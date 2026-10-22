"""Real Enroot/PID-namespace checks with tiny rootfs trees and synthetic labels.

No GPU, dataset, image download, or Slurm allocation is needed. Host /usr supplies
the standard-library Python and shell; only temporary test data is writable.
"""

import json
import logging
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import textwrap
import time
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from agents import run


@pytest.mark.skipif(
    not shutil.which("enroot") or not shutil.which("unshare") or not Path("/usr/bin/python3").exists(),
    reason="requires Linux Enroot, unshare, and system Python",
)
@pytest.mark.parametrize("crash_grader", [False, True])
def test_real_enroot_isolates_labels_and_reaps_detached_children(tmp_path, crash_grader):
    public = tmp_path / "public"
    private = tmp_path / "private"
    public.mkdir()
    private.mkdir()
    (public / "train.csv").write_text("public-training-data")
    (private / "labels.csv").write_text("synthetic-private-answer")
    task = tmp_path / "run"
    task.mkdir()
    competition = SimpleNamespace(id="isolation", public_dir=public, private_dir=private)
    agent = SimpleNamespace(name="isolation-test")
    logger = logging.getLogger(__name__)
    agent_mounts, _ = run._enroot_mounts(competition, agent, task, logger)
    grader_mounts, wrapper, control = run._enroot_grading_mounts(competition, agent, task, logger)
    (control / "enroot-config").mkdir()

    storage = tmp_path / "rootfs"
    for name in ("agent", "grader"):
        root = storage / name
        for directory in ("etc", "usr", "home/agent", "private", "mlebench/data_privateonly"):
            (root / directory).mkdir(parents=True, exist_ok=True)
        for alias in ("bin", "sbin", "lib", "lib64"):
            if Path(f"/{alias}").exists():
                (root / alias).symlink_to(f"usr/{alias}")
        (root / "etc/environment").touch()
        (root / "etc/fstab").touch()
        (root / "home/agent/shared.txt").write_text("original-grader-code")
        (root / "mlebench/data_privateonly/baked-labels").write_text("old-image-private-data")

    server_script = control / "synthetic_grader.py"
    server_script.write_text(textwrap.dedent("""
        import json, os, socket
        from http.server import BaseHTTPRequestHandler, HTTPServer
        from pathlib import Path
        generations = Path('/run/grading/generations')
        generation = int(generations.read_text()) + 1 if generations.exists() else 1
        generations.write_text(str(generation))
        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                self.send({'status': 'running', 'run_token': os.environ['AIDE_RUN_TOKEN'], 'generation': generation})
            def do_POST(self):
                self.rfile.read(int(self.headers.get('Content-Length', '0')))
                answer = Path('/private/data/isolation/prepared/private/labels.csv').read_text()
                assert answer == 'synthetic-private-answer'
                assert Path('/home/agent/shared.txt').read_text() == 'original-grader-code'
                if self.path == '/score':
                    self.send({'score': 0.75})
                else:
                    self.send({'result': 'Submission is valid.'})
                if generation == 1 and os.environ.get('CRASH_GRADER') == '1' and self.path == '/validate':
                    os._exit(17)
            def send(self, body):
                self.send_response(200)
                self.end_headers()
                self.wfile.write(json.dumps(body).encode())
                self.wfile.flush()
        with HTTPServer(('127.0.0.1', 0), Handler, bind_and_activate=False) as server:
            server.socket.close()
            server.socket = socket.socket(fileno=int(os.environ['GRADING_SERVER_FD']))
            ready = Path(os.environ['GRADING_SERVER_READY_FILE'])
            temporary = ready.with_suffix('.tmp')
            temporary.write_text(json.dumps({'port': server.socket.getsockname()[1], 'run_token': os.environ['AIDE_RUN_TOKEN']}))
            temporary.replace(ready)
            server.serve_forever()
    """))
    grader_mounts = [mount for mount in grader_mounts if ":/private/grading_server.py:" not in mount]
    grader_mounts.append(f"{server_script}:/private/grading_server.py:none:x-create=file,bind,ro")
    (storage / "agent/home/agent/heartbeat.py").write_text(textwrap.dedent("""
        import time
        from pathlib import Path
        while True:
            Path('/home/logs/heartbeat').write_text(str(time.time_ns()))
            time.sleep(0.02)
    """))
    agent_script = storage / "agent/home/agent/check.py"
    agent_script.write_text(textwrap.dedent("""
        import json, os, subprocess, time
        from pathlib import Path
        from urllib.request import Request, ProxyHandler, build_opener
        from urllib.error import URLError
        assert Path('/home/data/train.csv').read_text() == 'public-training-data'
        protected = [
            Path('/private/data/isolation/prepared/private/labels.csv'),
            Path('/mlebench/data_privateonly/baked-labels'),
            Path('/run/grading/grading_server.log'),
            Path(os.environ['PRIVATE_HOST_FILE']),
            Path(os.environ['CONTROL_HOST_DIR']) / 'grading_server.log',
        ]
        for process in Path('/proc').glob('[0-9]*'):
            protected.extend([
                process / 'root/private/data/isolation/prepared/private/labels.csv',
                process / 'root' / os.environ['PRIVATE_HOST_FILE'].lstrip('/'),
            ])
        for path in protected:
            try:
                path.read_bytes()
            except (FileNotFoundError, PermissionError, ProcessLookupError):
                continue
            raise AssertionError('private file accessible: ' + str(path))
        assert os.readlink('/proc/self/ns/pid') != os.environ['HOST_PID_NAMESPACE']
        assert int(os.readlink('/proc/self')) == os.getpid()
        Path('/home/agent/shared.txt').write_text('agent-modified-code')
        # The descendant deliberately escapes the ordinary process group.
        child = subprocess.Popen(['/usr/bin/python3', '/home/agent/heartbeat.py'], start_new_session=True)
        deadline = time.monotonic() + 5
        while not Path('/home/logs/heartbeat').exists():
            assert time.monotonic() < deadline
            time.sleep(0.01)
        http = build_opener(ProxyHandler({}))
        for key, expected in [('AIDE_GRADING_SERVER_URL', {'score': 0.75}),
                              ('AIDE_VALIDATE_SERVER_URL', {'result': 'Submission is valid.'})]:
            with http.open(Request(os.environ[key], data=b'submission'), timeout=5) as response:
                assert json.load(response) == expected
        if os.environ['CRASH_GRADER'] == '1':
            deadline = time.monotonic() + 15
            health_url = os.environ['AIDE_GRADING_SERVER_URL'].replace('/score', '/health')
            while time.monotonic() < deadline:
                try:
                    with http.open(health_url, timeout=0.2) as response:
                        if json.load(response)['generation'] == 2:
                            break
                except (OSError, URLError):
                    pass
                time.sleep(0.02)
            else:
                raise AssertionError('grader never recovered on the original URL')
            with http.open(Request(os.environ['AIDE_GRADING_SERVER_URL'], data=b'submission'), timeout=5) as response:
                assert json.load(response) == {'score': 0.75}
        Path('/home/logs/verified').write_text('labels hidden; grading works')
    """))
    common_env = {"PATH": "/usr/bin:/bin", "HOME": "/tmp", "AIDE_RUN_TOKEN": tmp_path.name}
    common_mounts = ["/usr:/usr:none:x-create=dir,bind,ro"]
    grader_command = run._enroot_start_command(
        "grader", common_mounts + grader_mounts,
        {**common_env, "GRADING_SERVER_READY_FILE": "/run/grading/grading_server_ready.json",
         "GRADING_SERVER_FD": None,
         "CRASH_GRADER": str(int(crash_grader)), "NVIDIA_VISIBLE_DEVICES": "void"},
        ["/usr/bin/python3", "/private/grading_server.py"],
    )
    agent_command = run._enroot_start_command(
        "agent", common_mounts + agent_mounts,
        {**common_env, "AIDE_GRADING_SERVER_URL": None, "AIDE_VALIDATE_SERVER_URL": None,
         "PRIVATE_HOST_FILE": str(private / "labels.csv"), "CONTROL_HOST_DIR": str(control),
         "HOST_PID_NAMESPACE": os.readlink("/proc/self/ns/pid"), "CRASH_GRADER": str(int(crash_grader))},
        ["/usr/bin/python3", "/home/agent/check.py"],
    )
    env = {
        **os.environ, "ENROOT_DATA_PATH": str(storage), "ENROOT_RUNTIME_PATH": str(tmp_path / "runtime"),
        "ENROOT_CACHE_PATH": str(tmp_path / "cache"), "ENROOT_CONFIG_PATH": str(control / "enroot-config"),
        "ENROOT_MOUNT_HOME": "no", "ENROOT_ALLOW_SUPERUSER": "no", "NVIDIA_VISIBLE_DEVICES": "void",
        "LOGS_DIR": str(control), "GRADING_SERVER_READY_FILE": str(control / "grading_server_ready.json"),
        "GRADING_SERVER_STARTUP_TIMEOUT_SECS": "15", "GRADING_SERVER_COMMAND_JSON": json.dumps(grader_command),
        "GRADING_SERVER_RESTART_DELAY_SECS": "0.05",
        "AIDE_RUN_TOKEN": tmp_path.name,
    }
    output = task / "wrapper.log"
    with output.open("w") as log:
        proc = subprocess.Popen([sys.executable, str(wrapper), *agent_command], env=env,
                                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    try:
        code = proc.wait(timeout=30)
        assert code == 0, output.read_text() + (control / "grading_server.log").read_text()
        assert (task / "logs/verified").read_text() == "labels hidden; grading works"
        heartbeat = task / "logs/heartbeat"
        before = heartbeat.read_bytes()
        # The detached child must be gone when the namespace init exits.
        time.sleep(0.2)
        assert heartbeat.read_bytes() == before
        if crash_grader:
            assert "Grading server exited (code 17)" in output.read_text()
            assert "Restarting grading server" in output.read_text()
            assert (control / "generations").read_text() == "2"
    finally:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait(timeout=5)
        # Enroot creates mode-000 mountpoints. Exercise its removal path instead
        # of leaving those trees for pytest's generic temporary-dir cleanup.
        with patch.dict(os.environ, env):
            removed = [run._remove_enroot_container(name, storage, logger) for name in ("grader", "agent")]
        assert all(removed)
        assert not (storage / "grader").exists()
        assert not (storage / "agent").exists()
