"""Exercise the readiness handshake with independent server/wrapper processes."""

import errno
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import textwrap
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from urllib.error import URLError
from urllib.request import ProxyHandler, Request, build_opener

import pytest

ROOT = Path(__file__).resolve().parents[1]
AGENT_PROGRAM = """
import json, os, time
from pathlib import Path
from urllib.request import ProxyHandler, build_opener
keys = ('GRADING_SERVER_PORT', 'AIDE_GRADING_SERVER_URL', 'AIDE_VALIDATE_SERVER_URL')
result = {key: os.environ[key] for key in keys}
result['pid'] = os.getpid()
base = result['AIDE_GRADING_SERVER_URL'].removesuffix('/score')
if not os.environ.get('TEST_AGENT_SKIP_HEALTH'):
    with build_opener(ProxyHandler({})).open(base + '/health', timeout=5) as response:
        result['health'] = json.load(response)
marker = Path(os.environ['TEST_AGENT_MARKER'])
temporary = marker.with_suffix('.tmp')
temporary.write_text(json.dumps(result))
temporary.replace(marker)
deadline = time.monotonic() + 60
while not Path(os.environ['TEST_AGENT_RELEASE']).exists():
    if time.monotonic() >= deadline:
        raise RuntimeError('test agent release timed out')
    time.sleep(0.05)
raise SystemExit(int(os.environ.get('TEST_AGENT_EXIT_CODE', '0')))
"""

RESTARTABLE_GRADER = """
import json, os, socket, threading, time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
directory = Path(os.environ['LOGS_DIR'])
attempts = directory / 'attempts'
generation = int(attempts.read_text()) + 1 if attempts.exists() else 1
(directory / 'server.pid').write_text(str(os.getpid()))
attempts.with_suffix('.tmp').write_text(str(generation))
attempts.with_suffix('.tmp').replace(attempts)
print('GRADER BOOT', generation, flush=True)
if generation <= int(os.environ.get('TEST_STARTUP_FAILURES', '0')) or (
    generation > 1 and (directory / 'fail-restarts').exists()
):
    raise SystemExit(23)
if generation > 1 and os.environ.get('TEST_BLOCK_RESTART'):
    while not (directory / 'release-grader').exists():
        time.sleep(0.01)
class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        failures = directory / 'health-failures'
        if self.path == '/health' and failures.exists():
            remaining = int(failures.read_text())
            if remaining > 0:
                failures.with_suffix('.tmp').write_text(str(remaining - 1))
                failures.with_suffix('.tmp').replace(failures)
                self.send({'status': 'unavailable', 'run_token': os.environ['AIDE_RUN_TOKEN']})
                return
        self.send({'status': 'running', 'run_token': os.environ['AIDE_RUN_TOKEN'], 'generation': generation})
    def do_POST(self):
        self.rfile.read(int(self.headers.get('Content-Length', '0')))
        self.send({'score': 0.75} if self.path == '/score' else {'result': 'Submission is valid.'})
    def send(self, body):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(json.dumps(body).encode())
def exit_when_requested():
    marker = directory / 'exit-grader'
    while not marker.exists():
        time.sleep(0.01)
    marker.unlink()
    code = int(os.environ['TEST_GRADER_EXIT_CODE'])
    if code < 0:
        os.kill(os.getpid(), -code)
    os._exit(code)
with HTTPServer(('127.0.0.1', 0), Handler, bind_and_activate=False) as server:
    server.socket.close()
    server.socket = socket.socket(fileno=int(os.environ['GRADING_SERVER_FD']))
    server.server_address = server.socket.getsockname()
    ready = Path(os.environ['GRADING_SERVER_READY_FILE'])
    temporary = ready.with_suffix('.tmp')
    temporary.write_text(json.dumps({'port': server.server_address[1], 'run_token': os.environ['AIDE_RUN_TOKEN']}))
    temporary.replace(ready)
    threading.Thread(target=exit_when_requested, daemon=True).start()
    server.serve_forever()
"""


@pytest.fixture
def launch_wrapper(tmp_path):
    runs = []

    def launch(*, script=None, hold=False, stale=False, fast_health_checks=False, **extra_env):
        directory = tmp_path / f"run-{len(runs)}"
        directory.mkdir()
        wrapper = directory / "grading_server_wrapper.py"
        shutil.copyfile(ROOT / "environment" / wrapper.name, wrapper)
        server = directory / "grading_server.py"
        if script is None:
            server.symlink_to(ROOT / "environment" / server.name)
        else:
            server.write_text(textwrap.dedent(script))
        ready = directory / "grading_server_ready.json"
        if stale:
            ready.write_text(json.dumps({"port": 5000, "run_token": "old"}))
        marker = directory / "agent-started.json"
        release = directory / "release-agent"
        if not hold:
            release.touch()
        env = {
            **os.environ,
            "PYTHONPATH": str(ROOT),
            "LOGS_DIR": str(directory),
            "GRADING_SERVER_READY_FILE": str(ready),
            "GRADING_SERVER_PORT": "0",
            "GRADING_SERVER_STARTUP_TIMEOUT_SECS": "45",
            "GRADING_SERVER_RESTART_DELAY_SECS": "0.05",
            "AIDE_RUN_TOKEN": directory.name,
            "COMPETITION_ID": "test-competition",
            "TEST_AGENT_MARKER": str(marker),
            "TEST_AGENT_RELEASE": str(release),
            # A profile's old URLs must be replaced after the handshake.
            "AIDE_GRADING_SERVER_URL": "http://127.0.0.1:1/score",
            "AIDE_VALIDATE_SERVER_URL": "http://127.0.0.1:1/validate",
            **extra_env,
        }
        wrapper_command = [sys.executable, str(wrapper)]
        if fast_health_checks:
            # Keep the production defaults slow; shorten only these test probes.
            wrapper_command = [sys.executable, "-c", textwrap.dedent(f"""
                import sys
                sys.path.insert(0, {str(directory)!r})
                import grading_server_wrapper as wrapper
                wrapper.HEALTH_CHECK_INTERVAL_SECS = 0.1
                wrapper.HEALTH_CHECK_TIMEOUT_SECS = 0.1
                raise SystemExit(wrapper.main())
            """)]
        output = directory / "wrapper.log"
        with output.open("w") as log:
            proc = subprocess.Popen(
                [*wrapper_command, sys.executable, "-c", AGENT_PROGRAM],
                env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
            )
        run = SimpleNamespace(
            proc=proc, directory=directory, ready=ready, marker=marker,
            release=release, output=output, token=env["AIDE_RUN_TOKEN"],
        )
        runs.append(run)
        return run

    yield launch
    for run in runs:
        try:
            os.killpg(run.proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            run.proc.wait(timeout=12)
        except subprocess.TimeoutExpired:
            os.killpg(run.proc.pid, signal.SIGKILL)
            run.proc.wait(timeout=5)


def _agent_started(run):
    deadline = time.monotonic() + 50
    while time.monotonic() < deadline:
        if run.marker.exists():
            return json.loads(run.marker.read_text())
        assert run.proc.poll() is None, run.output.read_text()
        time.sleep(0.05)
    pytest.fail(f"agent never started: {run.output.read_text()}")


def test_independent_runs_hold_distinct_ports_and_route_both_endpoints(launch_wrapper):
    runs = [launch_wrapper(hold=True) for _ in range(2)]
    ports = []
    for run in runs:
        agent = _agent_started(run)
        ready = json.loads(run.ready.read_text())
        ports.append(ready["port"])
        assert ready == {"port": ready["port"], "run_token": run.token}
        assert agent["health"] == {"status": "running", "run_token": run.token}
        assert agent["GRADING_SERVER_PORT"] == str(ready["port"])
        for key, endpoint in (("AIDE_GRADING_SERVER_URL", "score"), ("AIDE_VALIDATE_SERVER_URL", "validate")):
            assert agent[key] == f"http://127.0.0.1:{ready['port']}/{endpoint}"
        # Readiness must describe a socket the server continues to own.
        with socket.socket() as competing_socket, pytest.raises(OSError) as error:
            competing_socket.bind(("127.0.0.1", ready["port"]))
        assert error.value.errno == errno.EADDRINUSE
    assert len(set(ports)) == len(runs)
    for run in runs:
        run.release.touch()
        assert run.proc.wait(timeout=12) == 0, run.output.read_text()
        port = json.loads(run.ready.read_text())["port"]
        with socket.socket() as connection:
            connection.settimeout(1)
            assert connection.connect_ex(("127.0.0.1", port)) == errno.ECONNREFUSED


@pytest.mark.parametrize("returncode", [1, 124, 137])
def test_wrapper_preserves_agent_exit_status(launch_wrapper, returncode):
    run = launch_wrapper(TEST_AGENT_EXIT_CODE=str(returncode))
    assert run.proc.wait(timeout=50) == returncode, run.output.read_text()
    assert run.marker.exists()


def test_persistent_startup_failure_does_not_fail_agent(launch_wrapper):
    run = launch_wrapper(
        script="raise SystemExit(23)\n", TEST_AGENT_SKIP_HEALTH="1",
        GRADING_SERVER_STARTUP_TIMEOUT_SECS="0.5",
    )
    assert run.proc.wait(timeout=10) == 0
    assert run.marker.exists()
    assert "exited before readiness (code 23)" in run.output.read_text()
    assert "starting agent while recovery continues" in run.output.read_text()


def test_transient_startup_failures_are_retried_before_agent_starts(launch_wrapper):
    run = launch_wrapper(script=RESTARTABLE_GRADER, TEST_STARTUP_FAILURES="2")
    assert run.proc.wait(timeout=15) == 0, run.output.read_text()
    assert json.loads(run.marker.read_text())["health"]["generation"] == 3


def test_stale_readiness_is_removed_and_hung_startup_is_killed(launch_wrapper):
    run = launch_wrapper(
        script="""
        import os, time
        from pathlib import Path
        Path(os.environ['LOGS_DIR'], 'server.pid').write_text(str(os.getpid()))
        time.sleep(60)
        """,
        stale=True, GRADING_SERVER_STARTUP_TIMEOUT_SECS="0.5", TEST_AGENT_SKIP_HEALTH="1",
    )
    assert run.proc.wait(timeout=12) == 0
    assert run.marker.exists()
    assert not run.ready.exists()
    assert "starting agent while recovery continues" in run.output.read_text()
    pid = int((run.directory / "server.pid").read_text())
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)


def test_readiness_from_another_run_is_rejected(launch_wrapper):
    run = launch_wrapper(TEST_AGENT_SKIP_HEALTH="1", GRADING_SERVER_STARTUP_TIMEOUT_SECS="0.5", script="""
        import json, os, time
        from pathlib import Path
        record = {'port': 12345, 'run_token': 'another-run'}
        ready = Path(os.environ['GRADING_SERVER_READY_FILE'])
        ready.with_suffix('.tmp').write_text(json.dumps(record))
        ready.with_suffix('.tmp').replace(ready)
        time.sleep(60)
    """)
    assert run.proc.wait(timeout=12) == 0
    assert run.marker.exists()
    assert "readiness run token does not match" in run.output.read_text()


def test_health_from_another_run_is_rejected(launch_wrapper):
    run = launch_wrapper(TEST_AGENT_SKIP_HEALTH="1", GRADING_SERVER_STARTUP_TIMEOUT_SECS="0.5", script="""
        import json, os, socket
        from http.server import BaseHTTPRequestHandler, HTTPServer
        from pathlib import Path
        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                payload = json.dumps({'status': 'running', 'run_token': 'another-run'}).encode()
                self.send_response(200)
                self.end_headers()
                self.wfile.write(payload)
        with HTTPServer(('127.0.0.1', 0), Handler, bind_and_activate=False) as server:
            server.socket.close()
            server.socket = socket.socket(fileno=int(os.environ['GRADING_SERVER_FD']))
            record = {'port': server.socket.getsockname()[1], 'run_token': os.environ['AIDE_RUN_TOKEN']}
            ready = Path(os.environ['GRADING_SERVER_READY_FILE'])
            ready.with_suffix('.tmp').write_text(json.dumps(record))
            ready.with_suffix('.tmp').replace(ready)
            server.serve_forever()
    """)
    assert run.proc.wait(timeout=12) == 0
    assert run.marker.exists()
    assert "health check does not match" in run.output.read_text()


def _generation_ready(run, generation):
    url = json.loads(run.marker.read_text())["AIDE_GRADING_SERVER_URL"].removesuffix("/score") + "/health"
    http = build_opener(ProxyHandler({}))
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        assert run.proc.poll() is None, run.output.read_text()
        try:
            with http.open(url, timeout=0.2) as response:
                health = json.load(response)
            if health["generation"] >= generation:
                return health
        except (OSError, URLError):
            pass
        time.sleep(0.02)
    pytest.fail(f"grader never recovered: {run.output.read_text()}")


def _wait_for_log(run, message, count=1):
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        assert run.proc.poll() is None, run.output.read_text()
        if run.output.read_text().count(message) >= count:
            return
        time.sleep(0.02)
    pytest.fail(f"missing {message!r}: {run.output.read_text()}")


def test_unresponsive_server_restarts_without_interrupting_agent(launch_wrapper):
    run = launch_wrapper(hold=True, script=RESTARTABLE_GRADER, fast_health_checks=True)
    agent = _agent_started(run)
    _wait_for_log(run, "Grading server ready:")
    server_pid = int((run.directory / "server.pid").read_text())
    os.kill(server_pid, signal.SIGSTOP)
    _generation_ready(run, 2)
    assert "3 consecutive health check failures" in run.output.read_text()
    assert json.loads(run.ready.read_text())["port"] == int(agent["GRADING_SERVER_PORT"])
    assert json.loads(run.marker.read_text())["pid"] == agent["pid"]
    os.kill(agent["pid"], 0)
    with pytest.raises(ProcessLookupError):
        os.kill(server_pid, 0)
    run.release.touch()
    assert run.proc.wait(timeout=12) == 0, run.output.read_text()


def test_successful_health_check_resets_consecutive_failures(launch_wrapper):
    run = launch_wrapper(hold=True, script=RESTARTABLE_GRADER, fast_health_checks=True)
    _agent_started(run)
    _wait_for_log(run, "Grading server ready:")
    for recovery in (1, 2):
        pending = run.directory / "health-failures-next"
        pending.write_text("2")
        pending.replace(run.directory / "health-failures")
        _wait_for_log(run, "Grading server health check recovered", count=recovery)
        assert int((run.directory / "attempts").read_text()) == 1
    run.release.touch()
    assert run.proc.wait(timeout=12) == 0, run.output.read_text()


def test_health_checks_allow_an_in_progress_score(launch_wrapper):
    source = str(ROOT / "environment/grading_server.py")
    run = launch_wrapper(hold=True, fast_health_checks=True, script=f"""
        import importlib.util, os, sys, time
        from pathlib import Path
        spec = importlib.util.spec_from_file_location('test_grading_server', {source!r})
        server = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = server
        spec.loader.exec_module(server)
        directory = Path(os.environ['LOGS_DIR'])
        def grade(submission):
            (directory / 'grade-started').touch()
            deadline = time.monotonic() + 10
            while not (directory / 'release-grade').exists():
                if time.monotonic() >= deadline:
                    raise TimeoutError('test did not release grading')
                time.sleep(0.01)
            return 0.75
        server._load_and_grade = grade
        server.serve_with_readiness(Path(os.environ['GRADING_SERVER_READY_FILE']))
    """)
    agent = _agent_started(run)
    _wait_for_log(run, "Grading server ready:")
    payload = (
        b'--submission\r\nContent-Disposition: form-data; name="file"; filename="submission.csv"'
        b'\r\nContent-Type: text/csv\r\n\r\nid,pred\n0,0.5\n\r\n--submission--\r\n'
    )

    def submit():
        http = build_opener(ProxyHandler({}))
        request = Request(agent["AIDE_GRADING_SERVER_URL"], data=payload,
                          headers={"Content-Type": "multipart/form-data; boundary=submission"})
        with http.open(request, timeout=10) as response:
            return json.load(response)

    with ThreadPoolExecutor(max_workers=1) as pool:
        result = pool.submit(submit)
        try:
            deadline = time.monotonic() + 5
            while not (run.directory / "grade-started").exists():
                assert time.monotonic() < deadline, run.output.read_text()
                time.sleep(0.02)
            log = run.directory / "grading_server.log"
            probes = log.read_text().count("GET /health")
            time.sleep(1)
            assert not result.done(), "the score should still be running"
            assert log.read_text().count("GET /health") >= probes + 3
            assert "health check failed" not in run.output.read_text()
        finally:
            (run.directory / "release-grade").touch()
        assert result.result(timeout=5) == {"score": 0.75, "error": None}
    run.release.touch()
    assert run.proc.wait(timeout=12) == 0, run.output.read_text()


@pytest.mark.parametrize("exit_code", [0, 17, 124, -signal.SIGKILL])
def test_grader_restarts_on_same_url_without_restarting_agent(launch_wrapper, exit_code):
    run = launch_wrapper(hold=True, script=RESTARTABLE_GRADER, TEST_GRADER_EXIT_CODE=str(exit_code))
    agent = _agent_started(run)
    for generation in (2, 3):
        (run.directory / "exit-grader").touch()
        _generation_ready(run, generation)
        assert json.loads(run.ready.read_text())["port"] == int(agent["GRADING_SERVER_PORT"])
        assert json.loads(run.marker.read_text())["pid"] == agent["pid"]
        os.kill(agent["pid"], 0)
    run.release.touch()
    assert run.proc.wait(timeout=12) == 0, run.output.read_text()
    assert f"Grading server exited (code {exit_code})" in run.output.read_text()
    assert "GRADER BOOT 1" in (run.directory / "grading_server.log").read_text()
    assert "GRADER BOOT 3" in (run.directory / "grading_server.log").read_text()
    with pytest.raises(ProcessLookupError):
        os.kill(agent["pid"], 0)


def test_real_werkzeug_grader_reuses_listener_after_sigkill(launch_wrapper):
    source = str(ROOT / "environment/grading_server.py")
    run = launch_wrapper(hold=True, script=f"""
        import os, runpy
        from pathlib import Path
        Path(os.environ['LOGS_DIR'], 'server.pid').write_text(str(os.getpid()))
        runpy.run_path({source!r}, run_name='__main__')
    """)
    agent = _agent_started(run)
    original_pid = int((run.directory / "server.pid").read_text())
    os.kill(original_pid, signal.SIGKILL)
    http = build_opener(ProxyHandler({}))
    url = agent["AIDE_GRADING_SERVER_URL"].removesuffix("/score") + "/health"
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        assert run.proc.poll() is None, run.output.read_text()
        try:
            replacement_pid = int((run.directory / "server.pid").read_text())
            if replacement_pid != original_pid:
                with http.open(url, timeout=0.2) as response:
                    assert json.load(response) == {"status": "running", "run_token": run.token}
                break
        except (OSError, URLError, ValueError):
            pass
        time.sleep(0.02)
    else:
        pytest.fail(f"real grader did not restart: {run.output.read_text()}")
    assert json.loads(run.ready.read_text())["port"] == int(agent["GRADING_SERVER_PORT"])
    os.kill(agent["pid"], 0)
    run.release.touch()
    assert run.proc.wait(timeout=12) == 0, run.output.read_text()


@pytest.mark.parametrize("agent_code", [0, 7, 124, 137])
def test_failed_restarts_do_not_override_agent_exit_status(launch_wrapper, agent_code):
    run = launch_wrapper(hold=True, script=RESTARTABLE_GRADER,
                         TEST_GRADER_EXIT_CODE="17", TEST_AGENT_EXIT_CODE=str(agent_code))
    _agent_started(run)
    (run.directory / "fail-restarts").touch()
    (run.directory / "exit-grader").touch()
    deadline = time.monotonic() + 10
    while int((run.directory / "attempts").read_text()) < 3:
        assert run.proc.poll() is None, run.output.read_text()
        assert time.monotonic() < deadline
        time.sleep(0.02)
    run.release.touch()
    assert run.proc.wait(timeout=12) == agent_code, run.output.read_text()


def test_port_stays_reserved_and_agent_can_finish_during_hung_restart(launch_wrapper):
    run = launch_wrapper(hold=True, script=RESTARTABLE_GRADER,
                         TEST_GRADER_EXIT_CODE="17", TEST_BLOCK_RESTART="1")
    agent = _agent_started(run)
    (run.directory / "exit-grader").touch()
    deadline = time.monotonic() + 10
    while int((run.directory / "attempts").read_text()) < 2:
        assert time.monotonic() < deadline
        time.sleep(0.02)
    with socket.socket() as competing, pytest.raises(OSError) as error:
        competing.bind(("127.0.0.1", int(agent["GRADING_SERVER_PORT"])))
    assert error.value.errno == errno.EADDRINUSE
    os.kill(agent["pid"], 0)
    run.release.touch()
    started = time.monotonic()
    assert run.proc.wait(timeout=12) == 0, run.output.read_text()
    assert time.monotonic() - started < 5
    grader_pid = int((run.directory / "server.pid").read_text())
    with pytest.raises(ProcessLookupError):
        os.kill(grader_pid, 0)
