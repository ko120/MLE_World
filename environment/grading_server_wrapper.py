"""Restart failed graders without interrupting the agent or changing its URLs.

Uses only the standard library so the startup deadline also covers the grading
server's imports. Enroot runs this frozen script on the host, with independent
container commands; private labels and grader control files never enter the
agent container. Both commands inherit the wrapper's process group and token.
"""

import json
import math
import os
import signal
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import ProxyHandler, build_opener


HEALTH_CHECK_INTERVAL_SECS = 60.0
HEALTH_CHECK_TIMEOUT_SECS = 1.0
HEALTH_CHECK_FAILURE_THRESHOLD = 3


def _check_health(http, port: int, run_token: str, timeout_s: float) -> None:
    with http.open(f"http://127.0.0.1:{port}/health", timeout=timeout_s) as response:
        health = json.loads(response.read(4096))
    if (
        not isinstance(health, dict)
        or health.get("status") != "running"
        or health.get("run_token") != run_token
    ):
        raise RuntimeError("Grading-server health check does not match this run")


def wait_for_readiness(
    server: subprocess.Popen,
    ready_file: Path,
    run_token: str,
    timeout_s: float,
    *,
    expected_port: int | None = None,
    stop: threading.Event | None = None,
) -> int:
    """Require a record and a responding HTTP server belonging to this run."""
    deadline = time.monotonic() + timeout_s
    # Node-local grading must not be sent through a profile's HTTP proxy.
    http = build_opener(ProxyHandler({}))
    while time.monotonic() < deadline:
        if stop is not None and stop.is_set():
            raise InterruptedError("Grading supervisor stopped")
        if server.poll() is not None:
            raise RuntimeError(f"Grading server exited before readiness (code {server.returncode})")
        try:
            record = json.loads(ready_file.read_text())
        except FileNotFoundError:
            record = None
        if record is not None:
            if not isinstance(record, dict) or record.get("run_token") != run_token:
                raise RuntimeError("Grading-server readiness run token does not match this run")
            port = record.get("port")
            if type(port) is not int or not 0 < port < 65536:
                raise RuntimeError("Grading-server readiness contains an invalid port")
            if expected_port is not None and port != expected_port:
                raise RuntimeError("Grading server did not use this run's reserved port")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                _check_health(http, port, run_token, min(1.0, remaining))
            except (URLError, OSError):
                # The record is published after listen(), just before the
                # request loop starts. Retry briefly while that loop comes up.
                pass
            else:
                if server.poll() is not None:
                    raise RuntimeError("Grading server exited during its readiness check")
                return port
        delay = min(0.1, max(0.0, deadline - time.monotonic()))
        if stop is None:
            time.sleep(delay)
        else:
            stop.wait(delay)
    raise TimeoutError(f"Grading server did not become ready within {timeout_s:g}s")


def _stop_process(process: subprocess.Popen | None) -> None:
    if process is None:
        return
    try:
        if process.poll() is None:
            process.terminate()
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            print(f"Process {process.pid} survived grading-wrapper cleanup", file=sys.stderr)
    except ProcessLookupError:
        pass


def _exit_on_signal(signum, frame) -> None:
    raise SystemExit(128 + signum)


def supervise_grader(
    command: list[str], server_log, listener: socket.socket, ready_file: Path,
    run_token: str, timeout_s: float, restart_delay: float,
    ready: threading.Event, stop: threading.Event,
) -> None:
    """Retry until the agent finishes; the parent keeps ownership of the port."""
    server = None
    attempt = 0
    delay = min(restart_delay, 30.0)
    port = listener.getsockname()[1]
    http = build_opener(ProxyHandler({}))
    server_env = {
        **os.environ,
        "NVIDIA_VISIBLE_DEVICES": "void",
        "GRADING_SERVER_FD": str(listener.fileno()),
        "GRADING_SERVER_PORT": str(port),
    }
    try:
        while not stop.is_set():
            healthy_since = None
            try:
                # Do not accumulate replacements if an old process cannot exit.
                if server is not None and server.poll() is None:
                    raise RuntimeError("Previous grading server is still stopping")
                ready_file.unlink(missing_ok=True)
                attempt += 1
                print(f"Starting grading server (attempt {attempt})", flush=True)
                server = subprocess.Popen(
                    command, stdout=server_log, stderr=subprocess.STDOUT,
                    env=server_env, pass_fds=(listener.fileno(),),
                )
                wait_for_readiness(
                    server, ready_file, run_token, timeout_s, expected_port=port, stop=stop,
                )
                healthy_since = time.monotonic()
                ready.set()
                print(f"Grading server ready: http://127.0.0.1:{port} (attempt {attempt})", flush=True)
                next_health_check = healthy_since + HEALTH_CHECK_INTERVAL_SECS
                health_failures = 0
                while not stop.wait(0.1):
                    if server.poll() is not None:
                        raise RuntimeError(f"Grading server exited (code {server.returncode})")
                    if time.monotonic() < next_health_check:
                        continue
                    # Probes run only in this supervisor thread. The threaded
                    # HTTP server can answer /health while a submission grades.
                    try:
                        _check_health(http, port, run_token, HEALTH_CHECK_TIMEOUT_SECS)
                    except Exception as exc:
                        if stop.is_set():
                            break
                        health_failures += 1
                        print(
                            f"Grading server health check failed "
                            f"({health_failures}/{HEALTH_CHECK_FAILURE_THRESHOLD}): {exc}",
                            file=sys.stderr, flush=True,
                        )
                        if health_failures >= HEALTH_CHECK_FAILURE_THRESHOLD:
                            raise RuntimeError(
                                "Grading server is unresponsive after "
                                f"{health_failures} consecutive health check failures"
                            ) from exc
                    else:
                        if health_failures:
                            print("Grading server health check recovered", flush=True)
                        health_failures = 0
                    next_health_check = time.monotonic() + HEALTH_CHECK_INTERVAL_SECS
            except Exception as exc:
                if not stop.is_set():
                    print(f"Grading server attempt {attempt} failed: {exc}", file=sys.stderr, flush=True)
            finally:
                _stop_process(server)
            if stop.is_set():
                break
            # Rapid failures back off to 30s. A minute of healthy service resets
            # the delay, so an isolated crash recovers promptly even in long runs.
            if healthy_since is not None and time.monotonic() - healthy_since >= 60:
                delay = min(restart_delay, 30.0)
            print(f"Restarting grading server in {delay:g}s; agent may continue", file=sys.stderr, flush=True)
            stop.wait(delay)
            delay = min(delay * 2, 30.0)
    finally:
        _stop_process(server)


def main(argv: list[str] | None = None) -> int:
    command = sys.argv[1:] if argv is None else argv
    agent = supervisor = listener = server_log = None
    ready, stop = threading.Event(), threading.Event()
    previous_handlers = {
        sig: signal.signal(sig, _exit_on_signal) for sig in (signal.SIGTERM, signal.SIGINT)
    }
    try:
        if not command:
            raise ValueError("Missing agent command")
        ready_file = Path(os.environ["GRADING_SERVER_READY_FILE"])
        run_token = os.environ["AIDE_RUN_TOKEN"]
        if not run_token:
            raise ValueError("Missing run token")
        timeout_s = float(os.environ.get("GRADING_SERVER_STARTUP_TIMEOUT_SECS", "120"))
        if not math.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError("GRADING_SERVER_STARTUP_TIMEOUT_SECS must be positive and finite")
        restart_delay = float(os.environ.get("GRADING_SERVER_RESTART_DELAY_SECS", "1"))
        if not math.isfinite(restart_delay) or restart_delay <= 0:
            raise ValueError("GRADING_SERVER_RESTART_DELAY_SECS must be positive and finite")
        server_command = json.loads(os.environ["GRADING_SERVER_COMMAND_JSON"]) if (
            "GRADING_SERVER_COMMAND_JSON" in os.environ
        ) else [sys.executable, str(Path(__file__).with_name("grading_server.py"))]
        if not isinstance(server_command, list) or not server_command or any(
            not isinstance(arg, str) or not arg for arg in server_command
        ):
            raise ValueError("GRADING_SERVER_COMMAND_JSON must contain a nonempty command list")
        log_path = Path(os.environ["LOGS_DIR"]) / "grading_server.log"
        # The wrapper reserves the socket for the entire run, including outages.
        # Only graders inherit it: no close/rebind race or stale agent URL.
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.bind(("127.0.0.1", 0))
        listener.listen(128)
        port = listener.getsockname()[1]
        server_log = log_path.open("w")
        supervisor = threading.Thread(
            target=supervise_grader,
            args=(server_command, server_log, listener, ready_file, run_token,
                  timeout_s, restart_delay, ready, stop),
            name="grading-supervisor",
        )
        supervisor.start()
        if not ready.wait(timeout_s):
            print(
                f"Grading server unavailable after {timeout_s:g}s; starting agent while "
                f"recovery continues; see {log_path}", file=sys.stderr, flush=True,
            )
        base_url = f"http://127.0.0.1:{port}"
        agent_env = {
            **os.environ,
            "GRADING_SERVER_PORT": str(port),
            "AIDE_GRADING_SERVER_URL": f"{base_url}/score",
            "AIDE_VALIDATE_SERVER_URL": f"{base_url}/validate",
        }
        for key in ("GRADING_SERVER_COMMAND_JSON", "GRADING_SERVER_FD"):
            agent_env.pop(key, None)
        agent = subprocess.Popen(command, env=agent_env)
        returncode = agent.wait()
        return 128 - returncode if returncode < 0 else returncode
    finally:
        stop.set()
        _stop_process(agent)
        if supervisor is not None:
            supervisor.join()
        if server_log is not None:
            server_log.close()
        if listener is not None:
            listener.close()
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)


if __name__ == "__main__":
    sys.exit(main())
