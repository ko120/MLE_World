"""Execution regressions; requires the dependencies in agents/aide/src."""

import multiprocessing as mp
import queue
import signal
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "agents/aide/src"))
interpreter_mod = pytest.importorskip("aide.interpreter")
Interpreter = interpreter_mod.Interpreter


@contextmanager
def _watchdog(seconds=10):
    # A broken Queue.get timeout must fail this test instead of hanging pytest.
    def expired(signum, frame):
        pytest.fail("Interpreter.run() exceeded the independent watchdog")

    previous = signal.signal(signal.SIGALRM, expired)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)


@pytest.fixture
def short_cleanup_graces(monkeypatch):
    for name in (
        "AIDE_EXEC_GRACEFUL_SHUTDOWN_S",
        "AIDE_EXEC_INTERRUPT_GRACE_S",
        "AIDE_EXEC_TERM_GRACE_S",
        "AIDE_EXEC_KILL_GRACE_S",
    ):
        monkeypatch.setenv(name, "1")


@pytest.mark.parametrize("stream", ["stdout", "stderr"])
def test_forked_worker_output_drains_before_join(tmp_path, short_cleanup_graces, stream):
    payload = "λ🙂worker-output-" * 10000
    interp = Interpreter(tmp_path, timeout=5)
    with _watchdog(20):
        try:
            result = interp.run(
                "import multiprocessing as mp, sys\n"
                "def worker():\n"
                f"    print({payload!r}, file=sys.{stream}, flush=True)\n"
                "process = mp.Process(target=worker)\n"
                "process.start()\n"
                "process.join()\n"
                "assert process.exitcode == 0\n"
                "print('worker joined')\n"
            )
            assert result.exc_type is None
            assert "".join(result.term_out[:-1]) == payload + "\nworker joined\n"
            following = interp.run("print('next execution')", reset_session=False)
            assert following.exc_type is None
            assert "".join(following.term_out[:-1]) == "next execution\n"
        finally:
            interp.cleanup_session()


def test_streaming_output_still_times_out_and_recovers(tmp_path, short_cleanup_graces):
    interp = Interpreter(tmp_path, timeout=0.1)
    with _watchdog(15):
        try:
            result = interp.run(
                "import time\n"
                "while True:\n"
                "    print('progress', flush=True)\n"
                "    time.sleep(0.01)\n"
            )
            assert result.exc_type == "TimeoutError"
            assert "".join(result.term_out).startswith("progress\n")
            assert result.term_out[-1].startswith("TimeoutError: Execution exceeded")
            assert result.timeout_s == 0.1
            assert "<|EOF|>" not in result.term_out
            recovered = interp.run("print('recovered')")
            assert recovered.exc_type is None
            assert "".join(recovered.term_out[:-1]) == "recovered\n"
        finally:
            interp.cleanup_session()


def test_cancelled_output_capture_does_not_consume_next_session(tmp_path, short_cleanup_graces):
    interp = Interpreter(tmp_path, timeout=5)
    # The file confirms execution has started before the parent cancels it.
    started = tmp_path / "started"
    with _watchdog(20), ThreadPoolExecutor(max_workers=1) as pool:
        try:
            running = pool.submit(
                interp.run,
                "from pathlib import Path\n"
                "import time\n"
                "Path('started').touch()\n"
                "while True:\n"
                "    print('old session', flush=True)\n"
                "    time.sleep(0.01)\n",
            )
            while not started.exists():
                if running.done():
                    pytest.fail(f"Execution stopped before cancellation: {running.result()}")
                threading.Event().wait(0.01)
            interp.cleanup_session()
            following = interp.run("print('new session')")
            assert following.exc_type is None
            assert "".join(following.term_out[:-1]) == "new session\n"
            cancelled = running.result(timeout=5)
            assert cancelled.exc_type == "TimeoutError"
            assert "old session\n" in "".join(cancelled.term_out)
            assert "new session" not in "".join(cancelled.term_out)
        finally:
            interp.cleanup_session()


def test_abrupt_death_during_large_output_returns_and_recovers(tmp_path):
    interp = Interpreter(tmp_path, timeout=2)
    with _watchdog():
        try:
            result = interp.run(
                "import os, time\n"
                "print('before-crash', flush=True)\n"
                "print('x' * 1000000, flush=True)\n"
                "time.sleep(0.3)\n"
                "os._exit(9)\n"
            )
            assert result.exc_type == "RuntimeError"
            output = "".join(result.term_out)
            assert output.startswith("REPL child process died unexpectedly (exitcode=9)")
            assert "before-crash" in output
            recovered = interp.run("print('recovered')")
            assert recovered.exc_type is None
            assert "".join(recovered.term_out).startswith("recovered\n")
        finally:
            interp.cleanup_session()


def test_large_unicode_output_and_session_reuse(tmp_path):
    interp = Interpreter(tmp_path, timeout=5)
    try:
        result = interp.run("print('λ🙂x' * 100000)\nprint('last')")
        assert result.exc_type is None
        assert "".join(result.term_out).startswith("λ🙂x" * 100000 + "\nlast\n")
        following = interp.run("print('next execution')", reset_session=False)
        assert following.exc_type is None
        assert "".join(following.term_out).startswith("next execution\n")
    finally:
        interp.cleanup_session()


def test_large_exception_metadata_is_preserved(tmp_path):
    interp = Interpreter(tmp_path, timeout=5)
    try:
        result = interp.run("raise ValueError('detail' * 100000)")
        assert result.exc_type == "ValueError"
        assert result.exc_info["args"] == ["detail" * 100000]
        assert any(frame[0] == "runfile.py" for frame in result.exc_stack)
    finally:
        interp.cleanup_session()


def test_partial_queue_message_survives_timeout_until_writer_resumes(monkeypatch):
    source = mp.get_context("spawn").Queue(maxsize=1)
    reader = interpreter_mod._QueueReader(source)
    partial_sent = threading.Event()
    resume = threading.Event()
    send = source._writer._send

    def pause_mid_message(payload):
        if len(payload) > 16384:
            send(payload[:32])
            partial_sent.set()
            resume.wait(timeout=5)
            send(payload[32:])
        else:
            send(payload)

    # Delay an actual multiprocessing message after its header and part of
    # its payload. A timeout must keep these bytes for the next read.
    monkeypatch.setattr(source._writer, "_send", pause_mid_message)
    value = "λ🙂" * 10000
    try:
        source.put(value)
        assert partial_sent.wait(timeout=2)
        with _watchdog(), pytest.raises(queue.Empty):
            reader.get(timeout=0.05)
        with pytest.raises(queue.Full):
            source.put_nowait("must not release capacity for a partial message")
        resume.set()
        assert reader.get(timeout=2) == value
        source.put_nowait("next message")
        assert reader.get(timeout=2) == "next message"
    finally:
        resume.set()
        source.cancel_join_thread()
        source.close()


def test_script_argv_isolated_from_aide_and_previous_execution(tmp_path, monkeypatch):
    harness_argv = ["aide", "data_dir=/home/data/", "exp_name=exp"]
    monkeypatch.setattr(sys, "argv", harness_argv[:])
    code = (
        "import argparse, sys\n"
        "assert sys.argv == [__file__], sys.argv\n"
        "parser = argparse.ArgumentParser()\n"
        "parser.add_argument('--seed', type=int, default=42)\n"
        "print('seed:', parser.parse_args().seed)\n"
        "sys.argv.append('--leftover')\n"
    )
    interp = Interpreter(tmp_path, timeout=5)
    try:
        for reset in (True, False):
            result = interp.run(code, reset_session=reset)
            assert result.exc_type is None, result.term_out
            assert "seed: 42" in "".join(result.term_out)
        assert sys.argv == harness_argv
    finally:
        interp.cleanup_session()


@pytest.mark.parametrize("exit_code", ["", "None", "0", "False"])
def test_successful_system_exit_keeps_outputs(tmp_path, exit_code):
    interp = Interpreter(tmp_path, timeout=5)
    try:
        result = interp.run(
            "import sys\n"
            "from pathlib import Path\n"
            "Path('submission.csv').write_text('id,pred\\n0,1\\n')\n"
            "print('Validation score: 0.9')\n"
            f"sys.exit({exit_code})\n"
            "raise AssertionError('code after sys.exit must not run')\n"
        )
        assert result.exc_type is None, result.term_out
        assert "Validation score: 0.9" in "".join(result.term_out)
        assert "SystemExit" not in "".join(result.term_out)
        assert (tmp_path / "submission.csv").read_text() == "id,pred\n0,1\n"
        following = interp.run("print('session still usable')", reset_session=False)
        assert following.exc_type is None
    finally:
        interp.cleanup_session()


@pytest.mark.parametrize("exit_code", ["1", "'error'", "'0'", "0.0"])
def test_unsuccessful_system_exit_remains_an_error(tmp_path, exit_code):
    interp = Interpreter(tmp_path, timeout=5)
    try:
        result = interp.run(f"import sys\nsys.exit({exit_code})")
        assert result.exc_type == "SystemExit"
    finally:
        interp.cleanup_session()
