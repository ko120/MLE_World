"""Output boundaries and native/child fd capture through real workers."""

import time
from pathlib import Path

import pytest

from aide.interpreter import Interpreter


@pytest.fixture
def interp(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    for name in (
        "AIDE_EXEC_GRACEFUL_SHUTDOWN_S", "AIDE_EXEC_INTERRUPT_GRACE_S",
        "AIDE_EXEC_TERM_GRACE_S", "AIDE_EXEC_KILL_GRACE_S",
    ):
        monkeypatch.setenv(name, "1")
    instance = Interpreter(tmp_path, timeout=15)
    yield instance
    instance.cleanup_session()


def captured(result):
    assert result.exc_type is None, result.term_out
    return "".join(result.term_out[:-1])


def test_printed_eof_marker_is_text_and_does_not_shift_next_result(interp):
    result = interp.run("print('BEFORE')\nprint('<|EOF|>')\nprint('AFTER')")
    assert captured(result) == "BEFORE\n<|EOF|>\nAFTER\n"
    for code, expected in (("pass", ""), ("print('NEXT_SESSION')", "NEXT_SESSION\n")):
        assert captured(interp.run(code, reset_session=False)) == expected


def test_python_native_c_and_subprocess_stdout_stderr_are_captured(interp):
    result = interp.run(r"""
import ctypes, os, subprocess, sys
assert sys.stdout.fileno() == 1 and sys.stderr.fileno() == 2
print('PYTHON_STDOUT', flush=True)
print('PYTHON_STDERR', file=sys.stderr, flush=True)
sys.stdout.buffer.write(b'BUFFER_STDOUT\n')
sys.stderr.buffer.write(b'BUFFER_STDERR\n')
os.write(1, b'NATIVE_STDOUT\n')
os.write(2, b'NATIVE_STDERR\n')
subprocess.run([sys.executable, '-c',
    "import sys; print('CHILD_STDOUT'); print('CHILD_STDERR', file=sys.stderr)"], check=True)
# No explicit fflush: the harness must flush native stdio at script completion.
ctypes.CDLL(None).printf(b'C_STDOUT')
""")
    text = captured(result)
    for label in (
        "PYTHON_STDOUT", "PYTHON_STDERR", "BUFFER_STDOUT", "BUFFER_STDERR",
        "NATIVE_STDOUT", "NATIVE_STDERR", "CHILD_STDOUT", "CHILD_STDERR", "C_STDOUT",
    ):
        assert label in text
    following = interp.run("print('CLEAN_NEXT_RESULT')", reset_session=False)
    assert captured(following) == "CLEAN_NEXT_RESULT\n"


@pytest.mark.parametrize("method", ["fork", "spawn", "forkserver"])
def test_large_child_output_and_reused_forkserver(interp, method):
    payload = "λ🙂child-output-" * 10000
    for index in range(2):
        code = f"""
import multiprocessing as mp, os, sys
def worker():
    print({payload!r}, flush=True)
    os.write(2, b'CHILD_NATIVE_STDERR\\n')
if __name__ == '__main__':
    child = mp.get_context({method!r}).Process(target=worker)
    child.start()
    child.join()
    assert child.exitcode == 0
    print('JOINED_{index}')
"""
        text = captured(interp.run(code, reset_session=index == 0))
        assert text == payload + f"\nCHILD_NATIVE_STDERR\nJOINED_{index}\n"


def test_split_utf8_and_invalid_native_bytes_do_not_poison_next_result(interp):
    result = interp.run("""
import os, time
payload = 'λ🙂'.encode('utf-8')
for byte in payload:
    os.write(1, bytes([byte]))
    time.sleep(0.01)
os.write(2, b'\\xff\\xf0\\x9f')
""")
    assert captured(result) == "λ🙂\ufffd\ufffd"
    assert captured(interp.run("print('🙂')", reset_session=False)) == "🙂\n"


def test_native_bytes_survive_immediate_worker_death(interp, monkeypatch):
    # An immediate os._exit can also discard the queued state:ready message.
    # Bound that existing startup wait without delaying the crash after write().
    monkeypatch.setenv("AIDE_EXEC_SPAWN_TIMEOUT_S", "5")
    for _ in range(3):
        result = interp.run("import os\nos.write(2, b'BEFORE_EXIT\\n')\nos._exit(9)")
        assert result.exc_type == "RuntimeError", result.term_out
        assert "BEFORE_EXIT\n" in "".join(result.term_out)
    assert captured(interp.run("print('RECOVERED')")) == "RECOVERED\n"


def test_output_precedes_exception_traceback(interp):
    result = interp.run("import os\nos.write(2, b'BEFORE_ERROR\\n')\nraise ValueError('failure')")
    assert result.exc_type == "ValueError"
    text = "".join(result.term_out)
    assert text.index("BEFORE_ERROR\n") < text.index("Traceback")
    assert text.index("Traceback") < text.index("ValueError: failure")


def test_daemon_holding_output_pipe_does_not_block_completion(interp):
    started = time.monotonic()
    result = interp.run("""
import multiprocessing as mp, os, time
ready = mp.Event()
def worker():
    ready.set()
    while True:
        os.write(1, b'DAEMON\\n')
        time.sleep(0.01)
child = mp.Process(target=worker, daemon=True)
child.start()
assert ready.wait(5)
print('PARENT_FINISHED')
""")
    assert "PARENT_FINISHED\n" in captured(result)
    assert time.monotonic() - started < 10
