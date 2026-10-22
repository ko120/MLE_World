"""Script completion includes non-daemon multiprocessing work."""

import os
import signal
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from pathlib import Path

import pytest

from aide.interpreter import Interpreter


@pytest.fixture(autouse=True)
def short_cleanup_graces(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "AIDE_EXEC_GRACEFUL_SHUTDOWN_S",
        "AIDE_EXEC_INTERRUPT_GRACE_S",
        "AIDE_EXEC_TERM_GRACE_S",
        "AIDE_EXEC_KILL_GRACE_S",
    ):
        monkeypatch.setenv(name, "1")


@pytest.mark.parametrize("method", ["fork", "spawn", "forkserver"])
def test_generated_script_can_select_start_method(tmp_path: Path, method: str) -> None:
    interp = Interpreter(tmp_path, timeout=15)
    code = f"""
import multiprocessing as mp
from pathlib import Path

def worker():
    Path('child_method').write_text(mp.get_start_method())

if __name__ == '__main__':
    mp.set_start_method({method!r})
    child = mp.Process(target=worker)
    child.start()
    child.join()
    assert child.exitcode == 0
"""
    try:
        result = interp.run(code)
        assert result.exc_type is None, result.term_out
        assert (tmp_path / "child_method").read_text() == method
        following = interp.run("print('method:', mp.get_start_method())", reset_session=False)
        assert following.exc_type is None, following.term_out
        assert f"method: {method}" in "".join(following.term_out)
    finally:
        interp.cleanup_session()


@pytest.mark.parametrize("method", ["fork", "spawn"])
@pytest.mark.parametrize(
    ("ending", "expected_exception"),
    [("", None), ("sys.exit(0)", None), ("raise ValueError('parent failed')", "ValueError")],
)
def test_unjoined_child_finishes_before_result_and_script_removal(
    tmp_path: Path, method: str, ending: str, expected_exception: str | None
) -> None:
    interp = Interpreter(tmp_path, timeout=15)
    code = f"""
import multiprocessing as mp
import sys
import time
from pathlib import Path

def write_submission():
    time.sleep(0.1)
    # Both spawn bootstrap and the running child must still see the script.
    assert 'def write_submission' in Path(__file__).read_text()
    Path('submission.csv').write_text('id,prediction\\n1,0.5\\n')

if __name__ == '__main__':
    mp.get_context({method!r}).Process(target=write_submission).start()
    {ending}
"""
    try:
        result = interp.run(code)
        assert result.exc_type == expected_exception, result.term_out
        assert (tmp_path / "submission.csv").read_text() == "id,prediction\n1,0.5\n"
        assert not (tmp_path / "runfile.py").exists()
    finally:
        interp.cleanup_session()


def test_unjoined_fork_child_output_is_fully_captured(tmp_path: Path) -> None:
    interp = Interpreter(tmp_path, timeout=10)
    try:
        result = interp.run(
            "import multiprocessing as mp\n"
            "def worker():\n"
            "    print('child-output-' * 20000, flush=True)\n"
            "mp.get_context('fork').Process(target=worker).start()\n"
        )
        assert result.exc_type is None, result.term_out
        assert "".join(result.term_out[:-1]) == "child-output-" * 20000 + "\n"
    finally:
        interp.cleanup_session()


def test_timeout_interrupts_child_wait_and_reaps_child(tmp_path: Path) -> None:
    interp = Interpreter(tmp_path, timeout=1)
    code = """
import multiprocessing as mp
import os
import time
from pathlib import Path

def worker():
    Path('child_pid').write_text(str(os.getpid()))
    time.sleep(30)

mp.get_context('fork').Process(target=worker).start()
"""
    try:
        result = interp.run(code)
        assert result.exc_type == "TimeoutError", result.term_out
        child_pid = int((tmp_path / "child_pid").read_text())
        with pytest.raises(ProcessLookupError):
            os.kill(child_pid, 0)
        interp.timeout = 5
        recovered = interp.run("print('recovered')")
        assert recovered.exc_type is None, recovered.term_out
        assert "recovered" in "".join(recovered.term_out)
    finally:
        interp.cleanup_session()


def test_daemon_child_does_not_block_completion(tmp_path: Path) -> None:
    interp = Interpreter(tmp_path, timeout=5)
    try:
        result = interp.run(
            "import multiprocessing as mp, time\n"
            "ctx = mp.get_context('fork')\n"
            "ready = ctx.Event()\n"
            "def worker():\n"
            "    ready.set()\n"
            "    time.sleep(30)\n"
            "child = ctx.Process(target=worker, daemon=True)\n"
            "child.start()\n"
            "assert ready.wait(2)\n"
            "print('daemon started')\n"
        )
        assert result.exc_type is None, result.term_out
        assert "daemon started" in "".join(result.term_out)
    finally:
        interp.cleanup_session()


def test_interrupt_keeps_script_until_resistant_child_finishes(tmp_path: Path) -> None:
    interp = Interpreter(tmp_path, timeout=15)
    release = tmp_path / "release"
    code = """
import multiprocessing as mp
import signal
import time
from pathlib import Path

def worker():
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    Path('child_started').touch()
    while not Path('release').exists():
        time.sleep(0.01)
    assert Path(__file__).is_file()
    Path('submission.csv').write_text('complete')

mp.get_context('fork').Process(target=worker).start()
Path('parent_done').touch()
"""
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(interp.run, code)
        try:
            deadline = time.monotonic() + 10
            while not all((tmp_path / name).exists() for name in ("parent_done", "child_started")):
                assert time.monotonic() < deadline
                assert not future.done()
                time.sleep(0.01)
            with pytest.raises(FutureTimeout):
                future.result(timeout=0.1)
            assert interp._pgid == interp.process.pid
            os.killpg(interp._pgid, signal.SIGINT)
            with pytest.raises(FutureTimeout):
                future.result(timeout=0.2)
            assert (tmp_path / "runfile.py").is_file()
            release.touch()
            result = future.result(timeout=5)
            assert result.exc_type == "TimeoutError", result.term_out
            assert (tmp_path / "submission.csv").read_text() == "complete"
            assert not (tmp_path / "runfile.py").exists()
        finally:
            release.touch()
            interp.cleanup_session()


@pytest.mark.parametrize("backend", ["concurrent_futures", "joblib"])
def test_managed_process_pool_finishes_and_session_can_be_reused(
    tmp_path: Path, backend: str
) -> None:
    if backend == "joblib":
        pytest.importorskip("joblib")
        body = (
            "from joblib import Parallel, delayed\n"
            "results = Parallel(n_jobs=2)(delayed(square)(x) for x in [1, 2, 3])\n"
        )
    else:
        body = (
            "from concurrent.futures import ProcessPoolExecutor\n"
            "executor = ProcessPoolExecutor(2, mp_context=mp.get_context('spawn'))\n"
            "results = list(executor.map(square, [1, 2, 3]))\n"
        )
    code = (
        "import multiprocessing as mp\n"
        "def square(x):\n"
        "    return x * x\n"
        "if __name__ == '__main__':\n"
        + "".join("    " + line + "\n" for line in body.splitlines())
        + "    assert results == [1, 4, 9]\n"
        + "    print('pool finished')\n"
    )
    interp = Interpreter(tmp_path, timeout=15)
    try:
        for reset in (True, False):
            result = interp.run(code, reset_session=reset)
            assert result.exc_type is None, result.term_out
            assert "pool finished" in "".join(result.term_out)
    finally:
        interp.cleanup_session()
