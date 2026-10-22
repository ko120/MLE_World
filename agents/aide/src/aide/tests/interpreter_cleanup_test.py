"""Focused process-lifecycle tests for the real-code interpreter."""

import os
import signal
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from aide.interpreter import Interpreter, _process_group_exists


def _short_teardown(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AIDE_EXEC_GRACEFUL_SHUTDOWN_S", "0.2")
    monkeypatch.setenv("AIDE_EXEC_INTERRUPT_GRACE_S", "0.2")
    monkeypatch.setenv("AIDE_EXEC_TERM_GRACE_S", "0.5")
    monkeypatch.setenv("AIDE_EXEC_KILL_GRACE_S", "0.5")


def _wait_pid_gone(pid: int, timeout: float = 2.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        time.sleep(0.05)
    return False


def test_completed_session_exits_cooperatively(tmp_path: Path) -> None:
    interp = Interpreter(tmp_path, timeout=5)
    result = interp.run("print('finished')")
    pgid = interp._pgid

    interp.cleanup_session()

    assert result.exc_type is None
    assert "finished" in "".join(result.term_out)
    assert interp.last_cleanup_stage == "cooperative"
    assert interp.last_exitcode == 0
    assert not _process_group_exists(pgid)


def test_four_sessions_spawn_safely_from_bon_threads(tmp_path: Path) -> None:
    """BoN starts real interpreters concurrently from worker threads."""
    interps = []
    for i in range(4):
        workdir = tmp_path / f"cand_{i}"
        workdir.mkdir()
        interps.append(Interpreter(workdir, timeout=5))

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(
            pool.map(
                lambda item: item[1].run(f"print('candidate-{item[0]}')"),
                enumerate(interps),
            )
        )
    pgids = [interp._pgid for interp in interps]
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda interp: interp.cleanup_session(), interps))

    assert all(result.exc_type is None for result in results)
    assert all(
        f"candidate-{i}" in "".join(result.term_out)
        for i, result in enumerate(results)
    )
    assert all(interp.last_cleanup_stage == "cooperative" for interp in interps)
    assert all(not _process_group_exists(pgid) for pgid in pgids)


def test_cleanup_reaps_descendant_process_group(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _short_teardown(monkeypatch)
    interp = Interpreter(tmp_path, timeout=5)
    result = interp.run(
        """
import signal
import subprocess
import sys

p = subprocess.Popen([
    sys.executable,
    "-c",
    "import signal,time; "
    "signal.signal(signal.SIGINT, signal.SIG_IGN); "
    "time.sleep(60)",
])
print(f"DESCENDANT_PID={p.pid}")
"""
    )
    descendant_pid = int(
        "".join(result.term_out).split("DESCENDANT_PID=", 1)[1].splitlines()[0]
    )
    pgid = interp._pgid

    interp.cleanup_session()

    assert _wait_pid_gone(descendant_pid)
    assert not _process_group_exists(pgid)


def test_stale_run_thread_cannot_kill_a_newer_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """run()'s own cleanup must claim only the session it started.

    ParallelInterpreter cancels the real child from the main thread and returns
    the simulator result WITHOUT awaiting the real future
    (parallel_interpreter.py:150), so that thread keeps polling for up to 1s.
    When its child died without reporting state:finished (the SIGTERM/SIGKILL
    rungs, or an OOM kill) it reaches the died-unexpectedly branch — by which
    point validate_on_best (run.py:674) may already have published a NEW child
    on this same Interpreter. An unscoped claim there killed that new child and
    failed the step with a spurious RuntimeError.

    Driven directly rather than by racing two threads: the natural interleaving
    only lands in the window ~1 time in 8, so a timing-based test would pass
    vacuously most runs.
    """
    _short_teardown(monkeypatch)
    interp = Interpreter(tmp_path, timeout=60)

    # Session 1, cancelled the way ParallelInterpreter cancels it.
    interp.create_process()
    stale = interp.process
    interp.cleanup_session()
    assert interp.process is None

    # Session 2 — what validate_on_best starts moments later.
    marker = tmp_path / "started"
    code = """
import time
from pathlib import Path

Path("started").write_text("ready")
time.sleep(30)
"""
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(interp.run, code)
        deadline = time.monotonic() + 10
        while not marker.exists() and time.monotonic() < deadline:
            time.sleep(0.05)
        assert marker.exists()
        live_pgid = interp._pgid

        # The abandoned thread's late cleanup, arriving after session 2 is up.
        interp.cleanup_session(expect=stale)

        assert interp.process is not None, "stale claim tore down the new session"
        assert _process_group_exists(live_pgid)

        # An unscoped caller (ParallelInterpreter, BoN, run teardown) still reaps it.
        interp.cleanup_session()
        result = future.result(timeout=20)

    assert interp.process is None
    assert not _process_group_exists(live_pgid)
    # Torn down mid-run: TimeoutError when the child rides the SIGINT rung and
    # still reports state:finished, RuntimeError when it dies without reporting.
    assert result.exc_type in {"TimeoutError", "RuntimeError"}


def test_unresponsive_candidate_escalates_without_unbounded_join(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _short_teardown(monkeypatch)
    interp = Interpreter(tmp_path, timeout=60)
    marker = tmp_path / "started"
    code = """
import signal
import time
from pathlib import Path

signal.signal(signal.SIGINT, signal.SIG_IGN)
signal.signal(signal.SIGTERM, signal.SIG_IGN)
Path("started").write_text("ready")
while True:
    time.sleep(1)
"""

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(interp.run, code)
        deadline = time.monotonic() + 10
        while not marker.exists() and time.monotonic() < deadline:
            time.sleep(0.05)
        assert marker.exists()

        started = time.monotonic()
        interp.cleanup_session()
        result = future.result(timeout=10)

    assert time.monotonic() - started < 5
    assert interp.last_cleanup_stage == "SIGKILL"
    assert interp.last_exitcode == -signal.SIGKILL
    assert result.exc_type == "RuntimeError"


def test_session_handles_are_published_atomically(tmp_path: Path) -> None:
    """create_process() must publish process+queues+pgid as one group.

    Regression: the queues were assigned before the _cleanup_lock block, so a
    cleanup_session() that took the lock in that window claimed the OLD process
    but the NEW code_inq — sending the shutdown sentinel into the incoming
    session (whose child then read None as its first message and exited) while
    the outgoing child never got its sentinel at all.
    """
    interp = Interpreter(tmp_path, timeout=5)
    interp.create_process()
    first_process, first_code_inq = interp.process, interp.code_inq

    released = threading.Event()
    holding = threading.Event()

    def hold_lock() -> None:
        with interp._cleanup_lock:
            holding.set()
            released.wait(10)

    holder = threading.Thread(target=hold_lock, daemon=True)
    holder.start()
    assert holding.wait(5)

    with ThreadPoolExecutor(max_workers=1) as pool:
        creator = pool.submit(interp.create_process)
        # The lock is held, so NOTHING of the new session may be visible yet.
        time.sleep(0.5)
        assert interp.process is first_process
        assert interp.code_inq is first_code_inq
        released.set()
        creator.result(timeout=30)

    holder.join(timeout=5)
    assert interp.process is not first_process
    assert interp.code_inq is not first_code_inq
    interp.cleanup_session()

    # The first session was displaced without a cleanup_session(); retire it by
    # hand so pytest's exit-time join() of non-daemon children can't hang.
    first_code_inq.put(None)
    first_process.join(timeout=10)
    if first_process.exitcode is None:
        first_process.terminate()
        first_process.join(timeout=5)


def test_concurrent_cleanup_does_not_poison_new_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A stale run thread's scoped cleanup must not disturb the next session.

    Mirrors ParallelInterpreter abandoning its real future while run.py's
    validate_on_best publishes a new child on the same shared Interpreter.
    """
    _short_teardown(monkeypatch)
    interp = Interpreter(tmp_path, timeout=10)

    for i in range(8):
        # Retire the session run() published last round before displacing it.
        interp.cleanup_session()
        interp.create_process()
        stale = interp.process
        with ThreadPoolExecutor(max_workers=1) as pool:
            killer = pool.submit(interp.cleanup_session, stale)
            result = interp.run(f"print('session-{i}')")
            killer.result(timeout=30)
        assert result.exc_type is None, result.term_out
        assert f"session-{i}" in "".join(result.term_out)

    interp.cleanup_session()
