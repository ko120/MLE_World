"""Execute Python snippets in a subprocess with captured output and time limits."""

import codecs
import ctypes
import fcntl
import io
import logging
import multiprocessing as mp
import os
import pickle
import queue
import select
import signal
import struct
import sys
import termios
import threading
import time
import traceback
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import humanize
from dataclasses_json import DataClassJsonMixin

logger = logging.getLogger("aide")

# Output is always text; completion is a separate, non-text queue message.
_OUTPUT_EOF = None
# Distinct from None so a caller with no handle cannot claim a newer session.
_ANY_SESSION = object()


@dataclass
class ExecutionResult(DataClassJsonMixin):
    """Captured output, execution time, and exception or prediction metadata."""

    term_out: list[str]
    exec_time: float
    exc_type: str | None
    exc_info: dict | None = None
    exc_stack: list[tuple] | None = None
    # True iff this result came from the real subprocess, not the simulator.
    real_won: bool = False
    # Simulator-only prediction of whether the code wrote a submission.
    predicted_submission_exists: str | None = None
    # Wall-clock kill limit (s) run() actually enforced for THIS execution.
    timeout_s: float | None = None
    # Validation metric from a structured world-model response. This is a
    # prediction used for candidate selection, never a measured/test score.
    predicted_metric: float | None = None
    # Legacy recovery metadata; accepted when reading older execution records.
    submission_files_before: dict | None = None
    submission_processing: dict | None = None
    # Real CSV presence captured before an isolated race workspace is removed.
    # None means no evidence was captured, never a positive observation.
    observed_submission_exists: bool | None = None


def exception_summary(
    e: BaseException,
    working_dir: Path,
    exec_file_name: str,
    format_tb_ipython: bool,
) -> tuple[str, str, dict, list[tuple]]:
    """Format an exception and collect its attributes and traceback frames."""
    if format_tb_ipython:
        import IPython.core.ultratb

        # Skip the interpreter's own execution frame.
        formatter = IPython.core.ultratb.VerboseTB(tb_offset=1, color_scheme="NoColor")
        tb_str = str(formatter.text(*sys.exc_info()))
    else:
        tb_lines = traceback.format_exception(e)
        # Hide interpreter and import machinery from the displayed traceback.
        tb_str = "".join(
            line for line in tb_lines if "aide/" not in line and "importlib" not in line
        )

    # Keep the workspace path out of displayed tracebacks.
    tb_str = tb_str.replace(str(working_dir / exec_file_name), exec_file_name)

    exc_info = {}
    if hasattr(e, "args"):
        exc_info["args"] = [str(arg) for arg in e.args]
    for attribute in ("name", "msg", "obj"):
        if hasattr(e, attribute):
            exc_info[attribute] = str(getattr(e, attribute))

    frames = traceback.extract_tb(e.__traceback__)
    exc_stack = [(frame.filename, frame.lineno, frame.name, frame.line) for frame in frames]

    return tb_str, e.__class__.__name__, exc_info, exc_stack


class _FdOutputCapture:
    """Redirect fd 1/2 to a pipe drained by the supervising process.

    Keep the same pipe throughout a session so persistent forkservers still
    have valid output descriptors on later executions. No reader thread runs
    in this worker, and bytes already written survive an abrupt worker death.
    """

    def __init__(self, pipe: Any) -> None:
        self._pipe = pipe

    def __enter__(self) -> "_FdOutputCapture":
        self._original_streams = (sys.stdout, sys.stderr, sys.__stdout__, sys.__stderr__)
        self._flush_python(self._original_streams)
        self._saved_fds = (os.dup(1), os.dup(2))
        os.dup2(self._pipe.fileno(), 1)
        os.dup2(self._pipe.fileno(), 2)
        self._streams = tuple(
            io.TextIOWrapper(
                io.FileIO(fd, "w", closefd=False),
                encoding="utf-8", errors="replace", write_through=True,
            )
            for fd in (1, 2)
        )
        sys.stdout, sys.stderr = self._streams
        sys.__stdout__, sys.__stderr__ = self._streams
        return self

    @staticmethod
    def _flush_python(streams: Any) -> None:
        for stream in streams:
            try:
                stream.flush()
            except (AttributeError, OSError, ValueError):
                # Generated code may replace or close a standard stream.
                pass

    def flush(self) -> None:
        self._flush_python((sys.stdout, sys.stderr, *self._streams, *self._original_streams))
        # Native stdio is separately buffered; normal script exit flushes it.
        try:
            ctypes.CDLL(None).fflush(None)
        except (AttributeError, OSError):
            pass

    def __exit__(self, *_exc: Any) -> None:
        sys.stdout, sys.stderr, sys.__stdout__, sys.__stderr__ = self._original_streams
        for fd, saved in zip((1, 2), self._saved_fds):
            os.dup2(saved, fd)
            os.close(saved)
        for stream in self._streams:
            stream.close()
        self._pipe.close()


class _QueueReader:
    """Parent-side queue reader with a deadline for the entire message.

    Queue.get(timeout=...) only bounds the initial poll; its subsequent read
    blocks forever if a killed producer left a partial message in the pipe.
    Read the POSIX multiprocessing.Connection framing incrementally instead.
    Keep partial bytes across polls, and release the queue's capacity only
    after receiving a complete message, just as Queue.get does.
    """

    def __init__(self, source: Any) -> None:
        self._source = source
        self._buffer = bytearray()
        os.set_blocking(source._reader.fileno(), False)

    def get(self, timeout: float) -> Any:
        deadline = time.monotonic() + max(0.0, timeout)
        if not self._source._rlock.acquire(timeout=max(0.0, timeout)):
            raise queue.Empty
        try:
            return self._receive(deadline)
        finally:
            self._source._rlock.release()

    def _receive(self, deadline: float) -> Any:
        reader = self._source._reader
        while True:
            if len(self._buffer) >= 4:
                size = struct.unpack_from("!i", self._buffer)[0]
                header_size = 4
                if size == -1 and len(self._buffer) >= 12:
                    size = struct.unpack_from("!Q", self._buffer, 4)[0]
                    header_size = 12
                if size < -1:
                    raise OSError("invalid multiprocessing message length")
                if size >= 0 and len(self._buffer) >= header_size + size:
                    end = header_size + size
                    payload = self._buffer[header_size:end]
                    del self._buffer[:end]
                    self._source._sem.release()
                    return pickle.loads(payload)

            remaining = deadline - time.monotonic()
            if remaining <= 0 or not reader.poll(remaining):
                raise queue.Empty
            try:
                chunk = os.read(reader.fileno(), 65536)
            except BlockingIOError:
                continue
            if not chunk:
                raise queue.Empty
            self._buffer.extend(chunk)


class _OutputReader:
    """Drain one execution's output while the parent waits for state events.

    Reading only after state:finished can back up the output pipe and block
    workers, including code that joins a child producing a large output.
    """

    def __init__(self, source: Any, pipe: Any) -> None:
        self._source = source
        self._pipe = pipe
        self._buffer: queue.Queue[str | None | Exception] = queue.Queue()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._drain, name="aide-output-reader", daemon=True)

    def __enter__(self) -> "_OutputReader":
        self._thread.start()
        return self

    def __exit__(self, *_exc: Any) -> None:
        self._stop.set()
        # _QueueReader bounds partial-message reads; cancellation must also
        # remain bounded if this thread is deserializing a very large chunk.
        self._thread.join(timeout=1)

    def get(self, timeout: float) -> str | None:
        item = self._buffer.get(timeout=timeout)
        if isinstance(item, Exception):
            raise item
        return item

    def _drain(self) -> None:
        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        fd = self._pipe.fileno()
        control = self._source._source._reader
        poller = select.poll()
        poller.register(fd, select.POLLIN)
        poller.register(control.fileno(), select.POLLIN)
        remaining: int | None = None
        traceback_text = None
        pipe_closed = False
        while not self._stop.is_set():
            try:
                if remaining is None and control.poll(0):
                    try:
                        # This queue carries only completion and traceback text;
                        # printed text never enters the control protocol.
                        traceback_text = self._source.get(timeout=0.1)
                    except queue.Empty:
                        pass
                    else:
                        # All main/finished-child writes precede completion.
                        # A finite snapshot avoids waiting for daemon pipe EOF.
                        remaining = struct.unpack(
                            "i", fcntl.ioctl(fd, termios.FIONREAD, struct.pack("i", 0))
                        )[0]
                if remaining == 0:
                    tail = decoder.decode(b"", final=True)
                    if tail:
                        self._buffer.put(tail)
                    if traceback_text:
                        self._buffer.put(traceback_text)
                    self._buffer.put(_OUTPUT_EOF)
                    return
                if pipe_closed:
                    # A killed writer may never send completion. Keep reads
                    # bounded while the supervising thread detects its death.
                    control.poll(0.1)
                    continue
                try:
                    chunk = os.read(fd, 65536 if remaining is None else min(65536, remaining))
                except BlockingIOError:
                    poller.poll(100)
                    continue
                if not chunk:
                    pipe_closed = True
                    tail = decoder.decode(b"", final=True)
                    if tail:
                        self._buffer.put(tail)
                    decoder.reset()
                    continue
                if remaining is not None:
                    remaining -= len(chunk)
                text = decoder.decode(chunk)
                if text:
                    self._buffer.put(text)
            except Exception as error:
                # Surface read failures through get(), on the supervising thread.
                self._buffer.put(error)
                return


def _apply_task_resource_caps() -> None:
    """Apply best-effort memory limits before generated code runs.

    AIDE_MEM_PER_TASK_GB sets RLIMIT_DATA in GiB (default 128; 0 disables).
    This bounds heap and anonymous mmap allocations without limiting CUDA's
    virtual address reservations or file-backed mappings. oom_score_adj=1000
    makes the worker a preferred OOM victim if the host still runs out of RAM.

    Limits apply per process and are inherited by forked descendants; the
    launchers' memory watchdog bounds the whole tree. Ignore failures so hosts
    that refuse limits can still execute code without polluting term_out.
    """
    try:
        memory_gb = float(os.environ.get("AIDE_MEM_PER_TASK_GB") or 128)
    except ValueError:
        memory_gb = 128.0
    if memory_gb > 0:
        try:
            import resource

            cap = int(memory_gb * 1024**3)
            resource.setrlimit(resource.RLIMIT_DATA, (cap, cap))
        except (ImportError, ValueError, OSError):
            pass
    try:
        with open("/proc/self/oom_score_adj", "w") as f:
            f.write("1000")
    except OSError:
        pass


def _finish_child_processes() -> None:
    """Finish non-daemon child work before reporting a script's completion.

    Process-pool workers wait for a shutdown request even after their tasks
    finish. Ask loaded stdlib/loky executors to shut down first, as their
    interpreter-exit hooks would. Using each executor's public shutdown keeps
    those libraries usable by later snippets in a reused session.

    The supervising parent still enforces the execution timeout and signals
    the whole group. Remember SIGINT while reaping children so runfile.py stays
    available until they exit; resistant children reach the parent's existing
    TERM/KILL escalation rather than causing premature success or file removal.
    """
    interrupted: KeyboardInterrupt | None = None
    manager_types = tuple(
        manager_type
        for name in (
            "concurrent.futures.process",
            "joblib.externals.loky.process_executor",
            "loky.process_executor",
        )
        if isinstance(
            manager_type := getattr(sys.modules.get(name), "_ExecutorManagerThread", None),
            type,
        )
    )
    for thread in threading.enumerate():
        if isinstance(thread, manager_types):
            executor = thread.executor_reference()
            if executor is not None:
                try:
                    executor.shutdown(wait=True)
                except KeyboardInterrupt as error:
                    interrupted = error

    while True:
        children = [child for child in mp.active_children() if not child.daemon]
        if not children:
            break
        for child in children:
            try:
                child.join()
            except KeyboardInterrupt as error:
                interrupted = error
    if interrupted is not None:
        raise interrupted


def _run_session_worker(
    working_dir_str: str,
    agent_file_name: str,
    format_tb_ipython: bool,
    code_inq: Any,
    result_outq: Any,
    event_outq: Any,
    control_outq: Any,
    output_pipe: Any,
) -> None:
    """Fresh-interpreter entry point for generated code.

    This must remain module-level so the ``spawn`` multiprocessing context can
    import it without pickling the parent ``Interpreter`` (which owns locks and
    process handles).  The worker becomes a process-group leader before running
    generated code; DataLoader and similar descendants inherit that group.
    """
    try:
        os.setsid()
    except OSError:
        # This should not happen for a freshly spawned child, but report the
        # actual group so the parent can safely fall back to leader-only signals
        # instead of ever signaling its own process group.
        pass
    control_outq.put(("state:spawned", os.getpid(), os.getpgrp()))
    _apply_task_resource_caps()

    # Spawn bootstrap already selected a multiprocessing context. Give the
    # generated script the same unset context as a fresh Python invocation so
    # it can call set_start_method() without force=True. Otherwise it uses the
    # platform default (fork on the deployed Linux/Python 3.11 environment).
    # This does not change the explicit spawn context used by create_process().
    mp.set_start_method(None, force=True)

    # Disable warnings before importing generated-code dependencies.
    import shutup

    shutup.mute_warnings()
    working_dir = Path(working_dir_str)
    os.chdir(working_dir_str)
    sys.path.append(working_dir_str)

    # A registered __main__ makes exec-defined classes pickleable and enables
    # main guards. An absolute __file__ lets explicit spawn/forkserver children
    # re-import the script even after chdir(), matching `python runfile.py`.
    main_module = types.ModuleType("__main__")
    main_module.__file__ = str(working_dir / agent_file_name)
    main_module.__package__ = None
    main_module.__spec__ = None
    sys.modules["__main__"] = main_module
    global_scope: dict = main_module.__dict__
    global_scope["__builtins__"] = __builtins__
    with _FdOutputCapture(output_pipe) as output:
        while True:
            code = code_inq.get()
            if code is None:
                # Normal session shutdown. Returning (rather than receiving
                # SIGTERM) lets Python, DataLoader workers, queues, and CUDA runtime
                # state unwind in their normal order.
                return

            os.chdir(working_dir_str)
            exec_path = working_dir / agent_file_name
            with open(exec_path, "w") as f:
                f.write(code)

            # Spawn inherits AIDE's CLI arguments. Each execution is a standalone
            # script with no supplied arguments, including reused REPL sessions.
            sys.argv = [str(exec_path)]
            event_outq.put(("state:ready",))
            tb_str = None
            try:
                try:
                    exec(compile(code, agent_file_name, "exec"), global_scope)
                finally:
                    # Python normally waits for non-daemon children at script exit,
                    # including sys.exit() and exceptions. A reusable exec loop has
                    # no such exit: keep its script and output open until work ends.
                    _finish_child_processes()
            except BaseException as e:
                if isinstance(e, SystemExit) and (
                    e.code is None or (isinstance(e.code, int) and e.code == 0)
                ):
                    state = ("state:finished", None, None, None)
                else:
                    tb_str, exc_type, exc_info, exc_stack = exception_summary(
                        e,
                        working_dir,
                        agent_file_name,
                        format_tb_ipython,
                    )
                    if exc_type == "KeyboardInterrupt":
                        exc_type = "TimeoutError"
                    state = ("state:finished", exc_type, exc_info, exc_stack)
            else:
                state = ("state:finished", None, None, None)
            finally:
                exec_path.unlink(missing_ok=True)

            output.flush()
            # Output bytes are on their own pipe. This queue message marks a
            # boundary after all writes, and cannot be produced by printing.
            result_outq.put(tb_str)
            event_outq.put(state)


def _process_is_alive(process: Any) -> bool:
    """Treat a missing or concurrently closed process handle as stopped."""
    try:
        return process is not None and process.is_alive()
    except ValueError:
        return False


def _process_group_exists(pgid: int | None) -> bool:
    if pgid is None or pgid <= 1:
        return False
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _process_group_members(pgid: int | None) -> list[str]:
    """Best-effort process-state snapshot used immediately before escalation."""
    if pgid is None:
        return []
    members: list[str] = []
    proc = Path("/proc")
    try:
        entries = list(proc.iterdir())
    except OSError:
        return members
    for entry in entries:
        if not entry.name.isdigit():
            continue
        try:
            raw = (entry / "stat").read_text()
            end = raw.rfind(")")
            fields = raw[end + 2 :].split()
            # fields after comm: state, ppid, pgrp, ...
            if len(fields) < 3 or int(fields[2]) != pgid:
                continue
            state = fields[0]
            try:
                wchan = (entry / "wchan").read_text().strip()
            except OSError:
                wchan = "?"
            members.append(f"{entry.name}:{state}:{wchan}")
        except (OSError, ValueError):
            continue
    return members


def _signal_process_tree(process: Any, pgid: int | None, sig: int) -> None:
    """Signal the isolated group, falling back to its direct leader."""
    pid = getattr(process, "pid", None)
    try:
        # Only trust a group created by this worker (setsid => pgid == pid).
        if pid is not None and pgid == pid and pgid > 1:
            os.killpg(pgid, sig)
        elif pid is not None:
            os.kill(pid, sig)
    except (ProcessLookupError, PermissionError):
        pass


def _wait_for_process_tree(process: Any, pgid: int | None, timeout: float) -> bool:
    deadline = time.monotonic() + max(0.0, timeout)
    while True:
        try:
            process.join(timeout=0)
            leader_alive = process.exitcode is None
        except (AssertionError, ValueError):
            leader_alive = False
        group_alive = _process_group_exists(pgid)
        if not leader_alive and not group_alive:
            return True
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return False
        time.sleep(min(0.1, remaining))


def _env_seconds(name: str, default: float) -> float:
    try:
        return max(0.0, float(os.getenv(name, str(default))))
    except ValueError:
        return default


def _drain_queue_dump(result_outq: Any, deadline_s: float = 2.0) -> list[str]:
    """Collect and log available stdout/stderr after a child fails.

    Timed reads preserve in-flight writes that Queue.empty() can miss. The
    deadline also bounds draining a stuck child that keeps streaming output.
    """
    chunks: list[str] = []
    deadline = time.monotonic() + deadline_s
    while time.monotonic() < deadline:
        try:
            chunk = result_outq.get(timeout=0.2)
            if chunk is _OUTPUT_EOF:
                break
            chunks.append(chunk)
        except (queue.Empty, OSError, ValueError):
            break
    if chunks:
        joined = "".join(chunks)
        logger.error(
            "REPL output queue dump (%d chunks, %d chars): %s",
            len(chunks),
            len(joined),
            joined[:1000],
        )
    return chunks


class Interpreter:
    """Run Python code in a reusable subprocess with an execution time limit."""

    def __init__(
        self,
        working_dir: Path | str,
        timeout: int = 3600,
        format_tb_ipython: bool = False,
        agent_file_name: str = "runfile.py",
    ) -> None:
        """Configure the subprocess and its execution environment.

        Args:
            working_dir: Existing directory used for generated code and its files.
            timeout: Time limit in seconds for each execution.
            format_tb_ipython: Use IPython formatting for exception tracebacks.
            agent_file_name: Filename used for the generated Python script.
        """
        self.working_dir = Path(working_dir).resolve()
        assert self.working_dir.exists(), f"Working directory {self.working_dir} does not exist"
        self.timeout = timeout
        self.format_tb_ipython = format_tb_ipython
        self.agent_file_name = agent_file_name
        self.process: Any = None
        self._pgid: int | None = None
        self._cleanup_lock = threading.Lock()
        self.last_cleanup_stage: str | None = None
        self.last_exitcode: int | None = None

    def create_process(self) -> None:
        """Start a worker and record its isolated process group."""
        # Candidate execution can start from ThreadPoolExecutor threads, so
        # use spawn to avoid forking the multithreaded agent.
        ctx = mp.get_context("spawn")
        code_inq = ctx.Queue()
        result_outq = ctx.Queue()
        event_outq = ctx.Queue()
        # Keep the process-group handshake separate from execution events.
        control_outq = ctx.Queue()
        output_pipe, output_writer = ctx.Pipe(duplex=False)
        os.set_blocking(output_pipe.fileno(), False)
        process = ctx.Process(
            target=_run_session_worker,
            args=(
                str(self.working_dir),
                self.agent_file_name,
                self.format_tb_ipython,
                code_inq,
                result_outq,
                event_outq,
                control_outq,
                output_writer,
            ),
        )
        # Bound entire messages: output and exception metadata can exceed pipe
        # capacity and be truncated by a kill or concurrent cancellation.
        result_reader = _QueueReader(result_outq)
        event_reader = _QueueReader(event_outq)
        control_reader = _QueueReader(control_outq)
        # Publish the handle, queues, and disarmed pgid atomically. Cleanup
        # must never pair one session's process with another's group or queue.
        with self._cleanup_lock:
            self.code_inq = code_inq
            self.result_outq = result_reader
            self.event_outq = event_reader
            self.control_outq = control_reader
            self.output_pipe = output_pipe
            self.process = process
            self._pgid = None
        try:
            process.start()
        finally:
            # Only the worker and its descendants should hold a write end.
            output_writer.close()
        try:
            # Use the local queue in case another session is already published.
            state, pid, pgid = control_reader.get(
                timeout=_env_seconds("AIDE_EXEC_SPAWN_TIMEOUT_S", 60.0)
            )
            if state != "state:spawned" or pid != process.pid:
                raise RuntimeError(f"invalid interpreter spawn handshake: {(state, pid, pgid)}")
            if pgid == pid:
                # Arm group cleanup only if this child still owns the session.
                # A concurrent cleanup may already have claimed it leader-only.
                with self._cleanup_lock:
                    if self.process is process:
                        self._pgid = pgid
            else:
                logger.error(
                    "REPL child failed to isolate its process group "
                    "(pid=%s pgid=%s); teardown will signal only the leader",
                    pid,
                    pgid,
                )
        except queue.Empty:
            logger.error(
                "REPL child did not report process-group setup (pid=%s)",
                process.pid,
            )

    def cleanup_session(self, expect: Any = _ANY_SESSION) -> None:
        """Tear down the child process tree.

        Omit `expect` to claim the current session. Internal callers pass their
        process handle so a cancelled run cannot tear down a newer session
        published while its thread was still draining output.
        """
        # Claim once under the publication lock; concurrent cleanup returns.
        with self._cleanup_lock:
            process = self.process
            if process is None:
                return
            if expect is not _ANY_SESSION and process is not expect:
                return
            self.process = None
            pgid, self._pgid = self._pgid, None
            code_inq = self.code_inq

        pid = getattr(process, "pid", "?")
        gpu = os.getenv("NVIDIA_VISIBLE_DEVICES", os.getenv("CUDA_VISIBLE_DEVICES", "?"))
        started = time.monotonic()
        cleanup_stage = "cooperative"

        # Normal path: ask the idle REPL loop to return. If cleanup races an
        # active run, the sentinel waits behind its code and still permits a
        # natural exit if that execution finishes during the grace window.
        try:
            code_inq.put(None)
        except (OSError, ValueError):
            pass
        if _wait_for_process_tree(
            process,
            pgid,
            _env_seconds("AIDE_EXEC_GRACEFUL_SHUTDOWN_S", 30.0),
        ):
            logger.info(
                "Child process tree exited cooperatively (pid=%s pgid=%s gpu=%s elapsed=%.2fs)",
                pid,
                pgid,
                gpu,
                time.monotonic() - started,
            )
        else:
            for sig, env_name, default, label in (
                (signal.SIGINT, "AIDE_EXEC_INTERRUPT_GRACE_S", 15.0, "SIGINT"),
                (signal.SIGTERM, "AIDE_EXEC_TERM_GRACE_S", 30.0, "SIGTERM"),
                (signal.SIGKILL, "AIDE_EXEC_KILL_GRACE_S", 10.0, "SIGKILL"),
            ):
                members = _process_group_members(pgid)
                logger.warning(
                    "Child process tree still alive; sending %s "
                    "(pid=%s pgid=%s gpu=%s members=%s)",
                    label,
                    pid,
                    pgid,
                    gpu,
                    members or ["leader-only/unknown"],
                )
                _signal_process_tree(process, pgid, sig)
                if _wait_for_process_tree(process, pgid, _env_seconds(env_name, default)):
                    cleanup_stage = label
                    break
            else:
                cleanup_stage = "abandoned"
                logger.error(
                    "Child process tree unresponsive to SIGKILL; abandoning "
                    "(pid=%s pgid=%s gpu=%s members=%s). Host/container "
                    "backstop must reap it.",
                    pid,
                    pgid,
                    gpu,
                    _process_group_members(pgid),
                )

        self.last_cleanup_stage = cleanup_stage
        try:
            self.last_exitcode = process.exitcode
            if self.last_exitcode is not None:
                process.close()
        except ValueError:
            pass

    def _execution_failure(
        self,
        message: str,
        *,
        process: Any,
        timeout_s: float | None,
        result_outq: Any = None,
        start_time: float | None = None,
    ) -> ExecutionResult:
        """Capture a failed child's output and clean up only its session."""
        logger.critical(message)
        output = _drain_queue_dump(result_outq) if result_outq is not None else []
        self.cleanup_session(expect=process)
        return ExecutionResult(
            # Downstream infrastructure-error detection matches this prefix.
            term_out=[message] + output,
            # Preserve elapsed time, including output draining and cleanup.
            exec_time=0 if start_time is None else time.time() - start_time,
            exc_type="RuntimeError",
            exc_info={},
            exc_stack=[],
            real_won=True,
            timeout_s=timeout_s,
        )

    def run(self, code: str, reset_session: bool = True) -> ExecutionResult:
        recorder = getattr(self, "execution_cost_recorder", None)
        if recorder is not None:
            return recorder.run_attempt(self, code, lambda: self._run(code, reset_session))
        return self._run(code, reset_session)

    def _run(self, code: str, reset_session: bool = True) -> ExecutionResult:
        """Execute code and return captured output and execution metadata.

        By default, start a fresh session. Set reset_session=False to reuse an
        existing session's namespace; its first execution must have reset it.
        """
        logger.info("REPL is executing code (reset_session=%s)", reset_session)

        if reset_session:
            if self.process is not None:
                self.cleanup_session()
            self.create_process()
        else:
            assert self.process is not None

        # Snapshot one session under the publication lock. Concurrent cleanup
        # may clear self.process while this run is still draining its output.
        with self._cleanup_lock:
            process = self.process
            code_inq = self.code_inq
            result_outq = self.result_outq
            output_pipe = self.output_pipe
            event_outq = self.event_outq
            pgid = self._pgid
        timeout = self.timeout
        # Record the enforced limit before callers can restore self.timeout.
        timeout_s = float(timeout) if timeout is not None else None

        if not _process_is_alive(process):
            return self._execution_failure(
                "REPL child process failed to start execution",
                process=process,
                timeout_s=timeout_s,
            )

        # Keep output flowing during startup, execution, and failure cleanup.
        with _OutputReader(result_outq, output_pipe) as result_outq:
            code_inq.put(code)

            # Wait for execution to start so timeout interrupts cannot hit setup.
            try:
                state = event_outq.get(timeout=_env_seconds("AIDE_EXEC_SPAWN_TIMEOUT_S", 60.0))
            except queue.Empty:
                return self._execution_failure(
                    "REPL child process failed to start execution",
                    process=process,
                    timeout_s=timeout_s,
                    result_outq=result_outq,
                )
            assert state[0] == "state:ready", state
            start_time = time.time()

            # Death is unexpected until we have sent a timeout interrupt.
            child_in_overtime = False

            while True:
                try:
                    state = event_outq.get(timeout=1)  # wait for state:finished
                    assert state[0] == "state:finished", state
                    exec_time = time.time() - start_time
                    break
                except queue.Empty:
                    leader_alive = _process_is_alive(process)
                    if not child_in_overtime and not leader_alive:
                        try:
                            exitcode = process.exitcode
                        except ValueError:
                            exitcode = None
                        msg = f"REPL child process died unexpectedly (exitcode={exitcode})"
                        if exitcode == -signal.SIGKILL:
                            msg += (
                                " — SIGKILL from outside this harness; that is"
                                " usually the kernel OOM killer (out of memory)"
                            )
                        return self._execution_failure(
                            msg,
                            process=process,
                            timeout_s=timeout_s,
                            result_outq=result_outq,
                            start_time=start_time,
                        )

                    # Interrupt overdue code, then escalate if it does not finish.
                    if timeout is None:
                        continue
                    running_time = time.time() - start_time
                    if running_time > timeout:
                        assert reset_session, "Timeout ocurred in interactive session"

                        # Interrupt the whole candidate group so DataLoader workers
                        # do not outlive an interrupted CUDA-owning leader.
                        if not child_in_overtime:
                            _signal_process_tree(process, pgid, signal.SIGINT)
                            child_in_overtime = True
                        if running_time > timeout + 60:
                            logger.warning("Child failed to terminate, killing it..")
                            self.cleanup_session(expect=process)

                            state = (None, "TimeoutError", {}, [])
                            exec_time = timeout
                            break

            output: list[str] = []
            # Bound each read: a killed producer may never write the EOF marker.
            while True:
                try:
                    chunk = result_outq.get(timeout=5)
                except (queue.Empty, OSError, ValueError):
                    break
                if chunk is _OUTPUT_EOF:
                    break
                output.append(chunk)

            exc_type, exc_info, exc_stack = state[1:]

            if exc_type == "TimeoutError":
                output.append(
                    f"TimeoutError: Execution exceeded the time limit of {humanize.naturaldelta(timeout)}"
                )
            else:
                output.append(
                    f"Execution time: {humanize.naturaldelta(exec_time)} seconds (time limit is {humanize.naturaldelta(timeout)})."
                )
            return ExecutionResult(
                output,
                exec_time,
                exc_type,
                exc_info,
                exc_stack,
                real_won=True,
                timeout_s=timeout_s,
            )
