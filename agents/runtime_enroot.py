"""Enroot provisioning, isolated agent/grader execution, and bounded teardown.

Each task gets separate copy-on-write roots for agent and grader. Cleanup must
finish before its worker admits another task; cleanup failures stop admission.
"""

import json
import logging
import math
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

from agents.registry import Agent
from agents.runtime_common import (
    CONSTANTS,
    ENROOT_COMMAND_STOP_GRACE_SECS,
    ENROOT_DRIVER_TIMEOUT_SECS,
    ENROOT_EXIT_GRACE_SECS,
    ENROOT_PROVISION_TIMEOUT_SECS,
    ENROOT_REMOVE_WARN_SECS,
    _agent_cmd,
    _agent_env,
)
from agents.snapshots import _grading_snapshot
from mlebench.registry import Competition
from mlebench.utils import purple

# Bound concurrent unpack/clone operations to avoid exhausting the host file table.
_ENROOT_CREATE_SEM = threading.Semaphore(
    max(1, int(os.environ.get("AIDE_ENROOT_CREATE_CONCURRENCY", "6")))
)


# --------------------------------------------------------------------------
# Enroot runtime: rootfs provisioning
# --------------------------------------------------------------------------
# Unpack once per process, then give each task separate writable reflink clones.
# File contents share the base's storage until written, while inode changes stay
# isolated. Reflinks are required: full copies can exhaust node-local storage.
_ENROOT_BASE_LOCK = threading.Lock()
_ENROOT_BASE_NAMES: dict[tuple, str] = {}
_ENROOT_REFLINK_LOCK = threading.Lock()
_ENROOT_REFLINK_OK: dict[Path, bool] = {}


class EnrootCleanupError(RuntimeError):
    """Cleanup left resources behind; the scheduler must stop admitting tasks."""


def _enroot_seconds(variable: str, default: float) -> float:
    value = float(os.environ.get(variable, "") or default)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{variable} must be positive and finite")
    return value


def _remaining(deadline: float, phase: str) -> float:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError(f"Enroot {phase} exceeded its deadline")
    return remaining


@contextmanager
def _enroot_gate(gate, deadline: float, phase: str):
    if not gate.acquire(timeout=_remaining(deadline, phase)):
        raise TimeoutError(f"Timed out waiting for Enroot {phase}")
    try:
        yield
    finally:
        gate.release()


def _enroot_data_path() -> Path:
    """Harness storage, explicitly passed to every Enroot subprocess.

    ENROOT_DATA_PATH in the environment selects storage. Otherwise use the XDG
    default. Pinning this value in the child environment prevents enroot.conf
    from sending creates/removals to a different tree than our reflink clones.
    """
    explicit = os.environ.get("ENROOT_DATA_PATH")
    xdg = os.environ.get("XDG_DATA_HOME") or os.path.expanduser("~/.local/share")
    return Path(explicit or str(Path(xdg) / "enroot")).expanduser().resolve()


def _enroot_host_env(data_path: Path) -> dict[str, str]:
    return {**os.environ, "ENROOT_DATA_PATH": str(data_path)}


def _run_enroot_command(
    command: list[str],
    logger,
    *,
    phase: str,
    timeout: float | None,
    warn_after: float | None = None,
    env: dict[str, str] | None = None,
    quiet: bool = False,
) -> None:
    """Bound provisioning; with timeout=None, warn while waiting for cleanup.

    No pipes: descendants holding an output fd cannot extend the deadline.
    Unlike subprocess.run(timeout=...), this also kills unsquashfs children.
    Commands passed here must not contain credentials.
    """
    started = time.monotonic()
    proc = subprocess.Popen(
        command,
        env=env,
        start_new_session=True,
        stdout=subprocess.DEVNULL if quiet else None,
        stderr=subprocess.DEVNULL if quiet else None,
    )
    try:
        while True:
            try:
                proc.wait(timeout=timeout if timeout is not None else warn_after)
                break
            except subprocess.TimeoutExpired:
                if timeout is not None:
                    raise
                logger.warning(
                    f"[enroot] {phase} still running after {time.monotonic() - started:.1f}s; "
                    "waiting for cleanup to finish"
                )
        if proc.returncode:
            raise subprocess.CalledProcessError(proc.returncode, command)
    except BaseException as exc:
        _signal_enroot_pg(proc, signal.SIGTERM)
        try:
            proc.wait(timeout=ENROOT_COMMAND_STOP_GRACE_SECS)
        except subprocess.TimeoutExpired:
            pass
        # Even if the leader exited after TERM, its children may still be alive.
        _signal_enroot_pg(proc, signal.SIGKILL)
        try:
            proc.wait(timeout=ENROOT_COMMAND_STOP_GRACE_SECS)
        except subprocess.TimeoutExpired:
            logger.error(f"Enroot {phase} process survived SIGKILL; abandoning the wait")
        if isinstance(exc, subprocess.TimeoutExpired) and timeout is not None:
            raise TimeoutError(f"Enroot {phase} exceeded its {timeout:g}s deadline") from exc
        raise
    finally:
        logger.info(f"[enroot] {phase}: {time.monotonic() - started:.2f}s")


def _remove_enroot_container(name: str, data_path: Path, logger) -> bool:
    """Wait for removal to finish; slow deletion warns instead of being killed."""
    try:
        if not (data_path / name).exists():
            return True
        variable = (
            "AIDE_ENROOT_REMOVE_WARN_SECS"
            if os.environ.get("AIDE_ENROOT_REMOVE_WARN_SECS")
            else "AIDE_ENROOT_REMOVE_TIMEOUT_SECS"  # Legacy name now sets a warning interval.
        )
        try:
            warn_after = _enroot_seconds(variable, ENROOT_REMOVE_WARN_SECS)
        except ValueError:
            warn_after = ENROOT_REMOVE_WARN_SECS
            logger.warning(f"Invalid {variable}; using a {warn_after}s cleanup warning interval")
        _run_enroot_command(
            ["enroot", "remove", "-f", name],
            logger,
            phase=f"remove {name}",
            timeout=None,
            warn_after=warn_after,
            env=_enroot_host_env(data_path),
        )
        if (data_path / name).exists():
            raise RuntimeError("enroot remove exited successfully but the rootfs still exists")
    except Exception as exc:
        logger.error(f"Container {name} could not be removed: {exc}")
        return False
    logger.info(f"Container {name} removed.")
    return True


def _reflink_supported(data_path: Path, logger, deadline: float) -> bool:
    """Probe each storage path with a real reflink operation, once per process."""
    with _enroot_gate(_ENROOT_REFLINK_LOCK, deadline, "reflink probe lock"):
        if data_path in _ENROOT_REFLINK_OK:
            return _ENROOT_REFLINK_OK[data_path]
        src = data_path / f".reflink-probe-src-{os.getpid()}"
        dst = data_path / f".reflink-probe-dst-{os.getpid()}"
        try:
            data_path.mkdir(parents=True, exist_ok=True)
            src.write_bytes(bytes(4096))
            _run_enroot_command(
                ["cp", "--reflink=always", str(src), str(dst)],
                logger,
                phase="reflink probe",
                timeout=_remaining(deadline, "reflink probe"),
                quiet=True,
            )
            _ENROOT_REFLINK_OK[data_path] = True
        except (subprocess.CalledProcessError, OSError) as exc:
            # A timeout is transient and must remain an error, not a cached
            # assertion that the filesystem cannot reflink.
            if isinstance(exc, TimeoutError):
                raise
            _ENROOT_REFLINK_OK[data_path] = False
            logger.error(f"[enroot] reflink unavailable at {data_path}: {exc}")
        finally:
            for probe in (src, dst):
                try:
                    probe.unlink(missing_ok=True)
                except OSError:
                    pass
        return _ENROOT_REFLINK_OK[data_path]


def _ensure_enroot_base(sqsh_path: str, data_path: Path, logger, deadline: float) -> str:
    """Reuse only the same image version on the same storage path."""
    image = Path(sqsh_path).resolve()
    stat = image.stat()
    key = (data_path, str(image), stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
    with _enroot_gate(_ENROOT_BASE_LOCK, deadline, "shared base lock"):
        cached = _ENROOT_BASE_NAMES.get(key)
        if cached is not None and (data_path / cached).is_dir():
            return cached
        # A failed extraction must not poison the next attempt with a partial
        # directory. Publish in the cache only after a successful extraction.
        name = f"aide-base-{os.environ.get('SLURM_JOB_ID', 'local')}-{os.getpid()}-{uuid.uuid4().hex[:8]}"
        logger.info(purple(f"[enroot] unpacking shared base {name} (once per image/process)"))
        try:
            _run_enroot_command(
                ["enroot", "create", "--name", name, str(image)],
                logger,
                phase="base extraction",
                timeout=_remaining(deadline, "base extraction"),
                env=_enroot_host_env(data_path),
            )
        except BaseException as exc:
            if not _remove_enroot_container(name, data_path, logger):
                raise EnrootCleanupError(
                    f"Could not remove partial base {name}; original provisioning error: {exc}"
                ) from exc
            raise
        _ENROOT_BASE_NAMES[key] = name
        return name


def _provision_enroot_container(name: str, sqsh_path: str, logger, data_path: Path) -> None:
    """Reflink a base with one deadline covering lock waits, extraction and copy."""
    deadline = time.monotonic() + _enroot_seconds(
        "AIDE_ENROOT_PROVISION_TIMEOUT_SECS", ENROOT_PROVISION_TIMEOUT_SECS
    )
    if os.environ.get("AIDE_ENROOT_NO_COW") == "1":
        raise RuntimeError(
            "AIDE_ENROOT_NO_COW=1 is no longer supported: per-task full unpacks "
            "can exhaust node storage. Use a reflink-capable ENROOT_DATA_PATH."
        )
    if not _reflink_supported(data_path, logger, deadline):
        raise RuntimeError(
            f"Enroot requires reflink support at {data_path}; refusing a full unpack "
            "per task. Set ENROOT_DATA_PATH to node-local storage supporting reflinks."
        )
    base = _ensure_enroot_base(sqsh_path, data_path, logger, deadline)
    with _enroot_gate(_ENROOT_CREATE_SEM, deadline, "clone slot"):
        _run_enroot_command(
            ["cp", "-a", "--reflink=always", str(data_path / base), str(data_path / name)],
            logger,
            phase="rootfs clone",
            timeout=_remaining(deadline, "rootfs clone"),
        )


# --------------------------------------------------------------------------
# Enroot runtime: teardown
# --------------------------------------------------------------------------
# enroot shares the host PID namespace. `enroot start` runs in its own session
# (pgid == pid) so the whole container tree can be signalled; AIDE candidate
# interpreters deliberately leave that group, so a unique AIDE_RUN_TOKEN in
# every descendant's environment is the final ownership boundary.


def _signal_enroot_pg(proc: subprocess.Popen, sig: int) -> None:
    try:
        os.killpg(proc.pid, sig)
    except (ProcessLookupError, PermissionError):
        pass


def _pids_with_run_token(token: str) -> list[int]:
    """Processes (any group) whose environment carries this run's token."""
    needle = f"AIDE_RUN_TOKEN={token}".encode()
    found: list[int] = []
    try:
        entries = list(Path("/proc").iterdir())
    except OSError:
        return found
    for entry in entries:
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            environ = (entry / "environ").read_bytes().split(b"\0")
        except (OSError, PermissionError):
            continue
        if needle in environ:
            found.append(int(entry.name))
    return sorted(found)


def _sweep_run_token_processes(token: str, logger: logging.Logger) -> bool:
    """Bounded SIGINT -> SIGTERM -> SIGKILL sweep of token-carrying survivors."""
    stages = (
        (signal.SIGINT, 10.0, "SIGINT"),
        (signal.SIGTERM, 20.0, "SIGTERM"),
        (signal.SIGKILL, 10.0, "SIGKILL"),
    )
    for sig, grace_s, label in stages:
        pids = _pids_with_run_token(token)
        if not pids:
            return True
        logger.warning(
            f"Container descendant backstop: sending {label} to token={token} pids={pids}"
        )
        for pid in pids:
            try:
                os.kill(pid, sig)
            except (ProcessLookupError, PermissionError):
                pass
        deadline = time.monotonic() + grace_s
        while time.monotonic() < deadline:
            if not _pids_with_run_token(token):
                return True
            time.sleep(0.1)
    survivors = _pids_with_run_token(token)
    if survivors:
        logger.error(f"Container descendants survived SIGKILL: token={token} pids={survivors}")

    return not survivors


# --------------------------------------------------------------------------
# Enroot runtime: networking + GPU
# --------------------------------------------------------------------------


def _enroot_env(competition: Competition, agent: Agent, logger: logging.Logger) -> dict:
    """Same env as the docker path, with the docker-only host alias rewritten:
    enroot shares the host network namespace, so the vLLM server on the same
    node is reachable at localhost."""
    return {
        k: (v.replace("host.docker.internal", "localhost") if isinstance(v, str) else v)
        for k, v in _agent_env(competition, agent, logger).items()
    }


# GPU passthrough on hosts without libnvidia-container (alpha). enroot's
# 98-nvidia.sh hook shells out to `nvidia-container-cli` to inject the driver
# user-space; where that binary is absent the hook aborts `enroot start`. We
# reproduce it by hand: stage the driver libs into ONE private dir (never over a
# system lib dir, which would shadow the container's glibc), bind it plus the
# /dev/nvidia* nodes, and point the loader at it. Where the real hook exists
# (beta) all of this is a no-op.
_NVDRV_DIR = "/opt/nvdrv"
_nvdrv_lock = threading.Lock()
_nvdrv_stage: str | None = None  # None = not probed; "" = real hook available

# Only what CUDA compute needs; graphics/video libs are extra version-skew risk.
_NVDRV_LIB_GLOBS = (
    "libcuda.so*",
    "libnvidia-ml.so*",
    "libnvidia-nvvm.so*",
    "libnvidia-ptxjitcompiler.so*",
)
_NVDRV_LIB_DIRS = ("/lib/x86_64-linux-gnu", "/usr/lib/x86_64-linux-gnu", "/usr/lib64")


def _nvidia_driver_stage(logger: logging.Logger) -> str:
    """Stage the host NVIDIA user-space once per process; return its path, or
    "" when nvidia-container-cli exists and the normal hook should be used."""
    global _nvdrv_stage
    deadline = time.monotonic() + ENROOT_DRIVER_TIMEOUT_SECS
    with _enroot_gate(_nvdrv_lock, deadline, "NVIDIA driver staging lock"):
        if _nvdrv_stage is not None:
            return _nvdrv_stage
        if shutil.which("nvidia-container-cli"):
            _nvdrv_stage = ""
            return _nvdrv_stage

        # Sibling Python processes in one Slurm job must not overwrite a stage
        # while another process is mounting it. The launcher's data-path sweep
        # also reclaims these small process-local copies at job exit.
        stage = _enroot_data_path() / f".nvdrv-{os.getpid()}"
        stage.mkdir(parents=True, exist_ok=True)
        # cp -a keeps libcuda.so.1 as the RELATIVE symlink it is on the host;
        # resolving it would break the soname the loader looks for.
        libs = {}
        for lib_dir in _NVDRV_LIB_DIRS:
            for pattern in _NVDRV_LIB_GLOBS:
                for library in Path(lib_dir).glob(pattern):
                    # /lib often aliases /usr/lib. Keep each basename once,
                    # including symlinks, so cp can report real copy failures.
                    libs.setdefault(library.name, library)
        if libs:
            _run_enroot_command(
                ["cp", "-a", *map(str, libs.values()), str(stage)],
                logger,
                phase="NVIDIA driver staging",
                timeout=_remaining(deadline, "NVIDIA driver staging"),
            )
        smi = shutil.which("nvidia-smi")
        if smi:
            _run_enroot_command(
                ["cp", "-a", smi, str(stage)],
                logger,
                phase="nvidia-smi staging",
                timeout=_remaining(deadline, "nvidia-smi staging"),
            )
        if not (stage / "libcuda.so.1").exists():
            raise RuntimeError(
                f"NVIDIA driver staging failed: no libcuda.so.1 under {stage}. "
                f"Searched {_NVDRV_LIB_DIRS} and nvidia-container-cli is absent, "
                "so containers cannot get a GPU on this host."
            )
        logger.info(purple(f"[nvidia-fallback] staged driver -> {stage} ({len(libs)} files)"))
        _nvdrv_stage = str(stage)
        return _nvdrv_stage


def _resolve_enroot_gpus(gpus: list) -> list[str]:
    """Interpret numeric GPU selections as indices into the Slurm allocation."""
    allocation = [
        g.strip() for g in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if g.strip()
    ]
    devices = []
    for gpu in gpus:
        value = str(gpu).strip()
        try:
            index = int(value)
        except ValueError:
            devices.append(value)  # GPU/MIG UUIDs are already explicit identities.
            continue
        if index < 0 or (allocation and index >= len(allocation)):
            raise ValueError(f"GPU allocation index {index} is out of range")
        devices.append(allocation[index] if allocation else value)
    return devices


def _nvidia_manual_passthrough(
    gpus: list, logger: logging.Logger
) -> tuple[list[str], dict[str, str]]:
    """Mounts + env that replace the nvidia hook; ([], {}) where the hook works."""
    stage = _nvidia_driver_stage(logger)
    if not stage:
        return [], {}

    # Mount every device node the host exposes, then scope with
    # CUDA_VISIBLE_DEVICES (mounting only the assigned node risks enumeration gaps).
    mounts = [f"{stage}:{_NVDRV_DIR}"]
    for dev in sorted(Path("/dev").glob("nvidia*")):
        if not dev.is_dir():  # skip /dev/nvidia-caps
            mounts.append(f"{dev.as_posix()}:{dev.as_posix()}")

    phys = _resolve_enroot_gpus(gpus)

    env = {
        "CUDA_VISIBLE_DEVICES": ",".join(phys),
        "LD_LIBRARY_PATH": _NVDRV_DIR,
        "NVIDIA_DRIVER_PASSTHROUGH": "manual",
        "NVIDIA_VISIBLE_DEVICES": "void",
    }
    logger.info(
        purple(
            f"[nvidia-fallback] gpus={gpus} -> CUDA_VISIBLE_DEVICES={env['CUDA_VISIBLE_DEVICES']}"
        )
    )
    return mounts, env


# --------------------------------------------------------------------------
# Enroot runtime: the run
# --------------------------------------------------------------------------


def _enroot_mounts(
    competition: Competition, agent: Agent, run_dir: Path, logger: logging.Logger
) -> tuple[list[str], str | None]:
    """Bind mounts for one task, plus the snapshot sha for the start.sh handshake.

    x-create=dir/file: enroot does not auto-create mountpoints the way docker
    does. Output dirs are pre-created on the host so the agent writes straight
    into run_dir (names match save_output()'s docker extraction layout).
    """
    mounts = [
        f"{competition.public_dir.resolve().as_posix()}:/home/data:none:x-create=dir,bind,ro",
        # Also mask conventional data directories in older images. The grader
        # alone receives the actual private-data bind, in another rootfs.
        "tmpfs:/private:tmpfs:x-create=dir,ro,nosuid,nodev,noexec",
        "tmpfs:/mlebench/data:tmpfs:x-create=dir,ro,nosuid,nodev,noexec",
        "tmpfs:/mlebench/data_privateonly:tmpfs:x-create=dir,ro,nosuid,nodev,noexec",
    ]
    for dir_type in ("SUBMISSION_DIR", "LOGS_DIR", "CODE_DIR"):
        host_dir = run_dir / CONSTANTS[dir_type].rsplit("/", 1)[-1]
        host_dir.mkdir(parents=True, exist_ok=True)
        mounts.append(
            f"{host_dir.resolve().as_posix()}:{CONSTANTS[dir_type]}:none:x-create=dir,bind"
        )

    # Agent code always comes from the run group's immutable snapshot. Per-path
    # binds rather than a whole-dir overlay: start.sh writes into AGENT_DIR at
    # runtime, which a read-only dir bind would break.
    snapshot_sha = None
    if agent.name == "aide":
        snapshot = run_dir.parent / "agent_snapshot"
        provenance = snapshot / "PROVENANCE.json"
        if not provenance.exists():
            raise RuntimeError(
                f"agent snapshot missing at {snapshot} -- run_agent.py creates it "
                "(snapshot_agent_code) before launching workers"
            )
        snapshot_sha = json.loads(provenance.read_text())["src_aide_sha256"]
        snap = snapshot.resolve().as_posix()
        agent_dir = CONSTANTS["AGENT_DIR"]
        mounts += [
            f"{snap}/src/aide:{agent_dir}/src/aide:none:bind,ro",
            f"{snap}/start.sh:{agent_dir}/start.sh:none:bind,ro",
            f"{snap}/additional_notes.txt:{agent_dir}/additional_notes.txt:none:bind,ro",
            f"{snap}/snapshot_check.sh:{agent_dir}/snapshot_check.sh:none:x-create=file,bind,ro",
            f"{snap}/PROVENANCE.json:{agent_dir}/PROVENANCE.json:none:x-create=file,bind,ro",
        ]
    return mounts, snapshot_sha


def _enroot_grading_mounts(
    competition: Competition, agent: Agent, run_dir: Path, logger: logging.Logger
) -> tuple[list[str], Path, Path]:
    """Private inputs and control files belong only to the grader container."""
    grading_dir = Path(__file__).resolve().parents[1] / "environment"
    if agent.name == "aide":
        grading_dir = _grading_snapshot(run_dir.parent / "agent_snapshot")
    control_dir = run_dir.resolve() / "grading"
    control_dir.mkdir(mode=0o700, exist_ok=True)
    mounts = [
        f"{competition.private_dir.resolve().as_posix()}:"
        f"/private/data/{competition.id}/prepared/private:none:x-create=dir,bind,ro",
        f"{grading_dir / 'grading_server.py'}:/private/grading_server.py:"
        "none:x-create=file,bind,ro",
        f"{control_dir}:/run/grading:none:x-create=dir,bind",
    ]
    mlebench_snap = run_dir.parent / "agent_snapshot" / "mlebench"
    if mlebench_snap.is_dir():
        mounts.append(f"{mlebench_snap.resolve()}:/mlebench/mlebench:none:bind,ro")
    else:
        logger.warning(
            purple(
                f"[grading] no mlebench snapshot at {mlebench_snap} -- using the image's baked graders"
            )
        )
    return mounts, grading_dir / "grading_server_wrapper.py", control_dir


def _enroot_start_command(name: str, mounts: list[str], env: dict, command: list[str]) -> list[str]:
    # Enroot normally exposes host PIDs and /proc/<pid>/root. A new PID
    # namespace and freshly mounted procfs close that route to the grader and
    # host files. Namespace-init exit also kills descendants that called setsid.
    result = [
        "unshare",
        "--user",
        "--map-root-user",
        "--pid",
        "--fork",
        "--kill-child=KILL",
        "--mount-proc",
        "--",
        "enroot",
        "start",
        "--root",
        "--rw",
    ]
    for key, value in env.items():
        result += ["--env", key if value is None else f"{key}={value}"]
    for mount in mounts:
        result += ["--mount", mount]
    return [*result, name, *command]


def _wait_for_container(
    proc: subprocess.Popen, budget: int | None, time_limit: str, logger: logging.Logger
) -> bool:
    """Wait for `enroot start`, escalating SIGTERM -> SIGKILL on its process
    group once the watchdog budget passes. Returns True if the watchdog fired.
    A group that survives SIGKILL (uninterruptible I/O) is abandoned so the
    worker frees up; SLURM's job-exit cgroup sweep reaps the remains."""
    try:
        proc.wait(timeout=budget)
        return False
    except subprocess.TimeoutExpired:
        pass
    logger.error(
        f"Container still running {ENROOT_EXIT_GRACE_SECS}s past the agent deadline "
        f"(TIME_LIMIT_SECS={time_limit}) -- killing its process group"
    )
    _signal_enroot_pg(proc, signal.SIGTERM)
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        _signal_enroot_pg(proc, signal.SIGKILL)
    try:
        proc.wait(timeout=120)
    except subprocess.TimeoutExpired:
        logger.error(
            "Container process group survived SIGKILL (uninterruptible I/O?); "
            "abandoning the wait"
        )
    return True


def run_in_enroot(
    competition: Competition,
    agent: Agent,
    sqsh_path: str,
    retain_container: bool,
    run_dir: Path,
    logger: logging.Logger,
    gpus: list,
    cpus: str | None = None,
) -> Path:
    """Run a task with bounded provisioning/execution, then wait for cleanup.

    CPU affinity and GPU placement retain the launcher's allocation policy.
    Outputs stay in host binds even when execution or cleanup fails.
    """
    name = f"competition-{competition.id}-{uuid.uuid4().hex}"
    job_token = os.environ.get("AIDE_JOB_TOKEN", "")
    data_path = _enroot_data_path()
    mounts, snapshot_sha = _enroot_mounts(competition, agent, run_dir, logger)
    grading_mounts, wrapper, control_dir = _enroot_grading_mounts(
        competition, agent, run_dir, logger
    )
    grader_name = f"{name}-grader"
    started = time.monotonic()
    container_started = None
    proc = pump = None
    failure = None
    timings = {}
    provisioned = []
    logger.info(purple(f"Run started (enroot): {run_dir}"))
    try:
        try:
            for rootfs_name in (name, grader_name):
                provisioned.append(rootfs_name)
                _provision_enroot_container(rootfs_name, sqsh_path, logger, data_path)
        finally:
            timings["provision_seconds"] = time.monotonic() - started

        env = {
            "PATH": "/opt/conda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
            "HOME": "/tmp",
            **CONSTANTS,
            **_enroot_env(competition, agent, logger),
            # Filled by the host wrapper after this run's readiness handshake.
            "GRADING_SERVER_PORT": None,
            "AIDE_GRADING_SERVER_URL": None,
            "AIDE_VALIDATE_SERVER_URL": None,
            "AIDE_RUN_TOKEN": name,
        }
        for key in (
            "GRADING_SERVER_READY_FILE",
            "GRADING_SERVER_COMMAND_JSON",
            "GRADING_SERVER_FD",
        ):
            env.pop(key, None)
        if job_token:
            env["AIDE_JOB_TOKEN"] = job_token
        if snapshot_sha:
            env["AIDE_SNAPSHOT_SHA"] = snapshot_sha
        nv_mounts, nv_env = _nvidia_manual_passthrough(gpus, logger)
        run_env = _enroot_host_env(data_path)
        # Do not let user home/config bind mounts reintroduce host datasets.
        # System hooks (including GPU setup) still apply.
        config_dir = control_dir / "enroot-config"
        config_dir.mkdir(exist_ok=True)
        run_env.update(
            {
                "ENROOT_MOUNT_HOME": "no",
                "ENROOT_ALLOW_SUPERUSER": "no",
                "ENROOT_CONFIG_PATH": str(config_dir),
                "AIDE_RUN_TOKEN": name,
            }
        )
        if nv_env:
            env["PATH"] = f"{_NVDRV_DIR}:{env['PATH']}"
            if env.get("LD_LIBRARY_PATH"):
                nv_env = {**nv_env, "LD_LIBRARY_PATH": f"{_NVDRV_DIR}:{env['LD_LIBRARY_PATH']}"}
            env.update(nv_env)
            mounts.extend(nv_mounts)
            # Explicit void overrides both inherited and image NVIDIA settings.
            # Unsetting it would allow the image to reactivate the missing hook.
            env["NVIDIA_VISIBLE_DEVICES"] = run_env["NVIDIA_VISIBLE_DEVICES"] = "void"
        else:
            devices = ",".join(_resolve_enroot_gpus(gpus)) or "none"
            run_env["NVIDIA_VISIBLE_DEVICES"] = env["NVIDIA_VISIBLE_DEVICES"] = devices
            run_env["NVIDIA_DRIVER_CAPABILITIES"] = env["NVIDIA_DRIVER_CAPABILITIES"] = (
                "compute,utility"
            )

        grader_env = {
            "PATH": "/opt/conda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
            "HOME": "/tmp",
            "COMPETITION_ID": competition.id,
            "AIDE_RUN_TOKEN": name,
            "NVIDIA_VISIBLE_DEVICES": "void",
            "CUDA_VISIBLE_DEVICES": "",
            "GRADING_SERVER_READY_FILE": "/run/grading/grading_server_ready.json",
            "GRADING_SERVER_FD": None,
        }
        if job_token:
            grader_env["AIDE_JOB_TOKEN"] = job_token
        server_command = _enroot_start_command(
            grader_name,
            grading_mounts,
            grader_env,
            ["/opt/conda/bin/python", "/private/grading_server.py"],
        )
        run_env.update(
            {
                "LOGS_DIR": str(control_dir),
                "GRADING_SERVER_READY_FILE": str(control_dir / "grading_server_ready.json"),
                "GRADING_SERVER_COMMAND_JSON": json.dumps(server_command),
                "GRADING_SERVER_STARTUP_TIMEOUT_SECS": str(
                    env.get("GRADING_SERVER_STARTUP_TIMEOUT_SECS", "120")
                ),
                "GRADING_SERVER_RESTART_DELAY_SECS": str(
                    env.get("GRADING_SERVER_RESTART_DELAY_SECS", "1")
                ),
            }
        )
        start_cmd = [
            sys.executable,
            str(wrapper),
            *_enroot_start_command(name, mounts, env, _agent_cmd(agent)),
        ]
        if cpus:
            start_cmd = ["taskset", "-c", cpus] + start_cmd
            logger.info(purple(f"CPU affinity: taskset -c {cpus}"))

        container_started = time.monotonic()
        timings["setup_seconds"] = container_started - started - timings["provision_seconds"]
        proc = subprocess.Popen(
            start_cmd,
            env=run_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            errors="replace",
            bufsize=1,
            start_new_session=True,
        )

        def _pump_container_log() -> None:
            for line in proc.stdout:
                logger.info(f"[Container] {line.rstrip()}")

        pump = threading.Thread(target=_pump_container_log, daemon=True)
        pump.start()
        time_limit = str(env.get("TIME_LIMIT_SECS", ""))
        budget = int(time_limit) + ENROOT_EXIT_GRACE_SECS if time_limit.isdigit() else None
        watchdog_expired = _wait_for_container(proc, budget, time_limit, logger)
        if watchdog_expired:
            raise TimeoutError(
                f"Enroot container exceeded its {budget}s watchdog deadline "
                f"(last exit code: {proc.returncode}); see {run_dir / 'run.log'}"
            )
        if proc.returncode == 124 and agent.name == "aide" and budget is not None:
            logger.info("AIDE reached its time limit; keeping the saved run outputs.")
        elif proc.returncode != 0:
            # Do not include start_cmd: its environment arguments carry credentials.
            raise RuntimeError(
                f"Agent process exited with code {proc.returncode}; see {run_dir / 'run.log'}"
            )
    except BaseException as exc:
        failure = exc
        raise
    finally:
        cleanup_started = time.monotonic()
        timings["container_seconds"] = (
            cleanup_started - container_started if container_started is not None else 0.0
        )
        cleanup_ok = not isinstance(failure, EnrootCleanupError)
        try:
            if _sweep_run_token_processes(name, logger) is False:
                cleanup_ok = False
        except Exception as exc:
            cleanup_ok = False
            logger.error(f"Container descendant cleanup failed for {name}: {exc}")
        if proc is not None:
            _signal_enroot_pg(proc, signal.SIGKILL)
            if proc.returncode is None:
                try:
                    proc.wait(timeout=ENROOT_COMMAND_STOP_GRACE_SECS)
                except subprocess.TimeoutExpired:
                    cleanup_ok = False
                    logger.error(f"Container {name} survived final SIGKILL")
        if pump is not None:
            pump.join(timeout=10)
            if not pump.is_alive():
                proc.stdout.close()
        if not retain_container:
            for rootfs_name in reversed(provisioned):
                cleanup_ok = _remove_enroot_container(rootfs_name, data_path, logger) and cleanup_ok

        timings["cleanup_seconds"] = time.monotonic() - cleanup_started
        timings["total_seconds"] = time.monotonic() - started
        logger.info(f"[enroot] lifecycle timings: {json.dumps(timings, sort_keys=True)}")
        record = {
            **timings,
            "cleanup_success": cleanup_ok,
            "container_retained": retain_container,
            "returncode": proc.returncode if proc is not None else None,
        }
        temporary = run_dir / ".runtime.json.tmp"
        try:
            temporary.write_text(json.dumps(record, indent=2) + "\n")
            temporary.replace(run_dir / "runtime.json")
        except OSError as exc:
            logger.warning(f"Could not save Enroot runtime timings: {exc}")
        if not cleanup_ok:
            if isinstance(failure, EnrootCleanupError):
                raise failure
            message = f"Enroot cleanup failed for {name}; outputs remain in {run_dir}"
            if failure is not None:
                message += f"; original error: {failure}"
            raise EnrootCleanupError(message) from failure

    logger.info(f"Run completed in {timings['total_seconds']:.2f} seconds including cleanup.")
    return run_dir
