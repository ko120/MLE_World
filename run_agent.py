import argparse
import asyncio
import concurrent.futures
import yaml
import json
import logging
import os
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import docker

from agents.registry import Agent
from agents.registry import Registry
from agents.run import EnrootCleanupError, run_in_container, run_in_enroot, snapshot_agent_code
from agents.seeding import seeded_agent
from environment.defaults import DEFAULT_CONTAINER_CONFIG_PATH
from mlebench.data import is_dataset_prepared
from mlebench.registry import Competition, registry
from mlebench.utils import create_run_dir, get_logger, get_runs_dir, get_timestamp

logger = get_logger(__name__)


@dataclass(frozen=True)
class Task:
    run_id: str
    seed: int
    image: str
    path_to_run_group: Path
    path_to_run: Path
    agent: Agent
    competition: Competition
    container_config: dict[str, Any]


def _cpu_slices(n_workers: int, logger: logging.Logger) -> list[str] | None:
    """Per-worker taskset core lists, or None to leave affinity untouched.

    Restores the per-container CPU cap the docker path gets from
    container_configs/default.json (nano_cpus=8e9 -> 8 CPUs) and that
    run_in_enroot structurally cannot apply: enroot is not a cgroup manager, so
    each agent container otherwise sees every core in the job's SLURM cgroup
    (144 on a B200 node). OpenMP/MKL/OpenBLAS/LightGBM size their pools from
    that, generated code uses n_jobs=-1, and real-only BoN runs best_of_n_size
    candidates in parallel -> ~10k runnable threads on 144 cores (measured
    CPULoad 157-375 on 2026-07-30, exec times inflated 5-10x).

    Gated on AIDE_CPU_AFFINITY, which the enroot launchers now set to 1 by
    DEFAULT (2026-08-16; previously only in SHARED_NODE/scattered mode). That
    flip is a campaign boundary: runs before it let every container see every
    core, so exec times are not comparable across it.

    Width comes from one of two knobs:
      AIDE_CPU_PER_TASK   cores per agent, stated directly (launcher default 10)
      AIDE_CPU_OVERCOMMIT factor on the fair share ncpu/n_workers (default 1.0)
    Both produce sliding overlapping windows. Strictly disjoint slices (1.0) are
    not obviously right: agents spend much of the run blocked on LLM generation
    (measured exec duty median 0.49 on 2026-07-30), so at 1.0 about half the
    node's cores sit idle inside the slices of waiting agents. A width near
    fair_share/duty lets an executing agent use roughly what work-conserving
    scheduling gives it today while still bounding the worst case, since the
    expected number of simultaneously executing agents is n_workers * duty.

    Prefer AIDE_CPU_PER_TASK when the intent is "each agent gets N cores": the
    equivalent factor is per_task * n_workers / ncpu, which has to be recomputed
    every time the worker count or the allocation changes (63 workers on 75
    cores need 8.4x for 10 cores each; 27 workers on the same cores need 3.6x).
    """
    if os.environ.get("AIDE_CPU_AFFINITY", "").strip().lower() not in ("1", "true", "auto"):
        return None
    try:
        available = sorted(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        logger.warning("[cpu-affinity] sched_getaffinity unavailable; not pinning")
        return None
    if n_workers <= 0:
        return None
    ncpu = len(available)

    def _num(var: str, default: float) -> float:
        try:
            return float(os.environ.get(var, "") or default)
        except ValueError:
            logger.warning(f"[cpu-affinity] ignoring unparseable {var}={os.environ.get(var)!r}")
            return default

    per_task = int(_num("AIDE_CPU_PER_TASK", 0))
    overcommit = max(1.0, _num("AIDE_CPU_OVERCOMMIT", 1.0))
    if per_task > 0:
        # Explicit width wins over the factor. Derive the factor from it for the
        # log line only — never round-trip the width through the float, or
        # 75 * (10*63/75) / 63 lands on 9.999... and int() silently gives 9.
        width = min(ncpu, per_task)
        overcommit = width * n_workers / ncpu
    elif overcommit > 1.0:
        width = min(ncpu, max(1, int(ncpu * overcommit / n_workers)))
    else:
        width = 0  # exact-partition mode

    if width == 0:
        # Only the exact partition needs a core per worker; a sliding window is
        # >=1 core wide by construction, so it still works on an allocation
        # narrower than the worker count (63 agents over 40 cores at 10 each).
        if ncpu < n_workers:
            logger.warning(
                f"[cpu-affinity] {ncpu} cores for {n_workers} workers "
                "(<1 core each); not pinning"
            )
            return None
        # Exact partition: contiguous chunks (NUMA/cache locality) with the
        # remainder spread one core at a time so no core is left unassigned.
        per, extra = divmod(ncpu, n_workers)
        slices, start = [], 0
        for i in range(n_workers):
            n = per + (1 if i < extra else 0)
            slices.append(",".join(str(c) for c in available[start:start + n]))
            start += n
        width_desc = f"{per}{'-' + str(per + 1) if extra else ''}"
    else:
        # Sliding overlapping windows: same start spacing, wider span, wrapping
        # at the end so every worker gets the full width (computed above).
        slices = []
        for i in range(n_workers):
            s = i * ncpu // n_workers
            picked = sorted({available[(s + k) % ncpu] for k in range(width)})
            slices.append(",".join(str(c) for c in picked))
        width_desc = str(width)
    logger.info(
        f"[cpu-affinity] pinning {n_workers} workers over {ncpu} cores "
        f"({width_desc} each, overcommit {overcommit:g}x)"
    )
    return slices


def _write_run_group_metadata(directory: Path, metadata: dict[str, Any]) -> None:
    """Atomically checkpoint completed tasks so a later stalled task loses no records."""
    directory.mkdir(parents=True, exist_ok=True)
    temporary = directory / ".metadata.json.tmp"
    try:
        temporary.write_text(json.dumps(metadata, indent=4, default=str) + "\n")
        temporary.replace(directory / "metadata.json")
    finally:
        temporary.unlink(missing_ok=True)


async def worker(
    idx: int,
    queue: asyncio.Queue[Task],
    client: docker.DockerClient,
    tasks_outputs: dict[str, dict[str, Any]],
    cpu_slices: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
    cleanup_failed: asyncio.Event | None = None,
) -> None:
    if cleanup_failed is None:
        cleanup_failed = asyncio.Event()
    while True:
        task = await queue.get()

        run_logger = logger
        log_file_handler = None

        task_output = {
            "seed": task.seed,
            "competition": task.competition.id,
            "metric_name": task.competition.grader.name,
            "metric_lower_is_better": task.competition.grader.lower_is_better,
        }
        try:
            # Create logger for the run
            run_logger = get_logger(str(task.path_to_run))
            log_file_handler = logging.FileHandler(task.path_to_run / "run.log")
            log_file_handler.setFormatter(
                logging.getLogger().handlers[0].formatter
            )  # match the formatting we have
            run_logger.addHandler(log_file_handler)
            run_logger.propagate = False

            if cleanup_failed.is_set():
                task_output.update({
                    "success": False, "skipped": True,
                    "error": "Not started because Enroot cleanup failed in this run group",
                })
                run_logger.warning(task_output["error"])
                continue

            run_logger.info(
                f"[Worker {idx}] Running seed {task.seed} for {task.competition.id} and agent {task.agent.name}"
            )

            if args.runtime == "enroot":
                await asyncio.to_thread(
                    run_in_enroot,
                    competition=task.competition,
                    agent=task.agent,
                    sqsh_path=args.enroot_image,
                    retain_container=args.retain,
                    run_dir=task.path_to_run,
                    logger=run_logger,
                    gpus=[args.gpu_device[idx % len(args.gpu_device)]],
                    cpus=cpu_slices[idx] if cpu_slices else None,
                )
            else:
                await asyncio.to_thread(
                    run_in_container,
                    client=client,
                    competition=task.competition,
                    agent=task.agent,
                    image=task.agent.name,
                    container_config=task.container_config,
                    retain_container=args.retain,
                    run_dir=task.path_to_run,
                    logger=run_logger,
                    gpus=[args.gpu_device[idx % len(args.gpu_device)]],
                )
            task_output["success"] = True

            run_logger.info(
                f"[Worker {idx}] Finished running seed {task.seed} for {task.competition.id} and agent {task.agent.name}"
            )
        except Exception as e:
            if isinstance(e, EnrootCleanupError):
                cleanup_failed.set()
                task_output["cleanup_success"] = False
                if metadata is not None:
                    metadata["status"] = "stopping"
                    metadata.setdefault("abort_reason", f"Cleanup failed for {task.run_id}: {e}")
            stack_trace = traceback.format_exc()
            run_logger.error(type(e))
            run_logger.error(stack_trace)
            run_logger.error(
                f"Run failed for seed {task.seed}, agent {task.agent.id} and competition "
                f"{task.competition.id}"
            )
            task_output["success"] = False
            task_output["error"] = str(e)
        finally:
            task_output.setdefault("success", False)
            tasks_outputs[task.run_id] = task_output
            try:
                if metadata is not None:
                    _write_run_group_metadata(task.path_to_run_group, metadata)
            except OSError as exc:
                run_logger.error(f"Could not checkpoint run metadata: {exc}")
            finally:
                queue.task_done()
                if log_file_handler is not None:
                    run_logger.removeHandler(log_file_handler)
                    log_file_handler.close()


def apply_overrides(target: dict, overrides: list[str], what: str) -> None:
    """Apply launch-time `key=value` overrides on top of a profile dict
    (kwargs or env_vars). Keeps model/time/step axes orthogonal to the method
    profiles: any profile can be re-pointed at launch instead of being cloned
    per variant in the yaml."""
    for override in overrides:
        key, sep, value = override.partition("=")
        if not sep or not key:
            raise ValueError(
                f"--{what} entries must be key=value, got {override!r}"
            )
        logger.info(
            f"{what} override: {key}={value} "
            f"(profile had: {target.get(key, '<unset>')})"
        )
        target[key] = value


async def main(args) -> int:
    # asyncio.to_thread routes through the event loop's default
    # ThreadPoolExecutor, whose cap is min(32, cpu+4) = 32 on a many-core node.
    # That would silently throttle N_WORKERS > 32 to 32 concurrent containers
    # (so 54 workers -> 2 waves). Size the executor to n_workers so the requested
    # parallelism is real. Applies to both the docker and enroot paths.
    asyncio.get_running_loop().set_default_executor(
        concurrent.futures.ThreadPoolExecutor(max_workers=args.n_workers)
    )
    client = docker.from_env() if args.runtime == "docker" else None
    global registry
    registry = registry.set_data_dir(Path(args.data_dir))
    agent_registry = Registry(args.agent_id, args.agent_name, args.agent_dir)

    # If agent config is provided, use it to create the agent
    # otherwise search over the config file in the agent directory
    if args.agent_config:
        with open(args.agent_config, "r") as f:
            agent = agent_registry.get_agent_from_config(yaml.safe_load(f))
    else:
        agent = agent_registry.get_agent()

    if args.extra_kwargs:
        apply_overrides(agent.kwargs, args.extra_kwargs, "extra-kwargs")
    if args.extra_env:
        apply_overrides(agent.env_vars, args.extra_env, "extra-env")

    if agent.privileged and not (
        os.environ.get("I_ACCEPT_RUNNING_PRIVILEGED_CONTAINERS", "False").lower()
        in ("true", "1", "t")
    ):
        raise ValueError(
            "Agent requires running in a privileged container, but the environment variable `I_ACCEPT_RUNNING_PRIVILEGED_CONTAINERS` is not set to `True`! "
            "Carefully consider if you wish to run this agent before continuing. See agents/README.md for more details."
        )

    suffix = os.environ.get("RUN_GROUP_SUFFIX", "").strip()
    if suffix:
        run_group = f"{get_timestamp()}_{agent.name}_{suffix}"
    else:
        run_group = f"{get_timestamp()}_run-group_{agent.name}"

    # Load competition ids and check all are prepared
    with open(args.competition_set, "r") as f:
        competition_ids = [
            line.strip() for line in f.read().splitlines() if line.strip()
        ]
    for competition_id in competition_ids:
        competition = registry.get_competition(competition_id)
        if not is_dataset_prepared(competition):
            raise ValueError(
                f"Dataset for competition `{competition.id}` is not prepared! "
                f"Please run `mlebench prepare -c {competition.id}` to prepare the dataset."
            )

    with open(args.container_config, "r") as f:
        container_config = json.load(f)

    # Create tasks for each (competition * seed)
    logger.info(f"Launching run group: {run_group}")
    tasks = []
    seed_offset = getattr(args, "seed_offset", 0)
    for seed in range(seed_offset, seed_offset + args.n_seeds):
        for competition_id in competition_ids:
            competition = registry.get_competition(competition_id)
            run_dir = create_run_dir(competition.id, agent.id, run_group)
            run_id = run_dir.stem
            task = Task(
                run_id=run_id,
                seed=seed,
                image=agent.name,
                agent=seeded_agent(agent, seed),
                competition=competition,
                path_to_run_group=run_dir.parent,
                path_to_run=run_dir,
                container_config=container_config,
            )
            (run_dir / "task.json").write_text(json.dumps({
                "competition": competition.id, "seed": seed, "agent_id": agent.id,
                "runtime_seed": seed if agent.name == "aide" else None,
                "metric_name": competition.grader.name,
                "metric_lower_is_better": competition.grader.lower_is_better,
            }, indent=2) + "\n")
            tasks.append(task)

    # Freeze the agent code for this run group BEFORE any worker launches:
    # enroot containers bind src/aide + start.sh from this snapshot
    # (agents/run.py:run_in_enroot), pinning the whole group to launch-time
    # code — host edits during the run can't reach in-flight containers.
    if agent.name == "aide":
        snapshot_agent_code(agent.start.parent, get_runs_dir() / run_group, logger)

    logger.info(f"Creating {args.n_workers} workers to serve {len(tasks)} tasks...")

    # Create queue of tasks, and assign workers to run them
    queue = asyncio.Queue()
    for task in tasks:
        queue.put_nowait(task)
    workers = []
    tasks_outputs = {}
    run_group_dir = get_runs_dir() / run_group
    metadata = {
        "run_group": run_group,
        "created_at": get_timestamp(),
        "status": "running",
        "expected_runs": len(tasks),
        "runs": tasks_outputs,
    }
    _write_run_group_metadata(run_group_dir, metadata)
    cpu_slices = _cpu_slices(args.n_workers, logger) if args.runtime == "enroot" else None
    cleanup_failed = asyncio.Event()
    for idx in range(args.n_workers):
        w = asyncio.create_task(worker(
            idx, queue, client, tasks_outputs, cpu_slices, metadata, cleanup_failed,
        ))
        workers.append(w)

    # Wait for all tasks to be completed and collect results
    started_at = time.monotonic()
    await queue.join()
    time_taken = time.monotonic() - started_at

    for w in workers:
        w.cancel()  # Cancel all workers now that the queue is empty

    await asyncio.gather(*workers, return_exceptions=True)

    metadata["status"] = "aborted" if cleanup_failed.is_set() else "complete"
    _write_run_group_metadata(run_group_dir, metadata)
    logger.info(f"{args.n_workers} workers ran for {time_taken:.2f} seconds in total")
    failed = sum(not output["success"] for output in tasks_outputs.values())
    if failed:
        logger.error(f"{failed}/{len(tasks_outputs)} runs failed or were skipped; see {run_group_dir / 'metadata.json'}")
    return 1 if failed else 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run an agent on a set of competitions in a Docker container."
    )
    parser.add_argument(
        "--agent-id",
        help="Agent ID of the agent to run.",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--agent-name",
        help="Agent name of the agent to run.",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--agent-dir",
        help="Path to the directory containing the agent configurations.",
        type=str,
    )
    parser.add_argument(
        "--agent-config",
        help="Path to a YAML file with the agent configuration.",
        type=str,
    )
    parser.add_argument(
        "--competition-set",
        type=str,
        required=True,
        help="Path to a text file with a single competition ID on each line",
    )
    parser.add_argument(
        "--n-workers",
        type=int,
        required=False,
        default=1,
        help="Number of workers to run in parallel",
    )
    parser.add_argument(
        "--n-seeds",
        type=int,
        required=False,
        default=1,
        help="Number of seeds to run for each competition",
    )
    parser.add_argument(
        "--seed-offset", type=int, default=0,
        help="First runtime seed; permits an exact repetition retry without relabeling it.",
    )
    parser.add_argument(
        "--container-config",
        help="Path to a JSON file with an environment configuration; these args will be passed to `docker.from_env().containers.create`",
        type=str,
        required=False,
        default=DEFAULT_CONTAINER_CONFIG_PATH,
    )
    parser.add_argument(
        "--retain",
        help="Whether to retain the container after the run instead of removing it.",
        action="store_true",
        required=False,
        default=False,
    )
    parser.add_argument(
        "--run-dir",
        help="Path to the directory where all assets associated with the run are stored.",
        type=str,
        required=False,
        default=None,
    )
    parser.add_argument(
        "--data-dir",
        help="Path to the directory containing the competition data.",
        type=str,
        required=False,
        default=registry.get_data_dir(),
    )
    parser.add_argument(
        "--gpu-device",
        help="List of GPU devices to pass to the container.",
        type=str,
        required=False,
        nargs="+",
        default=[0],
    )
    parser.add_argument(
        "--runtime",
        help="Container runtime: 'docker' (default) or 'enroot' for single-node SLURM.",
        type=str,
        choices=["docker", "enroot"],
        default="docker",
    )
    parser.add_argument(
        "--enroot-image",
        help="Path to the enroot .sqsh image (used when --runtime enroot).",
        type=str,
        required=False,
        default=None,
    )
    parser.add_argument(
        "--extra-kwargs",
        help="Launch-time kwarg overrides applied on top of the profile's kwargs "
        "(space-separated key=value pairs, e.g. "
        "agent.code.model=google/gemini-3.1-pro-preview agent.time_limit=3600). "
        "Lets one method profile serve any model/budget without cloning it in "
        "the yaml.",
        type=str,
        required=False,
        nargs="*",
        default=[],
    )
    parser.add_argument(
        "--extra-env",
        help="Launch-time env_var overrides applied on top of the profile's "
        "env_vars (space-separated key=value pairs, e.g. TIME_LIMIT_SECS=3600 "
        "STEP_LIMIT=500). Same mechanism as --extra-kwargs, for the env "
        "channel (start.sh timeout, prompt notes, container watchdog).",
        type=str,
        required=False,
        nargs="*",
        default=[],
    )
    args = parser.parse_args()
    logger = get_logger(__name__)
    logger.info(f"Args: {args}")

    raise SystemExit(asyncio.run(main(args)))
