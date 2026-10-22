"""Configure execution, run the search loop, and save/clean up its artifacts.

Real-check gating and incumbent publication are implemented in real_validation;
the console entrypoint remains aide.run:run.
"""

import json
import logging
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

from omegaconf import OmegaConf
from rich.status import Status

from .agent import Agent
from .execution_gate import (
    outcome_gate_enabled,
)
from .interpreter import Interpreter
from .journal import Journal, Node
from .real_validation import _beats as _beats
from .real_validation import _rank_by_metric as _rank_by_metric
from .real_validation import _simulated_timeout as _simulated_timeout
from .real_validation import validate_on_best
from .utils.config import load_cfg, load_task_desc, prep_agent_workspace, save_run
from .utils.execution_cost import ExecutionCostRecorder
from .utils.seeding import seed_runtime
from .utils.submission_recovery import retry_pending_submissions

# Stop persistent failures while preserving the journal and cleaning up.
_MAX_CONSECUTIVE_STEP_FAILURES = 5


class VerboseFilter(logging.Filter):
    """Exclude verbose records from normal logs."""

    def filter(self, record):
        return not getattr(record, "verbose", False)


def journal_to_string_tree(journal: Journal) -> str:
    best_node = journal.get_best_node()
    tree_str = "Solution tree\n"

    def append_rec(node: Node, level: int):
        nonlocal tree_str
        indent = "  " * level
        if node.is_buggy:
            s = f"{indent}◍ bug (ID: {node.id})\n"
        else:
            marker = " (best)" if node is best_node else ""
            s = f"{indent}● {node.metric.value:.3f}{marker} (ID: {node.id})\n"
        tree_str += s
        for child in node.children:
            append_rec(child, level + 1)

    for n in journal.draft_nodes:
        append_rec(n, 0)

    return tree_str


def run():
    """Run the configured search within its wall-clock budget and clean up.

    Real validation shares that budget and leaves a reserve for saving artifacts
    and stopping interpreters, including when repeated steps fail.
    """
    # Include workspace preparation in the total time budget.
    run_start = time.time()
    cfg = load_cfg()
    steps_count_mode = getattr(cfg.agent, "steps_count_mode", "journal")
    if steps_count_mode not in {"journal", "completed"}:
        raise ValueError("agent.steps_count_mode must be journal or completed")
    shutdown_reserve_s = max(0.0, float(os.getenv("AIDE_SHUTDOWN_RESERVE_S", "180")))
    deadline = run_start + cfg.agent.time_limit
    log_format = "[%(asctime)s] %(levelname)s: %(message)s"
    logging.basicConfig(
        level=getattr(logging, cfg.log_level.upper()), format=log_format, handlers=[]
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)

    logger = logging.getLogger("aide")
    cfg.log_dir.mkdir(parents=True, exist_ok=True)

    # Keep verbose records in their own file; show normal records on stdout.
    file_handler = logging.FileHandler(cfg.log_dir / "aide.log")
    file_handler.setFormatter(logging.Formatter(log_format))
    file_handler.addFilter(VerboseFilter())

    verbose_file_handler = logging.FileHandler(cfg.log_dir / "aide.verbose.log")
    verbose_file_handler.setFormatter(logging.Formatter(log_format))

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter(log_format))
    console_handler.addFilter(VerboseFilter())

    logger.addHandler(file_handler)
    logger.addHandler(verbose_file_handler)
    logger.addHandler(console_handler)

    logger.info(f'Starting run "{cfg.exp_name}"')
    runtime_seed = seed_runtime(os.environ.get("AIDE_RUN_SEED"))
    if runtime_seed is not None:
        logger.info(f"[runtime-seed] Python and NumPy seed={runtime_seed}")

    task_desc = load_task_desc(cfg)

    with Status("Preparing agent workspace (copying and extracting files) ..."):
        prep_agent_workspace(cfg)

    journal = Journal()

    exec_cfg = OmegaConf.to_container(cfg.exec)  # type: ignore
    use_simulator = bool(exec_cfg.pop("simulator", False))
    sim_model = exec_cfg.pop("simulator_model", None)
    sim_deadline = float(exec_cfg.pop("simulator_deadline_s", 30.0))
    sim_start_delay = float(exec_cfg.pop("simulator_start_delay_s", 3.0))
    sim_only = bool(exec_cfg.pop("simulator_only", False))
    if sim_only and not use_simulator:
        raise ValueError(
            "exec.simulator_only=True requires exec.simulator=True — there is "
            "no simulator to run alone"
        )
    real_interpreter = Interpreter(cfg.workspace_dir, **exec_cfg)
    sim_predict = None
    race_predict = None
    sim_p_clean = None
    sim_metric_posterior = None
    if use_simulator:
        from .parallel_interpreter import ParallelInterpreter
        from .simulator import SimulatorInterpreter

        sim = SimulatorInterpreter(
            cfg.workspace_dir,
            model=sim_model,
            timeout=exec_cfg.get("timeout", 3600),
        )
        interpreter = ParallelInterpreter(
            real_interpreter,
            sim,
            deadline_s=sim_deadline,
            start_delay_s=sim_start_delay,
            sim_only=sim_only,
        )

        def sim_predict(code):
            push_data_overview()
            return sim.run(code, reset_session=True)

        def sim_p_clean(code):
            push_data_overview()
            return sim.exc_scores(code)

        def sim_metric_posterior(code, *, config, deadline):
            push_data_overview()
            return sim.metric_posterior(code, config=config, deadline=deadline)

        # Concurrent candidates need isolated working and submission directories.
        race_root = cfg.workspace_dir / "bonrace"

        def race_predict(code):
            push_data_overview()
            race_root.mkdir(exist_ok=True)
            cand_ws = Path(tempfile.mkdtemp(dir=race_root))
            (cand_ws / "submission").mkdir()
            (cand_ws / "working").mkdir()
            canonical_input = cfg.workspace_dir / "input"
            if canonical_input.exists():
                (cand_ws / "input").symlink_to(canonical_input)
            real = Interpreter(cand_ws, **exec_cfg)
            recorder = getattr(agent, "execution_cost_recorder", None)
            if recorder is not None:
                stage = getattr(agent, "_execution_cost_stage", None)
                real.execution_cost_recorder = recorder
                real.execution_cost_context = {
                    "stage": stage,
                    "policy": recorder.metadata.get("stage_policies", {}).get(stage),
                    "selection_step": agent.current_step,
                    "triggers": {"race": True},
                    "retry_index": 0,
                    "path": "candidate_race",
                    "configured_timeout_s": exec_cfg.get("timeout"),
                }
            pi = ParallelInterpreter(
                real,
                sim,
                deadline_s=sim_deadline,
                start_delay_s=sim_start_delay,
            )
            try:
                return pi.run(code, reset_session=True)
            finally:
                pi.close()
                real.cleanup_session()
                shutil.rmtree(cand_ws, ignore_errors=True)

        if sim_only:
            # Candidate selection falls back to sim_predict without a race oracle.
            if cfg.agent.bon_oracle == "race":
                logger.warning(
                    "[parallel] exec.simulator_only=True overrides "
                    "agent.bon_oracle=race; candidate verdicts come from the "
                    "simulator alone"
                )
            race_predict = None
        logger.info(
            f"ParallelInterpreter enabled: model={sim_model} "
            f"sim_only={sim_only} "
            f"deadline_s={sim_deadline} start_delay_s={sim_start_delay}"
        )
    else:
        interpreter = real_interpreter

    agent = Agent(
        task_desc=task_desc,
        cfg=cfg,
        journal=journal,
        sim_predict=sim_predict,
        race_predict=race_predict,
        sim_p_clean=sim_p_clean,
        sim_metric_posterior=sim_metric_posterior,
        deadline=deadline,
    )
    cost_recorder = (
        ExecutionCostRecorder.from_config(cfg, run_start=run_start)
        if getattr(cfg.agent, "record_execution_cost", False)
        else None
    )
    agent.execution_cost_recorder = cost_recorder
    real_interpreter.execution_cost_recorder = cost_recorder

    global_step = len(journal)
    data_overview_pushed = False

    def push_data_overview():
        # Candidate prediction can precede exec_callback on the first step.
        nonlocal data_overview_pushed
        if use_simulator and not data_overview_pushed:
            sim.set_data_overview(agent.data_preview)
            data_overview_pushed = True

    def exec_callback(*args, **kwargs):
        push_data_overview()
        executable_wall = deadline - shutdown_reserve_s - time.time()
        if executable_wall <= 0:
            if cost_recorder is not None:
                cost_recorder.decision(
                    agent._execution_cost_node,
                    triggers={"baseline": not use_simulator},
                    skip_reason="deadline_reserve",
                    remaining_execution_s=executable_wall,
                )
            raise TimeoutError("clean-shutdown reserve reached before execution")
        previous_timeout = real_interpreter.timeout
        effective_timeout = (
            executable_wall
            if previous_timeout is None
            else min(float(previous_timeout), executable_wall)
        )
        real_interpreter.timeout = max(1, int(effective_timeout))
        if cost_recorder is not None:
            real_interpreter.execution_cost_context = {
                **cost_recorder.node_fields(agent._execution_cost_node),
                "triggers": {
                    "baseline": not use_simulator,
                    "simulator_fallback": use_simulator and sim_only,
                    "race": use_simulator and not sim_only,
                },
                "retry_index": 0,
                "configured_timeout_s": previous_timeout,
                "remaining_execution_s": executable_wall,
                "path": "selected_node",
            }
        try:
            return interpreter.run(*args, **kwargs)
        finally:
            real_interpreter.timeout = previous_timeout

    fatal: BaseException | None = None
    step_failures = 0
    completed_steps = 0
    last_real_check_step = len(journal)
    try:
        while (
            completed_steps if steps_count_mode == "completed" else global_step
        ) < cfg.agent.steps:
            retry_pending_submissions(agent)
            remaining = deadline - time.time()
            if remaining <= shutdown_reserve_s:
                logger.info(
                    f"[shutdown] clean-teardown reserve reached "
                    f"({remaining:.1f}s <= {shutdown_reserve_s:.1f}s); "
                    "stopping search"
                )
                break
            # Retry transient failures; stop when consecutive failures persist.
            step_started = time.perf_counter()
            nodes_before_step = len(journal)
            step_error = None
            try:
                agent.step(exec_callback=exec_callback)
                # Validate simulated winners and periodic/low-confidence selections.
                if use_simulator and (
                    cfg.agent.use_validate_on_best or outcome_gate_enabled(cfg.agent)
                ):
                    interval = cfg.agent.real_validation_interval
                    periodic = interval > 0 and len(journal) - last_real_check_step >= interval
                    ran = validate_on_best(
                        agent,
                        journal,
                        real_interpreter,
                        cfg.workspace_dir,
                        deadline=deadline,
                        reserve_s=shutdown_reserve_s,
                        periodic=periodic,
                    )
                    if ran or journal.nodes[-1].real_won:
                        last_real_check_step = len(journal)
                if steps_count_mode == "completed" and len(journal) <= nodes_before_step:
                    raise RuntimeError("Outer step returned without committing a journal node")
                if (
                    completed_steps if steps_count_mode == "completed" else global_step
                ) == cfg.agent.steps - 1:
                    logger.info(journal_to_string_tree(journal))
                save_run(cfg, journal)
                if len(journal) > nodes_before_step:
                    completed_steps += 1
                step_failures = 0
            except Exception as exc:
                step_error = type(exc).__name__
                step_failures += 1
                logger.exception(
                    f"[step-guard] step failed ({step_failures}/"
                    f"{_MAX_CONSECUTIVE_STEP_FAILURES} consecutive); dropping step"
                )
                if step_failures >= _MAX_CONSECUTIVE_STEP_FAILURES:
                    fatal = exc
                    logger.error(
                        "[step-guard] too many consecutive step failures; stopping search early"
                    )
                    break
            finally:
                if cost_recorder is not None:
                    cost_recorder.emit(
                        "step_end",
                        elapsed_s=time.perf_counter() - step_started,
                        journal_nodes=len(journal),
                        completed_steps=completed_steps,
                        exception_type=step_error
                        or (sys.exc_info()[0].__name__ if sys.exc_info()[0] else None),
                    )
            global_step = len(journal)
    except BaseException as e:
        # Save and clean up even after interrupts or unexpected loop failures.
        fatal = e
        logger.exception("[run-guard] search loop aborted by unexpected error")

    if steps_count_mode == "completed" and completed_steps < cfg.agent.steps and fatal is None:
        fatal = TimeoutError(
            f"Completed-step target not reached: {completed_steps}/{cfg.agent.steps}"
        )
        logger.error(str(fatal))

    retry_pending_submissions(agent)
    try:
        save_run(cfg, journal)
    except Exception:
        logger.exception("[run-guard] final save_run failed")

    try:
        if (
            cost_recorder is not None
            and (not use_simulator or sim_only)
            and getattr(real_interpreter, "execution_cost_attempt_id", None) is not None
        ):
            cost_recorder.cleanup_attempt(real_interpreter)
        else:
            interpreter.cleanup_session()
    except Exception:
        logger.exception("[run-guard] interpreter cleanup failed")

    if cost_recorder is not None:
        cost_recorder.emit(
            "run_end",
            journal_nodes=len(journal),
            completed_steps=completed_steps,
            steps_count_mode=steps_count_mode,
            step_target=cfg.agent.steps,
            step_target_met=(
                (completed_steps >= cfg.agent.steps) if steps_count_mode == "completed" else None
            ),
            recorder_write_s=cost_recorder.write_seconds,
            status="failed" if fatal is not None else "completed",
        )

    if global_step == 0:
        shutil.rmtree(cfg.workspace_dir, ignore_errors=True)

    # Avoid hanging at exit while Python joins leaked children or race threads.
    logging.shutdown()
    os._exit(0 if fatal is None else 1)


if __name__ == "__main__":
    run()
