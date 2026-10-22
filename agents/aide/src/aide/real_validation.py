"""Real-execution checks and strict publication of simulated search candidates.

Gate decisions use predicted execution outcomes and confidence, never private
scores. A real result must pass review and strictly improve the incumbent before
its artifacts can be published. Ties preserve the incumbent.
"""

import logging
import os
import time

from .agent import confidence_gate_stages
from .execution_gate import (
    outcome_gate_confidence,
    outcome_gate_enabled,
    outcome_gate_requires_real,
    outcome_gate_threshold,
    random_gate_decision,
    random_gate_enabled,
    random_gate_metadata,
)
from .utils.artifacts import (
    preserve_published_artifacts,
    publish_real_solution,
    reset_submission_directory,
)
from .utils.metric import WorstMetricValue, strictly_improves

# Retry an interpreter failure once while there is still execution budget.
_MAX_REAL_VALIDATION_ATTEMPTS = 2


def _beats(cand, base) -> bool:
    """Keep the published incumbent unless the candidate strictly improves."""
    return strictly_improves(cand, base)


def _simulated_timeout(node):
    """Use the simulator's execution outcome, independently of its posterior."""
    return not node.real_won and node.exc_type == "TimeoutError"


def validate_on_best(
    agent,
    journal,
    real_interpreter,
    workspace_dir,
    *,
    deadline=None,
    reserve_s=180.0,
    force=False,
    periodic=False,
):
    """Execute the latest simulated best and publish if its real score improves.

    force or an execution gate also checks lower-scoring, predicted-buggy,
    or previously failed nodes. The outcome gate uses only exc_type to choose
    its threshold at every stage. Predicted TimeoutError always vetoes real
    execution, including explicit force checks and either confidence gate.
    When the outcome gate is disabled, the legacy stage gate is preserved.
    Best-candidate and explicit force checks can execute high-confidence nodes.
    Random mode replaces only discretionary confidence checks, preserving
    mandatory checks and missing-confidence fallback. Periodic checks pass
    through the configured gate when enabled; otherwise they
    retain the legacy unconditional real-check behavior.
    Leave reserve_s before deadline for review and saving. Return whether a
    real execution result was received, including code failures and timeouts.
    """
    log = logging.getLogger("aide")
    if len(journal) == 0:
        return False
    last = journal.nodes[-1]
    cost_recorder = getattr(agent, "execution_cost_recorder", None)
    if last.real_won:
        if cost_recorder is not None:
            cost_recorder.decision(last, triggers={}, skip_reason="already_real")
        return False
    universal_gate = outcome_gate_enabled(agent.acfg)
    random_gate = random_gate_enabled(agent.acfg)
    gate_fields = {}
    if universal_gate:
        threshold = outcome_gate_threshold(agent.acfg, last.exc_type)
        confidence_check = not random_gate and outcome_gate_requires_real(
            agent.acfg, last.exc_type, last.sim_p_max
        )
        gate_fields = {
            "threshold": threshold,
            "gate_outcome": "success" if last.exc_type is None else "failure",
            "gate_mode": "outcome",
            "gate_confidence": outcome_gate_confidence(agent.acfg, last.exc_type, last.sim_p_max),
            "gate_score_type": "raw_p_max",
        }
        if random_gate:
            gate_fields.update(
                random_gate_metadata(agent.acfg),
                threshold=None,
                gate_mode="random",
                gate_score_type="uniform_random",
            )
    else:
        threshold = getattr(agent.acfg, "real_validation_p_max_threshold", None)
        confidence_check = (
            threshold is not None
            and last.stage_name in confidence_gate_stages(agent.acfg)
            and (last.sim_p_max is None or last.sim_p_max < threshold)
        )
    triggers = {
        "best": None,
        "confidence": bool(confidence_check),
        "periodic_or_forced": bool(force or periodic),
    }
    if _simulated_timeout(last):
        log.info(
            f"[validate-on-best] node {last.id[:8]} simulator predicted "
            "TimeoutError; skipping all real checks"
        )
        if cost_recorder is not None:
            cost_recorder.decision(
                last, triggers=triggers, skip_reason="predicted_timeout", **gate_fields
            )
        return False
    force = force or confidence_check or (periodic and not universal_gate)
    # Unverified real scores and failed interpreter attempts must not block
    # later candidates. Keep simulated scores available for the agent's search.
    ranked = _rank_by_metric(
        [
            n
            for n in journal.good_nodes
            if not n.failure_reasons.get("real_validation")
            and not _simulated_timeout(n)
            and (
                not n.real_won
                or (
                    n.submission_validation == "valid"
                    and n.submission_sha256
                    and not n.pending_submission
                )
            )
        ]
    )
    best = ranked[0] if ranked else None
    triggers["best"] = (
        getattr(agent.acfg, "use_validate_on_best", True)
        and best is not None
        and not last.is_buggy
        and last.id == best.id
    )
    if random_gate:
        triggers["random"] = False
        triggers["missing_confidence"] = gate_fields["gate_confidence"] is None
        force = force or triggers["missing_confidence"]
        if not force and not triggers["best"]:
            if deadline is not None and deadline - reserve_s - time.time() < 1:
                if cost_recorder is not None:
                    cost_recorder.decision(
                        last, triggers=triggers, skip_reason="deadline_reserve", **gate_fields
                    )
                return False
            force, random_fields = random_gate_decision(
                agent.acfg,
                last,
                task=os.getenv("COMPETITION_ID")
                or getattr(agent.cfg, "task_name", "")
                or getattr(agent, "task_desc", ""),
                trial_seed=os.getenv("AIDE_RUN_SEED", "0"),
                method=getattr(agent, "stage_policy", {}).get(last.stage_name, ""),
            )
            triggers["random"] = force
            gate_fields.update(random_fields)
    if not force and not triggers["best"]:
        if cost_recorder is not None:
            cost_recorder.decision(
                last,
                triggers=triggers,
                skip_reason=(
                    "random_skipped"
                    if random_gate
                    else "confidence_accepted" if universal_gate else "not_eligible"
                ),
                **gate_fields,
            )
        return False
    if cost_recorder is not None:
        cost_recorder.decision(last, triggers=triggers, **gate_fields)
    saved_timeout = real_interpreter.timeout
    saved_cost_context = getattr(real_interpreter, "execution_cost_context", {})
    # Compare against the confirmed incumbent before overwriting the sim score.
    prior_real = journal.get_best_node(validated_only=True)
    reason = "exploration check" if force else "became best"
    if random_gate and triggers["random"]:
        reason = (
            f"random gate (q={gate_fields['gate_check_probability']:g}, "
            f"draw={gate_fields['gate_random_draw']:g})"
        )
    elif random_gate and triggers["missing_confidence"]:
        reason = "random gate (confidence unavailable; real required)"
    if confidence_check:
        confidence = "unavailable" if last.sim_p_max is None else f"{last.sim_p_max:g}"
        threshold_label = "always_real" if threshold is None else f"{threshold:g}"
        outcome_label = f", outcome={gate_fields['gate_outcome']}" if universal_gate else ""
        # Offline extractors match the p_max/threshold prefix; stage is appended.
        reason = (
            f"confidence gate (p_max={confidence}, threshold={threshold_label}, "
            f"stage={last.stage_name}{outcome_label})"
        )
    log.info(f"[validate-on-best] node {last.id[:8]} {reason}; running real")
    try:
        for attempt in range(1, _MAX_REAL_VALIDATION_ATTEMPTS + 1):
            remaining = None
            if deadline is not None:
                remaining = deadline - reserve_s - time.time()
                if remaining < 1:
                    log.info("[validate-on-best] no execution budget before reserve; skipping")
                    if cost_recorder is not None:
                        cost_recorder.decision(
                            last,
                            triggers=triggers,
                            skip_reason="deadline_reserve",
                            retry_index=attempt - 1,
                            remaining_execution_s=remaining,
                            **gate_fields,
                        )
                    return False
                real_interpreter.timeout = int(
                    remaining if saved_timeout is None else min(saved_timeout, remaining)
                )
            if cost_recorder is not None:
                real_interpreter.execution_cost_recorder = cost_recorder
                real_interpreter.execution_cost_context = {
                    **cost_recorder.node_fields(last),
                    "triggers": triggers,
                    "retry_index": attempt - 1,
                    "configured_timeout_s": saved_timeout,
                    "remaining_execution_s": remaining,
                    "confidence": last.sim_p_max,
                    "predicted_exception": last.exc_type,
                    "path": "validation",
                    **gate_fields,
                }
            prior_attempt_id = getattr(real_interpreter, "execution_cost_attempt_id", None)
            try:
                # Each attempt must produce its own file.
                reset_submission_directory(workspace_dir)
                with preserve_published_artifacts(workspace_dir):
                    real_res = real_interpreter.run(last.code)
            except Exception as exc:
                # No result was received: this does not establish a code bug.
                failure = f"{type(exc).__name__}: {exc}"
                last.failure_reasons.setdefault("real_validation", []).append(failure)
                log.warning(
                    f"[validate-on-best] node {last.id[:8]} execution attempt "
                    f"{attempt}/{_MAX_REAL_VALIDATION_ATTEMPTS} failed: {failure}"
                )
                try:
                    if (
                        cost_recorder is not None
                        and getattr(real_interpreter, "execution_cost_attempt_id", None)
                        != prior_attempt_id
                    ):
                        cost_recorder.cleanup_attempt(real_interpreter)
                    else:
                        real_interpreter.cleanup_session()
                except Exception as cleanup_exc:
                    log.warning(f"[validate-on-best] interpreter cleanup failed: {cleanup_exc!r}")
                    return False
            else:
                break
        else:
            return False
        last.failure_reasons.pop("real_validation", None)
        real_res.real_won = True
        last.sim_metric = last.metric.value
        try:
            if cost_recorder is None:
                agent.parse_exec_result(last, real_res)
            else:
                cost_recorder.review(last, lambda: agent.parse_exec_result(last, real_res))
        except BaseException as exc:
            # Review mutates the journal node in place. Keep observed execution
            # facts, but discard incomplete evaluation even on interruption.
            last.absorb_exec_result(real_res)
            last.is_buggy = True
            last.metric = WorstMetricValue()
            last.metric_trustworthy = None
            last.csv_exists = None
            last.test_metric = None
            last.leakage_check = None
            last.submission_validation = None
            last.submission_validation_message = None
            last.submission_sha256 = None
            last.submission_actions = []
            last.analysis = f"Real execution review failed: {type(exc).__name__}: {exc}"
            last.failure_reasons = {"review": [last.analysis]}
            if last.exc_type is not None:
                failure = f"Observed execution failed: {last.exc_type}; {last.exc_info or {}}."
                last.failure_reasons["execution"] = [failure]
                last.analysis = f"{failure}\n{last.analysis}"
            raise
        if cost_recorder is not None:
            cost_recorder.node_result(last, event="validation_result")
        beats_confirmed = (
            not last.is_buggy
            and last.submission_validation == "valid"
            and last.metric.value is not None
            and _beats(last.metric, prior_real.metric if prior_real is not None else None)
        )
        if beats_confirmed:
            if publish_real_solution(workspace_dir, last):
                if cost_recorder is not None:
                    cost_recorder.node_result(last, event="publication")
                log.info(
                    f"[validate-on-best] node {last.id[:8]} confirmed real best "
                    f"(metric={last.metric.value:.4f}); published real CSV"
                )
            else:
                log.info(
                    f"[validate-on-best] node {last.id[:8]} confirmed real best "
                    f"but real exec produced no CSV"
                )
        else:
            log.info(
                f"[validate-on-best] node {last.id[:8]} did not beat confirmed "
                f"real best "
                f"{(prior_real.id[:8] if prior_real is not None else None)}; "
                f"keeping prior published CSV"
            )
    except Exception as e:
        log.warning(f"[validate-on-best] crashed: {e!r}")
    finally:
        real_interpreter.timeout = saved_timeout
        if cost_recorder is not None:
            real_interpreter.execution_cost_context = saved_cost_context
    return True


def _rank_by_metric(nodes):
    """Rank within the majority metric direction, matching Journal.get_best_node."""
    from collections import defaultdict

    nodes = [n for n in nodes if n.metric is not None]
    if not nodes:
        return []
    by_dir: dict = defaultdict(list)
    for n in nodes:
        d = n.metric.maximize if n.metric.value is not None else None
        by_dir[d].append(n)
    best_dir = max(by_dir, key=lambda d: len(by_dir[d]))
    return sorted(by_dir[best_dir], key=lambda n: n.metric, reverse=True)
