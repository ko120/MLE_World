"""Search orchestration for the MLE-World AIDE fork.

Read Agent.step first: choose a parent, generate/select a candidate, execute or
reuse its evidence, review the result, and update the journal. Prompt rendering,
advanced candidate selection, and submission review live in the corresponding
base modules; Agent owns their shared state and budget-aware model callbacks.

The standalone review/selection helpers remain here because offline evaluation
tools extract their source without importing the agent's runtime dependencies.
"""

import json
import logging
import os
import random
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, cast

from . import preference, ucb
from .backend import FunctionSpec, query
from .candidate_selection import STAGES, CandidateSelectionMixin
from .execution_gate import (
    outcome_gate_enabled,
    outcome_gate_threshold,
    random_gate_enabled,
    validate_execution_gate,
)
from .fusion import (
    FusionPromptConfig,
)
from .generation_prompts import GenerationPromptsMixin, _prompt_opening
from .generation_prompts import format_time as format_time
from .interpreter import ExecutionResult, Interpreter
from .journal import Journal, Node
from .simulator import (
    is_bad_prediction,
    is_predicted_candidate_usable,
    is_predicted_crash,
    is_sim_internal_failure,
)
from .submission_review import SubmissionReviewMixin
from .submission_review import _existing_submission_digest as _existing_submission_digest
from .utils import data_preview
from .utils.artifacts import has_submission_csv as _has_submission_csv
from .utils.artifacts import publish_real_solution, reset_submission_directory
from .utils.config import Config
from .utils.metric import MetricValue, WorstMetricValue
from .utils.response import (
    extract_code,
    extract_text_up_to_code,
    trim_long_string,
    wrap_code,
)
from .utils.submission_recovery import retain_pending_submission

logger = logging.getLogger("aide")


def _draft_max_tokens() -> int | None:
    """Return a positive AIDE_DRAFT_MAX_TOKENS override, or None.

    Drafts otherwise use the global sampling cap. Vertex/Gemini ignores caps."""
    raw = os.environ.get("AIDE_DRAFT_MAX_TOKENS", "").strip()
    if not raw:
        return None
    try:
        v = int(raw)
    except ValueError:
        logger.warning(f"AIDE_DRAFT_MAX_TOKENS is not an int, ignoring: {raw!r}")
        return None
    return v if v > 0 else None


def _split_plan_code(completion_text: str) -> tuple[str, str, bool]:
    """Parse a completion into (plan, code, parsed_ok); the plan is optional.

    On failure, retain best-effort code so execution reports its syntax error."""
    code = extract_code(completion_text)
    nl_text = extract_text_up_to_code(completion_text)
    if code:  # plan optional; see plan_and_code_query (avoids ```fence leak)
        return nl_text, code, True
    return (
        "",
        extract_code(completion_text, fallback_to_invalid=True) or completion_text,
        False,
    )


def _exploration_prompt(prompt: dict, axis: str | None) -> dict:
    """Append an improvement directive while preserving the shared prompt prefix."""
    if axis is None:
        return prompt
    return {
        **prompt,
        "Exploration directive": (
            "Sibling attempts are already improving this same solution "
            "along other axes. For THIS attempt, focus your improvement "
            f"on: {axis}."
        ),
    }


ExecCallbackType = Callable[[str, bool], ExecutionResult]

# Policies vary candidate generation/selection independently of the evidence oracle.
STAGE_POLICIES = (
    "single",
    "sequential",
    "parallel",
    "pairwise",
    "fuse",
    "argmax",
    "ucb",
)
# Oracles: simulator prediction, deadline-capped real/sim race, or full real execution.
BON_ORACLES = ("sim", "race", "real")


def confidence_gate_stages(acfg) -> frozenset[str]:
    """Stages whose selected candidates face the p_max confidence gate.

    agent.real_validation_p_max_stages is "all", one stage name, or a
    comma-separated subset of draft/debug/improve; a sequence of names is
    also accepted. A missing setting keeps the original improve-only gate."""
    raw = getattr(acfg, "real_validation_p_max_stages", "improve")
    if raw is None:
        names = []
    elif isinstance(raw, str):
        names = [part.strip().lower() for part in raw.replace(",", " ").split()]
    else:
        names = [str(part).strip().lower() for part in raw]
    if "all" in names:
        return frozenset(STAGES)
    unknown = sorted(set(names) - set(STAGES))
    if not names or unknown:
        raise ValueError(
            f"agent.real_validation_p_max_stages={raw!r}; use 'all' or a "
            f"comma-separated subset of {list(STAGES)}"
        )
    return frozenset(names)


@dataclass
class CandidateOutcome:
    """Candidate evidence shared by all selection policies.

    When predicted=False, result is a completed execution to reuse; workspace
    holds its submission. WM predictions can be reused in simulator-only mode.
    Race results require re-execution even when
    result.real_won=True because their workspaces were removed. An optional
    review provides advisory findings."""

    result: ExecutionResult
    usable: bool
    submission_exists: bool | str | None
    predicted: bool
    workspace: Path | None = None  # real oracle: cand dir holding submission/
    # Optional reviewer findings; absent reviews add nothing to selection prompts.
    review: dict | None = None

    @property
    def term_out(self) -> str:
        # Match Node.term_out: chunks may be partial-line writes, not whole lines.
        return "".join(self.result.term_out or [])

    @property
    def analysis(self) -> str:
        """The reviewer's summary of THIS candidate; "" when not reviewed."""
        return ((self.review or {}).get("summary") or "").strip()


# Stable improve_axes_v1 directives; changing wording requires a new prompt version.
IMPROVE_EXPLORE_AXES = (
    "feature engineering: add, transform, or select input features",
    "a different model family or architecture than the current solution",
    "ensembling or stacking multiple models",
    "preprocessing or target transforms (scaling, encoding, imputation)",
    "regularization or the training schedule (early stopping, LR schedule, " "augmentation)",
    "hyperparameter changes to the existing pipeline",
)

review_func_spec = FunctionSpec(
    name="submit_review",
    json_schema={
        "type": "object",
        "properties": {
            "is_bug": {
                "type": "boolean",
                "description": "true only if the output log shows an execution failure, "
                "such as an unhandled exception, crash, or timeout; otherwise false. "
                "Report metric reliability separately in metric_trustworthy. "
                "The harness checks actual submission files and their format; guessed "
                "CSV row counts, IDs, columns, or paths are not execution failures.",
            },
            "has_csv_submission": {
                "type": "boolean",
                "description": "true if the code saves the predictions on the test data"
                " in a `submission.csv` file in the `./submission/` directory, otherwise false."
                " Note that the file MUST be saved in the ./submission/ directory for this to be evaluated as true."
                " Otherwise, it should be evaluated as false."
                " You can assume the ./submission/ directory exists and is writable.",
            },
            "summary": {
                "type": "string",
                "description": "write a short summary (2-3 sentences) describing "
                " the empirical findings. Alternatively mention if there is a bug or"
                " the submission.csv was not properly produced."
                " DO NOT suggest fixes or improvements.",
            },
            "metric": {
                "type": ["number", "null"],
                "description": "If the code ran successfully, report the value of the validation metric. Otherwise, leave it null.",
            },
            "lower_is_better": {
                "type": "boolean",
                "description": "true if the metric should be minimized (i.e. a lower metric value is better, such as with MSE), false if the metric should be maximized (i.e. a higher metric value is better, such as with accuracy).",
            },
            "metric_trustworthy": {
                "type": "boolean",
                "description": "true if the printed validation score is a trustworthy"
                " estimate of held-out/test performance: a sound hold-out split scored"
                " without refitting on the validation rows, AND the validation metric is the"
                " task's official metric on the same numeric SCALE; false if the code shows"
                " no real hold-out (resubstitution), OR computes the validation metric with a"
                " different formula/scale than the official metric (a square root omitted or"
                " added — MSE vs RMSE, MSLE vs RMSLE; log vs log1p; wrong reduction or"
                " denominator; negated sign) so the printed score is not comparable.",
            },
        },
        "required": [
            "is_bug",
            "has_csv_submission",
            "summary",
            "metric",
            "lower_is_better",
            "metric_trustworthy",
        ],
    },
    description="Submit a review evaluating the output of the training script.",
)

# Require a concrete hold-out or metric-scale flaw before rejecting reliability.
VALIDATION_REVIEW_GUIDANCE = (
    "Before reporting the metric, judge whether the printed VALIDATION score is a "
    "trustworthy estimate of held-out / test performance. Set `metric_trustworthy` to "
    "false when the code shows a concrete, identifiable flaw, e.g.:\n"
    "- No real hold-out: scored on training data — no split, or the model is refit on "
    "the full data (incl. the validation rows) before scoring (resubstitution).\n"
    "- Wrong metric SCALE: the validation score is computed with a different formula than "
    "the task's official metric so its numeric scale is not comparable — e.g. a square root "
    "omitted or added (MSE vs RMSE, MSLE vs RMSLE), log vs log1p, the wrong "
    "reduction/denominator, or a negated sign. (The official metric is named in the prompt; "
    "compare the code's computation to it.)\n"
    "Otherwise set `metric_trustworthy` to true. Computing the official metric BY HAND is NOT "
    "a flaw as long as it yields the SAME metric on the SAME scale — only flag a concrete "
    "formula/scale mismatch (or a missing hold-out)."
)


def normalize_review_response(response):
    """Normalize recognized tool values without Python string truthiness."""
    import math

    if not isinstance(response, dict):
        response = {}
    normalized = {
        "is_bug": False,
        "has_csv_submission": True,
        "metric": None,
        "lower_is_better": False,
        "metric_trustworthy": None,
        "summary": "(review unavailable: feedback model returned no usable tool call)",
    } | response
    issues = (
        [issue for issue in response.get("review_issues", []) if isinstance(issue, str)]
        if isinstance(response.get("review_issues"), list)
        else []
    )
    for key in ("is_bug", "has_csv_submission", "lower_is_better", "metric_trustworthy"):
        value = normalized[key]
        if isinstance(value, str):
            value = {
                "true": True,
                "1": True,
                "yes": True,
                "false": False,
                "0": False,
                "no": False,
            }.get(value.strip().lower())
        if value is not None and not isinstance(value, bool):
            value = None
        if key in response and value is None and response[key] is not None:
            issues.append(
                f"Reviewer returned an unrecognized boolean for {key}: {response[key]!r}."
            )
        normalized[key] = value
    metric = normalized["metric"]
    try:
        if isinstance(metric, bool) or not isinstance(metric, (int, float, str)):
            raise ValueError
        metric = float(metric)
        if not math.isfinite(metric):
            raise ValueError
    except (ValueError, OverflowError):
        metric = None
        normalized.setdefault(
            "metric_error",
            "Missing or invalid validation metric: expected a finite number, excluding booleans.",
        )
    if any("metric_trustworthy" in issue for issue in issues):
        metric = None
        normalized["metric_error"] = "Reviewer metric reliability verdict is malformed."
    normalized["metric"] = metric
    if not isinstance(normalized["summary"], str) or not normalized["summary"].strip():
        normalized["summary"] = "(review provided no summary)"
    normalized["review_issues"] = issues
    return normalized


def review_is_buggy(response, crashed, has_csv_submission, *, execution_observed=False):
    """Reject failed executions, missing metrics/CSVs, or untrustworthy scores.

    Real executions use the interpreter's crash status; predictions also use
    the reviewer's execution flag. CSV presence comes from the caller, never
    the reviewer. Missing trust verdicts remain compatible with older reviews.
    Keep this helper self-contained for offline AST-based consumers."""
    import math

    metric = response.get("metric")
    return (
        (not execution_observed and response.get("is_bug") is True)
        or crashed
        or isinstance(metric, bool)
        or not isinstance(metric, (int, float))
        or not math.isfinite(metric)
        or has_csv_submission == False
        or response.get("metric_trustworthy") is False
    )


def _candidate_is_usable(result: ExecutionResult) -> bool:
    """Use captured real facts for race winners and prediction rules for sims."""
    if result.real_won:
        return result.exc_type is None and result.observed_submission_exists is True
    return is_predicted_candidate_usable(result)


def _result_has_submission_csv(workspace_dir, exec_result: ExecutionResult) -> bool:
    """Use disk evidence for real runs and explicit CSV predictions for simulations.

    Unknown predictions never fall back to files left by earlier executions."""
    if exec_result.real_won:
        return _has_submission_csv(workspace_dir)
    return exec_result.predicted_submission_exists == "true"


def metric_grounded_in_term_out(metric, term_out):
    """Check that a finite reviewer metric appears in the execution output.

    Allow display rounding and sign changes, but reject magnitude errors such
    as reading 0.2676 as 26.76. Percent-only output does not ground a fraction.
    Keep this helper self-contained for offline AST-based consumers."""
    import math
    import re

    if (
        isinstance(metric, bool)
        or not isinstance(metric, (int, float))
        or not math.isfinite(metric)
    ):
        return False
    for tok in re.findall(r"\d*\.?\d+(?:[eE][-+]?\d+)?", term_out):
        try:
            # Allow sign conventions to differ while rejecting magnitude errors.
            if math.isclose(abs(metric), abs(float(tok)), rel_tol=1e-2, abs_tol=1e-4):
                return True
        except ValueError:
            continue
    return False


listwise_pick_func_spec = FunctionSpec(
    name="pick_best_candidate",
    json_schema={
        "type": "object",
        "properties": {
            "best_candidate": {
                "type": "integer",
                "description": "Index of the single best candidate, chosen from"
                " the candidate indices listed in the prompt.",
            },
            "reason": {
                "type": "string",
                "description": "1-2 sentences: why this candidate over the others.",
            },
        },
        "required": ["best_candidate", "reason"],
    },
    description="Select the best of the executed candidate solutions.",
)


class Agent(GenerationPromptsMixin, CandidateSelectionMixin, SubmissionReviewMixin):
    """Draft, debug, and improve solutions using configurable generation policies."""

    def __init__(
        self,
        task_desc: str,
        cfg: Config,
        journal: Journal,
        sim_predict: Callable[[str], ExecutionResult] | None = None,
        race_predict: Callable[[str], ExecutionResult] | None = None,
        sim_p_clean: Callable[[str], dict | float | None] | None = None,
        deadline: float | None = None,
        sim_metric_posterior: Callable[..., dict] | None = None,
    ):
        super().__init__()
        self.task_desc = task_desc
        self.cfg = cfg
        self.acfg = cfg.agent
        self.journal = journal
        self.data_preview: str | None = None
        self.start_time = time.time()
        self.deadline = (
            float(deadline)
            if deadline is not None
            else self.start_time + float(self.acfg.time_limit)
        )
        self.current_step = 0
        self.sim_predict = sim_predict
        self.race_predict = race_predict
        # Keep the raw scorer for UCB without enabling confidence in other stages.
        self._exc_scores = sim_p_clean
        self._metric_posterior = sim_metric_posterior
        # Posterior-based selectors need this hook; the listwise judge does not.
        self._p_clean = (
            sim_p_clean
            if sim_p_clean is not None
            and (
                bool(getattr(self.acfg, "use_p_clean_argmax_pick", False))
                # Argmax needs scoring even when the global confidence flags are off.
                or any(str(getattr(self.acfg.stage_policy, s)) == "argmax" for s in STAGES)
            )
            else None
        )
        self.oracle = str(self.acfg.bon_oracle)
        self._predict = (
            race_predict if self.oracle == "race" and race_predict is not None else sim_predict
        )
        self.stage_policy = self._resolve_stage_policies()

    def search_policy(self) -> Node | None:
        """AutoMind parent selection without HPO: draft, debug, then greedy/tree-best."""
        search_cfg = self.acfg.search
        self._parent_selection = "draft"

        # initial drafting
        if len(self.journal.draft_nodes) < search_cfg.num_drafts:
            logger.info("[search policy] drafting new node (not enough drafts)")
            return None

        # debugging
        if random.random() < search_cfg.debug_prob:
            # nodes that are buggy + leaf nodes + debug depth < max debug depth
            debuggable_nodes = [
                n
                for n in self.journal.buggy_nodes
                if (n.is_leaf and n.debug_depth < search_cfg.max_debug_depth)
            ]
            if debuggable_nodes:
                node_to_debug = random.choice(debuggable_nodes)
                self._parent_selection = "debug"
                logger.info(f"[search policy] debugging node {node_to_debug.id}")
                return node_to_debug

        # back to drafting if no nodes to improve
        if not self.journal.good_nodes:
            logger.info("[search policy] drafting new node (no good nodes)")
            return None

        # Explore uniformly across draft trees, giving each tree one candidate.
        # The global best stays in this pool, as in AutoMind.
        if random.random() >= getattr(search_cfg, "greedy_prob", 0.7):
            tree_best_nodes = self.journal.get_best_nodes_per_tree()
            if tree_best_nodes:
                parent = random.choice(tree_best_nodes)
                self._parent_selection = "tree_best"
                logger.info(f"[search policy] random tree-best node selected: node {parent.id}")
                return parent

        greedy_node = self.journal.get_best_node()
        self._parent_selection = "greedy"
        logger.info(f"[search policy] greedy node selected: node {greedy_node.id}")
        return greedy_node

    def step(self, exec_callback: ExecCallbackType):
        """Generate, execute/reuse, and review one node before updating the journal.

        Selection may retain a predicted result; real artifact validation and
        publication remain separate so simulated success cannot publish a CSV.
        """
        # clear the submission dir from previous steps
        reset_submission_directory(self.cfg.workspace_dir)

        if not self.journal.nodes or self.data_preview is None:
            self.update_data_preview()

        self._parent_selection = None
        parent_node = self.search_policy()
        logger.info(f"Agent is generating code, parent node type: {type(parent_node)}")

        if parent_node is None:
            stage, prompt = "draft", self._draft_prompt()
            # Only drafts receive the stage-specific output cap.
            max_tokens = _draft_max_tokens()
        elif parent_node.is_buggy:
            stage = "debug"
            prompt = self._build_debug_prompt(
                parent_node.code,
                parent_node.term_out,
                parent_node.analysis or "",
                parent_node.failure_reasons,
            )
            max_tokens = None
        else:
            stage, prompt = "improve", self._improve_prompt(parent_node)
            max_tokens = None
        # Preserve both log formats: offline extractors distinguish real BoN prompts.
        if self.oracle == "real" and self.stage_policy[stage] != "single":
            label = {"draft": "drafting", "debug": "debug", "improve": "improve"}[stage]
            logger.info(
                f"[real-only best-of-{int(self.acfg.best_of_n_size)}] {label} "
                f"prompt:\n\n{json.dumps(prompt, indent=4)}\n\n"
            )
        else:
            gerund = {"draft": "drafting", "debug": "debugging", "improve": "improving"}[stage]
            logger.info(f"Prompt for {gerund}: \n\n{json.dumps(prompt, indent=4)}\n\n")

        self._execution_cost_stage = stage
        plan, code, exec_result = self.generate_code(
            prompt, stage, parent_node, max_tokens=max_tokens
        )
        result_node = Node(
            plan=plan,
            code=code,
            parent=parent_node,
            parent_selection=self._parent_selection,
        )
        self._execution_cost_node = result_node
        cost_recorder = getattr(self, "execution_cost_recorder", None)
        if cost_recorder is not None:
            cost_recorder.emit("node_selected", **cost_recorder.node_fields(result_node))
        selection = getattr(self, "_selected_ucb", None)
        if selection is not None and selection[0] == code:
            result_node.ucb_selection = dict(selection[1])
        try:
            # Offline extractors match these node-lifecycle messages exactly.
            if parent_node is None:
                logger.info(f"Drafted new node {result_node.id}")
            elif stage == "debug":
                logger.info(f"Debugged node {parent_node.id} to create new node {result_node.id}")
            else:
                logger.info(f"Improved node {parent_node.id} to create new node {result_node.id}")
            if exec_result is None:
                # Execute selections without a reusable real or simulated result.
                exec_result = exec_callback(result_node.code, True)

            if cost_recorder is None:
                result_node = self.parse_exec_result(node=result_node, exec_result=exec_result)
            else:
                result_node = cost_recorder.review(
                    result_node,
                    lambda: self.parse_exec_result(node=result_node, exec_result=exec_result),
                )
            universal_gate = outcome_gate_enabled(self.acfg)
            score_for_gate = (
                (
                    result_node.exc_type != "TimeoutError"
                    and (
                        random_gate_enabled(self.acfg)
                        or outcome_gate_threshold(self.acfg, result_node.exc_type) is not None
                    )
                )
                if universal_gate
                else (
                    getattr(self.acfg, "real_validation_p_max_threshold", None) is not None
                    and stage in confidence_gate_stages(self.acfg)
                )
            )
            if not result_node.real_won and score_for_gate:
                (
                    result_node.sim_p_max,
                    result_node.sim_top_exc,
                    result_node.sim_p_top_exc,
                ) = self._selected_confidence(result_node.code)
            if not result_node.is_buggy and not _result_has_submission_csv(
                self.cfg.workspace_dir, exec_result
            ):
                result_node.is_buggy = True
                result_node.metric = WorstMetricValue()
                self._record_failure(
                    result_node,
                    "submission",
                    "Submission CSV disappeared after review; expected a nonempty submission/submission.csv.",
                )
                logger.info(f"Actually, node {result_node.id} did not produce a submission.csv")
            self.journal.append(result_node)
            if cost_recorder is not None:
                cost_recorder.node_result(result_node)
        except BaseException:
            # Node construction links the parent immediately. An aborted step
            # must leave that parent eligible for another debug attempt.
            if parent_node is not None and result_node not in self.journal.nodes:
                parent_node.children.discard(result_node)
            raise

        # Only validated real results compete for publication.
        best_real = self.journal.get_best_node(validated_only=True)
        if exec_result.real_won and best_real is result_node:
            if (
                publish_real_solution(self.cfg.workspace_dir, result_node)
                and cost_recorder is not None
            ):
                cost_recorder.node_result(result_node, event="publication")

        best_node = self.journal.get_best_node()
        if best_node is not None:
            if best_node.id == result_node.id:
                logger.info(f"Node {result_node.id} is the best node so far")
            else:
                logger.info(f"Node {result_node.id} is not the best node")
                logger.info(f"Node {best_node.id} is still the best node")
        self.current_step += 1

    def generate_code(
        self,
        prompt,
        stage: str,
        parent_node: "Node | None" = None,
        max_tokens=None,
    ) -> tuple[str, str, ExecutionResult | None]:
        """Return (plan, code, result) using the configured stage policy and oracle.

        A reusable real or simulated result skips execution by the caller."""
        # Confidence belongs to this selection, even if another step repeats code.
        self._selected_exc_scores = None
        self._selected_ucb = None
        method = {
            "single": self._single,
            "sequential": self._sequential,
            "parallel": self._parallel,
            "pairwise": self._pairwise,
            "fuse": self._fuse,
            "argmax": self._argmax,
            "ucb": self._ucb,
        }[self.stage_policy[stage]]
        return method(prompt, stage, parent_node, max_tokens)

    def _selected_confidence(self, code: str) -> tuple[float | None, str | None, float | None]:
        """Reuse the selected posterior; score only the committed code if absent."""
        cached = getattr(self, "_selected_exc_scores", None)
        if cached is not None and cached[0] == code:
            posterior = cached[1]  # None records a failed selection-time call.
        else:
            try:
                posterior = self._exc_scores(code)
            except Exception as exc:
                logger.warning(f"[confidence-gate] selected candidate scoring failed: {exc!r}")
                posterior = None
        if not isinstance(posterior, dict):
            return None, None, None
        p_max = ucb.finite_number(posterior.get("max_p"))
        top_exc = posterior.get("top_exc")
        p_top_exc = ucb.finite_number(posterior.get("p_top_exc"))
        return (
            p_max if p_max is not None and 0 <= p_max <= 1 else None,
            top_exc if isinstance(top_exc, str) and top_exc else None,
            p_top_exc if p_top_exc is not None and 0 <= p_top_exc <= 1 else None,
        )

    def _query(self, *, reserve_s=None, **kwargs):
        """Bound each LLM call and its retries by the remaining run allowance."""
        deadline = getattr(self, "deadline", None)
        if deadline is not None:
            if reserve_s is None:
                reserve_s = max(0.0, float(os.getenv("AIDE_SHUTDOWN_RESERVE_S", "180")))
            remaining = deadline - time.time() - reserve_s
            if remaining <= 0:
                raise TimeoutError("LLM query deadline exhausted before request")
            for key in ("retry_max_time", "retry_attempt_timeout"):
                limit = kwargs.get(key)
                kwargs[key] = remaining if limit is None else min(limit, remaining)
        return query(**kwargs)

    def plan_and_code_query(self, prompt, retries=3, max_tokens=None) -> tuple[str, str]:
        """Generate a plan and code, retrying unparseable completions.

        max_tokens overrides the sampling cap on backends that support it."""
        plan, code = "", ""
        for _ in range(retries):
            completion_text = self._query(
                system_message=prompt,
                user_message=None,
                model=self.acfg.code.model,
                temperature=self.acfg.code.temp,
                max_tokens=max_tokens,
                convert_system_to_user=self.acfg.convert_system_to_user,
            )

            plan, code, parsed = _split_plan_code(completion_text)
            if parsed:
                logger.info("Plan + code extraction successed")
                logger.info(f"Generated natural language plan: \n\n{plan}\n\n")
                logger.info(f"Generated code: \n\n{code}\n\n")
                return plan, code
            logger.info("Plan + code extraction failed, retrying...")
        logger.info("Final plan + code extraction attempt failed, giving up...")
        return plan, code or ""

    def update_data_preview(self) -> None:
        self.data_preview = data_preview.generate(self.cfg.workspace_dir)

    def parse_exec_result(self, node: Node, exec_result: ExecutionResult) -> Node:
        """Attach execution evidence, review, and artifact diagnoses to a node."""
        logger.info(f"Agent is parsing execution results for node {node.id}")

        node.absorb_exec_result(exec_result)
        node.pending_submission = None

        validation, processing = None, {}
        if exec_result.real_won:
            node.code, validation, processing = self._process_real_submission(
                self.cfg.workspace_dir, exec_result, node.code
            )
        node.submission_actions = processing.get("actions", [])
        node.failure_reasons = {}

        response = self._review(node.code, node.term_out, subject=f"Node {node.id}")

        has_csv_submission = _result_has_submission_csv(self.cfg.workspace_dir, exec_result)
        if not exec_result.real_won and exec_result.predicted_submission_exists is not None:
            # Use the checkpoint's explicit CSV prediction instead of reviewer inference.
            response = {**response, "has_csv_submission": has_csv_submission}
        node.csv_exists = has_csv_submission
        # Private grades are measurement-only and require a real submission.
        node.test_metric = (
            self._grade_test_submission()
            if (exec_result.real_won and has_csv_submission and self.cfg.agent.record_test_metric)
            else None
        )

        # Normalizing here as well protects alternate/offline reviewer adapters.
        response = normalize_review_response(response)
        node.analysis = response["summary"]
        for note in response.get("review_issues", []) + processing.get("diagnostics", []):
            node.analysis += f"\n{note}"
        # Real exception names are authoritative; predictions also need validity checks.
        crashed = (
            exec_result.exc_type is not None
            if exec_result.real_won
            else is_bad_prediction(exec_result)
        )
        node.is_buggy = review_is_buggy(
            response, crashed, has_csv_submission, execution_observed=exec_result.real_won
        )
        if crashed:
            detail = exec_result.exc_info or {}
            self._record_failure(
                node,
                "execution",
                f"{'Observed' if exec_result.real_won else 'Predicted'} "
                f"execution failed: {exec_result.exc_type or 'unusable simulator result'}; {detail}.",
            )
        elif response["is_bug"] is True:
            if exec_result.real_won:
                node.analysis += "\nHarness execution completed without an exception; the reviewer's execution flag was not confirmed. CSV validity is checked separately."
            else:
                self._record_failure(
                    node,
                    "execution",
                    "Reviewer identified a failure in the predicted execution output.",
                )
        if response["metric"] is None:
            self._record_failure(
                node,
                "metric",
                response.get("metric_error", "No usable validation metric was reported."),
            )
        if response.get("metric_trustworthy") is False:
            self._record_failure(
                node,
                "metric",
                "Validation metric rejected as untrustworthy; review the hold-out and official metric formula/scale. "
                + response["summary"],
            )
        if not has_csv_submission:
            expected = self.cfg.workspace_dir / "submission/submission.csv"
            reason = (
                f"Missing nonempty regular submission CSV at expected path {expected}."
                if exec_result.real_won
                else f"Simulator did not positively predict a submission CSV: {exec_result.predicted_submission_exists!r}. Expected path {expected}."
            )
            self._record_failure(node, "submission", reason)
        if self.acfg.eval_lower_is_better is None and response["lower_is_better"] is None:
            node.is_buggy = True
            self._record_failure(
                node,
                "metric",
                "Metric direction is unavailable: no configured direction or recognized reviewer verdict.",
            )
        # Keep the raw trust verdict; absent legacy fields remain None.
        node.metric_trustworthy = response.get("metric_trustworthy")

        # Re-execution invalidates the previous CSV's validation evidence.
        node.submission_validation = None
        node.submission_validation_message = None
        node.submission_sha256 = None
        node.submission_columns = []

        if node.is_buggy:
            node.metric = WorstMetricValue()
        else:
            # Prefer the configured metric direction over the reviewer's guess.
            if self.acfg.eval_lower_is_better is not None:
                maximize = not self.acfg.eval_lower_is_better
            else:
                maximize = not response["lower_is_better"]
            node.metric = MetricValue(response["metric"], maximize=maximize)

        if validation is not None:
            self._apply_submission_validation(node, validation)

        self._check_data_leakage(node)
        retain_pending_submission(self, node)

        if node.is_buggy:
            logger.info(
                f"Parsed results: Node {node.id} is buggy and/or did not produce a submission.csv"
            )
        else:
            logger.info(f"Parsed results: Node {node.id} is not buggy")
        return node

    def _resolve_stage_policies(self) -> dict[str, str]:
        """Validate stage/oracle routing before the first step.

        Missing width/scoring prerequisites downgrade to single with a warning.
        Invalid policies or incompatible oracles raise. Pairwise and fusion
        configuration is validated here rather than failing during generation."""
        acfg = self.acfg
        greedy_prob = ucb.finite_number(getattr(getattr(acfg, "search", None), "greedy_prob", 0.8))
        if greedy_prob is None or not 0 <= greedy_prob <= 1:
            raise ValueError("agent.search.greedy_prob must be in [0, 1]")
        validate_execution_gate(acfg)
        threshold = getattr(acfg, "real_validation_p_max_threshold", None)
        if outcome_gate_enabled(acfg):
            thresholds = (
                outcome_gate_threshold(acfg, None),
                outcome_gate_threshold(acfg, "failure"),
            )
            if not getattr(self.cfg.exec, "simulator", False):
                raise ValueError("agent.execution_gate requires exec.simulator=True")
            if (
                random_gate_enabled(acfg) or any(t is not None for t in thresholds)
            ) and self._exc_scores is None:
                raise ValueError(
                    "agent.execution_gate thresholds require the exception-posterior scorer"
                )
            if random_gate_enabled(acfg):
                logger.info(
                    "[execution-gate] random check probability: %g; mandatory checks retained",
                    acfg.execution_gate.random_check_probability,
                )
            else:
                logger.info(
                    "[confidence-gate] outcome thresholds on all stages: "
                    f"success={thresholds[0]}, failure={thresholds[1]} "
                    "(None=always real except timeouts); best/periodic checks retained"
                )
        elif threshold is not None:
            value = ucb.finite_number(threshold)
            if value is None or not 0 <= value <= 1:
                raise ValueError("agent.real_validation_p_max_threshold must be null or in [0, 1]")
            if (
                not acfg.use_validate_on_best
                or not getattr(self.cfg.exec, "simulator", False)
                or self._exc_scores is None
            ):
                raise ValueError(
                    "agent.real_validation_p_max_threshold requires use_validate_on_best=True "
                    "and exec.simulator=True with the exception-posterior scorer"
                )
            gated = confidence_gate_stages(acfg)
            logger.info(
                f"[confidence-gate] p_max<{value:g} triggers a real check on "
                f"stages {sorted(gated, key=STAGES.index)}"
            )
        if self.oracle not in BON_ORACLES:
            raise ValueError(
                f"agent.bon_oracle={self.oracle!r}; valid oracles are " f"{list(BON_ORACLES)}"
            )
        if getattr(acfg, "use_observed_metric_argmax_pick", False):
            if self.oracle != "real" or not getattr(acfg, "bon_candidate_review", True):
                raise ValueError(
                    "Observed metric selection requires bon_oracle=real and candidate reviews"
                )
            if getattr(acfg, "use_predicted_metric_argmax_pick", False):
                raise ValueError("Choose either observed or predicted metric selection")
        # A simulator-only run cannot execute real BoN candidates.
        if getattr(self.cfg.exec, "simulator_only", False) and self.oracle == "real":
            raise ValueError(
                "agent.bon_oracle=real is invalid under exec.simulator_only: "
                "the real oracle executes every candidate for real. Use "
                "bon_oracle sim."
            )
        routing: dict[str, str] = {}
        for stage in STAGES:
            name = str(getattr(acfg.stage_policy, stage))
            if name not in STAGE_POLICIES:
                raise ValueError(
                    f"agent.stage_policy.{stage}={name!r}; valid policies are "
                    f"{list(STAGE_POLICIES)}"
                )
            if name == "ucb":
                options = ucb.resolve_config(getattr(acfg, "ucb", None), stage)
                if self.oracle != "real" and int(acfg.best_of_n_size) > 1:
                    if self._predict is None:
                        raise ValueError(f"agent.stage_policy.{stage}=ucb requires exec.simulator")
                    scorer_name = (
                        "_metric_posterior" if options.metric_posterior.enabled else "_exc_scores"
                    )
                    if options.beta > 0 and getattr(self, scorer_name, None) is None:
                        scorer_kind = (
                            "metric-posterior"
                            if options.metric_posterior.enabled
                            else "exc-posterior"
                        )
                        raise ValueError(
                            f"agent.stage_policy.{stage}=ucb with beta>0 requires the "
                            f"world-model {scorer_kind} scorer; enable exec.simulator"
                        )
                logger.info(f"[ucb] {stage} settings: {options}")
            if name == "sequential" and self.oracle == "real":
                raise ValueError(
                    f"agent.stage_policy.{stage}=sequential needs a predictive "
                    "oracle: screening a draw with the real oracle costs a "
                    "full execution (that is the debug loop). Use bon_oracle "
                    "sim/race, or policy single/parallel."
                )
            if name != "single" and int(acfg.best_of_n_size) <= 1:
                logger.warning(
                    f"[stage policy] {stage}: {name} needs best_of_n_size>1; "
                    f"downgrading to single"
                )
                name = "single"
            if name == "argmax" and self._p_clean is None:
                logger.warning(
                    f"[stage policy] {stage}: {name} needs the exc-posterior "
                    f"scorer (exec.simulator with world_9_3); downgrading to "
                    f"single"
                )
                name = "single"
            if name != "single" and self.oracle != "real" and self._predict is None:
                logger.warning(
                    f"[stage policy] {stage}: {name} needs a simulator "
                    f"(exec.simulator) under bon_oracle={self.oracle}; "
                    f"downgrading to single"
                )
                name = "single"
            routing[stage] = name
        if "pairwise" in routing.values():
            preference.resolve_config(self._pairwise_variant())
        for stage, policy in routing.items():
            if policy != "fuse":
                continue
            # Validate the same report provenance and candidate order used during fusion.
            prompt_style, style_params = self._fuse_prompt_settings(stage)
            FusionPromptConfig(
                mode=str(acfg.fuse_mode),
                prompt_style=prompt_style,
                style_params=style_params,
                report_source="real" if self.oracle == "real" else "wm",
                candidate_order="input",
            )
        logger.info(f"[stage policy] oracle={self.oracle} routing: {routing}")
        return routing

    def _single(self, prompt, stage, parent_node, max_tokens):
        """One draw straight to execution: plain AIDE, no oracle."""
        del stage, parent_node
        plan, code = self.plan_and_code_query(prompt, max_tokens=max_tokens)
        return plan, code, None

    def _parallel(self, prompt, stage, parent_node, max_tokens, _selector=None):
        """Draw, observe, select, and commit a fixed-width candidate pool.

        _selector overrides global selection flags for this stage only."""
        candidates = self._draw_candidates(prompt, max_tokens, stage=stage)
        step_log_dir = self._new_step_log_dir()
        self._persist_draw_axes(step_log_dir)
        outcomes = self._observe(candidates, step_log_dir)
        context = (
            {"stage": stage, "parent_node": parent_node}
            if _selector in {"ucb", "p_clean_argmax"}
            else {}
        )
        picked = self._select(candidates, outcomes, step_log_dir, selector=_selector, **context)
        return self._commit(candidates, outcomes, picked, step_log_dir)

    def _argmax(self, prompt, stage, parent_node, max_tokens):
        """Use stage-local p_clean ranking with UCB's metric eligibility/fallbacks.

        The legacy global argmax flag on parallel stages keeps its own semantics."""
        return self._parallel(prompt, stage, parent_node, max_tokens, _selector="p_clean_argmax")

    def _ucb(self, prompt, stage, parent_node, max_tokens):
        """Use the shared optimistic selector for this stage's fixed N draws."""
        return self._parallel(prompt, stage, parent_node, max_tokens, _selector="ucb")

    def _pairwise(self, prompt, stage, parent_node, max_tokens):
        """Select by pairwise tournament, bypassing viability and listwise judging."""
        del parent_node
        candidates = self._draw_candidates(prompt, max_tokens, stage=stage)
        step_log_dir = self._new_step_log_dir()
        self._persist_draw_axes(step_log_dir)
        outcomes = self._observe(candidates, step_log_dir)
        picked = self._pairwise_select(stage, candidates, outcomes, step_log_dir)
        return self._commit(candidates, outcomes, picked, step_log_dir)

    def _fuse(self, prompt, stage, parent_node, max_tokens):
        """Synthesize one program from a candidate pool, then execute it.

        Real-oracle fusion costs N candidate executions plus the fused execution."""
        candidates = self._draw_candidates(prompt, max_tokens, stage=stage)
        step_log_dir = self._new_step_log_dir()
        self._persist_draw_axes(step_log_dir)
        outcomes = self._observe(candidates, step_log_dir)
        if str(self.acfg.fuse_mode) == "synth":
            # Synthesis only needs a fallback pick, so it spends no judge call.
            usable = [i for i, o in enumerate(outcomes) if o.usable]
            picked = usable[0] if usable else None
        else:
            # Editing modes need the same backbone selection as the parallel policy.
            picked = self._select(candidates, outcomes, step_log_dir)
        fused = self._fuse_candidates(
            prompt=prompt,
            stage=stage,
            parent_node=parent_node,
            candidates=candidates,
            outcomes=outcomes,
            picked_idx=picked,
            step_log_dir=step_log_dir,
        )
        if fused is None:
            # Fusion declined; commit the selected candidate.
            return self._commit(candidates, outcomes, picked, step_log_dir)
        # Only the fused program becomes a journal node; reject all source candidates.
        if outcomes[0].predicted:
            self._write_bon_candidate_artifacts(
                step_log_dir, candidates, [o.result for o in outcomes], None
            )
        else:
            for i in range(len(candidates)):
                result_file = step_log_dir / f"cand_{i}" / "result.json"
                audit = json.loads(result_file.read_text())
                audit["accepted"] = False
                audit["reject_reason"] = "not_executed:fused_program_ran_instead"
                result_file.write_text(json.dumps(audit))
            shutil.rmtree(self.cfg.workspace_dir / "bon", ignore_errors=True)
        (step_log_dir / "picked.txt").write_text("fused")
        logger.info(f"[best-of-{len(candidates)}] executing FUSED program ({stage} stage)")
        return (*fused, None)

    def _sequential(self, prompt, stage, parent_node, max_tokens):
        """Refine through N predictions, then select by metric or failed-pool judge.

        Every revision conditions on the preceding candidate and its evidence.
        Only the selected candidate becomes a journal node; intermediate
        revisions must not change the outer search tree."""
        del parent_node
        assert self._predict is not None  # guarded by _resolve_stage_policies
        n = int(self.acfg.best_of_n_size)
        step_log_dir = self._new_step_log_dir()
        attempts: list[tuple[str, str]] = []
        outcomes: list[CandidateOutcome] = []
        for attempt in range(n):
            cand_log = step_log_dir / f"cand_{attempt}"
            cand_log.mkdir()
            (cand_log / "prompt.json").write_text(json.dumps(prompt, indent=2, default=str))
            candidate = self._generate_n_candidates(prompt, 1, max_tokens=max_tokens)[0]
            sim_res = self._predict(candidate[1])
            outcome = CandidateOutcome(
                result=sim_res,
                usable=_candidate_is_usable(sim_res),
                submission_exists=(
                    sim_res.observed_submission_exists
                    if sim_res.real_won
                    else sim_res.predicted_submission_exists
                ),
                predicted=True,
            )
            attempts.append(candidate)
            outcomes.append(outcome)
            logger.info(
                f"[sequential] {stage}: revision {attempt + 1}/{n} "
                f"usable={outcome.usable} (exc_type={sim_res.exc_type!r})"
            )
            if attempt + 1 < n:
                prompt = self._sequential_refinement_prompt(candidate, outcome)

        # Oracle reports drive refinement directly; no per-candidate reviewer
        # is needed for numeric selection or the all-failed listwise fallback.
        picked = self._select(attempts, outcomes, step_log_dir, selector="predicted_metric_argmax")
        accepted = picked if picked is not None and outcomes[picked].usable else None
        (step_log_dir / "sequential.json").write_text(
            json.dumps(
                {
                    "mode": "refinement",
                    "stage": stage,
                    "budget": n,
                    "attempts": len(attempts),
                    "picked_attempt": picked,
                    "accepted_attempt": accepted,
                    # Historical flag: the budget produced no usable selection.
                    "exhausted": accepted is None,
                    "exc_types": [o.result.exc_type for o in outcomes],
                },
                default=str,
            )
        )
        return self._commit(attempts, outcomes, picked, step_log_dir)

    def _sequential_refinement_prompt(self, candidate, outcome: CandidateOutcome) -> dict:
        """Use the normal repair/improve prompts without linking temporary nodes."""
        plan, code = candidate
        previous = Node(code=code, plan=plan, analysis=outcome.analysis)
        previous.absorb_exec_result(outcome.result)
        metric = (
            ucb.finite_number(outcome.result.predicted_metric) if not previous.real_won else None
        )
        lower = getattr(self.acfg, "eval_lower_is_better", None)
        if metric is not None:
            previous.metric = MetricValue(
                metric, maximize=not lower if isinstance(lower, bool) else None
            )
        if outcome.usable:
            prompt = self._improve_prompt(previous)
        else:
            prompt = self._build_debug_prompt(code, previous.term_out, previous.analysis)
        prompt["Sequential candidate feedback"] = {
            "Evidence source": "Real execution" if previous.real_won else "World-model prediction",
            "Exception": str(previous.exc_type),
            "Submission exists": str(outcome.submission_exists),
            "Predicted validation metric": str(metric),
        }
        if not previous.real_won:
            prompt["Sequential candidate feedback"]["Evidence note"] = (
                "This is predicted feedback, not an observed execution. Missing or "
                "malformed predictions do not establish a code defect. Use the code "
                "and available report to guide the next revision."
            )
        return prompt

    def _draw_candidates(self, prompt, max_tokens, stage=None) -> list[tuple[str, str]]:
        """N candidates for a width policy; tolerate a short Vertex return."""
        n = int(self.acfg.best_of_n_size)
        self._last_draw_axes = None
        if (
            stage == "improve"
            and n > 1
            and bool(getattr(self.acfg, "improve_plan_axes", False))
            and isinstance(prompt, dict)
        ):
            candidates = self._generate_diverse_candidates(prompt, n, max_tokens)
        else:
            candidates = self._generate_n_candidates(prompt, n, max_tokens=max_tokens)
        if not candidates:
            raise RuntimeError("best-of-n generation returned no candidates")
        if len(candidates) != n:
            # Some backends return fewer choices; use the actual pool size.
            logger.warning(f"[best-of-{n}] got {len(candidates)}/{n} candidates")
        return candidates

    def _generate_n_candidates(
        self, prompt: Any, n: int, max_tokens=None, topup_rounds: int = 2
    ) -> list[tuple[str, str]]:
        """Batch N completions, then retry unparseable slots in batched top-ups.

        Successful replacements retain slot identity; exhausted slots keep their
        original best-effort code. Short backend replies leave unmatched slots
        untouched. max_tokens applies only on backends that support output caps."""

        def _sample(k: int) -> list[str]:
            texts = self._query(
                system_message=prompt,
                user_message=None,
                model=self.acfg.code.model,
                temperature=self.acfg.code.temp,
                max_tokens=max_tokens,
                convert_system_to_user=self.acfg.convert_system_to_user,
                n=k,
            )
            # query() returns a bare string for n=1 requests.
            return [texts] if isinstance(texts, str) else list(texts)

        parsed = [_split_plan_code(text) for text in _sample(n)]
        for round_index in range(topup_rounds):
            failed = [i for i, (_, _, ok) in enumerate(parsed) if not ok]
            if not failed:
                break
            logger.info(
                f"[best-of-N] {len(failed)}/{len(parsed)} candidates failed "
                f"plan+code split; top-up round {round_index + 1}/{topup_rounds}"
            )
            # A short reply replaces only the leading failed slots.
            for slot, text in zip(failed, _sample(len(failed))):
                replacement = _split_plan_code(text)
                if replacement[2]:
                    parsed[slot] = replacement
        candidates: list[tuple[str, str]] = []
        for nl_text, code, ok in parsed:
            if not ok:
                logger.info("[best-of-N] candidate failed plan+code split; best-effort code")
            candidates.append((nl_text, code))
        return candidates

    def _generate_diverse_candidates(self, prompt, n, max_tokens):
        """Draw one baseline and n-1 improve candidates with cycling exploration axes.

        Each uses the normal parser retries. Directives are appended to share the
        common prompt prefix, and input order determines candidate identity."""
        axes = [None] + [IMPROVE_EXPLORE_AXES[i % len(IMPROVE_EXPLORE_AXES)] for i in range(n - 1)]

        def _one(axis):
            return self.plan_and_code_query(
                _exploration_prompt(prompt, axis), max_tokens=max_tokens
            )

        with ThreadPoolExecutor(max_workers=n, thread_name_prefix="bon-axes") as pool:
            candidates = list(pool.map(_one, axes))
        self._last_draw_axes = ["baseline" if a is None else a for a in axes]
        logger.info(
            f"[improve-axes] improve_axes_v1: drew {n} candidates " f"(1 baseline + {n - 1} hinted)"
        )
        return candidates

    def _persist_draw_axes(self, step_log_dir: Path) -> None:
        """Persist exploration axes for offline analysis, then clear the pending list."""
        if getattr(self, "_last_draw_axes", None):
            (step_log_dir / "improve_axes.json").write_text(
                json.dumps(self._last_draw_axes, indent=1)
            )
        self._last_draw_axes = None

    def _observe(self, candidates, step_log_dir) -> list[CandidateOutcome]:
        """Collect candidate evidence from real execution or parallel sim/race calls.

        Optional per-candidate reviews are attached for every width policy."""
        if self.oracle == "real":
            outcomes = self._observe_real(candidates, step_log_dir)
        else:
            assert self._predict is not None  # guarded by _resolve_stage_policies
            with ThreadPoolExecutor(
                max_workers=len(candidates), thread_name_prefix="bon-sim"
            ) as pool:
                results = list(pool.map(lambda c: self._predict(c[1]), candidates))
            outcomes = [
                CandidateOutcome(
                    result=res,
                    usable=_candidate_is_usable(res),
                    submission_exists=(
                        res.observed_submission_exists
                        if res.real_won
                        else res.predicted_submission_exists
                    ),
                    # Race workspaces are discarded, so even observed winners
                    # require re-execution. result.real_won retains provenance.
                    predicted=True,
                )
                for res in results
            ]
        self._review_candidates(candidates, outcomes, step_log_dir)
        return outcomes

    def _observe_real(self, candidates, step_log_dir) -> list[CandidateOutcome]:
        """Execute candidates in isolated workspaces and retain their artifacts.

        Prefer explicitly valid submissions. Unknown validity is eligible only
        when no candidate is valid; invalid submissions are never eligible.
        Workspaces survive until commit/fusion so the winner's CSV can be copied."""
        n = len(candidates)
        bon_root = self.cfg.workspace_dir / "bon"
        shutil.rmtree(bon_root, ignore_errors=True)
        bon_root.mkdir(parents=True)
        canonical_input = self.cfg.workspace_dir / "input"
        cand_dirs: list[Path] = []
        for i in range(n):
            cand_dir = bon_root / f"cand_{i}"
            cand_dir.mkdir()
            (cand_dir / "submission").mkdir()
            (cand_dir / "working").mkdir()
            if canonical_input.exists():
                (cand_dir / "input").symlink_to(canonical_input)
            cand_dirs.append(cand_dir)

        # plan/solution.py up-front so they exist even if the exec crashes.
        cand_log_dirs: list[Path] = []
        for i, (plan, code) in enumerate(candidates):
            cand_log = step_log_dir / f"cand_{i}"
            cand_log.mkdir(parents=True, exist_ok=True)
            (cand_log / "plan.txt").write_text(plan)
            (cand_log / "solution.py").write_text(code)
            cand_log_dirs.append(cand_log)

        shutdown_reserve = max(0.0, float(os.getenv("AIDE_SHUTDOWN_RESERVE_S", "180")))
        executable_wall = self.deadline - time.time() - shutdown_reserve
        if executable_wall <= 0:
            raise TimeoutError("clean-shutdown reserve reached before best-of-N execution")
        exec_kwargs = dict(
            # Reserve wall time for SIGINT overtime and bounded process-tree cleanup.
            timeout=max(1, int(min(self.cfg.exec.timeout, executable_wall))),
            format_tb_ipython=bool(self.cfg.exec.format_tb_ipython),
            agent_file_name=str(self.cfg.exec.agent_file_name),
        )
        interps = [Interpreter(cand_dirs[i], **exec_kwargs) for i in range(n)]
        cost_recorder = getattr(self, "execution_cost_recorder", None)
        if cost_recorder is not None:
            stage = getattr(self, "_execution_cost_stage", None)
            for i, ip in enumerate(interps):
                ip.execution_cost_recorder = cost_recorder
                ip.execution_cost_context = {
                    "stage": stage,
                    "policy": cost_recorder.metadata.get("stage_policies", {}).get(stage),
                    "candidate_index": i,
                    "selection_step": self.current_step,
                    "triggers": {"real_oracle_pool": True},
                    "retry_index": 0,
                    "configured_timeout_s": self.cfg.exec.timeout,
                    "remaining_execution_s": executable_wall,
                    "path": "candidate_pool",
                }

        def run_one(i: int) -> ExecutionResult:
            return interps[i].run(candidates[i][1], reset_session=True)

        try:
            with ThreadPoolExecutor(max_workers=n, thread_name_prefix="bon-real") as pool:
                results: list[ExecutionResult] = list(pool.map(run_one, range(n)))
        finally:
            # Clean up concurrently so the grace period is paid once per step.
            with ThreadPoolExecutor(
                max_workers=n, thread_name_prefix="bon-cleanup"
            ) as cleanup_pool:
                list(
                    cleanup_pool.map(
                        lambda ip: (
                            cost_recorder.cleanup_attempt(ip)
                            if cost_recorder is not None
                            else ip.cleanup_session()
                        ),
                        interps,
                    )
                )

        csv_sizes: list[int] = []
        validations = []
        for i, res in enumerate(results):
            plan, original_code = candidates[i]
            code, validation, processing = self._process_real_submission(
                cand_dirs[i], res, original_code
            )
            candidates[i] = (plan, code)
            validations.append(validation)
            csv_path = cand_dirs[i] / "submission" / "submission.csv"
            # Generated code may create submission.csv as a directory.
            csv_sizes.append(csv_path.stat().st_size if csv_path.is_file() else 0)
            if res.exc_type is not None:
                reject_reason = f"exception:{res.exc_type}"
            elif csv_sizes[i] == 0:
                reject_reason = "no_csv_or_empty"
            else:
                reject_reason = None
            # Match journal serialization of raw write fragments.
            (cand_log_dirs[i] / "term_out.log").write_text("".join(res.term_out))
            audit = {
                "exc_type": res.exc_type,
                # The innermost runfile.py frame identifies the crash line.
                "crash_line": next(
                    (
                        ln
                        for fn, ln, *_ in reversed(res.exc_stack or [])
                        if fn.endswith("runfile.py")
                    ),
                    None,
                ),
                "exec_time": res.exec_time,
                "csv_size_bytes": csv_sizes[i],
                "accepted": False,
                "reject_reason": reject_reason,
                "submission_actions": processing["actions"],
                "submission_diagnostics": list(processing["diagnostics"]),
            }
            if validation is not None:
                audit["submission_validation"] = validation.status
                audit["submission_validation_message"] = validation.message
                audit["submission_diagnostics"].extend(validation.diagnostics)
                if validation.status == "invalid":
                    audit["reject_reason"] = "invalid_submission"
            (cand_log_dirs[i] / "result.json").write_text(json.dumps(audit))
            # Retain size-capped candidate submissions for offline grading.
            if 0 < csv_sizes[i] <= int(os.getenv("AIDE_BON_CSV_MAX_BYTES", "25000000")):
                shutil.copyfile(csv_path, cand_log_dirs[i] / "submission.csv")

        has_valid = any(v is not None and v.status == "valid" for v in validations)
        usable = [
            v is not None and (v.status == "valid" or (not has_valid and v.status == "unknown"))
            for v in validations
        ]
        return [
            CandidateOutcome(
                result=results[i],
                usable=usable[i],
                submission_exists="true" if csv_sizes[i] > 0 else "false",
                predicted=False,
                workspace=cand_dirs[i],
            )
            for i in range(n)
        ]

    def _review_candidates(self, candidates, outcomes, step_log_dir) -> None:
        """Attach optional advisory reviews without changing candidate usability.

        This costs one feedback call per candidate. Failed reviews remain None;
        predicted metrics are not observed scores. Use separate log anchors so
        offline extractors do not mistake candidate reviews for journal nodes."""
        if not getattr(getattr(self, "acfg", None), "bon_candidate_review", True):
            return
        if not outcomes:
            return

        def _one(item):
            i, ((_, code), outcome) = item
            try:
                return self._review(
                    code,
                    # Use the same output window as journal-node reviews.
                    trim_long_string(outcome.term_out),
                    predicted=outcome.predicted and not outcome.result.real_won,
                    # Candidate logs must not match journal-node extraction anchors.
                    prompt_log_prefix=f"[best-of-N candidate review] candidate {i} prompt",
                    subject=f"BoN candidate {i}",
                )
            except Exception as e:  # noqa: BLE001 — a review must not kill the step
                logger.warning(f"[best-of-N candidate review] candidate {i} failed: {e!r}")
                return None

        with ThreadPoolExecutor(
            max_workers=len(candidates), thread_name_prefix="bon-review"
        ) as pool:
            reviews = list(pool.map(_one, enumerate(zip(candidates, outcomes))))
        for i, (outcome, review) in enumerate(zip(outcomes, reviews)):
            outcome.review = review
            # Keep review evidence separate from the acceptance audit.
            cand_log = step_log_dir / f"cand_{i}"
            cand_log.mkdir(parents=True, exist_ok=True)
            (cand_log / "review.json").write_text(json.dumps(review or {}, default=str))
        logger.info(
            f"[best-of-{len(candidates)}] candidate reviews: "
            f"{sum(r is not None for r in reviews)}/{len(outcomes)} ok "
            f"(evidence={'predicted' if getattr(outcomes[0], 'predicted', False) else 'real'})"
        )

    def _select(
        self,
        candidates,
        outcomes,
        step_log_dir,
        selector=None,
        *,
        stage=None,
        parent_node=None,
    ) -> int | None:
        """Select a usable candidate, or salvage an entirely failed pool.

        Explicit selectors override global settings. In listwise mode, the
        predicted-metric flag changes only usable-pool selection; entirely failed
        pools always use the least-buggy judge. Legacy probability argmax retains
        its own numeric and line-fraction fallbacks.
        None delegates the candidate-zero fallback to the caller."""
        if selector == "ucb":
            return self._select_ucb(candidates, outcomes, step_log_dir, stage, parent_node)
        if selector == "p_clean_argmax" and stage is not None:
            return self._select_ucb(
                candidates,
                outcomes,
                step_log_dir,
                stage,
                parent_node,
                score_rule="p_clean",
            )
        usable = [i for i, o in enumerate(outcomes) if o.usable]
        metric_argmax = selector == "predicted_metric_argmax" or (
            selector is None and bool(getattr(self.acfg, "use_predicted_metric_argmax_pick", False))
        )
        observed_argmax = selector == "observed_metric_argmax" or (
            selector is None and bool(getattr(self.acfg, "use_observed_metric_argmax_pick", False))
        )
        if usable and observed_argmax:
            return self._select_observed_metric(candidates, outcomes, usable, step_log_dir)
        if usable and metric_argmax:
            return self._select_predicted_metric(candidates, outcomes, usable, step_log_dir)
        listwise = bool(self.acfg.use_best_of_n_listwise) or metric_argmax or observed_argmax
        # Stage-local selectors override the legacy global argmax flag.
        argmax = not (metric_argmax or observed_argmax) and (
            selector == "p_clean_argmax"
            if selector is not None
            else bool(getattr(self.acfg, "use_p_clean_argmax_pick", False))
        )
        if usable and (len(usable) < 2 or not (listwise or argmax)):
            return usable[0]
        if not usable and not (listwise or argmax):
            return None
        pool = usable or list(range(len(candidates)))
        cands_info = [
            {
                "idx": i,
                "code": candidates[i][1],
                "term_out": outcomes[i].term_out,
                "analysis": getattr(outcomes[i], "analysis", ""),
            }
            for i in pool
        ]
        # Only probability argmax needs posterior calls; listwise uses code/output.
        p_clean_fn = getattr(self, "_p_clean", None)
        score_name = "p_clean"
        if argmax and p_clean_fn is not None:
            for info in cands_info:
                scores = p_clean_fn(info["code"])
                if isinstance(scores, dict):
                    score_name = scores.get("score_name") or score_name
                    # Failed pools always rank on p_clean, not confidence in a failure.
                    rank = scores.get("p_clean") if not usable else scores.get("score")
                    info["rank_score"] = rank if rank is not None else scores.get("p_clean")
                else:
                    # Legacy scalar hook (tests, older wiring): p_clean only.
                    info["rank_score"] = scores
        line_frac = not usable and argmax and self.acfg.use_line_frac_salvage
        if line_frac:
            # This override belongs to legacy probability argmax, not listwise.
            picked, judge_raw = salvage_line_frac_pick(cands_info), "line_frac"
        elif argmax:
            if not any(c.get("rank_score") is not None for c in cands_info):
                # A scoring outage must not turn an argmax policy into a judge policy.
                logger.warning(
                    f"[best-of-{len(candidates)}] p_clean argmax: no candidate "
                    f"scored; falling back to "
                    f"{'first usable' if usable else 'caller fallback'} "
                    f"(no judge call)"
                )
                return usable[0] if usable else None
            picked, judge_raw = p_clean_argmax_pick(cands_info), "p_clean_argmax"
        else:
            picked, judge_raw = listwise_pick(
                cands_info,
                self.task_desc,
                self._judge_query,
                predicted=outcomes[0].predicted,
                all_failed=not usable,
            )
        # Rejected candidates never reach the journal; preserve the comparison here.
        audit = {"accepted": usable, "winner": picked, "judge": judge_raw}
        if argmax:
            # A string judge identifies a numeric rule to offline audit tools.
            # Record the effective score, which is always p_clean for failed pools.
            audit["score_name"] = score_name if usable else "p_clean"
            audit["scores"] = {c["idx"]: c.get("rank_score") for c in cands_info}
        (step_log_dir / "listwise.json").write_text(json.dumps(audit, default=str))
        rule = "p_clean argmax" if argmax else "list-wise"
        if usable:
            logger.info(
                f"[best-of-{len(candidates)}] {rule} pick: {picked} " f"among usable {usable}"
            )
        else:
            logger.info(
                f"[best-of-{len(candidates)}] all candidates unusable; salvage "
                f"pick ({'line_frac' if line_frac else rule}): "
                f"{picked}"
            )
        return picked

    def _judge_query(self, prompt) -> dict:
        """One feedback-LLM call with the listwise pick schema."""
        return cast(
            dict,
            self._query(
                system_message=prompt,
                user_message=None,
                func_spec=listwise_pick_func_spec,
                model=self.acfg.feedback.model,
                temperature=self.acfg.feedback.temp,
                convert_system_to_user=self.acfg.convert_system_to_user,
            ),
        )

    def _commit(self, candidates, outcomes, picked, step_log_dir):
        """Persist the selection and return its (plan, code, result) triple.

        Simulator-only WM selections reuse their prediction for node review.
        Other predicted candidates still need execution. Real candidates reuse
        their result and copy the selected CSV before workspace cleanup,
        including invalid salvage picks whose errors must reach the debug loop."""
        idx = picked if picked is not None else 0
        outcome = outcomes[idx]
        if outcome.predicted:
            self._write_bon_candidate_artifacts(
                step_log_dir, candidates, [o.result for o in outcomes], idx
            )
        (step_log_dir / "picked.txt").write_text(
            str(picked) if picked is not None else "fallback:0"
        )
        if picked is not None:
            logger.info(
                f"[best-of-{len(candidates)}] picked candidate {picked} "
                f"(exc_type={outcome.result.exc_type!r})"
            )
        else:
            logger.info(
                f"[best-of-{len(candidates)}] nothing usable and no salvage "
                f"judge; falling back to candidate 0"
            )
        if outcome.predicted:
            if (
                self.oracle == "sim"
                and getattr(getattr(self.cfg, "exec", None), "simulator", False)
                and getattr(self.cfg.exec, "simulator_only", False)
                and not outcome.result.real_won
                and not is_sim_internal_failure(outcome.result)
            ):
                # Keep selection and node review on the same WM prediction.
                # Real validation remains in run.py; abstentions and discarded
                # race workspaces still require the configured executor.
                logger.info(f"[best-of-{len(candidates)}] reusing candidate {idx} WM prediction")
                return (*candidates[idx], outcome.result)
            return (*candidates[idx], None)
        if outcome.workspace is not None:
            csv_path = outcome.workspace / "submission" / "submission.csv"
            canonical = self.cfg.workspace_dir / "submission" / "submission.csv"
            reset_submission_directory(self.cfg.workspace_dir)
            if csv_path.is_file() and csv_path.stat().st_size > 0:
                shutil.copy(csv_path, canonical)
        if outcome.usable:
            result_file = step_log_dir / f"cand_{idx}" / "result.json"
            audit = json.loads(result_file.read_text())
            audit["accepted"], audit["reject_reason"] = True, None
            result_file.write_text(json.dumps(audit))
        shutil.rmtree(self.cfg.workspace_dir / "bon", ignore_errors=True)
        return (*candidates[idx], outcome.result)

    def _new_step_log_dir(self):
        """Create log_dir/bon/step_{k}, adding an attempt suffix to avoid overwrites."""
        step_log_dir = self.cfg.log_dir / "bon" / f"step_{self.current_step}"
        attempt = 1
        while step_log_dir.exists():
            step_log_dir = self.cfg.log_dir / "bon" / f"step_{self.current_step}_{attempt}"
            attempt += 1
        # Create the directory before selectors write their audit files.
        step_log_dir.mkdir(parents=True)
        return step_log_dir

    def _write_bon_candidate_artifacts(
        self,
        step_log_dir,
        candidates: list[tuple[str, str]],
        sim_results: list[ExecutionResult],
        accepted_idx: int | None,
    ) -> None:
        """Persist predicted candidate evidence under step_log_dir/cand_{i}/.

        accepted_idx=None marks all candidates rejected because a fused program
        ran instead. Offline extractors rely on that distinction."""
        for i, ((plan, code), sim_res) in enumerate(zip(candidates, sim_results)):
            cand_log = step_log_dir / f"cand_{i}"
            cand_log.mkdir(parents=True, exist_ok=True)
            (cand_log / "plan.txt").write_text(plan)
            (cand_log / "solution.py").write_text(code)
            (cand_log / "predicted_term_out.log").write_text("".join(sim_res.term_out or []))
            crashed = (
                sim_res.exc_type is not None if sim_res.real_won else is_predicted_crash(sim_res)
            )
            submission_prediction = getattr(sim_res, "predicted_submission_exists", None)
            usable = _candidate_is_usable(sim_res)
            if accepted_idx is None:
                reject_reason = "not_executed:fused_program_ran_instead"
            elif i == accepted_idx:
                reject_reason = None
            elif crashed:
                prefix = "exception" if sim_res.real_won else "predicted_crash"
                reject_reason = f"{prefix}:{sim_res.exc_type}"
            elif sim_res.real_won and not usable:
                reject_reason = "no_observed_submission"
            elif submission_prediction in {"false", "unknown"}:
                reject_reason = f"predicted_submission:{submission_prediction}"
            else:
                reject_reason = None
            (cand_log / "result.json").write_text(
                json.dumps(
                    {
                        "exc_type": sim_res.exc_type,
                        "real_won": sim_res.real_won,
                        "observed_submission_exists": sim_res.observed_submission_exists,
                        "is_predicted_crash": crashed,
                        "predicted_submission_exists": submission_prediction,
                        "is_predicted_usable": usable,
                        "accepted": i == accepted_idx,
                        "reject_reason": reject_reason,
                    }
                )
            )

    def _review(
        self,
        code: str,
        term_out: str,
        *,
        predicted: bool = False,
        prompt_log_prefix: str = "Prompt for parsing execution results",
        subject: str = "Node",
    ) -> dict:
        """Review code/output and normalize and ground the returned metric.

        Shared by journal nodes and optional candidate reviews. predicted changes
        the evidence label only; committed nodes retain their existing wording.
        Only journal-node reviews may use the default prompt_log_prefix, which
        offline extractors treat as a node anchor."""
        opening = _prompt_opening(self.acfg.obfuscate)
        if predicted:
            introduction = opening + (
                "You have written code to solve this task. It has NOT been executed: "
                "a learned simulator predicted the terminal output of running it. "
                "You should evaluate that predicted output, determining if there were "
                "any bugs as well as reporting the empirical findings."
            )
        else:
            introduction = opening + (
                "You have written code to solve this task and now need to evaluate the output of the code execution. "
                "You should determine if there were any bugs as well as report the empirical findings."
            )
        # Place shared task/guidance first to enable prefix caching across reviews.
        prompt = {
            "Introduction": introduction,
            "Task description": self.task_desc,
        }
        if self.acfg.eval_metric:
            prompt["Evaluation metric"] = (
                f"The competition's official evaluation metric is **{self.acfg.eval_metric}** "
                f"(as defined in the Task description). Report the value of THIS metric from "
                f"the execution output; if the code reported a different metric or none, "
                f"leave the metric null."
            )
        prompt["Validation reliability"] = VALIDATION_REVIEW_GUIDANCE
        prompt["Implementation"] = wrap_code(code)
        prompt["Predicted execution output" if predicted else "Execution output"] = wrap_code(
            term_out, lang=""
        )
        logger.info(f"{prompt_log_prefix}: \n\n{json.dumps(prompt, indent=4)}\n\n")

        response = cast(
            dict,
            self._query(
                # Execution may use the shutdown reserve for its final review;
                # keep the last 30 seconds available for saving and cleanup.
                reserve_s=30.0,
                system_message=prompt,
                user_message=None,
                func_spec=review_func_spec,
                model=self.acfg.feedback.model,
                temperature=self.acfg.feedback.temp,
                convert_system_to_user=self.acfg.convert_system_to_user,
            ),
        )

        # Partial reviews use safe defaults; missing metrics route the node to debug.
        response = normalize_review_response(response)
        # Discard metrics that cannot be grounded in the shown output.
        if response["metric"] is not None and not metric_grounded_in_term_out(
            response["metric"], term_out
        ):
            logger.warning(
                f"{subject}: reviewer metric {response['metric']} not found in "
                f"execution output; discarding as ungrounded."
            )
            response["metric_error"] = (
                f"Reviewer validation metric {response['metric']} was not found in the execution output."
            )
            response["metric"] = None
        return response


def salvage_line_frac_pick(cands):
    """Pick the candidate that reached the largest fraction of its source code.

    Candidates contain idx/code/term_out. Missing runfile.py frames score zero;
    ties choose the first candidate. Keep this helper self-contained for AST
    extraction by offline evaluation tools."""
    import re

    crash_line_re = re.compile(r'File "runfile\.py", line (\d+)')
    best_frac, best_idx = -1.0, cands[0]["idx"]
    for c in cands:
        nloc = max((int(m) for m in crash_line_re.findall(c["term_out"])), default=0)
        frac = nloc / max(len(c["code"].splitlines()), 1)
        if frac > best_frac:
            best_frac, best_idx = frac, c["idx"]
    return best_idx


def p_clean_argmax_pick(cands):
    """Return the first candidate with the highest non-None rank_score.

    The caller supplies the ranking scalar and handles fully unscored pools.
    Keep this helper self-contained for offline AST-based consumers."""
    best_p, best_idx = float("-inf"), cands[0]["idx"]
    for c in cands:
        p = c.get("rank_score")
        if p is not None and float(p) > best_p:
            best_p, best_idx = float(p), c["idx"]
    return best_idx


def listwise_pick(
    cands,
    task_desc,
    query_func,
    term_out_tail=3000,
    predicted=False,
    all_failed=False,
    score_name="p_clean",
):
    """Compare code/output in one judge call; fall back to the first candidate.

    Include optional reviewer findings/metrics, but no confidence signals.
    score_name is ignored for compatibility with older callers. Keep this
    function self-contained for offline AST-based consumers.
    """
    fallback = cands[0]["idx"]
    try:
        indices = [c["idx"] for c in cands]
        evidence = (
            "Outputs are simulator predictions; the candidates have not been executed."
            if predicted
            else "Outputs are from real executions."
        )
        objective = (
            "All candidates failed. Select the least buggy candidate, closest to working."
            if all_failed
            else "Select the candidate most likely to achieve the best held-out test score."
        )
        prompt = {"Introduction": f"{evidence} {objective}", "Task description": task_desc}
        if not all_failed:
            prompt["metric reliability"] = (
                "Use the task's official metric and direction. Check the code for a real "
                "hold-out and the correct metric formula/scale. Discount scores only for "
                "concrete validation contamination or metric errors. Correct hand-written "
                "metrics and full-data refits after hold-out scoring are valid."
            )
        notes = []
        if any(c.get("metric") is not None for c in cands):
            notes.append("Reviewer metric is the validation value extracted from the output.")
        if any((c.get("analysis") or "").strip() for c in cands):
            notes.append("Reviewer findings summarize the code and output.")
        if notes:
            prompt["reviewer notes"] = " ".join(notes) + (
                " These are fallible readings of the same evidence; verify them yourself."
            )
        out_label = "Predicted execution output (tail)" if predicted else "Execution output (tail)"
        for c in cands:
            term = c["term_out"].rstrip()
            nl = term.rfind("\n")
            if term[nl + 1 :].startswith("Execution time: "):
                term = term[: max(nl, 0)]
            header = f"Reviewer metric = {c['metric']}\n\n" if c.get("metric") is not None else ""
            review = (c.get("analysis") or "").strip()
            findings = f"\n\nReviewer findings:\n{review}" if review else ""
            prompt[f"Candidate {c['idx']}"] = (
                f"{header}```python\n{c['code']}\n```\n\n"
                f"{out_label}:\n```\n{term[-term_out_tail:]}\n```{findings}"
            )
        prompt["Instructions"] = f"Pick exactly one candidate index from {indices}."
        resp = query_func(prompt)
        winner = resp["best_candidate"]
        if isinstance(winner, bool):
            return fallback, resp
        winner = int(winner)
        return (winner if winner in indices else fallback), resp
    except Exception as e:
        return fallback, f"{type(e).__name__}: {e}"
