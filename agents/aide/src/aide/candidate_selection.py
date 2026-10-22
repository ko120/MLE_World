"""Metric, uncertainty, pairwise, and fusion policies for candidate pools.

Agent owns the journal and execution lifecycle. These methods consume candidate
evidence, call its budget-aware query/scoring hooks, and persist selection audits.
They never publish a submission or use private test scores for selection.
"""

import inspect
import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any

from . import metric_posterior, preference, ucb
from .fusion import (
    FusionCandidate,
    FusionContext,
    FusionParent,
    FusionPromptConfig,
    FusionReport,
    build_fusion_prompt,
    parse_fusion_response,
)
from .journal import Node
from .utils.response import trim_long_string

if TYPE_CHECKING:
    from .agent import CandidateOutcome

logger = logging.getLogger("aide")
STAGES = ("draft", "debug", "improve")
# Match candidate and parent output windows without changing the prompt content.
_AIDE_TRIM_K: int = inspect.signature(trim_long_string).parameters["k"].default


class CandidateSelectionMixin:
    """Rank or fuse candidates using state and callbacks initialized by Agent.

    Reads configuration, journal, task, oracle, and remaining budget; delegates
    model calls to _query and fallback selection to _select. Stores selection
    evidence in _selected_exc_scores and _selected_ucb for the committed node.
    """

    def _select_observed_metric(self, candidates, outcomes, usable, step_log_dir):
        """Select real validation evidence; neither WM nor private-test scores qualify."""
        lower = getattr(self.acfg, "eval_lower_is_better", None)
        metrics = {}
        for i in usable:
            outcome = outcomes[i]
            review = outcome.review or {}
            if (
                outcome.predicted
                or review.get("is_bug") is True
                or review.get("metric_trustworthy") is False
            ):
                continue
            value = ucb.finite_number(review.get("metric"))
            if value is not None:
                metrics[i] = value
        fallback = None
        if metrics and isinstance(lower, bool):
            picked = (min if lower else max)(metrics, key=metrics.get)
        else:
            picked = usable[0]
            fallback = (
                "unknown_metric_direction"
                if not isinstance(lower, bool)
                else "no_finite_observed_metric"
            )
        audit = {
            "accepted": usable,
            "winner": picked,
            "judge": "observed_metric_argmax",
            "score_name": "observed_validation_metric",
            "scores": metrics,
            "metric_source": "real_candidate_review",
            "lower_is_better": lower,
            "fallback_reason": fallback,
        }
        (step_log_dir / "listwise.json").write_text(json.dumps(audit, indent=2, allow_nan=False))
        logger.info(
            "[best-of-%s] observed metric pick: %s, lower_is_better=%s, fallback=%s",
            len(candidates),
            picked,
            lower,
            fallback,
        )
        return picked

    @staticmethod
    def _predicted_metrics(outcomes, usable):
        """The BoN pool: usable predictions with finite structured WM metrics."""
        metrics = {}
        for i in usable:
            outcome = outcomes[i]
            if outcome.predicted and not outcome.result.real_won:
                value = ucb.finite_number(getattr(outcome.result, "predicted_metric", None))
                if value is not None:
                    metrics[i] = value
        return metrics

    def _select_predicted_metric(self, candidates, outcomes, usable, step_log_dir):
        """Choose a WM metric from a nonempty usable pool without a judge call."""
        lower = getattr(self.acfg, "eval_lower_is_better", None)
        metrics = self._predicted_metrics(outcomes, usable)
        fallback = None
        rule = "predicted_metric_argmax"
        if metrics and isinstance(lower, bool):
            # min/max retain input order on ties, including valid zero scores.
            picked = (min if lower else max)(metrics, key=metrics.get)
        else:
            picked = usable[0]
            fallback = (
                "unknown_metric_direction"
                if not isinstance(lower, bool)
                else "no_finite_predicted_metric"
            )
        audit = {
            "accepted": usable,
            "winner": picked,
            "judge": rule,
            "score_name": "predicted_metric",
            "scores": metrics,
            "lower_is_better": lower,
            "fallback_reason": fallback,
        }
        (step_log_dir / "listwise.json").write_text(json.dumps(audit, indent=2, allow_nan=False))
        logger.info(
            "[best-of-%s] predicted metric pick: %s, lower_is_better=%s, fallback=%s",
            len(candidates),
            picked,
            lower,
            fallback,
        )
        return picked

    def _select_ucb(
        self,
        candidates,
        outcomes,
        step_log_dir,
        stage,
        parent_node,
        *,
        score_rule="ucb",
    ):
        """Select with UCB; optionally reuse BoN's pool and fallback behavior."""
        options = ucb.resolve_config(getattr(self.acfg, "ucb", None), stage)
        usable = [i for i, outcome in enumerate(outcomes) if outcome.usable]
        bon_pool = options.pool_policy == "bon" and score_rule == "ucb"
        bon_metrics = self._predicted_metrics(outcomes, usable) if bon_pool else {}
        if (not usable and (bon_pool or options.all_failed_fallback == "judge")) or (
            bon_pool
            and (
                (options.beta == 0 and options.selection_rule == "ucb")
                or not bon_metrics
                or not isinstance(getattr(self.acfg, "eval_lower_is_better", None), bool)
            )
        ):
            # Reuse the exact BoN judge/first-usable/metric-greedy path. In
            # particular, beta=0 must reproduce BoN even at numerical extremes.
            return self._select(
                candidates, outcomes, step_log_dir, selector="predicted_metric_argmax"
            )
        maximize, reference, reference_source = self._ucb_context(parent_node)
        pool = []
        metric_sources = []
        for i, outcome in enumerate(outcomes):
            observed = not outcome.predicted or bool(getattr(outcome.result, "real_won", False))
            review = outcome.review or {}
            metric = (
                None
                if observed
                else ucb.finite_number(getattr(outcome.result, "predicted_metric", None))
            )
            if bon_pool:
                metric = bon_metrics.get(i)
            source = "world_model" if metric is not None else "missing"
            if (
                not bon_pool
                and metric is None
                and (
                    observed or getattr(outcome.result, "predicted_submission_exists", None) is None
                )
            ):
                metric = ucb.finite_number(review.get("metric"))
                if metric is not None:
                    source = "real_review" if observed else "predicted_review"
            if not bon_pool and (
                review.get("is_bug") is True or review.get("metric_trustworthy") is False
            ):
                metric, source = None, "review_rejected"
            pool.append(ucb.Candidate(i, outcome.usable, metric, observed))
            metric_sources.append(source)

        eligible = [c for c in pool if c.usable and c.metric is not None]
        use_metric_posterior = options.metric_posterior.enabled and score_rule == "ucb"
        if eligible and maximize is not None:
            needs_confidence = (
                (options.beta > 0 and not use_metric_posterior)
                or score_rule == "p_clean"
                or options.selection_rule == "expected_gain"
            )
            compared = eligible if needs_confidence and len(eligible) > 1 else []
        else:
            compared = [c for c in pool if c.usable] or pool
        to_score = [c for c in compared if not c.observed]
        scorer = getattr(self, "_exc_scores", None)
        start = time.monotonic()
        if scorer is not None and to_score:
            # Bound scoring concurrency separately from candidate width.
            def score(candidate):
                try:
                    return scorer(candidates[candidate.idx][1])
                except Exception as exc:  # a failed scorer cannot kill the step
                    logger.warning(f"[ucb] candidate {candidate.idx} scoring failed: {exc!r}")
                    return None

            with ThreadPoolExecutor(
                max_workers=min(options.score_workers, len(to_score)),
                thread_name_prefix="bon-ucb",
            ) as executor:
                scores = list(executor.map(score, to_score))
            for candidate, posterior in zip(to_score, scores):
                candidate.scores = posterior

        if (
            use_metric_posterior
            and options.beta > 0
            and len(eligible) > 1
            and isinstance(maximize, bool)
        ):
            to_sample = [c for c in eligible if not c.observed]
            sampler = getattr(self, "_metric_posterior", None)
            budget = options.metric_posterior.timeout_s
            if getattr(self, "deadline", None) is not None:
                reserve = max(0.0, float(os.getenv("AIDE_SHUTDOWN_RESERVE_S", "180")))
                budget = min(budget, max(0.0, self.deadline - time.time() - reserve))
            sampling_deadline = time.monotonic() + budget

            def sample(candidate):
                if time.monotonic() >= sampling_deadline:
                    return metric_posterior.unavailable(
                        options.metric_posterior, "deadline_exhausted"
                    )
                if sampler is None:
                    return metric_posterior.unavailable(options.metric_posterior, "missing_scorer")
                try:
                    return sampler(
                        candidates[candidate.idx][1],
                        config=options.metric_posterior,
                        deadline=sampling_deadline,
                    )
                except Exception as exc:
                    logger.warning(
                        f"[ucb] candidate {candidate.idx} metric sampling failed: {type(exc).__name__}"
                    )
                    return metric_posterior.unavailable(options.metric_posterior, "request_error")

            if to_sample:
                with ThreadPoolExecutor(
                    max_workers=min(options.score_workers, len(to_sample)),
                    thread_name_prefix="bon-metric-posterior",
                ) as executor:
                    posteriors = list(executor.map(sample, to_sample))
                for candidate, posterior in zip(to_sample, posteriors):
                    candidate.metric_posterior = posterior

        picked, audit = ucb.select(
            pool,
            options,
            maximize=maximize,
            reference_metric=reference,
            score_rule=score_rule,
        )
        if bon_pool and audit["fallback_reason"] == "nonfinite_utility":
            return self._select(
                candidates, outcomes, step_log_dir, selector="predicted_metric_argmax"
            )
        # Retain the actual winner's posterior, including a failed scoring call.
        # Greedy and single-eligible pools may not have scored their winner yet.
        selected = pool[picked if picked is not None else 0]
        self._selected_exc_scores = (
            (candidates[selected.idx][1], selected.scores) if selected in to_score else None
        )
        audit.update(
            stage=stage,
            oracle=self.oracle,
            reference_source=reference_source,
            score_seconds=time.monotonic() - start,
            uncertainty_kind=(
                "sampled_validation_metric_std"
                if use_metric_posterior
                else "exception_probability_proxy"
            ),
        )
        self._selected_ucb = (
            candidates[selected.idx][1],
            {
                "stage": stage,
                "rule": audit["rule"],
                "winner": picked,
                "greedy_winner": audit["greedy_winner"],
                "beta": options.beta,
            },
        )
        for row, source in zip(audit["candidates"], metric_sources):
            row["metric_source"] = source
        (step_log_dir / "ucb.json").write_text(json.dumps(audit, indent=2, allow_nan=False))
        logger.info(
            f"[ucb] {stage}: winner={picked}, greedy={audit['greedy_winner']}, "
            f"sigma={'metric_posterior' if use_metric_posterior else options.sigma}, "
            f"beta={options.beta:g}, rule={audit['rule']}, "
            f"fallback={audit['fallback_reason']}"
        )
        return picked

    def _ucb_context(self, parent_node):
        """Prefer the best validated real metric, then the parent's metric."""
        lower = getattr(self.acfg, "eval_lower_is_better", None)
        maximize = not lower if isinstance(lower, bool) else None
        nodes = []
        journal = getattr(self, "journal", None)
        if journal is not None:
            nodes.append(("best_real", journal.get_best_node(validated_only=True)))
        nodes.append(("parent", parent_node))
        for source, node in nodes:
            if node is None or node.is_buggy:
                continue
            metric = getattr(node, "metric", None)
            value = ucb.finite_number(getattr(metric, "value", None))
            direction = getattr(metric, "maximize", None)
            if value is None or not isinstance(direction, bool):
                continue
            if maximize is None:
                maximize = direction
            if direction == maximize:
                return maximize, value, source
        return maximize, None, "pool"

    def _pairwise_select(
        self,
        stage: str,
        candidates: list[tuple[str, str]],
        outcomes: "list[CandidateOutcome]",
        step_log_dir,
    ) -> int:
        """Run aide.preference's pairwise tournament and return the selected index.

        Optional execution reports are labeled by oracle provenance. _commit
        owns the selection artifacts; this method records the comparison audit."""
        variant = self._pairwise_variant()
        context = "\n\n".join(
            f"Previous solution {node.id}:\n"
            f"Plan: {node.plan or '(no plan recorded)'}\n"
            f"Results: {node.analysis or '(no analysis recorded)'}\n"
            f"Validation metric ({'real execution' if node.real_won else 'world-model prediction'}): "
            f"{node.metric.value}"
            for node in self.journal.good_nodes
            if node.metric is not None and node.metric.value is not None
        )
        case = {
            "case_id": f"{self.cfg.exp_name}:{self.current_step}",
            "task_id": getattr(self.cfg, "task_name", "") or "",
            "stage": stage,
            "evaluation_metric": self.acfg.eval_metric,
            "lower_is_better": self.acfg.eval_lower_is_better,
            "original_prompt": {"Task description": self.task_desc},
            "context_text": context,
            "candidates": [
                {"index": i, "plan": plan, "code": code}
                for i, (plan, code) in enumerate(candidates)
            ],
        }
        reports = None
        if bool(self.acfg.pairwise_include_wm_report):
            reports = {
                i: {
                    "submission_exists": outcome.submission_exists,
                    "observed_output": outcome.term_out,
                }
                for i, outcome in enumerate(outcomes)
            }

        model = (
            self.acfg.code.model
            if str(self.acfg.pairwise_model) == "code"
            else self.acfg.feedback.model
        )
        verdicts: dict[str, dict] = {}
        outcome_summary = None
        while True:
            pending, outcome_summary = preference.plan_calls(case, variant, verdicts)
            if not pending:
                break
            for spec in pending:
                text = preference.build_preference_prompt(
                    case,
                    spec,
                    variant,
                    reports,
                    reports_are_predictions=outcomes[0].predicted,
                )
                try:
                    raw = self._query(
                        system_message=text,
                        user_message=None,
                        model=model,
                        temperature=self.acfg.feedback.temp,
                        max_tokens=int(self.acfg.pairwise_max_tokens),
                        convert_system_to_user=self.acfg.convert_system_to_user,
                    )
                except Exception as exc:  # noqa: BLE001 — an abstention, not a crash
                    logger.warning(f"[pairwise] {spec['call_name']} failed: {exc}")
                    raw = ""
                verdicts[spec["call_name"]] = preference.parse_preference_response(str(raw or ""))

        selected = int(outcome_summary["selected"])
        (step_log_dir / "pairwise.json").write_text(
            json.dumps(
                {
                    "stage": stage,
                    "model": model,
                    "prompt_version": preference.PROMPT_VERSION,
                    "with_wm_report": reports is not None,
                    **{k: v for k, v in outcome_summary.items() if k != "pairs"},
                    "verdicts": [
                        {**spec, **verdicts.get(spec["call_name"], {})}
                        for spec in (outcome_summary.get("pairs") or [])
                    ],
                },
                default=str,
            )
        )
        logger.info(
            f"[pairwise] {stage}: selected candidate {selected} from "
            f"{len(outcome_summary.get('pairs') or [])} comparisons "
            f"(basis={outcome_summary['basis']})"
        )
        return selected

    def _pairwise_variant(self) -> dict:
        """The `aide.preference` variant block for this arm's config."""
        return {
            "id": f"online_pairwise_{self.acfg.pairwise_scheme}",
            "limits": {"max_model_calls": int(self.acfg.pairwise_max_calls)},
            "foreagent": {
                "pair_scheme": str(self.acfg.pairwise_scheme),
            },
        }

    def _fuse_prompt_settings(self, stage):
        styles = getattr(self.acfg, "fuse_prompt_style_by_stage", {}) or {}
        params = getattr(self.acfg, "fuse_style_params_by_stage", {}) or {}
        unknown = (set(styles) | set(params)) - set(STAGES)
        if unknown:
            raise ValueError(f"unknown fusion stage override(s): {sorted(unknown)}")
        return (
            str(styles.get(stage, self.acfg.fuse_prompt_style)),
            tuple(sorted((str(k), str(v)) for k, v in (params.get(stage) or {}).items())),
        )

    def _fuse_candidates(
        self,
        prompt: Any,
        stage: str,
        parent_node: "Node | None",
        candidates: list[tuple[str, str]],
        outcomes: "list[CandidateOutcome]",
        picked_idx: int | None,
        step_log_dir,
    ) -> tuple[str, str] | None:
        """Build a fusion prompt and generate one program, retrying parse failures.

        Prompt/transport failure or insufficient candidate code declines fusion.
        Otherwise return the parsed or best-effort program for execution without
        simulator screening. Persist the prompt identity and generation audit."""
        mode = str(self.acfg.fuse_mode)
        source = "wm" if outcomes[0].predicted else "real"
        if mode != "synth" and picked_idx is None:
            # Editing modes require a selected backbone.
            logger.info(f"[fuse] no pick to anchor {mode} mode; skipping fusion")
            return None

        try:
            # Only code, oracle reports, and parent evidence reach the fuser.
            fusion_candidates = [
                FusionCandidate(
                    index=i,
                    code=code,
                    # Keep plan/analysis/code/output aligned with the parent layout.
                    plan=plan,
                    analysis=getattr(outcome, "analysis", ""),
                    report=FusionReport(
                        source=source,
                        submission_exists=outcome.submission_exists,
                        observed_output=outcome.term_out,
                    ),
                )
                for i, ((plan, code), outcome) in enumerate(zip(candidates, outcomes))
                # Skip empty code rather than invalidating the entire fusion prompt.
                if code.strip()
            ]
            if len(fusion_candidates) < 2:
                logger.info(
                    f"[fuse] only {len(fusion_candidates)} candidate(s) with code; "
                    "nothing to fuse"
                )
                return None

            parent = None
            if stage != "draft" and parent_node is not None:
                # Some fusion families require an explicit parent outside the request.
                parent = FusionParent(
                    plan=parent_node.plan or "",
                    analysis=parent_node.analysis or "",
                    code=parent_node.code or "",
                    execution_output=parent_node.term_out or "",
                    source="real" if parent_node.real_won else "wm",
                )
            context = FusionContext(
                stage=stage,
                original_prompt=prompt,
                # Use the configured metric and direction, shared with offline fusion.
                evaluation_metric=self.acfg.eval_metric,
                lower_is_better=self.acfg.eval_lower_is_better,
                parent=parent,
            )
            prompt_style, style_params = self._fuse_prompt_settings(stage)
            config = FusionPromptConfig(
                mode=mode,
                report_source=source,
                # Preserve generation order; reports carry candidate quality.
                candidate_order="input",
                backbone_index=picked_idx if mode != "synth" else None,
                include_candidate_plans=bool(self.acfg.fuse_include_candidate_plans),
                include_candidate_code=True,
                prompt_style=prompt_style,
                style_params=style_params,
                promote_parent=True,
                # An empty allowlist omits the embedded task request.
                request_sections=(),
                collapse_progress_output=True,
                # Window candidate output exactly like the parent output.
                report_head_chars=_AIDE_TRIM_K,
                report_tail_chars=_AIDE_TRIM_K,
            )
            built = build_fusion_prompt(context, fusion_candidates, config)
        except (ValueError, TypeError) as exc:
            # Invalid or oversized prompts decline fusion for this step.
            logger.warning(f"[fuse] prompt build failed ({exc}); no fusion this step")
            return None

        # Transport failures consume retries; unparseable replies keep best-effort code.
        retries = max(1, int(self.acfg.fuse_retries))
        artifact = None
        for attempt in range(1, retries + 1):
            try:
                raw = self._query(
                    system_message=built.text,
                    user_message=None,
                    model=self.acfg.code.model,
                    temperature=self.acfg.code.temp,
                    max_tokens=int(self.acfg.fuse_max_tokens),
                    convert_system_to_user=self.acfg.convert_system_to_user,
                )
                candidate_artifact = parse_fusion_response(
                    raw if isinstance(raw, str) else str(raw)
                )
            except Exception as exc:  # noqa: BLE001 - never kill a step
                logger.warning(f"[fuse] generation attempt {attempt}/{retries} failed ({exc})")
                continue
            artifact = candidate_artifact
            if artifact.valid_python:
                break
            logger.info(
                f"[fuse] attempt {attempt}/{retries} produced no valid Python "
                f"(status={artifact.status}); retrying"
            )
        if artifact is None:
            # Every attempt died in transport; there is nothing to execute.
            logger.warning(f"[fuse] all {retries} generation attempts failed")
            return None
        attempts_used = attempt

        fused_log = step_log_dir / "cand_fused"
        fused_log.mkdir(parents=True, exist_ok=True)
        (fused_log / "plan.txt").write_text(artifact.plan)
        (fused_log / "solution.py").write_text(artifact.code)
        (fused_log / "result.json").write_text(
            json.dumps(
                {
                    "parse_status": artifact.status,
                    "valid_python": artifact.valid_python,
                    "attempts": attempts_used,
                    # The fused program goes directly to execution.
                    "accepted": True,
                    "reject_reason": None,
                }
            )
        )
        (step_log_dir / "fuse.json").write_text(
            json.dumps(
                {
                    "stage": stage,
                    "mode": mode,
                    "prompt_style": config.prompt_style,
                    "style_params": dict(config.style_params),
                    "parent_source": parent.source if parent is not None else None,
                    "report_source": config.report_source,
                    "prompt_version": built.version,
                    "prompt_sha256": built.prompt_sha256,
                    "prompt_chars": built.prompt_chars,
                    "input_candidate_indices": list(built.displayed_candidate_indices),
                    "backbone_display_name": built.backbone_display_name,
                    "parse_status": artifact.status,
                    "valid_python": artifact.valid_python,
                    "code_sha256": artifact.code_sha256,
                    "attempts": attempts_used,
                    "executed": True,
                },
                default=str,
            )
        )

        if not artifact.valid_python:
            logger.warning(
                f"[fuse] no valid Python after {attempts_used} attempt(s) "
                f"(status={artifact.status}); executing the best-effort code — "
                f"real exec will report the true error"
            )
        logger.info(
            f"[fuse] executing fused program ({stage}/{mode}, "
            f"{built.prompt_chars} prompt chars, {attempts_used} attempt(s), "
            f"from candidates {list(built.displayed_candidate_indices)})"
        )
        return artifact.plan, artifact.code
