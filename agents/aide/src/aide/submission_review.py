"""Validate real artifacts, review leakage, and record evaluation-only scores.

Agent owns the workspace and node lifecycle. These methods attach diagnoses to
nodes and validate the exact CSV being considered for publication. Private grading
is instrumentation only: its scores never enter prompts or candidate selection.
"""

import json
import logging
import math
import os
import time
from dataclasses import asdict, replace
from pathlib import Path

from .data_leakage import check_validation
from .journal import Node
from .utils.artifacts import has_submission_csv as _has_submission_csv
from .utils.metric import WorstMetricValue
from .utils.response import trim_long_string
from .utils.submission_checks import (
    LOCAL_CHECK_SECONDS,
    ContentCheck,
    SubmissionChecks,
    submission_signature,
)
from .utils.submission_validation import (
    SubmissionValidation,
    SubmissionValidator,
    submission_sha256,
)
from .utils.validation_schema import PublicValidationSchema

logger = logging.getLogger("aide")


def _existing_submission_digest(csv_path):
    """A disappearing or inaccessible artifact cannot establish its identity."""
    try:
        path = Path(csv_path)
        if path.is_file() and path.stat().st_size > 0:
            return submission_sha256(path)
    except OSError:
        pass
    return None


class SubmissionReviewMixin:
    """Review artifacts using Agent configuration, task description, and journal.

    Lazily caches the public schema, validators, and content checks on the same
    Agent instance. Methods annotate nodes; publication remains with the caller.
    """

    @staticmethod
    def _record_failure(node, category, reason):
        # Bound prompt-facing diagnoses; node.exc_info retains the original exception.
        reason = trim_long_string(reason)
        # A shared diagnosis can route repairs to multiple categories, but should
        # appear only once in the prompt-facing analysis.
        already_recorded = any(reason in reasons for reasons in node.failure_reasons.values())
        reasons = node.failure_reasons.setdefault(category, [])
        if reason not in reasons:
            reasons.append(reason)
        if not already_recorded:
            node.analysis = trim_long_string(
                f"{node.analysis or ''}\nHarness {category} rejection: {reason}".strip()
            )

    def _process_real_submission(self, workspace, result, code):
        """Share validation across BoN, ordinary execution, and real reruns."""
        csv_path = Path(workspace) / "submission/submission.csv"
        processing = getattr(result, "submission_processing", None)
        if processing is not None:
            saved = processing.get("validation")
            digest = _existing_submission_digest(csv_path)
            if (
                digest is not None
                and digest == processing.get("sha256")
                and saved is not None
                and saved["status"] in {"valid", "invalid"}
            ):
                return (
                    code,
                    SubmissionValidation(**saved) if saved is not None else None,
                    processing,
                )
        validation = (
            self._validate_submission_checks(csv_path)
            if result.exc_type is None and _has_submission_csv(workspace)
            else None
        )
        processing = {
            "sha256": validation.sha256 if validation is not None else None,
            "validation": asdict(validation) if validation is not None else None,
            "actions": [],
            "diagnostics": [],
        }
        result.submission_processing = processing
        return code, validation, processing

    def _validate_submission(self, csv_path, *, timeout_s=75.0) -> SubmissionValidation:
        """Return format evidence; transport/grader failures remain unknown."""
        if not hasattr(self, "_submission_validator"):
            self._submission_validator = SubmissionValidator()
        deadline = getattr(self, "deadline", None)
        if deadline is not None:
            timeout_s = max(0.0, min(timeout_s, deadline - time.time() - 30.0))
        return self._submission_validator.validate(
            csv_path,
            url=os.getenv("AIDE_VALIDATE_SERVER_URL", "http://localhost:5000/validate"),
            competition_id=os.getenv("COMPETITION_ID", ""),
            timeout=timeout_s,
        )

    def _validate_submission_checks(self, csv_path, *, timeout_s=75.0) -> SubmissionValidation:
        """Combine one server verdict with a fast, read-only local prefix check."""
        csv_path = Path(csv_path)
        deadline = getattr(self, "deadline", None)
        if deadline is not None:
            timeout_s = max(0.0, min(timeout_s, deadline - time.time() - 30.0))
        end = time.monotonic() + timeout_s
        signature = submission_signature(csv_path)
        initial = self._validate_submission(csv_path, timeout_s=timeout_s)
        if not hasattr(self, "_submission_checks"):
            self._submission_checks = SubmissionChecks(
                self.cfg.workspace_dir / "input",
                getattr(self.acfg, "eval_metric", None),
                getattr(self, "task_desc", ""),
            )
        checks = self._submission_checks
        if not hasattr(self, "_submission_content_cache"):
            self._submission_content_cache = {}

        def check_content(path, digest, expected_signature):
            if expected_signature is None or submission_signature(path) != expected_signature:
                return ContentCheck(
                    diagnostics=[
                        "Local CSV check skipped: submission changed or became unavailable."
                    ]
                )
            cached = self._submission_content_cache.get(digest)
            if cached is not None and cached.cacheable:
                return cached
            local = checks.check(path, min(end, time.monotonic() + LOCAL_CHECK_SECONDS))
            # Retry incomplete scans while retaining errors proved for these exact bytes.
            if cached is not None:
                local.format_errors = list(
                    dict.fromkeys(cached.format_errors + local.format_errors)
                )
                local.content_errors = list(
                    dict.fromkeys(cached.content_errors + local.content_errors)
                )
            if (
                (local.cacheable or local.format_errors or local.content_errors)
                and digest
                and submission_signature(path) == expected_signature
            ):
                self._submission_content_cache[digest] = local
                if len(self._submission_content_cache) > 256:
                    self._submission_content_cache.pop(next(iter(self._submission_content_cache)))
            return local

        local = check_content(csv_path, initial.sha256, signature)
        if signature is not None and submission_signature(csv_path) != signature:
            return replace(
                initial,
                status="unknown",
                message="Submission changed during validation; local results were discarded.",
            )
        diagnostics = list(local.diagnostics)
        # A grader may accept a different schema; still reject invalid numeric contents.
        if initial.status == "valid":
            diagnostics.extend(
                f"Sample-schema diagnostic: {issue}" for issue in local.format_errors
            )
            if local.content_errors:
                return replace(
                    initial,
                    status="invalid",
                    message=" ".join(local.content_errors),
                    diagnostics=tuple(diagnostics),
                )
            return replace(initial, diagnostics=tuple(diagnostics))
        errors = local.format_errors + local.content_errors
        return replace(
            initial,
            status="invalid" if errors else initial.status,
            message=" ".join([initial.message, *errors]),
            diagnostics=tuple(diagnostics),
        )

    def _apply_submission_validation(self, node, validation):
        node.submission_validation = validation.status
        node.submission_validation_message = validation.message
        node.submission_sha256 = validation.sha256
        node.submission_columns = list(getattr(validation, "columns", ()))
        if validation.status == "invalid":
            node.is_buggy = True
            node.metric = WorstMetricValue()
            self._record_failure(
                node, "submission", f"Submission validation failed: {validation.message}"
            )
        elif validation.status == "unknown":
            logger.warning("[validate] node %s remains unverified: %s", node.id, validation.message)
            node.analysis += f"\nSubmission validation unverified: {validation.message}"
        if validation.diagnostics:
            node.analysis += "\n" + "\n".join(validation.diagnostics)
        # Both improve and debug prompts already include node.analysis. Keep
        # the follow-up at the end so it survives their head/tail truncation.
        if validation.status == "invalid":
            node.analysis += (
                "\n\nSubmission validation follow-up:\n"
                "The CSV was explicitly rejected. In the next improve or debug step, "
                "fix the specific submission errors reported above. Check required columns, "
                "test IDs and row coverage, and prediction types/values against the task's "
                "submission requirements and available public examples. Preserve all required "
                "predictions and change training or validation logic only if a separate "
                "diagnosis requires it. The revised CSV needs a successful format verdict "
                "before it can be published."
            )
        elif validation.status == "unknown":
            node.analysis += (
                "\n\nSubmission validation follow-up:\n"
                "No definitive CSV-format verdict was obtained. A timeout or unavailable "
                "validator does not establish a code or CSV-format bug, and a reported "
                "validation metric does not confirm that the CSV can be published. In the "
                "next improve or debug step, inspect the export against the task's submission "
                "requirements and available public examples; fix only demonstrated CSV "
                "mismatches and preserve all required rows, IDs, and predictions. Do not "
                "change training, the validation split, or the reported metric solely to "
                "address a validator timeout. If no export error is found, keep the format "
                "status unverified. For an otherwise usable real result, the harness retains "
                "the CSV and retries validation without another training execution."
            )

    def _check_data_leakage(self, node: Node, *, timeout_s: float | None = None) -> None:
        """Review validation and submission paths before a prospective publication."""
        node.leakage_check = None
        if not getattr(self.acfg, "check_data_leakage", False) or node.is_buggy is not False:
            return
        metric = node.metric
        if (
            metric is None
            or metric.value is None
            or not math.isfinite(metric.value)
            or metric.maximize is None
        ):
            return
        evidence = {
            "metric": metric.value,
            "maximize": metric.maximize,
            "source": "real" if node.real_won else "sim",
        }
        started = time.monotonic()
        try:
            triggers = []
            extreme = (
                metric.value >= self.acfg.leakage_max_threshold
                if metric.maximize
                else metric.value <= self.acfg.leakage_min_threshold
            )
            if extreme:
                triggers.append("extreme_score")
            if (
                node.real_won
                and node.submission_validation == "valid"
                and getattr(self.acfg, "leakage_review_best", True)
            ):
                previous = [
                    n.metric.value
                    for n in self.journal.nodes
                    if n is not node
                    and n.real_won
                    and n.is_buggy is False
                    and n.submission_validation == "valid"
                    and not getattr(n, "pending_submission", None)
                    and n.metric is not None
                    and n.metric.value is not None
                    and math.isfinite(n.metric.value)
                    and n.metric.maximize == metric.maximize
                ]
                if not previous or (
                    metric.value >= max(previous)
                    if metric.maximize
                    else metric.value <= min(previous)
                ):
                    # Publication accepts ties too; review every prospective replacement.
                    triggers.append("best_real_candidate")
            if not triggers:
                return
            allowance = getattr(self.acfg, "leakage_timeout_s", 120.0)
            if timeout_s is not None:
                allowance = min(allowance, max(0.0, timeout_s))
            deadline = getattr(self, "deadline", None)
            if deadline is not None:
                allowance = min(allowance, max(0.0, deadline - time.time()))
            review_submission = "best_real_candidate" in triggers
            public_schema = None
            workspace = getattr(getattr(self, "cfg", None), "workspace_dir", None)
            if review_submission and workspace is not None:
                input_dir = Path(workspace) / "input"
                cached = getattr(self, "_public_validation_schema", None)
                if cached is None or cached.input_dir != input_dir:
                    cached = self._public_validation_schema = PublicValidationSchema(input_dir)
                schema_started = time.monotonic()
                public_schema = cached.load(schema_started + max(0.0, allowance))
                allowance = max(0.0, allowance - (time.monotonic() - schema_started))
            verdict = check_validation(
                task_desc=self.task_desc,
                code=node.code,
                public_schema=public_schema,
                validated_submission=(
                    {
                        "status": "valid",
                        "sha256": node.submission_sha256,
                        "columns": getattr(node, "submission_columns", []),
                    }
                    if node.real_won
                    and node.submission_validation == "valid"
                    and node.submission_sha256
                    else None
                ),
                review_submission=review_submission,
                submission_reject_on_detection=getattr(
                    self.acfg, "submission_reject_on_model", True
                ),
                submission_temperature=getattr(self.acfg, "submission_review_temperature", 0.0),
                model=self.acfg.feedback.model,
                temperature=self.acfg.feedback.temp,
                eval_metric=self.acfg.eval_metric,
                convert_system_to_user=self.acfg.convert_system_to_user,
                timeout_s=allowance,
                max_tokens=getattr(self.acfg, "leakage_max_tokens", 3072),
                reject_on_detection=getattr(self.acfg, "leakage_reject_on_model", True),
            )
            verdict["triggers"] = triggers
            verdict["elapsed_s"] = time.monotonic() - started
        except Exception as exc:
            node.leakage_check = {
                "status": "unavailable",
                "decision": "advisory",
                "reason": str(exc),
                "elapsed_s": time.monotonic() - started,
                **evidence,
            }
            logger.warning("[leakage] node %s check unavailable: %s", node.id, exc)
            return
        node.leakage_check = {**verdict, **evidence}
        log = logger.warning if verdict["status"] == "unavailable" else logger.info
        log("[leakage] node %s: %s", node.id, json.dumps(node.leakage_check))
        if verdict["decision"] == "advisory" and verdict.get("has_submission_mismatch"):
            node.analysis += (
                "\nSUBMISSION PREDICTION REVIEW (ADVISORY)\n"
                f"{verdict['reason']}\n"
                "Check these source-level concerns in the next improvement; confirm the "
                "reported mismatch before changing a working prediction mapping."
            )
        if verdict["decision"] == "reject":
            node.is_buggy = True
            node.metric = WorstMetricValue()
            node.metric_trustworthy = False
            mismatch = verdict.get("has_submission_mismatch", False)
            leakage = verdict.get("has_leakage", not mismatch)
            defects = []
            if leakage:
                defects.append("validation contamination")
            if mismatch:
                defects.append("validation/submission prediction mismatch")
            diagnosis = (
                f"SOURCE REVIEW REJECTED: {', '.join(defects)}\n"
                f"{verdict['reason']}\n"
                f"Candidate's {'observed' if node.real_won else 'predicted'} validation metric: {metric.value}.\n"
                "Apply the targeted repair and recompute validation through the repaired "
                "prediction/export path before producing a new submission."
            )
            if leakage:
                self._record_failure(node, "metric", diagnosis)
            if mismatch:
                self._record_failure(node, "submission_pipeline", diagnosis)

    def _grade_test_submission(self) -> float | None:
        """Record a private test score for offline analysis; failures return None.

        The score never enters prompts or node selection. Local imports keep this
        method usable by standalone evaluation tools."""
        # Keep logging available even if a local import fails.
        import logging as _logging

        try:
            import os

            import requests

            url = os.environ.get("AIDE_GRADING_SERVER_URL", "http://localhost:5000/score")
            sub_path = self.cfg.workspace_dir / "submission" / "submission.csv"
            with open(sub_path, "rb") as f:
                resp = requests.post(
                    url,
                    files={"file": f},
                    timeout=90,  # > the server's 60s grade time-box
                )
            body = resp.json()
            score = body.get("score")
            if score is None:
                # Retain server failure reasons in the log without failing the step.
                _logging.getLogger("aide").warning(
                    f"[grading] no test_metric for {sub_path}: "
                    f"{body.get('error') or 'server reported no error'}"
                )
            return float(score) if score is not None else None
        except Exception as e:
            _logging.getLogger("aide").warning(
                f"[grading] /score call failed: {type(e).__name__}: {e}"
            )
            return None
