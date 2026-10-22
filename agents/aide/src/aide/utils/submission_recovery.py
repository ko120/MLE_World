"""Retain and revalidate selected real submissions without another execution."""

import hashlib
import logging
import os
from pathlib import Path
import shutil
import tempfile
import time

from .artifacts import publish_real_solution
from .metric import strictly_improves
from .submission_validation import submission_sha256

logger = logging.getLogger("aide")
RETRY_DELAY_S = 60.0


def retain_pending_submission(agent, node):
    """Keep unknown CSVs outside the working directory cleared by each step."""
    if (
        not node.real_won or node.is_buggy or not node.csv_exists
        or node.submission_validation != "unknown"
    ):
        return
    workspace = Path(agent.cfg.workspace_dir)
    root = Path(getattr(agent.cfg, "log_dir", workspace)) / "pending_submissions"
    destination = root / node.id / "submission.csv"
    temporary = None
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as handle:
            temporary = Path(handle.name)
            with (workspace / "submission/submission.csv").open("rb") as source:
                shutil.copyfileobj(source, handle)
        digest = submission_sha256(temporary)
        if temporary.stat().st_size == 0 or (
            node.submission_sha256 and digest != node.submission_sha256
        ):
            raise ValueError("Submission changed before it could be retained")
        os.replace(temporary, destination)
        node.submission_sha256 = digest
        node.pending_submission = {
            "path": str(destination.resolve()),
            "sha256": digest,
            "code_sha256": hashlib.sha256(node.code.encode()).hexdigest(),
            "attempts": 0,
            "retry_after": time.time() + RETRY_DELAY_S,
        }
        logger.info("[validate-retry] retained CSV for node %s", node.id)
    except (OSError, ValueError) as exc:
        logger.warning("[validate-retry] cannot retain node %s: %s", node.id, exc)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def retry_pending_submissions(agent, *, timeout_s=195.0, reserve_s=30.0):
    """Retry due files and deferred source reviews within one bounded allowance.

    The journal owns the retry state, including across serialization. Unknown
    responses stay pending. Valid files must finish the normal source review
    before they can compete with confirmed results. No interpreter is called.
    """
    pending_nodes = [
        node for node in agent.journal.nodes
        if getattr(node, "pending_submission", None) and node.real_won and not node.is_buggy
        and node.metric is not None and node.metric.value is not None
    ]
    if not pending_nodes:
        return 0
    # Give the strongest retained candidate the first chance within the budget.
    pending_nodes.sort(
        key=lambda node: node.metric.value if node.metric.maximize else -node.metric.value,
        reverse=True,
    )
    budget = max(0.0, timeout_s)
    deadline = getattr(agent, "deadline", None)
    if deadline is not None:
        budget = min(budget, max(0.0, deadline - time.time() - reserve_s))
    end = time.monotonic() + budget
    resolved = 0
    for node in pending_nodes:
        remaining = end - time.monotonic()
        if remaining < 1.0:
            break
        pending = node.pending_submission
        if time.time() < pending["retry_after"]:
            continue
        pending["retry_after"] = time.time() + RETRY_DELAY_S
        path = Path(pending["path"])
        try:
            if (
                submission_sha256(path) != pending["sha256"]
                or hashlib.sha256(node.code.encode()).hexdigest() != pending["code_sha256"]
            ):
                raise ValueError("Retained CSV or candidate code changed; retry refused")
            incumbent = agent.journal.get_best_node(validated_only=True)
            if node.submission_validation != "valid":
                pending["attempts"] += 1
                validation = agent._validate_submission_checks(
                    path, timeout_s=min(75.0, max(0.0, end - time.monotonic()))
                )
                if validation.sha256 != pending["sha256"]:
                    raise ValueError("Validation verdict does not match the retained CSV")
                # Do not repeatedly append the same outage guidance to prompts.
                if (
                    node.submission_validation != validation.status
                    or node.submission_validation_message != validation.message
                ):
                    agent._apply_submission_validation(node, validation)
                if validation.status == "unknown":
                    continue
                node.analysis += "\nCSV validation retry resolved: " + validation.message
            if not node.is_buggy:
                remaining = end - time.monotonic()
                if remaining < 1.0:
                    # Keep the valid CSV pending until source review has time.
                    continue
                agent._check_data_leakage(node, timeout_s=remaining)
            node.pending_submission = None
            try:
                eligible = (
                    not node.is_buggy and node.submission_validation == "valid"
                    and strictly_improves(node.metric, incumbent.metric if incumbent is not None else None)
                )
                if eligible:
                    published = publish_real_solution(
                        Path(agent.cfg.workspace_dir), node, source_path=path
                    )
                    if not published:
                        node.pending_submission = pending
                        continue
            except Exception:
                node.pending_submission = pending
                raise
            resolved += 1
            recorder = getattr(agent, "execution_cost_recorder", None)
            if recorder is not None:
                try:
                    recorder.node_result(node, event="publication" if eligible else "validation_result")
                except Exception as exc:
                    # An audit failure cannot undo a completed publication or
                    # remove its metric from subsequent incumbent comparisons.
                    logger.warning("[validate-retry] result recording failed: %s", exc)
            logger.info("[validate-retry] resolved node %s: %s", node.id, node.submission_validation)
        except Exception as exc:
            pending["last_error"] = str(exc)
            logger.warning("[validate-retry] node %s remains pending: %s", node.id, exc)
    return resolved
