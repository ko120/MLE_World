"""Small, append-only execution records; no model calls or artifact copies.

An attempt measures Interpreter.run, including its internal termination work.
Explicit recovery cleanup after a raised interpreter error is an additional
event linked to that attempt. Offline accounting adds the two exactly once.
Writes are flushed by closing each append, without an fsync per event. A hard
kill can leave an unmatched start or a partial final line; neither is zero cost.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import socket
import threading
import time
import uuid
from pathlib import Path

from aide.execution_gate import (outcome_gate_enabled, outcome_gate_threshold,
                                 random_gate_enabled, random_gate_metadata)

logger = logging.getLogger("aide")


def _finite(value):
    return value if isinstance(value, (int, float)) and math.isfinite(value) else None


class ExecutionCostRecorder:
    def __init__(self, log_dir, metadata=None, *, run_start=None):
        self.path = Path(log_dir) / "real_execution_events.jsonl"
        self.metadata = dict(metadata or {})
        self.session_id = uuid.uuid4().hex
        self.run_start = time.time() if run_start is None else run_start
        self.cumulative_real_s = 0.0
        self.write_seconds = 0.0
        self.enabled = True
        self._directory_ready = False
        self._lock = threading.RLock()
        self._node_attempts = {}
        self.emit("run_start", **self.metadata)

    @classmethod
    def from_config(cls, cfg, *, run_start):
        stages = {s: str(getattr(cfg.agent.stage_policy, s)) for s in ("draft", "debug", "improve")}
        policies = set(stages.values())
        method = next(iter(policies)) if len(policies) == 1 else "/".join(stages.values())
        if not cfg.exec.simulator and method == "single":
            method = "base"
        elif method == "parallel":
            method = "bon_metric_argmax" if cfg.agent.use_predicted_metric_argmax_pick else "bon"
        elif method == "sequential":
            method = "seq"
        threshold = cfg.agent.real_validation_p_max_threshold
        mode = "real" if not cfg.exec.simulator else ("confidence" if threshold is not None else "best_only")
        universal_gate = outcome_gate_enabled(cfg.agent)
        random_gate = random_gate_enabled(cfg.agent)
        outcome_thresholds = None
        if universal_gate:
            outcome_thresholds = {
                "success": outcome_gate_threshold(cfg.agent, None),
                "failure": outcome_gate_threshold(cfg.agent, "failure"),
            }
            if cfg.exec.simulator:
                mode = "outcome_confidence"
        if random_gate:
            mode = "random_check"
            outcome_thresholds = None
        if cfg.exec.simulator and cfg.agent.real_validation_interval > 0:
            mode += "+periodic"
        return cls(cfg.log_dir, {
            "trial_id": os.getenv("AIDE_RUN_TOKEN") or str(Path(cfg.log_dir).resolve()),
            "task": os.getenv("COMPETITION_ID"), "seed": os.getenv("AIDE_RUN_SEED"),
            "method": method, "execution_mode": mode, "stage_policies": stages,
            "threshold": None if universal_gate else threshold,
            "confidence_stages": "all" if universal_gate else cfg.agent.real_validation_p_max_stages,
            "outcome_thresholds": outcome_thresholds,
            "gate_score_type": "uniform_random" if random_gate else "raw_p_max" if universal_gate else None,
            **({**random_gate_metadata(cfg.agent), "gate_mode": "random"} if random_gate else {}),
            "configured_timeout_s": cfg.exec.timeout, "time_limit_s": cfg.agent.time_limit,
            "hostname": socket.gethostname(), "slurm_job_id": os.getenv("SLURM_JOB_ID"),
            "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES"),
            "generator_model": cfg.agent.code.model, "feedback_model": cfg.agent.feedback.model,
            "wm_model": cfg.exec.simulator_model,
            "artifact_snapshots": False,
        }, run_start=run_start)

    def emit(self, event, **fields):
        if not self.enabled:
            return
        started = time.perf_counter()
        try:
            with self._lock:
                now = time.time()
                row = {
                    "schema_version": 1, "session_id": self.session_id, "event": event,
                    "timestamp": now, "trial_wall_s": max(0.0, now - self.run_start),
                    "cumulative_real_s": self.cumulative_real_s, **fields,
                }
                if not self._directory_ready:
                    self.path.parent.mkdir(parents=True, exist_ok=True)
                    self._directory_ready = True
                with self.path.open("a", encoding="utf-8") as output:
                    output.write(json.dumps(row, allow_nan=False, separators=(",", ":")) + "\n")
        except Exception:
            # Measurement must not turn a completed execution into a failed step.
            self.enabled = False
            logger.warning("[execution-cost] recording disabled after write failure", exc_info=True)
        finally:
            self.write_seconds += time.perf_counter() - started

    def node_fields(self, node):
        stage = node.stage_name
        return {
            "node_id": node.id, "stage": stage,
            "policy": self.metadata.get("stage_policies", {}).get(stage),
            "code_sha256": hashlib.sha256(node.code.encode()).hexdigest(),
        }

    def node_result(self, node, event="node_committed", **fields):
        self.emit(event, **self.node_fields(node), real_won=bool(node.real_won),
                  is_buggy=node.is_buggy, validation_status=node.submission_validation,
                  attempt_id=self._node_attempts.get(node.id) if node.real_won else None,
                  submission_sha256=node.submission_sha256,
                  validation_metric=_finite(getattr(node.metric, "value", None)),
                  test_metric=_finite(node.test_metric), **fields)

    def decision(self, node, *, triggers, skip_reason=None, **fields):
        threshold = fields.pop("threshold", self.metadata.get("threshold"))
        self.emit("execution_decision", **self.node_fields(node), triggers=triggers,
                  skip_reason=skip_reason, confidence=_finite(node.sim_p_max),
                  threshold=threshold, predicted_exception=node.exc_type,
                  **fields)

    def run_attempt(self, interpreter, code, execute):
        context = dict(getattr(interpreter, "execution_cost_context", {}))
        context.setdefault("code_sha256", hashlib.sha256(code.encode()).hexdigest())
        attempt_id = uuid.uuid4().hex
        interpreter.execution_cost_attempt_id = attempt_id
        if context.get("node_id"):
            self._node_attempts[context["node_id"]] = attempt_id
        self.emit("attempt_start", attempt_id=attempt_id, provenance="real",
                  enforced_timeout_s=interpreter.timeout, **context)
        started = time.perf_counter()
        result = None
        error = None
        try:
            result = execute()
            return result
        except BaseException as exc:
            error = type(exc).__name__
            raise
        finally:
            elapsed = time.perf_counter() - started
            with self._lock:
                self.cumulative_real_s += elapsed
                exc_type = getattr(result, "exc_type", None)
                self.emit("attempt_end", attempt_id=attempt_id, elapsed_s=elapsed,
                          reported_exec_s=_finite(getattr(result, "exec_time", None)),
                          status=("interpreter_error" if error else "timeout" if exc_type == "TimeoutError"
                                  else "execution_error" if exc_type else "success"),
                          exception_type=error or exc_type, **context)

    def cleanup_attempt(self, interpreter):
        """Measure existing explicit failure recovery without changing its policy."""
        started = time.perf_counter()
        error = None
        try:
            return interpreter.cleanup_session()
        except BaseException as exc:
            error = type(exc).__name__
            raise
        finally:
            elapsed = time.perf_counter() - started
            with self._lock:
                self.cumulative_real_s += elapsed
                self.emit("attempt_cleanup", attempt_id=getattr(interpreter, "execution_cost_attempt_id", None),
                          elapsed_s=elapsed, exception_type=error)

    def review(self, node, callback):
        started = time.perf_counter()
        error = None
        try:
            return callback()
        except BaseException as exc:
            error = type(exc).__name__
            raise
        finally:
            self.emit("review_end", **self.node_fields(node), elapsed_s=time.perf_counter() - started,
                      exception_type=error)
