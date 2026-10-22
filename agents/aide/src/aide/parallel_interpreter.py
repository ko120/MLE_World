"""Parallel real+simulator race wrapper for AIDE.

Starts the real `Interpreter` (multiprocessing.Process child) immediately and
admits the `SimulatorInterpreter` (vLLM streaming call) only if the real child
is still running after `start_delay_s`. Real wins if it returns within
`deadline_s` (success OR error). Otherwise the real child is torn down via
`cleanup_session()` and the simulator's prediction is returned. If the
simulator itself failed (SimulatorError / SimulatorEmpty — no usable
prediction), the race instead
falls back to the still-running real execution and commits its genuine
result (real_won=True), so a dead sim backend degrades to sequential AIDE
rather than fabricating buggy nodes.

This hedged start avoids sending simulator requests for fast real executions,
which removes a large source of vLLM cancellation churn. The compact
execution-result target itself is runtime-independent; the admission delay is
only a deployment scheduling policy. See docs/PARALLEL_SIMULATOR_INTEGRATION.md.

With `sim_only=True` there is no race at all: the real child is never started,
the simulator's prediction IS the node result, and `deadline_s` /
`start_delay_s` are unused. The abstain fallback survives — a sim that failed
to produce a prediction still degrades to a genuine real execution rather than
fabricating a buggy node — because that path is about the sim backend being
down, not about racing.
"""

import logging
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutTimeout

from .interpreter import Interpreter, ExecutionResult
from .utils.artifacts import has_submission_csv
from .simulator import (
    SimulatorInterpreter,
    is_bad_prediction,
    is_sim_internal_failure,
)

logger = logging.getLogger("aide")


class ParallelInterpreter:
    """Race real `Interpreter` against `SimulatorInterpreter`.

    `sim_only=True` turns the race off: the simulator alone produces every node
    result and the real interpreter is touched only on the abstain fallback.
    """

    def __init__(
        self,
        real: Interpreter,
        simulator: SimulatorInterpreter,
        deadline_s: float = 30.0,
        start_delay_s: float = 3.0,
        sim_only: bool = False,
    ):
        self.real = real
        self.simulator = simulator
        self.deadline_s = float(deadline_s)
        self.start_delay_s = float(start_delay_s)
        self.sim_only = bool(sim_only)
        if self.deadline_s <= 0:
            raise ValueError("deadline_s must be greater than zero")
        if not 0 <= self.start_delay_s < self.deadline_s:
            raise ValueError(
                "start_delay_s must be non-negative and less than deadline_s"
            )
        # One pool reused across calls — avoids spawning two threads per step.
        self._pool = ThreadPoolExecutor(
            max_workers=2, thread_name_prefix="parallel-interp"
        )

    def set_data_overview(self, text: str | None) -> None:
        self.simulator.set_data_overview(text)

    def cleanup_session(self) -> None:
        return self.real.cleanup_session()

    def close(self) -> None:
        """Shut down the race pool without joining a possibly-abandoned real
        thread (wait=False). For per-candidate BoN-race instances, which are
        created and discarded every step; the persistent node-exec instance
        keeps its pool for the run's lifetime and never calls this."""
        self._pool.shutdown(wait=False)

    def _record_real_result(self, result: ExecutionResult) -> ExecutionResult:
        """Retain observed CSV presence before the caller removes its workspace."""
        result.real_won = True
        result.observed_submission_exists = has_submission_csv(self.real.working_dir)
        return result

    def run(self, code: str, reset_session: bool = True) -> ExecutionResult:
        if self.sim_only:
            return self._run_sim_only(code, reset_session)
        sim_cancel = threading.Event()
        real_f = self._pool.submit(self.real.run, code, reset_session)
        deadline_at = time.monotonic() + self.deadline_s

        def return_real(result: ExecutionResult, sim_f=None) -> ExecutionResult:
            if sim_f is None:
                sim_status = (
                    f"sim not admitted (delay={self.start_delay_s:.2f}s)"
                )
            else:
                # cancel() prevents queued work from starting. The event covers
                # a simulator call that has already entered its worker thread.
                sim_cancel.set()
                sim_f.cancel()
                sim_status = "sim cancelled"
            # Mark ground-truth provenance so the journal's `real_won` flag
            # is set when the agent absorbs this result. validate-on-best
            # uses this to skip nodes that don't need re-execution.
            self._record_real_result(result)
            logger.info(
                f"[parallel] real won in {result.exec_time:.2f}s "
                f"(exc_type={result.exc_type}); {sim_status}"
            )
            return result

        try:
            return return_real(real_f.result(timeout=self.start_delay_s))
        except FutTimeout:
            pass

        logger.info(
            f"[parallel] real still running after {self.start_delay_s:.2f}s; "
            "sim admitted"
        )
        sim_f = self._pool.submit(
            self.simulator.run, code, reset_session, sim_cancel
        )

        try:
            # The hedge delay is part of deadline_s, not extra latency.
            remaining = max(0.0, deadline_at - time.monotonic())
            return return_real(real_f.result(timeout=remaining), sim_f)
        except FutTimeout:
            # Real didn't make it by the deadline. Block on the simulator
            # BEFORE tearing down the real child: if the simulator itself
            # failed there is no prediction to commit, and the still-running
            # real execution is the only source of truth (killing it first
            # would force a from-scratch re-run). If the sim call hangs
            # here, real keeps making progress in the meantime and the
            # abstain path below surfaces the failure — don't paper over it.
            sim_result = sim_f.result()
            if is_sim_internal_failure(sim_result):
                # Sim abstained (backend/vLLM down, empty output) — degrade
                # to plain sequential AIDE for this step: wait for real to
                # finish (its own exec timeout bounds it) and commit the
                # genuine result. Committing the sentinel instead would
                # fabricate a buggy node — the 2026-07-13 half1 run turned a
                # dead sim server into 218 spurious buggy nodes that way.
                # Deliberately logged WITHOUT the "real won"/"real timed out"
                # markers that grade_run_group.sh's fallback parser pairs
                # with journal nodes.
                logger.info(
                    f"[parallel] sim abstained ({sim_result.exc_type}) at "
                    f"{self.deadline_s}s; falling back to real"
                )
                result = real_f.result()
                self._record_real_result(result)
                logger.info(
                    f"[parallel] real (post-abstain) finished in "
                    f"{result.exec_time:.2f}s (exc_type={result.exc_type})"
                )
                return result
            # Usable prediction — now tear down the real candidate tree. The
            # existing cleanup_session() owns cooperative shutdown plus bounded
            # process-group INT→TERM→KILL escalation; do NOT reimplement it.
            self.real.cleanup_session()
            logger.info(
                f"[parallel] real timed out at {self.deadline_s}s; using sim"
            )
            self._reconcile_submission(sim_result)
            return sim_result

    def _run_sim_only(
        self, code: str, reset_session: bool = True
    ) -> ExecutionResult:
        """Pure world-model rollout: no real child, no race, no deadline.

        The one place the real interpreter is still reachable is the abstain
        fallback. A simulator-internal failure (dead vLLM, empty output) is not
        a prediction, and committing its sentinel would fabricate a buggy node —
        the failure mode that cost the 2026-07-13 half1 run 218 spurious buggy
        nodes. With no already-running real execution to fall back on, we start
        one here and commit its genuine result, so a broken sim backend degrades
        to sequential AIDE instead of poisoning the tree.
        """
        sim_result = self.simulator.run(code, reset_session)
        if is_sim_internal_failure(sim_result):
            # Keep "[parallel] sim abstained" as the literal prefix:
            # check_run_health.py's SIM_ABSTAIN anchors on it to score sim
            # backend health, and grade_run_group.sh's PARALLEL_RE
            # deliberately does NOT match it (this step commits a real
            # result, so it is not a sim-provenance event).
            logger.info(
                f"[parallel] sim abstained ({sim_result.exc_type}) with no "
                f"race (exec.simulator_only); falling back to real"
            )
            result = self.real.run(code, reset_session)
            self._record_real_result(result)
            logger.info(
                f"[parallel] real (post-abstain) finished in "
                f"{result.exec_time:.2f}s (exc_type={result.exc_type})"
            )
            return result
        logger.info(
            f"[parallel] sim-only: using sim "
            f"(exc_type={sim_result.exc_type}); real never started"
        )
        self._reconcile_submission(sim_result)
        return sim_result

    def _reconcile_submission(self, sim_result: ExecutionResult) -> None:
        """Remove unvalidated CSVs left by an interrupted real child.

        Structured simulator results carry submission_exists directly, so no
        file is needed. Only legacy formats without that field use an empty
        marker for a predicted success. Neither can publish a CSV until real
        validation produces one.
        """
        sub = self.real.working_dir / "submission" / "submission.csv"
        # The killed child may even have created submission.csv as a
        # DIRECTORY (generated code does that occasionally); unlink()/
        # write_bytes() on a dir raises IsADirectoryError, so clear it.
        if sub.is_dir():
            shutil.rmtree(sub, ignore_errors=True)
        # is_bad_prediction, not `exc_type is None`: a parsed-garbage
        # exc_type (a non-exception token from the predicted tail) is NOT a
        # predicted crash, so it gets the success placeholder like a clean
        # prediction — consistent with is_predicted_crash, which BoN
        # selection already uses. Genuine predicted crashes still drop the
        # killed child's leftover (simulator-internal failures never reach
        # here — they take the abstain path).
        predicted_submission = getattr(
            sim_result, "predicted_submission_exists", None
        )
        if is_bad_prediction(sim_result) or predicted_submission is not None:
            sub.unlink(missing_ok=True)
        else:
            sub.parent.mkdir(parents=True, exist_ok=True)
            sub.write_bytes(b"")
