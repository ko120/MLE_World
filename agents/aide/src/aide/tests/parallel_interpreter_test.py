"""Focused tests for hedged simulator admission and cancellation."""

import threading
import time

import pytest

from aide.interpreter import ExecutionResult
from aide.parallel_interpreter import ParallelInterpreter
from aide import simulator as simulator_mod


def _result(source, exec_time=0.0, exc_type=None, predicted_submission_exists=None):
    return ExecutionResult(
        term_out=[source],
        exec_time=exec_time,
        exc_type=exc_type,
        exc_info=None,
        exc_stack=None,
        predicted_submission_exists=predicted_submission_exists,
    )


class _FakeReal:
    def __init__(self, working_dir, delay_s=None):
        self.working_dir = working_dir
        self.delay_s = delay_s
        self.release = threading.Event()
        self.cleanup_calls = 0
        self.calls = 0

    def run(self, code, reset_session=True):
        self.calls += 1
        if self.delay_s is None:
            self.release.wait(timeout=1.0)
            elapsed = 1.0
        else:
            time.sleep(self.delay_s)
            elapsed = self.delay_s
        return _result("real", exec_time=elapsed)

    def cleanup_session(self):
        self.cleanup_calls += 1
        self.release.set()


class _FakeSimulator:
    def __init__(self, result=None, wait_for_cancel=False):
        self.result = result or _result("sim")
        self.wait_for_cancel = wait_for_cancel
        self.calls = 0
        self.started = threading.Event()
        self.cancelled = threading.Event()

    def set_data_overview(self, text):
        return None

    def run(self, code, reset_session=True, cancel_event=None):
        self.calls += 1
        self.started.set()
        if self.wait_for_cancel:
            if cancel_event.wait(timeout=1.0):
                self.cancelled.set()
            return _result("cancelled", exc_type="SimulatorCancelled")
        return self.result


def test_fast_real_skips_simulator_request(tmp_path):
    real = _FakeReal(tmp_path, delay_s=0.0)
    sim = _FakeSimulator()
    race = ParallelInterpreter(real, sim, deadline_s=0.2, start_delay_s=0.05)
    try:
        result = race.run("print('fast')")
    finally:
        race.close()

    assert result.real_won is True
    assert result.term_out == ["real"]
    assert sim.calls == 0


def test_real_win_after_admission_cancels_simulator(tmp_path):
    real = _FakeReal(tmp_path, delay_s=0.04)
    sim = _FakeSimulator(wait_for_cancel=True)
    race = ParallelInterpreter(real, sim, deadline_s=0.2, start_delay_s=0.01)
    try:
        result = race.run("print('late')")
        assert sim.cancelled.wait(timeout=0.5)
    finally:
        race.close()

    assert result.real_won is True
    assert sim.calls == 1


def test_simulator_wins_at_original_deadline(tmp_path):
    # Real finishes after the overall deadline but before delay + deadline. If
    # the implementation accidentally adds the hedge delay to the deadline,
    # this would incorrectly return the real result.
    real = _FakeReal(tmp_path, delay_s=0.17)
    sim = _FakeSimulator(result=_result("sim"))
    race = ParallelInterpreter(real, sim, deadline_s=0.12, start_delay_s=0.06)
    try:
        result = race.run("print('slow')")
    finally:
        race.close()

    assert result.term_out == ["sim"]
    assert result.real_won is False
    assert real.cleanup_calls == 1


@pytest.mark.parametrize("prediction", ["true", "false", "unknown"])
def test_structured_submission_removes_unvalidated_artifact(tmp_path, prediction):
    real = _FakeReal(tmp_path, delay_s=0.17)
    submission = tmp_path / "submission" / "submission.csv"
    submission.parent.mkdir()
    submission.write_text("id,pred\n0,999\n")
    sim = _FakeSimulator(result=_result(
        "sim", predicted_submission_exists=prediction,
    ))
    race = ParallelInterpreter(real, sim, deadline_s=0.12, start_delay_s=0.01)
    try:
        race.run("print('slow')")
    finally:
        race.close()

    assert not submission.exists()


def test_pre_cancelled_simulator_does_not_open_stream(tmp_path, monkeypatch):
    def unexpected_query(*args, **kwargs):
        pytest.fail("pre-cancelled work must not reach the HTTP query")

    monkeypatch.setattr(simulator_mod, "_streaming_query", unexpected_query)
    simulator = simulator_mod.SimulatorInterpreter(tmp_path, model="sim")
    cancel_event = threading.Event()
    cancel_event.set()

    result = simulator.run("print('unused')", cancel_event=cancel_event)

    assert result.exc_type == "SimulatorCancelled"
    assert result.exec_time == 0.0


def test_sim_only_never_starts_real(tmp_path):
    real = _FakeReal(tmp_path, delay_s=0.0)
    sim = _FakeSimulator()
    race = ParallelInterpreter(real, sim, sim_only=True)
    try:
        result = race.run("print('sim only')")
    finally:
        race.close()

    assert real.calls == 0
    assert real.cleanup_calls == 0
    assert sim.calls == 1
    assert result.term_out == ["sim"]
    assert result.real_won is False


def test_sim_only_abstain_falls_back_to_real(tmp_path):
    # A dead sim backend must degrade to a genuine execution, never commit the
    # sentinel as a buggy node.
    real = _FakeReal(tmp_path, delay_s=0.0)
    sim = _FakeSimulator(result=_result("abstain", exc_type="SimulatorError"))
    race = ParallelInterpreter(real, sim, sim_only=True)
    try:
        result = race.run("print('sim down')")
    finally:
        race.close()

    assert real.calls == 1
    assert result.term_out == ["real"]
    assert result.real_won is True


@pytest.mark.parametrize("prediction", ["true", "false", "unknown"])
def test_sim_only_needs_no_placeholder(tmp_path, prediction):
    real = _FakeReal(tmp_path, delay_s=0.0)
    sim = _FakeSimulator(result=_result(
        "sim", predicted_submission_exists=prediction,
    ))
    race = ParallelInterpreter(real, sim, sim_only=True)
    try:
        race.run("print('sim only')")
    finally:
        race.close()

    submission = tmp_path / "submission" / "submission.csv"
    assert not submission.exists()


def test_sim_only_predicted_crash_drops_leftover(tmp_path):
    (tmp_path / "submission").mkdir()
    leftover = tmp_path / "submission" / "submission.csv"
    leftover.write_text("id,label\n1,0\n")
    real = _FakeReal(tmp_path, delay_s=0.0)
    sim = _FakeSimulator(result=_result("sim", exc_type="KeyError"))
    race = ParallelInterpreter(real, sim, sim_only=True)
    try:
        race.run("print('boom')")
    finally:
        race.close()

    assert not leftover.exists()
