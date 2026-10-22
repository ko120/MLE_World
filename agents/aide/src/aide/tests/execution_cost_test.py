"""Measurement must count real work without changing search or copying artifacts."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from aide.agent import Agent
from aide.execution_gate import ExecutionGateConfig
from aide.interpreter import ExecutionResult, Interpreter
from aide.journal import Journal, Node
from aide.parallel_interpreter import ParallelInterpreter
from aide.run import validate_on_best
from aide.utils.execution_cost import ExecutionCostRecorder
from aide.utils.metric import MetricValue
from aide.utils.submission_validation import submission_sha256
from mlebench import costs as report


SOURCE = Path(__file__).resolve().parents[5]


def rows(recorder):
    return [json.loads(line) for line in recorder.path.read_text().splitlines()]


def interpreter(recorder, run):
    instance = object.__new__(Interpreter)
    instance.timeout = 20
    instance.working_dir = recorder.path.parent.parent
    instance.execution_cost_recorder = recorder
    instance.execution_cost_context = {"node_id": "n", "stage": "draft", "retry_index": 0, "triggers": {"baseline": True}}
    instance._run = run
    instance.cleanup_session = Mock()
    return instance


def test_measured_duration_includes_timeout_overrun_and_recovery(tmp_path, monkeypatch):
    now = [0.0]
    monkeypatch.setattr("aide.utils.execution_cost.time.perf_counter", lambda: now[0])
    recorder = ExecutionCostRecorder(tmp_path / "logs")

    def execute(*args):
        now[0] += 25
        return ExecutionResult([], 20, "TimeoutError", real_won=True)

    real = interpreter(recorder, execute)
    result = real.run("pass")
    real.cleanup_session.side_effect = lambda: now.__setitem__(0, now[0] + 3)
    recorder.cleanup_attempt(real)
    recorder.emit("run_end")
    summary, attempts, _ = report.summarize(tmp_path / "logs")
    assert result.exec_time == 20
    assert summary["known_real_s"] == 28
    assert summary["cost_complete"]
    assert attempts[0]["reported_exec_s"] == 20
    assert attempts[0]["real_s"] == 28
    assert summary["timeout_cost_share"] == 1


def test_simulator_skip_and_abstain_fallback_count_only_real_time(tmp_path, monkeypatch):
    now = [0.0]
    monkeypatch.setattr("aide.utils.execution_cost.time.perf_counter", lambda: now[0])
    recorder = ExecutionCostRecorder(tmp_path / "logs")

    def execute(*args):
        now[0] += 7
        return ExecutionResult([], 7, None, real_won=True)

    real = interpreter(recorder, execute)
    sim = SimpleNamespace(run=Mock(return_value=ExecutionResult([], 0, None, predicted_submission_exists="true")))
    wrapped = ParallelInterpreter(real, sim, sim_only=True)
    try:
        wrapped.run("pass")
        assert not [e for e in rows(recorder) if e["event"] == "attempt_start"]
        sim.run.return_value = ExecutionResult([], 0, "SimulatorError")
        wrapped.run("pass")
        assert recorder.cumulative_real_s == 7
        assert len([e for e in rows(recorder) if e["event"] == "attempt_start"]) == 1
    finally:
        wrapped.close()


def test_validation_retry_overlap_and_publication_metadata(tmp_path, monkeypatch):
    now = [0.0]
    monkeypatch.setattr("aide.utils.execution_cost.time.perf_counter", lambda: now[0])
    recorder = ExecutionCostRecorder(tmp_path / "logs", {"threshold": .7})
    node = Node(code="candidate", is_buggy=False, sim_p_max=.2)
    node.metric = MetricValue(.8, maximize=True)
    journal = Journal()
    journal.append(node)
    agent = SimpleNamespace(
        execution_cost_recorder=recorder, journal=journal,
        acfg=SimpleNamespace(real_validation_p_max_threshold=.7, real_validation_p_max_stages="all"),
    )
    calls = []

    def execute(*args):
        calls.append(1)
        now[0] += 4
        if len(calls) == 1:
            raise RuntimeError("queue failed")
        (tmp_path / "submission/submission.csv").write_text("id,pred\n1,.8\n")
        return ExecutionResult([], 4, None, real_won=True)

    real = interpreter(recorder, execute)
    real.cleanup_session.side_effect = lambda: now.__setitem__(0, now[0] + 2)

    def parse(node, result):
        node.absorb_exec_result(result)
        node.is_buggy = False
        node.submission_validation = "valid"
        node.submission_sha256 = submission_sha256(tmp_path / "submission/submission.csv")
        node.test_metric = .6
        return node

    agent.parse_exec_result = parse
    assert validate_on_best(agent, journal, real, tmp_path, force=True)
    events = rows(recorder)
    starts = [e for e in events if e["event"] == "attempt_start"]
    assert [e["retry_index"] for e in starts] == [0, 1]
    assert starts[0]["triggers"] == {"best": True, "confidence": True, "periodic_or_forced": True}
    assert recorder.cumulative_real_s == 10
    assert len([e for e in events if e["event"] == "publication"]) == 1
    assert (tmp_path / "best_submission/submission.csv").exists()
    assert sorted(p.name for p in (tmp_path / "logs").iterdir()) == ["real_execution_events.jsonl"]
    _, attempts, _ = report.summarize(tmp_path / "logs")
    assert attempts[0]["validation_status"] is None
    assert attempts[1]["validation_status"] == "valid"


@pytest.mark.parametrize("exc_type,threshold", [(None, 0.9), ("ValueError", 0.7), ("TimeoutError", 0.7)])
def test_outcome_gate_decisions_record_effective_branch_and_threshold(tmp_path, exc_type, threshold):
    recorder = ExecutionCostRecorder(tmp_path / "logs", {"threshold": 0.123})
    prior = Node(code="incumbent", is_buggy=False)
    prior.metric = MetricValue(0.95, maximize=True)
    node = Node(code="selected", parent=prior, is_buggy=exc_type is not None,
                exc_type=exc_type, sim_p_max=threshold)
    node.metric = MetricValue(0.4, maximize=True)
    journal = Journal()
    journal.append(prior)
    journal.append(node)
    agent = SimpleNamespace(
        execution_cost_recorder=recorder, journal=journal,
        acfg=SimpleNamespace(execution_gate=ExecutionGateConfig(True, 0.9, 0.7)),
    )
    real = SimpleNamespace(run=Mock(side_effect=AssertionError("accepted or timeout")))
    assert not validate_on_best(agent, journal, real, tmp_path)
    decision = [e for e in rows(recorder) if e["event"] == "execution_decision"][-1]
    assert decision["threshold"] == threshold
    assert decision["gate_outcome"] == ("success" if exc_type is None else "failure")
    assert decision["gate_mode"] == "outcome"
    assert decision["predicted_exception"] == exc_type
    assert decision["skip_reason"] == ("predicted_timeout" if exc_type == "TimeoutError" else "confidence_accepted")


def test_outcome_gate_recorder_metadata_includes_both_thresholds(tmp_path):
    cfg = SimpleNamespace(
        log_dir=tmp_path,
        exec=SimpleNamespace(simulator=True, timeout=20, simulator_model="wm"),
        agent=SimpleNamespace(
            stage_policy=SimpleNamespace(draft="parallel", debug="parallel", improve="parallel"),
            use_predicted_metric_argmax_pick=False,
            execution_gate=ExecutionGateConfig(True, 0.9, None),
            real_validation_p_max_threshold=0.123, real_validation_p_max_stages="improve",
            real_validation_interval=1, time_limit=100,
            code=SimpleNamespace(model="generator"), feedback=SimpleNamespace(model="judge"),
        ),
    )
    recorder = ExecutionCostRecorder.from_config(cfg, run_start=0)
    start = rows(recorder)[0]
    assert start["execution_mode"] == "outcome_confidence+periodic"
    assert start["confidence_stages"] == "all" and start["threshold"] is None
    assert start["outcome_thresholds"] == {"success": 0.9, "failure": None}


@pytest.mark.parametrize("valid", [True, False])
def test_base_step_records_actual_publication_only(tmp_path, valid):
    recorder = ExecutionCostRecorder(tmp_path / "logs")
    agent = object.__new__(Agent)
    agent.execution_cost_recorder = recorder
    agent.cfg = SimpleNamespace(workspace_dir=tmp_path)
    agent.acfg = SimpleNamespace()
    agent.oracle = "real"
    agent.stage_policy = {"draft": "single"}
    agent.current_step = 0
    agent.journal = Journal()
    agent.data_preview = "data"
    agent.update_data_preview = lambda: None
    agent.search_policy = lambda: None
    agent._draft_prompt = lambda: {}
    agent.generate_code = lambda *args, **kwargs: ("plan", "code", None)

    def execute(*args):
        (tmp_path / "submission/submission.csv").write_text("id,pred\n1,.8\n")
        return ExecutionResult([], 0, None, real_won=True)

    real = interpreter(recorder, execute)

    def parse(node, exec_result):
        node.absorb_exec_result(exec_result)
        node.metric = MetricValue(.8, maximize=True)
        node.is_buggy = not valid
        node.submission_validation = "valid" if valid else "invalid"
        node.submission_sha256 = submission_sha256(tmp_path / "submission/submission.csv")
        return node

    agent.parse_exec_result = parse

    def callback(*args):
        real.execution_cost_context = recorder.node_fields(agent._execution_cost_node)
        return real.run(*args)

    agent.step(callback)
    events = rows(recorder)
    assert len([e for e in events if e["event"] == "node_committed"]) == 1
    assert len([e for e in events if e["event"] == "publication"]) == int(valid)
    summary, _, _ = report.summarize(tmp_path / "logs")
    assert summary["real_check_coverage"] == 1
    assert summary["published_valid_submission"] is valid


@pytest.mark.parametrize("reason", ["predicted_timeout", "deadline_reserve", "not_eligible"])
def test_skipped_validation_never_counts_as_an_attempt(tmp_path, reason):
    recorder = ExecutionCostRecorder(tmp_path / "logs")
    node = Node(code="candidate", is_buggy=reason == "not_eligible")
    node.metric = MetricValue(.8, maximize=True)
    node.exc_type = "TimeoutError" if reason == "predicted_timeout" else None
    journal = Journal()
    journal.append(node)
    agent = SimpleNamespace(execution_cost_recorder=recorder, acfg=SimpleNamespace(), journal=journal)
    real = interpreter(recorder, Mock(side_effect=AssertionError("must not execute")))
    assert not validate_on_best(agent, journal, real, tmp_path, deadline=0 if reason == "deadline_reserve" else None)
    assert reason in [e.get("skip_reason") for e in rows(recorder)]
    assert not [e for e in rows(recorder) if e["event"] == "attempt_start"]


def test_interruption_and_logging_failure_preserve_execution_semantics(tmp_path):
    recorder = ExecutionCostRecorder(tmp_path / "logs")
    real = interpreter(recorder, Mock(side_effect=KeyboardInterrupt))
    with pytest.raises(KeyboardInterrupt):
        real.run("pass")
    assert rows(recorder)[-1]["exception_type"] == "KeyboardInterrupt"
    recorder.path = tmp_path  # a directory: force a write failure
    result = ExecutionResult([], 1, None, real_won=True)
    real._run = lambda *args: result
    assert real.run("pass") is result
    assert not recorder.enabled


def test_incomplete_and_truncated_attempts_remain_unknown(tmp_path):
    recorder = ExecutionCostRecorder(tmp_path / "logs")
    recorder.emit("attempt_start", attempt_id="unfinished", node_id="n")
    with recorder.path.open("a") as output:
        output.write('{"event":')
    summary, attempts, _ = report.summarize(tmp_path / "logs")
    assert not summary["cost_complete"]
    assert summary["incomplete_attempts"] == 1
    assert summary["malformed_lines"] == 1
    assert attempts[0]["real_s"] is None


def test_legacy_proxy_excludes_simulated_durations(tmp_path):
    (tmp_path / "journal.json").write_text(json.dumps({"nodes": [
        {"real_won": True, "exec_time": 3}, {"real_won": True, "exec_time": 2, "exc_type": "ValueError"},
        {"real_won": False, "exec_time": 100},
    ]}))
    row = report.legacy_summary(tmp_path)
    assert row["known_real_s"] == 5
    assert not row["cost_complete"]
    assert row["real_attempts"] is None


def test_model_usage_retains_scoring_with_unknown_tokens(tmp_path):
    (tmp_path / "aide.log").write_text(
        "INFO openai call: in_tokens=12 out_tokens=3 req_time=2.0s model=coder\n"
        "INFO sim call: in_tokens=5 out_tokens=2 req_time=1.0s model=world-model\n"
        "INFO exc_posterior: p_clean=0.2 max_p=0.8 method=single_token seconds=0.250\n"
    )
    result = report.model_usage(tmp_path, {"trial": "t", "generator_model": "coder", "wm_model": "world-model"})
    assert result[0]["role"] == "generator"
    assert result[0]["input_tokens"] == 12
    assert result[-1]["kind"] == "exc_posterior"
    assert result[-1]["input_tokens"] is None
    assert result[-1]["operation_s"] == .25


def test_flops_report_filters_sim_time_and_counts_all_check_reasons(tmp_path):
    if not (SOURCE / "scripts/sample/flops_report.py").is_file():
        pytest.skip("requires the optional workspace FLOPs analysis tool")
    module_spec = importlib.util.spec_from_file_location("flops_report_tested", SOURCE / "scripts/sample/flops_report.py")
    flops = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(flops)
    logs = tmp_path / "task/logs"
    logs.mkdir(parents=True)
    (logs / "journal.json").write_text(json.dumps({"nodes": [
        {"real_won": True, "exec_time": 2}, {"real_won": False, "exec_time": 100},
    ]}))
    (logs / "aide.log").write_text("\n".join(
        f"INFO [validate-on-best] node abc {reason}; running real"
        for reason in ("became best", "confidence gate (p_max=0.2)", "exploration check")
    ))
    result = flops.scan_group(str(tmp_path))
    assert result["exec_s"] == 2
    assert result["vob"] == 3


def test_paired_costs_average_seeds_then_tasks_and_exclude_incomplete():
    trials = []
    for task, seed, best, confidence in [("a", 0, 10, 20), ("a", 1, 10, 30), ("b", 0, 10, 5)]:
        for mode, cost in [("best_only", best), ("confidence", confidence)]:
            trials.append(dict(method="bon", task=task, seed=seed, execution_mode=mode,
                               known_real_s=cost, cost_complete=True))
    result = report.paired_modes(trials)[0]
    assert result["delta_real_s"] == 5  # mean(task a=15, task b=-5), not mean of three seeds
    assert result["paired_trials"] == 3
    assert result["ratio_of_task_mean_costs"] == 1.5
    trials[-1]["cost_complete"] = False
    assert report.paired_modes(trials)[0]["paired_tasks"] == 1


def test_report_exports_metadata_without_touching_submission_files(tmp_path):
    recorder = ExecutionCostRecorder(tmp_path / "trial/logs")
    real = interpreter(recorder, lambda *args: ExecutionResult([], 0, None, real_won=True))
    real.run("pass")
    recorder.emit("run_end")
    (tmp_path / "trial/submission.csv").write_text("keep me")
    report.main([str(tmp_path / "trial"), "--out-dir", str(tmp_path / "report")])
    assert (tmp_path / "trial/submission.csv").read_text() == "keep me"
    assert sorted(p.name for p in (tmp_path / "report").iterdir()) == [
        "attempts.csv", "breakdowns.csv", "model_usage.csv", "paired_modes.csv", "publications.csv", "trials.csv"]
