"""Recover selected CSVs after validator outages, without model or server calls."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import time

import pytest

from aide_submission_processing_test import harness, _code, _run, VALID
from aide.interpreter import ExecutionResult
from aide.journal import Node
from aide.utils import submission_recovery as recovery
from aide.utils import submission_validation as validation_module
from aide.utils.artifacts import publish_real_solution, reset_submission_directory
from aide.utils.metric import MetricValue, WorstMetricValue


@pytest.fixture
def clock(monkeypatch):
    state = SimpleNamespace(wall=time.time(), mono=time.monotonic())
    def advance(seconds):
        state.wall += seconds
        state.mono += seconds
    state.advance = advance
    fake_time = SimpleNamespace(time=lambda: state.wall, monotonic=lambda: state.mono)
    monkeypatch.setattr(recovery, "time", fake_time)
    monkeypatch.setattr(validation_module, "time", fake_time)
    return state


def pending_candidate(harness):
    agent, state, server, uploads = harness
    incumbent = _run(agent, _code())
    state['metric'] = 0.75  # Recovery publication requires a strict loss improvement.
    healthy_server = server.side_effect
    server.side_effect = lambda *args, **kwargs: SimpleNamespace(status_code=503)
    candidate_csv = VALID.replace("0.2", "0.6")
    node = _run(agent, _code(candidate_csv), metric=0.75)
    assert node.pending_submission and node.submission_validation == "unknown"
    return agent, server, uploads, incumbent, node, healthy_server, candidate_csv


@pytest.mark.parametrize('value,maximize', [(0.8, False), (0.9, False), (0.7, True)])
def test_recovered_tied_worse_or_incomparable_score_keeps_incumbent(harness, clock, value, maximize):
    agent, server, _, incumbent, node, healthy, _ = pending_candidate(harness)
    node.metric = MetricValue(value, maximize=maximize)
    server.side_effect = healthy
    clock.advance(61)
    assert recovery.retry_pending_submissions(agent) == 1
    assert node.pending_submission is None and node.submission_validation == 'valid'
    assert (agent.cfg.workspace_dir / 'best_submission/submission.csv').read_text() == VALID
    assert (agent.cfg.workspace_dir / 'best_solution/node_id.txt').read_text() == incumbent.id


def test_recovery_keeps_exact_csv_across_steps_and_serialization(harness, clock):
    agent, server, uploads, incumbent, node, healthy, csv_text = pending_candidate(harness)
    workspace = agent.cfg.workspace_dir
    retained = Path(node.pending_submission["path"])
    assert retained.read_text() == csv_text
    restored = Node.from_dict(node.to_dict())
    agent.journal.nodes[-1] = restored
    reset_submission_directory(workspace)
    working = workspace / "submission/submission.csv"
    working.write_text("another candidate's current output")
    assert (workspace / "best_submission/submission.csv").read_text() == VALID

    server.side_effect = healthy
    agent._review = Mock(side_effect=AssertionError("Must not rerun metric extraction"))
    agent.generate_code = Mock(side_effect=AssertionError("Must not generate another program"))
    source_review = Mock(wraps=agent._check_data_leakage)
    agent._check_data_leakage = source_review
    assert recovery.retry_pending_submissions(agent) == 0  # cooldown
    assert server.call_count == 2
    clock.advance(61)
    assert recovery.retry_pending_submissions(agent) == 1
    assert restored.pending_submission is None and restored.submission_validation == "valid"
    assert restored.submission_columns == ["id", "target"]
    assert Node.from_dict(restored.to_dict()).submission_columns == ["id", "target"]
    assert restored.metric.value == node.metric.value and not restored.is_buggy
    assert uploads[-1] == csv_text.encode()
    assert (workspace / "best_submission/submission.csv").read_text() == csv_text
    assert (workspace / "best_solution/solution.py").read_text() == restored.code
    assert (workspace / "best_solution/node_id.txt").read_text() == restored.id
    assert working.read_text() == "another candidate's current output"
    source_review.assert_called_once()
    agent._review.assert_not_called()
    agent.generate_code.assert_not_called()


def test_processing_wrapper_does_not_reuse_unknown_verdict(harness, clock):
    agent, _, server, _ = harness
    reset_submission_directory(agent.cfg.workspace_dir)
    (agent.cfg.workspace_dir / "submission/submission.csv").write_text(VALID)
    result = ExecutionResult(["Validation loss: 0.8"], 0.1, None, real_won=True)
    healthy = server.side_effect
    server.side_effect = lambda *args, **kwargs: SimpleNamespace(status_code=503)
    first = agent._process_real_submission(agent.cfg.workspace_dir, result, "pass")[1]
    assert first.status == "unknown"
    clock.advance(61)
    server.side_effect = healthy
    second = agent._process_real_submission(agent.cfg.workspace_dir, result, "pass")[1]
    assert second.status == "valid" and server.call_count == 2
    third = agent._process_real_submission(agent.cfg.workspace_dir, result, "pass")[1]
    assert third == second and server.call_count == 2  # definitive verdict remains cached


def test_repeated_outage_stays_pending_without_repeated_immediate_calls(harness, clock):
    agent, server, _, _, node, _, _ = pending_candidate(harness)
    analysis = node.analysis
    clock.advance(61)
    assert recovery.retry_pending_submissions(agent) == 0
    assert node.pending_submission["attempts"] == 1 and server.call_count == 3
    assert recovery.retry_pending_submissions(agent) == 0 and server.call_count == 3
    assert node.submission_validation == "unknown" and not node.is_buggy
    assert node.analysis == analysis
    assert (agent.cfg.workspace_dir / "best_submission/submission.csv").read_text() == VALID


def test_invalid_retry_routes_to_debug_and_preserves_incumbent(harness, clock):
    agent, server, _, incumbent, node, _, _ = pending_candidate(harness)
    server.side_effect = lambda *args, **kwargs: SimpleNamespace(
        status_code=200, json=lambda: {"result": "Submission invalid! Wrong test IDs."}
    )
    clock.advance(61)
    assert recovery.retry_pending_submissions(agent) == 1
    assert node.pending_submission is None and node.is_buggy
    assert node.metric.value is None and node.submission_validation == "invalid"
    assert "Wrong test IDs" in node.failure_reasons["submission"][0]
    assert agent.journal.get_best_node(validated_only=True) is incumbent
    assert (agent.cfg.workspace_dir / "best_submission/submission.csv").read_text() == VALID


def test_deferred_source_rejection_blocks_recovered_csv(harness, clock):
    agent, server, _, incumbent, node, healthy, _ = pending_candidate(harness)
    server.side_effect = healthy
    def reject(candidate, **kwargs):
        assert candidate.submission_validation == "valid"
        assert agent.journal.get_best_node(validated_only=True) is incumbent
        candidate.is_buggy = True
        candidate.metric = WorstMetricValue()
    agent._check_data_leakage = Mock(side_effect=reject)
    clock.advance(61)
    assert recovery.retry_pending_submissions(agent) == 1
    assert node.is_buggy and node.pending_submission is None
    agent._check_data_leakage.assert_called_once()
    assert (agent.cfg.workspace_dir / "best_submission/submission.csv").read_text() == VALID


@pytest.mark.parametrize("change", ["csv", "code", "missing_csv"])
def test_changed_retained_evidence_cannot_be_published(harness, clock, change):
    agent, server, _, _, node, healthy, _ = pending_candidate(harness)
    path = Path(node.pending_submission["path"])
    if change == "csv":
        path.write_text("id,target\n001,0.99\n")
    elif change == "code":
        node.code += "\n# modified after execution"
    else:
        path.unlink()
    server.side_effect = healthy
    clock.advance(61)
    assert recovery.retry_pending_submissions(agent) == 0
    assert server.call_count == 2
    assert node.pending_submission["last_error"]
    assert not node.is_buggy and node.submission_validation == "unknown"
    assert (agent.cfg.workspace_dir / "best_submission/submission.csv").read_text() == VALID


def test_format_recovery_waits_for_source_review_budget(harness, clock):
    agent, server, _, incumbent, node, healthy, _ = pending_candidate(harness)
    agent._check_data_leakage = Mock()
    def slow_server(*args, **kwargs):
        response = healthy(*args, **kwargs)
        clock.advance(2)
        return response
    server.side_effect = slow_server
    clock.advance(61)
    assert recovery.retry_pending_submissions(agent, timeout_s=2) == 0
    assert node.pending_submission and node.submission_validation == "valid"
    assert agent.journal.get_best_node(validated_only=True) is incumbent
    assert not publish_real_solution(
        agent.cfg.workspace_dir, node, source_path=Path(node.pending_submission["path"])
    )
    agent._check_data_leakage.assert_not_called()
    count = server.call_count
    clock.advance(61)
    assert recovery.retry_pending_submissions(agent) == 1
    assert server.call_count == count  # retry only the deferred source review
    agent._check_data_leakage.assert_called_once()


def test_recovery_preserves_final_saving_reserve(harness, clock):
    agent, server, _, _, node, healthy, _ = pending_candidate(harness)
    server.side_effect = healthy
    clock.advance(61)
    agent.deadline = clock.wall + 30
    assert recovery.retry_pending_submissions(agent) == 0
    assert server.call_count == 2 and node.pending_submission["attempts"] == 0


def test_recovered_lower_score_does_not_replace_incumbent(harness, clock):
    agent, server, _, incumbent, node, healthy, _ = pending_candidate(harness)
    node.metric = MetricValue(0.9, maximize=False)  # incumbent loss is 0.8
    server.side_effect = healthy
    clock.advance(61)
    assert recovery.retry_pending_submissions(agent) == 1
    assert node.submission_validation == "valid" and not node.pending_submission
    assert agent.journal.get_best_node(validated_only=True) is incumbent
    assert (agent.cfg.workspace_dir / "best_submission/submission.csv").read_text() == VALID


def test_recording_failure_does_not_unconfirm_a_published_recovery(harness, clock):
    agent, server, _, _, node, healthy, csv_text = pending_candidate(harness)
    server.side_effect = healthy
    node.metric = MetricValue(0.7, maximize=False)
    agent.execution_cost_recorder = SimpleNamespace(
        node_result=Mock(side_effect=OSError("audit log is unavailable"))
    )
    clock.advance(61)
    assert recovery.retry_pending_submissions(agent) == 1
    assert node.pending_submission is None
    assert agent.journal.get_best_node(validated_only=True) is node
    assert (agent.cfg.workspace_dir / "best_submission/submission.csv").read_text() == csv_text


def test_strongest_pending_metric_is_recovered_first(harness, clock):
    agent, server, _, _, older, healthy, _ = pending_candidate(harness)
    harness[1]['metric'] = 0.7
    newer = _run(agent, _code(VALID.replace("0.2", "0.7")), metric=0.7)
    assert newer.pending_submission
    order = []
    def source_review(node, **kwargs):
        order.append(node.id)
        clock.advance(2)  # exhaust this recovery batch after the best candidate
    agent._check_data_leakage = source_review
    server.side_effect = healthy
    clock.advance(61)
    assert recovery.retry_pending_submissions(agent, timeout_s=2) == 1
    assert order == [newer.id] and older.pending_submission
