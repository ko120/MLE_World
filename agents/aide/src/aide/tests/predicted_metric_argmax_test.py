"""Metric selection for usable pools; least-buggy judging for failed pools."""

import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from aide.agent import Agent, CandidateOutcome
from aide.interpreter import ExecutionResult


CANDS = [("plan", f"code_{i}") for i in range(3)]


def _agent():
    agent = object.__new__(Agent)
    agent.acfg = SimpleNamespace(
        use_best_of_n_listwise=True, use_predicted_metric_argmax_pick=True,
        use_p_clean_argmax_pick=False, use_line_frac_salvage=False,
        eval_lower_is_better=False,
    )
    agent.task_desc = "task"
    agent._judge_query = Mock(side_effect=AssertionError("No judge in metric selection."))
    agent._p_clean = Mock(side_effect=AssertionError("No posterior in metric selection."))
    return agent


@pytest.fixture
def agent():
    agent = _agent()
    yield agent
    agent._judge_query.assert_not_called()
    agent._p_clean.assert_not_called()


def _outcome(metric, *, usable=True, predicted=True, real_won=False, review=None):
    return CandidateOutcome(
        result=ExecutionResult(["output"], 0, None, real_won=real_won,
                               predicted_metric=metric),
        usable=usable, submission_exists="true", predicted=predicted, review=review,
    )


@pytest.mark.parametrize("lower,expected", [(False, 1), (True, 0)])
def test_best_metric_uses_configured_direction(agent, tmp_path, lower, expected):
    agent.acfg.eval_lower_is_better = lower
    outcomes = [_outcome(value) for value in (0.2, 0.9, 0.4)]
    assert agent._select(CANDS, outcomes, tmp_path) == expected
    audit = json.loads((tmp_path / "listwise.json").read_text())
    assert audit["judge"] == "predicted_metric_argmax"
    assert audit["lower_is_better"] is lower and audit["fallback_reason"] is None
    assert audit["scores"] == {"0": 0.2, "1": 0.9, "2": 0.4}


def test_only_usable_candidates_are_ranked(agent, tmp_path):
    outcomes = [_outcome(99, usable=False), _outcome(None), _outcome(-0.5)]
    assert agent._select(CANDS, outcomes, tmp_path) == 2


def test_zero_scores_are_valid_and_ties_keep_original_order(agent, tmp_path):
    outcomes = [_outcome(None), _outcome(0), _outcome(0)]
    assert agent._select(CANDS, outcomes, tmp_path) == 1


@pytest.mark.parametrize("invalid", [None, float("nan"), float("inf"), float("-inf"), True, {}, "bad"])
def test_invalid_metrics_do_not_become_zero(agent, tmp_path, invalid):
    assert agent._select(CANDS[:2], [_outcome(invalid), _outcome(-2)], tmp_path) == 1
    assert json.loads((tmp_path / "listwise.json").read_text())["scores"] == {"1": -2.0}


def test_no_metric_falls_back_to_first_usable_candidate(agent, tmp_path):
    outcomes = [_outcome(99, usable=False), _outcome(None), _outcome(float("nan"))]
    assert agent._select(CANDS, outcomes, tmp_path) == 1
    audit = json.loads((tmp_path / "listwise.json").read_text())
    assert audit["fallback_reason"] == "no_finite_predicted_metric"


@pytest.mark.parametrize("metric_argmax", [False, True])
@pytest.mark.parametrize("line_frac", [False, True])
def test_all_failed_listwise_pool_always_uses_judge(tmp_path, metric_argmax, line_frac):
    agent = _agent()
    agent.acfg.use_predicted_metric_argmax_pick = metric_argmax
    agent.acfg.use_line_frac_salvage = line_frac
    reply = {"best_candidate": 0, "reason": "The first candidate is easier to repair."}
    agent._judge_query = Mock(return_value=reply)
    candidates = [("plan", "a\nb\nc"), ("plan", "a\nb")]
    outcomes = [_outcome(0.1, usable=False), _outcome(99, usable=False)]
    outcomes[0].result.term_out = ['File "runfile.py", line 1']
    outcomes[1].result.term_out = ['File "runfile.py", line 2']
    assert agent._select(candidates, outcomes, tmp_path) == 0
    agent._judge_query.assert_called_once()
    agent._p_clean.assert_not_called()
    prompt = agent._judge_query.call_args.args[0]
    assert "All candidates failed" in prompt["Introduction"]
    assert "least buggy" in prompt["Introduction"]
    assert "Candidate 0" in prompt and "Candidate 1" in prompt
    audit = json.loads((tmp_path / "listwise.json").read_text())
    assert audit == {"accepted": [], "winner": 0, "judge": reply}


def test_metric_mode_failed_pool_judges_even_with_legacy_numeric_flags(tmp_path):
    agent = _agent()
    agent.acfg.use_best_of_n_listwise = False
    agent.acfg.use_p_clean_argmax_pick = True
    agent.acfg.use_line_frac_salvage = True
    agent._judge_query = Mock(return_value={"best_candidate": 1})
    outcomes = [_outcome(99, usable=False), _outcome(0.1, usable=False)]
    assert agent._select(CANDS[:2], outcomes, tmp_path) == 1
    agent._judge_query.assert_called_once()
    agent._p_clean.assert_not_called()


@pytest.mark.parametrize("metric_argmax", [False, True])
def test_failed_pool_judge_error_preserves_first_candidate_fallback(tmp_path, metric_argmax):
    agent = _agent()
    agent.acfg.use_predicted_metric_argmax_pick = metric_argmax
    agent._judge_query = Mock(side_effect=RuntimeError("judge unavailable"))
    outcomes = [_outcome(None, usable=False), _outcome(None, usable=False)]
    assert agent._select(CANDS[:2], outcomes, tmp_path) == 0
    agent._judge_query.assert_called_once()
    agent._p_clean.assert_not_called()
    assert "judge unavailable" in json.loads((tmp_path / "listwise.json").read_text())["judge"]


@pytest.mark.parametrize("predicted,real_won", [(False, True), (True, True)])
def test_observed_results_do_not_supply_stale_predicted_metrics(agent, tmp_path, predicted, real_won):
    outcomes = [_outcome(99, predicted=predicted, real_won=real_won), _outcome(0.4)]
    assert agent._select(CANDS[:2], outcomes, tmp_path) == 1


def test_reviewer_metrics_do_not_replace_missing_predictions(agent, tmp_path):
    outcomes = [_outcome(None, review={"metric": 99}), _outcome(-0.2)]
    assert agent._select(CANDS[:2], outcomes, tmp_path) == 1


def test_metric_flag_precedes_global_probability_argmax(agent, tmp_path):
    agent.acfg.use_p_clean_argmax_pick = True
    assert agent._select(CANDS[:2], [_outcome(0.1), _outcome(0.9)], tmp_path) == 1


@pytest.mark.parametrize("selector", ["ucb", "p_clean_argmax"])
def test_explicit_stage_selector_keeps_precedence(agent, tmp_path, selector):
    agent._select_ucb = Mock(return_value=0)
    outcomes = [_outcome(0.1), _outcome(0.9)]
    assert agent._select(CANDS[:2], outcomes, tmp_path, selector=selector, stage="improve") == 0
    agent._select_ucb.assert_called_once()


def test_missing_direction_uses_first_usable_without_guessing(agent, tmp_path):
    agent.acfg.eval_lower_is_better = None
    assert agent._select(CANDS[:2], [_outcome(0.1), _outcome(0.9)], tmp_path) == 0
    assert json.loads((tmp_path / "listwise.json").read_text())["fallback_reason"] == "unknown_metric_direction"


def test_disabling_metric_flag_restores_listwise_selection(tmp_path):
    agent = _agent()
    agent.acfg.use_predicted_metric_argmax_pick = False
    agent._judge_query = Mock(return_value={"best_candidate": 0})
    assert agent._select(CANDS[:2], [_outcome(0.1), _outcome(0.9)], tmp_path) == 0
    agent._judge_query.assert_called_once()
    agent._p_clean.assert_not_called()


def test_parallel_policy_commits_the_metric_winner(agent, tmp_path):
    outcomes = [_outcome(value) for value in (0.2, 0.9, 0.4)]
    agent._draw_candidates = Mock(return_value=CANDS)
    agent._new_step_log_dir = Mock(return_value=tmp_path)
    agent._persist_draw_axes = Mock()
    agent._observe = Mock(return_value=outcomes)
    agent._commit = Mock(return_value=("plan", "code_1", outcomes[1].result))
    result = agent._parallel({}, "improve", None, None)
    agent._commit.assert_called_once_with(CANDS, outcomes, 1, tmp_path)
    assert result[1] == "code_1"
