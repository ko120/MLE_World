"""Matched BoN eligibility/fallback and the opt-in failed-pool judge."""

import json
from unittest.mock import Mock

import pytest

from aide.agent import CandidateOutcome
from aide.journal import Node
from aide.tests.ucb_stage_policy_test import CODES, _agent
from aide.ucb import resolve_config
from aide.utils.metric import MetricValue


@pytest.mark.parametrize("stage", ["draft", "debug", "improve"])
@pytest.mark.parametrize("pool_policy", ["review_filtered", "bon"])
@pytest.mark.parametrize("beta", [0.5, 1.0])
def test_failed_pool_reuses_exact_bon_judge(tmp_path, stage, pool_policy, beta):
    agent, scorer, predictions = _agent(tmp_path / "ucb", stage, beta=beta)
    agent.acfg.ucb.pool_policy = pool_policy
    if pool_policy == "review_filtered":
        agent.acfg.ucb.all_failed_fallback = "judge"
    for result in predictions.values():
        result.exc_type = "TypeError"
    agent._judge_query = Mock(return_value={"best_candidate": 2})
    assert agent.generate_code({}, stage)[1] == CODES[2][1]
    scorer.assert_not_called()
    agent._judge_query.assert_called_once()
    audit = json.loads((tmp_path / "ucb/bon/step_0/listwise.json").read_text())
    assert audit["accepted"] == [] and audit["winner"] == 2

    bon, bon_scorer, bon_predictions = _agent(tmp_path / "bon", stage, policy="parallel")
    bon.acfg.use_predicted_metric_argmax_pick = True
    for result in bon_predictions.values():
        result.exc_type = "TypeError"
    bon._judge_query = Mock(return_value={"best_candidate": 2})
    assert bon.generate_code({}, stage)[1] == CODES[2][1]
    assert bon._judge_query.call_args == agent._judge_query.call_args
    bon_scorer.assert_not_called()


@pytest.mark.parametrize("reply", [None, {"best_candidate": 99}])
def test_failed_judge_preserves_bon_first_candidate_fallback(tmp_path, reply):
    agent, scorer, predictions = _agent(tmp_path)
    agent.acfg.ucb.all_failed_fallback = "judge"
    for result in predictions.values():
        result.exc_type = "TypeError"
    agent._judge_query = Mock(return_value=reply)
    if reply is None:
        agent._judge_query.side_effect = RuntimeError("judge unavailable")
    assert agent.generate_code({}, "improve")[1] == CODES[0][1]
    agent._judge_query.assert_called_once()
    scorer.assert_not_called()


@pytest.mark.parametrize("all_failed,missing_metric", [(False, False), (False, True), (True, False)])
def test_judge_switch_changes_only_an_entirely_failed_pool(tmp_path, all_failed, missing_metric):
    winners = []
    for fallback in ["p_clean", "judge"]:
        agent, scorer, predictions = _agent(tmp_path / fallback)
        agent.acfg.ucb.all_failed_fallback = fallback
        for result in predictions.values():
            if all_failed:
                result.exc_type = "TypeError"
            if missing_metric:
                result.predicted_metric = None
        scorer.side_effect = lambda code: {"p_clean": 0.9 if code == CODES[1][1] else 0.1,
                                          "max_p": 0.9}
        agent._judge_query = Mock(return_value={"best_candidate": 3})
        winners.append(agent.generate_code({}, "improve")[1])
        assert agent._judge_query.call_count == int(all_failed and fallback == "judge")
    assert (winners[0] != winners[1]) is all_failed


@pytest.mark.parametrize("lower", [False, True, None])
@pytest.mark.parametrize("case", ["normal", "zero_ties", "missing", "reviews", "real", "failed", "extreme"])
def test_bon_pool_beta_zero_matches_bon_on_identical_candidates(tmp_path, lower, case):
    agent, scorer, predictions = _agent(tmp_path, beta=0)
    agent.acfg.ucb.pool_policy = "bon"
    agent.acfg.eval_lower_is_better = lower
    values = {
        "zero_ties": [None, 0, 0, -1],
        "missing": [None, float("nan"), float("inf"), None],
        "extreme": [1e308, -1e308, 0, None],
    }.get(case, [0.9, 0.8, 0.7, 0.6])
    outcomes = []
    for i, (_, code) in enumerate(CODES):
        result = predictions[code]
        result.predicted_metric = values[i]
        result.real_won = case == "real" and i == 0
        review = {"metric": 99, "is_bug": True, "metric_trustworthy": False} if case == "reviews" else {}
        outcomes.append(CandidateOutcome(result, case != "failed" and i != 3, "true", True, review))
    agent._judge_query = Mock(return_value={"best_candidate": 2})
    expected = agent._select(CODES, outcomes, tmp_path, selector="predicted_metric_argmax")
    expected_audit = json.loads((tmp_path / "listwise.json").read_text())
    assert agent._select_ucb(CODES, outcomes, tmp_path, "improve", None) == expected
    assert json.loads((tmp_path / "listwise.json").read_text()) == expected_audit
    scorer.assert_not_called()


@pytest.mark.parametrize("beta,expected", [(0.5, 0), (1.0, 1)])
def test_bon_pool_keeps_review_rejected_metrics_and_beta_controls_bonus(tmp_path, beta, expected):
    agent, scorer, predictions = _agent(tmp_path, beta=beta, sigma="p_clean")
    agent.acfg.ucb.pool_policy = "bon"
    outcomes = []
    for i, (_, code) in enumerate(CODES):
        predictions[code].predicted_metric = [0.9, 0.75, 0.1, 0.1][i]
        outcomes.append(CandidateOutcome(predictions[code], True, "true", True,
                                         {"is_bug": True, "metric_trustworthy": False}))
    scorer.side_effect = lambda code: {"p_clean": 0.79 if code == CODES[1][1] else 0.99}
    assert agent._select_ucb(CODES, outcomes, tmp_path, "improve", None) == expected
    audit = json.loads((tmp_path / "ucb.json").read_text())
    assert audit["options"]["pool_policy"] == "bon"
    assert all(row["eligible"] for row in audit["candidates"])
    assert audit["greedy_winner"] == 0
    agent._judge_query.assert_not_called()


@pytest.mark.parametrize("case", ["missing", "unknown_direction", "real"])
def test_bon_pool_nonzero_beta_preserves_first_usable_fallback(tmp_path, case):
    agent, scorer, predictions = _agent(tmp_path)
    agent.acfg.ucb.pool_policy = "bon"
    if case == "unknown_direction":
        agent.acfg.eval_lower_is_better = None
    outcomes = []
    for i, (_, code) in enumerate(CODES):
        result = predictions[code]
        if case == "missing":
            result.predicted_metric = None
        result.real_won = case == "real"
        outcomes.append(CandidateOutcome(result, i > 0, "true", True, {"metric": 99}))
    parent = Node(code="parent", is_buggy=False, metric=MetricValue(0.8, maximize=True))
    assert agent._select_ucb(CODES, outcomes, tmp_path, "improve", parent) == 1
    scorer.assert_not_called()
    agent._judge_query.assert_not_called()


def test_bon_pool_numeric_overflow_falls_back_to_bon_metric_selection(tmp_path):
    agent, _, predictions = _agent(tmp_path)
    agent.acfg.ucb.pool_policy = "bon"
    agent.acfg.ucb.metric_scale = 1e-300
    predictions[CODES[1][1]].predicted_metric = 1e308
    assert agent.generate_code({}, "improve")[1] == CODES[1][1]
    audit = json.loads((tmp_path / "bon/step_0/listwise.json").read_text())
    assert audit["judge"] == "predicted_metric_argmax"


@pytest.mark.parametrize("stage", ["draft", "debug"])
def test_hybrid_leaves_draft_and_debug_on_bon(tmp_path, stage):
    agent, scorer, _ = _agent(tmp_path)
    agent.acfg.ucb.pool_policy = "bon"
    agent.acfg.use_predicted_metric_argmax_pick = True
    agent.acfg.stage_policy.draft = agent.acfg.stage_policy.debug = "parallel"
    agent.stage_policy = agent._resolve_stage_policies()
    assert agent.stage_policy == {"draft": "parallel", "debug": "parallel", "improve": "ucb"}
    assert agent.generate_code({}, stage)[1] == CODES[0][1]
    scorer.assert_not_called()
    agent._judge_query.assert_not_called()


@pytest.mark.parametrize("field", ["pool_policy", "all_failed_fallback"])
def test_invalid_pool_options_fail_during_configuration(field):
    with pytest.raises(ValueError, match=field):
        resolve_config({field: "typo"}, "improve")
