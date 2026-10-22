"""Posterior timeout warnings do not veto selection or expected gain."""
import json

import pytest

from aide import ucb
from aide.journal import Node
from aide.tests.ucb_stage_policy_test import CODES, _agent
from aide.utils.metric import MetricValue


@pytest.mark.parametrize("beta", [0.0, .25, 1.0])
@pytest.mark.parametrize("all_timeouts", [False, True])
def test_timeout_posterior_does_not_filter_candidates(tmp_path, beta, all_timeouts):
    agent, scorer, _ = _agent(tmp_path, beta=beta)
    scores = {code: {"p_clean": .9, "max_p": .9,
                    "top_exc": "TimeoutError" if all_timeouts or i == 0 else "ValueError",
                    "p_top_exc": 1e-12} for i, (_, code) in enumerate(CODES)}
    scorer.side_effect = scores.__getitem__
    chosen = agent.generate_code({}, "improve")[1]
    assert chosen == CODES[0][1]
    assert scorer.call_count == (4 if beta else 0)
    assert agent._selected_confidence(chosen) == (.9, "TimeoutError", 1e-12)
    assert scorer.call_count == (4 if beta else 1)
    audit = json.loads((tmp_path / "bon/step_0/ucb.json").read_text())
    assert all(candidate["eligible"] for candidate in audit["candidates"])
    agent._grade_test_submission.assert_not_called()


@pytest.mark.parametrize("usable", [False, True])
def test_timeout_posterior_does_not_filter_salvage(usable):
    candidates = [
        ucb.Candidate(0, usable, None, scores={"p_clean": .99, "top_exc": "TimeoutError"}),
        ucb.Candidate(1, usable, None, scores={"p_clean": .1, "top_exc": "ValueError"}),
    ]
    chosen, audit = ucb.select(candidates, ucb.Options(), maximize=True)
    assert chosen == 0 and audit["rule"] == "p_clean"


@pytest.mark.parametrize("maximize,metrics,parent", [(True, [.95, .9], .8), (False, [.05, .1], .2)])
def test_expected_gain_can_choose_smaller_improvement_with_higher_success(maximize, metrics, parent):
    candidates = [ucb.Candidate(i, True, metric, scores={"p_clean": [.2, .9][i]})
                  for i, metric in enumerate(metrics)]
    chosen, audit = ucb.select(candidates, ucb.Options(selection_rule="expected_gain"),
                             maximize=maximize, reference_metric=parent)
    assert audit["greedy_winner"] == 0 and chosen == 1
    assert audit["rule"] == "expected_gain"


@pytest.mark.parametrize("probability", [None, float("nan"), 1.1])
def test_missing_success_probability_falls_back_to_metric_greedy(probability):
    candidates = [ucb.Candidate(0, True, .95, scores={"p_clean": probability}),
                  ucb.Candidate(1, True, .9, scores={"p_clean": .9})]
    chosen, audit = ucb.select(candidates, ucb.Options(selection_rule="expected_gain"),
                             maximize=True, reference_metric=.8)
    assert chosen == 0 and audit["fallback_reason"] == "missing_p_clean"


def test_all_non_improvements_retain_best_metric():
    candidates = [ucb.Candidate(0, True, .7, scores={"p_clean": .99}),
                  ucb.Candidate(1, True, .8, scores={"p_clean": .1})]
    chosen, audit = ucb.select(candidates, ucb.Options(selection_rule="expected_gain"),
                             maximize=True, reference_metric=.9)
    assert chosen == 1 and audit["fallback_reason"] == "no_predicted_improvement"


@pytest.mark.parametrize("maximize", [False, True])
@pytest.mark.parametrize("mass", [1e-12, .99])
def test_timeout_posterior_does_not_filter_improve_parents(tmp_path, maximize, mass):
    agent, _, _ = _agent(tmp_path, beta=0)
    agent.acfg.search.num_drafts = 0
    agent.acfg.search.debug_prob = 0
    best = Node(code="best", is_buggy=False, sim_top_exc="TimeoutError", sim_p_top_exc=mass)
    best.metric = MetricValue(.999 if maximize else .001, maximize=maximize)
    other = Node(code="other", is_buggy=False, sim_top_exc="ValueError")
    other.metric = MetricValue(.8 if maximize else .2, maximize=maximize)
    agent.journal.append(best)
    assert agent.search_policy() is best
    agent.journal.append(other)
    assert agent.search_policy() is best
    assert agent.journal.get_best_node() is best


def test_invalid_selection_rule_rejected():
    with pytest.raises(ValueError, match="selection_rule"):
        ucb.resolve_config({"selection_rule": "typo"}, "improve")
