"""Strict publication and incumbent-retention regressions; no model/training calls."""


import pytest

from aide.utils.metric import MetricValue, strictly_improves


@pytest.mark.parametrize('maximize', [False, True])
@pytest.mark.parametrize('value,expected', [(0.8, False), (0.9, True), (0.7, False)])
def test_publication_requires_strict_improvement(maximize, value, expected):
    candidate = value if maximize else 1.6 - value
    assert strictly_improves(MetricValue(candidate, maximize=maximize), MetricValue(.8, maximize=maximize)) is expected


@pytest.mark.parametrize('value', [None, float('nan'), float('inf'), -float('inf')])
def test_nonfinite_scores_cannot_publish_or_displace_incumbent(value):
    assert not strictly_improves(MetricValue(value, maximize=True), None)
    assert not strictly_improves(MetricValue(value, maximize=True), MetricValue(.8, maximize=True))
    assert not strictly_improves(MetricValue(.9, maximize=True), MetricValue(value, maximize=True))


def test_unknown_or_different_direction_preserves_incumbent():
    assert strictly_improves(MetricValue(.8, maximize=True), None)
    assert not strictly_improves(MetricValue(.9, maximize=None), None)
    assert not strictly_improves(MetricValue(.9, maximize=True), MetricValue(.8, maximize=False))


@pytest.mark.parametrize('maximize', [False, True])
def test_real_validation_tie_preserves_csv_code_and_node_id(tmp_path, maximize):
    from aide_simulator_acceptance_test import _agent, _Real, PRIOR_CSV
    from aide.journal import Node
    from aide.run import validate_on_best
    agent = _agent(tmp_path)
    agent.acfg.eval_lower_is_better = not maximize
    prior = agent.journal.nodes[0]
    prior.metric = MetricValue(.8, maximize=maximize)
    candidate = Node(code='new model', parent=prior, is_buggy=False,
                     metric=MetricValue(.9 if maximize else .7, maximize=maximize))
    agent.journal.append(candidate)
    assert validate_on_best(agent, agent.journal, _Real(tmp_path, sim_only=True), tmp_path)
    assert candidate.real_won and candidate.submission_validation == 'valid'
    assert candidate.metric.value == prior.metric.value
    assert (tmp_path / 'best_submission/submission.csv').read_text() == PRIOR_CSV
    assert (tmp_path / 'best_solution/solution.py').read_text() == prior.code
    assert (tmp_path / 'best_solution/node_id.txt').read_text() == prior.id

