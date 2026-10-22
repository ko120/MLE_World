"""AutoMind parent selection without HPO, independent of UCB child selection."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from aide.agent import Agent
from aide.journal import Journal, Node
from aide.tests.ucb_stage_policy_test import CODES, _agent
from aide.utils.metric import MetricValue, WorstMetricValue
from aide.utils.serialize import dumps_json, loads_json


def _node(journal, score=None, *, parent=None, maximize=True, buggy=False, **kwargs):
    node = Node(
        code=f"node_{len(journal)}", plan="", parent=parent, is_buggy=buggy,
        metric=WorstMetricValue() if score is None else MetricValue(score, maximize=maximize),
        _term_out=[], exec_time=0, analysis="", **kwargs,
    )
    journal.append(node)
    return node


def _policy(journal, **search):
    agent = Agent.__new__(Agent)
    agent.journal = journal
    agent.acfg = SimpleNamespace(search=SimpleNamespace(**{
        "num_drafts": 2, "debug_prob": 0, "max_debug_depth": 3,
        "greedy_prob": .8, **search,
    }))
    return agent


@pytest.mark.parametrize("maximize", [False, True])
def test_tree_pool_has_one_good_winner_per_draft_and_ignores_uncommitted_children(maximize):
    journal = Journal()
    metric = lambda x: x if maximize else 1 - x
    a = _node(journal, metric(.90), maximize=maximize)
    b = _node(journal, metric(.85), maximize=maximize)
    c = _node(journal, buggy=True)
    _node(journal, buggy=True)  # A tree without a good node contributes nothing.
    b_best = _node(journal, metric(.88), parent=b, maximize=maximize)
    c_best = _node(journal, metric(.87), parent=c, maximize=maximize)
    _node(journal, metric(.90), parent=a, maximize=maximize)  # First wins ties.
    _node(journal, metric(.88), parent=b_best, maximize=maximize)
    _node(journal, metric(.99), parent=b, maximize=maximize, buggy=True)
    Node(code="uncommitted", parent=b, is_buggy=False,
         metric=MetricValue(metric(1.0), maximize=maximize))
    # Measurement-only scores cannot change the pool.
    b.test_metric = b.sim_metric = metric(1.0)
    assert journal.get_best_nodes_per_tree() == [a, b_best, c_best]
    assert journal.get_best_node() is a


def test_tree_pool_uses_existing_metric_direction_grouping():
    journal = Journal()
    root = _node(journal, .90)
    local_best = _node(journal, .10, parent=root, maximize=False)
    _node(journal, .20, parent=root, maximize=False)
    other = _node(journal, .80)
    assert journal.get_best_nodes_per_tree() == [local_best, other]


@pytest.mark.parametrize("draw,expected", [
    (.0, "greedy"), (.7999, "greedy"), (.8, "tree_best"), (.9999, "tree_best"),
])
def test_improve_parent_uses_configured_eighty_twenty_split(monkeypatch, draw, expected):
    journal = Journal()
    global_best = _node(journal, .90)
    other = _node(journal, .85)
    agent = _policy(journal)
    monkeypatch.setattr("aide.agent.random.random", lambda: draw)
    choose = Mock(return_value=other)
    monkeypatch.setattr("aide.agent.random.choice", choose)
    assert agent.search_policy() is (global_best if expected == "greedy" else other)
    assert agent._parent_selection == expected
    if expected == "tree_best":
        choose.assert_called_once_with([global_best, other])
    else:
        choose.assert_not_called()


@pytest.mark.parametrize("greedy_prob", [0, 1])
def test_probability_endpoints_restore_greedy_or_always_sample_trees(monkeypatch, greedy_prob):
    journal = Journal()
    global_best = _node(journal, .90)
    other = _node(journal, .85)
    agent = _policy(journal, greedy_prob=greedy_prob)
    monkeypatch.setattr("aide.agent.random.random", lambda: .5)
    monkeypatch.setattr("aide.agent.random.choice", lambda nodes: nodes[-1])
    assert agent.search_policy() is (global_best if greedy_prob else other)


def test_tree_best_can_be_revisited_and_changes_after_child_or_real_result(monkeypatch):
    journal = Journal()
    _node(journal, .90)
    other = _node(journal, .85)
    agent = _policy(journal, greedy_prob=0)
    monkeypatch.setattr("aide.agent.random.choice", lambda nodes: nodes[-1])
    # No UCB disagreement or real-validation provenance is required.
    assert not other.real_won and other.ucb_selection is None
    assert agent.search_policy() is other
    _node(journal, .84, parent=other, parent_selection="tree_best")
    assert agent.search_policy() is other
    improved = _node(journal, .88, parent=other, parent_selection="tree_best")
    assert agent.search_policy() is improved  # Still below the global .90.
    improved.real_won = True
    improved.metric = MetricValue(.83, maximize=True)
    assert agent.search_policy() is other
    agent.journal = loads_json(dumps_json(journal), Journal)
    assert agent.search_policy().id == other.id
    assert agent.journal.nodes[-1].parent_selection == "tree_best"


def test_ucb_disagreement_below_its_tree_best_gets_no_special_followup(monkeypatch):
    journal = Journal()
    best = _node(journal, .90)
    _node(journal, .85, parent=best, real_won=True, csv_exists=True,
          submission_validation="valid", submission_sha256="validated",
          ucb_selection={"stage": "improve", "rule": "ucb", "winner": 1, "greedy_winner": 0})
    agent = _policy(journal, num_drafts=1, greedy_prob=0)
    monkeypatch.setattr("aide.agent.random.choice", lambda nodes: nodes[0])
    assert agent.search_policy() is best
    assert agent._parent_selection == "tree_best"


def test_drafting_then_debugging_precede_improvement(monkeypatch):
    journal = Journal()
    good = _node(journal, .90)
    broken = _node(journal, buggy=True)
    agent = _policy(journal, num_drafts=3, debug_prob=.5)
    monkeypatch.setattr("aide.agent.random.random", lambda: .49)
    assert agent.search_policy() is None
    assert agent._parent_selection == "draft"
    agent.acfg.search.num_drafts = 2
    assert agent.search_policy() is broken
    assert agent._parent_selection == "debug"
    monkeypatch.setattr("aide.agent.random.random", lambda: .5)
    assert agent.search_policy() is good  # Debug gate is strict.
    assert agent._parent_selection == "greedy"


@pytest.mark.parametrize("depth,debug", [(2, True), (3, False)])
def test_debug_stops_at_automind_depth_limit(monkeypatch, depth, debug):
    journal = Journal()
    broken = _node(journal, buggy=True)
    for _ in range(depth):
        broken = _node(journal, parent=broken, buggy=True)
    agent = _policy(journal, num_drafts=1, debug_prob=1)
    monkeypatch.setattr("aide.agent.random.random", lambda: 0)
    assert broken.debug_depth == depth
    assert agent.search_policy() is (broken if debug else None)
    assert agent._parent_selection == ("debug" if debug else "draft")


@pytest.mark.parametrize("value", [-.1, 1.1, float("nan"), float("inf"), None, True])
def test_invalid_greedy_probability_fails_before_generation(value):
    agent = Agent.__new__(Agent)
    agent.acfg = SimpleNamespace(search=SimpleNamespace(greedy_prob=value))
    with pytest.raises(ValueError, match="agent.search.greedy_prob"):
        agent._resolve_stage_policies()


@pytest.mark.parametrize("beta", [0, 1])
def test_tree_parent_selection_keeps_ucb_candidate_selection_and_audit(tmp_path, monkeypatch, beta):
    agent, scorer, _ = _agent(tmp_path, beta=beta)
    agent.cfg.workspace_dir = tmp_path / "workspace"
    agent.cfg.workspace_dir.mkdir()
    agent.cfg.exec.simulator = agent.cfg.exec.simulator_only = True
    agent.acfg.check_data_leakage = agent.acfg.record_test_metric = False
    agent.acfg.search.num_drafts = 2
    agent.acfg.search.debug_prob = 0
    agent.acfg.search.greedy_prob = 0
    agent.data_preview = "test data"
    agent._improve_prompt = Mock(return_value={"task": "improve"})
    agent._review = Mock(side_effect=lambda code, output, **kwargs: {
        "is_bug": False, "has_csv_submission": True, "metric_trustworthy": True,
        "metric": float(output.split(": ")[1]), "lower_is_better": False,
        "summary": "Review of the supplied output.",
    })
    global_best = _node(agent.journal, .92, real_won=True)
    other = _node(agent.journal, .70)
    monkeypatch.setattr("aide.agent.random.choice", lambda nodes: nodes[-1])
    callback = Mock(side_effect=AssertionError("reuse the selected prediction"))
    agent.step(callback)
    child = agent.journal.nodes[-1]
    assert child.parent is other and child.parent_selection == "tree_best"
    assert child.code == CODES[beta][1]
    assert child.ucb_selection["winner"] == beta
    assert child.ucb_selection["greedy_winner"] == 0
    assert child.is_ucb_exploration is bool(beta)
    assert not child.real_won
    assert agent.journal.get_best_node() is global_best
    assert agent.search_policy() is child
    agent.acfg.search.greedy_prob = 1
    assert agent.search_policy() is global_best
    agent._improve_prompt.assert_called_once_with(other)
    assert agent._generate_n_candidates.call_args.args[1] == 4
    assert scorer.call_count == 4 * beta
    callback.assert_not_called()
    agent._judge_query.assert_not_called()
    agent._grade_test_submission.assert_not_called()
