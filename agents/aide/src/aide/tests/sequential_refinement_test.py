"""Sequential revisions retain feedback, use the full budget, and select once."""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from omegaconf import OmegaConf

from aide import agent as agent_mod
from aide.agent import Agent, STAGES
from aide.interpreter import ExecutionResult
from aide.journal import Journal, Node
from aide.utils.config import AgentConfig


CANDIDATES = [(f"plan {i}", f"candidate_{i} = True") for i in range(4)]


def _result(metric=0.5, *, exc_type=None, submission="true", real_won=False):
    return ExecutionResult(
        term_out=[f"report: metric={metric}; exception={exc_type}\n"],
        exec_time=0.1,
        exc_type=exc_type,
        predicted_metric=metric,
        predicted_submission_exists=submission,
        real_won=real_won,
        observed_submission_exists=submission == "true" if real_won else None,
    )


def _agent(tmp_path, results, *, lower=False):
    config = OmegaConf.load(Path(agent_mod.__file__).parent / "utils/config.yaml")
    config.agent = OmegaConf.merge(OmegaConf.structured(AgentConfig), config.agent)
    for stage in STAGES:
        config.agent.stage_policy[stage] = "sequential"
    config.agent.best_of_n_size = len(results)
    config.agent.eval_lower_is_better = lower
    # Sequential uses WM feedback directly even when width-policy reviews are on.
    config.agent.bon_candidate_review = True
    config.agent.use_best_of_n_listwise = False
    config.agent.use_predicted_metric_argmax_pick = False
    config.agent.use_p_clean_argmax_pick = False
    config.agent.check_data_leakage = False
    config.agent.record_test_metric = False
    config.exec.simulator = True
    config.exec.simulator_only = False
    cfg = SimpleNamespace(
        agent=config.agent, exec=config.exec, log_dir=tmp_path,
        workspace_dir=tmp_path / "workspace",
    )
    cfg.workspace_dir.mkdir()
    predictions = {code: result for (_, code), result in zip(CANDIDATES, results)}
    predictor = Mock(side_effect=predictions.__getitem__)
    agent = Agent("Predict the task target using its official metric.", cfg, Journal(),
                  sim_predict=predictor)
    agent.data_preview = "Training and test tables."
    agent._generate_n_candidates = Mock(
        side_effect=[[candidate] for candidate in CANDIDATES[:len(results)]]
    )
    agent._judge_query = Mock(side_effect=AssertionError("Usable pools must not call the judge"))
    agent._review = Mock(side_effect=AssertionError("No per-candidate reviews in sequential"))
    agent._p_clean = Mock(side_effect=AssertionError("No confidence ranking"))
    agent._grade_test_submission = Mock(side_effect=AssertionError("No test grading in selection"))
    return agent


@pytest.mark.parametrize("stage", STAGES)
@pytest.mark.parametrize("lower,expected", [(False, 1), (True, 2)])
def test_revisions_use_previous_code_and_feedback_then_select_best_usable(
    tmp_path, stage, lower, expected,
):
    results = [
        _result(99, exc_type="ValueError", submission="false"),
        _result(0.9), _result(0.3),
        _result(100, exc_type="TimeoutError", submission="false"),
    ]
    agent = _agent(tmp_path, results, lower=lower)
    parent = Node(code="outer parent", is_buggy=stage == "debug")
    agent.journal.append(parent)
    original_prompt = {"task": "initial stage prompt"}
    plan, code, result = agent.generate_code(original_prompt, stage, parent, max_tokens=123)

    assert (plan, code) == CANDIDATES[expected]
    assert result is None
    assert agent._generate_n_candidates.call_count == agent._predict.call_count == 4
    calls = agent._generate_n_candidates.call_args_list
    assert all(c.args[1] == 1 and c.kwargs["max_tokens"] == 123 for c in calls)
    assert original_prompt == {"task": "initial stage prompt"}
    repair = calls[1].args[0]
    assert CANDIDATES[0][1] in repair["Previous (buggy) implementation"]
    assert results[0].term_out[0].strip() in repair["Execution output"]
    assert repair["Sequential candidate feedback"]["Exception"] == "ValueError"
    for i in (2, 3):
        improve = calls[i].args[0]
        assert CANDIDATES[i - 1][1] in improve["Previous solution"]["Code"]
        assert improve["Previous solution"]["Validation Metric"] == str(results[i - 1].predicted_metric)
        assert improve["Previous solution"]["Metric direction"] == (
            "lower is better" if lower else "higher is better"
        )
        assert improve["Previous solution"]["Evidence source"] == "World-model prediction"
    assert parent.children == set() and agent.journal.nodes == [parent]
    agent._judge_query.assert_not_called()
    agent._review.assert_not_called()
    agent._p_clean.assert_not_called()
    agent._grade_test_submission.assert_not_called()

    log = tmp_path / "bon/step_0"
    audit = json.loads((log / "sequential.json").read_text())
    assert audit["mode"] == "refinement" and audit["attempts"] == audit["budget"] == 4
    assert audit["picked_attempt"] == audit["accepted_attempt"] == expected
    assert audit["exhausted"] is False
    assert (log / "picked.txt").read_text() == str(expected)
    for i, call in enumerate(calls):
        assert json.loads((log / f"cand_{i}/prompt.json").read_text()) == call.args[0]
        assert json.loads((log / f"cand_{i}/result.json").read_text())["accepted"] == (i == expected)


@pytest.mark.parametrize("metrics,lower,expected", [
    ([0.1, 0.9, 0.3, 0.2], False, 1),
    ([-1, 0, 0, -2], False, 1),
    ([None, float("nan"), float("inf"), -0.3], False, 3),
    ([None, float("nan"), float("inf"), None], False, 0),
    ([0.1, 0.9, 0.3, 0.2], None, 0),
])
def test_full_budget_and_metric_fallbacks_override_global_selectors(
    tmp_path, metrics, lower, expected,
):
    agent = _agent(tmp_path, [_result(m) for m in metrics], lower=lower)
    agent.acfg.use_p_clean_argmax_pick = True
    _, code, _ = agent.generate_code({}, "improve")
    assert code == CANDIDATES[expected][1]
    assert agent._predict.call_count == 4  # even when the first candidate is usable
    agent._judge_query.assert_not_called()
    agent._p_clean.assert_not_called()


@pytest.mark.parametrize("judge_fails,expected", [(False, 2), (True, 0)])
def test_all_unusable_uses_judge_even_with_global_judging_disabled(tmp_path, judge_fails, expected):
    agent = _agent(tmp_path, [_result(99, submission="false") for _ in range(4)])
    agent.acfg.use_p_clean_argmax_pick = True
    agent.acfg.use_line_frac_salvage = True
    agent._judge_query = (
        Mock(side_effect=RuntimeError("judge unavailable")) if judge_fails
        else Mock(return_value={"best_candidate": 2})
    )
    assert agent.generate_code({}, "debug")[1] == CANDIDATES[expected][1]
    agent._judge_query.assert_called_once()
    prompt = agent._judge_query.call_args.args[0]
    assert "All candidates failed" in prompt["Introduction"]
    assert all(f"Candidate {i}" in prompt for i in range(4))
    assert agent._predict.call_count == 4
    agent._p_clean.assert_not_called()
    audit = json.loads((tmp_path / "bon/step_0/sequential.json").read_text())
    assert audit["picked_attempt"] == expected and audit["accepted_attempt"] is None
    assert audit["exhausted"] is True


@pytest.mark.parametrize("oracle,real_won,reuse", [
    ("sim", False, True), ("sim", True, False), ("race", False, False),
])
def test_commit_reuses_only_eligible_simulator_predictions(tmp_path, oracle, real_won, reuse):
    results = [_result(submission="false") for _ in range(4)]
    results[1] = _result(0.9, real_won=real_won)
    agent = _agent(tmp_path, results)
    agent.oracle = oracle
    agent.cfg.exec.simulator_only = True
    _, code, result = agent.generate_code({}, "draft")
    assert code == CANDIDATES[1][1]
    assert result is (results[1] if reuse else None)
    assert agent._predict.call_count == 4
    agent._judge_query.assert_not_called()
    if real_won:
        prompt = agent._generate_n_candidates.call_args_list[2].args[0]
        assert prompt["Previous solution"]["Evidence source"] == "Real execution"
        assert "Validation Metric" not in prompt["Previous solution"]
        assert prompt["Sequential candidate feedback"]["Submission exists"] == "True"


def test_step_commits_one_node_without_repeating_the_selected_prediction(tmp_path):
    results = [_result(m) for m in (0.1, 0.9, 0.3, 0.2)]
    agent = _agent(tmp_path, results)
    agent.cfg.exec.simulator_only = True
    agent._draft_prompt = Mock(return_value={"task": "draft"})
    agent._review = Mock(return_value={
        "is_bug": False, "has_csv_submission": True, "metric": 0.9,
        "metric_trustworthy": True, "lower_is_better": False, "summary": "Clean prediction.",
    })
    execute = Mock(side_effect=AssertionError("The selected prediction must be reused"))
    agent.step(execute)
    assert len(agent.journal.nodes) == 1
    node = agent.journal.nodes[0]
    assert node.code == CANDIDATES[1][1] and node.parent is None
    assert node.metric.value == 0.9 and node.real_won is False
    assert agent._predict.call_count == 4
    agent._review.assert_called_once()  # selected node review only
    agent._judge_query.assert_not_called()
    execute.assert_not_called()


@pytest.mark.parametrize("stage", STAGES)
@pytest.mark.parametrize("first", [
    _result(None, exc_type="ValueError", submission="false"),
    _result(.7, submission="true"),
    _result(.7, submission="true", real_won=True),
])
def test_live_markdown_renderer_accepts_every_refinement_prompt(tmp_path, stage, first):
    """Exercise the live rendering boundary skipped by candidate-generation mocks."""
    from aide.backend.utils import compile_prompt_to_md
    agent = _agent(tmp_path, [first, _result(.9), _result(.3), _result(.1)])
    generated = iter(CANDIDATES)
    rendered = []

    def generate(prompt, n, max_tokens=None):
        rendered.append(compile_prompt_to_md(prompt))
        return [next(generated)]

    agent._generate_n_candidates = Mock(side_effect=generate)
    agent.generate_code({"Task": "Refine the solution."}, stage)
    assert len(rendered) == 4
    assert all("Sequential candidate feedback" in text for text in rendered[1:])
    assert "World-model prediction" in rendered[2]
