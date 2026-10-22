"""Regression checks for search-tree rollback, LLM budgets, and race evidence."""

import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import openai
import pytest
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "agents/aide/src"))
from aide import agent as agent_mod
from aide.agent import Agent, _candidate_is_usable
from aide.backend import backend_openai, utils as backend_utils
from aide.interpreter import ExecutionResult
from aide.journal import Journal, Node
from aide.parallel_interpreter import ParallelInterpreter
from aide.utils.config import AgentConfig
from aide.utils.metric import WorstMetricValue


REVIEW = {
    "is_bug": False,
    "has_csv_submission": True,
    "summary": "Validation score: 0.8",
    "metric": 0.8,
    "lower_is_better": False,
    "metric_trustworthy": True,
}


def make_agent(tmp_path, *, policy="single", race_predict=None):
    base = Path(agent_mod.__file__).parent
    config = OmegaConf.load(base / "utils/config.yaml")
    config.agent = OmegaConf.merge(OmegaConf.structured(AgentConfig), config.agent)
    config.agent.search.num_drafts = 1
    config.agent.search.debug_prob = 1.0
    config.agent.stage_policy.draft = policy
    config.agent.best_of_n_size = 2
    config.agent.bon_oracle = "race" if race_predict else "sim"
    config.agent.code.model = "gpt-test"
    config.agent.feedback.model = "gpt-test"
    cfg = SimpleNamespace(
        agent=config.agent, exec=config.exec, workspace_dir=tmp_path, log_dir=tmp_path / "logs"
    )
    agent = Agent("task", cfg, Journal(), race_predict=race_predict)
    agent.data_preview = "data already loaded"
    agent._check_data_leakage = Mock()
    return agent


@pytest.mark.parametrize("where", ["execution", "review"])
@pytest.mark.parametrize("error", [RuntimeError, TimeoutError, KeyboardInterrupt])
def test_aborted_child_leaves_parent_available_for_successful_retry(tmp_path, where, error):
    agent = make_agent(tmp_path)
    parent = Node(
        code="broken()",
        is_buggy=True,
        metric=WorstMetricValue(),
        analysis="Fix NameError",
        _term_out=["NameError"],
    )
    agent.journal.append(parent)
    agent._build_debug_prompt = Mock(return_value={"task": "debug"})
    agent.generate_code = Mock(return_value=("plan", "print(1)", None))
    result = ExecutionResult(["Validation score: 0.8"], 0, None, predicted_submission_exists="true")
    callback = Mock(return_value=result)
    agent._review = Mock(return_value=REVIEW.copy())
    failing_call = callback if where == "execution" else agent._review
    failing_call.side_effect = error("transient failure")

    with pytest.raises(error):
        agent.step(callback)

    assert agent.journal.nodes == [parent]
    assert parent.children == set()
    assert agent.search_policy() is parent
    assert agent.current_step == 0

    failing_call.side_effect = None
    agent.step(callback)
    child = agent.journal.nodes[-1]
    assert len(agent.journal) == 2
    assert child.parent is parent and parent.children == {child}
    assert child.is_buggy is False
    assert agent.current_step == 1


def invoke_query(agent, kind):
    if kind == "single":
        return agent.plan_and_code_query({"task": "draft"})
    if kind == "batch":
        return agent._generate_n_candidates({"task": "draft"}, 2)
    if kind == "judge":
        return agent._judge_query({"task": "choose"})
    return agent._review("print(0.8)", "Validation score: 0.8")


@pytest.mark.parametrize("kind", ["single", "batch", "judge", "review"])
def test_generation_and_review_receive_remaining_budget(tmp_path, monkeypatch, kind):
    agent = make_agent(tmp_path)
    agent.deadline = 1200.0
    monkeypatch.setenv("AIDE_SHUTDOWN_RESERVE_S", "180")
    monkeypatch.setattr(agent_mod.time, "time", lambda: 1000.0)

    def reply(**kwargs):
        if kwargs.get("func_spec") is not None:
            return REVIEW.copy()
        code = "```python\nprint(0.8)\n```"
        return [code] * kwargs["n"] if kwargs.get("n", 1) > 1 else code

    query = Mock(side_effect=reply)
    monkeypatch.setattr(agent_mod, "query", query)
    invoke_query(agent, kind)
    remaining = 170.0 if kind == "review" else 20.0
    assert query.call_args.kwargs["retry_max_time"] == remaining
    assert query.call_args.kwargs["retry_attempt_timeout"] == remaining
    assert "reserve_s" not in query.call_args.kwargs

    agent.deadline = 1000.0 + (30.0 if kind == "review" else 180.0)
    query.reset_mock()
    with pytest.raises(TimeoutError, match="deadline exhausted"):
        invoke_query(agent, kind)
    query.assert_not_called()


@pytest.mark.parametrize("kind", ["single", "batch"])
def test_parse_retries_share_run_deadline(tmp_path, monkeypatch, kind):
    agent = make_agent(tmp_path)
    clock, allowances = [1000.0], []
    agent.deadline = 1200.0
    monkeypatch.setenv("AIDE_SHUTDOWN_RESERVE_S", "180")
    monkeypatch.setattr(agent_mod.time, "time", lambda: clock[0])

    def reply(**kwargs):
        allowances.append(kwargs["retry_max_time"])
        clock[0] += 15.0
        answer = "please try again" if len(allowances) == 1 else "```python\nprint(1)\n```"
        return [answer] * kwargs["n"] if kwargs.get("n", 1) > 1 else answer

    monkeypatch.setattr(agent_mod, "query", reply)
    invoke_query(agent, kind)
    assert allowances == [20.0, 5.0]


def test_query_preserves_stricter_caller_limits(tmp_path, monkeypatch):
    agent = make_agent(tmp_path)
    agent.deadline = 1200.0
    monkeypatch.setenv("AIDE_SHUTDOWN_RESERVE_S", "180")
    monkeypatch.setattr(agent_mod.time, "time", lambda: 1000.0)
    query = Mock()
    monkeypatch.setattr(agent_mod, "query", query)
    agent._query(retry_max_time=7, retry_attempt_timeout=2)
    assert query.call_args.kwargs == {"retry_max_time": 7, "retry_attempt_timeout": 2}


def test_local_backend_outage_stops_at_agent_budget(tmp_path, monkeypatch):
    agent = make_agent(tmp_path)
    clock, timeouts = [0.0], []
    agent.deadline = 200.0
    monkeypatch.setenv("AIDE_SHUTDOWN_RESERVE_S", "180")
    monkeypatch.setenv("AIDE_OPENAI_TIMEOUT", "7")
    monkeypatch.setattr(agent_mod.time, "time", lambda: clock[0])
    monkeypatch.setattr(backend_utils.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(backend_utils.random, "uniform", lambda *_: 0.0)
    monkeypatch.setattr(backend_openai, "_apply_sampling_overrides", lambda _: None)

    def create(**kwargs):
        assert not {"retry_max_time", "retry_attempt_timeout"} & kwargs.keys()
        timeouts.append(kwargs["timeout"])
        clock[0] += kwargs["timeout"]
        raise openai.APITimeoutError(request=httpx.Request("POST", "http://test.invalid"))

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    monkeypatch.setattr(backend_openai, "_get_client_for_model", lambda _: client)
    with pytest.raises(openai.APITimeoutError):
        agent.plan_and_code_query({"task": "draft"})
    assert timeouts == [7.0, 7.0, 6.0]
    assert clock[0] == 20.0


@pytest.mark.parametrize("policy", ["parallel", "sequential"])
@pytest.mark.parametrize("bad_kind", ["missing", "empty", "directory", "custom_exception"])
def test_race_selection_uses_real_facts_after_workspace_cleanup(tmp_path, policy, bad_kind):
    calls = []

    def race_predict(code):
        calls.append(code)
        workspace = tmp_path / ("workspace_" + code)
        csv_path = workspace / "submission/submission.csv"
        csv_path.parent.mkdir(parents=True)
        if code == "good" or bad_kind == "custom_exception":
            csv_path.write_text("id,pred\n1,0.8\n")
        elif bad_kind == "empty":
            csv_path.touch()
        elif bad_kind == "directory":
            csv_path.mkdir()
        result = ExecutionResult(["Validation score: 0.8"], 0, None)
        if code == "bad" and bad_kind == "custom_exception":
            result.exc_type = "ArrowInvalid"
        # A stale prediction cannot override the actual file or exception.
        result.predicted_submission_exists = "true"
        real = SimpleNamespace(working_dir=workspace, run=Mock(return_value=result))
        simulator = Mock()
        race = ParallelInterpreter(real, simulator)
        try:
            return race.run(code)
        finally:
            race.close()
            shutil.rmtree(workspace)
            simulator.run.assert_not_called()

    agent = make_agent(tmp_path, policy=policy, race_predict=race_predict)
    candidates = [("bad plan", "bad"), ("good plan", "good")]
    agent._generate_n_candidates = (
        Mock(side_effect=[[candidate] for candidate in candidates])
        if policy == "sequential"
        else Mock(return_value=candidates)
    )
    agent._review = Mock(return_value=REVIEW.copy())
    agent._judge_query = Mock(side_effect=AssertionError("Only one candidate is usable"))
    plan, code, result = agent.generate_code({}, "draft")
    assert (plan, code, result) == ("good plan", "good", None)
    assert set(calls) == {"bad", "good"}
    agent._judge_query.assert_not_called()
    audits = [
        json.loads((tmp_path / f"logs/bon/step_0/cand_{i}/result.json").read_text())
        for i in range(2)
    ]
    assert [audit["is_predicted_usable"] for audit in audits] == [False, True]
    assert [audit["accepted"] for audit in audits] == [False, True]
    assert all(audit["real_won"] for audit in audits)
    assert audits[1]["observed_submission_exists"] is True
    if policy == "parallel":
        assert all(call.kwargs["predicted"] is False for call in agent._review.call_args_list)


@pytest.mark.parametrize("observed", [None, False, True])
def test_real_evidence_never_falls_back_to_submission_prediction(observed):
    result = ExecutionResult(
        [],
        0,
        None,
        real_won=True,
        predicted_submission_exists="true",
        observed_submission_exists=observed,
    )
    assert _candidate_is_usable(result) is (observed is True)
    result.exc_type = "ArrowInvalid"
    assert _candidate_is_usable(result) is False


@pytest.mark.parametrize("submission", [None, "true"])
def test_legacy_parser_garbage_does_not_establish_candidate_usability(submission):
    result = ExecutionResult(
        [], 0, "model.safetensors", predicted_submission_exists=submission
    )
    assert _candidate_is_usable(result) is False


@pytest.mark.parametrize("submission", [None, False, True, "", "tru", "True", [], {}])
def test_simulator_acceptance_requires_explicit_positive_csv_evidence(submission):
    result = ExecutionResult([], 0, None, predicted_submission_exists=submission)
    assert _candidate_is_usable(result) is False
