"""Random verification changes only discretionary checks, with replayable draws."""

import copy
import hashlib
import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "agents/aide/src"))
from aide.execution_gate import ExecutionGateConfig, random_gate_decision, validate_execution_gate
from aide.journal import Journal, Node
from aide.run import validate_on_best
from aide.utils.config import Config, _load_cfg, prep_cfg
from aide.utils.execution_cost import ExecutionCostRecorder
from aide.utils.metric import MetricValue
from aide.utils.serialize import dumps_json, loads_json
from aide_simulator_acceptance_test import _agent, _Real


def _gate(probability=0.5):
    return ExecutionGateConfig(
        enabled=True,
        mode="random",
        random_seed=123,
        random_check_probability=probability,
    )


def _trial(
    tmp_path,
    monkeypatch,
    probability=0.5,
    *,
    stage="improve",
    exc=None,
    confidence=0.99,
    policy="parallel",
    best=False,
):
    monkeypatch.setenv("COMPETITION_ID", "test-task")
    monkeypatch.setenv("AIDE_RUN_SEED", "38")
    agent = _agent(tmp_path)
    agent.acfg.execution_gate = _gate(probability)
    agent.stage_policy = {s: policy for s in ("draft", "debug", "improve")}
    prior = agent.journal.nodes[0]
    prior.metric = MetricValue(0.5 if best else 0.95, maximize=True)
    parent = None if stage == "draft" else prior
    if stage == "debug":
        parent = Node(code="broken parent", is_buggy=True)
        agent.journal.append(parent)
    node = Node(
        code="selected program",
        parent=parent,
        is_buggy=exc is not None,
        exc_type=exc,
        sim_p_max=confidence,
        metric=MetricValue(0.9, maximize=True),
    )
    agent.journal.append(node)
    agent.execution_cost_recorder = ExecutionCostRecorder(tmp_path / "logs")
    return agent, node, _Real(tmp_path, sim_only=True)


def _decisions(agent):
    return [
        row
        for line in agent.execution_cost_recorder.path.read_text().splitlines()
        if (row := json.loads(line))["event"] == "execution_decision"
    ]


@pytest.mark.parametrize("policy", ["parallel", "sequential", "pairwise", "fuse", "ucb"])
@pytest.mark.parametrize("probability", [0.0, 1.0])
def test_periodic_random_check_covers_all_methods(tmp_path, monkeypatch, policy, probability):
    agent, node, real = _trial(tmp_path, monkeypatch, probability, policy=policy)
    assert validate_on_best(agent, agent.journal, real, tmp_path, periodic=True) is bool(
        probability
    )
    assert real.calls == int(probability)
    decision = _decisions(agent)[0]
    assert decision["gate_mode"] == "random"
    assert decision["gate_score_type"] == "uniform_random"
    assert decision["gate_check_probability"] == probability
    assert decision["triggers"]["random"] is bool(probability)
    assert decision["triggers"]["confidence"] is False
    assert decision["code_sha256"] == hashlib.sha256(node.code.encode()).hexdigest()
    assert decision["gate_random_seed"] == 123
    assert decision["skip_reason"] == (None if probability else "random_skipped")
    if not probability:
        assert not node.real_won
        assert (tmp_path / "best_submission/submission.csv").read_text() == "id,pred\n0,0.25\n"


@pytest.mark.parametrize("stage", ["draft", "debug", "improve"])
@pytest.mark.parametrize("exc", [None, "ValueError"])
@pytest.mark.parametrize("probability", [0.0, 0.5, 1.0])
def test_one_probability_applies_to_all_stages_and_outcomes(
    tmp_path, monkeypatch, stage, exc, probability
):
    agent, node, real = _trial(tmp_path, monkeypatch, probability, stage=stage, exc=exc)
    ran = validate_on_best(agent, agent.journal, real, tmp_path, periodic=True)
    decision = _decisions(agent)[0]
    assert decision["gate_check_probability"] == probability
    expected = decision["gate_random_draw"] < probability
    assert ran is expected
    assert real.calls == int(expected)
    assert decision["stage"] == stage
    assert decision["gate_outcome"] == ("success" if exc is None else "failure")


@pytest.mark.parametrize("missing", [None, True, float("nan"), float("inf"), -0.1, 1.1, "0.9"])
def test_missing_confidence_is_checked_even_at_zero_rate(tmp_path, monkeypatch, missing):
    agent, node, real = _trial(tmp_path, monkeypatch, 0, confidence=missing)
    assert validate_on_best(agent, agent.journal, real, tmp_path, periodic=True)
    assert real.calls == 1
    decision = _decisions(agent)[0]
    assert decision["triggers"]["missing_confidence"] is True
    assert "gate_random_draw" not in decision


@pytest.mark.parametrize("best,force", [(True, False), (False, True)])
def test_mandatory_checks_bypass_zero_rate(tmp_path, monkeypatch, best, force):
    agent, node, real = _trial(tmp_path, monkeypatch, 0, best=best)
    monkeypatch.setattr(
        "aide.real_validation.random_gate_decision",
        Mock(side_effect=AssertionError("no coin for mandatory checks")),
    )
    assert validate_on_best(agent, agent.journal, real, tmp_path, periodic=True, force=force)
    assert real.calls == 1
    assert "gate_random_draw" not in _decisions(agent)[0]
    if best:
        assert node.real_won
        assert (tmp_path / "best_solution/node_id.txt").read_text() == node.id


@pytest.mark.parametrize("reason", ["predicted_timeout", "already_real", "deadline_reserve"])
def test_vetoes_prevent_random_draws_and_real_execution(tmp_path, monkeypatch, reason):
    agent, node, real = _trial(tmp_path, monkeypatch, 1)
    kwargs = {"periodic": True}
    if reason == "predicted_timeout":
        node.exc_type = "TimeoutError"
        kwargs["force"] = True
    elif reason == "already_real":
        node.real_won = True
    else:
        kwargs["deadline"] = 0
    monkeypatch.setattr(
        "aide.real_validation.random_gate_decision",
        Mock(side_effect=AssertionError("no coin after a veto")),
    )
    assert not validate_on_best(agent, agent.journal, real, tmp_path, **kwargs)
    assert real.calls == 0
    assert _decisions(agent)[0]["skip_reason"] == reason


def test_draw_survives_resume_and_does_not_depend_on_score_or_global_rng():
    cfg = SimpleNamespace(execution_gate=_gate())
    node = SimpleNamespace(step=7, stage_name="improve", exc_type=None, sim_p_max=0.1)
    context = dict(task="task", trial_seed=38, method="parallel")
    state = random.getstate()
    expected, fields = random_gate_decision(cfg, node, **context)
    restored = SimpleNamespace(**json.loads(json.dumps(vars(node))))
    restored.sim_p_max = 0.999
    restored.stage_name = "debug"
    restored.exc_type = "ValueError"
    assert random_gate_decision(cfg, restored, **context) == (expected, fields)
    assert random.getstate() == state
    for key, value in (("task", "other-task"), ("trial_seed", 39), ("method", "fuse")):
        assert (
            random_gate_decision(cfg, restored, **(context | {key: value}))[1]["gate_random_draw"]
            != fields["gate_random_draw"]
        )
    # Equality skips; changing the rate never redraws the coin.
    cfg.execution_gate.random_check_probability = fields["gate_random_draw"]
    assert random_gate_decision(cfg, restored, **context)[0] is False


def test_repeated_validation_and_interpreter_retry_reuse_one_decision(tmp_path, monkeypatch):
    agent, node, real = _trial(tmp_path, monkeypatch, 1)
    original_run = real.run
    contexts = []

    def flaky_run(*args, **kwargs):
        contexts.append(copy.deepcopy(real.execution_cost_context))
        if len(contexts) == 1:
            raise RuntimeError("transient interpreter failure")
        return original_run(*args, **kwargs)

    real.run = flaky_run
    assert validate_on_best(agent, agent.journal, real, tmp_path, periodic=True)
    assert len(contexts) == 2
    for key in ("gate_random_draw", "gate_check_probability", "gate_random_key"):
        assert contexts[0][key] == contexts[1][key]
    assert not validate_on_best(agent, agent.journal, real, tmp_path, periodic=True)
    assert len(contexts) == 2


def test_skipped_decision_is_reproducible_on_repeated_validation(tmp_path, monkeypatch):
    agent, node, real = _trial(tmp_path, monkeypatch, 0)
    assert not validate_on_best(agent, agent.journal, real, tmp_path, periodic=True)
    agent.journal = loads_json(dumps_json(agent.journal), Journal)
    assert not validate_on_best(agent, agent.journal, real, tmp_path, periodic=True)
    decisions = _decisions(agent)
    assert decisions[0]["gate_random_draw"] == decisions[1]["gate_random_draw"]
    assert decisions[0]["gate_random_key"] == decisions[1]["gate_random_key"]


def test_random_mode_scores_the_selected_program_and_retains_simulated_feedback(tmp_path):
    from aide.interpreter import ExecutionResult

    agent = _agent(tmp_path)
    agent.acfg.execution_gate = _gate(0)
    agent.stage_policy = {s: "parallel" for s in ("draft", "debug", "improve")}
    agent.journal.nodes[0].metric = MetricValue(0.95, maximize=True)
    agent._exc_scores = Mock(
        return_value={"max_p": 0.91, "top_exc": "ValueError", "p_top_exc": 0.09}
    )
    agent.step(
        lambda *args: ExecutionResult(
            ["Validation score: 0.9"], 0, None, predicted_submission_exists="true"
        )
    )
    node = agent.journal.nodes[-1]
    agent._exc_scores.assert_called_once()
    assert node.sim_p_max == 0.91
    real = _Real(tmp_path, sim_only=True)
    assert not validate_on_best(agent, agent.journal, real, tmp_path, periodic=True)
    assert real.calls == 0
    assert node.metric.value == 0.9
    assert "Validation score: 0.9" in node.term_out


@pytest.mark.parametrize("invalid", [True, -0.1, 1.1, float("nan"), float("inf"), None, "0.5"])
def test_invalid_probability_fails_validation(invalid):
    gate = _gate()
    gate.random_check_probability = invalid
    with pytest.raises(ValueError, match="random_check_probability"):
        validate_execution_gate(SimpleNamespace(execution_gate=gate))


@pytest.mark.parametrize("change", ["missing", "bad_mode", "bad_seed"])
def test_incomplete_or_invalid_random_configuration_fails(change):
    gate = _gate()
    if change == "missing":
        gate = ExecutionGateConfig(enabled=True, mode="random")
    elif change == "bad_mode":
        gate.mode = "rndom"
    else:
        gate.random_seed = -1
    with pytest.raises(ValueError):
        validate_execution_gate(SimpleNamespace(execution_gate=gate))


def test_random_schema_roundtrip_and_cost_metadata(tmp_path, monkeypatch):
    cfg = _load_cfg(use_cli_args=False)
    cfg.data_dir = str(tmp_path)
    cfg.goal = "Test random verification."
    cfg.exp_name = "random-gate-test"
    cfg.log_dir = str(tmp_path / "logs")
    cfg.workspace_dir = str(tmp_path / "workspaces")
    cfg.exec.simulator = True
    cfg.agent.execution_gate = OmegaConf.structured(_gate())
    cfg.agent.real_validation_interval = 1
    cfg = prep_cfg(cfg)
    restored = OmegaConf.merge(
        OmegaConf.structured(Config), OmegaConf.create(OmegaConf.to_yaml(cfg))
    )
    validate_execution_gate(restored.agent)
    monkeypatch.setenv("COMPETITION_ID", "task")
    monkeypatch.setenv("AIDE_RUN_SEED", "38")
    recorder = ExecutionCostRecorder.from_config(restored, run_start=0)
    metadata = json.loads(recorder.path.read_text().splitlines()[0])
    assert metadata["execution_mode"] == "random_check+periodic"
    assert metadata["outcome_thresholds"] is None
    assert metadata["gate_score_type"] == "uniform_random"
    assert metadata["gate_check_probability"] == 0.5
    assert metadata["gate_random_seed"] == 123


def test_random_gate_requires_scorer_even_without_thresholds():
    from aide.agent import Agent

    instance = object.__new__(Agent)
    instance.acfg = SimpleNamespace(execution_gate=_gate())
    instance.cfg = SimpleNamespace(exec=SimpleNamespace(simulator=True))
    instance._exc_scores = None
    with pytest.raises(ValueError, match="exception-posterior scorer"):
        instance._resolve_stage_policies()
