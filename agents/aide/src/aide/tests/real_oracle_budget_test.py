"""Real-score selection and completed-step budgets, without model or GPU calls."""
import json
import logging
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest
from omegaconf import OmegaConf

from aide import agent as agent_mod, run as run_mod
from aide.journal import Node


def selector(tmp_path, lower=False):
    agent = object.__new__(agent_mod.Agent)
    agent.acfg = NS(use_observed_metric_argmax_pick=True, use_best_of_n_listwise=True,
                    eval_lower_is_better=lower)
    agent.task_desc = "task"
    agent._judge_query = Mock(side_effect=AssertionError("Unexpected judge call"))
    return agent


def outcome(metric, *, usable=True, predicted=False, trustworthy=True):
    return NS(usable=usable, predicted=predicted, term_out="actual validation output",
              review={"metric": metric, "metric_trustworthy": trustworthy},
              result=NS(predicted_metric=999, test_metric=999))


@pytest.mark.parametrize("lower,expected", [(False, 1), (True, 0)])
def test_real_metric_direction_ignores_predictions_and_private_scores(tmp_path, lower, expected):
    agent = selector(tmp_path, lower)
    outcomes = [outcome(0), outcome(.8), outcome(1, predicted=True), outcome(100, usable=False)]
    assert agent._select([("p", "pass")] * 4, outcomes, tmp_path) == expected
    audit = json.loads((tmp_path / "listwise.json").read_text())
    assert audit["scores"] == {"0": 0, "1": .8}
    assert audit["metric_source"] == "real_candidate_review"
    agent._judge_query.assert_not_called()


@pytest.mark.parametrize("values,lower,expected,reason", [
    ([.4, .4], False, 0, None),
    ([float("nan"), .6], False, 1, None),
    ([None, float("inf")], False, 0, "no_finite_observed_metric"),
    ([.2, .8], None, 0, "unknown_metric_direction"),
])
def test_real_metric_ties_and_missing_evidence(tmp_path, values, lower, expected, reason):
    agent = selector(tmp_path, lower)
    assert agent._select([("p", "pass")] * 2, [outcome(v) for v in values], tmp_path) == expected
    assert json.loads((tmp_path / "listwise.json").read_text())["fallback_reason"] == reason


def test_untrustworthy_real_metric_is_not_ranked(tmp_path):
    outcomes = [outcome(100, trustworthy=False), outcome(.3)]
    assert selector(tmp_path)._select([("p", "pass")] * 2, outcomes, tmp_path) == 1


def test_all_failed_real_pool_retains_least_buggy_judge(tmp_path, monkeypatch):
    judge = Mock(return_value=(1, "least_buggy"))
    monkeypatch.setattr(agent_mod, "listwise_pick", judge)
    outcomes = [outcome(10, usable=False), outcome(None, usable=False)]
    assert selector(tmp_path)._select([("p", "pass")] * 2, outcomes, tmp_path) == 1
    assert judge.call_args.kwargs == {"predicted": False, "all_failed": True}


@pytest.mark.parametrize("mode,expected_attempts", [("completed", 3), ("journal", 2)])
def test_post_commit_failure_does_not_spend_completed_step_budget(tmp_path, monkeypatch, mode, expected_attempts):
    cfg = NS(agent=NS(time_limit=10000, steps=2, steps_count_mode=mode),
             exec=OmegaConf.create({"simulator": False, "timeout": 10}),
             log_dir=tmp_path / "logs", workspace_dir=tmp_path / "workspace",
             exp_name="completed-step-test", log_level="ERROR")
    cfg.workspace_dir.mkdir()
    interpreter, save = Mock(), Mock()
    attempts = []

    def create_agent(**kwargs):
        def step(**_):
            kwargs["journal"].append(Node(code="pass", is_buggy=True))
            attempts.append(len(kwargs["journal"]))
            if len(attempts) == 1:
                raise RuntimeError("failure after journal commit")
        return NS(step=step, journal=kwargs["journal"])

    monkeypatch.setattr(run_mod, "load_cfg", lambda: cfg)
    monkeypatch.setattr(run_mod, "load_task_desc", lambda _: "task")
    monkeypatch.setattr(run_mod, "prep_agent_workspace", lambda _: None)
    monkeypatch.setattr(run_mod, "Interpreter", Mock(return_value=interpreter))
    monkeypatch.setattr(run_mod, "Agent", create_agent)
    monkeypatch.setattr(run_mod, "save_run", save)
    monkeypatch.setattr(run_mod, "journal_to_string_tree", lambda _: "tree")
    monkeypatch.setattr(run_mod.os, "_exit", Mock(side_effect=SystemExit(0)))
    monkeypatch.setattr(run_mod.logging, "shutdown", lambda: None)
    logger = logging.getLogger("aide")
    previous = set(logger.handlers)
    try:
        with pytest.raises(SystemExit):
            run_mod.run()
        run_mod.os._exit.assert_called_once_with(0)
        assert len(attempts) == expected_attempts
        interpreter.cleanup_session.assert_called_once_with()
    finally:
        for handler in set(logger.handlers) - previous:
            logger.removeHandler(handler)
            handler.close()


def test_completed_step_deadline_shortfall_exits_failed(tmp_path, monkeypatch):
    cfg = NS(agent=NS(time_limit=10000, steps=69, steps_count_mode="completed"),
             exec=OmegaConf.create({"simulator": False, "timeout": 10}),
             log_dir=tmp_path / "logs", workspace_dir=tmp_path / "workspace",
             exp_name="deadline-test", log_level="ERROR")
    cfg.workspace_dir.mkdir()
    monkeypatch.setenv("AIDE_SHUTDOWN_RESERVE_S", "10001")
    monkeypatch.setattr(run_mod, "load_cfg", lambda: cfg)
    monkeypatch.setattr(run_mod, "load_task_desc", lambda _: "task")
    monkeypatch.setattr(run_mod, "prep_agent_workspace", lambda _: None)
    interpreter, agent = Mock(), Mock()
    monkeypatch.setattr(run_mod, "Interpreter", Mock(return_value=interpreter))
    def create_agent(**kwargs):
        agent.journal = kwargs["journal"]
        return agent
    monkeypatch.setattr(run_mod, "Agent", create_agent)
    monkeypatch.setattr(run_mod, "save_run", Mock())
    monkeypatch.setattr(run_mod.os, "_exit", Mock(side_effect=SystemExit(1)))
    monkeypatch.setattr(run_mod.logging, "shutdown", lambda: None)
    logger = logging.getLogger("aide")
    previous = set(logger.handlers)
    try:
        with pytest.raises(SystemExit):
            run_mod.run()
        run_mod.os._exit.assert_called_once_with(1)
        agent.step.assert_not_called()
        interpreter.cleanup_session.assert_called_once_with()
    finally:
        for handler in set(logger.handlers) - previous:
            logger.removeHandler(handler)
            handler.close()
