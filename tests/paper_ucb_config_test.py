"""The paper UCB profiles preserve the shared experiment controls."""

import copy
import json
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest
import yaml
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "agents/aide/src"))
from aide.agent import Agent, CandidateOutcome
from aide.interpreter import ExecutionResult
from aide.journal import Journal
from aide.ucb import resolve_config
from aide.utils.config import Config, _load_cfg


CONFIGS = Path(__file__).resolve().parents[1] / "configs"
SUITE = yaml.safe_load((CONFIGS / "paper.yaml").read_text())
UCB_ARMS = [arm for arm in SUITE["arms"]
            if arm["id"] in {"debug_pclean", "bon_improve_ucb_beta05"}]


def resolve_profile(path):
    # Same extends/remove/overrides format as the selected bench's builder.
    raw = yaml.safe_load(path.read_text())
    if "extends" not in raw:
        return copy.deepcopy(raw["profile"])
    profile = resolve_profile(path.parent / raw["extends"])
    for section, keys in raw.get("remove", {}).items():
        for key in keys:
            profile[section].pop(key, None)
    for section, changes in raw.get("overrides", {}).items():
        profile.setdefault(section, {}).update(changes)
    return profile


@pytest.mark.parametrize("arm", UCB_ARMS, ids=lambda arm: arm["id"])
def test_profiles_have_exact_treatment_deltas_and_resolve_all_stage_betas(arm, tmp_path):
    profile = resolve_profile(CONFIGS / arm["configuration"])
    base = resolve_profile(CONFIGS / "policies/debug_pclean.yaml")
    hybrid = arm["policy"] == "bon_improve_ucb_beta05"
    beta = 1.0 if arm["id"].endswith("beta1") else 0.5
    expected_changes = {
        "agent.ucb.all_failed_fallback": "judge",
        "agent.ucb.pool_policy": "bon",
    }
    if hybrid:
        expected_changes.update({
            "agent.stage_policy.draft": "parallel",
            "agent.stage_policy.debug": "parallel",
            "agent.use_predicted_metric_argmax_pick": True,
        })
    if beta == 1:
        expected_changes.update({f"agent.ucb.{stage}beta": beta
                                 for stage in ["", "draft.", "debug.", "improve."]})
    expected = copy.deepcopy(base)
    expected["kwargs"].update(expected_changes)
    assert profile == expected

    cli = OmegaConf.from_dotlist([f"{key}={value}" for key, value in profile["kwargs"].items()])
    raw = _load_cfg(use_cli_args=False)
    raw.data_dir = str(tmp_path)
    raw.exp_name = "ucb-bon-config-test"
    cfg = OmegaConf.merge(OmegaConf.structured(Config), raw, cli)
    cfg.log_dir = str(tmp_path)
    for stage in ["draft", "debug", "improve"]:
        options = resolve_config(cfg.agent.ucb, stage)
        assert options.beta == beta and options.sigma == "p_clean"
        assert options.selection_rule == "ucb" and not options.metric_posterior.enabled
        assert options.pool_policy == "bon"
        assert options.all_failed_fallback == "judge"
    agent = Agent("test task", cfg, Journal(), sim_predict=lambda code: None,
                  sim_p_clean=lambda code: {"p_clean": 0.9})
    assert agent.stage_policy == {
        "draft": "parallel" if hybrid else "ucb",
        "debug": "parallel" if hybrid else "ucb",
        "improve": "ucb",
    }
    assert cfg.agent.time_limit == 43200
    assert cfg.agent.real_validation_interval == 1 and cfg.agent.best_of_n_size == 4
    assert cfg.exec.simulator and cfg.exec.simulator_only
    assert not cfg.agent.execution_gate.enabled
    assert cfg.agent.submission_reject_on_model and cfg.agent.leakage_reject_on_model
    assert cfg.agent.submission_review_temperature == 0.0
    assert profile["env_vars"]["AIDE_SHUTDOWN_RESERVE_S"] == "480"
    if hybrid:
        bon = resolve_profile(CONFIGS / "policies/bon_metric_argmax.yaml")
        bon_cli = OmegaConf.from_dotlist([f"{key}={value}" for key, value in bon["kwargs"].items()])
        bon_cfg = OmegaConf.merge(OmegaConf.structured(Config), raw, bon_cli)
        actual = OmegaConf.to_container(cfg.agent)
        expected = OmegaConf.to_container(bon_cfg.agent)
        del actual["ucb"], expected["ucb"]
        actual["stage_policy"]["improve"] = "parallel"
        assert actual == expected
        assert cfg.exec == bon_cfg.exec and profile["env_vars"] == bon["env_vars"]

    # Exercise each resolved profile: either review flag must leave WM metrics
    # eligible. Equal confidence makes the highest predicted metric win.
    cfg.agent.eval_lower_is_better = False
    agent._judge_query = Mock(side_effect=AssertionError("usable pool needs no judge"))
    candidates = [(f"plan {i}", f"candidate_{i} = True") for i in range(4)]
    metrics = [0.9, 0.8, 0.7, 0.6]
    reviews = [
        {"is_bug": True, "metric_trustworthy": True},
        {"is_bug": False, "metric_trustworthy": False},
        {"is_bug": True, "metric_trustworthy": False},
        {"is_bug": False, "metric_trustworthy": True},
    ]
    outcomes = [
        CandidateOutcome(
            ExecutionResult([f"Validation: {metric}\n"], 0, None,
                            predicted_submission_exists="true", predicted_metric=metric),
            True, "true", True, review,
        )
        for metric, review in zip(metrics, reviews)
    ]
    for stage, policy in agent.stage_policy.items():
        step_dir = tmp_path / stage
        step_dir.mkdir()
        if policy == "ucb":
            assert agent._select_ucb(candidates, outcomes, step_dir, stage, None) == 0
            audit = json.loads((step_dir / "ucb.json").read_text())
            assert all(row["eligible"] for row in audit["candidates"])
            assert [row["metric"] for row in audit["candidates"]] == metrics
            assert all(row["metric_source"] == "world_model" for row in audit["candidates"])
        else:
            assert agent._select(candidates, outcomes, step_dir,
                                 selector="predicted_metric_argmax") == 0
    agent._judge_query.assert_not_called()
