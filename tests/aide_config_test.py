"""Configuration preparation and optional exports must not derail search."""

import json
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest
from omegaconf import OmegaConf
from omegaconf.errors import ValidationError

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "agents/aide/src"))
from aide import ucb
from aide.journal import Journal, Node
from aide.utils import config
from aide.utils.metric import MetricValue


@pytest.fixture
def raw_cfg(tmp_path):
    cfg = config._load_cfg(use_cli_args=False)
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "train.csv").write_text("x,y\n1,2\n")
    cfg.data_dir = str(data_dir)
    cfg.goal = "Predict y."
    cfg.eval = "Minimize validation error."
    cfg.exp_name = "test-run"
    cfg.log_dir = str(tmp_path / "logs")
    cfg.workspace_dir = str(tmp_path / "workspaces")
    cfg.preprocess_data = False
    return cfg


@pytest.mark.parametrize("path_type", [str, Path])
def test_preparation_accepts_paths_and_leaves_template_reusable(raw_cfg, path_type):
    raw_cfg.data_dir = path_type(raw_cfg.data_dir)
    original = OmegaConf.to_container(raw_cfg)

    first = config.prep_cfg(raw_cfg)
    second = config.prep_cfg(raw_cfg)

    assert first.data_dir == Path(raw_cfg.data_dir).resolve()
    assert first.log_dir == second.log_dir == Path(raw_cfg.log_dir) / raw_cfg.exp_name
    assert (
        first.workspace_dir
        == second.workspace_dir
        == (Path(raw_cfg.workspace_dir) / raw_cfg.exp_name)
    )
    assert OmegaConf.to_container(raw_cfg) == original


def test_preparation_accepts_dataclass_input(raw_cfg):
    settings = OmegaConf.to_object(OmegaConf.merge(OmegaConf.structured(config.Config), raw_cfg))
    prepared = config.prep_cfg(settings)
    assert prepared.data_dir == Path(raw_cfg.data_dir)
    assert prepared.agent == OmegaConf.structured(settings.agent)
    assert settings.log_dir == Path(raw_cfg.log_dir)


@pytest.mark.parametrize("path_type", [str, Path])
def test_relative_data_paths_use_the_working_directory(raw_cfg, path_type, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    raw_cfg.data_dir = path_type("data")
    prepared = config.prep_cfg(raw_cfg)
    assert prepared.data_dir == tmp_path / "data"


def test_schema_fills_omitted_options_without_changing_search_overrides(raw_cfg):
    del raw_cfg.agent.ucb.debug
    raw_cfg.agent.stage_policy.improve = "ucb"
    raw_cfg.agent.ucb.improve.beta = 2.0
    raw_cfg.agent.search.debug_prob = "0.25"

    prepared = config.prep_cfg(raw_cfg)

    assert prepared.agent.ucb.debug == OmegaConf.structured(config.UCBConfig).debug
    assert prepared.agent.stage_policy.improve == "ucb"
    assert prepared.agent.ucb.improve.beta == 2.0
    assert prepared.agent.search.debug_prob == 0.25
    assert "debug" not in raw_cfg.agent.ucb


@pytest.mark.parametrize("simulator_only", [False, True])
def test_simulation_without_confidence_does_not_require_posterior_scoring(
    raw_cfg,
    simulator_only,
):
    from aide.agent import Agent

    agent = Agent.__new__(Agent)
    agent.cfg = raw_cfg
    agent.acfg = raw_cfg.agent
    agent.oracle = "sim"
    agent._exc_scores = None
    agent._sim_predict = Mock()
    agent._race_predict = None
    raw_cfg.exec.simulator = True
    raw_cfg.exec.simulator_only = simulator_only
    raw_cfg.agent.real_validation_p_max_threshold = None
    for stage in ["draft", "debug", "improve"]:
        raw_cfg.agent.stage_policy[stage] = "single"
    assert set(agent._resolve_stage_policies().values()) == {"single"}


def test_invalid_settings_fail_before_creating_directories(raw_cfg):
    raw_cfg.agent.search.debug_prob = "invalid"
    original = OmegaConf.to_container(raw_cfg)

    with pytest.raises(ValidationError, match="debug_prob"):
        config.prep_cfg(raw_cfg)

    assert not Path(raw_cfg.log_dir).exists()
    assert not Path(raw_cfg.workspace_dir).exists()
    assert OmegaConf.to_container(raw_cfg) == original


def test_predicted_metric_argmax_defaults_off_and_cli_enables_it(raw_cfg, monkeypatch):
    assert config.AgentConfig.use_predicted_metric_argmax_pick is False
    assert raw_cfg.agent.use_predicted_metric_argmax_pick is False
    monkeypatch.setattr(sys, "argv", ["aide", "agent.use_predicted_metric_argmax_pick=true"])
    raw_cfg.agent = config._load_cfg().agent
    assert config.prep_cfg(raw_cfg).agent.use_predicted_metric_argmax_pick is True


def test_cli_overrides_merge_with_yaml_and_can_be_disabled(tmp_path, monkeypatch):
    settings = tmp_path / "settings.yaml"
    settings.write_text("agent:\n  steps: 5\n  search:\n    num_drafts: 2\n")
    monkeypatch.setattr(sys, "argv", ["aide", "agent.steps=7", "agent.search.debug_prob=0.3"])

    cfg = config._load_cfg(settings)

    assert cfg.agent.steps == 7
    assert cfg.agent.search.num_drafts == 2
    assert cfg.agent.search.debug_prob == 0.3
    assert config._load_cfg(settings, use_cli_args=False).agent.steps == 5


def test_normalized_mu_cli_flags_survive_schema_and_stage_resolution(raw_cfg, monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "aide",
            "agent.ucb.improve.normalize_mu=true",
            "agent.ucb.improve.ideal_metric=100",
            "agent.ucb.improve.scale_floor=0.001",
            "agent.ucb.improve.beta=0.3",
        ],
    )
    raw_cfg.agent.ucb = config._load_cfg().agent.ucb
    prepared = config.prep_cfg(raw_cfg)
    options = ucb.resolve_config(prepared.agent.ucb, "improve")
    assert options.normalize_mu is True
    assert options.ideal_metric == 100
    assert options.scale_floor == 0.001
    assert options.beta == 0.3
    assert ucb.resolve_config(prepared.agent.ucb, "draft") == ucb.Options()


@pytest.mark.parametrize("copy_data", [False, True])
def test_workspace_preserves_input_data(raw_cfg, copy_data):
    raw_cfg.copy_data = copy_data
    cfg = config.prep_cfg(raw_cfg)
    config.prep_agent_workspace(cfg)

    staged = cfg.workspace_dir / "input/train.csv"
    assert staged.read_text() == (cfg.data_dir / "train.csv").read_text()
    assert staged.is_symlink() is (not copy_data)
    assert (cfg.workspace_dir / "working").is_dir()
    assert (cfg.workspace_dir / "submission").is_dir()


def test_description_file_takes_precedence_over_goal_and_eval(raw_cfg, tmp_path, caplog):
    assert config.load_task_desc(raw_cfg) == {
        "Task goal": raw_cfg.goal,
        "Task evaluation": raw_cfg.eval,
    }
    desc = tmp_path / "description.md"
    desc.write_text("Train a regression model.")
    raw_cfg.desc_file = str(desc)
    assert config.load_task_desc(config.prep_cfg(raw_cfg)) == desc.read_text()
    assert "Ignoring goal and eval" in caplog.text


@pytest.mark.parametrize("plan", [None, "Train a model."])
def test_saving_keeps_published_code_even_if_tree_export_fails(raw_cfg, plan, caplog):
    cfg = config.prep_cfg(raw_cfg)
    journal = Journal()
    real = Node(
        code="print(0.6)",
        plan=plan,
        metric=MetricValue(0.6, maximize=True),
        is_buggy=False,
        real_won=True,
    )
    predicted = Node(
        code="print(0.9)",
        plan="Predicted improvement.",
        metric=MetricValue(0.9, maximize=True),
        is_buggy=False,
    )
    journal.append(real)
    journal.append(predicted)
    published = cfg.workspace_dir / "best_solution"
    published.mkdir(parents=True)
    (published / "solution.py").write_text(real.code)
    (published / "node_id.txt").write_text(real.id)

    config.save_run(cfg, journal)

    saved = json.loads((cfg.log_dir / "journal.json").read_text())
    assert len(saved["nodes"]) == 2
    assert (cfg.log_dir / "best_solution.py").read_text() == real.code
    assert journal.get_best_node() is predicted
    assert OmegaConf.load(cfg.log_dir / "config.yaml").agent == cfg.agent
    full_log = (cfg.log_dir / "full_log.txt").read_text()
    assert "File not found." in full_log
    assert real.id in full_log and predicted.id in full_log
    if plan is None:
        assert "Could not export the search tree" in caplog.text
    else:
        assert (cfg.log_dir / "tree_plot.html").is_file()


def test_empty_run_saves_without_visualization(raw_cfg, monkeypatch):
    generate = Mock()
    monkeypatch.setattr(config.tree_export, "generate", generate)
    cfg = config.prep_cfg(raw_cfg)
    config.save_run(cfg, Journal())

    generate.assert_not_called()
    assert json.loads((cfg.log_dir / "journal.json").read_text())["nodes"] == []
    assert (cfg.log_dir / "full_log.txt").is_file()
    assert not (cfg.log_dir / "best_solution.py").exists()


def test_journal_write_errors_are_not_suppressed(raw_cfg, monkeypatch):
    monkeypatch.setattr(config.serialize, "dump_json", Mock(side_effect=OSError("disk full")))
    with pytest.raises(OSError, match="disk full"):
        config.save_run(config.prep_cfg(raw_cfg), Journal())
