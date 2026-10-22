"""Checkout-based paper launches preserve the published policies and budgets."""

import json
import logging
from pathlib import Path
from types import SimpleNamespace

import pytest
import run_paper
import yaml
from agents.registry import Registry
from aide.agent import Agent
from aide.journal import Journal
from aide.utils.config import Config, _load_cfg
from omegaconf import OmegaConf

SUITE = run_paper.BENCH / "configs/paper.yaml"
ARMS = {
    "bon_metric_argmax": ("parallel", "parallel", "parallel", None),
    "bon_metric_argmax_s50_f56": ("parallel", "parallel", "parallel", 0.56),
    "debug_pclean": ("ucb", "ucb", "ucb", None),
    "debug_pclean_s50_f83": ("ucb", "ucb", "ucb", 0.83),
    "seq": ("sequential", "sequential", "sequential", None),
    "pairwise": ("pairwise", "pairwise", "pairwise", None),
    "fuse": ("fuse", "fuse", "fuse", None),
    "seq_s50_f56": ("sequential", "sequential", "sequential", 0.56),
    "pairwise_s50_f56": ("pairwise", "pairwise", "pairwise", 0.56),
    "fuse_s50_f83": ("fuse", "fuse", "fuse", 0.83),
    "base": ("single", "single", "single", None),
    "debug_pclean_s50_f56": ("ucb", "ucb", "ucb", 0.56),
    "bon_improve_ucb_beta05": ("parallel", "parallel", "ucb", None),
    "bon_improve_ucb_beta05_s50_f56": ("parallel", "parallel", "ucb", 0.56),
}


@pytest.mark.parametrize("arm", ARMS)
def test_profiles_construct_the_expected_agent(arm, tmp_path):
    suite, profiles = run_paper.compile_suite(
        SUITE,
        arms=[arm],
        coder_url="http://coder:8000/v1",
        feedback_url="http://feedback:8002/v1",
        world_url="http://world:8001/v1",
    )
    registered = Registry(arm, "aide").get_agent_from_config(profiles)
    cli = OmegaConf.from_dotlist([f"{k}={v}" for k, v in registered.kwargs.items()])
    raw = _load_cfg(use_cli_args=False)
    raw.data_dir = str(tmp_path)
    raw.exp_name = "paper-config-test"
    cfg = OmegaConf.merge(OmegaConf.structured(Config), raw, cli)
    cfg.log_dir = str(tmp_path)
    expected = ARMS[arm]
    agent = Agent(
        "test task",
        cfg,
        Journal(),
        sim_predict=lambda code: None,
        sim_p_clean=lambda code: {"p_clean": 0.9},
    )
    assert tuple(agent.stage_policy[s] for s in ("draft", "debug", "improve")) == expected[:3]
    assert cfg.agent.execution_gate.enabled is (expected[3] is not None)
    if expected[3] is not None:
        assert cfg.agent.execution_gate.success_threshold == 0.5
        assert cfg.agent.execution_gate.failure_threshold == expected[3]
    assert cfg.agent.time_limit == 43200
    assert cfg.exec.timeout == 3400
    assert registered.env_vars["TIME_LIMIT_SECS"] == "43200"
    assert registered.env_vars["AIDE_SHUTDOWN_RESERVE_S"] == "480"
    assert suite["seeds"] == [34, 35, 36] and len(suite["tasks"]) == 17
    assert registered.start == run_paper.BENCH / "agents/aide/start.sh"
    assert registered.env_vars["OPENAI_BASE_URL"] == "http://coder:8000/v1"
    assert json.loads(registered.env_vars["AIDE_MODEL_BASE_URLS"]) == {
        cfg.agent.feedback.model: "http://feedback:8002/v1"
    }
    assert cfg.exec.simulator is (arm != "base")
    assert cfg.agent.best_of_n_size == (1 if arm == "base" else 4)
    if arm != "base":
        assert cfg.exec.simulator_only
        assert cfg.agent.real_validation_interval == 1
        assert registered.env_vars["AIDE_SIMULATOR_BASE_URL"] == "http://world:8001/v1"


@pytest.mark.parametrize("regime", ["gated", "ungated"])
def test_regime_selection_has_seven_arms_and_357_trials(regime):
    args = run_paper.make_parser().parse_args(["--regime", regime, "--dry-run"])
    plan, profiles = run_paper.prepare_plan(args)
    assert len(profiles) == 7 and plan["total_trials"] == 357
    assert all(
        p["kwargs"]["agent.execution_gate.enabled"] is (regime == "gated")
        for p in profiles.values()
    )


def test_invalid_selection_fails_before_launch(tmp_path, monkeypatch):
    monkeypatch.setattr(run_paper.subprocess, "run", lambda *a, **k: pytest.fail("launched"))
    with pytest.raises(SystemExit):
        run_paper.main(["--arm", "unknown", "--output-dir", str(tmp_path / "bad")])
    with pytest.raises(ValueError, match="outside"):
        run_paper.compile_suite(SUITE, regime="gated", arms=["base"])
    assert not (tmp_path / "bad").exists()


def test_dry_run_never_writes_or_launches(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(run_paper.subprocess, "run", lambda *a, **k: pytest.fail("launched"))
    out = tmp_path / "dry"
    assert run_paper.main(["--dry-run", "--output-dir", str(out)]) == 0
    assert not out.exists()
    text = capsys.readouterr().out
    assert "714 trials" in text and "--seed-offset 34" in text
    assert "submission_rejection_wm4h30" not in text


def test_prepare_saves_runnable_config_without_credentials(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-secret-do-not-persist")
    monkeypatch.setenv("AIDE_SIMULATOR_API_KEY", "test-world-secret")
    monkeypatch.setattr(run_paper.subprocess, "run", lambda *a, **k: pytest.fail("launched"))
    out = tmp_path / "prepared"
    assert (
        run_paper.main(
            ["--prepare-only", "--arm", "debug_pclean_s50_f56", "--output-dir", str(out)]
        )
        == 0
    )
    config = yaml.safe_load((out / "agent-config.yaml").read_text())
    agent = Registry("debug_pclean_s50_f56", "aide").get_agent_from_config(config)
    assert agent.env_vars["OPENAI_API_KEY"] == "test-secret-do-not-persist"
    for path in out.iterdir():
        assert "test-secret-do-not-persist" not in path.read_text()
        assert "test-world-secret" not in path.read_text()
    plan = json.loads((out / "launch.json").read_text())
    assert plan["total_trials"] == 51
    assert len((out / "tasks.txt").read_text().splitlines()) == 17
    assert plan["source_root"] == str(run_paper.BENCH)


def test_dispatch_uses_main_checkout_and_stops_on_failure(tmp_path, monkeypatch):
    image = tmp_path / "aide.sqsh"
    image.touch()
    calls = []

    def execute(command, **kwargs):
        calls.append((command, kwargs))
        assert Path(command[1]) == run_paper.BENCH / "run_agent.py"
        assert command[command.index("--seed-offset") + 1] == "34"
        assert command[command.index("--n-seeds") + 1] == "3"
        assert command[command.index("--runtime") + 1] == "enroot"
        assert kwargs["cwd"] == run_paper.BENCH
        assert kwargs["env"]["MLEBENCH_RUNS_DIR"] == str(tmp_path / "run/runs")
        return SimpleNamespace(returncode=7)

    monkeypatch.setattr(run_paper.subprocess, "run", execute)
    assert (
        run_paper.main(
            [
                "--arm",
                "base",
                "debug_pclean",
                "--enroot-image",
                str(image),
                "--output-dir",
                str(tmp_path / "run"),
            ]
        )
        == 7
    )
    assert len(calls) == 1


def test_shards_must_be_subsets_of_the_suite(tmp_path):
    shard = tmp_path / "shard.txt"
    shard.write_text("aerial-cactus-identification\n")
    args = run_paper.make_parser().parse_args(["--competition-set", str(shard), "--arm", "base"])
    plan, _ = run_paper.prepare_plan(args)
    assert plan["total_trials"] == 3
    shard.write_text("not-a-paper-task\n")
    with pytest.raises(ValueError, match="subset"):
        run_paper.prepare_plan(args)


def test_inheritance_cycle_is_reported(tmp_path):
    path = tmp_path / "loop.yaml"
    path.write_text("schema_version: 1\nextends: loop.yaml\n")
    with pytest.raises(ValueError, match="Circular"):
        run_paper.resolve_profile(path)


def test_mismatched_timeouts_are_rejected(tmp_path):
    suite = yaml.safe_load(SUITE.read_text())
    suite["source_base"] = str(run_paper.BENCH)
    suite["arms"] = [
        {**suite["arms"][0], "configuration": str(SUITE.parent / suite["arms"][0]["configuration"])}
    ]
    suite["trial_seconds"] = 3600
    path = tmp_path / "bad-budget.yaml"
    path.write_text(yaml.safe_dump(suite))
    with pytest.raises(ValueError, match="agent.time_limit"):
        run_paper.compile_suite(path)


@pytest.fixture
def completed_group(tmp_path):
    task = "aerial-cactus-identification"
    group = tmp_path / "group"
    group.mkdir()
    runs = {
        f"{task}_trial{i}": {"competition": task, "seed": seed, "success": True}
        for i, seed in enumerate([34, 35])
    }
    metadata = group / "metadata.json"
    metadata.write_text(json.dumps({"status": "complete", "expected_runs": 2, "runs": runs}))
    return metadata, task


def test_grading_uses_frozen_source_and_counts_missing_submissions(completed_group, tmp_path):
    from agents.snapshots import snapshot_agent_code
    from mlebench.registry import Registry as CompetitionRegistry

    metadata, task = completed_group
    data = tmp_path / "data"
    competition = CompetitionRegistry(data).get_competition(task)
    competition.private_dir.mkdir(parents=True)
    competition.answers.write_text("id,has_cactus\na.jpg,0\nb.jpg,1\n")
    submission = metadata.parent / f"{task}_trial0/submission/submission.csv"
    submission.parent.mkdir(parents=True)
    submission.write_text("id,has_cactus\nb.jpg,0.9\na.jpg,0.1\n")
    snapshot_agent_code(
        run_paper.BENCH / "agents/aide", metadata.parent, logging.getLogger(__name__)
    )

    assert run_paper.grade_run_group(metadata, tasks=[task], seeds=[34, 35], data_dir=data) == 0
    reports = list(metadata.parent.glob("*_grading_report.json"))
    assert len(reports) == 1
    report = json.loads(reports[0].read_text())
    assert report["total_runs"] == 2
    assert report["total_runs_with_submissions"] == report["total_valid_submissions"] == 1
    assert [r["score"] for r in report["competition_reports"]] == [1.0, None]
    assert len((metadata.parent / "submission.jsonl").read_text().splitlines()) == 2


@pytest.mark.parametrize("problem", ["incomplete", "duplicate_seed", "failed"])
def test_grading_rejects_incomplete_grid(completed_group, tmp_path, monkeypatch, problem):
    metadata, task = completed_group
    payload = json.loads(metadata.read_text())
    if problem == "incomplete":
        payload["status"] = "running"
    elif problem == "duplicate_seed":
        payload["runs"][f"{task}_trial1"]["seed"] = 34
    else:
        payload["runs"][f"{task}_trial1"]["success"] = False
    metadata.write_text(json.dumps(payload))
    monkeypatch.setattr(
        run_paper.subprocess, "run", lambda *a, **k: pytest.fail("graded incomplete grid")
    )
    with pytest.raises(ValueError, match="Incomplete or mismatched"):
        run_paper.grade_run_group(metadata, tasks=[task], seeds=[34, 35], data_dir=tmp_path)


def test_grading_rejects_changed_frozen_source(completed_group, tmp_path, monkeypatch):
    metadata, task = completed_group
    snapshot = metadata.parent / "agent_snapshot"
    (snapshot / "mlebench").mkdir(parents=True)
    (snapshot / "mlebench/grade.py").write_text("raise RuntimeError('modified grader')\n")
    (snapshot / "PROVENANCE.json").write_text(json.dumps({"mlebench_sha256": "wrong"}))
    monkeypatch.setattr(
        run_paper.subprocess, "run", lambda *a, **k: pytest.fail("used changed grader")
    )
    with pytest.raises(ValueError, match="hash mismatch"):
        run_paper.grade_run_group(metadata, tasks=[task], seeds=[34, 35], data_dir=tmp_path)


def test_single_command_runs_grades_and_reports_costs(tmp_path, monkeypatch):
    from agents.snapshots import snapshot_agent_code
    from mlebench.registry import Registry as CompetitionRegistry

    task = "aerial-cactus-identification"
    data = tmp_path / "data"
    competition = CompetitionRegistry(data).get_competition(task)
    competition.private_dir.mkdir(parents=True)
    competition.answers.write_text("id,has_cactus\na.jpg,0\nb.jpg,1\n")
    image = tmp_path / "image.sqsh"
    image.touch()
    shard = tmp_path / "tasks.txt"
    shard.write_text(task + "\n")
    output = tmp_path / "suite"
    execute = run_paper.subprocess.run

    def synthetic_agent(command, **kwargs):
        if command[1] != str(run_paper.BENCH / "run_agent.py"):
            return execute(command, **kwargs)
        group = Path(kwargs["env"]["MLEBENCH_RUNS_DIR"]) / "synthetic"
        group.mkdir(parents=True)
        runs = {}
        for seed in [34, 35, 36]:
            run_id = f"{task}_{seed}"
            runs[run_id] = {"competition": task, "seed": seed, "success": True}
            submission = group / run_id / "submission/submission.csv"
            submission.parent.mkdir(parents=True)
            submission.write_text("id,has_cactus\na.jpg,0.1\nb.jpg,0.9\n")
            logs = group / run_id / "logs"
            logs.mkdir()
            # A legacy journal has unknown cost; the reporter must preserve that.
            (logs / "journal.json").write_text('{"nodes": []}')
        (group / "metadata.json").write_text(
            json.dumps(
                {
                    "status": "complete",
                    "expected_runs": 3,
                    "runs": runs,
                }
            )
        )
        snapshot_agent_code(run_paper.BENCH / "agents/aide", group, logging.getLogger(__name__))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(run_paper.subprocess, "run", synthetic_agent)
    assert (
        run_paper.main(
            [
                "--grade",
                "--arm",
                "base",
                "--competition-set",
                str(shard),
                "--data-dir",
                str(data),
                "--enroot-image",
                str(image),
                "--output-dir",
                str(output),
            ]
        )
        == 0
    )
    (report_path,) = (output / "runs/synthetic").glob("*_grading_report.json")
    assert json.loads(report_path.read_text())["total_valid_submissions"] == 3
    assert json.loads((output / "launch.json").read_text())["grade_after_arm"] is True
    assert (output / "costs/trials.csv").is_file()


@pytest.mark.parametrize("regime,count", [("all", 2), ("gated", 1), ("ungated", 1)])
def test_smoke_plan_is_bounded_and_preserves_gates(regime, count):
    args = run_paper.make_parser().parse_args(
        [
            "--smoke-test",
            "--regime",
            regime,
            "--feedback-model",
            "gpt-qwen3-vllm",
            "--coder-url",
            "http://shared:8000/v1",
            "--feedback-url",
            "http://shared:8000/v1",
        ]
    )
    plan, profiles = run_paper.prepare_plan(args)
    assert plan["mode"] == "smoke"
    assert plan["tasks"] == ["leaf-classification"]
    assert plan["seeds"] == [34]
    assert plan["trial_seconds"] == 1200
    assert plan["total_trials"] == count
    assert len(profiles) == count
    for arm, profile in profiles.items():
        kwargs, env = profile["kwargs"], profile["env_vars"]
        assert kwargs["agent.execution_gate.enabled"] == arm.endswith("_s50_f56")
        assert kwargs["agent.steps"] == 4
        assert kwargs["agent.search.num_drafts"] == 1
        assert kwargs["agent.best_of_n_size"] == 2
        assert kwargs["exec.timeout"] == 300
        assert kwargs["agent.time_limit"] == int(env["TIME_LIMIT_SECS"]) == 1200
        assert kwargs["agent.code.model"] == kwargs["agent.feedback.model"] == "gpt-qwen3-vllm"
        assert json.loads(env["AIDE_MODEL_BASE_URLS"]) == {
            "gpt-qwen3-vllm": "http://shared:8000/v1"
        }
        assert json.loads(env["AIDE_MODEL_SAMPLING"]) == {}
        sampling = json.loads(env["AIDE_SAMPLING"])
        assert sampling["max_tokens"] == 8192
        assert "chat_template_kwargs" not in sampling
    _, paper_profiles = run_paper.compile_suite(SUITE)
    assert len(paper_profiles) == 14
    assert all(p["kwargs"]["agent.time_limit"] == 43200 for p in paper_profiles.values())
    assert all(
        p["kwargs"]["agent.feedback.model"] == "gpt-coderbf16-fb-vllm"
        for p in paper_profiles.values()
    )


def test_shared_feedback_rejects_ambiguous_model_routing():
    with pytest.raises(ValueError, match="matching server URLs"):
        run_paper.compile_suite(SUITE, feedback_model="gpt-qwen3-vllm")


def test_smoke_rejects_multiple_tasks(tmp_path):
    tasks = tmp_path / "tasks.txt"
    tasks.write_text("leaf-classification\nspooky-author-identification\n")
    args = run_paper.make_parser().parse_args(
        [
            "--smoke-test",
            "--competition-set",
            str(tasks),
        ]
    )
    with pytest.raises(ValueError, match="exactly one competition"):
        run_paper.prepare_plan(args)


@pytest.mark.parametrize("valid,expected_status", [(True, 0), (False, 1)])
def test_smoke_exit_status_requires_valid_grading(tmp_path, monkeypatch, valid, expected_status):
    image = tmp_path / "image.sqsh"
    image.touch()
    launched = []

    def execute(command, **kwargs):
        if command[1] == str(run_paper.BENCH / "run_agent.py"):
            arm = command[command.index("--agent-id") + 1]
            launched.append(arm)
            group = Path(kwargs["env"]["MLEBENCH_RUNS_DIR"]) / arm
            group.mkdir(parents=True)
            (group / "metadata.json").write_text("{}")
        return SimpleNamespace(returncode=0)

    def grade(metadata_path, **kwargs):
        report = {
            "total_runs": 1,
            "total_valid_submissions": int(valid),
            "competition_reports": [{"valid_submission": valid, "score": 0.5 if valid else None}],
        }
        (metadata_path.parent / "test_grading_report.json").write_text(json.dumps(report))
        return 0

    monkeypatch.setattr(run_paper.subprocess, "run", execute)
    monkeypatch.setattr(run_paper, "grade_run_group", grade)
    assert (
        run_paper.main(
            [
                "--smoke-test",
                "--enroot-image",
                str(image),
                "--output-dir",
                str(tmp_path / "smoke"),
            ]
        )
        == expected_status
    )
    assert launched == ["bon_metric_argmax", "bon_metric_argmax_s50_f56"]
    plan = json.loads((tmp_path / "smoke/launch.json").read_text())
    assert plan["grade_after_arm"] is True


@pytest.mark.parametrize("score", [None, True, float("nan"), float("inf")])
def test_smoke_rejects_missing_or_nonfinite_grading_scores(tmp_path, score):
    report = {
        "total_runs": 1,
        "total_valid_submissions": 1,
        "competition_reports": [{"valid_submission": True, "score": score}],
    }
    (tmp_path / "test_grading_report.json").write_text(json.dumps(report))
    with pytest.raises(ValueError, match="valid, finite"):
        run_paper.verify_smoke_grade(tmp_path, 1)
