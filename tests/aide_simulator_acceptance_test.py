"""Exercise simulator results through Agent.step and real validation."""

import json
import sys
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "agents/aide/src"))
Agent = pytest.importorskip("aide.agent").Agent
from aide import simulator as simulator_mod
from aide.execution_gate import ExecutionGateConfig
from aide.interpreter import ExecutionResult
from aide.journal import Journal, Node
from aide.parallel_interpreter import ParallelInterpreter
from aide.run import validate_on_best
from aide.utils.metric import MetricValue
from aide.utils.serialize import dumps_json, loads_json
from aide.utils.submission_validation import SubmissionValidation, submission_sha256

PRIOR_CSV = "id,pred\n0,0.25\n"
VALIDATED_CSV = "id,pred\n0,0.8\n"


class _Real:
    def __init__(self, workspace, sim_only):
        self.working_dir = workspace
        self.sim_only = sim_only
        self.release = threading.Event()
        self.calls = 0
        self.timeout = 5

    def run(self, code, reset_session=True):
        self.calls += 1
        if not self.sim_only and self.calls == 1:
            self.release.wait(timeout=5)
            return ExecutionResult([], 0.1, "TimeoutError", real_won=True)
        (self.working_dir / "submission/submission.csv").write_text(VALIDATED_CSV)
        return ExecutionResult(["Validation score: 0.8"], 0.1, None, real_won=True)

    def cleanup_session(self):
        self.release.set()


def _agent(workspace):
    agent = object.__new__(Agent)
    agent.acfg = SimpleNamespace(record_test_metric=True, eval_lower_is_better=False)
    agent.cfg = SimpleNamespace(workspace_dir=workspace, agent=agent.acfg)
    agent.journal = Journal()
    prior = Node(code="prior solution")
    prior.metric = MetricValue(0.5, maximize=True)
    prior.is_buggy = False
    prior.real_won = True
    prior.submission_validation = "valid"
    agent.journal.append(prior)
    (workspace / "best_submission").mkdir()
    (workspace / "best_submission/submission.csv").write_text(PRIOR_CSV)
    prior.submission_sha256 = submission_sha256(workspace / "best_submission/submission.csv")
    agent.oracle = "sim"
    agent.current_step = 1
    agent.data_preview = "test data"
    agent.search_policy = lambda: None
    agent._draft_prompt = lambda: {}
    agent.generate_code = lambda *args, **kwargs: ("plan", "candidate solution", None)
    agent._review = lambda code, output, **kwargs: {
        "is_bug": False,
        "has_csv_submission": True,
        "metric": 0.9 if "0.9" in output else 0.8,
        "metric_trustworthy": True,
        "lower_is_better": False,
        "summary": "Training succeeded and a submission was saved.",
    }
    agent._grade_test_submission = Mock(return_value=0.75)
    agent._validate_submission = Mock(
        side_effect=lambda path, **kwargs: SubmissionValidation(
            "valid", "Submission is valid.", submission_sha256(path)
        )
    )
    (workspace / "best_solution").mkdir()
    (workspace / "best_solution/solution.py").write_text(prior.code)
    (workspace / "best_solution/node_id.txt").write_text(prior.id)
    return agent


@pytest.mark.parametrize("sim_only", [False, True])
@pytest.mark.parametrize("token_format", [False, True])
def test_sim_success_reaches_validation_without_a_placeholder(
    tmp_path, sim_only, token_format, monkeypatch
):
    agent = _agent(tmp_path)
    real = _Real(tmp_path, sim_only)
    monkeypatch.setenv(
        "AIDE_SIMULATOR_PROMPT_FORMAT",
        "env_v1_reorder_excfirst" + ("_token" if token_format else ""),
    )
    monkeypatch.setattr(
        simulator_mod,
        "_streaming_query",
        Mock(
            return_value=(
                f"<exc_type>{'EXC_None' if token_format else 'None'}</exc_type>\n"
                "<term_out>\nValidation score: 0.9\n</term_out>\n"
                "<val_metric>0.9</val_metric>\n<submission_exists>true"
            )
        ),
    )
    simulator = simulator_mod.SimulatorInterpreter(
        tmp_path, "qwen3_5-4b-sim", environment="python 3.11 · no network"
    )
    race = ParallelInterpreter(
        real, simulator, deadline_s=0.1, start_delay_s=0.01, sim_only=sim_only
    )
    try:
        agent.step(race.run)
        node = agent.journal.nodes[-1]
        assert node.is_buggy is False
        assert node.metric.value == 0.9
        assert node.csv_exists is True
        assert node.real_won is False
        assert agent.journal.get_best_node() is node
        assert not (tmp_path / "submission/submission.csv").exists()
        assert (tmp_path / "best_submission/submission.csv").read_text() == PRIOR_CSV
        assert (tmp_path / "best_solution/solution.py").read_text() == "prior solution"
        agent._grade_test_submission.assert_not_called()
        agent._validate_submission.assert_not_called()

        assert validate_on_best(agent, agent.journal, real, tmp_path) is True
        assert real.calls == (1 if sim_only else 2)
        assert node.real_won is True
        assert node.is_buggy is False
        assert node.metric.value == 0.8
        assert (tmp_path / "best_submission/submission.csv").read_text() == VALIDATED_CSV
        assert (tmp_path / "best_solution/solution.py").read_text() == node.code
        assert (tmp_path / "best_solution/node_id.txt").read_text() == node.id
        agent._grade_test_submission.assert_called_once_with()
        agent._validate_submission.assert_called_once_with(
            tmp_path / "submission/submission.csv", timeout_s=75.0
        )
    finally:
        real.cleanup_session()
        race.close()


@pytest.mark.parametrize(
    "prediction,exc_type", [("false", None), ("unknown", None), ("true", "ValueError")]
)
def test_sim_failure_cannot_reach_validation_or_replace_prior_csv(tmp_path, prediction, exc_type):
    agent = _agent(tmp_path)
    real = _Real(tmp_path, sim_only=True)
    result = ExecutionResult(
        ["Validation score: 0.9"],
        100.0,
        exc_type,
        predicted_submission_exists=prediction,
    )
    race = ParallelInterpreter(real, SimpleNamespace(run=lambda *args: result), sim_only=True)
    try:
        agent.step(race.run)
        node = agent.journal.nodes[-1]
        assert node.is_buggy is True
        assert node.csv_exists is (prediction == "true")
        assert validate_on_best(agent, agent.journal, real, tmp_path) is False
        assert real.calls == 0
        assert (tmp_path / "best_submission/submission.csv").read_text() == PRIOR_CSV
        agent._grade_test_submission.assert_not_called()
    finally:
        race.close()


def test_real_empty_csv_is_rejected_even_when_review_says_success(tmp_path):
    agent = _agent(tmp_path)

    def execute(code, reset_session=True):
        (tmp_path / "submission/submission.csv").write_bytes(b"")
        return ExecutionResult(
            ["Validation score: 0.9"],
            0.1,
            None,
            real_won=True,
            predicted_submission_exists="true",
        )

    agent.step(execute)
    node = agent.journal.nodes[-1]
    assert node.is_buggy is True
    assert node.csv_exists is False
    assert (tmp_path / "best_submission/submission.csv").read_text() == PRIOR_CSV
    agent._grade_test_submission.assert_not_called()


@pytest.mark.parametrize("sim_first", [False, True])
def test_invalid_real_csv_preserves_confirmed_artifacts(tmp_path, sim_first):
    agent = _agent(tmp_path)
    agent._validate_submission.side_effect = lambda path, **kwargs: SubmissionValidation(
        "invalid",
        "Submission invalid! Submission should have a Insult column",
        submission_sha256(path),
    )
    real = _Real(tmp_path, sim_only=True)
    if sim_first:
        agent.step(
            lambda *args: ExecutionResult(
                ["Validation score: 0.9"], 0, None, predicted_submission_exists="true"
            )
        )
        validate_on_best(agent, agent.journal, real, tmp_path)
    else:
        agent.step(real.run)
    node = agent.journal.nodes[-1]
    assert node.is_buggy is True
    assert node.metric.value is None
    assert "Insult column" in node.analysis
    assert agent.journal.get_best_node().code == "prior solution"
    assert (tmp_path / "best_submission/submission.csv").read_text() == PRIOR_CSV
    assert (tmp_path / "best_solution/solution.py").read_text() == "prior solution"


@pytest.mark.parametrize("outcome", ["invalid", "worse", "exception", "better"])
def test_real_candidate_direct_writes_require_review_before_publication(tmp_path, outcome):
    agent = _agent(tmp_path)
    prior = agent.journal.nodes[0]
    agent.step(
        lambda *args: ExecutionResult(
            ["Validation score: 0.9"], 0, None, predicted_submission_exists="true"
        )
    )
    candidate = agent.journal.nodes[-1]
    if outcome == "invalid":
        agent._validate_submission.side_effect = lambda path, **kwargs: SubmissionValidation(
            "invalid", "Wrong columns", submission_sha256(path)
        )
    if outcome == "worse":
        review = agent._review
        agent._review = lambda *args, **kwargs: {**review(*args, **kwargs), "metric": 0.1}

    def execute(code):
        (tmp_path / "best_submission/submission.csv").write_text("unreviewed")
        (tmp_path / "best_solution/solution.py").write_text("unreviewed code")
        (tmp_path / "best_solution/node_id.txt").write_text("unreviewed node")
        if outcome == "exception":
            raise RuntimeError("interpreter disconnected after writing")
        (tmp_path / "submission/submission.csv").write_text(VALIDATED_CSV)
        return ExecutionResult(["Validation score: 0.8"], 0.1, None, real_won=True)

    real = SimpleNamespace(run=execute, timeout=5, cleanup_session=Mock())
    validate_on_best(agent, agent.journal, real, tmp_path, force=True)
    published = candidate if outcome == "better" else prior
    assert (tmp_path / "best_submission/submission.csv").read_text() == (
        VALIDATED_CSV if outcome == "better" else PRIOR_CSV
    )
    assert (tmp_path / "best_solution/solution.py").read_text() == published.code
    assert (tmp_path / "best_solution/node_id.txt").read_text() == published.id


def test_publication_guard_restores_mount_symlink_and_prior_absence(tmp_path):
    from aide.utils.artifacts import preserve_published_artifacts

    output = tmp_path / "harness_output"
    output.mkdir()
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "best_submission").symlink_to(output, target_is_directory=True)
    with pytest.raises(KeyboardInterrupt):
        with preserve_published_artifacts(workspace):
            (output / "submission.csv").write_text("unreviewed")
            (workspace / "best_submission").unlink()
            (workspace / "best_submission").mkdir()
            (workspace / "best_submission/submission.csv").write_text("also unreviewed")
            raise KeyboardInterrupt()
    assert (workspace / "best_submission").is_symlink()
    assert (workspace / "best_submission").resolve() == output
    assert not (output / "submission.csv").exists()


def test_real_candidate_can_publish_below_unconfirmed_sim_score(tmp_path):
    agent = _agent(tmp_path)
    simulated = Node(code="unconfirmed")
    simulated.is_buggy = False
    simulated.metric = MetricValue(0.99, maximize=True)
    agent.journal.append(simulated)
    agent.step(_Real(tmp_path, sim_only=True).run)
    assert agent.journal.get_best_node() is simulated
    assert agent.journal.get_best_node(real_only=True) is agent.journal.nodes[-1]
    assert (tmp_path / "best_submission/submission.csv").read_text() == VALIDATED_CSV
    assert (tmp_path / "best_solution/solution.py").read_text() == "candidate solution"


def test_unverified_real_score_cannot_block_validated_candidate(tmp_path):
    agent = _agent(tmp_path)
    unverified = Node(code="unverified real solution")
    unverified.real_won, unverified.is_buggy = True, False
    unverified.metric = MetricValue(0.99, maximize=True)
    unverified.submission_validation = "unknown"
    agent.journal.append(unverified)
    agent.step(_Real(tmp_path, sim_only=True).run)
    assert agent.journal.get_best_node() is unverified
    assert agent.journal.get_best_node(validated_only=True) is agent.journal.nodes[-1]
    assert (tmp_path / "best_submission/submission.csv").read_text() == VALIDATED_CSV


def test_unverified_real_score_does_not_block_simulator_real_execution(tmp_path):
    agent = _agent(tmp_path)
    unverified = Node(code="unverified real solution")
    unverified.real_won, unverified.is_buggy = True, False
    unverified.metric = MetricValue(0.99, maximize=True)
    unverified.submission_validation = "unknown"
    agent.journal.append(unverified)
    agent.step(
        lambda *args: ExecutionResult(
            ["Validation score: 0.9"], 0, None, predicted_submission_exists="true"
        )
    )
    assert agent.journal.get_best_node() is unverified
    real = _Real(tmp_path, sim_only=True)
    assert validate_on_best(agent, agent.journal, real, tmp_path)
    assert real.calls == 1
    assert (tmp_path / "best_submission/submission.csv").read_text() == VALIDATED_CSV


@pytest.mark.parametrize("remaining", [0, 0.5, 12.8])
def test_real_validation_respects_remaining_budget(tmp_path, monkeypatch, remaining):
    agent = _agent(tmp_path)
    agent.step(
        lambda *args: ExecutionResult(
            ["Validation score: 0.9"], 0, None, predicted_submission_exists="true"
        )
    )
    real = _Real(tmp_path, sim_only=True)
    real.timeout = 3400
    original_run = real.run
    observed = []

    def run(*args, **kwargs):
        observed.append(real.timeout)
        return original_run(*args, **kwargs)

    real.run = run
    monkeypatch.setattr("aide.run.time.time", lambda: 1000)
    ran = validate_on_best(
        agent, agent.journal, real, tmp_path, deadline=1180 + remaining, reserve_s=180
    )
    assert ran is (remaining >= 1)
    assert observed == ([12] if remaining >= 1 else [])
    assert real.timeout == 3400


def test_exploration_validates_below_incumbent(tmp_path):
    agent = _agent(tmp_path)
    agent.journal.nodes[0].metric = MetricValue(0.95, maximize=True)
    agent.step(
        lambda *args: ExecutionResult(
            ["Validation score: 0.9"], 0, None, predicted_submission_exists="true"
        )
    )
    real = _Real(tmp_path, sim_only=True)
    assert not validate_on_best(agent, agent.journal, real, tmp_path)
    assert validate_on_best(agent, agent.journal, real, tmp_path, force=True)
    assert agent.journal.nodes[-1].real_won
    assert (tmp_path / "best_submission/submission.csv").read_text() == PRIOR_CSV


def test_exploration_probes_predicted_bug(tmp_path):
    agent = _agent(tmp_path)
    agent.step(
        lambda *args: ExecutionResult(
            ["Predicted error"], 0, "ValueError", predicted_submission_exists="false"
        )
    )
    assert agent.journal.nodes[-1].is_buggy
    real = _Real(tmp_path, sim_only=True)
    assert validate_on_best(agent, agent.journal, real, tmp_path, force=True)
    assert not agent.journal.nodes[-1].is_buggy
    assert (tmp_path / "best_submission/submission.csv").read_text() == VALIDATED_CSV


@pytest.mark.parametrize(
    "stage,threshold,p_max,already_real,force,ran",
    [
        ("improve", None, 0.6, False, False, False),
        ("improve", 0.95, 0.94, False, False, True),
        ("improve", 0.95, 0.95, False, False, False),
        ("improve", 0.95, 0.99, False, False, False),
        ("improve", 0.95, None, False, False, True),
        ("improve", 0.95, float("nan"), False, False, True),
        ("improve", 0.95, 1.1, False, False, True),
        ("improve", 0.95, 0.6, True, False, False),
        ("improve", 0.95, 0.99, False, True, True),
        ("draft", 0.95, 0.6, False, False, False),
        ("debug", 0.95, 0.6, False, False, False),
    ],
)
def test_confidence_gate_checks_only_selected_simulated_improve_nodes(
    tmp_path,
    stage,
    threshold,
    p_max,
    already_real,
    force,
    ran,
):
    agent = _agent(tmp_path)
    prior = agent.journal.nodes[0]
    prior.metric = MetricValue(0.95, maximize=True)
    parent = None if stage == "draft" else prior
    if stage == "debug":
        parent = Node(code="broken parent", is_buggy=True)
        agent.journal.append(parent)
    agent.search_policy = lambda: parent
    agent._improve_prompt = lambda _: {}
    agent._build_debug_prompt = lambda *args: {}
    agent.acfg.real_validation_p_max_threshold = threshold
    agent._exc_scores = Mock(
        return_value={
            "max_p": p_max,
            "top_exc": "TimeoutError",
            "p_top_exc": 0.1,
        }
    )
    real = _Real(tmp_path, sim_only=True)
    agent.step(
        real.run
        if already_real
        else lambda *args: ExecutionResult(
            ["Validation score: 0.9"], 0, None, predicted_submission_exists="true"
        )
    )
    node = agent.journal.nodes[-1]
    assert node.stage_name == stage
    assert agent._exc_scores.call_count == int(
        stage == "improve" and threshold is not None and not already_real
    )
    assert validate_on_best(agent, agent.journal, real, tmp_path, force=force) is ran
    assert real.calls == int(already_real) + int(ran)
    assert (tmp_path / "best_submission/submission.csv").read_text() == PRIOR_CSV
    restored = loads_json(dumps_json(agent.journal), Journal).nodes[-1]
    assert restored.sim_p_max == node.sim_p_max
    assert restored.sim_top_exc == node.sim_top_exc
    assert restored.sim_p_top_exc == node.sim_p_top_exc
    assert restored.stage_name == stage
    if ran:
        assert node.real_won
        assert not validate_on_best(agent, agent.journal, real, tmp_path)
        assert real.calls == 1


@pytest.mark.parametrize("stage", ["draft", "debug", "improve"])
@pytest.mark.parametrize(
    "stages,gated_stages",
    [
        ("improve", {"improve"}),
        ("all", {"draft", "debug", "improve"}),
        ("draft,debug", {"draft", "debug"}),
        (" Debug ", {"debug"}),
        (["draft", "improve"], {"draft", "improve"}),
    ],
)
def test_confidence_gate_stage_flag_selects_gated_stages(
    tmp_path,
    caplog,
    stage,
    stages,
    gated_stages,
):
    agent = _agent(tmp_path)
    prior = agent.journal.nodes[0]
    prior.metric = MetricValue(0.95, maximize=True)
    parent = None if stage == "draft" else prior
    if stage == "debug":
        parent = Node(code="broken parent", is_buggy=True)
        agent.journal.append(parent)
    agent.search_policy = lambda: parent
    agent._improve_prompt = lambda _: {}
    agent._build_debug_prompt = lambda *args: {}
    agent.acfg.real_validation_p_max_threshold = 0.95
    agent.acfg.real_validation_p_max_stages = stages
    agent._exc_scores = Mock(
        return_value={
            "max_p": 0.6,
            "top_exc": "ValueError",
            "p_top_exc": 0.3,
        }
    )
    agent.step(
        lambda *args: ExecutionResult(
            ["Validation score: 0.9"], 0, None, predicted_submission_exists="true"
        )
    )
    node = agent.journal.nodes[-1]
    gated = stage in gated_stages
    assert node.stage_name == stage
    assert agent._exc_scores.call_count == int(gated)
    assert node.sim_p_max == (0.6 if gated else None)
    real = _Real(tmp_path, sim_only=True)
    with caplog.at_level("INFO", logger="aide"):
        assert validate_on_best(agent, agent.journal, real, tmp_path) is gated
    assert real.calls == int(gated)
    assert node.real_won is gated
    gate_line = f"confidence gate (p_max=0.6, threshold=0.95, stage={stage}); running real"
    assert (gate_line in caplog.text) is gated
    assert (tmp_path / "best_submission/submission.csv").read_text() == PRIOR_CSV
    restored = loads_json(dumps_json(agent.journal), Journal).nodes[-1]
    assert restored.sim_p_max == node.sim_p_max and restored.stage_name == stage


@pytest.mark.parametrize("stage", ["draft", "debug", "improve"])
@pytest.mark.parametrize(
    "exc_type,p_max,success_threshold,failure_threshold,expected",
    [
        (None, 0.8, 0.9, 0.7, True),
        ("ValueError", 0.8, 0.9, 0.7, False),
        (None, 0.9, 0.9, 0.7, False),
        ("ValueError", 0.7, 0.9, 0.7, False),
        ("ValueError", 0.69, 0.9, 0.7, True),
        (None, None, 0.9, 0.7, True),
        ("ValueError", float("nan"), 0.9, 0.7, True),
        (None, float("inf"), 0.9, 0.7, True),
        ("ValueError", 1.1, 0.9, 0.7, True),
        (None, 1.0, None, 0.7, True),
        ("ValueError", 1.0, 0.9, None, True),
        (None, 0.0, 0.0, 0.7, False),
        (None, 1.0, 1.0, 0.7, False),
        ("TimeoutError", 0.1, 0.9, 0.7, False),
        ("TimeoutError", None, None, None, False),
    ],
)
def test_outcome_gate_checks_selected_program_independently_of_stage(
    tmp_path,
    stage,
    exc_type,
    p_max,
    success_threshold,
    failure_threshold,
    expected,
):
    agent = _agent(tmp_path)
    prior = agent.journal.nodes[0]
    prior.metric = MetricValue(0.95, maximize=True)
    parent = None if stage == "draft" else prior
    if stage == "debug":
        parent = Node(code="broken parent", is_buggy=True)
        agent.journal.append(parent)
    agent.search_policy = lambda: parent
    agent._improve_prompt = lambda _: {}
    agent._build_debug_prompt = lambda *args: {}
    agent.acfg.execution_gate = ExecutionGateConfig(True, success_threshold, failure_threshold)
    # The universal gate must never consult the old stage/threshold settings.
    agent.acfg.real_validation_p_max_threshold = 1.0
    agent.acfg.real_validation_p_max_stages = "invalid-but-ignored"
    agent._exc_scores = Mock(
        return_value={
            "max_p": p_max,
            "top_exc": "TimeoutError",
            "p_top_exc": 0.99,
        }
    )
    agent.step(
        lambda *args: ExecutionResult(
            ["Validation score: 0.9"],
            0,
            exc_type,
            predicted_submission_exists="true" if exc_type is None else "false",
        )
    )
    node = agent.journal.nodes[-1]
    assert node.stage_name == stage and node.exc_type == exc_type
    threshold = success_threshold if exc_type is None else failure_threshold
    assert agent._exc_scores.call_count == int(threshold is not None and exc_type != "TimeoutError")
    real = _Real(tmp_path, sim_only=True)
    assert validate_on_best(agent, agent.journal, real, tmp_path) is expected
    assert real.calls == int(expected)
    assert node.real_won is expected
    assert (tmp_path / "best_submission/submission.csv").read_text() == PRIOR_CSV
    if expected:
        assert not validate_on_best(agent, agent.journal, real, tmp_path, force=True)
        assert real.calls == 1


@pytest.mark.parametrize("stage", ["draft", "debug", "improve"])
@pytest.mark.parametrize("exc_type", [None, "ValueError", "TimeoutError"])
@pytest.mark.parametrize("force", [False, True])
def test_outcome_gate_keeps_best_and_forced_checks_but_never_executes_timeouts(
    tmp_path,
    stage,
    exc_type,
    force,
):
    agent = _agent(tmp_path)
    agent.acfg.execution_gate = ExecutionGateConfig(True, 0.9, 0.7)
    prior = agent.journal.nodes[0]
    parent = None if stage == "draft" else prior
    if stage == "debug":
        parent = Node(code="broken parent", is_buggy=True)
        agent.journal.append(parent)
    node = Node(
        code="selected",
        parent=parent,
        exc_type=exc_type,
        is_buggy=exc_type is not None,
        sim_p_max=0.99,
    )
    node.metric = MetricValue(0.9, maximize=True)
    agent.journal.append(node)
    real = _Real(tmp_path, sim_only=True)
    expected = exc_type != "TimeoutError" and (exc_type is None or force)
    assert validate_on_best(agent, agent.journal, real, tmp_path, force=force) is expected
    assert real.calls == int(expected)
    if expected:
        assert (tmp_path / "best_submission/submission.csv").read_text() == VALIDATED_CSV


@pytest.mark.parametrize("stage", ["draft", "debug", "improve"])
@pytest.mark.parametrize(
    "enabled,exc_type,p_max,expected",
    [
        (True, None, 0.99, False),
        (True, None, 0.8, True),
        (True, "ValueError", 0.99, False),
        (True, "ValueError", 0.6, True),
        (True, "ValueError", None, True),
        (True, "TimeoutError", 0.0, False),
        (False, None, 0.99, True),
        (False, "ValueError", 0.99, True),
        (False, "TimeoutError", 0.0, False),
    ],
)
def test_periodic_checks_pass_through_outcome_gate(
    tmp_path, stage, enabled, exc_type, p_max, expected
):
    agent = _agent(tmp_path)
    agent.acfg.execution_gate = ExecutionGateConfig(enabled, 0.9, 0.7)
    prior = agent.journal.nodes[0]
    parent = None if stage == "draft" else prior
    if stage == "debug":
        parent = Node(code="broken parent", is_buggy=True)
        agent.journal.append(parent)
    node = Node(
        code="selected",
        parent=parent,
        exc_type=exc_type,
        is_buggy=exc_type is not None,
        sim_p_max=p_max,
    )
    node.metric = MetricValue(0.1, maximize=True)  # not the best candidate
    agent.journal.append(node)
    real = _Real(tmp_path, sim_only=True)
    assert validate_on_best(agent, agent.journal, real, tmp_path, periodic=True) is expected
    assert real.calls == int(expected)
    if expected:
        assert not validate_on_best(agent, agent.journal, real, tmp_path, periodic=True)
        assert real.calls == 1


def test_outcome_gate_uses_exception_instead_of_reviewer_bug_flag(tmp_path):
    agent = _agent(tmp_path)
    agent.acfg.execution_gate = ExecutionGateConfig(True, 0.9, 0.7)
    node = Node(code="reviewer rejected metric", is_buggy=True, exc_type=None, sim_p_max=0.8)
    node.metric = MetricValue(0.4, maximize=True)
    agent.journal.append(node)
    real = _Real(tmp_path, sim_only=True)
    assert validate_on_best(agent, agent.journal, real, tmp_path)
    assert real.calls == 1


def test_outcome_gate_never_scores_or_reexecutes_a_real_result(tmp_path):
    agent = _agent(tmp_path)
    agent.acfg.execution_gate = ExecutionGateConfig(True, 0.9, 0.7)
    agent._exc_scores = Mock(side_effect=AssertionError("already executed"))
    real = _Real(tmp_path, sim_only=True)
    agent.step(real.run)
    assert agent.journal.nodes[-1].real_won
    agent._exc_scores.assert_not_called()
    assert not validate_on_best(agent, agent.journal, real, tmp_path, force=True)
    assert real.calls == 1


def test_outcome_gate_respects_deadline_and_keeps_published_incumbent(tmp_path):
    agent = _agent(tmp_path)
    agent.acfg.execution_gate = ExecutionGateConfig(True, None, None)
    node = Node(code="selected", is_buggy=False, exc_type=None)
    node.metric = MetricValue(0.9, maximize=True)
    agent.journal.append(node)
    real = _Real(tmp_path, sim_only=True)
    assert not validate_on_best(agent, agent.journal, real, tmp_path, deadline=0, force=True)
    assert not node.real_won and real.calls == 0 and real.timeout == 5
    assert (tmp_path / "best_submission/submission.csv").read_text() == PRIOR_CSV


@pytest.mark.parametrize("trigger", ["best", "confidence", "forced"])
@pytest.mark.parametrize("posterior_case", ["runner_up", "argmax", "unavailable"])
def test_timeout_posterior_does_not_veto_real_checks(
    tmp_path,
    caplog,
    trigger,
    posterior_case,
):
    agent = _agent(tmp_path)
    prior = agent.journal.nodes[0]
    prior.metric = MetricValue(0.5 if trigger == "best" else 0.95, maximize=True)
    agent.search_policy = lambda: prior
    agent._improve_prompt = lambda _: {}
    agent.acfg.real_validation_p_max_threshold = 0.95
    p_max = 0.6 if trigger == "confidence" else 0.99
    posterior = (
        None
        if posterior_case == "unavailable"
        else {
            "max_p": p_max,
            "argmax_class": "TimeoutError" if posterior_case == "argmax" else "None",
            "p_clean": 0.001 if posterior_case == "argmax" else p_max,
            "top_exc": "TimeoutError",
            "p_top_exc": p_max if posterior_case == "argmax" else 0.001,
        }
    )
    agent._exc_scores = Mock(return_value=posterior)
    agent.step(
        lambda *args: ExecutionResult(
            ["Validation score: 0.9"], 0, None, predicted_submission_exists="true"
        )
    )
    node = agent.journal.nodes[-1]
    assert node.exc_type is None and not node.is_buggy
    real = _Real(tmp_path, sim_only=True)
    with caplog.at_level("INFO", logger="aide"):
        assert validate_on_best(agent, agent.journal, real, tmp_path, force=trigger == "forced")
    agent._exc_scores.assert_called_once_with(node.code)
    assert real.calls == 1 and node.real_won
    assert "skipping all real checks" not in caplog.text
    expected_csv = VALIDATED_CSV if trigger == "best" else PRIOR_CSV
    assert (tmp_path / "best_submission/submission.csv").read_text() == expected_csv
    restored = loads_json(dumps_json(agent.journal), Journal).nodes[-1]
    assert restored.sim_top_exc == node.sim_top_exc
    assert restored.sim_p_top_exc == node.sim_p_top_exc


@pytest.mark.parametrize("stage", ["draft", "debug", "improve"])
@pytest.mark.parametrize("confidence_threshold", [None, 0.95])
@pytest.mark.parametrize("force", [False, True])
def test_simulated_timeout_vetoes_every_real_check(
    tmp_path,
    caplog,
    stage,
    confidence_threshold,
    force,
):
    agent = _agent(tmp_path)
    prior = agent.journal.nodes[0]
    parent = None if stage == "draft" else prior
    if stage == "debug":
        parent = Node(code="broken parent", is_buggy=True)
        agent.journal.append(parent)
    agent.search_policy = lambda: parent
    agent._improve_prompt = lambda _: {}
    agent._build_debug_prompt = lambda *args: {}
    agent.acfg.real_validation_p_max_threshold = confidence_threshold
    # The decoded TimeoutError must win even when the posterior names another failure.
    agent._exc_scores = Mock(
        return_value={
            "max_p": 0.6,
            "top_exc": "ValueError",
            "p_top_exc": 0.3,
        }
    )
    agent.step(
        lambda *args: ExecutionResult(
            ["TimeoutError: simulated execution exceeded its limit"],
            0,
            "TimeoutError",
            predicted_submission_exists="false",
        )
    )
    real = _Real(tmp_path, sim_only=True)
    with caplog.at_level("INFO", logger="aide"):
        assert not validate_on_best(agent, agent.journal, real, tmp_path, force=force)
        assert not validate_on_best(agent, agent.journal, real, tmp_path, force=True)
    assert real.calls == 0
    node = agent.journal.nodes[-1]
    assert node.stage_name == stage
    assert not node.real_won and node.exc_type == "TimeoutError"
    assert node.is_buggy and node not in agent.journal.good_nodes
    assert agent._exc_scores.call_count == int(
        stage == "improve" and confidence_threshold is not None
    )
    assert "simulator predicted TimeoutError; skipping all real checks" in caplog.text
    assert "running real" not in caplog.text
    assert (tmp_path / "best_submission/submission.csv").read_text() == PRIOR_CSV


def test_simulated_timeout_does_not_block_later_real_validation(tmp_path):
    agent = _agent(tmp_path)
    agent.step(
        lambda *args: ExecutionResult(
            ["TimeoutError: simulated execution exceeded its limit"],
            0,
            "TimeoutError",
            predicted_submission_exists="false",
            predicted_metric=0.999,
        )
    )
    skipped = agent.journal.nodes[-1]
    assert skipped.is_buggy and skipped.metric.value is None
    assert skipped not in agent.journal.good_nodes
    real = _Real(tmp_path, sim_only=True)
    assert not validate_on_best(agent, agent.journal, real, tmp_path, force=True)
    assert real.calls == 0

    agent.step(
        lambda *args: ExecutionResult(
            ["Validation score: 0.9"], 0, None, predicted_submission_exists="true"
        )
    )
    eligible = agent.journal.nodes[-1]
    assert validate_on_best(agent, agent.journal, real, tmp_path)
    assert real.calls == 1
    assert eligible.real_won and not skipped.real_won
    assert (tmp_path / "best_submission/submission.csv").read_text() == VALIDATED_CSV


def test_legacy_journal_without_timeout_posterior_keeps_confidence_check(tmp_path):
    agent = _agent(tmp_path)
    prior = agent.journal.nodes[0]
    prior.metric = MetricValue(0.95, maximize=True)
    node = Node(
        code="legacy simulated improve",
        parent=prior,
        is_buggy=False,
        metric=MetricValue(0.9, maximize=True),
        sim_p_max=0.6,
    )
    agent.journal.append(node)
    payload = json.loads(dumps_json(agent.journal))
    for row in payload["nodes"]:
        row.pop("sim_top_exc")
        row.pop("sim_p_top_exc")
    agent.journal = loads_json(json.dumps(payload), Journal)
    restored = agent.journal.nodes[-1]
    assert restored.sim_top_exc is restored.sim_p_top_exc is None
    agent.acfg.real_validation_p_max_threshold = 0.95
    (tmp_path / "submission").mkdir()
    real = _Real(tmp_path, sim_only=True)
    assert validate_on_best(agent, agent.journal, real, tmp_path)
    assert real.calls == 1
    assert restored.real_won


@pytest.mark.parametrize(
    "best,remaining,expected",
    [
        (False, 0, False),
        (False, 12.8, True),
        (True, 12.8, True),
    ],
)
def test_non_timeout_best_and_confidence_checks_respect_deadline(
    tmp_path,
    monkeypatch,
    best,
    remaining,
    expected,
):
    agent = _agent(tmp_path)
    prior = agent.journal.nodes[0]
    prior.metric = MetricValue(0.5 if best else 0.95, maximize=True)
    agent.search_policy = lambda: prior
    agent._improve_prompt = lambda _: {}
    agent.acfg.real_validation_p_max_threshold = 0.95
    agent._exc_scores = Mock(
        return_value={
            "max_p": 0.6,
            "top_exc": "ValueError",
            "p_top_exc": 0.3,
        }
    )
    agent.step(
        lambda *args: ExecutionResult(
            ["Validation score: 0.9"], 0, None, predicted_submission_exists="true"
        )
    )
    real = _Real(tmp_path, sim_only=True)
    real.timeout = 3400
    execute = real.run
    timeouts = []

    def run(*args, **kwargs):
        timeouts.append(real.timeout)
        return execute(*args, **kwargs)

    real.run = run
    monkeypatch.setattr("aide.run.time.time", lambda: 1000)
    assert (
        validate_on_best(
            agent, agent.journal, real, tmp_path, deadline=1180 + remaining, reserve_s=180
        )
        is expected
    )
    assert timeouts == ([12] if expected else [])
    assert real.timeout == 3400


def test_confidence_scoring_failure_falls_back_to_real_once(tmp_path, caplog):
    agent = _agent(tmp_path)
    prior = agent.journal.nodes[0]
    prior.metric = MetricValue(0.95, maximize=True)
    agent.search_policy = lambda: prior
    agent._improve_prompt = lambda _: {}
    agent.acfg.real_validation_p_max_threshold = 0.95
    agent._exc_scores = Mock(side_effect=RuntimeError("posterior unavailable"))
    agent.step(
        lambda *args: ExecutionResult(
            ["Validation score: 0.9"], 0, None, predicted_submission_exists="true"
        )
    )
    real = _Real(tmp_path, sim_only=True)
    with caplog.at_level("INFO", logger="aide"):
        assert validate_on_best(agent, agent.journal, real, tmp_path)
    assert "p_max=unavailable" in caplog.text
    agent._exc_scores.assert_called_once_with(agent.journal.nodes[-1].code)
    assert real.calls == 1


def test_explicit_format_rejection_is_returned_to_debugging(tmp_path, monkeypatch):
    agent = _agent(tmp_path)
    agent._validate_submission = Agent._validate_submission.__get__(agent)
    monkeypatch.setattr(
        "requests.post",
        lambda *args, **kwargs: SimpleNamespace(
            status_code=200,
            json=lambda: {"result": "Submission invalid! Submission should have a Insult column"},
        ),
    )
    agent.step(_Real(tmp_path, sim_only=True).run)
    assert agent.journal.nodes[-1].is_buggy
    assert "Insult column" in agent.journal.nodes[-1].analysis
    assert (tmp_path / "best_submission/submission.csv").read_text() == PRIOR_CSV


def test_grading_server_error_keeps_candidate_unverified_and_preserves_best(tmp_path, monkeypatch):
    agent = _agent(tmp_path)
    agent._validate_submission = Agent._validate_submission.__get__(agent)
    monkeypatch.setattr(
        "requests.post",
        lambda *args, **kwargs: SimpleNamespace(
            status_code=500,
            text="Grader unavailable",
        ),
    )
    agent.step(_Real(tmp_path, sim_only=True).run)
    assert not agent.journal.nodes[-1].is_buggy
    node = agent.journal.nodes[-1]
    assert node.submission_validation == "unknown"
    assert agent.journal.get_best_node(validated_only=True).code == "prior solution"
    assert (tmp_path / "best_submission/submission.csv").read_text() == PRIOR_CSV
    assert (tmp_path / "best_solution/solution.py").read_text() == "prior solution"


def test_failed_real_rerun_cannot_publish_leftover_csv(tmp_path):
    agent = _agent(tmp_path)
    agent.step(
        lambda *args: ExecutionResult(
            ["Validation score: 0.9"], 0, None, predicted_submission_exists="true"
        )
    )
    (tmp_path / "submission/submission.csv").write_text(VALIDATED_CSV)
    real = _Real(tmp_path, sim_only=True)
    real.run = lambda *args, **kwargs: ExecutionResult(
        ["Validation score: 0.9"], 1, None, real_won=True
    )
    validate_on_best(agent, agent.journal, real, tmp_path)
    assert agent.journal.nodes[-1].is_buggy
    assert (tmp_path / "best_submission/submission.csv").read_text() == PRIOR_CSV


@pytest.mark.parametrize("valid", [False, True])
def test_validation_uses_shared_format_check_and_publication(tmp_path, monkeypatch, valid):
    agent = _agent(tmp_path)
    agent._validate_submission.side_effect = lambda path, **kwargs: SubmissionValidation(
        "valid" if valid else "invalid",
        "Submission is valid." if valid else "Submission invalid!",
        submission_sha256(path),
    )
    agent.step(
        lambda *args: ExecutionResult(
            ["Validation score: 0.9"], 0, None, predicted_submission_exists="true"
        )
    )
    real = _Real(tmp_path, sim_only=True)
    monkeypatch.setattr("aide.run.time.time", lambda: 1000)
    assert validate_on_best(agent, agent.journal, real, tmp_path, deadline=1300)
    assert real.timeout == 5
    agent._grade_test_submission.assert_called_once_with()
    agent._validate_submission.assert_called_once_with(
        tmp_path / "submission/submission.csv", timeout_s=75.0
    )
    assert (tmp_path / "best_submission/submission.csv").read_text() == (
        VALIDATED_CSV if valid else PRIOR_CSV
    )
    assert (tmp_path / "best_solution/solution.py").read_text() == (
        "candidate solution" if valid else "prior solution"
    )


def test_validation_restores_timeout_after_interpreter_error(tmp_path, monkeypatch):
    agent = _agent(tmp_path)
    agent.step(
        lambda *args: ExecutionResult(
            ["Validation score: 0.9"], 0, None, predicted_submission_exists="true"
        )
    )
    real = _Real(tmp_path, sim_only=True)
    real.timeout = 3400
    real.run = Mock(side_effect=RuntimeError("interpreter failed"))
    real.cleanup_session = Mock()
    monkeypatch.setattr("aide.run.time.time", lambda: 1000)
    assert not validate_on_best(agent, agent.journal, real, tmp_path, deadline=1200)
    assert real.run.call_count == 2
    assert real.cleanup_session.call_count == 2
    assert real.timeout == 3400
    node = agent.journal.nodes[-1]
    assert not node.real_won
    assert not node.is_buggy
    assert node.metric.value == 0.9
    assert all("interpreter failed" in reason for reason in node.failure_reasons["real_validation"])
    assert (tmp_path / "best_submission/submission.csv").read_text() == PRIOR_CSV


@pytest.mark.parametrize("force", [False, True])
def test_failed_validation_does_not_block_later_validation(tmp_path, force):
    agent = _agent(tmp_path)
    agent.step(
        lambda *args: ExecutionResult(
            ["Validation score: 0.9"], 0, None, predicted_submission_exists="true"
        )
    )
    failed = agent.journal.nodes[-1]
    real = _Real(tmp_path, sim_only=True)
    execute = real.run
    failed_run = Mock(side_effect=RuntimeError("interpreter failed"))
    real.run = failed_run

    assert not validate_on_best(agent, agent.journal, real, tmp_path)
    assert not validate_on_best(agent, agent.journal, real, tmp_path)
    assert failed_run.call_count == 2
    assert not failed.real_won
    assert not failed.is_buggy
    assert failed.metric.value == 0.9
    assert failed.failure_reasons["real_validation"]
    assert agent.journal.get_best_node() is failed

    if not force:
        agent.generate_code = lambda *args, **kwargs: ("plan", "second candidate", None)
        agent.step(
            lambda *args: ExecutionResult(
                ["Validation score: 0.8"], 0, None, predicted_submission_exists="true"
            )
        )
        assert agent.journal.get_best_node() is failed
    real.run = Mock(side_effect=execute)
    candidate = agent.journal.nodes[-1]
    assert validate_on_best(agent, agent.journal, real, tmp_path, force=force)
    real.run.assert_called_once_with(candidate.code)
    assert candidate.real_won
    assert not candidate.is_buggy
    assert "real_validation" not in candidate.failure_reasons
    assert agent.journal.get_best_node(validated_only=True) is candidate
    assert (tmp_path / "best_submission/submission.csv").read_text() == VALIDATED_CSV


@pytest.mark.parametrize("exhaust_budget", [False, True])
def test_validation_retry_rechecks_budget_and_cleans_partial_csv(
    tmp_path, monkeypatch, exhaust_budget
):
    agent = _agent(tmp_path)
    agent.step(
        lambda *args: ExecutionResult(
            ["Validation score: 0.9"], 0, None, predicted_submission_exists="true"
        )
    )
    node = agent.journal.nodes[-1]
    real = _Real(tmp_path, sim_only=True)
    real.timeout = 3400
    real.cleanup_session = Mock()
    execute = real.run
    now = [1000]
    timeouts = []
    monkeypatch.setattr("aide.run.time.time", lambda: now[0])

    def run(code):
        timeouts.append(real.timeout)
        submission = tmp_path / "submission/submission.csv"
        if len(timeouts) == 1:
            submission.write_text("incomplete submission")
            now[0] += 20 if exhaust_budget else 7
            raise RuntimeError("execution queue disconnected")
        real.cleanup_session.assert_called_once_with()
        assert not submission.exists()
        return execute(code)

    real.run = Mock(side_effect=run)
    assert validate_on_best(agent, agent.journal, real, tmp_path, deadline=1200) is (
        not exhaust_budget
    )
    assert timeouts == ([20] if exhaust_budget else [20, 13])
    real.cleanup_session.assert_called_once_with()
    assert real.timeout == 3400
    assert not node.is_buggy
    assert node.real_won is (not exhaust_budget)
    if exhaust_budget:
        assert node.failure_reasons["real_validation"]
        assert node.metric.value == 0.9
    else:
        assert "real_validation" not in node.failure_reasons
        assert node.metric.value == 0.8
        assert node.sim_metric == 0.9
    assert (tmp_path / "best_submission/submission.csv").read_text() == (
        PRIOR_CSV if exhaust_budget else VALIDATED_CSV
    )


@pytest.mark.parametrize("exc_type", ["TimeoutError", "ValueError"])
def test_validation_does_not_retry_observed_execution_errors(tmp_path, exc_type):
    agent = _agent(tmp_path)
    agent.step(
        lambda *args: ExecutionResult(
            ["Validation score: 0.9"], 0, None, predicted_submission_exists="true"
        )
    )
    real = _Real(tmp_path, sim_only=True)
    real.run = Mock(return_value=ExecutionResult(["Training failed"], 2, exc_type))

    assert validate_on_best(agent, agent.journal, real, tmp_path)

    node = agent.journal.nodes[-1]
    real.run.assert_called_once_with(node.code)
    assert node.real_won
    assert node.is_buggy
    assert node.exc_type == exc_type
    assert "real_validation" not in node.failure_reasons
    assert (tmp_path / "best_submission/submission.csv").read_text() == PRIOR_CSV


@pytest.mark.parametrize("exc_type", [None, "ValueError"])
@pytest.mark.parametrize("review_error", [RuntimeError, KeyboardInterrupt])
def test_validation_review_error_invalidates_simulated_evaluation(
    tmp_path, monkeypatch, exc_type, review_error
):
    agent = _agent(tmp_path)
    agent.step(
        lambda *args: ExecutionResult(
            ["Validation score: 0.9"], 0.1, None, predicted_submission_exists="true"
        )
    )
    node = agent.journal.nodes[-1]
    real = _Real(tmp_path, sim_only=True)
    real.timeout = 3400
    result = ExecutionResult(
        ["ValueError: training failed" if exc_type else "Validation score: 0.8"],
        12.0,
        exc_type,
        real_won=True,
        timeout_s=20,
    )

    def execute(code):
        if exc_type is None:
            (tmp_path / "submission/submission.csv").write_text(VALIDATED_CSV)
        return result

    real.run = Mock(side_effect=execute)
    agent._review = Mock(side_effect=review_error("feedback backend unavailable"))
    monkeypatch.setattr("aide.run.time.time", lambda: 1000)

    if review_error is KeyboardInterrupt:
        with pytest.raises(KeyboardInterrupt, match="feedback backend unavailable"):
            validate_on_best(agent, agent.journal, real, tmp_path, deadline=1200)
    else:
        assert validate_on_best(agent, agent.journal, real, tmp_path, deadline=1200)

    assert node.real_won
    assert node.exc_type == exc_type
    assert node.term_out == "".join(result.term_out)
    assert node.exec_time == 12.0
    assert node.timeout_s == 20
    assert node.is_buggy
    assert node.metric.is_worst
    assert node.sim_metric == 0.9
    assert node.metric_trustworthy is None
    assert node.csv_exists is None
    assert node.test_metric is None
    assert node.leakage_check is None
    assert node.submission_validation is None
    assert node.submission_sha256 is None
    assert "feedback backend unavailable" in node.analysis
    assert "feedback backend unavailable" in node.failure_reasons["review"][0]
    if exc_type:
        assert exc_type in node.failure_reasons["execution"][0]
    assert node not in agent.journal.good_nodes
    assert agent.journal.get_best_node().code == "prior solution"
    assert not validate_on_best(agent, agent.journal, real, tmp_path, force=True)
    real.run.assert_called_once_with(node.code)
    assert real.timeout == 3400
    assert (tmp_path / "best_submission/submission.csv").read_text() == PRIOR_CSV
    assert (tmp_path / "best_solution/solution.py").read_text() == "prior solution"


def test_validation_publication_error_preserves_completed_review(tmp_path, monkeypatch):
    agent = _agent(tmp_path)
    agent.step(
        lambda *args: ExecutionResult(
            ["Validation score: 0.9"], 0.1, None, predicted_submission_exists="true"
        )
    )
    real = _Real(tmp_path, sim_only=True)
    publish = Mock(side_effect=OSError("disk full"))
    monkeypatch.setattr("aide.real_validation.publish_real_solution", publish)

    assert validate_on_best(agent, agent.journal, real, tmp_path)

    node = agent.journal.nodes[-1]
    publish.assert_called_once_with(tmp_path, node)
    assert node.real_won
    assert not node.is_buggy
    assert node.metric.value == 0.8
    assert node.sim_metric == 0.9
    assert node.submission_validation == "valid"
    assert node.submission_sha256
    assert agent.journal.get_best_node(validated_only=True) is node
    assert real.timeout == 5
    assert (tmp_path / "best_submission/submission.csv").read_text() == PRIOR_CSV
    assert (tmp_path / "best_solution/solution.py").read_text() == "prior solution"


def test_publisher_stages_all_files_before_replacing_any(tmp_path, monkeypatch):
    from aide.utils.artifacts import publish_real_solution

    agent = _agent(tmp_path)
    (tmp_path / "submission").mkdir()
    (tmp_path / "submission/submission.csv").write_text(VALIDATED_CSV)
    node = Node(code="replacement solution")
    node.real_won, node.is_buggy = True, False
    node.submission_validation = "valid"
    node.submission_sha256 = submission_sha256(tmp_path / "submission/submission.csv")
    monkeypatch.setattr(
        "aide.utils.artifacts.shutil.copyfileobj", Mock(side_effect=OSError("disk full"))
    )
    with pytest.raises(OSError, match="disk full"):
        publish_real_solution(tmp_path, node)
    assert (tmp_path / "best_solution/solution.py").read_text() == "prior solution"
    assert (tmp_path / "best_submission/submission.csv").read_text() == PRIOR_CSV
    assert not list(tmp_path.glob("*/.publish-*"))


def test_publisher_keeps_harness_directory_symlinks(tmp_path):
    from aide.utils.artifacts import publish_real_solution

    for name in ["best_solution", "best_submission"]:
        target = tmp_path / (name + "_mount")
        target.mkdir()
        (tmp_path / name).symlink_to(target, target_is_directory=True)
    (tmp_path / "submission").mkdir()
    (tmp_path / "submission/submission.csv").write_text(VALIDATED_CSV)
    node = Node(code="confirmed solution")
    node.real_won, node.is_buggy = True, False
    node.submission_validation = "valid"
    node.submission_sha256 = submission_sha256(tmp_path / "submission/submission.csv")
    assert publish_real_solution(tmp_path, node)
    assert (tmp_path / "best_solution").is_symlink()
    assert (tmp_path / "best_submission").is_symlink()
    assert (tmp_path / "best_solution_mount/solution.py").read_text() == node.code
    assert (tmp_path / "best_submission_mount/submission.csv").read_text() == VALIDATED_CSV


@pytest.mark.parametrize("prediction", ["true", "false", "unknown", None])
@pytest.mark.parametrize("artifact", [None, "", "id,pred\n0,999\n"])
def test_prediction_controls_csv_gate_regardless_of_disk_or_reviewer(
    tmp_path, prediction, artifact
):
    agent = _agent(tmp_path)
    review = agent._review

    def contradictory_review(*args, **kwargs):
        return {**review(*args, **kwargs), "has_csv_submission": prediction != "true"}

    agent._review = contradictory_review

    def execute(code, reset_session=True):
        if artifact is not None:
            (tmp_path / "submission/submission.csv").write_text(artifact)
        return ExecutionResult(
            ["Validation score: 0.9"],
            0.0,
            None,
            predicted_submission_exists=prediction,
        )

    agent.step(execute)
    node = agent.journal.nodes[-1]
    assert node.csv_exists is (prediction == "true")
    assert node.is_buggy is (prediction != "true")
    assert (tmp_path / "best_submission/submission.csv").read_text() == PRIOR_CSV
    assert (tmp_path / "best_solution/solution.py").read_text() == "prior solution"
    agent._grade_test_submission.assert_not_called()
    agent._validate_submission.assert_not_called()


@pytest.mark.parametrize("prediction", ["true", "false", "unknown", None])
@pytest.mark.parametrize(
    "artifact,exists",
    [
        ("missing", False),
        ("empty", False),
        ("directory", False),
        ("csv", True),
    ],
)
def test_real_csv_gate_uses_disk_even_when_prediction_disagrees(
    tmp_path, prediction, artifact, exists
):
    from aide.agent import _result_has_submission_csv

    submission = tmp_path / "submission/submission.csv"
    submission.parent.mkdir()
    if artifact == "empty":
        submission.touch()
    elif artifact == "directory":
        submission.mkdir()
    elif artifact == "csv":
        submission.write_text(VALIDATED_CSV)
    result = ExecutionResult([], 0.1, None, real_won=True, predicted_submission_exists=prediction)
    assert _result_has_submission_csv(tmp_path, result) is exists


@pytest.mark.parametrize("error", [FileNotFoundError, PermissionError, OSError])
def test_real_csv_gate_rejects_unavailable_file_without_crashing(tmp_path, monkeypatch, error):
    from aide.agent import _result_has_submission_csv

    submission = tmp_path / "submission/submission.csv"
    original_stat = Path.stat

    def unavailable(path, *args, **kwargs):
        if path == submission:
            raise error("submission unavailable")
        return original_stat(path, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", unavailable)
    result = ExecutionResult([], 0.1, None, real_won=True)
    assert _result_has_submission_csv(tmp_path, result) is False


def test_unknown_submission_survives_simulated_steps_until_format_retry(tmp_path, monkeypatch):
    from aide.utils.submission_recovery import retry_pending_submissions

    agent = _agent(tmp_path)
    agent._validate_submission = Agent._validate_submission.__get__(agent)
    now = [100.0]
    monkeypatch.setattr("aide.utils.submission_validation.time.monotonic", lambda: now[0])
    post = Mock(return_value=SimpleNamespace(status_code=503))
    monkeypatch.setattr("requests.post", post)
    real = _Real(tmp_path, sim_only=True)
    agent.step(real.run)
    node = agent.journal.nodes[-1]
    assert node.submission_validation == "unknown"
    assert not node.is_buggy
    # A simulated step clears the working file. Recovery must use the retained
    # real CSV, without treating simulation as format evidence or rerunning code.
    post.return_value = SimpleNamespace(
        status_code=200, json=lambda: {"result": "Submission is valid."}
    )
    now[0] += 61
    agent.step(
        lambda *args: ExecutionResult(
            ["Validation score: 0.9"], 0.1, None, predicted_submission_exists="true"
        )
    )
    post.assert_called_once()
    assert real.calls == 1
    assert node.submission_validation == "unknown"
    assert not node.is_buggy and node.metric.value == 0.8
    assert Path(node.pending_submission["path"]).read_text() == VALIDATED_CSV
    assert agent.journal.get_best_node(validated_only=True).code == "prior solution"
    assert (tmp_path / "best_submission/submission.csv").read_text() == PRIOR_CSV
    assert (tmp_path / "best_solution/solution.py").read_text() == "prior solution"
    node.pending_submission["retry_after"] = 0
    assert retry_pending_submissions(agent) == 1
    assert post.call_count == 2 and real.calls == 1
    assert node.submission_validation == "valid" and node.pending_submission is None
    assert agent.journal.get_best_node(validated_only=True) is node
    assert (tmp_path / "best_submission/submission.csv").read_text() == VALIDATED_CSV
    assert not (tmp_path / "submission/submission.csv").exists()


def test_publication_rejects_bytes_changed_after_validation(tmp_path):
    from aide.utils.artifacts import publish_real_solution

    agent = _agent(tmp_path)
    (tmp_path / "submission").mkdir()
    path = tmp_path / "submission/submission.csv"
    path.write_text(VALIDATED_CSV)
    node = Node(code="candidate")
    node.real_won, node.is_buggy = True, False
    node.submission_validation = "valid"
    node.submission_sha256 = submission_sha256(path)
    path.write_text("id,pred\n0,999\n")
    assert not publish_real_solution(tmp_path, node)
    assert node.submission_validation == "unknown"
    assert (tmp_path / "best_submission/submission.csv").read_text() == PRIOR_CSV
    assert (tmp_path / "best_solution/solution.py").read_text() == "prior solution"
    assert not list(tmp_path.glob("*/.publish-*"))


@pytest.mark.parametrize(
    "statuses,usable",
    [
        (["invalid", "invalid"], [False, False]),
        (["invalid", "unknown"], [False, True]),
        (["valid", "unknown"], [True, False]),
    ],
)
def test_real_candidate_gate_never_relaxes_explicit_rejections(
    tmp_path, monkeypatch, statuses, usable
):
    agent = _agent(tmp_path)
    agent.deadline = float("inf")
    agent.cfg.exec = SimpleNamespace(
        timeout=10, format_tb_ipython=False, agent_file_name="runfile.py"
    )

    class RealCandidate:
        def __init__(self, workspace, **kwargs):
            self.workspace = workspace

        def run(self, code, **kwargs):
            (self.workspace / "submission/submission.csv").write_text(VALIDATED_CSV)
            return ExecutionResult(["Validation score: 0.8"], 0.1, None, real_won=True)

        def cleanup_session(self):
            pass

    monkeypatch.setattr("aide.agent.Interpreter", RealCandidate)
    verdicts = iter(statuses)

    def validate(path, **kwargs):
        status = next(verdicts)
        return SubmissionValidation(
            status,
            "Submission invalid! Wrong IDs." if status == "invalid" else "format verdict",
            submission_sha256(path),
        )

    agent._validate_submission.side_effect = validate
    outcomes = agent._observe_real([("plan", "code"), ("plan", "code")], tmp_path / "logs")
    assert [o.usable for o in outcomes] == usable
    if statuses == ["invalid", "invalid"]:
        candidates = [("plan", "code"), ("plan", "code")]
        plan, code, result = agent._commit(candidates, outcomes, 0, tmp_path / "logs")
        node = agent.parse_exec_result(Node(code=code, plan=plan), result)
        assert node.is_buggy
        assert "Wrong IDs" in node.analysis
        assert (
            agent._validate_submission.call_count == 2
        )  # committed parser reuses the candidate's evidence
        assert (tmp_path / "best_submission/submission.csv").read_text() == PRIOR_CSV


@pytest.mark.parametrize("sim_first", [True, False])
@pytest.mark.parametrize("leakage_detected", [True, False])
@pytest.mark.parametrize("prior_score", [0.7, 0.8])
def test_candidate_publication_follows_leakage_verdict(
    tmp_path,
    monkeypatch,
    sim_first,
    leakage_detected,
    prior_score,
):
    agent = _agent(tmp_path)
    agent.journal.nodes[0].metric = MetricValue(prior_score, maximize=True)
    agent.task_desc = "Predict the target for each test ID."
    agent.acfg.check_data_leakage = True
    agent.acfg.leakage_max_threshold = 0.999
    agent.acfg.leakage_min_threshold = 1e-6
    agent.acfg.feedback = SimpleNamespace(model="gpt-coderbf16-fb-vllm", temp=0.5)
    agent.acfg.eval_metric = "accuracy"
    agent.acfg.convert_system_to_user = False
    check = Mock(
        return_value={
            "status": "checked",
            "decision": "reject" if leakage_detected else "allow",
            "has_leakage": leakage_detected,
            "reason": (
                "Validation rows enter fitting. Repair: fit on training rows only."
                if leakage_detected
                else "No validation contamination established."
            ),
        }
    )
    monkeypatch.setattr("aide.submission_review.check_validation", check)
    real = _Real(tmp_path, sim_only=True)
    if sim_first:
        agent.step(
            lambda *_: ExecutionResult(
                ["Validation score: 0.9"],
                0.1,
                None,
                predicted_submission_exists="true",
            )
        )
        check.assert_not_called()
        assert validate_on_best(agent, agent.journal, real, tmp_path)
    else:
        agent.step(real.run)
    check.assert_called_once()
    candidate = agent.journal.nodes[-1]
    assert candidate.leakage_check["triggers"] == ["best_real_candidate"]
    if leakage_detected:
        assert candidate.is_buggy and set(candidate.failure_reasons) == {"metric"}
        assert agent.journal.get_best_node(validated_only=True) is agent.journal.nodes[0]
        assert (tmp_path / "best_submission/submission.csv").read_text() == PRIOR_CSV
        assert (tmp_path / "best_solution/solution.py").read_text() == "prior solution"
    else:
        assert not candidate.is_buggy and not candidate.failure_reasons
        assert candidate.submission_validation == "valid"
        # Direct execution and validate_on_best both preserve tied incumbents,
        # while still accepting the clean node as real validation feedback.
        publishes = prior_score < 0.8
        assert (tmp_path / "best_submission/submission.csv").read_text() == (
            VALIDATED_CSV if publishes else PRIOR_CSV
        )
        assert (tmp_path / "best_solution/solution.py").read_text() == (
            candidate.code if publishes else "prior solution"
        )
