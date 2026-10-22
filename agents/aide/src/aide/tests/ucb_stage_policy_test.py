"""UCB uses the production routing, prediction, selection, and commit path."""

import json
import math
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from omegaconf import OmegaConf

from aide.agent import Agent, CandidateOutcome, STAGES, confidence_gate_stages
from aide.interpreter import ExecutionResult
from aide.journal import Journal, Node
from aide.parallel_interpreter import ParallelInterpreter
from aide.run import validate_on_best
from aide.simulator import _EXC_SUPPORT, _finalize_exc_posterior
from aide.ucb import resolve_config
from aide.utils.config import AgentConfig
from aide.utils.metric import MetricValue, WorstMetricValue
from aide.utils.submission_validation import SubmissionValidation, submission_sha256


CODES = [(f"plan {i}", f"candidate_{i} = True") for i in range(4)]


def _agent(tmp_path, stage="improve", *, beta=1.0, sigma="p_max", policy="ucb"):
    path = Path(__file__).resolve().parents[1] / "utils/config.yaml"
    config = OmegaConf.load(path)
    config.agent = OmegaConf.merge(OmegaConf.structured(AgentConfig), config.agent)
    config.agent.stage_policy[stage] = policy
    config.agent.ucb.beta = beta
    config.agent.ucb.sigma = sigma
    config.agent.ucb.metric_scale = 1.0
    config.agent.eval_lower_is_better = False
    config.agent.bon_candidate_review = False
    config.agent.use_p_clean_argmax_pick = False
    config.agent.use_best_of_n_listwise = True
    cfg = SimpleNamespace(agent=config.agent, exec=config.exec, log_dir=tmp_path)
    metrics = [0.90, 0.85, 0.70, 0.65]
    posteriors = [0.99, 0.60, 0.99, 0.99]
    predictions = {
        code: ExecutionResult(
            [f"Validation: {metrics[i]}\n"],
            0,
            None,
            predicted_submission_exists="true",
            predicted_metric=metrics[i],
        )
        for i, (_, code) in enumerate(CODES)
    }
    scores = {
        code: {
            "p_clean": posteriors[i],
            "max_p": posteriors[i],
            "top_exc": "TimeoutError",
            "p_top_exc": 0.3 if posteriors[i] < 0.9 else 0.005,
            "entropy": (1 - posteriors[i]) * math.log(24),
            "support_size": 24,
            "score": 0.0,
            "score_name": "p_clean",
        }
        for i, (_, code) in enumerate(CODES)
    }
    scorer = Mock(side_effect=lambda code: scores[code])
    agent = Agent("task", cfg, Journal(), sim_predict=predictions.__getitem__, sim_p_clean=scorer)
    agent._generate_n_candidates = Mock(return_value=CODES)
    agent._judge_query = Mock(side_effect=AssertionError("UCB must not call the listwise judge"))
    agent._review = Mock(side_effect=AssertionError("structured metric needs no reviewer"))
    agent._grade_test_submission = Mock(side_effect=AssertionError("selection must not grade test"))
    return agent, scorer, predictions


@pytest.mark.parametrize("stage", STAGES)
@pytest.mark.parametrize("sigma", ["p_clean", "p_max", "entropy"])
def test_ucb_routes_for_each_stage_and_commits_one_of_four(tmp_path, stage, sigma):
    agent, scorer, _ = _agent(tmp_path, stage, sigma=sigma)
    assert agent.stage_policy == {s: "ucb" if s == stage else "single" for s in STAGES}
    assert agent._exc_scores is scorer
    assert agent._p_clean is None
    plan, code, result = agent.generate_code({"task": "t"}, stage)
    assert (plan, code) == CODES[1]
    assert result is None  # the ordinary step executor runs this winner
    assert agent._generate_n_candidates.call_count == 1
    assert agent._generate_n_candidates.call_args.args[1] == 4
    assert scorer.call_count == 4
    audit_dir = tmp_path / "bon/step_0"
    audit = json.loads((audit_dir / "ucb.json").read_text())
    assert audit["stage"] == stage
    assert audit["winner"] == 1 and audit["greedy_winner"] == 0
    assert all(r["metric_source"] == "world_model" for r in audit["candidates"])
    accepted = [
        json.loads((audit_dir / f"cand_{i}/result.json").read_text())["accepted"] for i in range(4)
    ]
    assert accepted == [False, True, False, False]
    assert (audit_dir / "picked.txt").read_text() == "1"
    agent._judge_query.assert_not_called()
    agent._review.assert_not_called()
    agent._grade_test_submission.assert_not_called()


def test_zero_beta_skips_posterior_calls_and_selects_best_metric(tmp_path):
    agent, scorer, _ = _agent(tmp_path, beta=0)
    assert agent.generate_code({}, "improve")[1] == CODES[0][1]
    scorer.assert_not_called()


@pytest.mark.parametrize("beta,predicted_failure", [(0, False), (1, False), (1, True)])
@pytest.mark.parametrize("confidence_threshold", [None, 1.0])
@pytest.mark.parametrize("outcome_gate", [False, True])
def test_sim_only_reuses_selected_prediction_then_validates_once(
    tmp_path, beta, predicted_failure, confidence_threshold, outcome_gate,
):
    agent, scorer, predictions = _agent(tmp_path, beta=beta)
    agent.cfg.exec.simulator = True
    agent.cfg.exec.simulator_only = True
    agent.acfg.real_validation_p_max_threshold = confidence_threshold
    if outcome_gate:
        agent.acfg.execution_gate.enabled = confidence_threshold is not None
        agent.acfg.execution_gate.success_threshold = confidence_threshold
        agent.acfg.execution_gate.failure_threshold = confidence_threshold
        agent.acfg.real_validation_p_max_threshold = None
    agent.cfg.workspace_dir = tmp_path / "workspace"
    agent.cfg.workspace_dir.mkdir()
    agent.acfg.check_data_leakage = False
    agent.acfg.record_test_metric = False
    agent.acfg.search.num_drafts = 1
    agent.acfg.search.debug_prob = 0
    agent.data_preview = "test data"
    agent._improve_prompt = Mock(return_value={"task": "t"})
    parent = Node(code="prior solution", is_buggy=False)
    parent.metric = MetricValue(0.92, maximize=True)
    parent.real_won = True
    parent.submission_validation = "valid"
    parent.submission_sha256 = "prior-validated-csv"
    agent.journal.append(parent)
    if predicted_failure:
        for result in predictions.values():
            result.exc_type = "ValueError"
            result.predicted_submission_exists = "false"
            result.predicted_metric = None
            result.term_out = ["ValueError: predicted training failure"]
    predictor = Mock(side_effect=predictions.__getitem__)
    agent._predict = predictor
    agent._review = Mock(side_effect=lambda code, output, **kwargs: {
        "is_bug": "ValueError" in output,
        "has_csv_submission": "ValueError" not in output,
        "metric": None if "ValueError" in output else float(output.split(": ")[1]),
        "metric_trustworthy": True,
        "lower_is_better": False,
        "summary": "Review of the supplied output.",
    })
    agent._validate_submission = Mock(side_effect=lambda path, **kwargs: SubmissionValidation(
        "valid", "Submission is valid.", submission_sha256(path)
    ))
    # A repeated WM call would disagree and must never reach this interpreter.
    repeated_wm = SimpleNamespace(run=Mock(return_value=ExecutionResult(
        ["Validation: 0.1\n"], 0, None,
        predicted_submission_exists="true", predicted_metric=0.1,
    )))
    workspace = agent.cfg.workspace_dir

    def execute_real(code, reset_session=True):
        (workspace / "submission/submission.csv").write_text("id,pred\n0,0.95\n")
        return ExecutionResult(["Validation: 0.95\n"], 0.1, None, real_won=True)

    real = SimpleNamespace(working_dir=workspace, timeout=5, run=Mock(side_effect=execute_real))
    executor = ParallelInterpreter(real, repeated_wm, sim_only=True)
    try:
        agent.step(executor.run)
        node = agent.journal.nodes[-1]
        expected = CODES[1 if beta and not predicted_failure else 0][1]
        selected_result = predictions[expected]
        assert node.code == expected
        assert predictor.call_count == 4
        repeated_wm.run.assert_not_called()
        real.run.assert_not_called()
        assert node.real_won is False
        assert node.is_buggy is predicted_failure
        assert node.metric.value == selected_result.predicted_metric
        assert node.term_out == "".join(selected_result.term_out)
        assert agent._review.call_count == 1
        assert scorer.call_count == (4 if beta else int(confidence_threshold is not None))
        assert node.sim_p_max == (
            (0.60 if beta and not predicted_failure else 0.99)
            if confidence_threshold is not None else None
        )
        assert node.sim_top_exc == ("TimeoutError" if confidence_threshold is not None else None)
        assert node.sim_p_top_exc == (
            (0.3 if beta and not predicted_failure else 0.005)
            if confidence_threshold is not None else None
        )
        assert not (workspace / "best_submission/submission.csv").exists()
        agent._validate_submission.assert_not_called()
        agent._grade_test_submission.assert_not_called()

        # Either the interval force or confidence alone checks non-best/failure picks.
        ran = validate_on_best(
            agent, agent.journal, real, workspace, force=confidence_threshold is None
        )
        assert ran is True
        real.run.assert_called_once_with(expected)
        assert node.real_won is True and node.is_buggy is False
        assert node.sim_metric == selected_result.predicted_metric
        assert node.metric.value == 0.95
        assert agent._review.call_count == 2
        assert (workspace / "best_submission/submission.csv").read_text() == "id,pred\n0,0.95\n"
        assert (workspace / "best_solution/node_id.txt").read_text() == node.id
        assert validate_on_best(agent, agent.journal, real, workspace, force=True) is False
        real.run.assert_called_once_with(expected)
    finally:
        executor.close()


@pytest.mark.parametrize("beta", [0, 1])
def test_confidence_gate_scores_single_eligible_winner_once(tmp_path, beta):
    agent, scorer, predictions = _agent(tmp_path, beta=beta)
    agent.cfg.exec.simulator = agent.cfg.exec.simulator_only = True
    agent.acfg.real_validation_p_max_threshold = 0.95
    for _, code in CODES[1:]:
        predictions[code].predicted_metric = None
    _, code, _ = agent.generate_code({}, "improve")
    assert code == CODES[0][1]
    scorer.assert_not_called()
    assert agent._selected_confidence(code) == (0.99, "TimeoutError", 0.005)
    scorer.assert_called_once_with(code)


def test_confidence_gate_does_not_retry_failed_selection_scoring(tmp_path):
    agent, scorer, _ = _agent(tmp_path)
    agent.acfg.real_validation_p_max_threshold = 0.95
    scorer.side_effect = RuntimeError("scoring unavailable")
    _, code, _ = agent.generate_code({}, "improve")
    assert code == CODES[0][1]
    assert scorer.call_count == 4
    assert agent._selected_confidence(code) == (None, None, None)
    assert scorer.call_count == 4


def test_confidence_cache_is_reset_when_next_step_repeats_code(tmp_path):
    agent, scorer, predictions = _agent(tmp_path)
    _, code, _ = agent.generate_code({}, "improve")
    assert code == CODES[1][1]
    assert agent._selected_confidence(code) == (0.6, "TimeoutError", 0.3)
    assert scorer.call_count == 4
    agent.acfg.ucb.beta = 0
    predictions[code].predicted_metric = 0.91
    scorer.side_effect = lambda _: {"max_p": 0.98}
    _, next_code, _ = agent.generate_code({}, "improve")
    assert next_code == code
    assert agent._selected_confidence(next_code) == (0.98, None, None)
    assert scorer.call_count == 5


@pytest.mark.parametrize("threshold", [-0.1, 1.1, float("nan"), float("inf")])
def test_confidence_gate_rejects_invalid_threshold(tmp_path, threshold):
    agent, _, _ = _agent(tmp_path)
    agent.acfg.real_validation_p_max_threshold = threshold
    with pytest.raises(ValueError, match="real_validation_p_max_threshold"):
        agent._resolve_stage_policies()


@pytest.mark.parametrize("stages,expected", [
    ("improve", {"improve"}),
    ("all", set(STAGES)),
    ("draft, debug", {"draft", "debug"}),
    ("DEBUG,improve,all", set(STAGES)),
])
def test_confidence_gate_stage_flag_resolves(tmp_path, caplog, stages, expected):
    agent, _, _ = _agent(tmp_path)
    agent.cfg.exec.simulator = True
    agent.acfg.real_validation_p_max_threshold = 0.95
    agent.acfg.real_validation_p_max_stages = stages
    assert confidence_gate_stages(agent.acfg) == expected
    with caplog.at_level("INFO", logger="aide"):
        agent._resolve_stage_policies()
    listed = [s for s in STAGES if s in expected]
    assert f"[confidence-gate] p_max<0.95 triggers a real check on stages {listed}" in caplog.text


def test_confidence_gate_stage_flag_defaults_to_improve():
    assert confidence_gate_stages(SimpleNamespace()) == {"improve"}
    tuple_flag = SimpleNamespace(real_validation_p_max_stages=("draft",))
    assert confidence_gate_stages(tuple_flag) == {"draft"}


@pytest.mark.parametrize("stages", ["", "train", "draft,foo", "every"])
def test_confidence_gate_rejects_invalid_stages(tmp_path, stages):
    agent, _, _ = _agent(tmp_path)
    agent.cfg.exec.simulator = True
    agent.acfg.real_validation_p_max_threshold = 0.95
    agent.acfg.real_validation_p_max_stages = stages
    with pytest.raises(ValueError, match="real_validation_p_max_stages"):
        agent._resolve_stage_policies()
    # A disabled gate ignores the stage flag entirely.
    agent.acfg.real_validation_p_max_threshold = None
    agent._resolve_stage_policies()


@pytest.mark.parametrize("missing", ["simulator", "use_validate_on_best", "scorer"])
def test_confidence_gate_requires_real_validation_and_scoring(tmp_path, missing):
    agent, _, _ = _agent(tmp_path)
    agent.acfg.real_validation_p_max_threshold = 0.95
    agent.cfg.exec.simulator = missing != "simulator"
    agent.acfg.use_validate_on_best = missing != "use_validate_on_best"
    if missing == "scorer":
        agent._exc_scores = None
    with pytest.raises(ValueError, match="real_validation_p_max_threshold requires"):
        agent._resolve_stage_policies()


@pytest.mark.parametrize("field", ["success_threshold", "failure_threshold"])
@pytest.mark.parametrize("threshold", [-0.1, 1.1, float("nan"), float("inf")])
def test_outcome_gate_rejects_invalid_thresholds(tmp_path, field, threshold):
    agent, _, _ = _agent(tmp_path)
    agent.cfg.exec.simulator = True
    agent.acfg.execution_gate.enabled = True
    setattr(agent.acfg.execution_gate, field, threshold)
    with pytest.raises(ValueError, match=f"agent.execution_gate.{field}"):
        agent._resolve_stage_policies()


@pytest.mark.parametrize("missing", ["simulator", "scorer"])
def test_outcome_gate_requires_prediction_and_confidence_support(tmp_path, missing):
    agent, _, _ = _agent(tmp_path)
    agent.cfg.exec.simulator = missing != "simulator"
    agent.acfg.execution_gate.enabled = True
    agent.acfg.execution_gate.success_threshold = 0.9
    if missing == "scorer":
        agent._exc_scores = None
    with pytest.raises(ValueError, match="agent.execution_gate"):
        agent._resolve_stage_policies()


def test_outcome_gate_all_real_needs_no_scorer_and_ignores_legacy_stage_gate(tmp_path):
    agent, _, _ = _agent(tmp_path, beta=0)
    agent.cfg.exec.simulator = True
    agent.acfg.execution_gate.enabled = True
    agent.acfg.real_validation_p_max_threshold = 0.95
    agent.acfg.real_validation_p_max_stages = "invalid-but-ignored"
    agent._exc_scores = None
    agent._resolve_stage_policies()
    assert agent.acfg.execution_gate.success_threshold is None
    assert agent.acfg.execution_gate.failure_threshold is None


@pytest.mark.parametrize("oracle,simulator,sim_only,real_won,exc_type", [
    ("sim", False, True, False, None),
    ("sim", True, False, False, None),
    ("race", True, True, False, None),
    ("race", True, True, True, None),
    ("sim", True, True, True, None),
    ("sim", True, True, False, "SimulatorError"),
    ("sim", True, True, False, "SimulatorEmpty"),
    ("sim", True, True, False, "SimulatorCancelled"),
])
def test_prediction_reuse_preserves_required_execution(
    tmp_path, oracle, simulator, sim_only, real_won, exc_type
):
    agent, _, _ = _agent(tmp_path)
    agent.oracle = oracle
    agent.cfg.exec.simulator = simulator
    agent.cfg.exec.simulator_only = sim_only
    result = ExecutionResult(
        ["candidate output"], 0, exc_type, real_won=real_won,
        predicted_submission_exists="true", predicted_metric=0.9,
    )
    outcome = CandidateOutcome(result, True, "true", predicted=True)
    plan, code, committed = agent._commit([CODES[0]], [outcome], 0, tmp_path)
    assert (plan, code) == CODES[0]
    assert committed is None  # Agent.step must still invoke its configured executor.


def test_argmax_stage_uses_shared_eligibility_and_records_clean_probability(tmp_path):
    agent, scorer, predictions = _agent(tmp_path, policy="argmax", beta=0)
    predictions[CODES[2][1]].exc_type = "TypeError"
    predictions[CODES[3][1]].predicted_metric = None
    scorer.side_effect = lambda code: {"p_clean": 0.3 if code == CODES[0][1] else 0.99,
                                      "score": 0.0, "max_p": 1.0}
    assert agent.stage_policy["improve"] == "argmax"
    assert agent.generate_code({}, "improve")[1] == CODES[1][1]
    audit = json.loads((tmp_path / "bon/step_0/ucb.json").read_text())
    assert audit["score_rule"] == "p_clean" and audit["rule"] == "p_clean_argmax"
    assert [r["eligible"] for r in audit["candidates"]] == [True, True, False, False]
    assert sorted(call.args[0] for call in scorer.call_args_list) == sorted([CODES[0][1], CODES[1][1]])
    agent._judge_query.assert_not_called()


def test_argmax_stage_partial_scoring_failure_uses_metric_greedy(tmp_path):
    agent, scorer, _ = _agent(tmp_path, policy="argmax")
    scorer.side_effect = lambda code: None if code == CODES[0][1] else {"p_clean": 0.99}
    assert agent.generate_code({}, "improve")[1] == CODES[0][1]
    audit = json.loads((tmp_path / "bon/step_0/ucb.json").read_text())
    assert audit["fallback_reason"] == "missing_p_clean"


def test_ucb_does_not_enable_confidence_on_other_parallel_stages(tmp_path):
    agent, scorer, _ = _agent(tmp_path)
    agent.stage_policy["draft"] = "parallel"
    agent._judge_query = Mock(return_value={"best_candidate": 0, "reasoning": "first"})
    agent.generate_code({}, "draft")
    scorer.assert_not_called()
    agent._judge_query.assert_called_once()
    assert not (tmp_path / "bon/step_0/ucb.json").exists()


def test_ucb_stage_overrides_global_argmax_and_salvage_flags(tmp_path):
    agent, _, _ = _agent(tmp_path)
    agent.acfg.use_p_clean_argmax_pick = True
    agent.acfg.use_line_frac_salvage = True
    assert agent.generate_code({}, "improve")[1] == CODES[1][1]


def test_confidence_calls_run_concurrently(tmp_path):
    agent, scorer, _ = _agent(tmp_path)
    barrier = threading.Barrier(4)
    original = scorer.side_effect

    def score(code):
        barrier.wait(timeout=5)
        return original(code)

    scorer.side_effect = score
    assert agent.generate_code({}, "improve")[1] == CODES[1][1]


def test_scoring_exception_degrades_to_greedy_with_an_audit(tmp_path):
    agent, scorer, _ = _agent(tmp_path)
    scorer.side_effect = RuntimeError("world model unavailable")
    assert agent.generate_code({}, "improve")[1] == CODES[0][1]
    audit = json.loads((tmp_path / "bon/step_0/ucb.json").read_text())
    assert audit["fallback_reason"] == "missing_uncertainty"


@pytest.mark.parametrize("exc_type", ["TypeError", "TimeoutError"])
def test_bad_prediction_is_gated_before_confidence_scoring(tmp_path, exc_type):
    agent, scorer, predictions = _agent(tmp_path)
    predictions[CODES[1][1]].exc_type = exc_type
    predictions[CODES[1][1]].predicted_submission_exists = "false"
    assert agent.generate_code({}, "improve")[1] == CODES[0][1]
    assert CODES[1][1] not in [call.args[0] for call in scorer.call_args_list]


def test_buggy_debug_parent_uses_validated_journal_metric_without_reading_test(tmp_path):
    agent, _, _ = _agent(tmp_path, "debug")
    good = Node(code="good")
    good.is_buggy = False
    good.real_won = True
    good.submission_validation = "valid"
    good.submission_sha256 = "test-fixture-submission"
    good.metric = MetricValue(0.8, maximize=True)
    good.test_metric = float("nan")  # instrumentation never enters UCB context
    agent.journal.append(good)
    parent = Node(code="crash")
    parent.is_buggy = True
    parent.metric = WorstMetricValue()
    assert agent._ucb_context(parent) == (True, 0.8, "best_real")
    agent.generate_code({}, "debug", parent)
    audit = json.loads((tmp_path / "bon/step_0/ucb.json").read_text())
    assert audit["reference_metric"] == 0.8


def test_structured_metric_has_priority_and_review_can_reject_it(tmp_path):
    agent, _, predictions = _agent(tmp_path)
    outcomes = [
        CandidateOutcome(predictions[code], True, "true", True, review={"metric": 100 - i})
        for i, (_, code) in enumerate(CODES)
    ]
    assert agent._select_ucb(CODES, outcomes, tmp_path, "improve", None) == 1
    outcomes[1].review["metric_trustworthy"] = False
    assert agent._select_ucb(CODES, outcomes, tmp_path, "improve", None) == 0


def test_real_race_results_use_reviewed_real_metric_and_zero_prediction_sigma(tmp_path):
    agent, scorer, predictions = _agent(tmp_path)
    outcomes = []
    for i, (_, code) in enumerate(CODES):
        predictions[code].real_won = True
        outcomes.append(
            CandidateOutcome(predictions[code], True, "true", True, review={"metric": i / 10})
        )
    assert agent._select_ucb(CODES, outcomes, tmp_path, "improve", None) == 3
    scorer.assert_not_called()
    audit = json.loads((tmp_path / "ucb.json").read_text())
    assert all(r["sigma"] == 0 and r["metric_source"] == "real_review" for r in audit["candidates"])


def test_old_format_can_use_existing_review_metric(tmp_path):
    agent, _, predictions = _agent(tmp_path, beta=0)
    outcomes = []
    for i, (_, code) in enumerate(CODES):
        predictions[code].predicted_metric = None
        predictions[code].predicted_submission_exists = None
        outcomes.append(
            CandidateOutcome(predictions[code], True, "true", True, review={"metric": i / 10})
        )
    assert agent._select_ucb(CODES, outcomes, tmp_path, "improve", None) == 3


def test_explicit_missing_structured_metric_is_not_replaced_by_review(tmp_path):
    agent, _, predictions = _agent(tmp_path, beta=0)
    outcomes = [
        CandidateOutcome(predictions[code], True, "true", True, review={"metric": 100})
        for _, code in CODES
    ]
    predictions[CODES[1][1]].predicted_metric = None
    assert agent._select_ucb(CODES, outcomes, tmp_path, "improve", None) == 0
    audit = json.loads((tmp_path / "ucb.json").read_text())
    assert audit["candidates"][1]["eligible"] is False


def test_missing_dependencies_and_bad_settings_fail_at_startup(tmp_path):
    agent, _, _ = _agent(tmp_path)
    agent._exc_scores = None
    with pytest.raises(ValueError, match="exc-posterior scorer"):
        agent._resolve_stage_policies()
    agent.acfg.ucb.beta = 0
    assert agent._resolve_stage_policies()["improve"] == "ucb"
    agent._predict = None
    with pytest.raises(ValueError, match="exec.simulator"):
        agent._resolve_stage_policies()


def test_stage_overrides_survive_omegaconf_schema_merge(tmp_path):
    agent, _, _ = _agent(tmp_path)
    agent.acfg.ucb.improve.sigma = "entropy"
    agent.acfg.ucb.improve.beta = 0.2
    assert resolve_config(agent.acfg.ucb, "improve").beta == 0.2
    assert resolve_config(agent.acfg.ucb, "draft").beta == 1


def test_simulator_supplies_entropy_denominator():
    scores = _finalize_exc_posterior([1 / len(_EXC_SUPPORT)] * len(_EXC_SUPPORT))
    assert scores["support_size"] == len(_EXC_SUPPORT)
    assert scores["entropy"] == pytest.approx(math.log(scores["support_size"]))
