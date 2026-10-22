"""Leakage review must affect search/debugging without using private scores."""

import json
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from aide import agent as agent_mod
from aide import data_leakage, submission_review
from aide.agent import Agent
from aide.backend import compile_prompt_to_md
from aide.interpreter import ExecutionResult
from aide.journal import Journal, Node
from aide.utils.config import AgentConfig
from aide.utils.metric import MetricValue
from aide.utils.submission_validation import SubmissionValidation
from omegaconf import OmegaConf

CODE = "pred = y_val.copy()\nscore = roc_auc_score(y_val, pred)\n"
TRACE = "L1 copies validation targets into the predictions scored at L2."
FINDING = {
    "mechanism": "validation_targets_in_predictions",
    "data_flow": "Validation targets are copied into the scored predictions.",
    "why_not_permitted": "The targets become predictions, rather than selecting a checkpoint.",
    "source_lines": [1, 2],
    "repair": "Generate validation predictions with the training-fitted model.",
}
REASON = f"{FINDING['data_flow']} Repair: {FINDING['repair']}"
LEAKAGE = {
    "validation_protocol": "The reported score compares copied targets to themselves.",
    "counterevidence": "No model prediction replaces the copied targets.",
    "findings": [FINDING],
    "reason": FINDING["data_flow"],
    "verdict": "leak_detected",
}
CONFIRMATION = {
    "findings": [
        {
            "finding_index": 0,
            "source_lines": [1, 2],
            "analysis": FINDING["data_flow"],
            "disposition": "contamination_supported",
        }
    ]
}


@pytest.fixture
def setup_agent(tmp_path, monkeypatch):
    config = OmegaConf.load(Path(agent_mod.__file__).parent / "utils/config.yaml")
    config.agent = OmegaConf.merge(OmegaConf.structured(AgentConfig), config.agent)
    config.agent.eval_metric = "auc-roc"
    config.agent.eval_lower_is_better = False
    config.agent.search.num_drafts = 1
    config.agent.search.debug_prob = 1
    agent = object.__new__(Agent)
    agent.acfg = config.agent
    agent.cfg = SimpleNamespace(agent=config.agent, exec=config.exec, workspace_dir=tmp_path)
    agent.task_desc = "Train a classifier and produce test predictions."
    agent.journal = Journal()
    agent.deadline = time.time() + 600
    agent.current_step = 0
    agent.data_preview = "train.csv and test.csv"
    state = {
        "is_bug": False,
        "has_csv_submission": True,
        "summary": "Original execution review.",
        "metric": 0.9995,
        "lower_is_better": False,
        "metric_trustworthy": True,
    }
    primary = Mock(side_effect=lambda **kwargs: dict(state))
    specialist = Mock(side_effect=[TRACE, LEAKAGE, CONFIRMATION])
    monkeypatch.setattr(agent_mod, "query", primary)
    monkeypatch.setattr(data_leakage, "query", specialist)
    # These parse-result tests isolate leakage; the combined-review suite covers
    # the submission check's independent model calls and decisions.
    monkeypatch.setattr(
        data_leakage,
        "check_submission_consistency",
        Mock(
            return_value={
                "status": "checked",
                "decision": "allow",
                "has_leakage": False,
                "has_submission_mismatch": False,
                "reason": "No submission mismatch.",
                "judge": {"verdict": "no_mismatch_detected", "findings": []},
                "evidence": [],
                "attempts": 0,
                "usage": [],
                "errors": [],
            }
        ),
    )
    monkeypatch.delenv("AIDE_DEBUG_USE_ANALYSIS", raising=False)
    agent._validate_submission = Mock(
        return_value=SubmissionValidation("valid", "Valid CSV", "fixture-hash")
    )
    agent._process_real_submission = Mock(
        side_effect=lambda workspace, result, code: (
            code,
            agent._validate_submission(workspace / "submission/submission.csv"),
            {},
        )
    )
    agent._grade_test_submission = Mock(return_value=987654.321)
    submission = tmp_path / "submission/submission.csv"
    submission.parent.mkdir()
    submission.write_text("id,prediction\n1,0.9\n")
    return agent, state, primary, specialist


def parse(agent, state, *, real=True, node=None, exc_type=None):
    node = node if node is not None else Node(code=CODE, plan="Candidate plan")
    result = ExecutionResult(
        [f"Validation score: {state['metric']}\n"],
        1.0,
        exc_type,
        real_won=real,
        predicted_submission_exists=None if real else "true",
    )
    return agent.parse_exec_result(node, result)


@pytest.mark.parametrize("real", [True, False])
def test_large_exception_diagnosis_is_bounded_but_raw_evidence_survives(setup_agent, real):
    agent, _, _, _ = setup_agent
    code = "raise KeyError('missing column')"
    detail = {"args": ["MISSING_COLUMN_HEAD " + "array_value, " * 100000 + " MISSING_COLUMN_TAIL"]}
    result = ExecutionResult(
        ["KeyError: missing column\n"],
        1.0,
        "KeyError",
        exc_info=detail,
        real_won=real,
        predicted_submission_exists="false",
    )
    agent._process_real_submission = Mock(return_value=(code, None, {}))
    node = agent.parse_exec_result(Node(code=code), result)
    assert node.is_buggy
    assert node.exc_info is detail
    assert len(node.exc_info["args"][0]) > 1_000_000
    assert len(node.analysis) <= 5100
    (reason,) = node.failure_reasons["execution"]
    assert len(reason) <= 5100
    assert "MISSING_COLUMN_HEAD" in reason and "MISSING_COLUMN_TAIL" in reason
    debug = compile_prompt_to_md(agent._build_debug_prompt(node.code, node.term_out, node.analysis))
    assert len(debug) < 30000
    assert "MISSING_COLUMN_HEAD" in debug and "MISSING_COLUMN_TAIL" in debug


def test_combined_source_rejection_records_both_categories_with_one_diagnosis(
    setup_agent,
    monkeypatch,
):
    agent, state, _, _ = setup_agent
    reason = "Validation uses held-out targets and submission swaps the class columns."
    monkeypatch.setattr(
        submission_review,
        "check_validation",
        Mock(
            return_value={
                "status": "checked",
                "decision": "reject",
                "has_leakage": True,
                "has_submission_mismatch": True,
                "reason": reason,
            }
        ),
    )

    node = parse(agent, state)

    assert node.is_buggy and not node.metric_trustworthy and node.metric.value is None
    assert set(node.failure_reasons) == {"metric", "submission_pipeline"}
    assert node.failure_reasons["metric"] == node.failure_reasons["submission_pipeline"]
    assert reason in node.failure_reasons["metric"][0]
    assert node.analysis.count("SOURCE REVIEW REJECTED:") == 1
    assert node.analysis.count(reason) == 1
    prompt = agent._build_debug_prompt(
        node.code,
        node.term_out,
        node.analysis,
        node.failure_reasons,
    )
    assert prompt["Recorded failure categories"] == "metric, submission_pipeline"
    rendered = compile_prompt_to_md(prompt)
    assert rendered.count(reason) == 1
    assert "Repair the specific reported validation/metric defect" in rendered
    assert "Repair the diagnosed validation/submission prediction mismatch" in rendered
    assert agent._improve_prompt(node)["Previous solution"]["Results"].count(reason) == 1


@pytest.mark.parametrize("size", [100, 600000])
def test_prompt_boundaries_handle_preexisting_diagnoses(setup_agent, size):
    agent, _, _, _ = setup_agent
    analysis = "ROOT_CAUSE " + "x" * size + " REPAIR_HINT"
    node = Node(code="print('solution')", plan="Use the existing solution", analysis=analysis)
    debug = agent._build_debug_prompt(node.code, node.term_out, node.analysis)
    improve = agent._improve_prompt(node)
    for diagnosis in (
        debug["Diagnosis of the failure (from an expert reviewer)"],
        improve["Previous solution"]["Results"],
    ):
        assert len(diagnosis) <= 5100
        assert diagnosis.startswith("ROOT_CAUSE ") and diagnosis.endswith(" REPAIR_HINT")
        if size == 100:
            assert diagnosis == analysis
        else:
            assert "characters truncated" in diagnosis
    assert node.analysis == analysis


@pytest.mark.parametrize(
    "reasons,expected,unexpected",
    [
        (
            {"submission_pipeline": ["Class columns use the wrong training-label mapping."]},
            [
                "Repair the diagnosed validation/submission prediction mismatch",
                "Preserve the validation split",
            ],
            [
                "Repair the recorded execution failure",
                "Repair the specific reported validation/metric defect",
            ],
        ),
        (
            {"metric": ["The fitted predictor used the validation targets."]},
            ["Repair the specific reported validation/metric defect", "Ordinary early stopping"],
            [
                "Repair the diagnosed validation/submission prediction mismatch",
                "Repair the recorded execution failure",
            ],
        ),
        (
            {"execution": ["Observed execution failed: KeyError: missing column."]},
            ["Repair the recorded execution failure"],
            [
                "Repair the specific reported validation/metric defect",
                "Repair the diagnosed validation/submission prediction mismatch",
            ],
        ),
        (
            {"submission": ["CSV row IDs do not match the required rows."]},
            ["Repair the diagnosed CSV creation or format defect", "Preserve the validation split"],
            [
                "Repair the specific reported validation/metric defect",
                "Repair the recorded execution failure",
            ],
        ),
        (
            {
                "submission_pipeline": ["Class columns are swapped."],
                "metric": ["Validation targets entered fitting."],
            },
            [
                "Repair the diagnosed validation/submission prediction mismatch",
                "Repair the specific reported validation/metric defect",
            ],
            ["Preserve the validation split", "Repair the recorded execution failure"],
        ),
        (
            {"review": ["Real execution review failed before a diagnosis was available."]},
            ["Establish the cause from the recorded diagnosis"],
            [
                "Repair the specific reported validation/metric defect",
                "Repair the recorded execution failure",
            ],
        ),
    ],
)
def test_debug_step_routes_recorded_failure_categories(setup_agent, reasons, expected, unexpected):
    agent, _, _, _ = setup_agent
    node = Node(
        code="print('validation score: 0.82')",
        plan="Candidate plan",
        is_buggy=True,
        analysis="\n".join(reason for values in reasons.values() for reason in values),
        failure_reasons=reasons,
        _term_out=["Validation score: 0.82\nWrote submission.csv\n"],
    )
    agent.journal.append(node)
    agent.oracle = "sim"

    class StopBeforeGeneration(Exception):
        pass

    # Exercise Agent.step, including its real prompt wiring, without model calls.
    agent.generate_code = Mock(side_effect=StopBeforeGeneration)
    execute = Mock()
    with pytest.raises(StopBeforeGeneration):
        agent.step(execute)
    prompt, stage, parent = agent.generate_code.call_args.args
    assert stage == "debug" and parent is node
    rendered = compile_prompt_to_md(prompt)
    assert node.analysis in rendered
    assert prompt["Recorded failure categories"] == ", ".join(sorted(reasons))
    for text in expected:
        assert text in rendered
    for text in unexpected:
        assert text not in rendered
    assert "it was flagged because its validation score" not in rendered
    assert "code that ran BEFORE the failure point is verified" not in rendered
    execute.assert_not_called()


def test_legacy_debug_diagnosis_does_not_invent_metric_failure(setup_agent):
    agent, _, _, _ = setup_agent
    diagnosis = "Prediction columns use a different order from the trained class mapping."
    prompt = agent._build_debug_prompt("export_predictions()", "Wrote submission.csv", diagnosis)
    rendered = compile_prompt_to_md(prompt)
    assert diagnosis in rendered
    assert "Establish the cause from the recorded diagnosis" in rendered
    assert "Repair the specific reported validation/metric defect" not in rendered
    assert "it was flagged because its validation score" not in rendered


def test_leakage_rejects_metric_and_reaches_debug(setup_agent):
    agent, state, primary, specialist = setup_agent
    node = parse(agent, state)
    assert primary.call_count == 1 and specialist.call_count == 3
    assert node.is_buggy and node.metric.is_worst and node.metric_trustworthy is False
    assert node.leakage_check["status"] == "checked"
    assert node.leakage_check["metric"] == state["metric"]
    assert "Original execution review." in node.analysis and REASON in node.analysis
    baseline = Node(code="baseline", is_buggy=False, metric=MetricValue(0.8, maximize=True))
    agent.journal.append(baseline)
    agent.journal.append(node)
    assert agent.journal.get_best_node() is baseline
    assert agent.search_policy() is node
    debug = agent._build_debug_prompt(node.code, node.term_out, node.analysis)
    assert REASON in compile_prompt_to_md(debug)
    request = specialist.call_args_list[0].kwargs
    assert request["model"] == agent.acfg.feedback.model
    rendered = compile_prompt_to_md(request["user_message"])
    assert REASON not in rendered  # diagnosis is a result of this call
    assert "987654.321" not in rendered  # private test score never enters review
    assert "987654.321" not in json.dumps(node.leakage_check)


@pytest.mark.parametrize("verdict", ["no_leak_detected", "uncertain"])
def test_negative_or_uncertain_verdict_preserves_node(setup_agent, verdict):
    agent, state, _, specialist = setup_agent
    specialist.side_effect = [
        TRACE,
        {
            **LEAKAGE,
            "verdict": verdict,
            "findings": [],
            "reason": "No confirmed leak.",
        },
    ]
    node = parse(agent, state)
    assert node.is_buggy is False and node.metric.value == state["metric"]
    assert node.metric_trustworthy is True
    assert node.analysis == state["summary"]
    assert node.leakage_check["status"] == "checked"


@pytest.mark.parametrize(
    "lower,metric,checked",
    [
        (False, 0.999, True),
        (False, 0.998, False),
        (True, 1e-6, True),
        (True, 1e-5, False),
    ],
)
def test_trigger_uses_validation_metric_and_authoritative_direction(
    setup_agent, lower, metric, checked
):
    agent, state, _, specialist = setup_agent
    agent.acfg.eval_lower_is_better = lower
    agent.acfg.leakage_review_best = False  # Isolate the extreme-score trigger.
    state["lower_is_better"] = not lower  # deliberately wrong general-review direction
    state["metric"] = metric
    node = parse(agent, state)
    assert specialist.call_count == 3 * int(checked)
    if checked:
        assert node.leakage_check["maximize"] is not lower
    else:
        assert node.leakage_check is None and not node.is_buggy


@pytest.mark.parametrize(
    "skip", ["disabled", "execution_bug", "untrustworthy", "missing_csv", "invalid_csv"]
)
def test_existing_rejections_and_disabled_flag_skip_specialist(setup_agent, skip):
    agent, state, _, specialist = setup_agent
    if skip == "disabled":
        agent.acfg.check_data_leakage = False
    elif skip == "execution_bug":
        state["is_bug"] = True
    elif skip == "untrustworthy":
        state["metric_trustworthy"] = False
    elif skip == "missing_csv":
        (agent.cfg.workspace_dir / "submission/submission.csv").unlink()
    else:
        agent._validate_submission.return_value = SubmissionValidation("invalid", "Wrong IDs")
    node = parse(agent, state, exc_type="ValueError" if skip == "execution_bug" else None)
    specialist.assert_not_called()
    assert node.leakage_check is None


@pytest.mark.parametrize(
    "reply",
    [
        None,
        {},
        {"has_leakage": True},
        {"has_leakage": True, "confidence": "high", "leakage_reason": ""},
    ],
)
def test_malformed_checker_reply_is_unavailable_not_a_positive_finding(setup_agent, reply):
    agent, state, _, specialist = setup_agent
    specialist.side_effect = [TRACE, reply, reply]
    node = parse(agent, state)
    assert not node.is_buggy and node.metric.value == state["metric"]
    assert node.leakage_check["status"] == "unavailable"
    assert node.leakage_check["has_leakage"] is False


def test_backend_failure_is_recorded_without_aborting(setup_agent):
    agent, state, _, specialist = setup_agent
    specialist.side_effect = RuntimeError("feedback unavailable")
    node = parse(agent, state)
    assert not node.is_buggy
    assert node.leakage_check["status"] == "unavailable"
    assert "feedback unavailable" in node.leakage_check["reason"]


def test_predicted_evidence_is_labeled_and_reexecution_clears_stale_verdict(
    setup_agent,
):
    agent, state, _, specialist = setup_agent
    agent.acfg.leakage_review_best = False  # The real rerun below is not extreme.
    node = parse(agent, state, real=False)
    prompt = specialist.call_args_list[0].kwargs["user_message"]
    assert "Evidence source" not in prompt
    assert "Validation score:" not in json.dumps(prompt)
    assert node.leakage_check["source"] == "sim" and node.is_buggy
    agent._validate_submission.assert_not_called()
    agent._grade_test_submission.assert_not_called()
    state["metric"] = 0.9
    parse(agent, state, node=node)
    assert node.real_won and not node.is_buggy
    assert node.leakage_check is None and "DATA LEAKAGE" not in node.analysis
    assert specialist.call_count == 3


def test_leakage_rejection_applies_when_submission_format_is_unknown(setup_agent):
    agent, state, _, specialist = setup_agent
    agent._validate_submission.return_value = SubmissionValidation(
        "unknown", "Server unavailable", "fixture-hash"
    )
    node = parse(agent, state)
    assert node.is_buggy and node.submission_validation == "unknown"
    assert node.metric.is_worst and node.metric_trustworthy is False
    assert specialist.call_count == 3


def test_threshold_can_be_adjusted_for_task_scale(setup_agent):
    agent, state, _, specialist = setup_agent
    agent.acfg.leakage_max_threshold = 0.95
    state["metric"] = 0.96
    assert parse(agent, state).is_buggy
    assert specialist.call_count == 3
