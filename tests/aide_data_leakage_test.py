"""Model-only leakage decisions, attribution, and bounded failure behavior."""

import copy
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "agents/aide/src"))
from aide import agent as agent_module
from aide import data_leakage as checker
from aide import submission_review
from aide.agent import Agent
from aide.backend import compile_prompt_to_md
from aide.journal import Journal, Node
from aide.utils.config import AgentConfig, StageConfig
from aide.utils.metric import MetricValue

CODE = "pred = y_val.copy()\nvalidation_accuracy = accuracy_score(y_val, pred)\n"
TRACE = (
    "L1 copies the scored labels into predictions. L2 reports accuracy against those same labels."
)
UNSUPPORTED = dict(
    validation_protocol="A held-out score.",
    counterevidence="Fitting uses separate rows.",
    findings=[],
    reason="No contamination is established.",
    verdict="no_leak_detected",
)
SUPPORTED = dict(
    validation_protocol="Accuracy uses copied validation labels.",
    counterevidence="No guard replaces the copied labels with model predictions.",
    findings=[
        dict(
            mechanism="validation_targets_in_predictions",
            data_flow="L1 copies truth into predictions that L2 scores.",
            why_not_permitted="The targets become the predictions themselves; they do not select an epoch or checkpoint.",
            source_lines=[1, 2],
            repair="Predict using the training-fitted model and keep truth separate.",
        )
    ],
    reason="The score compares labels to themselves.",
    verdict="leak_detected",
)
CONFIRMED = dict(
    findings=[
        dict(
            finding_index=0,
            source_lines=[1, 2],
            analysis="The scored targets are copied into the predictions; no permitted operation covers this flow.",
            disposition="contamination_supported",
        )
    ]
)
PERMITTED = dict(
    findings=[
        dict(
            finding_index=0,
            source_lines=[1, 2],
            analysis="The alleged flow is exclusively permitted model selection.",
            disposition="permitted_model_selection",
        )
    ]
)


def review(**kwargs):
    return checker.check_data_leakage(
        task_desc="Predict using public training data.",
        code=CODE,
        model="gpt-coderbf16-fb-vllm",
        **kwargs,
    )


def test_trace_precedes_judgment_and_only_judge_decides(monkeypatch):
    query = Mock(side_effect=[TRACE, UNSUPPORTED])
    monkeypatch.setattr(checker, "query", query)
    result = review()
    assert result["decision"] == "allow" and not result["has_leakage"]
    assert result["audit"] == TRACE
    assert query.call_args_list[0].kwargs["func_spec"] is None
    assert query.call_args_list[1].kwargs["func_spec"] is None
    assert query.call_args_list[1].kwargs["response_format"]["type"] == "json_schema"
    assert TRACE in query.call_args_list[1].kwargs["user_message"]["Draft data-flow trace"]
    contract = query.call_args_list[1].kwargs["user_message"]["JSON response contract"]
    assert json.dumps(checker.DATA_LEAKAGE_CHECK_SPEC.json_schema) in contract
    assert "leak_detected means contamination IS present" in contract
    assert [u["phase"] for u in result["usage"]] == ["audit", "judge"]


@pytest.mark.parametrize(
    "response,decision,has_leakage",
    [
        (SUPPORTED, "reject", True),
        (UNSUPPORTED, "allow", False),
        ({**UNSUPPORTED, "verdict": "uncertain"}, "advisory", False),
    ],
)
def test_model_verdict_maps_to_default_decision(monkeypatch, response, decision, has_leakage):
    responses = [TRACE, response] + ([CONFIRMED] if has_leakage else [])
    monkeypatch.setattr(checker, "query", Mock(side_effect=responses))
    result = review()
    assert result["decision"] == decision and result["has_leakage"] is has_leakage
    assert result["status"] == "checked" and result["attempts"] == len(responses)


def test_independent_confirmation_can_overturn_a_false_rejection(monkeypatch):
    query = Mock(side_effect=[TRACE, SUPPORTED, PERMITTED])
    monkeypatch.setattr(checker, "query", query)
    result = review()
    assert result["decision"] == "allow" and not result["has_leakage"]
    assert result["initial_judge"] == SUPPORTED
    assert result["confirmation"] == PERMITTED and result["evidence"] == []
    confirmation_prompt = query.call_args_list[2].kwargs["user_message"]
    assert "Draft data-flow trace" not in confirmation_prompt
    assert TRACE not in json.dumps(confirmation_prompt)
    assert SUPPORTED["reason"] not in json.dumps(confirmation_prompt)
    claims = json.loads(confirmation_prompt["Untrusted data-flow claims to verify"])
    assert claims == [
        {
            "claim": SUPPORTED["findings"][0]["data_flow"],
            "source": [
                {"line": 1, "code": CODE.splitlines()[0]},
                {"line": 2, "code": CODE.splitlines()[1]},
            ],
        }
    ]
    # Exercise the actual backend prompt renderer, which accepts text lists
    # but does not support lists of dictionaries as section contents.
    rendered = compile_prompt_to_md(confirmation_prompt)
    assert CODE.splitlines()[0] in rendered
    assert [u["phase"] for u in result["usage"]] == ["audit", "judge", "confirm"]


def test_uncertain_confirmation_does_not_hard_reject(monkeypatch):
    uncertain = copy.deepcopy(CONFIRMED)
    uncertain["findings"][0]["disposition"] = "uncertain"
    monkeypatch.setattr(checker, "query", Mock(side_effect=[TRACE, SUPPORTED, uncertain]))
    result = review()
    assert result["status"] == "checked" and result["decision"] == "advisory"
    assert not result["has_leakage"] and result["initial_judge"] == SUPPORTED


def test_confirmation_keeps_only_supported_findings(monkeypatch):
    initial = copy.deepcopy(SUPPORTED)
    permitted = {**initial["findings"][0], "data_flow": "A permitted checkpoint-selection claim."}
    initial["findings"] = [permitted, initial["findings"][0]]
    confirmation = {
        "findings": [
            PERMITTED["findings"][0],
            {**CONFIRMED["findings"][0], "finding_index": 1},
        ]
    }
    monkeypatch.setattr(checker, "query", Mock(side_effect=[TRACE, initial, confirmation]))
    result = review()
    assert result["decision"] == "reject"
    assert result["judge"]["findings"] == SUPPORTED["findings"]
    assert "checkpoint-selection claim" not in result["reason"]
    assert result["initial_judge"] == initial


@pytest.mark.parametrize(
    "bad",
    [
        {"findings": []},
        {"findings": [CONFIRMED["findings"][0], CONFIRMED["findings"][0]]},
        {"findings": [{**CONFIRMED["findings"][0], "source_lines": [9999]}]},
        {"findings": [{**CONFIRMED["findings"][0], "disposition": "generally_allowed"}]},
    ],
)
def test_incomplete_or_unsupported_confirmation_cannot_reject(monkeypatch, bad):
    monkeypatch.setattr(checker, "query", Mock(side_effect=[TRACE, SUPPORTED, bad, bad]))
    result = review()
    assert result["status"] == "unavailable" and result["decision"] == "advisory"
    assert result["attempts"] == 4 and result["evidence"] == []
    assert all(error["phase"] == "confirm" for error in result["errors"])


def test_confirmation_uses_remaining_budget_and_failure_stays_advisory(monkeypatch):
    clock = [10.0]
    monkeypatch.setattr(checker.time, "monotonic", lambda: clock[0])
    responses = iter([TRACE, SUPPORTED])

    def query(**kwargs):
        clock[0] += 3
        if kwargs["retry_max_time"] == 2:
            raise TimeoutError("confirmation exhausted remaining allowance")
        return next(responses)

    call = Mock(side_effect=query)
    monkeypatch.setattr(checker, "query", call)
    result = review(timeout_s=8)
    assert [c.kwargs["retry_max_time"] for c in call.call_args_list] == [8, 5, 2]
    assert result["decision"] == "advisory" and result["status"] == "unavailable"
    assert not result["has_leakage"] and result["evidence"] == []
    assert result["usage"][-1]["phase"] == "confirm"


def test_no_deterministic_gate_survives(monkeypatch):
    # Even obvious source text is not rejected without a supported model verdict.
    monkeypatch.setattr(checker, "query", Mock(side_effect=[TRACE, UNSUPPORTED]))
    assert review()["decision"] == "allow"
    assert not hasattr(checker, "source_verdict")
    assert not (Path(checker.__file__).parent / "utils/leakage_evidence.py").exists()


def test_scores_outputs_and_simulation_never_enter_any_review_prompt(monkeypatch):
    query = Mock(side_effect=[TRACE, SUPPORTED, CONFIRMED, TRACE, SUPPORTED, CONFIRMED])
    monkeypatch.setattr(checker, "query", query)
    review(metric=0.82, term_out="SENTINEL_PUBLIC_LOG", predicted=False)
    review(metric=1.0, term_out="SENTINEL_PRIVATE_VALUE", predicted=True)
    calls = query.call_args_list
    assert len(calls) == 6
    for a, b in zip(calls[:3], calls[3:]):
        assert a.kwargs["user_message"] == b.kwargs["user_message"]
        text = json.dumps(b.kwargs, default=str)
        assert "SENTINEL_PRIVATE_VALUE" not in text and "SENTINEL_PUBLIC_LOG" not in text


def test_out_of_range_citation_is_retried_not_used_as_rejection(monkeypatch):
    wrong = copy.deepcopy(SUPPORTED)
    wrong["findings"][0]["source_lines"] = [9999]
    query = Mock(side_effect=[TRACE, wrong, UNSUPPORTED])
    monkeypatch.setattr(checker, "query", query)
    result = review()
    assert result["decision"] == "allow" and result["attempts"] == 3
    assert result["errors"][0]["phase"] == "judge"
    assert "Citation" in result["errors"][0]["reason"]


def test_model_selection_is_not_a_prohibited_mechanism(monkeypatch):
    wrong = copy.deepcopy(SUPPORTED)
    wrong["findings"][0]["mechanism"] = "early_stopping_or_checkpoint_selection"
    query = Mock(side_effect=[TRACE, wrong, UNSUPPORTED])
    monkeypatch.setattr(checker, "query", query)
    result = review()
    assert result["status"] == "checked" and result["decision"] == "allow"
    assert result["attempts"] == 3 and result["errors"][0]["phase"] == "judge"


def test_rejection_without_explaining_policy_exceptions_stays_advisory(monkeypatch):
    incomplete = copy.deepcopy(SUPPORTED)
    del incomplete["findings"][0]["why_not_permitted"]
    monkeypatch.setattr(checker, "query", Mock(side_effect=[TRACE, incomplete, incomplete]))
    result = review()
    assert result["status"] == "unavailable" and result["decision"] == "advisory"
    assert not result["has_leakage"] and result["attempts"] == 3


@pytest.mark.parametrize(
    "bad",
    [
        {},
        dict(UNSUPPORTED, verdict="leak_detected"),
        dict(SUPPORTED, verdict="no_leak_detected"),
        dict(SUPPORTED, verdict="supported"),
    ],
)
def test_malformed_judgment_stays_unavailable(monkeypatch, bad):
    query = Mock(side_effect=[TRACE, bad, bad])
    monkeypatch.setattr(checker, "query", query)
    result = review()
    assert result["status"] == "unavailable" and result["decision"] == "advisory"
    assert not result["has_leakage"] and result["attempts"] == 3
    assert all(c.kwargs["tool_call_max_attempts"] == 1 for c in query.call_args_list)


def test_truncated_trace_is_not_silently_accepted(monkeypatch):
    def query(**kw):
        kw["info_sink"]["finish_reason"] = "length"
        return TRACE

    call = Mock(side_effect=query)
    monkeypatch.setattr(checker, "query", call)
    result = review()
    assert result["status"] == "unavailable" and result["attempts"] == 2
    assert "audit" not in result
    assert all(c.kwargs["func_spec"] is None for c in call.call_args_list)


def test_empty_trace_retries_once(monkeypatch):
    query = Mock(side_effect=["", TRACE, UNSUPPORTED])
    monkeypatch.setattr(checker, "query", query)
    assert review()["decision"] == "allow"
    assert query.call_count == 3


def test_all_phases_and_retries_share_budget(monkeypatch):
    clock = [10.0]
    monkeypatch.setattr(checker.time, "monotonic", lambda: clock[0])
    responses = iter([TRACE, {}, UNSUPPORTED])

    def query(**kw):
        clock[0] += 3
        return next(responses)

    call = Mock(side_effect=query)
    monkeypatch.setattr(checker, "query", call)
    assert review(timeout_s=10)["decision"] == "allow"
    assert [c.kwargs["retry_max_time"] for c in call.call_args_list] == [10, 7, 4]


def test_deadline_expiring_between_phases_prevents_second_request(monkeypatch):
    clock = [10.0]
    monkeypatch.setattr(checker.time, "monotonic", lambda: clock[0])

    def query(**kw):
        clock[0] += 5
        return TRACE

    call = Mock(side_effect=query)
    monkeypatch.setattr(checker, "query", call)
    result = review(timeout_s=5)
    assert result["status"] == "unavailable" and call.call_count == 1


def test_zero_budget_does_not_call_model(monkeypatch):
    query = Mock()
    monkeypatch.setattr(checker, "query", query)
    result = review(timeout_s=0)
    query.assert_not_called()
    assert result["status"] == "unavailable" and result["attempts"] == 0


def test_backend_failure_preserves_attempt_metadata(monkeypatch):
    monkeypatch.setattr(checker, "query", Mock(side_effect=TimeoutError("unreachable")))
    result = review()
    assert result["status"] == "unavailable" and result["attempts"] == 1
    assert result["usage"] == [{"phase": "audit"}]


@pytest.fixture
def agent(monkeypatch):
    # Legacy agent tests isolate the unchanged leakage decision path.
    monkeypatch.setattr(
        checker,
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
    obj = object.__new__(Agent)
    obj.acfg = SimpleNamespace(
        check_data_leakage=True,
        leakage_max_threshold=0.999,
        leakage_min_threshold=1e-6,
        leakage_review_best=True,
        leakage_timeout_s=120,
        leakage_max_tokens=3072,
        feedback=SimpleNamespace(model="gpt-coderbf16-fb-vllm", temp=0.5),
        eval_metric="accuracy",
        convert_system_to_user=False,
    )
    obj.task_desc = "Use public training data."
    obj.journal = Journal()
    obj.deadline = time.time() + 600
    return obj


def node(*, score=0.82, real=True, maximize=True):
    return Node(
        code=CODE,
        metric=MetricValue(score, maximize=maximize),
        metric_trustworthy=True,
        is_buggy=False,
        real_won=real,
        submission_validation="valid" if real else None,
        analysis="original",
    )


def test_non_extreme_best_real_candidate_is_reviewed(agent, monkeypatch):
    call = Mock(return_value=dict(status="checked", decision="allow"))
    monkeypatch.setattr(submission_review, "check_validation", call)
    n = node()
    agent._check_data_leakage(n)
    call.assert_called_once()
    assert n.leakage_check["triggers"] == ["best_real_candidate"]


@pytest.mark.parametrize("real", [True, False])
def test_noncontender_has_no_hidden_source_check(agent, monkeypatch, real):
    agent.journal.nodes.append(node(score=0.95))
    call = Mock()
    monkeypatch.setattr(submission_review, "check_validation", call)
    n = node(real=real)
    agent._check_data_leakage(n)
    call.assert_not_called()
    assert n.leakage_check is None and not n.is_buggy


@pytest.mark.parametrize("maximize,score", [(True, 1.0), (False, 1e-7)])
def test_extreme_simulated_score_is_reviewed(agent, monkeypatch, maximize, score):
    call = Mock(return_value=dict(status="checked", decision="allow"))
    monkeypatch.setattr(submission_review, "check_validation", call)
    n = node(score=score, real=False, maximize=maximize)
    agent._check_data_leakage(n)
    assert n.leakage_check["triggers"] == ["extreme_score"]
    call.assert_called_once()
    assert call.call_args.kwargs["review_submission"] is False


@pytest.mark.parametrize("config_source", ["legacy", "yaml", "schema", "explicit"])
def test_positive_model_result_marks_buggy_and_sends_debug_feedback(
    agent, monkeypatch, config_source
):
    if config_source == "yaml":
        agent.acfg = OmegaConf.load(Path(agent_module.__file__).parent / "utils/config.yaml").agent
    elif config_source == "schema":
        agent.acfg = OmegaConf.structured(AgentConfig)
        agent.acfg.feedback = StageConfig(model="gpt-coderbf16-fb-vllm", temp=0.5)
        agent.acfg.convert_system_to_user = False
    elif config_source == "explicit":
        agent.acfg.leakage_reject_on_model = True
    monkeypatch.setattr(checker, "query", Mock(side_effect=[TRACE, SUPPORTED, CONFIRMED]))
    n = node()
    agent._check_data_leakage(n)
    assert n.leakage_check["has_leakage"] and n.leakage_check["decision"] == "reject"
    assert n.leakage_check["reject_on_detection"] is True
    assert n.is_buggy and not n.metric_trustworthy and n.metric.value is None
    assert "Predict using the training-fitted model" in n.failure_reasons["metric"][0]
    assert "source evidence" not in n.analysis


@pytest.mark.parametrize("response", [dict(UNSUPPORTED, verdict="uncertain"), UNSUPPORTED])
def test_advisory_or_negative_does_not_alter_node(agent, monkeypatch, response):
    monkeypatch.setattr(checker, "query", Mock(side_effect=[TRACE, response]))
    n = node()
    agent._check_data_leakage(n)
    assert (
        not n.is_buggy
        and n.metric_trustworthy
        and n.analysis == "original"
        and n.failure_reasons == {}
    )


def test_overturned_finding_preserves_the_working_candidate(agent, monkeypatch):
    monkeypatch.setattr(checker, "query", Mock(side_effect=[TRACE, SUPPORTED, PERMITTED]))
    n = node()
    agent._check_data_leakage(n)
    assert n.leakage_check["decision"] == "allow"
    assert n.leakage_check["initial_judge"]["verdict"] == "leak_detected"
    assert not n.is_buggy and n.metric_trustworthy and n.metric.value == 0.82
    assert n.analysis == "original" and n.failure_reasons == {}


def test_failure_and_disable_leave_task_alive(agent, monkeypatch):
    call = Mock(side_effect=RuntimeError("judge failed"))
    monkeypatch.setattr(submission_review, "check_validation", call)
    n = node()
    agent._check_data_leakage(n)
    assert n.leakage_check["status"] == "unavailable" and not n.is_buggy
    agent.acfg.check_data_leakage = False
    disabled = node()
    agent._check_data_leakage(disabled)
    assert disabled.leakage_check is None and call.call_count == 1


def test_original_source_text_is_attached_without_model_quoting(monkeypatch):
    raw = copy.deepcopy(SUPPORTED)
    monkeypatch.setattr(
        checker, "query", Mock(side_effect=[TRACE, json.dumps(raw), json.dumps(CONFIRMED)])
    )
    result = review()
    assert result["evidence"][0]["evidence"] == [
        {"line": 1, "code": CODE.splitlines()[0]},
        {"line": 2, "code": CODE.splitlines()[1]},
    ]
    assert raw == SUPPORTED  # no mutation of model input/response fixtures


def test_explicit_advisory_override_preserves_node(agent, monkeypatch):
    agent.acfg.leakage_reject_on_model = False
    monkeypatch.setattr(checker, "query", Mock(side_effect=[TRACE, SUPPORTED, CONFIRMED]))
    n = node()
    agent._check_data_leakage(n)
    assert n.leakage_check["has_leakage"] is True
    assert n.leakage_check["decision"] == "advisory"
    assert not n.leakage_check["reject_on_detection"]
    assert not n.is_buggy and n.metric.value == 0.82 and n.metric_trustworthy
    assert n.failure_reasons == {} and n.analysis == "original"


MISMATCH_CODE = "classes = ['dog', 'cat']\npred = model.predict_proba(test)\nsubmission = pd.DataFrame(pred, columns=['cat', 'dog'])\n"
PIPELINE_ALLOW = {
    **UNSUPPORTED,
    "submission_protocol": "No submission mismatch established.",
    "verdict": "no_mismatch_detected",
}
MISMATCH = {
    **PIPELINE_ALLOW,
    "submission_protocol": "Probability columns are exported with a different class order.",
    "verdict": "mismatch_detected",
    "findings": [
        dict(
            mechanism="prediction_target_misalignment",
            data_flow="The training class order at L1 differs from export column order at L3.",
            why_not_permitted="No class permutation aligns the probabilities before export.",
            source_lines=[1, 3],
            repair="Reindex probability columns using the trained class names.",
        )
    ],
    "reason": "The validation estimate is clean but the submission labels are incorrect.",
}
MISMATCH_CONFIRMED = {
    "findings": [
        dict(
            finding_index=0,
            source_lines=[1, 3],
            analysis="The CSV uses reversed class labels without rearranging probabilities.",
            disposition="submission_mismatch_supported",
        )
    ]
}


def pipeline_review(**kwargs):
    return checker.check_submission_consistency(
        task_desc="Predict each test row with the trained class mapping.",
        code=MISMATCH_CODE,
        model="gpt-coderbf16-fb-vllm",
        **kwargs,
    )


def test_submission_confirmation_does_not_inherit_the_draft(monkeypatch):
    query = Mock(side_effect=[TRACE, MISMATCH, MISMATCH_CONFIRMED])
    monkeypatch.setattr(checker, "query", query)
    pipeline_review()
    assert TRACE in query.call_args_list[1].kwargs["user_message"]["Draft data-flow trace"]
    assert "Draft data-flow trace" not in query.call_args_list[2].kwargs["user_message"]


def combined_review(**kwargs):
    return checker.check_validation(
        task_desc="Predict each test row with the trained class mapping.",
        code=MISMATCH_CODE,
        model="gpt-coderbf16-fb-vllm",
        **kwargs,
    )


def test_submission_bug_rejects_by_default_without_claiming_label_leakage(agent, monkeypatch):
    monkeypatch.setattr(
        checker,
        "check_submission_consistency",
        lambda **kw: checker._review_source(submission=True, **kw),
    )
    monkeypatch.setattr(
        checker,
        "query",
        Mock(side_effect=[TRACE, UNSUPPORTED, TRACE, MISMATCH, MISMATCH_CONFIRMED]),
    )
    n = node()
    n.code = MISMATCH_CODE
    agent._check_data_leakage(n)
    assert n.leakage_check["decision"] == "reject"
    assert n.leakage_check["has_submission_mismatch"]
    assert not n.leakage_check["has_leakage"]
    assert n.is_buggy and n.metric.value is None and not n.metric_trustworthy
    assert set(n.failure_reasons) == {"submission_pipeline"}
    assert MISMATCH["findings"][0]["repair"] in n.analysis
    assert "validation contamination" not in n.analysis
    assert AgentConfig.submission_reject_on_model is True
    assert (
        OmegaConf.load(
            Path(agent_module.__file__).parent / "utils/config.yaml"
        ).agent.submission_reject_on_model
        is True
    )


def test_submission_review_explicit_advisory_override_preserves_feedback(agent, monkeypatch):
    agent.acfg.submission_reject_on_model = False
    monkeypatch.setattr(
        checker,
        "check_submission_consistency",
        lambda **kw: checker._review_source(submission=True, **kw),
    )
    monkeypatch.setattr(
        checker,
        "query",
        Mock(side_effect=[TRACE, UNSUPPORTED, TRACE, MISMATCH, MISMATCH_CONFIRMED]),
    )
    n = node()
    n.code = MISMATCH_CODE
    agent._check_data_leakage(n)
    assert n.leakage_check["decision"] == "advisory"
    assert n.leakage_check["has_submission_mismatch"] and not n.leakage_check["has_leakage"]
    assert not n.is_buggy and n.metric.value == 0.82 and n.metric_trustworthy
    assert n.failure_reasons == {} and n.submission_validation == "valid"
    assert "SUBMISSION PREDICTION REVIEW (ADVISORY)" in n.analysis
    assert MISMATCH["findings"][0]["repair"] in n.analysis


def test_global_advisory_override_also_controls_submission_review(monkeypatch):
    monkeypatch.setattr(
        checker,
        "query",
        Mock(side_effect=[TRACE, UNSUPPORTED, TRACE, MISMATCH, MISMATCH_CONFIRMED]),
    )
    result = combined_review(reject_on_detection=False, submission_reject_on_detection=True)
    assert result["decision"] == "advisory"
    assert result["has_submission_mismatch"]
    assert result["submission_review"]["reject_on_detection"] is False


def test_submission_temperature_is_independent_from_leakage_feedback(monkeypatch):
    query = Mock(side_effect=[TRACE, UNSUPPORTED, TRACE, PIPELINE_ALLOW])
    monkeypatch.setattr(checker, "query", query)
    result = combined_review(temperature=0.5)
    assert result["decision"] == "allow"
    leakage_calls = query.call_args_list[:2]
    submission_calls = query.call_args_list[2:]
    assert all(c.kwargs["temperature"] == 0.5 for c in leakage_calls)
    assert all("temperature_override" not in c.kwargs for c in leakage_calls)
    assert all(c.kwargs["temperature"] == 0.0 for c in submission_calls)
    assert all(c.kwargs["temperature_override"] == 0.0 for c in submission_calls)
    assert AgentConfig.submission_review_temperature == 0.0
    cfg = OmegaConf.load(Path(agent_module.__file__).parent / "utils/config.yaml")
    assert cfg.agent.submission_review_temperature == 0.0


@pytest.mark.parametrize(
    "disposition,decision",
    [
        ("source_refutes_claim", "allow"),
        ("permitted_inference_difference", "allow"),
        ("uncertain", "advisory"),
    ],
)
def test_unconfirmed_submission_claim_preserves_candidate(monkeypatch, disposition, decision):
    confirmation = copy.deepcopy(MISMATCH_CONFIRMED)
    confirmation["findings"][0]["disposition"] = disposition
    monkeypatch.setattr(checker, "query", Mock(side_effect=[TRACE, MISMATCH, confirmation]))
    result = pipeline_review()
    assert result["decision"] == decision
    assert not result["has_leakage"] and not result["has_submission_mismatch"]
    assert result["evidence"] == []


def test_confirmation_cannot_change_a_mismatch_into_contamination(monkeypatch):
    wrong = copy.deepcopy(MISMATCH_CONFIRMED)
    wrong["findings"][0]["disposition"] = "contamination_supported"
    monkeypatch.setattr(checker, "query", Mock(side_effect=[TRACE, MISMATCH, wrong, wrong]))
    result = pipeline_review()
    assert result["decision"] == "advisory" and result["status"] == "unavailable"
    assert not result["has_leakage"] and not result["has_submission_mismatch"]


@pytest.mark.parametrize(
    "wrong",
    [
        {**MISMATCH, "verdict": "no_mismatch_detected"},
        {**MISMATCH, "verdict": "leak_detected"},
        {**MISMATCH, "findings": []},
        {**MISMATCH, "findings": SUPPORTED["findings"]},
    ],
)
def test_malformed_or_wrong_category_submission_verdict_cannot_reject(monkeypatch, wrong):
    monkeypatch.setattr(checker, "query", Mock(side_effect=[TRACE, wrong, wrong]))
    result = pipeline_review()
    assert result["decision"] == "advisory" and result["status"] == "unavailable"


def test_positive_leakage_review_skips_unnecessary_submission_review(monkeypatch):
    monkeypatch.setattr(checker, "query", Mock(side_effect=[TRACE, SUPPORTED, CONFIRMED]))
    submission = Mock()
    monkeypatch.setattr(checker, "check_submission_consistency", submission)
    result = combined_review()
    assert result["decision"] == "reject" and result["has_leakage"]
    assert result["submission_review"] is None
    submission.assert_not_called()


def test_caller_can_limit_extreme_simulated_checks_to_leakage(monkeypatch):
    monkeypatch.setattr(checker, "query", Mock(side_effect=[TRACE, UNSUPPORTED]))
    submission = Mock()
    monkeypatch.setattr(checker, "check_submission_consistency", submission)
    result = combined_review(review_submission=False)
    assert result["decision"] == "allow" and result["submission_review"] is None
    submission.assert_not_called()


@pytest.mark.parametrize("first", [UNSUPPORTED, {**UNSUPPORTED, "verdict": "uncertain"}])
def test_submission_rejection_survives_clean_or_uncertain_leakage(monkeypatch, first):
    query = Mock(side_effect=[TRACE, first, TRACE, MISMATCH, MISMATCH_CONFIRMED])
    monkeypatch.setattr(checker, "query", query)
    result = combined_review()
    assert result["decision"] == "reject" and result["has_submission_mismatch"]
    assert not result["has_leakage"]
    assert result["judge"]["verdict"] == first["verdict"]
    assert result["judge"]["submission_verdict"] == "mismatch_detected"
    # The submission trace starts independently, with neither leakage trace nor verdict.
    prompt = query.call_args_list[2].kwargs["user_message"]
    assert "Draft data-flow trace" not in prompt
    assert TRACE not in json.dumps(prompt) and first["reason"] not in json.dumps(prompt)


def test_both_reviews_and_confirmation_share_one_total_deadline(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(checker.time, "monotonic", lambda: clock[0])
    responses = iter([TRACE, UNSUPPORTED, TRACE, MISMATCH, MISMATCH_CONFIRMED])

    def query(**kwargs):
        clock[0] += 2
        return next(responses)

    call = Mock(side_effect=query)
    monkeypatch.setattr(checker, "query", call)
    result = combined_review(timeout_s=12)
    assert result["decision"] == "reject"
    assert [c.kwargs["retry_max_time"] for c in call.call_args_list] == [12, 10, 8, 6, 4]
    assert result["elapsed_s"] == 10


def test_exhausted_shared_budget_does_not_start_submission_request(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(checker.time, "monotonic", lambda: clock[0])
    responses = iter([TRACE, UNSUPPORTED])

    def query(**kwargs):
        clock[0] += 2
        return next(responses)

    call = Mock(side_effect=query)
    monkeypatch.setattr(checker, "query", call)
    result = combined_review(timeout_s=4)
    assert call.call_count == 2
    assert result["decision"] == "advisory" and result["status"] == "unavailable"
    assert result["submission_review"]["attempts"] == 0


def test_submission_advisory_override_and_failed_confirmation_do_not_reject(monkeypatch):
    monkeypatch.setattr(checker, "query", Mock(side_effect=[TRACE, MISMATCH, MISMATCH_CONFIRMED]))
    result = pipeline_review(reject_on_detection=False)
    assert result["decision"] == "advisory" and result["has_submission_mismatch"]
    monkeypatch.setattr(
        checker, "query", Mock(side_effect=[TRACE, MISMATCH, TimeoutError("budget")])
    )
    result = pipeline_review()
    assert result["decision"] == "advisory" and result["status"] == "unavailable"
    assert not result["has_submission_mismatch"]


@pytest.mark.parametrize("maximize", [True, False])
def test_tied_real_candidate_receives_review_before_replacing_incumbent(
    agent, monkeypatch, maximize
):
    agent.journal.nodes.append(node(score=0.82, maximize=maximize))
    call = Mock(return_value=dict(status="checked", decision="allow"))
    monkeypatch.setattr(submission_review, "check_validation", call)
    n = node(score=0.82, maximize=maximize)
    agent._check_data_leakage(n)
    call.assert_called_once()
    assert n.leakage_check["triggers"] == ["best_real_candidate"]
    assert call.call_args.kwargs["review_submission"] is True


def test_tie_trigger_obeys_review_best_switch(agent, monkeypatch):
    agent.acfg.leakage_review_best = False
    agent.journal.nodes.append(node())
    call = Mock()
    monkeypatch.setattr(submission_review, "check_validation", call)
    agent._check_data_leakage(node())
    call.assert_not_called()
