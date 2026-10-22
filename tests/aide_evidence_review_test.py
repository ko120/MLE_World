"""Integration regressions for evidence-guided review; no model/network calls."""

import copy
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace
from unittest.mock import Mock
import zipfile

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "agents/aide/src"))
from aide import data_leakage as checker
from aide import validation_evidence as evidence
from aide import validation_review
from aide.agent import Agent
from aide.journal import Journal, Node
from aide.utils.metric import MetricValue
from aide.utils import validation_schema

MODEL = "gpt-coderbf16-fb-vllm"
CODE = """def features(df):
    if "flair" not in df.columns:
        df["flair"] = 0
    return df[["flair"]]
X_val = features(val)
predictions = model.predict(X_val)
"""
SCHEMA = {
    "files": [
        {"file": "train.json", "rows": 2, "key_presence_counts": {"x": 2, "flair": 2, "y": 2}},
        {"file": "test.json", "rows": 1, "key_presence_counts": {"x": 1}},
    ],
    "designated_raw_target_fields": ["y"],
}
PROPOSAL = {
    "field": "flair",
    "source_lines": [2, 3, 4, 6],
    "validation_path": "Observed flair reaches X_val and predict.",
    "test_path": "Test uses default zero.",
    "reachability_guard_checked": "Validation column exists.",
    "validation_nondefault_value_reaches_prediction": True,
    "test_nondefault_value_reaches_prediction": False,
    "validation_and_test_both_use_same_constant": False,
}
VERIFIED = {
    "raw_field": "flair",
    "raw_read_lines": [4],
    "selected_feature_lines": [4, 5],
    "prediction_call_lines": [6],
    "actual_validation_call_and_branch": "features(val), missing-column branch bypassed",
    "field_value_at_prediction_input": "Observed flair in X_val.",
    "source_lines": [4, 5, 6],
    "role": "observed_predictor_input",
}
ALLOW_LEAK = {
    "validation_protocol": "Disjoint supervised fit.",
    "counterevidence": "No target use found.",
    "findings": [],
    "reason": "No contamination established.",
    "verdict": "no_leak_detected",
}
ALLOW_EXPORT = {"trace": "Values preserved.", "findings": [], "decision": "allow"}


def install_query(monkeypatch, responses):
    def answer(**kwargs):
        fmt = kwargs.get("response_format")
        name = fmt["json_schema"]["name"] if fmt else "audit"
        value = responses.get(name)
        if value is None:
            raise AssertionError("Unexpected review phase: " + name)
        if callable(value):
            return value(kwargs)
        if isinstance(value, Exception):
            raise value
        return copy.deepcopy(value)

    query = Mock(side_effect=answer)
    monkeypatch.setattr(checker, "query", query)
    return query


def review(**kwargs):
    return checker.check_submission_consistency(
        code=kwargs.pop("code", CODE),
        public_schema=kwargs.pop("public_schema", SCHEMA),
        model=MODEL,
        task_desc="Use public data.",
        **kwargs,
    )


def input_responses(role="observed_predictor_input"):
    return {
        "validation_inputs": {"trace": "Trace input.", "field_paths": [PROPOSAL]},
        "validation_input_verify_0": {**VERIFIED, "role": role},
    }


def test_public_schema_and_second_review_control_rejection(monkeypatch):
    responses = input_responses()
    # Bad test-side booleans from the first judge cannot override public facts.
    responses["validation_inputs"]["field_paths"] = [
        {
            **PROPOSAL,
            "test_nondefault_value_reaches_prediction": True,
            "validation_and_test_both_use_same_constant": True,
        }
    ]
    call = install_query(monkeypatch, responses)
    result = review(metric=999, term_out="PRIVATE_VALUE_DO_NOT_SEND")
    assert result["decision"] == "reject" and result["has_submission_mismatch"]
    assert not result["has_leakage"]
    assert result["evidence"][0]["evidence"][-1]["code"] == CODE.splitlines()[5]
    assert all("PRIVATE_VALUE_DO_NOT_SEND" not in str(c.kwargs) for c in call.call_args_list)
    assert all(c.kwargs["temperature_override"] == 0 for c in call.call_args_list)


@pytest.mark.parametrize(
    "role,decision", [("excluded_or_constant", "allow"), ("unresolved", "advisory")]
)
def test_unconfirmed_input_does_not_reject(monkeypatch, role, decision):
    install_query(monkeypatch, input_responses(role))
    result = review()
    assert result["decision"] == decision and not result["has_submission_mismatch"]


def test_target_labels_and_invented_missing_fields_are_not_input_candidates(monkeypatch):
    fields = [{**PROPOSAL, "field": f} for f in ("y", "not_in_public_train")]
    call = install_query(
        monkeypatch, {"validation_inputs": {"trace": "Suspects.", "field_paths": fields}}
    )
    assert review()["decision"] == "allow"
    assert call.call_count == 1


@pytest.mark.parametrize(
    "failure", [TimeoutError("server unavailable"), "length", "invalid_citation"]
)
def test_failed_or_malformed_verification_is_advisory(monkeypatch, failure):
    def broken(kwargs):
        if failure == "length":
            kwargs["info_sink"]["finish_reason"] = "length"
            return VERIFIED
        return {**VERIFIED, "source_lines": [999]}

    replies = input_responses()
    replies["validation_input_verify_0"] = failure if isinstance(failure, Exception) else broken
    install_query(monkeypatch, replies)
    result = review()
    assert result["decision"] == "advisory" and result["status"] == "unavailable"
    assert result["errors"] and not result["has_submission_mismatch"]


def test_numeric_selector_supplies_missing_proposal_but_still_requires_verification(monkeypatch):
    code = 'cols = [c for c in train.columns if train[c].dtype in ["int64", "float64"]]\nmodel.predict(val[cols])\n'
    schema = {
        **SCHEMA,
        "column_dtypes": {"train.json": {"x": "object", "flair": "int64", "y": "bool"}},
    }
    call = install_query(
        monkeypatch,
        {
            "validation_inputs": {"trace": "No fields proposed.", "field_paths": []},
            "validation_input_verify_0": {**VERIFIED, "source_lines": [1, 2],
                                          "raw_read_lines": [1], "selected_feature_lines": [1],
                                          "prediction_call_lines": [2]},
        },
    )
    result = review(code=code, public_schema=schema)
    assert result["decision"] == "reject" and call.call_count == 2
    assert result["phases"]["input_proposals"][0]["field"] == "flair"


def test_advisory_switch_retains_supported_input_feedback(monkeypatch):
    install_query(monkeypatch, input_responses())
    result = review(reject_on_detection=False)
    assert result["decision"] == "advisory" and result["has_submission_mismatch"]
    assert result["evidence"] and "Repair:" in result["reason"]


EXPORT_CODE = """for probability in predictions:
    output["breed"] = probability
submission = pd.DataFrame(output)
submission.to_csv("submission.csv")
"""
EXPORT_FINDING = {
    "trace": "Second assignment overwrites the first scalar.",
    "decision": "reject",
    "findings": [
        {
            "mechanism": "export_value_mismatch",
            "basis": "prediction_values",
            "lines": [2, 3],
            "actual_operations": "The final scalar is broadcast to every ID.",
            "why_incorrect": "Each ID loses its own prediction.",
            "counterevidence_checked": "No append or row-specific assignment.",
            "minimal_repair": "Append each prediction rather than overwriting a scalar.",
        }
    ],
}
CONFIRM = {
    "validation_prediction_path": "Per-record predictions.",
    "test_prediction_path": "CSV loop.",
    "claims": [
        {
            "claim_index": 0,
            "basis": "prediction_values",
            "actual_reached_operations": "Scalar overwrite then broadcast.",
            "source_lines": [2, 3],
            "decisive_guard_or_counterexample": "Distinct pa,pb become pb,pb.",
            "conclusion": "defect_proved",
        }
    ],
}


@pytest.mark.parametrize(
    "conclusion,decision",
    [
        ("defect_proved", "reject"),
        ("source_refutes", "allow"),
        ("insufficient_evidence", "advisory"),
    ],
)
def test_export_requires_complete_independent_confirmation(monkeypatch, conclusion, decision):
    confirmation = copy.deepcopy(CONFIRM)
    confirmation["claims"][0]["conclusion"] = conclusion
    call = install_query(
        monkeypatch,
        {"submission_export": EXPORT_FINDING, "submission_export_confirmation": confirmation},
    )
    result = review(code=EXPORT_CODE, public_schema={})
    assert result["decision"] == decision
    assert EXPORT_FINDING["trace"] not in call.call_args_list[1].kwargs["user_message"]
    assert all(c.kwargs["max_tokens"] <= 3072 for c in call.call_args_list)


def test_confirmation_must_address_each_claim_once(monkeypatch):
    install_query(
        monkeypatch,
        {
            "submission_export": EXPORT_FINDING,
            "submission_export_confirmation": {**CONFIRM, "claims": []},
        },
    )
    result = review(code=EXPORT_CODE, public_schema={})
    assert result["decision"] == "advisory" and result["errors"]


@pytest.mark.parametrize(
    "conclusion,decision",
    [("defect_proved", "reject"), ("source_refutes", "allow"), ("insufficient_evidence", "advisory")],
)
def test_inference_transform_mismatch_reaches_confirmation(monkeypatch, conclusion, decision):
    code = (
        "model.fit(scaler.fit_transform(X_train), y_train)\n"
        "val_pred = model.predict(scaler.transform(X_val))\n"
        "test_pred = model.predict(X_test)\n"
        "pd.DataFrame(test_pred).to_csv('submission.csv')\n"
    )
    finding = copy.deepcopy(EXPORT_FINDING)
    finding["findings"][0].update(
        mechanism="inference_transform_or_state_mismatch",
        actual_operations="Validation scales inputs; test prediction omits the fitted scaler.",
        minimal_repair="Apply the fitted scaler to test inputs before prediction.",
    )
    confirmation = copy.deepcopy(CONFIRM)
    confirmation["claims"][0]["conclusion"] = conclusion
    call = install_query(monkeypatch, {
        "submission_export": finding, "submission_export_confirmation": confirmation,
    })
    result = review(code=code, public_schema={})
    assert call.call_count == 2
    assert result["decision"] == decision
    assert result["has_submission_mismatch"] is (conclusion == "defect_proved")
    if conclusion == "defect_proved":
        assert result["evidence"][0]["mechanism"] == "inference_transform_or_state_mismatch"


@pytest.mark.parametrize("mechanism", ["target_contamination", "input_information_mismatch"])
def test_out_of_scope_export_finding_is_not_a_clean_allow(monkeypatch, mechanism):
    finding = copy.deepcopy(EXPORT_FINDING)
    finding["findings"][0]["mechanism"] = mechanism
    install_query(monkeypatch, {"submission_export": finding})
    result = review(code=EXPORT_CODE, public_schema={})
    assert result["decision"] == "advisory"
    assert result["uncertain_phases"]
    assert result["phases"]["export_out_of_scope"][0]["mechanism"] == mechanism


ID_CODE = """import pandas as pd
import os
def main():
    sample = pd.read_csv("sample_submission.csv")
    ids = sample["id"].values.tolist()
    paths = ["cat.jpg"]
    keys = [os.path.basename(p).replace(".jpg", "") for p in paths]
    preds = [0.25]
    mapping = dict(zip(keys, preds))
    output = [mapping.get(x, 0.5) for x in ids]
    frame = pd.DataFrame({"id": ids, "prediction": output})
    frame.to_csv("submission.csv")
if __name__ == "__main__":
    main()
"""
ID_SCHEMA = {
    "files": [
        {
            "file": "sample_submission.csv",
            "columns": ["id", "prediction"],
            "id_examples": ["cat.jpg"],
        }
    ]
}


@pytest.mark.parametrize("extension", [".jpg", ".png"])
def test_id_patterns_do_not_bypass_export_model_review(monkeypatch, extension):
    code = ID_CODE.replace(".jpg", extension)
    schema = copy.deepcopy(ID_SCHEMA)
    schema["files"][0]["id_examples"] = ["cat" + extension]
    call = install_query(monkeypatch, {"submission_export": ALLOW_EXPORT})
    result = review(code=code, public_schema=schema)
    assert result["decision"] == "allow" and not result["has_submission_mismatch"]
    assert call.call_count == 1
    assert "submission_export" in result["phases"]


@pytest.mark.parametrize(
    "conclusion,decision",
    [
        ("defect_proved", "reject"),
        ("source_refutes", "allow"),
        ("insufficient_evidence", "advisory"),
    ],
)
def test_id_export_findings_require_model_confirmation(monkeypatch, conclusion, decision):
    proposal = copy.deepcopy(EXPORT_FINDING)
    proposal["trace"] = "Prediction keys and requested public IDs use different formats."
    proposal["findings"][0].update(
        lines=[7, 9, 10],
        actual_operations="Keys remove the extension; unmodified public IDs select defaults.",
        why_incorrect="Export loses the predictions associated with the requested IDs.",
        counterevidence_checked="Public IDs are read without normalization.",
        minimal_repair="Use consistent key and lookup transformations.",
    )
    confirmation = copy.deepcopy(CONFIRM)
    confirmation["claims"][0].update(
        actual_reached_operations="Key normalization differs from lookup normalization.",
        source_lines=[7, 9, 10],
        decisive_guard_or_counterexample="Compare actual keys with public ID examples.",
        conclusion=conclusion,
    )
    call = install_query(
        monkeypatch,
        {"submission_export": proposal, "submission_export_confirmation": confirmation},
    )
    result = review(code=ID_CODE, public_schema=ID_SCHEMA)
    assert result["decision"] == decision
    assert result["has_submission_mismatch"] == (decision == "reject")
    assert call.call_count == 2


def test_unavailable_id_review_cannot_trigger_automatic_rejection(monkeypatch):
    call = install_query(monkeypatch, {"submission_export": TimeoutError("review unavailable")})
    result = review(code=ID_CODE, public_schema=ID_SCHEMA)
    assert result["decision"] == "advisory" and not result["has_submission_mismatch"]
    assert call.call_count == 1 and result["errors"]


AGG_CODE = """stats = train.groupby("user").agg(rate=("y", "mean"))
features = train.merge(stats, on="user")
train_part, val = train_test_split(features)
metric = score(val.y, model.predict(val))
"""


@pytest.mark.parametrize("review_submission", [False, True])
def test_named_target_statistic_does_not_add_review_phase(monkeypatch, review_submission):
    call = install_query(
        monkeypatch,
        {"audit": "Trace.", "check_data_leakage": ALLOW_LEAK},
    )
    result = checker.check_validation(
        task_desc="Predict y.",
        code=AGG_CODE,
        model=MODEL,
        public_schema={"designated_raw_target_fields": ["y"]},
        review_submission=review_submission,
    )
    assert result["decision"] == "allow" and not result["has_leakage"]
    assert not result["has_submission_mismatch"]
    assert not result["errors"] and call.call_count == 2
    assert "aggregate_review" not in result
    assert (result["submission_review"] is not None) == review_submission


@pytest.mark.parametrize("global_reject,decision", [(True, "reject"), (False, "advisory")])
def test_original_leakage_review_still_handles_target_statistics(
    monkeypatch, global_reject, decision
):
    finding = {
        "mechanism": "validation_targets_in_predictions",
        "data_flow": "All labeled targets supply a rate merged into the later validation rows.",
        "why_not_permitted": "Scored targets contribute to a prediction feature.",
        "source_lines": [1, 2, 3, 4],
        "repair": "Fit the target statistic on fitting rows only before validation prediction.",
    }
    positive = {
        **ALLOW_LEAK,
        "validation_protocol": "Validation is split after target-derived features are built.",
        "counterevidence": "The later split does not remove validation targets from the rate.",
        "findings": [finding],
        "reason": finding["data_flow"],
        "verdict": "leak_detected",
    }
    confirmation = {"findings": [{
        "finding_index": 0,
        "source_lines": [1, 2, 3, 4],
        "analysis": "The statistic includes targets from rows later scored in validation.",
        "disposition": "contamination_supported",
    }]}
    call = install_query(
        monkeypatch,
        {"audit": "Trace.", "check_data_leakage": positive,
         "verify_leakage_findings": confirmation},
    )
    result = checker.check_validation(
        task_desc="Predict y.",
        code=AGG_CODE,
        model=MODEL,
        public_schema={"designated_raw_target_fields": ["y"]},
        review_submission=False,
        reject_on_detection=global_reject,
    )
    assert result["decision"] == decision and result["has_leakage"]
    assert result["evidence"][0]["data_flow"] == finding["data_flow"]
    assert result["submission_review"] is None and not result["errors"]
    assert call.call_count == 3


def test_focused_verification_shares_remaining_original_budget(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(checker.time, "monotonic", lambda: clock[0])
    replies = {"audit": "Trace.", "check_data_leakage": ALLOW_LEAK, **input_responses()}

    def answer(**kwargs):
        phase = (
            kwargs["response_format"]["json_schema"]["name"]
            if kwargs.get("response_format")
            else "audit"
        )
        clock[0] += 2
        return copy.deepcopy(replies[phase])

    call = Mock(side_effect=answer)
    monkeypatch.setattr(checker, "query", call)
    result = checker.check_validation(
        task_desc="Predict.", code=CODE, model=MODEL, public_schema=SCHEMA, timeout_s=8
    )
    assert result["decision"] == "reject"
    assert [c.kwargs["retry_max_time"] for c in call.call_args_list] == [8, 6, 4, 2]
    clock[0] = 0
    call.reset_mock()
    result = checker.check_validation(
        task_desc="Predict.", code=CODE, model=MODEL, public_schema=SCHEMA, timeout_s=6
    )
    assert result["decision"] == "advisory" and call.call_count == 3


def test_public_json_schema_scans_all_records_and_excludes_template_target(tmp_path):
    (tmp_path / "train.json").write_text(json.dumps([{"x": 1, "y": True, "flair": 2}]))
    (tmp_path / "test.json").write_text(json.dumps([{"x": 1}, {"x": 2, "rare": 9}]))
    (tmp_path / "sample_submission.csv").write_text("id,y\na,0\nb,0\n")
    loader = validation_schema.PublicValidationSchema(tmp_path)
    schema = loader.load(time.monotonic() + 20)
    assert schema["files"][1]["key_presence_counts"]["rare"] == 1
    assert schema["designated_raw_target_fields"] == ["y"]
    assert schema["column_dtypes"]["train.json"]["y"] == "bool"
    assert evidence.absent_fields(schema) == {"flair"}
    (tmp_path / "test.json").write_text(json.dumps([{"x": 1, "flair": 4}]))
    assert evidence.absent_fields(loader.load(time.monotonic() + 20)) == set()


def test_partial_json_never_proves_absent_fields(tmp_path, monkeypatch):
    (tmp_path / "train.json").write_text('[{"flair": 2}]')
    (tmp_path / "test.json").write_text(json.dumps([{"x": "a" * 300}, {"flair": 2}]))
    (tmp_path / "sample_submission.csv").write_text("id,y\na,0\n")
    monkeypatch.setattr(validation_schema, "MAX_BYTES", 100)
    schema = validation_schema.PublicValidationSchema(tmp_path).load(time.monotonic() + 20)
    assert not evidence.absent_fields(schema)
    assert not evidence.schema_branch_facts(CODE, schema)
    compact = evidence.compact_schema(schema)
    assert not evidence.absent_fields(compact)
    assert compact["diagnostics"] == schema["diagnostics"]
    assert schema["diagnostics"]


def test_partial_sample_keeps_header_without_inventing_id_examples(tmp_path, monkeypatch):
    (tmp_path / "sample_submission.csv").write_text(
        "id,prediction\ncat.jpg,0\n" + "a" * 300 + ",0\n"
    )
    monkeypatch.setattr(validation_schema, "MAX_BYTES", 32)
    monkeypatch.setattr(validation_schema, "MAX_HEADER_BYTES", 32)
    schema = validation_schema.PublicValidationSchema(tmp_path).load(time.monotonic() + 20)
    assert schema["files"][0]["columns"] == ["id", "prediction"]
    assert "id_examples" not in schema["files"][0]


def test_zip_headers_and_ambiguous_test_selection(tmp_path):
    for name, content in [
        ("en_train.csv", "before,class,after\na,LETTERS,a\n"),
        ("en_test_2.csv", "before\na\n"),
        ("en_sample_submission.csv", "id,after\n1,a\n"),
    ]:
        with zipfile.ZipFile(tmp_path / (name + ".zip"), "w") as z:
            z.writestr(name, content)
    loader = validation_schema.PublicValidationSchema(tmp_path)
    assert evidence.absent_fields(loader.load(time.monotonic() + 20)) == {"class"}
    (tmp_path / "another_test.csv").write_text("before,class\na,LETTERS\n")
    assert not evidence.absent_fields(loader.load(time.monotonic() + 20))


def test_large_image_directory_does_not_discard_complete_root_sample(tmp_path):
    (tmp_path / "sample_submission.csv").write_text("id,prediction\ncat.jpg,0\n")
    images = tmp_path / "test"
    images.mkdir()
    for i in range(513):
        (images / f"{i}.jpg").touch()
    schema = validation_schema.PublicValidationSchema(tmp_path).load(time.monotonic() + 20)
    assert schema["files"][0]["id_examples"] == ["cat.jpg"]
    assert schema["files"][0]["columns"] == ["id", "prediction"]


@pytest.mark.parametrize("confirmed", [True, False])
def test_actual_agent_supplies_public_schema_and_rejects_only_confirmed_input(
    tmp_path, monkeypatch, confirmed
):
    inputs = tmp_path / "input"
    inputs.mkdir()
    (inputs / "train.json").write_text('[{"x":1,"flair":2,"y":true}]')
    (inputs / "test.json").write_text('[{"x":1}]')
    (inputs / "sample_submission.csv").write_text("id,y\na,0\n")
    agent = object.__new__(Agent)
    agent.cfg = SimpleNamespace(workspace_dir=tmp_path)
    agent.acfg = SimpleNamespace(
        check_data_leakage=True,
        leakage_max_threshold=0.999,
        leakage_min_threshold=1e-6,
        leakage_review_best=True,
        leakage_timeout_s=120,
        feedback=SimpleNamespace(model=MODEL, temp=0.5),
        eval_metric="accuracy",
        convert_system_to_user=False,
    )
    agent.task_desc = "Predict y."
    agent.journal = Journal()
    agent.deadline = time.time() + 600
    n = Node(
        code=CODE,
        metric=MetricValue(0.82, maximize=True),
        metric_trustworthy=True,
        is_buggy=False,
        real_won=True,
        submission_validation="valid",
        submission_sha256="a" * 64,
        submission_columns=["id", "y"],
        analysis="original",
    )
    responses = {"audit": "Trace.", "check_data_leakage": ALLOW_LEAK, **input_responses()}
    if not confirmed:
        responses["validation_input_verify_0"]["raw_read_lines"] = [6]
    install_query(monkeypatch, responses)
    agent._check_data_leakage(n)
    receipt = n.leakage_check["submission_review"]["validated_submission"]
    assert receipt == {"status": "valid", "sha256": "a" * 64, "actual_columns": ["id", "y"]}
    assert n.leakage_check["submission_policy_version"] == checker.SUBMISSION_POLICY_VERSION
    if not confirmed:
        assert n.is_buggy is False and n.metric_trustworthy and n.metric.value == 0.82
        assert n.leakage_check["decision"] == "advisory"
        assert not n.leakage_check["has_submission_mismatch"]
        assert not n.failure_reasons and n.analysis == "original"
        return
    assert n.is_buggy and not n.metric_trustworthy and n.metric.value is None
    assert n.leakage_check["has_submission_mismatch"]
    assert set(n.failure_reasons) == {"submission_pipeline"}
    assert "same information available" in n.analysis


@pytest.mark.parametrize("real", [False, True])
def test_extreme_only_review_skips_public_schema_loading(tmp_path, monkeypatch, real):
    agent = object.__new__(Agent)
    agent.cfg = SimpleNamespace(workspace_dir=tmp_path)
    agent.acfg = SimpleNamespace(
        check_data_leakage=True,
        leakage_max_threshold=0.999,
        leakage_min_threshold=1e-6,
        leakage_review_best=False,
        feedback=SimpleNamespace(model=MODEL, temp=0.5),
        eval_metric="accuracy",
        convert_system_to_user=False,
    )
    agent.task_desc = "Predict y."
    agent.journal = Journal()
    n = Node(
        code=AGG_CODE,
        metric=MetricValue(1.0, maximize=True),
        metric_trustworthy=True,
        is_buggy=False,
        real_won=real,
        submission_validation="valid" if real else None,
        analysis="original",
    )
    load_schema = Mock(side_effect=AssertionError("No schema needed for leakage-only review"))
    monkeypatch.setattr(validation_schema.PublicValidationSchema, "load", load_schema)
    call = install_query(monkeypatch, {"audit": "Trace.", "check_data_leakage": ALLOW_LEAK})
    agent._check_data_leakage(n)
    load_schema.assert_not_called()
    assert call.call_count == 2
    assert n.leakage_check["triggers"] == ["extreme_score"]
    assert n.leakage_check["decision"] == "allow" and not n.is_buggy
    assert n.leakage_check["submission_review"] is None
