"""Precision regressions spanning format authority and genuine value defects."""
import copy
import hashlib
import json
import time
from types import SimpleNamespace

import pytest

import aide_evidence_review_test as base
from aide import validation_evidence as evidence
from aide.utils.submission_validation import SubmissionValidator
from aide.utils.validation_schema import PublicValidationSchema


RECEIPT = {"status": "valid", "sha256": "a" * 64, "columns": ["id", "probability"]}


@pytest.mark.parametrize("basis", ["structural_format", "training_or_quality", "metric_only"])
def test_out_of_scope_complaint_cannot_veto_candidate(monkeypatch, basis):
    result = copy.deepcopy(base.EXPORT_FINDING)
    result["findings"][0]["basis"] = basis
    query = base.install_query(monkeypatch, {"submission_export": result})
    verdict = base.review(code=base.EXPORT_CODE, public_schema={}, validated_submission=RECEIPT)
    assert verdict["decision"] == "allow"
    assert not verdict["has_submission_mismatch"]
    assert verdict["phases"]["export_nondefect_claims_dropped"]
    assert query.call_count == 1


@pytest.mark.parametrize("basis,code,claim", [
    ("row_alignment", 'pred = model.predict(test)\n'
        'submission = pd.DataFrame({"id": test.id.iloc[::-1].to_numpy(), "target": pred})\n'
        'submission.to_csv("submission.csv")\n',
        "Predictions retain test order but IDs are reversed."),
    ("class_alignment", 'pred = model.predict_proba(test)\n'
        'submission = pd.DataFrame(pred[:, ::-1], columns=model.classes_)\n'
        'submission.to_csv("submission.csv")\n',
        "Prediction columns are reversed without reversing class names."),
    ("prediction_values", 'pred = model.predict_proba(test)[:, 1]\n'
        'submission = pd.DataFrame({"probability": pred.astype(int)})\n'
        'submission.to_csv("submission.csv")\n',
        "Casting probabilities to integers destroys fractional predictions."),
    ("inference_pipeline", 'model.fit(scaler.fit_transform(train), y)\n'
        'pred = model.predict(scaler.fit_transform(test))\n'
        'pd.DataFrame(pred).to_csv("submission.csv")\n',
        "Test refits the scaler instead of applying the fitted training transform."),
])
def test_valid_format_does_not_hide_value_or_inference_defect(monkeypatch, basis, code, claim):
    proposal = copy.deepcopy(base.EXPORT_FINDING)
    confirmation = copy.deepcopy(base.CONFIRM)
    proposal["findings"][0].update(basis=basis, actual_operations=claim)
    confirmation["claims"][0].update(basis=basis, actual_reached_operations=claim)
    query = base.install_query(monkeypatch, {
        "submission_export": proposal, "submission_export_confirmation": confirmation,
    })
    verdict = base.review(code=code, public_schema={}, validated_submission=RECEIPT)
    assert verdict["decision"] == "reject"
    assert all(json.loads(c.kwargs["user_message"])["validated_submission"]["sha256"] == RECEIPT["sha256"]
               for c in query.call_args_list)
    assert all(json.loads(c.kwargs["user_message"])["task_description"] == "Use public data."
               for c in query.call_args_list)


def test_independent_confirmation_can_recognize_misclassified_format_claim(monkeypatch):
    confirmation = copy.deepcopy(base.CONFIRM)
    confirmation["claims"][0]["basis"] = "structural_format"
    base.install_query(monkeypatch, {
        "submission_export": base.EXPORT_FINDING,
        "submission_export_confirmation": confirmation,
    })
    result = base.review(code=base.EXPORT_CODE, public_schema={}, validated_submission=RECEIPT)
    assert result["decision"] == "allow"
    assert result["phases"]["export_confirmation_claims_dropped"]


@pytest.mark.parametrize("conclusion,decision", [("defect_proved", "reject"), ("source_refutes", "allow")])
def test_wrong_record_claim_reaches_confirmation_despite_input_category(monkeypatch, conclusion, decision):
    code = ('pred = model.predict(train_features[:len(test)])\n'
            'submission = pd.DataFrame({"id": test.id, "target": pred})\n'
            'submission.to_csv("submission.csv")\n')
    proposal = copy.deepcopy(base.EXPORT_FINDING)
    proposal["findings"][0].update(
        mechanism="input_information_mismatch", basis="row_alignment",
        actual_operations="Training feature rows supply predictions paired with test IDs.")
    confirmation = copy.deepcopy(base.CONFIRM)
    confirmation["claims"][0].update(basis="row_alignment", conclusion=conclusion)
    query = base.install_query(monkeypatch, {
        "submission_export": proposal, "submission_export_confirmation": confirmation,
    })
    result = base.review(code=code, public_schema={}, validated_submission=RECEIPT)
    assert result["decision"] == decision and query.call_count == 2
    assert result["phases"]["export_alignment_claims_routed"][0]["mechanism"] == "input_information_mismatch"


@pytest.mark.parametrize("status,digest", [("unknown", "a"*64), ("invalid", "a"*64), ("valid", None)])
def test_unknown_invalid_or_unbound_verdict_is_not_format_authority(monkeypatch, status, digest):
    query = base.install_query(monkeypatch, {"submission_export": base.ALLOW_EXPORT})
    result = base.review(code=base.EXPORT_CODE, public_schema={},
                        validated_submission={"status": status, "sha256": digest})
    assert result["validated_submission"] is None
    assert json.loads(query.call_args.kwargs["user_message"])["validated_submission"] is None


@pytest.mark.parametrize("patch", [
    {"selected_feature_lines": []}, {"prediction_call_lines": [999]},
    {"role": "unresolved"}, {"raw_field": "another_field"},
    {"raw_read_lines": [6]},
])
def test_positive_role_requires_consistent_reachable_input_witness(monkeypatch, patch):
    replies = base.input_responses()
    replies["validation_input_verify_0"] = {**base.VERIFIED, **patch}
    base.install_query(monkeypatch, replies)
    result = base.review()
    assert result["decision"] == "advisory" and not result["has_submission_mismatch"]
    assert result["errors"] or result["uncertain_phases"]


def test_similarly_named_field_is_not_a_raw_field_read(monkeypatch):
    query = base.install_query(monkeypatch, {
        "validation_inputs": {"trace": "Incorrect proposal", "field_paths": [base.PROPOSAL]},
    })
    result = base.review(code=base.CODE.replace('"flair"', '"flair_available_at_request"'))
    assert result["decision"] == "advisory"
    assert query.call_count == 1


def test_cross_dataframe_guard_is_not_evaluated_as_two_copies_of_train():
    code = 'if "flair" in train.columns and "flair" in test.columns:\n    selected.append("flair")\n'
    assert evidence.schema_branch_facts(code, base.SCHEMA) == []
    facts = evidence.column_filter_facts(code, base.SCHEMA)
    assert "flair" not in facts["shared_raw_columns"]
    assert facts["possible_column_filters"][0]["line"] == 1


def test_template_integer_zeros_are_not_input_dtypes_or_identifier_examples(tmp_path):
    (tmp_path/'train.csv').write_text('feature,target\n2,1\n')
    (tmp_path/'test.csv').write_text('feature\n3\n')
    (tmp_path/'sample_submission.csv').write_text('target,Comment\n0,example\n')
    schema = PublicValidationSchema(tmp_path).load(time.monotonic()+20)
    assert 'sample_submission.csv' not in schema['column_dtypes']
    assert 'train.csv' in schema['column_dtypes']
    sample = next(f for f in schema['files'] if f['role']=='sample')
    assert sample['dtype_semantics']=='placeholder_values_not_output_constraints'
    assert 'id_examples' not in sample
    # Old serialized schemas must be safe too.
    schema['column_dtypes']['sample_submission.csv']={'target':'int64'}
    sample['id_examples']=['0']
    compact = evidence.compact_schema(schema)
    assert 'sample_submission.csv' not in compact['column_dtypes']
    assert 'id_examples' not in next(f for f in compact['files'] if f['role']=='sample')
    assert sample['id_examples'] == ['0'], 'Sanitization must not mutate the caller cache.'
    code='cols = [c for c in train.columns if train[c].dtype == "int64"]'
    assert all('sample_submission.csv' not in f['raw_public_columns_satisfying_predicate']
               for f in evidence.numeric_selection_facts(code,schema))


def test_receipt_header_and_hash_describe_uploaded_snapshot(tmp_path, monkeypatch):
    path=tmp_path/'submission.csv'
    data=b'prediction,id\n0.25,row\n'
    path.write_bytes(data)
    def post(url, *, files, timeout):
        assert files['file'][1].read()==data
        path.write_text('different,columns\n9,other\n')
        return SimpleNamespace(status_code=200,json=lambda:{'result':'Submission is valid.'})
    monkeypatch.setattr('requests.post',post)
    receipt=SubmissionValidator().validate(path,url='http://mock/validate')
    assert receipt.status=='valid'
    assert receipt.sha256==hashlib.sha256(data).hexdigest()
    assert receipt.columns==('prediction','id')
