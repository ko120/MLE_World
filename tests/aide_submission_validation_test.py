"""Validate the exact artifact and distinguish rejection from no evidence."""

import hashlib
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "agents/aide/src"))
from aide.utils.submission_validation import SubmissionValidator


CSV = b"id,pred\n0,0.8\n"
URL = "http://validator/validate"


def _response(payload=None, status=200):
    return SimpleNamespace(status_code=status, json=lambda: payload)


@pytest.mark.parametrize("status,payload,expected", [
    (200, {"result": "Submission is valid."}, "valid"),
    (200, {"result": "Submission invalid! Wrong columns."}, "invalid"),
    (200, {"result": "unknown"}, "unknown"),
    (200, {"result": None}, "unknown"),
    (200, {"result": True}, "unknown"),
    (200, {"error": "grader failed"}, "unknown"),
    (200, ["Submission is valid."], "unknown"),
    (500, {"result": "Submission is valid."}, "unknown"),
    (504, {"error": "timeout"}, "unknown"),
])
def test_only_explicit_verdicts_are_definitive(tmp_path, monkeypatch, status, payload, expected):
    path = tmp_path / "submission.csv"
    path.write_bytes(CSV)
    monkeypatch.setattr(requests, "post", Mock(return_value=_response(payload, status)))
    result = SubmissionValidator().validate(path, url=URL)
    assert result.status == expected
    assert result.sha256 == hashlib.sha256(CSV).hexdigest()


@pytest.mark.parametrize("artifact", ["missing", "empty", "directory"])
def test_missing_and_empty_files_do_not_contact_server(tmp_path, monkeypatch, artifact):
    path = tmp_path / "submission.csv"
    if artifact == "empty":
        path.touch()
    elif artifact == "directory":
        path.mkdir()
    post = Mock()
    monkeypatch.setattr(requests, "post", post)
    assert SubmissionValidator().validate(path, url=URL).status == "invalid"
    post.assert_not_called()


@pytest.mark.parametrize("message", ["Submission is valid.", "Submission invalid! Wrong IDs."])
def test_cache_follows_contents_and_competition_not_filename(tmp_path, monkeypatch, message):
    path = tmp_path / "submission.csv"
    copy = tmp_path / "copy.csv"
    path.write_bytes(CSV)
    copy.write_bytes(CSV)
    post = Mock(return_value=_response({"result": message}))
    monkeypatch.setattr(requests, "post", post)
    validator = SubmissionValidator()
    first = validator.validate(path, url=URL, competition_id="a")
    assert validator.validate(copy, url=URL, competition_id="a") == first
    assert post.call_count == 1
    path.write_bytes(b"id,pred\n0,0.1\n")
    assert validator.validate(path, url=URL, competition_id="a").sha256 != first.sha256
    validator.validate(copy, url=URL, competition_id="b")
    validator.validate(copy, url=URL + "-new", competition_id="a")
    assert post.call_count == 4


@pytest.mark.parametrize("failure", [
    requests.Timeout("timed out"),
    requests.ConnectionError("connection refused"),
    _response({"error": "grader failed"}, 500),
])
def test_unavailable_verdict_is_retried_and_can_recover(tmp_path, monkeypatch, failure):
    path = tmp_path / "submission.csv"
    path.write_bytes(CSV)
    post = Mock(side_effect=[failure, _response({"result": "Submission is valid."})])
    monkeypatch.setattr(requests, "post", post)
    validator = SubmissionValidator(retry_delay_s=0)
    assert validator.validate(path, url=URL).status == "unknown"
    assert validator.validate(path, url=URL).status == "valid"
    assert post.call_count == 2


def test_malformed_json_is_unknown(tmp_path, monkeypatch):
    path = tmp_path / "submission.csv"
    path.write_bytes(CSV)
    response = SimpleNamespace(status_code=200, json=Mock(side_effect=ValueError("bad JSON")))
    monkeypatch.setattr(requests, "post", Mock(return_value=response))
    assert SubmissionValidator().validate(path, url=URL).status == "unknown"


def test_upload_is_a_snapshot_matching_the_reported_hash(tmp_path, monkeypatch):
    path = tmp_path / "submission.csv"
    path.write_bytes(CSV)

    def post(url, *, files, timeout):
        path.write_bytes(b"different submission")
        assert files["file"][1].read() == CSV
        return _response({"result": "Submission is valid."})

    monkeypatch.setattr(requests, "post", post)
    result = SubmissionValidator().validate(path, url=URL)
    assert result.sha256 == hashlib.sha256(CSV).hexdigest()


def test_exhausted_budget_leaves_submission_unverified(tmp_path, monkeypatch):
    path = tmp_path / "submission.csv"
    path.write_bytes(CSV)
    post = Mock()
    monkeypatch.setattr(requests, "post", post)
    assert SubmissionValidator().validate(path, url=URL, timeout=0).status == "unknown"
    post.assert_not_called()


def test_server_cooldown_avoids_repeated_timeouts_across_candidates(tmp_path, monkeypatch):
    first = tmp_path / "first.csv"
    second = tmp_path / "second.csv"
    first.write_bytes(CSV)
    second.write_bytes(b"id,pred\n0,0.9\n")
    post = Mock(side_effect=[requests.Timeout(), _response({"result": "Submission is valid."})])
    monkeypatch.setattr(requests, "post", post)
    now = [100.0]
    monkeypatch.setattr("aide.utils.submission_validation.time.monotonic", lambda: now[0])
    validator = SubmissionValidator()
    assert validator.validate(first, url=URL).status == "unknown"
    assert validator.validate(second, url=URL).status == "unknown"
    assert post.call_count == 1
    now[0] += 61
    assert validator.validate(second, url=URL).status == "valid"
    assert post.call_count == 2
