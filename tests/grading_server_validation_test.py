"""Exercise validation uploads and the real process timeout boundary."""

import io
import multiprocessing
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("flask")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "environment"))
import grading_server as server
from mlebench import grade
from mlebench.grade_helpers import InvalidSubmissionError


def _post(content):
    with server.app.test_client() as client:
        return client.post("/validate", data={"file": (io.BytesIO(content), "submission.csv")})


@pytest.mark.parametrize("kind,status,prefix", [
    ("valid", 200, "Submission is valid."),
    ("invalid", 200, "Submission invalid!"),
    ("grader_error", 500, None),
])
def test_real_validation_returns_format_evidence_without_score(tmp_path, monkeypatch, kind, status, prefix):
    answers = tmp_path / "answers.csv"
    answers.write_text("id,pred\n0,1\n")

    def grader(submission, private_answers):
        if kind == "invalid":
            raise InvalidSubmissionError("Submission has wrong IDs.")
        if kind == "grader_error":
            raise RuntimeError("private_score=0.12345")
        return 0.12345

    competition = SimpleNamespace(id="test", answers=answers, grader=SimpleNamespace(grade_fn=grader))
    registry = SimpleNamespace(get_competition=lambda _: competition)
    monkeypatch.setattr(server, "registry", SimpleNamespace(set_data_dir=lambda _: registry))
    monkeypatch.setattr(grade, "is_dataset_prepared", lambda *args, **kwargs: True)
    response = _post(b"id,pred\n0,0.8\n")
    assert response.status_code == status
    body = response.get_json()
    if prefix is not None:
        assert body["result"].startswith(prefix)
    else:
        assert "result" not in body
    assert "score" not in body
    assert "0.12345" not in response.get_data(as_text=True)


def test_concurrent_requests_use_distinct_files_and_remove_them(monkeypatch):
    barrier = threading.Barrier(2, timeout=5)
    paths = []

    def validate(path):
        paths.append(path)
        barrier.wait()
        content = path.read_bytes()
        result = "Submission is valid." if content == b"valid" else "Submission invalid! Bad CSV."
        return {"result": result}, 200

    monkeypatch.setattr(server, "_run_validation_bounded", validate)
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(_post, b"valid")
        second = pool.submit(_post, b"invalid")
        assert first.result(timeout=10).get_json()["result"] == "Submission is valid."
        assert second.result(timeout=10).get_json()["result"].startswith("Submission invalid!")
    assert len(set(paths)) == 2
    assert all(not path.parent.exists() for path in paths)


def test_hung_validator_is_killed_and_server_recovers(tmp_path, monkeypatch):
    before = {child.pid for child in multiprocessing.active_children()}
    monkeypatch.setattr(server, "VALIDATE_TIMEOUT_SECS", 0.1)
    monkeypatch.setattr(server, "run_validation", lambda path: time.sleep(10))
    started = time.monotonic()
    response = _post(b"id,pred\n0,1\n")
    assert response.status_code == 504
    assert time.monotonic() - started < 5
    assert {child.pid for child in multiprocessing.active_children()} == before
    monkeypatch.setattr(server, "run_validation", lambda path: "Submission is valid.")
    assert _post(b"id,pred\n0,1\n").get_json() == {"result": "Submission is valid."}


def test_crashed_validator_is_unknown_and_upload_is_removed(monkeypatch):
    paths = []
    original = server._run_validation_bounded

    def bounded(path):
        paths.append(path)
        return original(path)

    monkeypatch.setattr(server, "_run_validation_bounded", bounded)
    monkeypatch.setattr(server, "run_validation", lambda path: os._exit(7))
    response = _post(b"id,pred\n0,1\n")
    assert response.status_code == 500
    assert "result" not in response.get_json()
    assert all(not path.parent.exists() for path in paths)


def test_missing_upload_returns_client_error():
    with server.app.test_client() as client:
        response = client.post("/validate")
    assert response.status_code == 400


def test_oversized_upload_does_not_start_grader(monkeypatch):
    monkeypatch.setattr(server, "MAX_VALIDATE_BYTES", 3)
    monkeypatch.setattr(server, "_run_validation_bounded", lambda path: pytest.fail("unexpected grader call"))
    assert _post(b"1234").status_code == 413
