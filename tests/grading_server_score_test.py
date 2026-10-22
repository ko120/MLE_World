"""Exercise scoring uploads, real graders, and the process deadline."""

import importlib
import io
import json
import multiprocessing
import os
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("flask")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "environment"))
import grading_server as server
from mlebench import grade
from mlebench.grade_helpers import InvalidSubmissionError
from mlebench.utils import load_answers, read_csv


@pytest.fixture(autouse=True)
def check_cleanup(tmp_path, monkeypatch):
    temporary_directory = server.tempfile.TemporaryDirectory

    def upload_directory(*args, **kwargs):
        return temporary_directory(*args, **{**kwargs, "dir": tmp_path})

    monkeypatch.setattr(server.tempfile, "TemporaryDirectory", upload_directory)
    monkeypatch.setattr(server, "SCORE_TIMEOUT_SECS", 3)
    before = {child.pid for child in multiprocessing.active_children()}
    yield
    remaining = [child for child in multiprocessing.active_children() if child.pid not in before]
    try:
        assert not remaining, "Grading left a live child process"
        assert not list(tmp_path.glob("submission-*")), "Grading left an upload directory"
    finally:
        for child in remaining:
            child.kill()
            child.join(timeout=5)
            child.close()


@pytest.fixture
def configure_grader(tmp_path, monkeypatch):
    def configure(grader, answers=b"id,pred\n0,1\n", *, competition_id="test", suffix=".csv"):
        answers_path = tmp_path / f"answers{suffix}"
        answers_path.write_bytes(answers)
        competition = SimpleNamespace(
            id=competition_id, answers=answers_path, grader=SimpleNamespace(grade_fn=grader)
        )
        registry = SimpleNamespace(get_competition=lambda _: competition)
        monkeypatch.setattr(server, "registry", SimpleNamespace(set_data_dir=lambda _: registry))
        monkeypatch.setattr(server, "COMPETITION_ID", competition_id)
        monkeypatch.setattr(grade, "is_dataset_prepared", lambda *args, **kwargs: True)
        return competition

    return configure


def _post(content=b"id,pred\n0,0.5\n", endpoint="score"):
    with server.app.test_client() as client:
        return client.post(f"/{endpoint}", data={"file": (io.BytesIO(content), "submission.csv")})


@pytest.mark.parametrize("value,expected", [(0.123456789, 0.12346), (0.0, 0.0), (None, None)])
def test_success_preserves_rounding_and_null_contract(configure_grader, value, expected):
    configure_grader(lambda submission, answers: value)
    response = _post()
    assert response.status_code == 200
    assert response.get_json() == {"score": expected, "error": None}


def test_grader_mutation_does_not_change_later_requests(configure_grader):
    def grader(submission, answers):
        result = answers.loc[0, "pred"]
        answers.loc[0, "pred"] = 99
        return result

    configure_grader(grader)
    for _ in range(2):
        assert _post().get_json() == {"score": 1.0, "error": None}


@pytest.mark.parametrize("exception,message", [
    (InvalidSubmissionError("Wrong IDs"), "invalid submission: Wrong IDs"),
    (RuntimeError("grader defect"), "RuntimeError: grader defect"),
])
def test_grader_errors_keep_their_classification(configure_grader, exception, message):
    def grader(submission, answers):
        raise exception

    configure_grader(grader)
    response = _post()
    assert response.status_code == 200
    assert response.get_json() == {"score": None, "error": message}


def test_concurrent_scores_grade_their_own_uploads(monkeypatch):
    barrier = multiprocessing.get_context("fork").Barrier(2, timeout=2)

    def read_after_both_uploads(path):
        barrier.wait()
        return float(path.read_text())

    monkeypatch.setattr(server, "_load_and_grade", read_after_both_uploads)
    bodies = [None, None]

    def request(index, content):
        bodies[index] = _post(content).get_json()

    # Werkzeug uses ordinary request threads; forking from ThreadPoolExecutor
    # workers can fail during Python's inherited executor shutdown instead.
    threads = [threading.Thread(target=request, args=(i, content)) for i, content in enumerate((b"1", b"2"))]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=8)
        assert not thread.is_alive()
    assert bodies == [{"score": 1.0, "error": None}, {"score": 2.0, "error": None}]


def test_long_grader_error_does_not_block_result_delivery(monkeypatch):
    message = "x" * (128 * 1024)

    def grader(path):
        raise RuntimeError(message)

    monkeypatch.setattr(server, "_load_and_grade", grader)
    monkeypatch.setattr(server.logger, "disabled", True)
    body = _post().get_json()
    assert body["score"] is None
    assert body["error"].startswith("RuntimeError: ")
    assert body["error"] == "RuntimeError: " + message


@pytest.mark.parametrize("stage", ["registry", "answers", "submission", "grade"])
def test_deadline_covers_loading_and_grading(configure_grader, monkeypatch, stage):
    competition = configure_grader(lambda submission, answers: 0.5)
    parent_pid = os.getpid()

    def hang(*args):
        assert os.getpid() != parent_pid, "Loading ran in the Flask parent"
        time.sleep(2)

    owner, attribute = {
        "registry": (server.registry, "set_data_dir"),
        "answers": (server, "load_answers"),
        "submission": (server, "read_csv"),
        "grade": (competition.grader, "grade_fn"),
    }[stage]
    monkeypatch.setattr(owner, attribute, hang)
    monkeypatch.setattr(server, "SCORE_TIMEOUT_SECS", 0.1)
    started = time.monotonic()
    body = _post().get_json()
    assert body["score"] is None
    assert "timed out" in body["error"]
    assert time.monotonic() - started < 1.5
    # The parent remains usable after killing the stalled child.
    monkeypatch.setattr(server, "_load_and_grade", lambda path: 0.5)
    assert _post().get_json() == {"score": 0.5, "error": None}


def test_crashed_child_returns_null_and_server_recovers(monkeypatch):
    monkeypatch.setattr(server, "_load_and_grade", lambda path: os._exit(7))
    body = _post().get_json()
    assert body["score"] is None
    assert "exit 7" in body["error"]
    monkeypatch.setattr(server, "_load_and_grade", lambda path: 0.5)
    assert _post().get_json() == {"score": 0.5, "error": None}


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_scores_return_null_with_a_reason(configure_grader, value):
    configure_grader(lambda submission, answers: value)
    response = _post()
    body = json.loads(response.get_data(as_text=True), parse_constant=lambda value: pytest.fail(f"Nonfinite JSON: {value}"))
    assert body["score"] is None
    assert "non-finite score" in body["error"]


def test_undefined_correlation_returns_null(configure_grader):
    def correlation(submission, answers):
        return submission["score"].corr(answers["score"])

    configure_grader(correlation, b"id,score\na,0.0\nb,0.5\nc,1.0\n")
    body = _post(b"id,score\na,0.5\nb,0.5\nc,0.5\n").get_json()
    assert body["score"] is None
    assert "non-finite score" in body["error"]


@pytest.mark.parametrize("language", ["russian", "english"])
def test_text_normalization_handles_missing_values(configure_grader, language):
    module = importlib.import_module(f"mlebench.competitions.text-normalization-challenge-{language}-language.grade")
    content = b"id,after\n0,one\n1,\n2,two\n"
    configure_grader(module.grade, content)
    assert _post(content).get_json() == {"score": 1.0, "error": None}


def test_spooky_wrong_ids_are_explicitly_invalid(configure_grader):
    module = importlib.import_module("mlebench.competitions.spooky-author-identification.grade")
    configure_grader(module.grade, b"id,EAP,HPL,MWS\na,1,0,0\nb,0,1,0\nc,0,0,1\n")
    body = _post(b"id,EAP,HPL,MWS\nx,0.2,0.3,0.5\nb,0.2,0.3,0.5\nc,0.2,0.3,0.5\n").get_json()
    assert body["error"].startswith("invalid submission: Mismatch in Ids")


@pytest.mark.parametrize("endpoint", ["score", "validate"])
def test_jsonl_answers_match_offline_grading(configure_grader, tmp_path, endpoint):
    def jsonl_grader(submission, answers):
        expected = {answer["id"]: answer["prediction"] for answer in answers}
        actual = dict(zip(submission["id"], submission["prediction"]))
        return float(actual == expected)

    answers = json.dumps({"id": "a", "prediction": "one"}) + "\n"
    competition = configure_grader(
        jsonl_grader, answers.encode(), suffix=".jsonl"
    )
    content = b"id,prediction\na,one\n"
    submission = tmp_path / "gold.csv"
    submission.write_bytes(content)
    assert jsonl_grader(read_csv(submission), load_answers(competition.answers)) == 1.0
    response = _post(content, endpoint)
    assert response.status_code == 200
    expected = {"score": 1.0, "error": None} if endpoint == "score" else {"result": "Submission is valid."}
    assert response.get_json() == expected


def test_missing_or_oversized_upload_never_starts_grader(monkeypatch):
    monkeypatch.setattr(server, "MAX_SCORE_BYTES", 3)
    monkeypatch.setattr(server, "_run_bounded", lambda *args: pytest.fail("unexpected grader call"))
    with server.app.test_client() as client:
        response = client.post("/score")
    assert response.status_code == 200
    assert response.get_json()["score"] is None
    response = _post(b"1234")
    assert response.status_code == 200
    assert response.get_json() == {"score": None, "error": "submission too large"}
