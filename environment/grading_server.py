import json
import logging
import math
import multiprocessing
import os
import tempfile
import time
from pathlib import Path

from flask import Flask, jsonify, request
from werkzeug.serving import make_server

from mlebench.grade import validate_submission
from mlebench.grade_helpers import InvalidSubmissionError
from mlebench.registry import registry
from mlebench.utils import load_answers, read_csv

app = Flask(__name__)

# Enroot captures stdout/stderr in grading/grading_server.log, outside the
# agent's log mount. Docker captures this process through its entrypoint.
# A stderr handler on the root logger is the whole delivery mechanism, and
# forked /score children inherit it.
# NOTE: mlebench.utils calls logging.basicConfig at import (utils.py:350), and
# the imports above pull it in, so in practice ITS format wins and these lines
# are a no-op — deliberately kept as the fallback for anyone importing this
# module without mlebench having configured the root logger first.
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("grading_server")

PRIVATE_DATA_DIR = "/private/data"
COMPETITION_ID = os.getenv(
    "COMPETITION_ID"
)  # This is populated for us at container runtime

# Docker uses its configured port. Enroot's host wrapper owns a listening
# socket and passes its fd to every replacement grader, keeping the URL stable.
PORT = int(os.getenv("GRADING_SERVER_PORT", "5000"))

# Load data and grade inside a child so hangs and grader crashes are bounded.
# Reject oversized submissions before parsing them.
MAX_SCORE_BYTES = 500 * 1024 * 1024
SCORE_TIMEOUT_SECS = 60
VALIDATE_TIMEOUT_SECS = 60
MAX_VALIDATE_BYTES = 500 * 1024 * 1024


def run_validation(submission: Path) -> str:
    new_registry = registry.set_data_dir(Path(PRIVATE_DATA_DIR))
    competition = new_registry.get_competition(COMPETITION_ID)
    is_valid, message = validate_submission(submission, competition)
    return message


def _validation_child(submission: Path, result_pipe) -> None:
    """Only format verdicts cross this boundary; private scores stay inside."""
    try:
        result_pipe.send(({"result": run_validation(submission)[:4096]}, 200))
    except Exception:
        logger.exception("submission validator failed")
        result_pipe.send(({"error": "Submission validation failed on the server."}, 500))
    finally:
        result_pipe.close()


def _run_bounded(target, submission: Path, timeout_s: float):
    """Receive the child's result before joining, then reap it on every path."""
    context = multiprocessing.get_context("fork")
    receive, send = context.Pipe(duplex=False)
    child = context.Process(target=target, args=(submission, send))
    deadline = time.monotonic() + timeout_s
    try:
        child.start()
        send.close()
        if not receive.poll(max(0.0, deadline - time.monotonic())):
            raise TimeoutError("Grading deadline exceeded")
        try:
            return receive.recv()
        except EOFError:
            child.join(timeout=0.1)
            raise ChildProcessError(
                f"grader exited without a result (exit {child.exitcode})"
            ) from None
    finally:
        send.close()
        receive.close()
        if child.pid is not None:
            child.join(timeout=0.1)
            if child.is_alive():
                child.kill()
                child.join()
            child.close()


def _run_validation_bounded(submission: Path) -> tuple[dict, int]:
    try:
        return _run_bounded(_validation_child, submission, VALIDATE_TIMEOUT_SECS)
    except TimeoutError:
        return {"error": "Submission validation timed out."}, 504
    except ChildProcessError:
        return {"error": "Submission validator exited without a verdict."}, 500


@app.route("/validate", methods=["POST"])
def validate():
    if "file" not in request.files:
        return jsonify({"error": "Missing submission file."}), 400
    # Concurrent requests must not overwrite each other's upload. Removal
    # happens after the grader exits, including timeout and exception paths.
    try:
        with tempfile.TemporaryDirectory(prefix="submission-validation-") as directory:
            submission_path = Path(directory) / "submission.csv"
            request.files["file"].save(submission_path)
            if submission_path.stat().st_size > MAX_VALIDATE_BYTES:
                return jsonify({"error": "Submission exceeds the validation size limit."}), 413
            body, status = _run_validation_bounded(submission_path)
            return jsonify(body), status
    except Exception:
        logger.exception("/validate failed")
        return jsonify({"error": "Submission validation failed on the server."}), 500


def _load_and_grade(submission: Path):
    """Resolve and load inside the timed child, including CSV or JSONL answers."""
    new_registry = registry.set_data_dir(Path(PRIVATE_DATA_DIR))
    competition = new_registry.get_competition(COMPETITION_ID)
    answers = load_answers(competition.answers)
    sub_df = read_csv(submission)
    # The Grader wrapper swallows exceptions. Call grade_fn directly to retain
    # the failure reason, while applying the same rounding as offline grading.
    score = competition.grader.grade_fn(sub_df, answers)
    return round(score, 5) if score is not None else None


def _grade_child(submission: Path, result_pipe) -> None:
    """Report a finite score or the reason grading failed."""
    try:
        score = _load_and_grade(submission)
        score = float(score) if score is not None else None
        if score is not None and not math.isfinite(score):
            raise ValueError("Grader returned a non-finite score")
        result = {"score": score, "error": None}
    except InvalidSubmissionError as e:
        # The SUBMISSION is at fault (wrong length/columns/ids/range). Expected
        # traffic, not a defect — warn without a traceback.
        logger.warning(f"invalid submission {submission}: {e}")
        result = {"score": None, "error": f"invalid submission: {e}"}
    except Exception as e:
        # A GRADER bug, or a grade_fn that a bad-but-not-rejected submission
        # walked into (e.g. sklearn "Input contains NaN"). Traceback wanted.
        logger.exception(f"grader raised on {submission}")
        result = {"score": None, "error": f"{type(e).__name__}: {e}"}
    try:
        result_pipe.send(result)
    finally:
        result_pipe.close()


@app.route("/score", methods=["POST"])
def score():
    """Grade a submission against the private test answers.

    Measurement-only endpoint for run instrumentation (per-node test metric,
    see aide journal.Node.test_metric). Always answers 200 {"score":
    float|null} and never raises past Flask: an ungradeable submission is a
    null score, not an error the agent should react to.
    """

    def _null(reason: str):
        """Fail-open answer, but recorded. Every null grade now costs one log
        line, so the campaign-wide test-metric coverage rate is diagnosable
        from grading_server.log instead of being a silent hole."""
        logger.warning(f"/score returning null: {reason}")
        return jsonify({"score": None, "error": reason})

    try:
        submission_file = request.files["file"]
        with tempfile.TemporaryDirectory(prefix="submission-score-") as directory:
            submission_path = Path(directory) / "submission.csv"
            submission_file.save(submission_path)
            if submission_path.stat().st_size > MAX_SCORE_BYTES:
                return _null("submission too large")
            try:
                result = _run_bounded(_grade_child, submission_path, SCORE_TIMEOUT_SECS)
            except TimeoutError:
                return _null(f"grading timed out after {SCORE_TIMEOUT_SECS}s")
            except ChildProcessError as e:
                return _null(str(e))
        # _grade_child already logged its own traceback; don't double-log.
        if result.get("score") is None and result.get("error") is None:
            logger.warning("/score returning null: grader returned no score")
        return jsonify(result)
    except Exception as e:
        logger.exception("/score failed in the parent")
        return _null(f"{type(e).__name__}: {e}")


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "running", "run_token": os.environ.get("AIDE_RUN_TOKEN", "")}), 200


def serve_with_readiness(ready_file: Path) -> None:
    """Publish an atomic readiness record for a socket this server still owns."""
    run_token = os.environ["AIDE_RUN_TOKEN"]
    if not run_token:
        raise RuntimeError("Grading-server readiness requires a run token")
    inherited_fd = os.environ.get("GRADING_SERVER_FD")
    fd = int(inherited_fd) if inherited_fd else None
    # Werkzeug duplicates the inherited descriptor. The wrapper retains its
    # copy while this grader owns another; restarting never releases the port.
    with make_server("127.0.0.1", 0, app, threaded=True, fd=fd) as server:
        if fd is not None:
            os.close(fd)
        port = server.socket.getsockname()[1]
        ready_file.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w", dir=ready_file.parent, prefix=f".{ready_file.name}.", delete=False
        ) as temporary:
            temporary_path = Path(temporary.name)
            try:
                json.dump({"port": port, "run_token": run_token}, temporary)
                temporary.close()
                temporary_path.replace(ready_file)
            finally:
                temporary_path.unlink(missing_ok=True)
        logger.info("Grading server listening on 127.0.0.1:%s", port)
        server.serve_forever()


if __name__ == "__main__":
    if ready_file := os.environ.get("GRADING_SERVER_READY_FILE"):
        serve_with_readiness(Path(ready_file))
    else:
        app.run(host="0.0.0.0", port=PORT)
