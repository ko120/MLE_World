"""Exercise read-only submission validation and independent rejection reasons."""

import csv
import io
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "agents/aide/src"))
from aide import agent as agent_module
from aide import submission_review
from aide.agent import Agent, normalize_review_response
from aide.backend import compile_prompt_to_md
from aide.interpreter import ExecutionResult
from aide.journal import Journal, Node
from aide.utils.config import AgentConfig
from aide.utils.submission_checks import SubmissionChecks
from aide.utils.submission_validation import submission_sha256
from omegaconf import OmegaConf

SAMPLE = "id,target\n001,0\n002,0\n003,0\n"
VALID = "id,target\n001,0.2\n002,0.3\n003,0.4\n"


@pytest.fixture
def harness(tmp_path, monkeypatch):
    cfg = OmegaConf.load(Path(agent_module.__file__).parent / "utils/config.yaml")
    cfg.agent = OmegaConf.merge(OmegaConf.structured(AgentConfig), cfg.agent)
    cfg.agent.eval_metric = "binary-log-loss"
    cfg.agent.eval_lower_is_better = True
    cfg.agent.record_test_metric = False
    cfg.agent.check_data_leakage = False
    agent = object.__new__(Agent)
    agent.acfg = cfg.agent
    agent.cfg = SimpleNamespace(agent=cfg.agent, exec=cfg.exec, workspace_dir=tmp_path)
    agent.journal, agent.current_step = Journal(), 0
    agent.task_desc, agent.data_preview = "Predict the target for each test ID.", "public input"
    agent.deadline, agent.oracle = float("inf"), "sim"
    agent.search_policy = lambda: None
    agent.update_data_preview = lambda: None
    agent._draft_prompt = lambda: {}
    state = dict(
        is_bug=False,
        has_csv_submission=True,
        metric=0.8,
        lower_is_better=True,
        metric_trustworthy=True,
        summary="Execution succeeded.",
    )
    monkeypatch.setattr(agent_module, "query", lambda **kwargs: dict(state))
    (tmp_path / "input").mkdir()
    (tmp_path / "input/sample_submission.csv").write_text(SAMPLE)
    (tmp_path / "working").mkdir()
    monkeypatch.chdir(tmp_path)
    uploads = []

    def post(url, *, files, timeout):
        data = files["file"][1].read()
        uploads.append(data)
        columns = next(csv.reader(io.StringIO(data.decode())))
        valid = columns == ["id", "target"]
        return SimpleNamespace(
            status_code=200,
            json=lambda: {
                "result": (
                    "Submission is valid."
                    if valid
                    else "Submission invalid! Required columns: id,target."
                )
            },
        )

    server = Mock(side_effect=post)
    monkeypatch.setattr("requests.post", server)
    return agent, state, server, uploads


def _code(csv_text=VALID, path="submission/submission.csv"):
    return f"from pathlib import Path\nPath({path!r}).write_text({csv_text!r})\n"


def _run(agent, code, *, metric=0.8):
    agent.generate_code = lambda *args, **kwargs: ("plan", code, None)

    def execute(program, reset_session):
        try:
            exec(compile(program, "candidate.py", "exec"), {})
            return ExecutionResult([f"Validation loss: {metric}\n"], 0.1, None, real_won=True)
        except Exception as exc:
            return ExecutionResult(
                [str(exc)], 0.1, type(exc).__name__, {"args": [str(exc)]}, real_won=True
            )

    agent.step(execute)
    return agent.journal.nodes[-1]


def test_execution_flag_cannot_invent_a_csv_size_requirement(harness):
    agent, state, server, _ = harness
    state.update(
        is_bug=True,
        has_csv_submission=False,
        summary="Execution succeeded but the submission must contain 6577200 rows.",
    )
    node = _run(agent, _code())
    assert not node.is_buggy and node.submission_validation == "valid"
    assert node.failure_reasons == {}
    assert server.call_count == 1
    assert (agent.cfg.workspace_dir / "best_submission/submission.csv").read_text() == VALID
    assert "was not confirmed" in node.analysis


@pytest.mark.parametrize(
    "is_bug,trust,buggy", [("false", "true", False), ("False", "false", True), ("no", "yes", False)]
)
def test_recognized_string_verdicts_reach_the_correct_gate(harness, is_bug, trust, buggy):
    agent, state, _, _ = harness
    state.update(is_bug=is_bug, metric_trustworthy=trust, lower_is_better="false", metric="0.8")
    node = _run(agent, _code())
    assert node.is_buggy is buggy
    assert node.metric_trustworthy is (trust != "false")
    if not buggy:
        assert node.metric.maximize is False  # configured direction overrides the reviewer
    else:
        assert "metric" in node.failure_reasons and "untrustworthy" in node.analysis


@pytest.mark.parametrize(
    "metric", [True, False, float("inf"), float("-inf"), float("nan"), "NaN", "1e309", {}, None]
)
def test_invalid_metric_does_not_hide_independent_csv_evidence(harness, metric):
    agent, state, server, _ = harness
    state["metric"] = metric
    node = _run(agent, _code())
    assert node.is_buggy and node.metric.value is None
    assert "finite number" in node.analysis
    assert node.submission_validation == "valid" and server.call_count == 1
    assert "metric" in node.failure_reasons and "execution" not in node.failure_reasons


def test_ungrounded_score_and_malformed_trust_have_specific_reasons(harness):
    agent, state, _, _ = harness
    state["metric"] = 80.0
    node = _run(agent, _code())
    assert "80.0 was not found" in node.analysis
    state.update(metric=0.8, metric_trustworthy="maybe")
    node = _run(agent, _code())
    assert node.is_buggy and "reliability verdict is malformed" in node.analysis
    assert "unrecognized boolean" in node.analysis


def test_unknown_metric_direction_is_not_guessed(harness):
    agent, state, _, _ = harness
    agent.acfg.eval_lower_is_better = None
    state["lower_is_better"] = "maybe"
    node = _run(agent, _code())
    assert node.is_buggy and "Metric direction is unavailable" in node.analysis


def test_crash_and_missing_file_have_separate_debug_reasons(harness):
    agent, _, server, _ = harness
    node = _run(agent, "raise ValueError('cannot load data')")
    assert node.is_buggy
    assert set(node.failure_reasons) == {"execution", "metric", "submission"}
    assert "cannot load data" in node.analysis and "submission/submission.csv" in node.analysis
    server.assert_not_called()


def test_rejected_headers_are_not_rewritten_or_revalidated(harness):
    agent, _, server, _ = harness
    original = VALID.replace("target", "prediction")
    program = _code(original)
    node = _run(agent, program)
    assert node.is_buggy and node.submission_validation == "invalid"
    assert server.call_count == 1 and not node.submission_actions
    assert node.code == program
    assert (agent.cfg.workspace_dir / "submission/submission.csv").read_text() == original
    assert not (agent.cfg.workspace_dir / "best_submission/submission.csv").exists()


def test_server_outage_never_turns_local_success_into_validity(harness):
    agent, _, server, _ = harness
    server.side_effect = lambda *args, **kwargs: SimpleNamespace(status_code=503)
    node = _run(agent, _code())
    assert not node.is_buggy and node.submission_validation == "unknown"
    assert server.call_count == 1
    assert not (agent.cfg.workspace_dir / "best_submission/submission.csv").exists()


@pytest.mark.parametrize("http_status", [503, 504])
@pytest.mark.parametrize("trustworthy", [False, True])
def test_unverified_submission_followup_reaches_next_prompt(harness, http_status, trustworthy):
    agent, state, server, _ = harness
    state.update(
        metric_trustworthy=trustworthy,
        summary="Execution succeeded. " + "Detailed reviewer observation. " * 300,
    )
    server.side_effect = lambda *args, **kwargs: SimpleNamespace(status_code=http_status)
    program = _code()
    node = _run(agent, program)

    # Feedback must preserve the existing execution/metric classification and
    # keep an unverified file out of publication, including after serialization.
    restored = Node.from_dict(node.to_dict())
    assert restored.is_buggy is (not trustworthy)
    assert restored.metric.value == (0.8 if trustworthy else None)
    assert restored.submission_validation == "unknown"
    assert restored.code == program
    assert set(restored.failure_reasons) == (set() if trustworthy else {"metric"})
    assert server.call_count == 1
    assert not (agent.cfg.workspace_dir / "best_submission/submission.csv").exists()

    prompt = (
        agent._improve_prompt(restored)
        if trustworthy
        else agent._build_debug_prompt(restored.code, restored.term_out, restored.analysis)
    )
    rendered = compile_prompt_to_md(prompt)
    assert f"Validation server returned HTTP {http_status}" in rendered
    assert "Submission validation follow-up:" in rendered
    assert "does not establish a code or CSV-format bug" in rendered
    assert "preserve all required rows, IDs, and predictions" in rendered
    assert "retries validation without another training execution" in rendered


@pytest.mark.parametrize(
    "path", ["submission.csv", "working/submission.csv", "output/submission.csv"]
)
def test_alternate_csv_paths_are_not_recovered(harness, path):
    agent, _, server, _ = harness
    (agent.cfg.workspace_dir / path).parent.mkdir(parents=True, exist_ok=True)
    original = _code(path=path)
    node = _run(agent, original)
    assert node.is_buggy and not node.csv_exists and not node.submission_actions
    assert node.code == original
    assert (agent.cfg.workspace_dir / path).read_text() == VALID
    assert not (agent.cfg.workspace_dir / "submission/submission.csv").exists()
    server.assert_not_called()


def test_old_execution_result_without_snapshot_cannot_recover(harness):
    agent, _, server, _ = harness
    (agent.cfg.workspace_dir / "submission.csv").write_text(VALID)
    node = agent.parse_exec_result(
        Node(code="pass"), ExecutionResult(["Validation loss: 0.8"], 1, None, real_won=True)
    )
    assert node.is_buggy and not node.csv_exists
    server.assert_not_called()


def test_server_rejection_and_missing_predictions_reach_debug(harness):
    agent, _, server, _ = harness
    server.side_effect = lambda *args, **kwargs: SimpleNamespace(
        status_code=200,
        json=lambda: {"result": "Submission invalid! Identifier values do not match."},
    )
    node = _run(agent, _code("id,prediction\n001,0.2\n999,0.3\n"))
    assert node.is_buggy and "Execution succeeded." in node.analysis
    assert "missing=['target']" in node.analysis
    debug = compile_prompt_to_md(agent._build_debug_prompt(node.code, node.term_out, node.analysis))
    assert "Identifier values do not match" in debug and "missing=['target']" in debug
    assert "Submission validation follow-up:" in debug
    assert "The CSV was explicitly rejected" in debug
    assert "fix the specific submission errors reported above" in debug
    restored = Node.from_dict(node.to_dict())
    assert restored.failure_reasons == node.failure_reasons


@pytest.mark.parametrize(
    "value,reason",
    [
        ("NaN", "finite numeric"),
        ("inf", "finite numeric"),
        ("oops", "finite numeric"),
        ("-0.2", "outside [0, 1]"),
        ("1.1", "outside [0, 1]"),
    ],
)
def test_numeric_and_probability_checks_catch_contents_the_server_misses(harness, value, reason):
    agent, _, _, _ = harness
    node = _run(agent, _code(VALID.replace("0.2", value)))
    assert node.is_buggy and reason in node.analysis
    assert node.submission_validation == "invalid"


def test_constant_predictions_do_not_trigger_extra_work_or_rejection(harness):
    agent, _, _, _ = harness
    node = _run(agent, _code(SAMPLE))
    assert not node.is_buggy and node.submission_validation == "valid"
    assert "Prediction diagnostic" not in node.analysis and not node.failure_reasons


def test_auc_scores_are_not_assumed_to_be_probabilities(harness):
    agent, _, _, _ = harness
    agent.acfg.eval_metric = "auc-roc"
    node = _run(agent, _code(VALID.replace("0.2", "42")))
    assert not node.is_buggy


def test_local_checks_do_not_compare_public_row_coverage(harness):
    agent, _, _, _ = harness
    (agent.cfg.workspace_dir / "input/test.csv").write_text("id,feature\n007,x\n008,y\n")
    node = _run(agent, _code("id,target\n007,0.2\n008,0.3\n"))
    assert not node.is_buggy
    assert "Row count:" not in node.analysis and "ID coverage" not in node.analysis


def test_server_can_accept_an_alternate_layout(harness):
    agent, _, server, _ = harness
    server.side_effect = lambda *args, **kwargs: SimpleNamespace(
        status_code=200, json=lambda: {"result": "Submission is valid."}
    )
    node = _run(agent, _code("rec_id,species,probability\n1,1,0.2\n"))
    assert not node.is_buggy and not node.submission_actions
    assert "Sample-schema diagnostic" in node.analysis


@pytest.mark.parametrize("verdict", ["valid", "invalid", "unknown"])
def test_extra_columns_follow_server_verdict_and_reach_node(harness, verdict):
    agent, _, server, _ = harness
    server.side_effect = lambda *args, **kwargs: SimpleNamespace(
        status_code=503 if verdict == "unknown" else 200,
        json=lambda: {
            "result": (
                "Submission is valid."
                if verdict == "valid"
                else "Submission invalid! Extra columns are not accepted."
            )
        },
    )
    original = "id,target,notes\n001,0.2,a\n002,0.3,b\n003,0.4,c\n"
    node = _run(agent, _code(original))
    assert node.submission_validation == verdict
    assert node.is_buggy is (verdict == "invalid")
    assert node.metric.value == (None if verdict == "invalid" else 0.8)
    assert "Extra columns" in node.analysis and "notes" in node.analysis
    assert server.call_count == 1 and not node.submission_actions
    assert (agent.cfg.workspace_dir / "submission/submission.csv").read_text() == original


def test_extra_columns_do_not_hide_missing_columns_during_server_outage(harness):
    agent, _, server, _ = harness
    server.side_effect = lambda *args, **kwargs: SimpleNamespace(status_code=503)
    node = _run(agent, _code("id,notes\n001,a\n002,b\n003,c\n"))
    assert node.is_buggy and node.submission_validation == "invalid"
    assert "missing=['target']" in node.submission_validation_message
    assert "Extra columns" in node.analysis and "notes" in node.analysis
    assert server.call_count == 1 and not node.submission_actions


@pytest.mark.parametrize("verdict", ["valid", "unknown"])
@pytest.mark.parametrize("layout", ["missing_date", "different_date", "different_id", "missing_id"])
def test_optional_metadata_and_identity_follow_server_verdict(harness, verdict, layout):
    agent, _, server, _ = harness
    public = agent.cfg.workspace_dir / "input"
    if "date" in layout:
        (public / "sample_submission.csv").write_text(
            "Insult,Date,Comment\n0,,first\n0,,second\n0,,third\n"
        )
        (public / "test.csv").write_text("Date,Comment\n,first\n,second\n,third\n")
        original = (
            "Insult,Comment\n0.2,first\n0.3,second\n0.4,third\n"
            if layout == "missing_date"
            else "Insult,Date,Comment\n0.2,today,first\n0.3,today,second\n0.4,today,third\n"
        )
    else:
        (public / "sample_submission.csv").write_text(
            "id,formation_energy_ev_natom,bandgap_energy_ev\n1,0,0\n2,0,0\n3,0,0\n"
        )
        (public / "test.csv").write_text("id,feature\n1,x\n2,y\n3,z\n")
        agent.acfg.eval_metric = "mean-column-wise-rmsle"
        original = (
            "formation_energy_ev_natom,bandgap_energy_ev\n0.2,1\n0.3,2\n0.4,3\n"
            if layout == "missing_id"
            else "id,formation_energy_ev_natom,bandgap_energy_ev\n9,0.2,1\n8,0.3,2\n7,0.4,3\n"
        )
    server.side_effect = lambda *args, **kwargs: SimpleNamespace(
        status_code=503 if verdict == "unknown" else 200,
        json=lambda: {"result": "Submission is valid."},
    )
    node = _run(agent, _code(original))
    assert node.submission_validation == verdict and not node.is_buggy
    if layout.startswith("missing"):
        assert "Missing sample metadata" in node.analysis
    assert "ID coverage" not in node.analysis
    assert not node.submission_actions
    assert (agent.cfg.workspace_dir / "submission/submission.csv").read_text() == original


def test_server_identity_rejection_remains_invalid(harness):
    agent, _, server, _ = harness
    server.side_effect = lambda *args, **kwargs: SimpleNamespace(
        status_code=200,
        json=lambda: {"result": "Submission invalid! Identifier values do not match."},
    )
    node = _run(agent, _code(VALID.replace("001", "999")))
    assert node.is_buggy and node.submission_validation == "invalid"
    assert "Identifier values do not match" in node.analysis and not node.submission_actions


def test_row_counts_are_left_to_the_grader_during_server_outage(harness):
    agent, _, server, _ = harness
    server.side_effect = lambda *args, **kwargs: SimpleNamespace(status_code=503)
    node = _run(agent, _code("id,target\n001,0.2\n002,0.3\n"))
    assert not node.is_buggy and node.submission_validation == "unknown"
    assert "Row count:" not in node.submission_validation_message


def test_text_metadata_is_preserved_and_not_checked_as_numeric(harness):
    agent, _, server, uploads = harness
    public = agent.cfg.workspace_dir / "input"
    (public / "sample_submission.csv").write_text("Insult,Date,Comment\n0,,NA\n0,today,Hello\n")
    (public / "test.csv").write_text("Date,Comment\n,NA\ntoday,Hello\n")
    original = "Insult,Date,Comment\n0.2,,NA\n0.3,today,Hello\n"

    def post(*args, **kwargs):
        data = kwargs["files"]["file"][1].read()
        uploads.append(data)
        return SimpleNamespace(status_code=200, json=lambda: {"result": "Submission is valid."})

    server.side_effect = post
    node = _run(agent, _code(original))
    assert not node.is_buggy and uploads[-1].decode() == original
    assert server.call_count == 1 and not node.submission_actions
    assert "finite numeric" not in node.analysis


@pytest.mark.parametrize(
    "description",
    [
        "Probabilities must sum to one.",
        "Probabilities are not required to sum to one.",
        "Predict separate probabilities for each label.",
    ],
)
def test_probability_normalization_is_left_to_the_grader(tmp_path, description):
    (tmp_path / "sample_submission.csv").write_text("id,A,B\n1,0.5,0.5\n")
    submission = tmp_path / "submission.csv"
    submission.write_text("id,A,B\n1,0.4,0.4\n")
    checked = SubmissionChecks(tmp_path, "multi-class-log-loss", description).check(
        submission, time.monotonic() + 10
    )
    assert not checked.content_errors


def test_local_checks_leave_identifier_values_to_the_grader(tmp_path):
    (tmp_path / "sample_submission.csv").write_text(SAMPLE)
    submission = tmp_path / "submission.csv"
    checks = SubmissionChecks(tmp_path, "binary-log-loss")
    for csv_text in [
        "id,target\n003,0.4\n001,0.2\n002,0.3\n",
        "id,target\n001,0.2\n001,0.3\n003,0.4\n",
    ]:
        submission.write_text(csv_text)
        result = checks.check(submission, time.monotonic() + 10)
        assert not result.format_errors and not result.content_errors
        assert not any("ID coverage" in note for note in result.diagnostics)


def test_unknown_boolean_tokens_are_not_truthy():
    response = normalize_review_response(
        {"is_bug": "maybe", "metric_trustworthy": "maybe", "metric": 0.8}
    )
    assert response["is_bug"] is None and response["metric_trustworthy"] is None
    assert response["metric"] is None and response["review_issues"]


@pytest.mark.parametrize("kind", ["columns", "alternate_path"])
def test_real_best_of_n_keeps_original_code_and_reuses_validation(harness, kind):
    agent, _, server, _ = harness
    agent.cfg.exec.timeout = 30
    candidate = (
        _code(VALID.replace("target", "prediction"))
        if kind == "columns"
        else _code(path="submission.csv")
    )
    good_code = _code() + "print('Validation loss: 0.8')\n"
    candidates = [
        ("bad candidate", candidate + "print('Validation loss: 0.8')\n"),
        ("good candidate", good_code),
    ]
    original_candidates = list(candidates)
    log_dir = agent.cfg.workspace_dir / "logs"
    outcomes = agent._observe_real(candidates, log_dir)
    assert not outcomes[0].usable and outcomes[1].usable
    assert candidates == original_candidates
    calls = server.call_count
    assert calls == (2 if kind == "columns" else 1)
    _, code, result = agent._commit(candidates, outcomes, 1, log_dir)
    node = agent.parse_exec_result(Node(code=code), result)
    assert not node.is_buggy and not node.submission_actions
    assert node.code == good_code and server.call_count == calls
    assert (log_dir / "cand_1/submission.csv").read_text() == VALID
    assert (log_dir / "cand_1/solution.py").read_text() == good_code
    assert not (log_dir / "cand_0/executed_solution.py").exists()


@pytest.mark.parametrize("spent", [50.0, 80.0])
def test_server_rejection_gets_one_call_within_the_existing_budget(harness, monkeypatch, spent):
    agent, _, server, _ = harness
    clock, timeouts = [0.0], []
    monkeypatch.setattr("aide.agent.time.monotonic", lambda: clock[0])
    original_post = server.side_effect

    def post(*args, **kwargs):
        timeouts.append(kwargs["timeout"][1])
        answer = original_post(*args, **kwargs)
        clock[0] += spent
        return answer

    server.side_effect = post
    original = VALID.replace("target", "prediction")
    node = _run(agent, _code(original))
    assert server.call_count == 1 and timeouts[0] <= 75
    assert node.is_buggy and not node.submission_actions
    assert (agent.cfg.workspace_dir / "submission/submission.csv").read_text() == original


def test_ambiguous_class_names_are_never_guessed(harness):
    agent, _, server, _ = harness
    (agent.cfg.workspace_dir / "input/sample_submission.csv").write_text("id,A,B\n001,0.5,0.5\n")
    node = _run(agent, _code("id,X,Y\n001,0.2,0.8\n"))
    assert node.is_buggy and not node.submission_actions and server.call_count == 1


@pytest.mark.parametrize("valid_index", [True, False])
def test_extra_index_columns_are_never_dropped(harness, valid_index):
    agent, _, server, _ = harness
    csv_text = ",id,target\n0,001,0.2\n1,002,0.3\n2,003,0.4\n"
    if not valid_index:
        csv_text = csv_text.replace("1,002", "9,002")
    node = _run(agent, _code(csv_text))
    assert node.is_buggy and server.call_count == 1 and not node.submission_actions
    assert (agent.cfg.workspace_dir / "submission/submission.csv").read_text() == csv_text


def test_simulation_never_recovers_or_locally_validates_a_real_file(harness):
    agent, _, server, _ = harness
    (agent.cfg.workspace_dir / "submission.csv").write_text(VALID)
    node = agent.parse_exec_result(
        Node(code="pass"),
        ExecutionResult(
            ["Validation loss: 0.8"],
            1,
            None,
            predicted_submission_exists="true",
            submission_files_before={"submission.csv": None},
        ),
    )
    assert not node.is_buggy and not node.real_won
    assert not node.submission_actions and node.submission_validation is None
    server.assert_not_called()


def test_deadline_cannot_erase_a_known_content_rejection(harness):
    agent, _, server, _ = harness
    node = _run(agent, _code(VALID.replace("0.2", "NaN")))
    assert node.is_buggy
    result = agent._validate_submission_checks(
        agent.cfg.workspace_dir / "submission/submission.csv", timeout_s=0
    )
    assert result.status == "invalid" and "finite numeric" in result.message
    assert server.call_count == 1


def test_prefix_failure_survives_a_later_exhausted_budget(harness):
    agent, _, server, _ = harness
    sample = "id,target\n" + "".join(f"{i},0.5\n" for i in range(2049))
    (agent.cfg.workspace_dir / "input/sample_submission.csv").write_text(sample)
    node = _run(agent, _code(sample.replace("0,0.5", "0,NaN", 1)))
    assert node.is_buggy and node.submission_validation == "invalid"
    cached = agent._submission_content_cache[node.submission_sha256]
    assert not cached.complete and cached.cacheable and cached.content_errors
    result = agent._validate_submission_checks(
        agent.cfg.workspace_dir / "submission/submission.csv", timeout_s=0
    )
    assert result.status == "invalid" and "finite numeric" in result.message
    assert server.call_count == 1


def test_failed_schema_load_does_not_cache_an_unchecked_pass(harness, monkeypatch):
    from aide.utils import submission_checks as checks_module

    agent, _, server, _ = harness
    clock, failed = [0.0], []
    monkeypatch.setattr(checks_module.time, "monotonic", lambda: clock[0])
    original = checks_module._schema

    def schema(*args):
        if not failed:
            failed.append(True)
            raise TimeoutError("reference preview interrupted")
        return original(*args)

    monkeypatch.setattr(checks_module, "_schema", schema)
    server.side_effect = lambda *args, **kwargs: SimpleNamespace(status_code=503)
    path = agent.cfg.workspace_dir / "submission/submission.csv"
    path.parent.mkdir()
    path.write_text("id,target\n001,NaN\n002,0.3\n")
    first = agent._validate_submission_checks(path)
    assert first.status == "unknown"
    assert first.sha256 not in agent._submission_content_cache
    clock[0] = 2.0
    second = agent._validate_submission_checks(path)
    assert second.status == "invalid" and "finite numeric" in second.message


def test_bounded_prefix_is_cached_without_an_extra_full_file_hash(harness, monkeypatch):
    agent, _, server, _ = harness
    path = agent.cfg.workspace_dir / "submission/submission.csv"
    path.parent.mkdir()
    path.write_text("id,target\n" + "".join(f"{i},0.5\n" for i in range(2049)))

    def unexpected_hash(*args):
        pytest.fail("Local validation must not perform another whole-file hash")

    monkeypatch.setattr(submission_review, "_existing_submission_digest", unexpected_hash)
    first = agent._validate_submission_checks(path)
    cached = agent._submission_content_cache[first.sha256]
    assert first.status == "valid" and cached.cacheable and not cached.complete
    check = Mock(wraps=agent._submission_checks.check)
    monkeypatch.setattr(agent._submission_checks, "check", check)
    second = agent._validate_submission_checks(path)
    assert second.status == "valid" and second.sha256 == first.sha256
    check.assert_not_called()
    assert server.call_count == 1

    path.write_text(path.read_text().replace("0,0.5", "0,NaN", 1))
    changed = agent._validate_submission_checks(path)
    assert changed.status == "invalid" and "finite numeric" in changed.message
    assert changed.sha256 != first.sha256 and check.call_count == 1
    assert server.call_count == 2


def test_submission_changed_during_validation_is_not_certified(harness):
    agent, _, server, _ = harness
    path = agent.cfg.workspace_dir / "submission/submission.csv"
    path.parent.mkdir()
    path.write_text(VALID)
    original = server.side_effect

    def mutate(*args, **kwargs):
        response = original(*args, **kwargs)
        # Change size too: rapid equal-size writes can share filesystem timestamps.
        path.write_text(VALID.replace("0.2", "NaN") + "\n")
        return response

    server.side_effect = mutate
    result = agent._validate_submission_checks(path)
    assert result.status == "unknown" and "changed during validation" in result.message
    assert not agent._submission_content_cache and server.call_count == 1


def test_previous_submission_directory_symlink_cannot_supply_a_stale_csv(harness):
    agent, _, server, _ = harness
    root = agent.cfg.workspace_dir
    (root / "previous_output").mkdir()
    (root / "previous_output/submission.csv").write_text(VALID)
    (root / "submission").symlink_to("previous_output", target_is_directory=True)
    node = _run(agent, "pass")
    assert node.is_buggy and not node.csv_exists
    assert not (root / "submission").is_symlink()
    assert (root / "previous_output/submission.csv").read_text() == VALID
    server.assert_not_called()
