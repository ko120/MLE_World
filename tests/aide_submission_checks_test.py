"""Fast CSV checks bound both the input consumed and the conclusions drawn."""

import csv
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "agents/aide/src"))
from aide.utils import submission_checks as checks_module
from aide.utils.submission_checks import SubmissionChecks


@pytest.fixture
def files(tmp_path):
    public = tmp_path / "input"
    public.mkdir()
    sample = "id,target\n" + "".join(f"{i},0.5\n" for i in range(2049))
    (public / "sample_submission.csv").write_text(sample)
    (public / "test.csv").write_text("id,feature\n" + "".join(f"{i},x\n" for i in range(2049)))
    submission = tmp_path / "submission.csv"
    submission.write_text(sample)
    return public, submission


@pytest.mark.parametrize(
    "value,reason",
    [
        ("NaN", "finite numeric"),
        ("inf", "finite numeric"),
        ("[0.3]", "finite numeric"),
        ("1.5", "outside [0, 1]"),
    ],
)
def test_bad_prediction_stops_after_the_first_bad_row(files, monkeypatch, value, reason):
    public, submission = files
    checks = SubmissionChecks(public, "log-loss")
    checks.load(time.monotonic() + 10)
    submission.write_text(submission.read_text().replace("0,0.5", f"0,{value}", 1))
    original, read = checks_module._lines, []

    def lines(source, deadline):
        for line in original(source, deadline):
            read.append(line)
            yield line

    monkeypatch.setattr(checks_module, "_lines", lines)
    result = checks.check(submission, time.monotonic() + 10)
    assert len(result.content_errors) == 1 and reason in result.content_errors[0]
    assert not result.complete and result.cacheable
    assert len(read) == 3  # Header probe, header, and the first data row.


def test_large_submission_checks_only_its_prefix(files, monkeypatch):
    public, submission = files
    checks = SubmissionChecks(public, "log-loss")
    checks.load(time.monotonic() + 10)
    submission.write_text(submission.read_text().replace("2048,0.5", "2048,NaN"))
    original, read = checks_module._lines, []

    def lines(source, deadline):
        for line in original(source, deadline):
            read.append(line)
            yield line

    monkeypatch.setattr(checks_module, "_lines", lines)
    result = checks.check(submission, time.monotonic() + 100)
    assert not result.complete and result.cacheable
    assert not result.format_errors and not result.content_errors
    assert any("first 1024" in note for note in result.diagnostics)
    assert len(read) == 1026


def test_schema_reads_only_a_sample_prefix_and_the_test_header(files, monkeypatch):
    public, _ = files
    original, read = checks_module._lines, {}

    def lines(source, deadline):
        for line in original(source, deadline):
            read[source.name] = read.get(source.name, 0) + 1
            yield line

    monkeypatch.setattr(checks_module, "_lines", lines)
    checks = SubmissionChecks(public, "log-loss")
    checks.load(time.monotonic() + 10)
    assert checks.predictions == ["target"]
    assert read[str(public / "test.csv")] == 1
    assert read[str(public / "sample_submission.csv")] <= 131


@pytest.mark.parametrize("contents", ["x" * 4096, 'id,target\n1,"' + "\nx" * 4096])
def test_character_limit_bounds_huge_rows_and_multiline_cells(files, monkeypatch, contents):
    public, submission = files
    checks = SubmissionChecks(public, "log-loss")
    checks.load(time.monotonic() + 10)
    monkeypatch.setattr(checks_module, "_CSV_READ_CHARACTERS", 256)
    submission.write_text(contents)
    result = checks.check(submission, time.monotonic() + 10)
    assert not result.complete and result.cacheable
    assert not result.content_errors and not result.format_errors
    assert any("read limit (256 characters)" in note for note in result.diagnostics)


def test_blank_lines_cannot_bypass_the_read_limit(files, monkeypatch):
    public, submission = files
    checks = SubmissionChecks(public, "log-loss")
    checks.load(time.monotonic() + 10)
    monkeypatch.setattr(checks_module, "_CSV_READ_CHARACTERS", 256)
    submission.write_text("id,target\n" + "\n" * 4096)
    result = checks.check(submission, time.monotonic() + 10)
    assert not result.format_errors and not result.complete
    assert any("read limit" in note for note in result.diagnostics)


def test_local_budget_is_capped_even_if_caller_allows_longer(files, monkeypatch):
    public, submission = files
    checks = SubmissionChecks(public, "log-loss")
    checks.load(time.monotonic() + 10)
    clock = [0.0]
    monkeypatch.setattr(checks_module.time, "monotonic", lambda: clock[0])
    original = checks_module._lines

    def slow_lines(source, deadline):
        for line in original(source, deadline):
            clock[0] += 0.2
            yield line

    monkeypatch.setattr(checks_module, "_lines", slow_lines)
    result = checks.check(submission, 100.0)
    assert not result.complete and not result.cacheable
    assert not result.format_errors and not result.content_errors
    assert any("budget" in note for note in result.diagnostics)
    assert clock[0] <= 1.2


def test_reference_load_recovers_after_a_transient_failure(files, monkeypatch):
    public, _ = files
    clock, failed = [0.0], []
    monkeypatch.setattr(checks_module.time, "monotonic", lambda: clock[0])
    original = checks_module._schema

    def schema(*args):
        if not failed:
            failed.append(True)
            raise TimeoutError("reference preview interrupted")
        return original(*args)

    monkeypatch.setattr(checks_module, "_schema", schema)
    checks = SubmissionChecks(public, "log-loss")
    checks.load(10.0)
    assert checks.sample is None
    clock[0] = 2.0
    checks.load(10.0)
    assert checks.predictions == ["target"] and not checks.notes


def test_persistent_reference_failure_has_bounded_retries(files, monkeypatch):
    public, _ = files
    clock, calls = [0.0], []
    monkeypatch.setattr(checks_module.time, "monotonic", lambda: clock[0])

    def unavailable(*args):
        calls.append(True)
        raise OSError("temporarily unavailable")

    monkeypatch.setattr(checks_module, "_schema", unavailable)
    checks = SubmissionChecks(public, "log-loss")
    for tick in [0.0, 0.1, 2.0, 4.0, 6.0, 8.0]:
        clock[0] = tick
        checks.load(tick + 10)
    assert len(calls) == 3
    assert len(checks.notes) == 1 and "retry limit reached" in checks.notes[0]


def test_expired_first_budget_can_recover(files, monkeypatch):
    public, _ = files
    clock = [0.0]
    monkeypatch.setattr(checks_module.time, "monotonic", lambda: clock[0])
    checks = SubmissionChecks(public, "log-loss")
    checks.load(-1.0)
    clock[0] = 2.0
    checks.load(10.0)
    assert checks.predictions == ["target"] and not checks.notes


def test_large_text_cells_within_the_read_budget_are_supported(tmp_path):
    public = tmp_path / "input"
    public.mkdir()
    text = "a" * 200000
    (public / "sample_submission.csv").write_text(f"id,after\n1,{text}\n")
    submission = tmp_path / "submission.csv"
    submission.write_text(f"id,after\n1,{text}\n")
    result = SubmissionChecks(public, "accuracy").check(submission, time.monotonic() + 10)
    assert result.complete and result.cacheable
    assert not result.format_errors and not result.content_errors and not result.diagnostics


def test_parser_field_limit_is_reported_separately(files):
    public, submission = files
    checks = SubmissionChecks(public, "log-loss")
    checks.load(time.monotonic() + 10)
    submission.write_text("id,target\n1," + "x" * 200 + "\n")
    limit = csv.field_size_limit(64)
    try:
        result = checks.check(submission, time.monotonic() + 10)
    finally:
        csv.field_size_limit(limit)
    assert not result.complete and not result.cacheable
    assert not result.format_errors and not result.content_errors
    assert any("parser field-size limit (64 characters)" in note for note in result.diagnostics)


@pytest.mark.parametrize(
    "contents,reason",
    [
        ("id,target\n1,0.5,extra\n", "3 fields"),
        ("id,target\n", "no prediction rows"),
        ("id,prediction\n1,0.5\n", "missing=['target']"),
    ],
)
def test_obvious_format_errors_survive_the_small_scan_budget(files, contents, reason):
    public, submission = files
    submission.write_text(contents)
    result = SubmissionChecks(public, "log-loss").check(submission, time.monotonic() + 10)
    assert result.cacheable and any(reason in error for error in result.format_errors)
