"""Public reference discovery across task filenames and extracted layouts."""

import gzip
import sys
import time
import zipfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "agents/aide/src"))
from aide.utils import preproc_data
from aide.utils import submission_checks as checks_module
from aide.utils.submission_checks import SubmissionChecks, _references


SAMPLE = "id,target\n001,0\n002,0\n"
TEST = "id,feature\n001,a\n002,b\n"


def write_csv(root, name, text):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.name.lower().endswith(".gz"):
        with gzip.open(path, "wt", encoding="utf-8") as target:
            target.write(text)
    else:
        path.write_text(text)
    return path


@pytest.mark.parametrize(
    "sample_name,test_name",
    [
        ("sample_submission.csv", "test.csv"),
        ("sampleSubmission.csv", "testFeatures.csv"),
        ("sample_submission_null.csv", "test_data.csv"),
        ("en_sample_submission_2.csv", "en_test_2.csv"),
        ("ru_sample_submission_2.csv", "ru_test_2.csv"),
        ("datasetSampleSubmission2026.CSV.GZ", "datasetTestFeatures2026.CSV.GZ"),
        ("ENSampleSubmission2.csv", "ENTestData2.csv"),
        ("dataset-submission-sample-v2.csv", "X_test_2.csv"),
        ("dataset.samplesubmission.3.csv", "dataset.testdata.3.csv"),
    ],
)
def test_task_filename_variants_enable_schema_and_id_checks(tmp_path, sample_name, test_name):
    sample = write_csv(tmp_path, sample_name, SAMPLE)
    test = write_csv(tmp_path, test_name, TEST)
    assert _references(tmp_path) == ([sample], [test])
    submission = write_csv(tmp_path, "submission.csv", "id,target\n001,0.2\n999,0.3\n")
    checks = SubmissionChecks(tmp_path, "auc-roc")
    result = checks.check(submission, time.monotonic() + 10)
    assert checks.sample.path == sample
    assert checks.predictions == ["target"]
    assert not result.format_errors
    assert not any("ID coverage" in note for note in result.diagnostics)


@pytest.mark.parametrize("language", ["en", "ru"])
def test_startup_extraction_makes_text_normalization_references_usable(tmp_path, language):
    filenames = {
        f"{language}_train.csv": "sentence_id,token_id,class,before,after\n0,0,PLAIN,Hello,Hello\n",
        f"{language}_test_2.csv": "sentence_id,token_id,before\n0,0,Hello\n",
        f"{language}_sample_submission_2.csv": "id,after\n0_0,Hello\n",
    }
    for name, content in filenames.items():
        with zipfile.ZipFile(tmp_path / f"{name}.zip", "w") as archive:
            archive.writestr(name, content)
    preproc_data(tmp_path)
    assert not list(tmp_path.glob("*.zip"))
    sample = tmp_path / f"{language}_sample_submission_2.csv"
    assert _references(tmp_path) == ([sample], [tmp_path / f"{language}_test_2.csv"])
    submission = write_csv(tmp_path, "submission.csv", "id,after\n0_0,Hello\n")
    checks = SubmissionChecks(tmp_path, "accuracy")
    result = checks.check(submission, time.monotonic() + 10)
    assert checks.sample.path == sample
    assert checks.predictions == ["after"]
    assert result.complete and not result.format_errors and not result.content_errors


@pytest.mark.parametrize(
    "sample_name,test_name",
    [
        ("sample_submission.csv", "nested/test.csv"),
        ("nested/sample_submission.csv", "test.csv"),
        ("samples/sample_submission.csv", "data/nested/test.csv"),
        ("a/b/c/d/sample_submission.csv", "data/test.csv"),
    ],
)
def test_sample_and_test_can_be_at_different_depths(tmp_path, sample_name, test_name):
    sample = write_csv(tmp_path, sample_name, SAMPLE)
    test = write_csv(tmp_path, test_name, TEST)
    assert _references(tmp_path) == ([sample], [test])


def test_test_only_input_is_returned_and_missing_schema_is_reported(tmp_path):
    test = write_csv(tmp_path, "test.csv", TEST)
    assert _references(tmp_path) == ([], [test])
    checks = SubmissionChecks(tmp_path)
    checks.load(time.monotonic() + 10)
    assert checks.sample is None
    assert any("no sample-submission CSV" in note for note in checks.notes)


def test_nearest_references_take_precedence_over_nested_examples(tmp_path):
    sample = write_csv(tmp_path, "sample_submission.csv", SAMPLE)
    test = write_csv(tmp_path, "test.csv", TEST)
    write_csv(tmp_path, "examples/sample_submission.csv", "other,prediction\n1,0\n")
    write_csv(tmp_path, "examples/test.csv", "other\n1\n")
    assert _references(tmp_path) == ([sample], [test])


def test_equally_near_samples_remain_ambiguous(tmp_path):
    samples = [
        write_csv(tmp_path, f"{folder}/sample_submission.csv", SAMPLE) for folder in ["a", "b"]
    ]
    write_csv(tmp_path, "test.csv", TEST)
    assert _references(tmp_path)[0] == samples
    checks = SubmissionChecks(tmp_path)
    checks.load(time.monotonic() + 10)
    assert checks.sample is None and any("ambiguous" in note for note in checks.notes)


def test_equally_near_tests_do_not_create_a_false_coverage_requirement(tmp_path):
    write_csv(tmp_path, "sample_submission.csv", SAMPLE)
    write_csv(tmp_path, "test_1.csv", TEST)
    write_csv(tmp_path, "test_2.csv", "id,feature\n999,c\n")
    checks = SubmissionChecks(tmp_path)
    checks.load(time.monotonic() + 10)
    assert checks.sample is not None and checks.predictions == ["target"]
    assert any("test CSVs are ambiguous" in note for note in checks.notes)


def test_training_labels_predictions_and_substring_lookalikes_are_ignored(tmp_path):
    sample = write_csv(tmp_path, "sample_submission.csv", SAMPLE)
    test = write_csv(tmp_path, "test.csv", TEST)
    for name in [
        "train.csv",
        "test_labels.csv",
        "testWithSolutions.csv",
        "test_predictions.csv",
        "train_test.csv",
        "test_targets.csv",
        "contest.csv",
        "latest.csv",
        "resubmissionsample.csv",
    ]:
        write_csv(tmp_path, name, "id,label\n999,0\n")
    assert _references(tmp_path) == ([sample], [test])


@pytest.mark.parametrize("with_test", [True, False])
def test_supplied_directory_links_are_followed_without_cycles_or_duplicate_schemas(
    tmp_path, with_test
):
    public = tmp_path / "public"
    write_csv(public, "sample_submission.csv", SAMPLE)
    if with_test:
        write_csv(public, "test.csv", TEST)
    (public / "sample_submission_alias.csv").symlink_to(public / "sample_submission.csv")
    root = tmp_path / "input"
    root.mkdir()
    (root / "data").symlink_to(public, target_is_directory=True)
    (root / "same_data").symlink_to(public, target_is_directory=True)
    (public / "loop").symlink_to(root, target_is_directory=True)
    notes = []
    expected_tests = [root / "data/test.csv"] if with_test else []
    assert _references(root, notes=notes) == ([root / "data/sample_submission.csv"], expected_tests)
    assert not notes


def test_incomplete_sample_level_cannot_be_mistaken_for_an_unambiguous_schema(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(checks_module, "_REFERENCE_MAX_ENTRIES", 2)
    write_csv(tmp_path, "sample_submission.csv", SAMPLE)
    write_csv(tmp_path, "test.csv", TEST)
    write_csv(tmp_path, "sample_submission_extra.csv", SAMPLE)
    notes = []
    assert _references(tmp_path, notes=notes) == ([], [])
    assert any("incomplete" in note for note in notes)


def test_deep_scan_limit_keeps_a_complete_root_sample(tmp_path, monkeypatch):
    monkeypatch.setattr(checks_module, "_REFERENCE_MAX_ENTRIES", 2)
    sample = write_csv(tmp_path, "sample_submission.csv", SAMPLE)
    for index in range(3):
        write_csv(tmp_path, f"images/{index}.csv", "pixels\n0\n")
    notes = []
    assert _references(tmp_path, notes=notes) == ([sample], [])
    assert any("incomplete" in note for note in notes)


def test_directory_budget_does_not_select_one_of_unvisited_peers(tmp_path, monkeypatch):
    monkeypatch.setattr(checks_module, "_REFERENCE_MAX_DIRECTORIES", 2)
    write_csv(tmp_path, "a/sample_submission.csv", SAMPLE)
    write_csv(tmp_path, "b/sample_submission.csv", SAMPLE)
    notes = []
    assert _references(tmp_path, notes=notes) == ([], [])
    assert any("incomplete" in note for note in notes)


def test_reference_discovery_respects_the_validation_deadline(tmp_path):
    write_csv(tmp_path, "sample_submission.csv", SAMPLE)
    notes = []
    assert _references(tmp_path, deadline=time.monotonic() - 1, notes=notes) == ([], [])
    assert any("deadline" in note for note in notes)
