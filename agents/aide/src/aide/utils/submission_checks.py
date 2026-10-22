"""Fast checks of public CSV schemas and a bounded submission prefix.

Full row/ID coverage and alternate layouts are left to the grading server.
"""

import csv
import gzip
import math
import os
import re
import time
from collections import deque
from dataclasses import dataclass, field
from itertools import islice
from pathlib import Path

# Python's 128 Ki-character default is too small for some generated text cells.
# csv uses a process-wide limit; preserve a larger limit configured by the caller.
_CSV_FIELD_LIMIT = 8 * 1024 * 1024
csv.field_size_limit(max(csv.field_size_limit(), _CSV_FIELD_LIMIT))
_NUMERIC_SAMPLE_ROWS = 128
LOCAL_CHECK_SECONDS = 1.0
_SUBMISSION_SAMPLE_ROWS = 1024
_CSV_READ_CHARACTERS = 1024 * 1024
_REFERENCE_LOAD_ATTEMPTS = 3
_REFERENCE_RETRY_SECONDS = 1.0


def normalized_column(name):
    return "".join(char for char in name.casefold() if char.isalnum())


def _csv_issue(exc):
    if isinstance(exc, csv.Error) and "field larger than field limit" in str(exc):
        return (
            f"CSV parser field-size limit ({csv.field_size_limit()} characters) exceeded; "
            "this is a parser limit, not proof of malformed CSV"
        )
    return str(exc)


def _open_csv(path):
    opener = gzip.open if path.name.lower().endswith(".gz") else open
    return opener(path, "rt", encoding="utf-8-sig", newline="")


class _ScanLimit(Exception):
    pass


def _lines(source, deadline):
    """Bound physical input too, including blank lines and multiline CSV cells."""
    remaining = _CSV_READ_CHARACTERS
    while True:
        if time.monotonic() >= deadline:
            raise TimeoutError("local CSV check exceeded its budget")
        line = source.readline(remaining + 1)
        if not line:
            return
        if len(line) > remaining:
            raise _ScanLimit(f"local CSV read limit ({_CSV_READ_CHARACTERS} characters) reached")
        remaining -= len(line)
        yield line


def submission_signature(path):
    """Cheap file-version guard around validation; no extra whole-file hash."""
    try:
        info = Path(path).stat()
        return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)
    except OSError:
        return None


def _header(path, deadline=float("inf")):
    with _open_csv(path) as source:
        columns = next(csv.reader(_lines(source, deadline), strict=True), [])
    if not columns or len(columns) != len(set(columns)):
        raise ValueError(f"Missing or duplicate CSV header in {path.name}")
    return columns


def _reference_kind(name):
    """Recognize CSV roles with prefixes, versions, separators, and camel case."""
    if not name.lower().endswith((".csv", ".csv.gz")):
        return None
    stem = re.sub(r"\.csv(?:\.gz)?$", "", name, flags=re.I)
    stem = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", stem)
    stem = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", stem)
    tokens = re.findall(r"[a-z]+|[0-9]+", stem.casefold())
    pairs = {a + b for a, b in zip(tokens, tokens[1:])}
    if (set(tokens) | pairs) & {"samplesubmission", "submissionsample"}:
        return "sample"
    # A broader test-name match must not select public labels or predictions.
    excluded = {
        "train",
        "training",
        "label",
        "labels",
        "answer",
        "answers",
        "target",
        "targets",
        "solution",
        "solutions",
        "submission",
        "prediction",
        "predictions",
    }
    if set(tokens) & {"test", "testdata", "testfeatures"} and not set(tokens) & excluded:
        return "test"
    return None


_REFERENCE_MAX_DIRECTORIES = 32
_REFERENCE_MAX_ENTRIES = 512


def _references(root, *, deadline=float("inf"), notes=None):
    """Find each role at its nearest depth after startup archive extraction.

    Scan complete breadth-first levels so equally near candidates remain
    ambiguous. Follow supplied directory links once, and retain references
    from complete shallower levels if a deeper search reaches its budget.
    """
    directories = deque([(Path(root), 0)])
    found = {"sample": [], "test": []}
    depths = {"sample": None, "test": None}
    seen_directories, seen_files = set(), set()
    incomplete_depth = float("inf")
    while directories:
        directory, depth = directories.popleft()
        if all(value is not None for value in depths.values()) and depth > max(depths.values()):
            break
        if len(seen_directories) >= _REFERENCE_MAX_DIRECTORIES or time.monotonic() >= deadline:
            incomplete_depth = min(incomplete_depth, depth)
            break
        try:
            info = directory.stat()
            identity = (info.st_dev, info.st_ino)
            if identity in seen_directories:
                continue
            seen_directories.add(identity)
            with os.scandir(directory) as source:
                entries = list(islice(source, _REFERENCE_MAX_ENTRIES + 1))
            if len(entries) > _REFERENCE_MAX_ENTRIES:
                incomplete_depth = min(incomplete_depth, depth)
                entries = entries[:_REFERENCE_MAX_ENTRIES]
            for number, entry in enumerate(sorted(entries, key=lambda item: item.name)):
                if number % 128 == 0 and time.monotonic() >= deadline:
                    incomplete_depth = min(incomplete_depth, depth)
                    break
                try:
                    if entry.is_dir():
                        directories.append((Path(entry.path), depth + 1))
                        continue
                    kind = _reference_kind(entry.name)
                    if kind is None or not entry.is_file():
                        continue
                    if depths[kind] is not None and depth > depths[kind]:
                        continue
                    info = entry.stat()
                    identity = (kind, info.st_dev, info.st_ino)
                    if identity in seen_files:
                        continue
                    seen_files.add(identity)
                    depths[kind] = depth
                    found[kind].append(Path(entry.path))
                except OSError:
                    incomplete_depth = min(incomplete_depth, depth)
        except OSError:
            incomplete_depth = min(incomplete_depth, depth)
    if incomplete_depth != float("inf"):
        for kind, depth in depths.items():
            if depth is not None and depth >= incomplete_depth:
                found[kind] = []
        if notes is not None:
            notes.append(
                "Public reference discovery was incomplete (scan limit, deadline, or unreadable "
                "directory); only references from fully searched shallower levels are used."
            )
    return found["sample"], found["test"]


def _number(value):
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (ValueError, OverflowError):
        return None


@dataclass
class Reference:
    path: Path
    columns: list[str]
    ids: list[str]
    numeric: set[str] = field(default_factory=set)


def _schema(path, ids, deadline):
    columns = _header(path, deadline)
    profile = Reference(path, columns, ids, numeric=set(columns))
    saw_row = False
    with _open_csv(path) as source:
        reader = csv.reader(_lines(source, deadline), strict=True)
        next(reader)
        for row in islice(reader, _NUMERIC_SAMPLE_ROWS):
            if not row:
                continue
            if len(row) != len(columns):
                raise ValueError(f"Malformed reference row in {path.name}")
            saw_row = True
            profile.numeric.difference_update(
                c for c, value in zip(columns, row) if _number(value) is None
            )
    if not saw_row:
        profile.numeric.clear()
    return profile


@dataclass
class ContentCheck:
    format_errors: list[str] = field(default_factory=list)
    content_errors: list[str] = field(default_factory=list)
    diagnostics: list[str] = field(default_factory=list)
    complete: bool = False
    # A completed prefix check can be reused without claiming full-file coverage.
    cacheable: bool = False


class SubmissionChecks:
    """Cache a small public schema and check only the submission's opening rows."""

    def __init__(self, input_dir, metric=None, task_desc=""):
        self.input_dir = Path(input_dir)
        self.metric = normalized_column(metric or "")
        self.task_desc = task_desc or ""
        self.sample = None
        self.predictions = []
        self.notes = []
        self._loaded = False
        self._load_attempts = 0
        self._retry_at = 0.0
        self._load_note = None

    def load(self, deadline):
        if self._loaded:
            return
        now = time.monotonic()
        deadline = min(deadline, now + LOCAL_CHECK_SECONDS)
        if self._load_attempts >= _REFERENCE_LOAD_ATTEMPTS or now < self._retry_at:
            return
        if now >= deadline:
            self._note_load_failure("no time remains to load public references")
            return
        self._load_attempts += 1
        if self._load_note is not None:
            self.notes.remove(self._load_note)
            self._load_note = None
        try:
            if self.sample is None:
                discovery_notes = []
                samples, tests = _references(
                    self.input_dir, deadline=deadline, notes=discovery_notes
                )
                if len(samples) != 1:
                    if discovery_notes:
                        raise TimeoutError("public reference discovery was incomplete")
                    self.notes.append(
                        "Local schema checks skipped: multiple supplied sample-submission CSVs are ambiguous."
                        if samples
                        else "Local schema checks skipped: no sample-submission CSV was found after preprocessing."
                    )
                    self._loaded = True
                    return
                columns = _header(samples[0], deadline)
                test = tests[0] if len(tests) == 1 else None
                test_columns = _header(test, deadline) if test else []
                shared = [c for c in columns if c in test_columns]
                ids = [
                    c
                    for c in columns
                    if normalized_column(c)
                    in (
                        "id",
                        "image",
                        "filename",
                        "imageid",
                        "rowid",
                        "recordid",
                        "recid",
                        "sampleid",
                    )
                    or c.endswith(("Id", "ID"))
                    or re.search(r"[_ -]id$", c, re.I)
                ]
                # These columns help exclude metadata from predictions. They
                # do not establish which identity fields the grader requires.
                ids = ids or shared
                self.sample = _schema(samples[0], ids, deadline)
                self.predictions = [c for c in columns if c not in ids and c not in shared]
                self.notes.extend(discovery_notes)
                if len(tests) > 1:
                    self.notes.append(
                        "Multiple public test CSVs are ambiguous; shared-column inference skipped."
                    )
            self._loaded = True
        except (OSError, ValueError, csv.Error, UnicodeError, TimeoutError, _ScanLimit) as exc:
            self._note_load_failure(_csv_issue(exc))

    def _note_load_failure(self, reason):
        if self._load_note is not None:
            self.notes.remove(self._load_note)
        self._retry_at = time.monotonic() + _REFERENCE_RETRY_SECONDS
        retry = (
            "retry limit reached"
            if self._load_attempts >= _REFERENCE_LOAD_ATTEMPTS
            else "a later validation may retry"
        )
        self._load_note = f"Local reference checks incomplete: {reason}; {retry}."
        self.notes.append(self._load_note)

    def check(self, path, deadline):
        deadline = min(deadline, time.monotonic() + LOCAL_CHECK_SECONDS)
        self.load(deadline)
        result = ContentCheck(diagnostics=list(self.notes))
        sample = self.sample
        try:
            columns = _header(path, deadline)
            # Extra columns may be accepted by the grader; report them without rejecting locally.
            if sample:
                missing = [c for c in self.predictions if c not in columns]
                metadata = [
                    c for c in sample.columns if c not in self.predictions and c not in columns
                ]
                extra = [c for c in columns if c not in sample.columns]
                if missing:
                    result.format_errors.append(
                        f"Columns differ from {sample.path.name}: missing={missing}; "
                        f"expected={sample.columns}, actual={columns}."
                    )
                if metadata:
                    result.diagnostics.append(
                        f"Missing sample metadata columns: {metadata}; "
                        "the grading server determines whether they are required."
                    )
                if extra:
                    result.diagnostics.append(
                        f"Extra columns beyond {sample.path.name}: {extra}; "
                        "acceptance is determined by the grading server."
                    )
                if missing:
                    result.cacheable = self._loaded
                    return result
            numeric_metric = any(
                word in self.metric
                for word in (
                    "auc",
                    "logloss",
                    "logarithmicloss",
                    "rmse",
                    "rmsle",
                    "mse",
                    "msle",
                    "mae",
                    "meansquared",
                    "meanabsolute",
                    "rootmean",
                    "r2",
                )
            )
            numeric = (
                [
                    c
                    for c in self.predictions
                    if c in columns
                    and (
                        (numeric_metric and c in sample.numeric) or "probab" in normalized_column(c)
                    )
                ]
                if sample
                else []
            )
            probability = {
                c
                for c in numeric
                if "logloss" in self.metric
                or "logarithmicloss" in self.metric
                or "probab" in normalized_column(c)
            }
            # AUC does not imply a probability range. Row-sum requirements can
            # differ from task prose; the grading server decides normalization.
            numeric_indices = {c: columns.index(c) for c in numeric}
            count = 0
            with _open_csv(path) as source:
                reader = csv.reader(_lines(source, deadline), strict=True)
                next(reader)
                for record, row in enumerate(reader, 1):
                    if not row:
                        if record == _SUBMISSION_SAMPLE_ROWS:
                            raise _ScanLimit("local CSV row limit reached")
                        continue
                    count += 1
                    if len(row) != len(columns):
                        result.format_errors.append(
                            f"CSV row {count + 1} has {len(row)} fields; expected {len(columns)}."
                        )
                        result.cacheable = self._loaded
                        return result
                    for c, i in numeric_indices.items():
                        value = row[i]
                        number = _number(value)
                        if number is None:
                            result.content_errors.append(
                                f"CSV row {count + 1}, column {c!r}: expected a finite "
                                f"numeric prediction, got {value[:100]!r}."
                            )
                        elif c in probability and not 0 <= number <= 1:
                            result.content_errors.append(
                                f"CSV row {count + 1}, column {c!r}: probability "
                                f"{value[:100]!r} is outside [0, 1]."
                            )
                        if result.content_errors:
                            result.cacheable = self._loaded
                            return result
                    if record == _SUBMISSION_SAMPLE_ROWS:
                        raise _ScanLimit(
                            f"local check inspected only the first {count} prediction rows"
                        )
            if count == 0:
                result.format_errors.append("Submission CSV has a header but no prediction rows.")
            result.complete = self._loaded
            result.cacheable = self._loaded
        except _ScanLimit as exc:
            result.diagnostics.append(
                f"{exc}; remaining contents and coverage are left to the grader."
            )
            result.cacheable = self._loaded
        except (OSError, ValueError, csv.Error, UnicodeError, TimeoutError) as exc:
            result.diagnostics.append(f"Local content check incomplete: {_csv_issue(exc)}.")
        return result
