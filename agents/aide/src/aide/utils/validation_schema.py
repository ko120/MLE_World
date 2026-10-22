"""Bounded, cached metadata from the agent's supplied public input directory.

CSV headers establish column availability without reading all rows. JSON key
absence and inferred dtypes require a complete bounded read. A prefix never
establishes that a JSON key is absent. ID examples are descriptive samples only.
"""

from collections import Counter, deque
from contextlib import contextmanager
import copy
import csv
import gzip
import io
from itertools import islice
import json
import os
from pathlib import Path
import re
import time
import zipfile

from .submission_checks import normalized_column, submission_signature

MAX_BYTES = 16 * 1024 * 1024
MAX_HEADER_BYTES = 256 * 1024
SCHEMA_SECONDS = 5.0


def _format(name):
    return re.sub(r"\.(gz|zip)$", "", name.lower())


def _role(name):
    name = _format(name)
    if not name.endswith((".csv", ".tsv", ".json", ".jsonl", ".ndjson")):
        return None
    tokens = re.findall(r"[a-z]+", re.sub(r"([a-z])([A-Z])", r"\1_\2", name).lower())
    joined = "".join(tokens)
    if "samplesubmission" in joined or "submissionsample" in joined:
        return "sample"
    if set(tokens) & {
        "private",
        "solution",
        "solutions",
        "answer",
        "answers",
        "prediction",
        "submission",
    }:
        return None
    if set(tokens) & {"test", "testdata", "testfeatures"}:
        return (
            "test" if not set(tokens) & {"train", "label", "labels", "target", "targets"} else None
        )
    if set(tokens) & {"train", "training", "labels"}:
        return "train"
    return None


@contextmanager
def _open(path):
    if path.name.lower().endswith(".zip"):
        with zipfile.ZipFile(path) as archive:
            members = [
                m for m in archive.infolist() if not m.is_dir() and _role(Path(m.filename).name)
            ]
            if len(members) != 1:
                raise ValueError("archive does not contain exactly one recognized data file")
            with archive.open(members[0]) as source:
                yield source, members[0].file_size
    elif path.name.lower().endswith(".gz"):
        with gzip.open(path, "rb") as source:
            yield source, None
    else:
        with path.open("rb") as source:
            yield source, path.stat().st_size


def _read(source, limit, deadline):
    chunks = []
    size = 0
    while size <= limit:
        if time.monotonic() >= deadline:
            raise TimeoutError("public schema read exceeded its allowance")
        # Public datasets commonly live on NFS; tiny read calls can consume
        # the whole allowance before a modest JSON file reaches EOF.
        chunk = source.read(min(1024 * 1024, limit + 1 - size))
        if not chunk:
            return b"".join(chunks), True
        chunks.append(chunk)
        size += len(chunk)
    return b"".join(chunks)[:limit], False


def _metadata(path, role, deadline):
    record = {"file": path.name, "role": role}
    if role == "sample":
        record["dtype_semantics"] = "placeholder_values_not_output_constraints"
    is_csv = _format(path.name).endswith((".csv", ".tsv"))
    with _open(path) as (source, size):
        # Large tabular inputs need only their header. Large JSON inputs have
        # no complete key schema and cannot support an absence claim.
        if size is not None and size > MAX_BYTES and not is_csv:
            raise ValueError("JSON exceeds the complete-schema read limit")
        limit = MAX_HEADER_BYTES if size is not None and size > MAX_BYTES else MAX_BYTES
        raw, complete = _read(source, limit, deadline)
        text = raw.decode("utf-8-sig", errors="strict" if complete else "ignore")
    dtypes = {}
    if is_csv:
        delimiter = "\t" if _format(path.name).endswith(".tsv") else ","
        reader = csv.reader(io.StringIO(text), delimiter=delimiter, strict=True)
        columns = next(reader, [])
        if not columns or len(set(columns)) != len(columns):
            raise ValueError("missing or duplicate header")
        if not complete and "\n" not in text:
            raise ValueError("header exceeds the bounded read")
        record["columns"] = columns
        if complete:
            if time.monotonic() >= deadline:
                return record, dtypes
            import pandas as pd

            frame = pd.read_csv(io.StringIO(text), sep=delimiter)
            record["rows"] = len(frame)
            dtypes = {str(k): str(v) for k, v in frame.dtypes.items()}
            if role == "sample":
                # A template can put a prediction first. Its position does not
                # establish an ID column (nor do placeholder zeros fix a dtype).
                identifiers = [c for c in columns if normalized_column(c) in {
                    "id", "image", "filename", "rowid", "imageid", "key",
                }]
                if len(identifiers) == 1:
                    record["id_column"] = identifiers[0]
                    record["id_examples"] = [str(v) for v in frame[identifiers[0]].head(3)]
                dtypes = {}
    else:
        if not complete:
            raise ValueError("JSON key availability requires a complete read")
        records = (
            [json.loads(line) for line in text.splitlines() if line.strip()]
            if _format(path.name).endswith((".jsonl", ".ndjson"))
            else json.loads(text)
        )
        if not isinstance(records, list) or not all(isinstance(r, dict) for r in records):
            raise ValueError("only JSON record arrays or JSONL have supported key schemas")
        counts = Counter(key for row in records for key in row)
        record.update(rows=len(records), key_presence_counts=dict(counts))
        if time.monotonic() < deadline:
            import pandas as pd

            dtypes = {str(k): str(v) for k, v in pd.DataFrame.from_records(records).dtypes.items()}
    return record, dtypes


class PublicValidationSchema:
    def __init__(self, input_dir):
        self.input_dir = Path(input_dir)
        self.cached = None
        self.signatures = {}

    def load(self, deadline):
        deadline = min(deadline, time.monotonic() + SCHEMA_SECONDS)
        if self.cached is not None and all(
            submission_signature(p) == sig for p, sig in self.signatures.items()
        ):
            return copy.deepcopy(self.cached)
        result = {
            "files": [],
            "column_dtypes": {},
            "diagnostics": [],
            "designated_raw_target_fields": [],
            "public_submission_columns": [],
        }
        found = {"train": [], "test": [], "sample": []}
        pending = deque([(self.input_dir, 0)])
        seen = set()
        directories = []
        incomplete_depth = float("inf")
        while pending and len(seen) < 16:
            directory, depth = pending.popleft()
            if time.monotonic() >= deadline:
                incomplete_depth = min(incomplete_depth, depth)
                break
            if all(found.values()) and depth > max(v[0][0] for v in found.values()):
                break
            try:
                st = directory.stat()
                identity = (st.st_dev, st.st_ino)
                if identity in seen:
                    continue
                seen.add(identity)
                directories.append(directory)
                with os.scandir(directory) as scan:
                    entries = list(islice(scan, 513))
                if len(entries) > 512:
                    # Do not certify uniqueness from an incomplete directory.
                    incomplete_depth = min(incomplete_depth, depth)
                    continue
                for entry in sorted(entries, key=lambda e: e.name):
                    if (
                        entry.is_dir()
                        and depth < 2
                        and entry.name.lower()
                        not in {
                            "private",
                            "solutions",
                            "answers",
                            "submission",
                            "__pycache__",
                        }
                    ):
                        pending.append((Path(entry.path), depth + 1))
                    elif entry.is_file():
                        role = _role(entry.name)
                        if role and (not found[role] or depth == found[role][0][0]):
                            found[role].append((depth, Path(entry.path)))
            except OSError as exc:
                result["diagnostics"].append(f"Public discovery: {exc}")
                incomplete_depth = min(incomplete_depth, depth)
        if pending and len(seen) >= 16:
            incomplete_depth = min(incomplete_depth, pending[0][1])
        if incomplete_depth != float("inf"):
            result["diagnostics"].append(
                "Deeper public discovery incomplete; only complete shallower levels are used."
            )
        signatures = {p: submission_signature(p) for p in directories}
        for role in ("train", "test", "sample"):
            # Prefer an extracted file to its compressed copy at the same depth.
            unique = {}
            for depth, path in found[role]:
                if depth >= incomplete_depth:
                    continue
                key = (path.parent, _format(path.name))
                previous = unique.get(key)
                if previous is None or len(path.suffixes) < len(previous.suffixes):
                    unique[key] = path
            if len(unique) != 1:
                result["diagnostics"].append(
                    f"{role}: expected one public file, found {len(unique)}"
                )
                continue
            path = next(iter(unique.values()))
            before = submission_signature(path)
            try:
                record, dtypes = _metadata(path, role, deadline)
                if before is None or submission_signature(path) != before:
                    raise ValueError("public file changed during schema read")
                signatures[path] = before
                result["files"].append(record)
                if dtypes and role != "sample":
                    result["column_dtypes"][path.name] = dtypes
            except (OSError, ValueError, csv.Error, TimeoutError, zipfile.BadZipFile) as exc:
                result["diagnostics"].append(f"{path.name}: {exc}")
        train = next((f for f in result["files"] if f["role"] == "train"), {})
        test = next((f for f in result["files"] if f["role"] == "test"), {})
        sample = next((f for f in result["files"] if f["role"] == "sample"), {})
        train_fields = set(train.get("columns", train.get("key_presence_counts", {})))
        test_fields = set(test.get("columns", test.get("key_presence_counts", {})))
        predictions = {
            c
            for c in sample.get("columns", [])
            if normalized_column(c) not in {"id", "image", "filename", "rowid", "imageid"}
        }
        result["public_submission_columns"] = sample.get("columns", [])
        result["designated_raw_target_fields"] = sorted((train_fields - test_fields) & predictions)
        if result["designated_raw_target_fields"]:
            result["target_field_provenance"] = {
                "file": sample["file"],
                "derivation": "Raw fields absent from test and named in the public submission template.",
            }
        # A successfully discovered but unsupported role remains explicit in
        # diagnostics. Retry missing/partial schemas on later reviews.
        if result["files"] and not result["diagnostics"]:
            self.cached = copy.deepcopy(result)
            self.signatures = signatures
        return result
