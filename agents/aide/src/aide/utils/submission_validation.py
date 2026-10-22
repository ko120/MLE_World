"""Submission-format evidence, independent of execution and model scores."""

import hashlib
import csv
import io
import logging
import stat
import tempfile
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import requests

logger = logging.getLogger("aide")


@dataclass(frozen=True)
class SubmissionValidation:
    status: Literal["valid", "invalid", "unknown"]
    message: str
    sha256: str | None = None
    diagnostics: tuple[str, ...] = ()
    repairs: tuple[dict, ...] = ()  # Legacy serialized evidence; no repairs are applied.
    columns: tuple[str, ...] = ()  # Header from the exact uploaded snapshot, not a template.

    def __bool__(self):
        raise TypeError("Check SubmissionValidation.status explicitly")


def submission_sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


class SubmissionValidator:
    """Cache definitive verdicts for the exact bytes sent to one competition.

    Unknown responses are never cached as CSV verdicts. A short server cooldown
    avoids repeated timeouts across sibling candidates; later calls can recover.
    Upload a snapshot so the hash and the validated CSV always agree.
    """

    def __init__(self, cache_size: int = 256, retry_delay_s: float = 60.0):
        self.cache_size = cache_size
        self.retry_delay_s = retry_delay_s
        self._cache: OrderedDict[tuple[str, str, str], SubmissionValidation] = OrderedDict()
        self._retry_after: dict[tuple[str, str], float] = {}

    def validate(
        self, csv_path, *, url: str, competition_id: str = "", timeout: float = 75.0,
    ) -> SubmissionValidation:
        started = time.monotonic()
        scope = (url, competition_id)
        digest = None
        try:
            path = Path(csv_path)
            info = path.stat()
            if not stat.S_ISREG(info.st_mode) or info.st_size == 0:
                return SubmissionValidation("invalid", "Submission CSV is missing or empty.")
            # Spill large CSVs to disk instead of holding them all in memory.
            with tempfile.SpooledTemporaryFile(max_size=8 * 1024 * 1024) as snapshot:
                hasher = hashlib.sha256()
                with path.open("rb") as source:
                    for chunk in iter(lambda: source.read(1024 * 1024), b""):
                        snapshot.write(chunk)
                        hasher.update(chunk)
                if snapshot.tell() == 0:
                    return SubmissionValidation("invalid", "Submission CSV is empty.")
                digest = hasher.hexdigest()
                key = (url, competition_id, digest)
                if key in self._cache:
                    self._cache.move_to_end(key)
                    return self._cache[key]
                if time.monotonic() < self._retry_after.get(scope, 0):
                    return SubmissionValidation("unknown", "Validation server is in retry cooldown.", digest)
                remaining = timeout - (time.monotonic() - started)
                if remaining <= 0:
                    return SubmissionValidation("unknown", "No submission-validation budget remains.", digest)
                snapshot.seek(0)
                try:
                    prefix = snapshot.read(256 * 1024).decode("utf-8-sig")
                    columns = tuple(next(csv.reader(io.StringIO(prefix), strict=True))) if "\n" in prefix else ()
                except (UnicodeError, csv.Error, StopIteration):
                    columns = ()
                snapshot.seek(0)
                response = requests.post(
                    url,
                    files={"file": ("submission.csv", snapshot, "text/csv")},
                    timeout=(min(5.0, remaining), remaining),
                )
                if response.status_code != 200:
                    result = SubmissionValidation(
                        "unknown", f"Validation server returned HTTP {response.status_code}.", digest
                    )
                else:
                    payload = response.json()
                    message = payload.get("result") if isinstance(payload, dict) else None
                    # Support the official server's text protocol. An arbitrary
                    # HTTP-200 body is neither a valid nor an invalid verdict.
                    if isinstance(message, str) and message.strip() == "Submission is valid.":
                        result = SubmissionValidation("valid", message.strip(), digest, columns=columns)
                    elif isinstance(message, str) and message.strip().startswith("Submission invalid!"):
                        result = SubmissionValidation("invalid", message.strip(), digest)
                    else:
                        result = SubmissionValidation("unknown", "Validation server returned no recognized verdict.", digest)
                if result.status != "unknown":
                    self._retry_after.pop(scope, None)
                    self._cache[key] = result
                    while len(self._cache) > self.cache_size:
                        self._cache.popitem(last=False)
                elif response.status_code == 200 or response.status_code >= 500:
                    # One outage must not cost a full timeout for every sibling
                    # in a best-of-N pool. Known cached verdicts still work.
                    self._retry_after[scope] = time.monotonic() + self.retry_delay_s
                logger.info("[validate] %s: %s (%.2fs)", path, result.status, time.monotonic() - started)
                return result
        except (FileNotFoundError, NotADirectoryError):
            return SubmissionValidation("invalid", "Submission CSV is missing.", digest)
        except (OSError, requests.RequestException, ValueError) as exc:
            # Inaccessibility, transport failures, and malformed JSON provide
            # no verdict about the submission's format.
            logger.warning("[validate] unavailable: %s: %s", type(exc).__name__, exc)
            if isinstance(exc, (requests.RequestException, ValueError)):
                self._retry_after[scope] = time.monotonic() + self.retry_delay_s
            return SubmissionValidation("unknown", f"Submission validation unavailable ({type(exc).__name__}).", digest)
