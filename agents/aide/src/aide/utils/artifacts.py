"""Publish artifacts only after a candidate has passed real execution review."""

import os
import logging
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path

from .submission_validation import submission_sha256

logger = logging.getLogger("aide")


def _remove_path(path):
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.exists():
        shutil.rmtree(path)


@contextmanager
def preserve_published_artifacts(workspace):
    """Undo candidate writes to published files before review can publish.

    Generated programs share the harness user and can reach the output mounts.
    Keep private temporary copies outside their workspace, including the absence
    of a first publication. This protects against accidental writes; it is not
    a security boundary against a deliberately hostile same-user process.
    """
    workspace = Path(workspace)
    names = {
        "best_solution": ("solution.py", "node_id.txt"),
        "best_submission": ("submission.csv",),
    }
    with tempfile.TemporaryDirectory(prefix="aide-published-") as temporary:
        records = []
        anchors = []
        for directory, files in names.items():
            alias = workspace / directory
            link = os.readlink(alias) if alias.is_symlink() else None
            target = alias.resolve()
            anchors.append((alias, target, link))
            for name in files:
                path = target / name
                backup = Path(temporary) / f"{directory}-{name}"
                if path.is_symlink():
                    raise ValueError(f"Published artifact must be a regular file: {path}")
                sha = None
                if path.exists():
                    if not path.is_file():
                        raise ValueError(f"Published artifact must be a regular file: {path}")
                    shutil.copy2(path, backup)
                    sha = submission_sha256(backup)
                records.append((path, backup, sha))
        try:
            yield
        finally:
            restored = []
            for alias, target, link in anchors:
                if link is not None:
                    if not alias.is_symlink() or os.readlink(alias) != link:
                        _remove_path(alias)
                        alias.symlink_to(link, target_is_directory=True)
                        restored.append(str(alias))
                elif alias.is_symlink() or (alias.exists() and not alias.is_dir()):
                    _remove_path(alias)
                    restored.append(str(alias))
                if target.is_symlink() or (target.exists() and not target.is_dir()):
                    _remove_path(target)
                target.mkdir(parents=True, exist_ok=True)
            for path, backup, sha in records:
                if sha is None:
                    if path.exists() or path.is_symlink():
                        _remove_path(path)
                        restored.append(str(path))
                    continue
                if (not path.is_symlink() and path.is_file()
                        and submission_sha256(path) == sha):
                    continue
                with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".restore-", delete=False) as handle:
                    staged = Path(handle.name)
                try:
                    shutil.copy2(backup, staged)
                    if path.is_dir() and not path.is_symlink():
                        shutil.rmtree(path)
                    os.replace(staged, path)
                finally:
                    staged.unlink(missing_ok=True)
                restored.append(str(path))
            if restored:
                logger.warning("[publish-guard] restored candidate writes before review: %s", restored)


def has_submission_csv(workspace) -> bool:
    """Check for a nonempty regular CSV; presence does not prove format validity."""
    csv_path = Path(workspace) / "submission" / "submission.csv"
    try:
        return csv_path.is_file() and csv_path.stat().st_size > 0
    except OSError:
        return False


def reset_submission_directory(workspace):
    """Require a fresh output for each execution, including after a symlink."""
    directory = Path(workspace) / "submission"
    if directory.is_symlink() or directory.is_file():
        directory.unlink()
    else:
        shutil.rmtree(directory, ignore_errors=True)
    directory.mkdir(parents=True, exist_ok=True)


def publish_real_solution(workspace: Path, node, *, source_path: Path | None = None) -> bool:
    """Stage the confirmed CSV and its code before replacing published files.

    Each replacement is atomic; the three files can live on separate harness
    mounts, so this is not a transaction across all three files.
    """
    source = source_path if source_path is not None else workspace / "submission" / "submission.csv"
    if (
        not node.real_won
        or node.is_buggy
        or node.submission_validation != "valid"
        or getattr(node, "pending_submission", None)
        or not node.submission_sha256
        or not source.is_file()
        or source.stat().st_size == 0
    ):
        return False
    destinations = [
        (workspace / "best_solution" / "solution.py", node.code),
        (workspace / "best_solution" / "node_id.txt", str(node.id)),
        (workspace / "best_submission" / "submission.csv", None),
    ]
    staged = []
    try:
        for destination, content in destinations:
            destination.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                dir=destination.parent, prefix=".publish-", delete=False
            ) as handle:
                temporary = Path(handle.name)
                staged.append((temporary, destination))
                if content is None:
                    with source.open("rb") as submission:
                        shutil.copyfileobj(submission, handle)
                else:
                    handle.write(content.encode("utf-8"))
            # Match normal published-file permissions, not tempfile's 0600.
            temporary.chmod(0o644)
            if content is None and submission_sha256(temporary) != node.submission_sha256:
                node.submission_validation = "unknown"
                node.submission_validation_message = "Submission changed after format validation."
                logger.warning("[publish] CSV changed since validation for node %s; preserving published files", node.id)
                return False
        for temporary, destination in staged:
            os.replace(temporary, destination)
    finally:
        for temporary, _ in staged:
            temporary.unlink(missing_ok=True)
    return True
