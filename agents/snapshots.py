"""Freeze and verify launch-time agent and grader code for a run group."""

import hashlib
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from agents.runtime_common import ENROOT_GRADING_PROTOCOL, GRADING_SCRIPTS
from mlebench.utils import purple


def _hash_tree(root: Path) -> str:
    """Content hash of every .py under root (path-sorted). Only .py: data files
    such as leaderboard.csv are git-LFS-hydrated in the repo but pointers in the
    image, which would make the hash differ for a non-behavioural reason."""
    h = hashlib.sha256()
    for p in sorted(root.rglob("*.py")):
        h.update(str(p.relative_to(root)).encode())
        h.update(p.read_bytes())
    return h.hexdigest()


def _grading_snapshot(snapshot: Path) -> Path:
    """Require frozen grading scripts; never fill an old snapshot from live code."""
    provenance_path = snapshot / "PROVENANCE.json"
    if not provenance_path.is_file():
        raise RuntimeError(
            f"Agent snapshot missing at {snapshot}; run_agent.py must create it before launch"
        )
    provenance = json.loads(provenance_path.read_text())
    directory = snapshot / "environment"
    expected = provenance.get("grading_code_sha256")
    if not expected or any(not (directory / name).is_file() for name in GRADING_SCRIPTS):
        raise RuntimeError(
            f"Snapshot {snapshot} has no frozen grading server. "
            "Start a new run group; existing snapshots are not modified."
        )
    if _hash_tree(directory) != expected:
        raise RuntimeError(f"Grading code hash mismatch in snapshot {snapshot}")
    if provenance.get("enroot_grading_protocol") != ENROOT_GRADING_PROTOCOL:
        raise RuntimeError(
            f"Snapshot {snapshot} uses an older Enroot grading protocol. Start a new run group; "
            "existing snapshots are not modified."
        )
    return directory


def snapshot_agent_code(agent_dir_host: Path, run_group_dir: Path, logger) -> Path:
    """Copy the agent dir + the mlebench package into runs/<group>/agent_snapshot
    plus grading scripts, and stamp their hashes in PROVENANCE.json.

    run_in_enroot binds src/aide, start.sh, etc. from this copy, so every task
    in the group runs launch-time code regardless of later host edits or a
    stale image. Reused as-is when it already exists, so a resumed group keeps
    its original code. Built under a tmp name and renamed so a crash mid-copy
    cannot leave a half-snapshot a later launch would trust.
    """
    snapshot = run_group_dir / "agent_snapshot"
    provenance_path = snapshot / "PROVENANCE.json"
    if provenance_path.exists():
        _grading_snapshot(snapshot)
        sha = json.loads(provenance_path.read_text())["src_aide_sha256"]
        logger.info(f"Reusing agent snapshot {snapshot} (src/aide sha {sha[:12]})")
        return snapshot

    tmp = run_group_dir / "agent_snapshot.tmp"
    if tmp.exists():
        shutil.rmtree(tmp)
    ignore = shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache", "*.egg-info")
    shutil.copytree(agent_dir_host, tmp, ignore=ignore)
    # The graders + grading_server's imports. The image bakes its own copy at
    # build time, so without this a repo grader fix never reaches a run.
    mlebench_src = Path(__file__).resolve().parents[1] / "mlebench"
    shutil.copytree(mlebench_src, tmp / "mlebench", ignore=ignore)
    grading_dir = tmp / "environment"
    grading_dir.mkdir()
    for name in GRADING_SCRIPTS:
        shutil.copy2(mlebench_src.parent / "environment" / name, grading_dir / name)

    # Same implementation the container re-runs at start (snapshot handshake),
    # so the two sides cannot drift on algorithm.
    sha = subprocess.run(
        ["bash", str(tmp / "snapshot_check.sh"), "hash", str(tmp / "src" / "aide")],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    git = {}
    for key, cmd in {
        "git_head": ["git", "rev-parse", "HEAD"],
        "git_status": ["git", "status", "--porcelain", "--", "."],  # agent subtree only
    }.items():
        try:
            git[key] = subprocess.run(
                cmd, cwd=agent_dir_host, capture_output=True, text=True, check=True
            ).stdout.strip()
        except Exception:
            git[key] = None
    mlebench_sha = _hash_tree(tmp / "mlebench")
    (tmp / "PROVENANCE.json").write_text(
        json.dumps(
            {
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "source": str(Path(agent_dir_host).resolve()),
                "src_aide_sha256": sha,
                "mlebench_sha256": mlebench_sha,
                "grading_code_sha256": _hash_tree(grading_dir),
                "enroot_grading_protocol": ENROOT_GRADING_PROTOCOL,
                "git_head": git["git_head"],
                "git_dirty": bool(git["git_status"]),
                "git_status_agent_subtree": (git["git_status"] or "")[:4000],
            },
            indent=2,
        )
    )
    tmp.rename(snapshot)
    dirty = " dirty" if git["git_status"] else ""
    logger.info(
        purple(
            f"Agent snapshot: {snapshot} (src/aide sha {sha[:12]}, "
            f"mlebench sha {mlebench_sha[:12]}, git {str(git['git_head'])[:12]}{dirty})"
        )
    )
    return snapshot
