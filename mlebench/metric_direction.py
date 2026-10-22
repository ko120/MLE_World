"""Audit the explicit metric metadata in each competition's config.yaml.

grader.name and grader.lower_is_better are the source of truth for agent
startup and offline grading. Directions were verified against all 82 bundled
leaderboards; the ventilator grader's name was corrected to mean-absolute-error
after checking its implementation. New competitions must declare both fields.

With hydrated leaderboard CSVs, check the declarations without grading any
submissions: python -m mlebench.metric_direction

The name-keyed maps below are compatibility views for older analysis scripts,
derived from the configs rather than maintained separately. Runtime code uses
competition.grader directly. Conflicting names are omitted from the legacy map.
"""

import csv
import math
from pathlib import Path

import yaml


COMPETITIONS_DIR = Path(__file__).with_name("competitions")


def _configured_graders(competitions_dir: Path):
    paths = sorted(competitions_dir.glob("*/config.yaml"))
    if not paths:
        raise ValueError(f"No competition configs found under {competitions_dir}")
    for path in paths:
        config = yaml.safe_load(path.read_text())
        grader = config["grader"]
        if not isinstance(grader.get("name"), str) or not grader["name"].strip():
            raise ValueError(f"Missing metric name in {path}")
        if type(grader.get("lower_is_better")) is not bool:
            raise ValueError(f"Missing boolean grader.lower_is_better in {path}")
        yield path, grader


def audit_metric_metadata(competitions_dir: Path = COMPETITIONS_DIR) -> int:
    """Check declared directions against finite, ordered leaderboard scores.

    Missing/unhydrated or degenerate leaderboards fail the audit; they never
    cause a guessed direction or modify the declarations.
    """
    count = 0
    for path, grader in _configured_graders(competitions_dir):
        leaderboard = path.with_name("leaderboard.csv")
        try:
            with leaderboard.open() as stream:
                scores = [float(row["score"]) for row in csv.DictReader(stream)]
        except (OSError, KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Cannot audit {leaderboard}; hydrate its leaderboard CSV") from exc
        if (
            len(scores) < 2
            or not all(math.isfinite(score) for score in scores)
            or scores[0] == scores[-1]
        ):
            raise ValueError(f"Nonfinite or degenerate scores in {leaderboard}")
        lower = grader["lower_is_better"]
        if not all(a <= b if lower else a >= b for a, b in zip(scores, scores[1:])):
            raise ValueError(f"grader.lower_is_better disagrees with {leaderboard}")
        count += 1
    return count


def _legacy_directions_by_name() -> dict[str, bool]:
    directions: dict[str, set[bool]] = {}
    for _, grader in _configured_graders(COMPETITIONS_DIR):
        directions.setdefault(grader["name"], set()).add(grader["lower_is_better"])
    return {name: next(iter(values)) for name, values in directions.items() if len(values) == 1}


LOWER_IS_BETTER = _legacy_directions_by_name()
# Names are now corrected in the competition configs, where they are unambiguous.
CLEAN_METRIC_NAME: dict[str, str] = {}


if __name__ == "__main__":
    count = audit_metric_metadata()
    print(f"Verified metric directions for {count} competitions against their leaderboards.")
