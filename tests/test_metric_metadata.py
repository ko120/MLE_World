"""Metric declarations must agree with the benchmark and need no runtime probing."""

import logging
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import pandas as pd
import yaml

from agents.run import _agent_env, _eval_metric_env
from mlebench.grade import grade_score
from mlebench.grade_helpers import Grader
from mlebench.metric_direction import (
    CLEAN_METRIC_NAME,
    COMPETITIONS_DIR,
    LOWER_IS_BETTER,
    audit_metric_metadata,
)
from mlebench.registry import registry


LOGGER = logging.getLogger(__name__)


def grader_config(competition_id):
    path = COMPETITIONS_DIR / competition_id / "config.yaml"
    return yaml.safe_load(path.read_text())["grader"]


class MetricResolutionTests(unittest.TestCase):
    def test_every_competition_resolves_without_submissions_or_grader_calls(self):
        # NaN used to incorrectly pin higher-is-better. Grading results (including
        # NaN) must now have no influence on the declared metadata.
        grade_fn = Mock(return_value=float("nan"))
        with tempfile.TemporaryDirectory() as directory:
            reg = registry.set_data_dir(Path(directory) / "data-not-downloaded")
            with patch("mlebench.registry.import_fn", return_value=Mock()):
                with patch("mlebench.grade_helpers.import_fn", return_value=grade_fn):
                    competitions = [reg.get_competition(cid) for cid in reg.list_competition_ids()]
            self.assertTrue(competitions)
            self.assertEqual(
                {comp.id for comp in competitions},
                set((COMPETITIONS_DIR.parents[1] / "configs/tasks.txt").read_text().split()),
            )
            with patch.object(pd, "read_csv", side_effect=AssertionError("Runtime data read")):
                for competition in competitions:
                    with self.subTest(competition=competition.id):
                        self.assertFalse(competition.answers.exists())
                        env = _eval_metric_env(competition, LOGGER)
                        self.assertEqual(env["METRIC_NAME"], competition.grader.name)
                        self.assertIn(env["METRIC_LOWER_IS_BETTER"], ("true", "false"))
                        self.assertEqual(
                            env["METRIC_LOWER_IS_BETTER"] == "true",
                            competition.grader.is_lower_better(),
                        )
        grade_fn.assert_not_called()

    def test_profile_cannot_replace_competition_metric(self):
        with patch("mlebench.grade_helpers.import_fn"):
            grader = Grader.from_dict(grader_config("denoising-dirty-documents"))
        competition = SimpleNamespace(id="denoising-dirty-documents", grader=grader)
        agent = SimpleNamespace(env_vars={
            "COMPETITION_ID": "wrong-competition", "METRIC_NAME": "accuracy",
            "METRIC_LOWER_IS_BETTER": "false", "TIME_LIMIT_SECS": "60",
        })
        for resolve in (_eval_metric_env, lambda comp, logger: _agent_env(comp, agent, logger)):
            env = resolve(competition, LOGGER)
            self.assertEqual(env["METRIC_NAME"], "root_mean_squared_error")
            self.assertEqual(env["METRIC_LOWER_IS_BETTER"], "true")
        self.assertEqual(env["COMPETITION_ID"], competition.id)
        self.assertEqual(env["TIME_LIMIT_SECS"], "60")

    def test_missing_or_nonboolean_direction_is_rejected(self):
        config = grader_config("denoising-dirty-documents")
        with patch("mlebench.grade_helpers.import_fn") as import_fn:
            for value in (None, "false", "true", 0, 1):
                with self.subTest(value=value):
                    with self.assertRaisesRegex(ValueError, "explicit boolean"):
                        Grader.from_dict({**config, "lower_is_better": value})
            del config["lower_is_better"]
            with self.assertRaisesRegex(TypeError, "lower_is_better"):
                Grader.from_dict(config)
        import_fn.assert_not_called()

    def test_offline_ranking_uses_the_declared_direction(self):
        for lower, scores, score in ((True, [1, 2, 3, 4], 0.5), (False, [4, 3, 2, 1], 4.5)):
            with self.subTest(lower=lower):
                with patch("mlebench.grade_helpers.import_fn"):
                    # Identical display names must not imply identical directions.
                    grader = Grader("shared-name", "unused:grade", lower)
                competition = SimpleNamespace(id="test", grader=grader)
                self.assertIs(grader.is_lower_better(pd.DataFrame()), lower)
                with patch("mlebench.grade.get_leaderboard", return_value=pd.DataFrame({"score": scores})):
                    report = grade_score(score, competition)
                self.assertIs(report.is_lower_better, lower)
                self.assertTrue(report.gold_medal)
                self.assertTrue(report.above_median)

    def test_denoising_declaration_matches_actual_rmse_grader(self):
        grader = Grader.from_dict(grader_config("denoising-dirty-documents"))
        answers = pd.DataFrame({"id": [1, 2], "value": [10.0, 20.0]})
        gold = answers.copy()
        baseline = gold.assign(value=[14.0, 24.0])
        self.assertEqual(grader.name, "root_mean_squared_error")
        self.assertTrue(grader.lower_is_better)
        self.assertEqual(grader(gold, answers.copy()), 0.0)
        self.assertEqual(grader(baseline, answers.copy()), 4.0)

    def test_cactus_declaration_matches_actual_auc_grader(self):
        grader = Grader.from_dict(grader_config("aerial-cactus-identification"))
        answers = pd.DataFrame({"id": ["a", "b"], "has_cactus": [0, 1]})
        gold = answers.copy()
        baseline = gold.assign(has_cactus=[1, 0])
        self.assertEqual(grader.name, "auc-roc")
        self.assertFalse(grader.lower_is_better)
        self.assertEqual(grader(gold, answers.copy()), 1.0)
        self.assertEqual(grader(baseline, answers.copy()), 0.0)

    def test_legacy_views_match_retained_competition_metrics(self):
        self.assertIs(LOWER_IS_BETTER["root_mean_squared_error"], True)
        self.assertIs(LOWER_IS_BETTER["auc-roc"], False)
        self.assertNotIn("auc-roc", CLEAN_METRIC_NAME)


class MetricAuditTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.competition = self.root / "test"
        self.competition.mkdir()
        (self.competition / "config.yaml").write_text(
            "grader:\n  name: mean-absolute-error\n  lower_is_better: true\n"
        )

    def test_audit_accepts_declared_direction_and_tied_neighbors(self):
        (self.competition / "leaderboard.csv").write_text("score\n0\n0\n1\n2\n")
        self.assertEqual(audit_metric_metadata(self.root), 1)

    def test_audit_rejects_missing_and_unhydrated_leaderboards(self):
        with self.assertRaisesRegex(ValueError, "hydrate"):
            audit_metric_metadata(self.root)
        (self.competition / "leaderboard.csv").write_text(
            "version https://git-lfs.github.com/spec/v1\noid sha256:test\nsize 123\n"
        )
        with self.assertRaisesRegex(ValueError, "hydrate"):
            audit_metric_metadata(self.root)

    def test_audit_rejects_nonfinite_degenerate_or_wrongly_ordered_scores(self):
        for scores in ([0, float("nan")], [0, float("inf")], [1, 1], [1], [2, 1, 0], [0, 2, 1]):
            with self.subTest(scores=scores):
                (self.competition / "leaderboard.csv").write_text(
                    "score\n" + "\n".join(map(str, scores)) + "\n"
                )
                with self.assertRaises(ValueError):
                    audit_metric_metadata(self.root)


if __name__ == "__main__":
    unittest.main()
