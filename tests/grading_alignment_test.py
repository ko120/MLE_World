"""Scores must follow named classes and unique row IDs, not input order."""
import importlib
import logging
import unittest

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from mlebench.grade_helpers import Grader, InvalidSubmissionError


JIGSAW = "jigsaw-toxic-comment-classification-challenge"
NOMAD = "nomad2018-predict-transparent-conductors"
TAXI = "new-york-city-taxi-fare-prediction"
TASKS = (JIGSAW, NOMAD, TAXI)
GRADERS = {
    task: importlib.import_module(f"mlebench.competitions.{task}.grade")
    for task in TASKS
}


def frames(task):
    if task == JIGSAW:
        # Distinct balanced columns expose class swaps that identical labels hide.
        labels = np.array([
            [0, 0, 1, 1], [0, 1, 0, 1], [0, 1, 1, 0],
            [1, 0, 0, 1], [1, 0, 1, 0], [1, 1, 0, 0],
        ]).T
        answers = pd.DataFrame(labels, columns=GRADERS[task].CLASSES)
        submission = pd.DataFrame(0.1 + 0.8 * labels, columns=GRADERS[task].CLASSES)
        answers.insert(0, "id", ["a", "b", "c", "d"])
        submission.insert(0, "id", answers["id"])
        return submission, answers, "id", 1.0
    if task == NOMAD:
        answers = pd.DataFrame({
            "id": [10, 20, 30],
            "formation_energy_ev_natom": [0.1, 1.0, 10.0],
            "bandgap_energy_ev": [1.0, 2.0, 5.0],
        })
        return answers.copy(), answers, "id", 0.0
    answers = pd.DataFrame({"key": ["a", "b", "c"], "fare_amount": [1.0, 2.0, 5.0]})
    return answers.copy(), answers, "key", 0.0


class GradingAlignmentTests(unittest.TestCase):
    def test_known_scores(self):
        for task in TASKS:
            with self.subTest(task=task):
                submission, answers, _, expected = frames(task)
                self.assertAlmostEqual(GRADERS[task].grade(submission, answers), expected)

    def test_submission_row_order_does_not_change_score(self):
        for task in TASKS:
            with self.subTest(task=task):
                submission, answers, _, expected = frames(task)
                self.assertAlmostEqual(GRADERS[task].grade(submission.iloc[::-1], answers), expected)

    def test_answer_row_order_does_not_change_score(self):
        for task in TASKS:
            with self.subTest(task=task):
                submission, answers, _, expected = frames(task)
                self.assertAlmostEqual(GRADERS[task].grade(submission, answers.iloc[::-1]), expected)

    def test_arbitrary_dataframe_indices_do_not_change_score(self):
        for task in TASKS:
            with self.subTest(task=task):
                submission, answers, _, expected = frames(task)
                submission.index = list(range(10, 10 + len(submission)))
                answers.index = list(range(30, 30 + len(answers)))
                self.assertAlmostEqual(GRADERS[task].grade(submission, answers), expected)

    def test_grading_does_not_mutate_inputs(self):
        for task in TASKS:
            with self.subTest(task=task):
                submission, answers, _, _ = frames(task)
                before_submission, before_answers = submission.copy(deep=True), answers.copy(deep=True)
                GRADERS[task].grade(submission, answers)
                assert_frame_equal(submission, before_submission)
                assert_frame_equal(answers, before_answers)

    def test_bad_submission_ids_are_invalid_submissions(self):
        for task in TASKS:
            for fault in ("missing_column", "wrong_ids", "duplicate", "null", "missing_row", "extra_row"):
                with self.subTest(task=task, fault=fault):
                    submission, answers, id_col, _ = frames(task)
                    if fault == "missing_column":
                        submission = submission.drop(columns=id_col)
                    elif fault == "wrong_ids":
                        submission[id_col] = [f"wrong-{i}" for i in range(len(submission))]
                    elif fault == "duplicate":
                        submission.loc[1, id_col] = submission.loc[0, id_col]
                    elif fault == "null":
                        submission[id_col] = submission[id_col].astype(object)
                        submission.loc[0, id_col] = None
                    elif fault == "missing_row":
                        submission = submission.iloc[:-1]
                    else:
                        submission = pd.concat([submission, submission.iloc[[0]]], ignore_index=True)
                    with self.assertRaises(InvalidSubmissionError):
                        GRADERS[task].grade(submission, answers)

    def test_duplicate_column_names_are_invalid_submissions(self):
        for task in TASKS:
            with self.subTest(task=task):
                submission, answers, id_col, _ = frames(task)
                submission = pd.concat([submission, submission[[id_col]]], axis=1)
                with self.assertRaises(InvalidSubmissionError):
                    GRADERS[task].grade(submission, answers)

    def test_bad_answer_ids_are_grader_faults(self):
        for task in TASKS:
            for fault in ("missing_column", "duplicate", "null"):
                with self.subTest(task=task, fault=fault):
                    submission, answers, id_col, _ = frames(task)
                    if fault == "missing_column":
                        answers = answers.drop(columns=id_col)
                    elif fault == "duplicate":
                        answers.loc[1, id_col] = answers.loc[0, id_col]
                    else:
                        answers[id_col] = answers[id_col].astype(object)
                        answers.loc[0, id_col] = None
                    with self.assertRaises(AssertionError):
                        GRADERS[task].grade(submission, answers)

    def test_jigsaw_named_class_order_does_not_change_score(self):
        submission, answers, _, _ = frames(JIGSAW)
        reordered = ["id", *reversed(GRADERS[JIGSAW].CLASSES)]
        for left, right in [(submission[reordered], answers), (submission, answers[reordered]),
                            (submission[reordered], answers[reordered])]:
            with self.subTest(left=list(left.columns), right=list(right.columns)):
                self.assertEqual(GRADERS[JIGSAW].grade(left, right), 1.0)

    def test_jigsaw_nonclass_columns_do_not_enter_metric(self):
        submission, answers, _, _ = frames(JIGSAW)
        submission["unused"] = "metadata"
        answers["unused"] = 42
        self.assertEqual(GRADERS[JIGSAW].grade(submission, answers), 1.0)

    def test_jigsaw_missing_class_is_invalid(self):
        submission, answers, _, _ = frames(JIGSAW)
        with self.assertRaises(InvalidSubmissionError):
            GRADERS[JIGSAW].grade(submission.drop(columns=GRADERS[JIGSAW].CLASSES[0]), answers)

    def test_jigsaw_nonfinite_scored_values_are_invalid(self):
        for value in (np.nan, np.inf, -np.inf):
            with self.subTest(value=value):
                submission, answers, _, _ = frames(JIGSAW)
                submission.loc[0, GRADERS[JIGSAW].CLASSES[0]] = value
                with self.assertRaises(InvalidSubmissionError):
                    GRADERS[JIGSAW].grade(submission, answers)

    def test_jigsaw_unscored_rows_remain_excluded(self):
        submission, answers, _, _ = frames(JIGSAW)
        classes = list(GRADERS[JIGSAW].CLASSES)
        submission.loc[len(submission)] = ["unscored", *([np.nan] * len(classes))]
        answers.loc[len(answers)] = ["unscored", *([-1] * len(classes))]
        self.assertEqual(GRADERS[JIGSAW].grade(submission.iloc[::-1], answers), 1.0)

    def test_grader_wrapper_classifies_wrong_ids_as_submission_faults(self):
        for task in TASKS:
            with self.subTest(task=task):
                submission, answers, id_col, _ = frames(task)
                submission[id_col] = [f"wrong-{i}" for i in range(len(submission))]
                grader = Grader("alignment-test", f"mlebench.competitions.{task}.grade:grade", task != JIGSAW)
                with self.assertLogs("mlebench.grade_helpers", level=logging.WARNING) as logs:
                    self.assertIsNone(grader(submission, answers))
                self.assertTrue(all(record.levelno == logging.WARNING for record in logs.records))


if __name__ == "__main__":
    unittest.main()
