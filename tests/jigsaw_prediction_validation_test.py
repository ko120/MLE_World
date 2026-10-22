"""Invalid model predictions must not be reported as grader infrastructure faults."""
import importlib

import numpy as np
import pandas as pd
import pytest

from mlebench.grade_helpers import InvalidSubmissionError

grader = importlib.import_module(
    "mlebench.competitions.jigsaw-toxic-comment-classification-challenge.grade"
)


def frames():
    answers = pd.DataFrame({"id": ["a", "b", "c"], **{
        name: [0, 1, -1] for name in grader.CLASSES
    }})
    submission = pd.DataFrame({"id": ["a", "b", "c"], **{
        name: [0.1, 0.9, 0.5] for name in grader.CLASSES
    }})
    return submission, answers


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_nonfinite_scored_prediction_is_invalid(value):
    submission, answers = frames()
    submission.loc[0, grader.CLASSES[0]] = value
    with pytest.raises(InvalidSubmissionError, match="NaN or infinity"):
        grader.grade(submission, answers)


def test_valid_and_unscored_predictions_keep_the_same_score():
    submission, answers = frames()
    assert grader.grade(submission, answers) == 1.0
    submission.loc[2, grader.CLASSES] = np.nan
    assert grader.grade(submission, answers) == 1.0
