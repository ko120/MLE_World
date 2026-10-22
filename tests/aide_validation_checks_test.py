"""The agent preserves reviewer trust verdicts for real and predicted results."""

import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "agents/aide/src"))
BAD_REDUCTION = textwrap.dedent(
    """
    import numpy as np
    targets = np.zeros((len(y), 99))
    def loss(probabilities, onehot):
        clipped = np.clip(probabilities, 1e-15, 1 - 1e-15)
        value = -np.mean(onehot * np.log(clipped))
        return value
    validation_loss = loss(predictions, targets)
"""
)


@pytest.mark.parametrize("predicted", [False, True])
@pytest.mark.parametrize("trustworthy", [False, True])
def test_pipeline_preserves_reviewer_trust_verdict(predicted, trustworthy, monkeypatch):
    import aide.agent as agent_module

    agent = object.__new__(agent_module.Agent)
    agent.task_desc = "Training task"
    agent.acfg = SimpleNamespace(
        obfuscate=False,
        eval_metric="multi-class-log-loss",
        convert_system_to_user=False,
        feedback=SimpleNamespace(model="mock", temp=0),
    )
    monkeypatch.setattr(
        agent_module,
        "query",
        lambda **kwargs: {
            "is_bug": False,
            "has_csv_submission": True,
            "metric": 0.9,
            "lower_is_better": True,
            "metric_trustworthy": trustworthy,
            "summary": "Reviewer assessment.",
        },
    )
    response = agent._review(BAD_REDUCTION, "Validation score: 0.9", predicted=predicted)
    assert response["metric"] == 0.9
    assert response["metric_trustworthy"] is trustworthy
    assert response["summary"] == "Reviewer assessment."
    assert agent_module.review_is_buggy(response, False, True) is (not trustworthy)
