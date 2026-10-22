"""UCB reference selection and signed normalization reach selection logs."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "agents/aide/src"))
from aide import ucb


def pool(metrics, probabilities=None):
    probabilities = probabilities or [0.9] * len(metrics)
    return [
        ucb.Candidate(i, True, metric, scores={"max_p": p})
        for i, (metric, p) in enumerate(zip(metrics, probabilities))
    ]


@pytest.mark.parametrize(
    "maximize,reference,metrics",
    [(True, 0.9, [0.92, 0.9, 0.88]), (False, 0.1, [0.08, 0.1, 0.12])],
)
def test_same_progress_in_accuracy_and_error_including_decreases(maximize, reference, metrics):
    winner, audit = ucb.select(
        pool(metrics),
        ucb.Options(normalize_mu=True),
        maximize=maximize,
        reference_metric=reference,
    )
    assert winner == 0
    assert [row["mu"] for row in audit["candidates"]] == pytest.approx([0.2, 0, -0.2])
    assert audit["metric_scale"] == pytest.approx(0.1)
    assert audit["mu_normalization"] == "remaining_gap"
    assert audit["ideal_metric"] == (1 if maximize else 0)


@pytest.mark.parametrize("normalize,beta,winner", [(False, 1, 1), (True, 1, 0), (True, 3, 1)])
def test_flag_changes_selection_and_beta_still_controls_exploration(normalize, beta, winner):
    picked, _ = ucb.select(
        pool([0.92, 0.91], [0.95, 0.9]),
        ucb.Options(normalize_mu=normalize, beta=beta),
        maximize=True,
        reference_metric=0.9,
    )
    assert picked == winner


@pytest.mark.parametrize("maximize,reference,metric", [(True, 1, 0.99), (False, 0, 0.01)])
def test_perfect_reference_uses_configured_floor(maximize, reference, metric):
    _, audit = ucb.select(
        pool([metric]),
        ucb.Options(normalize_mu=True, scale_floor=0.001),
        maximize=maximize,
        reference_metric=reference,
    )
    assert audit["metric_scale"] == 0.001
    assert audit["candidates"][0]["mu"] == pytest.approx(-10)
    json.dumps(audit, allow_nan=False)


def test_custom_ideal_and_explicit_scale_override():
    candidates = pool([92, 88])
    _, audit = ucb.select(
        candidates,
        ucb.Options(normalize_mu=True, ideal_metric=100),
        maximize=True,
        reference_metric=90,
    )
    assert [row["mu"] for row in audit["candidates"]] == pytest.approx([0.2, -0.2])
    _, fixed = ucb.select(
        candidates,
        ucb.Options(normalize_mu=True, ideal_metric=100, metric_scale=20),
        maximize=True,
        reference_metric=90,
    )
    assert fixed["mu_normalization"] == "fixed"
    assert [row["mu"] for row in fixed["candidates"]] == pytest.approx([0.1, -0.1])


def test_missing_reference_preserves_pool_fallback():
    candidates = pool([0.92, 0.91], [0.95, 0.9])
    winner, audit = ucb.select(candidates, ucb.Options(normalize_mu=True), maximize=True)
    old_winner, old = ucb.select(candidates, ucb.Options(), maximize=True)
    assert winner == old_winner
    assert audit["candidates"] == old["candidates"]
    assert audit["mu_normalization"] == "pool"
    assert audit["ideal_metric"] is None


@pytest.mark.parametrize(
    "override,field",
    [
        ({"normalize_mu": "yes"}, "normalize_mu"),
        ({"ideal_metric": float("nan")}, "ideal_metric"),
        ({"scale_floor": 0}, "scale_floor"),
        ({"scale_floor": -1}, "scale_floor"),
        ({"scale_floor": float("inf")}, "scale_floor"),
    ],
)
def test_invalid_normalization_options_fail_early(override, field):
    with pytest.raises(ValueError, match=field):
        ucb.resolve_config({"improve": override}, "improve")


@pytest.mark.parametrize("maximize", [True, False])
@pytest.mark.parametrize("has_real_reference", [False, True])
def test_agent_applies_reference_priority_and_records_normalized_terms(
    tmp_path, maximize, has_real_reference
):
    from aide.agent import Agent
    from aide.journal import Journal, Node
    from aide.utils.metric import MetricValue

    agent = Agent.__new__(Agent)
    agent.acfg = SimpleNamespace(
        ucb=ucb.UCBConfig(improve=ucb.UCBStageConfig(normalize_mu=True)),
        eval_lower_is_better=not maximize,
    )
    agent.oracle = "sim"
    agent.journal = Journal()
    agent._exc_scores = lambda code: {"max_p": {"a": 0.95, "b": 0.9}[code]}

    def metric_value(accuracy):
        return accuracy if maximize else 1 - accuracy

    parent = Node(
        code="parent",
        metric=MetricValue(metric_value(0.9), maximize=maximize),
        is_buggy=False,
    )
    agent.journal.append(parent)
    agent.journal.append(
        Node(
            code="unvalidated real candidate",
            metric=MetricValue(metric_value(0.99), maximize=maximize),
            is_buggy=False,
            real_won=True,
            submission_validation="unknown",
            submission_sha256="unvalidated",
        )
    )
    for accuracy in [0.4, 0.5] if has_real_reference else []:
        agent.journal.append(
            Node(
                code="validated real candidate",
                metric=MetricValue(metric_value(accuracy), maximize=maximize),
                is_buggy=False,
                real_won=True,
                submission_validation="valid",
                submission_sha256="validated",
            )
        )
    outcomes = [
        SimpleNamespace(
            predicted=True,
            usable=True,
            review={},
            result=SimpleNamespace(
                predicted_metric=metric_value(m), predicted_submission_exists=True
            ),
        )
        for m in [0.92, 0.91]
    ]
    picked = agent._select_ucb([("", "a"), ("", "b")], outcomes, tmp_path, "improve", parent)
    audit = json.loads((tmp_path / "ucb.json").read_text())
    assert picked == audit["winner"] == (1 if has_real_reference else 0)
    assert audit["options"]["normalize_mu"] is True
    assert audit["mu_normalization"] == "remaining_gap"
    assert audit["reference_source"] == ("best_real" if has_real_reference else "parent")
    assert audit["reference_metric"] == pytest.approx(
        metric_value(0.5 if has_real_reference else 0.9)
    )
    assert audit["metric_scale"] == pytest.approx(0.5 if has_real_reference else 0.1)
    expected_scores = [0.89, 0.92] if has_real_reference else [0.25, 0.2]
    assert [row["score"] for row in audit["candidates"]] == pytest.approx(expected_scores)
