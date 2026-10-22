"""Decision properties of fixed-pool UCB; no models or execution required."""

import math

import pytest

from aide.ucb import Candidate, Options, resolve_config, select, uncertainty


@pytest.mark.parametrize("source", ["p_clean", "p_max", "entropy"])
def test_uncertainty_can_trade_a_small_metric_loss_for_exploration(source):
    candidates = [
        Candidate(
            0,
            True,
            0.90,
            scores={
                "p_clean": 0.99,
                "max_p": 0.99,
                "entropy": 0.01 * math.log(24),
                "support_size": 24,
            },
        ),
        Candidate(
            1,
            True,
            0.85,
            scores={
                "p_clean": 0.60,
                "max_p": 0.60,
                "entropy": 0.40 * math.log(24),
                "support_size": 24,
            },
        ),
    ]
    picked, audit = select(
        candidates, Options(sigma=source, beta=0.5, metric_scale=1), maximize=True
    )
    assert picked == 1
    assert audit["greedy_winner"] == 0
    assert audit["candidates"][1]["score"] == pytest.approx(1.05)


def test_clean_probability_and_symmetric_confidence_are_different_signals():
    scores = {"p_clean": 0.05, "max_p": 0.90}
    assert uncertainty(scores, "p_clean") == pytest.approx(0.95)
    assert uncertainty(scores, "p_max") == pytest.approx(0.10)


def test_entropy_is_normalized_by_actual_support_size():
    assert uncertainty({"entropy": math.log(24), "support_size": 24}, "entropy") == 1
    assert uncertainty({"entropy": 0, "support_size": 24}, "entropy") == 0
    assert uncertainty({"entropy": 1}, "entropy") is None


@pytest.mark.parametrize("bad", [None, float("nan"), float("inf"), -0.1, 1.1, True])
def test_invalid_probability_is_missing_uncertainty(bad):
    assert uncertainty({"max_p": bad}, "p_max") is None


@pytest.mark.parametrize("maximize, winner", [(True, 0), (False, 1)])
def test_beta_zero_preserves_metric_direction_without_confidence(maximize, winner):
    picked, audit = select(
        [Candidate(0, True, 0.9), Candidate(1, True, 0.8)], Options(beta=0), maximize=maximize
    )
    assert picked == winner
    assert audit["rule"] == "metric_greedy"


def test_constant_sigma_and_ties_preserve_metric_greedy_order():
    candidates = [
        Candidate(i, True, v, scores={"max_p": 0.8}) for i, v in enumerate([0.85, 0.90, 0.90])
    ]
    assert select(candidates, Options(beta=2), maximize=True)[0] == 1


def test_auto_scaling_is_invariant_to_metric_units():
    candidates = [
        Candidate(0, True, 0.90, scores={"max_p": 0.99}),
        Candidate(1, True, 0.85, scores={"max_p": 0.6}),
    ]
    first, a = select(candidates, Options(), maximize=True, reference_metric=0.8)
    for candidate in candidates:
        candidate.metric *= 100
    second, b = select(candidates, Options(), maximize=True, reference_metric=80)
    assert first == second
    assert [r["score"] for r in a["candidates"]] == pytest.approx(
        [r["score"] for r in b["candidates"]]
    )


def test_missing_sigma_makes_entire_pool_greedy():
    candidates = [Candidate(0, True, 0.9), Candidate(1, True, 0.8, scores={"max_p": 0.1})]
    picked, audit = select(candidates, Options(), maximize=True)
    assert picked == 0
    assert audit["fallback_reason"] == "missing_uncertainty"
    assert all(r["bonus"] == 0 for r in audit["candidates"])


def test_known_execution_has_no_world_model_prediction_bonus():
    candidates = [
        Candidate(0, True, 0.9, observed=True, scores={"max_p": 0.1}),
        Candidate(1, True, 0.85, observed=True),
    ]
    picked, audit = select(candidates, Options(beta=10), maximize=True)
    assert picked == 0
    assert [r["sigma"] for r in audit["candidates"]] == [0, 0]


def test_crashes_and_missing_metrics_cannot_win_with_an_uncertainty_bonus():
    candidates = [
        Candidate(0, False, 10, scores={"max_p": 0.1}),
        Candidate(1, True, None, scores={"max_p": 0.1}),
        Candidate(2, True, 0.8, scores={"max_p": 0.99}),
    ]
    picked, audit = select(candidates, Options(beta=100), maximize=True)
    assert picked == 2
    assert [r["eligible"] for r in audit["candidates"]] == [False, False, True]


def test_all_failed_pool_prefers_success_probability_independent_of_sigma():
    candidates = [
        Candidate(0, False, None, scores={"p_clean": 0.1, "max_p": 0.5}),
        Candidate(1, False, None, scores={"p_clean": 0.3, "max_p": 0.7}),
    ]
    picked, audit = select(candidates, Options(sigma="entropy"), maximize=False)
    assert picked == 1
    assert audit["rule"] == "p_clean"
    assert audit["fallback_reason"] == "no_usable_metric"


def test_total_scoring_failure_falls_back_to_first_usable():
    picked, audit = select(
        [Candidate(0, False, None), Candidate(1, True, None)], Options(), maximize=True
    )
    assert picked == 1
    assert audit["rule"] == "first_usable"


def test_unknown_direction_does_not_assume_maximization():
    candidates = [
        Candidate(0, True, 0.9, scores={"p_clean": 0.1}),
        Candidate(1, True, 0.8, scores={"p_clean": 0.8}),
    ]
    picked, audit = select(candidates, Options(), maximize=None)
    assert picked == 1
    assert audit["fallback_reason"] == "unknown_metric_direction"


def test_stage_settings_inherit_independently_and_accept_client_alias():
    cfg = {
        "sigma": "max_p",
        "beta": 0.5,
        "metric_scale": 0.1,
        "improve": {"sigma": "entropy", "beta": 2},
    }
    assert resolve_config(cfg, "draft") == Options(sigma="p_max", beta=0.5, metric_scale=0.1)
    assert resolve_config(cfg, "improve") == Options(sigma="entropy", beta=2, metric_scale=0.1)


@pytest.mark.parametrize(
    "config",
    [
        {"sigma": "typo"},
        {"beta": -1},
        {"beta": float("nan")},
        {"metric_scale": 0},
        {"metric_scale": float("inf")},
        {"score_workers": 0},
        {"score_workers": True},
        {"improve": {"sigma": "typo"}},
    ],
)
def test_bad_config_is_rejected_before_search(config):
    with pytest.raises(ValueError):
        resolve_config(config, "improve")


def test_clean_argmax_uses_the_same_metric_gate_as_greedy_and_optimism():
    pool = [
        Candidate(0, True, 0.9, scores={"p_clean": 0.3, "max_p": 0.99}),
        Candidate(1, True, 0.8, scores={"p_clean": 0.7, "max_p": 0.7}),
        Candidate(2, True, None, scores={"p_clean": 0.99, "max_p": 0.99}),
        Candidate(3, False, 10, scores={"p_clean": 1, "max_p": 1}),
    ]
    audits = [select(pool, options, maximize=True, score_rule=rule)[1]
              for options, rule in [(Options(beta=0), "ucb"),
                                    (Options(), "ucb"), (Options(), "p_clean")]]
    assert [a["winner"] for a in audits] == [0, 1, 1]
    assert all([r["eligible"] for r in a["candidates"]] == [True, True, False, False]
               for a in audits)
    assert audits[2]["candidates"][1]["score"] == 0.7


@pytest.mark.parametrize("missing", [None, float("nan"), -0.1, 1.1, True])
@pytest.mark.parametrize("maximize,greedy", [(True, 0), (False, 1)])
def test_clean_argmax_missing_one_probability_falls_back_over_the_whole_pool(missing, maximize, greedy):
    pool = [Candidate(0, True, 0.9, scores={"p_clean": missing}),
            Candidate(1, True, 0.8, scores={"p_clean": 0.99})]
    winner, audit = select(pool, Options(beta=3), maximize=maximize, score_rule="p_clean")
    assert winner == greedy
    assert audit["fallback_reason"] == "missing_p_clean"
    assert audit["rule"] == "metric_greedy"


def test_clean_argmax_ties_preserve_generation_order():
    pool = [Candidate(0, True, 0.7, scores={"p_clean": 0.8}),
            Candidate(1, True, 0.9, scores={"p_clean": 0.8})]
    winner, audit = select(pool, Options(), maximize=True, score_rule="p_clean")
    assert winner == 0 and audit["greedy_winner"] == 1


def test_clean_argmax_no_metric_uses_the_shared_usable_first_salvage():
    pool = [Candidate(0, False, None, scores={"p_clean": 0.99}),
            Candidate(1, True, None, scores={"p_clean": 0.3}),
            Candidate(2, True, None, scores={"p_clean": 0.7})]
    for rule in ["ucb", "p_clean"]:
        winner, audit = select(pool, Options(), maximize=True, score_rule=rule)
        assert winner == 2 and audit["fallback_reason"] == "no_usable_metric"
