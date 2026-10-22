"""Fixed-pool optimistic selection using world-model uncertainty proxies.

No fitting or test metrics: sigma defaults to an exception-confidence proxy.
An opt-in sampler estimates validation-metric predictive standard deviation.
Kept independent of Agent so other candidate generators can use the same rule.
"""

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from statistics import median

from . import metric_posterior as metric_sampling


@dataclass
class UCBStageConfig:
    """None inherits the corresponding shared setting."""

    sigma: str | None = None
    beta: float | None = None
    metric_scale: float | None = None
    normalize_mu: bool | None = None
    ideal_metric: float | None = None
    scale_floor: float | None = None
    selection_rule: str | None = None


@dataclass
class UCBConfig:
    sigma: str = "p_max"
    beta: float = 1.0
    # None: abs(best validated real metric, else parent), else median abs(pool metric),
    # floored at 0.01. A fixed positive value overrides that normalization.
    metric_scale: float | None = None
    score_workers: int = 4
    draft: UCBStageConfig = field(default_factory=UCBStageConfig)
    debug: UCBStageConfig = field(default_factory=UCBStageConfig)
    improve: UCBStageConfig = field(default_factory=UCBStageConfig)
    # Opt in to improvement / remaining gap to ideal. Fixed metric_scale wins.
    normalize_mu: bool = False
    ideal_metric: float | None = None  # None: 1 for maximize, 0 for minimize.
    scale_floor: float = 0.01
    selection_rule: str = "ucb"
    # Shared across UCB stages. Defaults preserve the original selector.
    all_failed_fallback: str = "p_clean"  # p_clean | judge
    pool_policy: str = "review_filtered"  # review_filtered | bon
    metric_posterior: metric_sampling.MetricPosteriorConfig = field(
        default_factory=metric_sampling.MetricPosteriorConfig
    )


@dataclass(frozen=True)
class Options:
    sigma: str = "p_max"
    beta: float = 1.0
    metric_scale: float | None = None
    score_workers: int = 4
    normalize_mu: bool = False
    ideal_metric: float | None = None
    scale_floor: float = 0.01
    selection_rule: str = "ucb"
    all_failed_fallback: str = "p_clean"
    pool_policy: str = "review_filtered"
    metric_posterior: metric_sampling.MetricPosteriorConfig = field(
        default_factory=metric_sampling.MetricPosteriorConfig
    )


def finite_number(value) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def _get(config, key, default=None):
    return (
        config.get(key, default) if isinstance(config, Mapping) else getattr(config, key, default)
    )


def resolve_config(config, stage: str) -> Options:
    """Resolve shared settings and stage overrides; reject bad runs at startup."""
    if stage not in {"draft", "debug", "improve"}:
        raise ValueError(f"Unknown UCB stage: {stage!r}")
    defaults = Options()
    overlay = _get(config, stage)
    values = {}
    for key in ("sigma", "beta", "metric_scale", "normalize_mu", "ideal_metric", "scale_floor", "selection_rule"):
        override = _get(overlay, key)
        values[key] = (
            override if override is not None else _get(config, key, getattr(defaults, key))
        )
    # The posterior client spells this max_p; accept both config spellings.
    if values["sigma"] == "max_p":
        values["sigma"] = "p_max"
    if values["sigma"] not in {"p_clean", "entropy", "p_max"}:
        raise ValueError(f"agent.ucb.{stage}.sigma must be p_clean, entropy, or p_max")
    if values["selection_rule"] not in {"ucb", "expected_gain"}:
        raise ValueError(f"agent.ucb.{stage}.selection_rule must be ucb or expected_gain")
    beta = finite_number(values["beta"])
    if beta is None or beta < 0:
        raise ValueError(f"agent.ucb.{stage}.beta must be finite and nonnegative")
    scale = finite_number(values["metric_scale"])
    if values["metric_scale"] is not None and (scale is None or scale <= 0):
        raise ValueError(f"agent.ucb.{stage}.metric_scale must be null or finite and positive")
    if not isinstance(values["normalize_mu"], bool):
        raise ValueError(f"agent.ucb.{stage}.normalize_mu must be a boolean")
    ideal = finite_number(values["ideal_metric"])
    if values["ideal_metric"] is not None and ideal is None:
        raise ValueError(f"agent.ucb.{stage}.ideal_metric must be null or finite")
    floor = finite_number(values["scale_floor"])
    if floor is None or floor <= 0:
        raise ValueError(f"agent.ucb.{stage}.scale_floor must be finite and positive")
    workers = _get(config, "score_workers", defaults.score_workers)
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ValueError("agent.ucb.score_workers must be a positive integer")
    fallback = _get(config, "all_failed_fallback", defaults.all_failed_fallback)
    if fallback not in {"p_clean", "judge"}:
        raise ValueError("agent.ucb.all_failed_fallback must be p_clean or judge")
    pool_policy = _get(config, "pool_policy", defaults.pool_policy)
    if pool_policy not in {"review_filtered", "bon"}:
        raise ValueError("agent.ucb.pool_policy must be review_filtered or bon")
    posterior = metric_sampling.resolve_config(_get(config, "metric_posterior"))
    if posterior.enabled and values["selection_rule"] != "ucb":
        raise ValueError("agent.ucb.metric_posterior.enabled requires selection_rule=ucb")
    return Options(
        sigma=values["sigma"],
        beta=beta,
        metric_scale=scale,
        score_workers=workers,
        normalize_mu=values["normalize_mu"],
        ideal_metric=ideal,
        scale_floor=floor,
        selection_rule=values["selection_rule"],
        all_failed_fallback=fallback,
        pool_policy=pool_policy,
        metric_posterior=posterior,
    )


def uncertainty(scores, source: str) -> float | None:
    """1-p_clean, 1-max_p, or Shannon entropy / log(support size), in [0, 1]."""
    if not isinstance(scores, Mapping):
        scores = {"p_clean": scores}  # legacy scalar hooks contain only P(clean)
    if source in {"p_clean", "p_max"}:
        key = "p_clean" if source == "p_clean" else "max_p"
        p = finite_number(scores.get(key))
        return 1.0 - p if p is not None and 0 <= p <= 1 else None
    if source != "entropy":
        raise ValueError(f"Unknown UCB sigma source: {source!r}")
    entropy = finite_number(scores.get("entropy"))
    size = finite_number(scores.get("support_size"))
    if entropy is None or size is None or size < 2 or not size.is_integer():
        return None
    normalized = entropy / math.log(size)
    if not 0 <= normalized <= 1 + 1e-12:
        return None
    return min(normalized, 1.0)


@dataclass
class Candidate:
    idx: int
    usable: bool
    metric: float | None
    observed: bool = False
    scores: dict | float | None = None
    metric_posterior: dict | None = None


def select(
    candidates: list[Candidate],
    options: Options,
    *,
    maximize: bool | None,
    reference_metric: float | None = None,
    score_rule: str = "ucb",
) -> tuple[int, dict]:
    """Return one index and a JSON-safe audit. Ties preserve candidate order.

    Missing metrics use P(clean) salvage; missing sigma on ANY eligible
    prediction makes the whole comparison metric-greedy. A real observation
    has zero WM prediction uncertainty and needs no posterior scoring.
    """
    if not candidates:
        raise ValueError("UCB needs at least one candidate")
    if score_rule == "ucb":
        score_rule = options.selection_rule
    if score_rule not in {"ucb", "p_clean", "expected_gain"}:
        raise ValueError(f"Unknown candidate score rule: {score_rule!r}")
    use_metric_posterior = options.metric_posterior.enabled and score_rule == "ucb"
    rows = []
    for candidate in candidates:
        scores = candidate.scores
        if not isinstance(scores, Mapping):
            scores = {"p_clean": scores}
        metric = finite_number(candidate.metric)
        rows.append(
            {
                "idx": candidate.idx,
                "usable": candidate.usable,
                "metric": metric,
                "observed": candidate.observed,
                "eligible": candidate.usable and metric is not None,
                "top_exc": scores.get("top_exc"),
                "p_top_exc": finite_number(scores.get("p_top_exc")),
                "p_clean": finite_number(scores.get("p_clean")),
                "p_max": finite_number(scores.get("max_p")),
                "entropy": finite_number(scores.get("entropy")),
                "support_size": finite_number(scores.get("support_size")),
                "sigma": 0.0 if candidate.observed else uncertainty(scores, options.sigma),
                "mu": None,
                "bonus": None,
                "score": None,
            }
        )
        if use_metric_posterior:
            # Keep the original point metric (and eligibility) as the mean.
            # Numeric failures/None never become zeros in this distribution.
            rows[-1]["metric_posterior"] = candidate.metric_posterior
            rows[-1]["sigma"] = 0.0 if candidate.observed else None
    eligible = [r for r in rows if r["eligible"]]
    reference = finite_number(reference_metric)
    scale = options.metric_scale
    normalization = "fixed" if scale is not None else None
    ideal = None
    if scale is None and (reference is not None or eligible):
        normalization = "reference" if reference is not None else "pool"
        basis = (
            abs(reference) if reference is not None else median(abs(r["metric"]) for r in eligible)
        )
        if options.normalize_mu and reference is not None and isinstance(maximize, bool):
            ideal = options.ideal_metric if options.ideal_metric is not None else float(maximize)
            gap = finite_number(ideal - reference)
            if gap is not None:
                basis = abs(gap)
                normalization = "remaining_gap"
        scale = max(basis, options.scale_floor)
    if use_metric_posterior and scale is not None:
        for row in eligible:
            posterior = row["metric_posterior"]
            if row["observed"] or not isinstance(posterior, Mapping):
                continue
            std = finite_number(posterior.get("std"))
            if posterior.get("status") == "ok" and std is not None and std >= 0:
                row["sigma"] = finite_number(std / scale)
    audit = {
        "options": asdict(options),
        "score_rule": score_rule,
        "maximize": maximize,
        "reference_metric": reference,
        "metric_scale": scale,
        "mu_normalization": normalization,
        "ideal_metric": ideal,
        "candidates": rows,
        "greedy_winner": None,
        "fallback_reason": None,
    }
    if eligible and isinstance(maximize, bool):
        sign = 1.0 if maximize else -1.0
        for row in eligible:
            row["mu"] = sign * (row["metric"] / scale - (reference or 0.0) / scale)
        # A tiny configured scale must not inject infinities into the ranking.
        if all(math.isfinite(r["mu"]) for r in eligible):
            greedy = max(eligible, key=lambda r: r["mu"])["idx"]
            audit["greedy_winner"] = greedy
            if score_rule == "expected_gain":
                known = all(r["p_clean"] is not None and 0 <= r["p_clean"] <= 1 for r in eligible)
                positive = any(r["mu"] > 0 for r in eligible)
                if known and positive:
                    for row in eligible:
                        row["bonus"] = 0.0
                        row["score"] = row["p_clean"] * max(0.0, row["mu"])
                    picked = max(eligible, key=lambda r: (r["score"], r["mu"]))["idx"]
                    audit.update(winner=picked, rule="expected_gain")
                else:
                    for row in eligible:
                        row["bonus"], row["score"] = 0.0, row["mu"]
                    audit.update(winner=greedy, rule="metric_greedy",
                                 fallback_reason="missing_p_clean" if not known else "no_predicted_improvement")
                return audit["winner"], audit
            if score_rule == "p_clean":
                # The A arm changes only the score. It uses the same metric
                # eligibility, direction, ties, and whole-pool fallback as G/P.
                use_clean = len(eligible) > 1 and all(
                    r["p_clean"] is not None and 0 <= r["p_clean"] <= 1 for r in eligible
                )
                if len(eligible) > 1 and not use_clean:
                    audit["fallback_reason"] = "missing_p_clean"
                for row in eligible:
                    row["bonus"] = 0.0
                    row["score"] = row["p_clean"] if use_clean else row["mu"]
                picked = max(eligible, key=lambda r: r["score"])["idx"]
                audit.update(winner=picked, rule="p_clean_argmax" if use_clean else "metric_greedy")
                return picked, audit
            use_bonus = options.beta > 0 and len(eligible) > 1
            if use_bonus and any(r["sigma"] is None for r in eligible):
                use_bonus = False
                audit["fallback_reason"] = "missing_uncertainty"
            for row in eligible:
                row["bonus"] = options.beta * row["sigma"] if use_bonus else 0.0
                row["score"] = row["mu"] + row["bonus"]
            if all(math.isfinite(r["score"]) for r in eligible):
                picked = max(eligible, key=lambda r: r["score"])["idx"]
                audit.update(winner=picked, rule="ucb" if use_bonus else "metric_greedy")
                return picked, audit
        # Exceptional numeric scales degrade explicitly, never emit NaN JSON.
        for row in eligible:
            row.update(mu=None, bonus=None, score=None)
        audit["fallback_reason"] = "nonfinite_utility"
    else:
        audit["fallback_reason"] = "unknown_metric_direction" if eligible else "no_usable_metric"

    # Crashes/absent metrics have no meaningful optimistic metric. Prefer a
    # gate-passing candidate, then probability of success, then input order.
    pool = [r for r in rows if r["usable"]] or rows
    scored = [r for r in pool if r["p_clean"] is not None and 0 <= r["p_clean"] <= 1]
    picked = max(scored, key=lambda r: r["p_clean"])["idx"] if scored else pool[0]["idx"]
    rule = "p_clean" if scored else ("first_usable" if pool[0]["usable"] else "first_candidate")
    audit.update(winner=picked, rule=rule)
    return picked, audit
