"""Outcome thresholds and reproducible random real-check controls."""

from dataclasses import dataclass
import hashlib
import json
import math

@dataclass
class ExecutionGateConfig:
    enabled: bool = False
    # When enabled, None means always execute this outcome except timeouts.
    success_threshold: float | None = None
    failure_threshold: float | None = None
    mode: str = "confidence"
    # One probability per method profile, shared across all stages and outcomes.
    random_check_probability: float | None = None
    random_seed: int = 0


def outcome_gate_enabled(acfg) -> bool:
    return bool(getattr(getattr(acfg, "execution_gate", None), "enabled", False))


def random_gate_enabled(acfg) -> bool:
    return outcome_gate_enabled(acfg) and getattr(acfg.execution_gate, "mode", "confidence") == "random"


def random_gate_metadata(acfg) -> dict:
    return {"gate_check_probability": float(acfg.execution_gate.random_check_probability),
            "gate_random_seed": acfg.execution_gate.random_seed}


def random_gate_decision(acfg, node, *, task, trial_seed, method) -> tuple[bool, dict]:
    """One draw per committed step, reproducible after resume without global RNG state.

    The caller handles mandatory checks, missing confidence, and vetoes first.
    Neither confidence, code, nor the check probability enters the draw's key.
    """
    if type(node.step) is not int or node.step < 0:
        raise ValueError("random execution gate requires a committed node.step")
    probability = float(acfg.execution_gate.random_check_probability)
    key = json.dumps(["real-check-v1", acfg.execution_gate.random_seed,
                      str(trial_seed), str(task), method, node.step], separators=(",", ":"))
    digest = hashlib.sha256(key.encode()).digest()
    draw = (int.from_bytes(digest[:8], "big") >> 11) / (1 << 53)
    return draw < probability, {
        "gate_check_probability": probability, "gate_random_draw": draw,
        "gate_random_key": digest.hex(), "gate_random_step": node.step,
    }


def outcome_gate_threshold(acfg, exc_type) -> float | None:
    """Choose only by the simulated exception, never reviewer or stage labels."""
    field = "success_threshold" if exc_type is None else "failure_threshold"
    return getattr(acfg.execution_gate, field)


def outcome_gate_requires_real(acfg, exc_type, confidence) -> bool:
    if not outcome_gate_enabled(acfg) or exc_type == "TimeoutError":
        return False
    threshold = outcome_gate_threshold(acfg, exc_type)
    if threshold is None:
        return True
    confidence = outcome_gate_confidence(acfg, exc_type, confidence)
    if (isinstance(confidence, bool) or not isinstance(confidence, (int, float))
            or not math.isfinite(confidence) or not 0 <= confidence <= 1):
        return True
    return confidence < threshold


def outcome_gate_confidence(acfg, exc_type, confidence):
    if (type(confidence) not in (int, float) or not math.isfinite(confidence)
            or not 0 <= confidence <= 1):
        return None
    return confidence


def validate_execution_gate(acfg) -> None:
    if not outcome_gate_enabled(acfg):
        return
    gate = acfg.execution_gate
    mode = getattr(gate, "mode", "confidence")
    if mode not in {"confidence", "random"}:
        raise ValueError("agent.execution_gate.mode must be confidence or random")
    for field in ("success_threshold", "failure_threshold"):
        value = getattr(acfg.execution_gate, field)
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or not 0 <= value <= 1
        ):
            raise ValueError(f"agent.execution_gate.{field} must be null or in [0, 1]")
    if mode == "random":
        probability = gate.random_check_probability
        if type(probability) not in (int, float) or not math.isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError("agent.execution_gate.random_check_probability must be in [0, 1]")
        if type(gate.random_seed) is not int or gate.random_seed < 0:
            raise ValueError("agent.execution_gate.random_seed must be a nonnegative integer")
