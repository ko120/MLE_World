"""Sampled LM predictive variation in validation metrics, conditional on clean output.

Samples start at the original assistant prefix, before the execution prediction.
The reported deviation is a sample standard deviation (ddof=1), not a standard
error or a calibrated Bayesian confidence interval.
"""

import asyncio
import hashlib
import math
import re
import time
from collections.abc import Mapping
from dataclasses import dataclass
from statistics import fmean, stdev

import httpx
import openai


@dataclass
class MetricPosteriorConfig:
    enabled: bool = False
    samples: int = 16
    min_valid: int = 8
    temperature: float = 1.0
    max_tokens: int = 8192
    # Total wall budget for one pool, including queued requests.
    timeout_s: float = 60.0
    seed: int = 0


def resolve_config(config) -> MetricPosteriorConfig:
    defaults = MetricPosteriorConfig()
    values = {
        name: (
            config.get(name, default)
            if isinstance(config, Mapping)
            else getattr(config, name, default)
        )
        for name, default in vars(defaults).items()
    }
    prefix = "agent.ucb.metric_posterior"
    if not isinstance(values["enabled"], bool):
        raise ValueError(f"{prefix}.enabled must be a boolean")
    for name, minimum in (("samples", 2), ("min_valid", 2), ("max_tokens", 1), ("seed", 0)):
        value = values[name]
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"{prefix}.{name} must be an integer >= {minimum}")
    if values["min_valid"] > values["samples"]:
        raise ValueError(f"{prefix}.min_valid must not exceed samples")
    for name in ("temperature", "timeout_s"):
        value = values[name]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value <= 0
        ):
            raise ValueError(f"{prefix}.{name} must be finite and positive")
    return MetricPosteriorConfig(**values)


_METRIC_RESULT_RE = re.compile(
    r"\A\s*<exc_type>\s*(?P<exc>[A-Za-z_][\w.]*)\s*</exc_type>\s*"
    r"<term_out>.*?</term_out>\s*"
    r"<val_metric>\s*(?P<metric>.*?)\s*</val_metric>\s*\Z",
    re.S,
)
_SAMPLE_KINDS = (
    "numeric",
    "predicted_failure",
    "none",
    "nonfinite",
    "malformed",
    "truncated",
    "unavailable",
)


def unavailable(config, status: str) -> dict:
    """Missing inference is never counted as a predicted execution failure."""
    return {
        "status": status,
        "requested_samples": config.samples,
        "received_samples": 0,
        "counts": {kind: config.samples if kind == "unavailable" else 0 for kind in _SAMPLE_KINDS},
        "samples": [],
        "mean": None,
        "std": None,
        "ddof": 1,
    }


def summarize(choices, config) -> dict:
    result = unavailable(config, "insufficient_valid_samples")
    indices = [getattr(choice, "index", None) for choice in choices]
    if (
        len(indices) > config.samples
        or any(type(index) is not int or not 0 <= index < config.samples for index in indices)
        or len(set(indices)) != len(indices)
    ):
        result["status"] = "invalid_response"
        return result
    numeric = []
    for choice in sorted(choices, key=lambda item: item.index):
        text = getattr(choice, "text", None)
        finish = getattr(choice, "finish_reason", None)
        match = _METRIC_RESULT_RE.fullmatch(text) if isinstance(text, str) else None
        value = None
        exc = match.group("exc") if match else None
        if finish == "length":
            kind = "truncated"
        elif finish != "stop" or match is None:
            kind = "malformed"
        elif exc not in {"None", "EXC_None"}:
            kind = "predicted_failure"
        elif match.group("metric").strip() == "None":
            kind = "none"
        else:
            try:
                value = float(match.group("metric"))
                kind = "numeric" if math.isfinite(value) else "nonfinite"
            except (ValueError, OverflowError):
                kind = "malformed"
            if kind != "numeric":
                value = None
        if kind == "numeric":
            numeric.append(value)
        result["counts"][kind] += 1
        result["counts"]["unavailable"] -= 1
        result["samples"].append(
            {
                "index": choice.index,
                "kind": kind,
                "metric": value,
                "exc_type": exc,
                "finish_reason": finish,
            }
        )
    result["received_samples"] = len(choices)
    # A partial batch is a failed request even if it contains min_valid numbers.
    if len(choices) != config.samples:
        result["status"] = "incomplete_response"
    elif len(numeric) >= config.min_valid:
        try:
            mean, std = fmean(numeric), stdev(numeric)
        except (ValueError, OverflowError):
            mean = std = math.inf
        if math.isfinite(mean) and math.isfinite(std):
            result.update(status="ok", mean=mean, std=std)
        else:
            result["status"] = "nonfinite_statistics"
    return result


async def _complete(client, params, deadline):
    # Separate async transport lets wait_for cancel an in-flight HTTP request,
    # including a server that keeps sending bytes without completing. HTTP read
    # timeouts alone only bound inactivity. No retries or orphan worker request.
    async with openai.AsyncOpenAI(
        base_url=client.base_url,
        api_key=client.api_key,
        organization=client.organization,
        project=client.project,
        max_retries=0,
        timeout=max(0.001, deadline - time.monotonic()),
    ) as sampler:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise asyncio.TimeoutError
        return await asyncio.wait_for(sampler.completions.create(**params), timeout=remaining)


def sample(client, *, model: str, prefix: str, config, deadline: float) -> dict:
    """Sample a complete response batch with a shared monotonic pool deadline.

    Called from the selector's bounded worker pool. The existing deterministic
    simulator sampling overrides deliberately do not apply to this distribution.
    """
    start = time.monotonic()
    remaining = min(config.timeout_s, deadline - start)
    seed = (config.seed + int(hashlib.sha256(prefix.encode()).hexdigest()[:8], 16)) % (2**31)
    sampling = dict(
        n=config.samples,
        temperature=config.temperature,
        top_p=1.0,
        max_tokens=config.max_tokens,
        seed=seed,
        presence_penalty=0.0,
        frequency_penalty=0.0,
        stop=["</val_metric>"],
        extra_body={
            "top_k": -1,
            "min_p": 0.0,
            "repetition_penalty": 1.0,
            "add_special_tokens": False,
            "include_stop_str_in_output": True,
        },
    )
    if remaining <= 0:
        result = unavailable(config, "deadline_exhausted")
    else:
        try:
            response = asyncio.run(
                _complete(client, dict(model=model, prompt=prefix, **sampling), start + remaining)
            )
            result = summarize(response.choices, config)
            if response.usage is not None:
                result["usage"] = {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                }
        except (asyncio.TimeoutError, openai.APITimeoutError, httpx.TimeoutException):
            result = unavailable(config, "request_timeout")
        except Exception as exc:
            result = unavailable(config, "request_error")
            # Keep audit JSON small and avoid recording HTTP headers/credentials.
            result["error_type"] = type(exc).__name__
    result.update(
        scoring_seconds=time.monotonic() - start,
        sampling=sampling,
        prompt_sha256=hashlib.sha256(prefix.encode()).hexdigest(),
        method="full_response_sampling",
    )
    return result
