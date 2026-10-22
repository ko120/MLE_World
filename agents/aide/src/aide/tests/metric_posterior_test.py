"""Metric sampling contracts, UCB decisions, and deadline handling without a WM."""

import asyncio
import json
import statistics
import threading
import time
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import pytest
from omegaconf import OmegaConf

from aide import metric_posterior as mp
from aide import simulator as sim
from aide.agent import Agent, CandidateOutcome, STAGES
from aide.journal import Journal
from aide.tests.ucb_stage_policy_test import CODES, _agent
from aide.ucb import Candidate, Options, resolve_config, select
from aide.utils.config import AgentConfig


def choice(index, metric="0.8", exc="EXC_None", finish="stop"):
    return SimpleNamespace(
        index=index,
        text=(
            f"<exc_type>{exc}</exc_type>\n<term_out>validation log</term_out>\n"
            f"<val_metric>{metric}</val_metric>"
        ),
        finish_reason=finish,
    )


def posterior(values):
    return mp.summarize(
        [choice(i, str(value)) for i, value in enumerate(values)],
        mp.MetricPosteriorConfig(samples=len(values), min_valid=2),
    )


def attach_server(monkeypatch, handler):
    original = mp.openai.AsyncOpenAI

    def client(**kwargs):
        assert kwargs["max_retries"] == 0
        return original(
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)), **kwargs
        )

    monkeypatch.setattr(mp.openai, "AsyncOpenAI", client)
    monkeypatch.setattr(
        sim,
        "_sim_client",
        SimpleNamespace(
            base_url="http://test/wm/v1/",
            api_key="test-only",
            organization=None,
            project=None,
        ),
    )


def test_clean_numeric_distribution_and_failure_counts_are_separate():
    choices = [choice(i, str(i)) for i in range(8)]
    choices += [
        choice(8, "999", "EXC_TimeoutError"),
        choice(9, "None"),
        choice(10, "nan"),
        choice(11, "inf"),
        choice(12, "not numeric"),
        choice(13, "0.9", finish="length"),
        SimpleNamespace(index=14, text="<val_metric>0.9</val_metric>", finish_reason="stop"),
        choice(15, "None", "ValueError"),
    ]
    result = mp.summarize(choices, mp.MetricPosteriorConfig())
    assert result["status"] == "ok"
    assert result["mean"] == 3.5
    assert result["std"] == pytest.approx(statistics.stdev(range(8)))
    assert result["counts"] == dict(
        numeric=8,
        predicted_failure=2,
        none=1,
        nonfinite=2,
        malformed=2,
        truncated=1,
        unavailable=0,
    )
    assert result["samples"][0]["metric"] == 0.0
    assert all(row["metric"] is None for row in result["samples"][8:])
    json.dumps(result, allow_nan=False)


def test_insufficient_valid_and_incomplete_batches_abstain():
    cfg = mp.MetricPosteriorConfig()
    choices = [choice(i, "0.8" if i < 7 else "None") for i in range(16)]
    result = mp.summarize(choices, cfg)
    assert result["status"] == "insufficient_valid_samples"
    assert result["mean"] is result["std"] is None
    assert mp.summarize([choice(i) for i in range(8)], cfg)["status"] == "incomplete_response"
    assert mp.summarize([choice(0), choice(0)], cfg)["status"] == "invalid_response"
    assert mp.summarize([choice(16)], cfg)["status"] == "invalid_response"


def test_constant_samples_have_zero_uncertainty_and_plain_clean_is_supported():
    result = mp.summarize(
        [choice(i, "0", exc="None") for i in range(16)], mp.MetricPosteriorConfig()
    )
    assert result["status"] == "ok" and result["std"] == 0.0 and result["mean"] == 0.0


def test_extreme_numeric_samples_do_not_create_nonfinite_audits():
    result = posterior([1e308] * 8 + [-1e308] * 8)
    assert result["status"] == "nonfinite_statistics"
    assert result["std"] is None
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize(
    "key,value",
    [
        ("enabled", "true"),
        ("samples", 1),
        ("samples", True),
        ("samples", 2.5),
        ("min_valid", 1),
        ("min_valid", 17),
        ("max_tokens", 0),
        ("seed", -1),
        ("temperature", 0),
        ("temperature", float("nan")),
        ("temperature", True),
        ("timeout_s", 0),
        ("timeout_s", float("inf")),
    ],
)
def test_invalid_sampler_configuration_fails_at_startup(key, value):
    with pytest.raises(ValueError, match="agent.ucb.metric_posterior"):
        resolve_config({"metric_posterior": {key: value}}, "improve")


def test_flag_survives_structured_config_and_cli_merge():
    cfg = OmegaConf.merge(
        OmegaConf.structured(AgentConfig),
        OmegaConf.from_dotlist(["ucb.metric_posterior.enabled=true", "ucb.beta=0.5"]),
    )
    options = resolve_config(cfg.ucb, "improve")
    assert options.metric_posterior.enabled is True
    assert options.metric_posterior.samples == 16 and options.beta == 0.5
    cfg.ucb.metric_posterior.enabled = False
    assert resolve_config(cfg.ucb, "improve").metric_posterior.enabled is False
    cfg.ucb.metric_posterior.enabled = True
    cfg.ucb.selection_rule = "expected_gain"
    with pytest.raises(ValueError, match="requires selection_rule=ucb"):
        resolve_config(cfg.ucb, "improve")


@pytest.mark.parametrize(
    "prompt_format,clean",
    [
        ("env_v1_reorder_excfirst_token", "EXC_None"),
        ("env_v1_reorder_excfirst", "None"),
    ],
)
def test_sampler_uses_original_prompt_and_explicit_distribution(
    tmp_path, monkeypatch, prompt_format, clean
):
    calls = []
    monkeypatch.setenv("AIDE_SIMULATOR_PROMPT_FORMAT", prompt_format)
    monkeypatch.setenv(
        "AIDE_SIMULATOR_SAMPLING", '{"temperature":0,"n":1,"max_tokens":1,"top_p":0.1}'
    )
    interpreter = sim.SimulatorInterpreter(tmp_path, "wm", timeout=3400, environment="numpy 1.26.2")
    interpreter.set_data_overview("train has x and y")
    code = "print('Validation: 0.8')"

    async def handler(request):
        body = json.loads(request.content)
        calls.append(body)
        assert request.url.path == "/wm/v1/completions"
        assert body["prompt"] == sim._render_sim_prefix(
            interpreter.system_prompt, interpreter._user_message(code)
        )
        assert body["prompt"].endswith(sim._SIM_ASSISTANT_PREFIX)
        assert "<runtime_budget>3400s</runtime_budget>" in body["prompt"]
        assert "numpy 1.26.2" in body["prompt"] and "train has x and y" in body["prompt"]
        assert body["n"] == 16 and body["temperature"] == 1.0
        assert body["top_p"] == 1.0 and body["top_k"] == -1 and body["min_p"] == 0.0
        assert body["max_tokens"] == 8192 and body["repetition_penalty"] == 1.0
        assert body["presence_penalty"] == body["frequency_penalty"] == 0.0
        assert body["stop"] == ["</val_metric>"] and body["include_stop_str_in_output"]
        assert body["add_special_tokens"] is False
        assert "logprobs" not in body and not body.get("stream", False)
        return httpx.Response(
            200,
            json=dict(
                id="test",
                object="text_completion",
                created=0,
                model="wm",
                choices=[vars(choice(i, str(i / 10), clean)) for i in reversed(range(16))],
                usage=dict(prompt_tokens=100, completion_tokens=300, total_tokens=400),
            ),
        )

    attach_server(monkeypatch, handler)
    cfg = mp.MetricPosteriorConfig(enabled=True)
    result = interpreter.metric_posterior(code, config=cfg, deadline=time.monotonic() + 5)
    assert result["status"] == "ok"
    assert result["std"] == pytest.approx(statistics.stdev(i / 10 for i in range(16)))
    assert result["usage"]["completion_tokens"] == 300
    again = interpreter.metric_posterior(code, config=cfg, deadline=time.monotonic() + 5)
    assert again["sampling"]["seed"] == result["sampling"]["seed"]
    interpreter.metric_posterior(
        code + "\n# different input", config=cfg, deadline=time.monotonic() - 1
    )
    assert len(calls) == 2  # an expired deadline sends nothing


def test_request_error_is_not_an_execution_failure_and_is_not_retried(tmp_path, monkeypatch):
    calls = []

    async def handler(request):
        calls.append(request)
        return httpx.Response(503, json={"error": {"message": "unavailable"}})

    attach_server(monkeypatch, handler)
    interpreter = sim.SimulatorInterpreter(tmp_path, "wm", environment="numpy")
    result = interpreter.metric_posterior(
        "print(1)",
        config=mp.MetricPosteriorConfig(enabled=True),
        deadline=time.monotonic() + 5,
    )
    assert len(calls) == 1 and result["status"] == "request_error"
    assert result["counts"]["predicted_failure"] == 0
    assert result["counts"]["unavailable"] == 16


def test_wall_timeout_cancels_inflight_http(tmp_path, monkeypatch):
    cancelled = threading.Event()

    async def handler(request):
        try:
            await asyncio.sleep(10)
        finally:
            cancelled.set()

    attach_server(monkeypatch, handler)
    interpreter = sim.SimulatorInterpreter(tmp_path, "wm", environment="numpy")
    start = time.monotonic()
    result = interpreter.metric_posterior(
        "print(1)",
        config=mp.MetricPosteriorConfig(enabled=True, timeout_s=0.1),
        deadline=start + 5,
    )
    assert time.monotonic() - start < 2
    assert cancelled.is_set()
    assert result["status"] == "request_timeout" and result["std"] is None


@pytest.mark.parametrize("maximize", [True, False])
@pytest.mark.parametrize("units", [1, 100])
def test_ucb_uses_shared_scale_and_direction_without_changing_the_point_mean(maximize, units):
    sign = 1 if maximize else -1
    candidates = [
        Candidate(0, True, sign * 0.9 * units, metric_posterior=posterior([0.1 * units] * 16)),
        Candidate(
            1, True, sign * 0.85 * units, metric_posterior=posterior([0] * 8 + [0.4 * units] * 8)
        ),
    ]
    options = Options(
        beta=0.5,
        metric_scale=2 * units,
        metric_posterior=mp.MetricPosteriorConfig(enabled=True),
    )
    picked, audit = select(
        candidates, options, maximize=maximize, reference_metric=sign * 0.8 * units
    )
    assert picked == 1 and audit["greedy_winner"] == 0
    rows = audit["candidates"]
    assert [r["mu"] for r in rows] == pytest.approx([0.05, 0.025])
    assert rows[1]["sigma"] == pytest.approx(statistics.stdev([0] * 8 + [0.4] * 8) / 2)
    assert rows[1]["score"] == pytest.approx(rows[1]["mu"] + 0.5 * rows[1]["sigma"])
    assert select(candidates, replace(options, beta=0), maximize=maximize)[0] == 0
    assert candidates[0].metric == sign * 0.9 * units


def test_missing_metric_posterior_disables_bonus_for_the_whole_pool():
    candidates = [
        Candidate(0, True, 0.9, scores={"max_p": 0.1}),
        Candidate(1, True, 0.8, metric_posterior=posterior([0] * 8 + [100] * 8)),
    ]
    picked, audit = select(
        candidates,
        Options(metric_posterior=mp.MetricPosteriorConfig(enabled=True)),
        maximize=True,
    )
    assert picked == 0 and audit["fallback_reason"] == "missing_uncertainty"
    assert all(row["bonus"] == 0 for row in audit["candidates"])


def metric_agent(tmp_path, *, stage="improve", beta=1):
    agent, exc, predictions = _agent(tmp_path, stage, beta=beta)
    agent.acfg.ucb.metric_posterior.enabled = True
    values = {
        code: posterior([0] * 8 + ([1] * 8 if i == 2 else [0] * 8))
        for i, (_, code) in enumerate(CODES)
    }
    sampler = Mock(side_effect=lambda code, **kwargs: values[code])
    agent._metric_posterior = sampler
    agent._resolve_stage_policies()
    return agent, exc, predictions, sampler


@pytest.mark.parametrize("stage", STAGES)
def test_flag_on_selects_by_metric_uncertainty_and_preserves_validation_confidence(tmp_path, stage):
    agent, exc, _, sampler = metric_agent(tmp_path, stage=stage)
    assert agent.generate_code({}, stage)[1] == CODES[2][1]
    assert sampler.call_count == 4
    exc.assert_not_called()
    audit = json.loads((tmp_path / "bon/step_0/ucb.json").read_text())
    assert audit["winner"] == 2 and audit["greedy_winner"] == 0
    assert audit["uncertainty_kind"] == "sampled_validation_metric_std"
    assert all(row["metric_posterior"]["status"] == "ok" for row in audit["candidates"])
    assert len({call.kwargs["deadline"] for call in sampler.call_args_list}) == 1
    # Validation confidence still comes from exceptions for the actual winner.
    assert agent._selected_confidence(CODES[2][1]) == (0.99, "TimeoutError", 0.005)
    exc.assert_called_once_with(CODES[2][1])


@pytest.mark.parametrize(
    "enabled,beta,winner,exc_calls", [(False, 1, 1, 4), (False, 0, 0, 0), (True, 0, 0, 0)]
)
def test_disabled_or_zero_beta_makes_no_metric_sampling_calls(
    tmp_path, enabled, beta, winner, exc_calls
):
    agent, exc, _, sampler = metric_agent(tmp_path, beta=beta)
    agent.acfg.ucb.metric_posterior.enabled = enabled
    assert agent.generate_code({}, "improve")[1] == CODES[winner][1]
    sampler.assert_not_called()
    assert exc.call_count == exc_calls


@pytest.mark.parametrize("fault", ["failure", "timeout", "no_submission", "missing_metric"])
def test_ineligible_predictions_are_never_sampled_or_rescued(tmp_path, fault):
    agent, _, predictions, sampler = metric_agent(tmp_path)
    bad = predictions[CODES[2][1]]
    if fault in ("failure", "timeout"):
        bad.exc_type = "TimeoutError" if fault == "timeout" else "ValueError"
    elif fault == "no_submission":
        bad.predicted_submission_exists = "false"
    else:
        bad.predicted_metric = None
    assert agent.generate_code({}, "improve")[1] == CODES[0][1]
    assert CODES[2][1] not in [call.args[0] for call in sampler.call_args_list]


def test_single_eligible_and_unknown_direction_pools_skip_sampling(tmp_path):
    agent, _, predictions, sampler = metric_agent(tmp_path)
    for _, code in CODES[1:]:
        predictions[code].predicted_metric = None
    assert agent.generate_code({}, "improve")[1] == CODES[0][1]
    sampler.assert_not_called()
    agent.acfg.eval_lower_is_better = None
    agent.generate_code({}, "improve")
    sampler.assert_not_called()


def test_all_failed_pool_still_uses_exception_salvage(tmp_path):
    agent, exc, predictions, sampler = metric_agent(tmp_path)
    for result in predictions.values():
        result.exc_type = "ValueError"
        result.predicted_submission_exists = "false"
        result.predicted_metric = None
    agent.generate_code({}, "improve")
    sampler.assert_not_called()
    assert exc.call_count == 4
    audit = json.loads((tmp_path / "bon/step_0/ucb.json").read_text())
    assert audit["fallback_reason"] == "no_usable_metric"
    assert audit["rule"] == "p_clean"


def test_real_race_observations_get_zero_sigma_and_rejected_reviews_stay_ineligible(tmp_path):
    agent, exc, predictions, sampler = metric_agent(tmp_path)
    outcomes = [CandidateOutcome(predictions[code], True, "true", True) for _, code in CODES]
    outcomes[2].result.real_won = True
    outcomes[2].review = {"metric": 0.7}
    outcomes[1].review = {"metric_trustworthy": False}
    assert agent._select_ucb(CODES, outcomes, tmp_path, "improve", None) == 0
    assert {call.args[0] for call in sampler.call_args_list} == {CODES[0][1], CODES[3][1]}
    audit = json.loads((tmp_path / "ucb.json").read_text())
    assert audit["candidates"][2]["sigma"] == 0
    assert audit["candidates"][1]["eligible"] is False
    exc.assert_not_called()


def test_sampling_failure_is_audited_and_falls_back_to_greedy(tmp_path):
    agent, exc, _, sampler = metric_agent(tmp_path)
    sampler.side_effect = RuntimeError("offline")
    assert agent.generate_code({}, "improve")[1] == CODES[0][1]
    audit = json.loads((tmp_path / "bon/step_0/ucb.json").read_text())
    assert audit["fallback_reason"] == "missing_uncertainty"
    assert all(row["metric_posterior"]["status"] == "request_error" for row in audit["candidates"])
    exc.assert_not_called()


def test_shutdown_reserve_prevents_sampling(tmp_path, monkeypatch):
    agent, _, _, sampler = metric_agent(tmp_path)
    monkeypatch.setenv("AIDE_SHUTDOWN_RESERVE_S", "180")
    agent.deadline = time.time() + 100
    assert agent.generate_code({}, "improve")[1] == CODES[0][1]
    sampler.assert_not_called()
    audit = json.loads((tmp_path / "bon/step_0/ucb.json").read_text())
    assert all(
        row["metric_posterior"]["status"] == "deadline_exhausted" for row in audit["candidates"]
    )


def test_scoring_workers_share_budget_including_queued_candidates(tmp_path, monkeypatch):
    agent, _, _, sampler = metric_agent(tmp_path)
    agent.acfg.ucb.score_workers = 1
    agent.acfg.ucb.metric_posterior.timeout_s = 0.1
    real_clock = time.monotonic
    now = real_clock()
    monkeypatch.setattr(time, "monotonic", lambda: now)

    def consume_budget(code, **kwargs):
        nonlocal now
        now = kwargs["deadline"] + 0.01
        return mp.unavailable(kwargs["config"], "request_timeout")

    sampler.side_effect = consume_budget
    assert agent.generate_code({}, "improve")[1] == CODES[0][1]
    assert sampler.call_count == 1
    audit = json.loads((tmp_path / "bon/step_0/ucb.json").read_text())
    assert [row["metric_posterior"]["status"] for row in audit["candidates"]] == [
        "request_timeout",
        "deadline_exhausted",
        "deadline_exhausted",
        "deadline_exhausted",
    ]


def test_metric_scorer_is_required_at_startup_when_enabled(tmp_path):
    agent, exc, _, sampler = metric_agent(tmp_path)
    agent._metric_posterior = None
    with pytest.raises(ValueError, match="metric-posterior scorer"):
        agent._resolve_stage_policies()
    # The metric selector can run without an exception scorer when no confidence
    # gate requests one; salvage then keeps the existing first-usable fallback.
    Agent(
        "task",
        agent.cfg,
        Journal(),
        sim_predict=agent.sim_predict,
        sim_p_clean=None,
        sim_metric_posterior=sampler,
    )
