"""Exercise the served-tokenizer/forced-logprob contract through the real SDK."""

import json
import math
from types import SimpleNamespace

import httpx
import openai
import pytest
from aide import simulator as sim

BASE = [101, 102, 103]


class ScoringServer:
    def __init__(self, fault=None):
        self.fault = fault
        self.suffixes = [[200 + i, *([600] * (i % 3)), 700] for i in range(24)]
        self.logps = [[-0.01 * i, *([-0.3] * (i % 3)), -abs(i - 3)] for i in range(24)]
        self.calls = []
        self.warmed = False

    def handle(self, request):
        body = json.loads(request.content)
        self.calls.append((request.url.path, body))
        if request.url.path == "/wm/tokenize":
            assert body["add_special_tokens"] is False
            assert "<runtime_budget>3400s</runtime_budget>" in body["prompt"]
            assert body["return_token_strs"] is True
            assert body["prompt"].count("<runtime_budget>3400s</runtime_budget>") == 1
            assert body["prompt"].count(sim._SIM_ASSISTANT_PREFIX) == 24
            # Include a special-token occurrence in the input before our 24
            # packed assistant segments; unpacking must preserve it.
            ids, token_strs = [BASE[0]], ["<|im_start|>"]
            for suffix in self.suffixes:
                ids.extend([*BASE[1:], *suffix])
                token_strs.extend(["<|im_start|>", "assistant", *[str(t) for t in suffix]])
            if self.fault == "marker":
                token_strs = ["ordinary"] * len(ids)
            return httpx.Response(
                200,
                json={
                    "tokens": ids,
                    "token_strs": token_strs,
                    "count": len(ids),
                    "max_model_len": 2 if self.fault == "context" else 7,
                },
            )
        assert request.url.path == "/wm/v1/completions"
        assert body["prompt"] == BASE
        assert body["logprobs"] == 0 and not body.get("echo", False)
        assert body["ignore_eos"] and body["return_tokens_as_token_ids"]
        suffix = body["vllm_xargs"]["forced_token_ids"]
        i = self.suffixes.index(suffix)
        assert body["max_tokens"] == len(suffix)
        if i == 0:
            self.warmed = True
        else:
            assert self.warmed, "remaining branches must follow cache warmup"
        if self.fault == "backend":
            return httpx.Response(503, json={"error": {"message": "offline"}})
        tokens = [f"token_id:{t}" for t in suffix]
        values = list(self.logps[i])
        if self.fault == "ignored":
            tokens[-1] = "token_id:999"
        elif self.fault == "missing":
            values[-1] = None
        elif self.fault == "truncated":
            values.pop()
        return httpx.Response(
            200,
            json={
                "id": "cmpl-test",
                "object": "text_completion",
                "created": 0,
                "model": "wm",
                "choices": [
                    {
                        "index": 0,
                        "text": "unused",
                        "finish_reason": "length",
                        "logprobs": {
                            "tokens": tokens,
                            "token_logprobs": values,
                            "text_offset": list(range(len(tokens))),
                        },
                    }
                ],
            },
        )


@pytest.fixture
def interpreter(tmp_path, monkeypatch):
    # This fixture exercises the retained 24-class legacy protocol. The paper
    # default is the 82-class single-token protocol, tested separately.
    monkeypatch.setenv("AIDE_SIMULATOR_PROMPT_FORMAT", "env_v1_reorder_excfirst")
    monkeypatch.delenv("AIDE_EXC_SCORING", raising=False)
    monkeypatch.delenv("AIDE_PCLEAN_SCORE", raising=False)
    monkeypatch.setenv("AIDE_EXC_SCORE_WORKERS", "3")
    return sim.SimulatorInterpreter(tmp_path, "wm", timeout=3400, environment="numpy 1.26.2")


def attach_server(monkeypatch, server):
    client = openai.OpenAI(
        api_key="EMPTY",
        base_url="http://test/wm/v1",
        max_retries=0,
        http_client=httpx.Client(transport=httpx.MockTransport(server.handle)),
    )
    monkeypatch.setattr(sim, "_sim_client", client)
    return client


def test_legacy_format_scores_all_complete_classes_with_served_bpe(interpreter, monkeypatch):
    server = ScoringServer()
    with attach_server(monkeypatch, server):
        scores = interpreter.exc_scores("print('metric: 0.8')")
    weights = [math.exp(sum(lp)) for lp in server.logps]
    expected = sim._finalize_exc_posterior([w / sum(weights) for w in weights])
    for key, value in expected.items():
        assert (
            scores[key] == pytest.approx(value)
            if isinstance(value, (float, dict))
            else scores[key] == value
        )
    assert scores["scoring_method"] == "forced_suffix"
    assert scores["argmax_class"] == "NameError"  # First-token-only scoring would pick None.
    assert scores["support_size"] == 24
    assert scores["score"] == scores["p_clean"]
    assert len(server.calls) == 25


@pytest.mark.parametrize(
    "fault", ["ignored", "missing", "truncated", "context", "backend", "marker"]
)
def test_scoring_failure_abstains_without_expensive_echo(interpreter, monkeypatch, fault):
    server = ScoringServer(fault)
    with attach_server(monkeypatch, server):
        assert interpreter.exc_scores("print(1)") is None
    assert all(not body.get("echo") for _, body in server.calls)


def test_max_p_header_does_not_replace_raw_p_clean(interpreter, monkeypatch):
    interpreter.pclean_score = "max_p"
    server = ScoringServer()
    with attach_server(monkeypatch, server):
        scores = interpreter.exc_scores("print(1)")
    assert scores["score"] == scores["max_p"] > scores["p_clean"]
    assert scores["score_name"] == "max_p"


def test_echo_reference_excludes_prefix_and_honors_choice_indices(interpreter, monkeypatch):
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        choices = []
        for i, prompt in enumerate(kwargs["prompt"]):
            prefix, tail = prompt.rsplit("<exc_type>", 1)
            k = sim._EXC_SUPPORT.index(tail.split("</exc_type>")[0])
            choices.append(
                SimpleNamespace(
                    index=i,
                    logprobs=SimpleNamespace(
                        text_offset=[0, 5, len(prefix), len(prefix) + 4],
                        token_logprobs=[None, -9000.0 * k, -float(k), -1.0],
                    ),
                )
            )
        return SimpleNamespace(choices=list(reversed(choices)))

    monkeypatch.setenv("AIDE_EXC_SCORING", "echo")
    monkeypatch.setenv("AIDE_PCLEAN_CHUNK", "5")
    interpreter = sim.SimulatorInterpreter(
        interpreter.working_dir, "wm", timeout=3400, environment="numpy 1.26.2"
    )
    monkeypatch.setattr(
        sim, "_sim_client", SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    scores = interpreter.exc_scores("print(1)")
    assert scores["p_clean"] == pytest.approx(1 / sum(math.exp(-i) for i in range(24)))
    assert scores["scoring_method"] == "echo"
    assert [len(c["prompt"]) for c in calls] == [5, 5, 5, 5, 4]
    assert all(c["echo"] and c["max_tokens"] == 0 for c in calls)


@pytest.mark.parametrize("name", ["default", "failure_site", "execution_result", "typo"])
def test_legacy_prompt_configuration_fails_at_startup(tmp_path, monkeypatch, name):
    monkeypatch.setenv("AIDE_SIMULATOR_PROMPT_FORMAT", name)
    with pytest.raises(ValueError, match="Unsupported AIDE_SIMULATOR_PROMPT_FORMAT"):
        sim.SimulatorInterpreter(tmp_path, "wm", environment="numpy 1.26.2")


@pytest.mark.parametrize(
    "key,value",
    [
        ("AIDE_EXC_SCORING", "typo"),
        ("AIDE_EXC_SCORE_WORKERS", "0"),
        ("AIDE_PCLEAN_CHUNK", "-1"),
    ],
)
def test_bad_scoring_configuration_fails_at_startup(tmp_path, monkeypatch, key, value):
    monkeypatch.delenv("AIDE_SIMULATOR_PROMPT_FORMAT", raising=False)
    monkeypatch.setenv(key, value)
    with pytest.raises(ValueError):
        sim.SimulatorInterpreter(tmp_path, "wm", environment="numpy 1.26.2")
