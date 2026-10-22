"""EXC-token wire protocol, posterior, and canonical execution-result regression tests."""

import importlib.util
import json
import math
from pathlib import Path
import re
import sys
from unittest.mock import Mock

import httpx
import openai
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "agents/aide/src"))
from aide import simulator
from aide import ucb


@pytest.fixture(autouse=True)
def token_format(monkeypatch):
    monkeypatch.setenv("AIDE_SIMULATOR_PROMPT_FORMAT", "env_v1_reorder_excfirst_token")
    monkeypatch.delenv("AIDE_EXC_SCORING", raising=False)


def result_text(label="EXC_None", metric="0.9", submission="true"):
    return (
        f"<exc_type>{label}</exc_type>\n"
        "<term_out>\nLiteral EXC_None in stdout; &lt;tag&gt;\n</term_out>\n"
        f"<val_metric>{metric}</val_metric>\n"
        f"<submission_exists>{submission}</submission_exists>"
    )


def load_datagen(name):
    if not (REPO / "scripts/datagen" / f"{name}.py").is_file():
        pytest.skip("requires optional workspace training-data tools")
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts/datagen" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_support_and_prompt_match_training_contract(tmp_path):
    tokens = load_datagen("exception_tokens")
    builder = load_datagen("build_llamafactory_dataset")
    # The canonical paper checkpoint uses the historical 82-token support;
    # current datagen adds one class for later checkpoints.
    assert tokens.EXCEPTION_CLASSES[-1] == "PackageNotFoundError"
    assert simulator._EXC_TOKEN_SUPPORT == tokens.EXCEPTION_CLASSES[:-1]
    instance = simulator.SimulatorInterpreter(
        tmp_path, "wm", timeout=3400, environment="numpy 1.26.2"
    )
    instance.set_data_overview("train.csv: id,label")
    node = dict(
        code="print(1)",
        environment=instance.environment,
        data_overview=instance._data_overview,
        exec_timeout_config_s=3400,
        has_csv_submission=True,
    )
    assert instance.system_prompt == builder.system_prompt_for(tokens.TOKEN_FORMAT)
    assert instance._user_message(node["code"]) == builder._user_msg_from_node(
        node, tokens.TOKEN_FORMAT
    )


@pytest.mark.parametrize("exc", simulator._EXC_TOKEN_SUPPORT)
def test_all_registered_labels_decode_to_canonical_outcomes(exc):
    parsed = simulator._parse_env_result(
        result_text("EXC_" + exc, "0.9" if exc == "None" else "None")
    )
    assert parsed["exc_type"] == (None if exc == "None" else exc)
    assert parsed["observed_output"] == "Literal EXC_None in stdout; &lt;tag&gt;"


@pytest.mark.parametrize(
    "text",
    [
        result_text("None"),
        result_text("EXC_UnknownError", "None"),
        result_text("EXC_TimeoutError", "0.9"),
        result_text(metric="nan"),
        result_text(submission="tru"),
        result_text().split("<submission_exists>")[0],
    ],
)
def test_malformed_or_mismatched_token_output_abstains(text, tmp_path, monkeypatch):
    monkeypatch.setattr(simulator, "_streaming_query", Mock(return_value=text))
    instance = simulator.SimulatorInterpreter(tmp_path, "wm", environment="numpy 1.26.2")
    result = instance.run("print(1)")
    assert result.exc_type == "SimulatorEmpty"
    assert result.predicted_metric is None
    assert not simulator.is_predicted_candidate_usable(result)


@pytest.mark.parametrize(
    "exc,usable", [("None", True), ("TimeoutError", False), ("ArrowInvalid", False)]
)
def test_decoded_tokens_drive_gate_metric_and_timeout(exc, usable, tmp_path, monkeypatch):
    monkeypatch.setattr(
        simulator,
        "_streaming_query",
        Mock(return_value=result_text("EXC_" + exc, "0.9" if exc == "None" else "None")),
    )
    instance = simulator.SimulatorInterpreter(tmp_path, "wm", environment="numpy 1.26.2")
    result = instance.run("print(1)")
    assert result.exc_type == (None if exc == "None" else exc)
    assert result.predicted_metric == (0.9 if exc == "None" else None)
    assert simulator.is_predicted_candidate_usable(result) is usable
    assert "Literal EXC_None" in "".join(result.term_out)
    assert "".join(result.term_out).count(simulator._TIMEOUT_FOOTER_PREFIX) == (
        exc == "TimeoutError"
    )


@pytest.mark.parametrize("method", ["forced_suffix", "echo", "typo"])
def test_incompatible_scorer_fails_at_initialization(method, tmp_path, monkeypatch):
    monkeypatch.setenv("AIDE_EXC_SCORING", method)
    with pytest.raises(ValueError, match="incompatible"):
        simulator.SimulatorInterpreter(tmp_path, "wm", environment="numpy 1.26.2")


def test_legacy_method_is_selected_from_explicit_plain_format(monkeypatch):
    assert simulator._exc_scoring_method(simulator._PLAIN_PROMPT_FORMAT) == "forced_suffix"
    assert simulator._exc_scoring_method(simulator._PLAIN_PROMPT_FORMAT, "echo") == "echo"
    with pytest.raises(ValueError, match="incompatible"):
        simulator._exc_scoring_method(simulator._PLAIN_PROMPT_FORMAT, "single_token")


class TokenServer:
    """A deterministic tokenizer/model API with changing, non-checkpoint IDs.

    Only EXC_* labels and the chat delimiter are atomic; other text uses chars.
    The real checkpoint's BPE boundaries are covered by the live integration
    check. Here an ID shift between requests catches stale maps and hardcoding.
    """

    def __init__(self, failure=None):
        self.calls = []
        self.failure = failure
        self.version = 0
        self.labels = {}
        self.completion_bodies = []

    def __call__(self, request):
        body = json.loads(request.content)
        self.calls.append(request.url.path)
        if request.url.path == "/tokenize":
            self.version += 1
            assert body["prompt"].count("unique user program") == 1
            parts = re.findall(r"<\|im_start\|>|EXC_\w+|.", body["prompt"], re.S)
            if self.failure == "split_label":
                parts = [
                    piece
                    for part in parts
                    for piece in (list(part) if part == "EXC_None" else [part])
                ]
            mapping = {part: i + 1000 * self.version for i, part in enumerate(sorted(set(parts)))}
            self.labels = {
                name: mapping.get("EXC_" + name) for name in simulator._EXC_TOKEN_SUPPORT
            }
            return httpx.Response(
                200,
                json={
                    "tokens": [mapping[part] for part in parts],
                    "token_strs": parts,
                    "max_model_len": 32768,
                },
            )
        assert request.url.path == "/v1/completions"
        self.completion_bodies.append(body)
        assert body["logprobs"] == 82 and body["max_tokens"] == 1
        assert body["logprob_token_ids"] == list(self.labels.values())
        assert body["return_tokens_as_token_ids"] is True
        assert body["add_special_tokens"] is False
        assert "vllm_xargs" not in body and not body.get("echo")
        probs = {name: 0.2 / 80 for name in self.labels}
        probs["None"], probs["ArrowInvalid"] = 0.6, 0.2
        logps = {
            f"token_id:{self.labels[name]}": math.log(0.9 * value) for name, value in probs.items()
        }
        if self.failure == "missing_rare_label":
            logps.pop(f'token_id:{self.labels["ArrowInvalid"]}')
        if self.failure == "nonfinite":
            logps[f'token_id:{self.labels["ArrowInvalid"]}'] = None
        if self.failure == "bad_mass":
            logps = {key: -0.01 for key in logps}
        key = f'token_id:{self.labels["None"]}'
        return httpx.Response(
            200,
            json={
                "id": "test-completion",
                "object": "text_completion",
                "created": 0,
                "model": body["model"],
                "choices": [
                    {
                        "index": 0,
                        "text": "EXC_None",
                        "finish_reason": "length",
                        "logprobs": {
                            "tokens": [key],
                            "token_logprobs": [logps[key]],
                            "text_offset": [0],
                            "top_logprobs": [logps],
                        },
                    }
                ],
                "usage": {
                    "prompt_tokens": len(body["prompt"]),
                    "completion_tokens": 1,
                    "total_tokens": len(body["prompt"]) + 1,
                },
            },
        )


def use_server(monkeypatch, server):
    client = openai.OpenAI(
        api_key="test",
        base_url="http://simulator.test/v1",
        max_retries=0,
        http_client=httpx.Client(transport=httpx.MockTransport(server)),
    )
    monkeypatch.setattr(simulator, "_sim_client", client)
    return client


def test_sdk_returns_all_probabilities_once_and_refreshes_ids(tmp_path, monkeypatch):
    server = TokenServer()
    with use_server(monkeypatch, server):
        instance = simulator.SimulatorInterpreter(tmp_path, "wm", environment="numpy 1.26.2")
        for _ in range(2):
            scores = instance.exc_scores("unique user program <|im_start|>")
            assert scores["p_clean"] == pytest.approx(0.6)
            assert scores["p_exception"] == pytest.approx(0.4)
            assert scores["score"] == pytest.approx(0.6)
            assert scores["raw_label_mass"] == pytest.approx(0.9)
            assert scores["raw_other_mass"] == pytest.approx(0.1)
            assert scores["support_size"] == len(scores["probabilities"]) == 82
            assert scores["argmax_class"] == "None"
            assert scores["top_exc"] == "ArrowInvalid"
            assert scores["p_top_exc"] == pytest.approx(0.2)
            assert scores["scoring_method"] == "single_token"
            expected_h = -sum(p * math.log(p) for p in [0.6, 0.2] + [0.2 / 80] * 80)
            assert scores["entropy"] == pytest.approx(expected_h)
            assert ucb.uncertainty(scores, "entropy") == pytest.approx(expected_h / math.log(82))
            assert ucb.uncertainty(scores, "p_clean") == pytest.approx(0.4)
    assert server.calls == ["/tokenize", "/v1/completions"] * 2
    assert (
        server.completion_bodies[0]["logprob_token_ids"]
        != server.completion_bodies[1]["logprob_token_ids"]
    )


@pytest.mark.parametrize("failure", ["split_label", "missing_rare_label", "nonfinite", "bad_mass"])
def test_bad_tokenizer_or_probabilities_return_missing_confidence_without_fallback(
    tmp_path, monkeypatch, failure
):
    server = TokenServer(failure)
    with use_server(monkeypatch, server):
        instance = simulator.SimulatorInterpreter(tmp_path, "wm", environment="numpy 1.26.2")
        assert instance.exc_scores("unique user program") is None
    assert server.calls == (
        ["/tokenize"] if failure == "split_label" else ["/tokenize", "/v1/completions"]
    )
