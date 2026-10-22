"""Regression coverage for the context and reasoning-only failures in vLLM logs."""

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import json
import sys

import httpx
import openai
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "agents/aide/src"))
from aide.backend import backend_openai as backend
from aide.backend.utils import FunctionSpec


SPEC = FunctionSpec(
    "review", {"type": "object", "properties": {"ok": {"type": "boolean"}}}, "Review"
)


def choice(content="answer", *, finish="stop", reasoning=None, calls=None, **extra):
    return SimpleNamespace(
        message=SimpleNamespace(
            content=content, reasoning=reasoning, tool_calls=calls, model_extra=extra
        ),
        finish_reason=finish,
    )


def completion(*choices, prompt_tokens=10, output_tokens=8192):
    return SimpleNamespace(
        choices=list(choices),
        usage=SimpleNamespace(prompt_tokens=prompt_tokens, completion_tokens=output_tokens),
        system_fingerprint=None,
        model="gpt-qwen3-vllm",
        created=0,
    )


def bad_request(message):
    response = httpx.Response(400, request=httpx.Request("POST", "http://test.invalid"))
    return openai.BadRequestError(message, response=response, body=None)


def tool_choice(arguments='{"ok": true}', *, finish="tool_calls", reasoning=None):
    return choice(
        None,
        finish=finish,
        reasoning=reasoning,
        calls=[SimpleNamespace(function=SimpleNamespace(name="review", arguments=arguments))],
    )


@pytest.fixture
def fake_server(monkeypatch):
    for key in ("AIDE_SAMPLING", "AIDE_MODEL_SAMPLING", "AIDE_OPENAI_TIMEOUT"):
        monkeypatch.delenv(key, raising=False)
    calls, replies, clock = [], [], [0.0]
    durations = []
    monkeypatch.setattr(backend.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(backend.time, "time", lambda: clock[0])

    def create(**kwargs):
        calls.append(deepcopy(kwargs))
        clock[0] += durations.pop(0) if durations else 1.0
        assert replies, "Unexpected extra request"
        reply = replies.pop(0)
        if isinstance(reply, BaseException):
            raise reply
        return reply

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    monkeypatch.setattr(backend, "_get_client_for_model", lambda _: client)
    monkeypatch.setattr(backend, "_get_vertex_client", lambda _: client)
    return SimpleNamespace(calls=calls, replies=replies, clock=clock, durations=durations)


def query(**kwargs):
    return backend.query(
        "original system prompt", "original user prompt", model="gpt-qwen3-vllm", **kwargs
    )


def test_complete_response_preserves_prompt_sampling_and_return_contract(fake_server):
    fake_server.replies.append(completion(choice("complete answer"), output_tokens=20))
    result = query(max_tokens=8192, temperature=0.8, extra_body={"top_k": 40})
    assert result[:4] == ("complete answer", 1.0, 10, 20)
    assert fake_server.calls == [
        {
            "model": "gpt-qwen3-vllm",
            "max_tokens": 8192,
            "temperature": 0.8,
            "extra_body": {"top_k": 40},
            "messages": [
                {"role": "system", "content": "original system prompt"},
                {"role": "user", "content": "original user prompt"},
            ],
        }
    ]
    assert result[4]["response_attempts"] == 1


def test_batch_retries_only_failed_slots_and_counts_all_spent_tokens(fake_server):
    fake_server.replies.extend(
        [
            completion(
                choice("candidate A"),
                choice(None, finish="length", reasoning="unfinished"),
                choice("candidate C"),
                choice(" \n", finish="length", reasoning_content="unfinished"),
                output_tokens=18000,
            ),
            completion(choice("replacement B"), choice("replacement D"), output_tokens=10000),
        ]
    )
    output, seconds, in_tokens, out_tokens, info = query(
        n=4, max_tokens=8192, temperature=0.8, extra_body={"top_k": 40}
    )
    assert output == ["candidate A", "replacement B", "candidate C", "replacement D"]
    assert [c["n"] for c in fake_server.calls] == [4, 2]
    assert [c["max_tokens"] for c in fake_server.calls] == [8192, 16384]
    assert fake_server.calls[1]["extra_body"] == {"top_k": 40, "thinking_token_budget": 8192}
    assert fake_server.calls[0]["extra_body"] == {"top_k": 40}
    assert fake_server.calls[0]["messages"] == fake_server.calls[1]["messages"]
    assert (seconds, in_tokens, out_tokens) == (2.0, 20, 28000)
    assert info["response_attempts"] == 2
    assert info["finish_reason"] == "stop"
    assert info["reasoning"] is None  # Metadata belongs to the retained first sibling.
    assert info["finish_reasons"] == ["stop"] * 4


def test_exhausted_recovery_never_returns_incomplete_reasoning_as_code(fake_server):
    unfinished = "```python\nprint('partial program from reasoning')\n```\nStill thinking..."
    fake_server.replies.extend(
        [
            completion(choice(None, finish="length", reasoning=unfinished)),
            completion(choice(None, finish="length", reasoning=unfinished)),
        ]
    )
    output, _, _, tokens, info = query(max_tokens=8192)
    assert output == ""
    assert tokens == 16384
    assert info["finish_reason"] == "length"
    assert len(fake_server.calls) == 2


@pytest.mark.parametrize(
    "field", ["reasoning", "reasoning_content", "extra_reasoning", "extra_reasoning_content"]
)
def test_completed_reasoning_fallback_still_works(fake_server, field):
    reply = choice(None)
    if field.startswith("extra_"):
        reply.message.model_extra[field.removeprefix("extra_")] = "completed answer"
    else:
        setattr(reply.message, field, "completed answer")
    fake_server.replies.append(completion(reply))
    assert query()[0] == "completed answer"
    assert len(fake_server.calls) == 1


def test_partial_final_content_is_left_for_existing_code_parser(fake_server):
    answer = "Solution.\n```python\nprint('complete code')\n```"
    fake_server.replies.append(completion(choice(answer, finish="length")))
    assert query(max_tokens=8192)[0] == answer
    assert len(fake_server.calls) == 1


def test_gemma_truncated_code_retries_only_incomplete_batch_member(fake_server):
    good = "Plan.\n```python\nprint('complete')\n```"
    # This prefix compiles but has lost the remainder of the program.
    cut = "Plan.\n```python\nimport numpy as np\n"
    fake_server.replies.extend([
        completion(choice(good), choice(cut, finish="length", reasoning="thinking")),
        completion(choice(good)),
    ])
    output, _, _, _, info = query(n=2, max_tokens=8192)
    assert output == [good, good]
    assert [call['n'] for call in fake_server.calls] == [2, 1]
    assert [call['max_tokens'] for call in fake_server.calls] == [8192, 16384]
    assert info['finish_reasons'] == ['stop', 'stop']


def test_exhausted_truncated_code_is_not_executed(fake_server):
    cut = "```python\nprint('partial program')\n"
    fake_server.replies.extend([
        completion(choice(cut, finish="length")),
        completion(choice(cut, finish="length")),
    ])
    output, _, _, tokens, info = query(max_tokens=16384)
    assert output == ""
    assert tokens == 16384
    assert info['finish_reason'] == 'length'


@pytest.mark.parametrize('answer', [
    '```python\ns = "```"\n```\ntrailing prose',
    '~~~~python\nprint(1)\n~~~~',
])
def test_closed_fences_are_not_retried_when_trailing_text_is_cut(fake_server, answer):
    fake_server.replies.append(completion(choice(answer, finish='length')))
    assert query(max_tokens=8192)[0] == answer
    assert len(fake_server.calls) == 1


@pytest.mark.parametrize("budget", [None, 1000, 32000])
def test_explicit_thinking_budget_is_preserved(fake_server, budget):
    fake_server.replies.extend(
        [
            completion(choice(None, finish="length", reasoning="unfinished")),
            completion(choice()),
        ]
    )
    original = {"thinking_token_budget": budget, "top_k": 40}
    query(max_tokens=8192, extra_body=original)
    assert fake_server.calls[1]["extra_body"] == original
    assert original == {"thinking_token_budget": budget, "top_k": 40}


def test_older_vllm_can_reject_automatic_thinking_budget_without_losing_reply(fake_server):
    fake_server.replies.extend(
        [
            completion(choice(None, finish="length", reasoning="unfinished")),
            bad_request("thinking_token_budget is not yet supported by the V2 model runner"),
            completion(choice("recovered")),
        ]
    )
    assert query(max_tokens=8192)[0] == "recovered"
    assert fake_server.calls[1]["extra_body"]["thinking_token_budget"] == 8192
    assert "thinking_token_budget" not in fake_server.calls[2]["extra_body"]
    assert fake_server.calls[1]["max_tokens"] == fake_server.calls[2]["max_tokens"] == 16384


def test_hosted_openai_retry_uses_only_standard_parameters(fake_server):
    fake_server.replies.extend(
        [
            completion(choice(None, finish="length", reasoning="unfinished")),
            completion(choice()),
        ]
    )
    backend.query("system", None, model="gpt-test", max_completion_tokens=8192)
    assert fake_server.calls[1]["max_completion_tokens"] == 16384
    assert "max_tokens" not in fake_server.calls[1]
    assert "extra_body" not in fake_server.calls[1]


def test_vertex_retains_provider_defaults_even_on_truncation(fake_server):
    fake_server.replies.append(completion(choice(None, finish="length", reasoning="unfinished")))
    output, *_ = backend.query("system", None, model="google/gemini-test", max_tokens=8192)
    assert output == ""
    assert len(fake_server.calls) == 1
    assert "max_tokens" not in fake_server.calls[0]
    assert fake_server.calls[0]["extra_body"] == {"reasoning_effort": "medium"}


def test_recovery_deadline_keeps_successful_batch_siblings(fake_server):
    fake_server.durations.append(10)
    fake_server.replies.append(
        completion(choice("good"), choice(None, finish="length", reasoning="bad"))
    )
    output, *_ = query(n=2, max_tokens=8192, retry_max_time=10, retry_attempt_timeout=10)
    assert output == ["good", ""]
    assert len(fake_server.calls) == 1


def test_recovery_error_keeps_successful_batch_siblings(fake_server):
    fake_server.replies.extend(
        [
            completion(choice("good"), choice(None, finish="length", reasoning="bad")),
            bad_request("invalid model"),
        ]
    )
    assert query(n=2, max_tokens=8192)[0] == ["good", ""]


def test_recovery_receives_only_remaining_query_budget(fake_server):
    fake_server.durations.extend([17, 3])
    fake_server.replies.extend(
        [
            completion(choice(None, finish="length", reasoning="unfinished")),
            completion(choice()),
        ]
    )
    query(max_tokens=8192, retry_max_time=20, retry_attempt_timeout=20)
    assert [c["timeout"] for c in fake_server.calls] == [20, 3]
    for call in fake_server.calls:
        assert (
            not {"retry_max_time", "retry_attempt_timeout", "tool_call_max_attempts"} & call.keys()
        )


@pytest.mark.parametrize(
    "cap,expected", [(8192, 16384), (16384, 32768), (32768, 32768), (65536, 65536)]
)
def test_automatic_output_expansion_is_capped(fake_server, cap, expected):
    fake_server.replies.extend(
        [
            completion(choice(None, finish="length", reasoning="unfinished")),
            completion(choice()),
        ]
    )
    query(max_tokens=cap)
    assert fake_server.calls[1]["max_tokens"] == expected
    assert fake_server.calls[1]["extra_body"]["thinking_token_budget"] < expected


def test_server_default_output_limit_can_be_retried(fake_server):
    fake_server.replies.extend(
        [
            completion(choice(None, finish="length", reasoning="unfinished"), output_tokens=8192),
            completion(choice()),
        ]
    )
    query()
    assert "max_tokens" not in fake_server.calls[0]
    assert fake_server.calls[1]["max_tokens"] == 16384


def test_empty_text_choices_do_not_raise_index_error(fake_server):
    fake_server.replies.append(completion())
    assert query()[0] == ""


def test_empty_tool_choices_use_existing_bounded_recovery(fake_server):
    fake_server.replies.extend([completion(), completion(tool_choice())])
    output, _, in_tokens, out_tokens, info = query(func_spec=SPEC)
    assert output == {"ok": True}
    assert (in_tokens, out_tokens) == (20, 16384)
    assert info["response_attempts"] == 2


def test_truncated_tool_retry_gets_answer_space_and_counts_spent_tokens(fake_server):
    fake_server.replies.extend(
        [
            completion(tool_choice('{"ok":', finish="length", reasoning="unfinished")),
            completion(tool_choice(), output_tokens=100),
        ]
    )
    result = query(func_spec=SPEC, max_tokens=8192)
    assert result[0] == {"ok": True}
    assert result[2:4] == (20, 8292)
    assert fake_server.calls[1]["max_tokens"] == 16384
    assert fake_server.calls[1]["extra_body"]["thinking_token_budget"] == 8192
    assert fake_server.calls[1]["tools"] == fake_server.calls[0]["tools"]
    assert fake_server.calls[1]["tool_choice"] == fake_server.calls[0]["tool_choice"]


def test_wrong_tool_name_is_retried_instead_of_becoming_a_review(fake_server):
    wrong = tool_choice('{"ok": false}')
    wrong.message.tool_calls[0].function.name = 'unrelated_function'
    fake_server.replies.extend([completion(wrong), completion(tool_choice())])
    assert query(func_spec=SPEC)[0] == {'ok': True}
    assert len(fake_server.calls) == 2


@pytest.mark.parametrize('model', ['gpt-gemma4-26b-a4b-vllm', 'gpt-gemma-4-26b-vllm'])
def test_gemma_feedback_uses_schema_content_and_prompt_contract(fake_server, model):
    fake_server.replies.append(completion(choice('{"ok": true}', reasoning='review reasoning')))
    output, *_ = backend.query('review this log', None, model=model, func_spec=SPEC,
                               max_tokens=8192, convert_system_to_user=True)
    assert output == {'ok': True}
    request = fake_server.calls[0]
    assert 'tools' not in request and 'tool_choice' not in request
    assert request['response_format']['json_schema']['schema'] == SPEC.json_schema
    assert request['messages'][0] == {'role': 'user', 'content': 'review this log'}
    assert json.dumps(SPEC.json_schema) in request['messages'][-1]['content']
    assert SPEC.description in request['messages'][-1]['content']
    assert request['max_tokens'] == 8192


def test_gemma_malformed_structured_feedback_retries_without_parsing_python(fake_server):
    fake_server.replies.extend([
        completion(choice('```python\nreview(ok=True)\n```')),
        completion(choice('{"ok": true}')),
    ])
    output, *_ = backend.query('review this log', None, model='gpt-gemma4-vllm', func_spec=SPEC)
    assert output == {'ok': True}
    assert len(fake_server.calls) == 2
    assert fake_server.calls[0] == fake_server.calls[1]


@pytest.mark.parametrize('cap,budget', [(1536, 768), (2048, 1024), (4096, 2048), (8192, None), (16384, None)])
def test_gemma_short_reviews_reserve_answer_space_without_limiting_codegen(fake_server, cap, budget):
    fake_server.replies.append(completion(choice('completed review')))
    backend.query('audit source', None, model='gpt-gemma4-vllm', max_tokens=cap)
    assert fake_server.calls[0].get('extra_body', {}).get('thinking_token_budget') == budget
    assert fake_server.calls[0]['max_tokens'] == cap


def test_gemma_short_review_preserves_explicit_thinking_budget(fake_server):
    fake_server.replies.append(completion(choice('completed review')))
    backend.query('audit source', None, model='gpt-gemma4-vllm', max_tokens=2048,
                  extra_body={'thinking_token_budget': 1500})
    assert fake_server.calls[0]['extra_body']['thinking_token_budget'] == 1500


def test_gemma_initial_thinking_budget_has_unsupported_server_fallback(fake_server):
    fake_server.replies.extend([
        bad_request('thinking_token_budget is not supported'),
        completion(choice('completed review')),
    ])
    output, *_ = backend.query('audit source', None, model='gpt-gemma4-vllm', max_tokens=2048)
    assert output == 'completed review'
    assert fake_server.calls[0]['extra_body']['thinking_token_budget'] == 1024
    assert 'thinking_token_budget' not in fake_server.calls[1]['extra_body']


def test_truncated_but_valid_json_tool_is_not_accepted(fake_server):
    fake_server.replies.extend([
        completion(tool_choice('{"ok": false}', finish='length')),
        completion(tool_choice()),
    ])
    assert query(func_spec=SPEC, max_tokens=8192)[0] == {'ok': True}
    assert fake_server.calls[1]['max_tokens'] == 16384


@pytest.mark.parametrize(
    "explicit,sampling,expected",
    [
        ({"max_tokens": 123}, '{"max_completion_tokens": 456}', {"max_tokens": 123}),
        ({"max_completion_tokens": 123}, '{"max_tokens": 456}', {"max_completion_tokens": 123}),
        ({}, '{"max_tokens": 123,"max_completion_tokens": 456}', {"max_completion_tokens": 456}),
        ({"max_tokens": 123, "max_completion_tokens": 456}, "{}", {"max_completion_tokens": 456}),
    ],
)
def test_output_cap_aliases_cannot_override_explicit_call_limits(
    fake_server, monkeypatch, explicit, sampling, expected
):
    monkeypatch.setenv("AIDE_SAMPLING", sampling)
    fake_server.replies.append(completion(choice()))
    query(**explicit)
    assert {
        k: v for k, v in fake_server.calls[0].items() if k in backend._OUTPUT_CAP_KEYS
    } == expected


def oversized_debug_prompt(fence="```"):
    # A Python comment that looks like a prompt section must remain source code.
    code = (
        fence
        + "python\n# Memory\n"
        + "protected source\n" * 2000
        + "print('done')\n"
        + fence
        + "\n"
    )
    prefix = (
        "# Introduction\nRepair the bug.\n# Task description\nThe original task.\n# Previous (buggy) implementation\n"
        + code
    )
    diagnosis = "FIRST DIAGNOSIS\n" + "large diagnosis\n" * 20000 + "LAST DIAGNOSIS\n"
    suffix = "# Repair policy\nPreserve working code.\n# Instructions\n## Response format\nReturn Python code.\n"
    return (
        prefix + "# Diagnosis of the failure (from an expert reviewer)\n" + diagnosis + suffix,
        prefix,
        suffix,
    )


@pytest.mark.parametrize("fence", ["```", "~~~~"])
def test_context_overflow_compacts_evidence_and_preserves_core_prompt(fake_server, fence):
    prompt, prefix, suffix = oversized_debug_prompt(fence)
    fake_server.replies.extend(
        [
            bad_request(
                "This model's maximum context length is 131072 tokens. However, you requested "
                "8192 output tokens and your prompt contains at least 122881 input tokens."
            ),
            completion(choice()),
        ]
    )
    output, _, _, _, info = backend.query(prompt, None, model="gpt-qwen3-vllm", max_tokens=8192)
    assert output == "answer"
    assert fake_server.calls[0]["messages"][0]["content"] == prompt
    repaired = fake_server.calls[1]["messages"][0]["content"]
    assert repaired.startswith(prefix)
    assert repaired.endswith(suffix)
    assert "FIRST DIAGNOSIS" in repaired and "LAST DIAGNOSIS" in repaired
    assert len(repaired) < len(prompt) // 2
    assert fake_server.calls[1]["max_tokens"] == 8192
    assert info["context_retries"] == 1


def test_compaction_is_idempotent_and_leaves_ordinary_prompts_identical():
    prompt, _, _ = oversized_debug_prompt()
    messages = [{"role": "system", "content": prompt}]
    assert backend._compact_context(messages)
    compacted = deepcopy(messages)
    assert not backend._compact_context(messages)
    assert messages == compacted
    normal = [
        {"role": "system", "content": "# Task description\nA task\n# Memory\nA short memory.\n"}
    ]
    before = deepcopy(normal)
    assert not backend._compact_context(normal)
    assert normal == before


def test_unclosed_evidence_fence_cannot_cause_instruction_truncation():
    content = (
        "# Diagnosis of the failure (from an expert reviewer)\n```\n"
        + "large diagnosis\n" * 20000
        + "# Instructions\nKeep all required instructions.\n# Data Overview\n"
        + "large data\n" * 20000
    )
    messages = [{"role": "system", "content": content}]
    assert not backend._compact_context(messages)
    assert messages[0]["content"] == content


@pytest.mark.parametrize("cap_key", ["max_tokens", "max_completion_tokens"])
def test_exact_context_count_reduces_output_cap_without_changing_prompt(fake_server, cap_key):
    fake_server.replies.extend(
        [
            bad_request(
                "This model's maximum context length is 32768 tokens. However, you requested "
                "8192 output tokens and your prompt contains 30000 input tokens."
            ),
            completion(choice()),
        ]
    )
    query(**{cap_key: 8192})
    assert fake_server.calls[1][cap_key] == 2768
    assert fake_server.calls[1]["messages"] == fake_server.calls[0]["messages"]


def test_context_recovery_also_runs_during_tool_shape_retry(fake_server):
    fake_server.replies.extend(
        [
            completion(choice(None)),
            bad_request("maximum context length exceeded"),
            completion(tool_choice()),
        ]
    )
    assert query(func_spec=SPEC, max_tokens=8192)[0] == {"ok": True}
    assert "max_tokens" in fake_server.calls[1]
    assert "max_tokens" not in fake_server.calls[2]


def test_required_prompt_overflow_is_reported_without_discarding_task(fake_server):
    fake_server.replies.append(
        bad_request(
            "This model's maximum context length is 131072 tokens. However, you requested "
            "8192 output tokens and your prompt contains at least 131073 input tokens."
        )
    )
    task = "# Task description\n" + "essential task " * 50000
    with pytest.raises(openai.BadRequestError, match="at least 131073"):
        backend.query(task, None, model="gpt-qwen3-vllm", max_tokens=8192)
    assert len(fake_server.calls) == 1
    assert fake_server.calls[0]["messages"][0]["content"] == task


def test_context_recovery_is_bounded_even_without_query_deadline(fake_server):
    fake_server.replies.extend([bad_request("maximum context length exceeded") for _ in range(3)])
    prompt, _, _ = oversized_debug_prompt()
    with pytest.raises(openai.BadRequestError):
        backend.query(prompt, None, model="gpt-qwen3-vllm", max_tokens=8192)
    assert len(fake_server.calls) == 3


def test_unrelated_bad_request_does_not_trigger_context_recovery(fake_server):
    fake_server.replies.append(bad_request("model does not exist"))
    with pytest.raises(openai.BadRequestError, match="model does not exist"):
        query(max_tokens=8192)
    assert len(fake_server.calls) == 1


def test_context_recovery_resizes_automatic_thinking_budget(fake_server):
    fake_server.replies.extend(
        [
            completion(choice(None, finish="length", reasoning="unfinished")),
            bad_request(
                "This model's maximum context length is 32768 tokens. However, you requested "
                "16384 output tokens and your prompt contains 30000 input tokens."
            ),
            completion(choice()),
        ]
    )
    assert query(max_tokens=8192)[0] == "answer"
    assert fake_server.calls[2]["max_tokens"] == 2768
    assert fake_server.calls[2]["extra_body"]["thinking_token_budget"] == 1384


def test_total_allowance_alone_caps_every_request(fake_server):
    fake_server.durations.extend([17, 3])
    fake_server.replies.extend(
        [
            completion(choice(None, finish="length", reasoning="unfinished")),
            completion(choice()),
        ]
    )
    assert query(max_tokens=8192, retry_max_time=20)[0] == "answer"
    assert [c["timeout"] for c in fake_server.calls] == [20, 3]


def test_query_budget_respects_stricter_explicit_sdk_timeout(fake_server):
    fake_server.replies.append(completion(choice()))
    query(retry_max_time=20, timeout=2)
    assert fake_server.calls[0]["timeout"] == 2


def test_batch_usage_cannot_create_unbounded_automatic_output_cap(fake_server):
    fake_server.replies.extend(
        [
            completion(
                choice("good"),
                choice(None, finish="length", reasoning="unfinished"),
                output_tokens=100000,
            ),
            completion(choice()),
        ]
    )
    query(n=2)
    assert fake_server.calls[1]["max_tokens"] == 32768


def test_real_sdk_sends_vllm_extension_and_parses_reasoning_field(fake_server, monkeypatch):
    payloads = []

    def serve(request):
        payloads.append(json.loads(request.content))
        if len(payloads) == 1:
            choices = [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": "retained"},
                },
                {
                    "index": 1,
                    "finish_reason": "length",
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "reasoning_content": "unfinished",
                    },
                },
            ]
        else:
            assert len(payloads) == 2
            choices = [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": "recovered"},
                }
            ]
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "created": 0,
                "model": "gpt-qwen3-vllm",
                "choices": choices,
                "usage": {"prompt_tokens": 10, "completion_tokens": 8192, "total_tokens": 8202},
            },
        )

    with openai.OpenAI(
        base_url="http://test.invalid/v1",
        api_key="test",
        max_retries=0,
        http_client=httpx.Client(transport=httpx.MockTransport(serve)),
    ) as client:
        monkeypatch.setattr(backend, "_get_client_for_model", lambda _: client)
        output, _, inputs, outputs, _ = query(n=2, max_tokens=8192)
    assert output == ["retained", "recovered"]
    assert (inputs, outputs) == (20, 16384)
    assert payloads[1]["n"] == 1
    assert payloads[1]["max_tokens"] == 16384
    assert payloads[1]["thinking_token_budget"] == 8192
    assert "extra_body" not in payloads[1]
    assert payloads[0]["messages"] == payloads[1]["messages"]
