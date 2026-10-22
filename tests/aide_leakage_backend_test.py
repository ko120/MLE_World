"""Diagnostic retry allowances apply across context and tool-shape retries."""
import sys
from pathlib import Path
from types import SimpleNamespace

import httpx
import openai
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "agents/aide/src"))
from aide.backend import backend_openai as backend
from aide.data_leakage import DATA_LEAKAGE_CHECK_SPEC


def fake_backend(monkeypatch, *, malformed=False, overflow=False):
    clock, calls = [0.0], []
    monkeypatch.setattr(backend.time, 'monotonic', lambda: clock[0])
    choice = SimpleNamespace(message=SimpleNamespace(model_extra={}), finish_reason='stop')
    completion = SimpleNamespace(choices=[choice], usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1), model='gpt-coderbf16-fb-vllm', created=0, system_fingerprint=None)
    def create(**kwargs):
        calls.append(kwargs)
        clock[0] += 7.0
        if overflow and len(calls) == 1:
            response = httpx.Response(400, request=httpx.Request('POST', 'http://test.invalid'))
            raise openai.BadRequestError('maximum context length exceeded', response=response, body=None)
        return completion
    monkeypatch.setattr(backend, '_get_client_for_model', lambda _: SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create))))
    monkeypatch.setattr(backend, '_apply_sampling_overrides', lambda _: None)
    monkeypatch.setattr(backend, '_extract_tool_call', lambda *a: None if malformed else {'ok': True})
    return calls


def test_context_retry_uses_only_remaining_time_and_strips_control_kwargs(monkeypatch):
    calls = fake_backend(monkeypatch, overflow=True)
    backend.query('review', None, model='gpt-coderbf16-fb-vllm', func_spec=DATA_LEAKAGE_CHECK_SPEC,
                  max_tokens=20, retry_max_time=10, retry_attempt_timeout=10, tool_call_max_attempts=1)
    assert [c['timeout'] for c in calls] == [10.0, 3.0]
    assert 'max_tokens' in calls[0] and 'max_tokens' not in calls[1]
    for c in calls:
        assert not {'retry_max_time', 'retry_attempt_timeout', 'tool_call_max_attempts'} & c.keys()


def test_malformed_tool_retries_cannot_reset_budget(monkeypatch):
    calls = fake_backend(monkeypatch, malformed=True)
    with pytest.raises(TimeoutError, match='allowance exhausted'):
        backend.query('review', None, model='gpt-coderbf16-fb-vllm', func_spec=DATA_LEAKAGE_CHECK_SPEC,
                      retry_max_time=10, retry_attempt_timeout=10, tool_call_max_attempts=3)
    assert len(calls) == 2
    assert [c['timeout'] for c in calls] == [10.0, 3.0]


def test_caller_can_disable_hidden_tool_retries(monkeypatch):
    calls = fake_backend(monkeypatch, malformed=True)
    output, *_ = backend.query('review', None, model='gpt-coderbf16-fb-vllm', func_spec=DATA_LEAKAGE_CHECK_SPEC,
                              retry_max_time=120, retry_attempt_timeout=120, tool_call_max_attempts=1)
    assert output == {} and len(calls) == 1


def test_review_temperature_overrides_recipe_and_survives_context_retry(monkeypatch):
    calls = fake_backend(monkeypatch, overflow=True)
    monkeypatch.setattr(backend, '_apply_sampling_overrides',
                        lambda request: request.update(temperature=1.0))
    backend.query('review', None, model='gpt-coderbf16-fb-vllm',
                  func_spec=DATA_LEAKAGE_CHECK_SPEC, temperature=.5,
                  temperature_override=0.0, max_tokens=20,
                  retry_max_time=10, retry_attempt_timeout=10, tool_call_max_attempts=1)
    assert len(calls) == 2
    assert all(c['temperature'] == 0.0 for c in calls)
    assert all('temperature_override' not in c for c in calls)


def test_ordinary_feedback_retains_sampling_recipe(monkeypatch):
    calls = fake_backend(monkeypatch)
    monkeypatch.setattr(backend, '_apply_sampling_overrides',
                        lambda request: request.update(temperature=1.0))
    backend.query('feedback', None, model='gpt-coderbf16-fb-vllm',
                  func_spec=DATA_LEAKAGE_CHECK_SPEC, temperature=.5)
    assert calls[0]['temperature'] == 1.0
