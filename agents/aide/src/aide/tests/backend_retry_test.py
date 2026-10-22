from types import SimpleNamespace

import pytest

from aide.backend import backend_openai
from aide.backend import utils as backend_utils


class RetryableError(Exception):
    pass


def test_retry_deadline_caps_final_attempt(monkeypatch):
    """A 12s deadline with 10s attempts must not launch a third request."""
    clock = [0.0]
    timeouts: list[float] = []

    monkeypatch.setattr(backend_utils.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(
        backend_utils.time,
        "sleep",
        lambda seconds: clock.__setitem__(0, clock[0] + seconds),
    )
    monkeypatch.setattr(backend_utils.random, "uniform", lambda _low, high: high)

    def always_timeout(*, timeout):
        timeouts.append(timeout)
        clock[0] += timeout
        raise RetryableError("timed out")

    with pytest.raises(RetryableError, match="timed out"):
        backend_utils.backoff_create(
            always_timeout,
            (RetryableError,),
            retry_max_time=12,
            retry_attempt_timeout=10,
        )

    assert timeouts == pytest.approx([10.0, 0.5])
    assert clock[0] == pytest.approx(12.0)


def test_unset_deadline_preserves_retry_until_success(monkeypatch):
    monkeypatch.setattr(backend_utils.random, "uniform", lambda _low, _high: 0.0)
    attempts = 0

    def succeeds_on_third_attempt():
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise RetryableError("server restarting")
        return "ok"

    assert backend_utils.backoff_create(
        succeeds_on_third_attempt, (RetryableError,)
    ) == "ok"
    assert attempts == 3


def test_vertex_query_uses_medium_600s_and_ignores_output_caps(monkeypatch):
    captured = {}
    completion = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content="answer"), finish_reason="stop"
            )
        ],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=20),
        system_fingerprint=None,
        model="gemini-3.1-pro-preview",
        created=0,
    )

    def fake_backoff_create(_create_fn, _retry_exceptions, *args, **kwargs):
        captured.update(kwargs)
        return completion

    monkeypatch.delenv("AIDE_VERTEX_REASONING_EFFORT", raising=False)
    monkeypatch.delenv("AIDE_VERTEX_RETRY_MAX_TIME", raising=False)
    monkeypatch.delenv("AIDE_VERTEX_TIMEOUT", raising=False)
    monkeypatch.setattr(backend_openai, "backoff_create", fake_backoff_create)

    output, *_ = backend_openai.query(
        system_message="prompt",
        user_message=None,
        model="google/gemini-3.1-pro-preview",
        max_tokens=1234,
        max_completion_tokens=5678,
    )

    assert output == "answer"
    assert captured["extra_body"] == {"reasoning_effort": "medium"}
    assert "max_tokens" not in captured
    assert "max_completion_tokens" not in captured
    assert captured["retry_max_time"] == pytest.approx(1200.0)
    assert captured["retry_attempt_timeout"] == 600.0


def test_openai_timeout_default_and_override(monkeypatch):
    """Local (non-Vertex) client read-timeout is env-tunable, default 600s."""
    monkeypatch.delenv("AIDE_OPENAI_TIMEOUT", raising=False)
    assert backend_openai._openai_timeout_seconds() == 600.0

    monkeypatch.setenv("AIDE_OPENAI_TIMEOUT", "900")
    assert backend_openai._openai_timeout_seconds() == 900.0

    monkeypatch.setenv("AIDE_OPENAI_TIMEOUT", "0")
    with pytest.raises(ValueError):
        backend_openai._openai_timeout_seconds()
