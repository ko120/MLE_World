"""Provider adapters must honor search allowances without leaking control kwargs."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "agents/aide/src"))
from aide.backend import backend_anthropic, backend_openrouter, utils


class RetryableError(Exception):
    pass


def load_legacy_gdm(monkeypatch):
    # The legacy SDK is optional. Stub only its imports; execute the real adapter.
    modules = {}
    for name in (
        "google",
        "google.api_core",
        "google.api_core.exceptions",
        "google.generativeai",
        "google.generativeai.generative_models",
    ):
        module = ModuleType(name)
        module.__path__ = []
        modules[name] = module
        monkeypatch.setitem(sys.modules, name, module)
        if "." in name:
            parent, child = name.rsplit(".", 1)
            setattr(modules[parent], child, module)
    exceptions = modules["google.api_core.exceptions"]
    for name in ("RetryError", "TooManyRequests", "ResourceExhausted", "InternalServerError"):
        setattr(exceptions, name, RetryableError)
    modules["google.generativeai.generative_models"].generation_types = SimpleNamespace(
        GenerateContentResponse=object
    )
    path = Path(backend_openrouter.__file__).with_name("backend_gdm.py")
    spec = importlib.util.spec_from_file_location("aide.backend._gdm_budget_test", path)
    backend = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(backend)
    return backend


@pytest.mark.parametrize("provider", ["openrouter", "anthropic", "gdm"])
def test_provider_retries_fit_total_allowance_and_keep_request_cap(monkeypatch, provider):
    backend = (
        load_legacy_gdm(monkeypatch)
        if provider == "gdm"
        else backend_anthropic if provider == "anthropic" else backend_openrouter
    )
    clock, timeouts = [0.0], []
    monkeypatch.setattr(utils.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(utils.random, "uniform", lambda *_: 0.0)

    def create(**kwargs):
        assert not {"retry_max_time", "retry_attempt_timeout"} & kwargs.keys()
        if provider == "gdm":
            assert kwargs["request_options"]["retry"] is None
            timeout = kwargs["request_options"]["timeout"]
        else:
            timeout = kwargs["timeout"]
        timeouts.append(timeout)
        clock[0] += timeout
        raise RetryableError("server unavailable")

    if provider == "gdm":
        monkeypatch.setattr(backend, "_setup_gdm_client", lambda *_: None)
        monkeypatch.setattr(backend, "gdm_model", SimpleNamespace(generate_content=create))
    elif provider == "anthropic":
        monkeypatch.setattr(backend, "_setup_anthropic_client", lambda: None)
        monkeypatch.setattr(
            backend, "_client", SimpleNamespace(messages=SimpleNamespace(create=create))
        )
        monkeypatch.setattr(backend, "ANTHROPIC_TIMEOUT_EXCEPTIONS", (RetryableError,))
    else:
        monkeypatch.setattr(backend, "_setup_openrouter_client", lambda: None)
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        monkeypatch.setattr(backend, "_client", client)
        monkeypatch.setattr(backend, "OPENAI_TIMEOUT_EXCEPTIONS", (RetryableError,))

    with pytest.raises(RetryableError):
        backend.query(
            "task", None, model="test-model", retry_max_time=1250.0, retry_attempt_timeout=1250.0
        )
    assert timeouts == [600.0, 600.0, 50.0]
    assert clock[0] == 1250.0
