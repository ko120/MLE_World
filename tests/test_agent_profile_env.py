"""Characterize profile resolution so cleanup preserves the original behavior."""

from pathlib import Path

import pytest
import yaml
from agents.registry import Registry
from agents.utils import get_env_var, parse_env_var_values


def test_unset_required_secret_still_fails(monkeypatch):
    monkeypatch.delenv("AGENT_TEST_SECRET", raising=False)
    with pytest.raises(ValueError, match="AGENT_TEST_SECRET"):
        parse_env_var_values({"key": "${{ secrets.AGENT_TEST_SECRET }}"})


@pytest.mark.parametrize("value", ["https://my-provider.example/v1", ""])
def test_required_reference_preserves_environment_value(monkeypatch, value):
    monkeypatch.setenv("AGENT_TEST_ENDPOINT", value)
    profile = {"url": "${{ secrets.AGENT_TEST_ENDPOINT }}"}
    assert get_env_var(profile["url"]) == "AGENT_TEST_ENDPOINT"
    assert parse_env_var_values(profile) is profile
    assert profile["url"] == value


def test_unsupported_fallback_syntax_remains_literal(monkeypatch):
    monkeypatch.delenv("AGENT_TEST_ENDPOINT", raising=False)
    literal = "${{ secrets.AGENT_TEST_ENDPOINT:-https://old.example/v1 }}"
    assert get_env_var(literal) is None
    assert parse_env_var_values({"url": literal})["url"] == literal


def test_literal_values_and_required_secret_resolution_are_unchanged(monkeypatch):
    monkeypatch.setenv("AGENT_TEST_SECRET", "test-value")
    profile = {"key": "${{ secrets.AGENT_TEST_SECRET }}", "count": 3, "url": "literal"}
    assert parse_env_var_values(profile) == {"key": "test-value", "count": 3, "url": "literal"}


def test_historical_profile_keeps_its_literal_endpoint(monkeypatch):
    profile_path = Path(__file__).resolve().parents[1] / "agents/aide/config.yaml"
    env = yaml.safe_load(profile_path.read_text())["aide"]["env_vars"]
    monkeypatch.setenv("OPENAI_API_KEY", "test-value")
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    assert parse_env_var_values(dict(env))["OPENAI_BASE_URL"] == "https://vip.dmxapi.com/v1/"
    monkeypatch.setenv("OPENAI_BASE_URL", "https://my-provider.example/v1")
    assert parse_env_var_values(dict(env))["OPENAI_BASE_URL"] == "https://vip.dmxapi.com/v1/"


def test_registry_discovers_profile_from_an_unrelated_working_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPENAI_API_KEY", "test-value")
    agent = Registry("aide", "aide").get_agent()
    root = Path(__file__).resolve().parents[1] / "agents"
    assert agent.start == root / "aide/start.sh"
    assert agent.dockerfile == root / "aide/Dockerfile"
    assert agent.env_vars["OPENAI_API_KEY"] == "test-value"
