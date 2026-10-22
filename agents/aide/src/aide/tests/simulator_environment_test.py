"""The simulator's <environment> block must be the training-corpus text, byte for byte.

The corpus stamped <environment> from scripts/datagen/env_manifest.json (x86_64 run
groups: the x86 entry; aarch64 run groups: the "default" entry). A live probe of the
agent image never reproduces those bytes (missing import-failure line; older
imageio/albucore versions on aarch64), so deployment must send the corpus text.
"""

import hashlib
import json
from pathlib import Path

import pytest

from aide import simulator as sim
from aide import simulator_environment as env


IMPORT_FAILS_LINE = (
    "installed but import fails at startup: albumentations (ImportError: cannot import "
    "name 'preserve_channel_dim' from 'albucore.utils'); keras with the default "
    "TensorFlow backend (ModuleNotFoundError: No module named 'tensorflow')."
)


def _datagen_manifest():
    for parent in [Path(__file__).resolve().parents[5]]:
        candidate = parent / "scripts" / "datagen" / "env_manifest.json"
        if candidate.is_file():
            return json.loads(candidate.read_text(encoding="utf-8"))
    return None


@pytest.mark.parametrize("arch", ["x86_64", "aarch64"])
def test_training_text_matches_recorded_hash(arch):
    text = env.training_environment(arch)
    assert hashlib.sha256(text.encode("utf-8")).hexdigest() == env.TRAINING_ENVIRONMENT_SHA256[arch]
    lines = text.split("\n")
    assert len(lines) == 11
    assert lines[0] == "python 3.11 · network"
    assert lines[-1] == IMPORT_FAILS_LINE
    assert all(line == line.rstrip() for line in lines)


def test_training_text_matches_datagen_manifest():
    manifest = _datagen_manifest()
    if manifest is None:
        pytest.skip("scripts/datagen/env_manifest.json is not reachable from this tree")
    assert env.training_environment("aarch64") == manifest["default"]
    x86_entries = {text for group, text in manifest.items() if group != "default"}
    assert x86_entries == {env.training_environment("x86_64")}


def test_arch_variants_differ_only_in_pinned_versions():
    x86 = env.training_environment("x86_64").split("\n")
    a64 = env.training_environment("aarch64").split("\n")
    changed = [(a, b) for a, b in zip(x86, a64) if a != b]
    assert len(changed) == 2
    assert "imageio 2.37.4" in changed[0][0] and "imageio 2.37.3" in changed[0][1]
    assert "albucore 0.2.13" in changed[1][0] and "albucore 0.1.6" in changed[1][1]


def test_live_render_never_emits_the_import_failure_line():
    live = env.render_manifest(["numpy 1.26.2", "albumentations 1.4.14"], ["tensorflow"], "network", [])
    assert IMPORT_FAILS_LINE not in live
    assert live != env.training_environment("x86_64")


def test_resolve_defaults_to_training_text(monkeypatch):
    monkeypatch.delenv("AIDE_SIMULATOR_ENV_TEXT", raising=False)
    monkeypatch.setattr(env.platform, "machine", lambda: "x86_64")
    text, meta = env.resolve_environment()
    assert text == env.training_environment("x86_64")
    assert meta == {
        "source": "training",
        "arch": "x86_64",
        "lines": 11,
        "sha256": env.TRAINING_ENVIRONMENT_SHA256["x86_64"],
    }


def test_resolve_follows_arch(monkeypatch):
    monkeypatch.setenv("AIDE_SIMULATOR_ENV_TEXT", "training")
    monkeypatch.setattr(env.platform, "machine", lambda: "aarch64")
    text, meta = env.resolve_environment()
    assert text == env.training_environment("aarch64")
    assert meta["arch"] == "aarch64"


def test_resolve_live_uses_the_probe(monkeypatch):
    monkeypatch.setenv("AIDE_SIMULATOR_ENV_TEXT", "live")
    monkeypatch.setattr(env, "capture_environment", lambda: "python 3.11 · network\nnumpy 1.26.2")
    text, meta = env.resolve_environment()
    assert text == "python 3.11 · network\nnumpy 1.26.2"
    assert meta["source"] == "live" and meta["lines"] == 2


def test_resolve_rejects_unknown_source(monkeypatch):
    monkeypatch.setenv("AIDE_SIMULATOR_ENV_TEXT", "trainng")
    with pytest.raises(ValueError, match="AIDE_SIMULATOR_ENV_TEXT"):
        env.resolve_environment()


def test_unknown_arch_raises():
    with pytest.raises(ValueError, match="arch"):
        env.training_environment("riscv64")


def test_simulator_prompt_carries_training_text(tmp_path, monkeypatch):
    monkeypatch.delenv("AIDE_SIMULATOR_ENV_TEXT", raising=False)
    monkeypatch.setattr(env.platform, "machine", lambda: "x86_64")
    messages = []
    monkeypatch.setattr(sim.logger, "info", lambda msg, *a, **k: messages.append(msg))
    interpreter = sim.SimulatorInterpreter(tmp_path, model="wm", timeout=3400)
    expected = env.training_environment("x86_64")
    assert interpreter.environment == expected
    assert f"<environment>\n{expected}\n</environment>" in interpreter._user_message("print(1)")
    assert any(
        m == "SimulatorInterpreter environment: source=training arch=x86_64 lines=11 "
        f"sha256={env.TRAINING_ENVIRONMENT_SHA256['x86_64']}"
        for m in messages
    )


def test_explicit_environment_argument_still_wins(tmp_path):
    interpreter = sim.SimulatorInterpreter(
        tmp_path, model="wm", timeout=3400, environment="python 3.11 · network\nnumpy 1.26.2"
    )
    assert interpreter.environment == "python 3.11 · network\nnumpy 1.26.2"
