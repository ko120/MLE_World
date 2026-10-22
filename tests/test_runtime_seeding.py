"""Repetitions must change real RNG streams without mutating sibling tasks."""

from dataclasses import dataclass
import json
from pathlib import Path
import random
import subprocess

import numpy as np
import pytest

from agents.seeding import seeded_agent
from aide.utils.seeding import seed_runtime


@dataclass(frozen=True)
class Profile:
    name: str
    kwargs: dict
    env_vars: dict


def test_three_repetitions_have_independent_search_and_model_seeds():
    original = Profile("aide", {"agent.steps": 91}, {
        "AIDE_SAMPLING": '{"temperature":1}',
        "AIDE_MODEL_SAMPLING": '{"feedback":{"max_tokens":1024}}',
        "AIDE_SIMULATOR_SAMPLING": '{"seed":0,"temperature":0}',
    })
    copies = [seeded_agent(original, seed) for seed in range(3)]
    for seed, agent in enumerate(copies):
        assert agent.kwargs == original.kwargs
        assert agent.env_vars["AIDE_RUN_SEED"] == str(seed)
        assert agent.env_vars["PYTHONHASHSEED"] == str(seed)
        assert json.loads(agent.env_vars["AIDE_SAMPLING"]) == {"temperature": 1, "seed": seed}
        assert json.loads(agent.env_vars["AIDE_MODEL_SAMPLING"])["feedback"] == {"max_tokens": 1024, "seed": seed}
        assert agent.env_vars["AIDE_SIMULATOR_SAMPLING"] == original.env_vars["AIDE_SIMULATOR_SAMPLING"]
    copies[0].kwargs["new"] = True
    assert "new" not in copies[1].kwargs and "new" not in original.kwargs
    assert original.kwargs["agent.steps"] == 91
    assert "seed" not in json.loads(original.env_vars["AIDE_SAMPLING"])


def test_runtime_seeds_reproduce_python_and_numpy_draws():
    def draw(seed):
        assert seed_runtime(str(seed)) == seed
        return random.random(), float(np.random.random())
    first = draw(2)
    assert draw(1) != first
    assert draw(2) == first


@pytest.mark.parametrize("seed", [-1, 2**32, True, 1.5])
def test_invalid_harness_seed_fails_before_launch(seed):
    with pytest.raises(ValueError):
        seeded_agent(Profile("aide", {}, {}), seed)


def test_non_aide_profile_keeps_its_argument_contract():
    profile = Profile("other", {}, {})
    assert seeded_agent(profile, 2) is profile


def test_seeded_profile_passes_container_snapshot_handshake():
    agent_dir = Path(__file__).resolve().parents[1] / "agents/aide"
    script = agent_dir / "snapshot_check.sh"
    source_hash = subprocess.check_output(
        ["bash", str(script), "hash", str(agent_dir / "src/aide")], text=True
    ).strip()
    profile = seeded_agent(Profile("aide", {}, {}), 2)
    env = {"PATH": "/usr/bin:/bin", "AGENT_DIR": str(agent_dir),
           "AIDE_SNAPSHOT_SHA": source_hash, **profile.env_vars}
    accepted = subprocess.run(["bash", str(script)], env=env, capture_output=True, text=True)
    assert accepted.returncode == 0, accepted.stderr
    assert "SNAPSHOT_CHECK passed" in accepted.stdout
    rejected = subprocess.run(["bash", str(script)],
        env={**env, "AIDE_UNKNOWN_TEST_KNOB": "1"}, capture_output=True, text=True)
    assert rejected.returncode == 1
    assert "AIDE_UNKNOWN_TEST_KNOB" in rejected.stderr
