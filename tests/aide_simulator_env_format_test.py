"""Regression coverage for the world_low_9_3 checkpoint's deployment contract."""

import importlib.util
import sys
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "agents/aide/src"))
simulator = pytest.importorskip("aide.simulator")
from aide import simulator_environment as environment
from aide.interpreter import ExecutionResult
from aide.parallel_interpreter import ParallelInterpreter


def _load_script(name):
    if not (REPO / "scripts/datagen" / f"{name}.py").is_file():
        pytest.skip("requires optional workspace training-data tools")
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts/datagen" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ENVIRONMENT = "python 3.11 · no network\nnumpy 1.26.2, pandas 2.1.4\nnot installed: tensorflow"
COMPLETED = """<exc_type>None</exc_type>
<term_out>
&lt;literal output&gt;
ValueError: caught and handled
Validation score: 0.9
</term_out>
<val_metric>0.9</val_metric>
<submission_exists>true</submission_exists>"""


class _Stream:
    def __init__(self, text):
        self.text = text
        self.response = SimpleNamespace(close=lambda: None)

    def __iter__(self):
        for i in range(0, len(self.text), 7):
            yield SimpleNamespace(choices=[SimpleNamespace(
                delta=SimpleNamespace(content=self.text[i:i + 7])
            )])


class _Client:
    def __init__(self, *outputs):
        self.outputs = outputs
        self.calls = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        output = self.outputs[min(len(self.calls), len(self.outputs) - 1)]
        self.calls.append(kwargs)
        return _Stream(output)


@pytest.fixture
def sim(tmp_path, monkeypatch):
    monkeypatch.setenv("AIDE_SIMULATOR_PROMPT_FORMAT", "env_v1_reorder_excfirst")
    monkeypatch.delenv("AIDE_SIMULATOR_SAMPLING", raising=False)
    instance = simulator.SimulatorInterpreter(
        tmp_path, "qwen3_5-4b-sim", timeout=3400, environment=ENVIRONMENT
    )
    instance.set_data_overview("train.csv rows=10 cols=[id, label]")
    return instance


def test_prompt_and_input_match_training_builder(sim):
    builder = _load_script("build_llamafactory_dataset")
    node = {
        "code": "print('score: 0.9')", "environment": ENVIRONMENT,
        "data_overview": sim._data_overview, "exec_timeout_config_s": 3400,
        "has_csv_submission": True,
    }
    assert sim.system_prompt == builder.system_prompt_for(sim.prompt_format)
    assert sim._user_message(node["code"]) == builder._user_msg_from_node(node, sim.prompt_format)
    assert builder.stop_string_for(sim.prompt_format) == "</submission_exists>"


def test_empty_code_predicts_no_submission_without_querying(sim, monkeypatch):
    query = Mock(side_effect=AssertionError("empty code must not query the model"))
    monkeypatch.setattr(simulator, "_streaming_query", query)
    result = sim.run("\n  ")
    assert result.predicted_submission_exists == "false"
    assert not simulator.is_predicted_candidate_usable(result)
    query.assert_not_called()


@pytest.mark.parametrize("include_stop", [True, False])
def test_stream_reads_csv_flag_after_log_and_preserves_raw_output(sim, monkeypatch, include_stop):
    output = COMPLETED if include_stop else COMPLETED.removesuffix("</submission_exists>")
    client = _Client(output)
    monkeypatch.setattr(simulator, "_sim_client", client)
    result = sim.run("print('score: 0.9')")
    assert result.predicted_submission_exists == "true"
    assert result.predicted_metric == .9
    assert result.exc_type is None  # a handled error in stdout is not a crash
    assert simulator.is_predicted_candidate_usable(result)
    assert "&lt;literal output&gt;\n" in "".join(result.term_out)
    assert "reported_metric=0.9\n" in result.term_out[-1]
    request = client.calls[0]
    assert len(client.calls) == 1
    assert request["stop"] == ["</submission_exists>"]
    assert request["messages"][1]["content"] == sim._user_message("print('score: 0.9')")
    assert request["extra_body"]["chat_template_kwargs"] == {"enable_thinking": False}


@pytest.mark.parametrize("flag", ["false", "unknown"])
def test_csv_rejection_reaches_bon_gate(sim, monkeypatch, flag):
    monkeypatch.setattr(simulator, "_sim_client", _Client(
        COMPLETED.replace("<submission_exists>true", f"<submission_exists>{flag}")
    ))
    result = sim.run("print(1)")
    assert result.predicted_submission_exists == flag
    assert not simulator.is_predicted_candidate_usable(result)


@pytest.mark.parametrize("exc", ["TimeoutError", "ValueError", "ArrowInvalid"])
def test_crash_keeps_tail_prediction_and_reports_failure(sim, monkeypatch, exc):
    output = COMPLETED.replace("<exc_type>None", f"<exc_type>{exc}").replace(
        "<val_metric>0.9", "<val_metric>None"
    )
    monkeypatch.setattr(simulator, "_sim_client", _Client(output))
    result = sim.run("print(1)")
    assert result.exc_type == exc
    assert result.predicted_metric is None
    assert result.predicted_submission_exists == "true"  # may write a CSV before crashing
    assert not simulator.is_predicted_candidate_usable(result)
    if exc == "TimeoutError":
        assert "".join(result.term_out).count(simulator._TIMEOUT_FOOTER_PREFIX) == 1


@pytest.mark.parametrize("output", [
    COMPLETED.split("<val_metric>")[0],
    COMPLETED.split("<submission_exists>")[0],
    COMPLETED.replace("<submission_exists>true", "<submission_exists>tru"),
    COMPLETED.replace("<val_metric>0.9", "<val_metric>nan"),
    COMPLETED.replace("<val_metric>0.9", "<val_metric>inf"),
    COMPLETED.replace("<val_metric>0.9", "<val_metric>not-a-number"),
    COMPLETED.replace("<exc_type>None", "<exc_type>ValueError"),
])
def test_malformed_or_truncated_result_abstains(sim, monkeypatch, output):
    client = _Client(output)
    monkeypatch.setattr(simulator, "_sim_client", client)
    result = sim.run("print(1)")
    assert result.exc_type == "SimulatorEmpty"
    assert result.predicted_submission_exists is None
    assert not simulator.is_predicted_candidate_usable(result)
    assert len(client.calls) == 2  # one retry, then stop on the identical answer


def test_retry_recovers_missing_tail(sim, monkeypatch):
    client = _Client(COMPLETED.split("<submission_exists>")[0], COMPLETED)
    monkeypatch.setattr(simulator, "_sim_client", client)
    assert sim.run("print(1)").predicted_submission_exists == "true"
    assert len(client.calls) == 2


def test_distinct_malformed_replies_can_recover_on_last_attempt(sim, monkeypatch):
    client = _Client(
        COMPLETED.split("<val_metric>")[0],
        COMPLETED.split("<submission_exists>")[0],
        COMPLETED.replace("<val_metric>0.9", "<val_metric>nan"),
        COMPLETED,
    )
    monkeypatch.setattr(simulator, "_sim_client", client)
    result = sim.run("print(1)")
    assert result.predicted_submission_exists == "true"
    assert simulator.is_predicted_candidate_usable(result)
    assert len(client.calls) == 1 + simulator._MALFORMED_RETRIES


def test_repeated_malformed_reply_abstains_without_inventing_fields(sim, monkeypatch):
    incomplete = COMPLETED.split("<submission_exists>")[0]
    client = _Client(incomplete, incomplete, COMPLETED)
    monkeypatch.setattr(simulator, "_sim_client", client)
    result = sim.run("print(1)")
    assert result.exc_type == "SimulatorEmpty"
    assert result.predicted_metric is None
    assert result.predicted_submission_exists is None
    assert not simulator.is_predicted_candidate_usable(result)
    assert len(client.calls) == 2


def test_timeout_footer_is_not_duplicated_before_metadata(sim, monkeypatch):
    output = COMPLETED.replace("<exc_type>None", "<exc_type>TimeoutError").replace(
        "<val_metric>0.9", "<val_metric>None"
    ).replace("Validation score: 0.9", "TimeoutError: Execution exceeded the time limit of 1 hour")
    monkeypatch.setattr(simulator, "_sim_client", _Client(output))
    result = sim.run("print(1)")
    assert "".join(result.term_out).count(simulator._TIMEOUT_FOOTER_PREFIX) == 1


def test_midstream_cancellation_closes_stream_and_does_not_retry(sim, monkeypatch):
    cancel = threading.Event()
    close = Mock()

    class CancelledStream:
        response = SimpleNamespace(close=close)

        def __iter__(self):
            for chunk in _Stream(COMPLETED):
                cancel.set()
                yield chunk

    create = Mock(return_value=CancelledStream())
    monkeypatch.setattr(simulator, "_sim_client", SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    ))
    result = sim.run("print(1)", cancel_event=cancel)
    assert result.exc_type == "SimulatorCancelled"
    assert result.predicted_metric is None
    create.assert_called_once()
    assert close.called


def test_default_is_the_exception_token_training_prompt(tmp_path, monkeypatch):
    monkeypatch.delenv("AIDE_SIMULATOR_PROMPT_FORMAT", raising=False)
    instance = simulator.SimulatorInterpreter(tmp_path, "wm", environment=ENVIRONMENT)
    assert instance.prompt_format == "env_v1_reorder_excfirst_token"
    assert instance.system_prompt == simulator._ENV_V1_REORDER_EXCFIRST_TOKEN_SYSTEM_PROMPT
    assert instance.exc_scoring_method == "single_token"


@pytest.mark.parametrize("exc", ["SimulatorEmpty", "SimulatorError", "SimulatorCancelled"])
def test_backend_failure_cannot_enter_the_usable_candidate_set(exc):
    result = ExecutionResult([], 0.0, exc)
    assert not simulator.is_predicted_candidate_usable(result)


def test_truncated_prediction_falls_back_to_real_execution(sim, monkeypatch, tmp_path):
    monkeypatch.setattr(simulator, "_sim_client", _Client(COMPLETED.split("<val_metric>")[0]))
    real_result = ExecutionResult(["real output"], 0.1, None, real_won=True)
    real = SimpleNamespace(working_dir=tmp_path, run=Mock(return_value=real_result))
    race = ParallelInterpreter(real, sim, sim_only=True)
    try:
        assert race.run("print(1)") is real_result
        real.run.assert_called_once()
    finally:
        race.close()


def test_posterior_uses_the_same_environment_and_budget_prompt(sim, monkeypatch):
    posterior = Mock(return_value=None)
    monkeypatch.setattr(simulator, "predict_exc_posterior", posterior)
    assert sim.exc_scores("print(1)") is None
    assert posterior.call_args.kwargs["user_msg"] == sim._user_message("print(1)")


def test_environment_matches_training_manifest_format():
    capture = _load_script("capture_env_manifest")
    assert list(environment.PROBE.items()) == list(capture.PROBE.items())
    installed = [f"{name} 1.2.3" for name in environment.PROBE]
    args = installed, ["tensorflow", "polars"], "no network", ["resnet18", "resnet50"]
    assert environment.render_manifest(*args) == capture.render_manifest(*args)


def test_environment_probes_the_agent_and_caches_result(tmp_path, monkeypatch):
    monkeypatch.setattr(environment, "PROBE", {"numpy": None, "tensorflow": None})
    monkeypatch.setattr(environment.importlib.util, "find_spec", lambda name: object() if name == "numpy" else None)
    version = Mock(return_value="1.26.2")
    monkeypatch.setattr(environment.importlib.metadata, "version", version)
    monkeypatch.setattr(environment.socket, "create_connection", Mock(side_effect=OSError("offline")))
    monkeypatch.setenv("TORCH_HOME", str(tmp_path))
    cache = tmp_path / "hub/checkpoints"
    cache.mkdir(parents=True)
    (cache / "resnet18-abc.pth").touch()
    environment.capture_environment.cache_clear()
    try:
        manifest = environment.capture_environment()
        assert "numpy 1.26.2" in manifest
        assert "not installed: tensorflow" in manifest
        assert "no network" in manifest
        assert "torch hub cache: resnet18" in manifest
        assert environment.capture_environment() == manifest
        version.assert_called_once_with("numpy")
    finally:
        environment.capture_environment.cache_clear()
