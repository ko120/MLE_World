"""Best-of-N integration tests for structured simulator selection fields."""

import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from aide import agent as agent_mod
from aide.agent import Agent
from aide.interpreter import ExecutionResult


def _prediction(submission_exists: str) -> ExecutionResult:
    return ExecutionResult(
        term_out=[
            f"[simulator result] submission_exists={submission_exists}\n",
            "[simulator result] reported_metric=0.5\n",
        ],
        exec_time=0.0,
        exc_type=None,
        predicted_submission_exists=submission_exists,
    )


@pytest.mark.parametrize("policy", ["parallel", "sequential"])
@pytest.mark.parametrize("submission,exc_type", [
    ("false", None),
    ("unknown", None),
    (None, None),
    (False, None),
    ("tru", None),
    ("true", "model.safetensors"),
])
def test_bon_skips_prediction_without_clean_submission(tmp_path, policy, submission, exc_type):
    instance = object.__new__(Agent)
    instance.acfg = SimpleNamespace(
        best_of_n_size=2,
        # First-usable pick: this test asserts the submission gate itself.
        use_best_of_n_listwise=False,
        use_line_frac_salvage=False,
        bon_candidate_review=False,
        code=SimpleNamespace(model="code-model", temp=0.0),
        convert_system_to_user=False,
    )
    instance.cfg = SimpleNamespace(log_dir=tmp_path)
    instance.current_step = 0
    instance.oracle = "sim"
    instance.stage_policy = {s: policy for s in agent_mod.STAGES}
    # Prompt construction is covered by sequential_refinement_test; this test
    # isolates the shared submission gate with a minimal Agent stand-in.
    instance._sequential_refinement_prompt = Mock(return_value={})
    rejected = _prediction(submission)
    rejected.exc_type = exc_type
    instance._predict = lambda code: (
        rejected if "first" in code else _prediction("true")
    )

    candidates = [("first plan", "first = 1"), ("second plan", "second = 2")]
    instance._generate_n_candidates = (
        Mock(side_effect=[[candidate] for candidate in candidates])
        if policy == "sequential"
        else Mock(return_value=candidates)
    )

    _, selected_code, exec_result = instance.generate_code({}, "draft")

    assert selected_code.strip() == "second = 2"
    assert exec_result is None  # sim oracle: the caller executes the winner
    first_audit = json.loads(
        (tmp_path / "bon" / "step_0" / "cand_0" / "result.json").read_text()
    )
    assert first_audit["predicted_submission_exists"] == submission
    assert first_audit["is_predicted_usable"] is False
    assert first_audit["accepted"] is False
    if submission in {"false", "unknown"}:
        assert first_audit["reject_reason"] == f"predicted_submission:{submission}"
