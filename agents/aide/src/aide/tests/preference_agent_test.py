"""Behavioural tests for policy routing and pairwise selection.

Drives Agent.generate_code's policy methods with a stub code model, stub
oracle and stub judge, so the routing and the call ACCOUNTING are checked
without a model or a GPU.

Run:
    python -m pytest aide/tests/preference_agent_test.py -q
"""

from __future__ import annotations

import json
import types

import pytest

from aide import agent as agent_mod, preference
from aide.interpreter import ExecutionResult
from aide.journal import Journal, Node
from aide.utils.metric import MetricValue


def _exec_result(exc_type=None, submission="true", out="ok"):
    result = ExecutionResult(term_out=[out], exec_time=0.1, exc_type=exc_type,
                             exc_info=None, exc_stack=None)
    # The simulator formats carry this extra field; the viability gate reads it
    # via getattr, so set it the same way the sim path does.
    object.__setattr__(result, "predicted_submission_exists", submission)
    return result


class _Stub:
    """Minimal Agent stand-in: only what the policy paths touch."""

    def __init__(self, tmp_path, draws, sims, verdicts, **acfg):
        self.cfg = types.SimpleNamespace(
            log_dir=tmp_path, exp_name="t", task_name="leaf-classification",
            exec=types.SimpleNamespace(simulator=True, simulator_only=False),
        )
        self.acfg = types.SimpleNamespace(
            best_of_n_size=4,
            bon_oracle="sim",
            stage_policy=types.SimpleNamespace(
                draft="single", debug="single", improve="single"
            ),
            use_best_of_n_listwise=True,
            # No `_review` on this stub, and the reviewer is not under test
            # here; the guard's fallback is the config default (on).
            bon_candidate_review=False,
            use_line_frac_salvage=False,
            fuse_mode="synth",
            fuse_prompt_style="typed_paco_v13",
            pairwise_scheme="bracket",
            pairwise_confidence_gate=0.7,
            pairwise_include_wm_report=True,
            pairwise_model="feedback",
            pairwise_max_calls=24,
            pairwise_max_tokens=4096,
            eval_metric="log loss",
            eval_lower_is_better=True,
            convert_system_to_user=True,
            code=types.SimpleNamespace(model="coder"),
            feedback=types.SimpleNamespace(model="judge", temp=0.5),
        )
        for key, value in acfg.items():
            setattr(self.acfg, key, value)
        self.current_step = 0
        self.task_desc = "Classify 99 plant species."
        self.journal = Journal()
        self.oracle = "sim"
        self._draws = list(draws)
        self._sims = list(sims)
        self._verdicts = list(verdicts)
        self.code_calls = 0
        self.sim_calls = 0
        self.judge_calls = 0
        self.prompts: list[str] = []
        self._predict = self._sim

    def _sim(self, code):
        self.sim_calls += 1
        return self._sims.pop(0)

    def _generate_n_candidates(self, prompt, n, max_tokens=None):
        self.code_calls += 1
        return [self._draws.pop(0) for _ in range(n)]

    def _judge(self, **kwargs):
        self.judge_calls += 1
        self.prompts.append(kwargs["system_message"])
        return self._verdicts.pop(0)

    # Real implementations under test.
    _query = agent_mod.Agent._query
    _new_step_log_dir = agent_mod.Agent._new_step_log_dir
    _write_bon_candidate_artifacts = agent_mod.Agent._write_bon_candidate_artifacts
    _resolve_stage_policies = agent_mod.Agent._resolve_stage_policies
    _draw_candidates = agent_mod.Agent._draw_candidates
    _persist_draw_axes = agent_mod.Agent._persist_draw_axes
    _observe = agent_mod.Agent._observe
    _review_candidates = agent_mod.Agent._review_candidates
    _select = agent_mod.Agent._select
    _commit = agent_mod.Agent._commit
    _sequential = agent_mod.Agent._sequential
    _pairwise = agent_mod.Agent._pairwise
    _pairwise_select = agent_mod.Agent._pairwise_select
    _pairwise_variant = agent_mod.Agent._pairwise_variant


@pytest.fixture(autouse=True)
def _patch_query(monkeypatch):
    holder = {}
    monkeypatch.setattr(agent_mod, "query", lambda **kw: holder["fn"](**kw))
    return holder


# ---------------------------------------------------------------- routing ---

def test_resolver_rejects_sequential_under_the_real_oracle(tmp_path):
    stub = _Stub(tmp_path, draws=[], sims=[], verdicts=[])
    stub.oracle = "real"
    stub.acfg.stage_policy = types.SimpleNamespace(
        draft="sequential", debug="single", improve="single"
    )
    with pytest.raises(ValueError, match="predictive oracle"):
        stub._resolve_stage_policies()


def test_resolver_rejects_unknown_policy_names(tmp_path):
    stub = _Stub(tmp_path, draws=[], sims=[], verdicts=[])
    stub.acfg.stage_policy = types.SimpleNamespace(
        draft="widthwise", debug="single", improve="single"
    )
    with pytest.raises(ValueError, match="stage_policy"):
        stub._resolve_stage_policies()


def test_resolver_downgrades_to_single_without_a_predictor(tmp_path):
    stub = _Stub(tmp_path, draws=[], sims=[], verdicts=[])
    stub._predict = None
    stub.acfg.stage_policy = types.SimpleNamespace(
        draft="sequential", debug="parallel", improve="parallel"
    )
    assert stub._resolve_stage_policies() == {
        "draft": "single", "debug": "single", "improve": "single"
    }


# --------------------------------------------------------------- pairwise ---

def _pairwise_stub(tmp_path, verdicts, **acfg):
    return _Stub(
        tmp_path,
        draws=[(f"p{i}", f"A={i}") for i in range(4)],
        sims=[_exec_result(out=f"SENTINEL_OUT_{i}") for i in range(4)],
        verdicts=verdicts,
        **acfg,
    )


def test_pairwise_bracket_is_m_minus_1_calls_and_returns_the_winner(tmp_path, _patch_query):
    # Always prefer B => candidate 1 beats 0, candidate 3 beats 2,
    # then 3 beats 1.
    stub = _pairwise_stub(
        tmp_path, [r"\boxed{B}"] * 3)
    _patch_query["fn"] = stub._judge
    plan, code, exec_result = stub._pairwise("improve", "improve", None, None)
    assert code == "A=3", code
    assert exec_result is None
    assert stub.judge_calls == 3, stub.judge_calls
    assert (tmp_path / "bon" / "step_0" / "picked.txt").read_text() == "3"
    audit = json.loads((tmp_path / "bon" / "step_0" / "pairwise.json").read_text())
    assert audit["prompt_version"] == preference.PROMPT_VERSION
    assert audit["decided_pairs"] == 3
    assert "confidence_mass" not in audit and "wins" not in audit


def test_pairwise_prompt_carries_the_world_model_output(tmp_path, _patch_query):
    stub = _pairwise_stub(
        tmp_path, [r"\boxed{A}"] * 3)
    _patch_query["fn"] = stub._judge
    stub._pairwise("improve", "improve", None, None)
    first = stub.prompts[0]
    assert "Predicted terminal output:" in first
    assert "SENTINEL_OUT_0" in first and "SENTINEL_OUT_1" in first
    assert "learned simulator's prediction" in first
    assert "Predicted submission produced: true" in first
    assert "Candidate A -- Plan:\np0" in first
    assert "Candidate B -- Plan:\np1" in first
    assert "log loss (lower is better)" in first
    assert "confidence" not in first.lower()
    # Provenance only: no instruction to discount what the simulator predicts.
    assert "distrust" not in first.lower()


def test_pairwise_without_report_omits_execution_evidence(tmp_path, _patch_query):
    stub = _pairwise_stub(
        tmp_path,
        [r"\boxed{A}"] * 3,
        pairwise_include_wm_report=False,
    )
    _patch_query["fn"] = stub._judge
    stub._pairwise("improve", "improve", None, None)
    first = stub.prompts[0]
    assert "Predicted terminal output:" not in first
    assert "SENTINEL_OUT_0" not in first
    assert "simulator" not in first.lower()


def test_pairwise_judge_failure_abstains_instead_of_crashing(tmp_path, _patch_query):
    stub = _pairwise_stub(tmp_path, [])

    def boom(**kwargs):
        stub.judge_calls += 1
        raise RuntimeError("judge exploded")

    _patch_query["fn"] = boom
    _, code, _ = stub._pairwise("improve", "improve", None, None)
    assert code == "A=0"
    assert stub.judge_calls == 3
    audit = json.loads((tmp_path / "bon" / "step_0" / "pairwise.json").read_text())
    assert audit["basis"] == "bracket_fallback"
    assert audit["abstained_pairs"] == 3


def test_pairwise_uses_the_configured_model(tmp_path, _patch_query):
    seen = []

    for setting, expected in (("feedback", "judge"), ("code", "coder")):
        stub = _pairwise_stub(
            tmp_path / setting,
            [r"\boxed{A}"] * 3,
            pairwise_model=setting,
        )

        def capture(**kwargs):
            seen.append(kwargs["model"])
            return stub._verdicts.pop(0)

        _patch_query["fn"] = capture
        stub._pairwise("improve", "improve", None, None)
        assert seen[-1] == expected, (setting, seen[-1])


def test_pairwise_supplies_prior_validation_context_without_test_scores(tmp_path, _patch_query):
    stub = _pairwise_stub(tmp_path, [r"\boxed{B}"] * 3)
    for real, value in ((True, 0.42), (False, 0.37)):
        stub.journal.append(Node(
            code="old_code", plan=f"prior_plan_{real}", analysis="prior_results",
            is_buggy=False, metric=MetricValue(value=value, maximize=False),
            real_won=real, test_metric=987654.321,
        ))
    _patch_query["fn"] = stub._judge
    stub._pairwise("improve", "improve", None, None)
    prompt = stub.prompts[0]
    assert "prior_plan_True" in prompt and "prior_results" in prompt
    assert "Validation metric (real execution): 0.42" in prompt
    assert "Validation metric (world-model prediction): 0.37" in prompt
    assert "987654.321" not in prompt


def test_pairwise_ignores_legacy_confidence_gate(tmp_path, _patch_query):
    stub = _pairwise_stub(tmp_path, [r"\boxed{B}"] * 3, pairwise_confidence_gate=1.0)
    _patch_query["fn"] = stub._judge
    _, code, _ = stub._pairwise("improve", "improve", None, None)
    assert code == "A=3"
    assert stub.judge_calls == 3


def test_pairwise_short_single_candidate_pool_needs_no_judge(tmp_path, _patch_query):
    stub = _Stub(tmp_path, draws=[], sims=[_exec_result()], verdicts=[])
    stub._generate_n_candidates = lambda *a, **kw: [("only plan", "A=7")]
    _patch_query["fn"] = stub._judge
    _, code, _ = stub._pairwise("improve", "improve", None, None)
    assert code == "A=7" and stub.judge_calls == 0
