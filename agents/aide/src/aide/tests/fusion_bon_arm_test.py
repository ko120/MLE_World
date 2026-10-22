"""Tests for the deployed WM-Fuse arm (Agent._fuse_candidates).

The offline harness already covers prompt CONTENT (fusion_prompt_test.py,
scripts/eval/wm_fuse/wm_fuse_prompt_regress.py). What is only testable here is the
deployment wiring:

* the fused program is what executes, unscreened — no simulator verify gate and
  no fallback to the BoN pick, so an unparseable response is retried rather
  than swapped for a candidate;
* no feedback-LLM call is spent on a fused step, because synth fusion never
  consumes picked_idx — while non-fused stages still use the judge;
* the parent evidence actually reaches the prompt. `_improve_prompt` embeds
  only the parent's CODE, so a fusion arm that forgot to thread `parent_node`
  would still build a plausible prompt and silently measure the wrong thing.
"""

import json
from types import SimpleNamespace

from aide import agent as agent_mod
from aide.agent import Agent, CandidateOutcome
from aide.interpreter import ExecutionResult
from aide.journal import Node


def _prediction(submission_exists: str, exc_type=None) -> ExecutionResult:
    return ExecutionResult(
        term_out=[f"[simulator result] submission_exists={submission_exists}\n"],
        exec_time=0.0,
        exc_type=exc_type,
        predicted_submission_exists=submission_exists,
    )


def _agent(tmp_path, *, improve_policy="fuse", sim_verdict="true"):
    instance = object.__new__(Agent)
    instance.acfg = SimpleNamespace(
        best_of_n_size=4,
        use_best_of_n_listwise=False,
        use_line_frac_salvage=False,
        fuse_mode="synth",
        fuse_prompt_style="typed_paco_v13",
        fuse_include_candidate_plans=True,
        fuse_retries=3,
        fuse_max_tokens=8192,
        eval_metric=None,
        eval_lower_is_better=None,
        code=SimpleNamespace(model="code-model", temp=0.0),
        convert_system_to_user=False,
    )
    instance.cfg = SimpleNamespace(log_dir=tmp_path)
    instance.task_desc = "predict the target"
    instance.current_step = 0
    instance.oracle = "sim"
    instance.stage_policy = {
        "draft": "parallel",
        "debug": "parallel",
        "improve": improve_policy,
    }
    instance._predict = lambda code: _prediction(sim_verdict)
    return instance


def _parent() -> Node:
    node = Node(code="print('parent code')", plan="parent plan")
    node.analysis = "parent analysis"
    node._term_out = ["parent stdout\n"]
    return node


IMPROVE_PROMPT = {
    "Introduction": "improve the previous solution",
    "Task description": "predict the target",
    "Previous solution": {"Code": "print('parent code')"},
}
FUSED_RESPONSE = "Fusion rationale.\n```python\nprint('fused')\n```"


def _candidates(n=4):
    return [(f"plan {i}", f"print({i})") for i in range(n)]


def _outcomes(n=4, submission_exists="true"):
    return [
        CandidateOutcome(
            result=_prediction(submission_exists),
            usable=submission_exists == "true",
            submission_exists=submission_exists,
            predicted=True,
        )
        for _ in range(n)
    ]


def test_fused_program_replaces_the_pick(tmp_path, monkeypatch):
    instance = _agent(tmp_path)
    monkeypatch.setattr(agent_mod, "query", lambda **kwargs: FUSED_RESPONSE)

    result = instance._fuse_candidates(
        prompt=IMPROVE_PROMPT,
        stage="improve",
        parent_node=_parent(),
        candidates=_candidates(),
        outcomes=_outcomes(),
        picked_idx=2,
        step_log_dir=tmp_path,
    )

    assert result is not None
    plan, code = result
    # parse_fusion_response uses AIDE's production parser, which Black-formats
    # the extracted block — hence the normalized quotes.
    assert code.strip() == 'print("fused")'
    assert plan.strip() == "Fusion rationale."

    audit = json.loads((tmp_path / "fuse.json").read_text())
    assert audit["executed"] is True
    assert audit["attempts"] == 1
    assert audit["stage"] == "improve"
    assert audit["report_source"] == "wm"
    assert audit["prompt_style"] == "typed_paco_v13"
    assert audit["input_candidate_indices"] == [0, 1, 2, 3]
    assert (
        tmp_path / "cand_fused" / "solution.py"
    ).read_text().strip() == 'print("fused")'
    assert json.loads(
        (tmp_path / "cand_fused" / "result.json").read_text()
    )["accepted"] is True


def test_parent_evidence_reaches_the_fusion_prompt(tmp_path, monkeypatch):
    """The evidence AIDE logs but never shows the coder must be in the prompt.

    `_improve_prompt` carries the parent's code alone — no plan, no analysis,
    no execution output. Those three are the reason the paco families exist.
    """
    instance = _agent(tmp_path)
    seen = {}

    def _capture(**kwargs):
        seen["prompt"] = kwargs["system_message"]
        seen["max_tokens"] = kwargs["max_tokens"]
        return FUSED_RESPONSE

    monkeypatch.setattr(agent_mod, "query", _capture)
    instance._fuse_candidates(
        prompt=IMPROVE_PROMPT,
        stage="improve",
        parent_node=_parent(),
        candidates=_candidates(),
        outcomes=_outcomes(),
        picked_idx=0,
        step_log_dir=tmp_path,
    )

    prompt = seen["prompt"]
    assert "parent plan" in prompt
    assert "parent analysis" in prompt
    assert "parent stdout" in prompt
    # Rule 2 promises the parent and every candidate share one layout, so the
    # candidate plans must actually be there (paco_parent_v8's setting).
    for i in range(4):
        assert f'<candidate_plan id="solution_v{i}">' in prompt
        assert f"plan {i}" in prompt
    # ...but per-candidate analysis genuinely does not exist online (AIDE
    # reviews only the picked candidate), so that slot must stay unrendered
    # rather than emitting an empty block.
    assert "<candidate_analysis" not in prompt
    # Layout order per candidate: plan -> (analysis) -> code -> report.
    assert prompt.index('<candidate_plan id="solution_v0">') < prompt.index(
        '<candidate_code id="solution_v0">'
    ) < prompt.index('<execution_report id="solution_v0"')
    # Reports retain their provenance and the rules identify predictions.
    assert 'source="wm"' in prompt
    assert "not an observed fact" in prompt
    # request_sections=() drops the embedded task request entirely.
    assert "predict the target" not in prompt
    assert seen["max_tokens"] == 8192


def test_no_verify_gate_and_no_fallback(tmp_path, monkeypatch):
    """The fused program runs unscreened, whatever the simulator would say.

    There is deliberately no `_predict(fused)` call: this arm is a pure
    always-fuse policy, so its floor is NOT the select arm.
    """
    instance = _agent(tmp_path, sim_verdict="false")
    sim_calls = []
    instance._predict = lambda code: (
        sim_calls.append(code) or _prediction("false")
    )
    monkeypatch.setattr(agent_mod, "query", lambda **kwargs: FUSED_RESPONSE)

    result = instance._fuse_candidates(
        prompt=IMPROVE_PROMPT,
        stage="improve",
        parent_node=_parent(),
        candidates=_candidates(),
        outcomes=_outcomes(submission_exists="false"),
        picked_idx=1,
        step_log_dir=tmp_path,
    )

    assert result is not None and result[1].strip() == 'print("fused")'
    assert sim_calls == []  # the fused program is never screened
    audit = json.loads((tmp_path / "fuse.json").read_text())
    assert audit["executed"] is True


def test_stage_routing_dispatches_around_the_fuser(tmp_path, monkeypatch):
    """fuse gating is structural now: a draft step routes to its own policy,
    so the fuser cannot fire there — no stage-list check needed."""
    instance = _agent(tmp_path)
    called = []
    monkeypatch.setattr(
        Agent, "_parallel", lambda self, *a: called.append("parallel") or ("p", "c", None)
    )
    monkeypatch.setattr(
        Agent, "_fuse", lambda self, *a: called.append("fuse") or ("p", "c", None)
    )
    instance.generate_code({"Task description": "t"}, "draft")
    assert called == ["parallel"]
    instance.generate_code(IMPROVE_PROMPT, "improve", _parent())
    assert called == ["parallel", "fuse"]


def test_invalid_code_is_retried_then_executed_best_effort(tmp_path, monkeypatch):
    """No fallback: retry `fuse_retries` times, then run what we have."""
    instance = _agent(tmp_path)
    calls = []
    monkeypatch.setattr(
        agent_mod, "query",
        lambda **kwargs: (calls.append(1), "no code here at all")[1],
    )

    result = instance._fuse_candidates(
        prompt=IMPROVE_PROMPT,
        stage="improve",
        parent_node=_parent(),
        candidates=_candidates(),
        outcomes=_outcomes(),
        picked_idx=3,
        step_log_dir=tmp_path,
    )

    assert len(calls) == 3
    # Best-effort code is executed rather than reverting to a candidate; real
    # execution reports the true error.
    assert result is not None
    audit = json.loads((tmp_path / "fuse.json").read_text())
    assert audit["executed"] is True
    assert audit["valid_python"] is False
    assert audit["attempts"] == 3


def test_retry_stops_as_soon_as_code_parses(tmp_path, monkeypatch):
    instance = _agent(tmp_path)
    responses = ["no code at all", FUSED_RESPONSE, "should not be reached"]
    calls = []

    def _query(**kwargs):
        calls.append(1)
        return responses[len(calls) - 1]

    monkeypatch.setattr(agent_mod, "query", _query)
    result = instance._fuse_candidates(
        prompt=IMPROVE_PROMPT,
        stage="improve",
        parent_node=_parent(),
        candidates=_candidates(),
        outcomes=_outcomes(),
        picked_idx=0,
        step_log_dir=tmp_path,
    )

    assert len(calls) == 2
    assert result is not None and result[1].strip() == 'print("fused")'
    assert json.loads((tmp_path / "fuse.json").read_text())["attempts"] == 2


def test_transport_failure_exhausts_retries_then_declines(tmp_path, monkeypatch):
    """A wedged endpoint costs attempts, so it cannot spin forever."""
    instance = _agent(tmp_path)
    calls = []

    def _raise(**kwargs):
        calls.append(1)
        raise RuntimeError("vLLM dropped the connection")

    monkeypatch.setattr(agent_mod, "query", _raise)
    result = instance._fuse_candidates(
        prompt=IMPROVE_PROMPT,
        stage="improve",
        parent_node=_parent(),
        candidates=_candidates(),
        outcomes=_outcomes(),
        picked_idx=0,
        step_log_dir=tmp_path,
    )

    assert len(calls) == 3
    # Nothing was generated, so there is nothing to execute.
    assert result is None


def test_bon_query_spends_no_feedback_call_on_a_fused_step(tmp_path, monkeypatch):
    """End-to-end: fused program executes and the judge is never called.

    `use_best_of_n_listwise` is ON, and all four candidates are sim-approved —
    the select arm would spend one feedback-LLM call here. Synth fusion never
    consumes picked_idx, so that call must be skipped outright.
    """
    instance = _agent(tmp_path)
    instance.acfg.use_best_of_n_listwise = True
    instance.acfg.feedback = SimpleNamespace(model="feedback-model", temp=0.0)
    calls = []

    def _query(**kwargs):
        calls.append(kwargs)
        if kwargs.get("n"):  # candidate generation
            return [f"plan {i}\n```python\nprint({i})\n```" for i in range(kwargs["n"])]
        return FUSED_RESPONSE

    monkeypatch.setattr(agent_mod, "query", _query)
    plan, code, exec_result = instance.generate_code(
        IMPROVE_PROMPT, "improve", _parent()
    )

    assert code.strip() == 'print("fused")'
    assert exec_result is None
    assert (tmp_path / "bon" / "step_0" / "picked.txt").read_text() == "fused"
    # Exactly two calls: candidate generation, then fusion. No judge call.
    assert len(calls) == 2
    assert [c["model"] for c in calls] == ["code-model", "code-model"]
    assert not (tmp_path / "bon" / "step_0" / "listwise.json").exists()


def test_non_fused_stage_still_uses_the_judge(tmp_path, monkeypatch):
    """Skipping selection is scoped to fused steps only.

    draft routes to `parallel`, so it runs the plain select arm and must still
    spend the feedback call that picks among candidates.
    """
    instance = _agent(tmp_path, improve_policy="fuse")
    instance.acfg.use_best_of_n_listwise = True
    instance.acfg.feedback = SimpleNamespace(model="feedback-model", temp=0.0)
    models = []

    def _query(**kwargs):
        models.append(kwargs.get("model"))
        if kwargs.get("n"):
            return [f"plan {i}\n```python\nprint({i})\n```" for i in range(kwargs["n"])]
        return {"winner_idx": 0, "reason": "first"}

    monkeypatch.setattr(agent_mod, "query", _query)
    instance.generate_code({"Task description": "t"}, "draft")

    assert "feedback-model" in models


def test_fused_step_marks_no_candidate_accepted(tmp_path, monkeypatch):
    """On a fused step the executed program is cand_fused/, not a candidate.

    Crowning the first sim-usable candidate here would tell an offline
    extractor that it became the journal node, which is false.
    """
    instance = _agent(tmp_path)
    instance.acfg.use_best_of_n_listwise = True
    instance.acfg.feedback = SimpleNamespace(model="feedback-model", temp=0.0)

    def _query(**kwargs):
        if kwargs.get("n"):
            return [f"plan {i}\n```python\nprint({i})\n```" for i in range(kwargs["n"])]
        return FUSED_RESPONSE

    monkeypatch.setattr(agent_mod, "query", _query)
    instance.generate_code(IMPROVE_PROMPT, "improve", _parent())

    step = tmp_path / "bon" / "step_0"
    for i in range(4):
        result = json.loads((step / f"cand_{i}" / "result.json").read_text())
        assert result["accepted"] is False
        assert result["reject_reason"] == "not_executed:fused_program_ran_instead"
    assert json.loads((step / "cand_fused" / "result.json").read_text())["accepted"] is True
