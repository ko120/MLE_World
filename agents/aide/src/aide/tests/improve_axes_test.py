"""agent.improve_plan_axes (prompt family improve_axes_v1).

Contract pinned here: the flag fans improve-stage width draws out into
per-candidate prompts — candidate 0 UN-hinted (judge-failure fallback and the
per-step paired control), candidate i>0 carrying axis (i-1) — and everything
else (flag off, other stages, non-dict prompts) stays on the single batched
n=N request, byte-identical prompts included.

Run: python -m pytest aide/tests/improve_axes_test.py -q
"""

import json
from types import SimpleNamespace

from aide import agent as agent_mod
from aide.agent import IMPROVE_EXPLORE_AXES, Agent


def _agent(n=4, flag=True):
    inst = object.__new__(Agent)
    inst.acfg = SimpleNamespace(
        best_of_n_size=n,
        improve_plan_axes=flag,
        code=SimpleNamespace(model="m", temp=0.5),
        convert_system_to_user=False,
    )
    return inst


def _fake_query(calls):
    """Echo the exploration directive back inside the code block so candidate
    identity is checkable from the parsed (plan, code) tuples."""

    def fake(**kw):
        calls.append(kw)
        sm = kw["system_message"]
        directive = (
            sm.get("Exploration directive", "baseline")
            if isinstance(sm, dict)
            else "baseline"
        )
        text = f"plan\n```python\n# axis: {directive}\n```"
        n = kw.get("n")
        return text if n is None else [text] * n

    return fake


def test_improve_flag_fans_out_per_candidate_prompts(monkeypatch):
    calls = []
    monkeypatch.setattr(agent_mod, "query", _fake_query(calls))
    inst = _agent(n=4, flag=True)

    cands = inst._draw_candidates({"Task": "t"}, None, stage="improve")

    assert len(cands) == 4
    assert len(calls) == 4
    assert all("n" not in c or c["n"] is None for c in calls)
    # candidate 0 is the plain draw; 1..3 carry the first three axes in order
    assert "# axis: baseline" in cands[0][1]
    for i in range(1, 4):
        assert IMPROVE_EXPLORE_AXES[i - 1] in cands[i][1]
    assert inst._last_draw_axes == ["baseline", *IMPROVE_EXPLORE_AXES[:3]]
    # the base prompt object is never mutated
    hinted = [c for c in calls if isinstance(c["system_message"], dict)
              and "Exploration directive" in c["system_message"]]
    assert len(hinted) == 3


def test_flag_off_uses_single_batched_request(monkeypatch):
    calls = []
    monkeypatch.setattr(agent_mod, "query", _fake_query(calls))
    inst = _agent(n=4, flag=False)

    cands = inst._draw_candidates({"Task": "t"}, None, stage="improve")

    assert len(cands) == 4
    assert len(calls) == 1 and calls[0]["n"] == 4
    assert inst._last_draw_axes is None


def test_other_stages_ignore_the_flag(monkeypatch):
    calls = []
    monkeypatch.setattr(agent_mod, "query", _fake_query(calls))
    inst = _agent(n=4, flag=True)

    inst._draw_candidates({"Task": "t"}, None, stage="draft")

    assert len(calls) == 1 and calls[0]["n"] == 4
    assert inst._last_draw_axes is None


def test_axes_cycle_past_the_axis_list(monkeypatch):
    n = len(IMPROVE_EXPLORE_AXES) + 2  # forces one wrap-around
    calls = []
    monkeypatch.setattr(agent_mod, "query", _fake_query(calls))
    inst = _agent(n=n, flag=True)

    inst._draw_candidates({"Task": "t"}, None, stage="improve")

    assert inst._last_draw_axes[0] == "baseline"
    assert inst._last_draw_axes[1:] == [
        IMPROVE_EXPLORE_AXES[i % len(IMPROVE_EXPLORE_AXES)] for i in range(n - 1)
    ]


def test_persist_draw_axes_writes_and_clears(tmp_path):
    inst = _agent()
    inst._last_draw_axes = ["baseline", IMPROVE_EXPLORE_AXES[0]]
    inst._persist_draw_axes(tmp_path)
    assert json.loads((tmp_path / "improve_axes.json").read_text()) == [
        "baseline", IMPROVE_EXPLORE_AXES[0],
    ]
    assert inst._last_draw_axes is None

    # nothing recorded -> no file, no crash
    inst._persist_draw_axes(tmp_path / "empty")
