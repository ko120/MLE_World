"""Judge-free BoN selection (agent.use_p_clean_argmax_pick).

The arm's whole claim is that a world-model posterior scalar IS the selector,
so three things need pinning. The ranking scalar must follow AIDE_PCLEAN_SCORE
— p_clean by default, max_p for the A/B arm — EXCEPT on an all-failed pool,
which always ranks on p_clean because max_p there is confidence in the
predicted FAILURE and orders nothing useful. `_select` must reach the rule instead of the judge — including on the
all-failed pool, and including when use_best_of_n_listwise was left on. And a
step where nothing could be scored must degrade to first-usable, never fall
through to the feedback model: a zero-judge arm that silently becomes the
list-wise arm on the steps where the simulator was down is unanalysable.

p_clean_argmax_pick is AST-extracted rather than imported, like listwise_pick:
its body is stdlib-only on purpose so it runs without the agent's dep tree.
"""

import ast
import json
from types import SimpleNamespace

import pytest

from aide import agent as agent_mod
from aide.agent import Agent

AGENT_SRC = __import__("pathlib").Path(agent_mod.__file__).read_text()


def _extract(name):
    fn = next(n for n in ast.parse(AGENT_SRC).body
              if isinstance(n, ast.FunctionDef) and n.name == name)
    ns: dict = {}
    exec(compile(ast.Module([fn], []), f"<{name}>", "exec"), ns)
    return ns[name]


pick = _extract("p_clean_argmax_pick")


def _c(idx, p):
    return {"idx": idx, "code": "x = 1", "term_out": "ok", "rank_score": p}


# --- the rule itself -------------------------------------------------------

def test_picks_the_highest_p_clean():
    assert pick([_c(0, 0.2), _c(1, 0.9), _c(2, 0.5)]) == 1


def test_ties_go_to_the_first():
    assert pick([_c(0, 0.7), _c(1, 0.7)]) == 0


def test_unscored_candidates_rank_below_every_scored_one():
    """A scoring failure on one candidate must not hand it the pick, and must
    not knock the pool out of the rule either."""
    assert pick([_c(0, None), _c(1, 0.05), _c(2, None)]) == 1


def test_a_zero_probability_still_beats_unscored():
    """-inf seed, not -1.0: p=0.0 is a real (and orderable) posterior."""
    assert pick([_c(0, None), _c(1, 0.0)]) == 1


def test_idx_is_the_candidate_index_not_the_position():
    """_select passes only the judged pool, so positions and indices differ."""
    assert pick([_c(3, 0.1), _c(7, 0.8)]) == 7


# --- _select wiring --------------------------------------------------------

def _select_instance(tmp_path, *, p_clean_fn, argmax=True, listwise=False,
                     line_frac=False, judge=None):
    inst = object.__new__(Agent)
    inst.acfg = SimpleNamespace(
        use_best_of_n_listwise=listwise,
        use_p_clean_argmax_pick=argmax,
        use_line_frac_salvage=line_frac,
        feedback=SimpleNamespace(model="fb", temp=0.0),
        convert_system_to_user=False,
    )
    inst.cfg = SimpleNamespace(log_dir=tmp_path)
    inst.task_desc = "task"
    inst._p_clean = p_clean_fn
    inst._judge_query = judge or _forbidden_judge
    return inst


def _forbidden_judge(prompt):
    raise AssertionError("the feedback model must not be called in this arm")


def _outcome(usable):
    return SimpleNamespace(usable=usable, term_out="ok", predicted=True)


CANDS = [("p", "a = 1"), ("p", "b = 2"), ("p", "c = 3")]
SCORES = {"a = 1": 0.10, "b = 2": 0.80, "c = 3": 0.40}


def _hook(scored=None, *, key="p_clean", extra=None):
    def fn(code):
        if scored is not None:
            scored.append(code)
        out = {key: SCORES[code], "score": SCORES[code], "score_name": "p_clean"}
        out.update(extra or {})
        return out
    return fn


def test_argmax_replaces_the_judge(tmp_path):
    scored: list[str] = []
    inst = _select_instance(tmp_path, p_clean_fn=_hook(scored))
    outcomes = [_outcome(True), _outcome(True), _outcome(True)]

    assert inst._select(CANDS, outcomes, tmp_path) == 1
    assert scored == ["a = 1", "b = 2", "c = 3"]


def test_argmax_wins_over_a_listwise_flag_left_on(tmp_path):
    """Both knobs on must cost zero judge calls, not run two selectors."""
    inst = _select_instance(tmp_path, p_clean_fn=_hook(), listwise=True)
    outcomes = [_outcome(True), _outcome(True), _outcome(True)]
    assert inst._select(CANDS, outcomes, tmp_path) == 1


def test_only_the_gate_survivors_are_scored_and_ranked(tmp_path):
    """Candidate 1 has the best p_clean but failed the gate: the acceptance
    gate stays the gate, and a rejected candidate costs no scoring call."""
    scored: list[str] = []
    inst = _select_instance(tmp_path, p_clean_fn=_hook(scored))
    outcomes = [_outcome(True), _outcome(False), _outcome(True)]

    assert inst._select(CANDS, outcomes, tmp_path) == 2
    assert scored == ["a = 1", "c = 3"]


def test_single_survivor_still_short_circuits_without_scoring(tmp_path):
    scored: list[str] = []
    inst = _select_instance(tmp_path, p_clean_fn=_hook(scored))
    outcomes = [_outcome(True), _outcome(False), _outcome(False)]

    assert inst._select(CANDS, outcomes, tmp_path) == 0
    assert scored == []


def test_all_failed_pool_is_salvaged_by_the_same_rule(tmp_path):
    """p_clean IS the salvage question, so the argmax covers the all-crash
    pool too — over every candidate, since none survived the gate."""
    scored: list[str] = []
    inst = _select_instance(tmp_path, p_clean_fn=_hook(scored))
    outcomes = [_outcome(False), _outcome(False), _outcome(False)]

    assert inst._select(CANDS, outcomes, tmp_path) == 1
    assert scored == ["a = 1", "b = 2", "c = 3"]


def test_line_frac_still_owns_the_all_crash_case_when_on(tmp_path):
    """use_line_frac_salvage is scoped to the all-crash case by definition, so
    it keeps that case; candidate 2 crashed furthest into its own code."""
    cands = [
        ("p", "a = 1\nb = 2\nc = 3\nd = 4"),
        ("p", "a = 1\nb = 2\nc = 3\nd = 4"),
        ("p", "a = 1\nb = 2"),
    ]
    outs = [_outcome(False), _outcome(False), _outcome(False)]
    outs[0].term_out = 'File "runfile.py", line 1'
    outs[1].term_out = 'File "runfile.py", line 2'
    outs[2].term_out = 'File "runfile.py", line 2'
    inst = _select_instance(tmp_path, p_clean_fn=None, line_frac=True)

    assert inst._select(cands, outs, tmp_path) == 2
    assert json.loads((tmp_path / "listwise.json").read_text())["judge"] == "line_frac"


# --- ranking scalar --------------------------------------------------------

def _maxp_hook(code):
    """exc_scores under AIDE_PCLEAN_SCORE=max_p: `score` carries max_p, whose
    ordering here is the exact REVERSE of p_clean's, so the two arms cannot
    agree by accident."""
    return {
        "p_clean": SCORES[code],
        "score": 1.0 - SCORES[code],
        "score_name": "max_p",
        "argmax_class": "TimeoutError",
    }


def test_maxp_arm_ranks_on_the_surfaced_scalar(tmp_path):
    """AIDE_PCLEAN_SCORE=max_p: the argmax follows `score`, so it must pick the
    max_p winner (candidate 0), NOT the p_clean winner (candidate 1). This is
    the whole arm — one knob moves both the judge header and the argmax."""
    inst = _select_instance(tmp_path, p_clean_fn=_maxp_hook)
    outcomes = [_outcome(True), _outcome(True), _outcome(True)]
    assert inst._select(CANDS, outcomes, tmp_path) == 0


def test_maxp_arm_still_salvages_on_p_clean(tmp_path):
    """All-failed pool: max_p is confidence in the predicted FAILURE and orders
    nothing useful, so salvage normalises to p_clean whatever the knob says —
    the same thing listwise_pick does for the judge. Candidate 1 (best
    p_clean) must win even though candidate 0 has the best max_p."""
    inst = _select_instance(tmp_path, p_clean_fn=_maxp_hook)
    outcomes = [_outcome(False), _outcome(False), _outcome(False)]

    assert inst._select(CANDS, outcomes, tmp_path) == 1
    audit = json.loads((tmp_path / "listwise.json").read_text())
    assert audit["score_name"] == "p_clean"


def test_default_arm_is_unaffected_by_the_maxp_plumbing(tmp_path):
    """Knob unset: `score` IS p_clean, so the default arm ranks on p_clean."""
    inst = _select_instance(tmp_path, p_clean_fn=_hook())
    outcomes = [_outcome(True), _outcome(True), _outcome(True)]

    assert inst._select(CANDS, outcomes, tmp_path) == 1
    assert json.loads((tmp_path / "listwise.json").read_text())["score_name"] == "p_clean"


def test_hook_stamping_only_p_clean_still_ranks(tmp_path):
    """A dict hook with no `score` key (older wiring) must not go unscored."""
    inst = _select_instance(tmp_path, p_clean_fn=lambda c: {"p_clean": SCORES[c]})
    outcomes = [_outcome(True), _outcome(True), _outcome(True)]
    assert inst._select(CANDS, outcomes, tmp_path) == 1


def test_legacy_scalar_hook_still_ranks(tmp_path):
    """Older wiring (and the test suite) hand back a bare float."""
    inst = _select_instance(tmp_path, p_clean_fn=lambda c: SCORES[c])
    outcomes = [_outcome(True), _outcome(True), _outcome(True)]
    assert inst._select(CANDS, outcomes, tmp_path) == 1


# --- degradation -----------------------------------------------------------

def test_no_scoring_source_falls_back_to_first_usable_not_the_judge(tmp_path):
    """exec.simulator off / hook missing: the arm must not call the judge."""
    inst = _select_instance(tmp_path, p_clean_fn=None)
    outcomes = [_outcome(False), _outcome(True), _outcome(True)]
    assert inst._select(CANDS, outcomes, tmp_path) == 1


def test_unscorable_pool_falls_back_to_first_usable_not_the_judge(tmp_path):
    """Scoring reachable but returning None (non-default simulator prompt
    format, or a serve error) — same rule, no judge call."""
    inst = _select_instance(tmp_path, p_clean_fn=lambda c: None)
    outcomes = [_outcome(False), _outcome(True), _outcome(True)]
    assert inst._select(CANDS, outcomes, tmp_path) == 1


def test_unscorable_all_failed_pool_returns_none(tmp_path):
    """Nothing usable and nothing scored: hand the caller its fallback
    (candidate 0, real exec is the truth) rather than a judge call."""
    inst = _select_instance(tmp_path, p_clean_fn=lambda c: None)
    outcomes = [_outcome(False), _outcome(False), _outcome(False)]
    assert inst._select(CANDS, outcomes, tmp_path) is None


# --- audit trail -----------------------------------------------------------

def test_audit_records_the_rule_and_the_scores(tmp_path):
    inst = _select_instance(tmp_path, p_clean_fn=_hook())
    outcomes = [_outcome(True), _outcome(True), _outcome(True)]
    inst._select(CANDS, outcomes, tmp_path)

    audit = json.loads((tmp_path / "listwise.json").read_text())
    assert audit["winner"] == 1
    assert audit["accepted"] == [0, 1, 2]
    # `judge` must stay a STRING: bon_review_eval.py reads isinstance(judge,
    # dict) as "a list-wise judge really decided this set", and an argmax pick
    # credited to the judge would flatter or slander it.
    assert audit["judge"] == "p_clean_argmax"
    assert not isinstance(audit["judge"], dict)
    assert audit["score_name"] == "p_clean"
    assert audit["scores"] == {"0": 0.10, "1": 0.80, "2": 0.40}
