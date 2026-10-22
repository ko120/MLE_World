"""Per-stage `argmax` policy: routing, the silent-downgrade guard, and the
selector precedence that makes the treatment stage-local.

Guards the failure this policy exists to make impossible: the global
use_p_clean_argmax_pick could only be all-three-stages-or-none, so an
improve-only treatment was unrepresentable.
"""
from types import SimpleNamespace

from aide.agent import Agent, STAGE_POLICIES, STAGES


def _routing_instance(*, policies, p_clean, oracle="race", bon=4):
    inst = object.__new__(Agent)
    inst.acfg = SimpleNamespace(
        stage_policy=SimpleNamespace(**policies),
        best_of_n_size=bon,
        fuse_mode="synth",
        fuse_prompt_style="plain",
    )
    inst.oracle = oracle
    inst.cfg = SimpleNamespace(exec=SimpleNamespace(simulator_only=False))
    inst._p_clean = p_clean
    inst._predict = (lambda *_a, **_k: None)
    return inst


def test_argmax_is_a_valid_policy_name():
    assert "argmax" in STAGE_POLICIES


def test_improve_only_argmax_routes_and_leaves_the_others_parallel():
    """The whole point: one stage argmax, the other two untouched."""
    inst = _routing_instance(
        policies=dict(draft="parallel", debug="parallel", improve="argmax"),
        p_clean=lambda code: {"p_clean": 0.5},
    )
    assert inst._resolve_stage_policies() == {
        "draft": "parallel", "debug": "parallel", "improve": "argmax",
    }


def test_debug_only_argmax_routes():
    inst = _routing_instance(
        policies=dict(draft="parallel", debug="argmax", improve="parallel"),
        p_clean=lambda code: {"p_clean": 0.5},
    )
    assert inst._resolve_stage_policies()["debug"] == "argmax"


def test_argmax_downgrades_to_single_without_the_posterior_scorer():
    """🔴 The silent-failure guard: no exc-posterior scorer => not this arm."""
    inst = _routing_instance(
        policies=dict(draft="parallel", debug="parallel", improve="argmax"),
        p_clean=None,
    )
    assert inst._resolve_stage_policies()["improve"] == "single"


def test_argmax_downgrades_when_width_is_one():
    inst = _routing_instance(
        policies=dict(draft="parallel", debug="parallel", improve="argmax"),
        p_clean=lambda code: {"p_clean": 0.5},
        bon=1,
    )
    assert inst._resolve_stage_policies()["improve"] == "single"


def test_p_clean_scorer_is_wired_for_an_argmax_stage():
    """Check actual wiring without depending on the source spelling of a set."""
    scorer = lambda code: {"p_clean": .5}
    cfg = SimpleNamespace(
        exec=SimpleNamespace(simulator_only=False),
        agent=SimpleNamespace(
            stage_policy=SimpleNamespace(draft="single", debug="single", improve="argmax"),
            best_of_n_size=4, bon_oracle="sim", time_limit=60,
        ),
    )
    inst = Agent("task", cfg, None, sim_predict=lambda code: None, sim_p_clean=scorer)
    assert inst._p_clean is scorer
    assert inst.stage_policy["improve"] == "argmax"


# --- selector precedence in _select ---------------------------------------

def _select_instance(tmp_path, *, p_clean_fn, global_argmax, listwise=False):
    inst = object.__new__(Agent)
    inst.acfg = SimpleNamespace(
        use_best_of_n_listwise=listwise,
        use_p_clean_argmax_pick=global_argmax,
        use_line_frac_salvage=False,
        feedback=SimpleNamespace(model="fb", temp=0.0),
        convert_system_to_user=False,
    )
    inst.cfg = SimpleNamespace(log_dir=tmp_path)
    inst.task_desc = "task"
    inst._p_clean = p_clean_fn
    inst._judge_query = _forbidden_judge
    return inst


def _forbidden_judge(prompt):
    raise AssertionError("the feedback model must not be called for an argmax pick")


CANDS = [("p", "a = 1"), ("p", "b = 2"), ("p", "c = 3")]
SCORES = {"a = 1": 0.10, "b = 2": 0.80, "c = 3": 0.40}


def _hook(code):
    return {"p_clean": SCORES[code], "score": SCORES[code], "score_name": "p_clean"}


def _outcomes():
    return [SimpleNamespace(usable=True, term_out="ok", predicted=True)] * 3


def test_selector_overrides_a_FALSE_global_flag(tmp_path):
    """The stage policy must win: this is what makes it per-stage at all."""
    inst = _select_instance(tmp_path, p_clean_fn=_hook, global_argmax=False)
    picked = inst._select(CANDS, _outcomes(), tmp_path, selector="p_clean_argmax")
    assert picked == 1  # b = 2, the argmax of p_clean


def test_selector_none_falls_back_to_the_global_flag(tmp_path):
    """Existing profiles must be byte-identical in behaviour."""
    inst = _select_instance(tmp_path, p_clean_fn=_hook, global_argmax=True)
    assert inst._select(CANDS, _outcomes(), tmp_path) == 1


def test_listwise_json_records_the_argmax_judge(tmp_path):
    import json
    inst = _select_instance(tmp_path, p_clean_fn=_hook, global_argmax=False)
    inst._select(CANDS, _outcomes(), tmp_path, selector="p_clean_argmax")
    audit = json.loads((tmp_path / "listwise.json").read_text())
    assert audit["judge"] == "p_clean_argmax"
    assert audit["score_name"] == "p_clean"
