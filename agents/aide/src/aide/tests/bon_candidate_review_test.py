"""Per-candidate reviewer for the drawn best-of-N pool (agent.bon_candidate_review).

AIDE reviews only the candidate it picks, so a stock run holds no analysis of
the ones it rejected. This flag runs the SAME reviewer (`review_func_spec`) over
every drawn candidate and surfaces the summary to the list-wise judge and to
`FusionCandidate.analysis`.

Four things are worth pinning:

* flag off is a true no-op — zero extra feedback calls, and the judge prompt is
  byte-identical to the pre-review one, so an A/B measures only the reviewer;
* the committed-node prompt is unchanged, because it is what every banked
  review and the world-model corpus were built from;
* the candidate path never emits the datagen node anchors, which would forge
  phantom nodes in the run.log fallback parser (extract_aide_logs.py);
* the review is advisory — the acceptance gate is untouched by what it says.
"""

import ast
import json
from types import SimpleNamespace

import pytest

from aide import agent as agent_mod
from aide.agent import Agent, CandidateOutcome

AGENT_SRC = __import__("pathlib").Path(agent_mod.__file__).read_text()

LEGACY_INTRO = (
    "You are a Kaggle grandmaster attending a competition. "
    "You have written code to solve this task and now need to evaluate the output of the code execution. "
    "You should determine if there were any bugs as well as report the empirical findings."
)


def _extract_listwise_pick():
    fn = next(n for n in ast.parse(AGENT_SRC).body
              if isinstance(n, ast.FunctionDef) and n.name == "listwise_pick")
    ns: dict = {}
    exec(compile(ast.Module([fn], []), "<listwise_pick>", "exec"), ns)
    return ns["listwise_pick"]


def _agent(review_flag=True, obfuscate=False):
    a = object.__new__(Agent)
    a.acfg = SimpleNamespace(
        obfuscate=obfuscate,
        eval_metric=None,
        convert_system_to_user=False,
        feedback=SimpleNamespace(model="fb", temp=0.0),
        bon_candidate_review=review_flag,
    )
    a.task_desc = "task"
    return a


def _outcome(term_out="ok\n", predicted=True, usable=True):
    return CandidateOutcome(
        result=SimpleNamespace(term_out=[term_out], exc_type=None, real_won=False),
        usable=usable,
        submission_exists="true",
        predicted=predicted,
    )


@pytest.fixture
def recorder(monkeypatch):
    calls: list[dict] = []

    def query(**kw):
        calls.append(kw)
        return {"is_bug": False, "has_csv_submission": True,
                "summary": f"finding {len(calls)}", "metric": None,
                "lower_is_better": False, "metric_trustworthy": True}

    monkeypatch.setattr(agent_mod, "query", query)
    return calls


CANDS = [("plan a", "a = 1"), ("plan b", "b = 2")]


# --- flag off is a true no-op ---------------------------------------------

def test_flag_off_spends_no_calls_and_leaves_no_analysis(recorder, tmp_path):
    outcomes = [_outcome(), _outcome()]
    _agent(review_flag=False)._review_candidates(CANDS, outcomes, tmp_path)
    assert recorder == []
    assert [o.review for o in outcomes] == [None, None]
    assert [o.analysis for o in outcomes] == ["", ""]
    assert not list(tmp_path.iterdir())


def test_missing_flag_falls_back_to_the_config_default_on(recorder, tmp_path):
    """No attribute -> the AgentConfig default (True), not a silent opt-out.

    Only stand-ins built via object.__new__ can land here; a real run always
    carries a full AgentConfig, which always has the field. Keeping the
    fallback equal to that default means a stub cannot accidentally exercise a
    configuration nothing ships.
    """
    agent = _agent()
    del agent.acfg.bon_candidate_review
    outcomes = [_outcome()]
    agent._review_candidates(CANDS[:1], outcomes, tmp_path)
    assert len(recorder) == 1
    assert outcomes[0].review is not None


# --- flag on ---------------------------------------------------------------

def test_reviews_every_candidate_and_persists_each(recorder, tmp_path):
    outcomes = [_outcome(), _outcome()]
    _agent()._review_candidates(CANDS, outcomes, tmp_path)

    assert len(recorder) == 2
    assert {c["model"] for c in recorder} == {"fb"}
    assert all(c["func_spec"] is agent_mod.review_func_spec for c in recorder)
    # every candidate's OWN code was reviewed, not the pool's first
    reviewed = {c["system_message"]["Implementation"] for c in recorder}
    assert reviewed == {agent_mod.wrap_code("a = 1"), agent_mod.wrap_code("b = 2")}
    assert sorted(o.analysis for o in outcomes) == ["finding 1", "finding 2"]
    for i in (0, 1):
        saved = json.loads((tmp_path / f"cand_{i}" / "review.json").read_text())
        assert saved["summary"] == outcomes[i].analysis


def test_one_failing_review_does_not_kill_the_step(monkeypatch, tmp_path):
    def query(**kw):
        if "b = 2" in kw["system_message"]["Implementation"]:
            raise RuntimeError("judge exploded")
        return {"summary": "fine"}

    monkeypatch.setattr(agent_mod, "query", query)
    outcomes = [_outcome(), _outcome()]
    _agent()._review_candidates(CANDS, outcomes, tmp_path)
    assert outcomes[0].analysis == "fine"
    assert outcomes[1].review is None and outcomes[1].analysis == ""


def test_review_is_advisory_and_never_moves_the_gate(recorder, tmp_path):
    """A reviewer calling a candidate buggy must not un-accept it: the oracle's
    gate is what defines the arm."""
    outcomes = [_outcome(usable=True), _outcome(usable=False)]
    _agent()._review_candidates(CANDS, outcomes, tmp_path)
    assert [o.usable for o in outcomes] == [True, False]


# --- prompt provenance -----------------------------------------------------

def test_committed_node_review_wording_is_unchanged(recorder):
    _agent()._review("code", "out")
    sent = recorder[-1]["system_message"]
    assert sent["Introduction"] == LEGACY_INTRO
    assert "Execution output" in sent and "Predicted execution output" not in sent


def test_obfuscated_node_review_wording_is_unchanged(recorder):
    _agent(obfuscate=True)._review("code", "out")
    assert recorder[-1]["system_message"]["Introduction"] == LEGACY_INTRO.replace(
        "You are a Kaggle grandmaster attending a competition. ",
        "You are an expert machine learning engineer attempting a task. ",
    )


def test_predicted_evidence_is_labelled_as_a_prediction(recorder, tmp_path):
    outcomes = [_outcome(predicted=True)]
    _agent()._review_candidates(CANDS[:1], outcomes, tmp_path)
    sent = recorder[-1]["system_message"]
    assert "Predicted execution output" in sent
    assert "Execution output" not in sent
    assert "NOT been executed" in sent["Introduction"]


def test_real_evidence_keeps_the_observed_wording(recorder, tmp_path):
    outcomes = [_outcome(predicted=False)]
    _agent()._review_candidates(CANDS[:1], outcomes, tmp_path)
    sent = recorder[-1]["system_message"]
    assert "Execution output" in sent and "Predicted execution output" not in sent


def test_candidate_reviews_never_emit_the_datagen_node_anchors(
    recorder, tmp_path, caplog
):
    """extract_aide_logs.py keys the world-model corpus on these two strings;
    N candidate reviews per step must not look like N journal nodes."""
    with caplog.at_level("INFO", logger="aide"):
        _agent()._review_candidates(CANDS, [_outcome(), _outcome()], tmp_path)
    logged = "\n".join(r.getMessage() for r in caplog.records)
    assert "Prompt for parsing execution results" not in logged
    assert "Agent is parsing execution results for node" not in logged
    assert logged.count("[best-of-N candidate review]") == 2


# --- the judge sees it, and only when it exists ----------------------------

BASE = [{"idx": 0, "code": "a = 1", "term_out": "ok"},
        {"idx": 1, "code": "b = 2", "term_out": "ok"}]


def _judge_prompt(cands):
    seen: dict = {}

    def query(prompt):
        seen.clear()
        seen.update(prompt)
        return {"best_candidate": 0}

    _extract_listwise_pick()([dict(c) for c in cands], "task", query)
    return seen


def test_judge_prompt_without_analysis_is_byte_identical():
    plain = _judge_prompt(BASE)
    empty = _judge_prompt([dict(c, analysis="") for c in BASE])
    missing = _judge_prompt([dict(c, analysis=None) for c in BASE])
    assert plain == empty == missing
    assert "Reviewer findings" not in json.dumps(plain)


def test_judge_prompt_carries_the_findings_when_present():
    seen = _judge_prompt([dict(c, analysis=f"verdict {c['idx']}") for c in BASE])
    assert seen["Candidate 0"].endswith("Reviewer findings:\nverdict 0")
    assert "verdict 1" in seen["Candidate 1"]


def test_candidate_output_is_windowed_like_a_node_review(recorder, tmp_path):
    """CandidateOutcome.term_out is RAW (Node.term_out trims on the way out), so
    an unwindowed review prompt would put a progress-bar spam log on the wire."""
    outcomes = [_outcome(term_out="x" * 40_000)]
    _agent()._review_candidates(CANDS[:1], outcomes, tmp_path)
    sent = recorder[-1]["system_message"]["Predicted execution output"]
    assert "characters truncated" in sent
    assert len(sent) < 6_000


INVARIANT_HEAD = [
    "Introduction", "Task description", "Evaluation metric",
    "Validation reliability",
]


def test_node_prompt_puts_the_invariant_head_first_too(recorder):
    """One order for both callers: the node review recurs every step with the
    same task description + guidance, so it is the biggest prefix-cache win."""
    agent = _agent()
    agent.acfg.eval_metric = "AUC"
    agent._review("code", "out")
    assert list(recorder[-1]["system_message"]) == INVARIANT_HEAD + [
        "Implementation", "Execution output",
    ]


def test_candidate_prompt_puts_the_invariant_head_first(recorder, tmp_path):
    """Everything before Implementation is identical across the N reviews, so
    the server's prefix cache pays for it once."""
    agent = _agent()
    agent.acfg.eval_metric = "AUC"
    agent._review_candidates(CANDS, [_outcome(), _outcome()], tmp_path)
    keys = [list(c["system_message"]) for c in recorder]
    assert keys[0] == keys[1] == INVARIANT_HEAD + [
        "Implementation", "Predicted execution output",
    ]
    heads = {
        json.dumps({k: c["system_message"][k] for k in INVARIANT_HEAD}, sort_keys=True)
        for c in recorder
    }
    assert len(heads) == 1  # byte-identical shared prefix


def test_judge_prompt_unchanged_when_metric_fields_absent():
    """metric must be as inert as analysis when not supplied: the J0 baseline of
    any sweep has to be the prompt the arm really sends."""
    plain = _judge_prompt(BASE)
    nones = _judge_prompt([dict(c, metric=None, analysis=None) for c in BASE])
    assert plain == nones
    assert "reviewer notes" not in plain


def test_judge_prompt_renders_metric_when_present():
    seen = _judge_prompt([dict(c, metric=0.9 - 0.1 * c["idx"]) for c in BASE])
    assert "reviewer notes" in seen
    assert seen["Candidate 0"].startswith("Reviewer metric = 0.9\n")
    # the rubric names only the fields actually supplied
    assert "Reviewer findings" not in seen["reviewer notes"]


def test_trust_verdict_is_never_surfaced_to_the_judge():
    """metric_trustworthy stays a NODE-review field. It fires on 4.3% of
    candidates, added nothing over `analysis` in the 820-set replay, and shown
    to the judge it pushed toward discounting the candidate, not its score."""
    seen = _judge_prompt([dict(c, metric=0.9, metric_trust=(c["idx"] == 0))
                          for c in BASE])
    # scoped to the blocks the reviewer feeds: "UNTRUSTWORTHY" also appears in
    # the judge's own long-standing `metric reliability` rubric, which is not
    # the reviewer's verdict and must stay.
    touched = json.dumps([seen["reviewer notes"], seen["Candidate 0"], seen["Candidate 1"]])
    assert "Reviewer trust" not in touched
    assert "UNTRUSTWORTHY" not in touched


def test_shipped_default_is_on():
    """Pins the 2026-08-21 decision. config.yaml overrides the dataclass, so a
    flip that touches only one file silently does nothing -- assert the merged
    value, not either source."""
    from omegaconf import OmegaConf
    from aide.utils.config import AgentConfig, _load_cfg
    merged = OmegaConf.merge(OmegaConf.structured(AgentConfig),
                             _load_cfg(use_cli_args=False).agent)
    assert merged.bon_candidate_review is True
