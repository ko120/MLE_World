"""Compact listwise judging ignores legacy confidence inputs and scoring hooks."""

import ast
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from aide import agent as agent_mod
from aide.agent import Agent


# Keep exercising the standalone helper contract used by offline consumers.
source = ast.parse(Path(agent_mod.__file__).read_text())
function = next(n for n in source.body if isinstance(n, ast.FunctionDef) and n.name == "listwise_pick")
namespace = {}
exec(compile(ast.Module([function], []), "<listwise_pick>", "exec"), namespace)
pick = namespace["listwise_pick"]
CANDS = [
    {"idx": 3, "code": "a = 1", "term_out": "Validation score: 0.8"},
    {"idx": 7, "code": "b = 2", "term_out": "Validation score: 0.9"},
]


@pytest.mark.parametrize("predicted", [False, True])
@pytest.mark.parametrize("all_failed", [False, True])
def test_prompt_retains_evidence_source_and_selection_objective(predicted, all_failed):
    query = Mock(return_value={"best_candidate": 7, "reason": "Better candidate."})
    assert pick(CANDS, "Maximize AUC.", query, predicted=predicted, all_failed=all_failed)[0] == 7
    query.assert_called_once()
    prompt = query.call_args.args[0]
    assert prompt["Task description"] == "Maximize AUC."
    assert ("simulator predictions" if predicted else "real executions") in prompt["Introduction"]
    assert ("least buggy" if all_failed else "best held-out test score") in prompt["Introduction"]
    assert ("metric reliability" in prompt) is (not all_failed)
    assert ("Predicted execution output" in prompt["Candidate 3"]) is predicted
    assert "a = 1" in prompt["Candidate 3"]
    assert "[3, 7]" in prompt["Instructions"]


@pytest.mark.parametrize("score_name", ["p_clean", "max_p"])
@pytest.mark.parametrize("all_failed", [False, True])
def test_legacy_confidence_fields_do_not_change_the_prompt(score_name, all_failed):
    query = Mock(return_value={"best_candidate": 7})
    pick(CANDS, "task", query, all_failed=all_failed)
    baseline = query.call_args.args[0]
    with_confidence = [dict(c, p_clean=0.42, top_exc="CONFIDENCE_SENTINEL",
                            p_top_exc=0.38, argmax_class="CONFIDENCE_SENTINEL") for c in CANDS]
    pick(with_confidence, "task", query, all_failed=all_failed, score_name=score_name)
    assert query.call_args.args[0] == baseline


def test_output_tail_excludes_latency_footer_and_keeps_full_code():
    query = Mock(return_value={"best_candidate": 3})
    candidate = dict(CANDS[0], code="FULL_CODE_SENTINEL\n" + "x = 1\n" * 600,
                     term_out="OLD_OUTPUT\n" + "x" * 4000 + "\nRECENT_OUTPUT\nExecution time: 0.5 seconds")
    pick([candidate], "task", query)
    block = query.call_args.args[0]["Candidate 3"]
    assert candidate["code"] in block
    assert "RECENT_OUTPUT" in block and "OLD_OUTPUT" not in block
    assert "Execution time:" not in block


@pytest.mark.parametrize("response", [{}, {"best_candidate": 0}, {"best_candidate": True},
                                      {"best_candidate": "invalid"}, None])
def test_invalid_judge_reply_uses_first_pool_index(response):
    assert pick(CANDS, "task", Mock(return_value=response))[0] == 3


def test_judge_exception_uses_first_pool_index():
    winner, error = pick(CANDS, "task", Mock(side_effect=RuntimeError("judge down")))
    assert winner == 3 and "judge down" in error


def test_integer_string_index_remains_supported():
    assert pick(CANDS, "task", Mock(return_value={"best_candidate": "7"}))[0] == 7


def _agent(tmp_path):
    agent = object.__new__(Agent)
    agent.acfg = SimpleNamespace(use_best_of_n_listwise=True, use_line_frac_salvage=False)
    agent.task_desc = "task"
    agent._p_clean = Mock(side_effect=AssertionError("No confidence calls for listwise."))
    agent._judge_query = Mock(return_value={"best_candidate": 1, "reason": "Best score."})
    return agent


@pytest.mark.parametrize("usable", [(True, True, False), (False, False, False)])
def test_listwise_selection_never_requests_confidence(tmp_path, usable):
    agent = _agent(tmp_path)
    candidates = [("plan", f"code_{i}") for i in range(3)]
    outcomes = [SimpleNamespace(usable=ok, term_out="ok", predicted=True) for ok in usable]
    assert agent._select(candidates, outcomes, tmp_path) == 1
    agent._p_clean.assert_not_called()
    agent._judge_query.assert_called_once()
    prompt = agent._judge_query.call_args.args[0]
    expected = {f"Candidate {i}" for i, ok in enumerate(usable) if ok or not any(usable)}
    assert {key for key in prompt if key.startswith("Candidate ")} == expected
    assert json.loads((tmp_path / "listwise.json").read_text())["winner"] == 1


def test_single_usable_candidate_skips_judge_and_confidence(tmp_path):
    agent = _agent(tmp_path)
    candidates = [("plan", "failed"), ("plan", "usable")]
    outcomes = [SimpleNamespace(usable=ok) for ok in (False, True)]
    assert agent._select(candidates, outcomes, tmp_path) == 1
    agent._judge_query.assert_not_called()
    agent._p_clean.assert_not_called()
