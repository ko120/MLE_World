"""Pairwise prompt, answer parsing and elimination behavior without model calls."""

import pytest

from aide import preference


def _case(indices=(0, 1, 2, 3)):
    return {
        "task_id": "classification",
        "original_prompt": {"Task description": "Classify plants."},
        "evaluation_metric": "log loss",
        "lower_is_better": True,
        "context_text": "An earlier linear model had validation loss 0.6.",
        "candidates": [
            {"index": i, "plan": f"plan_{i}", "code": f"model_{i} = None"}
            for i in indices
        ],
    }


def _run(case, response=r"\boxed{B}", variant=None):
    verdicts = {}
    for _ in range(20):
        pending, outcome = preference.plan_calls(case, variant or {}, verdicts)
        if not pending:
            return outcome
        for spec in pending:
            assert spec["call_name"] not in verdicts
            verdicts[spec["call_name"]] = preference.parse_preference_response(response)
    pytest.fail("bracket did not terminate")


@pytest.mark.parametrize("size", [1, 2, 3, 4, 5, 8, 9])
def test_bracket_advances_winners_with_exactly_n_minus_one_calls(size):
    outcome = _run(_case(range(size)))
    assert outcome["selected"] == size - 1
    assert len(outcome["pairs"]) == size - 1
    assert outcome["decided_pairs"] == size - 1
    assert outcome["abstained_pairs"] == 0
    assert outcome["budget_exhausted"] is False
    assert all(pair["kind"] == "pair" for pair in outcome["pairs"])


def test_bracket_uses_display_positions_with_noncontiguous_candidate_indices():
    case = _case([21, 8, 42, 3])
    pending, outcome = preference.plan_calls(case, {}, {})
    assert outcome is None
    assert [(p["left"], p["right"]) for p in pending] == [(3, 8), (21, 42)]
    verdicts = {
        pending[0]["call_name"]: preference.parse_preference_response(r"\boxed{B}"),
        pending[1]["call_name"]: preference.parse_preference_response(r"\boxed{A}"),
    }
    final, _ = preference.plan_calls(case, {}, verdicts)
    assert [(p["left"], p["right"]) for p in final] == [(8, 21)]
    verdicts[final[0]["call_name"]] = preference.parse_preference_response(r"\boxed{B}")
    pending, outcome = preference.plan_calls(case, {}, verdicts)
    assert pending == [] and outcome["selected"] == 21


def test_partial_round_requests_only_missing_verdicts():
    case = _case()
    pairs, _ = preference.plan_calls(case, {}, {})
    verdicts = {pairs[0]["call_name"]: {"winner_position": 1}}
    pending, outcome = preference.plan_calls(case, {}, verdicts)
    assert pending == [pairs[1]] and outcome is None


def test_invalid_answers_use_left_fallback_without_extra_calls():
    outcome = _run(_case(range(5)), response="truncated answer")
    assert outcome["selected"] == 0
    assert len(outcome["pairs"]) == 4
    assert outcome["abstained_pairs"] == 4
    assert outcome["basis"] == "bracket_fallback"


@pytest.mark.parametrize("budget,selected", [(1, 1), (2, 1), (3, 3)])
def test_budget_preserves_completed_winners_and_never_overruns(budget, selected):
    outcome = _run(_case(), variant={"limits": {"max_model_calls": budget}})
    assert len(outcome["pairs"]) == budget
    assert outcome["selected"] == selected
    assert outcome["budget_exhausted"] == (budget < 3)


@pytest.mark.parametrize("raw,position", [
    (r"\boxed{A}", 0),
    (r"B has a stronger upgrade path. \boxed{B}", 1),
    (r"Reasoning: \boxed{A}. Final answer: \boxed{B}", 1),
    (r"<think>\boxed{A}</think>\boxed{B}", 1),
    (r"$\boxed{B}$", 1),
    (r"\[\boxed{A}\]", 0),
    ("\\boxed{ B }\n", 1),
])
def test_parser_reads_final_boxed_answer(raw, position):
    assert preference.parse_preference_response(raw) == {
        "status": "ok", "winner_position": position,
    }


@pytest.mark.parametrize("raw", [
    "", "A", r"\boxed{C}", r"\boxed{A} then \boxed{C}",
    r"<think>\boxed{A}</think>", r"<think>\boxed{B}",
    r"Example: \boxed{A} but the final answer is truncated",
    '{"predicted_best_index": 1, "confidence": 0.9}',
])
def test_parser_abstains_on_missing_invalid_or_unfinished_answers(raw):
    assert preference.parse_preference_response(raw)["winner_position"] is None


def test_prompt_maps_plans_code_and_world_model_output_to_the_displayed_pair():
    reports = {i: {"observed_output": f"OUTPUT_{i}", "submission_exists": "true"}
               for i in range(4)}
    prompt = preference.build_preference_prompt(
        _case(), {"left": 3, "right": 1}, {}, reports,
    )
    a, b = prompt.split("Candidate A -- Plan:\n", 1)[1].split("Candidate B -- Plan:\n")
    assert "plan_3" in a and "model_3" in a and "OUTPUT_3" in a
    assert "plan_1" in b and "model_1" in b and "OUTPUT_1" in b
    assert "OUTPUT_0" not in prompt and "OUTPUT_2" not in prompt
    assert "Classify plants." in prompt
    assert "log loss (lower is better)" in prompt
    assert "An earlier linear model" in prompt
    assert "World-model code output" in prompt
    assert "learned simulator's prediction" in prompt
    assert "confidence" not in prompt.lower()
    assert r"\boxed{A}" in prompt and r"\boxed{B}" in prompt


def test_real_reports_have_observed_provenance_and_preserve_the_output_tail():
    prompt = preference.build_preference_prompt(
        _case(), {"left": 0, "right": 1},
        {"foreagent": {"report_tail_chars": 12}},
        {0: {"observed_output": "START" + "x" * 100 + "IMPORTANTEND"}},
        reports_are_predictions=False,
    )
    assert "Observed terminal output:" in prompt and "real execution" in prompt
    assert "World-model code output" not in prompt and "Predicted" not in prompt
    assert "IMPORTANTEND" in prompt and "START" not in prompt


def test_prompt_without_history_or_reports_never_invents_evidence():
    case = _case()
    case.pop("context_text")
    prompt = preference.build_preference_prompt(case, {"left": 0, "right": 1}, {})
    assert "No previous solutions are available." in prompt
    assert "terminal output:" not in prompt and "simulator" not in prompt


@pytest.mark.parametrize("indices", [[], [1, 1]])
def test_empty_or_duplicate_candidate_pool_is_rejected(indices):
    with pytest.raises(preference.ForeagentError, match="distinct candidate indices"):
        preference.plan_calls(_case(indices), {}, {})


def test_old_round_robin_scheme_is_rejected_instead_of_silently_scoring():
    with pytest.raises(preference.ForeagentError, match="score tallies were removed"):
        preference.resolve_config({"foreagent": {"pair_scheme": "round_robin"}})
