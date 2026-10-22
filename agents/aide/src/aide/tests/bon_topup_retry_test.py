"""Tests for the BoN top-up retry in Agent._generate_n_candidates.

Without a retry, a first-sample completion with no extractable code entered
the candidate pool as raw prose (3-11% of BoN candidates measured on the
07-13/08-01 4B/9B/122B bestN_real runs), while the single-shot path
(plan_and_code_query, retries=3) ended at 0.6%. The top-up re-samples every
still-failed slot in one batched query per round.
"""

from types import SimpleNamespace

from aide import agent as agent_mod
from aide.agent import Agent


GOOD_A = "```python\na = 1\n```"
GOOD_B = "```python\nb = 2\n```"
GOOD_C = "```python\nc = 3\n```"
PROSE_1 = "The user wants me to improve the previous solution"
PROSE_2 = "Thinking Process:\n\n1.  **Analyze the Request:**"


def _agent() -> Agent:
    instance = object.__new__(Agent)
    instance.acfg = SimpleNamespace(
        code=SimpleNamespace(model="code-model", temp=1.0),
        convert_system_to_user=False,
    )
    return instance


def _install_query(monkeypatch, batches):
    """Fake query returning canned batches; records each request's n."""
    calls = []

    def fake_query(**kwargs):
        calls.append(kwargs["n"])
        batch = batches.pop(0)
        # Mirror the real backend contract: bare string for n=1.
        if kwargs["n"] == 1:
            assert len(batch) == 1
            return batch[0]
        return batch

    monkeypatch.setattr(agent_mod, "query", fake_query)
    return calls


def test_topup_replaces_failed_slots_in_place(monkeypatch):
    calls = _install_query(
        monkeypatch,
        [
            [GOOD_A, PROSE_1, PROSE_2],  # first sample: slots 1 and 2 fail
            [GOOD_B, PROSE_1],           # round 1: slot 1 recovers, slot 2 fails
            [GOOD_C],                    # round 2 (n=1 -> bare string): slot 2 recovers
        ],
    )
    candidates = _agent()._generate_n_candidates({}, 3)

    assert calls == [3, 2, 1]
    assert [code.strip() for _, code in candidates] == ["a = 1", "b = 2", "c = 3"]


def test_gives_up_after_rounds_and_keeps_raw_text(monkeypatch):
    calls = _install_query(
        monkeypatch,
        [
            [GOOD_A, PROSE_1],
            [PROSE_2],  # round 1 fails again (n=1 -> bare string)
            [PROSE_2],  # round 2 fails again
        ],
    )
    candidates = _agent()._generate_n_candidates({}, 2)

    assert calls == [2, 1, 1]
    assert candidates[0][1].strip() == "a = 1"
    # The dud survives as best-effort raw text (executor reports the real
    # SyntaxError), exactly like the pre-fix fallback. Failed replacements
    # are discarded, so the ORIGINAL sample's text is what survives.
    assert candidates[1] == ("", PROSE_1)


def test_no_topup_call_when_all_parse(monkeypatch):
    calls = _install_query(monkeypatch, [[GOOD_A, GOOD_B]])
    candidates = _agent()._generate_n_candidates({}, 2)

    assert calls == [2]
    assert [code.strip() for _, code in candidates] == ["a = 1", "b = 2"]


def test_short_topup_batch_leaves_remaining_slots_untouched(monkeypatch):
    # Vertex can return fewer choices than requested: round 1 asks n=2 but
    # gets one text back; the unmatched slot keeps failing into round 2.
    calls = _install_query(
        monkeypatch,
        [
            [PROSE_1, PROSE_2, GOOD_A],
            [GOOD_B],        # short batch for n=2
            [PROSE_1],       # round 2 for the remaining slot: fails
        ],
    )
    candidates = _agent()._generate_n_candidates({}, 3)

    assert calls == [3, 2, 1]
    assert candidates[0][1].strip() == "b = 2"
    assert candidates[1] == ("", PROSE_2)
    assert candidates[2][1].strip() == "a = 1"


def test_topup_rounds_zero_reproduces_legacy_single_shot(monkeypatch):
    calls = _install_query(monkeypatch, [[GOOD_A, PROSE_1]])
    candidates = _agent()._generate_n_candidates({}, 2, topup_rounds=0)

    assert calls == [2]
    assert candidates[1] == ("", PROSE_1)
