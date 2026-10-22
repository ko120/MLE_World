"""Pairwise A/B judging with a single-elimination bracket.

The judge prompt adapts Figure 7 of Foster et al., "AI Research Preference
Models", https://arxiv.org/abs/2608.13940 (CC BY 4.0,
https://creativecommons.org/licenses/by/4.0/). It adds metric metadata and
candidate execution output, and refers back to context instead of repeating it.

Each comparison advances one candidate. There are no confidence scores,
revival matches, or aggregate rankings. Failed verdicts advance the left
candidate and are recorded as abstentions. An uncapped N-candidate bracket
always takes N-1 calls, including failed calls.

Prompt/response changes require new versions so cached judge responses cannot
be mistaken for verdicts under the previous prompt.
"""

from __future__ import annotations

import re
from typing import Any, Mapping

PROMPT_VERSION = "wm-rpm-fig7-pairwise-v2"
CASE_REQUEST_VERSION = PROMPT_VERSION

DEFAULTS: dict[str, Any] = {
    "pair_scheme": "bracket",
    "include_task_description": True,
    "include_metric": True,
    "report_tail_chars": 1500,
}


class ForeagentError(RuntimeError):
    """Invalid pairwise selector configuration or candidate pool."""


def resolve_config(variant: Mapping[str, Any]) -> dict[str, Any]:
    """Resolve prompt options; only the simple elimination bracket is supported."""
    block = variant.get("foreagent") or {}
    unknown = sorted(set(block) - set(DEFAULTS))
    if unknown:
        raise ForeagentError(f"unknown pairwise option(s): {', '.join(unknown)}")
    config = {**DEFAULTS, **block}
    if config["pair_scheme"] != "bracket":
        raise ForeagentError("pair_scheme must be 'bracket'; score tallies were removed")
    config["report_tail_chars"] = int(config["report_tail_chars"])
    if config["report_tail_chars"] < 0:
        raise ForeagentError("report_tail_chars must be nonnegative (0 means unlimited)")
    return config


def plan_calls(
    case: Mapping[str, Any],
    variant: Mapping[str, Any],
    verdicts: Mapping[str, Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    """Replay adjacent pairs and return the next round's missing calls or winner.

    Odd candidates receive a bye. An invalid response advances the left
    candidate. If the call budget runs out, select the first remaining
    candidate, including winners from a partially completed round.
    """
    resolve_config(variant)
    active = sorted(int(candidate["index"]) for candidate in case["candidates"])
    if not active or len(set(active)) != len(active):
        raise ForeagentError("pairwise selection needs distinct candidate indices")
    max_calls = int((variant.get("limits") or {}).get("max_model_calls", 0))
    if max_calls < 0:
        raise ForeagentError("max_model_calls must be nonnegative (0 means unlimited)")

    pairs: list[dict[str, Any]] = []
    rounds = byes = decided = abstained = 0
    budget_exhausted = False
    while len(active) > 1:
        rounds += 1
        advanced: list[int] = []
        pending: list[dict[str, Any]] = []
        for offset in range(0, len(active), 2):
            if offset + 1 == len(active):
                advanced.append(active[offset])
                byes += 1
                continue
            if max_calls and len(pairs) >= max_calls:
                advanced.extend(active[offset:])
                budget_exhausted = True
                break
            left, right = active[offset:offset + 2]
            spec = {
                "call_index": len(pairs),
                "call_name": f"r{rounds}_pair_{left}_{right}",
                "left": left,
                "right": right,
                "round": rounds,
                "kind": "pair",
            }
            pairs.append(spec)
            if spec["call_name"] not in verdicts:
                pending.append(spec)
                continue
            position = verdicts[spec["call_name"]].get("winner_position")
            if position in (0, 1):
                decided += 1
            else:
                abstained += 1
            advanced.append(right if position == 1 else left)
        if pending:
            return pending, None
        active = advanced
        if budget_exhausted:
            break

    basis = "bracket_fallback" if abstained else "bracket"
    if budget_exhausted:
        basis += "_budget_capped"
    return [], {
        "selected": active[0],
        "basis": basis,
        "budget_exhausted": budget_exhausted,
        "decided_pairs": decided,
        "abstained_pairs": abstained,
        "rounds": rounds,
        "byes": byes,
        "pairs": pairs,
    }


def _report_block(
    report: Mapping[str, Any] | None, tail_chars: int, predicted: bool
) -> str:
    """Label predicted versus real output and retain the configured output tail."""
    if not report:
        return ""
    word = "Predicted" if predicted else "Observed"
    lines = []
    exists = report.get("submission_exists")
    if exists is not None:
        lines.append(f"{word} submission produced: {exists}")
    output = str(report.get("observed_output") or "").strip()
    if output:
        if tail_chars and len(output) > tail_chars:
            output = "...\n" + output[-tail_chars:]
        lines.extend([f"{word} terminal output:", "~~~text", output, "~~~"])
    return "\n".join(lines)


def build_preference_prompt(
    case: Mapping[str, Any],
    pair: Mapping[str, Any],
    variant: Mapping[str, Any],
    reports: Mapping[int, Mapping[str, Any]] | None = None,
    reports_are_predictions: bool = True,
) -> str:
    """Figure 7's judge with candidate plans, code, context and optional output.

    Reports map candidate indices to submission_exists / observed_output.
    Their values are evidence for the judge, never a numerical selection rule.
    """
    config = resolve_config(variant)
    reports = reports or {}
    task = str(case.get("task_id") or "unknown")
    if config["include_task_description"]:
        description = (case.get("original_prompt") or {}).get("Task description")
        if description:
            task += f"\n{description}"
    metric = case.get("evaluation_metric")
    if config["include_metric"] and metric:
        lower = case.get("lower_is_better")
        direction = "" if lower is None else (
            " (lower is better)" if lower else " (higher is better)"
        )
        task += f"\nEvaluation metric: {metric}{direction}."

    context = str(case.get("context_text") or "No previous solutions are available.")
    candidates = {int(c["index"]): c for c in case["candidates"]}
    sections = []
    for label, side in (("A", "left"), ("B", "right")):
        index = int(pair[side])
        if index not in candidates:
            raise ForeagentError(f"candidate {index} is absent from the comparison pool")
        candidate = candidates[index]
        section = (
            f"Candidate {label} -- Plan:\n{candidate.get('plan') or '(No plan provided.)'}\n"
            f"Candidate {label} -- Code:\n~~~python\n{candidate['code']}\n~~~"
        )
        report = _report_block(
            reports.get(index), config["report_tail_chars"], reports_are_predictions
        )
        if report:
            source = "World-model code output" if reports_are_predictions else "Observed code output"
            section += f"\nCandidate {label} -- {source}:\n{report}"
        sections.append(section)
    evidence = ""
    if reports:
        evidence = (
            "\nThe predicted terminal output is a learned simulator's prediction "
            "of running that program, not a measured execution.\n"
            if reports_are_predictions else
            "\nThe observed terminal output comes from a real execution of that program.\n"
        )

    # Figure 7 (CC BY 4.0); additions described in the module attribution.
    return f"""You are a principal investigator allocating compute budget to one of two branches. Decide which branch is
more likely to yield the best eventual test score after several iterations. Emphasize extensibility,
fixability, and promise relative to what has already been tried.

Task description:
{task}

Context from other solutions and their scores (not the candidates). Use this to identify promising gaps and
avoid redundant directions:
{context}
{evidence}
{sections[0]}

{sections[1]}

Step-by-step evaluation:
Problem-model fit:
Does each candidate's formulation and objective align with the task? Any risks of leakage or misalignment?
Note which issues are trivially fixable vs. fundamental.
Extensibility and upgrade path:
How modular is the code? How straightforward is it to add stronger models, features, or training strategies
in 1 to 3 iterations?
Learning curve projection:
Based on current choices, estimate how performance might improve over the next few iterations. Identify
low-hanging fruit (data cleaning, features, hyperparameters, regularization, architecture changes).
Context-informed novelty:
Relative to the context above, does the candidate explore a fresh, promising region or iterate intelligently on a
proven one? Avoid branches that mirror underperforming context without new leverage.
Risk-adjusted potential:
Balance upside (ceiling) against effort/risk to realize it. Bugs are acceptable if the approach is sound and
fixes are clear; penalize only for hard-to-remedy conceptual flaws.
Decision policy:
Choose the candidate whose direction offers higher expected long-term best test score and a credible path to
get there.
Output format (STRICT):
Provide reasoning following the steps above.
End with a single final answer: A for Candidate A or B for Candidate B.
Provide your answer inside a \\boxed{{A}} or \\boxed{{B}}.
"""


_BOXED_ANSWER_RE = re.compile(
    r"\\boxed\s*\{\s*([AB])\s*\}\s*(?:\$+|\\[\]\)])?\s*[.!]?\s*$"
)


def parse_preference_response(raw: str) -> dict[str, Any]:
    """Read the final boxed A/B answer, ignoring any earlier reasoning verdicts."""
    raw = raw or ""
    if "<think>" in raw and "</think>" not in raw:
        return {"status": "invalid_answer", "winner_position": None}
    answer = raw.rsplit("</think>", 1)[-1].strip()
    match = _BOXED_ANSWER_RE.search(answer)
    if match is None:
        return {"status": "invalid_answer", "winner_position": None}
    return {"status": "ok", "winner_position": 0 if match[1] == "A" else 1}
