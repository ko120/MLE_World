"""Shared WM-Fuse generation contract and response parsing.

This module is deliberately independent of Agent state.  The offline replay
harness and the deployed agent should call the same builder so prompt
experiments cannot drift from inference.

The builder accepts only fuser-visible evidence.  Real candidate labels and
grader scores belong to the evaluation harness and are not part of these data
classes.  ``report_source="real"`` means the reports are finished real
executions: the offline oracle-feedback bound, and online fusion under
``agent.bon_oracle=real`` (where every candidate really ran before fusing).

This file owns the data classes, evidence guards, candidate ordering, size
limits, hashing, and response parsing. The current prompt and its helpers live
in :mod:`aide.fusion_prompts.typed_paco_v13`. Keep its rendered wording and
version identifiers stable so existing experiment request hashes remain valid.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from hashlib import sha256
from typing import Any, Mapping, Sequence

from .fusion_prompts import PROMPT_FAMILIES

# Compatibility re-exports: these helpers lived here before the family split
# and are imported by tests and analysis scripts.
from .fusion_prompts.common import (  # noqa: F401
    _CODER_ONLY_SECTIONS,
    _cap_memory_section,
    _clip_output,
    _collapse_progress_output,
    _compile_prompt_value,
    _metric_direction_text,
    _report_output_text,
    _strip_coder_boilerplate,
    _tail,
)


# Historical configuration-fingerprint stamp. The rendered prompt has its own
# v13 version; keep this stamp stable so existing experiment hashes still match.
FUSION_PROMPT_VERSION = "wm-fuse-generation-v2"
FUSION_STAGES = frozenset({"draft", "improve", "debug"})
FUSION_MODES = frozenset({"backbone", "synth", "debug_patch", "revise_one"})
REPORT_SOURCES = frozenset({"wm", "real", "none"})
CANDIDATE_ORDERS = frozenset({"input", "worst_to_best", "best_to_worst"})
SUBMISSION_VALUES = frozenset({"true", "false", "unknown"})
PROMPT_STYLES = frozenset(PROMPT_FAMILIES)


def prompt_version_for_style(style: str) -> str:
    """Return the version stamp used in experiment request hashes."""

    try:
        family = PROMPT_FAMILIES[style]
    except KeyError as exc:
        raise ValueError(
            f"unknown prompt style {style!r}; expected one of "
            f"{sorted(PROMPT_FAMILIES)}"
        ) from exc
    return getattr(family, "CASE_REQUEST_VERSION", FUSION_PROMPT_VERSION)


def _nonempty(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _as_submission_value(value: str | bool | None) -> str:
    if value is True:
        return "true"
    if value is False:
        return "false"
    if value is None:
        return "unknown"
    normalized = str(value).strip().lower()
    if normalized not in SUBMISSION_VALUES:
        raise ValueError(
            "submission_exists must be true, false, unknown, bool, or None"
        )
    return normalized


@dataclass(frozen=True)
class FusionReport:
    """Minimal fuser-visible execution evidence."""

    source: str
    submission_exists: str | bool | None = "unknown"
    observed_output: str = ""

    def __post_init__(self) -> None:
        if self.source not in REPORT_SOURCES - {"none"}:
            raise ValueError("FusionReport.source must be 'wm' or 'real'")
        object.__setattr__(
            self, "submission_exists", _as_submission_value(self.submission_exists)
        )


@dataclass(frozen=True)
class FusionCandidate:
    """One candidate artifact supplied to the fuser.

    ``analysis`` is a post-execution review of THIS candidate, in the same
    class as ``FusionParent.analysis``.  AIDE reviews only the candidate it
    picks (that one becomes a journal node), so per-candidate analyses do not
    exist in a stock best-of-n log and the field is normally empty; families
    render the slot only when it is populated.  Online it is populated iff
    ``agent.bon_candidate_review`` is on, which reviews every drawn candidate
    -- the offline bench has no such cases, so a fused prompt built with the
    flag on carries a block none of the banked ones do.
    """

    index: int
    code: str
    plan: str = ""
    analysis: str = ""
    report: FusionReport | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.index, int) or self.index < 0:
            raise ValueError("candidate index must be a non-negative integer")
        _nonempty(self.code, "candidate code")


@dataclass(frozen=True)
class FusionParent:
    """Real, logged evidence about the node the candidates were built from.

    AIDE embeds only part of this in the coder request it sends: the improve
    request carries the parent's CODE alone, and the debug request adds its
    execution output plus (under ``AIDE_DEBUG_USE_ANALYSIS``) the reviewer's
    diagnosis.  The parent's plan is never embedded, and no improve request
    carries the output or the diagnosis at all — so a family that reconstructs
    the parent purely from ``original_prompt`` cannot show them.  This carries
    them explicitly, sourced from the journal node.

    Every field is a real observation from the search — the plan that produced
    the code, the reviewer's post-execution analysis, the code, the logged
    terminal output, and whether the harness found a submission afterwards.
    Grader scores and candidate labels remain evaluator-only and must not be
    routed through here.

    ``submission_exists`` is the parent's counterpart of
    :class:`FusionReport`'s field of the same meaning, and it is NOT derivable
    from anything else here.  Measured on the 120 dev debug cases: of the nine
    parents that ran to completion and wrote no submission, **five state in
    both their stdout AND the reviewer analysis that they saved one** ("Saved
    predictions to ./submission/submission.csv"), so a family that infers the
    fact from the output is reading a lie.  ``None`` means the bench predates
    the field — a family that needs it must reject that case rather than
    render a default, because defaulting to ``False`` asserts the opposite of
    the truth on 1,340 of the bench's 3,120 non-draft cases.
    """

    plan: str = ""
    analysis: str = ""
    code: str = ""
    execution_output: str = ""
    submission_exists: bool | None = None
    # Historical oracle cases contain real evidence. Online journal nodes can
    # instead contain a WM prediction; new families must preserve that source.
    source: str = "real"

    def __post_init__(self) -> None:
        if self.source not in {"real", "wm"}:
            raise ValueError("FusionParent.source must be 'real' or 'wm'")
        for name in ("plan", "analysis", "code", "execution_output"):
            if not isinstance(getattr(self, name), str):
                raise TypeError(f"FusionParent.{name} must be a string")
        if self.submission_exists is not None and not isinstance(
            self.submission_exists, bool
        ):
            raise TypeError(
                "FusionParent.submission_exists must be a bool or None"
            )


@dataclass(frozen=True)
class FusionContext:
    """Trusted task/search context that the original coder received.

    ``parent`` is the exception to "what the coder received": it is the
    parent node's own logged evidence, which AIDE only partially embeds in the
    request (see :class:`FusionParent`).  It stays optional so families that
    promote the parent out of ``original_prompt`` are unaffected.
    """

    stage: str
    original_prompt: Mapping[str, Any] | str
    task_id: str = ""
    evaluation_metric: str | None = None
    lower_is_better: bool | None = None
    parent: FusionParent | None = None

    def __post_init__(self) -> None:
        if self.stage not in FUSION_STAGES:
            raise ValueError(f"stage must be one of {sorted(FUSION_STAGES)}")
        if not isinstance(self.original_prompt, (str, Mapping)):
            raise TypeError("original_prompt must be a string or mapping")
        if self.parent is not None and not isinstance(self.parent, FusionParent):
            raise TypeError("parent must be a FusionParent or None")
        if self.stage == "draft" and self.parent is not None:
            raise ValueError("draft cases have no parent node")


@dataclass(frozen=True)
class FusionPromptConfig:
    """All prompt-varying choices; suitable for hashing and experiment audit."""

    mode: str
    report_source: str = "wm"
    candidate_order: str = "worst_to_best"
    ranking_worst_to_best: tuple[int, ...] | None = None
    backbone_index: int | None = None
    include_candidate_plans: bool = False
    include_candidate_code: bool = True
    report_tail_chars: int = 1500
    max_prompt_chars: int = 360_000
    design_instructions: tuple[str, ...] = ()
    prompt_style: str = "typed_paco_v13"
    report_head_chars: int = 0
    collapse_progress_output: bool = False
    memory_max_chars: int = 0
    # ``request_sections=None`` keeps
    # every original-request section; a tuple is an allowlist over top-level
    # keys, applied AFTER parent promotion so parent keys never need listing.
    request_sections: tuple[str, ...] | None = None
    promote_parent: bool = False
    crop_task_boilerplate: bool = False
    # Family-private (key, value) string pairs, hashed into the fingerprint.
    # An escape hatch for quick family-local experiments; promote a knob to a
    # typed field once it graduates.
    style_params: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if self.mode not in FUSION_MODES:
            raise ValueError(f"mode must be one of {sorted(FUSION_MODES)}")
        if self.prompt_style not in PROMPT_STYLES:
            raise ValueError(f"prompt_style must be one of {sorted(PROMPT_STYLES)}")
        if self.report_head_chars < 0:
            raise ValueError("report_head_chars must be non-negative")
        if self.memory_max_chars < 0:
            raise ValueError("memory_max_chars must be non-negative")
        if self.report_source not in REPORT_SOURCES:
            raise ValueError(
                f"report_source must be one of {sorted(REPORT_SOURCES)}"
            )
        if self.candidate_order not in CANDIDATE_ORDERS:
            raise ValueError(
                f"candidate_order must be one of {sorted(CANDIDATE_ORDERS)}"
            )
        if self.report_tail_chars < 0:
            raise ValueError("report_tail_chars must be non-negative")
        if self.max_prompt_chars <= 0:
            raise ValueError("max_prompt_chars must be positive")
        if self.report_source == "none" and self.ranking_worst_to_best is not None:
            raise ValueError(
                "no-report variants cannot use a report-derived ranking; "
                "use input order"
            )
        if self.report_source == "none" and self.candidate_order != "input":
            raise ValueError(
                "no-report variants must use input order to avoid hidden WM leakage"
            )
        if not self.include_candidate_code:
            if self.mode != "synth":
                raise ValueError(
                    "include_candidate_code=False is only coherent in synth "
                    "mode; backbone/debug_patch/revise_one edit code they "
                    "must be shown"
                )
            if not self.include_candidate_plans and self.report_source == "none":
                raise ValueError(
                    "at least one candidate payload (code, plans, or report) "
                    "must be included"
                )
        if (
            self.candidate_order != "input"
            and self.ranking_worst_to_best is None
            and self.backbone_index is None
        ):
            raise ValueError(
                "ordered variants require ranking_worst_to_best or a selected "
                "backbone; otherwise the order claim is unsupported"
            )
        if self.mode in {"backbone", "revise_one"} and self.backbone_index is None:
            raise ValueError(f"{self.mode} mode requires backbone_index")
        if self.mode == "synth" and self.backbone_index is not None:
            raise ValueError("synth mode must not designate a backbone")
        for instruction in self.design_instructions:
            _nonempty(instruction, "design instruction")
        if self.request_sections is not None:
            object.__setattr__(
                self, "request_sections", tuple(self.request_sections)
            )
            for section in self.request_sections:
                _nonempty(section, "request section")
        object.__setattr__(
            self, "style_params", tuple(tuple(pair) for pair in self.style_params)
        )
        for pair in self.style_params:
            if len(pair) != 2:
                raise ValueError("style_params must be (key, value) pairs")
            _nonempty(pair[0], "style_params key")
            if not isinstance(pair[1], str):
                raise ValueError("style_params values must be strings")


@dataclass(frozen=True)
class BuiltFusionPrompt:
    version: str
    text: str
    prompt_sha256: str
    displayed_candidate_indices: tuple[int, ...]
    display_name_by_index: Mapping[int, str]
    backbone_display_name: str | None
    report_source: str
    mode: str
    stage: str
    prompt_chars: int


@dataclass(frozen=True)
class PromptView:
    """Everything a prompt family may see, assembled centrally.

    Families receive ordering, display naming, and the all-failed flag already
    computed, so the evidence-leak guards cannot be bypassed and no family can
    reach beyond fuser-visible data.
    """

    context: FusionContext
    config: FusionPromptConfig
    ordered: tuple[FusionCandidate, ...]
    display_name_by_index: Mapping[int, str]
    backbone_name: str | None
    all_failed: bool


@dataclass(frozen=True)
class FusionArtifact:
    """Parsed final artifact from any text-producing fusion strategy."""

    status: str
    raw_response: str
    plan: str
    code: str
    code_sha256: str | None
    valid_python: bool


def _ordered_candidates(
    candidates: Sequence[FusionCandidate], config: FusionPromptConfig
) -> list[FusionCandidate]:
    by_index = {candidate.index: candidate for candidate in candidates}
    if len(by_index) != len(candidates):
        raise ValueError("candidate indices must be unique")
    if not candidates:
        raise ValueError("at least one candidate is required")

    if config.mode == "revise_one":
        assert config.backbone_index is not None
        try:
            return [by_index[config.backbone_index]]
        except KeyError as exc:
            raise ValueError("backbone_index is not present in candidates") from exc

    ranking = config.ranking_worst_to_best
    if ranking is not None:
        if len(ranking) != len(candidates) or set(ranking) != set(by_index):
            raise ValueError(
                "ranking_worst_to_best must contain every candidate index exactly once"
            )
        ordered_indices = list(ranking)
    elif config.backbone_index is not None:
        if config.backbone_index not in by_index:
            raise ValueError("backbone_index is not present in candidates")
        ordered_indices = [
            candidate.index
            for candidate in candidates
            if candidate.index != config.backbone_index
        ] + [config.backbone_index]
    else:
        ordered_indices = [candidate.index for candidate in candidates]

    if config.candidate_order == "input":
        ordered_indices = [candidate.index for candidate in candidates]
    else:
        # A declared edit target is intentionally placed at the recency-favored
        # end even when the supplied soft ranking disagrees.  The remaining
        # candidates retain their relative ranking.
        if config.backbone_index is not None:
            ordered_indices = [
                index for index in ordered_indices if index != config.backbone_index
            ] + [config.backbone_index]
        if config.candidate_order == "best_to_worst":
            ordered_indices.reverse()
    return [by_index[index] for index in ordered_indices]


def build_fusion_prompt(
    context: FusionContext,
    candidates: Sequence[FusionCandidate],
    config: FusionPromptConfig,
) -> BuiltFusionPrompt:
    """Build one stable fusion generation prompt."""

    ordered = _ordered_candidates(candidates, config)
    display_name_by_index = {
        candidate.index: f"solution_v{position}"
        for position, candidate in enumerate(ordered)
    }
    backbone_name = (
        display_name_by_index.get(config.backbone_index)
        if config.backbone_index is not None
        else None
    )

    for candidate in ordered:
        if config.report_source == "none":
            continue
        if candidate.report is None:
            raise ValueError(
                f"candidate {candidate.index} has no {config.report_source} report"
            )
        if candidate.report.source != config.report_source:
            raise ValueError(
                f"candidate {candidate.index} report source "
                f"{candidate.report.source!r} does not match variant source "
                f"{config.report_source!r}"
            )

    family = PROMPT_FAMILIES[config.prompt_style]
    family.validate(config, context)
    view = PromptView(
        context=context,
        config=config,
        ordered=tuple(ordered),
        display_name_by_index=display_name_by_index,
        backbone_name=backbone_name,
        all_failed=config.report_source != "none"
        and all(
            candidate.report is not None
            and candidate.report.submission_exists == "false"
            for candidate in ordered
        ),
    )
    return _finalize_built_prompt(
        family.render(view),
        family.VERSION,
        context,
        config,
        ordered,
        display_name_by_index,
        backbone_name,
    )


def _finalize_built_prompt(
    parts: list[str],
    version: str,
    context: FusionContext,
    config: FusionPromptConfig,
    ordered: Sequence[FusionCandidate],
    display_name_by_index: Mapping[int, str],
    backbone_name: str | None,
) -> BuiltFusionPrompt:
    text = "\n".join(parts).strip() + "\n"
    if len(text) > config.max_prompt_chars:
        raise ValueError(
            f"fusion prompt has {len(text)} characters, exceeding "
            f"max_prompt_chars={config.max_prompt_chars}; shorten task context or "
            "reports rather than silently truncating candidate code"
        )
    return BuiltFusionPrompt(
        version=version,
        text=text,
        prompt_sha256=sha256(text.encode("utf-8")).hexdigest(),
        displayed_candidate_indices=tuple(candidate.index for candidate in ordered),
        display_name_by_index=display_name_by_index,
        backbone_display_name=backbone_name,
        report_source=config.report_source,
        mode=config.mode,
        stage=context.stage,
        prompt_chars=len(text),
    )


def parse_fusion_response(raw_response: str) -> FusionArtifact:
    """Parse one rationale + code response with AIDE's production parser."""

    # Lazy import keeps prompt-only tests independent of Black and API modules.
    try:
        from .utils.response import (
            extract_code,
            extract_text_up_to_code,
            is_valid_python_script,
        )
    except ModuleNotFoundError:
        # Prompt-development hosts may intentionally omit Black and the rest of
        # the AIDE runtime.  Keep a formatting-free compatibility parser there;
        # deployed AIDE environments continue to use the production parser.
        import re

        def is_valid_python_script(script: str) -> bool:
            try:
                compile(script, "<fusion>", "exec")
            except SyntaxError:
                return False
            return True

        def _blocks(text: str) -> list[str]:
            return [
                match.group(1)
                for match in re.finditer(
                    r"```(?:python)?\s*\n?(.*?)\n?```", text, re.DOTALL
                )
            ]

        def extract_code(text: str, fallback_to_invalid: bool = False) -> str:
            blocks = _blocks(text)
            if not blocks and text.strip():
                blocks = [text.strip()]
            valid_blocks = [block for block in blocks if is_valid_python_script(block)]
            if valid_blocks:
                return "\n\n".join(valid_blocks)
            return "\n\n".join(blocks) if fallback_to_invalid else ""

        def extract_text_up_to_code(text: str) -> str:
            return text[: text.find("```")].strip() if "```" in text else ""

    raw = raw_response or ""
    # Inline <think> (offline vLLM keeps it; the retired server split it off):
    # parse ONLY the answer after the last </think>. extract_code JOINS all
    # valid fenced blocks, so parsing the full text concatenated think-draft
    # code with the final program — 81/1,240 Stage-1 artifacts executed such
    # merged programs before this fix (07-31). Fall back to the full text only
    # when the post-think region yields nothing.
    answer = raw.rsplit("</think>", 1)[-1] if "</think>" in raw else raw
    valid_code = extract_code(answer) or (
        extract_code(raw) if answer is not raw else ""
    )
    if valid_code:
        code = valid_code
        status = "ok"
        valid = True
    else:
        code = extract_code(answer, fallback_to_invalid=True) or extract_code(
            raw, fallback_to_invalid=True
        )
        valid = bool(code) and is_valid_python_script(code)
        status = "invalid_code" if code else "no_code"
    plan = extract_text_up_to_code(answer)
    return FusionArtifact(
        status=status,
        raw_response=raw,
        plan=plan,
        code=code,
        code_sha256=sha256(code.encode("utf-8")).hexdigest() if code else None,
        valid_python=valid,
    )


def config_fingerprint(config: FusionPromptConfig) -> str:
    """Stable hash used by the offline harness's variant audit."""

    payload = {
        "prompt_version": FUSION_PROMPT_VERSION,
        **config.__dict__,
    }
    return sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
