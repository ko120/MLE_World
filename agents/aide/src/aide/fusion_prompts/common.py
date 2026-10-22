"""Shared payload shapers for WM-Fuse prompt families.

These helpers turn fuser-visible payloads (the original coder request, candidate
execution output) into prompt text.  Family modules under
:mod:`aide.fusion_prompts` own *wording*; this module owns *payload mechanics*
so clipping/compilation behavior stays identical across families.

Nothing here may import API clients or Agent state, and nothing here may accept
evaluator labels.  Functions moved verbatim from the pre-registry
``aide.fusion`` — their bytes feed banked prompt hashes, so behavior changes
require a new family version, not an edit here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Mapping

if TYPE_CHECKING:  # types only; runtime import would be circular
    from aide.fusion import FusionContext, FusionPromptConfig, FusionReport


def _compile_prompt_value(value: Any, depth: int = 1) -> str:
    """Render the original AIDE prompt like backend.compile_prompt_to_md.

    Kept local and pure so prompt construction does not import API clients.
    """

    if isinstance(value, str):
        return value.strip() + "\n"
    if isinstance(value, (list, tuple)):
        return "\n".join(f"- {str(item).strip()}" for item in value) + "\n"
    if isinstance(value, Mapping):
        parts: list[str] = []
        prefix = "#" * depth
        for key, child in value.items():
            parts.append(f"{prefix} {key}\n")
            parts.append(_compile_prompt_value(child, depth + 1))
        return "\n".join(parts)
    return str(value).strip() + "\n"


def _metric_direction_text(context: "FusionContext") -> str:
    if not context.evaluation_metric:
        return "Use the competition's official evaluation metric."
    if context.lower_is_better is True:
        direction = "lower is better"
    elif context.lower_is_better is False:
        direction = "higher is better"
    else:
        direction = "use the direction stated in the task"
    return f"Official metric: {context.evaluation_metric} ({direction})."


def _tail(text: str, limit: int) -> str:
    if limit <= 0 or not text:
        return ""
    if len(text) <= limit:
        return text
    omitted = len(text) - limit
    return f"[... {omitted} earlier characters omitted ...]\n{text[-limit:]}"


def _collapse_progress_output(text: str) -> str:
    """Collapse carriage-return progress frames and repeated lines.

    tqdm-style output rewrites one line hundreds of times via ``\\r``; only the
    final frame carries information. Runs of byte-identical lines (epoch spam,
    repeated warnings) compress to the first occurrence plus a count, so a
    clipped window spends its budget on distinct content.
    """

    if not text:
        return text
    out: list[str] = []
    previous: str | None = None
    repeats = 0
    for raw_line in text.split("\n"):
        line = raw_line[:-1] if raw_line.endswith("\r") else raw_line
        if "\r" in line:
            line = line.rsplit("\r", 1)[-1]
        if line == previous:
            repeats += 1
            continue
        if repeats:
            out.append(f"[... previous line repeated {repeats} more times ...]")
            repeats = 0
        out.append(line)
        previous = line
    if repeats:
        out.append(f"[... previous line repeated {repeats} more times ...]")
    return "\n".join(out)


def _clip_output(text: str, head_chars: int, tail_chars: int) -> str:
    """Head+tail window; ``head_chars=0`` reproduces the legacy tail-only clip.

    The head shows setup prints and early errors, the tail the crash or final
    metrics — together they carry more signal per char than one long tail.
    """

    if head_chars <= 0:
        return _tail(text, tail_chars)
    if not text:
        return ""
    if len(text) <= head_chars + tail_chars:
        return text
    omitted = len(text) - head_chars - tail_chars
    head = text[:head_chars]
    if tail_chars <= 0:
        return f"{head}\n[... {omitted} trailing characters omitted ...]"
    return (
        f"{head}\n[... {omitted} middle characters omitted ...]\n"
        f"{text[-tail_chars:]}"
    )


def _report_output_text(
    report: "FusionReport", config: "FusionPromptConfig"
) -> str:
    output = report.observed_output
    if config.collapse_progress_output:
        output = _collapse_progress_output(output)
    return _clip_output(output, config.report_head_chars, config.report_tail_chars)


def _cap_memory_section(
    original_prompt: Mapping[str, Any] | str, cap: int
) -> Mapping[str, Any] | str:
    """Clip an oversized journal Memory section to a head+tail window.

    The improve-stage journal memory is heavy-tailed (p50 ~7k chars but >100k
    at the tail on the oracle bench); the newest and oldest entries bound the
    search story while the middle is droppable. ``cap=0`` keeps the request
    untouched.
    """

    if cap <= 0 or not isinstance(original_prompt, Mapping):
        return original_prompt
    memory = original_prompt.get("Memory")
    if not isinstance(memory, str) or len(memory) <= cap:
        return original_prompt
    capped = dict(original_prompt)
    head = cap // 2
    capped["Memory"] = _clip_output(memory, head, cap - head)
    return capped


_CODER_ONLY_SECTIONS = frozenset({"Introduction", "Response format"})


def _strip_coder_boilerplate(
    original_prompt: Mapping[str, Any] | str,
) -> Mapping[str, Any] | str:
    """Drop request sections that only instructed the original coder.

    The coder's role intro, response-format contract, and sketch guidelines
    describe how the candidates were produced; they do not constrain the fused
    program and they conflict with the fusion response contract. Environment
    constraints (implementation guideline, packages, data overview, previous
    solutions) survive. Empty sections are dropped outright.
    """

    if not isinstance(original_prompt, Mapping):
        return original_prompt
    stripped: dict[str, Any] = {}
    for key, value in original_prompt.items():
        if key in _CODER_ONLY_SECTIONS:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        if isinstance(value, Mapping):
            kept = {
                child_key: child
                for child_key, child in value.items()
                if child_key not in _CODER_ONLY_SECTIONS
                and "sketch guideline" not in child_key.lower()
            }
            if kept:
                stripped[key] = kept
            continue
        stripped[key] = value
    return stripped
