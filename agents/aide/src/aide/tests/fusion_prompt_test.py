"""Shared fusion guards, response parsing, and the deployed v13 prompt."""

from dataclasses import replace

import pytest

from aide.fusion import (
    PROMPT_STYLES,
    FusionCandidate,
    FusionContext,
    FusionParent,
    FusionPromptConfig,
    FusionReport,
    _clip_output,
    _collapse_progress_output,
    build_fusion_prompt,
    config_fingerprint,
    parse_fusion_response,
    prompt_version_for_style,
)


@pytest.fixture
def prompt_case():
    context = FusionContext(
        stage="improve",
        task_id="task-a",
        evaluation_metric="log-loss",
        lower_is_better=True,
        original_prompt={
            "Task description": "Fit a classifier.",
            "Data Overview": "input/train.csv\ninput/test.csv",
        },
        parent=FusionParent(
            plan="parent plan",
            analysis="parent review",
            code="print('parent')",
            execution_output="parent output",
        ),
    )
    candidates = [
        FusionCandidate(
            index=i,
            code=f"V{i} = {i}\nprint(V{i})\n",
            plan=f"candidate plan {i}",
            analysis=f"candidate review {i}",
            report=FusionReport(source="wm", submission_exists=True, observed_output=f"score {i}"),
        )
        for i in range(4)
    ]
    config = FusionPromptConfig(
        mode="synth",
        candidate_order="input",
        promote_parent=True,
        include_candidate_plans=True,
    )
    return context, candidates, config


@pytest.mark.parametrize(
    "order,expected",
    [("input", (0, 1, 2, 3)), ("worst_to_best", (3, 1, 2, 0)), ("best_to_worst", (0, 2, 1, 3))],
)
def test_prompt_is_deterministic_and_preserves_candidate_code(prompt_case, order, expected):
    context, candidates, config = prompt_case
    config = replace(config, candidate_order=order, ranking_worst_to_best=(3, 1, 2, 0))
    first = build_fusion_prompt(context, candidates, config)
    assert first == build_fusion_prompt(context, candidates, config)
    assert first.displayed_candidate_indices == expected
    assert first.version == prompt_version_for_style(config.prompt_style)
    for candidate in candidates:
        assert candidate.code.rstrip() in first.text
    assert "<TOTAL_TIME_REMAINING" not in first.text


def test_only_deployed_prompt_is_available(prompt_case):
    assert PROMPT_STYLES == {"typed_paco_v13"}
    with pytest.raises(ValueError, match="prompt_style"):
        replace(prompt_case[2], prompt_style="full")
    with pytest.raises(ValueError, match="unknown prompt style"):
        prompt_version_for_style("full")


def test_order_claim_requires_ranking(prompt_case):
    with pytest.raises(ValueError, match="ordered variants require"):
        replace(prompt_case[2], candidate_order="worst_to_best")


def test_report_source_mismatch_is_rejected(prompt_case):
    context, candidates, config = prompt_case
    with pytest.raises(ValueError, match="does not match variant source"):
        build_fusion_prompt(context, candidates, replace(config, report_source="real"))


@pytest.mark.parametrize("stage", ["improve", "debug"])
def test_parent_evidence_is_required(prompt_case, stage):
    context, candidates, config = prompt_case
    with pytest.raises(ValueError, match="need the journal parent"):
        build_fusion_prompt(replace(context, stage=stage, parent=None), candidates, config)
    with pytest.raises(ValueError, match="need the journal parent"):
        build_fusion_prompt(
            replace(context, stage=stage), candidates, replace(config, promote_parent=False)
        )


def test_prompt_budget_fails_instead_of_truncating_code(prompt_case):
    context, candidates, config = prompt_case
    with pytest.raises(ValueError, match="rather than silently truncating"):
        build_fusion_prompt(context, candidates, replace(config, max_prompt_chars=100))


def test_report_windows_flow_into_prompt(prompt_case):
    context, candidates, config = prompt_case
    long_output = "setup line\n" + ("x" * 4000) + "\nfinal metric 0.9"
    candidates = [
        replace(c, report=replace(c.report, observed_output=long_output)) for c in candidates
    ]
    config = replace(
        config, report_head_chars=10, report_tail_chars=16, collapse_progress_output=True
    )
    text = build_fusion_prompt(context, candidates, config).text
    assert "setup line\n[... " in text
    assert "middle characters omitted ...]\nfinal metric 0.9" in text


def test_memory_cap_clips_only_oversized_memory(prompt_case):
    context, candidates, config = prompt_case
    context = replace(
        context,
        original_prompt={
            **context.original_prompt,
            "Memory": "OLD " * 300 + "MID " * 300 + "NEW " * 300,
        },
    )
    capped = replace(config, memory_max_chars=400)
    text = build_fusion_prompt(context, candidates, capped).text
    assert "OLD" in text and "NEW" in text and "MID" not in text
    assert "middle characters omitted" in text
    assert "MID" in build_fusion_prompt(context, candidates, config).text
    small = replace(context, original_prompt={"Memory": "tiny"})
    text = build_fusion_prompt(small, candidates, capped).text
    assert "tiny" in text and "characters omitted" not in text


def test_parent_request_values_take_precedence_and_sections_are_filtered(prompt_case):
    context, candidates, config = prompt_case
    context = replace(
        context,
        original_prompt={
            "Task description": "excluded task",
            "Previous solution": {"Code": "```python\nprint('request parent')\n```"},
            "Execution output": "request output",
            "Instructions": {
                "Implementation guideline": ["retained guideline"],
                "Response format": "excluded format",
            },
        },
    )
    config = replace(config, request_sections=("Instructions.Implementation guideline",))
    text = build_fusion_prompt(context, candidates, config).text
    assert "print('request parent')" in text and "print('parent')" not in text
    assert "request output" in text and "parent output" not in text
    assert "parent plan" in text and "parent review" in text
    assert "retained guideline" in text
    assert "excluded task" not in text and "excluded format" not in text


def test_draft_has_no_parent_block(prompt_case):
    context, candidates, config = prompt_case
    context = replace(context, stage="draft", parent=None)
    assert "<previous_solution>" not in build_fusion_prompt(context, candidates, config).text
    with pytest.raises(ValueError, match="draft cases have no parent"):
        replace(context, parent=prompt_case[0].parent)


def test_config_fingerprint_covers_instructions(prompt_case):
    config = prompt_case[2]
    changed = replace(config, design_instructions=("Preserve the loader.",))
    assert config_fingerprint(config) != config_fingerprint(changed)


@pytest.mark.parametrize("option", ["report_head_chars", "memory_max_chars"])
def test_negative_window_is_rejected(prompt_case, option):
    with pytest.raises(ValueError, match=option):
        replace(prompt_case[2], **{option: -1})


def test_response_parser_retains_valid_and_invalid_code():
    valid = parse_fusion_response(
        "I retained the reliable loader.\n```python\nx = 1\nprint(x)\n```"
    )
    assert valid.status == "ok" and valid.valid_python
    assert valid.plan == "I retained the reliable loader."
    assert "x = 1" in valid.code
    invalid = parse_fusion_response("reason\n```python\nif True print('x')\n```")
    assert invalid.status == "invalid_code" and not invalid.valid_python
    assert "if True" in invalid.code


def test_parser_ignores_code_drafts_inside_think_block():
    raw = (
        "<think>plan it out\n```python\nx = 'draft'\nprint(x)\n```\nno, better:\n</think>\n"
        "Final answer.\n```python\ny = 'final'\nprint(y)\n```"
    )
    parsed = parse_fusion_response(raw)
    assert parsed.status == "ok"
    assert "final" in parsed.code and "draft" not in parsed.code
    fallback = parse_fusion_response("<think>\n```python\nz = 1\n```\n</think>\n")
    assert fallback.status == "ok" and "z = 1" in fallback.code


def test_collapse_progress_output_and_head_tail_clip():
    text = "epoch 1: 10%\repoch 1: 50%\repoch 1: 100%\nloss=1\nloss=1\nloss=1\ndone"
    assert (
        _collapse_progress_output(text)
        == "epoch 1: 100%\nloss=1\n[... previous line repeated 2 more times ...]\ndone"
    )
    assert _collapse_progress_output("a\r\nb\r\n") == "a\nb\n"
    blob = "A" * 10 + "B" * 10 + "C" * 10
    assert _clip_output(blob, 5, 5) == "AAAAA\n[... 20 middle characters omitted ...]\nCCCCC"
    assert _clip_output(blob, 0, 5) == "[... 25 earlier characters omitted ...]\nCCCCC"
    assert _clip_output(blob, 20, 20) == blob
