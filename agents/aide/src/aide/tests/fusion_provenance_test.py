from dataclasses import replace
from types import SimpleNamespace

import pytest

from aide.agent import Agent
from aide.fusion import (
    FusionCandidate,
    FusionContext,
    FusionParent,
    FusionPromptConfig,
    FusionReport,
    build_fusion_prompt,
)


def fixture(stage="improve", candidate_source="real", parent_source="real"):
    context = FusionContext(
        stage=stage,
        original_prompt={"Task description": "omitted task"},
        parent=FusionParent(
            code="print('parent')",
            plan="base",
            analysis="review",
            execution_output="parent output",
            source=parent_source,
        ),
        evaluation_metric="accuracy",
        lower_is_better=False,
    )
    candidates = [
        FusionCandidate(
            index=i,
            code=f"print({i})",
            plan=f"plan {i}",
            report=FusionReport(
                source=candidate_source,
                submission_exists="false",
                observed_output=f"candidate output {i}",
            ),
        )
        for i in range(4)
    ]
    config = FusionPromptConfig(
        mode="synth",
        report_source=candidate_source,
        candidate_order="input",
        prompt_style="typed_paco_v13",
        promote_parent=True,
        request_sections=(),
        include_candidate_plans=True,
    )
    return context, candidates, config


@pytest.mark.parametrize(
    "stage,expected_sha256",
    [
        ("improve", "5c875e38b50e96368e60bb622e8c396c9160ffe5a13db5b82957eef37c494c11"),
        ("debug", "cc42c01975807e7de0692c57765f587b219c48a14203d06e3cc299a1e4fd2052"),
    ],
)
def test_oracle_control_preserves_frozen_prompt_bytes(stage, expected_sha256):
    # Captured from the original v13 controls before consolidating their helpers.
    context, candidates, config = fixture(stage)
    current = build_fusion_prompt(context, candidates, config)
    assert current.prompt_sha256 == expected_sha256
    assert current.version == "wm-fuse-generation-v13-provenance-debug"


@pytest.mark.parametrize(
    "candidate_source,parent_source",
    [
        ("wm", "real"),
        ("wm", "wm"),
        ("real", "wm"),
        ("real", "real"),
    ],
)
def test_each_evidence_source_is_independent(candidate_source, parent_source):
    context, candidates, config = fixture(
        candidate_source=candidate_source, parent_source=parent_source
    )
    text = build_fusion_prompt(context, candidates, config).text
    assert f'<execution source="{parent_source}">' in text
    assert text.count(f'source="{candidate_source}"') == 4 + int(candidate_source == parent_source)
    if "wm" in (candidate_source, parent_source):
        assert "Each execution report is the logged output of actually running" not in text
        assert "not an observed fact" in text
    if candidate_source == "wm":
        assert text.count("Predicted nonempty submission CSV:") == 4
    assert "omitted task" not in text


def test_improve_transfer_is_the_same_rule_block_on_debug():
    context, candidates, config = fixture()
    improve = build_fusion_prompt(context, candidates, config).text
    debug = build_fusion_prompt(
        replace(context, stage="debug"),
        candidates,
        replace(config, style_params=(("debug_rules", "improve_transfer"),)),
    ).text
    assert improve == debug


def test_interface_arm_changes_only_one_rule():
    context, candidates, config = fixture("debug")
    control = build_fusion_prompt(context, candidates, config).text.splitlines()
    arm = build_fusion_prompt(
        context, candidates, replace(config, style_params=(("debug_rules", "interface_repair"),))
    ).text.splitlines()
    addition = [line for line in arm if line not in control]
    assert len(addition) == 1
    assert addition[0].startswith("14. Before finalizing a repair")
    assert "fewer epochs" in addition[0]
    assert [line for line in arm if line != addition[0]] == control


def test_predicted_parent_prefix_is_not_claimed_as_verified():
    context, candidates, config = fixture("debug", "wm", "wm")
    text = build_fusion_prompt(context, candidates, config).text
    assert "proves neither an executed prefix" in text
    assert "point is verified" not in text


def test_stage_settings_do_not_change_improve_when_debug_changes():
    agent = object.__new__(Agent)
    agent.acfg = SimpleNamespace(
        fuse_prompt_style="typed_paco_v13",
        fuse_prompt_style_by_stage={"improve": "typed_paco_v13", "debug": "typed_paco_v13"},
        fuse_style_params_by_stage={"debug": {"debug_rules": "improve_transfer"}},
    )
    assert agent._fuse_prompt_settings("improve") == ("typed_paco_v13", ())
    assert agent._fuse_prompt_settings("debug") == (
        "typed_paco_v13",
        (("debug_rules", "improve_transfer"),),
    )
    assert agent._fuse_prompt_settings("draft") == ("typed_paco_v13", ())
    agent.acfg.fuse_prompt_style_by_stage["imrpove"] = "typed_paco_v13"
    with pytest.raises(ValueError, match="unknown fusion stage"):
        agent._fuse_prompt_settings("improve")


def test_unknown_arm_fails_instead_of_silently_testing_control():
    context, candidates, config = fixture("debug")
    with pytest.raises(ValueError, match="unknown v13 debug_rules"):
        build_fusion_prompt(
            context, candidates, replace(config, style_params=(("debug_rules", "typo"),))
        )
