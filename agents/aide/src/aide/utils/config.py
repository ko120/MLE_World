"""Typed configuration, workspace preparation, and run artifacts.

Runtime defaults live in config.yaml; dataclass defaults fill omitted options.
"""

import json
import logging
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

import coolname
import rich
from omegaconf import DictConfig, OmegaConf
from rich.syntax import Syntax

from aide.journal import Journal, filter_journal
from aide.execution_gate import ExecutionGateConfig
from aide.ucb import UCBConfig

from . import copytree, preproc_data, serialize, tree_export

logger = logging.getLogger("aide")
_DEFAULT_CONFIG_PATH = Path(__file__).with_name("config.yaml")


@dataclass
class StageConfig:
    model: str
    temp: float


@dataclass
class SearchConfig:
    max_debug_depth: int
    debug_prob: float
    num_drafts: int
    # Otherwise choose uniformly among the best good nodes of each draft tree.
    greedy_prob: float = 0.8


@dataclass
class StagePolicyConfig:
    """Generation policy per stage, validated by Agent._resolve_stage_policies.

    single | sequential | parallel | pairwise | fuse | argmax | ucb.
    argmax selects by world-model confidence; ucb adds metric uncertainty.
    """

    draft: str = "single"
    debug: str = "single"
    improve: str = "single"


@dataclass
class AgentConfig:
    steps: int
    time_limit: int
    k_fold_validation: int
    expose_prediction: bool
    data_preview: bool
    convert_system_to_user: bool
    obfuscate: bool

    code: StageConfig
    feedback: StageConfig

    search: SearchConfig

    # Candidate generation and selection.
    # "completed" counts successful outer steps, excluding post-commit failures.
    steps_count_mode: str = "journal"
    stage_policy: StagePolicyConfig = field(default_factory=StagePolicyConfig)
    ucb: UCBConfig = field(default_factory=UCBConfig)
    # sim: prediction; race: real result before deadline, else prediction;
    # real: execute every candidate and reuse the winner's result.
    bon_oracle: str = "sim"
    best_of_n_size: int = 4
    use_validate_on_best: bool = True
    # Extra real check after N steps, excluding predicted timeouts.
    # 0 disables periodic checks.
    real_validation_interval: int = 0
    # Universal success/failure gate on the selected program. When enabled it
    # replaces the legacy stage gate, keeping best/periodic checks and the
    # timeout veto. Null thresholds always execute that outcome except timeouts.
    execution_gate: ExecutionGateConfig = field(default_factory=ExecutionGateConfig)
    # Extra real check for selected candidates with p_max below this threshold.
    # Legacy gate, used only when execution_gate.enabled is False.
    # None disables it; missing/failed confidence also triggers a check.
    real_validation_p_max_threshold: float | None = None
    # Stages the gate covers: "improve" (default), "all", or a comma-separated
    # subset of draft,debug,improve. Ignored while the threshold is None.
    real_validation_p_max_stages: str = "improve"
    # Judge accepted candidates together; fall back to first accepted on error.
    use_best_of_n_listwise: bool = True
    # Listwise mode: True picks the best finite WM metric from usable predictions
    # using eval_lower_is_better; False asks the judge. All-failed pools always
    # ask the least-buggy judge. No finite metric/direction: first usable fallback.
    # Explicit stage selectors (ucb/argmax) retain precedence.
    use_predicted_metric_argmax_pick: bool = False
    # Real-oracle counterpart: finite validation metrics from actual candidate reviews.
    use_observed_metric_argmax_pick: bool = False
    # Advisory analysis for judges/fusion; costs N extra feedback-model calls.
    bon_candidate_review: bool = True
    # Legacy setting accepted when loading old profiles; listwise ignores it.
    use_listwise_p_clean: bool = False
    # Overrides listwise selection with simulator-score argmax. Ranking uses
    # AIDE_PCLEAN_SCORE, but all-failed pools use p_clean unless line-frac
    # salvage is enabled. Disable bon_candidate_review to avoid review calls.
    use_p_clean_argmax_pick: bool = False
    # Legacy probability argmax only: salvage failures by crash-line/code-length.
    use_line_frac_salvage: bool = False
    # Give improve candidates after candidate 0 distinct direction hints.
    improve_plan_axes: bool = False

    # Fusion executes a new program without screening or a fallback candidate.
    # synth uses the whole pool; backbone edits the selected candidate.
    fuse_mode: str = "synth"
    # Family from aide.fusion_prompts.PROMPT_FAMILIES; typed_paco_v13 needs synth
    # and labels WM reports and a WM parent as hypothetical at every stage.
    fuse_prompt_style: str = "typed_paco_v13"
    # Per-stage family overrides (historical families stay immutable); empty
    # now that v13 is the default everywhere.
    fuse_prompt_style_by_stage: dict[str, str] = field(default_factory=dict)
    fuse_style_params_by_stage: dict[str, dict[str, str]] = field(default_factory=dict)
    fuse_include_candidate_plans: bool = True
    # Retry parse/transport failures; execute best-effort code on exhaustion.
    fuse_retries: int = 3
    fuse_max_tokens: int = 8192

    # Pairwise tournament bypasses the viability gate and listwise judge.
    # Figure 7 (arXiv:2608.13940): A/B winners advance, N-1 comparisons.
    pairwise_scheme: str = "bracket"
    # Legacy setting accepted when loading old profiles; pairwise ignores it.
    pairwise_confidence_gate: float = 0.7
    pairwise_include_wm_report: bool = True
    # Choose the feedback or code model for comparisons.
    pairwise_model: str = "feedback"
    pairwise_max_calls: int = 24
    # Allow room for reasoning followed by a boxed A/B answer.
    pairwise_max_tokens: int = 8192

    # Official metric/direction supplied by the harness; None uses model review.
    eval_metric: str | None = None
    eval_lower_is_better: bool | None = None
    # Leakage review for extreme/best scores; submission review for best real nodes (ties too).
    check_data_leakage: bool = True
    # Positive verdicts mark nodes buggy; False makes findings advisory.
    leakage_reject_on_model: bool = True
    # Confirmed prediction-path mismatches reject; False makes findings advisory.
    submission_reject_on_model: bool = True
    submission_review_temperature: float = 0.0
    leakage_max_threshold: float = 0.999
    leakage_min_threshold: float = 1e-6
    leakage_review_best: bool = True
    leakage_timeout_s: float = 120.0
    leakage_max_tokens: int = 3072
    # /score instrumentation uses search time but never informs search/prompts.
    # False skips grading entirely and leaves Node.test_metric unset.
    record_test_metric: bool = True
    # Append small execution/decision/publication metadata; never copy submissions.
    record_execution_cost: bool = False


@dataclass
class ExecConfig:
    timeout: int
    agent_file_name: str
    format_tb_ipython: bool
    # Real/simulator race; delays are measured from the start of execution.
    simulator: bool = False
    simulator_model: str | None = None
    simulator_deadline_s: float = 30.0
    # Included in simulator_deadline_s; fast real runs avoid a simulator call.
    simulator_start_delay_s: float = 3.0
    # Requires simulator=True; bypasses the race (real validation is separate).
    simulator_only: bool = False


@dataclass
class Config:
    data_dir: Path
    desc_file: Path | None

    goal: str | None
    eval: str | None

    log_dir: Path
    log_level: str
    workspace_dir: Path

    preprocess_data: bool
    copy_data: bool

    exp_name: str

    exec: ExecConfig
    agent: AgentConfig


def _load_cfg(path: Path = _DEFAULT_CONFIG_PATH, use_cli_args: bool = True) -> DictConfig:
    """Load raw YAML settings, optionally overridden by CLI arguments."""
    cfg = OmegaConf.load(path)
    if use_cli_args:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_cli())
    return cast(DictConfig, cfg)


def load_cfg(path: Path = _DEFAULT_CONFIG_PATH) -> Config:
    """Load YAML and CLI settings, validate them, and prepare run paths."""
    return prep_cfg(_load_cfg(path))


def prep_cfg(cfg: Config | DictConfig) -> Config:
    """Validate a copy of the settings and resolve experiment directories."""
    # Keep raw paths reusable: preparing one run must not nest later runs
    # inside the first run's log/workspace directories.
    cfg = OmegaConf.merge(cfg)
    if cfg.data_dir is None:
        raise ValueError("`data_dir` must be provided.")

    if cfg.desc_file is None and cfg.goal is None:
        raise ValueError(
            "Provide a task goal (`goal=...`) or a path to its description (`desc_file=...`)."
        )

    cfg.data_dir = Path(cfg.data_dir).resolve()

    if cfg.desc_file is not None:
        cfg.desc_file = Path(cfg.desc_file).resolve()

    top_log_dir = Path(cfg.log_dir).resolve()
    top_workspace_dir = Path(cfg.workspace_dir).resolve()

    cfg.exp_name = cfg.exp_name or coolname.generate_slug(3)
    cfg.log_dir = (top_log_dir / cfg.exp_name).resolve()
    cfg.workspace_dir = (top_workspace_dir / cfg.exp_name).resolve()

    # Merge schema defaults and reject invalid options before creating dirs.
    cfg = OmegaConf.merge(OmegaConf.structured(Config), cfg)
    top_log_dir.mkdir(parents=True, exist_ok=True)
    top_workspace_dir.mkdir(parents=True, exist_ok=True)

    return cast(Config, cfg)


def print_cfg(cfg: Config) -> None:
    rich.print(Syntax(OmegaConf.to_yaml(cfg), "yaml", theme="paraiso-dark"))


def load_task_desc(cfg: Config) -> str | dict[str, str]:
    """Read the description file, or build a description from goal/eval."""
    if cfg.desc_file is not None:
        if cfg.goal is not None or cfg.eval is not None:
            logger.warning("Ignoring goal and eval args because task description file is provided.")
        return Path(cfg.desc_file).read_text()

    if cfg.goal is None:
        raise ValueError(
            "`goal` (and optionally `eval`) must be provided when no "
            "task description file is provided."
        )

    task_desc = {"Task goal": cfg.goal}
    if cfg.eval is not None:
        task_desc["Task evaluation"] = cfg.eval

    return task_desc


def prep_agent_workspace(cfg: Config) -> None:
    """Create workspace directories, populate input, and optionally preprocess."""
    for name in ("input", "working", "submission"):
        (cfg.workspace_dir / name).mkdir(parents=True, exist_ok=True)

    copytree(cfg.data_dir, cfg.workspace_dir / "input", use_symlinks=not cfg.copy_data)
    if cfg.preprocess_data:
        preproc_data(cfg.workspace_dir / "input")


def save_run(cfg: Config, journal: Journal) -> None:
    """Save the journal, configuration, published solution, and readable logs."""
    cfg.log_dir.mkdir(parents=True, exist_ok=True)

    filtered_journal = filter_journal(journal)
    serialize.dump_json(journal, cfg.log_dir / "journal.json")
    serialize.dump_json(filtered_journal, cfg.log_dir / "filtered_journal.json")
    OmegaConf.save(config=cfg, f=cfg.log_dir / "config.yaml")
    if journal.nodes:
        # A visualization failure must not count as a failed search step.
        try:
            tree_export.generate(cfg, journal, cfg.log_dir / "tree_plot.html")
        except Exception:
            logger.warning("Could not export the search tree visualization.", exc_info=True)
    # The exported best code must describe the published CSV. Simulated search
    # leaders remain available in the journal and tree visualization.
    published_solution = cfg.workspace_dir / "best_solution" / "solution.py"
    if published_solution.is_file():
        shutil.copyfile(published_solution, cfg.log_dir / "best_solution.py")
    (cfg.log_dir / "full_log.txt").write_text(
        concat_logs(
            cfg.log_dir / "aide.log",
            cfg.workspace_dir / "best_solution" / "node_id.txt",
            cfg.log_dir / "filtered_journal.json",
        )
    )


def concat_logs(chrono_log: Path, best_node: Path, journal: Path) -> str:
    content = (
        "The following is a concatenation of the log files produced.\n"
        "If a file is missing, it will be indicated.\n\n"
    )

    content += "---First, a chronological, high level log of the AIDE run---\n"
    content += output_file_or_placeholder(chrono_log) + "\n\n"

    content += "---Next, the ID of the best node from the run---\n"
    content += output_file_or_placeholder(best_node) + "\n\n"

    content += "---Finally, the full journal of the run---\n"
    content += output_file_or_placeholder(journal) + "\n\n"

    return content


def output_file_or_placeholder(file: Path) -> str:
    if not file.exists():
        return "File not found."
    content = file.read_text()
    return json.dumps(json.loads(content), indent=4) if file.suffix == ".json" else content
