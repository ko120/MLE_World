"""
The journal is the core datastructure in AIDE that contains:
- the generated code samples
- information how code samples relate to each other (the tree structure)
- code execution results
- evaluation information such as metrics
...
"""

import copy
import math
import time
import uuid
from dataclasses import dataclass, field
from typing import Literal, Optional

from dataclasses_json import DataClassJsonMixin
from .interpreter import ExecutionResult
from .utils.metric import MetricValue
from .utils.response import trim_long_string


@dataclass(eq=False)
class Node(DataClassJsonMixin):
    """A single node in the solution tree. Contains code, execution results, and evaluation information."""

    # ---- code & plan ----
    code: str
    plan: str = field(default=None, kw_only=True)  # type: ignore

    # ---- general attrs ----
    step: int = field(default=None, kw_only=True)  # type: ignore
    id: str = field(default_factory=lambda: uuid.uuid4().hex, kw_only=True)
    ctime: float = field(default_factory=lambda: time.time(), kw_only=True)
    parent: Optional["Node"] = field(default=None, kw_only=True)
    children: set["Node"] = field(default_factory=set, kw_only=True)

    # ---- execution info ----
    _term_out: list[str] = field(default=None, kw_only=True)  # type: ignore
    exec_time: float = field(default=None, kw_only=True)  # type: ignore
    exc_type: str | None = field(default=None, kw_only=True)
    exc_info: dict | None = field(default=None, kw_only=True)
    exc_stack: list[tuple] | None = field(default=None, kw_only=True)
    # Wall-clock kill limit (s) the interpreter enforced for THIS execution —
    # run.py clamps it below the configured exec.timeout near the run deadline,
    # so datagen must read the budget from here, not from config, to label
    # timeouts truthfully. None: pre-2026-08-13 journals and sim-only results.
    timeout_s: float | None = field(default=None, kw_only=True)

    # ---- evaluation ----
    # post-execution result analysis (findings/feedback)
    analysis: str = field(default=None, kw_only=True)  # type: ignore
    metric: MetricValue = field(default=None, kw_only=True)  # type: ignore
    # whether the agent decided that the code is buggy
    # -> always True if exc_type is not None or no valid metric
    is_buggy: bool = field(default=None, kw_only=True)  # type: ignore
    # Effective reviewer/source-check verdict on whether the printed validation
    # score is a trustworthy held-out estimate, persisted alongside is_buggy.
    # Deterministic rejection details are also appended to analysis. None so
    # old journals and judges predating the field are unaffected — matches the
    # `.get(...) is False` gate in review_is_buggy.
    metric_trustworthy: bool | None = field(default=None, kw_only=True)
    # Specialist validation-leakage verdict for this execution. None means the
    # check was skipped; status="unavailable" records a failed checker call.
    leakage_check: dict | None = field(default=None, kw_only=True)
    # Nonempty submission/submission.csv for real runs; an explicit "true"
    # submission_exists prediction for simulated runs. Missing/unknown
    # predictions and legacy placeholders do not establish CSV existence.
    csv_exists: bool | None = field(default=None, kw_only=True)
    # Real format-validation evidence. None means not checked (including sims).
    # "unknown" means infrastructure failed; it must not count as validity.
    submission_validation: Literal["valid", "invalid", "unknown"] | None = field(default=None, kw_only=True)
    submission_validation_message: str | None = field(default=None, kw_only=True)
    submission_sha256: str | None = field(default=None, kw_only=True)
    submission_columns: list[str] = field(default_factory=list, kw_only=True)
    # Retained CSV identity and retry state. A pending source review also keeps
    # a format-valid CSV out of confirmed-best comparisons and publication.
    pending_submission: dict | None = field(default=None, kw_only=True)
    # Harness diagnoses are separate from the model's summary and also appear
    # in analysis so the existing debug and parent-memory prompts receive them.
    failure_reasons: dict[str, list[str]] = field(default_factory=dict, kw_only=True)
    # Legacy repair metadata retained for loading older journals.
    submission_actions: list[dict] = field(default_factory=list, kw_only=True)
    # True iff the execution result absorbed into this node was the real
    # subprocess (not the simulator). Drives validate-on-best (run.py).
    real_won: bool = field(default=False, kw_only=True)
    # Test-set score from the in-container grading server (/score), graded on
    # this node's submission.csv at end of step. Measurement-only: kept a
    # plain float (never MetricValue) so it cannot enter get_best_node()
    # comparisons or generate_summary() prompts. None when nothing gradeable
    # (no/empty CSV, grading error or timeout).
    test_metric: float | None = field(default=None, kw_only=True)
    # World-model predicted val, captured before validate-on-best (run.py)
    # overwrites `metric` with the real re-run value, so a post-run pass can
    # compare prediction vs reality. Plain float like test_metric (measurement-
    # only; never enters get_best_node()). None until a sim node is checked for real.
    sim_metric: float | None = field(default=None, kw_only=True)
    # Selected exception-posterior confidence used by the optional real gate.
    # Retained after real validation; None means unavailable or gate disabled.
    sim_p_max: float | None = field(default=None, kw_only=True)
    # Leading non-clean class and its unconditional posterior mass. A timeout
    # can be the leading exception even when clean is the overall argmax.
    sim_top_exc: str | None = field(default=None, kw_only=True)
    sim_p_top_exc: float | None = field(default=None, kw_only=True)
    # Same-pool selection evidence, captured before real validation changes the
    # metric. Audit-only; parent selection does not depend on this provenance.
    ucb_selection: dict | None = field(default=None, kw_only=True)
    # Why this node's parent was chosen: draft/debug/greedy/tree_best.
    parent_selection: str | None = field(default=None, kw_only=True)

    def __post_init__(self) -> None:
        if self.parent is not None:
            self.parent.children.add(self)

    @property
    def is_ucb_exploration(self) -> bool:
        """A genuine improve-stage bonus choice over the same-pool greedy node."""
        selection = self.ucb_selection or {}
        return (
            selection.get("stage") == "improve"
            and selection.get("rule") == "ucb"
            and selection.get("winner") is not None
            and selection.get("greedy_winner") is not None
            and selection["winner"] != selection["greedy_winner"]
        )

    @property
    def stage_name(self) -> Literal["draft", "debug", "improve"]:
        """
        Return the stage of the node:
        - "stage" if the node is an initial solution draft
        - "debug" if the node is the result of a debugging step
        - "improve" if the node is the result of an improvement step
        """
        if self.parent is None:
            return "draft"
        return "debug" if self.parent.is_buggy else "improve"

    def absorb_exec_result(self, exec_result: ExecutionResult):
        """Absorb the result of executing the code from this node."""
        self._term_out = exec_result.term_out
        self.exec_time = exec_result.exec_time
        self.exc_type = exec_result.exc_type
        self.exc_info = exec_result.exc_info
        self.exc_stack = exec_result.exc_stack
        self.timeout_s = getattr(exec_result, "timeout_s", None)
        # Carry the real/sim provenance so validate-on-best can tell whether
        # this node already has ground-truth execution.
        self.real_won = bool(getattr(exec_result, "real_won", False))

    @property
    def term_out(self) -> str:
        """Get the terminal output of the code execution (after truncating it)."""
        # Coerce defensively: a stray non-str chunk in _term_out (e.g. a
        # nested list) crashed parse_exec_result with TypeError and killed a
        # run (07-07 nomad2018). term_out is prompt/display material — never
        # worth dying for.
        chunks = self._term_out
        if chunks is None:
            chunks = []
        elif isinstance(chunks, str):
            chunks = [chunks]
        return trim_long_string(
            "".join(c if isinstance(c, str) else str(c) for c in chunks)
        )

    def fetch_child_memory(self) -> str:
        """Summarize completed child attempts, including failures, for improvement.

        Rank successful attempts by the parent's metric direction. Keep ties
        chronological because children are stored in a set. Only validation
        evidence belongs here; test_metric and sim_metric are measurement-only.
        """
        children = sorted(
            (n for n in self.children if n.is_buggy is not None),
            key=lambda n: (n.step is None, n.step or 0, n.ctime, n.id),
        )
        maximize = self.metric.maximize if self.metric is not None else None

        def quality(n: "Node") -> tuple[bool, bool, float]:
            metric = n.metric
            comparable = (
                n.is_buggy is False
                and metric is not None
                and metric.value is not None
                and math.isfinite(metric.value)
                and maximize is not None
                and metric.maximize == maximize
            )
            score = (
                metric.value if maximize else -metric.value
            ) if comparable else -math.inf
            return n.is_buggy is False, comparable, score

        children.sort(key=quality, reverse=True)
        summary = []
        for idx, n in enumerate(children, 1):
            parts = [
                f"Attempt #{idx} (node {n.id}):",
                f"Design: {n.plan or '(no plan recorded)'}",
                f"Status: {'Buggy or invalid' if n.is_buggy else 'Non-buggy'}",
                f"Evidence source: {'Real execution' if n.real_won else 'World-model prediction'}",
                f"Results: {trim_long_string(n.analysis or n.exc_type or n.term_out or '(no analysis recorded)')}",
            ]
            if not n.is_buggy and n.metric is not None and n.metric.value is not None:
                direction = (
                    " (higher is better)" if n.metric.maximize
                    else " (lower is better)" if n.metric.maximize is False
                    else ""
                )
                parts.append(f"Validation Metric: {n.metric.value}{direction}")
            if n.real_won and n.exec_time is not None:
                parts.append(f"Execution Time: {n.exec_time:.2f}s")
            summary.append("\n".join(parts))
        return "\n-------------------------------\n".join(summary)

    @property
    def is_leaf(self) -> bool:
        """Check if the node is a leaf node in the solution tree."""
        return not self.children

    def __eq__(self, other):
        return isinstance(other, Node) and self.id == other.id

    def __hash__(self):
        return hash(self.id)

    @property
    def debug_depth(self) -> int:
        """
        Length of the current debug path
        - 0 if the node is not a debug node (parent is not buggy)
        - 1 if the parent is buggy but the skip parent isn't
        - n if there were n consecutive debugging steps
        """
        if self.stage_name != "debug":
            return 0
        return self.parent.debug_depth + 1  # type: ignore


@dataclass
class InteractiveSession(DataClassJsonMixin):
    """
    A collection of nodes for an interaction session
    (when the agent interacts with a Jupyter notebook-like interface).
    """

    nodes: list[Node] = field(default_factory=list)
    completed: bool = False

    def append(self, node: Node) -> None:
        node.step = len(self.nodes)
        self.nodes.append(node)

    def generate_nb_trace(self, include_prompt, comment_headers=True) -> str:
        """Generate a trace of the interactive session in IPython format."""
        trace = []
        header_prefix = "## " if comment_headers else ""
        for n in self.nodes:
            trace.append(f"\n{header_prefix}In [{n.step+1}]:\n")
            trace.append(n.code)
            trace.append(f"\n{header_prefix}Out [{n.step+1}]:\n")
            trace.append(n.term_out)

        if include_prompt and self.nodes:
            trace.append(f"\n{header_prefix}In [{self.nodes[-1].step+2}]:\n")

        return "\n".join(trace).strip()


@dataclass
class Journal(DataClassJsonMixin):
    """A collection of nodes representing the solution tree."""

    nodes: list[Node] = field(default_factory=list)
    # eda: InteractiveSession = field(default_factory=lambda: InteractiveSession())

    def __getitem__(self, idx: int) -> Node:
        return self.nodes[idx]

    def __len__(self) -> int:
        """Return the number of nodes in the journal."""
        return len(self.nodes)

    def append(self, node: Node) -> None:
        """Append a new node to the journal."""
        node.step = len(self.nodes)
        self.nodes.append(node)

    @property
    def draft_nodes(self) -> list[Node]:
        """Return a list of nodes representing intial coding drafts"""
        return [n for n in self.nodes if n.parent is None]

    @property
    def buggy_nodes(self) -> list[Node]:
        """Return a list of nodes that are considered buggy by the agent."""
        return [n for n in self.nodes if n.is_buggy]

    @property
    def good_nodes(self) -> list[Node]:
        """Return a list of nodes that are not considered buggy by the agent."""
        return [n for n in self.nodes if not n.is_buggy]

    def get_metric_history(self) -> list[MetricValue]:
        """Return a list of all metric values in the journal."""
        return [n.metric for n in self.nodes]

    def get_best_node(self, only_good=True, *, real_only=False, validated_only=False) -> None | Node:
        """Return the best solution found so far (node with the highest validation metric)."""
        if only_good:
            nodes = self.good_nodes
            if not nodes:
                return None
        else:
            nodes = self.nodes
        if real_only:
            nodes = [n for n in nodes if n.real_won]
        if validated_only:
            nodes = [
                n for n in nodes
                if n.real_won and n.submission_validation == "valid"
                and not n.pending_submission
                and n.submission_sha256 and n.metric.value is not None
            ]
        if not nodes:
            return None
        # MetricValue.__gt__ asserts both sides share the same `.maximize`
        # direction. The agent occasionally produces inconsistent directions
        # across iterations (e.g. one node minimizes logloss while a later
        # one maximizes accuracy), which used to crash max() with
        # "AssertionError: Cannot compare different metrics" and abort the
        # whole task. Group by direction and pick within the majority group.
        from collections import defaultdict
        by_dir: dict = defaultdict(list)
        for n in nodes:
            d = n.metric.maximize if n.metric.value is not None else None
            by_dir[d].append(n)
        best_dir = max(by_dir, key=lambda d: len(by_dir[d]))
        return max(by_dir[best_dir], key=lambda n: n.metric)

    def get_best_nodes_per_tree(self) -> list[Node]:
        """One best good node per draft tree, in draft order.

        Use committed journal nodes and the same eligibility, metric-direction
        grouping, and first-wins ties as global greedy selection. Recompute on
        each call so real validation can change a tree's best node.
        """
        trees = {node.id: [] for node in self.draft_nodes}
        for node in self.nodes:
            root = node
            while root.parent is not None:
                root = root.parent
            if root.id in trees:
                trees[root.id].append(node)
        return [
            best for nodes in trees.values()
            if (best := Journal(nodes=nodes).get_best_node()) is not None
        ]

    def generate_summary(self, include_code: bool = False) -> str:
        """Generate a summary of the journal for the agent."""
        summary = []
        for n in self.good_nodes:
            summary_part = f"Design: {n.plan}\n"
            if include_code:
                summary_part += f"Code: {n.code}\n"
            summary_part += f"Results: {n.analysis}\n"
            summary_part += f"Validation Metric: {n.metric.value}\n"
            summary.append(summary_part)
        return "\n-------------------------------\n".join(summary)


def get_path_to_node(journal: Journal, node_id: str) -> list[str]:
    path = [node_id]

    node2parent = {n.id: n.parent.id for n in journal.nodes if n.parent is not None}
    while node_id in node2parent:
        parent_id = node2parent[node_id]
        path.append(parent_id)
        node_id = parent_id
    return path[::-1]


def get_longest_path(journal: Journal) -> list[str]:
    longest_path = []
    for node in journal.nodes:
        path = get_path_to_node(journal, node.id)
        if len(path) > len(longest_path):
            longest_path = path
    return longest_path


def filter_on_path(journal: Journal, path: list[str]) -> Journal:
    journal_copy = copy.deepcopy(journal)
    journal_copy.nodes = [n for n in journal_copy.nodes if n.id in path]
    # further filter nodes, setting their _term_out and exc_stack to "<OMITTED>"
    for n in journal_copy.nodes:
        n._term_out = "<OMITTED>"
        n.exc_stack = "<OMITTED>"

    return journal_copy


def filter_for_best_path(journal: Journal, best_node: str) -> Journal:
    path_to_best = get_path_to_node(journal, best_node)
    filtered_journal = filter_on_path(journal, path_to_best)
    return filtered_journal


def filter_for_longest_path(journal: Journal) -> Journal:
    longest_path = get_longest_path(journal)
    filtered_journal = filter_on_path(journal, longest_path)
    return filtered_journal


def filter_journal(journal: Journal) -> Journal:
    best_node = journal.get_best_node(only_good=True)

    if best_node is not None:
        filtered_journal = filter_for_best_path(journal, best_node.id)
    else:
        filtered_journal = filter_for_longest_path(journal)

    return filtered_journal
