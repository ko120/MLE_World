"""Public AIDE API.

Heavy runtime dependencies are imported lazily.  This keeps pure support
modules (for example :mod:`aide.fusion`) usable by offline data and prompt
tooling on hosts that do not have every execution/backend dependency.
"""

from dataclasses import dataclass
from importlib import import_module
from typing import Any


_LAZY_EXPORTS = {
    "compile_prompt_to_md": (".backend", "compile_prompt_to_md"),
    "Agent": (".agent", "Agent"),
    "Interpreter": (".interpreter", "Interpreter"),
    "Journal": (".journal", "Journal"),
    "Node": (".journal", "Node"),
}


def __getattr__(name: str) -> Any:
    try:
        module_name, attribute = _LAZY_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(module_name, __name__), attribute)
    globals()[name] = value
    return value


@dataclass
class Solution:
    code: str
    valid_metric: float


class Experiment:

    def __init__(self, data_dir: str, goal: str, eval: str | None = None):
        """Initialize a new experiment run.

        Args:
            data_dir (str): Path to the directory containing the data files.
            goal (str): Description of the goal of the task.
            eval (str | None, optional): Optional description of the preferred way for the agent to evaluate its solutions.
        """

        from omegaconf import OmegaConf
        from rich.status import Status

        from .agent import Agent
        from .interpreter import Interpreter
        from .journal import Journal
        from .utils.config import (
            _load_cfg,
            load_task_desc,
            prep_agent_workspace,
            prep_cfg,
            save_run,
        )

        _cfg = _load_cfg(use_cli_args=False)
        _cfg.data_dir = data_dir
        _cfg.goal = goal
        _cfg.eval = eval
        self.cfg = prep_cfg(_cfg)

        self.task_desc = load_task_desc(self.cfg)

        with Status("Preparing agent workspace (copying and extracting files) ..."):
            prep_agent_workspace(self.cfg)

        self.journal = Journal()
        self.agent = Agent(
            task_desc=self.task_desc,
            cfg=self.cfg,
            journal=self.journal,
        )
        self.interpreter = Interpreter(
            self.cfg.workspace_dir, **OmegaConf.to_container(self.cfg.exec)  # type: ignore
        )
        self._save_run = save_run

    def run(self, steps: int) -> Solution:
        for _i in range(steps):
            self.agent.step(exec_callback=self.interpreter.run)
            self._save_run(self.cfg, self.journal)
        self.interpreter.cleanup_session()

        best_node = self.journal.get_best_node()
        return Solution(code=best_node.code, valid_metric=best_node.metric.value)
