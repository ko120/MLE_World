"""An aborted search must salvage its journal and report failure to the harness."""

import logging
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from omegaconf import OmegaConf

from aide import run as run_mod
from aide.journal import Node


@pytest.mark.parametrize("recover", [False, True])
def test_step_failure_exit_status_and_cleanup(tmp_path, monkeypatch, recover):
    cfg = SimpleNamespace(
        agent=SimpleNamespace(time_limit=21600, steps=2),
        exec=OmegaConf.create({"simulator": False, "timeout": 10}),
        log_dir=tmp_path / "logs", workspace_dir=tmp_path / "workspace",
        exp_name="step-guard-regression", log_level="ERROR",
    )
    cfg.workspace_dir.mkdir()
    interpreter, agent, save = Mock(), Mock(), Mock()
    limit = run_mod._MAX_CONSECUTIVE_STEP_FAILURES
    attempts = 0

    def create_agent(**kwargs):
        agent.journal = kwargs["journal"]
        def step(**_):
            nonlocal attempts
            attempts += 1
            if not recover or attempts % limit:
                raise RuntimeError("simulated context overflow")
            kwargs["journal"].append(Node(code="pass", is_buggy=True))
        agent.step.side_effect = step
        return agent

    def exit_run(code):
        raise SystemExit(code)

    monkeypatch.setattr(run_mod, "load_cfg", lambda: cfg)
    monkeypatch.setattr(run_mod, "load_task_desc", lambda _: "test task")
    monkeypatch.setattr(run_mod, "prep_agent_workspace", lambda _: None)
    monkeypatch.setattr(run_mod, "Interpreter", Mock(return_value=interpreter))
    monkeypatch.setattr(run_mod, "Agent", create_agent)
    monkeypatch.setattr(run_mod, "save_run", save)
    monkeypatch.setattr(run_mod, "journal_to_string_tree", lambda _: "tree")
    monkeypatch.setattr(run_mod.os, "_exit", exit_run)
    monkeypatch.setattr(run_mod.logging, "shutdown", lambda: None)
    logger = logging.getLogger("aide")
    existing_handlers = set(logger.handlers)
    try:
        with pytest.raises(SystemExit) as stopped:
            run_mod.run()
        assert stopped.value.code == (0 if recover else 1)
        assert attempts == limit * (2 if recover else 1)
        interpreter.cleanup_session.assert_called_once_with()
        assert save.call_count == (3 if recover else 1)
        assert len(save.call_args.args[1]) == (2 if recover else 0)
    finally:
        for handler in set(logger.handlers) - existing_handlers:
            logger.removeHandler(handler)
            handler.close()
