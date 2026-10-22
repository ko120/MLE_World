"""Container paths, lifecycle limits, and the shared agent launch contract."""

import logging

from agents.registry import Agent
from mlebench.registry import Competition

CONSTANTS = {
    "SUBMISSION_DIR": "/home/submission",
    "LOGS_DIR": "/home/logs",
    "CODE_DIR": "/home/code",
    "AGENT_DIR": "/home/agent",
}

# Wall-clock allowed past the agent's TIME_LIMIT_SECS before the harness kills
# the container's process group: covers start.sh's preamble, `timeout -k`'s own
# SIGKILL escalation and teardown. Without a bound one wedged container blocks
# its worker until SLURM walltime and prevents the task queue from finishing.
ENROOT_EXIT_GRACE_SECS = 900
ENROOT_PROVISION_TIMEOUT_SECS = 1800
ENROOT_REMOVE_WARN_SECS = 120
ENROOT_DRIVER_TIMEOUT_SECS = 120
ENROOT_COMMAND_STOP_GRACE_SECS = 5
GRADING_SCRIPTS = ("grading_server.py", "grading_server_wrapper.py")
ENROOT_GRADING_PROTOCOL = 3


def _agent_cmd(agent: Agent) -> list[str]:
    """start.sh plus the profile's kwargs in the agent's argument style."""
    cmd = ["bash", f"{CONSTANTS['AGENT_DIR']}/start.sh"]
    if agent.kwargs_type == "argparse":
        for key, value in agent.kwargs.items():
            cmd += [f"--{key}", str(value)]
    elif agent.kwargs_type == "omegaconf":
        cmd += [f"{key}={value}" for key, value in agent.kwargs.items()]
    return cmd


def _agent_env(competition: Competition, agent: Agent, logger: logging.Logger) -> dict:
    """Env every container gets: competition id, pinned eval metric, profile env."""
    return {
        **agent.env_vars,
        "COMPETITION_ID": competition.id,
        **_eval_metric_env(competition, logger),
    }


def _eval_metric_env(competition: Competition, logger: logging.Logger) -> dict:
    """Use the verified metric declared in this competition's grader config.

    Resolving metadata never reads submissions, invokes graders, or guesses a
    direction. Grader rejects missing or invalid metadata when the config loads.
    """
    return {
        "METRIC_NAME": competition.grader.name,
        "METRIC_LOWER_IS_BETTER": str(competition.grader.lower_is_better).lower(),
    }
