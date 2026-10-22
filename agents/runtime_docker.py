"""Docker execution and artifact extraction for one agent trial."""

import codecs
import logging
import time
from pathlib import Path

import docker
from agents.registry import Agent
from agents.runtime_common import CONSTANTS, _agent_cmd, _agent_env
from docker.models.containers import Container
from environment.utils import (
    create_competition_container,
    extract_from_container,
    extract_from_container_sysbox,
)
from mlebench.registry import Competition
from mlebench.utils import purple


def save_output(container: Container, save_dir: Path, container_config: dict) -> Path:
    """Extract the submission, logs and code directories from the container."""
    if container_config.get("runtime") == "sysbox-runc":
        extraction_fn = extract_from_container_sysbox
    else:
        extraction_fn = extract_from_container
    for dir_type in ("SUBMISSION_DIR", "LOGS_DIR", "CODE_DIR"):
        extraction_fn(container, CONSTANTS[dir_type], save_dir)
    return save_dir


def execute_agent(container: Container, agent: Agent, logger: logging.Logger) -> None:
    """Run start.sh inside the container and stream its output to the log."""
    logger.info("Running agent...")
    exit_code, stream = container.exec_run(_agent_cmd(agent), stream=True, user="nonroot")
    decoder = codecs.getincrementaldecoder("utf-8")("replace")
    for chunk in stream:
        text = decoder.decode(chunk) if chunk else ""
        if text:
            logger.info(f"[Container] {text.rstrip()}")
    tail = decoder.decode(b"", final=True)
    if tail:
        logger.info(f"[Container] {tail.rstrip()}")
    if isinstance(exit_code, int) and exit_code != 0:
        logger.warning(f"Agent process exited with code {exit_code}")


def clean_up(container: Container, logger: logging.Logger, retain: bool = False) -> bool:
    """Stop (and unless retain, remove) the container. True on success."""
    logger.info(f"Cleaning up container: {container.name}")
    try:
        container.stop()
        if not retain:
            container.remove()
        logger.info(f"Container {container.name} stopped and removed.")
        return True
    except Exception as e:
        logger.error(
            f"Error cleaning up: {e}. You may wish to manually check the status "
            f"of the {container.name} container."
        )
        return False


def run_in_container(
    client: docker.DockerClient,
    competition: Competition,
    agent: Agent,
    image: str,
    container_config: dict,
    retain_container: bool,
    run_dir: Path,
    logger: logging.Logger,
    gpus: list,
) -> Path:
    """Create the competition container, run the agent, extract outputs to
    run_dir, and always clean the container up. Returns run_dir."""
    volumes_config = {
        competition.public_dir.resolve().as_posix(): {"bind": "/home/data", "mode": "ro"},
        competition.private_dir.resolve().as_posix(): {
            "bind": f"/private/data/{competition.id}/prepared/private/",
            "mode": "ro",
        },
    }
    container = create_competition_container(
        client=client,
        competition=competition,
        container_config=container_config,
        volumes_config=volumes_config,
        env_vars=_agent_env(competition, agent, logger),
        container_image=image,
        privileged=agent.privileged,
        gpus=gpus,
    )

    logger.info(purple(f"Run started: {run_dir}"))
    try:
        time_start = time.monotonic()
        container.start()
        execute_agent(container, agent, logger)
        save_output(container, run_dir, container_config)
        logger.info(f"Run completed in {time.monotonic() - time_start:.2f} seconds.")
        return run_dir
    finally:
        clean_up(container, logger, retain_container)
