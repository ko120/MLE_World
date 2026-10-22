"""Run the paper's 12-hour agent policies from this checkout with Enroot.

The existing run_agent.py harness owns execution, grading isolation, runtime
seeding and per-run source snapshots. This entrypoint resolves the policy YAMLs
and supplies their tasks, seeds, endpoints and budgets without cluster-specific
experiment controllers or frozen experiment directories.
"""

import argparse
import copy
import json
import math
import os
import platform
import shlex
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import yaml

BENCH = Path(__file__).resolve().parent
DEFAULT_KWARGS = {
    "agent.steps_count_mode": "journal",
    "agent.use_observed_metric_argmax_pick": False,
    "agent.use_predicted_metric_argmax_pick": False,
    "agent.execution_gate.enabled": False,
    "agent.execution_gate.mode": "confidence",
    "agent.execution_gate.random_check_probability": "null",
    "agent.execution_gate.success_threshold": "null",
    "agent.execution_gate.failure_threshold": "null",
    "agent.real_validation_interval": 0,
}


def resolve_profile(path, ancestors=()):
    path = Path(path).resolve()
    if path in ancestors:
        raise ValueError(f"Circular profile inheritance: {path}")
    raw = yaml.safe_load(path.read_text())
    if raw.get("schema_version") != 1:
        raise ValueError(f"Unsupported profile schema: {path}")
    if "extends" not in raw:
        return copy.deepcopy(raw["profile"])
    profile = resolve_profile(path.parent / raw["extends"], (*ancestors, path))
    for section, keys in raw.get("remove", {}).items():
        for key in keys:
            profile[section].pop(key, None)
    for section, updates in raw.get("overrides", {}).items():
        profile.setdefault(section, {}).update(updates)
    return profile


def compile_suite(
    suite_path,
    *,
    regime="all",
    arms=None,
    coder_url="http://localhost:8000/v1",
    feedback_url="http://localhost:8002/v1",
    world_url="http://localhost:8001/v1",
    feedback_model=None,
):
    suite_path = Path(suite_path).resolve()
    suite = yaml.safe_load(suite_path.read_text())
    if suite.get("schema_version") != 1:
        raise ValueError("Unsupported experiment suite schema")
    if (suite_path.parent / suite["source_base"]).resolve() != BENCH:
        raise ValueError("The suite must select this checkout with source_base: ..")
    seeds = suite["seeds"]
    if (
        not seeds
        or any(type(seed) is not int or not 0 <= seed < 2**32 for seed in seeds)
        or seeds != list(range(seeds[0], seeds[0] + len(seeds)))
    ):
        raise ValueError("Suite seeds must be consecutive integers in [0, 2**32)")
    if not suite["tasks"] or len(set(suite["tasks"])) != len(suite["tasks"]):
        raise ValueError("Suite tasks must be nonempty and unique")
    for task in suite["tasks"]:
        if not (BENCH / "mlebench/competitions" / task / "config.yaml").is_file():
            raise ValueError(f"Competition is absent from this checkout: {task}")
    definitions = {arm["id"]: arm for arm in suite["arms"]}
    if len(definitions) != len(suite["arms"]):
        raise ValueError("Suite arm IDs must be unique")
    requested = set(arms or definitions)
    unknown = requested - definitions.keys()
    if unknown:
        raise ValueError(f"Unknown arm(s): {', '.join(sorted(unknown))}")
    if regime not in ("all", "gated", "ungated"):
        raise ValueError(f"Unknown regime: {regime}")
    if arms and regime != "all":
        conflicting = [arm for arm in arms if definitions[arm]["regime"] != regime]
        if conflicting:
            raise ValueError(f"Arms outside the {regime} regime: {', '.join(conflicting)}")
    profiles = {}
    for arm, definition in definitions.items():
        if arm not in requested or (regime != "all" and definition["regime"] != regime):
            continue
        profile = resolve_profile(suite_path.parent / definition["configuration"])
        kwargs = profile["kwargs"] = {**DEFAULT_KWARGS, **profile["kwargs"]}
        env = profile["env_vars"]
        if feedback_model:
            previous_feedback = kwargs["agent.feedback.model"]
            kwargs["agent.feedback.model"] = feedback_model
            sampling = json.loads(env.get("AIDE_MODEL_SAMPLING", "{}"))
            feedback_sampling = sampling.pop(previous_feedback, None)
            if feedback_model == kwargs["agent.code.model"]:
                if feedback_url.rstrip("/") != coder_url.rstrip("/"):
                    raise ValueError("A shared code/feedback model requires matching server URLs")
            elif feedback_sampling is not None:
                sampling[feedback_model] = feedback_sampling
            env["AIDE_MODEL_SAMPLING"] = json.dumps(sampling, separators=(",", ":"))
        expected = {
            "agent.time_limit": suite["trial_seconds"],
            "exec.timeout": suite["execution_timeout_seconds"],
            "agent.execution_gate.enabled": definition["regime"] == "gated",
        }
        for key, value in expected.items():
            if kwargs.get(key) != value:
                raise ValueError(f"{arm}: {key} must be {value!r}")
        for key, value in {
            "TIME_LIMIT_SECS": suite["trial_seconds"],
            "AIDE_SHUTDOWN_RESERVE_S": suite["shutdown_reserve_seconds"],
            "STEP_LIMIT": kwargs["agent.steps"],
        }.items():
            if int(env[key]) != value:
                raise ValueError(f"{arm}: {key} must be {value}")
        env["OPENAI_API_KEY"] = (
            "${{ secrets.OPENAI_API_KEY }}" if os.environ.get("OPENAI_API_KEY") else "EMPTY"
        )
        env["OPENAI_BASE_URL"] = coder_url
        env["AIDE_MODEL_BASE_URLS"] = json.dumps(
            {kwargs["agent.feedback.model"]: feedback_url}, separators=(",", ":")
        )
        if kwargs["exec.simulator"]:
            env["AIDE_SIMULATOR_BASE_URL"] = world_url
            if os.environ.get("AIDE_SIMULATOR_API_KEY"):
                env["AIDE_SIMULATOR_API_KEY"] = "${{ secrets.AIDE_SIMULATOR_API_KEY }}"
        profile["start"] = str(BENCH / "agents/aide/start.sh")
        profile["dockerfile"] = str(BENCH / "agents/aide/Dockerfile")
        profiles[arm] = profile
    if not profiles:
        raise ValueError("No arms selected")
    return suite, profiles


def positive_int(value):
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return result


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, default=BENCH / "configs/paper.yaml")
    parser.add_argument("--arm", nargs="+", help="Select arm IDs from the suite")
    parser.add_argument("--regime", choices=["all", "gated", "ungated"], default="all")
    parser.add_argument("--coder-url", default="http://localhost:8000/v1")
    parser.add_argument("--feedback-url", default="http://localhost:8002/v1")
    parser.add_argument("--feedback-model", help="Override the feedback API model name")
    parser.add_argument("--world-url", default="http://localhost:8001/v1")
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run and grade one task and seed with four search steps and a 20-minute trial limit",
    )
    parser.add_argument("--competition-set", type=Path, help="Optional subset/shard of suite tasks")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache")))
        / "mle-bench/data",
    )
    image_name = "aide.x86_64.sqsh" if platform.machine() == "x86_64" else "aide.sqsh"
    parser.add_argument("--enroot-image", type=Path, default=Path.home() / image_name)
    parser.add_argument("--n-workers", type=positive_int, default=1)
    parser.add_argument("--gpu-device", nargs="+", default=["0"])
    parser.add_argument(
        "--container-config",
        type=Path,
        default=BENCH / "environment/config/container_configs/default.json",
    )
    parser.add_argument("--output-dir", type=Path, help="New directory for configs and run groups")
    parser.add_argument(
        "--grade",
        action="store_true",
        help="Grade final submissions with each run group's frozen graders",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--dry-run", action="store_true", help="Print commands without writing or launching"
    )
    mode.add_argument(
        "--prepare-only", action="store_true", help="Write configs and commands without launching"
    )
    return parser


def prepare_plan(args):
    arms = args.arm
    if args.smoke_test and not arms:
        arms = []
        if args.regime in ("all", "ungated"):
            arms.append("bon_metric_argmax")
        if args.regime in ("all", "gated"):
            arms.append("bon_metric_argmax_s50_f56")
    suite, profiles = compile_suite(
        args.suite,
        regime=args.regime,
        arms=arms,
        coder_url=args.coder_url,
        feedback_url=args.feedback_url,
        world_url=args.world_url,
        feedback_model=args.feedback_model,
    )
    tasks = suite["tasks"]
    if args.competition_set:
        tasks = [
            line.strip() for line in args.competition_set.read_text().splitlines() if line.strip()
        ]
        if not tasks or len(tasks) != len(set(tasks)) or not set(tasks) <= set(suite["tasks"]):
            raise ValueError("The competition set must be a nonempty, unique subset of suite tasks")
    if args.smoke_test:
        if args.competition_set and len(tasks) != 1:
            raise ValueError("A smoke test requires exactly one competition")
        tasks = ["leaf-classification" if "leaf-classification" in tasks else tasks[0]]
        suite = {**suite, "seeds": suite["seeds"][:1], "trial_seconds": 1200}
        for profile in profiles.values():
            profile["kwargs"].update(
                {
                    "agent.time_limit": 1200,
                    "agent.steps": 4,
                    "agent.search.num_drafts": 1,
                    "agent.best_of_n_size": 2,
                    "exec.timeout": 300,
                }
            )
            env = profile["env_vars"]
            env.update(
                {
                    "TIME_LIMIT_SECS": "1200",
                    "STEP_LIMIT": "4",
                    "AIDE_SHUTDOWN_RESERVE_S": "120",
                    "AIDE_DRAFT_MAX_TOKENS": "8192",
                }
            )
            sampling = json.loads(env["AIDE_SAMPLING"])
            sampling["max_tokens"] = 8192
            env["AIDE_SAMPLING"] = json.dumps(sampling, separators=(",", ":"))
            model_sampling = json.loads(env.get("AIDE_MODEL_SAMPLING", "{}"))
            for options in model_sampling.values():
                options["max_tokens"] = 8192
            env["AIDE_MODEL_SAMPLING"] = json.dumps(model_sampling, separators=(",", ":"))
            if "AIDE_SIMULATOR_SAMPLING" in env:
                simulator_sampling = json.loads(env["AIDE_SIMULATOR_SAMPLING"])
                simulator_sampling["max_tokens"] = 2048
                env["AIDE_SIMULATOR_SAMPLING"] = json.dumps(
                    simulator_sampling, separators=(",", ":")
                )
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = (args.output_dir or BENCH / "runs" / f"paper-{stamp}").resolve()
    commands = []
    for arm in profiles:
        command = [
            sys.executable,
            str(BENCH / "run_agent.py"),
            "--agent-id",
            arm,
            "--agent-name",
            "aide",
            "--agent-dir",
            str(BENCH / "agents"),
            "--agent-config",
            str(output / "agent-config.yaml"),
            "--competition-set",
            str(output / "tasks.txt"),
            "--runtime",
            "enroot",
            "--enroot-image",
            str(args.enroot_image.resolve()),
            "--data-dir",
            str(args.data_dir.resolve()),
            "--container-config",
            str(args.container_config.resolve()),
            "--n-seeds",
            str(len(suite["seeds"])),
            "--seed-offset",
            str(suite["seeds"][0]),
            "--n-workers",
            str(args.n_workers),
            "--gpu-device",
            *args.gpu_device,
        ]
        commands.append(
            {
                "arm": arm,
                "command": command,
                "environment": {
                    "RUN_GROUP_SUFFIX": (
                        f"smoke_{arm}_20m" if args.smoke_test else f"paper_{arm}_12h"
                    )
                },
            }
        )
    environment = {
        "MLEBENCH_RUNS_DIR": str(output / "runs"),
        "PYTHONPATH": os.pathsep.join(
            filter(
                None, [str(BENCH), str(BENCH / "agents/aide/src"), os.environ.get("PYTHONPATH", "")]
            )
        ),
        "AIDE_CPU_AFFINITY": os.environ.get("AIDE_CPU_AFFINITY", "1"),
        "ENROOT_DATA_PATH": os.environ.get("ENROOT_DATA_PATH")
        or str(Path(tempfile.gettempdir()) / f"mlebench-enroot-{os.getuid()}-{stamp}"),
    }
    if "AIDE_CPU_PER_TASK" in os.environ or "AIDE_CPU_OVERCOMMIT" not in os.environ:
        environment["AIDE_CPU_PER_TASK"] = os.environ.get("AIDE_CPU_PER_TASK", "10")
    plan = {
        "mode": "smoke" if args.smoke_test else "paper",
        "suite": str(args.suite.resolve()),
        "source_root": str(BENCH),
        "output_dir": str(output),
        "tasks": tasks,
        "seeds": suite["seeds"],
        "trial_seconds": suite["trial_seconds"],
        "trials_per_arm": len(tasks) * len(suite["seeds"]),
        "total_trials": len(profiles) * len(tasks) * len(suite["seeds"]),
        "grade_after_arm": args.grade or args.smoke_test,
        "environment": environment,
        "commands": commands,
    }
    return plan, profiles


def grade_run_group(metadata_path, *, tasks, seeds, data_dir):
    """Grade exactly the completed task/seed grid, using its launch-time source."""
    from agents.snapshots import _hash_tree

    metadata_path = Path(metadata_path).resolve()
    group = metadata_path.parent
    metadata = json.loads(metadata_path.read_text())
    runs = metadata["runs"]
    identities = [(run["competition"], run["seed"]) for run in runs.values()]
    expected = {(task, seed) for task in tasks for seed in seeds}
    if (
        metadata.get("status") != "complete"
        or metadata.get("expected_runs") != len(expected)
        or len(identities) != len(expected)
        or set(identities) != expected
        or any(run.get("success") is not True for run in runs.values())
    ):
        raise ValueError(f"Incomplete or mismatched task/seed grid: {metadata_path}")
    snapshot = group / "agent_snapshot"
    provenance = json.loads((snapshot / "PROVENANCE.json").read_text())
    if _hash_tree(snapshot / "mlebench") != provenance["mlebench_sha256"]:
        raise ValueError(f"Frozen grader hash mismatch: {snapshot}")
    submission = group / "submission.jsonl"
    commands = [
        [
            sys.executable,
            str(BENCH / "experiments/make_submission.py"),
            "--metadata",
            str(metadata_path),
            "--output",
            str(submission),
        ],
        [
            sys.executable,
            "-m",
            "mlebench.cli",
            "grade",
            "--submission",
            str(submission),
            "--data-dir",
            str(Path(data_dir).resolve()),
            "--output-dir",
            str(group),
        ],
    ]
    # Changing cwd as well as PYTHONPATH prevents the live checkout from taking
    # precedence over the frozen package when invoking `python -m`.
    environment = {
        **os.environ,
        "PYTHONPATH": str(snapshot),
        "MLEBENCH_CACHE_DIR": str(group / "grading_cache"),
    }
    for command in commands:
        print(shlex.join(command), flush=True)
        result = subprocess.run(command, cwd=group, env=environment)
        if result.returncode:
            return result.returncode
    return 0


def verify_smoke_grade(group, expected_trials):
    """A completed process alone is insufficient for a passing smoke test."""
    reports = list(Path(group).glob("*_grading_report.json"))
    if len(reports) != 1:
        raise ValueError(f"Expected one grading report in {group}; found {len(reports)}")
    report = json.loads(reports[0].read_text())
    rows = report.get("competition_reports", [])
    if (
        report.get("total_runs") != expected_trials
        or report.get("total_valid_submissions") != expected_trials
        or len(rows) != expected_trials
        or any(
            row.get("valid_submission") is not True
            or type(row.get("score")) not in (int, float)
            or not math.isfinite(row["score"])
            for row in rows
        )
    ):
        raise ValueError(
            f"Smoke test requires a valid, finite graded score for every trial: {reports[0]}"
        )


def main(argv=None):
    parser = make_parser()
    args = parser.parse_args(argv)
    args.grade = args.grade or args.smoke_test
    try:
        plan, profiles = prepare_plan(args)
        if not args.dry_run and not args.prepare_only and not args.enroot_image.is_file():
            raise ValueError(f"Enroot image does not exist: {args.enroot_image}")
        if not args.dry_run:
            output = Path(plan["output_dir"])
            output.mkdir(parents=True, exist_ok=False)
            (output / "agent-config.yaml").write_text(yaml.safe_dump(profiles, sort_keys=False))
            (output / "tasks.txt").write_text("\n".join(plan["tasks"]) + "\n")
            (output / "launch.json").write_text(json.dumps(plan, indent=2) + "\n")
    except (OSError, ValueError, KeyError, yaml.YAMLError) as exc:
        parser.error(str(exc))
    print(f"Source: {plan['source_root']}")
    print(
        f"{len(profiles)} arms; {len(plan['tasks'])} tasks; seeds {plan['seeds']}; "
        f"{plan['trial_seconds']} seconds per trial; {plan['total_trials']} trials"
    )
    print("Arms execute sequentially; --n-workers controls concurrent trials within each arm.")
    if args.grade:
        print("Final submissions will be graded after each arm with its frozen source.")
    for item in plan["commands"]:
        env_options = [
            f"{key}={value}"
            for key, value in {**plan["environment"], **item["environment"]}.items()
        ]
        print(shlex.join(["env", *env_options, *item["command"]]), flush=True)
    if args.dry_run or args.prepare_only:
        return 0
    smoke_failures = []
    for item in plan["commands"]:
        runs = Path(plan["environment"]["MLEBENCH_RUNS_DIR"])
        before = set(runs.glob("*/metadata.json")) if args.grade else set()
        result = subprocess.run(
            item["command"],
            cwd=BENCH,
            env={**os.environ, **plan["environment"], **item["environment"]},
        )
        if result.returncode:
            return result.returncode
        if args.grade:
            try:
                created = set(runs.glob("*/metadata.json")) - before
                if len(created) != 1:
                    raise ValueError(
                        f"Expected one completed run group for {item['arm']}; found {len(created)}"
                    )
                metadata_path = created.pop()
                code = grade_run_group(
                    metadata_path, tasks=plan["tasks"], seeds=plan["seeds"], data_dir=args.data_dir
                )
            except (OSError, ValueError, KeyError) as exc:
                parser.error(str(exc))
            if code:
                return code
            if args.smoke_test:
                try:
                    verify_smoke_grade(metadata_path.parent, plan["trials_per_arm"])
                except (OSError, ValueError) as exc:
                    smoke_failures.append(f"{item['arm']}: {exc}")
    if args.grade:
        command = [
            sys.executable,
            "-m",
            "mlebench.costs",
            plan["environment"]["MLEBENCH_RUNS_DIR"],
            "--out-dir",
            str(Path(plan["output_dir"]) / "costs"),
        ]
        print(shlex.join(command), flush=True)
        result = subprocess.run(command, cwd=BENCH, env={**os.environ, **plan["environment"]})
        if result.returncode:
            return result.returncode
    if smoke_failures:
        print("Smoke test failed:\n" + "\n".join(smoke_failures), file=sys.stderr)
        return 1
    if args.smoke_test:
        print(f"Smoke test passed: {plan['total_trials']} valid graded submissions.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
