"""Run from mle-bench with: python -m unittest discover -s tests -v."""

import asyncio
import io
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import run_agent
from agents import runtime_common
from agents import runtime_enroot as container_run
from agents import snapshots


class CpuSliceTests(unittest.TestCase):
    def slices(self, available, workers, **env):
        with patch.dict(os.environ, {"AIDE_CPU_AFFINITY": "1", **env}, clear=True):
            with patch.object(os, "sched_getaffinity", return_value=set(available)):
                result = run_agent._cpu_slices(workers, logging.getLogger(__name__))
        return [list(map(int, part.split(","))) for part in result] if result else None

    def test_exact_partition_uses_sparse_allowed_cpus_once(self):
        available = [2, 4, 8, 9, 17, 20, 24, 31, 40, 63]
        slices = self.slices(available, 3)
        self.assertEqual([len(part) for part in slices], [4, 3, 3])
        self.assertEqual([cpu for part in slices for cpu in part], available)

    def test_fixed_width_windows_stay_inside_the_allocation(self):
        for ncpu, workers, width in [(75, 63, 10), (144, 54, 10), (40, 63, 10), (4, 8, 10)]:
            with self.subTest(ncpu=ncpu, workers=workers):
                available = list(range(3, 3 + 2 * ncpu, 2))
                slices = self.slices(available, workers, AIDE_CPU_PER_TASK=str(width))
                self.assertEqual(len(slices), workers)
                for part in slices:
                    self.assertEqual(len(set(part)), min(width, ncpu))
                    self.assertTrue(set(part) <= set(available))
                self.assertEqual(set().union(*map(set, slices)), set(available))

    def test_window_spacing_does_not_round_an_integer_boundary_down(self):
        slices = self.slices(range(30), 22, AIDE_CPU_PER_TASK="2")
        self.assertEqual(slices[11], [15, 16])
        self.assertEqual(slices[-1], [28, 29])

    def test_overcommit_and_explicit_width_precedence(self):
        slices = self.slices(range(16), 8, AIDE_CPU_OVERCOMMIT="2")
        self.assertTrue(all(len(part) == 4 for part in slices))
        self.assertEqual(slices[-1], [0, 1, 14, 15])
        slices = self.slices(range(16), 8, AIDE_CPU_OVERCOMMIT="2", AIDE_CPU_PER_TASK="3")
        self.assertTrue(all(len(part) == 3 for part in slices))

    def test_affinity_disabled_and_insufficient_exact_partition(self):
        self.assertIsNone(self.slices(range(8), 2, AIDE_CPU_AFFINITY="0"))
        self.assertIsNone(self.slices(range(2), 3))

    @unittest.skipUnless(
        shutil.which("taskset") and hasattr(os, "sched_getaffinity"), "Linux affinity"
    )
    def test_taskset_affinity_reaches_child_and_grandchild(self):
        available = sorted(os.sched_getaffinity(0))
        expected = sorted({available[0], available[-1]})
        slices = self.slices(expected, 1, AIDE_CPU_PER_TASK="2")
        cpus = ",".join(map(str, slices[0]))
        child = "import os,json; print(json.dumps(sorted(os.sched_getaffinity(0))))"
        program = (
            "import os,json,subprocess,sys; "
            "print(json.dumps([sorted(os.sched_getaffinity(0)), "
            f"json.loads(subprocess.check_output([sys.executable, '-c', {child!r}]))]))"
        )
        result = subprocess.run(
            ["taskset", "-c", cpus, sys.executable, "-c", program],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
            start_new_session=True,
        )
        self.assertEqual(json.loads(result.stdout), [expected, expected])


class EnrootRunTests(unittest.TestCase):
    def setUp(self):
        self.temp = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.run_dir = self.temp / "group" / "run"
        self.run_dir.mkdir(parents=True)
        self.snapshot(self.run_dir.parent)
        self.agent = SimpleNamespace(
            name="aide",
            id="test-profile",
            kwargs_type="omegaconf",
            kwargs={},
            env_vars={"TIME_LIMIT_SECS": "60"},
            privileged=False,
            start=self.temp / "start.sh",
        )
        self.competition = self.competition_for("test")
        self.proc = Mock(stdout=io.StringIO(""), returncode=0)
        self.popen = self.enterContext(
            patch.object(container_run.subprocess, "Popen", return_value=self.proc)
        )
        self.remove = self.enterContext(
            patch.object(container_run, "_remove_enroot_container", return_value=True)
        )
        self.sweep = self.enterContext(patch.object(container_run, "_sweep_run_token_processes"))
        self.signal = self.enterContext(patch.object(container_run, "_signal_enroot_pg"))
        self.provision = self.enterContext(
            patch.object(container_run, "_provision_enroot_container")
        )
        self.passthrough = self.enterContext(
            patch.object(container_run, "_nvidia_manual_passthrough", return_value=([], {}))
        )
        self.enterContext(patch.object(runtime_common, "_eval_metric_env", return_value={}))
        self.enterContext(
            patch.dict(os.environ, {"AIDE_JOB_TOKEN": "test-job", "RUN_GROUP_SUFFIX": ""})
        )

    def snapshot(self, group):
        snapshot = group / "agent_snapshot"
        snapshot.mkdir(parents=True, exist_ok=True)
        grading = snapshot / "environment"
        grading.mkdir(exist_ok=True)
        for name in runtime_common.GRADING_SCRIPTS:
            (grading / name).write_text("# frozen grading script\n")
        (snapshot / "PROVENANCE.json").write_text(
            json.dumps(
                {
                    "src_aide_sha256": "test",
                    "grading_code_sha256": snapshots._hash_tree(grading),
                    "enroot_grading_protocol": runtime_common.ENROOT_GRADING_PROTOCOL,
                }
            )
        )
        (snapshot / "mlebench").mkdir(exist_ok=True)
        return snapshot

    def competition_for(self, name):
        return SimpleNamespace(
            id=name,
            public_dir=self.temp / "public",
            private_dir=self.temp / "private",
            grader=SimpleNamespace(name="mean-absolute-error", lower_is_better=True),
        )

    def run_container(self, **kwargs):
        self.proc.stdout = io.StringIO("")
        retain = kwargs.pop("retain_container", False)
        return container_run.run_in_enroot(
            self.competition,
            self.agent,
            "test.sqsh",
            retain,
            self.run_dir,
            logging.getLogger(__name__),
            ["0"],
            **kwargs,
        )

    def test_success_and_expected_budget_expiry_preserve_outputs(self):
        for returncode in (0, 124):
            with self.subTest(returncode=returncode):
                self.proc.returncode = returncode
                submission = self.run_dir / "submission" / "submission.csv"
                submission.parent.mkdir(exist_ok=True)
                submission.write_text("id,prediction\n1,0.5\n")
                self.assertEqual(self.run_container(), self.run_dir)
                self.assertEqual(submission.read_text(), "id,prediction\n1,0.5\n")

    def test_unexpected_exits_raise_and_still_clean_up(self):
        self.agent.env_vars["TEST_CREDENTIAL"] = "do-not-log-this-value"
        for returncode in (1, 2, 126, 127, 137, -9):
            with self.subTest(returncode=returncode):
                self.proc.returncode = returncode
                with self.assertRaisesRegex(
                    RuntimeError, f"exited with code {returncode}"
                ) as raised:
                    self.run_container()
                self.assertNotIn("do-not-log-this-value", str(raised.exception))
                self.sweep.assert_called()
                self.assertTrue(self.remove.call_args.args[0].startswith("competition-test-"))

    def test_124_requires_aide_and_a_configured_time_budget(self):
        self.proc.returncode = 124
        self.agent.env_vars = {}
        with self.assertRaisesRegex(RuntimeError, "exited with code 124"):
            self.run_container()
        self.agent.env_vars = {"TIME_LIMIT_SECS": "60"}
        self.agent.name = "another-agent"
        with self.assertRaisesRegex(RuntimeError, "exited with code 124"):
            self.run_container()

    def test_outer_watchdog_is_a_failure_even_after_clean_exit(self):
        self.proc.wait.side_effect = [subprocess.TimeoutExpired("enroot", 960), 0, 0]
        with self.assertRaisesRegex(TimeoutError, "watchdog deadline"):
            self.run_container()
        self.sweep.assert_called()

    def test_process_surviving_sigkill_does_not_report_success(self):
        self.proc.returncode = None
        self.proc.wait.side_effect = [
            subprocess.TimeoutExpired("enroot", n) for n in (960, 30, 120, 5)
        ]
        with self.assertRaisesRegex(
            container_run.EnrootCleanupError, "last exit code: None"
        ) as raised:
            self.run_container()
        self.assertIsInstance(raised.exception.__cause__, TimeoutError)

    def test_cpu_slice_wraps_the_enroot_command(self):
        self.run_container(cpus="2,8,31")
        cmd = self.popen.call_args.args[0]
        self.assertEqual(cmd[:4], ["taskset", "-c", "2,8,31", sys.executable])
        self.assertIn("--mount-proc", cmd)
        self.assertTrue(self.popen.call_args.kwargs["start_new_session"])

    def test_profile_cannot_replace_run_owned_grading_startup(self):
        self.agent.env_vars.update(
            {
                "GRADING_SERVER_PORT": "5000",
                "GRADING_SERVER_READY_FILE": "/tmp/another-run.json",
                "GRADING_SERVER_FD": "999999",
                "AIDE_RUN_TOKEN": "another-run",
            }
        )
        self.run_container()
        cmd = self.popen.call_args.args[0]
        env = self.command_env(cmd)
        self.assertIsNone(env["GRADING_SERVER_PORT"])
        self.assertIsNone(env["AIDE_GRADING_SERVER_URL"])
        self.assertIsNone(env["AIDE_VALIDATE_SERVER_URL"])
        self.assertNotIn("GRADING_SERVER_READY_FILE", env)
        self.assertTrue(env["AIDE_RUN_TOKEN"].startswith("competition-test-"))
        wrapper = (
            self.run_dir.parent / "agent_snapshot" / "environment" / "grading_server_wrapper.py"
        )
        self.assertEqual(cmd[:2], [sys.executable, str(wrapper)])
        host_env = self.popen.call_args.kwargs["env"]
        self.assertEqual(
            host_env["GRADING_SERVER_READY_FILE"],
            str(self.run_dir / "grading" / "grading_server_ready.json"),
        )
        grader_env = self.command_env(json.loads(host_env["GRADING_SERVER_COMMAND_JSON"]))
        self.assertEqual(grader_env["AIDE_RUN_TOKEN"], env["AIDE_RUN_TOKEN"])
        self.assertEqual(
            grader_env["GRADING_SERVER_READY_FILE"], "/run/grading/grading_server_ready.json"
        )
        self.assertIsNone(grader_env["GRADING_SERVER_FD"])
        self.assertNotIn("GRADING_SERVER_FD", env)

    @staticmethod
    def command_env(cmd):
        return {
            arg.partition("=")[0]: arg.partition("=")[2] if "=" in arg else None
            for i, arg in enumerate(cmd)
            if i and cmd[i - 1] == "--env"
        }

    def test_native_gpu_selection_uses_the_allocation_mapping(self):
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "2,5"}):
            self.run_container()
        self.assertEqual(self.popen.call_args.kwargs["env"]["NVIDIA_VISIBLE_DEVICES"], "2")

    def test_manual_passthrough_disables_inherited_and_profile_nvidia_hook(self):
        self.agent.env_vars.update(
            {
                "NVIDIA_VISIBLE_DEVICES": "all",
                "LD_LIBRARY_PATH": "/opt/custom-libs",
            }
        )
        self.passthrough.return_value = (
            [],
            {
                "CUDA_VISIBLE_DEVICES": "2",
                "LD_LIBRARY_PATH": "/opt/nvdrv",
                "NVIDIA_DRIVER_PASSTHROUGH": "manual",
            },
        )
        with patch.dict(os.environ, {"NVIDIA_VISIBLE_DEVICES": "all"}):
            self.run_container()
        cmd = self.popen.call_args.args[0]
        env = self.command_env(cmd)
        self.assertEqual(self.popen.call_args.kwargs["env"]["NVIDIA_VISIBLE_DEVICES"], "void")
        self.assertEqual(env["NVIDIA_VISIBLE_DEVICES"], "void")
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "2")
        self.assertEqual(env["LD_LIBRARY_PATH"], "/opt/nvdrv:/opt/custom-libs")

    def test_storage_path_is_pinned_for_provision_start_and_remove(self):
        path = self.temp / "rootfs"
        with patch.dict(os.environ, {"ENROOT_DATA_PATH": str(path)}):
            self.run_container()
        self.assertEqual(self.provision.call_args.args[3], path)
        self.assertEqual(self.popen.call_args.kwargs["env"]["ENROOT_DATA_PATH"], str(path))
        self.assertEqual(self.remove.call_args.args[1], path)

    def test_cleanup_failure_is_recorded_and_does_not_hide_an_agent_error(self):
        self.remove.return_value = False
        with self.assertRaisesRegex(container_run.EnrootCleanupError, "Enroot cleanup failed"):
            self.run_container()
        record = json.loads((self.run_dir / "runtime.json").read_text())
        self.assertFalse(record["cleanup_success"])
        self.assertGreaterEqual(record["total_seconds"], record["cleanup_seconds"])
        self.proc.returncode = 2
        with self.assertRaisesRegex(
            container_run.EnrootCleanupError, "exited with code 2"
        ) as raised:
            self.run_container()
        self.assertIsInstance(raised.exception.__cause__, RuntimeError)
        self.assertNotIsInstance(raised.exception.__cause__, container_run.EnrootCleanupError)

    def test_provision_timeout_still_cleans_partial_rootfs_and_records_timings(self):
        self.provision.side_effect = TimeoutError("base extraction deadline")
        with self.assertRaisesRegex(TimeoutError, "base extraction deadline"):
            self.run_container()
        self.popen.assert_not_called()
        self.remove.assert_called_once()
        record = json.loads((self.run_dir / "runtime.json").read_text())
        self.assertIsNone(record["returncode"])
        self.assertTrue(record["cleanup_success"])
        self.assertEqual(record["container_seconds"], 0)

    def test_retaining_container_still_stops_processes_and_records_retention(self):
        self.run_container(retain_container=True)
        self.remove.assert_not_called()
        self.sweep.assert_called_once()
        record = json.loads((self.run_dir / "runtime.json").read_text())
        self.assertTrue(record["container_retained"])

    def test_grading_mounts_use_the_snapshot(self):
        self.run_container()
        cmd = self.popen.call_args.args[0]
        source = self.run_dir.parent / "agent_snapshot" / "environment"
        self.assertEqual(cmd[1], str(source / "grading_server_wrapper.py"))
        grader_cmd = json.loads(self.popen.call_args.kwargs["env"]["GRADING_SERVER_COMMAND_JSON"])
        self.assertTrue(
            any(
                arg.startswith(f"{source}/grading_server.py:/private/grading_server.py:")
                for arg in grader_cmd
            )
        )
        self.assertTrue(any(":/mlebench/mlebench:none:bind,ro" in arg for arg in grader_cmd))

    def test_private_data_and_grader_control_are_only_mounted_in_separate_grader(self):
        with patch.dict(os.environ, {"ENROOT_MOUNT_HOME": "yes", "ENROOT_CONFIG_PATH": "/custom"}):
            self.run_container()
        agent_cmd = self.popen.call_args.args[0]
        host_env = self.popen.call_args.kwargs["env"]
        grader_cmd = json.loads(host_env["GRADING_SERVER_COMMAND_JSON"])
        private_bind = f"{self.competition.private_dir}:/private/data/test/prepared/private:"
        self.assertFalse(any(arg.startswith(private_bind) for arg in agent_cmd))
        self.assertTrue(any(arg.startswith(private_bind) for arg in grader_cmd))
        self.assertFalse(any(":/run/grading:" in arg for arg in agent_cmd))
        self.assertFalse(any(":/home/logs:" in arg for arg in grader_cmd))
        for cmd in (agent_cmd, grader_cmd):
            self.assertIn("--pid", cmd)
            self.assertIn("--mount-proc", cmd)
            self.assertIn("--kill-child=KILL", cmd)
        self.assertEqual(host_env["ENROOT_MOUNT_HOME"], "no")
        self.assertEqual(host_env["ENROOT_ALLOW_SUPERUSER"], "no")
        self.assertEqual(
            host_env["ENROOT_CONFIG_PATH"], str(self.run_dir / "grading" / "enroot-config")
        )
        self.assertEqual(self.command_env(grader_cmd)["NVIDIA_VISIBLE_DEVICES"], "void")
        roots = [call.args[0] for call in self.provision.call_args_list]
        self.assertEqual(len(set(roots)), 2)
        self.assertEqual([call.args[0] for call in self.remove.call_args_list], roots[::-1])

    def test_grader_provision_failure_removes_both_roots_and_never_launches_agent(self):
        self.provision.side_effect = [None, TimeoutError("grader clone deadline")]
        with self.assertRaisesRegex(TimeoutError, "grader clone deadline"):
            self.run_container()
        self.popen.assert_not_called()
        self.assertEqual(self.remove.call_count, 2)

    def test_failed_grader_removal_still_attempts_agent_removal(self):
        self.remove.side_effect = [False, True]
        with self.assertRaises(container_run.EnrootCleanupError):
            self.run_container()
        self.assertEqual(self.remove.call_count, 2)

    def configure_scheduler(self, workers=1, competition_ids=("first", "second")):
        profile = self.temp / "profile.yaml"
        profile.write_text("{}")
        split = self.temp / "split.txt"
        split.write_text("\n".join(competition_ids) + "\n")
        container_config = self.temp / "container.json"
        container_config.write_text("{}")
        args = SimpleNamespace(
            n_workers=workers,
            runtime="enroot",
            data_dir=self.temp,
            agent_id=self.agent.id,
            agent_name="aide",
            agent_dir=self.temp,
            agent_config=profile,
            extra_kwargs=[],
            extra_env=[],
            competition_set=split,
            container_config=container_config,
            n_seeds=1,
            enroot_image="test.sqsh",
            retain=False,
            gpu_device=["0"],
        )
        registry = Mock()
        registry.set_data_dir.return_value = registry
        registry.get_competition.side_effect = self.competition_for
        agent_registry = Mock()
        agent_registry.get_agent_from_config.return_value = self.agent

        def create_run_dir(competition_id, agent_id, group):
            path = self.temp / group / competition_id
            path.mkdir(parents=True, exist_ok=True)
            return path

        self.enterContext(patch.object(run_agent, "args", args, create=True))
        self.enterContext(patch.object(run_agent, "registry", registry))
        self.enterContext(patch.object(run_agent, "Registry", return_value=agent_registry))
        self.enterContext(patch.object(run_agent, "is_dataset_prepared", return_value=True))
        self.enterContext(patch.object(run_agent, "get_timestamp", return_value="test"))
        self.enterContext(patch.object(run_agent, "get_runs_dir", return_value=self.temp))
        self.enterContext(patch.object(run_agent, "create_run_dir", side_effect=create_run_dir))
        self.enterContext(
            patch.object(
                run_agent,
                "snapshot_agent_code",
                side_effect=lambda _, group, logger: self.snapshot(group),
            )
        )
        self.enterContext(patch.object(run_agent, "_cpu_slices", return_value=["2,4"] * workers))
        return args

    def test_scheduler_records_failures_finishes_queue_and_returns_nonzero(self):
        args = self.configure_scheduler()
        for returncode in (0, 1, 124, 137):
            with self.subTest(returncode=returncode):

                def launch(cmd, **kwargs):
                    first = any(arg.startswith("competition-first-") for arg in cmd)
                    if not first:
                        checkpoint = json.loads(
                            (self.temp / "test_run-group_aide" / "metadata.json").read_text()
                        )
                        self.assertEqual(checkpoint["status"], "running")
                        self.assertEqual(checkpoint["expected_runs"], 2)
                        self.assertIn("first", checkpoint["runs"])
                        self.assertNotIn("second", checkpoint["runs"])
                    code = returncode if first else 0
                    return Mock(stdout=io.StringIO(""), returncode=code)

                self.popen.side_effect = launch
                status = asyncio.run(asyncio.wait_for(run_agent.main(args), timeout=10))
                expected_success = returncode in (0, 124)
                self.assertEqual(status, 0 if expected_success else 1)
                metadata = json.loads(
                    (self.temp / "test_run-group_aide" / "metadata.json").read_text()
                )
                self.assertEqual(metadata["status"], "complete")
                self.assertEqual(metadata["runs"]["first"]["success"], expected_success)
                self.assertTrue(metadata["runs"]["second"]["success"])
                for cid in ("first", "second"):
                    self.assertEqual(metadata["runs"][cid]["metric_name"], "mean-absolute-error")
                    self.assertIs(metadata["runs"][cid]["metric_lower_is_better"], True)
                    task = json.loads(
                        (self.temp / "test_run-group_aide" / cid / "task.json").read_text()
                    )
                    self.assertEqual(task["metric_name"], "mean-absolute-error")
                    self.assertIs(task["metric_lower_is_better"], True)
                if not expected_success:
                    self.assertIn(
                        f"exited with code {returncode}", metadata["runs"]["first"]["error"]
                    )
                self.assertEqual(self.popen.call_args.args[0][:3], ["taskset", "-c", "2,4"])

    def test_scheduler_stops_new_tasks_after_cleanup_failure(self):
        args = self.configure_scheduler()
        self.remove.return_value = False
        for returncode in (0, 2):
            with self.subTest(agent_exit=returncode):
                self.popen.reset_mock()
                self.popen.side_effect = lambda *a, **kw: Mock(
                    stdout=io.StringIO(""),
                    returncode=returncode,
                )
                status = asyncio.run(asyncio.wait_for(run_agent.main(args), timeout=10))
                self.assertEqual(status, 1)
                self.popen.assert_called_once()
                metadata = json.loads(
                    (self.temp / "test_run-group_aide" / "metadata.json").read_text()
                )
                self.assertEqual(metadata["status"], "aborted")
                self.assertIn("first", metadata["abort_reason"])
                self.assertFalse(metadata["runs"]["first"]["cleanup_success"])
                self.assertTrue(metadata["runs"]["second"]["skipped"])
                self.assertFalse(metadata["runs"]["second"]["success"])
                if returncode:
                    self.assertIn("exited with code 2", metadata["runs"]["first"]["error"])

    def test_cleanup_failure_allows_in_flight_tasks_to_finish(self):
        args = self.configure_scheduler(workers=2, competition_ids=("first", "second", "third"))
        second_running = threading.Event()
        allow_second_to_finish = threading.Event()
        calls = []

        def run_task(**kwargs):
            task_id = kwargs["competition"].id
            calls.append(task_id)
            if task_id == "first":
                if not second_running.wait(timeout=3):
                    raise AssertionError("second task never started")
                raise container_run.EnrootCleanupError("first rootfs could not be removed")
            if task_id == "second":
                second_running.set()
                if not allow_second_to_finish.wait(timeout=5):
                    raise AssertionError("second task was not allowed to finish")
                return kwargs["run_dir"]
            raise AssertionError("third task must not start after cleanup failure")

        self.enterContext(patch.object(run_agent, "run_in_enroot", side_effect=run_task))
        metadata_path = self.temp / "test_run-group_aide" / "metadata.json"

        async def exercise():
            future = asyncio.create_task(run_agent.main(args))
            try:
                deadline = time.monotonic() + 3
                while time.monotonic() < deadline:
                    if metadata_path.exists():
                        checkpoint = json.loads(metadata_path.read_text())
                        if checkpoint["status"] == "stopping":
                            break
                    await asyncio.sleep(0.01)
                else:
                    self.fail("cleanup failure did not stop new task launches")
                self.assertFalse(future.done())
                self.assertNotIn("second", checkpoint["runs"])
            finally:
                allow_second_to_finish.set()
                status = await asyncio.wait_for(future, timeout=5)
            self.assertEqual(status, 1)

        asyncio.run(exercise())
        metadata = json.loads(metadata_path.read_text())
        self.assertEqual(metadata["status"], "aborted")
        self.assertTrue(metadata["runs"]["second"]["success"])
        self.assertTrue(metadata["runs"]["third"]["skipped"])
        self.assertEqual(sorted(calls), ["first", "second"])


class StartScriptTests(unittest.TestCase):
    def setUp(self):
        self.temp = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.bin_dir = self.temp / "bin"
        self.bin_dir.mkdir()
        for cmd in ("conda", "python", "cp", "sed", "cat", "envsubst"):
            self.command(cmd, "exit 0")
        self.command("nvidia-smi", "exit 1")
        self.command("timeout", 'exit "$MLE_TEST_AGENT_EXIT_CODE"')
        self.script = Path(__file__).resolve().parents[1] / "agents" / "aide" / "start.sh"

    def command(self, name, body):
        path = self.bin_dir / name
        path.write_text(f"#!/bin/bash\n{body}\n")
        path.chmod(0o755)

    def launch(self, returncode, **extra_env):
        agent_dir = self.temp / f"agent-{returncode}"
        agent_dir.mkdir()
        (agent_dir / "additional_notes.txt").write_text("")
        (agent_dir / "snapshot_check.sh").write_text("exit 1\n")
        env = {
            "PATH": f"{self.bin_dir}:/usr/bin:/bin",
            "AGENT_DIR": str(agent_dir),
            "LOGS_DIR": str(self.temp / "logs"),
            "CODE_DIR": str(self.temp / "code"),
            "SUBMISSION_DIR": str(self.temp / "submission"),
            "TIME_LIMIT_SECS": "1",
            "MLE_TEST_AGENT_EXIT_CODE": str(returncode),
            **extra_env,
        }
        return subprocess.run(
            ["bash", str(self.script)], env=env, capture_output=True, text=True, timeout=10
        )

    def test_shell_preserves_agent_and_timeout_exit_codes(self):
        for code in (0, 1, 2, 124, 137):
            with self.subTest(code=code):
                result = self.launch(code)
                self.assertEqual(result.returncode, code, result.stderr)

    def test_snapshot_failure_stops_startup(self):
        result = self.launch(0, AIDE_SNAPSHOT_SHA="test")
        self.assertEqual(result.returncode, 1)
        self.assertNotIn("conda activate", result.stderr)

    def test_real_gnu_timeout_is_reported_as_124(self):
        (self.bin_dir / "timeout").unlink()
        self.command("aide", "exec /usr/bin/sleep 5")
        result = self.launch(0)
        self.assertEqual(result.returncode, 124, result.stderr)


if __name__ == "__main__":
    unittest.main()
