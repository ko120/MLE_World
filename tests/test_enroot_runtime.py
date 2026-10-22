"""Enroot provisioning and snapshot regressions; no Enroot containers are launched."""

import json
import logging
import os
import runpy
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from agents import runtime_enroot as run
from agents import snapshots


class ProvisioningTests(unittest.TestCase):
    def setUp(self):
        self.temp = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.data = self.temp / "data"
        self.image = self.temp / "image.sqsh"
        self.image.write_bytes(b"image one")
        self.logger = logging.getLogger(__name__)
        self.enterContext(patch.object(run, "_ENROOT_BASE_NAMES", {}))
        self.enterContext(patch.object(run, "_ENROOT_REFLINK_OK", {}))
        self.enterContext(
            patch.dict(
                os.environ,
                {
                    "AIDE_ENROOT_NO_COW": "",
                    "AIDE_ENROOT_PROVISION_TIMEOUT_SECS": "10",
                },
            )
        )
        self.command = self.enterContext(
            patch.object(
                run,
                "_run_enroot_command",
                side_effect=self.command_succeeds,
            )
        )

    def command_succeeds(self, command, logger, **kwargs):
        if command[:2] == ["enroot", "create"]:
            (Path(kwargs["env"]["ENROOT_DATA_PATH"]) / command[3]).mkdir(parents=True)
        elif command[:2] == ["enroot", "remove"]:
            shutil.rmtree(Path(kwargs["env"]["ENROOT_DATA_PATH"]) / command[3])

    def base(self, image=None, data=None):
        return run._ensure_enroot_base(
            str(image or self.image),
            data or self.data,
            self.logger,
            time.monotonic() + 10,
        )

    def test_base_cache_is_scoped_by_image_version_and_storage(self):
        first = self.base()
        self.assertEqual(self.base(), first)
        self.assertEqual(self.command.call_count, 1)

        second_image = self.temp / "other.sqsh"
        second_image.write_bytes(b"image two")
        self.assertNotEqual(self.base(image=second_image), first)
        self.assertNotEqual(self.base(data=self.temp / "other-data"), first)

        self.image.write_bytes(b"replacement image with a different size")
        self.assertNotEqual(self.base(), first)
        self.assertEqual(self.command.call_count, 4)

    def test_missing_cached_base_is_recreated(self):
        first = self.base()
        shutil.rmtree(self.data / first)
        self.assertNotEqual(self.base(), first)

    def test_partial_base_is_removed_and_next_attempt_can_succeed(self):
        attempts = []

        def fail_once(command, logger, **kwargs):
            self.command_succeeds(command, logger, **kwargs)
            if command[:2] == ["enroot", "create"]:
                attempts.append(command[3])
                if len(attempts) == 1:
                    raise TimeoutError("stalled extraction")

        self.command.side_effect = fail_once
        with self.assertRaisesRegex(TimeoutError, "stalled extraction"):
            self.base()
        self.assertEqual(run._ENROOT_BASE_NAMES, {})
        self.assertFalse((self.data / attempts[0]).exists())
        self.assertNotEqual(self.base(), attempts[0])

    def test_partial_base_cleanup_failure_keeps_the_provisioning_cause(self):
        def fail(command, logger, **kwargs):
            if command[:2] == ["enroot", "create"]:
                self.command_succeeds(command, logger, **kwargs)
                raise TimeoutError("stalled extraction")
            raise subprocess.CalledProcessError(1, command)

        self.command.side_effect = fail
        with self.assertRaisesRegex(run.EnrootCleanupError, "partial base") as raised:
            self.base()
        self.assertIsInstance(raised.exception.__cause__, TimeoutError)
        self.assertIn("stalled extraction", str(raised.exception))
        self.assertEqual(run._ENROOT_BASE_NAMES, {})

    def test_reflink_probe_is_cached_per_storage_path(self):
        other = self.temp / "other-data"

        def probe(command, logger, **kwargs):
            if str(other) in command[-1]:
                raise subprocess.CalledProcessError(1, command)

        self.command.side_effect = probe
        deadline = time.monotonic() + 10
        self.assertTrue(run._reflink_supported(self.data, self.logger, deadline))
        self.assertTrue(run._reflink_supported(self.data, self.logger, deadline))
        self.assertFalse(run._reflink_supported(other, self.logger, deadline))
        self.assertEqual(self.command.call_count, 2)
        self.assertEqual(list(self.data.iterdir()), [])

    def test_reflink_timeout_is_not_cached_as_unsupported(self):
        self.command.side_effect = TimeoutError("probe deadline")
        with self.assertRaisesRegex(TimeoutError, "probe deadline"):
            run._reflink_supported(self.data, self.logger, time.monotonic() + 10)
        self.assertNotIn(self.data, run._ENROOT_REFLINK_OK)

    def test_unsupported_storage_never_falls_back_to_full_unpack(self):
        with patch.object(run, "_reflink_supported", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "requires reflink support"):
                run._provision_enroot_container("task", str(self.image), self.logger, self.data)
        self.command.assert_not_called()

    def test_legacy_full_unpack_override_is_rejected_before_provisioning(self):
        with patch.dict(os.environ, {"AIDE_ENROOT_NO_COW": "1"}):
            with self.assertRaisesRegex(RuntimeError, "no longer supported"):
                run._provision_enroot_container("task", str(self.image), self.logger, self.data)
        self.command.assert_not_called()

    def test_waiting_for_another_base_has_a_deadline(self):
        lock = threading.Lock()
        lock.acquire()
        with patch.object(run, "_ENROOT_BASE_LOCK", lock):
            try:
                with self.assertRaisesRegex(TimeoutError, "shared base lock"):
                    run._ensure_enroot_base(
                        str(self.image),
                        self.data,
                        self.logger,
                        time.monotonic() + 0.03,
                    )
            finally:
                lock.release()
        self.command.assert_not_called()

    def test_clone_slot_wait_has_a_deadline(self):
        with patch.object(run, "_reflink_supported", return_value=True):
            with patch.object(run, "_ENROOT_CREATE_SEM", threading.Semaphore(0)):
                with patch.dict(os.environ, {"AIDE_ENROOT_PROVISION_TIMEOUT_SECS": "0.03"}):
                    with self.assertRaisesRegex(TimeoutError, "clone slot"):
                        run._provision_enroot_container(
                            "task",
                            str(self.image),
                            self.logger,
                            self.data,
                        )
        self.assertTrue(all(c.args[0][0] == "enroot" for c in self.command.call_args_list))

    def test_removal_checks_exit_status_and_never_reports_false_success(self):
        (self.data / "task").mkdir(parents=True)
        self.command.side_effect = subprocess.CalledProcessError(1, ["enroot", "remove"])
        with self.assertLogs(level="ERROR") as messages:
            self.assertFalse(run._remove_enroot_container("task", self.data, self.logger))
        self.assertIn("could not be removed", "\n".join(messages.output))
        self.assertEqual(self.command.call_args.kwargs["env"]["ENROOT_DATA_PATH"], str(self.data))

    def test_removal_checks_that_the_rootfs_was_deleted(self):
        (self.data / "task").mkdir(parents=True)
        self.command.side_effect = None  # Command exits zero but leaves the tree behind.
        self.assertFalse(run._remove_enroot_container("task", self.data, self.logger))

    def test_removal_warning_interval_replaces_the_old_deadline(self):
        with patch.dict(
            os.environ,
            {
                "AIDE_ENROOT_REMOVE_WARN_SECS": "0.2",
                "AIDE_ENROOT_REMOVE_TIMEOUT_SECS": "0.1",
            },
        ):
            (self.data / "task").mkdir(parents=True)
            self.assertTrue(run._remove_enroot_container("task", self.data, self.logger))
            self.assertIsNone(self.command.call_args.kwargs["timeout"])
            self.assertEqual(self.command.call_args.kwargs["warn_after"], 0.2)
        with patch.dict(
            os.environ,
            {
                "AIDE_ENROOT_REMOVE_WARN_SECS": "",
                "AIDE_ENROOT_REMOVE_TIMEOUT_SECS": "0.1",
            },
        ):
            (self.data / "task").mkdir()
            self.assertTrue(run._remove_enroot_container("task", self.data, self.logger))
            self.assertEqual(self.command.call_args.kwargs["warn_after"], 0.1)

    def test_bad_warning_setting_does_not_prevent_required_cleanup(self):
        with patch.dict(os.environ, {"AIDE_ENROOT_REMOVE_WARN_SECS": "nan"}):
            (self.data / "task").mkdir(parents=True)
            self.assertTrue(run._remove_enroot_container("task", self.data, self.logger))
            self.assertEqual(self.command.call_args.kwargs["warn_after"], 120)

    def test_timeout_settings_reject_unbounded_values(self):
        for value in ("0", "-1", "nan", "inf"):
            with self.subTest(value=value), patch.dict(os.environ, {"TEST_ENROOT_TIMEOUT": value}):
                with self.assertRaises(ValueError):
                    run._enroot_seconds("TEST_ENROOT_TIMEOUT", 10)


class CommandTimeoutTests(unittest.TestCase):
    def test_slow_cleanup_warns_and_finishes_deleting_without_termination(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "remaining-rootfs-file"
            target.write_text("delete after warning")
            command = [
                sys.executable,
                "-c",
                "import sys,time; from pathlib import Path; time.sleep(0.2); Path(sys.argv[1]).unlink()",
                str(target),
            ]
            with patch.object(run, "_signal_enroot_pg") as send_signal:
                with self.assertLogs(__name__, level="WARNING") as messages:
                    run._run_enroot_command(
                        command,
                        logging.getLogger(__name__),
                        phase="slow removal",
                        timeout=None,
                        warn_after=0.05,
                    )
            send_signal.assert_not_called()
            self.assertFalse(target.exists())
            self.assertIn("waiting for cleanup to finish", "\n".join(messages.output))

    def test_unresponsive_leader_cannot_make_cleanup_wait_forever(self):
        proc = Mock(returncode=None)
        proc.wait.side_effect = [subprocess.TimeoutExpired("test", 1)] * 3
        with patch.object(run.subprocess, "Popen", return_value=proc):
            with patch.object(run, "_signal_enroot_pg") as send_signal:
                with self.assertRaisesRegex(TimeoutError, "exceeded"):
                    run._run_enroot_command(
                        ["test"],
                        logging.getLogger(__name__),
                        phase="test",
                        timeout=1,
                    )
        self.assertEqual(
            [c.args[1] for c in send_signal.call_args_list], [signal.SIGTERM, signal.SIGKILL]
        )
        self.assertTrue(all(c.kwargs["timeout"] is not None for c in proc.wait.call_args_list))

    @unittest.skipUnless(sys.platform == "linux", "Linux process-group cleanup")
    def test_timeout_kills_a_descendant_even_when_leader_exits_on_term(self):
        with tempfile.TemporaryDirectory() as tmp:
            marker = Path(tmp) / "child.pid"
            child = (
                "import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(60)"
            )
            parent = (
                "import subprocess,sys,time; from pathlib import Path; "
                f"child=subprocess.Popen([sys.executable,'-c',{child!r}]); "
                f"Path({str(marker)!r}).write_text(str(child.pid)); time.sleep(60)"
            )
            try:
                with self.assertRaisesRegex(TimeoutError, "test descendant"):
                    run._run_enroot_command(
                        [sys.executable, "-c", parent],
                        logging.getLogger(__name__),
                        phase="test descendant",
                        timeout=1,
                    )
                pid = int(marker.read_text())
                deadline = time.monotonic() + 2
                while time.monotonic() < deadline:
                    try:
                        state = Path(f"/proc/{pid}/stat").read_text().split(") ", 1)[1].split()[0]
                    except FileNotFoundError:
                        break
                    if state == "Z":
                        break
                    time.sleep(0.02)
                else:
                    self.fail("grandchild survived the timeout cleanup")
            finally:
                if marker.exists():
                    try:
                        os.kill(int(marker.read_text()), signal.SIGKILL)
                    except ProcessLookupError:
                        pass


class GpuMappingTests(unittest.TestCase):
    def test_allocation_mapping_supports_indices_and_uuids(self):
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "2,5"}):
            self.assertEqual(run._resolve_enroot_gpus(["0", "1"]), ["2", "5"])
            self.assertEqual(run._resolve_enroot_gpus(["GPU-explicit"]), ["GPU-explicit"])
            for value in ("-1", "2"):
                with self.assertRaises(ValueError):
                    run._resolve_enroot_gpus([value])
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "GPU-first,GPU-second"}):
            self.assertEqual(run._resolve_enroot_gpus(["1"]), ["GPU-second"])
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": ""}):
            self.assertEqual(run._resolve_enroot_gpus(["3"]), ["3"])

    def test_manual_passthrough_uses_the_shared_mapping(self):
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "2,5"}):
            with patch.object(run, "_nvidia_driver_stage", return_value="/tmp/test-driver"):
                _, env = run._nvidia_manual_passthrough(["1"], logging.getLogger(__name__))
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "5")
        self.assertEqual(env["NVIDIA_VISIBLE_DEVICES"], "void")


class DriverStagingTests(unittest.TestCase):
    def test_processes_stage_separately_and_library_aliases_copy_successfully(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            libs = root / "lib"
            libs.mkdir()
            (libs / "libcuda.so.999").write_bytes(b"test driver")
            (libs / "libcuda.so.1").symlink_to("libcuda.so.999")
            alias = root / "usr-lib"
            alias.symlink_to(libs, target_is_directory=True)
            data = root / "enroot-data"
            self.enterContext(patch.object(run, "_NVDRV_LIB_DIRS", (str(libs), str(alias))))
            self.enterContext(patch.object(run, "_enroot_data_path", return_value=data))
            self.enterContext(patch.object(run.shutil, "which", return_value=None))
            stages = []
            for pid in (101, 102):
                with (
                    patch.object(run, "_nvdrv_stage", None),
                    patch.object(run.os, "getpid", return_value=pid),
                ):
                    stage = Path(run._nvidia_driver_stage(logging.getLogger(__name__)))
                    stages.append(stage)
                    self.assertEqual(stage.parent, data)
                    self.assertTrue((stage / "libcuda.so.1").is_symlink())
                    self.assertEqual((stage / "libcuda.so.1").read_bytes(), b"test driver")
                    self.assertEqual(
                        run._nvidia_driver_stage(logging.getLogger(__name__)), str(stage)
                    )
            self.assertNotEqual(*stages)


@unittest.skipUnless(
    (Path(__file__).resolve().parents[1] / "scripts/eval/wm_fuse_eval.py").is_file(),
    "requires the optional workspace fusion replay tool",
)
class ReplayRuntimeTests(unittest.TestCase):
    def test_fusion_replay_resets_only_a_missing_driver_stage(self):
        script = Path(__file__).resolve().parents[1] / "scripts/eval/wm_fuse_eval.py"
        with patch.object(sys, "path", sys.path.copy()):
            replay = runpy.run_path(str(script))["replay"]
        args = SimpleNamespace(shard_id=0, num_shards=1)
        with tempfile.TemporaryDirectory() as directory:
            existing = str(Path(directory))
            missing = str(Path(directory) / "removed-stage")
            for cached, expected in [(missing, None), (existing, existing), ("", ""), (None, None)]:
                with self.subTest(cached=cached):
                    with patch.object(run, "_nvdrv_stage", cached):
                        # Stop before loading cases or provisioning any containers.
                        with patch.dict(
                            replay.__globals__,
                            selected_cases=Mock(side_effect=ValueError("reached case loading")),
                        ):
                            with self.assertRaisesRegex(ValueError, "reached case loading"):
                                replay(args)
                        self.assertEqual(run._nvdrv_stage, expected)


@unittest.skipUnless(
    (Path(__file__).resolve().parents[1] / "scripts/run/run_mlebench_aide_enroot.sh").is_file(),
    "requires the optional workspace Slurm launchers",
)
class LauncherCleanupTests(unittest.TestCase):
    def test_cleanup_command_warns_and_waits_in_both_launchers(self):
        launchers = Path(__file__).resolve().parents[1] / "scripts" / "run"
        for name in ("run_mlebench_aide_enroot.sh", "run_mlebench_aide_sim_enroot.sh"):
            with self.subTest(launcher=name):
                source = (launchers / name).read_text()
                body = source.split("enroot_cleanup_command() {\n", 1)[1].split("\n}\n", 1)[0]
                script = "enroot_cleanup_command() {\n" + body + '\n}\nenroot_cleanup_command "$@"'
                env = {
                    **os.environ,
                    "REPO_ROOT": str(launchers.parents[1]),
                    "AIDE_ENROOT_REMOVE_WARN_SECS": "0.05",
                }
                result = subprocess.run(
                    [
                        "bash",
                        "-c",
                        script,
                        "cleanup-test",
                        sys.executable,
                        "-c",
                        "import time; time.sleep(0.2); print('cleanup completed')",
                    ],
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=3,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("waiting for completion", result.stderr)
                self.assertIn("cleanup completed", result.stdout)
                result = subprocess.run(
                    [
                        "bash",
                        "-c",
                        script,
                        "cleanup-test",
                        sys.executable,
                        "-c",
                        "raise SystemExit(7)",
                    ],
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=3,
                )
                self.assertEqual(result.returncode, 7, result.stderr)
                self.assertIn("cleanup failed with exit code 7", result.stderr)

    def test_exit_cleanup_attempts_all_sweeps_and_reports_failure(self):
        launchers = Path(__file__).resolve().parents[1] / "scripts" / "run"
        for name in ("run_mlebench_aide_enroot.sh", "run_mlebench_aide_sim_enroot.sh"):
            source = (launchers / name).read_text()
            body = source.split("\ncleanup() {\n", 1)[1].split("\n}\n", 1)[0]
            setup = """
set -euo pipefail
_CLEANUP_RUNNING=0
SHARED_NODE=1
VLLM_PID=''
FB_VLLM_PID=''
SIMULATOR_VLLM_PID=''
GPU_GUARD_DEVICES=''
sweep_aide_job_processes() { return 0; }
sweep_enroot() { echo 'name sweep'; return 1; }
sweep_enroot_data() { echo 'data sweep'; return 0; }
postflight_gpu_guard() { echo 'postflight'; return 0; }
"""
            for original_status in (0, 7):
                with self.subTest(launcher=name, original_status=original_status):
                    script = (
                        setup + "\ncleanup() {\n" + body + "\n}\ntrap cleanup EXIT\n"
                        f"exit {original_status}\n"
                    )
                    result = subprocess.run(
                        ["bash", "-c", script],
                        capture_output=True,
                        text=True,
                        timeout=3,
                    )
                    self.assertEqual(result.returncode, original_status or 1, result.stderr)
                    self.assertEqual(
                        result.stdout.splitlines(), ["name sweep", "data sweep", "postflight"]
                    )


class GradingSnapshotTests(unittest.TestCase):
    def setUp(self):
        self.temp = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.repo = self.temp / "repo"
        self.agent = self.repo / "agents" / "aide"
        (self.agent / "src" / "aide").mkdir(parents=True)
        (self.agent / "src" / "aide" / "__init__.py").write_text("VERSION = 1\n")
        original = Path(snapshots.__file__).parent / "aide" / "snapshot_check.sh"
        shutil.copyfile(original, self.agent / "snapshot_check.sh")
        (self.repo / "mlebench").mkdir()
        (self.repo / "mlebench" / "__init__.py").write_text("# grader package\n")
        (self.repo / "environment").mkdir()
        for name in snapshots.GRADING_SCRIPTS:
            (self.repo / "environment" / name).write_text("VERSION = 1\n")
        self.group = self.temp / "group"
        self.group.mkdir()
        self.enterContext(
            patch.object(snapshots, "__file__", str(self.repo / "agents" / "snapshots.py"))
        )
        self.logger = logging.getLogger(__name__)

    def snapshot(self):
        return snapshots.snapshot_agent_code(self.agent, self.group, self.logger)

    def test_grading_scripts_stay_frozen_after_live_edits_and_reuse(self):
        snapshot = self.snapshot()
        for name in snapshots.GRADING_SCRIPTS:
            (self.repo / "environment" / name).write_text("VERSION = 2\n")
        self.assertEqual(self.snapshot(), snapshot)
        grading = snapshots._grading_snapshot(snapshot)
        for name in snapshots.GRADING_SCRIPTS:
            self.assertEqual((grading / name).read_text(), "VERSION = 1\n")
        provenance = json.loads((snapshot / "PROVENANCE.json").read_text())
        self.assertEqual(provenance["grading_code_sha256"], snapshots._hash_tree(grading))

    def test_old_snapshot_is_rejected_without_backfilling_live_code(self):
        snapshot = self.group / "agent_snapshot"
        snapshot.mkdir()
        provenance = snapshot / "PROVENANCE.json"
        original = json.dumps({"src_aide_sha256": "old"})
        provenance.write_text(original)
        with self.assertRaisesRegex(RuntimeError, "no frozen grading server"):
            self.snapshot()
        self.assertEqual(provenance.read_text(), original)
        self.assertFalse((snapshot / "environment").exists())

    def test_tampered_grading_snapshot_is_rejected(self):
        snapshot = self.snapshot()
        (snapshot / "environment" / "grading_server.py").write_text("VERSION = 3\n")
        with self.assertRaisesRegex(RuntimeError, "hash mismatch"):
            snapshots._grading_snapshot(snapshot)

    def test_pre_isolation_snapshot_requires_a_new_group(self):
        snapshot = self.snapshot()
        path = snapshot / "PROVENANCE.json"
        provenance = json.loads(path.read_text())
        del provenance["enroot_grading_protocol"]
        original = json.dumps(provenance)
        path.write_text(original)
        with self.assertRaisesRegex(RuntimeError, "older Enroot grading protocol"):
            self.snapshot()
        self.assertEqual(path.read_text(), original)


if __name__ == "__main__":
    unittest.main()
