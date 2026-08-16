import hashlib
import json
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from tools.training_panel.training_panel.commands import (
    DEFAULT_TASK,
    EvaluationParams,
    TrainingParams,
    VideoParams,
    resolve_spring_backend,
)
from tools.training_panel.training_panel.config import PanelPaths
from tools.training_panel.training_panel.history import HistoryStore
from tools.training_panel.training_panel.processes import (
    CudaPreflightError,
    EXTERNAL_GPU_ID_PREFIX,
    EXTERNAL_TRAINING_ID_PREFIX,
    EXTERNAL_VIDEO_ID_PREFIX,
    GpuHostLeaseBusy,
    ProcessInfo,
    ProcessRegistry,
    ProcessStartError,
    SpawnedProcess,
)


class ProcessRegistryTests(unittest.TestCase):
    def make_paths(self, root: Path) -> PanelPaths:
        conda_sh = root / "conda.sh"
        conda_sh.write_text("conda() { :; }\n", encoding="utf-8")
        isaaclab_root = root / "IsaacLab"
        isaaclab_root.mkdir()
        launcher = isaaclab_root / "isaaclab.sh"
        launcher.write_text(
            "#!/usr/bin/env bash\n"
            "echo fake isaaclab \"$@\"\n"
            "sleep 3\n",
            encoding="utf-8",
        )
        os.chmod(launcher, 0o755)
        return PanelPaths(
            repo_root=root,
            isaaclab_root=isaaclab_root,
            isaacsim_root=root / "isaacsim",
            conda_sh=conda_sh,
            conda_env="env",
        )

    @staticmethod
    def make_evaluation_params(root: Path) -> EvaluationParams:
        checkpoint = root / "model_1.pt"
        checkpoint.write_bytes(b"checkpoint")
        command_profile = root / "command_profile.json"
        command_profile.write_text('{"commands":[]}\n', encoding="utf-8")
        identity = "a" * 64
        return EvaluationParams(
            source_run_id="panel_source",
            checkpoint=str(checkpoint),
            checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            task="Template-Redrhex-Direct-v0",
            agent_entry_point="rsl_rl_cfg_entry_point",
            seed=42,
            evaluation_profile="stage1",
            curriculum_stage=1,
            command_profile_file=str(command_profile),
            command_profile_sha256=hashlib.sha256(command_profile.read_bytes()).hexdigest(),
            code_sha256=identity,
            config_sha256=identity,
            dependency_sha256=identity,
            reward_profile_sha256=identity,
            physics_identity_sha256=identity,
            spring_identity_sha256=identity,
            terrain_profile_sha256=identity,
            num_envs=1,
            device="cpu",
            campaign_id="campaign-launch-recovery",
            campaign_trial_id="trial-launch-recovery",
        )

    def test_training_record_preserves_tweak_metadata(self):
        class FakeProcess:
            pid = 12345

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = self.make_paths(root)
            history = HistoryStore(paths)
            registry = ProcessRegistry(paths, history)
            params = TrainingParams.from_dict(
                {
                    "task": "Template-Redrhex-Direct-v0",
                    "num_envs": 4,
                    "max_iterations": 8,
                    "device": "cpu",
                    "reward_preset_id": "tweak-run-one",
                    "reward_overrides": {"rew_scale_alive": 0.2},
                    "tweak_source_run_id": "run_one",
                    "tweak_source_label": "Run One",
                    "requester_id": "11111111-1111-4111-8111-111111111111",
                    "requester_label": "Jason",
                    "display_name": "stair warmup",
                }
            )
            with patch.object(registry, "_spawn_shell", return_value=SpawnedProcess(proc=FakeProcess())), patch("threading.Thread") as thread_cls:
                thread_cls.return_value.start = Mock()
                run = registry.start_training(params)

            record = history.get_run(run["id"])
            self.assertEqual(record["params"]["tweak_source_run_id"], "run_one")
            self.assertEqual(record["params"]["tweak_source_label"], "Run One")
            self.assertEqual(record["created_by"], "11111111-1111-4111-8111-111111111111")
            self.assertEqual(record["requester_label"], "Jason")
            self.assertEqual(record["reward_preset_id"], "tweak-run-one")
            self.assertEqual(record["display_name"], "stair warmup")
            self.assertEqual(record["params"]["display_name"], "stair warmup")

    def test_training_writes_valid_run_scoped_physics_profile(self):
        class FakeProcess:
            pid = 12345

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = self.make_paths(root)
            history = HistoryStore(paths)
            registry = ProcessRegistry(paths, history)
            params = TrainingParams.from_dict(
                {
                    "device": "cpu",
                    "physics_preset_id": "bench-measured",
                    "physics_overrides": {
                        "simulation_physics.mass.scale": 1.04,
                        "simulation_physics.ground.static_friction": 1.3,
                    },
                }
            )
            with patch.object(registry, "_spawn_shell", return_value=SpawnedProcess(proc=FakeProcess())), patch("threading.Thread") as thread_cls:
                thread_cls.return_value.start = Mock()
                run = registry.start_training(params)

            record = history.get_run(run["id"])
            profile_path = Path(record["physics_profile_file"])
            profile = json.loads(profile_path.read_text(encoding="utf-8"))
            self.assertTrue(profile_path.is_relative_to(paths.panel_log_root))
            self.assertEqual(profile["simulation_physics"]["mass"]["scale"], 1.04)
            self.assertEqual(profile["simulation_physics"]["ground"]["dynamic_friction"], 1.0)
            self.assertEqual(record["physics_preset_id"], "bench-measured")
            self.assertEqual(record["params"]["physics_overrides"], params.physics_overrides)
            self.assertIn("--physics-profile", record["command"])
            self.assertIn(str(profile_path), record["command"])

    def test_replay_prefers_the_profile_snapshotted_by_train(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = self.make_paths(root)
            history = HistoryStore(paths)
            run_dir = root / "logs" / "rsl_rl" / "redrhex_wheg" / "run_one"
            saved_profile = run_dir / "params" / "physics_profile.json"
            saved_profile.parent.mkdir(parents=True)
            saved_profile.write_text('{"schema_version": 1}\n', encoding="utf-8")
            history.add_run(
                {
                    "id": "run_one",
                    "source": "training_panel",
                    "status": "completed",
                    "created_at": "2026-08-13T10:00:00",
                    "log_dir": str(run_dir),
                    "params": {
                        "physics_preset_id": "old-preset",
                        "physics_overrides": {"simulation_physics.mass.scale": 2.0},
                    },
                }
            )
            registry = ProcessRegistry(paths, history)

            self.assertEqual(
                registry._write_process_physics_profile("play_test", "run_one"),
                str(saved_profile),
            )

    def test_queue_training_starts_immediately_when_gpu_is_free(self):
        class FakeProcess:
            pid = 12345

            def poll(self):
                return None

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = self.make_paths(root)
            history = HistoryStore(paths)
            registry = ProcessRegistry(paths, history)
            params = TrainingParams.from_dict({"task": "Template-Redrhex-Direct-v0", "num_envs": 4, "max_iterations": 8})
            with patch.object(registry, "_spawn_shell", return_value=SpawnedProcess(proc=FakeProcess())), patch("threading.Thread") as thread_cls:
                thread_cls.return_value.start = Mock()
                run = registry.queue_training(params)

            record = history.get_run(run["id"])
            self.assertEqual(record["status"], "running")
            self.assertEqual([process["kind"] for process in registry.running_isaac_processes()], ["training"])

    def test_queue_training_rejects_direct_explicit_params_before_writing_history(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = self.make_paths(root)
            history = HistoryStore(paths)
            registry = ProcessRegistry(paths, history)

            with self.assertRaisesRegex(ValueError, "quarantined.*120 Hz"):
                registry.queue_training(TrainingParams(spring_backend="explicit"))

            self.assertEqual(history.list_runs(), [])

    def test_cuda_preflight_blocks_training_before_history_record(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = self.make_paths(root)
            history = HistoryStore(paths)
            registry = ProcessRegistry(paths, history, cuda_preflight=True)
            params = TrainingParams.from_dict({"task": "Template-Redrhex-Direct-v0", "num_envs": 4, "max_iterations": 8, "device": "cuda:0"})
            with (
                patch.object(ProcessRegistry, "_loaded_nvidia_kernel_version", return_value="580.126.09"),
                patch.object(ProcessRegistry, "_nvidia_userspace_version", return_value="580.159.03"),
            ):
                with self.assertRaises(CudaPreflightError):
                    registry.queue_training(params)

            self.assertEqual(history.list_runs(), [])

    def test_cuda_preflight_does_not_check_cpu_training(self):
        class FakeProcess:
            pid = 12345

            def poll(self):
                return None

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = self.make_paths(root)
            history = HistoryStore(paths)
            registry = ProcessRegistry(paths, history, cuda_preflight=True)
            params = TrainingParams.from_dict({"task": "Template-Redrhex-Direct-v0", "num_envs": 4, "max_iterations": 8, "device": "cpu"})
            with (
                patch.object(ProcessRegistry, "_loaded_nvidia_kernel_version", side_effect=AssertionError("should not check cuda")),
                patch.object(registry, "_spawn_shell", return_value=SpawnedProcess(proc=FakeProcess())),
                patch("threading.Thread") as thread_cls,
            ):
                thread_cls.return_value.start = Mock()
                run = registry.queue_training(params)

            self.assertEqual(history.get_run(run["id"])["status"], "running")

    def test_queue_training_waits_behind_active_gpu_process(self):
        class FakeProcess:
            pid = 12345

            def poll(self):
                return None

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = self.make_paths(root)
            history = HistoryStore(paths)
            registry = ProcessRegistry(paths, history)
            params = TrainingParams.from_dict({"task": "Template-Redrhex-Direct-v0", "num_envs": 4, "max_iterations": 8})
            with patch.object(registry, "_spawn_shell", return_value=SpawnedProcess(proc=FakeProcess())), patch("threading.Thread") as thread_cls:
                thread_cls.return_value.start = Mock()
                active = registry.start_training(params)
                queued = registry.queue_training(params)

            self.assertEqual(history.get_run(active["id"])["status"], "running")
            self.assertEqual(history.get_run(queued["id"])["status"], "queued")
            self.assertIsNone(history.get_run(queued["id"]).get("pid"))

    def test_gpu_host_lease_serializes_two_registry_instances(self):
        class RunningProcess:
            pid = 12345

            def poll(self):
                return None

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = self.make_paths(root)
            history = HistoryStore(paths)
            first_registry = ProcessRegistry(paths, history)
            second_registry = ProcessRegistry(paths, history)
            with (
                patch.object(
                    first_registry,
                    "_spawn_shell",
                    return_value=SpawnedProcess(proc=RunningProcess()),
                ),
                patch.object(first_registry, "_raise_if_immediate_exit"),
                patch("tools.training_panel.training_panel.processes.threading.Thread") as thread_cls,
            ):
                thread_cls.return_value.start = Mock()
                first = first_registry.start_play("run_one", "/tmp/model.pt", device="cpu")

            with patch.object(second_registry, "_spawn_shell") as second_spawn:
                with self.assertRaises(GpuHostLeaseBusy) as caught:
                    second_registry.start_onnx_export("run_one", "/tmp/model.pt", device="cpu")

            self.assertEqual(caught.exception.payload["code"], "gpu_host_lease_busy")
            self.assertEqual(caught.exception.payload["lease"]["process_id"], first["id"])
            self.assertEqual(caught.exception.payload["lease"]["kind"], "play")
            second_spawn.assert_not_called()

            params = TrainingParams.from_dict(
                {"task": "Template-Redrhex-Direct-v0", "num_envs": 4, "max_iterations": 8}
            )
            with (
                patch.object(second_registry, "_spawn_shell") as queued_spawn,
                patch.object(second_registry, "_schedule_queued_training_start_locked") as schedule,
            ):
                queued = second_registry.queue_training(params)
            self.assertEqual(history.get_run(queued["id"])["status"], "queued")
            queued_spawn.assert_not_called()
            schedule.assert_called_once_with(1.0)
            first_registry._release_gpu_lease(first["id"])

    def test_gpu_host_lease_acquisition_is_atomic_across_registry_threads(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = self.make_paths(root)
            registries = [
                ProcessRegistry(paths, HistoryStore(paths)),
                ProcessRegistry(paths, HistoryStore(paths)),
            ]
            start_barrier = threading.Barrier(3)
            attempted_barrier = threading.Barrier(3)
            release_winner = threading.Event()
            outcomes: list[tuple[str, int]] = []

            def attempt(index: int) -> None:
                acquired = False
                start_barrier.wait()
                try:
                    registries[index]._acquire_gpu_lease(f"process_{index}", "training")
                    acquired = True
                    outcomes.append(("acquired", index))
                except GpuHostLeaseBusy:
                    outcomes.append(("busy", index))
                attempted_barrier.wait()
                if acquired:
                    release_winner.wait(timeout=2)
                    registries[index]._release_gpu_lease(f"process_{index}")

            threads = [threading.Thread(target=attempt, args=(index,)) for index in range(2)]
            for thread in threads:
                thread.start()
            start_barrier.wait()
            attempted_barrier.wait()
            self.assertEqual(sorted(outcome for outcome, _index in outcomes), ["acquired", "busy"])
            release_winner.set()
            for thread in threads:
                thread.join(timeout=2)
                self.assertFalse(thread.is_alive())

    def test_gpu_host_lease_is_released_when_spawn_fails(self):
        class RunningProcess:
            pid = 12346

            def poll(self):
                return None

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = self.make_paths(root)
            history = HistoryStore(paths)
            failed_registry = ProcessRegistry(paths, history)
            next_registry = ProcessRegistry(paths, history)
            with patch.object(failed_registry, "_spawn_shell", side_effect=OSError("spawn failed")):
                with self.assertRaisesRegex(OSError, "spawn failed"):
                    failed_registry.start_play("run_one", "/tmp/model.pt", device="cpu")

            with (
                patch.object(
                    next_registry,
                    "_spawn_shell",
                    return_value=SpawnedProcess(proc=RunningProcess()),
                ),
                patch.object(next_registry, "_raise_if_immediate_exit"),
                patch("tools.training_panel.training_panel.processes.threading.Thread") as thread_cls,
            ):
                thread_cls.return_value.start = Mock()
                started = next_registry.start_play("run_one", "/tmp/model.pt", device="cpu")

            self.assertIn(started["id"], next_registry._gpu_lease_fds)
            next_registry._release_gpu_lease(started["id"])

    def test_gpu_host_lease_parent_fd_is_closed_when_monitor_start_fails(self):
        class CompletedProcess:
            pid = 12348

            def poll(self):
                return 0

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = self.make_paths(root)
            history = HistoryStore(paths)
            history.add_run({"id": "run_one", "source": "training_panel", "status": "completed"})
            registry = ProcessRegistry(paths, history)
            with (
                patch.object(
                    registry,
                    "_spawn_shell",
                    return_value=SpawnedProcess(proc=CompletedProcess()),
                ),
                patch(
                    "tools.training_panel.training_panel.processes.threading.Thread.start",
                    side_effect=RuntimeError("monitor failed"),
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "monitor failed"):
                    registry.start_onnx_export("run_one", "/tmp/model.pt", device="cpu")

            self.assertEqual(registry._gpu_lease_fds, {})
            other_registry = ProcessRegistry(paths, history)
            other_registry._acquire_gpu_lease("next_process", "training")
            other_registry._release_gpu_lease("next_process")

    def test_gpu_host_lease_fd_is_inherited_and_released_after_monitor_exit(self):
        class CompletedProcess:
            pid = 12347

            def poll(self):
                return 0

            def wait(self):
                return 0

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = self.make_paths(root)
            history = HistoryStore(paths)
            registry = ProcessRegistry(paths, history, isaac_settle_seconds=0)
            proc = CompletedProcess()
            with (
                patch.object(
                    registry,
                    "_spawn_shell",
                    return_value=SpawnedProcess(proc=proc),
                ) as spawn,
                patch.object(registry, "_raise_if_immediate_exit"),
                patch.object(registry, "start_next_queued_training"),
                patch("tools.training_panel.training_panel.processes.threading.Thread") as thread_cls,
            ):
                thread_cls.return_value.start = Mock()
                result = registry.start_play("run_one", "/tmp/model.pt", device="cpu")

            inherited_fds = spawn.call_args.kwargs["inherited_fds"]
            self.assertEqual(len(inherited_fds), 1)
            self.assertEqual(inherited_fds[0], registry._gpu_lease_fds[result["id"]])
            registry._monitor_play(result["id"], proc)
            self.assertNotIn(result["id"], registry._gpu_lease_fds)

            other_registry = ProcessRegistry(paths, history)
            lease_fd = other_registry._acquire_gpu_lease("next_process", "training")
            self.assertGreaterEqual(lease_fd, 0)
            other_registry._release_gpu_lease("next_process")

    def test_queue_training_waits_during_isaac_settle_window(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = self.make_paths(root)
            history = HistoryStore(paths)
            registry = ProcessRegistry(paths, history, isaac_settle_seconds=10)
            params = TrainingParams.from_dict({"task": "Template-Redrhex-Direct-v0", "num_envs": 4, "max_iterations": 8})
            registry._last_isaac_exit_at = time.time()

            with patch.object(registry, "_schedule_queued_training_start_locked") as schedule:
                run = registry.queue_training(params)

            self.assertEqual(run["status"], "queued")
            self.assertIsNone(history.get_run(run["id"]).get("pid"))
            schedule.assert_called_once()

    def test_training_records_preserve_launch_folder_and_client_request_id(self):
        class FakeProcess:
            pid = 12345

            def poll(self):
                return None

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = self.make_paths(root)
            history = HistoryStore(paths)
            registry = ProcessRegistry(paths, history)
            params = TrainingParams.from_dict(
                {
                    "task": "Template-Redrhex-Direct-v0",
                    "num_envs": 4,
                    "max_iterations": 8,
                    "folder": "tests",
                    "client_request_id": "child-123",
                    "display_name": "Launch A",
                }
            )
            with patch.object(registry, "_spawn_shell", return_value=SpawnedProcess(proc=FakeProcess())), patch("threading.Thread") as thread_cls:
                thread_cls.return_value.start = Mock()
                run = registry.start_training(params)

            record = history.get_run(run["id"])
            self.assertEqual(record["display_name"], "Launch A")
            self.assertEqual(record["folder"], "tests")
            self.assertEqual(record["client_request_id"], "child-123")
            self.assertEqual(record["params"]["folder"], "tests")
            self.assertEqual(record["params"]["client_request_id"], "child-123")

    def test_client_request_id_is_atomic_across_registry_instances(self):
        class RunningProcess:
            pid = 12345

            def poll(self):
                return None

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = self.make_paths(root)
            registries = [
                ProcessRegistry(paths, HistoryStore(paths)),
                ProcessRegistry(paths, HistoryStore(paths)),
            ]
            params = [
                TrainingParams.from_dict(
                    {
                        "task": "Template-Redrhex-Direct-v0",
                        "num_envs": 4,
                        "max_iterations": 8,
                        "device": "cpu",
                        "seed": 42,
                        "client_request_id": "shared-launch-123",
                    }
                )
                for _ in registries
            ]
            barrier = threading.Barrier(3)
            results: list[dict] = []
            errors: list[BaseException] = []

            def launch(index: int) -> None:
                barrier.wait()
                try:
                    results.append(registries[index].start_training(params[index]))
                except BaseException as exc:  # pragma: no cover - asserted below
                    errors.append(exc)

            with (
                patch.object(
                    registries[0],
                    "_spawn_shell",
                    return_value=SpawnedProcess(proc=RunningProcess()),
                ) as first_spawn,
                patch.object(
                    registries[1],
                    "_spawn_shell",
                    return_value=SpawnedProcess(proc=RunningProcess()),
                ) as second_spawn,
                patch.object(registries[0], "_start_gpu_monitor"),
                patch.object(registries[1], "_start_gpu_monitor"),
            ):
                threads = [threading.Thread(target=launch, args=(index,)) for index in range(2)]
                for thread in threads:
                    thread.start()
                barrier.wait()
                for thread in threads:
                    thread.join(timeout=3)
                    self.assertFalse(thread.is_alive())

            self.assertEqual(errors, [])
            self.assertEqual(len(results), 2)
            self.assertEqual(results[0]["id"], results[1]["id"])
            self.assertEqual(first_spawn.call_count + second_spawn.call_count, 1)
            matching = [
                run
                for run in HistoryStore(paths).list_runs()
                if run.get("client_request_id") == "shared-launch-123"
            ]
            self.assertEqual(len(matching), 1)
            for registry in registries:
                registry._release_gpu_lease(results[0]["id"])

    def test_client_request_id_mismatch_fails_closed_and_completed_run_is_reused(self):
        class RunningProcess:
            pid = 12346

            def poll(self):
                return None

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = self.make_paths(root)
            first_registry = ProcessRegistry(paths, HistoryStore(paths))
            request = {
                "task": "Template-Redrhex-Direct-v0",
                "num_envs": 4,
                "max_iterations": 8,
                "device": "cpu",
                "seed": 42,
                "client_request_id": "stable-launch-123",
            }
            with (
                patch.object(
                    first_registry,
                    "_spawn_shell",
                    return_value=SpawnedProcess(proc=RunningProcess()),
                ),
                patch.object(first_registry, "_start_gpu_monitor"),
            ):
                first = first_registry.start_training(TrainingParams.from_dict(request))

            first_registry.history.update_run(first["id"], status="completed")
            first_registry._release_gpu_lease(first["id"])
            second_registry = ProcessRegistry(paths, HistoryStore(paths))
            with patch.object(second_registry, "_spawn_shell") as duplicate_spawn:
                repeated = second_registry.start_training(TrainingParams.from_dict(request))
            self.assertEqual(repeated["id"], first["id"])
            self.assertEqual(repeated["status"], "completed")
            duplicate_spawn.assert_not_called()

            changed = {**request, "max_iterations": 9}
            with patch.object(second_registry, "_spawn_shell") as mismatch_spawn:
                with self.assertRaises(ProcessStartError) as caught:
                    second_registry.start_training(TrainingParams.from_dict(changed))
            self.assertEqual(caught.exception.payload["code"], "client_request_id_conflict")
            self.assertEqual(caught.exception.payload["existing_run_id"], first["id"])
            mismatch_spawn.assert_not_called()

    def test_start_deploy_validation_records_process_metadata(self):
        class FakeProcess:
            pid = 12345

            def poll(self):
                return None

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = self.make_paths(root)
            history = HistoryStore(paths)
            run_dir = root / "logs" / "rsl_rl" / "redrhex_wheg" / "run_one"
            (run_dir / "exported").mkdir(parents=True)
            checkpoint = run_dir / "model_10.pt"
            checkpoint.write_text("model", encoding="utf-8")
            onnx = run_dir / "exported" / "policy.onnx"
            onnx.write_text("onnx", encoding="utf-8")
            history.add_run(
                {
                    "id": "run_one",
                    "source": "training_panel",
                    "status": "completed",
                    "created_at": "2026-06-01T10:00:00",
                    "log_dir": str(run_dir),
                    "latest_checkpoint": str(checkpoint),
                    "onnx_path": str(onnx),
                }
            )
            registry = ProcessRegistry(paths, history)
            with patch.object(registry, "_spawn_shell", return_value=SpawnedProcess(proc=FakeProcess())), patch("threading.Thread") as thread_cls:
                thread_cls.return_value.start = Mock()
                result = registry.start_deploy_validation("run_one", include_mujoco=False)

            self.assertTrue(result["id"].startswith("deploy_"))
            record = history.get_run("run_one")
            self.assertEqual(record["deploy_status"], "running")
            self.assertEqual(record["deploy_process_id"], result["id"])
            self.assertIn(sys.executable, record["deploy_command"])
            self.assertIn("tools.training_panel.deploy_pipeline", record["deploy_command"])
            self.assertNotIn("conda activate", record["deploy_command"])
            self.assertNotIn("env_isaaclab_bin", record["deploy_command"])
            self.assertFalse(record["deploy_options"]["include_mujoco"])

    def test_start_mujoco_playback_records_process_metadata(self):
        class FakeProcess:
            pid = 12345

            def poll(self):
                return None

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = self.make_paths(root)
            history = HistoryStore(paths)
            run_dir = root / "logs" / "rsl_rl" / "redrhex_wheg" / "run_one"
            (run_dir / "exported").mkdir(parents=True)
            onnx = run_dir / "exported" / "policy.onnx"
            onnx.write_text("onnx", encoding="utf-8")
            history.add_run(
                {
                    "id": "run_one",
                    "source": "training_panel",
                    "status": "completed",
                    "created_at": "2026-06-01T10:00:00",
                    "log_dir": str(run_dir),
                    "onnx_path": str(onnx),
                }
            )
            registry = ProcessRegistry(paths, history)
            with patch.object(registry, "_spawn_shell", return_value=SpawnedProcess(proc=FakeProcess())), patch("threading.Thread") as thread_cls:
                thread_cls.return_value.start = Mock()
                result = registry.start_mujoco_playback("run_one", mode="record", scenario="forward_mid")

            self.assertTrue(result["id"].startswith("mujoco_record_"))
            record = history.get_run("run_one")
            self.assertEqual(record["mujoco_playback_status"], "running")
            self.assertEqual(record["mujoco_playback_mode"], "record")
            self.assertEqual(record["mujoco_playback_scenario"], "forward_mid")
            self.assertEqual(record["mujoco_process_id"], result["id"])
            self.assertIn("tools.training_panel.mujoco_playback", record["mujoco_command"])
            self.assertIn("--mode record", record["mujoco_command"])
            self.assertEqual(registry.running_isaac_processes(), [])

    def test_monitor_mujoco_records_video_report(self):
        class DoneProcess:
            def wait(self):
                return 0

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = self.make_paths(root)
            history = HistoryStore(paths)
            run_dir = root / "logs" / "rsl_rl" / "redrhex_wheg" / "run_one"
            artifact_dir = run_dir / "deploy" / "mujoco_playback_mujoco_record_test"
            artifact_dir.mkdir(parents=True)
            video = artifact_dir / "mujoco_forward_mid.mp4"
            video.write_bytes(b"mp4")
            report = artifact_dir / "mujoco_playback_report.json"
            report.write_text(
                json.dumps({"status": "completed", "summary": "ok", "video_path": str(video)}),
                encoding="utf-8",
            )
            history.add_run(
                {
                    "id": "run_one",
                    "source": "training_panel",
                    "status": "completed",
                    "created_at": "2026-06-01T10:00:00",
                    "log_dir": str(run_dir),
                }
            )
            registry = ProcessRegistry(paths, history)
            registry._monitor_mujoco("run_one", "mujoco_record_test", DoneProcess())

            record = history.get_run("run_one")
            self.assertEqual(record["mujoco_playback_status"], "completed")
            self.assertEqual(record["latest_mujoco_video"], str(video))
            self.assertEqual(record["mujoco_video_report"], str(report))

    def test_start_next_queued_training_respects_isaac_settle_window(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = self.make_paths(root)
            history = HistoryStore(paths)
            registry = ProcessRegistry(paths, history, isaac_settle_seconds=10)
            params = TrainingParams.from_dict({"task": "Template-Redrhex-Direct-v0", "num_envs": 4, "max_iterations": 8})
            queued = registry._create_queued_training_run(params)
            registry._last_isaac_exit_at = time.time()

            with patch.object(registry, "_schedule_queued_training_start_locked") as schedule:
                started = registry.start_next_queued_training()

            self.assertIsNone(started)
            self.assertEqual(history.get_run(queued["id"])["status"], "queued")
            schedule.assert_called_once()

    def test_start_next_queued_training_fails_legacy_run_before_native_reinterpretation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = self.make_paths(root)
            history = HistoryStore(paths)
            history.add_run(
                {
                    "id": "queued_explicit",
                    "source": "training_panel",
                    "status": "queued",
                    "created_at": "2026-08-14T10:00:00",
                    "queued_at": "2026-08-14T10:00:00",
                    "params": {"task": "Template-Redrhex-Direct-v0"},
                }
            )
            registry = ProcessRegistry(paths, history)

            with patch.object(registry, "_spawn_shell") as spawn:
                started = registry.start_next_queued_training()

            self.assertIsNone(started)
            record = history.get_run("queued_explicit")
            self.assertEqual(record["status"], "failed")
            self.assertRegex(record["queue_error"], "quarantined.*120 Hz")
            spawn.assert_not_called()

    def test_cancel_queued_training(self):
        class FakeProcess:
            pid = 12345

            def poll(self):
                return None

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = self.make_paths(root)
            history = HistoryStore(paths)
            registry = ProcessRegistry(paths, history)
            params = TrainingParams.from_dict({"task": "Template-Redrhex-Direct-v0", "num_envs": 4, "max_iterations": 8})
            with patch.object(registry, "_spawn_shell", return_value=SpawnedProcess(proc=FakeProcess())), patch("threading.Thread") as thread_cls:
                thread_cls.return_value.start = Mock()
                registry.start_training(params)
                queued = registry.queue_training(params)

            self.assertTrue(registry.cancel_queued_training(queued["id"]))
            self.assertEqual(history.get_run(queued["id"])["status"], "cancelled")

    def test_start_next_queued_training_when_gpu_becomes_free(self):
        class MutableProcess:
            pid = 12345
            returncode = None

            def poll(self):
                return self.returncode

        class RunningProcess:
            pid = 12346

            def poll(self):
                return None

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = self.make_paths(root)
            history = HistoryStore(paths)
            registry = ProcessRegistry(paths, history)
            params = TrainingParams.from_dict({"task": "Template-Redrhex-Direct-v0", "num_envs": 4, "max_iterations": 8})
            active_proc = MutableProcess()
            with patch.object(registry, "_spawn_shell", return_value=SpawnedProcess(proc=active_proc)), patch("threading.Thread") as thread_cls:
                thread_cls.return_value.start = Mock()
                registry.start_training(params)
                queued = registry.queue_training(params)
            active_proc.returncode = 0
            with patch.object(registry, "_spawn_shell", return_value=SpawnedProcess(proc=RunningProcess())), patch("threading.Thread") as thread_cls:
                thread_cls.return_value.start = Mock()
                started = registry.start_next_queued_training()

            self.assertEqual(started["id"], queued["id"])
            self.assertEqual(history.get_run(queued["id"])["status"], "running")
            self.assertEqual(history.get_run(queued["id"])["pid"], 12346)

    def test_play_process_debug_streams_log_tail(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = self.make_paths(root)
            history = HistoryStore(paths)
            log_dir = paths.rsl_rl_log_root / "run_one"
            params_dir = log_dir / "params"
            params_dir.mkdir(parents=True)
            checkpoint = log_dir / "model_10.pt"
            checkpoint.write_text("x", encoding="utf-8")
            (params_dir / "env.yaml").write_text(
                "terrain:\n  terrain_type: plane\nterrain_curriculum_enable: false\n",
                encoding="utf-8",
            )
            history.add_run(
                {
                    "id": "run_one",
                    "source": "training_panel",
                    "status": "completed",
                    "log_dir": str(log_dir),
                    "params": {
                        "task": "Template-Redrhex-ForwardFast-Direct-v0",
                        "spring_backend": "native",
                        "seed": 43,
                    },
                }
            )
            registry = ProcessRegistry(paths, history)
            result = registry.start_play("run_one", str(checkpoint), device="cpu")
            try:
                debug = registry.get_process_debug(result["id"])
                self.assertIsNotNone(debug)
                self.assertEqual(debug["kind"], "play")
                self.assertIsNone(debug["returncode"])
                self.assertEqual(debug["source_run_id"], "run_one")
                self.assertIn("scripts/rsl_rl/play.py", debug["command"])
                self.assertIn("--spring-backend native", debug["command"])
                self.assertIn("--task Template-Redrhex-ForwardFast-Direct-v0", debug["command"])
                self.assertIn("--seed 43", debug["command"])
                self.assertIn("--initial_command forward", debug["command"])
                self.assertIn("--terrain_override_file", debug["command"])
                self.assertIn("--camera_follow_robot", debug["command"])
                self.assertIn("--camera_eye -3.0 -2.4 1.6", debug["command"])
                override_files = list(paths.process_override_dir.glob("*_terrain.json"))
                self.assertEqual(len(override_files), 1)
                self.assertIn("terrain.terrain_type", override_files[0].read_text(encoding="utf-8"))
                self.assertIn("fake isaaclab", debug["log_tail"])
            finally:
                proc = registry._processes.get(result["id"])
                registry.stop(result["id"])
                if proc:
                    proc.wait(timeout=8)
                time.sleep(0.1)

    def test_terrain_override_file_is_written_and_cleared(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = self.make_paths(root)
            registry = ProcessRegistry(paths, HistoryStore(paths))
            registry._write_terrain_override({"terrain.terrain_type": "plane", "terrain_curriculum_enable": False})
            self.assertTrue(paths.terrain_override_file.exists())
            text = paths.terrain_override_file.read_text(encoding="utf-8")
            self.assertIn("terrain.terrain_type", text)
            registry._write_terrain_override({})
            self.assertFalse(paths.terrain_override_file.exists())

    def test_video_recording_process_uses_headless_video_flags(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = self.make_paths(root)
            history = HistoryStore(paths)
            log_dir = paths.rsl_rl_log_root / "run_one"
            params_dir = log_dir / "params"
            params_dir.mkdir(parents=True)
            checkpoint = log_dir / "model_10.pt"
            checkpoint.write_text("x", encoding="utf-8")
            (params_dir / "env.yaml").write_text(
                "terrain:\n  terrain_type: plane\nterrain_curriculum_enable: false\n",
                encoding="utf-8",
            )
            history.add_run(
                {
                    "id": "run_one",
                    "source": "training_panel",
                    "status": "completed",
                    "created_at": "2026-05-15T11:00:00",
                    "log_dir": str(log_dir),
                    "params": {
                        "task": "Template-Redrhex-ForwardFast-Direct-v0",
                        "spring_backend": "native",
                        "seed": 43,
                    },
                }
            )
            registry = ProcessRegistry(paths, history)
            result = registry.start_video_recording(
                "run_one",
                str(checkpoint),
                device="cpu",
                video_params=VideoParams.from_preset("high"),
            )
            try:
                debug = registry.get_process_debug(result["id"])
                self.assertIsNotNone(debug)
                self.assertEqual(debug["kind"], "video")
                self.assertEqual(debug["source_run_id"], "run_one")
                self.assertIn("--headless", debug["command"])
                self.assertIn("--spring-backend native", debug["command"])
                self.assertIn("--video", debug["command"])
                self.assertIn("--video_length 1200", debug["command"])
                self.assertIn("--video_width 1920", debug["command"])
                self.assertIn("--video_height 1080", debug["command"])
                self.assertIn("--video_fps 30", debug["command"])
                self.assertIn("--rendering_mode quality", debug["command"])
                self.assertIn("--task Template-Redrhex-ForwardFast-Direct-v0", debug["command"])
                self.assertIn("--seed 43", debug["command"])
                self.assertIn("--initial_command forward", debug["command"])
                self.assertIn("--terrain_override_file", debug["command"])
                self.assertIn("--camera_follow_robot", debug["command"])
                self.assertIn("--camera_lookat 0.45 0.0 0.35", debug["command"])
                override_files = list(paths.process_override_dir.glob("*_terrain.json"))
                self.assertEqual(len(override_files), 1)
                self.assertIn("terrain.terrain_type", override_files[0].read_text(encoding="utf-8"))
                self.assertIn("attach_command", debug)
                run = history.get_run("run_one")
                self.assertEqual(run["video_status"], "recording")
                self.assertEqual(run["video_process_id"], result["id"])
                self.assertEqual(run["video_preset"], "high")
            finally:
                proc = registry._processes.get(result["id"])
                registry.stop(result["id"])
                if proc:
                    proc.wait(timeout=8)
                time.sleep(0.1)

    def test_checkpoint_zero_video_is_tagged_for_history(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = self.make_paths(root)
            registry = ProcessRegistry(paths, HistoryStore(paths))
            video = root / "rl-video-step-0.mp4"
            video.write_bytes(b"video")

            tagged = registry._tag_video_with_checkpoint(
                video,
                "video_fixture",
                {"video_checkpoint_iteration": 0},
            )

            self.assertEqual(Path(tagged).name, "model_0_video_fixture.mp4")
            self.assertTrue(Path(tagged).is_file())
            self.assertFalse(video.exists())

    def test_manual_forward_fast_video_recording_starts_forward(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = self.make_paths(root)
            history = HistoryStore(paths)
            history.add_run(
                {
                    "id": "forward_fast",
                    "source": "training_panel",
                    "status": "completed",
                    "params": {
                        "task": "Template-Redrhex-ForwardFast-Direct-v0",
                        "spring_backend": "native",
                    },
                }
            )
            registry = ProcessRegistry(paths, history)

            with patch.object(
                registry, "_spawn_shell", return_value=SpawnedProcess(proc=Mock(pid=123))
            ) as spawn, patch("tools.training_panel.training_panel.processes.threading.Thread") as thread_cls:
                thread_cls.return_value.start = Mock()
                registry.start_video_recording("forward_fast", "/tmp/model_7.pt", device="cpu")

            self.assertIn("--initial_command forward", spawn.call_args.args[1])

    def test_forward_fast_process_log_resolves_logged_root_for_live_and_completed_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = self.make_paths(root)
            history = HistoryStore(paths)
            experiment_root = root / "logs" / "rsl_rl" / "redrhex_forward_fast"
            log_dir = experiment_root / "2026-06-01_12-00-00_forward_fast_reform_v1"
            log_dir.mkdir(parents=True)
            process_log = paths.process_log_dir / "forward_fast.log"
            process_log.parent.mkdir(parents=True, exist_ok=True)
            process_log.write_text(
                f"[INFO] Logging experiment in directory: {experiment_root}\n"
                "Exact experiment name requested from command line: 2026-06-01_12-00-00\n",
                encoding="utf-8",
            )
            run = {
                "id": "forward_fast",
                "source": "training_panel",
                "status": "running",
                "process_log": str(process_log),
            }
            history.add_run(run)
            registry = ProcessRegistry(paths, history)

            self.assertEqual(registry._log_dir_from_process_log("forward_fast"), str(log_dir))
            self.assertEqual(registry._completed_log_for_run(run), str(log_dir))

    def test_explicit_forward_fast_root_without_match_does_not_fallback_to_legacy_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = self.make_paths(root)
            history = HistoryStore(paths)
            timestamp = "2026-06-01_12-00-00"
            experiment_root = root / "logs" / "rsl_rl" / "redrhex_forward_fast"
            (paths.rsl_rl_log_root / f"{timestamp}_wheg_locomotion").mkdir(parents=True)
            process_log = paths.process_log_dir / "forward_fast.log"
            process_log.parent.mkdir(parents=True, exist_ok=True)
            process_log.write_text(
                f"[INFO] Logging experiment in directory: {experiment_root}\n"
                f"Exact experiment name requested from command line: {timestamp}\n",
                encoding="utf-8",
            )
            run = {"id": "forward_fast", "process_log": str(process_log)}
            history.add_run(run)
            registry = ProcessRegistry(paths, history)

            self.assertIsNone(registry._log_dir_from_process_log("forward_fast"))
            self.assertIsNone(registry._completed_log_for_run(run))

    def test_explicit_experiment_root_with_spaces_is_resolved(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = self.make_paths(root)
            history = HistoryStore(paths)
            timestamp = "2026-06-01_12-00-00"
            experiment_root = root / "logs" / "rsl_rl" / "Forward Fast Runs"
            log_dir = experiment_root / f"{timestamp}_forward_fast_reform_v1"
            log_dir.mkdir(parents=True)
            process_log = paths.process_log_dir / "forward_fast.log"
            process_log.parent.mkdir(parents=True, exist_ok=True)
            process_log.write_text(
                f"[INFO] Logging experiment in directory: {experiment_root}\n"
                f"Exact experiment name requested from command line: {timestamp}\n",
                encoding="utf-8",
            )
            history.add_run({"id": "forward_fast", "process_log": str(process_log)})
            registry = ProcessRegistry(paths, history)

            self.assertEqual(registry._log_dir_from_process_log("forward_fast"), str(log_dir))

    def test_explicit_experiment_root_rejects_symlinked_candidate_outside_authority(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = self.make_paths(root)
            history = HistoryStore(paths)
            timestamp = "2026-06-01_12-00-00"
            experiment_root = root / "logs" / "rsl_rl" / "redrhex_forward_fast"
            outside_log = root / "outside" / f"{timestamp}_forward_fast_reform_v1"
            outside_log.mkdir(parents=True)
            experiment_root.mkdir(parents=True)
            (experiment_root / outside_log.name).symlink_to(outside_log, target_is_directory=True)
            process_log = paths.process_log_dir / "forward_fast.log"
            process_log.parent.mkdir(parents=True, exist_ok=True)
            process_log.write_text(
                f"[INFO] Logging experiment in directory: {experiment_root}\n"
                f"Exact experiment name requested from command line: {timestamp}\n",
                encoding="utf-8",
            )
            history.add_run({"id": "forward_fast", "process_log": str(process_log)})
            registry = ProcessRegistry(paths, history)

            self.assertIsNone(registry._log_dir_from_process_log("forward_fast"))

    def test_out_of_tree_experiment_root_is_not_used_or_legacy_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = self.make_paths(root)
            history = HistoryStore(paths)
            timestamp = "2026-06-01_12-00-00"
            experiment_root = root / "outside experiment root"
            (experiment_root / f"{timestamp}_forward_fast_reform_v1").mkdir(parents=True)
            (paths.rsl_rl_log_root / f"{timestamp}_wheg_locomotion").mkdir(parents=True)
            process_log = paths.process_log_dir / "forward_fast.log"
            process_log.parent.mkdir(parents=True, exist_ok=True)
            process_log.write_text(
                f"[INFO] Logging experiment in directory: {experiment_root}\n"
                f"Exact experiment name requested from command line: {timestamp}\n",
                encoding="utf-8",
            )
            history.add_run({"id": "forward_fast", "process_log": str(process_log)})
            registry = ProcessRegistry(paths, history)

            self.assertIsNone(registry._log_dir_from_process_log("forward_fast"))

    def test_legacy_exact_name_rejects_symlinked_candidate_outside_log_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = self.make_paths(root)
            history = HistoryStore(paths)
            timestamp = "2026-06-01_12-00-00"
            outside_log = root / "outside" / f"{timestamp}_wheg_locomotion"
            outside_log.mkdir(parents=True)
            paths.rsl_rl_log_root.mkdir(parents=True)
            (paths.rsl_rl_log_root / outside_log.name).symlink_to(
                outside_log, target_is_directory=True
            )
            process_log = paths.process_log_dir / "legacy.log"
            process_log.parent.mkdir(parents=True, exist_ok=True)
            process_log.write_text(
                f"Exact experiment name requested from command line: {timestamp}\n",
                encoding="utf-8",
            )
            history.add_run({"id": "legacy", "process_log": str(process_log)})
            registry = ProcessRegistry(paths, history)

            self.assertIsNone(registry._log_dir_from_process_log("legacy"))

    def test_textual_log_path_rejects_symlinked_directory_outside_log_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = self.make_paths(root)
            history = HistoryStore(paths)
            outside_log = root / "outside" / "2026-06-01_12-00-00_wheg_locomotion"
            outside_log.mkdir(parents=True)
            paths.rsl_rl_log_root.mkdir(parents=True)
            symlinked_log = paths.rsl_rl_log_root / outside_log.name
            symlinked_log.symlink_to(outside_log, target_is_directory=True)
            process_log = paths.process_log_dir / "legacy.log"
            process_log.parent.mkdir(parents=True, exist_ok=True)
            process_log.write_text(
                f"saved {symlinked_log / 'events.out.tfevents.test'}\n",
                encoding="utf-8",
            )
            history.add_run({"id": "legacy", "process_log": str(process_log)})
            registry = ProcessRegistry(paths, history)

            self.assertIsNone(registry._log_dir_from_process_log("legacy"))

    def test_completed_legacy_exact_name_without_match_does_not_time_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = self.make_paths(root)
            history = HistoryStore(paths)
            process_log = paths.process_log_dir / "legacy.log"
            process_log.parent.mkdir(parents=True, exist_ok=True)
            process_log.write_text(
                "Exact experiment name requested from command line: 2026-06-01_12-00-00\n",
                encoding="utf-8",
            )
            neighbor_log = paths.rsl_rl_log_root / "2026-06-01_12-01-00_wheg_locomotion"
            neighbor_log.mkdir(parents=True)
            run = {"process_log": str(process_log), "created_at": "2000-01-01T00:00:00"}
            registry = ProcessRegistry(paths, history)

            self.assertIsNone(registry._completed_log_for_run(run))

    def test_completed_forward_fast_training_starts_native_video_for_source_task(self):
        class CompletedProcess:
            def poll(self):
                return 0

            def wait(self):
                return 0

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = self.make_paths(root)
            history = HistoryStore(paths)
            experiment_root = root / "logs" / "rsl_rl" / "redrhex_forward_fast"
            log_dir = experiment_root / "2026-06-01_12-00-00_forward_fast_reform_v1"
            log_dir.mkdir(parents=True)
            checkpoint = log_dir / "model_7.pt"
            checkpoint.write_text("x", encoding="utf-8")
            process_log = paths.process_log_dir / "forward_fast.log"
            process_log.parent.mkdir(parents=True, exist_ok=True)
            process_log.write_text(
                f"[INFO] Logging experiment in directory: {experiment_root}\n"
                "Exact experiment name requested from command line: 2026-06-01_12-00-00\n",
                encoding="utf-8",
            )
            history.add_run(
                {
                    "id": "forward_fast",
                    "source": "training_panel",
                    "status": "running",
                    "process_log": str(process_log),
                    "params": {
                        "task": "Template-Redrhex-ForwardFast-Direct-v0",
                        "spring_backend": "native",
                        "device": "cpu",
                    },
                }
            )
            registry = ProcessRegistry(paths, history)

            with patch.object(
                registry, "_spawn_shell", return_value=SpawnedProcess(proc=Mock(pid=123))
            ) as spawn, patch.object(registry, "_refresh_tensorboard_summary"), patch(
                "tools.training_panel.training_panel.processes.threading.Thread"
            ) as thread_cls:
                thread_cls.return_value.start = Mock()
                registry._monitor_training("forward_fast", CompletedProcess(), 0)

            command = spawn.call_args.args[1]
            self.assertIn("--task Template-Redrhex-ForwardFast-Direct-v0", command)
            self.assertIn("--spring-backend native", command)
            self.assertIn("--initial_command forward", command)
            self.assertIn(f"--checkpoint {checkpoint}", command)

    def test_play_and_onnx_export_use_source_run_task(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = self.make_paths(root)
            history = HistoryStore(paths)
            history.add_run(
                {
                    "id": "forward_fast",
                    "source": "training_panel",
                    "status": "completed",
                    "params": {"task": "Template-Redrhex-ForwardFast-Direct-v0", "spring_backend": "native"},
                }
            )
            registry = ProcessRegistry(paths, history)

            with patch.object(
                registry, "_spawn_shell", return_value=SpawnedProcess(proc=Mock(pid=123))
            ) as spawn, patch.object(registry, "_raise_if_immediate_exit"), patch(
                "tools.training_panel.training_panel.processes.threading.Thread"
            ) as thread_cls:
                thread_cls.return_value.start = Mock()
                registry.start_play("forward_fast", "/tmp/model_7.pt", device="cpu")
                registry.start_onnx_export("forward_fast", "/tmp/model_7.pt", device="cpu")

            commands = [call.args[1] for call in spawn.call_args_list]
            self.assertTrue(all("--task Template-Redrhex-ForwardFast-Direct-v0" in command for command in commands))
            self.assertTrue(all("--spring-backend native" in command for command in commands))
            self.assertIn("--initial_command forward", commands[0])
            self.assertNotIn("--initial_command", commands[1])

    def test_play_and_onnx_export_default_task_for_legacy_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = self.make_paths(root)
            history = HistoryStore(paths)
            history.add_run({"id": "legacy", "source": "rsl_rl", "status": "completed"})
            registry = ProcessRegistry(paths, history)

            with patch.object(
                registry, "_spawn_shell", return_value=SpawnedProcess(proc=Mock(pid=123))
            ) as spawn, patch.object(registry, "_raise_if_immediate_exit"), patch(
                "tools.training_panel.training_panel.processes.threading.Thread"
            ) as thread_cls:
                thread_cls.return_value.start = Mock()
                registry.start_play("legacy", "/tmp/model_7.pt", device="cpu")
                registry.start_onnx_export("legacy", "/tmp/model_7.pt", device="cpu")

            commands = [call.args[1] for call in spawn.call_args_list]
            self.assertTrue(all(f"--task {DEFAULT_TASK}" in command for command in commands))

    def test_process_terrain_override_falls_back_to_run_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = self.make_paths(root)
            history = HistoryStore(paths)
            history.add_run(
                {
                    "id": "run_one",
                    "source": "training_panel",
                    "status": "completed",
                    "created_at": "2026-05-15T11:00:00",
                    "terrain_overrides": {"terrain.terrain_type": "plane"},
                }
            )
            registry = ProcessRegistry(paths, history)
            path = registry._write_process_terrain_override("play_test", "run_one")
            self.assertIsNotNone(path)
            self.assertIn("run metadata", Path(path).read_text(encoding="utf-8"))
            self.assertIn("terrain.terrain_type", Path(path).read_text(encoding="utf-8"))

    def test_process_terrain_override_prefers_panel_metadata_over_env_yaml(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = self.make_paths(root)
            history = HistoryStore(paths)
            log_dir = paths.rsl_rl_log_root / "run_one"
            params_dir = log_dir / "params"
            params_dir.mkdir(parents=True)
            (params_dir / "env.yaml").write_text(
                "terrain:\n  terrain_type: generator\n  max_init_terrain_level: 3\n",
                encoding="utf-8",
            )
            history.add_run(
                {
                    "id": "run_one",
                    "source": "training_panel",
                    "status": "completed",
                    "created_at": "2026-05-15T11:00:00",
                    "log_dir": str(log_dir),
                    "terrain_overrides": {
                        "terrain.terrain_type": "plane",
                        "terrain.max_init_terrain_level": 0,
                    },
                }
            )
            registry = ProcessRegistry(paths, history)

            path = registry._write_process_terrain_override("play_test", "run_one")
            payload = json.loads(Path(path).read_text(encoding="utf-8"))

            self.assertEqual(payload["source"], "run metadata")
            self.assertEqual(payload["overrides"]["terrain.terrain_type"], "plane")
            self.assertEqual(payload["overrides"]["terrain.max_init_terrain_level"], 0)

    def test_process_terrain_override_absent_for_old_runs_without_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = self.make_paths(root)
            history = HistoryStore(paths)
            history.add_run({"id": "old_run", "source": "rsl_rl", "status": "completed"})
            registry = ProcessRegistry(paths, history)
            self.assertIsNone(registry._write_process_terrain_override("play_test", "old_run"))

    def test_play_resolves_native_backend_from_discovered_run_yaml(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = self.make_paths(root)
            history = HistoryStore(paths)
            log_dir = paths.rsl_rl_log_root / "discovered_run"
            params_dir = log_dir / "params"
            params_dir.mkdir(parents=True)
            checkpoint = log_dir / "model_10.pt"
            checkpoint.write_text("x", encoding="utf-8")
            (params_dir / "torsion_spring.yaml").write_text(
                "spring_backend: native\n  spring_backend: explicit\n",
                encoding="utf-8",
            )
            history.add_run(
                {
                    "id": "discovered_run",
                    "source": "rsl_rl",
                    "status": "completed",
                    "log_dir": str(log_dir),
                }
            )
            registry = ProcessRegistry(paths, history)

            with patch.object(registry, "_spawn_shell", return_value=SpawnedProcess(proc=Mock(pid=123))) as spawn, patch.object(
                registry, "_raise_if_immediate_exit"
            ):
                registry.start_play("discovered_run", str(checkpoint), device="cpu")

            self.assertIn("--spring-backend native", spawn.call_args.args[1])

    def test_play_uses_explicit_backend_for_legacy_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = self.make_paths(root)
            history = HistoryStore(paths)
            history.add_run({"id": "legacy_run", "source": "rsl_rl", "status": "completed"})
            registry = ProcessRegistry(paths, history)

            with patch.object(registry, "_spawn_shell", return_value=SpawnedProcess(proc=Mock(pid=123))) as spawn, patch.object(
                registry, "_raise_if_immediate_exit"
            ):
                registry.start_play("legacy_run", "/tmp/model_10.pt", device="cpu")

            self.assertIn("--spring-backend explicit", spawn.call_args.args[1])

    def test_invalid_stored_backend_raises_before_spawning_play(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = self.make_paths(root)
            history = HistoryStore(paths)
            history.add_run(
                {
                    "id": "invalid_run",
                    "source": "training_panel",
                    "status": "completed",
                    "params": {"spring_backend": "unsupported"},
                }
            )
            registry = ProcessRegistry(paths, history)

            with patch.object(registry, "_spawn_shell") as spawn:
                with self.assertRaisesRegex(ValueError, "spring_backend"):
                    registry.start_play("invalid_run", "/tmp/model_10.pt", device="cpu")

            spawn.assert_not_called()

    def test_invalid_yaml_backend_raises_before_spawning_play(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = self.make_paths(root)
            history = HistoryStore(paths)
            log_dir = paths.rsl_rl_log_root / "invalid_yaml_run"
            params_dir = log_dir / "params"
            params_dir.mkdir(parents=True)
            checkpoint = log_dir / "model_10.pt"
            checkpoint.write_text("x", encoding="utf-8")
            (params_dir / "torsion_spring.yaml").write_text("spring_backend: unsupported\n", encoding="utf-8")
            history.add_run(
                {
                    "id": "invalid_yaml_run",
                    "source": "rsl_rl",
                    "status": "completed",
                    "log_dir": str(log_dir),
                }
            )
            registry = ProcessRegistry(paths, history)

            with patch.object(registry, "_spawn_shell") as spawn:
                with self.assertRaisesRegex(ValueError, "spring_backend"):
                    registry.start_play("invalid_yaml_run", str(checkpoint), device="cpu")

            spawn.assert_not_called()

    def test_unreadable_yaml_backend_raises_with_path_context(self):
        with tempfile.TemporaryDirectory() as tmp:
            yaml_path = Path(tmp) / "params" / "torsion_spring.yaml"
            yaml_path.parent.mkdir()
            yaml_path.write_text("spring_backend: native\n", encoding="utf-8")

            with patch("tools.training_panel.training_panel.commands.Path.read_text", side_effect=PermissionError("denied")):
                with self.assertRaisesRegex(OSError, str(yaml_path)):
                    resolve_spring_backend({"log_dir": str(yaml_path.parent.parent)})

    def test_yaml_backend_path_directory_raises_before_spawning_play(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = self.make_paths(root)
            history = HistoryStore(paths)
            log_dir = paths.rsl_rl_log_root / "directory_yaml_run"
            (log_dir / "params" / "torsion_spring.yaml").mkdir(parents=True)
            checkpoint = log_dir / "model_10.pt"
            checkpoint.write_text("x", encoding="utf-8")
            history.add_run(
                {
                    "id": "directory_yaml_run",
                    "source": "rsl_rl",
                    "status": "completed",
                    "log_dir": str(log_dir),
                }
            )
            registry = ProcessRegistry(paths, history)

            with patch.object(registry, "_spawn_shell") as spawn:
                with self.assertRaisesRegex(OSError, "torsion_spring.yaml"):
                    registry.start_play("directory_yaml_run", str(checkpoint), device="cpu")

            spawn.assert_not_called()

    def test_duplicate_top_level_yaml_backends_raise_before_spawning_play(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = self.make_paths(root)
            history = HistoryStore(paths)
            log_dir = paths.rsl_rl_log_root / "duplicate_yaml_run"
            params_dir = log_dir / "params"
            params_dir.mkdir(parents=True)
            checkpoint = log_dir / "model_10.pt"
            checkpoint.write_text("x", encoding="utf-8")
            (params_dir / "torsion_spring.yaml").write_text(
                "spring_backend: native\nspring_backend: unsupported\n",
                encoding="utf-8",
            )
            history.add_run(
                {
                    "id": "duplicate_yaml_run",
                    "source": "rsl_rl",
                    "status": "completed",
                    "log_dir": str(log_dir),
                }
            )
            registry = ProcessRegistry(paths, history)

            with patch.object(registry, "_spawn_shell") as spawn:
                with self.assertRaisesRegex(ValueError, "Duplicate"):
                    registry.start_play("duplicate_yaml_run", str(checkpoint), device="cpu")

            spawn.assert_not_called()

    def test_play_resolves_native_backend_from_checkpoint_parent_yaml(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = self.make_paths(root)
            history = HistoryStore(paths)
            checkpoint_dir = root / "checkpoint_parent"
            params_dir = checkpoint_dir / "params"
            params_dir.mkdir(parents=True)
            checkpoint = checkpoint_dir / "model_10.pt"
            checkpoint.write_text("x", encoding="utf-8")
            (params_dir / "torsion_spring.yaml").write_text("spring_backend: native\n", encoding="utf-8")
            history.add_run({"id": "run_without_log_dir", "source": "rsl_rl", "status": "completed"})
            registry = ProcessRegistry(paths, history)

            with patch.object(registry, "_spawn_shell", return_value=SpawnedProcess(proc=Mock(pid=123))) as spawn, patch.object(
                registry, "_raise_if_immediate_exit"
            ):
                registry.start_play("run_without_log_dir", str(checkpoint), device="cpu")

            self.assertIn("--spring-backend native", spawn.call_args.args[1])

    def test_onnx_export_process_uses_export_only_flags_and_updates_history(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = self.make_paths(root)
            history = HistoryStore(paths)
            log_dir = paths.rsl_rl_log_root / "run_one"
            log_dir.mkdir(parents=True)
            checkpoint = log_dir / "model_10.pt"
            checkpoint.write_text("x", encoding="utf-8")
            history.add_run(
                {
                    "id": "run_one",
                    "source": "training_panel",
                    "status": "completed",
                    "created_at": "2026-05-15T11:00:00",
                    "log_dir": str(log_dir),
                    "params": {"spring_backend": "native"},
                }
            )
            registry = ProcessRegistry(paths, history)
            result = registry.start_onnx_export("run_one", str(checkpoint), device="cpu")
            try:
                debug = registry.get_process_debug(result["id"])
                self.assertIsNotNone(debug)
                self.assertEqual(debug["kind"], "onnx")
                self.assertEqual(debug["source_run_id"], "run_one")
                self.assertIn("--headless", debug["command"])
                self.assertIn("--spring-backend native", debug["command"])
                self.assertIn("--export_policy_only", debug["command"])
                self.assertIn("attach_command", debug)
                run = history.get_run("run_one")
                self.assertEqual(run["onnx_status"], "exporting")
                self.assertEqual(run["onnx_process_id"], result["id"])
                self.assertEqual(run["onnx_pid"], result["pid"])
            finally:
                proc = registry._processes.get(result["id"])
                registry.stop(result["id"])
                if proc:
                    proc.wait(timeout=8)
                time.sleep(0.1)

    def test_successful_training_monitor_starts_video_recording(self):
        class CompletedProcess:
            def poll(self):
                return 0  # non-None → while loop body skipped immediately

            def wait(self):
                return 0

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = self.make_paths(root)
            history = HistoryStore(paths)
            log_dir = paths.rsl_rl_log_root / "run_one"
            log_dir.mkdir(parents=True)
            checkpoint = log_dir / "model_7.pt"
            checkpoint.write_text("x", encoding="utf-8")
            history.add_run(
                {
                    "id": "panel_train",
                    "source": "training_panel",
                    "status": "running",
                    "created_at": "2026-05-15T11:00:00",
                    "params": {"device": "cpu"},
                }
            )
            registry = ProcessRegistry(paths, history)
            registry.start_video_recording = Mock()
            registry._monitor_training("panel_train", CompletedProcess(), 0)

            registry.start_video_recording.assert_called_once_with(
                run_id="panel_train",
                checkpoint=str(checkpoint),
                device="cpu",
                video_params=VideoParams.from_preset("high"),
            )
            run = history.get_run("panel_train")
            self.assertEqual(run["status"], "completed")
            self.assertEqual(run["returncode"], 0)
            self.assertEqual(run["log_dir"], str(log_dir))

    def test_campaign_monitor_records_only_the_iteration_cap_checkpoint(self):
        class CompletedProcess:
            def poll(self):
                return 0

            def wait(self):
                return 0

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = self.make_paths(root)
            history = HistoryStore(paths)
            log_dir = paths.rsl_rl_log_root / "campaign_run"
            log_dir.mkdir(parents=True)
            exact = log_dir / "model_7.pt"
            exact.write_bytes(b"exact campaign output")
            (log_dir / "model_999.pt").write_bytes(b"unrelated later file")
            history.add_run(
                {
                    "id": "panel_campaign",
                    "source": "training_panel",
                    "status": "running",
                    "campaign_id": "campaign-one",
                    "created_at": "2026-08-14T12:00:00",
                    "params": {
                        "campaign_id": "campaign-one",
                        "max_iterations": 8,
                        "device": "cpu",
                    },
                }
            )
            registry = ProcessRegistry(paths, history)
            with patch.object(
                registry, "_log_dir_from_process_log", return_value=str(log_dir)
            ), patch.object(registry, "_refresh_tensorboard_summary"):
                registry._monitor_training("panel_campaign", CompletedProcess(), 0)

            run = history.get_run("panel_campaign")
            self.assertEqual(run["status"], "completed")
            self.assertEqual(run["output_checkpoint_path"], str(exact.resolve()))
            self.assertEqual(run["output_checkpoint_iteration"], 7)
            self.assertEqual(
                run["output_checkpoint_sha256"], hashlib.sha256(exact.read_bytes()).hexdigest()
            )

    def test_campaign_monitor_never_fallback_selects_another_run_directory(self):
        class CompletedProcess:
            def poll(self):
                return 0

            def wait(self):
                return 0

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = self.make_paths(root)
            history = HistoryStore(paths)
            unrelated = paths.rsl_rl_log_root / "unrelated"
            unrelated.mkdir(parents=True)
            (unrelated / "model_7.pt").write_bytes(b"wrong run")
            history.add_run(
                {
                    "id": "panel_campaign_missing_log",
                    "source": "training_panel",
                    "status": "running",
                    "campaign_id": "campaign-one",
                    "params": {"campaign_id": "campaign-one", "max_iterations": 8},
                }
            )
            registry = ProcessRegistry(paths, history)
            with patch.object(
                registry, "_log_dir_from_process_log", return_value=None
            ), patch.object(
                history, "find_latest_log_after", return_value=str(unrelated)
            ) as fallback, patch.object(registry, "_refresh_tensorboard_summary"):
                registry._monitor_training(
                    "panel_campaign_missing_log", CompletedProcess(), 0
                )

            fallback.assert_not_called()
            run = history.get_run("panel_campaign_missing_log")
            self.assertEqual(run["status"], "failed")
            self.assertEqual(run["failure_class"], "evidence")
            self.assertNotIn("output_checkpoint_path", run)

    def test_stop_all_for_run_stops_linked_play_process(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = self.make_paths(root)
            registry = ProcessRegistry(paths, HistoryStore(paths))
            result = registry.start_play("run_one", "/tmp/checkpoint.pt", device="cpu")
            proc = registry._processes.get(result["id"])
            try:
                stopped = registry.stop_all_for_run("run_one")
                self.assertEqual(stopped, [result["id"]])
                if proc:
                    proc.wait(timeout=8)
            finally:
                if proc and proc.poll() is None:
                    registry.stop(result["id"])
                    proc.wait(timeout=8)

    def test_running_for_run_filters_by_process_kind(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = self.make_paths(root)
            registry = ProcessRegistry(paths, HistoryStore(paths))
            result = registry.start_play("run_one", "/tmp/checkpoint.pt", device="cpu")
            proc = registry._processes.get(result["id"])
            try:
                play_processes = registry.running_for_run("run_one", kind="play")
                video_processes = registry.running_for_run("run_one", kind="video")
                self.assertEqual([process["run_id"] for process in play_processes], [result["id"]])
                self.assertEqual(video_processes, [])
                self.assertEqual([process["run_id"] for process in registry.running_media_processes()], [result["id"]])
            finally:
                registry.stop(result["id"])
                if proc:
                    proc.wait(timeout=8)

    def test_running_for_log_dir_blocks_unlinked_active_training_log(self):
        class FakeProcess:
            def poll(self):
                return None

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = self.make_paths(root)
            paths.ensure_dirs()
            history = HistoryStore(paths)
            registry = ProcessRegistry(paths, history)
            history.add_run({
                "id": "panel_run",
                "source": "training_panel",
                "status": "running",
                "created_at": "2026-05-17T01:35:27",
                "process_log": str(paths.process_log_dir / "panel_run.log"),
                "log_dir": None,
            })
            log_dir = paths.rsl_rl_log_root / "2026-05-17_01-35-34_wheg_locomotion_reform_v1"
            log_dir.mkdir(parents=True)
            os.utime(log_dir, (1778952937, 1778952937))
            process_log = paths.process_log_dir / "panel_run.log"
            process_log.write_text(f"Writing events to {log_dir}\n", encoding="utf-8")
            registry._processes["panel_run"] = FakeProcess()
            registry._infos["panel_run"] = ProcessInfo(
                kind="training",
                pid=12345,
                run_id="panel_run",
                log_file=str(process_log),
                started_at="2026-05-17T01:35:27",
                command="train.py --task Template-Redrhex-Direct-v0",
            )

            running = registry.running_for_log_dir(log_dir)

            self.assertEqual([process["run_id"] for process in running], ["panel_run"])

    def test_running_for_log_dir_uses_saved_start_time_for_external_training(self):
        class FakeProcess:
            def poll(self):
                return None

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = self.make_paths(root)
            paths.ensure_dirs()
            history = HistoryStore(paths)
            registry = ProcessRegistry(paths, history)
            history.add_run({
                "id": "panel_run",
                "source": "training_panel",
                "status": "running",
                "created_at": "2026-05-17T01:35:27",
                "process_log": str(paths.process_log_dir / "panel_run.log"),
                "log_dir": None,
            })
            log_dir = paths.rsl_rl_log_root / "2026-05-17_01-35-34_wheg_locomotion_reform_v1"
            log_dir.mkdir(parents=True)
            os.utime(log_dir, (1778952937, 1778952937))
            process_log = paths.process_log_dir / "panel_run.log"
            process_log.write_text(
                "Exact experiment name requested from command line: 2026-05-17_01-35-34\n",
                encoding="utf-8",
            )
            registry._processes["external_training_123"] = FakeProcess()
            registry._infos["external_training_123"] = ProcessInfo(
                kind="training",
                pid=12345,
                run_id="external_training_123",
                source_run_id="panel_run",
                log_file=str(process_log),
                started_at="",
                command="train.py --task Template-Redrhex-Direct-v0",
            )

            running = registry.running_for_log_dir(log_dir)

            self.assertEqual([process["run_id"] for process in running], ["external_training_123"])

    def test_log_dir_from_process_log_recovers_deleted_event_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = self.make_paths(root)
            paths.ensure_dirs()
            process_log = paths.process_log_dir / "panel_run.log"
            deleted_log_dir = paths.rsl_rl_log_root / "2026-05-17_01-35-34_wheg_locomotion_reform_v1"
            process_log.write_text(
                "FileNotFoundError: [Errno 2] No such file or directory: "
                f"b'{deleted_log_dir}/events.out.tfevents.1778952937.host.2617302.0'\n",
                encoding="utf-8",
            )
            history = HistoryStore(paths)
            history.add_run({
                "id": "panel_run",
                "source": "training_panel",
                "status": "running",
                "created_at": "2026-05-17T01:35:27",
                "process_log": str(process_log),
                "log_dir": None,
            })
            registry = ProcessRegistry(paths, history)

            self.assertEqual(registry._log_dir_from_process_log("panel_run"), str(deleted_log_dir))

    def test_training_command_match_rejects_conflicting_args(self):
        recorded = (
            "/IsaacLab/isaaclab.sh -p scripts/rsl_rl/train.py "
            "--task Template-Redrhex-Direct-v0 --num_envs 4 --max_iterations 10 --device cuda:0 --headless"
        )
        observed = (
            "bash /IsaacLab/isaaclab.sh -p scripts/rsl_rl/train.py "
            "--task Template-Redrhex-Direct-v0 --num_envs 4 --max_iterations 1 --device cuda:0 --headless"
        )
        self.assertFalse(ProcessRegistry._training_commands_match(recorded, observed))

    def test_training_command_match_accepts_same_core_args(self):
        recorded = (
            "/IsaacLab/isaaclab.sh -p scripts/rsl_rl/train.py "
            "--task Template-Redrhex-Direct-v0 --num_envs 4 --max_iterations 10 --device cuda:0 --headless"
        )
        observed = (
            "python scripts/rsl_rl/train.py "
            "--device cuda:0 --max_iterations 10 --num_envs 4 --task Template-Redrhex-Direct-v0 --spring-backend explicit"
        )
        self.assertTrue(ProcessRegistry._training_commands_match(recorded, observed))

    def test_training_command_match_rejects_different_spring_backends(self):
        recorded = (
            "python scripts/rsl_rl/train.py "
            "--task Template-Redrhex-Direct-v0 --num_envs 4 --max_iterations 10 --device cuda:0 --spring-backend native"
        )
        observed = (
            "python scripts/rsl_rl/train.py "
            "--task Template-Redrhex-Direct-v0 --num_envs 4 --max_iterations 10 --device cuda:0 --spring-backend explicit"
        )
        self.assertFalse(ProcessRegistry._training_commands_match(recorded, observed))

    def test_training_command_match_accepts_same_sensor_v2_pipeline(self):
        recorded = (
            "/IsaacLab/isaaclab.sh -p scripts/rsl_rl/train_sensor_v2_pipeline.py "
            "--num_envs 64 --teacher_iterations 1500 --distillation_iterations 800 "
            "--ppo_iterations 1500 --device cuda:0 --seed 42 --headless"
        )
        observed = (
            "python scripts/rsl_rl/train_sensor_v2_pipeline.py --seed 42 --device cuda:0 "
            "--ppo_iterations 1500 --distillation_iterations 800 --teacher_iterations 1500 --num_envs 64"
        )
        self.assertTrue(ProcessRegistry._training_commands_match(recorded, observed))

    def test_training_command_match_accepts_same_sensor_v2_full_pipeline(self):
        recorded = (
            "/IsaacLab/isaaclab.sh -p scripts/rsl_rl/train_sensor_v2_full_pipeline.py "
            "--isaaclab-launcher /IsaacLab/isaaclab.sh --f0-evidence /tmp/f0.json "
            f"--f0-evidence-sha256 {'a' * 64} --f4-profile /tmp/f4.json "
            f"--f4-profile-sha256 {'b' * 64} --f5-profile /tmp/f5.json "
            f"--f5-profile-sha256 {'c' * 64} --seeds 42 43 44 --num_envs 64 --device cuda:0"
        )
        observed = (
            "python scripts/rsl_rl/train_sensor_v2_full_pipeline.py --device cuda:0 "
            f"--f5-profile-sha256 {'c' * 64} --f5-profile /tmp/f5.json "
            f"--f4-profile-sha256 {'b' * 64} --f4-profile /tmp/f4.json "
            f"--f0-evidence-sha256 {'a' * 64} --f0-evidence /tmp/f0.json "
            "--isaaclab-launcher /IsaacLab/isaaclab.sh --num_envs 64 --seeds 42 43 44"
        )
        self.assertTrue(ProcessRegistry._training_commands_match(recorded, observed))

    def test_training_command_match_rejects_different_full_pipeline_seeds(self):
        recorded = (
            "python scripts/rsl_rl/train_sensor_v2_full_pipeline.py "
            "--isaaclab-launcher /IsaacLab/isaaclab.sh --f0-evidence /tmp/f0.json "
            f"--f0-evidence-sha256 {'a' * 64} --f4-profile /tmp/f4.json "
            f"--f4-profile-sha256 {'b' * 64} --f5-profile /tmp/f5.json "
            f"--f5-profile-sha256 {'c' * 64} --seeds 42 43 44 --num_envs 64 "
            "--robust_iterations 600 --device cuda:0"
        )
        observed = recorded.replace("--seeds 42 43 44", "--seeds 42 43 45")

        self.assertFalse(ProcessRegistry._training_commands_match(recorded, observed))

    def test_training_command_match_does_not_confuse_pipeline_with_child_stage(self):
        recorded = (
            "/IsaacLab/isaaclab.sh -p scripts/rsl_rl/train_sensor_v2_pipeline.py "
            "--num_envs 64 --teacher_iterations 1500 --distillation_iterations 800 "
            "--ppo_iterations 1500 --device cuda:0 --seed 42 --headless"
        )
        observed = (
            "python scripts/rsl_rl/train.py --task Template-Redrhex-ForwardSensorV2-Direct-v0 "
            "--num_envs 64 --max_iterations 1500 --device cuda:0 --seed 42 --headless"
        )
        self.assertFalse(ProcessRegistry._training_commands_match(recorded, observed))

    def test_running_isaac_processes_includes_onnx_exports(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = self.make_paths(root)
            history = HistoryStore(paths)
            history.add_run(
                {
                    "id": "run_one",
                    "source": "training_panel",
                    "status": "completed",
                    "params": {"task": "Template-Redrhex-ForwardFast-Direct-v0"},
                }
            )
            registry = ProcessRegistry(paths, history)
            result = registry.start_onnx_export("run_one", "/tmp/checkpoint.pt", device="cpu")
            proc = registry._processes.get(result["id"])
            try:
                self.assertEqual([process["run_id"] for process in registry.running_isaac_processes()], [result["id"]])
                debug = registry.get_process_debug(result["id"])
                self.assertIn("--task Template-Redrhex-ForwardFast-Direct-v0", debug["command"])
                self.assertNotIn("--initial_command", debug["command"])
            finally:
                registry.stop(result["id"])
                if proc:
                    proc.wait(timeout=8)

    def test_stop_all_for_run_returns_empty_for_nonexistent_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = self.make_paths(root)
            registry = ProcessRegistry(paths, HistoryStore(paths))
            self.assertEqual(registry.stop_all_for_run("missing_run"), [])

    def test_stop_all_for_run_ignores_already_exited_processes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = self.make_paths(root)
            registry = ProcessRegistry(paths, HistoryStore(paths))
            result = registry.start_play("run_one", "/tmp/checkpoint.pt", device="cpu")
            proc = registry._processes.get(result["id"])
            self.assertIsNotNone(proc)
            registry.stop(result["id"])
            proc.wait(timeout=8)
            self.assertEqual(registry.stop_all_for_run("run_one"), [])

    def test_source_run_id_from_play_command(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = self.make_paths(root)
            registry = ProcessRegistry(paths, HistoryStore(paths))
            command = (
                "python scripts/rsl_rl/play.py --checkpoint "
                "/home/lab_user1/Py/RedRHex/logs/rsl_rl/redrhex_wheg/2026_run/model_12.pt"
            )
            self.assertEqual(registry._source_run_id_from_command(command), "2026_run")

    def test_source_run_id_from_tensorboard_command(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = self.make_paths(root)
            registry = ProcessRegistry(paths, HistoryStore(paths))
            command = "tensorboard --logdir /repo/logs/rsl_rl/redrhex_wheg/2026_run --port 6008"
            self.assertEqual(registry._source_run_id_from_tensorboard_command(command), "2026_run")

    def test_external_tensorboard_ignores_tmux_wrapper_command(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = self.make_paths(root)
            registry = ProcessRegistry(paths, HistoryStore(paths))
            log_dir = paths.rsl_rl_log_root / "2026_run"
            log_dir.mkdir(parents=True)
            wrapper = f"bash -lc /usr/bin/tmux new-session -d -s tb -- tensorboard --logdir {log_dir}"
            child = f"python tensorboard --logdir {log_dir} --port 6006"
            output = f"111 111 Ss {wrapper}\n222 222 Sl {child}\n"
            with patch(
                "tools.training_panel.training_panel.processes.subprocess.check_output",
                return_value=output,
            ):
                processes = registry._external_tensorboard_processes()
            self.assertEqual([process["pid"] for process in processes], [222])

    def test_source_run_id_from_training_process_uses_panel_record_pid(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = self.make_paths(root)
            history = HistoryStore(paths)
            history.add_run(
                {
                    "id": "panel_train",
                    "source": "training_panel",
                    "pid": 123,
                    "created_at": "2026-05-15T11:00:00",
                    "command": f"{paths.isaaclab_launcher} -p scripts/rsl_rl/train.py --task Template-Redrhex-Direct-v0",
                }
            )
            registry = ProcessRegistry(paths, history)
            command = "python scripts/rsl_rl/train.py --task Template-Redrhex-Direct-v0"
            self.assertEqual(registry._source_run_id_from_training_process(123, 123, command), "panel_train")

    def test_source_run_id_from_training_process_prefers_exact_panel_log(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = self.make_paths(root)
            paths.ensure_dirs()
            history = HistoryStore(paths)
            command = (
                f"{paths.isaaclab_launcher} -p scripts/rsl_rl/train.py --task Template-Redrhex-Direct-v0 "
                "--num_envs 4096 --max_iterations 100000 --device cuda:0"
            )
            old_log = paths.process_log_dir / "panel_old.log"
            new_log = paths.process_log_dir / "panel_new.log"
            old_log.write_text("old training output\n", encoding="utf-8")
            history.add_run(
                {
                    "id": "panel_old",
                    "source": "training_panel",
                    "status": "interrupted",
                    "process_log": str(old_log),
                    "command": command,
                }
            )
            history.add_run(
                {
                    "id": "panel_new",
                    "source": "training_panel",
                    "status": "queued",
                    "process_log": str(new_log),
                    "command": command,
                }
            )
            observed = f"bash -lc set +e exec > >(tee -a {old_log}) 2>&1 {command}"
            registry = ProcessRegistry(paths, history)

            self.assertEqual(registry._source_run_id_from_training_process(999, 999, observed), "panel_old")
            self.assertEqual(registry._matching_training_log(999, 999, observed), old_log)

    def test_source_run_id_from_training_process_rejects_ambiguous_command_match(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = self.make_paths(root)
            history = HistoryStore(paths)
            command = (
                f"{paths.isaaclab_launcher} -p scripts/rsl_rl/train.py --task Template-Redrhex-Direct-v0 "
                "--num_envs 4 --max_iterations 10 --device cuda:0"
            )
            history.add_run({"id": "panel_one", "source": "training_panel", "command": command})
            history.add_run({"id": "panel_two", "source": "training_panel", "command": command})
            registry = ProcessRegistry(paths, history)

            self.assertIsNone(registry._source_run_id_from_training_process(999, 999, command))

    def test_external_training_process_maps_to_history_and_debug_log(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = self.make_paths(root)
            history = HistoryStore(paths)
            process_log = paths.process_log_dir / "panel_train.log"
            process_log.write_text("training is still running\n", encoding="utf-8")
            history.add_run(
                {
                    "id": "panel_train",
                    "source": "training_panel",
                    "pid": 222,
                    "created_at": "2026-05-15T11:00:00",
                    "process_log": str(process_log),
                    "command": (
                        f"{paths.isaaclab_launcher} -p scripts/rsl_rl/train.py --task Template-Redrhex-Direct-v0 "
                        "--num_envs 4 --max_iterations 8 --device cuda:0"
                    ),
                }
            )
            output = (
                f"222 222 Ss bash {paths.isaaclab_launcher} -p scripts/rsl_rl/train.py "
                "--task Template-Redrhex-Direct-v0 --num_envs 4 --max_iterations 8 --device cuda:0\n"
                "223 222 Rl python scripts/rsl_rl/train.py --task Template-Redrhex-Direct-v0 "
                "--num_envs 4 --max_iterations 8 --device cuda:0\n"
            )
            registry = ProcessRegistry(paths, history)
            with patch(
                "tools.training_panel.training_panel.processes.subprocess.check_output",
                return_value=output,
            ):
                processes = registry.list_processes()
                training = next(process for process in processes if process["kind"] == "training")
                self.assertEqual(training["run_id"], f"{EXTERNAL_TRAINING_ID_PREFIX}222")
                self.assertEqual(training["source_run_id"], "panel_train")
                debug = registry.get_process_debug(training["run_id"])
            self.assertIsNotNone(debug)
            self.assertIn("training is still running", debug["log_tail"])

    def test_external_training_process_without_history_still_blocks_gpu(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = self.make_paths(root)
            registry = ProcessRegistry(paths, HistoryStore(paths))
            output = (
                f"333 333 Sl python {paths.repo_root}/scripts/rsl_rl/train.py "
                "--task Template-Redrhex-Direct-v0 --num_envs 4 --max_iterations 8 --device cuda:0\n"
            )
            with patch(
                "tools.training_panel.training_panel.processes.subprocess.check_output",
                return_value=output,
            ):
                processes = registry.list_processes()

            self.assertEqual([process["run_id"] for process in processes], [f"{EXTERNAL_TRAINING_ID_PREFIX}333"])
            self.assertIsNone(processes[0]["source_run_id"])
            self.assertEqual([process["kind"] for process in processes], ["training"])

    def test_external_gpu_python_process_from_isaaclab_blocks_gpu(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = self.make_paths(root)
            registry = ProcessRegistry(paths, HistoryStore(paths))
            command = f"{paths.isaaclab_root}/python scripts/tutorials/00_sim/create_empty.py --headless"
            output = f"17627 17619 Rl {command}\n"
            with (
                patch.object(ProcessRegistry, "_gpu_device_pids", return_value={17627}),
                patch("tools.training_panel.training_panel.processes.subprocess.check_output", return_value=output),
            ):
                processes = registry.list_processes()

            self.assertEqual([process["run_id"] for process in processes], [f"{EXTERNAL_GPU_ID_PREFIX}17619"])
            self.assertEqual(processes[0]["kind"], "gpu")
            self.assertIsNone(processes[0]["source_run_id"])

    def test_external_video_process_is_classified_as_video_not_play(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = self.make_paths(root)
            history = HistoryStore(paths)
            log_dir = paths.rsl_rl_log_root / "run_one"
            log_dir.mkdir(parents=True)
            (log_dir / "model_9.pt").write_text("checkpoint", encoding="utf-8")
            command = (
                f"python scripts/rsl_rl/play.py --task Template-Redrhex-Direct-v0 --video "
                f"--checkpoint {log_dir / 'model_9.pt'}"
            )
            output = f"444 444 Ss {command}\n"
            registry = ProcessRegistry(paths, history)
            with patch(
                "tools.training_panel.training_panel.processes.subprocess.check_output",
                return_value=output,
            ):
                processes = registry.list_processes()
            self.assertEqual([process["kind"] for process in processes], ["video"])
            self.assertEqual(processes[0]["run_id"], f"{EXTERNAL_VIDEO_ID_PREFIX}444")
            self.assertEqual(processes[0]["source_run_id"], "run_one")

    def test_panel_owned_tmux_child_is_not_duplicated_as_external_process(self):
        class FakeProcess:
            pid = 222

            def poll(self):
                return None

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = self.make_paths(root)
            history = HistoryStore(paths)
            registry = ProcessRegistry(paths, history)
            log_file = paths.process_log_dir / "video_one.log"
            registry._processes["video_one"] = FakeProcess()
            registry._infos["video_one"] = ProcessInfo(
                kind="video",
                pid=222,
                run_id="video_one",
                log_file=str(log_file),
                started_at="2026-05-20T10:00:00",
                command="panel video command",
                source_run_id="run_one",
            )
            command = (
                f"bash -lc set +e exec > >(tee -a {log_file}) 2>&1 "
                "python scripts/rsl_rl/play.py --video --checkpoint /tmp/run_one/model_1.pt"
            )
            output = f"223 223 Ss {command}\n"
            with patch(
                "tools.training_panel.training_panel.processes.subprocess.check_output",
                return_value=output,
            ):
                processes = registry.list_processes()
            self.assertEqual([process["run_id"] for process in processes], ["video_one"])

    def test_registered_process_snapshot_includes_gpu_child_pid(self):
        class FakeProcess:
            pid = 222

            def poll(self):
                return None

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = self.make_paths(root)
            registry = ProcessRegistry(paths, HistoryStore(paths))
            info = ProcessInfo(
                kind="training",
                pid=222,
                run_id="panel_live",
                log_file=str(paths.process_log_dir / "panel_live.log"),
                started_at="2026-05-20T10:00:00",
                command="train.py",
            )
            with patch("tools.training_panel.training_panel.processes.os.getpgid") as getpgid, patch.object(
                registry, "_gpu_device_pids", return_value={333, 444}
            ):
                getpgid.side_effect = lambda pid: {222: 777, 333: 777, 444: 999}[pid]
                snapshot = registry._process_info_snapshot(info, FakeProcess())
            self.assertEqual(snapshot["process_group"], 777)
            self.assertEqual(snapshot["gpu_pid"], 333)
            self.assertEqual(snapshot["gpu_pids"], [333])

    def test_completed_process_snapshot_does_not_query_missing_tmux_session(self):
        class FakeProcess:
            pid = 222

            def poll(self):
                return 0

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = self.make_paths(root)
            registry = ProcessRegistry(paths, HistoryStore(paths))
            info = ProcessInfo(
                kind="training",
                pid=222,
                run_id="panel_done",
                log_file=str(paths.process_log_dir / "panel_done.log"),
                started_at="2026-05-20T10:00:00",
                command="train.py",
                tmux_session="missing_session",
            )
            with patch.object(registry, "_tmux_process_group", side_effect=AssertionError("tmux should not be queried")):
                snapshot = registry._process_info_snapshot(info, FakeProcess())
            self.assertEqual(snapshot["returncode"], 0)
            self.assertNotIn("process_group", snapshot)

    def test_tmux_server_title_does_not_count_as_training_process(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = self.make_paths(root)
            registry = ProcessRegistry(paths, HistoryStore(paths))
            command = (
                f"/usr/bin/tmux new-session -d -s redrhex_panel_fake -- bash -lc "
                f"{paths.isaaclab_launcher} -p scripts/rsl_rl/train.py --task Template-Redrhex-Direct-v0"
            )
            self.assertFalse(registry._is_repo_training_process(command, "123"))

    def test_sensor_v2_pipeline_counts_as_training_process(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = self.make_paths(root)
            registry = ProcessRegistry(paths, HistoryStore(paths))
            command = (
                f"python {paths.repo_root}/scripts/rsl_rl/train_sensor_v2_pipeline.py "
                "--num_envs 64 --teacher_iterations 1500 --distillation_iterations 800 "
                "--ppo_iterations 1500 --device cuda:0"
            )
            self.assertTrue(registry._is_repo_training_process(command, "123"))

    def test_sensor_v2_full_pipeline_counts_as_training_process(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = self.make_paths(root)
            registry = ProcessRegistry(paths, HistoryStore(paths))
            command = (
                f"python {paths.repo_root}/scripts/rsl_rl/train_sensor_v2_full_pipeline.py "
                "--isaaclab-launcher /IsaacLab/isaaclab.sh --f0-evidence /tmp/f0.json"
            )
            self.assertTrue(registry._is_repo_training_process(command, "123"))

    def test_reconcile_links_completed_panel_run_to_discovered_log(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = self.make_paths(root)
            history = HistoryStore(paths)
            process_log = paths.process_log_dir / "panel_done.log"
            process_log.write_text(
                "Exact experiment name requested from command line: 2026-05-15_13-58-07\n" + ("training...\n" * 20000),
                encoding="utf-8",
            )
            exit_file = paths.process_log_dir / "panel_done.exit"
            exit_file.write_text("0", encoding="utf-8")
            log_dir = paths.rsl_rl_log_root / "2026-05-15_13-58-07_wheg_locomotion"
            log_dir.mkdir(parents=True)
            (log_dir / "model_9999.pt").write_text("x", encoding="utf-8")
            newer_log_dir = paths.rsl_rl_log_root / "2026-05-15_15-18-22_wheg_locomotion"
            newer_log_dir.mkdir(parents=True)
            (newer_log_dir / "model_99.pt").write_text("x", encoding="utf-8")
            history.add_run(
                {
                    "id": "panel_done",
                    "source": "training_panel",
                    "status": "running",
                    "created_at": "2026-05-15T13:58:01",
                    "process_log": str(process_log),
                    "exit_file": str(exit_file),
                    "folder": "experiments",
                }
            )
            history.patch_run_metadata(log_dir.name, source="rsl_rl", log_dir=str(log_dir), folder="experiments")
            registry = ProcessRegistry(paths, history)
            with patch("tools.training_panel.training_panel.processes.subprocess.check_output", return_value=""):
                registry.reconcile_stale_history()

            runs = history.list_runs()
            panel = next(run for run in runs if run["id"] == "panel_done")
            self.assertEqual(panel["status"], "completed")
            self.assertEqual(panel["returncode"], 0)
            self.assertEqual(panel["log_dir"], str(log_dir))
            self.assertEqual(panel["latest_checkpoint"], str(log_dir / "model_9999.pt"))
            self.assertFalse(any(run["id"] == log_dir.name for run in runs))

    def test_campaign_reconciliation_finalizes_monitorless_training_after_exit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = self.make_paths(root)
            history = HistoryStore(paths)
            run_id = "panel_campaign_restart"
            process_log = paths.process_log_dir / f"{run_id}.log"
            process_log.parent.mkdir(parents=True, exist_ok=True)
            process_log.write_text(
                "Exact experiment name requested from command line: restart_run\n",
                encoding="utf-8",
            )
            log_dir = paths.rsl_rl_log_root / "restart_run_wheg_locomotion"
            log_dir.mkdir(parents=True)
            checkpoint = log_dir / "model_10.pt"
            checkpoint.write_text("checkpoint", encoding="utf-8")
            exit_file = paths.process_log_dir / f"{run_id}.exit"
            history.add_run(
                {
                    "id": run_id,
                    "source": "training_panel",
                    "status": "running",
                    "created_at": "2026-08-14T10:00:00",
                    "process_log": str(process_log),
                    "exit_file": str(exit_file),
                    "campaign_id": "campaign-one",
                    "campaign_trial_id": "trial-one",
                    "params": {
                        "campaign_id": "campaign-one",
                        "max_iterations": 11,
                    },
                }
            )
            restarted = ProcessRegistry(paths, HistoryStore(paths))

            unchanged = restarted.reconcile_campaign_process(run_id)
            self.assertEqual(unchanged["status"], "running")
            exit_file.write_text("0", encoding="utf-8")
            completed = restarted.reconcile_campaign_process(run_id)

            self.assertEqual(completed["status"], "completed")
            self.assertEqual(completed["returncode"], 0)
            self.assertEqual(completed["log_dir"], str(log_dir))
            self.assertEqual(completed["latest_checkpoint"], str(checkpoint))
            self.assertEqual(
                completed["output_checkpoint_path"], str(checkpoint.resolve())
            )
            self.assertEqual(completed["output_checkpoint_iteration"], 10)
            self.assertEqual(
                completed["output_checkpoint_sha256"],
                hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            )

    def test_fallback_training_persists_exit_receipt_before_spawn_and_reconciles(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = self.make_paths(root)
            paths.isaaclab_launcher.write_text(
                "#!/usr/bin/env bash\nexit 7\n",
                encoding="utf-8",
            )
            os.chmod(paths.isaaclab_launcher, 0o755)
            history = HistoryStore(paths)
            registry = ProcessRegistry(paths, history, isaac_settle_seconds=0)
            params = TrainingParams.from_dict(
                {
                    "task": "Template-Redrhex-Direct-v0",
                    "num_envs": 1,
                    "max_iterations": 1,
                    "device": "cpu",
                    "campaign_id": "campaign-fallback",
                    "campaign_trial_id": "trial-fallback",
                }
            )
            original_spawn = registry._spawn_shell
            recorded_before_spawn: list[str] = []

            def spawn_with_record_check(run_id, shell, log_file, **kwargs):
                recorded = history.get_run(run_id)
                recorded_before_spawn.append(str(recorded.get("exit_file") or ""))
                return original_spawn(run_id, shell, log_file, **kwargs)

            with (
                patch.object(registry, "_spawn_shell", side_effect=spawn_with_record_check),
                patch.object(registry, "_start_gpu_monitor"),
                patch("tools.training_panel.training_panel.processes.shutil.which", return_value=None),
            ):
                started = registry.start_training(params)

            process = registry._processes[started["id"]]
            self.assertEqual(process.wait(timeout=2), 7)
            registry._release_gpu_lease(started["id"])
            exit_file = paths.process_log_dir / f"{started['id']}.exit"
            self.assertEqual(recorded_before_spawn, [str(exit_file)])
            self.assertEqual(started["exit_file"], str(exit_file))
            self.assertEqual(exit_file.read_text(encoding="utf-8").strip(), "7")

            completed = ProcessRegistry(paths, HistoryStore(paths)).reconcile_campaign_process(
                started["id"]
            )
            self.assertEqual(completed["status"], "failed")
            self.assertEqual(completed["returncode"], 7)
            self.assertEqual(
                completed["failure_reason"],
                "training process exited with status 7",
            )

    def test_campaign_training_killed_after_history_add_before_spawn_is_infrastructure_failure(self):
        class SimulatedPanelDeath(BaseException):
            pass

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = self.make_paths(root)
            history = HistoryStore(paths)
            registry = ProcessRegistry(paths, history, launch_grace_seconds=0)
            params = TrainingParams.from_dict(
                {
                    "task": "Template-Redrhex-Direct-v0",
                    "num_envs": 1,
                    "max_iterations": 1,
                    "device": "cpu",
                    "campaign_id": "campaign-pre-spawn",
                    "campaign_trial_id": "trial-pre-spawn",
                }
            )
            with patch.object(
                registry,
                "_spawn_shell",
                side_effect=SimulatedPanelDeath("panel killed before spawn"),
            ):
                with self.assertRaises(SimulatedPanelDeath):
                    registry.start_training(params)

            intent = history.list_runs()[0]
            self.assertEqual(intent["launch_phase"], "prepared")
            self.assertRegex(intent["launch_owner_token"], r"^[0-9a-f]{32}$")
            registry._release_gpu_lease(intent["id"])

            grace_protected = ProcessRegistry(
                paths,
                HistoryStore(paths),
                launch_grace_seconds=60,
            ).reconcile_campaign_process(intent["id"])
            self.assertEqual(grace_protected["status"], "running")

            failed = ProcessRegistry(
                paths,
                HistoryStore(paths),
                launch_grace_seconds=0,
            ).reconcile_campaign_process(intent["id"])

            self.assertEqual(failed["status"], "failed")
            self.assertEqual(failed["launch_phase"], "failed")
            self.assertEqual(failed["failure_kind"], "infrastructure")
            self.assertIn("temporarily unavailable", failed["failure_reason"])
            self.assertIsNone(failed["returncode"])

    def test_campaign_evaluation_killed_after_history_add_before_spawn_is_infrastructure_failure(self):
        class SimulatedPanelDeath(BaseException):
            pass

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = self.make_paths(root)
            history = HistoryStore(paths)
            registry = ProcessRegistry(paths, history, launch_grace_seconds=0)
            with patch.object(
                registry,
                "_spawn_shell",
                side_effect=SimulatedPanelDeath("panel killed before spawn"),
            ):
                with self.assertRaises(SimulatedPanelDeath):
                    registry.start_evaluation(self.make_evaluation_params(root))

            intent = history.list_runs()[0]
            self.assertEqual(intent["source"], "autopilot_evaluation")
            self.assertEqual(intent["launch_phase"], "prepared")
            self.assertRegex(intent["launch_owner_token"], r"^[0-9a-f]{32}$")
            registry._release_gpu_lease(intent["id"])

            failed = ProcessRegistry(
                paths,
                HistoryStore(paths),
                launch_grace_seconds=0,
            ).reconcile_campaign_process(intent["id"])

            self.assertEqual(failed["status"], "failed")
            self.assertEqual(failed["launch_phase"], "failed")
            self.assertEqual(failed["failure_kind"], "infrastructure")
            self.assertIn("temporarily unavailable", failed["failure_reason"])
            self.assertIsNone(failed["returncode"])

    def test_live_monitorless_campaign_child_keeps_launch_intent_running(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = self.make_paths(root)
            history = HistoryStore(paths)
            registry = ProcessRegistry(paths, history, launch_grace_seconds=0)
            params = TrainingParams.from_dict(
                {
                    "task": "Template-Redrhex-Direct-v0",
                    "num_envs": 1,
                    "max_iterations": 1,
                    "device": "cpu",
                    "campaign_id": "campaign-live-child",
                    "campaign_trial_id": "trial-live-child",
                }
            )
            with (
                patch.object(registry, "_start_gpu_monitor"),
                patch("tools.training_panel.training_panel.processes.shutil.which", return_value=None),
            ):
                started = registry.start_training(params)
            proc = registry._processes[started["id"]]
            registry._drop_gpu_lease_reference(started["id"])
            try:
                recovered = ProcessRegistry(
                    paths,
                    HistoryStore(paths),
                    launch_grace_seconds=0,
                ).reconcile_campaign_process(started["id"])
                self.assertEqual(recovered["status"], "running")
                self.assertEqual(recovered["launch_phase"], "spawned")
            finally:
                if proc.poll() is None:
                    os.killpg(proc.pid, signal.SIGTERM)
                proc.wait(timeout=3)

    def test_restart_stop_resolves_exact_live_campaign_training_child(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = self.make_paths(root)
            paths.isaaclab_launcher.write_text(
                "#!/usr/bin/env bash\necho training-started\nsleep 30\n",
                encoding="utf-8",
            )
            os.chmod(paths.isaaclab_launcher, 0o755)
            history = HistoryStore(paths)
            original = ProcessRegistry(paths, history, launch_grace_seconds=0)
            params = TrainingParams.from_dict(
                {
                    "task": "Template-Redrhex-Direct-v0",
                    "num_envs": 1,
                    "max_iterations": 1,
                    "device": "cpu",
                    "campaign_id": "campaign-restart-stop-training",
                    "campaign_trial_id": "trial-restart-stop-training",
                }
            )
            with (
                patch.object(original, "_start_gpu_monitor"),
                patch("tools.training_panel.training_panel.processes.shutil.which", return_value=None),
            ):
                started = original.start_training(params)
            proc = original._processes[started["id"]]
            receipt = Path(started["launch_receipt_file"])
            deadline = time.time() + 2
            while not receipt.is_file() and time.time() < deadline:
                time.sleep(0.01)
            self.assertTrue(receipt.is_file())
            original._drop_gpu_lease_reference(started["id"])
            unrelated = subprocess.Popen(["sleep", "30"], start_new_session=True)
            try:
                restarted = ProcessRegistry(paths, HistoryStore(paths), launch_grace_seconds=0)
                with patch("tools.training_panel.training_panel.processes.threading.Thread") as thread_cls:
                    thread_cls.return_value.start = Mock()
                    self.assertTrue(restarted.stop(started["id"]))
                self.assertEqual(proc.wait(timeout=3), 130)
                self.assertIsNone(unrelated.poll())
                completed = restarted.reconcile_campaign_process(started["id"])
                self.assertEqual(completed["status"], "failed")
                self.assertEqual(completed["returncode"], 130)
                self.assertEqual(completed["launch_phase"], "finished")
            finally:
                if proc.poll() is None:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait(timeout=3)
                if unrelated.poll() is None:
                    os.killpg(unrelated.pid, signal.SIGTERM)
                    unrelated.wait(timeout=3)

    def test_restart_stop_resolves_exact_live_campaign_evaluation_child(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = self.make_paths(root)
            paths.isaaclab_launcher.write_text(
                "#!/usr/bin/env bash\necho evaluation-started\nsleep 30\n",
                encoding="utf-8",
            )
            os.chmod(paths.isaaclab_launcher, 0o755)
            history = HistoryStore(paths)
            original = ProcessRegistry(paths, history, launch_grace_seconds=0)
            with (
                patch.object(original, "_start_gpu_monitor"),
                patch("tools.training_panel.training_panel.processes.shutil.which", return_value=None),
            ):
                started = original.start_evaluation(self.make_evaluation_params(root))
            proc = original._processes[started["id"]]
            receipt = Path(started["launch_receipt_file"])
            deadline = time.time() + 2
            while not receipt.is_file() and time.time() < deadline:
                time.sleep(0.01)
            self.assertTrue(receipt.is_file())
            original._drop_gpu_lease_reference(started["id"])
            unrelated = subprocess.Popen(["sleep", "30"], start_new_session=True)
            try:
                restarted = ProcessRegistry(paths, HistoryStore(paths), launch_grace_seconds=0)
                with patch("tools.training_panel.training_panel.processes.threading.Thread") as thread_cls:
                    thread_cls.return_value.start = Mock()
                    self.assertTrue(restarted.stop(started["id"]))
                self.assertEqual(proc.wait(timeout=3), 130)
                self.assertIsNone(unrelated.poll())
                completed = restarted.reconcile_campaign_process(started["id"])
                self.assertEqual(completed["status"], "failed")
                self.assertEqual(completed["returncode"], 130)
                self.assertEqual(completed["launch_phase"], "finished")
            finally:
                if proc.poll() is None:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait(timeout=3)
                if unrelated.poll() is None:
                    os.killpg(unrelated.pid, signal.SIGTERM)
                    unrelated.wait(timeout=3)

    def test_restart_stop_rejects_swapped_receipt_and_preserves_unrelated_process(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = self.make_paths(root)
            paths.isaaclab_launcher.write_text(
                "#!/usr/bin/env bash\nsleep 30\n",
                encoding="utf-8",
            )
            os.chmod(paths.isaaclab_launcher, 0o755)
            history = HistoryStore(paths)
            original = ProcessRegistry(paths, history, launch_grace_seconds=0)
            params = TrainingParams.from_dict(
                {
                    "task": "Template-Redrhex-Direct-v0",
                    "num_envs": 1,
                    "max_iterations": 1,
                    "device": "cpu",
                    "campaign_id": "campaign-swapped-receipt",
                    "campaign_trial_id": "trial-swapped-receipt",
                }
            )
            with (
                patch.object(original, "_start_gpu_monitor"),
                patch("tools.training_panel.training_panel.processes.shutil.which", return_value=None),
            ):
                started = original.start_training(params)
            proc = original._processes[started["id"]]
            receipt_path = Path(started["launch_receipt_file"])
            deadline = time.time() + 2
            while not receipt_path.is_file() and time.time() < deadline:
                time.sleep(0.01)
            self.assertTrue(receipt_path.is_file())
            original._drop_gpu_lease_reference(started["id"])
            unrelated = subprocess.Popen(["sleep", "30"], start_new_session=True)
            try:
                receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
                receipt.update(pid=unrelated.pid, process_group=os.getpgid(unrelated.pid))
                receipt_path.write_text(json.dumps(receipt), encoding="utf-8")

                restarted = ProcessRegistry(paths, HistoryStore(paths), launch_grace_seconds=0)
                self.assertFalse(restarted.stop(started["id"]))
                self.assertIsNone(proc.poll())
                self.assertIsNone(unrelated.poll())
            finally:
                if proc.poll() is None:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait(timeout=3)
                if unrelated.poll() is None:
                    os.killpg(unrelated.pid, signal.SIGTERM)
                    unrelated.wait(timeout=3)

    def test_fallback_exit_receipt_preserves_interrupt_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = self.make_paths(root)
            registry = ProcessRegistry(paths, HistoryStore(paths))
            ready_file = root / "fallback-ready"
            with patch(
                "tools.training_panel.training_panel.processes.shutil.which",
                return_value=None,
            ):
                spawned = registry._spawn_shell(
                    "fallback_interrupt",
                    f"touch {ready_file}; sleep 10",
                    paths.process_log_dir / "fallback_interrupt.log",
                )
            deadline = time.time() + 2
            while not ready_file.exists() and time.time() < deadline:
                time.sleep(0.01)
            self.assertTrue(ready_file.exists())

            os.killpg(spawned.proc.pid, signal.SIGINT)

            self.assertEqual(spawned.proc.wait(timeout=2), 130)
            self.assertEqual(
                Path(spawned.exit_file or "").read_text(encoding="utf-8").strip(),
                "130",
            )

    def test_campaign_reconciliation_finalizes_monitorless_evaluation_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = self.make_paths(root)
            history = HistoryStore(paths)
            evaluation_id = "evaluation_campaign_restart"
            artifact_dir = paths.evaluation_dir / evaluation_id
            artifact_dir.mkdir(parents=True)
            command_csv = artifact_dir / "commands.csv"
            episode_csv = artifact_dir / "commands_episodes.csv"
            summary_csv = artifact_dir / "commands_summary.csv"
            command_csv.write_text("command,score\nforward,1\n", encoding="utf-8")
            episode_csv.write_text("command,episode\nforward,0\n", encoding="utf-8")
            summary_csv.write_text("metric,value\nevaluation.seed,42\n", encoding="utf-8")
            exit_file = paths.process_log_dir / f"{evaluation_id}.exit"
            history.add_run(
                {
                    "id": evaluation_id,
                    "source": "autopilot_evaluation",
                    "status": "running",
                    "created_at": "2026-08-14T10:00:00",
                    "exit_file": str(exit_file),
                    "command_csv": str(command_csv),
                    "episode_csv": str(episode_csv),
                    "summary_csv": str(summary_csv),
                    "campaign_id": "campaign-one",
                    "campaign_trial_id": "trial-one",
                    "params": {"campaign_id": "campaign-one"},
                }
            )
            restarted = ProcessRegistry(paths, HistoryStore(paths))

            self.assertEqual(
                restarted.reconcile_campaign_process(evaluation_id)["status"],
                "running",
            )
            exit_file.write_text("0", encoding="utf-8")
            completed = restarted.reconcile_campaign_process(evaluation_id)

            self.assertEqual(completed["status"], "completed")
            self.assertEqual(completed["returncode"], 0)
            self.assertEqual(completed["evaluation_command_count"], 1)
            self.assertEqual(completed["evaluation_episode_count"], 1)
            self.assertEqual(
                completed["command_csv_sha256"],
                hashlib.sha256(command_csv.read_bytes()).hexdigest(),
            )

    def test_reconcile_repairs_completed_panel_stub_from_process_log(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = self.make_paths(root)
            history = HistoryStore(paths)
            run_id = "panel_20260524_124402_128233"
            process_log = paths.process_log_dir / f"{run_id}.log"
            process_log.parent.mkdir(parents=True, exist_ok=True)
            process_log.write_text(
                "$ bash -lc <<'PANEL_COMMAND'\n"
                "exec train.py --max_iterations 10\n"
                "PANEL_COMMAND\n"
                "Exact experiment name requested from command line: 2026-05-24_12-44-08\n",
                encoding="utf-8",
            )
            exit_file = paths.process_log_dir / f"{run_id}.exit"
            exit_file.write_text("0", encoding="utf-8")
            log_dir = paths.rsl_rl_log_root / "2026-05-24_12-44-08_wheg_locomotion_reform_v1"
            log_dir.mkdir(parents=True)
            (log_dir / "model_9.pt").write_text("x", encoding="utf-8")
            history.add_run(
                {
                    "id": run_id,
                    "source": "training_panel",
                    "created_at": "2026-05-24T22:48:50",
                    "updated_at": "2026-05-24T22:48:50",
                    "log_dir": None,
                }
            )
            registry = ProcessRegistry(paths, history)
            with patch("tools.training_panel.training_panel.processes.subprocess.check_output", return_value=""):
                registry.reconcile_stale_history()

            run = history.get_run(run_id)
            self.assertEqual(run["status"], "completed")
            self.assertEqual(run["returncode"], 0)
            self.assertEqual(run["log_dir"], str(log_dir))
            self.assertEqual(run["process_log"], str(process_log))
            self.assertEqual(run["exit_file"], str(exit_file))
            self.assertEqual(run["command"], "exec train.py --max_iterations 10")
            self.assertEqual(run["created_at"], "2026-05-24T12:44:02")
            self.assertEqual(run["started_at"], "2026-05-24T12:44:02")
            self.assertIn("completed_at", run)

    def test_reconcile_persists_running_panel_log_dir_before_exit(self):
        class FakeProcess:
            pid = 12345

            def poll(self):
                return None

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = self.make_paths(root)
            history = HistoryStore(paths)
            process_log = paths.process_log_dir / "panel_running.log"
            process_log.write_text(
                "Exact experiment name requested from command line: 2026-05-17_01-35-34\n",
                encoding="utf-8",
            )
            log_dir = paths.rsl_rl_log_root / "2026-05-17_01-35-34_wheg_locomotion_reform_v1"
            log_dir.mkdir(parents=True)
            (log_dir / "model_1.pt").write_text("x", encoding="utf-8")
            history.add_run(
                {
                    "id": "panel_running",
                    "source": "training_panel",
                    "status": "running",
                    "created_at": "2026-05-17T01:35:27",
                    "process_log": str(process_log),
                    "log_dir": None,
                }
            )
            registry = ProcessRegistry(paths, history)
            registry._processes["panel_running"] = FakeProcess()
            registry._infos["panel_running"] = ProcessInfo(
                kind="training",
                pid=12345,
                run_id="panel_running",
                log_file=str(process_log),
                started_at="2026-05-17T01:35:27",
                command="train.py --task Template-Redrhex-Direct-v0",
            )

            with patch("tools.training_panel.training_panel.processes.subprocess.check_output", return_value=""):
                registry.reconcile_stale_history()

            raw = next(record for record in history._load_data()["runs"] if record["id"] == "panel_running")
            panel = history.get_run("panel_running")
            self.assertEqual(raw["status"], "running")
            self.assertEqual(raw["log_dir"], str(log_dir))
            self.assertEqual(panel["latest_checkpoint"], str(log_dir / "model_1.pt"))

    def test_reconcile_repairs_live_training_stub(self):
        class FakeProcess:
            pid = 12345

            def poll(self):
                return None

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = self.make_paths(root)
            history = HistoryStore(paths)
            process_log = paths.process_log_dir / "panel_live.log"
            process_log.write_text("training output\n", encoding="utf-8")
            history.add_run({"id": "panel_live", "source": "training_panel"})
            registry = ProcessRegistry(paths, history)
            registry._processes["panel_live"] = FakeProcess()
            registry._infos["panel_live"] = ProcessInfo(
                kind="training",
                pid=12345,
                run_id="panel_live",
                log_file=str(process_log),
                started_at="2026-05-17T01:35:27",
                command="train.py --task Template-Redrhex-Direct-v0",
                tmux_session="redrhex_panel_live",
                attach_command="tmux attach -t redrhex_panel_live",
                exit_file=str(paths.process_log_dir / "panel_live.exit"),
            )

            with patch.object(registry, "_process_group_for_info", return_value=777), patch.object(
                registry, "_gpu_pids_for_group", return_value=[888]
            ), patch("tools.training_panel.training_panel.processes.subprocess.check_output", return_value=""):
                registry.reconcile_stale_history()

            raw = next(record for record in history._load_data()["runs"] if record["id"] == "panel_live")
            self.assertEqual(raw["status"], "running")
            self.assertEqual(raw["process_log"], str(process_log))
            self.assertEqual(raw["process_group"], 777)
            self.assertEqual(raw["gpu_pid"], 888)
            self.assertEqual(raw["gpu_pids"], [888])
            self.assertEqual(raw["tmux_session"], "redrhex_panel_live")

    def test_reconcile_persists_fresh_discovered_log_before_exact_name(self):
        class FakeProcess:
            pid = 12346

            def poll(self):
                return None

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = self.make_paths(root)
            history = HistoryStore(paths)
            process_log = paths.process_log_dir / "panel_running.log"
            process_log.write_text("Isaac startup has not printed the experiment name yet.\n", encoding="utf-8")
            log_dir = paths.rsl_rl_log_root / "2026-05-17_10-21-24_wheg_locomotion_reform_v1"
            log_dir.mkdir(parents=True)
            (log_dir / "model_0.pt").write_text("x", encoding="utf-8")
            history.add_run(
                {
                    "id": "panel_20260517_102117_740732",
                    "source": "training_panel",
                    "status": "running",
                    "created_at": "2026-05-17T10:21:17",
                    "process_log": str(process_log),
                    "log_dir": None,
                }
            )
            registry = ProcessRegistry(paths, history)
            registry._processes["panel_20260517_102117_740732"] = FakeProcess()
            registry._infos["panel_20260517_102117_740732"] = ProcessInfo(
                kind="training",
                pid=12346,
                run_id="panel_20260517_102117_740732",
                log_file=str(process_log),
                started_at="2026-05-17T10:21:17",
                command="train.py --task Template-Redrhex-Direct-v0",
            )

            with patch("tools.training_panel.training_panel.processes.subprocess.check_output", return_value=""):
                registry.reconcile_stale_history()

            raw = next(record for record in history._load_data()["runs"] if record["id"] == "panel_20260517_102117_740732")
            runs = history.list_runs()
            self.assertEqual([run["id"] for run in runs], ["panel_20260517_102117_740732"])
            self.assertEqual(raw["status"], "running")
            self.assertEqual(raw["log_dir"], str(log_dir))

    def test_reconcile_failed_run_does_not_time_fallback_to_neighbor_log(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = self.make_paths(root)
            history = HistoryStore(paths)
            process_log = paths.process_log_dir / "panel_failed.log"
            process_log.write_text(
                "Exact experiment name requested from command line: 2026-05-17_10-31-01\n"
                "CUDA error: out of memory\n",
                encoding="utf-8",
            )
            exit_file = paths.process_log_dir / "panel_failed.exit"
            exit_file.write_text("1", encoding="utf-8")
            neighbor_log = paths.rsl_rl_log_root / "2026-05-17_10-30-57_wheg_locomotion_reform_v1"
            neighbor_log.mkdir(parents=True)
            (neighbor_log / "model_19.pt").write_text("x", encoding="utf-8")
            history.add_run(
                {
                    "id": "panel_failed",
                    "source": "training_panel",
                    "status": "running",
                    "created_at": "2026-05-17T10:30:55",
                    "process_log": str(process_log),
                    "exit_file": str(exit_file),
                    "log_dir": None,
                }
            )
            registry = ProcessRegistry(paths, history)

            with patch("tools.training_panel.training_panel.processes.subprocess.check_output", return_value=""):
                registry.reconcile_stale_history()

            run = history.get_run("panel_failed")
            self.assertEqual(run["status"], "failed")
            self.assertEqual(run["returncode"], 1)
            self.assertIsNone(run["log_dir"])
            self.assertIsNone(run["latest_checkpoint"])

    def test_reconcile_stale_history_marks_missing_panel_process_interrupted(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = self.make_paths(root)
            history = HistoryStore(paths)
            history.add_run(
                {
                    "id": "panel_stale",
                    "source": "training_panel",
                    "status": "running",
                    "created_at": "2026-05-15T11:00:00",
                    "pid": 456,
                }
            )
            registry = ProcessRegistry(paths, history)
            with patch("tools.training_panel.training_panel.processes.subprocess.check_output", return_value=""):
                registry.reconcile_stale_history()
            self.assertEqual(history.get_run("panel_stale")["status"], "interrupted")

    def test_sensor_v2_pipeline_result_resolves_final_ppo_log(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = self.make_paths(root)
            history = HistoryStore(paths)
            final_dir = paths.rsl_rl_log_root / "redrhex_forward_v2_ppo" / "final_run"
            final_dir.mkdir(parents=True)
            process_log = paths.process_log_dir / "panel_pipeline.log"
            result = {
                "status": "completed",
                "teacher_log_dir": str(paths.rsl_rl_log_root / "redrhex_forward_v2_teacher" / "teacher"),
                "distillation_log_dir": str(paths.rsl_rl_log_root / "redrhex_forward_v2_distillation" / "student"),
                "ppo_log_dir": str(final_dir),
            }
            process_log.write_text(
                f"SENSOR_V2_PIPELINE_RESULT: {json.dumps(result)}\n",
                encoding="utf-8",
            )
            history.add_run(
                {
                    "id": "panel_pipeline",
                    "source": "training_panel",
                    "status": "running",
                    "process_log": str(process_log),
                    "params": {"training_route": "sensor_v2_f1_f3"},
                }
            )
            registry = ProcessRegistry(paths, history)
            parsed = registry._sensor_v2_pipeline_result("panel_pipeline")
            self.assertIsNotNone(parsed)
            self.assertEqual(parsed["ppo_log_dir"], str(final_dir.resolve()))
            self.assertTrue(parsed["debug_only"])
            self.assertFalse(parsed["deployment_eligible"])
            self.assertFalse(parsed["promotion_eligible"])
            self.assertEqual(
                parsed["acceptance_screening"], "not_run_legacy_debug_only"
            )

    def test_sensor_v2_full_pipeline_result_resolves_primary_robust_seed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = self.make_paths(root)
            history = HistoryStore(paths)
            runs = []
            for seed in (42, 43, 44):
                run_dir = (
                    root
                    / "logs/rsl_rl/redrhex_forward_v2_robust_ppo"
                    / f"robust_seed_{seed}"
                )
                run_dir.mkdir(parents=True)
                checkpoint = run_dir / "model_600.pt"
                checkpoint.write_bytes(f"checkpoint-{seed}".encode())
                runs.append(
                    {
                        "stage": "f4_robust_ppo",
                        "seed": seed,
                        "run_dir": str(run_dir),
                        "checkpoint": str(checkpoint),
                        "checkpoint_sha256": hashlib.sha256(
                            checkpoint.read_bytes()
                        ).hexdigest(),
                    }
                )
            process_log = paths.process_log_dir / "panel_full_pipeline.log"
            result = {
                "schema": "redrhex.sensor-v2-full-pipeline.v2",
                "status": "passed",
                "seeds": [42, 43, 44],
                "f0_evidence": {"path": "/tmp/f0.json", "sha256": "a" * 64},
                "f4_profile": {"path": "/tmp/f4.json", "sha256": "b" * 64},
                "f5_profile": {"path": "/tmp/f5.json", "sha256": "c" * 64},
                "runs": runs,
            }
            process_log.write_text(
                f"SENSOR_V2_FULL_PIPELINE_RESULT: {json.dumps(result)}\n",
                encoding="utf-8",
            )
            history.add_run(
                {
                    "id": "panel_full_pipeline",
                    "source": "training_panel",
                    "status": "running",
                    "process_log": str(process_log),
                    "params": {
                        "training_route": "sensor_v2_full",
                        "seeds": [42, 43, 44],
                        "f0_evidence": "/tmp/f0.json",
                        "f0_evidence_sha256": "a" * 64,
                        "f4_profile": "/tmp/f4.json",
                        "f4_profile_sha256": "b" * 64,
                        "f5_profile": "/tmp/f5.json",
                        "f5_profile_sha256": "c" * 64,
                    },
                }
            )
            parsed = ProcessRegistry(paths, history)._sensor_v2_pipeline_result(
                "panel_full_pipeline"
            )
            self.assertIsNotNone(parsed)
            self.assertEqual(parsed["primary_seed"], 42)
            self.assertEqual(parsed["ppo_log_dir"], runs[0]["run_dir"])

            result["f5_profile"]["sha256"] = "d" * 64
            process_log.write_text(
                f"SENSOR_V2_FULL_PIPELINE_RESULT: {json.dumps(result)}\n",
                encoding="utf-8",
            )
            self.assertIsNone(
                ProcessRegistry(paths, history)._sensor_v2_pipeline_result(
                    "panel_full_pipeline"
                )
            )
            result["f5_profile"]["sha256"] = "c" * 64
            duplicate = {**runs[0], "checkpoint_sha256": runs[0]["checkpoint_sha256"]}
            result["runs"].append(duplicate)
            process_log.write_text(
                f"SENSOR_V2_FULL_PIPELINE_RESULT: {json.dumps(result)}\n",
                encoding="utf-8",
            )
            self.assertIsNone(
                ProcessRegistry(paths, history)._sensor_v2_pipeline_result(
                    "panel_full_pipeline"
                )
            )

    def test_full_pipeline_monitor_fails_closed_without_valid_result(self):
        class CompletedProcess:
            def poll(self):
                return 0

            def wait(self):
                return 0

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = self.make_paths(root)
            history = HistoryStore(paths)
            process_log = paths.process_log_dir / "panel_full_missing_result.log"
            process_log.write_text("pipeline child output only\n", encoding="utf-8")
            history.add_run(
                {
                    "id": "panel_full_missing_result",
                    "source": "training_panel",
                    "status": "running",
                    "process_log": str(process_log),
                    "params": {
                        "training_route": "sensor_v2_full",
                        "device": "cpu",
                        "seeds": [42, 43, 44],
                    },
                }
            )
            registry = ProcessRegistry(paths, history)
            registry.start_video_recording = Mock()

            with patch.object(registry, "_refresh_tensorboard_summary"):
                registry._monitor_training(
                    "panel_full_missing_result",
                    CompletedProcess(),
                    0,
                )

            run = history.get_run("panel_full_missing_result")
            self.assertEqual(run["status"], "failed")
            self.assertEqual(run["failure_class"], "evidence")
            self.assertIsNone(run["log_dir"])
            registry.start_video_recording.assert_not_called()


if __name__ == "__main__":
    unittest.main()
