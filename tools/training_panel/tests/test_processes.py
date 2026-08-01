import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from tools.training_panel.training_panel.commands import TrainingParams, VideoParams, resolve_spring_backend
from tools.training_panel.training_panel.config import PanelPaths
from tools.training_panel.training_panel.history import HistoryStore
from tools.training_panel.training_panel.processes import (
    CudaPreflightError,
    EXTERNAL_GPU_ID_PREFIX,
    EXTERNAL_TRAINING_ID_PREFIX,
    EXTERNAL_VIDEO_ID_PREFIX,
    ProcessInfo,
    ProcessRegistry,
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
                    "params": {"spring_backend": "native"},
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
                    "params": {"spring_backend": "native"},
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

    def test_running_isaac_processes_includes_onnx_exports(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = self.make_paths(root)
            registry = ProcessRegistry(paths, HistoryStore(paths))
            result = registry.start_onnx_export("run_one", "/tmp/checkpoint.pt", device="cpu")
            proc = registry._processes.get(result["id"])
            try:
                self.assertEqual([process["run_id"] for process in registry.running_isaac_processes()], [result["id"]])
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


if __name__ == "__main__":
    unittest.main()
