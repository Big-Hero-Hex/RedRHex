import ast
import hashlib
import tempfile
import unittest
from pathlib import Path

from tools.training_panel.training_panel.commands import (
    EvaluationParams,
    FORWARD_FAST_TASK,
    TrainingParams,
    VideoParams,
    export_onnx_argv,
    evaluation_argv,
    play_argv,
    shell_for_command,
    training_argv,
)
from tools.training_panel.training_panel.config import PanelPaths


class CommandTests(unittest.TestCase):
    def make_paths(self, root: Path) -> PanelPaths:
        return PanelPaths(
            repo_root=root,
            isaaclab_root=root / "IsaacLab",
            isaacsim_root=root / "isaacsim",
            conda_sh=root / "miniconda3" / "etc" / "profile.d" / "conda.sh",
            conda_env="env",
        )

    def test_shell_activates_conda_env_and_keeps_conda_lib_first(self):
        with tempfile.TemporaryDirectory() as tmp:
            shell = shell_for_command(self.make_paths(Path(tmp)), ["tensorboard", "--version"])
            self.assertIn("source ", shell)
            self.assertIn("conda activate env", shell)
            self.assertIn('export PATH="$CONDA_PREFIX/bin:$PATH"', shell)
            self.assertIn('export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"', shell)

    def test_play_cli_defaults_to_forward_command(self):
        repo_root = Path(__file__).resolve().parents[3]
        tree = ast.parse((repo_root / "scripts/rsl_rl/play.py").read_text(encoding="utf-8"))
        initial_command_call = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "add_argument"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and node.args[0].value == "--initial_command"
        )
        default = next(keyword.value for keyword in initial_command_call.keywords if keyword.arg == "default")
        self.assertEqual(ast.literal_eval(default), "forward")

    def test_training_defaults_to_forward_fast_task(self):
        self.assertEqual(TrainingParams().task, FORWARD_FAST_TASK)
        self.assertEqual(TrainingParams.from_dict({}).task, FORWARD_FAST_TASK)
        self.assertEqual(TrainingParams().reward_preset_id, "speed-focus")
        self.assertEqual(TrainingParams.from_dict({}).reward_preset_id, "speed-focus")

    def test_forward_fast_source_reward_profile_matches_panel_preset(self):
        repo_root = Path(__file__).resolve().parents[3]
        tree = ast.parse(
            (repo_root / "source/RedRhex/RedRhex/tasks/direct/redrhex/redrhex_env_cfg.py").read_text(encoding="utf-8")
        )
        forward_fast = next(
            node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "RedrhexForwardFastEnvCfg"
        )
        scales_node = next(
            node.value
            for node in forward_fast.body
            if isinstance(node, (ast.Assign, ast.AnnAssign))
            and (
                (isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "v2_reward_scales" for target in node.targets))
                or (isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id == "v2_reward_scales")
            )
        )
        scales = ast.literal_eval(scales_node)
        self.assertEqual(
            {key: scales[key] for key in (
                "forward_progress",
                "velocity_tracking",
                "axis_suppression",
                "height_maintain",
                "height_low_penalty",
                "leg_moving",
                "stall_penalty",
                "energy_per_distance",
            )},
            {
                "forward_progress": 3.0,
                "velocity_tracking": 6.0,
                "axis_suppression": 2.0,
                "height_maintain": 1.0,
                "height_low_penalty": 1.5,
                "leg_moving": 0.25,
                "stall_penalty": -3.0,
                "energy_per_distance": 0.0005,
            },
        )

    def test_play_argv_supports_headless_video_recording(self):
        params = VideoParams.from_preset("high")
        argv = play_argv(
            "/tmp/model_10.pt",
            device="cuda:0",
            headless=True,
            video=True,
            video_length=params.length,
            video_width=params.width,
            video_height=params.height,
            video_fps=params.fps,
            rendering_mode=params.rendering_mode,
            initial_command="forward",
        )
        self.assertIn("scripts/rsl_rl/play.py", argv)
        self.assertIn("--headless", argv)
        self.assertIn("--video", argv)
        self.assertIn("--video_length", argv)
        self.assertEqual(argv[argv.index("--video_length") + 1], "1200")
        self.assertEqual(argv[argv.index("--video_width") + 1], "1920")
        self.assertEqual(argv[argv.index("--video_height") + 1], "1080")
        self.assertEqual(argv[argv.index("--video_fps") + 1], "30")
        self.assertEqual(argv[argv.index("--rendering_mode") + 1], "quality")
        self.assertEqual(argv[argv.index("--initial_command") + 1], "forward")
        self.assertEqual(argv[argv.index("--checkpoint") + 1], "/tmp/model_10.pt")

    def test_play_argv_supports_terrain_replay_and_follow_camera(self):
        argv = play_argv(
            "/tmp/model_10.pt",
            terrain_override_file="/tmp/terrain.json",
            camera_follow_robot=True,
            camera_eye=(-3.0, -2.4, 1.6),
            camera_lookat=(0.45, 0.0, 0.35),
        )
        self.assertEqual(argv[argv.index("--terrain_override_file") + 1], "/tmp/terrain.json")
        self.assertIn("--camera_follow_robot", argv)
        self.assertEqual(argv[argv.index("--camera_eye") + 1 : argv.index("--camera_eye") + 4], ["-3.0", "-2.4", "1.6"])
        self.assertEqual(argv[argv.index("--camera_lookat") + 1 : argv.index("--camera_lookat") + 4], ["0.45", "0.0", "0.35"])

    def test_video_default_is_high_quality(self):
        high = VideoParams.from_preset(None)
        self.assertEqual(high.preset, "high")
        self.assertEqual(high.width, 1920)
        self.assertEqual(high.height, 1080)
        self.assertEqual(high.length, 1200)
        self.assertEqual(high.rendering_mode, "quality")

    def test_export_onnx_argv_uses_headless_export_only_play(self):
        argv = export_onnx_argv("/tmp/model_10.pt", device="cuda:0")
        self.assertIn("scripts/rsl_rl/play.py", argv)
        self.assertIn("--headless", argv)
        self.assertIn("--export_policy_only", argv)
        self.assertEqual(argv[argv.index("--spring-backend") + 1], "explicit")
        self.assertEqual(argv[argv.index("--device") + 1], "cuda:0")
        self.assertEqual(argv[argv.index("--checkpoint") + 1], "/tmp/model_10.pt")
        self.assertNotIn("--initial_command", argv)

    def test_export_onnx_argv_forwards_requested_spring_backend(self):
        argv = export_onnx_argv("/tmp/model_10.pt", spring_backend="native")

        self.assertEqual(argv[argv.index("--spring-backend") + 1], "native")

    def test_play_argv_preserves_explicit_legacy_default_and_allows_native_override(self):
        default_argv = play_argv("/tmp/model_10.pt")
        native_argv = play_argv("/tmp/model_10.pt", spring_backend="native")

        self.assertEqual(default_argv[default_argv.index("--spring-backend") + 1], "explicit")
        self.assertEqual(native_argv[native_argv.index("--spring-backend") + 1], "native")

    def test_training_params_store_and_serialize_requested_spring_backend(self):
        params = TrainingParams.from_dict({"spring_backend": "native"})

        self.assertEqual(params.spring_backend, "native")
        self.assertEqual(params.to_dict()["spring_backend"], "native")

    def test_training_params_default_spring_backend_is_native(self):
        params = TrainingParams.from_dict({})

        self.assertEqual(params.to_dict()["spring_backend"], "native")

    def test_training_params_can_parse_historical_explicit_backend(self):
        params = TrainingParams.from_dict({"spring_backend": "explicit"})

        self.assertEqual(params.spring_backend, "explicit")
        with self.assertRaisesRegex(ValueError, "quarantined.*120 Hz"):
            training_argv(params)

    def test_training_params_reject_unknown_spring_backend_before_queuing(self):
        with self.assertRaises(ValueError):
            TrainingParams.from_dict({"spring_backend": "unsupported"})

    def test_training_params_reject_present_falsey_spring_backend_values(self):
        for backend in ("", None, False, 0):
            with self.subTest(backend=backend), self.assertRaises(ValueError):
                TrainingParams.from_dict({"spring_backend": backend})

    def test_direct_play_and_export_argv_reject_unknown_spring_backend(self):
        with self.assertRaises(ValueError):
            play_argv("/tmp/model_10.pt", spring_backend="unsupported")
        with self.assertRaises(ValueError):
            export_onnx_argv("/tmp/model_10.pt", spring_backend="unsupported")

    def test_training_argv_includes_requested_spring_backend(self):
        argv = training_argv(TrainingParams.from_dict({"spring_backend": "native"}))

        self.assertEqual(argv[argv.index("--spring-backend") + 1], "native")

    def test_autopilot_training_argv_uses_immutable_profiles_and_strict_policy_fork(self):
        digest = "a" * 64
        params = TrainingParams.from_dict(
            {
                "checkpoint": "/tmp/model_10.pt",
                "checkpoint_sha256": digest,
                "initialization_mode": "policy_only",
                "strict_checkpoint_loading": True,
                "curriculum_stage": 2,
            }
        )

        argv = training_argv(
            params,
            reward_profile_file="/tmp/reward.json",
            reward_profile_sha256="b" * 64,
            terrain_profile_file="/tmp/terrain.json",
            terrain_profile_sha256="c" * 64,
        )

        self.assertNotIn("--panel_overrides", argv)
        self.assertIn("--resume_policy_only", argv)
        self.assertIn("--strict-checkpoint-loading", argv)
        self.assertEqual(argv[argv.index("--checkpoint-sha256") + 1], digest)
        self.assertEqual(argv[argv.index("--curriculum-stage") + 1], "2")
        self.assertEqual(argv[argv.index("--reward-profile") + 1], "/tmp/reward.json")

    def test_evaluation_argv_requires_exact_checkpoint_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "model_10.pt"
            checkpoint.write_bytes(b"checkpoint")
            command_profile = Path(tmp) / "command_profile.json"
            command_profile.write_text("{}", encoding="utf-8")
            command_profile_digest = hashlib.sha256(command_profile.read_bytes()).hexdigest()
            params = EvaluationParams(
                source_run_id="panel_1",
                checkpoint=str(checkpoint),
                checkpoint_sha256="a" * 64,
                task=FORWARD_FAST_TASK,
                agent_entry_point="rsl_rl_cfg_entry_point",
                seed=42,
                evaluation_profile="stage1",
                curriculum_stage=1,
                command_profile_file=str(command_profile),
                command_profile_sha256=command_profile_digest,
                code_sha256="b" * 64,
                config_sha256="c" * 64,
                dependency_sha256="d" * 64,
                reward_profile_sha256="e" * 64,
                physics_identity_sha256="f" * 64,
                spring_identity_sha256="1" * 64,
                terrain_profile_sha256="2" * 64,
            )

            argv = evaluation_argv(params, csv_file=str(Path(tmp) / "commands.csv"))

            self.assertIn("--strict-checkpoint-loading", argv)
            self.assertEqual(argv[argv.index("--checkpoint") + 1], str(checkpoint))
            self.assertEqual(argv[argv.index("--checkpoint-sha256") + 1], "a" * 64)
            self.assertEqual(argv[argv.index("--eval_profile") + 1], "stage1")
            self.assertEqual(argv[argv.index("--sweep_steps") + 1], "600")
            self.assertEqual(
                float(argv[argv.index("--expected-step-dt") + 1]), 1.0 / 60.0
            )
            self.assertEqual(
                argv[argv.index("--agent") + 1], "rsl_rl_cfg_entry_point"
            )
            self.assertEqual(argv[argv.index("--command-profile-sha256") + 1], command_profile_digest)
            self.assertEqual(argv[argv.index("--identity-reward-profile-sha256") + 1], "e" * 64)

    def test_physics_profile_is_forwarded_to_train_play_and_export(self):
        params = TrainingParams.from_dict(
            {
                "device": "cpu",
                "physics_preset_id": "measured",
                "physics_overrides": {
                    "simulation_physics.mass.scale": "1.05",
                    "simulation_physics.main_drive.effort_limit": 13.0,
                },
            }
        )
        profile_path = "/tmp/panel_physics.json"

        self.assertEqual(params.physics_preset_id, "measured")
        self.assertEqual(params.physics_overrides["simulation_physics.mass.scale"], 1.05)
        for argv in (
            training_argv(params, physics_profile_file=profile_path),
            play_argv("/tmp/model.pt", physics_profile_file=profile_path),
            export_onnx_argv("/tmp/model.pt", physics_profile_file=profile_path),
        ):
            self.assertEqual(argv[argv.index("--physics-profile") + 1], profile_path)

    def test_sensor_v2_pipeline_forwards_physics_profile(self):
        params = TrainingParams.from_dict({"training_route": "sensor_v2_full", "device": "cpu"})
        argv = training_argv(params, physics_profile_file="/tmp/panel_physics.json")

        self.assertIn("scripts/rsl_rl/train_sensor_v2_pipeline.py", argv)
        self.assertEqual(argv[argv.index("--spring-backend") + 1], "native")
        self.assertEqual(argv[argv.index("--physics-profile") + 1], "/tmp/panel_physics.json")

    def test_training_params_accept_terrain_overrides(self):
        params = TrainingParams.from_dict(
            {
                "task": "Template-Redrhex-Direct-v0",
                "num_envs": 4,
                "max_iterations": 1,
                "device": "cuda:0",
                "terrain_preset_id": "flat-debug",
                "terrain_overrides": {
                    "terrain.terrain_type": "plane",
                    "terrain_curriculum_enable": False,
                    "terrain_curriculum_levels": [0.0],
                },
            }
        )
        self.assertEqual(params.terrain_preset_id, "flat-debug")
        self.assertEqual(params.terrain_overrides["terrain.terrain_type"], "plane")
        self.assertEqual(params.terrain_overrides["terrain_curriculum_enable"], False)
        self.assertEqual(params.terrain_overrides["terrain_curriculum_levels"], [0.0])

    def test_training_params_accept_nested_v2_reward_overrides(self):
        params = TrainingParams.from_dict(
            {
                "task": "Template-Redrhex-Direct-v0",
                "num_envs": 4,
                "max_iterations": 8,
                "device": "cuda:0",
                "reward_overrides": {
                    "rew_scale_alive": "0.25",
                    "v2_reward_scales": {
                        "velocity_tracking": "5.5",
                        "energy_per_distance": 0.002,
                    },
                },
            }
        )

        self.assertEqual(params.reward_overrides["rew_scale_alive"], 0.25)
        self.assertEqual(params.reward_overrides["v2_reward_scales"]["velocity_tracking"], 5.5)
        self.assertEqual(params.reward_overrides["v2_reward_scales"]["energy_per_distance"], 0.002)

    def test_training_params_preserve_tweak_metadata_without_changing_argv(self):
        params = TrainingParams.from_dict(
            {
                "task": "Template-Redrhex-Direct-v0",
                "num_envs": 4,
                "max_iterations": 8,
                "device": "cuda:0",
                "tweak_source_run_id": "panel_123",
                "tweak_source_label": "Baseline trial",
            }
        )
        self.assertEqual(params.tweak_source_run_id, "panel_123")
        self.assertEqual(params.tweak_source_label, "Baseline trial")
        self.assertNotIn("tweak_source_run_id", " ".join(training_argv(params)))

    def test_training_params_preserve_requester_without_changing_argv(self):
        params = TrainingParams.from_dict(
            {
                "task": "Template-Redrhex-Direct-v0",
                "num_envs": 4,
                "max_iterations": 8,
                "device": "cuda:0",
                "requester_id": "11111111-1111-4111-8111-111111111111",
                "requester_label": "Jason",
            }
        )
        self.assertEqual(params.requester_id, "11111111-1111-4111-8111-111111111111")
        self.assertEqual(params.requester_label, "Jason")
        self.assertNotIn("requester_id", " ".join(training_argv(params)))

    def test_training_params_accept_display_name_without_changing_argv(self):
        params = TrainingParams.from_dict(
            {
                "task": "Template-Redrhex-Direct-v0",
                "num_envs": 4,
                "max_iterations": 8,
                "device": "cuda:0",
                "display_name": "  stair warmup  ",
            }
        )
        self.assertEqual(params.display_name, "stair warmup")
        self.assertEqual(params.to_dict()["display_name"], "stair warmup")
        self.assertNotIn("stair warmup", " ".join(training_argv(params)))

    def test_training_params_preserve_folder_and_client_request_id_without_changing_argv(self):
        params = TrainingParams.from_dict(
            {
                "task": "Template-Redrhex-Direct-v0",
                "num_envs": 4,
                "max_iterations": 8,
                "device": "cuda:0",
                "folder": "  tests  ",
                "client_request_id": "child-123",
            }
        )
        self.assertEqual(params.folder, "tests")
        self.assertEqual(params.client_request_id, "child-123")
        argv = " ".join(training_argv(params))
        self.assertNotIn("tests", argv)
        self.assertNotIn("child-123", argv)

    def test_training_params_reject_display_name_over_limit(self):
        with self.assertRaises(ValueError):
            TrainingParams.from_dict(
                {
                    "task": "Template-Redrhex-Direct-v0",
                    "num_envs": 4,
                    "max_iterations": 8,
                    "device": "cuda:0",
                    "display_name": "x" * 121,
                }
            )

    def test_sensor_v2_full_route_builds_sequential_pipeline_command(self):
        params = TrainingParams.from_dict(
            {
                "training_route": "sensor_v2_full",
                "num_envs": 64,
                "teacher_iterations": 1500,
                "distillation_iterations": 800,
                "ppo_iterations": 1500,
                "device": "cuda:0",
                "headless": True,
            }
        )
        argv = training_argv(params)
        self.assertEqual(params.task, "Template-Redrhex-ForwardSensorV2-Direct-v0")
        self.assertEqual(argv[0], "scripts/rsl_rl/train_sensor_v2_pipeline.py")
        self.assertEqual(argv[argv.index("--teacher_iterations") + 1], "1500")
        self.assertEqual(argv[argv.index("--distillation_iterations") + 1], "800")
        self.assertEqual(argv[argv.index("--ppo_iterations") + 1], "1500")
        self.assertNotIn("--panel_overrides", argv)

    def test_sensor_v2_stage_commands_use_explicit_legal_bootstrap_flags(self):
        teacher = TrainingParams.from_dict(
            {"training_route": "sensor_v2_teacher", "max_iterations": 5}
        )
        distillation = TrainingParams.from_dict(
            {
                "training_route": "sensor_v2_distillation",
                "checkpoint": "/tmp/teacher.pt",
                "resume": True,
                "max_iterations": 5,
            }
        )
        ppo = TrainingParams.from_dict(
            {
                "training_route": "sensor_v2_ppo",
                "checkpoint": "/tmp/distilled.pt",
                "resume": True,
                "max_iterations": 5,
            }
        )
        teacher_argv = training_argv(teacher)
        distillation_argv = training_argv(distillation)
        ppo_argv = training_argv(ppo)
        self.assertIn("rsl_rl_teacher_v2_cfg_entry_point", teacher_argv)
        self.assertEqual(distillation_argv[distillation_argv.index("--teacher_checkpoint") + 1], "/tmp/teacher.pt")
        self.assertEqual(ppo_argv[ppo_argv.index("--student_checkpoint") + 1], "/tmp/distilled.pt")
        self.assertNotIn("--resume", distillation_argv)
        self.assertNotIn("--resume", ppo_argv)

    def test_sensor_v2_play_selects_student_runner_explicitly(self):
        argv = play_argv(
            "/tmp/model_10.pt",
            task="Template-Redrhex-ForwardSensorV2-Direct-v0",
            agent="rsl_rl_ppo_v2_cfg_entry_point",
        )
        self.assertEqual(argv[argv.index("--agent") + 1], "rsl_rl_ppo_v2_cfg_entry_point")


if __name__ == "__main__":
    unittest.main()
