import argparse
import tempfile
import time
import unittest
from datetime import datetime
from pathlib import Path
from unittest import mock

from tools.training_panel.training_panel.config import PanelPaths
from tools.training_panel.training_panel import smoke_pipeline
from tools.training_panel.training_panel.smoke_pipeline import (
    SmokeConfig,
    SmokeValidationError,
    build_export_command,
    build_train_command,
    build_video_command,
    command_display,
    default_run_name,
    exact_experiment_name,
    run_smoke_pipeline,
    validate_export,
    validate_smoke_run,
    validate_video,
)


class SmokePipelineTests(unittest.TestCase):
    def make_paths(self, root: Path) -> PanelPaths:
        return PanelPaths(
            repo_root=root,
            isaaclab_root=root / "IsaacLab",
            isaacsim_root=root / "isaacsim",
            conda_sh=root / "conda.sh",
            conda_env="env",
        )

    def make_run(self, root: Path, run_id: str = "2026-06-13_12-00-00_smoke_case") -> Path:
        run = root / "logs" / "rsl_rl" / "redrhex_wheg" / run_id
        (run / "params").mkdir(parents=True)
        (run / "model_0.pt").write_text("checkpoint", encoding="utf-8")
        (run / "events.out.tfevents.test").write_text("events", encoding="utf-8")
        (run / "params" / "env.yaml").write_text("env: {}\n", encoding="utf-8")
        (run / "params" / "agent.yaml").write_text("agent: {}\n", encoding="utf-8")
        return run

    def test_default_run_name_is_disposable_and_timestamped(self):
        now = datetime(2026, 6, 13, 14, 5, 6)
        self.assertEqual(default_run_name(now), "disposable_smoke_20260613_140506")

    def test_exact_experiment_name_uses_last_match(self):
        output = "\n".join(
            [
                "Exact experiment name requested from command line: old",
                "other line",
                "Exact experiment name requested from command line: 2026-06-13_12-00-00",
            ]
        )
        self.assertEqual(exact_experiment_name(output), "2026-06-13_12-00-00")

    def test_build_train_command_has_disposable_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = self.make_paths(Path(tmp))
            config = SmokeConfig(run_name="smoke_case")
            argv = build_train_command(paths, config, ["--foo", "bar"])

            self.assertEqual(argv[0], str(paths.isaaclab_launcher))
            self.assertIn("scripts/rsl_rl/train.py", argv)
            self.assertIn("--experiment_name", argv)
            self.assertIn("redrhex_wheg", argv)
            self.assertIn("--run_name", argv)
            self.assertIn("smoke_case", argv)
            self.assertIn("--headless", argv)
            self.assertEqual(argv[-2:], ["--foo", "bar"])

    def test_build_export_and_video_commands_target_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = self.make_paths(Path(tmp))
            config = SmokeConfig(
                run_name="smoke_case",
                video_length=60,
                video_width=320,
                video_height=240,
                video_fps=24,
            )
            checkpoint = str(Path(tmp) / "logs" / "rsl_rl" / "redrhex_wheg" / "run" / "model_0.pt")

            export = build_export_command(paths, config, checkpoint)
            video = build_video_command(paths, config, checkpoint)

            self.assertIn("--export_policy_only", export)
            self.assertEqual(export[-1], checkpoint)
            self.assertIn("--video", video)
            self.assertIn("--disable_keyboard_control", video)
            self.assertIn("60", video)
            self.assertEqual(video[-1], checkpoint)

    def test_command_display_quotes_spaces(self):
        self.assertEqual(
            command_display(["python", "path with space.py", "--name", "a b"]),
            "python 'path with space.py' --name 'a b'",
        )

    def test_validate_smoke_run_discovers_artifacts_and_history(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = self.make_run(root)
            paths = self.make_paths(root)
            config = SmokeConfig(run_name="smoke_case")
            output = "Exact experiment name requested from command line: 2026-06-13_12-00-00\n"

            result = validate_smoke_run(paths, config, output=output, start_time=time.time() - 10)

            self.assertTrue(result["ok"])
            self.assertEqual(result["run_dir"], str(run))
            self.assertTrue(result["checkpoint"].endswith("model_0.pt"))
            self.assertEqual(result["local_history_run_id"], run.name)

    def test_validate_smoke_run_requires_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = self.make_run(root)
            (run / "model_0.pt").unlink()
            paths = self.make_paths(root)
            config = SmokeConfig(run_name="smoke_case")
            output = "Exact experiment name requested from command line: 2026-06-13_12-00-00\n"

            with self.assertRaisesRegex(SmokeValidationError, "missing model_\\*.pt checkpoint"):
                validate_smoke_run(paths, config, output=output, start_time=time.time() - 10)

    def test_validate_export_and_video_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = self.make_run(Path(tmp))
            exported = run / "exported"
            exported.mkdir()
            (exported / "policy.pt").write_text("jit", encoding="utf-8")
            (exported / "policy.onnx").write_text("onnx", encoding="utf-8")
            video_dir = run / "videos" / "play"
            video_dir.mkdir(parents=True)
            (video_dir / "rl-video-step-0.mp4").write_text("mp4", encoding="utf-8")

            self.assertTrue(validate_export(run)["policy_pt"].endswith("policy.pt"))
            self.assertTrue(validate_export(run)["policy_onnx"].endswith("policy.onnx"))
            self.assertTrue(validate_video(run)["video"].endswith(".mp4"))

    def test_dry_run_does_not_require_isaac_launcher(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = self.make_paths(Path(tmp))
            args = argparse.Namespace(
                task="Template-Redrhex-Direct-v0",
                experiment_name="redrhex_wheg",
                run_name="dry_check",
                num_envs=4,
                max_iterations=1,
                device="cuda:0",
                seed=1,
                no_headless=False,
                include_export=False,
                include_video=False,
                video_length=120,
                video_width=640,
                video_height=360,
                video_fps=30,
                timeout_seconds=1800,
                panel_url="",
                run_dir="",
                validate_only=False,
                dry_run=True,
                summary_file="",
                extra_train_args=[],
            )

            with mock.patch.object(smoke_pipeline, "_paths_from_env", return_value=paths):
                payload = run_smoke_pipeline(args)

            self.assertTrue(payload["dry_run"])
            self.assertEqual(payload["config"]["run_name"], "dry_check")
            self.assertIn("scripts/rsl_rl/train.py", payload["commands"]["train"])


if __name__ == "__main__":
    unittest.main()
