import importlib.util
import tempfile
import unittest
from pathlib import Path

from tools.training_panel.training_panel.config import PanelPaths
from tools.training_panel.training_panel.deploy import FALLBACK_CONTRACT, build_policy_manifest, run_deploy_validation
from tools.training_panel.training_panel.mujoco_rollout import (
    DEFAULT_PLAYBACK_FPS,
    DEFAULT_PLAYBACK_HEIGHT,
    DEFAULT_PLAYBACK_STEPS,
    DEFAULT_PLAYBACK_WIDTH,
    MujocoPlaybackConfig,
    MujocoScenario,
    contract_joint_to_mujoco,
    load_calibration_config,
    policy_decimation,
    prepare_mujoco_model,
    resolve_package_uris,
    run_mujoco_rollouts,
    scenario_by_name,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
MUJOCO_MODEL = REPO_ROOT / "test_7_description" / "test_7_description" / "urdf" / "test_7.urdf"
HAS_MUJOCO = importlib.util.find_spec("mujoco") is not None


class MujocoRolloutTests(unittest.TestCase):
    def make_paths(self, root: Path) -> PanelPaths:
        return PanelPaths(
            repo_root=REPO_ROOT,
            isaaclab_root=root / "IsaacLab",
            isaacsim_root=root / "isaacsim",
            conda_sh=root / "conda.sh",
            conda_env="env",
        )

    def make_run(self, root: Path) -> dict:
        run_dir = root / "logs" / "rsl_rl" / "redrhex_wheg" / "deploy_run"
        exported = run_dir / "exported"
        params = run_dir / "params"
        exported.mkdir(parents=True)
        params.mkdir(parents=True)
        checkpoint = run_dir / "model_10.pt"
        checkpoint.write_text("checkpoint", encoding="utf-8")
        (exported / "policy.onnx").write_text("fake onnx", encoding="utf-8")
        (exported / "policy.pt").write_text("fake torchscript", encoding="utf-8")
        (params / "env.yaml").write_text("env: test\n", encoding="utf-8")
        (params / "agent.yaml").write_text("agent: test\n", encoding="utf-8")
        return {
            "id": "deploy_run",
            "display_name": "Deploy Run",
            "log_dir": str(run_dir),
            "latest_checkpoint": str(checkpoint),
            "onnx_path": str(exported / "policy.onnx"),
        }

    def test_package_uri_resolution_and_joint_mapping(self):
        resolved = resolve_package_uris(
            'file="package://test_7_description/meshes/base_link.stl"',
            {"test_7_description": "/repo/test_7_description"},
        )
        self.assertIn("/repo/test_7_description/meshes/base_link.stl", resolved)
        self.assertEqual(contract_joint_to_mujoco("Revolute_15"), "Revolute 15")

    def test_playback_config_and_policy_decimation(self):
        self.assertEqual(policy_decimation(125.0, 0.002), 4)
        scenario = scenario_by_name("forward_mid", steps=DEFAULT_PLAYBACK_STEPS)
        self.assertEqual(scenario.name, "forward_mid")
        self.assertEqual(scenario.steps, DEFAULT_PLAYBACK_STEPS)
        config = MujocoPlaybackConfig(mode="record")
        self.assertEqual(config.width, DEFAULT_PLAYBACK_WIDTH)
        self.assertEqual(config.height, DEFAULT_PLAYBACK_HEIGHT)
        self.assertEqual(config.fps, DEFAULT_PLAYBACK_FPS)
        self.assertEqual(config.to_dict()["mode"], "record")

    @unittest.skipUnless(HAS_MUJOCO, "mujoco is not installed")
    def test_prepare_model_generates_mjcf_with_12_actuators(self):
        with tempfile.TemporaryDirectory() as tmp:
            artifact_dir = Path(tmp)
            config = load_calibration_config(repo_root=REPO_ROOT, model_path=MUJOCO_MODEL, contract=FALLBACK_CONTRACT)
            rollout_model, info = prepare_mujoco_model(model_path=MUJOCO_MODEL, artifact_dir=artifact_dir, config=config)
            self.assertTrue(rollout_model.is_file())
            self.assertEqual(info["nu"], 12)
            self.assertIn("Revolute_15_motor", info["actuators"])
            self.assertTrue((artifact_dir / "redrhex_resolved.urdf").is_file())
            text = rollout_model.read_text(encoding="utf-8")
            self.assertIn("deploy_mujoco_visual_floor", text)
            self.assertIn("deploy_mujoco_floor_grid", text)
            self.assertIn("deploy_mujoco_sky", text)

    @unittest.skipUnless(HAS_MUJOCO, "mujoco is not installed")
    def test_rollout_warns_until_calibration_is_explicit(self):
        with tempfile.TemporaryDirectory() as tmp:
            artifact_dir = Path(tmp)
            config = load_calibration_config(repo_root=REPO_ROOT, model_path=MUJOCO_MODEL, contract=FALLBACK_CONTRACT)
            report = run_mujoco_rollouts(
                repo_root=REPO_ROOT,
                model_path=MUJOCO_MODEL,
                policy_path=artifact_dir / "missing.onnx",
                artifact_dir=artifact_dir / "uncalibrated",
                config=config,
                obs_dim=56,
                action_dim=12,
                scenarios=[MujocoScenario("stand_zero", (0.0, 0.0, 0.0), steps=8)],
            )
            self.assertEqual(report.status, "warn")
            self.assertFalse(report.calibrated)
            self.assertEqual(report.scenarios[0].status, "pass")
            self.assertTrue(Path(report.artifacts["metrics"]).is_file())

            config.calibrated = True
            calibrated = run_mujoco_rollouts(
                repo_root=REPO_ROOT,
                model_path=MUJOCO_MODEL,
                policy_path=artifact_dir / "missing.onnx",
                artifact_dir=artifact_dir / "calibrated",
                config=config,
                obs_dim=56,
                action_dim=12,
                scenarios=[MujocoScenario("stand_zero", (0.0, 0.0, 0.0), steps=8)],
            )
            self.assertEqual(calibrated.status, "pass")
            self.assertTrue(calibrated.calibrated)

    @unittest.skipUnless(HAS_MUJOCO, "mujoco is not installed")
    def test_mujoco_only_deploy_report_writes_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = self.make_run(root)
            report = run_deploy_validation(
                self.make_paths(root),
                run,
                pipeline_id="mujoco_smoke",
                include_ros_mock=False,
                include_mujoco=True,
                mujoco_only=True,
                mujoco_model_path=str(MUJOCO_MODEL),
            )
            self.assertEqual(
                [stage.name for stage in report.stages],
                ["spring_calibration", "mujoco_readiness"],
            )
            self.assertEqual(report.stages[0].status, "fail")
            stage = report.stages[1]
            self.assertEqual(stage.status, "warn")
            self.assertTrue(Path(stage.artifacts["rollout_model"]).is_file())
            self.assertTrue(Path(stage.artifacts["metrics"]).is_file())


if __name__ == "__main__":
    unittest.main()
