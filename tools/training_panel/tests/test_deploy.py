import tempfile
import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from tools.training_panel.training_panel.config import PanelPaths
from tools.training_panel.training_panel.deploy import (
    _module_status,
    build_policy_manifest,
    deploy_defaults,
    run_deploy_validation,
    run_export_stage,
    validate_contract,
    validate_export_integrity,
    validate_safety_faults,
    validate_spring_calibration,
)


REPO_ROOT = Path(__file__).resolve().parents[3]


class DeployReadinessTests(unittest.TestCase):
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
        (params / "torsion_spring.yaml").write_text(
            "spring_backend: explicit\ncalibration_status: uncalibrated\n",
            encoding="utf-8",
        )
        return {
            "id": "deploy_run",
            "display_name": "Deploy Run",
            "log_dir": str(run_dir),
            "latest_checkpoint": str(checkpoint),
            "onnx_path": str(exported / "policy.onnx"),
        }

    def test_manifest_and_export_integrity_include_hashes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = build_policy_manifest(self.make_paths(root), self.make_run(root))
            self.assertEqual(manifest.run_id, "deploy_run")
            self.assertEqual(manifest.expected_obs_dim, 56)
            self.assertEqual(manifest.expected_action_dim, 12)
            self.assertIn("policy_onnx", manifest.hashes)
            self.assertIn("torsion_spring", manifest.hashes)
            self.assertGreater(manifest.sizes["policy_onnx"], 0)

            stage = validate_export_integrity(manifest)
            self.assertEqual(stage.status, "pass")
            self.assertIn("policy_onnx", stage.artifacts)

    def test_contract_and_safety_stages_run_without_optional_runtime_deps(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = self.make_paths(root)
            manifest = build_policy_manifest(paths, self.make_run(root))

            contract = validate_contract(paths, manifest)
            self.assertIn(contract.status, {"pass", "warn"})
            self.assertEqual(contract.details["expected_obs_dim"], 56)

            safety = validate_safety_faults(paths, manifest)
            self.assertEqual(safety.status, "pass")
            self.assertGreaterEqual(len(safety.details["cases"]), 5)

    def test_spring_calibration_is_a_fail_closed_deployment_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = build_policy_manifest(self.make_paths(root), self.make_run(root))

            rejected = validate_spring_calibration(manifest)
            self.assertEqual(rejected.status, "fail")
            self.assertIn("calibrated torsion-spring", rejected.summary)

            with patch(
                "tools.sim2real.checkpoint_spring.validate_checkpoint_spring_deployment",
                return_value={
                    "spring_backend": "native",
                    "calibration_status": "calibrated",
                    "profile_id": "measured-spring",
                    "profile_sha256": "a" * 64,
                },
            ):
                accepted = validate_spring_calibration(manifest)
            self.assertEqual(accepted.status, "pass")
            self.assertEqual(accepted.details["spring_backend"], "native")

    def test_deploy_defaults_expose_validation_runtime_dependencies(self):
        with tempfile.TemporaryDirectory() as tmp:
            defaults = deploy_defaults(self.make_paths(Path(tmp)))
            self.assertEqual(defaults["deploy_runtime_python"], sys.executable)
            self.assertIn("deploy_runtime_dependencies", defaults)
            for name in ("onnx", "onnxruntime", "mujoco", "torch"):
                self.assertIn(name, defaults["deploy_runtime_dependencies"])
                self.assertIn("installed", defaults["deploy_runtime_dependencies"][name])
                self.assertIn("version", defaults["deploy_runtime_dependencies"][name])

    def test_module_status_treats_missing_parent_package_as_unavailable(self):
        with patch(
            "tools.training_panel.training_panel.deploy.importlib.util.find_spec",
            side_effect=ModuleNotFoundError("No module named 'mujoco'"),
        ):
            status = _module_status("mujoco.viewer")

        self.assertFalse(status["installed"])
        self.assertEqual(status["version"], "")

    def test_deploy_validation_writes_report_even_when_fake_onnx_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = self.make_run(root)
            report = run_deploy_validation(
                self.make_paths(root),
                run,
                pipeline_id="test_pipeline",
                include_ros_mock=False,
                include_mujoco=False,
            )
            report_path = Path(report.artifacts["json_report"])
            markdown_path = Path(report.artifacts["markdown_report"])
            self.assertTrue(report_path.is_file())
            self.assertTrue(markdown_path.is_file())
            self.assertEqual(report.pipeline_id, "test_pipeline")
            self.assertTrue(any(stage.name == "export_integrity" for stage in report.stages))
            self.assertTrue(any(stage.name == "spring_calibration" for stage in report.stages))
            self.assertEqual(report.overall_status, "fail")
            self.assertEqual(report.runtime["python"], sys.executable)

    def test_export_stage_propagates_resolved_native_backend(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = self.make_paths(root)
            paths.isaaclab_root.mkdir()
            paths.isaaclab_launcher.write_text("#!/usr/bin/env bash\n", encoding="utf-8")
            run = self.make_run(root)
            run["params"] = {
                "task": "Template-Redrhex-ForwardFast-Direct-v0",
                "spring_backend": "native",
            }

            with patch(
                "tools.training_panel.training_panel.deploy.export_onnx_argv",
                return_value=["scripts/rsl_rl/play.py", "--spring-backend", "native"],
            ) as export, patch(
                "tools.training_panel.training_panel.deploy.subprocess.run",
                return_value=Mock(returncode=0, stdout=""),
            ):
                stage = run_export_stage(paths, run, device="cpu")

            self.assertEqual(stage.status, "pass")
            self.assertEqual(export.call_args.kwargs["task"], "Template-Redrhex-ForwardFast-Direct-v0")
            self.assertEqual(export.call_args.kwargs["spring_backend"], "native")


if __name__ == "__main__":
    unittest.main()
