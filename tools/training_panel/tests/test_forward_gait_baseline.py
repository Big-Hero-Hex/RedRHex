from __future__ import annotations

import ast
import importlib.util
import json
import math
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]


def _load_validator_module():
    path = REPO_ROOT / "scripts" / "rsl_rl" / "validate_forward_gait_baseline.py"
    spec = importlib.util.spec_from_file_location("redrhex_forward_gait_f0_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_f0_structural_report_accepts_contract_bound_v2_reset() -> None:
    validator = _load_validator_module()

    report = validator.build_report(REPO_ROOT)
    checks = {check["name"]: check for check in report["checks"]}

    assert checks["canonical_six_joint_order"]["status"] == "PASS"
    assert checks["direction_multipliers"]["status"] == "PASS"
    assert checks["tripod_partition"]["status"] == "PASS"
    assert checks["timing_and_rates"]["status"] == "PASS"
    assert checks["configured_reset_contract_parity"]["status"] == "PASS"
    assert checks["shared_decoder_parity"]["status"] == "PASS"
    assert checks["shared_decoder_parity"]["details"][
        "velocity_limit_saturation_exercised"
    ] is True
    assert checks["reset_effective_phase_coherence"]["status"] == "PASS"
    assert checks["time_warped_duty_cycle"]["status"] == "PASS"
    assert checks["main_drive_velocity_limit_binding"]["status"] == "PASS"
    assert checks["reset_effective_phase_coherence"]["details"][
        "tripod_separation_rad"
    ] == pytest.approx(0.0)
    assert checks["reset_effective_phase_coherence"]["details"][
        "tripod_a_mean_rad"
    ] == pytest.approx(-math.pi / 4.0)
    phase_lock = checks["initial_phase_lock_error_bound"]
    assert phase_lock["status"] == "PASS"
    assert phase_lock["details"]["maximum_abs_error_rad"] == pytest.approx(
        1.067335965642686
    )
    assert phase_lock["details"]["correction_saturated"] is False
    duty = checks["time_warped_duty_cycle"]["details"]
    assert duty["tripod_a_observed_duty"] == pytest.approx(0.65)
    assert duty["tripod_b_observed_duty"] == pytest.approx(0.65)
    assert duty["tripod_overlap_fraction"] == pytest.approx(0.30)
    assert duty["continuous_tripod_support_fraction"] == pytest.approx(1.0)
    velocity_limit = checks["main_drive_velocity_limit_binding"]["details"]
    assert velocity_limit["contract_limit_rad_s"] == pytest.approx(15.0)
    assert velocity_limit["physics_limit_rad_s"] == pytest.approx(15.0)
    assert report["overall_status"] == "PASS"
    assert report["simulator_rollout"]["status"] == "NOT_RUN"


def test_f0_cli_writes_machine_readable_structural_pass(tmp_path: Path) -> None:
    validator = _load_validator_module()
    report_path = tmp_path / "f0.json"

    returncode = validator.main(["--json", str(report_path)])

    assert returncode == 0
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "redrhex.forward-gait-f0.v2"
    assert payload["overall_status"] == "PASS"
    assert payload["structural_status"] == "PASS"
    assert payload["simulator_rollout"]["status"] == "NOT_RUN"
    assert payload["simulator_rollout"]["requested"] is False
    assert payload["gate_configuration"]["commands_vx_m_s"] == [0.22, 0.35, 0.42]
    assert all(set(check) == {"name", "status", "details"} for check in payload["checks"])


def test_f0_isaac_request_skips_launch_while_structural_gate_fails(
    tmp_path: Path,
    monkeypatch,
) -> None:
    validator = _load_validator_module()
    report_path = tmp_path / "f0-isaac-skipped.json"
    monkeypatch.setattr(
        validator,
        "build_report",
        lambda: {
            "schema_version": "redrhex.forward-gait-f0.v2",
            "overall_status": "FAIL",
            "checks": [
                {
                    "name": "reset_effective_phase_coherence",
                    "status": "FAIL",
                    "details": {},
                }
            ],
            "simulator_rollout": {"status": "NOT_RUN", "reason": "test fixture"},
        },
    )

    returncode = validator.main(
        ["--json", str(report_path), "--isaac", "--headless"]
    )

    assert returncode == 2
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["structural_status"] == "FAIL"
    assert payload["overall_status"] == "FAIL"
    assert payload["simulator_rollout"]["status"] == "SKIPPED"
    assert payload["simulator_rollout"]["requested"] is True
    assert payload["simulator_rollout"]["failed_structural_checks"] == [
        "reset_effective_phase_coherence"
    ]
    assert "not launched" in payload["simulator_rollout"]["reason"]


def test_f0_sources_forward_thresholds_from_command_sweep() -> None:
    validator = _load_validator_module()

    assert validator._load_forward_acceptance_thresholds(REPO_ROOT) == {
        "forward_abs_m_s": 0.15,
        "forward_command_ratio": 0.55,
        "lateral_leak_m_s": 0.12,
        "yaw_leak_rad_s": 0.30,
        "forward_tilt_bound_rad": 0.70,
        "forward_min_base_height_m": 0.085,
        "max_fall_rate": 0.20,
        "accept_duration_s": 2.0,
        "contiguous_env_ratio": 0.50,
    }


def test_v2_reset_writes_contract_pose_exactly_without_changing_v1_source() -> None:
    validator = _load_validator_module()
    snapshot = validator._load_snapshot(REPO_ROOT)
    expected_v2 = (
        math.pi / 4.0,
        math.pi / 4.0,
        math.pi / 4.0,
        -math.pi / 4.0,
        -math.pi / 4.0,
        -math.pi / 4.0,
    )
    expected_v1 = (
        math.pi / 4.0,
        math.pi / 4.0,
        math.pi / 4.0,
        -math.pi / 4.0,
        -math.pi / 4.0,
        -math.pi / 4.0,
    )
    assert snapshot["reset_main_position_rad"] == pytest.approx(expected_v2)
    assert snapshot["legacy_reset_main_position_rad"] == pytest.approx(expected_v1)

    env_path = (
        REPO_ROOT
        / "source"
        / "RedRhex"
        / "RedRhex"
        / "tasks"
        / "direct"
        / "redrhex"
        / "redrhex_sensor_v2_env.py"
    )
    tree = ast.parse(env_path.read_text(encoding="utf-8"))
    reset_method = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "_reset_idx"
    )
    reset_source = ast.unparse(reset_method)
    assert "self._action_contract_v2.initial_main_position_rad" in reset_source
    assert "self.robot.write_joint_state_to_sim" in reset_source
    assert "reset_main_velocity = torch.zeros_like(reset_main_position)" in reset_source
    assert "self.joint_pos[ids[:, None], self._main_drive_indices]" in reset_source
    assert "self.joint_vel[ids[:, None], self._main_drive_indices]" in reset_source
    assert "sample_uniform" not in reset_source


def test_f0_isaac_wiring_is_lazy_and_reports_required_metrics() -> None:
    path = REPO_ROOT / "scripts" / "rsl_rl" / "validate_forward_gait_baseline.py"
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    top_level_imports = [
        node
        for node in tree.body
        if isinstance(node, (ast.Import, ast.ImportFrom))
    ]

    assert all("isaac" not in ast.unparse(node).lower() for node in top_level_imports)
    assert "F0_FORWARD_COMMANDS_M_S = (0.22, 0.35, 0.42)" in source
    assert "bind_redrhex_source(REPO_ROOT)" in source
    assert "assert_redrhex_module_source(redrhex_tasks, REPO_ROOT)" in source
    assert "from isaaclab.app import AppLauncher" in source
    assert "simulation_app.close()" in source
    assert "os._exit(status)" in source
    assert "env.step(zero_actions)" in source
    assert "unwrapped.cfg.play_forward_compat_enable = False" in source
    assert "_reference_relative_tilt(unwrapped.projected_gravity, reference)" in source
    assert "_target_abad_pos" in source and "_abad_rest_pos" in source
    for field in (
        "mean_forward_displacement_m",
        "actual_forward_speed_mean_m_s",
        "actual_lateral_leak_mean_m_s",
        "actual_yaw_leak_mean_rad_s",
        "reference_relative_tilt_mean_rad",
        "reference_relative_tilt_max_rad",
        "base_height_mean_m",
        "base_height_min_m",
        "fall_rate",
    ):
        assert f'"{field}"' in source
