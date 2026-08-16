from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).parents[3]


def _load_module():
    path = ROOT / "scripts/rsl_rl/train_sensor_v2_full_pipeline.py"
    spec = importlib.util.spec_from_file_location("redrhex_sensor_v2_full_pipeline_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, payload: dict) -> str:
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _profile(purpose: str, *, artifact: str, evidence_sha256: str) -> dict:
    if purpose == "training_curriculum":
        parameters = {
            "sensor_dr_encoder_latency_steps_range": [0, 1],
            "sim2real_command_delay_steps": 1,
            "domain_randomization_enable": True,
            "dr_randomize_actuator_strength": True,
            "dr_main_actuator_strength_range": [0.9, 1.1],
            "dr_abad_actuator_strength_range": [0.92, 1.08],
        }
    else:
        parameters = {
            "sensor_dr_gyro_noise_std_range_rad_s": [0.001, 0.003],
            "sensor_dr_encoder_latency_steps_range": [1, 2],
            "sim2real_command_delay_steps": 2,
            "domain_randomization_enable": True,
            "dr_try_physical_material_randomization": True,
            "dr_randomize_actuator_strength": True,
            "dr_main_actuator_strength_range": [0.75, 1.25],
            "dr_abad_actuator_strength_range": [0.8, 1.2],
            "dr_randomize_friction": True,
            "dr_friction_range": [0.85, 1.15],
        }
    return {
        "schema": "redrhex.sensor-dr-profile.v2",
        "profile_id": f"test-{purpose}",
        "purpose": purpose,
        "evidence": [
            {"artifact": artifact, "sha256": evidence_sha256, "note": "test evidence"}
        ],
        "parameters": parameters,
    }


def _f0_payload(pipeline) -> dict:
    structural_names = (
        "canonical_six_joint_order",
        "direction_multipliers",
        "tripod_partition",
        "reset_effective_phase_coherence",
        "initial_phase_lock_error_bound",
        "time_warped_duty_cycle",
        "main_drive_velocity_limit_binding",
        "timing_and_rates",
        "configured_reset_contract_parity",
        "shared_decoder_parity",
    )
    validator = pipeline._f0_validator_module()
    action_contract = validator._shared_action_contract(
        ROOT, validator._load_snapshot(ROOT)
    )
    policy_step_dt_s = 1.0 / action_contract.policy_rate_hz
    commands = []
    row_check_names = (
        "finite_metrics",
        "forward_speed",
        "lateral_leak",
        "yaw_leak",
        "reference_relative_tilt",
        "base_height",
        "fall_rate",
        "contiguous_success",
        "forward_mae",
        "forward_displacement",
        "neutral_abad_target",
    )
    for command_vx in (0.22, 0.35, 0.42):
        commands.append(
            {
                "command_vx_m_s": command_vx,
                "status": "PASS",
                "sample_count": 1920,
                "mean_forward_displacement_m": 1.0,
                "actual_forward_speed_mean_m_s": command_vx,
                "actual_lateral_leak_mean_m_s": 0.0,
                "actual_yaw_leak_mean_rad_s": 0.0,
                "reference_relative_tilt_mean_rad": 0.0,
                "reference_relative_tilt_max_rad": 0.0,
                "base_height_mean_m": 0.1,
                "base_height_min_m": 0.1,
                "fall_events": 0,
                "episode_ends": 0,
                "fall_rate": 0.0,
                "forward_mae_m_s": 0.0,
                "success_sample_ratio": 1.0,
                "contiguous_success_env_ratio": 1.0,
                "gait_cycle_window_steps": action_contract.command_scaled_cycle_steps(
                    command_vx
                ),
                "gait_cycle_window_duration_s": (
                    action_contract.command_scaled_cycle_steps(command_vx)
                    * policy_step_dt_s
                ),
                "contiguous_success_semantics": (
                    "one_command_scaled_gait_cycle_velocity_means_with_"
                    "pointwise_tilt_height_and_episode_boundary_safety"
                ),
                "max_abad_target_offset_rad": 0.0,
                "checks": [
                    {"name": name, "status": "PASS"} for name in row_check_names
                ],
            }
        )
    thresholds = validator._load_forward_acceptance_thresholds()
    return {
        "schema_version": "redrhex.forward-gait-f0.v2",
        "overall_status": "PASS",
        "structural_status": "PASS",
        "provenance": pipeline._current_f0_provenance(),
        "checks": [{"name": name, "status": "PASS"} for name in structural_names],
        "gate_configuration": {
            "isaac_requested": True,
            "task": "Template-Redrhex-ForwardSensorV2-Direct-v0",
            "num_envs": 8,
            "seed": 42,
            "settle_steps": 120,
            "warmup_steps": 120,
            "measurement_steps": 240,
            "commands_vx_m_s": [0.22, 0.35, 0.42],
            "spring_backend": "native",
        },
        "simulator_rollout": {
            "status": "PASS",
            "requested": True,
            "mode": "isaac_zero_residual",
            "task": "Template-Redrhex-ForwardSensorV2-Direct-v0",
            "zero_residual_actions": True,
            "neutral_abad_actions": True,
            "spring_backend": "native",
            "seed": 42,
            "num_envs": 8,
            "policy_step_dt_s": policy_step_dt_s,
            "settle_steps": 120,
            "warmup_steps": 120,
            "measurement_steps": 240,
            "threshold_source": "scripts/rsl_rl/eval_command_sweep.py",
            "thresholds": thresholds,
            "commands_vx_m_s": [0.22, 0.35, 0.42],
            "commands": commands,
        },
    }


def test_f0_requires_both_structural_and_simulator_pass(tmp_path: Path) -> None:
    pipeline = _load_module()
    report = tmp_path / "f0.json"
    digest = _write_json(report, _f0_payload(pipeline))
    assert pipeline.validate_f0_evidence(report, digest)["sha256"] == digest

    not_run = _f0_payload(pipeline)
    not_run["simulator_rollout"] = {"status": "NOT_RUN"}
    digest = _write_json(report, not_run)
    try:
        pipeline.validate_f0_evidence(report, digest)
    except pipeline.PipelineGateError as exc:
        assert "Isaac simulator rollout PASS" in str(exc)
    else:
        raise AssertionError("structural-only F0 must not promote training")

    legacy = _f0_payload(pipeline)
    legacy["schema_version"] = "redrhex.forward-gait-f0.v1"
    digest = _write_json(report, legacy)
    try:
        pipeline.validate_f0_evidence(report, digest)
    except pipeline.PipelineGateError as exc:
        assert "gait-cycle-aware validator" in str(exc)
    else:
        raise AssertionError("legacy instantaneous-sample F0 evidence must migrate")

    fabricated = _f0_payload(pipeline)
    fabricated["simulator_rollout"]["commands"][0][
        "actual_forward_speed_mean_m_s"
    ] = -100.0
    fabricated["simulator_rollout"]["commands"][0][
        "actual_lateral_leak_mean_m_s"
    ] = 100.0
    digest = _write_json(report, fabricated)
    try:
        pipeline.validate_f0_evidence(report, digest)
    except pipeline.PipelineGateError as exc:
        assert "incomplete or not PASS" in str(exc)
    else:
        raise AssertionError("caller-supplied PASS labels must not override failed metrics")

    impossible = _f0_payload(pipeline)
    for row in impossible["simulator_rollout"]["commands"]:
        row["actual_lateral_leak_mean_m_s"] = -100.0
        row["actual_yaw_leak_mean_rad_s"] = -100.0
        row["reference_relative_tilt_mean_rad"] = -100.0
        row["reference_relative_tilt_max_rad"] = -100.0
        row["base_height_mean_m"] = -100.0
    digest = _write_json(report, impossible)
    try:
        pipeline.validate_f0_evidence(report, digest)
    except pipeline.PipelineGateError as exc:
        assert "incomplete or not PASS" in str(exc)
    else:
        raise AssertionError("physically impossible negative F0 metrics must not promote")


def test_profile_comparison_ignores_neutral_fields_but_keeps_active_distributions() -> None:
    pipeline = _load_module()
    training = SimpleNamespace(
        parameters={
            "sensor_dr_gyro_noise_std_range_rad_s": (0.001, 0.003),
            "sensor_dr_encoder_latency_steps_range": (1, 2),
            "sim2real_command_delay_steps": 2,
            "domain_randomization_enable": True,
            "dr_randomize_friction": True,
            "dr_friction_range": (0.85, 1.15),
        }
    )
    same_with_neutral_fields = SimpleNamespace(
        parameters={
            **training.parameters,
            "sensor_dr_gyro_bias_range_rad_s": (0.0, 0.0),
            "dr_randomize_mass": False,
            "dr_mass_range": (1.0, 1.0),
        }
    )
    held_out = SimpleNamespace(
        parameters={
            **training.parameters,
            "sensor_dr_encoder_latency_steps_range": (2, 3),
        }
    )

    canonical = pipeline._canonical_active_profile_parameters
    assert canonical(training) == canonical(same_with_neutral_fields)
    assert canonical(training) != canonical(held_out)


def test_f5_rejects_reused_required_category_distributions() -> None:
    pipeline = _load_module()
    training = SimpleNamespace(
        parameters={
            "sensor_dr_encoder_latency_steps_range": (1, 2),
            "sim2real_command_delay_steps": 2,
        }
    )
    held_out = SimpleNamespace(
        parameters={
            **training.parameters,
            "sensor_dr_gyro_noise_std_range_rad_s": (0.001, 0.003),
            "dr_friction_range": (0.85, 1.15),
        }
    )
    try:
        pipeline._validate_held_out_category_distributions(training, held_out)
    except pipeline.PipelineGateError as exc:
        assert "unchanged categories" in str(exc)
        assert "latency" in str(exc) or "actuator" in str(exc)
    else:
        raise AssertionError("F5 cannot relabel F4 actuator/latency ranges as held out")


def test_dry_run_plans_three_seed_f0_to_f5_with_distinct_profiles(tmp_path: Path) -> None:
    pipeline = _load_module()
    f0 = tmp_path / "f0.json"
    f4 = tmp_path / "f4.json"
    f5 = tmp_path / "f5.json"
    f4_evidence = tmp_path / "f4_bench.json"
    f5_evidence = tmp_path / "f5_bench.json"
    f4_evidence_hash = _write_json(f4_evidence, {"domain": "training"})
    f5_evidence_hash = _write_json(f5_evidence, {"domain": "held-out"})
    f0_hash = _write_json(f0, _f0_payload(pipeline))
    f4_hash = _write_json(
        f4,
        _profile(
            "training_curriculum",
            artifact=f4_evidence.name,
            evidence_sha256=f4_evidence_hash,
        ),
    )
    f5_hash = _write_json(
        f5,
        _profile(
            "held_out_evaluation",
            artifact=f5_evidence.name,
            evidence_sha256=f5_evidence_hash,
        ),
    )
    output = tmp_path / "pipeline.json"

    assert pipeline.main(
        [
            "--isaaclab-launcher",
            str(tmp_path / "isaaclab.sh"),
            "--f0-evidence",
            str(f0),
            "--f0-evidence-sha256",
            f0_hash,
            "--f4-profile",
            str(f4),
            "--f4-profile-sha256",
            f4_hash,
            "--f5-profile",
            str(f5),
            "--f5-profile-sha256",
            f5_hash,
            "--pipeline-id",
            "three-seed-test",
            "--output",
            str(output),
            "--dry-run",
        ]
    ) == 0

    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["status"] == "planned"
    assert result["deployment_eligible"] is False
    assert result["seeds"] == [42, 43, 44]
    assert len(result["runs"]) == 12
    assert len(result["evaluations"]) == 15
    commands = [" ".join(command) for command in result["planned_commands"]]
    assert any("rsl_rl_robust_ppo_v2_cfg_entry_point" in command for command in commands)
    assert sum("--sensor-dr-profile" in command for command in commands) == 6
    assert all("--spring-backend native" in command for command in commands)
