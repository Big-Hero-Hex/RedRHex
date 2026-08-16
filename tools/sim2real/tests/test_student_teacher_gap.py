from __future__ import annotations

import ast
import csv
from dataclasses import dataclass
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from redrhex_policy_io import (
    canonical_sha256,
    ForwardResidualActionContractV2,
    SensorCalibrationProfileV2,
    StudentObservationContractV2,
)
from tools.sim2real.generate_sensor_v2_promotion_gates import generate_promotion_gates
from tools.sim2real import replay_student_observation_v2 as replay_module
from tools.sim2real.import_sensor_v2_rosbag import (
    CAPTURE_ATTESTATION_SCHEMA_V2,
    JOINT_ORDER_V2,
    MAX_IMU_JOINT_SKEW_S_V2,
    MAX_PERIOD_ERROR_RATIO_V2,
    REQUIRED_TOPIC_TYPES_V2,
    SAMPLE_RATE_HZ_V2,
    TIMESTAMP_SEMANTICS_V2,
    sha256_path_v2,
)

from tools.sim2real.student_teacher_gap import (
    CANONICAL_ACCEPTANCE_PROTOCOL,
    EXPECTED_POLICY_IDENTITY,
    PROMOTION_GATES,
    REQUIRED_ABLATIONS,
    StudentTeacherGapError,
    _resolve_promotion_artifact,
    _resolve_run,
    evaluate_student_teacher_gap,
    write_gap_tensorboard,
)


_REAL_REPLAY_LOAD_BUNDLE = replay_module._load_bundle


_OBSERVATION = StudentObservationContractV2.validated_quaternion()
_ACTION = ForwardResidualActionContractV2()
_TRAINING_CALIBRATION = SensorCalibrationProfileV2.provisional(
    _OBSERVATION, _ACTION
)
_RUNTIME_CALIBRATION = SensorCalibrationProfileV2(
    profile_id="measured-hardware-v2",
    observation_contract_sha256=_OBSERVATION.sha256,
    action_contract_sha256=_ACTION.sha256,
    attitude_mode=_OBSERVATION.attitude_mode,
    imu_frame_id=_OBSERVATION.imu_frame_id,
    imu_to_body_wxyz=_OBSERVATION.imu_to_body_wxyz,
    main_counts_per_rad=(1000.0,) * 6,
    abad_counts_per_rad=(1000.0,) * 6,
    main_encoder_evidence=("measured-main",) * 6,
    abad_encoder_evidence=("measured-abad",) * 6,
    imu_mount_evidence="measured-mount",
    rest_gravity_evidence="measured-rest-gravity",
)
CONTRACT_HASHES = {
    "observation_contract_sha256": _OBSERVATION.sha256,
    "action_contract_sha256": _ACTION.sha256,
    "training_calibration_sha256": _TRAINING_CALIBRATION.sha256,
}
RUNTIME_CALIBRATION_SHA256 = _RUNTIME_CALIBRATION.sha256
_V2_FORWARD_CONTIGUOUS_SEMANTICS = (
    "one_command_scaled_gait_cycle_velocity_means_with_"
    "pointwise_tilt_height_and_episode_boundary_safety"
)
_LEGACY_FORWARD_CONTIGUOUS_SEMANTICS = "instantaneous_samples"


@dataclass
class _StructuralReplayRunner:
    observation_contract: StudentObservationContractV2
    action_contract: ForwardResidualActionContractV2
    bindings: dict[str, str]
    training_seed: int = 42

    @property
    def metadata(self) -> dict[str, object]:
        return {
            "calibration_sha256": self.bindings["runtime_calibration_sha256"],
            "training_calibration_sha256": self.bindings[
                "training_calibration_sha256"
            ],
            "checkpoint_sha256": self.bindings["checkpoint_sha256"],
            "architecture_sha256": self.bindings["architecture_sha256"],
            "config_sha256": self.bindings["config_sha256"],
            "canonical_config_sha256": self.bindings[
                "canonical_config_sha256"
            ],
            "training_seed": self.training_seed,
        }

    @property
    def runtime_calibration_sha256(self) -> str:
        return str(self.metadata["calibration_sha256"])

    @property
    def training_calibration_sha256(self) -> str:
        return str(self.metadata["training_calibration_sha256"])

    @property
    def checkpoint_sha256(self) -> str:
        return str(self.metadata["checkpoint_sha256"])

    @property
    def canonical_config_sha256(self) -> str:
        return str(self.metadata["canonical_config_sha256"])

    def run(self, sensor_history: np.ndarray, command: np.ndarray) -> object:
        assert sensor_history.shape == (60, 36)
        assert command.shape == (3,)
        return SimpleNamespace(
            actions=np.zeros(12, dtype=np.float32),
            base_velocity_estimate=np.zeros(3, dtype=np.float32),
        )


@pytest.fixture(autouse=True)
def _dependency_light_sensor_replay_bundle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep gap tests independent of ROS/ONNX; production uses the real loader."""

    def load_bundle(
        _onnx_path: Path,
        sidecar_path: Path,
        *,
        require_hardware_ready: bool,
    ) -> tuple[object, object, object, object]:
        payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
        contract = StudentObservationContractV2.from_dict(payload["contract"])
        action_contract = ForwardResidualActionContractV2.from_dict(
            payload["action_contract"]
        )
        calibration = SensorCalibrationProfileV2.from_dict(payload["calibration"])
        calibration.validate(require_hardware_ready=require_hardware_ready)
        metadata = payload["metadata"]
        bindings = {
            "observation_contract_sha256": metadata["contract_sha256"],
            "action_contract_sha256": metadata["action_contract_sha256"],
            "runtime_calibration_sha256": metadata["calibration_sha256"],
            "training_calibration_sha256": metadata[
                "training_calibration_sha256"
            ],
            "checkpoint_sha256": metadata["checkpoint_sha256"],
            "architecture_sha256": metadata["architecture_sha256"],
            "config_sha256": metadata["config_sha256"],
            "canonical_config_sha256": metadata["canonical_config_sha256"],
        }
        return (
            _StructuralReplayRunner(
                contract,
                action_contract,
                bindings,
                training_seed=int(metadata["training_seed"]),
            ),
            contract,
            action_contract,
            calibration,
        )

    monkeypatch.setattr(replay_module, "_load_bundle", load_bundle)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _run_identity(policy: str, seed: int, domain: str = "flat") -> dict[str, object]:
    agent, kind, stage = EXPECTED_POLICY_IDENTITY[policy]
    identity: dict[str, object] = {
        "agent_entry_point": agent,
        "checkpoint_kind": kind,
        "checkpoint_stage": stage,
        "checkpoint_sha256": (
            hashlib.sha256(b"hash-bound-checkpoint").hexdigest()
            if policy == "v2_ppo" and seed == 42
            else _digest(f"{policy}:{seed}:checkpoint")
        ),
        "config_sha256": (
            "f" * 64
            if policy == "v2_ppo" and seed == 42
            else _digest(f"{policy}:{seed}:config")
        ),
        "evaluation_protocol_sha256": _digest(f"{domain}:protocol"),
    }
    if policy != "legacy_student":
        identity.update(
            CONTRACT_HASHES,
            canonical_config_sha256=_digest(f"{policy}:canonical-config"),
            training_seed=seed,
            architecture_sha256=(
                "e" * 64
                if policy == "v2_ppo"
                else _digest(f"{policy}:architecture")
            ),
        )
    return identity


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _held_out_profile(root: Path) -> tuple[Path, dict[str, object]]:
    evidence = root / "fixture-held-out-evidence.json"
    evidence.write_text('{"source":"fixture bench"}\n', encoding="utf-8")
    parameters: dict[str, object] = {
        "sensor_dr_gyro_noise_std_range_rad_s": [0.002, 0.004],
        "sensor_dr_encoder_latency_steps_range": [2, 3],
        "sim2real_command_delay_steps": 3,
        "domain_randomization_enable": True,
        "dr_try_physical_material_randomization": True,
        "dr_randomize_friction": True,
        "dr_friction_range": [0.75, 1.25],
        "dr_randomize_actuator_strength": True,
        "dr_main_actuator_strength_range": [0.8, 1.2],
        "dr_abad_actuator_strength_range": [0.85, 1.15],
    }
    payload: dict[str, object] = {
        "schema": "redrhex.sensor-dr-profile.v2",
        "profile_id": "fixture-held-out-v2",
        "purpose": "held_out_evaluation",
        "evidence": [
            {
                "artifact": evidence.name,
                "sha256": _sha256(evidence),
                "note": "fixture evidence",
            }
        ],
        "parameters": parameters,
    }
    path = root / "fixture-held-out-profile.json"
    path.write_text(
        json.dumps(payload, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path, payload


def _write_run(
    root: Path,
    policy: str,
    seed: int,
    *,
    domain: str = "flat",
    accepted: bool = True,
    identity_override: dict[str, object] | None = None,
) -> dict[str, object]:
    forward_semantics = (
        _LEGACY_FORWARD_CONTIGUOUS_SEMANTICS
        if policy == "legacy_student"
        else _V2_FORWARD_CONTIGUOUS_SEMANTICS
    )
    command = root / f"{policy}-{seed}-{domain}.csv"
    with command.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "command",
                "skill",
                "cmd_vx",
                "cmd_vy",
                "cmd_wz",
                "sample_count",
                "success_sample_count",
                "fall_events",
                "episode_ends",
                "mae_vx",
                "mae_vy",
                "mae_wz",
                "success_ratio",
                "contiguous_success_env_ratio",
                "contiguous_success_semantics",
                "diag_sign_match_ratio",
                "yaw_tilt_ok_ratio",
                "fall_rate",
                "accept_pass",
                "fixture_run_id",
            ),
        )
        writer.writeheader()
        writer.writerow(
            {
                "command": "forward",
                "skill": "forward",
                "cmd_vx": 0.35,
                "cmd_vy": 0.0,
                "cmd_wz": 0.0,
                "sample_count": 120,
                "success_sample_count": 120 if accepted else 0,
                "fall_events": 0,
                "episode_ends": 1,
                "mae_vx": 0.05 + seed * 0.001 if accepted else 1.0,
                "mae_vy": 0.01,
                "mae_wz": 0.02,
                "success_ratio": 1.0 if accepted else 0.0,
                "contiguous_success_env_ratio": 1.0 if accepted else 0.0,
                "contiguous_success_semantics": forward_semantics,
                "diag_sign_match_ratio": 1.0,
                "yaw_tilt_ok_ratio": 1.0,
                "fall_rate": 0.0,
                "accept_pass": accepted,
                "fixture_run_id": f"{policy}:{seed}:{domain}",
            }
        )
    episode = root / f"{policy}-{seed}-{domain}-episodes.csv"
    with episode.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "command",
                "skill",
                "environment_index",
                "episode_index",
                "complete",
                "sample_count",
                "fall_count",
                "success_count",
                "mae_vx",
                "mae_vy",
                "mae_wz",
                "success_ratio",
                "fixture_run_id",
            ),
        )
        writer.writeheader()
        writer.writerow(
            {
                "command": "forward",
                "skill": "forward",
                "environment_index": 0,
                "episode_index": 0,
                "complete": True,
                "sample_count": 120,
                "fall_count": 0,
                "success_count": 120 if accepted else 0,
                "mae_vx": 0.05 + seed * 0.001 if accepted else 1.0,
                "mae_vy": 0.01,
                "mae_wz": 0.02,
                "success_ratio": 1.0 if accepted else 0.0,
                "fixture_run_id": f"{policy}:{seed}:{domain}",
            }
        )
    summary = root / f"{policy}-{seed}-{domain}-summary.csv"
    identity = _run_identity(policy, seed, domain)
    if identity_override is not None:
        identity.update(identity_override)
    with summary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("metric", "value"))
        writer.writeheader()
        identity_metrics = (
            ("evaluation.seed", seed),
            ("evaluation.num_envs", 1),
            ("evaluation.sweep_steps", 120),
            ("evaluation.domain", domain),
            ("evaluation.agent_entry_point", identity["agent_entry_point"]),
            ("evaluation.protocol_sha256", identity["evaluation_protocol_sha256"]),
            ("checkpoint.kind", identity["checkpoint_kind"]),
            ("checkpoint.stage", identity["checkpoint_stage"]),
            ("checkpoint.sha256", identity["checkpoint_sha256"]),
            ("checkpoint.config_sha256", identity["config_sha256"]),
            ("artifact.command_csv_sha256", _sha256(command)),
            ("artifact.episode_csv_sha256", _sha256(episode)),
            ("evidence.episode_row_count", 1),
        )
        for metric, value in identity_metrics:
            writer.writerow({"metric": metric, "value": value})
        if domain not in {"flat", "nominal"}:
            profile_path, profile_payload = _held_out_profile(root)
            held_out_parameters = profile_payload["parameters"]
            for metric, value in (
                ("sensor_dr.profile_id", "fixture-held-out-v2"),
                ("sensor_dr.profile_sha256", _sha256(profile_path)),
                ("sensor_dr.profile_path", str(profile_path.resolve())),
                ("sensor_dr.profile_purpose", "held_out_evaluation"),
                ("sensor_dr.active_categories", "actuator,friction,latency,noise,sensor"),
                (
                    "sensor_dr.parameters_json",
                    json.dumps(
                        held_out_parameters,
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                ),
            ):
                writer.writerow({"metric": metric, "value": value})
        if policy != "legacy_student":
            for field, metric in (
                ("observation_contract_sha256", "checkpoint.observation_contract_sha256"),
                ("action_contract_sha256", "checkpoint.action_contract_sha256"),
                ("training_calibration_sha256", "checkpoint.training_calibration_sha256"),
                ("architecture_sha256", "checkpoint.architecture_sha256"),
                ("canonical_config_sha256", "checkpoint.canonical_config_sha256"),
            ):
                writer.writerow({"metric": metric, "value": identity[field]})
            writer.writerow(
                {"metric": "checkpoint.training_seed", "value": identity["training_seed"]}
            )
        for metric, value in (
            ("acceptance.duration_s", 2.0),
            ("acceptance.contiguous_env_ratio_threshold", 0.50),
            ("acceptance.max_fall_rate", 0.20),
            ("acceptance.forward_vx_abs", 0.15),
            ("acceptance.forward_lin_ratio", 0.55),
            ("acceptance.forward_lateral_leak", 0.12),
            ("acceptance.forward_yaw_leak", 0.30),
            ("acceptance.forward_tilt_bound_rad", 0.70),
            ("acceptance.forward_min_base_height_m", 0.085),
            ("acceptance.lateral_vy_abs", 0.15),
            ("acceptance.lateral_forward_leak", 0.12),
            ("acceptance.lateral_yaw_leak", 0.30),
            ("acceptance.yaw_wz_abs", 0.40),
            ("acceptance.yaw_wz_ratio", 0.55),
            ("acceptance.diag_sign_ratio", 0.70),
            ("acceptance.diag_component_ratio", 0.50),
            ("acceptance.diag_yaw_leak", 0.35),
            ("acceptance.yaw_tilt_ratio", 0.70),
            ("acceptance.yaw_tilt_bound_rad", 0.60),
            ("acceptance.yaw_linear_leak", 0.18),
            ("acceptance.yaw_min_base_height_m", 0.12),
            ("acceptance.skill_pass_ratio_threshold", 0.60),
            ("acceptance.overall_pass_ratio_threshold", 0.70),
            ("acceptance.forward_contiguous_semantics", forward_semantics),
            ("acceptance.max_main_action_saturation_ratio", 0.05),
            ("acceptance.command_pass_ratio", 1.0 if accepted else 0.0),
            ("acceptance.min_skill_pass_ratio", 1.0 if accepted else 0.0),
            ("acceptance.max_command_fall_rate", 0.0),
            ("acceptance.skill_pass_ratio.forward", 1.0 if accepted else 0.0),
            ("tracking.mean_abs_vx", 0.05),
            ("tracking.mean_abs_vy", 0.01),
            ("tracking.mean_abs_wz", 0.02),
            ("stability.fall_rate", 0.03),
            ("stability.roll_rms", 0.04),
            ("stability.pitch_rms", 0.05),
            ("policy.main_action_saturation_ratio", 0.01),
            ("policy.abad_action_saturation_ratio", 0.0),
            ("policy.abad_action_magnitude_mean", 0.0),
        ):
            writer.writerow({"metric": metric, "value": value})
        writer.writerow({"metric": "acceptance.overall_status", "value": "PASS" if accepted else "FAIL"})
    return {
        "policy": policy,
        "seed": seed,
        "domain": domain,
        "command_csv": command.name,
        "episode_csv": episode.name,
        "summary_csv": summary.name,
    }


def _rewrite_run_acceptance(
    root: Path,
    run: dict[str, object],
    *,
    accepted: bool,
    update_summary_aggregates: bool = True,
) -> None:
    command_path = root / str(run["command_csv"])
    command_rows = list(csv.DictReader(command_path.open(encoding="utf-8")))
    assert len(command_rows) == 1
    command_rows[0]["mae_vx"] = "0.05" if accepted else "1.0"
    command_rows[0]["success_sample_count"] = "120" if accepted else "0"
    command_rows[0]["success_ratio"] = "1.0" if accepted else "0.0"
    command_rows[0]["contiguous_success_env_ratio"] = "1.0" if accepted else "0.0"
    command_rows[0]["accept_pass"] = str(accepted)
    with command_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(command_rows[0]))
        writer.writeheader()
        writer.writerows(command_rows)

    episode_path = root / str(run["episode_csv"])
    episode_rows = list(csv.DictReader(episode_path.open(encoding="utf-8")))
    assert len(episode_rows) == 1
    episode_rows[0]["mae_vx"] = "0.05" if accepted else "1.0"
    episode_rows[0]["success_count"] = "120" if accepted else "0"
    episode_rows[0]["success_ratio"] = "1.0" if accepted else "0.0"
    with episode_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(episode_rows[0]))
        writer.writeheader()
        writer.writerows(episode_rows)

    summary_path = root / str(run["summary_csv"])
    summary_rows = list(csv.DictReader(summary_path.open(encoding="utf-8")))
    updates: dict[str, object] = {
        "artifact.command_csv_sha256": _sha256(command_path),
        "artifact.episode_csv_sha256": _sha256(episode_path),
    }
    if update_summary_aggregates:
        ratio = 1.0 if accepted else 0.0
        updates.update(
            {
                "acceptance.command_pass_ratio": ratio,
                "acceptance.min_skill_pass_ratio": ratio,
                "acceptance.skill_pass_ratio.forward": ratio,
                "acceptance.overall_status": "PASS" if accepted else "FAIL",
            }
        )
    for row in summary_rows:
        if row["metric"] in updates:
            row["value"] = str(updates[row["metric"]])
    with summary_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("metric", "value"))
        writer.writeheader()
        writer.writerows(summary_rows)


def _write_sensor_replay_source_artifacts(
    root: Path,
    bindings: dict[str, str],
    *,
    bundle_sources: dict[str, dict[str, str]] | None = None,
) -> tuple[dict[str, dict[str, str]], dict[str, object]]:
    """Build structural evidence fixtures; these are not project hardware results."""

    source_bag = root / "sensor-replay-source.db3"
    source_bag.write_bytes(b"structural rosbag fixture; not hardware evidence")
    attestation_path = root / "sensor-replay-capture-attestation.json"
    attestation = {
        "schema": CAPTURE_ATTESTATION_SCHEMA_V2,
        "source_recorder_id": "ros2bag/redrhex-v2-recorder",
        "operator_id": "unit-test-structural-fixture",
        "capture_declaration": "physical_hardware",
        "attested_at_utc": "2026-08-16T00:00:00Z",
        "observation_contract_sha256": bindings[
            "observation_contract_sha256"
        ],
        "attitude_mode": _OBSERVATION.attitude_mode,
        "runtime_calibration_sha256": bindings[
            "runtime_calibration_sha256"
        ],
        "source_bag_sha256": sha256_path_v2(source_bag),
        "source_bag_hash_kind": "sha256-file-v1",
        "topics": dict(REQUIRED_TOPIC_TYPES_V2),
    }
    attestation_path.write_text(
        json.dumps(attestation, sort_keys=True),
        encoding="utf-8",
    )

    sample_count = 61
    timestamps = 1_000.0 + np.arange(sample_count, dtype=np.float64) / 60.0
    input_trace = root / "sensor-replay-input.npz"
    np.savez_compressed(
        input_trace,
        timestamp_s=timestamps,
        imu_source_timestamp_s=timestamps + 0.001,
        joint_validity_timestamp_s=timestamps,
        imu_gyro_rad_s=np.zeros((sample_count, 3), dtype=np.float64),
        imu_linear_accel_m_s2=np.tile(
            [0.0, 0.0, 9.80665],
            (sample_count, 1),
        ),
        imu_orientation_xyzw=np.tile(
            [0.0, 0.0, 0.0, 1.0],
            (sample_count, 1),
        ),
        imu_orientation_covariance=np.tile(
            np.eye(3, dtype=np.float64).reshape(9) * 1.0e-6,
            (sample_count, 1),
        ),
        imu_frame_id=np.asarray(_OBSERVATION.imu_frame_id),
        main_position_rad=np.zeros((sample_count, 6), dtype=np.float64),
        abad_position_rad=np.zeros((sample_count, 6), dtype=np.float64),
        command=np.tile([0.3, 0.0, 0.0], (sample_count, 1)),
    )
    receipt_path = root / "sensor-replay-import-receipt.json"
    receipt = {
        "schema": "redrhex.sensor-v2-rosbag-import.v1",
        "source_bag": {
            "path": str(source_bag.resolve()),
            "sha256": sha256_path_v2(source_bag),
            "hash_kind": "sha256-file-v1",
        },
        "output_trace": {
            "path": str(input_trace.resolve()),
            "sha256": sha256_path_v2(input_trace),
        },
        "topics": {
            topic: {"type": message_type, "message_count": sample_count}
            for topic, message_type in REQUIRED_TOPIC_TYPES_V2.items()
        },
        "joint_order": list(JOINT_ORDER_V2),
        "imu_frame_id": _OBSERVATION.imu_frame_id,
        "observation_contract_sha256": bindings[
            "observation_contract_sha256"
        ],
        "attitude_mode": _OBSERVATION.attitude_mode,
        "sample_rate_hz": SAMPLE_RATE_HZ_V2,
        "sample_count": sample_count,
        "max_period_error_ratio": MAX_PERIOD_ERROR_RATIO_V2,
        "max_imu_joint_skew_s": MAX_IMU_JOINT_SKEW_S_V2,
        "observed_max_imu_joint_skew_s": 0.001,
        "timestamp_semantics": dict(TIMESTAMP_SEMANTICS_V2),
        "capture_attestation": {
            "path": str(attestation_path.resolve()),
            "sha256": sha256_path_v2(attestation_path),
            "schema": attestation["schema"],
            "source_recorder_id": attestation["source_recorder_id"],
            "operator_id": attestation["operator_id"],
            "capture_declaration": attestation["capture_declaration"],
            "attested_at_utc": attestation["attested_at_utc"],
            "observation_contract_sha256": attestation[
                "observation_contract_sha256"
            ],
            "attitude_mode": attestation["attitude_mode"],
            "runtime_calibration_sha256": attestation[
                "runtime_calibration_sha256"
            ],
        },
    }
    receipt_path.write_text(json.dumps(receipt, sort_keys=True), encoding="utf-8")

    if bundle_sources is None:
        onnx_path = root / "sensor-replay-policy.onnx"
        onnx_path.write_bytes(b"structural ONNX fixture")
        sidecar_path = root / "sensor-replay-policy.onnx.json"
        sidecar_path.write_text(
            json.dumps(
                {
                    "metadata": {
                        "contract_sha256": bindings["observation_contract_sha256"],
                        "action_contract_sha256": bindings[
                            "action_contract_sha256"
                        ],
                        "calibration_sha256": bindings[
                            "runtime_calibration_sha256"
                        ],
                        "training_calibration_sha256": bindings[
                            "training_calibration_sha256"
                        ],
                        "checkpoint_sha256": bindings["checkpoint_sha256"],
                        "architecture_sha256": bindings["architecture_sha256"],
                        "config_sha256": bindings["config_sha256"],
                        "canonical_config_sha256": bindings[
                            "canonical_config_sha256"
                        ],
                        "training_seed": "42",
                    },
                    "contract": _OBSERVATION.to_dict(include_sha256=True),
                    "action_contract": _ACTION.to_dict(include_sha256=True),
                    "calibration": _RUNTIME_CALIBRATION.to_dict(
                        include_sha256=True
                    ),
                    "training_calibration": _TRAINING_CALIBRATION.to_dict(
                        include_sha256=True
                    ),
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
    else:
        onnx_path = Path(bundle_sources["onnx"]["path"])
        sidecar_path = Path(bundle_sources["sidecar"]["path"])
    output_npz = root / "sensor-replay-output.npz"
    with np.load(input_trace, allow_pickle=False) as replay_input:
        trace = {name: replay_input[name] for name in replay_input.files}
    canonical_outputs, canonical_summary = replay_module.replay_arrays(
        trace,
        contract=_OBSERVATION,
        action_contract=_ACTION,
        calibration=_RUNTIME_CALIBRATION,
        runner=_StructuralReplayRunner(_OBSERVATION, _ACTION, bindings),
        trace_kind="real",
        max_period_error_ratio=MAX_PERIOD_ERROR_RATIO_V2,
        max_main_action_saturation_fraction=0.05,
    )
    assert canonical_summary["status"] == "passed"
    np.savez_compressed(output_npz, **canonical_outputs)
    paths = {
        "source_bag": source_bag,
        "capture_attestation": attestation_path,
        "import_receipt": receipt_path,
        "input_trace": input_trace,
        "onnx": onnx_path,
        "sidecar": sidecar_path,
        "hardware_config": replay_module.DEPLOYMENT_HARDWARE_CONFIG_PATH_V2,
        "output_npz": output_npz,
    }
    return (
        {
            name: {"path": str(path.resolve()), "sha256": sha256_path_v2(path)}
            for name, path in paths.items()
        },
        canonical_summary["hardware_target_tightening"],
    )


def _replace_sensor_replay_with_executable_onnx(manifest_path: Path) -> None:
    """Rebuild replay outputs with the exact parity-proven F4 ONNX bundle."""

    pytest.importorskip("onnxruntime")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifact_record = payload["promotion_artifacts"]["sensor_replay"]
    artifact_path = manifest_path.parent / artifact_record["path"]
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    sources = artifact["source_artifacts"]
    onnx_path = Path(sources["onnx"]["path"])
    sidecar_path = Path(sources["sidecar"]["path"])
    runner, contract, action_contract, calibration = _REAL_REPLAY_LOAD_BUNDLE(
        onnx_path,
        sidecar_path,
        require_hardware_ready=True,
    )
    with np.load(Path(sources["input_trace"]["path"]), allow_pickle=False) as stream:
        trace = {name: stream[name] for name in stream.files}
    outputs, summary = replay_module.replay_arrays(
        trace,
        contract=contract,
        action_contract=action_contract,
        calibration=calibration,
        runner=runner,
        trace_kind="real",
        max_period_error_ratio=MAX_PERIOD_ERROR_RATIO_V2,
        max_main_action_saturation_fraction=0.05,
    )
    assert summary["status"] == "passed"
    output_path = Path(sources["output_npz"]["path"])
    np.savez_compressed(output_path, **outputs)
    sources["output_npz"]["sha256"] = _sha256(output_path)
    for name in (
        "status",
        "sample_count",
        "sensor_frame_count",
        "history_ready_count",
        "policy",
        "hardware_target_tightening",
    ):
        artifact[name] = summary[name]
    artifact_path.write_text(json.dumps(artifact, sort_keys=True), encoding="utf-8")
    artifact_record["sha256"] = _sha256(artifact_path)
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")


def _write_gate_artifact(
    root: Path,
    gate: str,
    *,
    status: str = "PASS",
    bindings: dict[str, str] | None = None,
    sensor_bundle_sources: dict[str, dict[str, str]] | None = None,
) -> dict[str, str]:
    path = root / f"{gate}.json"
    normalized_status = status.strip().upper()
    aliased_bindings = {
        "contract_sha256": (bindings or {}).get("observation_contract_sha256"),
        "action_contract_sha256": (bindings or {}).get("action_contract_sha256"),
        "calibration_sha256": (bindings or {}).get("runtime_calibration_sha256"),
        "training_calibration_sha256": (bindings or {}).get(
            "training_calibration_sha256"
        ),
        "checkpoint_sha256": (bindings or {}).get("checkpoint_sha256"),
        "architecture_sha256": (bindings or {}).get("architecture_sha256"),
        "config_sha256": (bindings or {}).get("config_sha256"),
        "canonical_config_sha256": (bindings or {}).get(
            "canonical_config_sha256"
        ),
    }
    aliased_bindings = {name: value for name, value in aliased_bindings.items() if value is not None}
    if gate == "torch_onnx_parity":
        payload = {
            "metadata": {
                "bundle_schema": "redrhex.sensor-policy-bundle.v2",
                **aliased_bindings,
            },
            "torch_onnx_parity": {
                "status": "passed" if normalized_status == "PASS" else "failed",
                "random_sample_count": 4,
                "recorded_sample_count": 2,
                "action_max_abs_error": 1.0e-6,
                "velocity_max_abs_error": 1.0e-6,
                "absolute_tolerance": 2.0e-5,
            },
            "recorded_parity_input": {"sha256": "d" * 64},
        }
    elif gate == "sensor_replay":
        replay_fixture = (
            _write_sensor_replay_source_artifacts(
                root,
                bindings or {},
                bundle_sources=sensor_bundle_sources,
            )
            if normalized_status == "PASS"
            else None
        )
        source_artifacts = None if replay_fixture is None else replay_fixture[0]
        hardware_target_tightening = (
            None if replay_fixture is None else replay_fixture[1]
        )
        payload = {
            "schema": "redrhex.sensor-v2-replay.v2",
            "status": "passed" if normalized_status == "PASS" else "failed",
            "trace_kind": "real",
            "sample_count": 61,
            "sensor_frame_count": 60,
            "history_ready_count": 1,
            "training_seed": 42,
            "policy": {
                "inference_count": 1,
                "action_abs_max": 0.0,
                "main_action_saturation_fraction": 0.0,
                "max_main_action_saturation_fraction": 0.05,
                "main_action_saturation_gate_passed": True,
                "main_action_saturation_limit_source": "interim safety gate",
                "main_action_saturation_sensitivity": {
                    "strict": 0.0,
                    "base": 0.05,
                    "relaxed": 0.1,
                },
                "abad_action_abs_max": 0.0,
            },
            **aliased_bindings,
        }
        if source_artifacts is not None:
            payload["source_artifacts"] = source_artifacts
            payload["hardware_target_tightening"] = hardware_target_tightening
    else:
        required_checks = {
            "no_privileged_leak": (
                "actor_inputs_exact",
                "forbidden_features_absent",
                "command_separate",
                "privileged_groups_training_only",
            ),
            "contract_provenance": (
                "observation_contract_hash",
                "action_contract_hash",
                "calibration_hash",
                "runtime_calibration_lineage",
                "checkpoint_manifest_binding",
                "architecture_config_binding",
            ),
        }[gate]
        payload = {
            "schema": f"redrhex.{gate.replace('_', '-')}-gate.v2",
            "gate": gate,
            "status": normalized_status,
            "checks": [
                {"name": name, "status": "PASS"} for name in required_checks
            ],
            **(bindings or {}),
        }
    path.write_text(
        json.dumps(payload, sort_keys=True),
        encoding="utf-8",
    )
    return {"path": path.name, "sha256": _sha256(path)}


def _canonical_generated_gates(root: Path) -> dict[str, dict[str, str]]:
    helper_path = Path(__file__).with_name("test_sensor_v2_promotion_gates.py")
    spec = importlib.util.spec_from_file_location(
        "redrhex_sensor_v2_promotion_fixture", helper_path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    onnx_path, sidecar_path, checkpoint_path = module._bundle(root / "bundle-source")
    parity_input_path = checkpoint_path.parent / module.PARITY_INPUT_NAME
    generated = generate_promotion_gates(
        onnx_path=onnx_path,
        sidecar_path=sidecar_path,
        checkpoint_path=checkpoint_path,
        parity_input_path=parity_input_path,
        parity_input_sha256=_sha256(parity_input_path),
        output_dir=root / "generated-gates",
    )
    return {
        gate: {
            "path": str(Path(record["path"]).relative_to(root)),
            "sha256": record["sha256"],
        }
        for gate, record in generated.items()
    }


def _manifest(tmp_path: Path) -> Path:
    canonical_artifacts = _canonical_generated_gates(tmp_path)
    parity_payload = json.loads(
        (tmp_path / canonical_artifacts["torch_onnx_parity"]["path"]).read_text(
            encoding="utf-8"
        )
    )
    candidate_bindings = {
        name: parity_payload["provenance"][name]
        for name in (
            "observation_contract_sha256",
            "action_contract_sha256",
            "runtime_calibration_sha256",
            "training_calibration_sha256",
            "checkpoint_sha256",
            "architecture_sha256",
            "config_sha256",
            "canonical_config_sha256",
        )
    }
    candidate_training_identity = {
        name: candidate_bindings[name]
        for name in (
            "observation_contract_sha256",
            "action_contract_sha256",
            "training_calibration_sha256",
            "checkpoint_sha256",
            "architecture_sha256",
            "config_sha256",
            "canonical_config_sha256",
        )
    }
    candidate_training_identity["training_seed"] = parity_payload["provenance"][
        "training_seed"
    ]
    policies = sorted(set(REQUIRED_ABLATIONS) | {"teacher_a"})
    runs = [
        _write_run(
            tmp_path,
            policy,
            seed,
            domain=domain,
            identity_override=(
                candidate_training_identity
                if policy == "v2_ppo" and seed == 42
                else {
                    "canonical_config_sha256": candidate_bindings[
                        "canonical_config_sha256"
                    ]
                }
                if policy == "v2_ppo"
                else None
            ),
        )
        for policy in policies
        for seed in (42, 43, 44)
        for domain in (
            ("flat",) if policy == "legacy_student" else ("flat", "held_out")
        )
    ]
    artifacts = {
        "sensor_replay": _write_gate_artifact(
            tmp_path,
            "sensor_replay",
            bindings=candidate_bindings,
            sensor_bundle_sources=parity_payload["source_artifacts"],
        )
    }
    artifacts.update(canonical_artifacts)
    path = tmp_path / "manifest.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "runs": runs,
                "deployment_candidate": {"seed": 42, "domain": "held_out"},
                "promotion_artifacts": artifacts,
            }
        ),
        encoding="utf-8",
    )
    return path


def test_gap_canonical_acceptance_thresholds_match_evaluator_defaults() -> None:
    option_by_protocol_name = {
        "duration_s": "--accept_duration_s",
        "contiguous_env_ratio": "--accept-contiguous-env-ratio",
        "max_fall_rate": "--accept_max_fall_rate",
        "forward_vx_abs": "--accept_vx_abs",
        "forward_lin_ratio": "--accept_lin_ratio",
        "forward_lateral_leak": "--accept_forward_lateral_leak",
        "forward_yaw_leak": "--accept_forward_yaw_leak",
        "forward_tilt_bound_rad": "--accept-forward-tilt-bound",
        "forward_min_base_height_m": "--accept-forward-min-base-height",
        "lateral_vy_abs": "--accept_vy_abs",
        "lateral_forward_leak": "--accept_lateral_forward_leak",
        "lateral_yaw_leak": "--accept_lateral_yaw_leak",
        "yaw_wz_abs": "--accept_wz_abs",
        "yaw_wz_ratio": "--accept_wz_ratio",
        "diag_sign_ratio": "--accept_diag_sign_ratio",
        "diag_component_ratio": "--accept_diag_component_ratio",
        "diag_yaw_leak": "--accept_diag_yaw_leak",
        "yaw_tilt_ratio": "--accept_yaw_tilt_ratio",
        "yaw_tilt_bound_rad": "--accept_yaw_tilt_bound",
        "yaw_linear_leak": "--accept_yaw_lin_leak",
        "yaw_min_base_height_m": "--accept_min_base_height",
        "skill_pass_ratio": "--accept_skill_pass_ratio",
        "overall_pass_ratio": "--accept_overall_pass_ratio",
        "max_main_action_saturation_ratio": (
            "--accept-max-main-action-saturation-ratio"
        ),
    }
    source_path = Path(__file__).parents[3] / "scripts/rsl_rl/eval_command_sweep.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    parser_defaults: dict[str, float] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr != "add_argument" or not node.args:
            continue
        if not isinstance(node.args[0], ast.Constant) or not isinstance(
            node.args[0].value, str
        ):
            continue
        option = node.args[0].value
        default = next(
            (
                ast.literal_eval(keyword.value)
                for keyword in node.keywords
                if keyword.arg == "default"
            ),
            None,
        )
        if option in option_by_protocol_name.values() and default is not None:
            parser_defaults[str(option)] = float(default)

    assert {
        name: parser_defaults[option]
        for name, option in option_by_protocol_name.items()
    } == CANONICAL_ACCEPTANCE_PROTOCOL


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("sample_count", "forward.sample_count does not match"),
        ("success_count", "forward.success_sample_count does not match"),
        ("fall_count", "forward.fall_events does not match"),
        ("mae_vx", "forward.mae_vx does not match episode-weighted"),
        ("success_ratio", "episode\\[0\\].success_ratio disagrees"),
    ),
)
def test_gap_rejects_rehashed_episode_command_contradictions(
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    run = _write_run(tmp_path, "teacher_a", 1)
    episode_path = tmp_path / str(run["episode_csv"])
    episode_rows = list(csv.DictReader(episode_path.open(encoding="utf-8")))
    row = episode_rows[0]
    if mutation == "sample_count":
        row["sample_count"] = "119"
        row["success_count"] = "119"
        row["success_ratio"] = "1.0"
    elif mutation == "success_count":
        row["success_count"] = "119"
        row["success_ratio"] = str(119.0 / 120.0)
    elif mutation == "fall_count":
        row["fall_count"] = "1"
    elif mutation == "mae_vx":
        row["mae_vx"] = "0.25"
    else:
        row["success_ratio"] = "0.5"
    with episode_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        writer.writeheader()
        writer.writerows(episode_rows)

    summary_path = tmp_path / str(run["summary_csv"])
    summary_rows = list(csv.DictReader(summary_path.open(encoding="utf-8")))
    next(
        summary_row
        for summary_row in summary_rows
        if summary_row["metric"] == "artifact.episode_csv_sha256"
    )["value"] = _sha256(episode_path)
    with summary_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("metric", "value"))
        writer.writeheader()
        writer.writerows(summary_rows)

    with pytest.raises(StudentTeacherGapError, match=message):
        _resolve_run(tmp_path, run)


def test_gap_allows_float64_csv_aggregate_roundtrip_noise(tmp_path: Path) -> None:
    run = _write_run(tmp_path, "teacher_a", 1)
    episode_path = tmp_path / str(run["episode_csv"])
    episode_rows = list(csv.DictReader(episode_path.open(encoding="utf-8")))
    episode_rows[0]["mae_vx"] = str(float(episode_rows[0]["mae_vx"]) + 5.0e-10)
    with episode_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(episode_rows[0]))
        writer.writeheader()
        writer.writerows(episode_rows)
    summary_path = tmp_path / str(run["summary_csv"])
    summary_rows = list(csv.DictReader(summary_path.open(encoding="utf-8")))
    next(
        row
        for row in summary_rows
        if row["metric"] == "artifact.episode_csv_sha256"
    )["value"] = _sha256(episode_path)
    with summary_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("metric", "value"))
        writer.writeheader()
        writer.writerows(summary_rows)

    resolved = _resolve_run(tmp_path, run)

    assert resolved["overall_pass"] is True


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("missing_canonical_config", "checkpoint.canonical_config_sha256"),
        ("wrong_training_seed", "training_seed must equal evaluation seed"),
    ),
)
def test_gap_requires_v2_canonical_config_and_matching_training_seed(
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    run = _write_run(tmp_path, "teacher_a", 1)
    summary_path = tmp_path / str(run["summary_csv"])
    summary_rows = list(csv.DictReader(summary_path.open(encoding="utf-8")))
    if mutation == "missing_canonical_config":
        summary_rows = [
            row
            for row in summary_rows
            if row["metric"] != "checkpoint.canonical_config_sha256"
        ]
    else:
        next(
            row
            for row in summary_rows
            if row["metric"] == "checkpoint.training_seed"
        )["value"] = "2"
    with summary_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("metric", "value"))
        writer.writeheader()
        writer.writerows(summary_rows)

    with pytest.raises(StudentTeacherGapError, match=message):
        _resolve_run(tmp_path, run)


def test_gap_report_requires_and_passes_hash_bound_promotion_evidence(tmp_path: Path) -> None:
    result = evaluate_student_teacher_gap(_manifest(tmp_path))

    assert result["promotion"]["pass"] is True
    assert result["contact_supervision"]["status"] == "blocked"
    assert set(result["teacher_gap"]) == {"legacy_student", "v2_distilled", "v2_ppo"}
    contract_bindings = result["promotion"]["artifacts"]["contract_provenance"]["bindings"]
    assert contract_bindings["runtime_calibration_sha256"] == RUNTIME_CALIBRATION_SHA256
    assert contract_bindings["training_calibration_sha256"] == CONTRACT_HASHES[
        "training_calibration_sha256"
    ]
    assert (
        result["promotion"]["artifacts"]["contract_provenance"][
            "checkpoint_stage"
        ]
        == "ppo_f4"
    )


def test_gap_rejects_real_replay_from_a_different_onnx_bundle(
    tmp_path: Path,
) -> None:
    manifest = _manifest(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    replay_record = payload["promotion_artifacts"]["sensor_replay"]
    replay_path = tmp_path / replay_record["path"]
    replay = json.loads(replay_path.read_text(encoding="utf-8"))
    different_onnx = tmp_path / "different-replay-policy.onnx"
    different_onnx.write_bytes(b"a separately rehashed replay graph")
    replay["source_artifacts"]["onnx"] = {
        "path": str(different_onnx.resolve()),
        "sha256": _sha256(different_onnx),
    }
    replay_path.write_text(json.dumps(replay, sort_keys=True), encoding="utf-8")
    replay_record["sha256"] = _sha256(replay_path)
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(
        StudentTeacherGapError,
        match="exact bundle proven by the canonical Torch/ONNX parity gate",
    ):
        evaluate_student_teacher_gap(manifest)


def test_gap_rejects_legacy_held_out_evidence(tmp_path: Path) -> None:
    run = _write_run(tmp_path, "legacy_student", 1, domain="held_out")

    with pytest.raises(StudentTeacherGapError, match="legacy_student evidence is nominal-only"):
        _resolve_run(tmp_path, run)


def test_gap_compares_legacy_only_to_nominal_teacher_metrics(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    for run in payload["runs"]:
        if run["policy"] != "teacher_a" or run["domain"] != "held_out":
            continue
        summary_path = tmp_path / str(run["summary_csv"])
        summary_path.write_text(
            summary_path.read_text(encoding="utf-8").replace(
                "tracking.mean_abs_vx,0.05", "tracking.mean_abs_vx,0.50"
            ),
            encoding="utf-8",
        )

    result = evaluate_student_teacher_gap(manifest)

    assert result["aggregate"]["teacher_a"]["summary.tracking.mean_abs_vx"][
        "mean"
    ] > 0.05
    assert result["teacher_gap"]["legacy_student"][
        "summary.tracking.mean_abs_vx"
    ] == pytest.approx(0.0)


@pytest.mark.parametrize(
    ("metric", "forged_value", "error_match"),
    (
        (
            "evaluation.agent_entry_point",
            "rsl_rl_ppo_v2_cfg_entry_point",
            "requires agent entry point 'rsl_rl_robust_ppo_v2_cfg_entry_point'",
        ),
        ("checkpoint.stage", "ppo_f3", "requires checkpoint stage 'ppo_f4'"),
    ),
)
def test_gap_requires_robust_f4_ppo_identity(
    tmp_path: Path,
    metric: str,
    forged_value: str,
    error_match: str,
) -> None:
    run = _write_run(tmp_path, "v2_ppo", 1)
    summary_path = tmp_path / str(run["summary_csv"])
    summary_rows = list(csv.DictReader(summary_path.open(encoding="utf-8")))
    next(row for row in summary_rows if row["metric"] == metric)["value"] = forged_value
    with summary_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("metric", "value"))
        writer.writeheader()
        writer.writerows(summary_rows)

    with pytest.raises(StudentTeacherGapError, match=error_match):
        _resolve_run(tmp_path, run)


def test_gap_rejects_noncanonical_stage_in_rehashed_promotion_source(
    tmp_path: Path,
) -> None:
    artifacts = _canonical_generated_gates(tmp_path)
    record = artifacts["contract_provenance"]
    artifact_path = tmp_path / record["path"]
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    sidecar_path = Path(artifact["source_artifacts"]["sidecar"]["path"])
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    sidecar["metadata"]["stage"] = "ppo_f3"
    sidecar_path.write_text(json.dumps(sidecar, sort_keys=True), encoding="utf-8")
    artifact["source_artifacts"]["sidecar"]["sha256"] = _sha256(sidecar_path)
    artifact_path.write_text(json.dumps(artifact, sort_keys=True), encoding="utf-8")
    record["sha256"] = _sha256(artifact_path)

    with pytest.raises(
        StudentTeacherGapError,
        match="canonical source verification failed:.*stage 'ppo_f4'",
    ):
        _resolve_promotion_artifact(
            tmp_path,
            "contract_provenance",
            record,
        )


def test_gap_reruns_parity_instead_of_trusting_rehashed_report(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    artifact_record = payload["promotion_artifacts"]["torch_onnx_parity"]
    artifact_path = tmp_path / artifact_record["path"]
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    artifact["torch_onnx_parity"]["random_sample_count"] += 1
    artifact_path.write_text(json.dumps(artifact, sort_keys=True), encoding="utf-8")
    artifact_record["sha256"] = _sha256(artifact_path)

    with pytest.raises(
        StudentTeacherGapError,
        match="random_sample_count differs from canonical verification",
    ):
        _resolve_promotion_artifact(
            tmp_path,
            "torch_onnx_parity",
            artifact_record,
        )


def test_gap_rejects_untyped_promotion_training_seed(tmp_path: Path) -> None:
    artifacts = _canonical_generated_gates(tmp_path)
    record = artifacts["contract_provenance"]
    artifact_path = tmp_path / record["path"]
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    artifact["provenance"]["training_seed"] = "1"
    artifact_path.write_text(json.dumps(artifact, sort_keys=True), encoding="utf-8")
    record["sha256"] = _sha256(artifact_path)

    with pytest.raises(
        StudentTeacherGapError,
        match="promotion artifact training_seed must be a non-negative integer",
    ):
        _resolve_promotion_artifact(
            tmp_path,
            "contract_provenance",
            record,
        )


def test_gap_allows_per_seed_configs_but_not_per_domain_configs(
    tmp_path: Path,
) -> None:
    manifest = _manifest(tmp_path)
    result = evaluate_student_teacher_gap(manifest)
    assert result["promotion"]["pass"] is True

    payload = json.loads(manifest.read_text(encoding="utf-8"))
    target = next(
        run
        for run in payload["runs"]
        if run["policy"] == "teacher_a"
        and run["seed"] == 43
        and run["domain"] == "held_out"
    )
    summary = tmp_path / target["summary_csv"]
    old_config = _digest("teacher_a:43:config")
    summary.write_text(
        summary.read_text(encoding="utf-8").replace(
            old_config,
            _digest("teacher_a:43:held-out-config"),
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        StudentTeacherGapError,
        match="teacher_a uses multiple configs for one training seed",
    ):
        evaluate_student_teacher_gap(manifest)


def test_gap_checks_required_execution_stripped_config_across_seeds(
    tmp_path: Path,
) -> None:
    manifest = _manifest(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    for run in payload["runs"]:
        if run["policy"] != "v2_ppo" or run["seed"] != 43:
            continue
        summary = tmp_path / run["summary_csv"]
        summary_rows = list(csv.DictReader(summary.open(encoding="utf-8")))
        next(
            row
            for row in summary_rows
            if row["metric"] == "checkpoint.canonical_config_sha256"
        )["value"] = _digest("v2_ppo:changed-canonical-config")
        with summary.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=("metric", "value"))
            writer.writeheader()
            writer.writerows(summary_rows)

    with pytest.raises(
        StudentTeacherGapError,
        match="v2_ppo execution-field-stripped configuration differs across seeds",
    ):
        evaluate_student_teacher_gap(manifest)


def test_gap_report_rejects_nonidentical_command_set(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    target = next(
        run
        for run in payload["runs"]
        if run["policy"] == "v2_ppo" and run["seed"] == 42
    )
    command = tmp_path / target["command_csv"]
    text = command.read_text(encoding="utf-8").replace("0.35", "0.36")
    command.write_text(text, encoding="utf-8")
    summary = tmp_path / target["summary_csv"]
    summary_rows = summary.read_text(encoding="utf-8").splitlines()
    summary.write_text(
        "\n".join(
            (
                f"artifact.command_csv_sha256,{_sha256(command)}"
                if line.startswith("artifact.command_csv_sha256,")
                else line
            )
            for line in summary_rows
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(StudentTeacherGapError, match="command set differs"):
        evaluate_student_teacher_gap(manifest)


def test_gap_report_rejects_legacy_boolean_schema_with_migration_hint(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["schema_version"] = 1
    payload["gates"] = {gate: True for gate in PROMOTION_GATES}
    payload.pop("promotion_artifacts")
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(StudentTeacherGapError, match="legacy schema_version 1.*migrate.*promotion_artifacts"):
        evaluate_student_teacher_gap(manifest)


def test_gap_report_rejects_gate_artifact_changed_after_manifest(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    artifact = tmp_path / payload["promotion_artifacts"]["sensor_replay"]["path"]
    artifact.write_text(artifact.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    with pytest.raises(StudentTeacherGapError, match="sensor_replay.*sha256 mismatch"):
        evaluate_student_teacher_gap(manifest)


def test_sensor_replay_branch_accepts_complete_rehashed_source_chain(
    tmp_path: Path,
) -> None:
    manifest = _manifest(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    result = _resolve_promotion_artifact(
        tmp_path,
        "sensor_replay",
        payload["promotion_artifacts"]["sensor_replay"],
    )

    assert result["status"] == "PASS"
    assert set(result["source_artifacts"]) == {
        "source_bag",
        "capture_attestation",
        "import_receipt",
        "input_trace",
        "onnx",
        "sidecar",
        "hardware_config",
        "output_npz",
    }


@pytest.mark.parametrize("source_name", ("source_bag", "input_trace", "output_npz"))
def test_gap_rehashes_sensor_replay_source_artifacts(
    tmp_path: Path,
    source_name: str,
) -> None:
    manifest = _manifest(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    artifact_path = tmp_path / payload["promotion_artifacts"]["sensor_replay"]["path"]
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    source_path = Path(artifact["source_artifacts"][source_name]["path"])
    source_path.write_bytes(source_path.read_bytes() + b"tampered")

    with pytest.raises(
        StudentTeacherGapError,
        match=rf"sensor_replay source artifact {source_name} sha256 mismatch",
    ):
        _resolve_promotion_artifact(
            tmp_path,
            "sensor_replay",
            payload["promotion_artifacts"]["sensor_replay"],
        )


def test_gap_rehashes_sensor_replay_hardware_config_record(
    tmp_path: Path,
) -> None:
    manifest = _manifest(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    artifact_record = payload["promotion_artifacts"]["sensor_replay"]
    artifact_path = tmp_path / artifact_record["path"]
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    artifact["source_artifacts"]["hardware_config"]["sha256"] = "0" * 64
    artifact_path.write_text(json.dumps(artifact, sort_keys=True), encoding="utf-8")
    artifact_record["sha256"] = _sha256(artifact_path)

    with pytest.raises(
        StudentTeacherGapError,
        match="sensor_replay source artifact hardware_config sha256 mismatch",
    ):
        _resolve_promotion_artifact(tmp_path, "sensor_replay", artifact_record)


def test_gap_rejects_rehashed_permissive_hardware_config(
    tmp_path: Path,
) -> None:
    manifest = _manifest(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    artifact_record = payload["promotion_artifacts"]["sensor_replay"]
    artifact_path = tmp_path / artifact_record["path"]
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    permissive_config = tmp_path / "permissive-hardware-config.yaml"
    canonical_text = replay_module.DEPLOYMENT_HARDWARE_CONFIG_PATH_V2.read_text(
        encoding="utf-8"
    )
    permissive_config.write_text(
        canonical_text.replace(
            "main_drive_vel_limit_rad_s: 9.0",
            "main_drive_vel_limit_rad_s: 15.0",
        ),
        encoding="utf-8",
    )
    replacement = {
        "path": str(permissive_config.resolve()),
        "sha256": _sha256(permissive_config),
    }
    artifact["source_artifacts"]["hardware_config"] = replacement
    artifact["hardware_target_tightening"]["deployment_config"] = replacement
    artifact_path.write_text(json.dumps(artifact, sort_keys=True), encoding="utf-8")
    artifact_record["sha256"] = _sha256(artifact_path)

    with pytest.raises(
        StudentTeacherGapError,
        match="hardware config bytes differ from the canonical",
    ):
        _resolve_promotion_artifact(tmp_path, "sensor_replay", artifact_record)


def test_gap_recomputes_hardware_target_tightening_report(
    tmp_path: Path,
) -> None:
    manifest = _manifest(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    artifact_record = payload["promotion_artifacts"]["sensor_replay"]
    artifact_path = tmp_path / artifact_record["path"]
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    artifact["hardware_target_tightening"]["velocity_limit"][
        "max_abs_delta_rad_s"
    ] = 1.0
    artifact_path.write_text(json.dumps(artifact, sort_keys=True), encoding="utf-8")
    artifact_record["sha256"] = _sha256(artifact_path)

    with pytest.raises(
        StudentTeacherGapError,
        match="differs from canonical deployment-decoder rerun",
    ):
        _resolve_promotion_artifact(tmp_path, "sensor_replay", artifact_record)


def test_gap_reruns_canonical_replay_and_rejects_rehashed_forged_output(
    tmp_path: Path,
) -> None:
    manifest = _manifest(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    artifact_record = payload["promotion_artifacts"]["sensor_replay"]
    artifact_path = tmp_path / artifact_record["path"]
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    output_path = Path(artifact["source_artifacts"]["output_npz"]["path"])
    with np.load(output_path, allow_pickle=False) as archive:
        forged = {name: archive[name] for name in archive.files}
    forged["sensor_frames"] = np.zeros_like(forged["sensor_frames"])
    forged["sensor_histories"] = np.zeros_like(forged["sensor_histories"])
    np.savez_compressed(output_path, **forged)
    artifact["source_artifacts"]["output_npz"]["sha256"] = _sha256(output_path)
    artifact_path.write_text(json.dumps(artifact, sort_keys=True), encoding="utf-8")
    artifact_record["sha256"] = _sha256(artifact_path)

    with pytest.raises(StudentTeacherGapError, match="differs from canonical rerun"):
        _resolve_promotion_artifact(
            tmp_path,
            "sensor_replay",
            artifact_record,
        )


def test_gap_rejects_rehashed_attestation_for_different_runtime_calibration(
    tmp_path: Path,
) -> None:
    manifest = _manifest(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    artifact_record = payload["promotion_artifacts"]["sensor_replay"]
    artifact_path = tmp_path / artifact_record["path"]
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    sources = artifact["source_artifacts"]
    attestation_path = Path(sources["capture_attestation"]["path"])
    attestation = json.loads(attestation_path.read_text(encoding="utf-8"))
    attestation["runtime_calibration_sha256"] = "0" * 64
    attestation_path.write_text(json.dumps(attestation, sort_keys=True), encoding="utf-8")
    sources["capture_attestation"]["sha256"] = _sha256(attestation_path)
    receipt_path = Path(sources["import_receipt"]["path"])
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["capture_attestation"]["sha256"] = _sha256(attestation_path)
    receipt["capture_attestation"]["runtime_calibration_sha256"] = "0" * 64
    receipt_path.write_text(json.dumps(receipt, sort_keys=True), encoding="utf-8")
    sources["import_receipt"]["sha256"] = _sha256(receipt_path)
    artifact_path.write_text(json.dumps(artifact, sort_keys=True), encoding="utf-8")
    artifact_record["sha256"] = _sha256(artifact_path)

    with pytest.raises(StudentTeacherGapError, match="attestation runtime calibration"):
        _resolve_promotion_artifact(
            tmp_path,
            "sensor_replay",
            artifact_record,
        )


def test_gap_rejects_synthetic_capture_attestation_even_when_rehashed(
    tmp_path: Path,
) -> None:
    manifest = _manifest(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    artifact_record = payload["promotion_artifacts"]["sensor_replay"]
    artifact_path = tmp_path / artifact_record["path"]
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    sources = artifact["source_artifacts"]

    attestation_path = Path(sources["capture_attestation"]["path"])
    attestation = json.loads(attestation_path.read_text(encoding="utf-8"))
    attestation["capture_declaration"] = "synthetic_fixture"
    attestation_path.write_text(json.dumps(attestation, sort_keys=True), encoding="utf-8")
    sources["capture_attestation"]["sha256"] = _sha256(attestation_path)

    receipt_path = Path(sources["import_receipt"]["path"])
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["capture_attestation"]["sha256"] = _sha256(attestation_path)
    receipt["capture_attestation"]["capture_declaration"] = "synthetic_fixture"
    receipt_path.write_text(json.dumps(receipt, sort_keys=True), encoding="utf-8")
    sources["import_receipt"]["sha256"] = _sha256(receipt_path)

    artifact_path.write_text(json.dumps(artifact, sort_keys=True), encoding="utf-8")
    artifact_record["sha256"] = _sha256(artifact_path)
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(StudentTeacherGapError, match="physical_hardware"):
        _resolve_promotion_artifact(
            tmp_path,
            "sensor_replay",
            payload["promotion_artifacts"]["sensor_replay"],
        )


def test_sensor_replay_branch_rejects_provisional_runtime_calibration(
    tmp_path: Path,
) -> None:
    manifest = _manifest(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    artifact_record = payload["promotion_artifacts"]["sensor_replay"]
    artifact_path = tmp_path / artifact_record["path"]
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    sidecar_path = Path(artifact["source_artifacts"]["sidecar"]["path"])
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    sidecar["calibration"] = _TRAINING_CALIBRATION.to_dict(include_sha256=True)
    sidecar["metadata"]["calibration_sha256"] = _TRAINING_CALIBRATION.sha256
    sidecar_path.write_text(json.dumps(sidecar, sort_keys=True), encoding="utf-8")
    artifact["calibration_sha256"] = _TRAINING_CALIBRATION.sha256
    artifact["source_artifacts"]["sidecar"]["sha256"] = _sha256(sidecar_path)
    sources = artifact["source_artifacts"]
    attestation_path = Path(sources["capture_attestation"]["path"])
    attestation = json.loads(attestation_path.read_text(encoding="utf-8"))
    attestation["runtime_calibration_sha256"] = _TRAINING_CALIBRATION.sha256
    attestation_path.write_text(json.dumps(attestation, sort_keys=True), encoding="utf-8")
    sources["capture_attestation"]["sha256"] = _sha256(attestation_path)
    receipt_path = Path(sources["import_receipt"]["path"])
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["capture_attestation"]["sha256"] = _sha256(attestation_path)
    receipt["capture_attestation"][
        "runtime_calibration_sha256"
    ] = _TRAINING_CALIBRATION.sha256
    receipt_path.write_text(json.dumps(receipt, sort_keys=True), encoding="utf-8")
    sources["import_receipt"]["sha256"] = _sha256(receipt_path)
    artifact_path.write_text(json.dumps(artifact, sort_keys=True), encoding="utf-8")
    artifact_record["sha256"] = _sha256(artifact_path)

    with pytest.raises(StudentTeacherGapError, match="hardware calibration is incomplete"):
        _resolve_promotion_artifact(
            tmp_path,
            "sensor_replay",
            artifact_record,
        )


def test_gap_report_uses_artifact_status_and_rejects_wrong_contract_binding(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    failed = _write_gate_artifact(tmp_path, "sensor_replay", status="FAIL")
    payload["promotion_artifacts"]["sensor_replay"] = failed
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    result = evaluate_student_teacher_gap(manifest)
    assert result["promotion"]["pass"] is False
    assert result["promotion"]["failed_gates"] == ["sensor_replay"]

    record = payload["promotion_artifacts"]["contract_provenance"]
    artifact_path = tmp_path / record["path"]
    artifact_payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    artifact_payload["provenance"]["training_calibration_sha256"] = "a" * 64
    artifact_path.write_text(json.dumps(artifact_payload), encoding="utf-8")
    record["sha256"] = _sha256(artifact_path)
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(StudentTeacherGapError, match="bindings differ from canonical"):
        evaluate_student_teacher_gap(manifest)


def test_gap_tensorboard_writes_available_metrics_and_no_fabricated_heads(tmp_path: Path) -> None:
    EventAccumulator = pytest.importorskip(
        "tensorboard.backend.event_processing.event_accumulator"
    ).EventAccumulator
    result = evaluate_student_teacher_gap(_manifest(tmp_path))
    log_dir = tmp_path / "tensorboard"

    event_path = write_gap_tensorboard(result, log_dir, step=7)

    assert event_path.is_file()
    accumulator = EventAccumulator(str(log_dir)).Reload()
    tags = set(accumulator.Tags()["scalars"])
    assert "Aggregate/v2_ppo/summary.tracking.mean_abs_vx/mean" in tags
    assert "TeacherGap/v2_ppo/summary.tracking.mean_abs_vx" in tags
    for tag in (
        "Student/vx_tracking_error",
        "Student/vy_leak",
        "Student/wz_leak",
        "Student/fall_rate",
        "Student/roll_rms",
        "Student/pitch_rms",
        "Student/action_saturation_main",
        "Student/action_saturation_abad",
    ):
        assert tag in tags
        assert accumulator.Scalars(tag)[0].step == 7
    assert not any(tag.startswith("Distill/") for tag in tags)
    assert "Estimator/contact_bce" not in tags
    assert "Estimator/contact_f1" not in tags


def test_gap_cli_optionally_writes_tensorboard_events(tmp_path: Path) -> None:
    EventAccumulator = pytest.importorskip(
        "tensorboard.backend.event_processing.event_accumulator"
    ).EventAccumulator
    manifest = _manifest(tmp_path)
    _replace_sensor_replay_with_executable_onnx(manifest)
    report = tmp_path / "report.json"
    aggregate = tmp_path / "aggregate.csv"
    log_dir = tmp_path / "cli-tensorboard"

    completed = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).parents[3] / "scripts/rsl_rl/eval_student_teacher_gap.py"),
            str(manifest),
            "--json",
            str(report),
            "--csv",
            str(aggregate),
            "--tensorboard-logdir",
            str(log_dir),
            "--tensorboard-step",
            "9",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert report.is_file() and aggregate.is_file()
    assert "TensorBoard event:" in completed.stdout
    accumulator = EventAccumulator(str(log_dir)).Reload()
    assert accumulator.Scalars("Student/vx_tracking_error")[0].step == 9


def test_gap_rejects_cross_policy_artifact_and_checkpoint_reuse(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    teacher = next(
        run
        for run in payload["runs"]
        if run["policy"] == "teacher_a" and run["seed"] == 42
    )
    ppo = next(
        run
        for run in payload["runs"]
        if run["policy"] == "v2_ppo" and run["seed"] == 42
    )
    teacher["command_csv"] = ppo["command_csv"]
    teacher["episode_csv"] = ppo["episode_csv"]
    teacher["summary_csv"] = ppo["summary_csv"]
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(
        StudentTeacherGapError,
        match="requires agent entry point|reuse the same command|artifact.command_csv_sha256",
    ):
        evaluate_student_teacher_gap(manifest)

    manifest = _manifest(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    teacher_runs = [
        run
        for run in payload["runs"]
        if run["policy"] == "teacher_a" and run["seed"] == 42
    ]
    ppo_run = next(
        run
        for run in payload["runs"]
        if run["policy"] == "v2_ppo" and run["seed"] == 42
    )
    with (tmp_path / ppo_run["summary_csv"]).open(
        newline="", encoding="utf-8"
    ) as stream:
        ppo_summary = {
            row["metric"]: row["value"] for row in csv.DictReader(stream)
        }
    ppo_checkpoint = ppo_summary["checkpoint.sha256"]
    for teacher_run in teacher_runs:
        summary = tmp_path / teacher_run["summary_csv"]
        summary.write_text(
            summary.read_text(encoding="utf-8").replace(
                str(_run_identity("teacher_a", 42)["checkpoint_sha256"]),
                str(ppo_checkpoint),
            ),
            encoding="utf-8",
        )
    with pytest.raises(StudentTeacherGapError, match="reused by policy labels"):
        evaluate_student_teacher_gap(manifest)


def test_gap_rejects_unbound_episode_or_reused_artifact_component(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    ppo = next(
        run
        for run in payload["runs"]
        if run["policy"] == "v2_ppo" and run["seed"] == 42
    )
    episode = tmp_path / ppo["episode_csv"]
    episode.write_text(episode.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(StudentTeacherGapError, match="artifact.episode_csv_sha256"):
        evaluate_student_teacher_gap(manifest)

    manifest = _manifest(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    teacher = next(
        run
        for run in payload["runs"]
        if run["policy"] == "teacher_a" and run["seed"] == 42
    )
    ppo = next(
        run
        for run in payload["runs"]
        if run["policy"] == "v2_ppo" and run["seed"] == 42
    )
    ppo["command_csv"] = teacher["command_csv"]
    ppo_summary = tmp_path / ppo["summary_csv"]
    teacher_command = tmp_path / teacher["command_csv"]
    lines = ppo_summary.read_text(encoding="utf-8").splitlines()
    ppo_summary.write_text(
        "\n".join(
            (
                f"artifact.command_csv_sha256,{_sha256(teacher_command)}"
                if line.startswith("artifact.command_csv_sha256,")
                else line
            )
            for line in lines
        )
        + "\n",
        encoding="utf-8",
    )
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(StudentTeacherGapError, match="reuse the same command CSV"):
        evaluate_student_teacher_gap(manifest)


def test_gap_requires_three_distinct_seeds_and_complete_ablations(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["runs"] = [run for run in payload["runs"] if run["seed"] == 42]
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(StudentTeacherGapError, match="three distinct seeds"):
        evaluate_student_teacher_gap(manifest)

    manifest = _manifest(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["runs"] = [
        run for run in payload["runs"] if run["policy"] != "v2_no_aux"
    ]
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    result = evaluate_student_teacher_gap(manifest)
    assert result["promotion"]["pass"] is False
    assert result["promotion"]["missing_ablations"] == ["v2_no_aux"]


def test_gap_requires_one_protocol_per_domain_across_seeds(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    for run in payload["runs"]:
        if run["seed"] != 43:
            continue
        summary = tmp_path / run["summary_csv"]
        summary.write_text(
            summary.read_text(encoding="utf-8").replace(
                _digest("flat:protocol"), _digest("seed-specific-protocol")
            ),
            encoding="utf-8",
        )

    with pytest.raises(
        StudentTeacherGapError,
        match="evaluation protocol differs across seeds.*flat",
    ):
        evaluate_student_teacher_gap(manifest)


def test_gap_rejects_jointly_loosened_acceptance_thresholds(
    tmp_path: Path,
) -> None:
    manifest = _manifest(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    for run in payload["runs"]:
        summary_path = tmp_path / str(run["summary_csv"])
        summary_rows = list(csv.DictReader(summary_path.open(encoding="utf-8")))
        next(
            row
            for row in summary_rows
            if row["metric"] == "acceptance.max_fall_rate"
        )["value"] = "0.25"
        with summary_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=("metric", "value"))
            writer.writeheader()
            writer.writerows(summary_rows)

    with pytest.raises(
        StudentTeacherGapError,
        match="thresholds differ from the canonical eval_command_sweep protocol",
    ):
        evaluate_student_teacher_gap(manifest)


def test_gap_requires_nominal_and_held_out_domains_and_held_out_candidate(
    tmp_path: Path,
) -> None:
    manifest = _manifest(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["runs"] = [run for run in payload["runs"] if run["domain"] == "flat"]
    payload["deployment_candidate"] = {"seed": 42, "domain": "flat"}
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(StudentTeacherGapError, match="both nominal.*held-out"):
        evaluate_student_teacher_gap(manifest)

    manifest = _manifest(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["deployment_candidate"] = {"seed": 42, "domain": "flat"}
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(StudentTeacherGapError, match="deployment_candidate.*held-out"):
        evaluate_student_teacher_gap(manifest)


def test_gap_gates_teacher_acceptance_and_relative_student_quality(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    teacher = next(
        run
        for run in payload["runs"]
        if run["policy"] == "teacher_a" and run["seed"] == 42
    )
    _rewrite_run_acceptance(tmp_path, teacher, accepted=False)
    result = evaluate_student_teacher_gap(manifest)
    assert result["promotion"]["pass"] is False
    assert result["promotion"]["failed_teacher_runs"] == [[42, "flat"]]

    manifest = _manifest(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    for run in payload["runs"]:
        if run["policy"] != "v2_ppo":
            continue
        summary = tmp_path / run["summary_csv"]
        summary.write_text(
            summary.read_text(encoding="utf-8").replace(
                "tracking.mean_abs_vx,0.05", "tracking.mean_abs_vx,0.50"
            ),
            encoding="utf-8",
        )
    result = evaluate_student_teacher_gap(manifest)
    assert result["promotion"]["pass"] is False
    assert any(
        failure["metric"] == "summary.tracking.mean_abs_vx"
        for failure in result["promotion"]["teacher_gap_gate"]["failures"]
    )


def test_gap_rejects_rehashed_failed_command_row_with_passing_summary(
    tmp_path: Path,
) -> None:
    manifest = _manifest(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    teacher = next(
        run
        for run in payload["runs"]
        if run["policy"] == "teacher_a"
        and run["seed"] == 42
        and run["domain"] == "flat"
    )
    _rewrite_run_acceptance(
        tmp_path,
        teacher,
        accepted=False,
        update_summary_aggregates=False,
    )

    with pytest.raises(
        StudentTeacherGapError,
        match="command_pass_ratio does not match command accept_pass",
    ):
        evaluate_student_teacher_gap(manifest)


def test_gap_rejects_forged_command_accept_pass_after_metric_failure(
    tmp_path: Path,
) -> None:
    manifest = _manifest(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    teacher = next(
        run
        for run in payload["runs"]
        if run["policy"] == "teacher_a"
        and run["seed"] == 42
        and run["domain"] == "flat"
    )
    command_path = tmp_path / str(teacher["command_csv"])
    command_rows = list(csv.DictReader(command_path.open(encoding="utf-8")))
    command_rows[0]["contiguous_success_env_ratio"] = "0.0"
    command_rows[0]["accept_pass"] = "True"
    with command_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(command_rows[0]))
        writer.writeheader()
        writer.writerows(command_rows)
    summary_path = tmp_path / str(teacher["summary_csv"])
    summary_rows = list(csv.DictReader(summary_path.open(encoding="utf-8")))
    next(
        row
        for row in summary_rows
        if row["metric"] == "artifact.command_csv_sha256"
    )["value"] = _sha256(command_path)
    with summary_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("metric", "value"))
        writer.writeheader()
        writer.writerows(summary_rows)

    with pytest.raises(
        StudentTeacherGapError,
        match="accept_pass disagrees with its command acceptance metrics",
    ):
        evaluate_student_teacher_gap(manifest)


def test_gap_rejects_forged_summary_pass_after_command_failure(
    tmp_path: Path,
) -> None:
    manifest = _manifest(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    teacher = next(
        run
        for run in payload["runs"]
        if run["policy"] == "teacher_a"
        and run["seed"] == 42
        and run["domain"] == "flat"
    )
    _rewrite_run_acceptance(tmp_path, teacher, accepted=False)
    summary_path = tmp_path / str(teacher["summary_csv"])
    summary_rows = list(csv.DictReader(summary_path.open(encoding="utf-8")))
    next(
        row
        for row in summary_rows
        if row["metric"] == "acceptance.overall_status"
    )["value"] = "PASS"
    with summary_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("metric", "value"))
        writer.writeheader()
        writer.writerows(summary_rows)

    with pytest.raises(
        StudentTeacherGapError,
        match="overall_status disagrees with recomputed",
    ):
        evaluate_student_teacher_gap(manifest)


def test_gap_applies_teacher_limits_per_seed_domain_without_aggregate_masking(
    tmp_path: Path,
) -> None:
    manifest = _manifest(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    candidate = next(
        run
        for run in payload["runs"]
        if run["policy"] == "v2_ppo"
        and run["seed"] == 42
        and run["domain"] == "held_out"
    )
    summary = tmp_path / candidate["summary_csv"]
    summary.write_text(
        summary.read_text(encoding="utf-8").replace(
            "tracking.mean_abs_vx,0.05", "tracking.mean_abs_vx,0.13"
        ),
        encoding="utf-8",
    )

    result = evaluate_student_teacher_gap(manifest)

    assert result["promotion"]["pass"] is False
    assert any(
        failure.get("seed") == 42
        and failure.get("domain") == "held_out"
        and failure["metric"] == "summary.tracking.mean_abs_vx"
        for failure in result["promotion"]["teacher_gap_gate"]["failures"]
    )


def test_gap_rejects_generic_pass_artifact_without_gate_schema(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    artifact = tmp_path / "generic-pass.json"
    artifact.write_text(json.dumps({"status": "PASS"}), encoding="utf-8")
    payload["promotion_artifacts"]["no_privileged_leak"] = {
        "path": artifact.name,
        "sha256": _sha256(artifact),
    }
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(StudentTeacherGapError, match="canonical schema"):
        evaluate_student_teacher_gap(manifest)
