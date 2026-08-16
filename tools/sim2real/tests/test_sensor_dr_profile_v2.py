from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.sim2real.sensor_dr_profile_v2 import (
    SensorDrProfileV2,
    SensorDrProfileErrorV2,
    apply_sensor_dr_profile_v2,
    load_sensor_dr_profile_v2,
)


def _payload(
    *, purpose: str = "training_curriculum", evidence_sha256: str = "a" * 64
) -> dict:
    return {
        "schema": "redrhex.sensor-dr-profile.v2",
        "profile_id": "bench-imu-2026-08",
        "purpose": purpose,
        "evidence": [
            {
                "artifact": "imu_stationary_summary.json",
                "sha256": evidence_sha256,
                "note": "stationary bench characterization",
            }
        ],
        "parameters": {
            "sensor_dr_gyro_noise_std_range_rad_s": [0.001, 0.004],
            "sensor_dr_encoder_latency_steps_range": [0, 1],
            "domain_randomization_enable": True,
            "dr_randomize_actuator_strength": True,
            "dr_main_actuator_strength_range": [0.95, 1.05],
            "dr_abad_actuator_strength_range": [0.97, 1.03],
            "sim2real_command_delay_steps": 1,
        },
    }


def test_profile_is_hash_bound_purpose_separated_and_applied(tmp_path) -> None:
    evidence = tmp_path / "imu_stationary_summary.json"
    evidence.write_text('{"gyro_std":0.002}\n', encoding="utf-8")
    evidence_digest = hashlib.sha256(evidence.read_bytes()).hexdigest()
    path = tmp_path / "profile.json"
    path.write_text(
        json.dumps(_payload(evidence_sha256=evidence_digest), sort_keys=True) + "\n",
        encoding="utf-8",
    )
    digest = hashlib.sha256(path.read_bytes()).hexdigest()

    profile, actual = load_sensor_dr_profile_v2(
        path,
        expected_sha256=digest,
        expected_purpose="training_curriculum",
    )
    cfg = SimpleNamespace(
        sensor_dr_gyro_noise_std_range_rad_s=(0.0, 0.0),
        sensor_dr_encoder_latency_steps_range=(0, 0),
        domain_randomization_enable=False,
        dr_randomize_actuator_strength=False,
        dr_main_actuator_strength_range=(1.0, 1.0),
        dr_abad_actuator_strength_range=(1.0, 1.0),
        sim2real_command_delay_steps=0,
        sensor_dr_evidence="",
        curriculum_stage_scales=[0.05],
    )
    apply_sensor_dr_profile_v2(cfg, profile, actual)

    assert cfg.sensor_dr_gyro_noise_std_range_rad_s == (0.001, 0.004)
    assert cfg.sensor_dr_encoder_latency_steps_range == (0, 1)
    assert cfg.dr_randomize_actuator_strength is True
    assert cfg.curriculum_stage_scales == [1.0]
    assert cfg.sensor_dr_physical_stage_scale == 1.0
    assert cfg.sensor_dr_profile_sha256 == digest
    assert cfg.sensor_dr_profile_purpose == "training_curriculum"
    assert cfg.sensor_dr_require_physical_material_writes is False
    assert evidence_digest in cfg.sensor_dr_evidence


def test_profile_rejects_tampering_wrong_domain_and_unknown_or_guessed_data(tmp_path) -> None:
    evidence = tmp_path / "imu_stationary_summary.json"
    evidence.write_text("evidence-v1\n", encoding="utf-8")
    evidence_digest = hashlib.sha256(evidence.read_bytes()).hexdigest()
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(_payload(evidence_sha256=evidence_digest)) + "\n", encoding="utf-8")

    with pytest.raises(SensorDrProfileErrorV2, match="SHA-256 mismatch"):
        load_sensor_dr_profile_v2(path, expected_sha256="0" * 64)
    with pytest.raises(SensorDrProfileErrorV2, match="purpose"):
        load_sensor_dr_profile_v2(path, expected_purpose="held_out_evaluation")

    unknown = _payload(evidence_sha256=evidence_digest)
    unknown["parameters"]["made_up_noise"] = [0.0, 1.0]
    path.write_text(json.dumps(unknown) + "\n", encoding="utf-8")
    with pytest.raises(SensorDrProfileErrorV2, match="unsupported"):
        load_sensor_dr_profile_v2(path)

    no_evidence = _payload(evidence_sha256=evidence_digest)
    no_evidence["evidence"] = []
    path.write_text(json.dumps(no_evidence) + "\n", encoding="utf-8")
    with pytest.raises(SensorDrProfileErrorV2, match="evidence"):
        load_sensor_dr_profile_v2(path)


def test_probability_and_latency_ranges_fail_closed(tmp_path) -> None:
    evidence = tmp_path / "imu_stationary_summary.json"
    evidence.write_text("evidence-v1\n", encoding="utf-8")
    evidence_digest = hashlib.sha256(evidence.read_bytes()).hexdigest()
    payload = _payload(evidence_sha256=evidence_digest)
    payload["parameters"] = {
        "sensor_dr_encoder_dropout_probability_range": [0.0, 1.2]
    }
    path = tmp_path / "bad_probability.json"
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    with pytest.raises(SensorDrProfileErrorV2, match=r"\[0, 1\]"):
        load_sensor_dr_profile_v2(path)

    payload = _payload(evidence_sha256=evidence_digest)
    payload["parameters"] = {"sensor_dr_encoder_latency_steps_range": [-1, 1]}
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    with pytest.raises(SensorDrProfileErrorV2, match="non-negative"):
        load_sensor_dr_profile_v2(path)


def test_profile_rejects_missing_or_tampered_evidence_artifact(tmp_path: Path) -> None:
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(_payload()) + "\n", encoding="utf-8")
    with pytest.raises(SensorDrProfileErrorV2, match="does not exist"):
        load_sensor_dr_profile_v2(path)

    evidence = tmp_path / "imu_stationary_summary.json"
    evidence.write_text("actual evidence\n", encoding="utf-8")
    with pytest.raises(SensorDrProfileErrorV2, match="evidence SHA-256 mismatch"):
        load_sensor_dr_profile_v2(path)


def test_profile_rejects_inert_or_unenabled_physical_ranges(tmp_path: Path) -> None:
    evidence = tmp_path / "imu_stationary_summary.json"
    evidence.write_text("evidence-v1\n", encoding="utf-8")
    evidence_digest = hashlib.sha256(evidence.read_bytes()).hexdigest()
    path = tmp_path / "profile.json"

    inert = _payload(evidence_sha256=evidence_digest)
    inert["parameters"] = {"domain_randomization_enable": True}
    path.write_text(json.dumps(inert) + "\n", encoding="utf-8")
    with pytest.raises(SensorDrProfileErrorV2, match="requires an explicit physical DR flag"):
        load_sensor_dr_profile_v2(path)

    unenabled = _payload(evidence_sha256=evidence_digest)
    unenabled["parameters"] = {"dr_mass_range": [0.9, 1.1]}
    path.write_text(json.dumps(unenabled) + "\n", encoding="utf-8")
    with pytest.raises(SensorDrProfileErrorV2, match="requires domain_randomization_enable"):
        load_sensor_dr_profile_v2(path)

    fallback_only = _payload(evidence_sha256=evidence_digest)
    fallback_only["parameters"] = {
        "domain_randomization_enable": True,
        "dr_randomize_friction": True,
        "dr_friction_range": [0.9, 1.1],
    }
    path.write_text(json.dumps(fallback_only) + "\n", encoding="utf-8")
    with pytest.raises(SensorDrProfileErrorV2, match="physical_material_randomization"):
        load_sensor_dr_profile_v2(path)


def test_command_delay_is_latency_not_actuator_strength_evidence() -> None:
    payload = _payload()
    payload["parameters"] = {
        "sensor_dr_gyro_noise_std_range_rad_s": [0.001, 0.004],
        "sim2real_command_delay_steps": 2,
    }

    profile = SensorDrProfileV2.from_mapping(payload)

    assert "latency" in profile.active_categories
    assert "actuator" not in profile.active_categories

def test_training_and_evaluation_cli_bind_separate_profile_purposes() -> None:
    root = Path(__file__).parents[3]
    train = (root / "scripts/rsl_rl/train.py").read_text(encoding="utf-8")
    evaluate = (root / "scripts/rsl_rl/eval_command_sweep.py").read_text(encoding="utf-8")

    for source in (train, evaluate):
        assert '"--sensor-dr-profile"' in source
        assert '"--sensor-dr-profile-sha256"' in source
        assert "apply_sensor_dr_profile_v2(" in source
        assert "PHYSICAL_PROFILE_PARAMETERS_V2" in source
    assert 'expected_purpose="training_curriculum"' in train
    assert 'expected_purpose="held_out_evaluation"' in evaluate
    assert '"sensor_dr.profile_sha256"' in evaluate
