from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from tools.sim2real.contracts import CalibrationProfileV1, ContractError
from tools.sim2real.cli import main
from tools.sim2real.promotion import evaluate_promotion
from tools.sim2real.traces import sha256_json


def _profile() -> CalibrationProfileV1:
    return CalibrationProfileV1.from_dict(
        {
            "schema_version": 1,
            "profile_id": "candidate",
            "hardware_mapping": {},
            "sensor_timing": {},
            "simulation_physics": {},
        }
    )


def _condition(
    condition_id: str,
    subsystem: str,
    role: str,
    *,
    held_out_by: list[str] | None = None,
    sim_value: float = 1.05,
) -> dict:
    condition = {
        "condition_id": condition_id,
        "subsystem": subsystem,
        "role": role,
        "real_episodes": [
            {
                "episode_id": f"{condition_id}-real",
                "trace_sha256": "1" * 64,
                "repetition_count": 3,
            }
        ],
        "metrics": {},
    }
    if role == "holdout":
        condition["held_out_by"] = held_out_by or ["leg"]
        condition["sim_trace_sha256"] = "2" * 64
        condition["metrics"] = {
            "steady_speed_rad_s": {
                "unit": "rad/s",
                "real_mean": 1.0,
                "real_std": 0.02,
                "instrument_uncertainty": 0.1,
                "sim_value": sim_value,
            }
        }
    return condition


def _evidence(profile: CalibrationProfileV1 | None = None) -> dict:
    profile = profile or _profile()
    return {
        "schema_version": 1,
        "candidate_profile_sha256": sha256_json(profile.to_dict()),
        "audit": {
            "units_pass": True,
            "frames_pass": True,
            "joint_sign_pass": True,
            "mass_pass": True,
            "imu_mount_pass": True,
            "contact_sensor_pass": True,
        },
        "fitted_subsystems": ["main_drive", "contact"],
        "conditions": [
            _condition("main-cal", "main_drive", "calibration"),
            _condition("main-held", "main_drive", "holdout"),
            _condition("contact-cal", "contact", "calibration"),
            _condition(
                "contact-held", "contact", "holdout", held_out_by=["load"]
            ),
        ],
        "actuator_model_envelope": {"main_drive": "inside"},
    }


def test_passing_evidence_reports_separate_subsystems_without_promoting() -> None:
    profile = _profile()

    result = evaluate_promotion(profile, _evidence(profile))

    assert result["eligible_for_review"] is True
    assert result["promotion_requires_reviewed_config_change"] is True
    assert set(result["subsystems"]) == {"main_drive", "contact"}
    assert all(item["pass"] for item in result["subsystems"].values())
    assert "score" not in str(result).lower()
    metric = result["subsystems"]["main_drive"]["holdout_conditions"][0][
        "metrics"
    ]["steady_speed_rad_s"]
    assert metric["tolerance"] == pytest.approx(0.1)
    assert metric["absolute_error"] == pytest.approx(0.05)
    assert metric["pass"] is True


@pytest.mark.parametrize(
    ("mutate", "failure"),
    [
        (
            lambda evidence: evidence["audit"].__setitem__("mass_pass", False),
            "audit.mass_pass",
        ),
        (
            lambda evidence: evidence["conditions"][0]["real_episodes"][0].__setitem__(
                "repetition_count", 2
            ),
            "three real repetitions",
        ),
        (
            lambda evidence: evidence["conditions"].__setitem__(
                slice(None),
                [
                    item
                    for item in evidence["conditions"]
                    if not (
                        item["subsystem"] == "contact" and item["role"] == "holdout"
                    )
                ],
            ),
            "contact.*holdout",
        ),
        (
            lambda evidence: evidence["conditions"][1]["metrics"][
                "steady_speed_rad_s"
            ].__setitem__("sim_value", 1.2),
            "steady_speed_rad_s.*outside",
        ),
    ],
    ids=("audit-blocker", "too-few-repeats", "missing-holdout", "metric-envelope"),
)
def test_promotion_gate_fails_closed_with_actionable_subsystem_reasons(
    mutate, failure: str
) -> None:
    profile = _profile()
    evidence = copy.deepcopy(_evidence(profile))
    mutate(evidence)

    result = evaluate_promotion(profile, evidence)

    assert result["eligible_for_review"] is False
    assert any(__import__("re").search(failure, reason) for reason in result["failures"])


def test_actuator_model_mismatch_is_reported_explicitly() -> None:
    profile = _profile()
    evidence = _evidence(profile)
    evidence["actuator_model_envelope"]["main_drive"] = "mismatch"

    result = evaluate_promotion(profile, evidence)

    assert result["eligible_for_review"] is False
    assert result["subsystems"]["main_drive"]["actuator_model_mismatch"] is True
    assert any("actuator-model mismatch" in reason for reason in result["failures"])


def test_evidence_rejects_wrong_profile_hash_unknown_fields_and_fake_holdout() -> None:
    profile = _profile()
    wrong_hash = _evidence(profile)
    wrong_hash["candidate_profile_sha256"] = "0" * 64
    with pytest.raises(ContractError, match="candidate profile hash"):
        evaluate_promotion(profile, wrong_hash)

    unknown = _evidence(profile)
    unknown["automatic_promotion"] = True
    with pytest.raises(ContractError, match="unknown validation evidence fields"):
        evaluate_promotion(profile, unknown)

    fake_holdout = _evidence(profile)
    fake_holdout["conditions"][1]["held_out_by"] = []
    with pytest.raises(ContractError, match="held_out_by"):
        evaluate_promotion(profile, fake_holdout)


def test_duplicate_condition_or_episode_identity_is_rejected() -> None:
    profile = _profile()
    duplicate_condition = _evidence(profile)
    duplicate_condition["conditions"][1]["condition_id"] = "main-cal"
    with pytest.raises(ContractError, match="condition_id.*unique"):
        evaluate_promotion(profile, duplicate_condition)

    duplicate_episode = _evidence(profile)
    duplicate_episode["conditions"][0]["real_episodes"].append(
        copy.deepcopy(duplicate_episode["conditions"][0]["real_episodes"][0])
    )
    with pytest.raises(ContractError, match="episode_id.*unique"):
        evaluate_promotion(profile, duplicate_episode)


def test_validate_promotion_cli_writes_report_and_returns_nonzero_on_failure(
    tmp_path: Path, capsys
) -> None:
    profile = _profile()
    profile_path = tmp_path / "profile.json"
    evidence_path = tmp_path / "evidence.json"
    output_path = tmp_path / "report.json"
    profile_path.write_text(json.dumps(profile.to_dict()), encoding="utf-8")
    evidence = _evidence(profile)
    evidence["audit"]["frames_pass"] = False
    evidence_path.write_text(json.dumps(evidence), encoding="utf-8")

    code = main(
        [
            "validate-promotion",
            str(profile_path),
            str(evidence_path),
            "--output",
            str(output_path),
        ]
    )

    emitted = json.loads(capsys.readouterr().out)
    persisted = json.loads(output_path.read_text(encoding="utf-8"))
    assert code == 3
    assert emitted == persisted
    assert emitted["eligible_for_review"] is False
