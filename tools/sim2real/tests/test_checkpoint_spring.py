from __future__ import annotations

import json
from pathlib import Path

import pytest


PROFILE_SHA256 = "a" * 64
SPRING_ALIASES = [f"damper_{index}" for index in range(6)]


def _validate(
    run_dir: Path,
    *,
    backend: str = "explicit",
    profile_id: str | None = None,
    profile_sha256: str | None = None,
) -> str:
    from tools.sim2real.checkpoint_spring import validate_checkpoint_spring_evaluation

    return validate_checkpoint_spring_evaluation(
        run_dir,
        selected_backend=backend,
        selected_profile_id=profile_id,
        selected_profile_sha256=profile_sha256,
    )


def _spring_metadata(run_dir: Path, *, backend: str, status: str) -> None:
    params = run_dir / "params"
    params.mkdir(parents=True, exist_ok=True)
    (params / "torsion_spring.yaml").write_text(
        f"spring_backend: {backend}\ncalibration_status: {status}\n",
        encoding="utf-8",
    )


def _profile_metadata(run_dir: Path, *, schema_version: int = 1) -> None:
    (run_dir / "params" / "physics_profile_metadata.json").write_text(
        json.dumps(
            {
                "schema_version": schema_version,
                "profile_id": "measured-spring",
                "profile_sha256": PROFILE_SHA256,
            }
        ),
        encoding="utf-8",
    )


def _deployment_snapshot(run_dir: Path) -> None:
    from tools.sim2real.contracts import CalibrationProfileV1
    from tools.sim2real.traces import sha256_json

    source = {
        "trace_sha256": "a" * 64,
        "metadata_sha256": "b" * 64,
        "scenario_id": "torsion-spring",
        "scenario_sha256": "c" * 64,
        "source": "real",
        "metric_kind": "torsional_spring",
        "frame": "damper_0",
        "repeat_count": 3,
        "dataset_id": "spring-bench",
        "episode_id": "spring-calibration",
        "applies_to": SPRING_ALIASES,
        "rest_position_rad": 0.7853981633974483,
        "episode_path": "/immutable/calibration",
        "quality_validation": {
            "accepted": True,
            "gates": {
                "r_squared": True,
                "heldout_rmse": True,
                "stiffness_cv": True,
                "hysteresis": True,
                "neutral_model_heldout_rmse": True,
            },
            "calibration_trace_sha256": "a" * 64,
            "calibration_metadata_sha256": "b" * 64,
            "holdout_trace_sha256": "d" * 64,
            "holdout_metadata_sha256": "e" * 64,
            "holdout_scenario_id": "torsion-spring-holdout",
            "holdout_scenario_sha256": "f" * 64,
            "source": "real",
            "dataset_id": "spring-bench",
            "episode_id": "spring-holdout",
            "episode_path": "/immutable/holdout",
        },
    }
    profile = {
        "schema_version": 1,
        "profile_id": "measured-spring",
        "hardware_mapping": {},
        "sensor_timing": {},
        "simulation_physics": {
            "passive_spring": {
                alias: {"stiffness": 200.0, "damping": 0.0}
                for alias in SPRING_ALIASES
            }
        },
        "measurement_sources": {"passive_spring:damper_0": source},
    }
    profile = CalibrationProfileV1.from_dict(profile).to_dict()
    params = run_dir / "params"
    params.mkdir(parents=True, exist_ok=True)
    (params / "physics_profile.json").write_text(
        json.dumps(profile), encoding="utf-8"
    )
    profile_sha256 = sha256_json(profile)
    (params / "physics_profile_metadata.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "profile_id": "measured-spring",
                "profile_sha256": profile_sha256,
            }
        ),
        encoding="utf-8",
    )
    (params / "torsion_spring.yaml").write_text(
        "\n".join(
            (
                "spring_backend: native",
                "calibration_status: calibrated",
                "profile_id: measured-spring",
                f"profile_sha256: {profile_sha256}",
                "joint_aliases: [damper_0, damper_1, damper_2, damper_3, damper_4, damper_5]",
                "stiffness_nm_per_rad: [200, 200, 200, 200, 200, 200]",
                "damping_nm_s_per_rad: [0, 0, 0, 0, 0, 0]",
                "neutral_angle_rad: [0.7853981633974483, 0.7853981633974483, -0.7853981633974483, 0.7853981633974483, 0.7853981633974483, 0.7853981633974483]",
            )
        ),
        encoding="utf-8",
    )


def test_checkpoint_without_spring_metadata_is_legacy_uncalibrated_compatible(
    tmp_path: Path,
) -> None:
    assert _validate(tmp_path) == "uncalibrated"


def test_uncalibrated_checkpoint_does_not_require_a_profile(tmp_path: Path) -> None:
    _spring_metadata(tmp_path, backend="explicit", status="uncalibrated")

    assert _validate(tmp_path, backend="native") == "uncalibrated"


def test_deployment_rejects_uncalibrated_or_legacy_checkpoint(tmp_path: Path) -> None:
    from tools.sim2real.checkpoint_spring import validate_checkpoint_spring_deployment

    with pytest.raises(ValueError, match="calibrated torsion-spring"):
        validate_checkpoint_spring_deployment(tmp_path)
    _spring_metadata(tmp_path, backend="explicit", status="uncalibrated")
    with pytest.raises(ValueError, match="calibrated torsion-spring"):
        validate_checkpoint_spring_deployment(tmp_path)


def test_deployment_reopens_bound_profile_and_spring_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import tools.sim2real.checkpoint_spring as checkpoint_spring

    _deployment_snapshot(tmp_path)
    monkeypatch.setattr(
        checkpoint_spring,
        "verify_representative_spring_source",
        lambda _source: {
            "calibration": {
                "neutral_stiffness_nm_per_rad": 200.0,
                "rest_position_rad": 0.7853981633974483,
            },
            "quality": {"accepted": True},
        },
    )

    result = checkpoint_spring.validate_checkpoint_spring_deployment(tmp_path)

    assert result["spring_backend"] == "native"
    assert result["calibration_status"] == "calibrated"
    assert result["profile_id"] == "measured-spring"
    assert len(result["profile_sha256"]) == 64

    spring_path = tmp_path / "params" / "torsion_spring.yaml"
    spring_path.write_text(
        spring_path.read_text(encoding="utf-8").replace(
            "stiffness_nm_per_rad: [200, 200, 200, 200, 200, 200]",
            "stiffness_nm_per_rad: [201, 200, 200, 200, 200, 200]",
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="stiffness"):
        checkpoint_spring.validate_checkpoint_spring_deployment(tmp_path)


def test_calibrated_checkpoint_accepts_exact_backend_and_profile(tmp_path: Path) -> None:
    _spring_metadata(tmp_path, backend="native", status="calibrated")
    _profile_metadata(tmp_path)

    assert (
        _validate(
            tmp_path,
            backend="native",
            profile_id="measured-spring",
            profile_sha256=PROFILE_SHA256,
        )
        == "calibrated"
    )


@pytest.mark.parametrize(
    ("backend", "profile_id", "profile_sha256", "message"),
    [
        ("explicit", "measured-spring", PROFILE_SHA256, "backend"),
        ("native", None, None, "requires --physics-profile"),
        ("native", "different-profile", PROFILE_SHA256, "profile_id"),
        ("native", "measured-spring", "b" * 64, "profile_sha256"),
    ],
)
def test_calibrated_checkpoint_rejects_changed_evaluation_physics(
    tmp_path: Path,
    backend: str,
    profile_id: str | None,
    profile_sha256: str | None,
    message: str,
) -> None:
    _spring_metadata(tmp_path, backend="native", status="calibrated")
    _profile_metadata(tmp_path)

    with pytest.raises(ValueError, match=message):
        _validate(
            tmp_path,
            backend=backend,
            profile_id=profile_id,
            profile_sha256=profile_sha256,
        )


@pytest.mark.parametrize("metadata_kind", ("missing", "malformed", "unsupported-schema"))
def test_calibrated_checkpoint_rejects_invalid_profile_binding_metadata(
    tmp_path: Path, metadata_kind: str
) -> None:
    _spring_metadata(tmp_path, backend="native", status="calibrated")
    if metadata_kind == "malformed":
        (tmp_path / "params" / "physics_profile_metadata.json").write_text(
            "{not-json", encoding="utf-8"
        )
    elif metadata_kind == "unsupported-schema":
        _profile_metadata(tmp_path, schema_version=999)

    with pytest.raises(ValueError, match="profile metadata"):
        _validate(
            tmp_path,
            backend="native",
            profile_id="measured-spring",
            profile_sha256=PROFILE_SHA256,
        )


@pytest.mark.parametrize(
    "payload",
    [
        "spring_backend: invalid\ncalibration_status: calibrated\n",
        "spring_backend: explicit\ncalibration_status: unknown\n",
        "spring_backend: [unterminated\n",
        "- not\n- an\n- object\n",
    ],
)
def test_malformed_new_checkpoint_spring_metadata_fails_closed(
    tmp_path: Path, payload: str
) -> None:
    params = tmp_path / "params"
    params.mkdir()
    (params / "torsion_spring.yaml").write_text(payload, encoding="utf-8")

    with pytest.raises(ValueError, match="torsion-spring metadata"):
        _validate(tmp_path)
