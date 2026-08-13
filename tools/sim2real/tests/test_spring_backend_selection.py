from __future__ import annotations

import json
import math
from types import SimpleNamespace

import numpy as np
import pytest

from tools.sim2real.contracts import ContractError


def _release_arrays(
    *,
    passivity_ratio: float = 0.0,
    peak_scale: float = 1.0,
    applied_sign: float = 1.0,
    fixture_position_error_rad: float = 0.0,
    fixture_velocity_rad_s: float = 0.0,
    unwrap_ambiguous: float = 0.0,
) -> dict[str, np.ndarray]:
    deflection = np.zeros((8, 6), dtype=np.float64)
    deflection[:, 0] = np.array(
        [0.1, -0.04, -0.01, 0.0, -0.2, 0.08, 0.02, 0.0]
    )
    deflection[1, 0] *= peak_scale
    deflection[5, 0] *= peak_scale
    model_torque = -200.0 * deflection
    applied_torque = model_torque.copy()
    applied_torque[[0, 4], 0] *= applied_sign
    energy = 0.5 * 200.0 * np.square(deflection)
    mechanical_power = np.zeros_like(deflection)
    for start, stop in ((0, 4), (4, 8)):
        for index in range(start + 1, stop):
            energy_delta = energy[index, 0] - energy[index - 1, 0]
            mechanical_power[index, 0] = (
                -2.0 * energy_delta / (1.0 / 120.0)
                - mechanical_power[index - 1, 0]
            )
    release_start = np.zeros(8, dtype=np.float64)
    release_start[[0, 4]] = 1.0
    residual = np.zeros((8, 1), dtype=np.float64)
    residual[1, 0] = passivity_ratio * energy[0].sum()
    residual[5, 0] = passivity_ratio * energy[4].sum()
    return {
        "spring_deflection": deflection,
        "spring_model_torque": model_torque,
        "spring_applied_torque_estimate": applied_torque,
        "spring_potential_energy": energy,
        "spring_mechanical_power": mechanical_power,
        "spring_passivity_residual": residual,
        "spring_release_start": release_start,
        "spring_fixture_position_error": np.full(
            8, fixture_position_error_rad, dtype=np.float64
        ),
        "spring_fixture_velocity": np.full(
            8, fixture_velocity_rad_s, dtype=np.float64
        ),
        "spring_unwrap_ambiguous": np.full(
            8, unwrap_ambiguous, dtype=np.float64
        ),
    }


def test_release_gate_accepts_linear_passive_trace() -> None:
    from tools.sim2real.spring_backend_selection import evaluate_release_arrays

    report = evaluate_release_arrays(
        _release_arrays(),
        backend="native",
        physics_dt_s=1.0 / 120.0,
        calibration_status="calibrated",
        profile_sha256="a" * 64,
    )

    assert report["passed"] is True
    assert report["static_torque_rmse_fraction"] == pytest.approx(0.0)
    assert report["energy_creation_fraction"] == pytest.approx(0.0)
    assert report["energy_work_residual_fraction"] == pytest.approx(0.0)
    assert report["restoring_sign_correct"] is True
    assert report["gates"]["completed_rebound"] is True


def test_release_gate_rejects_a_frozen_joint_without_a_completed_rebound() -> None:
    from tools.sim2real.spring_backend_selection import evaluate_release_arrays

    arrays = _release_arrays()
    arrays["spring_deflection"][:4, 0] = 0.1
    arrays["spring_deflection"][4:, 0] = -0.2
    arrays["spring_model_torque"] = -200.0 * arrays["spring_deflection"]
    arrays["spring_applied_torque_estimate"] = arrays["spring_model_torque"].copy()
    arrays["spring_potential_energy"] = (
        0.5 * 200.0 * np.square(arrays["spring_deflection"])
    )
    arrays["spring_mechanical_power"].fill(0.0)
    arrays["spring_passivity_residual"].fill(0.0)

    report = evaluate_release_arrays(
        arrays,
        backend="native",
        physics_dt_s=1.0 / 120.0,
        calibration_status="calibrated",
        profile_sha256="a" * 64,
    )

    assert report["passed"] is False
    assert report["gates"]["completed_rebound"] is False


@pytest.mark.parametrize(
    ("arrays", "failed_gate"),
    [
        (_release_arrays(applied_sign=-1.0), "restoring_sign"),
        (_release_arrays(passivity_ratio=0.021), "energy_creation"),
        (_release_arrays(passivity_ratio=-0.021), "energy_work_residual"),
        (_release_arrays(peak_scale=3.0), "runaway"),
        (
            _release_arrays(fixture_position_error_rad=2.0e-4),
            "fixture_position",
        ),
        (
            _release_arrays(fixture_velocity_rad_s=2.0e-4),
            "fixture_velocity",
        ),
        (_release_arrays(unwrap_ambiguous=1.0), "unwrap_unambiguous"),
    ],
)
def test_release_gate_rejects_wrong_sign_energy_creation_and_runaway(
    arrays: dict[str, np.ndarray], failed_gate: str
) -> None:
    from tools.sim2real.spring_backend_selection import evaluate_release_arrays

    report = evaluate_release_arrays(
        arrays,
        backend="explicit",
        physics_dt_s=1.0 / 120.0,
        calibration_status="calibrated",
        profile_sha256="a" * 64,
    )

    assert report["passed"] is False
    assert report["gates"][failed_gate] is False


def _backend_report(
    backend: str,
    hz: int,
    *,
    residual: float,
    peaks: tuple[float, ...] = (0.1, 0.2),
    passed: bool = True,
    calibrated: bool = True,
) -> dict[str, object]:
    return {
        "backend": backend,
        "physics_hz": hz,
        "physics_dt_s": 1.0 / hz,
        "calibration_status": "calibrated" if calibrated else "uncalibrated",
        "profile_sha256": "a" * 64 if calibrated else None,
        "seed": 0,
        "runtime_identity_sha256": "b" * 64,
        "effective_parameter_sha256": "c" * 64,
        "passed": passed,
        "energy_creation_fraction": residual,
        "energy_work_residual_fraction": residual,
        "release_peak_angles_rad": list(peaks),
        "gates": {},
    }


def test_selection_chooses_lower_residual_and_explicit_within_ten_percent() -> None:
    from tools.sim2real.spring_backend_selection import select_spring_backend

    native_wins = select_spring_backend(
        [
            _backend_report("explicit", 120, residual=0.01),
            _backend_report("explicit", 240, residual=0.01),
            _backend_report("native", 120, residual=0.005),
            _backend_report("native", 240, residual=0.005),
        ]
    )
    assert native_wins["selected_backend"] == "native"
    assert native_wins["eligible"] is True

    explicit_tie_break = select_spring_backend(
        [
            _backend_report("explicit", 120, residual=0.0095),
            _backend_report("explicit", 240, residual=0.0095),
            _backend_report("native", 120, residual=0.009),
            _backend_report("native", 240, residual=0.009),
        ]
    )
    assert explicit_tie_break["selected_backend"] == "explicit"


def test_selection_blocks_uncalibrated_failed_or_timestep_sensitive_runs() -> None:
    from tools.sim2real.spring_backend_selection import select_spring_backend

    uncalibrated = [
        _backend_report(backend, hz, residual=0.0, calibrated=False)
        for backend in ("explicit", "native")
        for hz in (120, 240)
    ]
    assert select_spring_backend(uncalibrated)["status"] == "blocked_uncalibrated"

    failed = [
        _backend_report(
            backend,
            hz,
            residual=0.0,
            passed=not (backend == "explicit" and hz == 120),
        )
        for backend in ("explicit", "native")
        for hz in (120, 240)
    ]
    assert select_spring_backend(failed)["status"] == "blocked_physics_validation"

    timestep_sensitive = [
        _backend_report(
            backend,
            hz,
            residual=0.0,
            peaks=(0.1, 0.2) if hz == 120 else (0.11, 0.2),
        )
        for backend in ("explicit", "native")
        for hz in (120, 240)
    ]
    result = select_spring_backend(timestep_sensitive)
    assert result["status"] == "blocked_physics_validation"
    assert result["backends"]["explicit"]["peak_angle_difference_fraction"] > 0.02


def test_selection_requires_exact_backend_hz_matrix() -> None:
    from tools.sim2real.spring_backend_selection import select_spring_backend

    with pytest.raises(ContractError, match="exactly one"):
        select_spring_backend(
            [_backend_report("native", 120, residual=0.0)]
        )


def test_rebound_peak_uses_quadratic_extremum_instead_of_first_sample() -> None:
    from tools.sim2real.spring_backend_selection import _first_rebound_peak

    def sample(dt: float) -> np.ndarray:
        time = np.arange(0.0, 0.035, dt)
        return 300.0 * np.square(time - 0.012) - 0.04

    low = _first_rebound_peak(sample(1.0 / 120.0))
    high = _first_rebound_peak(sample(1.0 / 240.0))

    assert low == pytest.approx(0.04, abs=1.0e-12)
    assert high == pytest.approx(0.04, abs=1.0e-12)


def test_selection_rejects_mismatched_seed_or_runtime_identity() -> None:
    from tools.sim2real.spring_backend_selection import select_spring_backend

    reports = [
        _backend_report(backend, hz, residual=0.0)
        for backend in ("explicit", "native")
        for hz in (120, 240)
    ]
    reports[0]["seed"] = 1
    result = select_spring_backend(reports)
    assert result["eligible"] is False
    assert result["seeds_match"] is False

    reports[0]["seed"] = 0
    reports[0]["runtime_identity_sha256"] = "d" * 64
    result = select_spring_backend(reports)
    assert result["eligible"] is False
    assert result["runtime_identity_matches"] is False


def test_release_loader_binds_seed_timestep_fixture_and_parameters_to_audit(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import tools.sim2real.spring_backend_selection as selection
    from tools.sim2real.traces import sha256_json

    joint_names = ["Revolute_5", *[f"joint_{index}" for index in range(17)]]
    parameters = {
        "backend": "explicit",
        "joint_order": [f"damper_{index}" for index in range(6)],
        "runtime_joint_names": [
            "Revolute_5",
            "Revolute_8",
            "Revolute_13",
            "Revolute_25",
            "Revolute_26",
            "Revolute_27",
        ],
        "stiffness_nm_per_rad": [200.0] * 6,
        "damping_nm_s_per_rad": [0.0] * 6,
        "neutral_angle_rad": [0.0] * 6,
        "locked_joint_names": joint_names[1:],
        "fixture_limit_half_width_rad": 1.0e-6,
        "fixture_selected_limit_rad": [None, None],
        "fixture_selected_range_kind": "continuous",
        "fixture_constraint_kind": "excluded_fixed_joint",
    }
    audit = {
        "mode": "fixed-base",
        "is_fixed_base": True,
        "physics_dt_s": 1.0 / 120.0,
        "joint_names": joint_names,
        "torsion_springs": parameters,
    }
    audit_hash = sha256_json(audit)
    metadata = {
        "spring_backend": "explicit",
        "calibration_status": "uncalibrated",
        "profile_sha256": None,
        "seed": 0,
        "calibration_constants": {
            "physics_dt_s": 1.0 / 120.0,
            "seed": 0,
            "runtime_audit_sha256": audit_hash,
        },
        **{
            name: "runtime"
            for name in (
                "git_sha",
                "asset_sha256",
                "config_sha256",
                "redrhex_module_sha256",
                "isaaclab_version",
                "isaacsim_version",
                    "characterization_runner_sha256",
                    "torsion_spring_model_sha256",
                    "runtime_bundle_sha256",
            )
        },
    }
    fake_loaded = SimpleNamespace(
        manifest=SimpleNamespace(
            source="sim",
            metadata=metadata,
            provenance={"trace_sha256": "a" * 64},
        ),
        arrays=_release_arrays(),
        metadata_sha256="b" * 64,
    )
    monkeypatch.setattr(selection, "load_trace", lambda *_args, **_kwargs: fake_loaded)
    (tmp_path / "runtime_audit.json").write_text(json.dumps(audit), encoding="utf-8")
    results = {
        "spring_backend": "explicit",
        "calibration_status": "uncalibrated",
        "profile_sha256": None,
        "seed": 0,
        "physics_dt_s": 1.0 / 120.0,
        "runtime_audit": "runtime_audit.json",
        "runtime_audit_sha256": audit_hash,
        "effective_spring_parameters": parameters,
    }
    (tmp_path / "results.json").write_text(json.dumps(results), encoding="utf-8")

    report = selection.load_release_report(tmp_path)
    assert report["seed"] == 0
    assert report["effective_parameter_sha256"]
    assert report["runtime_identity_sha256"]

    audit["physics_dt_s"] = 1.0 / 240.0
    (tmp_path / "runtime_audit.json").write_text(json.dumps(audit), encoding="utf-8")
    with pytest.raises(ContractError, match="runtime audit hash mismatch"):
        selection.load_release_report(tmp_path)


def test_cli_parses_the_four_release_runs_and_output() -> None:
    from tools.sim2real.cli import build_parser

    args = build_parser().parse_args(
        [
            "select-spring-backend",
            "--explicit-120",
            "/tmp/explicit-120",
            "--explicit-240",
            "/tmp/explicit-240",
            "--native-120",
            "/tmp/native-120",
            "--native-240",
            "/tmp/native-240",
            "--output",
            "/tmp/selection.json",
        ]
    )

    assert args.command == "select-spring-backend"
    assert args.explicit_120.name == "explicit-120"
    assert args.explicit_240.name == "explicit-240"
    assert args.native_120.name == "native-120"
    assert args.native_240.name == "native-240"
    assert args.output.name == "selection.json"


def test_cli_selection_writes_once_and_returns_the_report(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tools.sim2real.cli import _run, build_parser
    import tools.sim2real.spring_backend_selection as selection

    expected = {"schema_version": 1, "eligible": False, "status": "blocked_uncalibrated"}
    monkeypatch.setattr(selection, "evaluate_backend_runs", lambda **_: expected)
    output = tmp_path / "selection.json"
    args = build_parser().parse_args(
        [
            "select-spring-backend",
            "--explicit-120",
            "e120",
            "--explicit-240",
            "e240",
            "--native-120",
            "n120",
            "--native-240",
            "n240",
            "--output",
            str(output),
        ]
    )

    assert _run(args) == expected
    assert __import__("json").loads(output.read_text(encoding="utf-8")) == expected
    with pytest.raises(ValueError, match="output already exists"):
        _run(args)
