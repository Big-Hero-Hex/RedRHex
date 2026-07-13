from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from tools.sim2real.cli import main
from tools.sim2real.compare import compare_traces
from tools.sim2real.contracts import CalibrationProfileV1, ContractError
from tools.sim2real.promotion import evaluate_promotion
from tools.sim2real.scenarios import load_scenario
from tools.sim2real.traces import load_trace, sha256_file, sha256_json, write_trace


def _profile(profile_id: str, damping: float) -> CalibrationProfileV1:
    return CalibrationProfileV1.from_dict(
        {
            "schema_version": 1,
            "profile_id": profile_id,
            "hardware_mapping": {},
            "sensor_timing": {},
            "simulation_physics": {"main_drive": {"damping": damping}},
        }
    )


def _response_trace(
    directory: Path,
    *,
    scenario_id: str,
    source: str,
    speed_scale: float,
    profile: CalibrationProfileV1 | None = None,
) -> None:
    scenario = load_scenario(scenario_id)
    scenario_hash = sha256_json(scenario.to_dict())
    dt = 0.05
    cycle_s = sum(float(segment["duration_s"]) for segment in scenario.command_segments)
    duration_s = cycle_s * scenario.repeats
    time_s = np.arange(0.0, duration_s + dt * 0.5, dt)
    command = np.zeros_like(time_s)
    velocity = np.zeros_like(time_s)
    cumulative = np.cumsum([float(item["duration_s"]) for item in scenario.command_segments])
    for index, sample_time in enumerate(time_s):
        local = min(sample_time, duration_s - np.finfo(float).eps) % cycle_s
        segment_index = min(int(np.searchsorted(cumulative, local, side="right")), len(cumulative) - 1)
        command[index] = float(scenario.command_segments[segment_index]["value"])
        # The reviewed step/coast sequence uses fixed boundaries within each 5.5 s cycle.
        if 0.6 <= local < 1.5:
            velocity[index] = 2.0 * speed_scale
        elif 1.5 <= local < 1.9:
            velocity[index] = 2.0 * speed_scale * (1.9 - local) / 0.4
        elif 3.1 <= local < 4.0:
            velocity[index] = -1.5 * speed_scale
        elif 4.0 <= local < 4.4:
            velocity[index] = -1.5 * speed_scale * (4.4 - local) / 0.4
    position = np.cumsum(velocity) * dt
    metadata = {
        "units": {"command": "rad/s", "position": "rad"},
        "frames": {"command": scenario.joint, "position": scenario.joint},
        "joint_order": [scenario.joint],
        "clock": {
            "source": "test",
            "timestamp_semantics": "relative_monotonic",
            "time_unit": "s",
        },
        "git_sha": None,
        "asset_sha256": None,
        "config_sha256": None,
        "calibration_constants": {
            "position_mapping_source": (
                f"profile:{profile.profile_id}" if source == "real" and profile else "synthetic"
            ),
            "requested_command_source": (
                f"authenticated_probe_events:{scenario_hash}"
                if source == "real"
                else "synthetic"
            ),
            "probe_event_evidence": {
                "scenario_sha256": scenario_hash,
                "repetition_count": 3,
                "segment_count": 21,
                "complete_ticks": 990,
                "abad_output_disabled_verified": True,
                "receive_duration_s": duration_s,
                "receive_jitter_bound_s": 1.0 / 60.0,
            }
        },
    }
    source_path = None
    if source == "real":
        source_path = directory.parent / f"{directory.name}.raw"
        source_path.write_bytes(f"raw:{directory.name}".encode())
    write_trace(
        directory,
        {
            "command_time_s": time_s,
            "command": command,
            "position_time_s": time_s,
            "position": position,
        },
        scenario=scenario,
        source=source,
        source_path=source_path,
        profile=profile,
        metadata=metadata,
    )


def _write_json(path: Path, payload: dict) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return sha256_file(path)


def _fixture(root: Path, *, sim_scale: float = 1.02) -> tuple[CalibrationProfileV1, dict]:
    baseline = _profile("baseline", 0.1)
    candidate = _profile("candidate", 0.2)
    baseline_path = root / "baseline.json"
    _write_json(baseline_path, baseline.to_dict())
    calibration = root / "real-calibration"
    holdout = root / "real-holdout"
    simulated = root / "sim-holdout"
    _response_trace(
        calibration,
        scenario_id="suspended-main-0-step-coast",
        source="real",
        speed_scale=0.9,
        profile=candidate,
    )
    _response_trace(
        holdout,
        scenario_id="suspended-main-5-step-coast",
        source="real",
        speed_scale=1.0,
        profile=candidate,
    )
    _response_trace(
        simulated,
        scenario_id="suspended-main-5-step-coast",
        source="sim",
        speed_scale=sim_scale,
        profile=candidate,
    )
    real_trace = load_trace(holdout)
    sim_trace = load_trace(simulated)
    comparison = compare_traces(
        real_trace, sim_trace, scenario="suspended-main-5-step-coast"
    )
    sweep_results = {
        "schema_version": 1,
        "sweep_sha256": "3" * 64,
        "scenario_id": "suspended-main-5-step-coast",
        "candidates": [
            {
                "status": "completed",
                "trace_sha256": sim_trace.manifest.provenance["trace_sha256"],
                "comparison": comparison,
            }
        ],
    }
    sweep_path = root / "sweep-results.json"
    sweep_hash = _write_json(sweep_path, sweep_results)
    audit_path = root / "audit.json"
    audit_hash = _write_json(
        audit_path,
        {
            "schema_version": 1,
            "checks": {
                "units_pass": True,
                "frames_pass": True,
                "joint_sign_pass": True,
                "mass_pass": True,
                "imu_mount_pass": True,
                "contact_sensor_pass": True,
            },
        },
    )
    metric_path = "step.positive.steady_speed_rad_s"
    evidence = {
        "schema_version": 1,
        "candidate_profile_sha256": sha256_json(candidate.to_dict()),
        "baseline_profile": {
            "path": "baseline.json",
            "sha256": sha256_file(baseline_path),
        },
        "audit_artifact": {"path": "audit.json", "sha256": audit_hash},
        "conditions": [
            {
                "condition_id": "main-cal",
                "subsystem": "main_drive",
                "role": "calibration",
                "real_episodes": [
                    {
                        "episode_id": "main-cal-real",
                        "path": "real-calibration",
                        "trace_sha256": load_trace(calibration).manifest.provenance[
                            "trace_sha256"
                        ],
                    }
                ],
                "metrics": {},
            },
            {
                "condition_id": "main-held",
                "subsystem": "main_drive",
                "role": "holdout",
                "held_out_by": ["leg"],
                "real_episodes": [
                    {
                        "episode_id": "main-held-real",
                        "path": "real-holdout",
                        "trace_sha256": real_trace.manifest.provenance["trace_sha256"],
                    }
                ],
                "sim_artifact": {
                    "path": "sim-holdout",
                    "trace_sha256": sim_trace.manifest.provenance["trace_sha256"],
                },
                "metrics": {
                    metric_path: {
                        "unit": "rad/s",
                        "instrument_uncertainty": 0.15,
                    }
                },
            },
        ],
        "actuator_sweeps": {
            "main_drive": {
                "results_path": "sweep-results.json",
                "results_sha256": sweep_hash,
                "candidate_artifacts": [
                    {
                        "path": "sim-holdout",
                        "trace_sha256": sim_trace.manifest.provenance["trace_sha256"],
                    }
                ],
            }
        },
    }
    return candidate, evidence


def test_promotion_resolves_artifacts_and_derives_repetitions_metrics_and_fitted_subsystems(
    tmp_path: Path,
) -> None:
    profile, evidence = _fixture(tmp_path)

    result = evaluate_promotion(profile, evidence, artifact_root=tmp_path)

    assert result["eligible_for_review"] is True
    assert result["promotion_requires_reviewed_config_change"] is True
    assert set(result["subsystems"]) == {"main_drive"}
    assert result["subsystems"]["main_drive"]["pass"] is True
    condition = result["subsystems"]["main_drive"]["holdout_conditions"][0]
    assert condition["real_repetition_count"] == 3
    metric = condition["metrics"]["step.positive.steady_speed_rad_s"]
    assert metric["absolute_error"] <= metric["tolerance"]
    assert "score" not in str(result).lower()
    assert result["evidence_sha256"] == sha256_json(evidence)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda evidence: evidence["conditions"][0]["real_episodes"][0].__setitem__(
                "path", "missing"
            ),
            "missing|does not exist",
        ),
        (
            lambda evidence: evidence["conditions"][1]["sim_artifact"].__setitem__(
                "trace_sha256", "0" * 64
            ),
            "trace hash mismatch",
        ),
        (
            lambda evidence: evidence["conditions"][1]["real_episodes"][0].update(
                {
                    "path": evidence["conditions"][0]["real_episodes"][0]["path"],
                    "trace_sha256": evidence["conditions"][0]["real_episodes"][0][
                        "trace_sha256"
                    ],
                }
            ),
            "calibration and holdout.*disjoint",
        ),
        (
            lambda evidence: evidence["conditions"][1].__setitem__(
                "held_out_by", ["direction"]
            ),
            "held-out dimension direction",
        ),
    ],
    ids=("missing-artifact", "wrong-hash", "reused-trace", "fake-coordinate"),
)
def test_promotion_rejects_unbound_or_fake_artifact_claims(
    tmp_path: Path, mutation, message: str
) -> None:
    profile, evidence = _fixture(tmp_path)
    mutation(evidence)

    with pytest.raises(ContractError, match=message):
        evaluate_promotion(profile, evidence, artifact_root=tmp_path)


def test_audit_metric_and_model_envelope_fail_the_affected_subsystem(
    tmp_path: Path,
) -> None:
    profile, evidence = _fixture(tmp_path, sim_scale=1.3)
    audit_path = tmp_path / "audit.json"
    audit = json.loads(audit_path.read_text())
    audit["checks"]["mass_pass"] = False
    evidence["audit_artifact"]["sha256"] = _write_json(audit_path, audit)

    result = evaluate_promotion(profile, evidence, artifact_root=tmp_path)

    assert result["eligible_for_review"] is False
    main = result["subsystems"]["main_drive"]
    assert main["pass"] is False
    assert main["actuator_model_mismatch"] is True
    assert any("audit.mass_pass" in reason for reason in result["failures"])
    assert any("outside" in reason for reason in main["failures"])


def test_missing_holdout_fails_the_derived_fitted_subsystem(tmp_path: Path) -> None:
    profile, evidence = _fixture(tmp_path)
    evidence["conditions"] = [evidence["conditions"][0]]

    result = evaluate_promotion(profile, evidence, artifact_root=tmp_path)

    assert result["eligible_for_review"] is False
    assert result["subsystems"]["main_drive"]["pass"] is False
    assert any("missing a holdout" in reason for reason in result["failures"])


def test_validate_promotion_cli_resolves_paths_relative_to_evidence(
    tmp_path: Path, capsys
) -> None:
    profile, evidence = _fixture(tmp_path)
    profile_path = tmp_path / "candidate.json"
    evidence_path = tmp_path / "evidence.json"
    output_path = tmp_path / "report.json"
    _write_json(profile_path, profile.to_dict())
    _write_json(evidence_path, evidence)

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
    assert code == 0
    assert emitted == json.loads(output_path.read_text())
    assert emitted["eligible_for_review"] is True
