from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from tools.sim2real.contracts import (
    CalibrationProfileV1,
    ContractError,
    ScenarioSpecV1,
    load_profile,
)
from tools.sim2real.scenarios import load_scenario
from tools.sim2real.sweep import generate_one_factor_candidates
from tools.sim2real.traces import (
    sha256_file,
    sha256_json,
    sha256_path,
    write_trace,
)


def _profile() -> CalibrationProfileV1:
    main_joints = [f"main_{index}" for index in range(6)]
    return CalibrationProfileV1.from_dict(
        {
            "schema_version": 1,
            "profile_id": "baseline",
            "hardware_mapping": {
                "encoder_counts_per_rev": {
                    joint: 54984.83 for joint in main_joints
                },
                "encoder_zero_count": {joint: 0.0 for joint in main_joints},
                "encoder_sign": {joint: 1 for joint in main_joints},
                "joint_direction": {"main_0": 1},
                "pwm_scale": {"main_0": 1.0 / 120.0},
                "pwm_cap": {"main_0": 500.0 / 120.0},
            },
            "sensor_timing": {"aggregate_command_delay_s": 0.01},
            "simulation_physics": {
                "main_drive": {"damping": 0.2},
                "ground": {"static_friction": 0.8, "dynamic_friction": 0.7},
            },
        }
    )


def _candidates(count: int = 2) -> list[CalibrationProfileV1]:
    return generate_one_factor_candidates(
        _profile(),
        {"simulation_physics.main_drive.damping": [0.3, 0.4][:count]},
    )


def _effort_profiles() -> tuple[CalibrationProfileV1, CalibrationProfileV1]:
    payload = _profile().to_dict()
    payload["simulation_physics"]["main_drive"]["effort_limit"] = 1.0
    baseline = CalibrationProfileV1.from_dict(payload)
    candidate = generate_one_factor_candidates(
        baseline,
        {"simulation_physics.main_drive.effort_limit": [2.0]},
    )[0]
    return baseline, candidate


def _argument(command: list[str], flag: str) -> str:
    return command[command.index(flag) + 1]


def _response_arrays(*, scale: float = 1.0) -> dict[str, np.ndarray]:
    time_s = np.arange(0.0, 10.5 + 0.05, 0.05)
    phase = np.mod(time_s, 3.5)
    command = np.select(
        [
            (phase >= 0.5) & (phase < 1.5),
            (phase >= 2.0) & (phase < 3.0),
        ],
        [0.25, -0.25],
        default=0.0,
    )
    position = np.cumsum(command * scale) * 0.05
    return {
        "command_time_s": time_s,
        "position_time_s": time_s,
        "command": command,
        "position": position,
    }


def _trace_metadata(scenario: ScenarioSpecV1, **overrides: Any) -> dict[str, Any]:
    metadata = {
        "units": {"command": "rad/s", "position": "rad"},
        "frames": {"command": scenario.joint, "position": scenario.joint},
        "calibration_constants": {
            "calibration_source": "profile:baseline",
            "position_mapping_source": "profile:baseline",
            "requested_command_source": "profile:baseline",
        },
        "git_sha": "1" * 40,
        "asset_sha256": "a" * 64,
        "config_sha256": "b" * 64,
        "characterization_runner_sha256": "c" * 64,
        "runtime_bundle_sha256": "e" * 64,
    }
    metadata.update(overrides)
    return metadata


def _runtime_provenance(**overrides: Any) -> dict[str, str]:
    provenance = {
        "git_sha": "1" * 40,
        "asset_sha256": "a" * 64,
        "config_sha256": "b" * 64,
        "characterization_runner_sha256": "c" * 64,
        "sweep_runner_sha256": "d" * 64,
        "runtime_bundle_sha256": "e" * 64,
    }
    provenance.update(overrides)
    return provenance


_AUDIT_FIXTURES: dict[tuple[str, str], tuple[dict[str, Any], Path]] = {}


def _passing_audit(
    root: Path, profile: CalibrationProfileV1
) -> tuple[dict[str, Any], Path]:
    from tools.sim2real.tests.test_promotion import _audit_evidence

    profile_sha = sha256_json(profile.to_dict())
    key = (str(root.resolve()), profile_sha)
    cached = _AUDIT_FIXTURES.get(key)
    if cached is not None:
        return cached
    artifact_root = root / f"audit-{profile_sha[:12]}"
    binding = _audit_evidence(artifact_root, profile)
    result = (binding, artifact_root)
    _AUDIT_FIXTURES[key] = result
    return result


def _write_real_reference(output: Path, scenario: ScenarioSpecV1) -> None:
    dataset = output.parent.parent
    raw = dataset / "raw" / f"{output.name}.bin"
    raw.parent.mkdir(parents=True, exist_ok=True)
    raw.write_bytes(b"immutable managed real reference")
    manifest = write_trace(
        output,
        _response_arrays(scale=1.0),
        scenario=scenario,
        source="real",
        source_path=raw,
        profile=_profile(),
        metadata={
            **_trace_metadata(scenario),
        },
    )
    metadata_sha256 = sha256_file(output / "metadata.json")
    dataset_manifest = {
        "schema_version": 1,
        "dataset_id": dataset.name,
        "raw": [
            {
                "path": raw.relative_to(dataset).as_posix(),
                "sha256": sha256_path(raw),
            }
        ],
        "episodes": [
            {
                "episode_id": output.name,
                "scenario_id": scenario.scenario_id,
                "path": output.relative_to(dataset).as_posix(),
                "trace_sha256": manifest.provenance["trace_sha256"],
                "metadata_sha256": metadata_sha256,
                "raw_path": raw.relative_to(dataset).as_posix(),
            }
        ],
    }
    (dataset / "manifest.json").write_text(
        json.dumps(dataset_manifest) + "\n", encoding="utf-8"
    )


def _write_standalone_real_reference(output: Path, scenario: ScenarioSpecV1) -> None:
    raw = output.parent / f"{output.name}.bin"
    raw.write_bytes(b"standalone real reference")
    write_trace(
        output,
        _response_arrays(scale=1.0),
        scenario=scenario,
        source="real",
        source_path=raw,
        profile=_profile(),
        metadata=_trace_metadata(scenario),
    )


def _write_known_load_reference(output: Path, *, torque_scale: float = 2.0) -> None:
    scenario = load_scenario("manual-load")
    dataset = output.parent.parent
    raw = dataset / "raw" / f"{output.name}.bin"
    raw.parent.mkdir(parents=True, exist_ok=True)
    raw.write_bytes(b"immutable known-load measurement")
    time_s = np.arange(6, dtype=float)
    manifest = write_trace(
        output,
        {
            "load_force_time_s": time_s,
            "load_force": np.array([10.0, 10.0, 10.5, 10.5, 9.5, 9.5])
            * torque_scale,
            "lever_arm_time_s": time_s,
            "lever_arm": np.full(6, 0.1),
            "command_time_s": time_s,
            "command": np.array([0.2, -0.2] * 3),
            "direction_time_s": time_s,
            "direction": np.array([1.0, -1.0] * 3),
            "saturation_confirmed": np.ones(6),
            "repeat_index": np.repeat(np.arange(3), 2),
        },
        scenario=scenario,
        source="real",
        source_path=raw,
        profile=_profile(),
        metadata={
            "units": {
                "load_force": "N",
                "lever_arm": "m",
                "command": "normalized",
                "direction": "1",
                "saturation_confirmed": "1",
                "repeat_index": "1",
            },
            "frames": {
                "load_force": "main_0",
                "lever_arm": "main_0",
                "command": "main_0",
                "direction": "main_0",
                "saturation_confirmed": "main_0",
                "repeat_index": "main_0",
            },
            # Manual load evidence has no encoder-position channel and therefore
            # must not pretend to carry encoder-mapping provenance.
            "calibration_constants": {},
        },
    )
    dataset_manifest = {
        "schema_version": 1,
        "dataset_id": dataset.name,
        "raw": [
            {
                "path": raw.relative_to(dataset).as_posix(),
                "sha256": sha256_path(raw),
            }
        ],
        "episodes": [
            {
                "episode_id": output.name,
                "scenario_id": scenario.scenario_id,
                "path": output.relative_to(dataset).as_posix(),
                "trace_sha256": manifest.provenance["trace_sha256"],
                "metadata_sha256": sha256_file(output / "metadata.json"),
                "raw_path": raw.relative_to(dataset).as_posix(),
            }
        ],
    }
    (dataset / "manifest.json").write_text(
        json.dumps(dataset_manifest) + "\n", encoding="utf-8"
    )


def _write_sim_artifact(
    command: list[str], *, profile_override=None, metadata_override=None
) -> None:
    scenario = load_scenario(_argument(command, "--scenario"))
    profile = profile_override or load_profile(_argument(command, "--physics-profile"))
    output = Path(_argument(command, "--output"))
    manifest = write_trace(
        output,
        _response_arrays(scale=0.8),
        scenario=scenario,
        source="sim",
        profile=profile,
        metadata=metadata_override or _trace_metadata(scenario),
    )
    (output / "runtime_audit.json").write_text("{}\n", encoding="utf-8")
    (output / "results.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "scenario_id": scenario.scenario_id,
                "mode": _argument(command, "--mode"),
                "steps": 2,
                "physics_dt_s": 1.0 / 120.0,
                "trace_sha256": manifest.provenance["trace_sha256"],
                "profile_id": profile.profile_id,
                "contact_validation": None,
                "runtime_audit": "runtime_audit.json",
            }
        )
        + "\n",
        encoding="utf-8",
    )


def _execute(
    output: Path,
    *,
    candidates: list[CalibrationProfileV1] | None = None,
    scenario: ScenarioSpecV1 | None = None,
    run_process,
    generate_only: bool = False,
    real_trace: str | Path | None | object = ...,
    audit_artifact: dict[str, Any] | None | object = ...,
    audit_artifact_root: str | Path | None = None,
    provenance: dict[str, Any] | None = None,
    provenance_provider=lambda: _runtime_provenance(),
) -> dict[str, Any]:
    from tools.sim2real.sweep_runner import execute_sweep

    selected_scenario = scenario or load_scenario("main-step")
    if audit_artifact is ...:
        if generate_only:
            audit_artifact = None
        else:
            audit_artifact, audit_artifact_root = _passing_audit(
                output.parent, _profile()
            )
    if real_trace is ...:
        if generate_only:
            real_trace = None
        else:
            real_trace = (
                output.parent
                / "datasets"
                / "sim2real"
                / f"{output.name}-dataset"
                / "episodes"
                / f"{output.name}-real"
            )
            if not real_trace.exists():
                _write_real_reference(real_trace, selected_scenario)
    return execute_sweep(
        output=output,
        scenario=selected_scenario,
        base_profile=_profile(),
        candidates=candidates or _candidates(),
        sweep_mode="one-factor",
        scene_mode="fixed-base",
        headless=True,
        seed=17,
        device="cpu",
        provenance=provenance or {},
        provenance_provider=provenance_provider,
        command_prefix=("/opt/isaaclab/isaaclab.sh", "-p", "-m", "tools.sim2real"),
        generate_only=generate_only,
        real_trace=real_trace,
        known_load_trace=None,
        audit_artifact=audit_artifact,
        audit_artifact_root=audit_artifact_root,
        run_process=run_process,
    )


def test_execute_sweep_runs_fresh_process_per_candidate_and_verifies_outputs(
    tmp_path: Path,
) -> None:
    commands: list[list[str]] = []

    def run_process(command, **kwargs):
        command = list(command)
        commands.append(command)
        assert kwargs == {"check": False, "capture_output": True, "text": True}
        _write_sim_artifact(command)
        return subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")

    result = _execute(tmp_path / "sweep", run_process=run_process)

    assert len(commands) == 2
    assert commands[0][:5] == [
        "/opt/isaaclab/isaaclab.sh",
        "-p",
        "-m",
        "tools.sim2real",
        "run-sim",
    ]
    assert load_scenario(_argument(commands[0], "--scenario")).scenario_id == "main-step"
    assert _argument(commands[0], "--mode") == "fixed-base"
    assert _argument(commands[0], "--seed") == "17"
    assert _argument(commands[0], "--device") == "cpu"
    assert "--headless" in commands[0]
    assert _argument(commands[0], "--physics-profile") != _argument(
        commands[1], "--physics-profile"
    )
    assert _argument(commands[0], "--output") != _argument(commands[1], "--output")
    assert result["counts"] == {
        "cached": 0,
        "completed": 2,
        "failed": 0,
        "generated": 0,
        "pending": 0,
    }
    assert [item["status"] for item in result["candidates"]] == [
        "completed",
        "completed",
    ]
    assert json.loads((tmp_path / "sweep" / "index.json").read_text())[
        "sweep_sha256"
    ] == result["sweep_sha256"]
    provenance = json.loads((tmp_path / "sweep" / "provenance.json").read_text())
    assert provenance["git_sha"] == "1" * 40
    assert provenance["seed"] == 17
    assert len(provenance["audit_artifact_sha256"]) == 64
    assert len(provenance["audit_report_sha256"]) == 64
    assert json.loads((tmp_path / "sweep" / "results.json").read_text()) == result
    for candidate in result["candidates"]:
        assert set(candidate["metrics"]) == {"main_drive"}
        assert set(candidate["metrics"]["main_drive"]) == {"real", "sim", "delta"}
        assert "score" not in json.dumps(candidate["metrics"]).lower()
        comparison = tmp_path / "sweep" / candidate["comparison"]
        payload = json.loads(comparison.read_text(encoding="utf-8"))
        assert payload["real_trace_sha256"] == provenance["real_trace_sha256"]
        assert payload["sim_trace_sha256"] == candidate["trace_sha256"]
        assert candidate["comparison_sha256"] == sha256_json(payload)
        run = tmp_path / "sweep" / candidate["run_output"]
        assert candidate["metadata_sha256"] == sha256_file(run / "metadata.json")
        status = json.loads(
            (tmp_path / "sweep" / candidate["status_file"]).read_text()
        )
        assert status["metadata_sha256"] == candidate["metadata_sha256"]


def test_cached_candidate_rejects_metadata_changed_after_completion(
    tmp_path: Path,
) -> None:
    def first_run(command, **_kwargs):
        command = list(command)
        _write_sim_artifact(command)
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    output = tmp_path / "sweep"
    result = _execute(output, candidates=_candidates(1), run_process=first_run)
    metadata_path = output / result["candidates"][0]["run_output"] / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["metadata"]["operator_note"] = "tampered after completion"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ContractError, match="metadata.*hash mismatch"):
        _execute(
            output,
            candidates=_candidates(1),
            run_process=lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("tampered cache must fail before running")
            ),
        )


def test_execution_requires_real_reference_before_writing_output(tmp_path: Path) -> None:
    def must_not_run(*_args, **_kwargs):
        raise AssertionError("missing real evidence must fail before launching Isaac")

    output = tmp_path / "sweep"
    with pytest.raises(ContractError, match="real_trace is required"):
        _execute(
            output,
            candidates=_candidates(1),
            run_process=must_not_run,
            real_trace=None,
        )

    assert not output.exists()


def test_execution_requires_passing_prefit_audit_before_writing_output(
    tmp_path: Path,
) -> None:
    output = tmp_path / "sweep"

    with pytest.raises(ContractError, match="passing pre-fit audit"):
        _execute(
            output,
            candidates=_candidates(1),
            audit_artifact=None,
            run_process=lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("missing audit must fail before running")
            ),
        )

    assert not output.exists()


def test_execution_rejects_failed_prefit_audit_and_binds_pass_to_cache(
    tmp_path: Path,
) -> None:
    audit, audit_root = _passing_audit(tmp_path, _profile())
    physical_binding = audit["physical_measurements"]
    physical_path = audit_root / physical_binding["path"]
    physical = json.loads(physical_path.read_text())
    physical["mass_measurements_kg"] = [20.0, 20.0, 20.0]
    physical_path.write_text(
        json.dumps(physical, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    physical_binding["sha256"] = sha256_file(physical_path)

    output = tmp_path / "sweep"
    with pytest.raises(ContractError, match="pre-fit audit failed.*mass_pass"):
        _execute(
            output,
            candidates=_candidates(1),
            audit_artifact=audit,
            audit_artifact_root=audit_root,
            run_process=lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("failed audit must fail before running")
            ),
        )

    assert not output.exists()


def test_execution_rejects_standalone_real_reference_before_writing_output(
    tmp_path: Path,
) -> None:
    scenario = load_scenario("main-step")
    standalone = tmp_path / "standalone-real"
    _write_standalone_real_reference(standalone, scenario)
    output = tmp_path / "sweep"

    with pytest.raises(ContractError, match="managed dataset"):
        _execute(
            output,
            candidates=_candidates(1),
            scenario=scenario,
            real_trace=standalone,
            run_process=lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("standalone reference must fail before running")
            ),
        )

    assert not output.exists()


def test_effort_limit_sweep_requires_managed_known_load_evidence(
    tmp_path: Path,
) -> None:
    from tools.sim2real.sweep_runner import execute_sweep

    baseline, candidate = _effort_profiles()
    scenario = load_scenario("main-step")
    real = (
        tmp_path
        / "datasets"
        / "sim2real"
        / "main-reference-dataset"
        / "episodes"
        / "main-reference-real"
    )
    _write_real_reference(real, scenario)
    audit_artifact, audit_root = _passing_audit(tmp_path, baseline)

    with pytest.raises(ContractError, match="known-load.*effort_limit"):
        execute_sweep(
            output=tmp_path / "sweep",
            scenario=scenario,
            base_profile=baseline,
            candidates=[candidate],
            sweep_mode="one-factor",
            scene_mode="fixed-base",
            headless=True,
            seed=17,
            device="cpu",
            provenance={},
            provenance_provider=lambda: _runtime_provenance(),
            command_prefix=("/opt/isaaclab/isaaclab.sh", "-p", "-m", "tools.sim2real"),
            real_trace=real,
            known_load_trace=None,
            audit_artifact=audit_artifact,
            audit_artifact_root=audit_root,
            run_process=lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("missing known load must fail before running")
            ),
        )


def test_effort_limit_sweep_binds_managed_known_load_evidence(
    tmp_path: Path,
) -> None:
    from tools.sim2real.sweep_runner import execute_sweep

    baseline, candidate = _effort_profiles()
    scenario = load_scenario("main-step")
    real = (
        tmp_path
        / "datasets"
        / "sim2real"
        / "main-reference-dataset"
        / "episodes"
        / "main-reference-real"
    )
    known_load = (
        tmp_path
        / "datasets"
        / "sim2real"
        / "known-load-dataset"
        / "episodes"
        / "known-load-real"
    )
    _write_real_reference(real, scenario)
    _write_known_load_reference(known_load)
    audit_artifact, audit_root = _passing_audit(tmp_path, baseline)

    def run_process(command, **_kwargs):
        command = list(command)
        _write_sim_artifact(command)
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    result = execute_sweep(
        output=tmp_path / "sweep",
        scenario=scenario,
        base_profile=baseline,
        candidates=[candidate],
        sweep_mode="one-factor",
        scene_mode="fixed-base",
        headless=True,
        seed=17,
        device="cpu",
        provenance={},
        provenance_provider=lambda: _runtime_provenance(),
        command_prefix=("/opt/isaaclab/isaaclab.sh", "-p", "-m", "tools.sim2real"),
        real_trace=real,
        known_load_trace=known_load,
        audit_artifact=audit_artifact,
        audit_artifact_root=audit_root,
        run_process=run_process,
    )

    provenance = json.loads((tmp_path / "sweep" / "provenance.json").read_text())
    assert result["counts"]["completed"] == 1
    assert len(provenance["known_load_trace_sha256"]) == 64
    assert len(provenance["known_load_metadata_sha256"]) == 64


def test_effort_limit_sweep_must_cover_measured_known_load_envelope(
    tmp_path: Path,
) -> None:
    from tools.sim2real.sweep_runner import execute_sweep

    baseline, candidate = _effort_profiles()
    payload = candidate.to_dict()
    payload["simulation_physics"]["main_drive"]["effort_limit"] = 4.0
    candidate = CalibrationProfileV1.from_dict(payload)
    scenario = load_scenario("main-step")
    real = tmp_path / "datasets" / "sim2real" / "main" / "episodes" / "response"
    known_load = tmp_path / "datasets" / "sim2real" / "load" / "episodes" / "known"
    _write_real_reference(real, scenario)
    _write_known_load_reference(known_load)
    audit_artifact, audit_root = _passing_audit(tmp_path, baseline)

    with pytest.raises(ContractError, match="effort-limit candidates.*known-load envelope"):
        execute_sweep(
            output=tmp_path / "sweep",
            scenario=scenario,
            base_profile=baseline,
            candidates=[candidate],
            sweep_mode="one-factor",
            scene_mode="fixed-base",
            headless=True,
            seed=17,
            device="cpu",
            provenance={},
            provenance_provider=lambda: _runtime_provenance(),
            command_prefix=("/opt/isaaclab/isaaclab.sh", "-p", "-m", "tools.sim2real"),
            real_trace=real,
            known_load_trace=known_load,
            audit_artifact=audit_artifact,
            audit_artifact_root=audit_root,
            run_process=lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("mismatched known-load evidence must fail before running")
            ),
        )


def test_cached_comparison_is_recomputed_against_bound_real_trace(
    tmp_path: Path,
) -> None:
    def first_run(command, **_kwargs):
        command = list(command)
        _write_sim_artifact(command)
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    output = tmp_path / "sweep"
    first = _execute(output, candidates=_candidates(1), run_process=first_run)
    comparison_path = output / first["candidates"][0]["comparison"]
    comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
    comparison["subsystems"]["main_drive"]["delta"] = {"tampered": 999.0}
    comparison_path.write_text(json.dumps(comparison), encoding="utf-8")

    def must_not_run(*_args, **_kwargs):
        raise AssertionError("a corrupt comparison must fail before launching Isaac")

    with pytest.raises(ContractError, match="cached comparison mismatch"):
        _execute(output, candidates=_candidates(1), run_process=must_not_run)

    status = json.loads((output / "statuses" / "0001.json").read_text())
    assert status["status"] == "failed"


def test_resume_reuses_only_verified_completed_cache(tmp_path: Path) -> None:
    commands: list[list[str]] = []

    def first_run(command, **_kwargs):
        command = list(command)
        commands.append(command)
        _write_sim_artifact(command)
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    output = tmp_path / "sweep"
    _execute(output, candidates=_candidates(1), run_process=first_run)

    def must_not_run(*_args, **_kwargs):
        raise AssertionError("a verified cache entry must not launch Isaac again")

    result = _execute(output, candidates=_candidates(1), run_process=must_not_run)

    assert len(commands) == 1
    assert result["counts"]["cached"] == 1
    assert result["candidates"][0]["status"] == "cached"
    assert result["candidates"][0]["run_output"].endswith("attempt-0001")


def test_resume_runs_remaining_candidate_in_a_new_attempt(tmp_path: Path) -> None:
    call_count = 0

    def interrupted_run(command, **_kwargs):
        nonlocal call_count
        command = list(command)
        call_count += 1
        if call_count == 1:
            _write_sim_artifact(command)
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")
        return subprocess.CompletedProcess(command, 9, stdout="", stderr="sim failed")

    output = tmp_path / "sweep"
    with pytest.raises(ContractError, match="candidate 2.*exit code 9"):
        _execute(output, run_process=interrupted_run)

    resumed_commands: list[list[str]] = []

    def resumed_run(command, **_kwargs):
        command = list(command)
        resumed_commands.append(command)
        _write_sim_artifact(command)
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    result = _execute(output, run_process=resumed_run)

    assert len(resumed_commands) == 1
    assert _argument(resumed_commands[0], "--output").endswith("attempt-0002")
    assert [item["status"] for item in result["candidates"]] == [
        "cached",
        "completed",
    ]


def test_invalid_interrupted_artifact_records_failure_then_resumes_new_attempt(
    tmp_path: Path,
) -> None:
    def first_run(command, **_kwargs):
        command = list(command)
        _write_sim_artifact(command)
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    output = tmp_path / "sweep"
    first = _execute(output, candidates=_candidates(1), run_process=first_run)
    status_path = output / "statuses" / "0001.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    status["status"] = "running"
    status_path.write_text(json.dumps(status), encoding="utf-8")
    first_output = output / first["candidates"][0]["run_output"]
    (first_output / "trace.npz").write_bytes(b"interrupted")

    def must_not_run(*_args, **_kwargs):
        raise AssertionError("invalid interrupted output must be recorded before retry")

    with pytest.raises(ContractError, match="interrupted artifact verification failed"):
        _execute(output, candidates=_candidates(1), run_process=must_not_run)

    failed = json.loads(status_path.read_text(encoding="utf-8"))
    assert failed["status"] == "failed"
    assert failed["attempt"] == 1
    assert "trace hash mismatch" in failed["error"]

    resumed_commands: list[list[str]] = []

    def resumed_run(command, **_kwargs):
        command = list(command)
        resumed_commands.append(command)
        _write_sim_artifact(command)
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    resumed = _execute(
        output,
        candidates=_candidates(1),
        run_process=resumed_run,
    )

    assert len(resumed_commands) == 1
    assert _argument(resumed_commands[0], "--output").endswith("attempt-0002")
    assert first_output.is_dir()
    assert resumed["candidates"][0]["status"] == "completed"
    assert resumed["candidates"][0]["attempt"] == 2


def test_zero_exit_without_result_fails_closed_and_persists_status(tmp_path: Path) -> None:
    def no_result(command, **_kwargs):
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    output = tmp_path / "sweep"
    with pytest.raises(ContractError, match="candidate 1.*results.json"):
        _execute(output, candidates=_candidates(1), run_process=no_result)

    status = json.loads((output / "statuses" / "0001.json").read_text())
    results = json.loads((output / "results.json").read_text())
    assert status["status"] == "failed"
    assert results["counts"]["failed"] == 1


def test_cached_trace_hash_mismatch_fails_closed_without_subprocess(
    tmp_path: Path,
) -> None:
    output = tmp_path / "sweep"

    def first_run(command, **_kwargs):
        command = list(command)
        _write_sim_artifact(command)
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    first = _execute(output, candidates=_candidates(1), run_process=first_run)
    run_output = output / first["candidates"][0]["run_output"]
    (run_output / "trace.npz").write_bytes(b"tampered")

    def must_not_run(*_args, **_kwargs):
        raise AssertionError("a corrupt completed cache must fail closed")

    with pytest.raises(ContractError, match="candidate 1.*trace hash mismatch"):
        _execute(output, candidates=_candidates(1), run_process=must_not_run)

    status = json.loads((output / "statuses" / "0001.json").read_text())
    assert status["status"] == "failed"


def test_artifact_profile_hash_mismatch_fails_closed(tmp_path: Path) -> None:
    def wrong_profile(command, **_kwargs):
        command = list(command)
        _write_sim_artifact(command, profile_override=_profile())
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    with pytest.raises(ContractError, match="candidate 1.*profile hash mismatch"):
        _execute(
            tmp_path / "sweep",
            candidates=_candidates(1),
            run_process=wrong_profile,
        )


def test_artifact_provenance_mismatch_fails_closed(tmp_path: Path) -> None:
    def wrong_provenance(command, **_kwargs):
        command = list(command)
        _write_sim_artifact(
            command,
            metadata_override=_trace_metadata(
                load_scenario(_argument(command, "--scenario")),
                git_sha="different",
            ),
        )
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    with pytest.raises(ContractError, match="candidate 1.*git_sha mismatch"):
        _execute(
            tmp_path / "sweep",
            candidates=_candidates(1),
            run_process=wrong_provenance,
        )


def test_runtime_provenance_provider_hashes_production_inputs(tmp_path: Path) -> None:
    from tools.sim2real.runtime_provenance import (
        _BEHAVIOR_INPUTS,
        production_runtime_provenance,
    )
    from tools.sim2real.traces import sha256_file, sha256_json

    paths = {
        "asset_sha256": tmp_path / "RedRhex.usd",
        "config_sha256": (
            tmp_path
            / "source/RedRhex/RedRhex/tasks/direct/redrhex/redrhex_env_cfg.py"
        ),
        "characterization_runner_sha256": tmp_path / "tools/sim2real/isaac_runner.py",
        "sweep_runner_sha256": tmp_path / "tools/sim2real/sweep_runner.py",
    }
    for index, path in enumerate(paths.values(), start=1):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"input-{index}".encode())
    for index, relative in enumerate(_BEHAVIOR_INPUTS, start=1):
        path = tmp_path / relative
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(f"behavior-{index}".encode())

    def run_git(command, **kwargs):
        assert command == ["git", "rev-parse", "HEAD"]
        assert kwargs["cwd"] == tmp_path
        return subprocess.CompletedProcess(command, 0, stdout="e" * 40 + "\n", stderr="")

    result = production_runtime_provenance(tmp_path, run_git=run_git)

    assert result["git_sha"] == "e" * 40
    for field, path in paths.items():
        assert result[field] == sha256_file(path)
    assert result["runtime_bundle_sha256"] == sha256_json(
        {
            relative.as_posix(): sha256_file(tmp_path / relative)
            for relative in _BEHAVIOR_INPUTS
        }
    )


def test_runtime_provenance_is_required_and_cannot_be_user_overridden(
    tmp_path: Path,
) -> None:
    def must_not_run(*_args, **_kwargs):
        raise AssertionError("invalid provenance must fail before launching Isaac")

    missing_output = tmp_path / "missing"
    with pytest.raises(ContractError, match="runtime provenance missing.*config_sha256"):
        _execute(
            missing_output,
            candidates=_candidates(1),
            run_process=must_not_run,
            provenance_provider=lambda: {
                key: value
                for key, value in _runtime_provenance().items()
                if key != "config_sha256"
            },
        )
    assert not missing_output.exists()

    override_output = tmp_path / "override"
    with pytest.raises(ContractError, match="derived provenance fields cannot be overridden"):
        _execute(
            override_output,
            candidates=_candidates(1),
            run_process=must_not_run,
            provenance={"asset_sha256": "f" * 64},
        )
    assert not override_output.exists()


def test_runtime_asset_hash_changes_cache_identity(tmp_path: Path) -> None:
    def first_run(command, **_kwargs):
        command = list(command)
        _write_sim_artifact(command)
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    output = tmp_path / "sweep"
    _execute(output, candidates=_candidates(1), run_process=first_run)

    def must_not_run(*_args, **_kwargs):
        raise AssertionError("cache identity must reject before launching Isaac")

    with pytest.raises(ContractError, match="existing sweep index does not match"):
        _execute(
            output,
            candidates=_candidates(1),
            run_process=must_not_run,
            provenance_provider=lambda: _runtime_provenance(
                asset_sha256="f" * 64
            ),
        )


def test_prefit_audit_hash_changes_cache_identity(tmp_path: Path) -> None:
    def first_run(command, **_kwargs):
        command = list(command)
        _write_sim_artifact(command)
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    output = tmp_path / "sweep"
    audit, audit_root = _passing_audit(tmp_path, _profile())
    _execute(
        output,
        candidates=_candidates(1),
        audit_artifact=audit,
        audit_artifact_root=audit_root,
        run_process=first_run,
    )

    physical_binding = audit["physical_measurements"]
    physical_path = audit_root / physical_binding["path"]
    physical = json.loads(physical_path.read_text())
    physical["mass_instrument_uncertainty_kg"] = 0.3
    physical_path.write_text(
        json.dumps(physical, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    physical_binding["sha256"] = sha256_file(physical_path)

    with pytest.raises(ContractError, match="existing sweep index does not match"):
        _execute(
            output,
            candidates=_candidates(1),
            audit_artifact=audit,
            audit_artifact_root=audit_root,
            run_process=lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("changed audit must invalidate before running")
            ),
        )


def test_artifact_must_report_all_derived_runtime_provenance(tmp_path: Path) -> None:
    def incomplete_artifact(command, **_kwargs):
        command = list(command)
        scenario = load_scenario(_argument(command, "--scenario"))
        _write_sim_artifact(
            command,
            metadata_override=_trace_metadata(
                scenario,
                config_sha256=None,
                characterization_runner_sha256=None,
            ),
        )
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    with pytest.raises(ContractError, match="config_sha256 mismatch"):
        _execute(
            tmp_path / "sweep",
            candidates=_candidates(1),
            run_process=incomplete_artifact,
        )


def test_generate_only_writes_candidates_and_launches_nothing(tmp_path: Path) -> None:
    def must_not_run(*_args, **_kwargs):
        raise AssertionError("generate-only mode must not launch Isaac")

    output = tmp_path / "sweep"
    result = _execute(output, run_process=must_not_run, generate_only=True)

    assert result["counts"]["generated"] == 2
    assert [item["status"] for item in result["candidates"]] == [
        "generated",
        "generated",
    ]
    assert sorted(path.name for path in (output / "candidates").iterdir()) == [
        "0001.json",
        "0002.json",
    ]
    assert not (output / "runs").exists()


def test_execute_sweep_snapshots_custom_scenario_for_fresh_process(
    tmp_path: Path,
) -> None:
    payload = load_scenario("main-step").to_dict()
    payload["scenario_id"] = "custom-main-step"
    scenario = ScenarioSpecV1.from_dict(payload)
    commands: list[list[str]] = []

    def run_process(command, **_kwargs):
        command = list(command)
        commands.append(command)
        _write_sim_artifact(command)
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    output = tmp_path / "sweep"
    _execute(
        output,
        scenario=scenario,
        candidates=_candidates(1),
        run_process=run_process,
    )

    scenario_argument = Path(_argument(commands[0], "--scenario"))
    assert scenario_argument == (output / "scenario.json").resolve()
    assert load_scenario(scenario_argument).to_dict() == scenario.to_dict()


def test_sweep_cli_defaults_to_execution_and_forwards_runner_settings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tools.sim2real import sweep_runner
    from tools.sim2real.cli import _run, build_parser
    from tools.sim2real.runtime_provenance import production_runtime_provenance

    profile_path = tmp_path / "profile.json"
    profile_path.write_text(json.dumps(_profile().to_dict()), encoding="utf-8")
    audit_payload = {
        "runtime_trace": {"path": "audit-run", "trace_sha256": "a" * 64},
        "runtime_audit": {"path": "audit-run/runtime_audit.json", "sha256": "b" * 64},
        "physical_measurements": {"path": "physical.json", "sha256": "c" * 64},
    }
    audit_path = tmp_path / "audit-evidence.json"
    audit_path.write_text(json.dumps(audit_payload), encoding="utf-8")
    isaac_root = tmp_path / "IsaacLab"
    isaac_root.mkdir()
    (isaac_root / "isaaclab.sh").write_text("#!/bin/sh\n", encoding="utf-8")
    captured: dict[str, Any] = {}

    def fake_execute_sweep(**kwargs):
        captured.update(kwargs)
        return {"schema_version": 1, "executed": True}

    monkeypatch.setattr(sweep_runner, "execute_sweep", fake_execute_sweep)
    args = build_parser().parse_args(
        [
            "sweep",
            str(profile_path),
            "--scenario",
            "main-step",
            "--mode",
            "one-factor",
            "--space-json",
            '{"simulation_physics.main_drive.damping":[0.3]}',
            "--output",
            str(tmp_path / "output"),
            "--scene-mode",
            "fixed-base",
            "--isaaclab-root",
            str(isaac_root),
            "--seed",
            "23",
            "--device",
            "cpu",
            "--headless",
            "--real-trace",
            str(tmp_path / "real-reference"),
            "--audit-evidence",
            str(audit_path),
            "--provenance-json",
            '{"operator":"bench"}',
        ]
    )

    assert _run(args) == {"schema_version": 1, "executed": True}
    assert captured["generate_only"] is False
    assert captured["command_prefix"] == (
        str(isaac_root / "isaaclab.sh"),
        "-p",
        "-m",
        "tools.sim2real",
    )
    assert captured["scene_mode"] == "fixed-base"
    assert captured["seed"] == 23
    assert captured["device"] == "cpu"
    assert captured["headless"] is True
    assert captured["real_trace"] == tmp_path / "real-reference"
    assert captured["known_load_trace"] is None
    assert captured["audit_artifact"] == audit_payload
    assert captured["audit_artifact_root"] == tmp_path
    assert captured["base_profile"].to_dict() == _profile().to_dict()
    assert captured["provenance"]["operator"] == "bench"
    assert captured["provenance_provider"] is production_runtime_provenance


def test_sweep_cli_generate_only_needs_no_isaac_install(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tools.sim2real import sweep_runner
    from tools.sim2real.cli import _run, build_parser

    profile_path = tmp_path / "profile.json"
    profile_path.write_text(json.dumps(_profile().to_dict()), encoding="utf-8")
    captured: dict[str, Any] = {}

    def fake_execute_sweep(**kwargs):
        captured.update(kwargs)
        return {"schema_version": 1, "generated": True}

    monkeypatch.delenv("ISAACLAB_ROOT", raising=False)
    monkeypatch.setattr(sweep_runner, "execute_sweep", fake_execute_sweep)
    args = build_parser().parse_args(
        [
            "sweep",
            str(profile_path),
            "--scenario",
            "main-step",
            "--mode",
            "one-factor",
            "--space-json",
            '{"simulation_physics.main_drive.damping":[0.3]}',
            "--output",
            str(tmp_path / "output"),
            "--generate-only",
        ]
    )

    assert _run(args) == {"schema_version": 1, "generated": True}
    assert captured["generate_only"] is True
    assert captured["command_prefix"] is None
    assert captured["scene_mode"] == "fixed-base"
    assert captured["real_trace"] is None
    assert captured["audit_artifact"] is None
    assert captured["audit_artifact_root"] is None


def test_sweep_cli_execution_requires_explicit_real_trace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tools.sim2real.cli import _run, build_parser

    profile_path = tmp_path / "profile.json"
    profile_path.write_text(json.dumps(_profile().to_dict()), encoding="utf-8")
    isaac_root = tmp_path / "IsaacLab"
    isaac_root.mkdir()
    (isaac_root / "isaaclab.sh").write_text("#!/bin/sh\n", encoding="utf-8")
    args = build_parser().parse_args(
        [
            "sweep",
            str(profile_path),
            "--scenario",
            "main-step",
            "--mode",
            "one-factor",
            "--space-json",
            '{"simulation_physics.main_drive.damping":[0.3]}',
            "--output",
            str(tmp_path / "output"),
            "--isaaclab-root",
            str(isaac_root),
        ]
    )

    with pytest.raises(ValueError, match="--real-trace is required"):
        _run(args)


def test_sweep_cli_execution_requires_explicit_audit_evidence(
    tmp_path: Path,
) -> None:
    from tools.sim2real.cli import _run, build_parser

    profile_path = tmp_path / "profile.json"
    profile_path.write_text(json.dumps(_profile().to_dict()), encoding="utf-8")
    isaac_root = tmp_path / "IsaacLab"
    isaac_root.mkdir()
    (isaac_root / "isaaclab.sh").write_text("#!/bin/sh\n", encoding="utf-8")
    args = build_parser().parse_args(
        [
            "sweep",
            str(profile_path),
            "--scenario",
            "main-step",
            "--mode",
            "one-factor",
            "--space-json",
            '{"simulation_physics.main_drive.damping":[0.3]}',
            "--output",
            str(tmp_path / "output"),
            "--isaaclab-root",
            str(isaac_root),
            "--real-trace",
            str(tmp_path / "real-reference"),
        ]
    )

    with pytest.raises(ValueError, match="--audit-evidence is required"):
        _run(args)


def test_sweep_cli_requires_known_load_when_effort_limit_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tools.sim2real import sweep_runner
    from tools.sim2real.cli import _run, build_parser

    baseline, _ = _effort_profiles()
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(json.dumps(baseline.to_dict()), encoding="utf-8")
    isaac_root = tmp_path / "IsaacLab"
    isaac_root.mkdir()
    (isaac_root / "isaaclab.sh").write_text("#!/bin/sh\n", encoding="utf-8")
    monkeypatch.setattr(
        sweep_runner,
        "execute_sweep",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("CLI must reject before invoking the runner")
        ),
    )
    args = build_parser().parse_args(
        [
            "sweep",
            str(profile_path),
            "--scenario",
            "main-step",
            "--mode",
            "one-factor",
            "--space-json",
            '{"simulation_physics.main_drive.effort_limit":[2.0]}',
            "--output",
            str(tmp_path / "output"),
            "--isaaclab-root",
            str(isaac_root),
            "--real-trace",
            str(tmp_path / "real-reference"),
        ]
    )

    with pytest.raises(ValueError, match="--known-load-trace.*effort_limit"):
        _run(args)
