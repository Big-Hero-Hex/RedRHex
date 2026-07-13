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
from tools.sim2real.traces import write_trace


def _profile() -> CalibrationProfileV1:
    return CalibrationProfileV1.from_dict(
        {
            "schema_version": 1,
            "profile_id": "baseline",
            "hardware_mapping": {},
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


def _argument(command: list[str], flag: str) -> str:
    return command[command.index(flag) + 1]


def _write_sim_artifact(
    command: list[str], *, profile_override=None, metadata_override=None
) -> None:
    scenario = load_scenario(_argument(command, "--scenario"))
    profile = profile_override or load_profile(_argument(command, "--physics-profile"))
    output = Path(_argument(command, "--output"))
    manifest = write_trace(
        output,
        {
            "command_time_s": np.array([0.0, 1.0]),
            "position_time_s": np.array([0.0, 1.0]),
            "command": np.array([0.0, 0.25]),
            "position": np.array([0.0, 0.1]),
        },
        scenario=scenario,
        source="sim",
        profile=profile,
        metadata=metadata_override
        or {"git_sha": "abc123", "asset_sha256": "a" * 64},
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
) -> dict[str, Any]:
    from tools.sim2real.sweep_runner import execute_sweep

    return execute_sweep(
        output=output,
        scenario=scenario or load_scenario("main-step"),
        candidates=candidates or _candidates(),
        sweep_mode="one-factor",
        scene_mode="fixed-base",
        headless=True,
        seed=17,
        device="cpu",
        provenance={"git_sha": "abc123", "asset_sha256": "a" * 64},
        command_prefix=("/opt/isaaclab/isaaclab.sh", "-p", "-m", "tools.sim2real"),
        generate_only=generate_only,
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
    assert provenance["git_sha"] == "abc123"
    assert provenance["seed"] == 17
    assert json.loads((tmp_path / "sweep" / "results.json").read_text()) == result


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
            metadata_override={"git_sha": "different", "asset_sha256": "a" * 64},
        )
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    with pytest.raises(ContractError, match="candidate 1.*git_sha mismatch"):
        _execute(
            tmp_path / "sweep",
            candidates=_candidates(1),
            run_process=wrong_provenance,
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

    profile_path = tmp_path / "profile.json"
    profile_path.write_text(json.dumps(_profile().to_dict()), encoding="utf-8")
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
            "--provenance-json",
            '{"asset_sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"}',
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
    assert captured["provenance"]["asset_sha256"] == "a" * 64


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
