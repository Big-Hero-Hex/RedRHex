#!/usr/bin/env python3
"""Run the fail-closed, three-seed Sensor V2 F0 -> F5 pipeline.

This orchestrator never invents robustness ranges or simulator evidence.  It
requires a passed Isaac F0 artifact plus separate hash-bound F4 training and F5
held-out profiles.  Every model stage is screened with the repository's
``eval_command_sweep.py`` acceptance protocol before promotion.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import re
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.sim2real.sensor_dr_profile_v2 import (  # noqa: E402
    ACTIVE_CATEGORY_PARAMETERS_V2,
    SensorDrProfileV2,
    load_sensor_dr_profile_v2,
)


TASK = "Template-Redrhex-ForwardSensorV2-Direct-v0"
SCHEMA = "redrhex.sensor-v2-full-pipeline.v2"
DEFAULT_SEEDS = (42, 43, 44)
TRAIN_SCRIPT = REPO_ROOT / "scripts/rsl_rl/train.py"
EVAL_SCRIPT = REPO_ROOT / "scripts/rsl_rl/eval_command_sweep.py"

STAGES: dict[str, dict[str, str]] = {
    "f1_teacher": {
        "agent": "rsl_rl_teacher_v2_cfg_entry_point",
        "experiment": "redrhex_forward_v2_teacher",
    },
    "f2_distillation": {
        "agent": "rsl_rl_distillation_v2_cfg_entry_point",
        "experiment": "redrhex_forward_v2_distillation",
    },
    "f3_ppo": {
        "agent": "rsl_rl_ppo_v2_cfg_entry_point",
        "experiment": "redrhex_forward_v2_ppo",
    },
    "f4_robust_ppo": {
        "agent": "rsl_rl_robust_ppo_v2_cfg_entry_point",
        "experiment": "redrhex_forward_v2_robust_ppo",
    },
}


class PipelineGateError(RuntimeError):
    """Raised when a required artifact or empirical gate is not valid."""


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PipelineGateError(f"cannot read {label} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise PipelineGateError(f"{label} must contain a JSON object: {path}")
    return value


def _passed(value: Any) -> bool:
    return isinstance(value, str) and value.strip().lower() in {"pass", "passed"}


def _f0_validator_module() -> Any:
    path = REPO_ROOT / "scripts/rsl_rl/validate_forward_gait_baseline.py"
    spec = importlib.util.spec_from_file_location("redrhex_f0_evidence_validator", path)
    if spec is None or spec.loader is None:
        raise PipelineGateError(f"cannot load the canonical F0 validator: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _current_f0_provenance() -> dict[str, Any]:
    return dict(_f0_validator_module().build_evidence_provenance())


def validate_f0_evidence(path: Path, expected_sha256: str) -> dict[str, str]:
    resolved = path.expanduser().resolve()
    if _sha256_file(resolved) != expected_sha256:
        raise PipelineGateError("F0 evidence SHA-256 mismatch")
    report = _read_json(resolved, "F0 evidence")
    if report.get("schema_version") != "redrhex.forward-gait-f0.v2":
        raise PipelineGateError(
            "F0 evidence must use redrhex.forward-gait-f0.v2; regenerate legacy "
            "v1 instantaneous-sample evidence with the gait-cycle-aware validator"
        )
    if report.get("provenance") != _current_f0_provenance():
        raise PipelineGateError(
            "F0 evidence provenance does not match the current decoder/config/threshold sources"
        )
    overall = report.get("overall_status", report.get("status"))
    structural = report.get("structural_status")
    checks = report.get("checks")
    required_structural_checks = {
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
    }
    structural_names = {
        str(item.get("name"))
        for item in checks
        if isinstance(item, Mapping) and _passed(item.get("status"))
    } if isinstance(checks, list) else set()
    if (
        not _passed(structural)
        or not isinstance(checks, list)
        or not checks
        or any(not isinstance(item, Mapping) or not _passed(item.get("status")) for item in checks)
        or not required_structural_checks.issubset(structural_names)
    ):
        raise PipelineGateError("F0 evidence does not contain the complete structural PASS set")
    simulator = report.get("simulator_rollout", report.get("isaac_f0"))
    simulator_status = simulator.get("status") if isinstance(simulator, Mapping) else None
    if not _passed(overall) or not _passed(simulator_status):
        raise PipelineGateError(
            "F0 evidence must contain both structural PASS and an Isaac simulator rollout PASS"
        )
    assert isinstance(simulator, Mapping)
    required_simulator_fields = {
        "requested": True,
        "mode": "isaac_zero_residual",
        "task": TASK,
        "zero_residual_actions": True,
        "neutral_abad_actions": True,
    }
    for name, expected in required_simulator_fields.items():
        if simulator.get(name) != expected:
            raise PipelineGateError(
                f"F0 simulator evidence field {name!r} must equal {expected!r}"
            )
    gate_configuration = report.get("gate_configuration")
    if not isinstance(gate_configuration, Mapping):
        raise PipelineGateError("F0 evidence lacks its canonical gate configuration")
    for name, expected in {
        "isaac_requested": True,
        "task": TASK,
        "seed": 42,
        "spring_backend": "native",
    }.items():
        if gate_configuration.get(name) != expected:
            raise PipelineGateError(
                f"F0 gate configuration {name!r} must equal {expected!r}"
            )
    for name in ("num_envs", "settle_steps", "warmup_steps", "measurement_steps"):
        if (
            isinstance(gate_configuration.get(name), bool)
            or not isinstance(gate_configuration.get(name), int)
            or gate_configuration[name] <= 0
        ):
            raise PipelineGateError(
                f"F0 gate configuration {name!r} must be a positive integer"
            )
    if gate_configuration.get("commands_vx_m_s") != list((0.22, 0.35, 0.42)):
        raise PipelineGateError("F0 gate configuration changed the fixed command set")
    for simulator_name, configuration_name in (
        ("seed", "seed"),
        ("num_envs", "num_envs"),
        ("settle_steps", "settle_steps"),
        ("warmup_steps", "warmup_steps"),
        ("measurement_steps", "measurement_steps"),
        ("spring_backend", "spring_backend"),
    ):
        if simulator.get(simulator_name) != gate_configuration.get(configuration_name):
            raise PipelineGateError(
                f"F0 simulator field {simulator_name!r} disagrees with gate configuration"
            )
    expected_commands = (0.22, 0.35, 0.42)
    declared_commands = simulator.get("commands_vx_m_s")
    if not isinstance(declared_commands, list) or tuple(declared_commands) != expected_commands:
        raise PipelineGateError("F0 evidence does not use the fixed forward command set")
    commands = simulator.get("commands")
    if not isinstance(commands, list) or len(commands) != len(expected_commands):
        raise PipelineGateError("F0 simulator evidence must contain all three command rows")
    validator = _f0_validator_module()
    expected_thresholds = validator._load_forward_acceptance_thresholds()
    expected_action_contract = validator._shared_action_contract(
        REPO_ROOT, validator._load_snapshot(REPO_ROOT)
    )
    if (
        simulator.get("threshold_source") != "scripts/rsl_rl/eval_command_sweep.py"
        or simulator.get("thresholds") != expected_thresholds
    ):
        raise PipelineGateError(
            "F0 simulator evidence is not bound to the current command-sweep thresholds"
        )
    required_row_checks = {
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
    }
    required_metrics = {
        "mean_forward_displacement_m",
        "actual_forward_speed_mean_m_s",
        "actual_lateral_leak_mean_m_s",
        "actual_yaw_leak_mean_rad_s",
        "reference_relative_tilt_mean_rad",
        "reference_relative_tilt_max_rad",
        "base_height_mean_m",
        "base_height_min_m",
        "fall_rate",
        "forward_mae_m_s",
        "success_sample_ratio",
        "contiguous_success_env_ratio",
        "gait_cycle_window_duration_s",
        "max_abad_target_offset_rad",
    }
    for expected_vx, row in zip(expected_commands, commands, strict=True):
        row_checks = row.get("checks") if isinstance(row, Mapping) else None
        check_names = {
            str(check.get("name"))
            for check in row_checks
            if isinstance(check, Mapping)
        } if isinstance(row_checks, list) else set()
        finite_metrics = (
            isinstance(row, Mapping)
            and all(
                isinstance(row.get(name), (int, float))
                and not isinstance(row.get(name), bool)
                and math.isfinite(float(row[name]))
                for name in required_metrics
            )
        )
        required_forward = max(
            expected_thresholds["forward_abs_m_s"],
            expected_thresholds["forward_command_ratio"] * expected_vx,
        )
        forward_mae_limit = max(
            expected_thresholds["forward_abs_m_s"],
            (1.0 - expected_thresholds["forward_command_ratio"]) * expected_vx,
        )
        expected_cycle_steps = expected_action_contract.command_scaled_cycle_steps(
            expected_vx
        )
        fall_events = row.get("fall_events") if isinstance(row, Mapping) else None
        episode_ends = row.get("episode_ends") if isinstance(row, Mapping) else None
        count_metrics_valid = bool(
            isinstance(fall_events, int)
            and not isinstance(fall_events, bool)
            and isinstance(episode_ends, int)
            and not isinstance(episode_ends, bool)
            and 0 <= fall_events <= episode_ends
            and episode_ends <= int(row["sample_count"])
            and math.isclose(
                float(row["fall_rate"]),
                float(fall_events) / float(max(1, episode_ends)),
                rel_tol=0.0,
                abs_tol=1.0e-12,
            )
        ) if isinstance(row, Mapping) and isinstance(row.get("sample_count"), int) else False
        metrics_pass = bool(
            finite_metrics
            and count_metrics_valid
            and float(row["actual_forward_speed_mean_m_s"]) >= required_forward
            and abs(float(row["actual_forward_speed_mean_m_s"]) - expected_vx)
            <= float(row["forward_mae_m_s"]) + 1.0e-12
            and 0.0 <= float(row["actual_lateral_leak_mean_m_s"])
            and float(row["actual_lateral_leak_mean_m_s"])
            <= expected_thresholds["lateral_leak_m_s"]
            and 0.0 <= float(row["actual_yaw_leak_mean_rad_s"])
            and float(row["actual_yaw_leak_mean_rad_s"])
            <= expected_thresholds["yaw_leak_rad_s"]
            and 0.0 <= float(row["reference_relative_tilt_mean_rad"])
            and float(row["reference_relative_tilt_mean_rad"])
            <= float(row["reference_relative_tilt_max_rad"])
            and float(row["reference_relative_tilt_max_rad"])
            <= expected_thresholds["forward_tilt_bound_rad"]
            and float(row["base_height_min_m"])
            <= float(row["base_height_mean_m"])
            and float(row["base_height_min_m"])
            >= expected_thresholds["forward_min_base_height_m"]
            and float(row["fall_rate"]) <= expected_thresholds["max_fall_rate"]
            and float(row["contiguous_success_env_ratio"])
            >= expected_thresholds["contiguous_env_ratio"]
            and 0.0 <= float(row["forward_mae_m_s"])
            and float(row["forward_mae_m_s"]) <= forward_mae_limit
            and float(row["mean_forward_displacement_m"]) > 0.0
            and 0.0 <= float(row["max_abad_target_offset_rad"])
            and float(row["max_abad_target_offset_rad"]) <= 1.0e-8
            and 0.0 <= float(row["success_sample_ratio"]) <= 1.0
            and isinstance(row.get("gait_cycle_window_steps"), int)
            and not isinstance(row.get("gait_cycle_window_steps"), bool)
            and row["gait_cycle_window_steps"] == expected_cycle_steps
            and math.isclose(
                float(row["gait_cycle_window_duration_s"]),
                expected_cycle_steps * float(simulator["policy_step_dt_s"]),
                rel_tol=0.0,
                abs_tol=1.0e-12,
            )
            and row.get("contiguous_success_semantics")
            == (
                "one_command_scaled_gait_cycle_velocity_means_with_"
                "pointwise_tilt_height_and_episode_boundary_safety"
            )
        )
        if (
            not isinstance(row, Mapping)
            or not _passed(row.get("status"))
            or float(row.get("command_vx_m_s", math.nan)) != expected_vx
            or not isinstance(row_checks, list)
            or check_names != required_row_checks
            or any(
                not isinstance(check, Mapping) or not _passed(check.get("status"))
                for check in row_checks
            )
            or not metrics_pass
            or not isinstance(row.get("sample_count"), int)
            or row["sample_count"]
            != gate_configuration["measurement_steps"] * gate_configuration["num_envs"]
            or not 0.0 <= float(row["fall_rate"]) <= 1.0
            or not 0.0 <= float(row["contiguous_success_env_ratio"]) <= 1.0
            or float(row["mean_forward_displacement_m"]) <= 0.0
            or float(row["max_abad_target_offset_rad"]) > 1.0e-8
        ):
            raise PipelineGateError(
                f"F0 simulator command row {expected_vx} is incomplete or not PASS"
            )
    return {"path": str(resolved), "sha256": expected_sha256}


def _latest_checkpoint(run_dir: Path) -> Path:
    candidates: list[tuple[int, Path]] = []
    for path in run_dir.glob("model_*.pt"):
        match = re.fullmatch(r"model_(\d+)\.pt", path.name)
        if match and path.is_file():
            candidates.append((int(match.group(1)), path))
    if not candidates:
        raise PipelineGateError(f"training produced no model_*.pt checkpoint in {run_dir}")
    return max(candidates, key=lambda item: item[0])[1].resolve()


def _new_run_directory(experiment: str, run_name: str, before: set[Path]) -> Path:
    root = REPO_ROOT / "logs/rsl_rl" / experiment
    candidates = {
        path.resolve()
        for path in root.glob(f"*_{run_name}")
        if path.is_dir()
    } - before
    if len(candidates) != 1:
        raise PipelineGateError(
            f"expected exactly one new {experiment}/{run_name} directory, got {sorted(candidates)}"
        )
    return candidates.pop()


def _run(command: Sequence[str], *, dry_run: bool, planned: list[list[str]]) -> None:
    rendered = [str(item) for item in command]
    planned.append(rendered)
    print("[SENSOR_V2_FULL] " + " ".join(rendered), flush=True)
    if not dry_run:
        subprocess.run(rendered, cwd=REPO_ROOT, check=True)


def _training_command(
    args: argparse.Namespace,
    *,
    stage: str,
    seed: int,
    run_name: str,
    source_checkpoint: Path | None,
) -> list[str]:
    spec = STAGES[stage]
    iterations = {
        "f1_teacher": args.teacher_iterations,
        "f2_distillation": args.distillation_iterations,
        "f3_ppo": args.ppo_iterations,
        "f4_robust_ppo": args.robust_iterations,
    }[stage]
    command = [
        str(args.isaaclab_launcher),
        "-p",
        str(TRAIN_SCRIPT),
        "--task",
        TASK,
        "--agent",
        spec["agent"],
        "--seed",
        str(seed),
        "--num_envs",
        str(args.num_envs),
        "--max_iterations",
        str(iterations),
        "--device",
        args.device,
        "--spring-backend",
        "native",
        "--run_name",
        run_name,
    ]
    if not args.no_headless:
        command.append("--headless")
    if stage == "f2_distillation":
        command.extend(("--teacher_checkpoint", str(source_checkpoint)))
    elif stage == "f3_ppo":
        command.extend(("--student_checkpoint", str(source_checkpoint)))
    elif stage == "f4_robust_ppo":
        command.extend(
            (
                "--ppo_checkpoint",
                str(source_checkpoint),
                "--sensor-dr-profile",
                str(args.f4_profile),
                "--sensor-dr-profile-sha256",
                args.f4_profile_sha256,
            )
        )
    return command


def _train_stage(
    args: argparse.Namespace,
    *,
    stage: str,
    seed: int,
    pipeline_id: str,
    source_checkpoint: Path | None,
    dry_run: bool,
    planned: list[list[str]],
) -> tuple[Path, Path]:
    spec = STAGES[stage]
    run_name = f"{pipeline_id}_{stage}_seed{seed}"
    root = REPO_ROOT / "logs/rsl_rl" / spec["experiment"]
    before = {path.resolve() for path in root.glob(f"*_{run_name}") if path.is_dir()}
    _run(
        _training_command(
            args,
            stage=stage,
            seed=seed,
            run_name=run_name,
            source_checkpoint=source_checkpoint,
        ),
        dry_run=dry_run,
        planned=planned,
    )
    if dry_run:
        run_dir = args.artifact_root / "planned" / stage / f"seed-{seed}"
        return run_dir, run_dir / "model_FINAL.pt"
    run_dir = _new_run_directory(spec["experiment"], run_name, before)
    return run_dir, _latest_checkpoint(run_dir)


def _summary_status(path: Path) -> str:
    try:
        with path.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
    except OSError as exc:
        raise PipelineGateError(f"cannot read command-sweep summary {path}: {exc}") from exc
    values = {
        row.get("metric", ""): row.get("value", "")
        for row in rows
        if row.get("metric")
    }
    return str(values.get("acceptance.overall_status", ""))


def _screen_stage(
    args: argparse.Namespace,
    *,
    label: str,
    stage: str,
    seed: int,
    checkpoint: Path,
    held_out: bool,
    dry_run: bool,
    planned: list[list[str]],
) -> dict[str, Any]:
    output = args.artifact_root / "evaluation" / label / f"seed-{seed}" / "commands.csv"
    command = [
        str(args.isaaclab_launcher),
        "-p",
        str(EVAL_SCRIPT),
        "--task",
        TASK,
        "--agent",
        STAGES[stage]["agent"],
        "--checkpoint",
        str(checkpoint),
        "--seed",
        str(seed),
        "--num_envs",
        str(args.eval_num_envs),
        "--warmup_steps",
        str(args.eval_warmup_steps),
        "--sweep_steps",
        str(args.eval_sweep_steps),
        "--eval_profile",
        "stage1",
        "--evaluation-domain",
        "held_out" if held_out else "nominal",
        "--spring-backend",
        "native",
        "--csv",
        str(output),
        "--headless",
    ]
    if held_out:
        command.extend(
            (
                "--sensor-dr-profile",
                str(args.f5_profile),
                "--sensor-dr-profile-sha256",
                args.f5_profile_sha256,
            )
        )
    if not dry_run:
        output.parent.mkdir(parents=True, exist_ok=True)
    _run(command, dry_run=dry_run, planned=planned)
    summary = output.with_name("commands_summary.csv")
    if dry_run:
        return {
            "status": "planned",
            "command_csv": str(output),
            "summary_csv": str(summary),
        }
    if not output.is_file() or not summary.is_file():
        raise PipelineGateError(f"{label} seed {seed} did not produce both CSV artifacts")
    status = _summary_status(summary)
    if status != "PASS":
        raise PipelineGateError(f"{label} seed {seed} acceptance is {status or 'missing'}")
    return {
        "status": "passed",
        "command_csv": str(output.resolve()),
        "command_csv_sha256": _sha256_file(output),
        "summary_csv": str(summary.resolve()),
        "summary_csv_sha256": _sha256_file(summary),
    }


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--isaaclab-launcher", type=Path, required=True)
    parser.add_argument("--f0-evidence", type=Path, required=True)
    parser.add_argument("--f0-evidence-sha256", required=True)
    parser.add_argument("--f4-profile", type=Path, required=True)
    parser.add_argument("--f4-profile-sha256", required=True)
    parser.add_argument("--f5-profile", type=Path, required=True)
    parser.add_argument("--f5-profile-sha256", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--num_envs", type=int, default=64)
    parser.add_argument("--eval_num_envs", type=int, default=256)
    parser.add_argument("--teacher_iterations", type=int, default=1500)
    parser.add_argument("--distillation_iterations", type=int, default=800)
    parser.add_argument("--ppo_iterations", type=int, default=1500)
    parser.add_argument("--robust_iterations", type=int, default=600)
    parser.add_argument("--eval_warmup_steps", type=int, default=120)
    parser.add_argument("--eval_sweep_steps", type=int, default=600)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--pipeline-id", default="")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--no-headless", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if len(args.seeds) < 3 or len(set(args.seeds)) != len(args.seeds):
        parser.error("--seeds requires at least three unique experimental seeds")
    for name in (
        "num_envs",
        "eval_num_envs",
        "teacher_iterations",
        "distillation_iterations",
        "ppo_iterations",
        "robust_iterations",
        "eval_warmup_steps",
        "eval_sweep_steps",
    ):
        if getattr(args, name) <= 0:
            parser.error(f"--{name} must be positive")
    raw_id = args.pipeline_id.strip() or datetime.now(timezone.utc).strftime(
        "sensor_v2_%Y%m%dT%H%M%SZ"
    )
    pipeline_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", raw_id).strip("_.-")
    if not pipeline_id:
        parser.error("--pipeline-id has no usable characters")
    args.pipeline_id = pipeline_id[:96]
    args.output = (
        args.output
        or REPO_ROOT / "logs/rsl_rl/pipeline" / f"{args.pipeline_id}_f0_f5.json"
    ).resolve()
    args.artifact_root = args.output.parent / f"{args.output.stem}_artifacts"
    args.isaaclab_launcher = args.isaaclab_launcher.expanduser().resolve()
    if not args.dry_run and not args.isaaclab_launcher.is_file():
        parser.error(f"Isaac Lab launcher does not exist: {args.isaaclab_launcher}")
    return args


def _profile_record(
    path: Path, digest: str, *, purpose: str
) -> tuple[SensorDrProfileV2, dict[str, str]]:
    profile, actual = load_sensor_dr_profile_v2(
        path, expected_sha256=digest, expected_purpose=purpose
    )
    return profile, {
        "path": str(path.expanduser().resolve()),
        "sha256": actual,
        "profile_id": profile.profile_id,
        "purpose": profile.purpose,
    }


def _canonical_active_profile_parameters(profile: SensorDrProfileV2) -> dict[str, Any]:
    """Return only perturbation distributions that can change rollout samples."""

    active: dict[str, Any] = {}
    for name, value in profile.parameters.items():
        if name == "sim2real_command_delay_steps":
            if int(value) > 0:
                active[name] = int(value)
            continue
        if not isinstance(value, tuple):
            # Validated boolean fields only enable physical ranges; they are not
            # distributions and are implied by the active range retained below.
            continue
        numeric = tuple(float(item) for item in value)
        neutral = (1.0, 1.0) if name.startswith("dr_") else (0.0, 0.0)
        if numeric != neutral:
            active[name] = numeric
    return dict(sorted(active.items()))


def _active_profile_parameters_by_category(
    profile: SensorDrProfileV2,
) -> dict[str, dict[str, Any]]:
    active = _canonical_active_profile_parameters(profile)
    return {
        category: {
            name: active[name]
            for name in sorted(parameters)
            if name in active
        }
        for category, parameters in ACTIVE_CATEGORY_PARAMETERS_V2.items()
    }


def _validate_held_out_category_distributions(
    training: SensorDrProfileV2,
    held_out: SensorDrProfileV2,
) -> dict[str, dict[str, dict[str, Any]]]:
    training_categories = _active_profile_parameters_by_category(training)
    held_out_categories = _active_profile_parameters_by_category(held_out)
    required = ("noise", "latency", "actuator", "friction")
    reused = [
        category
        for category in required
        if training_categories[category] == held_out_categories[category]
    ]
    if reused:
        raise PipelineGateError(
            "F5 must use held-out distributions for every required category; "
            f"unchanged categories: {reused}"
        )
    return {
        "f4": training_categories,
        "f5": held_out_categories,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _arguments(argv)
    f0 = validate_f0_evidence(args.f0_evidence, args.f0_evidence_sha256)
    f4_profile_value, f4_profile = _profile_record(
        args.f4_profile, args.f4_profile_sha256, purpose="training_curriculum"
    )
    f5_profile_value, f5_profile = _profile_record(
        args.f5_profile, args.f5_profile_sha256, purpose="held_out_evaluation"
    )
    f4_evidence = {item.sha256 for item in f4_profile_value.evidence}
    f5_evidence = {item.sha256 for item in f5_profile_value.evidence}
    if (
        f4_profile["sha256"] == f5_profile["sha256"]
        or f4_profile_value.profile_id == f5_profile_value.profile_id
        or f4_evidence & f5_evidence
    ):
        raise PipelineGateError(
            "F5 held-out evaluation must use a distinct profile, profile_id, and "
            "evidence artifacts that were not used to construct the F4 curriculum"
        )
    if _canonical_active_profile_parameters(
        f4_profile_value
    ) == _canonical_active_profile_parameters(f5_profile_value):
        raise PipelineGateError(
            "F5 held-out active perturbation distributions must differ from the F4 curriculum"
        )
    required_categories = {
        "F4": {"sensor", "actuator"},
        "F5": {"noise", "latency", "actuator", "friction"},
    }
    for label, profile in (("F4", f4_profile_value), ("F5", f5_profile_value)):
        missing = required_categories[label] - set(profile.active_categories)
        if missing:
            raise PipelineGateError(
                f"{label} profile lacks required active perturbation categories: {sorted(missing)}"
            )
    held_out_category_distributions = _validate_held_out_category_distributions(
        f4_profile_value, f5_profile_value
    )

    planned: list[list[str]] = []
    checkpoints: dict[str, dict[int, Path]] = {stage: {} for stage in STAGES}
    runs: list[dict[str, Any]] = []
    evaluations: list[dict[str, Any]] = []

    source_stage = {
        "f1_teacher": None,
        "f2_distillation": "f1_teacher",
        "f3_ppo": "f2_distillation",
        "f4_robust_ppo": "f3_ppo",
    }
    for stage in STAGES:
        for seed in args.seeds:
            prior = source_stage[stage]
            source = checkpoints[prior][seed] if prior is not None else None
            run_dir, checkpoint = _train_stage(
                args,
                stage=stage,
                seed=seed,
                pipeline_id=args.pipeline_id,
                source_checkpoint=source,
                dry_run=args.dry_run,
                planned=planned,
            )
            checkpoints[stage][seed] = checkpoint
            record: dict[str, Any] = {
                "stage": stage,
                "seed": seed,
                "run_dir": str(run_dir),
                "checkpoint": str(checkpoint),
            }
            if not args.dry_run:
                record["checkpoint_sha256"] = _sha256_file(checkpoint)
            runs.append(record)

        # Teacher must pass before F2; distilled must pass before F3.  F3 and
        # F4 nominal screens also prevent robustness training from hiding a
        # nominal regression.  F5 below is a separate held-out domain.
        for seed in args.seeds:
            evidence = _screen_stage(
                args,
                label=f"{stage}_nominal",
                stage=stage,
                seed=seed,
                checkpoint=checkpoints[stage][seed],
                held_out=False,
                dry_run=args.dry_run,
                planned=planned,
            )
            evaluations.append({"gate": f"{stage}_nominal", "seed": seed, **evidence})

    for seed in args.seeds:
        evidence = _screen_stage(
            args,
            label="f5_held_out",
            stage="f4_robust_ppo",
            seed=seed,
            checkpoint=checkpoints["f4_robust_ppo"][seed],
            held_out=True,
            dry_run=args.dry_run,
            planned=planned,
        )
        evaluations.append({"gate": "f5_held_out", "seed": seed, **evidence})

    result = {
        "schema": SCHEMA,
        "status": "planned" if args.dry_run else "passed",
        "deployment_eligible": False,
        "limitation": (
            "simulation gates do not authorize motor output; real replay, hardware-ready "
            "calibration, preflight, and explicit operator authorization remain mandatory"
        ),
        "pipeline_id": args.pipeline_id,
        "task": TASK,
        "seeds": list(args.seeds),
        "f0_evidence": f0,
        "f4_profile": f4_profile,
        "f5_profile": f5_profile,
        "held_out_category_distributions": held_out_category_distributions,
        "runs": runs,
        "evaluations": evaluations,
        "planned_commands": planned,
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    _atomic_write_json(args.output, result)
    print("SENSOR_V2_FULL_PIPELINE_RESULT: " + json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
