from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import statistics
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping


SCHEMA_VERSION = 2
LEGACY_SCHEMA_VERSION = 1
CORE_POLICIES = ("teacher_a", "legacy_student", "v2_distilled", "v2_ppo")
REQUIRED_ABLATIONS = (
    "legacy_student",
    "v2_no_aux",
    "v2_velocity",
    "v2_velocity_dynamics",
    "v2_distilled",
    "v2_ppo",
)
PROMOTION_GATES = (
    "no_privileged_leak",
    "torch_onnx_parity",
    "sensor_replay",
    "contract_provenance",
)
V2_TRAINING_IDENTITY_HASH_FIELDS = (
    "observation_contract_sha256",
    "action_contract_sha256",
    "training_calibration_sha256",
    "checkpoint_sha256",
    "architecture_sha256",
    "config_sha256",
    "canonical_config_sha256",
)
ARTIFACT_BINDING_HASH_ALIASES = {
    "observation_contract_sha256": ("observation_contract_sha256", "contract_sha256"),
    "action_contract_sha256": ("action_contract_sha256",),
    "runtime_calibration_sha256": (
        "runtime_calibration_sha256",
        "calibration_sha256",
    ),
    "training_calibration_sha256": ("training_calibration_sha256",),
    "checkpoint_sha256": ("checkpoint_sha256",),
    "architecture_sha256": ("architecture_sha256",),
    "config_sha256": ("config_sha256",),
    "canonical_config_sha256": ("canonical_config_sha256",),
}
V2_POLICIES = {
    "teacher_a",
    "v2_no_aux",
    "v2_velocity",
    "v2_velocity_dynamics",
    "v2_distilled",
    "v2_ppo",
}
EXPECTED_POLICY_IDENTITY = {
    "teacher_a": (
        "rsl_rl_teacher_v2_cfg_entry_point",
        "teacher_v2",
        "teacher_a",
    ),
    "legacy_student": (
        "rsl_rl_distillation_cfg_entry_point",
        "legacy_distillation_v1",
        None,
    ),
    "v2_no_aux": (
        "rsl_rl_distillation_v2_no_aux_cfg_entry_point",
        "student_distilled_v2",
        "distillation_f2",
    ),
    "v2_velocity": (
        "rsl_rl_distillation_v2_velocity_cfg_entry_point",
        "student_distilled_v2",
        "distillation_f2",
    ),
    "v2_velocity_dynamics": (
        "rsl_rl_distillation_v2_velocity_dynamics_cfg_entry_point",
        "student_distilled_v2",
        "distillation_f2",
    ),
    "v2_distilled": (
        "rsl_rl_distillation_v2_cfg_entry_point",
        "student_distilled_v2",
        "distillation_f2",
    ),
    "v2_ppo": (
        "rsl_rl_robust_ppo_v2_cfg_entry_point",
        "student_ppo_v2",
        "ppo_f4",
    ),
}
MAX_PROMOTABLE_REAL_MAIN_ACTION_SATURATION_FRACTION = 0.05
REQUIRED_REAL_HARDWARE_TARGET_TIGHTENING_FRACTION = 0.0
REQUIRED_HARDWARE_TARGET_TIGHTENING_LIMIT_SOURCE_V2 = (
    "exact sim/deployment raw-target parity"
)
STUDENT_TENSORBOARD_ALIASES = {
    "Student/vx_tracking_error": (
        "summary.tracking.mean_abs_vx",
        "command.mae_vx",
    ),
    "Student/vy_leak": (
        "summary.tracking.mean_abs_vy",
        "command.mae_vy",
    ),
    "Student/wz_leak": (
        "summary.tracking.mean_abs_wz",
        "command.mae_wz",
    ),
    "Student/fall_rate": (
        "summary.stability.fall_rate",
        "command.fall_rate",
    ),
    "Student/roll_rms": ("summary.stability.roll_rms",),
    "Student/pitch_rms": ("summary.stability.pitch_rms",),
    "Student/action_saturation_main": (
        "summary.policy.action_saturation_main",
        "summary.policy.main_action_saturation_ratio",
        "command.action_saturation_main",
    ),
    "Student/action_saturation_abad": (
        "summary.policy.action_saturation_abad",
        "summary.policy.abad_action_saturation_ratio",
        "command.action_saturation_abad",
    ),
}
TEACHER_GAP_LIMITS = {
    # Absolute deterioration limits are intentionally half (or stricter for
    # fall rate) of eval_command_sweep.py's forward acceptance defaults.  This
    # makes "close to teacher" an independent gate instead of merely requiring
    # both policies to remain somewhere under the same broad absolute bound.
    "summary.tracking.mean_abs_vx": 0.075,
    "summary.tracking.mean_abs_vy": 0.060,
    "summary.tracking.mean_abs_wz": 0.150,
    "summary.stability.fall_rate": 0.050,
    "summary.stability.roll_rms": 0.350,
    "summary.stability.pitch_rms": 0.350,
}
TEACHER_GAP_LIMIT_SOURCE = "scripts/rsl_rl/eval_command_sweep.py forward defaults"
V2_FORWARD_CONTIGUOUS_SEMANTICS = (
    "one_command_scaled_gait_cycle_velocity_means_with_"
    "pointwise_tilt_height_and_episode_boundary_safety"
)
LEGACY_FORWARD_CONTIGUOUS_SEMANTICS = "instantaneous_samples"
CANONICAL_ACCEPTANCE_PROTOCOL = {
    "duration_s": 2.0,
    "contiguous_env_ratio": 0.50,
    "max_fall_rate": 0.20,
    "forward_vx_abs": 0.15,
    "forward_lin_ratio": 0.55,
    "forward_lateral_leak": 0.12,
    "forward_yaw_leak": 0.30,
    "forward_tilt_bound_rad": 0.70,
    "forward_min_base_height_m": 0.085,
    "lateral_vy_abs": 0.15,
    "lateral_forward_leak": 0.12,
    "lateral_yaw_leak": 0.30,
    "yaw_wz_abs": 0.40,
    "yaw_wz_ratio": 0.55,
    "diag_sign_ratio": 0.70,
    "diag_component_ratio": 0.50,
    "diag_yaw_leak": 0.35,
    "yaw_tilt_ratio": 0.70,
    "yaw_tilt_bound_rad": 0.60,
    "yaw_linear_leak": 0.18,
    "yaw_min_base_height_m": 0.12,
    "skill_pass_ratio": 0.60,
    "overall_pass_ratio": 0.70,
    "max_main_action_saturation_ratio": 0.05,
}
CANONICAL_ACCEPTANCE_PROTOCOL_SOURCE = (
    "scripts/rsl_rl/eval_command_sweep.py parser defaults"
)
COMMAND_SKILLS = {"forward", "lateral", "diagonal", "yaw"}


class StudentTeacherGapError(ValueError):
    """Raised when evaluation artifacts cannot support a valid comparison."""


def _reject_constant(value: str) -> None:
    raise StudentTeacherGapError(f"non-finite JSON constant {value}")


def _load_json(path: Path, *, label: str = "manifest") -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"), parse_constant=_reject_constant)
    except (OSError, json.JSONDecodeError) as exc:
        raise StudentTeacherGapError(f"cannot read {label} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise StudentTeacherGapError(f"{label} must be a JSON object")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdefABCDEF" for character in value)
    )


def _require_sha256(value: Any, label: str) -> str:
    if not _is_sha256(value):
        raise StudentTeacherGapError(f"{label} must be a 64-character hexadecimal sha256")
    return str(value).lower()


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise StudentTeacherGapError(f"{label} must be numeric")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise StudentTeacherGapError(f"{label} must be numeric") from exc
    if not math.isfinite(number):
        raise StudentTeacherGapError(f"{label} must be finite")
    return number


def _read_command_csv(path: Path) -> list[dict[str, str]]:
    try:
        with path.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
    except OSError as exc:
        raise StudentTeacherGapError(f"cannot read command CSV {path}: {exc}") from exc
    required = {
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
        "fall_rate",
        "accept_pass",
    }
    if not rows:
        raise StudentTeacherGapError(f"command CSV is empty: {path}")
    missing = required - set(rows[0])
    if missing:
        raise StudentTeacherGapError(
            f"command CSV {path} lacks columns: {', '.join(sorted(missing))}"
        )
    commands = [row["command"] for row in rows]
    if len(commands) != len(set(commands)):
        raise StudentTeacherGapError(f"command CSV has duplicate command names: {path}")
    return rows


def _read_summary_csv(path: Path) -> dict[str, str]:
    try:
        with path.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
    except OSError as exc:
        raise StudentTeacherGapError(f"cannot read summary CSV {path}: {exc}") from exc
    result: dict[str, str] = {}
    for row in rows:
        metric = row.get("metric")
        if not metric or metric in result:
            raise StudentTeacherGapError(f"invalid or duplicate summary metric in {path}")
        result[metric] = row.get("value", "")
    return result


def _read_episode_csv(path: Path) -> list[dict[str, str]]:
    try:
        with path.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
    except OSError as exc:
        raise StudentTeacherGapError(f"cannot read episode CSV {path}: {exc}") from exc
    required = {
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
    }
    if not rows:
        raise StudentTeacherGapError(f"episode CSV is empty: {path}")
    missing = required - set(rows[0])
    if missing:
        raise StudentTeacherGapError(
            f"episode CSV {path} lacks columns: {', '.join(sorted(missing))}"
        )
    for index, row in enumerate(rows):
        for name in (
            "environment_index",
            "episode_index",
            "sample_count",
            "fall_count",
            "success_count",
        ):
            number = _finite_number(row.get(name), f"episode[{index}].{name}")
            if not number.is_integer() or number < 0.0:
                raise StudentTeacherGapError(
                    f"episode[{index}].{name} must be a non-negative integer"
                )
        if row.get("complete", "").strip().lower() not in {
            "0",
            "1",
            "false",
            "true",
            "no",
            "yes",
        }:
            raise StudentTeacherGapError(
                f"episode[{index}].complete must be an explicit boolean"
            )
    return rows


def _required_summary_value(summary: Mapping[str, str], name: str) -> str:
    value = summary.get(name)
    if not isinstance(value, str) or not value.strip():
        raise StudentTeacherGapError(f"summary is missing required identity metric {name}")
    return value.strip()


def _required_summary_sha256(summary: Mapping[str, str], name: str) -> str:
    return _require_sha256(_required_summary_value(summary, name), f"summary {name}")


def _explicit_pass(value: Any, label: str) -> bool:
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "pass", "yes"}:
        return True
    if normalized in {"0", "false", "fail", "no"}:
        return False
    raise StudentTeacherGapError(f"{label} must explicitly mean PASS or FAIL")


def _unit_interval(value: Any, label: str) -> float:
    number = _finite_number(value, label)
    if not 0.0 <= number <= 1.0:
        raise StudentTeacherGapError(f"{label} must be between zero and one")
    return number


def _same_number(actual: float, expected: float) -> bool:
    return math.isclose(actual, expected, rel_tol=0.0, abs_tol=1.0e-12)


def _same_float64_aggregate(actual: float, expected: float) -> bool:
    return math.isclose(actual, expected, rel_tol=0.0, abs_tol=1.0e-9)


def _nonnegative_integer(value: Any, label: str) -> int:
    number = _finite_number(value, label)
    if not number.is_integer() or number < 0.0:
        raise StudentTeacherGapError(f"{label} must be a non-negative integer")
    return int(number)


def _typed_nonnegative_integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise StudentTeacherGapError(f"{label} must be a non-negative integer")
    return value


def _reconcile_command_episode_evidence(
    command_rows: Iterable[Mapping[str, str]],
    episode_rows: Iterable[Mapping[str, str]],
    summary: Mapping[str, str],
) -> None:
    commands = {str(row["command"]): row for row in command_rows}
    episodes = list(episode_rows)
    expected_episode_rows = _nonnegative_integer(
        _required_summary_value(summary, "evidence.episode_row_count"),
        "evidence.episode_row_count",
    )
    if expected_episode_rows != len(episodes):
        raise StudentTeacherGapError(
            "evidence.episode_row_count does not match the episode CSV"
        )
    num_envs = _nonnegative_integer(
        _required_summary_value(summary, "evaluation.num_envs"),
        "evaluation.num_envs",
    )
    sweep_steps = _nonnegative_integer(
        _required_summary_value(summary, "evaluation.sweep_steps"),
        "evaluation.sweep_steps",
    )
    if num_envs <= 0 or sweep_steps <= 0:
        raise StudentTeacherGapError(
            "evaluation.num_envs and evaluation.sweep_steps must be positive"
        )
    sample_capacity = num_envs * sweep_steps

    by_command: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen_episode_keys: set[tuple[str, int, int]] = set()
    for index, row in enumerate(episodes):
        command = str(row.get("command", "")).strip()
        if command not in commands:
            raise StudentTeacherGapError(
                f"episode[{index}] references unknown command {command!r}"
            )
        expected_skill = str(commands[command].get("skill", "")).strip()
        if str(row.get("skill", "")).strip() != expected_skill:
            raise StudentTeacherGapError(
                f"episode[{index}].skill differs from command {command!r}"
            )
        environment_index = _nonnegative_integer(
            row.get("environment_index"), f"episode[{index}].environment_index"
        )
        episode_index = _nonnegative_integer(
            row.get("episode_index"), f"episode[{index}].episode_index"
        )
        if environment_index >= num_envs:
            raise StudentTeacherGapError(
                f"episode[{index}].environment_index exceeds evaluation.num_envs"
            )
        episode_key = (command, environment_index, episode_index)
        if episode_key in seen_episode_keys:
            raise StudentTeacherGapError(
                f"episode CSV has duplicate command/environment/episode key {episode_key!r}"
            )
        seen_episode_keys.add(episode_key)

        sample_count = _nonnegative_integer(
            row.get("sample_count"), f"episode[{index}].sample_count"
        )
        success_count = _nonnegative_integer(
            row.get("success_count"), f"episode[{index}].success_count"
        )
        fall_count = _nonnegative_integer(
            row.get("fall_count"), f"episode[{index}].fall_count"
        )
        if sample_count <= 0:
            raise StudentTeacherGapError(
                f"episode[{index}].sample_count must be positive"
            )
        if success_count > sample_count:
            raise StudentTeacherGapError(
                f"episode[{index}].success_count exceeds sample_count"
            )
        complete = _explicit_pass(row.get("complete"), f"episode[{index}].complete")
        if fall_count > int(complete):
            raise StudentTeacherGapError(
                f"episode[{index}].fall_count is inconsistent with complete"
            )
        metrics: dict[str, float] = {}
        for name in ("mae_vx", "mae_vy", "mae_wz"):
            metric = _finite_number(row.get(name), f"episode[{index}].{name}")
            if metric < 0.0:
                raise StudentTeacherGapError(
                    f"episode[{index}].{name} must be non-negative"
                )
            metrics[name] = metric
        episode_success_ratio = _unit_interval(
            row.get("success_ratio"), f"episode[{index}].success_ratio"
        )
        if not _same_number(
            episode_success_ratio, success_count / sample_count
        ):
            raise StudentTeacherGapError(
                f"episode[{index}].success_ratio disagrees with its counts"
            )
        by_command[command].append(
            {
                "sample_count": sample_count,
                "success_count": success_count,
                "fall_count": fall_count,
                "complete": complete,
                **metrics,
            }
        )

    for command, command_row in commands.items():
        evidence = by_command.get(command, [])
        if not evidence:
            raise StudentTeacherGapError(
                f"command {command!r} has no episode evidence"
            )
        sample_count = sum(row["sample_count"] for row in evidence)
        success_count = sum(row["success_count"] for row in evidence)
        fall_events = sum(row["fall_count"] for row in evidence)
        episode_ends = sum(int(row["complete"]) for row in evidence)
        if sample_count > sample_capacity:
            raise StudentTeacherGapError(
                f"{command}.sample_count exceeds the evaluation sample capacity"
            )
        recorded_counts = {
            "sample_count": sample_count,
            "success_sample_count": success_count,
            "fall_events": fall_events,
            "episode_ends": episode_ends,
        }
        for name, expected in recorded_counts.items():
            actual = _nonnegative_integer(command_row.get(name), f"{command}.{name}")
            if actual != expected:
                raise StudentTeacherGapError(
                    f"{command}.{name} does not match the episode CSV"
                )
        for name in ("mae_vx", "mae_vy", "mae_wz"):
            weighted_metric = sum(
                row[name] * row["sample_count"] for row in evidence
            ) / sample_count
            command_metric = _finite_number(command_row.get(name), f"{command}.{name}")
            if command_metric < 0.0 or not _same_float64_aggregate(
                command_metric, weighted_metric
            ):
                raise StudentTeacherGapError(
                    f"{command}.{name} does not match episode-weighted evidence"
                )
        command_success_ratio = _unit_interval(
            command_row.get("success_ratio"), f"{command}.success_ratio"
        )
        if not _same_number(
            command_success_ratio, success_count / sample_capacity
        ):
            raise StudentTeacherGapError(
                f"{command}.success_ratio does not match episode success counts"
            )
        command_fall_rate = _unit_interval(
            command_row.get("fall_rate"), f"{command}.fall_rate"
        )
        if not _same_number(
            command_fall_rate, fall_events / max(1, episode_ends)
        ):
            raise StudentTeacherGapError(
                f"{command}.fall_rate does not match episode fall evidence"
            )


def _verify_run_acceptance(
    policy: str,
    rows: Iterable[Mapping[str, str]],
    summary: Mapping[str, str],
) -> dict[str, Any]:
    """Recompute the command and aggregate acceptance chain from bound evidence."""

    command_rows = list(rows)
    thresholds = {
        "duration_s": _finite_number(
            _required_summary_value(summary, "acceptance.duration_s"),
            "acceptance.duration_s",
        ),
        "contiguous_env_ratio": _unit_interval(
            _required_summary_value(
                summary, "acceptance.contiguous_env_ratio_threshold"
            ),
            "acceptance.contiguous_env_ratio_threshold",
        ),
        "max_fall_rate": _unit_interval(
            _required_summary_value(summary, "acceptance.max_fall_rate"),
            "acceptance.max_fall_rate",
        ),
        "forward_vx_abs": _finite_number(
            _required_summary_value(summary, "acceptance.forward_vx_abs"),
            "acceptance.forward_vx_abs",
        ),
        "forward_lin_ratio": _unit_interval(
            _required_summary_value(summary, "acceptance.forward_lin_ratio"),
            "acceptance.forward_lin_ratio",
        ),
        "forward_lateral_leak": _finite_number(
            _required_summary_value(summary, "acceptance.forward_lateral_leak"),
            "acceptance.forward_lateral_leak",
        ),
        "forward_yaw_leak": _finite_number(
            _required_summary_value(summary, "acceptance.forward_yaw_leak"),
            "acceptance.forward_yaw_leak",
        ),
        "forward_tilt_bound_rad": _finite_number(
            _required_summary_value(summary, "acceptance.forward_tilt_bound_rad"),
            "acceptance.forward_tilt_bound_rad",
        ),
        "forward_min_base_height_m": _finite_number(
            _required_summary_value(
                summary, "acceptance.forward_min_base_height_m"
            ),
            "acceptance.forward_min_base_height_m",
        ),
        "lateral_vy_abs": _finite_number(
            _required_summary_value(summary, "acceptance.lateral_vy_abs"),
            "acceptance.lateral_vy_abs",
        ),
        "lateral_forward_leak": _finite_number(
            _required_summary_value(summary, "acceptance.lateral_forward_leak"),
            "acceptance.lateral_forward_leak",
        ),
        "lateral_yaw_leak": _finite_number(
            _required_summary_value(summary, "acceptance.lateral_yaw_leak"),
            "acceptance.lateral_yaw_leak",
        ),
        "yaw_wz_abs": _finite_number(
            _required_summary_value(summary, "acceptance.yaw_wz_abs"),
            "acceptance.yaw_wz_abs",
        ),
        "yaw_wz_ratio": _unit_interval(
            _required_summary_value(summary, "acceptance.yaw_wz_ratio"),
            "acceptance.yaw_wz_ratio",
        ),
        "diag_sign_ratio": _unit_interval(
            _required_summary_value(summary, "acceptance.diag_sign_ratio"),
            "acceptance.diag_sign_ratio",
        ),
        "diag_component_ratio": _unit_interval(
            _required_summary_value(summary, "acceptance.diag_component_ratio"),
            "acceptance.diag_component_ratio",
        ),
        "diag_yaw_leak": _finite_number(
            _required_summary_value(summary, "acceptance.diag_yaw_leak"),
            "acceptance.diag_yaw_leak",
        ),
        "yaw_tilt_ratio": _unit_interval(
            _required_summary_value(summary, "acceptance.yaw_tilt_ratio"),
            "acceptance.yaw_tilt_ratio",
        ),
        "yaw_tilt_bound_rad": _finite_number(
            _required_summary_value(summary, "acceptance.yaw_tilt_bound_rad"),
            "acceptance.yaw_tilt_bound_rad",
        ),
        "yaw_linear_leak": _finite_number(
            _required_summary_value(summary, "acceptance.yaw_linear_leak"),
            "acceptance.yaw_linear_leak",
        ),
        "yaw_min_base_height_m": _finite_number(
            _required_summary_value(summary, "acceptance.yaw_min_base_height_m"),
            "acceptance.yaw_min_base_height_m",
        ),
        "skill_pass_ratio": _unit_interval(
            _required_summary_value(
                summary, "acceptance.skill_pass_ratio_threshold"
            ),
            "acceptance.skill_pass_ratio_threshold",
        ),
        "overall_pass_ratio": _unit_interval(
            _required_summary_value(
                summary, "acceptance.overall_pass_ratio_threshold"
            ),
            "acceptance.overall_pass_ratio_threshold",
        ),
        "max_main_action_saturation_ratio": _unit_interval(
            _required_summary_value(
                summary, "acceptance.max_main_action_saturation_ratio"
            ),
            "acceptance.max_main_action_saturation_ratio",
        ),
    }
    if thresholds["duration_s"] <= 0.0:
        raise StudentTeacherGapError("acceptance.duration_s must be positive")
    if thresholds["forward_vx_abs"] < 0.0:
        raise StudentTeacherGapError("acceptance.forward_vx_abs must be non-negative")
    if any(value < 0.0 for value in thresholds.values()):
        raise StudentTeacherGapError("acceptance thresholds must be non-negative")
    noncanonical_thresholds = {
        name: (thresholds[name], expected)
        for name, expected in CANONICAL_ACCEPTANCE_PROTOCOL.items()
        if not _same_number(thresholds[name], expected)
    }
    if noncanonical_thresholds:
        raise StudentTeacherGapError(
            "acceptance thresholds differ from the canonical eval_command_sweep "
            f"protocol: {noncanonical_thresholds}"
        )

    forward_semantics = _required_summary_value(
        summary, "acceptance.forward_contiguous_semantics"
    )
    expected_forward_semantics = (
        V2_FORWARD_CONTIGUOUS_SEMANTICS
        if policy in V2_POLICIES
        else LEGACY_FORWARD_CONTIGUOUS_SEMANTICS
    )
    if forward_semantics != expected_forward_semantics:
        raise StudentTeacherGapError(
            "acceptance.forward_contiguous_semantics is not the current protocol"
        )

    accepted_commands: list[str] = []
    skill_totals: dict[str, int] = defaultdict(int)
    skill_passes: dict[str, int] = defaultdict(int)
    command_fall_rates: list[float] = []
    for row in command_rows:
        command = str(row.get("command", "")).strip()
        skill = str(row.get("skill", "")).strip()
        if not command:
            raise StudentTeacherGapError("command name must be a non-empty string")
        if skill not in COMMAND_SKILLS:
            raise StudentTeacherGapError(
                f"{command}.skill is not part of the current command protocol"
            )
        contiguous_ratio = _unit_interval(
            row.get("contiguous_success_env_ratio"),
            f"{command}.contiguous_success_env_ratio",
        )
        fall_rate = _unit_interval(row.get("fall_rate"), f"{command}.fall_rate")
        derived_pass = (
            contiguous_ratio >= thresholds["contiguous_env_ratio"]
            and fall_rate <= thresholds["max_fall_rate"]
        )
        if skill == "forward":
            if row.get("contiguous_success_semantics", "").strip() != forward_semantics:
                raise StudentTeacherGapError(
                    f"{command}.contiguous_success_semantics differs from the summary protocol"
                )
            mae_vx = _finite_number(row.get("mae_vx"), f"{command}.mae_vx")
            if mae_vx < 0.0:
                raise StudentTeacherGapError(f"{command}.mae_vx must be non-negative")
            command_vx = _finite_number(row.get("cmd_vx"), f"{command}.cmd_vx")
            forward_limit = max(
                thresholds["forward_vx_abs"],
                (1.0 - thresholds["forward_lin_ratio"]) * abs(command_vx),
            )
            derived_pass = derived_pass and mae_vx <= forward_limit
        elif skill == "diagonal":
            derived_pass = derived_pass and _unit_interval(
                row.get("diag_sign_match_ratio"),
                f"{command}.diag_sign_match_ratio",
            ) >= thresholds["diag_sign_ratio"]
        elif skill == "yaw":
            derived_pass = derived_pass and _unit_interval(
                row.get("yaw_tilt_ok_ratio"),
                f"{command}.yaw_tilt_ok_ratio",
            ) >= thresholds["yaw_tilt_ratio"]

        recorded_pass = _explicit_pass(row.get("accept_pass"), f"{command}.accept_pass")
        if recorded_pass != derived_pass:
            raise StudentTeacherGapError(
                f"{command}.accept_pass disagrees with its command acceptance metrics"
            )
        skill_totals[skill] += 1
        if recorded_pass:
            accepted_commands.append(command)
            skill_passes[skill] += 1
        command_fall_rates.append(fall_rate)

    command_pass_ratio = len(accepted_commands) / len(command_rows)
    skill_pass_ratios = {
        skill: skill_passes[skill] / total
        for skill, total in sorted(skill_totals.items())
    }
    min_skill_pass_ratio = min(skill_pass_ratios.values())
    max_command_fall_rate = max(command_fall_rates)

    summary_command_ratio = _unit_interval(
        _required_summary_value(summary, "acceptance.command_pass_ratio"),
        "acceptance.command_pass_ratio",
    )
    summary_min_skill_ratio = _unit_interval(
        _required_summary_value(summary, "acceptance.min_skill_pass_ratio"),
        "acceptance.min_skill_pass_ratio",
    )
    summary_max_fall_rate = _unit_interval(
        _required_summary_value(summary, "acceptance.max_command_fall_rate"),
        "acceptance.max_command_fall_rate",
    )
    if not _same_number(summary_command_ratio, command_pass_ratio):
        raise StudentTeacherGapError(
            "acceptance.command_pass_ratio does not match command accept_pass values"
        )
    if not _same_number(summary_min_skill_ratio, min_skill_pass_ratio):
        raise StudentTeacherGapError(
            "acceptance.min_skill_pass_ratio does not match command skill ratios"
        )
    if not _same_number(summary_max_fall_rate, max_command_fall_rate):
        raise StudentTeacherGapError(
            "acceptance.max_command_fall_rate does not match command fall rates"
        )

    expected_skill_metrics = {
        f"acceptance.skill_pass_ratio.{skill}": ratio
        for skill, ratio in skill_pass_ratios.items()
    }
    recorded_skill_metrics = {
        name: value
        for name, value in summary.items()
        if name.startswith("acceptance.skill_pass_ratio.")
    }
    if set(recorded_skill_metrics) != set(expected_skill_metrics):
        raise StudentTeacherGapError(
            "acceptance skill pass-ratio metrics do not match command skills"
        )
    for name, expected in expected_skill_metrics.items():
        if not _same_number(_unit_interval(recorded_skill_metrics[name], name), expected):
            raise StudentTeacherGapError(
                f"{name} does not match command accept_pass values"
            )

    overall_pass = (
        command_pass_ratio >= thresholds["overall_pass_ratio"]
        and min_skill_pass_ratio >= thresholds["skill_pass_ratio"]
        and max_command_fall_rate <= thresholds["max_fall_rate"]
    )
    if policy in V2_POLICIES:
        main_saturation_ratio = _unit_interval(
            _required_summary_value(summary, "policy.main_action_saturation_ratio"),
            "policy.main_action_saturation_ratio",
        )
        abad_action_magnitude = _finite_number(
            _required_summary_value(summary, "policy.abad_action_magnitude_mean"),
            "policy.abad_action_magnitude_mean",
        )
        if abad_action_magnitude < 0.0:
            raise StudentTeacherGapError(
                "policy.abad_action_magnitude_mean must be non-negative"
            )
        overall_pass = (
            overall_pass
            and main_saturation_ratio
            <= thresholds["max_main_action_saturation_ratio"]
            and abad_action_magnitude <= 1.0e-6
        )

    raw_overall_status = _required_summary_value(
        summary, "acceptance.overall_status"
    ).upper()
    if raw_overall_status not in {"PASS", "FAIL"}:
        raise StudentTeacherGapError(
            "acceptance.overall_status must explicitly be PASS or FAIL"
        )
    if (raw_overall_status == "PASS") != overall_pass:
        raise StudentTeacherGapError(
            "acceptance.overall_status disagrees with recomputed command and aggregate gates"
        )
    return {
        "accepted_commands": sorted(accepted_commands),
        "overall_pass": overall_pass,
        "protocol": {
            **thresholds,
            "forward_contiguous_semantics": forward_semantics,
        },
    }


def _command_signature(rows: Iterable[Mapping[str, str]]) -> tuple[tuple[Any, ...], ...]:
    return tuple(
        (
            row["command"],
            str(row.get("skill", "")).strip(),
            _finite_number(row["cmd_vx"], f"{row['command']}.cmd_vx"),
            _finite_number(row["cmd_vy"], f"{row['command']}.cmd_vy"),
            _finite_number(row["cmd_wz"], f"{row['command']}.cmd_wz"),
        )
        for row in rows
    )


def _numeric_metrics(
    command_rows: Iterable[Mapping[str, str]], summary: Mapping[str, str]
) -> dict[str, float]:
    values: dict[str, list[float]] = defaultdict(list)
    ignored = {"command", "skill", "accept_pass"}
    for row in command_rows:
        for name, raw in row.items():
            if name in ignored or raw in (None, ""):
                continue
            try:
                values[f"command.{name}"].append(_finite_number(raw, name))
            except StudentTeacherGapError:
                continue
    result = {
        name: statistics.fmean(items) for name, items in values.items() if items
    }
    for name, raw in summary.items():
        try:
            result[f"summary.{name}"] = _finite_number(raw, name)
        except StudentTeacherGapError:
            continue
    return result


def _verify_sensor_replay_sources(
    artifact_path: Path,
    artifact: Mapping[str, Any],
    bindings: Mapping[str, str],
    *,
    training_seed: int,
) -> dict[str, dict[str, str]]:
    from redrhex_policy_io import (
        ContractError,
        ForwardResidualActionContractV2,
        SensorCalibrationProfileV2,
        StudentObservationContractV2,
        validate_calibration_lineage_v2,
    )
    import numpy as np
    from tools.sim2real.import_sensor_v2_rosbag import (
        sha256_path_v2,
        validate_sensor_v2_import_receipt,
    )

    expected_names = {
        "source_bag",
        "import_receipt",
        "capture_attestation",
        "input_trace",
        "onnx",
        "sidecar",
        "hardware_config",
        "output_npz",
    }
    raw_sources = artifact.get("source_artifacts")
    if not isinstance(raw_sources, Mapping) or set(raw_sources) != expected_names:
        raise StudentTeacherGapError(
            "sensor_replay PASS requires source_bag/import_receipt/"
            "capture_attestation/input_trace/onnx/sidecar/hardware_config/"
            "output_npz source artifacts"
        )
    paths: dict[str, Path] = {}
    verified: dict[str, dict[str, str]] = {}
    for name in sorted(expected_names):
        record = raw_sources[name]
        if not isinstance(record, Mapping) or set(record) != {"path", "sha256"}:
            raise StudentTeacherGapError(
                f"sensor_replay source artifact {name} must contain path and sha256"
            )
        raw_path = record["path"]
        if not isinstance(raw_path, str) or not raw_path.strip():
            raise StudentTeacherGapError(
                f"sensor_replay source artifact {name} path is invalid"
            )
        source_path = Path(raw_path).expanduser()
        if not source_path.is_absolute():
            source_path = artifact_path.parent / source_path
        source_path = source_path.resolve()
        expected_sha256 = _require_sha256(
            record["sha256"],
            f"sensor_replay source artifact {name} sha256",
        )
        try:
            actual_sha256 = sha256_path_v2(source_path)
        except OSError as exc:
            raise StudentTeacherGapError(
                f"cannot read sensor_replay source artifact {name}: {exc}"
            ) from exc
        if actual_sha256 != expected_sha256:
            raise StudentTeacherGapError(
                f"sensor_replay source artifact {name} sha256 mismatch"
            )
        paths[name] = source_path
        verified[name] = {
            "path": str(source_path),
            "sha256": actual_sha256,
        }
    if paths["input_trace"] == paths["output_npz"]:
        raise StudentTeacherGapError(
            "sensor_replay input trace and replay output NPZ must be distinct"
        )
    if len(set(paths.values())) != len(paths):
        raise StudentTeacherGapError(
            "sensor_replay source artifact paths must be distinct"
        )
    for name, source_path in paths.items():
        if name != "source_bag" and not source_path.is_file():
            raise StudentTeacherGapError(
                f"sensor_replay source artifact {name} must be a file"
            )
    try:
        receipt = validate_sensor_v2_import_receipt(
            paths["import_receipt"],
            expected_receipt_sha256=verified["import_receipt"]["sha256"],
            expected_trace_path=paths["input_trace"],
        )
    except (ContractError, OSError, ValueError) as exc:
        raise StudentTeacherGapError(
            f"sensor_replay import receipt verification failed: {exc}"
        ) from exc
    if (
        receipt.source_bag_path != paths["source_bag"]
        or receipt.source_bag_sha256 != verified["source_bag"]["sha256"]
    ):
        raise StudentTeacherGapError(
            "sensor_replay receipt does not bind the declared source rosbag"
        )
    if (
        receipt.capture_attestation_path != paths["capture_attestation"]
        or receipt.capture_attestation_sha256
        != verified["capture_attestation"]["sha256"]
    ):
        raise StudentTeacherGapError(
            "sensor_replay receipt does not bind the declared capture attestation"
        )
    if int(artifact.get("sample_count", 0)) != int(
        receipt.payload["sample_count"]
    ):
        raise StudentTeacherGapError(
            "sensor_replay summary sample count disagrees with import receipt"
        )
    if (
        receipt.capture_attestation["runtime_calibration_sha256"]
        != bindings["runtime_calibration_sha256"]
    ):
        raise StudentTeacherGapError(
            "sensor_replay capture attestation runtime calibration disagrees "
            "with replay bindings"
        )

    sidecar = _load_json(paths["sidecar"], label="sensor_replay ONNX sidecar")
    metadata = sidecar.get("metadata")
    contract = sidecar.get("contract")
    action_contract = sidecar.get("action_contract")
    runtime_calibration = sidecar.get("calibration")
    training_calibration = sidecar.get("training_calibration")
    if not all(
        isinstance(value, Mapping)
        for value in (
            metadata,
            contract,
            action_contract,
            runtime_calibration,
            training_calibration,
        )
    ):
        raise StudentTeacherGapError(
            "sensor_replay sidecar lacks contract/calibration lineage records"
        )
    expected_metadata = {
        "contract_sha256": bindings["observation_contract_sha256"],
        "action_contract_sha256": bindings["action_contract_sha256"],
        "calibration_sha256": bindings["runtime_calibration_sha256"],
        "training_calibration_sha256": bindings["training_calibration_sha256"],
        "checkpoint_sha256": bindings["checkpoint_sha256"],
        "architecture_sha256": bindings["architecture_sha256"],
        "config_sha256": bindings["config_sha256"],
        "canonical_config_sha256": bindings["canonical_config_sha256"],
        "training_seed": str(training_seed),
    }
    mismatches = [
        name for name, value in expected_metadata.items() if metadata.get(name) != value
    ]
    if mismatches:
        raise StudentTeacherGapError(
            "sensor_replay sidecar metadata disagrees with replay bindings: "
            + ", ".join(mismatches)
        )
    if (
        contract.get("imu_frame_id") != receipt.payload["imu_frame_id"]
        or contract.get("attitude_mode") != receipt.payload["attitude_mode"]
    ):
        raise StudentTeacherGapError(
            "sensor_replay receipt observation contract disagrees with ONNX sidecar"
        )
    try:
        observation_record = StudentObservationContractV2.from_dict(contract)
        action_record = ForwardResidualActionContractV2.from_dict(action_contract)
        runtime_record = SensorCalibrationProfileV2.from_dict(runtime_calibration)
        training_record = SensorCalibrationProfileV2.from_dict(training_calibration)
        validate_calibration_lineage_v2(
            training_record,
            runtime_record,
            observation_contract=observation_record,
            action_contract=action_record,
            require_runtime_hardware_ready=True,
        )
    except (ContractError, TypeError, ValueError) as exc:
        raise StudentTeacherGapError(
            f"sensor_replay sidecar calibration lineage is invalid: {exc}"
        ) from exc
    if (
        observation_record.sha256 != bindings["observation_contract_sha256"]
        or action_record.sha256 != bindings["action_contract_sha256"]
        or runtime_record.sha256 != bindings["runtime_calibration_sha256"]
        or training_record.sha256 != bindings["training_calibration_sha256"]
    ):
        raise StudentTeacherGapError(
            "sensor_replay sidecar contract/calibration records disagree with bindings"
        )
    if observation_record.sha256 != receipt.payload["observation_contract_sha256"]:
        raise StudentTeacherGapError(
            "sensor_replay receipt observation contract hash disagrees with sidecar"
        )
    policy = artifact["policy"]
    hardware_target_report = artifact["hardware_target_tightening"]
    if hardware_target_report.get("deployment_config") != verified[
        "hardware_config"
    ]:
        raise StudentTeacherGapError(
            "sensor_replay deployment config record disagrees with its source artifact"
        )
    from tools.sim2real import replay_student_observation_v2 as replay_module

    canonical_hardware_config_sha256 = sha256_path_v2(
        replay_module.DEPLOYMENT_HARDWARE_CONFIG_PATH_V2
    )
    if (
        verified["hardware_config"]["sha256"]
        != canonical_hardware_config_sha256
    ):
        raise StudentTeacherGapError(
            "sensor_replay hardware config bytes differ from the canonical "
            "ROS Sensor-V2 deployment config"
        )
    max_saturation_fraction = _finite_number(
        policy.get("max_main_action_saturation_fraction"),
        "sensor_replay max_main_action_saturation_fraction",
    )
    try:
        runner, replay_contract, replay_action_contract, replay_calibration = (
            replay_module._load_bundle(
                paths["onnx"],
                paths["sidecar"],
                require_hardware_ready=True,
            )
        )
        with np.load(paths["input_trace"], allow_pickle=False) as replay_input:
            canonical_trace = {
                name: replay_input[name] for name in replay_input.files
            }
        canonical_outputs, canonical_summary = replay_module.replay_arrays(
            canonical_trace,
            contract=replay_contract,
            action_contract=replay_action_contract,
            calibration=replay_calibration,
            runner=runner,
            trace_kind="real",
            max_period_error_ratio=receipt.payload["max_period_error_ratio"],
            max_main_action_saturation_fraction=max_saturation_fraction,
            deployment_hardware_config_path=paths["hardware_config"],
        )
    except Exception as exc:
        raise StudentTeacherGapError(
            f"sensor_replay canonical rerun failed: {exc}"
        ) from exc
    canonical_bindings = {
        "observation_contract_sha256": canonical_summary.get("contract_sha256"),
        "action_contract_sha256": canonical_summary.get("action_contract_sha256"),
        "runtime_calibration_sha256": canonical_summary.get(
            "runtime_calibration_sha256"
        ),
        "training_calibration_sha256": canonical_summary.get(
            "training_calibration_sha256"
        ),
        "checkpoint_sha256": canonical_summary.get("checkpoint_sha256"),
        "architecture_sha256": canonical_summary.get("architecture_sha256"),
        "config_sha256": canonical_summary.get("config_sha256"),
        "canonical_config_sha256": canonical_summary.get(
            "canonical_config_sha256"
        ),
    }
    if (
        canonical_summary.get("status") != "passed"
        or canonical_bindings != dict(bindings)
        or canonical_summary.get("training_seed") != training_seed
    ):
        raise StudentTeacherGapError(
            "sensor_replay canonical rerun status or provenance disagrees with artifact"
        )
    canonical_hardware_targets = canonical_summary.get(
        "hardware_target_tightening"
    )
    if (
        not isinstance(canonical_hardware_targets, Mapping)
        or artifact.get("hardware_target_tightening")
        != canonical_hardware_targets
    ):
        raise StudentTeacherGapError(
            "sensor_replay hardware target tightening report differs from "
            "canonical deployment-decoder rerun"
        )
    try:
        with np.load(paths["output_npz"], allow_pickle=False) as replay_output:
            if set(replay_output.files) != set(canonical_outputs):
                raise StudentTeacherGapError(
                    "sensor_replay output NPZ arrays differ from canonical replay"
                )
            recorded_outputs = {
                name: np.asarray(replay_output[name]) for name in replay_output.files
            }
    except (OSError, ValueError) as exc:
        raise StudentTeacherGapError(
            f"cannot validate sensor_replay output NPZ: {exc}"
        ) from exc
    exact_arrays = {
        "timestamp_s",
        "sensor_frame_timestamp_s",
        "history_timestamp_s",
        "command",
        "raw_contract_target_main_drive_velocity_rad_s",
        "action_clipped_contract_target_main_drive_velocity_rad_s",
        "hardware_slew_target_main_drive_velocity_rad_s",
        "hardware_target_main_drive_velocity_rad_s",
        "hardware_action_clip_applied",
        "hardware_slew_applied",
        "hardware_velocity_limit_applied",
    }
    for name, expected in canonical_outputs.items():
        recorded = recorded_outputs[name]
        if name == "policy_latency_ms":
            if (
                recorded.shape != np.asarray(expected).shape
                or not np.isfinite(recorded).all()
                or np.any(recorded < 0.0)
            ):
                raise StudentTeacherGapError(
                    "sensor_replay output policy latency is invalid"
                )
            continue
        matches = (
            np.array_equal(recorded, expected)
            if name in exact_arrays
            else recorded.shape == np.asarray(expected).shape
            and np.allclose(recorded, expected, rtol=1.0e-6, atol=1.0e-7)
        )
        if not matches:
            raise StudentTeacherGapError(
                f"sensor_replay output array {name} differs from canonical rerun"
            )
    frames = recorded_outputs["sensor_frames"]
    histories = recorded_outputs["sensor_histories"]
    actions = recorded_outputs["actions"]
    velocity = recorded_outputs["base_velocity_estimate"]
    raw_targets = recorded_outputs[
        "raw_contract_target_main_drive_velocity_rad_s"
    ]
    action_clipped_targets = recorded_outputs[
        "action_clipped_contract_target_main_drive_velocity_rad_s"
    ]
    slew_targets = recorded_outputs[
        "hardware_slew_target_main_drive_velocity_rad_s"
    ]
    hardware_targets = recorded_outputs[
        "hardware_target_main_drive_velocity_rad_s"
    ]
    tightening_flags = (
        recorded_outputs["hardware_action_clip_applied"],
        recorded_outputs["hardware_slew_applied"],
        recorded_outputs["hardware_velocity_limit_applied"],
    )
    if (
        frames.ndim != 2
        or frames.shape[1:] != (36,)
        or histories.ndim != 3
        or histories.shape[1:] != (60, 36)
        or actions.ndim != 2
        or actions.shape[1:] != (12,)
        or velocity.ndim != 2
        or velocity.shape[1:] != (3,)
        or histories.shape[0] <= 0
        or actions.shape[0] != histories.shape[0]
        or velocity.shape[0] != histories.shape[0]
        or any(
            target.shape != (histories.shape[0], 6)
            for target in (
                raw_targets,
                action_clipped_targets,
                slew_targets,
                hardware_targets,
            )
        )
        or any(flag.shape != (histories.shape[0],) for flag in tightening_flags)
        or not all(
            np.isfinite(values).all()
            for values in (
                frames,
                histories,
                actions,
                velocity,
                raw_targets,
                action_clipped_targets,
                slew_targets,
                hardware_targets,
            )
        )
    ):
        raise StudentTeacherGapError(
            "sensor_replay output NPZ has invalid shapes or values"
        )
    total_target_delta = np.abs(hardware_targets - raw_targets)
    total_changed = total_target_delta != 0.0
    total_target_count = int(total_target_delta.size)
    independently_recomputed_total = {
        "target_count": total_target_count,
        "tightened_target_count": int(np.count_nonzero(total_changed)),
        "tightening_fraction": (
            float(np.count_nonzero(total_changed) / total_target_count)
            if total_target_count
            else 0.0
        ),
        "max_abs_delta_rad_s": (
            float(np.max(total_target_delta)) if total_target_count else 0.0
        ),
    }
    if canonical_hardware_targets.get("total") != independently_recomputed_total:
        raise StudentTeacherGapError(
            "sensor_replay hardware target parity metrics disagree with output NPZ"
        )
    saturation_fraction = float(np.mean(np.abs(actions[:, :6]) >= 0.999))
    expected_metrics = {
        "action_abs_max": float(np.max(np.abs(actions))),
        "abad_action_abs_max": float(np.max(np.abs(actions[:, 6:]))),
        "main_action_saturation_fraction": saturation_fraction,
    }
    if (
        int(artifact.get("sensor_frame_count", 0)) != frames.shape[0]
        or int(artifact.get("history_ready_count", 0)) != histories.shape[0]
        or int(policy.get("inference_count", 0)) != actions.shape[0]
        or any(
            not math.isclose(
                _finite_number(policy.get(name), f"sensor_replay {name}"),
                value,
                rel_tol=0.0,
                abs_tol=1.0e-9,
            )
            for name, value in expected_metrics.items()
        )
    ):
        raise StudentTeacherGapError(
            "sensor_replay summary disagrees with its output NPZ"
        )
    return verified


def _resolve_promotion_artifact(
    root: Path,
    gate: str,
    record: Mapping[str, Any],
    *,
    expected_bundle_source_hashes: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    required = {"path", "sha256"}
    missing = required - set(record)
    unexpected = set(record) - required
    if missing or unexpected:
        detail = []
        if missing:
            detail.append("missing " + ", ".join(sorted(missing)))
        if unexpected:
            detail.append("unexpected " + ", ".join(sorted(unexpected)))
        raise StudentTeacherGapError(f"{gate} promotion artifact record is invalid: {'; '.join(detail)}")
    raw_path = record["path"]
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise StudentTeacherGapError(f"{gate} promotion artifact path must be a non-empty string")
    expected_sha256 = _require_sha256(record["sha256"], f"{gate} promotion artifact sha256")
    artifact_path = (root / raw_path).resolve()
    try:
        actual_sha256 = _sha256(artifact_path)
    except OSError as exc:
        raise StudentTeacherGapError(f"cannot read {gate} promotion artifact {artifact_path}: {exc}") from exc
    if actual_sha256 != expected_sha256:
        raise StudentTeacherGapError(
            f"{gate} promotion artifact sha256 mismatch: expected {expected_sha256}, got {actual_sha256}"
        )
    artifact = _load_json(artifact_path, label=f"{gate} promotion artifact")
    if "gate" in artifact and artifact["gate"] != gate:
        raise StudentTeacherGapError(f"{gate} promotion artifact declares gate {artifact.get('gate')!r}")
    status_values = []
    if "status" in artifact:
        status_values.append(artifact["status"])
    gate_result = artifact.get(gate)
    if isinstance(gate_result, Mapping) and "status" in gate_result:
        status_values.append(gate_result["status"])
    if len(status_values) != 1 or not isinstance(status_values[0], str):
        raise StudentTeacherGapError(
            f"{gate} promotion artifact must expose exactly one string status for that gate"
        )
    normalized_status = status_values[0].strip().upper()
    if normalized_status == "PASSED":
        normalized_status = "PASS"
    elif normalized_status == "FAILED":
        normalized_status = "FAIL"
    if normalized_status not in {"PASS", "FAIL"}:
        raise StudentTeacherGapError(f"{gate} promotion artifact status must mean PASS or FAIL")

    if normalized_status == "PASS":
        if gate == "torch_onnx_parity":
            report = artifact.get("torch_onnx_parity")
            recorded = artifact.get("recorded_parity_input")
            metadata = artifact.get("metadata")
            if (
                not isinstance(report, Mapping)
                or not isinstance(recorded, Mapping)
                or not isinstance(metadata, Mapping)
            ):
                raise StudentTeacherGapError(
                    "torch_onnx_parity PASS requires bundle parity and recorded input records"
                )
            random_count = int(report.get("random_sample_count", 0))
            recorded_count = int(report.get("recorded_sample_count", 0))
            action_error = _finite_number(
                report.get("action_max_abs_error"), "torch_onnx_parity action error"
            )
            velocity_error = _finite_number(
                report.get("velocity_max_abs_error"), "torch_onnx_parity velocity error"
            )
            absolute_tolerance = _finite_number(
                report.get("absolute_tolerance"), "torch_onnx_parity tolerance"
            )
            if (
                artifact.get("schema") != "redrhex.torch-onnx-parity-gate.v2"
                or artifact.get("gate") != gate
                or metadata.get("bundle_schema")
                != "redrhex.sensor-policy-bundle.v2"
                or random_count <= 0
                or recorded_count <= 0
                or action_error > absolute_tolerance
                or velocity_error > absolute_tolerance
                or not _is_sha256(recorded.get("sha256"))
            ):
                raise StudentTeacherGapError(
                    "torch_onnx_parity PASS lacks bounded random+recorded numerical evidence"
                )
            checks = artifact.get("checks")
            passed_checks = {
                str(check.get("name"))
                for check in checks
                if isinstance(check, Mapping)
                and isinstance(check.get("status"), str)
                and check["status"].strip().upper() in {"PASS", "PASSED"}
            } if isinstance(checks, list) else set()
            if not {
                "checkpoint_state_loaded_strictly",
                "random_history_parity",
                "recorded_history_parity",
                "source_artifact_hashes",
            }.issubset(passed_checks):
                raise StudentTeacherGapError(
                    "torch_onnx_parity PASS lacks the canonical verifier checks"
                )
        elif gate == "sensor_replay":
            policy = artifact.get("policy")
            hardware_targets = artifact.get("hardware_target_tightening")
            saturation_fraction = _finite_number(
                policy.get("main_action_saturation_fraction")
                if isinstance(policy, Mapping)
                else None,
                "sensor_replay main-action saturation fraction",
            )
            saturation_limit = _finite_number(
                policy.get("max_main_action_saturation_fraction")
                if isinstance(policy, Mapping)
                else None,
                "sensor_replay main-action saturation limit",
            )
            total_tightening = (
                hardware_targets.get("total")
                if isinstance(hardware_targets, Mapping)
                else None
            )
            deployment_config = (
                hardware_targets.get("deployment_config")
                if isinstance(hardware_targets, Mapping)
                else None
            )
            if (
                artifact.get("schema") != "redrhex.sensor-v2-replay.v2"
                or artifact.get("trace_kind") != "real"
                or int(artifact.get("history_ready_count", 0)) <= 0
                or not isinstance(policy, Mapping)
                or int(policy.get("inference_count", 0)) <= 0
                or _finite_number(
                    policy.get("action_abs_max"), "sensor_replay action_abs_max"
                )
                > 1.0
                or _finite_number(
                    policy.get("abad_action_abs_max"), "sensor_replay ABAD action max"
                )
                > 1.0e-6
                or not 0.0 <= saturation_limit <= 1.0
                or saturation_limit
                > MAX_PROMOTABLE_REAL_MAIN_ACTION_SATURATION_FRACTION
                or saturation_fraction > saturation_limit
                or policy.get("main_action_saturation_gate_passed") is not True
                or not isinstance(
                    policy.get("main_action_saturation_limit_source"), str
                )
                or not isinstance(
                    policy.get("main_action_saturation_sensitivity"), Mapping
                )
                or not isinstance(hardware_targets, Mapping)
                or not isinstance(total_tightening, Mapping)
                or not isinstance(deployment_config, Mapping)
                or set(deployment_config) != {"path", "sha256"}
                or not isinstance(deployment_config.get("path"), str)
                or not deployment_config.get("path", "").strip()
                or not _is_sha256(deployment_config.get("sha256"))
                or _finite_number(
                    hardware_targets.get("max_total_tightening_fraction"),
                    "sensor_replay maximum hardware target tightening fraction",
                )
                != REQUIRED_REAL_HARDWARE_TARGET_TIGHTENING_FRACTION
                or hardware_targets.get("limit_source")
                != REQUIRED_HARDWARE_TARGET_TIGHTENING_LIMIT_SOURCE_V2
                or hardware_targets.get("required_for_real_replay") is not True
                or hardware_targets.get("gate_passed") is not True
                or _nonnegative_integer(
                    total_tightening.get("target_count"),
                    "sensor_replay hardware target count",
                )
                != int(policy.get("inference_count", 0)) * 6
                or _nonnegative_integer(
                    total_tightening.get("tightened_target_count"),
                    "sensor_replay tightened hardware target count",
                )
                != 0
                or _finite_number(
                    total_tightening.get("tightening_fraction"),
                    "sensor_replay hardware target tightening fraction",
                )
                != REQUIRED_REAL_HARDWARE_TARGET_TIGHTENING_FRACTION
                or _finite_number(
                    total_tightening.get("max_abs_delta_rad_s"),
                    "sensor_replay hardware target maximum delta",
                )
                != 0.0
            ):
                raise StudentTeacherGapError(
                    "sensor_replay PASS requires real, non-empty, finite, "
                    "unsaturated inference and exact raw/deployment target parity"
                )
        else:
            expected_schema = f"redrhex.{gate.replace('_', '-')}-gate.v2"
            required_checks = {
                "no_privileged_leak": {
                    "actor_inputs_exact",
                    "forbidden_features_absent",
                    "command_separate",
                    "privileged_groups_training_only",
                },
                "contract_provenance": {
                    "observation_contract_hash",
                    "action_contract_hash",
                    "calibration_hash",
                    "runtime_calibration_lineage",
                    "checkpoint_manifest_binding",
                    "architecture_config_binding",
                },
            }[gate]
            checks = artifact.get("checks")
            passed_checks = {
                str(check.get("name"))
                for check in checks
                if isinstance(check, Mapping)
                and isinstance(check.get("status"), str)
                and check["status"].strip().upper() in {"PASS", "PASSED"}
            } if isinstance(checks, list) else set()
            if (
                artifact.get("schema") != expected_schema
                or artifact.get("gate") != gate
                or not required_checks.issubset(passed_checks)
            ):
                raise StudentTeacherGapError(
                    f"{gate} PASS lacks its canonical schema and complete required check set"
                )

    containers = [artifact]
    for name in ("metadata", "provenance"):
        container = artifact.get(name)
        if isinstance(container, Mapping):
            containers.append(container)
    bindings: dict[str, str] = {}
    for field, aliases in ARTIFACT_BINDING_HASH_ALIASES.items():
        declared = {
            _require_sha256(container[alias], f"{gate} promotion artifact {alias}")
            for container in containers
            for alias in aliases
            if alias in container and container[alias] is not None
        }
        if len(declared) > 1:
            raise StudentTeacherGapError(f"{gate} promotion artifact has conflicting {field} values")
        if declared:
            bindings[field] = declared.pop()
    artifact_provenance = artifact.get("provenance")
    raw_training_seed = (
        artifact.get("training_seed")
        if gate == "sensor_replay"
        else (
            artifact_provenance.get("training_seed")
            if isinstance(artifact_provenance, Mapping)
            else None
        )
    )
    training_seed = (
        _typed_nonnegative_integer(
            raw_training_seed, f"{gate} promotion artifact training_seed"
        )
        if normalized_status == "PASS"
        else None
    )
    required_binding_fields = {
        "no_privileged_leak": {
            "observation_contract_sha256",
            "action_contract_sha256",
            "checkpoint_sha256",
            "architecture_sha256",
            "config_sha256",
            "canonical_config_sha256",
        },
        "torch_onnx_parity": set(ARTIFACT_BINDING_HASH_ALIASES),
        "sensor_replay": set(ARTIFACT_BINDING_HASH_ALIASES),
        "contract_provenance": set(ARTIFACT_BINDING_HASH_ALIASES),
    }[gate]
    if normalized_status == "PASS" and not required_binding_fields.issubset(bindings):
        missing_bindings = sorted(required_binding_fields - set(bindings))
        raise StudentTeacherGapError(
            f"{gate} PASS is missing provenance bindings: {', '.join(missing_bindings)}"
        )
    verified_sources: dict[str, dict[str, str]] | None = None
    checkpoint_stage: str | None = None
    if normalized_status == "PASS" and gate == "sensor_replay":
        raw_sources = artifact.get("source_artifacts")
        if not isinstance(raw_sources, Mapping):
            raise StudentTeacherGapError(
                "sensor_replay PASS requires source artifacts"
            )
        if expected_bundle_source_hashes is not None:
            for source_name in ("onnx", "sidecar"):
                source_record = raw_sources.get(source_name)
                actual_hash = (
                    source_record.get("sha256")
                    if isinstance(source_record, Mapping)
                    else None
                )
                if (
                    _require_sha256(
                        actual_hash,
                        f"sensor_replay source artifact {source_name} sha256",
                    )
                    != expected_bundle_source_hashes[source_name]
                ):
                    raise StudentTeacherGapError(
                        "sensor_replay ONNX/sidecar must be the exact bundle proven "
                        "by the canonical Torch/ONNX parity gate"
                    )
        verified_sources = _verify_sensor_replay_sources(
            artifact_path,
            artifact,
            bindings,
            training_seed=training_seed,
        )
    if normalized_status == "PASS" and gate in {
        "no_privileged_leak",
        "torch_onnx_parity",
        "contract_provenance",
    }:
        raw_sources = artifact.get("source_artifacts")
        source_names = {"onnx", "sidecar", "checkpoint"}
        if gate == "torch_onnx_parity":
            source_names.add("parity_input")
        if not isinstance(raw_sources, Mapping) or set(raw_sources) != source_names:
            raise StudentTeacherGapError(
                f"{gate} PASS requires canonical source artifacts: "
                + ", ".join(sorted(source_names))
            )
        verified_paths: dict[str, Path] = {}
        verified_sources = {}
        for source_name in sorted(source_names):
            source_record = raw_sources[source_name]
            if not isinstance(source_record, Mapping) or set(source_record) != {
                "path",
                "sha256",
            }:
                raise StudentTeacherGapError(
                    f"{gate} source artifact {source_name} must contain path and sha256"
                )
            raw_source_path = source_record["path"]
            if not isinstance(raw_source_path, str) or not raw_source_path.strip():
                raise StudentTeacherGapError(
                    f"{gate} source artifact {source_name} path is invalid"
                )
            source_path = Path(raw_source_path)
            if not source_path.is_absolute():
                source_path = artifact_path.parent / source_path
            source_path = source_path.resolve()
            source_sha256 = _require_sha256(
                source_record["sha256"],
                f"{gate} source artifact {source_name} sha256",
            )
            try:
                actual_source_sha256 = _sha256(source_path)
            except OSError as exc:
                raise StudentTeacherGapError(
                    f"cannot read {gate} source artifact {source_name}: {exc}"
                ) from exc
            if actual_source_sha256 != source_sha256:
                raise StudentTeacherGapError(
                    f"{gate} source artifact {source_name} sha256 mismatch"
                )
            verified_paths[source_name] = source_path
            verified_sources[source_name] = {
                "path": str(source_path),
                "sha256": actual_source_sha256,
            }
        from tools.sim2real.generate_sensor_v2_promotion_gates import (
            ONNX_PARITY_ATOL_V2,
            ONNX_PARITY_RTOL_V2,
            PARITY_RANDOM_SAMPLE_COUNT,
            PARITY_RANDOM_SEED,
            PromotionGateGenerationError,
            validate_promotion_bundle,
            validate_torch_onnx_parity_bundle,
        )

        try:
            if gate == "torch_onnx_parity":
                parity_verification = validate_torch_onnx_parity_bundle(
                    onnx_path=verified_paths["onnx"],
                    sidecar_path=verified_paths["sidecar"],
                    checkpoint_path=verified_paths["checkpoint"],
                    parity_input_path=verified_paths["parity_input"],
                    parity_input_sha256=verified_sources["parity_input"]["sha256"],
                )
                verified_bindings = parity_verification["bindings"]
                verified_provenance = parity_verification["provenance"]
            else:
                verified_bindings, verified_provenance = validate_promotion_bundle(
                    onnx_path=verified_paths["onnx"],
                    sidecar_path=verified_paths["sidecar"],
                    checkpoint_path=verified_paths["checkpoint"],
                )
        except PromotionGateGenerationError as exc:
            raise StudentTeacherGapError(
                f"{gate} canonical source verification failed: {exc}"
            ) from exc
        if bindings != verified_bindings:
            raise StudentTeacherGapError(
                f"{gate} bindings differ from canonical source verification"
            )
        if not isinstance(artifact_provenance, Mapping) or any(
            artifact_provenance.get(name) != value
            for name, value in verified_provenance.items()
        ):
            raise StudentTeacherGapError(
                f"{gate} source provenance differs from canonical bundle verification"
            )
        checkpoint_stage = str(verified_provenance["checkpoint_stage"])
        if gate == "torch_onnx_parity":
            canonical_metadata = parity_verification["metadata"]
            if artifact.get("metadata") != canonical_metadata:
                raise StudentTeacherGapError(
                    "torch_onnx_parity metadata differs from canonical verification"
                )
            canonical_report = parity_verification["torch_onnx_parity"]
            artifact_report = artifact.get("torch_onnx_parity")
            if not isinstance(artifact_report, Mapping) or set(artifact_report) != set(
                canonical_report
            ):
                raise StudentTeacherGapError(
                    "torch_onnx_parity report shape differs from canonical verification"
                )
            for name, expected in canonical_report.items():
                actual = artifact_report.get(name)
                if isinstance(expected, int):
                    matches = (
                        isinstance(actual, int)
                        and not isinstance(actual, bool)
                        and actual == expected
                    )
                else:
                    matches = math.isclose(
                        _finite_number(actual, f"torch_onnx_parity {name}"),
                        float(expected),
                        rel_tol=0.0,
                        abs_tol=1.0e-12,
                    )
                if not matches:
                    raise StudentTeacherGapError(
                        f"torch_onnx_parity {name} differs from canonical verification"
                    )
            canonical_recorded = parity_verification["recorded_parity_input"]
            artifact_recorded = artifact.get("recorded_parity_input")
            if not isinstance(artifact_recorded, Mapping) or set(
                artifact_recorded
            ) != set(canonical_recorded):
                raise StudentTeacherGapError(
                    "recorded parity input record differs from canonical verification"
                )
            artifact_recorded_path = Path(str(artifact_recorded.get("path", "")))
            if not artifact_recorded_path.is_absolute():
                artifact_recorded_path = artifact_path.parent / artifact_recorded_path
            if (
                artifact_recorded_path.resolve() != verified_paths["parity_input"]
                or artifact_recorded.get("sha256")
                != canonical_recorded["sha256"]
                or artifact_recorded.get("source_sample_count")
                != canonical_recorded["source_sample_count"]
                or artifact_recorded.get("evaluated_sample_count")
                != canonical_recorded["evaluated_sample_count"]
            ):
                raise StudentTeacherGapError(
                    "recorded parity input record differs from canonical verification"
                )
            if artifact.get("verifier") != {
                "random_seed": PARITY_RANDOM_SEED,
                "random_sample_count": PARITY_RANDOM_SAMPLE_COUNT,
                "absolute_tolerance": ONNX_PARITY_ATOL_V2,
                "relative_tolerance": ONNX_PARITY_RTOL_V2,
            }:
                raise StudentTeacherGapError(
                    "torch_onnx_parity verifier settings are not canonical"
                )
    return {
        "path": str(artifact_path),
        "sha256": actual_sha256,
        "schema_version": artifact.get("schema_version"),
        "status": normalized_status,
        "bindings": bindings,
        "training_seed": training_seed,
        "checkpoint_stage": checkpoint_stage,
        "source_artifacts": verified_sources,
    }


def _resolve_promotion_artifacts(root: Path, value: Any) -> dict[str, dict[str, Any]]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise StudentTeacherGapError("manifest promotion_artifacts must be an object")
    unexpected = sorted(set(value) - set(PROMOTION_GATES))
    if unexpected:
        raise StudentTeacherGapError(
            "manifest promotion_artifacts has unknown gates: " + ", ".join(unexpected)
        )
    result: dict[str, dict[str, Any]] = {}
    expected_bundle_source_hashes: dict[str, str] | None = None
    for gate in PROMOTION_GATES:
        if gate not in value:
            continue
        record = value[gate]
        if not isinstance(record, Mapping):
            raise StudentTeacherGapError(f"{gate} promotion artifact record must be an object")
        result[gate] = _resolve_promotion_artifact(
            root,
            gate,
            record,
            expected_bundle_source_hashes=expected_bundle_source_hashes,
        )
        if gate == "torch_onnx_parity" and result[gate]["status"] == "PASS":
            sources = result[gate].get("source_artifacts")
            if not isinstance(sources, Mapping):
                raise StudentTeacherGapError(
                    "torch_onnx_parity PASS lacks verified bundle sources"
                )
            expected_bundle_source_hashes = {
                name: str(sources[name]["sha256"])
                for name in ("onnx", "sidecar")
            }
    return result


def _resolve_run(root: Path, value: Mapping[str, Any]) -> dict[str, Any]:
    required = {
        "policy",
        "seed",
        "domain",
        "command_csv",
        "episode_csv",
        "summary_csv",
    }
    missing = required - set(value)
    if missing:
        raise StudentTeacherGapError(f"run lacks keys: {', '.join(sorted(missing))}")
    policy = str(value["policy"])
    if policy not in EXPECTED_POLICY_IDENTITY:
        raise StudentTeacherGapError(f"unsupported comparison policy identity: {policy!r}")
    seed = int(value["seed"])
    domain = str(value["domain"])
    command_path = (root / str(value["command_csv"])).resolve()
    episode_path = (root / str(value["episode_csv"])).resolve()
    summary_path = (root / str(value["summary_csv"])).resolve()
    rows = _read_command_csv(command_path)
    episode_rows = _read_episode_csv(episode_path)
    summary = _read_summary_csv(summary_path)
    command_sha256 = _sha256(command_path)
    episode_sha256 = _sha256(episode_path)
    if (
        _required_summary_sha256(summary, "artifact.command_csv_sha256")
        != command_sha256
    ):
        raise StudentTeacherGapError(
            f"{policy} summary artifact.command_csv_sha256 does not match its command CSV"
        )
    if (
        _required_summary_sha256(summary, "artifact.episode_csv_sha256")
        != episode_sha256
    ):
        raise StudentTeacherGapError(
            f"{policy} summary artifact.episode_csv_sha256 does not match its episode CSV"
        )
    command_names = {row["command"] for row in rows}
    episode_command_names = {row["command"] for row in episode_rows}
    if episode_command_names != command_names:
        raise StudentTeacherGapError(
            f"{policy} episode command set differs from its command summary"
        )
    summary_domain = _required_summary_value(summary, "evaluation.domain")
    if summary_domain != domain:
        raise StudentTeacherGapError(
            f"{policy} manifest domain {domain!r} disagrees with summary domain "
            f"{summary_domain!r}"
        )
    if policy == "legacy_student" and summary_domain not in {"nominal", "flat"}:
        raise StudentTeacherGapError(
            "legacy_student evidence is nominal-only; Sensor-V2 held-out evidence "
            "must come from a V2 policy"
        )
    sensor_profile_sha256 = str(summary.get("sensor_dr.profile_sha256", "")).strip()
    sensor_profile_path_value = str(summary.get("sensor_dr.profile_path", "")).strip()
    sensor_profile_purpose = str(summary.get("sensor_dr.profile_purpose", "")).strip()
    sensor_profile_id = str(summary.get("sensor_dr.profile_id", "")).strip()
    sensor_profile_categories = {
        item.strip()
        for item in str(summary.get("sensor_dr.active_categories", "")).split(",")
        if item.strip()
    }
    sensor_profile_parameters = str(
        summary.get("sensor_dr.parameters_json", "")
    ).strip()
    if summary_domain in {"nominal", "flat"}:
        if any(
            (
                sensor_profile_sha256,
                sensor_profile_path_value,
                sensor_profile_purpose,
                sensor_profile_id,
                sensor_profile_categories,
                sensor_profile_parameters,
            )
        ):
            raise StudentTeacherGapError(
                f"nominal domain {summary_domain!r} unexpectedly carries a Sensor DR profile"
            )
    else:
        if not _is_sha256(sensor_profile_sha256):
            raise StudentTeacherGapError(
                f"domain {summary_domain!r} requires a hash-bound Sensor DR profile"
            )
        if sensor_profile_purpose != "held_out_evaluation" or not sensor_profile_id:
            raise StudentTeacherGapError(
                f"domain {summary_domain!r} requires held_out_evaluation profile provenance"
            )
        required_categories = {"noise", "latency", "actuator", "friction"}
        if not required_categories.issubset(sensor_profile_categories):
            raise StudentTeacherGapError(
                f"domain {summary_domain!r} lacks held-out Sensor DR categories"
            )
        try:
            parameters_value = json.loads(sensor_profile_parameters)
        except json.JSONDecodeError as exc:
            raise StudentTeacherGapError(
                f"domain {summary_domain!r} has invalid Sensor DR parameters JSON"
            ) from exc
        if not isinstance(parameters_value, dict) or not parameters_value:
            raise StudentTeacherGapError(
                f"domain {summary_domain!r} has no held-out parameter distributions"
            )
        if not sensor_profile_path_value:
            raise StudentTeacherGapError(
                f"domain {summary_domain!r} requires its Sensor DR profile path"
            )
        profile_path = Path(sensor_profile_path_value).expanduser()
        if not profile_path.is_absolute():
            profile_path = summary_path.parent / profile_path
        profile_path = profile_path.resolve()
        try:
            from tools.sim2real.sensor_dr_profile_v2 import (
                SensorDrProfileErrorV2,
                load_sensor_dr_profile_v2,
            )

            profile, _ = load_sensor_dr_profile_v2(
                profile_path,
                expected_sha256=sensor_profile_sha256,
                expected_purpose="held_out_evaluation",
            )
        except (OSError, SensorDrProfileErrorV2) as exc:
            raise StudentTeacherGapError(
                f"domain {summary_domain!r} Sensor DR profile verification failed: {exc}"
            ) from exc
        expected_parameters_json = json.dumps(
            profile.parameters,
            sort_keys=True,
            separators=(",", ":"),
        )
        if (
            profile.profile_id != sensor_profile_id
            or set(profile.active_categories) != sensor_profile_categories
            or expected_parameters_json != sensor_profile_parameters
        ):
            raise StudentTeacherGapError(
                f"domain {summary_domain!r} summary disagrees with its Sensor DR profile"
            )
    summary_seed = _finite_number(
        _required_summary_value(summary, "evaluation.seed"), "evaluation.seed"
    )
    if not summary_seed.is_integer() or int(summary_seed) != seed:
        raise StudentTeacherGapError(
            f"{policy} manifest seed {seed} disagrees with summary seed {summary_seed}"
        )
    expected_agent, expected_kind, expected_stage = EXPECTED_POLICY_IDENTITY[policy]
    actual_agent = _required_summary_value(summary, "evaluation.agent_entry_point")
    if actual_agent != expected_agent:
        raise StudentTeacherGapError(
            f"{policy} requires agent entry point {expected_agent!r}, got {actual_agent!r}"
        )
    actual_kind = _required_summary_value(summary, "checkpoint.kind")
    if actual_kind != expected_kind:
        raise StudentTeacherGapError(
            f"{policy} requires checkpoint kind {expected_kind!r}, got {actual_kind!r}"
        )
    actual_stage: Any = None
    if expected_stage is not None:
        actual_stage = _required_summary_value(summary, "checkpoint.stage")
        if actual_stage != expected_stage:
            raise StudentTeacherGapError(
                f"{policy} requires checkpoint stage {expected_stage!r}, "
                f"got {actual_stage!r}"
            )
    provenance: dict[str, Any] = {
        "checkpoint_kind": actual_kind,
        "checkpoint_stage": actual_stage,
        "checkpoint_sha256": _required_summary_sha256(summary, "checkpoint.sha256"),
        "config_sha256": _required_summary_sha256(summary, "checkpoint.config_sha256"),
        "canonical_config_sha256": None,
        "training_seed": None,
        "evaluation_protocol_sha256": _required_summary_sha256(
            summary, "evaluation.protocol_sha256"
        ),
        "evaluation_seed": seed,
        "evaluation_domain": summary_domain,
        "sensor_dr_profile_sha256": sensor_profile_sha256 or None,
        "sensor_dr_profile_path": sensor_profile_path_value or None,
        "sensor_dr_profile_id": sensor_profile_id or None,
        "sensor_dr_profile_purpose": sensor_profile_purpose or None,
        "sensor_dr_active_categories": tuple(sorted(sensor_profile_categories)),
        "sensor_dr_parameters_json": sensor_profile_parameters or None,
        "agent_entry_point": actual_agent,
    }
    if policy in V2_POLICIES:
        raw_training_seed = _required_summary_value(summary, "checkpoint.training_seed")
        training_seed = _nonnegative_integer(
            raw_training_seed, "checkpoint.training_seed"
        )
        if raw_training_seed != str(training_seed) or training_seed != seed:
            raise StudentTeacherGapError(
                f"{policy} checkpoint.training_seed must equal evaluation seed {seed}"
            )
        provenance.update(
            observation_contract_sha256=_required_summary_sha256(
                summary, "checkpoint.observation_contract_sha256"
            ),
            action_contract_sha256=_required_summary_sha256(
                summary, "checkpoint.action_contract_sha256"
            ),
            training_calibration_sha256=_required_summary_sha256(
                summary, "checkpoint.training_calibration_sha256"
            ),
            architecture_sha256=_required_summary_sha256(
                summary, "checkpoint.architecture_sha256"
            ),
            canonical_config_sha256=_required_summary_sha256(
                summary, "checkpoint.canonical_config_sha256"
            ),
            training_seed=training_seed,
        )
    declared_provenance = value.get("provenance")
    if declared_provenance not in (None, {}):
        if not isinstance(declared_provenance, Mapping):
            raise StudentTeacherGapError("run provenance must be an object")
        conflicts = {
            name: (declared_provenance[name], expected)
            for name, expected in provenance.items()
            if name in declared_provenance and declared_provenance[name] != expected
        }
        if conflicts:
            raise StudentTeacherGapError(
                f"{policy} caller provenance conflicts with summary evidence: {conflicts}"
            )
    _reconcile_command_episode_evidence(rows, episode_rows, summary)
    acceptance = _verify_run_acceptance(policy, rows, summary)
    return {
        "policy": policy,
        "seed": seed,
        "domain": domain,
        "key": (seed, domain),
        "signature": _command_signature(rows),
        "accepted_commands": acceptance["accepted_commands"],
        "metrics": _numeric_metrics(rows, summary),
        "overall_pass": acceptance["overall_pass"],
        "acceptance_protocol": acceptance["protocol"],
        "provenance": provenance,
        "command_csv": str(command_path),
        "command_csv_sha256": command_sha256,
        "episode_csv": str(episode_path),
        "episode_csv_sha256": episode_sha256,
        "summary_csv": str(summary_path),
        "summary_csv_sha256": _sha256(summary_path),
    }


def _aggregate(runs: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    by_policy_metric: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for run in runs:
        for metric, value in run["metrics"].items():
            by_policy_metric[str(run["policy"])][metric].append(float(value))
    result: dict[str, Any] = {}
    for policy, metrics in sorted(by_policy_metric.items()):
        result[policy] = {
            metric: {
                "count": len(values),
                "mean": statistics.fmean(values),
                "std": statistics.pstdev(values),
            }
            for metric, values in sorted(metrics.items())
        }
    return result


def evaluate_student_teacher_gap(manifest_path: str | Path) -> dict[str, Any]:
    path = Path(manifest_path).resolve()
    manifest = _load_json(path)
    schema_version = manifest.get("schema_version")
    if schema_version == LEGACY_SCHEMA_VERSION:
        raise StudentTeacherGapError(
            "legacy schema_version 1 trusts caller-supplied gate booleans; migrate to "
            "schema_version 2 promotion_artifacts records with path and sha256"
        )
    if schema_version != SCHEMA_VERSION:
        raise StudentTeacherGapError(f"schema_version must be {SCHEMA_VERSION}")
    if "gates" in manifest:
        raise StudentTeacherGapError(
            "schema_version 2 does not accept gates booleans; use promotion_artifacts records"
        )
    raw_runs = manifest.get("runs")
    if not isinstance(raw_runs, list) or not raw_runs:
        raise StudentTeacherGapError("manifest runs must be a non-empty list")
    if any(not isinstance(run, Mapping) for run in raw_runs):
        raise StudentTeacherGapError("each run must be an object")

    runs = [_resolve_run(path.parent, run) for run in raw_runs]
    by_policy: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen_run_keys: set[tuple[str, int, str]] = set()
    artifact_owners: dict[tuple[str, str, str], tuple[str, int, str]] = {}
    artifact_component_owners: dict[tuple[str, str], tuple[str, int, str]] = {}
    for run in runs:
        run_key = (run["policy"], int(run["seed"]), str(run["domain"]))
        if run_key in seen_run_keys:
            raise StudentTeacherGapError(
                f"duplicate policy/seed/domain run: {run_key}"
            )
        seen_run_keys.add(run_key)
        artifact_identity = (
            run["command_csv_sha256"],
            run["episode_csv_sha256"],
            run["summary_csv_sha256"],
        )
        owner = artifact_owners.setdefault(artifact_identity, run_key)
        if owner != run_key:
            raise StudentTeacherGapError(
                "different policy/seed/domain runs reuse the same command/episode/summary "
                f"artifacts: {owner!r} and {run_key!r}"
            )
        for kind in ("command", "episode", "summary"):
            component_identity = (kind, run[f"{kind}_csv_sha256"])
            component_owner = artifact_component_owners.setdefault(
                component_identity, run_key
            )
            if component_owner != run_key:
                raise StudentTeacherGapError(
                    f"different policy/seed/domain runs reuse the same {kind} CSV "
                    f"artifact: {component_owner!r} and {run_key!r}"
                )
        by_policy[run["policy"]].append(run)

    missing_core = sorted(set(CORE_POLICIES) - set(by_policy))
    if missing_core:
        raise StudentTeacherGapError(
            "comparison lacks core policies: " + ", ".join(missing_core)
        )
    reference_keys = {run["key"] for run in by_policy["teacher_a"]}
    reference_seeds = {int(run["seed"]) for run in by_policy["teacher_a"]}
    reference_domains = {str(run["domain"]) for run in by_policy["teacher_a"]}
    if len(reference_seeds) < 3:
        raise StudentTeacherGapError("Teacher A comparison requires at least three distinct seeds")
    nominal_domains = reference_domains & {"nominal", "flat"}
    held_out_domains = reference_domains - {"nominal", "flat"}
    if not nominal_domains or not held_out_domains:
        raise StudentTeacherGapError(
            "comparison requires both nominal and hash-bound held-out evaluation domains"
        )
    nominal_reference_keys = {
        run["key"]
        for run in by_policy["teacher_a"]
        if run["domain"] in {"nominal", "flat"}
    }
    reference_signatures = {
        run["key"]: run["signature"] for run in by_policy["teacher_a"]
    }
    reference_protocols = {
        run["key"]: run["provenance"]["evaluation_protocol_sha256"]
        for run in by_policy["teacher_a"]
    }
    reference_acceptance_protocols = {
        run["key"]: run["acceptance_protocol"]
        for run in by_policy["teacher_a"]
    }
    teacher_protocols_by_domain: dict[str, set[str]] = defaultdict(set)
    teacher_acceptance_protocols_by_domain: dict[str, set[str]] = defaultdict(set)
    for run in by_policy["teacher_a"]:
        domain = str(run["domain"])
        teacher_protocols_by_domain[domain].add(
            run["provenance"]["evaluation_protocol_sha256"]
        )
        teacher_acceptance_protocols_by_domain[domain].add(
            json.dumps(run["acceptance_protocol"], sort_keys=True)
        )
    inconsistent_protocol_domains = sorted(
        domain
        for domain, protocols in teacher_protocols_by_domain.items()
        if len(protocols) != 1
    )
    if inconsistent_protocol_domains:
        raise StudentTeacherGapError(
            "Teacher A evaluation protocol differs across seeds for domain(s): "
            + ", ".join(inconsistent_protocol_domains)
        )
    inconsistent_acceptance_domains = sorted(
        domain
        for domain, protocols in teacher_acceptance_protocols_by_domain.items()
        if len(protocols) != 1
    )
    if inconsistent_acceptance_domains:
        raise StudentTeacherGapError(
            "Teacher A acceptance thresholds differ across seeds for domain(s): "
            + ", ".join(inconsistent_acceptance_domains)
        )
    legacy_protocols_by_domain: dict[str, set[str]] = defaultdict(set)
    for run in by_policy.get("legacy_student", []):
        legacy_protocols_by_domain[str(run["domain"])].add(
            run["provenance"]["evaluation_protocol_sha256"]
        )
    inconsistent_legacy_protocol_domains = sorted(
        domain
        for domain, protocols in legacy_protocols_by_domain.items()
        if len(protocols) != 1
    )
    if inconsistent_legacy_protocol_domains:
        raise StudentTeacherGapError(
            "legacy_student evaluation protocol differs across seeds for domain(s): "
            + ", ".join(inconsistent_legacy_protocol_domains)
        )
    for domain in sorted(held_out_domains):
        domain_runs = [run for run in runs if run["domain"] == domain]
        lineage_fields = (
            "sensor_dr_profile_sha256",
            "sensor_dr_profile_id",
            "sensor_dr_profile_purpose",
            "sensor_dr_active_categories",
            "sensor_dr_parameters_json",
        )
        inconsistent = [
            field
            for field in lineage_fields
            if len({run["provenance"][field] for run in domain_runs}) != 1
        ]
        if inconsistent:
            raise StudentTeacherGapError(
                f"held-out domain {domain!r} changes Sensor DR profile lineage: "
                + ", ".join(inconsistent)
            )
    comparison_policies = set(CORE_POLICIES) | set(REQUIRED_ABLATIONS)
    missing_ablations = sorted(set(REQUIRED_ABLATIONS) - set(by_policy))
    for policy in sorted(comparison_policies & set(by_policy)):
        keys = {run["key"] for run in by_policy[policy]}
        expected_keys = (
            nominal_reference_keys if policy == "legacy_student" else reference_keys
        )
        if policy == "legacy_student" and len({key[0] for key in keys}) < 3:
            raise StudentTeacherGapError(
                "legacy_student nominal comparison requires at least three distinct seeds"
            )
        if keys != expected_keys:
            raise StudentTeacherGapError(
                f"{policy} seed/domain set differs from its required Teacher A "
                f"reference set: {sorted(keys)}"
            )
        for run in by_policy[policy]:
            if run["signature"] != reference_signatures[run["key"]]:
                raise StudentTeacherGapError(
                    f"{policy} command set differs for seed/domain {run['key']}"
                )
            if policy in V2_POLICIES and (
                run["provenance"]["evaluation_protocol_sha256"]
                != reference_protocols[run["key"]]
            ):
                raise StudentTeacherGapError(
                    f"{policy} evaluation protocol differs for seed/domain {run['key']}"
                )
            comparable_acceptance_protocol = {
                name: value
                for name, value in run["acceptance_protocol"].items()
                if name != "forward_contiguous_semantics"
            }
            comparable_reference_protocol = {
                name: value
                for name, value in reference_acceptance_protocols[run["key"]].items()
                if name != "forward_contiguous_semantics"
            }
            if comparable_acceptance_protocol != comparable_reference_protocol:
                raise StudentTeacherGapError(
                    f"{policy} acceptance thresholds differ for seed/domain {run['key']}"
                )

    checkpoint_owners: dict[str, str] = {}
    config_owners: dict[str, str] = {}
    canonical_config_owners: dict[str, str] = {}
    for policy in sorted(comparison_policies & set(by_policy)):
        config_by_seed: dict[int, set[str]] = defaultdict(set)
        checkpoint_by_seed: dict[int, set[str]] = defaultdict(set)
        for run in by_policy[policy]:
            config = run["provenance"]["config_sha256"]
            config_by_seed[int(run["seed"])].add(config)
            config_owner = config_owners.setdefault(config, policy)
            if config_owner != policy:
                raise StudentTeacherGapError(
                    f"config {config} is reused by policy labels "
                    f"{config_owner!r} and {policy!r}"
                )
            digest = run["provenance"]["checkpoint_sha256"]
            checkpoint_by_seed[int(run["seed"])].add(digest)
            owner = checkpoint_owners.setdefault(digest, policy)
            if owner != policy:
                raise StudentTeacherGapError(
                    f"checkpoint {digest} is reused by policy labels {owner!r} and {policy!r}"
                )
        if any(len(values) != 1 for values in config_by_seed.values()):
            raise StudentTeacherGapError(
                f"{policy} uses multiple configs for one training seed"
            )
        if any(len(values) != 1 for values in checkpoint_by_seed.values()):
            raise StudentTeacherGapError(
                f"{policy} uses multiple checkpoints for one training seed"
            )
        if len({next(iter(values)) for values in checkpoint_by_seed.values()}) != len(
            checkpoint_by_seed
        ):
            raise StudentTeacherGapError(
                f"{policy} reuses one checkpoint across independent training seeds"
            )
        canonical_configs = {
            run["provenance"]["canonical_config_sha256"]
            for run in by_policy[policy]
            if run["provenance"]["canonical_config_sha256"] is not None
        }
        if canonical_configs:
            if any(
                run["provenance"]["canonical_config_sha256"] is None
                for run in by_policy[policy]
            ):
                raise StudentTeacherGapError(
                    f"{policy} canonical config evidence is incomplete"
                )
            if len(canonical_configs) != 1:
                raise StudentTeacherGapError(
                    f"{policy} execution-field-stripped configuration differs across seeds"
                )
            canonical_config = canonical_configs.pop()
            canonical_owner = canonical_config_owners.setdefault(
                canonical_config, policy
            )
            if canonical_owner != policy:
                raise StudentTeacherGapError(
                    f"canonical config {canonical_config} is reused by policy labels "
                    f"{canonical_owner!r} and {policy!r}"
                )

    v2_contract_sets = {
        field: {
            run["provenance"][field]
            for policy in V2_POLICIES & set(by_policy)
            for run in by_policy[policy]
        }
        for field in (
            "observation_contract_sha256",
            "action_contract_sha256",
            "training_calibration_sha256",
        )
    }
    inconsistent_v2_contracts = [
        field for field, values in v2_contract_sets.items() if len(values) != 1
    ]
    if inconsistent_v2_contracts:
        raise StudentTeacherGapError(
            "V2 policies do not share the same training contract lineage: "
            + ", ".join(inconsistent_v2_contracts)
        )

    aggregates = _aggregate(runs)
    teacher_metrics = aggregates["teacher_a"]
    nominal_teacher_metrics = _aggregate(
        run
        for run in by_policy["teacher_a"]
        if run["domain"] in {"nominal", "flat"}
    )["teacher_a"]
    teacher_gap: dict[str, dict[str, float]] = {}
    for policy in ("legacy_student", "v2_distilled", "v2_ppo"):
        reference_metrics = (
            nominal_teacher_metrics if policy == "legacy_student" else teacher_metrics
        )
        teacher_gap[policy] = {}
        for metric in sorted(set(reference_metrics) & set(aggregates[policy])):
            teacher_gap[policy][metric] = (
                aggregates[policy][metric]["mean"] - reference_metrics[metric]["mean"]
            )

    promotion_artifacts = _resolve_promotion_artifacts(
        path.parent,
        manifest.get("promotion_artifacts"),
    )
    failed_gates = [
        name
        for name in PROMOTION_GATES
        if name not in promotion_artifacts or promotion_artifacts[name]["status"] != "PASS"
    ]
    ppo_by_key = {run["key"]: run for run in by_policy["v2_ppo"]}
    teacher_by_key = {run["key"]: run for run in by_policy["teacher_a"]}
    failed_ppo_runs = [list(key) for key, run in sorted(ppo_by_key.items()) if not run["overall_pass"]]
    failed_teacher_runs = [
        list(run["key"])
        for run in sorted(by_policy["teacher_a"], key=lambda value: value["key"])
        if not run["overall_pass"]
    ]
    accepted_set_mismatches = [
        list(key)
        for key in sorted(reference_keys)
        if ppo_by_key[key]["accepted_commands"] != teacher_by_key[key]["accepted_commands"]
    ]
    candidate_spec = manifest.get("deployment_candidate")
    if not isinstance(candidate_spec, Mapping) or set(candidate_spec) != {"seed", "domain"}:
        raise StudentTeacherGapError(
            "manifest deployment_candidate must contain exactly seed and domain"
        )
    candidate_key = (int(candidate_spec["seed"]), str(candidate_spec["domain"]))
    candidate_run = ppo_by_key.get(candidate_key)
    if candidate_run is None:
        raise StudentTeacherGapError(
            f"deployment_candidate {candidate_key} is not a V2 PPO evaluation run"
        )
    if candidate_key[1] in {"nominal", "flat"}:
        raise StudentTeacherGapError(
            "deployment_candidate must select a hash-bound held-out evaluation run"
        )
    provenance_hashes = {
        field: str(candidate_run["provenance"][field]).lower()
        for field in V2_TRAINING_IDENTITY_HASH_FIELDS
    }
    runtime_calibration_hashes: set[str] = set()
    ppo_binding_fields = {
        "observation_contract_sha256": "observation_contract_sha256",
        "action_contract_sha256": "action_contract_sha256",
        "training_calibration_sha256": "training_calibration_sha256",
        "checkpoint_sha256": "checkpoint_sha256",
        "architecture_sha256": "architecture_sha256",
        "config_sha256": "config_sha256",
        "canonical_config_sha256": "canonical_config_sha256",
    }
    for gate, artifact in promotion_artifacts.items():
        bindings = artifact["bindings"]
        for artifact_field, provenance_field in ppo_binding_fields.items():
            if artifact_field not in bindings:
                continue
            if bindings[artifact_field] != provenance_hashes[provenance_field]:
                raise StudentTeacherGapError(
                    f"{gate} promotion artifact {artifact_field} does not match "
                    "the selected V2 PPO deployment candidate"
                )
        runtime_hash = bindings.get("runtime_calibration_sha256")
        if runtime_hash is not None:
            runtime_calibration_hashes.add(runtime_hash)
        if artifact["status"] == "PASS" and artifact.get(
            "training_seed"
        ) != candidate_run["provenance"]["training_seed"]:
            raise StudentTeacherGapError(
                f"{gate} promotion artifact training_seed does not match "
                "the selected V2 PPO deployment candidate"
            )
        if artifact.get("checkpoint_stage") is not None and artifact[
            "checkpoint_stage"
        ] != candidate_run["provenance"]["checkpoint_stage"]:
            raise StudentTeacherGapError(
                f"{gate} promotion artifact checkpoint_stage does not match "
                "the selected V2 PPO deployment candidate"
            )
    if len(runtime_calibration_hashes) > 1:
        raise StudentTeacherGapError(
            "promotion artifacts reference different runtime calibration profiles"
        )
    runtime_calibration_sha256 = (
        next(iter(runtime_calibration_hashes))
        if runtime_calibration_hashes
        else None
    )
    provenance_failures: list[list[Any]] = []

    teacher_gap_failures: list[dict[str, Any]] = []
    for key in sorted(reference_keys):
        teacher_run = teacher_by_key[key]
        student_run = ppo_by_key[key]
        for metric, limit in TEACHER_GAP_LIMITS.items():
            teacher_value = teacher_run["metrics"].get(metric)
            student_value = student_run["metrics"].get(metric)
            if teacher_value is None or student_value is None:
                teacher_gap_failures.append(
                    {
                        "seed": key[0],
                        "domain": key[1],
                        "metric": metric,
                        "reason": "required metric missing",
                    }
                )
                continue
            deterioration = float(student_value) - float(teacher_value)
            if deterioration > limit:
                teacher_gap_failures.append(
                    {
                        "seed": key[0],
                        "domain": key[1],
                        "metric": metric,
                        "teacher_value": float(teacher_value),
                        "student_value": float(student_value),
                        "deterioration": deterioration,
                        "maximum_deterioration": limit,
                    }
                )

    promotion_pass = not any(
        (
            failed_gates,
            missing_ablations,
            failed_teacher_runs,
            failed_ppo_runs,
            accepted_set_mismatches,
            provenance_failures,
            teacher_gap_failures,
        )
    )
    public_runs = []
    for run in runs:
        public_runs.append({key: value for key, value in run.items() if key not in {"key", "signature"}})
    return {
        "schema_version": SCHEMA_VERSION,
        "source_manifest": str(path),
        "source_manifest_sha256": _sha256(path),
        "contact_supervision": {"status": "blocked", "reason": "validated simulator contact labels unavailable"},
        "runs": public_runs,
        "aggregate": aggregates,
        "teacher_gap": teacher_gap,
        "promotion": {
            "pass": promotion_pass,
            "failed_gates": failed_gates,
            "missing_ablations": missing_ablations,
            "failed_v2_ppo_runs": failed_ppo_runs,
            "failed_teacher_runs": failed_teacher_runs,
            "teacher_command_set_mismatches": accepted_set_mismatches,
            "provenance_failures": provenance_failures,
            "teacher_gap_gate": {
                "source": TEACHER_GAP_LIMIT_SOURCE,
                "limits": TEACHER_GAP_LIMITS,
                "failures": teacher_gap_failures,
            },
            "deployment_candidate": {
                "seed": candidate_key[0],
                "domain": candidate_key[1],
            },
            "v2_ppo_provenance_hashes": provenance_hashes,
            "runtime_calibration_sha256": runtime_calibration_sha256,
            "artifacts": promotion_artifacts,
        },
    }


def gap_tensorboard_scalars(result: Mapping[str, Any]) -> dict[str, float]:
    """Build only evidence-backed scalar tags from an evaluated gap report."""

    scalars: dict[str, float] = {}
    aggregate = result.get("aggregate", {})
    if isinstance(aggregate, Mapping):
        for policy, metrics in aggregate.items():
            if not isinstance(metrics, Mapping):
                continue
            for metric, statistics_value in metrics.items():
                if not isinstance(statistics_value, Mapping):
                    continue
                for statistic in ("count", "mean", "std"):
                    value = statistics_value.get(statistic)
                    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)):
                        scalars[f"Aggregate/{policy}/{metric}/{statistic}"] = float(value)

    teacher_gap = result.get("teacher_gap", {})
    if isinstance(teacher_gap, Mapping):
        for policy, metrics in teacher_gap.items():
            if not isinstance(metrics, Mapping):
                continue
            for metric, value in metrics.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)):
                    scalars[f"TeacherGap/{policy}/{metric}"] = float(value)

    promotion = result.get("promotion", {})
    if isinstance(promotion, Mapping) and isinstance(promotion.get("pass"), bool):
        scalars["Promotion/pass"] = float(promotion["pass"])

    v2_ppo_metrics = aggregate.get("v2_ppo", {}) if isinstance(aggregate, Mapping) else {}
    if isinstance(v2_ppo_metrics, Mapping):
        for tag, sources in STUDENT_TENSORBOARD_ALIASES.items():
            for source in sources:
                statistics_value = v2_ppo_metrics.get(source)
                if not isinstance(statistics_value, Mapping):
                    continue
                mean = statistics_value.get("mean")
                if isinstance(mean, (int, float)) and not isinstance(mean, bool) and math.isfinite(float(mean)):
                    scalars[tag] = float(mean)
                    break
    return dict(sorted(scalars.items()))


def write_gap_tensorboard(
    result: Mapping[str, Any],
    log_dir: str | Path,
    *,
    step: int = 0,
) -> Path:
    """Write available comparison scalars to one optional TensorBoard event stream."""

    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        raise StudentTeacherGapError("TensorBoard step must be a non-negative integer")
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError as exc:
        raise StudentTeacherGapError("TensorBoard output requires the tensorboard package") from exc

    target = Path(log_dir)
    try:
        target.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise StudentTeacherGapError(f"cannot create TensorBoard log directory {target}: {exc}") from exc
    existing = set(target.glob("events.out.tfevents.*"))
    writer = SummaryWriter(log_dir=str(target))
    try:
        for tag, value in gap_tensorboard_scalars(result).items():
            writer.add_scalar(tag, value, step)
        writer.flush()
    finally:
        writer.close()
    created = sorted(set(target.glob("events.out.tfevents.*")) - existing)
    if not created:
        created = sorted(target.glob("events.out.tfevents.*"))
    if not created:
        raise StudentTeacherGapError(f"TensorBoard writer created no event file in {target}")
    return created[-1]


def write_gap_report(result: Mapping[str, Any], json_path: str | Path, csv_path: str | Path) -> None:
    json_target = Path(json_path)
    csv_target = Path(csv_path)
    json_target.parent.mkdir(parents=True, exist_ok=True)
    csv_target.parent.mkdir(parents=True, exist_ok=True)
    json_payload = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    handle, temporary = tempfile.mkstemp(prefix=f".{json_target.name}-", dir=json_target.parent)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            stream.write(json_payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, json_target)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise

    handle, temporary = tempfile.mkstemp(prefix=f".{csv_target.name}-", dir=csv_target.parent)
    try:
        with os.fdopen(handle, "w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(("policy", "metric", "count", "mean", "std", "gap_from_teacher"))
            aggregate = result["aggregate"]
            gaps = result["teacher_gap"]
            for policy, metrics in aggregate.items():
                for metric, stats in metrics.items():
                    writer.writerow(
                        (
                            policy,
                            metric,
                            stats["count"],
                            stats["mean"],
                            stats["std"],
                            gaps.get(policy, {}).get(metric, ""),
                        )
                    )
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, csv_target)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
