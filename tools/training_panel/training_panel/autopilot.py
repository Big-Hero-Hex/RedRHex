from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Iterable, Mapping, Sequence


GOAL_SCHEMA_VERSION = "redrhex.autopilot.goal.v1"
REWARD_CATALOG_SCHEMA_VERSION = "redrhex.autopilot.reward-catalog.v1"
DECISION_SCHEMA_VERSION = "redrhex.autopilot.decision.v1"
EVALUATION_SCHEMA_VERSION = "redrhex.autopilot.evaluation.v1"
CAMPAIGN_SCHEMA_VERSION = "redrhex.autopilot.campaign.v1"
DECISION_CONTEXT_SCHEMA_VERSION = "redrhex.autopilot.decision-context.v1"
CAPABILITIES_SCHEMA_VERSION = "redrhex.autopilot.capabilities.v1"

DIRECT_TASK = "Template-Redrhex-Direct-v0"
FORWARD_FAST_TASK = "Template-Redrhex-ForwardFast-Direct-v0"
SUPPORTED_TASKS = (FORWARD_FAST_TASK, DIRECT_TASK)
SUPPORTED_GAITS = ("walk", "run")
DEFAULT_TRAINING_SEEDS = (42, 43, 44)
MAX_TRAINING_TRIALS = 24
MAX_GPU_HOURS = 72.0
MAX_CONNECTOR_POLLS = 300
REWARD_MOVE_MULTIPLIERS = (0.8, 0.9, 1.0, 1.1, 1.2)

SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
REWARD_KEY_RE = re.compile(r"^v2_reward_scales\.[A-Za-z0-9_]+$")


class AutopilotValidationError(ValueError):
    """A request violates a versioned autopilot contract."""


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise AutopilotValidationError(f"{name} must be an object")
    return value


def _strict_keys(
    value: Mapping[str, Any],
    *,
    required: Iterable[str],
    optional: Iterable[str] = (),
    name: str,
) -> None:
    required_set = set(required)
    allowed = required_set | set(optional)
    missing = sorted(required_set - set(value))
    unknown = sorted(set(value) - allowed)
    if missing:
        raise AutopilotValidationError(f"{name} is missing fields: {', '.join(missing)}")
    if unknown:
        raise AutopilotValidationError(f"{name} has unknown fields: {', '.join(unknown)}")


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise AutopilotValidationError(f"{name} must be a finite number, not a boolean")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise AutopilotValidationError(f"{name} must be a finite number") from exc
    if not math.isfinite(result):
        raise AutopilotValidationError(f"{name} must be finite")
    return result


def _integer(value: Any, name: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise AutopilotValidationError(f"{name} must be an integer")
    if minimum is not None and value < minimum:
        raise AutopilotValidationError(f"{name} must be at least {minimum}")
    return value


def _text(value: Any, name: str, *, maximum: int = 4096) -> str:
    if not isinstance(value, str) or not value.strip():
        raise AutopilotValidationError(f"{name} must be a non-empty string")
    result = value.strip()
    if len(result) > maximum:
        raise AutopilotValidationError(f"{name} must be at most {maximum} characters")
    return result


def _optional_text(value: Any, name: str, *, maximum: int = 4096) -> str | None:
    if value is None:
        return None
    return _text(value, name, maximum=maximum)


def _sha256(value: Any, name: str, *, optional: bool = False) -> str | None:
    if value is None and optional:
        return None
    if not isinstance(value, str) or not SHA256_RE.fullmatch(value):
        raise AutopilotValidationError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _schema(value: Any, expected: str, name: str) -> str:
    if value != expected:
        raise AutopilotValidationError(f"{name}.schema_version must be {expected!r}")
    return expected


def _json_finite(value: Any, name: str) -> None:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return
    if isinstance(value, (int, float)):
        _finite(value, name)
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise AutopilotValidationError(f"{name} object keys must be strings")
            _json_finite(item, f"{name}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _json_finite(item, f"{name}[{index}]")
        return
    raise AutopilotValidationError(f"{name} contains a non-JSON value")


# Intervals deliberately preserve the disjoint positive/negative ranges used by
# lateral, diagonal, and yaw commands. A single min/max range would incorrectly
# include near-zero commands that the environment does not train.
CommandIntervals = dict[str, tuple[tuple[float, float], ...]]


STAGE_COMMAND_PRESETS: dict[str, dict[int, CommandIntervals]] = {
    FORWARD_FAST_TASK: {
        1: {"vx": ((0.22, 0.42),), "vy": ((0.0, 0.0),), "wz": ((0.0, 0.0),)},
    },
    DIRECT_TASK: {
        1: {"vx": ((0.20, 0.45),), "vy": ((0.0, 0.0),), "wz": ((0.0, 0.0),)},
        2: {"vx": ((0.0, 0.0),), "vy": ((-0.64, -0.32), (0.32, 0.64)), "wz": ((0.0, 0.0),)},
        3: {"vx": ((0.34, 0.60),), "vy": ((-0.48, -0.28), (0.28, 0.48)), "wz": ((0.0, 0.0),)},
        4: {"vx": ((0.0, 0.0),), "vy": ((0.0, 0.0),), "wz": ((-0.62, -0.20), (0.20, 0.62))},
        5: {
            "vx": ((0.0, 0.0), (0.22, 0.45), (0.32, 0.56)),
            "vy": ((-0.60, -0.28), (-0.44, -0.26), (0.0, 0.0), (0.26, 0.44), (0.28, 0.60)),
            "wz": ((-0.70, -0.28), (0.0, 0.0), (0.28, 0.70)),
        },
    },
}

STAGE_DIRECTIONS = {
    1: ("forward",),
    2: ("left", "right"),
    3: ("forward_left", "forward_right"),
    4: ("yaw_ccw", "yaw_cw"),
    5: ("forward", "left", "right", "forward_left", "forward_right", "yaw_ccw", "yaw_cw"),
}

FORWARD_FAST_REWARD_KEYS = (
    "v2_reward_scales.forward_progress",
    "v2_reward_scales.velocity_tracking",
    "v2_reward_scales.axis_suppression",
    "v2_reward_scales.height_maintain",
    "v2_reward_scales.height_low_penalty",
    "v2_reward_scales.leg_moving",
    "v2_reward_scales.stall_penalty",
    "v2_reward_scales.energy_per_distance",
)

# Direct exposes shaping terms only. Targets, sigmas, caps, termination/fall
# values, physics, and health gates are intentionally absent.
DIRECT_REWARD_KEYS_BY_STAGE = {
    1: (
        "v2_reward_scales.forward_progress",
        "v2_reward_scales.velocity_tracking",
        "v2_reward_scales.axis_suppression",
        "v2_reward_scales.forward_prior_coherence",
        "v2_reward_scales.forward_prior_antiphase",
        "v2_reward_scales.forward_prior_duty",
        "v2_reward_scales.forward_prior_vel_ratio",
        "v2_reward_scales.forward_prior_overlap",
        "v2_reward_scales.height_maintain",
        "v2_reward_scales.height_low_penalty",
        "v2_reward_scales.leg_moving",
        "v2_reward_scales.stall_penalty",
        "v2_reward_scales.energy_per_distance",
    ),
    2: (
        "v2_reward_scales.velocity_tracking",
        "v2_reward_scales.mode_specialization",
        "v2_reward_scales.axis_suppression",
        "v2_reward_scales.lateral_drive_soft_penalty",
        "v2_reward_scales.lateral_speed_deficit_penalty",
        "v2_reward_scales.lateral_speed_bonus",
        "v2_reward_scales.height_maintain",
        "v2_reward_scales.height_low_penalty",
        "v2_reward_scales.leg_moving",
        "v2_reward_scales.stall_penalty",
        "v2_reward_scales.energy_per_distance",
    ),
    3: (
        "v2_reward_scales.velocity_tracking",
        "v2_reward_scales.mode_specialization",
        "v2_reward_scales.axis_suppression",
        "v2_reward_scales.diag_sign_bonus",
        "v2_reward_scales.diag_wrong_sign_penalty",
        "v2_reward_scales.diag_speed_bonus",
        "v2_reward_scales.height_maintain",
        "v2_reward_scales.height_low_penalty",
        "v2_reward_scales.leg_moving",
        "v2_reward_scales.stall_penalty",
        "v2_reward_scales.energy_per_distance",
    ),
    4: (
        "v2_reward_scales.velocity_tracking",
        "v2_reward_scales.mode_specialization",
        "v2_reward_scales.axis_suppression",
        "v2_reward_scales.yaw_mode_track_bonus",
        "v2_reward_scales.yaw_spin_bonus",
        "v2_reward_scales.yaw_roll_pitch_penalty",
        "v2_reward_scales.yaw_height_penalty",
        "v2_reward_scales.yaw_slip_penalty",
        "v2_reward_scales.yaw_cheat_penalty",
        "v2_reward_scales.height_maintain",
        "v2_reward_scales.height_low_penalty",
        "v2_reward_scales.leg_moving",
        "v2_reward_scales.stall_penalty",
        "v2_reward_scales.energy_per_distance",
    ),
}
DIRECT_REWARD_KEYS_BY_STAGE[5] = tuple(
    dict.fromkeys(key for stage in range(1, 5) for key in DIRECT_REWARD_KEYS_BY_STAGE[stage])
)


def eligible_reward_keys(task: str, stage: int) -> tuple[str, ...]:
    if task == FORWARD_FAST_TASK and stage == 1:
        return FORWARD_FAST_REWARD_KEYS
    if task == DIRECT_TASK and stage in DIRECT_REWARD_KEYS_BY_STAGE:
        return DIRECT_REWARD_KEYS_BY_STAGE[stage]
    raise AutopilotValidationError(f"Unsupported task/stage combination: {task!r}/stage{stage}")


def _split_interval(interval: tuple[float, float], gait: str) -> tuple[float, float]:
    low, high = interval
    if low == 0.0 and high == 0.0:
        return interval
    if high <= 0.0:
        min_magnitude, max_magnitude = abs(high), abs(low)
        midpoint = (min_magnitude + max_magnitude) / 2.0
        if gait == "walk":
            return (-midpoint, -min_magnitude)
        return (-max_magnitude, -midpoint)
    if low >= 0.0:
        midpoint = (low + high) / 2.0
        if gait == "walk":
            return (low, midpoint)
        return (midpoint, high)
    raise AutopilotValidationError("Command preset intervals may not cross zero")


def compile_command_envelope(task: str, stage: int, gait: str) -> CommandIntervals:
    """Compile walk/run into the lower/upper half of the checked-in stage ranges."""

    if gait not in SUPPORTED_GAITS:
        raise AutopilotValidationError(f"gait must be one of: {', '.join(SUPPORTED_GAITS)}")
    try:
        preset = STAGE_COMMAND_PRESETS[task][stage]
    except KeyError as exc:
        raise AutopilotValidationError(f"Unsupported task/stage combination: {task!r}/stage{stage}") from exc
    return {
        axis: tuple(_split_interval(interval, gait) for interval in intervals)
        for axis, intervals in preset.items()
    }


def command_envelope_to_dict(envelope: Mapping[str, Sequence[Sequence[float]]]) -> dict[str, list[list[float]]]:
    return {
        axis: [[float(interval[0]), float(interval[1])] for interval in envelope[axis]]
        for axis in ("vx", "vy", "wz")
    }


def _interval_samples(interval: Sequence[float], count: int = 2) -> tuple[float, ...]:
    low, high = float(interval[0]), float(interval[1])
    if abs(high - low) < 1e-12 or count <= 1:
        return ((low + high) / 2.0,)
    if count == 2:
        return (low, high)
    return (low, (low + high) / 2.0, high)


def compile_command_profile(
    task: str,
    stage: int,
    gait: str,
    directions: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Return the exact deterministic commands used to prove a goal envelope."""

    envelope = compile_command_envelope(task, stage, gait)
    selected_directions = tuple(directions or STAGE_DIRECTIONS[stage])
    if not selected_directions or any(direction not in STAGE_DIRECTIONS[stage] for direction in selected_directions):
        raise AutopilotValidationError("command profile directions are invalid for the selected stage")
    vx_nonzero = [value for value in envelope["vx"] if value != (0.0, 0.0)]
    vy_negative = [value for value in envelope["vy"] if value[1] < 0.0]
    vy_positive = [value for value in envelope["vy"] if value[0] > 0.0]
    wz_negative = [value for value in envelope["wz"] if value[1] < 0.0]
    wz_positive = [value for value in envelope["wz"] if value[0] > 0.0]

    commands: list[dict[str, Any]] = []

    def add(name: str, skill: str, vx: float, vy: float, wz: float) -> None:
        commands.append(
            {
                "name": name,
                "skill": skill,
                "vx": float(vx),
                "vy": float(vy),
                "wz": float(wz),
            }
        )

    for direction in selected_directions:
        if direction == "forward":
            interval = vx_nonzero[0]
            for index, vx in enumerate(_interval_samples(interval, 3), 1):
                add(f"forward_{index}", "forward", vx, 0.0, 0.0)
        elif direction in {"left", "right"}:
            intervals = vy_positive if direction == "left" else vy_negative
            # Stage 5 contains both lateral and diagonal vy bands. The outer
            # band is the pure lateral command contract.
            interval = max(intervals, key=lambda item: max(abs(item[0]), abs(item[1])))
            for index, vy in enumerate(_interval_samples(interval), 1):
                add(f"{direction}_{index}", "lateral", 0.0, vy, 0.0)
        elif direction in {"forward_left", "forward_right"}:
            intervals = vy_positive if direction == "forward_left" else vy_negative
            interval_y = min(intervals, key=lambda item: max(abs(item[0]), abs(item[1])))
            interval_x = vx_nonzero[-1]
            xs = _interval_samples(interval_x)
            ys = _interval_samples(interval_y)
            for index, (vx, vy) in enumerate(zip(xs, ys), 1):
                add(f"{direction}_{index}", "diagonal", vx, vy, 0.0)
        elif direction in {"yaw_ccw", "yaw_cw"}:
            intervals = wz_positive if direction == "yaw_ccw" else wz_negative
            interval = intervals[0]
            for index, wz in enumerate(_interval_samples(interval), 1):
                add(f"{direction}_{index}", "yaw", 0.0, 0.0, wz)
    if not commands:
        raise AutopilotValidationError("compiled command profile is empty")
    return {
        "schema_version": "redrhex.autopilot.command-profile.v1",
        "task": task,
        "stage": stage,
        "evaluation_profile": f"stage{stage}",
        "gait": gait,
        "directions": list(selected_directions),
        "command_envelope": command_envelope_to_dict(envelope),
        "commands": commands,
    }


def _command_envelope(value: Any, task: str, stage: int) -> CommandIntervals:
    obj = _mapping(value, "goal.command_envelope")
    _strict_keys(obj, required=("vx", "vy", "wz"), name="goal.command_envelope")
    preset = STAGE_COMMAND_PRESETS[task][stage]
    result: CommandIntervals = {}
    for axis in ("vx", "vy", "wz"):
        raw_intervals = obj[axis]
        if not isinstance(raw_intervals, (list, tuple)) or not raw_intervals:
            raise AutopilotValidationError(f"goal.command_envelope.{axis} must contain intervals")
        parsed: list[tuple[float, float]] = []
        for index, raw in enumerate(raw_intervals):
            if not isinstance(raw, (list, tuple)) or len(raw) != 2:
                raise AutopilotValidationError(
                    f"goal.command_envelope.{axis}[{index}] must be [minimum, maximum]"
                )
            minimum = _finite(raw[0], f"goal.command_envelope.{axis}[{index}][0]")
            maximum = _finite(raw[1], f"goal.command_envelope.{axis}[{index}][1]")
            if minimum > maximum:
                raise AutopilotValidationError(f"goal.command_envelope.{axis}[{index}] is reversed")
            if not any(base_min <= minimum and maximum <= base_max for base_min, base_max in preset[axis]):
                raise AutopilotValidationError(
                    f"goal.command_envelope.{axis}[{index}] is outside the stage command preset"
                )
            parsed.append((minimum, maximum))
        result[axis] = tuple(parsed)
    return result


DEFAULT_SKILL_GATES = {
    "min_command_pass_ratio": 0.70,
    "min_skill_pass_ratio": 0.60,
    "max_fall_rate": 0.20,
    "min_tracking_quality": 0.0,
    "min_stability_quality": 0.0,
    "min_direction_sign_ratio": 0.70,
    "max_linear_leak": 0.18,
    "max_yaw_leak": 0.35,
    "max_energy_per_distance": 500.0,
}


def _skill_gates(value: Any) -> dict[str, float]:
    obj = _mapping(value, "goal.skill_gates")
    _strict_keys(obj, required=DEFAULT_SKILL_GATES, name="goal.skill_gates")
    gates = {key: _finite(obj[key], f"goal.skill_gates.{key}") for key in DEFAULT_SKILL_GATES}
    for key in (
        "min_command_pass_ratio",
        "min_skill_pass_ratio",
        "min_tracking_quality",
        "min_stability_quality",
        "min_direction_sign_ratio",
    ):
        if not 0.0 <= gates[key] <= 1.0:
            raise AutopilotValidationError(f"goal.skill_gates.{key} must be between 0 and 1")
    for key in ("max_fall_rate", "max_linear_leak", "max_yaw_leak"):
        if gates[key] < 0.0:
            raise AutopilotValidationError(f"goal.skill_gates.{key} must be non-negative")
    if gates["max_energy_per_distance"] <= 0.0:
        raise AutopilotValidationError("goal.skill_gates.max_energy_per_distance must be positive")

    # The checked-in evaluator defaults are the loosest V1 safety policy. A
    # campaign may only make minimums larger or maximums smaller.
    for key in (
        "min_command_pass_ratio",
        "min_skill_pass_ratio",
        "min_tracking_quality",
        "min_stability_quality",
        "min_direction_sign_ratio",
    ):
        if gates[key] < DEFAULT_SKILL_GATES[key]:
            raise AutopilotValidationError(f"goal.skill_gates.{key} may not relax the default")
    for key in ("max_fall_rate", "max_linear_leak", "max_yaw_leak", "max_energy_per_distance"):
        if gates[key] > DEFAULT_SKILL_GATES[key]:
            raise AutopilotValidationError(f"goal.skill_gates.{key} may not relax the default")
    return gates


@dataclass(frozen=True)
class CampaignBudgetV1:
    max_training_trials: int = MAX_TRAINING_TRIALS
    max_gpu_hours: float = MAX_GPU_HOURS

    def __post_init__(self) -> None:
        trials = _integer(self.max_training_trials, "goal.budget.max_training_trials", minimum=1)
        hours = _finite(self.max_gpu_hours, "goal.budget.max_gpu_hours")
        if trials > MAX_TRAINING_TRIALS:
            raise AutopilotValidationError(f"training trial budget may not exceed {MAX_TRAINING_TRIALS}")
        if hours <= 0.0 or hours > MAX_GPU_HOURS:
            raise AutopilotValidationError(f"GPU-hour budget must be in (0, {MAX_GPU_HOURS:g}]")
        object.__setattr__(self, "max_training_trials", trials)
        object.__setattr__(self, "max_gpu_hours", hours)

    @classmethod
    def from_dict(cls, value: Any) -> "CampaignBudgetV1":
        obj = _mapping(value, "goal.budget")
        _strict_keys(obj, required=("max_training_trials", "max_gpu_hours"), name="goal.budget")
        return cls(obj["max_training_trials"], obj["max_gpu_hours"])

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class GoalSpecV1:
    description: str
    task: str
    stage: int
    evaluation_profile: str
    gait: str
    directions: tuple[str, ...]
    command_envelope: CommandIntervals
    skill_gates: dict[str, float]
    initialization_mode: str
    baseline_run_id: str | None
    baseline_checkpoint_iteration: int | None
    checkpoint_sha256: str | None
    physics_profile_sha256: str
    spring_profile_sha256: str
    code_sha256: str
    config_sha256: str
    command_profile_sha256: str
    training_seeds: tuple[int, ...]
    per_trial_iteration_cap: int
    budget: CampaignBudgetV1 = field(default_factory=CampaignBudgetV1)
    schema_version: str = GOAL_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _schema(self.schema_version, GOAL_SCHEMA_VERSION, "goal")
        object.__setattr__(self, "description", _text(self.description, "goal.description", maximum=2000))
        if self.task not in SUPPORTED_TASKS:
            raise AutopilotValidationError(f"goal.task must be one of: {', '.join(SUPPORTED_TASKS)}")
        stage = _integer(self.stage, "goal.stage", minimum=1)
        if stage not in STAGE_COMMAND_PRESETS[self.task]:
            raise AutopilotValidationError(f"Unsupported task/stage combination: {self.task!r}/stage{stage}")
        object.__setattr__(self, "stage", stage)
        expected_profile = f"stage{stage}"
        if self.evaluation_profile != expected_profile:
            raise AutopilotValidationError(f"goal.evaluation_profile must be {expected_profile!r}")
        if self.gait not in SUPPORTED_GAITS:
            raise AutopilotValidationError(f"goal.gait must be one of: {', '.join(SUPPORTED_GAITS)}")
        if not self.directions or len(set(self.directions)) != len(self.directions):
            raise AutopilotValidationError("goal.directions must contain unique directions")
        allowed_directions = set(STAGE_DIRECTIONS[stage])
        if any(direction not in allowed_directions for direction in self.directions):
            raise AutopilotValidationError("goal.directions contains a direction outside the selected stage")
        object.__setattr__(self, "directions", tuple(self.directions))
        object.__setattr__(self, "command_envelope", _command_envelope(self.command_envelope, self.task, stage))
        object.__setattr__(self, "skill_gates", _skill_gates(self.skill_gates))
        if self.initialization_mode not in ("fresh", "policy_only"):
            raise AutopilotValidationError("goal.initialization_mode must be 'fresh' or 'policy_only'")
        baseline = _optional_text(self.baseline_run_id, "goal.baseline_run_id", maximum=512)
        checkpoint_iteration = (
            None
            if self.baseline_checkpoint_iteration is None
            else _integer(
                self.baseline_checkpoint_iteration,
                "goal.baseline_checkpoint_iteration",
                minimum=0,
            )
        )
        checkpoint = _sha256(self.checkpoint_sha256, "goal.checkpoint_sha256", optional=True)
        if self.initialization_mode == "policy_only" and (
            baseline is None or checkpoint_iteration is None or checkpoint is None
        ):
            raise AutopilotValidationError(
                "policy_only initialization requires baseline_run_id, "
                "baseline_checkpoint_iteration, and checkpoint_sha256"
            )
        if self.initialization_mode == "fresh" and any(
            value is not None for value in (baseline, checkpoint_iteration, checkpoint)
        ):
            raise AutopilotValidationError(
                "fresh initialization may not specify baseline checkpoint identity"
            )
        object.__setattr__(self, "baseline_run_id", baseline)
        object.__setattr__(self, "baseline_checkpoint_iteration", checkpoint_iteration)
        object.__setattr__(self, "checkpoint_sha256", checkpoint)
        for name in (
            "physics_profile_sha256",
            "spring_profile_sha256",
            "code_sha256",
            "config_sha256",
            "command_profile_sha256",
        ):
            object.__setattr__(self, name, _sha256(getattr(self, name), f"goal.{name}"))
        seeds = tuple(_integer(seed, f"goal.training_seeds[{index}]", minimum=0) for index, seed in enumerate(self.training_seeds))
        if seeds != DEFAULT_TRAINING_SEEDS:
            raise AutopilotValidationError(f"goal.training_seeds must be {list(DEFAULT_TRAINING_SEEDS)} in V1")
        object.__setattr__(self, "training_seeds", seeds)
        object.__setattr__(
            self,
            "per_trial_iteration_cap",
            _integer(self.per_trial_iteration_cap, "goal.per_trial_iteration_cap", minimum=1),
        )
        if not isinstance(self.budget, CampaignBudgetV1):
            raise AutopilotValidationError("goal.budget must be CampaignBudgetV1")

    @classmethod
    def from_dict(cls, value: Any) -> "GoalSpecV1":
        obj = _mapping(value, "goal")
        required = (
            "schema_version",
            "description",
            "task",
            "stage",
            "evaluation_profile",
            "gait",
            "directions",
            "command_envelope",
            "skill_gates",
            "initialization_mode",
            "baseline_run_id",
            "baseline_checkpoint_iteration",
            "checkpoint_sha256",
            "physics_profile_sha256",
            "spring_profile_sha256",
            "code_sha256",
            "config_sha256",
            "command_profile_sha256",
            "training_seeds",
            "per_trial_iteration_cap",
            "budget",
        )
        _strict_keys(obj, required=required, name="goal")
        if not isinstance(obj["directions"], (list, tuple)):
            raise AutopilotValidationError("goal.directions must be an array")
        if not isinstance(obj["training_seeds"], (list, tuple)):
            raise AutopilotValidationError("goal.training_seeds must be an array")
        return cls(
            schema_version=obj["schema_version"],
            description=obj["description"],
            task=obj["task"],
            stage=obj["stage"],
            evaluation_profile=obj["evaluation_profile"],
            gait=obj["gait"],
            directions=tuple(obj["directions"]),
            command_envelope=dict(obj["command_envelope"]),
            skill_gates=dict(obj["skill_gates"]),
            initialization_mode=obj["initialization_mode"],
            baseline_run_id=obj["baseline_run_id"],
            baseline_checkpoint_iteration=obj["baseline_checkpoint_iteration"],
            checkpoint_sha256=obj["checkpoint_sha256"],
            physics_profile_sha256=obj["physics_profile_sha256"],
            spring_profile_sha256=obj["spring_profile_sha256"],
            code_sha256=obj["code_sha256"],
            config_sha256=obj["config_sha256"],
            command_profile_sha256=obj["command_profile_sha256"],
            training_seeds=tuple(obj["training_seeds"]),
            per_trial_iteration_cap=obj["per_trial_iteration_cap"],
            budget=CampaignBudgetV1.from_dict(obj["budget"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "description": self.description,
            "task": self.task,
            "stage": self.stage,
            "evaluation_profile": self.evaluation_profile,
            "gait": self.gait,
            "directions": list(self.directions),
            "command_envelope": command_envelope_to_dict(self.command_envelope),
            "skill_gates": dict(self.skill_gates),
            "initialization_mode": self.initialization_mode,
            "baseline_run_id": self.baseline_run_id,
            "baseline_checkpoint_iteration": self.baseline_checkpoint_iteration,
            "checkpoint_sha256": self.checkpoint_sha256,
            "physics_profile_sha256": self.physics_profile_sha256,
            "spring_profile_sha256": self.spring_profile_sha256,
            "code_sha256": self.code_sha256,
            "config_sha256": self.config_sha256,
            "command_profile_sha256": self.command_profile_sha256,
            "training_seeds": list(self.training_seeds),
            "per_trial_iteration_cap": self.per_trial_iteration_cap,
            "budget": self.budget.to_dict(),
        }


def compile_goal_spec(
    *,
    description: str,
    task: str,
    stage: int,
    gait: str,
    per_trial_iteration_cap: int,
    physics_profile_sha256: str,
    spring_profile_sha256: str,
    code_sha256: str,
    config_sha256: str,
    initialization_mode: str = "fresh",
    baseline_run_id: str | None = None,
    baseline_checkpoint_iteration: int | None = None,
    checkpoint_sha256: str | None = None,
    directions: Sequence[str] | None = None,
    skill_gates: Mapping[str, float] | None = None,
    budget: CampaignBudgetV1 | None = None,
) -> GoalSpecV1:
    envelope = compile_command_envelope(task, stage, gait)
    resolved_directions = tuple(
        STAGE_DIRECTIONS[stage] if directions is None else directions
    )
    command_profile_sha256 = sha256_json(
        compile_command_profile(task, stage, gait, resolved_directions)
    )
    return GoalSpecV1(
        description=description,
        task=task,
        stage=stage,
        evaluation_profile=f"stage{stage}",
        gait=gait,
        directions=resolved_directions,
        command_envelope=envelope,
        skill_gates=dict(DEFAULT_SKILL_GATES if skill_gates is None else skill_gates),
        initialization_mode=initialization_mode,
        baseline_run_id=baseline_run_id,
        baseline_checkpoint_iteration=baseline_checkpoint_iteration,
        checkpoint_sha256=checkpoint_sha256,
        physics_profile_sha256=physics_profile_sha256,
        spring_profile_sha256=spring_profile_sha256,
        code_sha256=code_sha256,
        config_sha256=config_sha256,
        command_profile_sha256=command_profile_sha256,
        training_seeds=DEFAULT_TRAINING_SEEDS,
        per_trial_iteration_cap=per_trial_iteration_cap,
        budget=CampaignBudgetV1() if budget is None else budget,
    )


@dataclass(frozen=True)
class RewardCatalogEntryV1:
    key: str
    description: str
    tasks: tuple[str, ...]
    stages: tuple[int, ...]
    start_value: float
    minimum: float
    maximum: float
    sign: str
    enabled: bool = True
    mutability_class: str = "shaping"
    schema_version: str = REWARD_CATALOG_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _schema(self.schema_version, REWARD_CATALOG_SCHEMA_VERSION, "reward_catalog_entry")
        if not isinstance(self.key, str) or not REWARD_KEY_RE.fullmatch(self.key):
            raise AutopilotValidationError("reward catalog key must be a v2_reward_scales.* key")
        object.__setattr__(self, "description", _text(self.description, "reward_catalog_entry.description", maximum=1000))
        tasks = tuple(self.tasks)
        if not tasks or any(task not in SUPPORTED_TASKS for task in tasks):
            raise AutopilotValidationError("reward catalog tasks must be supported and non-empty")
        stages = tuple(_integer(stage, "reward_catalog_entry.stage", minimum=1) for stage in self.stages)
        if not stages:
            raise AutopilotValidationError("reward catalog stages must be non-empty")
        for task in tasks:
            if any(stage not in STAGE_COMMAND_PRESETS[task] for stage in stages):
                raise AutopilotValidationError("reward catalog contains an unsupported task/stage")
            if any(self.key not in eligible_reward_keys(task, stage) for stage in stages):
                raise AutopilotValidationError("reward catalog key is not allowlisted for every task/stage")
        object.__setattr__(self, "tasks", tasks)
        object.__setattr__(self, "stages", stages)
        start = _finite(self.start_value, "reward_catalog_entry.start_value")
        minimum = _finite(self.minimum, "reward_catalog_entry.minimum")
        maximum = _finite(self.maximum, "reward_catalog_entry.maximum")
        if start == 0.0:
            raise AutopilotValidationError("zero-valued reward terms are immutable and may not enter the catalog")
        expected_sign = "positive" if start > 0.0 else "negative"
        if self.sign != expected_sign:
            raise AutopilotValidationError(f"reward catalog sign must be {expected_sign!r}")
        if minimum > start or maximum < start or minimum > maximum:
            raise AutopilotValidationError("reward bounds must contain the campaign-start value")
        hard_low, hard_high = sorted((start * 0.8, start * 1.2))
        tolerance = 1e-12
        if minimum < hard_low - tolerance or maximum > hard_high + tolerance:
            raise AutopilotValidationError("reward bounds may not exceed 80-120% of the campaign-start value")
        if minimum == 0.0 or maximum == 0.0 or minimum * start <= 0.0 or maximum * start <= 0.0:
            raise AutopilotValidationError("reward bounds may not change sign or reach zero")
        if not isinstance(self.enabled, bool):
            raise AutopilotValidationError("reward catalog enabled must be a boolean")
        if self.mutability_class != "shaping":
            raise AutopilotValidationError("only shaping rewards are mutable in V1")
        object.__setattr__(self, "start_value", start)
        object.__setattr__(self, "minimum", minimum)
        object.__setattr__(self, "maximum", maximum)

    @classmethod
    def from_dict(cls, value: Any) -> "RewardCatalogEntryV1":
        obj = _mapping(value, "reward_catalog_entry")
        required = (
            "schema_version", "key", "description", "tasks", "stages", "start_value", "minimum", "maximum",
            "sign", "enabled", "mutability_class",
        )
        _strict_keys(obj, required=required, name="reward_catalog_entry")
        if not isinstance(obj["tasks"], (list, tuple)) or not isinstance(obj["stages"], (list, tuple)):
            raise AutopilotValidationError("reward catalog tasks and stages must be arrays")
        return cls(
            schema_version=obj["schema_version"], key=obj["key"], description=obj["description"],
            tasks=tuple(obj["tasks"]), stages=tuple(obj["stages"]), start_value=obj["start_value"],
            minimum=obj["minimum"], maximum=obj["maximum"], sign=obj["sign"], enabled=obj["enabled"],
            mutability_class=obj["mutability_class"],
        )

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["tasks"] = list(self.tasks)
        result["stages"] = list(self.stages)
        return result


def build_reward_catalog(
    task: str,
    stage: int,
    reward_values: Mapping[str, Any],
    *,
    enabled_keys: Sequence[str] | None = None,
    narrowed_bounds: Mapping[str, Sequence[Any]] | None = None,
) -> tuple[RewardCatalogEntryV1, ...]:
    eligible = eligible_reward_keys(task, stage)
    selected = eligible if enabled_keys is None else tuple(enabled_keys)
    if len(set(selected)) != len(selected):
        raise AutopilotValidationError("Enabled reward keys must be unique")
    unknown = sorted(set(selected) - set(eligible))
    if unknown:
        raise AutopilotValidationError(f"Unknown or immutable reward keys: {', '.join(unknown)}")
    bounds_by_key = dict(narrowed_bounds or {})
    unknown_bounds = sorted(set(bounds_by_key) - set(selected))
    if unknown_bounds:
        raise AutopilotValidationError(
            "Reward bounds may only narrow selected keys: " + ", ".join(unknown_bounds)
        )
    entries = []
    for key in selected:
        if key not in reward_values:
            raise AutopilotValidationError(f"Missing campaign-start reward value for {key}")
        start = _finite(reward_values[key], f"reward_values.{key}")
        if start == 0.0:
            continue
        low, high = sorted((start * 0.8, start * 1.2))
        if key in bounds_by_key:
            requested = bounds_by_key[key]
            if (
                isinstance(requested, (str, bytes))
                or not isinstance(requested, Sequence)
                or len(requested) != 2
            ):
                raise AutopilotValidationError(f"reward_bounds.{key} must be [minimum, maximum]")
            low = _finite(requested[0], f"reward_bounds.{key}[0]")
            high = _finite(requested[1], f"reward_bounds.{key}[1]")
        entries.append(
            RewardCatalogEntryV1(
                key=key,
                description=f"Bounded shaping weight {key}",
                tasks=(task,),
                stages=(stage,),
                start_value=start,
                minimum=low,
                maximum=high,
                sign="positive" if start > 0.0 else "negative",
            )
        )
    if not entries:
        raise AutopilotValidationError("No nonzero eligible reward weights were selected")
    return tuple(entries)


DECISION_ACTIONS = ("propose_candidate", "pause", "request_patch_handoff")


@dataclass(frozen=True)
class AgentDecisionV1:
    campaign_id: str
    campaign_revision: int
    evidence_ids: tuple[str, ...]
    hypothesis: str
    action: str
    reward_key: str | None = None
    proposed_value: float | None = None
    expected_metric_effect: str | None = None
    rationale: str = ""
    schema_version: str = DECISION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _schema(self.schema_version, DECISION_SCHEMA_VERSION, "decision")
        object.__setattr__(self, "campaign_id", _text(self.campaign_id, "decision.campaign_id", maximum=128))
        object.__setattr__(self, "campaign_revision", _integer(self.campaign_revision, "decision.campaign_revision", minimum=0))
        if not isinstance(self.evidence_ids, tuple):
            object.__setattr__(self, "evidence_ids", tuple(self.evidence_ids))
        evidence = tuple(_text(value, "decision.evidence_id", maximum=256) for value in self.evidence_ids)
        if len(set(evidence)) != len(evidence):
            raise AutopilotValidationError("decision.evidence_ids must be unique")
        object.__setattr__(self, "evidence_ids", evidence)
        object.__setattr__(self, "hypothesis", _text(self.hypothesis, "decision.hypothesis", maximum=2000))
        object.__setattr__(self, "rationale", _text(self.rationale, "decision.rationale", maximum=4000))
        if self.action not in DECISION_ACTIONS:
            raise AutopilotValidationError(f"decision.action must be one of: {', '.join(DECISION_ACTIONS)}")
        if self.action == "propose_candidate":
            if not isinstance(self.reward_key, str) or not REWARD_KEY_RE.fullmatch(self.reward_key):
                raise AutopilotValidationError("candidate decisions require one v2_reward_scales.* reward_key")
            object.__setattr__(self, "proposed_value", _finite(self.proposed_value, "decision.proposed_value"))
            object.__setattr__(
                self,
                "expected_metric_effect",
                _text(self.expected_metric_effect, "decision.expected_metric_effect", maximum=1000),
            )
        elif self.reward_key is not None or self.proposed_value is not None or self.expected_metric_effect is not None:
            raise AutopilotValidationError("non-candidate decisions may not contain reward mutation fields")

    @classmethod
    def from_dict(cls, value: Any) -> "AgentDecisionV1":
        obj = _mapping(value, "decision")
        required = (
            "schema_version", "campaign_id", "campaign_revision", "evidence_ids", "hypothesis", "action",
            "reward_key", "proposed_value", "expected_metric_effect", "rationale",
        )
        _strict_keys(obj, required=required, name="decision")
        if not isinstance(obj["evidence_ids"], (list, tuple)):
            raise AutopilotValidationError("decision.evidence_ids must be an array")
        return cls(
            schema_version=obj["schema_version"], campaign_id=obj["campaign_id"],
            campaign_revision=obj["campaign_revision"], evidence_ids=tuple(obj["evidence_ids"]),
            hypothesis=obj["hypothesis"], action=obj["action"], reward_key=obj["reward_key"],
            proposed_value=obj["proposed_value"], expected_metric_effect=obj["expected_metric_effect"],
            rationale=obj["rationale"],
        )

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["evidence_ids"] = list(self.evidence_ids)
        return result


def validate_candidate_decision(
    decision: AgentDecisionV1,
    goal: GoalSpecV1,
    catalog: Sequence[RewardCatalogEntryV1],
    leader_reward_values: Mapping[str, Any],
) -> dict[str, float]:
    """Return a one-key candidate profile after all deterministic checks pass."""

    if decision.action != "propose_candidate":
        raise AutopilotValidationError("decision is not a candidate proposal")
    entries = {entry.key: entry for entry in catalog}
    entry = entries.get(str(decision.reward_key))
    if entry is None or not entry.enabled:
        raise AutopilotValidationError("decision reward key is unknown, immutable, or disabled")
    if goal.task not in entry.tasks or goal.stage not in entry.stages:
        raise AutopilotValidationError("decision reward key is incompatible with the campaign task/stage")
    current: dict[str, float] = {}
    unknown = sorted(set(leader_reward_values) - set(entries))
    if unknown:
        raise AutopilotValidationError(f"leader profile contains unknown reward keys: {', '.join(unknown)}")
    for key, value in leader_reward_values.items():
        current[key] = _finite(value, f"leader_reward_values.{key}")
    if entry.key not in current:
        raise AutopilotValidationError(f"leader profile is missing {entry.key}")
    value = _finite(decision.proposed_value, "decision.proposed_value")
    if not entry.minimum <= value <= entry.maximum:
        raise AutopilotValidationError("decision proposed value is outside the approved reward bounds")
    if value == 0.0 or value * entry.start_value <= 0.0:
        raise AutopilotValidationError("decision proposed value may not reach zero or change sign")
    if value == current[entry.key]:
        raise AutopilotValidationError("decision proposed value must change the selected reward")
    current[entry.key] = value
    return current


def reward_lattice_values(entry: RewardCatalogEntryV1) -> tuple[float, ...]:
    """Return the finite V1 move lattice clipped to human-approved bounds."""

    values: list[float] = []
    for multiplier in REWARD_MOVE_MULTIPLIERS:
        raw = entry.start_value * multiplier
        value = min(entry.maximum, max(entry.minimum, raw))
        if math.isclose(value, entry.minimum, rel_tol=1e-12, abs_tol=1e-15):
            value = entry.minimum
        elif math.isclose(value, entry.maximum, rel_tol=1e-12, abs_tol=1e-15):
            value = entry.maximum
        elif math.isclose(value, entry.start_value, rel_tol=1e-12, abs_tol=1e-15):
            value = entry.start_value
        else:
            value = float(format(value, ".15g"))
        if not any(
            math.isclose(value, prior, rel_tol=1e-12, abs_tol=1e-15)
            for prior in values
        ):
            values.append(value)
    return tuple(values)


def reward_move_lattice(
    catalog: Sequence[RewardCatalogEntryV1],
    leader_reward_values: Mapping[str, Any],
    decisions: Sequence[Mapping[str, Any]] = (),
) -> dict[str, list[dict[str, Any]]]:
    """Partition deterministic one-weight lattice moves into remaining and attempted."""

    entries = tuple(catalog)
    entry_by_key = {entry.key: entry for entry in entries}
    if len(entry_by_key) != len(entries):
        raise AutopilotValidationError("reward catalog keys must be unique")
    unknown = sorted(set(leader_reward_values) - set(entry_by_key))
    missing = sorted(set(entry_by_key) - set(leader_reward_values))
    if unknown:
        raise AutopilotValidationError(
            f"leader profile contains unknown reward keys: {', '.join(unknown)}"
        )
    if missing:
        raise AutopilotValidationError(
            f"leader profile is missing reward keys: {', '.join(missing)}"
        )
    current = {
        key: _finite(value, f"leader_reward_values.{key}")
        for key, value in leader_reward_values.items()
    }

    attempted: list[dict[str, Any]] = []
    attempted_points: list[tuple[str, float]] = []
    for decision in decisions:
        if decision.get("action") != "propose_candidate":
            continue
        key = decision.get("reward_key")
        if key not in entry_by_key:
            continue
        value = _finite(decision.get("proposed_value"), "decision.proposed_value")
        attempted_points.append((str(key), value))
        move = {"reward_key": str(key), "proposed_value": value}
        if decision.get("id"):
            move["decision_id"] = str(decision["id"])
        attempted.append(move)

    remaining: list[dict[str, Any]] = []
    for entry in entries:
        if not entry.enabled:
            continue
        current_value = current[entry.key]
        for proposed_value in reward_lattice_values(entry):
            if math.isclose(
                proposed_value, current_value, rel_tol=1e-12, abs_tol=1e-15
            ):
                continue
            if any(
                attempted_key == entry.key
                and math.isclose(
                    proposed_value,
                    attempted_value,
                    rel_tol=1e-12,
                    abs_tol=1e-15,
                )
                for attempted_key, attempted_value in attempted_points
            ):
                continue
            remaining.append(
                {
                    "reward_key": entry.key,
                    "current_value": current_value,
                    "proposed_value": proposed_value,
                    "delta_from_leader": proposed_value - current_value,
                    "delta_from_campaign_start": proposed_value - entry.start_value,
                }
            )
    return {"remaining": remaining, "attempted": attempted}


@dataclass(frozen=True)
class EvaluationReportV1:
    id: str
    trial_id: str
    checkpoint_sha256: str
    config_sha256: str
    reward_profile_sha256: str
    physics_profile_sha256: str
    spring_profile_sha256: str
    command_profile_sha256: str
    seed: int
    evaluation_profile: str
    strict_checkpoint_load: bool
    episode_artifact_sha256: str
    command_metrics: tuple[dict[str, Any], ...]
    episode_metrics: tuple[dict[str, Any], ...]
    artifact_ids: tuple[str, ...]
    hard_gates: dict[str, bool] = field(default_factory=dict)
    gate_margins: dict[str, float] = field(default_factory=dict)
    ranking: dict[str, Any] = field(default_factory=dict)
    failure_reason: str | None = None
    schema_version: str = EVALUATION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _schema(self.schema_version, EVALUATION_SCHEMA_VERSION, "evaluation")
        object.__setattr__(self, "id", _text(self.id, "evaluation.id", maximum=128))
        object.__setattr__(self, "trial_id", _text(self.trial_id, "evaluation.trial_id", maximum=128))
        for name in (
            "checkpoint_sha256", "config_sha256", "reward_profile_sha256", "physics_profile_sha256",
            "spring_profile_sha256", "command_profile_sha256",
        ):
            object.__setattr__(self, name, _sha256(getattr(self, name), f"evaluation.{name}"))
        object.__setattr__(self, "seed", _integer(self.seed, "evaluation.seed", minimum=0))
        if self.evaluation_profile not in {"stage1", "stage2", "stage3", "stage4", "stage5"}:
            raise AutopilotValidationError("evaluation.evaluation_profile must be stage1-stage5")
        if self.strict_checkpoint_load is not True:
            raise AutopilotValidationError("evaluation.strict_checkpoint_load must be true")
        object.__setattr__(
            self,
            "episode_artifact_sha256",
            _sha256(self.episode_artifact_sha256, "evaluation.episode_artifact_sha256"),
        )
        if not isinstance(self.command_metrics, tuple):
            object.__setattr__(self, "command_metrics", tuple(self.command_metrics))
        if not self.command_metrics:
            raise AutopilotValidationError("evaluation.command_metrics must be non-empty")
        names = set()
        required_command = (
            "name", "skill", "command", "accept_pass", "tracking_quality", "stability_quality", "fall_rate",
            "energy_per_distance", "direction_sign_ratio", "linear_leak", "yaw_leak",
        )
        parsed_commands = []
        for index, raw in enumerate(self.command_metrics):
            item = _mapping(raw, f"evaluation.command_metrics[{index}]")
            _strict_keys(item, required=required_command, name=f"evaluation.command_metrics[{index}]")
            name = _text(item["name"], f"evaluation.command_metrics[{index}].name", maximum=128)
            if name in names:
                raise AutopilotValidationError("evaluation command names must be unique")
            names.add(name)
            skill = item["skill"]
            if skill not in ("forward", "lateral", "diagonal", "yaw"):
                raise AutopilotValidationError("evaluation command skill is invalid")
            command = _mapping(item["command"], f"evaluation.command_metrics[{index}].command")
            _strict_keys(command, required=("vx", "vy", "wz"), name=f"evaluation.command_metrics[{index}].command")
            if not isinstance(item["accept_pass"], bool):
                raise AutopilotValidationError("evaluation command accept_pass must be a boolean")
            parsed = {"name": name, "skill": skill, "accept_pass": item["accept_pass"]}
            parsed["command"] = {axis: _finite(command[axis], f"evaluation.command.{axis}") for axis in ("vx", "vy", "wz")}
            for key in required_command[4:]:
                parsed[key] = _finite(item[key], f"evaluation.command_metrics[{index}].{key}")
            for key in ("tracking_quality", "stability_quality", "direction_sign_ratio"):
                if not 0.0 <= parsed[key] <= 1.0:
                    raise AutopilotValidationError(f"evaluation command {key} must be between 0 and 1")
            for key in ("fall_rate", "energy_per_distance", "linear_leak", "yaw_leak"):
                if parsed[key] < 0.0:
                    raise AutopilotValidationError(f"evaluation command {key} must be non-negative")
            parsed_commands.append(parsed)
        object.__setattr__(self, "command_metrics", tuple(parsed_commands))
        if not isinstance(self.episode_metrics, tuple):
            object.__setattr__(self, "episode_metrics", tuple(self.episode_metrics))
        if not self.episode_metrics:
            raise AutopilotValidationError("evaluation.episode_metrics must be non-empty")
        for index, item in enumerate(self.episode_metrics):
            _mapping(item, f"evaluation.episode_metrics[{index}]")
            _json_finite(item, f"evaluation.episode_metrics[{index}]")
        artifacts = tuple(_text(value, "evaluation.artifact_id", maximum=256) for value in self.artifact_ids)
        if not artifacts:
            raise AutopilotValidationError("evaluation.artifact_ids must be non-empty")
        if len(set(artifacts)) != len(artifacts):
            raise AutopilotValidationError("evaluation.artifact_ids must be unique")
        object.__setattr__(self, "artifact_ids", artifacts)
        if not isinstance(self.hard_gates, dict) or any(not isinstance(value, bool) for value in self.hard_gates.values()):
            raise AutopilotValidationError("evaluation.hard_gates must contain boolean values")
        margins = {key: _finite(value, f"evaluation.gate_margins.{key}") for key, value in self.gate_margins.items()}
        object.__setattr__(self, "gate_margins", margins)
        if not isinstance(self.ranking, dict):
            raise AutopilotValidationError("evaluation.ranking must be an object")
        _json_finite(self.ranking, "evaluation.ranking")
        object.__setattr__(self, "failure_reason", _optional_text(self.failure_reason, "evaluation.failure_reason", maximum=2000))

    @classmethod
    def from_dict(cls, value: Any) -> "EvaluationReportV1":
        obj = _mapping(value, "evaluation")
        required = (
            "schema_version", "id", "trial_id", "checkpoint_sha256", "config_sha256", "reward_profile_sha256",
            "physics_profile_sha256", "spring_profile_sha256", "command_profile_sha256", "seed", "evaluation_profile", "command_metrics",
            "strict_checkpoint_load", "episode_artifact_sha256", "episode_metrics", "artifact_ids", "hard_gates",
            "gate_margins", "ranking", "failure_reason",
        )
        _strict_keys(obj, required=required, name="evaluation")
        for key in ("command_metrics", "episode_metrics", "artifact_ids"):
            if not isinstance(obj[key], (list, tuple)):
                raise AutopilotValidationError(f"evaluation.{key} must be an array")
        return cls(
            schema_version=obj["schema_version"], id=obj["id"], trial_id=obj["trial_id"],
            checkpoint_sha256=obj["checkpoint_sha256"], config_sha256=obj["config_sha256"],
            reward_profile_sha256=obj["reward_profile_sha256"], physics_profile_sha256=obj["physics_profile_sha256"],
            spring_profile_sha256=obj["spring_profile_sha256"],
            command_profile_sha256=obj["command_profile_sha256"], seed=obj["seed"],
            evaluation_profile=obj["evaluation_profile"], strict_checkpoint_load=obj["strict_checkpoint_load"],
            episode_artifact_sha256=obj["episode_artifact_sha256"], command_metrics=tuple(obj["command_metrics"]),
            episode_metrics=tuple(obj["episode_metrics"]), artifact_ids=tuple(obj["artifact_ids"]),
            hard_gates=dict(obj["hard_gates"]), gate_margins=dict(obj["gate_margins"]),
            ranking=dict(obj["ranking"]), failure_reason=obj["failure_reason"],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version, "id": self.id, "trial_id": self.trial_id,
            "checkpoint_sha256": self.checkpoint_sha256, "config_sha256": self.config_sha256,
            "reward_profile_sha256": self.reward_profile_sha256,
            "physics_profile_sha256": self.physics_profile_sha256,
            "spring_profile_sha256": self.spring_profile_sha256,
            "command_profile_sha256": self.command_profile_sha256, "seed": self.seed,
            "evaluation_profile": self.evaluation_profile, "strict_checkpoint_load": self.strict_checkpoint_load,
            "episode_artifact_sha256": self.episode_artifact_sha256,
            "command_metrics": [dict(item) for item in self.command_metrics],
            "episode_metrics": [dict(item) for item in self.episode_metrics],
            "artifact_ids": list(self.artifact_ids), "hard_gates": dict(self.hard_gates),
            "gate_margins": dict(self.gate_margins), "ranking": dict(self.ranking),
            "failure_reason": self.failure_reason,
        }


def _minimum_margin(actual: float, threshold: float) -> float:
    denominator = max(abs(threshold), 1e-9)
    return (actual - threshold) / denominator


def _maximum_margin(actual: float, threshold: float) -> float:
    denominator = max(abs(threshold), 1e-9)
    return (threshold - actual) / denominator


def evaluate_report(
    report: EvaluationReportV1,
    goal: GoalSpecV1,
    *,
    expected_trial_checkpoint_sha256: str,
    original_reward_values: Mapping[str, Any],
    candidate_reward_values: Mapping[str, Any],
) -> EvaluationReportV1:
    """Apply fail-closed identity/safety gates and a deterministic ranking key."""

    gates: dict[str, bool] = {}
    margins: dict[str, float] = {}
    expected_checkpoint = _sha256(
        expected_trial_checkpoint_sha256,
        "expected_trial_checkpoint_sha256",
    )
    identity_values = {
        "config_identity": report.config_sha256 == goal.config_sha256,
        "physics_identity": report.physics_profile_sha256 == goal.physics_profile_sha256,
        "spring_identity": report.spring_profile_sha256 == goal.spring_profile_sha256,
        "command_identity": report.command_profile_sha256 == goal.command_profile_sha256,
        "evaluation_profile": report.evaluation_profile == goal.evaluation_profile,
        "seed_protocol": report.seed in goal.training_seeds,
        # goal.checkpoint_sha256 identifies the frozen source lineage. The
        # evaluation must bind to this trial's trained output checkpoint.
        "checkpoint_identity": report.checkpoint_sha256 == expected_checkpoint,
        "strict_checkpoint_load": report.strict_checkpoint_load,
        "episode_evidence": bool(report.episode_metrics) and bool(report.episode_artifact_sha256),
        "evaluation_complete": report.failure_reason is None,
    }
    gates.update(identity_values)
    margins.update({key: 1.0 if passed else -1.0 for key, passed in identity_values.items()})

    commands = report.command_metrics
    command_pass_ratio = sum(1 for item in commands if item["accept_pass"]) / len(commands)
    skill_ratios = []
    for skill in sorted({item["skill"] for item in commands}):
        matching = [item for item in commands if item["skill"] == skill]
        skill_ratios.append(sum(1 for item in matching if item["accept_pass"]) / len(matching))
    min_skill_pass_ratio = min(skill_ratios)
    max_fall_rate = max(item["fall_rate"] for item in commands)
    min_tracking = min(item["tracking_quality"] for item in commands)
    min_stability = min(item["stability_quality"] for item in commands)
    min_direction = min(item["direction_sign_ratio"] for item in commands)
    max_linear_leak = max(item["linear_leak"] for item in commands)
    max_yaw_leak = max(item["yaw_leak"] for item in commands)
    max_energy = max(item["energy_per_distance"] for item in commands)
    observed = {
        "command_pass_ratio": (command_pass_ratio, goal.skill_gates["min_command_pass_ratio"], "minimum"),
        "skill_pass_ratio": (min_skill_pass_ratio, goal.skill_gates["min_skill_pass_ratio"], "minimum"),
        "fall_rate": (max_fall_rate, goal.skill_gates["max_fall_rate"], "maximum"),
        "tracking_quality": (min_tracking, goal.skill_gates["min_tracking_quality"], "minimum"),
        "stability_quality": (min_stability, goal.skill_gates["min_stability_quality"], "minimum"),
        "direction_sign_ratio": (min_direction, goal.skill_gates["min_direction_sign_ratio"], "minimum"),
        "linear_leak": (max_linear_leak, goal.skill_gates["max_linear_leak"], "maximum"),
        "yaw_leak": (max_yaw_leak, goal.skill_gates["max_yaw_leak"], "maximum"),
        # The evaluator clamps saturated/undefined ratios to its configured
        # ceiling.  Treat equality as invalid evidence rather than allowing a
        # saturated measurement to masquerade as exactly-on-budget energy.
        "energy_per_distance": (
            max_energy,
            goal.skill_gates["max_energy_per_distance"],
            "maximum_exclusive",
        ),
    }
    for key, (actual, threshold, kind) in observed.items():
        if kind == "minimum":
            passed = actual >= threshold
        elif kind == "maximum_exclusive":
            passed = actual < threshold
        else:
            passed = actual <= threshold
        gates[key] = passed
        margins[key] = _minimum_margin(actual, threshold) if kind == "minimum" else _maximum_margin(actual, threshold)

    original = {key: _finite(value, f"original_reward_values.{key}") for key, value in original_reward_values.items()}
    candidate = {key: _finite(value, f"candidate_reward_values.{key}") for key, value in candidate_reward_values.items()}
    if set(original) != set(candidate):
        raise AutopilotValidationError("original and candidate reward profiles must contain identical keys")
    reward_distance = sum(
        abs(candidate[key] - original[key]) / max(abs(original[key]), 1e-12) for key in sorted(original)
    )
    eligible = all(gates.values())
    passed_gate_count = sum(gates.values())
    worst_margin = min(margins.values())
    mean_tracking = sum(item["tracking_quality"] for item in commands) / len(commands)
    mean_energy = sum(item["energy_per_distance"] for item in commands) / len(commands)
    sort_key = [
        1 if eligible else 0,
        passed_gate_count,
        round(worst_margin, 12),
        round(mean_tracking, 12),
        round(-mean_energy, 12),
        round(-reward_distance, 12),
    ]
    ranking = {
        "eligible": eligible,
        "passed_gate_count": passed_gate_count,
        "gate_count": len(gates),
        "worst_normalized_gate_margin": worst_margin,
        "mean_tracking_quality": mean_tracking,
        "mean_energy_per_distance": mean_energy,
        "reward_distance": reward_distance,
        "sort_key": sort_key,
    }
    return replace(report, hard_gates=gates, gate_margins=margins, ranking=ranking)


def evaluation_rank_key(report: EvaluationReportV1) -> tuple[Any, ...]:
    if not report.ranking or "sort_key" not in report.ranking:
        raise AutopilotValidationError("evaluation report has not been deterministically evaluated")
    raw = report.ranking["sort_key"]
    if not isinstance(raw, list) or len(raw) != 6:
        raise AutopilotValidationError("evaluation ranking sort_key is invalid")
    return (*tuple(_finite(value, "evaluation.ranking.sort_key") for value in raw), report.id)


CAMPAIGN_STATES = (
    "draft", "armed", "control_training", "control_evaluating", "awaiting_advisor",
    "candidate_training", "candidate_evaluating", "confirming", "simulation_goal_met",
    "paused", "waiting_for_chatgpt", "patch_handoff", "budget_exhausted", "blocked_safety",
    "stopped", "failed",
)

TERMINAL_STATES = frozenset(
    {"simulation_goal_met", "patch_handoff", "budget_exhausted", "blocked_safety", "stopped", "failed"}
)
HOST_SLOT_STATES = frozenset(set(CAMPAIGN_STATES) - TERMINAL_STATES - {"draft"})

ALLOWED_TRANSITIONS = {
    "draft": {"armed"},
    "armed": {"control_training", "paused", "stopped", "failed", "blocked_safety"},
    "control_training": {"control_evaluating", "paused", "stopped", "failed", "budget_exhausted", "blocked_safety"},
    "control_evaluating": {"awaiting_advisor", "simulation_goal_met", "paused", "stopped", "failed", "budget_exhausted", "blocked_safety"},
    "awaiting_advisor": {"candidate_training", "waiting_for_chatgpt", "paused", "patch_handoff", "stopped", "failed", "budget_exhausted"},
    "candidate_training": {"candidate_evaluating", "paused", "stopped", "failed", "budget_exhausted", "blocked_safety"},
    "candidate_evaluating": {"awaiting_advisor", "confirming", "paused", "patch_handoff", "stopped", "failed", "budget_exhausted", "blocked_safety"},
    "confirming": {"candidate_training", "candidate_evaluating", "simulation_goal_met", "patch_handoff", "paused", "stopped", "failed", "budget_exhausted", "blocked_safety"},
    "paused": {"armed", "control_training", "control_evaluating", "awaiting_advisor", "candidate_training", "candidate_evaluating", "confirming", "stopped"},
    "waiting_for_chatgpt": {"awaiting_advisor", "paused", "patch_handoff", "stopped", "budget_exhausted"},
}


def validate_transition(current: str, target: str, *, resume_state: str | None = None) -> None:
    if current not in CAMPAIGN_STATES or target not in CAMPAIGN_STATES:
        raise AutopilotValidationError("unknown campaign state")
    if current in TERMINAL_STATES:
        raise AutopilotValidationError(f"terminal campaign state {current!r} cannot transition")
    allowed = ALLOWED_TRANSITIONS.get(current, set())
    if target not in allowed:
        raise AutopilotValidationError(f"campaign may not transition from {current!r} to {target!r}")
    if current == "paused" and resume_state is not None and target != resume_state and target != "stopped":
        raise AutopilotValidationError(f"paused campaign must resume to {resume_state!r}")


def next_permitted_actions(state: str, resume_state: str | None = None) -> list[str]:
    if state in TERMINAL_STATES:
        return []
    if state == "draft":
        return ["update", "arm"]
    if state == "paused":
        return ["resume", "stop"]
    actions = ["pause", "stop"]
    if state == "awaiting_advisor":
        actions = ["propose_candidate", "pause", "request_patch_handoff", "stop"]
    elif state == "waiting_for_chatgpt":
        # A returning advisor records its visit through the decision write.
        # Keeping this identical to ``awaiting_advisor`` avoids charging a
        # second connector poll merely to wake the durable controller.
        actions = ["propose_candidate", "pause", "request_patch_handoff", "stop"]
    return actions


@dataclass(frozen=True)
class CampaignSnapshotV1:
    id: str
    revision: int
    state: str
    goal: GoalSpecV1
    reward_catalog: tuple[RewardCatalogEntryV1, ...]
    leader: dict[str, Any]
    budget: dict[str, Any]
    active_process: dict[str, Any] | None
    candidate_lineage: tuple[dict[str, Any], ...]
    decisions: tuple[dict[str, Any], ...]
    evaluations: tuple[dict[str, Any], ...]
    connector: dict[str, Any]
    resume_state: str | None
    terminal_reason: str | None
    created_at: str
    updated_at: str
    schema_version: str = CAMPAIGN_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _schema(self.schema_version, CAMPAIGN_SCHEMA_VERSION, "campaign")
        object.__setattr__(self, "id", _text(self.id, "campaign.id", maximum=128))
        object.__setattr__(self, "revision", _integer(self.revision, "campaign.revision", minimum=0))
        if self.state not in CAMPAIGN_STATES:
            raise AutopilotValidationError("campaign.state is invalid")
        if not isinstance(self.goal, GoalSpecV1):
            raise AutopilotValidationError("campaign.goal must be GoalSpecV1")
        catalog = tuple(self.reward_catalog)
        if not catalog or any(not isinstance(item, RewardCatalogEntryV1) for item in catalog):
            raise AutopilotValidationError("campaign.reward_catalog must be non-empty")
        keys = [item.key for item in catalog]
        if len(set(keys)) != len(keys):
            raise AutopilotValidationError("campaign.reward_catalog keys must be unique")
        object.__setattr__(self, "reward_catalog", catalog)
        for name in ("leader", "budget", "connector"):
            _mapping(getattr(self, name), f"campaign.{name}")
            _json_finite(getattr(self, name), f"campaign.{name}")
        if self.active_process is not None:
            _mapping(self.active_process, "campaign.active_process")
            _json_finite(self.active_process, "campaign.active_process")
        for name in ("candidate_lineage", "decisions", "evaluations"):
            values = tuple(getattr(self, name))
            for index, value in enumerate(values):
                _mapping(value, f"campaign.{name}[{index}]")
                _json_finite(value, f"campaign.{name}[{index}]")
            object.__setattr__(self, name, values)
        if self.resume_state is not None and self.resume_state not in CAMPAIGN_STATES:
            raise AutopilotValidationError("campaign.resume_state is invalid")
        object.__setattr__(self, "terminal_reason", _optional_text(self.terminal_reason, "campaign.terminal_reason", maximum=2000))
        object.__setattr__(self, "created_at", _text(self.created_at, "campaign.created_at", maximum=64))
        object.__setattr__(self, "updated_at", _text(self.updated_at, "campaign.updated_at", maximum=64))

    @classmethod
    def from_dict(cls, value: Any) -> "CampaignSnapshotV1":
        obj = _mapping(value, "campaign")
        required = (
            "schema_version", "id", "revision", "state", "goal", "reward_catalog", "leader", "budget",
            "active_process", "candidate_lineage", "decisions", "evaluations", "connector", "resume_state",
            "terminal_reason", "created_at", "updated_at",
        )
        _strict_keys(obj, required=required, optional=("next_permitted_actions",), name="campaign")
        for key in ("reward_catalog", "candidate_lineage", "decisions", "evaluations"):
            if not isinstance(obj[key], (list, tuple)):
                raise AutopilotValidationError(f"campaign.{key} must be an array")
        return cls(
            schema_version=obj["schema_version"], id=obj["id"], revision=obj["revision"], state=obj["state"],
            goal=GoalSpecV1.from_dict(obj["goal"]),
            reward_catalog=tuple(RewardCatalogEntryV1.from_dict(item) for item in obj["reward_catalog"]),
            leader=dict(obj["leader"]), budget=dict(obj["budget"]), active_process=obj["active_process"],
            candidate_lineage=tuple(obj["candidate_lineage"]), decisions=tuple(obj["decisions"]),
            evaluations=tuple(obj["evaluations"]), connector=dict(obj["connector"]),
            resume_state=obj["resume_state"], terminal_reason=obj["terminal_reason"],
            created_at=obj["created_at"], updated_at=obj["updated_at"],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version, "id": self.id, "revision": self.revision, "state": self.state,
            "goal": self.goal.to_dict(), "reward_catalog": [item.to_dict() for item in self.reward_catalog],
            "leader": dict(self.leader), "budget": dict(self.budget),
            "active_process": None if self.active_process is None else dict(self.active_process),
            "candidate_lineage": [dict(item) for item in self.candidate_lineage],
            "decisions": [dict(item) for item in self.decisions],
            "evaluations": [dict(item) for item in self.evaluations], "connector": dict(self.connector),
            "resume_state": self.resume_state, "terminal_reason": self.terminal_reason,
            "created_at": self.created_at, "updated_at": self.updated_at,
            "next_permitted_actions": next_permitted_actions(self.state, self.resume_state),
        }


def autopilot_capabilities(*, enabled: bool = False) -> dict[str, Any]:
    return {
        "schema_version": CAPABILITIES_SCHEMA_VERSION,
        "enabled": bool(enabled),
        "supported_tasks": list(SUPPORTED_TASKS),
        "supported_stages": {task: sorted(stages) for task, stages in STAGE_COMMAND_PRESETS.items()},
        "gaits": list(SUPPORTED_GAITS),
        "states": list(CAMPAIGN_STATES),
        "max_training_trials": MAX_TRAINING_TRIALS,
        "max_gpu_hours": MAX_GPU_HOURS,
        "default_seeds": list(DEFAULT_TRAINING_SEEDS),
        "command_profiles": {
            task: {
                f"stage{stage}": {
                    gait: command_envelope_to_dict(compile_command_envelope(task, stage, gait))
                    for gait in SUPPORTED_GAITS
                }
                for stage in stages
            }
            for task, stages in STAGE_COMMAND_PRESETS.items()
        },
        "default_reward_keys": {
            task: {f"stage{stage}": list(eligible_reward_keys(task, stage)) for stage in stages}
            for task, stages in STAGE_COMMAND_PRESETS.items()
        },
    }
