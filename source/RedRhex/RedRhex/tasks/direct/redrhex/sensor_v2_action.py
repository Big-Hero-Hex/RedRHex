"""Dependency-light construction of the simulator's Sensor-V2 action contract."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

from redrhex_policy_io import ForwardResidualActionContractV2


def _first(value: Any, fallback: float) -> float:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if not value:
            return float(fallback)
        return float(value[0])
    if value is None:
        return float(fallback)
    return float(value)


def _configured_joint_positions(cfg: Any, names: Sequence[str]) -> tuple[float, ...]:
    robot_cfg = getattr(cfg, "robot_cfg", None)
    init_state = getattr(robot_cfg, "init_state", None)
    joint_positions = getattr(init_state, "joint_pos", None)
    if not isinstance(joint_positions, dict):
        raise ValueError("Sensor V2 robot_cfg.init_state.joint_pos must be an explicit mapping")
    try:
        return tuple(float(joint_positions[name]) for name in names)
    except KeyError as exc:
        raise ValueError(f"Sensor V2 initial joint position is missing {exc.args[0]!r}") from exc


def _configured_v2_main_reset(cfg: Any, names: Sequence[str]) -> tuple[float, ...]:
    configured = getattr(cfg, "v2_reset_main_position_rad", None)
    if configured is None:
        return _configured_joint_positions(cfg, names)
    try:
        values = tuple(float(value) for value in configured)
    except (TypeError, ValueError) as exc:
        raise ValueError("Sensor V2 reset main position must be a finite six-vector") from exc
    if len(values) != len(names) or any(not math.isfinite(value) for value in values):
        raise ValueError("Sensor V2 reset main position must be a finite six-vector")
    return values


def forward_residual_action_contract_v2_from_config(
    cfg: Any,
) -> ForwardResidualActionContractV2:
    """Build and fail-closed validate the contract used by training and export."""

    drive_scale = _first(
        getattr(cfg, "stage_drive_vel_scale", None),
        getattr(cfg, "main_drive_vel_scale", 8.0),
    )
    residual_ratio = _first(
        getattr(cfg, "stage_forward_policy_drive_residual_scale", None),
        getattr(cfg, "main_drive_residual_scale", 0.1),
    )
    warmup_steps = _first(getattr(cfg, "stage_action_warmup_steps", None), 0.0)
    if not math.isfinite(warmup_steps) or not warmup_steps.is_integer() or warmup_steps < 0.0:
        raise ValueError("Sensor V2 action warmup must be a non-negative integer")
    defaults = ForwardResidualActionContractV2()
    contract = ForwardResidualActionContractV2(
        main_residual_scale_rad_s=drive_scale * residual_ratio,
        forward_command_reference_m_s=float(getattr(cfg, "drive_bias_vx_ref", 0.45)),
        forward_bias_scale=_first(
            getattr(cfg, "stage_forward_bias_scale", None),
            getattr(cfg, "forward_drive_action_scale", 1.0),
        ),
        phase_lock_gain=float(getattr(cfg, "forward_phase_lock_gain", 1.2)),
        phase_correction_limit_rad_s=2.0,
        residual_cap_ratio=_first(
            getattr(cfg, "stage_forward_residual_cap_ratio", None),
            getattr(cfg, "forward_residual_cap_ratio", 0.26),
        ),
        action_warmup_steps=int(warmup_steps),
        # This is the raw contract limit and must match the simulator actuator
        # ceiling.  ROS may tighten it further as a separately reported
        # hardware-safety layer, but Isaac must not silently clip a larger raw
        # parity target inside PhysX.
        main_velocity_limit_rad_s=float(
            getattr(
                cfg,
                "main_drive_contract_velocity_limit_rad_s",
                defaults.main_velocity_limit_rad_s,
            )
        ),
        initial_main_position_rad=_configured_v2_main_reset(
            cfg,
            defaults.MAIN_JOINT_ORDER,
        ),
        abad_neutral_position_rad=_configured_joint_positions(
            cfg,
            defaults.ABAD_JOINT_ORDER,
        ),
    )

    fixed_checks = (
        ("action_space", int(getattr(cfg, "action_space", -1)), contract.ACTION_DIM),
        (
            "main_drive_joint_names",
            tuple(getattr(cfg, "main_drive_joint_names", ())),
            contract.MAIN_JOINT_ORDER,
        ),
        (
            "abad_joint_names",
            tuple(getattr(cfg, "abad_joint_names", ())),
            contract.ABAD_JOINT_ORDER,
        ),
        (
            "tripod_a_leg_indices",
            tuple(getattr(cfg, "tripod_a_leg_indices", ())),
            contract.TRIPOD_A,
        ),
        (
            "tripod_b_leg_indices",
            tuple(getattr(cfg, "tripod_b_leg_indices", ())),
            contract.TRIPOD_B,
        ),
        (
            "leg_direction_multiplier",
            tuple(float(value) for value in getattr(cfg, "leg_direction_multiplier", ())),
            contract.LEG_DIRECTION_MULTIPLIER,
        ),
    )
    for name, actual, expected in fixed_checks:
        if actual != expected:
            raise ValueError(f"Sensor V2 {name} {actual!r} disagrees with action contract {expected!r}")

    scalar_checks = (
        ("base_gait_frequency", float(cfg.base_gait_frequency), contract.NOMINAL_GAIT_FREQUENCY_HZ),
        ("stance_phase_start", float(cfg.stance_phase_start), contract.STANCE_PHASE_START_RAD),
        ("stance_phase_end", float(cfg.stance_phase_end), contract.STANCE_PHASE_END_RAD),
        (
            "stance_duty_cycle",
            float(
                getattr(
                    cfg,
                    "stance_duty_cycle",
                    contract.STANCE_DUTY_CYCLE,
                )
            ),
            contract.STANCE_DUTY_CYCLE,
        ),
        ("stance_velocity_ratio", float(cfg.stance_velocity_ratio), contract.STANCE_VELOCITY_RATIO),
        ("swing_velocity_ratio", float(cfg.swing_velocity_ratio), contract.SWING_VELOCITY_RATIO),
        ("tripod_phase_offset", float(cfg.tripod_phase_offset), contract.TRIPOD_PHASE_OFFSET_RAD),
        (
            "mode_forward_min_vx",
            float(getattr(cfg, "mode_forward_min_vx", contract.FORWARD_ACTIVE_MIN_M_S)),
            contract.FORWARD_ACTIVE_MIN_M_S,
        ),
        (
            "mode_lin_zero_thresh",
            float(
                getattr(
                    cfg,
                    "mode_lin_zero_thresh",
                    contract.MAX_ABS_LATERAL_COMMAND_M_S,
                )
            ),
            contract.MAX_ABS_LATERAL_COMMAND_M_S,
        ),
        (
            "mode_yaw_zero_thresh",
            float(
                getattr(
                    cfg,
                    "mode_yaw_zero_thresh",
                    contract.MAX_ABS_YAW_COMMAND_RAD_S,
                )
            ),
            contract.MAX_ABS_YAW_COMMAND_RAD_S,
        ),
    )
    for name, actual, expected in scalar_checks:
        if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1.0e-12):
            raise ValueError(f"Sensor V2 {name}={actual!r} disagrees with action contract {expected!r}")
    reset_gait_phase = float(getattr(cfg, "v2_reset_gait_phase_rad", 0.0))
    if not math.isfinite(reset_gait_phase):
        raise ValueError("Sensor V2 reset gait phase must be finite")
    if not bool(getattr(cfg, "strict_forward_residual_actions", False)):
        raise ValueError("Sensor V2 requires strict_forward_residual_actions=True")
    return contract
