# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Additive configuration for the sensor-only forward research route."""

from __future__ import annotations

from isaaclab.utils import configclass

from .redrhex_env_cfg import RedrhexForwardFastEnvCfg


@configclass
class RedrhexForwardSensorV2EnvCfg(RedrhexForwardFastEnvCfg):
    """Forward-only V2 task without changing the legacy 56/280-D tasks.

    ``policy`` is a flat Gym compatibility view of the two semantic actor inputs
    (60 x 36 sensor history followed by the current 3-D command).  The V2 runner
    consumes the separately returned ``sensor_history_v2`` and ``command_v2``
    groups; the command is never replicated into the temporal history.
    """

    sensor_contract_id = "redrhex.student-observation.v2"
    action_contract_id = "redrhex.forward-residual-action.v2"
    calibration_profile_id = "sim_nominal_sensor_v2"

    sensor_frame_dim = 36
    sensor_history_length = 60
    sensor_sample_hz = 60
    command_dim = 3
    observation_space = sensor_history_length * sensor_frame_dim + command_dim
    state_space = 65

    # The simulator provides an exact quaternion.  The alternative estimator is
    # explicit and contract-hashed; there is no runtime fallback between modes.
    sensor_attitude_mode = "validated_quaternion"
    sensor_imu_frame_id = "base_link"
    sensor_imu_mount_rpy_rad = (0.0, 0.0, 0.0)
    sensor_quaternion_covariance_max = 1.0e-6
    sensor_rest_gravity_tolerance = 0.05
    sensor_gravity_correction_gain = 0.02
    sensor_accel_norm_gate_mps2 = (7.5, 12.0)

    # The policy produces residuals around the existing procedural forward CPG.
    # ABAD is structurally disabled for F0--F5, including non-forward edge cases.
    strict_forward_residual_actions = True
    lock_abad_in_forward = True
    lock_main_drive_in_lateral = True

    # One-conv TCN history must be chronological at the environment boundary.
    sensor_history_order = "oldest_to_newest"
    sensor_history_prefill = "repeat_first_valid_sample"

    # Outcome-oriented forward reward.  Gait priors are deliberately disabled;
    # non-zero regularizers use the documented half/base/double sensitivity knob.
    v2_regularizer_sensitivity = "base"
    v2_regularizer_sensitivity_multipliers = {
        "half": 0.5,
        "base": 1.0,
        "double": 2.0,
    }
    v2_action_smoothness_penalty = 0.02
    v2_action_saturation_penalty = 0.03
    v2_action_saturation_start = 0.95
    v2_tilt_penalty = 0.80
    v2_lateral_velocity_penalty = 1.30
    v2_yaw_velocity_penalty = 0.65

    v2_reward_scales = {
        "forward_progress": 5.5,
        "velocity_tracking": 4.5,
        "mode_specialization": 0.0,
        "axis_suppression": 0.0,
        "lateral_drive_soft_penalty": 0.0,
        "lateral_speed_deficit_penalty": 0.0,
        "lateral_speed_target_ratio": 0.70,
        "lateral_speed_bonus": 0.0,
        "diag_sign_bonus": 0.0,
        "diag_wrong_sign_penalty": 0.0,
        "diag_speed_bonus": 0.0,
        "forward_prior_coherence": 0.0,
        "forward_prior_antiphase": 0.0,
        "forward_prior_duty": 0.0,
        "forward_prior_vel_ratio": 0.0,
        "forward_prior_overlap": 0.0,
        "height_maintain": 0.9,
        "target_base_height": 0.12,
        "height_sigma": 0.08,
        "height_low_penalty": 1.2,
        "leg_moving": 0.20,
        "stall_penalty": -2.5,
        "fall": -8.0,
        "fall_height_threshold": 0.085,
        "fall_tilt_threshold": 1.70,
        "fall_roll_threshold": 1.30,
        "fall_pitch_threshold": 1.30,
        "yaw_mode_track_bonus": 0.0,
        "yaw_spin_bonus": 0.0,
        "yaw_roll_pitch_penalty": 0.0,
        "yaw_height_penalty": 0.0,
        "yaw_target_base_height": 0.12,
        "yaw_slip_penalty": 0.0,
        "yaw_slip_cap": 2.0,
        "yaw_cheat_penalty": 0.0,
        "yaw_cheat_min_wz": 0.4,
        "yaw_cheat_tilt_thresh": 0.30,
        "lin_tracking_sigma": 0.30,
        "yaw_tracking_sigma": 0.35,
        "energy_per_distance": 0.0005,
    }
