# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Sensor-only V2 environment layered on the proven forward controller."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch

from isaaclab.utils.math import quat_apply_inverse

from .redrhex_env import RedrhexEnv
from .redrhex_sensor_v2_env_cfg import RedrhexForwardSensorV2EnvCfg


def _wrapped_difference(current: torch.Tensor, previous: torch.Tensor) -> torch.Tensor:
    """Return the shortest signed angular difference in radians."""

    return torch.atan2(torch.sin(current - previous), torch.cos(current - previous))


def _normalize_vector(vector: torch.Tensor, *, eps: float = 1.0e-6) -> torch.Tensor:
    return vector / torch.linalg.vector_norm(vector, dim=-1, keepdim=True).clamp(min=eps)


class RedrhexForwardSensorV2Env(RedrhexEnv):
    """Additive forward task with a physical-feedback-only actor contract.

    The procedural CPG remains inside the action decoder.  It is deliberately
    absent from every student observation.  Simulator truth is retained only in
    the teacher/critic groups and auxiliary targets.
    """

    cfg: RedrhexForwardSensorV2EnvCfg

    def _setup_buffers(self) -> None:
        super()._setup_buffers()

        history = int(self.cfg.sensor_history_length)
        frame_dim = int(self.cfg.sensor_frame_dim)
        if history != 60 or frame_dim != 36 or int(self.cfg.sensor_sample_hz) != 60:
            raise ValueError("Sensor V2 requires a fixed 60 x 36 history sampled at 60 Hz")
        if self.cfg.sensor_history_order != "oldest_to_newest":
            raise ValueError("Sensor V2 history order must be oldest_to_newest")

        self._sensor_history_v2 = torch.zeros(
            self.num_envs, history, frame_dim, device=self.device
        )
        self._sensor_history_valid_count_v2 = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self._sensor_history_ready_v2 = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._sensor_prev_main_pos_v2 = torch.zeros(
            self.num_envs, self.num_main_drive_joints, device=self.device
        )
        self._sensor_prev_abad_pos_v2 = torch.zeros(
            self.num_envs, self.num_abad_joints, device=self.device
        )
        self._sensor_encoder_initialized_v2 = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._sensor_last_sample_step_v2 = torch.full(
            (self.num_envs,), -1, dtype=torch.long, device=self.device
        )
        self._sensor_frame_v2 = torch.zeros(self.num_envs, frame_dim, device=self.device)
        self._sensor_prev_root_lin_vel_w_v2 = torch.zeros(self.num_envs, 3, device=self.device)
        self._sensor_gravity_estimate_v2 = _normalize_vector(self.reference_projected_gravity.clone())

        # The base environment's legacy histories are not consumed by V2.  Keep
        # tiny buffers because reset and diagnostics still reference their names.
        self._obs_history = torch.zeros(self.num_envs, 1, frame_dim, device=self.device)
        self._policy_obs_history = torch.zeros(self.num_envs, 1, 1, device=self.device)

        for key in (
            "rew_sensor_v2_tilt",
            "rew_sensor_v2_lateral",
            "rew_sensor_v2_yaw",
            "rew_sensor_v2_smoothness",
            "rew_sensor_v2_saturation",
            "diag_sensor_v2_history_ready",
        ):
            self.episode_sums[key] = torch.zeros(self.num_envs, device=self.device)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        if actions.shape[-1] != 12:
            raise ValueError(f"Sensor V2 action contract requires 12 actions, got {actions.shape[-1]}")
        strict_actions = actions.clone()
        if bool(getattr(self.cfg, "strict_forward_residual_actions", True)):
            strict_actions[..., 6:12] = 0.0
        super()._pre_physics_step(strict_actions)

    def _validated_projected_gravity(self, root_quat: torch.Tensor) -> torch.Tensor:
        if str(self.cfg.sensor_imu_frame_id).strip() != "base_link":
            raise RuntimeError("simulated validated_quaternion mode requires IMU frame_id=base_link")
        finite = torch.isfinite(root_quat).all(dim=-1)
        norm = torch.linalg.vector_norm(root_quat, dim=-1)
        valid = finite & (torch.abs(norm - 1.0) <= 1.0e-3)
        if not bool(valid.all()):
            bad = torch.nonzero(~valid, as_tuple=False).flatten().tolist()
            raise RuntimeError(f"invalid simulated IMU quaternion for environments {bad[:8]}")
        normalized = root_quat / norm.unsqueeze(-1)
        return _normalize_vector(quat_apply_inverse(normalized, self._gravity_vec_w))

    def _estimated_projected_gravity(
        self,
        root_quat: torch.Tensor,
        root_lin_vel_w: torch.Tensor,
        gyro_body: torch.Tensor,
        new_sample: torch.Tensor,
    ) -> torch.Tensor:
        """Causal gyro propagation with gated accelerometer correction."""

        dt = 1.0 / float(self.cfg.sensor_sample_hz)
        world_accel = (root_lin_vel_w - self._sensor_prev_root_lin_vel_w_v2) / dt
        gravity_mps2_w = self._gravity_vec_w * 9.81
        specific_force_body = quat_apply_inverse(root_quat, world_accel - gravity_mps2_w)

        estimate = self._sensor_gravity_estimate_v2
        propagated = _normalize_vector(estimate - torch.cross(gyro_body, estimate, dim=-1) * dt)
        accel_norm = torch.linalg.vector_norm(specific_force_body, dim=-1)
        low, high = tuple(float(v) for v in self.cfg.sensor_accel_norm_gate_mps2)
        correction_valid = (
            torch.isfinite(specific_force_body).all(dim=-1)
            & (accel_norm >= low)
            & (accel_norm <= high)
        )
        accel_gravity = _normalize_vector(-specific_force_body)
        gain = float(self.cfg.sensor_gravity_correction_gain)
        corrected = _normalize_vector((1.0 - gain) * propagated + gain * accel_gravity)
        updated = torch.where(correction_valid.unsqueeze(-1), corrected, propagated)
        self._sensor_gravity_estimate_v2 = torch.where(
            new_sample.unsqueeze(-1), updated, self._sensor_gravity_estimate_v2
        )
        return self._sensor_gravity_estimate_v2

    def _build_sensor_frame_v2(self) -> tuple[torch.Tensor, torch.Tensor]:
        root_quat = self.robot.data.root_quat_w
        root_lin_vel_w = self.robot.data.root_lin_vel_w
        gyro_body = quat_apply_inverse(root_quat, self.robot.data.root_ang_vel_w)
        gyro_body = torch.nan_to_num(gyro_body, nan=0.0, posinf=0.0, neginf=0.0)

        sample_step = torch.full(
            (self.num_envs,),
            int(self._sim_step_counter // max(int(self.cfg.decimation), 1)),
            dtype=torch.long,
            device=self.device,
        )
        new_sample = sample_step != self._sensor_last_sample_step_v2

        mode = str(self.cfg.sensor_attitude_mode)
        if mode == "validated_quaternion":
            gravity_body = self._validated_projected_gravity(root_quat)
        elif mode == "causal_gyro_accel":
            gravity_body = self._estimated_projected_gravity(
                root_quat, root_lin_vel_w, gyro_body, new_sample
            )
        else:
            raise ValueError(
                "sensor_attitude_mode must be validated_quaternion or causal_gyro_accel; "
                f"got {mode!r}"
            )

        main_pos = self.joint_pos[:, self._main_drive_indices]
        abad_pos = self.joint_pos[:, self._abad_indices]
        dt = 1.0 / float(self.cfg.sensor_sample_hz)
        main_velocity = _wrapped_difference(main_pos, self._sensor_prev_main_pos_v2) / dt
        abad_velocity = (abad_pos - self._sensor_prev_abad_pos_v2) / dt
        initialized = self._sensor_encoder_initialized_v2.unsqueeze(-1)
        main_velocity = torch.where(initialized, main_velocity, torch.zeros_like(main_velocity))
        abad_velocity = torch.where(initialized, abad_velocity, torch.zeros_like(abad_velocity))

        candidate = torch.cat(
            (
                gyro_body,
                gravity_body,
                torch.sin(main_pos),
                torch.cos(main_pos),
                main_velocity,
                abad_pos - self._abad_rest_pos.expand(self.num_envs, -1),
                abad_velocity,
            ),
            dim=-1,
        )
        if candidate.shape[-1] != 36:
            raise RuntimeError(f"Sensor V2 frame construction produced {candidate.shape[-1]} values")
        if not bool(torch.isfinite(candidate).all()):
            raise RuntimeError("Sensor V2 simulator event preprocessing produced non-finite values")

        # get_observations may be called more than once for the same physics step.
        # Only a new causal sample advances velocities and history.
        frame = torch.where(new_sample.unsqueeze(-1), candidate, self._sensor_frame_v2)
        self._sensor_frame_v2 = frame
        self._sensor_prev_main_pos_v2 = torch.where(
            new_sample.unsqueeze(-1), main_pos, self._sensor_prev_main_pos_v2
        )
        self._sensor_prev_abad_pos_v2 = torch.where(
            new_sample.unsqueeze(-1), abad_pos, self._sensor_prev_abad_pos_v2
        )
        self._sensor_prev_root_lin_vel_w_v2 = torch.where(
            new_sample.unsqueeze(-1), root_lin_vel_w, self._sensor_prev_root_lin_vel_w_v2
        )
        self._sensor_encoder_initialized_v2 |= new_sample
        self._sensor_last_sample_step_v2 = torch.where(
            new_sample, sample_step, self._sensor_last_sample_step_v2
        )
        return frame, new_sample

    def _append_sensor_history_v2(self, frame: torch.Tensor, new_sample: torch.Tensor) -> None:
        if not bool(new_sample.any()):
            return
        ids = torch.nonzero(new_sample, as_tuple=False).flatten()
        history = self._sensor_history_v2[ids]
        history = torch.roll(history, shifts=-1, dims=1)
        history[:, -1, :] = frame[ids]

        first_sample = self._sensor_history_valid_count_v2[ids] == 0
        if bool(first_sample.any()):
            first_ids = torch.nonzero(first_sample, as_tuple=False).flatten()
            history[first_ids] = frame[ids[first_ids]].unsqueeze(1).expand(-1, history.shape[1], -1)

        self._sensor_history_v2[ids] = history
        self._sensor_history_valid_count_v2[ids] = torch.clamp(
            self._sensor_history_valid_count_v2[ids] + 1,
            max=int(self.cfg.sensor_history_length),
        )
        self._sensor_history_ready_v2[ids] = (
            self._sensor_history_valid_count_v2[ids] >= int(self.cfg.sensor_history_length)
        )

    def _teacher_physical_v2(self, sensor_frame: torch.Tensor) -> torch.Tensor:
        teacher = torch.cat(
            (
                sensor_frame,
                self.commands,
                self.base_lin_vel,
                self.robot.data.root_pos_w[:, 2:3],
                self._main_strength_scale_per_leg,
                self._abad_strength_scale_per_leg,
                self._fault_mask.float(),
                self._mass_scale.unsqueeze(-1),
                self._friction_scale.unsqueeze(-1),
                self._terrain_level.unsqueeze(-1),
                self._push_events_step.unsqueeze(-1),
            ),
            dim=-1,
        )
        if teacher.shape[-1] != 65:
            raise RuntimeError(f"Teacher A V2 must be 65-D, got {teacher.shape[-1]}")
        return torch.nan_to_num(teacher, nan=0.0, posinf=10.0, neginf=-10.0)

    def _get_observations(self) -> dict[str, torch.Tensor]:
        self._update_state()
        sensor_frame, new_sample = self._build_sensor_frame_v2()
        self._append_sensor_history_v2(sensor_frame, new_sample)

        history = self._sensor_history_v2
        command = self.commands.clone()
        teacher_physical = self._teacher_physical_v2(sensor_frame)
        teacher_ablation = torch.cat(
            (
                teacher_physical,
                self._target_drive_vel,
                self._target_abad_pos - self._abad_rest_pos.expand(self.num_envs, -1),
            ),
            dim=-1,
        )
        policy_compat = torch.cat((history.reshape(self.num_envs, -1), command), dim=-1)

        return {
            "policy": policy_compat,
            "sensor_history_v2": history,
            "sensor_frame_v2": sensor_frame,
            "command_v2": command,
            "teacher": teacher_physical,
            "teacher_physical_v2": teacher_physical,
            "teacher_internal_target_ablation_v2": teacher_ablation,
            "critic": teacher_physical,
            "critic_privileged_v2": teacher_physical,
            "aux_base_vel_target": self.base_lin_vel.clone(),
            "history_ready_v2": self._sensor_history_ready_v2.float().unsqueeze(-1),
        }

    def _compute_simplified_rewards(self) -> torch.Tensor:
        reward = super()._compute_simplified_rewards()
        sensitivity = str(self.cfg.v2_regularizer_sensitivity)
        multipliers = dict(self.cfg.v2_regularizer_sensitivity_multipliers)
        if sensitivity not in multipliers:
            raise ValueError(f"unknown V2 regularizer sensitivity {sensitivity!r}")
        multiplier = float(multipliers[sensitivity])

        gravity_alignment = torch.sum(
            self.projected_gravity * self.reference_projected_gravity, dim=-1
        ).clamp(-1.0, 1.0)
        tilt = torch.acos(gravity_alignment)
        lateral = torch.abs(self.base_lin_vel[:, 1])
        yaw = torch.abs(self.base_ang_vel[:, 2])
        delta_action = self.actions[:, :6] - self.last_actions[:, :6]
        smoothness = torch.mean(torch.square(delta_action), dim=-1)
        saturation_start = float(self.cfg.v2_action_saturation_start)
        saturation = torch.mean(
            torch.square(torch.relu(torch.abs(self.actions[:, :6]) - saturation_start)), dim=-1
        )

        penalties = {
            "rew_sensor_v2_tilt": -tilt * float(self.cfg.v2_tilt_penalty),
            "rew_sensor_v2_lateral": -lateral * float(self.cfg.v2_lateral_velocity_penalty),
            "rew_sensor_v2_yaw": -yaw * float(self.cfg.v2_yaw_velocity_penalty),
            "rew_sensor_v2_smoothness": -smoothness
            * float(self.cfg.v2_action_smoothness_penalty)
            * multiplier,
            "rew_sensor_v2_saturation": -saturation
            * float(self.cfg.v2_action_saturation_penalty)
            * multiplier,
        }
        for name, value in penalties.items():
            reward = reward + value
            self.episode_sums[name] += value
        self.episode_sums["diag_sensor_v2_history_ready"] += self._sensor_history_ready_v2.float()
        return reward

    def _reset_idx(self, env_ids: Sequence[int]) -> None:
        super()._reset_idx(env_ids)
        ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        self._sensor_history_v2[ids] = 0.0
        self._sensor_history_valid_count_v2[ids] = 0
        self._sensor_history_ready_v2[ids] = False
        self._sensor_encoder_initialized_v2[ids] = False
        self._sensor_last_sample_step_v2[ids] = -1
        self._sensor_frame_v2[ids] = 0.0
        self._sensor_prev_main_pos_v2[ids] = self.joint_pos[ids][:, self._main_drive_indices]
        self._sensor_prev_abad_pos_v2[ids] = self.joint_pos[ids][:, self._abad_indices]
        self._sensor_prev_root_lin_vel_w_v2[ids] = self.robot.data.root_lin_vel_w[ids]
        self._sensor_gravity_estimate_v2[ids] = _normalize_vector(
            self.reference_projected_gravity[ids]
        )
