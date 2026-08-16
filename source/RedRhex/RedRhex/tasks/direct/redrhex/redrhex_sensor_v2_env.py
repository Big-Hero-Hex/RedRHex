# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Sensor-only V2 environment layered on the proven forward controller."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch

from isaaclab.utils.math import quat_apply_inverse
from redrhex_policy_io import (
    build_sensor_frame_torch,
    decode_forward_residual_action_v2_torch,
)

from .redrhex_env import RedrhexEnv
from .redrhex_sensor_v2_env_cfg import RedrhexForwardSensorV2EnvCfg
from .sensor_v2_action import forward_residual_action_contract_v2_from_config
from .sensor_domain_randomization_v2 import (
    SensorDomainRandomizationV2Config,
    SensorDomainRandomizerV2,
    real_history_sample_mask_v2,
)


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

        self._action_contract_v2 = forward_residual_action_contract_v2_from_config(self.cfg)
        self._v2_raw_contract_target_drive_vel = torch.zeros(
            self.num_envs, self.num_main_drive_joints, device=self.device
        )
        self._v2_raw_contract_target_abad_pos = torch.zeros(
            self.num_envs, self.num_abad_joints, device=self.device
        )
        self._v2_simulator_target_drive_vel = torch.zeros_like(
            self._v2_raw_contract_target_drive_vel
        )
        self._v2_contract_gait_phase_rad = torch.zeros(self.num_envs, device=self.device)
        self._v2_motion_step = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )

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
        self._sensor_prev_encoder_sample_step_v2 = torch.full(
            (self.num_envs,), -1, dtype=torch.long, device=self.device
        )
        self._sensor_last_sample_step_v2 = torch.full(
            (self.num_envs,), -1, dtype=torch.long, device=self.device
        )
        self._sensor_frame_v2 = torch.zeros(self.num_envs, frame_dim, device=self.device)
        self._sensor_prev_root_lin_vel_w_v2 = torch.zeros(self.num_envs, 3, device=self.device)
        self._sensor_gravity_estimate_v2 = _normalize_vector(self.reference_projected_gravity.clone())

        sensor_dr_config = SensorDomainRandomizationV2Config.from_namespace(self.cfg)
        base_seed = int(getattr(self.cfg, "seed", 0) or 0)
        self._sensor_dr_v2 = SensorDomainRandomizerV2(
            sensor_dr_config,
            num_envs=self.num_envs,
            num_main_joints=self.num_main_drive_joints,
            num_abad_joints=self.num_abad_joints,
            sample_hz=float(self.cfg.sensor_sample_hz),
            device=self.device,
            seed=base_seed + int(getattr(self.cfg, "sensor_dr_seed_offset", 0)),
        )

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
            "diag_sensor_dr_encoder_stale",
            "diag_sensor_dr_encoder_dropout",
        ):
            self.episode_sums[key] = torch.zeros(self.num_envs, device=self.device)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        if actions.shape[-1] != 12:
            raise ValueError(f"Sensor V2 action contract requires 12 actions, got {actions.shape[-1]}")
        strict_actions = actions.clone()
        if bool(getattr(self.cfg, "strict_forward_residual_actions", True)):
            strict_actions[..., 6:12] = 0.0
        # A newly reset policy has no causal cycle of evidence yet.  Keep the
        # learned residual at zero until all 60 slots contain real samples;
        # the deterministic forward controller remains responsible for warmup.
        if bool(getattr(self.cfg, "sensor_history_action_gate", True)):
            strict_actions = torch.where(
                self._sensor_history_ready_v2.unsqueeze(-1),
                strict_actions,
                torch.zeros_like(strict_actions),
            )
        super()._pre_physics_step(strict_actions)

    def _apply_action(self) -> None:
        """Apply the shared Torch action contract, then simulator-only perturbations."""

        self._apply_explicit_torsion_spring()
        if getattr(self, "_action_targets_computed", False):
            self._apply_sim2real_command_delay(
                self._requested_target_drive_vel,
                self._requested_target_abad_pos,
            )
            return

        main_position = self.joint_pos[:, self._main_drive_indices]
        contract_phase = self._v2_contract_gait_phase_rad
        self._v2_contract_gait_phase_rad = contract_phase
        decoded = decode_forward_residual_action_v2_torch(
            self.actions,
            self.commands,
            contract_phase,
            main_position,
            control_step=self._v2_motion_step,
            contract=self._action_contract_v2,
        )
        raw_drive = decoded.target_main_velocity_rad_s
        raw_abad = decoded.target_abad_position_rad
        self._v2_raw_contract_target_drive_vel = raw_drive.clone()
        self._v2_raw_contract_target_abad_pos = raw_abad.clone()
        self._base_velocity = decoded.nominal_main_velocity_rad_s.clone()
        self._action_warmup_scale = decoded.action_warmup_scale.clone()

        # Advance the hidden procedural clock only while a strict forward
        # command is active.  Command magnitude scales both physical velocity
        # and phase progress; the same startup ramp prevents the reference
        # angle from running ahead of the legs.  This state is controller-known
        # and never enters the student observation.
        command_scale = torch.clamp(
            self.commands[:, 0]
            / self._action_contract_v2.forward_command_reference_m_s,
            min=0.0,
            max=1.0,
        )
        moving = self.commands[:, 0] > self._action_contract_v2.FORWARD_ACTIVE_MIN_M_S
        phase_increment = (
            2.0
            * math.pi
            * self._action_contract_v2.NOMINAL_GAIT_FREQUENCY_HZ
            / self._action_contract_v2.policy_rate_hz
            * command_scale
            * decoded.action_warmup_scale
        )
        next_phase = torch.remainder(contract_phase + phase_increment, 2.0 * math.pi)
        self._v2_contract_gait_phase_rad = torch.where(
            moving, next_phase, contract_phase
        )
        self._v2_motion_step = torch.where(
            moving,
            self._v2_motion_step + 1,
            torch.zeros_like(self._v2_motion_step),
        )
        self.gait_phase = self._v2_contract_gait_phase_rad.clone()

        effective_position = main_position * self._direction_multiplier
        leg_phase = torch.remainder(effective_position, 2.0 * math.pi)
        self._current_leg_in_stance = self._in_stance_phase(leg_phase)
        self._contact_count = self._current_leg_in_stance.float().sum(dim=1)
        mode_fwd, mode_lat, mode_diag, mode_yaw, mode_id = self._resolve_command_modes()
        self._mode_fwd = mode_fwd
        self._mode_lat = mode_lat
        self._mode_diag = mode_diag
        self._mode_yaw = mode_yaw
        self._mode_id = mode_id
        self._is_pure_lateral = mode_lat

        # Physical-domain perturbations belong after the raw action contract.
        # The named raw buffer is the simulator/ROS parity target; this final
        # buffer is the potentially perturbed actuator target used by Isaac.
        simulator_drive = self._compute_main_drive_targets(
            raw_drive,
            self._action_contract_v2.main_velocity_limit_rad_s,
        )
        self._v2_simulator_target_drive_vel = simulator_drive.clone()
        self._apply_sim2real_command_delay(simulator_drive, raw_abad)
        self._action_targets_computed = True

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

    def _specific_force_body(
        self, root_quat: torch.Tensor, root_lin_vel_w: torch.Tensor
    ) -> torch.Tensor:
        dt = 1.0 / float(self.cfg.sensor_sample_hz)
        world_accel = (root_lin_vel_w - self._sensor_prev_root_lin_vel_w_v2) / dt
        gravity_mps2_w = self._gravity_vec_w * 9.81
        return quat_apply_inverse(root_quat, world_accel - gravity_mps2_w)

    def _estimated_projected_gravity(
        self,
        gyro_body: torch.Tensor,
        specific_force_body: torch.Tensor,
        new_sample: torch.Tensor,
    ) -> torch.Tensor:
        """Causal gyro propagation with gated accelerometer correction."""

        dt = 1.0 / float(self.cfg.sensor_sample_hz)
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
        if not bool(torch.isfinite(gyro_body).all()):
            raise RuntimeError("Sensor V2 simulated gyro contains NaN or Inf")

        sample_step = torch.full(
            (self.num_envs,),
            int(self._sim_step_counter // max(int(self.cfg.decimation), 1)),
            dtype=torch.long,
            device=self.device,
        )
        new_sample = sample_step != self._sensor_last_sample_step_v2

        main_pos = self.joint_pos[:, self._main_drive_indices]
        abad_pos = self.joint_pos[:, self._abad_indices]
        mode = str(self.cfg.sensor_attitude_mode)
        if mode == "validated_quaternion":
            raw_gravity_body = self._validated_projected_gravity(root_quat)
            event = self._sensor_dr_v2.process(
                gyro_body=gyro_body,
                gravity_body=raw_gravity_body,
                main_position=main_pos,
                abad_position=abad_pos,
                new_sample=new_sample,
            )
            if event.gravity_body is None:
                raise RuntimeError("Sensor V2 randomizer omitted validated gravity")
            gravity_body = event.gravity_body
        elif mode == "causal_gyro_accel":
            raw_specific_force_body = self._specific_force_body(root_quat, root_lin_vel_w)
            event = self._sensor_dr_v2.process(
                gyro_body=gyro_body,
                specific_force_body=raw_specific_force_body,
                main_position=main_pos,
                abad_position=abad_pos,
                new_sample=new_sample,
            )
            if event.specific_force_body is None:
                raise RuntimeError("Sensor V2 randomizer omitted causal accelerometer data")
            gravity_body = self._estimated_projected_gravity(
                event.gyro_body, event.specific_force_body, new_sample
            )
        else:
            raise ValueError(
                "sensor_attitude_mode must be validated_quaternion or causal_gyro_accel; "
                f"got {mode!r}"
            )

        gyro_body = event.gyro_body
        main_pos = event.main_position
        abad_pos = event.abad_position
        accepted_sample = event.accepted_sample
        history_sample = real_history_sample_mask_v2(
            accepted_sample,
            self._sensor_encoder_initialized_v2,
        )
        elapsed_steps = torch.where(
            self._sensor_prev_encoder_sample_step_v2 >= 0,
            sample_step - self._sensor_prev_encoder_sample_step_v2,
            torch.ones_like(sample_step),
        ).clamp(min=1)
        elapsed_s = elapsed_steps.float().unsqueeze(-1) / float(self.cfg.sensor_sample_hz)
        main_velocity = _wrapped_difference(main_pos, self._sensor_prev_main_pos_v2) / elapsed_s
        abad_velocity = (abad_pos - self._sensor_prev_abad_pos_v2) / elapsed_s
        initialized = self._sensor_encoder_initialized_v2.unsqueeze(-1)
        main_velocity = torch.where(initialized, main_velocity, torch.zeros_like(main_velocity))
        abad_velocity = torch.where(initialized, abad_velocity, torch.zeros_like(abad_velocity))

        # Use the same feature composer exported to replay and ROS.  Causal
        # event generation stays here; feature ordering and neutral subtraction
        # have a single implementation.
        candidate = build_sensor_frame_torch(
            gyro_body,
            gravity_body,
            main_pos,
            main_velocity,
            abad_pos,
            abad_velocity,
            abad_neutral_position_rad=self._abad_rest_pos.squeeze(0),
        )
        if candidate.shape[-1] != 36:
            raise RuntimeError(f"Sensor V2 frame construction produced {candidate.shape[-1]} values")
        if not bool(torch.isfinite(candidate).all()):
            raise RuntimeError("Sensor V2 simulator event preprocessing produced non-finite values")

        # get_observations may be called more than once for the same physics step.
        # The first accepted encoder event primes the finite-difference baseline;
        # only later accepted events are eligible for real-sample history.
        frame = torch.where(accepted_sample.unsqueeze(-1), candidate, self._sensor_frame_v2)
        self._sensor_frame_v2 = frame
        self._sensor_prev_main_pos_v2 = torch.where(
            accepted_sample.unsqueeze(-1), main_pos, self._sensor_prev_main_pos_v2
        )
        self._sensor_prev_abad_pos_v2 = torch.where(
            accepted_sample.unsqueeze(-1), abad_pos, self._sensor_prev_abad_pos_v2
        )
        self._sensor_prev_encoder_sample_step_v2 = torch.where(
            accepted_sample, sample_step, self._sensor_prev_encoder_sample_step_v2
        )
        self._sensor_prev_root_lin_vel_w_v2 = torch.where(
            new_sample.unsqueeze(-1), root_lin_vel_w, self._sensor_prev_root_lin_vel_w_v2
        )
        self._sensor_encoder_initialized_v2 |= accepted_sample
        self._sensor_last_sample_step_v2 = torch.where(
            new_sample, sample_step, self._sensor_last_sample_step_v2
        )
        # A missing or replayed physical encoder generation invalidates the causal window;
        # production ROS applies the same fail-closed reset.  Keeping an old
        # ready history here would train/evaluate stale-policy continuation that
        # can never occur in deployment.  The next accepted generation primes a
        # fresh finite-difference baseline and is not appended to history.
        invalid_sample = new_sample & (event.encoder_dropout | event.encoder_stale)
        if bool(invalid_sample.any()):
            invalid_ids = torch.nonzero(invalid_sample, as_tuple=False).flatten()
            self._sensor_history_v2[invalid_ids] = 0.0
            self._sensor_history_valid_count_v2[invalid_ids] = 0
            self._sensor_history_ready_v2[invalid_ids] = False
            self._sensor_encoder_initialized_v2[invalid_ids] = False
            self._sensor_prev_main_pos_v2[invalid_ids] = 0.0
            self._sensor_prev_abad_pos_v2[invalid_ids] = 0.0
            self._sensor_prev_encoder_sample_step_v2[invalid_ids] = -1
        self.episode_sums["diag_sensor_dr_encoder_stale"] += event.encoder_stale.float()
        self.episode_sums["diag_sensor_dr_encoder_dropout"] += event.encoder_dropout.float()
        return frame, history_sample

    def _append_sensor_history_v2(self, frame: torch.Tensor, new_sample: torch.Tensor) -> None:
        if not bool(new_sample.any()):
            return
        ids = torch.nonzero(new_sample, as_tuple=False).flatten()
        history = self._sensor_history_v2[ids]
        history = torch.roll(history, shifts=-1, dims=1)
        history[:, -1, :] = frame[ids]

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

        # Hand ownership to the caller.  Runners step the environment before
        # storing rollout labels, so exposing this mutable ring would alias H_t
        # with H_{t+1} and silently corrupt distillation trajectories.
        history = self._sensor_history_v2.clone()
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

    def sensor_dr_sampled_statistics_v2(self) -> dict[str, float]:
        """Expose the sampled Sensor V2 perturbation statistics for diagnostics."""

        return self._sensor_dr_v2.sampled_statistics()

    def _reset_idx(self, env_ids: Sequence[int]) -> None:
        super()._reset_idx(env_ids)
        if bool(
            getattr(self.cfg, "sensor_dr_require_physical_material_writes", False)
        ):
            missing_physical_writes: list[str] = []
            if bool(getattr(self.cfg, "dr_randomize_mass", False)) and not bool(
                self._mass_physical_randomized
            ):
                missing_physical_writes.append("mass")
            if bool(getattr(self.cfg, "dr_randomize_friction", False)) and not bool(
                self._friction_physical_randomized
            ):
                missing_physical_writes.append("friction")
            if missing_physical_writes:
                raise RuntimeError(
                    "Sensor V2 F4/F5 requires verified PhysX material writes; "
                    "controller-target fallback is forbidden for: "
                    + ", ".join(missing_physical_writes)
                )
        ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        reset_main_position = torch.as_tensor(
            self._action_contract_v2.initial_main_position_rad,
            dtype=self.joint_pos.dtype,
            device=self.device,
        ).unsqueeze(0).expand(ids.numel(), -1)
        reset_main_velocity = torch.zeros_like(reset_main_position)
        # The legacy reset intentionally retains small joint noise.  V2 removes
        # it only for the six main drives and writes the contract-hashed pose
        # back to both PhysX and the environment's cloned state buffers.
        self.robot.write_joint_state_to_sim(
            reset_main_position,
            reset_main_velocity,
            joint_ids=self._main_drive_indices,
            env_ids=ids,
        )
        self.joint_pos[ids[:, None], self._main_drive_indices] = reset_main_position
        self.joint_vel[ids[:, None], self._main_drive_indices] = reset_main_velocity
        # The V2 action contract has a deterministic clock origin shared with
        # deployment.  Legacy tasks retain the base environment's random phase.
        reset_gait_phase = float(self.cfg.v2_reset_gait_phase_rad)
        self.gait_phase[ids] = reset_gait_phase
        self._v2_raw_contract_target_drive_vel[ids] = 0.0
        self._v2_raw_contract_target_abad_pos[ids] = 0.0
        self._v2_simulator_target_drive_vel[ids] = 0.0
        self._v2_contract_gait_phase_rad[ids] = reset_gait_phase
        self._v2_motion_step[ids] = 0
        self._sensor_dr_v2.reset(ids)
        self._sensor_history_v2[ids] = 0.0
        self._sensor_history_valid_count_v2[ids] = 0
        self._sensor_history_ready_v2[ids] = False
        self._sensor_encoder_initialized_v2[ids] = False
        self._sensor_prev_encoder_sample_step_v2[ids] = -1
        self._sensor_last_sample_step_v2[ids] = -1
        self._sensor_frame_v2[ids] = 0.0
        self._sensor_prev_main_pos_v2[ids] = self.joint_pos[ids][:, self._main_drive_indices]
        self._sensor_prev_abad_pos_v2[ids] = self.joint_pos[ids][:, self._abad_indices]
        self._sensor_prev_root_lin_vel_w_v2[ids] = self.robot.data.root_lin_vel_w[ids]
        self._sensor_gravity_estimate_v2[ids] = _normalize_vector(
            self.reference_projected_gravity[ids]
        )
        if self._sensor_dr_v2.config.enabled:
            self.extras.setdefault("log", {})[
                "SensorDR/physical_mass_applied"
            ] = float(bool(self._mass_physical_randomized))
            self.extras.setdefault("log", {})[
                "SensorDR/physical_friction_applied"
            ] = float(bool(self._friction_physical_randomized))
            for name, value in self._sensor_dr_v2.sampled_statistics(ids).items():
                self.extras.setdefault("log", {})[f"SensorDR/{name}"] = value
