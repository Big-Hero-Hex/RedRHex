---
id: student-distillation-v2-audit
title: Sensor-Only Student Distillation V2 Code-Path Audit
lang: en
audience: developer
type: audit
status: published
owner: training
last_reviewed: 2026-08-14
---

<a id="scope"></a>
## Scope

This audit reviews the RedRHex observation, action, RSL-RL, ONNX, ROS 2, encoder-bridge, and sim-to-real paths that constrain an additive sensor-only teacher–student route. It establishes compatibility facts; it does not claim trained-policy, recorded-hardware, or physical-robot results.

<a id="method"></a>
## Method

The review followed the executable paths from Gym registration through environment observation construction, runner selection, checkpoint loading, export, ROS inference, sensor ingestion, and calibration/replay tooling. Installed RSL-RL 3.1.2 source was inspected where the repository delegates behavior upstream. Findings were accepted only when supported by current code, configuration, or an explicit absence check.

<a id="findings"></a>
## Findings

- V1 is a 56-D current frame plus four prior frames (280-D actor input). It includes simulator base linear velocity, a procedural gait clock, and the previous action; the privileged group also includes internal drive and ABAD targets. These semantics cannot be safely repurposed as a hardware sensor contract.
- The existing distillation route is real but is a flat-MLP behavior-cloning path. Upstream RSL-RL executes student actions and optimizes one whole-action MSE/Huber loss; it has no causal TCN, rollout mixture, next-frame target, auxiliary loss, or PPO teacher-BC hook.
- The forward environment already interprets the six main outputs as residuals around a procedural gait and suppresses forward ABAD outputs. That decoder can be versioned for the trial; its phase must remain internal rather than becoming an actor input.
- The simulator contact sensor is disabled. Current “contact” state is derived from encoder phase, so contact supervision and export must remain disabled.
- ROS V1 accepts only one 56- or 280-D ONNX input, zero-fills base velocity, can use commanded ABAD state, zero-pads an incomplete history, and fills history only after policy execution starts. These behaviors are prohibited in V2 but must remain intact for V1 rollback.
- The low-level bridge already receives six raw ABAD encoders but does not publish calibrated ABAD joint feedback. Counts-per-radian and some encoder zeros remain provisional.
- The repository has no production IMU publisher or recorded evidence proving quaternion covariance, frame identity, mount calibration, and rest-gravity behavior. Hardware V2 therefore has no approved attitude mode yet.
- Existing Torch/ONNX parity begins from an already assembled observation, while real-trace import omits ABAD and `cmd_vel`. A shared causal preprocessor and raw-event replay gate are required.
- The initial audit found Training Panel configuration and history discovery tied to V1 experiment roots with no V2 runner selector. A browser implementation was recovered, but is isolated on the stacked Panel physics/calibration proposal branch pending separate review; this core branch does not claim that gap is merged.

<a id="actions"></a>
## Actions

- [x] Preserve V1 registrations, loaders, preprocessing, ROS YAML, and panel routes.
- [x] Approve an additive `Template-Redrhex-ForwardSensorV2-Direct-v0` research route with independently hashed observation and action contracts.
- [ ] Review and merge the recovered kind-checked Panel stage transitions, sequential checkpoint handoff, and final-F3 history routing while preserving standard training.
- [ ] Pass the deterministic forward baseline and three-seed Teacher A, distilled-student, and PPO gates.
- [ ] Prove all twelve encoder calibrations and one explicit IMU attitude mode using reviewed recorded evidence.
- [ ] Run raw-event replay, shared-preprocessor parity, ONNX Runtime parity, and synthetic ROS safety tests before any deployment promotion.

<a id="evidence"></a>
## Evidence

Primary code evidence is in the [legacy environment](../../source/RedRhex/RedRhex/tasks/direct/redrhex/redrhex_env.py), [environment configuration](../../source/RedRhex/RedRhex/tasks/direct/redrhex/redrhex_env_cfg.py), [V2 training backends](../../source/RedRhex/RedRhex/tasks/direct/redrhex/agents/sensor_v2/backends.py), [training entry point](../../scripts/rsl_rl/train.py), [full pipeline](../../scripts/rsl_rl/train_sensor_v2_pipeline.py), [ROS observation builder](../../ros2_ws/src/redrhex_rl_controller/redrhex_rl_controller/observation_builder.py), [ONNX runner](../../ros2_ws/src/redrhex_rl_controller/redrhex_rl_controller/policy_onnx_runner.py), [Rinbo bridge](../../ros2_ws/src/redrhex_lowlevel_bridge/redrhex_lowlevel_bridge/rinbo_ros_backend.py), and [real-trace importer](../../tools/sim2real/import_real.py). The recovered Panel paths are evidence for the stacked proposal, not this core branch.

<a id="follow-up"></a>
## Follow-up

Re-audit when the F5 promotion evidence exists or when any observation feature, attitude mode, action decoder, calibration profile, runner role, or checkpoint transition changes. A one-update F1/F2/F3/pipeline Isaac smoke now exists; production-length, multi-seed, recorded-sensor, and physical tests remain explicitly pending rather than inferred from it or from unit tests.
