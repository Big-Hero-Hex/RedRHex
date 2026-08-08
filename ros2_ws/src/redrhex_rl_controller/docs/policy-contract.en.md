---
id: ros2-policy-contract
title: ROS 2 policy contract
lang: en
audience: developer
type: reference
status: active
owner: deployment
last_reviewed: 2026-08-07
---

<a id="ros2-policy-contract"></a>
# ROS 2 policy contract

The controller contract must match the exported policy and the training environment. Do not infer dimensions, joint order, scale, or rate from another robot.

<a id="tensor-contract"></a>
## Tensor contract

| Property | Required value |
| --- | --- |
| Single observation | 56 floats |
| Policy history | 5 observations / 280 floats |
| Policy action | 12 floats |
| Simulation step | 1/120 s |
| Decimation | 2 |
| Policy rate | 60 Hz |

The detailed 56-value observation ordering is defined in the shared [policy contract](../../../../docs/reference/policy-contract.en.md). `policy_hz: 0.0` means derive 60 Hz from the repository contract; it does not mean inference is disabled.

<a id="actuator-contract"></a>
## Actuator contract

The output controls six main-drive motors and six ABAD motors. Damper joints are simulation-only passive joints and must not be added to the hardware command. Signs and zero offsets remain identity until single-joint tests prove a correction.

Default limits in `config/redrhex_policy.yaml` include a normalized action clip of 1.0, main-drive velocity limit of 30 rad/s, ABAD position limit of 0.7 rad, and slew-rate limits. Treat the YAML beside the deployed package as the runtime source of truth.

<a id="command-and-feedback-contract"></a>
## Command and feedback contract

Default command bounds are `vx` 0.0–0.56 m/s, `vy` -0.60–0.60 m/s, and `wz` -0.70–0.70 rad/s. The default base-linear-velocity source is `zero`; switching to odometry or an estimator requires contract validation.

Main-drive encoder position is required for current hardware. Missing main-drive velocity can be estimated. ABAD position may be sourced from the last command only when configured; this must remain visible as an evidence limitation.

<a id="validation-tools"></a>
## Validation tools

Use `scripts/check_onnx_io.py` for graph I/O and `scripts/compare_onnx_with_torch.py` for Torch/ONNX parity. The Training Panel [deployment readiness check](../../../../tools/training_panel/docs/deploy-readiness.en.md) combines those checks with contract and safety evidence.
