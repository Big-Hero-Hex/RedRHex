---
id: policy-contract-reference
title: Policy and Deployment Contract
lang: en
audience: shared
type: reference
status: active
owner: deployment
last_reviewed: 2026-08-14
---

<a id="dimensions"></a>
## Dimensions and rate

The following is the legacy V1 contract and remains the default for legacy tasks and ROS YAML:

- Single observation frame: 56 values
- Policy history: 5 frames, 280 values
- Actions: 12 values
- Simulation step: `1/120 s`
- Decimation: `2`
- Policy/control rate: `60 Hz`

<a id="observation-order"></a>
## Observation order

| Slice | Values |
| --- | ---: |
| base linear velocity | 0–2 |
| base angular velocity | 3–5 |
| projected gravity | 6–8 |
| main-drive position sine/cosine | 9–20 |
| scaled main-drive velocity | 21–26 |
| scaled ABAD position | 27–32 |
| ABAD velocity | 33–38 |
| velocity command | 39–41 |
| gait phase sine/cosine | 42–43 |
| previous actions | 44–55 |

<a id="sensor-v2-contract"></a>
## Sensor-only V2 contract

V2 is selected only by `Template-Redrhex-ForwardSensorV2-Direct-v0` and a V2 runner/ROS route. Its sensor frame is 36 physical-feedback values: body gyro `0:3`, projected gravity `3:6`, six main sine values `6:12`, six main cosine values `12:18`, six main velocities `18:24`, six measured neutral-relative ABAD positions `24:30`, and six measured ABAD velocities `30:36`.

The actor receives `sensor_history [1,60,36]` in oldest-to-newest order and a separate current `command [1,3]`. It never receives base velocity, gait clock, previous action, odometry, commanded ABAD feedback, controller target, or privilege. The fixed ONNX outputs are `actions [1,12]` and `base_velocity_estimate [1,3]`. Contact output is unavailable.

<a id="actions"></a>
## Actions and joints

Actions control six main-drive joints followed by six ABAD joints. The six damper joints are passive and are not commanded on hardware. Exact joint names and stage-5 scaling constants are defined in `redrhex_contract.py` and checked against the deployment YAML and training configuration.

For V2 forward F0–F5, the first six outputs are learned residuals around a versioned procedural CPG and the final six outputs are forced neutral. Decoder semantics and both observation/action SHA-256 values are stored in every V2 checkpoint and deployment bundle.

<a id="commands"></a>
## Command envelope

The mirrored deployment limits are `vx` from `0.0` to `0.56 m/s`, `vy` from `-0.60` to `0.60 m/s`, and yaw rate from `-0.70` to `0.70 rad/s`. A specific training stage may sample a narrower range.

<a id="change-rule"></a>
## Change rule

Any dimension, order, normalization, rate, stage scaling, joint mapping, command envelope, or frame change requires training, export, panel readiness, ONNX, ROS, and hardware-preflight review. Update parity tests and both operator/developer documentation in the same change.

V2 additionally requires an exact contract, action-decoder, and calibration hash match. Changing the explicit IMU attitude mode retrains and re-exports the bundle. Hardware V2 remains blocked until one attitude mode and all twelve encoder calibrations are supported by reviewed recorded evidence.
