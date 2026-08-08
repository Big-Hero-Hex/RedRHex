---
id: policy-contract-reference
title: Policy and Deployment Contract
lang: en
audience: shared
type: reference
status: active
owner: deployment
last_reviewed: 2026-08-07
---

<a id="dimensions"></a>
## Dimensions and rate

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

<a id="actions"></a>
## Actions and joints

Actions control six main-drive joints followed by six ABAD joints. The six damper joints are passive and are not commanded on hardware. Exact joint names and stage-5 scaling constants are defined in `redrhex_contract.py` and checked against the deployment YAML and training configuration.

<a id="commands"></a>
## Command envelope

The mirrored deployment limits are `vx` from `0.0` to `0.56 m/s`, `vy` from `-0.60` to `0.60 m/s`, and yaw rate from `-0.70` to `0.70 rad/s`. A specific training stage may sample a narrower range.

<a id="change-rule"></a>
## Change rule

Any dimension, order, normalization, rate, stage scaling, joint mapping, command envelope, or frame change requires training, export, panel readiness, ONNX, ROS, and hardware-preflight review. Update parity tests and both operator/developer documentation in the same change.
