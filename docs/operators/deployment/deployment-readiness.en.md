---
id: operator-deployment-readiness
title: Validate a Policy for Deployment
lang: en
audience: operator
type: safety
status: active
owner: deployment
last_reviewed: 2026-08-07
---

<a id="gate"></a>
## Deployment gate

Never move directly from a successful training run to powered hardware. Select one `model_*.pt`, export it, and keep its configuration and readiness report together. A `blocked` readiness result stops deployment; `review` means warnings or missing optional evidence still require human review.

<a id="panel-check"></a>
## Run panel readiness

Open the Training Panel Deploy page, select the completed run, and choose Validate Existing ONNX or Export ONNX + Validate. Required stages cover export integrity, static ONNX, ONNX Runtime, Torch/ONNX parity, the observation/action contract, and safety fault injection. ROS mock and uncalibrated MuJoCo results are supporting evidence, not permission to power the robot.

Reports are written under `<run>/deploy/readiness_<pipeline-id>.json` and `.md`.

<a id="jetson-preflight"></a>
## Run Jetson preflight

Copy the reviewed ONNX, deployment YAML, and readiness report. On Jetson:

```bash
source /opt/ros/humble/setup.bash
source /path/to/ros2_ws/install/setup.bash
ros2 run redrhex_rl_controller preflight_check \
  --onnx /path/to/policy.onnx \
  --config /path/to/redrhex_policy.yaml
```

The current contract is 56 single-frame observations or 280 with history, 12 actions, and a 60 Hz policy loop.

<a id="hardware-boundary"></a>
## Hardware boundary

Before any motor enable, prepare a physical E-stop and power cutoff, validate one bridge publisher, verify heartbeat, keep `enable_policy_on_start=false` and `enable_motor_output_on_start=false`, test one ABAD at low power, and test one main drive with the robot lifted. Continue with the colocated [ROS bring-up guide](../../../ros2_ws/src/redrhex_rl_controller/docs/bring-up.en.md).
