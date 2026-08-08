---
id: ros2-deployment-architecture
title: ROS 2 deployment architecture
lang: en
audience: developer
type: explanation
status: active
owner: deployment
last_reviewed: 2026-08-07
---

<a id="ros2-deployment-architecture"></a>
# ROS 2 deployment architecture

The deployment stack separates learned inference, safety decisions, and hardware transport so an exported action is never sent directly to a motor driver.

<a id="data-path"></a>
## Data path

```text
IMU + joint states + command
  -> ObservationBuilder
  -> PolicyONNXRunner
  -> ActionDecoder
  -> SafetyFilter + RedRhexStateMachine
  -> /redrhex/motor_commands
  -> redrhex_lowlevel_bridge
  -> low-level board and motor drivers
```

`rl_controller_node` owns the high-level path and publishes observation, raw-action, safe-action, motor-command, state, and diagnostic topics. A low-level bridge owns the hardware protocol. Run only one publisher that can reach the physical motor-command topic.

<a id="package-boundaries"></a>
## Package boundaries

- `ObservationBuilder` constructs the trained observation contract and its five-frame history.
- `PolicyONNXRunner` validates ONNX input/output shapes and performs inference.
- `ActionDecoder` maps 12 normalized outputs into six main-drive and six ABAD targets.
- `SafetyFilter` clips commands and enforces configured motion limits.
- `RedrhexStateMachine` controls readiness, warm-up, policy, motor-output, and protective-stop gates.
- `redrhex_lowlevel_bridge` translates the safe command into `mock`, `serial`, `sbrio_udp`, or Rinbo/Biorola transport.

The six simulated damper joints are passive spring mechanisms. They have neither policy actions nor physical motor commands.

<a id="interfaces"></a>
## Interfaces

The controller subscribes to `/imu/data`, `/joint_states`, `/cmd_vel`, `/estop`, enable/recovery topics, low-level heartbeat, motor feedback, battery state, and optional odometry. It publishes under `/redrhex/` including `observation`, `policy_action_raw`, `policy_action_safe`, `motor_commands`, `state_machine_state`, and `diagnostics`.

The launch file can start fake sensors and a selected bridge for dry-run tests. Both policy and motor output default to disabled.

<a id="hardware-feedback-assumptions"></a>
## Hardware feedback assumptions

The current MVP assumes true IMU data and six main-drive encoder positions. It can estimate main-drive velocity by differentiation and, with `abad_feedback_source: commanded`, substitutes the previous ABAD command when physical ABAD feedback is unavailable. That substitution is a documented sim-to-real limitation, not equivalent to an encoder.

<a id="evolution-boundary"></a>
## Evolution boundary

Keep the bridge abstraction until the low-level protocol is stable. A future `ros2_control` hardware plugin can replace transport without changing the learned-policy contract, but adopting it is not current behavior.
