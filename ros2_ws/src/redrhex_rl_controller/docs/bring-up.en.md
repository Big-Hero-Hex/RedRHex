---
id: ros2-safe-bring-up
title: Bring up the ROS 2 deployment stack
lang: en
audience: operator
type: how-to
status: active
owner: deployment
last_reviewed: 2026-08-07
---

<a id="bring-up-the-ros2-deployment-stack"></a>
# Bring up the ROS 2 deployment stack

This procedure proves the software graph without granting motor authority. Keep the robot unpowered or physically isolated during the first graph test.

<a id="prerequisites"></a>
## Prerequisites

- ROS 2 Humble and the repository dependencies are installed.
- The exported ONNX artifact passes the [deployment readiness check](../../../../tools/training_panel/docs/deploy-readiness.en.md).
- A physical E-stop, current limiting, and a stable lifted fixture are available before hardware work.

<a id="build-and-source"></a>
## Build and source

```bash
source /opt/ros/humble/setup.bash
cd ros2_ws
colcon build --symlink-install
source install/setup.bash
```

For Rinbo/Biorola hardware, source its workspace before this repository workspace so the controller and low-level bridge can discover the hardware messages.

<a id="start-a-mock-graph"></a>
## Start a mock graph

Launch with fake sensors and a mock bridge. Leave both startup enable flags false:

```bash
ros2 launch redrhex_rl_controller redrhex_policy_bringup.launch.py \
  use_fake_sensors:=true \
  start_bridge:=true \
  bridge_backend:=mock \
  enable_policy_on_start:=false \
  enable_motor_output_on_start:=false
```

In a second terminal, source the workspace and inspect:

```bash
ros2 topic echo /redrhex/state_machine_state
ros2 topic echo /redrhex/diagnostics
ros2 topic echo /redrhex/motor_commands
```

Confirm that observations are fresh, inference dimensions are accepted, and motor commands remain disabled.

<a id="prove-the-safety-controls"></a>
## Prove the safety controls

```bash
ros2 run redrhex_rl_controller estop_tool assert
ros2 run redrhex_rl_controller estop_tool clear --confirm-clear
```

The clear operation intentionally requires confirmation. A ROS E-stop is only a high-level protection input; it never replaces the physical E-stop or lower-level watchdog.

<a id="exit-criteria"></a>
## Exit criteria

Proceed to [hardware deployment](deployment.en.md) only after the mock graph remains healthy, policy and motor enables are independently gated, E-stop assertion drops the latches, and no unexpected publisher can command motors.
