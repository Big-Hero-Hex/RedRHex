---
id: system-architecture
title: RedRHex System Architecture
lang: en
audience: developer
type: explanation
status: active
owner: core
last_reviewed: 2026-08-07
---

<a id="layers"></a>
## System layers

RedRHex currently connects four maintained systems:

```text
Training Panel / Reward Agent
            -> train, play, evaluation scripts
            -> RedRHex Isaac Lab task and RSL-RL
            -> checkpoints, events, exports, reports
            -> ROS2 ONNX controller and low-level bridge
```

The panel and reward agent orchestrate existing script interfaces. The `RedRhex` extension owns simulation and training behavior. The ROS2 workspace consumes exported ONNX and mirrors the deployment contract. Sim-to-real tooling creates authenticated evidence and optional explicit physics profiles.

<a id="source-ownership"></a>
## Source ownership

- `source/RedRhex/RedRhex/tasks/direct/redrhex/` owns the Isaac task, environment configuration, behavior, and agent entry points.
- `scripts/rsl_rl/` owns training, playback, staged training, validation, and command-sweep entry points.
- `tools/training_panel/` owns local and remote operations, artifacts, and deployment readiness.
- `tools/reward_agent/` owns bounded reward candidate sessions and trial orchestration.
- `tools/sim2real/` owns characterization evidence, comparisons, profile validation, and promotion gates.
- `ros2_ws/src/` owns deployment messages, policy control, safety, and the hardware bridge.

<a id="stable-interfaces"></a>
## Stable interfaces

The current public boundaries are the two Gym task IDs, command-line entry points, RSL-RL checkpoint layout, panel run/artifact discovery, the 56/280 observation and 12-action policy contract, the 60 Hz control rate, and ROS messages/topics. A change crossing one of these boundaries requires operator or developer documentation plus compatibility review.

<a id="known-coupling"></a>
## Known coupling

The Isaac environment still combines simulator I/O, reward/observation math, gait and command state, randomization, buffers, and logging. Some contract facts are mirrored into ROS and protected by parity tests rather than generated from one package. The proposed core-first reboot documents a possible extraction; it is not the active architecture.

<a id="next"></a>
## Related documents

- [Training and policy architecture](training-and-policy.en.md)
- [Reward and energy model](reward-and-energy.en.md)
- [Sim-to-real architecture](sim-to-real.en.md)
- [Subsystem ownership](../subsystems/ownership.en.md)
