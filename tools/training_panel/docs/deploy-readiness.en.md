---
id: training-panel-deploy-readiness
title: Training Panel Deployment Readiness
lang: en
audience: operator
type: safety
status: active
owner: deployment
last_reviewed: 2026-08-07
---

<a id="levels"></a>
## Readiness levels

`ready` means required export, ONNX, runtime, parity, contract, and safety checks passed. `review` means no required failure but at least one warning or skipped supporting check. `blocked` means at least one required stage failed; do not continue to robot bring-up.

<a id="runtime"></a>
## Runtime split

Training, playback, video, and export run through the Isaac launcher. Readiness analysis runs in the panel Python and needs `onnx`, `onnxruntime`, `torch`, and optionally `mujoco`. The Deploy defaults API and process log identify the exact interpreter and dependency status.

<a id="stages"></a>
## Stages

Required stages verify checkpoint/export files and hashes, ONNX shape and inference, Torch/ONNX parity, 56/280 observation and 12-action contract, 60 Hz, joint/scaling limits, and synthetic safety faults. ROS mock is optional. MuJoCo is advisory while its configuration is marked `calibrated=false` and becomes meaningful only with reviewed calibration.

<a id="artifacts"></a>
## Artifacts

Readiness JSON/Markdown and MuJoCo traces are written below the selected run's `deploy/` directory. They belong to that run and are not source artifacts. Copy the reviewed ONNX, deployment YAML, and readiness report together for Jetson handoff.

<a id="hardware"></a>
## Hardware gate

Before hardware enable, run Jetson preflight, source the correct ROS workspace, verify physical E-stop and cutoff, bridge heartbeat, one publisher, low-power single ABAD, lifted single main drive, and disabled policy/motor startup flags. A clean MuJoCo video never replaces these checks.

<a id="rollback"></a>
## Rollback

On unexpected behavior, assert software and physical stop, disable motor output, restore the last reviewed bundle, rerun readiness and preflight, and record the failing report path in run notes.
