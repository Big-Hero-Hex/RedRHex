---
id: ros2-deployment-troubleshooting
title: Troubleshoot ROS 2 deployment
lang: en
audience: shared
type: troubleshooting
status: active
owner: deployment
last_reviewed: 2026-08-07
---

<a id="troubleshoot-ros2-deployment"></a>
# Troubleshoot ROS 2 deployment

Keep motor output disabled while diagnosing. If motion is unexpected, assert the physical E-stop and remove motor power first.

<a id="policy-enable-is-rejected"></a>
## Policy enable is rejected

Inspect `/redrhex/state_machine_state` and `/redrhex/diagnostics`. Policy enable is expected to be rejected before `POLICY_READY`, after an E-stop or protective fault, when required signals are stale, or when the configured rest-attitude gate fails.

Do not loosen a threshold to silence a fault. Correct the frame, signal, timing, or state transition and repeat the dry-run evidence.

<a id="motor-output-remains-disabled"></a>
## Motor output remains disabled

This can be correct even while inference runs. Confirm `/redrhex/enable_motors` was deliberately requested, the state permits it, low-level heartbeat requirements are met, and no safety fault cleared the latch. Policy enable and motor enable are independent.

<a id="observation-or-inference-dimension-fails"></a>
## Observation or inference dimension fails

Check the ONNX graph against 56 single-frame or 280 five-frame input and 12 output values. Verify the deployed config and policy originated from the same environment contract. Do not pad, trim, or reorder tensors to make a foreign artifact load.

<a id="robot-appears-tilted-while-level"></a>
## Robot appears tilted while level

The IMU frame likely differs from the trained body frame. Measure and set `imu_mount_rpy_deg`, then compare projected gravity at rest with training evidence. Do not compensate by widening roll/pitch safety limits.

<a id="joint-motion-has-the-wrong-direction-or-zero"></a>
## Joint motion has the wrong direction or zero

Stop all multi-joint tests. With the robot lifted and current-limited, repeat one-joint preview and command tests. Change only the proven entry in `main_drive_sign`, `abad_sign`, or the corresponding zero-offset array, then re-run readiness and the one-joint test.

<a id="heartbeat-or-sensor-is-stale"></a>
## Heartbeat or sensor is stale

Inspect timestamps, network route, workspace sourcing, topic names, and publisher counts. The default heartbeat timeout is 0.10 s, sensor timeout 0.10 s, motor feedback timeout 0.25 s, and command timeout 0.25 s. Fix the producer or transport; do not bypass a required heartbeat for hardware operation.

<a id="more-than-one-motor-command-publisher-exists"></a>
## More than one motor-command publisher exists

Keep the relay off. Stop competing controllers such as calibration, standing, tripod, or an earlier RL process. Recheck the graph until exactly one authorized command path remains.

<a id="recovery-checklist"></a>
## Recovery checklist

After correcting the cause, return to disabled output, reassert and deliberately clear the software E-stop, repeat the mock or preview stage, and regain hardware authority one stage at a time. Never resume directly at the failed stage after a protective stop.
