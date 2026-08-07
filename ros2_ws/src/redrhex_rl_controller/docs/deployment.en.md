---
id: ros2-hardware-deployment
title: Deploy a policy to RedRHex hardware
lang: en
audience: operator
type: how-to
status: active
owner: deployment
last_reviewed: 2026-08-07
---

<a id="deploy-a-policy-to-redrhex-hardware"></a>
# Deploy a policy to RedRHex hardware

Do not skip stages. The first hardware goal is verified transport and direction, not autonomous walking.

<a id="prepare-the-station"></a>
## Prepare the station

Keep a physical E-stop in hand, apply conservative current limits, and lift the robot securely. Confirm that only one `grpccore` and one `fpga_driver` are running and that no competing controller publishes motor commands.

Source the hardware workspace, then this repository's `ros2_ws`. Generate a read-only terminal plan if needed:

```bash
ros2 run redrhex_lowlevel_bridge biorola_bringup_plan \
  --sbrio-ip 192.168.30.12 \
  --orin-ip 192.168.30.164 \
  --onnx-path /home/jetson/redrhex_models/policy.onnx
```

The command prints steps and does not publish motor commands.

<a id="verify-low-level-readiness"></a>
## Verify low-level readiness

```bash
ros2 run redrhex_lowlevel_bridge biorola_bringup_check --message-timeout-s 5.0
ros2 run redrhex_lowlevel_bridge biorola_power_tool status
```

Do not energize the relay unless bring-up has no error and digital, signal, and power state can be read. Complete `rinbo_cali` and `rinbo_standing`, then stop their controller before starting the RedRHex bridge.

<a id="preview-without-authority"></a>
## Preview without authority

Start the Rinbo/Biorola bridge with `allow_enable=false`. Inspect both mapping and heartbeat:

```bash
ros2 topic echo /redrhex/rinbo_motor_command_preview --once
ros2 topic echo /redrhex/lowlevel_heartbeat
ros2 topic echo /redrhex/lowlevel_diagnostics --once
```

Do not continue if the preview order, sign, scale, heartbeat, or publisher count is wrong.

<a id="fixed-sim-to-real-probe"></a>
## Capture the fixed sim-to-real probe

Use `sim2real_probe` only after preview succeeds and before policy control. It is an immutable suspended single-main-drive step/coast sequence: three repetitions at 60 Hz, 990 command ticks, 16.5 s total, ±0.25 rad/s drive segments, and a probe-only physical PWM ceiling of 30.0. Main indices 0–4 are calibration; index 5 is the holdout.

Preview the JSON without creating a ROS node or publishing:

```bash
ros2 run redrhex_rl_controller sim2real_probe --main-index 0 --dry-run
```

Do not energize unless the preview reports the expected scenario ID, SHA-256, rate, repeats, ticks, duration, speed cap, and PWM cap. Before the enabled run, prove the physical E-stop, conservative current limiting, secure suspension, cable clearance, and sbRIO watchdog. Isolate ABAD power, or physically verify the disabled servo mode and then set the bridge interlock `probe_abad_disable_verified: true`. CLI confirmation cannot replace that hardware evidence.

Stop every other motor-command publisher. The probe must be the only publisher to `/redrhex/motor_commands`, with a visible subscriber, fresh true heartbeat, fresh joint state, and explicit `/estop=false`. Record the raw BioRoLa topics rather than only derived feedback:

```bash
ros2 bag record -o redrhex_probe_main0_raw \
  /motor/command \
  /motor/state \
  /redrhex/motor_commands \
  /redrhex/sim2real_probe/events \
  /redrhex/lowlevel_heartbeat \
  /joint_states \
  /estop \
  /imu/data
```

After recording begins, grant both explicit authorizations in another terminal:

```bash
ros2 run redrhex_rl_controller sim2real_probe --main-index 0 --enable --confirm-risk --confirm-abad-disable
```

The scheduler uses absolute 60 Hz deadlines. If a command is late by one period, about 16.7 ms, it aborts before publishing that tick and sends disabled packets. It never catches up a missed tick or bursts delayed enabled commands. Any E-stop, heartbeat, joint-state, graph-ownership, subscriber, or process anomaly requires physical E-stop and diagnosis before a new attempt.

<a id="start-the-gated-controller"></a>
## Start the gated controller

Start `redrhex_policy_bringup.launch.py` with the real ONNX path and both `enable_policy_on_start` and `enable_motor_output_on_start` set to false. First prove policy dry-run while watching state, diagnostics, raw action, safe action, and disabled motor output.

`/redrhex/enable_policy` permits closed-loop inference only when the state machine is ready. `/redrhex/enable_motors` independently permits enabled motor output. Opening the first gate must not open the second.

<a id="increase-authority-in-stages"></a>
## Increase authority in stages

1. With the robot lifted, test one ABAD at a small angle.
2. Test one main-drive motor at low speed and verify encoder direction.
3. Prove lifted `INIT_STAND` with conservative limits.
4. Run the policy lifted while monitoring timeouts, attitude, current, temperature, and heartbeat.
5. Only after all earlier stages pass, conduct a brief low-speed ground test with the physical E-stop ready.

Record any sign or zero-offset correction in the deployed YAML and repeat the single-joint evidence. Never route raw policy action directly to `/motor/command`.

<a id="abort-conditions"></a>
## Abort conditions

Assert E-stop and remove motor power for any stale heartbeat, stale sensor, excessive attitude, unexpected motion, mapping error, second command publisher, current/temperature violation, or loss of operator control. Diagnose with [Troubleshooting](troubleshooting.en.md) before resuming.
