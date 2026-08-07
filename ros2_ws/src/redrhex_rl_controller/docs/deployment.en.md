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
