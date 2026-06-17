# Step 9 Policy Run From Cold Start

This is the start-of-day SOP for an airborne Step 9 policy run.

It is intentionally strict. Do not bypass `POLICY_READY` on hardware.

## Why The Last Policy Run Did Not Work

The run did not fail because of ONNX. The ONNX loaded correctly as `56 -> 12`.

It failed because the hardware/controller gates were not satisfied:

- The old SOP did not clearly mark long-running terminals. You pasted follow-up commands into terminals that were busy running bridges/controllers.
- Two `microstrain_inertial_driver` nodes were running at the same time. ROS warned about duplicate node names, and this can cause unstable graph behavior.
- The IMU frame was initially wrong. The controller saw roll near `-3.12 rad`. This has been fixed in `config/microstrain_redrhex.yaml`.
- The controller later reached `INIT_STAND`, but it cannot enter `POLICY_READY` unless all required main-drive encoder positions reach the init-stand target.
- `rinbo_disabled_legs:=l1` only stops the lowlevel bridge from commanding L1. It does not teach the RL controller or the policy to ignore L1.
- A five-leg real policy run is not supported by this SOP unless the controller code is changed to explicitly ignore that leg in init-stand and safety checks.

Bottom line: if the robot cannot complete calibration/standing/init-stand for the required legs, Step 9 must stop before policy enable.

## Terminal Rules

Use five Orin terminals:

- `T1_RINBO`: runs `rinbo_ros_bridge`. It keeps running. Do not paste other commands here.
- `T_IMU`: runs the MicroStrain IMU driver. It keeps running. Do not paste other commands here.
- `T3_BRIDGE`: runs `redrhex_lowlevel_bridge`. It keeps running. Do not paste other commands here.
- `T4_CONTROLLER`: runs `redrhex_rl_controller`. It keeps running. Do not paste other commands here.
- `T2_CONTROL`: the only terminal where you paste one-shot commands, checks, enable commands, and stop commands.

If a terminal is printing logs and has no shell prompt, it is busy. Open another terminal or press `Ctrl+C` to stop that process.

Any command using `ros2 launch`, `ros2 run rinbo_ros_bridge`, or `ros2 topic pub --rate` is expected to keep running unless wrapped in `timeout`.

Use `timeout` for every check that could otherwise block forever.

## Current Safety Reset

Use this before a new attempt if terminals are messy.

In `T2_CONTROL`:

```bash
source /opt/ros/humble/setup.bash
source ~/rinbo_ros_ws/install/setup.bash
source ~/RedRhex/RedRhex/ros2_ws/install/setup.bash

timeout 3 ros2 topic pub --once /redrhex/enable_policy std_msgs/msg/Bool "{data: false}" || true
timeout 3 ros2 topic pub --once /redrhex/enable_motors std_msgs/msg/Bool "{data: false}" || true
timeout 3 ros2 topic pub --once /power/command rinbo_msgs/msg/PowerCmdStamped "{header: {seq: 1, stamp: {sec: 0, nanosec: 0}}, digital: true, signal: true, power: false}" || true

pkill -INT -f '[r]edrhex_lowlevel_bridge' || true
pkill -INT -f '[r]edrhex_rl_controller' || true
pkill -INT -f '[m]icrostrain_inertial_driver' || true
pkill -INT -f '[m]icrostrain_launch.py' || true
sleep 2
pkill -TERM -f '[r]edrhex_lowlevel_bridge' || true
pkill -TERM -f '[r]edrhex_rl_controller' || true
pkill -TERM -f '[m]icrostrain_inertial_driver' || true
pkill -TERM -f '[m]icrostrain_launch.py' || true

ps -ef | grep -E "rinbo_ros_bridge|redrhex_lowlevel_bridge|redrhex_rl_controller|microstrain" | grep -v grep
timeout 5 ros2 topic echo /power/state --once
```

Expected before continuing:

- `rinbo_ros_bridge` may still be running.
- No RedRhex bridge/controller process.
- No MicroStrain process.
- `/power/state` shows `power: false`.

## 0. Start Clean On Orin

In `T2_CONTROL`:

```bash
cd ~/RedRhex/RedRhex/ros2_ws
source /opt/ros/humble/setup.bash
source ~/rinbo_ros_ws/install/setup.bash
source install/setup.bash

pkill -INT -f '[r]inbo_tripod' || true
pkill -INT -f '[r]inbo_standing' || true
pkill -INT -f '[r]inbo_cali' || true
```

If code changed:

```bash
colcon build --symlink-install
source install/setup.bash
```

## 1. Start sbRIO Drivers

Open the sbRIO shell:

```bash
ssh admin@192.168.30.2
```

On sbRIO:

```bash
cd ~/rinbo_sbRIO_ws/rinbo_fpga_driver/build/
pkill -f fpga_driver || true
pkill -f grpccore || true
export CORE_LOCAL_IP=192.168.30.2
export CORE_MASTER_ADDR=192.168.30.2:50051
nohup /home/admin/rinbo_sbRIO_ws/install/bin/grpccore >/tmp/grpccore.log 2>&1 &
nohup /home/admin/rinbo_sbRIO_ws/rinbo_fpga_driver/build/fpga_driver >/tmp/fpga_driver.log 2>&1 &
ps -ef | egrep "grpccore|fpga_driver" | grep -v grep
netstat -tn | grep 50051 || netstat -ltn | grep 50051 || echo "NO TCP on 50051"
```

The sbRIO shell can be left alone after this.

## 2. Start Rinbo ROS Bridge

In `T1_RINBO`, keep this running:

```bash
cd ~/rinbo_ros_ws
source /opt/ros/humble/setup.bash
source install/setup.bash
export CORE_MASTER_ADDR=192.168.30.2:50051
export CORE_LOCAL_IP=192.168.30.8
ros2 run rinbo_ros_bridge rinbo_ros_bridge
```

Do not paste more commands into `T1_RINBO` while it is running.

## 3. Power Sensor Rails Only

In `T2_CONTROL`:

```bash
source /opt/ros/humble/setup.bash
source ~/rinbo_ros_ws/install/setup.bash
source ~/RedRhex/RedRhex/ros2_ws/install/setup.bash

timeout 5 ros2 topic list | grep -E "motor|power"
timeout 5 ros2 topic echo /power/state --once
timeout 5 ros2 topic echo /motor/state --once
```

Turn on digital and signal. Keep motor relay off:

```bash
timeout 5 ros2 topic pub --once /power/command rinbo_msgs/msg/PowerCmdStamped "{header: {seq: 1, stamp: {sec: 0, nanosec: 0}}, digital: true, signal: false, power: false}"
timeout 5 ros2 topic pub --once /power/command rinbo_msgs/msg/PowerCmdStamped "{header: {seq: 1, stamp: {sec: 0, nanosec: 0}}, digital: true, signal: true, power: false}"
timeout 5 ros2 topic echo /power/state --once
```

Required:

- `digital: true`
- `signal: true`
- `power: false`

## 4. Start Exactly One IMU Driver

In `T_IMU`, keep this running:

```bash
source /opt/ros/humble/setup.bash
ros2 launch microstrain_inertial_driver microstrain_launch.py \
  params_file:=/home/jetson/RedRhex/RedRhex/ros2_ws/src/redrhex_rl_controller/config/microstrain_redrhex.yaml
```

Do not start a second IMU driver.

In `T2_CONTROL`, verify:

```bash
source /opt/ros/humble/setup.bash
ros2 node list | grep microstrain
timeout 5 ros2 topic echo /imu/data --once
timeout 6 ros2 topic hz /imu/data
```

Required:

- Exactly one `/microstrain_inertial_driver` node.
- `/imu/data` near `100 Hz`.
- Roll/pitch should be small when the robot is stationary and upright.

Quick roll/pitch check:

```bash
python3 - <<'PY'
import math, subprocess, yaml
out = subprocess.check_output(
    ["bash", "-lc", "source /opt/ros/humble/setup.bash; timeout 5 ros2 topic echo /imu/data --once"],
    text=True,
)
data = yaml.safe_load(out.split("---")[0])
q = data["orientation"]
x, y, z, w = q["x"], q["y"], q["z"], q["w"]
roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
sinp = 2 * (w * y - z * x)
pitch = math.copysign(math.pi / 2, sinp) if abs(sinp) >= 1 else math.asin(sinp)
print(f"roll={roll:.4f} rad, pitch={pitch:.4f} rad")
PY
```

Stop if `abs(roll)` or `abs(pitch)` is near `0.7 rad` or larger while the robot is upright.

## 5. Calibration And Standing

Open motor relay only for calibration/standing:

```bash
timeout 5 ros2 topic pub --once /power/command rinbo_msgs/msg/PowerCmdStamped "{header: {seq: 1, stamp: {sec: 0, nanosec: 0}}, digital: true, signal: true, power: true}"
timeout 5 ros2 topic echo /power/state --once
```

Required:

- `power: true`
- Motor bus voltages look reasonable.
- Current does not jump abnormally.

Run:

```bash
ros2 run rinbo_fsm rinbo_cali
ros2 run rinbo_fsm rinbo_standing
```

These commands are allowed to move the robot. Stop if a leg binds, current jumps, or calibration never reaches all required legs.

After standing:

```bash
pkill -INT -f '[r]inbo_tripod' || true
pkill -INT -f '[r]inbo_standing' || true
pkill -INT -f '[r]inbo_cali' || true
ros2 topic info /motor/command -v
```

Do not continue if `/motor/command` has `rinbo_tripod` or any unknown publisher.

Important: if calibration or standing cannot finish because a leg is broken, this SOP cannot safely continue to Step 9.

## 6. Start RedRhex Lowlevel Bridge

In `T3_BRIDGE`, keep this running:

```bash
cd ~/RedRhex/RedRhex/ros2_ws
source /opt/ros/humble/setup.bash
source ~/rinbo_ros_ws/install/setup.bash
source install/setup.bash

ros2 launch redrhex_lowlevel_bridge lowlevel_bridge.launch.py \
  bridge_backend:=biorola_ros \
  rinbo_allow_enable:=true \
  rinbo_main_max_pwm:=80.0
```

Do not use `rinbo_disabled_legs` for a real Step 9 policy run unless the controller has also been updated to ignore those legs.

If you must protect a broken leg during diagnostics, use this only for preview or manual tests:

```bash
ros2 launch redrhex_lowlevel_bridge lowlevel_bridge.launch.py \
  bridge_backend:=biorola_ros \
  rinbo_allow_enable:=false \
  rinbo_main_max_pwm:=80.0 \
  rinbo_disabled_legs:=l1
```

## 7. Start Policy Controller

In `T4_CONTROLLER`, keep this running:

```bash
cd ~/RedRhex/RedRhex/ros2_ws
source /opt/ros/humble/setup.bash
source ~/rinbo_ros_ws/install/setup.bash
source install/setup.bash

ros2 launch redrhex_rl_controller redrhex_policy_bringup.launch.py \
  onnx_path:=/home/jetson/RedRHex/policy.onnx
```

Expected:

```text
Loaded ONNX policy:
input=obs shape=[1, 56]
output=actions shape=[1, 12]
```

## 8. Pre-Run Gate Check

In `T2_CONTROL`:

```bash
timeout 5 ros2 topic echo /redrhex/lowlevel_heartbeat --once
timeout 5 ros2 topic echo /redrhex/lowlevel_diagnostics --once
timeout 5 ros2 topic echo /redrhex/power_safety_trip --once
timeout 5 ros2 topic echo /imu/data --once
timeout 5 ros2 topic echo /power/state --once
ros2 topic info /redrhex/motor_commands -v
ros2 topic info /motor/command -v
ros2 node list | sort
```

Required:

- `/redrhex/lowlevel_heartbeat`: `data: true`
- `/redrhex/power_safety_trip`: `data: false`
- `/power/state`: `digital: true`, `signal: true`, `power: true`
- Lowlevel diagnostics: `message: ok`, `allow_enable=True`
- Exactly one `/microstrain_inertial_driver`
- Exactly one `/redrhex_lowlevel_bridge`
- Exactly one `/redrhex_rl_controller`
- `/redrhex/motor_commands`: publisher from controller, subscriber from lowlevel bridge
- `/motor/command`: publisher from lowlevel bridge, subscriber from `rinbo_ros2_bridge`

Stop here if any required check fails.

## 9. Enable Motors And Wait For POLICY_READY

In `T2_CONTROL`:

```bash
timeout 5 ros2 run redrhex_rl_controller estop_tool off
timeout 5 ros2 topic pub --once /redrhex/enable_motors std_msgs/msg/Bool "{data: true}"
timeout 20 ros2 topic echo /redrhex/state_machine_state
```

You may see `INIT_STAND: moving to init stand` first.

Continue only after you see:

```text
POLICY_READY: policy ready, waiting enable_policy
```

Hard stop:

- Do not continue from `PROTECTIVE_STOP`.
- Do not continue from `INIT_STAND`.
- Do not continue if the message mentions IMU timeout, joint timeout, roll/pitch too large, current too high, deadline miss, or motor fault.

## 10. Run Policy For 2 Seconds

Only run this after `POLICY_READY`.

In `T2_CONTROL`:

```bash
timeout 5 ros2 topic pub --once /cmd_vel geometry_msgs/msg/Twist "{linear: {x: 0.0, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}"
timeout 5 ros2 topic pub --once /redrhex/enable_policy std_msgs/msg/Bool "{data: true}"
timeout 2s ros2 topic pub --rate 10 /cmd_vel geometry_msgs/msg/Twist "{linear: {x: 0.05, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}" || true
```

Then stop immediately.

## 11. Stop Immediately

In `T2_CONTROL`:

```bash
timeout 5 ros2 run redrhex_rl_controller estop_tool on || true
timeout 5 ros2 topic pub --once /redrhex/enable_policy std_msgs/msg/Bool "{data: false}" || true
timeout 5 ros2 topic pub --once /redrhex/enable_motors std_msgs/msg/Bool "{data: false}" || true
timeout 5 ros2 topic pub --once /cmd_vel geometry_msgs/msg/Twist "{linear: {x: 0.0, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}" || true
```

Check:

```bash
timeout 5 ros2 topic echo /redrhex/diagnostics --once
timeout 5 ros2 topic echo /redrhex/lowlevel_diagnostics --once
timeout 5 ros2 topic echo /redrhex/power_safety_trip --once
```

## 12. Power Off And Close Terminals

In `T2_CONTROL`:

```bash
timeout 5 ros2 topic pub --once /power/command rinbo_msgs/msg/PowerCmdStamped "{header: {seq: 1, stamp: {sec: 0, nanosec: 0}}, digital: true, signal: true, power: false}" || true
timeout 5 ros2 topic echo /power/state --once
```

Stop these terminals with `Ctrl+C`:

- `T4_CONTROLLER`
- `T3_BRIDGE`
- `T_IMU`
- `T1_RINBO`, only if you are done with all Rinbo work

End-of-day full power off:

```bash
timeout 5 ros2 topic pub --once /power/command rinbo_msgs/msg/PowerCmdStamped "{header: {seq: 1, stamp: {sec: 0, nanosec: 0}}, digital: false, signal: false, power: false}" || true
timeout 5 ros2 topic echo /power/state --once
```

Optional sbRIO stop:

```bash
ssh admin@192.168.30.2
pkill -f fpga_driver || true
pkill -f grpccore || true
ps -ef | egrep "grpccore|fpga_driver" | grep -v grep
```

## Fast Abort

Use this in `T2_CONTROL` if anything looks wrong:

```bash
source /opt/ros/humble/setup.bash
source ~/rinbo_ros_ws/install/setup.bash
source ~/RedRhex/RedRhex/ros2_ws/install/setup.bash

timeout 3 ros2 run redrhex_rl_controller estop_tool on || true
timeout 3 ros2 topic pub --once /redrhex/enable_policy std_msgs/msg/Bool "{data: false}" || true
timeout 3 ros2 topic pub --once /redrhex/enable_motors std_msgs/msg/Bool "{data: false}" || true
timeout 3 ros2 topic pub --once /cmd_vel geometry_msgs/msg/Twist "{linear: {x: 0.0, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}" || true
timeout 3 ros2 topic pub --once /power/command rinbo_msgs/msg/PowerCmdStamped "{header: {seq: 1, stamp: {sec: 0, nanosec: 0}}, digital: true, signal: true, power: false}" || true
timeout 5 ros2 topic echo /power/state --once
```

Then stop `T4_CONTROLLER`, `T3_BRIDGE`, and `T_IMU` with `Ctrl+C`.
