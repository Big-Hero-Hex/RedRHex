# Five-Leg Fast Policy Run SOP

Use this only for the five-leg setup where Rinbo leg `l3` has no physical leg installed. This SOP uses the isolated `redrhex_five_leg_mode` package:

- Low-level bridge disables Rinbo leg `l3`.
- Policy controller ignores policy leg `Revolute_24` / main index `5`.
- Controller enters `POLICY_READY` after the timed init window without requiring the missing leg to reach stand pose.

Do not use the regular six-leg bringup for this hardware state.

## Key Fix For Missing Encoders

If controller diagnostics says `missing main encoders`, the policy is not stuck because of the policy. It means `/joint_states` is stale or missing because the Rinbo state stream is stale or missing.

The fix that made the run work was:

1. Restart sbRIO `grpccore` and `fpga_driver` if `/motor/state` and `/power/state` do not produce real messages.
2. Fully restart the Orin `rinbo_ros_bridge` after that.
3. Continue only after the raw state check reports `/motor/state: message received` and `/power/state: message received`.

Topic names alone are not enough. The topics can exist while no state messages are streaming.

## 0. Optional Build

Run this after code changes.

```bash
cd ~/RedRhex/RedRhex/ros2_ws
source /opt/ros/humble/setup.bash
source ~/rinbo_ros_ws/install/setup.bash
colcon build --symlink-install --packages-select redrhex_rl_controller redrhex_five_leg_mode
source install/setup.bash
```

## 1. Start Rinbo Bridge

Terminal 1:

```bash
cd ~/rinbo_ros_ws
source /opt/ros/humble/setup.bash
source install/setup.bash
export CORE_MASTER_ADDR=192.168.30.2:50051
export CORE_LOCAL_IP=192.168.30.8
ros2 run rinbo_ros_bridge rinbo_ros_bridge
```

If this is a retry after missing encoders, stop the old Orin bridge first with `Ctrl-C`, then run:

```bash
source /opt/ros/humble/setup.bash
ros2 daemon stop || true
pgrep -af "rinbo_ros_bridge|redrhex_lowlevel_bridge|five_leg_rl_controller|microstrain_inertial_driver|ros2 launch|ros2 run" || true
```

Start fresh only after the old `rinbo_ros_bridge` process is gone.

## 2. Start IMU

Terminal 2:

```bash
source /opt/ros/humble/setup.bash
ros2 launch microstrain_inertial_driver microstrain_launch.py \
  params_file:=/home/jetson/RedRhex/RedRhex/ros2_ws/src/redrhex_rl_controller/config/microstrain_redrhex.yaml
```

## 3. Raw Rinbo State Check

Control terminal:

```bash
source /opt/ros/humble/setup.bash
source ~/rinbo_ros_ws/install/setup.bash
export CORE_MASTER_ADDR=192.168.30.2:50051
export CORE_LOCAL_IP=192.168.30.8
timeout 10 ros2 run redrhex_lowlevel_bridge rinbo_bringup_check \
  --message-timeout-s 6 --require-power-state
```

Continue only if `/motor/state` and `/power/state` both report `message received`. If the check says TCP is connected but no state messages arrive, stop here; the policy cannot reach `POLICY_READY` without encoder feedback.

If TCP is connected but no state messages arrive, restart the sbRIO core/FPGA driver in the sbRIO terminal:

```bash
cd ~/rinbo_sbRIO_ws/rinbo_fpga_driver/build/
pkill -f fpga_driver || true
pkill -f grpccore || true
sleep 1

export CORE_LOCAL_IP=192.168.30.2
export CORE_MASTER_ADDR=192.168.30.2:50051

nohup /home/admin/rinbo_sbRIO_ws/install/bin/grpccore >/tmp/grpccore.log 2>&1 &
sleep 1
nohup /home/admin/rinbo_sbRIO_ws/rinbo_fpga_driver/build/fpga_driver >/tmp/fpga_driver.log 2>&1 &
sleep 2

ps -ef | egrep "grpccore|fpga_driver" | grep -v grep
netstat -tn | grep 50051 || netstat -ltn | grep 50051 || echo "NO TCP on 50051"
```

Then restart the Orin Rinbo bridge and repeat this raw state check.

If state was good but later `missing main encoders` comes back, do not continue by only publishing `/redrhex/enable_policy`. Stop the local stack, restart the Orin bridge from Step 1, then repeat this raw state check before starting low-level/controller again.

## 4. Start Five-Leg Low-Level Bridge

Terminal 3:

```bash
cd ~/RedRhex/RedRhex/ros2_ws
source /opt/ros/humble/setup.bash
source ~/rinbo_ros_ws/install/setup.bash
source install/setup.bash
ros2 launch redrhex_five_leg_mode five_leg_lowlevel_bridge.launch.py
```

## 5. Start Five-Leg Policy Controller

Terminal 4:

```bash
cd ~/RedRhex/RedRhex/ros2_ws
source /opt/ros/humble/setup.bash
source ~/rinbo_ros_ws/install/setup.bash
source install/setup.bash
ros2 launch redrhex_five_leg_mode five_leg_policy_bringup.launch.py \
  onnx_path:=/home/jetson/RedRHex/policy.onnx
```

## 6. Quick Gate Check

Control terminal:

```bash
cd ~/RedRhex/RedRhex/ros2_ws
source /opt/ros/humble/setup.bash
source ~/rinbo_ros_ws/install/setup.bash
source install/setup.bash

ros2 node list | sort
timeout 5 ros2 topic echo /redrhex/lowlevel_diagnostics --once
timeout 5 ros2 topic echo /redrhex/diagnostics --once
```

Expected:

- Exactly one low-level bridge and one policy controller.
- Low-level diagnostics includes `disabled_legs=['l3']`.
- Controller diagnostics eventually reaches `POLICY_READY` after motors are enabled.

## 7. Enable Five-Leg Policy Run

Control terminal:

```bash
timeout 5 ros2 topic pub --once /redrhex/enable_policy std_msgs/msg/Bool "{data: false}"
timeout 5 ros2 topic pub --once /redrhex/enable_motors std_msgs/msg/Bool "{data: false}"
timeout 5 ros2 topic pub --once /cmd_vel geometry_msgs/msg/Twist "{linear: {x: 0.0, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}"

python3 - <<'PY'
import time
import rclpy
from rinbo_msgs.msg import PowerCmdStamped

rclpy.init()
node = rclpy.create_node("redrhex_power_on_sequence")
pub = node.create_publisher(PowerCmdStamped, "/power/command", 10)
deadline = time.monotonic() + 3.0
while time.monotonic() < deadline and pub.get_subscription_count() == 0:
    rclpy.spin_once(node, timeout_sec=0.05)
print(f"power subscribers={pub.get_subscription_count()}")

for seq, (digital, signal, power) in enumerate([(True, False, False), (True, True, False), (True, True, True)]):
    msg = PowerCmdStamped()
    msg.header.seq = seq
    msg.header.frame_id = "redrhex_power_on_sequence"
    msg.digital = digital
    msg.signal = signal
    msg.power = power
    if hasattr(msg, "clean"):
        msg.clean = False
    if hasattr(msg, "trigger"):
        msg.trigger = False
    for _ in range(10):
        msg.header.stamp = node.get_clock().now().to_msg()
        pub.publish(msg)
        rclpy.spin_once(node, timeout_sec=0.02)
        time.sleep(0.05)
    print(f"published digital={digital} signal={signal} power={power}")
    time.sleep(0.4)

node.destroy_node()
rclpy.shutdown()
PY

timeout 5 ros2 topic echo /power/state --once
timeout 6 ros2 run redrhex_rl_controller estop_tool off --wait-s 2 --duration-s 2 --rate-hz 20
timeout 5 ros2 topic pub --once /redrhex/enable_motors std_msgs/msg/Bool "{data: true}"

timeout 15 ros2 topic echo /redrhex/diagnostics
```

Wait until diagnostics says `POLICY_READY: policy ready, waiting enable_policy`.

## 8. Run For 20 Seconds

Control terminal:

```bash
timeout 5 ros2 topic pub --once /cmd_vel geometry_msgs/msg/Twist "{linear: {x: 0.0, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}"
timeout 5 ros2 topic pub --once /redrhex/enable_policy std_msgs/msg/Bool "{data: true}"

python3 - <<'PY'
import time
import rclpy
from geometry_msgs.msg import Twist

rclpy.init()
node = rclpy.create_node("redrhex_five_leg_20s_cmd")
pub = node.create_publisher(Twist, "/cmd_vel", 10)
msg = Twist()
msg.linear.x = 0.05

end = time.monotonic() + 20.0
while time.monotonic() < end:
    pub.publish(msg)
    rclpy.spin_once(node, timeout_sec=0.0)
    time.sleep(0.1)

node.destroy_node()
rclpy.shutdown()
PY
```

## 9. Stop And Power Off

Run this after every test, or immediately if anything looks wrong.

```bash
timeout 5 ros2 run redrhex_rl_controller estop_tool on --wait-s 1 --duration-s 2 --rate-hz 20 || true
timeout 5 ros2 topic pub --once /redrhex/enable_policy std_msgs/msg/Bool "{data: false}" || true
timeout 5 ros2 topic pub --once /redrhex/enable_motors std_msgs/msg/Bool "{data: false}" || true
timeout 5 ros2 topic pub --once /cmd_vel geometry_msgs/msg/Twist "{linear: {x: 0.0, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}" || true
timeout 5 ros2 topic pub --once /power/command rinbo_msgs/msg/PowerCmdStamped \
  "{header: {seq: 1, stamp: {sec: 0, nanosec: 0}}, digital: false, signal: false, power: false}" || true

timeout 5 ros2 topic echo /redrhex/diagnostics --once || true
timeout 5 ros2 topic echo /redrhex/lowlevel_diagnostics --once || true
timeout 5 ros2 topic echo /power/state --once || true
```

Then stop the four long-running terminals with `Ctrl-C`.

## Fast Abort

If the robot should stop immediately, run this first:

```bash
source /opt/ros/humble/setup.bash
source ~/rinbo_ros_ws/install/setup.bash
source ~/RedRhex/RedRhex/ros2_ws/install/setup.bash
timeout 5 ros2 run redrhex_rl_controller estop_tool on --wait-s 1 --duration-s 2 --rate-hz 20 || true
timeout 5 ros2 topic pub --once /redrhex/enable_policy std_msgs/msg/Bool "{data: false}" || true
timeout 5 ros2 topic pub --once /redrhex/enable_motors std_msgs/msg/Bool "{data: false}" || true
timeout 5 ros2 topic pub --once /cmd_vel geometry_msgs/msg/Twist "{linear: {x: 0.0, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}" || true
timeout 5 ros2 topic pub --once /power/command rinbo_msgs/msg/PowerCmdStamped \
  "{header: {seq: 1, stamp: {sec: 0, nanosec: 0}}, digital: false, signal: false, power: false}" || true
```
