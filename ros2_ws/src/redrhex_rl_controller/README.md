# RedRhex ONNX Sim2Real 安全部署手冊

這份文件是給你明天真的要把 `policy.onnx` 放到 Jetson Orin Nano / Jetson Nano，接上 sbRIO / BioRoLaROS2，讓 RedRhex 六足機器人開始跑 RL policy 的逐步手冊。

請先記住一句話：**第一次實機不要讓 policy 直接接管整台機器人。**
一定要照順序做：ONNX check -> mock -> sbRIO heartbeat -> preview-only -> 單顆馬達 -> 架空 policy -> 落地低速。

## 不能跳過的安全摘要

1. 不要同時跑 `rinbo_tripod` 和 RL controller。
2. 不要同時開兩個 `fpga_driver` 或兩個 `grpccore`。
3. 不要把 raw ONNX action 直接送 `/motor/command`。
4. 不要一開始就 `rinbo_allow_enable:=true`。
5. 不要一開始就 `enable_policy=true`。
6. 第一次只用 bench-safe profile：
   - main drive velocity limit: `1.0 rad/s`
   - main drive slew rate: `4.0 rad/s^2`
   - ABAD angle limit: `0.15 rad`
   - ABAD slew rate: `1.0 rad/s`
   - Rinbo PWM limit: `150`
   - main channel current limit: `3.0 A`
   - bus current limit: `12.0 A`
7. ROS2 軟體限流不是硬體急停。第一次實機仍然要架空、限流供電、手放實體急停。

## 目前這包做了什麼

這個 repo 現在新增三個 ROS2 package：

```text
ros2_ws/src/redrhex_msgs
ros2_ws/src/redrhex_rl_controller
ros2_ws/src/redrhex_lowlevel_bridge
```

資料流：

```text
/imu/data + /joint_states + /cmd_vel
  -> ObservationBuilder
  -> PolicyONNXRunner
  -> ActionDecoder
  -> SafetyFilter
  -> /redrhex/motor_commands
  -> redrhex_lowlevel_bridge
  -> BioRoLa / Rinbo / sbRIO / FPGA
  -> motors
```

這個分層參考 Lite3 的 Sim2Real 做法：`policy runner -> state machine -> safety -> robot interface`。但 RedRhex 的 observation order、action order、joint order 完全以本 repo 的 IsaacLab task 為準，不照 Lite3 的機器人定義。

參考專案：

- Lite3 RL deploy: https://github.com/DeepRoboticsLab/Lite3_rl_deploy
- Jetson BioRoLaROS2: https://github.com/JasonLiaoJCS/BioRoLaROS2
- sbRIO FPGA driver: https://github.com/ShuWei-Yang/rinbo_sbRIO_ws

## 你的 policy 路徑

訓練電腦目前的 ONNX：

```bash
/home/jasonliao/RedRhex/RedRhex/logs/rsl_rl/redrhex_wheg/2026-02-08_15-06-43_wheg_locomotion_v3/exported/policy.onnx
```

Jetson 上固定放：

```bash
/home/jetson/RedRHex/policy.onnx
```

從訓練電腦複製到 Jetson：

```bash
ssh jetson@yahboom "mkdir -p /home/jetson/RedRHex"
scp /home/jasonliao/RedRhex/RedRhex/logs/rsl_rl/redrhex_wheg/2026-02-08_15-06-43_wheg_locomotion_v3/exported/policy.onnx \
  jetson@yahboom:/home/jetson/RedRHex/policy.onnx
```

到 Jetson 確認：

```bash
ssh jetson@yahboom
ls -lh /home/jetson/RedRHex/policy.onnx
```

## 已確認的 IsaacLab contract

從 `redrhex_env.py` / `redrhex_env_cfg.py` 已確認：

- task: `Template-Redrhex-Direct-v0`
- policy input: `56`
- policy output: `12`
- sim dt: `1/250`
- decimation: `2`
- policy frequency: `1 / (1/250 * 2) = 125 Hz`
- ONNX export: `export_policy_as_onnx(policy_nn, normalizer=normalizer, ...)`
- 所以 ONNX 很可能已包含 RSL-RL observation normalizer，Jetson 端不要再 normalize 一次。

Observation 56 維順序：

| Index | Field |
| --- | --- |
| `0:3` | `base_lin_vel` |
| `3:6` | `base_ang_vel` |
| `6:9` | `projected_gravity` |
| `9:15` | `sin(main_drive_pos)` |
| `15:21` | `cos(main_drive_pos)` |
| `21:27` | `main_drive_vel / base_gait_angular_vel` |
| `27:33` | `abad_pos / abad_pos_scale` |
| `33:39` | `abad_vel` |
| `39:42` | `velocity_command [vx, vy, wz]` |
| `42:44` | `gait_phase [sin, cos]` |
| `44:56` | `last_actions` |

Action 12 維順序：

| Index | Meaning |
| --- | --- |
| `0:6` | 6 顆 main drive，policy order: RF, RM, RR, LF, LM, LR |
| `6:12` | 6 顆 ABAD，policy order: RF, RM, RR, LF, LM, LR |

Policy joint order：

| Policy index | Leg | Main drive | ABAD | Damper |
| --- | --- | --- | --- | --- |
| `0` | RF | `Revolute_15` | `Revolute_14` | `Revolute_5` |
| `1` | RM | `Revolute_7` | `Revolute_6` | `Revolute_8` |
| `2` | RR | `Revolute_12` | `Revolute_11` | `Revolute_13` |
| `3` | LF | `Revolute_18` | `Revolute_17` | `Revolute_25` |
| `4` | LM | `Revolute_23` | `Revolute_22` | `Revolute_26` |
| `5` | LR | `Revolute_24` | `Revolute_21` | `Revolute_27` |

Damper 是模擬彈簧腳，不是實體馬達，不在 action space。

Action decoding：

- 原始 action 先 clamp 到 `[-1, 1]`
- main drive residual: `action[0:6] * 8.0 * 0.40`
- ABAD: `action[6:12] * 0.61096 rad`
- 環境有 command-mode gating：
  - forward mode: ABAD action 歸零
  - lateral mode: main drive action 歸零，並進入 lateral step 邏輯
- 本部署包會先複製基本 gating，再加 bench-safe 限幅。

## 真機回授限制

你目前真機能拿到：

- IMU
- 6 顆 main drive encoder position
- main drive 由 PWM / speed-like 方式控制
- ABAD 位置控制，但沒有回授
- damper 沒有實體馬達

所以本 MVP 做法：

- `base_lin_vel` 初期設為 0，只能 bench / 架空 / 低速測試。
- `base_ang_vel` 從 IMU gyro 取。
- `projected_gravity` 從 IMU quaternion 算。
- main drive position / velocity 從 `/motor/state` 轉成 `/joint_states`。
- ABAD position 用「上一個 command target」當估計值。

風險：如果 `base_lin_vel` 長期設 0，落地 locomotion 可能有嚴重 sim2real mismatch。後續要補 leg odometry 或外部 odom。

## Jetson 安裝需求

假設：

- Ubuntu 22.04
- ROS2 Humble
- JetPack / Jetson Linux 已可正常跑 ROS2
- BioRoLaROS2 已在 `~/rinbo_ros_ws` build 完

Jetson 上安裝 Python 套件：

```bash
source /opt/ros/humble/setup.bash
python3 -m pip install --user "numpy<2" onnx onnxruntime pyserial
```

第一次先用 CPU ONNX Runtime。等整個流程穩定後再考慮 CUDA/TensorRT。

## 在 Jetson build 這包

假設你把這個 repo 放在：

```bash
~/RedRhex/RedRhex
```

Build：

```bash
cd ~/RedRhex/RedRhex/ros2_ws
source /opt/ros/humble/setup.bash
source ~/rinbo_ros_ws/install/setup.bash
colcon build --symlink-install
source install/setup.bash
```

如果 `source ~/rinbo_ros_ws/install/setup.bash` 失敗，代表 BioRoLaROS2 還沒 build。先回去 build `~/rinbo_ros_ws`。

## Step 1：ONNX I/O check

在 Jetson：

```bash
cd ~/RedRhex/RedRhex/ros2_ws
source /opt/ros/humble/setup.bash
source ~/rinbo_ros_ws/install/setup.bash
source install/setup.bash

ros2 run redrhex_rl_controller preflight_check \
  --onnx /home/jetson/RedRHex/policy.onnx \
  --hardware-profile bench
```

再跑：

```bash
ros2 run redrhex_rl_controller check_onnx_io /home/jetson/RedRHex/policy.onnx
```

你要看到：

- input 最後一維是 `56`
- output 最後一維是 `12`
- zero observation 可以推論
- output 沒有 NaN / Inf

如果這一步失敗，不要接真機。

## Step 2：不接真機 mock test

Terminal A：

```bash
cd ~/RedRhex/RedRhex/ros2_ws
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 launch redrhex_lowlevel_bridge lowlevel_bridge.launch.py bridge_backend:=mock
```

Terminal B：

```bash
cd ~/RedRhex/RedRhex/ros2_ws
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 run redrhex_rl_controller fake_redrhex_sensors
```

Terminal C：

```bash
cd ~/RedRhex/RedRhex/ros2_ws
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 launch redrhex_rl_controller redrhex_policy_bringup.launch.py \
  onnx_path:=/home/jetson/RedRHex/policy.onnx
```

Terminal D 檢查 topic：

```bash
source /opt/ros/humble/setup.bash
source ~/RedRhex/RedRhex/ros2_ws/install/setup.bash

ros2 topic echo /redrhex/state_machine_state
ros2 topic hz /redrhex/observation
ros2 topic echo /redrhex/policy_action_raw --once
ros2 topic echo /redrhex/motor_commands --once
```

清掉軟體 E-stop：

```bash
ros2 run redrhex_rl_controller estop_tool off
```

mock 模式中測 INIT_STAND：

```bash
ros2 topic pub --once /redrhex/enable_motors std_msgs/msg/Bool "{data: true}"
```

mock 模式中才可以測 policy run：

```bash
ros2 topic pub --once /redrhex/enable_policy std_msgs/msg/Bool "{data: true}"
ros2 topic pub --rate 10 /cmd_vel geometry_msgs/msg/Twist \
  "{linear: {x: 0.05, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}"
```

停止：

```bash
ros2 run redrhex_rl_controller estop_tool on
```

## Step 3：啟動 sbRIO / FPGA driver

在電腦 PowerShell 或 SSH terminal 連 sbRIO：

```bash
ssh admin@<SBRIO_IP>
```

常見 IP 是：

```bash
192.168.0.100
```

進入 driver 資料夾：

```bash
cd rinbo_sbRIO_ws/rinbo_fpga_driver/build/
```

設定 sbRIO 端環境變數：

```bash
export CORE_LOCAL_IP=<SBRIO_IP>
export CORE_MASTER_ADDR=<SBRIO_IP>:50051
```

先確認沒有舊 process：

```bash
pkill -f grpccore || true
pkill -f fpga_driver || true
```

啟動：

```bash
nohup /home/admin/rinbo_sbRIO_ws/install/bin/grpccore >/tmp/grpccore.log 2>&1 &
nohup /home/admin/rinbo_sbRIO_ws/rinbo_fpga_driver/build/fpga_driver >/tmp/fpga_driver.log 2>&1 &
```

確認只有一組：

```bash
ps -ef | egrep "grpccore|fpga_driver" | grep -v grep
netstat -tn | grep 50051 || echo "NO TCP on 50051"
```

如果看到兩個 `fpga_driver`，先停掉重來。

## Step 4：啟動 Jetson 的 BioRoLa ROS bridge

Jetson Terminal 1：

```bash
cd ~/rinbo_ros_ws
source /opt/ros/humble/setup.bash
source install/setup.bash
export CORE_MASTER_ADDR=<SBRIO_IP>:50051
export CORE_LOCAL_IP=<ORIN_IP>
ros2 run rinbo_ros_bridge rinbo_ros_bridge
```

Jetson Terminal 2 檢查 topic：

```bash
source /opt/ros/humble/setup.bash
source ~/rinbo_ros_ws/install/setup.bash
ros2 topic list | grep -E "motor|power"
```

你應該看到：

```text
/motor/command
/motor/state
/power/command
/power/state
```

再跑：

```bash
ros2 run redrhex_lowlevel_bridge rinbo_bringup_check
```

## Step 5：開 power，但還不要跑 RL

Jetson Terminal 2：

```bash
source /opt/ros/humble/setup.bash
source ~/rinbo_ros_ws/install/setup.bash
source ~/RedRhex/RedRhex/ros2_ws/install/setup.bash
```

開 digital：

```bash
ros2 run redrhex_lowlevel_bridge rinbo_power_tool digital
```

開 sensor signal：

```bash
ros2 run redrhex_lowlevel_bridge rinbo_power_tool signal
```

先看 `/power/state`：

```bash
ros2 topic echo /power/state --once
```

確認電壓電流合理後，才開 relay：

```bash
ros2 run redrhex_lowlevel_bridge rinbo_power_tool relay
```

## Step 6：校正與 standing

此時還不要開 RL controller。

校正：

```bash
ros2 run rinbo_fsm rinbo_cali
```

腳應該會往 Hall sensor 方向轉並停止。

站姿：

```bash
ros2 run rinbo_fsm rinbo_standing
```

完成後，確認沒有舊的運動 node 繼續佔用 `/motor/command`：

```bash
pkill -f rinbo_tripod || true
pkill -f rinbo_standing || true
pkill -f rinbo_cali || true
ros2 topic info /motor/command -v
```

如果 `/motor/command` 還有其他 publisher，不要啟動 RL。

## Step 7：BioRoLa preview-only，不送真馬達

這一步會讓 RedRhex RL pipeline 跑起來，但 bridge 不會真的發 enabled command 到 `/motor/command`。

Jetson Terminal 3：

```bash
cd ~/RedRhex/RedRhex/ros2_ws
source /opt/ros/humble/setup.bash
source ~/rinbo_ros_ws/install/setup.bash
source install/setup.bash

ros2 launch redrhex_lowlevel_bridge lowlevel_bridge.launch.py \
  bridge_backend:=biorola_ros \
  rinbo_allow_enable:=false \
  rinbo_main_max_pwm:=150.0
```

Jetson Terminal 4：

```bash
cd ~/RedRhex/RedRhex/ros2_ws
source /opt/ros/humble/setup.bash
source ~/rinbo_ros_ws/install/setup.bash
source install/setup.bash

ros2 launch redrhex_rl_controller redrhex_policy_bringup.launch.py \
  onnx_path:=/home/jetson/RedRHex/policy.onnx
```

清 E-stop：

```bash
ros2 run redrhex_rl_controller estop_tool off
```

允許 controller 進 INIT_STAND，但 bridge 仍是 preview-only：

```bash
ros2 topic pub --once /redrhex/enable_motors std_msgs/msg/Bool "{data: true}"
```

檢查 preview：

```bash
ros2 topic echo /redrhex/rinbo_motor_command_preview --once
ros2 topic echo /redrhex/lowlevel_diagnostics --once
ros2 topic echo /redrhex/power_safety_trip --once
```

這一步如果有 current trip、NaN、topic timeout，不要進下一步。

## Step 8：單顆馬達測試

這一步會真的讓 bridge 發 enabled command。機器人要架空，手上要有急停。

停止 RL controller，只留下 lowlevel bridge。

重新啟動 bridge，允許 enable，但 PWM 上限仍很低：

```bash
ros2 launch redrhex_lowlevel_bridge lowlevel_bridge.launch.py \
  bridge_backend:=biorola_ros \
  rinbo_allow_enable:=true \
  rinbo_main_max_pwm:=80.0
```

先測一顆 ABAD，超小角度：

```bash
ros2 run redrhex_rl_controller motor_command_tool abad \
  --index 0 \
  --position 0.03 \
  --enable \
  --confirm-risk
```

立刻 disable：

```bash
ros2 run redrhex_rl_controller motor_command_tool disable
```

再測一顆 main drive，超低速度：

```bash
ros2 run redrhex_rl_controller motor_command_tool main \
  --index 0 \
  --velocity 0.10 \
  --enable \
  --confirm-risk
```

立刻 disable：

```bash
ros2 run redrhex_rl_controller motor_command_tool disable
```

看電流：

```bash
ros2 topic echo /redrhex/lowlevel_diagnostics --once
ros2 topic echo /power/state --once
```

如果 `/redrhex/power_safety_trip` 是 true：

```bash
ros2 topic pub --once /redrhex/enable_motors std_msgs/msg/Bool "{data: false}"
ros2 service call /redrhex/clear_power_safety_trip std_srvs/srv/Trigger "{}"
```

注意：只有 `rinbo_allow_enable:=false` 或已停止 enable 狀態後，才允許清除 latch。

## Step 9：架空 policy dry-run

機器人架空，PWM 上限先用 `80`。

Terminal 3：bridge：

```bash
ros2 launch redrhex_lowlevel_bridge lowlevel_bridge.launch.py \
  bridge_backend:=biorola_ros \
  rinbo_allow_enable:=true \
  rinbo_main_max_pwm:=80.0
```

Terminal 4：controller：

```bash
ros2 launch redrhex_rl_controller redrhex_policy_bringup.launch.py \
  onnx_path:=/home/jetson/RedRHex/policy.onnx
```

清 E-stop：

```bash
ros2 run redrhex_rl_controller estop_tool off
```

只讓它 INIT_STAND，不開 policy：

```bash
ros2 topic pub --once /redrhex/enable_motors std_msgs/msg/Bool "{data: true}"
ros2 topic echo /redrhex/state_machine_state
```

確認狀態到 `POLICY_READY` 後，再開非常小的命令：

```bash
ros2 topic pub --once /redrhex/enable_policy std_msgs/msg/Bool "{data: true}"
ros2 topic pub --rate 10 /cmd_vel geometry_msgs/msg/Twist \
  "{linear: {x: 0.05, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}"
```

只跑 2 秒，然後停：

```bash
ros2 run redrhex_rl_controller estop_tool on
ros2 topic pub --once /redrhex/enable_policy std_msgs/msg/Bool "{data: false}"
ros2 topic pub --once /redrhex/enable_motors std_msgs/msg/Bool "{data: false}"
```

檢查：

```bash
ros2 topic echo /redrhex/diagnostics --once
ros2 topic echo /redrhex/lowlevel_diagnostics --once
ros2 topic echo /redrhex/power_safety_trip --once
```

## Step 10：落地低速 2 秒

只有在以下全部通過後才做：

- ONNX check passed
- mock passed
- sbRIO heartbeat ok
- `/motor/state` ok
- `/power/state` ok
- calibration ok
- standing ok
- preview-only ok
- 單顆 ABAD ok
- 單顆 main ok
- 架空 policy dry-run ok
- 沒有 power safety trip
- 沒有異常電流
- 旁邊有人拿急停

落地後仍使用：

- `rinbo_main_max_pwm:=80.0`
- `cmd_vel x <= 0.05`
- 只跑 2 秒

不要一開始側移、轉向、斜走。先只測很小的 forward。

## 常用停機指令

軟體 E-stop：

```bash
ros2 run redrhex_rl_controller estop_tool on
```

disable RedRhex high-level command：

```bash
ros2 run redrhex_rl_controller motor_command_tool disable
```

關 power relay：

```bash
ros2 run redrhex_lowlevel_bridge rinbo_power_tool signal
```

全部 power off：

```bash
ros2 run redrhex_lowlevel_bridge rinbo_power_tool off
```

停掉舊 controller：

```bash
pkill -f redrhex_rl_controller || true
pkill -f redrhex_lowlevel_bridge || true
pkill -f rinbo_tripod || true
```

## Topics

Controller subscriptions：

- `/imu/data`: `sensor_msgs/msg/Imu`
- `/joint_states`: `sensor_msgs/msg/JointState`
- `/cmd_vel`: `geometry_msgs/msg/Twist`
- `/estop`: `std_msgs/msg/Bool`
- `/redrhex/enable_motors`: `std_msgs/msg/Bool`
- `/redrhex/enable_policy`: `std_msgs/msg/Bool`
- `/redrhex/lowlevel_heartbeat`: `std_msgs/msg/Bool`
- `/motor_feedback`: `redrhex_msgs/msg/RedRhexMotorState`

Controller publications：

- `/redrhex/observation`
- `/redrhex/policy_action_raw`
- `/redrhex/policy_action_safe`
- `/redrhex/motor_commands`
- `/redrhex/state_machine_state`
- `/redrhex/diagnostics`

Lowlevel bridge publications：

- `/redrhex/lowlevel_heartbeat`
- `/redrhex/power_safety_trip`
- `/redrhex/lowlevel_diagnostics`
- `/redrhex/rinbo_motor_command_preview`

Lowlevel bridge service：

- `/redrhex/clear_power_safety_trip`: `std_srvs/srv/Trigger`

## YAML 重要參數

`config/redrhex_policy.yaml`：

```yaml
policy.onnx_path: "/home/jetson/RedRHex/policy.onnx"
safety.estop_on_start: true
safety.enable_motor_output_on_start: false
safety.main_drive_vel_limit_rad_s: 1.0
safety.main_drive_slew_rate_rad_s2: 4.0
safety.abad_pos_limit_rad: 0.15
safety.abad_slew_rate_rad_s: 1.0
observation.base_lin_vel_source: "zero"
observation.abad_feedback_source: "commanded"
```

`redrhex_lowlevel_bridge/config/lowlevel_bridge.yaml`：

```yaml
bridge.backend: "mock"
rinbo.allow_enable: false
rinbo.main_max_pwm: 150.0
rinbo.require_power_state_when_enabled: true
rinbo.max_main_channel_current_a: 3.0
rinbo.max_bus_current_a: 12.0
rinbo.command_timeout_s: 0.10
```

`rinbo.min_bus_voltage_v` 和 `rinbo.max_bus_voltage_v` 現在用 `0.0` 表示不啟用，因為 ROS2 YAML parameter 對 null 不方便。

## 已知風險與 TODO

- `base_lin_vel` 初期設 0，真機 locomotion 可能不穩。
- ABAD 沒有回授，目前用 command target 當 observation，會有 sim2real gap。
- ABAD servo encoder `counts_per_rad=1000` 是初始估計，需要實測校正。
- main drive 正反方向可能和 IsaacLab joint axis 相反，需要單顆馬達測試修正。
- encoder zero offset 可能和 IsaacLab init pose 不一致，需要校正。
- `/power/state` channel mapping 目前依 sbRIO driver 與操作經驗假設：`i_1..i_6` 是 main channels，`i_7` 是 bus current。
- Jetson ROS2 軟體限流不是硬體 current limit。
- `rinbo_ros_bridge` / sbRIO gRPC 網路延遲、封包遺失會造成 sim2real gap。
- 不要跳過 INIT_STAND。
- 不要把 raw action 直接送 `/motor/command`。

## 明天最短安全路線

照這個順序打勾：

```text
[ ] 1. policy.onnx 複製到 /home/jetson/RedRHex/policy.onnx
[ ] 2. colcon build 成功
[ ] 3. preflight_check PASS
[ ] 4. check_onnx_io PASS
[ ] 5. mock test 有 /redrhex/observation 和 /redrhex/motor_commands
[ ] 6. sbRIO 只開一個 grpccore / fpga_driver
[ ] 7. rinbo_ros_bridge 看到 /motor/state /power/state
[ ] 8. power digital -> signal -> relay 正常
[ ] 9. rinbo_cali 正常
[ ] 10. rinbo_standing 正常
[ ] 11. 確認沒有 rinbo_tripod 或其他 /motor/command publisher
[ ] 12. BioRoLa preview-only 正常
[ ] 13. 單顆 ABAD 小角度正常
[ ] 14. 單顆 main 小速度正常
[ ] 15. 架空 INIT_STAND 正常
[ ] 16. 架空 policy 2 秒正常
[ ] 17. 落地 forward 0.05 m/s 2 秒
```

只要任何一步怪怪的，停下來，不要往下一步。
