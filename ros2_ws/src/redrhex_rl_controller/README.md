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

注意：如果 zero observation 的 raw action 大於 `[-1, 1]`，不一定代表 policy 壞掉。這個 IsaacLab task 在 `_pre_physics_step()` 會先 clamp action，本部署包也會在 `ActionDecoder` 先 clip 再做限速、限角與 PWM 限制。真正不能接受的是 NaN / Inf、維度錯、ONNX 無法載入。

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

## Step 3：標準 R-Slip 硬體 bringup 網路模式

以下硬體 bringup 流程依照實驗室標準 SOP：

```text
/home/jetson/Downloads/R-Slip實驗操作流程.md
```

先選一個網路模式，後面整段都不要混用 IP：

| 模式 | Orin Nano IP | sbRIO IP | 用途 |
| --- | --- | --- | --- |
| A. 實驗室網路線 | `192.168.30.8` | `192.168.30.2` | 目前實驗室常用 |
| B. 機器人 Wi-Fi | `192.168.0.101` | `192.168.0.100` | 連機器人 Wi-Fi 分享器時使用 |

規則：

- `CORE_MASTER_ADDR` 永遠是 `<SBRIO_IP>:50051`。
- sbRIO 上的 `CORE_LOCAL_IP` 填 sbRIO 自己的 IP。
- Orin 上的 `CORE_LOCAL_IP` 填 Orin 自己的 IP。
- `Orin1` 的 `rinbo_ros_bridge` 啟動後必須保持開啟。
- 標準 R-Slip SOP 會在最後跑 `rinbo_tripod`；**要接 RL controller 時不要跑 `rinbo_tripod`**。

快速連線檢查可以在 Windows PowerShell 執行：

模式 A：

```powershell
Test-NetConnection 192.168.30.2 -Port 22
Test-NetConnection 192.168.30.8 -Port 22
arp -a
```

模式 B：

```powershell
Test-NetConnection 192.168.0.100 -Port 22
Test-NetConnection 192.168.0.101 -Port 22
arp -a
```

## Step 4：啟動 sbRIO / FPGA driver

### 模式 A：實驗室網路線

在 sbRIO terminal：

```bash
ssh admin@192.168.30.2
```

登入後貼上：

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

### 模式 B：機器人 Wi-Fi

在 sbRIO terminal：

```bash
ssh admin@192.168.0.100
```

登入後貼上：

```bash
cd ~/rinbo_sbRIO_ws/rinbo_fpga_driver/build/
pkill -f fpga_driver || true
pkill -f grpccore || true
export CORE_LOCAL_IP=192.168.0.100
export CORE_MASTER_ADDR=192.168.0.100:50051
nohup /home/admin/rinbo_sbRIO_ws/install/bin/grpccore >/tmp/grpccore.log 2>&1 &
nohup /home/admin/rinbo_sbRIO_ws/rinbo_fpga_driver/build/fpga_driver >/tmp/fpga_driver.log 2>&1 &
ps -ef | egrep "grpccore|fpga_driver" | grep -v grep
netstat -tn | grep 50051 || netstat -ltn | grep 50051 || echo "NO TCP on 50051"
```

正常情況：

- `ps` 只看到一組 `grpccore` 和一組 `fpga_driver`。
- `netstat` 看到 `50051`。如果只有 listen，通常是 `rinbo_ros_bridge` 還沒連上；啟動 Orin bridge 後再檢查一次。

## Step 5：啟動 Jetson / Orin 的 Rinbo ROS Bridge

開一個 Orin terminal，後面叫 `Orin1`。這個 terminal 啟動 bridge 後不要關。

### 模式 A：實驗室網路線

```bash
ssh jetson@192.168.30.8
```

登入 Orin 後：

```bash
cd ~/rinbo_ros_ws/
source /opt/ros/humble/setup.bash
source ~/rinbo_ros_ws/install/setup.bash
export CORE_MASTER_ADDR=192.168.30.2:50051
export CORE_LOCAL_IP=192.168.30.8
ros2 run rinbo_ros_bridge rinbo_ros_bridge
```

### 模式 B：機器人 Wi-Fi

```bash
ssh jetson@192.168.0.101
```

登入 Orin 後：

```bash
cd ~/rinbo_ros_ws/
source /opt/ros/humble/setup.bash
source ~/rinbo_ros_ws/install/setup.bash
export CORE_MASTER_ADDR=192.168.0.100:50051
export CORE_LOCAL_IP=192.168.0.101
ros2 run rinbo_ros_bridge rinbo_ros_bridge
```

再開第二個 Orin terminal，後面叫 `Orin2`。所有檢查、power、calibration、RL command 都在 `Orin2` 做。

`Orin2` 模式 A 先登入：

```bash
ssh jetson@192.168.30.8
```

登入 Orin 後貼上：

```bash
source /opt/ros/humble/setup.bash
source ~/rinbo_ros_ws/install/setup.bash
source ~/RedRhex/RedRhex/ros2_ws/install/setup.bash
export CORE_MASTER_ADDR=192.168.30.2:50051
export CORE_LOCAL_IP=192.168.30.8
```

`Orin2` 模式 B 先登入：

```bash
ssh jetson@192.168.0.101
```

登入 Orin 後貼上：

```bash
source /opt/ros/humble/setup.bash
source ~/rinbo_ros_ws/install/setup.bash
source ~/RedRhex/RedRhex/ros2_ws/install/setup.bash
export CORE_MASTER_ADDR=192.168.0.100:50051
export CORE_LOCAL_IP=192.168.0.101
```

在 `Orin2` 確認 topic：

```bash
ros2 topic list | grep -E "motor|power"
```

你應該看到：

```text
/motor/command
/motor/state
/power/command
/power/state
```

也可以跑：

```bash
ros2 run redrhex_lowlevel_bridge rinbo_bringup_check
```

## Step 6：標準 power、校正與 standing

此時還不要開 RL controller。

以下 power 指令使用標準 R-Slip SOP 的 `/power/command` 直接發布方式；這是目前實驗室已驗證的開機方式。

在 `Orin2` 依序貼上，每貼完一個等它執行完再貼下一個。

開啟機器人 digital：

```bash
ros2 topic pub --once /power/command rinbo_msgs/msg/PowerCmdStamped "{header: {seq: 1, stamp: {sec: 0, nanosec: 0}}, digital: true, signal: false, power: false}"
```

開啟感測器 signal：

```bash
ros2 topic pub --once /power/command rinbo_msgs/msg/PowerCmdStamped "{header: {seq: 1, stamp: {sec: 0, nanosec: 0}}, digital: true, signal: true, power: false}"
```

先看 `/power/state`：

```bash
ros2 topic echo /power/state --once
```

確認電壓電流合理後，才開 relay：

```bash
ros2 topic pub --once /power/command rinbo_msgs/msg/PowerCmdStamped "{header: {seq: 1, stamp: {sec: 0, nanosec: 0}}, digital: true, signal: true, power: true}"
ros2 topic echo /power/state --once
```

接著校正：

```bash
ros2 run rinbo_fsm rinbo_cali
```

腳應該會往 Hall sensor 方向轉並停止。要看到 6/6 legs 完成；如果卡在 5/6 或某顆 Hall 一直沒觸發，停止，不要進 RL。

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

如果 `/motor/command` 還有 `rinbo_tripod` 或其他未知 publisher，不要啟動 RL。

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

停止 RL controller，只留下 `rinbo_ros_bridge` 和本步驟啟動的 `redrhex_lowlevel_bridge`。

注意目前 Jetson 上有兩個 overlay：

- `~/RedRhex/RedRhex/ros2_ws`：本 repo 的 RL lowlevel bridge，支援 `rinbo_disabled_legs:=l1`。
- `~/rinbo_ros_ws`：標準 R-Slip/Rinbo workspace，`motor_command_tool` 的命令名稱是 `single-abad` / `single-main-velocity`。

所以 Step 8 請照下面兩個 terminal 的 source 順序跑，不要混在同一個 terminal 裡猜 overlay。

重要：**不要用 `env -i ... bash --noprofile --norc` 來跑 Step 8 motor command。**  
`env -i` 會清掉 ROS/DDS discovery 需要的環境，結果就是明明 bridge 開著，`motor_command_tool` 仍然報：

```text
No subscriber on /redrhex/motor_commands. Start redrhex_lowlevel_bridge before manual motor tests.
```

正確做法是在 command terminal 裡只清掉 ROS overlay 變數，再 source `~/rinbo_ros_ws`。

### Terminal 3：啟動 lowlevel bridge，保持開著

如果 L1 腳目前有問題，本次測試用 `rinbo_disabled_legs:=l1`。如果沒有壞腳，把最後一行刪掉即可。

```bash
cd ~/RedRhex/RedRhex/ros2_ws
source /opt/ros/humble/setup.bash
source ~/rinbo_ros_ws/install/setup.bash
source install/setup.bash

ros2 launch redrhex_lowlevel_bridge lowlevel_bridge.launch.py \
  bridge_backend:=biorola_ros \
  rinbo_allow_enable:=true \
  rinbo_main_max_pwm:=80.0 \
  rinbo_disabled_legs:=l1
```

你要看到類似：

```text
BioRoLa/Rinbo backend connected. allow_enable=True; command_topic=/motor/command; disabled_legs=['l1']
```

這個 terminal 不要按 `Ctrl+C`。如果 bridge 停了，後面的 `motor_command_tool` 只會發布到 `/redrhex/motor_commands`，不會有東西轉成 `/motor/command`，馬達就不會動。

### Terminal 4：確認 bridge 與 power

```bash
source /opt/ros/humble/setup.bash
source ~/rinbo_ros_ws/install/setup.bash
source ~/RedRhex/RedRhex/ros2_ws/install/setup.bash

ros2 topic list | grep -E "motor|power|redrhex"
ros2 topic echo /redrhex/lowlevel_heartbeat --once
ros2 topic echo /redrhex/lowlevel_diagnostics --once
ros2 topic echo /redrhex/power_safety_trip --once
ros2 topic echo /power/state --once
```

要確認：

- `/redrhex/lowlevel_heartbeat` 是 `true`。
- `/redrhex/power_safety_trip` 是 `false`。
- `/power/state` 裡 `digital: true`、`signal: true`、`power: true`。
- diagnostics 裡 `allow_enable` 是 `True`，如果使用壞腳隔離，`disabled_legs` 是 `['l1']`。

### Terminal 4：使用新版 motor_command_tool 做單顆測試

這裡要先清掉目前 shell 裡可能殘留的 overlay，再只 source `~/rinbo_ros_ws`，因為這個 workspace 裡的工具命令名稱才是 `single-abad` / `single-main-velocity`。

不要開 `env -i` clean shell；那會讓 command tool 看不到正在跑的 lowlevel bridge。

```bash
unset AMENT_PREFIX_PATH COLCON_PREFIX_PATH CMAKE_PREFIX_PATH PYTHONPATH LD_LIBRARY_PATH
source /opt/ros/humble/setup.bash
source ~/rinbo_ros_ws/install/setup.bash
```

先確認這個 terminal 現在用的是 `rinbo_ros_ws` 的新版工具：

```bash
ros2 pkg prefix redrhex_rl_controller
ros2 run redrhex_rl_controller motor_command_tool --help
```

你必須看到：

```text
/home/jetson/rinbo_ros_ws/install/redrhex_rl_controller
...
single-abad
single-main-velocity
```

再確認 command tool 可以看到 lowlevel bridge subscriber：

```bash
ros2 topic info /redrhex/motor_commands -v
ros2 topic echo /redrhex/lowlevel_heartbeat --once
ros2 topic echo /redrhex/power_safety_trip --once
```

你必須看到：

```text
Subscription count: 1
data: true
data: false
```

如果這裡 `Subscription count: 0` 或 `Unknown topic`，不要跑馬達命令；回去確認 Terminal 3 的 `redrhex_lowlevel_bridge` 還活著。

可以列出 policy-order joint index：

```bash
ros2 run redrhex_rl_controller motor_command_tool list-joints
```

先測一顆 ABAD，超小角度，持續 2 秒：

```bash
ros2 run redrhex_rl_controller motor_command_tool single-abad --index 0 --position 0.03 --enable --confirm-risk --duration 2.0 --wait-for-subscriber-s 10.0
```

立刻 disable：

```bash
ros2 run redrhex_rl_controller motor_command_tool disable --wait-for-subscriber-s 10.0
```

再測一顆 main drive，超低速度：

```bash
ros2 run redrhex_rl_controller motor_command_tool single-main-velocity --index 0 --velocity 0.10 --enable --confirm-risk --duration 2.0 --wait-for-subscriber-s 10.0
```

立刻 disable：

```bash
ros2 run redrhex_rl_controller motor_command_tool disable --wait-for-subscriber-s 10.0
```

看電流：

```bash
ros2 topic echo /redrhex/lowlevel_diagnostics --once
ros2 topic echo /power/state --once
```

如果完全沒有動，照順序查：

```bash
ros2 topic echo /redrhex/lowlevel_heartbeat --once
ros2 topic info /redrhex/motor_commands -v
ros2 topic info /motor/command -v
ros2 topic echo /redrhex/rinbo_motor_command_preview --once
```

常見原因：

- Terminal 3 的 lowlevel bridge 被 `Ctrl+C` 停掉了。
- `Orin1` 的 `rinbo_ros_bridge` 沒有保持開啟。
- power relay 沒開，`/power/state` 不是 `power: true`。
- 用錯 overlay，跑到了舊版 `motor_command_tool`。
- 測的是已禁用的腿；`rinbo_disabled_legs:=l1` 會強制 Rinbo `l1` 不輸出。

如果 `/redrhex/power_safety_trip` 是 true：

```bash
ros2 topic pub --once /redrhex/enable_motors std_msgs/msg/Bool "{data: false}"
ros2 service call /redrhex/clear_power_safety_trip std_srvs/srv/Trigger "{}"
```

注意：只有 `rinbo_allow_enable:=false` 或已停止 enable 狀態後，才允許清除 latch。

## Step 9：架空 policy dry-run

從冷開機直接跑 Step 9 時，不要照 README 片段手動拼命令。直接看這份 SOP：

```text
STEP9_POLICY_RUN_COLD_START.md
```

這份 SOP 現在包含：

- 哪些 terminal 是長時間運行的，不要再貼其他命令進去。
- 哪個 terminal 是唯一 control terminal。
- 如何清掉卡住的 bridge/controller/IMU terminal。
- 如何避免重複啟動 MicroStrain IMU。
- 為什麼必須等 `POLICY_READY`，不能從 `INIT_STAND` 或 `PROTECTIVE_STOP` 直接開 policy。

重要限制：`rinbo_disabled_legs:=l1` 只會讓 lowlevel bridge 不輸出 L1；它不會讓 RL controller 或 policy 真的支援五條腿。若 calibration / standing / INIT_STAND 因壞腿無法完成，Step 9 必須停在 policy enable 之前。

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
ros2 topic pub --once /power/command rinbo_msgs/msg/PowerCmdStamped "{header: {seq: 1, stamp: {sec: 0, nanosec: 0}}, digital: true, signal: true, power: false}"
ros2 topic echo /power/state --once
```

全部 power off：

```bash
ros2 topic pub --once /power/command rinbo_msgs/msg/PowerCmdStamped "{header: {seq: 1, stamp: {sec: 0, nanosec: 0}}, digital: false, signal: false, power: false}"
ros2 topic echo /power/state --once
```

停掉舊 controller：

```bash
pkill -f redrhex_rl_controller || true
pkill -f redrhex_lowlevel_bridge || true
pkill -f rinbo_tripod || true
```

停掉 sbRIO driver：

```bash
ssh admin@<SBRIO_IP>
pkill -f fpga_driver || true
pkill -f grpccore || true
ps -ef | egrep "grpccore|fpga_driver" | grep -v grep
```

如果最後一行沒有輸出，代表 `grpccore` 和 `fpga_driver` 都已關閉。

## 硬體 bringup 故障排除

### SSH 連不上 sbRIO / Orin

先確認你選的網路模式：

| 模式 | sbRIO SSH | Orin SSH |
| --- | --- | --- |
| 實驗室網路線 | `ssh admin@192.168.30.2` | `ssh jetson@192.168.30.8` |
| 機器人 Wi-Fi | `ssh admin@192.168.0.100` | `ssh jetson@192.168.0.101` |

不要把 `192.168.30.255` 或 `192.168.0.255` 拿來 SSH；那是 broadcast address。

### `NO TCP on 50051`

在 sbRIO terminal 檢查：

```bash
ps -ef | egrep "grpccore|fpga_driver" | grep -v grep
tail -n 50 /tmp/grpccore.log
tail -n 50 /tmp/fpga_driver.log
netstat -tn | grep 50051 || netstat -ltn | grep 50051 || echo "NO TCP on 50051"
```

如果沒有看到 `grpccore` 或 `fpga_driver`，回到 Step 4 重新啟動。

### `/motor/state` 或 `/power/state` 看不到

先確認 `Orin1` 的 bridge 還在跑：

```bash
ps -ef | grep rinbo_ros_bridge | grep -v grep
```

如果沒有，回到 Step 5，依照目前網路模式重開 `rinbo_ros_bridge`。

最常見錯誤是 Orin 上的 IP 寫錯。Orin terminal 應該是：

模式 A：

```bash
export CORE_MASTER_ADDR=192.168.30.2:50051
export CORE_LOCAL_IP=192.168.30.8
```

模式 B：

```bash
export CORE_MASTER_ADDR=192.168.0.100:50051
export CORE_LOCAL_IP=192.168.0.101
```

不要把 Orin 上的 `CORE_LOCAL_IP` 填成 sbRIO IP。

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
[ ] 1. 已選網路模式 A 實驗室網路線或 B 機器人 Wi-Fi
[ ] 2. policy.onnx 複製到 /home/jetson/RedRHex/policy.onnx
[ ] 3. colcon build 成功
[ ] 4. preflight_check PASS
[ ] 5. check_onnx_io PASS
[ ] 6. mock test 有 /redrhex/observation 和 /redrhex/motor_commands
[ ] 7. sbRIO 只開一個 grpccore / fpga_driver
[ ] 8. Orin1 的 rinbo_ros_bridge 保持開啟
[ ] 9. Orin2 看到 /motor/state /power/state
[ ] 10. 已用 /power/command 依序 digital -> signal -> relay
[ ] 11. /power/state 電壓電流合理
[ ] 12. rinbo_cali 正常，6/6 legs 完成
[ ] 13. rinbo_standing 正常
[ ] 14. 確認沒有 rinbo_tripod 或其他 /motor/command publisher
[ ] 15. BioRoLa preview-only 正常
[ ] 16. 單顆 ABAD 小角度正常
[ ] 17. 單顆 main 小速度正常
[ ] 18. 架空 INIT_STAND 正常
[ ] 19. 架空 policy 2 秒正常
[ ] 20. 落地 forward 0.05 m/s 2 秒
```

只要任何一步怪怪的，停下來，不要往下一步。
