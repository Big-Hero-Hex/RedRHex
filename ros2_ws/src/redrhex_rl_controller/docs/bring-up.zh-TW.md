---
id: ros2-safe-bring-up
title: 啟動 ROS 2 部署堆疊
lang: zh-TW
audience: operator
type: how-to
status: active
owner: deployment
last_reviewed: 2026-08-07
---

<a id="bring-up-the-ros2-deployment-stack"></a>
# 啟動 ROS 2 部署堆疊

此程序在不授予馬達控制權的情況下證明軟體 graph。第一次 graph 測試時，機器人應保持斷電或實體隔離。

<a id="prerequisites"></a>
## 先決條件

- 已安裝 ROS 2 Humble 與 repository dependencies。
- 匯出的 ONNX artifact 已通過[部署就緒檢查](../../../../tools/training_panel/docs/deploy-readiness.zh-TW.md)。
- 在進行硬體工作前，已備妥實體急停、限流與穩固架空支架。

<a id="build-and-source"></a>
## 建置並載入環境

```bash
source /opt/ros/humble/setup.bash
cd ros2_ws
colcon build --symlink-install
source install/setup.bash
```

若使用 Rinbo/Biorola 硬體，先 source 它的 workspace，再 source 此 repository workspace，讓控制器與低階 bridge 能找到硬體 messages。

<a id="start-a-mock-graph"></a>
## 啟動 mock graph

使用 fake sensors 與 mock bridge 啟動，兩個啟動 enable flags 均保持 false：

```bash
ros2 launch redrhex_rl_controller redrhex_policy_bringup.launch.py \
  use_fake_sensors:=true \
  start_bridge:=true \
  bridge_backend:=mock \
  enable_policy_on_start:=false \
  enable_motor_output_on_start:=false
```

在第二個 terminal source workspace 並檢查：

```bash
ros2 topic echo /redrhex/state_machine_state
ros2 topic echo /redrhex/diagnostics
ros2 topic echo /redrhex/motor_commands
```

確認觀測未逾時、推論維度已接受，而且馬達命令保持停用。

<a id="prove-the-safety-controls"></a>
## 證明安全控制

```bash
ros2 run redrhex_rl_controller estop_tool assert
ros2 run redrhex_rl_controller estop_tool clear --confirm-clear
```

Clear 操作刻意要求確認。ROS E-stop 只是高階保護輸入，絕不能取代實體急停或低階 watchdog。

<a id="exit-criteria"></a>
## 通過條件

只有在 mock graph 持續健康、policy 與 motor enable 能獨立受閘門控制、E-stop assertion 會解除 latches，且沒有非預期 publisher 能控制馬達後，才進入[硬體部署](deployment.zh-TW.md)。
