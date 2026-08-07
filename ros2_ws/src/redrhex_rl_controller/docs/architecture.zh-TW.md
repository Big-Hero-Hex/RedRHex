---
id: ros2-deployment-architecture
title: ROS 2 部署架構
lang: zh-TW
audience: developer
type: explanation
status: active
owner: deployment
last_reviewed: 2026-08-07
---

<a id="ros2-deployment-architecture"></a>
# ROS 2 部署架構

部署堆疊分離學習推論、安全決策與硬體傳輸，因此匯出的 action 絕不會直接送往馬達驅動器。

<a id="data-path"></a>
## 資料路徑

```text
IMU + joint states + command
  -> ObservationBuilder
  -> PolicyONNXRunner
  -> ActionDecoder
  -> SafetyFilter + RedRhexStateMachine
  -> /redrhex/motor_commands
  -> redrhex_lowlevel_bridge
  -> 低階控制板與馬達驅動器
```

`rl_controller_node` 負責高階路徑，並發布 observation、raw-action、safe-action、motor-command、state 與 diagnostic topics。低階 bridge 負責硬體協定。任何時候只能有一個 publisher 能到達實體 motor-command topic。

<a id="package-boundaries"></a>
## 套件邊界

- `ObservationBuilder` 建立訓練時的觀測契約及五幀歷史。
- `PolicyONNXRunner` 驗證 ONNX 輸入輸出形狀並執行推論。
- `ActionDecoder` 將 12 個正規化輸出映射成六顆 main-drive 與六顆 ABAD 目標。
- `SafetyFilter` 裁切命令並執行設定的動作限制。
- `RedrhexStateMachine` 控制 readiness、warm-up、policy、motor-output 與 protective-stop 閘門。
- `redrhex_lowlevel_bridge` 將安全命令轉成 `mock`、`serial`、`sbrio_udp` 或 Rinbo/Biorola 傳輸。

六個模擬 damper joints 是被動彈簧機構，沒有 policy action，也沒有實體馬達命令。

<a id="interfaces"></a>
## 介面

控制器訂閱 `/imu/data`、`/joint_states`、`/cmd_vel`、`/estop`、啟用／復原 topics、低階 heartbeat、馬達回授、電池狀態及選用 odometry。它在 `/redrhex/` 下發布 `observation`、`policy_action_raw`、`policy_action_safe`、`motor_commands`、`state_machine_state` 與 `diagnostics` 等 topics。

Launch file 可啟動 fake sensors 與指定 bridge 進行 dry-run。Policy 與馬達輸出預設皆為停用。

<a id="hardware-feedback-assumptions"></a>
## 硬體回授假設

目前 MVP 假設有真實 IMU 資料與六顆 main-drive encoder position。它可用差分估計 main-drive velocity；若沒有實體 ABAD 回授，`abad_feedback_source: commanded` 會以先前 ABAD 命令代替。這項替代是已知 sim-to-real 限制，不等同於 encoder。

<a id="evolution-boundary"></a>
## 演進邊界

在低階協定穩定前保留 bridge 抽象。未來可用 `ros2_control` hardware plugin 替換傳輸而不改變學習策略契約，但這並非目前行為。
