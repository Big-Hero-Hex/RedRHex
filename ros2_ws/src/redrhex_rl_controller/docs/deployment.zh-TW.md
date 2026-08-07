---
id: ros2-hardware-deployment
title: 將策略部署到 RedRHex 硬體
lang: zh-TW
audience: operator
type: how-to
status: active
owner: deployment
last_reviewed: 2026-08-07
---

<a id="deploy-a-policy-to-redrhex-hardware"></a>
# 將策略部署到 RedRHex 硬體

不可跳過階段。第一次硬體工作的目標是驗證傳輸與方向，不是自主行走。

<a id="prepare-the-station"></a>
## 準備工作站

實體急停握在手上、套用保守限流，並穩固架空機器人。確認只執行一個 `grpccore` 與一個 `fpga_driver`，也沒有其他控制器發布馬達命令。

先 source 硬體 workspace，再 source 此 repository 的 `ros2_ws`。需要時可產生唯讀 terminal 計畫：

```bash
ros2 run redrhex_lowlevel_bridge biorola_bringup_plan \
  --sbrio-ip 192.168.30.12 \
  --orin-ip 192.168.30.164 \
  --onnx-path /home/jetson/redrhex_models/policy.onnx
```

此命令只印出步驟，不會發布馬達命令。

<a id="verify-low-level-readiness"></a>
## 驗證低階就緒狀態

```bash
ros2 run redrhex_lowlevel_bridge biorola_bringup_check --message-timeout-s 5.0
ros2 run redrhex_lowlevel_bridge biorola_power_tool status
```

在 bring-up 沒有 error 且能讀取 digital、signal、power 狀態前，不可開啟 relay。完成 `rinbo_cali` 與 `rinbo_standing`，停止其控制器後才啟動 RedRHex bridge。

<a id="preview-without-authority"></a>
## 在無控制權下預覽

以 `allow_enable=false` 啟動 Rinbo/Biorola bridge，檢查 mapping 與 heartbeat：

```bash
ros2 topic echo /redrhex/rinbo_motor_command_preview --once
ros2 topic echo /redrhex/lowlevel_heartbeat
ros2 topic echo /redrhex/lowlevel_diagnostics --once
```

若 preview 順序、sign、scale、heartbeat 或 publisher 數量錯誤，不可繼續。

<a id="fixed-sim-to-real-probe"></a>
## 擷取固定 sim-to-real probe

只有 preview 成功後、policy control 前才使用 `sim2real_probe`。它是不可調整的架空單顆 main-drive step/coast sequence：以 60 Hz 執行三次、共 990 個 command ticks、總長 16.5 秒、drive segments 為 ±0.25 rad/s，且 probe-only 實體 PWM 上限為 30.0。Main indices 0–4 用於 calibration；index 5 是 holdout。

先預覽 JSON；此動作不建立 ROS node，也不發布訊息：

```bash
ros2 run redrhex_rl_controller sim2real_probe --main-index 0 --dry-run
```

Preview 回報的 scenario ID、SHA-256、rate、repeats、ticks、duration、speed cap 與 PWM cap 全部符合前，不可上電。Enabled run 前必須證明實體急停、保守限流、穩固架空、線材安全與 sbRIO watchdog。隔離 ABAD 電源，或實測 disabled servo mode 後才設定 bridge interlock `probe_abad_disable_verified: true`。CLI confirmation 不能取代這項硬體證據。

停止其他所有 motor-command publishers。Probe 必須是 `/redrhex/motor_commands` 唯一 publisher，並且能看到 subscriber、未逾時且為 true 的 heartbeat、未逾時的 joint state，以及明確的 `/estop=false`。直接記錄 raw BioRoLa topics，不可只錄衍生 feedback：

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

開始錄製後，才在另一個 terminal 給予兩項明確授權：

```bash
ros2 run redrhex_rl_controller sim2real_probe --main-index 0 --enable --confirm-risk --confirm-abad-disable
```

Scheduler 使用 absolute 60 Hz deadlines。若 command lateness 達到一個 period，約 16.7 ms，它會在發布該 tick 前 abort 並送出 disabled packets。不會補送漏掉的 tick，也不會把延遲的 enabled commands 集中 burst。任何 E-stop、heartbeat、joint-state、graph ownership、subscriber 或 process 異常，都必須先按實體急停並診斷，才可重新嘗試。

<a id="start-the-gated-controller"></a>
## 啟動具閘門的控制器

以真實 ONNX path 啟動 `redrhex_policy_bringup.launch.py`，並將 `enable_policy_on_start` 與 `enable_motor_output_on_start` 設為 false。先證明 policy dry-run，同時觀察 state、diagnostics、raw action、safe action 與保持停用的馬達輸出。

`/redrhex/enable_policy` 只會在 state machine 就緒時允許 closed-loop inference。`/redrhex/enable_motors` 獨立允許啟用的馬達輸出。開啟第一道閘門不得連帶開啟第二道。

<a id="increase-authority-in-stages"></a>
## 分階段提高控制權

1. 機器人架空，以小角度測試單顆 ABAD。
2. 以低速測試單顆 main-drive 馬達並驗證 encoder 方向。
3. 以保守限制證明架空 `INIT_STAND`。
4. 架空執行策略並監看 timeouts、姿態、電流、溫度與 heartbeat。
5. 只有前面所有階段通過後，才在實體急停就緒下進行短時間低速落地測試。

任何 sign 或 zero-offset 修正都要記錄在部署 YAML，並重做單關節證據。絕不可將 raw policy action 直接送到 `/motor/command`。

<a id="abort-conditions"></a>
## 中止條件

若發生 heartbeat 或 sensor 逾時、姿態過大、非預期動作、mapping 錯誤、第二個 command publisher、電流／溫度違規或失去操作控制，立即 assert E-stop 並移除馬達電源。重新開始前依[疑難排解](troubleshooting.zh-TW.md)診斷。
