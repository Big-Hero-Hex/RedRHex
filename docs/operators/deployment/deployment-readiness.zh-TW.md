---
id: operator-deployment-readiness
title: 驗證 Policy 部署就緒狀態
lang: zh-TW
audience: operator
type: safety
status: active
owner: deployment
last_reviewed: 2026-08-07
---

<a id="gate"></a>
## 部署 gate

絕不可從成功的訓練直接跳到通電真機。請選定一個 `model_*.pt`、匯出，並讓其設定與 readiness report 一起保存。Readiness 為 `blocked` 時必須停止部署；`review` 代表仍有警告或缺少選用證據，需由人員審查。

<a id="panel-check"></a>
## 執行面板 readiness

開啟 Training Panel 的 Deploy 頁面，選擇完成的 run，再使用 Validate Existing ONNX 或 Export ONNX + Validate。必要階段涵蓋 export integrity、static ONNX、ONNX Runtime、Torch/ONNX parity、observation/action contract 與 safety fault injection。ROS mock 與未校準 MuJoCo 結果只是輔助證據，不代表可以讓機器人通電。

報告寫入 `<run>/deploy/readiness_<pipeline-id>.json` 與 `.md`。

<a id="jetson-preflight"></a>
## 執行 Jetson preflight

複製經審查的 ONNX、部署 YAML 與 readiness report。在 Jetson 執行：

```bash
source /opt/ros/humble/setup.bash
source /path/to/ros2_ws/install/setup.bash
ros2 run redrhex_rl_controller preflight_check \
  --onnx /path/to/policy.onnx \
  --config /path/to/redrhex_policy.yaml
```

目前 contract 為單幀 56 維 observation、含 history 時 280 維、12 維 action，以及 60 Hz policy loop。

<a id="hardware-boundary"></a>
## 硬體邊界

任何 motor enable 前，必須準備實體急停與斷電方式、確認只有一個 bridge publisher、驗證 heartbeat、保持 `enable_policy_on_start=false` 與 `enable_motor_output_on_start=false`，再以低功率測試單顆 ABAD，並在架空狀態測試單顆 main drive。後續請依共置的 [ROS bring-up 指南](../../../ros2_ws/src/redrhex_rl_controller/docs/bring-up.zh-TW.md)操作。
