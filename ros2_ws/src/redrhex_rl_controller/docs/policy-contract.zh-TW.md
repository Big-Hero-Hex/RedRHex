---
id: ros2-policy-contract
title: ROS 2 策略契約
lang: zh-TW
audience: developer
type: reference
status: active
owner: deployment
last_reviewed: 2026-08-07
---

<a id="ros2-policy-contract"></a>
# ROS 2 策略契約

控制器契約必須符合匯出的策略與訓練環境。不可從其他機器人猜測維度、joint 順序、尺度或頻率。

<a id="tensor-contract"></a>
## 張量契約

| 屬性 | 必要值 |
| --- | --- |
| 單幀觀測 | 56 個浮點數 |
| 策略歷史 | 5 幀觀測／280 個浮點數 |
| 策略 action | 12 個浮點數 |
| 模擬步長 | 1/120 秒 |
| Decimation | 2 |
| 策略頻率 | 60 Hz |

56 個數值的詳細觀測順序定義在共用[策略契約](../../../../docs/reference/policy-contract.zh-TW.md)。`policy_hz: 0.0` 表示從 repository 契約推導 60 Hz，不代表停用推論。

<a id="actuator-contract"></a>
## 致動器契約

輸出控制六顆 main-drive 馬達與六顆 ABAD 馬達。Damper joints 是僅存在於模擬的被動 joints，不得加入硬體命令。Signs 與 zero offsets 在單關節測試證明需要修正前保持 identity。

`config/redrhex_policy.yaml` 預設限制包括正規化 action clip 1.0、main-drive 速度上限 30 rad/s、ABAD 位置上限 0.7 rad 及 slew-rate 限制。部署套件旁的 YAML 是執行時真實來源。

<a id="command-and-feedback-contract"></a>
## 命令與回授契約

預設命令範圍為 `vx` 0.0–0.56 m/s、`vy` -0.60–0.60 m/s、`wz` -0.70–0.70 rad/s。預設 base-linear-velocity 來源是 `zero`；切換至 odometry 或 estimator 前必須驗證契約。

目前硬體需要 main-drive encoder position，缺少的 main-drive velocity 可估算。ABAD position 只能在有設定時取自上一筆命令，且必須持續標示為證據限制。

<a id="validation-tools"></a>
## 驗證工具

使用 `scripts/check_onnx_io.py` 檢查 graph I/O，並以 `scripts/compare_onnx_with_torch.py` 比對 Torch/ONNX 一致性。Training Panel 的[部署就緒檢查](../../../../tools/training_panel/docs/deploy-readiness.zh-TW.md)會合併這些檢查、契約與安全證據。
