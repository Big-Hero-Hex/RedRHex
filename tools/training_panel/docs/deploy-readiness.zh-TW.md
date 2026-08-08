---
id: training-panel-deploy-readiness
title: Training Panel 部署就緒檢查
lang: zh-TW
audience: operator
type: safety
status: active
owner: deployment
last_reviewed: 2026-08-07
---

<a id="levels"></a>
## Readiness 等級

`ready` 表示必要 export、ONNX、runtime、parity、contract 與 safety check 通過。`review` 表示沒有必要項失敗，但至少一個輔助檢查警告或跳過。`blocked` 表示至少一個必要階段失敗；不可繼續 robot bring-up。

<a id="runtime"></a>
## Runtime 分工

Training、playback、video 與 export 透過 Isaac launcher 執行。Readiness analysis 在 panel Python 執行，需要 `onnx`、`onnxruntime`、`torch` 與可選的 `mujoco`。Deploy defaults API 與 process log 會指出確切 interpreter 與 dependency 狀態。

<a id="stages"></a>
## 階段

必要階段驗證 checkpoint/export file 與 hash、ONNX shape/inference、Torch/ONNX parity、56/280 observation 與 12-action contract、60 Hz、joint/scaling limit 與 synthetic safety fault。ROS mock 為可選。MuJoCo 在 configuration 標示 `calibrated=false` 時只是 advisory；只有結合已審查 calibration 才具有更強意義。

<a id="artifacts"></a>
## 產物

Readiness JSON/Markdown 與 MuJoCo trace 寫在選定 run 的 `deploy/` 目錄下。它們只屬於該 run，不是 source artifact。交付 Jetson 時，請一起複製已審查 ONNX、deployment YAML 與 readiness report。

<a id="hardware"></a>
## 硬體 gate

Hardware enable 前執行 Jetson preflight、source 正確 ROS workspace、確認實體 E-stop 與 cutoff、bridge heartbeat、單一 publisher、低功率單顆 ABAD、架空單顆 main drive，以及停用 policy/motor startup flag。乾淨的 MuJoCo 影片不能取代這些檢查。

<a id="rollback"></a>
## Rollback

出現非預期行為時，觸發 software 與 physical stop、停用 motor output、還原上一個已審查 bundle、重新執行 readiness/preflight，並在 run note 記錄失敗 report 路徑。
