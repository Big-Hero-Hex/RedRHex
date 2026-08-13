---
id: sensor-distillation-v2-architecture
title: Sensor-Only Distillation V2 架構
lang: zh-TW
audience: developer
type: explanation
status: active
owner: training
last_reviewed: 2026-08-14
---

<a id="purpose"></a>
## 用途

Sensor Distillation V2 是 additive forward-only research architecture，用來測試一秒 measurable temporal feedback 能否取代 simulator velocity 與 controller-state inputs，同時不影響 V1。Task ID 是 `Template-Redrhex-ForwardSensorV2-Direct-v0`；選擇任何 legacy task 就回到 legacy contract。

<a id="data-flow"></a>
## 資料流

Timestamped IMU 與 measured main/ABAD encoder events 進入 `redrhex_policy_io`。其 causal preprocessor 產生 strict 36-D frame。60-sample oldest-to-newest buffer 與 current three-value command 共同輸入 TCN actor。六個 learned main outputs 是 versioned forward CPG 周圍的 residuals；六個 ABAD outputs 全為 neutral。True base velocity 與 physical randomization state 只能作為 training targets 或 Teacher A/critic inputs，絕不是 actor inputs。

相同 contracts 與 hashes 會經過 simulation、real-trace replay、checkpoint manifests、custom two-input ONNX exporter、V2 ROS builder 與 deployment preflight。Mismatch 會停止載入，不會從 tensor dimensions 猜測 semantics。

<a id="training-stages"></a>
## 訓練階段

- F0 以 zero policy residuals 證明 procedural controller。
- F1 訓練 65-D physically privileged Teacher A。加入 controller targets 的 Teacher B 是隔離的 ablation。
- F2 透過 rollout mixing 與 velocity/next-frame auxiliaries 將 Teacher A distill 到 causal TCN。
- F3 使用 asymmetric physical critic、annealed teacher BC 與持續 auxiliaries fine-tune 完全相同的 actor。
- F4 加入有 evidence 的 sensor/actuator randomization 與 raw-event replay。
- F5 使用既有 command-sweep acceptance core，比較 three-seed policy lineages 與 ablations。

後續 stage 不得補償前一 stage 的 mapping、parity 或 provenance gate failure。

可執行的 V2 backends 在 F1 保留既有 RSL-RL PPO implementation，但以 strict V2 format 取代 checkpoint writer。F2 負責三條 action streams 與 terminal-masked next-frame targets。F3 負責 asymmetric rollout、GAE/minibatches、distilled actor exact bootstrap，以及保留的 Teacher A state。CLI 可啟動各 stage，或啟動 fail-closed 的 F1 → F2 → F3 sequential pipeline；它絕不從 tensor shape 猜測 transition。復原出的 browser route 留在堆疊的 Panel proposal branch，等待分開審查。

<a id="boundaries"></a>
## 邊界

Contact supervision 不存在，因為目前 simulator contact state 是 phase proxy。Production IMU attitude mode 與 ABAD calibration 尚未驗證。Learned ABAD、direct targets、lateral/yaw expansion、hardware motor enable 與 physical promotion 都不在此 architecture 範圍內，直到有新 evidence 與 design approval。堆疊的 Panel proposal 僅負責 launch、strict checkpoint handoff、monitoring 與 final-artifact routing；它不能 promote model，也不能放寬 evidence gate。

<a id="verification-status"></a>
## 驗證狀態

Dependency-light contract、training、replay 與 ROS tests 可建立 implementation correctness。F1、F2、F3 與 sequential pipeline 的 one-update Isaac gates 已完成。Deterministic forward acceptance、full-run quality、three-seed results、recorded-sensor replay 與 hardware preflight 仍是分開且待完成的 evidence gates。目前完成狀態請看 [active plan](../../plans/active/2026-08-13-student-distillation-v2.zh-TW.md)，精確 interfaces 請看 [approved design](../../designs/active/2026-08-13-student-distillation-v2.zh-TW.md)。
