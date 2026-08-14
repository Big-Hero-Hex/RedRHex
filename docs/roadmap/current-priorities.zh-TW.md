---
id: project-roadmap
title: 目前專案優先事項
lang: zh-TW
audience: shared
type: roadmap
status: active
owner: project
last_reviewed: 2026-08-14
---

<a id="validation"></a>
## 證據與移動驗證

- 關閉 reward-preset resolution、command bias、exploration-scale diagnostic、evaluation method identity、energy provenance 與 training/deployment observation parity 的 correctness gate。
- 建立 IMU frame、base velocity、contact behavior、mass/CoM/inertia、joint stop、friction/backlash、passive spring、actuation 與 electrical energy 的 hardware ground truth，並綁定 held-out calibration evidence。
- Baseline comparison 前，凍結 task、command envelope、metric、resolved configuration、code/dependency revision、checkpoint、hardware revision 與 immutable held-out suite。
- One-seed screening 只用於排除不良 candidate。探索至少使用三個 independent seed，confirmatory result 最好使用五個；保留 per-episode row 並回報 interval，不把 environment-time sample 當成獨立資料。
- 使用 matched commanded 與 achieved speed、tracking、success、fall、recovery、temperature、peak current 及 measured electrical cost of transport 驗證 energy 變更。機構允許時，以 randomized paired hardware trial 比較 passive-spring condition。
- 只有在同時提供 protocol、configuration、calibration evidence、checkpoint、per-episode data、failure 與 representative video 時才發布結果。宣稱 novelty 前完成專門的 literature 與 prior-art review。
- 只有在共同 task、command envelope、metric、seed 與 hardware condition 定義完成後，才能比較 RL 與 MPC。

Evidence gate 與解讀記錄於 [2026-08-13 研究就緒度稽核](../research/2026-08-13-research-readiness-audit.zh-TW.md)。

<a id="core"></a>
## Core 可維護性

- 決定是否核准並執行 proposed core-first、simulation-first soft reboot。
- 若核准，任何 golden baseline 前先驗證 legacy gravity、frame、mass、contact、timing 與 determinism。
- 把 stable contract 與 pure Torch 行為從 Isaac adapter 分離；抽離期間不改變外部 task 或 artifact。

<a id="operations"></a>
## 操作與部署

- 面板更廣泛暴露到 LAN 前，解決或明確接受 authentication 邊界。
- 加入 deployed base-linear-velocity estimator，或採用有證據的 training alternative。
- 完成 hardware bring-up 證據，並維持 ROS contract parity 為 60 Hz。
- 依 active design 與 plan 實作並驗證已核准的 Windows remote launcher。

<a id="reward-agent"></a>
## Reward Agent

- 保留有界 candidate planning、dry-run inspection、explicit legacy launch、trial persistence 與 metric-based report，作為 armed Autopilot campaign 之外的 manual workflow。
- Legacy session import 必須維持不可 arm 並保留 source JSON；不可把 legacy score 重新解讀成 deterministic campaign evidence。

<a id="autopilot-rollout"></a>
## Autopilot rollout

- 在 target training host 通過 fake-advisor loop、restart recovery、idempotency 與 single-GPU host serialization 前，保持 `REDRHEX_AUTOPILOT_ENABLED` 預設關閉。
- 在 repository 之外 provision ChatGPT Scheduled 與 OpenAI Secure MCP Tunnel 並保存 runtime credential；任何 unattended launch 前，先驗證 shadow proposal 與 tunnel-loss waiting。
- 從四個 trial 的 ForwardFast pilot、restart recovery，逐步進入 opt-in 24-trial qualification campaign。上述 gate 通過後才能啟用 Direct stage 2–5。
- Sensor V2、remote-child campaign control、automatic source application、policy export/deployment、hardware promotion 與 cross-campaign learning 都不納入 V1。變更 `3.7.0-remote-parity` protocol 前必須另做 compatibility design。

<a id="documentation"></a>
## 文件

- Operator/reference/roadmap 每 90 天 review，developer architecture 每 180 天 review。
- 只有 evidence 改變 baseline、recommendation、decision 或 result 時才發布 experiment summary。
- Documentation-system v1 checkpoint tag 完成後，才開始 Git history 與 branch reorganization。
