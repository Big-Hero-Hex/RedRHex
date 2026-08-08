---
id: project-roadmap
title: 目前專案優先事項
lang: zh-TW
audience: shared
type: roadmap
status: active
owner: project
last_reviewed: 2026-08-07
---

<a id="validation"></a>
## 證據與移動驗證

- 對完整與 ForwardFast task 執行固定 protocol 的長訓練與 command-sweep 比較。
- 依技能使用 tracking、success、fall、cost-of-transport proxy 與 power-per-motion 準則驗證 energy 變更。
- 在提出更強 sim-to-real 宣稱前，建立 IMU frame、base velocity、contact behavior、mass/CoM、actuation 與 electrical energy 的 hardware ground truth。
- 只有在共同 task、command envelope、metric、seed 與 hardware condition 定義完成後，才能比較 RL 與 MPC。

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

- 保留有界 candidate planning、dry-run inspection、明確 launch、trial persistence 與 metric-based report。
- Proposal UI 與更深入 Training Panel integration 需另行 design；不可靜默修改 reward source，也不可讓 LLM 在沒有 metric 時宣告成功。

<a id="documentation"></a>
## 文件

- Operator/reference/roadmap 每 90 天 review，developer architecture 每 180 天 review。
- 只有 evidence 改變 baseline、recommendation、decision 或 result 時才發布 experiment summary。
- Documentation-system v1 checkpoint tag 完成後，才開始 Git history 與 branch reorganization。
