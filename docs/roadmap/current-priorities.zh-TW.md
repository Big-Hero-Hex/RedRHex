---
id: project-roadmap
title: 目前專案優先事項
lang: zh-TW
audience: shared
type: roadmap
status: active
owner: project
last_reviewed: 2026-08-15
---

<a id="integration"></a>
## 整合與發布紀律

- 以目前的 `origin/main` 為起點，將 Autopilot 與 Student V2 recovery snapshot 分別重建成可審查的變更；不可直接合併 recovery commit，也不可遺失它們的基底之後已進入 mainline 的 desktop remote、UI 與 code CI 修正。
- 在繼續 integration 前，先讓 repository root 回到最新且 clean 的 `main`，並將每個 recovery reconstruction 放在各自的 `.worktrees/` checkout。
- 維持 `main` 為唯一已發布 baseline。Training Panel 3.8 Autopilot 與 Student V2 follow-up 的重建 PR 在通過文件、service、browser 與 target-environment gate 前，都只視為 branch-local preview。
- 在相依關係允許時，將 27,000 行的 Autopilot snapshot 拆成可審查的 contract、controller、connector、UI 與文件 concern。保留精確的跨層 identity，並在每個整合步驟前執行完整 combined suite。
- 擴充 lightweight code CI，納入 Autopilot MCP adapter、Reward Agent 與 ROS contract suite。Mother 或 Child UI 行為有變更時，加入 browser job，或要求保存 local Playwright evidence。
- 以單獨的 hygiene 變更移除已追蹤的 `.vscode/browse.vc.db-shm` 與 `.vscode/browse.vc.db-wal` generated database file，之後持續忽略這些檔案。

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
- 完成剩餘的 Windows PowerShell 5.1+ 與受支援 macOS host smoke，涵蓋安裝、interactive SSH、fixed forward、browser capability state、timeout、tunnel shutdown 與 existing-tunnel reuse。Source-level launcher test 不能關閉這些 host gate。
- 在接受 production remote job 前，以真實環境執行 Child 3.7 Supabase staging smoke，涵蓋 role/RLS rejection、old/new worker compatibility、queue 與 cancellation、media 與 Drive flow、deployment job、activity attribution 與 admin deletion。

<a id="student-v2"></a>
## Student Distillation V2

- 以目前的 `main` 重建已保存的 Student V2 follow-up，保留其 simulator stop-point evidence，並與 Panel physics proposal 分開審查。
- 使用 canonical hash、golden fixture、zero-residual/action mapping、causal observation 與 simulator-to-ROS trace parity 凍結 V1 compatibility 與共用 `redrhex_policy_io` contract。
- 關閉 F1/F2/F3 evidence gate：三個 Teacher A seed、named two-input ONNX export 與 runtime parity、finite update/save/resume check、strict actor/normalizer transfer，以及三 seed forward screening。
- 將每個 promoted artifact 綁定 measured sensor calibration 與 held-out raw-event replay；完成 V2 ROS builder、warm-up、validity/dropout behavior 與 offline inference parity，過程中不得啟用 motor。
- 只有在同一 command protocol 下完成 Teacher A、legacy、distilled、PPO 與必要 ablation 的比較，三個 PPO seed 全部通過，且 provenance 與 teacher-gap evidence 完整時才能 promotion。

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

- 以 read-only 方式將 PM agent 連接到 GitHub `main`，並透過 deduplicated Update Log intake 將 repository change 投影到 PM Control Center。維持 Drive 中的 maintained Markdown 副本為 non-authoritative，並以 canonical GitHub link 取代會反覆更新的 snapshot link。
- Operator/reference/roadmap 每 90 天 review，developer architecture 每 180 天 review。
- 只有 evidence 改變 baseline、recommendation、decision 或 result 時才發布 experiment summary。
- 在重建工作經 review 進入 `main` 或被明確 archive 前，維持 branch-preservation manifest 與 local recovery bundle 的精確性；只有 reachability 與 clean-state check 通過後，才能移除 redundant branch 與 worktree。
- Active design 與 plan 的真實 host、staging、simulation 或 hardware gate 關閉後，才將其解析為 durable architecture、release、audit 或 roadmap record。
