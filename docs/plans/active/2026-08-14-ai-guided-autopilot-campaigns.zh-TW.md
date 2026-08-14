---
id: plan-ai-guided-autopilot-campaigns
title: AI 引導的 Autopilot Campaign 實作
lang: zh-TW
audience: developer
type: plan
status: active
owner: project
last_reviewed: 2026-08-14
---

<a id="objective"></a>
## 目標

交付已核准且預設關閉的 campaign controller、deterministic evaluator integration、有界 ChatGPT connector、panel workspace 及復原證據，使標準 PPO reward 實驗能在人員離開時安全運行。

<a id="context"></a>
## 背景

核准輸入為 [Autopilot campaign 設計](../../designs/active/2026-08-14-ai-guided-autopilot-campaigns.zh-TW.md)。Panel 仍是執行與成功判定的唯一 authority。V1 涵蓋 ForwardFast stage 1 與 Direct stage 1–5，維持既有 remote-child protocol 不變，並排除 Sensor V2、deployment、hardware、自動 source edit 與自動擴大 budget。

<a id="phased-checklist"></a>
## 分階段檢查清單

<a id="phase-foundation"></a>
### 安全與可重現性基礎

- [x] 在保存的 configuration、history diff 與 tweak reconstruction 中發現巢狀 `v2_reward_scales`。
- [x] 拒絕 boolean 與非有限 reward value，並提供精確 resolved-key 驗證。
- [x] 為標準 training 加入 typed stage、initialization、strict checkpoint、profile path、profile hash 與 campaign ownership 欄位。
- [x] 以每個 run 的不可變 SHA-256 snapshot 取代 panel launch 共用的 global reward 與 terrain mutation。
- [x] 將 policy-only initialization 綁定到已記錄的 run、確切 checkpoint iteration 與 SHA-256，不掃描 run directory 來尋找 latest checkpoint；V1 也會拒絕 non-default terrain override。
- [ ] 完成所有 campaign lifecycle state 的跨 process host lease、heartbeat 與 startup reconciliation 覆蓋。Lease 與 active training/evaluation recovery path 已有 coverage；all-state crash matrix 仍是 rollout gate。

<a id="phase-controller"></a>
### 耐久 deterministic controller

- [x] 交付 SQLite WAL campaign contract、event log、idempotency result、單一 armed campaign invariant、budget、trial、report 與 artifact record。
- [x] 透過 panel-owned process service 序列化 control/candidate training 與 exact-checkpoint command evaluation。
- [x] 以 fail closed 方式解析 command 與 episode CSV，先套用 hard gate 再排名，並保留 confirmation 容量。
- [x] 將 advisor proposal 限制於 campaign-start value 的有限 80/90/100/110/120 lattice，並在 decision context 暴露完整/剩餘 value、attempted move、leader delta、evidence ID 與 remaining budget。
- [x] 將 evaluated report 保存成第四個 immutable artifact，在延後復原與 confirmation 時重新驗證其 binding，並在 controller failure terminalization 前耐久地停止 campaign-owned work。
- [x] 以 dependency-light fake-process test 證明 control、screen、leader、early confirmation、three-seed success、non-improvement、budget、pause、stop 與 patch-handoff transition。

<a id="phase-advisor-and-ui"></a>
### Advisor 界線與 panel workspace

- [x] 提供有 version 的 `/api/autopilot` read route，以及具 idempotency/revision 檢查的 write route。
- [x] 交付 loopback MCP adapter，包含五個 read tool、五個有界 decision/lifecycle write tool、一個 no-op active-campaign heartbeat、safety annotation、stdio 開發 transport 與 Streamable HTTP tunnel transport。
- [x] 交付 advisor skill 與耐久的 15 分鐘 same-chat task prompt；記錄 heartbeat、declared model、prompt/skill version、proposal、validation 與 process identifier。External task creation 仍是 rollout step。
- [x] 交付 Autopilot workspace，用於 goal preview、explicit arming、lifecycle、budget、heartbeat、trial/evaluation comparison、campaign-owned stop control 與 patch export。

<a id="phase-rollout"></a>
### 分階段推出

- [ ] 執行 deterministic fake-advisor end-to-end loop 與 crash/restart matrix。
- [ ] 執行不會 launch 的 ChatGPT shadow proposal。
- [ ] 執行 four-trial ForwardFast pilot 與 restart/recovery pilot。
- [ ] 執行完整 24-trial ForwardFast qualification campaign。
- [ ] 只有前述 gate 通過後才啟用 Direct stage 2–5。

<a id="verification"></a>
## 驗證

執行 contract、validation、state transition、ranking、hard gate、budget、hash 及 corrupt input 的 unit/property test；fake-process lifecycle/restart/idempotency/concurrency test；MCP schema/security test；browser accessibility/mobile/recovery test；既有 Reward Agent、Training Panel 與 sim2real suite；以及 `python -m tools.documentation validate --all`。短版 shadow campaign 前先執行 one-iteration Isaac smoke。高成本 three-seed campaign 維持明確 opt-in。

<a id="completion-summary"></a>
## 完成摘要

預設關閉的 repository implementation、deterministic fake-process/restart coverage、有限 reward lattice、immutable evaluation-report evidence、durable controller-failure stop recovery、Streamable HTTP boundary 與 browser workspace 已存在。External ChatGPT Scheduled task 與 Secure MCP Tunnel 尚未 provision，而且 shadow、real Isaac、four-trial、restart-host、24-trial 與 Direct rollout gate 尚未完成，因此本計畫維持 active。本計畫的任何結果都不得稱為 hardware-ready。
