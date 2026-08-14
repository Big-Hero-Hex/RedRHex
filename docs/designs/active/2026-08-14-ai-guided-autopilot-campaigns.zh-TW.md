---
id: design-ai-guided-autopilot-campaigns
title: AI 引導的 Autopilot Campaign
lang: zh-TW
audience: developer
type: design
status: approved
owner: project
last_reviewed: 2026-08-14
---

<a id="problem"></a>
## 問題

Training Panel 已能啟動與觀察 PPO run，Reward Agent 已能保存有界候選值，command sweep 也能量測移動能力，但三者尚未形成可在重新啟動後安全復原的實驗迴圈。因此人員離開時 GPU 會閒置；反之，若讓 AI 直接控制 shell、reward 原始碼、成功判定或部署，也會違反專案安全界線。

<a id="goals-and-non-goals"></a>
## 目標與非目標

- 目標：讓人員核准一個數值化移動目標、有界 reward catalog、trial 上限、iteration 上限及有效 GPU 小時上限。
- 目標：以 SQLite WAL transaction 保存 campaign、decision、trial、evaluation、artifact、revision、idempotency 結果與復原事件。
- 目標：讓 ChatGPT 讀取精簡證據並提議一項 allowlist 內的 reward 變更；驗證、執行、排名、預算與成功判定仍由 deterministic panel 程式負責。
- 目標：支援 ForwardFast stage 1 與 Direct curriculum stage 1–5 的標準 PPO，並固定 checkpoint、seed、task、command、physics、spring、configuration 與 code identity。
- 非目標：Sensor V2、修改 source tree、policy export 或 deployment、hardware promotion、remote-child campaign control，以及 panel 端模型或 API key。

<a id="proposal-and-interfaces"></a>
## 提案與介面

耐久流程為 `draft → armed → control training/evaluation → awaiting advisor → candidate training/evaluation → confirmation → simulation goal met`，並具備明確的 paused、waiting、patch-handoff、budget、safety、stopped 與 failed 結果。每台 host 同時只可有一個 armed campaign。

`GoalSpecV1` 保存數值 `{vx, vy, wz}` 範圍、task 與 stage、evaluation profile、方向、安全 gate、精確 initialization checkpoint identity、seed、iteration cap 與 campaign budget。Policy-only baseline 是不可拆分的 `baseline_run_id + baseline_checkpoint_iteration + checkpoint_sha256` tuple；panel 只解析已記錄於 History 的 checkpoint identity，絕不掃描 run directory 來選取「latest」檔案。「walk」與「run」等顯示標籤分別編譯為既有 task/stage 範圍的下半部與上半部，但 arming 前一定顯示實際數值。`RewardCatalogEntryV1` 只允許與所選 task/stage 相容且非零的 shaping weight；符號固定，絕對界線不可超過 campaign 起始值的 80–120%。Advisor move 限於 campaign-start value 的有限 lattice：80%、90%、100%、110% 與 120%；human 縮窄範圍後會裁切並去除重複值，且排除目前及已嘗試的 point。termination、fall 與 health gate、physics、target、sigma、cap、command range 和 terrain 都不可變更。由於 deterministic evaluation 不會套用 baseline terrain override，V1 會拒絕任何含非預設 terrain override 的 baseline。

`AgentDecisionV1` 只能為一個 catalog key 提議一個尚未使用的 lattice point、暫停或請求僅供審查的 patch handoff。Decision context 包含每個 key 的 campaign-start/current value、hard bound、完整與剩餘 lattice value、baseline-to-leader delta、attempted move、recent decision/evaluation、evidence ID，以及剩餘的 trial/GPU/confirmation/poll budget。`EvaluationReportV1` 將 command 與 episode 證據綁定至精確 checkpoint、seed、evaluator profile、configuration 及 artifact hash。Command-sweep horizon identity 也會綁定 evaluator 記錄的 `num_envs`、`sweep_steps=600`、V1 control timestep `step_dt = 1/60` 秒，以及 `duration_s = sweep_steps × step_dt`；每個 command/environment 都必須提供完整的 600-sample horizon，而且每個 command success duration 必須等於 episode reconciliation 後的 success ratio 乘以 `duration_s`。除了 immutable command、episode 與 summary artifact，evaluated report 本身也會保存成第四個 SHA-256 addressed artifact，並在延後復原或最終 confirmation 前重新開啟驗證。無效、缺漏、截斷、遭竄改、非有限值、partial load、fallback-selected、artifact divergence 或 identity mismatch 的證據一律 fail closed。`CampaignSnapshotV1` 是精簡 read model。

Panel 提供有 schema version 的 `/api/autopilot` endpoint，用於 capability、campaign draft 與 lifecycle、decision context、有界 decision、event、comparison、artifact 及 patch export。每個 write 都需要 idempotency key 與 expected revision。獨立的 loopback MCP server 僅暴露五個 read tool、五個有界 decision/lifecycle write tool，以及一個 active-campaign no-op heartbeat；它不代理 panel administrative API。Streamable HTTP 用於 OpenAI Secure MCP Tunnel；本機 stdio 則保留給已安裝的開發工具。

每次 panel launch 都取得不可變且以 SHA-256 綁定的 reward、terrain、physics、command 與 checkpoint input。已安裝 dependency file 會以內容 hash 綁定；editable dependency 則綁定其 scoped Git tree、dirty diff 與 untracked content，且不暴露 absolute origin。Campaign completion 只記錄確切的 `model_{iteration_cap-1}.pt` path 與 digest，包括沒有 monitor 的 restart reconciliation；它不會從 History 僅供顯示的 latest-checkpoint scan 推導 output。Training 使用 fresh initialization，或使用 strict policy-only fork 並重設 optimizer 與 iteration。Candidate policy 不可串接：control 與每個 candidate 都從相同 frozen initialization 開始。Evaluator 是序列化的一級 GPU job，輸出 command 與 environment-episode 證據。Training reward 與 TensorBoard 資料只作診斷，不作成功證明。

24-trial funnel 保留一個 seed-42 control、最多十九個 seed-42 screen，以及四個 seed-43/44 control/winner confirmation。Screen 通過後立即開始 confirmation。Simulation success 必須三份 candidate report 全部有效、至少兩份通過所有 goal gate、median tracking 優於 paired control，且 energy 低於絕對 cap。

<a id="failure-modes"></a>
## 失敗模式

- Panel 或 host 重新啟動：依 durable state 及精確 process/artifact identity 對帳；不得重啟已完成或仍存活的 idempotent action。
- Campaign work 執行中發生 controller validation 或 internal failure：先保存確切且只屬於該 campaign 的 stop intent，再向 process 發出 signal；若尚無法確認停止，必須保留 active identity，並在後續 tick 或 restart 完成 GPU accounting 與 terminalization。Controller exception 不得默默留下 GPU work，也不得結束 panel controller thread。
- ChatGPT、schedule 或 tunnel 無法使用：完成並保存進行中的本機工作，再進入 `waiting_for_chatgpt`；不得使用隱藏模型 fallback。
- 證據無效、identity mismatch、fall/health failure 或非有限值：拒絕 trial，並依 deterministic state table 阻擋或暫停。
- Infrastructure launch failure：只可用相同 input 重試一次；configuration、divergence、safety 與 evidence failure 不得自動重試。
- 四個有效但無改善的 screen、無可用 move，或 confirmation 預算不足：進入 patch handoff，不得放寬 bounds。
- Patch proposal：只保存 hash-addressed unified-diff artifact，永不套用。任何被接受的 source edit 都因 code identity 改變而建立新的 linked campaign。

<a id="acceptance"></a>
## 驗收

- [ ] 在無人觀看下完成兩次 advisor-to-training-to-evaluation iteration。
- [ ] 重新啟動復原後不產生重複 GPU 工作或重複 campaign event。
- [ ] 所有 budget 與 safety guardrail 都能 deterministic 地自動停止或暫停。
- [ ] 成功只依賴有效 command-sweep 證據，不依賴 ChatGPT 意見或 training reward。
- [ ] Connector 無法修改 source、deployment、hardware、無關 process、budget 或 arming state。
- [ ] 通過 one-iteration Isaac smoke 與短版 ForwardFast shadow campaign；three-seed qualification campaign 維持 opt-in。

<a id="resolution"></a>
## 結果

核准以預設關閉的方式分階段推出。進行中的實作與 pilot gate 記錄於 [Autopilot campaign 實作計畫](../../plans/active/2026-08-14-ai-guided-autopilot-campaigns.zh-TW.md)，權限界線永久記錄於 [ADR-0003](../../decisions/adr-0003-ai-advisor-deterministic-authority.zh-TW.md)。
