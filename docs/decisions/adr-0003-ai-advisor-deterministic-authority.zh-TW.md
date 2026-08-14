---
id: adr-0003-ai-advisor-deterministic-authority
title: "ADR-0003：AI Advisor 與 Deterministic Authority 界線"
lang: zh-TW
audience: developer
type: decision
status: accepted
owner: project
last_reviewed: 2026-08-14
---

<a id="context"></a>
## 背景

反覆 reward 實驗可受益於能根據近期證據形成 hypothesis 的 AI，但機器人 training 成本高、具狀態且涉及安全。自然語言判斷不是可重現的 acceptance test；若讓 unattended connector 存取 administrative panel、shell、source tree、deployment 或 hardware，就會把 advisor 的不確定性轉成未受控 actuation。

<a id="decision"></a>
## 決策

ChatGPT 只能擔任 advisor。它可以讀取精簡的 structured campaign evidence，並提交恰好一個 allowlist 內的 reward-weight proposal、請求暫停、請求 stop-after-current，或保存僅供審查的 patch proposal。它不能 arm 或 resume campaign、放寬 bounds 或 budgets、變更 immutable goal/physics/safety field、選擇任意 launch argument、宣告成功、修改 source、export/deploy policy、操作 hardware，或停止無關工作。

Deterministic panel 程式對 schema 與 identity validation、idempotency、revision ordering、GPU serialization、trial allocation、hard safety gate、ranking、confirmation、budget、state transition 與 `simulation_goal_met` 宣告擁有最終 authority。缺漏、非有限值、mismatch、partial-load、fallback-selected 或 corrupt evidence 一律 fail closed。Training reward 與模型評論只作診斷，不是成功證據。

Connector 是具有分離 read/write tool 的狹窄 loopback MCP service。ChatGPT Scheduled 可以回到同一個 chat，但其缺席或故障只會讓 campaign 在本機進行中工作完成後等待；不得使用 panel-side model fallback。

<a id="alternatives"></a>
## 替代方案

- 讓 AI 直接操作完整 panel 或 shell：拒絕，因為 tool scope 會包含無關 process、file、deployment 與不安全 parameter change。
- 讓 AI 排名 evidence 並宣告成功：拒絕，因為結果不 deterministic、不可重現，也無法獨立 audit。
- ChatGPT 缺席時啟動隱藏 panel-side model：拒絕，因為會引入第二個 authority、secret storage，以及使用者無法在 scheduled chat 觀察的行為。
- 所有 iteration 都保持 manual：保留作為 fallback，但不作唯一 workflow，因為會使已安全核准的實驗容量閒置。

<a id="consequences"></a>
## 後果

Campaign 只能在 human-armed goal、reward catalog 與 budget 範圍內自動前進。Panel 必須維護耐久且具 revision 的 state，以及足夠的 structured evidence，讓 advisor 不需 raw log 或 secret 即可行動。ChatGPT 品質可影響下一個被測試的有效 hypothesis，但不能削弱 constraint，也不能把失敗實驗變成成功。Hardware readiness 與 deployment 仍是分離且由人員治理的流程。

<a id="supersession"></a>
## 取代關係

未來任何授予模型更廣 mutation 或 success authority 的設計，都必須明確 supersede 本 ADR，並提供新的 authentication、safety、evaluation 與 human-approval 論證。
