---
id: reward-agent-architecture
title: Reward Agent 架構
lang: zh-TW
audience: developer
type: explanation
status: active
owner: reward-agent
last_reviewed: 2026-08-14
---

<a id="boundary"></a>
## 邊界

Reward Agent 是保留的 headless、manually driven experiment-orchestration foundation。它會建立 session、產生一次調整一個權重的有界 candidate、把 candidate 轉成 Training Panel parameter、預覽或排入 trial、評估外部提供的 metric，並建立 comparison report。它不會修改 reward source code、自動抓取 TensorBoard、為 Autopilot campaign 提供 advice，也不允許 language model 自行判定成功。

<a id="modules"></a>
## 模組

- `planner.py` 將 candidate 變更限制在宣告範圍，並記錄改變欄位與 hypothesis。
- `experiment_store.py` 在 `logs/reward_agent/` 保存 session、candidate、trial、evaluation 與 report。
- `agent.py` 把 candidate override 轉成 `TrainingParams`，並記錄 dry-run 或 queued trial。
- `launcher.py` 為明確 launch mode 建立現有 Training Panel process registry。
- `evaluator.py` 對完整 metric set 計分，並針對 supplied baseline 的 regression 加上 penalty。
- `reports.py` 排序完整 evaluation，並把最佳 candidate 連到 panel run。

<a id="safety"></a>
## 安全性質

實際執行需要 `queue-trials --launch`；`--dry-run` 獨立提供，且 operator workflow 要求先使用。Candidate generation 一次只改一個已宣告權重、保留其他 override，並套用 minimum/maximum bound。Report 會優先完整 evaluation，但不能證明 metric 在科學上有效。

<a id="integration"></a>
## 整合狀態

Standalone CLI 與 Training Panel registry adapter 會保留，供 explicit legacy workflow 使用。Training Panel 3.8 Autopilot preview 是另一套由 panel 擁有的 lifecycle，具有獨立 UI、SQLite state、exact-checkpoint evaluator、deterministic acceptance 與 narrow external-advisor connector。`logs/reward_agent/sessions.json` 的 Reward Agent runtime JSON 會匯入 Autopilot store 作為不可 arm 的 historical provenance；來源不會刪除，也不會被提升成 campaign。

不要把 `queue-trials --launch` 當成 armed campaign 的替代 controller。它不會提交 `AgentDecisionV1`、消耗 campaign budget 或建立 campaign success。新的 unattended work 應從 panel 的 Autopilot workspace 開始；legacy CLI 仍可用於該 lifecycle 之外的 manual dry-run planning 與 comparison。Source patch handoff 也屬於 Autopilot，而且只會儲存 review artifact；它不會授權任一元件修改 source tree。

Runtime session data 仍會被忽略，不是 canonical documentation 或 published experiment evidence。新的 authority 與 compatibility boundary 請見 [Training Panel Autopilot API 參考](../../training_panel/docs/autopilot-api.zh-TW.md)。

<a id="tests"></a>
## 測試

執行 `python -m unittest discover -s tools/reward_agent/tests -p 'test_*.py'`。測試涵蓋 store round trip、有界 candidate、metric completeness/regression、report ranking、dry-run trial persistence、透過 fake registry 的明確 launch，以及 launcher construction。Autopilot contract、legacy import、controller recovery 與 connector scope 由 Training Panel 及 plugin suite 覆蓋，不屬於此 legacy suite。
