---
id: reward-agent-architecture
title: Reward Agent 架構
lang: zh-TW
audience: developer
type: explanation
status: active
owner: reward-agent
last_reviewed: 2026-08-07
---

<a id="boundary"></a>
## 邊界

Reward Agent 是 headless experiment-orchestration foundation。它會建立 session、產生一次調整一個權重的有界 candidate、把 candidate 轉成 Training Panel parameter、預覽或排入 trial、評估外部提供的 metric，並建立 comparison report。它不會修改 reward source code、自動抓取 TensorBoard，也不允許 language model 自行判定成功。

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

CLI 與 Training Panel registry adapter 已實作。Proposal UI、自動 evaluation ingestion、source-code proposal 與更深入 panel integration 都屬 roadmap，需另行 design。Runtime session data 維持 ignored，不是 canonical documentation 或 published experiment evidence。

<a id="tests"></a>
## 測試

執行 `python -m unittest discover -s tools/reward_agent/tests -p 'test_*.py'`。測試涵蓋 store round trip、有界 candidate、metric completeness/regression、report ranking、dry-run trial persistence、透過 fake registry 的明確 launch，以及 launcher construction。
