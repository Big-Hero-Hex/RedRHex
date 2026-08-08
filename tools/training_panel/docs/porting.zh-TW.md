---
id: training-panel-porting
title: 擴充或移植 Training Panel
lang: zh-TW
audience: developer
type: how-to
status: active
owner: panel
last_reviewed: 2026-08-07
---

<a id="preserve"></a>
## 保留邊界

Mother 對本機 process/file operation 保持 authoritative；child 保持 static 與 team-scoped；worker 維持 execution bridge。重用 `TrainingParams`、`ProcessRegistry`、`HistoryStore`、role/job definition 與 artifact discovery，不建立第二套 launcher 或 store。

<a id="add-local"></a>
## 加入本機功能

先定義 data/command contract、加入 backend test、暴露一個 API，再加入 local UI。任何 Isaac action 都必須參與 GPU lock、queue、settle window、process console、stop handling、activity 與 history reconciliation。允許 open、compact 或 delete 前，驗證 path 仍在 repository-owned artifact root 內。

<a id="add-remote"></a>
## 加入遠端功能

一起加入 job type、role permission、必要的 Supabase schema/policy、worker execution、sync/artifact representation 與 child UI。預設 denied 或 paused。Static asset 絕不可放 machine credential。Request 必須 idempotent，並保留 requester attribution。

<a id="version"></a>
## 版本與 release

只有真實 release 才一起更新所有 3.4.10 version surface。加入 canonical 雙語 release document；不可事後虛構版本缺口。Contract 改變時，重新套用並測試 Supabase schema。

<a id="verify"></a>
## 驗證

執行相關 `tools/training_panel/tests`、remote web Node test 與 local UI test。檢查 mother workflow、child role boundary、queue/GPU lock、history convergence、artifact preservation、remote sync 與未改變的 Pages root。任何可見 workflow 變更都要更新 operator documentation。
