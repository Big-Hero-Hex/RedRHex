---
id: child-panel-remote-parity
title: Child Panel 3.7 遠端同等升級
lang: zh-TW
audience: developer
type: design
status: approved
owner: panel
last_reviewed: 2026-08-14
---

<a id="problem"></a>
## 問題

RedRHex To Go 已具備 authentication、role、remote queue、共享 Reward 與 Terrain preset、folder 及 History，但 navigation、feedback、training route、Physics、deployment evidence、detection visibility 與 team activity trail 仍落後 Mother 3.6.4。Operator 需要在手機上使用相同且安全的 mental model，同時不能暴露 Mother 的 administrative surface。

<a id="experience"></a>
## 核准的體驗

To Go 維持 static、buildless、team-scoped 與 phone-first。Desktop 使用持續顯示的 Mother-style sidebar；手機與平板使用 Dashboard、Train、History、More，More 包含 Rewards、Terrain、Physics、Deploy、Detection、Activity 與 Connection。URL state 保留 view、folder、run、search、status 與 sort。共享 design token、focus state、notice、toast、freshness 與 responsive card，並在 390 px、768 px 及 desktop width 均不得水平 overflow。

Train 提供 Standard、F1、F2、F3 與完整 pipeline route，固定使用 Native spring，省略無關欄位，checkpoint 僅以 run ID 加 iteration 表示。Reward、Terrain 與 sparse Physics preset 為 team-shared，built-in 受保護。History 預設顯示 all runs，並管理 folder、filter、keyboard selection、drag/drop、bulk move、admin-only bulk deletion、comparison、progress、bounded curve、provenance、checkpoint evolution 與 remote-safe run action。

<a id="boundary"></a>
## 安全邊界

Viewer 僅能檢視。Operator 可編輯 metadata、preset 與 folder，並執行 non-destructive job。Admin 另外可刪除。Worker 解析 authoritative profile role；不得信任 browser 提供的 role 欄位。Terminal access、raw log、worker administration、任意 host path、GUI viewer、convergence 編輯與 physical robot actuation 仍僅限 Mother。

Remote Deploy 僅接受 repository-owned model 與列舉的 MuJoCo scenario。它可驗證既有 ONNX、export 並驗證、執行 MuJoCo smoke、錄製 MuJoCo MP4，並可選擇 ROS mock；不可開啟 viewer 或驅動 hardware。Detection setting 為 read-only。

<a id="protocol"></a>
## Protocol 與相容性

Contract version 為 `3.7.0-remote-parity`。Additive Supabase migration 新增 machine capability、Physics preset、request idempotency、bounded run projection、activity source identity、受限 metadata/cancellation RPC 與 authoritative job attribution。既有 3.4.10 row 仍有效。Schema 或 worker 較舊時，sign-in 與 inspection 仍可使用，但所有 mutation 會停用，並顯示精確的 migration 與 restart 指引。

<a id="rollout"></a>
## Rollout 與 rollback

先 pause acceptance、套用 additive migration、更新 Mother、restart worker、確認 capability row 與 heartbeat，再發布 Child asset 並恢復 acceptance。若順序不同，Child 保持 read-only。Rollback 時 pause acceptance，還原上一版 Child 與 worker asset；additive database object 保留，因為舊 client 會忽略它們。

<a id="acceptance"></a>
## 驗收邊界

本機完成條件包含 Node protocol test、Python worker/schema test、專用 Child Playwright coverage、完整 Mother UI 與 Training Panel suite、documentation validation 及 diff check。Staging 完成條件另包含 Supabase login、RLS spoof rejection、old-worker fallback、new-worker activation、queue/stop/media/Drive/Deploy flow、audit attribution 與 admin deletion。Hardware actuation 明確不在本設計範圍內。
