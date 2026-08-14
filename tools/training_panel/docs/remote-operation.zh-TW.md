---
id: training-panel-remote-operation
title: 操作 RedRHex To Go
lang: zh-TW
audience: operator
type: how-to
status: active
owner: panel
last_reviewed: 2026-08-14
---

<a id="architecture"></a>
## 遠端邊界

GitHub Pages 託管 static Child UI。Supabase 儲存 team identity、role、job、run、artifact、event、capability、shared preset 與 machine heartbeat。Training PC 上的 worker poll 並執行已接受工作。Cloudflare Tunnel 可暴露 TensorBoard；Supabase function 傳送 requester-scoped Discord notification。

Child 為 team-scoped 且 remote-safe。它不會暴露 terminal、raw process log、worker administration、任意 host path、GUI viewer、convergence edit 或 physical robot deployment。需要這些本機管理能力時，透過可信任 SSH tunnel 使用 Mother。

<a id="secrets"></a>
## 設定 secret

在 training PC 建立 `~/.redrhex_remote.env`，包含 `REDRHEX_SUPABASE_URL`、`REDRHEX_SUPABASE_ANON_KEY`、`REDRHEX_SUPABASE_MACHINE_TOKEN`、`REDRHEX_MACHINE_ID`、`REDRHEX_REMOTE_ACCEPT_JOBS` 與可選 tunnel 設定。權限設為 `600`。Child 只能包含公開 Supabase URL 與 anonymous/publishable key；machine/service-role token 必須保持私密。

<a id="start-worker"></a>
## 啟動並驗證 worker

在 Control Center 啟動、停止、重新啟動、選擇 tmux 或 child-process mode、啟用 auto-start，並 accept/pause job。手動驗證：

```bash
source ~/.redrhex_remote.env
python -m tools.training_panel.remote_worker --once
```

連續執行前，確認正確 machine ID、`online: true`、新 heartbeat 與預期 `accept_jobs` 狀態。

3.7.0 rollout 時，先 pause acceptance 並套用 `supabase/migrations/20260814_370_remote_parity.sql`，再 restart Mother 與 worker。確認 heartbeat 及 `machine_capabilities.protocol_version` 都回報 `3.7.0-remote-parity`、選定正確 machine，且沒有 schema warning。只 restart Mother 仍會保留已在執行的舊 worker。

任一 version 較舊時，sign-in 與 inspection 仍可使用，但所有 mutation 都會停用。精確遵循 Child banner：套用 migration、更新 Mother、restart worker，再 refresh。Compatibility 未解決時，不要透過其他 client 樂觀送出 job。

<a id="roles"></a>
## Role 與 action

Viewer 可檢視所有 team-safe data。Operator 可編輯 run metadata、preset 與 folder，並執行所有 non-destructive remote action：route-aware training、direct stop、resume/tweak、TensorBoard、video、ONNX、private Drive export、compaction、deployment validation、export-and-validate、MuJoCo smoke/MP4、queued cancellation 與 notification delivery。Admin 另可刪除單一或選定 runs。Bulk deletion 需要輸入 `DELETE`；每個 worker delete 仍需 exact run ID，並逐一回報成功或失敗。

Database 會 stamp job identity 與 authoritative profile role。Worker 再次解析該 role，並忽略 browser 提供的 `actor_role`。Preset/folder trigger、metadata RPC 與 worker event 會建立 authenticated audit entry。

<a id="sync"></a>
## Sync 預期

Version 3.7.0 同步 machine-scoped run/job/artifact state、明確 metadata clearing、folder state/tombstone、Reward/Terrain/Physics preset、bounded progress/scalar curve、Git/spring provenance、divergence、deployment/MuJoCo evidence、completed private Drive link、redacted idempotent activity、queue/lock state、capability 與 worker health。它絕不同步 TensorBoard event file、raw log、credential 或 unrestricted path。

Training、video、ONNX 與 export-and-validate 共用 Isaac GPU lock。Stop 維持優先。Drive export、existing-ONNX validation 與 MuJoCo-only 工作不取得該 lock。Retry 前先檢查 action-local status 與 History；request ID 會防止 duplicate submission。

<a id="rollout"></a>
## Rollout 或 rollback

Pause acceptance、套用 additive migration、deploy/update Mother、restart worker、驗證 capability row、heartbeat、role 與 schema、發布 Child asset，之後才恢復 acceptance。若順序不同，讓 Child 保持 read-only。Rollback 時 pause acceptance 並還原先前 Child/worker asset。Additive table、column、policy、RPC 與 trigger 保留，因為舊 client 會忽略它們。
