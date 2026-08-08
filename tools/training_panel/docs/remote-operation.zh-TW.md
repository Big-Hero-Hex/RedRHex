---
id: training-panel-remote-operation
title: 操作 RedRHex To Go
lang: zh-TW
audience: operator
type: how-to
status: active
owner: panel
last_reviewed: 2026-08-07
---

<a id="architecture"></a>
## 遠端邊界

GitHub Pages 託管 static child UI。Supabase 儲存 team identity、role、job、run、artifact、event 與 machine heartbeat。Training PC 上的 worker poll 並執行已接受工作。Cloudflare Tunnel 可暴露 live panel/TensorBoard service；Supabase function 傳送 requester-scoped Discord notification。

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

<a id="roles"></a>
## Role 與 action

Viewer 檢視資料。Operator 可啟動 training 與非破壞性操作，包括 stop、video、export、TensorBoard、compaction 與 missed-notification delivery。Admin 另外可 delete。Mother panel 保留 terminal、本機檔案、worker administration 與完整 debugging 能力。

<a id="sync"></a>
## Sync 預期

Version 3.4.10 同步 machine-scoped run/job/artifact state、明確 metadata clearing、folder state/tombstone、active preset、queue/lock status 與 worker health。Release note 要求時重新套用 checked-in Supabase schema。若 mother 與 child 不一致，先 pause job 並診斷 sync，不要同時修改兩邊。
