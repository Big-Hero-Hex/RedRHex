---
id: training-panel-troubleshooting
title: 排除 Training Panel 問題
lang: zh-TW
audience: operator
type: troubleshooting
status: active
owner: panel
last_reviewed: 2026-08-14
---

<a id="startup"></a>
## 面板無法啟動

確認 Python 可 import `tools.training_panel`、8080 未被占用，且 `REDRHEX_ROOT` 指向此 repository。先綁定 `127.0.0.1` 啟動，並在改 host/port 前檢查 terminal traceback。

<a id="cuda"></a>
## CUDA preflight 失敗

面板在 GPU 工作前比較 loaded NVIDIA kernel 與 userspace driver version。Driver 更新後先 reboot；若版本仍不同，必須修復 NVIDIA 安裝再啟動 Isaac。降低 environment 數量無法修正 driver mismatch。

<a id="queue"></a>
## Run 持續 queued

檢查 Process Console 與 GPU lock。本機 training、playback、video 與 export 共用 lock。Remote training、video、ONNX 與 export-and-validate 共用該 lock；Stop 維持優先。Drive export、existing-ONNX validation 與 MuJoCo-only 工作不應等待 Isaac lock。只有 requester 或 admin 可取消 queued job。再次送出前先檢查 History；相同 `client_request_id` 會被拒絕為 duplicate。

<a id="history"></a>
## History 或 folder 不一致

Pause remote acceptance。確認 Mother history、worker heartbeat/protocol、選定 machine、capability row、schema 與 tombstone。不要同時修改 Mother 與 Child metadata。Machine-scoped query 與明確 clearing 可避免 stale name、note 或 folder 回復；3.7 metadata write 必須使用受限 RPC。

<a id="remote"></a>
## Worker offline 或 disabled

執行 `source ~/.redrhex_remote.env` 與 `python -m tools.training_panel.remote_worker --once`。驗證必要 variable、token scope、system time、machine ID、heartbeat freshness 與 `accept_jobs`。Environment file 權限保持 `600`。

<a id="compatibility"></a>
## Child 登入後為 read-only

所選 machine heartbeat 或 capability row 低於 `3.7.0-remote-parity`，或 additive schema 缺失時，read-only fallback 是預期行為。Pause acceptance。套用 `supabase/migrations/20260814_370_remote_parity.sql`、更新 Mother、restart worker，確認兩個 protocol field 都相符後再 refresh Child。不要用手動 insert job 繞過 banner。

若 migration 已套用，確認 worker token 能 upsert 自身的 `machine_capabilities` row、browser 選定正確 machine、realtime 已 subscribed，且 system clock 正確。在 mismatch 原因釐清前維持 pause acceptance。

<a id="artifacts"></a>
## Video、export 或 readiness 失敗

在 Mother 開啟選定 process output，確認 run 有真正 `model_*.pt`。在 Child 重新選擇 run 與 checkpoint iteration；browser-supplied path 會刻意被拒絕。Video/ONNX/export-and-validate 前先停止其他 Isaac 工作。Readiness 請檢查 per-stage result，以及回報的 panel Python、`onnx`、`onnxruntime`、`torch` 與可選 MuJoCo dependency。Remote scenario 必須從已發布 capability list 選擇。缺少可選 stage 產生 review；必要項失敗產生 blocked。
