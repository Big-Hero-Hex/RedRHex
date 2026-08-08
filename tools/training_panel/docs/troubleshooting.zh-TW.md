---
id: training-panel-troubleshooting
title: 排除 Training Panel 問題
lang: zh-TW
audience: operator
type: troubleshooting
status: active
owner: panel
last_reviewed: 2026-08-07
---

<a id="startup"></a>
## 面板無法啟動

確認 Python 可 import `tools.training_panel`、8080 未被占用，且 `REDRHEX_ROOT` 指向此 repository。先綁定 `127.0.0.1` 啟動，並在改 host/port 前檢查 terminal traceback。

<a id="cuda"></a>
## CUDA preflight 失敗

面板在 GPU 工作前比較 loaded NVIDIA kernel 與 userspace driver version。Driver 更新後先 reboot；若版本仍不同，必須修復 NVIDIA 安裝再啟動 Isaac。降低 environment 數量無法修正 driver mismatch。

<a id="queue"></a>
## Run 持續 queued

檢查 Process Console 與 GPU lock。Training、playback、video 與 export 共用 lock。停止或等待 active Isaac job 與 settle window。只有確認沒有 orphan tmux 或 Isaac process 後，才 cancel 並重新送出。

<a id="history"></a>
## History 或 folder 不一致

Pause remote acceptance。確認 mother history、worker heartbeat/version、machine ID、schema 與 tombstone。不要同時修改 mother 與 child metadata。Version 3.4.10 需要 machine-scoped query 與明確 clearing，避免 stale name、note 或 folder 回復。

<a id="remote"></a>
## Worker offline 或 disabled

執行 `source ~/.redrhex_remote.env` 與 `python -m tools.training_panel.remote_worker --once`。驗證必要 variable、token scope、system time、machine ID、heartbeat freshness 與 `accept_jobs`。Environment file 權限保持 `600`。

<a id="artifacts"></a>
## Video、export 或 readiness 失敗

開啟選定 process output，確認 run 有真正 `model_*.pt`。Video/export 前停止其他 Isaac 工作。Readiness 請檢查回報的 panel Python，以及 `onnx`、`onnxruntime`、`torch` 與可選 MuJoCo dependency。缺少可選 stage 產生 review；必要項失敗產生 blocked。
