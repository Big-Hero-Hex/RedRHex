---
id: runtime-layout
title: 執行路徑與產物配置
lang: zh-TW
audience: shared
type: reference
status: active
owner: project
last_reviewed: 2026-08-14
---

<a id="configuration"></a>
## 執行設定

Training Panel 讀取 `REDRHEX_ROOT`、`ISAACLAB_ROOT`、`ISAACSIM_ROOT`、`CONDA_SH` 與 `REDRHEX_CONDA_ENV`。預設值只是特定機器的便利設定，不是可攜式專案需求。Script 應取得明確路徑，或從 repository 根目錄執行。Local campaign service 另由 `REDRHEX_AUTOPILOT_ENABLED` 啟用；其 optional panel bearer token 與 MCP/tunnel credential 都是 process environment secret，不可進入 repository file 或 artifact。

<a id="training-artifacts"></a>
## 訓練產物

```text
logs/rsl_rl/<experiment>/<timestamp>_<run-name>/
├── model_*.pt
├── events.out.tfevents.*
├── params/
├── exported/policy.pt
├── exported/policy.onnx
├── videos/play/
└── deploy/
```

只有 `model_*.pt` 是 training runner checkpoint。已匯出 model 是部署產物。

<a id="panel-state"></a>
## 面板狀態

`logs/training_panel/` 包含 process log、per-run override snapshot、note、history、activity、remote state、convergence configuration 與 cross-process `gpu_process.lock`。Autopilot 另加入 `autopilot.sqlite3`、`autopilot_artifacts/` 下的 SHA-256 addressed file，以及 `evaluations/` 下的 command/episode evidence。這些都是 runtime record，不是 tracked documentation。`tools/training_panel/` 下的 active override 檔案是暫時 IPC。手動 `train.py` 除非帶有 `--panel_overrides`，否則會忽略它們。

<a id="evidence"></a>
## 校準與實驗證據

Raw run log、完整 trace、video、campaign database/artifact 與本機 calibration artifact 維持 ignored。只提交小型 canonical configuration、manifest、已審查 summary 或明確核准 fixture。Reward Agent session 位於所選 repository root 的 `logs/reward_agent/`；Autopilot 會匯入成不可 arm 的 reference，且不會刪除來源。

<a id="repository-rule"></a>
## 儲存庫規則

Generated HTML、staged site file、cache、runtime log、raw experiment artifact、secret 與 worktree metadata 都不可追蹤。Git 儲存 durable source 與已審查 summary，不儲存操作狀態。
