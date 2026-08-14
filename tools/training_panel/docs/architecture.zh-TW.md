---
id: training-panel-architecture
title: Training Panel 架構與 Contract
lang: zh-TW
audience: developer
type: explanation
status: active
owner: panel
last_reviewed: 2026-08-14
---

<a id="components"></a>
## 元件

本機 mother 是 Python `ThreadingHTTPServer`，包含 static asset 與 API，後端由 configuration、process registry、history、activity、preset、convergence、deployment 與 remote-worker management 組成。GPU action 以 subprocess 或 detached tmux session 執行。Child 是 static ES-module app。Supabase 保存遠端協作 state；只有 worker 會在 training PC 執行 job。

<a id="process-contract"></a>
## Process 與 artifact contract

`TrainingParams` 建立既有 `train.py` 介面，並傳入 `--panel_overrides`。Process registry 序列化 Isaac/GPU 工作、記錄 command/log、reconcile RSL-RL artifact，並把 panel request ID 與探索到的 run directory 關聯。History write 由 `RLock` 與 atomic replacement 保護。

<a id="physics-profile-contract"></a>
## Physics profile contract

`physics.py` 擁有 browser-facing schema 與 local sparse preset store。其 113 個 field 是目前 Isaac integration 會使用、可獨立調整的 `CalibrationProfileV1` value。API 與 command payload 只接受 schema key 及 finite bounded number。Cross-field validation 由 `CalibrationProfileV1` 負責；materialize candidate 時會補齊 coupled ground friction 與 passive-spring requirement。

六個扭轉彈簧保留穩定的 `damper_0` 到 `damper_5` profile aliases。其未校準 damping 預設為零。相容性用 uniform spring stiffness/damping fields 會對映到 backend-aware 有效彈簧參數；大型 effort/velocity 數值讓 actuator-path limits 維持 nonbinding，不會 clip 或 brake spring law。

Process registry 把每個 non-empty candidate 寫入 `logs/training_panel/process_overrides/<process>_physics.json`，並透過 `--physics-profile` 傳遞。`train.py` 透過 sim2real integration 套用 profile，並把確切 applied contract snapshot 到 run 的 `params/`。History 保存 preset identity、sparse value 與 candidate path。Evaluation 與 export 優先使用 immutable run snapshot；只有 snapshot 不存在時才重建。Empty Baseline 不傳 profile，並移除 stale process candidate。

<a id="remote-contract"></a>
## Remote contract

Remote role 與 job type 集中定義於 `remote_config.py`。Heartbeat 回報 panel version、machine ID、path、active job、queue depth、GPU lock、acceptance state、tunnel 與時間。Worker claim 已授權 job，透過同一 process registry 執行，並同步 run、artifact、metadata、folder、tombstone 與 notification。

<a id="spring-contract"></a>
## Spring-backend 契約

Run metadata 只接受 `explicit` 或 `native`。新 policy request 預設使用 Native；由於目前 120 Hz 的未校準 model 會發生數值不穩定，process boundary 會在寫入 history 或 spawn process 前拒絕 Explicit training。Explicit 仍可用於 sim-to-real characterization path。

Play、automatic video、export、deployment validation 與 remote synchronization 都重用儲存的 backend，spring metadata 無效時 fail closed。已 stamp 的 uncalibrated checkpoint 會拒絕 backend mismatch；沒有 metadata 的 legacy run 保留歷史 Explicit fallback，不會被靜默重新標記。Play 與 recording 會明確加入 `--initial_command forward`；export 不會加入移動命令。

<a id="security"></a>
## 安全邊界

Mother 沒有內建 authentication，且可啟動或刪除本機工作；預設邊界是 localhost 加 SSH。Child 只能暴露 publishable configuration。Service-role/machine token 留在 worker host。Role check 是 defense in depth，不能取代 Supabase policy 與 secret handling。

<a id="version"></a>
## 版本 contract

本機 Mother package 與 UI 的 release 是 `3.6.2-route-clarity`。獨立部署的 remote Child asset、Child release metadata、heartbeat schema 與 worker synchronization contract 仍為 `3.4.10-sync-health`。本機 UI release 不會暗中改變 remote protocol。更新 release contract 所屬的每個 surface、保留 compatibility evidence，並新增雙語 release entry。

V3.5 新增 progress parsing、TensorBoard summary、divergence monitoring、Git provenance 與已記錄的 random seed。V3.6 新增 URL-backed navigation、action-local error reporting、first-load skeleton、keyboard-focus tooltip、run-card action menu 與 backend-freshness state。V3.6.1 quarantine Explicit policy training、把 Native 設為新 run 的暫定預設，並保留已 stamp checkpoint 的 backend identity。V3.6.2 讓 Train form 能依 route 顯示欄位、恢復可靠的 hidden-state rendering，並讓 browser request 省略無關 stage fields。測試涵蓋 command construction、history、progress/convergence/provenance、queue/process behavior、remote role/sync、notification、contract parity、deployment 與 UI asset。

<a id="pages"></a>
## Pages artifact

Checked-in remote web source 維持 GitHub Pages 根目錄。Documentation-site staging 必須把雙語 docs 放在 `/docs/`，且不得改變 remote asset path 或行為。
