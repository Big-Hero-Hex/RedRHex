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

Autopilot 新增一個長期運行且由 panel 擁有的 `AutopilotService`、SQLite store、deterministic controller、exact-checkpoint evaluator process 與本機 Goals workspace。HTTP handler 會呼叫此 service；read 不會 reconcile 或 mutate state。Service 刻意不包含 model client 或 model API key。獨立的 `redrhex-autopilot` plugin 會把 narrow MCP tool set 轉成具 revision 的 panel request。

<a id="process-contract"></a>
## Process 與 artifact contract

`TrainingParams` 建立既有 `train.py` 介面，並傳入 `--panel_overrides`。Process registry 序列化 Isaac/GPU 工作、記錄 command/log、reconcile RSL-RL artifact，並把 panel request ID 與探索到的 run directory 關聯。`logs/training_panel/gpu_process.lock` 上的 nonblocking `fcntl` lease 會跨 registry instance/process 序列化 GPU launch；child 會繼承 descriptor，因此即使 Mother restart，kernel 仍會在 process exit 後釋放 slot。History write 由 `RLock` 與 atomic replacement 保護。

Campaign training 新增 typed stage/evaluation selection、immutable reward/terrain snapshot path 與 hash、exact source-checkpoint identity、strict policy-only loading，以及 campaign/trial ownership。每個 control 與 candidate 都從相同 frozen checkpoint 加 optimizer reset 開始，或採 fresh initialization；candidate checkpoint 絕不成為下一個 candidate 的 policy source。Command sweep 是 first-class serialized process，會依 immutable command profile 評估確切 trained output checkpoint。

<a id="autopilot-contract"></a>
## Autopilot campaign contract

`autopilot.py` 擁有 V1 schema、allowlist、command compilation、hard gate、ranking、state 與 transition table。`autopilot_store.py` 以 SQLite WAL mode 保存 revisioned campaign snapshot、強制 one-host-slot invariant，並 append immutable event、trial、decision、evaluation、artifact 與 idempotency result。`autopilot_service.py` 會依 resolved repository state 編譯 human draft、擁有 worker/recovery tick，且是唯一可以推進 campaign 或宣告 `simulation_goal_met` 的程式。

Controller 在 24-trial 上限內配置一個 seed-42 control、最多十九個 adaptive seed-42 screen，以及四個保留的 seed-43/44 control/winner confirmation。Hard identity/safety gate 先於 deterministic ranking。Training reward 與 TensorBoard curve 仍只供診斷。Campaign success 要求所有 paired replica 都有 valid evidence、三個 candidate 中至少兩個通過、candidate median tracking 優於 control，且 energy 低於 cap。

SQLite 位於 `logs/training_panel/autopilot.sqlite3`；content-addressed evidence 位於 `logs/training_panel/autopilot_artifacts/`。既有 Reward Agent JSON 會保留，且只匯入為不可 arm 的 provenance。Schema、route、mutation header、MCP tool 與 patch-handoff contract 請見 [Autopilot API 參考](autopilot-api.zh-TW.md)。

<a id="google-drive-contract"></a>
## Google Drive 匯出 contract

`google_drive.py` 負責 host-side rclone readiness check、Mother-wide destination setting、account-reconnect lifecycle 與背景 video export。Remote 名稱維持固定為 `redrhex-drive:`。`POST /api/google-drive/settings` 接受經驗證的相對 My Drive path，或 HTTPS `drive.google.com/drive/folders/<id>` URL。Path mode 以 argument-vector `rclone mkdir` 驗證或建立私人 folder；link mode 會解析 folder ID 與 optional legacy resource key，再以 `rclone lsjson --stat` 搭配 `--drive-root-folder-id`，必要時加上 `--drive-resource-key`，驗證 directory access。選定的 destination 以 mode `600` 保存於 `logs/training_panel/google_drive_settings.json`；OAuth output 與 resource key 不會進入 API state、activity 或 run history。

`POST /api/google-drive/reconnect` 會合併 concurrent request，並以 bounded background process 執行 `rclone config reconnect redrhex-drive:`，在 training PC 開啟授權。Folder 變更或 account reconnect 成功時，會遞增本機 destination revision。Export command 會套用 link validation 使用的相同 root-folder option，因此有效目的地為 `<configured-path>/<sanitized-run-id>/<sanitized-video-name>` 或 `<linked-folder>/<sanitized-run-id>/<sanitized-video-name>`。

Server 先解析最新或指定 checkpoint 的影片；若檔案不存在、來源不是 MP4、run directory 位於 repository RSL-RL root 之外，或影片位於所選 run directory 之外，都會在 rclone 啟動前拒絕。這會接受 `redrhex_wheg`、`redrhex_forward_fast` 與 Sensor V2 等 task-specific root，同時不放寬 per-run containment。

每個 run 會保存 `google_drive_video_exports`，以 run-relative video path 為 key。Entry 記錄來源 size 與 nanosecond mtime、checkpoint iteration、lifecycle state、不含 secret 的 destination mode/display/URL 與 revision、Drive file ID、private view URL、timestamp，以及長度受限且已遮蔽敏感資訊的錯誤。只有 source fingerprint 與 destination identity 都相符時，才會重用 completed entry。同一來源的 concurrent click 會合併；變更或失敗的來源會開始新嘗試；active upload 執行時會拒絕 settings 變更；startup 會把 stale uploading entry 轉成 interrupted。Export 不取得 GPU lock。Rclone `copyto` 接收 argument vector，不使用 shell command；`lsjson --stat` 提供 Drive ID。Exporter 絕不呼叫 `rclone link`，也不改變分享權限。

<a id="physics-profile-contract"></a>
## Physics profile contract

`physics.py` 擁有 browser-facing schema 與 local sparse preset store。其 113 個 field 是目前 Isaac integration 會使用、可獨立調整的 `CalibrationProfileV1` value。API 與 command payload 只接受 schema key 及 finite bounded number。Cross-field validation 由 `CalibrationProfileV1` 負責；materialize candidate 時會補齊 coupled ground friction 與 passive-spring requirement。

六個扭轉彈簧保留穩定的 `damper_0` 到 `damper_5` profile aliases。其未校準 damping 預設為零。相容性用 uniform spring stiffness/damping fields 會對映到 backend-aware 有效彈簧參數；大型 effort/velocity 數值讓 actuator-path limits 維持 nonbinding，不會 clip 或 brake spring law。

Process registry 把每個 non-empty candidate 寫入 `logs/training_panel/process_overrides/<process>_physics.json`，並透過 `--physics-profile` 傳遞。`train.py` 透過 sim2real integration 套用 profile，並把確切 applied contract snapshot 到 run 的 `params/`。History 保存 preset identity、sparse value 與 candidate path。Evaluation 與 export 優先使用 immutable run snapshot；只有 snapshot 不存在時才重建。Empty Baseline 不傳 profile，並移除 stale process candidate。

<a id="physics-robot-preview"></a>
## Physics 機器人預覽

`robot_geometry.py` 解析 deploy pipeline 的 canonical URDF，並於 `GET /api/physics/robot-geometry` 提供 layout。Layout 以 canonical leg index 為索引，逐腿記錄 body mount point，以及 ABAD、main-drive、扭轉彈簧三個 joint 的 origin、axis、canonical id、URDF name 與靜止角度。Joint 順序取自共用 contract，不重新推導；扭轉彈簧靜止角度採用該可調 field 的 schema 預設值，因此未修改的 preset 會顯示真實 spawn pose。Layout 以 URDF 的大小與修改時間為鍵快取。URDF 缺失或無法解析時回傳標記 `source: "fallback"` 的六腿平面 layout，而不是讓頁面失敗。

Layout 另外回報 `label_audit`。由 URDF 推得的腿部位置在右側與 `_LEG_LABELS` 不一致：index 0、1、2 命名為右前、右中、右後，實際卻安裝在右中、右後、右前的位置。Canonical index、actuator 分組與 tripod 歸屬皆不受影響，因此這是命名缺陷而非訓練缺陷。面板因此依 URDF 位置繪製並明確標示此不一致，而非暗示兩者相符。

`robot_view.js` 負責繪製該 layout。它是面板唯一的 ES module；`index.html` 透過 import map 將裸 `three` specifier 對映到 `static/vendor/` 下已釘選版本的 three.js，讓面板維持沒有 bundler、也沒有網路請求。`app.js` 仍為 classic script，透過 `window.RedRHexRobotView` 驅動該 module。幾何為程序化生成，因此不需提供 mesh 資產。每個 joint family 依其實際機構繪製，而非使用可互換的標記：ABAD 是位於其 abduction 軸上、附端蓋的 hinge pin，並顯示其擺動弧線；main drive 是位於側向軸上的馬達本體，附 rotor key 與掃掠弧線，表示其受 velocity control 的連續旋轉；扭轉彈簧則是裸金屬螺旋線圈，因為被動而不上色。Joint origin 與 axis 直接取自 URDF，其間的結構也一併繪製——通往各 hip 的支柱、把外張中腿撐出殼體的 outrigger，以及 main drive 與扭轉彈簧之間的連桿——讓各部件呈現為相連的機構。殼體尺寸由資料推導而非猜測：寬度收在最近 leg mount 內側，使 hip 硬體突出於側面之外；頂面則到達由 mount 與 ABAD offset 推得的 hip 線。URDF frame 本身即為 Y-up，與 three.js 一致，故原樣繪製；模擬器的 spawn rotation 用於把資產對映到 Isaac 的 Z-up world，此處刻意不再套用。以模擬器 frame 撰寫的 field（例如質心偏移）則逐值轉換。

此預覽顯示的是 preset 數值，而非機器人狀態；面板並無 joint telemetry。無法誠實以空間方式呈現的量（包含 linear damping、angular damping 與 aggregate command delay）以文字顯示於模型旁。選取腿部沿用既有的 Physics 搜尋篩選，不另外引入第二套篩選機制。此預覽具備 sticky 行為：field 清單捲動時它會固定於視窗頂端並收合為精簡列，同時移除 readout strip 並縮短標籤警告。Stuck 狀態透過 frame-throttled scroll listener，量測位於預覽靜止位置的零高度 anchor。不使用 IntersectionObserver 的原因是：瞬間捲動可能讓 sentinel 直接從畫面下方越過至上方而從未 intersect，ratio 始終未跨越 threshold，observer 因而不會觸發。不量測預覽自身 rect 的原因是：收合會改變其高度，使收合本身推翻了產生該狀態的條件而卡住。Anchor 不會被收合移動，因此保持正確；同時檢查 computed position，因為在螢幕高度不足時會停用固定。Physics row 設有對應的 `scroll-margin-top`，確保 focus 與 `scrollIntoView` 不會把 field 停在固定列底下；在螢幕高度不足時則停用固定。Render loop 採延遲掛載，並在 Physics view 或文件隱藏時停止。當 WebGL 不可用或 context 遺失時，viewer 會降級為俯視 SVG 示意圖，仍保留腿部選取與質心標記；`?robot3d=off` 可強制走此路徑以便測試。

<a id="remote-contract"></a>
## Remote contract

Remote role 與 job type 集中定義於 `remote_config.py`。Protocol `3.7.0-remote-parity` 新增 host-safe `machine_capabilities` row，包含 feature flag、route catalog、Physics field schema、列舉 deployment scenario、read-only detection setting 與 integration readiness。Heartbeat 與 sync summary 使用相同 protocol version。所選 machine 與 capability row 未同時符合時，Child 進入 inspection-only mode。

Worker claim 已授權 job，透過同一 process registry 執行，並同步 run、artifact、metadata、folder、tombstone、notification、bounded scalar series、progress、provenance、spring identity、divergence、deployment evidence、MuJoCo artifact、Drive state 與 allowlisted Mother activity projection。Browser checkpoint input 為 `{run_id, checkpoint_iteration}`；worker 解析並 containment-check path。Remote deployment 只接受 repository-owned model 與列舉 scenario，絕不接受 path 或 shell fragment。

`start_training`、video、ONNX 與 export-and-validate 由 Isaac GPU lock 序列化。Stop 維持優先。Drive export、existing-ONNX validation 與 MuJoCo-only 工作不取得該 lock。`client_request_id` 對 machine 與 actor 唯一，因此 retry 不會重複 enqueue。

<a id="spring-contract"></a>
## Spring-backend 契約

Run metadata 只接受 `explicit` 或 `native`。新 policy request 預設使用 Native；由於目前 120 Hz 的未校準 model 會發生數值不穩定，process boundary 會在寫入 history 或 spawn process 前拒絕 Explicit training。Explicit 仍可用於 sim-to-real characterization path。

Play、automatic video、export、deployment validation 與 remote synchronization 都重用儲存的 backend，spring metadata 無效時 fail closed。已 stamp 的 uncalibrated checkpoint 會拒絕 backend mismatch；沒有 metadata 的 legacy run 保留歷史 Explicit fallback，不會被靜默重新標記。Play 與 recording 會明確加入 `--initial_command forward`；export 不會加入移動命令。

<a id="security"></a>
## 安全邊界

Mother 沒有內建 authentication，且可啟動或刪除本機工作；預設邊界是 localhost 加 SSH。Child 只能暴露 publishable configuration。Service-role/machine token 與 Google Drive credential 留在 worker host。Role check 是 defense in depth，不能取代 Supabase policy 與 secret handling。

Autopilot 另由 `REDRHEX_AUTOPILOT_ENABLED` 控制，預設關閉。Panel 與 MCP endpoint 應維持 loopback。Narrow adapter 不能 arm、resume、擴大 constraint、存取 generic shell/file API、套用 patch、deploy 或操作 hardware。OpenAI Secure MCP Tunnel 與 ChatGPT Scheduled 是由 operator 管理的外部 service：repository 只提供 local adapter 與 recurring prompt，不會 provision 其 identity、API key、connection 或 schedule。

Viewer 為 read-only。Operator 可修改共享 metadata、preset 與 folder，並送出 non-destructive job。Admin 另可刪除。Supabase 透過 RPC 限制 run metadata，queued cancellation 僅限 requester 或 admin；trigger/RPC 與 worker event 會 audit authenticated actor。Worker 解析 authoritative profile role，並忽略 browser 提供的 `actor_role`。Terminal、raw log、worker control、任意 host file、convergence edit、GUI viewer 與 robot actuation 絕不投影到 Child。

<a id="version"></a>
## 版本 contract

Mother release `3.8.0-autopilot-preview` 新增本機、預設關閉的 campaign surface，但不改變 remote protocol。Remote Child asset、capability row、cache key、schema label 與 worker synchronization 仍為 `3.7.0-remote-parity`。Remote migration 為 additive；既有 3.4.10 row 與 read path 仍有效。舊 schema 或 worker 仍可 authentication 與 inspection，但 Child 會停用 mutation，直到兩個 remote compatibility signal 都相符。更新 release contract 所屬的每個 surface、保留 compatibility evidence，並新增雙語 release entry。

V3.5 新增 progress parsing、TensorBoard summary、divergence monitoring、Git provenance 與已記錄的 random seed。V3.6 新增 URL-backed navigation、action-local error reporting、first-load skeleton、keyboard-focus tooltip、run-card action menu 與 backend-freshness state。V3.6.1 quarantine Explicit policy training、把 Native 設為新 run 的暫定預設，並保留已 stamp checkpoint 的 backend identity。V3.6.2 讓 Train form 能依 route 顯示欄位、恢復可靠的 hidden-state rendering，並讓 browser request 省略無關 stage fields。V3.6.3 讓 run comparison 使用獨立 panel、在顯示為進行中之前先確認 History 的破壞性操作、保留 run-list filter，並為 run list 加入鍵盤操作。V3.6.4 透過 host-configured rclone remote 新增 private、checkpoint-aware Google Drive video export。V3.7 在 remote-safe Child 採用上述 Mother-grade contract，並新增前述 capability/security protocol。V3.8 新增 local Autopilot preview、deterministic campaign/evaluation authority 與 narrow external-advisor boundary。測試涵蓋 command construction、history、progress/convergence/provenance、queue/process behavior、remote role/sync、notification、contract parity、deployment、Drive export、campaign、MCP scope 與兩套 UI surface。

<a id="pages"></a>
## Pages artifact

Checked-in remote web source 維持 GitHub Pages 根目錄。Documentation-site staging 必須把雙語 docs 放在 `/docs/`，且不得改變 remote asset path 或行為。
