---
id: training-panel-operator-guide
title: Training Panel 3.7.0 操作指南
lang: zh-TW
audience: operator
type: how-to
status: active
owner: panel
last_reviewed: 2026-08-14
---

<a id="start"></a>
## 啟動 Mother

```bash
python -m tools.training_panel --host 127.0.0.1 --port 8080
```

開啟 `http://127.0.0.1:8080`。從其他電腦使用時，優先建立 SSH tunnel。本機面板是未驗證身分的管理介面；只有可信任 LAN 才可綁定 `0.0.0.0`。

使用 VS Code Remote SSH 時，先在遠端 terminal 啟動面板，再從 Ports view 轉送遠端 `8080` 與 `6006`，並開啟其本機轉送 URL。面板應維持綁定 `127.0.0.1`；透過 SSH tunnel 存取即可，不必把管理介面暴露到 LAN。

<a id="train"></a>
## 訓練與 queue

進入 Train 後，先選擇 **Training Mode**。表單只顯示該 mode 會用到的 control：**Standard PPO — No Distillation** 只有一個 Iterations field，不會顯示 F1/F2/F3；完整 Sensor V2 pipeline 顯示三個 stage iteration fields；advanced single-stage route 則會依該 stage 重新標示唯一的 Iterations field。新 policy run 暫時預設使用 Native。Explicit 會顯示但無法用於 policy training，因為目前未校準的 `200 N*m/rad` model 在 120 Hz 會發生數值不穩定；請用[扭轉彈簧校準與 characterization workflow](../../../docs/operators/calibration/torsion-spring-calibration.zh-TW.md#backend)進行調查。此 quarantine 不代表已選擇 Native 用於 production。

Isaac/GPU action（training、playback、video 與 ONNX export）會序列化。已有 GPU action 時，新 training request 變成 queued；可從 History 取消。完成的 Isaac 工作之間有 settle window。

Panel 啟動的 standard training 使用 run-scoped override snapshot，並傳入 `--panel_overrides` 及 optional `--physics-profile`。Built-in reward、terrain 與 physics preset 為 read-only；修改前先 duplicate。Sensor V2 route 會隱藏不使用的 task、reward 與 terrain controls，因為它們採用下方說明的固定 versioned contract；所選 physics profile 仍可使用。**Use Smoke Defaults** 與 **Use Debug Defaults** 都會更新目前生效的 iteration controls：standard 或 single-stage training 更新一個 field，完整 pipeline 則更新全部三個 fields。

只訓練直走步態時，Panel 現在預設使用 `Template-Redrhex-ForwardFast-Direct-v0`，並預設啟用 `Straight Forward Focus`；啟動前請確認兩者。先使用 4 個 environment 與 1 iteration 做 stack smoke test；通過後，再使用預定的 environment 數量，並在範圍受限的 ForwardFast profile 中最多執行 1,500 iterations。

<a id="sensor-v2-distillation"></a>
### 訓練 Sensor V2 Teacher 與 Student

在 **Training Mode** 選擇 **Sensor V2 — Full F1 → F2 → F3 Pipeline**。Production 預設值為 64 個 environments、F1 Teacher 1,500 updates、F2 distillation 800 updates，以及 F3 student PPO 1,500 updates。設定 run name；在遠端機器保持 Headless 啟用，然後只需按一次 **Start Training**。完整 pipeline 不使用 task、single-stage Iterations、checkpoint、reward 與 terrain controls，因此表單會隱藏它們。

Pipeline 會依序執行 F1、F2 與 F3。它使用 `--teacher_checkpoint` 把完成的 `teacher_v2` checkpoint 傳給 F2，再使用 `--student_checkpoint` 把完成的 `student_distilled_v2` checkpoint 傳給 F3。任一 stage 失敗就停止 pipeline。Sensor V2 使用固定的 forward reward contract，不套用 Panel reward 或 terrain override files。三個 stage 都會收到相同的 spring backend；若有選擇，也會收到相同的 run-scoped physics profile。

若只執行單一 stage，使用 advanced 選項 **F1 — Teacher only**、**F2 — Distillation only** 或 **F3 — Student PPO only**。表單會依所選 stage 標示唯一的 Iterations field。F2 需要 Teacher checkpoint，F3 需要 distilled checkpoint：先在 History 選擇來源 run，按 **Resume to Train** 填入 required checkpoint field，再選擇目標 mode。F1 可從頭開始，也可 resume 相容的 Teacher checkpoint。Strict loader 會拒絕錯誤的 checkpoint kind。

完整 pipeline 完成後，History 會連到最後的 F3 PPO directory。從 **Process Console** 可查看目前 stage 與確切 checkpoint handoff；從 **TensorBoard** 可查看最後的 F3 metrics。F1 與 F2 stage directories 仍分別保留在 `logs/rsl_rl/redrhex_forward_v2_teacher` 與 `logs/rsl_rl/redrhex_forward_v2_distillation`。一次完成的 run 只代表一個 seed 的訓練，不代表已取得 three-seed、recorded-sensor 或 hardware promotion evidence。

<a id="physics-presets"></a>
### 調整 physical quantity

開啟 **Physics**，選擇 **Baseline** 即可繼承所有 repository 與 USD physical default。要建立 candidate，先 duplicate Baseline 或建立 preset。使用 Search 依 body、joint、limit、unit 或 description 搜尋；使用 **Show changed only** 稽核 sparse candidate。空白 field 代表 inherit。**Reset** 只清除一個 override，不會把 default 寫入 preset。

Editor 公開 113 個可獨立調整的 simulation quantity：rigid-body damping；mass scale、added root mass 與 root center-of-mass offset；contact friction 與 restitution；aggregate command delay；每個 actuator group 的 stiffness、damping、effort limit、velocity limit、armature 與 friction；全部 18 個 joint 的 static、dynamic 與 viscous friction；六個 passive spring；以及六組 ABAD target scale 與 offset。Ground static/dynamic friction 是 coupled contract。Invalid value 會在啟動前被拒絕。

扭轉彈簧 damping 預設為零。`damper_0` 到 `damper_5` 保留為穩定 profile aliases。扭轉彈簧 actuator 的大型 effort/velocity limits 是 nonbinding，不是 spring torque clipping 或人工 velocity brake。在沒有 reviewed physical evidence 前，不得使用任意 armature 或統一 mass scaling 作為不穩定修正。

保存 preset 後可在之後重用。即使尚未保存，選定 draft 仍會用於下一個 run，Train page 會顯示其名稱。Training 會把確切 `CalibrationProfileV1` snapshot 到 run。Play、Record Video 與 Export ONNX 會重用該 snapshot，不會改用目前選定的 preset。這些設定只改變 simulation behavior；hardware calibration、E-stop 準備與 motor-enable authorization 仍是獨立 gate。

<a id="history"></a>
## 使用 History

History 結合 panel request 與探索到的 RSL-RL run。選擇 run 後可檢查 configuration、checkpoint、spring backend/calibration status、reward/terrain difference、saved physics metadata、note、folder、event state、video、export 與 readiness evidence。Play、recording、export 與 deployment checks 重用已記錄的 backend，並拒絕不相容的 spring metadata。Explicit checkpoint 仍可用於檢視與保留 provenance 的 playback，但 Resume to Train 會被阻擋；backend characterization 使用 deterministic workflow，而不是 learned checkpoint。可用 action 包括 TensorBoard、Play、Record Video、Export ONNX、Resume to Train、Compare、Compact Run 與 Process Console。

執行中的 card 會顯示從 process log 解析的 iteration progress、throughput 與 ETA。Run detail 會從本機 TensorBoard scalar 繪製 mean reward 與 episode-length curve。Run record 也會保存啟動時使用的 Git commit、branch 與 dirty state。Seed 留白時，面板會自行選擇並記錄 seed，讓面板啟動的 run 可重現。
Play 與 Record Video 會重用所選 run 保存的 task，並以前進命令開始，等同按下 `W`。Export ONNX 會重用保存的 task，但不加入移動命令。兩項檢查都以 Process Console 顯示的命令為準。

Windows 與 macOS launcher 會用明確的 desktop-remote marker 開啟 panel。在此 mode，**TensorBoard** 會於轉送的 `6006` port 啟動或重用單一 all-runs server，並強制使用 headless training。**Play**、**Open MuJoCo Viewer** 與 host file-manager button 會停用，因為這些視窗只會開在 training PC，不會出現在遠端 browser。請改用錄製的 Isaac video、MuJoCo MP4、browser console output 與 copy-path control。

Compaction 會保留最高編號的 top-level `model_*.pt`，並保留 event、parameter、video、export、note 與 deployment report。刪除需要輸入確切 run ID；相關 process 執行中時會拒絕。批次刪除需要輸入 `DELETE` 確認。只有在確認後，run 才會顯示為刪除中。

Search、status、sort 與 folder 選擇會在重新載入後保留；當 filter 隱藏了 run 時，run 數量顯示為 `N of M`，並出現 **Clear filters**。Search 會比對 run 名稱、id、task、status、folder、note 內容與 preset id。依 status 排序時，running、stopping 與 queued 會排在已結束的 run 之前。按 `/` 聚焦 search，`j`/`k` 或方向鍵移動選擇，`Escape` 清除 search 或關閉 comparison。Shift-click checkbox 可選取範圍。把 run card 拖曳到側邊欄的 folder 即可移動；若拖曳的 run 屬於目前的選取範圍，會移動整組選取。拖曳進行中時，所有可接收的 folder 都會標示，已包含全部拖曳 run 的 folder 會變淡並拒絕放下，游標所在的 folder 則會在放下前顯示確切結果。

**Compare** 會在 run list 旁開啟獨立的 comparison panel，不影響 run details panel；對其他 run 按 **Compare** 可替換被比較的欄位。切換選擇時，未儲存的 notes 會依 run 保留並標示為未儲存草稿，離開頁面前也會提示。

<a id="deploy"></a>
## 檢查 deployment readiness

Deploy view 用於在 Jetson ROS2 bring-up 之前驗證已匯出的 policy。先選擇 run，讀取 artifact 列，再執行一項檢查。**Checkpoint**、**ONNX** 與 **Last report** 會以 status pill 呈現，沿用面板其餘部分相同的顏色語彙；按鈕上方的一句話會指出下一個唯一動作：run 有 checkpoint 但沒有 ONNX 時先匯出，已有 ONNX 但尚無 report 時先驗證，已有 report 時則檢視警告或失敗的 stage。

三項檢查的成本不同。**Validate Existing ONNX** 直接對磁碟上既有的檔案執行 readiness pipeline。**Export ONNX + Validate** 會先以最新 checkpoint 重新產生 ONNX，適用於再次訓練之後。**Run MuJoCo Smoke** 只執行 advisory 物理 rollout。無法執行的檢查會維持 disabled，並在滑鼠停留時說明自身原因，包含 Isaac/GPU lock 被其他動作占用的情況。Device、MuJoCo model 路徑、runtime provider 選擇與 ROS mock stage 集中在 **Advanced options**，預設為收合。

Readiness stage 的 pass、warn、fail 與 skipped 數量顯示在標題旁。完整 JSON report 與 deploy console 在需要前保持收合；有 process 正在輸出時，console 會自行展開。Readiness level 與各 stage 的意義定義於[部署就緒](deploy-readiness.zh-TW.md)。

<a id="convergence"></a>
## 監控 convergence

Convergence view 包含兩條各自獨立的規則。**Plateau rule** 判定 reward 曲線何時趨於穩定：可選擇 Loose、Default、Strict 或 Custom，並可在訓練結束後自動錄製影片。**Divergence guard** 監看 non-finite scalar value 與 reward 持續崩塌，並透過已設定的通知 channel 發送警示。Automatic stop 是 opt-in；先將 action 維持在 `notify`，用目前 task 的 reward scale 驗證 detector，再只在確實需要時選擇 `stop`。此 view 會以文字說明目前選用的是兩種行為中的哪一種。

關閉主開關時，其管轄的控制項會一併 disabled，而不是維持可編輯的外觀。**Effective Settings** 列出 mother 已儲存的內容，包含 window、threshold、最少 iteration 數、通知 cooldown 與 TensorBoard scalar tag。修改表單後會標示 **Unsaved** 並啟用 **Save Settings**。已儲存的值套用於下一次訓練，不影響進行中的 run。

<a id="navigation"></a>
## 導覽與診斷 UI

目前 view 與選定 run 會保存於 URL，因此 refresh 或分享連結都能保留 context。Top bar 會回報 backend freshness；若顯示 stale，操作人員應先停止送出新 action，直到理解連線狀態。初次載入時使用 skeleton，不會誤報沒有資料；Rewards、Terrain、Convergence、Activity 與 Control Center 的 action failure 會顯示在目前 view。

<a id="child"></a>
## 安全使用 Child

RedRHex To Go 在 desktop 使用 persistent sidebar，在手機與平板使用 Dashboard、Train、History、More。More 包含 Rewards、Terrain、Physics、Deploy、Detection、Activity 與 Connection。URL 與 local draft 會保留目前 view、選定 run/folder、filter、sort 與尚未完成的 training input。Dashboard 維持 remote landing page，顯示 acceptance、machine readiness、GPU/queue state、compatibility、active job 與 recent run。

Train 採用與 Mother 相同的 Standard/F1/F2/F3/full-pipeline route 與 Native spring contract，但 checkpoint 以 run 加 iteration 選擇。History 預設顯示 all runs，並包含 folder、keyboard navigation、comparison、progress、bounded curve、provenance、checkpoint evolution、bulk move 與 remote-safe action。Deploy 僅產生 evidence：ONNX validation、export-and-validate、MuJoCo smoke/MP4 與 optional ROS mock 都使用 repository-owned input。Detection 只顯示 Mother setting，無法編輯。

Viewer 僅能檢視。Operator 可編輯共享 metadata/preset/folder，並執行 non-destructive action。Admin 另可刪除；bulk deletion 需要輸入 `DELETE`，而且每個 run 都會回報各自結果。若 Connection 回報舊 schema 或 worker，保持 read-only，套用指定 migration、更新 Mother、restart worker，再 refresh。Child 絕不提供 terminal、raw log、host path、GUI viewer、worker control、convergence edit 或 robot actuation。

<a id="console"></a>
## 使用 Process Console

Launch Command 是面板要求的命令；Output 是捕捉到的 process stream。若有 tmux，工作在 detached session 執行，console 會提供 attach command。Stop Process 先送出 interrupt，只有 Isaac 未關閉才 escalation。

<a id="artifacts"></a>
## 匯出與錄影

Export ONNX 會從選定 training checkpoint 產生 `exported/policy.pt` 與 `exported/policy.onnx`。預設高品質 video preset 為 1920×1080、1,200 steps 與 30 FPS。Standard run 會使用已保存的 terrain override；Sensor V2 會使用固定 reward/terrain contract 與已保存 runner。兩種 route 都會重用 run 已保存的 physics profile。Video 也會傳入 `--initial_command forward`。

若要使用 one-touch Google Drive 匯出，請在 training PC 安裝 `rclone`，並透過 `rclone config` 建立名稱必須完全是 `redrhex-drive` 的 Google Drive remote。個人 My Drive 建議使用 `drive.file` scope。確認 `rclone listremotes` 包含 `redrhex-drive:`，且 `rclone lsd redrhex-drive:` 成功，再重新啟動 Mother，讓 `/api/system` 將 integration 回報為已設定。

在 History 顯示最新或 checkpoint 影片，然後按 **Export to Drive**。上傳會在背景繼續，不占用 Isaac/GPU lock，並寫入 `RedRHex Videos/<run-id>/<video-filename>`。相同且未變更的來源不會重複上傳。**Open in Drive** 使用回傳的 Drive file ID 開啟私人檔案，不會建立 anyone-with-link 權限。失敗或因 restart 中斷的匯出會保留錯誤並可重試。

<a id="next"></a>
## 下一步

- [遠端操作](remote-operation.zh-TW.md)
- [部署就緒檢查](deploy-readiness.zh-TW.md)
- [疑難排解](troubleshooting.zh-TW.md)
