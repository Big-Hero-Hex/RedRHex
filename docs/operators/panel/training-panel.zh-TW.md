---
id: operator-training-panel
title: 操作 Training Panel
lang: zh-TW
audience: operator
type: how-to
status: active
owner: panel
last_reviewed: 2026-08-14
---

<a id="start"></a>
## 啟動本機面板

```bash
python -m tools.training_panel --host 127.0.0.1 --port 8080
```

開啟 `http://127.0.0.1:8080`。若要從另一台可信任電腦連線，請讓面板維持綁定 localhost，並使用 `ssh -L 8080:127.0.0.1:8080 user@host`。綁定 `0.0.0.0` 會把未驗證身分的管理介面暴露到 LAN，只能用於可信任網路。

<a id="launch-and-history"></a>
## 啟動與管理 run

在 Train 選擇 task、environment 數量、iteration、reward、terrain、physics preset、spring backend 與 resume 選項。新 policy run 暫時預設使用 `native`。`explicit` training 會顯示但無法選擇，因為目前未校準的 `200 N*m/rad` model 在 120 Hz 會發生數值不穩定；請透過[扭轉彈簧校準與 characterization workflow](../calibration/torsion-spring-calibration.zh-TW.md#backend)調查 backend。這是 operational quarantine，不代表已選擇 Native 用於 production。

面板會傳入 `--panel_overrides`；non-empty physics candidate 另傳入 `--physics-profile`，把每個工作綁定到已保存 input。History 會探索 checkpoint、event、export、video、note、folder 與部署報告。執行中的 card 會顯示 iteration progress、throughput 與 ETA；detail 包含 reward/episode-length curve 與啟動時的 Git provenance。現有 checkpoint 的 playback 與 export 會重用已記錄 backend，已 stamp 的 checkpoint 不得被靜默改用另一個 backend 評估。

Seed 留白時，面板會選擇並記錄一個值。先以目前 task 的 reward scale 驗證 divergence handling，再啟用 automatic stop；驗證前保持只通知。目前 view/run 會透過 URL 在 refresh 後保留，而 topbar freshness indicator 會顯示 backend 是否仍有回應。

同一時間只能執行一個 Isaac GPU 工作。Queue 會在工作之間保留 settle window。請從 Process Console 停止選定程序，並等候完全結束後再啟動另一個工作。

<a id="physics-presets"></a>
## 使用 physics preset

Physics 公開 113 個 schema-validated simulation quantity，涵蓋 mass 與 center of mass、contact、actuator limit 與 constant、所有 joint-friction term、passive spring、command delay 與 ABAD calibration。扭轉彈簧 damping 預設為零；為了 profile 相容性，穩定的 `damper_0` 到 `damper_5` aliases 繼續保留。彈簧 actuator 的大型 effort 與 velocity limits 是 nonbinding，不是 spring-law clipping 或 velocity brake。

Schema validation 不代表 physical value 已證實。Baseline 會繼承 repository 與 USD default。先 duplicate Baseline，只設定已量測或有意調整的 override，保存 preset，並在 Train 確認其名稱。Search 與 **Show changed only** 可讓大型 profile 維持可 review。

每個 non-empty candidate 都會成為 run-scoped `CalibrationProfileV1`。Play、video 與 ONNX export 會重用該 run 已保存的 profile。Physics preset 只影響 simulation experiment，不會授權 hardware operation。

<a id="google-drive-export"></a>
## 將錄製影片匯出到 Google Drive

在 training PC 安裝 `rclone`，然後執行 `rclone config`。建立名稱必須完全是 `redrhex-drive` 的 Google Drive remote；若使用個人 My Drive，建議選擇 `drive.file` scope，讓 remote 只存取由它建立的檔案。重新啟動 Mother 前，先在 host 驗證連線：

```bash
rclone listremotes
rclone lsd redrhex-drive:
```

在 History 選擇 run，再選擇最新錄影或較舊 checkpoint 的影片，然後按 **Export to Drive**。背景上傳目的地為 `RedRHex Videos/<run-id>/<video-filename>`。未變更且已完成的匯出會直接重用；失敗或中斷的工作可重試；**Open in Drive** 會開啟已連線帳號中的私人檔案，不會改變分享權限。Rclone credential 只留在 training PC，Panel API 不會回傳。

<a id="remote"></a>
## 使用遠端團隊模式

Remote worker 需要 Supabase URL、anonymous key、machine token、machine ID 與明確的 accept-jobs 設定。請將它們存於權限為 `600` 的 `~/.redrhex_remote.env`；絕不可把 service-role 或 machine token 放入 GitHub Pages 或已提交檔案。

從 Control Center 啟動及監督 worker。設定與擁有者尚未確認前，保持關閉遠端工作接受功能。角色分為 viewer、operator 與 admin；破壞性操作僅限 admin。

安裝 3.6.1 spring-safety update 後，先 pause acceptance，並同時 restart remote worker 與 Mother。Mother startup 不會替換已在執行的 worker。再次接受 job 前，確認 heartbeat 來自更新後的 worker；checked-in 3.4.10 Child 會省略 `spring_backend`，因此必須由更新後的 worker 補上安全的 Native default。

<a id="safety"></a>
## 操作邊界

面板只會啟動現有腳本，不會讓訓練結果自動適合真機。Export、deploy readiness、ROS preflight、實體急停準備與分階段馬達 enable 仍是獨立 gate。壓縮或刪除前，先保留選定 checkpoint 與證據路徑。

<a id="component-docs"></a>
## 元件文件

架構、遠端 contract、部署、疑難排解與版本資訊請見 [Training Panel 元件入口](../../../tools/training_panel/docs/index.zh-TW.md)。
