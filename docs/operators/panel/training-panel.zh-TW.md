---
id: operator-training-panel
title: 操作 Training Panel
lang: zh-TW
audience: operator
type: how-to
status: active
owner: panel
last_reviewed: 2026-08-07
---

<a id="start"></a>
## 啟動本機面板

```bash
python -m tools.training_panel --host 127.0.0.1 --port 8080
```

開啟 `http://127.0.0.1:8080`。若要從另一台可信任電腦連線，請讓面板維持綁定 localhost，並使用 `ssh -L 8080:127.0.0.1:8080 user@host`。綁定 `0.0.0.0` 會把未驗證身分的管理介面暴露到 LAN，只能用於可信任網路。

<a id="launch-and-history"></a>
## 啟動與管理 run

在 Train 選擇 task、environment 數量、iteration、reward/terrain preset 與 resume 選項。面板會傳入 `--panel_overrides`，把每個工作綁定到其儲存的 override 檔案。History 會探索 checkpoint、event、export、video、note、folder 與部署報告。

同一時間只能執行一個 Isaac GPU 工作。Queue 會在工作之間保留 settle window。請從 Process Console 停止選定程序，並等候完全結束後再啟動另一個工作。

<a id="remote"></a>
## 使用遠端團隊模式

Remote worker 需要 Supabase URL、anonymous key、machine token、machine ID 與明確的 accept-jobs 設定。請將它們存於權限為 `600` 的 `~/.redrhex_remote.env`；絕不可把 service-role 或 machine token 放入 GitHub Pages 或已提交檔案。

從 Control Center 啟動及監督 worker。設定與擁有者尚未確認前，保持關閉遠端工作接受功能。角色分為 viewer、operator 與 admin；破壞性操作僅限 admin。

<a id="safety"></a>
## 操作邊界

面板只會啟動現有腳本，不會讓訓練結果自動適合真機。Export、deploy readiness、ROS preflight、實體急停準備與分階段馬達 enable 仍是獨立 gate。壓縮或刪除前，先保留選定 checkpoint 與證據路徑。

<a id="component-docs"></a>
## 元件文件

架構、遠端 contract、部署、疑難排解與版本資訊請見 [Training Panel 元件入口](../../../tools/training_panel/docs/index.zh-TW.md)。
