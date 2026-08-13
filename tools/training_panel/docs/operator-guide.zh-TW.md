---
id: training-panel-operator-guide
title: Training Panel 3.4.10 操作指南
lang: zh-TW
audience: operator
type: how-to
status: active
owner: panel
last_reviewed: 2026-08-13
---

<a id="start"></a>
## 啟動 Mother

```bash
python -m tools.training_panel --host 127.0.0.1 --port 8080
```

開啟 `http://127.0.0.1:8080`。從其他電腦使用時，優先建立 SSH tunnel。本機面板是未驗證身分的管理介面；只有可信任 LAN 才可綁定 `0.0.0.0`。

<a id="train"></a>
## 訓練與 queue

在 Train 選擇 task、environment 數量、iteration、device、reward preset、terrain preset、spring backend 與 resume mode。Isaac/GPU action（training、playback、video 與 ONNX export）會序列化。已有 GPU action 時，新 training request 變成 queued；可從 History 取消。完成的 Isaac 工作之間有 settle window。

面板啟動的訓練使用 run-scoped override snapshot，並傳入 `--panel_overrides`。Built-in reward/terrain preset 為 read-only；修改前先 duplicate。

<a id="history"></a>
## 使用 History

History 結合 panel request 與探索到的 RSL-RL run。選擇 run 後可檢查 configuration、checkpoint、spring backend/calibration status、reward/terrain difference、note、folder、event state、video、export 與 readiness evidence。Play、recording、export 與 deployment checks 重用已記錄的 backend，並拒絕不相容的 spring metadata。可用 action 包括 TensorBoard、Play、Record Video、Export ONNX、Resume to Train、Compare、Compact Run 與 Process Console。

Compaction 會保留最高編號的 top-level `model_*.pt`，並保留 event、parameter、video、export、note 與 deployment report。刪除需要輸入確切 run ID；相關 process 執行中時會拒絕。

<a id="console"></a>
## 使用 Process Console

Launch Command 是面板要求的命令；Output 是捕捉到的 process stream。若有 tmux，工作在 detached session 執行，console 會提供 attach command。Stop Process 先送出 interrupt，只有 Isaac 未關閉才 escalation。

<a id="artifacts"></a>
## 匯出與錄影

Export ONNX 會從選定 training checkpoint 產生 `exported/policy.pt` 與 `exported/policy.onnx`。預設高品質 video preset 為 1920×1080、1,200 steps 與 30 FPS。若 run 有保存 terrain override，兩項 action 都會使用。

<a id="next"></a>
## 下一步

- [遠端操作](remote-operation.zh-TW.md)
- [部署就緒檢查](deploy-readiness.zh-TW.md)
- [疑難排解](troubleshooting.zh-TW.md)
