---
id: reward-agent-weight-tuning
title: 使用 Reward Agent 調整 Reward 權重
lang: zh-TW
audience: developer
type: how-to
status: active
owner: reward-agent
last_reviewed: 2026-08-07
---

<a id="baseline"></a>
## 建立 baseline

選定一個 task、checkpoint policy、command profile、seed set、environment count、iteration budget 與 evaluation metric contract。必要 score input 為 command tracking、skill pass、stability、energy penalty 與 fall penalty。Baseline 尚未完整且可重現前，不要開始調整。

<a id="session"></a>
## 建立 session 與 candidate

```bash
python -m tools.reward_agent create-session --objective "improve forward tracking without worse falls"
python -m tools.reward_agent propose-candidates \
  --session-id SESSION_ID \
  --base-overrides-json '{"v2_reward_scales":{"velocity_tracking":4.0}}' \
  --scale velocity_tracking:3.5:4.5
```

預設 multiplier 為 0.8 與 1.2，接著 clamp 到 supplied bound。Generated ID 與 change record 對宣告順序具有 deterministic 結果。

<a id="preview"></a>
## 預覽 trial

傳入完整 `TrainingParams` JSON object，並檢查已儲存 dry-run record：

```bash
python -m tools.reward_agent queue-trials \
  --session-id SESSION_ID \
  --base-params-json '{"task":"Template-Redrhex-Direct-v0","num_envs":4,"max_iterations":1,"device":"cuda:0"}' \
  --limit 1 \
  --dry-run
```

Launch 前確認 task、run budget、device、candidate ID、reward override 與 client request ID。

<a id="launch"></a>
## 明確啟動

把檢查過的命令改為 `--launch` 再執行。Adapter 透過 Training Panel registry 排入工作，使 run history 與 override snapshot 沿用既有操作路徑。一個 candidate 尚未通過 smoke 與 evaluation 前，不可啟動大型 batch。

<a id="evaluate"></a>
## 評估與報告

Baseline 與 candidate 都收集相同必要 metric。不完整 evaluation 排在完整 evaluation 之後。較高 score 只用於選擇輔助；仍需檢查 component metric、regression、run configuration、artifact 與行為。只有結果改變 accepted baseline 或 recommendation 時，才提交雙語 experiment summary。
