---
id: operator-launch-training
title: 啟動訓練工作
lang: zh-TW
audience: operator
type: how-to
status: active
owner: training
last_reviewed: 2026-08-14
---

<a id="choose-task"></a>
## 選擇 task

- `Template-Redrhex-Direct-v0` 是完整移動 task，支援五階段 curriculum。
- `Template-Redrhex-ForwardFast-Direct-v0` 是範圍受限的快速 forward-only 設定。

除非刻意選擇 teacher 或 distillation agent 設定，否則請使用一般 PPO。

若要執行範圍受限的直走步態實驗，請使用完整 task ID `Template-Redrhex-ForwardFast-Direct-v0`；這現在也是 Training Panel 的預設值。原始 profile 與 Panel 中目前啟用的 `Straight Forward Focus` preset 使用相同的直走追蹤權重，並設定目前生效的簡化 reward dictionary；舊的扁平 `rew_scale_*` 欄位不會調整簡化 reward 路徑。

<a id="smoke-run"></a>
## 執行 smoke 訓練

設定 `ISAACLAB_ROOT` 後，先跑小型 headless 工作：

```bash
"$ISAACLAB_ROOT/isaaclab.sh" -p scripts/rsl_rl/train.py \
  --task Template-Redrhex-Direct-v0 \
  --num_envs 4 \
  --max_iterations 1 \
  --headless
```

成功代表程序完成一次更新，並在 `logs/rsl_rl/redrhex_wheg/` 寫入 run 目錄。Smoke 結果只能證明 stack 可執行，不能證明移動品質。

<a id="full-run"></a>
## 啟動較長訓練

Smoke 通過後才能增加 environment 數量與 iteration。實際值取決於 GPU 記憶體與實驗 protocol：

```bash
"$ISAACLAB_ROOT/isaaclab.sh" -p scripts/rsl_rl/train.py \
  --task Template-Redrhex-Direct-v0 \
  --num_envs 4096 \
  --max_iterations 8000 \
  --headless \
  --run_name baseline
```

手動執行預設會忽略面板產生的 reward 與 terrain override 檔案。只有在訓練刻意綁定這些檔案時才加入 `--panel_overrides`。

ForwardFast 請先跑一次 one-iteration smoke test；通過後，再使用此 profile 的 1,500-iteration 訓練範圍：

```bash
"$ISAACLAB_ROOT/isaaclab.sh" -p scripts/rsl_rl/train.py \
  --task Template-Redrhex-ForwardFast-Direct-v0 \
  --num_envs 4096 \
  --max_iterations 1500 \
  --headless \
  --run_name forward_spring_baseline
```

<a id="resume"></a>
## 接續訓練

完整 resume 會還原 policy、optimizer 與 iteration 狀態：

```bash
"$ISAACLAB_ROOT/isaaclab.sh" -p scripts/rsl_rl/train.py \
  --task Template-Redrhex-Direct-v0 \
  --resume \
  --load_run RUN_DIRECTORY \
  --checkpoint model_ITERATION.pt \
  --headless
```

只有在刻意進行 policy 權重交接時才使用 `--resume_policy_only`。它不保留 optimizer 連續性，必須記錄在實驗備註。

<a id="monitor"></a>
## 監看訓練

接著閱讀[監看訓練](monitor-training.zh-TW.md)。使用 `Ctrl+C` 停止工作，並等待 Isaac Sim 完全退出後再啟動另一個 GPU 工作。
