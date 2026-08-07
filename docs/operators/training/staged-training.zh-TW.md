---
id: operator-staged-training
title: 執行五階段 Curriculum
lang: zh-TW
audience: operator
type: how-to
status: active
owner: training
last_reviewed: 2026-08-07
---

<a id="stage-purpose"></a>
## 各階段目的

完整 task 將 forward、lateral、diagonal、yaw 與 mixed 移動拆成第 1–5 階段。Pipeline 在所有階段沿用同一個 run tag，並預設完整 checkpoint resume，使 policy、optimizer 與 iteration 狀態保持連續。

<a id="launch"></a>
## 啟動 pipeline

```bash
bash scripts/rsl_rl/train_stage_pipeline.sh \
  --run_tag experiment-name \
  --num_envs 4096
```

預設 iteration 依序為 8,000、8,000、9,000、10,000 與 12,000，可用 `--s1` 到 `--s5` 覆寫。過夜 headless 訓練前可加 `--precheck_gui 1` 進行短暫視覺檢查。

<a id="health-gate"></a>
## 理解 health gate

預設啟用的穩定性 gate 會在每階段後讀取 episode length 與 termination 指標。缺少指標時，除非啟用 strict mode，否則只會警告；明顯不健康的指標會停止 pipeline。Pipeline log 位於 `logs/rsl_rl/pipeline/`。

<a id="restart"></a>
## 從後續階段重新開始

沿用原本的 run tag，並選擇 `--start_stage 2` 到 `5`。前一階段的 checkpoint 必須仍可在 `logs/rsl_rl/redrhex_wheg/` 找到。

Resume 時不要更換 run tag。只有刻意進行 policy-only 交接時才使用 `--resume_policy_only 1`；此時 model 編號與 optimizer 狀態不再代表同一條連續 curriculum。

<a id="acceptance"></a>
## 驗收階段

不要只看總 reward 就驗收。請同時檢查 episode length、termination、command tracking、該技能專用指標、playback 與下一階段 health gate。高 return 若伴隨倒地、幾乎不動或逃避命令，仍不算成功。
