---
id: operator-play-export-video
title: 播放、匯出與錄製 Policy
lang: zh-TW
audience: operator
type: how-to
status: active
owner: training
last_reviewed: 2026-08-14
---

<a id="checkpoint"></a>
## 選擇 checkpoint

請使用名為 `model_*.pt` 的訓練 checkpoint，不要使用 TensorBoard event 或已匯出的 `policy.pt`。保留其 run 目錄，因為設定與自動 stage 推斷會使用 checkpoint 路徑。

<a id="play"></a>
## 播放 policy

```bash
"$ISAACLAB_ROOT/isaaclab.sh" -p scripts/rsl_rl/play.py \
  --task Template-Redrhex-Direct-v0 \
  --load_run RUN_DIRECTORY \
  --checkpoint model_ITERATION.pt \
  --initial_command forward \
  --num_envs 1
```

鍵盤控制預設以前進命令開始，等同按下 `W`；在保存的命令中加入 `--initial_command forward` 可明確表達此意圖。若要靜止起步，請使用 `--initial_command stop`；若要保留環境取樣命令，請使用 `--disable_keyboard_control`。Checkpoint 路徑若含 `_stage1` 到 `_stage5`，會自動設定 `env.stage`；可用 `--disable_auto_stage_from_checkpoint` 關閉。

<a id="export"></a>
## 匯出 JIT 與 ONNX

```bash
"$ISAACLAB_ROOT/isaaclab.sh" -p scripts/rsl_rl/play.py \
  --task Template-Redrhex-Direct-v0 \
  --load_run RUN_DIRECTORY \
  --checkpoint model_ITERATION.pt \
  --export_policy_only \
  --headless
```

此命令會建立 `exported/policy.pt` 與 `exported/policy.onnx`。將 ONNX 複製到硬體前，必須執行[部署就緒檢查](../deployment/deployment-readiness.zh-TW.md)。

<a id="video"></a>
## 錄製影片

```bash
"$ISAACLAB_ROOT/isaaclab.sh" -p scripts/rsl_rl/play.py \
  --task Template-Redrhex-Direct-v0 \
  --load_run RUN_DIRECTORY \
  --checkpoint model_ITERATION.pt \
  --initial_command forward \
  --video \
  --video_length 1200 \
  --headless
```

若固定相機跟不上機器人，可加入 `--camera_follow_robot`。結果會寫入 run 的 `videos/play/` 目錄。

Training Panel 的 Play 與 Record Video 會重用所選 run 保存的 task，並明確以前進命令開始。ONNX export 也會重用保存的 task，但不需要初始移動命令。把靜止影片判定為 policy 失敗前，先在 Process Console 確認命令同時包含預期的 `--task` 與 `--initial_command forward`。

<a id="evaluate"></a>
## 評估行為

使用 `scripts/rsl_rl/eval_command_sweep.py` 執行可重複的 command profile 並產生 CSV。請依技能比較 tracking、success、fall 與 energy 指標，不要只憑一段 playback 影片判斷。
