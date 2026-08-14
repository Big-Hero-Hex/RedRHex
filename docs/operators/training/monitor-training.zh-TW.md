---
id: operator-monitor-training
title: 監看訓練
lang: zh-TW
audience: operator
type: how-to
status: active
owner: training
last_reviewed: 2026-08-14
---

<a id="tensorboard"></a>
## 啟動 TensorBoard

```bash
tensorboard --logdir logs/rsl_rl --host 127.0.0.1 --port 6006
```

開啟 `http://127.0.0.1:6006`。Training Panel 也能為指定 run 啟動 TensorBoard；若 6006 已被占用，面板會選擇其他可用連接埠。

<a id="signals"></a>
## 判讀主要訊號

Reward 項目必須與 `Mean episode length`、`Episode_Termination/terminated`、command tracking error，以及 task 專用診斷一起檢查。節能實驗還要查看 mechanical power、cost-of-transport proxy、spring recovery 與 motion speed；若功率下降只是因為走得更慢或失敗，不能視為改善。

ForwardFast 請先比較 `Episode_Reward/diag_cmd_vx` 與 `Episode_Reward/diag_forward_vel`，再檢查 `diag_main_drive_target_vel_mean`、`diag_main_drive_vel_mean`、`diag_abad_forward_lock_error`、`rew_stall` 與 `rew_energy_per_distance`。在 Process Console 確認命令包含 run 的 task 與 `--initial_command forward` 前，不能把靜止的 Panel 影片當成訓練失敗證據。

<a id="artifacts"></a>
## 找到產物

RSL-RL run 位於 `logs/rsl_rl/<experiment>/<timestamp>_<run-name>/`。一般包含 `model_*.pt`、TensorBoard event 與 `params/`。Playback 會建立 `videos/play/`；export 會在 checkpoint 旁建立 `exported/policy.pt` 與 `exported/policy.onnx`。

面板的 process log、notes、activity 與 history 位於 `logs/training_panel/`。它們是執行期狀態，不可提交。

<a id="stop-conditions"></a>
## 停止條件

若 termination 暴增、episode length 崩落、數值變成非有限值、GPU 記憶體耗盡、Isaac 回報致命錯誤，或機器人取得 reward 卻沒有執行命令動作，應停止並診斷。刪除或壓縮產物前，先保留 run ID 與相關指標。
