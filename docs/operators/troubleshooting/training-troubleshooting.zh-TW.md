---
id: operator-training-troubleshooting
title: 排除訓練與 Playback 問題
lang: zh-TW
audience: operator
type: troubleshooting
status: active
owner: training
last_reviewed: 2026-08-16
---

<a id="imports"></a>
## Import 或 task 錯誤

若無法 import `isaaclab`、`pxr`、`redrhex_policy_io` 或 RedRHex task，請執行 `isaaclab.sh -p scripts/install_redrhex.py`，再用 `scripts/list_envs.py` 驗證。Installer 會使用同一個 interpreter，並在擴充套件之前先安裝儲存庫內的共用 distribution。不要把一般 shell Python 與 Isaac Lab interpreter 混用。

<a id="assets"></a>
## USD 資產遺失或過小

執行 `git lfs install` 與 `git lfs pull`。約 130 bytes 的 USD 通常是尚未解析的 LFS pointer，不是有效的機器人資產。

<a id="memory"></a>
## CUDA 記憶體不足

降低 `--num_envs`、停止其他占用 GPU 記憶體的 Isaac 或 TensorBoard 程序，再重新執行 smoke。記憶體使用穩定前不要再次增加 environment 數量。

<a id="checkpoint"></a>
## 找不到或拒絕 checkpoint

請選擇 `model_*.pt` 訓練 checkpoint。使用相對檔名時，也要提供對應的 `--load_run`。已匯出的 `policy.pt` 與 TensorBoard event 不是 runner checkpoint。

<a id="behavior"></a>
## 彈飛、倒地或完全不動

先用小型固定 task 與原 checkpoint 設定重現。檢查自動 stage 推斷、terrain/reward override、termination 指標、base height/tilt、command tracking，以及 playback 是否從 `stop` 開始。不要還沒排除過期假設就直接修改 reward 權重。

<a id="panel"></a>
## 面板或 remote worker 問題

確認 8080 未被占用、查看 Process Console，並驗證設定的 repository 與 Isaac 路徑。遠端模式先以私有 environment file 執行 worker 一次，檢查 machine heartbeat 與 `accept_jobs`，並避免讓秘密出現在瀏覽器可見設定。

<a id="deployment"></a>
## 部署問題

任何 blocked readiness report、ONNX shape 不符、非有限值 inference、contract 不符、多個 motor publisher、heartbeat 遺失或 E-stop fault 都是停止條件。請接著閱讀 [ROS 疑難排解指南](../../../ros2_ws/src/redrhex_rl_controller/docs/troubleshooting.zh-TW.md)。
