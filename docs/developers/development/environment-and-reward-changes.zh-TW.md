---
id: environment-reward-development
title: 修改環境或 Reward
lang: zh-TW
audience: developer
type: how-to
status: active
owner: core
last_reviewed: 2026-08-07
---

<a id="classify"></a>
## 分類變更

先判斷變更影響 simulator physics、observation、action、reward、curriculum、randomization、termination、training configuration，或只影響內部結構。跨領域決策要建立 ADR；重大功能要有 approved design；多步驟工作使用暫時 plan。

<a id="trace"></a>
## 追蹤 contract

從 `redrhex_env_cfg.py` 開始，接著追蹤 `redrhex_env.py` 的使用處，再檢查 PPO/distillation 設定、train/play/evaluation script、面板參數建構、部署 parity 與 ROS observation/action 程式。搜尋鏡像的 dimension、rate、joint order、scale、command limit 與 artifact name。

<a id="test-first"></a>
## 先建立證據

先加入能針對預期行為失敗的最小測試。Pure helper 與 contract fact 應有 CPU test；Isaac 行為需要受限 simulator validation。Observation/action 變更需要 shape/order test 與 Torch/ONNX/ROS parity。Reward 變更需要 component-level diagnostic 及 evaluation 或 ablation protocol。

<a id="implement"></a>
## 一次只改一個語意軸

不要把 reward 變更與 physics、timing、logging 或大範圍 refactor 混在一起。除非 design 明確變更，否則保留舊介面。沒有 `--panel_overrides` 時，面板 override 檔案不可影響手動訓練；沒有 `--physics-profile` 時，不可載入 calibration candidate。

<a id="verify"></a>
## 驗證與文件

執行相關 CPU suite；Isaac stack 變更執行 `validate_reform_stack.py`；訓練變更執行短 PPO smoke；contract 變更執行 deployment parity。更新受影響操作/開發旅程的雙語文件，已交付行為加入 component release entry，並在 PR 宣告 documentation impact。
