---
id: milestone-training-stack-reform
title: 2026-07-10 訓練 Stack 改革里程碑
lang: zh-TW
audience: shared
type: release
status: published
owner: project
last_reviewed: 2026-08-07
---

<a id="scope"></a>
## 範圍

此日期式專案 milestone 記錄 commit `5cdc824` 與之後文件來源線上可用的訓練及部署 hardening。它不是全域 semantic version。

<a id="training"></a>
## 訓練與環境

- 手動 `train.py` 只有帶 `--panel_overrides` 才套用面板 override 檔案；面板啟動工作會明確加入此參數。
- Action intent 改為每 control step 計算一次，不再於 physics substep 重複累積時間。
- 修正 observation-noise slice、初始化設定、height termination、global step counting 與重複 legacy reward 處理。
- 已停用物理上不一致的 left-right augmentation。
- Full、ForwardFast、privileged-teacher 與 distillation 路徑維持註冊。

<a id="deployment"></a>
## 部署與操作

- 鏡像 deployment contract 採用訓練的 60 Hz、ABAD normalization `0.60` 與 60-degree stage limit。
- Contract parity test 保護鏡像 constant。
- IMU mount rotation 與 rest projected-gravity 檢查會 gate policy enable，但仍需現場硬體驗證。
- Training Panel history write 使用 lock 與 atomic replacement。

<a id="remaining"></a>
## 尚存限制

Contact sensing、base linear-velocity estimation、模糊的 diagonal reward magnitude、observation-side state mutation、physics mass/actuator 假設、convergence-window semantics 與更廣泛 modularization 並未在此 milestone 解決。它們仍是明確 priority，不可暗示已修正。
