---
id: reward-energy-model
title: Reward 與能量模型
lang: zh-TW
audience: developer
type: explanation
status: active
owner: training
last_reviewed: 2026-08-07
---

<a id="current-reward"></a>
## 目前 reward 邊界

現行環境使用簡化、command-aware reward 路徑。它結合 forward progress、linear/angular tracking、mode specialization、不需要軸向的 suppression、gait/height shaping、movement/alive 行為、stall/fall penalty，以及單一 energy 項。`RedrhexEnvCfg.v2_reward_scales` 是目前真實來源。

先前四項提案（`power_efficiency`、`spring_recovery`、`spring_utilization` 與 `torque_penalty`）並非目前啟用的 reward 介面。Spring 與 power 數量仍可作為診斷與實驗假設。

<a id="mechanical-model"></a>
## 機械模型

對主動 joint，估計瞬時 mechanical power 以 `abs(torque * angular_velocity)` 為基礎。Main-drive torque 由 velocity-control error 做有界估計；ABAD torque 由 position/velocity error 做有界估計。被動 damper energy 使用設定的線性 torsion model，stiffness 為 `200 N·m/rad`，damping 為 `20 N·m·s/rad`。

這些是 simulator/controller 估計值，不是電池電功率。除非另外量測，current、voltage、gearing、friction、hysteresis、driver loss 與 sensor error 均不在估計範圍內。

<a id="active-energy-term"></a>
## 啟用中的 energy 項

`energy_per_distance` 將累積估計 mechanical energy 除以 command 正方向位移，再以 epsilon 與最大值 clamp 處理。預設權重為 `0.001`；ForwardFast 使用 `0.0005`。它仍次於 tracking 與 gait 項。

對 yaw、lateral 與 diagonal 工作，應比較 command-aware motion 定義，而不是只看 raw forward distance。若 proxy 降低是因為變慢、停滯或倒地，就不能視為效率提升。

<a id="validation"></a>
## 驗證 protocol

在相同 seed 與 command profile 下比較 baseline 與有界 candidate。Tracking 與 success 必須維持可接受、fall rate 不得惡化，且 cost-of-transport proxy 與 power per motion 都要改善。結果需分別回報 forward、lateral、diagonal 與 yaw。只有證據改變建議或 baseline 時才發布 experiment summary。

<a id="limitations"></a>
## 限制

目前模型使用 torque 與 contact proxy，尚未證明真機電能確實節省。在 robot mass、velocity、current、voltage、timing 與 sensor calibration 綁定到已審查證據前，cost of transport 只能視為比較 proxy。
