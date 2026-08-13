---
id: torsion-spring-rollout-plan
title: 扭轉彈簧校準與 Policy Rollout
lang: zh-TW
audience: developer
type: plan
status: active
owner: sim2real
last_reviewed: 2026-08-13
---

<a id="objective"></a>
## 目標

在不改變 12-action/56-observation contract 的前提下，將已實作的 explicit/native spring system 轉為有證據支持的 selected backend，以及通過驗收的 ForwardFast 與 Direct policies。

<a id="context"></a>
## 背景

Implementation、profile binding、Panel propagation、fail-closed deployment checks、characterization 與 acceptance validators 都已存在。V11 未校準，且沒有選出 backend。實體 spring calibration 與 holdout 是外部阻擋輸入。

<a id="phased-checklist"></a>
## 分階段清單

<a id="physical"></a>
### 實體證據

- [ ] 取得 mechanical-owner fixture approval，且只含一個 safe envelope。
- [ ] 為 `torsion-spring` 記錄三次 signed loading/unloading repeats，並建立另一個 `torsion-spring-holdout` episode。
- [ ] 匯入 immutable episodes 並要求所有 linear-model quality gates 通過；否則停止並指定 nonlinear model。

<a id="physics"></a>
### Physics 選擇

- [ ] 建立 authenticated profile，把通過的代表性 stiffness 套用到六個 aliases，並保持 damping 為零。
- [ ] 在相同 provenance 下執行 explicit/native 的 120 Hz 與 240 Hz `spring-release` characterization。
- [ ] 只有兩種實作都通過 deterministic gate 時才選擇 backend；否則保存 blocked report。

<a id="policy"></a>
### Policy 驗收

- [ ] 訓練 ForwardFast seeds 42–44、執行固定 command sweep，並要求至少兩個 passing seeds。
- [ ] ForwardFast 通過後，才訓練與評估完整 Direct seeds 42–44，並要求至少兩個 passing seeds。
- [ ] 歷史 high-gain-hold comparison 的 physics metadata 不同，因此維持 observational。

<a id="integration"></a>
### 整合

- [ ] 執行完整 sim-to-real 與 Training Panel suites，以及真正 Panel video/export 的 recorded-backend reuse 檢查。
- [ ] 更新 canonical operator/developer documentation，並發布有證據支持的 release 與 experiment records。
- [ ] 只有 calibrated evidence、selected backend、accepted policies 與 reviewed integration 全部存在後，才解決本 plan 與 approved design。

<a id="verification"></a>
## 驗證

必要證據包含 immutable source hashes、profile revalidation、四個 matched characterization artifacts、backend selection output、六個 ForwardFast command/summary files、六個 Direct files、deployment 對 uncalibrated checkpoint 的拒絕、Panel command/history assertions，以及可讀取的 recorded video。

<a id="completion-summary"></a>
## 完成摘要

目前仍等待實體 calibration 與 holdout；尚未選出 production backend 或可部署的 torsion policy。
