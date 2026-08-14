---
id: torsion-spring-rollout-plan
title: 扭轉彈簧校準與 Policy Rollout
lang: zh-TW
audience: developer
type: plan
status: active
owner: sim2real
last_reviewed: 2026-08-14
---

<a id="objective"></a>
## 目標

在不改變 12-action/56-observation contract 的前提下，將已實作的 explicit/native spring system 轉為有證據支持的 selected backend，以及通過驗收的 ForwardFast 與 Direct policies。

<a id="context"></a>
## 背景

Implementation、profile binding、Panel propagation、fail-closed deployment checks、characterization 與 acceptance validators 都已存在。V11 未校準，且沒有選出 backend。決定性重跑已證明 Explicit 目前的 200 N*m/rad model 在 120 與 240 Hz 會發生數值不穩定，因此 policy training 進入 quarantine，characterization 仍可使用。實體 spring calibration、per-link mass/inertia evidence 與 holdout 是外部阻擋輸入。

<a id="phased-checklist"></a>
## 分階段清單

<a id="physical"></a>
### 實體證據

- [ ] 取得 mechanical-owner fixture approval，且只含一個 safe envelope。
- [ ] 為 `torsion-spring` 記錄三次 signed loading/unloading repeats，並建立另一個 `torsion-spring-holdout` episode。
- [ ] 匯入 immutable episodes 並要求所有 linear-model quality gates 通過；否則停止並指定 nonlinear model。

<a id="physics"></a>
### Physics 選擇

- [x] 在沒有 policy actions 的情況下重現 Explicit runaway、發布數值不穩定證據，並 quarantine 新的 Explicit policy training。
- [ ] 依 reviewed physical evidence 稽核並修正 per-link mass 與 inertia；不得把統一 mass scaling 或任意 armature 提升為修正。
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
- [x] 發布 Explicit instability experiment、quarantine behavior、operator guidance 與 3.6.1 safety release。
- [ ] 在 backend selection 與 policy acceptance 後，發布最終 calibrated-backend documentation 與 release evidence。
- [ ] 只有 calibrated evidence、selected backend、accepted policies 與 reviewed integration 全部存在後，才解決本 plan 與 approved design。

<a id="verification"></a>
## 驗證

必要證據包含 immutable source hashes、profile revalidation、四個 matched characterization artifacts、backend selection output、六個 ForwardFast command/summary files、六個 Direct files、deployment 對 uncalibrated checkpoint 的拒絕、Panel command/history assertions，以及可讀取的 recorded video。

<a id="completion-summary"></a>
## 完成摘要

Explicit policy training 已進入 quarantine，Native 只是暫定 operational default。實體 calibration、per-link mass/inertia evidence 與 holdout 仍待完成；尚未選出 production backend 或可部署的 torsion policy。
