---
id: torsion-spring-backends-design
title: 被動扭轉彈簧 Backends
lang: zh-TW
audience: developer
type: design
status: approved
owner: sim2real
last_reviewed: 2026-08-14
---

<a id="problem"></a>
## 問題

六個被動腿部 joints 需要已校準的 spring model，且在 training、playback、command sweeps、characterization、Training Panel actions 與 deployment evidence 中維持一致。暫定 simulator result 不得被誤認為已選擇或可部署的實體模型。

<a id="contract"></a>
## 穩定契約

Canonical aliases 為 `damper_0` 到 `damper_5`。每個 joint 遵循 `tau = -k(q-q0)-c*qdot`，並使用 unwrapped continuous-joint displacement。這些 joints 維持被動且不進入 policy action vector，因此 policy contract 仍為 12 actions 與 56 observations。

Training、playback、evaluation、sim-to-real execution 與 Panel processes 只接受 `explicit` 或 `native`。每次 run 都記錄選擇的 backend、依序有效 parameters、calibration status、profile identity/hash、deflection、torque estimate、potential energy、power 與 passivity diagnostics。

<a id="backends"></a>
## Backends

- `explicit` 將 PhysX spring gains 歸零，並在每個 physics substep 套用 restoring effort。
- `native` 把 stiffness 與 damping 寫入固定 neutral-angle target 的 PhysX implicit drives。

兩種 backend 都不增加 spring-law clip、人工 velocity brake 或 policy-controlled spring action。Native applied-torque channel 是 implicit-PD estimate，不是 force-sensor evidence。Physical defaults 仍是未校準的 `200 N*m/rad`、zero damping 與暫定 neutral angles。新的 environment 與 policy-training entry points 暫時預設使用 `native`；這是針對 Explicit 數值 runaway 的 operational quarantine，不是 production backend choice。

<a id="evidence"></a>
## 證據與選擇

實體校準使用代表性 `damper_0`、immutable calibration/holdout episodes、mechanical-owner approval 與 fail-closed quality gates。Calibrated fit 把 neutral-constrained stiffness 傳播到所有 aliases；在另行量測前 damping 保持零。Backend selection 要求兩種實作都先通過相符的 120/240 Hz release characterization，才可重新訓練。

<a id="panel"></a>
## Training Panel 傳播

Panel 建立 run 時驗證並儲存 `spring_backend`；Play、automatic video、export、deployment validation、history 與 remote synchronization 都重用記錄值。Policy-training entry points 會拒絕 `explicit`，而 deterministic sim-to-real characterization 與歷史 playback 仍保留此 backend。已 stamp 的 uncalibrated checkpoints 現在也會像 calibrated checkpoints 一樣拒絕 backend mismatch。Panel Play 與每個 Panel recording 都會提供 `--initial_command forward`；export 不會新增 motion command。

<a id="non-goals"></a>
## 非目標

本設計不會在沒有 calibrated evidence 時選擇 backend、不會用 static data 識別 damping、不改變 policy tensor contract、不重新設計 rewards、不授權 hardware deployment，也不把 V11 smoke checkpoint 當作 production evidence。
