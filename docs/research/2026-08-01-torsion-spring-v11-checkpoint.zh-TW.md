---
id: torsion-spring-v11-checkpoint
title: 扭轉彈簧 V11 暫定 Checkpoint
lang: zh-TW
audience: developer
type: experiment-summary
status: published
owner: sim2real
last_reviewed: 2026-08-13
---

<a id="question"></a>
## 問題

在實體校準前，暫定 linear spring implementation 是否提供足夠 simulator evidence 來選擇 `explicit` 或 `native`？

<a id="method"></a>
## 方法

Checkpoint 使用 source commit `185a4fd5627ae6e8c0d33caad6ab38cea3b09e0a`、seed 0，以及 `200 N*m/rad`、zero damping 與暫定 neutral angles 的 uncalibrated defaults。四個 `spring-release` traces 在 runtime bundle hash `dba80874b37fb0895ac6b90d353eed375b2f90a35edc88b06cdbb89162f69ec7` 下，比較 explicit/native 的 120 Hz 與 240 Hz。ForwardFast seed-42 的 one-iteration smoke runs 檢查 environment creation、metadata、checkpoint shape 與 playback。

<a id="results"></a>
## 結果

Selection status 為 `blocked_uncalibrated`；`physics_passed` 是 false，`selected_backend` 是 null。

- Explicit 在兩種 timestep 都 runaway；maximum amplitude ratio 在 120 Hz 約 2,391.54、240 Hz 約 1,767.15。它未通過 energy、fixture、unwrap-ambiguity、runaway 與 cross-timestep gates；rebound peak difference 約 85.53%。
- Native 保持 finite、完成 rebounds 並通過 fixture checks，但未通過 energy/work balance 與 timestep agreement。Residual fraction 在 120 Hz 為 1.00、240 Hz 約 2.78，而限制為 0.02；rebound peak difference 約 61.76%。
- 兩個 smoke runs 都產生 12-action/56-observation checkpoints，標記為 `uncalibrated`，且 deployment validation 拒絕它們。Native playback 成功 render 120 frames。

<a id="limitations"></a>
## 限制

沒有實體 calibration 或 holdout，因此 simulator comparison 無法建立 real-spring fidelity。Applied-torque comparison 使用 implicit-PD estimate，不是 measured PhysX joint torque。One-iteration smoke runs 只驗證整合，不驗證 locomotion quality。

<a id="decision"></a>
## 決策影響

V11 保留為 implementation checkpoint，不是 backend-selection 或 deployment evidence。下一步必須先取得已核准的實體 `damper_0` / `Revolute_5` calibration 與 holdout，再使用 authenticated profile 重做四次 characterization。

<a id="provenance"></a>
## Provenance

原始詳細報告與 commands 仍可從 `archive/source/torsion-spring-2026-08-13` 的 `docs/torsion_spring_workflow.md` 取得。Raw outputs 與 logs 維持為未提交 runtime artifacts。
