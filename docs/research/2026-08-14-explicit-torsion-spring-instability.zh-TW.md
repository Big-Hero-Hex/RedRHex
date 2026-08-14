---
id: explicit-torsion-spring-instability-2026-08-14
title: Explicit 扭轉彈簧數值不穩定
lang: zh-TW
audience: developer
type: experiment-summary
status: published
owner: sim2real
last_reviewed: 2026-08-14
---

<a id="question"></a>
## 問題

為什麼使用 `explicit` 扭轉彈簧 backend 時機器人會抖動並反覆 reset？在實體校準前，證據支持哪種操作？

<a id="baseline"></a>
## 基準

決定性基準為 fixed-base、zero-gravity 的 `spring-release`，不使用 policy 或 action input，seed 為 `0`，physics 為 120 Hz，並使用未校準的 `200 N*m/rad`、zero-damping spring。此基準使用 source commit `e86da0055d0b7db6da6af0017e33c8882f4b1413`、runtime bundle `5010e8362e522ebb45d2131338d62253cf9f6acbc76242376548ebed1b33a707` 與 trace `505e62f936200542e9deda1f177e73a2dbe6e1161d8d4c3061915d3a246eb193`。

<a id="method"></a>
## 方法

此 run 隔離 `Revolute_5`、鎖定其他 joints，並從 `+0.1 rad` 釋放彈簧。檢查內容包含 requested torque、applied-effort path、PhysX 與 actuator gains、前四個 physics samples、runtime mass properties，以及 semi-implicit stability condition。另外以只供診斷的變體，將六個 stiffness 降為 `20 N*m/rad`，或將機器人總質量從 `1.7985 kg` 統一縮放至 `14 kg`；兩者均用相同正式 release gates 評估。

<a id="results"></a>
## 結果

Restoring law 與方向正確：`+0.1 rad` 會要求 `-20 N*m`，static torque RMSE 為零，effort 在每個 physics substep 只寫入一次，且 explicit PhysX gains 為零。在 `0`、`8.33`、`16.67` 與 `25 ms` 的前四個 deflection 約為 `+0.100`、`-2.357`、`+23.417` 與 `-233.263 rad`；spring energy 由 `1.0 J` 增至約 `5.44 MJ`。

第一個 step 推導出有效 joint inertia 約為 `0.00056538 kg*m^2`。在 `k=200 N*m/rad` 時，undamped natural frequency 約為 `594.8 rad/s`，因此 120 Hz 的 `dt*omega` 約為 `4.956`，240 Hz 約為 `2.478`；兩者都超過 semi-implicit stability boundary `2`。Runtime articulation mass 為 `1.7985 kg`，而非設定診斷假設的約 `14 kg`。

基準 maximum amplitude ratio 約為 `2391.54`，energy/work residual fraction 約為 `5.872e15`。將 stiffness 降至 `20 N*m/rad` 可把 ratio 降至約 `1.4566`，但 residual fraction 仍約為 `15.68`。統一縮放至 14 kg 後，ratio 約為 `2.3602`，residual fraction 約為 `67.34`。三者均未通過 runaway、energy-creation 與 energy/work gates；這兩個診斷變體都不是已驗證的修正。

<a id="decision-impact"></a>
## 決策影響

`explicit` policy training 進入 quarantine。新的 environment 與 Training Panel policy runs 以 `native` 作為暫定 operational default，同時已記錄的 checkpoint backend identity 不得變更。`explicit` 仍保留用於決定性 spring characterization 與調查。這不代表已選擇 `native` 作為 production backend；仍需完成實體校準、相符的 backend characterization 與 policy acceptance。

<a id="limitations"></a>
## 限制

目前沒有實體 stiffness、damping、per-link mass、inertia 或 release-response 量測。診斷用 mass scale 不是 per-link calibration，降低後的 stiffness 也不是已量測彈簧。先前 V11 實驗中 Native 維持 finite，但未通過嚴格 energy/work 與 timestep-agreement gates。Raw traces 是本機忽略的 artifacts，不是已提交的 evidence packages。

<a id="artifacts-and-addenda"></a>
## 成品與附錄

基準 runtime artifact 為 `outputs/sim2real/explicit-shake-baseline-120-seed0`；診斷變體為 `explicit-shake-k20-120-seed0` 與 `explicit-shake-mass14-120-seed0`。前一份證據記錄為[扭轉彈簧 V11 暫定 checkpoint](2026-08-01-torsion-spring-v11-checkpoint.zh-TW.md)。
