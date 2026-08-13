---
id: operator-torsion-spring-calibration
title: 校準並驗證被動扭轉彈簧
lang: zh-TW
audience: operator
type: how-to
status: draft
owner: sim2real
last_reviewed: 2026-08-13
---

<a id="status"></a>
## 目前狀態

Torsion feature branch 已有彈簧實作與驗證工具，但尚未完成實體校準、backend 選擇與 production retraining。Repository defaults 與 V11 checkpoint 都是 uncalibrated。在 calibrated-physics 與 policy-acceptance gates 都通過前，不得 promote 或 deploy checkpoint。

<a id="safety"></a>
## 先核准實體安全範圍

施力前，mechanical owner 必須識別 owner 與 fixture，並只核准一個正值上限：`maximum_safe_deflection_rad`、`maximum_safe_load_n` 或 `maximum_safe_torque_nm`。若缺少核准、fixture identity 或上限，立即停止。Calibration 與 holdout 必須使用相同核准及有限 neutral angle。

只量測代表性的 `damper_0` / `Revolute_5` 組件。只有 evidence gates 通過後，結果才可傳播到六個 aliases：

| Alias | Runtime joint | 暫定 neutral angle |
| --- | --- | --- |
| `damper_0` | `Revolute_5` | `+pi/4` rad |
| `damper_1` | `Revolute_8` | `+pi/4` rad |
| `damper_2` | `Revolute_13` | `-pi/4` rad |
| `damper_3` | `Revolute_25` | `+pi/4` rad |
| `damper_4` | `Revolute_26` | `+pi/4` rad |
| `damper_5` | `Revolute_27` | `+pi/4` rad |

<a id="capture"></a>
## 擷取 calibration 與 holdout episodes

記錄 angle、非負 load force、非負 lever arm、torque direction、sweep branch 與 repeat index。Signed torque 為 `load_force * lever_arm * torque_direction`；direction 必須恰為 `-1` 或 `+1`，branch 在 ordered loading 時為 `+1`、ordered unloading 時為 `-1`。

- `torsion-spring`：三次 repeat；兩個方向及兩種 branch 都使用核准範圍的 20%、40%、60%、80%。
- `torsion-spring-holdout`：另一個三次 repeat episode；在相同核准下使用 30%、50%、70%。

NPZ import 必須提供 `angle_time_s`、`angle`、`load_force_time_s`、`load_force`、`lever_arm_time_s`、`lever_arm`、`torque_direction`、`sweep_branch` 與 `repeat_index`。最後三項使用 angle clock。呼叫 `python -m tools.sim2real import-real` 時，同時提供 immutable neutral angle、依序為 `damper_0` 到 `damper_5` 的 alias list、mechanical approval object，以及明確 latency clock。

<a id="quality"></a>
## 套用線性模型 gates

只有所有 gates 都通過時才接受代表性線性 fit：

- calibration R² 至少 `0.98`；
- held-out torque RMSE 不超過 holdout full scale 的 5%；
- stiffness coefficient of variation 不超過 5%；
- loading/unloading hysteresis width 不超過 calibration full scale 的 10%；
- neutral-constrained model 的 held-out RMSE 也不超過 full scale 的 5%。

任何 gate 失敗都應停止，並報告需要 nonlinear 或 hysteretic model。通過的 fit 會把 neutral-constrained stiffness 複製到六個 aliases、保留各自 configured neutral angle，並在另有 dynamic measurement 識別前把 damping 設為零。Profile 持續綁定精確 calibration/holdout files、identities 與 hashes。

<a id="backend"></a>
## 選擇 simulator backend

使用已驗證的 calibrated profile，讓 `explicit` 與 `native` 各自在 120 Hz 與 240 Hz 執行 `spring-release`；四次執行必須有相同 seed、runtime、profile 與 parameters。接著執行：

```bash
python -m tools.sim2real select-spring-backend \
  --explicit-120 OUTPUT_EXPLICIT_120 \
  --explicit-240 OUTPUT_EXPLICIT_240 \
  --native-120 OUTPUT_NATIVE_120 \
  --native-240 OUTPUT_NATIVE_240 \
  --output outputs/sim2real/spring-backend-selection-calibrated.json
```

兩個 backends 都必須通過 restoring-sign、finite-state、rebound、passivity、fixture、static estimated-torque 與 cross-timestep gates。Energy/work residual 較低者勝出；差異不超過 10% 時，為了 auditability 選 `explicit`。Exit status `3` 表示沒有 eligible backend。

<a id="policy"></a>
## 重新訓練並接受 policies

使用 selected backend 與精確 calibrated profile 訓練 ForwardFast seeds 42、43、44。至少兩個 seeds 必須讓每個 command 達到：forward speed 至少 `0.15 m/s`、lateral leak 不超過 `0.12 m/s`、yaw leak 不超過 `0.30 rad/s`、fall rate 不超過 `0.20`。只有通過後才訓練相同 seeds 的完整 Direct task；至少兩個 seeds 的 overall pass ratio 必須達 `0.70`、每個 skill 達 `0.60`，而每個 command 的 fall rate 不超過 `0.20`。

執行 `python -m tools.sim2real validate-policy-acceptance --help` 查看六個 command/summary artifact arguments。若 evidence 未校準、seed 錯誤、混用 backend/profile identity，或 summary hash 與 command CSV 不符，validator 會拒絕。若 ForwardFast 失敗就停止；不得因單一 noisy training seed 而切換 backend。

<a id="evidence"></a>
## 證據邊界

[V11 checkpoint 摘要](../../research/2026-08-01-torsion-spring-v11-checkpoint.zh-TW.md)只屬於 implementation evidence，且未選出 backend。Promotion 邊界請同時遵循完整的[物理校準流程](physics-calibration.zh-TW.md)與 [sim-to-real 架構](../../developers/architecture/sim-to-real.zh-TW.md)。
