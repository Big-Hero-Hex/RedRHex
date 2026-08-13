---
id: operator-torsion-spring-calibration
title: Calibrate and Validate the Passive Torsion Springs
lang: en
audience: operator
type: how-to
status: draft
owner: sim2real
last_reviewed: 2026-08-13
---

<a id="status"></a>
## Current status

The spring implementation and validation tooling exist on the torsion feature branch, but physical calibration, backend selection, and production retraining are not complete. Repository defaults and the V11 checkpoint are uncalibrated. Do not promote or deploy a checkpoint until both the calibrated-physics and policy-acceptance gates pass.

<a id="safety"></a>
## Approve the physical envelope first

Before applying a load, a mechanical owner must identify the owner and fixture and approve exactly one positive limit: `maximum_safe_deflection_rad`, `maximum_safe_load_n`, or `maximum_safe_torque_nm`. Stop if approval, fixture identity, or the limit is absent. Calibration and holdout must use the same approval and finite neutral angle.

Measure only the representative `damper_0` / `Revolute_5` assembly. The accepted result may propagate to all six aliases only after the evidence gates pass:

| Alias | Runtime joint | Provisional neutral angle |
| --- | --- | --- |
| `damper_0` | `Revolute_5` | `+pi/4` rad |
| `damper_1` | `Revolute_8` | `+pi/4` rad |
| `damper_2` | `Revolute_13` | `-pi/4` rad |
| `damper_3` | `Revolute_25` | `+pi/4` rad |
| `damper_4` | `Revolute_26` | `+pi/4` rad |
| `damper_5` | `Revolute_27` | `+pi/4` rad |

<a id="capture"></a>
## Capture calibration and holdout episodes

Record angle, non-negative load force, non-negative lever arm, torque direction, sweep branch, and repeat index. Signed torque is `load_force * lever_arm * torque_direction`; direction is exactly `-1` or `+1`, and branch is `+1` for ordered loading or `-1` for ordered unloading.

- `torsion-spring`: three repeats using 20%, 40%, 60%, and 80% of the approved envelope in both directions and both branches.
- `torsion-spring-holdout`: a distinct three-repeat episode using 30%, 50%, and 70% under the same approval.

For NPZ import, provide `angle_time_s`, `angle`, `load_force_time_s`, `load_force`, `lever_arm_time_s`, `lever_arm`, `torque_direction`, `sweep_branch`, and `repeat_index`. The last three use the angle clock. Supply the immutable neutral angle, the ordered alias list `damper_0` through `damper_5`, the mechanical approval object, and an explicit latency clock to `python -m tools.sim2real import-real`.

<a id="quality"></a>
## Apply the linear-model gates

Accept the representative linear fit only when all gates pass:

- calibration R² is at least `0.98`;
- held-out torque RMSE is at most 5% of holdout full scale;
- stiffness coefficient of variation is at most 5%;
- loading/unloading hysteresis width is at most 10% of calibration full scale;
- the neutral-constrained model also has held-out RMSE at most 5% of full scale.

If any gate fails, stop and report that a nonlinear or hysteretic model is required. An accepted fit copies the neutral-constrained stiffness to all six aliases, keeps their configured neutral angles, and sets damping to zero until a separate dynamic measurement identifies it. The profile remains bound to the exact calibration and holdout files, identities, and hashes.

<a id="backend"></a>
## Select the simulator backend

With an authenticated calibrated profile, run `spring-release` for `explicit` and `native` at both 120 Hz and 240 Hz with identical seed, runtime, profile, and parameters. Then run:

```bash
python -m tools.sim2real select-spring-backend \
  --explicit-120 OUTPUT_EXPLICIT_120 \
  --explicit-240 OUTPUT_EXPLICIT_240 \
  --native-120 OUTPUT_NATIVE_120 \
  --native-240 OUTPUT_NATIVE_240 \
  --output outputs/sim2real/spring-backend-selection-calibrated.json
```

Both backends must pass restoring-sign, finite-state, rebound, passivity, fixture, static estimated-torque, and cross-timestep gates. The lower energy/work residual wins; when residuals differ by no more than 10%, `explicit` wins for auditability. Exit status `3` means no backend is eligible.

<a id="policy"></a>
## Retrain and accept policies

Train ForwardFast seeds 42, 43, and 44 with the selected backend and exact calibrated profile. At least two seeds must pass every command with minimum forward speed `0.15 m/s`, lateral leak at most `0.12 m/s`, yaw leak at most `0.30 rad/s`, and fall rate at most `0.20`. Only then train the full Direct task for the same seeds; at least two seeds need overall pass ratio `0.70`, per-skill pass ratio `0.60`, and per-command fall rate at most `0.20`.

Use `python -m tools.sim2real validate-policy-acceptance --help` for the six command/summary artifact arguments. The validator rejects uncalibrated evidence, wrong seeds, mixed backend/profile identity, or summary hashes that do not match their command CSV. If ForwardFast fails, stop; do not switch backend based on one noisy training seed.

<a id="evidence"></a>
## Evidence boundary

The [V11 checkpoint summary](../../research/2026-08-01-torsion-spring-v11-checkpoint.en.md) is implementation evidence only and selected no backend. Follow the broader [physics calibration workflow](physics-calibration.en.md) and [sim-to-real architecture](../../developers/architecture/sim-to-real.en.md) for promotion boundaries.
