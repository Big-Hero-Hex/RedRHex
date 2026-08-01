# RedRHex torsion-spring workflow

This document is the operating procedure for the six passive leg torsion springs. It separates implementation verification, physical calibration, backend selection, and policy acceptance so that a provisional simulator result cannot be mistaken for a deployable model.

Current status: the spring implementation and its validation tooling exist, but no physical spring calibration or production retraining has been completed. The provisional v11 characterization did not select a backend. Do not promote or deploy a checkpoint from this workflow until the calibrated physics and policy gates below both pass.

## Implemented joints and invariant policy contract

The canonical spring order is also the stable calibration-alias order:

| Alias | Runtime joint | Leg | Provisional neutral angle |
| --- | --- | --- | --- |
| `damper_0` | `Revolute_5` | right front | +45 degrees (`+pi/4` rad) |
| `damper_1` | `Revolute_8` | right middle | +45 degrees (`+pi/4` rad) |
| `damper_2` | `Revolute_13` | right rear | -45 degrees (`-pi/4` rad) |
| `damper_3` | `Revolute_25` | left front | +45 degrees (`+pi/4` rad) |
| `damper_4` | `Revolute_26` | left middle | +45 degrees (`+pi/4` rad) |
| `damper_5` | `Revolute_27` | left rear | +45 degrees (`+pi/4` rad) |

For each joint, the shared model is

```text
tau = -k (q - q0) - c qdot
U   = 0.5 k (q - q0)^2
P   = tau qdot
```

`q - q0` is accumulated as an unwrapped continuous-joint displacement, so crossing `-pi`/`+pi` does not introduce a discontinuity. The provisional defaults are `k = 200 N*m/rad` and `c = 0 N*m*s/rad` on all six joints, with the per-joint neutral angles in the table. The repository starts on the explicit backend, but that is a provisional starting point rather than a production selection. Static calibration may replace stiffness, but damping stays zero until a separate physical release-response measurement identifies it.

The two interchangeable backends have the same effective parameters:

- `explicit` sets the spring joints' PhysX stiffness and damping to zero and applies the restoring effort on every physics substep. The production environment runs physics at 120 Hz.
- `native` writes the effective stiffness and damping to the PhysX implicit drives and keeps a fixed position target at `q0` and a fixed zero-velocity target. These targets are initialized/reset from the spring profile and never depend on policy actions or move during an episode.

Neither backend adds a spring-law torque clip, physical hard stop, or artificial velocity brake inside the validated operating range. The spring joints remain passive and outside the policy action vector. The checkpoint contract therefore remains 12 actions and 56 policy observations.

Training, playback, command sweeps, and characterization accept `--spring-backend explicit` or `--spring-backend native`. A profile is applied consistently to both the environment configuration and live articulation: explicit mode updates the torque-law tensors while keeping PhysX gains zero; native mode updates the PhysX gains and fixed neutral targets.

Each run records the effective joint order, stiffness, damping, neutral angles, backend, calibration status, profile identity/hash, deflection, model torque, potential energy, mechanical power, and passivity diagnostics. In characterization output, `spring_applied_torque_estimate` is the implicit-PD applied-torque estimate exposed by the articulation; it is not a measured PhysX joint torque. Treat the static-torque comparison that uses this channel as an estimate-based gate, not force-sensor evidence.

## Physical calibration gate

Before applying any physical load, a mechanical owner must approve the fixture and exactly one positive safe envelope:

- `maximum_safe_deflection_rad`, or
- `maximum_safe_load_n`, or
- `maximum_safe_torque_nm`.

The approval metadata must contain only `owner`, `fixture_id`, and the one selected envelope. The calibration and holdout episodes must use the same approval and the same finite `rest_position_rad`. Stop if fixture approval or the safe envelope is absent; simulator tooling is not a substitute for that safety decision.

Measure the representative physical assembly `damper_0` / `Revolute_5`. Record immutable numeric channels for angle in radians, non-negative load force in newtons, non-negative lever arm in metres, `torque_direction`, `sweep_branch`, and `repeat_index`. Signed torque is

```text
measured_torque = load_force * lever_arm * torque_direction
```

`torque_direction` must be exactly `-1` or `+1`. `sweep_branch` must be `+1` for ordered loading and `-1` for ordered unloading. Every one of the three repeats must include both torque directions and both branches.

Use these exact envelope levels:

- Calibration scenario `torsion-spring`: 20%, 40%, 60%, and 80% in every signed loading and unloading branch, for three repeats.
- Distinct holdout scenario `torsion-spring-holdout`: 30%, 50%, and 70% in every signed loading and unloading branch, for three repeats.

The importer tolerates at most 0.02 absolute error in the declared envelope fraction and rejects samples outside the approved maximum. The managed source declares

```json
{
  "rest_position_rad": 0.7853981633974483,
  "applies_to_spring_aliases": [
    "damper_0",
    "damper_1",
    "damper_2",
    "damper_3",
    "damper_4",
    "damper_5"
  ],
  "mechanical_owner_approval": {
    "owner": "MECHANICAL_OWNER",
    "fixture_id": "APPROVED_FIXTURE_ID",
    "maximum_safe_deflection_rad": "APPROVED_POSITIVE_NUMBER"
  }
}
```

This is a template, not importable JSON: replace the strings with the approved owner/fixture and replace `APPROVED_POSITIVE_NUMBER` with a positive JSON number. If the owner approves load or torque instead, replace `maximum_safe_deflection_rad` with the one corresponding field; never include two envelope fields. The same immutable constants accompany the holdout episode.

For an NPZ source, the required numeric arrays are `angle_time_s`, `angle`, `load_force_time_s`, `load_force`, `lever_arm_time_s`, `lever_arm`, `torque_direction`, `sweep_branch`, and `repeat_index`. The final three channels use the angle clock. Import each episode with `python -m tools.sim2real import-real`, `--scenario torsion-spring` or `--scenario torsion-spring-holdout`, and `--calibration-constants-json` containing the approved object. NPZ imports also require an explicit `--latency-clock`. See `docs/sim2real_calibration.md` for the full immutable-dataset import and profile-building example.

Accept the real-world linear model only when all five gates pass:

- calibration `R^2 >= 0.98`;
- held-out torque RMSE is at most 5% of measured holdout full-scale torque;
- stiffness coefficient of variation across the three repeats is at most 5%;
- loading/unloading hysteresis width is at most 10% of calibration full-scale torque;
- the model constrained to the configured neutral angle also has held-out RMSE at most 5% of full scale.

If any gate fails, stop before characterization-based selection or retraining and report that a nonlinear or hysteretic spring model is required. Do not tune the simulator around a failed linear fit.

An accepted representative fit propagates its neutral-constrained stiffness to all aliases `damper_0` through `damper_5`; the original representative trace, holdout trace, dataset/episode identities, and hashes remain attached to `passive_spring:damper_0`. Damping is overwritten to zero. The other five joints retain their configured per-joint neutral angles.

## Calibration status and profile binding

A run without an authenticated representative calibration and accepted holdout is stamped `uncalibrated`. This includes repository defaults, a profile with manually entered spring constants, and a calibration-only profile. Such runs may be used only for clearly named provisional smoke tests; they cannot be promoted, accepted for policy rollout, or deployed.

A run is stamped `calibrated` only when the profile points to the exact file-backed real calibration and holdout episodes, all quality gates pass, the source declares all six aliases in canonical order, all six effective stiffness values match the recomputed representative fit, damping is zero, and the configured representative neutral angle matches the measurement source. The files and hashes are reopened and recomputed when the profile is applied. Editing the trace, metadata, source record, profile parameters, or profile hash invalidates the stamp.

Training writes `params/torsion_spring.yaml` and, for a supplied profile, `params/physics_profile_metadata.json`. Playback and command-sweep evaluation require a calibrated checkpoint to use the same backend, profile ID, and profile SHA-256 with which it was trained.

## Four-run simulator characterization and backend selection

Run this only after producing an authenticated calibrated profile. All four jobs must use the same profile, seed, code/runtime identity, and effective spring parameters. The `spring-release` scenario fixes the root, sets gravity to zero, locks every non-tested joint, and releases `damper_0` from `+0.1`, `-0.1`, `+0.2`, and `-0.2` rad for 0.25 s each.

From the repository root:

```bash
export PYTHONPATH="$PWD/source/RedRhex:$PWD${PYTHONPATH:+:$PYTHONPATH}"
export SPRING_PROFILE="$PWD/profiles/torsion-spring-calibrated.json"

$ISAACLAB_ROOT/isaaclab.sh -p -m tools.sim2real run-sim \
  --scenario spring-release \
  --mode fixed-base \
  --physics-profile "$SPRING_PROFILE" \
  --spring-backend explicit \
  --physics-hz 120 \
  --seed 0 \
  --output outputs/sim2real/spring-release-explicit-120-calibrated \
  --headless \
  --device cuda:0

$ISAACLAB_ROOT/isaaclab.sh -p -m tools.sim2real run-sim \
  --scenario spring-release \
  --mode fixed-base \
  --physics-profile "$SPRING_PROFILE" \
  --spring-backend explicit \
  --physics-hz 240 \
  --seed 0 \
  --output outputs/sim2real/spring-release-explicit-240-calibrated \
  --headless \
  --device cuda:0

$ISAACLAB_ROOT/isaaclab.sh -p -m tools.sim2real run-sim \
  --scenario spring-release \
  --mode fixed-base \
  --physics-profile "$SPRING_PROFILE" \
  --spring-backend native \
  --physics-hz 120 \
  --seed 0 \
  --output outputs/sim2real/spring-release-native-120-calibrated \
  --headless \
  --device cuda:0

$ISAACLAB_ROOT/isaaclab.sh -p -m tools.sim2real run-sim \
  --scenario spring-release \
  --mode fixed-base \
  --physics-profile "$SPRING_PROFILE" \
  --spring-backend native \
  --physics-hz 240 \
  --seed 0 \
  --output outputs/sim2real/spring-release-native-240-calibrated \
  --headless \
  --device cuda:0
```

Do not pass `--steps`; the scenario derives the exact 120- or 240-frame duration from its one-second signed-release schedule. Output directories are immutable and must not already exist.

Apply the deterministic selection gate:

```bash
python -m tools.sim2real select-spring-backend \
  --explicit-120 outputs/sim2real/spring-release-explicit-120-calibrated \
  --explicit-240 outputs/sim2real/spring-release-explicit-240-calibrated \
  --native-120 outputs/sim2real/spring-release-native-120-calibrated \
  --native-240 outputs/sim2real/spring-release-native-240-calibrated \
  --output outputs/sim2real/spring-backend-selection-calibrated.json
```

The command exits with status 3 when no backend is eligible. It verifies matching provenance and parameters, calibrated stamping, restoring sign, finite state, no runaway, no continuous-angle unwrap ambiguity, a completed opposite-sign rebound in every release, static estimated-torque RMSE at most 1%, energy creation and absolute energy/work residual at most 2% of initial spring energy, settled fixture position/velocity errors at most `1e-4` rad and `1e-4` rad/s, and at most 2% interpolated first-rebound peak-angle difference between 120 and 240 Hz. Both backends must pass. The lower energy/work residual wins; if the residuals differ by no more than 10%, explicit wins for auditability. Both implementations remain available after selection.

## Retraining and policy acceptance

Do not begin formal retraining until the backend report has `status: selected`. Before calibration, only short, explicitly provisional smoke runs are allowed.

Set the selected backend and calibrated profile, then train ForwardFast seed 42 on both backends for 1,500 iterations:

```bash
export SPRING_PROFILE="$PWD/profiles/torsion-spring-calibrated.json"

for BACKEND in explicit native; do
  $ISAACLAB_ROOT/isaaclab.sh -p scripts/rsl_rl/train.py \
    --task Template-Redrhex-ForwardFast-Direct-v0 \
    --spring-backend "$BACKEND" \
    --physics-profile "$SPRING_PROFILE" \
    --seed 42 \
    --max_iterations 1500 \
    --num_envs 4096 \
    --run_name "torsion_${BACKEND}_seed42" \
    --headless \
    --device cuda:0
done
```

Then train ForwardFast seeds 43 and 44 only for the backend selected by the physics report:

```bash
export SELECTED_BACKEND=explicit  # set this from selected_backend in the report

for SEED in 43 44; do
  $ISAACLAB_ROOT/isaaclab.sh -p scripts/rsl_rl/train.py \
    --task Template-Redrhex-ForwardFast-Direct-v0 \
    --spring-backend "$SELECTED_BACKEND" \
    --physics-profile "$SPRING_PROFILE" \
    --seed "$SEED" \
    --max_iterations 1500 \
    --num_envs 4096 \
    --run_name "torsion_${SELECTED_BACKEND}_seed${SEED}" \
    --headless \
    --device cuda:0
done
```

Evaluate each selected-backend checkpoint with the command sweep. Replace each checkpoint variable with its actual `model_*.pt` path:

```bash
$ISAACLAB_ROOT/isaaclab.sh -p scripts/rsl_rl/eval_command_sweep.py \
  --task Template-Redrhex-ForwardFast-Direct-v0 \
  --checkpoint "$FORWARDFAST_CHECKPOINT_42" \
  --spring-backend "$SELECTED_BACKEND" \
  --physics-profile "$SPRING_PROFILE" \
  --seed 42 \
  --eval_profile stage1 \
  --csv outputs/acceptance/forwardfast-seed-42.csv \
  --headless \
  --device cuda:0
```

Repeat that exact command for seeds 43 and 44, changing the checkpoint, `--seed`, and CSV basename. Each evaluation produces the requested command CSV and a sibling summary CSV named by adding `_summary` before `.csv`, for example `forwardfast-seed-42_summary.csv`. Validate all six artifacts together:

```bash
python -m tools.sim2real validate-policy-acceptance \
  --stage forwardfast \
  --seed-42-command outputs/acceptance/forwardfast-seed-42.csv \
  --seed-42-summary outputs/acceptance/forwardfast-seed-42_summary.csv \
  --seed-43-command outputs/acceptance/forwardfast-seed-43.csv \
  --seed-43-summary outputs/acceptance/forwardfast-seed-43_summary.csv \
  --seed-44-command outputs/acceptance/forwardfast-seed-44.csv \
  --seed-44-summary outputs/acceptance/forwardfast-seed-44_summary.csv \
  --output outputs/acceptance/forwardfast-acceptance.json
```

At least two seeds must pass. For every ForwardFast command, the minimum forward speed must be at least 0.15 m/s, maximum lateral leak at most 0.12 m/s, maximum yaw leak at most 0.30 rad/s, and fall rate at most 0.20. The validator rejects uncalibrated evidence, incorrect seeds, mixed backends/profiles, or a command CSV whose SHA-256 does not match its summary.

Only after ForwardFast acceptance, train the full Direct task for 2,500 iterations on seeds 42, 43, and 44:

```bash
for SEED in 42 43 44; do
  $ISAACLAB_ROOT/isaaclab.sh -p scripts/rsl_rl/train.py \
    --task Template-Redrhex-Direct-v0 \
    --spring-backend "$SELECTED_BACKEND" \
    --physics-profile "$SPRING_PROFILE" \
    --seed "$SEED" \
    --max_iterations 2500 \
    --num_envs 4096 \
    --run_name "torsion_${SELECTED_BACKEND}_seed${SEED}" \
    --headless \
    --device cuda:0
done
```

Evaluate each Direct checkpoint with the same command-sweep invocation, changing the task to `Template-Redrhex-Direct-v0`, using `--eval_profile stage5`, and writing `direct-seed-42.csv`, `direct-seed-43.csv`, and `direct-seed-44.csv`. Then run:

```bash
python -m tools.sim2real validate-policy-acceptance \
  --stage direct \
  --seed-42-command outputs/acceptance/direct-seed-42.csv \
  --seed-42-summary outputs/acceptance/direct-seed-42_summary.csv \
  --seed-43-command outputs/acceptance/direct-seed-43.csv \
  --seed-43-summary outputs/acceptance/direct-seed-43_summary.csv \
  --seed-44-command outputs/acceptance/direct-seed-44.csv \
  --seed-44-summary outputs/acceptance/direct-seed-44_summary.csv \
  --output outputs/acceptance/direct-acceptance.json
```

At least two Direct seeds must have an overall command pass ratio of at least 0.70, a pass ratio of at least 0.60 for every one of `forward`, `lateral`, `diagonal`, and `yaw`, and a fall rate of at most 0.20 for every command.

If the selected backend fails ForwardFast acceptance, stop the rollout. Do not silently switch backend based on one noisy training seed.

Run the existing high-gain-hold checkpoint through the same command table and report tracking, falls, spring behavior, and energy per distance, but label that comparison observational. Its training physics and spring metadata differ from the new system, so it cannot establish backend acceptance, calibrated equivalence, promotion eligibility, or deployment readiness. Preserve its results separately from the six calibrated acceptance artifacts.

## Provisional v11 checkpoint

The v11 simulator checkpoint was generated from branch commit `185a4fd5627ae6e8c0d33caad6ab38cea3b09e0a` with the uncalibrated repository defaults (`200 N*m/rad`, zero damping, provisional neutral angles) and seed 0. Its four-run selection report is `outputs/sim2real/spring-backend-selection-v11.json`; all four traces share runtime bundle hash `dba80874b37fb0895ac6b90d353eed375b2f90a35edc88b06cdbb89162f69ec7`.

- Selection status is `blocked_uncalibrated`; `physics_passed` is false and `selected_backend` is null.
- Explicit is a severe runaway at both timesteps: maximum amplitude ratios are approximately 2,391.54 at 120 Hz and 1,767.15 at 240 Hz. It fails the energy, fixture, unwrap-ambiguity, runaway, and cross-timestep gates; the interpolated rebound peak difference is approximately 85.53%.
- Native does not create energy, remains unambiguous, completes every rebound, and passes its fixture checks, but it loses/fails to balance far too much energy: energy/work residual fractions are 1.00 at 120 Hz and approximately 2.78 at 240 Hz versus the 0.02 limit. Its interpolated rebound peak difference is approximately 61.76% versus the 0.02 limit.
- One-iteration ForwardFast smoke runs at seed 42 successfully instantiated both backends and wrote 12-action/56-observation checkpoints. They are `logs/rsl_rl/redrhex_forward_fast/2026-08-01_18-54-15_torsion_explicit_v11_provisional_smoke` and `logs/rsl_rl/redrhex_forward_fast/2026-08-01_18-54-31_torsion_native_v11_provisional_smoke`. Both are stamped `uncalibrated` and the deployment validator rejects them.
- The native smoke checkpoint was also loaded through the playback entry point and rendered for 120 frames. Its visual artifact is `logs/rsl_rl/redrhex_forward_fast/2026-08-01_18-54-31_torsion_native_v11_provisional_smoke/videos/play/rl-video-step-0.mp4`.

Therefore v11 is a useful implementation checkpoint, not evidence that either backend matches the real spring. The next externally blocked step is the approved physical `Revolute_5` calibration and holdout measurement; only then should the four characterization runs be repeated with the authenticated profile.
