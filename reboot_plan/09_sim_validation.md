# 09 — Mandatory Pre-Baseline Simulation and Gravity Gate

P1 determines whether the legacy simulation is internally correct and sufficiently
grounded in sourced physical facts to become the P2 oracle. It does not claim complete
sim-to-real equivalence; later hardware residuals remain necessary. P1 must pass before
any golden rollout, validated tag, checkpoint oracle, or reference-training run.

Physical/frame inputs are recorded in
[`evidence/sim_validation/FACTS.md`](evidence/sim_validation/FACTS.md); results go to
[`evidence/sim_validation/RESULTS.md`](evidence/sim_validation/RESULTS.md).

## Current hypothesis, not diagnosis

The configured world gravity is `(0, 0, -9.81)`. The more suspicious boundary is the
configured `wxyz` quaternion near `(0.7071068, 0.7071068, 0, 0)`, a +90° X rotation. If
that is the runtime root pose, analytic projected gravity is near `(0, -1, 0)`.

That may be intentional. It is wrong only if asset, root, policy, observation, command,
reward, or deploy code uses incompatible semantic axes. Density-derived mass, body
damping, actuator softness, and phase-proxy contact are also unverified suspects. No
source change is justified until the probes localize a failed layer.

## Status and blocking policy

Required G-checks use only `PASS`, `FAIL`, or `BLOCKED`.

- Every G0–G10 check must be `PASS` before P2. There is no generic waiver for wrong
  gravity, unexplained frames, unavailable physical facts, invalid mass/inertia,
  unvalidated actuators, contact instability, reward-axis errors, or timing drift.
- A threshold may be corrected only with an approved ADR explaining why the original
  criterion was invalid, followed by a fresh affected run. It does not convert evidence
  directly to PASS.
- C1 is a separate frozen-ROS compatibility finding. `MISMATCH` or
  `BLOCKED_ON_IMU_GROUND_TRUTH` does not force the simulator to copy an unvalidated ROS
  transform and does not modify `ros2_ws/**`; it remains a declared P7/post-reboot risk.

## Execution rules

- Use one environment, fixed seed, and no DR, observation noise, pushes, policy, or gait
  logic unless a check explicitly requires them.
- Do not use `zero_agent.py` as a gravity probe: zero actions can still produce implicit
  actuator and damper forces.
- Every check emits raw machine-readable data, fitted summary, predeclared thresholds,
  and status.
- Stop failed descendants, not unrelated branches. The prerequisite graph—not numeric
  check order—controls what can still run.
- Test caches/bytecode go outside frozen paths.

## Prerequisite graph

```text
G0 -> G1 -> G2
      +----> G3
      +----> G4

G2 + G3 + G4 ------> G5
G3 + G4 + G5 ------> G6a ideal-constrained contact
G3 + G4 -----------> G8 actuator/rest-hold
G6a + G8 ----------> G6b actuator-held settle
G1 + G3 -----------> G7a analytic frames
G6b ----------------> G7b settled-rest frame
G7a + G7b + G8 ----> G9 rewards/dones
G0 + G1 + G3 ------> G10a timing/freshness characterization
G6b + G8 + G9 -----> G10b determinism holdout
G7 + G9 ------------> C1 frozen ROS comparison
```

G3 and G4 may proceed independently. For example, missing measured mass blocks G4 and
its descendants but does not prevent analytic frame work in G7.

## G0 — Reproducible run manifest

Record the P0 toolchain-manifest ID plus source commit/dirty policy, USD/config hashes,
task, seed/RNG, dt/decimation, device/GPU, exact command, and every disabled override,
randomization, noise, and push setting.

**Pass:** the manifest is complete, one environment is resolved, and there are no hidden
inputs or unapproved dirty states.

## G1 — Composed stage, units, and world gravity

Dump composed-stage `upAxis`, `metersPerUnit`, every PhysicsScene and the active scene.
Archive the raw gravity direction, nonnegative magnitude, normalized direction, ground
normal, and the SI vector after converting stage-length units with `metersPerUnit`.

**Pass:** one active scene; Z-up; documented units; normalized gravity points world -Z;
SI vector differs from `(0,0,-9.81) m/s²` by less than `1e-5`; ground normal is +Z.

Per-body gravity flags are intentionally checked in G4/G5, not used to misclassify an
asset problem as a world-gravity problem.

## G2 — Canonical-body free fall

Drop two isolated primitive bodies of different masses from 10 m for 0.5 s with no
ground/contact, damping, controller, or external force. Record SI position/velocity at
each physics step; fit velocity and integrator-aware position independently.

**Pass:** `|az + 9.81| <= 0.049 m/s²`; horizontal acceleration magnitude below
`0.01 m/s²`; mass-to-mass difference below `0.01 m/s²`; fit disagreement below
`0.02 m/s²`.

Failure here is stage/simulator physics, not the robot asset.

## G3 — Asset, root, semantic axes, and quaternion convention

Before running, declare with sources:

- handedness, world axes, and semantic robot forward/left/up;
- quaternion storage (`wxyz`), direction (root-to-world), multiplication/composition
  order, and the fixed root-to-policy transform;
- intended spawn transform and CAD dimensions.

Inspect asset/source units, transforms, visual/collision bounds, and reset quaternion.
Transform semantic basis markers through the spawn pose.

**Pass:** dimensions agree with sourced CAD/measurement within 2%; semantic basis lands
on intended world axes within 2°; reset quaternion is within 0.05° of intended pose. The
root-to-policy transform must be a proper rotation: `max|R^T R - I| <= 1e-6`,
`|det(R)-1| <= 1e-6`, `max|R^-1-R^T| <= 1e-6`, and norm preservation within `1e-6`.
If quaternion-backed, quaternion norm and independently constructed matrix agreement are
also within `1e-6`. Missing semantic/CAD truth is `BLOCKED`, not inferred from whichever
current code path is convenient.

## G4 — Mass, density provenance, center of mass, and inertia

From the live articulation, record for every link:

- prim/name and gravity-enabled flag;
- effective mass source: explicit mass, density × computed volume, inherited override,
  or fallback;
- effective density/volume and sensitivity to relevant density overrides;
- world/local CoM and full inertia about the CoM;
- eigenvalues, triangle inequalities, principal ratio, and scale-aware `I/(m*L²)` using
  sourced link dimensions.

Compare total/modules with weighed/CAD facts. Check each CoM against the sourced physical
envelope or aggregate visual hull; do not assume a collision AABB is always the body.

**Pass:** total within 5% of measured deployed mass; sourced modules within 15%; every
body has intended gravity; mass provenance is explained; inertias are finite/SPD,
dimensionally plausible, CAD-consistent where available, and satisfy triangle
inequalities; CoMs lie in sourced physical envelopes. Missing measurements/CAD block G4.

## G5 — Robot whole-COM free fall and damping isolation

Spawn the articulation high without terrain, policy, controller targets, or self-contact.
Compute whole-system COM position and velocity from each link's mass, COM position, and
COM velocity—not link-origin velocity.

Run linear/angular damping combinations:

```text
(0, 0), (configured linear, 0), (0, configured angular), (configured linear, configured angular)
```

Also run a vacuum spin-down probe for angular damping.

**Pass:** zero-damping COM acceleration is within 1% of `(0,0,-9.81)` and horizontal
acceleration below `0.02 m/s²`; linear-only affects translation as predicted; angular-only
spin-down matches the declared model; no unexplained actuator/external force appears.
Configured damping must fit a band sourced before the run from hardware/CAD/model intent;
without that band, its physical-fidelity result is BLOCKED.

## G6 — Physical contact, clearance, drop, and settle

G6a first records signed spawn/rest-pose clearance, then drops/settles on flat ground
using ideal diagnostic joint constraints—not configured actuators—and no policy/CPG.
This isolates collision/contact behavior. G6b repeats with configured actuator rest-hold
only after G8 passes, so actuator weakness cannot masquerade as contact failure. Define
separation sign and log all external contact pairs, points, normals, separation,
impulses/forces, colliders, whole-COM/root state, chassis angular speed, constraint or
actuator forces, and kinetic energy at physics rate.

**Pass for both G6a and G6b:** pre-step rest-pose penetration below 1 mm; steady maximum penetration below
3 mm; no nominal base-ground contact; final 0.5 s average vertical support within 5% of
`M*g` after summing all external contacts; horizontal net force below 5% of `M*g`; settle
within 1 s and remain 0.5 s with COM speed below `0.01 m/s`, chassis angular speed below
`0.02 rad/s`, and base-height standard deviation below 1 mm. G6b actuator forces must
also remain inside the sourced G8 rest-hold bands. A G6a pass/G6b fail is an actuator or
rest-controller finding, not a contact failure.

The gait-phase `_contact_count` is not physical contact evidence. If later promoted as a
proxy, require physical-contact precision and recall each at least 0.90.

## G7 — Full-vector frame and projected-gravity truth

Independently generate complete numeric vector oracles for neutral, semantic ±10° and
±90° roll/pitch, and multiple world-yaw rotations about +Z using the declared
root-to-world composition order. For every pose record:

- runtime quaternion and full expected/actual projected-gravity vector;
- expected/actual policy-frame linear velocity for injected world motions;
- expected/actual policy-frame angular rate for semantic roll/pitch/yaw;
- vector errors and norms, not only dot products.

After G6b, separately capture the settled-rest projected-gravity mean and standard
deviation; exact teleported `q0` is not a substitute for deployed rest.

**Pass:** every full vector/sign matches the independent oracle within `0.002` for
projected gravity and the predeclared velocity/rate tolerance; observation values match
the independently recomputed values within `1e-5`; norm within `0.002`; world yaw leaves
gravity invariant. Settled-rest statistics are stable and explained by the declared
root-to-policy transform. Against an independently sourced neutral chassis attitude,
the settled-rest mean angular error must be at most 2° and angular standard deviation at
most 0.2° over the final 0.5 s. Missing neutral-pose truth is `BLOCKED`.

## G8 — Actuator and action-intent response

Without a learned policy, probe both action families:

- main-drive velocity steps across the valid range: target, measured velocity, applied
  torque, saturation, rise time, steady-state error, and torque-speed behavior;
- ABAD/rest-hold and position steps: target/actual position, torque, overshoot, settling,
  limit enforcement, and gravity-load hold.

**Pass:** pure action decoding obeys current scales/order/limits; simulator targets and
applied forces are finite and repeatable; saturation/limits are enforced; response bands
match pre-run actuator datasheet/measured/model facts. Missing actuator truth blocks G8.

## G9 — Task-specific rewards, terminations, and command domains

For both frozen Gym IDs, build a matrix from the task's actual configured command domain.
Do not demand negative-forward symmetry if a task excludes negative `vx`. For each valid
forward/lateral/yaw command and analytic G7 state, record every reward component, total,
termination cause, and done mask with precomputed numeric/sign expectations.

**Pass:** each tracking term is maximal for its matching semantic motion; required sign
symmetries and asymmetric task rules match explicit numeric expectations; tilt/fall terms
respond only to semantic orientation; termination causes match their declared thresholds.

If confirmed, record the dormant legacy `sum(projected_gravity[:2]^2)` assumption as a
dormant defect; do not enable, delete, or fix it during P1.

## G10 — Timing, freshness, write cadence, and determinism

Before execution, fill an expected step-index table in `FACTS.md` for action intent,
physics state, observation, each reward group, termination/done, and reset mutation. Use
symbols such as pre-step `S_k`, applied intent `A_k`, and post-decimation `S_(k+1)` so
“fresh” has an exact meaning rather than a log-message interpretation.

Instrument actual execution over a fixed input/command rollout:

- physics-step rate, control rate/decimation, and substep count;
- action-intent computation count (must be once per control step);
- adapter target-flush/write count and exact substep cadence (measured, not assumed once);
- expected versus actual step index for every state/observation/reward/done consumer;
- episode duration, command/gait timers, phase advancement, resets, and cooldowns in
  simulated seconds;
- same-seed, fresh-process repeatability for root/link/joint state, observation slices,
  reward components/total, termination/done/reset sequences, and gait/command state.

The determinism algorithm is fixed before data:

1. Declare per-channel numeric floors `F_m`, normalization scales, rollout length, seed,
   input sequence, and metrics: maximum absolute error, normalized RMS error, first
   threshold-crossing step, and exact equality for integers/booleans/event sequences.
2. Run five characterization repeats in fresh simulator processes. For each float metric,
   let `D_m` be the maximum pairwise divergence and freeze `T_m = max(F_m, 2*D_m)` in a
   hashed envelope file. Do not inspect holdout output before that file is written.
3. Restart the runtime and run three holdout repeats. Every float metric must remain
   within its frozen `T_m`; done/reset/event sequences and integer fields must be exact.
4. If the envelope grows with rollout length, hides a semantic divergence, or requires
   a physically meaningless tolerance, G10 fails even if the formula can contain it.

**Pass:** rates and durations equal resolved cfg/declared behavior; intent computes once
per control step; target flushing matches the measured/documented Isaac adapter contract;
the actual step-index table exactly matches the approved expected table; no
substep-multiplied timer exists; all three fresh holdouts satisfy the previously frozen
envelope and exact event rules. The hashed G10 envelope becomes the P2 replay tolerance.

## C1 — Frozen ROS/deploy compatibility finding

After G7/G9, feed equivalent synthetic state through frozen deploy observation/action
logic read-only. Report exactly one of:

- `MATCH` with numeric residuals;
- `MISMATCH` with the differing frame/fact and downstream risk;
- `BLOCKED_ON_IMU_GROUND_TRUTH` with the missing measurement.

C1 does not force the simulator to match unvalidated deploy transforms and does not edit
ROS. The finding is carried into P4 read-only contract comparison and P7 risk review.

## Diagnostic implementation shape

After design approval, prefer small scripts sharing one manifest/output helper:

```text
scripts/diagnostics/audit_stage_asset.py
scripts/diagnostics/probe_gravity.py --case canonical|robot --damping <case>
scripts/diagnostics/probe_contact_settle.py
scripts/diagnostics/probe_frames_rewards.py
scripts/diagnostics/probe_actuators.py
scripts/diagnostics/probe_timing_determinism.py
```

Graduate stable invariants into explicit Isaac tests under `tests/sim/validation/` after
P1. P0 reclassifies the existing root velocity scripts but does not treat their current
hard-coded paths/parameters as validated probes.

## Evidence bundle

Tracked summary: `reboot_plan/evidence/sim_validation/RESULTS.md`.

Raw bundle under `artifacts/reboot/<run-id>/` includes manifest/resolved config,
stage/axis/asset dump, mass/density/inertia tables, gravity/contact/frame/actuator/timing
traces, fitted summaries with thresholds, plots, and relevant overlays. Every tracked
result links artifact hashes.

## Localization guide

```text
G1/G2 fail                -> stage/world/simulator gravity
G2 passes; G3/G4/G5 fail -> asset transforms, body flags, mass/inertia, damping/forces
G5 passes; G6 fails       -> collision/contact/rest control
G3 passes; G7 fails       -> root/policy/observation frame
G3/G4 pass; G8 fails      -> action decoder/actuator model
G7/G8 pass; G9 fails      -> reward/termination/command assumption
G10 fails                 -> timing, freshness, substep, or reproducibility
C1 mismatch               -> frozen deploy compatibility risk, not a simulator rewrite
all G0–G10 pass           -> legacy simulator may become the P2 oracle
```

The user can add the original visual symptom later to select traces/video for inspection;
it does not bypass this dependency graph.
