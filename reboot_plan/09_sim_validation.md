# 09 — Simulation Validation Ladder (is the sim working as intended?)

The migration plan (03) proves **code fidelity**: new code == old code, step for step.
This document proves **model fidelity**: the simulation == the intended physical robot.
Parity tests cannot catch a wrong mass, a too-soft actuator, or a flipped frame — they
would faithfully preserve them. This ladder checks the sim itself, level by level, from
static asset facts up to whole-robot dynamics and cross-simulator/hardware comparison.

**Known suspects going in** (from the 2026-07 review — expect red results here):
masses via `density=2500` with comments claiming UPE ≈ 940 kg/m³ (#9); main-drive
velocity tracking extremely soft (damping=1 vs effort 15) (#10); no contact sensors —
all stance logic is phase-proxy (#8); `linear_damping=0.05`/`angular_damping=0.10` as
fake drag on every body incl. the base (#12); 90°-about-X USD frame convention (#32).

## 0. How the ladder works

- Every check = a **script** (under `scripts/diagnostics/` or `tests/sim/validation/`)
  + a **written pass criterion** + an **evidence snapshot**.
- Results live in `experiments/reports/sim_validation/RESULTS.md`: one row per check,
  status ✅ pass / ❌ fail / ⚠️ pass-with-caveat / ⏸ blocked-on-ground-truth, with a link
  to the evidence (log, plot, printed table).
- Checks that are cheap invariants (rates, frames, limits, slices) graduate into
  **permanent tests** (`-m isaac` tier, run in `make preflight`). One-time
  characterizations (step responses, incline slip) stay as diagnostics with archived
  reports, re-run after any physics change.
- Pass thresholds below are **starting points**: tightening is free; loosening one
  after a first run requires a one-line justification in RESULTS.md.
- The ladder is **read-only on the sim code**. It *finds* problems; it does not fix
  them inline. Every ❌ becomes a triaged item in the Phase 4.3 physics backlog, fixed
  one at a time with an ADR + retrain validation (03 standing rule 3). Exception — the
  escape hatch: a *catastrophic* L0/L1 finding (e.g. total mass off by >2×, wrong units)
  is worth fixing **before** Phase 0.3's reference training run, because a baseline
  trained on absurd physics is a weak oracle. That call is the human's (07 §4).

## 1. Ground truth to collect first (without this, sim-vs-sim is circular)

Fill `docs/sim_facts.md` (template: `templates/sim_facts.md`). Every entry has a
**source**: measured / datasheet / CAD / assumed (assumed entries are flagged in every
check that depends on them).

| Fact | How to get it | Blocks |
|---|---|---|
| Total robot mass (± 0.05 kg) | weigh the robot (with battery, as deployed) | L0.2 |
| Per-module masses (leg, battery, chassis) if feasible | scale, CAD BOM | L0.2, L0.3 |
| Main-drive motor: stall torque, no-load speed, gear ratio, rated V/I | datasheet | L2.1–L2.3 |
| ABAD actuator specs (torque, speed, range) | datasheet | L2.4 |
| Leg/wheg geometry (radius, shape) | CAD | L4.1 |
| Leg-tip ↔ lab-floor friction estimate | incline slip test with the real robot or leg module | L4.3 |
| IMU mounting orientation on chassis | photo + CAD | L5, Phase 5.4 |
| Battery voltage under load | multimeter during stall test | L2.2 |
| Material density actually used (UPE?) | CAD/BOM | L0.2 |

## 2. The ladder

### L0 — Model & asset integrity (USD/URDF inspection; minutes)
| # | Check | Pass criterion |
|---|---|---|
| L0.1 | Joint inventory: names, order, axes, limits — USD == `contract.py` == URDF source (`assets/robot_description`) | exact match (script dumps all three, diffs) |
| L0.2 | Mass audit: per-link masses *after* density application, printed + summed | total within ±5% of weighed mass; per-module within ±15% where known |
| L0.3 | Inertia sanity: principal moments > 0, ratios < 10³, CoM inside body envelope | all links pass |
| L0.4 | Collision geometry: dump/visualize collision shapes | leg contact surfaces exist; no oversized phantom colliders; counts match expectation |
| L0.5 | Joint limits vs hardware spec table | sim limits ⊆ hardware limits (sim never commands past hardware) |
| L0.6 | Scale/units: bounding-box dims vs CAD | within ±2% (catches mm-vs-m class errors) |

### L1 — Static behavior (GPU; zero-action agent; minutes)
| # | Check | Pass criterion |
|---|---|---|
| L1.1 | Drop-settle: spawn at init height, zero actions, 3 s | settles < 1 s; base-height std < 1 mm afterwards; |roll|,|pitch| < 2°; no NaN |
| L1.2 | Ground penetration at rest | max penetration < 3 mm |
| L1.3 | Rest projected gravity: record obs[6:9] at rest | matches analytic value for the init rotation ± 0.02; **snapshot feeds `expected_rest_projected_gravity` in `redrhex_policy.yaml`** (closes the loop on review #32) |
| L1.4 | Phantom drift: base xy over 5 s at rest | < 1 cm |
| L1.5 | Holding torques at rest | each joint |τ| < 30% of effort limit (plausible static load) |

### L2 — Actuator & joint level (GPU; single-robot scripted commands)
| # | Check | Pass criterion |
|---|---|---|
| L2.1 | Main-drive velocity step (e.g. 0→5 rad/s): rise time, steady-state error, applied torque trace | SSE and rise time within a band derived from the intended actuator model; **quantifies review #10 softness** — record even if "pass" |
| L2.2 | Torque saturation: command beyond capability | τ clamps at effort limit; with DCMotor model, saturation follows the datasheet torque-speed line ± 15% |
| L2.3 | Velocity limit enforcement | never exceeded by > 2% |
| L2.4 | ABAD position step: rise, overshoot, SSE; range sweep | overshoot < 15%; hard stop exactly at cfg limits; limits == contract |
| L2.5 | Gravity-load hold: leg commanded to hold horizontal | holding torque ≈ analytic m·g·r ± 25% |
| L2.6 | Formalize `test_joint_velocity*.py` into `scripts/diagnostics/` with the above criteria | scripts run headless, emit pass/fail |

### L3 — Timing & integration integrity (GPU; mostly graduates to permanent tests)
| # | Check | Pass criterion |
|---|---|---|
| L3.1 | Rates: sim.dt = 1/120, decimation = 2 → control 60 Hz; episode length in seconds correct | asserted against `contract.py` (permanent test) |
| L3.2 | Action targets written exactly once per control step | counted via instrumentation (regression for the July substep bug — permanent) |
| L3.3 | Gait phase advances 2π·f per sim-second; FSM timers/cooldowns run in real sim time | analytic check over 10 s rollout (permanent) |
| L3.4 | Determinism envelope: same seed, 5 runs | spread recorded → pins golden tolerances (05 §3) |
| L3.5 | Reward/state freshness: which step's kinematics rewards consume | documented now; after step 2.5, rewards use current-step state (permanent) |

### L4 — Whole-robot dynamics (GPU; scripted open-loop, NO policy)
| # | Check | Pass criterion |
|---|---|---|
| L4.1 | Scripted tripod gait (fixed CPG, no learning): walk 10 m | stable tripod; forward speed consistent with wheg-geometry × rotation-rate expectation ± 25%; heading drift < 10°/10 m |
| L4.2 | Push recovery: standing robot, calibrated lateral impulse | survives small push, falls under large one — thresholds recorded (baseline for policy comparisons later) |
| L4.3 | Incline slip: static robot on ramp, sweep angle | slips at atan(μ) ± 5° vs measured friction |
| L4.4 | Energetics during scripted gait: Σ|τ·ω| | ≤ total rated motor power; cost of transport in RHex-literature range (~0.5–3) — record the number |
| L4.5 | 10 cm drop test | lands without solver explosion; peak forces finite; settles < 1 s |
| L4.6 | Fake-drag quantification: base free-fall / swing with body damping on vs off | effect measured and documented (input to the Phase-4.3 decision on review #12-sim) |

### L5 — Observation & frame truth (GPU; teleport-based analytic checks; graduates to permanent)
| # | Check | Pass criterion |
|---|---|---|
| L5.1 | Set base to known quaternions (0°, ±90° roll/pitch/yaw) → projected-gravity obs | matches closed-form ± 0.02 for every pose |
| L5.2 | Constant forward motion → base_lin_vel obs | +x in body frame regardless of world heading |
| L5.3 | Spin about +z → base_ang_vel obs | sign and magnitude match |
| L5.4 | Obs slice layout vs contract | already tier-2 (05) — cross-referenced here for completeness |
| L5.5 | Command frame: velocity command executed in the intended (body/world) frame | scripted check documents the convention |
| L5.6 | ROS `observation_builder` fed a synthetic IMU quat equal to the sim rest attitude → identical obs vector as sim | bitwise/1e-6 match (extends the existing golden-obs deploy readiness) |

### L6 — Cross-validation against external references
| # | Check | Pass criterion |
|---|---|---|
| L6.1 | MuJoCo cross-sim (existing `mujoco_rollout.py`): same scripted L4.1 gait | both stable; forward speed within 20% of each other; disagreement = finding, not failure |
| L6.2 | Hardware open-loop: same scripted gait on the real robot (bench, then floor) — **human-present** (Phase 5) | speed, current draw, gait video vs sim; residuals recorded |
| L6.3 | Sim2real residual table maintained as hardware data arrives | every known gap listed with size + mitigation status |

## 3. Scheduling (how this weaves into 03)

| Sub-phase | What | When |
|---|---|---|
| **V0 — build** | Write the ladder scripts + `docs/sim_facts.md` skeleton + RESULTS.md skeleton | during Phase 1 (agent work, CPU-heavy, GPU-light) |
| **V0.5 — early screen** | Run L0 + L1 + L3.1/3.4 only | **before Phase 0.3's reference training run** — this is where the escape hatch applies |
| **V1 — full run** | Execute L0–L5 completely, triage all findings in RESULTS.md | parallel to Phase 2 (read-only, so it can't conflict with extractions; respect one-GPU discipline 04 §4) |
| **V2 — fix** | ❌ items fixed one at a time: ADR + fix + re-run affected ladder level + 3-seed retrain check | = Phase 4.3, now *driven by RESULTS.md* instead of hand-picked |
| **V3 — hardware** | L6.2/L6.3 + IMU capture | Phase 5.4/5.5 |
| **Standing** | Graduated permanent checks run in `make preflight`; full ladder re-run after ANY physics/asset/actuator change and before any hardware session | forever |

Ordering note: V0.5 exists because Phase 0's reference run is only a useful oracle if
the physics under it isn't absurd. L0+L1 take under an hour and de-risk the whole
baseline investment.

## 4. Deliverables checklist

- [ ] `docs/sim_facts.md` filled, every entry sourced (measured/datasheet/CAD/assumed)
- [ ] `scripts/diagnostics/` ladder scripts, each emitting machine-readable pass/fail
- [ ] `experiments/reports/sim_validation/RESULTS.md` — full L0–L5 run recorded
- [ ] Permanent subset wired into `tests/sim/validation/` + `make preflight`
- [ ] Rest-attitude snapshot (L1.3) written into `redrhex_policy.yaml`
- [ ] Phase-4.3 backlog = triaged ❌/⚠️ list, priority-ordered by expected sim2real impact
- [ ] Ladder re-run recorded after each physics fix (evidence chain per ADR)
