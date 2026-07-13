# 03 — Migration Plan (the actual to-do list)

Strangler-fig migration: the old env keeps working at every step; new modules replace
organs one at a time behind parity gates. Every step = one or a few commits, each
independently revertible. **Never two extraction steps in flight at once.**

Effort estimates assume AI-heavy development (agent implements, harness verifies,
human reviews + judges) with one shared GPU machine.

---

## Phase 0 — Freeze & Baseline (½–1 day) — DO THIS BEFORE ANY RESTRUCTURING

| # | Step | Verification gate |
|---|---|---|
| 0.1 | Merge `fix/review-2026-07` → `main`; tag `v0-pre-reboot` | CI/tests on branch pass; tag pushed |
| 0.2 | **Golden rollout dump**: script `scripts/diagnostics/dump_golden_rollout.py` — fixed seed, 4 envs, ~500 steps on current env; save obs/actions/rewards/dones/joint-states per step to `baselines/golden_v0/*.pt` + manifest (commit hash, task, cfg snapshot) | Dump loads; re-running the script byte-reproduces it (determinism check — if nondeterministic, record tolerance now) |
| 0.3 | **Reference training run**: ~300 iters, current best config, seed 42 (3 seeds if GPU time allows); export TB scalars to `baselines/ref_run_v0/` | Learning curve sane vs historical runs |
| 0.4 | Copy best checkpoint(s) + eval metrics into `baselines/checkpoints/` with manifest | play.py runs them |
| 0.5 | Hygiene commit: create `assets/`, `attic/`, `tests/`, `experiments/`, `baselines/` (gitignored); move `RedRhex.usd` + `test_7_description/` → `assets/` (fix cfg paths); patches + skrl → `attic/`; `test_joint_velocity*` → `scripts/diagnostics/`; untrack `.vscode/*.db*` | Old task still trains 5 iters after the moves (`make smoke`) |

**Exit:** tag exists, golden dump exists, reference curve exists, tree is clean.
*Nothing below this line starts until Phase 0 is done.*

---

## Phase 1 — Scaffolding (2–3 days)

| # | Step | Verification gate |
|---|---|---|
| 1.1 | `Makefile` with the canonical targets (templates/Makefile): `lint`, `test-fast`, `test-contract`, `test-golden`, `smoke`, `preflight`, `contract`, `train-ref` | Every target runs on this machine |
| 1.2 | `pyproject.toml`: pytest markers (`fast`, `isaac`, `golden`), ruff (replaces flake8/black — one tool), pre-commit update | `make lint` clean (allow initial noqa waves in old files) |
| 1.3 | `tests/` skeleton + port existing tests (panel tests, reward_agent tests, contract parity test) into the tiered layout | `make test-fast && make test-contract` green |
| 1.4 | CI: `.github/workflows/ci.yml` running lint + fast + contract tiers on CPU (no GPU in CI — see 05 §4) | CI green on a trial PR |
| 1.5 | CLAUDE.md hierarchy from templates: root + `source/RedRhex/` + `tools/training_panel/` + `ros2_ws/` | Fresh AI session can run `make smoke` from CLAUDE.md alone, zero human hints |
| 1.6 | `docs/INDEX.md` + `docs/adr/0001-soft-reboot.md` (records this plan as accepted) | — |

**Exit:** a fresh clone + fresh AI session reaches green `make test-fast` and `make smoke` unaided.

---

## Phase V — Simulation validation ladder (parallel track; full spec in 09)

Parity (Phase 2) proves new code == old code; this track proves the sim == the intended
robot. It is **read-only on sim code**, so it runs alongside the other phases without
conflicting — findings are triaged into Phase 4.3, not fixed inline.

| # | Sub-phase | When | Gate |
|---|---|---|---|
| V0 | Build ladder scripts + `docs/sim_facts.md` (ground truth, sourced) + RESULTS.md skeleton | during Phase 1 | scripts run headless, emit pass/fail |
| V0.5 | **Early screen: L0 (asset/mass/joints) + L1 (static) + L3.1/3.4 (rates/determinism)** | **before Phase 0.3's reference run** | no catastrophic finding — else escape hatch: human decides to fix pre-baseline |
| V1 | Full L0–L5 run; every check ✅/❌/⚠️/⏸ recorded with evidence | parallel to Phase 2 | RESULTS.md complete; ❌ items triaged into Phase 4.3 backlog |
| V2 | Fix ❌ items one at a time (ADR + re-run affected level + 3-seed retrain check) | = Phase 4.3 | ladder level green after each fix |
| V3 | Hardware cross-checks (L6.2/L6.3, IMU capture) | Phase 5.4/5.5 | residual table current |

Standing: graduated invariant checks (rates, frames, limits, action-write count) join
`make preflight`; the full ladder re-runs after ANY physics/asset change and before any
hardware session.

---

## Phase 2 — Strangler extraction of the env (1.5–2.5 weeks; the core of the reboot)

Per-module recipe (repeat 6×):

```
a. Agent reads the relevant env region, writes core/<module>.py as pure functions
   (same math, no isaaclab imports) + unit tests with hand-computable cases.
b. Golden parity test: feed recorded inputs from baselines/golden_v0 through the new
   module; outputs must match recorded outputs (atol from 0.2 determinism check).
c. Env class switched to call the module; dead in-env code deleted in the same commit.
d. make test-fast && make test-golden && make smoke  → commit ("refactor(core): extract X").
e. /code-review on the diff before merge.
```

Extraction order (dependency-driven, easiest first):

| # | Module | Notes / traps |
|---|---|---|
| 2.1 | `core/kinematics.py` | quat/gravity helpers; trivially testable |
| 2.2 | `core/buffers.py` | batched episode-sums `(num_envs, n_terms)`; fixes review #13 perf + #19 per-second logging in one move — **behavior-changing for logging only**: declare it, don't parity-gate the log values |
| 2.3 | `core/observations.py` | layout slices imported from `contract.py`; obs-noise slices get direct unit tests (July bug class) |
| 2.4 | `core/gait.py` | GaitState dataclass; dt passed once per control step — the July substep bug becomes unrepresentable; FSM gets a pure-CPU simulation test over synthetic command sequences |
| 2.5 | `core/commands.py` | move resampling/push OUT of `_get_observations` (review #15). **Behavior-changing** (1-step timing shift): ADR + 3-seed short-run comparison vs reference curve instead of step parity |
| 2.6 | `core/rewards.py` | simplified path only; per-term unit tests; resolve review #12 (diag-sign double count) HERE with an explicit human decision recorded in the ADR |
| 2.7 | `core/domain_rand.py` | samplers pure; slice indices tested against contract layout |
| 2.8 | **Delete legacy full-reward path** (~1,000 lines) + `redrhex_symmetry.py` | `use_simplified_rewards` flag removed; recovery = `v0-pre-reboot` tag |
| 2.9 | Cfg modularization: grouped configclasses + `validate()`; kill alias/deprecated fields; stages as overlays | Old flat names kept as a thin compat shim for panel presets until Phase 5; validation test tier |
| 2.10 | Consolidate 4 PPO cfgs → base + variants; extract `scripts/common/checkpoint_utils.py` (3 copies → 1) | train/play/eval all resolve the same checkpoint in a test |

**Exit criteria:** env < 800 lines; `make test-fast` < 30 s covering all core modules;
full golden parity green (or ADR'd exceptions: 2.2 logging, 2.5 timing, 2.6 decision);
**Phase-2 gate run:** 300-iter training, 3 seeds, curve within noise band of
`baselines/ref_run_v0`.

---

## Phase 3 — Contract unification & deploy hardening (3–5 days)

| # | Step | Verification gate |
|---|---|---|
| 3.1 | `contract.py` as importable single source (env cfg + obs layout consume it) | golden + smoke still green |
| 3.2 | `scripts/gen_contract.py` → generated `ros2_ws/.../redrhex_contract.py` with DO-NOT-EDIT header + source hash | `make contract` idempotent; "generated file up to date" test in CI |
| 3.3 | Extend `deploy.py validate_contract`: rates, scales, limits, `CONTRACT_VERSION` (closes the 125 Hz bug class permanently — review #31/#33) | Deliberately corrupt a constant → validation fails |
| 3.4 | ONNX equivalence test: export policy, compare onnxruntime vs torch outputs on golden obs (atol 1e-5) | test in `make preflight` |
| 3.5 | Checkpoint metadata carries contract version + git hash; deploy refuses mismatches | unit test |

**Exit:** hand-editing the ROS2 contract is impossible without CI screaming.

---

## Phase 4 — Research features on the clean base (ongoing; now cheap)

Priority-ordered backlog (each item = experiment overlay + report per 06):

| # | Item | Origin |
|---|---|---|
| 4.1 | Contact sensors: author contact-reporter API into the USD (`assets/`), instantiate ContactSensor, replace phase-proxy stance rewards; A/B vs proxy | review #8 — biggest sim-fidelity gap |
| 4.2 | base_lin_vel at deploy: lin-vel obs dropout during training and/or simple estimator node; pick via eval | review #34 |
| 4.3 | Physics fidelity pass, **driven by the Phase-V RESULTS.md triage** (09): explicit per-link masses (drop density=2500 hack), main-drive actuator damping/effort revisit, remove fake body drag, plus whatever else the ladder flags — one at a time, each with ADR + ladder re-run + 3-seed validation run | review #9/#10/#12(sim) + 09 |
| 4.4 | Teacher–student pipeline productionized on new structure (midterm work: privileged teacher + distillation smoke-validated) | 2026_Midterm.md |
| 4.5 | Terrain curriculum consolidation + `validate_reform_stack`-style checks into `tests/sim/` | midterm |
| 4.6 | Correct morphological symmetry (if pursued): derive the true symmetry group of the tripod grouping first; ADR before code | review #20 |
| 4.7 | MPC comparison harness: scripted disturbance/efficiency benchmark scenarios, both controllers, auto-report (the thesis claim generator) | strategy doc §6 |

---

## Phase 5 — Panel & hardware readiness (interleave with Phase 4)

| # | Step | Gate |
|---|---|---|
| 5.1 | Panel: per-run explicit config files replace `active_*_override.json` IPC; overlays from `cfg/experiments/` | queued runs can't race (test with 2 queued runs) |
| 5.2 | Panel: minimal token auth + bind-localhost default | manual check + doc |
| 5.3 | Panel perf/semantics: history poll caching, convergence window fix | existing panel tests extended |
| 5.4 | Hardware IMU capture: record rest `projected_gravity` on the real robot → fill `imu_mount_rpy_deg` / `expected_rest_projected_gravity` in `redrhex_policy.yaml` | rest-attitude gate passes on hardware; **human-present step** |
| 5.5 | HIL dry run: preflight + safety filter + zero-policy, then baseline policy, legs off ground first | hardware checklist (human-gated, see 04 §5) |

---

## Standing rules during the whole migration

1. Training experiments may continue on `main` between phases — but not *during* an
   extraction step (keep the golden oracle meaningful).
2. Any test that flakes gets fixed or quarantined the same day; a red-but-ignored suite
   destroys the AI workflow (the agent must be able to trust green).
3. Every behavior-changing step needs: an ADR line, a baseline update, and a validation
   run. No silent drift — that is the disease this reboot cures.
4. If a phase stalls > 1 week, cut scope (e.g. skip 2.7's purity, keep DR in env),
   don't extend the freeze on panel/features indefinitely.
