# 05 — Testing & CI

## 1. Test tiers

| Tier | Marker / dir | Runs on | Time | Contents |
|---|---|---|---|---|
| 0 | `make lint` | CPU, CI | s | ruff (format+lint, replaces black/flake8), pre-commit |
| 1 | `tests/unit` `-m fast` | CPU, CI | < 30 s | pure-torch core modules: reward terms vs hand-computed values, obs assembly/normalization/noise slices, gait FSM over synthetic sequences, DR samplers (bounds/shapes/dtypes), cfg `validate()` (bad cfgs must raise), checkpoint_utils |
| 2 | `tests/contract` | CPU, CI | s | generated-ROS2-contract freshness (regenerate+diff), AST parity env↔contract (July test), obs-layout slice invariants, action-decoder vs env-gating equivalence on input grids, panel/deploy `validate_contract` unit tests, checkpoint-metadata version checks |
| 3 | `tests/golden` `-m golden` | CPU, CI (uses saved tensors from `baselines/golden_v0`) | < 1 min | each core module replayed on recorded inputs → must match recorded outputs; full obs/reward/done step-parity replay |
| 4 | `tests/sim` `-m isaac` | GPU + Isaac Sim, **local only** | 3–15 min | env construction, reset/step, obs dims vs contract, determinism (same seed twice → same trajectory within tol), 5-iter smoke train (loss finite, no NaN), zero/random agent sanity, terrain generator instantiation |
| 5 | deploy readiness | GPU/CPU, local, pre-hardware | min | ONNX-vs-torch equivalence on golden obs (atol 1e-5), MuJoCo cross-sim rollout sanity (`mujoco_rollout.py`), preflight + safety-filter unit tests, rest-attitude gate |

Existing tests get ported into this layout in Phase 1.3 (panel tests, reward_agent
tests, `test_contract_parity.py`).

**Simulation-truth checks** (the ladder in 09) sit alongside tier 4: one-time
characterizations (actuator step responses, incline slip, energetics) live in
`scripts/diagnostics/` with archived reports; the invariant subset (rates, frames,
limits, action-write count, obs-frame analytic checks) graduates into
`tests/sim/validation/` and runs in `make preflight`. Tier 4 answers "does the code
run the model correctly"; the ladder answers "is the model the real robot".

## 2. What each bug-class from the 2026-07 review maps to

| Bug class (review #) | Now caught by |
|---|---|
| Rate/constant drift sim↔deploy (#31, #33) | Tier 2 freshness + version checks; extended `validate_contract` |
| Obs slice/noise index errors (#2) | Tier 1 slice tests + Tier 3 replay |
| Per-substep time accumulation (#1) | Structural (dt passed once) + Tier 1 FSM timing test |
| Double-counted reward terms (#3, #5) | Tier 1 per-term tests + `total_reward` decomposition test (sum of parts = total) |
| Cfg typos / dead attrs (#4, #6) | cfg `validate()` + ruff + Tier 1 construction test |
| Frame convention (#32) | Tier 5 rest-attitude gate + hardware capture (Phase 5.4) |
| History-store races (#26) | ported panel concurrency test |
| Silent global overrides (#22, #27) | `--panel_overrides` gating test; Phase-5 per-run configs |

## 3. Determinism policy

Measured once in Phase 0.2 and pinned in `tests/conftest.py`:
- If the current env is bitwise-deterministic under fixed seed → golden tests use exact
  compare for ints/bools, `atol=0` floats.
- If not (PhysX/GPU nondeterminism) → record the observed spread over 5 runs, set
  tolerance = 10× spread, and document it. **Tolerances are frozen constants; widening
  one requires an ADR** (see 04 §5.2 — this is the anti-"AI loosens the test" rule).
- Golden tiers 1–3 avoid the problem entirely where possible: they replay *recorded*
  inputs through pure functions, so sim nondeterminism never enters.

## 4. CI reality: one lab GPU, no cloud GPU

- **GitHub Actions (every push/PR):** tiers 0–3 only. Torch CPU wheel; `isaaclab` never
  imported (enforced by a tier-1 test that fails if any `core/` module imports it).
  Total budget < 5 min.
- **`make preflight` (local, GPU):** tiers 0–5 minus long runs. Required before merging
  to `main` — enforced socially + by PR checklist (templates/pr_checklist.md), since
  there's no GPU runner to enforce it mechanically. If the machine is ever idle enough,
  a self-hosted runner label `gpu` can make it mechanical — optional, not planned.
- **Nightly (optional, cron/panel queue):** `make preflight` + 100-iter smoke train with
  metric bounds (reward at iter 100 within band of reference). Catches slow drift.

## 5. Phase gates (from 03) as test events

| Gate | Test evidence required |
|---|---|
| Phase 0 exit | golden dump reproduces; reference curve archived; **V0.5 early screen (09): L0+L1+rates run before the reference training run** |
| Each extraction (2.x) | tiers 0–3 green + smoke; behavior-changing steps: ADR + 3-seed 300-iter curve within noise band of `ref_run_v0` |
| Phase 2 exit | full parity suite green; env < 800 lines; `make test-fast` < 30 s |
| Phase 3 exit | corrupt-a-constant drill: mutate any contract constant → at least one tier-2 test fails |
| Pre-hardware (5.5) | tier 5 all green + human checklist |

## 6. Conventions

- Test names state the invariant: `test_obs_noise_touches_only_configured_slices`, not
  `test_noise2`.
- Every regression fix adds the test that would have caught it, in the cheapest tier
  that can express it (July's contract-parity test is the model).
- No test sleeps, no test depends on `logs/` contents, no network in tiers 0–3.
- Flaky = quarantined same day with an issue note in the phase checklist; a suite the
  agent can't trust is worse than no suite (04 standing rule).
