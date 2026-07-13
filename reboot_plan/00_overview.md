# 00 — Overview: Why, What, and Definition of Done

## 1. Why a soft reboot now

The project has succeeded at its research goals faster than its structure could keep up:

- **The core env is a 4,478-line monolith** (`redrhex_env.py`) mixing a legacy ~50-term
  reward path (dead by default), the simplified reward path, the lateral FSM/CPG, domain
  randomization, visualization, and buffer management. The cfg is 1,806 lines with ~250
  scalars including deprecated fields and alias pairs that silently desync.
- **Cross-boundary constants drift.** The sim↔ROS2 deploy contract was hand-mirrored and
  drifted (125 Hz vs 60 Hz policy rate, ABAD scale/limits) — the single most dangerous
  class of bug found in the 2026-07 review. A parity test now exists, but the structure
  still invites drift.
- **Verification is slow and manual.** Almost nothing can be tested without launching
  Isaac Sim on the GPU. That is the #1 bottleneck for AI-heavy development: an agent
  that must wait minutes (or a human eyeball) for every check is throttled to human speed.
- **The 2026-07 review found 35 issues**; the high-severity ones are fixed on
  `fix/review-2026-07`, but roughly a third were *structural* (duplication, dead code,
  fragile IPC via JSON files, missing validation) — symptoms that will regenerate new
  bugs unless the structure changes.

A hard rewrite would throw away validated physics knowledge, reward-shaping experience,
a working deploy stack, and months of training insight. A soft reboot keeps all of that
and replaces only the structure.

## 2. Scope

**In scope (changes):**
- Repo layout, module boundaries, config architecture inside `source/RedRhex`.
- Single-source contract generation for the ROS2 stack.
- Test harness, CI, Makefile, CLAUDE.md hierarchy, experiment management.
- Repo hygiene (stray root files, asset placement).
- Deferred review items get scheduled slots (contact sensors, lin-vel estimator,
  physics tuning, panel auth, perf).

**Out of scope (kept as-is unless a phase explicitly touches them):**
- The research direction and reward design philosophy (midterm report stands).
- Training panel feature set (frozen during migration; auth + IPC cleanup scheduled later).
- Trained checkpoints, logs, tensorboard history (kept locally, never in git).
- Isaac Lab / Isaac Sim versions (Isaac Lab 2.3.2, Isaac Sim 5.1.0-rc.19,
  conda `env_isaaclab_bin`) — do not upgrade mid-migration.

## 3. The five principles

1. **Behavior-preserving by proof, not by hope.** Every structural change is gated by a
   parity test against a frozen golden baseline (fixed-seed rollout dump). A change
   either provably preserves behavior, or is explicitly declared behavior-changing and
   gets its own baseline update + short validation run.

2. **Testable without the simulator.** Observation math, reward math, gait/CPG state
   machines, DR sampling, and command logic become pure-torch functions with no
   `isaaclab` import. The env class shrinks to orchestration. Result: the fast test
   tier runs on CPU in seconds — the precondition for AI agents to self-verify.

3. **One source of truth per fact.** Control rate, joint ordering, obs layout, action
   scaling, ABAD limits: defined once in `contract.py`, consumed by the env, exported
   to the ROS2 stack by a generator, guarded by parity tests. Same principle for
   checkpoint-resolution helpers (currently 3 drifting copies) and PPO configs
   (currently 4 near-copies).

4. **AI does the work; harnesses do the checking; humans do the judgment.** The agent
   plans, implements, tests, reviews, and documents. Humans decide reward-shaping
   intent, physics plausibility, research direction, and anything touching hardware.
   The boundary is enforced by workflow (04) and guardrails, not by trust.

5. **Small, reversible steps.** One issue per commit (this already worked well in the
   July fix branch). Extraction order chosen so each step is shippable and revertible.
   No long-lived divergent branches during migration.

## 4. Definition of done (reboot complete when…)

- [ ] `redrhex_env.py` < 800 lines; all math lives in `core/` modules with unit tests.
- [ ] `pytest -m fast` passes on CPU in < 30 s with no Isaac Sim installed.
- [ ] Golden-parity suite proves the restructured env is step-for-step identical to the
      pre-reboot env for a fixed seed (or every intentional difference is documented in
      an ADR with a validation run).
- [ ] ROS2 `redrhex_contract.py` is generated, not hand-written; `make contract` +
      parity test in CI; `deploy.py validate_contract` checks rates and scales.
- [ ] Config is layered (base/stage/experiment); no alias fields; stage-list lengths and
      cross-field constraints validated at construction.
- [ ] Legacy full-reward path deleted (recoverable via git tag `v0-pre-reboot`).
- [ ] CI (CPU tiers) green on every PR; `make preflight` (GPU tiers) documented and
      required before merge to main.
- [ ] CLAUDE.md hierarchy in place; a fresh AI session can run smoke train, tests, and
      a standard experiment without human path-hunting.
- [ ] A reference training run on the rebooted code matches the frozen baseline run's
      learning curve within noise (3-seed check).
- [ ] Simulation validation ladder (09) executed L0–L5 with evidence in
      `experiments/reports/sim_validation/RESULTS.md`; every ❌ finding either fixed
      (ADR + ladder re-run + retrain check) or explicitly accepted with a reason; the
      invariant subset runs permanently in `make preflight`.

## 5. What "smoothest and fastest development" concretely means after the reboot

| Task | Before | After |
|---|---|---|
| Verify a reward-math change | Launch Isaac Sim, watch robot / read TB | `pytest tests/unit/test_rewards.py` in ~2 s, then one smoke train |
| Change ABAD limit | Edit cfg + remember to edit ROS2 contract + panel | Edit `contract.py`; `make contract` regenerates everything; CI enforces |
| Add an experiment variant | Edit the 1,806-line cfg in place, lose the old values | Drop a 20-line experiment overlay file; both versions coexist |
| AI agent works overnight | Not safe — no way to self-check | Agent runs `make preflight`, files report, commits per issue |
| Find why a run regressed | Diff memory + guesswork | Diff two checked-in experiment overlays + decision log |
