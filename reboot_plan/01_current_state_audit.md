# 01 — Current State Audit: Keep / Refactor / Drop

Snapshot: 2026-07-13, branch `fix/review-2026-07` (clean). 249 tracked files.
Full defect list: `docs/project_review_2026-07-09.md` (35 findings; high-severity items
fixed on this branch, deferred items listed in its "Fix status" section).

## 1. Component inventory and verdicts

### Core RL environment — `source/RedRhex/RedRhex/tasks/direct/redrhex/`
| File | Size | Verdict |
|---|---|---|
| `redrhex_env.py` | 4,478 lines | **REFACTOR (strangler-fig)** — the crown jewel behaviorally; structurally the biggest liability. Contains: obs building, simplified rewards, dead legacy ~50-term reward path (~1,000 lines), lateral FSM/CPG, DR, command resampling, push logic, visualization, buffers. |
| `redrhex_env_cfg.py` | 1,806 lines | **REFACTOR** — ~250 flat scalars, deprecated fields, alias pairs (`randomize_mass = dr_randomize_mass`) that desync under Hydra overrides, unvalidated length-5 stage lists. Replace with grouped configclasses + validation. |
| `redrhex_symmetry.py` | 119 lines | **DROP** — mirror augmentation is physically wrong for the non-mirror-symmetric tripod grouping (review #20); already disabled everywhere. Delete; a *correct* morphological-symmetry approach is a Phase-4 research item, started fresh. |
| `agents/` (4 PPO cfgs) | ~95% copy-paste | **CONSOLIDATE** into one base cfg + small variants. |

Known live issues to carry into the plan (deferred from review): contact sensors never
instantiated (all contact rewards use phase proxies), density-based mass instead of
explicit per-link masses, soft main-drive velocity tracking (damping=1 vs effort 15),
body-level linear/angular damping acting as fake drag, reward-on-stale-state (state
mutation inside `_get_observations`), episode_sums = ~140 per-step kernel launches.

### Training scripts — `scripts/`
| Item | Verdict |
|---|---|
| `rsl_rl/train.py`, `play.py`, eval scripts | **KEEP + DEDUPE** — checkpoint-resolution helpers exist in 3 drifting copies; extract to one shared module. `--panel_overrides` gating (fixed in July) stays. |
| `skrl/` scripts | **PARK** — not the active framework; move to `attic/` (git-preserved) unless actively used. |
| `list_envs.py`, `zero_agent.py`, `random_agent.py` | **KEEP** — cheap, useful smoke tools. |

### Training panel — `tools/training_panel/` (~24 backend modules + static/remote web + supabase)
**KEEP, FREEZE during migration.** It works and is valuable (queueing, presets, history,
convergence detection, MuJoCo rollout, remote worker). Scheduled cleanups (Phase 5):
- Replace JSON-file IPC (`active_*_override.json`) with per-run config files passed
  explicitly (races documented in review #27).
- Minimal token auth before any non-localhost exposure (review #28).
- `reconcile_stale_history()` on every poll → cache (review #30).
- Convergence detector window semantics (review #29).
The AST-based `tests/test_contract_parity.py` (added July) is a keeper and becomes part
of the contract pipeline (02 §4).

### Reward agent — `tools/reward_agent/`
**KEEP.** AI-driven reward search (agent/planner/evaluator/experiment_store) aligns
exactly with the AI-heavy direction; integrate with the new experiment management (06)
instead of its private store once the migration lands.

### ROS2 deploy stack — `ros2_ws/src/`
| Item | Verdict |
|---|---|
| `redrhex_rl_controller` (contract, obs builder, action decoder, ONNX runner, safety filter, preflight) | **KEEP — architecture is good** (review #35 called it out as a strength). Change: `redrhex_contract.py` becomes a *generated* file (02 §4). Extend `validate_contract` to check rates/scales, not just dims/names. |
| `redrhex_lowlevel_bridge`, `redrhex_msgs` | **KEEP** as-is. |
| Open hardware items | IMU mount values (`imu_mount_rpy_deg`, `expected_rest_projected_gravity`) still need hardware capture; `base_lin_vel_source: zero` needs estimator or lin-vel-dropout training (Phase 4/5). |

### Robot assets
| Item | Verdict |
|---|---|
| `RedRhex.usd` (repo root) | **MOVE** to `assets/` — root is not a place for a binary asset. Update the cfg USD path (and delete the stale `/home/jasonliao` comment lineage). |
| `test_7_description/` (URDF + meshes + converted USDs, 45 tracked files) | **MOVE + RENAME** to `assets/robot_description/` — it is the robot's source description, currently named like a scratch folder. Also reconcile with the copy in `~/Downloads/test_7_description`. |

### Repo hygiene (root strays)
| Item | Verdict |
|---|---|
| `curriculum_dr.patch`, `redrhex_multiskill_fix.patch` | **ARCHIVE** to `attic/patches/` with a one-line note each, or delete (git history has them). |
| `test_joint_velocity.py`, `test_joint_velocity2.py` | **MOVE** to `scripts/diagnostics/` or delete. |
| `MUJOCO_LOG.TXT` | Already gitignored (July); delete local copy. |
| `.vscode/browse.vc.db-*` | **UNTRACK** — IDE database files don't belong in git. |
| `logs/`, `outputs/` | Already untracked ✓ — keep it that way. |

### Docs — `docs/`
**KEEP ALL**, add `docs/INDEX.md`. The literature/strategy reports
(`redrhex_improvement_strategy_full.md`, `2026_Midterm.md`) are the research spine.
The review file stays as the historical defect record. New structural docs go to
`docs/adr/` (decision records) going forward.

## 2. What is genuinely good (protect during migration)

1. **The deploy-readiness mindset**: safety filter, preflight, state machine, golden-obs
   validation, MuJoCo cross-check tooling — rare discipline; the reboot builds *on* it.
2. **The simplified-reward path** — the product of long shaping iteration; its *behavior*
   is the thing parity tests protect.
3. **The panel + queue + history** — a real experiment-ops asset most labs don't have.
4. **Docs culture** — long-form reasoning docs exist; the reboot adds structure
   (index, ADRs, experiment reports), not volume.
5. **The July fix discipline** (one issue per commit, review file as state) — adopted
   as the standard workflow (08).

## 3. Baseline assets to freeze before touching anything (Phase 0 input)

- Best current checkpoint(s) under `logs/rsl_rl/redrhex_wheg/` — copy to
  `baselines/` (local, gitignored) with a manifest (task, commit, iterations, metrics).
- A fixed-seed golden rollout dump from the *current* env (obs/actions/rewards/dones
  for ~500 steps, 4 envs) — the parity oracle for the whole migration.
- A reference training run (~300 iters, 3 seeds if time allows) with TB scalars
  exported — the learning-curve oracle for "did the reboot change training?".
