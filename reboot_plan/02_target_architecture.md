# 02 — Target Architecture

## 1. Design rules (dependency law)

```
Rule 1: core/ modules are pure torch — NO isaaclab/omni imports. CPU-runnable.
Rule 2: the env class orchestrates; it computes nothing non-trivial itself.
Rule 3: contract.py is the ONLY place cross-boundary constants are written by hand.
        Everything else (env cfg, ROS2 contract, panel checks) derives or is generated.
Rule 4: configs are layered; an experiment never edits a base file.
Rule 5: anything an AI agent may edit has a test that fails when it's wrong.
```

Rule 1 is the keystone: it is what makes tests fast, which is what makes AI-heavy
development work (see 04). Isaac Lab's `DirectRLEnv` still owns sim stepping; the env
class passes tensors *into* pure functions and writes results back.

## 2. Target repo layout

```
RedRHex/
├── CLAUDE.md                      # root agent brief (template in templates/)
├── Makefile                       # ALL verification entry points (templates/Makefile)
├── pyproject.toml                 # ruff config, pytest config, markers
├── README.md                      # rewritten: 1-page quickstart, points at docs/INDEX.md
├── assets/
│   ├── RedRhex.usd
│   └── robot_description/         # ← test_7_description (URDF, meshes, converted USDs)
├── source/RedRhex/RedRhex/        # installable package (path kept — Isaac Lab template compat)
│   ├── contract.py                # ★ single source of truth (see §4)
│   ├── core/                      # ★ pure torch, no isaaclab imports
│   │   ├── observations.py        # obs assembly + normalization + layout (from contract)
│   │   ├── rewards.py             # simplified reward terms as pure functions
│   │   ├── gait.py                # CPG phase, tripod grouping, lateral FSM
│   │   ├── commands.py            # command sampling / resampling / curriculum gates
│   │   ├── domain_rand.py         # DR sampling + obs noise (correct slices, tested)
│   │   ├── kinematics.py          # frame math, projected gravity, quat helpers
│   │   └── buffers.py             # episode sums (batched!), history buffers
│   ├── tasks/direct/redrhex/
│   │   ├── redrhex_env.py         # thin orchestrator (< 800 lines): sim I/O + core calls
│   │   ├── cfg/
│   │   │   ├── base.py            # grouped configclasses: SimCfg, RobotCfg, ObsCfg,
│   │   │   │                      #   RewardCfg, CommandCfg, DRCfg, StageCfg (+validate())
│   │   │   ├── stages.py          # stage 1..5 overlays
│   │   │   └── experiments/       # 20-line experiment overlays (checked in, immutable)
│   │   └── agents/
│   │       └── rsl_rl_ppo_cfg.py  # ONE base cfg + small named variants
│   └── (gym registration keeps Template-Redrhex-Direct-v0 working; adds Redrhex-v1)
├── scripts/
│   ├── rsl_rl/                    # train/play/eval — import shared checkpoint_utils
│   ├── common/checkpoint_utils.py # ← the 3 drifting copies, unified
│   ├── diagnostics/               # test_joint_velocity*.py etc.
│   └── gen_contract.py            # contract.py → ros2_ws generated file (§4)
├── ros2_ws/src/                   # unchanged layout; redrhex_contract.py now GENERATED
├── tools/
│   ├── training_panel/            # frozen during migration; Phase-5 cleanups
│   └── reward_agent/
├── tests/                         # ★ consolidated test root (see 05)
│   ├── unit/                      # -m fast: pure CPU, no isaaclab (seconds)
│   ├── contract/                  # parity: env↔contract↔ros2↔panel (seconds)
│   ├── golden/                    # parity vs frozen baseline dump (CPU, uses saved tensors)
│   └── sim/                       # -m isaac: needs Isaac Sim + GPU (minutes)
├── baselines/                     # gitignored: golden dumps, reference checkpoints, manifests
├── experiments/                   # experiment reports (md), decision log (see 06)
├── docs/                          # existing docs + INDEX.md + adr/
├── attic/                         # parked code (skrl scripts, patches) — git-preserved, excluded from lint
└── .github/workflows/ci.yml       # CPU tiers on every push (see 05)
```

Notes:
- `source/RedRhex` path is **kept** so Isaac Lab tooling, existing panel process
  launcher, and `pip install -e` targets don't break. The reboot happens *inside* it.
- Old task id `Template-Redrhex-Direct-v0` keeps working throughout the migration
  (checkpoints, panel presets, muscle memory). A cleaner `Redrhex-v1` id is added at the
  end and both point at the same env.

## 3. The env decomposition (what moves where)

| Concern (today: inside redrhex_env.py) | Target module | Shape of the API |
|---|---|---|
| Obs vector assembly, normalization, noise | `core/observations.py` | `build_obs(state: ObsInputs, cfg, layout) -> Tensor`; `apply_obs_noise(obs, cfg, layout, gen)` |
| Simplified reward terms | `core/rewards.py` | one pure fn per term `(state, cfg) -> Tensor`, plus `total_reward(...) -> (Tensor, dict[str, Tensor])` |
| Gait phase / CPG / tripod / lateral FSM | `core/gait.py` | explicit `GaitState` dataclass of tensors; `step_gait(state, actions, dt, cfg) -> GaitState` — dt passed *once per control step* (July substep bug becomes structurally impossible) |
| Command sampling, curriculum, pushes | `core/commands.py` | `resample(...)`, `should_push(...)` — called from `_pre_physics_step` / post-step hooks, **never** from `_get_observations` (kills the state-mutation-in-obs problem, review #15) |
| DR (mass/friction/actuator/fault) | `core/domain_rand.py` | samplers return per-env tensors; env applies them to sim handles |
| Episode sums / logging | `core/buffers.py` | ONE `(num_envs, num_terms)` tensor + name list — replaces ~140 individual tensors/kernel launches (review #13); per-second normalization uses *actual* episode length (review #19) |
| Dead legacy full-reward path (~1,000 lines) | **deleted** | recoverable at tag `v0-pre-reboot` |

The env class keeps: scene/robot construction, actuator writes, resets, sim stepping,
`_get_dones`, visualization hooks. Target < 800 lines.

## 4. Contract flow (single source of truth)

```
source/RedRhex/RedRhex/contract.py          ← hand-written, reviewed, versioned
  │  CONTROL_HZ=60, joint order, obs layout slices, action scales,
  │  ABAD limits, stage gates, CONTRACT_VERSION
  │
  ├──> env cfg + core/observations.py import it directly (no copies)
  ├──> scripts/gen_contract.py  ──writes──> ros2_ws/.../redrhex_contract.py
  │        (generated file carries "GENERATED — DO NOT EDIT" header + source hash)
  ├──> tools/training_panel deploy.py validate_contract:
  │        checks dims + names + RATES + SCALES + CONTRACT_VERSION vs checkpoint metadata
  └──> tests/contract/: (a) generated file is up to date (regenerate & diff),
           (b) AST parity vs env cfg (July test, kept), (c) ONNX export I/O shape check
```

Additional contract-adjacent invariants to encode as tests: obs slice layout
(sin/cos/vel/abad boundaries — the July DR-noise bug class), action decoder gating
equivalence (env gating vs `action_decoder` behavior on a grid of inputs), checkpoint
metadata carries `CONTRACT_VERSION` so deploy refuses mismatched policies.

## 5. Config layering

```
cfg/base.py          # physical truth + defaults. Grouped, validated:
                     #   RedrhexEnvCfg(sim=SimCfg(), robot=RobotCfg(), obs=ObsCfg(),
                     #                 rewards=RewardCfg(), commands=CommandCfg(),
                     #                 dr=DRCfg(), stages=StagesCfg())
                     #   __post_init__ → validate(): stage-list lengths, limit ordering,
                     #   rate divisibility, no NaN scales. Fail LOUDLY at construction.
cfg/stages.py        # stage overlays only (what changes per curriculum stage)
cfg/experiments/*.py # ≤ ~20 lines each: named deltas from base for a run
                     #   e.g. exp_2026_07_20_linvel_dropout.py
```

Rules: no alias fields (the `randomize_mass = dr_randomize_mass` desync class dies);
deprecated fields are *deleted*, not kept; every experiment overlay is immutable once a
run has used it (new idea → new file). The panel's reward/terrain overrides map onto
experiment overlays in Phase 5 instead of global JSON files.

## 6. What deliberately does NOT change

- Isaac Lab direct-workflow architecture (no move to manager-based envs — churn without
  payoff for this task).
- RSL-RL as the training framework; teacher-student configs stay in `agents/`.
- Panel ↔ train process model (subprocess + logs) — only the override IPC changes later.
- ROS2 node graph and safety-filter design.
- Pinned versions: Isaac Lab 2.3.2 / Isaac Sim 5.1.0-rc.19 / `env_isaaclab_bin` /
  RTX 5080 16 GB. Upgrades are their own post-reboot project with the parity suite as
  the safety net (a hidden payoff of this whole plan).
