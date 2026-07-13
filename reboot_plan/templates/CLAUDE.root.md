# RedRHex — Agent Brief (root CLAUDE.md)

<!-- Copy to /CLAUDE.md at Phase 1.5. Keep < 150 lines. Update in the same commit that
     makes any line stale. Area details live in per-directory CLAUDE.md files. -->

RL locomotion for the RedRHex hexapod (RHex-style whegs: per-leg main drive + ABAD).
Isaac Lab (direct workflow) + RSL-RL PPO in sim; ROS2 deploy stack for hardware.
Research goal: demonstrate RL advantages over MPC; sim-to-real.

## Machine facts (this machine only — do not guess paths)
- Repo: `/home/lab_user1/Py/RedRHex` (run everything from here so logs land in-repo)
- Isaac Lab: `/home/lab_user1/isaac_lab_ws/IsaacLab` (v2.3.2) — launcher `isaaclab.sh`
- Isaac Sim: `/home/lab_user1/isaacsim` (5.1.0-rc.19)
- Conda env: `env_isaaclab_bin` (NEVER `base` or `env_isaaclab`)
- GPU: single RTX 5080 16 GB — check `nvidia-smi` / panel queue before GPU targets;
  never start smoke/preflight beside a live training run
- Shell prelude: `export TERM=xterm; conda activate env_isaaclab_bin`
- Task ids: `Template-Redrhex-Direct-v0` (legacy, keep working), `Redrhex-v1` (new)

## Verification ladder (the ONLY sanctioned entry points)
```
make lint           # ruff — every edit
make test-fast      # CPU unit tests, <30s — every core/cfg change
make test-contract  # contract/layout parity — anything touching contract or deploy
make test-golden    # replay vs frozen baseline — every extraction/refactor
make smoke          # GPU, 5-iter train, ~5min — env/integration changes
make preflight      # GPU, full local gate — before merging to main / ending a session
make contract       # regenerate ROS2 contract from contract.py
```
Run the cheapest tier that can catch your change; run `preflight` before merge.

## Hard rules (violating these is never OK)
1. NEVER edit `ros2_ws/**/redrhex_contract.py` — it is generated. Edit
   `source/RedRhex/RedRhex/contract.py` then `make contract`.
2. NEVER widen a parity tolerance or skip a golden test to get green. A parity failure
   is a finding: stop and report it.
3. NEVER touch hardware paths (`enable_policy`, bridge on real robot) — human-present only.
4. `baselines/` is append-only. `cfg/experiments/*` overlays are immutable once used.
5. Training runs > 30 GPU-min: only if asked, or pre-approved overnight queue slot.
6. One issue per commit; prefix `fix:/feat:/refactor:/docs:/test:/chore:`; body says why
   and names the plan step / finding.
7. Reward/physics *intent* is the human's call — implement, measure, report, propose;
   don't unilaterally re-shape.

## Where things are
- Env package: `source/RedRhex/RedRhex/` — see its CLAUDE.md for module ownership + obs
  layout. Core math: `core/` (pure torch, no isaaclab imports — enforced by test).
- Configs: `.../tasks/direct/redrhex/cfg/` (base → stages → experiments overlays).
- Training scripts: `scripts/rsl_rl/` (checkpoint utils in `scripts/common/`).
- Panel: `tools/training_panel/` (see its CLAUDE.md; launcher/queue/history).
- Deploy: `ros2_ws/src/` (see its CLAUDE.md; contract is generated).
- Tests: `tests/{unit,contract,golden,sim}` — tiers in docs; markers `fast|golden|isaac`.
- Experiments ledger: `experiments/LOG.md` (append a line when starting a run; grep it
  before proposing an experiment). Reports: `experiments/reports/`.
- Plan/status: `reboot_plan/03_migration_plan.md` checkboxes = current migration state.
- Sim validation: ladder spec `reboot_plan/09_sim_validation.md`; ground truth
  `docs/sim_facts.md`; results `experiments/reports/sim_validation/RESULTS.md`.
  After ANY physics/asset/actuator change: re-run the affected ladder level.
- Machine ops detail: `docs/COMMANDS.md`. Decisions: `docs/adr/`.

## Standard loops
- Structural change: plan → implement → ladder up to smoke → /code-review → commit →
  tick the plan checkbox.
- Experiment: hypothesis line in LOG.md → new overlay → smoke → launch → pull TB
  scalars → report from `experiments/_template.md` → human verdict.
- Unattended session: scoped task list only; end with preflight + session report;
  stop (don't improvise) on unexplained parity failure.
