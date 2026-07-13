# RedRHex Soft Reboot — Master Plan (2026-07)

This folder is the complete plan for the soft reboot of RedRHex: restructuring the
project for robustness while keeping everything that works (trained knowledge, deploy
stack, panel, docs), and switching to an AI-heavy development workflow where iteration
speed is bounded by *verification speed*, not typing speed.

**Soft reboot means:** same repo, same git history, same research goal (RL locomotion on
the RHex-style hexapod that beats MPC, sim-to-real via ROS2). What changes is the
*structure* — module boundaries, single sources of truth, test harnesses, and a
development loop designed so an AI agent can do most of the work safely.

## Reading order

| Doc | What it answers |
|---|---|
| [00_overview.md](00_overview.md) | Why reboot, the 5 principles, definition of done |
| [01_current_state_audit.md](01_current_state_audit.md) | What exists today: keep / refactor / drop verdict per component |
| [02_target_architecture.md](02_target_architecture.md) | The end-state repo structure, module boundaries, contract flow, config layering |
| [03_migration_plan.md](03_migration_plan.md) | Step-by-step phases with verification gates (the actual to-do list) |
| [04_ai_workflow.md](04_ai_workflow.md) | How to develop with AI: CLAUDE.md design, loop, guardrails, prompt recipes |
| [05_testing_and_ci.md](05_testing_and_ci.md) | Test tiers, GPU-aware CI, the preflight gate |
| [06_experiment_management.md](06_experiment_management.md) | Runs, seeds, baselines, ablation protocol, MPC comparison |
| [07_roadmap.md](07_roadmap.md) | Milestones, calendar estimate, risk register |
| [08_conventions.md](08_conventions.md) | Commits, branches, naming, language, ADRs |
| [09_sim_validation.md](09_sim_validation.md) | Step-by-step ladder proving the sim itself works as intended (L0 assets → L6 hardware) |
| [templates/](templates/) | Ready-to-copy CLAUDE.md, Makefile, ADR, experiment report, PR checklist, sim facts sheet |

## The 30-second version

1. **Freeze a baseline** (golden rollout dump + reference training run) so every later
   change can be proven behavior-preserving or knowingly behavior-changing.
2. **Extract the 4,478-line env monolith into pure-torch modules** (observations,
   rewards, gait FSM, DR, commands) using strangler-fig migration — each extraction
   gated by parity tests against the frozen baseline. Pure-torch modules run on CPU
   without Isaac Sim → tests run in seconds → AI can iterate 100× faster.
3. **One source of truth** for every constant crossing the sim→deploy boundary
   (`contract.py` generates the ROS2 contract; parity tests prevent drift).
4. **Layered config** (base → stage → experiment) replaces the ~250-scalar flat cfg.
5. **AI workflow**: CLAUDE.md hierarchy + Makefile verification targets + code-review
   loop + experiment log, so the agent can plan, edit, verify, and report autonomously,
   while reward-shaping judgment, physics plausibility, and hardware safety stay
   human-gated.
6. **Simulation validation ladder** (09): parity proves new code == old code; the ladder
   proves the sim == the real robot — step-by-step checks from asset masses and joint
   limits up through actuator responses, timing, frames, whole-robot dynamics, and
   cross-sim/hardware comparison, each with a written pass criterion and recorded
   evidence.

## Status

- Written 2026-07-13, on branch `fix/review-2026-07` (all 2026-07 review fixes committed).
- Grounded in: `docs/project_review_2026-07-09.md` (35 findings + fix status),
  `docs/2026_Midterm.md` (research direction), `docs/COMMANDS.md` (machine setup),
  and direct inspection of the current tree.
