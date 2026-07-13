# 08 — Conventions

Short by design. Anything not covered: match surrounding code.

## 1. Git

- **Branches**: `main` is always trainable (smoke passes). Work on short-lived branches:
  `refactor/<step>` (migration), `exp/<overlay>` (only if code changes; overlays alone
  go straight to main), `fix/<thing>`. Delete after merge. No branch lives > 1 week.
- **Commits**: conventional prefixes already in use — `fix:`, `feat:`, `refactor:`,
  `docs:`, `test:`, `chore:`. **One issue/step per commit** (the July pattern). Body
  says *why* + names the plan step or review finding (`(migration 2.3)`, `(review #8)`).
- **Tags**: `v0-pre-reboot` (Phase 0), `v1-clean-core` (M2), `v2-deploy-hardened` (M3),
  plus a tag per thesis-cited result campaign.
- **Merging**: PR checklist (templates/pr_checklist.md) even when self-merging —
  it's the mechanical stand-in for a GPU CI gate.

## 2. Language

- **Code, comments, commit messages, tests, CLAUDE.md**: English (tooling and AI work
  best; the codebase is already English).
- **Research reports / thesis-facing docs** (`docs/`, `experiments/reports/`):
  Traditional Chinese or English at the author's preference — existing zh-TW docs stay.
  The ledger (`experiments/LOG.md`) is English one-liners for grep-ability.

## 3. Python

- ruff (format + lint) replaces black + flake8 — one tool, one config in
  `pyproject.toml`. Line length 100 (matches current style closer than 88).
- Type hints required in `core/` and `contract.py` (the AI-edited hot zones); elsewhere
  encouraged, not enforced.
- No new module > ~600 lines; no function > ~80 lines in `core/` (review-ability caps).
- Tensor code: shape comments on non-obvious tensors (`# (num_envs, n_legs)`);
  device/dtype explicit at creation in `core/`.
- Comments state invariants and physical meaning, not narration. Units in names or
  comments for every physical quantity (`_rad`, `_hz`, `dt_s`, `torque_nm`).

## 4. Configs & constants

- A number appearing in two places is a bug; it belongs in `contract.py` (cross-boundary)
  or the owning cfg group (single-domain).
- Experiment overlays are immutable after first use (06 §1). Base cfg edits require a
  `refactor:`/`feat:` commit + a note whether checkpoints remain loadable.
- Deprecated = deleted (git remembers). No `_old`, `_v2`, commented-out blocks.

## 5. ADRs — `docs/adr/NNNN-title.md` (template in templates/adr.md)

Required for: behavior-changing migration steps, baseline promotions, parity-tolerance
changes, physics-model changes (masses, actuators, damping), symmetry approach,
benchmark-scenario definitions, any dependency upgrade. One page max.

## 6. Documentation placement

| Content | Lives in |
|---|---|
| How to run things on THIS machine | root CLAUDE.md + docs/COMMANDS.md |
| Architecture / module ownership | `source/RedRhex/CLAUDE.md` (agent-facing) + 02 of this plan (human-facing) |
| Decisions | `docs/adr/` |
| Experiment outcomes | `experiments/` (06) |
| Research narrative / literature | `docs/` long-form reports (as today) |
| Plan status | checkboxes in `reboot_plan/03_migration_plan.md`, updated in the commit that completes a step |

## 7. Naming

- Runs: `<MMDD><letter>_<overlay>_s<seed>` (06 §3).
- Tests: `test_<invariant_in_words>`.
- Overlays: `exp_<yyyy_mm_dd>_<slug>.py`.
- Core functions: verb-first, physics-explicit (`compute_projected_gravity`, not
  `get_pg`).
- The robot description folder is `assets/robot_description/` — nothing shipped is
  named `test_*` except tests.
