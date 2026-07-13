# 06 — Experiment Management

The reboot's second product (after clean code) is a clean *experiment record* — the raw
material of the thesis/reports. Rule zero: **a run whose config, commit, and outcome
can't be reconstructed later did not happen.**

## 1. Anatomy of an experiment

```
experiments/
├── LOG.md                        # append-only one-liner ledger (see §2)
└── reports/
    └── 2026-07-20_linvel-dropout_s42.md    # from templates/experiment_report.md

source/RedRhex/.../cfg/experiments/
└── exp_2026_07_20_linvel_dropout.py        # ≤20-line overlay; immutable once used

logs/rsl_rl/redrhex_wheg/<ts>_<exp-name>_s<seed>/   # checkpoints+TB (local, gitignored)
```

Every run records (in checkpoint metadata + report header): git commit hash, experiment
overlay name, seed, CONTRACT_VERSION, GPU, Isaac Lab/Sim versions, start/end, and the
one-sentence hypothesis.

## 2. The ledger — `experiments/LOG.md`

One line per run, append-only, written when the run *starts* (agent fills the verdict
when it ends):

```
| date | run id | overlay | seed(s) | hypothesis | verdict |
| 2026-07-20 | 0720a | linvel_dropout | 42,43,44 | dropout closes deploy lin-vel gap w/o curve cost | ✅ curve -2% (noise), deploy eval +18% |
```

This is the file the human scans weekly and the agent greps before proposing anything
("has this been tried?").

## 3. Naming and seeds

- Run name: `<date-compact><letter>_<overlay>_s<seed>` (e.g. `0720a_linvel_dropout_s42`).
- Seed policy: **exploration = 1 seed (42); any claim = 3 seeds (42/43/44)**; thesis
  headline numbers = 5 seeds. Never compare a 1-seed curve against a 3-seed band and
  conclude anything.
- The reference baseline is always `baselines/ref_run_v0` (Phase 0.3) until an ADR
  promotes a new one (`ref_run_v1` after Phase 2, expected).

## 4. Comparison discipline

1. **One variable per experiment.** The overlay diff *is* the variable; if the overlay
   changes two things, split it.
2. **Fixed evaluation**: `eval_command_sweep` (existing) on the final checkpoint +
   the 3 standard metrics (tracking error, energy proxy, termination rate) — same
   sweep for every run, versioned with the code.
3. **Curve comparison**: agent overlays TB scalars of run vs baseline band in the
   report (matplotlib PNG committed alongside the report — small, worth it).
4. **Negative results get reports too** — they're the cheapest thing the AI writes and
   the thing you'll wish you had in November.

## 5. Roles

- **Agent**: creates overlay + ledger line, launches (via panel queue or `make train`),
  monitors, pulls scalars, drafts the report with a *proposed* verdict, links follow-ups.
- **Human**: confirms/edits the verdict (shaping and physics judgment), picks the next
  hypothesis, promotes baselines.
- **reward_agent tool**: its automated search runs follow the same ledger + overlay
  discipline (its experiment_store gets a thin adapter in Phase 4) — no parallel
  bookkeeping universe.

## 6. Relationship to the panel

The panel remains the launcher/queue/monitor UI. Migration target (Phase 5.1): a panel
run = (overlay name, seed, commit) — the same triple as a CLI run, writing the same
ledger. Until then, panel runs must still get a ledger line (agent reconciles from
panel history if needed).

## 7. The MPC-comparison campaign (thesis endgame)

Runs like any other experiment family, but with its own fixed benchmark suite
(Phase 4.7): disturbance-recovery scenarios, efficiency (CoT), terrain generalization —
identical scenario seeds for RL and MPC. The suite is code (`tests/benchmarks/` or
`scripts/benchmarks/`), versioned, so the November numbers are regenerable from a tag.
Design decisions for scenarios go through the strategy doc §6 + an ADR.
