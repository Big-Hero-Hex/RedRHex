# 05 — Testing, Evidence, and CI

## Current baseline

From the Desktop worktree on 2026-07-13:

| Command | Result |
|---|---|
| `python -m pytest -q tools/reward_agent/tests tools/training_panel/tests` | `282 passed, 41 subtests` |
| `node --test tools/training_panel/remote_web/*.test.mjs` | `2 passed` |
| `python -m pytest -q tools/training_panel/ui_tests` | `7 passed` |
| `python -m pytest --collect-only -q` | `289 collected, 2 errors` |

The collection errors are root Isaac diagnostics, and the blanket `tests/` ignore hides
an intended two-test reward evaluator in the source checkout. P0 repairs discovery before
the project treats any root-level green result as trustworthy.

## Target tiers

| Tier | Runs on | Scope | Gate |
|---|---|---|---|
| Foundation | CPU | collection completeness, ignore rules, frozen-tree guard, runtime manifest checker | P0 |
| Contract | CPU, stdlib | ordering/shapes/units/rates/scales/slices/version; legacy and frozen-ROS read-only parity | P4–P7 |
| Core | CPU + Torch | hand-computable functions, explicit state/RNG/time, forbidden-import checks | P3–P7 |
| Golden-reduced | CPU + committed fixture | seam replay for every extracted slice | P2–P7, CI |
| Adapter | Isaac environment | both Gym IDs, cfg compatibility, reset/step, shape/range, train/play/export smoke | P3–P7 |
| Simulation validation | Isaac/GPU | G0–G10 gravity, frames, asset, contacts, actuators, rewards, timing, deterministic invariants | P1 and P7 |
| Frozen consumers | CPU/read-only | existing panel, remote-web, reward-agent suites and CLI construction | every cutover/P7 |
| Golden-full/reference | Isaac/GPU/local | full rollout parity and fixed-seed training/evaluation comparison | P2 and P7 |

Panel/reward tests remain in their current frozen directories. They are invoked as
boundary regressions, not moved or edited during the reboot.

## P0 discovery policy

- Root pytest configuration enumerates intended CPU test paths and registers `fast`,
  `golden`, and `isaac` markers.
- Default CPU collection must never import or initialize Isaac/Omniverse.
- Simulator diagnostics use explicit scripts or `tests/sim/**` with an explicit Isaac
  command; filenames at repository root must not masquerade as CPU tests.
- `.gitignore` must not hide source tests. Probe the intended root and nested paths in
  verification.
- After recreating the ignored evaluator's two invariants under the root reboot tests,
  the explicit expected inventory is 291 Python tests plus 41 subtests: 282 frozen
  panel/reward tests, 2 external black-box reward tests, and 7 panel UI tests. The 2
  Node remote-web tests are separately required. Inventory changes must be explained.
- Do not claim pre-commit as a gate until its missing license-header configuration is
  repaired in a separately scoped change.

## Golden artifact policy

The original plan was contradictory: it ignored `baselines/` while asking CI to consume
files there. The split is now explicit:

```text
tests/fixtures/reboot/<baseline-id>/   # small, deterministic, committed, CI-safe
baselines/reboot/<baseline-id>/        # full rollouts/checkpoints/TB, local + ignored
artifacts/reboot/<run-id>/              # raw diagnostic CSV/log/plots, local + ignored
reboot_plan/evidence/**                 # tracked manifests/results/hashes
```

A fixture is never overwritten. A changed validated source, physics model, schema, or
approved tolerance creates a new baseline ID and explains the relationship in an ADR.

## Required manifest fields

- source commit and branch purpose;
- working-tree/external-runtime dirty-state policy;
- robot USD and relevant source-description hashes;
- complete resolved config and overrides;
- task, seed/RNG state, environment count, steps/iterations, device;
- Isaac Lab SHA/tag, Isaac Sim build, Python and important package versions;
- exact command and start/end timestamps;
- schema/contract version and artifact SHA-256 values;
- threshold set used and per-check PASS/FAIL/BLOCKED result.

## CI model

### CPU CI on every change

- safe collection/completeness and frozen-path guard;
- contract and core unit tiers;
- reduced golden replay;
- frozen panel/reward/remote-web regressions;
- formatting/lint only after the toolchain has a valid configuration.

CPU CI must install/import `redrhex_contract` and `redrhex_core` without Isaac.

### Local Isaac preflight

- provenance check first;
- both Gym task smoke checks;
- adapter and simulation invariant tests;
- full local golden replay;
- export/read-only deploy parity where applicable.

Long reference training is a named P2/P7 protocol, not part of every preflight.

## Gate evidence

| Gate | Required test evidence |
|---|---|
| P0 | complete 291-test Python/UI + 2-test Node inventory, full pass, frozen-change drill, toolchain provenance pass, existing interface snapshots, future diagnostic/artifact schema checks |
| P1 | G0 diagnostic CLI non-GPU dry-run plus every G0–G10 result `PASS`; C1 frozen compatibility recorded separately; no G-check `FAIL`/`BLOCKED` before baseline capture |
| P2 | deterministic-envelope report, committed fixture hash, full local manifest, reference manifests |
| P3 | both packages independently install/import; forbidden-import tests; both legacy task smokes |
| Each P4/P5 slice | focused unit, aggregate CPU, reduced/full golden, frozen regression, adapter smoke |
| P6 | task/CLI/checkpoint/export/artifact compatibility suite plus P1 invariants |
| P7 | all tiers fresh plus reference comparison and acceptance report |

## Test conventions

- Name the invariant (`test_projected_gravity_matches_analytic_pose`), not the incident.
- Put a regression at the cheapest layer that can express it.
- No network, log-directory dependence, or sleeps in CPU tiers.
- Missing evidence is BLOCKED; flaky tests are fixed or explicitly quarantined with an
  owner/reason before a gate can advance.
- A failing result is never resolved by editing the expected fixture first.
