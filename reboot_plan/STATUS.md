# Soft Reboot Status

Last updated: 2026-07-13

| Field | Value |
|---|---|
| Design direction | Approved: core-first, simulation-first, frozen consumers |
| Detailed design checkpoint | Review requested |
| Core extraction | Not started |
| Current gate | P0 — foundation and interface freeze |
| Baseline | **Not frozen**; P1 simulation/gravity validation must pass first |
| Source snapshot | `fix/review-2026-07@5cdc824` |
| Reboot branch | `reboot/core-sim-first` |
| Desktop worktree | `/home/lab_user1/Desktop/RedRHex-core-sim-first` |
| Planning baseline commit | `a795c06` |

## Gate status

| Gate | State | Required evidence |
|---|---|---|
| P0 Foundation/interface freeze | In progress | [foundation results](evidence/foundation/RESULTS.md), [interface freeze](evidence/interface_freeze.md), [toolchain provenance](evidence/provenance/TOOLCHAIN.md), canonical command/artifact contract |
| P1 Legacy simulation/gravity validation | Blocked by P0 | [physical facts](evidence/sim_validation/FACTS.md) and `evidence/sim_validation/RESULTS.md` with G0–G10 passing and C1 recorded |
| P2 Validated legacy baseline | Not started | validated tag, full local dump, committed reduced fixture, reference training manifests |
| P3 Package scaffolding | Not started | independently importable `redrhex_contract` and `redrhex_core`, adapter smoke |
| P4 Contract extraction | Not started | read-only parity against frozen interfaces |
| P5 Core extraction | Not started | pure-Torch unit and golden parity per slice |
| P6 Isaac adapter cutover | Not started | unchanged task/CLI/checkpoint behavior, Isaac smoke |
| P7 Acceptance | Not started | rebooted reference comparison and final evidence report |

## Current evidence

- The frozen panel/reward CPU suites in this worktree pass: `282 passed, 41 subtests`.
- The frozen remote-web suite passes: `2 passed`.
- The tracked panel UI suite passes separately: `7 passed`.
- Repository-wide collection is not safe yet: `289 tests collected, 2 errors` because
  `test_joint_velocity.py` and `test_joint_velocity2.py` import Isaac Lab during
  collection in a non-Isaac Python environment.
- Root `tests/` is ignored by `.gitignore`; an ignored two-test reward-agent evaluator
  file exists in the source checkout but is absent from this isolated checkout.
- The Isaac Lab checkout used by the project is externally dirty. Its exact state must
  be resolved or pinned before a simulation result can become evidence.

Do not advance a gate by editing this file alone. A gate changes only when its command,
artifact, and commit/ADR evidence all exist.
