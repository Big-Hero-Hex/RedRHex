# 01 — Current State Audit

Behavioral snapshot: `fix/review-2026-07@5cdc824` (2026-07-13). Completed July review
fixes are recorded in `docs/project_review_2026-07-09.md`; deferred findings remain and
must not be described as fixed.

## Verdicts

| Component | Verdict during reboot | Reason |
|---|---|---|
| `redrhex_env.py` | **EXTRACT** | Preserve behavior while moving pure logic into sibling `redrhex_core`; retain simulator orchestration in `RedRhex`. |
| `redrhex_env_cfg.py` | **KEEP/ADAPT** | It remains the compatibility source during extraction. Broad config redesign is deferred. |
| Gym registration and agents | **KEEP STABLE** | Both current task IDs, entry-point keys, checkpoints, and CLI behavior are public boundaries. |
| Robot USD/description | **KEEP IN PLACE** | Audit as simulator evidence; asset moves and physical changes are deferred. |
| `tools/training_panel/**` | **FROZEN** | Includes local panel, remote worker/web, Supabase, command construction, and tests. Read-only regression only. |
| `tools/reward_agent/**` | **FROZEN** | Read-only regression only; no experiment-store integration during the core reboot. |
| `ros2_ws/**` | **FROZEN** | Read-only contract comparison only; no generated files, messages, topics, nodes, or YAML changes. |
| Training/play/eval scripts | **KEEP STABLE** | Existing process boundary used by humans and frozen consumers. Internal adapter-compatible changes only when directly required. |
| Root `test_joint_velocity*.py` | **RECLASSIFY IN P0** | Diagnostics that launch Isaac at import time, use old absolute paths, and currently break CPU collection. |
| Root test/config tooling | **ESTABLISH IN P0** | No root pytest config/Makefile; blanket `.gitignore` rule hides tests. |

## Core issues carried forward

- Environment and config are large and coupled; pure logic cannot be imported without
  traversing eager task registration and Isaac dependencies.
- Stable facts such as observation layout, action ordering, rates, scales, and frame
  conventions are duplicated across simulator/deploy boundaries.
- The +90°-about-X initialization and projected-gravity convention need analytic runtime
  validation before any gravity fix or baseline claim.
- Density-based mass, actuator softness, body damping, and phase-proxy contact logic are
  suspects, not yet measured diagnoses.
- `source/RedRhex/setup.py` installs only `packages=["RedRhex"]`, and importing the package
  eagerly imports tasks/UI; independent CPU packages need explicit packaging boundaries.

## Foundation defects confirmed on the source machine

1. `.gitignore` ignores every directory named `tests/` except selected panel tests. An
   ignored `tools/reward_agent/tests/test_evaluator.py` therefore did not enter the clean
   reboot worktree.
2. Repository collection finds 289 tests but errors on both root velocity diagnostics
   because the base Python environment has no `isaaclab`.
3. Targeted tracked suites pass (`282 passed, 41 subtests`), and the two remote-web tests
   pass. The ignored evaluator adds two more passing tests only in the source checkout.
4. CI deploys panel Pages but does not run Python test tiers.
5. Documented pre-commit configuration references missing license-header files, so it is
   not a valid P0 gate and must be repaired separately rather than claimed green.
6. External runtime provenance is identifiable but not clean: Isaac Lab is based on
   `v2.3.2` / `37ddf626871758333d6ed89cf64ad702aef127d0` with a local Flatdict change;
   Isaac Sim reports `5.1.0-rc.19+release.26219.9c81211b.gl`.

## What must be protected

- Simplified reward behavior and learned policy/checkpoint compatibility.
- Both current Gym registrations.
- Panel command construction, history, remote workflows, and artifact expectations.
- Reward-agent behavior and storage.
- ROS safety, observation/action contract, topics/messages, and hardware workflows.
- Existing research records, checkpoints, logs, and long-form documentation.

## Baseline inputs captured only after P1 passes

- Complete seam-level rollout data, not only totals: raw state, actions, observations,
  reward components, gait/command state, reset events, cfg, seeds/RNG, and hashes.
- Best compatible checkpoints and evaluation metrics with immutable manifests.
- A named fixed-seed reference training protocol and comparison rules.
- A reduced committed CI fixture plus full ignored local tensors/artifacts.

Until P1 passes, the current simulator is an object under test—not a golden oracle.
