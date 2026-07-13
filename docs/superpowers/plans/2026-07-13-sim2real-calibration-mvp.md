# RedRHex Sim-to-Real Physics Calibration MVP Implementation Plan

> **For agentic workers:** Use test-driven development. Each task must leave its focused tests passing and commit its changes before review.

**Goal:** Add a safe, reproducible measurement-first calibration harness for RedRHex without changing the active training physics by default.

**Architecture:** A pure-Python `tools.sim2real` package owns versioned scenarios, trace/profile contracts, metrics, comparison, and bounded sweeps. The ROS deployment stack adds fail-closed per-actuator command semantics and a bounded single-main-drive probe. A dedicated Isaac Lab runner reuses the production asset/config while bypassing the RL control policy and applies candidate profiles only when explicitly requested.

**Tech stack:** Python 3.10+, NumPy, ROS 2/rclpy, Isaac Lab 2.3.2, pytest/unittest.

## Global Constraints

- Existing training physics remains unchanged unless `--physics-profile` is explicitly supplied.
- Do not add Optuna, learned actuators, chirps, automated ABAD dynamics, drops, or locomotion replay.
- Hardware commands fail closed: one selected main drive, ABAD disabled, E-stop/stale-state/exit always disable output.
- Raw data is immutable; each derived episode has numeric-only `trace.npz` plus versioned JSON metadata with independent time bases and provenance hashes.
- Fit and report subsystem metrics separately; never emit one global sim-to-real score.
- Put CPU tests under `tools/sim2real/tests/`, because the repository ignores root `tests/`.

---

### Task 1: Pure Calibration Contracts, Metrics, Scenarios, and CLI

Create `tools/sim2real/` as a lazy-loading package with `list`, `import-real`, `compare`, `sweep`, and `validate-profile` commands. Define and validate `ScenarioSpecV1`, `TraceManifestV1`, and `CalibrationProfileV1`; write numeric-only NPZ traces atomically; reject invalid versions, object arrays, non-finite values, missing channels, shape mismatches, non-monotonic time, and hash mismatches. Add reviewed JSON scenarios for audit, main step, main coast, manual load, mass/CoM, ABAD static, spring, and friction measurements. Implement offline position-derived velocity and subsystem metrics for onset delay, steady speed, rise time, overshoot, coast time, stiffness, mass/CoM, and friction. Comparison must produce separate subsystem results. Implement deterministic one-factor/coarse-grid candidate generation and cache keys, but no optimizer dependency. Add CPU tests covering all behavior.

### Task 2: ROS Fail-Safe Commanding and Main-Drive Probe

Extend `RedRhexMotorCommand.msg` with fixed `bool[6] main_drive_enable` and `bool abad_output_enable`. Update every in-repo producer/backend: normal policy commands explicitly enable all main drives and ABAD; disabled commands override masks; BioRoLa and mock honor the main mask; unsupported/provisional enabled paths fail closed. Add central `/estop` gating and an emergency-disable path in the low-level bridge, including stale raw-state handling. Ensure the manual tool sends terminal disable packets. Add `sim2real_probe` as a bounded 60 Hz, one-main-drive, ABAD-disabled step/coast publisher with dry-run, preview, heartbeat/state freshness, `--enable --confirm-risk`, amplitude/duration limits, event markers, and repeated disable in `finally`. Add pure/unit ROS tests or AST tests that can run without hardware and verify command-mask and shutdown behavior.

### Task 3: Isaac Characterization Runner and Explicit Profile Integration

Add `run-sim` to `tools.sim2real`, lazy-importing Isaac only after `AppLauncher`. Build a finite one-environment `InteractiveScene` that clones production simulation/robot/material configuration, bypasses the RL action transforms, supports fixed-base actuator and free-root/contact modes, logs every 1/120 s physics step, and writes the Task 1 trace/manifest format. Audit runtime joint/body names, masses, inertias, CoM, actuator properties, and contact availability; refuse contact scenarios when the contact probe has no measurable force. Change implicit actuator limits to `effort_limit_sim`/`velocity_limit_sim`. Add profile application helpers for rigid damping, main/ABAD/damper actuator values, joint friction, timing/sensor metadata, mass correction, and ground material. Add explicit `--physics-profile` to train/play without changing defaults. Add CPU/static tests plus a finite headless Isaac smoke run when available.

### Task 4: Integrated Verification and Documentation

Document the operator workflow, data layout, exact ROS bag topics, safety prerequisites, calibration/holdout rules, and promotion gate. Run the complete CPU suite, ROS package tests/build where dependencies exist, CLI dry-runs, and a headless Isaac probe. Confirm the original training-panel baseline remains green. Request a whole-branch code review and resolve all critical/important findings.
