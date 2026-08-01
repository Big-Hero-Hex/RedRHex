# Training Panel Spring Backend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the Training Panel carry a selected torsion-spring backend from training through playback, automatic video recording, and ONNX export, then launch and inspect a 200-iteration native ForwardFast run.

**Architecture:** Add `spring_backend` to the panel's canonical `TrainingParams`, validate it at the API boundary, and include it in every Isaac command builder. Resolve replay/export physics from the originating run first and its immutable `params/torsion_spring.yaml` second, defaulting to `explicit` only for legacy runs with no backend metadata. The browser form and run views expose the value without changing any simulator, task, reward, or checkpoint contracts.

**Tech Stack:** Python dataclasses and `unittest`/`pytest`, the existing Training Panel HTTP server and process registry, vanilla JavaScript/HTML, Isaac Lab/RSL-RL.

---

## Task 1: Lock the command and parameter contract with failing tests

**Files:**

- Modify: `tools/training_panel/tests/test_commands.py`
- Test: `tools/training_panel/tests/test_commands.py`

- [ ] Add a test that `TrainingParams.from_dict({"spring_backend": "native"})` stores and serializes `native`.
- [ ] Add a legacy/default test proving an omitted value serializes as `explicit`.
- [ ] Add a validation test proving a value outside `explicit` and `native` raises `ValueError` before queuing.
- [ ] Extend the training command test to require `--spring-backend native`.
- [ ] Extend playback and ONNX export command tests to require their requested backend and confirm their default remains `explicit`.
- [ ] Run `PYTHONPATH=. pytest -q tools/training_panel/tests/test_commands.py` and confirm the new assertions fail for the missing interface, not because of fixture errors.

## Task 2: Implement the backend command contract

**Files:**

- Modify: `tools/training_panel/training_panel/commands.py`
- Test: `tools/training_panel/tests/test_commands.py`

- [ ] Define the canonical allowed backend values in one constant.
- [ ] Add `spring_backend: str = "explicit"` to `TrainingParams`.
- [ ] Parse the field in `from_dict()` and reject anything outside the canonical values in `validate()`.
- [ ] Append `--spring-backend <backend>` in `training_argv()`.
- [ ] Add a keyword-only `spring_backend="explicit"` parameter to `play_argv()` and pass it to `scripts/rsl_rl/play.py`.
- [ ] Add the same backward-compatible keyword to `export_onnx_argv()` and forward it to `play_argv()`.
- [ ] Run `PYTHONPATH=. pytest -q tools/training_panel/tests/test_commands.py` and confirm it passes.
- [ ] Commit the command contract as `feat(panel): add torsion spring backend parameter`.

## Task 3: Lock run-backend resolution with failing tests

**Files:**

- Modify: `tools/training_panel/tests/test_processes.py`
- Modify: `tools/training_panel/tests/test_deploy.py`
- Test: `tools/training_panel/tests/test_processes.py`
- Test: `tools/training_panel/tests/test_deploy.py`

- [ ] Add a process test whose run has `params.spring_backend = native` and assert play/video/export commands contain `--spring-backend native`.
- [ ] Add a discovered-run test with no panel parameter but `params/torsion_spring.yaml` containing `spring_backend: native`; assert the command is native.
- [ ] Add a legacy-run test with neither source and assert explicit is used.
- [ ] Add a malformed stored-backend test and assert it raises rather than silently changing physics.
- [ ] Extend the deployment export-stage test to prove a run's resolved backend reaches `export_onnx_argv()`.
- [ ] Run `PYTHONPATH=. pytest -q tools/training_panel/tests/test_processes.py tools/training_panel/tests/test_deploy.py` and confirm failures identify missing resolution/propagation.

## Task 4: Implement one resolver and propagate it through panel processes

**Files:**

- Modify: `tools/training_panel/training_panel/commands.py`
- Modify: `tools/training_panel/training_panel/processes.py`
- Modify: `tools/training_panel/training_panel/deploy.py`
- Test: `tools/training_panel/tests/test_processes.py`
- Test: `tools/training_panel/tests/test_deploy.py`

- [ ] Add a small resolver that accepts run metadata and an optional checkpoint path, checking in order: `run["params"]["spring_backend"]`, the run log's `params/torsion_spring.yaml`, the checkpoint's run directory, then the legacy default `explicit`.
- [ ] Parse only the top-level scalar `spring_backend` from the saved YAML so the lightweight panel server does not gain a mandatory PyYAML dependency.
- [ ] Treat an absent value as legacy, but reject a present invalid value.
- [ ] In `ProcessRegistry.start_play()`, `start_video_recording()`, and `start_onnx_export()`, fetch the source run, resolve the backend, and pass it to the command builder.
- [ ] In deployment `run_export_stage()`, resolve from its supplied run and checkpoint so deployment validation exports under matching physics.
- [ ] Run `PYTHONPATH=. pytest -q tools/training_panel/tests/test_processes.py tools/training_panel/tests/test_deploy.py` and confirm they pass.

## Task 5: Add the panel selector and browser-state coverage

**Files:**

- Modify: `tools/training_panel/static/index.html`
- Modify: `tools/training_panel/static/app.js`
- Modify: relevant files under `tools/training_panel/ui_tests/`
- Test: `tools/training_panel/ui_tests/`

- [ ] Add a required `Spring Backend` select with `explicit` first and `native` second.
- [ ] Preserve the selected string in form submission; no simulator fields or numeric conversions change.
- [ ] Restore `params.spring_backend || "explicit"` when tweaking an existing run.
- [ ] Show the backend in run summary/detail/comparison metadata wherever training parameters are rendered.
- [ ] Add UI/static assertions proving both options exist, `explicit` is the default, and a native tweak restores `native`.
- [ ] Run `PYTHONPATH=. pytest -q tools/training_panel/ui_tests/test_local_panel_ui.py` and confirm it passes.

## Task 6: Verify the complete panel change and review the diff

**Files:**

- Verify: `tools/training_panel/`

- [ ] Run `PYTHONPATH=. pytest -q tools/training_panel/tests` for all Training Panel Python tests.
- [ ] Run `PYTHONPATH=. pytest -q tools/training_panel/ui_tests/test_local_panel_ui.py` for all Training Panel browser/UI tests.
- [ ] Inspect `git diff --check` and `git diff --stat`.
- [ ] Confirm the only pre-existing unrelated worktree change is the generated tracked `scripts/rsl_rl/__pycache__/cli_args.cpython-311.pyc`; do not stage or modify it.
- [ ] Request an independent code review focused on backend mismatches and legacy compatibility; address any substantiated findings with a new failing test first.
- [ ] Commit the completed panel integration as `feat(panel): preserve torsion spring backend across workflows`.

## Task 7: Launch the native 200-iteration checkpoint through the panel

**Files/artifacts:**

- Run: worktree-local Training Panel server on an unused local port
- Produce: `logs/rsl_rl/redrhex_forward_fast/<timestamp>_torsion_native_v11_provisional_200/`

- [ ] Start the panel from this worktree with `REDRHEX_ROOT` pointing to this worktree and the known Isaac Lab/conda environment.
- [ ] Read `/api/training/defaults` and verify it reports `spring_backend: explicit`.
- [ ] Submit exactly one run through `/api/training/start`: task `Template-Redrhex-ForwardFast-Direct-v0`, native backend, seed 42, 4096 environments, 200 iterations, headless, display name `torsion_native_v11_provisional_200`, baseline reward and terrain presets.
- [ ] Inspect the recorded process command and stored run parameters before letting the run proceed; both must say `native`.
- [ ] Monitor the run to completion, reporting progress and checking logs for NaN, traceback, runaway state, or checkpoint failures.
- [ ] Confirm a final checkpoint exists and its `params/torsion_spring.yaml` says `spring_backend: native` and `calibration_status: uncalibrated`.

## Task 8: Verify the panel's automatic video and decide integration

**Files/artifacts:**

- Verify: the run's panel-generated high-quality video
- Verify: final logs and checkpoint metadata

- [ ] Confirm the successful training monitor automatically starts video recording with the same native backend, 1920x1080 resolution, 1200 steps, 30 fps, and quality rendering.
- [ ] Wait for the video process to complete and confirm its artifact is non-empty and readable.
- [ ] Sample frames near the start, middle, and end and inspect posture, leg/spring motion, camera framing, ground interaction, and obvious simulation instability.
- [ ] Report the run metrics and visual result, explicitly retaining the provisional/uncalibrated status.
- [ ] Before claiming completion, run the verification-before-completion checklist and fresh panel tests.
- [ ] If the video and runtime checks pass, inspect the base worktree and remote tracking state before merging. Merge only into the known parent branch without disturbing unrelated changes, then push the authorized branch update. If the target or worktree state is ambiguous, stop and ask for the exact integration target rather than guessing.
