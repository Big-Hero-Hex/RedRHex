# ForwardFast Panel Recording Command Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Record ForwardFast panel videos with a deterministic forward command while preserving every non-recording and non-ForwardFast launch path.

**Architecture:** Extend the existing `play_argv()` command builder with one optional playback-command argument. The video process registry supplies `forward` only for the exact ForwardFast task; all other callers omit it and retain the current `play.py` default.

**Tech Stack:** Python, `unittest`/`pytest`, the existing Training Panel process registry, Isaac Lab playback.

---

### Task 1: Lock the recording behavior with a failing test

**Files:**
- Modify: `tools/training_panel/tests/test_processes.py`
- Test: `tools/training_panel/tests/test_processes.py`

- [ ] Add assertions to the existing ForwardFast automatic-video test:

```python
self.assertIn("--initial_command forward", command)
```

- [ ] Add a Direct recording assertion:

```python
self.assertNotIn("--initial_command", debug["command"])
```

- [ ] Run `PYTHONPATH=. pytest -q tools/training_panel/tests/test_processes.py -k 'forward_fast or video_recording_process'` and confirm the ForwardFast assertion fails because the flag is absent.

### Task 2: Implement the minimal command propagation

**Files:**
- Modify: `tools/training_panel/training_panel/commands.py`
- Modify: `tools/training_panel/training_panel/processes.py`
- Test: `tools/training_panel/tests/test_processes.py`

- [ ] Add the exact task constant and optional builder argument:

```python
FORWARD_FAST_TASK = "Template-Redrhex-ForwardFast-Direct-v0"

def play_argv(..., initial_command: str | None = None, ...) -> list[str]:
    ...
    if initial_command is not None:
        argv.extend(["--initial_command", initial_command])
```

- [ ] Supply it only from video recording:

```python
initial_command="forward" if task == FORWARD_FAST_TASK else None,
```

- [ ] Run the focused test command and confirm it passes.

### Task 3: Verify and exercise the real panel path

**Files:**
- Verify: `tools/training_panel/`
- Artifact: existing ForwardFast native `model_199.pt` video

- [ ] Run `PYTHONPATH=. pytest -q tools/training_panel/tests`.
- [ ] Run `git diff --check` and inspect the scoped diff.
- [ ] Restart the worktree panel on port 8080.
- [ ] Start one panel video recording for the existing `torsion spring` run.
- [ ] Confirm the persisted command includes `--initial_command forward` and the playback log reports `(vx, vy, wz)=(0.42, 0, 0)`.
- [ ] Confirm the recording exits successfully and produces a non-empty playable artifact.
