# Reward Agent CLI Trial Launch Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a manual CLI workflow for previewing and launching candidate reward trials from saved Reward Agent sessions.

**Architecture:** Extend the existing headless reward-agent package. Add dry-run trial previews to `agent.py`, add a `build_panel_registry()` adapter for the existing Training Panel process registry, and add a `queue-trials` CLI command that requires either `--dry-run` or `--launch`.

**Tech Stack:** Python stdlib, `unittest`, existing `ExperimentStore`, existing `TrainingParams`, existing Training Panel `ProcessRegistry`.

## Global Constraints

- Keep the existing Training Panel focused on manual operations.
- Do not let the agent silently edit reward source code.
- Do not let the LLM decide success without metric-based scoring.
- Do not duplicate training launch, history, or TensorBoard parsing code already owned by the panel.
- New reward code proposal support is out of scope for this plan.
- Web UI integration is out of scope for this plan.
- Actual launch must require explicit `--launch`; dry-run must be available for safe manual inspection.

---

## File Structure

- `tools/reward_agent/agent.py`: factor candidate-to-`TrainingParams` conversion, add `preview_candidate_trials`, add `limit` support for `queue_candidate_trials`.
- `tools/reward_agent/launcher.py`: create the real panel `ProcessRegistry` for CLI launch mode.
- `tools/reward_agent/__main__.py`: add `queue-trials` CLI command with `--dry-run` and `--launch`.
- `tools/reward_agent/tests/test_agent.py`: add dry-run preview and limit tests.
- `tools/reward_agent/tests/test_cli.py`: add CLI dry-run and fake-launch tests.
- `tools/reward_agent/tests/test_launcher.py`: verify the launcher uses repo root and constructs a registry-compatible object without invoking training.

### Task 1: Dry-Run Preview and Limit Support

**Files:**
- Modify: `tools/reward_agent/agent.py`
- Modify: `tools/reward_agent/tests/test_agent.py`

**Interfaces:**
- Produces: `candidate_training_params(base_params: dict, candidate: dict, session_id: str, max_iterations: int | None = None) -> TrainingParams`
- Produces: `preview_candidate_trials(store: ExperimentStore, session_id: str, base_params: dict, candidates: list[dict], max_iterations: int | None = None, limit: int | None = None) -> list[dict]`
- Updates: `queue_candidate_trials(..., limit: int | None = None) -> list[dict]`

- [x] **Step 1: Write failing tests**

Add these tests to `tools/reward_agent/tests/test_agent.py`:

```python
    def test_preview_candidate_trials_persists_dry_run_without_registry(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = ExperimentStore(RewardAgentPaths.from_repo_root(Path(tmp)))
            session = store.create_session({"objective": "preview trials"})

            trials = preview_candidate_trials(
                store,
                session["id"],
                {"task": "Template-Redrhex-Direct-v0", "num_envs": 4, "max_iterations": 100, "device": "cpu"},
                [
                    {"id": "cand_1", "reward_overrides": {"v2_reward_scales": {"velocity_tracking": 5.0}}},
                    {"id": "cand_2", "reward_overrides": {"v2_reward_scales": {"velocity_tracking": 3.5}}},
                ],
                max_iterations=7,
                limit=1,
            )

            self.assertEqual(len(trials), 1)
            self.assertEqual(trials[0]["status"], "dry_run")
            self.assertIsNone(trials[0]["panel_run_id"])
            self.assertEqual(trials[0]["params"]["max_iterations"], 7)
            self.assertEqual(store.load_trials(session["id"])[0]["candidate_id"], "cand_1")

    def test_queue_candidate_trials_respects_limit(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = ExperimentStore(RewardAgentPaths.from_repo_root(Path(tmp)))
            session = store.create_session({"objective": "limit trials"})
            registry = FakeRegistry()

            trials = queue_candidate_trials(
                store,
                session["id"],
                registry,
                {"task": "Template-Redrhex-Direct-v0", "num_envs": 4, "max_iterations": 100, "device": "cpu"},
                [
                    {"id": "cand_1", "reward_overrides": {"v2_reward_scales": {"velocity_tracking": 5.0}}},
                    {"id": "cand_2", "reward_overrides": {"v2_reward_scales": {"velocity_tracking": 3.5}}},
                ],
                max_iterations=5,
                limit=1,
            )

            self.assertEqual(len(trials), 1)
            self.assertEqual(len(registry.params), 1)
            self.assertEqual(trials[0]["candidate_id"], "cand_1")
```

- [x] **Step 2: Verify red**

Run: `python -m unittest tools.reward_agent.tests.test_agent -v`

Expected: FAIL with import/name errors for `preview_candidate_trials` and unexpected `limit`.

- [x] **Step 3: Implement agent helpers**

Update imports in `test_agent.py` to import `preview_candidate_trials`. Update `agent.py` by extracting candidate parameter creation, adding `_select_candidates`, implementing `preview_candidate_trials`, and adding the `limit` parameter to `queue_candidate_trials`.

- [x] **Step 4: Verify green**

Run: `python -m unittest tools.reward_agent.tests.test_agent -v`

Expected: PASS.

### Task 2: Panel Registry Launcher Adapter

**Files:**
- Create: `tools/reward_agent/launcher.py`
- Create: `tools/reward_agent/tests/test_launcher.py`

**Interfaces:**
- Produces: `build_panel_registry(repo_root: Path)`

- [x] **Step 1: Write failing test**

Create `tools/reward_agent/tests/test_launcher.py`:

```python
import tempfile
import unittest
from pathlib import Path

from tools.reward_agent.launcher import build_panel_registry


class LauncherTests(unittest.TestCase):
    def test_build_panel_registry_uses_requested_repo_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry = build_panel_registry(Path(tmp))

            self.assertEqual(registry.paths.repo_root, Path(tmp).resolve())
            self.assertEqual(registry.history.paths.repo_root, Path(tmp).resolve())


if __name__ == "__main__":
    unittest.main()
```

- [x] **Step 2: Verify red**

Run: `python -m unittest tools.reward_agent.tests.test_launcher -v`

Expected: FAIL with `ModuleNotFoundError` for `tools.reward_agent.launcher`.

- [x] **Step 3: Implement launcher**

Create `tools/reward_agent/launcher.py` using `dataclasses.replace(PanelPaths.from_env(), repo_root=Path(repo_root).resolve())`, `HistoryStore`, and `ProcessRegistry`.

- [x] **Step 4: Verify green**

Run: `python -m unittest tools.reward_agent.tests.test_launcher -v`

Expected: PASS.

### Task 3: Queue-Trials CLI

**Files:**
- Modify: `tools/reward_agent/__main__.py`
- Modify: `tools/reward_agent/tests/test_cli.py`

**Interfaces:**
- Produces: `python -m tools.reward_agent --repo-root <path> queue-trials --session-id <id> --base-params-json <json> --dry-run`
- Produces: `python -m tools.reward_agent --repo-root <path> queue-trials --session-id <id> --base-params-json <json> --launch`
- Updates: `main(argv: list[str] | None = None, registry_factory=None) -> int`

- [x] **Step 1: Write failing tests**

Add tests to `tools/reward_agent/tests/test_cli.py`:

```python
    def test_queue_trials_dry_run_uses_saved_candidates(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            store = ExperimentStore(RewardAgentPaths.from_repo_root(repo_root))
            session = store.create_session({"objective": "manual queue"})
            store.save_candidates(session["id"], [{"id": "cand_1", "reward_overrides": {"v2_reward_scales": {"velocity_tracking": 5.0}}}])

            out = io.StringIO()
            with redirect_stdout(out):
                code = main(
                    [
                        "--repo-root",
                        str(repo_root),
                        "queue-trials",
                        "--session-id",
                        session["id"],
                        "--base-params-json",
                        "{\"task\":\"Template-Redrhex-Direct-v0\",\"num_envs\":4,\"max_iterations\":100,\"device\":\"cpu\"}",
                        "--max-iterations",
                        "5",
                        "--dry-run",
                    ]
                )

            self.assertEqual(code, 0)
            self.assertIn("dry_run_trials: 1", out.getvalue())
            self.assertEqual(store.load_trials(session["id"])[0]["status"], "dry_run")

    def test_queue_trials_launch_uses_registry_factory(self):
        class FakeRegistry:
            def __init__(self):
                self.params = []

            def queue_training(self, params):
                self.params.append(params)
                return {"id": "panel_1", "status": "queued"}

        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            store = ExperimentStore(RewardAgentPaths.from_repo_root(repo_root))
            session = store.create_session({"objective": "manual launch"})
            store.save_candidates(session["id"], [{"id": "cand_1", "reward_overrides": {"v2_reward_scales": {"velocity_tracking": 5.0}}}])
            registry = FakeRegistry()

            with redirect_stdout(io.StringIO()):
                code = main(
                    [
                        "--repo-root",
                        str(repo_root),
                        "queue-trials",
                        "--session-id",
                        session["id"],
                        "--base-params-json",
                        "{\"task\":\"Template-Redrhex-Direct-v0\",\"num_envs\":4,\"max_iterations\":100,\"device\":\"cpu\"}",
                        "--launch",
                    ],
                    registry_factory=lambda _repo_root: registry,
                )

            self.assertEqual(code, 0)
            self.assertEqual(registry.params[0].reward_preset_id, "cand_1")
            self.assertEqual(store.load_trials(session["id"])[0]["panel_run_id"], "panel_1")
```

- [x] **Step 2: Verify red**

Run: `python -m unittest tools.reward_agent.tests.test_cli -v`

Expected: FAIL because `queue-trials` is not a known command.

- [x] **Step 3: Implement CLI command**

Update `__main__.py` to import `preview_candidate_trials`, `queue_candidate_trials`, and `build_panel_registry`. Add a `queue-trials` parser with required mutually exclusive `--dry-run` / `--launch`, `--session-id`, `--base-params-json`, optional `--max-iterations`, optional `--limit`. In `main`, load candidates, parse base params JSON, call dry-run or launch path, and print `dry_run_trials: N` or `queued_trials: N`.

- [x] **Step 4: Verify green**

Run: `python -m unittest tools.reward_agent.tests.test_cli -v`

Expected: PASS.

### Task 4: Focused Verification and Commit

**Files:**
- Test all touched modules.

- [x] **Step 1: Run focused tests**

Run:

```bash
python -m unittest discover tools/reward_agent/tests -v
python -m unittest tools.training_panel.tests.test_reward_overrides tools.training_panel.tests.test_commands.CommandTests.test_training_params_accept_nested_v2_reward_overrides -v
python -m py_compile tools/reward_agent/agent.py tools/reward_agent/launcher.py tools/reward_agent/__main__.py
```

Expected: PASS.

- [x] **Step 2: Commit**

Stage only this milestone:

```bash
git add \
  docs/superpowers/plans/2026-06-19-reward-agent-cli-trial-launch.md \
  tools/reward_agent/__main__.py \
  tools/reward_agent/agent.py \
  tools/reward_agent/launcher.py \
  tools/reward_agent/tests/test_agent.py \
  tools/reward_agent/tests/test_cli.py \
  tools/reward_agent/tests/test_launcher.py
git commit -m "Add reward agent CLI trial launch"
```

## Self-Review Notes

- This plan adds a manual launch path but keeps launch explicit with `--launch`.
- The CLI defaults to no action unless the user chooses `--dry-run` or `--launch`.
- The implementation reuses existing Training Panel queueing through `ProcessRegistry`.
