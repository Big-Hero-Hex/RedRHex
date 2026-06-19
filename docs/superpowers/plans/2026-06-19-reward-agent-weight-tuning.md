# Reward Agent Weight Tuning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build Milestone 2 of the Reward Agent Lab: generate bounded reward-weight candidates, queue short training trials through the existing panel backend, rank evaluated trials, and persist comparison reports.

**Architecture:** Keep this milestone headless and API-first. Add focused `planner`, `agent`, and `reports` modules under `tools.reward_agent`, extend the JSON store for candidates/evaluations/reports, and expose minimal CLI commands for creating sessions and proposing candidates.

**Tech Stack:** Python stdlib, `unittest`, existing `TrainingParams`, existing reward override normalization, existing `ExperimentStore`.

## Global Constraints

- Keep the existing Training Panel focused on manual operations.
- Do not let the agent silently edit reward source code.
- Do not let the LLM decide success without metric-based scoring.
- Do not duplicate training launch, history, or TensorBoard parsing code already owned by the panel.
- New reward code proposal support is out of scope for this Milestone 2 plan.
- Web UI integration is out of scope for this Milestone 2 plan.

---

## File Structure

- `tools/reward_agent/planner.py`: create bounded reward-weight candidate dictionaries.
- `tools/reward_agent/agent.py`: queue candidate trials through an injected process registry and persist trial records.
- `tools/reward_agent/reports.py`: rank evaluations and build a comparison report.
- `tools/reward_agent/experiment_store.py`: add candidates, evaluations, and report persistence helpers.
- `tools/reward_agent/__main__.py`: add `create-session` and `propose-candidates` CLI commands.
- `tools/reward_agent/tests/test_planner.py`: candidate generation tests.
- `tools/reward_agent/tests/test_agent.py`: queueing tests using a fake registry.
- `tools/reward_agent/tests/test_reports.py`: ranking/report tests.
- `tools/reward_agent/tests/test_experiment_store.py`: persistence extension tests.
- `tools/reward_agent/tests/test_cli.py`: CLI command tests.

### Task 1: Candidate Planner

**Files:**
- Create: `tools/reward_agent/planner.py`
- Create: `tools/reward_agent/tests/test_planner.py`

**Interfaces:**
- Produces: `RewardWeightSpec(name: str, minimum: float, maximum: float, multipliers: tuple[float, ...] = (0.8, 1.2), group: str = "v2_reward_scales")`
- Produces: `generate_weight_candidates(base_overrides: dict, specs: list[RewardWeightSpec], parent_candidate_id: str = "baseline") -> list[dict]`

- [ ] **Step 1: Write failing tests**

Create `tools/reward_agent/tests/test_planner.py`:

```python
import unittest

from tools.reward_agent.planner import RewardWeightSpec, generate_weight_candidates


class PlannerTests(unittest.TestCase):
    def test_generate_candidates_changes_one_weight_at_a_time_with_bounds(self):
        candidates = generate_weight_candidates(
            {"v2_reward_scales": {"velocity_tracking": 4.0, "energy_per_distance": 0.001}},
            [
                RewardWeightSpec(
                    "velocity_tracking",
                    minimum=3.5,
                    maximum=4.5,
                    multipliers=(0.5, 1.5),
                )
            ],
        )

        self.assertEqual([candidate["id"] for candidate in candidates], ["cand-001-velocity_tracking-x0_5", "cand-002-velocity_tracking-x1_5"])
        self.assertEqual(candidates[0]["reward_overrides"]["v2_reward_scales"]["velocity_tracking"], 3.5)
        self.assertEqual(candidates[1]["reward_overrides"]["v2_reward_scales"]["velocity_tracking"], 4.5)
        self.assertEqual(candidates[0]["reward_overrides"]["v2_reward_scales"]["energy_per_distance"], 0.001)
        self.assertEqual(candidates[0]["parent_candidate_id"], "baseline")
        self.assertIn("velocity_tracking", candidates[0]["hypothesis"])

    def test_generate_candidates_supports_top_level_reward_scales(self):
        candidates = generate_weight_candidates(
            {"rew_scale_alive": 0.5},
            [RewardWeightSpec("rew_scale_alive", minimum=0.1, maximum=1.0, multipliers=(1.2,), group="top_level")],
        )

        self.assertEqual(candidates[0]["reward_overrides"]["rew_scale_alive"], 0.6)
        self.assertEqual(candidates[0]["changed"]["rew_scale_alive"]["from"], 0.5)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Verify red**

Run: `python -m unittest tools.reward_agent.tests.test_planner -v`

Expected: FAIL with `ModuleNotFoundError` for `tools.reward_agent.planner`.

- [ ] **Step 3: Implement planner**

Create `tools/reward_agent/planner.py` with dataclass spec, deep-copy helpers, group-aware value reads/writes, deterministic candidate IDs, clamp behavior, and candidate dictionaries containing `id`, `parent_candidate_id`, `reward_overrides`, `changed`, `hypothesis`, and `risk_notes`.

- [ ] **Step 4: Verify green**

Run: `python -m unittest tools.reward_agent.tests.test_planner -v`

Expected: PASS.

### Task 2: Store Candidates, Evaluations, and Reports

**Files:**
- Modify: `tools/reward_agent/experiment_store.py`
- Modify: `tools/reward_agent/tests/test_experiment_store.py`

**Interfaces:**
- Produces: `ExperimentStore.save_candidates(session_id: str, candidates: list[dict]) -> None`
- Produces: `ExperimentStore.load_candidates(session_id: str) -> list[dict]`
- Produces: `ExperimentStore.save_evaluations(session_id: str, evaluations: list[dict]) -> None`
- Produces: `ExperimentStore.load_evaluations(session_id: str) -> list[dict]`
- Produces: `ExperimentStore.save_report(session_id: str, report_name: str, report: dict) -> Path`

- [ ] **Step 1: Write failing tests**

Add to `ExperimentStoreTests`:

```python
    def test_candidates_evaluations_and_report_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = RewardAgentPaths.from_repo_root(Path(tmp))
            store = ExperimentStore(paths)
            session = store.create_session({"objective": "improve yaw"})

            store.save_candidates(session["id"], [{"id": "cand_1"}])
            store.save_evaluations(session["id"], [{"candidate_id": "cand_1", "overall_score": 1.2}])
            report_path = store.save_report(session["id"], "comparison", {"best_candidate_id": "cand_1"})

            self.assertEqual(store.load_candidates(session["id"])[0]["id"], "cand_1")
            self.assertEqual(store.load_evaluations(session["id"])[0]["overall_score"], 1.2)
            self.assertTrue(report_path.is_file())
            self.assertIn("best_candidate_id", report_path.read_text(encoding="utf-8"))
```

- [ ] **Step 2: Verify red**

Run: `python -m unittest tools.reward_agent.tests.test_experiment_store.ExperimentStoreTests.test_candidates_evaluations_and_report_round_trip -v`

Expected: FAIL with `AttributeError` for `save_candidates`.

- [ ] **Step 3: Implement store methods**

Use existing `_write_json` and `_read_json`. Store candidates in `candidates.json`, evaluations in `evaluations.json`, and reports in `reports/<report_name>.json`.

- [ ] **Step 4: Verify green**

Run: `python -m unittest tools.reward_agent.tests.test_experiment_store -v`

Expected: PASS.

### Task 3: Queue Candidate Trials

**Files:**
- Create: `tools/reward_agent/agent.py`
- Create: `tools/reward_agent/tests/test_agent.py`

**Interfaces:**
- Produces: `queue_candidate_trials(store: ExperimentStore, session_id: str, process_registry: object, base_params: dict, candidates: list[dict], max_iterations: int | None = None) -> list[dict]`

- [ ] **Step 1: Write failing tests**

Create `tools/reward_agent/tests/test_agent.py`:

```python
import tempfile
import unittest
from pathlib import Path

from tools.reward_agent.agent import queue_candidate_trials
from tools.reward_agent.experiment_store import ExperimentStore, RewardAgentPaths


class FakeRegistry:
    def __init__(self):
        self.params = []

    def queue_training(self, params):
        self.params.append(params)
        return {"id": f"panel_{len(self.params)}", "status": "queued"}


class AgentTests(unittest.TestCase):
    def test_queue_candidate_trials_uses_training_params_and_persists_trials(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = ExperimentStore(RewardAgentPaths.from_repo_root(Path(tmp)))
            session = store.create_session({"objective": "improve diagonal"})
            registry = FakeRegistry()

            trials = queue_candidate_trials(
                store,
                session["id"],
                registry,
                {"task": "Template-Redrhex-Direct-v0", "num_envs": 4, "max_iterations": 100, "device": "cpu"},
                [{"id": "cand_1", "reward_overrides": {"v2_reward_scales": {"velocity_tracking": 5.0}}}],
                max_iterations=5,
            )

            self.assertEqual(trials[0]["panel_run_id"], "panel_1")
            self.assertEqual(trials[0]["candidate_id"], "cand_1")
            self.assertEqual(registry.params[0].max_iterations, 5)
            self.assertEqual(registry.params[0].reward_preset_id, "cand_1")
            self.assertEqual(registry.params[0].reward_overrides["v2_reward_scales"]["velocity_tracking"], 5.0)
            self.assertEqual(store.load_trials(session["id"])[0]["panel_run_id"], "panel_1")


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Verify red**

Run: `python -m unittest tools.reward_agent.tests.test_agent -v`

Expected: FAIL with `ModuleNotFoundError` for `tools.reward_agent.agent`.

- [ ] **Step 3: Implement queueing**

Create `tools/reward_agent/agent.py`. Convert `base_params` plus each candidate into `TrainingParams.from_dict`, set `reward_preset_id` to candidate ID, set `reward_overrides`, optionally override `max_iterations`, set `client_request_id` to `<session_id>:<candidate_id>`, call `process_registry.queue_training(params)`, and persist trials.

- [ ] **Step 4: Verify green**

Run: `python -m unittest tools.reward_agent.tests.test_agent -v`

Expected: PASS.

### Task 4: Rank Evaluations and Build Reports

**Files:**
- Create: `tools/reward_agent/reports.py`
- Create: `tools/reward_agent/tests/test_reports.py`

**Interfaces:**
- Produces: `rank_evaluations(evaluations: list[dict]) -> list[dict]`
- Produces: `build_comparison_report(session: dict, trials: list[dict], evaluations: list[dict]) -> dict`

- [ ] **Step 1: Write failing tests**

Create `tools/reward_agent/tests/test_reports.py`:

```python
import unittest

from tools.reward_agent.reports import build_comparison_report, rank_evaluations


class ReportTests(unittest.TestCase):
    def test_rank_evaluations_prefers_complete_high_scores(self):
        ranked = rank_evaluations(
            [
                {"candidate_id": "cand_low", "overall_score": 1.0, "complete": True},
                {"candidate_id": "cand_incomplete", "overall_score": 9.0, "complete": False},
                {"candidate_id": "cand_high", "overall_score": 2.0, "complete": True},
            ]
        )

        self.assertEqual([item["candidate_id"] for item in ranked], ["cand_high", "cand_low", "cand_incomplete"])

    def test_build_comparison_report_links_best_trial(self):
        report = build_comparison_report(
            {"id": "session_1", "goal": {"objective": "improve yaw"}},
            [{"candidate_id": "cand_high", "panel_run_id": "panel_1"}],
            [{"candidate_id": "cand_high", "overall_score": 2.0, "complete": True}],
        )

        self.assertEqual(report["best_candidate_id"], "cand_high")
        self.assertEqual(report["best_panel_run_id"], "panel_1")
        self.assertIn("improve yaw", report["summary"])


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Verify red**

Run: `python -m unittest tools.reward_agent.tests.test_reports -v`

Expected: FAIL with `ModuleNotFoundError` for `tools.reward_agent.reports`.

- [ ] **Step 3: Implement reports**

Create `tools/reward_agent/reports.py` with complete-first ranking by `overall_score` descending and report dictionaries containing `session_id`, `objective`, `best_candidate_id`, `best_panel_run_id`, `ranked_candidates`, and `summary`.

- [ ] **Step 4: Verify green**

Run: `python -m unittest tools.reward_agent.tests.test_reports -v`

Expected: PASS.

### Task 5: CLI for Session Creation and Candidate Proposal

**Files:**
- Modify: `tools/reward_agent/__main__.py`
- Create: `tools/reward_agent/tests/test_cli.py`

**Interfaces:**
- Produces: `python -m tools.reward_agent --repo-root <path> create-session --objective <text>`
- Produces: `python -m tools.reward_agent --repo-root <path> propose-candidates --session-id <id> --base-overrides-json <json> --scale <name:min:max>`

- [ ] **Step 1: Write failing tests**

Create `tools/reward_agent/tests/test_cli.py`:

```python
import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from tools.reward_agent.__main__ import main
from tools.reward_agent.experiment_store import ExperimentStore, RewardAgentPaths


class CliTests(unittest.TestCase):
    def test_create_session_and_propose_candidates(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            out = io.StringIO()
            with redirect_stdout(out):
                self.assertEqual(main(["--repo-root", str(repo_root), "create-session", "--objective", "improve diagonal"]), 0)
            session_id = out.getvalue().strip().split()[-1]

            with redirect_stdout(io.StringIO()):
                code = main(
                    [
                        "--repo-root",
                        str(repo_root),
                        "propose-candidates",
                        "--session-id",
                        session_id,
                        "--base-overrides-json",
                        "{\"v2_reward_scales\":{\"velocity_tracking\":4.0}}",
                        "--scale",
                        "velocity_tracking:2.0:8.0",
                    ]
                )

            self.assertEqual(code, 0)
            store = ExperimentStore(RewardAgentPaths.from_repo_root(repo_root))
            self.assertEqual(len(store.load_candidates(session_id)), 2)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Verify red**

Run: `python -m unittest tools.reward_agent.tests.test_cli -v`

Expected: FAIL because the CLI does not have `create-session`.

- [ ] **Step 3: Implement CLI commands**

Update `tools/reward_agent/__main__.py` to import `json`, `RewardWeightSpec`, and `generate_weight_candidates`. Add `create-session` and `propose-candidates` subcommands. Parse `--scale name:min:max` into `RewardWeightSpec(name, float(min), float(max))`; save generated candidates through the store and print the count.

- [ ] **Step 4: Run focused verification**

Run:

```bash
python -m unittest discover tools/reward_agent/tests -v
python -m unittest tools.training_panel.tests.test_reward_overrides tools.training_panel.tests.test_commands.CommandTests.test_training_params_accept_nested_v2_reward_overrides -v
python -m py_compile tools/reward_agent/planner.py tools/reward_agent/agent.py tools/reward_agent/reports.py tools/reward_agent/__main__.py
```

Expected: PASS.

- [ ] **Step 5: Commit**

Stage only files created or modified by this milestone:

```bash
git add \
  docs/superpowers/plans/2026-06-19-reward-agent-weight-tuning.md \
  tools/reward_agent/__main__.py \
  tools/reward_agent/agent.py \
  tools/reward_agent/experiment_store.py \
  tools/reward_agent/planner.py \
  tools/reward_agent/reports.py \
  tools/reward_agent/tests/test_agent.py \
  tools/reward_agent/tests/test_cli.py \
  tools/reward_agent/tests/test_experiment_store.py \
  tools/reward_agent/tests/test_planner.py \
  tools/reward_agent/tests/test_reports.py
git commit -m "Add reward agent weight tuning loop"
```

## Self-Review Notes

- This plan implements only Milestone 2.
- Code proposals, web UI, panel card integration, and autonomous long-running scheduling remain out of scope.
- The process registry is injected so unit tests do not start Isaac or require a GPU.
