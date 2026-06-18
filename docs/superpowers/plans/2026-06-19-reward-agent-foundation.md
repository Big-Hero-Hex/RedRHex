# Reward Agent Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build Milestone 1 of the Reward Agent Lab: nested active reward overrides, local reward-agent storage, and deterministic scoring for existing runs.

**Architecture:** Keep the first implementation small and testable. Add shared reward override helpers under the existing training panel package so both the panel and `scripts/rsl_rl/train.py` use the same behavior, then add a separate `tools.reward_agent` package for storage and evaluation without autonomous training.

**Tech Stack:** Python stdlib, `unittest`, existing Training Panel modules, TensorBoard `EventAccumulator` when installed.

## Global Constraints

- Keep the existing Training Panel focused on manual operations.
- Do not let the agent silently edit reward source code.
- Do not let the LLM decide success without metric-based scoring.
- Do not duplicate training launch, history, or TensorBoard parsing code already owned by the panel.
- New reward code proposal support is out of scope for this Milestone 1 plan.
- Autonomous training launch is out of scope for this Milestone 1 plan.

---

## File Structure

- `tools/training_panel/training_panel/reward_overrides.py`: new helper for normalizing, validating, and applying top-level and nested reward overrides.
- `tools/training_panel/training_panel/commands.py`: use reward override normalization in `TrainingParams.from_dict`.
- `scripts/rsl_rl/train.py`: delegate reward override application to the shared helper.
- `tools/training_panel/tests/test_reward_overrides.py`: unit tests for direct and nested reward override behavior.
- `tools/training_panel/tests/test_commands.py`: add a regression test that `TrainingParams` preserves nested `v2_reward_scales`.
- `tools/reward_agent/experiment_store.py`: persistent JSON/JSONL storage for sessions, trials, evaluations, proposals, and conversation.
- `tools/reward_agent/evaluator.py`: deterministic evaluator for existing run records and scalar dictionaries.
- `tools/reward_agent/__init__.py` and `tools/reward_agent/__main__.py`: importable package and minimal CLI entrypoint.
- `tools/reward_agent/tests/test_experiment_store.py`: storage persistence tests.
- `tools/reward_agent/tests/test_evaluator.py`: scoring and missing-metric tests.

### Task 1: Reward Override Helper

**Files:**
- Create: `tools/training_panel/training_panel/reward_overrides.py`
- Create: `tools/training_panel/tests/test_reward_overrides.py`

**Interfaces:**
- Produces: `normalize_reward_overrides(overrides: dict) -> dict`
- Produces: `apply_reward_overrides(env_cfg: object, overrides: dict) -> list[str]`
- Produces: `RewardOverrideError(ValueError)`

- [ ] **Step 1: Write the failing tests**

Add `tools/training_panel/tests/test_reward_overrides.py`:

```python
import unittest

from tools.training_panel.training_panel.reward_overrides import (
    RewardOverrideError,
    apply_reward_overrides,
    normalize_reward_overrides,
)


class DummyEnvCfg:
    rew_scale_alive = 0.5

    def __init__(self):
        self.v2_reward_scales = {
            "velocity_tracking": 4.0,
            "energy_per_distance": 0.001,
        }


class RewardOverrideTests(unittest.TestCase):
    def test_apply_direct_and_nested_v2_reward_scales(self):
        cfg = DummyEnvCfg()

        applied = apply_reward_overrides(
            cfg,
            {
                "rew_scale_alive": "0.25",
                "v2_reward_scales": {"velocity_tracking": "5.5"},
                "v2_reward_scales.energy_per_distance": "0.002",
            },
        )

        self.assertEqual(cfg.rew_scale_alive, 0.25)
        self.assertEqual(cfg.v2_reward_scales["velocity_tracking"], 5.5)
        self.assertEqual(cfg.v2_reward_scales["energy_per_distance"], 0.002)
        self.assertEqual(
            applied,
            [
                "rew_scale_alive=0.25",
                "v2_reward_scales.velocity_tracking=5.5",
                "v2_reward_scales.energy_per_distance=0.002",
            ],
        )

    def test_normalize_rejects_non_numeric_nested_value(self):
        with self.assertRaises(RewardOverrideError):
            normalize_reward_overrides({"v2_reward_scales": {"velocity_tracking": "fast"}})

    def test_unknown_top_level_key_is_ignored_not_created(self):
        cfg = DummyEnvCfg()

        applied = apply_reward_overrides(cfg, {"not_a_reward": 1.0})

        self.assertEqual(applied, [])
        self.assertFalse(hasattr(cfg, "not_a_reward"))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest tools.training_panel.tests.test_reward_overrides -v`

Expected: FAIL with `ModuleNotFoundError` for `tools.training_panel.training_panel.reward_overrides`.

- [ ] **Step 3: Write minimal implementation**

Create `tools/training_panel/training_panel/reward_overrides.py`:

```python
from __future__ import annotations

from collections.abc import Mapping
from typing import Any


class RewardOverrideError(ValueError):
    """Raised when reward override payloads are malformed."""


def _to_float(key: str, value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise RewardOverrideError(f"Reward override {key!r} must be numeric") from exc


def normalize_reward_overrides(overrides: Mapping[str, Any] | None) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for raw_key, raw_value in (overrides or {}).items():
        key = str(raw_key)
        if key == "v2_reward_scales":
            if not isinstance(raw_value, Mapping):
                raise RewardOverrideError("v2_reward_scales override must be an object")
            normalized[key] = {
                str(scale_key): _to_float(f"v2_reward_scales.{scale_key}", scale_value)
                for scale_key, scale_value in raw_value.items()
            }
            continue
        normalized[key] = _to_float(key, raw_value)
    return normalized


def _iter_overrides(overrides: Mapping[str, Any]) -> list[tuple[str, float]]:
    normalized = normalize_reward_overrides(overrides)
    flattened: list[tuple[str, float]] = []
    for key, value in normalized.items():
        if key == "v2_reward_scales":
            for nested_key, nested_value in value.items():
                flattened.append((f"v2_reward_scales.{nested_key}", nested_value))
        else:
            flattened.append((key, value))
    return flattened


def apply_reward_overrides(env_cfg: object, overrides: Mapping[str, Any] | None) -> list[str]:
    applied: list[str] = []
    for key, value in _iter_overrides(overrides or {}):
        if key.startswith("v2_reward_scales."):
            scale_name = key.split(".", 1)[1]
            current = getattr(env_cfg, "v2_reward_scales", None)
            if not isinstance(current, Mapping):
                continue
            merged = dict(current)
            merged[scale_name] = value
            setattr(env_cfg, "v2_reward_scales", merged)
            applied.append(f"{key}={value}")
            continue
        if hasattr(env_cfg, key):
            setattr(env_cfg, key, value)
            applied.append(f"{key}={value}")
    return applied
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m unittest tools.training_panel.tests.test_reward_overrides -v`

Expected: PASS.

### Task 2: Training Params and Train Script Integration

**Files:**
- Modify: `tools/training_panel/training_panel/commands.py`
- Modify: `scripts/rsl_rl/train.py`
- Modify: `tools/training_panel/tests/test_commands.py`

**Interfaces:**
- Consumes: `normalize_reward_overrides(overrides: dict) -> dict`
- Consumes: `apply_reward_overrides(env_cfg: object, overrides: dict) -> list[str]`
- Produces: `TrainingParams.reward_overrides` supports floats and nested `v2_reward_scales` dictionaries.

- [ ] **Step 1: Write the failing test**

Add this test to `CommandTests` in `tools/training_panel/tests/test_commands.py`:

```python
    def test_training_params_accept_nested_v2_reward_overrides(self):
        params = TrainingParams.from_dict(
            {
                "task": "Template-Redrhex-Direct-v0",
                "num_envs": 4,
                "max_iterations": 8,
                "device": "cuda:0",
                "reward_overrides": {
                    "rew_scale_alive": "0.25",
                    "v2_reward_scales": {
                        "velocity_tracking": "5.5",
                        "energy_per_distance": 0.002,
                    },
                },
            }
        )

        self.assertEqual(params.reward_overrides["rew_scale_alive"], 0.25)
        self.assertEqual(params.reward_overrides["v2_reward_scales"]["velocity_tracking"], 5.5)
        self.assertEqual(params.reward_overrides["v2_reward_scales"]["energy_per_distance"], 0.002)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest tools.training_panel.tests.test_commands.CommandTests.test_training_params_accept_nested_v2_reward_overrides -v`

Expected: FAIL with `TypeError` from trying to cast the nested dict to float.

- [ ] **Step 3: Update `TrainingParams.from_dict`**

In `tools/training_panel/training_panel/commands.py`, import the helper:

```python
from .reward_overrides import normalize_reward_overrides
```

Replace the `reward_overrides` assignment in `TrainingParams.from_dict` with:

```python
            reward_overrides=normalize_reward_overrides(raw_overrides),
```

- [ ] **Step 4: Update `scripts/rsl_rl/train.py`**

Replace the inline reward override application block with:

```python
    # Apply reward scale overrides written by the training panel or Reward Agent Lab (if any)
    _override_file = Path(__file__).parents[2] / "tools" / "training_panel" / "active_reward_override.json"
    if _override_file.exists():
        import json as _json
        from tools.training_panel.training_panel.reward_overrides import apply_reward_overrides

        _overrides = _json.loads(_override_file.read_text(encoding="utf-8"))
        _applied = apply_reward_overrides(env_cfg, _overrides)
        if _applied:
            print(f"[INFO] Training panel reward overrides applied: {', '.join(_applied)}")
```

- [ ] **Step 5: Run focused tests**

Run:

```bash
python -m unittest \
  tools.training_panel.tests.test_reward_overrides \
  tools.training_panel.tests.test_commands.CommandTests.test_training_params_accept_nested_v2_reward_overrides \
  -v
```

Expected: PASS.

### Task 3: Reward Agent Experiment Store

**Files:**
- Create: `tools/reward_agent/__init__.py`
- Create: `tools/reward_agent/experiment_store.py`
- Create: `tools/reward_agent/tests/test_experiment_store.py`

**Interfaces:**
- Produces: `RewardAgentPaths.from_repo_root(repo_root: Path) -> RewardAgentPaths`
- Produces: `ExperimentStore(paths: RewardAgentPaths)`
- Produces: `ExperimentStore.create_session(goal: dict) -> dict`
- Produces: `ExperimentStore.append_conversation(session_id: str, entry: dict) -> None`
- Produces: `ExperimentStore.save_trials(session_id: str, trials: list[dict]) -> None`
- Produces: `ExperimentStore.load_trials(session_id: str) -> list[dict]`

- [ ] **Step 1: Write the failing test**

Add `tools/reward_agent/tests/test_experiment_store.py`:

```python
import tempfile
import unittest
from pathlib import Path

from tools.reward_agent.experiment_store import ExperimentStore, RewardAgentPaths


class ExperimentStoreTests(unittest.TestCase):
    def test_create_session_persists_goal_and_index(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = RewardAgentPaths.from_repo_root(Path(tmp))
            store = ExperimentStore(paths)

            session = store.create_session({"objective": "improve diagonal", "baseline_run_id": "run_1"})

            self.assertEqual(session["goal"]["objective"], "improve diagonal")
            self.assertTrue((paths.session_dir(session["id"]) / "goal.json").is_file())
            self.assertEqual(store.list_sessions()[0]["id"], session["id"])

    def test_trials_and_conversation_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = RewardAgentPaths.from_repo_root(Path(tmp))
            store = ExperimentStore(paths)
            session = store.create_session({"objective": "reduce energy"})

            store.save_trials(session["id"], [{"id": "trial_1", "panel_run_id": "panel_1"}])
            store.append_conversation(session["id"], {"role": "user", "content": "focus on stability"})

            self.assertEqual(store.load_trials(session["id"])[0]["panel_run_id"], "panel_1")
            log_text = (paths.session_dir(session["id"]) / "conversation.jsonl").read_text(encoding="utf-8")
            self.assertIn("focus on stability", log_text)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest tools.reward_agent.tests.test_experiment_store -v`

Expected: FAIL with `ModuleNotFoundError` for `tools.reward_agent`.

- [ ] **Step 3: Write minimal implementation**

Create `tools/reward_agent/__init__.py`:

```python
"""Reward Agent Lab foundation package."""
```

Create `tools/reward_agent/experiment_store.py` with dataclass paths, atomic JSON writes, session index updates, trials load/save, and JSONL append exactly as exercised by the tests.

```python
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


SESSION_ID_RE = re.compile(r"[^A-Za-z0-9_.-]+")


def _timestamp_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def _safe_session_id(value: str) -> str:
    return SESSION_ID_RE.sub("_", value).strip("._") or f"session_{_timestamp_id()}"


def _read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return default


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


@dataclass(frozen=True)
class RewardAgentPaths:
    root: Path

    @classmethod
    def from_repo_root(cls, repo_root: Path) -> "RewardAgentPaths":
        return cls(Path(repo_root) / "logs" / "reward_agent")

    @property
    def sessions_file(self) -> Path:
        return self.root / "sessions.json"

    def session_dir(self, session_id: str) -> Path:
        return self.root / "sessions" / _safe_session_id(session_id)


class ExperimentStore:
    def __init__(self, paths: RewardAgentPaths):
        self.paths = paths

    def list_sessions(self) -> list[dict[str, Any]]:
        data = _read_json(self.paths.sessions_file, {"sessions": []})
        sessions = data.get("sessions") if isinstance(data, dict) else []
        return list(sessions) if isinstance(sessions, list) else []

    def create_session(self, goal: dict[str, Any]) -> dict[str, Any]:
        now = datetime.now().isoformat(timespec="seconds")
        session = {
            "id": f"session_{_timestamp_id()}",
            "created_at": now,
            "updated_at": now,
            "status": "idle",
            "goal": dict(goal),
        }
        session_dir = self.paths.session_dir(session["id"])
        _write_json(session_dir / "goal.json", session)
        index = {"sessions": [*self.list_sessions(), session]}
        _write_json(self.paths.sessions_file, index)
        return session

    def append_conversation(self, session_id: str, entry: dict[str, Any]) -> None:
        path = self.paths.session_dir(session_id) / "conversation.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"created_at": datetime.now().isoformat(timespec="seconds"), **entry}
        with path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(payload, sort_keys=True) + "\n")

    def save_trials(self, session_id: str, trials: list[dict[str, Any]]) -> None:
        _write_json(self.paths.session_dir(session_id) / "trials.json", {"trials": trials})

    def load_trials(self, session_id: str) -> list[dict[str, Any]]:
        data = _read_json(self.paths.session_dir(session_id) / "trials.json", {"trials": []})
        trials = data.get("trials") if isinstance(data, dict) else []
        return list(trials) if isinstance(trials, list) else []
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m unittest tools.reward_agent.tests.test_experiment_store -v`

Expected: PASS.

### Task 4: Deterministic Evaluator

**Files:**
- Create: `tools/reward_agent/evaluator.py`
- Create: `tools/reward_agent/tests/test_evaluator.py`

**Interfaces:**
- Produces: `EvaluationWeights`
- Produces: `evaluate_metrics(metrics: dict, baseline: dict | None = None, weights: EvaluationWeights | None = None) -> dict`

- [ ] **Step 1: Write the failing test**

Add `tools/reward_agent/tests/test_evaluator.py`:

```python
import unittest

from tools.reward_agent.evaluator import EvaluationWeights, evaluate_metrics


class EvaluatorTests(unittest.TestCase):
    def test_scores_complete_metrics_and_penalizes_regression(self):
        report = evaluate_metrics(
            {
                "command_tracking_score": 0.75,
                "skill_pass_score": 0.80,
                "stability_score": 0.90,
                "energy_penalty": 0.10,
                "fall_penalty": 0.05,
            },
            baseline={"overall_score": 2.50},
            weights=EvaluationWeights(regression_penalty=0.5),
        )

        self.assertTrue(report["complete"])
        self.assertLess(report["overall_score"], 2.50)
        self.assertIn("regression_penalty", report["components"])

    def test_missing_metrics_produce_incomplete_report(self):
        report = evaluate_metrics({"command_tracking_score": 0.5})

        self.assertFalse(report["complete"])
        self.assertIn("skill_pass_score", report["missing_metrics"])
        self.assertIsInstance(report["overall_score"], float)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest tools.reward_agent.tests.test_evaluator -v`

Expected: FAIL with `ModuleNotFoundError` for `tools.reward_agent.evaluator`.

- [ ] **Step 3: Write minimal implementation**

Create `tools/reward_agent/evaluator.py` with:

```python
from __future__ import annotations

from dataclasses import dataclass


REQUIRED_METRICS = (
    "command_tracking_score",
    "skill_pass_score",
    "stability_score",
    "energy_penalty",
    "fall_penalty",
)


@dataclass(frozen=True)
class EvaluationWeights:
    command_tracking_score: float = 1.0
    skill_pass_score: float = 1.0
    stability_score: float = 1.0
    energy_penalty: float = 1.0
    fall_penalty: float = 1.0
    regression_penalty: float = 1.0
```

Implement `evaluate_metrics` so it sums weighted positive scores, subtracts weighted penalties, subtracts a regression penalty when the score is below `baseline["overall_score"]`, and returns `complete`, `missing_metrics`, `components`, and `overall_score`.

```python
def _metric(metrics: dict, key: str) -> float:
    try:
        return float(metrics.get(key, 0.0))
    except (TypeError, ValueError):
        return 0.0


def evaluate_metrics(
    metrics: dict,
    baseline: dict | None = None,
    weights: EvaluationWeights | None = None,
) -> dict:
    weights = weights or EvaluationWeights()
    missing = [key for key in REQUIRED_METRICS if key not in metrics]
    components = {
        "command_tracking_score": _metric(metrics, "command_tracking_score") * weights.command_tracking_score,
        "skill_pass_score": _metric(metrics, "skill_pass_score") * weights.skill_pass_score,
        "stability_score": _metric(metrics, "stability_score") * weights.stability_score,
        "energy_penalty": -_metric(metrics, "energy_penalty") * weights.energy_penalty,
        "fall_penalty": -_metric(metrics, "fall_penalty") * weights.fall_penalty,
    }
    base_score = sum(components.values())
    regression = 0.0
    if baseline and "overall_score" in baseline:
        baseline_score = _metric(baseline, "overall_score")
        if base_score < baseline_score:
            regression = (baseline_score - base_score) * weights.regression_penalty
    components["regression_penalty"] = -regression
    return {
        "complete": not missing,
        "missing_metrics": missing,
        "components": components,
        "overall_score": round(sum(components.values()), 6),
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m unittest tools.reward_agent.tests.test_evaluator -v`

Expected: PASS.

### Task 5: Minimal CLI and Focused Verification

**Files:**
- Create: `tools/reward_agent/__main__.py`
- Test: focused unit test commands below

**Interfaces:**
- Consumes: `RewardAgentPaths.from_repo_root`
- Produces: `python -m tools.reward_agent --repo-root <path> status`

- [ ] **Step 1: Write the failing smoke command expectation**

Run: `python -m tools.reward_agent --help`

Expected before implementation: FAIL with no `__main__` module.

- [ ] **Step 2: Write minimal CLI implementation**

Create `tools/reward_agent/__main__.py` with argparse supporting:

```bash
python -m tools.reward_agent --repo-root /path/to/repo status
```

The `status` command should print the reward-agent root path and number of sessions.

```python
from __future__ import annotations

import argparse
from pathlib import Path

from .experiment_store import ExperimentStore, RewardAgentPaths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Reward Agent Lab foundation commands.")
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("status", help="Show reward-agent storage status.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    paths = RewardAgentPaths.from_repo_root(args.repo_root)
    store = ExperimentStore(paths)
    if args.command == "status":
        print(f"reward_agent_root: {paths.root}")
        print(f"sessions: {len(store.list_sessions())}")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 3: Run focused tests**

Run:

```bash
python -m unittest \
  tools.training_panel.tests.test_reward_overrides \
  tools.training_panel.tests.test_commands.CommandTests.test_training_params_accept_nested_v2_reward_overrides \
  tools.reward_agent.tests.test_experiment_store \
  tools.reward_agent.tests.test_evaluator \
  -v
```

Expected: PASS.

- [ ] **Step 4: Run CLI smoke check**

Run: `python -m tools.reward_agent --repo-root /tmp/redrhex-reward-agent-smoke status`

Expected: command exits 0 and prints `sessions: 0`.

- [ ] **Step 5: Commit the foundation changes**

Stage only files created or modified by this plan:

```bash
git add \
  docs/superpowers/plans/2026-06-19-reward-agent-foundation.md \
  scripts/rsl_rl/train.py \
  tools/training_panel/training_panel/commands.py \
  tools/training_panel/training_panel/reward_overrides.py \
  tools/training_panel/tests/test_commands.py \
  tools/training_panel/tests/test_reward_overrides.py \
  tools/reward_agent
git commit -m "Add reward agent foundation"
```

## Self-Review Notes

- This plan implements Milestone 1 only. Milestones 2-4 remain out of scope and should get separate plans.
- The plan covers nested `v2_reward_scales` override support, persistent reward-agent state, deterministic scoring, and dry CLI access.
- No task adds autonomous training or direct reward-code patch application.
