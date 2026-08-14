from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import sqlite3
import sys
import tempfile
import unittest
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Mapping
from unittest import mock

from tools.training_panel.training_panel.autopilot import (
    DEFAULT_SKILL_GATES,
    FORWARD_FAST_TASK,
    TERMINAL_STATES,
    AgentDecisionV1,
    AutopilotValidationError,
    GoalSpecV1,
    RewardCatalogEntryV1,
    command_envelope_to_dict,
    compile_command_envelope,
    sha256_json,
)
from tools.training_panel.training_panel.autopilot_service import AutopilotService
from tools.training_panel.training_panel.autopilot_identity import (
    AUTOPILOT_CODE_IDENTITY_PATHS,
    dependency_manifest_sha256,
    source_code_identities,
)
from tools.training_panel.training_panel.autopilot_store import AutopilotConflictError
from tools.training_panel.training_panel.config import PanelPaths
from tools.training_panel.training_panel.processes import GpuHostLeaseBusy
from tools.training_panel.training_panel.physics import write_physics_profile


REWARD_VALUES = {
    "v2_reward_scales.forward_progress": 3.0,
    "v2_reward_scales.velocity_tracking": 6.0,
    "v2_reward_scales.axis_suppression": 2.0,
    "v2_reward_scales.height_maintain": 1.0,
    "v2_reward_scales.height_low_penalty": 1.5,
    "v2_reward_scales.leg_moving": 0.25,
    "v2_reward_scales.stall_penalty": -3.0,
    "v2_reward_scales.energy_per_distance": 0.0005,
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class FakeHistory:
    def __init__(self) -> None:
        self.runs: dict[str, dict[str, Any]] = {}

    def add(self, run: dict[str, Any]) -> dict[str, Any]:
        self.runs[str(run["id"])] = run
        return run

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        return self.runs.get(str(run_id))

    def list_runs(self) -> list[dict[str, Any]]:
        return list(reversed(list(self.runs.values())))


class FakeProcesses:
    COMMAND_FIELDS = [
        "command", "skill", "cmd_vx", "cmd_vy", "cmd_wz", "mae_vx", "mae_vy",
        "mae_wz", "actual_forward_speed_mean", "actual_lateral_leak_mean",
        "actual_yaw_leak_mean", "success_duration_s", "success_ratio",
        "success_vy_duration_s", "success_wz_duration_s", "diag_sign_match_ratio",
        "yaw_tilt_ok_ratio", "fall_rate", "energy_mech_power_main_mean",
        "energy_mech_power_total_mean", "energy_cost_of_transport_proxy",
        "energy_spring_energy_mean", "energy_spring_release_power_mean",
        "energy_spring_store_power_mean", "energy_spring_recovery_ratio",
        "energy_motion_speed_mean", "energy_progress_speed_mean", "energy_cost_mean",
        "energy_progress_distance_mean", "energy_per_distance", "energy_power_per_motion",
        "tracking_quality", "stability_quality", "score", "accept_pass",
    ]
    EPISODE_FIELDS = [
        "command", "skill", "environment_index", "episode_index", "complete",
        "sample_count", "fall_count", "mae_vx", "mae_vy", "mae_wz",
        "success_ratio", "energy_mech_power_total_mean", "energy_effort_mean",
    ]

    def __init__(self, root: Path, history: FakeHistory) -> None:
        self.root = root
        self.history = history
        self.training_params: list[Any] = []
        self.evaluation_params: list[Any] = []
        self.stopped: list[str] = []
        self._sequence = 0

    def _id(self, prefix: str) -> str:
        self._sequence += 1
        return f"{prefix}-{self._sequence}"

    def _profile(self, process_id: str, kind: str, value: Any) -> tuple[str, str]:
        path = self.root / "process_profiles" / f"{process_id}-{kind}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False),
            encoding="utf-8",
        )
        return str(path), _sha(path)

    def queue_training(self, params: Any) -> dict[str, Any]:
        params.validate()
        self.training_params.append(params)
        process_id = self._id("train")
        reward_file, reward_sha = self._profile(
            process_id, "reward", params.reward_overrides
        )
        terrain_file, terrain_sha = self._profile(
            process_id, "terrain", params.terrain_overrides
        )
        physics_file = None
        if params.physics_overrides:
            physics_path = self.root / "process_profiles" / f"{process_id}-physics.json"
            written = write_physics_profile(
                physics_path,
                profile_id=f"test-{process_id}",
                description="Autopilot test physics",
                values=params.physics_overrides,
            )
            physics_file = None if written is None else str(written)
        run = {
            "id": process_id,
            "source": "training_panel",
            "status": "queued",
            "params": asdict(params),
            "campaign_id": params.campaign_id,
            "campaign_trial_id": params.campaign_trial_id,
            "reward_profile_file": reward_file,
            "reward_profile_sha256": reward_sha,
            "terrain_profile_file": terrain_file,
            "terrain_profile_sha256": terrain_sha,
            "physics_profile_file": physics_file,
            "queued_at": _now(),
            "created_at": _now(),
        }
        return self.history.add(run)

    def start_evaluation(self, params: Any) -> dict[str, Any]:
        params.validate()
        self.evaluation_params.append(params)
        process_id = self._id("evaluation")
        run = {
            "id": process_id,
            "source": "autopilot_evaluation",
            "status": "running",
            "params": asdict(params),
            "campaign_id": params.campaign_id,
            "campaign_trial_id": params.campaign_trial_id,
            "started_at": _now(),
            "created_at": _now(),
        }
        return self.history.add(run)

    def stop(self, process_id: str) -> bool:
        self.stopped.append(process_id)
        run = self.history.get_run(process_id)
        if run:
            run["status"] = "stopped"
        return True

    def complete_training(self, process_id: str) -> dict[str, Any]:
        run = self.history.get_run(process_id)
        assert run is not None
        log_dir = self.root / "runs" / process_id
        final_iteration = int(run["params"]["max_iterations"]) - 1
        checkpoint = log_dir / f"model_{final_iteration}.pt"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_bytes(f"checkpoint:{process_id}".encode("utf-8"))
        run.update(
            {
                "status": "completed",
                "started_at": run.get("started_at") or _now(),
                "completed_at": _now(),
                "updated_at": _now(),
                "log_dir": str(log_dir),
                "latest_checkpoint": str(checkpoint),
                "output_checkpoint_path": str(checkpoint.resolve()),
                "output_checkpoint_iteration": final_iteration,
                "output_checkpoint_sha256": hashlib.sha256(
                    checkpoint.read_bytes()
                ).hexdigest(),
            }
        )
        return run

    @staticmethod
    def _command_row(
        command: dict[str, Any],
        *,
        passed: bool,
        tracking: float,
        fall_rate: float = 0.0,
    ) -> dict[str, Any]:
        vx, vy, wz = command["vx"], command["vy"], command["wz"]
        tracking_error = 1.0 - tracking
        mae_vx, mae_vy, mae_wz = 0.01, 0.01, 0.01
        if command["skill"] == "forward":
            mae_vx = abs(vx) * tracking_error
        elif command["skill"] == "lateral":
            mae_vy = abs(vy) * tracking_error
        elif command["skill"] == "diagonal":
            mae_vx = abs(vx) * tracking_error
            mae_vy = abs(vy) * tracking_error
        elif command["skill"] == "yaw":
            mae_wz = abs(wz) * tracking_error
        success_duration = 9.0 if passed else 1.0
        success_ratio = success_duration / 10.0
        stability = min(1.0, max(0.0, 1.0 - fall_rate / 0.20))
        accept_pass = success_duration >= 2.0 and fall_rate <= 0.20
        values = {
            "command": command["name"],
            "skill": command["skill"],
            "cmd_vx": vx,
            "cmd_vy": vy,
            "cmd_wz": wz,
            "mae_vx": mae_vx,
            "mae_vy": mae_vy,
            "mae_wz": mae_wz,
            "actual_forward_speed_mean": vx,
            "actual_lateral_leak_mean": 0.01,
            "actual_yaw_leak_mean": 0.01,
            "success_duration_s": success_duration,
            "success_ratio": success_ratio,
            "success_vy_duration_s": success_duration,
            "success_wz_duration_s": success_duration,
            "diag_sign_match_ratio": success_ratio,
            "yaw_tilt_ok_ratio": success_ratio,
            "fall_rate": fall_rate,
            "energy_mech_power_main_mean": 1.0,
            "energy_mech_power_total_mean": 1.0,
            "energy_cost_of_transport_proxy": 1.0,
            "energy_spring_energy_mean": 1.0,
            "energy_spring_release_power_mean": 1.0,
            "energy_spring_store_power_mean": 1.0,
            "energy_spring_recovery_ratio": 0.5,
            "energy_motion_speed_mean": 0.3,
            "energy_progress_speed_mean": 0.3,
            "energy_cost_mean": 1.0,
            "energy_progress_distance_mean": 1.0,
            "energy_per_distance": 1.0,
            "energy_power_per_motion": 1.0,
            "tracking_quality": tracking,
            "stability_quality": stability,
            "score": 0.8,
            "accept_pass": accept_pass,
        }
        return values

    @staticmethod
    def _episode_row(
        command: dict[str, Any],
        command_row: Mapping[str, Any],
        *,
        environment_index: int,
        fall_count: int = 0,
    ) -> dict[str, Any]:
        return {
            "command": command["name"],
            "skill": command["skill"],
            "environment_index": environment_index,
            "episode_index": 0,
            "complete": True,
            "sample_count": 100,
            "fall_count": fall_count,
            "mae_vx": command_row["mae_vx"],
            "mae_vy": command_row["mae_vy"],
            "mae_wz": command_row["mae_wz"],
            "success_ratio": command_row["success_ratio"],
            "energy_mech_power_total_mean": command_row["energy_mech_power_total_mean"],
            "energy_effort_mean": command_row["energy_per_distance"],
        }

    def complete_evaluation(
        self,
        process_id: str,
        *,
        passed: bool,
        tracking: float,
        fall_rate: float = 0.0,
        malformed: str | None = None,
        summary_overrides: dict[str, Any] | None = None,
        episode_mutator: Callable[[list[dict[str, Any]]], None] | None = None,
        command_mutator: Callable[[list[dict[str, Any]]], None] | None = None,
    ) -> dict[str, Any]:
        run = self.history.get_run(process_id)
        assert run is not None
        params = run["params"]
        profile = json.loads(Path(params["command_profile_file"]).read_text(encoding="utf-8"))
        commands = list(profile["commands"])
        command_rows = [
            self._command_row(
                command,
                passed=passed,
                tracking=tracking,
                fall_rate=fall_rate,
            )
            for command in commands
        ]
        num_envs = int(params["num_envs"])
        fall_environments = int(round(fall_rate * num_envs))
        if not math.isclose(
            fall_environments / float(num_envs), fall_rate, abs_tol=1.0e-12
        ):
            raise AssertionError("fake fall_rate must be exactly representable by num_envs")
        command_by_name = {row["command"]: row for row in command_rows}
        episode_rows = [
            self._episode_row(
                command,
                command_by_name[command["name"]],
                environment_index=environment_index,
                fall_count=int(environment_index < fall_environments),
            )
            for command in commands
            for environment_index in range(num_envs)
        ]
        for row in episode_rows:
            row["sample_count"] = int(params["sweep_steps"])
        if malformed == "nonfinite":
            command_rows[0]["tracking_quality"] = "nan"
        elif malformed == "missing_command":
            command_rows.pop()
        if command_mutator is not None:
            command_mutator(command_rows)
        if episode_mutator is not None:
            episode_mutator(episode_rows)

        output_dir = self.root / "evaluations" / process_id
        output_dir.mkdir(parents=True, exist_ok=True)
        command_csv = output_dir / "commands.csv"
        episode_csv = output_dir / "episodes.csv"
        summary_csv = output_dir / "summary.csv"
        with command_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.COMMAND_FIELDS)
            writer.writeheader()
            writer.writerows(command_rows)
        with episode_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.EPISODE_FIELDS)
            writer.writeheader()
            writer.writerows(episode_rows)

        physics_profile = (
            json.loads(Path(params["physics_profile_file"]).read_text(encoding="utf-8"))
            if params.get("physics_profile_file")
            else None
        )
        summary = {
            "evaluation.seed": params["seed"],
            "evaluation.num_envs": num_envs,
            "evaluation.sweep_steps": params["sweep_steps"],
            "evaluation.step_dt": params["expected_step_dt"],
            "evaluation.duration_s": params["sweep_steps"] * params["expected_step_dt"],
            "eval.profile": params["evaluation_profile"],
            "evaluation.agent_entry_point": params["agent_entry_point"],
            "command.profile_sha256": params["command_profile_sha256"],
            "checkpoint.path": params["checkpoint"],
            "checkpoint.sha256": params["checkpoint_sha256"],
            "checkpoint.strict_load": True,
            "energy.strict_evidence": True,
            "identity.code_sha256": params["code_sha256"],
            "identity.config_sha256": params["config_sha256"],
            "identity.dependency_sha256": params["dependency_sha256"],
            "identity.reward_profile_sha256": params["reward_profile_sha256"],
            "identity.physics.sha256": params["physics_identity_sha256"],
            "identity.spring.sha256": params["spring_identity_sha256"],
            "identity.terrain.sha256": params["terrain_profile_sha256"],
            "spring.backend": params["spring_backend"],
            "spring.calibration_status": "uncalibrated",
            "spring.checkpoint_calibration_status": "uncalibrated",
            "spring.profile_id": "" if physics_profile is None else physics_profile["profile_id"],
            "spring.profile_sha256": "" if physics_profile is None else sha256_json(physics_profile),
            "artifact.command_csv_sha256": _sha(command_csv),
            "artifact.episode_csv_sha256": _sha(episode_csv),
            "evidence.episode_row_count": len(episode_rows),
        }
        summary.update(summary_overrides or {})
        with summary_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["metric", "value"])
            writer.writeheader()
            writer.writerows(
                {"metric": key, "value": value} for key, value in summary.items()
            )
        run.update(
            {
                "status": "completed",
                "completed_at": _now(),
                "updated_at": _now(),
                "command_csv": str(command_csv),
                "episode_csv": str(episode_csv),
                "summary_csv": str(summary_csv),
                "command_csv_sha256": _sha(command_csv),
                "episode_csv_sha256": _sha(episode_csv),
                "summary_csv_sha256": _sha(summary_csv),
            }
        )
        return run


class AutopilotServiceIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.num_envs_patch = mock.patch.dict(
            os.environ, {"REDRHEX_AUTOPILOT_NUM_ENVS": "2"}
        )
        self.num_envs_patch.start()
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        for relative in AUTOPILOT_CODE_IDENTITY_PATHS:
            path = self.root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            content = f"fixture:{relative}\n"
            if relative.endswith("redrhex_env_cfg.py"):
                content += (
                    "v2_reward_scales = {\"forward_progress\": 3.0, "
                    "\"velocity_tracking\": 5.5}\n"
                )
            path.write_text(content, encoding="utf-8")
        self.paths = PanelPaths(
            repo_root=self.root,
            isaaclab_root=self.root / "isaaclab",
            isaacsim_root=self.root / "isaacsim",
            conda_sh=self.root / "conda.sh",
            conda_env="test",
        )
        self.paths.ensure_dirs()
        self.history = FakeHistory()
        self.processes = FakeProcesses(self.root, self.history)
        self.fake_dependency_manifest = {
            "schema_version": "redrhex.autopilot.dependency-manifest.v1",
            "python": {"implementation": "CPython", "version": "3.11.15"},
            "distributions": [{"name": "fixture", "version": "1.0"}],
            "simulator": {"components": []},
            "source_files": [],
        }

        def fake_runtime_identities(
            root: Path,
            **_kwargs: Any,
        ) -> tuple[dict[str, str], dict[str, Any]]:
            return (
                {
                    **source_code_identities(root),
                    "dependency": dependency_manifest_sha256(
                        self.fake_dependency_manifest
                    ),
                },
                dict(self.fake_dependency_manifest),
            )

        self.identity_patch = mock.patch(
            "tools.training_panel.training_panel.autopilot_service.runtime_source_identities",
            side_effect=fake_runtime_identities,
        )
        self.identity_patch.start()
        self.reward_patch = mock.patch(
            "tools.training_panel.training_panel.autopilot_service.reward_defaults",
            return_value=dict(REWARD_VALUES),
        )
        self.reward_patch.start()
        self.service = AutopilotService(
            self.paths,
            self.history,  # type: ignore[arg-type]
            self.processes,  # type: ignore[arg-type]
            enabled=True,
            start_worker=False,
            identity_python=sys.executable,
        )
        self.key_sequence = 0

    def tearDown(self) -> None:
        self.service.close()
        self.reward_patch.stop()
        self.identity_patch.stop()
        self.temporary.cleanup()
        self.num_envs_patch.stop()

    def key(self, prefix: str) -> str:
        self.key_sequence += 1
        return f"{prefix}-{self.key_sequence:04d}"

    def payload(
        self,
        *,
        initialization_mode: str = "fresh",
        baseline_run_id: str | None = None,
        baseline_checkpoint_iteration: int | None = None,
        checkpoint_sha256: str | None = None,
    ) -> dict[str, Any]:
        return {
            "schema_version": "redrhex.autopilot.goal.v1",
            "description": "Walk forward under deterministic evaluation.",
            "task": FORWARD_FAST_TASK,
            "stage": 1,
            "evaluation_profile": "stage1",
            "gait": "walk",
            "directions": ["forward"],
            "command_envelope": command_envelope_to_dict(
                compile_command_envelope(FORWARD_FAST_TASK, 1, "walk")
            ),
            "skill_gates": dict(DEFAULT_SKILL_GATES),
            "initialization_mode": initialization_mode,
            "baseline_run_id": baseline_run_id,
            "baseline_checkpoint_iteration": baseline_checkpoint_iteration,
            "checkpoint_sha256": checkpoint_sha256,
            "training_seeds": [42, 43, 44],
            "per_trial_iteration_cap": 10,
            "budget": {"max_training_trials": 24, "max_gpu_hours": 72.0},
            "tunable_reward_keys": list(REWARD_VALUES),
        }

    def create_and_arm(self, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        created = self.service.create_campaign(
            payload or self.payload(), idempotency_key=self.key("create")
        )
        return self.service.arm_campaign(
            created["id"],
            expected_revision=created["revision"],
            idempotency_key=self.key("arm"),
        )

    def pending_training(self, campaign_id: str) -> list[dict[str, Any]]:
        return [
            run for run in self.history.list_runs()
            if run.get("source") == "training_panel"
            and run.get("campaign_id") == campaign_id
            and run.get("status") in {"queued", "running"}
        ]

    def pending_evaluations(self, campaign_id: str) -> list[dict[str, Any]]:
        return [
            run for run in self.history.list_runs()
            if run.get("source") == "autopilot_evaluation"
            and run.get("campaign_id") == campaign_id
            and run.get("status") in {"queued", "running"}
        ]

    def complete_pending(
        self,
        campaign_id: str,
        outcome: Callable[[dict[str, Any]], tuple[bool, float]],
    ) -> None:
        for run in self.pending_training(campaign_id):
            self.processes.complete_training(run["id"])
        snapshot = self.service.get_campaign(campaign_id)
        trials = {trial["id"]: trial for trial in snapshot["candidate_lineage"]}
        for run in self.pending_evaluations(campaign_id):
            trial = trials[run["campaign_trial_id"]]
            passed, tracking = outcome(trial)
            self.processes.complete_evaluation(
                run["id"], passed=passed, tracking=tracking
            )

    def drive(
        self,
        campaign_id: str,
        terminal: set[str],
        outcome: Callable[[dict[str, Any]], tuple[bool, float]],
        *,
        limit: int = 80,
    ) -> dict[str, Any]:
        for _ in range(limit):
            self.service.tick()
            self.complete_pending(campaign_id, outcome)
            snapshot = self.service.get_campaign(campaign_id)
            if snapshot["state"] in terminal:
                return snapshot
        self.fail(f"campaign did not reach {terminal}: {snapshot['state']}")

    def reach_awaiting_advisor(self) -> dict[str, Any]:
        armed = self.create_and_arm()
        return self.drive(
            armed["id"],
            {"awaiting_advisor"},
            lambda trial: (False, 0.4),
        )

    def complete_with_evidence_mutation(
        self,
        *,
        episode_mutator: Callable[[list[dict[str, Any]]], None] | None = None,
        command_mutator: Callable[[list[dict[str, Any]]], None] | None = None,
    ) -> dict[str, Any]:
        armed = self.create_and_arm()
        self.service.tick()
        training = self.pending_training(armed["id"])[0]
        self.processes.complete_training(training["id"])
        self.service.tick()
        evaluation = self.pending_evaluations(armed["id"])[0]
        self.processes.complete_evaluation(
            evaluation["id"],
            passed=False,
            tracking=0.4,
            episode_mutator=episode_mutator,
            command_mutator=command_mutator,
        )
        self.service.tick()
        return self.service.get_campaign(armed["id"])

    def propose_candidate(
        self,
        snapshot: dict[str, Any],
        *,
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        key = "v2_reward_scales.forward_progress"
        decision = AgentDecisionV1(
            campaign_id=snapshot["id"],
            campaign_revision=snapshot["revision"],
            evidence_ids=(snapshot["evaluations"][-1]["id"],),
            hypothesis="The control undershoots forward commands.",
            action="propose_candidate",
            reward_key=key,
            proposed_value=3.3,
            expected_metric_effect="Improve forward command tracking.",
            rationale="Increase one bounded progress shaping term and re-evaluate.",
        )
        return self.service.submit_decision(
            snapshot["id"],
            decision.to_dict(),
            expected_revision=snapshot["revision"],
            idempotency_key=idempotency_key or self.key("decision"),
        )

    def test_draft_compiles_panel_identities_and_exact_command_artifact(self) -> None:
        payload = self.payload()
        payload["code_sha256"] = "a" * 64
        with self.assertRaisesRegex(AutopilotValidationError, "code_sha256"):
            self.service.create_campaign(payload, idempotency_key=self.key("invalid"))

        created = self.service.create_campaign(
            self.payload(), idempotency_key=self.key("create")
        )
        runtime = self.service.store.get_runtime(created["id"])
        command_file = Path(runtime["command_profile_file"])
        command_profile = json.loads(command_file.read_text(encoding="utf-8"))

        self.assertEqual(_sha(command_file), created["goal"]["command_profile_sha256"])
        self.assertEqual(command_file.stem, created["goal"]["command_profile_sha256"])
        self.assertEqual(command_profile["command_envelope"], created["goal"]["command_envelope"])
        self.assertEqual(command_profile["directions"], ["forward"])
        artifacts = self.service.list_artifacts(created["id"])
        self.assertEqual(
            sorted(item["kind"] for item in artifacts),
            ["command_profile", "dependency_manifest"],
        )
        dependency_artifact = next(
            item for item in artifacts if item["kind"] == "dependency_manifest"
        )
        self.assertEqual(
            dependency_artifact["sha256"], runtime["dependency_sha256"]
        )
        _artifact_metadata, dependency_content = self.service.store.get_artifact(
            created["id"], dependency_artifact["id"]
        )
        self.assertEqual(
            json.loads(dependency_content), runtime["dependency_manifest"]
        )

    def test_draft_persists_human_narrowed_reward_bounds(self) -> None:
        payload = self.payload()
        key = "v2_reward_scales.forward_progress"
        payload["tunable_reward_keys"] = [key]
        payload["reward_bounds"] = {key: [2.7, 3.3]}

        created = self.service.create_campaign(payload, idempotency_key=self.key("bounds"))

        self.assertEqual(len(created["reward_catalog"]), 1)
        self.assertEqual(created["reward_catalog"][0]["minimum"], 2.7)
        self.assertEqual(created["reward_catalog"][0]["maximum"], 3.3)

    def test_arm_reconciles_identity_artifacts_after_create_commit_crash(self) -> None:
        class SimulatedCrash(BaseException):
            pass

        with mock.patch.object(
            self.service,
            "_store_identity_artifacts",
            side_effect=SimulatedCrash("after campaign commit"),
        ):
            with self.assertRaises(SimulatedCrash):
                self.service.create_campaign(
                    self.payload(), idempotency_key=self.key("create-artifact-crash")
                )

        draft = self.service.list_campaigns()[0]
        self.assertEqual(self.service.list_artifacts(draft["id"]), [])
        self.service.close()
        self.service = AutopilotService(
            self.paths,
            self.history,  # type: ignore[arg-type]
            self.processes,  # type: ignore[arg-type]
            enabled=True,
            start_worker=False,
            identity_python=sys.executable,
        )
        armed = self.service.arm_campaign(
            draft["id"],
            expected_revision=draft["revision"],
            idempotency_key=self.key("arm-after-create-crash"),
        )

        self.assertEqual(armed["state"], "armed")
        self.assertEqual(
            {item["kind"] for item in self.service.list_artifacts(draft["id"])},
            {"dependency_manifest", "command_profile"},
        )

    def test_arm_reconciles_new_command_artifact_after_update_commit_crash(self) -> None:
        draft = self.service.create_campaign(
            self.payload(), idempotency_key=self.key("create-before-update-crash")
        )
        updated_payload = self.payload()
        updated_payload["gait"] = "run"
        updated_payload["command_envelope"] = command_envelope_to_dict(
            compile_command_envelope(FORWARD_FAST_TASK, 1, "run")
        )

        class SimulatedCrash(BaseException):
            pass

        with mock.patch.object(
            self.service,
            "_store_identity_artifacts",
            side_effect=SimulatedCrash("after draft update commit"),
        ):
            with self.assertRaises(SimulatedCrash):
                self.service.update_draft(
                    draft["id"],
                    updated_payload,
                    expected_revision=draft["revision"],
                    idempotency_key=self.key("update-artifact-crash"),
                )

        interrupted = self.service.get_campaign(draft["id"])
        new_command_sha = interrupted["goal"]["command_profile_sha256"]
        self.assertNotIn(
            ("command_profile", new_command_sha),
            {
                (item["kind"], item["sha256"])
                for item in self.service.list_artifacts(draft["id"])
            },
        )
        self.service.close()
        self.service = AutopilotService(
            self.paths,
            self.history,  # type: ignore[arg-type]
            self.processes,  # type: ignore[arg-type]
            enabled=True,
            start_worker=False,
            identity_python=sys.executable,
        )
        armed = self.service.arm_campaign(
            draft["id"],
            expected_revision=interrupted["revision"],
            idempotency_key=self.key("arm-after-update-crash"),
        )

        self.assertEqual(armed["state"], "armed")
        self.assertIn(
            ("command_profile", new_command_sha),
            {
                (item["kind"], item["sha256"])
                for item in self.service.list_artifacts(draft["id"])
            },
        )

    def test_rest_draft_rejects_missing_or_mistyped_required_contracts(self) -> None:
        invalid_cases = (
            ("budget", None, "budget must be an object"),
            ("skill_gates", [], "skill_gates must be an object"),
            ("directions", "forward", "directions must be an array"),
            ("command_envelope", None, "command_envelope must be an object"),
            ("directions", [], "directions must contain unique directions"),
            ("skill_gates", {}, "goal.skill_gates"),
            ("initialization_mode", "", "fresh or strict policy_only"),
        )
        for field, value, message in invalid_cases:
            with self.subTest(field=field):
                payload = self.payload()
                payload[field] = value
                with self.assertRaisesRegex(AutopilotValidationError, message):
                    self.service.create_campaign(payload, idempotency_key=self.key(f"bad-{field}"))
        missing = self.payload()
        del missing["tunable_reward_keys"]
        with self.assertRaisesRegex(AutopilotValidationError, "missing fields"):
            self.service.create_campaign(missing, idempotency_key=self.key("missing-tunables"))
        for field in (
            "baseline_run_id",
            "baseline_checkpoint_iteration",
            "checkpoint_sha256",
        ):
            with self.subTest(empty_fresh_identity=field):
                payload = self.payload()
                payload[field] = ""
                with self.assertRaisesRegex(AutopilotValidationError, "fresh initialization"):
                    self.service.create_campaign(
                        payload,
                        idempotency_key=self.key(f"empty-{field}"),
                    )

    def test_capabilities_publish_task_stage_reward_start_and_hard_bounds(self) -> None:
        capabilities = self.service.capabilities()
        entries = capabilities["reward_catalog"][FORWARD_FAST_TASK]["stage1"]
        forward = next(
            entry for entry in entries
            if entry["key"] == "v2_reward_scales.forward_progress"
        )

        self.assertEqual(forward["start_value"], 3.0)
        self.assertAlmostEqual(forward["minimum"], 2.4)
        self.assertAlmostEqual(forward["maximum"], 3.6)

    def test_policy_only_uses_exact_checkpoint_and_strict_frozen_inputs(self) -> None:
        checkpoint = self.root / "baseline" / "model_100.pt"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_bytes(b"baseline-policy")
        newer_checkpoint = checkpoint.with_name("model_200.pt")
        newer_checkpoint.write_bytes(b"newer-but-unselected-policy")
        checkpoint_sha = _sha(checkpoint)
        self.history.add(
            {
                "id": "baseline-run",
                "source": "training_panel",
                "status": "completed",
                "log_dir": str(checkpoint.parent),
                "latest_checkpoint": str(newer_checkpoint),
                "checkpoint_history": [
                    {"iteration": 100, "checkpoint": str(checkpoint)},
                    {"iteration": 200, "checkpoint": str(newer_checkpoint)},
                ],
                "params": {
                    "task": FORWARD_FAST_TASK,
                    "spring_backend": "native",
                    "num_envs": 16,
                    "device": "cpu",
                },
            }
        )
        wrong = self.payload(
            initialization_mode="policy_only",
            baseline_run_id="baseline-run",
            baseline_checkpoint_iteration=100,
            checkpoint_sha256="f" * 64,
        )
        with self.assertRaisesRegex(AutopilotValidationError, "no longer matches"):
            self.service.create_campaign(wrong, idempotency_key=self.key("wrong"))

        advertised = [
            item
            for item in self.service.capabilities()["baselines"]
            if item["run_id"] == "baseline-run"
        ]
        self.assertEqual(
            [(item["checkpoint_iteration"], item["checkpoint_sha256"]) for item in advertised],
            [(100, checkpoint_sha), (200, _sha(newer_checkpoint))],
        )

        armed = self.create_and_arm(
            self.payload(
                initialization_mode="policy_only",
                baseline_run_id="baseline-run",
                baseline_checkpoint_iteration=100,
                checkpoint_sha256=checkpoint_sha,
            )
        )
        self.service.tick()
        params = self.processes.training_params[-1]
        snapshot = self.service.get_campaign(armed["id"])
        trial = snapshot["candidate_lineage"][0]

        self.assertEqual(params.checkpoint, str(checkpoint.resolve()))
        self.assertEqual(params.checkpoint_sha256, checkpoint_sha)
        self.assertEqual(params.initialization_mode, "policy_only")
        self.assertTrue(params.strict_checkpoint_loading)
        self.assertEqual(params.curriculum_stage, 1)
        self.assertEqual(params.evaluation_profile, "stage1")
        self.assertEqual(params.seed, 42)
        self.assertEqual(params.max_iterations, 10)
        self.assertEqual(trial["source_checkpoint_sha256"], checkpoint_sha)
        self.assertEqual(trial["metadata"]["code_sha256"], snapshot["goal"]["code_sha256"])
        self.assertEqual(
            trial["metadata"]["command_profile_sha256"],
            snapshot["goal"]["command_profile_sha256"],
        )

    def _physics_baseline_payload(self) -> dict[str, Any]:
        checkpoint = self.root / "physics-baseline" / "model_100.pt"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_bytes(b"physics-baseline-policy")
        self.history.add(
            {
                "id": "physics-baseline-run",
                "source": "training_panel",
                "status": "completed",
                "latest_checkpoint": str(checkpoint),
                "physics_preset_id": "bench-measured",
                "physics_overrides": {
                    "simulation_physics.mass.scale": 1.04,
                    "simulation_physics.ground.static_friction": 1.3,
                },
                "params": {
                    "task": FORWARD_FAST_TASK,
                    "spring_backend": "native",
                    "num_envs": 2,
                    "device": "cuda:0",
                },
            }
        )
        return self.payload(
            initialization_mode="policy_only",
            baseline_run_id="physics-baseline-run",
            baseline_checkpoint_iteration=100,
            checkpoint_sha256=_sha(checkpoint),
        )

    def test_v1_rejects_nondefault_terrain_instead_of_false_attestation(self) -> None:
        checkpoint = self.root / "terrain-baseline" / "model_100.pt"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_bytes(b"terrain-baseline-policy")
        self.history.add(
            {
                "id": "terrain-baseline-run",
                "source": "training_panel",
                "status": "completed",
                "latest_checkpoint": str(checkpoint),
                "terrain_overrides": {"terrain.terrain_type": "generator"},
                "params": {
                    "task": FORWARD_FAST_TASK,
                    "spring_backend": "native",
                    "num_envs": 16,
                    "device": "cpu",
                },
            }
        )
        payload = self.payload(
            initialization_mode="policy_only",
            baseline_run_id="terrain-baseline-run",
            baseline_checkpoint_iteration=100,
            checkpoint_sha256=_sha(checkpoint),
        )

        with self.assertRaisesRegex(
            AutopilotValidationError, "default terrain profile"
        ):
            self.service.create_campaign(
                payload, idempotency_key=self.key("terrain-rejected")
            )

    def test_nonempty_physics_profile_is_bound_and_parsed_without_fallback(self) -> None:
        armed = self.create_and_arm(self._physics_baseline_payload())
        self.service.tick()
        training = self.pending_training(armed["id"])[0]
        self.assertTrue(training["physics_profile_file"])
        self.processes.complete_training(training["id"])
        self.service.tick()
        evaluation = self.pending_evaluations(armed["id"])[0]
        self.processes.complete_evaluation(
            evaluation["id"], passed=False, tracking=0.4
        )
        self.service.tick()

        final = self.service.get_campaign(armed["id"])
        self.assertEqual(final["state"], "awaiting_advisor")
        self.assertEqual(len(final["evaluations"]), 1)

    def test_mutated_physics_profile_is_rejected_before_evaluation(self) -> None:
        armed = self.create_and_arm(self._physics_baseline_payload())
        self.service.tick()
        training = self.pending_training(armed["id"])[0]
        self.processes.complete_training(training["id"])
        physics_file = Path(str(training["physics_profile_file"]))
        physics_file.write_bytes(physics_file.read_bytes() + b"\n")

        self.service.tick()

        final = self.service.get_campaign(armed["id"])
        self.assertEqual(final["state"], "blocked_safety")
        self.assertIn("physics profile changed", final["terminal_reason"])

    def test_control_train_evaluate_awaits_advisor_and_candidate_is_one_change(self) -> None:
        snapshot = self.reach_awaiting_advisor()
        self.assertEqual(len(snapshot["candidate_lineage"]), 1)
        self.assertEqual(len(snapshot["evaluations"]), 1)
        self.assertFalse(snapshot["evaluations"][0]["ranking"]["eligible"])
        episode_index = snapshot["evaluations"][0]["episode_metrics"]
        self.assertEqual(len(episode_index), 1)
        self.assertEqual(
            episode_index[0]["schema_version"],
            "redrhex.autopilot.episode-evidence-index.v1",
        )
        self.assertGreater(episode_index[0]["row_count"], 0)

        self.propose_candidate(snapshot)
        self.service.tick()
        current = self.service.get_campaign(snapshot["id"])
        candidate = current["candidate_lineage"][-1]
        changed = [
            key for key, value in candidate["reward_profile"].items()
            if value != REWARD_VALUES[key]
        ]
        self.assertEqual(changed, ["v2_reward_scales.forward_progress"])
        self.assertEqual(current["state"], "candidate_training")
        self.assertEqual(len(self.pending_training(snapshot["id"])), 1)

    def test_decision_context_exposes_finite_moves_and_leader_delta(self) -> None:
        awaiting = self.reach_awaiting_advisor()
        initial_context = self.service.decision_context(awaiting["id"])

        self.assertEqual(initial_context["remaining_allowable_move_count"], 32)
        self.assertEqual(len(initial_context["remaining_allowable_moves"]), 32)
        self.assertEqual(initial_context["attempted_moves"], [])
        self.assertEqual(
            initial_context["campaign_start_reward_values"], REWARD_VALUES
        )

        off_lattice = AgentDecisionV1(
            campaign_id=awaiting["id"],
            campaign_revision=awaiting["revision"],
            evidence_ids=(awaiting["evaluations"][-1]["id"],),
            hypothesis="Try an arbitrary bounded progress value.",
            action="propose_candidate",
            reward_key="v2_reward_scales.forward_progress",
            proposed_value=3.2,
            expected_metric_effect="Change tracking.",
            rationale="This is intentionally not a V1 lattice point.",
        )
        with self.assertRaisesRegex(AutopilotValidationError, "finite approved"):
            self.service.submit_decision(
                awaiting["id"],
                off_lattice.to_dict(),
                expected_revision=awaiting["revision"],
                idempotency_key=self.key("off-lattice"),
            )
        self.assertEqual(
            self.service.get_campaign(awaiting["id"])["revision"],
            awaiting["revision"],
        )

        self.propose_candidate(awaiting)
        updated = self.drive(
            awaiting["id"],
            {"awaiting_advisor"},
            lambda trial: (False, 0.9),
        )
        context = self.service.decision_context(updated["id"])

        self.assertAlmostEqual(
            context["baseline_to_leader_reward_deltas"][
                "v2_reward_scales.forward_progress"
            ],
            0.3,
        )
        self.assertEqual(len(context["attempted_moves"]), 1)
        self.assertFalse(
            any(
                move["reward_key"] == "v2_reward_scales.forward_progress"
                and math.isclose(move["proposed_value"], 3.3)
                for move in context["remaining_allowable_moves"]
            )
        )

    def test_eligible_move_check_uses_full_lattice_remainder(self) -> None:
        snapshot = self.reach_awaiting_advisor()
        context = self.service.decision_context(snapshot["id"])
        endpoint_decisions = []
        all_move_decisions = []
        for constraint in context["constraints"]:
            current = constraint["current_value"]
            values = [
                value for value in constraint["lattice_values"]
                if not math.isclose(value, current)
            ]
            for value in (values[0], values[-1]):
                endpoint_decisions.append(
                    {
                        "action": "propose_candidate",
                        "reward_key": constraint["reward_key"],
                        "proposed_value": value,
                    }
                )
            for value in values:
                all_move_decisions.append(
                    {
                        "action": "propose_candidate",
                        "reward_key": constraint["reward_key"],
                        "proposed_value": value,
                    }
                )

        endpoint_snapshot = {**snapshot, "decisions": endpoint_decisions}
        exhausted_snapshot = {**snapshot, "decisions": all_move_decisions}

        self.assertTrue(self.service._has_eligible_move(endpoint_snapshot))
        self.assertFalse(self.service._has_eligible_move(exhausted_snapshot))

    def test_duplicate_candidate_decision_returns_current_result_without_duplicate(self) -> None:
        snapshot = self.reach_awaiting_advisor()
        key = self.key("duplicate-decision")
        first = self.propose_candidate(snapshot, idempotency_key=key)
        repeated = self.propose_candidate(snapshot, idempotency_key=key)

        self.assertEqual(repeated, first)
        self.service.tick()
        self.service.tick()
        current = self.service.get_campaign(snapshot["id"])
        self.assertEqual(
            len([trial for trial in current["candidate_lineage"] if trial["kind"] == "candidate"]),
            1,
        )
        self.assertEqual(len(self.pending_training(snapshot["id"])), 1)

    def test_safe_better_partial_candidate_advances_accumulated_leader(self) -> None:
        awaiting = self.reach_awaiting_advisor()
        self.propose_candidate(awaiting)
        partial = self.drive(
            awaiting["id"],
            {"awaiting_advisor"},
            lambda trial: (False, 0.9),
        )
        first_candidate = next(
            trial for trial in partial["candidate_lineage"]
            if trial["kind"] == "candidate"
        )
        self.assertEqual(partial["leader"]["trial_id"], first_candidate["id"])
        self.assertFalse(partial["evaluations"][-1]["ranking"]["eligible"])

        second = AgentDecisionV1(
            campaign_id=partial["id"],
            campaign_revision=partial["revision"],
            evidence_ids=(partial["evaluations"][-1]["id"],),
            hypothesis="Tracking still misses the evaluator duration gate.",
            action="propose_candidate",
            reward_key="v2_reward_scales.velocity_tracking",
            proposed_value=6.6,
            expected_metric_effect="Improve command tracking duration.",
            rationale="Retain the safe partial leader and change one additional bounded term.",
        )
        self.service.submit_decision(
            partial["id"],
            second.to_dict(),
            expected_revision=partial["revision"],
            idempotency_key=self.key("second-decision"),
        )
        self.service.tick()
        current = self.service.get_campaign(partial["id"])
        newest = current["candidate_lineage"][-1]
        self.assertEqual(newest["reward_profile"]["v2_reward_scales.forward_progress"], 3.3)
        self.assertEqual(newest["reward_profile"]["v2_reward_scales.velocity_tracking"], 6.6)

        second_complete = self.drive(
            partial["id"],
            {"awaiting_advisor"},
            lambda trial: (False, 0.95),
        )
        candidates = [
            trial for trial in second_complete["candidate_lineage"]
            if trial["kind"] == "candidate"
        ]
        self.assertEqual(len(candidates), 2)
        self.assertEqual(len(second_complete["evaluations"]), 3)

        # Reverting the second key recreates the first profile. The controller
        # rejects it before recording or launching instead of replaying old
        # evidence or consuming a hidden/unaccounted trial.
        third = AgentDecisionV1(
            campaign_id=second_complete["id"],
            campaign_revision=second_complete["revision"],
            evidence_ids=(second_complete["evaluations"][-1]["id"],),
            hypothesis="The second shaping change increased energy without enough tracking gain.",
            action="propose_candidate",
            reward_key="v2_reward_scales.velocity_tracking",
            proposed_value=6.0,
            expected_metric_effect="Restore the lower-energy first candidate profile.",
            rationale="Re-evaluate the prior values under a new immutable trial identity.",
        )
        with self.assertRaisesRegex(AutopilotConflictError, "already been attempted"):
            self.service.submit_decision(
                second_complete["id"],
                third.to_dict(),
                expected_revision=second_complete["revision"],
                idempotency_key=self.key("third-decision"),
            )
        replay_guard = self.service.get_campaign(second_complete["id"])
        candidate_trials = [
            trial for trial in replay_guard["candidate_lineage"]
            if trial["kind"] == "candidate"
        ]
        self.assertEqual(len(candidate_trials), 2)
        self.assertEqual(replay_guard["revision"], second_complete["revision"])

    def test_safety_gate_failure_cannot_become_leader(self) -> None:
        awaiting = self.reach_awaiting_advisor()
        baseline_leader = awaiting["leader"]["trial_id"]
        self.propose_candidate(awaiting)
        self.service.tick()
        training = self.pending_training(awaiting["id"])[0]
        self.processes.complete_training(training["id"])
        self.service.tick()
        evaluation = self.pending_evaluations(awaiting["id"])[0]
        self.processes.complete_evaluation(
            evaluation["id"],
            passed=False,
            tracking=0.99,
            fall_rate=0.5,
        )
        self.service.tick()
        current = self.service.get_campaign(awaiting["id"])

        self.assertEqual(current["state"], "awaiting_advisor")
        self.assertEqual(current["leader"]["trial_id"], baseline_leader)
        self.assertFalse(current["evaluations"][-1]["hard_gates"]["fall_rate"])

    def test_safe_partial_candidate_outranks_unsafe_high_scoring_control(self) -> None:
        armed = self.create_and_arm()
        self.service.tick()
        control_training = self.pending_training(armed["id"])[0]
        self.processes.complete_training(control_training["id"])
        self.service.tick()
        control_evaluation = self.pending_evaluations(armed["id"])[0]
        self.processes.complete_evaluation(
            control_evaluation["id"],
            passed=True,
            tracking=0.99,
            command_mutator=lambda rows: [row.__setitem__("mae_vy", 0.3) for row in rows],
            episode_mutator=lambda rows: [row.__setitem__("mae_vy", 0.3) for row in rows],
        )
        self.service.tick()
        awaiting = self.service.get_campaign(armed["id"])
        control = awaiting["evaluations"][-1]
        self.assertEqual(awaiting["state"], "awaiting_advisor")
        self.assertFalse(control["hard_gates"]["linear_leak"])

        self.propose_candidate(awaiting)
        self.service.tick()
        candidate_training = self.pending_training(armed["id"])[0]
        self.processes.complete_training(candidate_training["id"])
        self.service.tick()
        candidate_evaluation = self.pending_evaluations(armed["id"])[0]
        self.processes.complete_evaluation(
            candidate_evaluation["id"], passed=False, tracking=0.9
        )
        self.service.tick()

        current = self.service.get_campaign(armed["id"])
        candidate_trial = current["candidate_lineage"][-1]
        candidate_report = current["evaluations"][-1]
        self.assertTrue(self.service._safety_gates_pass(candidate_report))
        self.assertGreater(
            control["ranking"]["passed_gate_count"],
            candidate_report["ranking"]["passed_gate_count"],
        )
        self.assertEqual(current["leader"]["trial_id"], candidate_trial["id"])

    def test_restart_reuses_bound_training_without_duplicate_launch(self) -> None:
        armed = self.create_and_arm()
        self.service.tick()
        self.assertEqual(len(self.pending_training(armed["id"])), 1)
        self.service.close()
        self.service = AutopilotService(
            self.paths,
            self.history,  # type: ignore[arg-type]
            self.processes,  # type: ignore[arg-type]
            enabled=True,
            start_worker=False,
            identity_python=sys.executable,
        )

        self.service.tick()
        self.service.tick()

        self.assertEqual(len(self.pending_training(armed["id"])), 1)
        self.assertEqual(
            len(
                [
                    run for run in self.history.list_runs()
                    if run.get("source") == "training_panel"
                    and run.get("campaign_id") == armed["id"]
                ]
            ),
            1,
        )

    def test_controller_does_not_lose_active_campaign_behind_newer_drafts(self) -> None:
        armed = self.create_and_arm()
        self.service.tick()
        training = self.pending_training(armed["id"])[0]
        active = self.service.get_campaign(armed["id"])
        goal = GoalSpecV1.from_dict(active["goal"])
        catalog = tuple(
            RewardCatalogEntryV1.from_dict(item)
            for item in active["reward_catalog"]
        )
        for index in range(101):
            self.service.store.create_campaign(
                goal,
                catalog,
                idempotency_key=f"newer-draft-{index:03d}",
                runtime={},
            )
        self.processes.complete_training(training["id"])

        self.service.tick()

        current = self.service.get_campaign(armed["id"])
        self.assertEqual(current["candidate_lineage"][0]["status"], "evaluating")
        self.assertEqual(len(self.pending_evaluations(armed["id"])), 1)

    def test_restart_finishes_persisted_evaluation_transition_without_duplicate(self) -> None:
        armed = self.create_and_arm()
        self.service.tick()
        training = self.pending_training(armed["id"])[0]
        self.processes.complete_training(training["id"])
        self.service.tick()
        evaluation = self.pending_evaluations(armed["id"])[0]
        self.processes.complete_evaluation(
            evaluation["id"], passed=False, tracking=0.4
        )
        snapshot = self.service.get_campaign(armed["id"])
        runtime = self.service.store.get_runtime(armed["id"])

        class SimulatedCrash(BaseException):
            pass

        with mock.patch.object(
            self.service,
            "_after_evaluation",
            side_effect=SimulatedCrash("after evaluation commit"),
        ):
            with self.assertRaises(SimulatedCrash):
                self.service._poll_evaluation_state(snapshot, runtime)
        persisted = self.service.get_campaign(armed["id"])
        self.assertEqual(persisted["state"], "control_evaluating")
        self.assertEqual(persisted["candidate_lineage"][0]["status"], "evaluated")

        self.service.close()
        self.service = AutopilotService(
            self.paths,
            self.history,  # type: ignore[arg-type]
            self.processes,  # type: ignore[arg-type]
            enabled=True,
            start_worker=False,
            identity_python=sys.executable,
        )
        self.service.tick()
        recovered = self.service.get_campaign(armed["id"])
        self.assertEqual(recovered["state"], "awaiting_advisor")
        self.assertEqual(len(recovered["evaluations"]), 1)
        self.assertEqual(len(self.processes.evaluation_params), 1)

    def test_restart_rejects_corrupt_evidence_before_delayed_outcome(self) -> None:
        armed = self.create_and_arm()
        self.service.tick()
        training = self.pending_training(armed["id"])[0]
        self.processes.complete_training(training["id"])
        self.service.tick()
        evaluation = self.pending_evaluations(armed["id"])[0]
        self.processes.complete_evaluation(
            evaluation["id"], passed=False, tracking=0.4
        )
        snapshot = self.service.get_campaign(armed["id"])
        runtime = self.service.store.get_runtime(armed["id"])

        class SimulatedCrash(BaseException):
            pass

        with mock.patch.object(
            self.service,
            "_after_evaluation",
            side_effect=SimulatedCrash("after evaluation commit"),
        ):
            with self.assertRaises(SimulatedCrash):
                self.service._poll_evaluation_state(snapshot, runtime)
        persisted = self.service.get_campaign(armed["id"])
        artifact_id = persisted["evaluations"][0]["artifact_ids"][0]
        artifact, _content = self.service.get_artifact(armed["id"], artifact_id)
        artifact_path = (
            self.paths.autopilot_artifact_dir
            / artifact["sha256"][:2]
            / artifact["sha256"]
        )
        artifact_path.write_bytes(b"corrupt evidence")

        self.service.close()
        self.service = AutopilotService(
            self.paths,
            self.history,  # type: ignore[arg-type]
            self.processes,  # type: ignore[arg-type]
            enabled=True,
            start_worker=False,
            identity_python=sys.executable,
        )
        self.service.tick()

        blocked = self.service.get_campaign(armed["id"])
        self.assertEqual(blocked["state"], "blocked_safety")
        self.assertIn("missing or corrupt", blocked["terminal_reason"])

    def test_restart_rejects_sqlite_report_tampering_against_immutable_json(self) -> None:
        armed = self.create_and_arm()
        self.service.tick()
        training = self.pending_training(armed["id"])[0]
        self.processes.complete_training(training["id"])
        self.service.tick()
        evaluation = self.pending_evaluations(armed["id"])[0]
        self.processes.complete_evaluation(
            evaluation["id"], passed=False, tracking=0.4
        )
        snapshot = self.service.get_campaign(armed["id"])
        runtime = self.service.store.get_runtime(armed["id"])

        class SimulatedCrash(BaseException):
            pass

        with mock.patch.object(
            self.service,
            "_after_evaluation",
            side_effect=SimulatedCrash("after evaluation commit"),
        ):
            with self.assertRaises(SimulatedCrash):
                self.service._poll_evaluation_state(snapshot, runtime)

        connection = sqlite3.connect(self.paths.autopilot_db_file)
        try:
            row = connection.execute(
                "SELECT id, report_json FROM campaign_evaluations WHERE campaign_id=?",
                (armed["id"],),
            ).fetchone()
            assert row is not None
            tampered = json.loads(row[1])
            tampered["ranking"]["eligible"] = True
            tampered["ranking"]["mean_tracking_quality"] = 1.0
            connection.execute(
                "UPDATE campaign_evaluations SET report_json=? WHERE id=?",
                (json.dumps(tampered, sort_keys=True), row[0]),
            )
            connection.commit()
        finally:
            connection.close()

        self.service.close()
        self.service = AutopilotService(
            self.paths,
            self.history,  # type: ignore[arg-type]
            self.processes,  # type: ignore[arg-type]
            enabled=True,
            start_worker=False,
            identity_python=sys.executable,
        )
        self.service.tick()

        blocked = self.service.get_campaign(armed["id"])
        self.assertEqual(blocked["state"], "blocked_safety")
        self.assertIn("immutable evaluation report", blocked["terminal_reason"])

    def test_restart_finishes_candidate_confirmation_outcome_after_runtime_commit(self) -> None:
        awaiting = self.reach_awaiting_advisor()
        self.propose_candidate(awaiting)
        self.service.tick()
        training = self.pending_training(awaiting["id"])[0]
        self.processes.complete_training(training["id"])
        self.service.tick()
        evaluation = self.pending_evaluations(awaiting["id"])[0]
        self.processes.complete_evaluation(
            evaluation["id"], passed=True, tracking=0.9
        )

        class SimulatedCrash(BaseException):
            pass

        original_transition = self.service.store.transition_campaign

        def crash_before_confirming(campaign_id, target, **kwargs):
            if target == "confirming":
                raise SimulatedCrash("after confirmation_started")
            return original_transition(campaign_id, target, **kwargs)

        with mock.patch.object(
            self.service.store,
            "transition_campaign",
            side_effect=crash_before_confirming,
        ):
            with self.assertRaises(SimulatedCrash):
                self.service.tick()

        interrupted = self.service.get_campaign(awaiting["id"])
        self.assertEqual(interrupted["state"], "candidate_evaluating")
        self.assertTrue(
            self.service.store.get_runtime(awaiting["id"])[
                "confirmation_winner_trial_id"
            ]
        )

        self.service.close()
        self.service = AutopilotService(
            self.paths,
            self.history,  # type: ignore[arg-type]
            self.processes,  # type: ignore[arg-type]
            enabled=True,
            start_worker=False,
            identity_python=sys.executable,
        )
        self.service.tick()
        recovered = self.service.get_campaign(awaiting["id"])
        self.assertEqual(recovered["state"], "confirming")

        self.service.tick()
        confirmation_training = self.pending_training(awaiting["id"])[0]
        self.processes.complete_training(confirmation_training["id"])
        self.service.tick()
        confirmation_evaluation = self.pending_evaluations(awaiting["id"])[0]
        self.processes.complete_evaluation(
            confirmation_evaluation["id"], passed=False, tracking=0.4
        )
        before_record = self.service.get_campaign(awaiting["id"])
        before_runtime = self.service.store.get_runtime(awaiting["id"])
        with mock.patch.object(
            self.service,
            "_after_evaluation",
            side_effect=SimulatedCrash("after confirmation evaluation record"),
        ):
            with self.assertRaises(SimulatedCrash):
                self.service._poll_evaluation_state(before_record, before_runtime)
        recorded = self.service.get_campaign(awaiting["id"])
        self.assertEqual(recorded["state"], "candidate_evaluating")
        self.assertEqual(recorded["candidate_lineage"][-1]["kind"], "confirmation_control")
        self.assertEqual(recorded["candidate_lineage"][-1]["status"], "evaluated")

        self.service.close()
        self.service = AutopilotService(
            self.paths,
            self.history,  # type: ignore[arg-type]
            self.processes,  # type: ignore[arg-type]
            enabled=True,
            start_worker=False,
            identity_python=sys.executable,
        )
        self.service.tick()
        self.assertEqual(
            self.service.get_campaign(awaiting["id"])["state"], "confirming"
        )

    def test_restart_finishes_candidate_screen_outcome_after_runtime_commit(self) -> None:
        awaiting = self.reach_awaiting_advisor()
        self.propose_candidate(awaiting)
        self.service.tick()
        training = self.pending_training(awaiting["id"])[0]
        self.processes.complete_training(training["id"])
        self.service.tick()
        evaluation = self.pending_evaluations(awaiting["id"])[0]
        self.processes.complete_evaluation(
            evaluation["id"], passed=False, tracking=0.9
        )

        class SimulatedCrash(BaseException):
            pass

        original_transition = self.service.store.transition_campaign

        def crash_before_advisor(campaign_id, target, **kwargs):
            if target == "awaiting_advisor":
                raise SimulatedCrash("after candidate_screened")
            return original_transition(campaign_id, target, **kwargs)

        with mock.patch.object(
            self.service.store,
            "transition_campaign",
            side_effect=crash_before_advisor,
        ):
            with self.assertRaises(SimulatedCrash):
                self.service.tick()

        interrupted = self.service.get_campaign(awaiting["id"])
        self.assertEqual(interrupted["state"], "candidate_evaluating")
        runtime = self.service.store.get_runtime(awaiting["id"])
        screened_trial = next(
            trial
            for trial in interrupted["candidate_lineage"]
            if trial["kind"] == "candidate"
        )
        self.assertEqual(runtime["last_screened_trial_id"], screened_trial["id"])

        self.service.close()
        self.service = AutopilotService(
            self.paths,
            self.history,  # type: ignore[arg-type]
            self.processes,  # type: ignore[arg-type]
            enabled=True,
            start_worker=False,
            identity_python=sys.executable,
        )
        self.service.tick()
        recovered = self.service.get_campaign(awaiting["id"])
        self.assertEqual(recovered["state"], "awaiting_advisor")

    def test_restart_terminalizes_failed_candidate_evaluation_before_older_evidence(self) -> None:
        awaiting = self.reach_awaiting_advisor()
        self.propose_candidate(awaiting)
        first_screen = self.drive(
            awaiting["id"],
            {"awaiting_advisor"},
            lambda trial: (False, 0.9),
        )
        context = self.service.decision_context(awaiting["id"])
        move = next(
            item
            for item in context["remaining_allowable_moves"]
            if item["reward_key"] != "v2_reward_scales.forward_progress"
        )
        decision = AgentDecisionV1(
            campaign_id=first_screen["id"],
            campaign_revision=first_screen["revision"],
            evidence_ids=(first_screen["evaluations"][-1]["id"],),
            hypothesis="Test a second bounded shaping dimension.",
            action="propose_candidate",
            reward_key=move["reward_key"],
            proposed_value=move["proposed_value"],
            expected_metric_effect="Improve the remaining tracking gate.",
            rationale="Exercise failure recovery after earlier candidate evidence.",
        )
        self.service.submit_decision(
            first_screen["id"],
            decision.to_dict(),
            expected_revision=first_screen["revision"],
            idempotency_key=self.key("second-candidate"),
        )
        self.service.tick()
        training = self.pending_training(awaiting["id"])[0]
        self.processes.complete_training(training["id"])
        self.service.tick()
        evaluation = self.pending_evaluations(awaiting["id"])[0]
        evaluation.update(
            status="failed",
            failure_class="configuration",
            failure_reason="evaluation profile mismatch",
            completed_at=_now(),
        )

        class SimulatedCrash(BaseException):
            pass

        with mock.patch.object(
            self.service,
            "_finish_failed_evaluation",
            side_effect=SimulatedCrash("after failed trial commit"),
        ):
            with self.assertRaises(SimulatedCrash):
                self.service.tick()

        interrupted = self.service.get_campaign(awaiting["id"])
        failed_trial = interrupted["candidate_lineage"][-1]
        self.assertEqual(interrupted["state"], "candidate_evaluating")
        self.assertEqual(failed_trial["status"], "failed")
        trial_count = len(interrupted["candidate_lineage"])
        evaluation_attempts = len(self.processes.evaluation_params)

        self.service.close()
        self.service = AutopilotService(
            self.paths,
            self.history,  # type: ignore[arg-type]
            self.processes,  # type: ignore[arg-type]
            enabled=True,
            start_worker=False,
            identity_python=sys.executable,
        )
        self.service.tick()

        recovered = self.service.get_campaign(awaiting["id"])
        self.assertEqual(recovered["state"], "blocked_safety")
        self.assertEqual(len(recovered["candidate_lineage"]), trial_count)
        self.assertEqual(len(self.processes.evaluation_params), evaluation_attempts)
        self.assertIn("evaluator failed", recovered["terminal_reason"])

    def test_restart_terminalizes_invalid_evidence_after_failed_trial_commit(self) -> None:
        armed = self.create_and_arm()
        self.service.tick()
        training = self.pending_training(armed["id"])[0]
        self.processes.complete_training(training["id"])
        self.service.tick()
        evaluation = self.pending_evaluations(armed["id"])[0]
        self.processes.complete_evaluation(
            evaluation["id"],
            passed=True,
            tracking=0.9,
            malformed="nonfinite",
        )

        class SimulatedCrash(BaseException):
            pass

        with mock.patch.object(
            self.service,
            "_finish_failed_evaluation",
            side_effect=SimulatedCrash("after invalid-evidence trial commit"),
        ):
            with self.assertRaises(SimulatedCrash):
                self.service.tick()

        interrupted = self.service.get_campaign(armed["id"])
        self.assertEqual(interrupted["state"], "control_evaluating")
        self.assertEqual(interrupted["candidate_lineage"][0]["status"], "failed")
        evaluation_attempts = len(self.processes.evaluation_params)

        self.service.close()
        self.service = AutopilotService(
            self.paths,
            self.history,  # type: ignore[arg-type]
            self.processes,  # type: ignore[arg-type]
            enabled=True,
            start_worker=False,
            identity_python=sys.executable,
        )
        self.service.tick()

        recovered = self.service.get_campaign(armed["id"])
        self.assertEqual(recovered["state"], "blocked_safety")
        self.assertEqual(len(self.processes.evaluation_params), evaluation_attempts)
        self.assertIn("invalid evaluation evidence", recovered["terminal_reason"])

    def test_paused_evaluation_completion_and_failure_resume_deterministically(self) -> None:
        armed = self.create_and_arm()
        self.service.tick()
        training = self.pending_training(armed["id"])[0]
        self.processes.complete_training(training["id"])
        self.service.tick()
        evaluation = self.pending_evaluations(armed["id"])[0]
        current = self.service.get_campaign(armed["id"])
        self.service.pause_campaign(
            armed["id"],
            expected_revision=current["revision"],
            idempotency_key=self.key("pause-evaluation"),
        )
        self.processes.complete_evaluation(
            evaluation["id"], passed=False, tracking=0.4
        )
        self.service.tick()

        paused = self.service.get_campaign(armed["id"])
        self.assertEqual(paused["state"], "paused")
        self.assertEqual(paused["candidate_lineage"][0]["status"], "evaluated")
        self.assertIsNone(paused["active_process"])
        self.service.tick()
        self.assertEqual(self.service.get_campaign(armed["id"])["state"], "paused")

        resumed = self.service.resume_campaign(
            armed["id"],
            expected_revision=paused["revision"],
            idempotency_key=self.key("resume-evaluation"),
        )
        self.service.tick()
        completed = self.service.get_campaign(armed["id"])
        self.assertEqual(resumed["state"], "control_evaluating")
        self.assertEqual(completed["state"], "awaiting_advisor")
        self.service.stop_campaign(
            armed["id"],
            expected_revision=completed["revision"],
            idempotency_key=self.key("stop-completed-evaluation"),
            after_current=False,
        )

        second = self.create_and_arm()
        self.service.tick()
        training = self.pending_training(second["id"])[0]
        self.processes.complete_training(training["id"])
        self.service.tick()
        evaluation = self.pending_evaluations(second["id"])[0]
        current = self.service.get_campaign(second["id"])
        self.service.pause_campaign(
            second["id"],
            expected_revision=current["revision"],
            idempotency_key=self.key("pause-failed-evaluation"),
        )
        evaluation.update(
            status="failed",
            failure_class="configuration",
            failure_reason="command profile mismatch",
            completed_at=_now(),
        )
        self.service.tick()

        paused_failure = self.service.get_campaign(second["id"])
        self.assertEqual(paused_failure["state"], "paused")
        self.assertEqual(
            paused_failure["candidate_lineage"][0]["status"], "failed"
        )
        self.assertIsNone(paused_failure["active_process"])
        self.service.tick()
        self.assertEqual(self.service.get_campaign(second["id"])["state"], "paused")

        self.service.resume_campaign(
            second["id"],
            expected_revision=paused_failure["revision"],
            idempotency_key=self.key("resume-failed-evaluation"),
        )
        self.service.tick()
        blocked = self.service.get_campaign(second["id"])
        self.assertEqual(blocked["state"], "blocked_safety")
        self.assertIn("evaluator failed", blocked["terminal_reason"])

    def test_evaluation_gpu_contention_waits_without_failing_campaign(self) -> None:
        armed = self.create_and_arm()
        self.service.tick()
        training = self.pending_training(armed["id"])[0]
        self.processes.complete_training(training["id"])

        busy = GpuHostLeaseBusy(
            "Another Isaac/GPU process owns the host lease.",
            {
                "code": "gpu_host_lease_busy",
                "lease": {"process_id": "unrelated-play", "kind": "play"},
            },
        )
        with mock.patch.object(self.processes, "start_evaluation", side_effect=busy):
            self.service.tick()

        waiting = self.service.get_campaign(armed["id"])
        trial = waiting["candidate_lineage"][0]
        self.assertEqual(waiting["state"], "control_evaluating")
        self.assertEqual(trial["status"], "trained")
        self.assertEqual(trial["metadata"]["evaluation_wait_reason"], "gpu_host_contention")
        self.assertIsNone(waiting["active_process"])

        self.service.tick()
        recovered = self.service.get_campaign(armed["id"])
        self.assertEqual(recovered["candidate_lineage"][0]["status"], "evaluating")
        self.assertEqual(len(self.processes.evaluation_params), 1)

    def test_evaluation_infrastructure_failure_retries_once_with_identical_inputs(self) -> None:
        armed = self.create_and_arm()
        self.service.tick()
        training = self.pending_training(armed["id"])[0]
        self.processes.complete_training(training["id"])
        self.service.tick()
        first = self.pending_evaluations(armed["id"])[0]
        first.update(
            status="failed",
            failure_reason="failed to create CUDA context",
            completed_at=_now(),
        )

        self.service.tick()
        reserved = self.service.get_campaign(armed["id"])
        trial = reserved["candidate_lineage"][0]
        self.assertEqual(trial["status"], "trained")
        self.assertEqual(trial["evaluation_retry_count"], 1)
        self.assertEqual(
            trial["evaluation_retry_reason"],
            "failed to create CUDA context",
        )
        self.service.tick()
        attempts = [
            run for run in self.history.list_runs()
            if run.get("source") == "autopilot_evaluation"
            and run.get("campaign_id") == armed["id"]
        ]
        self.assertEqual(len(attempts), 2)
        self.assertEqual(
            asdict(self.processes.evaluation_params[0]),
            asdict(self.processes.evaluation_params[1]),
        )

        retry = self.pending_evaluations(armed["id"])[0]
        retry.update(
            status="failed",
            failure_reason="temporarily unavailable",
            completed_at=_now(),
        )
        self.service.tick()
        final = self.service.get_campaign(armed["id"])
        self.assertEqual(final["state"], "blocked_safety")
        self.assertEqual(len(self.processes.evaluation_params), 2)

    def test_evaluation_configuration_failure_never_retries(self) -> None:
        armed = self.create_and_arm()
        self.service.tick()
        training = self.pending_training(armed["id"])[0]
        self.processes.complete_training(training["id"])
        self.service.tick()
        evaluation = self.pending_evaluations(armed["id"])[0]
        evaluation.update(
            status="failed",
            failure_class="configuration",
            failure_reason="command profile mismatch; temporarily unavailable",
            completed_at=_now(),
        )

        self.service.tick()
        final = self.service.get_campaign(armed["id"])
        self.assertEqual(final["state"], "blocked_safety")
        self.assertEqual(final["candidate_lineage"][0]["evaluation_retry_count"], 0)
        self.assertEqual(len(self.processes.evaluation_params), 1)

    def test_training_infrastructure_retry_does_not_rebind_failed_attempt(self) -> None:
        armed = self.create_and_arm()
        self.service.tick()
        first = self.pending_training(armed["id"])[0]
        first.update(
            status="failed",
            failure_reason="failed to create CUDA context",
            completed_at=_now(),
        )

        self.service.tick()
        attempts = [
            run for run in self.history.list_runs()
            if run.get("source") == "training_panel"
            and run.get("campaign_id") == armed["id"]
        ]
        snapshot = self.service.get_campaign(armed["id"])
        trial = snapshot["candidate_lineage"][0]

        self.assertEqual(len(attempts), 2)
        self.assertNotEqual(attempts[0]["id"], attempts[1]["id"])
        self.assertEqual(trial["retry_count"], 1)
        self.assertEqual(trial["metadata"]["retry_source_run_id"], first["id"])
        self.assertNotEqual(trial["run_id"], first["id"])

    def test_failed_training_retries_charge_every_attempt_once(self) -> None:
        armed = self.create_and_arm()
        self.service.tick()
        first = self.pending_training(armed["id"])[0]
        first.update(
            status="failed",
            started_at=(datetime.now(timezone.utc) - timedelta(seconds=20)).isoformat(),
            completed_at=_now(),
            failure_reason="failed to create CUDA context",
        )

        self.service.tick()
        attempts = [
            run for run in self.history.list_runs()
            if run.get("source") == "training_panel"
            and run.get("campaign_id") == armed["id"]
        ]
        self.assertEqual(len(attempts), 2)
        retry = next(run for run in attempts if run["id"] != first["id"])
        retry.update(
            status="failed",
            started_at=(datetime.now(timezone.utc) - timedelta(seconds=30)).isoformat(),
            completed_at=_now(),
            failure_reason="configuration mismatch",
            failure_class="configuration",
        )

        self.service.tick()
        final = self.service.get_campaign(armed["id"])
        self.assertEqual(final["state"], "failed")
        self.assertGreater(final["budget"]["used_gpu_hours"], 49.0 / 3600.0)
        accounting = final["candidate_lineage"][0]["metadata"]["gpu_process_accounting"]
        self.assertEqual(set(accounting), {first["id"], retry["id"]})
        used_before = final["budget"]["used_gpu_hours"]
        self.service.tick()
        self.assertEqual(
            self.service.get_campaign(armed["id"])["budget"]["used_gpu_hours"],
            used_before,
        )

    def test_gpu_budget_stop_is_durable_and_retried_after_restart(self) -> None:
        payload = self.payload()
        payload["budget"] = {"max_training_trials": 24, "max_gpu_hours": 0.001}
        armed = self.create_and_arm(payload)
        self.service.tick()
        running = self.pending_training(armed["id"])[0]
        running.update(
            status="running",
            started_at=(datetime.now(timezone.utc) - timedelta(seconds=10)).isoformat(),
        )

        with mock.patch.object(self.processes, "stop", side_effect=RuntimeError("busy")):
            self.service.tick()
        exhausted = self.service.get_campaign(armed["id"])
        self.assertEqual(exhausted["state"], "budget_exhausted")
        self.assertEqual(exhausted["active_process"]["process_id"], running["id"])
        self.assertGreaterEqual(exhausted["budget"]["used_gpu_hours"], 0.001)

        self.service.close()
        self.service = AutopilotService(
            self.paths,
            self.history,  # type: ignore[arg-type]
            self.processes,  # type: ignore[arg-type]
            enabled=True,
            start_worker=False,
            identity_python=sys.executable,
        )
        self.service.tick()
        recovered = self.service.get_campaign(armed["id"])
        self.assertEqual(recovered["state"], "budget_exhausted")
        self.assertIsNone(recovered["active_process"])
        self.assertEqual(self.processes.stopped, [running["id"]])

    def test_orphaned_bound_process_is_rebound_before_budget_stop(self) -> None:
        payload = self.payload()
        payload["budget"] = {"max_training_trials": 24, "max_gpu_hours": 0.001}
        armed = self.create_and_arm(payload)
        self.service.tick()
        running = self.pending_training(armed["id"])[0]
        running.update(
            status="running",
            started_at=(datetime.now(timezone.utc) - timedelta(seconds=10)).isoformat(),
        )
        connection = sqlite3.connect(self.paths.autopilot_db_file)
        try:
            connection.execute(
                "UPDATE campaigns SET active_process_json=NULL WHERE id=?",
                (armed["id"],),
            )
            connection.commit()
        finally:
            connection.close()

        self.service.tick()
        exhausted = self.service.get_campaign(armed["id"])

        self.assertEqual(exhausted["state"], "budget_exhausted")
        self.assertIsNone(exhausted["active_process"])
        self.assertEqual(self.processes.stopped, [running["id"]])
        event_types = [
            event["type"] for event in self.service.list_events(armed["id"])
        ]
        self.assertIn("training_process_rebound", event_types)
        self.assertIn("gpu_budget_process_stopped", event_types)

    def test_running_gpu_accounting_is_coarsened_but_cap_checks_remain_live(self) -> None:
        armed = self.create_and_arm()
        self.service.tick()
        running = self.pending_training(armed["id"])[0]
        running.update(
            status="running",
            started_at=(datetime.now(timezone.utc) - timedelta(seconds=10)).isoformat(),
        )
        before = self.service.get_campaign(armed["id"])

        self.service.tick()
        coarsened = self.service.get_campaign(armed["id"])
        self.assertEqual(coarsened["revision"], before["revision"])
        self.assertEqual(coarsened["budget"]["used_gpu_hours"], 0.0)

        running["started_at"] = (
            datetime.now(timezone.utc) - timedelta(seconds=70)
        ).isoformat()
        self.service.tick()
        charged = self.service.get_campaign(armed["id"])
        self.assertGreater(charged["revision"], coarsened["revision"])
        self.assertGreater(charged["budget"]["used_gpu_hours"], 69.0 / 3600.0)

    def test_emergency_stop_charges_the_stopped_attempt_before_terminal_state(self) -> None:
        armed = self.create_and_arm()
        self.service.tick()
        running = self.pending_training(armed["id"])[0]
        running.update(
            status="running",
            started_at=(datetime.now(timezone.utc) - timedelta(seconds=30)).isoformat(),
        )
        current = self.service.get_campaign(armed["id"])

        request_key = self.key("emergency-stop")
        stopped = self.service.stop_campaign(
            armed["id"],
            expected_revision=current["revision"],
            idempotency_key=request_key,
            after_current=False,
            reason="test emergency stop",
        )

        self.assertEqual(stopped["state"], "stopped")
        self.assertIsNone(stopped["active_process"])
        self.assertGreater(stopped["budget"]["used_gpu_hours"], 29.0 / 3600.0)
        accounting = stopped["candidate_lineage"][0]["metadata"]["gpu_process_accounting"]
        self.assertIn(running["id"], accounting)
        self.assertEqual(self.processes.stopped, [running["id"]])
        duplicate = self.service.stop_campaign(
            armed["id"],
            expected_revision=current["revision"],
            idempotency_key=request_key,
            after_current=False,
            reason="test emergency stop",
        )
        self.assertEqual(duplicate, stopped)

    def test_restart_finalizes_durable_emergency_intent_after_signal_crash(self) -> None:
        armed = self.create_and_arm()
        self.service.tick()
        running = self.pending_training(armed["id"])[0]
        running.update(
            status="running",
            started_at=(datetime.now(timezone.utc) - timedelta(seconds=30)).isoformat(),
        )
        current = self.service.get_campaign(armed["id"])

        class SimulatedCrash(BaseException):
            pass

        with mock.patch.object(
            self.service.store,
            "stop_campaign",
            side_effect=SimulatedCrash("after process signal"),
        ):
            with self.assertRaises(SimulatedCrash):
                self.service.stop_campaign(
                    armed["id"],
                    expected_revision=current["revision"],
                    idempotency_key=self.key("crash-stop"),
                    after_current=False,
                    reason="crash boundary test",
                )

        interrupted = self.service.get_campaign(armed["id"])
        self.assertEqual(interrupted["state"], "paused")
        self.assertIsNotNone(interrupted["active_process"])
        self.assertTrue(
            self.service.store.get_runtime(armed["id"])["emergency_stop_intent"]
        )
        self.assertEqual(self.processes.stopped, [running["id"]])

        self.service.close()
        self.service = AutopilotService(
            self.paths,
            self.history,  # type: ignore[arg-type]
            self.processes,  # type: ignore[arg-type]
            enabled=True,
            start_worker=False,
            identity_python=sys.executable,
        )
        self.service.tick()

        recovered = self.service.get_campaign(armed["id"])
        self.assertEqual(recovered["state"], "stopped")
        self.assertIsNone(recovered["active_process"])
        self.assertGreater(recovered["budget"]["used_gpu_hours"], 29.0 / 3600.0)
        self.assertNotIn(
            "emergency_stop_intent", self.service.store.get_runtime(armed["id"])
        )
        self.assertEqual(self.processes.stopped, [running["id"]])

    def test_connector_poll_cap_terminally_stops_only_the_bound_campaign_job(self) -> None:
        armed = self.create_and_arm()
        self.service.tick()
        active = self.service.get_campaign(armed["id"])
        running = self.pending_training(armed["id"])[0]
        running["status"] = "running"
        budget = dict(active["budget"])
        budget["connector_polls"] = budget["max_connector_polls"] - 1
        connection = sqlite3.connect(self.paths.autopilot_db_file)
        try:
            connection.execute(
                "UPDATE campaigns SET budget_json=? WHERE id=?",
                (json.dumps(budget, sort_keys=True), active["id"]),
            )
            connection.commit()
        finally:
            connection.close()

        exhausted = self.service.advisor_heartbeat(
            active["id"],
            advisor_metadata={
                "schema_version": "redrhex.autopilot.advisor-metadata.v1",
                "skill_version": "skill-v1",
                "prompt_version": "prompt-v1",
                "declared_model": "test-model",
                "reasoning_effort": "medium",
            },
            expected_revision=active["revision"],
            idempotency_key=self.key("heartbeat"),
        )
        self.assertEqual(exhausted["state"], "budget_exhausted")
        self.assertEqual(exhausted["active_process"]["process_id"], running["id"])

        self.service.tick()
        stopped = self.service.get_campaign(active["id"])
        self.assertIsNone(stopped["active_process"])
        self.assertEqual(self.processes.stopped, [running["id"]])
        with self.assertRaisesRegex(AutopilotConflictError, "not resumable"):
            self.service.resume_campaign(
                active["id"],
                expected_revision=stopped["revision"],
                idempotency_key=self.key("resume"),
            )

    def test_last_nonpassing_screen_enters_patch_handoff_when_only_confirmation_budget_remains(self) -> None:
        payload = self.payload()
        payload["budget"] = {"max_training_trials": 6, "max_gpu_hours": 72.0}
        armed = self.create_and_arm(payload)
        awaiting = self.drive(
            armed["id"],
            {"awaiting_advisor"},
            lambda trial: (False, 0.4),
        )
        self.propose_candidate(awaiting)

        final = self.drive(
            awaiting["id"],
            {"patch_handoff", "failed", "blocked_safety"},
            lambda trial: (False, 0.9),
        )

        self.assertEqual(final["state"], "patch_handoff")
        self.assertIn("reserved confirmation", final["terminal_reason"])

    def test_restart_recovers_decision_committed_before_candidate_runtime(self) -> None:
        awaiting = self.reach_awaiting_advisor()
        decision = AgentDecisionV1(
            campaign_id=awaiting["id"],
            campaign_revision=awaiting["revision"],
            evidence_ids=(awaiting["evaluations"][-1]["id"],),
            hypothesis="The control undershoots forward commands.",
            action="propose_candidate",
            reward_key="v2_reward_scales.forward_progress",
            proposed_value=3.3,
            expected_metric_effect="Improve forward tracking.",
            rationale="Change exactly one bounded shaping term.",
        )
        self.service.store.record_decision(
            awaiting["id"],
            decision,
            expected_revision=awaiting["revision"],
            idempotency_key=self.key("interrupted-decision"),
        )
        self.service.close()
        self.service = AutopilotService(
            self.paths,
            self.history,  # type: ignore[arg-type]
            self.processes,  # type: ignore[arg-type]
            enabled=True,
            start_worker=False,
            identity_python=sys.executable,
        )

        self.service.tick()
        self.service.tick()
        recovered = self.service.get_campaign(awaiting["id"])
        candidates = [
            trial for trial in recovered["candidate_lineage"]
            if trial["kind"] == "candidate"
        ]
        self.assertEqual(len(candidates), 1)
        self.assertEqual(recovered["state"], "candidate_training")
        self.assertEqual(len(self.pending_training(awaiting["id"])), 1)

    def test_pause_preserves_completed_work_and_stop_after_current_is_terminal(self) -> None:
        armed = self.create_and_arm()
        self.service.tick()
        running = self.pending_training(armed["id"])[0]
        snapshot = self.service.get_campaign(armed["id"])
        paused = self.service.pause_campaign(
            armed["id"],
            expected_revision=snapshot["revision"],
            idempotency_key=self.key("pause"),
        )
        self.processes.complete_training(running["id"])
        self.service.tick()
        paused = self.service.get_campaign(armed["id"])
        self.assertEqual(paused["state"], "paused")
        self.assertEqual(paused["candidate_lineage"][0]["status"], "trained")
        self.assertFalse(self.pending_evaluations(armed["id"]))

        resumed = self.service.resume_campaign(
            armed["id"],
            expected_revision=paused["revision"],
            idempotency_key=self.key("resume"),
        )
        self.service.tick()
        self.assertEqual(resumed["state"], "control_training")
        self.assertEqual(len(self.pending_evaluations(armed["id"])), 1)
        current = self.service.get_campaign(armed["id"])
        self.service.stop_campaign(
            armed["id"],
            expected_revision=current["revision"],
            idempotency_key=self.key("emergency-stop"),
            after_current=False,
        )

        second = self.create_and_arm()
        self.service.tick()
        run = self.pending_training(second["id"])[0]
        current = self.service.get_campaign(second["id"])
        self.service.stop_campaign(
            second["id"],
            expected_revision=current["revision"],
            idempotency_key=self.key("stop-after"),
            after_current=True,
        )
        self.processes.complete_training(run["id"])
        self.service.tick()
        stopped = self.service.get_campaign(second["id"])
        self.assertEqual(stopped["state"], "stopped")
        self.assertFalse(self.pending_evaluations(second["id"]))

    def test_advisor_pause_and_boundary_stop_commit_heartbeat_atomically(self) -> None:
        metadata = {
            "schema_version": "redrhex.autopilot.advisor-metadata.v1",
            "skill_version": "skill-v1",
            "prompt_version": "prompt-v1",
            "declared_model": "test-model",
            "reasoning_effort": "medium",
        }
        armed = self.create_and_arm()
        self.service.tick()
        before_pause = self.service.get_campaign(armed["id"])
        paused = self.service.pause_campaign(
            armed["id"],
            expected_revision=before_pause["revision"],
            idempotency_key=self.key("advisor-pause"),
            advisor_metadata=metadata,
        )
        self.assertEqual(paused["revision"], before_pause["revision"] + 1)
        self.assertEqual(paused["state"], "paused")
        self.assertEqual(paused["budget"]["connector_polls"], 1)
        self.assertEqual(paused["connector"]["declared_model"], "test-model")

        resumed = self.service.resume_campaign(
            armed["id"],
            expected_revision=paused["revision"],
            idempotency_key=self.key("resume-after-advisor-pause"),
        )
        stopped_later = self.service.stop_campaign(
            armed["id"],
            expected_revision=resumed["revision"],
            idempotency_key=self.key("advisor-boundary-stop"),
            after_current=True,
            advisor_metadata=metadata,
        )
        self.assertEqual(stopped_later["revision"], resumed["revision"] + 1)
        self.assertEqual(stopped_later["budget"]["connector_polls"], 2)
        self.assertTrue(
            self.service.store.get_runtime(armed["id"])["stop_after_current"]
        )

    def test_restart_honors_stop_after_training_commit_before_evaluation_launch(self) -> None:
        armed = self.create_and_arm()
        self.service.tick()
        training = self.pending_training(armed["id"])[0]
        current = self.service.get_campaign(armed["id"])
        self.service.stop_campaign(
            armed["id"],
            expected_revision=current["revision"],
            idempotency_key=self.key("stop-after-training-crash"),
            after_current=True,
        )
        self.processes.complete_training(training["id"])

        class SimulatedCrash(BaseException):
            pass

        original_complete = self.service.store.complete_training

        def commit_then_crash(*args, **kwargs):
            original_complete(*args, **kwargs)
            raise SimulatedCrash("after training completion commit")

        with mock.patch.object(
            self.service.store, "complete_training", side_effect=commit_then_crash
        ):
            with self.assertRaises(SimulatedCrash):
                self.service.tick()

        self.service.close()
        self.service = AutopilotService(
            self.paths,
            self.history,  # type: ignore[arg-type]
            self.processes,  # type: ignore[arg-type]
            enabled=True,
            start_worker=False,
            identity_python=sys.executable,
        )
        self.service.tick()

        recovered = self.service.get_campaign(armed["id"])
        self.assertEqual(recovered["state"], "stopped")
        self.assertFalse(self.pending_evaluations(armed["id"]))

    def test_restart_honors_stop_after_evaluation_commit_before_funnel_advance(self) -> None:
        armed = self.create_and_arm()
        self.service.tick()
        training = self.pending_training(armed["id"])[0]
        self.processes.complete_training(training["id"])
        self.service.tick()
        evaluation = self.pending_evaluations(armed["id"])[0]
        current = self.service.get_campaign(armed["id"])
        self.service.stop_campaign(
            armed["id"],
            expected_revision=current["revision"],
            idempotency_key=self.key("stop-after-evaluation-crash"),
            after_current=True,
        )
        self.processes.complete_evaluation(
            evaluation["id"], passed=False, tracking=0.4
        )

        class SimulatedCrash(BaseException):
            pass

        original_record = self.service.store.record_evaluation

        def commit_then_crash(*args, **kwargs):
            original_record(*args, **kwargs)
            raise SimulatedCrash("after evaluation completion commit")

        with mock.patch.object(
            self.service.store, "record_evaluation", side_effect=commit_then_crash
        ):
            with self.assertRaises(SimulatedCrash):
                self.service.tick()

        self.service.close()
        self.service = AutopilotService(
            self.paths,
            self.history,  # type: ignore[arg-type]
            self.processes,  # type: ignore[arg-type]
            enabled=True,
            start_worker=False,
            identity_python=sys.executable,
        )
        self.service.tick()

        recovered = self.service.get_campaign(armed["id"])
        self.assertEqual(recovered["state"], "stopped")
        self.assertEqual(len(recovered["candidate_lineage"]), 1)

    def test_stale_emergency_stop_never_signals_campaign_process(self) -> None:
        armed = self.create_and_arm()
        self.service.tick()
        current = self.service.get_campaign(armed["id"])

        with self.assertRaises(AutopilotConflictError):
            self.service.stop_campaign(
                armed["id"],
                expected_revision=current["revision"] - 1,
                idempotency_key=self.key("stale-emergency"),
                after_current=False,
            )

        self.assertEqual(self.processes.stopped, [])
        self.assertEqual(self.service.get_campaign(armed["id"])["state"], "control_training")

    def test_confirmation_success_uses_three_paired_valid_replicas(self) -> None:
        awaiting = self.reach_awaiting_advisor()
        self.propose_candidate(awaiting)

        final = self.drive(
            awaiting["id"],
            {"simulation_goal_met", "patch_handoff", "failed", "blocked_safety"},
            lambda trial: (
                (trial["kind"] in {"candidate", "confirmation_candidate"}),
                0.9 if trial["kind"] in {"candidate", "confirmation_candidate"} else 0.4,
            ),
        )

        self.assertEqual(final["state"], "simulation_goal_met")
        self.assertEqual(len(final["candidate_lineage"]), 6)
        self.assertEqual(len(final["evaluations"]), 6)
        self.assertEqual(
            [(item["kind"], item["seed"]) for item in final["candidate_lineage"]],
            [
                ("control", 42),
                ("candidate", 42),
                ("confirmation_control", 43),
                ("confirmation_candidate", 43),
                ("confirmation_control", 44),
                ("confirmation_candidate", 44),
            ],
        )

    def test_confirmation_failure_creates_patch_handoff(self) -> None:
        awaiting = self.reach_awaiting_advisor()
        self.propose_candidate(awaiting)

        final = self.drive(
            awaiting["id"],
            {"simulation_goal_met", "patch_handoff", "failed", "blocked_safety"},
            lambda trial: (
                trial["kind"] == "candidate",
                0.9 if trial["kind"] == "candidate" else 0.4,
            ),
        )

        self.assertEqual(final["state"], "patch_handoff")
        self.assertIn("confirmation failed closed", final["terminal_reason"])
        self.assertIn(
            "patch_context",
            {item["kind"] for item in self.service.list_artifacts(final["id"])},
        )

    def test_budget_impossible_after_control_enters_patch_handoff(self) -> None:
        payload = self.payload()
        payload["budget"] = {"max_training_trials": 5, "max_gpu_hours": 72.0}
        armed = self.create_and_arm(payload)

        final = self.drive(
            armed["id"],
            {"patch_handoff", "failed", "blocked_safety"},
            lambda trial: (False, 0.4),
        )

        self.assertEqual(final["state"], "patch_handoff")
        self.assertIn("insufficient training budget", final["terminal_reason"])
        self.assertEqual(len(final["candidate_lineage"]), 1)

    def test_patch_proposal_is_idempotent_and_never_applied(self) -> None:
        awaiting = self.reach_awaiting_advisor()
        handoff_decision = AgentDecisionV1(
            campaign_id=awaiting["id"],
            campaign_revision=awaiting["revision"],
            evidence_ids=(awaiting["evaluations"][-1]["id"],),
            hypothesis="Bounded reward shaping is insufficient.",
            action="request_patch_handoff",
            rationale="Prepare an allowlisted source proposal for human review only.",
        )
        handoff = self.service.submit_decision(
            awaiting["id"],
            handoff_decision.to_dict(),
            expected_revision=awaiting["revision"],
            idempotency_key=self.key("patch-handoff"),
        )
        context = self.service.decision_context(awaiting["id"])["patch_handoff"]
        source = "source/RedRhex/RedRhex/tasks/direct/redrhex/redrhex_env_cfg.py"
        snippet = next(
            item for item in context["source_snippets"] if item["source"] == source
        )
        original = (self.root / source).read_bytes()
        proposal = {
            "schema_version": "redrhex.autopilot.patch-proposal.v1",
            "target_symbols": ["RedrhexEnvCfg.v2_reward_scales"],
            "base_blob_hashes": {source: context["allowed_source_blobs"][source]},
            "unified_diff": (
                f"--- a/{source}\n+++ b/{source}\n"
                f"@@ -{snippet['line_start']} +{snippet['line_start']} @@\n"
                f"-fixture:{source}\n+fixture:{source}  # proposed only\n"
            ),
            "rationale": "Expose one source-level shaping alternative.",
            "test_plan": "Run contract tests and a new linked shadow campaign.",
            "rollback_notes": "Discard this unapplied artifact.",
        }
        patch_decision = AgentDecisionV1(
            campaign_id=handoff["id"],
            campaign_revision=handoff["revision"],
            evidence_ids=(awaiting["evaluations"][-1]["id"],),
            hypothesis="A source-level reward expression may be required.",
            action="request_patch_handoff",
            rationale="Submit a review artifact without mutating the repository.",
        )
        key = self.key("patch-proposal")
        first = self.service.submit_patch_proposal(
            handoff["id"],
            patch_decision,
            proposal,
            advisor_metadata=None,
            expected_revision=handoff["revision"],
            idempotency_key=key,
        )
        (self.root / source).write_bytes(original + b"# later external edit\n")
        frozen_again = self.service.decision_context(handoff["id"])["patch_handoff"]
        self.assertEqual(
            frozen_again["allowed_source_blobs"][source],
            context["allowed_source_blobs"][source],
        )
        repeated = self.service.submit_patch_proposal(
            handoff["id"],
            patch_decision,
            proposal,
            advisor_metadata=None,
            expected_revision=handoff["revision"],
            idempotency_key=key,
        )

        self.assertEqual(repeated, first)
        self.assertEqual((self.root / source).read_bytes(), original + b"# later external edit\n")
        artifact, content = self.service.patch_export(handoff["id"])
        self.assertEqual(artifact["kind"], "patch_proposal")
        self.assertIn(b'"unified_diff"', content)

    def test_patch_proposal_scope_is_bound_to_frozen_symbols_hashes_and_snippets(self) -> None:
        awaiting = self.reach_awaiting_advisor()
        decision = AgentDecisionV1(
            campaign_id=awaiting["id"],
            campaign_revision=awaiting["revision"],
            evidence_ids=(awaiting["evaluations"][-1]["id"],),
            hypothesis="Bounded reward shaping is insufficient.",
            action="request_patch_handoff",
            rationale="Prepare a source proposal for human review.",
        )
        handoff = self.service.submit_decision(
            awaiting["id"],
            decision.to_dict(),
            expected_revision=awaiting["revision"],
            idempotency_key=self.key("patch-scope-handoff"),
        )
        context = self.service.decision_context(handoff["id"])["patch_handoff"]
        source = "source/RedRhex/RedRhex/tasks/direct/redrhex/redrhex_env_cfg.py"
        snippet = next(
            item for item in context["source_snippets"] if item["source"] == source
        )
        proposal = {
            "schema_version": "redrhex.autopilot.patch-proposal.v1",
            "target_symbols": ["RedrhexEnvCfg.v2_reward_scales"],
            "base_blob_hashes": {source: context["allowed_source_blobs"][source]},
            "unified_diff": (
                f"--- a/{source}\n+++ b/{source}\n"
                f"@@ -{snippet['line_start']} +{snippet['line_start']} @@\n"
                "-fixture\n+fixture # proposal\n"
            ),
            "rationale": "Try a source-level shaping change.",
            "test_plan": "Run contracts and a linked campaign.",
            "rollback_notes": "Discard this unapplied artifact.",
        }

        invalid_symbol = dict(proposal, target_symbols=["os.system"])
        with self.assertRaisesRegex(AutopilotValidationError, "frozen handoff context"):
            self.service._validate_patch_proposal(
                invalid_symbol, patch_context=context
            )

        extra_hash = dict(proposal)
        extra_hash["base_blob_hashes"] = dict(context["allowed_source_blobs"])
        with self.assertRaisesRegex(AutopilotValidationError, "exactly one frozen base hash"):
            self.service._validate_patch_proposal(extra_hash, patch_context=context)

        outside_hunk = dict(
            proposal,
            unified_diff=(
                f"--- a/{source}\n+++ b/{source}\n@@ -999 +999 @@\n-old\n+new\n"
            ),
        )
        with self.assertRaisesRegex(AutopilotValidationError, "outside the frozen"):
            self.service._validate_patch_proposal(
                outside_hunk, patch_context=context
            )

    def test_episode_numeric_domains_fail_closed(self) -> None:
        invalid_values = (
            ("mae_vx", -0.01, "mae_vx must be non-negative"),
            ("mae_vy", -0.01, "mae_vy must be non-negative"),
            ("mae_wz", -0.01, "mae_wz must be non-negative"),
            (
                "energy_mech_power_total_mean",
                -0.01,
                "energy_mech_power_total_mean must be non-negative",
            ),
            ("energy_effort_mean", -0.01, "energy_effort_mean must be non-negative"),
            ("success_ratio", 1.01, "success_ratio must be between 0 and 1"),
            ("fall_count", 2, "fall_count must be 0 or 1"),
        )
        for field, value, expected_reason in invalid_values:
            with self.subTest(field=field):
                final = self.complete_with_evidence_mutation(
                    episode_mutator=lambda rows, field=field, value=value: rows[0].__setitem__(
                        field, value
                    )
                )

                self.assertEqual(final["state"], "blocked_safety")
                self.assertIn(expected_reason, final["terminal_reason"])

    def test_episode_identity_and_topology_fail_closed(self) -> None:
        def duplicate(rows: list[dict[str, Any]]) -> None:
            rows.append(dict(rows[0]))

        def missing_environment(rows: list[dict[str, Any]]) -> None:
            command = rows[0]["command"]
            rows[:] = [
                row
                for row in rows
                if not (row["command"] == command and row["environment_index"] == 1)
            ]

        def index_gap(rows: list[dict[str, Any]]) -> None:
            rows[0]["episode_index"] = 1

        def nonfinal_incomplete(rows: list[dict[str, Any]]) -> None:
            rows[0]["complete"] = False
            following = dict(rows[0])
            following["episode_index"] = 1
            following["complete"] = True
            rows.append(following)

        def incomplete_fall(rows: list[dict[str, Any]]) -> None:
            rows[0]["complete"] = False
            rows[0]["fall_count"] = 1

        invalid_topologies = (
            (duplicate, "duplicate command/environment/episode"),
            (missing_environment, "does not cover every frozen environment"),
            (index_gap, "indices are not contiguous"),
            (nonfinal_incomplete, "only the final episode may be incomplete"),
            (incomplete_fall, "cannot record a fall on an incomplete episode"),
        )
        for mutator, expected_reason in invalid_topologies:
            with self.subTest(expected_reason=expected_reason):
                final = self.complete_with_evidence_mutation(episode_mutator=mutator)

                self.assertEqual(final["state"], "blocked_safety")
                self.assertIn(expected_reason, final["terminal_reason"])

    def test_episode_aggregates_must_match_command_csv(self) -> None:
        mismatches = (
            ("mae_vx", 0.15),
            ("mae_vy", 0.15),
            ("mae_wz", 0.15),
            ("success_ratio", 0.5),
            ("energy_mech_power_total_mean", 2.0),
            ("energy_effort_mean", 2.0),
            ("fall_count", 1),
        )
        for field, value in mismatches:
            with self.subTest(field=field):
                final = self.complete_with_evidence_mutation(
                    episode_mutator=lambda rows, field=field, value=value: rows[0].__setitem__(
                        field, value
                    )
                )

                self.assertEqual(final["state"], "blocked_safety")
                self.assertIn("aggregate mismatch", final["terminal_reason"])

    def test_evidence_is_parsed_from_the_exact_hashed_byte_buffers(self) -> None:
        armed = self.create_and_arm()
        self.service.tick()
        training = self.pending_training(armed["id"])[0]
        self.processes.complete_training(training["id"])
        self.service.tick()
        evaluation = self.pending_evaluations(armed["id"])[0]
        self.processes.complete_evaluation(
            evaluation["id"], passed=False, tracking=0.4
        )
        command_path = Path(str(evaluation["command_csv"])).resolve()
        original_read_bytes = Path.read_bytes
        swapped = False

        def read_then_swap(path: Path) -> bytes:
            nonlocal swapped
            content = original_read_bytes(path)
            if path.resolve() == command_path and not swapped:
                swapped = True
                path.write_text("malformed,after,hash\n", encoding="utf-8")
            return content

        with mock.patch.object(Path, "read_bytes", read_then_swap):
            self.service.tick()

        final = self.service.get_campaign(armed["id"])
        self.assertTrue(swapped)
        self.assertEqual(final["state"], "awaiting_advisor")
        self.assertEqual(len(final["evaluations"]), 1)

    def test_coherent_evaluation_artifact_rewrite_after_completion_fails_closed(self) -> None:
        armed = self.create_and_arm()
        self.service.tick()
        training = self.pending_training(armed["id"])[0]
        self.processes.complete_training(training["id"])
        self.service.tick()
        evaluation = self.pending_evaluations(armed["id"])[0]
        self.processes.complete_evaluation(
            evaluation["id"], passed=False, tracking=0.4
        )

        command_path = Path(evaluation["command_csv"])
        episode_path = Path(evaluation["episode_csv"])
        summary_path = Path(evaluation["summary_csv"])
        command_path.write_bytes(command_path.read_bytes() + b"\n")
        episode_path.write_bytes(episode_path.read_bytes() + b"\n")
        with summary_path.open(newline="", encoding="utf-8") as handle:
            summary_rows = list(csv.DictReader(handle))
        for row in summary_rows:
            if row["metric"] == "artifact.command_csv_sha256":
                row["value"] = hashlib.sha256(command_path.read_bytes()).hexdigest()
            elif row["metric"] == "artifact.episode_csv_sha256":
                row["value"] = hashlib.sha256(episode_path.read_bytes()).hexdigest()
        with summary_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["metric", "value"])
            writer.writeheader()
            writer.writerows(summary_rows)

        self.service.tick()
        final = self.service.get_campaign(armed["id"])

        self.assertEqual(final["state"], "blocked_safety")
        self.assertIn("changed after process completion", final["terminal_reason"])

    def test_all_evaluation_csv_types_reject_blank_or_duplicate_headers(self) -> None:
        for name in ("commands.csv", "episodes.csv", "summary.csv"):
            with self.subTest(name=name):
                with self.assertRaisesRegex(
                    AutopilotValidationError, "blank or duplicate headers"
                ):
                    self.service._read_csv_bytes(b"metric,metric\na,b\n", name)
                with self.assertRaisesRegex(
                    AutopilotValidationError, "blank or duplicate headers"
                ):
                    self.service._read_csv_bytes(b"metric,\na,b\n", name)

    def test_derived_command_metrics_cannot_be_tampered(self) -> None:
        mutations = (
            (
                "tracking",
                lambda rows: rows[0].__setitem__("tracking_quality", 0.9),
                "derived tracking_quality mismatch",
            ),
            (
                "stability",
                lambda rows: rows[0].__setitem__("stability_quality", 0.5),
                "derived stability_quality mismatch",
            ),
            (
                "acceptance",
                lambda rows: rows[0].__setitem__("accept_pass", True),
                "derived accept_pass mismatch",
            ),
            (
                "energy",
                lambda rows: rows[0].update(
                    energy_per_distance=2.0,
                    energy_power_per_motion=2.0,
                ),
                "aggregate mismatch",
            ),
        )
        for name, mutator, reason in mutations:
            with self.subTest(name=name):
                final = self.complete_with_evidence_mutation(command_mutator=mutator)

                self.assertEqual(final["state"], "blocked_safety")
                self.assertIn(reason, final["terminal_reason"])

    def test_weighted_episode_aggregates_allow_one_trailing_partial_episode(self) -> None:
        def split_first_environment(rows: list[dict[str, Any]]) -> None:
            first = rows[0]
            base = {
                key: float(first[key])
                for key in (
                    "mae_vx",
                    "mae_vy",
                    "mae_wz",
                    "success_ratio",
                    "energy_mech_power_total_mean",
                    "energy_effort_mean",
                )
            }
            first.update(
                {
                    "sample_count": int(first["sample_count"]) // 4,
                    **{key: value * 1.6 for key, value in base.items()},
                }
            )
            trailing = dict(first)
            trailing.update(
                {
                    "episode_index": 1,
                    "complete": False,
                    "sample_count": int(first["sample_count"]) * 3,
                    **{key: value * 0.8 for key, value in base.items()},
                }
            )
            rows.append(trailing)

        final = self.complete_with_evidence_mutation(
            episode_mutator=split_first_environment
        )

        self.assertEqual(final["state"], "awaiting_advisor")
        self.assertEqual(len(final["evaluations"]), 1)

    def test_truncated_episode_horizon_fails_closed(self) -> None:
        final = self.complete_with_evidence_mutation(
            episode_mutator=lambda rows: [
                row.__setitem__("sample_count", 1) for row in rows
            ]
        )

        self.assertEqual(final["state"], "blocked_safety")
        self.assertIn("sample horizon mismatch", final["terminal_reason"])

    def test_per_environment_episode_horizon_cannot_be_cross_compensated(self) -> None:
        def shift_samples_between_environments(rows: list[dict[str, Any]]) -> None:
            command = rows[0]["command"]
            peer = next(
                row
                for row in rows
                if row["command"] == command and row["environment_index"] != rows[0]["environment_index"]
            )
            rows[0]["sample_count"] += 1
            peer["sample_count"] -= 1

        final = self.complete_with_evidence_mutation(
            episode_mutator=shift_samples_between_environments
        )

        self.assertEqual(final["state"], "blocked_safety")
        self.assertIn("environment sample horizon mismatch", final["terminal_reason"])

    def test_inflated_evaluation_timestep_cannot_promote_low_success_ratio(self) -> None:
        armed = self.create_and_arm()
        self.service.tick()
        training = self.pending_training(armed["id"])[0]
        self.processes.complete_training(training["id"])
        self.service.tick()
        evaluation = self.pending_evaluations(armed["id"])[0]

        def inflate_duration(rows: list[dict[str, Any]]) -> None:
            for row in rows:
                row["success_duration_s"] = 2.0
                row["accept_pass"] = True

        self.processes.complete_evaluation(
            evaluation["id"],
            passed=False,
            tracking=0.4,
            command_mutator=inflate_duration,
            summary_overrides={
                "evaluation.step_dt": 1.0 / 30.0,
                "evaluation.duration_s": 20.0,
            },
        )
        self.service.tick()
        final = self.service.get_campaign(armed["id"])

        self.assertEqual(final["state"], "blocked_safety")
        self.assertIn("evaluation horizon identity mismatch", final["terminal_reason"])

    def test_malformed_and_nonfinite_csv_fail_closed_as_blocked_safety(self) -> None:
        for malformed in ("nonfinite", "missing_command"):
            with self.subTest(malformed=malformed):
                armed = self.create_and_arm()
                self.service.tick()
                training = self.pending_training(armed["id"])[0]
                self.processes.complete_training(training["id"])
                self.service.tick()
                evaluation = self.pending_evaluations(armed["id"])[0]
                self.processes.complete_evaluation(
                    evaluation["id"],
                    passed=True,
                    tracking=0.9,
                    malformed=malformed,
                )
                self.service.tick()
                final = self.service.get_campaign(armed["id"])

                self.assertEqual(final["state"], "blocked_safety")
                self.assertIn("invalid evaluation evidence", final["terminal_reason"])
                self.assertEqual(final["candidate_lineage"][0]["status"], "failed")

    def test_summary_identity_mismatch_and_command_profile_tamper_fail_closed(self) -> None:
        armed = self.create_and_arm()
        self.service.tick()
        training = self.pending_training(armed["id"])[0]
        self.processes.complete_training(training["id"])
        self.service.tick()
        evaluation = self.pending_evaluations(armed["id"])[0]
        self.processes.complete_evaluation(
            evaluation["id"],
            passed=True,
            tracking=0.9,
            summary_overrides={"checkpoint.strict_load": False},
        )
        self.service.tick()
        final = self.service.get_campaign(armed["id"])
        self.assertEqual(final["state"], "blocked_safety")
        self.assertIsNone(final["leader"]["trial_id"])

        second = self.create_and_arm()
        self.service.tick()
        training = self.pending_training(second["id"])[0]
        self.processes.complete_training(training["id"])
        self.service.tick()
        evaluation = self.pending_evaluations(second["id"])[0]
        self.processes.complete_evaluation(
            evaluation["id"], passed=True, tracking=0.9
        )
        runtime = self.service.store.get_runtime(second["id"])
        Path(runtime["command_profile_file"]).write_text("{}", encoding="utf-8")
        self.service.tick()
        final = self.service.get_campaign(second["id"])
        self.assertEqual(final["state"], "blocked_safety")

    def test_evaluation_without_strict_energy_evidence_fails_closed(self) -> None:
        armed = self.create_and_arm()
        self.service.tick()
        training = self.pending_training(armed["id"])[0]
        self.processes.complete_training(training["id"])
        self.service.tick()
        evaluation = self.pending_evaluations(armed["id"])[0]
        self.processes.complete_evaluation(
            evaluation["id"],
            passed=True,
            tracking=0.9,
            summary_overrides={"energy.strict_evidence": False},
        )

        self.service.tick()
        final = self.service.get_campaign(armed["id"])
        self.assertEqual(final["state"], "blocked_safety")
        self.assertIn("strict energy evidence", final["terminal_reason"])

    def test_evaluation_runtime_identity_mismatch_fails_closed(self) -> None:
        armed = self.create_and_arm()
        self.service.tick()
        training = self.pending_training(armed["id"])[0]
        self.processes.complete_training(training["id"])
        self.service.tick()
        evaluation = self.pending_evaluations(armed["id"])[0]
        self.processes.complete_evaluation(
            evaluation["id"],
            passed=True,
            tracking=0.9,
            summary_overrides={"identity.code_sha256": "f" * 64},
        )

        self.service.tick()
        final = self.service.get_campaign(armed["id"])
        self.assertEqual(final["state"], "blocked_safety")
        self.assertIn("runtime identity mismatch", final["terminal_reason"])

    def test_evaluation_uses_frozen_output_checkpoint_not_mutable_latest_pointer(self) -> None:
        armed = self.create_and_arm()
        self.service.tick()
        training = self.pending_training(armed["id"])[0]
        self.processes.complete_training(training["id"])
        exact_checkpoint = Path(str(training["output_checkpoint_path"])).resolve()

        replacement = exact_checkpoint.with_name("model_999.pt")
        replacement.write_bytes(b"unrelated later checkpoint")
        training["latest_checkpoint"] = str(replacement)

        self.service.tick()
        current = self.service.get_campaign(armed["id"])
        trial = current["candidate_lineage"][0]
        self.assertEqual(
            trial["metadata"]["output_checkpoint_path"], str(exact_checkpoint)
        )
        evaluation = self.pending_evaluations(armed["id"])[0]
        self.assertEqual(
            Path(str(evaluation["params"]["checkpoint"])).resolve(), exact_checkpoint
        )

        self.processes.complete_evaluation(
            evaluation["id"], passed=False, tracking=0.4
        )
        self.service.tick()

        final = self.service.get_campaign(armed["id"])
        self.assertEqual(final["state"], "awaiting_advisor")
        self.assertEqual(
            final["candidate_lineage"][0]["metadata"]["output_checkpoint_path"],
            str(exact_checkpoint),
        )

    def test_pure_yaw_energy_cannot_collapse_to_zero(self) -> None:
        command = {"name": "yaw_ccw", "skill": "yaw", "vx": 0.0, "vy": 0.0, "wz": 0.7}
        row = self.processes._command_row(command, passed=True, tracking=0.9)
        row["energy_per_distance"] = 0.0
        row["energy_power_per_motion"] = 0.0

        with self.assertRaisesRegex(AutopilotValidationError, "pure-yaw energy"):
            self.service._command_row(row, 0, evaluation_duration_s=10.0)

        row["energy_per_distance"] = row["energy_mech_power_total_mean"]
        row["energy_power_per_motion"] = row["energy_mech_power_total_mean"]
        parsed = self.service._command_row(row, 0, evaluation_duration_s=10.0)
        self.assertEqual(parsed["energy_per_distance"], 1.0)

    def test_diagonal_sign_and_yaw_tilt_are_part_of_derived_acceptance(self) -> None:
        cases = (
            (
                {"name": "diag", "skill": "diagonal", "vx": 0.3, "vy": 0.3, "wz": 0.0},
                "diag_sign_match_ratio",
            ),
            (
                {"name": "yaw", "skill": "yaw", "vx": 0.0, "vy": 0.0, "wz": 0.7},
                "yaw_tilt_ok_ratio",
            ),
        )
        for command, field in cases:
            with self.subTest(skill=command["skill"]):
                row = self.processes._command_row(
                    command, passed=True, tracking=0.9
                )
                row[field] = 0.69
                row["accept_pass"] = False
                parsed = self.service._command_row(
                    row, 0, evaluation_duration_s=10.0
                )
                self.assertFalse(parsed["accept_pass"])
                if command["skill"] == "diagonal":
                    self.assertEqual(parsed["direction_sign_ratio"], 0.69)

                row["accept_pass"] = True
                with self.assertRaisesRegex(
                    AutopilotValidationError, "derived accept_pass mismatch"
                ):
                    self.service._command_row(
                        row, 0, evaluation_duration_s=10.0
                    )

    def test_diagonal_success_duration_must_match_reconciled_ratio(self) -> None:
        command = {
            "name": "diag",
            "skill": "diagonal",
            "vx": 0.3,
            "vy": 0.3,
            "wz": 0.0,
        }
        row = self.processes._command_row(command, passed=False, tracking=0.9)
        row["success_duration_s"] = 9.0
        row["accept_pass"] = True

        with self.assertRaisesRegex(
            AutopilotValidationError, "success duration/ratio mismatch"
        ):
            self.service._command_row(
                row, 0, evaluation_duration_s=10.0
            )

    def test_code_identity_drift_stops_only_campaign_work_and_blocks(self) -> None:
        armed = self.create_and_arm()
        self.service.tick()
        training = self.pending_training(armed["id"])[0]
        source = self.root / "scripts/rsl_rl/train.py"
        source.write_text(source.read_text(encoding="utf-8") + "# changed\n", encoding="utf-8")

        self.service.tick()

        final = self.service.get_campaign(armed["id"])
        self.assertEqual(final["state"], "blocked_safety")
        self.assertIn("code_sha256", final["terminal_reason"])
        self.assertEqual(self.processes.stopped, [training["id"]])

    def test_controller_failure_stop_is_durable_across_false_and_exception(self) -> None:
        cases = (
            (
                "false",
                AutopilotValidationError("unsafe identity drift"),
                mock.Mock(return_value=False),
                "blocked_safety",
            ),
            (
                "exception",
                RuntimeError("controller exploded"),
                mock.Mock(side_effect=RuntimeError("stop transport failed")),
                "failed",
            ),
        )
        for name, controller_error, stop_mock, expected_state in cases:
            with self.subTest(name=name):
                armed = self.create_and_arm()
                self.service.tick()
                training = self.pending_training(armed["id"])[0]
                with mock.patch.object(
                    self.service,
                    "_assert_campaign_identity",
                    side_effect=controller_error,
                ), mock.patch.object(self.processes, "stop", stop_mock):
                    self.service.tick()

                interrupted = self.service.get_campaign(armed["id"])
                self.assertNotIn(interrupted["state"], TERMINAL_STATES)
                self.assertEqual(
                    interrupted["active_process"]["process_id"], training["id"]
                )
                self.assertIn(
                    "controller_failure_stop_intent",
                    self.service.store.get_runtime(armed["id"]),
                )

                self.service.close()
                self.service = AutopilotService(
                    self.paths,
                    self.history,  # type: ignore[arg-type]
                    self.processes,  # type: ignore[arg-type]
                    enabled=True,
                    start_worker=False,
                    identity_python=sys.executable,
                )
                self.service.tick()

                recovered = self.service.get_campaign(armed["id"])
                self.assertEqual(recovered["state"], expected_state)
                self.assertIsNone(recovered["active_process"])
                self.assertNotIn(
                    "controller_failure_stop_intent",
                    self.service.store.get_runtime(armed["id"]),
                )
                self.assertEqual(
                    self.history.get_run(training["id"])["status"], "stopped"
                )

    def test_identity_drift_prevents_draft_from_being_armed(self) -> None:
        draft = self.service.create_campaign(
            self.payload(), idempotency_key=self.key("drift-draft")
        )
        source = self.root / "scripts/rsl_rl/train.py"
        source.write_text(source.read_text(encoding="utf-8") + "# changed\n", encoding="utf-8")

        with self.assertRaisesRegex(AutopilotValidationError, "code_sha256"):
            self.service.arm_campaign(
                draft["id"],
                expected_revision=draft["revision"],
                idempotency_key=self.key("drift-arm"),
            )

        self.assertEqual(self.service.get_campaign(draft["id"])["state"], "draft")

    def test_controller_and_runner_identity_drift_prevent_arming(self) -> None:
        previously_omitted_inputs = (
            "tools/training_panel/training_panel/autopilot.py",
            "scripts/rsl_rl/runner_factory.py",
        )
        for relative in previously_omitted_inputs:
            with self.subTest(relative=relative):
                draft = self.service.create_campaign(
                    self.payload(), idempotency_key=self.key("identity-draft")
                )
                source = self.root / relative
                original = source.read_text(encoding="utf-8")
                source.write_text(original + "# identity drift\n", encoding="utf-8")
                try:
                    with self.assertRaisesRegex(
                        AutopilotValidationError, "code_sha256"
                    ):
                        self.service.arm_campaign(
                            draft["id"],
                            expected_revision=draft["revision"],
                            idempotency_key=self.key("identity-arm"),
                        )
                finally:
                    source.write_text(original, encoding="utf-8")

    def test_dependency_runtime_drift_prevents_arming(self) -> None:
        draft = self.service.create_campaign(
            self.payload(), idempotency_key=self.key("dependency-draft")
        )
        source_identities = source_code_identities(self.root)
        changed_identities = {
            **source_identities,
            "dependency": "f" * 64,
        }

        with mock.patch(
            "tools.training_panel.training_panel.autopilot_service.runtime_source_identities",
            return_value=(changed_identities, {"schema_version": "changed"}),
        ):
            with self.assertRaisesRegex(
                AutopilotValidationError, "dependency_sha256"
            ):
                self.service.arm_campaign(
                    draft["id"],
                    expected_revision=draft["revision"],
                    idempotency_key=self.key("dependency-arm"),
                )


if __name__ == "__main__":
    unittest.main()
