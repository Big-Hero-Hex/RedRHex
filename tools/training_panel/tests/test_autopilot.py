from __future__ import annotations

import math
import unittest

from tools.training_panel.training_panel.autopilot import (
    AgentDecisionV1,
    AutopilotValidationError,
    CampaignSnapshotV1,
    DIRECT_TASK,
    EvaluationReportV1,
    FORWARD_FAST_REWARD_KEYS,
    FORWARD_FAST_TASK,
    RewardCatalogEntryV1,
    autopilot_capabilities,
    build_reward_catalog,
    compile_command_envelope,
    compile_command_profile,
    compile_goal_spec,
    evaluate_report,
    evaluation_rank_key,
    reward_lattice_values,
    reward_move_lattice,
    validate_candidate_decision,
    validate_transition,
)


HASH_A = "a" * 64
HASH_B = "b" * 64
HASH_C = "c" * 64
HASH_D = "d" * 64
HASH_E = "e" * 64


def make_goal(*, initialization_mode: str = "fresh"):
    return compile_goal_spec(
        description="Walk forward while tracking the requested speed.",
        task=FORWARD_FAST_TASK,
        stage=1,
        gait="walk",
        per_trial_iteration_cap=100,
        initialization_mode=initialization_mode,
        baseline_run_id="baseline-run" if initialization_mode == "policy_only" else None,
        baseline_checkpoint_iteration=100 if initialization_mode == "policy_only" else None,
        checkpoint_sha256=HASH_A if initialization_mode == "policy_only" else None,
        physics_profile_sha256=HASH_C,
        spring_profile_sha256=HASH_D,
        code_sha256=HASH_E,
        config_sha256=HASH_A,
    )


def make_command(*, accept_pass: bool = True, energy: float = 1.0):
    return {
        "name": "forward_walk",
        "skill": "forward",
        "command": {"vx": 0.25, "vy": 0.0, "wz": 0.0},
        "accept_pass": accept_pass,
        "tracking_quality": 0.9,
        "stability_quality": 0.95,
        "fall_rate": 0.0,
        "energy_per_distance": energy,
        "direction_sign_ratio": 1.0,
        "linear_leak": 0.0,
        "yaw_leak": 0.0,
    }


def make_report(
    goal,
    *,
    report_id: str = "eval-1",
    checkpoint_sha256: str = HASH_B,
    command: dict | None = None,
    episode_metrics: tuple[dict, ...] = ({"episode": 1, "duration_s": 10.0, "fall_rate": 0.0},),
    strict_checkpoint_load: bool = True,
    failure_reason: str | None = None,
):
    return EvaluationReportV1(
        id=report_id,
        trial_id="trial-1",
        checkpoint_sha256=checkpoint_sha256,
        config_sha256=goal.config_sha256,
        reward_profile_sha256=HASH_E,
        physics_profile_sha256=goal.physics_profile_sha256,
        spring_profile_sha256=goal.spring_profile_sha256,
        command_profile_sha256=goal.command_profile_sha256,
        seed=42,
        evaluation_profile=goal.evaluation_profile,
        strict_checkpoint_load=strict_checkpoint_load,
        episode_artifact_sha256=HASH_D,
        command_metrics=(command or make_command(),),
        episode_metrics=episode_metrics,
        artifact_ids=("command-csv", "episode-csv"),
        failure_reason=failure_reason,
    )


class CommandGoalContractTests(unittest.TestCase):
    def test_walk_and_run_split_signed_stage_ranges(self):
        walk = compile_command_envelope(DIRECT_TASK, 2, "walk")
        run = compile_command_envelope(DIRECT_TASK, 2, "run")

        self.assertEqual(walk["vy"], ((-0.48, -0.32), (0.32, 0.48)))
        self.assertEqual(run["vy"], ((-0.64, -0.48), (0.48, 0.64)))
        self.assertEqual(walk["vx"], ((0.0, 0.0),))

    def test_goal_round_trip_preserves_disjoint_command_envelope(self):
        goal = make_goal(initialization_mode="policy_only")

        restored = type(goal).from_dict(goal.to_dict())

        self.assertEqual(restored, goal)
        self.assertEqual(restored.command_envelope["vx"], ((0.22, 0.32),))
        self.assertEqual(restored.checkpoint_sha256, HASH_A)

    def test_every_direct_stage_and_gait_compiles_exact_in_envelope_commands(self):
        for stage in range(1, 6):
            for gait in ("walk", "run"):
                with self.subTest(stage=stage, gait=gait):
                    envelope = compile_command_envelope(DIRECT_TASK, stage, gait)
                    profile = compile_command_profile(DIRECT_TASK, stage, gait)
                    self.assertEqual(profile["stage"], stage)
                    self.assertEqual(profile["gait"], gait)
                    self.assertTrue(profile["commands"])
                    self.assertEqual(
                        len({item["name"] for item in profile["commands"]}),
                        len(profile["commands"]),
                    )
                    for command in profile["commands"]:
                        for axis in ("vx", "vy", "wz"):
                            value = float(command[axis])
                            self.assertTrue(
                                any(low <= value <= high for low, high in envelope[axis]),
                                f"{command['name']}.{axis} is outside {envelope[axis]}",
                            )

    def test_command_profile_can_be_narrowed_to_requested_directions(self):
        profile = compile_command_profile(
            DIRECT_TASK,
            5,
            "run",
            directions=("forward_right", "yaw_cw"),
        )

        self.assertEqual(profile["directions"], ["forward_right", "yaw_cw"])
        self.assertEqual(
            {item["skill"] for item in profile["commands"]},
            {"diagonal", "yaw"},
        )
        self.assertTrue(all(item["vy"] < 0.0 for item in profile["commands"] if item["skill"] == "diagonal"))
        self.assertTrue(all(item["wz"] < 0.0 for item in profile["commands"] if item["skill"] == "yaw"))

    def test_policy_only_requires_exact_source_checkpoint_identity(self):
        with self.assertRaisesRegex(AutopilotValidationError, "requires baseline_run_id"):
            compile_goal_spec(
                description="Walk forward",
                task=FORWARD_FAST_TASK,
                stage=1,
                gait="walk",
                per_trial_iteration_cap=10,
                initialization_mode="policy_only",
                physics_profile_sha256=HASH_A,
                spring_profile_sha256=HASH_B,
                code_sha256=HASH_C,
                config_sha256=HASH_D,
            )

    def test_goal_cannot_relax_default_safety_gate(self):
        gates = {
            "min_command_pass_ratio": 0.70,
            "min_skill_pass_ratio": 0.60,
            "max_fall_rate": 0.21,
            "min_tracking_quality": 0.0,
            "min_stability_quality": 0.0,
            "min_direction_sign_ratio": 0.70,
            "max_linear_leak": 0.18,
            "max_yaw_leak": 0.35,
            "max_energy_per_distance": 500.0,
        }
        with self.assertRaisesRegex(AutopilotValidationError, "may not relax"):
            compile_goal_spec(
                description="Walk forward",
                task=FORWARD_FAST_TASK,
                stage=1,
                gait="walk",
                per_trial_iteration_cap=10,
                physics_profile_sha256=HASH_A,
                spring_profile_sha256=HASH_B,
                code_sha256=HASH_C,
                config_sha256=HASH_D,
                skill_gates=gates,
            )

    def test_capabilities_are_off_by_default_and_publish_profiles(self):
        capabilities = autopilot_capabilities()

        self.assertFalse(capabilities["enabled"])
        self.assertIn("stage1", capabilities["command_profiles"][FORWARD_FAST_TASK])
        self.assertEqual(
            capabilities["default_reward_keys"][FORWARD_FAST_TASK]["stage1"],
            list(FORWARD_FAST_REWARD_KEYS),
        )


class RewardDecisionContractTests(unittest.TestCase):
    def setUp(self):
        self.goal = make_goal()
        self.values = {
            "v2_reward_scales.forward_progress": 3.0,
            "v2_reward_scales.velocity_tracking": 6.0,
            "v2_reward_scales.axis_suppression": 2.0,
            "v2_reward_scales.height_maintain": 1.0,
            "v2_reward_scales.height_low_penalty": 1.5,
            "v2_reward_scales.leg_moving": 0.25,
            "v2_reward_scales.stall_penalty": -3.0,
            "v2_reward_scales.energy_per_distance": 0.0005,
        }
        self.catalog = build_reward_catalog(FORWARD_FAST_TASK, 1, self.values)

    def decision(self, value=3.3, key="v2_reward_scales.forward_progress"):
        return AgentDecisionV1(
            campaign_id="campaign-1",
            campaign_revision=3,
            evidence_ids=("eval-1",),
            hypothesis="More progress shaping may improve forward tracking.",
            action="propose_candidate",
            reward_key=key,
            proposed_value=value,
            expected_metric_effect="Increase minimum tracking quality.",
            rationale="The leader is stable but consistently undershoots vx.",
        )

    def test_one_known_bounded_weight_is_applied(self):
        candidate = validate_candidate_decision(self.decision(), self.goal, self.catalog, self.values)

        changed = [key for key in candidate if candidate[key] != self.values[key]]
        self.assertEqual(changed, ["v2_reward_scales.forward_progress"])
        self.assertEqual(candidate[changed[0]], 3.3)

    def test_unknown_out_of_bounds_boolean_and_nonfinite_values_are_rejected(self):
        with self.assertRaises(AutopilotValidationError):
            validate_candidate_decision(
                self.decision(key="v2_reward_scales.fall"), self.goal, self.catalog, self.values
            )
        with self.assertRaisesRegex(AutopilotValidationError, "outside"):
            validate_candidate_decision(self.decision(value=4.0), self.goal, self.catalog, self.values)
        with self.assertRaisesRegex(AutopilotValidationError, "boolean"):
            self.decision(value=True)
        with self.assertRaisesRegex(AutopilotValidationError, "finite"):
            self.decision(value=math.nan)

    def test_catalog_cannot_widen_bounds_or_tune_zero(self):
        with self.assertRaisesRegex(AutopilotValidationError, "80-120%"):
            RewardCatalogEntryV1(
                key="v2_reward_scales.forward_progress",
                description="Forward shaping",
                tasks=(FORWARD_FAST_TASK,),
                stages=(1,),
                start_value=3.0,
                minimum=2.0,
                maximum=3.6,
                sign="positive",
            )
        values = dict(self.values)
        values["v2_reward_scales.forward_progress"] = 0.0
        catalog = build_reward_catalog(FORWARD_FAST_TASK, 1, values)
        self.assertNotIn("v2_reward_scales.forward_progress", {entry.key for entry in catalog})

    def test_human_can_only_narrow_selected_reward_bounds(self):
        key = "v2_reward_scales.forward_progress"
        catalog = build_reward_catalog(
            FORWARD_FAST_TASK,
            1,
            self.values,
            enabled_keys=[key],
            narrowed_bounds={key: [2.7, 3.3]},
        )

        self.assertEqual((catalog[0].minimum, catalog[0].maximum), (2.7, 3.3))
        with self.assertRaisesRegex(AutopilotValidationError, "80-120%"):
            build_reward_catalog(
                FORWARD_FAST_TASK,
                1,
                self.values,
                enabled_keys=[key],
                narrowed_bounds={key: [2.0, 3.3]},
            )
        with self.assertRaisesRegex(AutopilotValidationError, "selected keys"):
            build_reward_catalog(
                FORWARD_FAST_TASK,
                1,
                self.values,
                enabled_keys=[key],
                narrowed_bounds={"v2_reward_scales.velocity_tracking": [5.0, 6.0]},
            )

    def test_finite_move_lattice_clips_to_narrowed_bounds(self):
        key = "v2_reward_scales.forward_progress"
        full_entry = next(entry for entry in self.catalog if entry.key == key)
        self.assertEqual(
            len(reward_lattice_values(full_entry)),
            5,
        )
        for actual, expected in zip(
            reward_lattice_values(full_entry),
            (2.4, 2.7, 3.0, 3.3, 3.6),
        ):
            self.assertAlmostEqual(actual, expected)

        narrowed = build_reward_catalog(
            FORWARD_FAST_TASK,
            1,
            self.values,
            enabled_keys=[key],
            narrowed_bounds={key: [2.85, 3.15]},
        )[0]
        self.assertEqual(reward_lattice_values(narrowed), (2.85, 3.0, 3.15))

    def test_move_lattice_excludes_current_and_attempted_points(self):
        decisions = (
            {
                "id": "decision-1",
                "action": "propose_candidate",
                "reward_key": "v2_reward_scales.forward_progress",
                "proposed_value": 3.3,
            },
        )

        lattice = reward_move_lattice(self.catalog, self.values, decisions)
        forward_values = [
            move["proposed_value"]
            for move in lattice["remaining"]
            if move["reward_key"] == "v2_reward_scales.forward_progress"
        ]

        self.assertEqual(len(lattice["attempted"]), 1)
        self.assertEqual(lattice["attempted"][0]["decision_id"], "decision-1")
        self.assertNotIn(3.0, forward_values)
        self.assertNotIn(3.3, forward_values)
        self.assertEqual(len(forward_values), 3)


class EvaluationContractTests(unittest.TestCase):
    def setUp(self):
        self.goal = make_goal(initialization_mode="policy_only")
        self.original = {"v2_reward_scales.forward_progress": 3.0}
        self.candidate = {"v2_reward_scales.forward_progress": 3.3}

    def evaluated(self, report):
        return evaluate_report(
            report,
            self.goal,
            expected_trial_checkpoint_sha256=HASH_B,
            original_reward_values=self.original,
            candidate_reward_values=self.candidate,
        )

    def test_output_checkpoint_is_not_compared_to_source_checkpoint(self):
        report = self.evaluated(make_report(self.goal, checkpoint_sha256=HASH_B))

        self.assertNotEqual(self.goal.checkpoint_sha256, report.checkpoint_sha256)
        self.assertTrue(report.hard_gates["checkpoint_identity"])
        self.assertTrue(report.ranking["eligible"])

    def test_wrong_trial_output_checkpoint_fails_closed(self):
        report = make_report(self.goal, checkpoint_sha256=HASH_C)

        evaluated = self.evaluated(report)

        self.assertFalse(evaluated.hard_gates["checkpoint_identity"])
        self.assertFalse(evaluated.ranking["eligible"])

    def test_saturated_energy_at_the_evaluator_ceiling_fails_closed(self):
        evaluated = self.evaluated(
            make_report(self.goal, command=make_command(energy=500.0))
        )

        self.assertFalse(evaluated.hard_gates["energy_per_distance"])
        self.assertFalse(evaluated.ranking["eligible"])

    def test_strict_load_and_per_episode_evidence_are_required(self):
        with self.assertRaisesRegex(AutopilotValidationError, "strict_checkpoint_load"):
            make_report(self.goal, strict_checkpoint_load=False)
        with self.assertRaisesRegex(AutopilotValidationError, "episode_metrics must be non-empty"):
            make_report(self.goal, episode_metrics=())

    def test_nonfinite_command_or_episode_evidence_is_rejected(self):
        invalid_command = make_command()
        invalid_command["tracking_quality"] = math.nan
        with self.assertRaisesRegex(AutopilotValidationError, "finite"):
            make_report(self.goal, command=invalid_command)
        with self.assertRaisesRegex(AutopilotValidationError, "finite"):
            make_report(self.goal, episode_metrics=({"energy": math.inf},))

    def test_failure_reason_is_a_hard_gate(self):
        evaluated = self.evaluated(make_report(self.goal, failure_reason="episode CSV was truncated"))

        self.assertFalse(evaluated.hard_gates["evaluation_complete"])
        self.assertFalse(evaluated.ranking["eligible"])

    def test_passing_candidate_outranks_failure_and_lower_energy_breaks_tie(self):
        lower_energy = self.evaluated(
            make_report(self.goal, report_id="eval-low", command=make_command(energy=1.0))
        )
        higher_energy = self.evaluated(
            make_report(self.goal, report_id="eval-high", command=make_command(energy=2.0))
        )
        failed = self.evaluated(
            make_report(
                self.goal,
                report_id="eval-failed",
                command=make_command(accept_pass=False, energy=0.1),
            )
        )

        ranked = sorted((failed, higher_energy, lower_energy), key=evaluation_rank_key, reverse=True)

        self.assertEqual([item.id for item in ranked], ["eval-low", "eval-high", "eval-failed"])

    def test_evaluation_round_trip_requires_new_identity_fields(self):
        report = make_report(self.goal)

        restored = EvaluationReportV1.from_dict(report.to_dict())

        self.assertEqual(restored, report)
        data = report.to_dict()
        del data["episode_artifact_sha256"]
        with self.assertRaisesRegex(AutopilotValidationError, "missing fields"):
            EvaluationReportV1.from_dict(data)


class CampaignStateContractTests(unittest.TestCase):
    def test_transitions_are_explicit_and_terminal_states_cannot_resume(self):
        validate_transition("draft", "armed")
        validate_transition("awaiting_advisor", "candidate_training")
        validate_transition("paused", "candidate_training", resume_state="candidate_training")
        with self.assertRaisesRegex(AutopilotValidationError, "must resume"):
            validate_transition("paused", "awaiting_advisor", resume_state="candidate_training")
        with self.assertRaisesRegex(AutopilotValidationError, "terminal"):
            validate_transition("simulation_goal_met", "armed")

    def test_campaign_snapshot_round_trip_exposes_permitted_actions(self):
        goal = make_goal()
        values = {key: 1.0 for key in FORWARD_FAST_REWARD_KEYS}
        values["v2_reward_scales.stall_penalty"] = -1.0
        catalog = build_reward_catalog(FORWARD_FAST_TASK, 1, values)
        snapshot = CampaignSnapshotV1(
            id="campaign-1",
            revision=0,
            state="draft",
            goal=goal,
            reward_catalog=catalog,
            leader={"candidate_id": "baseline", "reward_values": values, "evaluation_id": None},
            budget={
                "max_training_trials": 24,
                "max_gpu_hours": 72.0,
                "training_trials_used": 0,
                "gpu_hours_used": 0.0,
                "remaining_training_trials": 24,
                "remaining_gpu_hours": 72.0,
            },
            active_process=None,
            candidate_lineage=(),
            decisions=(),
            evaluations=(),
            connector={"last_heartbeat_at": None, "polls_used": 0, "max_polls": 300},
            resume_state=None,
            terminal_reason=None,
            created_at="2026-08-14T10:00:00Z",
            updated_at="2026-08-14T10:00:00Z",
        )

        data = snapshot.to_dict()
        restored = CampaignSnapshotV1.from_dict(data)

        self.assertEqual(restored, snapshot)
        self.assertEqual(data["next_permitted_actions"], ["update", "arm"])


if __name__ == "__main__":
    unittest.main()
