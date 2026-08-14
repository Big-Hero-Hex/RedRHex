from __future__ import annotations

import hashlib
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tools.training_panel.training_panel.autopilot import (
    AgentDecisionV1,
    AutopilotValidationError,
    FORWARD_FAST_TASK,
    build_reward_catalog,
    compile_goal_spec,
)
from tools.training_panel.training_panel.autopilot_store import (
    AutopilotBudgetError,
    AutopilotConflictError,
    AutopilotStore,
    AutopilotStoreError,
)


HASH_A = "a" * 64
HASH_B = "b" * 64
HASH_C = "c" * 64
HASH_D = "d" * 64


def make_goal():
    return compile_goal_spec(
        description="Walk forward with bounded reward tuning.",
        task=FORWARD_FAST_TASK,
        stage=1,
        gait="walk",
        per_trial_iteration_cap=100,
        physics_profile_sha256=HASH_A,
        spring_profile_sha256=HASH_B,
        code_sha256=HASH_C,
        config_sha256=HASH_D,
    )


def make_reward_values():
    return {
        "v2_reward_scales.forward_progress": 3.0,
        "v2_reward_scales.velocity_tracking": 6.0,
        "v2_reward_scales.axis_suppression": 2.0,
        "v2_reward_scales.height_maintain": 1.0,
        "v2_reward_scales.height_low_penalty": 1.5,
        "v2_reward_scales.leg_moving": 0.25,
        "v2_reward_scales.stall_penalty": -3.0,
        "v2_reward_scales.energy_per_distance": 0.0005,
    }


class AutopilotStoreTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        root = Path(self.temporary.name)
        self.database = root / "autopilot.sqlite3"
        self.artifacts = root / "artifacts"
        self.store = AutopilotStore(self.database, self.artifacts, enabled=True)
        self.goal = make_goal()
        self.reward_values = make_reward_values()
        self.catalog = build_reward_catalog(FORWARD_FAST_TASK, 1, self.reward_values)
        self.key_sequence = 0

    def tearDown(self):
        self.temporary.cleanup()

    def key(self, prefix="request"):
        self.key_sequence += 1
        return f"{prefix}-{self.key_sequence:04d}"

    def create(self):
        return self.store.create_campaign(
            self.goal,
            self.catalog,
            idempotency_key=self.key("create"),
        )

    def arm(self, campaign):
        return self.store.arm_campaign(
            campaign["id"],
            expected_revision=campaign["revision"],
            idempotency_key=self.key("arm"),
        )

    def test_wal_schema_and_append_only_triggers_are_installed(self):
        connection = sqlite3.connect(self.database)
        try:
            journal_mode = connection.execute("PRAGMA journal_mode").fetchone()[0]
            schema_version = connection.execute(
                "SELECT value FROM autopilot_meta WHERE key='schema_version'"
            ).fetchone()[0]
            trigger_names = {
                row[0]
                for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type='trigger'"
                ).fetchall()
            }
        finally:
            connection.close()

        self.assertEqual(journal_mode, "wal")
        self.assertEqual(schema_version, "1")
        self.assertIn("campaign_events_no_update", trigger_names)
        self.assertIn("campaign_events_no_delete", trigger_names)

    def test_create_is_idempotent_and_list_is_read_only(self):
        key = self.key("create")
        first = self.store.create_campaign(self.goal, self.catalog, idempotency_key=key)
        repeated = self.store.create_campaign(self.goal, self.catalog, idempotency_key=key)

        listed = self.store.list_campaigns()

        self.assertEqual(repeated, first)
        self.assertEqual([item["id"] for item in listed], [first["id"]])
        self.assertEqual(self.store.get_campaign(first["id"])["revision"], 0)
        self.assertEqual(len(self.store.list_events(first["id"])), 1)

    def test_only_one_campaign_can_hold_the_host_slot(self):
        first = self.create()
        second = self.create()
        self.arm(first)

        with self.assertRaisesRegex(AutopilotConflictError, "host execution slot"):
            self.arm(second)

        self.assertEqual(self.store.get_campaign(second["id"])["state"], "draft")
        self.assertEqual(self.store.get_campaign(second["id"])["revision"], 0)

    def test_arming_is_blocked_when_capability_is_disabled(self):
        root = Path(self.temporary.name)
        disabled = AutopilotStore(root / "disabled.sqlite3", root / "disabled-artifacts")
        campaign = disabled.create_campaign(
            self.goal,
            self.catalog,
            idempotency_key="disabled-create-0001",
        )

        with self.assertRaisesRegex(AutopilotConflictError, "disabled"):
            disabled.arm_campaign(
                campaign["id"],
                expected_revision=0,
                idempotency_key="disabled-arm-0001",
            )

    def test_stale_revision_does_not_write_an_event(self):
        campaign = self.create()

        with self.assertRaises(AutopilotConflictError) as caught:
            self.store.arm_campaign(
                campaign["id"],
                expected_revision=9,
                idempotency_key=self.key("arm"),
            )

        self.assertEqual(caught.exception.current_revision, 0)
        self.assertEqual(len(self.store.list_events(campaign["id"])), 1)

    def test_duplicate_trial_reservation_returns_prior_result_without_double_counting(self):
        campaign = self.arm(self.create())
        key = self.key("reserve")
        arguments = {
            "kind": "control",
            "seed": 42,
            "reward_profile": self.reward_values,
            "source_checkpoint_sha256": None,
            "expected_revision": campaign["revision"],
            "idempotency_key": key,
        }

        first = self.store.reserve_trial(campaign["id"], **arguments)
        repeated = self.store.reserve_trial(campaign["id"], **arguments)

        self.assertEqual(repeated, first)
        current = self.store.get_campaign(campaign["id"])
        self.assertEqual(len(current["candidate_lineage"]), 1)
        self.assertEqual(current["budget"]["used_training_trials"], 1)
        self.assertEqual(current["budget"]["remaining_training_trials"], 23)
        self.assertEqual(
            [event["type"] for event in self.store.list_events(campaign["id"])].count("trial_reserved"),
            1,
        )

        with self.assertRaisesRegex(AutopilotConflictError, "equivalent trial"):
            self.store.reserve_trial(
                campaign["id"],
                kind="control",
                seed=42,
                reward_profile=self.reward_values,
                source_checkpoint_sha256=None,
                expected_revision=current["revision"],
                idempotency_key=self.key("reserve"),
            )

    def test_snapshot_preserves_insertion_order_when_timestamps_are_equal(self):
        with mock.patch(
            "tools.training_panel.training_panel.autopilot_store._now",
            return_value="2026-08-14T00:00:00+00:00",
        ):
            current = self.arm(self.create())
            first = self.store.reserve_trial(
                current["id"],
                kind="control",
                seed=42,
                reward_profile=self.reward_values,
                source_checkpoint_sha256=None,
                expected_revision=current["revision"],
                idempotency_key=self.key("control"),
            )
            candidate_profile = dict(self.reward_values)
            candidate_profile["v2_reward_scales.forward_progress"] = 3.3
            second = self.store.reserve_trial(
                current["id"],
                kind="candidate",
                seed=42,
                reward_profile=candidate_profile,
                source_checkpoint_sha256=None,
                expected_revision=first["campaign"]["revision"],
                idempotency_key=self.key("candidate"),
            )
            third = self.store.reserve_trial(
                current["id"],
                kind="confirmation_control",
                seed=43,
                reward_profile=self.reward_values,
                source_checkpoint_sha256=None,
                expected_revision=second["campaign"]["revision"],
                idempotency_key=self.key("confirmation"),
            )

        snapshot = self.store.get_campaign(current["id"])
        self.assertEqual(
            [trial["id"] for trial in snapshot["candidate_lineage"]],
            [first["trial"]["id"], second["trial"]["id"], third["trial"]["id"]],
        )

    def test_screening_candidates_cannot_consume_four_confirmation_slots(self):
        current = self.arm(self.create())
        control = self.store.reserve_trial(
            current["id"],
            kind="control",
            seed=42,
            reward_profile=self.reward_values,
            source_checkpoint_sha256=None,
            expected_revision=current["revision"],
            idempotency_key=self.key("control"),
        )
        current = control["campaign"]

        for index in range(19):
            profile = dict(self.reward_values)
            profile["v2_reward_scales.forward_progress"] = 2.41 + index * 0.06
            reserved = self.store.reserve_trial(
                current["id"],
                kind="candidate",
                seed=42,
                reward_profile=profile,
                source_checkpoint_sha256=None,
                expected_revision=current["revision"],
                idempotency_key=self.key("candidate"),
            )
            current = reserved["campaign"]

        self.assertEqual(current["budget"]["used_training_trials"], 20)
        self.assertEqual(current["budget"]["remaining_training_trials"], 4)
        self.assertEqual(current["budget"]["reserved_confirmation_trials"], 4)
        profile = dict(self.reward_values)
        profile["v2_reward_scales.forward_progress"] = 3.1
        with self.assertRaisesRegex(AutopilotBudgetError, "reserved for confirmation"):
            self.store.reserve_trial(
                current["id"],
                kind="candidate",
                seed=42,
                reward_profile=profile,
                source_checkpoint_sha256=None,
                expected_revision=current["revision"],
                idempotency_key=self.key("candidate"),
            )

        confirmation_control = self.store.reserve_trial(
            current["id"],
            kind="confirmation_control",
            seed=43,
            reward_profile=self.reward_values,
            source_checkpoint_sha256=None,
            expected_revision=current["revision"],
            idempotency_key=self.key("confirmation"),
        )
        confirmation_candidate = self.store.reserve_trial(
            current["id"],
            kind="confirmation_candidate",
            seed=43,
            reward_profile=self.reward_values,
            source_checkpoint_sha256=None,
            expected_revision=confirmation_control["campaign"]["revision"],
            idempotency_key=self.key("confirmation"),
        )

        self.assertEqual(
            confirmation_candidate["campaign"]["budget"]["reserved_confirmation_trials"],
            2,
        )
        self.assertEqual(
            confirmation_candidate["campaign"]["budget"]["remaining_training_trials"],
            2,
        )

    def test_artifact_is_content_addressed_and_verified_on_read(self):
        campaign = self.create()
        content = b'{"metric":"tracking","value":0.9}'

        first = self.store.store_artifact(
            campaign["id"],
            kind="evaluation_json",
            content=content,
            media_type="application/json",
            metadata={"trial_id": "trial-1"},
        )
        repeated = self.store.store_artifact(
            campaign["id"],
            kind="evaluation_json",
            content=content,
            media_type="application/json",
            metadata={"trial_id": "trial-1"},
        )
        metadata, restored = self.store.get_artifact(campaign["id"], first["id"])

        self.assertEqual(first, repeated)
        self.assertEqual(metadata["sha256"], hashlib.sha256(content).hexdigest())
        self.assertEqual(restored, content)
        self.assertEqual(self.store.list_artifacts(campaign["id"]), [first])

        artifact_path = self.artifacts / first["sha256"][:2] / first["sha256"]
        artifact_path.write_bytes(b"tampered")
        with self.assertRaisesRegex(AutopilotStoreError, "hash verification"):
            self.store.get_artifact(campaign["id"], first["id"])

    def test_decision_context_contains_bounds_recent_decisions_and_budget(self):
        current = self.arm(self.create())
        for state in ("control_training", "control_evaluating", "awaiting_advisor"):
            current = self.store.transition_campaign(
                current["id"],
                state,
                expected_revision=current["revision"],
                idempotency_key=self.key("transition"),
            )
        decision = AgentDecisionV1(
            campaign_id=current["id"],
            campaign_revision=current["revision"],
            evidence_ids=(),
            hypothesis="Increase progress shaping to reduce undershoot.",
            action="propose_candidate",
            reward_key="v2_reward_scales.forward_progress",
            proposed_value=3.3,
            expected_metric_effect="Improve forward tracking quality.",
            rationale="The bounded move preserves the sign and current safety contract.",
        )
        current = self.store.record_decision(
            current["id"],
            decision,
            expected_revision=current["revision"],
            idempotency_key=self.key("decision"),
        )

        context = self.store.decision_context(current["id"])

        self.assertEqual(context["campaign_revision"], current["revision"])
        self.assertEqual(len(context["constraints"]), len(self.catalog))
        self.assertEqual(
            context["campaign_start_reward_values"], self.reward_values
        )
        self.assertTrue(
            all(value == 0.0 for value in context["baseline_to_leader_reward_deltas"].values())
        )
        self.assertEqual(context["remaining_allowable_move_count"], 31)
        self.assertEqual(len(context["remaining_allowable_moves"]), 31)
        self.assertEqual(
            context["attempted_moves"],
            [
                {
                    "reward_key": "v2_reward_scales.forward_progress",
                    "proposed_value": 3.3,
                    "decision_id": context["recent_decisions"][-1]["id"],
                }
            ],
        )
        self.assertEqual(context["recent_decisions"][-1]["action"], "propose_candidate")
        self.assertEqual(context["remaining_budget"]["training_trials"], 24)
        self.assertIn("propose_candidate", context["next_permitted_actions"])

    def test_decision_store_rejects_off_lattice_bounded_value(self):
        current = self.arm(self.create())
        for state in ("control_training", "control_evaluating", "awaiting_advisor"):
            current = self.store.transition_campaign(
                current["id"],
                state,
                expected_revision=current["revision"],
                idempotency_key=self.key("transition"),
            )
        decision = AgentDecisionV1(
            campaign_id=current["id"],
            campaign_revision=current["revision"],
            evidence_ids=(),
            hypothesis="Try an arbitrary bounded value.",
            action="propose_candidate",
            reward_key="v2_reward_scales.forward_progress",
            proposed_value=3.2,
            expected_metric_effect="Change tracking.",
            rationale="This value is bounded but is not a V1 lattice point.",
        )

        with self.assertRaisesRegex(AutopilotValidationError, "finite approved"):
            self.store.record_decision(
                current["id"],
                decision,
                expected_revision=current["revision"],
                idempotency_key=self.key("off-lattice"),
            )

        unchanged = self.store.get_campaign(current["id"])
        self.assertEqual(unchanged["revision"], current["revision"])
        self.assertEqual(unchanged["decisions"], [])

    def test_corrupt_and_nonfinite_persisted_data_fail_closed(self):
        corrupt = self.create()
        connection = sqlite3.connect(self.database)
        try:
            connection.execute("UPDATE campaigns SET goal_json='{' WHERE id=?", (corrupt["id"],))
            connection.commit()
        finally:
            connection.close()
        with self.assertRaises(json.JSONDecodeError):
            self.store.get_campaign(corrupt["id"])

        finite = self.create()
        budget = dict(finite["budget"])
        budget["used_gpu_hours"] = float("nan")
        connection = sqlite3.connect(self.database)
        try:
            connection.execute(
                "UPDATE campaigns SET budget_json=? WHERE id=?",
                (json.dumps(budget), finite["id"]),
            )
            connection.commit()
        finally:
            connection.close()
        with self.assertRaisesRegex(AutopilotValidationError, "finite"):
            self.store.get_campaign(finite["id"])

    def test_nonfinite_reservation_rolls_back_without_budget_or_event(self):
        campaign = self.arm(self.create())
        profile = dict(self.reward_values)
        profile["v2_reward_scales.forward_progress"] = float("nan")

        with self.assertRaisesRegex(ValueError, "finite"):
            self.store.reserve_trial(
                campaign["id"],
                kind="candidate",
                seed=42,
                reward_profile=profile,
                source_checkpoint_sha256=None,
                expected_revision=campaign["revision"],
                idempotency_key=self.key("reserve"),
            )

        current = self.store.get_campaign(campaign["id"])
        self.assertEqual(current["budget"]["used_training_trials"], 0)
        self.assertEqual(current["candidate_lineage"], [])

    def test_events_cannot_be_updated_or_deleted(self):
        campaign = self.arm(self.create())
        before = self.store.list_events(campaign["id"])
        connection = sqlite3.connect(self.database)
        try:
            with self.assertRaisesRegex(sqlite3.IntegrityError, "append-only"):
                connection.execute(
                    "UPDATE campaign_events SET event_type='rewritten' WHERE campaign_id=?",
                    (campaign["id"],),
                )
            connection.rollback()
            with self.assertRaisesRegex(sqlite3.IntegrityError, "append-only"):
                connection.execute(
                    "DELETE FROM campaign_events WHERE campaign_id=?",
                    (campaign["id"],),
                )
            connection.rollback()
        finally:
            connection.close()

        after = self.store.list_events(campaign["id"])
        self.assertEqual(after, before)
        self.assertEqual([item["sequence"] for item in after], sorted(item["sequence"] for item in after))
        self.assertTrue(all(item["campaign_id"] == campaign["id"] for item in after))

    def test_process_gpu_high_water_mark_survives_retries_and_store_restart(self):
        current = self.arm(self.create())
        reserved = self.store.reserve_trial(
            current["id"],
            kind="control",
            seed=42,
            reward_profile=self.reward_values,
            source_checkpoint_sha256=None,
            expected_revision=current["revision"],
            idempotency_key=self.key("reserve"),
        )
        bound = self.store.update_trial(
            current["id"],
            reserved["trial"]["id"],
            expected_revision=reserved["campaign"]["revision"],
            idempotency_key=self.key("bind"),
            status="training",
            run_id="train-attempt-1",
            active_process={
                "kind": "training",
                "process_id": "train-attempt-1",
                "trial_id": reserved["trial"]["id"],
            },
        )
        key = self.key("gpu")
        first = self.store.account_process_gpu_usage(
            current["id"],
            reserved["trial"]["id"],
            process_id="train-attempt-1",
            process_kind="training",
            cumulative_gpu_hours=0.5,
            force=True,
            expected_revision=bound["campaign"]["revision"],
            idempotency_key=key,
        )
        duplicate = self.store.account_process_gpu_usage(
            current["id"],
            reserved["trial"]["id"],
            process_id="train-attempt-1",
            process_kind="training",
            cumulative_gpu_hours=0.5,
            force=True,
            expected_revision=bound["campaign"]["revision"],
            idempotency_key=key,
        )
        self.assertEqual(duplicate, first)

        restarted = AutopilotStore(self.database, self.artifacts, enabled=True)
        second = restarted.account_process_gpu_usage(
            current["id"],
            reserved["trial"]["id"],
            process_id="train-attempt-1",
            process_kind="training",
            cumulative_gpu_hours=0.75,
            force=True,
            expected_revision=first["campaign"]["revision"],
            idempotency_key=self.key("gpu"),
        )

        self.assertAlmostEqual(second["campaign"]["budget"]["used_gpu_hours"], 0.75)
        marker = second["trial"]["metadata"]["gpu_process_accounting"]["train-attempt-1"]
        self.assertEqual(marker["kind"], "training")
        self.assertAlmostEqual(marker["accounted_gpu_hours"], 0.75)

    def test_connector_poll_cap_transitions_to_terminal_budget_state_idempotently(self):
        current = self.arm(self.create())
        budget = dict(current["budget"])
        budget["connector_polls"] = budget["max_connector_polls"] - 1
        connection = sqlite3.connect(self.database)
        try:
            connection.execute(
                "UPDATE campaigns SET budget_json=? WHERE id=?",
                (json.dumps(budget, sort_keys=True), current["id"]),
            )
            connection.commit()
        finally:
            connection.close()

        key = self.key("heartbeat")
        arguments = {
            "expected_revision": current["revision"],
            "idempotency_key": key,
            "prompt_version": "prompt-v1",
            "skill_version": "skill-v1",
            "declared_model": "test-model",
            "reasoning_effort": "medium",
            "metadata_schema": "redrhex.autopilot.advisor-metadata.v1",
        }
        exhausted = self.store.record_connector_heartbeat(current["id"], **arguments)
        duplicate = self.store.record_connector_heartbeat(current["id"], **arguments)

        self.assertEqual(duplicate, exhausted)
        self.assertEqual(exhausted["state"], "budget_exhausted")
        self.assertEqual(exhausted["budget"]["connector_polls"], 300)
        self.assertEqual(exhausted["terminal_reason"], "Connector poll budget is exhausted")
        self.assertIsNone(exhausted["resume_state"])
        events = self.store.list_events(current["id"])
        self.assertEqual(
            [event["type"] for event in events].count("connector_poll_budget_exhausted"),
            1,
        )

    def test_last_allowed_advisor_decision_is_recorded_then_campaign_terminates(self):
        current = self.arm(self.create())
        for state in ("control_training", "control_evaluating", "awaiting_advisor"):
            current = self.store.transition_campaign(
                current["id"],
                state,
                expected_revision=current["revision"],
                idempotency_key=self.key("transition"),
            )
        budget = dict(current["budget"])
        budget["connector_polls"] = budget["max_connector_polls"] - 1
        connection = sqlite3.connect(self.database)
        try:
            connection.execute(
                "UPDATE campaigns SET budget_json=? WHERE id=?",
                (json.dumps(budget, sort_keys=True), current["id"]),
            )
            connection.commit()
        finally:
            connection.close()
        decision = AgentDecisionV1(
            campaign_id=current["id"],
            campaign_revision=current["revision"],
            evidence_ids=(),
            hypothesis="No more autonomous polling is permitted.",
            action="pause",
            rationale="Stop at the durable connector guardrail.",
        )
        key = self.key("decision")

        exhausted = self.store.record_decision(
            current["id"],
            decision,
            expected_revision=current["revision"],
            idempotency_key=key,
        )
        duplicate = self.store.record_decision(
            current["id"],
            decision,
            expected_revision=current["revision"],
            idempotency_key=key,
        )

        self.assertEqual(duplicate, exhausted)
        self.assertEqual(exhausted["state"], "budget_exhausted")
        self.assertEqual(len(exhausted["decisions"]), 1)
        self.assertEqual(exhausted["next_permitted_actions"], [])


if __name__ == "__main__":
    unittest.main()
