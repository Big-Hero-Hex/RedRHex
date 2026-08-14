from __future__ import annotations

import io
import json
import os
import unittest
from email.message import Message
from types import SimpleNamespace
from unittest.mock import patch

from tools.training_panel.training_panel.autopilot_store import (
    AutopilotConflictError,
    CampaignNotFoundError,
)
from tools.training_panel.training_panel import server as server_module
from tools.training_panel.training_panel.server import PanelHandler, PanelState


ADVISOR_METADATA = {
    "schema_version": "redrhex.autopilot.advisor-metadata.v1",
    "skill_version": "redrhex-autopilot/0.1.0",
    "prompt_version": "scheduled-advisor.v1",
    "declared_model": "gpt-5.6-terra",
    "reasoning_effort": "medium",
}


class FakeAutopilotService:
    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def capabilities(self):
        self.calls.append(("capabilities",))
        return {"schema_version": "redrhex.autopilot.capabilities.v1", "enabled": True}

    def list_campaigns(self, *, state=None, limit=100):
        self.calls.append(("list_campaigns", state, limit))
        return [{"id": "campaign-1", "state": state or "draft"}]

    def get_campaign(self, campaign_id):
        self.calls.append(("get_campaign", campaign_id))
        return {"id": campaign_id, "revision": 3, "state": "draft"}

    def decision_context(self, campaign_id):
        self.calls.append(("decision_context", campaign_id))
        return {
            "schema_version": "redrhex.autopilot.decision-context.v1",
            "campaign_id": campaign_id,
        }

    def list_events(self, campaign_id, *, after=0, limit=500):
        self.calls.append(("list_events", campaign_id, after, limit))
        return [{"sequence": 1, "campaign_id": campaign_id}]

    def list_artifacts(self, campaign_id):
        self.calls.append(("list_artifacts", campaign_id))
        return [{"id": "artifact-1", "campaign_id": campaign_id}]

    def get_artifact(self, campaign_id, artifact_id):
        self.calls.append(("get_artifact", campaign_id, artifact_id))
        return (
            {
                "id": artifact_id,
                "campaign_id": campaign_id,
                "media_type": "application/json",
                "sha256": "a" * 64,
            },
            b'{"immutable":true}\n',
        )

    def compare_trials(self, campaign_id, trial_ids):
        self.calls.append(("compare_trials", campaign_id, list(trial_ids)))
        return {"campaign_id": campaign_id, "trial_ids": list(trial_ids)}

    def patch_export(self, campaign_id):
        self.calls.append(("patch_export", campaign_id))
        return (
            {"id": "artifact-patch", "media_type": "application/json"},
            b'{"source_mutated":false}\n',
        )

    def create_campaign(self, payload, *, idempotency_key):
        self.calls.append(("create_campaign", payload, idempotency_key))
        return {"id": "campaign-created", "revision": 0, "state": "draft"}

    def advisor_heartbeat(
        self,
        campaign_id,
        *,
        advisor_metadata,
        expected_revision,
        idempotency_key,
    ):
        self.calls.append(
            (
                "advisor_heartbeat",
                campaign_id,
                advisor_metadata,
                expected_revision,
                idempotency_key,
            )
        )
        return {
            "id": campaign_id,
            "revision": expected_revision + 1,
            "state": "control_training",
        }

    def update_draft(self, campaign_id, payload, *, expected_revision, idempotency_key):
        self.calls.append(
            ("update_draft", campaign_id, payload, expected_revision, idempotency_key)
        )
        return {"id": campaign_id, "revision": expected_revision + 1, "state": "draft"}

    def arm_campaign(self, campaign_id, *, expected_revision, idempotency_key):
        self.calls.append(("arm_campaign", campaign_id, expected_revision, idempotency_key))
        return {"id": campaign_id, "revision": expected_revision + 1, "state": "armed"}

    def pause_campaign(
        self,
        campaign_id,
        *,
        expected_revision,
        idempotency_key,
        reason="operator request",
        advisor_metadata=None,
    ):
        self.calls.append(
            (
                "pause_campaign",
                campaign_id,
                expected_revision,
                idempotency_key,
                reason,
                advisor_metadata,
            )
        )
        return {"id": campaign_id, "revision": expected_revision + 1, "state": "paused"}

    def resume_campaign(self, campaign_id, *, expected_revision, idempotency_key):
        self.calls.append(("resume_campaign", campaign_id, expected_revision, idempotency_key))
        return {"id": campaign_id, "revision": expected_revision + 1, "state": "armed"}

    def stop_campaign(
        self,
        campaign_id,
        *,
        expected_revision,
        idempotency_key,
        after_current=True,
        reason="operator request",
        advisor_metadata=None,
    ):
        self.calls.append(
            (
                "stop_campaign",
                campaign_id,
                expected_revision,
                idempotency_key,
                after_current,
                reason,
                advisor_metadata,
            )
        )
        return {"id": campaign_id, "revision": expected_revision + 1, "state": "stopped"}

    def submit_decision(
        self,
        campaign_id,
        payload,
        *,
        expected_revision,
        idempotency_key,
    ):
        self.calls.append(
            (
                "submit_decision",
                campaign_id,
                payload,
                expected_revision,
                idempotency_key,
            )
        )
        return {"id": campaign_id, "revision": expected_revision + 1, "state": "candidate_training"}

    def submit_patch_proposal(
        self,
        campaign_id,
        decision,
        patch_proposal,
        *,
        advisor_metadata,
        expected_revision,
        idempotency_key,
    ):
        self.calls.append(
            (
                "submit_patch_proposal",
                campaign_id,
                decision,
                patch_proposal,
                advisor_metadata,
                expected_revision,
                idempotency_key,
            )
        )
        return {"id": campaign_id, "revision": expected_revision + 1, "state": "patch_handoff"}


def _handler(
    method: str,
    path: str,
    service,
    payload: dict | None = None,
    headers: list[tuple[str, str]] | None = None,
):
    handler = object.__new__(PanelHandler)
    handler.command = method
    handler.path = path
    handler.state = SimpleNamespace(autopilot=service)
    body = b"" if payload is None else json.dumps(payload).encode("utf-8")
    message = Message()
    if payload is not None:
        message.add_header("Content-Type", "application/json")
        message.add_header("Content-Length", str(len(body)))
    for name, value in headers or []:
        message.add_header(name, value)
    handler.headers = message
    handler.rfile = io.BytesIO(body)
    handler.wfile = io.BytesIO()
    handler.json_responses = []
    handler.response_statuses = []
    handler.response_headers = []
    handler._json = lambda value, status=200: handler.json_responses.append((value, status))
    handler.send_response = lambda status: handler.response_statuses.append(status)
    handler.send_header = lambda name, value: handler.response_headers.append((name, value))
    handler.end_headers = lambda: None
    return handler


def _write_headers(revision: int, key: str = "request-key-0001") -> list[tuple[str, str]]:
    return [
        ("Idempotency-Key", key),
        ("If-Match", f'"{revision}"'),
        ("X-Expected-Revision", str(revision)),
    ]


class AutopilotReadRouteTests(unittest.TestCase):
    def test_missing_service_advertises_disabled_capability(self):
        handler = _handler("GET", "/api/autopilot/capabilities", None)

        handler._do_GET()

        payload, status = handler.json_responses[0]
        self.assertEqual(status, 200)
        self.assertFalse(payload["enabled"])
        self.assertEqual(payload["service_state"], "unavailable")

    def test_read_routes_map_to_the_service_without_mutation(self):
        service = FakeAutopilotService()
        requests = [
            ("/api/autopilot/capabilities", "capabilities"),
            ("/api/autopilot/campaigns?state=awaiting_advisor&limit=5", "list_campaigns"),
            ("/api/autopilot/campaigns/campaign-1", "get_campaign"),
            ("/api/autopilot/campaigns/campaign-1/decision-context", "decision_context"),
            ("/api/autopilot/campaigns/campaign-1/events", "list_events"),
            ("/api/autopilot/campaigns/campaign-1/artifacts", "list_artifacts"),
            (
                "/api/autopilot/campaigns/campaign-1/compare?trial_ids=trial-1,trial-2",
                "compare_trials",
            ),
        ]

        for path, expected_call in requests:
            with self.subTest(path=path):
                service.calls.clear()
                handler = _handler("GET", path, service)
                handler._do_GET()
                self.assertEqual(handler.json_responses[0][1], 200)
                self.assertEqual(service.calls[0][0], expected_call)

    def test_artifact_link_omits_content_and_download_is_hash_verified_by_service(self):
        service = FakeAutopilotService()
        detail = _handler(
            "GET",
            "/api/autopilot/campaigns/campaign-1/artifacts/artifact-1",
            service,
        )

        detail._do_GET()

        payload, status = detail.json_responses[0]
        self.assertEqual(status, 200)
        self.assertEqual(
            payload["artifact"]["download_url"],
            "/api/autopilot/campaigns/campaign-1/artifacts/artifact-1/download",
        )
        self.assertNotIn("content", payload["artifact"])

        download = _handler(
            "GET",
            "/api/autopilot/campaigns/campaign-1/artifacts/artifact-1/download",
            service,
        )
        download._do_GET()
        self.assertEqual(download.response_statuses, [200])
        self.assertEqual(download.wfile.getvalue(), b'{"immutable":true}\n')
        self.assertIn(("Cache-Control", "no-store"), download.response_headers)

    def test_event_log_supports_bounded_cursor_pagination(self):
        service = FakeAutopilotService()
        handler = _handler(
            "GET",
            "/api/autopilot/campaigns/campaign-1/events?after=500&limit=125",
            service,
        )

        handler._do_GET()

        self.assertEqual(handler.json_responses[0][1], 200)
        self.assertEqual(service.calls, [("list_events", "campaign-1", 500, 125)])

        for query in ("after=-1", "limit=0", "limit=501", "after=nope"):
            with self.subTest(query=query):
                invalid = _handler(
                    "GET",
                    f"/api/autopilot/campaigns/campaign-1/events?{query}",
                    service,
                )
                invalid._do_GET()
                self.assertEqual(invalid.json_responses[0][1], 400)

    def test_patch_export_streams_service_artifact_without_applying_it(self):
        service = FakeAutopilotService()
        handler = _handler(
            "GET",
            "/api/autopilot/campaigns/campaign-1/patch-export",
            service,
        )

        handler._do_GET()

        self.assertEqual(handler.response_statuses, [200])
        self.assertEqual(handler.wfile.getvalue(), b'{"source_mutated":false}\n')
        self.assertIn(("Cache-Control", "no-store"), handler.response_headers)

    def test_compare_rejects_more_than_twelve_trials(self):
        service = FakeAutopilotService()
        trial_ids = ",".join(f"trial-{index}" for index in range(13))
        handler = _handler(
            "GET",
            f"/api/autopilot/campaigns/campaign-1/compare?trial_ids={trial_ids}",
            service,
        )

        handler._do_GET()

        self.assertEqual(handler.json_responses[0][1], 400)
        self.assertEqual(handler.json_responses[0][0]["code"], "invalid_query")
        self.assertEqual(service.calls, [])


class AutopilotWriteRouteTests(unittest.TestCase):
    def test_create_uses_flat_ui_payload_and_revision_zero(self):
        service = FakeAutopilotService()
        payload = {
            "schema_version": "redrhex.autopilot.goal.v1",
            "expected_revision": 0,
            "description": "Walk forward",
            "tunable_reward_keys": ["v2_reward_scales.velocity_tracking"],
        }
        handler = _handler(
            "POST",
            "/api/autopilot/campaigns",
            service,
            payload,
            _write_headers(0),
        )

        handler._do_POST()

        result, status = handler.json_responses[0]
        self.assertEqual(status, 201)
        self.assertEqual(result["campaign"]["id"], "campaign-created")
        self.assertEqual(
            service.calls,
            [
                (
                    "create_campaign",
                    {
                        "schema_version": "redrhex.autopilot.goal.v1",
                        "description": "Walk forward",
                        "tunable_reward_keys": ["v2_reward_scales.velocity_tracking"],
                    },
                    "request-key-0001",
                )
            ],
        )

    def test_heartbeat_is_a_metadata_only_revisioned_mutation(self):
        service = FakeAutopilotService()
        handler = _handler(
            "POST",
            "/api/autopilot/campaigns/campaign-1/heartbeat",
            service,
            {
                "expected_revision": 8,
                "advisor_metadata": ADVISOR_METADATA,
            },
            _write_headers(8, "request-heartbeat-1"),
        )

        handler._do_POST()

        payload, status = handler.json_responses[0]
        self.assertEqual(status, 200)
        self.assertEqual(payload["campaign"]["revision"], 9)
        self.assertEqual(
            service.calls,
            [
                (
                    "advisor_heartbeat",
                    "campaign-1",
                    ADVISOR_METADATA,
                    8,
                    "request-heartbeat-1",
                )
            ],
        )

    def test_patch_and_lifecycle_routes_use_concrete_service_methods(self):
        service = FakeAutopilotService()
        cases = [
            (
                "PATCH",
                "/api/autopilot/campaigns/campaign-1",
                {"schema_version": "redrhex.autopilot.goal.v1", "description": "Updated"},
                "update_draft",
            ),
            ("POST", "/api/autopilot/campaigns/campaign-1/arm", {}, "arm_campaign"),
            (
                "POST",
                "/api/autopilot/campaigns/campaign-1/pause",
                {
                    "reason": "  evidence review  ",
                    "advisor_metadata": ADVISOR_METADATA,
                },
                "pause_campaign",
            ),
            ("POST", "/api/autopilot/campaigns/campaign-1/resume", {}, "resume_campaign"),
            (
                "POST",
                "/api/autopilot/campaigns/campaign-1/stop",
                {
                    "mode": "emergency",
                    "reason": "local operator",
                    "advisor_metadata": ADVISOR_METADATA,
                },
                "stop_campaign",
            ),
        ]
        recorded_calls = {}
        for index, (method, path, fields, expected_call) in enumerate(cases):
            with self.subTest(path=path):
                service.calls.clear()
                payload = {"expected_revision": 3, **fields}
                handler = _handler(
                    method,
                    path,
                    service,
                    payload,
                    _write_headers(3, f"request-key-{index:04d}"),
                )
                getattr(handler, f"_do_{method}")()
                self.assertEqual(handler.json_responses[0][1], 200)
                self.assertEqual(service.calls[0][0], expected_call)
                recorded_calls[expected_call] = service.calls[0]

        self.assertEqual(recorded_calls["pause_campaign"][4], "evidence review")
        self.assertEqual(recorded_calls["pause_campaign"][5], ADVISOR_METADATA)
        self.assertFalse(recorded_calls["stop_campaign"][4])
        self.assertEqual(recorded_calls["stop_campaign"][6], ADVISOR_METADATA)

    def test_candidate_and_patch_wrapper_share_validated_decision_surface(self):
        service = FakeAutopilotService()
        decision = {
            "schema_version": "redrhex.autopilot.decision.v1",
            "campaign_id": "campaign-1",
            "campaign_revision": 3,
            "action": "propose_candidate",
        }
        candidate = _handler(
            "POST",
            "/api/autopilot/campaigns/campaign-1/decisions",
            service,
            {
                "expected_revision": 3,
                "decision": decision,
                "advisor_metadata": ADVISOR_METADATA,
            },
            _write_headers(3),
        )
        candidate._do_POST()
        submitted = service.calls[-1]
        self.assertEqual(submitted[0], "submit_decision")
        self.assertNotIn("expected_revision", submitted[2])
        self.assertEqual(submitted[2]["advisor_metadata"], ADVISOR_METADATA)

        service.calls.clear()
        wrapper = {
            "expected_revision": 3,
            "decision": {**decision, "action": "request_patch_handoff"},
            "patch_proposal": {"schema_version": "redrhex.autopilot.patch-proposal.v1"},
            "advisor_metadata": ADVISOR_METADATA,
        }
        patch_handler = _handler(
            "POST",
            "/api/autopilot/campaigns/campaign-1/decisions",
            service,
            wrapper,
            _write_headers(3, "request-key-patch"),
        )
        patch_handler._do_POST()
        self.assertEqual(service.calls[0][0], "submit_patch_proposal")
        self.assertEqual(service.calls[0][4], ADVISOR_METADATA)

        invalid = _handler(
            "POST",
            "/api/autopilot/campaigns/campaign-1/decisions",
            service,
            {
                "expected_revision": 3,
                "decision": decision,
                "advisor_metadata": {"declared_model": "unversioned"},
            },
            _write_headers(3, "request-key-invalid-metadata"),
        )
        invalid._do_POST()
        self.assertEqual(invalid.json_responses[0][1], 400)
        self.assertEqual(
            invalid.json_responses[0][0]["code"],
            "invalid_advisor_metadata",
        )

    def test_all_writes_require_matching_idempotency_and_revision_headers(self):
        service = FakeAutopilotService()
        cases = [
            ([], "invalid_idempotency_key"),
            ([("Idempotency-Key", "request-key-0001")], "invalid_if_match"),
            (_write_headers(4), "revision_header_mismatch"),
        ]
        for headers, expected_code in cases:
            with self.subTest(expected_code=expected_code):
                handler = _handler(
                    "POST",
                    "/api/autopilot/campaigns/campaign-1/pause",
                    service,
                    {"expected_revision": 3},
                    headers,
                )
                handler._do_POST()
                payload, status = handler.json_responses[0]
                self.assertEqual(status, 400)
                self.assertEqual(payload["schema_version"], "redrhex.autopilot.error.v1")
                self.assertEqual(payload["code"], expected_code)
        self.assertEqual(service.calls, [])

    def test_draft_wrapper_rejects_unknown_or_unsupported_schema(self):
        service = FakeAutopilotService()
        payloads = [
            {"expected_revision": 0, "schema_version": "redrhex.autopilot.goal.v2"},
            {
                "expected_revision": 0,
                "schema_version": "redrhex.autopilot.goal.v1",
                "hydra_args": ["task.env.commands=unsafe"],
            },
        ]
        for index, payload in enumerate(payloads):
            with self.subTest(payload=payload):
                handler = _handler(
                    "POST",
                    "/api/autopilot/campaigns",
                    service,
                    payload,
                    _write_headers(0, f"request-schema-{index}"),
                )
                handler._do_POST()
                self.assertEqual(handler.json_responses[0][1], 400)
        self.assertEqual(service.calls, [])

    def test_missing_service_and_store_conflicts_fail_closed_with_typed_errors(self):
        missing = _handler(
            "POST",
            "/api/autopilot/campaigns",
            None,
            {"expected_revision": 0},
            _write_headers(0),
        )
        missing._do_POST()
        self.assertEqual(missing.json_responses[0][1], 503)
        self.assertEqual(missing.json_responses[0][0]["code"], "autopilot_unavailable")

        class ConflictService(FakeAutopilotService):
            def pause_campaign(self, *_args, **_kwargs):
                raise AutopilotConflictError("stale revision", current_revision=7)

        conflict = _handler(
            "POST",
            "/api/autopilot/campaigns/campaign-1/pause",
            ConflictService(),
            {"expected_revision": 3},
            _write_headers(3),
        )
        conflict._do_POST()
        payload, status = conflict.json_responses[0]
        self.assertEqual(status, 409)
        self.assertEqual(payload["code"], "campaign_conflict")
        self.assertEqual(payload["details"], {"current_revision": 7})

    def test_optional_bearer_token_protects_reads_and_writes(self):
        service = FakeAutopilotService()
        with patch.dict(os.environ, {"REDRHEX_AUTOPILOT_BEARER_TOKEN": "secret-token-value"}):
            denied = _handler("GET", "/api/autopilot/campaigns", service)
            denied._do_GET()
            self.assertEqual(denied.json_responses[0][1], 401)
            self.assertEqual(denied.json_responses[0][0]["code"], "unauthorized")

            allowed = _handler(
                "GET",
                "/api/autopilot/campaigns",
                service,
                headers=[("Authorization", "Bearer secret-token-value")],
            )
            allowed._do_GET()
            self.assertEqual(allowed.json_responses[0][1], 200)

    def test_not_found_store_error_is_typed(self):
        class MissingCampaignService(FakeAutopilotService):
            def get_campaign(self, _campaign_id):
                raise CampaignNotFoundError("campaign missing")

        handler = _handler(
            "GET",
            "/api/autopilot/campaigns/campaign-missing",
            MissingCampaignService(),
        )
        handler._do_GET()
        payload, status = handler.json_responses[0]
        self.assertEqual(status, 404)
        self.assertEqual(payload["code"], "campaign_not_found")


class AutopilotStartupOrderingTests(unittest.TestCase):
    def test_process_recovery_finishes_before_autopilot_worker_is_constructed(self):
        events = []

        class Dummy:
            def __init__(self, *_args, **_kwargs):
                pass

        class FakeProcesses(Dummy):
            def __init__(self, *_args, **_kwargs):
                events.append("process_registry")

            def reconcile_stale_history(self):
                events.append("reconcile")

            def start_next_queued_training(self):
                events.append("start_next")

        class FakeAutopilot(Dummy):
            def __init__(self, *_args, **_kwargs):
                events.append("autopilot")

        class FakeRemoteWorker(Dummy):
            def autostart_if_enabled(self):
                events.append("remote_autostart")

        paths = SimpleNamespace(
            physics_preset_file="physics.json",
            remote_state_file="remote.json",
        )
        with patch.multiple(
            server_module,
            HistoryStore=Dummy,
            GoogleDriveExporter=Dummy,
            ProcessRegistry=FakeProcesses,
            PresetStore=Dummy,
            TerrainPresetStore=Dummy,
            PhysicsPresetStore=Dummy,
            ActivityStore=Dummy,
            RemoteStateStore=Dummy,
            RemoteWorkerManager=FakeRemoteWorker,
            AutopilotService=FakeAutopilot,
        ):
            state = PanelState(paths)

        self.assertIsInstance(state.autopilot, FakeAutopilot)
        self.assertLess(events.index("reconcile"), events.index("autopilot"))
        self.assertLess(events.index("start_next"), events.index("autopilot"))


if __name__ == "__main__":
    unittest.main()
