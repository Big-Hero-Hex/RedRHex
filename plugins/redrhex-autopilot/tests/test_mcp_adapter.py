from __future__ import annotations

import importlib.util
import http.client
import json
import math
import os
import subprocess
import sys
import threading
import unittest
from unittest.mock import patch
from pathlib import Path

from tools.training_panel.training_panel.autopilot import AgentDecisionV1, GoalSpecV1


MODULE_PATH = Path(__file__).parents[1] / "scripts" / "redrhex_autopilot_mcp.py"
SPEC = importlib.util.spec_from_file_location("redrhex_autopilot_mcp", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
mcp = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = mcp
SPEC.loader.exec_module(mcp)


class RecordingClient:
    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def get(self, path, *, query=None):
        self.calls.append(("GET", path, query))
        return {"path": path, "query": query}

    def post(self, path, payload, *, idempotency_key, expected_revision):
        self.calls.append(
            ("POST", path, payload, idempotency_key, expected_revision)
        )
        return {"path": path, "revision": expected_revision}


class FakeResponse:
    def __init__(self, payload, status=200) -> None:
        self.status = status
        self._body = json.dumps(payload).encode("utf-8")

    def read(self, _limit=-1):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False


class RecordingOpener:
    def __init__(self, payload=None) -> None:
        self.payload = payload if payload is not None else {"ok": True}
        self.requests = []

    def open(self, request, timeout):
        self.requests.append((request, timeout))
        return FakeResponse(self.payload)


def valid_goal():
    return {
        "description": "Walk forward",
        "task": "Template-Redrhex-ForwardFast-Direct-v0",
        "stage": 1,
        "evaluation_profile": "stage1",
        "gait": "walk",
        "directions": ["forward"],
        "command_envelope": {
            "vx": [[0.22, 0.32]],
            "vy": [[0.0, 0.0]],
            "wz": [[0.0, 0.0]],
        },
        "skill_gates": {
            "min_command_pass_ratio": 0.70,
            "min_skill_pass_ratio": 0.60,
            "max_fall_rate": 0.20,
            "min_tracking_quality": 0.0,
            "min_stability_quality": 0.0,
            "min_direction_sign_ratio": 0.70,
            "max_linear_leak": 0.18,
            "max_yaw_leak": 0.35,
            "max_energy_per_distance": 500.0,
        },
        "initialization_mode": "fresh",
        "baseline_run_id": None,
        "baseline_checkpoint_iteration": None,
        "checkpoint_sha256": None,
        "training_seeds": [42, 43, 44],
        "per_trial_iteration_cap": 1000,
        "tunable_reward_keys": ["v2_reward_scales.velocity_tracking"],
        "budget": {
            "max_training_trials": 24,
            "max_gpu_hours": 72,
        },
    }


class LoopbackBoundaryTests(unittest.TestCase):
    def test_accepts_only_loopback_origins(self):
        self.assertEqual(
            mcp.validate_loopback_url("http://127.0.0.1:8080/"),
            "http://127.0.0.1:8080",
        )
        self.assertEqual(
            mcp.validate_loopback_url("https://[::1]:9443"),
            "https://[::1]:9443",
        )
        self.assertEqual(
            mcp.validate_loopback_url("http://localhost:8768"),
            "http://localhost:8768",
        )
        for unsafe in (
            "http://example.com:8080",
            "http://127.0.0.1:8080/api",
            "http://user:secret@127.0.0.1:8080",
            "file:///tmp/panel.sock",
            "http://127.0.0.1:8080?next=evil",
        ):
            with self.subTest(unsafe=unsafe):
                with self.assertRaises(mcp.AdapterConfigurationError):
                    mcp.validate_loopback_url(unsafe)

    def test_panel_client_adds_auth_idempotency_and_revision_headers(self):
        opener = RecordingOpener({"campaign_id": "campaign-1"})
        client = mcp.PanelClient(
            "http://127.0.0.1:8080",
            "top-secret",
            timeout=5,
            opener=opener,
        )
        result = client.post(
            "/api/autopilot/campaigns/campaign-1/pause",
            {"expected_revision": 7, "reason": "review"},
            idempotency_key="request-1234",
            expected_revision=7,
        )
        self.assertEqual(result["campaign_id"], "campaign-1")
        request, timeout = opener.requests[0]
        self.assertEqual(timeout, 5)
        self.assertEqual(request.full_url, "http://127.0.0.1:8080/api/autopilot/campaigns/campaign-1/pause")
        self.assertEqual(request.get_header("Authorization"), "Bearer top-secret")
        self.assertEqual(request.get_header("Idempotency-key"), "request-1234")
        self.assertEqual(request.get_header("If-match"), '"7"')
        self.assertEqual(request.get_header("X-expected-revision"), "7")

    def test_panel_client_rejects_non_autopilot_path(self):
        client = mcp.PanelClient(
            "http://127.0.0.1:8080", timeout=5, opener=RecordingOpener()
        )
        with self.assertRaises(mcp.AdapterConfigurationError):
            client.get("/api/runs")

    def test_connector_token_is_never_forwarded_to_the_panel(self):
        opener = RecordingOpener()
        with patch.dict(
            os.environ,
            {"REDRHEX_AUTOPILOT_MCP_TOKEN": "connector-only-token-1234"},
            clear=True,
        ):
            client = mcp.PanelClient(opener=opener)
            client.get("/api/autopilot/campaigns")
        request, _timeout = opener.requests[0]
        self.assertIsNone(request.get_header("Authorization"))


class ToolContractTests(unittest.TestCase):
    def setUp(self):
        self.client = RecordingClient()
        self.gateway = mcp.ToolGateway(self.client)

    def test_exposes_exactly_the_eleven_bounded_tools(self):
        expected = {
            "redrhex_list_campaigns",
            "redrhex_get_campaign",
            "redrhex_get_decision_context",
            "redrhex_compare_trials",
            "redrhex_get_artifact_link",
            "redrhex_advisor_heartbeat",
            "redrhex_create_goal_draft",
            "redrhex_propose_candidate",
            "redrhex_pause_campaign",
            "redrhex_request_stop_after_current",
            "redrhex_submit_patch_proposal",
        }
        definitions = {tool["name"]: tool for tool in mcp.TOOL_DEFINITIONS}
        self.assertEqual(set(definitions), expected)
        self.assertNotIn("redrhex_arm_campaign", definitions)
        self.assertNotIn("redrhex_resume_campaign", definitions)
        for name, definition in definitions.items():
            self.assertFalse(definition["annotations"]["openWorldHint"], name)
            self.assertFalse(definition["inputSchema"]["additionalProperties"], name)
        self.assertTrue(
            definitions["redrhex_request_stop_after_current"]["annotations"][
                "destructiveHint"
            ]
        )

    def test_heartbeat_records_fixed_advisor_metadata_without_a_decision(self):
        self.gateway.call(
            "redrhex_advisor_heartbeat",
            {
                "campaign_id": "campaign-1",
                "idempotency_key": "heartbeat-request-1",
                "expected_revision": 5,
            },
        )
        method, path, payload, key, revision = self.client.calls[0]
        self.assertEqual(method, "POST")
        self.assertEqual(path, "/api/autopilot/campaigns/campaign-1/heartbeat")
        self.assertEqual((key, revision), ("heartbeat-request-1", 5))
        self.assertEqual(payload["expected_revision"], 5)
        self.assertEqual(payload["advisor_metadata"]["skill_version"], "redrhex-autopilot/0.1.0")
        self.assertNotIn("decision", payload)

    def test_read_tools_map_only_to_fixed_routes(self):
        self.gateway.call("redrhex_list_campaigns", {"state": "awaiting_advisor", "limit": 5})
        self.gateway.call("redrhex_get_campaign", {"campaign_id": "campaign-1"})
        self.gateway.call("redrhex_get_decision_context", {"campaign_id": "campaign-1"})
        self.gateway.call(
            "redrhex_compare_trials",
            {"campaign_id": "campaign-1", "trial_ids": ["trial-1", "trial-2"]},
        )
        self.gateway.call(
            "redrhex_get_artifact_link",
            {"campaign_id": "campaign-1", "artifact_id": "artifact-1"},
        )
        self.assertEqual(
            self.client.calls,
            [
                ("GET", "/api/autopilot/campaigns", {"state": "awaiting_advisor", "limit": 5}),
                ("GET", "/api/autopilot/campaigns/campaign-1", None),
                ("GET", "/api/autopilot/campaigns/campaign-1/decision-context", None),
                (
                    "GET",
                    "/api/autopilot/campaigns/campaign-1/compare",
                    {"trial_ids": "trial-1,trial-2"},
                ),
                (
                    "GET",
                    "/api/autopilot/campaigns/campaign-1/artifacts/artifact-1",
                    None,
                ),
            ],
        )

    def test_campaign_reads_strip_host_local_paths(self):
        class CampaignClient(RecordingClient):
            def get(self, path, *, query=None):
                self.calls.append(("GET", path, query))
                return {
                    "campaign": {
                        "id": "campaign-1",
                        "candidate_lineage": [
                            {
                                "id": "trial-1",
                                "metadata": {
                                    "output_checkpoint_path": "/secret/model_1.pt",
                                    "evaluation_input_sha256": "a" * 64,
                                },
                            }
                        ],
                    }
                }

        gateway = mcp.ToolGateway(CampaignClient())
        result = gateway.call(
            "redrhex_get_campaign", {"campaign_id": "campaign-1"}
        )

        metadata = result["campaign"]["candidate_lineage"][0]["metadata"]
        self.assertNotIn("output_checkpoint_path", metadata)
        self.assertEqual(metadata["evaluation_input_sha256"], "a" * 64)

    def test_all_mutation_responses_strip_host_local_paths(self):
        class CampaignClient(RecordingClient):
            def post(self, path, payload, *, idempotency_key, expected_revision):
                self.calls.append(
                    ("POST", path, payload, idempotency_key, expected_revision)
                )
                return {
                    "campaign": {
                        "id": "campaign-1",
                        "candidate_lineage": [
                            {
                                "id": "trial-1",
                                "metadata": {
                                    "output_checkpoint_path": "/secret/model_1.pt",
                                    "checkpoint_path": "/secret/source.pt",
                                    "command_profile_file": "/secret/commands.json",
                                    "reward_profile_file": "/secret/reward.json",
                                    "terrain_profile_file": "/secret/terrain.json",
                                    "physics_profile_file": "/secret/physics.json",
                                    "log_dir": "/secret/run",
                                    "process_log": "/secret/process.log",
                                    "evaluation_input_sha256": "a" * 64,
                                },
                            }
                        ],
                    }
                }

        source_path = "source/RedRhex/RedRhex/tasks/direct/redrhex/redrhex_env.py"
        mutations = {
            "redrhex_advisor_heartbeat": {
                "campaign_id": "campaign-1",
                "idempotency_key": "heartbeat-redaction-1",
                "expected_revision": 5,
            },
            "redrhex_create_goal_draft": {
                "goal": valid_goal(),
                "idempotency_key": "draft-redaction-1",
                "expected_revision": 0,
            },
            "redrhex_propose_candidate": {
                "campaign_id": "campaign-1",
                "idempotency_key": "candidate-redaction-1",
                "expected_revision": 6,
                "evidence_ids": ["eval-1"],
                "hypothesis": "Increase tracking weight slightly.",
                "reward_key": "v2_reward_scales.velocity_tracking",
                "proposed_value": 1.1,
                "expected_metric_effect": "Improve command tracking.",
                "rationale": "The current leader misses its tracking gate.",
            },
            "redrhex_pause_campaign": {
                "campaign_id": "campaign-1",
                "idempotency_key": "pause-redaction-1",
                "expected_revision": 7,
                "reason": "Review the current evidence.",
            },
            "redrhex_request_stop_after_current": {
                "campaign_id": "campaign-1",
                "idempotency_key": "stop-redaction-1",
                "expected_revision": 8,
                "reason": "End after the current process.",
            },
            "redrhex_submit_patch_proposal": {
                "campaign_id": "campaign-1",
                "idempotency_key": "patch-redaction-1",
                "expected_revision": 9,
                "target_symbols": ["RedrhexEnv._compute_simplified_rewards"],
                "base_blob_hashes": {source_path: "a" * 64},
                "unified_diff": (
                    f"diff --git a/{source_path} b/{source_path}\n"
                    f"--- a/{source_path}\n"
                    f"+++ b/{source_path}\n"
                    "@@ -1 +1 @@\n-old\n+new\n"
                ),
                "rationale": "Bounded runtime tuning is exhausted.",
                "test_plan": "Run the focused reward tests.",
                "rollback_notes": "Discard the proposal artifact.",
            },
        }

        def assert_redacted(value):
            if isinstance(value, dict):
                self.assertFalse(set(value) & mcp._CAMPAIGN_PATH_FIELDS)
                for item in value.values():
                    assert_redacted(item)
            elif isinstance(value, list):
                for item in value:
                    assert_redacted(item)

        gateway = mcp.ToolGateway(CampaignClient())
        for tool_name, arguments in mutations.items():
            with self.subTest(tool_name=tool_name):
                result = gateway.call(tool_name, arguments)
                assert_redacted(result)
                metadata = result["campaign"]["candidate_lineage"][0]["metadata"]
                self.assertEqual(metadata["evaluation_input_sha256"], "a" * 64)

    def test_create_goal_draft_is_unarmed_and_capped(self):
        self.gateway.call(
            "redrhex_create_goal_draft",
            {
                "goal": valid_goal(),
                "idempotency_key": "draft-request-1",
                "expected_revision": 0,
            },
        )
        method, path, payload, key, revision = self.client.calls[0]
        self.assertEqual((method, path), ("POST", "/api/autopilot/campaigns"))
        self.assertEqual((key, revision), ("draft-request-1", 0))
        self.assertEqual(payload["schema_version"], "redrhex.autopilot.goal.v1")
        self.assertNotIn("goal", payload)
        self.assertNotIn("arm", payload)
        self.assertEqual(payload["budget"]["max_training_trials"], 24)
        core_goal = {
            key: value
            for key, value in payload.items()
            if key not in {"expected_revision", "tunable_reward_keys", "reward_bounds"}
        }
        core_goal.update(
            {
                "physics_profile_sha256": "a" * 64,
                "spring_profile_sha256": "b" * 64,
                "code_sha256": "c" * 64,
                "config_sha256": "d" * 64,
                "command_profile_sha256": "e" * 64,
            }
        )
        self.assertEqual(GoalSpecV1.from_dict(core_goal).schema_version, payload["schema_version"])

        narrowed = valid_goal()
        narrowed["reward_bounds"] = {"v2_reward_scales.velocity_tracking": [0.9, 1.1]}
        self.gateway.call(
            "redrhex_create_goal_draft",
            {
                "goal": narrowed,
                "idempotency_key": "draft-request-bounds",
                "expected_revision": 0,
            },
        )
        self.assertEqual(
            self.client.calls[-1][2]["reward_bounds"],
            {"v2_reward_scales.velocity_tracking": [0.9, 1.1]},
        )

        over_budget = valid_goal()
        over_budget["budget"]["max_training_trials"] = 25
        with self.assertRaises(mcp.ToolInputError):
            self.gateway.call(
                "redrhex_create_goal_draft",
                {
                    "goal": over_budget,
                    "idempotency_key": "draft-request-2",
                    "expected_revision": 0,
                },
            )

    def test_candidate_maps_to_decisions_with_one_finite_value(self):
        arguments = {
            "campaign_id": "campaign-1",
            "idempotency_key": "candidate-request-1",
            "expected_revision": 6,
            "evidence_ids": ["eval-1", "eval-2"],
            "hypothesis": "A small tracking increase may reduce forward error.",
            "reward_key": "v2_reward_scales.velocity_tracking",
            "proposed_value": 1.1,
            "expected_metric_effect": "Lower tracking error without extra falls.",
            "rationale": "Both matched evaluations miss the tracking gate.",
        }
        self.gateway.call("redrhex_propose_candidate", arguments)
        method, path, payload, key, revision = self.client.calls[0]
        self.assertEqual(method, "POST")
        self.assertEqual(path, "/api/autopilot/campaigns/campaign-1/decisions")
        self.assertEqual((key, revision), ("candidate-request-1", 6))
        decision = payload["decision"]
        self.assertEqual(decision["action"], "propose_candidate")
        self.assertEqual(decision["schema_version"], "redrhex.autopilot.decision.v1")
        self.assertEqual(decision["campaign_revision"], 6)
        self.assertEqual(payload["expected_revision"], 6)
        self.assertEqual(decision["reward_key"], "v2_reward_scales.velocity_tracking")
        self.assertNotIn("hydra_args", payload)
        self.assertEqual(AgentDecisionV1.from_dict(decision).campaign_revision, 6)
        self.assertEqual(
            payload["advisor_metadata"],
            {
                "schema_version": "redrhex.autopilot.advisor-metadata.v1",
                "skill_version": "redrhex-autopilot/0.1.0",
                "prompt_version": "scheduled-advisor.v1",
                "declared_model": "gpt-5.6-terra",
                "reasoning_effort": "medium",
            },
        )

        for bad_value in (math.inf, math.nan, True):
            invalid = dict(arguments, proposed_value=bad_value)
            with self.subTest(bad_value=bad_value):
                with self.assertRaises(mcp.ToolInputError):
                    self.gateway.call("redrhex_propose_candidate", invalid)

    def test_pause_and_stop_routes_include_expected_revision(self):
        common = {
            "campaign_id": "campaign-1",
            "idempotency_key": "lifecycle-request-1",
            "expected_revision": 9,
            "reason": "operator requested review",
        }
        self.gateway.call("redrhex_pause_campaign", common)
        common["idempotency_key"] = "lifecycle-request-2"
        self.gateway.call("redrhex_request_stop_after_current", common)
        pause = self.client.calls[0]
        stop = self.client.calls[1]
        self.assertEqual(pause[1], "/api/autopilot/campaigns/campaign-1/pause")
        self.assertEqual(pause[2]["expected_revision"], 9)
        self.assertEqual(pause[2]["advisor_metadata"]["declared_model"], "gpt-5.6-terra")
        self.assertEqual(stop[1], "/api/autopilot/campaigns/campaign-1/stop")
        self.assertEqual(stop[2]["mode"], "after_current")

    def test_patch_proposal_is_stored_as_a_decision_not_applied(self):
        source_path = "source/RedRhex/RedRhex/tasks/direct/redrhex/redrhex_env.py"
        self.gateway.call(
            "redrhex_submit_patch_proposal",
            {
                "campaign_id": "campaign-1",
                "idempotency_key": "patch-request-1",
                "expected_revision": 12,
                "target_symbols": ["reward_term"],
                "base_blob_hashes": {source_path: "a" * 64},
                "unified_diff": (
                    f"diff --git a/{source_path} b/{source_path}\n"
                    f"--- a/{source_path}\n"
                    f"+++ b/{source_path}\n"
                    "@@ -1 +1 @@\n-old\n+new\n"
                ),
                "rationale": "Bounded weights were exhausted.",
                "test_plan": "Run focused reward tests.",
                "rollback_notes": "Discard the proposal artifact.",
            },
        )
        _method, path, payload, _key, _revision = self.client.calls[0]
        self.assertEqual(path, "/api/autopilot/campaigns/campaign-1/decisions")
        self.assertEqual(payload["decision"]["action"], "request_patch_handoff")
        self.assertEqual(
            payload["patch_proposal"]["schema_version"],
            "redrhex.autopilot.patch-proposal.v1",
        )
        self.assertEqual(
            AgentDecisionV1.from_dict(payload["decision"]).action,
            "request_patch_handoff",
        )
        self.assertEqual(payload["advisor_metadata"]["prompt_version"], "scheduled-advisor.v1")
        self.assertNotIn("apply", payload)

    def test_declared_advisor_model_comes_only_from_process_environment(self):
        arguments = {
            "campaign_id": "campaign-1",
            "idempotency_key": "pause-model-request",
            "expected_revision": 4,
            "reason": "safety review",
        }
        with patch.dict(
            os.environ,
            {"REDRHEX_AUTOPILOT_ADVISOR_MODEL": "deterministic-fake-advisor"},
        ):
            self.gateway.call("redrhex_pause_campaign", arguments)
        self.assertEqual(
            self.client.calls[0][2]["advisor_metadata"]["declared_model"],
            "deterministic-fake-advisor",
        )

    def test_unknown_arguments_and_tools_fail_closed(self):
        with self.assertRaises(mcp.ToolInputError):
            self.gateway.call("redrhex_get_campaign", {"campaign_id": "campaign-1", "url": "http://evil"})
        with self.assertRaises(mcp.ToolInputError):
            self.gateway.call("redrhex_run_shell", {"command": "echo nope"})


class MCPProtocolTests(unittest.TestCase):
    def setUp(self):
        self.client = RecordingClient()
        self.server = mcp.MCPServer(mcp.ToolGateway(self.client))

    def test_initialize_and_list_tools(self):
        initialized = self.server.handle(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {"protocolVersion": "2025-06-18"},
            }
        )
        self.assertEqual(initialized["result"]["protocolVersion"], "2025-06-18")
        listed = self.server.handle(
            {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}}
        )
        self.assertEqual(len(listed["result"]["tools"]), 11)

    def test_tool_call_returns_structured_content(self):
        response = self.server.handle(
            {
                "jsonrpc": "2.0",
                "id": "call-1",
                "method": "tools/call",
                "params": {
                    "name": "redrhex_get_campaign",
                    "arguments": {"campaign_id": "campaign-1"},
                },
            }
        )
        result = response["result"]
        self.assertFalse(result["isError"])
        self.assertTrue(result["structuredContent"]["ok"])
        self.assertEqual(
            result["structuredContent"]["data"]["path"],
            "/api/autopilot/campaigns/campaign-1",
        )

    def test_invalid_tool_input_is_a_safe_tool_error(self):
        response = self.server.handle(
            {
                "jsonrpc": "2.0",
                "id": "call-2",
                "method": "tools/call",
                "params": {
                    "name": "redrhex_get_campaign",
                    "arguments": {"campaign_id": "../../api/runs"},
                },
            }
        )
        result = response["result"]
        self.assertTrue(result["isError"])
        self.assertEqual(
            result["structuredContent"]["error"]["code"], "invalid_tool_input"
        )
        self.assertEqual(self.client.calls, [])

    def test_notifications_produce_no_response(self):
        self.assertIsNone(
            self.server.handle(
                {
                    "jsonrpc": "2.0",
                    "method": "notifications/initialized",
                    "params": {},
                }
            )
        )

    def test_stdio_entrypoint_emits_only_json_rpc(self):
        request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {"protocolVersion": "2025-06-18"},
        }
        completed = subprocess.run(
            [sys.executable, str(MODULE_PATH)],
            input=(json.dumps(request) + "\n").encode("utf-8"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=5,
            check=True,
        )
        response = json.loads(completed.stdout.decode("utf-8"))
        self.assertEqual(response["id"], 1)
        self.assertEqual(response["result"]["serverInfo"]["name"], "redrhex-autopilot")
        self.assertEqual(completed.stderr, b"")


class HTTPTransportTests(unittest.TestCase):
    TOKEN = "connector-token-1234567890"

    def setUp(self):
        self.client = RecordingClient()
        self.server = mcp.create_http_server(
            "127.0.0.1",
            0,
            gateway=mcp.ToolGateway(self.client),
            bearer_token=self.TOKEN,
        )
        self.port = self.server.server_address[1]
        self.origin = f"http://127.0.0.1:{self.port}"
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)

    def request(self, method, path="/mcp", payload=None, *, headers=None):
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        body = None if payload is None else json.dumps(payload).encode("utf-8")
        request_headers = {
            "Authorization": f"Bearer {self.TOKEN}",
            "Origin": self.origin,
        }
        if method == "POST":
            request_headers.update(
                {
                    "Content-Type": "application/json",
                    "Accept": "application/json, text/event-stream",
                    "MCP-Protocol-Version": "2025-06-18",
                }
            )
        if headers:
            request_headers.update(headers)
        connection.request(method, path, body=body, headers=request_headers)
        response = connection.getresponse()
        raw = response.read()
        result = (response.status, dict(response.getheaders()), raw)
        connection.close()
        return result

    def test_streamable_http_initialize_and_tool_call(self):
        initialize = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {"protocolVersion": "2025-06-18"},
        }
        status, headers, body = self.request("POST", payload=initialize)
        self.assertEqual(status, 200)
        self.assertIn("application/json", headers["Content-Type"])
        self.assertEqual(json.loads(body)["result"]["protocolVersion"], "2025-06-18")

        tool_call = {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {
                "name": "redrhex_get_campaign",
                "arguments": {"campaign_id": "campaign-1"},
            },
        }
        status, _headers, body = self.request("POST", payload=tool_call)
        self.assertEqual(status, 200)
        self.assertTrue(json.loads(body)["result"]["structuredContent"]["ok"])
        self.assertEqual(
            self.client.calls,
            [("GET", "/api/autopilot/campaigns/campaign-1", None)],
        )

    def test_notification_returns_accepted_without_body(self):
        notification = {
            "jsonrpc": "2.0",
            "method": "notifications/initialized",
            "params": {},
        }
        status, _headers, body = self.request("POST", payload=notification)
        self.assertEqual(status, 202)
        self.assertEqual(body, b"")

    def test_get_explicitly_declines_sse_stream(self):
        status, headers, body = self.request("GET")
        self.assertEqual(status, 405)
        self.assertEqual(headers["Allow"], "POST")
        self.assertEqual(body, b"")

    def test_rejects_bad_origin_and_bad_connector_token(self):
        initialize = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {"protocolVersion": "2025-06-18"},
        }
        status, _headers, _body = self.request(
            "POST", payload=initialize, headers={"Origin": "https://evil.example"}
        )
        self.assertEqual(status, 403)
        status, headers, _body = self.request(
            "POST", payload=initialize, headers={"Authorization": "Bearer wrong-token-value"}
        )
        self.assertEqual(status, 401)
        self.assertIn("Bearer", headers["WWW-Authenticate"])
        self.assertEqual(self.client.calls, [])

    def test_rejects_wrong_path_media_types_and_protocol(self):
        initialize = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {"protocolVersion": "2025-06-18"},
        }
        status, _headers, _body = self.request("POST", "/other", initialize)
        self.assertEqual(status, 404)
        status, _headers, _body = self.request(
            "POST", payload=initialize, headers={"Accept": "application/json"}
        )
        self.assertEqual(status, 406)
        status, _headers, _body = self.request(
            "POST", payload=initialize, headers={"MCP-Protocol-Version": "2099-01-01"}
        )
        self.assertEqual(status, 400)

    def test_rejects_oversized_body_before_dispatch(self):
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        connection.putrequest("POST", "/mcp")
        connection.putheader("Authorization", f"Bearer {self.TOKEN}")
        connection.putheader("Origin", self.origin)
        connection.putheader("Content-Type", "application/json")
        connection.putheader("Accept", "application/json, text/event-stream")
        connection.putheader("Content-Length", str(mcp.MAX_INPUT_BYTES + 1))
        connection.endheaders()
        response = connection.getresponse()
        response.read()
        self.assertEqual(response.status, 413)
        connection.close()

    def test_http_server_refuses_non_loopback_bind(self):
        with self.assertRaises(mcp.AdapterConfigurationError):
            mcp.create_http_server("0.0.0.0", 8787, bearer_token=self.TOKEN)


if __name__ == "__main__":
    unittest.main()
