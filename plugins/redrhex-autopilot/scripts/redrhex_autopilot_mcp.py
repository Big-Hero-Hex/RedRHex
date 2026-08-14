#!/usr/bin/env python3
"""Narrow stdio MCP adapter for the loopback RedRHex Autopilot API.

This module intentionally uses only the Python standard library.  It is not a
generic HTTP proxy: every path, method, query, and request body is constructed
by one of the eleven allowlisted tools below.
"""

from __future__ import annotations

import argparse
import ipaddress
import json
import math
import os
import re
import secrets
import socket
import sys
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Mapping
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode, urlsplit, urlunsplit
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener


SERVER_NAME = "redrhex-autopilot"
SERVER_VERSION = "0.1.0"
LATEST_PROTOCOL_VERSION = "2025-06-18"
SUPPORTED_PROTOCOL_VERSIONS = {
    "2024-11-05",
    "2025-03-26",
    LATEST_PROTOCOL_VERSION,
}
DEFAULT_PANEL_URL = "http://127.0.0.1:8080"
DEFAULT_MCP_HOST = "127.0.0.1"
DEFAULT_MCP_PORT = 8787
MCP_HTTP_PATH = "/mcp"
MAX_INPUT_BYTES = 1_048_576
MAX_RESPONSE_BYTES = 2_097_152
DEFAULT_TIMEOUT_SECONDS = 30.0
GOAL_SCHEMA_VERSION = "redrhex.autopilot.goal.v1"
DECISION_SCHEMA_VERSION = "redrhex.autopilot.decision.v1"
PATCH_SCHEMA_VERSION = "redrhex.autopilot.patch-proposal.v1"
ADVISOR_METADATA_SCHEMA_VERSION = "redrhex.autopilot.advisor-metadata.v1"
SKILL_VERSION = "redrhex-autopilot/0.1.0"
PROMPT_VERSION = "scheduled-advisor.v1"
DEFAULT_DECLARED_MODEL = "gpt-5.6-terra"
DECLARED_REASONING_EFFORT = "medium"
DIRECT_TASK = "Template-Redrhex-Direct-v0"
FORWARD_FAST_TASK = "Template-Redrhex-ForwardFast-Direct-v0"

_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_IDEMPOTENCY_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{7,127}$")
_SHA256_RE = re.compile(r"^[a-f0-9]{64}$")
_REWARD_KEY_RE = re.compile(r"^v2_reward_scales\.[A-Za-z0-9_]+$")
_PATCH_SYMBOL_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_.]{0,127}$")
_MODEL_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}$")
PATCH_SOURCE_ALLOWLIST = {
    "source/RedRhex/RedRhex/tasks/direct/redrhex/redrhex_env.py",
    "source/RedRhex/RedRhex/tasks/direct/redrhex/redrhex_env_cfg.py",
}


class AdapterConfigurationError(ValueError):
    """Raised for unsafe or invalid adapter process configuration."""


class ToolInputError(ValueError):
    """Raised when an MCP tool call does not satisfy the local contract."""


@dataclass(frozen=True)
class PanelAPIError(RuntimeError):
    status: int
    code: str
    message: str
    details: Any = None

    def as_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "status": self.status,
            "code": self.code,
            "message": self.message,
        }
        if self.details is not None:
            result["details"] = self.details
        return result


class _RejectRedirects(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: ANN001
        return None


def validate_loopback_url(raw_url: str) -> str:
    """Return a canonical panel origin after enforcing a loopback-only URL."""

    if not isinstance(raw_url, str) or not raw_url.strip():
        raise AdapterConfigurationError("panel URL must be a non-empty string")
    parsed = urlsplit(raw_url.strip())
    if parsed.scheme not in {"http", "https"}:
        raise AdapterConfigurationError("panel URL scheme must be http or https")
    if parsed.username is not None or parsed.password is not None:
        raise AdapterConfigurationError("panel URL must not contain credentials")
    if parsed.query or parsed.fragment:
        raise AdapterConfigurationError("panel URL must not contain a query or fragment")
    if parsed.path not in {"", "/"}:
        raise AdapterConfigurationError("panel URL must be an origin without a path")
    hostname = parsed.hostname
    if hostname is None:
        raise AdapterConfigurationError("panel URL must include a host")
    is_loopback = hostname.lower() == "localhost"
    if not is_loopback:
        try:
            is_loopback = ipaddress.ip_address(hostname).is_loopback
        except ValueError:
            is_loopback = False
    if not is_loopback:
        raise AdapterConfigurationError("panel URL host must be loopback")
    try:
        port = parsed.port
    except ValueError as exc:
        raise AdapterConfigurationError("panel URL has an invalid port") from exc
    if port is not None and not 1 <= port <= 65535:
        raise AdapterConfigurationError("panel URL port is outside 1..65535")
    host = f"[{hostname}]" if ":" in hostname else hostname
    netloc = f"{host}:{port}" if port is not None else host
    return urlunsplit((parsed.scheme, netloc, "", "", ""))


def validate_loopback_host(raw_host: str) -> str:
    """Validate an HTTP bind host without performing DNS resolution."""

    if not isinstance(raw_host, str) or not raw_host.strip():
        raise AdapterConfigurationError("MCP bind host must be non-empty")
    host = raw_host.strip()
    if host.lower() == "localhost":
        return "localhost"
    try:
        address = ipaddress.ip_address(host)
    except ValueError as exc:
        raise AdapterConfigurationError(
            "MCP bind host must be localhost or a literal loopback address"
        ) from exc
    if not address.is_loopback:
        raise AdapterConfigurationError("MCP bind host must be loopback")
    return address.compressed


def canonical_origin(raw_origin: str) -> str:
    """Canonicalize one exact HTTP Origin allowlist entry."""

    if not isinstance(raw_origin, str) or not raw_origin.strip():
        raise AdapterConfigurationError("MCP allowed origins must be non-empty")
    parsed = urlsplit(raw_origin.strip())
    if parsed.scheme.lower() not in {"http", "https"}:
        raise AdapterConfigurationError("MCP allowed origins must use http or https")
    if parsed.username is not None or parsed.password is not None:
        raise AdapterConfigurationError("MCP allowed origins must not contain credentials")
    if parsed.path not in {"", "/"} or parsed.query or parsed.fragment:
        raise AdapterConfigurationError("MCP allowed origins must be exact origins without paths")
    hostname = parsed.hostname
    if hostname is None:
        raise AdapterConfigurationError("MCP allowed origin must include a host")
    try:
        port = parsed.port
    except ValueError as exc:
        raise AdapterConfigurationError("MCP allowed origin has an invalid port") from exc
    default_port = 80 if parsed.scheme.lower() == "http" else 443
    normalized_host = hostname.lower()
    if ":" in normalized_host:
        normalized_host = f"[{normalized_host}]"
    netloc = normalized_host if port in {None, default_port} else f"{normalized_host}:{port}"
    return f"{parsed.scheme.lower()}://{netloc}"


def allowed_origins_from_env(host: str, port: int) -> frozenset[str]:
    """Load an exact origin allowlist, with safe local defaults."""

    configured = os.environ.get("REDRHEX_AUTOPILOT_MCP_ALLOWED_ORIGINS", "").strip()
    if configured:
        entries = [entry.strip() for entry in configured.split(",") if entry.strip()]
        if not entries:
            raise AdapterConfigurationError("MCP allowed origin list is empty")
        return frozenset(canonical_origin(entry) for entry in entries)
    local_hosts = {host, "127.0.0.1", "localhost", "::1"}
    origins = set()
    for local_host in local_hosts:
        rendered = f"[{local_host}]" if ":" in local_host else local_host
        origins.add(canonical_origin(f"http://{rendered}:{port}"))
    return frozenset(origins)


def mcp_http_token_from_env() -> str:
    token = os.environ.get("REDRHEX_AUTOPILOT_MCP_TOKEN", "").strip()
    if token and (len(token) < 16 or len(token) > 4096 or any(char.isspace() for char in token)):
        raise AdapterConfigurationError(
            "REDRHEX_AUTOPILOT_MCP_TOKEN must be 16..4096 non-whitespace characters"
        )
    return token


def advisor_metadata() -> dict[str, str]:
    declared_model = os.environ.get(
        "REDRHEX_AUTOPILOT_ADVISOR_MODEL",
        DEFAULT_DECLARED_MODEL,
    ).strip()
    if _MODEL_NAME_RE.fullmatch(declared_model) is None:
        raise ToolInputError(
            "REDRHEX_AUTOPILOT_ADVISOR_MODEL must be a safe 1..128 character model name"
        )
    return {
        "schema_version": ADVISOR_METADATA_SCHEMA_VERSION,
        "skill_version": SKILL_VERSION,
        "prompt_version": PROMPT_VERSION,
        "declared_model": declared_model,
        "reasoning_effort": DECLARED_REASONING_EFFORT,
    }


def _timeout_from_env() -> float:
    raw = os.environ.get("REDRHEX_AUTOPILOT_TIMEOUT_SECONDS", "").strip()
    if not raw:
        return DEFAULT_TIMEOUT_SECONDS
    try:
        value = float(raw)
    except ValueError as exc:
        raise AdapterConfigurationError("timeout must be numeric") from exc
    if not math.isfinite(value) or not 1.0 <= value <= 120.0:
        raise AdapterConfigurationError("timeout must be finite and within 1..120 seconds")
    return value


class PanelClient:
    """Minimal client that can address only the Autopilot API namespace."""

    def __init__(
        self,
        panel_url: str | None = None,
        bearer_token: str | None = None,
        *,
        timeout: float | None = None,
        opener=None,
    ) -> None:
        self.base_url = validate_loopback_url(
            panel_url
            if panel_url is not None
            else os.environ.get("REDRHEX_AUTOPILOT_PANEL_URL", DEFAULT_PANEL_URL)
        )
        token = (
            bearer_token
            if bearer_token is not None
            else os.environ.get("REDRHEX_AUTOPILOT_BEARER_TOKEN", "")
        )
        self.bearer_token = token.strip()
        self.timeout = timeout if timeout is not None else _timeout_from_env()
        if not math.isfinite(self.timeout) or not 1.0 <= self.timeout <= 120.0:
            raise AdapterConfigurationError("timeout must be finite and within 1..120 seconds")
        self.opener = opener or build_opener(ProxyHandler({}), _RejectRedirects())

    def get(
        self,
        path: str,
        *,
        query: Mapping[str, str | int] | None = None,
    ) -> Any:
        return self._request("GET", path, query=query)

    def post(
        self,
        path: str,
        payload: Mapping[str, Any],
        *,
        idempotency_key: str,
        expected_revision: int,
    ) -> Any:
        return self._request(
            "POST",
            path,
            payload=payload,
            idempotency_key=idempotency_key,
            expected_revision=expected_revision,
        )

    def _request(
        self,
        method: str,
        path: str,
        *,
        query: Mapping[str, str | int] | None = None,
        payload: Mapping[str, Any] | None = None,
        idempotency_key: str | None = None,
        expected_revision: int | None = None,
    ) -> Any:
        if not path.startswith("/api/autopilot/") and path != "/api/autopilot/campaigns":
            raise AdapterConfigurationError("adapter attempted a non-Autopilot panel path")
        url = self.base_url + path
        if query:
            url += "?" + urlencode(query)
        headers = {"Accept": "application/json", "User-Agent": f"{SERVER_NAME}/{SERVER_VERSION}"}
        data = None
        if payload is not None:
            try:
                data = json.dumps(
                    payload,
                    ensure_ascii=False,
                    allow_nan=False,
                    separators=(",", ":"),
                ).encode("utf-8")
            except (TypeError, ValueError) as exc:
                raise ToolInputError("request contains a non-JSON or non-finite value") from exc
            headers["Content-Type"] = "application/json"
        if idempotency_key is not None:
            headers["Idempotency-Key"] = idempotency_key
        if expected_revision is not None:
            headers["If-Match"] = f'"{expected_revision}"'
            headers["X-Expected-Revision"] = str(expected_revision)
        if self.bearer_token:
            headers["Authorization"] = f"Bearer {self.bearer_token}"
        request = Request(url, data=data, headers=headers, method=method)
        try:
            with self.opener.open(request, timeout=self.timeout) as response:
                return self._decode_response(response, int(response.status))
        except HTTPError as exc:
            payload_or_text = self._decode_error_body(exc)
            code = "panel_http_error"
            message = f"panel returned HTTP {exc.code}"
            details: Any = payload_or_text
            if isinstance(payload_or_text, dict):
                code = str(payload_or_text.get("code") or code)
                message = str(payload_or_text.get("message") or message)
                details = payload_or_text.get("details", payload_or_text)
            raise PanelAPIError(int(exc.code), code, message, details) from None
        except (URLError, TimeoutError, OSError) as exc:
            raise PanelAPIError(
                503,
                "panel_unavailable",
                "loopback Training Panel is unavailable",
            ) from exc

    @staticmethod
    def _limited_read(response) -> bytes:  # noqa: ANN001
        body = response.read(MAX_RESPONSE_BYTES + 1)
        if len(body) > MAX_RESPONSE_BYTES:
            raise PanelAPIError(
                502,
                "panel_response_too_large",
                "panel response exceeded the adapter limit",
            )
        return body

    @classmethod
    def _decode_response(cls, response, status: int) -> Any:  # noqa: ANN001
        body = cls._limited_read(response)
        if status == 204 or not body:
            return {"status": status}
        try:
            return json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise PanelAPIError(
                502,
                "invalid_panel_response",
                "panel returned non-JSON data",
            ) from exc

    @classmethod
    def _decode_error_body(cls, response) -> Any:  # noqa: ANN001
        try:
            body = cls._limited_read(response)
        except PanelAPIError:
            return None
        if not body:
            return None
        try:
            return json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return body.decode("utf-8", errors="replace")[:2048]


READ_ANNOTATIONS = {
    "readOnlyHint": True,
    "destructiveHint": False,
    "idempotentHint": True,
    "openWorldHint": False,
}
WRITE_ANNOTATIONS = {
    "readOnlyHint": False,
    "destructiveHint": False,
    "idempotentHint": True,
    "openWorldHint": False,
}
STOP_ANNOTATIONS = {
    "readOnlyHint": False,
    "destructiveHint": True,
    "idempotentHint": True,
    "openWorldHint": False,
}

IDENTIFIER_SCHEMA = {
    "type": "string",
    "minLength": 1,
    "maxLength": 128,
    "pattern": _IDENTIFIER_RE.pattern,
}
IDEMPOTENCY_SCHEMA = {
    "type": "string",
    "minLength": 8,
    "maxLength": 128,
    "pattern": _IDEMPOTENCY_RE.pattern,
}
REVISION_SCHEMA = {"type": "integer", "minimum": 0}
SHA256_SCHEMA = {"type": "string", "pattern": _SHA256_RE.pattern}

COMMAND_INTERVAL_SCHEMA = {
    "type": "array",
    "minItems": 2,
    "maxItems": 2,
    "items": {"type": "number"},
}
COMMAND_AXIS_SCHEMA = {
    "type": "array",
    "minItems": 1,
    "maxItems": 8,
    "items": COMMAND_INTERVAL_SCHEMA,
}
SKILL_GATE_PROPERTIES = {
    "min_command_pass_ratio": {"type": "number", "minimum": 0.70, "maximum": 1.0},
    "min_skill_pass_ratio": {"type": "number", "minimum": 0.60, "maximum": 1.0},
    "max_fall_rate": {"type": "number", "minimum": 0.0, "maximum": 0.20},
    "min_tracking_quality": {"type": "number", "minimum": 0.0, "maximum": 1.0},
    "min_stability_quality": {"type": "number", "minimum": 0.0, "maximum": 1.0},
    "min_direction_sign_ratio": {"type": "number", "minimum": 0.70, "maximum": 1.0},
    "max_linear_leak": {"type": "number", "minimum": 0.0, "maximum": 0.18},
    "max_yaw_leak": {"type": "number", "minimum": 0.0, "maximum": 0.35},
    "max_energy_per_distance": {"type": "number", "exclusiveMinimum": 0.0, "maximum": 500.0},
}

GOAL_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "description",
        "task",
        "stage",
        "evaluation_profile",
        "gait",
        "directions",
        "command_envelope",
        "skill_gates",
        "initialization_mode",
        "baseline_run_id",
        "baseline_checkpoint_iteration",
        "checkpoint_sha256",
        "training_seeds",
        "per_trial_iteration_cap",
        "budget",
        "tunable_reward_keys",
    ],
    "properties": {
        "description": {"type": "string", "minLength": 1, "maxLength": 2000},
        "task": {"type": "string", "enum": [FORWARD_FAST_TASK, DIRECT_TASK]},
        "stage": {"type": "integer", "minimum": 1, "maximum": 5},
        "evaluation_profile": {
            "type": "string",
            "enum": ["stage1", "stage2", "stage3", "stage4", "stage5"],
        },
        "gait": {"type": "string", "enum": ["walk", "run"]},
        "directions": {
            "type": "array",
            "minItems": 1,
            "uniqueItems": True,
            "items": {
                "type": "string",
                "enum": [
                    "forward",
                    "left",
                    "right",
                    "forward_left",
                    "forward_right",
                    "yaw_ccw",
                    "yaw_cw",
                ],
            },
        },
        "command_envelope": {
            "type": "object",
            "additionalProperties": False,
            "required": ["vx", "vy", "wz"],
            "properties": {
                "vx": COMMAND_AXIS_SCHEMA,
                "vy": COMMAND_AXIS_SCHEMA,
                "wz": COMMAND_AXIS_SCHEMA,
            },
        },
        "skill_gates": {
            "type": "object",
            "additionalProperties": False,
            "required": list(SKILL_GATE_PROPERTIES),
            "properties": SKILL_GATE_PROPERTIES,
        },
        "initialization_mode": {"type": "string", "enum": ["fresh", "policy_only"]},
        "baseline_run_id": {"type": ["string", "null"], "maxLength": 128},
        "baseline_checkpoint_iteration": {
            "type": ["integer", "null"],
            "minimum": 0,
        },
        "checkpoint_sha256": {"anyOf": [SHA256_SCHEMA, {"type": "null"}]},
        "training_seeds": {
            "type": "array",
            "enum": [[42, 43, 44]],
        },
        "per_trial_iteration_cap": {"type": "integer", "minimum": 1},
        "tunable_reward_keys": {
            "type": "array",
            "uniqueItems": True,
            "items": {"type": "string", "pattern": _REWARD_KEY_RE.pattern},
        },
        "reward_bounds": {
            "type": "object",
            "propertyNames": {"pattern": _REWARD_KEY_RE.pattern},
            "additionalProperties": {
                "type": "array",
                "minItems": 2,
                "maxItems": 2,
                "items": {"type": "number"},
            },
        },
        "budget": {
            "type": "object",
            "additionalProperties": False,
            "required": ["max_training_trials", "max_gpu_hours"],
            "properties": {
                "max_training_trials": {"type": "integer", "minimum": 1, "maximum": 24},
                "max_gpu_hours": {"type": "number", "exclusiveMinimum": 0, "maximum": 72},
            },
        },
    },
}


def _object_schema(
    properties: Mapping[str, Any],
    required: list[str],
) -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": required,
        "properties": dict(properties),
    }


MUTATION_PROPERTIES = {
    "campaign_id": IDENTIFIER_SCHEMA,
    "idempotency_key": IDEMPOTENCY_SCHEMA,
    "expected_revision": REVISION_SCHEMA,
}

TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "name": "redrhex_list_campaigns",
        "title": "List RedRHex Autopilot campaigns",
        "description": "List bounded campaign summaries without advancing or reconciling work.",
        "inputSchema": _object_schema(
            {
                "state": {"type": "string", "minLength": 1, "maxLength": 64},
                "limit": {"type": "integer", "minimum": 1, "maximum": 100},
            },
            [],
        ),
        "annotations": READ_ANNOTATIONS,
    },
    {
        "name": "redrhex_get_campaign",
        "title": "Get a RedRHex Autopilot campaign",
        "description": "Read one versioned campaign snapshot and its next permitted action.",
        "inputSchema": _object_schema({"campaign_id": IDENTIFIER_SCHEMA}, ["campaign_id"]),
        "annotations": READ_ANNOTATIONS,
    },
    {
        "name": "redrhex_get_decision_context",
        "title": "Get bounded advisor context",
        "description": "Read compact deterministic evidence and server-allowed reward moves for one campaign.",
        "inputSchema": _object_schema({"campaign_id": IDENTIFIER_SCHEMA}, ["campaign_id"]),
        "annotations": READ_ANNOTATIONS,
    },
    {
        "name": "redrhex_compare_trials",
        "title": "Compare campaign trials",
        "description": "Compare two to twelve explicitly named trials within one campaign.",
        "inputSchema": _object_schema(
            {
                "campaign_id": IDENTIFIER_SCHEMA,
                "trial_ids": {
                    "type": "array",
                    "minItems": 2,
                    "maxItems": 12,
                    "uniqueItems": True,
                    "items": IDENTIFIER_SCHEMA,
                },
            },
            ["campaign_id", "trial_ids"],
        ),
        "annotations": READ_ANNOTATIONS,
    },
    {
        "name": "redrhex_get_artifact_link",
        "title": "Get a campaign artifact link",
        "description": "Resolve metadata or a panel-local link for one immutable campaign artifact.",
        "inputSchema": _object_schema(
            {"campaign_id": IDENTIFIER_SCHEMA, "artifact_id": IDENTIFIER_SCHEMA},
            ["campaign_id", "artifact_id"],
        ),
        "annotations": READ_ANNOTATIONS,
    },
    {
        "name": "redrhex_advisor_heartbeat",
        "title": "Record an advisor poll heartbeat",
        "description": "Record one idempotent scheduled-advisor visit for an active nonterminal campaign without launching or deciding work.",
        "inputSchema": _object_schema(
            MUTATION_PROPERTIES,
            ["campaign_id", "idempotency_key", "expected_revision"],
        ),
        "annotations": WRITE_ANNOTATIONS,
    },
    {
        "name": "redrhex_create_goal_draft",
        "title": "Create a RedRHex goal draft",
        "description": "Create an unarmed bounded locomotion-goal draft; the panel resolves immutable identity hashes and this cannot start training.",
        "inputSchema": _object_schema(
            {
                "goal": GOAL_SCHEMA,
                "idempotency_key": IDEMPOTENCY_SCHEMA,
                "expected_revision": {"type": "integer", "const": 0},
            },
            ["goal", "idempotency_key", "expected_revision"],
        ),
        "annotations": WRITE_ANNOTATIONS,
    },
    {
        "name": "redrhex_propose_candidate",
        "title": "Propose one bounded reward candidate",
        "description": "Submit one reward-key/value hypothesis for deterministic server validation.",
        "inputSchema": _object_schema(
            {
                **MUTATION_PROPERTIES,
                "evidence_ids": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 32,
                    "uniqueItems": True,
                    "items": IDENTIFIER_SCHEMA,
                },
                "hypothesis": {"type": "string", "minLength": 1, "maxLength": 2000},
                "reward_key": {"type": "string", "minLength": 1, "maxLength": 128},
                "proposed_value": {"type": "number"},
                "expected_metric_effect": {"type": "string", "minLength": 1, "maxLength": 1000},
                "rationale": {"type": "string", "minLength": 1, "maxLength": 4000},
            },
            [
                "campaign_id",
                "idempotency_key",
                "expected_revision",
                "evidence_ids",
                "hypothesis",
                "reward_key",
                "proposed_value",
                "expected_metric_effect",
                "rationale",
            ],
        ),
        "annotations": WRITE_ANNOTATIONS,
    },
    {
        "name": "redrhex_pause_campaign",
        "title": "Pause a RedRHex campaign",
        "description": "Request a scheduling pause; running local work finishes and remains recorded.",
        "inputSchema": _object_schema(
            {**MUTATION_PROPERTIES, "reason": {"type": "string", "minLength": 1, "maxLength": 1000}},
            ["campaign_id", "idempotency_key", "expected_revision", "reason"],
        ),
        "annotations": WRITE_ANNOTATIONS,
    },
    {
        "name": "redrhex_request_stop_after_current",
        "title": "Stop a campaign after current work",
        "description": "Request a graceful terminal stop after the campaign-owned active job completes.",
        "inputSchema": _object_schema(
            {**MUTATION_PROPERTIES, "reason": {"type": "string", "minLength": 1, "maxLength": 1000}},
            ["campaign_id", "idempotency_key", "expected_revision", "reason"],
        ),
        "annotations": STOP_ANNOTATIONS,
    },
    {
        "name": "redrhex_submit_patch_proposal",
        "title": "Store a reward-source patch proposal",
        "description": "Store a review-only diff artifact during patch handoff; never applies repository changes.",
        "inputSchema": _object_schema(
            {
                **MUTATION_PROPERTIES,
                "target_symbols": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 32,
                    "uniqueItems": True,
                    "items": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": 128,
                        "pattern": _PATCH_SYMBOL_RE.pattern,
                    },
                },
                "base_blob_hashes": {
                    "type": "object",
                    "minProperties": 1,
                    "maxProperties": 2,
                    "propertyNames": {"enum": sorted(PATCH_SOURCE_ALLOWLIST)},
                    "additionalProperties": SHA256_SCHEMA,
                },
                "unified_diff": {"type": "string", "minLength": 1, "maxLength": 200000},
                "rationale": {"type": "string", "minLength": 1, "maxLength": 4000},
                "test_plan": {"type": "string", "minLength": 1, "maxLength": 8000},
                "rollback_notes": {"type": "string", "minLength": 1, "maxLength": 4000},
            },
            [
                "campaign_id",
                "idempotency_key",
                "expected_revision",
                "target_symbols",
                "base_blob_hashes",
                "unified_diff",
                "rationale",
                "test_plan",
                "rollback_notes",
            ],
        ),
        "annotations": WRITE_ANNOTATIONS,
    },
]


def _ensure_object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ToolInputError(f"{label} must be an object")
    return value


def _check_keys(
    arguments: dict[str, Any],
    *,
    allowed: set[str],
    required: set[str],
) -> None:
    unknown = set(arguments) - allowed
    missing = required - set(arguments)
    if unknown:
        raise ToolInputError(f"unknown argument(s): {', '.join(sorted(unknown))}")
    if missing:
        raise ToolInputError(f"missing argument(s): {', '.join(sorted(missing))}")


def _identifier(arguments: Mapping[str, Any], key: str) -> str:
    value = arguments.get(key)
    if not isinstance(value, str) or _IDENTIFIER_RE.fullmatch(value) is None:
        raise ToolInputError(f"{key} must be a safe identifier")
    return value


def _bounded_string(arguments: Mapping[str, Any], key: str, maximum: int) -> str:
    value = arguments.get(key)
    if not isinstance(value, str) or not value.strip() or len(value) > maximum:
        raise ToolInputError(f"{key} must be a non-empty string of at most {maximum} characters")
    return value


def _mutation_metadata(arguments: Mapping[str, Any]) -> tuple[str, str, int]:
    campaign_id = _identifier(arguments, "campaign_id")
    key = arguments.get("idempotency_key")
    if not isinstance(key, str) or _IDEMPOTENCY_RE.fullmatch(key) is None:
        raise ToolInputError("idempotency_key must be a stable safe identifier of 8..128 characters")
    revision = arguments.get("expected_revision")
    if isinstance(revision, bool) or not isinstance(revision, int) or revision < 0:
        raise ToolInputError("expected_revision must be a non-negative integer")
    return campaign_id, key, revision


def _finite_number(arguments: Mapping[str, Any], key: str) -> float:
    value = arguments.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ToolInputError(f"{key} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ToolInputError(f"{key} must be finite")
    return result


_CAMPAIGN_PATH_FIELDS = {
    "output_checkpoint_path",
    "checkpoint_path",
    "command_profile_file",
    "reward_profile_file",
    "terrain_profile_file",
    "physics_profile_file",
    "log_dir",
    "process_log",
}


def _without_local_paths(value: Any) -> Any:
    """Remove host-local paths from campaign data crossing the MCP boundary."""

    if isinstance(value, Mapping):
        return {
            key: _without_local_paths(item)
            for key, item in value.items()
            if key not in _CAMPAIGN_PATH_FIELDS
        }
    if isinstance(value, list):
        return [_without_local_paths(item) for item in value]
    return value


class ToolGateway:
    def __init__(self, client: PanelClient) -> None:
        self.client = client

    def call(self, name: str, arguments: Any) -> Any:
        args = _ensure_object(arguments, "arguments")
        handler = getattr(self, f"_tool_{name}", None)
        if handler is None:
            raise ToolInputError(f"unknown tool: {name}")
        # The panel's administrative DTOs intentionally retain host-local
        # paths for the operator UI.  No MCP tool, including a write tool's
        # response, may carry those paths across the connector boundary.
        return _without_local_paths(handler(args))

    def _tool_redrhex_list_campaigns(self, args: dict[str, Any]) -> Any:
        _check_keys(args, allowed={"state", "limit"}, required=set())
        query: dict[str, str | int] = {}
        if "state" in args:
            query["state"] = _bounded_string(args, "state", 64)
        if "limit" in args:
            limit = args["limit"]
            if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 100:
                raise ToolInputError("limit must be an integer within 1..100")
            query["limit"] = limit
        return _without_local_paths(
            self.client.get("/api/autopilot/campaigns", query=query or None)
        )

    def _tool_redrhex_get_campaign(self, args: dict[str, Any]) -> Any:
        _check_keys(args, allowed={"campaign_id"}, required={"campaign_id"})
        campaign_id = quote(_identifier(args, "campaign_id"), safe="")
        return _without_local_paths(
            self.client.get(f"/api/autopilot/campaigns/{campaign_id}")
        )

    def _tool_redrhex_get_decision_context(self, args: dict[str, Any]) -> Any:
        _check_keys(args, allowed={"campaign_id"}, required={"campaign_id"})
        campaign_id = quote(_identifier(args, "campaign_id"), safe="")
        return self.client.get(f"/api/autopilot/campaigns/{campaign_id}/decision-context")

    def _tool_redrhex_compare_trials(self, args: dict[str, Any]) -> Any:
        _check_keys(args, allowed={"campaign_id", "trial_ids"}, required={"campaign_id", "trial_ids"})
        campaign_id = quote(_identifier(args, "campaign_id"), safe="")
        raw_ids = args["trial_ids"]
        if not isinstance(raw_ids, list) or not 2 <= len(raw_ids) <= 12:
            raise ToolInputError("trial_ids must contain 2..12 identifiers")
        trial_ids: list[str] = []
        for value in raw_ids:
            trial_ids.append(_identifier({"trial_id": value}, "trial_id"))
        if len(set(trial_ids)) != len(trial_ids):
            raise ToolInputError("trial_ids must be unique")
        return self.client.get(
            f"/api/autopilot/campaigns/{campaign_id}/compare",
            query={"trial_ids": ",".join(trial_ids)},
        )

    def _tool_redrhex_get_artifact_link(self, args: dict[str, Any]) -> Any:
        _check_keys(args, allowed={"campaign_id", "artifact_id"}, required={"campaign_id", "artifact_id"})
        campaign_id = quote(_identifier(args, "campaign_id"), safe="")
        artifact_id = quote(_identifier(args, "artifact_id"), safe="")
        return self.client.get(f"/api/autopilot/campaigns/{campaign_id}/artifacts/{artifact_id}")

    def _tool_redrhex_advisor_heartbeat(self, args: dict[str, Any]) -> Any:
        allowed = {"campaign_id", "idempotency_key", "expected_revision"}
        _check_keys(args, allowed=allowed, required=allowed)
        campaign_id, key, revision = _mutation_metadata(args)
        return self.client.post(
            f"/api/autopilot/campaigns/{quote(campaign_id, safe='')}/heartbeat",
            {
                "expected_revision": revision,
                "advisor_metadata": advisor_metadata(),
            },
            idempotency_key=key,
            expected_revision=revision,
        )

    def _tool_redrhex_create_goal_draft(self, args: dict[str, Any]) -> Any:
        _check_keys(
            args,
            allowed={"goal", "idempotency_key", "expected_revision"},
            required={"goal", "idempotency_key", "expected_revision"},
        )
        key = args["idempotency_key"]
        revision = args["expected_revision"]
        if not isinstance(key, str) or _IDEMPOTENCY_RE.fullmatch(key) is None:
            raise ToolInputError("idempotency_key must be a stable safe identifier of 8..128 characters")
        if isinstance(revision, bool) or revision != 0:
            raise ToolInputError("new drafts require expected_revision=0")
        goal = self._validate_goal(_ensure_object(args["goal"], "goal"))
        tunable_reward_keys = goal.pop("tunable_reward_keys")
        reward_bounds = goal.pop("reward_bounds", {})
        return self.client.post(
            "/api/autopilot/campaigns",
            {
                **goal,
                "schema_version": GOAL_SCHEMA_VERSION,
                "tunable_reward_keys": tunable_reward_keys,
                "reward_bounds": reward_bounds,
                "expected_revision": 0,
            },
            idempotency_key=key,
            expected_revision=0,
        )

    def _tool_redrhex_propose_candidate(self, args: dict[str, Any]) -> Any:
        allowed = {
            "campaign_id",
            "idempotency_key",
            "expected_revision",
            "evidence_ids",
            "hypothesis",
            "reward_key",
            "proposed_value",
            "expected_metric_effect",
            "rationale",
        }
        _check_keys(args, allowed=allowed, required=allowed)
        campaign_id, key, revision = _mutation_metadata(args)
        evidence_ids = args["evidence_ids"]
        if not isinstance(evidence_ids, list) or not 1 <= len(evidence_ids) <= 32:
            raise ToolInputError("evidence_ids must contain 1..32 identifiers")
        evidence = [_identifier({"evidence_id": value}, "evidence_id") for value in evidence_ids]
        if len(set(evidence)) != len(evidence):
            raise ToolInputError("evidence_ids must be unique")
        reward_key = _bounded_string(args, "reward_key", 128)
        if _REWARD_KEY_RE.fullmatch(reward_key) is None:
            raise ToolInputError("reward_key must be an allowlisted v2_reward_scales.* key")
        decision = {
            "schema_version": DECISION_SCHEMA_VERSION,
            "campaign_id": campaign_id,
            "campaign_revision": revision,
            "action": "propose_candidate",
            "evidence_ids": evidence,
            "hypothesis": _bounded_string(args, "hypothesis", 2000),
            "reward_key": reward_key,
            "proposed_value": _finite_number(args, "proposed_value"),
            "expected_metric_effect": _bounded_string(args, "expected_metric_effect", 1000),
            "rationale": _bounded_string(args, "rationale", 4000),
        }
        payload = {
            "expected_revision": revision,
            "decision": decision,
            "advisor_metadata": advisor_metadata(),
        }
        return self.client.post(
            f"/api/autopilot/campaigns/{quote(campaign_id, safe='')}/decisions",
            payload,
            idempotency_key=key,
            expected_revision=revision,
        )

    def _tool_redrhex_pause_campaign(self, args: dict[str, Any]) -> Any:
        allowed = {"campaign_id", "idempotency_key", "expected_revision", "reason"}
        _check_keys(args, allowed=allowed, required=allowed)
        campaign_id, key, revision = _mutation_metadata(args)
        return self.client.post(
            f"/api/autopilot/campaigns/{quote(campaign_id, safe='')}/pause",
            {
                "expected_revision": revision,
                "reason": _bounded_string(args, "reason", 1000),
                "advisor_metadata": advisor_metadata(),
            },
            idempotency_key=key,
            expected_revision=revision,
        )

    def _tool_redrhex_request_stop_after_current(self, args: dict[str, Any]) -> Any:
        allowed = {"campaign_id", "idempotency_key", "expected_revision", "reason"}
        _check_keys(args, allowed=allowed, required=allowed)
        campaign_id, key, revision = _mutation_metadata(args)
        return self.client.post(
            f"/api/autopilot/campaigns/{quote(campaign_id, safe='')}/stop",
            {
                "expected_revision": revision,
                "mode": "after_current",
                "reason": _bounded_string(args, "reason", 1000),
                "advisor_metadata": advisor_metadata(),
            },
            idempotency_key=key,
            expected_revision=revision,
        )

    def _tool_redrhex_submit_patch_proposal(self, args: dict[str, Any]) -> Any:
        allowed = {
            "campaign_id",
            "idempotency_key",
            "expected_revision",
            "target_symbols",
            "base_blob_hashes",
            "unified_diff",
            "rationale",
            "test_plan",
            "rollback_notes",
        }
        _check_keys(args, allowed=allowed, required=allowed)
        campaign_id, key, revision = _mutation_metadata(args)
        symbols = args["target_symbols"]
        if not isinstance(symbols, list) or not 1 <= len(symbols) <= 32:
            raise ToolInputError("target_symbols must contain 1..32 entries")
        normalized_symbols = [
            _bounded_string({"symbol": value}, "symbol", 128) for value in symbols
        ]
        if len(set(normalized_symbols)) != len(normalized_symbols):
            raise ToolInputError("target_symbols must be unique")
        if any(_PATCH_SYMBOL_RE.fullmatch(value) is None for value in normalized_symbols):
            raise ToolInputError("target_symbols contains an invalid Python symbol")
        hashes = _ensure_object(args["base_blob_hashes"], "base_blob_hashes")
        if not 1 <= len(hashes) <= len(PATCH_SOURCE_ALLOWLIST):
            raise ToolInputError("base_blob_hashes must contain 1..2 allowlisted source paths")
        normalized_hashes: dict[str, str] = {}
        for source_id, digest in hashes.items():
            if source_id not in PATCH_SOURCE_ALLOWLIST:
                raise ToolInputError("base_blob_hashes contains a non-allowlisted source path")
            if not isinstance(digest, str) or _SHA256_RE.fullmatch(digest) is None:
                raise ToolInputError("base_blob_hashes values must be SHA-256 hex digests")
            normalized_hashes[source_id] = digest.lower()
        unified_diff = _bounded_string(args, "unified_diff", 200000)
        if "GIT binary patch" in unified_diff or "Binary files " in unified_diff:
            raise ToolInputError("binary patch proposals are forbidden")
        header_paths: set[str] = set()
        for line in unified_diff.splitlines():
            if not (line.startswith("--- ") or line.startswith("+++ ")):
                continue
            raw_path = line[4:].split("\t", 1)[0]
            if raw_path == "/dev/null":
                raise ToolInputError("patch proposals may not create or delete source files")
            header_paths.add(raw_path.removeprefix("a/").removeprefix("b/"))
        if not header_paths or header_paths - PATCH_SOURCE_ALLOWLIST:
            raise ToolInputError("unified_diff touches a non-allowlisted source path")
        proposal = {
            "schema_version": PATCH_SCHEMA_VERSION,
            "target_symbols": normalized_symbols,
            "base_blob_hashes": normalized_hashes,
            "unified_diff": unified_diff,
            "rationale": _bounded_string(args, "rationale", 4000),
            "test_plan": _bounded_string(args, "test_plan", 8000),
            "rollback_notes": _bounded_string(args, "rollback_notes", 4000),
        }
        decision = {
            "schema_version": DECISION_SCHEMA_VERSION,
            "campaign_id": campaign_id,
            "campaign_revision": revision,
            "evidence_ids": [],
            "hypothesis": "Bounded reward-weight search requires a human-reviewed source proposal.",
            "action": "request_patch_handoff",
            "reward_key": None,
            "proposed_value": None,
            "expected_metric_effect": None,
            "rationale": _bounded_string(args, "rationale", 4000),
        }
        payload = {
            "expected_revision": revision,
            "decision": decision,
            "patch_proposal": proposal,
            "advisor_metadata": advisor_metadata(),
        }
        return self.client.post(
            f"/api/autopilot/campaigns/{quote(campaign_id, safe='')}/decisions",
            payload,
            idempotency_key=key,
            expected_revision=revision,
        )

    @staticmethod
    def _validate_goal(goal: dict[str, Any]) -> dict[str, Any]:
        allowed = set(GOAL_SCHEMA["properties"])
        required = set(GOAL_SCHEMA["required"])
        _check_keys(goal, allowed=allowed, required=required)
        _bounded_string(goal, "description", 2000)
        if goal["task"] not in {FORWARD_FAST_TASK, DIRECT_TASK}:
            raise ToolInputError("task must be a supported standard PPO task")
        stage = goal["stage"]
        if isinstance(stage, bool) or not isinstance(stage, int) or not 1 <= stage <= 5:
            raise ToolInputError("stage must be an integer within 1..5")
        if goal["task"] == FORWARD_FAST_TASK and stage != 1:
            raise ToolInputError("ForwardFast supports stage 1 only")
        if goal["evaluation_profile"] != f"stage{stage}":
            raise ToolInputError("evaluation_profile must match the selected stage")
        if goal["gait"] not in {"walk", "run"}:
            raise ToolInputError("gait must be walk or run")
        if goal["initialization_mode"] not in {"fresh", "policy_only"}:
            raise ToolInputError("initialization_mode must be fresh or policy_only")
        if goal["initialization_mode"] == "policy_only":
            if (
                goal["baseline_run_id"] is None
                or goal["baseline_checkpoint_iteration"] is None
                or goal["checkpoint_sha256"] is None
            ):
                raise ToolInputError(
                    "policy_only initialization requires baseline_run_id, "
                    "baseline_checkpoint_iteration, and checkpoint_sha256"
                )
        elif any(
            goal[name] is not None
            for name in (
                "baseline_run_id",
                "baseline_checkpoint_iteration",
                "checkpoint_sha256",
            )
        ):
            raise ToolInputError("fresh initialization may not specify checkpoint identity")
        if goal["baseline_run_id"] is not None:
            _identifier(goal, "baseline_run_id")
        if goal["checkpoint_sha256"] is not None:
            digest = goal["checkpoint_sha256"]
            if not isinstance(digest, str) or _SHA256_RE.fullmatch(digest) is None:
                raise ToolInputError("checkpoint_sha256 must be a lowercase SHA-256 digest")
        checkpoint_iteration = goal["baseline_checkpoint_iteration"]
        if checkpoint_iteration is not None and (
            isinstance(checkpoint_iteration, bool)
            or not isinstance(checkpoint_iteration, int)
            or checkpoint_iteration < 0
        ):
            raise ToolInputError("baseline_checkpoint_iteration must be a non-negative integer")
        envelope = _ensure_object(goal["command_envelope"], "command_envelope")
        _check_keys(envelope, allowed={"vx", "vy", "wz"}, required={"vx", "vy", "wz"})
        for axis in ("vx", "vy", "wz"):
            intervals = envelope[axis]
            if not isinstance(intervals, list) or not 1 <= len(intervals) <= 8:
                raise ToolInputError(f"command_envelope.{axis} must contain 1..8 intervals")
            for index, interval in enumerate(intervals):
                if not isinstance(interval, list) or len(interval) != 2:
                    raise ToolInputError(f"command_envelope.{axis}[{index}] must be [minimum, maximum]")
                minimum = _finite_number({"minimum": interval[0]}, "minimum")
                maximum = _finite_number({"maximum": interval[1]}, "maximum")
                if minimum > maximum:
                    raise ToolInputError(f"command_envelope.{axis}[{index}] is reversed")
        directions = goal["directions"]
        directions_by_stage = {
            1: {"forward"},
            2: {"left", "right"},
            3: {"forward_left", "forward_right"},
            4: {"yaw_ccw", "yaw_cw"},
            5: {"forward", "left", "right", "forward_left", "forward_right", "yaw_ccw", "yaw_cw"},
        }
        if (
            not isinstance(directions, list)
            or not directions
            or len(set(directions)) != len(directions)
            or any(not isinstance(value, str) or value not in directions_by_stage[stage] for value in directions)
        ):
            raise ToolInputError("directions must be unique and valid for the selected stage")
        gates = _ensure_object(goal["skill_gates"], "skill_gates")
        _check_keys(
            gates,
            allowed=set(SKILL_GATE_PROPERTIES),
            required=set(SKILL_GATE_PROPERTIES),
        )
        gate_values = {key: _finite_number(gates, key) for key in SKILL_GATE_PROPERTIES}
        minimums = {
            "min_command_pass_ratio": 0.70,
            "min_skill_pass_ratio": 0.60,
            "min_tracking_quality": 0.0,
            "min_stability_quality": 0.0,
            "min_direction_sign_ratio": 0.70,
        }
        maximums = {
            "max_fall_rate": 0.20,
            "max_linear_leak": 0.18,
            "max_yaw_leak": 0.35,
            "max_energy_per_distance": 500.0,
        }
        for key, minimum in minimums.items():
            if not minimum <= gate_values[key] <= 1.0:
                raise ToolInputError(f"skill_gates.{key} relaxes the V1 safety minimum")
        for key, maximum in maximums.items():
            if gate_values[key] < 0.0 or gate_values[key] > maximum:
                raise ToolInputError(f"skill_gates.{key} relaxes the V1 safety maximum")
        if gate_values["max_energy_per_distance"] <= 0.0:
            raise ToolInputError("skill_gates.max_energy_per_distance must be positive")
        seeds = goal["training_seeds"]
        if seeds != [42, 43, 44]:
            raise ToolInputError("training_seeds must be [42, 43, 44] in V1")
        iteration_cap = goal["per_trial_iteration_cap"]
        if isinstance(iteration_cap, bool) or not isinstance(iteration_cap, int) or iteration_cap < 1:
            raise ToolInputError("per_trial_iteration_cap must be a positive integer")
        keys = goal["tunable_reward_keys"]
        if not isinstance(keys, list) or not keys or len(set(keys)) != len(keys):
            raise ToolInputError("tunable_reward_keys must be a non-empty unique list")
        for value in keys:
            if not isinstance(value, str) or _REWARD_KEY_RE.fullmatch(value) is None:
                raise ToolInputError("tunable_reward_keys must contain v2_reward_scales.* keys")
        bounds = goal.get("reward_bounds", {})
        if not isinstance(bounds, dict):
            raise ToolInputError("reward_bounds must be an object")
        unknown_bounds = sorted(set(bounds) - set(keys))
        if unknown_bounds:
            raise ToolInputError("reward_bounds may only contain selected tunable_reward_keys")
        for key, interval in bounds.items():
            if _REWARD_KEY_RE.fullmatch(key) is None:
                raise ToolInputError("reward_bounds keys must be v2_reward_scales.* keys")
            if not isinstance(interval, list) or len(interval) != 2:
                raise ToolInputError(f"reward_bounds.{key} must be [minimum, maximum]")
            minimum = _finite_number({"minimum": interval[0]}, "minimum")
            maximum = _finite_number({"maximum": interval[1]}, "maximum")
            if minimum > maximum:
                raise ToolInputError(f"reward_bounds.{key} is reversed")
        budget = _ensure_object(goal["budget"], "budget")
        _check_keys(
            budget,
            allowed={"max_training_trials", "max_gpu_hours"},
            required={"max_training_trials", "max_gpu_hours"},
        )
        trials = budget["max_training_trials"]
        if isinstance(trials, bool) or not isinstance(trials, int) or not 1 <= trials <= 24:
            raise ToolInputError("max_training_trials must be an integer within 1..24")
        hours = _finite_number(budget, "max_gpu_hours")
        if not 0 < hours <= 72:
            raise ToolInputError("max_gpu_hours must be within (0, 72]")
        # Round-trip through strict JSON serialization to detach caller-owned objects and
        # reject any non-JSON value not covered by the focused checks above.
        try:
            return json.loads(json.dumps(goal, allow_nan=False))
        except (TypeError, ValueError) as exc:
            raise ToolInputError("goal must contain only finite JSON values") from exc


def _tool_result(payload: Any) -> dict[str, Any]:
    structured = {"ok": True, "data": payload}
    return {
        "content": [
            {
                "type": "text",
                "text": json.dumps(structured, ensure_ascii=False, allow_nan=False, separators=(",", ":")),
            }
        ],
        "structuredContent": structured,
        "isError": False,
    }


def _tool_error(code: str, message: str, details: Any = None) -> dict[str, Any]:
    error: dict[str, Any] = {"code": code, "message": message}
    if details is not None:
        error["details"] = details
    structured = {"ok": False, "error": error}
    return {
        "content": [
            {
                "type": "text",
                "text": json.dumps(structured, ensure_ascii=False, allow_nan=False, separators=(",", ":")),
            }
        ],
        "structuredContent": structured,
        "isError": True,
    }


_MISSING = object()


class MCPServer:
    def __init__(self, gateway: ToolGateway | None = None) -> None:
        self._gateway_instance = gateway

    def _gateway(self) -> ToolGateway:
        if self._gateway_instance is None:
            self._gateway_instance = ToolGateway(PanelClient())
        return self._gateway_instance

    def handle(self, message: Any) -> dict[str, Any] | None:
        if not isinstance(message, dict):
            return self._rpc_error(None, -32600, "request must be a JSON object")
        request_id = message.get("id", _MISSING)
        is_notification = request_id is _MISSING or request_id is None
        if message.get("jsonrpc") != "2.0" or not isinstance(message.get("method"), str):
            return None if is_notification else self._rpc_error(request_id, -32600, "invalid JSON-RPC request")
        method = message["method"]
        params = message.get("params", {})
        if method in {"notifications/initialized", "notifications/cancelled", "notifications/progress"}:
            return None
        if is_notification:
            return None
        try:
            if method == "initialize":
                params_object = _ensure_object(params, "initialize params")
                requested = params_object.get("protocolVersion")
                protocol = requested if requested in SUPPORTED_PROTOCOL_VERSIONS else LATEST_PROTOCOL_VERSION
                result = {
                    "protocolVersion": protocol,
                    "capabilities": {"tools": {"listChanged": False}},
                    "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION},
                    "instructions": (
                        "The Training Panel is authoritative. Read campaign context before one bounded write; "
                        "never arm, resume, deploy, edit files, or claim success."
                    ),
                }
            elif method == "ping":
                result = {}
            elif method == "tools/list":
                params_object = _ensure_object(params, "tools/list params")
                if params_object.get("cursor") not in {None, ""}:
                    raise ToolInputError("pagination cursors are not supported")
                result = {"tools": TOOL_DEFINITIONS}
            elif method == "tools/call":
                params_object = _ensure_object(params, "tools/call params")
                name = params_object.get("name")
                if not isinstance(name, str):
                    raise ToolInputError("tool name must be a string")
                arguments = params_object.get("arguments", {})
                try:
                    result = _tool_result(self._gateway().call(name, arguments))
                except ToolInputError as exc:
                    result = _tool_error("invalid_tool_input", str(exc))
                except AdapterConfigurationError as exc:
                    result = _tool_error("adapter_configuration_error", str(exc))
                except PanelAPIError as exc:
                    result = _tool_error(exc.code, exc.message, exc.details)
                except Exception:
                    result = _tool_error("adapter_internal_error", "adapter failed safely")
            else:
                return self._rpc_error(request_id, -32601, f"method not found: {method}")
        except ToolInputError as exc:
            return self._rpc_error(request_id, -32602, str(exc))
        return {"jsonrpc": "2.0", "id": request_id, "result": result}

    @staticmethod
    def _rpc_error(request_id: Any, code: int, message: str) -> dict[str, Any]:
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {"code": code, "message": message},
        }


def dispatch_payload(server: MCPServer, incoming: Any) -> Any | None:
    """Dispatch a single JSON-RPC message or batch for either transport."""

    if isinstance(incoming, list):
        if not incoming:
            return MCPServer._rpc_error(None, -32600, "empty JSON-RPC batch")
        responses = [
            response
            for item in incoming
            if (response := server.handle(item)) is not None
        ]
        return responses or None
    return server.handle(incoming)


def _accepted_media_types(raw_accept: str) -> set[str]:
    return {
        item.split(";", 1)[0].strip().lower()
        for item in raw_accept.split(",")
        if item.strip()
    }


class AutopilotMCPHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(
        self,
        server_address,
        request_handler_class,
        *,
        mcp_server: MCPServer,
        allowed_origins: frozenset[str],
        bearer_token: str,
    ) -> None:
        super().__init__(server_address, request_handler_class)
        self.mcp_server = mcp_server
        self.allowed_origins = allowed_origins
        self.bearer_token = bearer_token


class AutopilotMCPHTTPServerV6(AutopilotMCPHTTPServer):
    address_family = socket.AF_INET6


class MCPHTTPHandler(BaseHTTPRequestHandler):
    """Stateless Streamable HTTP transport for one fixed MCP endpoint."""

    protocol_version = "HTTP/1.1"
    server_version = "RedRHexAutopilotMCP/0.1"
    sys_version = ""

    @property
    def autopilot_server(self) -> AutopilotMCPHTTPServer:
        return self.server  # type: ignore[return-value]

    def log_message(self, _format: str, *_args: Any) -> None:
        # Avoid recording request data or authentication material in runtime logs.
        return

    def do_GET(self) -> None:  # noqa: N802
        if not self._authorize_request():
            return
        # This stateless server never initiates requests or notifications. MCP permits
        # a 405 response when no standalone SSE stream is offered.
        self.send_response(405)
        self.send_header("Allow", "POST")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_POST(self) -> None:  # noqa: N802
        if not self._authorize_request():
            return
        content_type = self.headers.get("Content-Type", "").split(";", 1)[0].strip().lower()
        if content_type != "application/json":
            self._send_http_error(415, "invalid_content_type", "Content-Type must be application/json")
            return
        accepted = _accepted_media_types(self.headers.get("Accept", ""))
        if not {"application/json", "text/event-stream"}.issubset(accepted):
            self._send_http_error(
                406,
                "invalid_accept_header",
                "Accept must list application/json and text/event-stream",
            )
            return
        if self.headers.get("Transfer-Encoding"):
            self._send_http_error(400, "unsupported_transfer_encoding", "chunked requests are not supported")
            return
        lengths = self.headers.get_all("Content-Length", failobj=[])
        if len(lengths) != 1:
            self._send_http_error(411, "content_length_required", "one Content-Length header is required")
            return
        try:
            content_length = int(lengths[0], 10)
        except (TypeError, ValueError):
            self._send_http_error(400, "invalid_content_length", "Content-Length must be an integer")
            return
        if content_length < 0:
            self._send_http_error(400, "invalid_content_length", "Content-Length must be non-negative")
            return
        if content_length > MAX_INPUT_BYTES:
            self._send_http_error(413, "request_too_large", "request exceeded the adapter limit")
            return
        raw_protocol = self.headers.get("MCP-Protocol-Version")
        if raw_protocol is not None and raw_protocol not in SUPPORTED_PROTOCOL_VERSIONS:
            self._send_http_error(400, "unsupported_protocol_version", "unsupported MCP protocol version")
            return
        raw = self.rfile.read(content_length)
        if len(raw) != content_length:
            self._send_http_error(400, "incomplete_request", "request body was incomplete")
            return
        try:
            incoming = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            self._send_json(
                400,
                MCPServer._rpc_error(None, -32700, "invalid JSON"),
            )
            return
        response = dispatch_payload(self.autopilot_server.mcp_server, incoming)
        if response is None:
            self.send_response(202)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        self._send_json(200, response)

    def _authorize_request(self) -> bool:
        parsed_path = urlsplit(self.path)
        if parsed_path.path != MCP_HTTP_PATH or parsed_path.query or parsed_path.fragment:
            self._send_http_error(404, "not_found", "MCP endpoint not found")
            return False
        origins = self.headers.get_all("Origin", failobj=[])
        if len(origins) > 1:
            self._send_http_error(403, "invalid_origin", "multiple Origin headers are forbidden")
            return False
        if origins:
            try:
                origin = canonical_origin(origins[0])
            except AdapterConfigurationError:
                self._send_http_error(403, "invalid_origin", "Origin is not allowed")
                return False
            if origin not in self.autopilot_server.allowed_origins:
                self._send_http_error(403, "invalid_origin", "Origin is not allowed")
                return False
        token = self.autopilot_server.bearer_token
        if token:
            authorization = self.headers.get_all("Authorization", failobj=[])
            expected = f"Bearer {token}"
            if len(authorization) != 1 or not secrets.compare_digest(authorization[0], expected):
                self.close_connection = True
                self.send_response(401)
                self.send_header("WWW-Authenticate", 'Bearer realm="redrhex-autopilot"')
                self.send_header("Connection", "close")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return False
        return True

    def _send_http_error(self, status: int, code: str, message: str) -> None:
        # Most transport errors are detected before the declared request body is
        # consumed. Closing prevents unread bytes from becoming a second request.
        self.close_connection = True
        self._send_json(
            status,
            {
                "jsonrpc": "2.0",
                "id": None,
                "error": {"code": -32000, "message": message, "data": {"code": code}},
            },
        )

    def _send_json(self, status: int, payload: Any) -> None:
        body = json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        ).encode("utf-8")
        if len(body) > MAX_RESPONSE_BYTES:
            body = json.dumps(
                MCPServer._rpc_error(None, -32603, "response exceeded the adapter limit"),
                separators=(",", ":"),
            ).encode("utf-8")
            status = 500
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        if self.close_connection:
            self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)


def create_http_server(
    host: str = DEFAULT_MCP_HOST,
    port: int = DEFAULT_MCP_PORT,
    *,
    gateway: ToolGateway | None = None,
    allowed_origins: frozenset[str] | None = None,
    bearer_token: str | None = None,
) -> AutopilotMCPHTTPServer:
    safe_host = validate_loopback_host(host)
    if isinstance(port, bool) or not isinstance(port, int) or not 0 <= port <= 65535:
        raise AdapterConfigurationError("MCP port must be an integer within 0..65535")
    # Port zero is accepted for tests; command-line use requires an explicit real port.
    origin_port = port or DEFAULT_MCP_PORT
    origins = allowed_origins or allowed_origins_from_env(safe_host, origin_port)
    token = mcp_http_token_from_env() if bearer_token is None else bearer_token
    if token and (len(token) < 16 or len(token) > 4096 or any(char.isspace() for char in token)):
        raise AdapterConfigurationError("MCP bearer token must be 16..4096 non-whitespace characters")
    server_class = AutopilotMCPHTTPServerV6 if ":" in safe_host else AutopilotMCPHTTPServer
    server = server_class(
        (safe_host, port),
        MCPHTTPHandler,
        mcp_server=MCPServer(gateway),
        allowed_origins=origins,
        bearer_token=token,
    )
    if port == 0 and allowed_origins is None:
        actual_port = int(server.server_address[1])
        server.allowed_origins = allowed_origins_from_env(safe_host, actual_port)
    return server


def _emit(payload: Any) -> None:
    sys.stdout.write(json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n")
    sys.stdout.flush()


def serve_stdio() -> int:
    server = MCPServer()
    while True:
        raw = sys.stdin.buffer.readline(MAX_INPUT_BYTES + 1)
        if not raw:
            return 0
        if len(raw) > MAX_INPUT_BYTES:
            while raw and not raw.endswith(b"\n"):
                raw = sys.stdin.buffer.readline(MAX_INPUT_BYTES + 1)
            _emit(MCPServer._rpc_error(None, -32700, "request exceeded the adapter limit"))
            continue
        try:
            incoming = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            _emit(MCPServer._rpc_error(None, -32700, "invalid JSON"))
            continue
        response = dispatch_payload(server, incoming)
        if response is not None:
            _emit(response)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="RedRHex Autopilot MCP adapter")
    parser.add_argument(
        "--http",
        action="store_true",
        help="serve stateless Streamable HTTP instead of stdio",
    )
    parser.add_argument("--host", default=DEFAULT_MCP_HOST, help="loopback bind host for --http")
    parser.add_argument("--port", default=DEFAULT_MCP_PORT, type=int, help="bind port for --http")
    options = parser.parse_args(argv)
    if not options.http:
        if options.host != DEFAULT_MCP_HOST or options.port != DEFAULT_MCP_PORT:
            parser.error("--host and --port require --http")
        return serve_stdio()
    if not 1 <= options.port <= 65535:
        parser.error("--port must be within 1..65535")
    try:
        server = create_http_server(options.host, options.port)
    except AdapterConfigurationError as exc:
        parser.error(str(exc))
    rendered_host = f"[{options.host}]" if ":" in options.host else options.host
    sys.stderr.write(f"RedRHex Autopilot MCP listening on http://{rendered_host}:{options.port}{MCP_HTTP_PATH}\n")
    try:
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
