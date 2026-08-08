"""Frontmatter parsing and metadata schema helpers."""

from __future__ import annotations

import dataclasses as _dataclasses
import datetime as _datetime
import re as _re


@_dataclasses.dataclass(frozen=True)
class _ParsedDocument:
    metadata: dict[str, str]
    body_lines: tuple[str, ...]


_FIELD_LINE = _re.compile(r"([a-z][a-z_]*): (.*)\Z")
_BLOCK_SCALAR = _re.compile(
    r"[|>](?:(?:[1-9][+-]?)|(?:[+-][1-9]?))?(?:[ \t]+#.*)?\Z"
)
_NON_SCALAR_PREFIX = _re.compile(r"(?:[!&*#]|[-?:][ \t])")
_REQUIRED_FIELDS = frozenset(
    {
        "id",
        "title",
        "lang",
        "audience",
        "type",
        "status",
        "owner",
        "last_reviewed",
    }
)
_AUDIENCES = frozenset({"operator", "developer", "shared"})
_OWNERS = frozenset(
    {"project", "core", "training", "panel", "deployment", "sim2real", "reward-agent"}
)
_KNOWLEDGE_STATUSES = frozenset({"draft", "active", "deprecated"})
_STATUS_BY_TYPE = {
    "index": _KNOWLEDGE_STATUSES,
    "tutorial": _KNOWLEDGE_STATUSES,
    "how-to": _KNOWLEDGE_STATUSES,
    "reference": _KNOWLEDGE_STATUSES,
    "explanation": _KNOWLEDGE_STATUSES,
    "safety": _KNOWLEDGE_STATUSES,
    "troubleshooting": _KNOWLEDGE_STATUSES,
    "decision": frozenset({"accepted", "superseded"}),
    "design": frozenset({"proposed", "approved", "implemented", "rejected", "superseded"}),
    "plan": frozenset({"draft", "active", "blocked", "completed", "cancelled"}),
    "roadmap": frozenset({"active"}),
    "release": frozenset({"published"}),
    "experiment-summary": frozenset({"published"}),
    "audit": frozenset({"published"}),
}
_ID = _re.compile(r"[a-z][a-z0-9]*(?:-[a-z0-9]+)*\Z")
_ISO_DATE = _re.compile(r"\d{4}-\d{2}-\d{2}\Z")


def _parse_frontmatter(text: str) -> _ParsedDocument | None:
    lines = text.splitlines()
    if not lines or lines[0] != "---":
        return None
    try:
        closing_index = lines.index("---", 1)
    except ValueError:
        return None

    metadata: dict[str, str] = {}
    for line in lines[1:closing_index]:
        match = _FIELD_LINE.fullmatch(line)
        if match is None:
            return None
        key, raw_value = match.groups()
        value = raw_value.strip()
        if not value or key in metadata:
            return None
        if (
            value.startswith(("[", "{"))
            or _BLOCK_SCALAR.fullmatch(value)
            or _NON_SCALAR_PREFIX.match(value)
        ):
            return None
        metadata[key] = value
    return _ParsedDocument(metadata, tuple(lines[closing_index + 1 :]))


def _has_exact_fields(metadata: dict[str, str]) -> bool:
    return metadata.keys() == _REQUIRED_FIELDS


def _has_valid_enums(metadata: dict[str, str]) -> bool:
    document_type = metadata["type"]
    return (
        metadata["lang"] in {"en", "zh-TW"}
        and metadata["audience"] in _AUDIENCES
        and metadata["owner"] in _OWNERS
        and document_type in _STATUS_BY_TYPE
        and metadata["status"] in _STATUS_BY_TYPE.get(document_type, ())
    )


def _metadata_identity_error(
    metadata: dict[str, str], filename_locale: str
) -> str | None:
    if _ID.fullmatch(metadata["id"]) is None:
        return "id must be lowercase kebab-case"
    reviewed = metadata["last_reviewed"]
    if _ISO_DATE.fullmatch(reviewed) is None:
        return "last_reviewed must be a real ISO date"
    try:
        _datetime.date.fromisoformat(reviewed)
    except ValueError:
        return "last_reviewed must be a real ISO date"
    if metadata["lang"] != filename_locale:
        return "filename locale does not match lang"
    return None
