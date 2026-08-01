import datetime
import re
from pathlib import Path


_SLUG = r"[a-z][a-z0-9]*(?:-[a-z0-9]+)*"
_NORMAL_NAME = re.compile(rf"^{_SLUG}$")
_ID = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
_DATED_NAME = re.compile(rf"^(\d{{4}})-(\d{{2}})-(\d{{2}})-({_SLUG})$")
_ADR_NAME = re.compile(rf"^adr-\d{{4}}-({_SLUG})$")
_FRONTMATTER_LINE = re.compile(r"^([a-z_]+):[ \t]*(.*?)$")
_BLOCK_SCALAR = re.compile(r"^[|>](?:(?:[+-][1-9]?)|(?:[1-9][+-]?))?$")
REQUIRED_FIELDS = {
    "id",
    "title",
    "lang",
    "audience",
    "type",
    "status",
    "owner",
    "last_reviewed",
}
LANGUAGES = {"en", "zh-TW"}
AUDIENCES = {"operator", "developer", "shared"}
OWNERS = {"project", "core", "training", "panel", "deployment", "sim2real", "reward-agent"}
KNOWLEDGE_TYPES = {
    "index",
    "tutorial",
    "how-to",
    "reference",
    "explanation",
    "safety",
    "troubleshooting",
}
STATUS_BY_TYPE = {
    **{document_type: {"draft", "active", "deprecated"} for document_type in KNOWLEDGE_TYPES},
    "decision": {"accepted", "superseded"},
    "design": {"proposed", "approved", "implemented", "rejected", "superseded"},
    "plan": {"draft", "active", "blocked", "completed", "cancelled"},
    "roadmap": {"active"},
    "release": {"published"},
    "experiment-summary": {"published"},
    "audit": {"published"},
}
CENTRAL_SECTIONS = {
    "operators": ("operator", KNOWLEDGE_TYPES - {"index"}),
    "developers": ("developer", KNOWLEDGE_TYPES - {"index"}),
    "reference": ("shared", {"reference"}),
    "decisions": ("developer", {"decision"}),
    "designs": ("developer", {"design"}),
    "plans": ("developer", {"plan"}),
    "roadmap": ("shared", {"roadmap"}),
    "releases": ("shared", {"release"}),
    "research": ("developer", {"experiment-summary", "audit", "explanation"}),
    "governance": ("developer", {"reference"}),
}


def is_candidate_name(name: str) -> bool:
    lower_name = name.lower()
    return lower_name.endswith(".en.md") or lower_name.endswith(".zh-tw.md")


def is_valid_document_name(path: Path) -> bool:
    if path.name.endswith(".zh-TW.md"):
        stem = path.name[: -len(".zh-TW.md")]
    elif path.name.endswith(".en.md"):
        stem = path.name[: -len(".en.md")]
    else:
        return False

    if stem == "index":
        return True
    if stem.startswith("adr-"):
        return _ADR_NAME.fullmatch(stem) is not None
    if stem[:1].isdigit():
        match = _DATED_NAME.fullmatch(stem)
        if match is None:
            return False
        try:
            datetime.date(*(int(part) for part in match.group(1, 2, 3)))
        except ValueError:
            return False
        return True
    return _NORMAL_NAME.fullmatch(stem) is not None


def parse_frontmatter(text: str):
    lines = text.splitlines()
    if not lines or lines[0] != "---":
        return None
    try:
        closing_index = lines.index("---", 1)
    except ValueError:
        return None

    metadata = {}
    for line in lines[1:closing_index]:
        match = _FRONTMATTER_LINE.fullmatch(line)
        if match is None:
            return None
        key, value = match.groups()
        value = value.strip()
        if not value or key in metadata:
            return None
        if value.startswith("[") or value.startswith("{"):
            return None
        if _BLOCK_SCALAR.fullmatch(value):
            return None
        metadata[key] = value
    return metadata, "\n".join(lines[closing_index + 1 :])


def has_exact_metadata_fields(metadata: dict[str, str]) -> bool:
    return set(metadata) == REQUIRED_FIELDS


def has_valid_enum_values(metadata: dict[str, str]) -> bool:
    document_type = metadata["type"]
    return (
        metadata["lang"] in LANGUAGES
        and metadata["audience"] in AUDIENCES
        and metadata["owner"] in OWNERS
        and document_type in STATUS_BY_TYPE
        and metadata["status"] in STATUS_BY_TYPE.get(document_type, set())
    )


def has_valid_identity(metadata: dict[str, str], path: Path) -> bool:
    if _ID.fullmatch(metadata["id"]) is None:
        return False
    reviewed = metadata["last_reviewed"]
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", reviewed) is None:
        return False
    try:
        datetime.date.fromisoformat(reviewed)
    except ValueError:
        return False
    filename_language = "zh-TW" if path.name.endswith(".zh-TW.md") else "en"
    return metadata["lang"] == filename_language


def has_valid_location(relative_path: Path, metadata: dict[str, str]) -> bool:
    parts = relative_path.parts
    if not parts or parts[0] != "docs":
        return True
    if len(parts) == 2:
        return (
            parts[1] in {"index.en.md", "index.zh-TW.md"}
            and metadata["audience"] == "shared"
            and metadata["type"] == "index"
        )
    section = CENTRAL_SECTIONS.get(parts[1])
    if section is None:
        return False
    expected_audience, allowed_types = section
    is_portal = len(parts) == 3 and parts[2] in {"index.en.md", "index.zh-TW.md"}
    if is_portal:
        return metadata["audience"] == expected_audience and metadata["type"] == "index"
    return metadata["audience"] == expected_audience and metadata["type"] in allowed_types
