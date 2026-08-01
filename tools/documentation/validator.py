"""Validate canonical documentation maintained in a repository."""

from dataclasses import dataclass
from datetime import date
from pathlib import Path
import re
from urllib.parse import unquote

from .schema import CENTRAL_LOCATION_RULES, ENUM_FIELDS, REQUIRED_FIELDS, STATUS_BY_TYPE


__all__ = ["Issue", "validate_repository"]


_NORMAL_STEM = re.compile(r"^(?:index|[a-z0-9]+(?:-[a-z0-9]+)*)$")
_DATED_STEM = re.compile(r"^(\d{4}-\d{2}-\d{2})-[a-z0-9]+(?:-[a-z0-9]+)*$")
_ADR_STEM = re.compile(r"^adr-\d{4}-[a-z0-9]+(?:-[a-z0-9]+)*$")
_SCALAR_LINE = re.compile(r"^([a-z_]+): (.*)$")
_KEBAB = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
_ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_ANY_ANCHOR_LINE = re.compile(r'^<a id="([^"]*)"></a>$')
_ATX_HEADING = re.compile(r"^ {0,3}#{1,6}(?:[ \t]+|$)")
_FENCE_OPEN = re.compile(r"^ {0,3}(`{3,}|~{3,}).*$")
_INLINE_OPEN = re.compile(r"!?\[[^\]]*\]\(")
_EXPLICIT_SCHEME = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*:")
_REFERENCE_DEFINITION = re.compile(r"^ {0,3}\[([^\]]+)\]:[ \t]*(<[^>]+>|\S+)")
_REFERENCE_USE = re.compile(r"!?\[([^\]]*)\]\[([^\]]*)\]")
_SHORTCUT_REFERENCE = re.compile(r"!?\[([^\]]+)\]")
_EXCLUDED_DIRECTORIES = {
    ".git",
    ".worktrees",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".tox",
    ".nox",
    "build",
    "dist",
    "site",
}


@dataclass(frozen=True)
class Issue:
    path: Path
    code: str
    message: str


def _candidate_documents(repo_root: Path) -> list[Path]:
    return sorted(
        path
        for path in repo_root.rglob("*.md")
        if path.name.lower().endswith((".en.md", ".zh-tw.md"))
        and not _excluded(path.relative_to(repo_root))
    )


def _excluded(relative_path: Path) -> bool:
    parts = relative_path.parts
    return bool(_EXCLUDED_DIRECTORIES.intersection(parts)) or parts[:3] == (
        "docs",
        "governance",
        "templates",
    )


def _stem(path: Path) -> str:
    locale_suffix = ".zh-TW.md" if path.name.endswith(".zh-TW.md") else ".en.md"
    return path.name[: -len(locale_suffix)]


def _locale(path: Path) -> str:
    return "zh-TW" if path.name.endswith(".zh-TW.md") else "en"


def _companion(path: Path) -> Path:
    if path.name.endswith(".en.md"):
        return path.with_name(f"{_stem(path)}.zh-TW.md")
    return path.with_name(f"{_stem(path)}.en.md")


def _valid_stem(stem: str) -> bool:
    if re.match(r"^\d{4}", stem):
        match = _DATED_STEM.fullmatch(stem)
        if not match:
            return False
        try:
            date.fromisoformat(match.group(1))
        except ValueError:
            return False
        return True
    if stem.startswith("adr-"):
        return bool(_ADR_STEM.fullmatch(stem))
    return bool(_NORMAL_STEM.fullmatch(stem))


def _valid_iso_date(value: str) -> bool:
    if not _ISO_DATE.fullmatch(value):
        return False
    try:
        date.fromisoformat(value)
    except ValueError:
        return False
    return True


def _valid_location(relative_path: Path, metadata: dict[str, str]) -> bool:
    parts = relative_path.parts
    if parts[0] != "docs":
        return True
    if len(parts) == 2:
        return (
            _stem(relative_path) == "index"
            and metadata["type"] == "index"
            and metadata["audience"] == "shared"
        )
    rule = CENTRAL_LOCATION_RULES.get(parts[1])
    if rule is None:
        return False
    audience, allowed_types = rule
    if metadata["type"] == "index":
        return (
            len(parts) == 3
            and _stem(relative_path) == "index"
            and metadata["audience"] == audience
        )
    return metadata["audience"] == audience and metadata["type"] in allowed_types


def _frontmatter_lines(path: Path) -> list[str] | None:
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines or lines[0] != "---":
        return None
    try:
        closing = lines.index("---", 1)
    except ValueError:
        return None
    return lines[1:closing]


def _parse_frontmatter(path: Path) -> dict[str, str] | None:
    lines = _frontmatter_lines(path)
    if lines is None:
        return None
    metadata = {}
    for line in lines:
        match = _SCALAR_LINE.fullmatch(line)
        if not match:
            return None
        key, value = match.groups()
        stripped = value.strip()
        if (
            key in metadata
            or not stripped
            or stripped[:1] in {"[", "{"}
            or stripped in {"|", ">"}
        ):
            return None
        metadata[key] = value
    return metadata


def _link_issue(repo_root: Path, source: Path, destination: str) -> Issue | None:
    if _EXPLICIT_SCHEME.match(destination):
        return None
    raw_path, separator, raw_fragment = destination.partition("#")
    path_text = unquote(raw_path)
    if path_text.startswith("/"):
        return Issue(source.relative_to(repo_root), "broken-link", "absolute link target")
    target = source if not path_text else source.parent / path_text
    resolved_root = repo_root.resolve()
    resolved_target = target.resolve()
    if not resolved_target.is_relative_to(resolved_root):
        return Issue(source.relative_to(repo_root), "broken-link", "link escapes repository")
    if not resolved_target.exists():
        return Issue(source.relative_to(repo_root), "broken-link", "missing link target")
    if separator and raw_fragment:
        fragment = unquote(raw_fragment)
        if (
            not resolved_target.is_file()
            or resolved_target.suffix.lower() != ".md"
            or fragment not in _explicit_anchors(resolved_target)
        ):
            return Issue(
                source.relative_to(repo_root),
                "missing-link-anchor",
                "missing explicit target anchor",
            )
    return None


def _inline_destination(raw_destination: str) -> str:
    parts = raw_destination.strip().split(maxsplit=1)
    return parts[0] if parts else ""


def _inline_destinations(line: str) -> list[str]:
    destinations = []
    for opening in _INLINE_OPEN.finditer(line):
        start = opening.end()
        depth = 1
        index = start
        while index < len(line):
            character = line[index]
            if character == "\\":
                index += 2
                continue
            if character == "(":
                depth += 1
            elif character == ")":
                depth -= 1
                if depth == 0:
                    destinations.append(line[start:index])
                    break
            index += 1
    return destinations


def _reference_label(value: str) -> str:
    return " ".join(value.split()).casefold()


def _reference_destination(value: str) -> str:
    return value[1:-1] if value.startswith("<") and value.endswith(">") else value


def _visible_line_indices(lines: list[str], start: int = 0) -> list[int]:
    visible_indices = []
    fence_marker = None
    fence_length = 0
    for index in range(start, len(lines)):
        line = lines[index]
        if fence_marker is None:
            opening = _FENCE_OPEN.match(line)
            if opening:
                marker = opening.group(1)
                fence_marker = marker[0]
                fence_length = len(marker)
                continue
            visible_indices.append(index)
            continue
        closing_fence = re.fullmatch(
            rf" {{0,3}}{re.escape(fence_marker)}{{{fence_length},}}[ \t]*", line
        )
        if closing_fence:
            fence_marker = None
            fence_length = 0
    return visible_indices


def _explicit_anchors(path: Path) -> set[str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    anchors = set()
    for index in _visible_line_indices(lines):
        match = _ANY_ANCHOR_LINE.fullmatch(lines[index])
        if match:
            anchors.add(match.group(1))
    return anchors


def _markdown_structure(repo_root: Path, path: Path) -> tuple[list[Issue], list[str]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    closing = lines.index("---", 1)
    issues = []
    anchors = []
    seen_anchors = set()
    visible_indices = _visible_line_indices(lines, closing + 1)
    definitions = {}
    for index in visible_indices:
        definition = _REFERENCE_DEFINITION.match(lines[index])
        if definition:
            definitions[_reference_label(definition.group(1))] = _reference_destination(
                definition.group(2)
            )
    for index in visible_indices:
        anchor_match = _ANY_ANCHOR_LINE.fullmatch(lines[index])
        if anchor_match:
            anchor_id = anchor_match.group(1)
            if _KEBAB.fullmatch(anchor_id):
                anchors.append(anchor_id)
                if anchor_id in seen_anchors:
                    issues.append(
                        Issue(
                            path.relative_to(repo_root),
                            "duplicate-anchor",
                            "duplicate explicit anchor id",
                        )
                    )
                seen_anchors.add(anchor_id)
            else:
                issues.append(
                    Issue(path.relative_to(repo_root), "heading-anchor", "invalid anchor id")
                )
        if _ATX_HEADING.match(lines[index]):
            previous = lines[index - 1] if index else ""
            if not _ANY_ANCHOR_LINE.fullmatch(previous):
                issues.append(
                    Issue(
                        path.relative_to(repo_root),
                        "heading-anchor",
                        "heading lacks an immediately preceding explicit anchor",
                    )
                )
        for raw_destination in _inline_destinations(lines[index]):
            issue = _link_issue(repo_root, path, _inline_destination(raw_destination))
            if issue:
                issues.append(issue)
        reference_uses = list(_REFERENCE_USE.finditer(lines[index]))
        for reference in reference_uses:
            label = reference.group(2) or reference.group(1)
            destination = definitions.get(_reference_label(label))
            if destination is None:
                issues.append(
                    Issue(
                        path.relative_to(repo_root),
                        "broken-link",
                        "missing reference link definition",
                    )
                )
                continue
            issue = _link_issue(repo_root, path, destination)
            if issue:
                issues.append(issue)
        covered_spans = [match.span() for match in reference_uses]
        covered_spans.extend(match.span() for match in _INLINE_OPEN.finditer(lines[index]))
        definition = _REFERENCE_DEFINITION.match(lines[index])
        if definition:
            covered_spans.append(definition.span())
        for shortcut in _SHORTCUT_REFERENCE.finditer(lines[index]):
            start, end = shortcut.span()
            if any(start < covered_end and end > covered_start for covered_start, covered_end in covered_spans):
                continue
            destination = definitions.get(_reference_label(shortcut.group(1)))
            if destination is None:
                continue
            issue = _link_issue(repo_root, path, destination)
            if issue:
                issues.append(issue)
    return issues, anchors


def validate_repository(repo_root: Path) -> list[Issue]:
    """Return validation issues for canonical documents in *repo_root*."""
    issues = []
    valid_metadata = {}
    anchor_sequences = {}
    for path in _candidate_documents(repo_root):
        if not path.name.endswith((".en.md", ".zh-TW.md")):
            issues.append(
                Issue(path.relative_to(repo_root), "invalid-name", "invalid locale suffix")
            )
            continue
        if not _companion(path).is_file():
            issues.append(
                Issue(path.relative_to(repo_root), "missing-pair", "missing locale companion")
            )
        if not _valid_stem(_stem(path)):
            issues.append(
                Issue(path.relative_to(repo_root), "invalid-name", "invalid canonical filename")
            )
        metadata = _parse_frontmatter(path)
        if metadata is None:
            issues.append(
                Issue(path.relative_to(repo_root), "frontmatter", "invalid frontmatter")
            )
            continue
        markdown_issues, anchors = _markdown_structure(repo_root, path)
        issues.extend(markdown_issues)
        if not markdown_issues:
            anchor_sequences[path] = anchors
        if not REQUIRED_FIELDS <= metadata.keys():
            issues.append(
                Issue(path.relative_to(repo_root), "invalid-metadata", "missing required field")
            )
            continue
        elif metadata.keys() - REQUIRED_FIELDS:
            issues.append(
                Issue(path.relative_to(repo_root), "invalid-metadata", "unsupported metadata field")
            )
            continue
        if any(metadata[field] not in allowed for field, allowed in ENUM_FIELDS.items()):
            issues.append(
                Issue(path.relative_to(repo_root), "invalid-metadata", "invalid enum value")
            )
            continue
        if metadata["status"] not in STATUS_BY_TYPE[metadata["type"]]:
            issues.append(
                Issue(path.relative_to(repo_root), "invalid-metadata", "status does not match type")
            )
            continue
        if not _KEBAB.fullmatch(metadata["id"]):
            issues.append(
                Issue(path.relative_to(repo_root), "invalid-metadata", "invalid document id")
            )
            continue
        if not _valid_iso_date(metadata["last_reviewed"]):
            issues.append(
                Issue(path.relative_to(repo_root), "invalid-metadata", "invalid review date")
            )
            continue
        if metadata["lang"] != _locale(path):
            issues.append(
                Issue(path.relative_to(repo_root), "invalid-metadata", "language does not match filename")
            )
            continue
        valid_metadata[path] = metadata
        if not _valid_location(path.relative_to(repo_root), metadata):
            issues.append(
                Issue(path.relative_to(repo_root), "invalid-location", "metadata does not match location")
            )
    paired_fields = REQUIRED_FIELDS - {"title", "lang"}
    for path, anchors in anchor_sequences.items():
        if not path.name.endswith(".en.md") or _companion(path) not in anchor_sequences:
            continue
        if anchors != anchor_sequences[_companion(path)]:
            issues.append(
                Issue(path.relative_to(repo_root), "pair-anchors", "locale anchor sequences differ")
            )
    for path, metadata in valid_metadata.items():
        if not path.name.endswith(".en.md") or _companion(path) not in valid_metadata:
            continue
        companion_metadata = valid_metadata[_companion(path)]
        if any(metadata[field] != companion_metadata[field] for field in paired_fields):
            issues.append(
                Issue(path.relative_to(repo_root), "pair-metadata", "locale metadata differs")
            )
    documents_by_id = {}
    for path, metadata in valid_metadata.items():
        if not path.name.endswith(".en.md") or _companion(path) not in valid_metadata:
            continue
        logical_path = path.with_name(_stem(path))
        representatives = documents_by_id.setdefault(metadata["id"], {})
        representatives[logical_path] = path
    for representatives in documents_by_id.values():
        for logical_path in sorted(representatives)[1:]:
            representative = representatives[logical_path]
            issues.append(
                Issue(
                    representative.relative_to(repo_root),
                    "duplicate-id",
                    "document id reused by another logical pair",
                )
            )
    return sorted(issues, key=lambda issue: (issue.path, issue.code, issue.message))
