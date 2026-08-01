from dataclasses import dataclass as _dataclass
from pathlib import Path as _Path
import re as _re
from urllib.parse import unquote as _unquote

from .schema import is_candidate_name as _is_candidate_name
from .schema import is_valid_document_name as _is_valid_document_name
from .schema import has_exact_metadata_fields as _has_exact_metadata_fields
from .schema import has_valid_enum_values as _has_valid_enum_values
from .schema import has_valid_identity as _has_valid_identity
from .schema import has_valid_location as _has_valid_location
from .schema import parse_frontmatter as _parse_frontmatter

__all__ = ["Issue", "validate_repository"]

_EXCLUDED_DIRECTORY_NAMES = {
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
_HEADING = _re.compile(r"^ {0,3}#{1,6}(?:[ \t]+|$)")
_EXPLICIT_ANCHOR = _re.compile(r'^ {0,3}<a id="([^"]+)"></a>[ \t]*$')
_VALID_ANCHOR_ID = _re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
_URI_SCHEME = _re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*:")
_REFERENCE_DEFINITION = _re.compile(r"^ {0,3}\[([^\]]+)\]:[ \t]*(\S+)")
_REFERENCE_USE = _re.compile(r"!?\[[^\]]*\]\[([^\]]+)\]")


@_dataclass(frozen=True)
class Issue:
    path: _Path
    code: str
    message: str


@_dataclass(frozen=True)
class _Document:
    path: _Path
    metadata: dict[str, str]
    body: str


def _companion_path(path: _Path) -> _Path:
    name = path.name
    if name.endswith(".en.md"):
        return path.with_name(f"{name[:-len('.en.md')]}.zh-TW.md")
    return path.with_name(f"{name[:-len('.zh-TW.md')]}.en.md")


def _logical_path(path: _Path) -> _Path:
    name = path.name
    suffix = ".zh-TW.md" if name.endswith(".zh-TW.md") else ".en.md"
    return path.with_name(name[: -len(suffix)])


def _fence_marker(line: str):
    stripped = line.lstrip(" ")
    if len(line) - len(stripped) > 3 or not stripped or stripped[0] not in {"`", "~"}:
        return None
    marker_character = stripped[0]
    marker_length = len(stripped) - len(stripped.lstrip(marker_character))
    if marker_length < 3:
        return None
    return marker_character, marker_length, stripped[marker_length:]


def _lines_outside_fences(body: str) -> list[str]:
    outside_lines = []
    open_fence = None
    for line in body.splitlines():
        marker = _fence_marker(line)
        if open_fence is None:
            if marker is None:
                outside_lines.append(line)
            else:
                open_fence = marker[:2]
                outside_lines.append("")
            continue
        if (
            marker is not None
            and marker[0] == open_fence[0]
            and marker[1] >= open_fence[1]
            and not marker[2].strip()
        ):
            open_fence = None
    return outside_lines


def _missing_heading_anchor_issues(document: _Document) -> list[Issue]:
    issues = []
    lines = _lines_outside_fences(document.body)
    for line in lines:
        anchor_match = _EXPLICIT_ANCHOR.fullmatch(line)
        if anchor_match is not None and _VALID_ANCHOR_ID.fullmatch(anchor_match.group(1)) is None:
            issues.append(
                Issue(document.path, "heading-anchor", "explicit anchor ID is invalid")
            )
    for index, line in enumerate(lines):
        if _HEADING.match(line) is None:
            continue
        anchor_match = None if index == 0 else _EXPLICIT_ANCHOR.fullmatch(lines[index - 1])
        if anchor_match is None:
            issues.append(
                Issue(document.path, "heading-anchor", "heading is missing an explicit anchor")
            )
    return issues


def _duplicate_anchor_issues(document: _Document) -> list[Issue]:
    seen = set()
    duplicates = set()
    for line in _lines_outside_fences(document.body):
        match = _EXPLICIT_ANCHOR.fullmatch(line)
        if match is None or _VALID_ANCHOR_ID.fullmatch(match.group(1)) is None:
            continue
        anchor_id = match.group(1)
        if anchor_id in seen:
            duplicates.add(anchor_id)
        seen.add(anchor_id)
    return [
        Issue(document.path, "duplicate-anchor", f"duplicate anchor: {anchor_id}")
        for anchor_id in sorted(duplicates)
    ]


def _anchor_sequence_if_valid(document: _Document):
    if _missing_heading_anchor_issues(document) or _duplicate_anchor_issues(document):
        return None
    lines = _lines_outside_fences(document.body)
    sequence = []
    for line in lines:
        match = _EXPLICIT_ANCHOR.fullmatch(line)
        if match is not None:
            sequence.append(match.group(1))
    return tuple(sequence)


def _inline_destinations(text: str):
    search_from = 0
    while True:
        link_start = text.find("](", search_from)
        if link_start < 0:
            return
        position = link_start + 2
        if position >= len(text):
            return
        if text[position] == "<":
            angle_end = text.find(">", position + 1)
            if angle_end < 0:
                search_from = position + 1
                continue
            destination = text[position + 1 : angle_end]
            position = angle_end + 1
            while position < len(text) and text[position].isspace():
                position += 1
            if position < len(text) and text[position] in {'"', "'"}:
                quote = text[position]
                title_end = text.find(quote, position + 1)
                if title_end < 0:
                    search_from = position + 1
                    continue
                position = title_end + 1
                while position < len(text) and text[position].isspace():
                    position += 1
            if position < len(text) and text[position] == ")":
                yield destination
                search_from = position + 1
            else:
                search_from = position
            continue
        destination_start = position
        depth = 0
        while position < len(text):
            character = text[position]
            if character == "(":
                depth += 1
            elif character == ")":
                if depth == 0:
                    yield text[destination_start:position]
                    search_from = position + 1
                    break
                depth -= 1
            elif character.isspace() and depth == 0:
                destination = text[destination_start:position]
                while position < len(text) and text[position].isspace():
                    position += 1
                if position >= len(text) or text[position] not in {'"', "'"}:
                    search_from = position
                    break
                quote = text[position]
                title_start = position + 1
                title_end = text.find(quote, title_start)
                if title_end < 0:
                    search_from = position + 1
                    break
                position = title_end + 1
                while position < len(text) and text[position].isspace():
                    position += 1
                if position < len(text) and text[position] == ")":
                    yield destination
                    search_from = position + 1
                else:
                    search_from = position
                break
            position += 1
        else:
            return


def _basic_destination_issue(repo_root: _Path, document: _Document, destination: str):
    if not destination:
        return None
    if (
        destination.startswith("/")
        or destination.startswith("\\\\")
        or _re.match(r"^[A-Za-z]:[\\/]", destination)
    ):
        return Issue(document.path, "broken-link", f"absolute link target is not allowed: {destination}")
    if _URI_SCHEME.match(destination):
        return None
    target_part, separator, fragment_part = destination.partition("#")
    target_text = _unquote(target_part)
    fragment = _unquote(fragment_part) if separator else ""
    resolved_root = repo_root.resolve()
    if target_text:
        target = (resolved_root / document.path.parent / target_text).resolve()
    else:
        target = (resolved_root / document.path).resolve()
    if not target.is_relative_to(resolved_root):
        return Issue(document.path, "broken-link", f"link target escapes repository: {destination}")
    if not target.is_file():
        return Issue(document.path, "broken-link", f"link target does not exist: {destination}")
    if fragment and target.suffix.lower() == ".md":
        anchor_ids = {
            match.group(1)
            for line in _lines_outside_fences(target.read_text(encoding="utf-8"))
            if (match := _EXPLICIT_ANCHOR.fullmatch(line)) is not None
        }
        if fragment not in anchor_ids:
            return Issue(
                document.path,
                "missing-link-anchor",
                f"link anchor does not exist: {destination}",
            )
    return None


def _basic_link_issues(repo_root: _Path, document: _Document) -> list[Issue]:
    issues = []
    lines = _lines_outside_fences(document.body)
    text = "\n".join(lines)
    for destination in _inline_destinations(text):
        issue = _basic_destination_issue(repo_root, document, destination)
        if issue is not None:
            issues.append(issue)

    definitions = {}
    for line in lines:
        match = _REFERENCE_DEFINITION.match(line)
        if match is not None:
            definitions[" ".join(match.group(1).lower().split())] = match.group(2)
    for match in _REFERENCE_USE.finditer(text):
        label = " ".join(match.group(1).lower().split())
        destination = definitions.get(label)
        if destination is None:
            issues.append(
                Issue(document.path, "broken-link", f"reference definition is missing: {label}")
            )
            continue
        issue = _basic_destination_issue(repo_root, document, destination)
        if issue is not None:
            issues.append(issue)
    return issues


def _discover_document_paths(repo_root: _Path):
    for path in repo_root.rglob("*.md"):
        relative_path = path.relative_to(repo_root)
        if not path.is_file():
            continue
        if any(part in _EXCLUDED_DIRECTORY_NAMES for part in relative_path.parts[:-1]):
            continue
        if relative_path.parts[:3] == ("docs", "governance", "templates"):
            continue
        if _is_candidate_name(path.name):
            yield path, relative_path


def validate_repository(repo_root: _Path) -> list[Issue]:
    """Return deterministic validation issues sorted by path, code, and message."""
    issues = []
    documents = []
    for path, relative_path in _discover_document_paths(repo_root):
        if not _is_valid_document_name(path):
            issues.append(
                Issue(relative_path, "invalid-name", "invalid documentation filename")
            )
            continue
        parsed = _parse_frontmatter(path.read_text(encoding="utf-8"))
        if parsed is None:
            issues.append(
                Issue(relative_path, "frontmatter", "invalid frontmatter")
            )
            continue
        metadata, body = parsed
        if not _has_exact_metadata_fields(metadata):
            issues.append(
                Issue(relative_path, "invalid-metadata", "metadata fields must match the schema")
            )
            continue
        if not _has_valid_enum_values(metadata):
            issues.append(
                Issue(relative_path, "invalid-metadata", "metadata contains an invalid value")
            )
            continue
        if not _has_valid_identity(metadata, path):
            issues.append(
                Issue(relative_path, "invalid-metadata", "metadata identity or date is invalid")
            )
            continue
        if not _has_valid_location(relative_path, metadata):
            issues.append(
                Issue(relative_path, "invalid-location", "metadata does not match document location")
            )
            continue
        documents.append(_Document(relative_path, metadata, body))

    documents_by_path = {document.path: document for document in documents}
    for document in documents:
        companion_path = _companion_path(document.path)
        if companion_path not in documents_by_path:
            issues.append(
                Issue(document.path, "missing-pair", "locale companion is missing")
            )
            continue
        if document.path.as_posix() > companion_path.as_posix():
            continue
        companion = documents_by_path[companion_path]
        comparable = {
            key: value for key, value in document.metadata.items() if key not in {"title", "lang"}
        }
        companion_comparable = {
            key: value for key, value in companion.metadata.items() if key not in {"title", "lang"}
        }
        if comparable != companion_comparable:
            issues.append(
                Issue(document.path, "pair-metadata", "locale pair metadata differs")
            )
        sequence = _anchor_sequence_if_valid(document)
        companion_sequence = _anchor_sequence_if_valid(companion)
        if sequence is not None and companion_sequence is not None and sequence != companion_sequence:
            issues.append(
                Issue(document.path, "pair-anchors", "locale pair anchor sequences differ")
            )

    logical_paths_by_id = {}
    for document in documents:
        logical_path = _logical_path(document.path)
        logical_paths_by_id.setdefault(document.metadata["id"], set()).add(logical_path)
    for document_id, logical_paths in logical_paths_by_id.items():
        if len(logical_paths) > 1:
            representative = min(
                document.path
                for document in documents
                if document.metadata["id"] == document_id
            )
            issues.append(
                Issue(representative, "duplicate-id", f"id is reused: {document_id}")
            )
    for document in documents:
        issues.extend(_missing_heading_anchor_issues(document))
        issues.extend(_duplicate_anchor_issues(document))
        issues.extend(_basic_link_issues(repo_root, document))
    return sorted(issues, key=lambda issue: (issue.path.as_posix(), issue.code, issue.message))
