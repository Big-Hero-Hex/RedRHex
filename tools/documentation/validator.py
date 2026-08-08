"""Validate canonical repository documentation."""

from __future__ import annotations

import dataclasses as _dataclasses
import datetime as _datetime
import os as _os
import re as _re
import urllib.parse as _urllib_parse
from pathlib import Path as _Path

from .schema import (
    _ParsedDocument,
    _has_exact_fields,
    _has_valid_enums,
    _metadata_identity_error,
    _parse_frontmatter,
)

__all__ = ["Issue", "validate_repository"]


@_dataclasses.dataclass(frozen=True)
class Issue:
    path: _Path
    code: str
    message: str


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


def _split_locale(filename: str) -> tuple[str, str] | None:
    for suffix, locale in ((".zh-TW.md", "zh-TW"), (".en.md", "en")):
        if filename.endswith(suffix):
            return filename[: -len(suffix)], locale
    return None


def _discover_candidates(repo_root: _Path) -> list[_Path]:
    candidates: list[_Path] = []
    for root_string, directory_names, file_names in _os.walk(repo_root):
        root = _Path(root_string)
        relative_root = root.relative_to(repo_root)
        directory_names[:] = sorted(
            name
            for name in directory_names
            if name not in _EXCLUDED_DIRECTORY_NAMES
            and not (
                relative_root == _Path("docs/governance") and name == "templates"
            )
        )
        for filename in sorted(file_names):
            path = root / filename
            if _split_locale(filename) is None:
                continue
            try:
                if not path.is_file():
                    continue
            except OSError:
                continue
            candidates.append(path)
    return sorted(candidates, key=lambda path: path.relative_to(repo_root).as_posix())


_SLUG = r"[a-z][a-z0-9]*(?:-[a-z0-9]+)*"
_NORMAL_NAME = _re.compile(rf"{_SLUG}\Z")
_DATE_NAME = _re.compile(rf"(\d{{4}}-\d{{2}}-\d{{2}})-({_SLUG})\Z")
_ADR_NAME = _re.compile(rf"adr-\d{{4}}-{_SLUG}\Z")
_ANCHOR_ID = _re.compile(rf"{_SLUG}\Z")
_EXPLICIT_ANCHOR = _re.compile(r' {0,3}<a id="([^"]*)"></a>[ \t]*\Z')
_ATX_HEADING = _re.compile(r" {0,3}#{1,6}(?:[ \t]+|$)")
_FENCE_OPEN = _re.compile(r" {0,3}(`{3,}|~{3,}).*\Z")
_REFERENCE_DEFINITION = _re.compile(r" {0,3}\[([^]]+)]\:[ \t]*(.*)\Z")
_EXPLICIT_SCHEME = _re.compile(r"[A-Za-z][A-Za-z0-9+.-]*:")
_WINDOWS_DRIVE_ROOT = _re.compile(r"[A-Za-z]:[\\/]")
_CENTRAL_SECTIONS = {
    "operators": (
        "operator",
        frozenset({"tutorial", "how-to", "reference", "explanation", "safety", "troubleshooting"}),
    ),
    "developers": (
        "developer",
        frozenset({"tutorial", "how-to", "reference", "explanation", "safety", "troubleshooting"}),
    ),
    "reference": ("shared", frozenset({"reference"})),
    "decisions": ("developer", frozenset({"decision"})),
    "designs": ("developer", frozenset({"design"})),
    "plans": ("developer", frozenset({"plan"})),
    "roadmap": ("shared", frozenset({"roadmap"})),
    "releases": ("shared", frozenset({"release"})),
    "research": (
        "developer",
        frozenset({"experiment-summary", "audit", "explanation"}),
    ),
    "governance": ("developer", frozenset({"reference"})),
}


def _valid_stem(stem: str) -> bool:
    if stem == "index":
        return True
    if stem.startswith("adr-"):
        return _ADR_NAME.fullmatch(stem) is not None
    if stem and stem[0].isdigit():
        match = _DATE_NAME.fullmatch(stem)
        if match is None:
            return False
        try:
            _datetime.date.fromisoformat(match.group(1))
        except ValueError:
            return False
        return True
    return _NORMAL_NAME.fullmatch(stem) is not None


def _valid_location(
    relative_path: _Path, stem: str, metadata: dict[str, str]
) -> bool:
    parts = relative_path.parts
    if not parts or parts[0] != "docs":
        return True
    inside_docs = parts[1:]
    if len(inside_docs) == 1:
        return (
            stem == "index"
            and metadata["audience"] == "shared"
            and metadata["type"] == "index"
        )
    section = _CENTRAL_SECTIONS.get(inside_docs[0])
    if section is None:
        return False
    expected_audience, allowed_types = section
    if len(inside_docs) == 2 and stem == "index":
        return (
            metadata["audience"] == expected_audience
            and metadata["type"] == "index"
        )
    if stem == "index" or metadata["type"] == "index":
        return False
    return (
        metadata["audience"] == expected_audience
        and metadata["type"] in allowed_types
    )


def _analyze_anchors(
    body_lines: tuple[str, ...], relative_path: _Path
) -> tuple[tuple[str, ...], bool, list[Issue]]:
    outside_fence = _outside_fence_mask(body_lines)
    anchors: list[str] = []
    seen: set[str] = set()
    issues: list[Issue] = []
    eligible = True
    for index, line in enumerate(body_lines):
        if not outside_fence[index]:
            continue
        anchor_match = _EXPLICIT_ANCHOR.fullmatch(line)
        if anchor_match is not None:
            anchor = anchor_match.group(1)
            followed_by_heading = (
                index + 1 < len(body_lines)
                and outside_fence[index + 1]
                and _ATX_HEADING.match(body_lines[index + 1]) is not None
            )
            if _ANCHOR_ID.fullmatch(anchor) is None:
                issues.append(
                    Issue(
                        relative_path,
                        "heading-anchor",
                        f"invalid explicit anchor: {anchor}",
                    )
                )
                eligible = False
            elif not followed_by_heading:
                issues.append(
                    Issue(
                        relative_path,
                        "heading-anchor",
                        f"explicit anchor is not immediately followed by a heading: {anchor}",
                    )
                )
                eligible = False
            else:
                anchors.append(anchor)
                if anchor in seen:
                    issues.append(
                        Issue(
                            relative_path,
                            "duplicate-anchor",
                            f"duplicate explicit anchor: {anchor}",
                        )
                    )
                    eligible = False
                seen.add(anchor)
        if _ATX_HEADING.match(line) is not None:
            previous_anchor = (
                _EXPLICIT_ANCHOR.fullmatch(body_lines[index - 1])
                if index > 0 and outside_fence[index - 1]
                else None
            )
            if previous_anchor is None:
                issues.append(
                    Issue(
                        relative_path,
                        "heading-anchor",
                        "heading lacks preceding explicit anchor",
                    )
                )
                eligible = False
    return tuple(anchors), eligible, issues


def _outside_fence_mask(body_lines: tuple[str, ...]) -> tuple[bool, ...]:
    outside: list[bool] = []
    marker_character: str | None = None
    marker_length = 0
    for line in body_lines:
        if marker_character is not None:
            outside.append(False)
            closing = _re.fullmatch(
                rf" {{0,3}}{_re.escape(marker_character)}{{{marker_length},}}[ \t]*",
                line,
            )
            if closing is not None:
                marker_character = None
                marker_length = 0
            continue
        opening = _FENCE_OPEN.fullmatch(line)
        if opening is None:
            outside.append(True)
            continue
        marker = opening.group(1)
        marker_character = marker[0]
        marker_length = len(marker)
        outside.append(False)
    return tuple(outside)


def _is_escaped(text: str, index: int) -> bool:
    backslashes = 0
    index -= 1
    while index >= 0 and text[index] == "\\":
        backslashes += 1
        index -= 1
    return backslashes % 2 == 1


def _matching_bracket(text: str, opening_index: int) -> int | None:
    depth = 0
    for index in range(opening_index, len(text)):
        if _is_escaped(text, index):
            continue
        if text[index] == "[":
            depth += 1
        elif text[index] == "]":
            depth -= 1
            if depth == 0:
                return index
    return None


def _title_and_close(text: str, index: int) -> int | None:
    while index < len(text) and text[index].isspace():
        index += 1
    if index < len(text) and text[index] == ")":
        return index + 1
    if index >= len(text):
        return None
    delimiter = text[index]
    if delimiter in {'"', "'"}:
        index += 1
        while index < len(text):
            if text[index] == delimiter and not _is_escaped(text, index):
                index += 1
                break
            index += 1
        else:
            return None
    elif delimiter == "(":
        index += 1
        while index < len(text):
            if text[index] == ")" and not _is_escaped(text, index):
                index += 1
                break
            index += 1
        else:
            return None
    else:
        return None
    while index < len(text) and text[index].isspace():
        index += 1
    if index >= len(text) or text[index] != ")":
        return None
    return index + 1


def _parse_inline_destination(
    text: str, opening_parenthesis: int
) -> tuple[str, int] | None:
    index = opening_parenthesis + 1
    while index < len(text) and text[index].isspace():
        index += 1
    if index < len(text) and text[index] == "<":
        start = index + 1
        index = start
        while index < len(text):
            if text[index] == ">" and not _is_escaped(text, index):
                destination = text[start:index]
                closing = _title_and_close(text, index + 1)
                return (destination, closing) if closing is not None else None
            index += 1
        return None

    start = index
    parenthesis_depth = 0
    while index < len(text):
        character = text[index]
        if _is_escaped(text, index):
            index += 1
        elif character == "(":
            parenthesis_depth += 1
        elif character == ")":
            if parenthesis_depth == 0:
                return text[start:index], index + 1
            parenthesis_depth -= 1
        elif character.isspace() and parenthesis_depth == 0:
            destination = text[start:index]
            closing = _title_and_close(text, index)
            return (destination, closing) if closing is not None else None
        index += 1
    return None


def _extract_inline_destinations(body_lines: tuple[str, ...]) -> list[str]:
    destinations: list[str] = []
    outside_fence = _outside_fence_mask(body_lines)
    for line_index, line in enumerate(body_lines):
        if not outside_fence[line_index]:
            continue
        index = 0
        while index < len(line):
            if line[index] != "[" or _is_escaped(line, index):
                index += 1
                continue
            closing_bracket = _matching_bracket(line, index)
            if closing_bracket is None or closing_bracket + 1 >= len(line):
                index += 1
                continue
            if line[closing_bracket + 1] != "(":
                index = closing_bracket + 1
                continue
            parsed = _parse_inline_destination(line, closing_bracket + 1)
            if parsed is None:
                index = closing_bracket + 1
                continue
            destination, end_index = parsed
            destinations.append(destination)
            index = end_index
    return destinations


def _normalize_reference_label(label: str) -> str:
    return " ".join(label.split()).casefold()


def _definition_title_is_valid(text: str) -> bool:
    text = text.strip()
    if not text:
        return True
    delimiter = text[0]
    closing = ")" if delimiter == "(" else delimiter
    if delimiter not in {'"', "'", "("}:
        return False
    index = 1
    while index < len(text):
        if text[index] == closing and not _is_escaped(text, index):
            return not text[index + 1 :].strip()
        index += 1
    return False


def _parse_definition_destination(text: str) -> str | None:
    text = text.lstrip()
    if not text:
        return None
    if text[0] == "<":
        index = 1
        while index < len(text):
            if text[index] == ">" and not _is_escaped(text, index):
                if _definition_title_is_valid(text[index + 1 :]):
                    return text[1:index]
                return None
            index += 1
        return None
    index = 0
    parenthesis_depth = 0
    while index < len(text):
        character = text[index]
        if _is_escaped(text, index):
            index += 1
        elif character == "(":
            parenthesis_depth += 1
        elif character == ")" and parenthesis_depth:
            parenthesis_depth -= 1
        elif character.isspace() and parenthesis_depth == 0:
            break
        index += 1
    destination = text[:index]
    if not destination or not _definition_title_is_valid(text[index:]):
        return None
    return destination


def _extract_reference_destinations(
    body_lines: tuple[str, ...]
) -> tuple[list[str], list[str]]:
    outside_fence = _outside_fence_mask(body_lines)
    definitions: dict[str, str] = {}
    definition_lines: set[int] = set()
    for line_index, line in enumerate(body_lines):
        if not outside_fence[line_index]:
            continue
        match = _REFERENCE_DEFINITION.fullmatch(line)
        if match is None:
            continue
        destination = _parse_definition_destination(match.group(2))
        if destination is None:
            continue
        label = _normalize_reference_label(match.group(1))
        definitions.setdefault(label, destination)
        definition_lines.add(line_index)

    missing_labels: list[str] = []
    for line_index, line in enumerate(body_lines):
        if not outside_fence[line_index] or line_index in definition_lines:
            continue
        index = 0
        while index < len(line):
            if line[index] != "[" or _is_escaped(line, index):
                index += 1
                continue
            closing_bracket = _matching_bracket(line, index)
            if closing_bracket is None:
                index += 1
                continue
            first_label = line[index + 1 : closing_bracket]
            following = closing_bracket + 1
            if following < len(line) and line[following] == "(":
                parsed_inline = _parse_inline_destination(line, following)
                index = parsed_inline[1] if parsed_inline is not None else following + 1
                continue
            if following < len(line) and line[following] == "[":
                second_close = _matching_bracket(line, following)
                if second_close is None:
                    index = following + 1
                    continue
                explicit_label = line[following + 1 : second_close]
                label = _normalize_reference_label(explicit_label or first_label)
                if label not in definitions:
                    missing_labels.append(label)
                index = second_close + 1
                continue
            label = _normalize_reference_label(first_label)
            if label in definitions:
                index = closing_bracket + 1
                continue
            index = closing_bracket + 1
    return list(definitions.values()), missing_labels


def _is_absolute_local_path(path: str) -> bool:
    return (
        path.startswith("/")
        or path.startswith("\\\\")
        or _WINDOWS_DRIVE_ROOT.match(path) is not None
    )


def _markdown_explicit_anchors(path: _Path) -> set[str] | None:
    try:
        lines = tuple(path.read_text(encoding="utf-8").splitlines())
    except (OSError, UnicodeError, ValueError):
        return None
    outside_fence = _outside_fence_mask(lines)
    anchors: set[str] = set()
    for index, line in enumerate(lines):
        if not outside_fence[index]:
            continue
        match = _EXPLICIT_ANCHOR.fullmatch(line)
        if match is not None:
            anchors.add(match.group(1))
    return anchors


def _validate_local_destination(
    repo_root: _Path, source_path: _Path, raw_destination: str
) -> Issue | None:
    raw_path, separator, raw_fragment = raw_destination.partition("#")
    relative_source = source_path.relative_to(repo_root)
    if _is_absolute_local_path(raw_path):
        return Issue(
            relative_source,
            "broken-link",
            f"invalid local link target: {raw_destination}",
        )
    if _EXPLICIT_SCHEME.match(raw_path) is not None:
        return None

    decoded_path = _urllib_parse.unquote(raw_path)
    decoded_fragment = _urllib_parse.unquote(raw_fragment) if separator else ""
    if _is_absolute_local_path(decoded_path):
        return Issue(
            relative_source,
            "broken-link",
            f"invalid local link target: {raw_destination}",
        )
    try:
        root_resolved = repo_root.resolve()
        target = (
            source_path.resolve()
            if not decoded_path
            else (source_path.parent / decoded_path).resolve()
        )
        target.relative_to(root_resolved)
        if not target.is_file():
            raise FileNotFoundError
    except (OSError, RuntimeError, ValueError):
        return Issue(
            relative_source,
            "broken-link",
            f"invalid local link target: {raw_destination}",
        )

    if separator:
        if target.suffix.lower() != ".md":
            return Issue(
                relative_source,
                "broken-link",
                f"invalid local link target: {raw_destination}",
            )
        anchors = _markdown_explicit_anchors(target)
        if anchors is None or decoded_fragment not in anchors:
            return Issue(
                relative_source,
                "missing-link-anchor",
                f"missing explicit anchor: {decoded_fragment}",
            )
    return None


def _link_issues(
    repo_root: _Path, source_path: _Path, body_lines: tuple[str, ...]
) -> list[Issue]:
    inline_destinations = _extract_inline_destinations(body_lines)
    reference_destinations, missing_labels = _extract_reference_destinations(
        body_lines
    )
    issues: list[Issue] = []
    for destination in inline_destinations + reference_destinations:
        issue = _validate_local_destination(repo_root, source_path, destination)
        if issue is not None:
            issues.append(issue)
    relative_source = source_path.relative_to(repo_root)
    issues.extend(
        Issue(
            relative_source,
            "broken-link",
            f"missing reference definition: {label}",
        )
        for label in missing_labels
    )
    return issues


def validate_repository(repo_root: _Path) -> list[Issue]:
    """Return deterministic validation issues sorted by path, code, and message."""
    issues: list[Issue] = []
    canonical_paths: list[_Path] = []
    parsed_documents: dict[_Path, _ParsedDocument] = {}
    metadata_valid_documents: dict[_Path, _ParsedDocument] = {}
    anchor_sequences: dict[_Path, tuple[str, ...]] = {}
    anchor_eligible_paths: set[_Path] = set()
    for path in _discover_candidates(repo_root):
        stem, locale = _split_locale(path.name) or ("", "")
        if not _valid_stem(stem):
            issues.append(
                Issue(
                    path.relative_to(repo_root),
                    "invalid-name",
                    "filename is not canonical",
                )
            )
            continue
        canonical_paths.append(path)
        try:
            parsed = _parse_frontmatter(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError):
            parsed = None
        if parsed is None:
            issues.append(
                Issue(
                    path.relative_to(repo_root),
                    "frontmatter",
                    "invalid frontmatter",
                )
            )
            continue
        parsed_documents[path] = parsed
        if not _has_exact_fields(parsed.metadata):
            issues.append(
                Issue(
                    path.relative_to(repo_root),
                    "invalid-metadata",
                    "metadata fields must match schema",
                )
            )
            continue
        if not _has_valid_enums(parsed.metadata):
            issues.append(
                Issue(
                    path.relative_to(repo_root),
                    "invalid-metadata",
                    "metadata value is invalid",
                )
            )
            continue
        identity_error = _metadata_identity_error(parsed.metadata, locale)
        if identity_error is not None:
            issues.append(
                Issue(
                    path.relative_to(repo_root),
                    "invalid-metadata",
                    identity_error,
                )
            )
            continue
        metadata_valid_documents[path] = parsed
        relative_path = path.relative_to(repo_root)
        if not _valid_location(relative_path, stem, parsed.metadata):
            issues.append(
                Issue(
                    relative_path,
                    "invalid-location",
                    "metadata does not match document location",
                )
            )
    for path, parsed in parsed_documents.items():
        relative_path = path.relative_to(repo_root)
        anchors, eligible, anchor_issues = _analyze_anchors(
            parsed.body_lines, relative_path
        )
        anchor_sequences[path] = anchors
        if eligible:
            anchor_eligible_paths.add(path)
        issues.extend(anchor_issues)
        issues.extend(_link_issues(repo_root, path, parsed.body_lines))
    canonical_path_set = set(canonical_paths)
    for path in canonical_paths:
        stem, locale = _split_locale(path.name) or ("", "")
        other_locale = "zh-TW" if locale == "en" else "en"
        companion = path.with_name(f"{stem}.{other_locale}.md")
        if companion not in canonical_path_set:
            issues.append(
                Issue(
                    path.relative_to(repo_root),
                    "missing-pair",
                    f"missing locale companion: {other_locale}",
                )
            )
    for path, parsed in metadata_valid_documents.items():
        stem, locale = _split_locale(path.name) or ("", "")
        if locale != "en":
            continue
        companion = path.with_name(f"{stem}.zh-TW.md")
        other = metadata_valid_documents.get(companion)
        if other is None:
            continue
        left = {
            key: value
            for key, value in parsed.metadata.items()
            if key not in {"title", "lang"}
        }
        right = {
            key: value
            for key, value in other.metadata.items()
            if key not in {"title", "lang"}
        }
        if left != right:
            issues.append(
                Issue(
                    path.relative_to(repo_root),
                    "pair-metadata",
                    "locale pair metadata differs",
                )
            )
    for path, anchors in anchor_sequences.items():
        stem, locale = _split_locale(path.name) or ("", "")
        if locale != "en" or path not in anchor_eligible_paths:
            continue
        companion = path.with_name(f"{stem}.zh-TW.md")
        if companion not in anchor_eligible_paths:
            continue
        if anchors != anchor_sequences.get(companion):
            issues.append(
                Issue(
                    path.relative_to(repo_root),
                    "pair-anchors",
                    "locale pair anchor sequences differ",
                )
            )
    paths_by_id: dict[str, list[_Path]] = {}
    for path, parsed in metadata_valid_documents.items():
        paths_by_id.setdefault(parsed.metadata["id"], []).append(path)
    for document_id, paths in paths_by_id.items():
        logical_documents = {
            (path.parent, (_split_locale(path.name) or ("", ""))[0])
            for path in paths
        }
        if len(logical_documents) <= 1:
            continue
        representative = min(
            paths,
            key=lambda candidate: candidate.relative_to(repo_root).as_posix(),
        )
        issues.append(
            Issue(
                representative.relative_to(repo_root),
                "duplicate-id",
                f"id reused by multiple logical documents: {document_id}",
            )
        )
    return sorted(
        set(issues),
        key=lambda issue: (issue.path.as_posix(), issue.code, issue.message),
    )
