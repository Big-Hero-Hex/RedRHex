"""Validate canonical RedRHex documentation."""

import dataclasses as _dataclasses
import datetime as _datetime
import os as _os
import pathlib as _pathlib
import re as _re
import urllib.parse as _urllib_parse

from . import schema as _schema

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
_TEMPLATE_DIRECTORY = _pathlib.Path("docs/governance/templates")
_SLUG_RE = _re.compile(r"[a-z][a-z0-9]*(?:-[a-z0-9]+)*\Z")
_DATE_NAME_RE = _re.compile(
    r"(?P<date>\d{4}-\d{2}-\d{2})-(?P<slug>[a-z][a-z0-9]*(?:-[a-z0-9]+)*)\Z"
)
_ADR_NAME_RE = _re.compile(r"adr-\d{4}-[a-z][a-z0-9]*(?:-[a-z0-9]+)*\Z")
_ISO_DATE_RE = _re.compile(r"\d{4}-\d{2}-\d{2}\Z")
_KEY_RE = _re.compile(r"[A-Za-z_][A-Za-z0-9_-]*\Z")
_BLOCK_SCALAR_RE = _re.compile(
    r"[|>](?:(?:[1-9][+-]?)|(?:[+-][1-9]?))?(?:\s+#.*)?\Z"
)
_HEADING_RE = _re.compile(r" {0,3}#{1,6}(?:[ \t]+|$)")
_ANCHOR_RE = _re.compile(r'<a id="([a-z][a-z0-9]*(?:-[a-z0-9]+)*)"></a>\Z')
_ANCHOR_LIKE_RE = _re.compile(
    r'''\s*<a\s+id\s*=\s*["']([^"']*)["']\s*>\s*</a>\s*\Z'''
)
_FENCE_OPEN_RE = _re.compile(r" {0,3}(`{3,}|~{3,}).*\Z")
_REFERENCE_DEFINITION_RE = _re.compile(r" {0,3}\[([^]]+)\]:[ \t]*(.*)\Z")


@_dataclasses.dataclass(frozen=True)
class Issue:
    path: _pathlib.Path
    code: str
    message: str


@_dataclasses.dataclass
class _ParsedDocument:
    path: _pathlib.Path
    relative_path: _pathlib.Path
    stem: str
    locale: str
    metadata: dict[str, str]
    body: str


@_dataclasses.dataclass
class _AnchorAnalysis:
    anchors: tuple[str, ...]
    eligible: bool


@_dataclasses.dataclass
class _ReferenceAnalysis:
    definitions: dict[str, str]
    used_labels: tuple[str, ...]
    missing_labels: tuple[str, ...]


def _discover_candidates(repo_root: _pathlib.Path) -> list[_pathlib.Path]:
    root = repo_root.resolve()
    candidates: list[_pathlib.Path] = []
    for current, directory_names, file_names in _os.walk(root):
        current_path = _pathlib.Path(current)
        kept: list[str] = []
        for name in sorted(directory_names):
            child = current_path / name
            relative = child.relative_to(root)
            if name in _EXCLUDED_DIRECTORY_NAMES or relative == _TEMPLATE_DIRECTORY:
                continue
            kept.append(name)
        directory_names[:] = kept
        for name in sorted(file_names):
            if not (name.endswith(".en.md") or name.endswith(".zh-TW.md")):
                continue
            path = current_path / name
            if path.is_file():
                candidates.append(path)
    return sorted(candidates, key=lambda path: path.relative_to(root).as_posix())


def _filename_parts(path: _pathlib.Path) -> tuple[str, str] | None:
    if path.name.endswith(".zh-TW.md"):
        stem = path.name[: -len(".zh-TW.md")]
        locale = "zh-TW"
    else:
        stem = path.name[: -len(".en.md")]
        locale = "en"
    if stem == "index" or _ADR_NAME_RE.fullmatch(stem):
        return stem, locale
    if stem.startswith("adr-"):
        return None
    date_match = _DATE_NAME_RE.fullmatch(stem)
    if date_match:
        try:
            _datetime.date.fromisoformat(date_match.group("date"))
        except ValueError:
            return None
        return stem, locale
    if stem[:1].isdigit():
        return None
    if _SLUG_RE.fullmatch(stem):
        return stem, locale
    return None


def _parse_document(
    path: _pathlib.Path,
    relative_path: _pathlib.Path,
    stem: str,
    locale: str,
) -> _ParsedDocument | str:
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines or lines[0] != "---":
        return "frontmatter must start on line 1"
    try:
        closing_index = lines.index("---", 1)
    except ValueError:
        return "frontmatter closing delimiter is missing"
    metadata: dict[str, str] = {}
    for line in lines[1:closing_index]:
        key, separator, raw_value = line.partition(":")
        value = raw_value.strip()
        if (
            not separator
            or key != key.strip()
            or not _KEY_RE.fullmatch(key)
            or not value
            or value.startswith(("[", "{"))
            or _BLOCK_SCALAR_RE.fullmatch(value)
        ):
            return "frontmatter supports one nonempty scalar key/value per line"
        if key in metadata:
            return f"duplicate frontmatter key: {key}"
        metadata[key] = value
    return _ParsedDocument(
        path=path,
        relative_path=relative_path,
        stem=stem,
        locale=locale,
        metadata=metadata,
        body="\n".join(lines[closing_index + 1 :]),
    )


def _metadata_field_error(document: _ParsedDocument) -> str | None:
    actual = set(document.metadata)
    if actual == _schema.REQUIRED_FIELDS:
        return None
    missing = sorted(_schema.REQUIRED_FIELDS - actual)
    extra = sorted(actual - _schema.REQUIRED_FIELDS)
    details = []
    if missing:
        details.append("missing: " + ", ".join(missing))
    if extra:
        details.append("unsupported: " + ", ".join(extra))
    return "; ".join(details)


def _metadata_value_error(document: _ParsedDocument) -> str | None:
    metadata = document.metadata
    if metadata["lang"] not in _schema.LANGUAGES:
        return "unsupported lang"
    if metadata["audience"] not in _schema.AUDIENCES:
        return "unsupported audience"
    if metadata["owner"] not in _schema.OWNERS:
        return "unsupported owner"
    allowed_statuses = _schema.STATUS_BY_TYPE.get(metadata["type"])
    if allowed_statuses is None:
        return "unsupported type"
    if metadata["status"] not in allowed_statuses:
        return "status is not allowed for type"
    if not _SLUG_RE.fullmatch(metadata["id"]):
        return "id must be lowercase kebab case"
    reviewed = metadata["last_reviewed"]
    if not _ISO_DATE_RE.fullmatch(reviewed):
        return "last_reviewed must be a real ISO date"
    try:
        parsed_date = _datetime.date.fromisoformat(reviewed)
    except ValueError:
        return "last_reviewed must be a real ISO date"
    if parsed_date.isoformat() != reviewed:
        return "last_reviewed must be a real ISO date"
    if metadata["lang"] != document.locale:
        return "filename locale must match lang"
    return None


def _location_error(document: _ParsedDocument) -> str | None:
    parts = document.relative_path.parts
    if not parts or parts[0] != "docs":
        return None
    metadata = document.metadata
    if len(parts) == 2:
        if (
            document.stem == "index"
            and metadata["audience"] == "shared"
            and metadata["type"] == "index"
        ):
            return None
        return "only docs/index may be a document directly under docs"
    section_rule = _schema.CENTRAL_LOCATIONS.get(parts[1])
    if section_rule is None:
        return "unknown central documentation section"
    required_audience, allowed_types = section_rule
    is_direct_portal = len(parts) == 3 and document.stem == "index"
    if is_direct_portal:
        if metadata["audience"] == required_audience and metadata["type"] == "index":
            return None
        return "section portal must use its section audience and index type"
    if document.stem == "index" or metadata["type"] == "index":
        return "index type is allowed only for a direct section portal"
    if metadata["audience"] != required_audience:
        return "audience is not allowed in this section"
    if metadata["type"] not in allowed_types:
        return "type is not allowed in this section"
    return None


def _outside_fence_lines(body: str) -> list[str]:
    visible: list[str] = []
    fence_character: str | None = None
    fence_length = 0
    for line in body.splitlines():
        if fence_character is None:
            opener = _FENCE_OPEN_RE.fullmatch(line)
            if opener:
                marker = opener.group(1)
                fence_character = marker[0]
                fence_length = len(marker)
                continue
            visible.append(line)
            continue
        leading_spaces = len(line) - len(line.lstrip(" "))
        candidate = line.lstrip(" ").rstrip(" \t")
        if (
            leading_spaces <= 3
            and len(candidate) >= fence_length
            and candidate
            and set(candidate) == {fence_character}
        ):
            fence_character = None
            fence_length = 0
    return visible


def _is_escaped(text: str, index: int) -> bool:
    backslashes = 0
    index -= 1
    while index >= 0 and text[index] == "\\":
        backslashes += 1
        index -= 1
    return backslashes % 2 == 1


def _closing_bracket(text: str, opener: int) -> int | None:
    depth = 1
    for index in range(opener + 1, len(text)):
        if _is_escaped(text, index):
            continue
        if text[index] == "[":
            depth += 1
        elif text[index] == "]":
            depth -= 1
            if depth == 0:
                return index
    return None


def _parenthesized_content(text: str, opener: int) -> tuple[str, int] | None:
    depth = 1
    quote: str | None = None
    index = opener + 1
    while index < len(text):
        character = text[index]
        if _is_escaped(text, index):
            index += 1
        elif quote is not None:
            if character == quote:
                quote = None
        elif character in {'"', "'"}:
            quote = character
        elif character == "(":
            depth += 1
        elif character == ")":
            depth -= 1
            if depth == 0:
                return text[opener + 1 : index], index
        index += 1
    return None


def _title_is_valid(text: str) -> bool:
    if not text:
        return True
    if len(text) < 2:
        return False
    if text[0] in {'"', "'"}:
        return text[-1] == text[0]
    if text[0] == "(":
        depth = 0
        for index, character in enumerate(text):
            if _is_escaped(text, index):
                continue
            if character == "(":
                depth += 1
            elif character == ")":
                depth -= 1
                if depth == 0 and index != len(text) - 1:
                    return False
                if depth < 0:
                    return False
        return depth == 0
    return False


def _destination_from_content(content: str) -> str | None:
    content = content.strip()
    if not content:
        return ""
    if content.startswith("<"):
        closing = None
        for index in range(1, len(content)):
            if content[index] == ">" and not _is_escaped(content, index):
                closing = index
                break
        if closing is None:
            return None
        destination = content[1:closing]
        title = content[closing + 1 :].strip()
        return destination if _title_is_valid(title) else None
    depth = 0
    split_at = len(content)
    for index, character in enumerate(content):
        if _is_escaped(content, index):
            continue
        if character == "(":
            depth += 1
        elif character == ")" and depth:
            depth -= 1
        elif character.isspace() and depth == 0:
            split_at = index
            break
    destination = content[:split_at]
    title = content[split_at:].strip()
    return destination if _title_is_valid(title) else None


def _inline_destinations(body: str) -> list[str]:
    destinations: list[str] = []
    for line in _outside_fence_lines(body):
        index = 0
        while index < len(line):
            if line[index] != "[" or _is_escaped(line, index):
                index += 1
                continue
            closing_bracket = _closing_bracket(line, index)
            if closing_bracket is None or closing_bracket + 1 >= len(line):
                index += 1
                continue
            if line[closing_bracket + 1] != "(":
                index = closing_bracket + 1
                continue
            parsed = _parenthesized_content(line, closing_bracket + 1)
            if parsed is None:
                index += 1
                continue
            content, closing_parenthesis = parsed
            destination = _destination_from_content(content)
            if destination is not None:
                destinations.append(destination)
            index = closing_parenthesis + 1
    return destinations


def _normalize_reference_label(label: str) -> str:
    return " ".join(label.split()).casefold()


def _analyze_references(body: str) -> _ReferenceAnalysis:
    lines = _outside_fence_lines(body)
    definitions: dict[str, str] = {}
    definition_lines: set[int] = set()
    for line_number, line in enumerate(lines):
        match = _REFERENCE_DEFINITION_RE.fullmatch(line)
        if match is None:
            continue
        label = _normalize_reference_label(match.group(1))
        destination = _destination_from_content(match.group(2))
        if label and destination is not None:
            definitions.setdefault(label, destination)
            definition_lines.add(line_number)

    used: list[str] = []
    missing: list[str] = []
    for line_number, line in enumerate(lines):
        if line_number in definition_lines:
            continue
        index = 0
        while index < len(line):
            if (
                line[index] != "["
                or _is_escaped(line, index)
                or (index > 0 and line[index - 1] == "]")
            ):
                index += 1
                continue
            first_close = _closing_bracket(line, index)
            if first_close is None:
                index += 1
                continue
            next_index = first_close + 1
            if next_index < len(line) and line[next_index] == "(":
                inline = _parenthesized_content(line, next_index)
                index = inline[1] + 1 if inline else first_close + 1
                continue
            first_label = _normalize_reference_label(line[index + 1 : first_close])
            if next_index < len(line) and line[next_index] == "[":
                second_close = _closing_bracket(line, next_index)
                if second_close is None:
                    index = first_close + 1
                    continue
                explicit_label = _normalize_reference_label(
                    line[next_index + 1 : second_close]
                )
                label = explicit_label or first_label
                if label in definitions:
                    used.append(label)
                else:
                    missing.append(label)
                index = second_close + 1
                continue
            if first_label in definitions:
                used.append(first_label)
            index = first_close + 1
    return _ReferenceAnalysis(definitions, tuple(used), tuple(missing))


def _explicit_anchors(path: _pathlib.Path) -> set[str]:
    text = path.read_text(encoding="utf-8")
    anchors = set()
    for line in _outside_fence_lines(text):
        match = _ANCHOR_RE.fullmatch(line)
        if match:
            anchors.add(match.group(1))
    return anchors


def _destination_issue(
    document: _ParsedDocument, repo_root: _pathlib.Path, destination: str
) -> Issue | None:
    raw_path = destination.split("#", 1)[0]
    if (
        raw_path.startswith("/")
        or raw_path.startswith("\\\\")
        or _re.match(r"[A-Za-z]:[\\/]", raw_path)
    ):
        return Issue(
            document.relative_path,
            "broken-link",
            f"local link must be relative: {destination}",
        )
    split = _urllib_parse.urlsplit(destination)
    if split.scheme:
        return None
    decoded_path = _urllib_parse.unquote(split.path)
    fragment = _urllib_parse.unquote(split.fragment)
    if (
        decoded_path.startswith("/")
        or decoded_path.startswith("\\\\")
        or _re.match(r"[A-Za-z]:[\\/]", decoded_path)
    ):
        return Issue(
            document.relative_path,
            "broken-link",
            f"local link must be relative: {destination}",
        )
    if decoded_path:
        target = (document.path.parent / decoded_path).resolve()
    else:
        target = document.path.resolve()
    resolved_root = repo_root.resolve()
    if not target.is_relative_to(resolved_root):
        return Issue(
            document.relative_path,
            "broken-link",
            f"local link escapes repository: {destination}",
        )
    if not target.is_file():
        return Issue(
            document.relative_path,
            "broken-link",
            f"local link target is not a file: {destination}",
        )
    if fragment and (not target.name.endswith(".md") or fragment not in _explicit_anchors(target)):
        return Issue(
            document.relative_path,
            "missing-link-anchor",
            f"explicit target anchor is missing: {destination}",
        )
    return None


def _validate_links(
    document: _ParsedDocument, repo_root: _pathlib.Path
) -> list[Issue]:
    issues: list[Issue] = []
    for destination in _inline_destinations(document.body):
        issue = _destination_issue(document, repo_root, destination)
        if issue:
            issues.append(issue)
    references = _analyze_references(document.body)
    for destination in references.definitions.values():
        issue = _destination_issue(document, repo_root, destination)
        if issue:
            issues.append(issue)
    for label in references.missing_labels:
        issues.append(
            Issue(
                document.relative_path,
                "broken-link",
                f"reference definition is missing: {label}",
            )
        )
    return issues


def _analyze_anchors(document: _ParsedDocument) -> tuple[_AnchorAnalysis, list[Issue]]:
    lines = _outside_fence_lines(document.body)
    issues: list[Issue] = []
    anchors: list[str] = []
    seen: set[str] = set()
    anchor_like = [bool(_ANCHOR_LIKE_RE.fullmatch(line)) for line in lines]
    for index, line in enumerate(lines):
        if anchor_like[index]:
            match = _ANCHOR_RE.fullmatch(line)
            if match is None:
                issues.append(
                    Issue(
                        document.relative_path,
                        "heading-anchor",
                        "explicit anchor is not lowercase kebab case",
                    )
                )
            else:
                anchor = match.group(1)
                anchors.append(anchor)
                if anchor in seen:
                    issues.append(
                        Issue(
                            document.relative_path,
                            "duplicate-anchor",
                            f"explicit anchor is duplicated: {anchor}",
                        )
                    )
                seen.add(anchor)
                if index + 1 >= len(lines) or not _HEADING_RE.match(lines[index + 1]):
                    issues.append(
                        Issue(
                            document.relative_path,
                            "heading-anchor",
                            "explicit anchor must immediately precede an ATX heading",
                        )
                    )
        if _HEADING_RE.match(line) and (index == 0 or not anchor_like[index - 1]):
            issues.append(
                Issue(
                    document.relative_path,
                    "heading-anchor",
                    "ATX heading is missing an immediately preceding explicit anchor",
                )
            )
    eligible = not any(
        issue.code in {"heading-anchor", "duplicate-anchor"} for issue in issues
    )
    return _AnchorAnalysis(tuple(anchors), eligible), issues


def validate_repository(repo_root: _pathlib.Path) -> list[Issue]:
    """Return deterministic validation issues sorted by path, code, and message."""
    root = repo_root.resolve()
    issues = []
    physical: dict[
        tuple[_pathlib.Path, str], dict[str, _pathlib.Path]
    ] = {}
    metadata_valid: dict[
        tuple[_pathlib.Path, str], dict[str, _ParsedDocument]
    ] = {}
    anchor_analyses: dict[
        tuple[_pathlib.Path, str], dict[str, _AnchorAnalysis]
    ] = {}
    for path in _discover_candidates(root):
        parts = _filename_parts(path)
        if parts is None:
            issues.append(
                Issue(
                    path.relative_to(root),
                    "invalid-name",
                    "filename does not follow canonical naming rules",
                )
            )
            continue
        stem, locale = parts
        relative_path = path.relative_to(root)
        logical_key = (relative_path.parent, stem)
        physical.setdefault(logical_key, {})[locale] = relative_path
        parsed = _parse_document(path, path.relative_to(root), *parts)
        if isinstance(parsed, str):
            issues.append(Issue(path.relative_to(root), "frontmatter", parsed))
            continue
        anchor_analysis, anchor_issues = _analyze_anchors(parsed)
        anchor_analyses.setdefault(logical_key, {})[locale] = anchor_analysis
        issues.extend(anchor_issues)
        issues.extend(_validate_links(parsed, root))
        metadata_error = _metadata_field_error(parsed)
        if metadata_error is None:
            metadata_error = _metadata_value_error(parsed)
        if metadata_error:
            issues.append(
                Issue(parsed.relative_path, "invalid-metadata", metadata_error)
            )
            continue
        metadata_valid.setdefault(logical_key, {})[locale] = parsed
        location_error = _location_error(parsed)
        if location_error:
            issues.append(Issue(parsed.relative_path, "invalid-location", location_error))
    for locales in physical.values():
        if len(locales) == 1:
            path = next(iter(locales.values()))
            issues.append(
                Issue(path, "missing-pair", "canonical locale companion is missing")
            )
    for locales in metadata_valid.values():
        if set(locales) != {"en", "zh-TW"}:
            continue
        english = locales["en"]
        chinese = locales["zh-TW"]
        comparable_english = {
            key: value
            for key, value in english.metadata.items()
            if key not in {"title", "lang"}
        }
        comparable_chinese = {
            key: value
            for key, value in chinese.metadata.items()
            if key not in {"title", "lang"}
        }
        if comparable_english != comparable_chinese:
            representative = min(english.relative_path, chinese.relative_path)
            issues.append(
                Issue(
                    representative,
                    "pair-metadata",
                    "paired metadata differs outside title and lang",
                )
            )
    documents_by_id: dict[
        str, dict[tuple[_pathlib.Path, str], list[_ParsedDocument]]
    ] = {}
    for logical_key, locales in metadata_valid.items():
        for document in locales.values():
            documents_by_id.setdefault(document.metadata["id"], {}).setdefault(
                logical_key, []
            ).append(document)
    for document_id, documents_by_logical_path in documents_by_id.items():
        if len(documents_by_logical_path) < 2:
            continue
        representative = min(
            document.relative_path
            for documents in documents_by_logical_path.values()
            for document in documents
        )
        issues.append(
            Issue(
                representative,
                "duplicate-id",
                f"id is reused by multiple logical documents: {document_id}",
            )
        )
    for logical_key, locales in anchor_analyses.items():
        if set(locales) != {"en", "zh-TW"}:
            continue
        english = locales["en"]
        chinese = locales["zh-TW"]
        if not english.eligible or not chinese.eligible:
            continue
        if english.anchors != chinese.anchors:
            representative = min(physical[logical_key].values())
            issues.append(
                Issue(
                    representative,
                    "pair-anchors",
                    "paired documents have different ordered anchor sequences",
                )
            )
    return sorted(issues, key=lambda issue: (issue.path.as_posix(), issue.code, issue.message))
