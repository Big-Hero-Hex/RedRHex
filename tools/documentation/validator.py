from dataclasses import dataclass as _dataclass
from datetime import date as _date
import os as _os
from pathlib import Path as _Path
import re as _re
import urllib.parse as _urllib_parse

from . import schema as _schema

__all__ = ["Issue", "validate_repository"]


@_dataclass(frozen=True)
class Issue:
    path: _Path
    code: str
    message: str


@_dataclass(frozen=True)
class _Document:
    path: _Path
    stem: str
    locale: str
    metadata: dict[str, str]
    body_lines: tuple[str, ...]


_EXCLUDED = {".git", ".worktrees", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", ".tox", ".nox", "build", "dist", "site"}


def _discover_candidates(repo_root: _Path) -> list[_Path]:
    found = []
    for directory, directory_names, file_names in _os.walk(repo_root, topdown=True):
        current = _Path(directory)
        relative = current.relative_to(repo_root)
        if relative.parts[:3] == ("docs", "governance", "templates"):
            directory_names[:] = []
            continue
        directory_names[:] = sorted(name for name in directory_names if name not in _EXCLUDED)
        for name in sorted(file_names):
            if name.endswith((".en.md", ".zh-TW.md")):
                path = current / name
                if path.is_file():
                    found.append(path)
    return sorted(found, key=lambda path: path.relative_to(repo_root).as_posix())


_SLUG = r"[a-z][a-z0-9]*(?:-[a-z0-9]+)*"


def _filename_info(path: _Path) -> tuple[str, str] | None:
    for suffix, locale in ((".zh-TW.md", "zh-TW"), (".en.md", "en")):
        if path.name.endswith(suffix):
            stem = path.name[: -len(suffix)]
            break
    else:
        return None
    if stem.startswith("adr-"):
        valid = _re.fullmatch(rf"adr-\d{{4}}-{_SLUG}", stem) is not None
    elif stem and stem[0].isdigit():
        match = _re.fullmatch(rf"(\d{{4}})-(\d{{2}})-(\d{{2}})-({_SLUG})", stem)
        valid = match is not None
        if match:
            try:
                _date(*(int(match.group(index)) for index in (1, 2, 3)))
            except ValueError:
                valid = False
    else:
        valid = stem == "index" or _re.fullmatch(_SLUG, stem) is not None
    return (stem, locale) if valid else None


_BLOCK = _re.compile(r"[|>](?:(?:[1-9][+-]?)|(?:[+-][1-9]?)|[+-])?(?:\s+#.*)?")


def _parse(path: _Path, stem: str, locale: str) -> tuple[_Document | None, str | None]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as error:
        return None, f"cannot read document: {error}"
    if not lines or lines[0] != "---":
        return None, "frontmatter must start on line 1"
    try:
        closing = lines.index("---", 1)
    except ValueError:
        return None, "frontmatter closing delimiter is missing"
    metadata = {}
    for line in lines[1:closing]:
        match = _re.fullmatch(r"([a-z][a-z0-9_]*):\s*(.*)", line)
        if not match:
            return None, "frontmatter requires one scalar key/value per line"
        key, value = match.groups()
        if key in metadata:
            return None, f"duplicate frontmatter key: {key}"
        if not value:
            return None, f"empty frontmatter value: {key}"
        if value.startswith(("[", "{", "- ")) or _BLOCK.fullmatch(value):
            return None, f"unsupported collection or multiline value: {key}"
        metadata[key] = value
    return _Document(path, stem, locale, metadata, tuple(lines[closing + 1 :])), None


def _shape_error(metadata: dict[str, str]) -> str | None:
    missing = sorted(_schema.REQUIRED_FIELDS - metadata.keys())
    extra = sorted(metadata.keys() - _schema.REQUIRED_FIELDS)
    if not missing and not extra:
        return None
    parts = []
    if missing:
        parts.append("missing: " + ", ".join(missing))
    if extra:
        parts.append("unsupported: " + ", ".join(extra))
    return "; ".join(parts)


def _values_error(metadata: dict[str, str]) -> str | None:
    if metadata["lang"] not in _schema.LANGUAGES:
        return f"invalid lang: {metadata['lang']}"
    if metadata["audience"] not in _schema.AUDIENCES:
        return f"invalid audience: {metadata['audience']}"
    if metadata["owner"] not in _schema.OWNERS:
        return f"invalid owner: {metadata['owner']}"
    allowed = _schema.STATUS_BY_TYPE.get(metadata["type"])
    if allowed is None:
        return f"invalid type: {metadata['type']}"
    if metadata["status"] not in allowed:
        return f"invalid status {metadata['status']} for type {metadata['type']}"
    return None


def _identity_error(document: _Document) -> str | None:
    if _re.fullmatch(_SLUG, document.metadata["id"]) is None:
        return f"invalid id: {document.metadata['id']}"
    reviewed = document.metadata["last_reviewed"]
    if _re.fullmatch(r"\d{4}-\d{2}-\d{2}", reviewed) is None:
        return f"invalid last_reviewed date: {reviewed}"
    try:
        _date.fromisoformat(reviewed)
    except ValueError:
        return f"invalid last_reviewed date: {reviewed}"
    if document.locale != document.metadata["lang"]:
        return f"filename locale {document.locale} does not match lang {document.metadata['lang']}"
    return None


def _location_error(relative: _Path, stem: str, metadata: dict[str, str]) -> str | None:
    parts = relative.parts
    if not parts or parts[0] != "docs":
        return None
    if len(parts) == 2:
        if stem == "index" and metadata["audience"] == "shared" and metadata["type"] == "index":
            return None
        return "direct docs files must be the shared index portal"
    section = _schema.CENTRAL_SECTIONS.get(parts[1])
    if section is None:
        return f"unknown central documentation section: {parts[1]}"
    audience, allowed_types = section
    if stem == "index":
        if len(parts) != 3:
            return "index portal must be directly under its central section"
        if metadata["audience"] != audience or metadata["type"] != "index":
            return f"{parts[1]} portal requires audience {audience} and type index"
        return None
    if metadata["type"] == "index":
        return "type index is reserved for direct section portals"
    if metadata["audience"] != audience or metadata["type"] not in allowed_types:
        return f"invalid audience/type for docs/{parts[1]}"
    return None


_ANCHOR = _re.compile(rf'\s*<a id="({_SLUG})"></a>\s*')
_ANY_ANCHOR = _re.compile(r'\s*<a id="([^"]*)"></a>\s*')
_HEADING = _re.compile(r" {0,3}#{1,6}(?:\s+|$)")


_FENCE_OPEN = _re.compile(r"^ {0,3}(`{3,}|~{3,})(.*)$")
_FENCE_CLOSE = _re.compile(r"^ {0,3}([`~]+)[ \t]*$")


def _active_lines(lines: tuple[str, ...]) -> tuple[bool, ...]:
    active = []
    fence_character = None
    fence_length = 0
    for line in lines:
        if fence_character is not None:
            active.append(False)
            closing = _FENCE_CLOSE.fullmatch(line)
            if closing and set(closing.group(1)) == {fence_character} and len(closing.group(1)) >= fence_length:
                fence_character = None
                fence_length = 0
            continue
        opening = _FENCE_OPEN.fullmatch(line)
        if opening and not (opening.group(1)[0] == "`" and "`" in opening.group(2)):
            marker = opening.group(1)
            fence_character = marker[0]
            fence_length = len(marker)
            active.append(False)
            continue
        active.append(True)
    return tuple(active)


def _anchor_result(document: _Document) -> tuple[tuple[str, ...], frozenset[str]]:
    lines = document.body_lines
    active = _active_lines(lines)
    anchors = []
    bad_structure = False
    for index, line in enumerate(lines):
        if not active[index]:
            continue
        valid_anchor = _ANCHOR.fullmatch(line)
        any_anchor = _ANY_ANCHOR.fullmatch(line)
        if any_anchor:
            if not valid_anchor:
                bad_structure = True
            else:
                anchors.append(valid_anchor.group(1))
                if index + 1 >= len(lines) or not active[index + 1] or not _HEADING.match(lines[index + 1]):
                    bad_structure = True
        if _HEADING.match(line):
            if index == 0 or not active[index - 1] or _ANCHOR.fullmatch(lines[index - 1]) is None:
                bad_structure = True
    codes = set()
    if bad_structure:
        codes.add("heading-anchor")
    if len(anchors) != len(set(anchors)):
        codes.add("duplicate-anchor")
    return tuple(anchors), frozenset(codes)


def _escaped(value: str, index: int) -> bool:
    backslashes = 0
    index -= 1
    while index >= 0 and value[index] == "\\":
        backslashes += 1
        index -= 1
    return backslashes % 2 == 1


def _closing_bracket(line: str, opening: int) -> int | None:
    depth = 1
    for index in range(opening + 1, len(line)):
        if _escaped(line, index):
            continue
        if line[index] == "[":
            depth += 1
        elif line[index] == "]":
            depth -= 1
            if depth == 0:
                return index
    return None


def _inline_destination(line: str, opening: int) -> tuple[str, int] | None:
    position = opening + 1
    while position < len(line) and line[position].isspace():
        position += 1
    if position >= len(line):
        return None
    if line[position] == "<":
        end = position + 1
        while end < len(line) and (line[end] != ">" or _escaped(line, end)):
            end += 1
        if end >= len(line):
            return None
        destination = line[position + 1 : end]
        position = end + 1
    else:
        start = position
        nested = 0
        while position < len(line):
            character = line[position]
            if _escaped(line, position):
                position += 1
                continue
            if character == "(":
                nested += 1
            elif character == ")":
                if nested == 0:
                    return line[start:position], position + 1
                nested -= 1
            elif character.isspace() and nested == 0:
                break
            position += 1
        destination = line[start:position]
    while position < len(line) and line[position].isspace():
        position += 1
    if position < len(line) and line[position] == ")":
        return destination, position + 1
    if position >= len(line) or line[position] not in "\"'(":
        return None
    opener = line[position]
    closer = ")" if opener == "(" else opener
    position += 1
    title_depth = 1
    while position < len(line):
        character = line[position]
        if _escaped(line, position):
            position += 1
        elif opener == "(" and character == "(":
            title_depth += 1
        elif character == closer:
            title_depth -= 1
            if title_depth == 0:
                position += 1
                break
        position += 1
    else:
        return None
    while position < len(line) and line[position].isspace():
        position += 1
    if position >= len(line) or line[position] != ")":
        return None
    return destination, position + 1


def _inline_uses(lines: tuple[str, ...]) -> list[tuple[str, int, int, int]]:
    uses = []
    active = _active_lines(lines)
    for line_number, line in enumerate(lines):
        if not active[line_number]:
            continue
        position = 0
        while position < len(line):
            if line[position] != "[" or _escaped(line, position):
                position += 1
                continue
            closing = _closing_bracket(line, position)
            if closing is None or closing + 1 >= len(line) or line[closing + 1] != "(":
                position += 1
                continue
            parsed = _inline_destination(line, closing + 1)
            if parsed is None:
                position += 1
                continue
            destination, end = parsed
            uses.append((destination, line_number, position, end))
            position = end
    return uses


def _inline_destinations(lines: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(use[0] for use in _inline_uses(lines))


_REFERENCE_DEFINITION = _re.compile(r"^ {0,3}\[([^]]+)\]:[ \t]*(.+)$")


def _reference_label(value: str) -> str:
    return " ".join(value.split()).casefold()


def _reference_data(lines: tuple[str, ...]) -> tuple[tuple[str, ...], tuple[str, ...], int]:
    active = _active_lines(lines)
    definitions = {}
    definition_lines = set()
    for line_number, line in enumerate(lines):
        if not active[line_number]:
            continue
        match = _REFERENCE_DEFINITION.fullmatch(line)
        if not match:
            continue
        synthetic = "(" + match.group(2) + ")"
        parsed = _inline_destination(synthetic, 0)
        if parsed is None or parsed[1] != len(synthetic):
            continue
        label = _reference_label(match.group(1))
        if label not in definitions:
            definitions[label] = parsed[0]
        definition_lines.add(line_number)
    resolved = []
    missing = 0
    for line_number, line in enumerate(lines):
        if not active[line_number] or line_number in definition_lines:
            continue
        position = 0
        while position < len(line):
            if line[position] != "[" or _escaped(line, position):
                position += 1
                continue
            closing = _closing_bracket(line, position)
            if closing is None:
                position += 1
                continue
            if closing + 1 < len(line) and line[closing + 1] == "(":
                parsed = _inline_destination(line, closing + 1)
                position = parsed[1] if parsed else closing + 1
                continue
            first_label = line[position + 1 : closing]
            if closing + 1 < len(line) and line[closing + 1] == "[":
                second_close = _closing_bracket(line, closing + 1)
                if second_close is None:
                    position = closing + 1
                    continue
                second_label = line[closing + 2 : second_close]
                label = _reference_label(second_label or first_label)
                if label in definitions:
                    resolved.append(definitions[label])
                else:
                    missing += 1
                position = second_close + 1
                continue
            label = _reference_label(first_label)
            if label in definitions:
                resolved.append(definitions[label])
            position = closing + 1
    return tuple(definitions.values()), tuple(resolved), missing


def _explicit_anchors(path: _Path) -> frozenset[str]:
    try:
        lines = tuple(path.read_text(encoding="utf-8").splitlines())
    except (OSError, UnicodeError, ValueError):
        return frozenset()
    active = _active_lines(lines)
    return frozenset(
        match.group(1)
        for index, line in enumerate(lines)
        if active[index] and (match := _ANCHOR.fullmatch(line)) is not None
    )


def _target_issue(repo_root: _Path, document: _Document, destination: str) -> tuple[str, str] | None:
    decoded = _urllib_parse.unquote(destination)
    if decoded.startswith("/") or decoded.startswith("\\\\") or _re.match(r"^[A-Za-z]:[\\/]", decoded):
        return "broken-link", f"absolute local link: {destination}"
    if _re.match(r"^[A-Za-z][A-Za-z0-9+.-]*:", decoded):
        return None
    path_text, separator, fragment = decoded.partition("#")
    try:
        root_resolved = repo_root.resolve()
        target = document.path if not path_text else document.path.parent / path_text
        target = target.resolve()
        target.relative_to(root_resolved)
    except (OSError, RuntimeError, ValueError):
        return "broken-link", f"invalid or escaping local link: {destination}"
    try:
        is_file = target.is_file()
    except (OSError, ValueError):
        is_file = False
    if not is_file:
        return "broken-link", f"link target is not a file: {destination}"
    if separator and fragment not in _explicit_anchors(target):
        return "missing-link-anchor", f"missing explicit anchor {fragment}: {destination}"
    return None


def _link_issues(repo_root: _Path, document: _Document) -> list[Issue]:
    definitions, _resolved, missing_references = _reference_data(document.body_lines)
    destinations = set(_inline_destinations(document.body_lines)) | set(definitions)
    relative = document.path.relative_to(repo_root)
    issues = []
    for destination in sorted(destinations):
        result = _target_issue(repo_root, document, destination)
        if result:
            issues.append(Issue(relative, result[0], result[1]))
    if missing_references:
        issues.append(Issue(relative, "broken-link", f"missing reference definition ({missing_references} uses)"))
    return issues


def validate_repository(repo_root: _Path) -> list[Issue]:
    """Return deterministic validation issues sorted by path, code, and message."""
    issues = []
    physical = []
    metadata_valid = []
    parsed_documents = []
    for path in _discover_candidates(repo_root):
        filename = _filename_info(path)
        if filename is None:
            issues.append(Issue(path.relative_to(repo_root), "invalid-name", "invalid canonical filename"))
            continue
        physical.append((path, *filename))
    keys = {(path.relative_to(repo_root).parent.as_posix(), stem, locale) for path, stem, locale in physical}
    for path, stem, locale in physical:
        relative = path.relative_to(repo_root)
        companion = "zh-TW" if locale == "en" else "en"
        if (relative.parent.as_posix(), stem, companion) not in keys:
            issues.append(Issue(relative, "missing-pair", f"missing {companion} companion"))
    for path, stem, locale in physical:
        document, error = _parse(path, stem, locale)
        if error:
            issues.append(Issue(path.relative_to(repo_root), "frontmatter", error))
            continue
        parsed_documents.append(document)
        error = _shape_error(document.metadata)
        if error:
            issues.append(Issue(path.relative_to(repo_root), "invalid-metadata", error))
            continue
        error = _values_error(document.metadata)
        if error:
            issues.append(Issue(path.relative_to(repo_root), "invalid-metadata", error))
            continue
        error = _identity_error(document)
        if error:
            issues.append(Issue(path.relative_to(repo_root), "invalid-metadata", error))
            continue
        relative = path.relative_to(repo_root)
        metadata_valid.append(document)
        error = _location_error(relative, document.stem, document.metadata)
        if error:
            issues.append(Issue(relative, "invalid-location", error))
    metadata_pairs = {}
    for document in metadata_valid:
        relative = document.path.relative_to(repo_root)
        key = (relative.parent.as_posix(), document.stem)
        metadata_pairs.setdefault(key, {})[document.locale] = document
    for locales in metadata_pairs.values():
        if set(locales) != {"en", "zh-TW"}:
            continue
        english, chinese = locales["en"], locales["zh-TW"]
        left = {key: value for key, value in english.metadata.items() if key not in {"title", "lang"}}
        right = {key: value for key, value in chinese.metadata.items() if key not in {"title", "lang"}}
        if left != right:
            representative = min(english.path, chinese.path, key=lambda path: path.relative_to(repo_root).as_posix())
            issues.append(Issue(representative.relative_to(repo_root), "pair-metadata", "locale metadata differs"))
    id_groups = {}
    for document in metadata_valid:
        id_groups.setdefault(document.metadata["id"], []).append(document)
    for doc_id, documents in id_groups.items():
        logical_paths = {
            (document.path.relative_to(repo_root).parent.as_posix(), document.stem)
            for document in documents
        }
        if len(logical_paths) > 1:
            representative = min(documents, key=lambda document: document.path.relative_to(repo_root).as_posix())
            issues.append(Issue(representative.path.relative_to(repo_root), "duplicate-id", f"id reused across logical documents: {doc_id}"))
    anchor_results = {}
    for document in parsed_documents:
        anchors, codes = _anchor_result(document)
        anchor_results[document.path] = (anchors, codes)
        relative = document.path.relative_to(repo_root)
        for code in codes:
            message = "duplicate explicit anchor" if code == "duplicate-anchor" else "headings require immediate valid explicit anchors"
            issues.append(Issue(relative, code, message))
    parsed_pairs = {}
    for document in parsed_documents:
        relative = document.path.relative_to(repo_root)
        parsed_pairs.setdefault((relative.parent.as_posix(), document.stem), {})[document.locale] = document
    for locales in parsed_pairs.values():
        if set(locales) != {"en", "zh-TW"}:
            continue
        english, chinese = locales["en"], locales["zh-TW"]
        english_anchors, english_codes = anchor_results[english.path]
        chinese_anchors, chinese_codes = anchor_results[chinese.path]
        if not english_codes and not chinese_codes and english_anchors != chinese_anchors:
            representative = min(english.path, chinese.path, key=lambda path: path.relative_to(repo_root).as_posix())
            issues.append(Issue(representative.relative_to(repo_root), "pair-anchors", "locale anchor sequences differ"))
    for document in parsed_documents:
        issues.extend(_link_issues(repo_root, document))
    return sorted(issues, key=lambda issue: (issue.path.as_posix(), issue.code, issue.message))
