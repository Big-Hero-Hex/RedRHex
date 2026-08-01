"""Repository-wide documentation validation."""

from dataclasses import dataclass
from datetime import date
from pathlib import Path
import os
import re
from urllib.parse import unquote, urlsplit

from .schema import ALLOWED_VALUES, REQUIRED_FIELDS, STATUS_BY_TYPE


@dataclass(frozen=True)
class Issue:
    path: Path
    code: str
    message: str


@dataclass
class Document:
    path: Path
    absolute_path: Path
    locale: str | None
    metadata: dict[str, str] | None
    body_lines: list[str]
    anchors: list[str]


_LOCALE_NAME = re.compile(r"^(.+)\.(en|zh-TW)\.md$")
_LOCALE_LIKE_NAME = re.compile(r"^.+\.(?:en|zh[-_]tw)\.md$", re.IGNORECASE)
_KEBAB = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
_ANCHOR = re.compile(r'^<a id="([^"]+)"></a>$')
_HEADING = re.compile(r"^ {0,3}#{1,6}(?:\s|$)")
_LINK = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")
_SKIP_DIRECTORIES = {".git", ".worktrees", "__pycache__", ".cache", "build", "dist", "cache"}


def _issue(path: Path, code: str, message: str) -> Issue:
    return Issue(path, code, message)


def _discover(repo_root: Path) -> list[Path]:
    paths: list[Path] = []
    for directory, names, filenames in os.walk(repo_root):
        names[:] = [name for name in names if name not in _SKIP_DIRECTORIES]
        relative_directory = Path(directory).relative_to(repo_root)
        if relative_directory.parts[:3] == ("docs", "governance", "templates"):
            names[:] = []
            continue
        for filename in filenames:
            if filename.endswith((".en.md", ".zh-TW.md")) or _LOCALE_LIKE_NAME.match(filename):
                paths.append(Path(directory) / filename)
    return sorted(paths, key=lambda path: str(path.relative_to(repo_root)))


def _parse_frontmatter(lines: list[str]) -> tuple[dict[str, str] | None, list[str], str | None]:
    if not lines or lines[0] != "---":
        return None, lines, "frontmatter must start on line 1 with ---"
    try:
        close_index = lines.index("---", 1)
    except ValueError:
        return None, lines, "frontmatter must close with ---"
    metadata: dict[str, str] = {}
    for line in lines[1:close_index]:
        match = re.fullmatch(r"([a-z_]+): (.+)", line)
        if not match:
            return None, lines[close_index + 1 :], "frontmatter lines must be scalar key: value pairs"
        key, value = match.groups()
        if key in metadata:
            return None, lines[close_index + 1 :], f"duplicate frontmatter field: {key}"
        if key not in REQUIRED_FIELDS:
            return None, lines[close_index + 1 :], f"unsupported frontmatter field: {key}"
        if value != value.strip() or value.startswith(("[", "{", "|", ">")) or value.endswith(("]", "}")):
            return None, lines[close_index + 1 :], "frontmatter values must be nonempty scalars"
        metadata[key] = value
    return metadata, lines[close_index + 1 :], None


def _without_fences(lines: list[str]) -> list[tuple[int, str]]:
    result: list[tuple[int, str]] = []
    fenced = False
    for index, line in enumerate(lines):
        if re.match(r"^\s*(```|~~~)", line):
            fenced = not fenced
            continue
        if not fenced:
            result.append((index, line))
    return result


def _valid_stem(stem: str) -> bool:
    if stem == "index":
        return True
    if re.match(r"^\d{4}-\d{2}-\d{2}-", stem):
        date_part, slug = stem[:10], stem[11:]
        try:
            date.fromisoformat(date_part)
        except ValueError:
            return False
        return bool(_KEBAB.fullmatch(slug))
    if stem.startswith("adr-"):
        return bool(re.fullmatch(r"adr-\d{4}-[a-z0-9]+(?:-[a-z0-9]+)*", stem))
    return bool(_KEBAB.fullmatch(stem))


def _name_parts(path: Path) -> tuple[str, str] | None:
    match = _LOCALE_NAME.fullmatch(path.name)
    return match.groups() if match else None


def _validate_metadata(document: Document, issues: list[Issue]) -> None:
    assert document.metadata is not None
    metadata = document.metadata
    fields = set(metadata)
    required = set(REQUIRED_FIELDS)
    if fields != required:
        missing = sorted(required - fields)
        extra = sorted(fields - required)
        detail = ", ".join(([f"missing {item}" for item in missing] + [f"unsupported {item}" for item in extra]))
        issues.append(_issue(document.path, "invalid-metadata", detail))
    for field, choices in ALLOWED_VALUES.items():
        if field in metadata and metadata[field] not in choices:
            issues.append(_issue(document.path, "invalid-metadata", f"invalid {field}: {metadata[field]}"))
    if "id" in metadata and not _KEBAB.fullmatch(metadata["id"]):
        issues.append(_issue(document.path, "invalid-metadata", "id must be lowercase kebab case"))
    if "last_reviewed" in metadata:
        try:
            if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", metadata["last_reviewed"]):
                raise ValueError
            date.fromisoformat(metadata["last_reviewed"])
        except ValueError:
            issues.append(_issue(document.path, "invalid-metadata", "last_reviewed must be an ISO date"))
    if metadata.get("type") in STATUS_BY_TYPE and metadata.get("status") not in STATUS_BY_TYPE[metadata["type"]]:
        issues.append(_issue(document.path, "invalid-metadata", "status is not allowed for type"))
    if document.locale and metadata.get("lang") != document.locale:
        issues.append(_issue(document.path, "invalid-metadata", "filename locale must equal lang"))


_LOCATIONS = {
    "operators": ("operator", {"index", "tutorial", "how-to", "reference", "explanation", "safety", "troubleshooting"}),
    "developers": ("developer", {"index", "tutorial", "how-to", "reference", "explanation", "safety", "troubleshooting"}),
    "reference": ("shared", {"reference"}),
    "decisions": ("developer", {"decision"}),
    "designs": ("developer", {"design"}),
    "plans": ("developer", {"plan"}),
    "roadmap": ("shared", {"roadmap"}),
    "releases": ("shared", {"release"}),
    "research": ("developer", {"experiment-summary", "audit", "explanation"}),
    "governance": ("developer", {"reference"}),
}


def _validate_location(document: Document, issues: list[Issue]) -> None:
    assert document.metadata is not None
    parts = document.path.parts
    metadata = document.metadata
    if parts[:1] != ("docs",):
        return
    if len(parts) == 2 and document.path.stem.split(".")[0] == "index":
        expected_audience, expected_types = "shared", {"index"}
    elif len(parts) >= 3 and parts[1] in _LOCATIONS:
        if document.path.stem.split(".")[0] == "index":
            expected_audience, expected_types = _LOCATIONS[parts[1]][0], {"index"}
        else:
            expected_audience, expected_types = _LOCATIONS[parts[1]]
    else:
        return
    if metadata.get("audience") != expected_audience or metadata.get("type") not in expected_types:
        issues.append(_issue(document.path, "invalid-location", "metadata does not match its docs section"))


def _anchors_and_headings(document: Document, issues: list[Issue]) -> list[str]:
    visible = _without_fences(document.body_lines)
    by_index = dict(visible)
    anchors: list[str] = []
    seen: set[str] = set()
    for index, line in visible:
        match = _ANCHOR.fullmatch(line)
        if match:
            anchor = match.group(1)
            anchors.append(anchor)
            if not _KEBAB.fullmatch(anchor):
                issues.append(_issue(document.path, "heading-anchor", "anchor ids must be lowercase kebab case"))
            if anchor in seen:
                issues.append(_issue(document.path, "duplicate-anchor", f"duplicate anchor: {anchor}"))
            seen.add(anchor)
        if _HEADING.match(line):
            previous = by_index.get(index - 1)
            match = _ANCHOR.fullmatch(previous) if previous is not None else None
            if match is None or not _KEBAB.fullmatch(match.group(1)):
                issues.append(_issue(document.path, "heading-anchor", "every heading needs its preceding explicit anchor"))
    return anchors


def _logical_path(document: Document) -> Path:
    assert document.locale is not None
    return document.path.with_name(document.path.name[: -(len(document.locale) + 4)])


def _validate_link(document: Document, target: str, repo_root: Path, documents: dict[Path, Document], issues: list[Issue]) -> None:
    target = target.strip()
    if target.startswith("<") and target.endswith(">"):
        target = target[1:-1]
    if re.match(r"^[A-Za-z]:[\\/]", target) or target.startswith("/"):
        issues.append(_issue(document.path, "broken-link", f"absolute link: {target}"))
        return
    parsed = urlsplit(target)
    if parsed.scheme:
        return
    local_path = unquote(parsed.path)
    fragment = unquote(parsed.fragment)
    absolute_target = document.absolute_path if not local_path else (document.absolute_path.parent / local_path).resolve()
    try:
        relative_target = absolute_target.relative_to(repo_root)
    except ValueError:
        issues.append(_issue(document.path, "broken-link", f"link escapes repository: {target}"))
        return
    if not absolute_target.exists():
        issues.append(_issue(document.path, "broken-link", f"missing link target: {target}"))
        return
    if fragment:
        target_document = documents.get(relative_target)
        if target_document is not None:
            anchors = target_document.anchors
        elif absolute_target.suffix.lower() == ".md":
            try:
                target_lines = absolute_target.read_text(encoding="utf-8").splitlines()
            except UnicodeDecodeError:
                target_lines = []
            anchors = [match.group(1) for _, line in _without_fences(target_lines) if (match := _ANCHOR.fullmatch(line))]
        else:
            anchors = []
        if fragment not in anchors:
            issues.append(_issue(document.path, "missing-link-anchor", f"missing explicit anchor: {fragment}"))


def validate_repository(repo_root: Path) -> list[Issue]:
    """Return deterministic validation issues sorted by path, code, and message."""
    root = repo_root.resolve()
    issues: list[Issue] = []
    documents: list[Document] = []
    discovered_paths = _discover(root)
    canonical_paths = {
        absolute_path.relative_to(root)
        for absolute_path in discovered_paths
        if _name_parts(absolute_path.relative_to(root)) is not None
    }
    for absolute_path in discovered_paths:
        path = absolute_path.relative_to(root)
        name_parts = _name_parts(path)
        locale = name_parts[1] if name_parts else None
        if name_parts is None or not _valid_stem(name_parts[0]):
            issues.append(_issue(path, "invalid-name", "filename must use a supported canonical pattern"))
        try:
            lines = absolute_path.read_text(encoding="utf-8").splitlines()
        except UnicodeDecodeError:
            issues.append(_issue(path, "frontmatter", "document must be UTF-8 text"))
            continue
        metadata, body_lines, error = _parse_frontmatter(lines)
        if error:
            issues.append(_issue(path, "frontmatter", error))
            continue
        document = Document(path, absolute_path.resolve(), locale, metadata, body_lines, [])
        documents.append(document)
        _validate_metadata(document, issues)
        _validate_location(document, issues)
        document.anchors = _anchors_and_headings(document, issues)

    exact_documents = [document for document in documents if document.locale is not None]
    by_path = {document.path: document for document in documents}
    for document in exact_documents:
        companion = document.path.with_name(
            document.path.name[: -(len(document.locale) + 3)] + ("zh-TW.md" if document.locale == "en" else "en.md")
        )
        if companion not in canonical_paths:
            issues.append(_issue(document.path, "missing-pair", "missing locale companion"))
            continue
        other = by_path.get(companion)
        if other is None:
            continue
        assert document.metadata is not None and other.metadata is not None
        if any(document.metadata.get(key) != other.metadata.get(key) for key in REQUIRED_FIELDS if key not in {"title", "lang"}):
            issues.append(_issue(document.path, "pair-metadata", "pair metadata differs outside title and lang"))
        if document.anchors != other.anchors:
            issues.append(_issue(document.path, "pair-anchors", "pair anchor sequences differ"))

    ids: dict[str, set[Path]] = {}
    for document in exact_documents:
        assert document.metadata is not None
        identifier = document.metadata.get("id")
        if identifier:
            ids.setdefault(identifier, set()).add(_logical_path(document))
    for identifier, logical_paths in ids.items():
        if len(logical_paths) > 1:
            for logical_path in logical_paths:
                issues.append(_issue(logical_path, "duplicate-id", f"id reused by multiple documents: {identifier}"))

    for document in documents:
        for _, line in _without_fences(document.body_lines):
            for target in _LINK.findall(line):
                _validate_link(document, target, root, by_path, issues)
    return sorted(issues, key=lambda issue: (str(issue.path), issue.code, issue.message))


def document_count(repo_root: Path) -> int:
    """Return the number of exact canonical-document candidates."""
    root = repo_root.resolve()
    return sum(1 for path in _discover(root) if _name_parts(path.relative_to(root)))
