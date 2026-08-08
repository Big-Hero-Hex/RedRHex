"""Validate the site source manifest and stage canonical documentation."""

from __future__ import annotations

import dataclasses
import json
import os
import posixpath
import re
import shutil
import stat
import urllib.parse
from pathlib import Path, PurePosixPath

from .errors import DocumentationOperationError
from .validator import _outside_fence_mask, _split_locale, _valid_stem


@dataclasses.dataclass(frozen=True)
class SiteSource:
    """One resolved source directory and relative staging destination."""

    source: Path
    destination: Path


def _relative_posix_path(raw: object) -> Path:
    if not isinstance(raw, str) or not raw or "\0" in raw or "\\" in raw:
        raise DocumentationOperationError("invalid documentation site manifest")
    path = PurePosixPath(raw)
    if (
        path.is_absolute()
        or any(part == ".." for part in path.parts)
        or re.match(r"^[A-Za-z]:", raw) is not None
    ):
        raise DocumentationOperationError("invalid documentation site manifest")
    return Path(*path.parts)


def load_site_sources(repo_root: Path) -> tuple[SiteSource, ...]:
    """Load and validate resolved sources from the checked-in manifest."""
    manifest_path = repo_root / "docs/site-manifest.json"
    try:
        value = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise DocumentationOperationError(
            "invalid documentation site manifest"
        ) from error
    if (
        not isinstance(value, dict)
        or set(value) != {"schema_version", "sources"}
        or type(value["schema_version"]) is not int
        or value["schema_version"] != 1
        or not isinstance(value["sources"], list)
        or not value["sources"]
    ):
        raise DocumentationOperationError("invalid documentation site manifest")

    try:
        root_resolved = repo_root.resolve(strict=True)
    except OSError as error:
        raise DocumentationOperationError(
            "invalid documentation site manifest"
        ) from error
    sources: list[SiteSource] = []
    entries_seen: set[tuple[Path, Path]] = set()
    destinations_seen: set[Path] = set()
    for entry in value["sources"]:
        if not isinstance(entry, dict) or set(entry) != {"source", "destination"}:
            raise DocumentationOperationError("invalid documentation site manifest")
        source_relative = _relative_posix_path(entry["source"])
        destination = _relative_posix_path(entry["destination"])
        identity = (source_relative, destination)
        if identity in entries_seen or destination in destinations_seen:
            raise DocumentationOperationError("invalid documentation site manifest")
        entries_seen.add(identity)
        destinations_seen.add(destination)
        source = repo_root / source_relative
        try:
            source_resolved = source.resolve(strict=True)
            source_resolved.relative_to(root_resolved)
            is_directory = source.is_dir()
        except (OSError, RuntimeError, ValueError) as error:
            raise DocumentationOperationError(
                "invalid documentation site manifest"
            ) from error
        if not is_directory:
            raise DocumentationOperationError("invalid documentation site manifest")
        sources.append(SiteSource(source, destination))
    return tuple(sources)


_STAGING_EXCLUDED_DIRECTORIES = {
    ".git",
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


def _canonical_copy_plan(
    repo_root: Path, sources: tuple[SiteSource, ...], output: Path
) -> list[tuple[Path, Path]]:
    def reject_traversal_error(error: OSError) -> None:
        raise DocumentationOperationError(
            "unsafe documentation site source"
        ) from error

    plan: list[tuple[Path, Path]] = []
    for site_source in sources:
        current = repo_root
        try:
            source_parts = site_source.source.relative_to(repo_root).parts
        except ValueError as error:
            raise DocumentationOperationError(
                "unsafe documentation site source"
            ) from error
        for part in source_parts:
            current /= part
            if current.is_symlink():
                raise DocumentationOperationError("unsafe documentation site source")
        for root_string, directory_names, file_names in os.walk(
            site_source.source, onerror=reject_traversal_error
        ):
            root = Path(root_string)
            if any((root / name).is_symlink() for name in directory_names):
                raise DocumentationOperationError("unsafe documentation site source")
            directory_names[:] = sorted(
                name
                for name in directory_names
                if name not in _STAGING_EXCLUDED_DIRECTORIES
            )
            for filename in sorted(file_names):
                relative = (root / filename).relative_to(site_source.source)
                split = _split_locale(filename)
                is_document = (
                    "templates" not in relative.parts
                    and split is not None
                    and _valid_stem(split[0])
                )
                is_asset = filename.endswith(".md.template") or (
                    filename == "migration-manifest.csv"
                )
                if not is_document and not is_asset:
                    continue
                source = root / filename
                try:
                    source_mode = source.lstat().st_mode
                except OSError as error:
                    raise DocumentationOperationError(
                        "unsafe documentation site source"
                    ) from error
                if not stat.S_ISREG(source_mode):
                    raise DocumentationOperationError(
                        "unsafe documentation site source"
                    )
                destination = output / site_source.destination / relative
                plan.append((source, destination))
    return sorted(plan, key=lambda item: item[1].as_posix())


_INLINE_SITE_LINK = re.compile(
    r"(?P<prefix>\]\(\s*<?)(?P<destination>[^\s)>]+)(?P<suffix>>?)"
)
_REFERENCE_SITE_LINK = re.compile(
    r"(?P<prefix>^\s*\[[^\]]+\]:\s*<?)(?P<destination>[^\s>]+)(?P<suffix>>?)"
)


def _rewrite_site_destination(
    raw_destination: str,
    source: Path,
    staged_source: Path,
    staged_paths: dict[Path, Path],
) -> str:
    raw_path, separator, fragment = raw_destination.partition("#")
    if (
        not raw_path
        or raw_path.startswith("/")
        or re.match(r"^[A-Za-z][A-Za-z0-9+.-]*:", raw_path) is not None
    ):
        return raw_destination
    try:
        target = (source.parent / urllib.parse.unquote(raw_path)).resolve()
    except (OSError, RuntimeError, ValueError):
        return raw_destination
    staged_target = staged_paths.get(target)
    if staged_target is None:
        return raw_destination
    split = _split_locale(staged_target.name)
    if split is not None:
        staged_target = staged_target.with_name(f"{split[0]}.md")
    relative = posixpath.relpath(
        staged_target.as_posix(), start=staged_source.parent.as_posix()
    )
    encoded = urllib.parse.quote(relative, safe="/@:+-._~")
    return f"{encoded}#{fragment}" if separator else encoded


def _rewrite_staged_markdown(
    source: Path,
    destination: Path,
    staged_paths: dict[Path, Path],
) -> None:
    text = source.read_text(encoding="utf-8")
    lines = text.splitlines(keepends=True)
    mask_lines = tuple(line.rstrip("\r\n") for line in lines)
    outside_fence = _outside_fence_mask(mask_lines)

    def rewrite_match(match: re.Match[str]) -> str:
        rewritten = _rewrite_site_destination(
            match.group("destination"), source, destination, staged_paths
        )
        return f'{match.group("prefix")}{rewritten}{match.group("suffix")}'

    rewritten_lines: list[str] = []
    for index, line in enumerate(lines):
        if outside_fence[index]:
            line = _INLINE_SITE_LINK.sub(rewrite_match, line)
            line = _REFERENCE_SITE_LINK.sub(rewrite_match, line)
        rewritten_lines.append(line)
    destination.write_text("".join(rewritten_lines), encoding="utf-8")
    shutil.copystat(source, destination)


def _is_within(path: Path, directory: Path) -> bool:
    try:
        path.relative_to(directory)
    except ValueError:
        return False
    return True


def _safe_output_path(
    repo_root: Path, sources: tuple[SiteSource, ...], output: Path
) -> Path:
    try:
        resolved = output.resolve(strict=False)
        repo_resolved = repo_root.resolve(strict=True)
        source_resolved = [source.source.resolve(strict=True) for source in sources]
        output_is_symlink = output.is_symlink()
        existing_ancestor = resolved
        while True:
            try:
                ancestor_mode = existing_ancestor.stat().st_mode
            except FileNotFoundError:
                parent = existing_ancestor.parent
                if parent == existing_ancestor:
                    raise
                existing_ancestor = parent
                continue
            break
    except (OSError, RuntimeError, ValueError) as error:
        raise DocumentationOperationError(
            "unsafe or nonempty site staging output"
        ) from error
    if (
        output_is_symlink
        or not stat.S_ISDIR(ancestor_mode)
        or _is_within(resolved, repo_resolved)
        or any(
            _is_within(resolved, source) for source in source_resolved
        )
    ):
        raise DocumentationOperationError("unsafe or nonempty site staging output")
    if output.exists():
        try:
            empty_directory = output.is_dir() and next(output.iterdir(), None) is None
        except (OSError, ValueError) as error:
            raise DocumentationOperationError(
                "unsafe or nonempty site staging output"
            ) from error
        if not empty_directory:
            raise DocumentationOperationError(
                "unsafe or nonempty site staging output"
            )
    return resolved


def stage_site(repo_root: Path, output: Path) -> int:
    """Stage canonical site sources and return the copied document count."""
    sources = load_site_sources(repo_root)
    safe_output = _safe_output_path(repo_root, sources, output)
    plan = _canonical_copy_plan(repo_root, sources, safe_output)
    destinations = [destination for _, destination in plan]
    for index, destination in enumerate(destinations):
        for other in destinations[index + 1 :]:
            if (
                destination == other
                or destination in other.parents
                or other in destination.parents
            ):
                raise DocumentationOperationError(
                    "site staging plan has destination collisions"
                )
    try:
        safe_output.mkdir(parents=True, exist_ok=True)
    except (OSError, ValueError) as error:
        raise DocumentationOperationError(
            "unsafe or nonempty site staging output"
        ) from error
    try:
        staged_paths = {
            source.resolve(): destination for source, destination in plan
        }
        for source, destination in plan:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            if _split_locale(source.name) is not None:
                _rewrite_staged_markdown(source, destination, staged_paths)
    except (OSError, UnicodeError, ValueError) as error:
        raise DocumentationOperationError(
            "unable to stage documentation site"
        ) from error
    return sum(_split_locale(source.name) is not None for source, _ in plan)
