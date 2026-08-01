"""Select documentation paths changed through Git."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

from .errors import DocumentationOperationError
from .validator import Issue, _split_locale, _valid_stem


def parse_name_status(data: bytes) -> set[Path]:
    """Return paths from NUL-delimited Git name-status output."""
    if not data:
        return set()
    if not data.endswith(b"\0"):
        raise DocumentationOperationError("invalid Git name-status output")
    fields = data[:-1].split(b"\0")
    paths: set[Path] = set()
    index = 0
    while index < len(fields):
        status = fields[index]
        index += 1
        path_count = 2 if status.startswith((b"R", b"C")) else 1
        if not status or status[:1] not in {b"A", b"M", b"T", b"D", b"R", b"C"}:
            raise DocumentationOperationError("invalid Git name-status output")
        if index + path_count > len(fields):
            raise DocumentationOperationError("invalid Git name-status output")
        record_paths = fields[index : index + path_count]
        if any(not path for path in record_paths):
            raise DocumentationOperationError("invalid Git name-status output")
        paths.update(Path(os.fsdecode(path)) for path in record_paths)
        index += path_count
    return paths


def _run_git(repo_root: Path, arguments: list[str], failure: str) -> bytes:
    try:
        result = subprocess.run(
            ["git", *arguments],
            cwd=repo_root,
            capture_output=True,
            check=False,
        )
    except OSError as error:
        raise DocumentationOperationError(failure) from error
    if result.returncode != 0:
        raise DocumentationOperationError(failure)
    return result.stdout


def select_staged_paths(repo_root: Path) -> set[Path]:
    """Return paths represented in the staged Git change set."""
    output = _run_git(
        repo_root,
        ["diff", "--cached", "--name-status", "-z", "--find-renames"],
        "unable to inspect staged Git changes",
    )
    return parse_name_status(output)


def select_changed_paths(repo_root: Path, reference: str) -> set[Path]:
    """Return paths committed between a verified reference and ``HEAD``."""
    resolved = _run_git(
        repo_root,
        [
            "rev-parse",
            "--verify",
            "--quiet",
            "--end-of-options",
            f"{reference}^{{commit}}",
        ],
        "invalid Git reference",
    )
    try:
        commit = resolved.decode("ascii").strip()
    except UnicodeError as error:
        raise DocumentationOperationError("invalid Git reference") from error
    if not commit:
        raise DocumentationOperationError("invalid Git reference")
    output = _run_git(
        repo_root,
        [
            "diff",
            "--name-status",
            "-z",
            "--find-renames",
            f"{commit}...HEAD",
            "--",
        ],
        "unable to inspect changed Git paths",
    )
    return parse_name_status(output)


def changed_pair_issues(paths: set[Path]) -> list[Issue]:
    """Return one issue for each canonical stem changed in only one locale."""
    paths_by_stem: dict[Path, list[Path]] = {}
    locales_by_stem: dict[Path, set[str]] = {}
    for path in paths:
        parts = path.parts
        if parts[:3] == ("docs", "governance", "templates") or parts[:2] in {
            (".agents", "skills"),
            (".claude", "skills"),
        }:
            continue
        split = _split_locale(path.name)
        if split is None:
            continue
        stem, locale = split
        if not _valid_stem(stem):
            continue
        logical_stem = path.parent / stem
        paths_by_stem.setdefault(logical_stem, []).append(path)
        locales_by_stem.setdefault(logical_stem, set()).add(locale)
    issues = [
        Issue(
            min(paths_by_stem[stem], key=lambda path: path.as_posix()),
            "changed-pair",
            "locale companion is not in selected change set",
        )
        for stem, locales in locales_by_stem.items()
        if locales != {"en", "zh-TW"}
    ]
    return sorted(
        issues,
        key=lambda issue: (issue.path.as_posix(), issue.code, issue.message),
    )
