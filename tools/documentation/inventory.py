"""Build deterministic documentation inventory data."""

from __future__ import annotations

import collections
import datetime
from pathlib import Path
from typing import Any

from .errors import DocumentationOperationError
from .schema import _parse_frontmatter
from .validator import _discover_candidates, _split_locale


def _stale_after_days(relative_path: Path, metadata: dict[str, str]) -> int | None:
    document_type = metadata["type"]
    if document_type in {"decision", "release"}:
        return None
    if (
        relative_path.parts[:2] == ("docs", "operators")
        or document_type in {"reference", "roadmap"}
    ):
        return 90
    if (
        relative_path.parts[:3] == ("docs", "developers", "architecture")
        and document_type == "explanation"
    ):
        return 180
    return None


def build_inventory(repo_root: Path, as_of: datetime.date) -> dict[str, Any]:
    """Return deterministic inventory data as of the injected date."""
    documents: list[dict[str, Any]] = []
    logical_documents: set[tuple[Path, str]] = set()
    summaries = {
        "by_audience": collections.Counter(),
        "by_owner": collections.Counter(),
        "by_status": collections.Counter(),
        "by_type": collections.Counter(),
    }
    for path in _discover_candidates(repo_root):
        relative_path = path.relative_to(repo_root)
        split = _split_locale(path.name)
        parsed = _parse_frontmatter(path.read_text(encoding="utf-8"))
        if split is None or parsed is None:
            raise DocumentationOperationError("unable to build documentation inventory")
        stem, locale = split
        other_locale = "zh-TW" if locale == "en" else "en"
        pair_path = relative_path.with_name(f"{stem}.{other_locale}.md")
        reviewed = datetime.date.fromisoformat(parsed.metadata["last_reviewed"])
        age = (as_of - reviewed).days
        if age < 0:
            raise DocumentationOperationError(
                "last_reviewed is after inventory as_of: "
                f"{relative_path.as_posix()}"
            )
        threshold = _stale_after_days(relative_path, parsed.metadata)
        stale = threshold is not None and age > threshold
        row: dict[str, Any] = {
            "path": relative_path.as_posix(),
            "pair_path": pair_path.as_posix(),
            **parsed.metadata,
            "days_since_review": age,
            "stale_after_days": threshold,
            "stale": stale,
        }
        documents.append(row)
        logical_documents.add((relative_path.parent, stem))
        for field, summary_key in (
            ("audience", "by_audience"),
            ("owner", "by_owner"),
            ("status", "by_status"),
            ("type", "by_type"),
        ):
            summaries[summary_key][parsed.metadata[field]] += 1

    return {
        "as_of": as_of.isoformat(),
        "document_count": len(documents),
        "documents": documents,
        "logical_document_count": len(logical_documents),
        "schema_version": 1,
        "summary": {
            key: dict(sorted(counts.items()))
            for key, counts in summaries.items()
        }
        | {"stale_documents": sum(bool(row["stale"]) for row in documents)},
    }
