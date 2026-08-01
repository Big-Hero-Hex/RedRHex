from __future__ import annotations

import datetime
import tempfile
import unittest
from pathlib import Path


def _document(
    *,
    lang: str,
    title: str,
    document_id: str = "guide",
    audience: str = "developer",
    document_type: str = "explanation",
    status: str = "active",
    owner: str = "core",
    last_reviewed: str = "2026-08-01",
) -> str:
    return (
        "---\n"
        f"id: {document_id}\n"
        f"title: {title}\n"
        f"lang: {lang}\n"
        f"audience: {audience}\n"
        f"type: {document_type}\n"
        f"status: {status}\n"
        f"owner: {owner}\n"
        f"last_reviewed: {last_reviewed}\n"
        "---\n\n"
        '<a id="overview"></a>\n'
        "## Overview\n"
    )


class InventoryContractTests(unittest.TestCase):
    def test_rows_counts_companions_and_summaries_are_deterministic(self) -> None:
        try:
            from tools.documentation.inventory import build_inventory
        except ImportError as error:
            self.fail(f"inventory builder is missing: {error}")

        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            docs = repo / "component/資料 空間"
            docs.mkdir(parents=True)
            (docs / "guide.en.md").write_text(
                _document(lang="en", title="Guide"),
                encoding="utf-8",
            )
            (docs / "guide.zh-TW.md").write_text(
                _document(lang="zh-TW", title="指南"),
                encoding="utf-8",
            )

            inventory = build_inventory(repo, datetime.date(2026, 8, 2))

        common = {
            "id": "guide",
            "audience": "developer",
            "type": "explanation",
            "status": "active",
            "owner": "core",
            "last_reviewed": "2026-08-01",
            "days_since_review": 1,
            "stale_after_days": None,
            "stale": False,
        }
        self.assertEqual(
            inventory,
            {
                "as_of": "2026-08-02",
                "document_count": 2,
                "documents": [
                    {
                        **common,
                        "path": "component/資料 空間/guide.en.md",
                        "pair_path": "component/資料 空間/guide.zh-TW.md",
                        "title": "Guide",
                        "lang": "en",
                    },
                    {
                        **common,
                        "path": "component/資料 空間/guide.zh-TW.md",
                        "pair_path": "component/資料 空間/guide.en.md",
                        "title": "指南",
                        "lang": "zh-TW",
                    },
                ],
                "logical_document_count": 1,
                "schema_version": 1,
                "summary": {
                    "by_audience": {"developer": 2},
                    "by_owner": {"core": 2},
                    "by_status": {"active": 2},
                    "by_type": {"explanation": 2},
                    "stale_documents": 0,
                },
            },
        )

    def test_staleness_families_and_strict_age_boundaries(self) -> None:
        from tools.documentation.inventory import _stale_after_days, build_inventory

        cases = (
            (Path("docs/operators/setup.en.md"), "tutorial", "operator", 90),
            (Path("component/api.en.md"), "reference", "developer", 90),
            (Path("docs/roadmap/now.en.md"), "roadmap", "shared", 90),
            (
                Path("docs/developers/architecture/core.en.md"),
                "explanation",
                "developer",
                180,
            ),
            (Path("docs/decisions/adr-0001-choice.en.md"), "decision", "developer", None),
            (Path("docs/releases/release.en.md"), "release", "shared", None),
            (Path("component/guide.en.md"), "explanation", "developer", None),
        )
        for path, document_type, audience, expected in cases:
            with self.subTest(path=path):
                self.assertEqual(
                    _stale_after_days(
                        path,
                        {"type": document_type, "audience": audience},
                    ),
                    expected,
                )

        as_of = datetime.date(2026, 8, 2)
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)

            def write_pair(
                stem: str,
                *,
                age: int,
                document_id: str,
                audience: str,
                document_type: str,
            ) -> None:
                reviewed = (as_of - datetime.timedelta(days=age)).isoformat()
                for suffix, lang, title in (
                    ("en", "en", "Guide"),
                    ("zh-TW", "zh-TW", "指南"),
                ):
                    path = repo / f"{stem}.{suffix}.md"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(
                        _document(
                            lang=lang,
                            title=title,
                            document_id=document_id,
                            audience=audience,
                            document_type=document_type,
                            last_reviewed=reviewed,
                        ),
                        encoding="utf-8",
                    )

            write_pair(
                "docs/operators/operator-90",
                age=90,
                document_id="operator-90",
                audience="operator",
                document_type="how-to",
            )
            write_pair(
                "docs/operators/operator-91",
                age=91,
                document_id="operator-91",
                audience="operator",
                document_type="how-to",
            )
            write_pair(
                "docs/developers/architecture/architecture-180",
                age=180,
                document_id="architecture-180",
                audience="developer",
                document_type="explanation",
            )
            write_pair(
                "docs/developers/architecture/architecture-181",
                age=181,
                document_id="architecture-181",
                audience="developer",
                document_type="explanation",
            )
            inventory = build_inventory(repo, as_of)

        stale_by_id = {
            row["id"]: (row["stale_after_days"], row["stale"])
            for row in inventory["documents"]
        }
        self.assertEqual(stale_by_id["operator-90"], (90, False))
        self.assertEqual(stale_by_id["operator-91"], (90, True))
        self.assertEqual(stale_by_id["architecture-180"], (180, False))
        self.assertEqual(stale_by_id["architecture-181"], (180, True))
        self.assertEqual(inventory["summary"]["stale_documents"], 4)

    def test_future_review_date_is_an_operational_error(self) -> None:
        from tools.documentation.errors import DocumentationOperationError
        from tools.documentation.inventory import build_inventory

        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            component = repo / "component"
            component.mkdir()
            (component / "guide.en.md").write_text(
                _document(
                    lang="en",
                    title="Guide",
                    last_reviewed="2026-08-03",
                ),
                encoding="utf-8",
            )
            (component / "guide.zh-TW.md").write_text(
                _document(
                    lang="zh-TW",
                    title="指南",
                    last_reviewed="2026-08-03",
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(
                DocumentationOperationError,
                "^last_reviewed is after inventory as_of: component/guide.en.md$",
            ):
                build_inventory(repo, datetime.date(2026, 8, 2))


if __name__ == "__main__":
    unittest.main()
