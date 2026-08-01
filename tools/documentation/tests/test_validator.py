import dataclasses
import importlib
import inspect
import tempfile
import unittest
from pathlib import Path
from typing import get_type_hints

import tools.documentation.validator as validator_module
from tools.documentation.validator import validate_repository


def _document(
    *,
    lang: str,
    title: str = "Title",
    doc_id: str = "sample",
    audience: str = "developer",
    doc_type: str = "explanation",
    status: str = "active",
    owner: str = "project",
    reviewed: str = "2026-08-01",
    body: str = '<a id="purpose"></a>\n## Purpose\n',
) -> str:
    return (
        "---\n"
        f"id: {doc_id}\n"
        f"title: {title}\n"
        f"lang: {lang}\n"
        f"audience: {audience}\n"
        f"type: {doc_type}\n"
        f"status: {status}\n"
        f"owner: {owner}\n"
        f"last_reviewed: {reviewed}\n"
        "---\n\n"
        f"{body}"
    )


def _write_pair(root: Path, relative_stem: str = "component/sample") -> None:
    stem = root / relative_stem
    stem.parent.mkdir(parents=True, exist_ok=True)
    stem.with_name(f"{stem.name}.en.md").write_text(
        _document(lang="en", title="Title"), encoding="utf-8"
    )
    stem.with_name(f"{stem.name}.zh-TW.md").write_text(
        _document(lang="zh-TW", title="標題"), encoding="utf-8"
    )


class ValidatorInterfaceTests(unittest.TestCase):
    def test_public_interface(self) -> None:
        module = importlib.import_module("tools.documentation.validator")

        self.assertEqual(
            {name for name in vars(module) if not name.startswith("_")},
            {"Issue", "validate_repository"},
        )
        self.assertEqual(module.__all__, ["Issue", "validate_repository"])
        self.assertTrue(dataclasses.is_dataclass(module.Issue))
        self.assertEqual(
            [field.name for field in dataclasses.fields(module.Issue)],
            ["path", "code", "message"],
        )
        issue = module.Issue(Path("doc.en.md"), "sample", "message")
        with self.assertRaises(dataclasses.FrozenInstanceError):
            issue.code = "changed"
        self.assertEqual(
            list(inspect.signature(module.validate_repository).parameters),
            ["repo_root"],
        )
        self.assertEqual(
            get_type_hints(module.validate_repository),
            {"repo_root": Path, "return": list[module.Issue]},
        )
        self.assertEqual(
            module.validate_repository.__doc__,
            "Return deterministic validation issues sorted by path, code, and message.",
        )


class ValidRepositoryTests(unittest.TestCase):
    def test_empty_and_valid_pair(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.assertEqual(validate_repository(root), [])
            _write_pair(root)
            self.assertEqual(validate_repository(root), [])


class DiscoveryTests(unittest.TestCase):
    def test_boundary_exclusions_files_and_order(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            expected = [Path("a.en.md"), Path("nested/b.zh-TW.md")]
            for relative in expected:
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("untracked", encoding="utf-8")

            ignored_files = [
                "README.md",
                "legacy.md",
                "sample.md.template",
                "data.csv",
                "AGENTS.en.md.template",
                "wrong.zh-tw.md",
                "docs/governance/templates/template.en.md",
            ]
            ignored_directories = [
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
            ]
            for relative in ignored_files:
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("ignored", encoding="utf-8")
            for name in ignored_directories:
                path = root / name / "ignored.en.md"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("ignored", encoding="utf-8")
            (root / "directory.en.md").mkdir()

            actual = [
                path.relative_to(root)
                for path in validator_module._discover_candidates(root)
            ]
            self.assertEqual(actual, expected)


class FilenameTests(unittest.TestCase):
    def test_all_name_families(self) -> None:
        valid = [
            "index.en.md",
            "lowercase-kebab-case.en.md",
            "2026-08-01-report.en.md",
            "adr-0001-record.en.md",
        ]
        invalid = [
            "Upper.en.md",
            "under_score.en.md",
            "space name.en.md",
            "two--hyphens.en.md",
            "-leading.en.md",
            "trailing-.en.md",
            "adr-001-record.en.md",
            "adr-0001.en.md",
            "2026-02-30-report.en.md",
            "2026-2-30-report.en.md",
            "20260801-report.en.md",
            "2026-08-01-.en.md",
            "2fast.en.md",
            "wrong.ZH-TW.en.md",
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in valid:
                (root / name).write_text(_document(lang="en"), encoding="utf-8")
            for name in invalid:
                (root / name).write_text("content", encoding="utf-8")
            (root / "wrong.zh-tw.md").write_text("ignored", encoding="utf-8")

            actual = [
                (issue.path.as_posix(), issue.code)
                for issue in validate_repository(root)
                if issue.code == "invalid-name"
            ]

        self.assertEqual(
            actual,
            sorted((name, "invalid-name") for name in invalid),
        )


class FrontmatterTests(unittest.TestCase):
    def test_scalar_grammar_comments_and_non_cascade(self) -> None:
        malformed = {
            "absent-opener.en.md": "id: sample\n---\n",
            "late-opener.en.md": "\n---\nid: sample\n---\n",
            "missing-closer.en.md": "---\nid: sample\n",
            "malformed-line.en.md": "---\nid sample\n---\n",
            "duplicate-key.en.md": "---\nid: one\nid: two\n---\n",
            "empty-value.en.md": "---\nid:\n---\n",
            "list-value.en.md": "---\nid: [one, two]\n---\n",
            "mapping-value.en.md": "---\nid: {one: two}\n---\n",
            "list-line.en.md": "---\n- item\n---\n",
        }
        indicators = [
            "|",
            ">",
            "|-",
            "|+",
            ">-",
            ">+",
            "|2-",
            "|+2",
            ">2+",
            ">+2",
            "|2- # note",
            ">+2 # note",
        ]
        for index, indicator in enumerate(indicators):
            malformed[f"block-{index}.en.md"] = (
                f"---\ntitle: {indicator}\ncontinued text\n---\n"
            )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, content in malformed.items():
                (root / name).write_text(content, encoding="utf-8")
            (root / "valid-scalar.en.md").write_text(
                _document(lang="en", title="Pipe | text"), encoding="utf-8"
            )

            actual = [
                (issue.path.as_posix(), issue.code)
                for issue in validate_repository(root)
                if issue.code == "frontmatter"
            ]

        self.assertEqual(
            actual,
            sorted((name, "frontmatter") for name in malformed),
        )


class MetadataFieldTests(unittest.TestCase):
    def test_exact_required_fields(self) -> None:
        fields = [
            "id",
            "title",
            "lang",
            "audience",
            "type",
            "status",
            "owner",
            "last_reviewed",
        ]
        invalid_names = []
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            valid_text = _document(lang="en")
            for field in fields:
                name = f"missing-{field.replace('_', '-')}.en.md"
                invalid_names.append(name)
                lines = [
                    line
                    for line in valid_text.splitlines()
                    if not line.startswith(f"{field}:")
                ]
                (root / name).write_text("\n".join(lines) + "\n", encoding="utf-8")
            extra_name = "extra-field.en.md"
            invalid_names.append(extra_name)
            (root / extra_name).write_text(
                valid_text.replace("---\n\n", "unsupported: value\n---\n\n", 1),
                encoding="utf-8",
            )
            (root / "exact-fields.en.md").write_text(valid_text, encoding="utf-8")

            actual = [
                (issue.path.as_posix(), issue.code)
                for issue in validate_repository(root)
                if issue.code == "invalid-metadata"
            ]

        self.assertEqual(
            actual,
            sorted((name, "invalid-metadata") for name in invalid_names),
        )


class MetadataEnumTests(unittest.TestCase):
    def test_all_enums_and_status_mappings(self) -> None:
        knowledge_types = [
            "index",
            "tutorial",
            "how-to",
            "reference",
            "explanation",
            "safety",
            "troubleshooting",
        ]
        status_map = {
            **{kind: ["draft", "active", "deprecated"] for kind in knowledge_types},
            "decision": ["accepted", "superseded"],
            "design": [
                "proposed",
                "approved",
                "implemented",
                "rejected",
                "superseded",
            ],
            "plan": ["draft", "active", "blocked", "completed", "cancelled"],
            "roadmap": ["active"],
            "release": ["published"],
            "experiment-summary": ["published"],
            "audit": ["published"],
        }
        invalid_names = []
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            counter = 0

            def add_valid(**overrides: str) -> None:
                nonlocal counter
                counter += 1
                lang = overrides.pop("lang", "en")
                suffix = "zh-TW" if lang == "zh-TW" else "en"
                path = root / f"valid-{counter}.{suffix}.md"
                path.write_text(_document(lang=lang, **overrides), encoding="utf-8")

            add_valid(lang="en")
            add_valid(lang="zh-TW")
            for audience in ["operator", "developer", "shared"]:
                add_valid(audience=audience)
            for owner in [
                "project",
                "core",
                "training",
                "panel",
                "deployment",
                "sim2real",
                "reward-agent",
            ]:
                add_valid(owner=owner)
            for doc_type, statuses in status_map.items():
                for status in statuses:
                    add_valid(doc_type=doc_type, status=status)

            invalid_cases = [
                {"lang": "EN"},
                {"audience": "everyone"},
                {"owner": "unknown"},
                {"doc_type": "guide"},
                {"doc_type": "tutorial", "status": "published"},
                {"doc_type": "decision", "status": "active"},
                {"doc_type": "design", "status": "active"},
                {"doc_type": "plan", "status": "approved"},
                {"doc_type": "roadmap", "status": "draft"},
                {"doc_type": "release", "status": "active"},
                {"doc_type": "experiment-summary", "status": "active"},
                {"doc_type": "audit", "status": "active"},
            ]
            for index, values in enumerate(invalid_cases):
                name = f"invalid-enum-{index}.en.md"
                invalid_names.append(name)
                (root / name).write_text(
                    _document(lang=values.pop("lang", "en"), **values),
                    encoding="utf-8",
                )

            actual = sorted(
                issue.path.as_posix()
                for issue in validate_repository(root)
                if issue.code == "invalid-metadata"
            )

        self.assertEqual(actual, sorted(invalid_names))


class MetadataShapeTests(unittest.TestCase):
    def test_id_date_and_filename_language(self) -> None:
        cases = {
            "upper-id.en.md": {"doc_id": "Upper-Id"},
            "underscore-id.en.md": {"doc_id": "under_score"},
            "double-hyphen-id.en.md": {"doc_id": "double--hyphen"},
            "edge-hyphen-id.en.md": {"doc_id": "-edge"},
            "impossible-date.en.md": {"reviewed": "2026-02-30"},
            "short-date.en.md": {"reviewed": "2026-2-01"},
            "datetime-review.en.md": {"reviewed": "2026-08-01T00:00:00"},
            "english-mismatch.en.md": {"lang": "zh-TW"},
            "chinese-mismatch.zh-TW.md": {"lang": "en"},
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, values in cases.items():
                lang = values.get("lang", "en")
                overrides = {key: value for key, value in values.items() if key != "lang"}
                (root / name).write_text(
                    _document(lang=lang, **overrides), encoding="utf-8"
                )
            (root / "valid-shape.en.md").write_text(
                _document(lang="en", doc_id="valid-shape", reviewed="2024-02-29"),
                encoding="utf-8",
            )
            actual = sorted(
                issue.path.as_posix()
                for issue in validate_repository(root)
                if issue.code == "invalid-metadata"
            )
        self.assertEqual(actual, sorted(cases))


class LocationTests(unittest.TestCase):
    def test_exhaustive_central_matrix(self) -> None:
        knowledge = [
            "tutorial",
            "how-to",
            "reference",
            "explanation",
            "safety",
            "troubleshooting",
        ]
        sections = {
            "operators": ("operator", knowledge),
            "developers": ("developer", knowledge),
            "reference": ("shared", ["reference"]),
            "decisions": ("developer", ["decision"]),
            "designs": ("developer", ["design"]),
            "plans": ("developer", ["plan"]),
            "roadmap": ("shared", ["roadmap"]),
            "releases": ("shared", ["release"]),
            "research": (
                "developer",
                ["experiment-summary", "audit", "explanation"],
            ),
            "governance": ("developer", ["reference"]),
        }
        wrong_types = {
            "operators": "decision",
            "developers": "release",
            "reference": "explanation",
            "decisions": "design",
            "designs": "decision",
            "plans": "roadmap",
            "roadmap": "release",
            "releases": "roadmap",
            "research": "release",
            "governance": "decision",
        }
        status_for = {
            "index": "active",
            "tutorial": "active",
            "how-to": "active",
            "reference": "active",
            "explanation": "active",
            "safety": "active",
            "troubleshooting": "active",
            "decision": "accepted",
            "design": "approved",
            "plan": "active",
            "roadmap": "active",
            "release": "published",
            "experiment-summary": "published",
            "audit": "published",
        }
        invalid_paths = []
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            def write(
                relative: str, *, audience: str, doc_type: str, doc_id: str
            ) -> None:
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(
                    _document(
                        lang="en",
                        audience=audience,
                        doc_type=doc_type,
                        status=status_for[doc_type],
                        doc_id=doc_id,
                    ),
                    encoding="utf-8",
                )

            write(
                "docs/index.en.md",
                audience="shared",
                doc_type="index",
                doc_id="docs-portal",
            )
            for section, (audience, allowed_types) in sections.items():
                write(
                    f"docs/{section}/index.en.md",
                    audience=audience,
                    doc_type="index",
                    doc_id=f"{section}-portal",
                )
                for index, doc_type in enumerate(allowed_types):
                    write(
                        f"docs/{section}/valid-{index}.en.md",
                        audience=audience,
                        doc_type=doc_type,
                        doc_id=f"{section}-valid-{index}",
                    )
                wrong_audience = "developer" if audience != "developer" else "shared"
                audience_path = f"docs/{section}/wrong-audience.en.md"
                invalid_paths.append(audience_path)
                write(
                    audience_path,
                    audience=wrong_audience,
                    doc_type=allowed_types[0],
                    doc_id=f"{section}-wrong-audience",
                )
                type_path = f"docs/{section}/wrong-type.en.md"
                invalid_paths.append(type_path)
                wrong_type = wrong_types[section]
                write(
                    type_path,
                    audience=audience,
                    doc_type=wrong_type,
                    doc_id=f"{section}-wrong-type",
                )
                non_index_portal = f"docs/{section}/not-index.en.md"
                invalid_paths.append(non_index_portal)
                write(
                    non_index_portal,
                    audience=audience,
                    doc_type="index",
                    doc_id=f"{section}-not-index",
                )
                nested_portal = f"docs/{section}/nested/index.en.md"
                invalid_paths.append(nested_portal)
                write(
                    nested_portal,
                    audience=audience,
                    doc_type="index",
                    doc_id=f"{section}-nested-index",
                )

            invalid_paths.extend(["docs/unknown/page.en.md", "docs/root-page.en.md"])
            write(
                "docs/unknown/page.en.md",
                audience="developer",
                doc_type="explanation",
                doc_id="unknown-page",
            )
            write(
                "docs/root-page.en.md",
                audience="shared",
                doc_type="reference",
                doc_id="root-page",
            )
            write(
                "component/free-location.en.md",
                audience="shared",
                doc_type="release",
                doc_id="free-location",
            )

            actual = sorted(
                issue.path.as_posix()
                for issue in validate_repository(root)
                if issue.code == "invalid-location"
            )

        self.assertEqual(actual, sorted(invalid_paths))


class PhysicalPairTests(unittest.TestCase):
    def test_presence_is_content_independent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "lone.en.md").write_text(_document(lang="en"), encoding="utf-8")

            (root / "malformed.en.md").write_text(
                _document(lang="en", doc_id="malformed"), encoding="utf-8"
            )
            (root / "malformed.zh-TW.md").write_text(
                "not frontmatter", encoding="utf-8"
            )

            (root / "bad-metadata.en.md").write_text(
                _document(lang="en", doc_id="bad-metadata"), encoding="utf-8"
            )
            (root / "bad-metadata.zh-TW.md").write_text(
                _document(lang="zh-TW", doc_id="Bad_Metadata"), encoding="utf-8"
            )

            (root / "lookalike.en.md").write_text(
                _document(lang="en", doc_id="lookalike"), encoding="utf-8"
            )
            (root / "lookalike.BAD.zh-TW.md").write_text(
                _document(lang="zh-TW", doc_id="lookalike"), encoding="utf-8"
            )

            actual = sorted(
                issue.path.as_posix()
                for issue in validate_repository(root)
                if issue.code == "missing-pair"
            )

        self.assertEqual(actual, ["lone.en.md", "lookalike.en.md"])


class PairMetadataTests(unittest.TestCase):
    def test_parity_once_and_location_independence(self) -> None:
        expected = []
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            def pair(name: str, en: dict[str, str], zh: dict[str, str]) -> None:
                en_path = root / f"{name}.en.md"
                zh_path = root / f"{name}.zh-TW.md"
                en_path.parent.mkdir(parents=True, exist_ok=True)
                en_path.write_text(_document(lang="en", **en), encoding="utf-8")
                zh_path.write_text(
                    _document(lang="zh-TW", title="標題", **zh), encoding="utf-8"
                )

            pair("localized", {"doc_id": "localized"}, {"doc_id": "localized"})
            drift_cases = {
                "drift-id": ({"doc_id": "first"}, {"doc_id": "second"}),
                "drift-audience": (
                    {"doc_id": "audience", "audience": "developer"},
                    {"doc_id": "audience", "audience": "shared"},
                ),
                "drift-type": (
                    {"doc_id": "type", "doc_type": "explanation"},
                    {"doc_id": "type", "doc_type": "reference"},
                ),
                "drift-status": (
                    {"doc_id": "status", "doc_type": "plan", "status": "active"},
                    {"doc_id": "status", "doc_type": "plan", "status": "blocked"},
                ),
                "drift-owner": (
                    {"doc_id": "owner", "owner": "project"},
                    {"doc_id": "owner", "owner": "core"},
                ),
                "drift-review": (
                    {"doc_id": "review"},
                    {"doc_id": "review", "reviewed": "2026-07-31"},
                ),
            }
            for name, (english, chinese) in drift_cases.items():
                expected.append(f"{name}.en.md")
                pair(name, english, chinese)

            expected.append("docs/operators/location-drift.en.md")
            pair(
                "docs/operators/location-drift",
                {"doc_id": "location-drift", "audience": "developer"},
                {"doc_id": "location-drift", "audience": "shared"},
            )
            pair(
                "invalid-metadata",
                {"doc_id": "valid-side", "owner": "project"},
                {"doc_id": "Invalid_Side", "owner": "core"},
            )

            actual = sorted(
                issue.path.as_posix()
                for issue in validate_repository(root)
                if issue.code == "pair-metadata"
            )

        self.assertEqual(actual, sorted(expected))


class DuplicateIdTests(unittest.TestCase):
    def test_complete_incomplete_and_location_independence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            def pair(stem: str, doc_id: str, **values: str) -> None:
                base = root / stem
                base.parent.mkdir(parents=True, exist_ok=True)
                base.with_name(f"{base.name}.en.md").write_text(
                    _document(lang="en", doc_id=doc_id, **values), encoding="utf-8"
                )
                base.with_name(f"{base.name}.zh-TW.md").write_text(
                    _document(lang="zh-TW", doc_id=doc_id, **values),
                    encoding="utf-8",
                )

            pair("complete-a", "complete-duplicate")
            pair("complete-b", "complete-duplicate")
            for stem in ["incomplete-c", "incomplete-d"]:
                (root / f"{stem}.en.md").write_text(
                    _document(lang="en", doc_id="incomplete-duplicate"),
                    encoding="utf-8",
                )
            pair("docs/unknown/location-a", "location-duplicate")
            pair("docs/unknown/location-b", "location-duplicate")
            pair("single-logical", "single-logical")
            pair("invalid-a", "invalid-excluded", owner="unknown")
            pair("invalid-b", "invalid-excluded", owner="unknown")

            issues = validate_repository(root)
            duplicates = [
                issue.path.as_posix()
                for issue in issues
                if issue.code == "duplicate-id"
            ]
            missing = [
                issue.path.as_posix() for issue in issues if issue.code == "missing-pair"
            ]

        self.assertEqual(
            duplicates,
            [
                "complete-a.en.md",
                "docs/unknown/location-a.en.md",
                "incomplete-c.en.md",
            ],
        )
        self.assertEqual(missing, ["incomplete-c.en.md", "incomplete-d.en.md"])

    def test_each_metadata_valid_locale_participates_when_pair_ids_drift(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for stem, english_id, chinese_id in [
                ("a-drift", "english-only-id", "reused-chinese-id"),
                ("b-other", "reused-chinese-id", "reused-chinese-id"),
            ]:
                (root / f"{stem}.en.md").write_text(
                    _document(lang="en", doc_id=english_id), encoding="utf-8"
                )
                (root / f"{stem}.zh-TW.md").write_text(
                    _document(lang="zh-TW", doc_id=chinese_id), encoding="utf-8"
                )

            duplicates = [
                issue.path.as_posix()
                for issue in validate_repository(root)
                if issue.code == "duplicate-id"
            ]

        self.assertEqual(duplicates, ["a-drift.zh-TW.md"])


class AnchorStructureTests(unittest.TestCase):
    def test_rules_non_cascade_and_independence(self) -> None:
        bodies = {
            "missing.en.md": "## Missing\n",
            "invalid-followed.en.md": '<a id="Bad_Anchor"></a>\n## Heading\n',
            "standalone.en.md": '<a id="standalone"></a>\nParagraph\n',
            "duplicate.en.md": (
                '<a id="same"></a>\n## One\n\n'
                '<a id="same"></a>\n## Two\n'
            ),
            "invalid-metadata.en.md": "## Metadata Independent\n",
            "docs/unknown/invalid-location.en.md": "## Location Independent\n",
        }
        expected = [
            ("docs/unknown/invalid-location.en.md", "heading-anchor"),
            ("duplicate.en.md", "duplicate-anchor"),
            ("invalid-followed.en.md", "heading-anchor"),
            ("invalid-metadata.en.md", "heading-anchor"),
            ("missing.en.md", "heading-anchor"),
            ("standalone.en.md", "heading-anchor"),
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, body in bodies.items():
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                owner = "unknown" if name == "invalid-metadata.en.md" else "project"
                path.write_text(
                    _document(lang="en", doc_id="anchor-case", owner=owner, body=body),
                    encoding="utf-8",
                )
            (root / "valid-anchor.en.md").write_text(
                _document(lang="en", doc_id="valid-anchor"), encoding="utf-8"
            )

            actual = [
                (issue.path.as_posix(), issue.code)
                for issue in validate_repository(root)
                if issue.code in {"heading-anchor", "duplicate-anchor"}
            ]

        self.assertEqual(actual, expected)


class PairAnchorTests(unittest.TestCase):
    def test_sequence_once_and_only_structure_disqualifies(self) -> None:
        def body(anchor: str, extra: str = "") -> str:
            return f'<a id="{anchor}"></a>\n## Heading\n{extra}'

        expected = [
            "broken-link.en.md",
            "docs/unknown/invalid-location.en.md",
            "invalid-metadata.en.md",
            "metadata-drift.en.md",
            "mismatch.en.md",
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            def pair(
                stem: str,
                en_body: str,
                zh_body: str,
                en_values: dict[str, str] | None = None,
                zh_values: dict[str, str] | None = None,
            ) -> None:
                base = root / stem
                base.parent.mkdir(parents=True, exist_ok=True)
                english = {"doc_id": stem.replace("/", "-")}
                chinese = dict(english)
                english.update(en_values or {})
                chinese.update(zh_values or {})
                base.with_name(f"{base.name}.en.md").write_text(
                    _document(lang="en", body=en_body, **english), encoding="utf-8"
                )
                base.with_name(f"{base.name}.zh-TW.md").write_text(
                    _document(lang="zh-TW", body=zh_body, **chinese),
                    encoding="utf-8",
                )

            pair("equal", body("same"), body("same"))
            pair("mismatch", body("one"), body("two"))
            pair(
                "invalid-metadata",
                body("one"),
                body("two"),
                {"owner": "unknown"},
                {"owner": "unknown"},
            )
            pair(
                "docs/unknown/invalid-location", body("one"), body("two")
            )
            pair(
                "metadata-drift",
                body("one"),
                body("two"),
                {"owner": "project"},
                {"owner": "core"},
            )
            pair(
                "broken-link",
                body("one", "[missing](missing.md)\n"),
                body("two", "[missing](missing.md)\n"),
            )
            pair("structure-defect", "## Missing\n", body("different"))

            actual = [
                issue.path.as_posix()
                for issue in validate_repository(root)
                if issue.code == "pair-anchors"
            ]

        self.assertEqual(actual, expected)


class FenceTests(unittest.TestCase):
    def test_marker_character_length_indent_and_ignored_content(self) -> None:
        body = """visible-before
```python
## ignored heading
<a id="Bad_Anchor"></a>
[inline](missing.md)
![image](missing.png)
[full][missing]
![collapsed][]
[definition]: missing.md
~~~
``` not-a-closer
## still ignored
```
visible-middle
   ~~~~ info
## tilde ignored
[reference][missing]
```
~~~
## still tilde ignored
~~~~<TRAILING-SPACES>
visible-after
    ```
## visible heading
""".replace("<TRAILING-SPACES>", "   ")
        self.assertEqual(
            validator_module._outside_fence_lines(body),
            [
                "visible-before",
                "visible-middle",
                "visible-after",
                "    ```",
                "## visible heading",
            ],
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "fences.en.md").write_text(
                _document(lang="en", doc_id="fences", body=body), encoding="utf-8"
            )
            actual = [
                issue.code
                for issue in validate_repository(root)
                if issue.code in {"heading-anchor", "duplicate-anchor"}
            ]
        self.assertEqual(actual, ["heading-anchor"])


class InlineLinkTests(unittest.TestCase):
    def test_openers_destinations_titles_and_negatives(self) -> None:
        body = r"""
[simple](target.txt)
![image](image.png "caption")
[angle](<target file.txt>)
[balanced](dir/file_(one).txt)
[double title](double.txt "title (with parentheses)")
[single title](single.txt 'title')
[paren title](paren.txt (title (nested)))
plain text](missing-plain.txt)
\[escaped](missing-escaped.txt)
[malformed(missing-malformed.txt)
```
[fenced](missing-fenced.txt)
![fenced image](missing-fenced.png)
```
"""
        self.assertEqual(
            validator_module._inline_destinations(body),
            [
                "target.txt",
                "image.png",
                "target file.txt",
                "dir/file_(one).txt",
                "double.txt",
                "single.txt",
                "paren.txt",
            ],
        )


class ReferenceLinkTests(unittest.TestCase):
    def test_full_collapsed_shortcut_definitions_and_negatives(self) -> None:
        body = r"""
[Full Label]: target.txt "Title (with parentheses)"
[collapsed]: <target file.txt> 'Title'
[shortcut]: shortcut.txt (Parenthesized title)
[image label]: image.png

[text][ full   LABEL ]
![alt][IMAGE LABEL]
[collapsed][]
![shortcut][]
[shortcut]
![image label]
[text][missing full]
![missing collapsed][]
[ordinary prose]
\[shortcut]
[shortcut](inline.txt)
plain text][full label]
[broken][
"""
        analysis = validator_module._analyze_references(body)
        self.assertEqual(
            analysis.definitions,
            {
                "full label": "target.txt",
                "collapsed": "target file.txt",
                "shortcut": "shortcut.txt",
                "image label": "image.png",
            },
        )
        self.assertEqual(
            analysis.used_labels,
            (
                "full label",
                "image label",
                "collapsed",
                "shortcut",
                "shortcut",
                "image label",
            ),
        )
        self.assertEqual(
            analysis.missing_labels,
            ("missing full", "missing collapsed"),
        )


class LocalTargetTests(unittest.TestCase):
    def test_absolute_containment_percent_file_and_fragments(self) -> None:
        body = r"""<a id="source-anchor"></a>
## Source
[http](http://example.com/page)
[https](https://example.com/page)
[mail](mailto:docs@example.com)
[other scheme](custom:opaque)
[encoded](target%20file.txt)
[angle](<target file.txt>)
[current fragment](#source-anchor)
[target fragment](target.md#target%2Danchor)
[missing](missing.txt)
![missing image](missing.png)
[posix](/absolute/path)
[drive forward](C:/secret/file.md)
[drive back](C:\secret\file.md)
[unc](\\server\share\file.md)
[outside](../outside.txt)
[directory](folder)
[missing fragment file](missing.md#anchor)
[target missing anchor](target.md#generated-heading)
[current missing anchor](#missing-current)
[bad ref]: missing-reference.txt
[anchor ref]: target.md#missing-definition-anchor
[use bad][bad ref]
[use anchor][anchor ref]
[missing full][absent]
![missing collapsed][]
[ordinary shortcut prose]
\[escaped](missing-escaped.txt)
plain text](missing-plain.txt)
```
[fenced](missing-fenced.txt)
```
"""
        with tempfile.TemporaryDirectory() as directory:
            outer = Path(directory)
            root = outer / "repo"
            root.mkdir()
            (outer / "outside.txt").write_text("outside", encoding="utf-8")
            (root / "target file.txt").write_text("target", encoding="utf-8")
            (root / "folder").mkdir()
            (root / "target.md").write_text(
                '<a id="target-anchor"></a>\n## Target\n## Generated Heading\n',
                encoding="utf-8",
            )
            (root / "source.en.md").write_text(
                _document(lang="en", doc_id="source", body=body), encoding="utf-8"
            )

            link_codes = [
                issue.code
                for issue in validate_repository(root)
                if issue.code in {"broken-link", "missing-link-anchor"}
            ]

        self.assertEqual(
            link_codes,
            ["broken-link"] * 12 + ["missing-link-anchor"] * 3,
        )


class AggregationTests(unittest.TestCase):
    def test_exact_independence_non_cascade_and_order(self) -> None:
        def anchor_body(anchor: str, extra: str = "") -> str:
            return f'<a id="{anchor}"></a>\n## Heading\n{extra}'

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            def pair(
                stem: str,
                en_text: str,
                zh_text: str,
            ) -> None:
                base = root / stem
                base.parent.mkdir(parents=True, exist_ok=True)
                base.with_name(f"{base.name}.en.md").write_text(en_text, encoding="utf-8")
                base.with_name(f"{base.name}.zh-TW.md").write_text(
                    zh_text, encoding="utf-8"
                )

            pair(
                "parse",
                "not frontmatter",
                _document(lang="zh-TW", doc_id="parse"),
            )
            pair(
                "invalid-meta",
                _document(
                    lang="en",
                    doc_id="invalid-meta",
                    owner="unknown",
                    body=anchor_body("same", "[bad](missing-meta.txt)\n"),
                ),
                _document(
                    lang="zh-TW",
                    doc_id="invalid-meta",
                    owner="unknown",
                    body=anchor_body("same", "[bad](missing-meta.txt)\n"),
                ),
            )
            pair(
                "docs/unknown/alpha",
                _document(
                    lang="en",
                    doc_id="shared-duplicate",
                    body=anchor_body("one", "[bad](missing-alpha.txt)\n"),
                ),
                _document(
                    lang="zh-TW",
                    doc_id="shared-duplicate",
                    body=anchor_body("two"),
                ),
            )
            pair(
                "docs/unknown/beta",
                _document(
                    lang="en",
                    doc_id="shared-duplicate",
                    owner="project",
                    body=anchor_body("three"),
                ),
                _document(
                    lang="zh-TW",
                    doc_id="shared-duplicate",
                    owner="core",
                    body=anchor_body("three"),
                ),
            )
            (root / "lone.en.md").write_text(
                _document(lang="en", doc_id="lone"), encoding="utf-8"
            )
            pair(
                "structure",
                _document(
                    lang="en",
                    doc_id="structure",
                    body=(
                        '<a id="same"></a>\n## One\n'
                        '<a id="same"></a>\n## Two\n'
                    ),
                ),
                _document(
                    lang="zh-TW",
                    doc_id="structure",
                    body=anchor_body("different"),
                ),
            )

            actual = validate_repository(root)

        expected = [
            validator_module.Issue(
                Path("docs/unknown/alpha.en.md"),
                "broken-link",
                "local link target is not a file: missing-alpha.txt",
            ),
            validator_module.Issue(
                Path("docs/unknown/alpha.en.md"),
                "duplicate-id",
                "id is reused by multiple logical documents: shared-duplicate",
            ),
            validator_module.Issue(
                Path("docs/unknown/alpha.en.md"),
                "invalid-location",
                "unknown central documentation section",
            ),
            validator_module.Issue(
                Path("docs/unknown/alpha.en.md"),
                "pair-anchors",
                "paired documents have different ordered anchor sequences",
            ),
            validator_module.Issue(
                Path("docs/unknown/alpha.zh-TW.md"),
                "invalid-location",
                "unknown central documentation section",
            ),
            validator_module.Issue(
                Path("docs/unknown/beta.en.md"),
                "invalid-location",
                "unknown central documentation section",
            ),
            validator_module.Issue(
                Path("docs/unknown/beta.en.md"),
                "pair-metadata",
                "paired metadata differs outside title and lang",
            ),
            validator_module.Issue(
                Path("docs/unknown/beta.zh-TW.md"),
                "invalid-location",
                "unknown central documentation section",
            ),
            validator_module.Issue(
                Path("invalid-meta.en.md"),
                "broken-link",
                "local link target is not a file: missing-meta.txt",
            ),
            validator_module.Issue(
                Path("invalid-meta.en.md"), "invalid-metadata", "unsupported owner"
            ),
            validator_module.Issue(
                Path("invalid-meta.zh-TW.md"),
                "broken-link",
                "local link target is not a file: missing-meta.txt",
            ),
            validator_module.Issue(
                Path("invalid-meta.zh-TW.md"), "invalid-metadata", "unsupported owner"
            ),
            validator_module.Issue(
                Path("lone.en.md"),
                "missing-pair",
                "canonical locale companion is missing",
            ),
            validator_module.Issue(
                Path("parse.en.md"),
                "frontmatter",
                "frontmatter must start on line 1",
            ),
            validator_module.Issue(
                Path("structure.en.md"),
                "duplicate-anchor",
                "explicit anchor is duplicated: same",
            ),
        ]
        self.assertEqual(actual, expected)


class ActualRepositoryTests(unittest.TestCase):
    def test_current_repository_is_valid(self) -> None:
        repo_root = Path(__file__).resolve().parents[3]
        self.assertEqual(validate_repository(repo_root), [])


if __name__ == "__main__":
    unittest.main()
