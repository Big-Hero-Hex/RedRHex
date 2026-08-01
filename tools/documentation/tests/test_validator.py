from __future__ import annotations

import dataclasses
import importlib
import socket
import tempfile
import unittest
from pathlib import Path

from tools.documentation.validator import Issue, validate_repository


class ValidatorInterfaceTests(unittest.TestCase):
    def test_public_interface(self) -> None:
        validator = importlib.import_module("tools.documentation.validator")

        self.assertEqual(validator.__all__, ["Issue", "validate_repository"])
        self.assertTrue(dataclasses.is_dataclass(validator.Issue))
        self.assertTrue(validator.Issue.__dataclass_params__.frozen)
        self.assertEqual(
            [field.name for field in dataclasses.fields(validator.Issue)],
            ["path", "code", "message"],
        )
        issue = validator.Issue(Path("doc.en.md"), "code", "message")
        with self.assertRaises(dataclasses.FrozenInstanceError):
            issue.code = "changed"
        self.assertEqual(
            validator.validate_repository.__doc__,
            "Return deterministic validation issues sorted by path, code, and message.",
        )


class RepositoryValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary_directory = tempfile.TemporaryDirectory()
        self.repo = Path(self._temporary_directory.name)

    def tearDown(self) -> None:
        self._temporary_directory.cleanup()

    def write(self, relative_path: str, content: str) -> Path:
        path = self.repo / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return path

    def document(
        self,
        *,
        document_id: str = "guide",
        title: str = "Guide",
        lang: str = "en",
        audience: str = "developer",
        document_type: str = "explanation",
        status: str = "active",
        owner: str = "project",
        last_reviewed: str = "2026-08-01",
        body: str = '<a id="overview"></a>\n## Overview\n',
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
            f"{body}"
        )

    def write_pair(
        self,
        stem: str = "docs/developers/guide",
        **metadata: str,
    ) -> None:
        self.write(f"{stem}.en.md", self.document(**metadata))
        self.write(
            f"{stem}.zh-TW.md",
            self.document(title="指南", lang="zh-TW", **metadata),
        )

    def test_empty_repository_and_valid_bilingual_pair_pass(self) -> None:
        self.assertEqual(validate_repository(self.repo), [])
        self.write_pair()
        self.assertEqual(validate_repository(self.repo), [])

    @unittest.skipUnless(hasattr(socket, "AF_UNIX"), "requires Unix sockets")
    def test_discovery_boundaries_and_is_file(self) -> None:
        self.write("z/BAD.zh-TW.md", "not frontmatter\n")
        self.write("a/UPPER.en.md", "not frontmatter\n")
        for directory in (
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
            "docs/governance/templates",
        ):
            self.write(f"{directory}/IGNORED.en.md", "not frontmatter\n")
        self.write("WRONG.zh-tw.md", "not frontmatter\n")
        self.write("WRONG.en.MD", "not frontmatter\n")
        self.write("WRONG.md", "not frontmatter\n")
        (self.repo / "DIRECTORY.en.md").mkdir()
        socket_path = self.repo / "SOCKET.en.md"
        unix_socket = socket.socket(socket.AF_UNIX)
        try:
            unix_socket.bind(str(socket_path))
            self.assertEqual(
                validate_repository(self.repo),
                [
                    Issue(Path("a/UPPER.en.md"), "invalid-name", "filename is not canonical"),
                    Issue(Path("z/BAD.zh-TW.md"), "invalid-name", "filename is not canonical"),
                ],
            )
        finally:
            unix_socket.close()

    def test_complete_filename_grammar(self) -> None:
        accepted = {
            "index": "index-doc",
            "simple-guide": "simple-doc",
            "guide-v2": "version-doc",
            "2026-08-01-release-notes": "dated-doc",
            "adr-0001-use-anchors": "adr-doc",
        }
        for stem, document_id in accepted.items():
            self.write_pair(f"component/{stem}", document_id=document_id)

        rejected = (
            "Upper",
            "under_score",
            "has space",
            "two--hyphens",
            "-leading",
            "trailing-",
            "2026-2-30-report",
            "2026-02-30-report",
            "20260801-report",
            "2026-08-report",
            "2026-08-01",
            "2026-08-01--report",
            "123-report",
            "adr-001-report",
            "adr-00001-report",
            "adr-0001",
            "adr-0001-Upper",
            "guide.ZH-tw",
        )
        for stem in rejected:
            self.write(f"invalid/{stem}.en.md", "not parsed\n")

        invalid_issues = [
            issue
            for issue in validate_repository(self.repo)
            if issue.code == "invalid-name"
        ]
        self.assertEqual(len(invalid_issues), len(rejected))
        self.assertEqual(
            {issue.path.name for issue in invalid_issues},
            {f"{stem}.en.md" for stem in rejected},
        )
        self.assertTrue(
            all(issue.message == "filename is not canonical" for issue in invalid_issues)
        )

    def test_frontmatter_boundaries_and_scalar_grammar_do_not_cascade(self) -> None:
        valid = self.document()
        malformed = [
            valid.removeprefix("---\n"),
            "\n" + valid,
            valid.replace("\n---\n\n", "\n", 1),
            valid.replace("title: Guide", "title Guide"),
            valid.replace("title: Guide", ": Guide"),
            valid.replace("title: Guide", "title:"),
            valid.replace("title: Guide", "title:    "),
            valid.replace("title: Guide", "title: [Guide]"),
            valid.replace("title: Guide", "title: {name: Guide}"),
            valid.replace("owner: project", "owner: project\nowner: core"),
        ]
        block_indicators = (
            "|",
            ">",
            "|-",
            "|+",
            ">-",
            ">+",
            "|2",
            ">2",
            "|2-",
            "|-2",
            "|2+",
            "|+2",
            ">2-",
            ">-2",
            ">2+",
            ">+2",
            "|2- # note",
            ">+2 # note",
        )
        malformed.extend(
            valid.replace("title: Guide", f"title: {indicator}")
            for indicator in block_indicators
        )
        for index, content in enumerate(malformed):
            stem = f"frontmatter/case-{index}"
            self.write(f"{stem}.en.md", content)
            self.write(
                f"{stem}.zh-TW.md",
                self.document(
                    document_id=f"frontmatter-case-{index}",
                    title="有效",
                    lang="zh-TW",
                ),
            )

        issues = validate_repository(self.repo)
        self.assertEqual(len(issues), len(malformed))
        self.assertEqual([issue.code for issue in issues], ["frontmatter"] * len(malformed))
        self.assertEqual(
            {issue.path for issue in issues},
            {Path(f"frontmatter/case-{index}.en.md") for index in range(len(malformed))},
        )
        self.assertTrue(all(issue.message == "invalid frontmatter" for issue in issues))

    def test_frontmatter_rejects_leading_yaml_indicators_but_accepts_quoted_scalars(
        self,
    ) -> None:
        invalid_titles = (
            "# comment",
            "!!seq [Guide]",
            "&items [Guide]",
            "!!str |",
            "&body >-",
            "*items",
            "- item",
            "? key",
            ": value",
        )
        expected: list[Issue] = []
        for index, title in enumerate(invalid_titles):
            stem = f"frontmatter/leading-indicator-{index}"
            document_id = f"leading-indicator-{index}"
            self.write(f"{stem}.en.md", self.document(document_id=document_id, title=title))
            self.write(
                f"{stem}.zh-TW.md",
                self.document(document_id=document_id, title="有效", lang="zh-TW"),
            )
            expected.append(
                Issue(Path(f"{stem}.en.md"), "frontmatter", "invalid frontmatter")
            )

        quoted_titles = ("'!!seq [Guide]'", '"&items [Guide]"', "'# comment'")
        for index, title in enumerate(quoted_titles):
            stem = f"frontmatter/quoted-scalar-{index}"
            document_id = f"quoted-scalar-{index}"
            self.write(f"{stem}.en.md", self.document(document_id=document_id, title=title))
            self.write(
                f"{stem}.zh-TW.md",
                self.document(document_id=document_id, title="有效", lang="zh-TW"),
            )

        self.assertEqual(validate_repository(self.repo), expected)

    def test_frontmatter_requires_exact_field_set(self) -> None:
        fields = (
            "id",
            "title",
            "lang",
            "audience",
            "type",
            "status",
            "owner",
            "last_reviewed",
        )
        valid = self.document()
        for field in fields:
            stem = f"fields/missing-{field.replace('_', '-')}"
            lines = [
                line
                for line in valid.splitlines()
                if not line.startswith(f"{field}: ")
            ]
            self.write(f"{stem}.en.md", "\n".join(lines) + "\n")
            self.write(
                f"{stem}.zh-TW.md",
                self.document(document_id=f"valid-{field.replace('_', '-')}", title="有效", lang="zh-TW"),
            )
        self.write(
            "fields/extra-field.en.md",
            valid.replace("owner: project", "owner: project\nextra: unsupported"),
        )
        self.write(
            "fields/extra-field.zh-TW.md",
            self.document(document_id="valid-extra", title="有效", lang="zh-TW"),
        )

        issues = validate_repository(self.repo)
        self.assertEqual(len(issues), len(fields) + 1)
        self.assertEqual([issue.code for issue in issues], ["invalid-metadata"] * 9)
        self.assertTrue(
            all(issue.message == "metadata fields must match schema" for issue in issues)
        )

    def test_all_enums_and_status_mappings(self) -> None:
        status_by_type = {
            "index": ("draft", "active", "deprecated"),
            "tutorial": ("draft", "active", "deprecated"),
            "how-to": ("draft", "active", "deprecated"),
            "reference": ("draft", "active", "deprecated"),
            "explanation": ("draft", "active", "deprecated"),
            "safety": ("draft", "active", "deprecated"),
            "troubleshooting": ("draft", "active", "deprecated"),
            "decision": ("accepted", "superseded"),
            "design": ("proposed", "approved", "implemented", "rejected", "superseded"),
            "plan": ("draft", "active", "blocked", "completed", "cancelled"),
            "roadmap": ("active",),
            "release": ("published",),
            "experiment-summary": ("published",),
            "audit": ("published",),
        }
        counter = 0
        for document_type, statuses in status_by_type.items():
            for status in statuses:
                self.write_pair(
                    f"component/valid-{counter}",
                    document_id=f"valid-{counter}",
                    document_type=document_type,
                    status=status,
                )
                counter += 1
        for audience in ("operator", "developer", "shared"):
            self.write_pair(
                f"component/audience-{audience}",
                document_id=f"audience-{audience}",
                audience=audience,
            )
        for owner in (
            "project",
            "core",
            "training",
            "panel",
            "deployment",
            "sim2real",
            "reward-agent",
        ):
            self.write_pair(
                f"component/owner-{owner}",
                document_id=f"owner-{owner}",
                owner=owner,
            )

        def write_invalid_pair(stem: str, **overrides: str) -> None:
            en_values = dict(overrides)
            zh_values = dict(overrides)
            if "lang" not in overrides:
                en_values["lang"] = "en"
                zh_values["lang"] = "zh-TW"
            self.write(
                f"component/{stem}.en.md",
                self.document(document_id=stem, title="Invalid", **en_values),
            )
            self.write(
                f"component/{stem}.zh-TW.md",
                self.document(document_id=stem, title="無效", **zh_values),
            )

        write_invalid_pair("invalid-lang", lang="fr")
        write_invalid_pair("invalid-audience", audience="robot")
        write_invalid_pair("invalid-type", document_type="unknown")
        write_invalid_pair("invalid-status", status="unknown")
        write_invalid_pair("invalid-owner", owner="nobody")
        invalid_status = {
            "index": "published",
            "tutorial": "published",
            "how-to": "published",
            "reference": "published",
            "explanation": "published",
            "safety": "published",
            "troubleshooting": "published",
            "decision": "active",
            "design": "active",
            "plan": "accepted",
            "roadmap": "draft",
            "release": "active",
            "experiment-summary": "active",
            "audit": "active",
        }
        for document_type, status in invalid_status.items():
            stem = f"invalid-status-{document_type}"
            write_invalid_pair(stem, document_type=document_type, status=status)

        issues = validate_repository(self.repo)
        self.assertEqual(len(issues), 2 * (5 + len(invalid_status)))
        self.assertEqual({issue.code for issue in issues}, {"invalid-metadata"})
        self.assertTrue(all(issue.message == "metadata value is invalid" for issue in issues))

    def test_ids_dates_and_filename_locale_identity(self) -> None:
        self.write_pair(
            "component/identity-valid",
            document_id="identity-valid-v2",
            last_reviewed="2024-02-29",
        )
        invalid_ids = (
            "Upper",
            "under_score",
            "has space",
            "two--hyphens",
            "-leading",
            "trailing-",
            "123-start",
        )
        for index, document_id in enumerate(invalid_ids):
            stem = f"bad-id-{index}"
            self.write(
                f"component/{stem}.en.md",
                self.document(document_id=document_id),
            )
            self.write(
                f"component/{stem}.zh-TW.md",
                self.document(document_id=document_id, title="無效", lang="zh-TW"),
            )
        invalid_dates = ("2026-8-1", "2026-02-30", "2026-08-01T00:00:00", "not-a-date")
        for index, reviewed in enumerate(invalid_dates):
            stem = f"bad-date-{index}"
            self.write(
                f"component/{stem}.en.md",
                self.document(document_id=stem, last_reviewed=reviewed),
            )
            self.write(
                f"component/{stem}.zh-TW.md",
                self.document(
                    document_id=stem,
                    title="無效",
                    lang="zh-TW",
                    last_reviewed=reviewed,
                ),
            )
        self.write(
            "component/bad-locale.en.md",
            self.document(document_id="bad-locale", lang="zh-TW"),
        )
        self.write(
            "component/bad-locale.zh-TW.md",
            self.document(document_id="bad-locale", title="無效", lang="en"),
        )

        issues = validate_repository(self.repo)
        self.assertEqual(len(issues), 2 * (len(invalid_ids) + len(invalid_dates) + 1))
        self.assertEqual({issue.code for issue in issues}, {"invalid-metadata"})
        self.assertEqual(
            {issue.message for issue in issues},
            {
                "id must be lowercase kebab-case",
                "last_reviewed must be a real ISO date",
                "filename locale does not match lang",
            },
        )

    def test_exhaustive_central_location_matrix(self) -> None:
        section_rules = {
            "operators": (
                "operator",
                ("tutorial", "how-to", "reference", "explanation", "safety", "troubleshooting"),
            ),
            "developers": (
                "developer",
                ("tutorial", "how-to", "reference", "explanation", "safety", "troubleshooting"),
            ),
            "reference": ("shared", ("reference",)),
            "decisions": ("developer", ("decision",)),
            "designs": ("developer", ("design",)),
            "plans": ("developer", ("plan",)),
            "roadmap": ("shared", ("roadmap",)),
            "releases": ("shared", ("release",)),
            "research": ("developer", ("experiment-summary", "audit", "explanation")),
            "governance": ("developer", ("reference",)),
        }
        valid_status = {
            "index": "active",
            "tutorial": "active",
            "how-to": "active",
            "reference": "active",
            "explanation": "active",
            "safety": "active",
            "troubleshooting": "active",
            "decision": "accepted",
            "design": "proposed",
            "plan": "active",
            "roadmap": "active",
            "release": "published",
            "experiment-summary": "published",
            "audit": "published",
        }
        scenarios: list[tuple[str, str, str, str, bool]] = [
            ("root portal valid", "docs/index", "shared", "index", True),
            ("root portal invalid audience", "docs/index", "developer", "index", False),
            ("root portal invalid type", "docs/index", "shared", "reference", False),
        ]
        for section, (audience, allowed_types) in section_rules.items():
            other_audience = "shared" if audience != "shared" else "developer"
            scenarios.extend(
                [
                    (f"{section} portal valid", f"docs/{section}/index", audience, "index", True),
                    (
                        f"{section} portal invalid audience",
                        f"docs/{section}/index",
                        other_audience,
                        "index",
                        False,
                    ),
                    (
                        f"{section} portal invalid type",
                        f"docs/{section}/index",
                        audience,
                        "reference",
                        False,
                    ),
                ]
            )
            for document_type in allowed_types:
                scenarios.append(
                    (
                        f"{section} nonportal valid {document_type}",
                        f"docs/{section}/guide",
                        audience,
                        document_type,
                        True,
                    )
                )
            invalid_type = "reference" if section in {"decisions", "designs", "plans", "roadmap", "releases", "research"} else "plan"
            scenarios.extend(
                [
                    (
                        f"{section} nonportal invalid audience",
                        f"docs/{section}/guide",
                        other_audience,
                        allowed_types[0],
                        False,
                    ),
                    (
                        f"{section} nonportal invalid type",
                        f"docs/{section}/guide",
                        audience,
                        invalid_type,
                        False,
                    ),
                    (
                        f"{section} nested index",
                        f"docs/{section}/nested/index",
                        audience,
                        "index",
                        False,
                    ),
                    (
                        f"{section} non-index portal type",
                        f"docs/{section}/not-portal",
                        audience,
                        "index",
                        False,
                    ),
                ]
            )
        scenarios.extend(
            [
                ("direct docs non-index", "docs/guide", "shared", "reference", False),
                ("unknown docs section", "docs/unknown/guide", "developer", "explanation", False),
                ("colocated exception", "source/component/guide", "shared", "release", True),
            ]
        )

        mismatches: list[str] = []
        for label, stem, audience, document_type, valid in scenarios:
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                for suffix, lang, title in (("en", "en", "Guide"), ("zh-TW", "zh-TW", "指南")):
                    path = root / f"{stem}.{suffix}.md"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(
                        self.document(
                            document_id="location-case",
                            title=title,
                            lang=lang,
                            audience=audience,
                            document_type=document_type,
                            status=valid_status[document_type],
                        ),
                        encoding="utf-8",
                    )
                expected = []
                if not valid:
                    expected = [
                        Issue(
                            Path(f"{stem}.en.md"),
                            "invalid-location",
                            "metadata does not match document location",
                        ),
                        Issue(
                            Path(f"{stem}.zh-TW.md"),
                            "invalid-location",
                            "metadata does not match document location",
                        ),
                    ]
                if validate_repository(root) != expected:
                    mismatches.append(label)
        self.assertEqual(mismatches, [])

    def test_physical_pair_presence_is_independent_of_content(self) -> None:
        self.write(
            "component/lonely.en.md",
            self.document(document_id="lonely"),
        )
        self.write(
            "component/present.en.md",
            self.document(document_id="present"),
        )
        self.write(
            "component/present.zh-TW.md",
            self.document(
                document_id="present",
                title="存在",
                lang="zh-TW",
                owner="nobody",
            ),
        )

        issues = sorted(
            validate_repository(self.repo),
            key=lambda issue: (issue.path.as_posix(), issue.code, issue.message),
        )
        self.assertEqual(
            issues,
            [
                Issue(
                    Path("component/lonely.en.md"),
                    "missing-pair",
                    "missing locale companion: zh-TW",
                ),
                Issue(
                    Path("component/present.zh-TW.md"),
                    "invalid-metadata",
                    "metadata value is invalid",
                ),
            ],
        )

    def test_pair_metadata_is_once_localized_and_location_independent(self) -> None:
        self.write_pair(
            "component/localized",
            document_id="localized",
        )
        self.write(
            "docs/unknown/drift.en.md",
            self.document(document_id="drift", owner="project"),
        )
        self.write(
            "docs/unknown/drift.zh-TW.md",
            self.document(
                document_id="drift",
                title="中文標題",
                lang="zh-TW",
                owner="core",
            ),
        )

        issues = sorted(
            validate_repository(self.repo),
            key=lambda issue: (issue.path.as_posix(), issue.code, issue.message),
        )
        self.assertEqual(
            issues,
            [
                Issue(
                    Path("docs/unknown/drift.en.md"),
                    "invalid-location",
                    "metadata does not match document location",
                ),
                Issue(
                    Path("docs/unknown/drift.en.md"),
                    "pair-metadata",
                    "locale pair metadata differs",
                ),
                Issue(
                    Path("docs/unknown/drift.zh-TW.md"),
                    "invalid-location",
                    "metadata does not match document location",
                ),
            ],
        )

    def test_duplicate_ids_across_all_eligible_logical_documents(self) -> None:
        self.write_pair("component/second-a", document_id="second-shared")
        self.write(
            "component/second-b.en.md",
            self.document(document_id="second-own"),
        )
        self.write(
            "component/second-b.zh-TW.md",
            self.document(document_id="second-shared", title="重複", lang="zh-TW"),
        )
        self.write(
            "component/incomplete-a.en.md",
            self.document(document_id="incomplete-shared"),
        )
        self.write(
            "component/incomplete-b.en.md",
            self.document(document_id="incomplete-shared"),
        )
        self.write_pair("docs/unknown/location-a", document_id="location-shared")
        self.write_pair("docs/unknown/location-b", document_id="location-shared")
        for stem in ("ignored-a", "ignored-b"):
            self.write(
                f"component/{stem}.en.md",
                self.document(document_id="ignored-shared", owner="nobody"),
            )
            self.write(
                f"component/{stem}.zh-TW.md",
                self.document(
                    document_id="ignored-shared",
                    title="忽略",
                    lang="zh-TW",
                    owner="nobody",
                ),
            )

        issues = validate_repository(self.repo)
        code_counts = {
            code: sum(issue.code == code for issue in issues)
            for code in {issue.code for issue in issues}
        }
        self.assertEqual(
            code_counts,
            {
                "duplicate-id": 3,
                "invalid-location": 4,
                "invalid-metadata": 4,
                "missing-pair": 2,
                "pair-metadata": 1,
            },
        )
        duplicates = sorted(
            (issue for issue in issues if issue.code == "duplicate-id"),
            key=lambda issue: issue.path.as_posix(),
        )
        self.assertEqual(
            duplicates,
            [
                Issue(
                    Path("component/incomplete-a.en.md"),
                    "duplicate-id",
                    "id reused by multiple logical documents: incomplete-shared",
                ),
                Issue(
                    Path("component/second-a.en.md"),
                    "duplicate-id",
                    "id reused by multiple logical documents: second-shared",
                ),
                Issue(
                    Path("docs/unknown/location-a.en.md"),
                    "duplicate-id",
                    "id reused by multiple logical documents: location-shared",
                ),
            ],
        )

    def test_heading_anchor_structure_and_independent_markdown_eligibility(self) -> None:
        valid_body = '<a id="overview"></a>\n## Overview\n'

        def write_bodies(stem: str, en_body: str, zh_body: str = valid_body) -> None:
            self.write(
                f"{stem}.en.md",
                self.document(document_id=stem.rsplit("/", 1)[-1], body=en_body),
            )
            self.write(
                f"{stem}.zh-TW.md",
                self.document(
                    document_id=stem.rsplit("/", 1)[-1],
                    title="標題",
                    lang="zh-TW",
                    body=zh_body,
                ),
            )

        write_bodies("component/valid-anchor", valid_body)
        write_bodies("component/missing-anchor", "## Missing\n")
        write_bodies(
            "component/invalid-anchor",
            '<a id="Bad_Anchor"></a>\n## Invalid\n',
        )
        write_bodies(
            "component/standalone-anchor",
            '<a id="orphan"></a>\nParagraph only.\n',
        )
        write_bodies(
            "component/duplicate-anchor",
            '<a id="same"></a>\n## One\n<a id="same"></a>\n## Two\n',
            '<a id="one"></a>\n## One\n<a id="two"></a>\n## Two\n',
        )
        self.write(
            "docs/unknown/independent.en.md",
            self.document(document_id="independent", owner="nobody", body="## Missing\n"),
        )
        self.write(
            "docs/unknown/independent.zh-TW.md",
            self.document(
                document_id="independent",
                title="獨立",
                lang="zh-TW",
                body=valid_body,
            ),
        )

        issues = validate_repository(self.repo)
        anchor_issues = sorted(
            (
                issue
                for issue in issues
                if issue.code in {"heading-anchor", "duplicate-anchor"}
            ),
            key=lambda issue: (issue.path.as_posix(), issue.code, issue.message),
        )
        self.assertEqual(
            anchor_issues,
            [
                Issue(
                    Path("component/duplicate-anchor.en.md"),
                    "duplicate-anchor",
                    "duplicate explicit anchor: same",
                ),
                Issue(
                    Path("component/invalid-anchor.en.md"),
                    "heading-anchor",
                    "invalid explicit anchor: Bad_Anchor",
                ),
                Issue(
                    Path("component/missing-anchor.en.md"),
                    "heading-anchor",
                    "heading lacks preceding explicit anchor",
                ),
                Issue(
                    Path("component/standalone-anchor.en.md"),
                    "heading-anchor",
                    "explicit anchor is not immediately followed by a heading: orphan",
                ),
                Issue(
                    Path("docs/unknown/independent.en.md"),
                    "heading-anchor",
                    "heading lacks preceding explicit anchor",
                ),
            ],
        )
        self.assertEqual(sum(issue.code == "invalid-metadata" for issue in issues), 1)
        self.assertEqual(sum(issue.code == "invalid-location" for issue in issues), 1)

    def test_pair_anchor_sequences_are_independent_except_for_structure(self) -> None:
        def body(anchor: str, link: str = "") -> str:
            return f'<a id="{anchor}"></a>\n## Heading\n{link}'

        self.write(
            "component/mismatch.en.md",
            self.document(document_id="mismatch", body=body("english")),
        )
        self.write(
            "component/mismatch.zh-TW.md",
            self.document(
                document_id="mismatch",
                title="不同",
                lang="zh-TW",
                body=body("chinese"),
            ),
        )
        self.write(
            "docs/unknown/independent-anchors.en.md",
            self.document(
                document_id="independent-anchors",
                owner="nobody",
                body=body("english"),
            ),
        )
        self.write(
            "docs/unknown/independent-anchors.zh-TW.md",
            self.document(
                document_id="independent-anchors",
                title="獨立",
                lang="zh-TW",
                body=body("chinese"),
            ),
        )
        self.write(
            "component/link-anchors.en.md",
            self.document(
                document_id="link-anchors",
                body=body("english", "[missing](missing.txt)\n"),
            ),
        )
        self.write(
            "component/link-anchors.zh-TW.md",
            self.document(
                document_id="link-anchors",
                title="連結",
                lang="zh-TW",
                body=body("chinese"),
            ),
        )
        self.write(
            "component/structure-ineligible.en.md",
            self.document(document_id="structure-ineligible", body="## Missing\n"),
        )
        self.write(
            "component/structure-ineligible.zh-TW.md",
            self.document(
                document_id="structure-ineligible",
                title="結構",
                lang="zh-TW",
                body=body("other"),
            ),
        )

        issues = validate_repository(self.repo)
        pair_issues = sorted(
            (issue for issue in issues if issue.code == "pair-anchors"),
            key=lambda issue: issue.path.as_posix(),
        )
        self.assertEqual(
            pair_issues,
            [
                Issue(
                    Path("component/link-anchors.en.md"),
                    "pair-anchors",
                    "locale pair anchor sequences differ",
                ),
                Issue(
                    Path("component/mismatch.en.md"),
                    "pair-anchors",
                    "locale pair anchor sequences differ",
                ),
                Issue(
                    Path("docs/unknown/independent-anchors.en.md"),
                    "pair-anchors",
                    "locale pair anchor sequences differ",
                ),
            ],
        )
        self.assertEqual(
            sum(
                issue.path == Path("component/structure-ineligible.en.md")
                and issue.code == "heading-anchor"
                for issue in issues
            ),
            1,
        )

    def test_fences_preserve_source_lines_and_hide_all_markdown_syntax(self) -> None:
        visible = '<a id="visible"></a>\n## Visible\n'

        def pair(stem: str, en_body: str, zh_body: str | None = None) -> None:
            self.write(
                f"component/{stem}.en.md",
                self.document(document_id=stem, body=en_body),
            )
            self.write(
                f"component/{stem}.zh-TW.md",
                self.document(
                    document_id=stem,
                    title="範例",
                    lang="zh-TW",
                    body=zh_body if zh_body is not None else en_body,
                ),
            )

        pair(
            "fence-barrier",
            '<a id="before"></a>\n```text\nignored\n```\n## After fence\n',
            visible,
        )
        ignored_syntax = (
            "```markdown\n"
            "## Hidden heading\n"
            '<a id="Bad_Anchor"></a>\n'
            "![image](missing-image.png)\n"
            "[inline](missing-inline.txt)\n"
            "[full][missing-full]\n"
            "[collapsed][]\n"
            "[shortcut]\n"
            "[missing-full]: missing-definition.txt\n"
            "```\n"
            + visible
        )
        pair("fenced-all-syntax", ignored_syntax)
        pair(
            "backtick-marker",
            "```text\n~~~\n## Hidden\n<a id=\"hidden\"></a>\n```\n" + visible,
        )
        pair(
            "tilde-marker-length",
            "~~~~ text\n~~~\n````\n## Hidden\n<a id=\"hidden\"></a>\n~~~~~\n" + visible,
        )
        pair(
            "three-space-fence",
            "   ```text\n## Hidden\n<a id=\"hidden\"></a>\n   ```\n" + visible,
        )
        pair(
            "four-space-not-fence",
            "    ```text\n## Missing\n    ```\n",
            visible,
        )

        anchor_issues = sorted(
            (
                issue
                for issue in validate_repository(self.repo)
                if issue.code in {"heading-anchor", "duplicate-anchor"}
            ),
            key=lambda issue: (issue.path.as_posix(), issue.code, issue.message),
        )
        self.assertEqual(
            anchor_issues,
            [
                Issue(
                    Path("component/fence-barrier.en.md"),
                    "heading-anchor",
                    "explicit anchor is not immediately followed by a heading: before",
                ),
                Issue(
                    Path("component/fence-barrier.en.md"),
                    "heading-anchor",
                    "heading lacks preceding explicit anchor",
                ),
                Issue(
                    Path("component/four-space-not-fence.en.md"),
                    "heading-anchor",
                    "heading lacks preceding explicit anchor",
                ),
            ],
        )

    def test_inline_link_and_image_destination_scanner(self) -> None:
        from tools.documentation.validator import _extract_inline_destinations

        lines = (
            "[plain](plain.txt)",
            "![image](image.png)",
            "[angle](<target file.txt> \"angle title\")",
            "[balanced](folder/(thing).md 'single title')",
            "[title-parens](target.md \"title (with parentheses)\")",
            "text](not-a-link.txt)",
            r"\[escaped](not-a-link.txt)",
            "[malformed(not-a-link.txt)",
            "```markdown",
            "[fenced](not-a-link.txt)",
            "```",
        )
        self.assertEqual(
            _extract_inline_destinations(lines),
            [
                "plain.txt",
                "image.png",
                "target file.txt",
                "folder/(thing).md",
                "target.md",
            ],
        )

    def test_reference_link_and_image_scanner(self) -> None:
        from tools.documentation.validator import _extract_reference_destinations

        lines = (
            '[Full Label]: full.txt "double title"',
            "[collapsed label]: <collapsed file.txt> 'single title'",
            "[shortcut]: shortcut.txt (parenthesized title)",
            "[Image Label]: image.png",
            "[unused]: unused.txt",
            "[text][  FULL   LABEL ]",
            "![alt][image label]",
            "[Collapsed   Label][]",
            "![collapsed label][]",
            "[SHORTCUT]",
            "![shortcut]",
            "[text][missing full]",
            "![alt][missing image]",
            "[missing collapsed][]",
            "[plain unmatched shortcut]",
            r"\[escaped][missing escaped]",
            "[inline text](inline.txt)",
            "```markdown",
            "[fenced][missing fenced]",
            "[fenced]: fenced.txt",
            "```",
        )
        self.assertEqual(
            _extract_reference_destinations(lines),
            (
                [
                    "full.txt",
                    "collapsed file.txt",
                    "shortcut.txt",
                    "image.png",
                    "unused.txt",
                ],
                ["missing full", "missing image", "missing collapsed"],
            ),
        )

    def test_public_local_target_validation_complete_raw_semantics(self) -> None:
        self.write("component/existing.txt", "exists\n")
        self.write("component/target file.txt", "exists\n")
        self.write("component/dir/(thing).txt", "exists\n")
        self.write("component/literal#name.txt", "exists\n")
        self.write(
            "component/anchor#target.md",
            '<a id="present"></a>\n## Target\n',
        )
        self.write(
            "component/target.md",
            '<a id="present"></a>\n## Present\n```html\n<a id="fenced"></a>\n```\n',
        )
        self.write("component/target.txt", '<a id="present"></a>\n')
        (self.repo / "component/directory").mkdir(parents=True)

        broken_destinations = [
            "z-missing.txt",
            "a-missing.txt",
            "missing%3Afile.txt",
            "%68ttps%3Afoo",
            "image-missing.png",
            "../../escape.txt",
            "/absolute/path",
            "/docs/page",
            "C:/secret/file.md",
            r"C:\secret\file.md",
            r"\\server\share\file.md",
            "%2Fdecoded/path",
            "C%3A/decoded/file.md",
            "C%3A%5Cdecoded%5Cfile.md",
            "%5C%5Cserver%5Cshare%5Cfile.md",
            "bad%00.txt",
            "directory",
            "target.txt#present",
            "reference-missing.txt",
        ]
        source_lines = [
            '<a id="present"></a>',
            "## Present",
            "[literal hash](literal%23name.txt)",
            "[encoded colon](missing%3Afile.txt)",
            "[encoded hash fragment](anchor%23target.md#pre%73ent)",
            "[current fragment](#pre%73ent)",
            "[markdown fragment](target.md#present)",
            "[missing anchor](target.md#absent)",
            "[fenced anchor](target.md#fenced)",
            "[nonmarkdown fragment](target.txt#present)",
            "[angle](<target file.txt>)",
            "[balanced](dir/(thing).txt \"title (parentheses)\")",
            "[title](existing.txt 'optional title')",
            "![missing image](image-missing.png)",
            "[z first](z-missing.txt)",
            "[a second](a-missing.txt)",
            "[decoded scheme remains local](%68ttps%3Afoo)",
            "[escape](../../escape.txt)",
            "[posix absolute](/absolute/path)",
            "[site absolute](/docs/page)",
            "[drive slash](C:/secret/file.md)",
            r"[drive backslash](C:\secret\file.md)",
            r"[unc](\\server\share\file.md)",
            "[decoded posix](%2Fdecoded/path)",
            "[decoded drive slash](C%3A/decoded/file.md)",
            "[decoded drive backslash](C%3A%5Cdecoded%5Cfile.md)",
            "[decoded unc](%5C%5Cserver%5Cshare%5Cfile.md)",
            "[nul](bad%00.txt)",
            "[directory](directory)",
            "[http](http://example.com)",
            "[https](https://example.com)",
            "[mail](mailto:docs@example.com)",
            "[custom](custom:thing)",
            "[good reference][good ref]",
            "[Good Ref][]",
            "[GOOD REF]",
            "[text][missing definition]",
            "![alt][missing image definition]",
            "[collapsed missing][]",
            "[plain unmatched shortcut]",
            r"\[escaped](ignored-missing.txt)",
            "plain text](ignored-missing.txt)",
            '[good ref]: existing.txt "title"',
            "[angle ref]: <target file.txt> 'title'",
            "[balanced ref]: dir/(thing).txt (title)",
            "[bad ref]: reference-missing.txt",
            "```markdown",
            "[fenced missing](ignored-missing.txt)",
            "[fenced ref]: ignored-missing.txt",
            "```",
        ]
        self.write(
            "component/source.en.md",
            self.document(document_id="source", body="\n".join(source_lines) + "\n"),
        )
        self.write(
            "component/source.zh-TW.md",
            self.document(
                document_id="source",
                title="來源",
                lang="zh-TW",
                body='<a id="present"></a>\n## Present\n',
            ),
        )
        self.write(
            "docs/unknown/independent-links.en.md",
            self.document(
                document_id="independent-links",
                owner="nobody",
                body='<a id="same"></a>\n## Same\n[missing](independent-missing.txt)\n',
            ),
        )
        self.write(
            "docs/unknown/independent-links.zh-TW.md",
            self.document(
                document_id="independent-links",
                title="獨立連結",
                lang="zh-TW",
                body='<a id="same"></a>\n## Same\n',
            ),
        )

        expected = [
            Issue(
                Path("component/source.en.md"),
                "broken-link",
                f"invalid local link target: {destination}",
            )
            for destination in broken_destinations
        ]
        expected.extend(
            [
                Issue(
                    Path("component/source.en.md"),
                    "missing-link-anchor",
                    "missing explicit anchor: absent",
                ),
                Issue(
                    Path("component/source.en.md"),
                    "missing-link-anchor",
                    "missing explicit anchor: fenced",
                ),
                Issue(
                    Path("component/source.en.md"),
                    "broken-link",
                    "missing reference definition: missing definition",
                ),
                Issue(
                    Path("component/source.en.md"),
                    "broken-link",
                    "missing reference definition: missing image definition",
                ),
                Issue(
                    Path("component/source.en.md"),
                    "broken-link",
                    "missing reference definition: collapsed missing",
                ),
                Issue(
                    Path("docs/unknown/independent-links.en.md"),
                    "broken-link",
                    "invalid local link target: independent-missing.txt",
                ),
            ]
        )
        actual = [
            issue
            for issue in validate_repository(self.repo)
            if issue.code in {"broken-link", "missing-link-anchor"}
        ]
        key = lambda issue: (issue.path.as_posix(), issue.code, issue.message)
        self.assertEqual(sorted(actual, key=key), sorted(expected, key=key))

    def test_issue_sort_and_dedup_uses_path_code_and_message(self) -> None:
        self.maxDiff = None
        self.write(
            "a/order.en.md",
            self.document(
                document_id="order",
                owner="nobody",
                body=(
                    "## Missing anchor\n"
                    "[z first](z-missing.txt)\n"
                    "[a second](a-missing.txt)\n"
                    "[duplicate one](same-missing.txt)\n"
                    "[duplicate two](same-missing.txt)\n"
                ),
            ),
        )
        self.write(
            "a/order.zh-TW.md",
            self.document(
                document_id="order",
                title="排序",
                lang="zh-TW",
                body='<a id="valid"></a>\n## Valid\n',
            ),
        )
        self.write("z-order/bad.en.md", "not frontmatter\n")
        self.write(
            "z-order/bad.zh-TW.md",
            self.document(document_id="bad", title="有效", lang="zh-TW"),
        )

        self.assertEqual(
            validate_repository(self.repo),
            [
                Issue(
                    Path("a/order.en.md"),
                    "broken-link",
                    "invalid local link target: a-missing.txt",
                ),
                Issue(
                    Path("a/order.en.md"),
                    "broken-link",
                    "invalid local link target: same-missing.txt",
                ),
                Issue(
                    Path("a/order.en.md"),
                    "broken-link",
                    "invalid local link target: z-missing.txt",
                ),
                Issue(
                    Path("a/order.en.md"),
                    "heading-anchor",
                    "heading lacks preceding explicit anchor",
                ),
                Issue(
                    Path("a/order.en.md"),
                    "invalid-metadata",
                    "metadata value is invalid",
                ),
                Issue(
                    Path("z-order/bad.en.md"),
                    "frontmatter",
                    "invalid frontmatter",
                ),
            ],
        )

    def test_actual_current_repository_is_valid(self) -> None:
        repo_root = Path(__file__).resolve().parents[3]
        self.assertEqual(validate_repository(repo_root), [])


if __name__ == "__main__":
    unittest.main()
