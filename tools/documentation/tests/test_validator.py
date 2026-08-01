from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from tools.documentation.validator import validate_repository


METADATA = {
    "id": "sample-guide",
    "title": "Sample Guide",
    "lang": "en",
    "audience": "developer",
    "type": "explanation",
    "status": "active",
    "owner": "core",
    "last_reviewed": "2026-08-01",
}


def write_document(root: Path, relative: str, **overrides: str) -> Path:
    metadata = {**METADATA, **overrides}
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = "\n".join(f"{key}: {value}" for key, value in metadata.items())
    path.write_text(
        f"---\n{fields}\n---\n\n<a id=\"overview\"></a>\n## Overview\n",
        encoding="utf-8",
    )
    return path


def write_raw(root: Path, relative: str, content: str) -> Path:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def issue_codes(root: Path) -> list[str]:
    return [issue.code for issue in validate_repository(root)]


class ValidateRepositoryTests(unittest.TestCase):
    def test_valid_paired_repository_has_no_issues(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            write_document(root, "guides/sample-guide.en.md")
            write_document(
                root,
                "guides/sample-guide.zh-TW.md",
                title="範例指南",
                lang="zh-TW",
            )

            self.assertEqual(validate_repository(root), [])

    def test_invalid_filename_forms_fail(self) -> None:
        for filename in (
            "guides/Bad.en.md",
            "guides/bad_name.en.md",
            "guides/bad--name.en.md",
            "guides/2026-02-30-report.en.md",
            "guides/adr-001-short.en.md",
            "guides/sample.zh-tw.md",
        ):
            with self.subTest(filename=filename), TemporaryDirectory() as directory:
                root = Path(directory)
                write_document(root, filename)
                self.assertIn("invalid-name", issue_codes(root))

    def test_missing_malformed_extra_and_duplicate_frontmatter_fail(self) -> None:
        cases = (
            "# No frontmatter\n",
            "---\nid: sample\ntitle:\n---\n",
            "---\nid: sample\nid: again\n---\n",
            "---\nid: sample\nunknown: value\n---\n",
            "---\nid: [sample]\n---\n",
        )
        for content in cases:
            with self.subTest(content=content), TemporaryDirectory() as directory:
                root = Path(directory)
                write_raw(root, "guides/sample.en.md", content)
                self.assertTrue(issue_codes(root))
                self.assertEqual(issue_codes(root)[0], "frontmatter")

    def test_invalid_metadata_values_and_filename_locale_mismatch_fail(self) -> None:
        invalid_overrides = (
            {"audience": "reader"},
            {"type": "memo"},
            {"status": "published"},
            {"owner": "someone"},
            {"id": "Not kebab"},
            {"last_reviewed": "2026-02-30"},
            {"last_reviewed": "20260801"},
            {"lang": "zh-TW"},
        )
        for overrides in invalid_overrides:
            with self.subTest(overrides=overrides), TemporaryDirectory() as directory:
                root = Path(directory)
                write_document(root, "guides/sample-guide.en.md", **overrides)
                self.assertIn("invalid-metadata", issue_codes(root))

    def test_central_location_mismatch_fails(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            write_document(root, "docs/operators/guide.en.md")
            self.assertIn("invalid-location", issue_codes(root))

    def test_missing_pair_fails(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            write_document(root, "guides/sample-guide.en.md")
            self.assertIn("missing-pair", issue_codes(root))

    def test_pair_metadata_drift_fails_but_localized_fields_may_differ(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            write_document(root, "guides/sample-guide.en.md")
            write_document(
                root,
                "guides/sample-guide.zh-TW.md",
                title="範例指南",
                lang="zh-TW",
                owner="project",
            )
            self.assertIn("pair-metadata", issue_codes(root))

    def test_id_reused_for_two_logical_pairs_fails(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            for stem in ("first-guide", "second-guide"):
                write_document(root, f"guides/{stem}.en.md")
                write_document(root, f"guides/{stem}.zh-TW.md", title="指南", lang="zh-TW")
            self.assertIn("duplicate-id", issue_codes(root))

    def test_heading_anchors_are_required_valid_and_unique(self) -> None:
        cases = (
            "---\n" + "\n".join(f"{key}: {value}" for key, value in METADATA.items()) + "\n---\n## Missing\n",
            "---\n" + "\n".join(f"{key}: {value}" for key, value in METADATA.items()) + "\n---\n<a id=\"Bad_anchor\"></a>\n## Invalid\n",
            "---\n" + "\n".join(f"{key}: {value}" for key, value in METADATA.items()) + "\n---\n<a id=\"same\"></a>\n## One\n<a id=\"same\"></a>\n## Two\n",
        )
        for content in cases:
            with self.subTest(content=content), TemporaryDirectory() as directory:
                root = Path(directory)
                write_raw(root, "guides/sample-guide.en.md", content)
                codes = issue_codes(root)
                self.assertTrue({"heading-anchor", "duplicate-anchor"} & set(codes))

    def test_pair_anchor_sequences_must_match(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            write_document(root, "guides/sample-guide.en.md")
            chinese = write_document(root, "guides/sample-guide.zh-TW.md", title="指南", lang="zh-TW")
            chinese.write_text(chinese.read_text(encoding="utf-8").replace("overview", "different"), encoding="utf-8")
            self.assertIn("pair-anchors", issue_codes(root))

    def test_fenced_headings_anchors_and_links_are_ignored(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            for locale, title in (("en", "Sample Guide"), ("zh-TW", "範例指南")):
                path = write_document(root, f"guides/sample-guide.{locale}.md", title=title, lang=locale)
                path.write_text(
                    path.read_text(encoding="utf-8") + "\n```markdown\n## Not a heading\n<a id=\"Bad_anchor\"></a>\n[bad](missing.md)\n```\n",
                    encoding="utf-8",
                )
            self.assertEqual(validate_repository(root), [])

    def test_missing_absolute_and_outside_links_fail(self) -> None:
        targets = ("missing.md", "/docs/index.en.md", "../../outside.md")
        for target in targets:
            with self.subTest(target=target), TemporaryDirectory() as directory:
                root = Path(directory)
                for locale, title in (("en", "Sample Guide"), ("zh-TW", "範例指南")):
                    path = write_document(root, f"guides/sample-guide.{locale}.md", title=title, lang=locale)
                    path.write_text(path.read_text(encoding="utf-8") + f"\n[bad]({target})\n", encoding="utf-8")
                self.assertIn("broken-link", issue_codes(root))

    def test_local_fragment_requires_explicit_target_anchor(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            write_document(root, "guides/target.en.md", id="target", title="Target")
            write_document(root, "guides/target.zh-TW.md", id="target", title="目標", lang="zh-TW")
            source = write_document(root, "guides/sample-guide.en.md")
            write_document(root, "guides/sample-guide.zh-TW.md", title="範例指南", lang="zh-TW")
            source.write_text(source.read_text(encoding="utf-8") + "\n[target](target.en.md#missing)\n", encoding="utf-8")
            self.assertIn("missing-link-anchor", issue_codes(root))
            source.write_text(source.read_text(encoding="utf-8").replace("#missing", "#overview"), encoding="utf-8")
            self.assertNotIn("missing-link-anchor", issue_codes(root))

    def test_local_fragment_may_target_noncanonical_markdown(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            write_raw(root, "other.md", "<a id=\"target\"></a>\n# Other\n")
            for locale, title in (("en", "Sample Guide"), ("zh-TW", "範例指南")):
                source = write_document(root, f"guides/sample-guide.{locale}.md", title=title, lang=locale)
                source.write_text(source.read_text(encoding="utf-8") + "\n[target](../other.md#target)\n", encoding="utf-8")
            self.assertNotIn("missing-link-anchor", issue_codes(root))

    def test_issue_order_is_deterministic_and_frontmatter_does_not_cascade(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            write_raw(root, "z/bad.en.md", "not frontmatter\n")
            write_document(root, "a/sample.en.md")
            issues = validate_repository(root)
            self.assertEqual(issues, sorted(issues, key=lambda issue: (str(issue.path), issue.code, issue.message)))
            bad_issues = [issue for issue in issues if issue.path == Path("z/bad.en.md")]
            self.assertEqual([issue.code for issue in bad_issues], ["frontmatter"])

    def test_malformed_companion_does_not_create_missing_pair_cascade(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            write_document(root, "guides/sample-guide.en.md")
            write_raw(root, "guides/sample-guide.zh-TW.md", "not frontmatter\n")

            codes = issue_codes(root)

            self.assertEqual(codes, ["frontmatter"])
