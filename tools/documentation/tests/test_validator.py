import dataclasses
import inspect
import tempfile
import unittest
from pathlib import Path


class PublicInterfaceTests(unittest.TestCase):
    def test_complete_public_interface(self):
        import tools.documentation.validator as validator

        self.assertEqual(
            {name for name in vars(validator) if not name.startswith("_")},
            {"Issue", "validate_repository"},
        )
        self.assertEqual(validator.__all__, ["Issue", "validate_repository"])
        self.assertTrue(dataclasses.is_dataclass(validator.Issue))
        self.assertEqual(
            [field.name for field in dataclasses.fields(validator.Issue)],
            ["path", "code", "message"],
        )
        issue = validator.Issue(Path("doc.en.md"), "code", "message")
        with self.assertRaises(dataclasses.FrozenInstanceError):
            issue.code = "changed"

        signature = inspect.signature(validator.validate_repository)
        self.assertEqual(list(signature.parameters), ["repo_root"])
        self.assertIs(signature.parameters["repo_root"].annotation, Path)
        self.assertEqual(signature.return_annotation, list[validator.Issue])
        self.assertEqual(
            validator.validate_repository.__doc__,
            "Return deterministic validation issues sorted by path, code, and message.",
        )


class ValidatorBehaviorTests(unittest.TestCase):
    def setUp(self):
        self._temporary_directory = tempfile.TemporaryDirectory()
        self.repo = Path(self._temporary_directory.name)

    def tearDown(self):
        self._temporary_directory.cleanup()

    def write_document(
        self,
        relative_path,
        *,
        document_id="guide",
        title="Guide",
        lang=None,
        audience="developer",
        document_type="explanation",
        status="active",
        owner="project",
        last_reviewed="2026-08-01",
        body="",
    ):
        path = self.repo / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        if lang is None:
            lang = "zh-TW" if relative_path.endswith(".zh-TW.md") else "en"
        path.write_text(
            "\n".join(
                [
                    "---",
                    f"id: {document_id}",
                    f"title: {title}",
                    f"lang: {lang}",
                    f"audience: {audience}",
                    f"type: {document_type}",
                    f"status: {status}",
                    f"owner: {owner}",
                    f"last_reviewed: {last_reviewed}",
                    "---",
                    body,
                ]
            ),
            encoding="utf-8",
        )
        return path

    def write_pair(self, relative_stem="component/guide", **kwargs):
        self.write_document(f"{relative_stem}.en.md", **kwargs)
        self.write_document(f"{relative_stem}.zh-TW.md", **kwargs)

    def write_raw(self, relative_path, text):
        path = self.repo / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return path

    def issue_codes(self):
        from tools.documentation.validator import validate_repository

        return [issue.code for issue in validate_repository(self.repo)]

    def test_valid_paired_mini_repository_passes(self):
        from tools.documentation.validator import validate_repository

        self.write_pair()

        self.assertEqual(validate_repository(self.repo), [])

    def test_invalid_filename_forms_fail(self):
        invalid_names = [
            "Bad.en.md",
            "bad_name.en.md",
            "bad name.en.md",
            "bad--name.en.md",
            "-bad.en.md",
            "bad-.en.md",
            "2026-2-30-report.en.md",
            "2026-02-30-report.en.md",
            "20260801-report.en.md",
            "adr-001-report.en.md",
            "guide.zh-tw.md",
        ]
        for name in invalid_names:
            (self.repo / name).write_text("not parsed yet", encoding="utf-8")

        self.assertEqual(
            self.issue_codes(),
            ["invalid-name"] * len(invalid_names),
        )

    def test_all_canonical_filename_families_are_accepted(self):
        self.write_pair("component/index", document_id="index-document")
        self.write_pair("component/normal-name", document_id="normal-name")
        self.write_pair("component/2026-08-01-report", document_id="dated-report")
        self.write_pair("component/adr-0001-decision", document_id="adr-decision")

        self.assertEqual(self.issue_codes(), [])

    def test_discovery_ignores_nonfiles_exclusions_and_noncanonical_files(self):
        excluded_directories = [
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
        ]
        for directory in excluded_directories:
            path = self.repo / directory / "Bad.en.md"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("ignored", encoding="utf-8")
        (self.repo / "Bad.en.md").mkdir()
        for name in ["README.md", "legacy.md", "guide.md.template", "docs.csv", "AGENTS.md"]:
            (self.repo / name).write_text("ignored", encoding="utf-8")

        self.assertEqual(self.issue_codes(), [])

    def test_frontmatter_requires_boundaries_and_scalar_key_value_lines(self):
        malformed_documents = {
            "opening.en.md": "\n---\nid: opening\n---\n",
            "closing.en.md": "---\nid: closing\n",
            "line.en.md": "---\nid line\n---\n",
            "duplicate.en.md": "---\nid: duplicate\nid: again\n---\n",
            "empty.en.md": "---\nid: \n---\n",
            "collection.en.md": "---\nid: [collection]\n---\n",
            "sequence.en.md": "---\n- id: sequence\n---\n",
        }
        for path, content in malformed_documents.items():
            self.write_raw(path, content)

        self.assertEqual(self.issue_codes(), ["frontmatter"] * len(malformed_documents))

    def test_frontmatter_rejects_all_yaml_block_scalar_indicator_orders(self):
        indicators = [
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
        ]
        for index, indicator in enumerate(indicators):
            self.write_raw(
                f"block-{index}.en.md",
                f"---\nid: {indicator}\n---\n",
            )

        self.assertEqual(self.issue_codes(), ["frontmatter"] * len(indicators))

    def test_metadata_requires_exact_field_set(self):
        missing = self.write_document("missing.en.md", document_id="missing")
        missing.write_text(
            missing.read_text(encoding="utf-8").replace("owner: project\n", ""),
            encoding="utf-8",
        )
        extra = self.write_document("extra.en.md", document_id="extra")
        extra.write_text(
            extra.read_text(encoding="utf-8").replace(
                "last_reviewed: 2026-08-01\n",
                "last_reviewed: 2026-08-01\nextra: unsupported\n",
            ),
            encoding="utf-8",
        )

        self.assertEqual(self.issue_codes(), ["invalid-metadata", "invalid-metadata"])

    def test_metadata_enforces_all_enums_and_status_type_mappings(self):
        invalid_overrides = [
            {"lang": "fr"},
            {"audience": "reader"},
            {"owner": "nobody"},
            {"document_type": "memo"},
        ]
        for document_type in [
            "index",
            "tutorial",
            "how-to",
            "reference",
            "explanation",
            "safety",
            "troubleshooting",
        ]:
            invalid_overrides.append({"document_type": document_type, "status": "published"})
        invalid_overrides.extend(
            [
                {"document_type": "decision", "status": "active"},
                {"document_type": "design", "status": "active"},
                {"document_type": "plan", "status": "published"},
                {"document_type": "roadmap", "status": "draft"},
                {"document_type": "release", "status": "active"},
                {"document_type": "experiment-summary", "status": "active"},
                {"document_type": "audit", "status": "active"},
            ]
        )
        for index, overrides in enumerate(invalid_overrides):
            self.write_document(
                f"invalid-metadata-{index}.en.md",
                document_id=f"invalid-metadata-{index}",
                **overrides,
            )

        self.assertEqual(self.issue_codes(), ["invalid-metadata"] * len(invalid_overrides))

    def test_metadata_enforces_id_review_date_and_filename_language(self):
        self.write_document("invalid-id.en.md", document_id="Invalid_ID")
        self.write_document("invalid-date.en.md", document_id="invalid-date", last_reviewed="2026-02-30")
        self.write_document("invalid-date-form.en.md", document_id="invalid-date-form", last_reviewed="2026-8-01")
        self.write_document("language.en.md", document_id="language", lang="zh-TW")

        self.assertEqual(self.issue_codes(), ["invalid-metadata"] * 4)

    def test_central_documents_enforce_section_location_contract(self):
        self.write_document(
            "docs/operators/wrong.en.md",
            document_id="operator-wrong",
            audience="developer",
        )
        self.write_document(
            "docs/reference/wrong.en.md",
            document_id="reference-wrong",
            audience="shared",
            document_type="explanation",
        )
        self.write_document(
            "docs/operators/nested/index.en.md",
            document_id="nested-index",
            audience="operator",
            document_type="index",
        )
        self.write_document(
            "docs/operators/not-index.en.md",
            document_id="not-index",
            audience="operator",
            document_type="index",
        )
        self.write_document(
            "docs/unknown/guide.en.md",
            document_id="unknown-guide",
        )
        self.write_document(
            "docs/index.en.md",
            document_id="root-index",
            audience="developer",
            document_type="index",
        )
        self.write_pair(
            "docs/operators/index",
            document_id="operator-index",
            audience="operator",
            document_type="index",
        )

        self.assertEqual(self.issue_codes(), ["invalid-location"] * 6)

    def test_metadata_valid_document_requires_locale_companion(self):
        self.write_document("component/lonely.en.md", document_id="lonely")

        self.assertEqual(self.issue_codes(), ["missing-pair"])

    def test_pair_metadata_allows_localization_and_reports_drift_once(self):
        self.write_document("localized/guide.en.md", document_id="localized", title="Guide")
        self.write_document("localized/guide.zh-TW.md", document_id="localized", title="指南")
        self.write_document("drift/guide.en.md", document_id="drift", owner="project")
        self.write_document("drift/guide.zh-TW.md", document_id="drift", owner="core")

        self.assertEqual(self.issue_codes(), ["pair-metadata"])

    def test_id_reuse_across_complete_logical_pairs_fails_once(self):
        self.write_pair("first/guide", document_id="shared-id")
        self.write_pair("second/guide", document_id="shared-id")

        self.assertEqual(self.issue_codes(), ["duplicate-id"])

    def test_duplicate_id_detection_includes_incomplete_pairs(self):
        self.write_document("first/guide.en.md", document_id="shared-id")
        self.write_document("second/guide.en.md", document_id="shared-id")

        self.assertEqual(
            self.issue_codes(),
            ["duplicate-id", "missing-pair", "missing-pair"],
        )

    def test_every_atx_heading_requires_immediately_preceding_anchor(self):
        self.write_pair("component/guide", body="## Missing anchor")

        self.assertEqual(self.issue_codes(), ["heading-anchor", "heading-anchor"])

    def test_invalid_explicit_anchor_is_one_root_cause_per_heading(self):
        body = '<a id="Bad_anchor"></a>\n## Heading'
        self.write_pair("component/guide", body=body)

        self.assertEqual(self.issue_codes(), ["heading-anchor", "heading-anchor"])

    def test_invalid_standalone_explicit_anchor_fails(self):
        self.write_pair("component/guide", body='<a id="Invalid_anchor"></a>')

        self.assertEqual(self.issue_codes(), ["heading-anchor", "heading-anchor"])

    def test_duplicate_anchor_ids_fail_once_per_file(self):
        body = (
            '<a id="same"></a>\n## One\n'
            '<a id="same"></a>\n## Two'
        )
        self.write_pair("component/guide", body=body)

        self.assertEqual(self.issue_codes(), ["duplicate-anchor", "duplicate-anchor"])

    def test_locale_pair_anchor_sequences_must_match_once(self):
        english_body = (
            '<a id="one"></a>\n## One\n'
            '<a id="two"></a>\n## Two'
        )
        chinese_body = (
            '<a id="one"></a>\n## 一\n'
            '<a id="three"></a>\n## 三'
        )
        self.write_document("component/guide.en.md", body=english_body)
        self.write_document("component/guide.zh-TW.md", body=chinese_body)

        self.assertEqual(self.issue_codes(), ["pair-anchors"])

    def test_locale_pair_sequence_includes_standalone_explicit_anchors(self):
        self.write_document(
            "component/guide.en.md",
            body='<a id="english-standalone"></a>',
        )
        self.write_document(
            "component/guide.zh-TW.md",
            body='<a id="chinese-standalone"></a>',
        )

        self.assertEqual(self.issue_codes(), ["pair-anchors"])

    def test_fences_track_character_length_and_three_space_indentation(self):
        body = "\n".join(
            [
                "````python",
                "## ignored backtick heading",
                '<a id="ignored"></a>',
                "~~~",
                "## opposite marker did not close",
                "```",
                "## shorter marker did not close",
                "````",
                "   ~~~ text",
                "## ignored tilde heading",
                '<a id="ignored"></a>',
                "```",
                "## opposite marker still ignored",
                "   ~~~~",
                '<a id="real"></a>',
                "## Real heading",
            ]
        )
        self.write_pair("component/guide", body=body)

        self.assertEqual(self.issue_codes(), [])

    def test_links_and_images_inside_fences_are_ignored(self):
        body = "\n".join(
            [
                "```markdown",
                "[missing](missing.txt)",
                "![missing](missing.png)",
                "```",
            ]
        )
        self.write_pair("component/guide", body=body)

        self.assertEqual(self.issue_codes(), [])

    def test_inline_links_and_images_validate_missing_local_targets(self):
        body = "\n".join(
            [
                "[missing](missing.txt)",
                "![missing image](missing-image.png)",
                "[web](https://example.com/page)",
                "[mail](mailto:docs@example.com)",
            ]
        )
        self.write_document("component/guide.en.md", body=body)
        self.write_document("component/guide.zh-TW.md")

        self.assertEqual(self.issue_codes(), ["broken-link", "broken-link"])

    def test_reference_links_and_images_validate_definition_targets(self):
        target = self.repo / "component/target.txt"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("target", encoding="utf-8")
        body = "\n".join(
            [
                "[missing document][doc]",
                "![missing image][image]",
                "[existing target][ok]",
                "[doc]: missing.txt",
                "[image]: missing.png 'image title'",
                '[ok]: target.txt "target title"',
            ]
        )
        self.write_document("component/guide.en.md", body=body)
        self.write_document("component/guide.zh-TW.md")

        self.assertEqual(self.issue_codes(), ["broken-link", "broken-link"])

    def test_reference_destinations_apply_absolute_containment_and_fragment_rules(self):
        outside = self.repo.parent / f"{self.repo.name}-outside-reference.txt"
        outside.write_text("outside", encoding="utf-8")
        self.addCleanup(outside.unlink, missing_ok=True)
        target = self.repo / "component/target.md"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text('<a id="present"></a>\n', encoding="utf-8")
        body = "\n".join(
            [
                "[drive use][drive]",
                "[escape use][escape]",
                "[missing anchor use][missing-anchor]",
                "[valid anchor use][valid-anchor]",
                "[drive]: C:/secret/file.md",
                f"[escape]: ../../{outside.name}",
                "[missing-anchor]: target.md#absent",
                "[valid-anchor]: target.md#present",
            ]
        )
        self.write_document("component/guide.en.md", body=body)
        self.write_document("component/guide.zh-TW.md")

        self.assertEqual(
            self.issue_codes(),
            ["broken-link", "broken-link", "missing-link-anchor"],
        )

    def test_inline_optional_single_and_double_quoted_titles_are_not_destinations(self):
        body = "\n".join(
            [
                '[missing double](double.txt "optional title")',
                "[missing single](single.txt 'optional title')",
            ]
        )
        self.write_document("component/guide.en.md", body=body)
        self.write_document("component/guide.zh-TW.md")

        self.assertEqual(self.issue_codes(), ["broken-link", "broken-link"])

    def test_inline_unquoted_destinations_support_balanced_parentheses(self):
        target = self.repo / "component/target(name).txt"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("target", encoding="utf-8")
        self.write_document(
            "component/guide.en.md",
            body="[balanced](target(name).txt)",
        )
        self.write_document("component/guide.zh-TW.md")

        self.assertEqual(self.issue_codes(), [])

    def test_parentheses_inside_quoted_inline_title_do_not_change_destination(self):
        self.write_document(
            "component/guide.en.md",
            body='[missing](target.txt "Title (with parentheses)")',
        )
        self.write_document("component/guide.zh-TW.md")

        self.assertEqual(self.issue_codes(), ["broken-link"])

    def test_inline_angle_destinations_preserve_spaces_and_optional_title(self):
        target = self.repo / "component/target file.txt"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("target", encoding="utf-8")
        body = "\n".join(
            [
                "[existing](<target file.txt>)",
                '[missing](<missing file.txt> "optional (title)")',
            ]
        )
        self.write_document("component/guide.en.md", body=body)
        self.write_document("component/guide.zh-TW.md")

        self.assertEqual(self.issue_codes(), ["broken-link"])

    def test_absolute_paths_fail_before_uri_scheme_exclusion(self):
        body = "\n".join(
            [
                r"[drive backslash](C:\secret\file.md)",
                "[drive slash](C:/secret/file.md)",
                r"[unc](\\server\share\file.md)",
                "[posix](/absolute/path)",
                "[site root](/docs/page)",
            ]
        )
        self.write_document("component/guide.en.md", body=body)
        self.write_document("component/guide.zh-TW.md")

        self.assertEqual(self.issue_codes(), ["broken-link"] * 5)

    def test_local_link_paths_are_percent_decoded(self):
        target = self.repo / "component/target file.txt"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("target", encoding="utf-8")
        self.write_document(
            "component/guide.en.md",
            body="[encoded](target%20file.txt)",
        )
        self.write_document("component/guide.zh-TW.md")

        self.assertEqual(self.issue_codes(), [])

    def test_existing_link_target_outside_repository_is_rejected(self):
        outside = self.repo.parent / f"{self.repo.name}-outside.txt"
        outside.write_text("outside", encoding="utf-8")
        self.addCleanup(outside.unlink, missing_ok=True)
        self.write_document(
            "component/guide.en.md",
            body=f"[escape](../../{outside.name})",
        )
        self.write_document("component/guide.zh-TW.md")

        self.assertEqual(self.issue_codes(), ["broken-link"])

    def test_local_link_target_must_be_a_file_not_a_directory(self):
        (self.repo / "component/directory").mkdir(parents=True)
        self.write_document(
            "component/guide.en.md",
            body="[directory](directory)",
        )
        self.write_document("component/guide.zh-TW.md")

        self.assertEqual(self.issue_codes(), ["broken-link"])

    def test_local_fragments_are_percent_decoded_and_require_explicit_anchor(self):
        target = self.repo / "component/target.md"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text('<a id="target-anchor"></a>\n', encoding="utf-8")
        english_body = "\n".join(
            [
                '<a id="source-anchor"></a>',
                "## Source",
                "[target](target.md#target%2Danchor)",
                "[self](#source%2Danchor)",
                "[missing](target.md#missing-anchor)",
            ]
        )
        chinese_body = '<a id="source-anchor"></a>\n## 來源'
        self.write_document("component/guide.en.md", body=english_body)
        self.write_document("component/guide.zh-TW.md", body=chinese_body)

        self.assertEqual(self.issue_codes(), ["missing-link-anchor"])

    def test_broken_link_does_not_suppress_pair_anchor_mismatch(self):
        english_body = "\n".join(
            [
                '<a id="english-anchor"></a>',
                "## English",
                "[missing](missing.txt)",
            ]
        )
        chinese_body = '<a id="chinese-anchor"></a>\n## 中文'
        self.write_document("component/guide.en.md", body=english_body)
        self.write_document("component/guide.zh-TW.md", body=chinese_body)

        self.assertEqual(self.issue_codes(), ["broken-link", "pair-anchors"])

    def test_issue_aggregation_is_exact_deterministic_and_non_cascading(self):
        from tools.documentation.validator import validate_repository

        self.write_raw("a.en.md", "---\nid: broken\n")
        self.write_document(
            "invalid/guide.en.md",
            document_id="invalid-anchor-pair",
            body='<a id="Invalid"></a>\n## Heading',
        )
        self.write_document(
            "invalid/guide.zh-TW.md",
            document_id="invalid-anchor-pair",
            body='<a id="different"></a>\n## 標題',
        )
        self.write_document(
            "z/guide.en.md",
            document_id="drift",
            owner="project",
            body='<a id="english"></a>\n## English',
        )
        self.write_document(
            "z/guide.zh-TW.md",
            document_id="drift",
            owner="core",
            body='<a id="chinese"></a>\n## 中文',
        )

        first = validate_repository(self.repo)
        second = validate_repository(self.repo)
        self.assertEqual(first, second)
        self.assertEqual(
            [(issue.path.as_posix(), issue.code) for issue in first],
            [
                ("a.en.md", "frontmatter"),
                ("invalid/guide.en.md", "heading-anchor"),
                ("z/guide.en.md", "pair-anchors"),
                ("z/guide.en.md", "pair-metadata"),
            ],
        )


class ActualRepositoryTests(unittest.TestCase):
    def test_actual_current_repository_passes(self):
        from tools.documentation.validator import validate_repository

        repo_root = Path(__file__).resolve().parents[3]
        self.assertTrue((repo_root / ".git").exists())
        self.assertEqual(validate_repository(repo_root), [])


if __name__ == "__main__":
    unittest.main()
