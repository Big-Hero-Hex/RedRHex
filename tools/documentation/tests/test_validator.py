from dataclasses import FrozenInstanceError
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from tools.documentation import validator as validator_module
from tools.documentation.validator import validate_repository


class ValidatorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.repo = Path(self.temp_dir.name)

    def write_document(
        self,
        relative_path: str,
        *,
        doc_id: str = "guide",
        title: str = "Guide",
        lang: str | None = None,
        audience: str = "developer",
        doc_type: str = "explanation",
        status: str = "active",
        owner: str = "project",
        last_reviewed: str = "2026-08-01",
        body: str = '<a id="overview"></a>\n## Overview\n',
    ) -> Path:
        path = self.repo / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        locale = lang or ("zh-TW" if relative_path.endswith(".zh-TW.md") else "en")
        path.write_text(
            "\n".join(
                [
                    "---",
                    f"id: {doc_id}",
                    f"title: {title}",
                    f"lang: {locale}",
                    f"audience: {audience}",
                    f"type: {doc_type}",
                    f"status: {status}",
                    f"owner: {owner}",
                    f"last_reviewed: {last_reviewed}",
                    "---",
                    "",
                    body.rstrip("\n"),
                    "",
                ]
            ),
            encoding="utf-8",
        )
        return path

    def write_pair(self, stem: str = "components/guide", **metadata: str) -> None:
        metadata.setdefault("doc_id", Path(stem).name.lower().replace("_", "-"))
        self.write_document(f"{stem}.en.md", **metadata)
        self.write_document(f"{stem}.zh-TW.md", title="指南", **metadata)

    def write_raw(self, relative_path: str, content: str) -> Path:
        path = self.repo / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return path

    def replace_in_document(self, relative_path: str, old: str, new: str) -> None:
        path = self.repo / relative_path
        path.write_text(path.read_text(encoding="utf-8").replace(old, new), encoding="utf-8")

    def test_valid_paired_repository_passes(self) -> None:
        self.write_pair()

        self.assertEqual(validate_repository(self.repo), [])

    def test_public_validator_surface_is_issue_and_validate_repository(self) -> None:
        self.assertEqual(
            validator_module.__all__,
            ["Issue", "validate_repository"],
        )
        self.assertFalse(hasattr(validator_module, "document_count"))

    def test_issue_value_object_is_frozen(self) -> None:
        issue = validator_module.Issue(Path("guide.en.md"), "code", "message")

        with self.assertRaises(FrozenInstanceError):
            issue.code = "changed"

    def test_invalid_normal_filename_forms_fail(self) -> None:
        self.write_pair("components/Bad_name")

        issues = validate_repository(self.repo)

        self.assertEqual([issue.code for issue in issues], ["invalid-name", "invalid-name"])

    def test_digit_leading_date_names_require_exact_real_dates(self) -> None:
        self.write_pair("components/2026-2-30-report")
        self.write_pair("components/2026-02-30-report")
        self.write_pair("components/20260801-report")
        self.write_pair("components/2026-08-01-report")

        issues = validate_repository(self.repo)

        self.assertEqual([issue.code for issue in issues], ["invalid-name"] * 6)

    def test_digit_leading_non_date_kebab_name_remains_valid(self) -> None:
        self.write_pair("components/3d-model")

        self.assertEqual(validate_repository(self.repo), [])

    def test_adr_names_require_exact_four_digit_number_and_slug(self) -> None:
        self.write_pair("components/adr-001-decision")
        self.write_pair("components/adr-00001-decision")
        self.write_pair("components/adr-0001-decision")

        issues = validate_repository(self.repo)

        self.assertEqual([issue.code for issue in issues], ["invalid-name"] * 4)

    def test_discovery_ignores_templates_build_and_cache_directories(self) -> None:
        excluded = [
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
        for directory in excluded:
            self.write_document(f"{directory}/Bad_name.en.md")
        self.write_pair()

        self.assertEqual(validate_repository(self.repo), [])

    def test_wrong_locale_suffix_case_fails_filename_validation(self) -> None:
        self.write_document("components/guide.EN.md")
        self.write_document("components/guide.zh-tw.md")

        issues = validate_repository(self.repo)

        self.assertEqual([issue.code for issue in issues], ["invalid-name", "invalid-name"])

    def test_frontmatter_must_start_on_line_one_and_close(self) -> None:
        self.write_pair()
        self.write_raw("components/guide.en.md", "# No frontmatter\n")

        issues = validate_repository(self.repo)

        self.assertEqual([issue.code for issue in issues], ["frontmatter"])

    def test_frontmatter_rejects_malformed_scalar_lines(self) -> None:
        self.write_pair()
        self.replace_in_document("components/guide.en.md", "id: guide", "id guide")

        issues = validate_repository(self.repo)

        self.assertEqual([issue.code for issue in issues], ["frontmatter"])

    def test_frontmatter_rejects_duplicate_keys(self) -> None:
        self.write_pair()
        self.replace_in_document(
            "components/guide.en.md", "owner: project", "owner: project\nowner: core"
        )

        issues = validate_repository(self.repo)

        self.assertEqual([issue.code for issue in issues], ["frontmatter"])

    def test_frontmatter_rejects_empty_values(self) -> None:
        self.write_pair()
        self.replace_in_document("components/guide.en.md", "title: Guide", "title: ")

        issues = validate_repository(self.repo)

        self.assertEqual([issue.code for issue in issues], ["frontmatter"])

    def test_frontmatter_rejects_yaml_collection_and_multiline_syntax(self) -> None:
        self.write_pair("components/list-value")
        self.write_pair("components/map-value")
        self.write_pair("components/multiline-value")
        self.replace_in_document(
            "components/list-value.en.md", "title: Guide", "title: [Guide]"
        )
        self.replace_in_document(
            "components/map-value.en.md", "title: Guide", "title: {name: Guide}"
        )
        self.replace_in_document(
            "components/multiline-value.en.md", "title: Guide", "title: |"
        )

        issues = validate_repository(self.repo)

        self.assertEqual([issue.code for issue in issues], ["frontmatter"] * 3)

    def test_metadata_requires_every_required_field(self) -> None:
        self.write_pair()
        self.replace_in_document("components/guide.en.md", "owner: project\n", "")

        issues = validate_repository(self.repo)

        self.assertEqual([issue.code for issue in issues], ["invalid-metadata"])

    def test_metadata_rejects_unsupported_extra_fields(self) -> None:
        self.write_pair()
        self.replace_in_document(
            "components/guide.en.md", "owner: project", "owner: project\ncategory: extra"
        )

        issues = validate_repository(self.repo)

        self.assertEqual([issue.code for issue in issues], ["invalid-metadata"])

    def test_metadata_enums_require_exact_governance_values(self) -> None:
        self.write_pair("components/bad-lang")
        self.write_pair("components/bad-audience")
        self.write_pair("components/bad-type")
        self.write_pair("components/bad-owner")
        self.replace_in_document("components/bad-lang.en.md", "lang: en", "lang: EN")
        self.replace_in_document(
            "components/bad-audience.en.md", "audience: developer", "audience: engineers"
        )
        self.replace_in_document(
            "components/bad-type.en.md", "type: explanation", "type: article"
        )
        self.replace_in_document(
            "components/bad-owner.en.md", "owner: project", "owner: unknown"
        )

        issues = validate_repository(self.repo)

        self.assertEqual([issue.code for issue in issues], ["invalid-metadata"] * 4)

    def test_status_must_match_document_type_lifecycle(self) -> None:
        self.write_pair()
        self.replace_in_document(
            "components/guide.en.md", "type: explanation", "type: decision"
        )

        issues = validate_repository(self.repo)

        self.assertEqual([issue.code for issue in issues], ["invalid-metadata"])

    def test_document_id_must_be_lowercase_kebab_case(self) -> None:
        self.write_pair()
        self.replace_in_document("components/guide.en.md", "id: guide", "id: Guide_ID")

        issues = validate_repository(self.repo)

        self.assertEqual([issue.code for issue in issues], ["invalid-metadata"])

    def test_last_reviewed_must_be_an_exact_real_iso_date(self) -> None:
        self.write_pair("components/bad-shape")
        self.write_pair("components/impossible-date")
        self.replace_in_document(
            "components/bad-shape.en.md", "last_reviewed: 2026-08-01", "last_reviewed: 2026-8-1"
        )
        self.replace_in_document(
            "components/impossible-date.en.md",
            "last_reviewed: 2026-08-01",
            "last_reviewed: 2026-02-30",
        )

        issues = validate_repository(self.repo)

        self.assertEqual([issue.code for issue in issues], ["invalid-metadata"] * 2)

    def test_filename_locale_must_equal_metadata_language(self) -> None:
        self.write_pair()
        self.replace_in_document("components/guide.en.md", "lang: en", "lang: zh-TW")

        issues = validate_repository(self.repo)

        self.assertEqual([issue.code for issue in issues], ["invalid-metadata"])

    def test_central_sections_enforce_governed_audience_and_type(self) -> None:
        cases = [
            ("docs/operators/operator-bad", {"audience": "shared"}),
            ("docs/developers/developer-bad", {"audience": "operator"}),
            (
                "docs/reference/reference-bad",
                {"audience": "developer", "doc_type": "reference"},
            ),
            ("docs/decisions/decision-bad", {}),
            ("docs/designs/design-bad", {}),
            ("docs/plans/plan-bad", {}),
            (
                "docs/roadmap/roadmap-bad",
                {"doc_type": "roadmap", "audience": "developer"},
            ),
            (
                "docs/releases/release-bad",
                {"doc_type": "release", "status": "published", "audience": "developer"},
            ),
            (
                "docs/research/research-bad",
                {"doc_type": "audit", "status": "published", "audience": "shared"},
            ),
            (
                "docs/governance/governance-bad",
                {"doc_type": "reference", "audience": "shared"},
            ),
        ]
        for stem, metadata in cases:
            self.write_pair(stem, **metadata)

        issues = validate_repository(self.repo)

        self.assertEqual([issue.code for issue in issues], ["invalid-location"] * 20)

    def test_index_exception_applies_only_to_direct_section_landing_file(self) -> None:
        self.write_pair(
            "docs/operators/index",
            audience="operator",
            doc_type="index",
            doc_id="operators-index",
        )
        self.write_pair(
            "docs/operators/nested/index",
            audience="operator",
            doc_type="index",
            doc_id="operators-nested-index",
        )
        self.write_pair("docs/operators/not-index", audience="operator", doc_type="index")

        issues = validate_repository(self.repo)

        self.assertEqual([issue.code for issue in issues], ["invalid-location"] * 4)

    def test_unrecognized_central_docs_section_fails_location_validation(self) -> None:
        self.write_pair("docs/misc/guide")

        issues = validate_repository(self.repo)

        self.assertEqual([issue.code for issue in issues], ["invalid-location"] * 2)

    def test_docs_root_allows_only_shared_index_portal(self) -> None:
        self.write_pair("docs/index", audience="shared", doc_type="index")
        self.write_pair("docs/guide")

        issues = validate_repository(self.repo)

        self.assertEqual([issue.code for issue in issues], ["invalid-location"] * 2)

    def test_canonical_document_requires_companion_locale(self) -> None:
        self.write_document("components/guide.en.md")

        issues = validate_repository(self.repo)

        self.assertEqual([issue.code for issue in issues], ["missing-pair"])

    def test_pair_metadata_drift_is_reported_once_while_title_and_lang_may_differ(self) -> None:
        self.write_pair()
        self.replace_in_document(
            "components/guide.zh-TW.md", "owner: project", "owner: core"
        )

        issues = validate_repository(self.repo)

        self.assertEqual([issue.code for issue in issues], ["pair-metadata"])
        self.assertEqual(issues[0].path, Path("components/guide.en.md"))

    def test_document_id_cannot_be_reused_by_another_logical_pair(self) -> None:
        self.write_pair("components/alpha", doc_id="shared-id")
        self.write_pair("components/beta", doc_id="shared-id")

        issues = validate_repository(self.repo)

        self.assertEqual([issue.code for issue in issues], ["duplicate-id"])
        self.assertEqual(issues[0].path, Path("components/beta.en.md"))

    def test_pair_with_invalid_metadata_skips_pair_and_duplicate_id_checks(self) -> None:
        self.write_pair("components/alpha", doc_id="shared-id")
        self.write_pair("components/beta", doc_id="shared-id")
        self.replace_in_document(
            "components/alpha.en.md", "owner: project", "owner: unknown"
        )

        issues = validate_repository(self.repo)

        self.assertEqual([issue.code for issue in issues], ["invalid-metadata"])

    def test_atx_heading_requires_immediately_preceding_explicit_anchor(self) -> None:
        self.write_pair()
        self.replace_in_document("components/guide.en.md", '<a id="overview"></a>\n', "")

        issues = validate_repository(self.repo)

        self.assertEqual([issue.code for issue in issues], ["heading-anchor"])

    def test_invalid_anchor_before_heading_reports_one_root_cause_issue(self) -> None:
        self.write_pair()
        self.replace_in_document(
            "components/guide.en.md", 'id="overview"', 'id="Bad_Anchor"'
        )

        issues = validate_repository(self.repo)

        self.assertEqual([issue.code for issue in issues], ["heading-anchor"])
        self.assertIn("invalid anchor", issues[0].message)

    def test_duplicate_explicit_anchor_id_fails_once_per_repeat(self) -> None:
        self.write_pair()
        self.replace_in_document(
            "components/guide.en.md",
            "## Overview\n",
            '## Overview\n\n<a id="overview"></a>\n## Repeated\n',
        )

        issues = validate_repository(self.repo)

        self.assertEqual([issue.code for issue in issues], ["duplicate-anchor"])

    def test_pair_anchor_sequence_mismatch_is_reported_once(self) -> None:
        self.write_pair()
        self.replace_in_document(
            "components/guide.zh-TW.md", 'id="overview"', 'id="translated-overview"'
        )

        issues = validate_repository(self.repo)

        self.assertEqual([issue.code for issue in issues], ["pair-anchors"])
        self.assertEqual(issues[0].path, Path("components/guide.en.md"))

    def test_fences_track_marker_character_length_and_leading_spaces(self) -> None:
        body = "\n".join(
            [
                '<a id="overview"></a>',
                "## Overview",
                "   ````markdown",
                '<a id="Bad_Anchor"></a>',
                "# Hidden backtick heading",
                "~~~",
                "[missing](missing.md)",
                "```",
                "   ````",
                "   ~~~markdown",
                '<a id="Bad_Anchor"></a>',
                "# Hidden tilde heading",
                "```",
                "![missing](missing.png)",
                "   ~~~",
            ]
        )
        self.write_pair(body=body)

        self.assertEqual(validate_repository(self.repo), [])

    def test_missing_absolute_and_outside_inline_links_and_images_fail(self) -> None:
        self.write_pair()
        self.replace_in_document(
            "components/guide.en.md",
            "## Overview\n",
            "## Overview\n"
            "[missing](missing.md)\n"
            "![missing image](missing.png)\n"
            "[absolute](/etc/passwd)\n"
            "[outside](../../outside.md)\n",
        )

        issues = validate_repository(self.repo)

        self.assertEqual([issue.code for issue in issues], ["broken-link"] * 4)

    def test_inline_link_accepts_optional_title(self) -> None:
        self.write_pair()
        self.write_raw("components/target.txt", "target\n")
        self.replace_in_document(
            "components/guide.en.md",
            "## Overview\n",
            '## Overview\n[target](target.txt "Optional title")\n',
        )

        self.assertEqual(validate_repository(self.repo), [])

    def test_inline_link_destination_supports_balanced_parentheses(self) -> None:
        self.write_pair()
        self.write_raw("components/target_(one).txt", "target\n")
        self.replace_in_document(
            "components/guide.en.md",
            "## Overview\n",
            "## Overview\n[target](target_(one).txt)\n",
        )

        self.assertEqual(validate_repository(self.repo), [])

    def test_missing_reference_style_link_and_image_targets_fail(self) -> None:
        self.write_pair()
        self.replace_in_document(
            "components/guide.en.md",
            "## Overview\n",
            "## Overview\n[missing][unknown]\n![missing image][unknown-image]\n",
        )

        issues = validate_repository(self.repo)

        self.assertEqual([issue.code for issue in issues], ["broken-link"] * 2)

    def test_shortcut_reference_uses_resolve_definitions_and_report_missing(self) -> None:
        self.write_pair()
        self.write_raw("components/target.txt", "target\n")
        self.write_raw("components/image.png", "image\n")
        self.replace_in_document(
            "components/guide.en.md",
            "## Overview\n",
            "## Overview\n"
            "[valid-link]\n"
            "![valid-image]\n"
            "[missing-link]\n"
            "![missing-image]\n"
            '[valid-link]: target.txt "Optional title"\n'
            "[valid-image]: image.png\n"
            "[missing-link]: missing.txt\n"
            "[missing-image]: missing.png\n",
        )

        issues = validate_repository(self.repo)

        self.assertEqual([issue.code for issue in issues], ["broken-link"] * 2)

    def test_bracketed_text_without_definition_is_not_a_shortcut_link(self) -> None:
        self.write_pair()
        self.replace_in_document(
            "components/guide.en.md",
            "## Overview\n",
            "## Overview\n[plain bracketed text]\n- [ ] unchecked item\n",
        )

        self.assertEqual(validate_repository(self.repo), [])

    def test_local_fragments_require_exact_explicit_target_anchor(self) -> None:
        self.write_pair()
        self.write_raw("components/target.md", '<a id="present"></a>\n')
        self.replace_in_document(
            "components/guide.en.md",
            "## Overview\n",
            "## Overview\n"
            "[valid](target.md#present)\n"
            "[encoded](target%2Emd#pre%73ent)\n"
            "[missing target anchor](target.md#missing)\n"
            "[missing current anchor](#missing)\n",
        )

        issues = validate_repository(self.repo)

        self.assertEqual([issue.code for issue in issues], ["missing-link-anchor"] * 2)

    def test_issue_order_is_deterministic_and_frontmatter_does_not_cascade(self) -> None:
        self.write_pair("components/alpha")
        self.replace_in_document(
            "components/alpha.zh-TW.md", "owner: project", "owner: core"
        )
        self.write_pair("components/beta")
        self.write_raw("components/beta.en.md", "# malformed\n")
        self.write_document("components/zeta.en.md", doc_id="zeta")

        first = validate_repository(self.repo)
        second = validate_repository(self.repo)
        expected = [
            (Path("components/alpha.en.md"), "pair-metadata"),
            (Path("components/beta.en.md"), "frontmatter"),
            (Path("components/zeta.en.md"), "missing-pair"),
        ]

        self.assertEqual([(issue.path, issue.code) for issue in first], expected)
        self.assertEqual(first, second)

    def test_actual_current_repository_passes(self) -> None:
        repo_root = Path(__file__).resolve().parents[3]

        self.assertEqual(validate_repository(repo_root), [])


if __name__ == "__main__":
    unittest.main()
