import dataclasses
import socket
import tempfile
import unittest
from pathlib import Path


def text(*, doc_id="sample", title="Sample", lang="en", audience="developer", doc_type="explanation", status="active", owner="project", reviewed="2026-08-01", body='<a id="overview"></a>\n## Overview\n'):
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


def pair(root, relative="component/sample", **values):
    base = root / relative
    base.parent.mkdir(parents=True, exist_ok=True)
    base.with_suffix(".en.md").write_text(text(**values), encoding="utf-8")
    base.with_suffix(".zh-TW.md").write_text(
        text(**{**values, "title": "範例", "lang": "zh-TW"}), encoding="utf-8"
    )


class Cycle01Interface(unittest.TestCase):
    def test_exact_public_interface(self):
        from tools.documentation import validator

        self.assertEqual(validator.__all__, ["Issue", "validate_repository"])
        self.assertEqual({name for name in vars(validator) if not name.startswith("_")}, {"Issue", "validate_repository"})
        self.assertTrue(dataclasses.is_dataclass(validator.Issue))
        self.assertEqual([field.name for field in dataclasses.fields(validator.Issue)], ["path", "code", "message"])
        issue = validator.Issue(Path("x"), "code", "message")
        with self.assertRaises(dataclasses.FrozenInstanceError):
            issue.code = "changed"
        self.assertEqual(
            validator.validate_repository.__doc__,
            "Return deterministic validation issues sorted by path, code, and message.",
        )


class Cycle02Baseline(unittest.TestCase):
    def test_empty_and_valid_pair(self):
        from tools.documentation.validator import validate_repository

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.assertEqual(validate_repository(root), [])
            pair(root)
            self.assertEqual(validate_repository(root), [])


class Cycle03Discovery(unittest.TestCase):
    def test_exact_candidates_exclusions_order_and_real_is_file(self):
        from tools.documentation.validator import _discover_candidates

        excluded = (".git", ".worktrees", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", ".tox", ".nox", "build", "dist", "site")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in excluded:
                directory = root / name
                directory.mkdir()
                (directory / "bad.en.md").write_text("bad", encoding="utf-8")
            template = root / "docs/governance/templates"
            template.mkdir(parents=True)
            (template / "bad.en.md").write_text("bad", encoding="utf-8")
            (root / "z.zh-TW.md").write_text("x", encoding="utf-8")
            (root / "a.en.md").write_text("x", encoding="utf-8")
            for name in ("README.md", "x.EN.md", "x.zh-tw.md", "x.en.md.template"):
                (root / name).write_text("x", encoding="utf-8")
            socket_path = root / "bound.en.md"
            bound = socket.socket(socket.AF_UNIX)
            try:
                bound.bind(str(socket_path))
                self.assertFalse(socket_path.is_file())
                self.assertEqual(
                    [path.relative_to(root).as_posix() for path in _discover_candidates(root)],
                    ["a.en.md", "z.zh-TW.md"],
                )
            finally:
                bound.close()


class Cycle04Names(unittest.TestCase):
    REJECTED = (
        "Upper.en.md", "bad_name.en.md", "bad name.en.md", "-bad.en.md", "bad-.en.md", "bad--name.en.md",
        "2026-2-30-report.en.md", "2026-02-30-report.en.md", "20260801-report.en.md", "2026-08-01-Report.en.md", "123-report.en.md",
        "adr-001-record.en.md", "adr-00001-record.en.md", "adr-0001-Record.en.md", "adr-0001-bad--record.en.md",
    )

    def test_all_name_families(self):
        from tools.documentation.validator import _filename_info

        for name in ("index.en.md", "quick-start.en.md", "guide2.zh-TW.md", "2024-02-29-report.en.md", "adr-0001-record.zh-TW.md"):
            self.assertIsNotNone(_filename_info(Path(name)), name)

    def test_public_validator_wires_discovery_and_names(self):
        from tools.documentation.validator import validate_repository

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in self.REJECTED:
                (root / name).write_text("not parsed", encoding="utf-8")
            ignored = root / "build"
            ignored.mkdir()
            (ignored / "Also-Bad.en.md").write_text("not parsed", encoding="utf-8")
            (root / "wrong.EN.md").write_text("not canonical", encoding="utf-8")
            issues = validate_repository(root)
            self.assertEqual([issue.code for issue in issues], ["invalid-name"] * len(self.REJECTED))
            self.assertEqual([issue.path.as_posix() for issue in issues], sorted(self.REJECTED))


class Cycle05Frontmatter(unittest.TestCase):
    def test_all_scalar_grammar_and_public_non_cascade(self):
        from tools.documentation.validator import validate_repository

        block = ("|", ">", "|-", "|+", ">-", ">+", "|2", ">2", "|2-", "|2+", ">-2", ">+2", "|2- # note", ">+2 # note")
        mutations = [
            lambda value: "\n" + value,
            lambda value: value.replace("---\n\n<a", "--\n\n<a", 1),
            lambda value: value.replace("title: Sample", "title Sample", 1),
            lambda value: value.replace("title: Sample", "title: Sample\ntitle: Again", 1),
            lambda value: value.replace("title: Sample", "title:", 1),
            lambda value: value.replace("title: Sample", "title: [Sample]", 1),
            lambda value: value.replace("title: Sample", "title: {name: Sample}", 1),
            lambda value: value.replace("title: Sample", "title: Sample\n  continuation", 1),
        ]
        mutations += [lambda value, indicator=indicator: value.replace("title: Sample", f"title: {indicator}", 1) for indicator in block]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for index, mutation in enumerate(mutations):
                stem = f"case-{index:02d}"
                (root / f"{stem}.en.md").write_text(mutation(text(doc_id=stem)), encoding="utf-8")
                (root / f"{stem}.zh-TW.md").write_text(text(doc_id=stem, title="範例", lang="zh-TW"), encoding="utf-8")
            issues = validate_repository(root)
            self.assertEqual(len(issues), len(mutations))
            self.assertEqual([issue.code for issue in issues], ["frontmatter"] * len(mutations))


class Cycle06MetadataShape(unittest.TestCase):
    def test_required_and_extra_fields_publicly(self):
        from tools.documentation.validator import validate_repository

        fields = ("id", "title", "lang", "audience", "type", "status", "owner", "last_reviewed")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for index, field in enumerate(fields):
                stem = f"missing-{index}"
                invalid = "\n".join(line for line in text(doc_id=stem).splitlines() if not line.startswith(f"{field}:")) + "\n"
                (root / f"{stem}.en.md").write_text(invalid, encoding="utf-8")
                (root / f"{stem}.zh-TW.md").write_text(text(doc_id=stem, title="範例", lang="zh-TW"), encoding="utf-8")
            (root / "extra.en.md").write_text(text(doc_id="extra").replace("---\n\n", "extra: value\n---\n\n", 1), encoding="utf-8")
            (root / "extra.zh-TW.md").write_text(text(doc_id="extra", title="範例", lang="zh-TW"), encoding="utf-8")
            self.assertEqual([issue.code for issue in validate_repository(root)], ["invalid-metadata"] * 9)


class Cycle07Enums(unittest.TestCase):
    BASE = {"id": "sample", "title": "Sample", "lang": "en", "audience": "developer", "type": "explanation", "status": "active", "owner": "project", "last_reviewed": "2026-08-01"}
    STATUSES = {
        "index": {"draft", "active", "deprecated"}, "tutorial": {"draft", "active", "deprecated"}, "how-to": {"draft", "active", "deprecated"},
        "reference": {"draft", "active", "deprecated"}, "explanation": {"draft", "active", "deprecated"}, "safety": {"draft", "active", "deprecated"}, "troubleshooting": {"draft", "active", "deprecated"},
        "decision": {"accepted", "superseded"}, "design": {"proposed", "approved", "implemented", "rejected", "superseded"},
        "plan": {"draft", "active", "blocked", "completed", "cancelled"}, "roadmap": {"active"}, "release": {"published"}, "experiment-summary": {"published"}, "audit": {"published"},
    }

    def test_every_enum_and_status_helper_branch(self):
        from tools.documentation.validator import _values_error

        for lang in ("en", "zh-TW"):
            self.assertIsNone(_values_error({**self.BASE, "lang": lang}))
        for audience in ("operator", "developer", "shared"):
            self.assertIsNone(_values_error({**self.BASE, "audience": audience}))
        for owner in ("project", "core", "training", "panel", "deployment", "sim2real", "reward-agent"):
            self.assertIsNone(_values_error({**self.BASE, "owner": owner}))
        for doc_type, statuses in self.STATUSES.items():
            for status in statuses:
                self.assertIsNone(_values_error({**self.BASE, "type": doc_type, "status": status}))
        for changes in (
            {"lang": "zh-tw"}, {"audience": "everyone"}, {"owner": "unknown"}, {"type": "manual"},
            {"type": "explanation", "status": "accepted"}, {"type": "decision", "status": "active"}, {"type": "design", "status": "active"},
            {"type": "plan", "status": "accepted"}, {"type": "roadmap", "status": "draft"}, {"type": "release", "status": "active"},
            {"type": "experiment-summary", "status": "active"}, {"type": "audit", "status": "active"},
        ):
            self.assertIsNotNone(_values_error({**self.BASE, **changes}))

    def test_public_invalid_metadata_integration(self):
        from tools.documentation.validator import validate_repository

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pair(root)
            path = root / "component/sample.en.md"
            path.write_text(path.read_text(encoding="utf-8").replace("audience: developer", "audience: everyone"), encoding="utf-8")
            self.assertEqual([issue.code for issue in validate_repository(root)], ["invalid-metadata"])


class Cycle08Identity(unittest.TestCase):
    def test_id_date_and_filename_language_publicly(self):
        from tools.documentation.validator import validate_repository

        invalid = ({"doc_id": "Upper"}, {"doc_id": "-leading"}, {"doc_id": "trailing-"}, {"doc_id": "double--hyphen"}, {"reviewed": "2026-2-03"}, {"reviewed": "2026-02-30"}, {"reviewed": "not-a-date"}, {"lang": "zh-TW"})
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for index, changes in enumerate(invalid):
                stem = f"identity-{index}"
                (root / f"{stem}.en.md").write_text(text(**{"doc_id": stem, **changes}), encoding="utf-8")
                (root / f"{stem}.zh-TW.md").write_text(text(doc_id=stem, title="範例", lang="zh-TW"), encoding="utf-8")
            self.assertEqual([issue.code for issue in validate_repository(root)], ["invalid-metadata"] * len(invalid))


class Cycle09Locations(unittest.TestCase):
    SECTIONS = {
        "operators": ("operator", {"tutorial", "how-to", "reference", "explanation", "safety", "troubleshooting"}),
        "developers": ("developer", {"tutorial", "how-to", "reference", "explanation", "safety", "troubleshooting"}),
        "reference": ("shared", {"reference"}), "decisions": ("developer", {"decision"}), "designs": ("developer", {"design"}),
        "plans": ("developer", {"plan"}), "roadmap": ("shared", {"roadmap"}), "releases": ("shared", {"release"}),
        "research": ("developer", {"experiment-summary", "audit", "explanation"}), "governance": ("developer", {"reference"}),
    }

    def test_exhaustive_location_helper_matrix(self):
        from tools.documentation.validator import _location_error

        self.assertIsNone(_location_error(Path("docs/index.en.md"), "index", {"audience": "shared", "type": "index"}))
        self.assertIsNotNone(_location_error(Path("docs/index.en.md"), "index", {"audience": "developer", "type": "index"}))
        self.assertIsNotNone(_location_error(Path("docs/index.en.md"), "index", {"audience": "shared", "type": "reference"}))
        for section, (audience, types) in self.SECTIONS.items():
            portal = Path(f"docs/{section}/index.en.md")
            self.assertIsNone(_location_error(portal, "index", {"audience": audience, "type": "index"}))
            wrong = "shared" if audience != "shared" else "developer"
            self.assertIsNotNone(_location_error(portal, "index", {"audience": wrong, "type": "index"}))
            self.assertIsNotNone(_location_error(portal, "index", {"audience": audience, "type": sorted(types)[0]}))
            for doc_type in types:
                self.assertIsNone(_location_error(Path(f"docs/{section}/topic-{doc_type}.en.md"), f"topic-{doc_type}", {"audience": audience, "type": doc_type}))
            self.assertIsNotNone(_location_error(Path(f"docs/{section}/wrong-audience.en.md"), "wrong-audience", {"audience": wrong, "type": sorted(types)[0]}))
            self.assertIsNotNone(_location_error(Path(f"docs/{section}/wrong-type.en.md"), "wrong-type", {"audience": audience, "type": "release" if "release" not in types else "decision"}))
        self.assertIsNotNone(_location_error(Path("docs/operators/nested/index.en.md"), "index", {"audience": "operator", "type": "index"}))
        self.assertIsNotNone(_location_error(Path("docs/operators/topic.en.md"), "topic", {"audience": "operator", "type": "index"}))
        self.assertIsNotNone(_location_error(Path("docs/orphan.en.md"), "orphan", {"audience": "shared", "type": "reference"}))
        self.assertIsNotNone(_location_error(Path("docs/unknown/topic.en.md"), "topic", {"audience": "developer", "type": "explanation"}))
        self.assertIsNone(_location_error(Path("component/topic.en.md"), "topic", {"audience": "shared", "type": "release"}))

    def test_public_root_and_every_direct_portal_branch(self):
        from tools.documentation.validator import validate_repository

        cases = [("docs/index", "shared", "index", True), ("docs/index", "developer", "index", False), ("docs/index", "shared", "reference", False)]
        for section, (audience, _types) in self.SECTIONS.items():
            wrong = "shared" if audience != "shared" else "developer"
            cases += [(f"docs/{section}/index", audience, "index", True), (f"docs/{section}/index", wrong, "index", False), (f"docs/{section}/index", audience, "reference", False)]
        observed = []
        expected = []
        for relative, audience, doc_type, valid in cases:
            with tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                pair(root, relative, audience=audience, doc_type=doc_type, status="active")
                observed.extend(issue.code for issue in validate_repository(root))
                expected.extend([] if valid else ["invalid-location", "invalid-location"])
        self.assertEqual(observed, expected)


class Cycle10PhysicalPairs(unittest.TestCase):
    def test_presence_is_independent_of_companion_contents(self):
        from tools.documentation.validator import validate_repository

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            lone = root / "component/lone.en.md"
            lone.parent.mkdir(parents=True)
            lone.write_text(text(doc_id="lone"), encoding="utf-8")
            pair(root, "component/bad-meta", doc_id="bad-meta")
            bad_meta = root / "component/bad-meta.zh-TW.md"
            bad_meta.write_text(bad_meta.read_text(encoding="utf-8").replace("owner: project", "owner: unknown"), encoding="utf-8")
            pair(root, "component/bad-front", doc_id="bad-front")
            bad_front = root / "component/bad-front.zh-TW.md"
            bad_front.write_text("\n" + bad_front.read_text(encoding="utf-8"), encoding="utf-8")
            issues = validate_repository(root)
            self.assertEqual([issue.code for issue in issues], ["frontmatter", "invalid-metadata", "missing-pair"])
            self.assertEqual([issue.path for issue in issues if issue.code == "missing-pair"], [Path("component/lone.en.md")])


class Cycle11PairMetadata(unittest.TestCase):
    def test_localized_fields_drift_once_and_location_independence(self):
        from tools.documentation.validator import validate_repository

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pair(root, "component/localized", doc_id="localized")
            self.assertEqual(validate_repository(root), [])
            pair(root, "docs/unknown/drift", doc_id="drift")
            chinese = root / "docs/unknown/drift.zh-TW.md"
            chinese.write_text(chinese.read_text(encoding="utf-8").replace("owner: project", "owner: core"), encoding="utf-8")
            issues = validate_repository(root)
            self.assertEqual(sum(issue.code == "pair-metadata" for issue in issues), 1)
            self.assertEqual(sum(issue.code == "invalid-location" for issue in issues), 2)
            self.assertEqual(next(issue.path for issue in issues if issue.code == "pair-metadata"), Path("docs/unknown/drift.en.md"))


class Cycle12DuplicateIds(unittest.TestCase):
    def test_complete_incomplete_invalid_location_and_invalid_metadata_eligibility(self):
        from tools.documentation.validator import validate_repository

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            component = root / "component"
            component.mkdir()
            for stem in ("first", "second"):
                (component / f"{stem}.en.md").write_text(text(doc_id="shared-id"), encoding="utf-8")
            pair(root, "component/good", doc_id="unique-id")
            (component / "bad.en.md").write_text(text(doc_id="unique-id", owner="unknown"), encoding="utf-8")
            pair(root, "docs/unknown/loc-one", doc_id="location-id")
            pair(root, "docs/unknown/loc-two", doc_id="location-id")
            issues = validate_repository(root)
            duplicates = [issue for issue in issues if issue.code == "duplicate-id"]
            self.assertEqual([issue.path for issue in duplicates], [Path("component/first.en.md"), Path("docs/unknown/loc-one.en.md")])
            self.assertEqual(sum(issue.code == "missing-pair" for issue in issues), 3)
            self.assertEqual(sum(issue.code == "invalid-metadata" for issue in issues), 1)

    def test_duplicate_existing_only_in_second_locale_regression(self):
        from tools.documentation.validator import validate_repository

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pair(root, "component/one", doc_id="one-english")
            pair(root, "component/two", doc_id="two-english")
            for stem, english_id in (("one", "one-english"), ("two", "two-english")):
                chinese = root / f"component/{stem}.zh-TW.md"
                chinese.write_text(
                    chinese.read_text(encoding="utf-8").replace(f"id: {english_id}", "id: shared-chinese"),
                    encoding="utf-8",
                )
            issues = validate_repository(root)
            self.assertEqual(
                [issue.path for issue in issues if issue.code == "duplicate-id"],
                [Path("component/one.zh-TW.md")],
            )
            self.assertEqual(sum(issue.code == "pair-metadata" for issue in issues), 2)


class Cycle13AnchorStructure(unittest.TestCase):
    def test_structure_independence_and_source_line_fence_barrier(self):
        from tools.documentation.validator import validate_repository

        bodies = {
            "missing": "## Missing\n",
            "invalid": '<a id="Bad"></a>\n## Bad\n',
            "standalone": '<a id="orphan"></a>\nparagraph\n',
            "duplicate": '<a id="same"></a>\n## One\n<a id="same"></a>\n## Two\n',
            "barrier": '<a id="after"></a>\n```text\nignored\n```\n## After\n',
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for stem, body in bodies.items():
                pair(root, f"component/{stem}", doc_id=stem)
                path = root / f"component/{stem}.en.md"
                path.write_text(text(doc_id=stem, body=body), encoding="utf-8")
            pair(root, "docs/unknown/independent", doc_id="independent")
            independent = root / "docs/unknown/independent.en.md"
            independent.write_text(text(doc_id="independent", owner="unknown", body="## Missing\n"), encoding="utf-8")
            issues = validate_repository(root)
            self.assertEqual(sum(issue.code == "heading-anchor" for issue in issues), 5)
            self.assertEqual(sum(issue.code == "duplicate-anchor" for issue in issues), 1)
            self.assertEqual(sum(issue.code == "invalid-metadata" for issue in issues), 1)
            self.assertEqual(sum(issue.code == "invalid-location" for issue in issues), 1)


class Cycle14PairAnchors(unittest.TestCase):
    def test_once_independent_and_only_structure_disqualifies(self):
        from tools.documentation.validator import validate_repository

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pair(root, "component/mismatch", doc_id="mismatch")
            (root / "component/mismatch.en.md").write_text(text(doc_id="mismatch", body='<a id="one"></a>\n## One\n<a id="two"></a>\n## Two\n'), encoding="utf-8")
            (root / "component/mismatch.zh-TW.md").write_text(text(doc_id="mismatch", title="範例", lang="zh-TW", body='<a id="one"></a>\n## 壹\n<a id="three"></a>\n## 參\n'), encoding="utf-8")
            pair(root, "docs/unknown/independent-anchors", doc_id="independent-anchors")
            (root / "docs/unknown/independent-anchors.en.md").write_text(text(doc_id="independent-anchors", owner="unknown", body='<a id="english"></a>\n## English\n[bad](missing.txt)\n'), encoding="utf-8")
            (root / "docs/unknown/independent-anchors.zh-TW.md").write_text(text(doc_id="independent-anchors", title="範例", lang="zh-TW", body='<a id="chinese"></a>\n## 中文\n'), encoding="utf-8")
            pair(root, "component/disqualified", doc_id="disqualified")
            (root / "component/disqualified.en.md").write_text(text(doc_id="disqualified", body='<a id="same"></a>\n## One\n<a id="same"></a>\n## Two\n'), encoding="utf-8")
            issues = validate_repository(root)
            pair_issues = [issue for issue in issues if issue.code == "pair-anchors"]
            self.assertEqual([issue.path for issue in pair_issues], [Path("component/mismatch.en.md"), Path("docs/unknown/independent-anchors.en.md")])
            self.assertEqual(sum(issue.code == "duplicate-anchor" for issue in issues), 1)


class Cycle15Fences(unittest.TestCase):
    def test_marker_character_length_indent_and_cross_marker_helper(self):
        from tools.documentation.validator import _active_lines

        lines = (
            "   ````python", "## hidden", "~~~", "[x](missing.txt)", "`````", "visible",
            "~~~", '<a id="hidden"></a>', "## hidden", "   ~~~   ",
            "    ```", "## visible",
            "```", "hidden", "``", "still hidden", "```", "visible again",
        )
        self.assertEqual(
            _active_lines(lines),
            (False, False, False, False, False, True, False, False, False, False, True, True, False, False, False, False, False, True),
        )

    def test_all_markdown_syntax_inside_fences_is_ignored_publicly(self):
        from tools.documentation.validator import validate_repository

        hidden = (
            "~~~markdown\n## Hidden\n<a id=\"Bad\"></a>\n[inline](missing.txt)\n![image](missing.png)\n"
            "[full][missing]\n[collapsed][]\n[shortcut]\n[missing]: missing.txt\n~~~\n"
            "   ````\n## Hidden too\n<a id=\"also-bad\"></a>\n`````\n"
        )
        body = '<a id="visible"></a>\n## Visible\n' + hidden
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pair(root, body=body)
            self.assertEqual(validate_repository(root), [])


class Cycle16InlineLinks(unittest.TestCase):
    def test_real_openers_angles_balanced_destinations_and_titles(self):
        from tools.documentation.validator import _inline_destinations

        lines = (
            '[plain](target.txt "optional title")',
            "![image](image.png 'optional title')",
            "[angle](<target file.txt>)",
            "![angle image](<image file.png> \"image title\")",
            "[balanced](docs/file_(one).md)",
            '[title parens](target-two.txt "title (with parentheses)")',
            "text](missing.txt)",
            r"\[escaped](missing.txt)",
            "[malformed(missing.txt)",
            "[reference][label]",
            "[label]: definition.txt",
            "```",
            "[hidden](missing.txt)",
            "```",
        )
        self.assertEqual(
            _inline_destinations(lines),
            ("target.txt", "image.png", "target file.txt", "image file.png", "docs/file_(one).md", "target-two.txt"),
        )


class Cycle17References(unittest.TestCase):
    def test_full_collapsed_shortcut_images_normalization_definitions_and_negatives(self):
        from tools.documentation.validator import _reference_data

        lines = (
            '[My   Label]: target-one.txt "title"',
            "[collapsed]: target-two.txt 'title'",
            "[shortcut]: target-three.txt (parenthesized title)",
            "[image ref]: image.png",
            "[angle]: <target file.txt> \"title\"",
            "[text][my label] and ![alt][MY LABEL]",
            "[Collapsed][] and ![collapsed][]",
            "[Shortcut] and ![Image Ref] and [angle use][angle]",
            "[No Definition] remains plain",
            "[explicit][missing] and [missing collapsed][]",
            r"\[Shortcut] is escaped",
            "[inline](inline.txt)",
            "```",
            "[hidden][missing]",
            "[hidden]: missing.txt",
            "```",
        )
        definitions, resolved, missing = _reference_data(lines)
        self.assertEqual(definitions, ("target-one.txt", "target-two.txt", "target-three.txt", "image.png", "target file.txt"))
        self.assertEqual(resolved, ("target-one.txt", "target-one.txt", "target-two.txt", "target-two.txt", "target-three.txt", "image.png", "target file.txt"))
        self.assertEqual(missing, 2)


class Cycle18LocalTargets(unittest.TestCase):
    def test_public_inline_reference_and_all_target_rules_including_nul(self):
        from tools.documentation.validator import validate_repository

        body = r'''<a id="overview"></a>
## Overview
[existing](../existing.txt)
[space](<../target file.txt>)
[balanced](../file_(one).txt "title (with parentheses)")
[target anchor](../target.md#present)
[current fragment](#overview)
[decoded fragment](#%6Fverview)
[external](https://example.com) [mail](mailto:user@example.com) [custom](custom:value)
[missing](../missing.txt)
[absolute](/absolute/path)
[site root](/docs/page)
[windows backslash](C:\secret\file.md)
[windows slash](C:/secret/file.md)
[unc](\\server\share\file.md)
[escape](../../outside.txt)
[directory](../directory)
[nul](bad%00.txt)
[missing anchor](../target.md#absent)
[full][ref] ![image][ref] [ref][] [ref]
[explicit missing][undefined]
[No Definition]
[ref]: ../existing.txt "title"
[baddef]: ../missing-def.txt (title)
'''
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in ("existing.txt", "target file.txt", "file_(one).txt"):
                (root / name).write_text("target", encoding="utf-8")
            (root / "target.md").write_text('<a id="present"></a>\n# Target\n', encoding="utf-8")
            (root / "directory").mkdir()
            pair(root, body=body)
            (root / "component/sample.zh-TW.md").write_text(
                text(title="範例", lang="zh-TW", body='<a id="overview"></a>\n## 概觀\n'),
                encoding="utf-8",
            )
            issues = validate_repository(root)
            self.assertEqual(sum(issue.code == "broken-link" for issue in issues), 11)
            self.assertEqual(sum(issue.code == "missing-link-anchor" for issue in issues), 1)
            self.assertTrue(any("bad%00.txt" in issue.message for issue in issues if issue.code == "broken-link"))
            self.assertFalse(any(issue.code in {"heading-anchor", "pair-anchors"} for issue in issues))


class Cycle19Aggregation(unittest.TestCase):
    def test_exact_order_once_only_and_independence_matrix(self):
        from tools.documentation.validator import validate_repository

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pair(root, "component/a", doc_id="a")
            (root / "component/a.en.md").write_text(text(doc_id="a", owner="unknown", body="## Missing\n[bad](missing.txt)\n"), encoding="utf-8")
            pair(root, "component/b", doc_id="b")
            (root / "component/b.en.md").write_text(text(doc_id="b", body='<a id="english"></a>\n## English\n[bad](missing.txt)\n'), encoding="utf-8")
            (root / "component/b.zh-TW.md").write_text(text(doc_id="b", title="範例", lang="zh-TW", body='<a id="chinese"></a>\n## 中文\n'), encoding="utf-8")
            pair(root, "docs/unknown/c", doc_id="c")
            chinese = root / "docs/unknown/c.zh-TW.md"
            chinese.write_text(chinese.read_text(encoding="utf-8").replace("owner: project", "owner: core"), encoding="utf-8")
            pair(root, "component/d", doc_id="d")
            malformed = root / "component/d.en.md"
            malformed.write_text("\n" + malformed.read_text(encoding="utf-8"), encoding="utf-8")
            first = validate_repository(root)
            second = validate_repository(root)
            expected = [
                ("component/a.en.md", "broken-link"), ("component/a.en.md", "heading-anchor"), ("component/a.en.md", "invalid-metadata"),
                ("component/b.en.md", "broken-link"), ("component/b.en.md", "pair-anchors"),
                ("component/d.en.md", "frontmatter"),
                ("docs/unknown/c.en.md", "invalid-location"), ("docs/unknown/c.en.md", "pair-metadata"),
                ("docs/unknown/c.zh-TW.md", "invalid-location"),
            ]
            self.assertEqual([(issue.path.as_posix(), issue.code) for issue in first], expected)
            self.assertEqual(first, second)


class Cycle22ActualRepository(unittest.TestCase):
    def test_actual_current_repository_passes(self):
        from tools.documentation.validator import validate_repository

        repository = Path(__file__).resolve().parents[3]
        self.assertEqual(validate_repository(repository), [])


class ReviewCoverageAcceptance(unittest.TestCase):
    def test_missing_inline_image_and_invalid_location_link_compose(self):
        from tools.documentation.validator import validate_repository

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pair(root, "docs/unknown/image-link", doc_id="image-link")
            english = root / "docs/unknown/image-link.en.md"
            english.write_text(
                text(doc_id="image-link", body='<a id="overview"></a>\n## Overview\n![missing](missing.png)\n'),
                encoding="utf-8",
            )
            issues = validate_repository(root)
            self.assertEqual(sum(issue.code == "broken-link" for issue in issues), 1)
            self.assertEqual(sum(issue.code == "invalid-location" for issue in issues), 2)


if __name__ == "__main__":
    unittest.main()
