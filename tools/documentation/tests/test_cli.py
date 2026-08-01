from __future__ import annotations

import contextlib
import datetime
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from tools.documentation.cli import _parser


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _document(lang: str, title: str) -> str:
    return (
        "---\n"
        "id: guide\n"
        f"title: {title}\n"
        f"lang: {lang}\n"
        "audience: developer\n"
        "type: explanation\n"
        "status: active\n"
        "owner: project\n"
        "last_reviewed: 2026-08-01\n"
        "---\n\n"
        '<a id="overview"></a>\n'
        "## Overview\n"
    )


class DocumentationCliTests(unittest.TestCase):
    def git(self, repo: Path, *arguments: str) -> subprocess.CompletedProcess[bytes]:
        return subprocess.run(
            ["git", *arguments],
            cwd=repo,
            check=True,
            capture_output=True,
        )

    def initialize_git_repository(self, repo: Path) -> None:
        self.git(repo, "init", "--quiet")
        self.git(repo, "config", "user.email", "docs@example.com")
        self.git(repo, "config", "user.name", "Docs Tests")

    def run_cli(self, cwd: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
        environment = os.environ.copy()
        existing_pythonpath = environment.get("PYTHONPATH")
        environment["PYTHONPATH"] = os.pathsep.join(
            part
            for part in (str(PROJECT_ROOT), existing_pythonpath)
            if part
        )
        return subprocess.run(
            [sys.executable, "-m", "tools.documentation", *arguments],
            cwd=cwd,
            env=environment,
            text=True,
            capture_output=True,
            check=False,
        )

    def test_success_failure_and_exact_command_surface(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / ".git").mkdir()
            (root / "component").mkdir()
            (root / "component/guide.en.md").write_text(
                _document("en", "Guide"), encoding="utf-8"
            )
            (root / "component/guide.zh-TW.md").write_text(
                _document("zh-TW", "指南"), encoding="utf-8"
            )
            success = self.run_cli(root, "validate", "--all")
            self.assertEqual(success.returncode, 0)
            self.assertEqual(
                success.stdout,
                "documentation validation passed (2 documents)\n",
            )
            self.assertEqual(success.stderr, "")

            (root / "component/guide.en.md").unlink()
            (root / "component/guide.zh-TW.md").unlink()
            (root / "bad.en.md").write_text("not frontmatter\n", encoding="utf-8")
            failure = self.run_cli(root, "validate", "--all")
            self.assertEqual(failure.returncode, 1)
            self.assertEqual(failure.stdout, "")
            self.assertEqual(
                failure.stderr,
                "bad.en.md: frontmatter: invalid frontmatter\n"
                "bad.en.md: missing-pair: missing locale companion: zh-TW\n"
                "documentation validation failed (2 issues)\n",
            )

            malformed_commands = (
                (),
                ("validate",),
                ("validate", "--a"),
                ("validate", "--al"),
                ("inventory",),
                ("stage-site",),
                ("validate", "--all", "extra"),
            )
            for arguments in malformed_commands:
                with self.subTest(arguments=arguments):
                    malformed = self.run_cli(root, *arguments)
                    self.assertEqual(malformed.returncode, 2)
                    self.assertEqual(malformed.stdout, "")
                    self.assertIn("usage:", malformed.stderr)

    def test_complete_parser_surface_and_invalid_shapes(self) -> None:
        valid_shapes = (
            ("validate", "--all"),
            ("validate", "--staged"),
            ("validate", "--changed-from", "base"),
            ("inventory", "--format", "json"),
            ("stage-site", "--output", "target"),
        )
        for arguments in valid_shapes:
            with self.subTest(arguments=arguments):
                self.assertEqual(_parser().parse_args(arguments).command, arguments[0])

        invalid_shapes = (
            ("validate",),
            ("validate", "--all", "--staged"),
            ("validate", "--changed-from"),
            ("validate", "--sta"),
            ("inventory", "--format", "yaml"),
            ("inventory", "--format", "json", "extra"),
            ("stage-site", "--output"),
            ("stage-site", "--out", "target"),
        )
        for arguments in invalid_shapes:
            with self.subTest(arguments=arguments):
                with contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit) as raised:
                        _parser().parse_args(arguments)
                self.assertEqual(raised.exception.code, 2)

    def test_repeated_stable_options_are_invalid_argparse_shapes(self) -> None:
        repeated_shapes = (
            ("validate", "--all", "--all"),
            ("validate", "--staged", "--staged"),
            (
                "validate",
                "--changed-from",
                "base",
                "--changed-from",
                "other",
            ),
            ("validate", "--changed-from=base", "--changed-from=other"),
            ("inventory", "--format", "json", "--format", "json"),
            ("inventory", "--format=json", "--format=json"),
            ("stage-site", "--output", "first", "--output", "second"),
            ("stage-site", "--output=first", "--output=second"),
        )
        for arguments in repeated_shapes:
            with self.subTest(arguments=arguments):
                with contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit) as raised:
                        _parser().parse_args(arguments)
                self.assertEqual(raised.exception.code, 2)

    def test_nested_directory_finds_git_directory_and_git_file_roots(self) -> None:
        for marker_kind in ("directory", "file"):
            with self.subTest(marker_kind=marker_kind):
                with tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    if marker_kind == "directory":
                        (root / ".git").mkdir()
                    else:
                        (root / ".git").write_text(
                            "gitdir: /tmp/shared.git/worktrees/example\n",
                            encoding="utf-8",
                        )
                    (root / "bad.en.md").write_text(
                        "not frontmatter\n", encoding="utf-8"
                    )
                    nested = root / "one/two/three"
                    nested.mkdir(parents=True)

                    result = self.run_cli(nested, "validate", "--all")
                    self.assertEqual(result.returncode, 1)
                    self.assertEqual(result.stdout, "")
                    self.assertEqual(
                        result.stderr,
                        "bad.en.md: frontmatter: invalid frontmatter\n"
                        "bad.en.md: missing-pair: missing locale companion: zh-TW\n"
                        "documentation validation failed (2 issues)\n",
                    )

    def test_real_all_and_staged_selectors_combine_full_and_pair_issues(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            self.initialize_git_repository(repo)
            component = repo / "component"
            component.mkdir()
            en_path = component / "guide.en.md"
            zh_path = component / "guide.zh-TW.md"
            en_path.write_text(_document("en", "Guide"), encoding="utf-8")
            zh_path.write_text(_document("zh-TW", "指南"), encoding="utf-8")
            self.git(repo, "add", "component")
            self.git(repo, "commit", "--quiet", "-m", "base")

            all_result = self.run_cli(repo, "validate", "--all")
            self.assertEqual(all_result.returncode, 0)
            self.assertEqual(
                all_result.stdout,
                "documentation validation passed (2 documents)\n",
            )

            en_path.write_text(
                _document("en", "Guide").replace(
                    '<a id="overview"></a>\n## Overview\n',
                    "## Missing anchor\n",
                ),
                encoding="utf-8",
            )
            self.git(repo, "add", "component/guide.en.md")
            failed = self.run_cli(repo, "validate", "--staged")
            self.assertEqual(failed.returncode, 1)
            self.assertEqual(failed.stdout, "")
            self.assertEqual(
                failed.stderr,
                "component/guide.en.md: changed-pair: locale companion is not in selected change set\n"
                "component/guide.en.md: heading-anchor: heading lacks preceding explicit anchor\n"
                "documentation validation failed (2 issues)\n",
            )

            en_path.write_text(
                _document("en", "Guide") + "English change.\n",
                encoding="utf-8",
            )
            zh_path.write_text(
                _document("zh-TW", "指南") + "中文變更。\n",
                encoding="utf-8",
            )
            self.git(repo, "add", "component")
            passed = self.run_cli(repo, "validate", "--staged")
            self.assertEqual(passed.returncode, 0)
            self.assertEqual(
                passed.stdout,
                "documentation validation passed (2 documents)\n",
            )
            self.assertEqual(passed.stderr, "")

    def test_real_changed_from_selector_requires_both_committed_locales(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            self.initialize_git_repository(repo)
            component = repo / "component"
            component.mkdir()
            en_path = component / "guide.en.md"
            zh_path = component / "guide.zh-TW.md"
            en_path.write_text(_document("en", "Guide"), encoding="utf-8")
            zh_path.write_text(_document("zh-TW", "指南"), encoding="utf-8")
            self.git(repo, "add", "component")
            self.git(repo, "commit", "--quiet", "-m", "base")
            base = self.git(repo, "rev-parse", "HEAD").stdout.decode().strip()

            en_path.write_text(
                _document("en", "Guide") + "English change.\n",
                encoding="utf-8",
            )
            self.git(repo, "add", "component/guide.en.md")
            self.git(repo, "commit", "--quiet", "-m", "english")
            failed = self.run_cli(repo, "validate", "--changed-from", base)
            self.assertEqual(failed.returncode, 1)
            self.assertEqual(
                failed.stderr,
                "component/guide.en.md: changed-pair: locale companion is not in selected change set\n"
                "documentation validation failed (1 issues)\n",
            )

            zh_path.write_text(
                _document("zh-TW", "指南") + "中文變更。\n",
                encoding="utf-8",
            )
            self.git(repo, "add", "component/guide.zh-TW.md")
            self.git(repo, "commit", "--quiet", "-m", "chinese")
            passed = self.run_cli(repo, "validate", "--changed-from", base)
            self.assertEqual(passed.returncode, 0)
            self.assertEqual(
                passed.stdout,
                "documentation validation passed (2 documents)\n",
            )
            self.assertEqual(passed.stderr, "")

    def test_invalid_git_ref_is_concise_operational_failure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            self.initialize_git_repository(repo)

            result = self.run_cli(
                repo,
                "validate",
                "--changed-from",
                ";touch injected",
            )

            self.assertEqual(result.returncode, 1)
            self.assertEqual(result.stdout, "")
            self.assertEqual(
                result.stderr,
                "documentation error: invalid Git reference\n",
            )
            self.assertNotIn("Traceback", result.stderr)
            self.assertFalse((repo / "injected").exists())

    def test_inventory_json_is_validation_gated_utf8_and_never_partial(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            (repo / ".git").mkdir()
            component = repo / "component"
            component.mkdir()
            (component / "guide.en.md").write_text(
                _document("en", "Guide"), encoding="utf-8"
            )
            (component / "guide.zh-TW.md").write_text(
                _document("zh-TW", "指南"), encoding="utf-8"
            )

            valid = self.run_cli(repo, "inventory", "--format", "json")
            self.assertEqual(valid.returncode, 0)
            self.assertEqual(valid.stderr, "")
            payload = json.loads(valid.stdout)
            self.assertEqual(
                valid.stdout,
                json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2)
                + "\n",
            )
            self.assertIn("指南", valid.stdout)
            self.assertEqual(payload["document_count"], 2)

            (component / "guide.en.md").write_text(
                "invalid frontmatter\n", encoding="utf-8"
            )
            invalid = self.run_cli(repo, "inventory", "--format", "json")
            self.assertEqual(invalid.returncode, 1)
            self.assertEqual(invalid.stdout, "")
            self.assertEqual(
                invalid.stderr,
                "component/guide.en.md: frontmatter: invalid frontmatter\n"
                "documentation validation failed (1 issues)\n",
            )

            tomorrow = (datetime.date.today() + datetime.timedelta(days=1)).isoformat()
            (component / "guide.en.md").write_text(
                _document("en", "Guide").replace("2026-08-01", tomorrow),
                encoding="utf-8",
            )
            (component / "guide.zh-TW.md").write_text(
                _document("zh-TW", "指南").replace("2026-08-01", tomorrow),
                encoding="utf-8",
            )
            future = self.run_cli(repo, "inventory", "--format", "json")
            self.assertEqual(future.returncode, 1)
            self.assertEqual(future.stdout, "")
            self.assertEqual(
                future.stderr,
                "documentation error: last_reviewed is after inventory as_of: "
                "component/guide.en.md\n",
            )
            self.assertNotIn("Traceback", future.stderr)

    def test_stage_site_cli_success_relative_output_and_operational_failures(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            top = Path(directory)
            repo = top / "repo"
            (repo / ".git").mkdir(parents=True)
            docs = repo / "docs"
            docs.mkdir()
            (docs / "site-manifest.json").write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "sources": [{"destination": ".", "source": "docs"}],
                    }
                ),
                encoding="utf-8",
            )
            (docs / "guide.en.md").write_text("english\n", encoding="utf-8")
            (docs / "guide.zh-TW.md").write_text("中文\n", encoding="utf-8")

            success = self.run_cli(repo, "stage-site", "--output", "../staged")
            self.assertEqual(success.returncode, 0)
            self.assertEqual(
                success.stdout,
                "documentation site staged (2 documents)\n",
            )
            self.assertEqual(success.stderr, "")
            self.assertEqual((top / "staged/guide.en.md").read_text(), "english\n")
            self.assertEqual((top / "staged/guide.zh-TW.md").read_text(), "中文\n")

            (docs / "site-manifest.json").write_text("{invalid", encoding="utf-8")
            malformed_output = top / "malformed-output"
            malformed = self.run_cli(
                repo,
                "stage-site",
                "--output",
                str(malformed_output),
            )
            self.assertEqual(malformed.returncode, 1)
            self.assertEqual(malformed.stdout, "")
            self.assertEqual(
                malformed.stderr,
                "documentation error: invalid documentation site manifest\n",
            )
            self.assertNotIn("Traceback", malformed.stderr)
            self.assertFalse(malformed_output.exists())

            (docs / "site-manifest.json").write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "sources": [{"destination": ".", "source": "docs"}],
                    }
                ),
                encoding="utf-8",
            )
            nonempty_output = top / "nonempty"
            nonempty_output.mkdir()
            (nonempty_output / "keep.txt").write_text("keep\n", encoding="utf-8")
            unsafe = self.run_cli(
                repo,
                "stage-site",
                "--output",
                str(nonempty_output),
            )
            self.assertEqual(unsafe.returncode, 1)
            self.assertEqual(unsafe.stdout, "")
            self.assertEqual(
                unsafe.stderr,
                "documentation error: unsafe or nonempty site staging output\n",
            )
            self.assertEqual(
                sorted(path.name for path in nonempty_output.iterdir()),
                ["keep.txt"],
            )

    def test_stage_site_cli_rejects_nul_destination_without_partial_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            top = Path(directory)
            repo = top / "repo"
            (repo / ".git").mkdir(parents=True)
            docs = repo / "docs"
            docs.mkdir()
            (docs / "site-manifest.json").write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "sources": [{"destination": "\0", "source": "docs"}],
                    }
                ),
                encoding="utf-8",
            )
            (docs / "guide.en.md").write_text("english\n", encoding="utf-8")
            output = top / "nul-output"

            result = self.run_cli(
                repo,
                "stage-site",
                "--output",
                str(output),
            )

            self.assertEqual(result.returncode, 1)
            self.assertEqual(result.stdout, "")
            self.assertEqual(
                result.stderr,
                "documentation error: invalid documentation site manifest\n",
            )
            self.assertNotIn("Traceback", result.stderr)
            self.assertFalse(output.exists())

    def test_stage_site_cli_preflights_missing_output_below_a_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            top = Path(directory)
            repo = top / "repo"
            (repo / ".git").mkdir(parents=True)
            docs = repo / "docs"
            docs.mkdir()
            (docs / "site-manifest.json").write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "sources": [{"destination": ".", "source": "docs"}],
                    }
                ),
                encoding="utf-8",
            )
            (docs / "guide.en.md").write_text("english\n", encoding="utf-8")
            blocking_ancestor = top / "regular-parent"
            blocking_ancestor.write_text("keep\n", encoding="utf-8")

            result = self.run_cli(
                repo,
                "stage-site",
                "--output",
                str(blocking_ancestor / "child"),
            )

            self.assertEqual(result.returncode, 1)
            self.assertEqual(result.stdout, "")
            self.assertEqual(
                result.stderr,
                "documentation error: unsafe or nonempty site staging output\n",
            )
            self.assertNotIn("Traceback", result.stderr)
            self.assertEqual(
                blocking_ancestor.read_text(encoding="utf-8"),
                "keep\n",
            )


if __name__ == "__main__":
    unittest.main()
