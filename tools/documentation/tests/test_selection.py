from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path

from tools.documentation.validator import Issue


class NameStatusParsingTests(unittest.TestCase):
    def test_nul_records_cover_all_selected_statuses_and_path_characters(self) -> None:
        try:
            from tools.documentation.selection import parse_name_status
        except ImportError as error:
            self.fail(f"selection parser is missing: {error}")

        data = (
            b"A\0docs/added.en.md\0"
            b"M\0docs/space name.zh-TW.md\0"
            b"T\0docs/type.en.md\0"
            b"D\0docs/deleted.zh-TW.md\0"
            b"R100\0docs/old.en.md\0docs/renamed.en.md\0"
            b"C075\0docs/source.zh-TW.md\0docs/\xe8\xa4\x87\xe8\xa3\xbd.zh-TW.md\0"
        )

        self.assertEqual(
            parse_name_status(data),
            {
                Path("docs/added.en.md"),
                Path("docs/space name.zh-TW.md"),
                Path("docs/type.en.md"),
                Path("docs/deleted.zh-TW.md"),
                Path("docs/old.en.md"),
                Path("docs/renamed.en.md"),
                Path("docs/source.zh-TW.md"),
                Path("docs/複製.zh-TW.md"),
            },
        )

    def test_malformed_or_unsupported_records_are_operational_errors(self) -> None:
        try:
            from tools.documentation.errors import DocumentationOperationError
            from tools.documentation.selection import parse_name_status
        except ImportError as error:
            self.fail(f"selection operational boundary is missing: {error}")

        malformed = (
            b"M\0missing-final-terminator",
            b"R100\0only-old\0",
            b"X\0unknown\0",
            b"M\0\0",
        )
        for data in malformed:
            with self.subTest(data=data):
                with self.assertRaisesRegex(
                    DocumentationOperationError,
                    "invalid Git name-status output",
                ):
                    parse_name_status(data)


class ChangedPairTests(unittest.TestCase):
    def test_paired_change_forms_pass_and_noncanonical_paths_are_ignored(self) -> None:
        try:
            from tools.documentation.selection import changed_pair_issues
        except ImportError as error:
            self.fail(f"changed-pair checker is missing: {error}")

        changes = {
            Path("docs/add.en.md"),
            Path("docs/add.zh-TW.md"),
            Path("docs/edit.en.md"),
            Path("docs/edit.zh-TW.md"),
            Path("docs/old-name.en.md"),
            Path("docs/old-name.zh-TW.md"),
            Path("docs/new-name.en.md"),
            Path("docs/new-name.zh-TW.md"),
            Path("docs/delete.en.md"),
            Path("docs/delete.zh-TW.md"),
            Path("README.md"),
            Path("component/README.md"),
            Path("docs/governance/templates/reference.en.md"),
            Path("docs/data.csv"),
            Path(".agents/skills/writing/SKILL.md"),
            Path(".agents/skills/writing/guide.en.md"),
            Path(".claude/skills/review/guide.zh-TW.md"),
            Path("docs/Bad.en.md"),
            Path("docs/legacy.md"),
        }

        self.assertEqual(changed_pair_issues(changes), [])

    def test_unpaired_changes_fail_once_per_stem_at_lexical_representative(self) -> None:
        try:
            from tools.documentation.selection import changed_pair_issues
        except ImportError as error:
            self.fail(f"changed-pair checker is missing: {error}")

        self.assertEqual(
            changed_pair_issues(
                {
                    Path("z/only-zh.zh-TW.md"),
                    Path("a/only-en.en.md"),
                    Path("router/README.md"),
                }
            ),
            [
                Issue(
                    Path("a/only-en.en.md"),
                    "changed-pair",
                    "locale companion is not in selected change set",
                ),
                Issue(
                    Path("z/only-zh.zh-TW.md"),
                    "changed-pair",
                    "locale companion is not in selected change set",
                ),
            ],
        )


class SelectionGitTests(unittest.TestCase):
    def git(self, repo: Path, *arguments: str) -> subprocess.CompletedProcess[bytes]:
        return subprocess.run(
            ["git", *arguments],
            cwd=repo,
            check=True,
            capture_output=True,
        )

    def test_staged_paths_come_from_real_nul_git_diff(self) -> None:
        try:
            from tools.documentation.selection import select_staged_paths
        except ImportError as error:
            self.fail(f"staged selector is missing: {error}")

        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            self.git(repo, "init", "--quiet")
            self.git(repo, "config", "user.email", "docs@example.com")
            self.git(repo, "config", "user.name", "Docs Tests")
            (repo / "docs").mkdir()
            tracked = repo / "docs/guide.en.md"
            tracked.write_text("before\n", encoding="utf-8")
            self.git(repo, "add", "docs/guide.en.md")
            self.git(repo, "commit", "--quiet", "-m", "base")

            tracked.write_text("after\n", encoding="utf-8")
            unicode_path = repo / "docs/空 白.zh-TW.md"
            unicode_path.write_text("new\n", encoding="utf-8")
            self.git(repo, "add", "--all")

            self.assertEqual(
                select_staged_paths(repo),
                {Path("docs/guide.en.md"), Path("docs/空 白.zh-TW.md")},
            )

    def test_changed_from_uses_real_committed_diff_and_rejects_invalid_ref(self) -> None:
        try:
            from tools.documentation.errors import DocumentationOperationError
            from tools.documentation.selection import select_changed_paths
        except ImportError as error:
            self.fail(f"changed-from selector is missing: {error}")

        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            self.git(repo, "init", "--quiet")
            self.git(repo, "config", "user.email", "docs@example.com")
            self.git(repo, "config", "user.name", "Docs Tests")
            (repo / "docs").mkdir()
            (repo / "docs/base.txt").write_text("base\n", encoding="utf-8")
            self.git(repo, "add", "docs/base.txt")
            self.git(repo, "commit", "--quiet", "-m", "base")
            base = self.git(repo, "rev-parse", "HEAD").stdout.decode().strip()

            changed = repo / "docs/committed name.en.md"
            changed.write_text("committed\n", encoding="utf-8")
            self.git(repo, "add", "docs/committed name.en.md")
            self.git(repo, "commit", "--quiet", "-m", "change")

            self.assertEqual(
                select_changed_paths(repo, base),
                {Path("docs/committed name.en.md")},
            )
            malicious = ";touch injected"
            with self.assertRaisesRegex(
                DocumentationOperationError,
                "^invalid Git reference$",
            ):
                select_changed_paths(repo, malicious)
            self.assertFalse((repo / "injected").exists())


if __name__ == "__main__":
    unittest.main()
