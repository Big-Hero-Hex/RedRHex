from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


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
                ("validate", "--staged"),
                ("validate", "--changed-from", "main"),
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


if __name__ == "__main__":
    unittest.main()
