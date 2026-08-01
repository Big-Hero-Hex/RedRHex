from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest

from tools.documentation.tests.test_validator import write_document


SOURCE_ROOT = Path(__file__).resolve().parents[3]


class DocumentationCliTests(unittest.TestCase):
    def run_cli(self, root: Path, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
        environment = {**os.environ, "PYTHONPATH": str(SOURCE_ROOT)}
        return subprocess.run(
            [sys.executable, "-m", "tools.documentation", "validate", "--all"],
            cwd=cwd or root,
            env=environment,
            text=True,
            capture_output=True,
            check=False,
        )

    def test_validate_all_reports_success(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            write_document(root, "guides/sample-guide.en.md")
            write_document(root, "guides/sample-guide.zh-TW.md", title="範例指南", lang="zh-TW")

            result = self.run_cli(root)

            self.assertEqual(result.returncode, 0)
            self.assertEqual(result.stdout, "documentation validation passed (2 documents)\n")
            self.assertEqual(result.stderr, "")

    def test_validate_all_reports_sorted_failures(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            write_document(root, "guides/sample-guide.en.md")

            result = self.run_cli(root)

            self.assertEqual(result.returncode, 1)
            self.assertIn("guides/sample-guide.en.md: missing-pair:", result.stderr)
            self.assertTrue(result.stderr.endswith("documentation validation failed (1 issues)\n"))

    def test_validate_all_uses_the_containing_workspace_root(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / ".git").mkdir()
            write_document(root, "root-only.en.md")
            write_document(root, "guides/sample-guide.en.md")
            write_document(root, "guides/sample-guide.zh-TW.md", title="範例指南", lang="zh-TW")

            result = self.run_cli(root, root / "guides")

            self.assertEqual(result.returncode, 1)
            self.assertIn("root-only.en.md: missing-pair:", result.stderr)
