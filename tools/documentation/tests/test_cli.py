from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest

from tools.documentation.cli import main


class CliTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.repo = Path(self.temp_dir.name)

    def write_pair(self) -> None:
        for locale, title in (("en", "Guide"), ("zh-TW", "指南")):
            path = self.repo / f"components/guide.{locale}.md"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                "\n".join(
                    [
                        "---",
                        "id: guide",
                        f"title: {title}",
                        f"lang: {locale}",
                        "audience: developer",
                        "type: explanation",
                        "status: active",
                        "owner: project",
                        "last_reviewed: 2026-08-01",
                        "---",
                        "",
                        '<a id="overview"></a>',
                        "## Overview",
                        "",
                    ]
                ),
                encoding="utf-8",
            )

    def run_main(self, arguments: list[str]) -> tuple[int, str, str]:
        previous_cwd = Path.cwd()
        stdout = StringIO()
        stderr = StringIO()
        try:
            os.chdir(self.repo)
            with redirect_stdout(stdout), redirect_stderr(stderr):
                exit_code = main(arguments)
        finally:
            os.chdir(previous_cwd)
        return exit_code, stdout.getvalue(), stderr.getvalue()

    def test_validate_all_success_prints_document_count_and_returns_zero(self) -> None:
        self.write_pair()

        result = self.run_main(["validate", "--all"])

        self.assertEqual(result, (0, "documentation validation passed (2 documents)\n", ""))

    def test_validate_all_failure_prints_issues_and_returns_one(self) -> None:
        self.write_pair()
        (self.repo / "components/guide.zh-TW.md").unlink()

        result = self.run_main(["validate", "--all"])

        self.assertEqual(
            result,
            (
                1,
                "",
                "components/guide.en.md: missing-pair: missing locale companion\n"
                "documentation validation failed (1 issue)\n",
            ),
        )

    def test_module_entrypoint_runs_validate_all(self) -> None:
        repo_root = Path(__file__).resolve().parents[3]

        completed = subprocess.run(
            [sys.executable, "-m", "tools.documentation", "validate", "--all"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(completed.returncode, 0)
        self.assertRegex(
            completed.stdout, r"^documentation validation passed \(\d+ documents\)\n$"
        )
        self.assertEqual(completed.stderr, "")

    def test_unimplemented_command_shapes_use_argparse_exit_two(self) -> None:
        repo_root = Path(__file__).resolve().parents[3]
        invalid_shapes = [
            [],
            ["validate"],
            ["validate", "--staged"],
            ["validate", "--changed-from", "main"],
            ["inventory"],
            ["stage-site"],
        ]

        for arguments in invalid_shapes:
            with self.subTest(arguments=arguments):
                completed = subprocess.run(
                    [sys.executable, "-m", "tools.documentation", *arguments],
                    cwd=repo_root,
                    capture_output=True,
                    text=True,
                    check=False,
                )
                self.assertEqual(completed.returncode, 2)


if __name__ == "__main__":
    unittest.main()
