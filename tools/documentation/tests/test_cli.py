import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class DocumentationCliTests(unittest.TestCase):
    def setUp(self):
        self._temporary_directory = tempfile.TemporaryDirectory()
        self.repo = Path(self._temporary_directory.name)
        (self.repo / ".git").mkdir()
        self.project_root = Path(__file__).resolve().parents[3]

    def tearDown(self):
        self._temporary_directory.cleanup()

    def write_document(self, relative_path, *, lang, document_id="guide", body=""):
        path = self.repo / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "\n".join(
                [
                    "---",
                    f"id: {document_id}",
                    "title: Guide",
                    f"lang: {lang}",
                    "audience: developer",
                    "type: explanation",
                    "status: active",
                    "owner: project",
                    "last_reviewed: 2026-08-01",
                    "---",
                    body,
                ]
            ),
            encoding="utf-8",
        )

    def write_pair(self, relative_stem="component/guide"):
        self.write_document(f"{relative_stem}.en.md", lang="en")
        self.write_document(f"{relative_stem}.zh-TW.md", lang="zh-TW")

    def run_cli(self, *arguments, cwd=None):
        environment = os.environ.copy()
        existing_pythonpath = environment.get("PYTHONPATH")
        environment["PYTHONPATH"] = str(self.project_root)
        if existing_pythonpath:
            environment["PYTHONPATH"] += os.pathsep + existing_pythonpath
        return subprocess.run(
            [sys.executable, "-m", "tools.documentation", *arguments],
            cwd=cwd or self.repo,
            env=environment,
            text=True,
            capture_output=True,
            check=False,
        )

    def test_validate_all_success_output_and_exit_zero(self):
        self.write_pair()

        result = self.run_cli("validate", "--all")

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, "documentation validation passed (2 documents)\n")
        self.assertEqual(result.stderr, "")

    def test_validate_all_failure_output_and_exit_one(self):
        (self.repo / "Bad.en.md").write_text("invalid", encoding="utf-8")

        result = self.run_cli("validate", "--all")

        self.assertEqual(result.returncode, 1)
        self.assertEqual(result.stdout, "")
        self.assertEqual(
            result.stderr,
            "Bad.en.md: invalid-name: invalid documentation filename\n"
            "documentation validation failed (1 issue)\n",
        )

    def test_unimplemented_or_invalid_command_shapes_use_argparse_exit_two(self):
        invalid_shapes = [
            (),
            ("validate",),
            ("validate", "--staged"),
            ("validate", "--changed-from", "HEAD"),
            ("inventory",),
            ("stage-site",),
        ]
        for arguments in invalid_shapes:
            with self.subTest(arguments=arguments):
                result = self.run_cli(*arguments)
                self.assertEqual(result.returncode, 2)

    def test_nested_cli_finds_worktree_root_for_git_directory_and_file(self):
        (self.repo / "Bad.en.md").write_text("invalid", encoding="utf-8")
        nested = self.repo / "component/deep/nested"
        nested.mkdir(parents=True)

        directory_marker_result = self.run_cli("validate", "--all", cwd=nested)

        self.assertEqual(directory_marker_result.returncode, 1)
        self.assertIn("Bad.en.md: invalid-name", directory_marker_result.stderr)

        shutil.rmtree(self.repo / ".git")
        (self.repo / ".git").write_text("gitdir: /tmp/shared/worktrees/example\n", encoding="utf-8")

        file_marker_result = self.run_cli("validate", "--all", cwd=nested)

        self.assertEqual(file_marker_result.returncode, 1)
        self.assertIn("Bad.en.md: invalid-name", file_marker_result.stderr)


if __name__ == "__main__":
    unittest.main()
