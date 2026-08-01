import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from tools.documentation.tests.test_validator import pair


REPOSITORY = Path(__file__).resolve().parents[3]


def run_cli(cwd, *arguments):
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(REPOSITORY) + os.pathsep + environment.get("PYTHONPATH", "")
    return subprocess.run(
        [sys.executable, "-m", "tools.documentation", *arguments],
        cwd=cwd,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )


class Cycle20Cli(unittest.TestCase):
    def test_success_contract_and_document_count(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / ".git").mkdir()
            pair(root)
            result = run_cli(root, "validate", "--all")
            self.assertEqual(result.returncode, 0)
            self.assertEqual(result.stdout, "documentation validation passed (2 documents)\n")
            self.assertEqual(result.stderr, "")

    def test_failure_contract(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / ".git").mkdir()
            (root / "Bad.en.md").write_text("bad", encoding="utf-8")
            result = run_cli(root, "validate", "--all")
            self.assertEqual(result.returncode, 1)
            self.assertEqual(result.stdout, "")
            self.assertEqual(
                result.stderr,
                "Bad.en.md: invalid-name: invalid canonical filename\n"
                "documentation validation failed (1 issue)\n",
            )

    def test_only_exact_validate_all_and_no_abbreviations(self):
        with tempfile.TemporaryDirectory() as temporary:
            invalid_shapes = (
                (), ("validate",), ("validate", "--all", "extra"),
                ("validate", "--staged"), ("validate", "--changed-from", "main"),
                ("inventory",), ("stage-site",), ("validate", "--a"), ("validate", "--al"),
            )
            self.assertEqual(
                [run_cli(temporary, *shape).returncode for shape in invalid_shapes],
                [2] * len(invalid_shapes),
            )


class Cycle21RepositoryRoot(unittest.TestCase):
    def test_nested_start_for_git_directory_and_worktree_git_file(self):
        results = []
        for git_kind in ("directory", "file"):
            with tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary) / "repository"
                nested = root / "nested/deep"
                nested.mkdir(parents=True)
                if git_kind == "directory":
                    (root / ".git").mkdir()
                else:
                    (root / ".git").write_text("gitdir: /tmp/shared/worktrees/example\n", encoding="utf-8")
                (root / "Bad.en.md").write_text("bad", encoding="utf-8")
                results.append(run_cli(nested, "validate", "--all"))
        self.assertEqual([result.returncode for result in results], [1, 1])
        self.assertTrue(all("Bad.en.md: invalid-name" in result.stderr for result in results))


if __name__ == "__main__":
    unittest.main()
