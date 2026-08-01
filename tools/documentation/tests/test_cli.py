import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


WORKTREE_ROOT = Path(__file__).resolve().parents[3]


def _document(lang: str, title: str) -> str:
    return (
        "---\n"
        "id: cli-sample\n"
        f"title: {title}\n"
        f"lang: {lang}\n"
        "audience: developer\n"
        "type: explanation\n"
        "status: active\n"
        "owner: project\n"
        "last_reviewed: 2026-08-01\n"
        "---\n\n"
        '<a id="purpose"></a>\n'
        "## Purpose\n"
    )


def _run_cli(cwd: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    existing = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        f"{WORKTREE_ROOT}{os.pathsep}{existing}" if existing else str(WORKTREE_ROOT)
    )
    return subprocess.run(
        [sys.executable, "-m", "tools.documentation", *arguments],
        cwd=cwd,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )


class CliContractTests(unittest.TestCase):
    def test_success_failure_and_command_surface(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "sample.en.md").write_text(
                _document("en", "Sample"), encoding="utf-8"
            )
            (root / "sample.zh-TW.md").write_text(
                _document("zh-TW", "範例"), encoding="utf-8"
            )
            success = _run_cli(root, "validate", "--all")
            self.assertEqual(success.returncode, 0)
            self.assertEqual(
                success.stdout, "documentation validation passed (2 documents)\n"
            )
            self.assertEqual(success.stderr, "")

            (root / "Bad.en.md").write_text("bad", encoding="utf-8")
            failure = _run_cli(root, "validate", "--all")
            self.assertEqual(failure.returncode, 1)
            self.assertEqual(failure.stdout, "")
            self.assertEqual(
                failure.stderr,
                "Bad.en.md: invalid-name: filename does not follow canonical naming rules\n"
                "documentation validation failed (1 issue)\n",
            )

            invalid_shapes = [
                (),
                ("validate",),
                ("validate", "--staged"),
                ("validate", "--changed-from", "main"),
                ("inventory",),
                ("stage-site", "--output", "site"),
            ]
            for arguments in invalid_shapes:
                with self.subTest(arguments=arguments):
                    self.assertEqual(_run_cli(root, *arguments).returncode, 2)


class CliRootDiscoveryTests(unittest.TestCase):
    def test_nested_git_directory_and_worktree_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            outer = Path(directory)
            normal = outer / "normal"
            normal.mkdir()
            subprocess.run(
                ["git", "init", "-q"], cwd=normal, check=True, capture_output=True
            )
            (normal / "Bad.en.md").write_text("bad", encoding="utf-8")
            normal_nested = normal / "nested" / "deeper"
            normal_nested.mkdir(parents=True)
            normal_result = _run_cli(normal_nested, "validate", "--all")
            self.assertEqual(normal_result.returncode, 1)
            self.assertIn("Bad.en.md: invalid-name:", normal_result.stderr)

            main = outer / "main"
            main.mkdir()
            subprocess.run(
                ["git", "init", "-q"], cwd=main, check=True, capture_output=True
            )
            subprocess.run(
                ["git", "config", "user.email", "test@example.com"],
                cwd=main,
                check=True,
            )
            subprocess.run(
                ["git", "config", "user.name", "Test User"], cwd=main, check=True
            )
            (main / "seed.txt").write_text("seed", encoding="utf-8")
            subprocess.run(["git", "add", "seed.txt"], cwd=main, check=True)
            subprocess.run(
                ["git", "commit", "-q", "-m", "seed"], cwd=main, check=True
            )
            linked = outer / "linked"
            subprocess.run(
                ["git", "worktree", "add", "-q", "-b", "linked", str(linked)],
                cwd=main,
                check=True,
            )
            self.assertTrue((linked / ".git").is_file())
            (linked / "Bad.en.md").write_text("bad", encoding="utf-8")
            linked_nested = linked / "nested" / "deeper"
            linked_nested.mkdir(parents=True)
            linked_result = _run_cli(linked_nested, "validate", "--all")
            self.assertEqual(linked_result.returncode, 1)
            self.assertIn("Bad.en.md: invalid-name:", linked_result.stderr)


if __name__ == "__main__":
    unittest.main()
