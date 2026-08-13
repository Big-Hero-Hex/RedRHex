import subprocess
import tempfile
import unittest
from pathlib import Path

from tools.training_panel.training_panel.commands import TrainingParams, training_argv
from tools.training_panel.training_panel.provenance import git_provenance


def _git(args: list[str], cwd: Path) -> None:
    subprocess.run(["git", *args], cwd=cwd, check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


class GitProvenanceTests(unittest.TestCase):
    def test_clean_repo_reports_commit_and_branch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _git(["init", "-b", "trunk"], root)
            _git(["config", "user.email", "test@example.com"], root)
            _git(["config", "user.name", "Test"], root)
            (root / "a.txt").write_text("hello")
            _git(["add", "a.txt"], root)
            _git(["commit", "-m", "first"], root)

            info = git_provenance(root)
            self.assertEqual(len(info["commit"]), 40)
            self.assertTrue(info["commit"].startswith(info["short"]))
            self.assertEqual(info["branch"], "trunk")
            self.assertFalse(info["dirty"])

    def test_dirty_repo_reports_dirty(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _git(["init", "-b", "trunk"], root)
            _git(["config", "user.email", "test@example.com"], root)
            _git(["config", "user.name", "Test"], root)
            (root / "a.txt").write_text("hello")
            _git(["add", "a.txt"], root)
            _git(["commit", "-m", "first"], root)
            (root / "a.txt").write_text("changed")

            self.assertTrue(git_provenance(root)["dirty"])

    def test_non_repo_returns_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(git_provenance(Path(tmp)), {})

    def test_missing_directory_returns_empty(self):
        self.assertEqual(git_provenance(Path("/nonexistent/redrhex/path")), {})


class SeedDefaultTests(unittest.TestCase):
    def test_blank_seed_gets_generated(self):
        params = TrainingParams.from_dict({"task": "x", "num_envs": 4, "max_iterations": 1})
        self.assertIsInstance(params.seed, int)
        self.assertGreaterEqual(params.seed, 0)
        self.assertLess(params.seed, 2 ** 31)

    def test_generated_seed_reaches_the_command_line(self):
        params = TrainingParams.from_dict({"task": "x", "num_envs": 4, "max_iterations": 1})
        argv = training_argv(params)
        self.assertIn("--seed", argv)
        self.assertEqual(argv[argv.index("--seed") + 1], str(params.seed))

    def test_explicit_seed_is_preserved(self):
        params = TrainingParams.from_dict({"task": "x", "num_envs": 4, "max_iterations": 1, "seed": 42})
        self.assertEqual(params.seed, 42)

    def test_two_blank_seed_runs_differ(self):
        seeds = {TrainingParams.from_dict({"task": "x", "num_envs": 4, "max_iterations": 1}).seed
                 for _ in range(20)}
        self.assertGreater(len(seeds), 1)


if __name__ == "__main__":
    unittest.main()
