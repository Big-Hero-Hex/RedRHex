import tempfile
import unittest
from pathlib import Path

from tools.reward_agent.launcher import build_panel_registry


class LauncherTests(unittest.TestCase):
    def test_build_panel_registry_uses_requested_repo_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry = build_panel_registry(Path(tmp))

            self.assertEqual(registry.paths.repo_root, Path(tmp).resolve())
            self.assertEqual(registry.history.paths.repo_root, Path(tmp).resolve())


if __name__ == "__main__":
    unittest.main()
