import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from tools.reward_agent.__main__ import main
from tools.reward_agent.experiment_store import ExperimentStore, RewardAgentPaths


class CliTests(unittest.TestCase):
    def test_create_session_and_propose_candidates(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            out = io.StringIO()
            with redirect_stdout(out):
                self.assertEqual(main(["--repo-root", str(repo_root), "create-session", "--objective", "improve diagonal"]), 0)
            session_id = out.getvalue().strip().split()[-1]

            with redirect_stdout(io.StringIO()):
                code = main(
                    [
                        "--repo-root",
                        str(repo_root),
                        "propose-candidates",
                        "--session-id",
                        session_id,
                        "--base-overrides-json",
                        '{"v2_reward_scales":{"velocity_tracking":4.0}}',
                        "--scale",
                        "velocity_tracking:2.0:8.0",
                    ]
                )

            self.assertEqual(code, 0)
            store = ExperimentStore(RewardAgentPaths.from_repo_root(repo_root))
            self.assertEqual(len(store.load_candidates(session_id)), 2)


if __name__ == "__main__":
    unittest.main()
