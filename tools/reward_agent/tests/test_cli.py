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

    def test_queue_trials_dry_run_uses_saved_candidates(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            store = ExperimentStore(RewardAgentPaths.from_repo_root(repo_root))
            session = store.create_session({"objective": "manual queue"})
            store.save_candidates(
                session["id"],
                [{"id": "cand_1", "reward_overrides": {"v2_reward_scales": {"velocity_tracking": 5.0}}}],
            )

            out = io.StringIO()
            with redirect_stdout(out):
                code = main(
                    [
                        "--repo-root",
                        str(repo_root),
                        "queue-trials",
                        "--session-id",
                        session["id"],
                        "--base-params-json",
                        '{"task":"Template-Redrhex-Direct-v0","num_envs":4,"max_iterations":100,"device":"cpu"}',
                        "--max-iterations",
                        "5",
                        "--dry-run",
                    ]
                )

            self.assertEqual(code, 0)
            self.assertIn("dry_run_trials: 1", out.getvalue())
            self.assertEqual(store.load_trials(session["id"])[0]["status"], "dry_run")

    def test_queue_trials_launch_uses_registry_factory(self):
        class FakeRegistry:
            def __init__(self):
                self.params = []

            def queue_training(self, params):
                self.params.append(params)
                return {"id": "panel_1", "status": "queued"}

        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            store = ExperimentStore(RewardAgentPaths.from_repo_root(repo_root))
            session = store.create_session({"objective": "manual launch"})
            store.save_candidates(
                session["id"],
                [{"id": "cand_1", "reward_overrides": {"v2_reward_scales": {"velocity_tracking": 5.0}}}],
            )
            registry = FakeRegistry()

            with redirect_stdout(io.StringIO()):
                code = main(
                    [
                        "--repo-root",
                        str(repo_root),
                        "queue-trials",
                        "--session-id",
                        session["id"],
                        "--base-params-json",
                        '{"task":"Template-Redrhex-Direct-v0","num_envs":4,"max_iterations":100,"device":"cpu"}',
                        "--launch",
                    ],
                    registry_factory=lambda _repo_root: registry,
                )

            self.assertEqual(code, 0)
            self.assertEqual(registry.params[0].reward_preset_id, "cand_1")
            self.assertEqual(store.load_trials(session["id"])[0]["panel_run_id"], "panel_1")


if __name__ == "__main__":
    unittest.main()
