import tempfile
import unittest
from pathlib import Path

from tools.reward_agent.agent import preview_candidate_trials, queue_candidate_trials
from tools.reward_agent.experiment_store import ExperimentStore, RewardAgentPaths


class FakeRegistry:
    def __init__(self):
        self.params = []

    def queue_training(self, params):
        self.params.append(params)
        return {"id": f"panel_{len(self.params)}", "status": "queued"}


class AgentTests(unittest.TestCase):
    def test_queue_candidate_trials_uses_training_params_and_persists_trials(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = ExperimentStore(RewardAgentPaths.from_repo_root(Path(tmp)))
            session = store.create_session({"objective": "improve diagonal"})
            registry = FakeRegistry()

            trials = queue_candidate_trials(
                store,
                session["id"],
                registry,
                {"task": "Template-Redrhex-Direct-v0", "num_envs": 4, "max_iterations": 100, "device": "cpu"},
                [{"id": "cand_1", "reward_overrides": {"v2_reward_scales": {"velocity_tracking": 5.0}}}],
                max_iterations=5,
            )

            self.assertEqual(trials[0]["panel_run_id"], "panel_1")
            self.assertEqual(trials[0]["candidate_id"], "cand_1")
            self.assertEqual(registry.params[0].max_iterations, 5)
            self.assertEqual(registry.params[0].reward_preset_id, "cand_1")
            self.assertEqual(registry.params[0].reward_overrides["v2_reward_scales"]["velocity_tracking"], 5.0)
            self.assertEqual(store.load_trials(session["id"])[0]["panel_run_id"], "panel_1")

    def test_preview_candidate_trials_persists_dry_run_without_registry(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = ExperimentStore(RewardAgentPaths.from_repo_root(Path(tmp)))
            session = store.create_session({"objective": "preview trials"})

            trials = preview_candidate_trials(
                store,
                session["id"],
                {"task": "Template-Redrhex-Direct-v0", "num_envs": 4, "max_iterations": 100, "device": "cpu"},
                [
                    {"id": "cand_1", "reward_overrides": {"v2_reward_scales": {"velocity_tracking": 5.0}}},
                    {"id": "cand_2", "reward_overrides": {"v2_reward_scales": {"velocity_tracking": 3.5}}},
                ],
                max_iterations=7,
                limit=1,
            )

            self.assertEqual(len(trials), 1)
            self.assertEqual(trials[0]["status"], "dry_run")
            self.assertIsNone(trials[0]["panel_run_id"])
            self.assertEqual(trials[0]["params"]["max_iterations"], 7)
            self.assertEqual(store.load_trials(session["id"])[0]["candidate_id"], "cand_1")

    def test_queue_candidate_trials_respects_limit(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = ExperimentStore(RewardAgentPaths.from_repo_root(Path(tmp)))
            session = store.create_session({"objective": "limit trials"})
            registry = FakeRegistry()

            trials = queue_candidate_trials(
                store,
                session["id"],
                registry,
                {"task": "Template-Redrhex-Direct-v0", "num_envs": 4, "max_iterations": 100, "device": "cpu"},
                [
                    {"id": "cand_1", "reward_overrides": {"v2_reward_scales": {"velocity_tracking": 5.0}}},
                    {"id": "cand_2", "reward_overrides": {"v2_reward_scales": {"velocity_tracking": 3.5}}},
                ],
                max_iterations=5,
                limit=1,
            )

            self.assertEqual(len(trials), 1)
            self.assertEqual(len(registry.params), 1)
            self.assertEqual(trials[0]["candidate_id"], "cand_1")


if __name__ == "__main__":
    unittest.main()
