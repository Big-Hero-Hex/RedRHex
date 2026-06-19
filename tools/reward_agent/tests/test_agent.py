import tempfile
import unittest
from pathlib import Path

from tools.reward_agent.agent import queue_candidate_trials
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


if __name__ == "__main__":
    unittest.main()
