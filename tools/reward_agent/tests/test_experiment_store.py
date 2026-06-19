import tempfile
import unittest
from pathlib import Path

from tools.reward_agent.experiment_store import ExperimentStore, RewardAgentPaths


class ExperimentStoreTests(unittest.TestCase):
    def test_create_session_persists_goal_and_index(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = RewardAgentPaths.from_repo_root(Path(tmp))
            store = ExperimentStore(paths)

            session = store.create_session({"objective": "improve diagonal", "baseline_run_id": "run_1"})

            self.assertEqual(session["goal"]["objective"], "improve diagonal")
            self.assertTrue((paths.session_dir(session["id"]) / "goal.json").is_file())
            self.assertEqual(store.list_sessions()[0]["id"], session["id"])

    def test_trials_and_conversation_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = RewardAgentPaths.from_repo_root(Path(tmp))
            store = ExperimentStore(paths)
            session = store.create_session({"objective": "reduce energy"})

            store.save_trials(session["id"], [{"id": "trial_1", "panel_run_id": "panel_1"}])
            store.append_conversation(session["id"], {"role": "user", "content": "focus on stability"})

            self.assertEqual(store.load_trials(session["id"])[0]["panel_run_id"], "panel_1")
            log_text = (paths.session_dir(session["id"]) / "conversation.jsonl").read_text(encoding="utf-8")
            self.assertIn("focus on stability", log_text)

    def test_candidates_evaluations_and_report_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = RewardAgentPaths.from_repo_root(Path(tmp))
            store = ExperimentStore(paths)
            session = store.create_session({"objective": "improve yaw"})

            store.save_candidates(session["id"], [{"id": "cand_1"}])
            store.save_evaluations(session["id"], [{"candidate_id": "cand_1", "overall_score": 1.2}])
            report_path = store.save_report(session["id"], "comparison", {"best_candidate_id": "cand_1"})

            self.assertEqual(store.load_candidates(session["id"])[0]["id"], "cand_1")
            self.assertEqual(store.load_evaluations(session["id"])[0]["overall_score"], 1.2)
            self.assertTrue(report_path.is_file())
            self.assertIn("best_candidate_id", report_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
