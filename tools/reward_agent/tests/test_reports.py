import unittest

from tools.reward_agent.reports import build_comparison_report, rank_evaluations


class ReportTests(unittest.TestCase):
    def test_rank_evaluations_prefers_complete_high_scores(self):
        ranked = rank_evaluations(
            [
                {"candidate_id": "cand_low", "overall_score": 1.0, "complete": True},
                {"candidate_id": "cand_incomplete", "overall_score": 9.0, "complete": False},
                {"candidate_id": "cand_high", "overall_score": 2.0, "complete": True},
            ]
        )

        self.assertEqual([item["candidate_id"] for item in ranked], ["cand_high", "cand_low", "cand_incomplete"])

    def test_build_comparison_report_links_best_trial(self):
        report = build_comparison_report(
            {"id": "session_1", "goal": {"objective": "improve yaw"}},
            [{"candidate_id": "cand_high", "panel_run_id": "panel_1"}],
            [{"candidate_id": "cand_high", "overall_score": 2.0, "complete": True}],
        )

        self.assertEqual(report["best_candidate_id"], "cand_high")
        self.assertEqual(report["best_panel_run_id"], "panel_1")
        self.assertIn("improve yaw", report["summary"])


if __name__ == "__main__":
    unittest.main()
