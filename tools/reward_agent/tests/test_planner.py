import unittest

from tools.reward_agent.planner import RewardWeightSpec, generate_weight_candidates


class PlannerTests(unittest.TestCase):
    def test_generate_candidates_changes_one_weight_at_a_time_with_bounds(self):
        candidates = generate_weight_candidates(
            {"v2_reward_scales": {"velocity_tracking": 4.0, "energy_per_distance": 0.001}},
            [
                RewardWeightSpec(
                    "velocity_tracking",
                    minimum=3.5,
                    maximum=4.5,
                    multipliers=(0.5, 1.5),
                )
            ],
        )

        self.assertEqual(
            [candidate["id"] for candidate in candidates],
            ["cand-001-velocity_tracking-x0_5", "cand-002-velocity_tracking-x1_5"],
        )
        self.assertEqual(candidates[0]["reward_overrides"]["v2_reward_scales"]["velocity_tracking"], 3.5)
        self.assertEqual(candidates[1]["reward_overrides"]["v2_reward_scales"]["velocity_tracking"], 4.5)
        self.assertEqual(candidates[0]["reward_overrides"]["v2_reward_scales"]["energy_per_distance"], 0.001)
        self.assertEqual(candidates[0]["parent_candidate_id"], "baseline")
        self.assertIn("velocity_tracking", candidates[0]["hypothesis"])

    def test_generate_candidates_supports_top_level_reward_scales(self):
        candidates = generate_weight_candidates(
            {"rew_scale_alive": 0.5},
            [RewardWeightSpec("rew_scale_alive", minimum=0.1, maximum=1.0, multipliers=(1.2,), group="top_level")],
        )

        self.assertEqual(candidates[0]["reward_overrides"]["rew_scale_alive"], 0.6)
        self.assertEqual(candidates[0]["changed"]["rew_scale_alive"]["from"], 0.5)


if __name__ == "__main__":
    unittest.main()
