import unittest

from tools.training_panel.training_panel.reward_overrides import (
    RewardOverrideError,
    apply_reward_overrides,
    normalize_reward_overrides,
)


class DummyEnvCfg:
    rew_scale_alive = 0.5

    def __init__(self):
        self.v2_reward_scales = {
            "velocity_tracking": 4.0,
            "energy_per_distance": 0.001,
        }


class RewardOverrideTests(unittest.TestCase):
    def test_apply_direct_and_nested_v2_reward_scales(self):
        cfg = DummyEnvCfg()

        applied = apply_reward_overrides(
            cfg,
            {
                "rew_scale_alive": "0.25",
                "v2_reward_scales": {"velocity_tracking": "5.5"},
                "v2_reward_scales.energy_per_distance": "0.002",
            },
        )

        self.assertEqual(cfg.rew_scale_alive, 0.25)
        self.assertEqual(cfg.v2_reward_scales["velocity_tracking"], 5.5)
        self.assertEqual(cfg.v2_reward_scales["energy_per_distance"], 0.002)
        self.assertEqual(
            applied,
            [
                "rew_scale_alive=0.25",
                "v2_reward_scales.velocity_tracking=5.5",
                "v2_reward_scales.energy_per_distance=0.002",
            ],
        )

    def test_normalize_rejects_non_numeric_nested_value(self):
        with self.assertRaises(RewardOverrideError):
            normalize_reward_overrides({"v2_reward_scales": {"velocity_tracking": "fast"}})

    def test_unknown_top_level_key_is_ignored_not_created(self):
        cfg = DummyEnvCfg()

        applied = apply_reward_overrides(cfg, {"not_a_reward": 1.0})

        self.assertEqual(applied, [])
        self.assertFalse(hasattr(cfg, "not_a_reward"))


if __name__ == "__main__":
    unittest.main()
