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

    def test_normalize_rejects_boolean_and_non_finite_values(self):
        for value in (True, False, float("nan"), float("inf"), float("-inf")):
            with self.subTest(value=value), self.assertRaises(RewardOverrideError):
                normalize_reward_overrides({"rew_scale_alive": value})

    def test_unknown_top_level_key_is_ignored_not_created(self):
        cfg = DummyEnvCfg()

        applied = apply_reward_overrides(cfg, {"not_a_reward": 1.0})

        self.assertEqual(applied, [])
        self.assertFalse(hasattr(cfg, "not_a_reward"))

    def test_require_all_rejects_unknown_nested_key(self):
        with self.assertRaisesRegex(RewardOverrideError, "unknown_term"):
            apply_reward_overrides(
                DummyEnvCfg(),
                {"v2_reward_scales": {"unknown_term": 1.0}},
                require_all=True,
            )

    def test_required_override_must_match_the_resolved_runtime_value(self):
        class CoercingConfig:
            def __init__(self):
                self._weight = 1.0

            @property
            def weight(self):
                return self._weight

            @weight.setter
            def weight(self, value):
                self._weight = round(float(value))

        with self.assertRaisesRegex(RewardOverrideError, "did not resolve exactly"):
            apply_reward_overrides(CoercingConfig(), {"weight": 1.25}, require_all=True)


if __name__ == "__main__":
    unittest.main()
