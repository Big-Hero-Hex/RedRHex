import json
import math
import tempfile
import unittest
from pathlib import Path

from tools.training_panel.training_panel.convergence import (
    ConvergenceChecker,
    ConvergenceConfig,
    apply_settings,
    load_convergence_config,
)


def _checker_with(scalars):
    checker = ConvergenceChecker()
    checker.read_scalars = lambda log_dir, tag: scalars
    return checker


def _healthy(n=400, start=1.0, end=9.0):
    step = (end - start) / max(n - 1, 1)
    return [(i, start + i * step) for i in range(n)]


class DivergenceDetectionTests(unittest.TestCase):
    def test_nan_is_detected(self):
        scalars = _healthy(300) + [(300, float("nan"))]
        result = _checker_with(scalars).check_divergence(Path("/fake"), ConvergenceConfig())
        self.assertTrue(result.detected)
        self.assertEqual(result.kind, "nan")
        self.assertEqual(result.iteration, 300)

    def test_inf_is_detected(self):
        scalars = _healthy(300) + [(300, float("inf"))]
        result = _checker_with(scalars).check_divergence(Path("/fake"), ConvergenceConfig())
        self.assertTrue(result.detected)
        self.assertEqual(result.kind, "nan")

    def test_sustained_collapse_is_detected(self):
        # Peaks at 10.0, then sits at 1.0 (10% of peak) for 150 iterations.
        scalars = [(i, 10.0) for i in range(200)] + [(200 + i, 1.0) for i in range(150)]
        config = ConvergenceConfig(divergence_patience_iterations=100, divergence_collapse_pct=40.0)
        result = _checker_with(scalars).check_divergence(Path("/fake"), config)
        self.assertTrue(result.detected)
        self.assertEqual(result.kind, "collapse")

    def test_brief_dip_is_not_a_collapse(self):
        scalars = [(i, 10.0) for i in range(200)] + [(200 + i, 1.0) for i in range(20)]
        config = ConvergenceConfig(divergence_patience_iterations=100, divergence_collapse_pct=40.0)
        result = _checker_with(scalars).check_divergence(Path("/fake"), config)
        self.assertFalse(result.detected)

    def test_healthy_run_never_fires(self):
        result = _checker_with(_healthy(500)).check_divergence(Path("/fake"), ConvergenceConfig())
        self.assertFalse(result.detected)

    def test_short_run_never_fires(self):
        result = _checker_with(_healthy(20)).check_divergence(Path("/fake"), ConvergenceConfig())
        self.assertFalse(result.detected)

    def test_no_data_never_fires(self):
        result = _checker_with([]).check_divergence(Path("/fake"), ConvergenceConfig())
        self.assertFalse(result.detected)

    def test_negative_reward_curve_does_not_false_positive(self):
        # All-negative rewards improving toward zero: max is the latest value.
        scalars = [(i, -100.0 + i * 0.2) for i in range(400)]
        result = _checker_with(scalars).check_divergence(Path("/fake"), ConvergenceConfig())
        self.assertFalse(result.detected)

    def test_disabled_config_never_fires(self):
        scalars = _healthy(300) + [(300, float("nan"))]
        config = ConvergenceConfig(divergence_enabled=False)
        result = _checker_with(scalars).check_divergence(Path("/fake"), config)
        self.assertFalse(result.detected)

    def test_early_outlier_does_not_cause_collapse(self):
        # One fluke-high spike near the start, then ~500 iterations oscillating
        # healthily in a 5-8 band. An all-time peak of 50 would poison the
        # collapse ratio forever; a bounded reference window should not.
        scalars = [(0, 50.0)]
        for i in range(1, 501):
            scalars.append((i, 5.0 if i % 2 == 0 else 8.0))
        result = _checker_with(scalars).check_divergence(Path("/fake"), ConvergenceConfig())
        self.assertFalse(result.detected)

    def test_curriculum_step_down_does_not_cause_collapse_once_settled(self):
        # Climbs to ~9.0 over 900 iterations (harder terrain unlocked), then
        # legitimately settles at a new, lower plateau of ~3.0 for 400 more
        # iterations. Once the reference window is entirely inside the new
        # plateau, this must read as healthy, not as a collapse.
        climb = _healthy(900, start=1.0, end=9.0)
        plateau = [(900 + i, 3.0) for i in range(400)]
        scalars = climb + plateau
        result = _checker_with(scalars).check_divergence(Path("/fake"), ConvergenceConfig())
        self.assertFalse(result.detected)


class DivergenceConfigTests(unittest.TestCase):
    def test_defaults(self):
        config = ConvergenceConfig()
        self.assertTrue(config.divergence_enabled)
        self.assertEqual(config.divergence_action, "notify")
        self.assertEqual(config.divergence_patience_iterations, 100)
        self.assertEqual(config.divergence_collapse_pct, 40.0)

    def test_settings_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_file = Path(tmp) / "convergence_config.json"
            apply_settings({"divergence_action": "stop", "divergence_patience_iterations": 250},
                           config_file)
            reloaded = load_convergence_config(config_file)
            self.assertEqual(reloaded.divergence_action, "stop")
            self.assertEqual(reloaded.divergence_patience_iterations, 250)

    def test_invalid_action_falls_back_to_notify(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_file = Path(tmp) / "convergence_config.json"
            apply_settings({"divergence_action": "rm -rf"}, config_file)
            self.assertEqual(load_convergence_config(config_file).divergence_action, "notify")

    def test_existing_config_file_without_divergence_keys_loads(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_file = Path(tmp) / "convergence_config.json"
            config_file.write_text(json.dumps({"enabled": True, "preset": "strict"}))
            config = load_convergence_config(config_file)
            self.assertEqual(config.preset, "strict")
            self.assertEqual(config.divergence_action, "notify")


if __name__ == "__main__":
    unittest.main()
