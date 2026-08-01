import unittest

from tools.training_panel.training_panel.progress import parse_progress, progress_snapshot


FULL_BLOCK = (
    "################################################################################\n"
    "                       \x1b[1m Learning iteration 8/10 \x1b[0m                     \n"
    "\n"
    "                       Computation: 21453 steps/s (collection: 0.412s, learning 0.221s)\n"
    "             Mean action noise std: 1.02\n"
    "           Mean value_function loss: 0.0431\n"
    "                       Mean reward: 12.34\n"
    "               Mean episode length: 431.20\n"
    "--------------------------------------------------------------------------------\n"
    "                   Total timesteps: 491520\n"
    "                    Iteration time: 0.97s\n"
    "                      Time elapsed: 00:00:09\n"
    "                               ETA: 00:01:41\n"
)

NO_REWARD_BLOCK = (
    "################################################################################\n"
    "                       \x1b[1m Learning iteration 1/500 \x1b[0m                    \n"
    "\n"
    "                       Computation: 18000 steps/s (collection: 0.500s, learning 0.300s)\n"
    "             Mean action noise std: 1.00\n"
    "--------------------------------------------------------------------------------\n"
    "                   Total timesteps: 61440\n"
    "                    Iteration time: 0.80s\n"
    "                      Time elapsed: 00:00:01\n"
    "                               ETA: 00:06:39\n"
)


class ParseProgressTests(unittest.TestCase):
    def test_parses_full_block(self):
        result = parse_progress(FULL_BLOCK)
        self.assertIsNotNone(result)
        self.assertEqual(result["iteration"], 8)
        self.assertEqual(result["total_iterations"], 10)
        self.assertEqual(result["percent"], 80.0)
        self.assertEqual(result["steps_per_second"], 21453.0)
        self.assertEqual(result["mean_reward"], 12.34)
        self.assertEqual(result["mean_episode_length"], 431.2)
        self.assertEqual(result["total_timesteps"], 491520)
        self.assertEqual(result["iteration_seconds"], 0.97)
        self.assertEqual(result["eta_seconds"], 101)

    def test_uses_last_complete_block(self):
        text = FULL_BLOCK + FULL_BLOCK.replace("iteration 8/10", "iteration 9/10")
        result = parse_progress(text)
        self.assertEqual(result["iteration"], 9)

    def test_ignores_incomplete_trailing_block(self):
        partial = "\x1b[1m Learning iteration 9/10 \x1b[0m\n     Computation: 900 steps/s"
        result = parse_progress(FULL_BLOCK + partial)
        self.assertEqual(result["iteration"], 8)
        self.assertEqual(result["eta_seconds"], 101)

    def test_reward_lines_optional(self):
        result = parse_progress(NO_REWARD_BLOCK)
        self.assertEqual(result["iteration"], 1)
        self.assertEqual(result["total_iterations"], 500)
        self.assertNotIn("mean_reward", result)
        self.assertEqual(result["eta_seconds"], 399)

    def test_no_block_returns_none(self):
        self.assertIsNone(parse_progress("Isaac Sim starting up...\nloading assets\n"))

    def test_empty_text_returns_none(self):
        self.assertIsNone(parse_progress(""))

    def test_malformed_eta_is_dropped_not_fatal(self):
        text = FULL_BLOCK.replace("ETA: 00:01:41", "ETA: n/a")
        result = parse_progress(text)
        self.assertEqual(result["iteration"], 8)
        self.assertNotIn("eta_seconds", result)

    def test_zero_total_iterations_has_no_percent(self):
        text = FULL_BLOCK.replace("iteration 8/10", "iteration 0/0")
        result = parse_progress(text)
        self.assertEqual(result["iteration"], 0)
        self.assertNotIn("percent", result)


class ProgressSnapshotTests(unittest.TestCase):
    def test_snapshot_adds_updated_at(self):
        snapshot = progress_snapshot(FULL_BLOCK)
        self.assertEqual(snapshot["iteration"], 8)
        self.assertIn("updated_at", snapshot)
        self.assertRegex(snapshot["updated_at"], r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}$")

    def test_snapshot_returns_none_without_block(self):
        self.assertIsNone(progress_snapshot("nothing here"))


if __name__ == "__main__":
    unittest.main()
