import unittest

from tools.training_panel.training_panel.convergence import (
    SCALAR_TAG_ALLOWLIST,
    downsample,
    requested_tags,
)


class DownsampleTests(unittest.TestCase):
    def test_short_series_is_unchanged(self):
        points = [(i, float(i)) for i in range(10)]
        self.assertEqual(downsample(points, 200), points)

    def test_long_series_is_reduced_to_limit(self):
        points = [(i, float(i)) for i in range(1000)]
        reduced = downsample(points, 100)
        self.assertLessEqual(len(reduced), 100)
        self.assertGreater(len(reduced), 1)

    def test_downsample_keeps_first_and_last(self):
        points = [(i, float(i)) for i in range(1000)]
        reduced = downsample(points, 100)
        self.assertEqual(reduced[0], points[0])
        self.assertEqual(reduced[-1], points[-1])

    def test_empty_series(self):
        self.assertEqual(downsample([], 100), [])

    def test_limit_of_one_returns_last_point(self):
        points = [(i, float(i)) for i in range(50)]
        self.assertEqual(downsample(points, 1), [points[-1]])


class RequestedTagsTests(unittest.TestCase):
    def test_defaults_when_empty(self):
        tags = requested_tags("")
        self.assertEqual(tags, ["Train/mean_reward", "Train/mean_episode_length"])

    def test_allowlisted_tags_pass(self):
        self.assertEqual(requested_tags("Train/mean_reward"), ["Train/mean_reward"])

    def test_unknown_tag_is_rejected(self):
        self.assertEqual(requested_tags("Train/mean_reward,../../etc/passwd"), ["Train/mean_reward"])

    def test_all_rejected_falls_back_to_defaults(self):
        self.assertEqual(requested_tags("bogus,alsobogus"),
                         ["Train/mean_reward", "Train/mean_episode_length"])

    def test_duplicates_collapse(self):
        self.assertEqual(requested_tags("Train/mean_reward,Train/mean_reward"), ["Train/mean_reward"])

    def test_allowlist_covers_the_convergence_primary_tag(self):
        self.assertIn("Train/mean_reward", SCALAR_TAG_ALLOWLIST)


if __name__ == "__main__":
    unittest.main()
