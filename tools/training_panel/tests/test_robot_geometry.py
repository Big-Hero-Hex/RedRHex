import math
import unittest
from pathlib import Path
from types import SimpleNamespace

from tools.training_panel.training_panel.config import PanelPaths
from tools.training_panel.training_panel.deploy import FALLBACK_CONTRACT
from tools.training_panel.training_panel.physics import FIELD_MAP, _LEG_LABELS
from tools.training_panel.training_panel.robot_geometry import (
    LAYOUT_VERSION,
    build_layout,
    robot_geometry,
)


def _paths() -> PanelPaths:
    return PanelPaths.from_env()


class RobotGeometryTests(unittest.TestCase):
    def setUp(self):
        self.layout = build_layout(_paths())

    def test_layout_is_parsed_from_the_urdf(self):
        self.assertEqual(self.layout["source"], "urdf")
        self.assertEqual(self.layout["version"], LAYOUT_VERSION)
        self.assertTrue(self.layout["model_path"].endswith("test_7.urdf"))
        self.assertEqual(len(self.layout["legs"]), 6)

    def test_every_leg_carries_its_canonical_joint_ids_and_urdf_names(self):
        for index, leg in enumerate(self.layout["legs"]):
            self.assertEqual(leg["index"], index)
            self.assertEqual(leg["contract_label"], _LEG_LABELS[index])
            joints = leg["joints"]
            self.assertEqual(set(joints), {"main", "abad", "damper"})
            for group in ("main", "abad", "damper"):
                self.assertEqual(joints[group]["canonical_id"], f"{group}_{index}")
                self.assertEqual(len(joints[group]["origin_m"]), 3)
                self.assertTrue(all(math.isfinite(v) for v in joints[group]["origin_m"]))

            # The URDF spells the contract's Revolute_15 as "Revolute 15".
            expected = {
                "main": FALLBACK_CONTRACT.MAIN_DRIVE_JOINT_NAMES[index],
                "abad": FALLBACK_CONTRACT.ABAD_JOINT_NAMES[index],
                "damper": FALLBACK_CONTRACT.DAMPER_JOINT_NAMES[index],
            }
            for group, canonical in expected.items():
                self.assertEqual(joints[group]["urdf_name"], canonical.replace("_", " ", 1))

    def test_leg_axes_are_unit_length_and_non_degenerate(self):
        for leg in self.layout["legs"]:
            for group in ("main", "abad", "damper"):
                axis = leg["joints"][group]["axis"]
                self.assertAlmostEqual(math.sqrt(sum(a * a for a in axis)), 1.0, places=6)

    def test_damper_rest_angle_defaults_to_the_tunable_schema_value(self):
        """The preview must show the inherited pose, not a hard-coded zero."""

        for index, leg in enumerate(self.layout["legs"]):
            key = f"simulation_physics.passive_spring.damper_{index}.rest_position_rad"
            self.assertAlmostEqual(leg["joints"]["damper"]["init_rad"], FIELD_MAP[key].default)

    def test_legs_are_split_three_per_side_with_the_middle_leg_splayed_outboard(self):
        centre = self.layout["body"]["lateral_center_m"]
        sides = {"left": [], "right": []}
        for leg in self.layout["legs"]:
            sides[leg["side"]].append(leg)
        self.assertEqual(len(sides["left"]), 3)
        self.assertEqual(len(sides["right"]), 3)

        for members in sides.values():
            rows = {leg["row"]: leg for leg in members}
            self.assertEqual(set(rows), {"front", "middle", "rear"})
            # Fore-aft ordering is what "front" and "rear" mean.
            self.assertLess(rows["front"]["mount"][0], rows["middle"]["mount"][0])
            self.assertLess(rows["middle"]["mount"][0], rows["rear"]["mount"][0])
            # RHex splays the middle leg so it clears its neighbours.
            offsets = {row: abs(leg["mount"][2] - centre) for row, leg in rows.items()}
            self.assertGreater(offsets["middle"], offsets["front"])
            self.assertGreater(offsets["middle"], offsets["rear"])

    def test_label_audit_reports_the_right_side_naming_mismatch(self):
        """Regression guard for a real defect: _LEG_LABELS disagrees with the URDF.

        Indices 0-2 are labelled right front/middle/rear but the URDF places them at the
        right middle/rear/front mounts. The indices themselves are correct, so training is
        unaffected; the viewer surfaces the mismatch rather than implying agreement.
        Fixing _LEG_LABELS should flip this assertion to `consistent`.
        """

        audit = self.layout["label_audit"]
        self.assertFalse(audit["consistent"])
        self.assertEqual(
            {(item["index"], item["contract_label"], item["geometric_label"]) for item in audit["mismatched"]},
            {
                (0, "Right front", "Right middle"),
                (1, "Right middle", "Right rear"),
                (2, "Right rear", "Right front"),
            },
        )
        # The left side has always agreed; only the right side is rotated.
        self.assertTrue(all(item["index"] < 3 for item in audit["mismatched"]))

    def test_missing_urdf_degrades_to_a_usable_fallback_layout(self):
        paths = SimpleNamespace(repo_root=Path("/nonexistent-redrhex-root"))
        layout = build_layout(paths)

        self.assertEqual(layout["source"], "fallback")
        self.assertTrue(layout["fallback_reason"])
        self.assertEqual(len(layout["legs"]), 6)
        for index, leg in enumerate(layout["legs"]):
            self.assertEqual(leg["index"], index)
            self.assertEqual(set(leg["joints"]), {"main", "abad", "damper"})
        self.assertEqual(len(layout["body"]["nominal_com_m"]), 3)

    def test_cached_layout_matches_a_fresh_build(self):
        self.assertEqual(robot_geometry(_paths()), self.layout)


if __name__ == "__main__":
    unittest.main()
