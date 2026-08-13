import json
import tempfile
import unittest
from pathlib import Path

from tools.training_panel.training_panel.physics import (
    FIELD_SCHEMA,
    PhysicsPresetStore,
    normalize_physics_values,
    physics_catalog,
    values_to_profile,
    write_physics_profile,
)


class PhysicsProfileTests(unittest.TestCase):
    def test_catalog_covers_all_runtime_physics_groups_with_unique_keys(self):
        catalog = physics_catalog()
        keys = {field.key for field in FIELD_SCHEMA}
        categories = {field.category for field in FIELD_SCHEMA}

        self.assertEqual(catalog["field_count"], 113)
        self.assertEqual(len(keys), len(FIELD_SCHEMA))
        self.assertTrue({
            "Rigid body",
            "Mass and center of mass",
            "Contact material",
            "Main drive actuator",
            "ABAD actuator",
            "Damper actuator",
            "Passive springs",
            "Timing",
            "ABAD calibration",
        }.issubset(categories))
        self.assertIn("simulation_physics.joint_friction.main_0", keys)
        self.assertIn("simulation_physics.joint_viscous_friction.damper_5", keys)

    def test_normalize_rejects_unknown_nonfinite_and_out_of_range_values(self):
        for values in (
            {"not.a.quantity": 1.0},
            {"simulation_physics.mass.scale": float("inf")},
            {"simulation_physics.ground.restitution": 1.1},
            {"hardware_mapping.abad_target_scale.abad_0": 0.4},
        ):
            with self.subTest(values=values), self.assertRaises(ValueError):
                normalize_physics_values(values)

    def test_values_build_a_valid_nested_calibration_profile(self):
        profile = values_to_profile(
            "panel-test",
            normalize_physics_values(
                {
                    "simulation_physics.mass.scale": 1.08,
                    "simulation_physics.mass.com_offset_m.1": -0.012,
                    "simulation_physics.main_drive.effort_limit": 13.5,
                    "simulation_physics.joint_dynamic_friction.main_0": 0.2,
                    "simulation_physics.passive_spring.damper_5.rest_position_rad": 0.7,
                    "hardware_mapping.abad_target_offset_rad.abad_5": -0.01,
                    "sensor_timing.aggregate_command_delay_s": 0.004,
                }
            ),
        ).to_dict()

        self.assertEqual(profile["simulation_physics"]["mass"]["com_offset_m"], [0.0, -0.012, 0.0])
        self.assertEqual(profile["simulation_physics"]["main_drive"]["effort_limit"], 13.5)
        self.assertEqual(profile["simulation_physics"]["joint_dynamic_friction"]["main_0"], 0.2)
        self.assertEqual(profile["hardware_mapping"]["abad_target_offset_rad"]["abad_5"], -0.01)
        self.assertEqual(profile["sensor_timing"]["aggregate_command_delay_s"], 0.004)

    def test_write_profile_is_sparse_and_baseline_removes_candidate_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "candidate.json"
            result = write_physics_profile(
                path,
                "panel-test",
                "candidate",
                {"simulation_physics.ground.static_friction": 1.3},
            )
            self.assertEqual(result, path)
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(
                payload["simulation_physics"]["ground"],
                {"static_friction": 1.3, "dynamic_friction": 1.0},
            )

            self.assertIsNone(write_physics_profile(path, "panel-test", "baseline", {}))
            self.assertFalse(path.exists())

    def test_preset_store_persists_custom_presets_and_protects_baseline(self):
        with tempfile.TemporaryDirectory() as tmp:
            preset_file = Path(tmp) / "physics_presets.json"
            store = PhysicsPresetStore(preset_file)
            created = store.create_preset(
                "Measured August",
                "bench candidate",
                {"simulation_physics.mass.scale": 1.05},
            )
            store.set_active_preset(created["id"])
            updated = store.update_preset(
                created["id"],
                values={"simulation_physics.main_drive.velocity_limit": 12.0},
            )

            reloaded = PhysicsPresetStore(preset_file)
            self.assertEqual(reloaded.get_active_preset_id(), created["id"])
            self.assertEqual(updated["values"], {"simulation_physics.main_drive.velocity_limit": 12.0})
            self.assertEqual(reloaded.get_preset(created["id"])["values"], updated["values"])
            with self.assertRaises(ValueError):
                reloaded.update_preset("baseline", name="changed")
            with self.assertRaises(ValueError):
                reloaded.delete_preset("baseline")


if __name__ == "__main__":
    unittest.main()
