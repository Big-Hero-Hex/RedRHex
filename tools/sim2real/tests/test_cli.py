from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tools.sim2real.cli import build_parser, main
from tools.sim2real.contracts import CalibrationProfileV1, ContractError
from tools.sim2real.dataset import import_real_dataset
from tools.sim2real.import_real import import_real_trace, validate_latency_clock


def _profile_file(path: Path) -> Path:
    payload = CalibrationProfileV1.from_dict(
        {
            "schema_version": 1,
            "profile_id": "baseline",
            "hardware_mapping": {},
            "sensor_timing": {"aggregate_command_delay_s": 0.01},
            "simulation_physics": {"main_drive": {"damping": 0.2}},
        }
    ).to_dict()
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _npz(path: Path) -> Path:
    time_s = np.arange(0.0, 3.01, 0.05)
    command = np.where((time_s >= 0.5) & (time_s < 2.0), 0.25, 0.0)
    position = np.cumsum(np.where((time_s >= 0.7) & (time_s < 2.0), 2.0, 0.0)) * 0.05
    np.savez(
        path,
        command_time_s=time_s,
        command=command,
        position_time_s=time_s,
        position=position,
    )
    return path


def _stub_rosbag_extraction(monkeypatch) -> dict[str, object]:
    import tools.sim2real.import_real as importer

    constants: dict[str, object] = {
        "calibration_source": "provisional_repository_defaults",
        "encoder_counts_per_rev": 54984.83,
        "encoder_zero_count": 0.0,
        "encoder_sign": 1.0,
        "pwm_scale": 0.002,
        "pwm_cap": 1.0,
        "selected_leg": "r1",
        "positive_direction_bit": False,
    }
    arrays = {
        "command_time_s": np.array([0.0, 0.1]),
        "command": np.array([0.25, -0.25]),
        "position_time_s": np.array([0.01, 0.11]),
        "position": np.array([0.0, np.pi / 2.0]),
    }
    monkeypatch.setattr(
        importer,
        "_load_rosbag",
        lambda path, scenario, profile: (arrays, {}, constants),
    )
    return constants


@pytest.mark.parametrize("clock", ["", " ", "\t\n"])
def test_latency_clock_rejects_empty_or_whitespace_only_names(clock: str) -> None:
    with pytest.raises(ContractError, match="non-empty"):
        validate_latency_clock(clock)


def test_parser_exposes_the_five_pure_python_commands() -> None:
    parser = build_parser()
    choices = next(
        action.choices
        for action in parser._actions
        if getattr(action, "choices", None)
    )

    assert {"list", "import-real", "compare", "sweep", "validate-profile"}.issubset(
        choices
    )
    parsed = parser.parse_args(
        [
            "import-real",
            "bag",
            "--scenario",
            "main-step",
            "--output",
            ".",
            "--dataset-id",
            "dataset",
            "--episode-id",
            "episode",
            "--profile",
            "profile.json",
        ]
    )
    assert parsed.profile == Path("profile.json")


def test_list_validate_and_sweep_commands_emit_json(tmp_path: Path, capsys) -> None:
    assert main(["list"]) == 0
    listed = json.loads(capsys.readouterr().out)
    assert {item["scenario_id"] for item in listed["scenarios"]} >= {
        "main-step",
        "friction",
    }

    profile = _profile_file(tmp_path / "profile.json")
    assert main(["validate-profile", str(profile)]) == 0
    assert json.loads(capsys.readouterr().out)["valid"] is True

    output = tmp_path / "sweep"
    assert (
        main(
            [
                "sweep",
                str(profile),
                "--scenario",
                "main-step",
                "--mode",
                "one-factor",
                "--space-json",
                '{"simulation_physics.main_drive.damping":[0.2,0.3]}',
                "--output",
                str(output),
            ]
        )
        == 0
    )
    summary = json.loads(capsys.readouterr().out)
    assert summary["candidate_count"] == 1
    assert (output / "candidates" / "0001.json").is_file()
    assert (output / "index.json").is_file()


def test_one_factor_sweep_cli_enforces_max_candidates(tmp_path: Path, capsys) -> None:
    profile = _profile_file(tmp_path / "profile.json")
    output = tmp_path / "sweep"

    code = main(
        [
            "sweep",
            str(profile),
            "--scenario",
            "main-step",
            "--mode",
            "one-factor",
            "--space-json",
            '{"simulation_physics.main_drive.damping":[0.1,0.2,0.3]}',
            "--max-candidates",
            "1",
            "--output",
            str(output),
        ]
    )

    assert code == 2
    assert "max_candidates=1" in capsys.readouterr().err
    assert not output.exists()


def test_import_real_numeric_trace_and_compare_cli(tmp_path: Path, capsys) -> None:
    source = _npz(tmp_path / "raw.npz")
    units = '{"command":"normalized","position":"rad"}'
    frames = '{"command":"actuator","position":"main_0"}'
    real = tmp_path / "real"
    sim = tmp_path / "sim"
    args = [
        "import-real",
        str(source),
        "--scenario",
        "main-step",
        "--units-json",
        units,
        "--frames-json",
        frames,
        "--latency-clock",
        "bag_receive_time",
        "--dataset-id",
        "lab-run",
        "--episode-id",
        "real-episode",
    ]
    assert main([*args, "--output", str(tmp_path)]) == 0
    imported = json.loads(capsys.readouterr().out)
    dataset = tmp_path / "datasets" / "sim2real" / "lab-run"
    real = dataset / "episodes" / "real-episode"
    assert imported["dataset"] == str(dataset)
    assert (dataset / "manifest.json").is_file()
    assert (dataset / "raw" / source.name).is_file()
    assert (real / "trace.npz").is_file()
    assert (real / "metadata.json").is_file()
    # A second numeric import stands in for a simulator trace in this CPU CLI test.
    import_real_trace(
        source,
        sim,
        scenario="main-step",
        source_kind="sim",
        units=json.loads(units),
        frames=json.loads(frames),
        latency_clock="bag_receive_time",
    )

    assert main(["compare", str(real), str(sim), "--scenario", "main-step"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert set(result["subsystems"]) == {"main_drive"}


@pytest.mark.parametrize(
    "clock",
    [
        "synchronized_encoder_clock",
        "BAG_RECEIVE_TIME",
        " bag_receive_time ",
        "/motor_feedback.header",
        "motor_feedback.header",
        "/motor_feedback/header",
        "motor_feedback/header",
        "/motor_feedback.header.stamp",
        "motor_feedback.header.stamp",
        "/motor_feedback/header/stamp",
        "motor_feedback/header/stamp",
        "motor_feedback_header_stamp",
    ],
)
def test_rosbag_rejects_every_clock_other_than_bag_receive_time(
    tmp_path: Path, monkeypatch, clock: str
) -> None:
    bag = tmp_path / "bag"
    bag.mkdir()
    _stub_rosbag_extraction(monkeypatch)

    with pytest.raises(ContractError, match="bag.receive.time"):
        import_real_trace(
            bag,
            tmp_path / "episode",
            scenario="main-step",
            latency_clock=clock,
        )

    assert not (tmp_path / "episode").exists()


def test_numeric_npz_records_its_declared_clock(tmp_path: Path) -> None:
    source = _npz(tmp_path / "raw.npz")

    manifest = import_real_trace(
        source,
        tmp_path / "episode",
        scenario="main-step",
        latency_clock="synchronized_encoder_clock",
    )

    assert manifest.metadata["clock"]["source"] == "synchronized_encoder_clock"


def test_rosbag_rejects_caller_metadata_that_conflicts_with_used_constants(
    tmp_path: Path, monkeypatch
) -> None:
    bag = tmp_path / "bag"
    bag.mkdir()
    _stub_rosbag_extraction(monkeypatch)

    with pytest.raises(ContractError, match="calibration_constants.*pwm_scale"):
        import_real_trace(
            bag,
            tmp_path / "episode",
            scenario="main-step",
            metadata={"calibration_constants": {"pwm_scale": 99.0}},
        )

    assert not (tmp_path / "episode").exists()


def test_rosbag_records_the_clock_and_constants_used_during_extraction(
    tmp_path: Path, monkeypatch
) -> None:
    bag = tmp_path / "bag"
    bag.mkdir()
    expected_constants = _stub_rosbag_extraction(monkeypatch)

    manifest = import_real_trace(
        bag,
        tmp_path / "episode",
        scenario="main-step",
        latency_clock="bag_receive_time",
        metadata={"calibration_constants": {"operator_note": "bench-a"}},
    )

    assert manifest.metadata["clock"]["source"] == "bag_receive_time"
    assert manifest.metadata["calibration_constants"] == {
        "operator_note": "bench-a",
        **expected_constants,
    }


def test_rosbag_dependency_is_imported_only_when_a_bag_is_read(
    tmp_path: Path, monkeypatch
) -> None:
    bag = tmp_path / "bag"
    bag.mkdir()
    monkeypatch.setitem(sys.modules, "rosbag2_py", None)

    with pytest.raises(RuntimeError, match="rosbag2_py.*required"):
        import_real_trace(
            bag,
            tmp_path / "episode",
            scenario="main-step",
            latency_clock="bag_receive_time",
        )


def test_rosbag_reader_extracts_signed_main_leg_and_optional_sensor_clocks(
    tmp_path: Path, monkeypatch
) -> None:
    import tools.sim2real.import_real as importer

    def leg(*, voltage=0.0, direction=False, position=0.0):
        return SimpleNamespace(
            enable=True,
            voltage=voltage,
            direction=direction,
            position=position,
            tick_count=0,
            hall_effect=False,
        )

    def motor(**overrides):
        fields = {name: leg() for name in ("l1", "l2", "l3", "r1", "r2", "r3")}
        fields.update(overrides)
        return SimpleNamespace(**fields)

    imu = lambda value: SimpleNamespace(
        linear_acceleration=SimpleNamespace(x=value, y=0.0, z=9.8),
        angular_velocity=SimpleNamespace(x=0.0, y=value, z=0.0),
        orientation=SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0),
    )
    power = lambda value: SimpleNamespace(
        **{**{f"v_{i}": value for i in range(8)}, **{f"i_{i}": 0.1 for i in range(8)}}
    )
    records = [
        ("/motor/command", motor(r1=leg(voltage=125.0, direction=False)), 1_000_000_000),
        ("/motor/state", motor(r1=leg(position=0.0)), 1_010_000_000),
        ("/imu/data", imu(0.0), 1_015_000_000),
        ("/power/state", power(24.0), 1_020_000_000),
        ("/motor/command", motor(r1=leg(voltage=125.0, direction=True)), 1_100_000_000),
        ("/motor/state", motor(r1=leg(position=54984.83 / 4.0)), 1_110_000_000),
        ("/imu/data", imu(1.0), 1_115_000_000),
        ("/power/state", power(23.5), 1_120_000_000),
    ]

    class Reader:
        def __init__(self):
            self.records = iter(records)
            self.next_record = None

        def open(self, storage, converter):
            return None

        def get_all_topics_and_types(self):
            return [
                SimpleNamespace(name=topic, type="fake/Msg")
                for topic in {item[0] for item in records}
            ]

        def has_next(self):
            try:
                self.next_record = next(self.records)
            except StopIteration:
                return False
            return True

        def read_next(self):
            return self.next_record

    fake_rosbag = SimpleNamespace(
        SequentialReader=Reader,
        StorageOptions=lambda **kwargs: kwargs,
        ConverterOptions=lambda **kwargs: kwargs,
    )
    monkeypatch.setattr(
        importer,
        "_rosbag_dependencies",
        lambda: (fake_rosbag, lambda data, _type: data, lambda name: name),
    )
    bag = tmp_path / "bag"
    bag.mkdir()
    manifest = import_real_trace(
        bag,
        tmp_path / "episode",
        scenario="main-step",
        latency_clock="bag_receive_time",
    )
    from tools.sim2real.traces import load_trace

    trace = load_trace(tmp_path / "episode")

    np.testing.assert_allclose(trace.arrays["command_time_s"], [0.0, 0.1])
    np.testing.assert_allclose(trace.arrays["position_time_s"], [0.01, 0.11])
    np.testing.assert_allclose(trace.arrays["command"], [0.25, -0.25])
    np.testing.assert_allclose(trace.arrays["position"], [0.0, np.pi / 2.0])
    np.testing.assert_allclose(
        trace.arrays["motor_command_pwm_raw"][:, 3], [125.0, -125.0]
    )
    np.testing.assert_allclose(
        trace.arrays["motor_state_encoder_raw"][:, 3], [0.0, 54984.83 / 4.0]
    )
    assert trace.manifest.metadata["calibration_constants"][
        "calibration_source"
    ] == "provisional_repository_defaults"
    assert trace.manifest.metadata["clock"]["source"] == "bag_receive_time"
    assert manifest.time_bases["imu_acceleration"] == "imu_time_s"
    assert manifest.time_bases["power_voltage"] == "power_time_s"

    calibrated_profile = CalibrationProfileV1.from_dict(
        {
            "schema_version": 1,
            "profile_id": "bag-calibration",
            "hardware_mapping": {
                "encoder_counts_per_rev": {"main_0": 100.0},
                "encoder_zero_count": {"main_0": 10.0},
                "encoder_sign": {"main_0": -1},
                "pwm_scale": {"main_0": 0.01},
                "pwm_cap": {"main_0": 0.5},
            },
            "sensor_timing": {},
            "simulation_physics": {},
        }
    )
    import_real_trace(
        bag,
        tmp_path / "calibrated-episode",
        scenario="main-step",
        profile=calibrated_profile,
        latency_clock="bag_receive_time",
    )
    calibrated = load_trace(tmp_path / "calibrated-episode")
    np.testing.assert_allclose(calibrated.arrays["command"], [0.5, -0.5])
    assert calibrated.manifest.metadata["calibration_constants"][
        "calibration_source"
    ] == "profile:bag-calibration"
    assert calibrated.manifest.metadata["calibration_constants"]["pwm_scale"] == 0.01
    assert calibrated.manifest.metadata["calibration_constants"]["pwm_cap"] == 0.5
    assert "profile_sha256" in calibrated.manifest.provenance


def test_dataset_accepts_new_repetitions_but_refuses_raw_or_episode_overwrite(
    tmp_path: Path, capsys,
) -> None:
    first_source = _npz(tmp_path / "repeat-1.npz")
    second_source = _npz(tmp_path / "repeat-2.npz")
    kwargs = {
        "output_root": tmp_path,
        "dataset_id": "repeatability",
        "scenario": "main-step",
        "latency_clock": "bag_receive_time",
    }
    import_real_dataset(first_source, episode_id="repeat-1", **kwargs)
    result = import_real_dataset(second_source, episode_id="repeat-2", **kwargs)

    assert len(result.manifest["episodes"]) == 2
    assert len(result.manifest["raw"]) == 2
    third_source = _npz(tmp_path / "repeat-3.npz")
    with np.load(third_source) as archive:
        changed = {name: np.array(archive[name], copy=True) for name in archive.files}
    changed["position"] *= 1.5
    np.savez(third_source, **changed)
    assert main(
        [
            "import-real",
            str(third_source),
            "--scenario",
            "main-step",
            "--output",
            str(tmp_path),
            "--dataset-id",
            "repeatability",
            "--episode-id",
            "repeat-3",
        ]
    ) == 0
    reported = json.loads(capsys.readouterr().out)
    from tools.sim2real.traces import load_trace

    third = load_trace(
        tmp_path
        / "datasets"
        / "sim2real"
        / "repeatability"
        / "episodes"
        / "repeat-3"
    )
    assert reported["trace_sha256"] == third.manifest.provenance["trace_sha256"]
    with pytest.raises(Exception, match="episode.*already exists|raw.*already exists"):
        import_real_dataset(second_source, episode_id="repeat-2", **kwargs)
    with pytest.raises(Exception, match="identifier"):
        import_real_dataset(first_source, episode_id="../escape", **kwargs)
