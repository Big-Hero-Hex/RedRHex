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
from tools.sim2real.scenarios import load_scenario
from tools.sim2real.traces import load_trace, sha256_json


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


def _install_fake_rosbag_reader(monkeypatch, records) -> None:
    import tools.sim2real.import_real as importer

    records = sorted(records, key=lambda item: item[2])

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


def _probe_events(
    scenario_id: str = "suspended-main-0-step-coast",
    *,
    scenario_sha256: str | None = None,
    main_index: int = 0,
) -> list[tuple[str, SimpleNamespace, int]]:
    scenario = load_scenario(scenario_id)
    digest = scenario_sha256 or sha256_json(scenario.to_dict())
    common = {
        "schema_version": 1,
        "scenario_id": scenario.scenario_id,
        "scenario_schema_version": scenario.schema_version,
        "scenario_sha256": digest,
        "main_index": main_index,
        "abad_output_enable": False,
    }
    timestamp_ns = 1_000_000_000
    payloads: list[dict[str, object]] = [
        {
            **common,
            "event": "scenario",
            "rate_hz": 60.0,
            "repeats": scenario.repeats,
            "ticks": 990,
            "duration_s": 16.5,
        }
    ]
    tick_index = 0
    for repetition in range(1, scenario.repeats + 1):
        elapsed_s = tick_index / 60.0
        payloads.append(
            {
                **common,
                "event": "repetition",
                "repetition": repetition,
                "scheduled_elapsed_s": elapsed_s,
                "actual_elapsed_s": elapsed_s,
                "lateness_s": 0.0,
            }
        )
        for segment_index, segment in enumerate(scenario.command_segments):
            elapsed_s = tick_index / 60.0
            payloads.append(
                {
                    **common,
                    "event": "segment",
                    "repetition": repetition,
                    "segment_index": segment_index,
                    "segment": segment["label"],
                    "tick_index": tick_index,
                    "scheduled_elapsed_s": elapsed_s,
                    "actual_elapsed_s": elapsed_s,
                    "lateness_s": 0.0,
                }
            )
            tick_index += round(float(segment["duration_s"]) * 60.0)
    payloads.append(
        {
            **common,
            "event": "complete",
            "ticks": tick_index,
            "scheduled_elapsed_s": tick_index / 60.0,
            "actual_elapsed_s": tick_index / 60.0,
            "lateness_s": 0.0,
        }
    )
    return [
        (
            "/redrhex/sim2real_probe/events",
            SimpleNamespace(data=json.dumps(payload, allow_nan=False)),
            timestamp_ns
            + round(float(payload.get("actual_elapsed_s", 0.0)) * 1.0e9)
            + ordinal,
        )
        for ordinal, payload in enumerate(payloads)
    ]


def _fake_main_messages(*, enabled: bool = True, voltage: float = 125.0):
    def leg(name: str, *, position: float = 0.0):
        return SimpleNamespace(
            enable=enabled and name == "r1",
            voltage=voltage,
            direction=False,
            position=position,
        )

    command = SimpleNamespace(
        **{name: leg(name) for name in ("l1", "l2", "l3", "r1", "r2", "r3")}
    )
    state = SimpleNamespace(
        **{
            name: SimpleNamespace(position=0.0)
            for name in ("l1", "l2", "l3", "r1", "r2", "r3")
        }
    )
    return command, state


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
    assert parsed.latency_clock is None


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
                "--generate-only",
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
            "--generate-only",
        ]
    )

    assert code == 2
    assert "max_candidates=1" in capsys.readouterr().err
    assert not output.exists()


def test_import_real_numeric_trace_cli_and_compare_rejects_missing_mapping_provenance(
    tmp_path: Path, capsys
) -> None:
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

    assert main(["compare", str(real), str(sim), "--scenario", "main-step"]) == 2
    assert "mapping provenance" in capsys.readouterr().err


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


def test_numeric_npz_trace_api_requires_an_explicit_clock_before_writing(
    tmp_path: Path,
) -> None:
    source = _npz(tmp_path / "raw.npz")
    episode = tmp_path / "episode"

    with pytest.raises(ContractError, match="NPZ.*explicit.*clock"):
        import_real_trace(source, episode, scenario="main-step")

    assert not episode.exists()


def test_numeric_npz_dataset_api_requires_an_explicit_clock_before_creation(
    tmp_path: Path,
) -> None:
    source = _npz(tmp_path / "raw.npz")

    with pytest.raises(ContractError, match="NPZ.*explicit.*clock"):
        import_real_dataset(
            source,
            tmp_path,
            dataset_id="missing-clock",
            episode_id="episode",
            scenario="main-step",
        )

    assert not (tmp_path / "datasets").exists()


def test_numeric_npz_cli_requires_an_explicit_clock_before_creation(
    tmp_path: Path, capsys
) -> None:
    source = _npz(tmp_path / "raw.npz")

    code = main(
        [
            "import-real",
            str(source),
            "--scenario",
            "main-step",
            "--output",
            str(tmp_path),
            "--dataset-id",
            "missing-clock",
            "--episode-id",
            "episode",
        ]
    )

    assert code == 2
    assert "NPZ" in capsys.readouterr().err
    assert not (tmp_path / "datasets").exists()


def test_rosbag_api_defaults_to_bag_receive_time(
    tmp_path: Path, monkeypatch
) -> None:
    bag = tmp_path / "bag"
    bag.mkdir()
    _stub_rosbag_extraction(monkeypatch)

    manifest = import_real_trace(
        bag,
        tmp_path / "episode",
        scenario="main-step",
    )

    assert manifest.metadata["clock"]["source"] == "bag_receive_time"


def test_rosbag_cli_defaults_to_bag_receive_time(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    bag = tmp_path / "bag"
    bag.mkdir()
    _stub_rosbag_extraction(monkeypatch)

    assert main(
        [
            "import-real",
            str(bag),
            "--scenario",
            "main-step",
            "--output",
            str(tmp_path),
            "--dataset-id",
            "bag-default-clock",
            "--episode-id",
            "episode",
        ]
    ) == 0
    capsys.readouterr()
    from tools.sim2real.traces import load_trace

    trace = load_trace(
        tmp_path
        / "datasets"
        / "sim2real"
        / "bag-default-clock"
        / "episodes"
        / "episode"
    )
    assert trace.manifest.metadata["clock"]["source"] == "bag_receive_time"


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
    def leg(*, enable=False, voltage=0.0, direction=False, position=0.0):
        return SimpleNamespace(
            enable=enable,
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
        (
            "/motor/command",
            motor(
                r1=leg(enable=True, voltage=125.0, direction=False),
                header=SimpleNamespace(stamp=SimpleNamespace(sec=999, nanosec=0)),
            ),
            1_000_000_000,
        ),
        ("/motor/state", motor(r1=leg(position=0.0)), 1_010_000_000),
        ("/imu/data", imu(0.0), 1_015_000_000),
        ("/power/state", power(24.0), 1_020_000_000),
        (
            "/motor/command",
            motor(
                r1=leg(enable=True, voltage=125.0, direction=True),
                header=SimpleNamespace(stamp=SimpleNamespace(sec=999, nanosec=0)),
            ),
            1_100_000_000,
        ),
        ("/motor/state", motor(r1=leg(position=54984.83 / 4.0)), 1_110_000_000),
        ("/imu/data", imu(1.0), 1_115_000_000),
        ("/power/state", power(23.5), 1_120_000_000),
    ]
    _install_fake_rosbag_reader(monkeypatch, records)
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
    np.testing.assert_allclose(trace.arrays["imu_time_s"], [0.015, 0.115])
    np.testing.assert_allclose(trace.arrays["power_time_s"], [0.02, 0.12])
    np.testing.assert_allclose(trace.arrays["command"], [0.25, -0.25])
    np.testing.assert_allclose(trace.arrays["position"], [0.0, np.pi / 2.0])
    np.testing.assert_allclose(
        trace.arrays["motor_command_pwm_raw"][:, 3], [125.0, -125.0]
    )
    np.testing.assert_allclose(
        trace.arrays["motor_command_canonical"], [0.25, -0.25]
    )
    np.testing.assert_allclose(
        trace.arrays["motor_state_encoder_raw"][:, 3], [0.0, 54984.83 / 4.0]
    )
    assert trace.manifest.metadata["calibration_constants"][
        "calibration_source"
    ] == "provisional_repository_defaults"
    assert trace.manifest.metadata["calibration_constants"][
        "raw_enabled_leg_binding_verified"
    ] is True
    assert trace.manifest.metadata["clock"]["source"] == "bag_receive_time"
    assert manifest.time_bases["imu_acceleration"] == "imu_time_s"
    assert manifest.time_bases["power_voltage"] == "power_time_s"
    assert manifest.time_bases["motor_command_canonical"] == "motor_command_time_s"

    calibrated_profile = CalibrationProfileV1.from_dict(
        {
            "schema_version": 1,
            "profile_id": "bag-calibration",
            "hardware_mapping": {
                "joint_direction": {"main_0": -1},
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
    np.testing.assert_allclose(calibrated.arrays["command"], [-0.5, 0.5])
    np.testing.assert_allclose(
        calibrated.arrays["motor_command_canonical"], [-0.5, 0.5]
    )
    assert calibrated.manifest.metadata["calibration_constants"][
        "calibration_source"
    ] == "profile:bag-calibration"
    assert calibrated.manifest.metadata["calibration_constants"]["pwm_scale"] == 0.01
    assert calibrated.manifest.metadata["calibration_constants"]["pwm_cap"] == 0.5
    assert calibrated.manifest.metadata["calibration_constants"]["joint_direction"] == -1
    assert calibrated.manifest.metadata["calibration_constants"][
        "position_mapping_source"
    ] == "profile:bag-calibration"
    assert calibrated.manifest.metadata["calibration_constants"][
        "requested_command_source"
    ] == "profile:bag-calibration"
    assert "profile_sha256" in calibrated.manifest.provenance


@pytest.mark.parametrize(
    "enabled_legs",
    [(), ("l1",), ("r1", "l1")],
    ids=("selected-never-enabled", "wrong-leg", "multiple-legs"),
)
def test_rosbag_main_drive_import_binds_raw_enable_mask_to_scenario_joint(
    tmp_path: Path, monkeypatch, enabled_legs: tuple[str, ...]
) -> None:
    def leg(name: str, *, position: float = 0.0):
        return SimpleNamespace(
            enable=name in enabled_legs,
            voltage=30.0,
            direction=False,
            position=position,
        )

    command = SimpleNamespace(
        **{name: leg(name) for name in ("l1", "l2", "l3", "r1", "r2", "r3")}
    )
    state = SimpleNamespace(
        **{
            name: SimpleNamespace(position=0.0)
            for name in ("l1", "l2", "l3", "r1", "r2", "r3")
        }
    )
    records = [
        ("/motor/command", command, 1_000_000_000),
        ("/motor/state", state, 1_010_000_000),
    ]
    _install_fake_rosbag_reader(monkeypatch, records)
    bag = tmp_path / "bag"
    bag.mkdir()

    with pytest.raises(ContractError, match="raw main-drive command"):
        import_real_trace(
            bag,
            tmp_path / "episode",
            scenario="main-step",
        )

    assert not (tmp_path / "episode").exists()


def test_bound_probe_import_requires_completion_events(
    tmp_path: Path, monkeypatch
) -> None:
    command, state = _fake_main_messages()
    _install_fake_rosbag_reader(
        monkeypatch,
        [
            ("/motor/command", command, 2_000_000_000),
            ("/motor/state", state, 2_010_000_000),
        ],
    )
    bag = tmp_path / "bag"
    bag.mkdir()

    with pytest.raises(ContractError, match="probe events"):
        import_real_trace(
            bag,
            tmp_path / "episode",
            scenario="suspended-main-0-step-coast",
        )

    assert not (tmp_path / "episode").exists()


@pytest.mark.parametrize(
    ("events", "message"),
    [
        (_probe_events(scenario_sha256="0" * 64), "scenario hash"),
        (_probe_events(main_index=1), "main index"),
    ],
    ids=("wrong-scenario-hash", "wrong-main-index"),
)
def test_bound_probe_import_rejects_mismatched_event_provenance(
    tmp_path: Path,
    monkeypatch,
    events: list[tuple[str, SimpleNamespace, int]],
    message: str,
) -> None:
    command, state = _fake_main_messages()
    _install_fake_rosbag_reader(
        monkeypatch,
        [
            *events,
            ("/motor/command", command, 2_000_000_000),
            ("/motor/state", state, 2_010_000_000),
        ],
    )
    bag = tmp_path / "bag"
    bag.mkdir()

    with pytest.raises(ContractError, match=message):
        import_real_trace(
            bag,
            tmp_path / "episode",
            scenario="suspended-main-0-step-coast",
        )


def test_bound_probe_import_rejects_abort_or_missing_complete(
    tmp_path: Path, monkeypatch
) -> None:
    events = _probe_events()
    topic, message, timestamp_ns = events[-1]
    aborted = json.loads(message.data)
    aborted["event"] = "abort"
    aborted["reason"] = "test abort"
    events[-1] = (
        topic,
        SimpleNamespace(data=json.dumps(aborted, allow_nan=False)),
        timestamp_ns,
    )
    command, state = _fake_main_messages()
    _install_fake_rosbag_reader(
        monkeypatch,
        [
            *events,
            ("/motor/command", command, 2_000_000_000),
            ("/motor/state", state, 2_010_000_000),
        ],
    )
    bag = tmp_path / "bag"
    bag.mkdir()

    with pytest.raises(ContractError, match="abort|complete"):
        import_real_trace(
            bag,
            tmp_path / "episode",
            scenario="suspended-main-0-step-coast",
        )


@pytest.mark.parametrize(
    "receive_clock_scale",
    [1.0e-6, 2.0],
    ids=("compressed", "stretched"),
)
def test_bound_probe_import_rejects_forged_receive_clock_duration(
    tmp_path: Path,
    monkeypatch,
    receive_clock_scale: float,
) -> None:
    events = _probe_events()
    receive_origin_ns = events[0][2]
    forged_events = [
        (
            topic,
            message,
            receive_origin_ns
            + round((receive_time_ns - receive_origin_ns) * receive_clock_scale),
        )
        for topic, message, receive_time_ns in events
    ]
    command, state = _fake_main_messages()
    _install_fake_rosbag_reader(
        monkeypatch,
        [
            *forged_events,
            ("/motor/state", state, 1_010_000_000),
            ("/motor/command", command, 2_000_000_000),
        ],
    )
    bag = tmp_path / "bag"
    bag.mkdir()

    with pytest.raises(ContractError, match="receive time"):
        import_real_trace(
            bag,
            tmp_path / "episode",
            scenario="suspended-main-0-step-coast",
        )


def test_bound_probe_import_rejects_per_event_receive_clock_jitter(
    tmp_path: Path, monkeypatch
) -> None:
    events = _probe_events()
    topic, message, receive_time_ns = events[5]
    events[5] = (topic, message, receive_time_ns + 100_000_000)
    command, state = _fake_main_messages()
    _install_fake_rosbag_reader(
        monkeypatch,
        [
            *events,
            ("/motor/state", state, 1_010_000_000),
            ("/motor/command", command, 2_000_000_000),
        ],
    )
    bag = tmp_path / "bag"
    bag.mkdir()

    with pytest.raises(ContractError, match="receive time"):
        import_real_trace(
            bag,
            tmp_path / "episode",
            scenario="suspended-main-0-step-coast",
        )


def test_bound_probe_import_uses_event_start_as_derived_neutral_baseline(
    tmp_path: Path, monkeypatch
) -> None:
    enabled, state = _fake_main_messages(enabled=True)
    disabled, _ = _fake_main_messages(enabled=False)
    _install_fake_rosbag_reader(
        monkeypatch,
        [
            *_probe_events(),
            ("/motor/state", state, 1_010_000_000),
            ("/motor/command", enabled, 2_000_000_000),
            ("/motor/state", state, 2_010_000_000),
            ("/motor/command", disabled, 2_100_000_000),
            ("/motor/state", state, 2_110_000_000),
        ],
    )
    bag = tmp_path / "bag"
    bag.mkdir()

    manifest = import_real_trace(
        bag,
        tmp_path / "episode",
        scenario="suspended-main-0-step-coast",
    )
    trace = load_trace(tmp_path / "episode")

    scenario = load_scenario("suspended-main-0-step-coast")
    expected_commands = np.tile(
        [float(segment["value"]) for segment in scenario.command_segments],
        scenario.repeats,
    )
    np.testing.assert_allclose(trace.arrays["command"], expected_commands)
    assert trace.arrays["command_time_s"][0] >= 0.0
    transitions = trace.arrays["command"][
        np.r_[True, np.diff(trace.arrays["command"]) != 0.0]
    ]
    np.testing.assert_allclose(
        transitions,
        [0.0, 0.25, 0.0, -0.25, 0.0, 0.25, 0.0, -0.25, 0.0, 0.25, 0.0, -0.25, 0.0],
    )
    np.testing.assert_allclose(
        trace.arrays["motor_command_pwm_raw"][:, 3], [125.0, 0.0]
    )
    np.testing.assert_allclose(trace.arrays["motor_command_canonical"], [0.25, 0.0])
    np.testing.assert_allclose(
        trace.arrays["motor_command_time_s"], [1.0, 1.1], atol=1.0e-9
    )
    assert manifest.time_bases["motor_command_pwm_raw"] == "motor_command_time_s"
    evidence = manifest.metadata["calibration_constants"]["probe_event_evidence"]
    assert evidence["scenario_sha256"] == sha256_json(
        scenario.to_dict()
    )
    assert evidence["repetition_count"] == 3
    assert evidence["segment_count"] == 21
    assert evidence["complete_ticks"] == 990


def test_bound_probe_import_authenticates_commands_separately_from_position_mapping(
    tmp_path: Path, monkeypatch
) -> None:
    command, state = _fake_main_messages()
    _install_fake_rosbag_reader(
        monkeypatch,
        [
            *_probe_events(),
            ("/motor/state", state, 1_010_000_000),
            ("/motor/command", command, 2_000_000_000),
        ],
    )
    profile = CalibrationProfileV1.from_dict(
        {
            "schema_version": 1,
            "profile_id": "encoder-audited-all-main",
            "hardware_mapping": {
                "encoder_counts_per_rev": {
                    f"main_{index}": 54984.83 for index in range(6)
                },
                "encoder_zero_count": {
                    f"main_{index}": 0.0 for index in range(6)
                },
                "encoder_sign": {
                    f"main_{index}": 1.0 for index in range(6)
                },
            },
            "sensor_timing": {},
            "simulation_physics": {},
        }
    )
    bag = tmp_path / "bag"
    bag.mkdir()

    import_real_trace(
        bag,
        tmp_path / "episode",
        scenario="suspended-main-0-step-coast",
        profile=profile,
    )
    trace = load_trace(tmp_path / "episode")
    np.testing.assert_allclose(
        trace.arrays["main_joint_position_canonical"], np.zeros((1, 6))
    )
    assert trace.manifest.metadata["frames"]["main_joint_position_canonical"] == (
        "canonical_main_joint_order"
    )
    constants = trace.manifest.metadata["calibration_constants"]
    scenario = load_scenario("suspended-main-0-step-coast")

    assert constants["position_mapping_source"] == "profile:encoder-audited-all-main"
    assert constants["requested_command_source"] == (
        f"authenticated_probe_events:{sha256_json(scenario.to_dict())}"
    )
    assert constants["calibration_source"].endswith(":with_provisional_fallbacks")
    initial_state = constants["replay_initial_state"]
    assert initial_state == {
        "schema_version": 1,
        "joint_order": [f"main_{index}" for index in range(6)],
        "position_source_channel": "main_joint_position_canonical",
        "position_rad": [0.0] * 6,
        "velocity_rad_s": [0.0] * 6,
        "velocity_source": "reviewed_initial_neutral",
        "fixture_mode": "fixed_base",
        "fixture_frame": "world",
        "root_pose_source": "production_asset_default",
        "sample_time_s": pytest.approx(0.01),
        "scenario_time_s": pytest.approx(0.0),
        "sample_offset_s": pytest.approx(0.01),
    }
    assert constants["replay_initial_state_sha256"] == sha256_json(initial_state)


def test_imported_normal_probe_exposes_all_three_bidirectional_repetitions(
    tmp_path: Path, monkeypatch
) -> None:
    from tools.sim2real.metrics import compute_subsystem_metrics

    names = ("l1", "l2", "l3", "r1", "r2", "r3")

    def raw_command(*, enabled: bool, direction: bool):
        legs = {
            name: SimpleNamespace(
                enable=enabled and name == "r1",
                voltage=125.0,
                direction=direction,
            )
            for name in names
        }
        return SimpleNamespace(**legs)

    def raw_state(position_count: float):
        return SimpleNamespace(
            **{
                name: SimpleNamespace(position=position_count if name == "r1" else 0.0)
                for name in names
            }
        )

    base_ns = 1_000_000_000
    records = list(_probe_events())
    position_rad = 0.0
    dt = 1.0 / 60.0
    for tick in range(990):
        elapsed = (tick + 1) * dt
        cycle_time = (elapsed - np.finfo(float).eps) % 5.5
        velocity = 0.0
        if 0.6 <= cycle_time < 1.5:
            velocity = 2.0
        elif 1.5 <= cycle_time < 1.9:
            velocity = 2.0 * (1.9 - cycle_time) / 0.4
        elif 3.1 <= cycle_time < 4.0:
            velocity = -1.5
        elif 4.0 <= cycle_time < 4.4:
            velocity = -1.5 * (4.4 - cycle_time) / 0.4
        position_rad += velocity * dt
        count = position_rad * 54984.83 / (2.0 * np.pi)
        records.append(
            ("/motor/state", raw_state(count), base_ns + round(elapsed * 1.0e9) - 100)
        )
    for repetition in range(3):
        cycle_start = repetition * 5.5
        for elapsed, enabled, direction in (
            (cycle_start + 0.5, True, False),
            (cycle_start + 1.5, False, False),
            (cycle_start + 3.0, True, True),
            (cycle_start + 4.0, False, False),
        ):
            records.append(
                (
                    "/motor/command",
                    raw_command(enabled=enabled, direction=direction),
                    base_ns + round(elapsed * 1.0e9) + 100,
                )
            )
    _install_fake_rosbag_reader(monkeypatch, records)
    bag = tmp_path / "bag"
    bag.mkdir()
    scenario = load_scenario("suspended-main-0-step-coast")

    import_real_trace(bag, tmp_path / "episode", scenario=scenario)
    trace = load_trace(tmp_path / "episode", scenario=scenario)
    metrics = compute_subsystem_metrics(scenario, trace)

    for family in ("step", "coast"):
        assert metrics[family]["positive"]["repeat_count"] == 3.0
        assert metrics[family]["negative"]["repeat_count"] == 3.0


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
            "--latency-clock",
            "bag_receive_time",
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
