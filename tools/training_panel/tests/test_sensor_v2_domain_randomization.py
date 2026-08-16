from __future__ import annotations

import ast
import importlib.util
import sys
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[3]
TASK_DIR = ROOT / "source" / "RedRhex" / "RedRhex" / "tasks" / "direct" / "redrhex"


def _load_module():
    path = TASK_DIR / "sensor_domain_randomization_v2.py"
    spec = importlib.util.spec_from_file_location("redrhex_sensor_dr_v2_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


DR = _load_module()


def _randomizer(config, *, num_envs: int = 4, seed: int = 17):
    return DR.SensorDomainRandomizerV2(
        config,
        num_envs=num_envs,
        num_main_joints=6,
        num_abad_joints=6,
        sample_hz=60.0,
        device="cpu",
        seed=seed,
    )


def _event_inputs(num_envs: int, value: float = 0.0) -> dict[str, torch.Tensor]:
    return {
        "gyro_body": torch.full((num_envs, 3), value),
        "main_position": torch.full((num_envs, 6), value),
        "abad_position": torch.full((num_envs, 6), value),
        "gravity_body": torch.tensor([[0.0, 0.0, -1.0]]).expand(num_envs, -1).clone(),
        "specific_force_body": torch.tensor([[0.0, 0.0, 9.81]])
        .expand(num_envs, -1)
        .clone(),
        "new_sample": torch.ones(num_envs, dtype=torch.bool),
    }


@pytest.mark.parametrize(
    "override",
    (
        {"gyro_noise_std_range_rad_s": (0.0, 0.01)},
        {"gyro_filter_time_constant_range_s": (0.01, 0.02)},
        {"gyro_latency_jitter_steps_range": (0, 1)},
        {"encoder_dropout_probability_range": (0.0, 0.1)},
        {"accel_bias_range_m_s2": (-0.1, 0.1)},
    ),
)
def test_non_neutral_range_requires_evidence(override: dict[str, object]) -> None:
    with pytest.raises(ValueError, match="requires non-empty evidence"):
        DR.SensorDomainRandomizationV2Config(**override)


def test_disabled_configuration_is_exact_identity() -> None:
    randomizer = _randomizer(DR.SensorDomainRandomizationV2Config())
    inputs = _event_inputs(4, value=0.25)

    event = randomizer.process(**inputs)

    assert torch.equal(event.gyro_body, inputs["gyro_body"])
    assert torch.equal(event.main_position, inputs["main_position"])
    assert torch.equal(event.abad_position, inputs["abad_position"])
    assert torch.equal(event.gravity_body, inputs["gravity_body"])
    assert torch.equal(event.specific_force_body, inputs["specific_force_body"])
    assert torch.equal(event.accepted_sample, inputs["new_sample"])
    assert not bool(event.encoder_stale.any())
    assert not bool(event.encoder_dropout.any())


def test_first_accepted_encoder_timestamp_only_primes_real_history() -> None:
    timestamps = (100, 101, 103, 104, 107)
    accepted_pattern = torch.tensor(
        [
            [False, True],
            [True, False],
            [False, True],
            [True, False],
            [True, True],
        ]
    )
    initialized = torch.zeros(2, dtype=torch.bool)
    history_timestamps: list[list[int]] = [[], []]

    for timestamp, accepted in zip(timestamps, accepted_pattern, strict=True):
        eligible = DR.real_history_sample_mask_v2(accepted, initialized)
        for env_index in range(2):
            if bool(eligible[env_index]):
                history_timestamps[env_index].append(timestamp)
        initialized |= accepted

    assert history_timestamps == [[104, 107], [103, 107]]


def test_seed_reproduces_parameters_events_and_partial_resets() -> None:
    config = DR.SensorDomainRandomizationV2Config(
        evidence="unit-test fixture",
        gyro_noise_std_range_rad_s=(0.01, 0.02),
        gyro_bias_range_rad_s=(-0.03, 0.04),
        gyro_drift_std_range_rad_s_sqrt_s=(0.001, 0.002),
        gyro_filter_time_constant_range_s=(0.01, 0.03),
        gyro_latency_steps_range=(1, 2),
        gyro_latency_jitter_steps_range=(-1, 1),
        imu_mount_roll_range_rad=(-0.02, 0.02),
        imu_mount_pitch_range_rad=(-0.03, 0.03),
        imu_mount_yaw_range_rad=(-0.04, 0.04),
        encoder_zero_offset_range_rad=(-0.02, 0.02),
        encoder_noise_std_range_rad=(0.001, 0.003),
        encoder_quantization_range_rad=(0.0005, 0.001),
        encoder_latency_steps_range=(0, 2),
        encoder_stale_probability_range=(0.1, 0.2),
        encoder_dropout_probability_range=(0.1, 0.2),
        accel_noise_std_range_m_s2=(0.01, 0.02),
        accel_bias_range_m_s2=(-0.03, 0.03),
    )
    first = _randomizer(config, seed=1234)
    second = _randomizer(config, seed=1234)

    for name, value in first.sampled_parameters().items():
        assert torch.equal(value, second.sampled_parameters()[name])
    inputs = _event_inputs(4, value=0.4)
    first_event = first.process(**inputs)
    second_event = second.process(**inputs)
    for name in (
        "gyro_body",
        "main_position",
        "abad_position",
        "gravity_body",
        "specific_force_body",
        "accepted_sample",
        "encoder_stale",
        "encoder_dropout",
    ):
        assert torch.equal(getattr(first_event, name), getattr(second_event, name))

    before = first.sampled_parameters()
    first.reset(torch.tensor([1, 3]))
    second.reset(torch.tensor([1, 3]))
    after = first.sampled_parameters()
    assert not torch.equal(after["gyro_bias_rad_s"][[1, 3]], before["gyro_bias_rad_s"][[1, 3]])
    for name, value in after.items():
        assert torch.equal(value, second.sampled_parameters()[name])
        assert torch.equal(value[[0, 2]], before[name][[0, 2]])


def test_reset_samples_stay_within_configured_bounds_and_stats_are_finite() -> None:
    config = DR.SensorDomainRandomizationV2Config(
        evidence="unit-test fixture",
        gyro_noise_std_range_rad_s=(0.01, 0.02),
        gyro_bias_range_rad_s=(-0.03, 0.04),
        gyro_latency_steps_range=(1, 3),
        imu_mount_roll_range_rad=(-0.2, -0.1),
        encoder_zero_offset_range_rad=(0.05, 0.08),
        encoder_stale_probability_range=(0.2, 0.3),
        accel_bias_range_m_s2=(-0.5, 0.6),
    )
    randomizer = _randomizer(config, num_envs=256)
    sampled = randomizer.sampled_parameters()

    assert sampled["gyro_noise_std_rad_s"].min() >= 0.01
    assert sampled["gyro_noise_std_rad_s"].max() <= 0.02
    assert sampled["gyro_bias_rad_s"].min() >= -0.03
    assert sampled["gyro_bias_rad_s"].max() <= 0.04
    assert sampled["gyro_latency_steps"].min() >= 1
    assert sampled["gyro_latency_steps"].max() <= 3
    assert sampled["imu_mount_rpy_rad"][:, 0].min() >= -0.2
    assert sampled["imu_mount_rpy_rad"][:, 0].max() <= -0.1
    assert sampled["encoder_zero_offset_rad"].min() >= 0.05
    assert sampled["encoder_zero_offset_rad"].max() <= 0.08
    assert sampled["encoder_stale_probability"].min() >= 0.2
    assert sampled["encoder_stale_probability"].max() <= 0.3
    assert sampled["accel_bias_m_s2"].min() >= -0.5
    assert sampled["accel_bias_m_s2"].max() <= 0.6
    assert all(torch.isfinite(torch.tensor(value)) for value in randomizer.sampled_statistics().values())


def test_latency_is_causal_and_stale_packets_are_held_but_rejected() -> None:
    latency = _randomizer(
        DR.SensorDomainRandomizationV2Config(
            evidence="unit-test fixture",
            gyro_latency_steps_range=(2, 2),
            encoder_latency_steps_range=(2, 2),
        ),
        num_envs=1,
    )
    gyro_values: list[float] = []
    encoder_values: list[float] = []
    for value in range(4):
        event = latency.process(**_event_inputs(1, float(value)))
        gyro_values.append(float(event.gyro_body[0, 0]))
        encoder_values.append(float(event.main_position[0, 0]))
    assert gyro_values == [0.0, 0.0, 0.0, 1.0]
    assert encoder_values == [0.0, 0.0, 0.0, 1.0]

    stale = _randomizer(
        DR.SensorDomainRandomizationV2Config(
            evidence="unit-test fixture",
            encoder_stale_probability_range=(1.0, 1.0),
        ),
        num_envs=1,
    )
    first = stale.process(**_event_inputs(1, 0.25))
    second = stale.process(**_event_inputs(1, 1.25))
    assert bool(first.accepted_sample.item())
    assert not bool(first.encoder_stale.item())
    assert not bool(second.accepted_sample.item())
    assert bool(second.encoder_stale.item())
    assert float(second.main_position[0, 0]) == pytest.approx(0.25)
    assert not bool(
        DR.real_history_sample_mask_v2(
            second.accepted_sample,
            torch.ones(1, dtype=torch.bool),
        ).item()
    )


def test_gyro_filter_is_causal_and_initialized_from_first_real_sample() -> None:
    randomizer = DR.SensorDomainRandomizerV2(
        DR.SensorDomainRandomizationV2Config(
            evidence="unit-test fixture",
            gyro_filter_time_constant_range_s=(1.0 / 60.0, 1.0 / 60.0),
        ),
        num_envs=1,
        num_main_joints=6,
        num_abad_joints=6,
        sample_hz=60.0,
        device="cpu",
        seed=5,
    )
    first = randomizer.process(**_event_inputs(1, 0.0))
    second = randomizer.process(**_event_inputs(1, 1.0))

    assert float(first.gyro_body[0, 0]) == pytest.approx(0.0)
    assert float(second.gyro_body[0, 0]) == pytest.approx(0.5)


def test_dropout_rejects_history_sample_and_outputs_remain_finite() -> None:
    randomizer = _randomizer(
        DR.SensorDomainRandomizationV2Config(
            evidence="unit-test fixture",
            gyro_noise_std_range_rad_s=(0.1, 0.1),
            gyro_bias_range_rad_s=(-0.1, 0.1),
            imu_mount_pitch_range_rad=(-0.1, 0.1),
            encoder_noise_std_range_rad=(0.01, 0.01),
            encoder_quantization_range_rad=(0.001, 0.001),
            encoder_dropout_probability_range=(1.0, 1.0),
            accel_noise_std_range_m_s2=(0.1, 0.1),
            accel_bias_range_m_s2=(-0.2, 0.2),
        )
    )

    event = randomizer.process(**_event_inputs(4, 0.5))

    assert not bool(event.accepted_sample.any())
    assert bool(event.encoder_dropout.all())
    for value in (
        event.gyro_body,
        event.main_position,
        event.abad_position,
        event.gravity_body,
        event.specific_force_body,
    ):
        assert value is not None and bool(torch.isfinite(value).all())


def test_environment_wires_randomization_before_differences_and_history() -> None:
    config_source = (TASK_DIR / "redrhex_sensor_v2_env_cfg.py").read_text(encoding="utf-8")
    env_source = (TASK_DIR / "redrhex_sensor_v2_env.py").read_text(encoding="utf-8")

    config_tree = ast.parse(config_source)
    config_class = next(
        node
        for node in config_tree.body
        if isinstance(node, ast.ClassDef) and node.name == "RedrhexForwardSensorV2EnvCfg"
    )
    defaults = {
        item.targets[0].id: ast.literal_eval(item.value)
        for item in config_class.body
        if isinstance(item, ast.Assign)
        and len(item.targets) == 1
        and isinstance(item.targets[0], ast.Name)
        and item.targets[0].id.startswith("sensor_dr_")
    }
    assert defaults["sensor_dr_evidence"] == ""
    assert defaults["sensor_dr_seed_offset"] == 0
    assert defaults["sensor_dr_require_physical_material_writes"] is False
    for name, value in defaults.items():
        if name in {
            "sensor_dr_evidence",
            "sensor_dr_seed_offset",
            "sensor_dr_require_physical_material_writes",
        }:
            continue
        assert value == (0.0, 0.0) or value == (0, 0)
    process_at = env_source.index("self._sensor_dr_v2.process(")
    difference_at = env_source.index("_wrapped_difference(main_pos")
    history_at = env_source.index("self._append_sensor_history_v2(sensor_frame")
    assert process_at < difference_at < history_at
    assert "return frame, history_sample" in env_source
    assert "real_history_sample_mask_v2" in env_source
    assert "self._sensor_dr_v2.reset(ids)" in env_source
    assert "sensor_dr_sampled_statistics_v2" in env_source
