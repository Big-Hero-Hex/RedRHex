from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tools.sim2real.contracts import ContractError
from tools.sim2real.scenarios import load_scenario
from tools.sim2real.traces import load_trace, sha256_path, write_trace


def _valid_arrays() -> dict[str, np.ndarray]:
    return {
        "command_time_s": np.array([0.0, 0.5, 1.0]),
        "command": np.array([0.0, 0.25, 0.25]),
        "position_time_s": np.array([0.0, 0.25, 0.5, 0.75, 1.0]),
        "position": np.array([0.0, 0.01, 0.04, 0.09, 0.16]),
    }


def test_trace_round_trip_preserves_independent_time_bases_and_hashes(tmp_path: Path) -> None:
    raw = tmp_path / "raw.csv"
    raw.write_bytes(b"immutable source\n")
    scenario = load_scenario("main-step")

    manifest = write_trace(
        tmp_path / "episode",
        _valid_arrays(),
        scenario=scenario,
        source="real",
        source_path=raw,
    )
    loaded = load_trace(tmp_path / "episode", scenario=scenario)

    assert manifest.sample_counts == {"command": 3, "position": 5}
    assert manifest.time_bases == {
        "command": "command_time_s",
        "position": "position_time_s",
    }
    assert set(manifest.provenance) == {
        "trace_sha256",
        "scenario_sha256",
        "source_sha256",
    }
    np.testing.assert_array_equal(loaded.arrays["position"], _valid_arrays()["position"])
    assert loaded.manifest == manifest
    assert (tmp_path / "episode" / "metadata.json").is_file()
    assert not (tmp_path / "episode" / "manifest.json").exists()
    assert not list((tmp_path / "episode").glob("*.tmp"))


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda arrays: arrays.__setitem__("position", np.array([object()])), "numeric"),
        (lambda arrays: arrays["position"].__setitem__(2, np.nan), "finite"),
        (lambda arrays: arrays.pop("position"), "missing"),
        (lambda arrays: arrays.__setitem__("position", arrays["position"][:-1]), "shape"),
        (
            lambda arrays: arrays.__setitem__(
                "position_time_s", np.array([0.0, 0.25, 0.2, 0.75, 1.0])
            ),
            "strictly increasing",
        ),
    ],
)
def test_trace_writer_rejects_invalid_arrays(tmp_path: Path, mutate, message: str) -> None:
    arrays = _valid_arrays()
    mutate(arrays)

    with pytest.raises(ContractError, match=message):
        write_trace(
            tmp_path / "episode",
            arrays,
            scenario=load_scenario("main-step"),
            source="real",
        )


def test_trace_loader_rejects_hash_mismatch(tmp_path: Path) -> None:
    episode = tmp_path / "episode"
    write_trace(
        episode,
        _valid_arrays(),
        scenario=load_scenario("main-step"),
        source="sim",
    )
    manifest_path = episode / "metadata.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["provenance"]["trace_sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ContractError, match="hash mismatch"):
        load_trace(episode)


def test_loader_rejects_object_array_even_with_a_matching_manifest(tmp_path: Path) -> None:
    episode = tmp_path / "episode"
    manifest = write_trace(
        episode,
        _valid_arrays(),
        scenario=load_scenario("main-step"),
        source="sim",
    )
    np.savez(episode / "trace.npz", position=np.array([object()], dtype=object))
    payload = manifest.to_dict()
    from tools.sim2real.traces import sha256_file

    payload["provenance"]["trace_sha256"] = sha256_file(episode / "trace.npz")
    (episode / "metadata.json").write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ContractError, match="object|numeric"):
        load_trace(episode)


def test_rosbag_directory_hash_is_recursive_and_trace_artifacts_are_immutable(
    tmp_path: Path,
) -> None:
    bag = tmp_path / "bag"
    (bag / "nested").mkdir(parents=True)
    (bag / "metadata.yaml").write_text("version: 1\n", encoding="utf-8")
    (bag / "nested" / "data.db3").write_bytes(b"payload")
    first_hash = sha256_path(bag)
    episode = tmp_path / "episode"
    manifest = write_trace(
        episode,
        _valid_arrays(),
        scenario=load_scenario("main-step"),
        source="real",
        source_path=bag,
    )

    assert manifest.provenance["source_sha256"] == first_hash
    (bag / "nested" / "data.db3").write_bytes(b"changed")
    assert sha256_path(bag) != first_hash
    with pytest.raises(ContractError, match="already exists"):
        write_trace(
            episode,
            _valid_arrays(),
            scenario=load_scenario("main-step"),
            source="sim",
        )


def test_optional_channels_require_explicit_independent_time_bases(tmp_path: Path) -> None:
    arrays = _valid_arrays()
    arrays["imu_time_s"] = np.array([0.0, 0.01, 0.02, 0.03])
    arrays["imu_acceleration"] = np.ones((4, 3))

    manifest = write_trace(
        tmp_path / "episode",
        arrays,
        scenario=load_scenario("main-step"),
        source="sim",
        time_bases={"imu_acceleration": "imu_time_s"},
    )

    assert manifest.sample_counts["imu_acceleration"] == 4
    assert manifest.time_bases["imu_acceleration"] == "imu_time_s"

    bad = _valid_arrays()
    bad["imu_time_s"] = np.array([0.0, 0.01])
    bad["imu_acceleration"] = np.ones((3, 3))
    with pytest.raises(ContractError, match="shape mismatch"):
        write_trace(
            tmp_path / "bad-episode",
            bad,
            scenario=load_scenario("main-step"),
            source="sim",
            time_bases={"imu_acceleration": "imu_time_s"},
        )
