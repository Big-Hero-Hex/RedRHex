from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET


REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
ROS_SOURCE = REPOSITORY_ROOT / "ros2_ws/src"
ADAPTER_ROOT = ROS_SOURCE / "redrhex_policy_io"
CONTROLLER_ROOT = ROS_SOURCE / "redrhex_rl_controller"


def _install_ament_python_package(
    package_root: Path,
    destination: Path,
    scratch: Path,
) -> None:
    egg_base = scratch / "egg"
    egg_base.mkdir(parents=True)
    command = [
        sys.executable,
        "setup.py",
        "egg_info",
        "--egg-base",
        str(egg_base),
        "build",
        "--build-base",
        str(scratch / "build"),
        "install",
        "--prefix",
        str(destination),
        "--single-version-externally-managed",
        "--record",
        str(scratch / "record.txt"),
    ]
    subprocess.run(
        command,
        cwd=package_root,
        check=True,
        text=True,
        capture_output=True,
    )


def test_controller_declares_the_colcon_policy_io_runtime_dependency() -> None:
    controller_xml = ET.parse(CONTROLLER_ROOT / "package.xml").getroot()
    dependencies = {
        element.text
        for element in controller_xml.findall("exec_depend")
    }
    assert "redrhex_policy_io" in dependencies
    setup_source = (CONTROLLER_ROOT / "setup.py").read_text(encoding="utf-8")
    assert '"redrhex-policy-io==2.0.0"' in setup_source

    adapter_xml = ET.parse(ADAPTER_ROOT / "package.xml").getroot()
    assert adapter_xml.findtext("name") == "redrhex_policy_io"
    assert adapter_xml.findtext("export/build_type") == "ament_python"
    adapter_setup = (ADAPTER_ROOT / "setup.py").read_text(encoding="utf-8")
    assert "../../../source/redrhex_policy_io" in adapter_setup
    assert not (ADAPTER_ROOT / "redrhex_policy_io").exists()


def test_isolated_install_contains_policy_io_and_v2_runtime_data(tmp_path) -> None:
    prefix = tmp_path / "install"
    _install_ament_python_package(
        ADAPTER_ROOT,
        prefix,
        tmp_path / "policy_io_build",
    )
    _install_ament_python_package(
        CONTROLLER_ROOT,
        prefix,
        tmp_path / "controller_build",
    )

    site_packages = next(prefix.glob("lib/python*/site-packages"))
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(site_packages)
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from importlib.resources import files; "
                "import redrhex_policy_io; "
                "from redrhex_policy_io.contracts import StudentObservationContractV2; "
                "from redrhex_rl_controller.observation_builder_v2 import "
                "SensorObservationBuilderV2; "
                "from redrhex_rl_controller.policy_onnx_runner_v2 import "
                "SensorPolicyONNXRunnerV2; "
                "assert StudentObservationContractV2.SENSOR_FRAME_DIM == 36; "
                "assert SensorObservationBuilderV2.__module__.startswith("
                "'redrhex_rl_controller.'); "
                "assert SensorPolicyONNXRunnerV2.__module__.startswith("
                "'redrhex_rl_controller.'); "
                "assert files('redrhex_policy_io').joinpath("
                "'data/sensor_frame_v2_golden.json').is_file(); "
                "print(redrhex_policy_io.__file__)"
            ),
        ],
        cwd=tmp_path,
        env=environment,
        check=True,
        text=True,
        capture_output=True,
    )
    assert str(site_packages) in result.stdout
    assert (
        prefix
        / "share/ament_index/resource_index/packages/redrhex_policy_io"
    ).is_file()
    assert (
        prefix
        / "share/redrhex_rl_controller/launch/redrhex_policy_sensor_v2.launch.py"
    ).is_file()
    assert (
        prefix
        / "share/redrhex_rl_controller/config/redrhex_policy_sensor_v2.yaml"
    ).is_file()
    assert (
        prefix
        / "lib/redrhex_rl_controller/rl_controller_node_v2"
    ).is_file()
    installed_node = (
        site_packages
        / "redrhex_rl_controller/rl_controller_node_v2.py"
    ).read_text(encoding="utf-8")
    assert "from redrhex_policy_io" not in installed_node
    assert "SensorObservationBuilderV2" in installed_node
    installed_builder = (
        site_packages
        / "redrhex_rl_controller/observation_builder_v2.py"
    ).read_text(encoding="utf-8")
    assert "from redrhex_policy_io.contracts import" in installed_builder
