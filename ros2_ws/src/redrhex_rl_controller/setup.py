from glob import glob
from setuptools import find_packages, setup

package_name = "redrhex_rl_controller"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml", "README.md"]),
        (f"share/{package_name}/config", glob("config/*.yaml")),
        (f"share/{package_name}/launch", glob("launch/*.launch.py")),
        (f"share/{package_name}/scripts", glob("scripts/*.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Jason Liao",
    maintainer_email="jason@example.com",
    description="Bench-safe RedRhex ONNX policy controller for ROS 2.",
    license="MIT",
    entry_points={
        "console_scripts": [
            "redrhex_rl_controller = redrhex_rl_controller.rl_controller_node:main",
            "fake_redrhex_sensors = redrhex_rl_controller.fake_sensor_node:main",
            "preflight_check = redrhex_rl_controller.preflight_check:main",
            "check_onnx_io = redrhex_rl_controller.check_onnx_io:main",
            "compare_onnx_with_torch = redrhex_rl_controller.compare_onnx_with_torch:main",
            "motor_command_tool = redrhex_rl_controller.motor_command_tool:main",
            "estop_tool = redrhex_rl_controller.estop_tool:main",
        ],
    },
)
