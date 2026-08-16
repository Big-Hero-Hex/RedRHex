"""Fail-closed bringup for the explicit Sensor-Only V2 controller route."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def _maybe_add(parameters: dict, name: str, value: str, value_type) -> None:
    text = value.strip()
    if not text:
        return
    parameters[name] = value_type(text)


def _launch_setup(context, *args, **kwargs):
    controller_parameters = [LaunchConfiguration("config").perform(context)]
    controller_overrides: dict[str, object] = {}
    _maybe_add(
        controller_overrides,
        "policy.onnx_path",
        LaunchConfiguration("onnx_path").perform(context),
        str,
    )
    _maybe_add(
        controller_overrides,
        "policy.sidecar_path",
        LaunchConfiguration("sidecar_path").perform(context),
        str,
    )
    if controller_overrides:
        controller_parameters.append(controller_overrides)

    bridge_parameters = [LaunchConfiguration("bridge_config").perform(context)]
    bridge_overrides: dict[str, object] = {}
    _maybe_add(
        bridge_overrides,
        "backend",
        LaunchConfiguration("bridge_backend").perform(context),
        str,
    )
    if bridge_overrides:
        bridge_parameters.append(bridge_overrides)

    return [
        Node(
            package="redrhex_rl_controller",
            executable="rl_controller_node_v2",
            name="redrhex_rl_controller_v2",
            output="screen",
            parameters=controller_parameters,
        ),
        Node(
            package="redrhex_lowlevel_bridge",
            executable="lowlevel_bridge_node",
            name="redrhex_lowlevel_bridge",
            output="screen",
            parameters=bridge_parameters,
            condition=IfCondition(LaunchConfiguration("start_bridge")),
        ),
    ]


def generate_launch_description() -> LaunchDescription:
    default_controller_config = PathJoinSubstitution(
        [
            FindPackageShare("redrhex_rl_controller"),
            "config",
            "redrhex_policy_sensor_v2.yaml",
        ]
    )
    default_bridge_config = PathJoinSubstitution(
        [
            FindPackageShare("redrhex_lowlevel_bridge"),
            "config",
            "lowlevel_bridge_sensor_v2.yaml",
        ]
    )
    return LaunchDescription(
        [
            DeclareLaunchArgument("config", default_value=default_controller_config),
            DeclareLaunchArgument("bridge_config", default_value=default_bridge_config),
            DeclareLaunchArgument("start_bridge", default_value="true"),
            DeclareLaunchArgument("onnx_path", default_value=""),
            DeclareLaunchArgument("sidecar_path", default_value=""),
            DeclareLaunchArgument(
                "bridge_backend",
                default_value="",
                description="Set rinbo_ros only after the V2 bridge calibration gates are verified.",
            ),
            OpaqueFunction(function=_launch_setup),
        ]
    )
