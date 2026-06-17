from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import PathJoinSubstitution


def generate_launch_description():
    config = LaunchConfiguration("config")
    onnx_path = LaunchConfiguration("onnx_path")
    fake_sensors = LaunchConfiguration("fake_sensors")

    controller = Node(
        package="redrhex_rl_controller",
        executable="redrhex_rl_controller",
        name="redrhex_rl_controller",
        output="screen",
        parameters=[config, {"policy.onnx_path": onnx_path}],
    )

    fake = Node(
        package="redrhex_rl_controller",
        executable="fake_redrhex_sensors",
        name="fake_redrhex_sensors",
        output="screen",
        condition=None,
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "config",
                default_value=PathJoinSubstitution(
                    [FindPackageShare("redrhex_rl_controller"), "config", "redrhex_policy.yaml"]
                ),
            ),
            DeclareLaunchArgument("onnx_path", default_value="/home/jetson/RedRHex/policy.onnx"),
            DeclareLaunchArgument("fake_sensors", default_value="false"),
            controller,
            # ROS 2 launch conditions are intentionally not used here to keep this file
            # simple for beginners. Start fake sensors separately in the mock test.
        ]
    )
