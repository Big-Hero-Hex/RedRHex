from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    config = LaunchConfiguration("config")
    onnx_path = LaunchConfiguration("onnx_path")

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "config",
                default_value=PathJoinSubstitution(
                    [FindPackageShare("redrhex_five_leg_mode"), "config", "five_leg_policy.yaml"]
                ),
            ),
            DeclareLaunchArgument("onnx_path", default_value="/home/jetson/RedRHex/policy.onnx"),
            Node(
                package="redrhex_five_leg_mode",
                executable="five_leg_rl_controller",
                name="redrhex_rl_controller",
                output="screen",
                parameters=[config, {"policy.onnx_path": onnx_path}],
            ),
        ]
    )
