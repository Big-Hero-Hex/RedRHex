from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    config = LaunchConfiguration("config")
    bridge_backend = LaunchConfiguration("bridge_backend")
    rinbo_allow_enable = LaunchConfiguration("rinbo_allow_enable")
    rinbo_main_max_pwm = LaunchConfiguration("rinbo_main_max_pwm")
    rinbo_disabled_legs = LaunchConfiguration("rinbo_disabled_legs")

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "config",
                default_value=PathJoinSubstitution(
                    [FindPackageShare("redrhex_lowlevel_bridge"), "config", "lowlevel_bridge.yaml"]
                ),
            ),
            DeclareLaunchArgument("bridge_backend", default_value="biorola_ros"),
            DeclareLaunchArgument("rinbo_allow_enable", default_value="true"),
            DeclareLaunchArgument("rinbo_main_max_pwm", default_value="80.0"),
            DeclareLaunchArgument("rinbo_disabled_legs", default_value="l3"),
            Node(
                package="redrhex_lowlevel_bridge",
                executable="redrhex_lowlevel_bridge",
                name="redrhex_lowlevel_bridge",
                output="screen",
                parameters=[
                    config,
                    {
                        "bridge.backend": bridge_backend,
                        "rinbo.allow_enable": ParameterValue(rinbo_allow_enable, value_type=bool),
                        "rinbo.main_max_pwm": ParameterValue(rinbo_main_max_pwm, value_type=float),
                        "rinbo.disabled_legs": ParameterValue(rinbo_disabled_legs, value_type=str),
                    },
                ],
            ),
        ]
    )
