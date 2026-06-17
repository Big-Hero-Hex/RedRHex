from __future__ import annotations

import argparse
from typing import Sequence


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Print a copy-paste BioRoLa/sbRIO bench-safe bringup plan.")
    parser.add_argument("--sbrio-ip", default="192.168.0.100")
    parser.add_argument("--orin-ip", default="<ORIN_IP>")
    parser.add_argument("--onnx", default="/home/jetson/RedRHex/policy.onnx")
    args = parser.parse_args(argv)
    print(
        f"""
Bench-safe RedRhex Sim2Real bringup plan

sbRIO terminal:
  ssh admin@{args.sbrio_ip}
  cd rinbo_sbRIO_ws/rinbo_fpga_driver/build/
  export CORE_LOCAL_IP={args.sbrio_ip}
  export CORE_MASTER_ADDR={args.sbrio_ip}:50051
  pkill -f grpccore || true
  pkill -f fpga_driver || true
  nohup /home/admin/rinbo_sbRIO_ws/install/bin/grpccore >/tmp/grpccore.log 2>&1 &
  nohup /home/admin/rinbo_sbRIO_ws/rinbo_fpga_driver/build/fpga_driver >/tmp/fpga_driver.log 2>&1 &
  ps -ef | egrep "grpccore|fpga_driver" | grep -v grep
  netstat -tn | grep 50051 || echo "NO TCP on 50051"

Jetson terminal 1:
  cd ~/rinbo_ros_ws
  source /opt/ros/humble/setup.bash
  source install/setup.bash
  export CORE_MASTER_ADDR={args.sbrio_ip}:50051
  export CORE_LOCAL_IP={args.orin_ip}
  ros2 run rinbo_ros_bridge rinbo_ros_bridge

Jetson terminal 2:
  cd ~/RedRhex/RedRhex/ros2_ws
  source /opt/ros/humble/setup.bash
  source ~/rinbo_ros_ws/install/setup.bash
  colcon build --symlink-install
  source install/setup.bash
  ros2 run redrhex_rl_controller preflight_check --onnx {args.onnx} --hardware-profile bench
  ros2 launch redrhex_lowlevel_bridge lowlevel_bridge.launch.py bridge_backend:=biorola_ros rinbo_allow_enable:=false
  ros2 launch redrhex_rl_controller redrhex_policy_bringup.launch.py onnx_path:={args.onnx}

Preview-only first. For real motor output later, restart lowlevel bridge with:
  ros2 launch redrhex_lowlevel_bridge lowlevel_bridge.launch.py bridge_backend:=biorola_ros rinbo_allow_enable:=true
"""
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
