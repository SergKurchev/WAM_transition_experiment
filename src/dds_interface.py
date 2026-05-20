"""DDS interface for WAM inference loop.

Subscribes to rt/lowstate (robot joint state from Isaac Sim).
Publishes rt/run_command/cmd (velocity command to GEAR-SONIC).

Command format: JSON string "[vx, vy, wz, body_height]"
  vx, vy       — body-frame linear velocity (m/s)
  wz           — yaw rate (rad/s)
  body_height  — height offset (m), 0.0 = policy default
"""

import json
import os
import sys
import threading
import time
from dataclasses import dataclass

# unitree_sdk2_python is bind-mounted at /workspace/unitree_sdk2py_src
sys.path.insert(0, "/workspace/unitree_sdk2py_src")

try:
    from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber, ChannelFactoryInitialize
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_
    from unitree_sdk2py.idl.std_msgs.msg.dds_ import String_
    DDS_AVAILABLE = True
except ImportError:
    DDS_AVAILABLE = False
    print("[dds] WARNING: unitree_sdk2_python not found — running in no-DDS stub mode", flush=True)


@dataclass
class RobotState:
    """Parsed snapshot of rt/lowstate."""
    q: list[float]       # joint positions, 29 DOF
    dq: list[float]      # joint velocities, 29 DOF
    tau: list[float]     # joint torques, 29 DOF
    timestamp: float     # wall time of last update


class DDSInterface:
    """Thin DDS wrapper for the WAM inference loop."""

    def __init__(self, iface: str = "lo"):
        self._iface = iface
        self._state: RobotState | None = None
        self._state_lock = threading.Lock()
        self._publisher = None
        self._subscriber = None

    def init(self) -> None:
        if not DDS_AVAILABLE:
            print("[dds] Stub mode: no DDS traffic.", flush=True)
            return

        ChannelFactoryInitialize(0, self._iface)

        self._subscriber = ChannelSubscriber("rt/lowstate", LowState_)
        self._subscriber.Init(self._on_lowstate, 10)

        self._publisher = ChannelPublisher("rt/run_command/cmd", String_)
        self._publisher.Init()

        print(f"[dds] Initialized on interface '{self._iface}'", flush=True)

    def _on_lowstate(self, msg: LowState_) -> None:
        with self._state_lock:
            self._state = RobotState(
                q=[j.q for j in msg.motor_state],
                dq=[j.dq for j in msg.motor_state],
                tau=[j.tau_est for j in msg.motor_state],
                timestamp=time.time(),
            )

    def get_state(self) -> RobotState | None:
        with self._state_lock:
            return self._state

    def send_command(self, vx: float, vy: float, wz: float, body_height: float = 0.0) -> None:
        """Publish a velocity command to GEAR-SONIC."""
        payload = json.dumps([vx, vy, wz, body_height])
        if self._publisher is not None:
            msg = String_()
            msg.data = payload
            self._publisher.Write(msg)
        else:
            print(f"[dds] stub cmd: {payload}", flush=True)
