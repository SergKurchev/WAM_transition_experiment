from __future__ import annotations

"""DDS interface for WAM inference loop.

Subscribes to rt/lowstate (robot joint state from Isaac Sim, 50 Hz).
Publishes rt/lowcmd (direct 29-DOF joint position targets → Isaac Sim).
Publishes rt/run_command/cmd (velocity command → GEAR-SONIC, legacy).

Joint mapping (G1 29-DOF, unitree_hg convention):
  Joints  0-13 : both arms (7 left + 7 right)  ← WAM model controls these
  Joints 14-28 : legs / torso / gripper         ← held at current position

LowCmd motor_cmd fields (per joint):
  q    — target position (rad)
  dq   — target velocity (rad/s), set 0 for position control
  kp   — position gain (N⋅m/rad)
  kd   — velocity / damping gain (N⋅m⋅s/rad)
  tau  — feedforward torque (N⋅m), set 0 for pure PD
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
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_, LowCmd_, MotorCmd_
    from unitree_sdk2py.idl.std_msgs.msg.dds_ import String_
    DDS_AVAILABLE = True
except ImportError:
    DDS_AVAILABLE = False
    MotorCmd_ = None
    print("[dds] WARNING: unitree_sdk2_python not found — running in no-DDS stub mode", flush=True)


# ── Control gains ──────────────────────────────────────────────────────────────
# Conservative values for simulation.  Raise kp carefully if response is sluggish.
ARM_KP  = 80.0    # N·m/rad  — arm position stiffness
ARM_KD  = 2.0     # N·m·s/rad — arm damping
LEG_KP  = 200.0   # N·m/rad  — leg/torso stiffness (hold-position)
LEG_KD  = 10.0    # N·m·s/rad — leg damping

# G1 joint layout in unitree_hg IDL (35 motor slots):
#   [00-05] left leg   · [06-11] right leg  · [12-13] waist
#   [14-20] left arm   · [21-27] right arm  · [28] gripper
#   [29-34] unused (zeros)
TOTAL_MOTORS    = 35   # unitree_hg IDL fixed array size
ARM_JOINT_START = 14   # first arm index in LowState/LowCmd
ARM_JOINT_END   = 28   # last arm index + 1  → q[14:28] = 14 DOF
NUM_ARM_JOINTS  = ARM_JOINT_END - ARM_JOINT_START  # 14


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
        self._publisher = None          # rt/run_command/cmd  (GEAR-SONIC, legacy)
        self._lowcmd_publisher = None   # rt/lowcmd           (direct joint control)
        self._subscriber = None

    def init(self) -> None:
        if not DDS_AVAILABLE:
            print("[dds] Stub mode: no DDS traffic.", flush=True)
            return

        ChannelFactoryInitialize(0, self._iface)

        # ── Subscriber: robot state from Isaac Sim ─────────────────────────────
        self._subscriber = ChannelSubscriber("rt/lowstate", LowState_)
        self._subscriber.Init(self._on_lowstate, 10)

        # ── Publisher: direct joint control → Isaac Sim ────────────────────────
        self._lowcmd_publisher = ChannelPublisher("rt/lowcmd", LowCmd_)
        self._lowcmd_publisher.Init()

        # ── Publisher: velocity commands → GEAR-SONIC (legacy, optional) ───────
        self._publisher = ChannelPublisher("rt/run_command/cmd", String_)
        self._publisher.Init()

        print(f"[dds] Initialized on interface '{self._iface}'", flush=True)
        print(f"[dds] Publishing rt/lowcmd: arms at slots {ARM_JOINT_START}-{ARM_JOINT_END-1}, legs held 0-{ARM_JOINT_START-1}", flush=True)

    # ── DDS callbacks ──────────────────────────────────────────────────────────

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

    # ── Command publishers ─────────────────────────────────────────────────────

    def publish_lowcmd(
        self,
        arm_joints_14: list[float],
    ) -> None:
        """Publish joint position targets to Isaac Sim via rt/lowcmd.

        G1 joint layout (unitree_hg, 35 slots):
          [00-13] legs + waist  → held at current state (PD hold, LEG_KP/KD)
          [14-27] arms          → commanded from arm_joints_14 (ARM_KP/KD)
          [28-34] gripper/unused → passive (mode=0, zero gains)

        Args:
            arm_joints_14: 14 arm joint targets in radians.
                           Index 0-6  → left arm  (LowState slots 14-20).
                           Index 7-13 → right arm (LowState slots 21-27).
        """
        if not DDS_AVAILABLE or self._lowcmd_publisher is None:
            return

        state = self.get_state()

        try:
            # ── Build all 35 MotorCmd_ entries ─────────────────────────────────

            motor_cmds = []

            # [00-13] Legs + waist: hold at current lowstate position
            for i in range(ARM_JOINT_START):
                q_val = float(state.q[i]) if (state is not None and i < len(state.q)) else 0.0
                motor_cmds.append(
                    MotorCmd_(mode=1, q=q_val, dq=0.0, tau=0.0,
                              kp=LEG_KP, kd=LEG_KD, reserve=0)
                )

            # [14-27] Arms: commanded by WAM model
            for i in range(NUM_ARM_JOINTS):
                q_target = float(arm_joints_14[i]) if i < len(arm_joints_14) else 0.0
                motor_cmds.append(
                    MotorCmd_(mode=1, q=q_target, dq=0.0, tau=0.0,
                              kp=ARM_KP, kd=ARM_KD, reserve=0)
                )

            # [28-34] Gripper + unused: passive
            for i in range(TOTAL_MOTORS - ARM_JOINT_END):
                q_val = float(state.q[ARM_JOINT_END + i]) if (
                    state is not None and ARM_JOINT_END + i < len(state.q)) else 0.0
                motor_cmds.append(
                    MotorCmd_(mode=0, q=q_val, dq=0.0, tau=0.0,
                              kp=0.0, kd=0.0, reserve=0)
                )

            # ── Assemble and publish ────────────────────────────────────────────
            cmd = LowCmd_(
                mode_pr=0,
                mode_machine=0,
                motor_cmd=motor_cmds,
                reserve=[0, 0, 0, 0],
                crc=0,
            )
            self._lowcmd_publisher.Write(cmd)

        except Exception as e:
            print(f"[dds] publish_lowcmd failed: {e}", flush=True)

    def send_command(self, vx: float, vy: float, wz: float, body_height: float = 0.0) -> None:
        """Publish a velocity command to GEAR-SONIC (legacy).

        GEAR-SONIC must be running for this to have any effect.
        Currently not used — WAM publishes rt/lowcmd directly instead.
        """
        payload = json.dumps([vx, vy, wz, body_height])
        if self._publisher is not None:
            self._publisher.Write(String_(data=payload))
        else:
            print(f"[dds] stub cmd: {payload}", flush=True)
