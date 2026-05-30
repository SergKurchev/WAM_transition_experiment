"""LingBot-VA client for WAM stack.

Architecture:
  - LingBot-VA server runs in a separate container (lingbot-server)
    executing wan_va/wan_va_server.py on port 29056 (WebSocket).
  - This module is the client: sends camera frames + state, receives
    30D action chunk, extracts arm joint targets.

30D RoboTwin action layout (from va_robotwin_cfg.py):
  [0:3]   left  EEF  xyz  (metres)
  [3:7]   left  EEF  quaternion (xyzw)
  [7:10]  right EEF  xyz
  [10:14] right EEF  quaternion
  [14:28] unused (zeros) — joint angles slot, not predicted by robotwin cfg
  [28]    left  gripper  [0, 1]
  [29]    right gripper  [0, 1]

For G1 we need joint angles.  Two paths:
  1. IK (pinocchio) on EEF → joint targets  (enabled when LINGBOT_USE_IK=1)
  2. Direct EEF use — only valid after G1-specific fine-tune that fills [14:28]

Reference:
  Server:        repos/lingbot-va/wan_va/wan_va_server.py
  Client proto:  repos/lingbot-va/evaluation/robotwin/websocket_client_policy.py
  Config:        repos/lingbot-va/wan_va/configs/va_robotwin_cfg.py
"""

from __future__ import annotations

import logging
import os
import time
from collections import deque
from typing import Optional

import cv2
import msgpack
import msgpack_numpy
import numpy as np
import websockets.sync.client

log = logging.getLogger(__name__)

# ── constants ────────────────────────────────────────────────────────────────

SERVER_HOST = os.environ.get("LINGBOT_HOST", "127.0.0.1")
SERVER_PORT = int(os.environ.get("LINGBOT_PORT", "29056"))
CAMERA_SHM_PATH       = "/run/mws/camera.rgb"
CAMERA_LEFT_WRIST_SHM = "/run/mws/camera_left_wrist.rgb"
CAMERA_RIGHT_WRIST_SHM = "/run/mws/camera_right_wrist.rgb"
CAMERA_W, CAMERA_H = 1280, 720
MODEL_W, MODEL_H = 320, 256         # robotwin config: width=320, height=256
G1_ARM_DOF = 14                     # arm slots 14..27 in rt/lowcmd
ACTION_DIM = 30
ACTION_CHUNK = 16                   # action_per_frame=16 in robotwin config
RECONNECT_DELAY = 5.0


# ── msgpack helpers (mirror of lingbot-va/evaluation/robotwin/msgpack_numpy.py) ─

msgpack_numpy.patch()


def _pack(obj) -> bytes:
    return msgpack.packb(obj, default=msgpack_numpy.encode)


def _unpack(data: bytes):
    return msgpack.unpackb(data, object_hook=msgpack_numpy.decode, raw=False)


# ── camera SHM reader (same pattern as unifolm.py) ──────────────────────────

def _read_shm_frame() -> Optional[np.ndarray]:
    """Return latest RGB frame from Isaac Sim SHM, or None."""
    if not os.path.exists(CAMERA_SHM_PATH):
        return None
    try:
        with open(CAMERA_SHM_PATH, "rb") as f:
            raw = f.read(CAMERA_W * CAMERA_H * 3)
        if len(raw) < CAMERA_W * CAMERA_H * 3:
            return None
        frame = np.frombuffer(raw, dtype=np.uint8).reshape(CAMERA_H, CAMERA_W, 3)
        return frame.copy()
    except Exception:
        return None


def _resize_frame(frame: np.ndarray, w: int, h: int) -> np.ndarray:
    return cv2.resize(frame, (w, h), interpolation=cv2.INTER_LINEAR)


# ── IK bridge (optional, requires pinocchio) ────────────────────────────────

_ik_robot = None


def _init_ik(urdf_path: str):
    global _ik_robot
    try:
        import pinocchio as pin
        _ik_robot = pin.RobotWrapper.BuildFromURDF(urdf_path)
        log.info("[LingBot] Pinocchio IK initialised from %s", urdf_path)
    except Exception as e:
        log.warning("[LingBot] Pinocchio unavailable (%s) — IK disabled", e)


def _eef_to_joints(
    left_xyz: np.ndarray,
    left_quat: np.ndarray,
    right_xyz: np.ndarray,
    right_quat: np.ndarray,
) -> np.ndarray:
    """Convert dual EEF poses to 14D G1 arm joint targets via IK.

    Returns zeros if pinocchio not available (caller should handle gracefully).
    """
    if _ik_robot is None:
        return np.zeros(G1_ARM_DOF, dtype=np.float32)

    import pinocchio as pin

    joints = np.zeros(G1_ARM_DOF, dtype=np.float32)
    try:
        # Left arm (G1 joints 14–20, pinocchio frames vary by URDF)
        T_left = pin.SE3(
            pin.Quaternion(left_quat[3], *left_quat[:3]).toRotationMatrix(),
            left_xyz,
        )
        q_left = pin.computeGeneralizedGravity(
            _ik_robot.model, _ik_robot.data, _ik_robot.q0
        )  # placeholder — real IK call depends on exact URDF joint names
        joints[:7] = np.zeros(7)  # TODO: fill with real IK result

        # Right arm (G1 joints 21–27)
        joints[7:] = np.zeros(7)  # TODO: fill with real IK result
    except Exception as e:
        log.debug("[LingBot] IK failed: %s", e)

    return joints


# ── WebSocket client ─────────────────────────────────────────────────────────

class LingBotVAModel:
    """LingBot-VA WebSocket client.

    Implements the same callable interface as UnifoLMModel:
        action_traj, state_traj, video = model(robot_state)
    where action_traj is [16, 14] arm joint targets.
    """

    def __init__(
        self,
        prompt: str = "pick and place green cube in white basket",
        use_ik: bool = False,
        urdf_path: str = "",
    ):
        self._prompt = prompt
        self._use_ik = use_ik
        self._ws: Optional[websockets.sync.client.ClientConnection] = None
        self._action_queue: deque[np.ndarray] = deque()
        self._step = 0
        self._cached_frame: Optional[np.ndarray] = None

        if use_ik and urdf_path:
            _init_ik(urdf_path)

        self._connect()

    # ── connection ──────────────────────────────────────────────────────────

    def _connect(self):
        uri = f"ws://{SERVER_HOST}:{SERVER_PORT}"
        log.info("[LingBot] Connecting to %s …", uri)
        while True:
            try:
                self._ws = websockets.sync.client.connect(
                    uri,
                    compression=None,
                    max_size=None,
                    ping_interval=None,
                    close_timeout=10,
                )
                metadata = _unpack(self._ws.recv())
                log.info("[LingBot] Connected. Server metadata: %s", metadata)
                return
            except Exception as e:
                log.warning("[LingBot] Server not ready (%s) — retry in %.0fs", e, RECONNECT_DELAY)
                time.sleep(RECONNECT_DELAY)

    def _ensure_connected(self):
        if self._ws is None:
            self._connect()

    # ── observation building ─────────────────────────────────────────────────

    def _read_cam(self, shm_path: str) -> Optional[np.ndarray]:
        """Read one camera frame from SHM; returns None on failure."""
        if not os.path.exists(shm_path):
            return None
        try:
            with open(shm_path, "rb") as f:
                raw = f.read(CAMERA_W * CAMERA_H * 3)
            if len(raw) < CAMERA_W * CAMERA_H * 3:
                return None
            return np.frombuffer(raw, dtype=np.uint8).reshape(CAMERA_H, CAMERA_W, 3).copy()
        except Exception:
            return None

    def _get_obs_images(self) -> dict[str, np.ndarray]:
        """Read head + wrist cameras from SHM. Falls back to cached/zeros."""
        head_raw = _read_shm_frame()
        left_raw  = self._read_cam(CAMERA_LEFT_WRIST_SHM)
        right_raw = self._read_cam(CAMERA_RIGHT_WRIST_SHM)

        # Fallback: use head camera for any missing wrist camera
        if head_raw is None:
            head_raw = self._cached_frame or np.zeros((CAMERA_H, CAMERA_W, 3), dtype=np.uint8)
        else:
            self._cached_frame = head_raw

        if left_raw is None:
            log.debug("[LingBot] left wrist SHM not ready — using head cam")
            left_raw = head_raw
        if right_raw is None:
            log.debug("[LingBot] right wrist SHM not ready — using head cam")
            right_raw = head_raw

        def to_chw(frame: np.ndarray) -> np.ndarray:
            return _resize_frame(frame, MODEL_W, MODEL_H).transpose(2, 0, 1)[None]

        return {
            "observation.images.cam_high":        to_chw(head_raw),
            "observation.images.cam_left_wrist":  to_chw(left_raw),
            "observation.images.cam_right_wrist": to_chw(right_raw),
        }

    def _build_state(self, arm_q: np.ndarray) -> np.ndarray:
        """Build 30D state from G1 14-DOF arm joints.

        RoboTwin state mirrors the 30D action layout.  We fill joints and
        leave EEF slots zero — the server normalises before encoding.
        """
        state = np.zeros(ACTION_DIM, dtype=np.float32)
        # Left arm: G1 slots 0..6 → state channels 7..13
        state[7:14] = arm_q[:7]
        # Right arm: G1 slots 7..13 → state channels 22..28 (but norm uses 0-13)
        # Mirror the robotwin convention: left first (0..6), right second (7..13)
        state[0:7] = arm_q[:7]   # left  joints → position 0..6
        state[7:14] = arm_q[7:]  # right joints → position 7..13
        return state[None]  # [1, 30]

    # ── inference ────────────────────────────────────────────────────────────

    def _infer(self, arm_q: np.ndarray, reset: bool = False) -> np.ndarray:
        """Send one observation to the server; return 30D action chunk [T, 30]."""
        obs = {
            "obs":               list(self._get_obs_images().values()),
            "state":             self._build_state(arm_q),
            "reset":             reset,
            "prompt":            self._prompt,
            "compute_kv_cache":  reset or self._step == 0,
        }

        self._ensure_connected()
        try:
            self._ws.send(_pack(obs))
            response = self._ws.recv()
            if isinstance(response, str):
                raise RuntimeError(f"Server error: {response}")
            result = _unpack(response)
            return np.array(result["action"], dtype=np.float32)  # [T, 30]
        except Exception as e:
            log.error("[LingBot] Inference failed: %s", e)
            self._ws = None
            return np.zeros((ACTION_CHUNK, ACTION_DIM), dtype=np.float32)

    # ── action extraction ────────────────────────────────────────────────────

    def _extract_arm_joints(self, action_chunk: np.ndarray) -> np.ndarray:
        """Convert 30D action chunk → [T, 14] G1 arm joint targets.

        Strategy:
          - If IK enabled: use EEF xyz+quat (channels 0..13) → Pinocchio IK
          - Else: use joint channels 14..27 if non-zero (post G1 fine-tune),
            otherwise fall back to IK or zeros.
        """
        T = action_chunk.shape[0]
        arm_targets = np.zeros((T, G1_ARM_DOF), dtype=np.float32)

        joint_channels = action_chunk[:, 14:28]  # [T, 14] — empty pre-finetune
        eef_channels = action_chunk[:, 0:14]      # [T, 14] — EEF poses

        if np.any(np.abs(joint_channels) > 1e-4):
            # Post-finetune: model fills joint channels directly
            arm_targets = joint_channels
            log.debug("[LingBot] Using direct joint channels (post-finetune)")
        elif self._use_ik and _ik_robot is not None:
            for t in range(T):
                arm_targets[t] = _eef_to_joints(
                    left_xyz=eef_channels[t, 0:3],
                    left_quat=eef_channels[t, 3:7],
                    right_xyz=eef_channels[t, 7:10],
                    right_quat=eef_channels[t, 10:14],
                )
            log.debug("[LingBot] Using Pinocchio IK")
        else:
            log.debug("[LingBot] No joint data and IK disabled — returning zeros")

        return arm_targets

    # ── public interface (matches UnifoLMModel.__call__) ─────────────────────

    def __call__(self, robot_state):
        """Run one control step.

        Args:
            robot_state: object with .q attribute [29] G1 joint positions.

        Returns:
            action_traj:  np.ndarray [16, 14]  arm joint targets
            state_traj:   np.ndarray [16, 14]  (zeros — no world model state)
            video_output: None                 (server keeps video internally)
        """
        arm_q = np.array(robot_state.q[14:28], dtype=np.float32)

        reset = self._step == 0
        action_chunk = self._infer(arm_q, reset=reset)   # [T, 30]
        self._step += 1

        action_traj = self._extract_arm_joints(action_chunk)  # [T, 14]
        state_traj = np.zeros_like(action_traj)

        norm = float(np.linalg.norm(action_traj[0]))
        log.info(
            "[LingBot] step=%d  action_0[0:3]=%s  norm=%.3f",
            self._step,
            action_traj[0, :3],
            norm,
        )

        return action_traj, state_traj, None
