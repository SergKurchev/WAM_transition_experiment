"""DDS bridge between the slim Isaac Sim runtime and ``unitree_sdk2py``.

Subscribes to ``rt/lowcmd``, publishes ``rt/lowstate``. Topic names, message
layout, and joint order match ``modules/gwbc/gear_sonic/utils/mujoco_sim/
unitree_sdk2py_bridge.py`` — that reference is the contract for the C++
``g1_deploy_onnx_ref`` consumer.

Isaac Lab DOF ↔ hardware DOF reordering uses the two index maps exported by
``gear_sonic.envs.manager_env.robots.g1`` — no joint-name lists live here.
"""
from __future__ import annotations

import mmap
import os
import threading
import time
from typing import Any, Optional

import traceback

import numpy as np
import torch
from unitree_sdk2py.core.channel import (
    ChannelFactoryInitialize,
    ChannelPublisher,
    ChannelSubscriber,
)
from unitree_sdk2py.idl.default import (
    unitree_hg_msg_dds__IMUState_ as IMUState_default,
    unitree_hg_msg_dds__LowCmd_ as LowCmd_default,
    unitree_hg_msg_dds__LowState_ as LowState_default,
    unitree_hg_msg_dds__OdoState_ as OdoState_default,
)
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import IMUState_, LowCmd_, LowState_, OdoState_

from sim.isaac.mws_msgs import CameraSignal_, LidarSignal_

from gear_sonic.envs.manager_env.robots.g1 import G1_ISAACLAB_TO_MUJOCO_DOF
from gear_sonic.isaac_utils.rotations import quat_rotate_inverse


NUM_BODY_MOTOR: int = 29
GRAVITY_Z: float = -9.81

# Shared-memory path and sizes for the Isaac -> bridge sensor transports.
# Both the Isaac container (writer) and the bridge container (reader) mount the
# same named volume at /run/mws in docker/dimos-compose.yml.
_CAMERA_SHM_PATH: str = "/run/mws/camera.rgb"
_CAMERA_SHM_SIZE: int = 1280 * 720 * 3  # D435i rgb8 at configured camera resolution
_LIDAR_SHM_PATH: str = "/run/mws/lidar.xyzi"
_LIDAR_POINT_STEP: int = 16  # x/y/z/intensity float32
_LIDAR_SHM_SIZE: int = 512 * 1024  # MID-360 approximation: ~20k points/scan


class IsaacDdsBridge:
    """DDS sub/pub pair for the slim Isaac Sim G1 runtime."""

    def __init__(self, domain_id: int = 0, network_interface: str = "lo") -> None:
        self._domain_id = domain_id
        self._network_interface = network_interface
        self._low_cmd: LowCmd_ = LowCmd_default()
        self._low_state: LowState_ = LowState_default()
        self._torso_imu_state: IMUState_ = IMUState_default()
        self._low_cmd_lock = threading.Lock()
        self._low_cmd_received: bool = False

        self._odo_state: OdoState_ = OdoState_default()
        self._low_state_puber: Optional[ChannelPublisher] = None
        self._torso_imu_puber: Optional[ChannelPublisher] = None
        self._odo_state_puber: Optional[ChannelPublisher] = None
        self._camera_puber: Optional[ChannelPublisher] = None
        self._lidar_puber: Optional[ChannelPublisher] = None
        self._low_cmd_suber: Optional[ChannelSubscriber] = None

        # Sensor SHM transport: camera pixels and lidar point bytes go to
        # /run/mws/*. The DDS topics only carry tiny metadata signals.
        self._camera_shm: Optional[mmap.mmap] = None
        self._camera_rgb_annotator: Any | None = None
        self._camera_depth_annotator: Any | None = None
        self._camera_render_product_path: str | None = None
        self._camera_signal_seq: int = 0
        self._camera_signal_msg: CameraSignal_ = CameraSignal_(
            tick=0, seq=0, width=1280, height=720, step=3840
        )
        self._lidar_shm: Optional[mmap.mmap] = None
        self._lidar_signal_seq: int = 0
        self._lidar_signal_msg: LidarSignal_ = LidarSignal_(
            tick=0,
            seq=0,
            width=0,
            height=1,
            point_step=_LIDAR_POINT_STEP,
            row_step=0,
            data_size=0,
        )

        # Reorder indices live on CPU: we copy joint data to CPU for DDS
        # serialization before reordering, so no device transfer per step.
        self._isaaclab_to_mujoco_np = np.asarray(G1_ISAACLAB_TO_MUJOCO_DOF, dtype=np.int64)

        # Split-instrumentation for publish_state(): the single opaque pub=
        # bucket in g1_sim's main-loop probe cannot tell us whether the
        # 100-320 ms first-call spikes live in (a) GPU->CPU copies / IMU
        # math, (b) Python message fill, or (c) DDS Write().  Log per-phase
        # timings for the first N calls so the warmup design has real data.
        # See task notes (2026-04-18 plan).
        self._pub_probe_max: int = 12
        self._pub_probe_count: int = 0

    def start(self) -> None:
        """Initialize DDS endpoints. ``ChannelSubscriber.Init`` registers the handler thread."""
        ChannelFactoryInitialize(self._domain_id, self._network_interface)

        self._low_state_puber = ChannelPublisher("rt/lowstate", LowState_)
        self._low_state_puber.Init()

        self._odo_state_puber = ChannelPublisher("rt/odostate", OdoState_)
        self._odo_state_puber.Init()

        self._camera_shm = self._open_shm_file(_CAMERA_SHM_PATH, _CAMERA_SHM_SIZE)

        self._camera_puber = ChannelPublisher("rt/camera/signal", CameraSignal_)
        self._camera_puber.Init()

        self._lidar_shm = self._open_shm_file(_LIDAR_SHM_PATH, _LIDAR_SHM_SIZE)

        self._lidar_puber = ChannelPublisher("rt/lidar/signal", LidarSignal_)
        self._lidar_puber.Init()

        # gear-sonic's g1_deploy_onnx_ref subscribes to rt/secondary_imu as a
        # separate channel (HG_IMU_TORSO); GatherRobotStateToLogger() aborts the
        # control loop if either LowState or this torso IMUState buffer is empty.
        self._torso_imu_puber = ChannelPublisher("rt/secondary_imu", IMUState_)
        self._torso_imu_puber.Init()

        self._low_cmd_suber = ChannelSubscriber("rt/lowcmd", LowCmd_)
        self._low_cmd_suber.Init(self._low_cmd_handler, 1)

    def stop(self) -> None:
        if self._low_cmd_suber is not None:
            try:
                self._low_cmd_suber.Close()
            except Exception:
                pass
        if self._low_state_puber is not None:
            try:
                self._low_state_puber.Close()
            except Exception:
                pass
        if self._torso_imu_puber is not None:
            try:
                self._torso_imu_puber.Close()
            except Exception:
                pass
        if self._odo_state_puber is not None:
            try:
                self._odo_state_puber.Close()
            except Exception:
                pass
        if self._camera_puber is not None:
            try:
                self._camera_puber.Close()
            except Exception:
                pass
        if self._lidar_puber is not None:
            try:
                self._lidar_puber.Close()
            except Exception:
                pass
        if self._camera_rgb_annotator is not None:
            try:
                if self._camera_render_product_path is not None:
                    self._camera_rgb_annotator.detach([self._camera_render_product_path])
                else:
                    self._camera_rgb_annotator.detach()
            except Exception:
                pass
            self._camera_rgb_annotator = None
        if self._camera_depth_annotator is not None:
            try:
                if self._camera_render_product_path is not None:
                    self._camera_depth_annotator.detach([self._camera_render_product_path])
                else:
                    self._camera_depth_annotator.detach()
            except Exception:
                pass
            self._camera_depth_annotator = None
            self._camera_render_product_path = None
        if self._camera_shm is not None:
            try:
                self._camera_shm.close()
            except Exception:
                pass
            try:
                os.unlink(_CAMERA_SHM_PATH)
            except OSError:
                pass
        if self._lidar_shm is not None:
            try:
                self._lidar_shm.close()
            except Exception:
                pass
            try:
                os.unlink(_LIDAR_SHM_PATH)
            except OSError:
                pass

    def _open_shm_file(self, path: str, size: int) -> mmap.mmap:
        """Create or resize a SHM-backed file and return an mmap handle."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
        try:
            os.ftruncate(fd, size)
            return mmap.mmap(fd, size)
        finally:
            os.close(fd)

    def _low_cmd_handler(self, msg: LowCmd_) -> None:
        with self._low_cmd_lock:
            self._low_cmd = msg
            self._low_cmd_received = True

    def configure_camera_export(self, camera: Any) -> None:
        """Attach a dedicated CPU RGB annotator to the camera render product.

        The image-export path should bypass ``camera.data.output`` entirely.
        That Isaac Lab cache is appropriate for in-process sensor consumers, but
        the bridge needs a stable CPU frame for SHM export after each render.

        Args:
            camera: Isaac Lab camera sensor object from the scene.
        """
        import omni.replicator.core as rep

        render_product_paths = list(camera.render_product_paths)
        if len(render_product_paths) != 1:
            raise RuntimeError(
                f"Expected exactly one camera render product, got {len(render_product_paths)}."
            )
        render_product_path = render_product_paths[0]
        if (
            self._camera_render_product_path == render_product_path
            and self._camera_rgb_annotator is not None
        ):
            return
        if self._camera_rgb_annotator is not None and self._camera_render_product_path is not None:
            try:
                self._camera_rgb_annotator.detach([self._camera_render_product_path])
            except Exception:
                pass
        if self._camera_depth_annotator is not None and self._camera_render_product_path is not None:
            try:
                self._camera_depth_annotator.detach([self._camera_render_product_path])
            except Exception:
                pass
            self._camera_depth_annotator = None

        annotator = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
        # Mirror Isaac Lab's own camera setup: attach the annotator directly to
        # the single render-product path rather than wrapping it in a list.
        annotator.attach(render_product_path)
        self._camera_rgb_annotator = annotator
        self._camera_render_product_path = render_product_path

    def read_camera_depth_meters(self) -> Optional[np.ndarray]:
        """Return the latest D435i depth image in meters without publishing it.

        This is intentionally passive: the current stack exports only RGB over
        DDS/LCM, but callers can opt into D435i depth from the same render
        product after ``configure_camera_export()`` and a render pass.
        """
        if self._camera_render_product_path is None:
            raise RuntimeError("configure_camera_export() must run before reading camera depth.")
        if self._camera_depth_annotator is None:
            import omni.replicator.core as rep

            annotator = rep.AnnotatorRegistry.get_annotator(
                "distance_to_image_plane",
                device="cpu",
            )
            annotator.attach(self._camera_render_product_path)
            self._camera_depth_annotator = annotator

        output = self._camera_depth_annotator.get_data(device="cpu")
        depth = output["data"] if isinstance(output, dict) else output
        if depth is None:
            return None

        depth_np = np.asarray(depth, dtype=np.float32)
        if depth_np.size == 0:
            return None
        if depth_np.ndim == 4:
            if depth_np.shape[0] != 1:
                raise RuntimeError(f"Expected one depth frame, got batch shape {depth_np.shape}.")
            depth_np = depth_np[0]
        if depth_np.ndim == 3 and depth_np.shape[-1] == 1:
            depth_np = depth_np[..., 0]
        if depth_np.ndim != 2:
            raise RuntimeError(f"Expected depth image with 2 dims, got shape {depth_np.shape}.")
        return np.ascontiguousarray(depth_np)

    def read_cmd(
        self, device: torch.device | str = "cpu"
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Return the latest commanded ``(q, dq)`` in hardware order, or ``(None, None)``.

        PhysX's implicit PD uses both position and velocity targets; dropping
        ``dq`` would make the damping term always pull joints toward zero
        velocity, fighting policy-commanded leg-swing velocities during walking.

        Shape: each tensor is ``[NUM_BODY_MOTOR]`` on ``device``. Returns
        ``(None, None)`` until at least one ``rt/lowcmd`` message has been
        received.
        """
        with self._low_cmd_lock:
            if not self._low_cmd_received:
                return None, None
            q = [self._low_cmd.motor_cmd[i].q for i in range(NUM_BODY_MOTOR)]
            dq = [self._low_cmd.motor_cmd[i].dq for i in range(NUM_BODY_MOTOR)]
        q_t = torch.tensor(q, dtype=torch.float32, device=device)
        dq_t = torch.tensor(dq, dtype=torch.float32, device=device)
        return q_t, dq_t

    def publish_state(
        self,
        robot,
        body_idx: torch.Tensor,
        root_body_idx: int,
        torso_body_idx: int,
        sim_time: float,
    ) -> None:
        """Read ``robot.data`` and publish a ``LowState_`` on ``rt/lowstate``.

        Args:
            robot: ``Articulation`` from the scene.
            body_idx: 29 Isaac Lab DOF indices into ``robot.data.joint_pos`` (env 0).
            root_body_idx: Index of the root link (e.g., ``pelvis``) in
                ``robot.data.body_names`` — used for IMU acceleration readout.
            sim_time: Current sim time in seconds; encoded as ``tick`` in ms.
        """
        if self._low_state_puber is None or self._torso_imu_puber is None:
            raise RuntimeError("IsaacDdsBridge.start() must be called before publish_state().")

        _probe = self._pub_probe_count < self._pub_probe_max
        _t0 = time.perf_counter() if _probe else 0.0

        # Phase A: gather every needed tensor on GPU, then issue ONE
        # GPU->CPU copy.  Previously this method did ~13 separate
        # `.detach().cpu().numpy()` calls — each one a synchronous CUDA
        # memcpy that waits on the stream.  Batching cuts the per-call
        # sync count from ~13 to 1, which dominates publish_state's
        # steady-state cost (the actual bytes copied are negligible).
        joint_pos = robot.data.joint_pos[0, body_idx]                          # (29,)
        joint_vel = robot.data.joint_vel[0, body_idx]                          # (29,)
        joint_acc = robot.data.joint_acc[0, body_idx]                          # (29,)
        joint_tau = robot.data.applied_torque[0, body_idx]                     # (29,)
        root_quat_w = robot.data.root_quat_w[0]                                # (4,) (w, x, y, z)
        root_ang_vel_b = robot.data.root_ang_vel_b[0]                          # (3,)
        # Mujoco ref publishes ``mj_data.qacc[:3]`` — world-frame linear
        # acceleration of the floating base, no rotation to body frame.
        # Isaac's body_lin_acc_w is also world-frame, publish as-is.
        root_lin_acc_w = robot.data.body_lin_acc_w[0, root_body_idx, :]        # (3,)
        # Torso IMU: body-frame angular velocity needs quat_rotate_inverse,
        # done on GPU before the cat.  Accelerometer field is left zero
        # (mujoco ref does the same — see comment further down).
        torso_quat_w = robot.data.body_quat_w[0, torso_body_idx]               # (4,)
        torso_ang_vel_w = robot.data.body_ang_vel_w[0, torso_body_idx]
        torso_ang_vel_b = quat_rotate_inverse(
            torso_quat_w.unsqueeze(0), torso_ang_vel_w.unsqueeze(0), False
        )[0]                                                                    # (3,)
        # OdoState fields — same single-shot batch.
        root_pos_w = robot.data.root_pos_w[0]                                  # (3,)
        root_lin_vel_w = robot.data.root_lin_vel_w[0]                          # (3,)
        root_ang_vel_w = robot.data.root_ang_vel_w[0]                          # (3,)

        # One sync.  Layout:
        #   [0:29]    joint_pos          [29:58]   joint_vel
        #   [58:87]   joint_acc          [87:116]  joint_tau
        #   [116:120] root_quat          [120:123] root_gyro_b
        #   [123:126] root_acc_w         [126:130] torso_quat
        #   [130:133] torso_gyro_b       [133:136] root_pos
        #   [136:139] root_lin_vel       [139:142] root_ang_vel
        batch_np = torch.cat([
            joint_pos, joint_vel, joint_acc, joint_tau,
            root_quat_w, root_ang_vel_b, root_lin_acc_w,
            torso_quat_w, torso_ang_vel_b,
            root_pos_w, root_lin_vel_w, root_ang_vel_w,
        ]).detach().cpu().numpy()

        joint_pos_isaac = batch_np[0:29]
        joint_vel_isaac = batch_np[29:58]
        joint_acc_isaac = batch_np[58:87]
        joint_tau_isaac = batch_np[87:116]
        quat_np = batch_np[116:120]
        gyro_np = batch_np[120:123]
        acc_np = batch_np[123:126]
        torso_quat_np = batch_np[126:130]
        torso_gyro_np = batch_np[130:133]
        pos_np = batch_np[133:136]
        lin_vel_np = batch_np[136:139]
        ang_vel_np = batch_np[139:142]

        # Sanitize NaN on CPU (cheap; previously a torch.isfinite on GPU
        # which forced an extra sync before the batched copy existed).
        if not np.isfinite(acc_np).all():
            acc_np = np.zeros(3, dtype=acc_np.dtype)

        q_hw = joint_pos_isaac[self._isaaclab_to_mujoco_np]
        dq_hw = joint_vel_isaac[self._isaaclab_to_mujoco_np]
        ddq_hw = joint_acc_isaac[self._isaaclab_to_mujoco_np]
        tau_hw = joint_tau_isaac[self._isaaclab_to_mujoco_np]

        _t_a = time.perf_counter() if _probe else 0.0

        # Phase B: Python message fill.  Should be cheap and steady (pure CPU).
        for i in range(NUM_BODY_MOTOR):
            m = self._low_state.motor_state[i]
            m.q = float(q_hw[i])
            m.dq = float(dq_hw[i])
            m.ddq = float(ddq_hw[i])
            m.tau_est = float(tau_hw[i])

        self._low_state.imu_state.quaternion[:] = [float(x) for x in quat_np]
        self._low_state.imu_state.gyroscope[:] = [float(x) for x in gyro_np]
        self._low_state.imu_state.accelerometer[:] = [float(x) for x in acc_np]

        self._low_state.tick = int(sim_time * 1e3)

        _t_b = time.perf_counter() if _probe else 0.0

        self._torso_imu_state.quaternion[:] = [float(x) for x in torso_quat_np]
        self._torso_imu_state.gyroscope[:] = [float(x) for x in torso_gyro_np]
        self._torso_imu_state.accelerometer[:] = [0.0, 0.0, 0.0]

        # Phase C: DDS Write().  First call(s) can block on CycloneDDS
        # participant discovery / QoS handshake with gear-sonic.
        self._low_state_puber.Write(self._low_state)
        self._torso_imu_puber.Write(self._torso_imu_state)

        # OdoState: base pose and twist for the MWS bridge → DimOS /odom path.
        # root_quat_w is (w, x, y, z); OdoState_.orientation is [x, y, z, w].
        # quat_np / pos_np / lin_vel_np / ang_vel_np come from the batched
        # GPU->CPU copy at the top of this method.
        try:
            self._odo_state.position[:] = [float(pos_np[0]), float(pos_np[1]), float(pos_np[2])]
            self._odo_state.orientation[:] = [float(quat_np[1]), float(quat_np[2]), float(quat_np[3]), float(quat_np[0])]
            self._odo_state.linear_velocity[:] = [float(lin_vel_np[0]), float(lin_vel_np[1]), float(lin_vel_np[2])]
            self._odo_state.angular_velocity[:] = [float(ang_vel_np[0]), float(ang_vel_np[1]), float(ang_vel_np[2])]
            self._odo_state.tick = int(sim_time * 1e3)
            self._odo_state_puber.Write(self._odo_state)
        except Exception as exc:
            print(f"[dds_bridge] odostate publish failed: {exc}", flush=True)

        if _probe:
            _t_c = time.perf_counter()
            a_ms = (_t_a - _t0) * 1e3
            b_ms = (_t_b - _t_a) * 1e3
            c_ms = (_t_c - _t_b) * 1e3
            total_ms = (_t_c - _t0) * 1e3
            print(
                f"[dds_bridge::pub_probe] call={self._pub_probe_count} "
                f"total={total_ms:.1f}ms  "
                f"A_gpu_copy={a_ms:.1f}ms  B_msg_fill={b_ms:.1f}ms  "
                f"C_dds_write={c_ms:.1f}ms",
                flush=True,
            )
            self._pub_probe_count += 1

    def _read_camera_rgb(self) -> Optional[np.ndarray]:
        """Return a C-contiguous (H, W, 3) uint8 RGB array, or None on failure.

        Reads directly from a CPU Replicator annotator attached to the camera's
        render product. This keeps the exporter on the supported "render ->
        CPU array -> SHM" path and avoids Isaac Lab's internal ``camera.data``
        caching for the bridge use-case.
        """
        if self._camera_rgb_annotator is None:
            raise RuntimeError("configure_camera_export() must be called before publish_camera().")

        output = self._camera_rgb_annotator.get_data(device="cpu")
        if output is None:
            return None
        rgb = output["data"] if isinstance(output, dict) else output
        if rgb is None:
            return None

        rgb_np = np.asarray(rgb, dtype=np.uint8)
        if rgb_np.size == 0:
            return None
        if rgb_np.ndim == 4:
            if rgb_np.shape[0] != 1:
                raise RuntimeError(f"Expected one camera frame, got batch shape {rgb_np.shape}.")
            rgb_np = rgb_np[0]
        if rgb_np.ndim != 3:
            raise RuntimeError(f"Expected camera frame with 3 dims, got shape {rgb_np.shape}.")
        if rgb_np.shape[2] < 3:
            raise RuntimeError(f"Expected at least 3 channels in camera frame, got shape {rgb_np.shape}.")
        return np.ascontiguousarray(rgb_np[..., :3])

    def publish_camera(self, sim_time: float) -> None:
        """Write camera sensor RGB output to SHM and publish a CameraSignal_ on rt/camera/signal.

        Camera data is only valid after sim.render() and after
        ``configure_camera_export()`` has attached the CPU annotator.
        Publishing is a no-op when no camera publisher is available (e.g.
        before start()).

        Args:
            sim_time: Current simulation time in seconds.
        """
        if self._camera_puber is None:
            return
        try:
            rgb_np = self._read_camera_rgb()
            if rgb_np is None:
                return
            h, w = rgb_np.shape[:2]
            raw = rgb_np.tobytes()
            self._camera_shm.seek(0)
            self._camera_shm.write(raw)
            self._camera_signal_msg.tick = int(sim_time * 1e3)
            self._camera_signal_msg.seq = self._camera_signal_seq
            self._camera_signal_msg.width = w
            self._camera_signal_msg.height = h
            self._camera_signal_msg.step = w * 3
            self._camera_puber.Write(self._camera_signal_msg)
            self._camera_signal_seq = (self._camera_signal_seq + 1) & 0xFFFFFFFF
        except Exception as exc:
            print(f"[dds_bridge] camera publish failed: {exc}", flush=True)
            traceback.print_exc()

    def publish_lidar(self, ray_caster, sim_time: float) -> None:
        """Write RayCaster hit positions to SHM and publish LidarSignal_ on rt/lidar/signal.

        RayCaster does not require a render pass — this method can be called
        every physics step or at any sub-rate.

        Args:
            ray_caster: Isaac Lab RayCaster sensor object from the scene.
            sim_time: Current simulation time in seconds.
        """
        if self._lidar_puber is None:
            return
        try:
            ray_hits = ray_caster.data.ray_hits_w  # (N_envs, N_rays, 3)
            hits = ray_hits[0]  # (N_rays, 3)
            valid_mask = torch.isfinite(hits).all(dim=-1)
            valid_hits = hits[valid_mask].detach().cpu().numpy().astype(np.float32)
            n = len(valid_hits)
            if n == 0:
                return
            # Pack XYZI: x, y, z float32 + intensity=0.0 → 16 bytes per point
            intensity = np.zeros((n, 1), dtype=np.float32)
            point_data = np.ascontiguousarray(
                np.hstack([valid_hits, intensity])
            )  # (N, 4) float32
            data_bytes = point_data.tobytes()
            data_size = len(data_bytes)
            if data_size > _LIDAR_SHM_SIZE:
                raise RuntimeError(
                    f"Lidar payload {data_size} exceeds SHM capacity {_LIDAR_SHM_SIZE}."
                )
            self._lidar_shm.seek(0)
            self._lidar_shm.write(data_bytes)
            self._lidar_signal_msg.tick = int(sim_time * 1e3)
            self._lidar_signal_msg.seq = self._lidar_signal_seq
            self._lidar_signal_msg.width = n
            self._lidar_signal_msg.height = 1
            self._lidar_signal_msg.point_step = _LIDAR_POINT_STEP
            self._lidar_signal_msg.row_step = _LIDAR_POINT_STEP * n
            self._lidar_signal_msg.data_size = data_size
            self._lidar_puber.Write(self._lidar_signal_msg)
            self._lidar_signal_seq = (self._lidar_signal_seq + 1) & 0xFFFFFFFF
        except Exception as exc:
            print(f"[dds_bridge] lidar publish failed: {exc}", flush=True)
