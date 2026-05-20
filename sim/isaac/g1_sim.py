"""Slim Isaac Sim runtime for the G1 robot.

Bare ``InteractiveScene`` + ``SimulationContext`` step loop. No env class,
no manager stubs. See ``docs/research/2026-04-17-slim-isaac-env-design.md``.
"""
from __future__ import annotations

import copy
import contextlib
import math
import os
import time
from dataclasses import MISSING

import numpy as np

import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import CameraCfg, RayCasterCfg
from isaaclab.sensors.ray_caster.patterns import LidarPatternCfg
from isaaclab.sim import SimulationCfg, SimulationContext
from isaaclab.sim.spawners.lights import DistantLightCfg
from isaaclab.sim.spawners.materials import RigidBodyMaterialCfg
from isaaclab.sim.spawners.sensors import PinholeCameraCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

from gear_sonic.envs.env_utils.joint_utils import get_body_joint_indices
from gear_sonic.envs.manager_env.robots.g1 import (
    G1_CYLINDER_MODEL_12_DEX_CFG,
    G1_MUJOCO_TO_ISAACLAB_DOF,
)

from sim.isaac.dds_bridge import IsaacDdsBridge
from sim.isaac.startup_support import IsaacStartupSupport


SIM_DT: float = 0.005
DECIMATION: int = 4
# Publish camera every 5 env-steps (50/5 = 10 Hz); lidar every 5 (10 Hz).
# 5 chosen to align with render_every=10 — GUI renders always coincide with camera
# renders, avoiding orphan sim.render() calls.
CAMERA_PUBLISH_EVERY_N_STEPS: int = 5
LIDAR_PUBLISH_EVERY_N_STEPS: int = 5

_CAMERA_FOCAL_LENGTH_MM: float = 24.0
_D435I_WIDTH: int = 1280
_D435I_HEIGHT: int = 720
_D435I_FPS: float = 30.0
# Unitree's G1 FOV diagram shows the installed D435i vertical FOV as 55.2 deg;
# derive the horizontal pinhole aperture from the 16:9 stream aspect ratio.
_D435I_VERTICAL_FOV_DEG: float = 55.2
_D435I_HORIZONTAL_FOV_DEG: float = math.degrees(
    2.0
    * math.atan(
        math.tan(math.radians(_D435I_VERTICAL_FOV_DEG) / 2.0)
        * (_D435I_WIDTH / _D435I_HEIGHT)
    )
)
_MID360_SCAN_RATE_HZ: float = 10.0
_MID360_POINTS_PER_SECOND: int = 200_000
_MID360_POINTS_PER_SCAN: int = int(_MID360_POINTS_PER_SECOND / _MID360_SCAN_RATE_HZ)
_MID360_VERTICAL_SAMPLES: int = 40
_MID360_HORIZONTAL_SAMPLES: int = int(_MID360_POINTS_PER_SCAN / _MID360_VERTICAL_SAMPLES)
_MID360_HORIZONTAL_RES_DEG: float = 360.0 / _MID360_HORIZONTAL_SAMPLES
_MID360_MOUNT_DOWN_TILT_DEG: float = 2.3
_MID360_WORLD_VERTICAL_FOV_RANGE_DEG: tuple[float, float] = (-52.0, 7.0)
_MID360_LOCAL_VERTICAL_FOV_RANGE_DEG: tuple[float, float] = (
    _MID360_WORLD_VERTICAL_FOV_RANGE_DEG[0] + _MID360_MOUNT_DOWN_TILT_DEG,
    _MID360_WORLD_VERTICAL_FOV_RANGE_DEG[1] + _MID360_MOUNT_DOWN_TILT_DEG,
)
_REQUIRED_ROBOT_USD_SENSOR_FRAMES = (
    "/g1/pelvis/imu_in_pelvis",
    "/g1/torso_link/imu_in_torso",
    "/g1/torso_link/d435_link/d435_optical_frame",
    "/g1/torso_link/mid360_sensor_frame",
)


def _pinhole_horizontal_aperture(
    focal_length_mm: float,
    horizontal_fov_deg: float,
) -> float:
    """Return aperture in millimeters for a target horizontal field of view."""
    return 2.0 * focal_length_mm * math.tan(math.radians(horizontal_fov_deg) / 2.0)


@configclass
class G1SceneCfg(InteractiveSceneCfg):
    """Flat-ground scene with a G1 robot and distant lighting."""

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        physics_material=RigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=1.0,
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
        ),
        visual_material=None,
        debug_vis=False,
    )
    robot: ArticulationCfg = MISSING
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )
    # Unitree G1 uses a head-mounted Intel RealSense D435i. The preconverted
    # robot USD provides d435_link/d435_optical_frame, so the camera attaches
    # to that model frame with no authored pose offset.
    camera: CameraCfg = CameraCfg(
        prim_path="/World/envs/env_.*/Robot/torso_link/d435_link/d435_optical_frame/camera",
        update_period=1.0 / _D435I_FPS,
        height=_D435I_HEIGHT,
        width=_D435I_WIDTH,
        data_types=["rgb"],
        spawn=PinholeCameraCfg(
            focal_length=_CAMERA_FOCAL_LENGTH_MM,
            focus_distance=400.0,
            horizontal_aperture=_pinhole_horizontal_aperture(
                _CAMERA_FOCAL_LENGTH_MM,
                _D435I_HORIZONTAL_FOV_DEG,
            ),
            clipping_range=(0.1, 100.0),
        ),
        offset=CameraCfg.OffsetCfg(),
    )
    # Unitree G1 uses a Livox MID-360 at the head. Attach to the explicit
    # mid360_sensor_frame from the preconverted robot USD; this avoids the mesh-frame
    # rotation on mid360_link while keeping the mount in the robot model.
    ray_caster: RayCasterCfg = RayCasterCfg(
        prim_path="/World/envs/env_.*/Robot/torso_link/mid360_sensor_frame",
        mesh_prim_paths=["/World/ground"],
        pattern_cfg=LidarPatternCfg(
            channels=_MID360_VERTICAL_SAMPLES,
            # Unitree's diagram gives the absolute G1 envelope as 7 deg above
            # and 52 deg below horizontal. The URDF frame is already pitched
            # 2.3 deg down from horizontal (92.3 deg from vertical), and
            # ray_alignment="base" applies that full sensor-frame rotation.
            # Therefore the local pattern is shifted up by 2.3 deg.
            vertical_fov_range=_MID360_LOCAL_VERTICAL_FOV_RANGE_DEG,
            horizontal_res=_MID360_HORIZONTAL_RES_DEG,
            horizontal_fov_range=(0.0, 360.0),
        ),
        max_distance=70.0,
        ray_alignment="base",
    )


def _build_robot_cfg(robot_usd: str) -> ArticulationCfg:
    """Return a deep-copied ``G1_CYLINDER_MODEL_12_DEX_CFG`` with slim-env overrides."""
    if not os.path.isfile(robot_usd):
        raise FileNotFoundError(f"Configured G1 robot USD not found: {robot_usd}")
    _validate_robot_usd_sensor_frames(robot_usd)

    cfg = copy.deepcopy(G1_CYLINDER_MODEL_12_DEX_CFG)
    cfg.prim_path = "/World/envs/env_.*/Robot"
    # The DDS bridge reads body_lin_acc_w for the IMU accelerometer.  With
    # retain_accelerations=False (the gwbc default for RL training), Isaac Lab
    # falls back to on-demand finite-difference compute on every read — extra
    # GPU sync that contributes to publish_state spikes at startup.  Override
    # locally; do not edit the gwbc submodule.
    cfg.spawn.rigid_props.retain_accelerations = True
    # gwbc's defaults (8 / 4) are training-tuned for stability under stiff
    # contact and high learning-rate policies.  For inference of an already-
    # trained policy on flat ground, fewer PhysX solver iterations are enough
    # and shave 1-2 ms off the per-substep `step` cost.  Velocity iter=0 is
    # the main stability risk: it can produce unphysical foot-strike rebound
    # under high-impact contact.  Raise velocity iter first if the robot
    # bounces or wobbles on landing; raise position iter for joint penetration.
    cfg.spawn.articulation_props.solver_position_iteration_count = 2
    cfg.spawn.articulation_props.solver_velocity_iteration_count = 0
    urdf_spawn = cfg.spawn
    cfg.spawn = sim_utils.UsdFileCfg(
        usd_path=robot_usd,
        activate_contact_sensors=urdf_spawn.activate_contact_sensors,
        rigid_props=copy.deepcopy(urdf_spawn.rigid_props),
        articulation_props=copy.deepcopy(urdf_spawn.articulation_props),
    )
    _probe(f"using required G1 robot USD: {robot_usd}")
    # Spawn 1 m back from scene origin so the robot isn't inside the scene asset.
    pos = cfg.init_state.pos
    cfg.init_state.pos = (pos[0] - 2.0, pos[1], pos[2])
    return cfg


def _validate_robot_usd_sensor_frames(robot_usd: str) -> None:
    """Fail early when runtime is pointed at a stale converted robot USD."""
    from pxr import Usd

    stage = Usd.Stage.Open(robot_usd)
    if stage is None:
        raise RuntimeError(f"Failed to open configured G1 robot USD: {robot_usd}")
    missing = [
        prim_path
        for prim_path in _REQUIRED_ROBOT_USD_SENSOR_FRAMES
        if not stage.GetPrimAtPath(prim_path).IsValid()
    ]
    if missing:
        raise RuntimeError(
            "Configured G1 robot USD is missing sensor frame(s) "
            f"{missing}; regenerate it with scripts/convert-g1-robot-usd.sh"
        )


def _reset_robot(scene: InteractiveScene) -> None:
    """Explicit reset per design §4 / gotcha #5."""
    robot = scene.articulations["robot"]
    default_root_state = robot.data.default_root_state.clone()
    default_root_state[:, :3] += scene.env_origins
    robot.write_root_pose_to_sim(default_root_state[:, :7])
    robot.write_root_velocity_to_sim(default_root_state[:, 7:])
    joint_pos = robot.data.default_joint_pos.clone()
    joint_vel = robot.data.default_joint_vel.clone()
    robot.write_joint_state_to_sim(joint_pos, joint_vel)
    scene.reset()


def _configure_rendering(
    sim: SimulationContext,
    *,
    headless: bool,
    no_render: bool,
    render_interval: int | None,
) -> tuple[bool, int]:
    """Configure SimulationContext rendering knobs and return render cadence."""
    render_every = max(1, render_interval or 10)
    if no_render:
        with contextlib.suppress(Exception):
            sim.render_interval = 1_000_000
        with contextlib.suppress(Exception):
            sim.render_mode = "offscreen"
        print("[g1_sim] rendering disabled via --no_render")
        return False, render_every
    if headless:
        with contextlib.suppress(Exception):
            sim.render_mode = "offscreen"
        with contextlib.suppress(Exception):
            sim.render_interval = max(1, render_interval or 1)
        return False, max(1, render_interval or 1)
    if render_interval is not None:
        with contextlib.suppress(Exception):
            sim.render_interval = render_every
    print()
    print("***  Please left-click on the Sim window to activate rendering.  ***")
    print()
    return True, render_every


def _set_default_camera_view(sim: SimulationContext, scene: InteractiveScene) -> None:
    """Place a simple free camera so noVNC starts on the robot, not the origin."""
    robot = scene.articulations["robot"]
    root_pos = robot.data.root_pos_w[0].detach().cpu()
    eye = (root_pos + torch.tensor([2.0, 2.0, 1.5])).tolist()
    target = root_pos.tolist()
    with contextlib.suppress(Exception):
        sim.set_camera_view(eye=eye, target=target)


def _probe(msg: str) -> None:
    print(f"[g1_sim::probe] {msg}", flush=True)


_RAYCASTER_SKIP_KEYWORDS = ("ceiling",)
_RAYCASTER_OUTPUT_PRIM = "/World/RayCasterMesh"


def _build_raycaster_mesh(stage: object, scene_prim_path: str) -> str:
    """Merge all scene meshes into a single prim for RayCasterCfg at runtime.

    Vertices are transformed to Isaac world space (Z-up, metres) using each
    mesh's full local-to-world transform, which already includes the Y-up→Z-up
    and cm→m ops on the scene root.  The merged prim is placed outside the
    scene root so those ops are not applied a second time.
    """
    from pxr import Gf, Usd, UsdGeom, Vt

    if stage.GetPrimAtPath(_RAYCASTER_OUTPUT_PRIM):
        stage.RemovePrim(_RAYCASTER_OUTPUT_PRIM)

    xform_cache = UsdGeom.XformCache()
    all_verts: list[np.ndarray] = []
    all_counts: list[int] = []
    all_indices: list[int] = []
    vert_offset = 0
    included = skipped = 0

    scene_prim = stage.GetPrimAtPath(scene_prim_path)
    if not scene_prim.IsValid():
        raise RuntimeError(f"Scene prim not found: {scene_prim_path}")
    for prim in Usd.PrimRange(scene_prim):
        if not prim.IsA(UsdGeom.Mesh):
            continue
        path_str = str(prim.GetPath())
        if any(kw in path_str.lower() for kw in _RAYCASTER_SKIP_KEYWORDS):
            skipped += 1
            continue

        mesh = UsdGeom.Mesh(prim)
        points = mesh.GetPointsAttr().Get()
        counts = mesh.GetFaceVertexCountsAttr().Get()
        indices = mesh.GetFaceVertexIndicesAttr().Get()
        if points is None or counts is None or indices is None or len(points) == 0:
            skipped += 1
            continue

        tf = np.array(
            [[xform_cache.GetLocalToWorldTransform(prim)[r][c] for c in range(4)] for r in range(4)],
            dtype=np.float64,
        )
        pts = np.array([[p[0], p[1], p[2], 1.0] for p in points], dtype=np.float64)
        transformed = (tf @ pts.T).T[:, :3].astype(np.float32)

        all_verts.append(transformed)
        all_counts.extend(list(counts))
        all_indices.extend(idx + vert_offset for idx in list(indices))
        vert_offset += len(points)
        included += 1

    if not all_verts:
        raise RuntimeError(f"No mesh geometry found under {scene_prim_path} after filtering.")

    merged_verts = np.vstack(all_verts)
    merged = UsdGeom.Mesh.Define(stage, _RAYCASTER_OUTPUT_PRIM)
    merged.GetPointsAttr().Set(
        Vt.Vec3fArray([Gf.Vec3f(float(v[0]), float(v[1]), float(v[2])) for v in merged_verts])
    )
    merged.GetFaceVertexCountsAttr().Set(Vt.IntArray(all_counts))
    merged.GetFaceVertexIndicesAttr().Set(Vt.IntArray(all_indices))
    UsdGeom.Imageable(merged).MakeInvisible()

    _probe(
        f"RayCasterMesh built: {included} mesh(es), {skipped} skipped, "
        f"{len(merged_verts):,} verts → {_RAYCASTER_OUTPUT_PRIM}"
    )
    return _RAYCASTER_OUTPUT_PRIM


def run(
    simulation_app,
    publish_every_n_sub_steps: int = 1,
    dds_domain_id: int = 0,
    dds_interface: str = "lo",
    headless: bool = False,
    no_render: bool = False,
    render_interval: int | None = None,
    scene_usd: str | None = None,
    robot_usd: str | None = None,
) -> None:
    """Run the bare sim loop until the SimulationApp stops.

    Args:
        simulation_app: The running ``SimulationApp`` returned by ``AppLauncher``.
        publish_every_n_sub_steps: Emit ``rt/lowstate`` once per N physics
            sub-steps.  Physics rate is ``1 / SIM_DT = 200 Hz`` (sim time), so
            ``publish_every_n_sub_steps=1`` publishes at the physics rate.
            Publishing inside the decimation loop (rather than once per env
            step) keeps ``rt/lowstate`` fresh for gear-sonic's CONTROL-loop
            watchdog even when the sim runs below real-time.
        dds_domain_id: CycloneDDS domain id; must match gear-sonic (0).
        dds_interface: Network interface for DDS multicast; ``lo`` matches
            gear-sonic's host-network setup.
        headless: Whether AppLauncher started a headless Kit experience.
        no_render: Disable rendering even when a GUI is available.
        render_interval: Render every N env steps in GUI mode.
        scene_usd: Optional external scene USD to reference under ``/World/Scene``.
        robot_usd: Preconverted G1 robot USD to spawn. Runtime does not
            convert or fall back to URDF.
    """
    if robot_usd is None:
        raise ValueError("robot_usd is required; prepare the robot USD before launching runtime.")

    _probe(f"entered run(); app.is_running={simulation_app.is_running()}")
    _startup_t0 = time.perf_counter()
    _startup_last = _startup_t0

    def _startup_timing(label: str) -> None:
        """Log startup timing deltas for scene-load regression tracking."""
        nonlocal _startup_last
        now = time.perf_counter()
        _probe(
            f"startup timing: {label}: "
            f"+{now - _startup_last:.3f}s total={now - _startup_t0:.3f}s"
        )
        _startup_last = now

    # Load scene reference before SimulationContext so the MDL/texture network is
    # present when the render session initialises and textures resolve correctly.
    _stage = None
    if scene_usd is not None:
        import omni.usd
        from pxr import Gf, UsdGeom
        _stage = omni.usd.get_context().get_stage()
        _scene_prim = _stage.DefinePrim("/World/Scene", "Xform")
        _scene_prim.GetReferences().AddReference(scene_usd)
        # Source is Y-up, centimeters. Isaac Sim is Z-up, meters.
        # xformOpOrder applies left-to-right: scale cm→m first, then rotate Y-up→Z-up.
        _xf = UsdGeom.Xformable(_scene_prim)
        _xf.ClearXformOpOrder()
        _xf.AddRotateXOp().Set(90.0)
        _xf.AddScaleOp().Set(Gf.Vec3f(0.01, 0.01, 0.01))
        _probe(f"scene USD loaded: {scene_usd} (Y-up→Z-up, cm→m)")
    _startup_timing("scene USD loaded")

    _probe("creating SimulationContext")
    sim = SimulationContext(SimulationCfg(dt=SIM_DT, device="cuda:0"))
    _probe(f"SimulationContext created; app.is_running={simulation_app.is_running()}")

    should_render, render_every = _configure_rendering(
        sim,
        headless=headless,
        no_render=no_render,
        render_interval=render_interval,
    )
    _probe(f"rendering configured; should_render={should_render}, render_every={render_every}")

    _probe("building G1SceneCfg + robot cfg")
    scene_cfg = G1SceneCfg(num_envs=1, env_spacing=2.0)
    scene_cfg.robot = _build_robot_cfg(robot_usd=robot_usd)

    if _stage is not None:
        raycaster_prim = _build_raycaster_mesh(_stage, "/World/Scene")
        scene_cfg.ray_caster.mesh_prim_paths = [raycaster_prim]
    _startup_timing("RayCasterMesh built")

    _probe("instantiating InteractiveScene")
    scene = InteractiveScene(scene_cfg)
    _probe(f"InteractiveScene created; app.is_running={simulation_app.is_running()}")
    _startup_timing("InteractiveScene created")

    _probe("calling sim.reset()")
    sim.reset()
    _probe(f"sim.reset() returned; app.is_running={simulation_app.is_running()}")
    _startup_timing("sim.reset() returned")
    # Warm-up update populates scene buffers before we reach into robot.data (design §1).
    _probe("first scene.update()")
    scene.update(dt=SIM_DT)
    _probe("calling _reset_robot()")
    _reset_robot(scene)
    _probe("post-reset scene.update()")
    scene.update(dt=SIM_DT)
    if should_render:
        _set_default_camera_view(sim, scene)
        sim.render()

    robot = scene.articulations["robot"]
    body_idx = get_body_joint_indices(robot)
    mujoco_to_isaac = torch.as_tensor(
        G1_MUJOCO_TO_ISAACLAB_DOF, dtype=torch.long, device=robot.device
    )
    root_body_idx = robot.data.body_names.index("pelvis")
    torso_body_idx = robot.data.body_names.index("torso_link")
    _probe(
        f"robot ready; body_idx shape={tuple(body_idx.shape) if hasattr(body_idx, 'shape') else 'n/a'}; "
        f"root_body_idx={root_body_idx} (pelvis); torso_body_idx={torso_body_idx} (torso_link)"
    )

    camera_sensor = scene.sensors["camera"]
    ray_caster_sensor = scene.sensors["ray_caster"]
    _probe("camera and ray_caster sensors acquired")

    # Wait until Isaac is fully started up before creating the DDS bridge.
    # Two distinct readiness signals, in order:
    #   1. USD stage load complete.  Asset cook / Fabric prim materialisation /
    #      MDL resolve are async background work driven by the Kit event loop.
    #      `simulation_app.update()` pumps that loop; `get_stage_loading_status`
    #      exposes the actual pending-asset count, which is the real completion
    #      signal — no thresholds, no caps.
    #   2. First render + first physics step.  These trigger RTX shader compile
    #      and PhysX scene flattening + CUDA kernel JIT respectively; both are
    #      one-shot costs that would otherwise land on main-loop iter 0 and
    #      starve DDS long enough for gear-sonic's LowState watchdog to fire.
    import omni.usd
    _usd_ctx = omni.usd.get_context()
    _probe("Kit warmup: waiting for USD stage to finish loading")
    _wait_frames = 0
    while True:
        simulation_app.update()
        _wait_frames += 1
        _msg, _loaded, _loading = _usd_ctx.get_stage_loading_status()
        if _loading == 0:
            break
        if _wait_frames % 50 == 0:
            _probe(
                f"  stage-loading: msg='{_msg}' loaded={_loaded} loading={_loading}"
            )
    _probe(
        f"Kit warmup: USD stage loaded after {_wait_frames} app.update() calls"
    )

    _probe("Kit warmup: commit render (flushes shader compile)")
    sim.render()
    _probe("Kit warmup: commit step (flattens PhysX, primes CUDA kernels)")
    sim.step(render=True)
    _probe("Kit warmup: done")

    _probe("starting IsaacDdsBridge")
    bridge = IsaacDdsBridge(domain_id=dds_domain_id, network_interface=dds_interface)
    bridge.start()
    bridge.configure_camera_export(camera_sensor)
    _probe("creating IsaacStartupSupport")
    support = IsaacStartupSupport(scene, sim, device=robot.device)

    # Pre-live burn-in.  The split-instrumentation in dds_bridge.publish_state
    # (see task notes 2026-04-18) showed the watchdog killer is CUDA / PhysX
    # first-call cost on the live-loop hot path: A_gpu_copy of 228->138->60 ms,
    # support.apply() spike of ~104 ms, scene.write_data_to_sim() spike of
    # ~77 ms in env_steps 0-1.  Run the FULL hot path here until per-iter
    # wall-clock is stable, so first-call costs land before gear-sonic's
    # 500 ms watchdog window.
    #
    # Critically the burn-in must include `set_joint_position_target` — the
    # action-ingress call invoked once gear-sonic sends its first rt/lowcmd.
    # Without exercising that path here, its first-call CUDA cost (~100 ms+)
    # lands inside the live loop, opens a publish gap, and trips the
    # watchdog right after gear-sonic prints "Init Done".  We feed the
    # default joint pose as a no-op target — same kernel signature as a real
    # command, but doesn't disturb the held robot.
    #
    # Convergence: last 10 iterations have max <= 1.2 * min — measured
    # stability, no absolute threshold.
    import time as _time_burnin
    _BURNIN_WINDOW = 10
    _BURNIN_RATIO = 1.2
    _probe("pre-live burn-in: warming CUDA / PhysX hot path (incl. action ingress)")
    _default_q = robot.data.default_joint_pos[0, body_idx].clone().unsqueeze(0)
    _zero_dq = torch.zeros_like(_default_q)
    _burnin_times: list[float] = []
    _burnin_iter = 0
    while True:
        _bi_t0 = _time_burnin.perf_counter()
        # Exercise the action-ingress path with the default pose so its CUDA
        # kernels are warm before the first real rt/lowcmd arrives.
        robot.set_joint_position_target(_default_q, joint_ids=body_idx)
        robot.set_joint_velocity_target(_zero_dq, joint_ids=body_idx)
        for _ in range(DECIMATION):
            if not support.released:
                support.apply()
            scene.write_data_to_sim()
            sim.step(render=False)
            scene.update(dt=SIM_DT)
            bridge.publish_state(
                robot, body_idx, root_body_idx, torso_body_idx,
                sim_time=sim.current_time,
            )
        _bi_dt = _time_burnin.perf_counter() - _bi_t0
        _burnin_times.append(_bi_dt)
        _burnin_iter += 1
        if len(_burnin_times) >= _BURNIN_WINDOW:
            _recent = _burnin_times[-_BURNIN_WINDOW:]
            _max_ms = max(_recent) * 1e3
            _min_ms = min(_recent) * 1e3
            if _min_ms > 0 and _max_ms / _min_ms <= _BURNIN_RATIO:
                _probe(
                    f"pre-live burn-in: stable after {_burnin_iter} iters "
                    f"(last {_BURNIN_WINDOW}: min={_min_ms:.1f}ms "
                    f"max={_max_ms:.1f}ms ratio={_max_ms/_min_ms:.2f})"
                )
                break
        # Hard cap: prevent indefinite hang under GPU scheduling jitter.
        if _burnin_iter >= 200:
            _probe(
                f"pre-live burn-in: cap hit at {_burnin_iter} iters without convergence; proceeding"
            )
            break

    # Readiness signal for orchestration.  Manual ordering (start isaac, then
    # gear-sonic only after isaac is warm) was confirmed to work; this file
    # turns that into an automatable handshake.  External tooling (compose
    # healthcheck, wrapper script) gates gear-sonic startup on this file so
    # gear-sonic's DDS subscriber comes online to a steady-state stream and
    # never opens a watchdog gap during its own init.
    try:
        with open("/tmp/isaac_ready", "w") as _ready_f:
            _ready_f.write("ready\n")
        _probe("readiness file written: /tmp/isaac_ready")
    except OSError as _e:
        _probe(f"WARNING: failed to write /tmp/isaac_ready: {_e}")

    # Reset-sim listener: receives a one-byte PUSH from control.py reset-sim,
    # teleports the robot back to spawn, and rearmed the startup-support wrench.
    import zmq as _zmq
    _reset_zmq_ctx = _zmq.Context.instance()
    _reset_zmq_sock = _reset_zmq_ctx.socket(_zmq.PULL)
    _reset_zmq_sock.setsockopt(_zmq.LINGER, 0)
    _reset_zmq_sock.bind("tcp://*:5559")
    _reset_zmq_poller = _zmq.Poller()
    _reset_zmq_poller.register(_reset_zmq_sock, _zmq.POLLIN)
    _probe("reset-sim listener bound on tcp://*:5559")

    _probe(f"DDS bridge + support ready; entering loop; app.is_running={simulation_app.is_running()}")

    env_step = 0
    import math as _math
    import time as _time
    _rate_t0 = _time.perf_counter()
    _rate_last_step = 0
    _cmd_recv_since_log = 0
    _cmd_miss_since_log = 0
    import time as _time2
    try:
        while simulation_app.is_running():
            _iter_t0 = _time2.perf_counter()

            # Non-blocking poll: handle reset-sim signal from control.py reset-sim
            if dict(_reset_zmq_poller.poll(0)):
                try:
                    _reset_zmq_sock.recv(flags=_zmq.NOBLOCK)
                except _zmq.Again:
                    pass
                else:
                    _probe("[g1_sim] reset-sim signal received — teleporting robot and rearming support")
                    _reset_robot(scene)
                    # Flush the teleport writes to PhysX so root_state_w reflects
                    # the new spawn pose before rearm() reads it for the support target.
                    scene.write_data_to_sim()
                    sim.step(render=False)
                    scene.update(dt=SIM_DT)
                    support.rearm()

            if env_step < 3:
                _probe(f"main-loop iter start env_step={env_step}")
            if env_step > 0 and env_step % 50 == 0:
                _now = _time.perf_counter()
                _dt = _now - _rate_t0
                _hz = (env_step - _rate_last_step) / _dt if _dt > 0 else 0.0
                _rtf = _hz / 50.0
                _probe(f"loop rate: env_step={env_step} hz={_hz:.1f} rtf={_rtf:.2f}")
                _rate_t0 = _now
                _rate_last_step = env_step
            if env_step > 0 and env_step % 25 == 0:
                _root = robot.data.root_state_w[0]
                _root_z = _root[2].item()
                _qw, _qx, _qy, _qz = (
                    _root[3].item(), _root[4].item(),
                    _root[5].item(), _root[6].item(),
                )
                # body-Z in world frame: z-component = 1 - 2*(x^2 + y^2)
                _bz = max(-1.0, min(1.0, 1.0 - 2.0 * (_qx * _qx + _qy * _qy)))
                _tilt_deg = _math.degrees(_math.acos(_bz))
                _vx, _vy, _vz = _root[7].item(), _root[8].item(), _root[9].item()
                _v = _math.sqrt(_vx * _vx + _vy * _vy + _vz * _vz)
                _rel = "RELEASED" if support.released else "SUPPORT"
                _probe(
                    f"state [{_rel}] env_step={env_step} root_z={_root_z:.3f} "
                    f"tilt_deg={_tilt_deg:.1f} |v|={_v:.2f} "
                    f"cmd_recv={_cmd_recv_since_log}/{_cmd_recv_since_log + _cmd_miss_since_log}"
                )
                _cmd_recv_since_log = 0
                _cmd_miss_since_log = 0
            q_hw, dq_hw = bridge.read_cmd(device=robot.device)
            if q_hw is not None:
                _cmd_recv_since_log += 1
            else:
                _cmd_miss_since_log += 1
            # While the startup-support wrench is holding the floating base,
            # forwarding gear-sonic's CONTROL-state policy output to the joints
            # causes the arms/legs to thrash against the pinned body (the
            # policy's balance assumptions don't match a body suspended in
            # mid-air).  Hold at default_q until release — this mirrors the
            # real-robot sequence where the operator physically stabilizes the
            # robot until the policy has produced a steady pose, then lets go.
            # While support is active, hold default_q regardless of what the
            # policy outputs — the balance assumptions don't match a suspended
            # body, and a future accel-fix regression must not cause thrashing.
            # Once released, forward policy output at full authority (matches
            # the gear-sonic MuJoCo reference which does not mask or blend).
            # PhysX implicit PD: τ = stiffness·(q_tgt − q) + damping·(dq_tgt − dq).
            # Forwarding only q_tgt leaves dq_tgt=0, so damping always pulls each
            # joint toward zero velocity — which fights policy-commanded swing
            # velocities during walking and traps the robot in a quasi-static
            # standing pose. Forward dq_tgt from rt/lowcmd alongside q_tgt.
            if q_hw is not None:
                q_isaac = q_hw[mujoco_to_isaac].unsqueeze(0)
                dq_isaac = dq_hw[mujoco_to_isaac].unsqueeze(0)
                if support.released:
                    q_target = q_isaac
                    dq_target = dq_isaac
                else:
                    q_target = _default_q
                    dq_target = _zero_dq
                robot.set_joint_position_target(q_target, joint_ids=body_idx)
                robot.set_joint_velocity_target(dq_target, joint_ids=body_idx)
            else:
                robot.set_joint_position_target(_default_q, joint_ids=body_idx)
                robot.set_joint_velocity_target(_zero_dq, joint_ids=body_idx)
            # --- Previous support-gated blend (kept for reference) -------------
            # if q_hw is not None:
            #     q_isaac = q_hw[mujoco_to_isaac].unsqueeze(0)
            #     alpha = 1.0 - support.support_scale
            #     if alpha >= 1.0:
            #         q_target = q_isaac
            #     elif alpha <= 0.0:
            #         q_target = _default_q
            #     else:
            #         q_target = _default_q + alpha * (q_isaac - _default_q)
            #     robot.set_joint_position_target(q_target, joint_ids=body_idx)
            # else:
            #     robot.set_joint_position_target(_default_q, joint_ids=body_idx)

            for sub in range(DECIMATION):
                _t_a = _time2.perf_counter()
                if not support.released:
                    support.apply()
                _t_b = _time2.perf_counter()
                scene.write_data_to_sim()
                _t_c = _time2.perf_counter()
                sim.step(render=False)
                _t_d = _time2.perf_counter()
                scene.update(dt=SIM_DT)
                _t_e = _time2.perf_counter()
                _sub_step_counter = env_step * DECIMATION + sub
                if _sub_step_counter % publish_every_n_sub_steps == 0:
                    bridge.publish_state(
                        robot, body_idx, root_body_idx, torso_body_idx,
                        sim_time=sim.current_time,
                    )
                _t_f = _time2.perf_counter()
                if env_step < 3:
                    _probe(
                        f"  sub={sub} apply={(_t_b-_t_a)*1e3:.1f}ms "
                        f"write={(_t_c-_t_b)*1e3:.1f}ms step={(_t_d-_t_c)*1e3:.1f}ms "
                        f"update={(_t_e-_t_d)*1e3:.1f}ms pub={(_t_f-_t_e)*1e3:.1f}ms"
                    )

            support.check_and_release()
            env_step += 1
            if env_step <= 3:
                _probe(f"main-loop iter end env_step={env_step - 1} total={(_time2.perf_counter()-_iter_t0)*1e3:.1f}ms")
            # Camera requires a render pass; lidar (RayCaster) does not.
            # Fold GUI render into the camera render when they coincide to
            # avoid double render calls on the same step.
            _render_for_camera = not no_render and env_step % CAMERA_PUBLISH_EVERY_N_STEPS == 0
            _render_for_gui = should_render and env_step % render_every == 0
            if _render_for_camera or _render_for_gui:
                sim.render()
                if _render_for_camera:
                    bridge.publish_camera(sim_time=sim.current_time)
            if env_step % LIDAR_PUBLISH_EVERY_N_STEPS == 0:
                bridge.publish_lidar(ray_caster_sensor, sim_time=sim.current_time)
            if sim.is_stopped():
                print("[g1_sim] sim stopped")
                break
    finally:
        bridge.stop()
