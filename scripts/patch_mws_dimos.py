#!/usr/bin/env python3
"""Apply WAM-specific patches to external mws-dimos code.

Usage
-----
    python3 scripts/patch_mws_dimos.py [--dry-run] [--verbose]

Options
-------
--dry-run   Print what would change without writing anything.
--verbose   Show full diff for each patch.

Why a script instead of a git patch?
-------------------------------------
mws-dimos is maintained by the MWS team and must stay on
``feat/real-transfer`` without local commits.  A patch script is:
  - Idempotent (safe to run multiple times; skips already-applied patches).
  - Self-documenting (each PATCH entry explains the rationale).
  - Integrated into deploy.sh so it runs automatically before every
    ``docker compose up``.

All patches are documented in detail in ``wam-stack/PATCHES.md``.

Adding a new patch
------------------
Append an entry to the PATCHES list below.  Required keys:
  file        Absolute path on the server.
  description Short one-line description.
  sentinel    A string that appears only in the patched version (for idempotency).
  old         The exact substring to replace (use raw strings r"..." for backslashes).
  new         The replacement substring.
"""
from __future__ import annotations

import argparse
import difflib
import shutil
import sys
import time
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# Patch definitions
# Each entry is applied in order.  All are idempotent.
# ─────────────────────────────────────────────────────────────────────────────

PATCHES: list[dict] = [

    # ── Patch 1: arm commands in SUPPORT mode ────────────────────────────────
    {
        "file": "/root/skurchev/workspace/mws-dimos/sim/isaac/g1_sim.py",
        "description": (
            "Allow arm joint commands from rt/lowcmd in Isaac SUPPORT mode "
            "(WAM without GEAR-SONIC WBC)"
        ),
        # Sentinel: unique string present ONLY after patch is applied.
        "sentinel": "# ── WAM-patch: arm-in-support-mode",
        # ── old ──────────────────────────────────────────────────────────────
        # g1_sim.py normally ignores rt/lowcmd while the startup-support wrench
        # is active, holding all joints at the default standing pose.  This is
        # fine for GEAR-SONIC (which needs to ramp up after release) but blocks
        # WAM from actuating the arms at all.
        "old": """\
                else:
                    q_target = _default_q
                    dq_target = _zero_dq""",
        # ── new ──────────────────────────────────────────────────────────────
        # In SUPPORT mode the external wrench holds the robot root; the joints
        # are still fully actuated by the Isaac Lab articulation PD controller.
        # WAM patch: keep leg/waist joints at _default_q (safe standing pose)
        # but apply arm targets from rt/lowcmd.
        #
        # Joint-space remapping:
        #   Hardware/MuJoCo ordering (used in rt/lowcmd):
        #     DOF  0-13 → legs + waist
        #     DOF 14-27 → left arm (14-20) + right arm (21-27)
        #     DOF 28    → gripper
        #   Isaac Lab ordering (used by set_joint_position_target):
        #     mujoco_to_isaac[14:28] gives the corresponding IL indices.
        "new": """\
                else:
                    # ── WAM-patch: arm-in-support-mode ───────────────────────
                    # Applied by wam-stack/scripts/patch_mws_dimos.py
                    # Documented in wam-stack/PATCHES.md (Patch 1).
                    # Keep legs/waist at default; apply arm targets from lowcmd.
                    _arm_mujoco_idx = torch.arange(14, 28, dtype=torch.long,
                                                   device=q_hw.device)
                    _arm_isaac_idx = mujoco_to_isaac[_arm_mujoco_idx]
                    q_target = _default_q.clone()
                    q_target[0, _arm_isaac_idx] = q_hw[_arm_mujoco_idx]
                    dq_target = _zero_dq
                    # ─────────────────────────────────────────────────────────""",
    },

    # ── Patch 3: kinematic robot root — follows WAM commands exactly, no physics fall ──
    # Controlled by env var G1_KINEMATIC_ROBOT=1 in compose.yml (sim-isaac).
    # fix_root_link=True anchors the robot pelvis to its spawn position.
    # The arm/leg joints are still driven by the Isaac Lab PD controller
    # (i.e. WAM joint targets work normally), but gravity/forces cannot
    # push the base — robot never falls regardless of WBC or wrench state.
    # Collision geometry remains active so the robot can interact with
    # physics objects (cube, storage box).
    {
        "file": "/root/skurchev/workspace/mws-dimos/sim/isaac/g1_sim.py",
        "description": (
            "Kinematic robot root: fix_root_link when G1_KINEMATIC_ROBOT=1 "
            "(robot follows WAM commands exactly, never falls)"
        ),
        "sentinel": "# WAM-patch: kinematic-robot",
        "old": (
            "    cfg.spawn.articulation_props.solver_position_iteration_count = 2\n"
            "    cfg.spawn.articulation_props.solver_velocity_iteration_count = 0"
        ),
        "new": (
            "    cfg.spawn.articulation_props.solver_position_iteration_count = 2\n"
            "    cfg.spawn.articulation_props.solver_velocity_iteration_count = 0\n"
            "    if os.environ.get(\"G1_KINEMATIC_ROBOT\", \"0\") == \"1\":  # WAM-patch: kinematic-robot\n"
            "        cfg.spawn.articulation_props.fix_root_link = True\n"
            "        # Robot pelvis anchored to spawn; joints driven by WAM targets.\n"
            "        # Collision geometry active → can interact with physics objects.\n"
            "        # Applied by wam-stack/scripts/patch_mws_dimos.py (Patch 3)."
        ),
    },

    # ── Patch 2: unifolm attention.py — remove xformers assertion ────────────
    {
        "file": (
            "/root/skurchev/workspace/wam-stack/repos/unifolm"
            "/src/unifolm_wma/modules/attention.py"
        ),
        "description": (
            "Remove xformers assertion that blocks vanilla attention "
            "(container has PyTorch 2.7/cu126, xformers requires cu128)"
        ),
        "sentinel": "# ── WAM-patch: xformers-assert-removed",
        "old": 'assert 1 > 2, ">>> ERROR: should setup xformers"',
        "new": (
            "# ── WAM-patch: xformers-assert-removed ─────────────────────────\n"
            "                # Applied by wam-stack/scripts/patch_mws_dimos.py\n"
            "                # Documented in wam-stack/PATCHES.md (Patch 2).\n"
            "                # xformers built for cu128; container uses cu126.\n"
            "                # Vanilla PyTorch attention (below) is fully functional.\n"
            "                # ─────────────────────────────────────────────────"
        ),
        # attention.py was manually patched before this script existed —
        # the assert was deleted directly without leaving a sentinel.
        # old_missing_ok lets the script treat that case as already-done.
        "old_missing_ok": True,
    },

    # ── Patch 4: wrist camera CameraCfg fields in G1SceneCfg ────────────────
    # Adds two CameraCfg sensors (left_wrist_yaw_link / right_wrist_yaw_link)
    # to G1SceneCfg so Isaac Lab spawns camera prims on both wrists.
    # These cameras are consumed by Patches 5-7 and published to SHM by Patch 9.
    {
        "file": "/root/skurchev/workspace/mws-dimos/sim/isaac/g1_sim.py",
        "description": "Add camera_left_wrist and camera_right_wrist CameraCfg to G1SceneCfg",
        "sentinel": "# ── WAM-patch: wrist-cameras-cfg-v3",
        "old": (
            "        offset=CameraCfg.OffsetCfg(),\n"
            "    )\n"
            "    # Unitree G1 uses a Livox MID-360 at the head."
        ),
        "new": (
            "        offset=CameraCfg.OffsetCfg(),\n"
            "    )\n"
            "    # ── WAM-patch: wrist-cameras-cfg-v3 ──────────────────────────────\n"
            "    # Applied by wam-stack/scripts/patch_mws_dimos.py (Patch 4).\n"
            "    # Wrist cameras for LingBot-VA 3-camera observation.\n"
            "    #\n"
            "    # Derived from diagnostic captures (30 May 2026):\n"
            "    #   identity rot=(1,0,0,0)  → camera -Z = parent -Z = world DOWN\n"
            "    #     → grey ground plane visible  → camera IS at wrist position ✓\n"
            "    #   fingers extend along parent +X  (palm_link offset +0.041 X)\n"
            "    #   rot=(0.7071,0,-0.7071,0) = -90° around Y (OpenGL):\n"
            "    #     camera -Z  →  parent +X  →  toward fingers / workspace\n"
            "    #   pos=(0.06,0,0) pushes 6 cm along arm axis away from wrist centre.\n"
            "    camera_left_wrist: CameraCfg = CameraCfg(\n"
            "        prim_path=\"/World/envs/env_.*/Robot/left_wrist_yaw_link/camera_left_wrist\",\n"
            "        update_period=1.0 / _D435I_FPS,\n"
            "        height=_D435I_HEIGHT,\n"
            "        width=_D435I_WIDTH,\n"
            "        data_types=[\"rgb\"],\n"
            "        spawn=PinholeCameraCfg(\n"
            "            focal_length=_CAMERA_FOCAL_LENGTH_MM,\n"
            "            focus_distance=400.0,\n"
            "            horizontal_aperture=_pinhole_horizontal_aperture(\n"
            "                _CAMERA_FOCAL_LENGTH_MM,\n"
            "                _D435I_HORIZONTAL_FOV_DEG,\n"
            "            ),\n"
            "            clipping_range=(0.02, 5.0),\n"
            "        ),\n"
            "        offset=CameraCfg.OffsetCfg(\n"
            "            pos=(0.06, 0.0, 0.0),\n"
            "            rot=(0.7071, 0.0, -0.7071, 0.0),\n"
            "        ),\n"
            "    )\n"
            "    camera_right_wrist: CameraCfg = CameraCfg(\n"
            "        prim_path=\"/World/envs/env_.*/Robot/right_wrist_yaw_link/camera_right_wrist\",\n"
            "        update_period=1.0 / _D435I_FPS,\n"
            "        height=_D435I_HEIGHT,\n"
            "        width=_D435I_WIDTH,\n"
            "        data_types=[\"rgb\"],\n"
            "        spawn=PinholeCameraCfg(\n"
            "            focal_length=_CAMERA_FOCAL_LENGTH_MM,\n"
            "            focus_distance=400.0,\n"
            "            horizontal_aperture=_pinhole_horizontal_aperture(\n"
            "                _CAMERA_FOCAL_LENGTH_MM,\n"
            "                _D435I_HORIZONTAL_FOV_DEG,\n"
            "            ),\n"
            "            clipping_range=(0.02, 5.0),\n"
            "        ),\n"
            "        offset=CameraCfg.OffsetCfg(\n"
            "            pos=(0.06, 0.0, 0.0),\n"
            "            rot=(0.7071, 0.0, -0.7071, 0.0),\n"
            "        ),\n"
            "    )\n"
            "    # ─────────────────────────────────────────────────────────────────\n"
            "    # Unitree G1 uses a Livox MID-360 at the head."
        ),
    },

    # ── Patch 5: acquire wrist sensors in run() ──────────────────────────────
    {
        "file": "/root/skurchev/workspace/mws-dimos/sim/isaac/g1_sim.py",
        "description": "Acquire camera_left_wrist / camera_right_wrist from scene.sensors",
        "sentinel": "# ── WAM-patch: wrist-cameras-acquire",
        "old": (
            "    camera_sensor = scene.sensors[\"camera\"]\n"
            "    ray_caster_sensor = scene.sensors[\"ray_caster\"]\n"
            "    _probe(\"camera and ray_caster sensors acquired\")"
        ),
        "new": (
            "    camera_sensor = scene.sensors[\"camera\"]\n"
            "    ray_caster_sensor = scene.sensors[\"ray_caster\"]\n"
            "    # ── WAM-patch: wrist-cameras-acquire ─────────────────────────────\n"
            "    left_wrist_cam = scene.sensors.get(\"camera_left_wrist\")\n"
            "    right_wrist_cam = scene.sensors.get(\"camera_right_wrist\")\n"
            "    # ─────────────────────────────────────────────────────────────────\n"
            "    _probe(\"camera and ray_caster sensors acquired\")"
        ),
    },

    # ── Patch 6: configure wrist cameras in bridge setup ─────────────────────
    {
        "file": "/root/skurchev/workspace/mws-dimos/sim/isaac/g1_sim.py",
        "description": "Call bridge.configure_wrist_cameras_export() after head cam configure",
        "sentinel": "# ── WAM-patch: wrist-cameras-configure",
        "old": (
            "    bridge.configure_camera_export(camera_sensor)\n"
            "    _probe(\"creating IsaacStartupSupport\")"
        ),
        "new": (
            "    bridge.configure_camera_export(camera_sensor)\n"
            "    # ── WAM-patch: wrist-cameras-configure ───────────────────────────\n"
            "    if left_wrist_cam is not None and right_wrist_cam is not None:\n"
            "        bridge.configure_wrist_cameras_export(left_wrist_cam, right_wrist_cam)\n"
            "    # ─────────────────────────────────────────────────────────────────\n"
            "    _probe(\"creating IsaacStartupSupport\")"
        ),
    },

    # ── Patch 7: publish wrist cameras in render loop ────────────────────────
    {
        "file": "/root/skurchev/workspace/mws-dimos/sim/isaac/g1_sim.py",
        "description": "Call bridge.publish_wrist_cameras() alongside publish_camera()",
        "sentinel": "# ── WAM-patch: wrist-cameras-publish",
        "old": (
            "                if _render_for_camera:\n"
            "                    bridge.publish_camera(sim_time=sim.current_time)"
        ),
        "new": (
            "                if _render_for_camera:\n"
            "                    bridge.publish_camera(sim_time=sim.current_time)\n"
            "                    # ── WAM-patch: wrist-cameras-publish ─────────────\n"
            "                    bridge.publish_wrist_cameras(sim_time=sim.current_time)\n"
            "                    # ─────────────────────────────────────────────────"
        ),
    },

    # ── Patch 8: wrist SHM constants in dds_bridge.py ────────────────────────
    {
        "file": "/root/skurchev/workspace/mws-dimos/sim/isaac/dds_bridge.py",
        "description": "Add _CAMERA_LEFT/RIGHT_WRIST_SHM_PATH constants to dds_bridge.py",
        "sentinel": "# ── WAM-patch: wrist-cameras-shm-constants",
        "old": (
            "_CAMERA_SHM_SIZE: int = 1280 * 720 * 3  # D435i rgb8 at configured camera resolution\n"
            "_LIDAR_SHM_PATH: str = \"/run/mws/lidar.xyzi\""
        ),
        "new": (
            "_CAMERA_SHM_SIZE: int = 1280 * 720 * 3  # D435i rgb8 at configured camera resolution\n"
            "# ── WAM-patch: wrist-cameras-shm-constants ──────────────────────────────\n"
            "# Applied by wam-stack/scripts/patch_mws_dimos.py (Patch 8).\n"
            "_CAMERA_LEFT_WRIST_SHM_PATH: str = \"/run/mws/camera_left_wrist.rgb\"\n"
            "_CAMERA_RIGHT_WRIST_SHM_PATH: str = \"/run/mws/camera_right_wrist.rgb\"\n"
            "_CAMERA_WRIST_SHM_SIZE: int = 1280 * 720 * 3\n"
            "# ─────────────────────────────────────────────────────────────────────────\n"
            "_LIDAR_SHM_PATH: str = \"/run/mws/lidar.xyzi\""
        ),
    },

    # ── Patch 9: wrist SHM members in __init__ ───────────────────────────────
    {
        "file": "/root/skurchev/workspace/mws-dimos/sim/isaac/dds_bridge.py",
        "description": "Add wrist SHM and annotator members to IsaacDdsBridge.__init__",
        "sentinel": "# ── WAM-patch: wrist-cameras-members",
        "old": (
            "        self._lidar_shm: Optional[mmap.mmap] = None\n"
            "        self._lidar_signal_seq: int = 0"
        ),
        "new": (
            "        self._lidar_shm: Optional[mmap.mmap] = None\n"
            "        # ── WAM-patch: wrist-cameras-members ─────────────────────────\n"
            "        self._left_wrist_shm: Optional[mmap.mmap] = None\n"
            "        self._right_wrist_shm: Optional[mmap.mmap] = None\n"
            "        self._left_wrist_annotator: Any | None = None\n"
            "        self._right_wrist_annotator: Any | None = None\n"
            "        self._left_wrist_render_product_path: str | None = None\n"
            "        self._right_wrist_render_product_path: str | None = None\n"
            "        # ─────────────────────────────────────────────────────────────\n"
            "        self._lidar_signal_seq: int = 0"
        ),
    },

    # ── Patch 10: initialize wrist SHMs in start() ───────────────────────────
    {
        "file": "/root/skurchev/workspace/mws-dimos/sim/isaac/dds_bridge.py",
        "description": "Open wrist SHM files in IsaacDdsBridge.start()",
        "sentinel": "# ── WAM-patch: wrist-cameras-start",
        "old": (
            "        self._camera_shm = self._open_shm_file(_CAMERA_SHM_PATH, _CAMERA_SHM_SIZE)\n"
            "\n"
            "        self._camera_puber = ChannelPublisher(\"rt/camera/signal\", CameraSignal_)"
        ),
        "new": (
            "        self._camera_shm = self._open_shm_file(_CAMERA_SHM_PATH, _CAMERA_SHM_SIZE)\n"
            "        # ── WAM-patch: wrist-cameras-start ───────────────────────────\n"
            "        self._left_wrist_shm = self._open_shm_file(\n"
            "            _CAMERA_LEFT_WRIST_SHM_PATH, _CAMERA_WRIST_SHM_SIZE)\n"
            "        self._right_wrist_shm = self._open_shm_file(\n"
            "            _CAMERA_RIGHT_WRIST_SHM_PATH, _CAMERA_WRIST_SHM_SIZE)\n"
            "        # ─────────────────────────────────────────────────────────────\n"
            "\n"
            "        self._camera_puber = ChannelPublisher(\"rt/camera/signal\", CameraSignal_)"
        ),
    },

    # ── Patch 11: close wrist SHMs in close() ────────────────────────────────
    {
        "file": "/root/skurchev/workspace/mws-dimos/sim/isaac/dds_bridge.py",
        "description": "Close and unlink wrist SHM files in IsaacDdsBridge.close()",
        "sentinel": "# ── WAM-patch: wrist-cameras-close",
        "old": (
            "        if self._camera_shm is not None:\n"
            "            try:\n"
            "                self._camera_shm.close()\n"
            "            except Exception:\n"
            "                pass\n"
            "            try:\n"
            "                os.unlink(_CAMERA_SHM_PATH)\n"
            "            except OSError:\n"
            "                pass\n"
            "        if self._lidar_shm is not None:"
        ),
        "new": (
            "        if self._camera_shm is not None:\n"
            "            try:\n"
            "                self._camera_shm.close()\n"
            "            except Exception:\n"
            "                pass\n"
            "            try:\n"
            "                os.unlink(_CAMERA_SHM_PATH)\n"
            "            except OSError:\n"
            "                pass\n"
            "        # ── WAM-patch: wrist-cameras-close ───────────────────────────\n"
            "        for _wrist_shm, _wrist_path in [\n"
            "            (self._left_wrist_shm, _CAMERA_LEFT_WRIST_SHM_PATH),\n"
            "            (self._right_wrist_shm, _CAMERA_RIGHT_WRIST_SHM_PATH),\n"
            "        ]:\n"
            "            if _wrist_shm is not None:\n"
            "                try:\n"
            "                    _wrist_shm.close()\n"
            "                except Exception:\n"
            "                    pass\n"
            "                try:\n"
            "                    os.unlink(_wrist_path)\n"
            "                except OSError:\n"
            "                    pass\n"
            "        for _ann, _path in [\n"
            "            (self._left_wrist_annotator, self._left_wrist_render_product_path),\n"
            "            (self._right_wrist_annotator, self._right_wrist_render_product_path),\n"
            "        ]:\n"
            "            if _ann is not None:\n"
            "                try:\n"
            "                    if _path is not None:\n"
            "                        _ann.detach([_path])\n"
            "                    else:\n"
            "                        _ann.detach()\n"
            "                except Exception:\n"
            "                    pass\n"
            "        # ─────────────────────────────────────────────────────────────\n"
            "        if self._lidar_shm is not None:"
        ),
    },

    # ── Patch 12: configure_wrist_cameras_export() and publish_wrist_cameras() ─
    {
        "file": "/root/skurchev/workspace/mws-dimos/sim/isaac/dds_bridge.py",
        "description": (
            "Add configure_wrist_cameras_export() and publish_wrist_cameras() "
            "to IsaacDdsBridge"
        ),
        "sentinel": "# ── WAM-patch: wrist-cameras-methods",
        "old": "    def read_cmd(",
        "new": (
            "    def configure_wrist_cameras_export(\n"
            "        self, left_cam: Any, right_cam: Any\n"
            "    ) -> None:\n"
            "        \"\"\"Attach CPU RGB annotators to left and right wrist camera render products.\n"
            "\n"
            "        # ── WAM-patch: wrist-cameras-methods ─────────────────────────\n"
            "        # Applied by wam-stack/scripts/patch_mws_dimos.py (Patch 12).\n"
            "        \"\"\"\n"
            "        import omni.replicator.core as rep\n"
            "\n"
            "        for cam, attr_ann, attr_path in [\n"
            "            (left_cam,  \"_left_wrist_annotator\",  \"_left_wrist_render_product_path\"),\n"
            "            (right_cam, \"_right_wrist_annotator\", \"_right_wrist_render_product_path\"),\n"
            "        ]:\n"
            "            rpp = list(cam.render_product_paths)\n"
            "            if len(rpp) != 1:\n"
            "                print(f\"[dds_bridge] wrist cam expected 1 render product, got {len(rpp)}\",\n"
            "                      flush=True)\n"
            "                continue\n"
            "            rp = rpp[0]\n"
            "            ann = rep.AnnotatorRegistry.get_annotator(\"rgb\", device=\"cpu\")\n"
            "            ann.attach(rp)\n"
            "            setattr(self, attr_ann, ann)\n"
            "            setattr(self, attr_path, rp)\n"
            "\n"
            "    def _read_wrist_rgb(self, annotator: Any) -> Optional[np.ndarray]:\n"
            "        \"\"\"Read RGB from a wrist camera annotator; returns None on failure.\"\"\"\n"
            "        output = annotator.get_data(device=\"cpu\")\n"
            "        if output is None:\n"
            "            return None\n"
            "        rgb = output[\"data\"] if isinstance(output, dict) else output\n"
            "        if rgb is None:\n"
            "            return None\n"
            "        rgb_np = np.asarray(rgb, dtype=np.uint8)\n"
            "        if rgb_np.size == 0:\n"
            "            return None\n"
            "        if rgb_np.ndim == 4:\n"
            "            rgb_np = rgb_np[0]\n"
            "        if rgb_np.ndim != 3:\n"
            "            return None\n"
            "        return np.ascontiguousarray(rgb_np[..., :3])\n"
            "\n"
            "    def publish_wrist_cameras(self, sim_time: float) -> None:\n"
            "        \"\"\"Write left and right wrist camera RGB to SHM.\n"
            "\n"
            "        No DDS signal is published — WAM reads the SHM directly.\n"
            "        No-op if wrist cameras were not configured.\n"
            "        \"\"\"\n"
            "        for ann, shm in [\n"
            "            (self._left_wrist_annotator,  self._left_wrist_shm),\n"
            "            (self._right_wrist_annotator, self._right_wrist_shm),\n"
            "        ]:\n"
            "            if ann is None or shm is None:\n"
            "                continue\n"
            "            try:\n"
            "                rgb_np = self._read_wrist_rgb(ann)\n"
            "                if rgb_np is None:\n"
            "                    continue\n"
            "                shm.seek(0)\n"
            "                shm.write(rgb_np.tobytes())\n"
            "            except Exception as exc:\n"
            "                print(f\"[dds_bridge] wrist camera publish failed: {exc}\", flush=True)\n"
            "\n"
            "    def read_cmd("
        ),
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# Core logic
# ─────────────────────────────────────────────────────────────────────────────

GREEN  = "\033[32m"
YELLOW = "\033[33m"
RED    = "\033[31m"
BLUE   = "\033[34m"
BOLD   = "\033[1m"
RESET  = "\033[0m"


def apply_patch(patch: dict, *, dry_run: bool = False, verbose: bool = False) -> str:
    """Apply a single patch entry.

    Returns
    -------
    "skipped"   — patch already applied (sentinel found), or manually applied
                  with old_missing_ok=True (target already absent).
    "applied"   — patch applied successfully.
    "error"     — old string not found (file may have changed upstream).

    Optional patch keys
    -------------------
    old_missing_ok : bool (default False)
        If True and the target ``old`` string is not found (and sentinel is
        absent too), treat the file as already manually patched instead of
        reporting an error.  Use when the change was applied by hand before
        the patch script existed.
    """
    path = Path(patch["file"])
    description = patch["description"]
    sentinel = patch["sentinel"]
    old = patch["old"]
    new = patch["new"]
    old_missing_ok = patch.get("old_missing_ok", False)

    print(f"\n{'─'*70}")
    print(f"{BOLD}Patch:{RESET} {description}")
    print(f"File:  {path}")

    if not path.exists():
        print(f"{RED}  ✗ File not found — skipping.{RESET}")
        return "error"

    content = path.read_text(encoding="utf-8")

    # Idempotency check
    if sentinel in content:
        print(f"{GREEN}  ✓ Already applied (sentinel found) — skipping.{RESET}")
        return "skipped"

    if old not in content:
        if old_missing_ok:
            print(f"{YELLOW}  ○ Target string absent and sentinel missing — file was likely{RESET}")
            print(f"{YELLOW}    patched manually before this script existed. Treating as OK.{RESET}")
            return "skipped"
        print(f"{RED}  ✗ Target string not found — file may have changed upstream.{RESET}")
        print(f"    Looking for:\n    {old!r}")
        return "error"

    new_content = content.replace(old, new, 1)

    if verbose:
        diff = difflib.unified_diff(
            content.splitlines(keepends=True),
            new_content.splitlines(keepends=True),
            fromfile=f"a/{path.name}",
            tofile=f"b/{path.name}",
            n=3,
        )
        print("".join(list(diff)[:60]))

    if dry_run:
        print(f"{YELLOW}  ○ DRY RUN — would apply patch (not writing).{RESET}")
        return "applied"

    # Backup
    backup = path.with_suffix(path.suffix + f".wam-bak.{int(time.time())}")
    shutil.copy2(path, backup)
    print(f"  Backup: {backup.name}")

    path.write_text(new_content, encoding="utf-8")
    print(f"{GREEN}  ✓ Patch applied.{RESET}")
    return "applied"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Apply WAM patches to mws-dimos and unifolm external repos."
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would change without writing.")
    parser.add_argument("--verbose", action="store_true",
                        help="Show unified diff for each patch.")
    args = parser.parse_args()

    print(f"\n{BOLD}{'═'*70}")
    print(f"WAM patch script — mws-dimos & unifolm external code")
    print(f"{'═'*70}{RESET}")
    if args.dry_run:
        print(f"{YELLOW}DRY RUN mode — no files will be written.{RESET}")

    results: dict[str, int] = {"applied": 0, "skipped": 0, "error": 0}

    for patch in PATCHES:
        result = apply_patch(patch, dry_run=args.dry_run, verbose=args.verbose)
        results[result] += 1

    print(f"\n{'═'*70}")
    print(
        f"{BOLD}Summary:{RESET}  "
        f"{GREEN}applied={results['applied']}{RESET}  "
        f"{YELLOW}skipped(already done)={results['skipped']}{RESET}  "
        f"{RED}errors={results['error']}{RESET}"
    )

    if results["error"] > 0:
        print(f"\n{RED}Some patches failed — check output above.{RESET}")
        return 1

    print(f"\n{GREEN}All patches OK.{RESET}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
