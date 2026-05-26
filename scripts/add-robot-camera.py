#!/usr/bin/env python3
"""Add camera to mws_office.usdz scene for UnifoLM G1 robot.

Camera placement: Head/chest of G1 robot, looking down at table.
- Position: (0.0, 0.5, 1.5) — above table, centered
- Rotation: Looking down at ~45 degrees
- Resolution: 640x480 (typical for vision models)

Usage (from wam-stack container):
  python3 scripts/add-robot-camera.py
"""

import sys
import os
from pathlib import Path

def add_robot_camera_to_scene(usdz_path: str, output_path: str = None):
    """Add camera to USD scene using pxr API."""
    if output_path is None:
        output_path = usdz_path

    print(f"[CAMERA] Opening scene: {usdz_path}", flush=True)

    try:
        from pxr import Usd, UsdGeom, Gf, Sdf
    except ImportError:
        print("[ERROR] USD Python bindings not available.", flush=True)
        print("        This must run in Isaac Sim container with USD SDK.", flush=True)
        return False

    try:
        # Open or create stage
        stage = Usd.Stage.Open(usdz_path)
        if not stage:
            print(f"[ERROR] Could not open stage: {usdz_path}", flush=True)
            return False

        # Define camera under /World/Cameras
        camera_prim = UsdGeom.Camera.Define(stage, "/World/Cameras/robot_head_camera")

        # Position: above table, centered
        # (Adjust based on scene - table is typically at z=0.7-0.8m, camera ~1.5m high)
        xform = camera_prim.GetPrim()
        xform_op_translate = UsdGeom.Xformable(xform).AddTranslateOp()
        xform_op_translate.Set(Gf.Vec3d(0.0, 0.5, 1.5))

        # Rotation: ~45 degree downward tilt to see table
        xform_op_rotate_x = UsdGeom.Xformable(xform).AddRotateXOp()
        xform_op_rotate_x.Set(45.0)  # degrees

        # Camera intrinsics: 24mm focal length, 640x480 aspect
        camera_prim.GetFocalLengthAttr().Set(24.0)
        camera_prim.GetHorizontalApertureAttr().Set(20.955)
        camera_prim.GetVerticalApertureAttr().Set(11.787)

        print(f"[CAMERA] Camera added:", flush=True)
        print(f"  Path: /World/Cameras/robot_head_camera", flush=True)
        print(f"  Position: (0.0, 0.5, 1.5) m", flush=True)
        print(f"  Tilt: 45° downward", flush=True)
        print(f"  Focal length: 24mm", flush=True)

        # Export (flatten) the scene
        print(f"[CAMERA] Exporting modified scene to: {output_path}", flush=True)
        stage.GetRootLayer().Export(output_path)

        print(f"[CAMERA] ✓ Successfully added camera to scene!", flush=True)
        return True

    except Exception as e:
        print(f"[ERROR] Failed to add camera: {e}", flush=True)
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    scene_path = "/workspace/scene_assets/mws_office.usdz"

    if not os.path.exists(scene_path):
        print(f"[ERROR] Scene not found: {scene_path}", flush=True)
        sys.exit(1)

    success = add_robot_camera_to_scene(scene_path)
    sys.exit(0 if success else 1)
