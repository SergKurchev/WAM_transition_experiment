#!/usr/bin/env python3
"""Add wrist camera mount Xform frames to the G1 USD asset.

Why a separate script (not patch_mws_dimos.py)?
------------------------------------------------
patch_mws_dimos.py does text-based replacement in .py files.
The G1 USD (g1_29dof.usd) is a binary USDC file — requires pxr Python API.

Why add Xforms to the USD (not create them at runtime in g1_sim.py)?
--------------------------------------------------------------------
Isaac Sim 5.x (Fabric) does NOT propagate physics rigid body transforms
to dynamically-created child prims.  Cameras created as DIRECT children
of physics bodies (left_wrist_yaw_link, right_wrist_yaw_link) stay at a
fixed world position instead of following the articulation.

The head camera works because `d435_link` and `d435_optical_frame` are
NON-PHYSICS Xform prims defined IN THE G1 USD, sitting between torso_link
(physics body) and the camera prim.  This chain propagates correctly.

We mirror the same pattern for wrist cameras:
  left_wrist_yaw_link  [physics, from USD]
    └── left_wrist_cam_frame  [Xform, added by this script]
         └── camera_left_wrist  [Camera, added by CameraCfg in g1_sim.py Patch 4]

Idempotent — safe to run multiple times.
Called from deploy.sh step 2b, before docker compose up.

Usage:
    python3 scripts/patch_g1_usd.py [--dry-run]
"""
from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

try:
    from pxr import Usd, UsdGeom
except ImportError:
    print("ERROR: pxr (usd-core) not found.  Run: pip install usd-core", file=sys.stderr)
    sys.exit(1)

GREEN  = "\033[32m"
YELLOW = "\033[33m"
RED    = "\033[31m"
RESET  = "\033[0m"

USD_PATH = Path("/root/skurchev/workspace/mws-dimos/assets/robots/g1/g1_29dof.usd")

# Non-physics Xform prims to create.  These sit between the physics body
# (wrist_yaw_link) and the camera prim created by CameraCfg.
WRIST_CAM_FRAMES = [
    "/g1/left_wrist_yaw_link/left_wrist_cam_frame",
    "/g1/right_wrist_yaw_link/right_wrist_cam_frame",
]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Add wrist camera Xform frames to G1 USD."
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would change without writing.")
    args = parser.parse_args()

    print(f"\n{'═'*60}")
    print("patch_g1_usd — wrist camera mount frames")
    print(f"{'═'*60}")
    print(f"USD: {USD_PATH}")

    if not USD_PATH.exists():
        print(f"{RED}ERROR: USD not found: {USD_PATH}{RESET}", file=sys.stderr)
        return 1

    stage = Usd.Stage.Open(str(USD_PATH))
    if stage is None:
        print(f"{RED}ERROR: Failed to open USD stage.{RESET}", file=sys.stderr)
        return 1

    changes: list[str] = []
    for frame_path in WRIST_CAM_FRAMES:
        prim = stage.GetPrimAtPath(frame_path)
        if prim.IsValid():
            print(f"{GREEN}  ✓ Already exists — skipping: {frame_path}{RESET}")
        else:
            changes.append(frame_path)
            print(f"  + Would add: {frame_path}")

    if not changes:
        print(f"\n{GREEN}G1 USD already patched — no changes needed.{RESET}")
        return 0

    if args.dry_run:
        print(f"\n{YELLOW}DRY RUN — no files written.{RESET}")
        return 0

    # Backup before writing
    backup = USD_PATH.with_suffix(f".usd.wam-bak.{int(time.time())}")
    shutil.copy2(USD_PATH, backup)
    print(f"\n  Backup: {backup.name}")

    for frame_path in changes:
        # Verify parent exists (wrist_yaw_link must be in the USD)
        parent_path = "/".join(frame_path.split("/")[:-1])
        parent = stage.GetPrimAtPath(parent_path)
        if not parent.IsValid():
            print(f"{RED}  ✗ Parent not found: {parent_path}{RESET}", file=sys.stderr)
            return 1
        UsdGeom.Xform.Define(stage, frame_path)
        print(f"{GREEN}  ✓ Added Xform: {frame_path}{RESET}")

    stage.GetRootLayer().Save()
    print(f"\n{GREEN}G1 USD saved.  {len(changes)} frame(s) added.{RESET}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
