#!/usr/bin/env python3
"""Apply scene object positions and physics from scene_config.yaml to mws_office.usd.

Idempotent — safe to run multiple times.
Run automatically by deploy.sh before docker compose up.

Usage:
    python3 scripts/patch_scene.py [--config path/to/scene_config.yaml]
"""

import argparse
import os
import sys
import tempfile
import zipfile

try:
    import yaml
except ImportError:
    print("ERROR: pyyaml not found. Run: pip install pyyaml", file=sys.stderr)
    sys.exit(1)

try:
    from pxr import Usd, UsdGeom, UsdPhysics, Gf
except ImportError:
    print("ERROR: pxr (usd-core) not found. Run: pip install usd-core", file=sys.stderr)
    sys.exit(1)

GREEN  = "\033[0;32m"
YELLOW = "\033[1;33m"
RED    = "\033[0;31m"
RESET  = "\033[0m"

def ok(msg):   print(f"{GREEN}  ✓ {msg}{RESET}")
def warn(msg): print(f"{YELLOW}  ○ {msg}{RESET}")
def err(msg):  print(f"{RED}  ✗ {msg}{RESET}", file=sys.stderr)


def apply_object(stage, obj_cfg):
    path = obj_cfg["path"]
    translate = obj_cfg.get("translate")
    physics_cfg = obj_cfg.get("physics")

    prim = stage.GetPrimAtPath(path)
    if not prim.IsValid():
        err(f"prim not found: {path}")
        return False

    # ── position ──────────────────────────────────────────────────────────────
    if translate is not None:
        attr = prim.GetAttribute("xformOp:translate")
        if not attr.IsValid():
            xform = UsdGeom.Xformable(prim)
            attr = xform.AddTranslateOp()
        current = attr.Get()
        new_val = Gf.Vec3d(*translate)
        if current != new_val:
            attr.Set(new_val)
            ok(f"{path}  translate {list(current)} → {translate}")
        else:
            warn(f"{path}  translate already {translate}")

    # ── physics ───────────────────────────────────────────────────────────────
    if not physics_cfg:
        return True

    if physics_cfg.get("rigid_body"):
        if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
            UsdPhysics.RigidBodyAPI.Apply(prim)
            ok(f"{path}  RigidBodyAPI applied")
        rb = UsdPhysics.RigidBodyAPI(prim)
        rb.CreateRigidBodyEnabledAttr(True)
        rb.CreateKinematicEnabledAttr(False)

        mass_kg = physics_cfg.get("mass_kg")
        if mass_kg is not None:
            if not prim.HasAPI(UsdPhysics.MassAPI):
                UsdPhysics.MassAPI.Apply(prim)
            UsdPhysics.MassAPI(prim).CreateMassAttr(float(mass_kg))

    # static_collision: just add CollisionAPI to the prim itself (no rigid body)
    if physics_cfg.get("static_collision"):
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            UsdPhysics.CollisionAPI.Apply(prim)
            ok(f"{path}  CollisionAPI applied (static)")
        else:
            warn(f"{path}  CollisionAPI already present (static)")
        return True

    approx = physics_cfg.get("approximation", "convexHull")

    # If a specific mesh child is named, use it; otherwise apply to all Mesh prims under path
    collision_mesh_path = physics_cfg.get("collision_mesh")
    if collision_mesh_path:
        mesh_prims = [stage.GetPrimAtPath(collision_mesh_path)]
    else:
        mesh_prims = [
            p for p in stage.Traverse()
            if str(p.GetPath()).startswith(path) and p.GetTypeName() == "Mesh"
        ]

    # Remove stale CollisionAPI from the xform root (wrong layer for dynamic bodies)
    if prim.HasAPI(UsdPhysics.CollisionAPI):
        prim.RemoveAPI(UsdPhysics.CollisionAPI)
        warn(f"{path}  removed stale CollisionAPI from xform")

    for mesh_prim in mesh_prims:
        if not mesh_prim.IsValid():
            err(f"  collision mesh not found: {mesh_prim}")
            continue
        if not mesh_prim.HasAPI(UsdPhysics.CollisionAPI):
            UsdPhysics.CollisionAPI.Apply(mesh_prim)
        mc = UsdPhysics.MeshCollisionAPI.Apply(mesh_prim)
        mc.CreateApproximationAttr(approx)

    ok(f"{path}  collision({approx}) on {len(mesh_prims)} mesh(es)")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=os.path.join(os.path.dirname(__file__), "..", "scene_config.yaml"),
    )
    parser.add_argument(
        "--restart", action="store_true",
        help="restart sim-isaac container after patching (avoids 'fetch' prompt)",
    )
    args = parser.parse_args()

    config_path = os.path.abspath(args.config)
    if not os.path.exists(config_path):
        err(f"config not found: {config_path}")
        sys.exit(1)

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    usdz_path = cfg["usdz_path"]

    if not os.path.exists(usdz_path):
        err(f"USDZ not found: {usdz_path}")
        sys.exit(1)

    print(f"[patch_scene] config: {config_path}")
    print(f"[patch_scene] USDZ:   {usdz_path}")

    # Extract USD from USDZ into a temp file, edit, repack, delete temp
    with tempfile.TemporaryDirectory() as tmpdir:
        # Unpack
        with zipfile.ZipFile(usdz_path, "r") as z:
            names = z.namelist()
            usd_name = next(n for n in names if n.endswith(".usd") or n.endswith(".usdc"))
            z.extractall(tmpdir)
        usd_tmp = os.path.join(tmpdir, usd_name)

        stage = Usd.Stage.Open(usd_tmp)

        # Ensure PhysicsScene with gravity
        ps_prim = stage.GetPrimAtPath("/scene/physicsScene")
        if not ps_prim.IsValid():
            ps = UsdPhysics.Scene.Define(stage, "/scene/physicsScene")
            ok("PhysicsScene created")
        else:
            ps = UsdPhysics.Scene.Get(stage, "/scene/physicsScene")
        ps.CreateGravityDirectionAttr(Gf.Vec3f(0, -1, 0))
        ps.CreateGravityMagnitudeAttr(981.0)   # cm/s² (scene units are cm)

        errors = 0
        for obj_cfg in cfg.get("objects", []):
            if not apply_object(stage, obj_cfg):
                errors += 1

        stage.Save()

        # Repack USDZ (temp file disappears automatically after this block)
        with zipfile.ZipFile(usdz_path, "w", zipfile.ZIP_STORED) as z:
            z.write(usd_tmp, usd_name)

    size_mb = os.path.getsize(usdz_path) / 1024 / 1024
    ok(f"Repacked {usdz_path}  ({size_mb:.1f} MB)  [no loose .usd left on disk]")

    if errors:
        err(f"{errors} error(s) — check output above")
        sys.exit(1)

    print(f"[patch_scene] done  errors={errors}")

    if args.restart:
        import subprocess
        compose_file = os.path.join(os.path.dirname(__file__), "..", "compose.yml")
        print("[patch_scene] restarting sim-isaac...")
        subprocess.run(
            ["docker", "compose", "-f", os.path.abspath(compose_file),
             "restart", "sim-isaac"],
            check=True,
        )
        ok("sim-isaac restarted — no fetch prompt")


if __name__ == "__main__":
    main()
