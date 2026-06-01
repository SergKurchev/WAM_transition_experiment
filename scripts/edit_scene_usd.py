#!/usr/bin/env python3
"""
Edit office_demo.usdz locally without GPU
Add task objects and position robot
"""

import os
import sys
from pathlib import Path

_WS = os.environ.get("SERVER_WORKSPACE", "/root/skurchev/workspace")

try:
    from pxr import Usd, UsdGeom, Gf
    print("✅ pxr.Usd loaded")
except ImportError:
    print("❌ ERROR: pxr not installed")
    print("\nInstall with:")
    print("  pip install usd-core")
    sys.exit(1)

def add_cube(stage, name, position, scale, parent_path="/"):
    """Add a cube mesh to the stage"""

    # Create Xform for positioning
    xform_path = Usd.Sdf.Path(f"{parent_path}{name}")
    xform = UsdGeom.Xform.Define(stage, xform_path)

    # Set transform
    xform.AddTranslateOp().Set(Gf.Vec3f(*position))
    xform.AddScaleOp().Set(Gf.Vec3f(*scale))

    # Create mesh (unit cube)
    mesh_path = Usd.Sdf.Path(f"{parent_path}{name}/mesh")
    mesh = UsdGeom.Mesh.Define(stage, mesh_path)

    # Cube vertices (unit cube centered at origin)
    points = [
        (-0.5, -0.5, -0.5), (0.5, -0.5, -0.5),
        (0.5, 0.5, -0.5), (-0.5, 0.5, -0.5),
        (-0.5, -0.5, 0.5), (0.5, -0.5, 0.5),
        (0.5, 0.5, 0.5), (-0.5, 0.5, 0.5),
    ]
    mesh.GetPointsAttr().Set([Gf.Vec3f(*p) for p in points])

    # Face indices (6 faces, 4 verts each)
    face_indices = [
        0, 1, 2, 3,    # bottom
        4, 7, 6, 5,    # top
        0, 4, 5, 1,    # front
        2, 6, 7, 3,    # back
        0, 3, 7, 4,    # left
        1, 5, 6, 2,    # right
    ]
    mesh.GetFaceVertexIndicesAttr().Set(face_indices)
    mesh.GetFaceVertexCountsAttr().Set([4, 4, 4, 4, 4, 4])

    # Add color material (random color for visibility)
    color_attr = mesh.CreateDisplayColorAttr()
    color_attr.Set([(0.5, 0.5, 0.8)])  # Light blue

    print(f"  ✅ Added cube: {name} at {position}")
    return xform

def edit_office_scene():
    """Main function to edit office scene"""

    print("\n" + "="*60)
    print("🏢 USD Scene Editor (No GPU Required)")
    print("="*60)

    # Paths
    base_path = Path(__file__).parent.parent / "blender_assets"
    input_file = base_path / "office_demo.usdz"
    output_file = base_path / "office_demo_edited.usdz"

    print(f"\n📂 Working directory: {base_path}")
    print(f"📥 Input:  {input_file.name}")
    print(f"📤 Output: {output_file.name}")

    # Check input file exists
    if not input_file.exists():
        print(f"\n❌ ERROR: {input_file} not found")
        return False

    print(f"✅ Input file found ({input_file.stat().st_size / 1e6:.1f}MB)")

    # Open stage
    print("\n📖 Opening stage...")
    stage = Usd.Stage.Open(str(input_file))
    if not stage:
        print("❌ Failed to open stage")
        return False

    print(f"✅ Stage opened (default prim: {stage.GetDefaultPrimPath()})")

    # Get or create World prim
    world_path = Usd.Sdf.Path("/World")
    world = stage.GetPrimAtPath(world_path)
    if not world:
        print("\n📝 Creating World prim...")
        world = UsdGeom.Xform.Define(stage, world_path)
        stage.SetDefaultPrim(world.GetPrim())

    # Add task objects
    print("\n🎯 Adding task objects...")

    add_cube(stage, "Robot", (-2.0, 0.0, 0.0), (0.4, 0.4, 2.0), "/World/")
    add_cube(stage, "Box", (0.5, 0.0, 0.5), (0.5, 0.5, 0.25), "/World/")
    add_cube(stage, "Shelf", (0.5, -1.0, 0.3), (1.0, 2.0, 0.1), "/World/")
    add_cube(stage, "Camera", (1.5, 0.0, 0.8), (0.1, 0.1, 0.1), "/World/")

    # Export
    print(f"\n💾 Exporting to {output_file.name}...")
    if stage.Export(str(output_file)):
        size = output_file.stat().st_size / 1e6
        print(f"✅ Export successful ({size:.1f}MB)")
    else:
        print("❌ Export failed")
        return False

    print("\n" + "="*60)
    print("✨ SCENE EDITING COMPLETE!")
    print("="*60)
    print(f"\n📍 Next steps:")
    print(f"   1. Copy {output_file.name} to server:")
    print(f"      scp -P 2221 '{output_file}' root@176.109.83.84:{_WS}/assets/")
    print(f"   2. Update compose.yml: SCENE_FILE=office_demo_edited.usdz")
    print(f"   3. Redeploy: bash scripts/remote-deploy.sh --build")

    return True

if __name__ == "__main__":
    success = edit_office_scene()
    sys.exit(0 if success else 1)
