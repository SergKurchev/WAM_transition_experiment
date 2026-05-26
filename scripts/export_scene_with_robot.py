#!/usr/bin/env python3
"""
Export office scene with G1 robot positioned at X=-2.0
Creates a composite USD file for editing in Blender
"""
import os
import sys
from pathlib import Path

def create_composite_usd():
    """Create a USD file that includes office_demo.usdz + G1 robot"""

    # Paths
    workspace = Path("/root/skurchev/workspace")
    scene_assets = workspace / "assets"
    robot_assets = workspace / "mws-dimos/assets/robots/g1"
    output_file = workspace / "assets/office_demo_with_robot.usd"

    # Check files exist
    office_usdz = scene_assets / "office_demo.usdz"
    g1_usd = robot_assets / "g1_29dof.usd"

    if not office_usdz.exists():
        print(f"❌ Office scene not found: {office_usdz}")
        return False

    if not g1_usd.exists():
        print(f"❌ G1 robot not found: {g1_usd}")
        return False

    print(f"📦 Office scene: {office_usdz}")
    print(f"🤖 G1 robot: {g1_usd}")

    # Create composite USD
    usd_content = f"""#usda 1.0
(
    defaultPrim = "World"
    upAxis = "Z"
)

def Xform "World"
{{
    # Import office environment
    def "OfficeEnvironment" (
        prepend references = @{office_usdz}@
    )
    {{
    }}

    # Import G1 robot at position X=-2.0
    def "G1" (
        prepend references = @{g1_usd}@
    )
    {{
        double3 xformOp:translate = (-2.0, 0.0, 0.0)
        uniform token[] xformOpOrder = ["xformOp:translate"]
    }}
}}
"""

    # Write composite USD
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(usd_content)

    print(f"\n✅ Composite USD created: {output_file}")
    print(f"📝 File size: {output_file.stat().st_size} bytes")
    print(f"\n📋 Next steps:")
    print(f"   1. Copy to local: scp -P 2221 root@176.109.83.84:{output_file} ~/projects/wam-stack/office_demo_with_robot.usd")
    print(f"   2. Open in Blender: File > Open {output_file.name}")
    print(f"   3. Edit scene (add box, camera, shelf)")
    print(f"   4. Export as office_demo_edited.usdz")

    return True

if __name__ == "__main__":
    if create_composite_usd():
        print("\n✨ Ready for Blender editing!")
        sys.exit(0)
    else:
        sys.exit(1)
