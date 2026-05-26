#!/bin/bash
# Save the current Isaac Sim scene to USD file using Isaac Sim API
# Usage: bash scripts/save-isaac-scene.sh [filename]
# Default: office_demo_<timestamp>.usdz (e.g., office_demo_20260525_143000.usdz)

TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
SCENE_FILENAME="${1:-office_demo_${TIMESTAMP}.usdz}"
SCENE_PATH="/workspace/scene_assets/$SCENE_FILENAME"

echo "Saving Isaac Sim scene to: $SCENE_PATH"
echo ""

# Python script to save the current scene using Isaac Sim core API
# Using unquoted EOF to allow variable expansion
docker exec wam-isaac-sim python3 << EOF
import sys
import os

try:
    # Import Isaac Sim stage utilities (correct API from official docs)
    import isaacsim.core.utils.stage as stage_utils
    from omni.isaac.core import World

    scene_path = "/workspace/scene_assets/$SCENE_FILENAME"

    print(f"[DEBUG] Attempting to save to: {scene_path}")

    # Check if we can access the world/stage
    try:
        world = World.instance()
        if world and world.stage:
            print(f"[DEBUG] World stage found")
            stage_path = world.stage.GetRootLayer().GetRealPath()
            print(f"[DEBUG] Current stage path: {stage_path}")
        else:
            print(f"[DEBUG] No world instance or stage available")
    except Exception as e:
        print(f"[DEBUG] Could not access world: {e}")

    # Try to save the current stage
    print(f"[DEBUG] Calling save_stage()...")
    success = stage_utils.save_stage(scene_path, save_and_reload_in_place=False)

    print(f"[DEBUG] save_stage returned: {success}")

    if success:
        # Verify file was created
        if os.path.exists(scene_path):
            size_mb = os.path.getsize(scene_path) / (1024*1024)
            print(f"✓ Scene saved successfully!")
            print(f"  Path: {scene_path}")
            print(f"  Size: {size_mb:.1f} MB")
        else:
            print(f"⚠ ERROR: save_stage said success but file doesn't exist!")
            print(f"  Path: {scene_path}")

            # List what IS in the directory
            import glob
            files = glob.glob("/workspace/scene_assets/*")
            print(f"[DEBUG] Files in /workspace/scene_assets/: {files}")
            sys.exit(1)
    else:
        print(f"✗ save_stage returned False")
        sys.exit(1)

except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
EOF

SAVE_RESULT=$?

echo ""
if [ $SAVE_RESULT -eq 0 ]; then
    echo "✓ Scene saved successfully!"
    echo ""
    echo "Next: Download to local machine:"
    echo "  bash scripts/copy-media-from-server.sh --scenes"
else
    echo "✗ Failed to save scene. Check logs above."
    exit 1
fi
