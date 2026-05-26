#!/usr/bin/env python3
"""Debug script to understand unifolm_wma_dual.ckpt structure."""

import torch
import sys

checkpoint_path = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/unifolm_wma_dual.ckpt"

print(f"\n=== Checkpoint Analysis: {checkpoint_path} ===\n")

try:
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    print(f"Type: {type(state_dict)}")

    if isinstance(state_dict, dict):
        print(f"Top-level keys: {list(state_dict.keys())}\n")

        # Check for model state dict
        if "model" in state_dict:
            model_dict = state_dict["model"]
            print(f"Found 'model' key containing: {len(model_dict)} parameters\n")

            # Print first 20 layer names and shapes
            print("Layer structure:")
            for i, (name, param) in enumerate(model_dict.items()):
                if i < 20:
                    shape = param.shape if hasattr(param, "shape") else type(param)
                    print(f"  [{i:2d}] {name:50s} {str(shape)}")
                else:
                    print(f"  ... ({len(model_dict) - 20} more layers)")
                    break

            # Look for video-related layers
            print("\n" + "="*80)
            print("Searching for video/prediction layers:")
            video_keywords = ["video", "pred", "output", "head", "decode", "frame", "img"]
            found_any = False
            for name in model_dict.keys():
                for keyword in video_keywords:
                    if keyword.lower() in name.lower():
                        param = model_dict[name]
                        shape = param.shape if hasattr(param, "shape") else type(param)
                        print(f"  {name:60s} {str(shape)}")
                        found_any = True
                        break
            if not found_any:
                print("  (No video/prediction layers found)")

            # Check for config or model type info
            print("\n" + "="*80)
            if "config" in state_dict:
                print("Config found:", state_dict["config"])

        else:
            print("No 'model' key found. Showing first 10 top-level items:\n")
            for i, (key, val) in enumerate(list(state_dict.items())[:10]):
                if hasattr(val, "shape"):
                    print(f"  {key}: {val.shape}")
                else:
                    print(f"  {key}: {type(val)}")
    else:
        print(f"Checkpoint is {type(state_dict)}, not a dict")
        if hasattr(state_dict, "to"):
            print("  (appears to be a model object)")

except Exception as e:
    print(f"Error loading checkpoint: {e}")
    import traceback
    traceback.print_exc()
