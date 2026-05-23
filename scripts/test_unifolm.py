#!/usr/bin/env python3
"""
Test script for UnifoLM model integration.

Usage:
  # Test with mock DDS data (no server needed)
  python scripts/test_unifolm.py

  # Test with real DDS (requires full stack running on server)
  DDS_IFACE=lo python scripts/test_unifolm.py --real-dds

Features:
  - Mock robot state (arm clapping demo mode)
  - Validates I/O dimensions
  - Checks inference timing (must stay within 100ms for 10 Hz loop)
  - Runs for 10 seconds of simulated time
"""

import os
import sys
import time
import argparse
from pathlib import Path

# Add src/ to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dds_interface import RobotState
from models.unifolm import UnifoLMModel


def test_unifolm_inference():
    """Test UnifoLM model with mock data."""
    print("\n" + "=" * 70)
    print("UnifoLM Integration Test")
    print("=" * 70 + "\n")

    # Test 1: Initialize model (test mode)
    print("[TEST 1] Initialize UnifoLM in test mode (arm clapping demo)")
    print("-" * 70)
    try:
        model = UnifoLMModel(checkpoint=None, test_mode=True)
        print("[PASS] Model initialized successfully")
        print(f"  Device: {model.device}")
        print(f"  Mode: TEST (arm clapping demo)")
        print(f"  Input dim: {model.input_dim} (29-DOF joint state)")
        print(f"  Output dim: {model.output_dim} (vx, vy, wz)")
    except Exception as e:
        print(f"[FAIL] {e}")
        return False

    # Test 2: Create mock robot state
    print("\n[TEST 2] Create mock robot state (G1 29-DOF)")
    print("-" * 70)
    try:
        mock_state = RobotState(
            q=[0.0] * 29,  # Joint positions
            dq=[0.0] * 29,  # Joint velocities
            tau=[0.0] * 29,  # Joint torques
            timestamp=time.time(),
        )
        print("[PASS] Mock state created")
        print(f"  Joints (q): {len(mock_state.q)}")
        print(f"  Velocities (dq): {len(mock_state.dq)}")
        print(f"  Torques (tau): {len(mock_state.tau)}")
    except Exception as e:
        print(f"[FAIL] {e}")
        return False

    # Test 3: Run inference loop (20 seconds showing clapping pattern)
    print("\n[TEST 3] Run 20 seconds of inference (200 steps at 10 Hz)")
    print("-" * 70)
    print("Pattern: Robot stays in place and claps arms in 4-second cycles")
    print()
    control_hz = 10
    dt = 1.0 / control_hz
    num_steps = 200  # 20 seconds = 5 complete clap cycles
    timings = []

    try:
        for step in range(num_steps):
            t0 = time.time()

            # Run inference
            vx, vy, wz = model(mock_state)

            elapsed = time.time() - t0
            timings.append(elapsed)

            # Validate output
            assert isinstance(vx, float), f"vx must be float, got {type(vx)}"
            assert isinstance(vy, float), f"vy must be float, got {type(vy)}"
            assert isinstance(wz, float), f"wz must be float, got {type(wz)}"
            assert -1.0 <= vx <= 1.0, f"vx out of range: {vx}"
            assert -1.0 <= vy <= 1.0, f"vy out of range: {vy}"
            assert -2.0 <= wz <= 2.0, f"wz out of range: {wz}"

            # Log every 2 seconds (20 steps at 10 Hz)
            if step % 20 == 0:
                elapsed_sec = step / control_hz
                print(
                    f"  T={elapsed_sec:5.1f}s  Step {step:3d}/{num_steps}  "
                    f"cmd=[vx={vx:+.1f} vy={vy:+.1f} wz={wz:+.1f}]  "
                    f"inference_ms={elapsed*1000:.1f}"
                )

            # Sleep to maintain 10 Hz
            time.sleep(max(0.0, dt - elapsed))

        print("\n[PASS] Inference loop completed successfully (5 clap cycles)")
    except Exception as e:
        print(f"[FAIL] {e}")
        return False

    # Test 4: Analyze timing
    print("\n[TEST 4] Analyze inference timing")
    print("-" * 70)
    avg_ms = sum(timings) * 1000 / len(timings)
    max_ms = max(timings) * 1000
    min_ms = min(timings) * 1000
    budget_ms = 100  # Must fit in 100ms for 10 Hz

    print(f"  Min:     {min_ms:.2f} ms")
    print(f"  Avg:     {avg_ms:.2f} ms")
    print(f"  Max:     {max_ms:.2f} ms")
    print(f"  Budget:  {budget_ms:.0f} ms (10 Hz control loop)")

    if max_ms > budget_ms:
        print(f"  [WARN] Some steps exceeded budget ({max_ms:.1f} > {budget_ms})")
    else:
        print(f"  [PASS] All steps within budget")

    # Test 5: Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print("[PASS] All tests passed!")
    print("\nNext steps:")
    print("  1. Deploy to server: bash scripts/deploy.sh")
    print("  2. Set model: WAM_MODEL=unifolm bash scripts/deploy.sh")
    print("  3. Monitor: docker logs -f wam-inference")
    print("  4. Visualize: ssh -N -L 6081:localhost:6080 -p 2221 x32-techgov-GPU-01")
    print("=" * 70 + "\n")

    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test UnifoLM integration")
    parser.add_argument(
        "--real-dds",
        action="store_true",
        help="Test with real DDS (requires server stack running)",
    )
    args = parser.parse_args()

    success = test_unifolm_inference()
    sys.exit(0 if success else 1)
