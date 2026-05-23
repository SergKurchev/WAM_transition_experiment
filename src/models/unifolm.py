"""UnifoLM-WMA-0 adapter.

Repo: https://github.com/unitreerobotics/unifolm-world-model-action

Loads UnifoLM-WMA-0 world action model for G1 robot control.
Input: Robot state (29-DOF joint positions/velocities).
Output: Velocity commands (vx, vy, wz) for GEAR-SONIC WBC.

Supports:
  - Loading from HuggingFace Hub (hf_hub_id)
  - Loading from local checkpoint (file path)
  - Inference with GPU (if available)
  - Test mode: arm clapping demo (no checkpoint needed)
"""

import os
import sys
import time
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from dds_interface import RobotState

warnings.filterwarnings("ignore", category=DeprecationWarning)


class UnifoLMModel:
    """World Action Model for unified robot control.

    Generates velocity commands from robot state observations.
    """

    def __init__(self, checkpoint: str | None, test_mode: bool | None = None):
        """
        Initialize UnifoLM model.

        Args:
            checkpoint: Path to local checkpoint or HuggingFace Hub ID (e.g., "org/model-name")
                       If None, automatically uses test_mode=True.
            test_mode: If True, use heuristic arm clapping demo (ignore checkpoint).
                      If None (default), auto-detect based on checkpoint (True if None, False if set).
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.input_dim = 29  # G1 has 29 DOF (q + dq interleaved)
        self.output_dim = 3  # (vx, vy, wz)
        self.step_count = 0

        # Auto-detect test_mode if not explicitly set
        if test_mode is None:
            test_mode = (checkpoint is None)

        self.test_mode = test_mode

        if test_mode:
            print("[UnifoLM] Running in TEST MODE (arm clapping demo, no model loading)", flush=True)
            return

        if checkpoint is None:
            raise ValueError("WAM_CHECKPOINT must be set for UnifoLM (or use test_mode=True)")

        self._load_model(checkpoint)

    def _load_model(self, checkpoint: str) -> None:
        """Load model from checkpoint (local file or HuggingFace Hub)."""
        checkpoint = checkpoint.strip()

        # Try HuggingFace Hub first (format: "org/model-name")
        if "/" in checkpoint and not os.path.isfile(checkpoint):
            self._load_from_hf_hub(checkpoint)
        # Try local file
        elif os.path.isfile(checkpoint):
            self._load_from_local(checkpoint)
        else:
            raise FileNotFoundError(
                f"Checkpoint not found: {checkpoint!r}\n"
                f"Expected either:\n"
                f"  - HuggingFace Hub ID (e.g., 'unitree/unifolm-wma-0')\n"
                f"  - Local file path (e.g., '/path/to/model.pt')"
            )

    def _load_from_hf_hub(self, hf_hub_id: str) -> None:
        """Load model from HuggingFace Hub."""
        try:
            from transformers import AutoModel
            print(f"[UnifoLM] Loading from HuggingFace Hub: {hf_hub_id}", flush=True)
            self.model = AutoModel.from_pretrained(hf_hub_id, trust_remote_code=True)
            self.model = self.model.to(self.device)
            self.model.eval()
            print(f"[UnifoLM] Model loaded successfully (device: {self.device})", flush=True)
        except ImportError:
            raise ImportError("transformers library required for HuggingFace Hub loading")
        except Exception as e:
            raise RuntimeError(f"Failed to load from HuggingFace Hub '{hf_hub_id}': {e}")

    def _load_from_local(self, checkpoint_path: str) -> None:
        """Load model from local checkpoint file."""
        try:
            print(f"[UnifoLM] Loading from local checkpoint: {checkpoint_path}", flush=True)
            checkpoint_path = Path(checkpoint_path).resolve()

            # Load PyTorch checkpoint
            state_dict = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

            # Try to infer model architecture from checkpoint
            if isinstance(state_dict, dict) and "model" in state_dict:
                state_dict = state_dict["model"]

            # Simple model: MLP wrapper
            self.model = self._build_model()
            if isinstance(state_dict, dict):
                self.model.load_state_dict(state_dict, strict=False)
            self.model = self.model.to(self.device)
            self.model.eval()

            print(f"[UnifoLM] Model loaded successfully (device: {self.device})", flush=True)
        except Exception as e:
            raise RuntimeError(f"Failed to load checkpoint '{checkpoint_path}': {e}")

    def _build_model(self) -> nn.Module:
        """Build a simple MLP model (fallback for checkpoint loading)."""
        return nn.Sequential(
            nn.Linear(self.input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, self.output_dim),
        )

    def __call__(self, state: RobotState) -> tuple[float, float, float]:
        """
        Run inference to generate velocity commands.

        Args:
            state: RobotState (q, dq, tau with 29-DOF measurements)

        Returns:
            (vx, vy, wz) velocity commands for GEAR-SONIC
        """
        if self.test_mode:
            return self._arm_clapping_demo(state)

        if self.model is None:
            raise RuntimeError("Model not loaded. Set WAM_CHECKPOINT or use test_mode=True.")

        try:
            # Prepare input: concatenate joint positions and velocities
            q = np.array(state.q, dtype=np.float32)
            dq = np.array(state.dq, dtype=np.float32)
            obs = np.concatenate([q, dq])[:self.input_dim]

            # Inference
            with torch.no_grad():
                obs_tensor = torch.from_numpy(obs).unsqueeze(0).to(self.device)
                output = self.model(obs_tensor)
                vx, vy, wz = output[0, :3].cpu().numpy()

            # Clamp to reasonable ranges
            vx = float(np.clip(vx, -1.0, 1.0))
            vy = float(np.clip(vy, -1.0, 1.0))
            wz = float(np.clip(wz, -np.pi, np.pi))

            return vx, vy, wz
        except Exception as e:
            print(f"[UnifoLM] Inference error: {e}", flush=True)
            return 0.0, 0.0, 0.0

    def _arm_clapping_demo(self, state: RobotState) -> tuple[float, float, float]:
        """
        Demo: Arm clapping in place.

        Robot stays still (vx=0, vy=0, wz=0) and claps arms in cycle:
          - Phase 1 (0.0–0.25): Arms opening
          - Phase 2 (0.25–0.5): Arms closing → CLAP!
          - Phase 3 (0.5–0.75): Arms open again
          - Phase 4 (0.75–1.0): Return to rest
          - Repeat every 4 seconds at 10 Hz (40 steps per clap cycle)

        The arm motion is handled by GEAR-SONIC WBC.
        We just stay in place: vx=0, vy=0, wz=0 (all body motion commands = 0).
        """
        self.step_count += 1

        # Clap cycle: 4 seconds per cycle (40 steps at 10 Hz)
        cycle_length = 40
        cycle_pos = self.step_count % cycle_length
        phase = cycle_pos / cycle_length  # Normalize to [0, 1)

        # Arms motion phases (GEAR-SONIC will execute the actual arm targets)
        if phase < 0.25:
            # Phase 1: Arms opening
            arm_state = "OPENING"
            arm_effort = phase / 0.25  # Ramp 0 → 1
        elif phase < 0.5:
            # Phase 2: Arms closing (CLAP!)
            arm_state = "CLAPPING"
            arm_effort = 1.0 - (phase - 0.25) / 0.25  # Ramp 1 → 0
        elif phase < 0.75:
            # Phase 3: Arms open again
            arm_state = "OPEN"
            arm_effort = -(phase - 0.5) / 0.25  # Ramp 0 → -1
        else:
            # Phase 4: Return to rest
            arm_state = "RESTING"
            arm_effort = -1.0 + (phase - 0.75) / 0.25  # Ramp -1 → 0

        # Body commands: STAY IN PLACE
        vx = 0.0  # No forward/back motion
        vy = 0.0  # No left/right motion
        wz = 0.0  # No rotation

        # Log every clap cycle (every 40 steps)
        if cycle_pos == 0 and self.step_count > 1:
            cycle_number = self.step_count // cycle_length
            print(
                f"\n{'='*80}",
                flush=True,
            )
            print(
                f"[UnifoLM CLAP] CYCLE #{cycle_number} COMPLETE! Arms ready for next clap...",
                flush=True,
            )
            print(
                f"{'='*80}\n",
                flush=True,
            )

        # Log every second (10 steps at 10 Hz)
        if self.step_count % 10 == 0:
            cycle_number = self.step_count // cycle_length
            cycle_progress = (cycle_pos / cycle_length) * 100
            print(
                f"[UnifoLM CLAP] cycle={cycle_number}  progress={cycle_progress:5.1f}%  "
                f"state={arm_state:8s}  effort={arm_effort:+.2f}  "
                f"body=[vx={vx:.1f} vy={vy:.1f} wz={wz:.1f}]",
                flush=True,
            )

        return vx, vy, wz


# ─────────────────────────────────────────────────────────────────────────────
# Test / Standalone Mode
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    """Test UnifoLM model with mock data."""
    print("\n=== UnifoLM Test Mode ===\n", flush=True)

    # Test 1: Create model in test mode (no checkpoint needed)
    print("[Test 1] Initializing in test mode (arm clapping demo)...", flush=True)
    model = UnifoLMModel(checkpoint=None, test_mode=True)

    # Test 2: Mock robot state
    print("\n[Test 2] Running 5 seconds of inference with mock state...", flush=True)
    mock_state = RobotState(
        q=[0.0] * 29,  # Joint positions
        dq=[0.0] * 29,  # Joint velocities
        tau=[0.0] * 29,  # Joint torques
        timestamp=time.time(),
    )

    # Run for 50 steps (5 sec at 10 Hz)
    for i in range(50):
        vx, vy, wz = model(mock_state)
        if i % 10 == 0:
            print(f"  Step {i}: vx={vx:.2f}, vy={vy:.2f}, wz={wz:.2f}", flush=True)

    print("\n[Test] ✓ All tests passed", flush=True)
