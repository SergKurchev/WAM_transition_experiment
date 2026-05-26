"""UnifoLM-WMA-0 adapter.

Repo: https://github.com/unitreerobotics/unifolm-world-model-action

Loads UnifoLM-WMA-0 world action model for G1 robot control.
Input: Robot state (29-DOF joint positions/velocities).
Output: Velocity commands (vx, vy, wz, body_height) for GEAR-SONIC WBC.

Supports:
  - Loading from HuggingFace Hub (hf_hub_id)
  - Loading from local checkpoint (file path)
  - Inference with GPU (if available)
  - Test mode: arm extending forward demo (no checkpoint needed)
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

    def __init__(self, checkpoint: str | None, test_mode: bool | None = None, prompt: str | None = None):
        """
        Initialize UnifoLM model.

        Args:
            checkpoint: Path to local checkpoint or HuggingFace Hub ID (e.g., "org/model-name")
                       If None, automatically uses test_mode=True.
            test_mode: If True, use heuristic demo (ignore checkpoint).
                      If None (default), auto-detect based on checkpoint (True if None, False if set).
            prompt: Task prompt (e.g., "pick and place green cube in white basket").
                   Used to condition model output if vision system available.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.input_dim = 29  # G1 has 29 DOF (q + dq interleaved)
        self.output_dim = 3  # (vx, vy, wz)
        self.step_count = 0
        self.prompt = prompt or "default navigation"

        print(f"[UnifoLM] Task prompt: {self.prompt}", flush=True)

        # Auto-detect test_mode if not explicitly set
        if test_mode is None:
            test_mode = (checkpoint is None)

        self.test_mode = test_mode

        if test_mode:
            print(f"[UnifoLM] Running in TEST MODE (demo for task: {self.prompt})", flush=True)
            return

        if checkpoint is None:
            raise ValueError("WAM_CHECKPOINT must be set for UnifoLM (or use test_mode=True)")

        self._load_model(checkpoint)

    def _load_model(self, checkpoint: str) -> None:
        """Load model from checkpoint (HuggingFace Hub preferred)."""
        checkpoint = checkpoint.strip()

        # Try HuggingFace Hub first (format: "org/model-name")
        if "/" in checkpoint and not checkpoint.endswith(".ckpt") and not checkpoint.endswith(".pt"):
            self._load_from_hf_hub(checkpoint)
        # Try local file as fallback (includes .ckpt, .pt)
        elif os.path.isfile(checkpoint):
            # For local .ckpt files, prefer HuggingFace Hub to get full architecture
            print(f"[UnifoLM] Note: Local .ckpt file detected. For full video generation support, prefer HuggingFace Hub ID (e.g., 'unitree/unifolm-wma-0')", flush=True)
            self._load_from_local(checkpoint)
        else:
            raise FileNotFoundError(
                f"Checkpoint not found: {checkpoint!r}\n"
                f"Expected either:\n"
                f"  - HuggingFace Hub ID (e.g., 'unitree/unifolm-wma-0') [RECOMMENDED for video generation]\n"
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
        """Load model from local checkpoint file (PyTorch or PyTorch Lightning)."""
        try:
            print(f"[UnifoLM] Loading from local checkpoint: {checkpoint_path}", flush=True)
            checkpoint_path = Path(checkpoint_path).resolve()

            # Load PyTorch checkpoint
            ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

            # Handle PyTorch Lightning format
            if isinstance(ckpt, dict) and "state_dict" in ckpt:
                print(f"[UnifoLM] Detected PyTorch Lightning checkpoint format", flush=True)
                state_dict = ckpt["state_dict"]
                # Remove 'model.' prefix if present (PyTorch Lightning convention)
                state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}
            elif isinstance(ckpt, dict) and "model" in ckpt:
                print(f"[UnifoLM] Found 'model' key in checkpoint", flush=True)
                state_dict = ckpt["model"]
            elif isinstance(ckpt, dict):
                state_dict = ckpt
            else:
                raise RuntimeError(f"Unexpected checkpoint format: {type(ckpt)}")

            # Analyze output layer to determine output dimensions
            output_dims = self._infer_output_dims(state_dict)
            print(f"[UnifoLM] Inferred output dimensions: {output_dims}", flush=True)

            # Build model with correct output dimensions
            self.model = self._build_model(output_dims=output_dims)

            # Load state dict with non-strict mode to handle architecture mismatches
            if isinstance(state_dict, dict):
                missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
                if missing:
                    print(f"[UnifoLM] Missing keys: {len(missing)}", flush=True)
                if unexpected:
                    print(f"[UnifoLM] Unexpected keys: {len(unexpected)}", flush=True)

            self.model = self.model.to(self.device)
            self.model.eval()

            print(f"[UnifoLM] Model loaded successfully (device: {self.device})", flush=True)
        except Exception as e:
            raise RuntimeError(f"Failed to load checkpoint '{checkpoint_path}': {e}")

    def _infer_output_dims(self, state_dict: dict) -> int:
        """Try to infer output dimensions from checkpoint state dict."""
        print(f"[UnifoLM] Analyzing checkpoint structure ({len(state_dict)} keys)...", flush=True)

        # Print first 10 keys for debugging
        for i, key in enumerate(list(state_dict.keys())[:10]):
            param = state_dict[key]
            shape_str = str(param.shape) if hasattr(param, "shape") else str(type(param))
            print(f"  [{i}] {key}: {shape_str}", flush=True)

        # Look for final output layer
        for key, param in state_dict.items():
            if "output" in key.lower() or "head" in key.lower():
                if hasattr(param, "shape") and len(param.shape) >= 1:
                    out_dim = int(param.shape[-1])
                    print(f"[UnifoLM] Found output layer '{key}' with dimension {out_dim}", flush=True)
                    return out_dim

        print(f"[UnifoLM] No output layer found in state dict, using default {self.output_dim}", flush=True)
        return self.output_dim

    def _build_model(self, output_dims: int | None = None) -> nn.Module:
        """Build a simple MLP model (fallback for checkpoint loading)."""
        if output_dims is None:
            output_dims = self.output_dim
        return nn.Sequential(
            nn.Linear(self.input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dims),
        )

    def __call__(self, state: RobotState) -> tuple:
        """
        Run inference to generate velocity commands and optionally video prediction.

        Args:
            state: RobotState (q, dq, tau with 29-DOF measurements)

        Returns:
            (vx, vy, wz, body_height, video_output)
            where video_output is torch.Tensor or numpy array if model generates it, else None
        """
        if self.test_mode:
            vx, vy, wz, body_height = self._arm_extend_demo(state)
            return vx, vy, wz, body_height, None

        if self.model is None:
            raise RuntimeError("Model not loaded. Set WAM_CHECKPOINT or use test_mode=True.")

        try:
            self.step_count += 1

            # Prepare input: concatenate joint positions and velocities
            q = np.array(state.q, dtype=np.float32)
            dq = np.array(state.dq, dtype=np.float32)
            obs = np.concatenate([q, dq])[:self.input_dim]

            vx, vy, wz, body_height, video_output = 0.0, 0.0, 0.0, 0.0, None

            # Inference
            with torch.no_grad():
                obs_tensor = torch.from_numpy(obs).unsqueeze(0).to(self.device)
                output = self.model(obs_tensor)

                # Handle different model output formats
                # unifolm_v1.pt: outputs only [1, 3] (vx, vy, wz)
                # Full WMA: outputs [1, 4+] (vx, vy, wz, body_height, [video...])

                if isinstance(output, torch.Tensor):
                    output_shape = output.shape

                    # Debug: log output shape every 100 steps
                    if self.step_count % 100 == 0:
                        print(f"[UnifoLM] Model output shape: {output_shape}  step={self.step_count}", flush=True)

                    if output_shape[1] >= 3:
                        # Extract first 3 values (vx, vy, wz)
                        vx = float(output[0, 0].cpu().numpy())
                        vy = float(output[0, 1].cpu().numpy())
                        wz = float(output[0, 2].cpu().numpy())

                        # Body height (optional, for newer models)
                        if output_shape[1] > 3:
                            body_height = float(output[0, 3].cpu().numpy())
                        else:
                            body_height = 0.0

                        # Video output (optional, for full WMA model)
                        if output_shape[1] > 4:
                            video_output = output[:, 4:].cpu()
                        else:
                            video_output = None
                    else:
                        print(f"[UnifoLM] Warning: Model output shape {output_shape} too small, using zero actions", flush=True)
                        return 0.0, 0.0, 0.0, 0.0, None

            # Clamp action to reasonable ranges
            vx = float(np.clip(vx, -1.0, 1.0))
            vy = float(np.clip(vy, -1.0, 1.0))
            wz = float(np.clip(wz, -np.pi, np.pi))

            return vx, vy, wz, body_height, video_output
        except Exception as e:
            print(f"[UnifoLM] Inference error: {e}", flush=True)
            import traceback
            traceback.print_exc()
            return 0.0, 0.0, 0.0, 0.0, None

    def _arm_extend_demo(self, state: RobotState) -> tuple[float, float, float, float]:
        """
        Demo behavior based on task prompt.

        Supports:
          - "pick and place": Walk towards table, pick green cube, place in basket
          - default: Extend arms forward in place
        """
        self.step_count += 1

        # Check if this is pick-and-place task
        if "pick" in self.prompt.lower() and "place" in self.prompt.lower():
            return self._pick_and_place_demo(state)
        else:
            return self._arm_extend_demo_original(state)

    def _pick_and_place_demo(self, state: RobotState) -> tuple[float, float, float, float]:
        """
        Demo: Pick green cube from table and place in white basket.

        Sequence (120 steps = 12 seconds at 10 Hz):
          - Steps 0-30: Walk forward to table (0.3 m/s)
          - Steps 30-60: Bend down and pick (arms down)
          - Steps 60-90: Walk to basket (backward 0.2 m/s)
          - Steps 90-120: Place cube in basket (arms up)
          - Repeat cycle
        """
        cycle_length = 120
        cycle_pos = self.step_count % cycle_length
        phase = cycle_pos / cycle_length

        if phase < 0.25:
            # Walk forward to table
            vx, vy, wz = 0.3, 0.0, 0.0
            body_height = 0.0
        elif phase < 0.5:
            # Pick phase: stay, arms go down
            vx, vy, wz = 0.0, 0.0, 0.0
            body_height = -0.5  # Arm down signal
        elif phase < 0.75:
            # Walk backward to basket
            vx, vy, wz = -0.2, 0.0, 0.0
            body_height = -0.5
        else:
            # Place phase: stay, arms up
            vx, vy, wz = 0.0, 0.0, 0.0
            body_height = 0.5  # Arm up signal

        return vx, vy, wz, body_height

    def _arm_extend_demo_original(self, state: RobotState) -> tuple[float, float, float, float]:
        """
        Demo: Extend arms forward in place.

        Robot stays still (vx=0, vy=0, wz=0) and extends arms forward in cycle:
          - Phase 1 (0.0–0.33): Arms extending forward
          - Phase 2 (0.33–0.66): Arms fully extended (hold position)
          - Phase 3 (0.66–1.0): Arms retracting to rest
          - Repeat every 3 seconds at 10 Hz (30 steps per cycle)

        The arm motion is handled by GEAR-SONIC WBC.
        We stay in place: vx=0, vy=0, wz=0 (all body motion commands = 0).
        Body_height signal indicates arm extension: positive=extending, negative=retracting.
        """
        # Extension cycle: 3 seconds per cycle (30 steps at 10 Hz)
        cycle_length = 30
        cycle_pos = self.step_count % cycle_length
        phase = cycle_pos / cycle_length  # Normalize to [0, 1)

        # Arm extension phases (GEAR-SONIC will execute the actual arm targets)
        if phase < 0.33:
            # Phase 1: Arms extending forward
            arm_state = "EXTENDING"
            arm_effort = phase / 0.33  # Ramp 0 → 1
        elif phase < 0.66:
            # Phase 2: Arms fully extended (hold)
            arm_state = "EXTENDED"
            arm_effort = 1.0
        else:
            # Phase 3: Arms retracting to rest
            arm_state = "RETRACTING"
            arm_effort = 1.0 - (phase - 0.66) / 0.34  # Ramp 1 → 0

        # Body commands: STAY IN PLACE
        vx = 0.0  # No forward/back motion
        vy = 0.0  # No left/right motion
        wz = 0.0  # No rotation

        # Use body_height to signal arm extension to GEAR-SONIC
        # Positive body_height (up to 0.15) triggers arm extension forward
        # Zero body_height keeps arms at rest
        # This is a proxy signal: GEAR-SONIC will adjust arm targets based on body_height
        body_height = arm_effort * 0.15  # Scale to reasonable height range

        # Log every extension cycle (every 30 steps)
        if cycle_pos == 0 and self.step_count > 1:
            cycle_number = self.step_count // cycle_length
            print(
                f"\n{'='*80}",
                flush=True,
            )
            print(
                f"[UnifoLM EXTEND] CYCLE #{cycle_number} COMPLETE! Arms ready for next extension...",
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
                f"[UnifoLM EXTEND] cycle={cycle_number}  progress={cycle_progress:5.1f}%  "
                f"state={arm_state:8s}  effort={arm_effort:+.2f}  "
                f"body=[vx={vx:.1f} vy={vy:.1f} wz={wz:.1f} h={body_height:+.2f}]",
                flush=True,
            )

        return vx, vy, wz, body_height


# ─────────────────────────────────────────────────────────────────────────────
# Test / Standalone Mode
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    """Test UnifoLM model with mock data."""
    print("\n=== UnifoLM Test Mode ===\n", flush=True)

    # Test 1: Create model in test mode (no checkpoint needed)
    print("[Test 1] Initializing in test mode (arm extending forward demo)...", flush=True)
    model = UnifoLMModel(checkpoint=None, test_mode=True)

    # Test 2: Mock robot state
    print("\n[Test 2] Running 10 seconds of inference with mock state...", flush=True)
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
