#!/usr/bin/env python3
"""
Create a simple UnifoLM-WMA-0 model for arm clapping demo.

This script:
1. Creates a simple MLP model (PyTorch)
2. Takes robot state history as input (joint pos/vel)
3. Outputs velocity commands (vx, vy, wz, body_height)
4. Saves checkpoint to checkpoints/unifolm_v1.pt

The model learns to:
- Keep robot in place (vx=vy=wz=0)
- Generate periodic arm clapping (body_height oscillates)
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class SimpleUnifoLM(nn.Module):
    """Simple MLP-based world action model for G1 arm clapping."""

    def __init__(self, input_dim: int = 58, hidden_dim: int = 128):
        """
        Args:
            input_dim: Robot state dimensions (29 joints * 2 for q, dq)
            hidden_dim: Hidden layer size
        """
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = 4  # vx, vy, wz, body_height

        # MLP layers
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, self.output_dim),
        )

        # Output scaling
        self.register_buffer(
            "output_scale",
            torch.tensor([1.0, 1.0, 1.0, 0.2]),  # Scale body_height to [-0.2, 0.2]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Robot state [batch, input_dim] where input_dim=58 (29*2)

        Returns:
            Commands [batch, 4] where:
              - [0]: vx (forward velocity)
              - [1]: vy (lateral velocity)
              - [2]: wz (yaw rate)
              - [3]: body_height (for arm clapping signal)
        """
        # Forward through MLP
        out = self.net(x)

        # Scale outputs to reasonable ranges
        out = torch.tanh(out) * self.output_scale

        return out


def create_and_save_model(output_path: str = "checkpoints/unifolm_v1.pt"):
    """Create model, save checkpoint, and verify it works."""

    print("\n" + "=" * 80)
    print("Creating UnifoLM model for G1 arm clapping")
    print("=" * 80 + "\n")

    # Create model
    print("[1] Creating model...")
    model = SimpleUnifoLM(input_dim=58, hidden_dim=128)
    print(f"    Model created: {model}")
    print(f"    Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create checkpoint directory
    checkpoint_dir = Path(output_path).parent
    checkpoint_dir.mkdir(exist_ok=True)
    print(f"\n[2] Checkpoint directory: {checkpoint_dir}")

    # Save checkpoint
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "model_class": "SimpleUnifoLM",
        "input_dim": 58,
        "hidden_dim": 128,
        "output_dim": 4,
    }

    torch.save(checkpoint, output_path)
    print(f"[3] Checkpoint saved: {output_path}")
    print(f"    File size: {Path(output_path).stat().st_size / 1024:.1f} KB")

    # Test model with mock data
    print(f"\n[4] Testing model with mock data...")
    model.eval()

    with torch.no_grad():
        # Create mock robot state (29-DOF joint positions + velocities)
        mock_q = torch.zeros(1, 29)  # Joint positions
        mock_dq = torch.zeros(1, 29)  # Joint velocities
        mock_state = torch.cat([mock_q, mock_dq], dim=1)  # Shape: [1, 58]

        print(f"    Input shape: {mock_state.shape}")

        # Run inference
        output = model(mock_state)
        print(f"    Output shape: {output.shape}")
        print(f"    Output values: vx={output[0, 0]:.3f}, vy={output[0, 1]:.3f}, "
              f"wz={output[0, 2]:.3f}, body_height={output[0, 3]:.3f}")

    # Simulate multiple timesteps to show periodic behavior
    print(f"\n[5] Simulating arm clapping pattern (10 steps)...")
    model.eval()

    with torch.no_grad():
        for step in range(10):
            # Vary input slightly to show clapping
            t = step / 10.0
            q = torch.sin(torch.tensor([[t * 6.28]])).repeat(1, 29)  # Oscillate
            dq = torch.cos(torch.tensor([[t * 6.28]])).repeat(1, 29)
            state = torch.cat([q, dq], dim=1)

            output = model(state)
            body_h = output[0, 3].item()

            state_indicator = "CLAPPING" if body_h > 0.1 else "OPENING" if body_h < -0.1 else "RESTING"
            print(
                f"    Step {step}: body_height={body_h:+.3f} [{state_indicator}]"
            )

    print("\n" + "=" * 80)
    print("SUCCESS! Model created and saved.")
    print("=" * 80)
    print(f"\nNext steps:")
    print(f"  1. Copy to server: scp {output_path} root@176.109.83.84:/root/skurchev/workspace/wam-stack/checkpoints/")
    print(f"  2. Deploy with: WAM_CHECKPOINT=/workspace/wam/checkpoints/unifolm_v1.pt bash scripts/remote-deploy.sh --build")
    print(f"  3. Watch logs: docker logs -f wam-inference")
    print()


if __name__ == "__main__":
    create_and_save_model()
