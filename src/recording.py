"""Recording module for WAM inference: video, frames, and command logs.

Saves:
  - input_frames/: RobotState visualization (q, dq, tau as images)
  - command_logs/: Velocity commands (vx, vy, wz, body_height) as CSV
  - isaac_frames/: Screenshots from Isaac Sim (if available via /tmp/isaac_frame.png)
"""

import os
import csv
import time
from pathlib import Path
from datetime import datetime


class MediaRecorder:
    """Record inference inputs, outputs, and simulator frames."""

    def __init__(self, media_dir: str = "/workspace/wam/media"):
        """Initialize recording directories.

        Args:
            media_dir: Base directory for all media (will be created if missing)
        """
        self.media_dir = Path(media_dir)
        self.media_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        self.input_frames_dir = self.media_dir / "input_frames"
        self.command_logs_dir = self.media_dir / "command_logs"
        self.isaac_frames_dir = self.media_dir / "isaac_frames"

        for d in [self.input_frames_dir, self.command_logs_dir, self.isaac_frames_dir]:
            d.mkdir(exist_ok=True)

        # CSV log for commands
        self.command_log_file = self.command_logs_dir / f"commands_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        self._init_command_log()

        self.frame_count = 0
        print(f"[RECORDING] Media directory: {self.media_dir.resolve()}", flush=True)
        print(f"[RECORDING] Command log: {self.command_log_file.resolve()}", flush=True)

    def _init_command_log(self):
        """Initialize CSV file for command logging."""
        with open(self.command_log_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "step", "vx", "vy", "wz", "body_height"])

    def save_input_frame(self, step: int, state):
        """Save RobotState as a text frame for inspection.

        Args:
            step: Control loop step number
            state: RobotState object (q, dq, tau, timestamp)
        """
        frame_file = self.input_frames_dir / f"state_{step:06d}.txt"

        with open(frame_file, "w") as f:
            f.write(f"Step: {step}\n")
            f.write(f"Timestamp: {state.timestamp}\n\n")
            f.write(f"Joint Positions (q) [{len(state.q)}]:\n")
            f.write("  " + " ".join(f"{q:+.4f}" for q in state.q) + "\n\n")
            f.write(f"Joint Velocities (dq) [{len(state.dq)}]:\n")
            f.write("  " + " ".join(f"{dq:+.4f}" for dq in state.dq) + "\n\n")
            f.write(f"Joint Torques (tau) [{len(state.tau)}]:\n")
            f.write("  " + " ".join(f"{tau:+.4f}" for tau in state.tau) + "\n")

        return str(frame_file.resolve())

    def save_command(self, step: int, vx: float, vy: float, wz: float, body_height: float):
        """Log velocity command to CSV.

        Args:
            step: Control loop step number
            vx, vy, wz, body_height: Command values
        """
        timestamp = datetime.now().isoformat()
        with open(self.command_log_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([timestamp, step, f"{vx:.4f}", f"{vy:.4f}", f"{wz:.4f}", f"{body_height:.4f}"])

    def save_isaac_frame(self, step: int):
        """Copy Isaac Sim frame from /tmp/isaac_frame.png to media directory.

        Args:
            step: Control loop step number

        Returns:
            Full path to saved frame, or None if source doesn't exist
        """
        isaac_src = Path("/tmp/isaac_frame.png")
        if not isaac_src.exists():
            return None

        isaac_dest = self.isaac_frames_dir / f"isaac_{step:06d}.png"

        try:
            import shutil
            shutil.copy2(isaac_src, isaac_dest)
            return str(isaac_dest.resolve())
        except Exception as e:
            print(f"[RECORDING] Failed to copy Isaac frame: {e}", flush=True)
            return None

    def finalize(self):
        """Print summary of all saved media with full paths."""
        print(f"\n{'='*80}", flush=True)
        print(f"[RECORDING] SUMMARY", flush=True)
        print(f"{'='*80}", flush=True)

        # Count files in each directory
        input_count = len(list(self.input_frames_dir.glob("state_*.txt")))
        isaac_count = len(list(self.isaac_frames_dir.glob("isaac_*.png")))

        print(f"[RECORDING] Input frames saved: {input_count}", flush=True)
        print(f"            Location: {self.input_frames_dir.resolve()}", flush=True)

        print(f"[RECORDING] Isaac Sim frames saved: {isaac_count}", flush=True)
        print(f"            Location: {self.isaac_frames_dir.resolve()}", flush=True)

        print(f"[RECORDING] Command log (CSV):", flush=True)
        print(f"            {self.command_log_file.resolve()}", flush=True)

        print(f"[RECORDING] All media in: {self.media_dir.resolve()}", flush=True)
        print(f"{'='*80}\n", flush=True)
