"""Recording module for WAM inference: video, frames, and command logs.

Saves:
  - input_frames/: RobotState visualization (q, dq, tau as text)
  - command_logs/: Velocity commands (vx, vy, wz, body_height) as CSV
  - robot_camera/: D435i camera frames from ROS2 /g1/camera/color/image_raw (1280x720 RGB)
  - isaac_frames/: Screenshots from Isaac Sim (if available)
  - model_output/: Model predictions (reserved for future video generation)
"""

import os
import csv
import time
import threading
import mmap
from pathlib import Path
from datetime import datetime
from queue import Queue

try:
    import numpy as np
except ImportError:
    np = None

try:
    from PIL import Image
except ImportError:
    Image = None


class MediaRecorder:
    """Record inference inputs, outputs, and camera frames from ROS2."""

    def __init__(self, media_dir: str = "/workspace/wam/media", prompt: str = "default"):
        """Initialize recording directories and ROS2 camera subscriber.

        Args:
            media_dir: Base directory for all media (will be created if missing)
            prompt: Task prompt (logged for reference)
        """
        self.media_dir = Path(media_dir)
        self.media_dir.mkdir(parents=True, exist_ok=True)
        self.prompt = prompt

        # Create subdirectories
        self.input_frames_dir = self.media_dir / "input_frames"
        self.command_logs_dir = self.media_dir / "command_logs"
        self.isaac_frames_dir = self.media_dir / "isaac_frames"
        self.robot_camera_dir = self.media_dir / "robot_camera"
        self.model_video_dir = self.media_dir / "model_output"

        for d in [self.input_frames_dir, self.command_logs_dir, self.isaac_frames_dir,
                  self.robot_camera_dir, self.model_video_dir]:
            d.mkdir(exist_ok=True)

        # CSV log for commands
        self.command_log_file = self.command_logs_dir / f"commands_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        self._init_command_log()

        # Write metadata file
        metadata_file = self.media_dir / "metadata.txt"
        with open(metadata_file, "w") as f:
            f.write(f"Timestamp: {datetime.now().isoformat()}\n")
            f.write(f"Task Prompt: {self.prompt}\n")
            f.write(f"Media Directory: {self.media_dir.resolve()}\n")
            f.write(f"Camera: Intel RealSense D435i (1280×720 RGB, 30 FPS)\n")
            f.write(f"Camera Topic: /g1/camera/color/image_raw (ROS2)\n")

        # Shared memory camera (read from /run/mws/camera.rgb, 1280×720 RGB)
        self.camera_shm_path = "/run/mws/camera.rgb"
        self.camera_shm_size = 1280 * 720 * 3  # RGB, 1280×720
        self.camera_shm_map = None
        self.latest_camera_frame = None
        self._init_camera_shm()

        self.frame_count = 0
        print(f"[RECORDING] Media directory: {self.media_dir.resolve()}", flush=True)
        print(f"[RECORDING] Task prompt: {self.prompt}", flush=True)
        print(f"[RECORDING] Command log: {self.command_log_file.resolve()}", flush=True)
        print(f"[RECORDING] Camera: ROS2 /g1/camera/color/image_raw (D435i)", flush=True)

    def _init_camera_shm(self):
        """Initialize shared memory camera (reads from /run/mws/camera.rgb)."""
        if np is None:
            print(f"[RECORDING] NumPy not available - camera frames disabled", flush=True)
            return

        try:
            if not os.path.exists(self.camera_shm_path):
                print(f"[RECORDING] Camera shared memory not found at {self.camera_shm_path}", flush=True)
                return

            self.camera_shm_map = open(self.camera_shm_path, "r+b")
            print(f"[RECORDING] ✓ Camera SHM: {self.camera_shm_path} (D435i, 1280×720 RGB)", flush=True)
        except Exception as e:
            print(f"[RECORDING] Camera SHM init warning: {e} - continuing without camera", flush=True)

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

    def save_robot_camera_frame(self, step: int):
        """Save robot camera frame from shared memory /run/mws/camera.rgb.

        Captures view from Intel RealSense D435i mounted on G1 head.
        Resolution: 1280×720 RGB (sim) or 640×480 (real robot)

        Args:
            step: Control loop step number

        Returns:
            Full path to saved frame, or None if no frame available
        """
        if np is None or self.camera_shm_map is None:
            return None

        try:
            # Read RGB bytes from shared memory (1280×720×3)
            self.camera_shm_map.seek(0)
            raw_bytes = self.camera_shm_map.read(self.camera_shm_size)

            # Convert to numpy array: (1280×720×3) uint8
            rgb_array = np.frombuffer(raw_bytes, dtype=np.uint8).reshape(720, 1280, 3)

            # Save as PNG
            camera_dest = self.robot_camera_dir / f"camera_{step:06d}.png"
            if Image is not None:
                # Use PIL to save PNG (no dependencies needed)
                img = Image.fromarray(rgb_array, mode='RGB')
                img.save(str(camera_dest))
            else:
                # Fallback: save as numpy file if PIL not available
                with open(camera_dest.with_suffix('.npz'), 'wb') as f:
                    np.save(f, rgb_array)

            return str(camera_dest.resolve())
        except Exception as e:
            return None

    def _prune_recordings(self, directory: Path, pattern: str, keep_count: int = 10) -> int:
        """Keep only first & last N files, delete the middle ones (save space).

        Args:
            directory: Directory to prune
            pattern: Glob pattern (e.g., "camera_*.png")
            keep_count: Number to keep at start and end

        Returns:
            Number of files deleted
        """
        if not directory.exists():
            return 0

        files = sorted(list(directory.glob(pattern)))
        if len(files) <= 2 * keep_count:
            return 0  # Not enough files, keep all

        # Keep first keep_count and last keep_count
        to_delete = files[keep_count:-keep_count]
        deleted = 0
        for f in to_delete:
            try:
                f.unlink()
                deleted += 1
            except Exception:
                pass

        return deleted

    def finalize(self):
        """Print summary, prune old recordings, then finalize."""
        print(f"\n{'='*80}", flush=True)
        print(f"[RECORDING] SUMMARY", flush=True)
        print(f"{'='*80}", flush=True)

        # Count files BEFORE pruning
        input_files = sorted(list(self.input_frames_dir.glob("state_*.txt")))
        isaac_files = sorted(list(self.isaac_frames_dir.glob("isaac_*.png")))
        camera_files = sorted(list(self.robot_camera_dir.glob("camera_*.png")))

        input_count_before = len(input_files)
        isaac_count_before = len(isaac_files)
        camera_count_before = len(camera_files)

        # Prune: keep first 10 and last 10, delete middle
        input_deleted = self._prune_recordings(self.input_frames_dir, "state_*.txt", keep_count=10)
        isaac_deleted = self._prune_recordings(self.isaac_frames_dir, "isaac_*.png", keep_count=10)
        camera_deleted = self._prune_recordings(self.robot_camera_dir, "camera_*.png", keep_count=10)

        print(f"[RECORDING] Input frames: {input_count_before} recorded, {input_deleted} pruned → kept {input_count_before - input_deleted}", flush=True)
        print(f"            Location: {self.input_frames_dir.resolve()}", flush=True)

        print(f"[RECORDING] Isaac Sim frames: {isaac_count_before} recorded, {isaac_deleted} pruned → kept {isaac_count_before - isaac_deleted}", flush=True)
        print(f"            Location: {self.isaac_frames_dir.resolve()}", flush=True)

        print(f"[RECORDING] Robot camera frames: {camera_count_before} recorded, {camera_deleted} pruned → kept {camera_count_before - camera_deleted}", flush=True)
        print(f"            Location: {self.robot_camera_dir.resolve()}", flush=True)

        print(f"[RECORDING] Command log (CSV):", flush=True)
        print(f"            {self.command_log_file.resolve()}", flush=True)

        print(f"[RECORDING] All media in: {self.media_dir.resolve()}", flush=True)
        print(f"{'='*80}\n", flush=True)
