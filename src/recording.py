"""Recording module for WAM inference.

Each inference step gets its own subfolder:

    media/
    ├── steps/
    │   ├── step_000001/
    │   │   ├── camera_input.png   ← D435i кадр (вход модели)
    │   │   ├── state_input.txt    ← суставные углы/скорости (вход модели)
    │   │   ├── model_output.mp4   ← видео предсказание (выход модели, 16 кадров)
    │   │   └── isaac_input.png    ← скриншот Isaac Sim (если доступен)
    │   ├── step_000002/
    │   │   └── ...
    │   └── ...  (хранятся только первые 10 + последние 10 шагов)
    └── command_logs/
        └── commands_YYYYMMDD_HHMMSS.csv

Автопрунинг: после каждого сохранения model_output.mp4 удаляются средние шаги,
остаются только первые KEEP_FIRST и последние KEEP_LAST папок.
"""

import os
import csv
import shutil
import time
from pathlib import Path
from datetime import datetime

try:
    import numpy as np
except ImportError:
    np = None

try:
    from PIL import Image
except ImportError:
    Image = None

KEEP_FIRST = 10   # сколько первых шагов хранить
KEEP_LAST  = 10   # сколько последних шагов хранить


class MediaRecorder:
    """Запись входов и выходов WAM inference, сгруппированная по шагам."""

    def __init__(self, media_dir: str = "/workspace/wam/media", prompt: str = "default"):
        self.media_dir = Path(media_dir)
        self.prompt = prompt

        # Директории
        self.steps_dir       = self.media_dir / "steps"
        self.command_logs_dir = self.media_dir / "command_logs"

        self.steps_dir.mkdir(parents=True, exist_ok=True)
        self.command_logs_dir.mkdir(parents=True, exist_ok=True)

        # Метаданные сессии
        meta = self.media_dir / "session.txt"
        with open(meta, "w") as f:
            f.write(f"started:  {datetime.now().isoformat()}\n")
            f.write(f"prompt:   {self.prompt}\n")
            f.write(f"keeps:    first {KEEP_FIRST} + last {KEEP_LAST} steps\n")
            f.write(f"camera:   D435i /run/mws/camera.rgb (1280×720 RGB)\n")

        # CSV лог команд
        self.command_log_file = (
            self.command_logs_dir
            / f"commands_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        )
        self._init_command_log()

        # Shared memory камера
        self._camera_shm_path = "/run/mws/camera.rgb"
        self._camera_shm_size = 1280 * 720 * 3
        self._camera_shm_map  = None
        self._init_camera_shm()

        print(f"[RECORDING] steps dir:   {self.steps_dir}", flush=True)
        print(f"[RECORDING] command log: {self.command_log_file}", flush=True)
        print(f"[RECORDING] pruning:     keep first {KEEP_FIRST} + last {KEEP_LAST} steps", flush=True)

    # ── internal helpers ──────────────────────────────────────────────────────

    def _step_dir(self, step: int) -> Path:
        """Вернуть (и создать) папку для шага step."""
        d = self.steps_dir / f"step_{step:06d}"
        d.mkdir(exist_ok=True)
        return d

    def _init_camera_shm(self) -> None:
        if np is None:
            return
        try:
            if not os.path.exists(self._camera_shm_path):
                print(f"[RECORDING] Camera SHM not found at {self._camera_shm_path}", flush=True)
                return
            self._camera_shm_map = open(self._camera_shm_path, "r+b")
            print(f"[RECORDING] ✓ Camera SHM: {self._camera_shm_path} (D435i, 1280×720 RGB)", flush=True)
        except Exception as e:
            print(f"[RECORDING] Camera SHM warning: {e}", flush=True)

    def _init_command_log(self) -> None:
        with open(self.command_log_file, "w", newline="") as f:
            csv.writer(f).writerow(["timestamp", "step", "action_norm", "traj_norm"])

    # ── public save API ───────────────────────────────────────────────────────

    def save_input_frame(self, step: int, state) -> str | None:
        """Сохранить суставное состояние робота (вход модели).

        Файл: steps/step_XXXXXX/state_input.txt
        """
        try:
            dest = self._step_dir(step) / "state_input.txt"
            with open(dest, "w") as f:
                f.write(f"step:      {step}\n")
                f.write(f"timestamp: {state.timestamp}\n\n")
                f.write(f"q  (joint positions,  {len(state.q)} DOF):\n")
                f.write("  " + "  ".join(f"{v:+.4f}" for v in state.q) + "\n\n")
                f.write(f"dq (joint velocities, {len(state.dq)} DOF):\n")
                f.write("  " + "  ".join(f"{v:+.4f}" for v in state.dq) + "\n\n")
                f.write(f"tau (joint torques,   {len(state.tau)} DOF):\n")
                f.write("  " + "  ".join(f"{v:+.4f}" for v in state.tau) + "\n")
            return str(dest)
        except Exception as e:
            print(f"[RECORDING] save_input_frame failed: {e}", flush=True)
            return None

    def save_robot_camera_frame(self, step: int) -> str | None:
        """Сохранить кадр с D435i (вход модели).

        Файл: steps/step_XXXXXX/camera_input.png
        """
        if np is None or self._camera_shm_map is None:
            return None
        try:
            self._camera_shm_map.seek(0)
            raw = self._camera_shm_map.read(self._camera_shm_size)
            if len(raw) < self._camera_shm_size:
                return None
            rgb = np.frombuffer(raw, dtype=np.uint8).reshape(720, 1280, 3)

            dest = self._step_dir(step) / "camera_input.png"
            if Image is not None:
                Image.fromarray(rgb, mode="RGB").save(str(dest))
            else:
                np.save(str(dest.with_suffix(".npy")), rgb)
            return str(dest)
        except Exception:
            return None

    def save_isaac_frame(self, step: int) -> str | None:
        """Скопировать скриншот Isaac Sim редактора (если есть).

        Файл: steps/step_XXXXXX/isaac_input.png
        """
        src = Path("/tmp/isaac_frame.png")
        if not src.exists():
            return None
        try:
            dest = self._step_dir(step) / "isaac_input.png"
            shutil.copy2(src, dest)
            return str(dest)
        except Exception as e:
            print(f"[RECORDING] save_isaac_frame failed: {e}", flush=True)
            return None

    def save_model_output(self, step: int, video_output) -> str | None:
        """Сохранить видео предсказание модели (выход модели).

        Файл: steps/step_XXXXXX/model_output.mp4
        Формат входа: тензор (B,C,T,H,W) или (C,T,H,W) в диапазоне [-1,1].

        После сохранения запускает прунинг: оставляет первые KEEP_FIRST
        и последние KEEP_LAST шагов, удаляет средние.
        """
        if video_output is None:
            return None

        try:
            import cv2 as _cv2
            import torch

            # 1. → float32 numpy
            if isinstance(video_output, torch.Tensor):
                arr = video_output.detach().cpu().float().numpy()
            else:
                arr = np.array(video_output, dtype=np.float32)

            # 2. → (T, H, W, C)
            if arr.ndim == 5:
                arr = arr[0]                          # (B,C,T,H,W) → (C,T,H,W)
            if arr.ndim == 4:
                if arr.shape[0] in (1, 3, 4) and arr.shape[0] < arr.shape[1]:
                    arr = arr.transpose(1, 2, 3, 0)   # (C,T,H,W) → (T,H,W,C)

            # 3. → uint8 [0, 255]
            if arr.min() >= -1.5 and arr.max() <= 1.5:
                arr = (arr + 1.0) / 2.0 * 255.0
            elif arr.max() <= 1.0:
                arr = arr * 255.0
            arr = np.clip(arr, 0, 255).astype(np.uint8)
            arr = np.ascontiguousarray(arr)

            # 4. Записать MP4 (OpenCV работает в BGR)
            dest = self._step_dir(step) / "model_output.mp4"
            T, H, W, C = arr.shape
            fourcc = _cv2.VideoWriter_fourcc(*"mp4v")
            writer = _cv2.VideoWriter(str(dest), fourcc, 10.0, (W, H))
            for frame in arr:
                if C == 3:
                    bgr = np.ascontiguousarray(frame[:, :, ::-1])
                elif C == 1:
                    bgr = _cv2.cvtColor(frame, _cv2.COLOR_GRAY2BGR)
                else:
                    bgr = np.ascontiguousarray(frame[:, :, :3][:, :, ::-1])
                writer.write(bgr)
            writer.release()

            print(f"[RECORDING] step {step:06d} → {dest}", flush=True)

            # 5. Прунинг: держим только первые + последние
            self._prune_steps()

            return str(dest)

        except Exception as e:
            print(f"[RECORDING] save_model_output failed: {e}", flush=True)
            import traceback
            traceback.print_exc()
            return None

    def save_command(self, step: int, action_norm: float = 0.0, traj_norm: float = 0.0) -> None:
        """Добавить строку в CSV лог."""
        try:
            with open(self.command_log_file, "a", newline="") as f:
                csv.writer(f).writerow([
                    datetime.now().isoformat(), step,
                    f"{action_norm:.4f}", f"{traj_norm:.4f}",
                ])
        except Exception:
            pass

    # ── pruning ───────────────────────────────────────────────────────────────

    def _prune_steps(self) -> int:
        """Удалить средние шаги, оставив первые KEEP_FIRST и последние KEEP_LAST.

        Returns:
            Количество удалённых папок.
        """
        dirs = sorted(
            d for d in self.steps_dir.iterdir()
            if d.is_dir() and d.name.startswith("step_")
        )
        total = len(dirs)
        if total <= KEEP_FIRST + KEEP_LAST:
            return 0

        to_delete = dirs[KEEP_FIRST : total - KEEP_LAST]
        deleted = 0
        for d in to_delete:
            try:
                shutil.rmtree(d)
                deleted += 1
            except Exception:
                pass

        if deleted:
            remaining = total - deleted
            print(
                f"[RECORDING] pruned {deleted} steps "
                f"(kept first {KEEP_FIRST} + last {KEEP_LAST}, "
                f"{remaining} remain)",
                flush=True,
            )
        return deleted

    # ── finalize ──────────────────────────────────────────────────────────────

    def finalize(self) -> None:
        """Итоговый прунинг + сводка."""
        print(f"\n{'='*70}", flush=True)
        print(f"[RECORDING] FINAL SUMMARY", flush=True)
        print(f"{'='*70}", flush=True)

        dirs = sorted(
            d for d in self.steps_dir.iterdir()
            if d.is_dir() and d.name.startswith("step_")
        )
        total_before = len(dirs)
        deleted = self._prune_steps()
        remaining = total_before - deleted

        print(f"[RECORDING] Steps recorded: {total_before}", flush=True)
        print(f"[RECORDING] Steps pruned:   {deleted}", flush=True)
        print(f"[RECORDING] Steps kept:     {remaining}", flush=True)
        if dirs:
            kept = sorted(
                d for d in self.steps_dir.iterdir()
                if d.is_dir() and d.name.startswith("step_")
            )
            if kept:
                print(f"[RECORDING] First step: {kept[0].name}", flush=True)
                print(f"[RECORDING] Last step:  {kept[-1].name}", flush=True)

        print(f"[RECORDING] Steps dir:    {self.steps_dir}", flush=True)
        print(f"[RECORDING] Command log:  {self.command_log_file}", flush=True)
        print(f"{'='*70}\n", flush=True)
