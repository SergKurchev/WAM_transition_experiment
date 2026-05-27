"""WAM inference entry point.

Runs a control loop:
  1. Read robot state from Isaac Sim via rt/lowstate (DDS).
  2. Run WAM model inference → trajectory predictions (16, 14).
  3. Record model outputs: action trajectories, state predictions, video frames.

To swap models set WAM_MODEL env var: unifolm | eva
"""

import os
import time
import signal
import numpy as np
import torch

from dds_interface import DDSInterface
from recording import MediaRecorder


# unifolm_wma is installed during Docker build (cloned from GitHub and pip install -e)

STATUS_LOG_INTERVAL = 5.0  # seconds between periodic status lines


# ── model registry ────────────────────────────────────────────────────────────

def load_model(name: str, checkpoint: str | None, prompt: str | None = None):
    """Return an inference callable: fn(state) -> (action_traj, state_traj, video_output).

    action_traj: (16, 14) tensor - 16 timesteps × 14 DOF (both arms)
    state_traj: (16, 14) tensor - world model state predictions
    video_output: optional video frames from diffusion model
    """
    if name == "unifolm":
        from models.unifolm import UnifoLMModel
        return UnifoLMModel(checkpoint, prompt=prompt)
    elif name == "eva":
        from models.eva import EVAModel
        return EVAModel(checkpoint)
    elif name == "stub":
        return _stub_model
    else:
        raise ValueError(f"Unknown WAM_MODEL: {name!r}. Set to 'unifolm', 'eva', or 'stub'.")


def _stub_model(state):
    """Placeholder: walk slowly forward. Replace with real model."""
    return 0.15, 0.0, 0.0, 0.0   # vx, vy, wz, body_height


# ── main loop ─────────────────────────────────────────────────────────────────

CONTROL_HZ = 10   # command publish rate


def main():
    model_name = os.environ.get("WAM_MODEL", "stub")
    checkpoint  = os.environ.get("WAM_CHECKPOINT") or None
    prompt      = os.environ.get("WAM_PROMPT", "default")
    dds_iface   = os.environ.get("DDS_IFACE", "lo")
    media_dir   = os.environ.get("WAM_MEDIA_DIR", "/workspace/wam/media")

    print(f"[WAM] starting  model={model_name}  checkpoint={checkpoint}  prompt={prompt}  iface={dds_iface}", flush=True)

    dds = DDSInterface(iface=dds_iface)
    dds.init()

    recorder = MediaRecorder(media_dir=media_dir, prompt=prompt)

    model = load_model(model_name, checkpoint, prompt=prompt)
    print(f"[WAM] model loaded ({model_name}). Waiting for rt/lowstate from Isaac Sim...", flush=True)

    wait_start = time.time()
    last_wait_log = wait_start
    while dds.get_state() is None:
        now = time.time()
        if now - last_wait_log >= 5.0:
            print(f"[WAM] still waiting for rt/lowstate  ({now - wait_start:.0f}s elapsed)...", flush=True)
            last_wait_log = now
        time.sleep(0.1)

    print(f"[WAM] rt/lowstate received  ({time.time() - wait_start:.1f}s wait). Control loop starting at {CONTROL_HZ} Hz.", flush=True)

    dt = 1.0 / CONTROL_HZ
    loop_count = 0
    last_status_t = time.time()

    def graceful_shutdown(signum, frame):
        """Handle Ctrl+C to finalize recording."""
        print(f"\n[WAM] received SIGINT, finalizing recording...", flush=True)
        recorder.finalize()
        exit(0)

    signal.signal(signal.SIGINT, graceful_shutdown)

    try:
        while True:
            t0 = time.time()

            state = dds.get_state()

            # ── 1. Записываем ВХОДЫ модели ────────────────────────────────
            recorder.save_input_frame(loop_count, state)       # state_input.txt
            recorder.save_robot_camera_frame(loop_count)       # camera_input.png
            recorder.save_isaac_frame(loop_count)              # isaac_input.png (если есть)

            # ── 2. Inference ──────────────────────────────────────────────
            result = model(state)
            if isinstance(result, tuple) and len(result) >= 3:
                action_traj, state_traj, video_output = result
            else:
                print(f"[WAM] ERROR: unexpected model result: {result}", flush=True)
                action_traj, state_traj, video_output = None, None, None

            # ── 3. Записываем ВЫХОД модели ────────────────────────────────
            if video_output is not None:
                recorder.save_model_output(loop_count, video_output)  # model_output.mp4

            # ── 4. Метрики ────────────────────────────────────────────────
            if action_traj is not None and hasattr(action_traj, '__len__') and len(action_traj) > 0:
                if isinstance(action_traj, torch.Tensor):
                    action_0      = action_traj[0].cpu().numpy()
                    action_traj_np = action_traj.cpu().numpy()
                else:
                    action_0      = action_traj[0]
                    action_traj_np = action_traj
                action_0_norm  = float((action_0      ** 2).sum() ** 0.5)
                action_traj_norm = float((action_traj_np ** 2).sum() ** 0.5)
            else:
                action_0       = np.zeros(14)
                action_0_norm  = 0.0
                action_traj_norm = 0.0

            # CSV лог
            recorder.save_command(loop_count, action_0_norm, action_traj_norm)

            loop_count += 1

            # ── 5. Периодический status-print ─────────────────────────────
            now = time.time()
            if now - last_status_t >= STATUS_LOG_INTERVAL:
                age = now - state.timestamp
                sample = f"{action_0[0]:+.3f} {action_0[1]:+.3f} {action_0[2]:+.3f}"
                print(
                    f"[WAM] loop={loop_count}"
                    f"  action_0[0:3]=[{sample}]"
                    f"  norm={action_0_norm:.3f}"
                    f"  traj_norm={action_traj_norm:.3f}"
                    f"  state_age={age*1000:.0f}ms"
                    f"  q0={state.q[0]:.3f}",
                    flush=True,
                )
                last_status_t = now

            elapsed = time.time() - t0
            time.sleep(max(0.0, dt - elapsed))
    except KeyboardInterrupt:
        recorder.finalize()
        raise


if __name__ == "__main__":
    main()
