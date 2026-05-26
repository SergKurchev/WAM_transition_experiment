"""WAM inference entry point.

Runs a control loop:
  1. Read robot state from Isaac Sim via rt/lowstate (DDS).
  2. Run WAM model inference → velocity command.
  3. Publish rt/run_command/cmd → GEAR-SONIC drives the robot.

To swap models set WAM_MODEL env var: unifolm | eva
"""

import os
import time
import signal

from dds_interface import DDSInterface
from recording import MediaRecorder

STATUS_LOG_INTERVAL = 5.0  # seconds between periodic status lines


# ── model registry ────────────────────────────────────────────────────────────

def load_model(name: str, checkpoint: str | None, prompt: str | None = None):
    """Return an inference callable: fn(state) -> (vx, vy, wz)."""
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
    last_cmd = (0.0, 0.0, 0.0)

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

            # Record input state
            input_frame_path = recorder.save_input_frame(loop_count, state)

            result = model(state)
            vx, vy, wz, body_height = result[0], result[1], result[2], result[3]
            video_output = result[4] if len(result) > 4 else None

            dds.send_command(vx, vy, wz, body_height=body_height)
            last_cmd = (vx, vy, wz)

            # Record command output
            recorder.save_command(loop_count, vx, vy, wz, body_height)

            # Record model video output if available
            if video_output is not None:
                recorder.save_model_output(loop_count, video_output)

            # Try to save Isaac Sim frame (if available)
            isaac_frame_path = recorder.save_isaac_frame(loop_count)

            # Try to save robot camera frame (if available)
            camera_frame_path = recorder.save_robot_camera_frame(loop_count)

            loop_count += 1

            now = time.time()
            if now - last_status_t >= STATUS_LOG_INTERVAL:
                age = now - state.timestamp
                print(
                    f"[WAM] loop={loop_count}  cmd=[vx={last_cmd[0]:.2f} vy={last_cmd[1]:.2f} wz={last_cmd[2]:.2f}]"
                    f"  state_age={age*1000:.0f}ms  q0={state.q[0]:.3f}",
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
