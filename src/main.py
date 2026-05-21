"""WAM inference entry point.

Runs a control loop:
  1. Read robot state from Isaac Sim via rt/lowstate (DDS).
  2. Run WAM model inference → velocity command.
  3. Publish rt/run_command/cmd → GEAR-SONIC drives the robot.

To swap models set WAM_MODEL env var: unifolm | eva
"""

import os
import time

from dds_interface import DDSInterface

STATUS_LOG_INTERVAL = 5.0  # seconds between periodic status lines


# ── model registry ────────────────────────────────────────────────────────────

def load_model(name: str, checkpoint: str | None):
    """Return an inference callable: fn(state) -> (vx, vy, wz)."""
    if name == "unifolm":
        from models.unifolm import UnifoLMModel
        return UnifoLMModel(checkpoint)
    elif name == "eva":
        from models.eva import EVAModel
        return EVAModel(checkpoint)
    elif name == "stub":
        return _stub_model
    else:
        raise ValueError(f"Unknown WAM_MODEL: {name!r}. Set to 'unifolm', 'eva', or 'stub'.")


def _stub_model(state):
    """Placeholder: walk slowly forward. Replace with real model."""
    return 0.15, 0.0, 0.0   # vx, vy, wz


# ── main loop ─────────────────────────────────────────────────────────────────

CONTROL_HZ = 10   # command publish rate


def main():
    model_name = os.environ.get("WAM_MODEL", "stub")
    checkpoint  = os.environ.get("WAM_CHECKPOINT") or None
    dds_iface   = os.environ.get("DDS_IFACE", "lo")

    print(f"[WAM] starting  model={model_name}  checkpoint={checkpoint}  iface={dds_iface}", flush=True)

    dds = DDSInterface(iface=dds_iface)
    dds.init()

    model = load_model(model_name, checkpoint)
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

    while True:
        t0 = time.time()

        state = dds.get_state()
        vx, vy, wz = model(state)
        dds.send_command(vx, vy, wz, body_height=0.0)
        last_cmd = (vx, vy, wz)
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


if __name__ == "__main__":
    main()
