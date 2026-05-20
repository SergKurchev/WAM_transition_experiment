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

    print(f"[main] WAM_MODEL={model_name}  checkpoint={checkpoint}  DDS_IFACE={dds_iface}", flush=True)

    dds = DDSInterface(iface=dds_iface)
    dds.init()

    model = load_model(model_name, checkpoint)
    print("[main] Model loaded. Waiting for first robot state...", flush=True)

    # Wait until Isaac Sim publishes at least one lowstate
    while dds.get_state() is None:
        time.sleep(0.1)
    print("[main] Robot state received. Starting control loop.", flush=True)

    dt = 1.0 / CONTROL_HZ
    while True:
        t0 = time.time()

        state = dds.get_state()
        vx, vy, wz = model(state)
        dds.send_command(vx, vy, wz, body_height=0.0)

        elapsed = time.time() - t0
        time.sleep(max(0.0, dt - elapsed))


if __name__ == "__main__":
    main()
