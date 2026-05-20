"""Entry point for the slim G1 Isaac Sim runtime.

Two-phase imports are mandatory: ``AppLauncher`` must start the ``SimulationApp``
before any ``isaaclab.*`` / scene modules are imported. See Isaac Lab docs.
"""
from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Slim Isaac Sim runtime for the G1 robot.")
    parser.add_argument(
        "--render_interval",
        type=int,
        default=None,
        help=(
            "Render every N env steps in GUI mode. Defaults to 10 when unset. "
            "Headless mode uses 1 when unset."
        ),
    )
    parser.add_argument(
        "--no_render",
        action="store_true",
        default=False,
        help="Disable rendering entirely, even when a GUI is available.",
    )
    parser.add_argument(
        "--publish-rate-hz",
        type=float,
        default=100.0,
        help=(
            "rt/lowstate publish rate in Hz (sim time). Physics sub-step rate is "
            "200 Hz; the effective cadence is round(200/rate_hz) sub-steps. Default "
            "100 publishes every other sub-step. The 500 ms gear-sonic CONTROL "
            "watchdog tolerates ~50 missed publishes at this rate before tripping; "
            "wall-clock margin scales with the sim's real-time factor. Raise toward "
            "200 only if the watchdog actually trips. Values >200 clamp to 200."
        ),
    )
    parser.add_argument(
        "--dds-domain-id",
        type=int,
        default=0,
        help="CycloneDDS domain id. Must match gear-sonic (0).",
    )
    parser.add_argument(
        "--dds-interface",
        type=str,
        default="lo",
        help="Network interface for DDS multicast. 'lo' matches gear-sonic's host-mode setup.",
    )
    parser.add_argument(
        "--scene-usd",
        type=str,
        default=None,
        help="Absolute path to a USD file to load as a scene reference at /World/Scene.",
    )
    parser.add_argument(
        "--robot-usd",
        type=str,
        required=True,
        help="Absolute path to a preconverted G1 robot USD file.",
    )
    AppLauncher.add_app_launcher_args(parser)
    return parser.parse_args()


def _publish_every_n_sub_steps(rate_hz: float, sub_step_hz: float = 200.0) -> int:
    """Translate a desired publish rate (Hz, sim time) to a sub-step cadence.

    Physics sub-step rate is ``1 / SIM_DT = 200 Hz`` in g1_sim.py. The returned
    integer N means "publish once every N physics sub-steps", so the effective
    publish rate is ``sub_step_hz / N`` in sim time (the wall-clock rate is
    scaled by how fast the sim loop runs).
    """
    if rate_hz <= 0:
        raise ValueError(f"publish-rate-hz must be positive, got {rate_hz}")
    return max(1, round(sub_step_hz / rate_hz))


def main() -> None:
    args = _parse_args()
    # G1SceneCfg always includes a CameraCfg; AppLauncher requires enable_cameras=True
    # or it raises at sim.reset() time.  Force it on regardless of CLI input.
    args.enable_cameras = True
    print(f"[launch_g1::probe] parsed args; headless={getattr(args, 'headless', '?')}", flush=True)
    print("[launch_g1::probe] creating AppLauncher", flush=True)
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app
    print(
        f"[launch_g1::probe] SimulationApp alive; is_running={simulation_app.is_running()}",
        flush=True,
    )

    # Imports that pull in isaaclab.scene / omni must happen AFTER SimulationApp is alive.
    from sim.isaac.g1_sim import run

    try:
        print("[launch_g1::probe] calling run()", flush=True)
        run(
            simulation_app,
            publish_every_n_sub_steps=_publish_every_n_sub_steps(args.publish_rate_hz),
            dds_domain_id=args.dds_domain_id,
            dds_interface=args.dds_interface,
            headless=bool(getattr(args, "headless", False)),
            no_render=args.no_render,
            render_interval=args.render_interval,
            scene_usd=args.scene_usd,
            robot_usd=args.robot_usd,
        )
        print("[launch_g1::probe] run() returned normally", flush=True)
    except BaseException as exc:
        print(f"[launch_g1::probe] run() raised: {type(exc).__name__}: {exc}", flush=True)
        raise
    finally:
        print("[launch_g1::probe] closing SimulationApp", flush=True)
        simulation_app.close()


if __name__ == "__main__":
    main()
