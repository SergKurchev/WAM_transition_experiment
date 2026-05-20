"""Prepare the preconverted G1 robot USD used by the Isaac runtime.

This script is intentionally separate from ``launch_g1.py`` and ``g1_sim.py``:
runtime only consumes a USD asset, while this script owns URDF conversion,
material patching, and sensor-frame validation.
"""
from __future__ import annotations

import argparse
import copy
import os
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from isaaclab.app import AppLauncher


_D435I_USD_FRAME = "/g1/torso_link/d435_link"
_D435I_OPTICAL_USD_FRAME = f"{_D435I_USD_FRAME}/d435_optical_frame"
_MID360_USD_FRAME = "/g1/torso_link/mid360_sensor_frame"
_IMU_IN_TORSO_USD_FRAME = "/g1/torso_link/imu_in_torso"
_IMU_IN_PELVIS_USD_FRAME = "/g1/pelvis/imu_in_pelvis"
_D435I_FRAME_ROT_WXYZ = (0.9149596678498247, 0.0, 0.40354529635239006, 0.0)
_D435I_OPTICAL_FRAME_ROT_WXYZ = (
    0.5000000000000001,
    -0.5,
    0.4999999999999999,
    -0.5,
)
_MID360_FRAME_ROT_WXYZ = (0.9997985784932998, 0.0, 0.020069938783589012, 0.0)


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments before Isaac app startup."""
    parser = argparse.ArgumentParser(description="Convert the configured G1 URDF to runtime USD.")
    parser.add_argument(
        "--robot-usd",
        required=True,
        help="Output path for the preconverted G1 robot USD.",
    )
    parser.add_argument(
        "--robot-color-mjcf",
        default=None,
        help="Optional MJCF file whose per-mesh rgba values are applied to converted G1 USD visuals.",
    )
    AppLauncher.add_app_launcher_args(parser)
    return parser.parse_args()


def _probe(message: str) -> None:
    """Emit a converter log line."""
    print(f"[prepare_g1_robot_usd::probe] {message}", flush=True)


def _resolve_g1_urdf_asset_path(cfg: Any) -> None:
    """Rewrite gear_sonic's CWD-relative URDF path to an absolute path."""
    if not os.path.isabs(cfg.spawn.asset_path):
        import gear_sonic

        gwbc_root = Path(gear_sonic.__file__).resolve().parents[1]
        cfg.spawn.asset_path = str(gwbc_root / cfg.spawn.asset_path)


def _default_g1_mjcf_path() -> str:
    """Return the GWBC MJCF path whose mesh colors match the MuJoCo visual."""
    import gear_sonic

    gwbc_root = Path(gear_sonic.__file__).resolve().parents[1]
    candidates = [
        gwbc_root / "gear_sonic_deploy/g1/g1_29dof_with_hand.xml",
        gwbc_root / "gear_sonic/data/assets/robot_description/mjcf/g1_29dof_rev_1_0.xml",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    return str(candidates[0])


def _load_mjcf_mesh_colors(mjcf_path: str) -> dict[str, tuple[float, float, float]]:
    """Load first authored per-mesh RGB color from a MuJoCo XML file."""
    root = ET.parse(mjcf_path).getroot()
    mesh_colors: dict[str, tuple[float, float, float]] = {}
    for geom in root.iter("geom"):
        mesh_name = geom.attrib.get("mesh")
        rgba = geom.attrib.get("rgba")
        if not mesh_name or not rgba or mesh_name in mesh_colors:
            continue
        values = rgba.split()
        if len(values) < 3:
            continue
        mesh_colors[mesh_name] = (float(values[0]), float(values[1]), float(values[2]))
    return mesh_colors


def _apply_mjcf_colors_to_usd(robot_usd_path: str, mjcf_path: str) -> None:
    """Patch converted USD mesh materials to match MuJoCo per-geom colors."""
    from pxr import Gf, Sdf, Usd, UsdShade

    mesh_colors = _load_mjcf_mesh_colors(mjcf_path)
    if not mesh_colors:
        _probe(f"WARNING: no mesh colors found in MJCF: {mjcf_path}")
        return

    robot_usd = Path(robot_usd_path)
    base_usd = robot_usd.with_name("configuration") / f"{robot_usd.stem}_base.usd"
    if not base_usd.is_file():
        _probe(f"WARNING: converted base USD not found for color patch: {base_usd}")
        return

    stage = Usd.Stage.Open(str(base_usd))
    if stage is None:
        _probe(f"WARNING: failed to open converted base USD for color patch: {base_usd}")
        return

    patched = 0
    missing: set[str] = set()
    for prim in stage.Traverse():
        if prim.GetTypeName() != "Shader":
            continue
        path_parts = prim.GetPath().pathString.split("/")
        try:
            mesh_name = path_parts[path_parts.index("meshes") + 1]
        except (ValueError, IndexError):
            continue
        color = mesh_colors.get(mesh_name)
        if color is None:
            missing.add(mesh_name)
            continue
        shader = UsdShade.Shader(prim)
        shader.CreateInput("diffuse_color_constant", Sdf.ValueTypeNames.Color3f).Set(
            Gf.Vec3f(*color)
        )
        patched += 1

    stage.GetRootLayer().Save()
    _probe(
        f"applied MuJoCo colors to converted G1 USD: {patched} shader(s) patched "
        f"from {mjcf_path}"
    )
    if missing:
        _probe(f"WARNING: {len(missing)} USD mesh material(s) had no MJCF color")


def _assert_usd_frame_translation(
    stage: Any,
    prim_path: str,
    expected: tuple[float, float, float],
) -> None:
    """Assert that a converted USD frame exists with the expected local translation."""
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise RuntimeError(f"Converted G1 USD is missing required sensor frame: {prim_path}")
    translate_attr = prim.GetAttribute("xformOp:translate")
    if not translate_attr.IsValid():
        raise RuntimeError(f"Converted G1 USD frame has no local translate op: {prim_path}")
    actual = tuple(float(value) for value in translate_attr.Get())
    max_error = max(
        abs(actual_value - expected_value)
        for actual_value, expected_value in zip(actual, expected)
    )
    if max_error > 1e-5:
        raise RuntimeError(
            f"Converted G1 USD frame {prim_path} has translation {actual}, expected {expected}"
        )


def _assert_usd_frame_rotation(
    stage: Any,
    prim_path: str,
    expected_wxyz: tuple[float, float, float, float],
) -> None:
    """Assert that a converted USD frame has the expected local orientation."""
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise RuntimeError(f"Converted G1 USD is missing required sensor frame: {prim_path}")
    orient_attr = prim.GetAttribute("xformOp:orient")
    if not orient_attr.IsValid():
        raise RuntimeError(f"Converted G1 USD frame has no local orient op: {prim_path}")
    quat = orient_attr.Get()
    imag = quat.GetImaginary()
    actual = (float(quat.GetReal()), float(imag[0]), float(imag[1]), float(imag[2]))
    direct_error = max(
        abs(actual_value - expected_value)
        for actual_value, expected_value in zip(actual, expected_wxyz)
    )
    negated_error = max(
        abs(actual_value + expected_value)
        for actual_value, expected_value in zip(actual, expected_wxyz)
    )
    if min(direct_error, negated_error) > 1e-5:
        raise RuntimeError(
            f"Converted G1 USD frame {prim_path} has rotation {actual}, expected {expected_wxyz}"
        )


def _validate_converted_g1_sensor_frames(robot_usd_path: str) -> None:
    """Validate generated USD frames that Isaac sensors attach to at runtime."""
    from pxr import Usd

    stage = Usd.Stage.Open(robot_usd_path)
    if stage is None:
        raise RuntimeError(f"Failed to open converted G1 USD for sensor-frame validation: {robot_usd_path}")

    _assert_usd_frame_translation(stage, _D435I_USD_FRAME, (0.0576235, 0.01753, 0.41987))
    _assert_usd_frame_translation(stage, _D435I_OPTICAL_USD_FRAME, (0.0, 0.0, 0.0))
    _assert_usd_frame_translation(stage, _MID360_USD_FRAME, (0.0002835, 0.00003, 0.41618))
    _assert_usd_frame_translation(stage, _IMU_IN_TORSO_USD_FRAME, (-0.03959, -0.00224, 0.14792))
    _assert_usd_frame_translation(stage, _IMU_IN_PELVIS_USD_FRAME, (0.04525, 0.0, -0.08339))
    _assert_usd_frame_rotation(stage, _D435I_USD_FRAME, _D435I_FRAME_ROT_WXYZ)
    _assert_usd_frame_rotation(stage, _D435I_OPTICAL_USD_FRAME, _D435I_OPTICAL_FRAME_ROT_WXYZ)
    _assert_usd_frame_rotation(stage, _MID360_USD_FRAME, _MID360_FRAME_ROT_WXYZ)
    _probe(
        "validated converted G1 sensor frames: "
        f"{_D435I_OPTICAL_USD_FRAME}, {_MID360_USD_FRAME}, "
        f"{_IMU_IN_TORSO_USD_FRAME}, {_IMU_IN_PELVIS_USD_FRAME}"
    )


def convert_g1_urdf_to_usd(output_path: str, color_mjcf: str | None = None) -> None:
    """Convert the configured G1 URDF asset to a persistent USD file."""
    from isaaclab.sim.converters import UrdfConverter

    from gear_sonic.envs.manager_env.robots.g1 import G1_CYLINDER_MODEL_12_DEX_CFG

    cfg = copy.deepcopy(G1_CYLINDER_MODEL_12_DEX_CFG)
    _resolve_g1_urdf_asset_path(cfg)
    output_path = os.path.abspath(output_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cfg.spawn.usd_dir = os.path.dirname(output_path)
    cfg.spawn.usd_file_name = os.path.basename(output_path)
    cfg.spawn.force_usd_conversion = True
    _probe(f"converting G1 URDF to USD: {cfg.spawn.asset_path} -> {output_path}")
    converter = UrdfConverter(cfg.spawn)
    _probe(f"G1 robot USD written: {converter.usd_path}")
    color_source = color_mjcf or _default_g1_mjcf_path()
    if os.path.isfile(color_source):
        _apply_mjcf_colors_to_usd(converter.usd_path, color_source)
    else:
        _probe(f"WARNING: G1 MuJoCo color source not found, leaving USD colors unchanged: {color_source}")
    _validate_converted_g1_sensor_frames(converter.usd_path)


def main() -> None:
    """Run conversion inside an initialized Isaac app."""
    args = _parse_args()
    _probe(f"creating AppLauncher; headless={getattr(args, 'headless', '?')}")
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app
    try:
        convert_g1_urdf_to_usd(args.robot_usd, color_mjcf=args.robot_color_mjcf)
    finally:
        _probe("closing SimulationApp")
        simulation_app.close()


if __name__ == "__main__":
    main()
