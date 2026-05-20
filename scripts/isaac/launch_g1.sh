#!/usr/bin/env bash
set -euo pipefail

if [ ! -f /opt/conda/etc/profile.d/conda.sh ]; then
    echo "[launch_g1] /opt/conda/etc/profile.d/conda.sh not found — wrong image?" >&2
    exit 1
fi

source /opt/conda/etc/profile.d/conda.sh
conda activate unitree_sim_env

cd /workspace/wam-stack

SCENE_USD_ARG=()
if [ -n "${SCENE_FILE:-}" ]; then
    SCENE_USD_ARG=(--scene-usd "/workspace/scene_assets/${SCENE_FILE}")
fi

ROBOT_USD_ARG=()
if [ -n "${ROBOT_USD_FILE:-}" ]; then
    ROBOT_USD_ARG=(--robot-usd "/workspace/robot_assets/${ROBOT_USD_FILE}")
fi

exec python sim/isaac/launch_g1.py "${SCENE_USD_ARG[@]}" "${ROBOT_USD_ARG[@]}" "$@"
