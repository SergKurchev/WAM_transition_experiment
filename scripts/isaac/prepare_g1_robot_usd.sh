#!/usr/bin/env bash
# Prepare the preconverted G1 robot USD inside the shared Isaac Sim / Isaac Lab env.

set -euo pipefail

if [ ! -f /opt/conda/etc/profile.d/conda.sh ]; then
    echo "[prepare_g1_robot_usd] /opt/conda/etc/profile.d/conda.sh not found; wrong image?" >&2
    exit 1
fi

# shellcheck disable=SC1091
source /opt/conda/etc/profile.d/conda.sh
conda activate unitree_sim_env

cd /workspace/mws-dimos

exec python sim/isaac/prepare_g1_robot_usd.py "$@"
