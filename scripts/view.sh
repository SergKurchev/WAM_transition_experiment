#!/usr/bin/env bash
# Open SSH tunnel to view Isaac Sim via noVNC in your local browser.
# Isaac Sim renders to a virtual display (Xvfb) inside the container.
# x11vnc + websockify + noVNC serve it on port 6180 (noVNC web).
#
# Usage:
#   ./scripts/view.sh                    # tunnel to GPU-01
#   ./scripts/view.sh x32-techgov-GPU-02 # tunnel to GPU-02
#
# Then open: http://localhost:6180

set -euo pipefail

HOST="${1:-x32-techgov-GPU-01}"
LOCAL_PORT="${2:-6180}"
REMOTE_PORT=6180

echo "Opening noVNC tunnel: localhost:${LOCAL_PORT} → ${HOST}:${REMOTE_PORT}"
echo "Open your browser at: http://localhost:${LOCAL_PORT}"
echo "(Press Ctrl+C to close the tunnel)"
echo ""

# -N: no remote command (tunnel only)
# -L: local port forward
ssh -N -L "${LOCAL_PORT}:localhost:${REMOTE_PORT}" "${HOST}"
