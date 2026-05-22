#!/usr/bin/env bash
# walk.sh — send a velocity command to GEAR-SONIC via DDS rt/run_command/cmd.
#
# Usage:
#   bash scripts/walk.sh forward              # vx=0.3 m/s, 6s
#   bash scripts/walk.sh back                 # vx=-0.3 m/s, 6s
#   bash scripts/walk.sh yaw                  # wz=0.4 rad/s, 6s
#   bash scripts/walk.sh stop                 # vx=vy=wz=0
#   bash scripts/walk.sh custom --vx 0.2 --wz 0.3 --duration 10

set -euo pipefail

SERVER_HOST="176.109.83.84"
SERVER_PORT="2221"
SERVER_USER="root"
REMOTE="$SERVER_USER@$SERVER_HOST"
MWS_DIR="/root/skurchev/workspace/mws-dimos"
SSH_KEY="$HOME/.ssh/id_ed25519"
UV="$HOME/.cache/uv/archive-v0/Ut-jV_1sYOe6u-ol2xGx5/uv-0.9.7.data/scripts/uv"

G='\033[0;32m'; Y='\033[1;33m'; R='\033[0;31m'; N='\033[0m'

# WSL fallback
if [ ! -f "$SSH_KEY" ]; then
    WIN_HOME=$(wslpath "$(cmd.exe /c 'echo %USERPROFILE%' 2>/dev/null | tr -d '\r')" 2>/dev/null || true)
    [ -n "$WIN_HOME" ] && SSH_KEY="$WIN_HOME/.ssh/id_ed25519"
fi

if ! ssh-add -l &>/dev/null 2>&1; then
    eval "$(ssh-agent -s)" > /dev/null
    ssh-add "$SSH_KEY" 2>&1 || { echo -e "${R}SSH key failed${N}" >&2; exit 1; }
fi
SSH="ssh -p $SERVER_PORT -o StrictHostKeyChecking=no -o ConnectTimeout=10 $REMOTE"

SCENARIO="${1:-forward}"
shift || true

# back = custom vx=-0.3
if [[ "$SCENARIO" == "back" ]]; then
    EXTRA="--vx -0.3 $*"
    SCENARIO="custom"
else
    EXTRA="$*"
fi

echo -e "${Y}  →${N} Sending '${SCENARIO}' command to robot... (Ctrl+C to stop early)"

$SSH "cd $MWS_DIR && $UV run python tools/control/dds_cmd_publisher.py lo $SCENARIO $EXTRA 2>&1"

echo -e "${G}  ✓${N} Done — stop command sent"
