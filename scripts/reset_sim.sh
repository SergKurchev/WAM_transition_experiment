#!/usr/bin/env bash
# reset_sim.sh — teleport robot back to spawn without restarting any container.
#
# Sends a one-byte ZMQ PUSH to the reset listener in g1_sim.py (tcp://localhost:6559).
# Isaac Sim responds: _reset_robot() + support.rearm() — robot stands up again.
#
# Usage:
#   bash scripts/reset_sim.sh
#
# Equivalent one-liner (no script needed):
#   ssh x32-techgov-GPU-02 "docker exec wam-isaac-sim python3 -c \
#       \"import zmq; s=zmq.Context().socket(zmq.PUSH); s.connect('tcp://localhost:6559'); s.send(b'r')\""

set -euo pipefail

SERVER_HOST="176.109.83.84"
SERVER_PORT="2222"
SERVER_USER="root"
REMOTE="$SERVER_USER@$SERVER_HOST"
SSH_KEY="$HOME/.ssh/id_ed25519"

SSH="ssh -p $SERVER_PORT -o StrictHostKeyChecking=no -o ConnectTimeout=10"

G='\033[0;32m'; Y='\033[1;33m'; R='\033[0;31m'; N='\033[0m'

# ── SSH agent ─────────────────────────────────────────────────────────────────
if ! ssh-add -l &>/dev/null 2>&1; then
    eval "$(ssh-agent -s)" > /dev/null
    ssh-add "$SSH_KEY" 2>&1 || { echo -e "${R}SSH key failed${N}" >&2; exit 1; }
fi

# ── Check container is running ────────────────────────────────────────────────
STATUS=$($SSH "$REMOTE" "docker ps --format '{{.Names}} {{.Status}}' 2>/dev/null | grep wam-isaac-sim" 2>/dev/null || true)
if [[ -z "$STATUS" ]]; then
    echo -e "${R}  ✗${N} wam-isaac-sim is not running. Start the stack first: bash scripts/deploy.sh"
    exit 1
fi

# ── Send reset signal ─────────────────────────────────────────────────────────
echo -e "${Y}  →${N} Sending reset signal to Isaac Sim..."

$SSH "$REMOTE" "docker exec wam-isaac-sim conda run -n unitree_sim_env python3 -c \"
import zmq
s = zmq.Context().socket(zmq.PUSH)
s.setsockopt(zmq.LINGER, 0)
s.setsockopt(zmq.SNDTIMEO, 2000)
s.connect('tcp://localhost:6559')
s.send(b'r')
s.close()
print('reset signal sent', flush=True)
\""

echo -e "${G}  ✓${N} Done — robot teleported to spawn, startup support rearmed."
echo -e "     Watch it stand up: http://localhost:6081"
