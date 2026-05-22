#!/usr/bin/env bash
# sim_start.sh — Start mws-dimos sim stack in correct order.
#
# Usage (run locally):
#   bash scripts/sim_start.sh           # start sim-isaac, wait, then start sim-gear-sonic
#   bash scripts/sim_start.sh --stop    # stop both containers
#   bash scripts/sim_start.sh --reset   # stop + start fresh

set -euo pipefail

SERVER_HOST="176.109.83.84"
SERVER_PORT="2221"
SERVER_USER="root"
REMOTE="$SERVER_USER@$SERVER_HOST"
MWS_DIR="/root/skurchev/workspace/mws-dimos"
COMPOSE="$MWS_DIR/deploy/sim/ros2/compose.yml"
SSH_KEY="$HOME/.ssh/id_ed25519"
LOCAL_NOVNC_PORT="6181"
# Isaac Sim takes ~2 min to load; timeout after 3 min
ISAAC_TIMEOUT=180

G='\033[0;32m'; Y='\033[1;33m'; R='\033[0;31m'; C='\033[0;36m'; N='\033[0m'
ok()   { echo -e "${G}  ✓${N} $*"; }
warn() { echo -e "${Y}  ⚠${N} $*"; }
err()  { echo -e "${R}  ✗${N} $*" >&2; }
hdr()  { echo -e "\n${C}──── $* ────${N}"; }

# ── SSH agent ─────────────────────────────────────────────────────────────────
if ! ssh-add -l &>/dev/null 2>&1; then
    eval "$(ssh-agent -s)" > /dev/null
    ssh-add "$SSH_KEY" 2>&1 || { err "SSH key failed"; exit 1; }
fi
SSH="ssh -p $SERVER_PORT -o StrictHostKeyChecking=no -o ConnectTimeout=15 $REMOTE"

# ── Args ──────────────────────────────────────────────────────────────────────
STOP=false
for arg in "$@"; do
    case "$arg" in
        --stop)  STOP=true ;;
        --reset) STOP=true ;;  # stop first, then fall through to start
    esac
done

# ── Stop ──────────────────────────────────────────────────────────────────────
if [[ "$STOP" == true ]]; then
    hdr "Stop"
    $SSH "docker compose -f $COMPOSE stop sim-gear-sonic sim-isaac 2>&1 | grep -E 'Stopp|Container' || true"
    ok "Containers stopped"
    [[ "$1" == "--stop" ]] && exit 0
fi

# ── Step 1: Start Isaac Sim ────────────────────────────────────────────────────
hdr "Step 1 — Isaac Sim"
$SSH "docker compose -f $COMPOSE up -d sim-isaac 2>&1 | tail -3"
ok "sim-isaac started"

# ── Step 2: Wait for Isaac Sim healthy ────────────────────────────────────────
hdr "Step 2 — Waiting for Isaac Sim (up to ${ISAAC_TIMEOUT}s)"
echo -n "  "
elapsed=0
while true; do
    STATUS=$($SSH "docker inspect mws-sim-isaac --format '{{.State.Health.Status}}' 2>/dev/null" 2>/dev/null || echo "unknown")
    if [[ "$STATUS" == "healthy" ]]; then
        echo " ✓"
        ok "Isaac Sim is healthy"
        break
    fi
    if [[ "$STATUS" == "unhealthy" ]]; then
        echo ""
        err "Isaac Sim became unhealthy. Logs:"
        $SSH "docker logs mws-sim-isaac --tail=20 2>&1"
        exit 1
    fi
    if [[ $elapsed -ge $ISAAC_TIMEOUT ]]; then
        echo " (timeout)"
        err "Isaac Sim did not become healthy in ${ISAAC_TIMEOUT}s"
        $SSH "docker logs mws-sim-isaac --tail=10 2>&1"
        exit 1
    fi
    echo -n "."
    sleep 3
    elapsed=$((elapsed + 3))
done

# ── Step 3: Start GEAR-SONIC ──────────────────────────────────────────────────
hdr "Step 3 — GEAR-SONIC"
$SSH "docker compose -f $COMPOSE up -d sim-gear-sonic 2>&1 | tail -3"
ok "sim-gear-sonic started"

# ── Step 4: Verify both healthy ───────────────────────────────────────────────
hdr "Step 4 — Verify"
echo -n "  Waiting for GEAR-SONIC "
elapsed=0
while true; do
    STATUS=$($SSH "docker inspect mws-sim-gear-sonic-policy --format '{{.State.Health.Status}}' 2>/dev/null" 2>/dev/null || echo "unknown")
    if [[ "$STATUS" == "healthy" ]]; then
        echo " ✓"
        ok "GEAR-SONIC is healthy"
        break
    fi
    CSTATE=$($SSH "docker inspect mws-sim-gear-sonic-policy --format '{{.State.Status}}' 2>/dev/null" 2>/dev/null || echo "")
    if [[ "$CSTATE" == "exited" ]]; then
        echo ""
        err "GEAR-SONIC exited! Logs:"
        $SSH "docker logs mws-sim-gear-sonic-policy --tail=20 2>&1"
        exit 1
    fi
    if [[ $elapsed -ge 60 ]]; then
        echo " (timeout — may still be starting)"
        break
    fi
    echo -n "."
    sleep 3
    elapsed=$((elapsed + 3))
done

# ── Status ────────────────────────────────────────────────────────────────────
hdr "Status"
$SSH "docker ps --format 'table {{.Names}}\t{{.Status}}' | grep -E 'NAMES|mws-sim'"

# Quick sanity check: look for InitDone in gear-sonic logs
INIT_DONE=$($SSH "docker logs mws-sim-gear-sonic-policy 2>&1 | grep -c 'InitDone received'" 2>/dev/null || echo "0")
DROP_SENT=$($SSH "docker logs mws-sim-gear-sonic-policy 2>&1 | grep -c 'Isaac drop sent'" 2>/dev/null || echo "0")

echo ""
if [[ "$INIT_DONE" -ge 1 ]]; then
    ok "InitDone received from Isaac Sim"
else
    warn "InitDone NOT seen yet in GEAR-SONIC logs"
fi
if [[ "$DROP_SENT" -ge 1 ]]; then
    ok "Isaac drop sent — robot released into simulation"
else
    warn "Drop signal not sent yet"
fi

# ── Tunnel info ───────────────────────────────────────────────────────────────
echo ""
echo -e "${G}┌─────────────────────────────────────────────────────────┐${N}"
echo -e "${G}│  Visual monitoring — run in a NEW terminal:             │${N}"
echo -e "${G}│                                                         │${N}"
echo -e "${G}│  ssh -N -L ${LOCAL_NOVNC_PORT}:localhost:6080 x32-techgov-GPU-01   │${N}"
echo -e "${G}│  Then open:  http://localhost:${LOCAL_NOVNC_PORT}                  │${N}"
echo -e "${G}│                                                         │${N}"
echo -e "${G}│  Stop:   bash scripts/sim_start.sh --stop               │${N}"
echo -e "${G}│  Reset:  bash scripts/sim_start.sh --reset              │${N}"
echo -e "${G}└─────────────────────────────────────────────────────────┘${N}"
