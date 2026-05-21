#!/usr/bin/env bash
set -e

ip link set lo multicast on 2>/dev/null || true
ip route add 224.0.0.0/4 dev lo 2>/dev/null || true

sysctl -w net.core.rmem_max=67108864 2>/dev/null || true
sysctl -w net.core.rmem_default=67108864 2>/dev/null || true

pkill Xvfb 2>/dev/null || true
rm -f /tmp/.X99-lock
Xvfb :99 -screen 0 1920x1080x24 +extension GLX -ac &
sleep 1
export DISPLAY=:99

x11vnc -display :99 -nopw -forever -shared -bg -quiet -noxdamage

cp /workspace/wam-stack/docker/isaac-sim/novnc-index.html /usr/share/novnc/index.html

pkill -f "websockify.*6080" 2>/dev/null || true
if ss -ltn "( sport = :6080 )" | grep -q LISTEN; then
    echo "[entrypoint] WARNING: noVNC port 6080 already in use" >&2
else
    echo "[entrypoint] noVNC on port 6080"
    websockify --web=/usr/share/novnc 6080 localhost:5900 &
fi

# ── Log file setup: tee stdout+stderr to file AND container stdout ─────────────
LOG_FILE=/tmp/wam_log.log
mkfifo /tmp/wam_fifo 2>/dev/null || true
# tee reads from fifo → writes to log file + stdout (docker logs picks it up)
tee -a "$LOG_FILE" < /tmp/wam_fifo &

# ── Log terminal in noVNC ──────────────────────────────────────────────────────
# Appears on top of Isaac Sim viewport once the sim is ready (wait for /tmp/isaac_ready).
# xterm is not in the base image; install once (~2s, cached by apt after first run).
if ! command -v xterm >/dev/null 2>&1; then
    echo "[entrypoint] installing xterm..."
    apt-get update -qq 2>/dev/null && apt-get install -y --no-install-recommends xterm 2>&1 | tail -1
fi

(
    # Wait until Isaac Sim signals it is ready, then open terminal on top of viewport
    timeout 300 bash -c 'until [ -f /tmp/isaac_ready ]; do sleep 2; done' 2>/dev/null || true
    sleep 2
    DISPLAY=:99 xterm \
        -geometry 240x20+0+840 \
        -bg '#0d1117' -fg '#c9d1d9' -cr '#58a6ff' \
        -fa 'Monospace' -fs 9 \
        -title "WAM Stack — Live Log" \
        -e bash -c "echo '=== WAM Isaac Sim Log ==='; tail -n 60 -F $LOG_FILE"
) &

# ── Launch main command, redirect to fifo (→ tee → log + docker stdout) ───────
exec "$@" > /tmp/wam_fifo 2>&1
