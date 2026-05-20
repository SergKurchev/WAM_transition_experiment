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

exec "$@"
