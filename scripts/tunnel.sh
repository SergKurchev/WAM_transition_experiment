#!/bin/bash
# Create SSH port forwarding tunnel for noVNC and monitoring
#
# Usage:
#   bash scripts/tunnel.sh [<port>] [<server>]
#
# Defaults:
#   port   = 6081 (local machine port)
#   server = x32-techgov-GPU-01 with port 2221

LOCAL_PORT="${1:-6081}"
SERVER="${2:-x32-techgov-GPU-01}"
SERVER_PORT="2221"

echo "🔌 Creating SSH tunnel:"
echo "  Local:  http://localhost:$LOCAL_PORT"
echo "  Server: $SERVER:$SERVER_PORT → localhost:6080 (Isaac Sim noVNC)"
echo ""
echo "Press Ctrl+C to stop tunnel"
echo ""

# Kill any existing tunnel on this port
lsof -ti:$LOCAL_PORT 2>/dev/null | xargs kill -9 2>/dev/null || true
sleep 1

# Create tunnel (will run in foreground, can be Ctrl+C to stop)
ssh -N -L $LOCAL_PORT:localhost:6080 -p $SERVER_PORT $SERVER
