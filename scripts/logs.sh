#!/usr/bin/env bash
# logs.sh — stream container logs from GPU-01 to stdout (and optionally to a file).
#
# Usage:
#   bash scripts/logs.sh                    # stream all 3 containers interleaved
#   bash scripts/logs.sh isaac              # only wam-isaac-sim
#   bash scripts/logs.sh gear               # only wam-gear-sonic
#   bash scripts/logs.sh wam               # only wam-inference
#   bash scripts/logs.sh isaac > /tmp/isaac.log   # save to file

set -euo pipefail

_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[ -f "${_SCRIPT_DIR}/../.env" ] && source "${_SCRIPT_DIR}/../.env"
_WS="${SERVER_WORKSPACE:-/root/skurchev/workspace}"

SERVER_HOST="176.109.83.84"
SERVER_PORT="2221"
SERVER_USER="root"
REMOTE="$SERVER_USER@$SERVER_HOST"
SSH_KEY="$HOME/.ssh/id_ed25519"
if [ ! -f "$SSH_KEY" ]; then
    WIN_HOME=$(wslpath "$(cmd.exe /c 'echo %USERPROFILE%' 2>/dev/null | tr -d '\r')" 2>/dev/null || true)
    [ -n "$WIN_HOME" ] && SSH_KEY="$WIN_HOME/.ssh/id_ed25519"
fi
if [[ "$SSH_KEY" == /mnt/* ]]; then
    _TMP_KEY=$(mktemp /tmp/id_ed25519.XXXXXX)
    cp "$SSH_KEY" "$_TMP_KEY" && chmod 600 "$_TMP_KEY"
    SSH_KEY="$_TMP_KEY"
    trap 'rm -f "$_TMP_KEY"' EXIT
fi

SSH="ssh -p $SERVER_PORT -i $SSH_KEY -o StrictHostKeyChecking=no -o ConnectTimeout=15"

TARGET="${1:-all}"

case "$TARGET" in
    isaac)  CONTAINERS="wam-isaac-sim" ;;
    gear)   CONTAINERS="wam-gear-sonic" ;;
    wam)    CONTAINERS="wam-inference" ;;
    all)    CONTAINERS="wam-isaac-sim wam-gear-sonic wam-inference" ;;
    *)
        echo "Usage: bash scripts/logs.sh [isaac|gear|wam|all]"
        exit 1 ;;
esac

echo "Streaming logs from: $CONTAINERS  (Ctrl-C to stop)" >&2
echo "Server: $REMOTE" >&2
echo "" >&2

# For a single container, just follow its logs directly.
# For multiple, run docker compose logs -f which interleaves and prefixes names.
if [ "$(echo "$CONTAINERS" | wc -w)" -eq 1 ]; then
    $SSH "$REMOTE" "docker logs -f --tail=50 $CONTAINERS 2>&1"
else
    $SSH "$REMOTE" "cd ${_WS}/wam-stack && docker compose logs -f --tail=20 2>&1"
fi
