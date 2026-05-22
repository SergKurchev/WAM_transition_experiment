#!/usr/bin/env bash
# deploy.sh — sync code to GPU-02, start the stack, print ports.
#
# Usage (from wam-stack/ or anywhere):
#   bash scripts/deploy.sh            # sync + start
#   bash scripts/deploy.sh --build    # sync + rebuild WAM image + start
#   bash scripts/deploy.sh --reset    # full restart (useful when robot fell)
#
# Run in PowerShell:  bash scripts/deploy.sh
# Run in Git Bash:    ./scripts/deploy.sh

set -euo pipefail

# ── Config ────────────────────────────────────────────────────────────────────
SERVER_HOST="176.109.83.84"
SERVER_PORT="2221"
SERVER_USER="root"
REMOTE="$SERVER_USER@$SERVER_HOST"
REMOTE_DIR="/root/skurchev/workspace/wam-stack"
SSH_KEY="$HOME/.ssh/id_ed25519"
# WSL fallback: key lives in Windows home, not Linux home
if [ ! -f "$SSH_KEY" ]; then
    WIN_HOME=$(wslpath "$(cmd.exe /c 'echo %USERPROFILE%' 2>/dev/null | tr -d '\r')" 2>/dev/null || true)
    [ -n "$WIN_HOME" ] && SSH_KEY="$WIN_HOME/.ssh/id_ed25519"
fi
# WSL /mnt/ fix: NTFS mounts have 0777 perms; SSH refuses them — copy to tmp with 600
if [[ "$SSH_KEY" == /mnt/* ]]; then
    _TMP_KEY=$(mktemp /tmp/id_ed25519.XXXXXX)
    cp "$SSH_KEY" "$_TMP_KEY" && chmod 600 "$_TMP_KEY"
    SSH_KEY="$_TMP_KEY"
    trap 'rm -f "$_TMP_KEY"' EXIT
fi
LOCAL_NOVNC_PORT="6181"   # local port for noVNC tunnel (6180 often busy on Windows)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WAM_ROOT="$(dirname "$SCRIPT_DIR")"

SSH="ssh -p $SERVER_PORT -o StrictHostKeyChecking=no -o ConnectTimeout=15"
SCP="scp -P $SERVER_PORT -o StrictHostKeyChecking=no"

# ── Colors ────────────────────────────────────────────────────────────────────
G='\033[0;32m'; Y='\033[1;33m'; R='\033[0;31m'; C='\033[0;36m'; N='\033[0m'
ok()   { echo -e "${G}  ✓${N} $*"; }
warn() { echo -e "${Y}  ⚠${N} $*"; }
err()  { echo -e "${R}  ✗${N} $*" >&2; }
hdr()  { echo -e "\n${C}──── $* ────${N}"; }

# ── Args ──────────────────────────────────────────────────────────────────────
BUILD=false
RESET=false
for arg in "$@"; do
    case "$arg" in
        --build|-b) BUILD=true ;;
        --reset|-r) RESET=true ;;
        --help|-h)
            sed -n '2,7p' "$0" | sed 's/^# \?//'
            exit 0 ;;
    esac
done

# ── 1. SSH agent ──────────────────────────────────────────────────────────────
hdr "SSH"
if ! ssh-add -l &>/dev/null 2>&1; then
    eval "$(ssh-agent -s)" > /dev/null
    ssh-add "$SSH_KEY" 2>&1 || { err "SSH key failed — check passphrase"; exit 1; }
    ok "Key loaded into agent"
else
    ok "Agent already has key"
fi

# ── 2. Sync ───────────────────────────────────────────────────────────────────
hdr "Sync  →  $SERVER_HOST:$REMOTE_DIR"

# Check if rsync is available (not always present in Windows Git Bash)
if command -v rsync &>/dev/null; then
    rsync -az --info=progress2 \
        --exclude='.git/' \
        --exclude='__pycache__/' \
        --exclude='*.pyc' \
        --exclude='*.egg-info/' \
        --exclude='modules/gwbc/.git/' \
        --exclude='modules/gwbc/gear_sonic_deploy/policy/release/*.onnx' \
        --exclude='modules/gwbc/gear_sonic_deploy/planner/' \
        --exclude='modules/gwbc/gear_sonic_deploy/thirdparty/' \
        --exclude='modules/gwbc/external_dependencies/' \
        -e "ssh -p $SERVER_PORT -o StrictHostKeyChecking=no" \
        "$WAM_ROOT/" \
        "$REMOTE:$REMOTE_DIR/"
    ok "rsync complete"
else
    # Fallback: tar over SSH (no rsync needed on Windows)
    warn "rsync not found — using tar+ssh (slower but works)"
    cd "$WAM_ROOT"
    tar czf - \
        --exclude='.git' \
        --exclude='__pycache__' \
        --exclude='*.pyc' \
        --exclude='modules/gwbc/.git' \
        --exclude='modules/gwbc/gear_sonic_deploy/policy' \
        --exclude='modules/gwbc/gear_sonic_deploy/planner' \
        --exclude='modules/gwbc/gear_sonic_deploy/thirdparty' \
        --exclude='modules/gwbc/external_dependencies' \
        src docker scripts compose.yml STATUS.md README.md \
        2>/dev/null \
    | $SSH "$REMOTE" "cd $REMOTE_DIR && tar xzf -"
    ok "tar+ssh sync complete"
fi

# Fix Windows CRLF line endings (safe to run every time)
$SSH "$REMOTE" "
    find $REMOTE_DIR/docker -name '*.sh' -exec sed -i 's/\r//' {} + 2>/dev/null || true
    find $REMOTE_DIR/scripts -name '*.sh' -exec sed -i 's/\r//' {} + 2>/dev/null || true
" && ok "Line endings OK"

# ── 3. Rebuild check + WAM image ─────────────────────────────────────────────
hdr "WAM image"
if [[ "$BUILD" != true ]]; then
    IMAGE_EXISTS=$($SSH "$REMOTE" "docker images -q wam-inference:latest 2>/dev/null" 2>/dev/null || echo "")
    CURRENT_HASH=$($SSH "$REMOTE" "sha256sum $REMOTE_DIR/docker/wam/Dockerfile 2>/dev/null | cut -d' ' -f1" 2>/dev/null || echo "")
    STORED_HASH=$($SSH "$REMOTE"  "cat $REMOTE_DIR/.wam_build_hash 2>/dev/null || echo ''" 2>/dev/null || echo "")

    if [[ -z "$IMAGE_EXISTS" ]]; then
        warn "wam-inference image not found — first build required (~10 min)"
        echo -n "  Build now? [Y/n] "
        read -r REPLY
        if [[ -z "$REPLY" || "$REPLY" =~ ^[Yy]$ ]]; then
            BUILD=true
        else
            err "Cannot start without image. Run: bash scripts/deploy.sh --build"
            exit 1
        fi
    elif [[ -z "$STORED_HASH" || "$CURRENT_HASH" != "$STORED_HASH" ]]; then
        warn "docker/wam/Dockerfile changed since last build"
        echo "  current: ${CURRENT_HASH:0:12}...  last build: ${STORED_HASH:0:12}${STORED_HASH:+...}${STORED_HASH:-  (never recorded)}"
        echo -n "  Rebuild WAM image? [y/N] "
        read -r REPLY
        [[ "$REPLY" =~ ^[Yy]$ ]] && BUILD=true || ok "Skipping rebuild — using existing image"
    else
        ok "WAM image up-to-date (Dockerfile unchanged since last build)"
    fi
fi

if [[ "$BUILD" == true ]]; then
    warn "Building WAM image — ~10 min (torch + CycloneDDS from source)"
    $SSH "$REMOTE" "cd $REMOTE_DIR && docker compose build wam 2>&1 | tail -15"
    $SSH "$REMOTE" "sha256sum $REMOTE_DIR/docker/wam/Dockerfile | cut -d' ' -f1 > $REMOTE_DIR/.wam_build_hash"
    ok "WAM image rebuilt + hash stored"
fi

# ── 4. Start stack ────────────────────────────────────────────────────────────
hdr "Stack"
$SSH "$REMOTE" "
    cd $REMOTE_DIR
    docker compose down 2>&1 | grep -E 'Stopp|Remov' || true
    docker compose up -d 2>&1
"

# ── 5. Health check ───────────────────────────────────────────────────────────
hdr "Health check"
echo -n "  Waiting "
for i in $(seq 1 90); do
    STATUS=$($SSH "$REMOTE" "docker ps --format '{{.Names}} {{.Status}}' 2>/dev/null" 2>/dev/null || echo "")

    if echo "$STATUS" | grep -qE "wam.*(Exited|Error)"; then
        echo " ✗"
        err "Container crashed! Last logs:"
        $SSH "$REMOTE" "cd $REMOTE_DIR && docker compose logs --tail=30 2>&1"
        exit 1
    fi

    HEALTHY=$(echo "$STATUS" | grep -cE "wam.*(healthy)" 2>/dev/null || true)
    WAM_UP=$(echo "$STATUS"  | grep -cE "wam-inference.*Up" 2>/dev/null || true)

    if [[ "${HEALTHY:-0}" -ge 2 && "${WAM_UP:-0}" -ge 1 ]]; then
        echo " ✓"
        break
    fi

    if [[ "$i" -eq 90 ]]; then echo " (timeout)"; fi
    echo -n "."
    sleep 2
done

# ── 6. Status ─────────────────────────────────────────────────────────────────
hdr "Status"
$SSH "$REMOTE" "docker ps --format 'table {{.Names}}\t{{.Status}}' 2>/dev/null | grep -E 'NAMES|wam'"

echo ""
NOVNC=$($SSH "$REMOTE" "ss -tlnp 2>/dev/null | grep ':6180'" 2>/dev/null || echo "")
if [[ -n "$NOVNC" ]]; then
    ok "noVNC listening on server :6180"
else
    warn "noVNC not yet on :6180 (Isaac Sim may still be loading — wait 30s)"
fi

WAM_LOG=$($SSH "$REMOTE" "docker logs wam-inference --tail=2 2>&1" 2>/dev/null || echo "")
if echo "$WAM_LOG" | grep -q "control loop"; then
    ok "WAM: receiving DDS data, control loop running"
elif echo "$WAM_LOG" | grep -q "Waiting for first"; then
    warn "WAM: waiting for first robot state (GEAR-SONIC still starting)"
fi

# ── 7. Tunnel instructions ────────────────────────────────────────────────────
echo ""
echo -e "${G}┌─────────────────────────────────────────────────────────┐${N}"
echo -e "${G}│  Visual monitoring — run in a NEW terminal:             │${N}"
echo -e "${G}│                                                         │${N}"
echo -e "${G}│  ssh -N -L ${LOCAL_NOVNC_PORT}:localhost:6180 x32-techgov-GPU-01   │${N}"
echo -e "${G}│  Then open:  http://localhost:${LOCAL_NOVNC_PORT}                  │${N}"
echo -e "${G}│                                                         │${N}"
echo -e "${G}│  To reset simulation (robot fell — fast, no restart):   │${N}"
echo -e "${G}│  bash scripts/reset_sim.sh                              │${N}"
echo -e "${G}│                                                         │${N}"
echo -e "${G}│  Full restart (if reset_sim.sh fails):                  │${N}"
echo -e "${G}│  bash scripts/deploy.sh --reset                         │${N}"
echo -e "${G}└─────────────────────────────────────────────────────────┘${N}"
