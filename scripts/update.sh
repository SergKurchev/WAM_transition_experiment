#!/usr/bin/env bash
# update.sh — pull latest commits on GPU-02, update submodules, restart stack.
#
# Workflow: push to GitHub locally → run this script → server pulls & restarts.
#
# Usage (from wam-stack/ or anywhere):
#   bash scripts/update.sh            # git pull + submodule update + restart
#   bash scripts/update.sh --no-restart  # pull only, skip docker restart

set -euo pipefail

# ── Config ────────────────────────────────────────────────────────────────────
SERVER_HOST="176.109.83.84"
SERVER_PORT="2222"
SERVER_USER="root"
REMOTE="$SERVER_USER@$SERVER_HOST"
REMOTE_DIR="/root/skurchev/workspace/wam-stack"
SSH_KEY="$HOME/.ssh/id_ed25519"

SSH="ssh -p $SERVER_PORT -o StrictHostKeyChecking=no -o ConnectTimeout=15"

# ── Colors ────────────────────────────────────────────────────────────────────
G='\033[0;32m'; Y='\033[1;33m'; R='\033[0;31m'; C='\033[0;36m'; N='\033[0m'
ok()   { echo -e "${G}  ✓${N} $*"; }
warn() { echo -e "${Y}  ⚠${N} $*"; }
err()  { echo -e "${R}  ✗${N} $*" >&2; }
hdr()  { echo -e "\n${C}──── $* ────${N}"; }

# ── Args ──────────────────────────────────────────────────────────────────────
RESTART=true
for arg in "$@"; do
    case "$arg" in
        --no-restart) RESTART=false ;;
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

# ── 2. git pull on server ─────────────────────────────────────────────────────
hdr "git pull  →  $SERVER_HOST:$REMOTE_DIR"
$SSH "$REMOTE" "
    set -e
    cd $REMOTE_DIR
    echo '  branch: '\$(git rev-parse --abbrev-ref HEAD)
    DIRTY=\$(git status --porcelain 2>/dev/null | wc -l)
    if [ \"\$DIRTY\" -gt 0 ]; then
        echo '  stashing \$DIRTY locally-modified file(s) on server...'
        git stash push -m 'auto-stash before update.sh pull'
    fi
    git pull --ff-only 2>&1
"
ok "Repository updated"

# ── 3. Submodules (gwbc) ──────────────────────────────────────────────────────
hdr "Submodules (gwbc)"
$SSH "$REMOTE" "
    cd $REMOTE_DIR
    if [ -f .gitmodules ]; then
        git submodule update --init --recursive 2>&1 | tail -10
    else
        echo '  (no .gitmodules — skipping)'
    fi
"
ok "Submodules updated"

# ── 4. Fix permissions ────────────────────────────────────────────────────────
hdr "Permissions"
$SSH "$REMOTE" "
    find $REMOTE_DIR/scripts -name '*.sh' -exec chmod +x {} +
    find $REMOTE_DIR/docker  -name '*.sh' -exec chmod +x {} +
"
ok "All .sh files are executable"

# ── 5. Restart stack ──────────────────────────────────────────────────────────
if [[ "$RESTART" == true ]]; then
    hdr "Restart stack"
    $SSH "$REMOTE" "
        cd $REMOTE_DIR
        docker compose down 2>&1 | grep -E 'Stopp|Remov' || true
        docker compose up -d 2>&1
    "
    ok "Stack restarted"
else
    warn "Skipping restart (--no-restart)"
fi

# ── 5. Quick status ───────────────────────────────────────────────────────────
hdr "Status"
sleep 3
$SSH "$REMOTE" "
    echo '  commit: '\$(cd $REMOTE_DIR && git log -1 --format='%h %s')
    docker ps --format 'table {{.Names}}\t{{.Status}}' 2>/dev/null | grep -E 'NAMES|wam' || true
"

echo ""
ok "Done — server is on latest commit"
echo "  noVNC:    bash scripts/view.sh"
echo "  Logs:     ssh x32-techgov-GPU-02 'docker logs wam-inference --tail=20'"
