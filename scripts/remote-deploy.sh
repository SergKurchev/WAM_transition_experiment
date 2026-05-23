#!/bin/bash
# Deploy wam-stack remotely to server (from local machine)
#
# Usage:
#   bash scripts/remote-deploy.sh [--build] [--reset]
#
# What it does:
#   1. Sync local code to server (src/, docker/)
#   2. SSH to server and run deploy.sh with WAM_MODEL=unifolm
#   3. Display logs
#
# Examples:
#   bash scripts/remote-deploy.sh                    # Normal deploy
#   bash scripts/remote-deploy.sh --build            # Rebuild Docker image
#   bash scripts/remote-deploy.sh --reset            # Stop and restart all services

set -e

# ============================================================================
# Configuration
# ============================================================================

SERVER_USER="root"
SERVER_IP="176.109.83.84"
SERVER_PORT="2221"
SERVER_PATH="/root/skurchev/workspace/wam-stack"
LOCAL_KEY="$HOME/.ssh/id_ed25519"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_blue() { echo -e "${BLUE}[REMOTE]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }

# ============================================================================
# Main
# ============================================================================

log_info "Remote deploy script (wam-stack to $SERVER_USER@$SERVER_IP)"
echo ""

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STACK_ROOT="$(dirname "$SCRIPT_DIR")"

# ============================================================================
# Step 1: Sync code to server
# ============================================================================

log_info "Syncing local code to server..."

# Build rsync exclude list (don't sync Docker artifacts, git history, etc)
RSYNC_EXCLUDE=(
    "--exclude=.git"
    "--exclude=.gitignore"
    "--exclude=__pycache__"
    "--exclude=*.pyc"
    "--exclude=.pytest_cache"
    "--exclude=.venv"
    "--exclude=venv"
    "--exclude=.env"
    "--exclude=*.log"
)

# Sync src/ and docker/ (code that changes frequently)
rsync -avz \
    -e "ssh -p $SERVER_PORT -i $LOCAL_KEY" \
    "${RSYNC_EXCLUDE[@]}" \
    "$STACK_ROOT/src/" \
    "$SERVER_USER@$SERVER_IP:$SERVER_PATH/src/" \
    2>/dev/null | grep -E "^(src/|sending|receiving|total)" || true

log_info "✓ Code synced"

# ============================================================================
# Step 2: SSH to server and run deploy.sh
# ============================================================================

echo ""
log_blue "Running deploy.sh on server (WAM_MODEL=unifolm)..."
echo ""

# Prepare deploy flags
DEPLOY_FLAGS=""
if [[ "$@" == *"--build"* ]]; then
    DEPLOY_FLAGS="--build"
    log_warn "Using --build flag (will rebuild Docker image)"
fi
if [[ "$@" == *"--reset"* ]]; then
    DEPLOY_FLAGS="$DEPLOY_FLAGS --reset"
    log_warn "Using --reset flag (will stop and restart services)"
fi

# SSH to server and run deploy.sh with environment variables
ssh -p "$SERVER_PORT" -i "$LOCAL_KEY" "$SERVER_USER@$SERVER_IP" \
    "cd $SERVER_PATH && export WAM_MODEL=unifolm && export WAM_CHECKPOINT='${WAM_CHECKPOINT}' && bash scripts/deploy.sh $DEPLOY_FLAGS"

# ============================================================================
# Step 3: Show next steps
# ============================================================================

echo ""
echo -e "${GREEN}================================${NC}"
log_info "Deploy complete! Next steps:"
echo -e "${GREEN}================================${NC}"
echo ""
echo "Terminal 2️⃣ (watch logs on server):"
echo "  ssh -p $SERVER_PORT -i $LOCAL_KEY $SERVER_USER@$SERVER_IP"
echo "  docker logs -f wam-inference"
echo ""
echo "Terminal 3️⃣ (SSH tunnel for video from local machine):"
echo "  ssh -N -L 6081:localhost:6080 -p $SERVER_PORT -i $LOCAL_KEY $SERVER_USER@$SERVER_IP"
echo ""
echo "Then open browser: http://localhost:6081"
echo ""
