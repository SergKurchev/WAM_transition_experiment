#!/bin/bash
# Deploy wam-stack: verify structure, build images, start docker compose
#
# Usage:
#   bash scripts/deploy.sh [--build] [--reset]
#
# Options:
#   --build    rebuild wam-inference image
#   --reset    stop and restart (docker compose down && up)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STACK_ROOT="$(dirname "$SCRIPT_DIR")"
MWS_ROOT="$(dirname "$STACK_ROOT")/mws-dimos"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }
log_debug() { echo -e "${BLUE}[DEBUG]${NC} $1"; }

# ============================================================================
# 0. Check directory structure (NEW CHECK)
# ============================================================================

log_info "Verifying directory structure..."

if [ ! -f "$STACK_ROOT/compose.yml" ]; then
    log_error "compose.yml not found at $STACK_ROOT/compose.yml"
    echo ""
    echo "This script must be run from wam-stack/ directory:"
    echo "  cd /path/to/wam-stack"
    echo "  bash scripts/deploy.sh"
    echo ""
    echo "Expected structure:"
    echo "  workspace/"
    echo "  ├── wam-stack/"
    echo "  │   ├── compose.yml          ← you are here"
    echo "  │   ├── scripts/deploy.sh    ← this script"
    echo "  │   ├── src/"
    echo "  │   ├── docker/"
    echo "  │   └── ..."
    echo "  └── mws-dimos/"
    echo "      └── (feat/real-transfer branch)"
    exit 1
fi

if [ ! -d "$STACK_ROOT/src" ] || [ ! -d "$STACK_ROOT/docker" ]; then
    log_error "Missing src/ or docker/ directory"
    echo "Current directory: $STACK_ROOT"
    echo "Contents:"
    ls -la "$STACK_ROOT" | head -15
    exit 1
fi

log_info "✓ Directory structure correct"

# ============================================================================
# 1. Verify mws-dimos dependency
# ============================================================================

if [ ! -d "$MWS_ROOT" ]; then
    log_error "mws-dimos not found at $MWS_ROOT"
    echo ""
    echo "Expected structure:"
    echo "  workspace/"
    echo "  ├── wam-stack/     ← current location ($(dirname "$STACK_ROOT"))"
    echo "  └── mws-dimos/     ← missing!"
    echo ""
    echo "Clone it first:"
    echo "  cd $(dirname "$STACK_ROOT")"
    echo "  git clone --branch feat/real-transfer https://github.com/MWS-Physical-AI/mws-dimos.git"
    echo ""
    exit 1
fi

log_info "✓ mws-dimos found at $MWS_ROOT"

# Verify branch
cd "$MWS_ROOT"
CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "unknown")
if [ "$CURRENT_BRANCH" != "feat/real-transfer" ]; then
    log_warn "mws-dimos on branch '$CURRENT_BRANCH', expected 'feat/real-transfer'"
    log_info "Checking out feat/real-transfer..."
    git checkout feat/real-transfer 2>&1 | tail -2
fi
log_info "✓ mws-dimos on feat/real-transfer"

# ============================================================================
# 2. Build ros2-bridge image (required, not pre-built locally)
# ============================================================================

log_info "Checking ros2-bridge image..."
cd "$MWS_ROOT"
if ! docker images | grep -q "mws-sim-ros2-bridge:latest"; then
    log_info "Building mws-sim-ros2-bridge:latest (this takes ~2 min)..."
    docker compose -f deploy/sim/ros2/compose.yml build sim-ros2-bridge 2>&1 | tail -5
else
    log_info "✓ mws-sim-ros2-bridge:latest already available"
fi

# ============================================================================
# 3. Handle --reset flag
# ============================================================================

cd "$STACK_ROOT"
if [[ "$@" == *"--reset"* ]]; then
    log_warn "Stopping existing stack (--reset flag)..."
    docker compose down 2>&1 | grep -E "Stopping|Removing|Removed" || true
    sleep 2
fi

# ============================================================================
# 4. Start stack
# ============================================================================

log_info "Starting wam-stack..."
BUILD_FLAG=""
if [[ "$@" == *"--build"* ]]; then
    BUILD_FLAG="--build"
    log_info "  (with --build flag)"
fi

docker compose up -d $BUILD_FLAG

# ============================================================================
# 5. Wait and report
# ============================================================================

log_info "Waiting for services to initialize (1–2 minutes)..."
sleep 10

log_info "Current status:"
docker compose ps

# ============================================================================
# 6. Port forwarding hint
# ============================================================================

echo ""
log_info "Stack started. Next steps:"
echo ""
echo "${BLUE}Terminal 1 (keep running):${NC}"
echo "  docker compose logs -f wam"
echo ""
echo "${BLUE}Terminal 2 (local machine, for visual monitoring):${NC}"
echo "  ssh -N -L 6081:localhost:6080 -p 2221 x32-techgov-GPU-01"
echo "  # Then open: http://localhost:6081"
echo ""
echo "${BLUE}Terminal 3 (if you need to check other services):${NC}"
echo "  docker compose ps"
echo "  docker compose logs -f wam-gear-sonic"
echo "  docker compose logs -f wam-isaac-sim"
echo "  docker compose logs -f wam-ros2-bridge"
echo ""
