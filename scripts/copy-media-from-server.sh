#!/bin/bash
# Copy WAM media (frames, logs, video) from server to local machine
#
# Usage:
#   bash scripts/copy-media-from-server.sh [--all] [--frames-only] [--logs-only]
#
# What it does:
#   - Connects to GPU-01 server via SSH
#   - Copies /root/skurchev/workspace/wam-stack/media/ to local
#   - Preserves directory structure and timestamps
#   - Logs full destination paths
#
# Examples:
#   bash scripts/copy-media-from-server.sh              # Copy all media
#   bash scripts/copy-media-from-server.sh --frames-only  # Copy only frames
#   bash scripts/copy-media-from-server.sh --logs-only    # Copy only logs

set -e

# ============================================================================
# Configuration
# ============================================================================

SERVER_USER="root"
SERVER_IP="176.109.83.84"
SERVER_PORT="2221"
SERVER_PATH="/root/skurchev/workspace/wam-stack/media"
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
# Parse arguments
# ============================================================================

COPY_FRAMES=true
COPY_LOGS=true

for arg in "$@"; do
    case "$arg" in
        --frames-only)
            COPY_LOGS=false
            ;;
        --logs-only)
            COPY_FRAMES=false
            ;;
        --all)
            COPY_FRAMES=true
            COPY_LOGS=true
            ;;
        *)
            log_warn "Unknown argument: $arg"
            ;;
    esac
done

# ============================================================================
# Setup
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STACK_ROOT="$(dirname "$SCRIPT_DIR")"

# Local destination
LOCAL_MEDIA_DIR="$STACK_ROOT/media_local"
LOCAL_MEDIA_DIR="$(cd "$STACK_ROOT" 2>/dev/null && echo "$(pwd)/media_local" || echo "$STACK_ROOT/media_local")"

mkdir -p "$LOCAL_MEDIA_DIR"

# Get SSH agent ready
eval "$(ssh-agent -s)" 2>/dev/null || true
if [[ -f /tmp/ssh_pass.sh ]]; then
    DISPLAY=1 SSH_ASKPASS=/tmp/ssh_pass.sh ssh-add "$LOCAL_KEY" 2>&1 | tail -1 || true
fi

# ============================================================================
# Main
# ============================================================================

log_info "Copying media from server..."
echo "  From: $SERVER_USER@$SERVER_IP:$SERVER_PATH"
echo "  To:   $LOCAL_MEDIA_DIR"
echo ""

RSYNC_EXCLUDE=()
if [[ "$COPY_FRAMES" != "true" ]]; then
    RSYNC_EXCLUDE+=("--exclude=input_frames" "--exclude=isaac_frames")
fi
if [[ "$COPY_LOGS" != "true" ]]; then
    RSYNC_EXCLUDE+=("--exclude=command_logs")
fi

# Sync using rsync (preserves timestamps, efficient)
rsync -avz \
    -e "ssh -p $SERVER_PORT -i $LOCAL_KEY -o StrictHostKeyChecking=no" \
    "${RSYNC_EXCLUDE[@]}" \
    "$SERVER_USER@$SERVER_IP:$SERVER_PATH/" \
    "$LOCAL_MEDIA_DIR/" \
    2>/dev/null | grep -E "^(input_frames|command_logs|isaac_frames|sending|receiving|total|\..*)" || true

# ============================================================================
# Summary
# ============================================================================

echo ""
echo -e "${GREEN}================================${NC}"
log_info "Media copy complete!"
echo -e "${GREEN}================================${NC}"
echo ""

# List what was copied
if [[ "$COPY_FRAMES" == "true" ]]; then
    INPUT_COUNT=$(find "$LOCAL_MEDIA_DIR/input_frames" -name "state_*.txt" 2>/dev/null | wc -l || echo 0)
    ISAAC_COUNT=$(find "$LOCAL_MEDIA_DIR/isaac_frames" -name "isaac_*.png" 2>/dev/null | wc -l || echo 0)

    echo "Input frames (RobotState): $INPUT_COUNT"
    echo "  Location: $(cd "$LOCAL_MEDIA_DIR/input_frames" 2>/dev/null && pwd || echo "$LOCAL_MEDIA_DIR/input_frames")"
    echo ""
    echo "Isaac Sim frames: $ISAAC_COUNT"
    echo "  Location: $(cd "$LOCAL_MEDIA_DIR/isaac_frames" 2>/dev/null && pwd || echo "$LOCAL_MEDIA_DIR/isaac_frames")"
    echo ""
fi

if [[ "$COPY_LOGS" == "true" ]]; then
    CSV_FILE=$(find "$LOCAL_MEDIA_DIR/command_logs" -name "commands_*.csv" 2>/dev/null | head -1)
    if [[ -n "$CSV_FILE" ]]; then
        LINES=$(wc -l < "$CSV_FILE" || echo 0)
        echo "Command log (CSV): $LINES lines"
        echo "  Location: $(cd "$(dirname "$CSV_FILE")" && pwd)/$(basename "$CSV_FILE")"
        echo ""
    fi
fi

echo "All media in:"
echo "  $(cd "$LOCAL_MEDIA_DIR" && pwd)"
echo ""

log_info "Next steps:"
echo "  1. Review input_frames/ to see RobotState data"
echo "  2. Check command_logs/*.csv for velocity commands over time"
echo "  3. View isaac_frames/ to see simulator snapshots"
echo "  4. Use these for analysis or video generation"
echo ""
