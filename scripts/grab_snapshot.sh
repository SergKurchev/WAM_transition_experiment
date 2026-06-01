#!/usr/bin/env bash
# grab_snapshot.sh — Capture left + right wrist cameras from the LIVE simulation.
#
# Usage:
#   bash scripts/grab_snapshot.sh                # capture current state
#   bash scripts/grab_snapshot.sh calibrate-on   # stop overriding camera pos (for manual calibration)
#   bash scripts/grab_snapshot.sh calibrate-off  # restore normal operation
#
# Calibration workflow:
#   1.  bash scripts/grab_snapshot.sh calibrate-on
#       → sim stops forcing cameras to the computed position
#   2.  In Isaac Sim viewport: drag the camera prims to the desired position
#       (sim can be paused or running — just don't unpause until you're done dragging)
#   3.  Unpause the sim (if paused) so one render cycle runs
#   4.  bash scripts/grab_snapshot.sh
#       → captures the image from YOUR camera position
#       → cam_calibrate_result.json shows the measured offset (local coords)
#   5.  Copy left/right offset values to scripts/wrist_cam_offset.json
#   6.  bash scripts/grab_snapshot.sh calibrate-off
#       → restores normal operation with the new offset
set -euo pipefail

MODE=${1:-capture}
SERVER_HOST="176.109.83.84"
SERVER_PORT="2221"
SERVER_USER="root"
SSH_KEY="$HOME/.ssh/id_ed25519"
REMOTE_WS="/root/skurchev/workspace"
LOCAL_OUT="media/camera_test"

CONTROL_SOCKET="/tmp/ssh_mux_wam_grab"
trap 'ssh -o ControlPath="$CONTROL_SOCKET" -O stop "$SERVER_USER@$SERVER_HOST" 2>/dev/null || true' EXIT

SSH_BASE="ssh -p $SERVER_PORT -i $SSH_KEY \
    -o StrictHostKeyChecking=no \
    -o ControlMaster=auto \
    -o ControlPath=$CONTROL_SOCKET \
    -o ControlPersist=300"
SSH="$SSH_BASE $SERVER_USER@$SERVER_HOST"
SCP="scp -P $SERVER_PORT -i $SSH_KEY \
    -o StrictHostKeyChecking=no \
    -o ControlMaster=auto \
    -o ControlPath=$CONTROL_SOCKET \
    -o ControlPersist=300"

# ── one passphrase ────────────────────────────────────────────────────────────
$SSH_BASE -N -f "$SERVER_USER@$SERVER_HOST" 2>/dev/null || true

# ── mode: calibrate-on ────────────────────────────────────────────────────────
if [ "$MODE" = "calibrate-on" ]; then
    echo "=== Calibrate mode ON ==="
    $SSH "docker exec wam-inference sh -c 'touch /run/mws/cam_calibrate_mode'"
    echo "  Flag created: /run/mws/cam_calibrate_mode"
    echo "  Cameras will NO LONGER be forced to the computed position."
    echo ""
    echo "  Next steps:"
    echo "   1. In Isaac Sim viewport, drag the camera prims to the desired position."
    echo "   2. Unpause the sim if paused (one render cycle needed to measure)."
    echo "   3. Run:  bash scripts/grab_snapshot.sh"
    echo "      → image shows your camera angle"
    echo "      → media/camera_test/cam_calibrate_result.json shows the measured offset"
    echo "   4. Copy values to scripts/wrist_cam_offset.json"
    echo "   5. Run:  bash scripts/grab_snapshot.sh calibrate-off"
    exit 0
fi

# ── mode: calibrate-off ───────────────────────────────────────────────────────
if [ "$MODE" = "calibrate-off" ]; then
    echo "=== Calibrate mode OFF ==="
    $SSH "docker exec wam-inference sh -c 'rm -f /run/mws/cam_calibrate_mode /run/mws/cam_calibrate_result.json'"
    # Upload the (possibly updated) offset and capture to confirm
    $SCP scripts/wrist_cam_offset.json "$SERVER_USER@$SERVER_HOST:/tmp/wrist_cam_offset.json"
    $SSH "docker cp /tmp/wrist_cam_offset.json wam-inference:/run/mws/camera_offset.json"
    echo "  Normal operation restored. Offset from wrist_cam_offset.json applied."
    echo "  Capturing to confirm..."
    sleep 2
    # fall through to capture below
fi

# ── capture (default + after calibrate-off) ───────────────────────────────────
echo "=== Capture wrist snapshots ==="

# Upload offset (if file exists and we're not in calibrate mode still)
if $SSH "docker exec wam-inference sh -c '[ ! -f /run/mws/cam_calibrate_mode ] && echo ok || echo skip'" | grep -q ok; then
    $SCP scripts/wrist_cam_offset.json "$SERVER_USER@$SERVER_HOST:/tmp/wrist_cam_offset.json"
    $SSH "docker cp /tmp/wrist_cam_offset.json wam-inference:/run/mws/camera_offset.json"
    sleep 2
fi

$SCP scripts/capture_snapshot.py "$SERVER_USER@$SERVER_HOST:$REMOTE_WS/wam-stack/scripts/"
$SSH "docker cp $REMOTE_WS/wam-stack/scripts/capture_snapshot.py wam-inference:/tmp/"
$SSH "docker exec wam-inference python3 /tmp/capture_snapshot.py"

mkdir -p "$LOCAL_OUT"
for f in snapshot_left_latest snapshot_right_latest; do
    $SCP "$SERVER_USER@$SERVER_HOST:$REMOTE_WS/wam-stack/media/camera_test/$f.png"  "$LOCAL_OUT/$f.png"
    $SCP "$SERVER_USER@$SERVER_HOST:$REMOTE_WS/wam-stack/media/camera_test/$f.json" "$LOCAL_OUT/$f.json"
done
$SCP "$SERVER_USER@$SERVER_HOST:$REMOTE_WS/wam-stack/media/camera_test/snapshot_composite_latest.png" \
    "$LOCAL_OUT/snapshot_composite_latest.png"

# Download calibrate result if present
$SCP "$SERVER_USER@$SERVER_HOST:$REMOTE_WS/wam-stack/media/camera_test/cam_calibrate_result.json" \
    "$LOCAL_OUT/cam_calibrate_result.json" 2>/dev/null && {
    echo ""
    echo "=== Measured offset (copy to wrist_cam_offset.json) ==="
    cat "$LOCAL_OUT/cam_calibrate_result.json"
} || true

echo ""
ls -lh "$LOCAL_OUT/snapshot_"*"_latest".*
