#!/usr/bin/env bash
# Sync wam-stack/ to remote server.
# Also syncs mws-dimos/ if --mws flag is passed (needed for first setup or infra changes).
#
# Usage:
#   ./scripts/sync.sh x32-techgov-GPU-01:/root/skurchev/workspace
#   ./scripts/sync.sh x32-techgov-GPU-01:/root/skurchev/workspace --mws

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <user@server:/workspace> [--mws]"
  exit 1
fi

DEST="$1"
SYNC_MWS=false
[[ "${2:-}" == "--mws" ]] && SYNC_MWS=true

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WAM_ROOT="$(dirname "$SCRIPT_DIR")"
MWS_ROOT="$(dirname "$WAM_ROOT")/mws-dimos"

RSYNC_EXCLUDES=(
  --exclude '.git'
  --exclude '__pycache__/'
  --exclude '*.pyc'
  --exclude '.venv/'
  --exclude '*.egg-info/'
  --exclude '.mypy_cache/'
  --exclude '.ruff_cache/'
  --exclude '*.trt'
  --exclude '*.engine'
  --exclude '*.plan'
)

echo "==> Syncing wam-stack/ → ${DEST}/wam-stack/"
rsync -avz --delete "${RSYNC_EXCLUDES[@]}" \
  "${WAM_ROOT}/" \
  "${DEST}/wam-stack/"

if [[ "$SYNC_MWS" == true ]]; then
  echo "==> Syncing mws-dimos/ → ${DEST}/mws-dimos/"
  rsync -avz --delete "${RSYNC_EXCLUDES[@]}" \
    --exclude 'node_modules/' \
    --exclude '.claude/' \
    --exclude 'dimos/data/' \
    "${MWS_ROOT}/" \
    "${DEST}/mws-dimos/"
fi

echo "==> Done."
echo ""
echo "Next: ssh into server and run:"
echo "  cd ${DEST##*:}/wam-stack"
echo "  scripts/start.sh --scene /root/skurchev/workspace/assets/office_demo.usdz"
