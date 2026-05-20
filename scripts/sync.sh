#!/usr/bin/env bash
# Sync wam-stack/ to remote server.
#
# Usage:
#   ./scripts/sync.sh x32-techgov-GPU-01:/root/skurchev/workspace

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <user@server:/workspace>"
  exit 1
fi

DEST="$1"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WAM_ROOT="$(dirname "$SCRIPT_DIR")"

rsync -avz --delete \
  --exclude '.git' \
  --exclude '__pycache__/' \
  --exclude '*.pyc' \
  --exclude '.venv/' \
  --exclude '*.egg-info/' \
  --exclude '*.trt' \
  --exclude '*.engine' \
  --exclude '*.plan' \
  --exclude 'modules/gwbc/.git' \
  --exclude 'assets/robots/g1/configuration/*_base.usd' \
  "${WAM_ROOT}/" \
  "${DEST}/wam-stack/"

echo "==> Done."
echo "On server: cd ${DEST##*:}/wam-stack && git submodule update --init --recursive"
