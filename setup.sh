#!/usr/bin/env bash
# Full setup and deploy of wam-stack to remote GPU server.
#
# ── First-time setup ─────────────────────────────────────────────────────────
#
#   # 1. Clone this repo locally
#   git clone --branch G1_unifolm_with_GS https://github.com/SergKurchev/WAM_transition_experiment wam-stack && cd wam-stack
#
#   # 2. Run full setup (syncs code, clones unifolm, downloads ~16 GB checkpoint, starts stack)
#   bash setup.sh --build
#
# ── Normal workflow ───────────────────────────────────────────────────────────
#
#   # After editing src/ only (no Dockerfile changes):
#   bash setup.sh
#
#   # After changing Dockerfile or docker/wam/:
#   bash setup.sh --build
#
#   # Full restart (stop all containers, rebuild, restart):
#   bash setup.sh --build --reset
#
#   # Only download/resume checkpoint (don't touch containers):
#   bash setup.sh --weights-only
#
#   # Check checkpoint download progress:
#   bash setup.sh --check-weights
#
# ── Options ───────────────────────────────────────────────────────────────────
#
#   --build          Rebuild wam-inference Docker image
#   --reset          Stop all containers before starting (docker compose down)
#   --weights-only   Download checkpoint only; skip code sync and deploy
#   --check-weights  Print checkpoint download progress and exit
#   --skip-unifolm   Don't re-clone unifolm (use whatever's already on server)
#   --logs           Tail WAM logs after deploy (blocks this terminal)
#   -h, --help       Show this help text
#
# ── What this script does ─────────────────────────────────────────────────────
#
#   1. rsync wam-stack/ → server (excludes .git, __pycache__, media/)
#   2. SSH: rm -rf repos/unifolm && git clone unifolm-world-model-action
#   3. SSH: wget checkpoint (~16 GB) to checkpoints/ if not already complete
#   4. SSH: bash scripts/deploy.sh [--build] [--reset]
#   5. Print: tunnel command, log tailing, media watching
#
# ── Checkpoint source ─────────────────────────────────────────────────────────
#
#   https://huggingface.co/unitreerobotics/UnifoLM-WMA-0-Dual/blob/main/unifolm_wma_dual.ckpt
#

set -euo pipefail

# ── Load server config (.env) ─────────────────────────────────────────────────
# Edit .env to set SERVER_WORKSPACE for your server account.

_SETUP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[ -f "${_SETUP_DIR}/.env" ] && source "${_SETUP_DIR}/.env"

# ── Configuration ─────────────────────────────────────────────────────────────

SERVER_USER="root"
SERVER_IP="176.109.83.84"
SERVER_PORT="2221"
SERVER_KEY="${HOME}/.ssh/id_ed25519"
WORKSPACE="${SERVER_WORKSPACE:-/root/skurchev/workspace}"
STACK_PATH="${WORKSPACE}/wam-stack"

CHECKPOINT_URL="https://huggingface.co/unitreerobotics/UnifoLM-WMA-0-Dual/resolve/main/unifolm_wma_dual.ckpt"
CHECKPOINT_DEST="${STACK_PATH}/checkpoints/unifolm_wma_dual.ckpt"
CHECKPOINT_MIN_BYTES=16000000000  # 16 GB lower bound; real file is ~16.9 GB

UNIFOLM_REPO="https://github.com/unitreerobotics/unifolm-world-model-action"

# ── Color helpers ──────────────────────────────────────────────────────────────

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'
RED='\033[0;31m'; BOLD='\033[1m'; NC='\033[0m'

info()  { echo -e "${GREEN}[INFO]${NC}  $1"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $1"; }
step()  { echo -e "\n${BOLD}${BLUE}══ $1 ══${NC}"; }
die()   { echo -e "${RED}[ERROR]${NC} $1"; exit 1; }

# ── Argument parsing ───────────────────────────────────────────────────────────

DO_BUILD=false
DO_RESET=false
WEIGHTS_ONLY=false
CHECK_WEIGHTS=false
SKIP_UNIFOLM=false
SHOW_LOGS=false

for arg in "$@"; do
  case "$arg" in
    --build)         DO_BUILD=true ;;
    --reset)         DO_RESET=true ;;
    --weights-only)  WEIGHTS_ONLY=true ;;
    --check-weights) CHECK_WEIGHTS=true ;;
    --skip-unifolm)  SKIP_UNIFOLM=true ;;
    --logs)          SHOW_LOGS=true ;;
    -h|--help)
      grep '^#' "$0" | sed 's/^# \?//' | sed '/^!/d'
      exit 0 ;;
    *) warn "Unknown argument: $arg (ignored)" ;;
  esac
done

SSH="ssh -p ${SERVER_PORT} -i ${SERVER_KEY} -o StrictHostKeyChecking=no ${SERVER_USER}@${SERVER_IP}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── --check-weights shortcut ───────────────────────────────────────────────────

if $CHECK_WEIGHTS; then
  step "Checkpoint status"
  $SSH bash -s << SSHEOF
CKPT="${WORKSPACE}/wam-stack/checkpoints/unifolm_wma_dual.ckpt"
if [ ! -f "\$CKPT" ]; then
  echo "  ✗ Not found: \$CKPT"
  exit 0
fi
SIZE=\$(stat -c%s "\$CKPT" 2>/dev/null || stat -f%z "\$CKPT" 2>/dev/null || echo 0)
echo "  File: \$CKPT"
echo "  Size: \$(numfmt --to=iec \$SIZE 2>/dev/null || echo "\${SIZE} bytes")"
if [ \$SIZE -ge 16000000000 ]; then
  echo "  Status: ✓ Complete"
else
  echo "  Status: ⟳ Downloading..."
  if [ -f /tmp/checkpoint_download.pid ]; then
    PID=\$(cat /tmp/checkpoint_download.pid)
    kill -0 \$PID 2>/dev/null && echo "  PID \$PID is still running" || echo "  wget process has exited"
  fi
  echo ""
  echo "  Last log lines:"
  tail -5 /tmp/checkpoint_download.log 2>/dev/null || echo "  (no log file)"
fi
SSHEOF
  exit 0
fi

# ── Step 1: Sync code to server ────────────────────────────────────────────────

if ! $WEIGHTS_ONLY; then
  step "1/4  Syncing wam-stack/ → server"
  RSYNC_RSH="ssh -p ${SERVER_PORT} -i ${SERVER_KEY} -o StrictHostKeyChecking=no" \
    bash "${SCRIPT_DIR}/scripts/sync.sh" \
    "${SERVER_USER}@${SERVER_IP}:${WORKSPACE}"
  info "✓ Code synced"
fi

# ── Step 2: Clone unifolm on server ───────────────────────────────────────────

if ! $SKIP_UNIFOLM && ! $WEIGHTS_ONLY; then
  step "2/4  Cloning unifolm on server"
  $SSH bash -s << EOF
set -e
mkdir -p ${STACK_PATH}/repos
cd ${STACK_PATH}/repos
if [ -d unifolm ]; then
  rm -rf unifolm
  echo "  Removed old unifolm dir"
fi
git clone ${UNIFOLM_REPO} unifolm 2>&1 | tail -3
echo "  ✓ unifolm cloned → ${STACK_PATH}/repos/unifolm"
EOF
  info "✓ unifolm cloned"
fi

# ── Step 3: Download checkpoint ───────────────────────────────────────────────

step "3/4  Checkpoint"
$SSH bash -s << SSHEOF
CKPT="${WORKSPACE}/wam-stack/checkpoints/unifolm_wma_dual.ckpt"
URL="https://huggingface.co/unitreerobotics/UnifoLM-WMA-0-Dual/resolve/main/unifolm_wma_dual.ckpt"
MIN_BYTES=16000000000   # 16 GB lower bound (real file ~16.9 GB)

mkdir -p "\$(dirname \$CKPT)"

# ① Already downloaded and complete — skip
if [ -f "\$CKPT" ]; then
  SIZE=\$(stat -c%s "\$CKPT" 2>/dev/null || echo 0)
  if [ "\$SIZE" -ge "\$MIN_BYTES" ]; then
    echo "  ✓ Checkpoint already complete (\$(numfmt --to=iec \$SIZE 2>/dev/null || echo \${SIZE} bytes))"
    exit 0
  fi
fi

# ② wget still running — don't start a second one
if [ -f /tmp/checkpoint_download.pid ]; then
  PID=\$(cat /tmp/checkpoint_download.pid)
  if kill -0 "\$PID" 2>/dev/null; then
    SIZE=\$(stat -c%s "\$CKPT" 2>/dev/null || echo 0)
    echo "  ⟳ Download already in progress (PID=\$PID, \$(numfmt --to=iec \$SIZE 2>/dev/null || echo \${SIZE} bytes) so far)"
    echo "  Monitor: bash setup.sh --check-weights"
    exit 0
  fi
fi

# ③ Start (or resume) download
if [ -f "\$CKPT" ]; then
  SIZE=\$(stat -c%s "\$CKPT" 2>/dev/null || echo 0)
  echo "  Resuming incomplete download (\$(numfmt --to=iec \$SIZE 2>/dev/null || echo \${SIZE} bytes) already downloaded)..."
else
  echo "  Checkpoint not found, starting download (~16 GB)..."
fi

nohup wget -c "\$URL" -O "\$CKPT" --progress=dot:giga \
  >> /tmp/checkpoint_download.log 2>&1 &
echo \$! > /tmp/checkpoint_download.pid
sleep 1
echo "  Download started (PID=\$(cat /tmp/checkpoint_download.pid))"
echo "  Monitor: bash setup.sh --check-weights"
echo "  ⚠ WAM will start but model returns zeros until download completes (~20 min on 100 Mbit)"
SSHEOF

# ── Step 4: Run deploy.sh on server ───────────────────────────────────────────

if ! $WEIGHTS_ONLY; then
  step "4/4  Running deploy.sh on server"

  DEPLOY_FLAGS=""
  $DO_BUILD && DEPLOY_FLAGS="$DEPLOY_FLAGS --build"
  $DO_RESET && DEPLOY_FLAGS="$DEPLOY_FLAGS --reset"

  $SSH "cd ${STACK_PATH} && bash scripts/deploy.sh${DEPLOY_FLAGS:+ $DEPLOY_FLAGS}"
  info "✓ deploy.sh finished"
fi

# ── Summary ───────────────────────────────────────────────────────────────────

echo ""
echo -e "${BOLD}${GREEN}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BOLD}${GREEN}║  Setup complete                                              ║${NC}"
echo -e "${BOLD}${GREEN}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${BLUE}▸ Watch WAM inference logs:${NC}"
echo "    ssh -p ${SERVER_PORT} ${SERVER_USER}@${SERVER_IP} 'docker logs -f wam-inference'"
echo "  Expected after model loads:"
echo "    [WAM] loop=N  action_0[0:3]=[-0.42 +0.42 -0.17]  norm=1.18  traj_norm=4.96"
echo ""
echo -e "${BLUE}▸ Monitor checkpoint download (if still in progress):${NC}"
echo "    bash setup.sh --check-weights"
echo "    # or live:"
echo "    ssh -p ${SERVER_PORT} ${SERVER_USER}@${SERVER_IP} 'tail -f /tmp/checkpoint_download.log'"
echo ""
echo -e "${BLUE}▸ Open Isaac Sim in browser — run locally, then open http://localhost:6081:${NC}"
echo "    ssh -N -L 6081:localhost:6080 -p ${SERVER_PORT} ${SERVER_USER}@${SERVER_IP}"
echo ""
echo -e "${BLUE}▸ Watch media being saved (model_output/ + robot_camera/ + command_logs/):${NC}"
echo "    ssh -p ${SERVER_PORT} ${SERVER_USER}@${SERVER_IP} \\"
echo "      'watch -n3 \"echo === MEDIA ===; find ${STACK_PATH}/media -type f | wc -l; echo --- newest model output ---; ls -lt ${STACK_PATH}/media/model_output/ 2>/dev/null | head -5; echo --- newest camera frames ---; ls -lt ${STACK_PATH}/media/robot_camera/ 2>/dev/null | head -5\"'"
echo ""
echo -e "${BLUE}▸ Copy media to local machine (frames + videos):${NC}"
echo "    bash scripts/copy-media-from-server.sh"
echo ""
echo -e "${BLUE}▸ Restart WAM only (after editing src/, no rebuild needed):${NC}"
echo "    ssh -p ${SERVER_PORT} ${SERVER_USER}@${SERVER_IP} 'docker restart wam-inference'"
echo ""
echo -e "${BLUE}▸ Stop everything:${NC}"
echo "    ssh -p ${SERVER_PORT} ${SERVER_USER}@${SERVER_IP} 'cd ${STACK_PATH} && docker compose down'"
echo ""

if $SHOW_LOGS; then
  info "Tailing WAM logs (Ctrl+C to stop)..."
  $SSH "docker logs -f wam-inference"
fi
