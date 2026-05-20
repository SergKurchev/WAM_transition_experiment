#!/usr/bin/env bash
# Start the WAM simulation stack (Isaac Sim + GEAR-SONIC + WAM inference).
#
# Usage:
#   scripts/start.sh                                      # start all services
#   scripts/start.sh --scene /path/to/scene.usd           # load USD scene
#   scripts/start.sh --build                              # rebuild WAM image, then start
#   scripts/start.sh --build-only                         # build only, do not start

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DEFAULT_ROBOT_USD="$PROJECT_ROOT/assets/robots/g1/g1_29dof.usd"

SCENE_PATH=""
ROBOT_USD_PATH="$DEFAULT_ROBOT_USD"
BUILD=false
BUILD_ONLY=false
COMPOSE_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --scene)     SCENE_PATH="$2"; shift 2 ;;
    --robot-usd) ROBOT_USD_PATH="$2"; shift 2 ;;
    --build)     BUILD=true; shift ;;
    --build-only) BUILD_ONLY=true; shift ;;
    *) COMPOSE_ARGS+=("$1"); shift ;;
  esac
done

if [[ -n "$SCENE_PATH" ]]; then
  [[ -f "$SCENE_PATH" ]] || { echo "Error: scene not found: $SCENE_PATH" >&2; exit 1; }
  export SCENE_DIR SCENE_FILE
  SCENE_DIR="$(dirname "$SCENE_PATH")"
  SCENE_FILE="$(basename "$SCENE_PATH")"
fi

if [[ -f "$ROBOT_USD_PATH" ]]; then
  export ROBOT_USD_DIR ROBOT_USD_FILE
  ROBOT_USD_DIR="$(dirname "$ROBOT_USD_PATH")"
  ROBOT_USD_FILE="$(basename "$ROBOT_USD_PATH")"
fi

# Auto-detect GPU SM → TensorRT image (A100 = SM 80)
if [[ -z "${TENSORRT_IMAGE:-}" ]]; then
  GPU_SM=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader,nounits 2>/dev/null \
           | head -1 | tr -d '.' || echo "80")
  case "${GPU_SM:-80}" in
    12*) TENSORRT_IMAGE="nvcr.io/nvidia/tensorrt:25.02-py3" ;;
    *)   TENSORRT_IMAGE="nvcr.io/nvidia/tensorrt:24.12-py3" ;;
  esac
  echo "GPU SM ${GPU_SM}, TensorRT image: ${TENSORRT_IMAGE}"
  export TENSORRT_IMAGE
fi

cd "$PROJECT_ROOT"

if [[ "$BUILD_ONLY" == true ]]; then
  exec docker compose -f compose.yml build "${COMPOSE_ARGS[@]}"
elif [[ "$BUILD" == true ]]; then
  docker compose -f compose.yml build wam "${COMPOSE_ARGS[@]}"
  exec docker compose -f compose.yml up "${COMPOSE_ARGS[@]}"
else
  exec docker compose -f compose.yml up "${COMPOSE_ARGS[@]}"
fi
