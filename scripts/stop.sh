#!/usr/bin/env bash
# Stop the WAM stack.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$(dirname "$SCRIPT_DIR")"
exec docker compose -f compose.yml down "$@"
