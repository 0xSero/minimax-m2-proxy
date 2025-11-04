#!/usr/bin/env bash
# Helper entrypoint for running the MiniMax-M2 proxy and developer workflows.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${PYTHONPATH:-$ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8001}"

usage() {
  cat <<'USAGE'
Usage: scripts/proxy.sh <command> [args...]

Commands:
  dev [uvicorn args...]     Run the proxy with auto-reload enabled.
  serve [uvicorn args...]   Run the proxy in foreground without reload.
  test [pytest args...]     Execute the pytest suite.
  help                      Show this message.

Environment variables:
  PYTHON_BIN   Python executable to use (default: python3)
  HOST         Bind address (default: 0.0.0.0)
  PORT         Bind port (default: 8001)
USAGE
}

run_dev() {
  exec "$PYTHON_BIN" -m uvicorn proxy.main:app --reload --host "$HOST" --port "$PORT" "$@"
}

run_serve() {
  exec "$PYTHON_BIN" -m uvicorn proxy.main:app --host "$HOST" --port "$PORT" "$@"
}

run_tests() {
  exec "$PYTHON_BIN" -m pytest "$@"
}

COMMAND="${1:-help}"
shift || true

case "$COMMAND" in
  dev) run_dev "$@" ;;
  serve) run_serve "$@" ;;
  test) run_tests "$@" ;;
  help|--help|-h) usage ;;
  *)
    echo "Unknown command: $COMMAND" >&2
    echo
    usage
    exit 1
    ;;
esac
