#!/usr/bin/env bash
# One control surface for the private model, public gateway, and HTTPS tunnel.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BACKEND="$SCRIPT_DIR/serve_h3_glowfish_d6.sh"
GATEWAY="$SCRIPT_DIR/serve_h3_live_demo.sh"

start_gateway() {
  "$GATEWAY" start
  for _ in $(seq 1 40); do
    if curl -fsS --max-time 1 http://127.0.0.1:7860/health >/dev/null; then
      return
    fi
    sleep 0.5
  done
  echo "H3 Live gateway did not become healthy" >&2
  exit 1
}

case "${1:-}" in
  start)
    "$BACKEND" start
    start_gateway
    ;;
  start-public)
    "$BACKEND" start
    start_gateway
    "$GATEWAY" tunnel-start
    ;;
  status)
    "$BACKEND" status || true
    "$GATEWAY" status || true
    "$GATEWAY" tunnel-status || true
    ;;
  stop)
    "$GATEWAY" tunnel-stop || true
    "$GATEWAY" stop || true
    "$BACKEND" stop || true
    ;;
  *) echo "usage: $0 {start|start-public|status|stop}" >&2; exit 2 ;;
esac
