#!/usr/bin/env bash
set -euo pipefail

WORKSPACE=/lustre/raplab/client/sylarl/minimax-h3-native
SOURCE="$WORKSPACE/vllm-omni-h3-dlo-latest/vllm-omni"
PYTHON="$WORKSPACE/vllm-omni-h3-dlo-latest/.venv-vllm028/bin/python"
RUNTIME_DIR="${H3_DEMO_RUNTIME_DIR:-$WORKSPACE/services/minimax-h3-live-demo}"
PID_FILE="$RUNTIME_DIR/gateway.pid"
LOG_FILE="$RUNTIME_DIR/gateway.log"
HOST="${H3_DEMO_HOST:-0.0.0.0}"
PORT="${H3_DEMO_PORT:-7860}"
CLOUDFLARED="${H3_DEMO_CLOUDFLARED:-$WORKSPACE/tools/cloudflared-2026.8.3}"
TUNNEL_PID_FILE="$RUNTIME_DIR/tunnel.pid"
TUNNEL_LOG_FILE="$RUNTIME_DIR/tunnel.log"
PUBLIC_URL_FILE="$RUNTIME_DIR/public-url.txt"

is_running() {
  [[ -f "$PID_FILE" ]] || return 1
  local pid
  IFS= read -r pid <"$PID_FILE" || return 1
  [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null
}

start() {
  [[ -x "$PYTHON" ]] || { echo "missing Python: $PYTHON" >&2; exit 2; }
  mkdir -p "$RUNTIME_DIR"
  if is_running; then
    echo "H3 Live gateway is already running (pid $(<"$PID_FILE"))"
    return
  fi
  if curl -fsS --max-time 2 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
    echo "port $PORT already hosts a healthy service" >&2
    exit 2
  fi
  cd "$SOURCE"
  export H3_DEMO_RUNTIME_DIR="$RUNTIME_DIR"
  setsid "$PYTHON" -m uvicorn tools.minimax_h3.demo.app:app \
    --host "$HOST" --port "$PORT" --no-access-log >"$LOG_FILE" 2>&1 &
  echo "$!" >"$PID_FILE"
  echo "starting H3 Live gateway on http://$HOST:$PORT (pid $!)"
}

status() {
  local pid=unknown
  [[ -f "$PID_FILE" ]] && IFS= read -r pid <"$PID_FILE" || true
  if curl -fsS --max-time 2 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
    echo "ready: http://127.0.0.1:$PORT (pid $pid)"
  elif is_running; then
    echo "starting: pid $pid"
  else
    echo "stopped"
    return 1
  fi
}

stop() {
  if ! is_running; then echo "H3 Live gateway is not running"; return; fi
  local pid
  IFS= read -r pid <"$PID_FILE"
  kill -TERM -- "-$pid"
  echo "sent SIGTERM to H3 Live gateway process group $pid"
}

tunnel_is_running() {
  [[ -f "$TUNNEL_PID_FILE" ]] || return 1
  local pid
  IFS= read -r pid <"$TUNNEL_PID_FILE" || return 1
  [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null
}

tunnel_start() {
  local public_url
  [[ -x "$CLOUDFLARED" ]] || {
    echo "missing verified cloudflared binary: $CLOUDFLARED" >&2
    exit 2
  }
  curl -fsS --max-time 2 "http://127.0.0.1:$PORT/health" >/dev/null || {
    echo "start the H3 Live gateway before opening its tunnel" >&2
    exit 2
  }
  if tunnel_is_running; then
    echo "H3 Live tunnel is already running (pid $(<"$TUNNEL_PID_FILE"))"
    [[ -f "$PUBLIC_URL_FILE" ]] && cat "$PUBLIC_URL_FILE"
    return
  fi
  : >"$TUNNEL_LOG_FILE"
  rm -f "$PUBLIC_URL_FILE"
  setsid "$CLOUDFLARED" tunnel --no-autoupdate --protocol http2 \
    --url "http://127.0.0.1:$PORT" >"$TUNNEL_LOG_FILE" 2>&1 &
  echo "$!" >"$TUNNEL_PID_FILE"
  for _ in $(seq 1 40); do
    public_url="$(grep -Eo 'https://[-a-z0-9]+\.trycloudflare\.com' \
      "$TUNNEL_LOG_FILE" | tail -1 || true)"
    if [[ -n "$public_url" ]]; then
      printf '%s\n' "$public_url" >"$PUBLIC_URL_FILE"
      echo "public H3 Live demo: $public_url"
      return
    fi
    tunnel_is_running || break
    sleep 0.5
  done
  echo "cloudflared did not publish a quick-tunnel URL" >&2
  tail -n 50 "$TUNNEL_LOG_FILE" >&2
  exit 1
}

tunnel_status() {
  if tunnel_is_running; then
    echo "tunnel running: pid $(<"$TUNNEL_PID_FILE")"
    [[ -f "$PUBLIC_URL_FILE" ]] && cat "$PUBLIC_URL_FILE"
  else
    echo "tunnel stopped"
    return 1
  fi
}

tunnel_stop() {
  if ! tunnel_is_running; then echo "H3 Live tunnel is not running"; return; fi
  local pid
  IFS= read -r pid <"$TUNNEL_PID_FILE"
  kill -TERM -- "-$pid"
  echo "sent SIGTERM to H3 Live tunnel process group $pid"
}

case "${1:-}" in
  start) start ;;
  status) status ;;
  stop) stop ;;
  logs) tail -n 200 -f "$LOG_FILE" ;;
  tunnel-start) tunnel_start ;;
  tunnel-status) tunnel_status ;;
  tunnel-stop) tunnel_stop ;;
  tunnel-logs) tail -n 200 -f "$TUNNEL_LOG_FILE" ;;
  public-url) [[ -f "$PUBLIC_URL_FILE" ]] && cat "$PUBLIC_URL_FILE" ;;
  *) echo "usage: $0 {start|status|stop|logs|tunnel-start|tunnel-status|tunnel-stop|tunnel-logs|public-url}" >&2; exit 2 ;;
esac
