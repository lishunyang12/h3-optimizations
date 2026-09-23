#!/usr/bin/env bash
# Serve the full-MP4 MiniMax-H3 A/B page locally or through a temporary HTTPS tunnel.
set -euo pipefail

WORKSPACE=/lustre/raplab/client/sylarl/minimax-h3-native
SOURCE="$WORKSPACE/vllm-omni-h3-dlo-latest/vllm-omni"
PYTHON="$WORKSPACE/vllm-omni-h3-dlo-latest/.venv-vllm028/bin/python"
RUNTIME_DIR="${H3_COMPARE_RUNTIME_DIR:-$WORKSPACE/services/minimax-h3-compare}"
PID_FILE="$RUNTIME_DIR/server.pid"
LOG_FILE="$RUNTIME_DIR/server.log"
HOST="${H3_COMPARE_HOST:-0.0.0.0}"
PORT="${H3_COMPARE_PORT:-7861}"
CLOUDFLARED="${H3_COMPARE_CLOUDFLARED:-$WORKSPACE/tools/cloudflared-2026.8.3}"
TUNNEL_PID_FILE="$RUNTIME_DIR/tunnel.pid"
TUNNEL_LOG_FILE="$RUNTIME_DIR/tunnel.log"
PUBLIC_URL_FILE="$RUNTIME_DIR/public-url.txt"

usage() {
  echo "usage: $0 {start|start-public|status|stop|logs|tunnel-start|tunnel-status|tunnel-stop|tunnel-logs|public-url}" >&2
}

read_pid() {
  local file="$1"
  local pid
  [[ -f "$file" ]] || return 1
  IFS= read -r pid <"$file" || return 1
  [[ "$pid" =~ ^[0-9]+$ ]] || return 1
  printf '%s\n' "$pid"
}

is_running() {
  local pid
  pid="$(read_pid "$PID_FILE")" || return 1
  kill -0 "$pid" 2>/dev/null
}

tunnel_is_running() {
  local pid
  pid="$(read_pid "$TUNNEL_PID_FILE")" || return 1
  kill -0 "$pid" 2>/dev/null
}

health_json() {
  curl -fsS --max-time 2 "http://127.0.0.1:$PORT/health"
}

comparison_ready() {
  local health
  health="$(health_json)" || return 1
  [[ "$health" == *'"status":"ok"'* ]]
}

start_service() {
  [[ -x "$PYTHON" ]] || { echo "missing Python: $PYTHON" >&2; exit 2; }
  [[ -f "$SOURCE/tools/minimax_h3/compare/static/index.html" ]] || {
    echo "missing comparison page" >&2
    exit 2
  }
  mkdir -p "$RUNTIME_DIR"
  if is_running; then
    echo "MiniMax-H3 comparison is already running (pid $(read_pid "$PID_FILE"), port $PORT)"
    return
  fi
  if health_json >/dev/null 2>&1; then
    echo "port $PORT already hosts a healthy HTTP service; refusing to replace it" >&2
    exit 2
  fi

  cd "$SOURCE"
  setsid "$PYTHON" -m uvicorn tools.minimax_h3.compare.app:app \
    --host "$HOST" --port "$PORT" --no-access-log >"$LOG_FILE" 2>&1 &
  echo "$!" >"$PID_FILE"

  # Importing the serving venv can initialize optional accelerator modules;
  # leave enough room for that one-time startup without declaring failure.
  for _ in $(seq 1 120); do
    if health_json >/dev/null 2>&1; then
      echo "MiniMax-H3 full-MP4 comparison: http://127.0.0.1:$PORT (pid $!)"
      return
    fi
    is_running || break
    sleep 0.25
  done
  echo "comparison server did not become healthy" >&2
  tail -n 80 "$LOG_FILE" >&2 || true
  exit 1
}

status_service() {
  local pid=unknown
  local health
  if read_pid "$PID_FILE" >/dev/null; then
    pid="$(read_pid "$PID_FILE")"
  fi
  if health="$(health_json 2>/dev/null)"; then
    if [[ "$health" == *'"status":"ok"'* ]]; then
      echo "ready: http://127.0.0.1:$PORT (pid $pid; both full MP4 files present)"
    else
      echo "preparing: http://127.0.0.1:$PORT (pid $pid; $health)"
    fi
  elif is_running; then
    echo "starting: pid $pid, port $PORT"
  else
    echo "stopped"
    return 1
  fi
}

stop_service() {
  local pid
  if ! pid="$(read_pid "$PID_FILE")" || ! kill -0 "$pid" 2>/dev/null; then
    echo "MiniMax-H3 comparison is not running"
    rm -f "$PID_FILE"
    return
  fi
  kill -TERM -- "-$pid"
  for _ in $(seq 1 40); do
    kill -0 "$pid" 2>/dev/null || break
    sleep 0.25
  done
  if kill -0 "$pid" 2>/dev/null; then
    echo "comparison process group $pid is still stopping" >&2
    return 1
  fi
  rm -f "$PID_FILE"
  echo "stopped MiniMax-H3 comparison process group $pid"
}

tunnel_start() {
  local public_url
  [[ -x "$CLOUDFLARED" ]] || {
    echo "missing verified cloudflared binary: $CLOUDFLARED" >&2
    exit 2
  }
  comparison_ready || {
    echo "both complete MP4 files must be ready before opening the public tunnel" >&2
    health_json >&2 || true
    exit 2
  }
  if tunnel_is_running; then
    echo "comparison tunnel is already running (pid $(read_pid "$TUNNEL_PID_FILE"))"
    [[ -f "$PUBLIC_URL_FILE" ]] && sed -n '1p' "$PUBLIC_URL_FILE"
    return
  fi

  : >"$TUNNEL_LOG_FILE"
  rm -f "$PUBLIC_URL_FILE"
  setsid "$CLOUDFLARED" tunnel --no-autoupdate --protocol http2 \
    --url "http://127.0.0.1:$PORT" >"$TUNNEL_LOG_FILE" 2>&1 &
  echo "$!" >"$TUNNEL_PID_FILE"

  for _ in $(seq 1 40); do
    public_url="$(grep -Eo 'https://[-a-z0-9]+\.trycloudflare\.com' "$TUNNEL_LOG_FILE" | tail -1 || true)"
    if [[ -n "$public_url" ]]; then
      printf '%s\n' "$public_url" >"$PUBLIC_URL_FILE"
      echo "public MiniMax-H3 comparison: $public_url"
      return
    fi
    tunnel_is_running || break
    sleep 0.5
  done
  echo "cloudflared did not publish a quick-tunnel URL" >&2
  tail -n 80 "$TUNNEL_LOG_FILE" >&2 || true
  exit 1
}

tunnel_status() {
  if tunnel_is_running; then
    echo "tunnel running: pid $(read_pid "$TUNNEL_PID_FILE")"
    [[ -f "$PUBLIC_URL_FILE" ]] && sed -n '1p' "$PUBLIC_URL_FILE"
  else
    echo "tunnel stopped"
    return 1
  fi
}

tunnel_stop() {
  local pid
  if ! pid="$(read_pid "$TUNNEL_PID_FILE")" || ! kill -0 "$pid" 2>/dev/null; then
    rm -f "$TUNNEL_PID_FILE" "$PUBLIC_URL_FILE"
    echo "comparison tunnel is not running"
    return
  fi
  kill -TERM -- "-$pid"
  for _ in $(seq 1 40); do
    kill -0 "$pid" 2>/dev/null || break
    sleep 0.25
  done
  rm -f "$TUNNEL_PID_FILE" "$PUBLIC_URL_FILE"
  echo "stopped comparison tunnel process group $pid"
}

case "${1:-}" in
  start)
    start_service
    ;;
  start-public)
    start_service
    tunnel_start
    ;;
  status)
    status_service || true
    tunnel_status || true
    ;;
  stop)
    tunnel_stop || true
    stop_service || true
    ;;
  logs)
    tail -n 200 -f "$LOG_FILE"
    ;;
  tunnel-start)
    tunnel_start
    ;;
  tunnel-status)
    tunnel_status
    ;;
  tunnel-stop)
    tunnel_stop
    ;;
  tunnel-logs)
    tail -n 200 -f "$TUNNEL_LOG_FILE"
    ;;
  public-url)
    tunnel_is_running && [[ -f "$PUBLIC_URL_FILE" ]] && sed -n '1p' "$PUBLIC_URL_FILE"
    ;;
  *)
    usage
    exit 2
    ;;
esac
