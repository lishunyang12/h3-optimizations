#!/usr/bin/env bash
set -Eeuo pipefail

# Capture exactly one steady MiniMax-H3 denoise update from the production
# 8x RTX PRO 5000 SM120 path.  The default action is intentionally read-only;
# use `run` explicitly to launch the eight-GPU server and request.
#
# Usage:
#   bash tools/minimax_h3/nsys_profile_h3_sm120_sp8_step.sh check
#   bash tools/minimax_h3/nsys_profile_h3_sm120_sp8_step.sh run

ACTION="${1:-check}"
case "$ACTION" in
  check|run) ;;
  *)
    echo "Usage: bash $0 [check|run]" >&2
    exit 2
    ;;
esac

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
VLLM_SOURCE="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
WORKSPACE_DIR="$(cd -- "$VLLM_SOURCE/../.." && pwd)"
CHAMPION_LAUNCHER="${MINIMAX_H3_CHAMPION_LAUNCHER:-$WORKSPACE_DIR/FastVideo/run_vllm_omni_h3_sm120_sp8_adaln.sh}"
VLLM_VENV="${VLLM_OMNI_VENV:-$WORKSPACE_DIR/vllm-omni-h3-dlo-latest/.venv-vllm028}"
PYTHON_BIN="$VLLM_VENV/bin/python"
NSYS_BIN="${NSYS_BIN:-$(command -v nsys || true)}"
PORT="${VLLM_OMNI_PORT:-8091}"
READY_TIMEOUT="${VLLM_OMNI_READY_TIMEOUT:-3600}"
REQUEST_TIMEOUT="${VLLM_OMNI_REQUEST_TIMEOUT:-7200}"
CAPTURE_STEP="${VLLM_OMNI_MINIMAX_H3_NVTX_DENOISE_STEP:-3}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUTPUT_DIR_OVERRIDE="${VLLM_OMNI_NSYS_OUTPUT_DIR:-}"
NVTX_DOMAIN="vllm_omni.minimax_h3"

PROMPT='傍晚小厨房的真人实拍的手 与手绘发光2d动画 融合在一起的影像。夕阳余晖残留在窗边，生活感十足的小厨房里有旧木桌、洗到一半的马克杯、起雾的玻璃瓶、悬挂的抹布。画面带有智能手机单手拍摄的手抖、近距离对焦的犹豫、逆光曝光波动。要像在家中慌忙拍下某个不可思议事件的自然质感，不要广告影像的精心整理。声音只用厨房环境声与手绘生物柔和的电子音、小小的叫声。'

die() {
  echo "ERROR: $*" >&2
  exit 1
}

[[ -n "$NSYS_BIN" && -x "$NSYS_BIN" ]] || die "Nsight Systems CLI (nsys) is missing"
[[ -f "$CHAMPION_LAUNCHER" ]] || die "Champion launcher is missing: $CHAMPION_LAUNCHER"
[[ -x "$PYTHON_BIN" ]] || die "Champion Python is missing: $PYTHON_BIN"
"$PYTHON_BIN" -c 'import nvtx' >/dev/null 2>&1 || die \
  "Champion Python must provide NVIDIA's nvtx package for registered-string triggers"
[[ "$PORT" =~ ^[0-9]+$ ]] || die "VLLM_OMNI_PORT must be an integer"
[[ "$READY_TIMEOUT" =~ ^[1-9][0-9]*$ ]] || die "VLLM_OMNI_READY_TIMEOUT must be positive"
[[ "$REQUEST_TIMEOUT" =~ ^[1-9][0-9]*$ ]] || die "VLLM_OMNI_REQUEST_TIMEOUT must be positive"
[[ "$CAPTURE_STEP" =~ ^[1-9][0-9]*$ ]] || die \
  "VLLM_OMNI_MINIMAX_H3_NVTX_DENOISE_STEP must be in [1, 49]"
CAPTURE_STEP="$((10#$CAPTURE_STEP))"
(( CAPTURE_STEP <= 49 )) || die \
  "VLLM_OMNI_MINIMAX_H3_NVTX_DENOISE_STEP must be in [1, 49]"
printf -v RANGE_NAME 'minimax_h3.denoise.step_%02d' "$CAPTURE_STEP"
printf -v STEP_LABEL '%02d' "$CAPTURE_STEP"
OUTPUT_DIR="${OUTPUT_DIR_OVERRIDE:-$WORKSPACE_DIR/results/vllm-omni-h3-sm120-sp8-step${STEP_LABEL}-nsys-$STAMP}"
PROFILE_PREFIX="$OUTPUT_DIR/minimax-h3-sp8-step${STEP_LABEL}"
PROFILE_REPORT="$PROFILE_PREFIX.nsys-rep"
SERVER_LOG="$OUTPUT_DIR/server-nsys.log"
STATS_LOG="$OUTPUT_DIR/nsys-stats.txt"
REQUEST_LOG="$OUTPUT_DIR/request-curl.log"

export VLLM_OMNI_SOURCE="$VLLM_SOURCE"
export VLLM_OMNI_VENV="$VLLM_VENV"
export VLLM_OMNI_PORT="$PORT"
export VLLM_OMNI_MINIMAX_H3_NVTX_DENOISE_STEP="$CAPTURE_STEP"
export VLLM_OMNI_FLASHINFER_ULYSSES_QK_PRODUCER_DIRECT=1
export MINIMAX_H3_VAE_NVENC_PIPELINE=0
# NVIDIA's Python NVTX package treats any non-empty NVTX_DISABLE value,
# including "0", as disabled.  This runner requires active registered strings.
unset NVTX_DISABLE

echo "MiniMax-H3 single-step Nsight Systems capture"
echo "Source       : $VLLM_SOURCE"
echo "Launcher     : $CHAMPION_LAUNCHER"
echo "NVTX range   : $RANGE_NAME@$NVTX_DOMAIN (DiT rank 0 only)"
echo "Trace        : cuda,nvtx,cudnn,cublas,osrt"
echo "Contract     : original 50-sigma FL2VA, exact AdaLN, TP1/SP8, all-RDMA QK-direct"
echo "Output       : $PROFILE_REPORT"

if [[ "$ACTION" == check ]]; then
  "$NSYS_BIN" --version
  echo "Preflight only; pass 'run' to launch the eight-GPU capture."
  exit 0
fi

command -v curl >/dev/null 2>&1 || die "curl is missing"
command -v setsid >/dev/null 2>&1 || die "setsid is missing"
if curl -fsS --max-time 1 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
  die "Port $PORT already serves a healthy process; refusing to interfere"
fi

mkdir -p "$OUTPUT_DIR"
{
  echo "utc_start=$STAMP"
  echo "source=$VLLM_SOURCE"
  echo "launcher=$CHAMPION_LAUNCHER"
  echo "capture_step=$CAPTURE_STEP"
  echo "nvtx_range=$RANGE_NAME@$NVTX_DOMAIN"
  echo "trace=cuda,nvtx,cudnn,cublas,osrt"
  echo "capture_range_end=stop-shutdown"
  echo "kill=sigterm"
  echo "request=1280x720,24fps,15s,50_sigma_points,shift_12_3,seed_1101"
  echo "weights=original_FL2VA_no_distillation"
  echo "adaln=exact_50_sigma_sidecar"
  echo "topology=TP1_SP8_ring1"
  echo "communication=FlashInfer_PR4876_all_RDMA_QK_producer_direct"
  echo "attention=CUDNN_ATTN_dense"
  echo "vsa=disabled"
  echo "approximate_dit_cache=disabled"
  echo "nvenc=disabled"
} >"$OUTPUT_DIR/config.txt"

PROFILE_PID=""
PROFILE_PGID=""

stop_profile_group() {
  local pgid="${1:-}"
  [[ -n "$pgid" ]] || return 0
  if kill -0 -- "-$pgid" 2>/dev/null; then
    kill -TERM -- "-$pgid" 2>/dev/null || true
    for _ in {1..30}; do
      kill -0 -- "-$pgid" 2>/dev/null || return 0
      sleep 1
    done
    kill -KILL -- "-$pgid" 2>/dev/null || true
  fi
}

cleanup() {
  local status=$?
  trap - EXIT INT TERM
  stop_profile_group "$PROFILE_PGID"
  exit "$status"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

# `setsid` makes cleanup target only this profiler tree.  Nsight's
# stop-shutdown plus --kill=sigterm terminates the launched server process
# group as soon as rank 0 pops the selected range; the explicit cleanup is a
# second guard for re-parented multiprocessing workers.
setsid "$NSYS_BIN" profile \
  --output="$PROFILE_PREFIX" \
  --force-overwrite=true \
  --capture-range=nvtx \
  --nvtx-capture="$RANGE_NAME@$NVTX_DOMAIN" \
  --capture-range-end=stop-shutdown \
  --kill=sigterm \
  --trace=cuda,nvtx,cudnn,cublas,osrt \
  --sample=none \
  --cpuctxsw=none \
  --wait=all \
  bash "$CHAMPION_LAUNCHER" serve >"$SERVER_LOG" 2>&1 &
PROFILE_PID=$!
PROFILE_PGID=$PROFILE_PID
echo "$PROFILE_PID" >"$OUTPUT_DIR/nsys.pid"

deadline=$((SECONDS + READY_TIMEOUT))
until curl -fsS --max-time 2 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; do
  if ! kill -0 "$PROFILE_PID" 2>/dev/null; then
    tail -n 240 "$SERVER_LOG" >&2 || true
    die "Profiled server exited before readiness"
  fi
  if (( SECONDS >= deadline )); then
    tail -n 240 "$SERVER_LOG" >&2 || true
    die "Profiled server did not become ready within ${READY_TIMEOUT}s"
  fi
  sleep 5
done

echo "Server ready; issuing the exact 15-second, 50-sigma T2VA request"
curl_status=0
if curl --fail-with-body --silent --show-error --max-time "$REQUEST_TIMEOUT" \
  -X POST "http://127.0.0.1:$PORT/v1/videos/sync" \
  -F "prompt=$PROMPT" \
  -F 'width=1280' \
  -F 'height=720' \
  -F 'fps=24' \
  -F 'num_inference_steps=50' \
  -F 'flow_shift=12' \
  -F 'seed=1101' \
  -F 'aspect_ratio=16:9' \
  -F 'quality=lossless' \
  -F 'extra_params={"task":"t2va","duration":15,"audio_flow_shift":3.0}' \
  -o "$OUTPUT_DIR/expected-incomplete-response.mp4" \
  2>"$REQUEST_LOG"; then
  :
else
  curl_status=$?
fi
echo "curl_status=$curl_status" >>"$OUTPUT_DIR/config.txt"

nsys_status=0
if wait "$PROFILE_PID"; then
  :
else
  nsys_status=$?
fi
PROFILE_PID=""
# The expected path is already terminated by --kill=sigterm.  Clean the full
# dedicated process group in case a multiprocessing worker outlived Nsight.
stop_profile_group "$PROFILE_PGID"
PROFILE_PGID=""
echo "nsys_status=$nsys_status" >>"$OUTPUT_DIR/config.txt"

[[ -s "$PROFILE_REPORT" ]] || {
  tail -n 240 "$SERVER_LOG" >&2 || true
  die "Nsight report was not created: $PROFILE_REPORT"
}
"$NSYS_BIN" stats \
  --report nvtx_pushpop_sum,nvtx_gpu_proj_sum,cuda_gpu_kern_sum,cuda_api_sum \
  "$PROFILE_REPORT" >"$STATS_LOG"
grep -Fq "$RANGE_NAME" "$STATS_LOG" || die \
  "Report does not contain the requested NVTX range: $RANGE_NAME"

echo "Capture complete: $PROFILE_REPORT"
echo "Summary         : $STATS_LOG"
echo "The curl failure is expected because stop-shutdown ends the server after step $CAPTURE_STEP."
