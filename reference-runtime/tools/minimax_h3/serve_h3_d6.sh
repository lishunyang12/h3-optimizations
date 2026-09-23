#!/usr/bin/env bash
# Serve the exact D6 MiniMax-H3 stack qualified by the 2026-09-08 repeat-3 run.
#
# The command itself is reused from the immutable benchmark evidence directory
# so the large AdaLN/QKV provenance bindings cannot drift during hand copying.

set -euo pipefail

WORKSPACE=/lustre/raplab/client/sylarl/minimax-h3-native
VLLM_SOURCE="$WORKSPACE/vllm-omni-h3-dlo-latest/vllm-omni"
VLLM_VENV="$WORKSPACE/vllm-omni-h3-dlo-latest/.venv-vllm028"
FASTVIDEO_KERNEL_PYTHON="$WORKSPACE/FastVideo/fastvideo-kernel/python"
FLASHINFER_ROOT="$WORKSPACE/fi-pr4944-pr4876-combined"
FLASHINFER_WORKSPACE="$WORKSPACE/.flashinfer-pr4944-pr4876-vllm028-cache"
CUDA_TOOLKIT="$WORKSPACE/toolchains/cuda-13.2.78-full/nvidia/cu13"
TAEH3_CHECKPOINT="$WORKSPACE/cache/taeh3-62f7591f59dfbb4c3c02b7a621d180a9eeaba26c.safetensors"
EVIDENCE_DIR="${VLLM_OMNI_H3_SERVICE_EVIDENCE_DIR:-$WORKSPACE/results/vllm-omni-fasth3-cross-rank-audio-profiler-off-repeat3-20260908-resume2/vllm-omni-tp1-sp8-fasth3-vsa-flashinfer-topk162-rdma-qk-direct-qkv-e4m3-all-main-fp8-o-bundle-taeh3-fp16-persistent-rdma-cross-rank-audio}"
SERVER_COMMAND="$EVIDENCE_DIR/server-command.txt"

RUNTIME_DIR="${VLLM_OMNI_H3_SERVICE_DIR:-$WORKSPACE/services/minimax-h3-d6}"
PID_FILE="$RUNTIME_DIR/server.pid"
LOG_FILE="$RUNTIME_DIR/server.log"
PORT=8091
FLASHINFER_MAX_BYTES="${VLLM_OMNI_H3_SERVICE_FLASHINFER_MAX_BYTES:-175056896}"

usage() {
  echo "usage: $0 {start|status|stop|logs}"
}

require_runtime() {
  [[ -x "$VLLM_VENV/bin/vllm" ]] || {
    echo "missing vLLM executable: $VLLM_VENV/bin/vllm" >&2
    exit 2
  }
  [[ -f "$SERVER_COMMAND" ]] || {
    echo "missing qualified server command: $SERVER_COMMAND" >&2
    exit 2
  }
  [[ -f "$TAEH3_CHECKPOINT" ]] || {
    echo "missing pinned TAEH3 checkpoint: $TAEH3_CHECKPOINT" >&2
    exit 2
  }
  [[ -x "$CUDA_TOOLKIT/bin/nvcc" ]] || {
    echo "missing CUDA 13.2 toolkit: $CUDA_TOOLKIT" >&2
    exit 2
  }
}

export_runtime() {
  export CUDA_DEVICE_ORDER=PCI_BUS_ID
  export CUDA_VISIBLE_DEVICES=0,4,1,5,2,6,3,7
  export VLLM_WORKER_MULTIPROC_METHOD=spawn
  export VLLM_OMNI_VIDEO_SYNC_TIMEOUT=7200
  export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
  export PYTHONNOUSERSITE=1
  export HF_HOME="${HF_HOME:-$WORKSPACE/hf-cache}"
  export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$WORKSPACE/xdg-cache}"
  export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$WORKSPACE/torchinductor-cache}"
  export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$WORKSPACE/triton-cache}"
  export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
  export NCCL_DEBUG_SUBSYS="${NCCL_DEBUG_SUBSYS:-INIT,GRAPH,TUNING}"

  export PYTHONPATH="$FLASHINFER_ROOT:$FASTVIDEO_KERNEL_PYTHON:$VLLM_SOURCE${PYTHONPATH:+:$PYTHONPATH}"
  export CUDA_HOME="$CUDA_TOOLKIT"
  export PATH="$CUDA_HOME/bin:$PATH"
  export LD_LIBRARY_PATH="$CUDA_HOME/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  export TORCH_CUDA_ARCH_LIST=12.0
  export FLASHINFER_CUDA_ARCH_LIST=12.0f
  export FLASHINFER_NVCC="$CUDA_HOME/bin/nvcc"
  export FLASHINFER_EXTRA_LDFLAGS="-L$CUDA_HOME/lib"
  export FLASHINFER_WORKSPACE_BASE="$FLASHINFER_WORKSPACE"
  export FLASHINFER_ULYSSES_PCIE_ROUTE=rdma
  export FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED=0
  unset FLASH_ATTENTION_CUTE_DSL_CACHE_DIR

  export MINIMAX_H3_VAE_NVENC_PIPELINE=0
  export VLLM_OMNI_MINIMAX_H3_GPU_UINT8_OUTPUT=0
  export VLLM_OMNI_MINIMAX_H3_CHUNKED_CPU_MP4_OUTPUT=1
  export VLLM_OMNI_MINIMAX_H3_CHUNKED_CPU_MP4_SLOTS=2
  export VLLM_OMNI_MINIMAX_H3_VIDEO_DECODE_BACKEND=taeh3
  export VLLM_OMNI_MINIMAX_H3_TAEH3_CHECKPOINT="$TAEH3_CHECKPOINT"
  export VLLM_OMNI_MINIMAX_H3_TAEH3_DTYPE=fp16
  export VLLM_OMNI_MINIMAX_H3_TAEH3_CHUNK_SIZE=5
  export VLLM_OMNI_MINIMAX_H3_TAEH3_AUDIO_DECODE_OVERLAP=0
  export VLLM_OMNI_MINIMAX_H3_TAEH3_CROSS_RANK_AUDIO_DECODE=1
  export VLLM_OMNI_MINIMAX_H3_TAEH3_CROSS_RANK_AUDIO_ENCODE=0

  export VLLM_OMNI_FASTVIDEO_VSA_FUSED_TILE_PACK=1
  export VLLM_OMNI_FASTVIDEO_VSA_FUSED_UNTILE=1
  export VLLM_OMNI_FASTVIDEO_VSA_FUSED_GATE_UNTILE=0
  export VLLM_OMNI_FASTVIDEO_VSA_DEFERRED_GATE_SP=0
  export VLLM_OMNI_FASTVIDEO_VSA_O_BUNDLE=1
  export VLLM_OMNI_FASTVIDEO_VSA_GATE_COMPUTE_OVERLAP=0
  export VLLM_OMNI_FASTVIDEO_VSA_DIRECT_Q2K=1
  export FASTVIDEO_VSA_SM100A=0

  export VLLM_OMNI_MINIMAX_H3_VAE_TEMPORAL_PRE_GATHER_PRUNE=0
  export VLLM_OMNI_MINIMAX_H3_VAE_TEMPORAL_CHUNK_PARALLEL=0
  export VLLM_OMNI_MINIMAX_H3_VAE_DECODER_TILE_SIZE=0
  export VLLM_OMNI_MINIMAX_H3_VAE_EXACT_OPS_SM120=0
  export MINIMAX_H3_VAE_DECODER_STREAM_TEMPORAL_CAT=1
  export VLLM_OMNI_MINIMAX_H3_QCHUNK_PIPELINE=0

  export VLLM_OMNI_ULYSSES_A2A_BACKEND=flashinfer-pcie
  export VLLM_OMNI_FLASHINFER_ULYSSES_QCHUNK2_DIRECT=0
  export VLLM_OMNI_FLASHINFER_ULYSSES_QCHUNK2_QKV_FUSED=0
  export VLLM_OMNI_FLASHINFER_ULYSSES_QK_PRODUCER_DIRECT=1
  export VLLM_OMNI_FLASHINFER_ULYSSES_O_PRODUCER_DIRECT=1
  export VLLM_OMNI_FLASHINFER_ULYSSES_QKV_E4M3_TRANSPORT=1
  export VLLM_OMNI_FLASHINFER_ULYSSES_RELEASE_AFTER_DENOISE=0
  export VLLM_OMNI_FLASHINFER_ULYSSES_REQUIRE_RDMA=1
  export VLLM_OMNI_FLASHINFER_ULYSSES_MAX_BYTES="$FLASHINFER_MAX_BYTES"

  export VLLM_OMNI_MINIMAX_H3_FP8_SCOPE=all-main
  export VLLM_OMNI_MINIMAX_H3_FP8_CONFIG_SHA256=6a4c6ed2b05085cd111f9c4652062da67318ac870c6cbac0d309a0d8bdc4a3c0
  export VLLM_OMNI_MINIMAX_H3_FP8_CONTRACT_SHA256=21aab1eadd0550ea676161829a609a5697aa445c3528cff87f132c65b2d794f4

  unset VLLM_OMNI_FLASHINFER_ULYSSES_QKV_SINGLE_PHASE
  unset VLLM_OMNI_MINIMAX_H3_QKV_TRANSPORT_OBSERVER_MODE
  unset VLLM_OMNI_MINIMAX_H3_QKV_TRANSPORT_OBSERVER_SNAPSHOT_DIR
  unset VLLM_OMNI_MINIMAX_H3_QKV_TRANSPORT_OBSERVER_SCALES_PATH
}

is_running() {
  [[ -f "$PID_FILE" ]] || return 1
  local pid
  pid="$(<"$PID_FILE")"
  [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null
}

start_service() {
  require_runtime
  mkdir -p "$RUNTIME_DIR" "$FLASHINFER_WORKSPACE"
  if is_running; then
    echo "MiniMax-H3 D6 is already running (pid $(<"$PID_FILE"), port $PORT)"
    return
  fi
  if curl -fsS --max-time 2 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
    echo "port $PORT already has a healthy HTTP service; refusing to replace it" >&2
    exit 2
  fi
  export_runtime
  cd "$VLLM_SOURCE"
  setsid bash "$SERVER_COMMAND" >"$LOG_FILE" 2>&1 &
  echo "$!" >"$PID_FILE"
  echo "starting MiniMax-H3 D6 (pid $!, port $PORT); log: $LOG_FILE"
}

status_service() {
  local pid=unknown
  if [[ -f "$PID_FILE" ]]; then
    IFS= read -r pid <"$PID_FILE" || pid=unknown
  fi
  if curl -fsS --max-time 2 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
    echo "ready: http://127.0.0.1:$PORT (pid $pid)"
  elif is_running; then
    echo "starting: pid $pid, port $PORT"
  else
    echo "stopped"
    return 1
  fi
}

stop_service() {
  if ! is_running; then
    echo "MiniMax-H3 D6 is not running"
    return
  fi
  local pid
  pid="$(<"$PID_FILE")"
  kill -TERM -- "-$pid"
  echo "sent SIGTERM to MiniMax-H3 D6 process group $pid"
}

case "${1:-}" in
  start) start_service ;;
  status) status_service ;;
  stop) stop_service ;;
  logs) tail -n 200 -f "$LOG_FILE" ;;
  *) usage; exit 2 ;;
esac
