#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

set -Eeuo pipefail

# Profile only the second decode invocation of the current FastH3 realtime
# stack. Invocation one is the benchmark warmup; invocation two is the single
# measured request. The NVTX range is published by DiT rank 0, while Nsight
# records the complete process tree so rank1 Audio VAE, rank0 TAEH3, NCCL, D2H,
# libx264, AAC, and mux activity share one timeline.

ACTION="${1:-check}"
ACTION_MODE=check
AAC_PREENCODE=0
case "$ACTION" in
  check|run)
    ACTION_MODE="$ACTION"
    ;;
  check-aac)
    AAC_PREENCODE=1
    ;;
  run-aac)
    ACTION_MODE=run
    AAC_PREENCODE=1
    ;;
  *)
    echo "Usage: bash $0 [check|run|check-aac|run-aac]" >&2
    exit 2
    ;;
esac

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
VLLM_SOURCE="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
WORKSPACE_DIR="$(cd -- "$VLLM_SOURCE/../.." && pwd)"
VLLM_VENV="${VLLM_OMNI_VENV:-$WORKSPACE_DIR/vllm-omni-h3-dlo-latest/.venv-vllm028}"
PYTHON_BIN="$VLLM_VENV/bin/python"
NSYS_BIN="${NSYS_BIN:-$(command -v nsys || true)}"
BENCHMARK_RUNNER="$WORKSPACE_DIR/FastVideo/benchmark_vllm_omni_h3_comm_15s.sh"
ANALYZER="$SCRIPT_DIR/analyze_h3_nsys_step.py"
BENCHMARK_ACTION="vllm-omni-tp1-sp8-fasth3-vsa-flashinfer-topk162-rdma-qk-direct-qkv-e4m3-all-main-fp8-o-bundle-taeh3-fp16-persistent-rdma-cross-rank-audio"
PROFILE_VARIANT=cross-rank-decode
if [[ "$AAC_PREENCODE" == 1 ]]; then
  BENCHMARK_ACTION="${BENCHMARK_ACTION}-aac"
  PROFILE_VARIANT=cross-rank-aac
fi
NVTX_DOMAIN="vllm_omni.minimax_h3"
DECODE_REQUEST=2
RANGE_NAME="minimax_h3.decode.request_02"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUTPUT_DIR="${VLLM_OMNI_NSYS_OUTPUT_DIR:-$WORKSPACE_DIR/results/vllm-omni-fasth3-taeh3-cross-rank-decode-nsys-$STAMP}"
PROFILE_PREFIX="$OUTPUT_DIR/minimax-h3-taeh3-$PROFILE_VARIANT"
PROFILE_REPORT="$PROFILE_PREFIX.nsys-rep"
BENCHMARK_OUTPUT="$OUTPUT_DIR/benchmark"
CASE_DIR="$BENCHMARK_OUTPUT/$BENCHMARK_ACTION"
ANALYSIS_MD="$OUTPUT_DIR/decode-analysis.md"
ANALYSIS_JSON="$OUTPUT_DIR/decode-analysis.json"
ANALYSIS_STATS="$OUTPUT_DIR/decode-analysis-nsys-stats.txt"

die() {
  echo "ERROR: $*" >&2
  exit 1
}

[[ -n "$NSYS_BIN" && -x "$NSYS_BIN" ]] || die "Nsight Systems CLI (nsys) is missing"
[[ -x "$PYTHON_BIN" ]] || die "vLLM Python is missing: $PYTHON_BIN"
[[ -f "$BENCHMARK_RUNNER" ]] || die "FastH3 benchmark runner is missing: $BENCHMARK_RUNNER"
[[ -f "$ANALYZER" ]] || die "H3 Nsight analyzer is missing: $ANALYZER"
bash -n "$BENCHMARK_RUNNER"

if [[ "$ACTION_MODE" == check ]]; then
  echo "Nsight Systems : $($NSYS_BIN --version)"
  echo "Benchmark       : $BENCHMARK_ACTION"
  echo "Capture         : $RANGE_NAME@$NVTX_DOMAIN"
  echo "Decode request  : $DECODE_REQUEST (warmup=1, measured=1)"
  echo "AAC pre-encode  : $AAC_PREENCODE"
  echo "Trace           : cuda,nvtx,cudnn,cublas,osrt"
  echo "Status          : ready"
  exit 0
fi

mkdir -p "$OUTPUT_DIR"
{
  echo "benchmark_action=$BENCHMARK_ACTION"
  echo "nvtx_range=$RANGE_NAME@$NVTX_DOMAIN"
  echo "nvtx_decode_request=$DECODE_REQUEST"
  echo "warmups=1"
  echo "repeats=1"
  echo "diffusion_pipeline_profiler=0"
  echo "trace=cuda,nvtx,cudnn,cublas,osrt"
  echo "capture_range_end=stop"
  echo "quality_contract=diagnostic_only_production_route_unchanged"
  echo "aac_preencode=$AAC_PREENCODE"
} >"$OUTPUT_DIR/config.txt"

export VLLM_OMNI_OUTPUT_DIR="$BENCHMARK_OUTPUT"
export VLLM_OMNI_WARMUPS=1
export VLLM_OMNI_REPEATS=1
export VLLM_OMNI_UPLOAD_VIDEOS=0
export VLLM_OMNI_GPU_BALANCE_HEALTH_GATE=0
export VLLM_OMNI_ENABLE_DIFFUSION_PIPELINE_PROFILER=0
export VLLM_OMNI_MINIMAX_H3_NVTX_DECODE_REQUEST="$DECODE_REQUEST"
unset VLLM_OMNI_MINIMAX_H3_NVTX_DENOISE_STEP
export VLLM_OMNI_MINIMAX_H3_QKV_E4M3_EXPECTED_WHOLE_SHA256="${VLLM_OMNI_MINIMAX_H3_QKV_E4M3_EXPECTED_WHOLE_SHA256:-a06dd0f41f0df8dd3cdcf019e96191fc83543b39064ca6687003cac6f969a6d9}"
export VLLM_OMNI_MINIMAX_H3_QKV_E4M3_EXPECTED_PAYLOAD_SHA256="${VLLM_OMNI_MINIMAX_H3_QKV_E4M3_EXPECTED_PAYLOAD_SHA256:-fa7a218ad28dbc2d9447d1475351d3e82ec6591d1644667c87b2ba6329748a1e}"

echo "Profiling $BENCHMARK_ACTION"
echo "Capture range: $RANGE_NAME@$NVTX_DOMAIN"
"$NSYS_BIN" profile \
  --force-overwrite=true \
  --output="$PROFILE_PREFIX" \
  --capture-range=nvtx \
  --nvtx-capture="$RANGE_NAME@$NVTX_DOMAIN" \
  --capture-range-end=stop \
  --trace=cuda,nvtx,cudnn,cublas,osrt \
  --sample=none \
  --cpuctxsw=none \
  --trace-fork-before-exec=false \
  bash "$BENCHMARK_RUNNER" "$BENCHMARK_ACTION"

[[ -s "$PROFILE_REPORT" ]] || die "Nsight report was not produced: $PROFILE_REPORT"
[[ -s "$CASE_DIR/server.log" ]] || die "Profiled server log is missing: $CASE_DIR/server.log"
[[ "$(grep -Fc "[MiniMaxH3NVTX] capturing decode request=2 range=$RANGE_NAME" "$CASE_DIR/server.log")" == 1 ]] || die \
  "Expected exactly one rank0 decode-capture marker"

"$PYTHON_BIN" "$ANALYZER" "$PROFILE_REPORT" \
  --nvtx-range "$RANGE_NAME" \
  --nvtx-domain "$NVTX_DOMAIN" \
  --server-log "$CASE_DIR/server.log" \
  --json-output "$ANALYSIS_JSON" \
  --output "$ANALYSIS_MD" \
  --raw-nsys-stats "$ANALYSIS_STATS"

echo "Report   : $PROFILE_REPORT"
echo "Analysis : $ANALYSIS_MD"
