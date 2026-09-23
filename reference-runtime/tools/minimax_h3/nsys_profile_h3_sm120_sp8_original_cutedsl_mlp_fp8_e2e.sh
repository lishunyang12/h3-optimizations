#!/usr/bin/env bash
set -Eeuo pipefail

# Marker-free, one-request Nsight Systems capture for the quality-first FastH3
# lane.  Nsight is attached before server startup, collection starts only after
# /health succeeds, and collection is stopped after the complete MP4 response.
# No H3 step/decode/VSA NVTX selector is enabled.

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
VLLM_VENV="${VLLM_OMNI_VENV:-$WORKSPACE_DIR/vllm-omni-h3-dlo-latest/.venv-vllm028}"
PYTHON_BIN="$VLLM_VENV/bin/python"
NSYS_BIN="${NSYS_BIN:-$(command -v nsys || true)}"
BENCHMARK_DRIVER="$WORKSPACE_DIR/FastVideo/benchmark_vllm_omni_h3_comm_15s.sh"
BENCHMARK_ACTION=vllm-omni-tp1-sp8-fasth3-vsa-flashinfer-topk162-rdma-qk-direct-mlp-fp8-o-bundle-taeh3-fp16-persistent-rdma-cross-rank-audio
FLASHINFER_ROOT="$WORKSPACE_DIR/fi-original-cutedsl-pr4876"
FLASHINFER_BASE_COMMIT=750dbfd5
FLASHINFER_HEAD=ecd450bd9002e60081fa3c1eac0f16b05d06108c
FLASHINFER_WORKSPACE="$WORKSPACE_DIR/.flashinfer-pr4944-pr4876-vllm028-cache"
RDMA_SO="$FLASHINFER_WORKSPACE/.cache/flashinfer/0.6.18/120f/cached_ops/ulysses_pcie/ulysses_pcie.so"
GPU_ORDER="${VLLM_OMNI_CUDA_VISIBLE_DEVICES:-0,4,1,5,2,6,3,7}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUTPUT_DIR="${VLLM_OMNI_NSYS_OUTPUT_DIR:-$WORKSPACE_DIR/results/vllm-omni-fasth3-original-cutedsl-mlp-fp8-bf16-wire-e2e-nsys-$STAMP}"
BENCHMARK_OUTPUT="$OUTPUT_DIR/benchmark"
PROFILE_PREFIX="$OUTPUT_DIR/minimax-h3-original-cutedsl-mlp-fp8-bf16-wire-e2e"
PROFILE_REPORT="$PROFILE_PREFIX.nsys-rep"
PROFILE_SQLITE="$PROFILE_PREFIX.sqlite"
SESSION_NAME="h3e2e${STAMP//[^A-Za-z0-9]/}"
ANALYZER="$SCRIPT_DIR/analyze_h3_nsys_step.py"
ANALYSIS_MD="$OUTPUT_DIR/e2e-analysis.md"
ANALYSIS_JSON="$OUTPUT_DIR/e2e-analysis.json"
ANALYSIS_STATS="$OUTPUT_DIR/e2e-analysis-nsys-stats.txt"
NSYS_STATS="$OUTPUT_DIR/nsys-stats.txt"
CONTRACT="$OUTPUT_DIR/contract.txt"

die() {
  echo "ERROR: $*" >&2
  exit 1
}

[[ -x "$PYTHON_BIN" ]] || die "vLLM Python is missing: $PYTHON_BIN"
[[ -n "$NSYS_BIN" && -x "$NSYS_BIN" ]] || die "Nsight Systems CLI is missing"
[[ -f "$BENCHMARK_DRIVER" ]] || die "Benchmark driver is missing: $BENCHMARK_DRIVER"
[[ -f "$ANALYZER" ]] || die "Nsight analyzer is missing: $ANALYZER"
[[ "$(git -C "$FLASHINFER_ROOT" rev-parse HEAD)" == "$FLASHINFER_HEAD" ]] || die \
  "FlashInfer RDMA head changed"
[[ -f "$FLASHINFER_ROOT/flashinfer/data/csrc/ulysses_pcie.cu" ]] || die \
  "FlashInfer RDMA package-data source is missing"
[[ -s "$RDMA_SO" ]] || die "Prebuilt FlashInfer RDMA extension is missing: $RDMA_SO"
bash -n "$BENCHMARK_DRIVER"

kernel_files=(
  flashinfer/cute_dsl/sparse/bsa_attn_sm120.py
  flashinfer/cute_dsl/sparse/sm120_blk64/batched_static_scheduler.py
  flashinfer/cute_dsl/sparse/sm120_blk64/flash_fwd_sm120.py
)
for relative_path in "${kernel_files[@]}"; do
  observed="$(git -C "$FLASHINFER_ROOT" hash-object "$FLASHINFER_ROOT/$relative_path")"
  expected="$(git -C "$FLASHINFER_ROOT" rev-parse "$FLASHINFER_BASE_COMMIT:$relative_path")"
  [[ "$observed" == "$expected" ]] || die \
    "Original CuTeDSL blob mismatch for $relative_path: $observed != $expected"
done
for relative_path in "${kernel_files[@]}"; do
  if grep -Eiq 'skip[_-]?softmax|skip_unused_v|skip.*v.*tile' \
    "$FLASHINFER_ROOT/$relative_path"; then
    die "Original CuTeDSL runtime source unexpectedly contains a skipped-softmax/V-tile optimization"
  fi
done

grep -Fq "$BENCHMARK_ACTION" "$BENCHMARK_DRIVER" || die \
  "Benchmark driver does not register the MLP-only BF16-wire E2E action"

FP8_OUTPUT="$($PYTHON_BIN - <<'PY'
from vllm_omni.diffusion.models.minimax_h3.fp8_contract import (
    minimax_h3_mlp_fp8_contract,
    minimax_h3_mlp_fp8_contract_sha256,
)

contract = minimax_h3_mlp_fp8_contract()
inventory = contract["inventory"]
expected = {
    "target_module_count": 100,
    "fc1_count": 50,
    "fc2_count": 50,
    "unexpected_fp8_module_count": 0,
    "fallback_count": 0,
}
for key, value in expected.items():
    if inventory.get(key) != value:
        raise SystemExit(f"MLP FP8 inventory {key}={inventory.get(key)!r}, expected {value!r}")
if contract.get("scope") != "mlp":
    raise SystemExit(f"MLP FP8 scope changed: {contract.get('scope')!r}")
print("__MLP_FP8_CONTRACT__" + minimax_h3_mlp_fp8_contract_sha256())
PY
)" || die "MLP-only FP8 contract validation failed"
FP8_CONTRACT="$(sed -n 's/^__MLP_FP8_CONTRACT__//p' <<<"$FP8_OUTPUT" | tail -n 1)"
[[ "$FP8_CONTRACT" =~ ^[0-9a-f]{64}$ ]] || die "Invalid MLP FP8 contract digest"

echo "MiniMax-H3 marker-free full-request Nsight profile"
echo "Attention : PR #4259 original SM120 CuTeDSL, tile64, top-k 162"
echo "Compute   : MLP FC1/FC2 FP8 only; QKV projection and O projection BF16"
echo "Transport : BF16 Q/K/V and reverse-O over #4876 RDMA"
echo "Output    : TAEH3 FP16 + cross-rank audio + complete MP4"
echo "Schedule  : FastH3 VSA/Data-Free, exactly four transformer forwards"
echo "Warmup    : none"
echo "Markers   : no H3 step/decode/VSA capture markers"
echo "Report    : $PROFILE_REPORT"

if [[ "$ACTION" == check ]]; then
  "$NSYS_BIN" --version
  echo "Static preflight passed; no GPU process was started."
  exit 0
fi

busy_gpu_processes="$(nvidia-smi \
  --query-compute-apps=gpu_bus_id,pid,process_name,used_memory \
  --format=csv,noheader,nounits 2>/dev/null)" || die "Unable to query GPU process state"
if [[ -n "${busy_gpu_processes//[[:space:]]/}" ]]; then
  echo "$busy_gpu_processes" >&2
  die "Refusing to profile over or preempt another GPU process"
fi
[[ ! -e "$OUTPUT_DIR" ]] || die "Output path already exists: $OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"

export VLLM_OMNI_OUTPUT_DIR="$BENCHMARK_OUTPUT"
export VLLM_OMNI_WARMUPS=0
export VLLM_OMNI_REPEATS=1
export VLLM_OMNI_UPLOAD_VIDEOS=0
export VLLM_OMNI_GPU_IDLE_HEALTH_GATE=0
export VLLM_OMNI_GPU_BALANCE_HEALTH_GATE=0
export VLLM_OMNI_ENABLE_DIFFUSION_PIPELINE_PROFILER=0
export VLLM_OMNI_CUDA_VISIBLE_DEVICES="$GPU_ORDER"
export VLLM_OMNI_FLASHINFER_VSA_ROOT="$FLASHINFER_ROOT"
export VLLM_OMNI_FLASHINFER_VSA_WORKSPACE="$FLASHINFER_WORKSPACE"
export VLLM_OMNI_FLASHINFER_ULYSSES_QKV_E4M3_TRANSPORT=0
export VLLM_OMNI_FASTVIDEO_VSA_SKIP_SOFTMAX_THRESHOLD_SCALE_FACTOR=0
export VLLM_OMNI_NSYS_INTERACTIVE_SESSION="$SESSION_NAME"
export VLLM_OMNI_NSYS_INTERACTIVE_OUTPUT="$PROFILE_PREFIX"
export VLLM_OMNI_NSYS_BIN="$NSYS_BIN"
export OMP_NUM_THREADS=28
export MKL_NUM_THREADS=28
export OPENBLAS_NUM_THREADS=28
unset VLLM_OMNI_MINIMAX_H3_NVTX_DENOISE_STEP
unset VLLM_OMNI_MINIMAX_H3_NVTX_DECODE_REQUEST
unset VLLM_OMNI_MINIMAX_H3_VSA_NVTX
unset VLLM_OMNI_MINIMAX_H3_QKV_E4M3_SCALE_SIDECAR
unset VLLM_OMNI_MINIMAX_H3_QKV_TRANSPORT_OBSERVER_MODE
unset VLLM_OMNI_MINIMAX_H3_QKV_TRANSPORT_OBSERVER_SNAPSHOT_DIR
unset VLLM_OMNI_MINIMAX_H3_QKV_TRANSPORT_OBSERVER_SCALES_PATH

{
  echo "utc_start=$STAMP"
  echo "benchmark_action=$BENCHMARK_ACTION"
  echo "vllm_source=$VLLM_SOURCE"
  echo "vllm_head=$(git -C "$VLLM_SOURCE" rev-parse HEAD)"
  echo "flashinfer_root=$FLASHINFER_ROOT"
  echo "flashinfer_head=$FLASHINFER_HEAD"
  echo "vsa_kernel_origin=flashinfer_pr4259_commit_$FLASHINFER_BASE_COMMIT"
  for relative_path in "${kernel_files[@]}"; do
    echo "vsa_blob_${relative_path//\//_}=$(git -C "$FLASHINFER_ROOT" hash-object "$FLASHINFER_ROOT/$relative_path")"
  done
  echo "schedule=fastH3_vsa_datafree_4_forward"
  echo "parallelism=TP1_SP8"
  echo "vsa_topk=162"
  echo "fp8_scope=mlp_fc1_fc2_only"
  echo "fp8_target_modules=100"
  echo "fp8_contract_sha256=$FP8_CONTRACT"
  echo "qkv_projection_dtype=bf16"
  echo "qkv_wire_dtype=bf16"
  echo "o_projection_dtype=bf16"
  echo "reverse_o_wire_dtype=bf16"
  echo "qkv_e4m3_transport=0"
  echo "skip_softmax=0"
  echo "request_warmups=0"
  echo "request_repeats=1"
  echo "custom_h3_nvtx_markers=0"
  echo "nsys_control=external_interactive_start_after_health_stop_after_mp4"
  echo "trace=cuda,nvtx,cudnn,cublas,osrt"
  echo "omp_threads_per_worker=28"
} >"$CONTRACT"

bash "$BENCHMARK_DRIVER" "$BENCHMARK_ACTION" 2>&1 | tee "$OUTPUT_DIR/runner.log"

[[ -s "$PROFILE_REPORT" ]] || die "Nsight report was not produced"
CASE_DIR="$BENCHMARK_OUTPUT/$BENCHMARK_ACTION"
SERVER_LOG="$CASE_DIR/server.log"
[[ -s "$SERVER_LOG" ]] || die "Server log is missing"
[[ -s "$CASE_DIR/out_t2va1.mp4" ]] || die "Complete MP4 is missing"

for required in \
  'FastH3 adapter active: sigma points [0.999, 0.749, 0.5, 0.25, 0.0] for 4 transformer forwards' \
  'FASTVIDEO_VSA H3 compute kernel: FlashInfer vsa_sm120_blk64_cute_dsl' \
  'MINIMAX_H3_MLP_FP8_AUDIT_V1' \
  'target_modules=100 fc1=50 fc2=50' \
  'dtype=torch.bfloat16' \
  'MINIMAX_H3_TAEH3_CROSS_RANK_AUDIO_DECODE_ACTIVE_V1'; do
  grep -Fq "$required" "$SERVER_LOG" || die "Required runtime proof is missing: $required"
done
for forbidden in \
  MINIMAX_H3_ALL_MAIN_FP8_AUDIT_V1 \
  MINIMAX_H3_MLP_OUT_FP8_AUDIT_V1 \
  MINIMAX_H3_MLP_OUT_QKV_FP8_AUDIT_V1 \
  'MiniMax H3 E4M3 QKV transport scales installed:' \
  'FASTVIDEO_VSA H3 FP8 QKV transport receive path:' \
  'skip-softmax' \
  '[MiniMaxH3NVTX]'; do
  if grep -Fq "$forbidden" "$SERVER_LOG"; then
    die "Forbidden runtime path appeared: $forbidden"
  fi
done
[[ "$(grep -Fc 'MINIMAX_H3_MLP_FP8_AUDIT_V1' "$SERVER_LOG")" == 8 ]] || die \
  "MLP-only FP8 audit did not cover exactly eight ranks"

"$NSYS_BIN" export --type=sqlite --force-overwrite=true \
  --output="$PROFILE_SQLITE" "$PROFILE_REPORT" >/dev/null
[[ -s "$PROFILE_SQLITE" ]] || die "Nsight SQLite export is missing"
"$NSYS_BIN" stats \
  --report nvtx_pushpop_sum,nvtx_gpu_proj_sum,cuda_gpu_kern_sum,cuda_gpu_mem_time_sum,cuda_api_sum \
  "$PROFILE_REPORT" >"$NSYS_STATS"
"$PYTHON_BIN" "$ANALYZER" "$PROFILE_REPORT" \
  --nvtx-range none \
  --server-log "$SERVER_LOG" \
  --barriers-per-layer 8 \
  --top 100 \
  --json-output "$ANALYSIS_JSON" \
  --output "$ANALYSIS_MD" \
  --raw-nsys-stats "$ANALYSIS_STATS"

sha256sum "$PROFILE_REPORT" "$PROFILE_SQLITE" "$CASE_DIR/out_t2va1.mp4" \
  "$ANALYSIS_JSON" "$ANALYSIS_MD" >"$OUTPUT_DIR/SHA256SUMS.txt"
echo "RESULT_DIR=$OUTPUT_DIR"
echo "NSYS_REPORT=$PROFILE_REPORT"
echo "NSYS_SQLITE=$PROFILE_SQLITE"
echo "MP4=$CASE_DIR/out_t2va1.mp4"
echo "ANALYSIS=$ANALYSIS_MD"
