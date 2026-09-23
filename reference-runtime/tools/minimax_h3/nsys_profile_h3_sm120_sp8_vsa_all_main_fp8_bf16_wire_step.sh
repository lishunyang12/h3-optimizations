#!/usr/bin/env bash
set -Eeuo pipefail

# Capture one complete DiT update from the accepted MiniMax-H3 quality lane:
#
#   * vLLM-Omni, FastH3 VSA/Data-Free, four DiT forwards
#   * TP1/SP8 on eight RTX PRO 5000 (SM120) GPUs
#   * FlashInfer PR #4944 CuTeDSL VSA, tile 64, top-k 162
#   * all-main FP8 linear compute (250 main-block linears)
#   * BF16 Q/K wire transport -- E4M3 QKV transport is forbidden
#   * PR #4876 RDMA, Q/K and reverse-O producer-direct
#   * direct ordered q2k and reverse-O bundle, Moon kmax 279
#
# Nsight starts tracing at the rank-0 NVTX range for denoise step 3 and uses
# stop-shutdown after the range closes.  The request is therefore intentionally
# interrupted before video decode/MP4 output.  This is a DiT/VSA diagnostic,
# not a complete-response latency run.
#
# The canonical benchmark driver remains the source of truth for dynamic AdaLN
# binding and the all-main FP8 quantization contract.  Its exact bytes are
# pinned below, and this wrapper independently regenerates and records both
# contracts before launching it.  This prevents a changed driver from silently
# turning the diagnostic back into the rejected E4M3-wire lane.
#
# Usage:
#   bash tools/minimax_h3/nsys_profile_h3_sm120_sp8_vsa_all_main_fp8_bf16_wire_step.sh check
#   bash tools/minimax_h3/nsys_profile_h3_sm120_sp8_vsa_all_main_fp8_bf16_wire_step.sh run

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
ANALYZER="$SCRIPT_DIR/analyze_h3_nsys_step.py"
BENCHMARK_DRIVER="$WORKSPACE_DIR/FastVideo/benchmark_vllm_omni_h3_comm_15s.sh"
BENCHMARK_ACTION="vllm-omni-tp1-sp8-fasth3-vsa-flashinfer-topk162-rdma-qk-direct-quality-moon-all-main-fp8-bf16-qkv-o-bundle-full-vae"
PROMPT_FILE="$SCRIPT_DIR/prompts/moon_teahouse_signal_15s.txt"
MEDIA_PROFILE="$SCRIPT_DIR/prompts/moon_teahouse_signal_15s.media-profile.json"
FASTH3_ADAPTER="${VLLM_OMNI_FASTH3_LORA:-$WORKSPACE_DIR/FastH3-LoRA/vsa-datafree/adapter_model.safetensors}"
ADALN_CACHE="${MINIMAX_H3_ADALN_CACHE:-$WORKSPACE_DIR/cache/minimax-h3-fasth3-vsa-datafree-42dc502a-t2va-4step-shift12-3.safetensors}"

BENCHMARK_DRIVER_SHA256=2ada9d3e8ce1617730c8cf516e83ea5e4a393612a9349dbc529ad072cf6f441e
PROMPT_FILE_SHA256=42f62c2ecb9f90880041d51d3fa38fa6c236945ba80b6a862e080151824a50c2
PROMPT_PAYLOAD_SHA256=5abb96b31333eea455c241aa17e61d1eaf75d7e78625c80f17302b34ce11fad3
MEDIA_PROFILE_SHA256=19c8ebdf5823294ddeb65cb51e660fb76bbefbd5c41f24d1d3d974b065ab419b
FASTH3_ADAPTER_SIZE=5339117712
FASTH3_ADAPTER_SHA256=42dc502a2078f166c396a1fa75f29728d1844363652d345d5ef3e2b444ed6470

GPU_ORDER="${VLLM_OMNI_CUDA_VISIBLE_DEVICES:-0,4,1,5,2,6,3,7}"
CAPTURE_STEP=3
RANGE_NAME=minimax_h3.denoise.step_03
NVTX_DOMAIN=vllm_omni.minimax_h3
VSA_NVTX_DOMAIN=vllm_omni.minimax_h3.vsa
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUTPUT_DIR="${VLLM_OMNI_NSYS_OUTPUT_DIR:-$WORKSPACE_DIR/results/vllm-omni-fasth3-vsa-all-main-fp8-bf16-wire-o-bundle-sm120-sp8-step03-nsys-$STAMP}"
BENCHMARK_OUTPUT="$OUTPUT_DIR/benchmark-run"
PROFILE_PREFIX="$OUTPUT_DIR/minimax-h3-vsa-all-main-fp8-bf16-wire-sp8-step03"
PROFILE_REPORT="$PROFILE_PREFIX.nsys-rep"
PROFILE_SQLITE="$PROFILE_PREFIX.sqlite"
NSYS_LOG="$OUTPUT_DIR/nsys-profile.log"
STATS_LOG="$OUTPUT_DIR/nsys-stats.txt"
ANALYSIS_MD="$OUTPUT_DIR/h3-analysis.md"
ANALYSIS_JSON="$OUTPUT_DIR/h3-analysis.json"
ANALYSIS_STATS="$OUTPUT_DIR/h3-analysis-nsys-stats.txt"
VALIDATION_LOG="$OUTPUT_DIR/stack-validation.txt"
CONTRACT_LOG="$OUTPUT_DIR/contract.txt"

die() {
  echo "ERROR: $*" >&2
  exit 1
}

sha256_of() {
  local value
  value="$(sha256sum "$1")"
  printf '%s\n' "${value%% *}"
}

require_sha256() {
  local path="$1"
  local expected="$2"
  local observed
  observed="$(sha256_of "$path")"
  [[ "$observed" == "$expected" ]] || die \
    "SHA256 mismatch for $path: observed=$observed expected=$expected"
}

for command_name in git grep nvidia-smi ps setsid sha256sum; do
  command -v "$command_name" >/dev/null 2>&1 || die \
    "Required command is missing: $command_name"
done
[[ -n "$NSYS_BIN" && -x "$NSYS_BIN" ]] || die \
  "Nsight Systems CLI (nsys) is missing"
[[ -x "$PYTHON_BIN" ]] || die "vLLM Python is missing: $PYTHON_BIN"
[[ -f "$ANALYZER" ]] || die "H3 Nsight analyzer is missing: $ANALYZER"
[[ -f "$BENCHMARK_DRIVER" ]] || die \
  "Canonical H3 benchmark driver is missing: $BENCHMARK_DRIVER"
[[ -f "$PROMPT_FILE" ]] || die "Moon prompt is missing: $PROMPT_FILE"
[[ -f "$MEDIA_PROFILE" ]] || die "Moon media profile is missing: $MEDIA_PROFILE"
[[ -f "$FASTH3_ADAPTER" ]] || die "FastH3 adapter is missing: $FASTH3_ADAPTER"
[[ -f "$ADALN_CACHE" ]] || die "FastH3 AdaLN cache is missing: $ADALN_CACHE"

require_sha256 "$BENCHMARK_DRIVER" "$BENCHMARK_DRIVER_SHA256"
require_sha256 "$PROMPT_FILE" "$PROMPT_FILE_SHA256"
require_sha256 "$MEDIA_PROFILE" "$MEDIA_PROFILE_SHA256"
observed_adapter_size="$(stat -c %s "$FASTH3_ADAPTER")"
[[ "$observed_adapter_size" == "$FASTH3_ADAPTER_SIZE" ]] || die \
  "FastH3 adapter size $observed_adapter_size != $FASTH3_ADAPTER_SIZE"
require_sha256 "$FASTH3_ADAPTER" "$FASTH3_ADAPTER_SHA256"

PROMPT="$(<"$PROMPT_FILE")"
observed_prompt_payload_sha="$(printf '%s' "$PROMPT" | sha256sum)"
observed_prompt_payload_sha="${observed_prompt_payload_sha%% *}"
[[ "$observed_prompt_payload_sha" == "$PROMPT_PAYLOAD_SHA256" ]] || die \
  "Moon prompt payload SHA256 $observed_prompt_payload_sha != $PROMPT_PAYLOAD_SHA256"

MEDIA_PROFILE_PATH="$MEDIA_PROFILE" \
EXPECTED_PROMPT_SHA256="$PROMPT_PAYLOAD_SHA256" \
"$PYTHON_BIN" - <<'PY'
import json
import os

path = os.environ["MEDIA_PROFILE_PATH"]
profile = json.load(open(path, encoding="utf-8"))
expected = {
    "width": 1280,
    "height": 720,
    "effective_width": 1280,
    "effective_height": 704,
    "fps": 24,
    "num_frames": 362,
    "requested_duration_seconds": 15.0,
    "seed": 1101,
    "video_shape": [107, 22, 40],
    "video_tile_count": 1620,
    "prefix_tile_count": 27,
    "compact_row_count": 95823,
    "padded_row_count": 105408,
    "prompt_sha256": os.environ["EXPECTED_PROMPT_SHA256"],
}
for key, value in expected.items():
    if profile.get(key) != value:
        raise SystemExit(
            f"Moon media profile {key}={profile.get(key)!r}, expected {value!r}"
        )
if profile["padded_row_count"] != (
    profile["video_tile_count"] + profile["prefix_tile_count"]
) * 64:
    raise SystemExit("Moon tile-64 padded geometry is inconsistent")
PY

# The accepted BF16-wire lane delegates launch details to this registered
# action.  Pin both its registration and its explicit E4M3=0 assertion.
grep -Fq "$BENCHMARK_ACTION" "$BENCHMARK_DRIVER" || die \
  "Canonical benchmark no longer registers the BF16-wire Moon action"
grep -Fq 'Moon FP8-compute diagnostic requires BF16 QKV wire, O-bundle, and the full H3 VAE' \
  "$BENCHMARK_DRIVER" || die \
  "Canonical benchmark lost the BF16-wire fail-closed assertion"
grep -Fq 'VLLM_OMNI_MINIMAX_H3_VSA_NVTX' \
  "$VLLM_SOURCE/vllm_omni/diffusion/attention/backends/fastvideo_vsa.py" || die \
  "Current VSA backend lacks nested diagnostic NVTX support"

# Regenerate the exact AdaLN binding rather than trusting a cache path alone.
ADALN_OUTPUT="$(
  ADALN_CACHE_PATH="$ADALN_CACHE" \
  FASTH3_ADAPTER_PATH="$FASTH3_ADAPTER" \
  "$PYTHON_BIN" - <<'PY'
import json
import os

from vllm_omni.diffusion.models.minimax_h3.adaln_cache import (
    prevalidate_minimax_h3_fasth3_adaln_sidecar,
)

binding = prevalidate_minimax_h3_fasth3_adaln_sidecar(
    os.environ["ADALN_CACHE_PATH"],
    adapter_path=os.environ["FASTH3_ADAPTER_PATH"],
    model_variant="fl2va",
    mode="t2va",
    base_schedule=(0.999, 0.749, 0.5, 0.25, 0.0),
    flow_shift=12.0,
    audio_flow_shift=3.0,
)
print(
    "__ADALN__"
    + json.dumps(binding.to_dict(), allow_nan=False, separators=(",", ":"), sort_keys=True)
)
PY
)" || die "FastH3 adapter-bound AdaLN prevalidation failed"
ADALN_BINDING_JSON="$(sed -n 's/^__ADALN__//p' <<<"$ADALN_OUTPUT" | tail -n 1)"
[[ -n "$ADALN_BINDING_JSON" ]] || die "AdaLN prevalidation returned no binding JSON"

# Generate and inspect the current all-main FP8 config/contract.  In this lane
# the QKV GEMM is FP8, but its epilogue output and the communicated Q/K tensors
# remain BF16.  The 50 VSA gate projections are included, hence 250 targets.
FP8_OUTPUT="$("$PYTHON_BIN" - <<'PY'
import json

from vllm_omni.diffusion.models.minimax_h3.fp8_contract import (
    MINIMAX_H3_ALL_MAIN_FP8_DIT_QUANT_MODE,
    minimax_h3_all_main_fp8_config_sha256,
    minimax_h3_all_main_fp8_contract,
    minimax_h3_all_main_fp8_contract_json,
    minimax_h3_all_main_fp8_contract_sha256,
    minimax_h3_all_main_fp8_quantization_config_json,
)

contract = minimax_h3_all_main_fp8_contract()
inventory = contract.get("inventory", {})
expected = {
    "target_module_count": 250,
    "fc1_count": 50,
    "fc2_count": 50,
    "out_proj_count": 50,
    "qkv_proj_count": 50,
    "to_gate_compress_count": 50,
    "qkv_proj_output_dtype": "bfloat16",
    "to_gate_compress_output_dtype": "bfloat16",
    "unexpected_fp8_module_count": 0,
    "fallback_count": 0,
}
for key, value in expected.items():
    if inventory.get(key) != value:
        raise SystemExit(
            f"all-main FP8 inventory {key}={inventory.get(key)!r}, expected {value!r}"
        )
if contract.get("scope") != "all-main":
    raise SystemExit("all-main FP8 contract scope changed")
if contract.get("dit_quant_mode") != MINIMAX_H3_ALL_MAIN_FP8_DIT_QUANT_MODE:
    raise SystemExit("all-main FP8 DiT quantization mode changed")
print("__FP8_CONFIG__" + minimax_h3_all_main_fp8_quantization_config_json())
print("__FP8_CONFIG_SHA256__" + minimax_h3_all_main_fp8_config_sha256())
print("__FP8_CONTRACT__" + minimax_h3_all_main_fp8_contract_json())
print("__FP8_CONTRACT_SHA256__" + minimax_h3_all_main_fp8_contract_sha256())
PY
)" || die "Dynamic all-main FP8 contract generation failed"
FP8_CONFIG_JSON="$(sed -n 's/^__FP8_CONFIG__//p' <<<"$FP8_OUTPUT" | tail -n 1)"
FP8_CONFIG_SHA256="$(sed -n 's/^__FP8_CONFIG_SHA256__//p' <<<"$FP8_OUTPUT" | tail -n 1)"
FP8_CONTRACT_JSON="$(sed -n 's/^__FP8_CONTRACT__//p' <<<"$FP8_OUTPUT" | tail -n 1)"
FP8_CONTRACT_SHA256="$(sed -n 's/^__FP8_CONTRACT_SHA256__//p' <<<"$FP8_OUTPUT" | tail -n 1)"
for required_value in \
  "$FP8_CONFIG_JSON" "$FP8_CONFIG_SHA256" \
  "$FP8_CONTRACT_JSON" "$FP8_CONTRACT_SHA256"; do
  [[ -n "$required_value" ]] || die "Dynamic all-main FP8 contract output is incomplete"
done

IFS=',' read -r -a GPU_IDS <<<"$GPU_ORDER"
[[ "${#GPU_IDS[@]}" -eq 8 ]] || die "GPU order must contain exactly eight devices"
declare -A SEEN_GPU=()
for gpu_id in "${GPU_IDS[@]}"; do
  [[ "$gpu_id" =~ ^[0-9]+$ ]] || die "Invalid GPU id: $gpu_id"
  [[ -z "${SEEN_GPU[$gpu_id]:-}" ]] || die "GPU $gpu_id occurs more than once"
  SEEN_GPU[$gpu_id]=1
done

echo "MiniMax-H3 VSA Nsight Systems step profile"
echo "Lane         : all-main FP8 compute + BF16 QKV wire (E4M3 wire forbidden)"
echo "Model        : FastH3 VSA/Data-Free, four forwards"
echo "Attention    : FlashInfer #4944 SM120 CuTeDSL, tile64, top-k 162"
echo "Topology     : TP1/SP8, GPU order $GPU_ORDER"
echo "Communication: RDMA BF16 Q/K-direct + BF16 O-direct, O-bundle kmax279"
echo "NVTX         : $RANGE_NAME@$NVTX_DOMAIN; nested VSA@$VSA_NVTX_DOMAIN"
echo "Moon payload : $PROMPT_PAYLOAD_SHA256"
echo "Report       : $PROFILE_REPORT"

if [[ "$ACTION" == check ]]; then
  "$NSYS_BIN" --version
  echo "Static preflight passed; no GPU process was started."
  exit 0
fi

busy_gpu_processes="$(nvidia-smi \
  --query-compute-apps=gpu_bus_id,pid,process_name,used_memory \
  --format=csv,noheader,nounits 2>/dev/null)" || die \
  "Unable to query GPU process state"
if [[ -n "${busy_gpu_processes//[[:space:]]/}" ]]; then
  echo "GPU compute processes are already present:" >&2
  echo "$busy_gpu_processes" >&2
  die "Refusing to profile over or preempt another GPU process"
fi
[[ ! -e "$OUTPUT_DIR" ]] || die \
  "Output path already exists; refusing stale data: $OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"

# Every value that distinguishes this lane is forced explicitly.  The
# canonical benchmark independently rejects inconsistent combinations.
export VLLM_OMNI_OUTPUT_DIR="$BENCHMARK_OUTPUT"
export VLLM_OMNI_H3_PROMPT_FILE="$PROMPT_FILE"
export VLLM_OMNI_H3_PROMPT_SHA256="$PROMPT_PAYLOAD_SHA256"
export VLLM_OMNI_H3_PREFIX_TILE_COUNT=27
export VLLM_OMNI_H3_COMPACT_ROW_COUNT=95823
export VLLM_OMNI_H3_PADDED_ROW_COUNT=105408
export VLLM_OMNI_H3_SEQUENCE_ROW_COUNT=95872
export VLLM_OMNI_H3_LOCAL_ROW_COUNT=11984
export VLLM_OMNI_H3_O_BUNDLE_KMAX=279
export VLLM_OMNI_CUDA_VISIBLE_DEVICES="$GPU_ORDER"
export VLLM_OMNI_WARMUPS=0
export VLLM_OMNI_REPEATS=1
export VLLM_OMNI_UPLOAD_VIDEOS=0
export VLLM_OMNI_ENABLE_DIFFUSION_PIPELINE_PROFILER=0
export VLLM_OMNI_MINIMAX_H3_NVTX_DENOISE_STEP="$CAPTURE_STEP"
export VLLM_OMNI_MINIMAX_H3_VSA_NVTX=1
export VLLM_OMNI_FLASHINFER_ULYSSES_QKV_E4M3_TRANSPORT=0
export VLLM_OMNI_ULYSSES_A2A_BACKEND=flashinfer-pcie
export FLASHINFER_ULYSSES_PCIE_ROUTE=rdma
export VLLM_OMNI_FLASHINFER_ULYSSES_REQUIRE_RDMA=1
export VLLM_OMNI_FLASHINFER_ULYSSES_QK_PRODUCER_DIRECT=1
export VLLM_OMNI_FLASHINFER_ULYSSES_O_PRODUCER_DIRECT=1
export VLLM_OMNI_FLASHINFER_ULYSSES_RELEASE_AFTER_DENOISE=1
export VLLM_OMNI_FASTVIDEO_VSA_FUSED_TILE_PACK=1
export VLLM_OMNI_FASTVIDEO_VSA_FUSED_UNTILE=1
export VLLM_OMNI_FASTVIDEO_VSA_DIRECT_Q2K=1
export VLLM_OMNI_FASTVIDEO_VSA_O_BUNDLE=1
export VLLM_OMNI_FASTVIDEO_VSA_DEFERRED_GATE_SP=0
export VLLM_OMNI_FASTVIDEO_VSA_FUSED_GATE_UNTILE=0
export VLLM_OMNI_FASTVIDEO_VSA_SKIP_SOFTMAX_THRESHOLD_SCALE_FACTOR=0
export VLLM_OMNI_MINIMAX_H3_QCHUNK_PIPELINE=0
export VLLM_OMNI_MINIMAX_H3_VAE_EXACT_OPS_SM120=1
export VLLM_OMNI_MINIMAX_H3_GPU_UINT8_OUTPUT=0
export VLLM_OMNI_MINIMAX_H3_CHUNKED_CPU_MP4_OUTPUT=1
export VLLM_OMNI_MINIMAX_H3_CHUNKED_CPU_MP4_SLOTS=2
export VLLM_OMNI_GPU_BALANCE_HEALTH_GATE=0
export OMP_NUM_THREADS=28
export MKL_NUM_THREADS=28
export OPENBLAS_NUM_THREADS=28
unset NVTX_DISABLE
for forbidden_env in \
  VLLM_OMNI_MINIMAX_H3_QKV_E4M3_SCALE_SIDECAR \
  VLLM_OMNI_MINIMAX_H3_QKV_TRANSPORT_OBSERVER_MODE \
  VLLM_OMNI_MINIMAX_H3_QKV_TRANSPORT_OBSERVER_SNAPSHOT_DIR \
  VLLM_OMNI_MINIMAX_H3_QKV_TRANSPORT_OBSERVER_SCALES_PATH; do
  unset "$forbidden_env"
done

{
  echo "utc_start=$STAMP"
  echo "source=$VLLM_SOURCE"
  echo "source_git_head=$(git -C "$VLLM_SOURCE" rev-parse HEAD)"
  echo "benchmark_driver=$BENCHMARK_DRIVER"
  echo "benchmark_driver_sha256=$BENCHMARK_DRIVER_SHA256"
  echo "benchmark_action=$BENCHMARK_ACTION"
  echo "prompt_file=$PROMPT_FILE"
  echo "prompt_file_sha256=$PROMPT_FILE_SHA256"
  echo "prompt_payload_sha256=$PROMPT_PAYLOAD_SHA256"
  echo "media_profile=$MEDIA_PROFILE"
  echo "media_profile_sha256=$MEDIA_PROFILE_SHA256"
  echo "request=1280x720_effective_1280x704_362_frames_24fps_seed1101"
  echo "schedule=0.999,0.749,0.5,0.25,0.0"
  echo "capture_step=3_of_4"
  echo "nvtx_range=$RANGE_NAME@$NVTX_DOMAIN"
  echo "vsa_nvtx_domain=$VSA_NVTX_DOMAIN"
  echo "parallelism=TP1_SP8"
  echo "attention=FlashInfer_PR4944_SM120_CuTeDSL_tile64_topk162"
  echo "fp8_scope=all-main"
  echo "fp8_target_module_count=250"
  echo "fp8_config_sha256=$FP8_CONFIG_SHA256"
  echo "fp8_contract_sha256=$FP8_CONTRACT_SHA256"
  echo "fp8_config=$FP8_CONFIG_JSON"
  echo "fp8_contract=$FP8_CONTRACT_JSON"
  echo "qkv_projection_output_dtype=bfloat16"
  echo "qkv_wire_dtype=bfloat16"
  echo "qkv_e4m3_transport=0"
  echo "rdma_qk_producer_direct=1"
  echo "rdma_o_producer_direct=1"
  echo "vsa_direct_q2k=1"
  echo "vsa_o_bundle=1"
  echo "vsa_o_bundle_kmax=279"
  echo "skip_softmax_threshold_scale_factor=0"
  echo "adaln_cache=$ADALN_CACHE"
  echo "adaln_binding=$ADALN_BINDING_JSON"
  echo "omp_threads_per_worker=28"
  echo "clock_policy=node_default_dynamic_unlocked"
} >"$CONTRACT_LOG"

PROFILE_PID=""
PROFILE_PGID=""

verify_owned_session() {
  local pid="$1"
  local observed_pgid=""
  local observed_sid=""
  for _ in {1..100}; do
    read -r observed_pgid observed_sid < <(ps -o pgid=,sid= -p "$pid") || true
    if [[ "$observed_pgid" == "$pid" && "$observed_sid" == "$pid" ]]; then
      return 0
    fi
    kill -0 -- "$pid" 2>/dev/null || break
    sleep 0.02
  done
  echo "ERROR: profiler pid=$pid is not its own PGID/SID " \
    "(pgid=${observed_pgid:-missing}, sid=${observed_sid:-missing})" >&2
  return 1
}

stop_owned_group() {
  local pid="${1:-}"
  local pgid="${2:-}"
  local deadline
  [[ -n "$pid" ]] || return 0
  [[ "$pid" =~ ^[1-9][0-9]*$ && "$pgid" == "$pid" && "$pid" != "$$" ]] || {
    echo "ERROR: refusing unsafe cleanup pid=$pid pgid=$pgid" >&2
    return 1
  }
  kill -0 -- "$pid" 2>/dev/null || kill -0 -- "-$pgid" 2>/dev/null || return 0
  kill -TERM -- "-$pgid" 2>/dev/null || true
  deadline=$((SECONDS + 30))
  while kill -0 -- "$pid" 2>/dev/null || kill -0 -- "-$pgid" 2>/dev/null; do
    if (( SECONDS >= deadline )); then
      kill -KILL -- "-$pgid" 2>/dev/null || true
      break
    fi
    sleep 1
  done
}

cleanup() {
  local status=$?
  trap - EXIT INT TERM
  stop_owned_group "$PROFILE_PID" "$PROFILE_PGID" || true
  exit "$status"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

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
  --wait=primary \
  bash "$BENCHMARK_DRIVER" "$BENCHMARK_ACTION" \
  >"$NSYS_LOG" 2>&1 &
PROFILE_PID=$!
if ! verify_owned_session "$PROFILE_PID"; then
  kill -TERM -- "$PROFILE_PID" 2>/dev/null || true
  wait "$PROFILE_PID" 2>/dev/null || true
  PROFILE_PID=""
  die "Nsight launcher did not establish a runner-owned session"
fi
PROFILE_PGID=$PROFILE_PID
echo "$PROFILE_PID" >"$OUTPUT_DIR/nsys.pid"
echo "$PROFILE_PGID" >"$OUTPUT_DIR/nsys.pgid"

nsys_status=0
if wait "$PROFILE_PID"; then
  :
else
  nsys_status=$?
fi
PROFILE_PID=""
PROFILE_PGID=""
echo "nsys_status=$nsys_status" >>"$CONTRACT_LOG"
[[ "$nsys_status" -eq 0 ]] || {
  tail -n 240 "$NSYS_LOG" >&2 || true
  die "Nsight Systems exited with status $nsys_status"
}
[[ -s "$PROFILE_REPORT" ]] || {
  tail -n 240 "$NSYS_LOG" >&2 || true
  die "Nsight report was not created: $PROFILE_REPORT"
}

SERVER_LOG="$BENCHMARK_OUTPUT/$BENCHMARK_ACTION/server.log"
[[ -s "$SERVER_LOG" ]] || die "Profiled vLLM server log is missing: $SERVER_LOG"

SERVER_LOG_PATH="$SERVER_LOG" \
VALIDATION_PATH="$VALIDATION_LOG" \
EXPECTED_FP8_CONFIG_SHA256="$FP8_CONFIG_SHA256" \
EXPECTED_FP8_CONTRACT_SHA256="$FP8_CONTRACT_SHA256" \
"$PYTHON_BIN" - <<'PY'
import os
import re
from collections import Counter
from pathlib import Path

text = Path(os.environ["SERVER_LOG_PATH"]).read_text(encoding="utf-8", errors="replace")
problems: list[str] = []

audit_marker = "MINIMAX_H3_ALL_MAIN_FP8_AUDIT_V1"
audit_lines = [line for line in text.splitlines() if audit_marker in line]
ranks = []
for line in audit_lines:
    match = re.search(r"\brank=([0-7])\b", line)
    if match:
        ranks.append(int(match.group(1)))
    required = (
        "scope=all-main",
        "target_modules=250",
        "qkv_proj=50",
        "qkv_proj_output_dtype=bfloat16",
        "to_gate_compress=50",
        "to_gate_compress_output_dtype=bfloat16",
        "unexpected_fp8_modules=0",
        "fallback_count=0",
        f"config_sha256={os.environ['EXPECTED_FP8_CONFIG_SHA256']}",
        f"contract_sha256={os.environ['EXPECTED_FP8_CONTRACT_SHA256']}",
        "status=ok",
    )
    missing = [token for token in required if token not in line]
    if missing:
        problems.append(f"all-main FP8 audit is missing {missing!r}")
if Counter(ranks) != Counter({rank: 1 for rank in range(8)}):
    problems.append(f"all-main FP8 audit rank coverage is {Counter(ranks)!r}")

ranked_markers = {
    "bf16_qk_direct": (
        "FlashInfer PCIe Ulysses Q/K producer-direct exchange active:",
        r"shape=\(1,\s*11984,\s*56,\s*128\).*dtype=torch\.bfloat16",
    ),
    "bf16_o_direct": (
        "FlashInfer PCIe Ulysses O producer-direct exchange active:",
        r"shape=\(1,\s*98104,\s*7,\s*128\).*dtype=torch\.bfloat16",
    ),
    "direct_q2k": (
        "FASTVIDEO_VSA H3 metadata: direct ordered q2k enabled by "
        "VLLM_OMNI_FASTVIDEO_VSA_DIRECT_Q2K=1",
        r"",
    ),
    "o_bundle_producer": (
        "FASTVIDEO_VSA H3 reverse-O bundle producer active:",
        r"fine=\(1,\s*95872,\s*7,\s*128\).*bundle=\(1,\s*98104,\s*7,\s*128\).*kmax=279",
    ),
    "o_bundle_consumer": (
        "MiniMax H3 VSA reverse-O bundle SP active:",
        r"local_gate=\(1,\s*11984,\s*56,\s*128\).*reversed_bundle=\(1,\s*12263,\s*56,\s*128\).*kmax=279",
    ),
}
for name, (marker, suffix_pattern) in ranked_markers.items():
    marker_ranks = []
    for line in text.splitlines():
        if marker not in line:
            continue
        match = re.search(r"DiffusionWorker_SP([0-7])\s+pid=", line)
        if match and (not suffix_pattern or re.search(suffix_pattern, line)):
            marker_ranks.append(int(match.group(1)))
    if Counter(marker_ranks) != Counter({rank: 1 for rank in range(8)}):
        problems.append(f"{name} rank coverage is {Counter(marker_ranks)!r}")

required_once = (
    "FastH3 adapter active: sigma points [0.999, 0.749, 0.5, 0.25, 0.0] for 4 transformer forwards",
    "FASTVIDEO_VSA H3 compute kernel: FlashInfer vsa_sm120_blk64_cute_dsl",
    "[MiniMaxH3NVTX] capturing step=3/4 range=minimax_h3.denoise.step_03",
)
for marker in required_once:
    if text.count(marker) != 1:
        problems.append(f"expected one marker {marker!r}, got {text.count(marker)}")

forbidden = (
    "FlashInfer PCIe Ulysses Q/K/V producer-direct exchange active:",
    "MiniMax H3 E4M3 QKV transport scales installed:",
    "FASTVIDEO_VSA H3 FP8 QKV transport receive path:",
    "FASTVIDEO_VSA falling back to SDPA",
    "VSA-H3 kernel failed",
    "FlashInfer PCIe Ulysses unavailable; using NCCL",
    "transport=p2p",
    "CUDA out of memory",
    "torch.OutOfMemoryError",
)
for marker in forbidden:
    if marker in text:
        problems.append(f"forbidden/failure marker appeared: {marker!r}")

report = [
    "required_lane=all-main_fp8_compute_bf16_qkv_wire",
    "qkv_e4m3_transport=0",
    f"fp8_audit_rank_counts={dict(sorted(Counter(ranks).items()))}",
]
report.extend(f"validation_error={problem}" for problem in problems)
report.append(f"validation_status={'failed' if problems else 'ok'}")
Path(os.environ["VALIDATION_PATH"]).write_text("\n".join(report) + "\n", encoding="utf-8")
if problems:
    raise SystemExit("; ".join(problems))
PY

"$NSYS_BIN" export \
  --type=sqlite \
  --force-overwrite=true \
  --output="$PROFILE_SQLITE" \
  "$PROFILE_REPORT" >/dev/null
[[ -s "$PROFILE_SQLITE" ]] || die "Nsight SQLite export is missing"

"$NSYS_BIN" stats \
  --filter-nvtx "$RANGE_NAME@$NVTX_DOMAIN" \
  --report nvtx_pushpop_sum,nvtx_gpu_proj_sum,cuda_gpu_kern_sum,cuda_gpu_mem_time_sum,cuda_api_sum \
  "$PROFILE_REPORT" >"$STATS_LOG"
for vsa_range in \
  vsa.layout.tile_pack \
  vsa.coarse.pool_qk \
  vsa.coarse.qk_scores \
  vsa.route.topk_q2k \
  vsa.fine.block_sparse_attention \
  vsa.coarse.pool_v \
  vsa.coarse.softmax \
  vsa.coarse.pv \
  vsa.output.o_bundle; do
  grep -Fq "$vsa_range" "$STATS_LOG" || die \
    "Nested VSA NVTX range is missing from Nsight stats: $vsa_range"
done

"$PYTHON_BIN" "$ANALYZER" "$PROFILE_REPORT" \
  --nvtx-range "$RANGE_NAME" \
  --nvtx-domain "$NVTX_DOMAIN" \
  --server-log "$SERVER_LOG" \
  --barriers-per-layer 8 \
  --top 80 \
  --output "$ANALYSIS_MD" \
  --json-output "$ANALYSIS_JSON" \
  --raw-nsys-stats "$ANALYSIS_STATS"

ANALYSIS_JSON_PATH="$ANALYSIS_JSON" "$PYTHON_BIN" - <<'PY'
import json
import os

result = json.load(open(os.environ["ANALYSIS_JSON_PATH"], encoding="utf-8"))
scope = result["scope"]
if scope["name"] != "minimax_h3.denoise.step_03":
    raise SystemExit(f"analyzer selected the wrong NVTX range: {scope!r}")
if scope["domain"] != "vllm_omni.minimax_h3":
    raise SystemExit(f"analyzer selected the wrong NVTX domain: {scope!r}")
categories = {item["category"]: item for item in result["kernel_categories"]}
if categories.get("vsa_attention", {}).get("count", 0) <= 0:
    raise SystemExit("analyzer did not identify the PR #4944 VSA kernel")
barriers = result["flashinfer_ulysses_barriers"]
if not barriers["available"]:
    raise SystemExit("analyzer did not identify FlashInfer RDMA barriers")
if barriers["barriers_per_layer"] != 8:
    raise SystemExit(f"expected eight Stage-2 barriers/layer, got {barriers!r}")
if barriers["devices"] != list(range(8)):
    raise SystemExit(f"expected RDMA evidence on devices 0..7, got {barriers!r}")
if barriers["observed_layer_cycles"] != 50:
    raise SystemExit(f"expected 50 complete H3 layer cycles, got {barriers!r}")
PY

sha256sum \
  "$PROFILE_REPORT" \
  "$PROFILE_SQLITE" \
  "$STATS_LOG" \
  "$ANALYSIS_MD" \
  "$ANALYSIS_JSON" \
  "$VALIDATION_LOG" \
  "$CONTRACT_LOG" \
  >"$OUTPUT_DIR/SHA256SUMS.txt"

echo "Capture complete : $PROFILE_REPORT"
echo "SQLite export    : $PROFILE_SQLITE"
echo "VSA/NVTX stats   : $STATS_LOG"
echo "Analysis         : $ANALYSIS_MD"
echo "Machine data     : $ANALYSIS_JSON"
echo "Stack validation : $VALIDATION_LOG"
echo "No MP4 is expected: stop-shutdown ends the request after denoise step 3."
