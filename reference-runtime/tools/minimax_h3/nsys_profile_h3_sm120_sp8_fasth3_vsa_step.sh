#!/usr/bin/env bash
set -Eeuo pipefail

# Capture exactly the third denoise update from the current cumulative
# MiniMax-H3 serving stack on 8x RTX PRO 5000 (SM120).  This runner is kept
# separate from nsys_profile_h3_sm120_sp8_step.sh, which intentionally remains
# the original dense Base-H3/49-forward profile contract.
#
# Current contract:
#   - FastH3 VSA/Data-Free student, four DiT forwards
#   - TP1/SP8, Qwen TP8, VAE patch-parallel 8
#   - FlashInfer #4944 SM120 CuTeDSL VSA, top-k 162
#   - exact online FP8 for 200 main-block QKV/out/MLP linears
#   - calibrated static E4M3 Q/K/V wire transport; BF16 QKV projection output
#   - FlashInfer #4876 all-RDMA Ulysses, Q/K/V producer-direct
#   - P4a fused compact->tile64, P4c fused untile, P4d O producer-direct
#   - P4e direct ordered q2k metadata (no bool-map materialization/full scan)
#   - P4f opt-in bit-exact SM120 video-VAE operators and FP16 decoder linears
#   - Stage-2 reverse-O compact-coarse piggyback (required)
#   - exact adapter-bound AdaLN sidecar
#   - persistent communicator and two-slot chunked CPU MP4 output
#
# This is an experimental performance profile, not a lossless-quality claim:
# both online DiT FP8 and E4M3 QKV transport are lossy.  GPU clocks remain at
# the node default and are deliberately neither changed nor locked here.
#
# The NVTX stop-shutdown trigger deliberately terminates the request after
# step 3.  No complete MP4 is produced or uploaded by this diagnostic run.
#
# Usage:
#   bash tools/minimax_h3/nsys_profile_h3_sm120_sp8_fasth3_vsa_step.sh check
#   bash tools/minimax_h3/nsys_profile_h3_sm120_sp8_fasth3_vsa_step.sh run

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
VLLM_BIN="$VLLM_VENV/bin/vllm"
MODEL="${VLLM_OMNI_MODEL:-$WORKSPACE_DIR/MiniMax-H3/FL2VA}"
FASTH3_ADAPTER="${VLLM_OMNI_FASTH3_LORA:-$WORKSPACE_DIR/FastH3-LoRA/vsa-datafree/adapter_model.safetensors}"
ADALN_CACHE="${MINIMAX_H3_ADALN_CACHE:-$WORKSPACE_DIR/cache/minimax-h3-fasth3-vsa-datafree-42dc502a-t2va-4step-shift12-3.safetensors}"
QKV_E4M3_SCALE_SIDECAR="${VLLM_OMNI_MINIMAX_H3_QKV_E4M3_SCALE_SIDECAR:-$WORKSPACE_DIR/cache/minimax-h3-fasth3-vsa-datafree-mlp-out-qkv-fp8-bound-qkv-e4m3-sm120-sp8-topk162-15s-seed1101.safetensors}"
QKV_RUNTIME_MANIFEST="${VLLM_OMNI_MINIMAX_H3_QKV_RUNTIME_CHECKPOINT_MANIFEST_PATH:-$WORKSPACE_DIR/cache/minimax-h3-fl2va-73372e6cf53e-qkv-input-conditioning-v1.manifest.json}"
FASTVIDEO_KERNEL_PYTHON="${VLLM_OMNI_FASTVIDEO_KERNEL_PYTHON:-$WORKSPACE_DIR/FastVideo/fastvideo-kernel/python}"
FLASHINFER_ROOT="${VLLM_OMNI_FLASHINFER_VSA_ROOT:-$WORKSPACE_DIR/fi-pr4944-pr4876-combined}"
FLASHINFER_WORKSPACE="${VLLM_OMNI_FLASHINFER_VSA_WORKSPACE:-$WORKSPACE_DIR/.flashinfer-pr4944-pr4876-vllm028-cache}"
CUDA_TOOLKIT="${VLLM_OMNI_CUDA_HOME:-$WORKSPACE_DIR/toolchains/cuda-13.2.78-full/nvidia/cu13}"
NSYS_BIN="${NSYS_BIN:-$(command -v nsys || true)}"
ANALYZER="$SCRIPT_DIR/analyze_h3_nsys_step.py"
GPU_HEALTH_GATE="${VLLM_OMNI_GPU_HEALTH_GATE_SOURCE:-$WORKSPACE_DIR/FastVideo/scripts/benchmark_h3_gpu_health_gate.py}"

FASTH3_EXPECTED_SIZE=5339117712
FASTH3_EXPECTED_SHA256=42dc502a2078f166c396a1fa75f29728d1844363652d345d5ef3e2b444ed6470
FLASHINFER_EXPECTED_VERSION=0.6.18
FLASHINFER_EXPECTED_COMMIT=ecd450bd9002e60081fa3c1eac0f16b05d06108c
FP8_SCOPE=mlp-out-qkv
FP8_CONFIG_EXPECTED_SHA256=4bc1578f27c2e0139fa781b71fe795894bb93a8d7b26e273baf130a14c8680a2
FP8_CONTRACT_EXPECTED_SHA256=b26bf3715717f29e5a2eaa13db5afc6a86ed3c4f25fe384a265ba4dccede99b7
FP8_DIT_QUANT_MODE_EXPECTED=mlp-out-qkv-fp8-e4m3fn-w-per-tensor-a-per-token-cutlass-sm120-v1
QKV_E4M3_EXPECTED_SIZE=10616
QKV_E4M3_EXPECTED_WHOLE_SHA256=5cfb3ebfdcb44b0f9a8088fa7571633d30b39646f20dca088c082f1938e50934
QKV_E4M3_EXPECTED_PAYLOAD_SHA256=bd5cd1840dd532391988599749c276f129b71d43502d8f9b623cfb6f84ae1f8d
QKV_E4M3_SOURCE_MODEL_REVISION=9bfb6693f2cf6de171db46d1aa586f67d773a1da
QKV_RUNTIME_CHECKPOINT_REVISION=73372e6cf53e414edd3ab03e357717fb0602e758
QKV_RUNTIME_MANIFEST_EXPECTED_SHA256=8e48eec173930435ca2c4f17af0c04e48cdfc68b6746c6a148c9230b800606ee
QKV_E4M3_CAMPAIGN_UUID=30518044-e7f8-4153-bd71-b6817daf84f1
QKV_E4M3_RUN_CONTRACT_SHA256=e7c9f9f201d9a35417862e26f63ce2b78ef17cc5a5ce5a7d1bf52e1ec4e41b79
QKV_E4M3_KERNEL_PROVIDER=flashinfer-pr4944-cutedsl-sm120
QKV_E4M3_KERNEL_ABI=minimax-h3-vsa-qkv-e4m3-bf16-o-v1
QKV_E4M3_PROMPT_SHA256=613e0a1beccf83566e3531852c52c2e11daffa3b255222cd67726cdf73c1a563
VSA_O_BUNDLE="${VLLM_OMNI_FASTVIDEO_VSA_O_BUNDLE:-1}"
FLASHINFER_BASE_MAX_BYTES=171114496
FLASHINFER_O_BUNDLE_MAX_BYTES=175056896
if [[ "$VSA_O_BUNDLE" == 1 ]]; then
  FLASHINFER_MAX_BYTES="$FLASHINFER_O_BUNDLE_MAX_BYTES"
else
  FLASHINFER_MAX_BYTES="$FLASHINFER_BASE_MAX_BYTES"
fi
FLASHINFER_CACHED_SO="$FLASHINFER_WORKSPACE/.cache/flashinfer/$FLASHINFER_EXPECTED_VERSION/120f/cached_ops/ulysses_pcie/ulysses_pcie.so"
GPU_ORDER="${VLLM_OMNI_CUDA_VISIBLE_DEVICES:-0,4,1,5,2,6,3,7}"
PORT="${VLLM_OMNI_PORT:-8091}"
READY_TIMEOUT="${VLLM_OMNI_READY_TIMEOUT:-3600}"
REQUEST_TIMEOUT="${VLLM_OMNI_REQUEST_TIMEOUT:-7200}"
NSYS_EXIT_TIMEOUT="${VLLM_OMNI_NSYS_EXIT_TIMEOUT:-300}"
REQUEST_CLEANUP_TIMEOUT="${VLLM_OMNI_REQUEST_CLEANUP_TIMEOUT:-15}"
VSA_DIRECT_Q2K="${VLLM_OMNI_FASTVIDEO_VSA_DIRECT_Q2K:-1}"
VAE_EXACT_OPS_SM120="${VLLM_OMNI_MINIMAX_H3_VAE_EXACT_OPS_SM120:-1}"
if [[ "$VAE_EXACT_OPS_SM120" == 1 ]]; then
  STACK_TAG=p4f-vae-exact-ops
else
  STACK_TAG=p4e-vae-reference
fi
if [[ "$VSA_O_BUNDLE" == 1 ]]; then
  STACK_TAG="$STACK_TAG-o-bundle-stage2"
fi
CAPTURE_STEP=3
RANGE_NAME=minimax_h3.denoise.step_03
NVTX_DOMAIN=vllm_omni.minimax_h3
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUTPUT_DIR="${VLLM_OMNI_NSYS_OUTPUT_DIR:-$WORKSPACE_DIR/results/vllm-omni-fasth3-vsa-$STACK_TAG-sm120-sp8-step03-nsys-$STAMP}"
PROFILE_PREFIX="$OUTPUT_DIR/minimax-fasth3-vsa-$STACK_TAG-sp8-step03"
PROFILE_REPORT="$PROFILE_PREFIX.nsys-rep"
SERVER_LOG="$OUTPUT_DIR/server-nsys.log"
REQUEST_LOG="$OUTPUT_DIR/request-curl.log"
STATS_LOG="$OUTPUT_DIR/nsys-stats.txt"
ANALYSIS_MD="$OUTPUT_DIR/h3-analysis.md"
ANALYSIS_JSON="$OUTPUT_DIR/h3-analysis.json"
ANALYSIS_STATS="$OUTPUT_DIR/h3-analysis-nsys-stats.txt"
VALIDATION_LOG="$OUTPUT_DIR/current-stack-validation.txt"

ATTENTION_CONFIG='{"default":{"backend":"FASTVIDEO_VSA","fastvideo_vsa_h3_kernel_backend":"flashinfer","fastvideo_vsa_topk":162},"per_role":{"minimax_h3.token_refiner":{"backend":"CUDNN_ATTN"}}}'
PROMPT='傍晚小厨房的真人实拍的手 与手绘发光2d动画 融合在一起的影像。夕阳余晖残留在窗边，生活感十足的小厨房里有旧木桌、洗到一半的马克杯、起雾的玻璃瓶、悬挂的抹布。画面带有智能手机单手拍摄的手抖、近距离对焦的犹豫、逆光曝光波动。要像在家中慌忙拍下某个不可思议事件的自然质感，不要广告影像的精心整理。声音只用厨房环境声与手绘生物柔和的电子音、小小的叫声。'

die() {
  echo "ERROR: $*" >&2
  exit 1
}

for command_name in curl git grep nvidia-smi numactl ps setsid sha256sum; do
  command -v "$command_name" >/dev/null 2>&1 || die "Required command is missing: $command_name"
done
[[ -n "$NSYS_BIN" && -x "$NSYS_BIN" ]] || die "Nsight Systems CLI (nsys) is missing"
[[ -x "$PYTHON_BIN" ]] || die "vLLM Python is missing: $PYTHON_BIN"
[[ -x "$VLLM_BIN" ]] || die "vLLM entrypoint is missing: $VLLM_BIN"
[[ -f "$ANALYZER" ]] || die "H3 Nsight analyzer is missing: $ANALYZER"
[[ -f "$GPU_HEALTH_GATE" ]] || die "GPU idle-health gate is missing: $GPU_HEALTH_GATE"
[[ -f "$MODEL/model_index.json" ]] || die "MiniMax-H3 FL2VA model is missing: $MODEL"
[[ -f "$MODEL/transformer/model.safetensors.index.json" ]] || die "MiniMax-H3 transformer index is missing"
[[ -f "$FASTH3_ADAPTER" ]] || die "FastH3 VSA/Data-Free adapter is missing: $FASTH3_ADAPTER"
[[ -f "$ADALN_CACHE" ]] || die "FastH3 adapter-bound AdaLN sidecar is missing: $ADALN_CACHE"
[[ -f "$QKV_E4M3_SCALE_SIDECAR" ]] || die \
  "FP8-bound QKV E4M3 scale sidecar is missing: $QKV_E4M3_SCALE_SIDECAR"
[[ -f "$QKV_RUNTIME_MANIFEST" ]] || die \
  "QKV runtime checkpoint manifest is missing: $QKV_RUNTIME_MANIFEST"
[[ -f "$FASTVIDEO_KERNEL_PYTHON/fastvideo_kernel/__init__.py" ]] || die \
  "FastVideo kernel Python source is missing: $FASTVIDEO_KERNEL_PYTHON"
[[ -f "$FLASHINFER_ROOT/flashinfer/cute_dsl/sparse/bsa_attn_sm120.py" ]] || die \
  "FlashInfer #4944 SM120 VSA kernel is missing: $FLASHINFER_ROOT"
[[ -f "$FLASHINFER_ROOT/flashinfer/_build_meta.py" ]] || die \
  "FlashInfer combined source is missing build metadata: $FLASHINFER_ROOT"
[[ -f "$FLASHINFER_CACHED_SO" ]] || die \
  "FlashInfer #4876 SM120 Ulysses artifact is missing: $FLASHINFER_CACHED_SO"
[[ -x "$CUDA_TOOLKIT/bin/nvcc" && -f "$CUDA_TOOLKIT/include/cuda.h" ]] || die \
  "Complete CUDA 13.2 toolkit is missing: $CUDA_TOOLKIT"
[[ "$PORT" =~ ^[0-9]+$ ]] || die "VLLM_OMNI_PORT must be an integer"
[[ "$READY_TIMEOUT" =~ ^[1-9][0-9]*$ ]] || die "VLLM_OMNI_READY_TIMEOUT must be positive"
[[ "$REQUEST_TIMEOUT" =~ ^[1-9][0-9]*$ ]] || die "VLLM_OMNI_REQUEST_TIMEOUT must be positive"
[[ "$NSYS_EXIT_TIMEOUT" =~ ^[1-9][0-9]*$ ]] || die \
  "VLLM_OMNI_NSYS_EXIT_TIMEOUT must be positive"
[[ "$REQUEST_CLEANUP_TIMEOUT" =~ ^[1-9][0-9]*$ ]] || die \
  "VLLM_OMNI_REQUEST_CLEANUP_TIMEOUT must be positive"
[[ "$VSA_DIRECT_Q2K" == 0 || "$VSA_DIRECT_Q2K" == 1 ]] || die \
  "VLLM_OMNI_FASTVIDEO_VSA_DIRECT_Q2K must be 0 or 1"
[[ "$VSA_O_BUNDLE" == 0 || "$VSA_O_BUNDLE" == 1 ]] || die \
  "VLLM_OMNI_FASTVIDEO_VSA_O_BUNDLE must be 0 or 1"
[[ "$VSA_O_BUNDLE" == 1 ]] || die \
  "The current-fastest nsys contract requires VLLM_OMNI_FASTVIDEO_VSA_O_BUNDLE=1"
[[ "$VSA_DIRECT_Q2K" == 1 ]] || die \
  "The current-fastest nsys contract requires VLLM_OMNI_FASTVIDEO_VSA_DIRECT_Q2K=1"
[[ "${VLLM_OMNI_FASTVIDEO_VSA_DEFERRED_GATE_SP:-0}" == 0 ]] || die \
  "The current-fastest nsys contract requires deferred-gate SP to be disabled"
[[ "${VLLM_OMNI_FASTVIDEO_VSA_FUSED_GATE_UNTILE:-0}" == 0 ]] || die \
  "The current-fastest nsys contract requires fused-gate untile to be disabled"
[[ "$VAE_EXACT_OPS_SM120" == 0 || "$VAE_EXACT_OPS_SM120" == 1 ]] || die \
  "VLLM_OMNI_MINIMAX_H3_VAE_EXACT_OPS_SM120 must be 0 or 1"
[[ "$VAE_EXACT_OPS_SM120" == 1 ]] || die \
  "The current-fastest nsys contract requires SM120 exact VAE operators"

IFS=',' read -r -a GPU_IDS <<<"$GPU_ORDER"
[[ "${#GPU_IDS[@]}" -eq 8 ]] || die "GPU order must contain exactly eight devices"
declare -A SEEN_GPU=()
for gpu_id in "${GPU_IDS[@]}"; do
  [[ "$gpu_id" =~ ^[0-9]+$ ]] || die "Invalid GPU id: $gpu_id"
  [[ -z "${SEEN_GPU[$gpu_id]:-}" ]] || die "GPU $gpu_id occurs more than once"
  SEEN_GPU[$gpu_id]=1
done

observed_adapter_size="$(stat -c %s "$FASTH3_ADAPTER")"
[[ "$observed_adapter_size" == "$FASTH3_EXPECTED_SIZE" ]] || die \
  "FastH3 adapter size $observed_adapter_size != $FASTH3_EXPECTED_SIZE"
observed_adapter_sha="$(sha256sum "$FASTH3_ADAPTER")"
observed_adapter_sha="${observed_adapter_sha%% *}"
[[ "$observed_adapter_sha" == "$FASTH3_EXPECTED_SHA256" ]] || die \
  "FastH3 adapter SHA256 $observed_adapter_sha != $FASTH3_EXPECTED_SHA256"
observed_qkv_sidecar_size="$(stat -c %s "$QKV_E4M3_SCALE_SIDECAR")"
[[ "$observed_qkv_sidecar_size" == "$QKV_E4M3_EXPECTED_SIZE" ]] || die \
  "QKV E4M3 sidecar size $observed_qkv_sidecar_size != $QKV_E4M3_EXPECTED_SIZE"
observed_qkv_sidecar_sha="$(sha256sum "$QKV_E4M3_SCALE_SIDECAR")"
observed_qkv_sidecar_sha="${observed_qkv_sidecar_sha%% *}"
[[ "$observed_qkv_sidecar_sha" == "$QKV_E4M3_EXPECTED_WHOLE_SHA256" ]] || die \
  "QKV E4M3 sidecar SHA256 $observed_qkv_sidecar_sha != $QKV_E4M3_EXPECTED_WHOLE_SHA256"
observed_prompt_sha="$(printf '%s' "$PROMPT" | sha256sum)"
observed_prompt_sha="${observed_prompt_sha%% *}"
[[ "$observed_prompt_sha" == "$QKV_E4M3_PROMPT_SHA256" ]] || die \
  "Prompt SHA256 $observed_prompt_sha != calibrated $QKV_E4M3_PROMPT_SHA256"
observed_flashinfer_commit="$(git -C "$FLASHINFER_ROOT" rev-parse HEAD)"
[[ "$observed_flashinfer_commit" == "$FLASHINFER_EXPECTED_COMMIT" ]] || die \
  "FlashInfer combined checkout is $observed_flashinfer_commit, expected $FLASHINFER_EXPECTED_COMMIT"

# Use one coherent FlashInfer tree containing both #4944 and #4876.  Import
# only FastVideo's kernel package source; importing its separate environment
# would replace this vLLM environment's CUDA-13.2 PyTorch.
export PYTHONPATH="$FLASHINFER_ROOT:$FASTVIDEO_KERNEL_PYTHON:$VLLM_SOURCE${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONNOUSERSITE=1
export FLASHINFER_WORKSPACE_BASE="$FLASHINFER_WORKSPACE"
export FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED=0
unset FLASH_ATTENTION_CUTE_DSL_CACHE_DIR

export CUDA_HOME="$CUDA_TOOLKIT"
export FLASHINFER_NVCC="$CUDA_HOME/bin/nvcc"
export FLASHINFER_EXTRA_LDFLAGS="-L$CUDA_HOME/lib"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export FLASHINFER_CUDA_ARCH_LIST=12.0f
export TORCH_CUDA_ARCH_LIST=12.0

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$GPU_ORDER"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_OMNI_VIDEO_SYNC_TIMEOUT="$REQUEST_TIMEOUT"
export VLLM_OMNI_GPU_BALANCE_HEALTH_GATE=0
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export HF_HOME="${HF_HOME:-$WORKSPACE_DIR/hf-cache}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$WORKSPACE_DIR/xdg-cache}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$WORKSPACE_DIR/torchinductor-cache}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$WORKSPACE_DIR/triton-cache}"
export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
export NCCL_DEBUG_SUBSYS="${NCCL_DEBUG_SUBSYS:-INIT,GRAPH,TUNING}"

export FASTVIDEO_VSA_SM100A=0
export VLLM_OMNI_ULYSSES_A2A_BACKEND=flashinfer-pcie
export FLASHINFER_ULYSSES_PCIE_ROUTE=rdma
export VLLM_OMNI_FLASHINFER_ULYSSES_REQUIRE_RDMA=1
export VLLM_OMNI_FLASHINFER_ULYSSES_MAX_BYTES="$FLASHINFER_MAX_BYTES"
export VLLM_OMNI_FLASHINFER_ULYSSES_QK_PRODUCER_DIRECT=1
export VLLM_OMNI_FLASHINFER_ULYSSES_O_PRODUCER_DIRECT=1
export VLLM_OMNI_FLASHINFER_ULYSSES_QKV_E4M3_TRANSPORT=1
export VLLM_OMNI_FLASHINFER_ULYSSES_RELEASE_AFTER_DENOISE=1
export VLLM_OMNI_FLASHINFER_ULYSSES_QCHUNK2_DIRECT=0
export VLLM_OMNI_FLASHINFER_ULYSSES_QCHUNK2_QKV_FUSED=0
export VLLM_OMNI_MINIMAX_H3_QCHUNK_PIPELINE=0
export VLLM_OMNI_FASTVIDEO_VSA_FUSED_TILE_PACK=1
export VLLM_OMNI_FASTVIDEO_VSA_FUSED_UNTILE=1
export VLLM_OMNI_FASTVIDEO_VSA_DIRECT_Q2K="$VSA_DIRECT_Q2K"
export VLLM_OMNI_FASTVIDEO_VSA_O_BUNDLE="$VSA_O_BUNDLE"
export VLLM_OMNI_FASTVIDEO_VSA_DEFERRED_GATE_SP=0
export VLLM_OMNI_FASTVIDEO_VSA_FUSED_GATE_UNTILE=0

export VLLM_OMNI_MINIMAX_H3_FP8_SCOPE="$FP8_SCOPE"
export VLLM_OMNI_MINIMAX_H3_FP8_CONFIG_SHA256="$FP8_CONFIG_EXPECTED_SHA256"
export VLLM_OMNI_MINIMAX_H3_FP8_CONTRACT_SHA256="$FP8_CONTRACT_EXPECTED_SHA256"
for observer_env in \
  VLLM_OMNI_MINIMAX_H3_QKV_TRANSPORT_OBSERVER_MODE \
  VLLM_OMNI_MINIMAX_H3_QKV_TRANSPORT_OBSERVER_SNAPSHOT_DIR \
  VLLM_OMNI_MINIMAX_H3_QKV_TRANSPORT_OBSERVER_SCALES_PATH \
  VLLM_OMNI_MINIMAX_H3_QKV_TRANSPORT_OBSERVER_CAMPAIGN_UUID \
  VLLM_OMNI_MINIMAX_H3_QKV_TRANSPORT_OBSERVER_RUN_CONTRACT_SHA256 \
  VLLM_OMNI_MINIMAX_H3_QKV_TRANSPORT_OBSERVER_EXPECTED_OBSERVATION_COUNT \
  VLLM_OMNI_MINIMAX_H3_QKV_TRANSPORT_OBSERVER_SOURCE_MODEL_REVISION \
  VLLM_OMNI_MINIMAX_H3_QKV_TRANSPORT_OBSERVER_RUNTIME_CHECKPOINT_REVISION \
  VLLM_OMNI_MINIMAX_H3_QKV_TRANSPORT_OBSERVER_RUNTIME_CHECKPOINT_MANIFEST_SHA256; do
  unset "$observer_env"
done

export MINIMAX_H3_VAE_NVENC_PIPELINE=0
export VLLM_OMNI_MINIMAX_H3_GPU_UINT8_OUTPUT=0
export VLLM_OMNI_MINIMAX_H3_CHUNKED_CPU_MP4_OUTPUT=1
export VLLM_OMNI_MINIMAX_H3_CHUNKED_CPU_MP4_SLOTS=2
export VLLM_OMNI_MINIMAX_H3_VAE_TEMPORAL_PRE_GATHER_PRUNE=0
export MINIMAX_H3_VAE_DECODER_STREAM_TEMPORAL_CAT=1
# This cumulative runner is the P4f action. Dispatch remains default-off for
# every process that does not explicitly export this startup-only gate.
export VLLM_OMNI_MINIMAX_H3_VAE_EXACT_OPS_SM120="$VAE_EXACT_OPS_SM120"
export VLLM_OMNI_UPLOAD_VIDEOS=0

export VLLM_OMNI_MINIMAX_H3_NVTX_DENOISE_STEP="$CAPTURE_STEP"
# NVIDIA's Python NVTX package treats every non-empty NVTX_DISABLE value,
# including "0", as disabled.  Registered strings are required by the
# --nvtx-capture trigger.
unset NVTX_DISABLE

"$PYTHON_BIN" -c 'import nvtx' >/dev/null 2>&1 || die \
  "The vLLM Python environment must provide NVIDIA's nvtx package"
"$PYTHON_BIN" -c \
  'from vllm_omni.diffusion.models.minimax_h3.chunked_cpu_output import validate_chunked_cpu_mp4_runtime; validate_chunked_cpu_mp4_runtime()' \
  >/dev/null 2>&1 || die \
  "The vLLM Python environment must provide the chunked PyAV/libx264 output path"

IMPORTED_SOURCE="$($PYTHON_BIN -c \
  'import pathlib, vllm_omni; print(pathlib.Path(vllm_omni.__file__).resolve().parents[1])' \
  | tail -n 1)"
[[ "$IMPORTED_SOURCE" == "$(realpath "$VLLM_SOURCE")" ]] || die \
  "vllm_omni imports from $IMPORTED_SOURCE, expected $VLLM_SOURCE"

EXPECTED_FLASHINFER_ROOT="$FLASHINFER_ROOT" \
EXPECTED_FLASHINFER_VERSION="$FLASHINFER_EXPECTED_VERSION" \
EXPECTED_FLASHINFER_COMMIT="$FLASHINFER_EXPECTED_COMMIT" \
EXPECTED_FASTVIDEO_ROOT="$FASTVIDEO_KERNEL_PYTHON" \
EXPECTED_CUDA_HOME="$CUDA_HOME" \
"$PYTHON_BIN" - <<'PY'
import importlib.util
import os
import re
import subprocess
from pathlib import Path

import flashinfer
import torch
from flashinfer.comm.ulysses import missing_ulysses_pcie_dependencies
from vllm_omni.diffusion.distributed.flashinfer_ulysses import (
    ensure_flashinfer_pcie_available,
)

fi_root = Path(os.environ["EXPECTED_FLASHINFER_ROOT"]).resolve()
fi_origin = Path(flashinfer.__file__).resolve()
if not fi_origin.is_relative_to(fi_root):
    raise SystemExit(f"FlashInfer imports from {fi_origin}, expected root {fi_root}")
if flashinfer.__version__ != os.environ["EXPECTED_FLASHINFER_VERSION"]:
    raise SystemExit(
        f"FlashInfer version {flashinfer.__version__!r} != "
        f"{os.environ['EXPECTED_FLASHINFER_VERSION']!r}"
    )
if flashinfer.__git_commit__ != os.environ["EXPECTED_FLASHINFER_COMMIT"]:
    raise SystemExit(
        f"FlashInfer commit {flashinfer.__git_commit__!r} != "
        f"{os.environ['EXPECTED_FLASHINFER_COMMIT']!r}"
    )
vsa_spec = importlib.util.find_spec("flashinfer.cute_dsl.sparse.bsa_attn_sm120")
if vsa_spec is None or vsa_spec.origin is None:
    raise SystemExit("FlashInfer #4944 SM120 VSA module cannot be resolved")

fastvideo_root = Path(os.environ["EXPECTED_FASTVIDEO_ROOT"]).resolve()
fastvideo_spec = importlib.util.find_spec("fastvideo_kernel")
if fastvideo_spec is None or fastvideo_spec.origin is None:
    raise SystemExit("fastvideo_kernel cannot be resolved")
fastvideo_origin = Path(fastvideo_spec.origin).resolve()
if not fastvideo_origin.is_relative_to(fastvideo_root):
    raise SystemExit(
        f"fastvideo_kernel imports from {fastvideo_origin}, expected root {fastvideo_root}"
    )

if torch.version.cuda != "13.2":
    raise SystemExit(f"Expected PyTorch CUDA 13.2, got {torch.version.cuda!r}")
nvcc = subprocess.check_output(
    [str(Path(os.environ["EXPECTED_CUDA_HOME"]) / "bin/nvcc"), "--version"],
    text=True,
)
match = re.search(r"release\s+(\d+\.\d+)", nvcc)
if match is None or match.group(1) != torch.version.cuda:
    raise SystemExit(
        f"nvcc/PyTorch CUDA mismatch: nvcc={match.group(1) if match else 'unknown'}, "
        f"torch={torch.version.cuda}"
    )
missing = missing_ulysses_pcie_dependencies()
if missing:
    raise SystemExit(f"FlashInfer PCIe JIT dependencies missing: {missing}")
ensure_flashinfer_pcie_available()
print(
    "Current FlashInfer stack verified: "
    f"origin={fi_origin} commit={flashinfer.__git_commit__} "
    f"vsa={Path(vsa_spec.origin).resolve()} fastvideo={fastvideo_origin}"
)
PY

MODEL_INDEX="$MODEL/model_index.json" "$PYTHON_BIN" - <<'PY'
import json
import os

path = os.environ["MODEL_INDEX"]
release = json.load(open(path, encoding="utf-8")).get("_minimax_h3") or {}
if str(release.get("partition", "")).lower() != "fl2va":
    raise SystemExit(f"Not an original FL2VA partition: {path}")
if release.get("base_schedule") is not None:
    raise SystemExit(f"Base checkpoint unexpectedly pins a distilled schedule: {path}")
PY

ADALN_BINDING_OUTPUT=""
if ! ADALN_BINDING_OUTPUT="$(
  ADALN_CACHE_PATH="$ADALN_CACHE" \
  FASTH3_ADAPTER_PATH="$FASTH3_ADAPTER" \
  FASTH3_EXPECTED_SHA256="$FASTH3_EXPECTED_SHA256" \
  FASTH3_EXPECTED_SIZE="$FASTH3_EXPECTED_SIZE" \
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
adapter = binding.fasth3_adapter
if adapter is None:
    raise SystemExit("FastH3 AdaLN binding contains no adapter identity")
if adapter.sha256 != os.environ["FASTH3_EXPECTED_SHA256"]:
    raise SystemExit(f"AdaLN binding adapter SHA256 mismatch: {adapter.sha256}")
if adapter.size_bytes != int(os.environ["FASTH3_EXPECTED_SIZE"]):
    raise SystemExit(f"AdaLN binding adapter size mismatch: {adapter.size_bytes}")
print(json.dumps(binding.to_dict(), allow_nan=False, separators=(",", ":"), sort_keys=True))
PY
)"; then
  die "FastH3 AdaLN sidecar prevalidation failed"
fi

# Import-time INFO messages from vLLM may be written to stdout.  Preserve them
# as diagnostics, but feed only the final non-empty line to json.loads.  The
# second validation prevents an unrelated trailing line from being accepted as
# the adapter-bound cache identity.
if ! ADALN_BINDING_JSON="$(
  ADALN_BINDING_OUTPUT="$ADALN_BINDING_OUTPUT" \
  FASTH3_EXPECTED_SHA256="$FASTH3_EXPECTED_SHA256" \
  FASTH3_EXPECTED_SIZE="$FASTH3_EXPECTED_SIZE" \
  "$PYTHON_BIN" - <<'PY'
import json
import os
import sys

lines = os.environ["ADALN_BINDING_OUTPUT"].splitlines()
if not lines:
    raise SystemExit("FastH3 AdaLN sidecar returned no stdout")
for diagnostic in lines[:-1]:
    print(diagnostic, file=sys.stderr)
candidate = lines[-1]
try:
    binding = json.loads(candidate)
except json.JSONDecodeError as error:
    raise SystemExit(
        f"final AdaLN prevalidation stdout line is not JSON: {candidate!r}"
    ) from error
if not isinstance(binding, dict):
    raise SystemExit("FastH3 AdaLN binding JSON must be an object")
adapter = binding.get("fasth3_adapter")
if not isinstance(adapter, dict):
    raise SystemExit("FastH3 AdaLN binding JSON has no adapter identity")
expected_schedule = [0.999, 0.749, 0.5, 0.25, 0.0]
expected = {
    "model_variant": "fl2va",
    "mode": "t2va",
    "base_schedule": expected_schedule,
    "flow_shift": 12.0,
    "audio_flow_shift": 3.0,
}
for key, value in expected.items():
    if binding.get(key) != value:
        raise SystemExit(
            f"FastH3 AdaLN binding field {key!r}={binding.get(key)!r}, expected {value!r}"
        )
if adapter.get("sha256") != os.environ["FASTH3_EXPECTED_SHA256"]:
    raise SystemExit("FastH3 AdaLN binding JSON adapter SHA256 mismatch")
if adapter.get("size_bytes") != int(os.environ["FASTH3_EXPECTED_SIZE"]):
    raise SystemExit("FastH3 AdaLN binding JSON adapter size mismatch")
print(json.dumps(binding, allow_nan=False, separators=(",", ":"), sort_keys=True))
PY
)"; then
  die "FastH3 AdaLN sidecar stdout did not end in the expected binding JSON"
fi

if ! FP8_CONTRACT_OUTPUT="$(
  FP8_EXPECTED_CONFIG_SHA256="$FP8_CONFIG_EXPECTED_SHA256" \
  FP8_EXPECTED_CONTRACT_SHA256="$FP8_CONTRACT_EXPECTED_SHA256" \
  FP8_EXPECTED_DIT_QUANT_MODE="$FP8_DIT_QUANT_MODE_EXPECTED" \
  "$PYTHON_BIN" - <<'PY'
import json
import os

from vllm_omni.diffusion.models.minimax_h3.fp8_contract import (
    MINIMAX_H3_MLP_OUT_QKV_FP8_DIT_QUANT_MODE,
    minimax_h3_mlp_out_qkv_fp8_config_sha256,
    minimax_h3_mlp_out_qkv_fp8_contract_json,
    minimax_h3_mlp_out_qkv_fp8_contract_sha256,
    minimax_h3_mlp_out_qkv_fp8_quantization_config_json,
)

config_json = minimax_h3_mlp_out_qkv_fp8_quantization_config_json()
config_sha256 = minimax_h3_mlp_out_qkv_fp8_config_sha256()
contract_json = minimax_h3_mlp_out_qkv_fp8_contract_json()
contract_sha256 = minimax_h3_mlp_out_qkv_fp8_contract_sha256()
dit_quant_mode = MINIMAX_H3_MLP_OUT_QKV_FP8_DIT_QUANT_MODE
if config_sha256 != os.environ["FP8_EXPECTED_CONFIG_SHA256"]:
    raise SystemExit(f"canonical FP8 config SHA256 changed: {config_sha256}")
if contract_sha256 != os.environ["FP8_EXPECTED_CONTRACT_SHA256"]:
    raise SystemExit(f"canonical FP8 contract SHA256 changed: {contract_sha256}")
if dit_quant_mode != os.environ["FP8_EXPECTED_DIT_QUANT_MODE"]:
    raise SystemExit(f"canonical FP8 DiT mode changed: {dit_quant_mode}")

config = json.loads(config_json)
contract = json.loads(contract_json)
ignored = config.get("transformer", {}).get("ignored_layers")
expected_ignored = {"condition_proj"}
for block in range(2):
    expected_ignored.update(
        f"token_refiner.blocks.{block}.{suffix}"
        for suffix in ("attn.qkv_proj", "attn.out_proj", "mlp.fc1", "mlp.fc2")
    )
expected_ignored.update(
    f"blocks.{block}.attn.to_gate_compress" for block in range(50)
)
if config.get("transformer", {}).get("method") != "fp8":
    raise SystemExit("canonical mlp-out-qkv config no longer selects FP8")
if not isinstance(ignored, list) or set(ignored) != expected_ignored:
    raise SystemExit("canonical mlp-out-qkv ignored-layer inventory changed")
inventory = contract.get("inventory", {})
expected_inventory = {
    "target_module_count": 200,
    "fc1_count": 50,
    "fc2_count": 50,
    "out_proj_count": 50,
    "qkv_proj_count": 50,
    "qkv_proj_output_dtype": "bfloat16",
    "to_gate_compress_bf16_count": 50,
    "unexpected_fp8_module_count": 0,
    "fallback_count": 0,
}
for key, expected in expected_inventory.items():
    if inventory.get(key) != expected:
        raise SystemExit(
            f"canonical mlp-out-qkv inventory {key}={inventory.get(key)!r}, "
            f"expected {expected!r}"
        )
if contract.get("scope") != "mlp-out-qkv":
    raise SystemExit("canonical FP8 contract scope is not mlp-out-qkv")
if contract.get("dit_quant_mode") != dit_quant_mode:
    raise SystemExit("canonical FP8 contract carries the wrong DiT mode")
if contract.get("quantization_config") != config:
    raise SystemExit("canonical FP8 contract/config mismatch")
print("__FP8_CONFIG__" + config_json)
print("__FP8_CONTRACT__" + contract_json)
PY
)"; then
  die "Canonical MiniMax H3 mlp-out-qkv exact-online-FP8 validation failed"
fi
FP8_CONFIG_JSON="$(sed -n 's/^__FP8_CONFIG__//p' <<<"$FP8_CONTRACT_OUTPUT")"
FP8_CONTRACT_JSON="$(sed -n 's/^__FP8_CONTRACT__//p' <<<"$FP8_CONTRACT_OUTPUT")"
[[ -n "$FP8_CONFIG_JSON" && -n "$FP8_CONTRACT_JSON" ]] || die \
  "Canonical MiniMax H3 FP8 validation returned incomplete tagged output"
"$PYTHON_BIN" -c 'import json,sys; json.loads(sys.argv[1]); json.loads(sys.argv[2])' \
  "$FP8_CONFIG_JSON" "$FP8_CONTRACT_JSON" || die \
  "Canonical MiniMax H3 FP8 validation returned invalid JSON"

if ! QKV_PREVALIDATION_OUTPUT="$(
  ADALN_BINDING_JSON="$ADALN_BINDING_JSON" \
  MODEL_ROOT="$MODEL" \
  QKV_SCALE_PATH="$QKV_E4M3_SCALE_SIDECAR" \
  QKV_EXPECTED_PAYLOAD_SHA256="$QKV_E4M3_EXPECTED_PAYLOAD_SHA256" \
  QKV_SOURCE_MODEL_REVISION="$QKV_E4M3_SOURCE_MODEL_REVISION" \
  QKV_RUNTIME_CHECKPOINT_REVISION="$QKV_RUNTIME_CHECKPOINT_REVISION" \
  QKV_RUNTIME_MANIFEST_PATH="$QKV_RUNTIME_MANIFEST" \
  QKV_RUNTIME_MANIFEST_SHA256="$QKV_RUNTIME_MANIFEST_EXPECTED_SHA256" \
  QKV_CAMPAIGN_UUID="$QKV_E4M3_CAMPAIGN_UUID" \
  QKV_RUN_CONTRACT_SHA256="$QKV_E4M3_RUN_CONTRACT_SHA256" \
  QKV_KERNEL_PROVIDER="$QKV_E4M3_KERNEL_PROVIDER" \
  QKV_KERNEL_ABI="$QKV_E4M3_KERNEL_ABI" \
  QKV_DIT_QUANT_MODE="$FP8_DIT_QUANT_MODE_EXPECTED" \
  QKV_PROMPT_SHA256="$QKV_E4M3_PROMPT_SHA256" \
  "$PYTHON_BIN" - <<'PY'
import json
import os
from pathlib import Path

import torch

from vllm_omni.diffusion.models.minimax_h3.adaln_cache import (
    MiniMaxH3AdalnCacheBinding,
)
from vllm_omni.diffusion.models.minimax_h3.qkv_transport_scales import (
    MiniMaxH3QKVTransportMediaProfile,
    MiniMaxH3QKVTransportScales,
    prevalidate_minimax_h3_qkv_transport_scale_sidecar,
    read_minimax_h3_runtime_checkpoint_manifest,
)

model_root = Path(os.environ["MODEL_ROOT"]).resolve()
metadata_root = (
    model_root.parent / ".cache" / "huggingface" / "download" / model_root.name
)
relative_paths = [Path("model_index.json")]
for directory_name in ("processor", "tokenizer", "text_encoder", "transformer"):
    directory = model_root / directory_name
    if not directory.is_dir():
        raise SystemExit(f"runtime checkpoint is missing {directory_name}/")
    relative_paths.extend(
        path.relative_to(model_root)
        for path in directory.rglob("*")
        if path.is_file() and ".cache" not in path.relative_to(model_root).parts
    )
revisions = set()
for relative_path in relative_paths:
    metadata_path = metadata_root / f"{relative_path.as_posix()}.metadata"
    if not metadata_path.is_file():
        raise SystemExit(
            f"runtime checkpoint is missing Hugging Face revision metadata: {metadata_path}"
        )
    lines = metadata_path.read_text(encoding="utf-8").splitlines()
    if not lines:
        raise SystemExit(f"empty Hugging Face revision metadata: {metadata_path}")
    revision = lines[0].strip().lower()
    if len(revision) != 40:
        raise SystemExit(f"invalid runtime checkpoint revision in {metadata_path}")
    try:
        int(revision, 16)
    except ValueError as exc:
        raise SystemExit(
            f"non-hex runtime checkpoint revision in {metadata_path}"
        ) from exc
    revisions.add(revision)
if revisions != {os.environ["QKV_RUNTIME_CHECKPOINT_REVISION"]}:
    raise SystemExit(
        f"runtime checkpoint revisions {sorted(revisions)!r} do not match the calibrated revision"
    )

manifest_digest = read_minimax_h3_runtime_checkpoint_manifest(
    os.environ["QKV_RUNTIME_MANIFEST_PATH"]
)
if manifest_digest != os.environ["QKV_RUNTIME_MANIFEST_SHA256"]:
    raise SystemExit(
        f"runtime checkpoint manifest digest {manifest_digest} does not match the calibrated digest"
    )

adaln_binding = MiniMaxH3AdalnCacheBinding.from_dict(
    json.loads(os.environ["ADALN_BINDING_JSON"])
)
adapter = adaln_binding.fasth3_adapter
if adapter is None:
    raise SystemExit("QKV transport requires an adapter-bound AdaLN sidecar")
adapter_header = json.loads(adapter.header_identity)
if adapter_header.get("base_revision") != os.environ["QKV_SOURCE_MODEL_REVISION"]:
    raise SystemExit("QKV source revision does not match the FastH3 adapter header")

media = MiniMaxH3QKVTransportMediaProfile(
    width=1280,
    height=720,
    fps=24,
    num_frames=362,
    requested_duration_seconds=15.0,
    seed=1101,
    prompt_sha256=os.environ["QKV_PROMPT_SHA256"],
    effective_width=1280,
    effective_height=704,
    video_shape=(107, 22, 40),
    video_tile_count=1620,
    prefix_tile_count=21,
    compact_row_count=95487,
    padded_row_count=105024,
)
binding = prevalidate_minimax_h3_qkv_transport_scale_sidecar(
    os.environ["QKV_SCALE_PATH"],
    source_model_revision=os.environ["QKV_SOURCE_MODEL_REVISION"],
    runtime_checkpoint_revision=os.environ["QKV_RUNTIME_CHECKPOINT_REVISION"],
    runtime_checkpoint_manifest_sha256=manifest_digest,
    campaign_uuid=os.environ["QKV_CAMPAIGN_UUID"],
    run_contract_sha256=os.environ["QKV_RUN_CONTRACT_SHA256"],
    expected_observation_count=4,
    adapter=adapter,
    base_schedule=(0.999, 0.749, 0.5, 0.25, 0.0),
    flow_shift=12.0,
    audio_flow_shift=3.0,
    kernel_provider=os.environ["QKV_KERNEL_PROVIDER"],
    kernel_abi=os.environ["QKV_KERNEL_ABI"],
    media_profile=media,
    expected_payload_sha256=os.environ["QKV_EXPECTED_PAYLOAD_SHA256"],
    dit_quant_mode=os.environ["QKV_DIT_QUANT_MODE"],
)
loader = MiniMaxH3QKVTransportScales(
    os.environ["QKV_SCALE_PATH"], runtime_binding=binding
)
rows = loader.load()
audit = loader.audit_tensors
required_audit = {
    "observed_absmax",
    "finite_count",
    "nonfinite_count",
    "numel",
    "observation_count",
    "saturation_count",
}
if set(audit) != required_audit:
    raise SystemExit(
        "formal QKV E4M3 sidecar audit tensor inventory mismatch: "
        f"missing={sorted(required_audit - set(audit))!r}, "
        f"unexpected={sorted(set(audit) - required_audit)!r}"
    )
if bool((audit["nonfinite_count"] != 0).any()):
    raise SystemExit("QKV E4M3 calibration audit contains non-finite observations")
if bool((audit["saturation_count"] != 0).any()):
    raise SystemExit("QKV E4M3 validation audit contains saturated values")
if not torch.equal(audit["finite_count"] + audit["nonfinite_count"], audit["numel"]):
    raise SystemExit("QKV E4M3 audit sample counts are inconsistent")
if not bool((audit["observation_count"] == 32).all()):
    raise SystemExit("QKV E4M3 sidecar must contain 8 ranks x 4 observations per cell")
scales = torch.tensor(rows, dtype=torch.float32)
summary = {
    "audit_tensor_names": sorted(audit),
    "finite_count": int(audit["finite_count"].sum().item()),
    "nonfinite_count": int(audit["nonfinite_count"].sum().item()),
    "observation_count_min": int(audit["observation_count"].min().item()),
    "observation_count_max": int(audit["observation_count"].max().item()),
    "observed_absmax_max": float(audit["observed_absmax"].max().item()),
    "saturation_count": int(audit["saturation_count"].sum().item()),
    "scale_shape": list(scales.shape),
    "scale_order": ["q", "k", "v"],
    "scale_min": float(scales.min().item()),
    "scale_max": float(scales.max().item()),
}
if summary["finite_count"] != 410674790400:
    raise SystemExit(f"QKV E4M3 finite sample count changed: {summary['finite_count']}")
if summary["observed_absmax_max"] != 1944.0:
    raise SystemExit(f"QKV E4M3 observed absmax changed: {summary['observed_absmax_max']}")
print(
    "__QKV_E4M3_BINDING__"
    + json.dumps(binding.to_dict(), allow_nan=False, separators=(",", ":"), sort_keys=True)
)
print(
    "__QKV_E4M3_AUDIT__"
    + json.dumps(summary, allow_nan=False, separators=(",", ":"), sort_keys=True)
)
PY
)"; then
  die "Adapter/media/runtime-bound QKV E4M3 sidecar prevalidation failed"
fi
QKV_E4M3_BINDING_JSON="$(
  sed -n 's/^__QKV_E4M3_BINDING__//p' <<<"$QKV_PREVALIDATION_OUTPUT"
)"
QKV_E4M3_AUDIT_JSON="$(
  sed -n 's/^__QKV_E4M3_AUDIT__//p' <<<"$QKV_PREVALIDATION_OUTPUT"
)"
[[ -n "$QKV_E4M3_BINDING_JSON" && -n "$QKV_E4M3_AUDIT_JSON" ]] || die \
  "QKV E4M3 sidecar prevalidation returned incomplete tagged output"
"$PYTHON_BIN" -c 'import json,sys; json.loads(sys.argv[1]); json.loads(sys.argv[2])' \
  "$QKV_E4M3_BINDING_JSON" "$QKV_E4M3_AUDIT_JSON" || die \
  "QKV E4M3 sidecar prevalidation returned invalid JSON"

CACHE_CONFIG="$($PYTHON_BIN -c \
  'import json,sys; print(json.dumps({"minimax_h3_adaln_cache_path":sys.argv[1],"minimax_h3_adaln_cache_binding":json.loads(sys.argv[2]),"minimax_h3_qkv_transport_scale_path":sys.argv[3],"minimax_h3_qkv_transport_scale_binding":json.loads(sys.argv[4]),"minimax_h3_qkv_transport_runtime_checkpoint_revision":sys.argv[5],"minimax_h3_qkv_transport_runtime_checkpoint_manifest_path":sys.argv[6]},separators=(",",":"),sort_keys=True))' \
  "$ADALN_CACHE" "$ADALN_BINDING_JSON" \
  "$QKV_E4M3_SCALE_SIDECAR" "$QKV_E4M3_BINDING_JSON" \
  "$QKV_RUNTIME_CHECKPOINT_REVISION" "$QKV_RUNTIME_MANIFEST")"

help_output="$($VLLM_BIN serve --omni --help=all 2>&1)" || die "Unable to inspect vLLM serve flags"
for required_flag in --omni --tensor-parallel-size --usp --ring \
  --ulysses-mode --ulysses-a2a-permute --text-encoder-tp-size \
  --vae-patch-parallel-size --diffusion-attention-config --task-type \
  --diffusion-quantization-config --lora-path --cache-config; do
  [[ "$help_output" == *"$required_flag"* ]] || die "vLLM build lacks $required_flag"
done

SERVER_ARGS=(
  serve "$MODEL"
  --omni
  --host 127.0.0.1
  --port "$PORT"
  --trust-remote-code
  --num-gpus 8
  --tensor-parallel-size 1
  --usp 8
  --ring 1
  --ulysses-mode strict
  --ulysses-a2a-permute
  --text-encoder-tp-size 8
  --vae-patch-parallel-size 8
  --vae-parallel-mode tile
  --vae-use-tiling
  --diffusion-attention-config "$ATTENTION_CONFIG"
  --diffusion-quantization-config "$FP8_CONFIG_JSON"
  --cache-backend none
  --enable-diffusion-pipeline-profiler
  --cache-config "$CACHE_CONFIG"
  --task-type fl2va
  --lora-path "$FASTH3_ADAPTER"
  --lora-scale 1.0
)

echo "MiniMax-H3 current cumulative single-step Nsight Systems capture"
echo "Source       : $VLLM_SOURCE"
echo "Model        : $MODEL + FastH3 VSA/Data-Free"
echo "NVTX range   : $RANGE_NAME@$NVTX_DOMAIN (DiT rank 0 only)"
echo "Trace        : cuda,nvtx,cudnn,cublas,osrt"
echo "Topology     : TP1/SP8, Qwen TP8, VAE PP8"
echo "Attention    : FlashInfer #4944 SM120 CuTeDSL VSA, top-k 162"
echo "Compute      : exact online FP8, scope=mlp-out-qkv, 200 main-block linears"
echo "Communication: FlashInfer #4876 all-RDMA, E4M3 Q/K/V-direct, BF16 O-direct"
echo "Layout       : P4a tile pack + P4c untile + P4e direct-q2k=$VSA_DIRECT_Q2K"
echo "Stage-2      : reverse-O compact-coarse bundle=$VSA_O_BUNDLE, capacity=$FLASHINFER_MAX_BYTES"
echo "QKV sidecar  : $QKV_E4M3_SCALE_SIDECAR"
echo "Clock policy : node-default dynamic clocks; no lock/change"
echo "Output route : chunked CPU MP4 slots=2 (initialized, not reached)"
echo "Upload       : disabled"
echo "Report       : $PROFILE_REPORT"
printf 'Server       :'
printf ' %q' numactl --interleave=0,1 "$VLLM_BIN" "${SERVER_ARGS[@]}"
printf '\n'

if [[ "$ACTION" == check ]]; then
  "$NSYS_BIN" --version
  echo "Preflight only; pass 'run' to launch the eight-GPU capture."
  exit 0
fi

if ! busy_gpu_processes="$(nvidia-smi \
  --query-compute-apps=gpu_bus_id,pid,process_name,used_memory \
  --format=csv,noheader,nounits 2>/dev/null)"; then
  die "Unable to query GPU process state; refusing to launch without an idle-GPU check"
fi
if [[ -n "${busy_gpu_processes//[[:space:]]/}" ]]; then
  echo "GPU compute processes are already present:" >&2
  echo "$busy_gpu_processes" >&2
  die "Refusing to preempt or profile over another GPU process"
fi
if PORT_TO_CHECK="$PORT" "$PYTHON_BIN" - <<'PY'
import os
import socket

with socket.socket() as sock:
    sock.settimeout(0.5)
    status = sock.connect_ex(("127.0.0.1", int(os.environ["PORT_TO_CHECK"])))
raise SystemExit(0 if status == 0 else 1)
PY
then
  die "Port $PORT already has a listener; refusing to interfere"
fi

[[ ! -e "$OUTPUT_DIR" ]] || die \
  "Output path already exists; refusing to accept a stale report: $OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"

# Keep the benchmark contract at node-default dynamic clocks.  This gate is
# read-only: it records telemetry and rejects a busy, hot, throttled,
# non-default-clock/power, or otherwise inconsistent node without attempting
# to change clocks or power limits.  Volatile uncorrectable ECC is checked
# separately because the shared idle-health helper intentionally does not
# query ECC counters.
"$PYTHON_BIN" "$GPU_HEALTH_GATE" idle \
  --gpu-ids "$GPU_ORDER" \
  --samples 3 \
  --grace-seconds 5 \
  --interval-seconds 1 \
  --max-utilization-percent 5 \
  --max-memory-mib 256 \
  --max-temperature-c 70 \
  --output "$OUTPUT_DIR/gpu-idle-health.json" || die \
  "GPU idle/default-clock/temperature health gate failed"
nvidia-smi -q >"$OUTPUT_DIR/nvidia-smi-q-preflight.txt" || die \
  "Unable to save raw nvidia-smi -q preflight telemetry"
nvidia-smi \
  --query-gpu=index,uuid,pci.bus_id,temperature.gpu,ecc.errors.uncorrected.volatile.total \
  --format=csv,noheader,nounits \
  >"$OUTPUT_DIR/nvidia-smi-ecc-preflight.csv" || die \
  "Unable to query volatile uncorrectable ECC counters"
GPU_ECC_REPORT="$OUTPUT_DIR/nvidia-smi-ecc-preflight.csv" \
EXPECTED_GPU_IDS="$GPU_ORDER" "$PYTHON_BIN" - <<'PY'
import csv
import os
from pathlib import Path

path = Path(os.environ["GPU_ECC_REPORT"])
expected = {int(value) for value in os.environ["EXPECTED_GPU_IDS"].split(",")}
rows = list(csv.reader(path.read_text(encoding="utf-8").splitlines()))
if len(rows) != 8:
    raise SystemExit(f"expected eight ECC telemetry rows, got {len(rows)}")
observed = set()
for row_number, raw in enumerate(rows, 1):
    values = [value.strip() for value in raw]
    if len(values) != 5:
        raise SystemExit(f"ECC telemetry row {row_number} has {len(values)} columns")
    index_raw, uuid, bus_id, temperature_raw, ecc_raw = values
    try:
        index = int(index_raw)
        temperature = int(temperature_raw)
        volatile_uncorrectable_ecc = int(ecc_raw)
    except ValueError as exc:
        raise SystemExit(
            f"ECC telemetry row {row_number} contains a non-integer metric: {values!r}"
        ) from exc
    if not uuid or not bus_id:
        raise SystemExit(f"ECC telemetry row {row_number} has missing identity: {values!r}")
    if index in observed:
        raise SystemExit(f"duplicate GPU index in ECC telemetry: {index}")
    observed.add(index)
    if temperature > 70:
        raise SystemExit(f"GPU {index} idle temperature {temperature} C exceeds 70 C")
    if volatile_uncorrectable_ecc != 0:
        raise SystemExit(
            f"GPU {index} has {volatile_uncorrectable_ecc} volatile uncorrectable ECC errors"
        )
if observed != expected:
    raise SystemExit(
        f"ECC telemetry GPU inventory {sorted(observed)} != expected {sorted(expected)}"
    )
PY
{
  echo "utc_start=$STAMP"
  echo "source=$VLLM_SOURCE"
  echo "source_git_head=$(git -C "$VLLM_SOURCE" rev-parse HEAD)"
  echo "model=$MODEL"
  echo "checkpoint=original_FL2VA_BF16_plus_FastH3_vsa-datafree_fused_student"
  echo "fasth3_adapter=$FASTH3_ADAPTER"
  echo "fasth3_adapter_size=$FASTH3_EXPECTED_SIZE"
  echo "fasth3_adapter_sha256=$FASTH3_EXPECTED_SHA256"
  echo "fasth3_base_schedule=0.999 0.749 0.5 0.25 0.0"
  echo "adaln_cache=$ADALN_CACHE"
  echo "fasth3_adaln_binding=$ADALN_BINDING_JSON"
  echo "request=1280x720_effective_1280x704,24fps,362_frames,15s,4_forwards,shift_12_3,seed_1101"
  echo "gpu_clock_policy=node_default_dynamic_unlocked"
  echo "gpu_clock_mutation=forbidden"
  echo "gpu_balance_health_gate=0"
  echo "gpu_idle_health_gate=1"
  echo "gpu_idle_health_report=$OUTPUT_DIR/gpu-idle-health.json"
  echo "gpu_raw_nvidia_smi_report=$OUTPUT_DIR/nvidia-smi-q-preflight.txt"
  echo "gpu_volatile_uncorrectable_ecc_report=$OUTPUT_DIR/nvidia-smi-ecc-preflight.csv"
  echo "nvtx_range=$RANGE_NAME@$NVTX_DOMAIN"
  echo "capture_step=3_of_4"
  echo "trace=cuda,nvtx,cudnn,cublas,osrt"
  echo "capture_range_end=stop-shutdown"
  echo "nsys_wait=primary"
  echo "request_cleanup_timeout=$REQUEST_CLEANUP_TIMEOUT"
  echo "parallelism=TP1_SP8_ring1_TE8_VAE_PP8"
  echo "attention=FASTVIDEO_VSA_H3_tile64"
  echo "vsa_provider=flashinfer_pr4944_sm120_cutedsl"
  echo "vsa_topk=162"
  echo "vsa_fused_tile_pack=1"
  echo "vsa_fused_untile=1"
  echo "vsa_direct_q2k=$VSA_DIRECT_Q2K"
  echo "vsa_o_bundle=$VSA_O_BUNDLE"
  echo "vsa_o_bundle_quality_contract=bf16_bit_exact_relative_to_same_fp8_qkv_e4m3_lane"
  echo "flashinfer_commit=$FLASHINFER_EXPECTED_COMMIT"
  echo "flashinfer_route=rdma"
  echo "flashinfer_max_bytes=$FLASHINFER_MAX_BYTES"
  echo "flashinfer_qk_producer_direct=1"
  echo "flashinfer_o_producer_direct=1"
  echo "flashinfer_release_after_denoise=1"
  echo "qkv_e4m3_transport=1"
  echo "qkv_projection_compute_dtype=float8_e4m3fn"
  echo "qkv_projection_output_dtype=bfloat16"
  echo "qkv_wire_dtype=float8_e4m3fn"
  echo "reverse_o_dtype=bfloat16"
  echo "qkv_transport_quality_contract=lossy_static_per_block_e4m3_transport"
  echo "qkv_e4m3_scale_sidecar=$QKV_E4M3_SCALE_SIDECAR"
  echo "qkv_e4m3_scale_sidecar_size=$QKV_E4M3_EXPECTED_SIZE"
  echo "qkv_e4m3_scale_sidecar_sha256=$QKV_E4M3_EXPECTED_WHOLE_SHA256"
  echo "qkv_e4m3_payload_sha256=$QKV_E4M3_EXPECTED_PAYLOAD_SHA256"
  echo "qkv_e4m3_source_model_revision=$QKV_E4M3_SOURCE_MODEL_REVISION"
  echo "qkv_runtime_checkpoint_revision=$QKV_RUNTIME_CHECKPOINT_REVISION"
  echo "qkv_runtime_checkpoint_manifest=$QKV_RUNTIME_MANIFEST"
  echo "qkv_runtime_checkpoint_manifest_sha256=$QKV_RUNTIME_MANIFEST_EXPECTED_SHA256"
  echo "qkv_e4m3_campaign_uuid=$QKV_E4M3_CAMPAIGN_UUID"
  echo "qkv_e4m3_run_contract_sha256=$QKV_E4M3_RUN_CONTRACT_SHA256"
  echo "qkv_e4m3_kernel_provider=$QKV_E4M3_KERNEL_PROVIDER"
  echo "qkv_e4m3_kernel_abi=$QKV_E4M3_KERNEL_ABI"
  echo "qkv_e4m3_binding=$QKV_E4M3_BINDING_JSON"
  echo "qkv_e4m3_audit=$QKV_E4M3_AUDIT_JSON"
  echo "qkv_remote_wire_saving_bytes_per_rank_request=44917555200"
  echo "minimax_h3_fp8_scope=$FP8_SCOPE"
  echo "minimax_h3_fp8_target_modules=200"
  echo "minimax_h3_fp8_config=$FP8_CONFIG_JSON"
  echo "minimax_h3_fp8_config_sha256=$FP8_CONFIG_EXPECTED_SHA256"
  echo "minimax_h3_fp8_contract=$FP8_CONTRACT_JSON"
  echo "minimax_h3_fp8_contract_sha256=$FP8_CONTRACT_EXPECTED_SHA256"
  echo "minimax_h3_fp8_dit_quant_mode=$FP8_DIT_QUANT_MODE_EXPECTED"
  echo "minimax_h3_fp8_quality_contract=lossy_online_per_tensor_fp8"
  echo "h3_qchunk_pipeline=0"
  echo "chunked_cpu_mp4_output=1"
  echo "chunked_cpu_mp4_slots=2"
  echo "vae_temporal_pre_gather_prune=0"
  echo "vae_exact_ops_sm120=$VAE_EXACT_OPS_SM120"
  echo "approximate_dit_cache=disabled"
  echo "upload=disabled"
} >"$OUTPUT_DIR/config.txt"

git -C "$VLLM_SOURCE" status --short --branch >"$OUTPUT_DIR/source-git-status.txt"
git -C "$VLLM_SOURCE" diff --binary >"$OUTPUT_DIR/source-git-diff.patch"
while IFS= read -r relative_path; do
  sha256sum "$VLLM_SOURCE/$relative_path"
done < <(git -C "$VLLM_SOURCE" ls-files --others --exclude-standard) \
  >"$OUTPUT_DIR/source-untracked-sha256.txt"
sha256sum \
  "${BASH_SOURCE[0]}" \
  "$ANALYZER" \
  "$GPU_HEALTH_GATE" \
  "$VLLM_SOURCE/vllm_omni/diffusion/attention/backends/fastvideo_vsa.py" \
  "$VLLM_SOURCE/vllm_omni/diffusion/attention/ops/minimax_h3_vsa_layout.py" \
  "$VLLM_SOURCE/vllm_omni/diffusion/attention/ops/minimax_h3_vsa_o_bundle.py" \
  "$VLLM_SOURCE/vllm_omni/diffusion/attention/ops/minimax_h3_vsa_owner_route.py" \
  "$VLLM_SOURCE/vllm_omni/diffusion/attention/parallel/ulysses.py" \
  "$VLLM_SOURCE/vllm_omni/diffusion/distributed/flashinfer_ulysses.py" \
  "$VLLM_SOURCE/vllm_omni/diffusion/models/minimax_h3/fp8_contract.py" \
  "$VLLM_SOURCE/vllm_omni/diffusion/models/minimax_h3/pipeline_minimax_h3.py" \
  "$VLLM_SOURCE/vllm_omni/diffusion/models/minimax_h3/qkv_transport_scales.py" \
  "$VLLM_SOURCE/vllm_omni/diffusion/models/minimax_h3/vae.py" \
  "$VLLM_SOURCE/vllm_omni/diffusion/models/minimax_h3/ops/vae/dispatch.py" \
  "$QKV_RUNTIME_MANIFEST" \
  "$QKV_E4M3_SCALE_SIDECAR" \
  >"$OUTPUT_DIR/source-sha256.txt"
printf '%q ' numactl --interleave=0,1 "$VLLM_BIN" "${SERVER_ARGS[@]}" \
  >"$OUTPUT_DIR/server-command.txt"
printf '\n' >>"$OUTPUT_DIR/server-command.txt"

PROFILE_PID=""
PROFILE_PGID=""
REQUEST_PID=""
REQUEST_PGID=""

# Every mutable process is launched through its own `setsid`.  Verify that the
# PID returned by Bash is also the new PGID/SID before group signaling is ever
# enabled; this prevents cleanup from reaching a caller-owned process group.
verify_owned_session() {
  local pid="$1"
  local label="$2"
  local observed_pgid=""
  local observed_sid=""
  # Bash can return $! just before the child reaches exec(setsid).  Retry this
  # narrow identity check rather than falsely rejecting that harmless race.
  for _ in {1..100}; do
    observed_pgid=""
    observed_sid=""
    read -r observed_pgid observed_sid < <(ps -o pgid=,sid= -p "$pid") || true
    if [[ "$observed_pgid" == "$pid" && "$observed_sid" == "$pid" ]]; then
      return 0
    fi
    kill -0 -- "$pid" 2>/dev/null || break
    sleep 0.02
  done
  echo "ERROR: $label PID $pid is not its own PGID/SID " \
    "(pgid=${observed_pgid:-missing}, sid=${observed_sid:-missing})" >&2
  return 1
}

owned_process_group_alive() {
  local pid="$1"
  local pgid="$2"
  kill -0 -- "$pid" 2>/dev/null || kill -0 -- "-$pgid" 2>/dev/null
}

stop_owned_process_group() {
  local label="$1"
  local pid="${2:-}"
  local pgid="${3:-}"
  local timeout_seconds="$4"
  local deadline

  [[ -n "$pid" || -n "$pgid" ]] || return 0
  if [[ ! "$pid" =~ ^[1-9][0-9]*$ || "$pid" == "$$" ]]; then
    echo "ERROR: refusing unsafe $label cleanup identity pid=$pid pgid=$pgid" >&2
    return 1
  fi
  # A signal can arrive in the few instructions between `$!` and successful
  # setsid verification.  In that window, target only the exact child PID and
  # never infer a group identity.
  if [[ -z "$pgid" ]]; then
    kill -0 -- "$pid" 2>/dev/null || return 0
    kill -TERM -- "$pid" 2>/dev/null || true
    deadline=$((SECONDS + timeout_seconds))
    while kill -0 -- "$pid" 2>/dev/null; do
      if (( SECONDS >= deadline )); then
        kill -KILL -- "$pid" 2>/dev/null || true
        for _ in {1..5}; do
          kill -0 -- "$pid" 2>/dev/null || return 0
          sleep 1
        done
        echo "ERROR: $label pid=$pid survived bounded PID cleanup" >&2
        return 1
      fi
      sleep 1
    done
    return 0
  fi
  if [[ ! "$pgid" =~ ^[1-9][0-9]*$ || "$pid" != "$pgid" ]]; then
    echo "ERROR: refusing unsafe $label cleanup identity pid=$pid pgid=$pgid" >&2
    return 1
  fi
  owned_process_group_alive "$pid" "$pgid" || return 0

  # Both targets are exact identities created above.  The PID signal is
  # redundant while the leader lives, but also covers a transient group-query
  # failure without broad name- or user-based matching.
  kill -TERM -- "-$pgid" 2>/dev/null || true
  kill -TERM -- "$pid" 2>/dev/null || true
  deadline=$((SECONDS + timeout_seconds))
  while owned_process_group_alive "$pid" "$pgid"; do
    if (( SECONDS >= deadline )); then
      kill -KILL -- "-$pgid" 2>/dev/null || true
      kill -KILL -- "$pid" 2>/dev/null || true
      for _ in {1..5}; do
        owned_process_group_alive "$pid" "$pgid" || return 0
        sleep 1
      done
      echo "ERROR: $label pid=$pid pgid=$pgid survived bounded cleanup" >&2
      return 1
    fi
    sleep 1
  done
}

trace_report_exported() {
  # Nsight prints the absolute report path only after the qdstrm conversion has
  # completed.  The output directory is required to be new, so this cannot
  # match a report or log from an earlier run.
  [[ -s "$PROFILE_REPORT" && -s "$SERVER_LOG" ]] &&
    grep -Fq "Generated:" "$SERVER_LOG" &&
    grep -Fq "$PROFILE_REPORT" "$SERVER_LOG"
}

curl_status=""
request_cleanup="not_started"

finalize_request_after_trace() {
  local request_pid="$REQUEST_PID"
  local request_pgid="$REQUEST_PGID"

  [[ -n "$request_pid" ]] || return 0
  request_cleanup=natural_before_trace_cleanup
  if owned_process_group_alive "$request_pid" "$request_pgid"; then
    request_cleanup=terminated_after_trace_export
    stop_owned_process_group \
      request "$request_pid" "$request_pgid" "$REQUEST_CLEANUP_TIMEOUT" || return 1
  fi
  curl_status=0
  if wait "$request_pid"; then
    :
  else
    curl_status=$?
  fi
  REQUEST_PID=""
  REQUEST_PGID=""
  {
    echo "request_cleanup=$request_cleanup"
    echo "curl_status=$curl_status"
  } >>"$OUTPUT_DIR/config.txt"
}

cleanup() {
  local status=$?
  trap - EXIT INT TERM
  stop_owned_process_group \
    request "$REQUEST_PID" "$REQUEST_PGID" "$REQUEST_CLEANUP_TIMEOUT" || true
  stop_owned_process_group profile "$PROFILE_PID" "$PROFILE_PGID" 30 || true
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
  numactl --interleave=0,1 "$VLLM_BIN" "${SERVER_ARGS[@]}" \
  >"$SERVER_LOG" 2>&1 &
PROFILE_PID=$!
if ! verify_owned_session "$PROFILE_PID" profile; then
  kill -TERM -- "$PROFILE_PID" 2>/dev/null || true
  wait "$PROFILE_PID" 2>/dev/null || true
  PROFILE_PID=""
  die "Nsight launcher did not establish a runner-owned session"
fi
PROFILE_PGID=$PROFILE_PID
echo "$PROFILE_PID" >"$OUTPUT_DIR/nsys.pid"
echo "$PROFILE_PGID" >"$OUTPUT_DIR/nsys.pgid"

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

echo "Server ready; issuing the exact 15-second FastH3 four-forward request"
setsid curl --fail-with-body --silent --show-error --max-time "$REQUEST_TIMEOUT" \
  -X POST "http://127.0.0.1:$PORT/v1/videos/sync" \
  -F "prompt=$PROMPT" \
  -F 'width=1280' \
  -F 'height=720' \
  -F 'fps=24' \
  -F 'num_inference_steps=4' \
  -F 'flow_shift=12' \
  -F 'seed=1101' \
  -F 'aspect_ratio=16:9' \
  -F 'quality=lossless' \
  -F 'extra_params={"task":"t2va","duration":15,"audio_flow_shift":3.0}' \
  -o "$OUTPUT_DIR/expected-incomplete-response.mp4" \
  2>"$REQUEST_LOG" &
REQUEST_PID=$!
if ! verify_owned_session "$REQUEST_PID" request; then
  kill -TERM -- "$REQUEST_PID" 2>/dev/null || true
  wait "$REQUEST_PID" 2>/dev/null || true
  REQUEST_PID=""
  die "curl launcher did not establish a runner-owned session"
fi
REQUEST_PGID=$REQUEST_PID
echo "$REQUEST_PID" >"$OUTPUT_DIR/request.pid"
echo "$REQUEST_PGID" >"$OUTPUT_DIR/request.pgid"

exit_deadline=$((SECONDS + NSYS_EXIT_TIMEOUT))
trace_export_observed=0
while kill -0 "$PROFILE_PID" 2>/dev/null; do
  if [[ "$trace_export_observed" -eq 0 ]] && trace_report_exported; then
    trace_export_observed=1
    echo "trace_export_complete=1" >>"$OUTPUT_DIR/config.txt"
    echo "Nsight report exported; releasing the runner-owned HTTP request"
    finalize_request_after_trace || die \
      "Unable to clean the runner-owned curl process group after trace export"
  fi
  if (( SECONDS >= exit_deadline )); then
    tail -n 240 "$SERVER_LOG" >&2 || true
    if stop_owned_process_group profile "$PROFILE_PID" "$PROFILE_PGID" 30; then
      wait "$PROFILE_PID" 2>/dev/null || true
      PROFILE_PID=""
      PROFILE_PGID=""
    fi
    die "Nsight did not exit within ${NSYS_EXIT_TIMEOUT}s after the profiled request"
  fi
  sleep 1
done

nsys_status=0
if wait "$PROFILE_PID"; then
  :
else
  nsys_status=$?
fi
echo "nsys_status=$nsys_status" >>"$OUTPUT_DIR/config.txt"
[[ "$nsys_status" -eq 0 ]] || die "Nsight Systems exited with status $nsys_status"

[[ -s "$PROFILE_REPORT" ]] || {
  tail -n 240 "$SERVER_LOG" >&2 || true
  die "Nsight report was not created: $PROFILE_REPORT"
}
if [[ "$trace_export_observed" -eq 0 ]]; then
  trace_report_exported || die \
    "Nsight report exists but its completed-export marker is missing"
  trace_export_observed=1
  echo "trace_export_complete=1" >>"$OUTPUT_DIR/config.txt"
fi

# `wait` above returns only after Nsight has finalized PROFILE_REPORT and its
# primary target has stopped.  Any re-parented workers are still confined to
# the exact profiler session and are cleaned here.  The request was normally
# released as soon as the completed-export marker appeared; the call below is
# the race-safe fallback for an unusually fast profiler exit.
stop_owned_process_group profile "$PROFILE_PID" "$PROFILE_PGID" 30 || die \
  "Unable to clean the runner-owned Nsight process group"
PROFILE_PID=""
PROFILE_PGID=""

finalize_request_after_trace || die \
  "Unable to clean the runner-owned curl process group after trace export"
[[ -n "$curl_status" ]] || die "curl request exited without a recorded status"
[[ "$curl_status" -ne 0 ]] || die \
  "Request unexpectedly completed; step-3 stop-shutdown did not interrupt the four-step request"

SERVER_LOG_PATH="$SERVER_LOG" VALIDATION_PATH="$VALIDATION_LOG" \
EXPECTED_QKV_PAYLOAD_SHA256="$QKV_E4M3_EXPECTED_PAYLOAD_SHA256" \
EXPECTED_QKV_KERNEL_PROVIDER="$QKV_E4M3_KERNEL_PROVIDER" \
EXPECTED_QKV_KERNEL_ABI="$QKV_E4M3_KERNEL_ABI" \
EXPECTED_FP8_DIT_QUANT_MODE="$FP8_DIT_QUANT_MODE_EXPECTED" \
EXPECTED_FP8_CONFIG_SHA256="$FP8_CONFIG_EXPECTED_SHA256" \
EXPECTED_FP8_CONTRACT_SHA256="$FP8_CONTRACT_EXPECTED_SHA256" \
"$PYTHON_BIN" - <<'PY'
import os
import re
from collections import Counter
from pathlib import Path

server_log = Path(os.environ["SERVER_LOG_PATH"])
validation_path = Path(os.environ["VALIDATION_PATH"])
text = server_log.read_text(encoding="utf-8", errors="replace")
lines = text.splitlines()
worker_re = re.compile(r"DiffusionWorker_SP([0-7])\s+pid=[0-9]+")


def worker_rank(line: str, offset: int) -> int | None:
    matches = worker_re.findall(line[:offset])
    return int(matches[-1]) if matches else None


armed_marker = "FlashInfer PCIe Ulysses armed:"
bf16_qk_marker = "FlashInfer PCIe Ulysses Q/K producer-direct exchange active:"
qkv_e4m3_marker = "FlashInfer PCIe Ulysses Q/K/V producer-direct exchange active:"
qkv_e4m3_install_marker = "MiniMax H3 E4M3 QKV transport scales installed: "
qkv_e4m3_receive_marker = (
    "FASTVIDEO_VSA H3 FP8 QKV transport receive path: fused "
    "E4M3->BF16 dequant + tile64 pack enabled "
)
gate_marker = "FlashInfer PCIe Ulysses VSA gate exchange active:"
o_marker = "FlashInfer PCIe Ulysses O producer-direct exchange active:"
o_bundle_enabled = os.environ.get("VLLM_OMNI_FASTVIDEO_VSA_O_BUNDLE") == "1"
expected_capacity = 175056896 if o_bundle_enabled else 171114496
expected_o_rows = 97688 if o_bundle_enabled else 95488
armed_re = re.compile(
    r"group=([^\s]+).*?world=8.*?device=cuda:([0-7]).*?"
    rf"transport=rdma.*?capacity={expected_capacity}"
)
bf16_qkg_re = re.compile(
    r"rank=([0-7])\s+group=([^\s]+).*?"
    r"shape=\(1,\s*11936,\s*56,\s*128\).*?dtype=torch\.bfloat16"
)
qkv_e4m3_re = re.compile(
    r"rank=([0-7])\s+group=([^\s]+).*?"
    r"shape=\(1,\s*11936,\s*56,\s*128\).*?dtype=torch\.float8_e4m3fn"
)
qkv_e4m3_install_re = re.compile(
    r"blocks=50\s+provider=([^\s]+)\s+abi=([^\s]+)\s+"
    r"payload_sha256=([0-9a-f]{64})\s+"
    r"scale_min=([^\s]+)\s+scale_max=([^\s]+)"
)
qkv_e4m3_receive_re = re.compile(
    r"compact_rows=95487,\s*padded_rows=105024"
)
o_re = re.compile(
    r"rank=([0-7])\s+group=([^\s]+).*?"
    rf"shape=\(1,\s*{expected_o_rows},\s*7,\s*128\).*?dtype=torch\.bfloat16"
)
o_bundle_producer_marker = "FASTVIDEO_VSA H3 reverse-O bundle producer active:"
o_bundle_producer_re = re.compile(
    r"fine=\(1,\s*95488,\s*7,\s*128\),\s+"
    r"compact_coarse=\(1,\s*1641,\s*7,\s*128\),\s+"
    r"bundle=\(1,\s*97688,\s*7,\s*128\),\s+"
    r"kmax=275,\s+destination=registered_reverse_o"
)
o_bundle_consumer_marker = "MiniMax H3 VSA reverse-O bundle SP active:"
o_bundle_consumer_re = re.compile(
    r"rank=([0-7])\s+"
    r"local_gate=\(1,\s*11936,\s*56,\s*128\)\s+"
    r"reversed_bundle=\(1,\s*12211,\s*56,\s*128\)\s+"
    r"kmax=275\s+elided_gate_operand_bytes=171114496\s+"
    r"elided_gate_remote_wire_bytes=149725184\s+"
    r"piggyback_tail_operand_bytes=3942400\s+"
    r"piggyback_tail_remote_wire_bytes=3449600"
)
deferred_producer_marker = "FASTVIDEO_VSA H3 deferred gate producer active:"
deferred_consumer_marker = "MiniMax H3 VSA deferred-gate SP active:"

armed = []
bf16_qk = []
qkv_e4m3 = []
qkv_e4m3_installs = []
qkv_e4m3_receives = []
gate = []
o_direct = []
o_bundle_producer = []
o_bundle_consumer = []
malformed = []
for line_number, line in enumerate(lines, 1):
    for kind, marker, pattern, destination in (
        ("armed", armed_marker, armed_re, armed),
        ("bf16-qk", bf16_qk_marker, bf16_qkg_re, bf16_qk),
        ("qkv-e4m3", qkv_e4m3_marker, qkv_e4m3_re, qkv_e4m3),
        ("gate", gate_marker, bf16_qkg_re, gate),
        ("o", o_marker, o_re, o_direct),
    ):
        offset = line.find(marker)
        if offset < 0:
            continue
        owner = worker_rank(line, offset)
        detail = pattern.search(line[offset:])
        if owner is None or detail is None:
            malformed.append((line_number, kind, line))
            continue
        if kind == "armed":
            group, device = detail.groups()
            destination.append((owner, group, int(device)))
        else:
            rank, group = int(detail.group(1)), detail.group(2)
            if owner != rank:
                malformed.append((line_number, f"{kind}-rank-mismatch", line))
            else:
                destination.append((rank, group))

    offset = line.find(qkv_e4m3_install_marker)
    if offset >= 0:
        owner = worker_rank(line, offset)
        detail = qkv_e4m3_install_re.search(line[offset:])
        if owner is None or detail is None:
            malformed.append((line_number, "qkv-e4m3-install", line))
        else:
            provider, abi, payload_sha256, scale_min, scale_max = detail.groups()
            try:
                scale_min_value = float(scale_min)
                scale_max_value = float(scale_max)
            except ValueError:
                malformed.append((line_number, "qkv-e4m3-install-scale", line))
            else:
                qkv_e4m3_installs.append(
                    (
                        owner,
                        provider,
                        abi,
                        payload_sha256,
                        scale_min_value,
                        scale_max_value,
                    )
                )

    offset = line.find(qkv_e4m3_receive_marker)
    if offset >= 0:
        owner = worker_rank(line, offset)
        if owner is None or qkv_e4m3_receive_re.search(line[offset:]) is None:
            malformed.append((line_number, "qkv-e4m3-receive", line))
        else:
            qkv_e4m3_receives.append(owner)

    offset = line.find(o_bundle_producer_marker)
    if offset >= 0:
        owner = worker_rank(line, offset)
        if owner is None or o_bundle_producer_re.search(line[offset:]) is None:
            malformed.append((line_number, "o-bundle-producer", line))
        else:
            o_bundle_producer.append(owner)

    offset = line.find(o_bundle_consumer_marker)
    if offset >= 0:
        owner = worker_rank(line, offset)
        detail = o_bundle_consumer_re.search(line[offset:])
        if owner is None or detail is None:
            malformed.append((line_number, "o-bundle-consumer", line))
        else:
            rank = int(detail.group(1))
            if owner != rank:
                malformed.append((line_number, "o-bundle-consumer-rank-mismatch", line))
            else:
                o_bundle_consumer.append(rank)

problems = []
armed_set = set(armed)
groups = {item[1] for item in armed}
if len(groups) != 1:
    problems.append(f"communicator markers do not share one group: {sorted(groups)!r}")
group = next(iter(groups), "")
expected_armed = {(rank, group, rank) for rank in range(8)}
expected_exchange = {(rank, group) for rank in range(8)}
if len(armed) != 8 or armed_set != expected_armed:
    problems.append(f"all-RDMA arm tuples mismatch: observed={sorted(armed_set)!r}")
for name, values in (("E4M3 Q/K/V", qkv_e4m3), ("O", o_direct)):
    if len(values) != 8 or set(values) != expected_exchange:
        problems.append(f"{name} producer/exchange tuples mismatch: observed={sorted(set(values))!r}")
if bf16_qk:
    problems.append(
        "BF16 Q/K producer-direct markers appeared in the required E4M3 lane: "
        f"observed={sorted(set(bf16_qk))!r}"
    )
expected_rank_counts = Counter({rank: 1 for rank in range(8)})
qkv_install_rank_counts = Counter(item[0] for item in qkv_e4m3_installs)
if qkv_install_rank_counts != expected_rank_counts:
    problems.append(
        "E4M3 QKV sidecar install ownership mismatch: "
        f"{qkv_install_rank_counts!r}"
    )
for rank, provider, abi, payload_sha256, scale_min, scale_max in qkv_e4m3_installs:
    if provider != os.environ["EXPECTED_QKV_KERNEL_PROVIDER"]:
        problems.append(f"rank {rank} installed unexpected QKV provider {provider!r}")
    if abi != os.environ["EXPECTED_QKV_KERNEL_ABI"]:
        problems.append(f"rank {rank} installed unexpected QKV ABI {abi!r}")
    if payload_sha256 != os.environ["EXPECTED_QKV_PAYLOAD_SHA256"]:
        problems.append(f"rank {rank} installed unexpected QKV payload {payload_sha256!r}")
    if not (0.0 < scale_min <= scale_max < float("inf")):
        problems.append(
            f"rank {rank} installed invalid QKV scale range [{scale_min!r}, {scale_max!r}]"
        )
qkv_receive_rank_counts = Counter(qkv_e4m3_receives)
if qkv_receive_rank_counts != expected_rank_counts:
    problems.append(
        "E4M3 QKV fused receive/dequant ownership mismatch: "
        f"{qkv_receive_rank_counts!r}"
    )
full_gate_marker_count = text.count(gate_marker)
deferred_producer_count = text.count(deferred_producer_marker)
deferred_consumer_count = text.count(deferred_consumer_marker)
if o_bundle_enabled:
    if gate or full_gate_marker_count:
        problems.append(
            "Stage-2 must elide the full VSA gate exchange: "
            f"parsed={len(gate)}, markers={full_gate_marker_count}"
        )
    if Counter(o_bundle_producer) != expected_rank_counts:
        problems.append(
            "reverse-O bundle producer ownership mismatch: "
            f"{Counter(o_bundle_producer)!r}"
        )
    if Counter(o_bundle_consumer) != expected_rank_counts:
        problems.append(
            "reverse-O bundle consumer ownership mismatch: "
            f"{Counter(o_bundle_consumer)!r}"
        )
    if deferred_producer_count or deferred_consumer_count:
        problems.append(
            "Stage-2 must not run the Stage-1 compact-coarse gather route: "
            f"producer={deferred_producer_count}, consumer={deferred_consumer_count}"
        )
else:
    if len(gate) != 8 or set(gate) != expected_exchange:
        problems.append(
            "VSA gate producer/exchange tuples mismatch: "
            f"observed={sorted(set(gate))!r}"
        )
    if o_bundle_producer or o_bundle_consumer:
        problems.append(
            "reverse-O bundle markers appeared while Stage-2 was disabled: "
            f"producer={Counter(o_bundle_producer)!r}, "
            f"consumer={Counter(o_bundle_consumer)!r}"
        )
    if deferred_producer_count or deferred_consumer_count:
        problems.append(
            "Stage-1 deferred-gate markers appeared outside this runner contract: "
            f"producer={deferred_producer_count}, consumer={deferred_consumer_count}"
        )

fp8_audit_marker = "MINIMAX_H3_MLP_OUT_QKV_FP8_AUDIT_V1"
all_fp8_audit_markers = (
    "MINIMAX_H3_MLP_FP8_AUDIT_V1",
    "MINIMAX_H3_MLP_OUT_FP8_AUDIT_V1",
    fp8_audit_marker,
)
fp8_audit_lines = [
    line for line in lines if any(marker in line for marker in all_fp8_audit_markers)
]
fp8_expected = {
    "world_size": "8",
    "scope": "mlp-out-qkv",
    "dit_quant_mode": os.environ["EXPECTED_FP8_DIT_QUANT_MODE"],
    "target_modules": "200",
    "fc1": "50",
    "fc2": "50",
    "out_proj": "50",
    "qkv_proj": "50",
    "qkv_proj_output_dtype": "bfloat16",
    "to_gate_compress_bf16": "50",
    "sequence_parallel_size": "8",
    "ulysses_degree": "8",
    "ring_degree": "1",
    "allgather_degree": "1",
    "quant_method": "Fp8PerTensorOnlineLinearMethod",
    "linear_kernel": "CutlassFP8ScaledMMLinearKernel",
    "weight_dtype": "float8_e4m3fn",
    "weight_quant": "static_per_tensor_symmetric",
    "activation_quant": "dynamic_per_token_symmetric",
    "input_dtype": "bfloat16",
    "output_dtype": "bfloat16",
    "source_weight_dtype": "bfloat16",
    "weight_scale_dtype": "float32",
    "tp_size": "1",
    "gpu_architecture": "sm120",
    "unexpected_fp8_modules": "0",
    "fallback_count": "0",
    "config_sha256": os.environ["EXPECTED_FP8_CONFIG_SHA256"],
    "contract_sha256": os.environ["EXPECTED_FP8_CONTRACT_SHA256"],
    "status": "ok",
}
fp8_expected_keys = {"rank", *fp8_expected}
fp8_audit_ranks = []
for index, line in enumerate(fp8_audit_lines):
    context = f"fp8-audit[{index}]"
    if fp8_audit_marker not in line:
        problems.append(f"{context}: wrong-scope selective-FP8 marker")
        continue
    offset = line.find(fp8_audit_marker)
    owner = worker_rank(line, offset)
    fields = {}
    for token in line.split(fp8_audit_marker, 1)[1].strip().split():
        if "=" not in token:
            problems.append(f"{context}: malformed token {token!r}")
            continue
        key, value = token.split("=", 1)
        if key in fields:
            problems.append(f"{context}: duplicate field {key!r}")
        fields[key] = value.rstrip(",")
    if set(fields) != fp8_expected_keys:
        problems.append(
            f"{context}: fields={sorted(fields)!r}, expected={sorted(fp8_expected_keys)!r}"
        )
    try:
        rank = int(fields.get("rank", ""))
    except ValueError:
        rank = -1
    fp8_audit_ranks.append(rank)
    if owner != rank:
        problems.append(f"{context}: worker_rank={owner!r}, rank={rank}")
    for name, expected in fp8_expected.items():
        if fields.get(name) != expected:
            problems.append(
                f"{context}: {name}={fields.get(name)!r}, expected={expected!r}"
            )
if Counter(fp8_audit_ranks) != expected_rank_counts:
    problems.append(
        "exact-200 mlp-out-qkv FP8 audit ownership mismatch: "
        f"{Counter(fp8_audit_ranks)!r}"
    )
if malformed:
    problems.append(f"malformed markers at lines={[item[0] for item in malformed]!r}")

rank_markers = {
    "fast_h3": "FastH3 adapter active: sigma points [0.999, 0.749, 0.5, 0.25, 0.0] for 4 transformer forwards",
    "chunked_output": (
        "MiniMax H3 output route enabled: "
        "route=finalized_chunk_gpu_u8_pinned_d2h_pyav_libx264 "
        "gate=VLLM_OMNI_MINIMAX_H3_CHUNKED_CPU_MP4_OUTPUT "
        "worker_transport=final_mp4_bytes"
    ),
    "tile_pack": "FASTVIDEO_VSA H3 tile pack: fused Triton compact->tile64 enabled",
}
untile_to_o_marker = (
    "FASTVIDEO_VSA H3 output untile: fused Triton tile64->aligned enabled "
    "(valid_rows=95487, aligned_rows=95488, tiled_rows=105024, "
    "destination=registered_reverse_o)"
)
if o_bundle_enabled:
    if untile_to_o_marker in text:
        problems.append("legacy standalone untile-to-O marker appeared in Stage-2")
else:
    rank_markers["untile_to_o"] = untile_to_o_marker
if os.environ.get("VLLM_OMNI_FASTVIDEO_VSA_DIRECT_Q2K") == "1":
    rank_markers["direct_q2k"] = (
        "FASTVIDEO_VSA H3 metadata: direct ordered q2k enabled by "
        "VLLM_OMNI_FASTVIDEO_VSA_DIRECT_Q2K=1"
    )
    forbidden_metadata_marker = "FASTVIDEO_VSA H3 metadata: legacy bool-map conversion active"
else:
    rank_markers["legacy_q2k"] = "FASTVIDEO_VSA H3 metadata: legacy bool-map conversion active"
    forbidden_metadata_marker = "FASTVIDEO_VSA H3 metadata: direct ordered q2k enabled"
rank_marker_counts = {}
for name, marker in rank_markers.items():
    ranks = []
    for line in lines:
        offset = line.find(marker)
        if offset >= 0:
            rank = worker_rank(line, offset)
            if rank is not None:
                ranks.append(rank)
    rank_marker_counts[name] = Counter(ranks)
    if Counter(ranks) != Counter({rank: 1 for rank in range(8)}):
        problems.append(f"{name} marker ownership mismatch: {Counter(ranks)!r}")
if forbidden_metadata_marker in text:
    problems.append(f"forbidden VSA metadata route marker observed: {forbidden_metadata_marker!r}")

vae_exact_marker = "MiniMax H3 VAE exact operators installed: "
vae_exact_re = re.compile(
    r"rank=([0-7])\s+device=cuda:([0-7])\s+blocks=36\s+"
    r"linear_dtype=float16\s+counters=enabled"
)
vae_exact_installs = []
for line_number, line in enumerate(lines, 1):
    offset = line.find(vae_exact_marker)
    if offset >= 0:
        owner = worker_rank(line, offset)
        detail = vae_exact_re.search(line[offset:])
        if owner is None or detail is None:
            problems.append(f"malformed exact-VAE install marker at line {line_number}")
            continue
        rank, device = map(int, detail.groups())
        if owner != rank or rank != device:
            problems.append(
                "exact-VAE install worker/rank/device mismatch at line "
                f"{line_number}: owner={owner}, rank={rank}, device={device}"
            )
            continue
        vae_exact_installs.append(rank)
if os.environ.get("VLLM_OMNI_MINIMAX_H3_VAE_EXACT_OPS_SM120") == "1":
    if Counter(vae_exact_installs) != expected_rank_counts:
        problems.append(
            "SM120 VAE exact-op install marker mismatch: "
            f"rank_counts={Counter(vae_exact_installs)!r}"
        )
elif vae_exact_installs:
    problems.append(
        "SM120 VAE exact-op marker appeared while gate was disabled: "
        f"ranks={vae_exact_installs!r}"
    )

routing_re = re.compile(
    r"FASTVIDEO_VSA H3 routing: seq_len=95487, .*?"
    r"video_shape=\(107, 22, 40\), .*?topk=162, .*?compute_backend=flashinfer"
)
if len(routing_re.findall(text)) != 1:
    problems.append("expected exactly one current FastH3 VSA routing marker")
compute_marker = "FASTVIDEO_VSA H3 compute kernel: FlashInfer vsa_sm120_blk64_cute_dsl"
if text.count(compute_marker) != 1:
    problems.append("expected exactly one FlashInfer #4944 SM120 CuTeDSL marker")
nvtx_marker = "[MiniMaxH3NVTX] capturing step=3/4 range=minimax_h3.denoise.step_03"
if text.count(nvtx_marker) != 1:
    problems.append("expected exactly one step-3-of-4 NVTX ownership marker")

failure_patterns = (
    r"CUDA out of memory",
    r"torch\.OutOfMemoryError",
    r"FASTVIDEO_VSA falling back to SDPA",
    r"VSA-H3 kernel failed",
    r"using the reference tile pack",
    r"FASTVIDEO_VSA H3 output untile fallback:",
    r"FlashInfer PCIe Ulysses O producer-direct exchange fell back:",
    r"(?:MiniMax-H3|H3 VSA) reverse-O bundl(?:e|ing).*"
    r"(?:requires|absent|stale|untrusted|mismatch|unsupported|disappeared|wrong|failed|fallback|fell back|lost|did not)",
    r"MiniMax-H3 VSA backend did not retain its registered reverse-O landing",
    r"FlashInfer PCIe Ulysses unavailable; using NCCL",
    r"transport=p2p",
    r"PCIe Ulysses communicator entered BROKEN state",
    r"E4M3 QKV.*(?:missing|mismatch|unsupported|wrong dtype|fallback|fell back)",
    r"QKV.*sidecars bind different",
    r"QKV scale.*(?:does not match|require)",
    r"QKV transport.*(?:requires|unsupported|wrong dtype|fallback|fell back)",
    r"QKV transport scales have already been installed",
    r"MiniMax-H3 selective FP8.*(?:missing|mismatch|unsupported|fallback|failed)",
)
failure_hits = [pattern for pattern in failure_patterns if re.search(pattern, text, re.IGNORECASE)]
if failure_hits:
    problems.append(f"fallback/failure evidence matched: {failure_hits!r}")

required_stack = (
    "FastH3_VSA_DataFree_P4e_direct_q2k_P4f_vae_exact_ops_sm120_"
    "Stage2_reverse_o_bundle_MLP_OUT_QKV_exact200_FP8_QKV_E4M3"
)
report = [
    f"required_stack={required_stack}",
    "required_nvtx_step=3/4",
    "vae_exact_ops_sm120="
    + os.environ.get("VLLM_OMNI_MINIMAX_H3_VAE_EXACT_OPS_SM120", "missing"),
    f"vsa_o_bundle={int(o_bundle_enabled)}",
    f"expected_flashinfer_capacity_bytes={expected_capacity}",
    f"expected_o_producer_rows={expected_o_rows}",
    f"vae_exact_ops_install_rank_counts={dict(sorted(Counter(vae_exact_installs).items()))}",
    f"armed_marker_count={len(armed)}",
    f"forbidden_bf16_qk_direct_marker_count={len(bf16_qk)}",
    f"qkv_e4m3_direct_marker_count={len(qkv_e4m3)}",
    f"qkv_e4m3_sidecar_install_rank_counts={dict(sorted(qkv_install_rank_counts.items()))}",
    f"qkv_e4m3_receive_rank_counts={dict(sorted(qkv_receive_rank_counts.items()))}",
    f"fp8_exact200_audit_rank_counts={dict(sorted(Counter(fp8_audit_ranks).items()))}",
    f"vsa_gate_marker_count={len(gate)}",
    f"o_direct_marker_count={len(o_direct)}",
    f"o_bundle_producer_rank_counts={dict(sorted(Counter(o_bundle_producer).items()))}",
    f"o_bundle_consumer_rank_counts={dict(sorted(Counter(o_bundle_consumer).items()))}",
    f"forbidden_deferred_producer_marker_count={deferred_producer_count}",
    f"forbidden_deferred_consumer_marker_count={deferred_consumer_count}",
    f"communicator_group={group}",
]
for name, counts in rank_marker_counts.items():
    report.append(f"{name}_rank_counts={dict(sorted(counts.items()))}")
for line_number, kind, line in malformed:
    report.append(f"malformed_marker={line_number}:{kind}:{line}")
for problem in problems:
    report.append(f"validation_error={problem}")
report.append(f"validation_status={'failed' if problems else 'ok'}")
validation_path.write_text("\n".join(report) + "\n", encoding="utf-8")
if problems:
    raise SystemExit("; ".join(problems))
PY

"$NSYS_BIN" stats \
  --report nvtx_pushpop_sum,nvtx_gpu_proj_sum,cuda_gpu_kern_sum,cuda_gpu_mem_time_sum,cuda_api_sum \
  "$PROFILE_REPORT" >"$STATS_LOG"
grep -Fq "$RANGE_NAME" "$STATS_LOG" || die \
  "Nsight report does not contain the requested NVTX range: $RANGE_NAME"

# Kernel names alone cannot reveal that Stage-2 bundled/elided the standalone
# gate exchange.  Pin the required O-bundle layout instead of letting the
# generic analyzer infer the legacy ten-barrier VSA layout.
"$PYTHON_BIN" "$ANALYZER" "$PROFILE_REPORT" \
  --nvtx-range "$RANGE_NAME" \
  --nvtx-domain "$NVTX_DOMAIN" \
  --server-log "$SERVER_LOG" \
  --barriers-per-layer 8 \
  --top 50 \
  --output "$ANALYSIS_MD" \
  --json-output "$ANALYSIS_JSON" \
  --raw-nsys-stats "$ANALYSIS_STATS"

ANALYSIS_JSON_PATH="$ANALYSIS_JSON" "$PYTHON_BIN" - <<'PY'
import json
import os

result = json.load(open(os.environ["ANALYSIS_JSON_PATH"], encoding="utf-8"))
scope = result["scope"]
if scope["name"] != "minimax_h3.denoise.step_03":
    raise SystemExit(f"analyzer selected the wrong range: {scope!r}")
if scope["domain"] != "vllm_omni.minimax_h3":
    raise SystemExit(f"analyzer selected the wrong NVTX domain: {scope!r}")
categories = {item["category"]: item for item in result["kernel_categories"]}
if categories["vsa_attention"]["count"] <= 0:
    raise SystemExit("analyzer did not identify the FlashInfer #4944 VSA kernel")
barriers = result["flashinfer_ulysses_barriers"]
if not barriers["available"]:
    raise SystemExit("analyzer did not identify FlashInfer #4876 Ulysses barriers")
o_bundle_enabled = os.environ.get("VLLM_OMNI_FASTVIDEO_VSA_O_BUNDLE") == "1"
expected_barriers_per_layer = 8 if o_bundle_enabled else 10
if barriers["barriers_per_layer"] != expected_barriers_per_layer:
    raise SystemExit(
        f"expected {expected_barriers_per_layer} VSA barriers/layer, got {barriers!r}"
    )
if barriers["devices"] != list(range(8)):
    raise SystemExit(f"expected RDMA barrier evidence on devices 0..7, got {barriers!r}")
if barriers["observed_layer_cycles"] != 50:
    raise SystemExit(f"expected 50 complete H3 layer cycles, got {barriers!r}")
PY

echo "Capture complete : $PROFILE_REPORT"
echo "Stack validation: $VALIDATION_LOG"
echo "Analysis         : $ANALYSIS_MD"
echo "Machine data     : $ANALYSIS_JSON"
echo "The curl failure/incomplete MP4 is expected because stop-shutdown ends the server after step 3."
