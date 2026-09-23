#!/usr/bin/env bash
set -euo pipefail

# Generate two prompt-generalization A/B pairs without a warmup request.
#
# The quality control and candidate intentionally share FastH3 4-step,
# FlashInfer's PR #4944 VSA kernel, top-k 162, TP1/SP8, NCCL Ulysses, seed,
# prompt, canvas, and output codec.  Only the two admitted lossy factors differ:
#
#   control:   BF16 main DiT + BF16 QKV wire + full H3 VAE
#   candidate: all-main FP8 + BF16 QKV wire + TAEH3 FP16
#
# E4M3 QKV transport is explicitly disabled.  RDMA/O-bundle are omitted from
# this shape-generalization test because their current runner qualification is
# pinned to the 1280x720 / 362-frame production geometry; NCCL is held fixed on
# both sides and therefore cannot confound the visual A/B.

WORKSPACE=/lustre/raplab/client/sylarl/minimax-h3-native
VLLM_SOURCE="$WORKSPACE/vllm-omni-h3-dlo-latest/vllm-omni"
VENV="$WORKSPACE/vllm-omni-h3-dlo-latest/.venv-vllm028"
PYTHON="$VENV/bin/python"
VLLM="$VENV/bin/vllm"
MODEL="$WORKSPACE/MiniMax-H3/FL2VA"
FASTVIDEO_KERNEL_PYTHON="$WORKSPACE/FastVideo/fastvideo-kernel/python"
FLASHINFER_ROOT="$WORKSPACE/fi-pr4944-pr4876-combined"
FLASHINFER_COMMIT=ecd450bd9002e60081fa3c1eac0f16b05d06108c
FLASHINFER_WORKSPACE="$WORKSPACE/.flashinfer-pr4944-pr4876-vllm028-cache"
FASTH3_LORA="$WORKSPACE/FastH3-LoRA/vsa-datafree/adapter_model.safetensors"
ADALN_CACHE="$WORKSPACE/cache/minimax-h3-fasth3-vsa-datafree-42dc502a-t2va-4step-shift12-3.safetensors"
TAEH3_CHECKPOINT="$WORKSPACE/cache/taeh3-62f7591f59dfbb4c3c02b7a621d180a9eeaba26c.safetensors"
FFPROBE="$WORKSPACE/ffmpeg-tools/bin/ffprobe"
PROMPT_ROOT="$VLLM_SOURCE/tools/minimax_h3/prompts"
PROMPT_TABLETOP="$PROMPT_ROOT/tabletop_emblem_1x1_6s.txt"
PROMPT_PANCAKE="$PROMPT_ROOT/pancake_robot_9x16_12s.txt"
OUTPUT_ROOT="${1:-$WORKSPACE/results/h3-two-prompt-quality-ab-$(date -u +%Y%m%dT%H%M%SZ)}"
PHASE="${2:-both}"
PORT="${VLLM_OMNI_PORT:-8091}"
GPU_ORDER="${VLLM_OMNI_GPU_ORDER:-0,4,1,5,2,6,3,7}"
SERVER_PID=""

die() {
  echo "ERROR: $*" >&2
  exit 1
}

case "$PHASE" in
  both|control|candidate) ;;
  *) die "phase must be one of: both, control, candidate (got $PHASE)" ;;
esac

stop_server() {
  if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill -TERM -- "-$SERVER_PID" 2>/dev/null || kill -TERM "$SERVER_PID" 2>/dev/null || true
    for _ in $(seq 1 30); do
      kill -0 "$SERVER_PID" 2>/dev/null || break
      sleep 1
    done
    if kill -0 "$SERVER_PID" 2>/dev/null; then
      kill -KILL -- "-$SERVER_PID" 2>/dev/null || kill -KILL "$SERVER_PID" 2>/dev/null || true
    fi
    wait "$SERVER_PID" 2>/dev/null || true
  fi
  SERVER_PID=""
}

trap stop_server EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

require_idle_gpus() {
  local applications
  applications="$(nvidia-smi \
    --query-compute-apps=pid,gpu_bus_id,used_memory,process_name \
    --format=csv,noheader,nounits)" || die "nvidia-smi compute-process query failed"
  if [[ -n "${applications//[[:space:]]/}" ]]; then
    echo "GPU compute processes are already active:" >&2
    printf '%s\n' "$applications" >&2
    die "all eight GPUs must be idle before starting an A/B server"
  fi
}

for required in \
  "$PYTHON" \
  "$VLLM" \
  "$MODEL/model_index.json" \
  "$FASTVIDEO_KERNEL_PYTHON/fastvideo_kernel/__init__.py" \
  "$FLASHINFER_ROOT/flashinfer/cute_dsl/sparse/bsa_attn_sm120.py" \
  "$FASTH3_LORA" \
  "$ADALN_CACHE" \
  "$TAEH3_CHECKPOINT" \
  "$FFPROBE" \
  "$PROMPT_TABLETOP" \
  "$PROMPT_PANCAKE"; do
  [[ -e "$required" ]] || die "required artifact is missing: $required"
done

observed_flashinfer_commit="$(git -C "$FLASHINFER_ROOT" rev-parse HEAD)"
[[ "$observed_flashinfer_commit" == "$FLASHINFER_COMMIT" ]] || die \
  "FlashInfer checkout $observed_flashinfer_commit != pinned $FLASHINFER_COMMIT"

require_idle_gpus

if ss -ltn "sport = :$PORT" | tail -n +2 | grep -q .; then
  die "port $PORT is already in use"
fi

mkdir -p "$OUTPUT_ROOT"
cp -- "$PROMPT_TABLETOP" "$OUTPUT_ROOT/prompt-1-tabletop.txt"
cp -- "$PROMPT_PANCAKE" "$OUTPUT_ROOT/prompt-2-pancake.txt"

export PYTHONNOUSERSITE=1
export PYTHONPATH="$FLASHINFER_ROOT:$FASTVIDEO_KERNEL_PYTHON:$VLLM_SOURCE${PYTHONPATH:+:$PYTHONPATH}"
export FLASHINFER_WORKSPACE_BASE="$FLASHINFER_WORKSPACE"
export FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED=0
unset FLASH_ATTENTION_CUTE_DSL_CACHE_DIR
export FASTVIDEO_VSA_SM100A=0
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$GPU_ORDER"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_OMNI_VIDEO_SYNC_TIMEOUT=3600
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export HF_HOME="${HF_HOME:-$WORKSPACE/hf-cache}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$WORKSPACE/xdg-cache}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$WORKSPACE/torchinductor-cache}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$WORKSPACE/triton-cache}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export OMP_NUM_THREADS=28
export MKL_NUM_THREADS=28
export OPENBLAS_NUM_THREADS=28
export VLLM_OMNI_MINIMAX_H3_CHUNKED_CPU_MP4_OUTPUT=1
export VLLM_OMNI_MINIMAX_H3_CHUNKED_CPU_MP4_SLOTS=2
export VLLM_OMNI_MINIMAX_H3_GPU_UINT8_OUTPUT=0
export MINIMAX_H3_VAE_NVENC_PIPELINE=0
export VLLM_OMNI_MINIMAX_H3_TAEH3_AUDIO_DECODE_OVERLAP=0
export VLLM_OMNI_MINIMAX_H3_TAEH3_CROSS_RANK_AUDIO_DECODE=0
export VLLM_OMNI_MINIMAX_H3_TAEH3_CROSS_RANK_AUDIO_ENCODE=0
export VLLM_OMNI_MINIMAX_H3_QCHUNK_PIPELINE=0
export VLLM_OMNI_FASTVIDEO_VSA_FUSED_TILE_PACK=0
export VLLM_OMNI_FASTVIDEO_VSA_FUSED_UNTILE=0
export VLLM_OMNI_FASTVIDEO_VSA_FUSED_GATE_UNTILE=0
export VLLM_OMNI_FASTVIDEO_VSA_DEFERRED_GATE_SP=0
export VLLM_OMNI_FASTVIDEO_VSA_O_BUNDLE=0
export VLLM_OMNI_FASTVIDEO_VSA_GATE_COMPUTE_OVERLAP=0
export VLLM_OMNI_FASTVIDEO_VSA_DIRECT_Q2K=0
export VLLM_OMNI_FASTVIDEO_VSA_SKIP_SOFTMAX_THRESHOLD_SCALE_FACTOR=0
export VLLM_OMNI_FLASHINFER_ULYSSES_QK_PRODUCER_DIRECT=0
export VLLM_OMNI_FLASHINFER_ULYSSES_O_PRODUCER_DIRECT=0
export VLLM_OMNI_FLASHINFER_ULYSSES_QKV_E4M3_TRANSPORT=0
export VLLM_OMNI_FLASHINFER_ULYSSES_RELEASE_AFTER_DENOISE=0
unset VLLM_OMNI_FLASHINFER_ULYSSES_QKV_SINGLE_PHASE
unset VLLM_OMNI_MINIMAX_H3_QKV_TRANSPORT_OBSERVER_MODE
unset VLLM_OMNI_MINIMAX_H3_QKV_TRANSPORT_OBSERVER_SNAPSHOT_DIR
unset VLLM_OMNI_MINIMAX_H3_QKV_TRANSPORT_OBSERVER_SCALES_PATH

mkdir -p "$FLASHINFER_WORKSPACE_BASE"

adaln_prevalidation="$($PYTHON - "$ADALN_CACHE" "$FASTH3_LORA" <<'PY'
import json
import sys

from vllm_omni.diffusion.models.minimax_h3.adaln_cache import (
    prevalidate_minimax_h3_fasth3_adaln_sidecar,
)

binding = prevalidate_minimax_h3_fasth3_adaln_sidecar(
    sys.argv[1],
    adapter_path=sys.argv[2],
    model_variant="fl2va",
    mode="t2va",
    base_schedule=(0.999, 0.749, 0.5, 0.25, 0.0),
    flow_shift=12.0,
    audio_flow_shift=3.0,
)
print(
    "__FASTH3_ADALN_BINDING__"
    + json.dumps(binding.to_dict(), allow_nan=False, separators=(",", ":"), sort_keys=True)
)
PY
)"
adaln_binding="$(printf '%s\n' "$adaln_prevalidation" \
  | sed -n 's/^__FASTH3_ADALN_BINDING__//p' \
  | tail -n 1)"
[[ -n "$adaln_binding" ]] || die "FastH3 AdaLN sidecar prevalidation returned an empty binding"

cache_config="$($PYTHON - "$ADALN_CACHE" "$adaln_binding" <<'PY'
import json
import sys

print(json.dumps({
    "minimax_h3_adaln_cache_path": sys.argv[1],
    "minimax_h3_adaln_cache_binding": json.loads(sys.argv[2]),
}, allow_nan=False, separators=(",", ":"), sort_keys=True))
PY
)"

attention_config="$($PYTHON - <<'PY'
import json

print(json.dumps({
    "default": {
        "backend": "FASTVIDEO_VSA",
        "fastvideo_vsa_h3_kernel_backend": "flashinfer",
        "fastvideo_vsa_topk": 162,
    },
    "per_role": {
        "minimax_h3.token_refiner": {"backend": "CUDNN_ATTN"},
    },
}, separators=(",", ":"), sort_keys=True))
PY
)"

fp8_prevalidation="$($PYTHON - <<'PY'
from vllm_omni.diffusion.models.minimax_h3.fp8_contract import (
    minimax_h3_all_main_fp8_config_sha256,
    minimax_h3_all_main_fp8_contract_sha256,
    minimax_h3_all_main_fp8_quantization_config_json,
)
print("__FP8_CONFIG__" + minimax_h3_all_main_fp8_quantization_config_json())
print("__FP8_CONFIG_SHA256__" + minimax_h3_all_main_fp8_config_sha256())
print("__FP8_CONTRACT_SHA256__" + minimax_h3_all_main_fp8_contract_sha256())
PY
)"
fp8_config="$(printf '%s\n' "$fp8_prevalidation" | sed -n 's/^__FP8_CONFIG__//p' | tail -n 1)"
fp8_config_sha256="$(printf '%s\n' "$fp8_prevalidation" | sed -n 's/^__FP8_CONFIG_SHA256__//p' | tail -n 1)"
fp8_contract_sha256="$(printf '%s\n' "$fp8_prevalidation" | sed -n 's/^__FP8_CONTRACT_SHA256__//p' | tail -n 1)"
[[ -n "$fp8_config" && -n "$fp8_config_sha256" && -n "$fp8_contract_sha256" ]] \
  || die "all-main FP8 contract prevalidation returned an empty value"

start_server() {
  local variant=$1
  local vae_parallel_size=8
  local variant_dir="$OUTPUT_ROOT/$variant"
  local -a quant_args=()

  require_idle_gpus
  if ss -ltn "sport = :$PORT" | tail -n +2 | rg -q .; then
    die "port $PORT is already in use before starting $variant"
  fi
  mkdir -p "$variant_dir"
  if [[ "$variant" == optimized-fp8-taeh3 ]]; then
    vae_parallel_size=1
    export VLLM_OMNI_MINIMAX_H3_VIDEO_DECODE_BACKEND=taeh3
    export VLLM_OMNI_MINIMAX_H3_TAEH3_CHECKPOINT="$TAEH3_CHECKPOINT"
    export VLLM_OMNI_MINIMAX_H3_TAEH3_DTYPE=fp16
    export VLLM_OMNI_MINIMAX_H3_TAEH3_CHUNK_SIZE=5
    export VLLM_OMNI_MINIMAX_H3_FP8_SCOPE=all-main
    export VLLM_OMNI_MINIMAX_H3_FP8_CONFIG_SHA256="$fp8_config_sha256"
    export VLLM_OMNI_MINIMAX_H3_FP8_CONTRACT_SHA256="$fp8_contract_sha256"
    quant_args=(--diffusion-quantization-config "$fp8_config")
  else
    unset VLLM_OMNI_MINIMAX_H3_VIDEO_DECODE_BACKEND
    unset VLLM_OMNI_MINIMAX_H3_TAEH3_CHECKPOINT
    unset VLLM_OMNI_MINIMAX_H3_TAEH3_DTYPE
    unset VLLM_OMNI_MINIMAX_H3_TAEH3_CHUNK_SIZE
    unset VLLM_OMNI_MINIMAX_H3_FP8_SCOPE
    unset VLLM_OMNI_MINIMAX_H3_FP8_CONFIG_SHA256
    unset VLLM_OMNI_MINIMAX_H3_FP8_CONTRACT_SHA256
  fi

  local -a server_args=(
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
    --text-encoder-tp-size 8
    --vae-patch-parallel-size "$vae_parallel_size"
    --vae-parallel-mode tile
    --vae-use-tiling
    --diffusion-attention-config "$attention_config"
    --cache-backend none
    --cache-config "$cache_config"
    --no-ulysses-a2a-permute
    --task-type fl2va
    --lora-path "$FASTH3_LORA"
    --lora-scale 1.0
    "${quant_args[@]}"
  )

  printf '%q ' numactl --interleave=0,1 "$VLLM" "${server_args[@]}" >"$variant_dir/server-command.txt"
  printf '\n' >>"$variant_dir/server-command.txt"
  setsid numactl --interleave=0,1 "$VLLM" "${server_args[@]}" >"$variant_dir/server.log" 2>&1 &
  SERVER_PID=$!
  echo "$SERVER_PID" >"$variant_dir/server.pid"

  local deadline=$((SECONDS + 1200))
  until curl -fsS --max-time 2 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
      tail -n 240 "$variant_dir/server.log" >&2 || true
      die "$variant server exited before readiness"
    fi
    (( SECONDS < deadline )) || die "$variant server did not become ready"
    sleep 5
  done
  kill -0 "$SERVER_PID" 2>/dev/null || die "$variant server exited immediately after readiness"
  if [[ "$variant" == optimized-fp8-taeh3 ]]; then
    local marker_deadline=$((SECONDS + 180))
    until [[ "$(rg -c 'MINIMAX_H3_ALL_MAIN_FP8_AUDIT_V1 .*status=ok' "$variant_dir/server.log" || true)" -ge 8 ]] \
      && rg -q 'MINIMAX_H3_TAEH3 marker=v1 status=active .*output_rank=True dtype=fp16' "$variant_dir/server.log"; do
      kill -0 "$SERVER_PID" 2>/dev/null || die "$variant server exited before FP8/TAEH3 audit markers"
      (( SECONDS < marker_deadline )) || die "$variant server did not emit complete FP8/TAEH3 audit markers"
      sleep 2
    done
    if rg -q 'E4M3 QKV transport scales installed|FP8 QKV transport receive path' "$variant_dir/server.log"; then
      die "$variant unexpectedly enabled E4M3 QKV transport"
    fi
    rg -q 'qkv_proj_output_dtype=bfloat16' "$variant_dir/server.log" \
      || die "$variant did not confirm BF16 QKV projection output"
  fi
  echo "$variant server ready"
}

request_video() {
  local variant=$1
  local prompt_id=$2
  local prompt_file=$3
  local width=$4
  local height=$5
  local aspect_ratio=$6
  local requested_duration=$7
  local expected_frames=$8
  local output="$OUTPUT_ROOT/$variant/$prompt_id.mp4"
  local output_part="${output}.part.$$"
  local metadata="$OUTPUT_ROOT/$variant/$prompt_id.json"
  local metadata_part="${metadata}.part.$$"
  local prompt start_ns end_ns elapsed

  prompt="$(<"$prompt_file")"
  start_ns="$(date +%s%N)"
  curl --fail-with-body --silent --show-error --max-time 3600 \
    -X POST "http://127.0.0.1:$PORT/v1/videos/sync" \
    --form-string "prompt=$prompt" \
    -F "width=$width" \
    -F "height=$height" \
    -F fps=24 \
    -F num_inference_steps=4 \
    -F flow_shift=12 \
    -F seed=1101 \
    -F "aspect_ratio=$aspect_ratio" \
    -F quality=lossless \
    --form-string "extra_params={\"task\":\"t2va\",\"duration\":$requested_duration,\"aspect_ratio\":\"$aspect_ratio\",\"audio_flow_shift\":3.0}" \
    -o "$output_part"
  end_ns="$(date +%s%N)"
  elapsed="$($PYTHON -c 'import sys; print(f"{(int(sys.argv[2])-int(sys.argv[1]))/1e9:.3f}")' "$start_ns" "$end_ns")"

  PROMPT_ID="$prompt_id" VARIANT="$variant" PROMPT_FILE="$prompt_file" \
    OUTPUT="$output_part" ELAPSED="$elapsed" WIDTH="$width" HEIGHT="$height" \
    ASPECT_RATIO="$aspect_ratio" REQUESTED_DURATION="$requested_duration" FFPROBE="$FFPROBE" \
    EXPECTED_FRAMES="$expected_frames" "$PYTHON" - <<'PY' >"$metadata_part"
import hashlib
import json
import os
import subprocess
from fractions import Fraction

output = os.environ["OUTPUT"]
raw = subprocess.check_output([
    os.environ["FFPROBE"], "-v", "error", "-show_entries",
    "format=duration,size:stream=index,codec_type,codec_name,width,height,avg_frame_rate,nb_frames,sample_rate,channels",
    "-of", "json", output,
], text=True)
probe = json.loads(raw)
video = [item for item in probe.get("streams", []) if item.get("codec_type") == "video"]
audio = [item for item in probe.get("streams", []) if item.get("codec_type") == "audio"]
if len(video) != 1 or len(audio) != 1:
    raise SystemExit(f"expected one video and one audio stream, got video={len(video)} audio={len(audio)}")
if video[0].get("codec_name") != "h264" or audio[0].get("codec_name") != "aac":
    raise SystemExit(f"unexpected codecs: video={video[0].get('codec_name')} audio={audio[0].get('codec_name')}")
expected = (int(os.environ["WIDTH"]), int(os.environ["HEIGHT"]))
actual = (int(video[0].get("width", 0)), int(video[0].get("height", 0)))
if actual != expected:
    raise SystemExit(f"unexpected video geometry {actual}, expected {expected}")
expected_frames = int(os.environ["EXPECTED_FRAMES"])
actual_frames = int(video[0].get("nb_frames", 0))
if actual_frames != expected_frames:
    raise SystemExit(f"unexpected frame count {actual_frames}, expected {expected_frames}")
if Fraction(video[0].get("avg_frame_rate", "0/1")) != Fraction(24, 1):
    raise SystemExit(f"unexpected frame rate {video[0].get('avg_frame_rate')}, expected 24/1")
if int(audio[0].get("sample_rate", 0)) != 32000 or int(audio[0].get("channels", 0)) != 2:
    raise SystemExit(
        f"unexpected audio contract sample_rate={audio[0].get('sample_rate')} "
        f"channels={audio[0].get('channels')}, expected 32000 Hz stereo"
    )
expected_media_seconds = expected_frames / 24
actual_media_seconds = float(probe.get("format", {}).get("duration", 0.0))
if abs(actual_media_seconds - expected_media_seconds) > 0.05:
    raise SystemExit(
        f"unexpected media duration {actual_media_seconds:.6f}, "
        f"expected approximately {expected_media_seconds:.6f}"
    )
payload = open(output, "rb").read()
prompt_payload = open(os.environ["PROMPT_FILE"], "rb").read().rstrip(b"\n")
print(json.dumps({
    "prompt_id": os.environ["PROMPT_ID"],
    "variant": os.environ["VARIANT"],
    "prompt_sha256": hashlib.sha256(prompt_payload).hexdigest(),
    "seed": 1101,
    "requested_duration_seconds": float(os.environ["REQUESTED_DURATION"]),
    "aligned_frame_count": expected_frames,
    "aligned_media_seconds": expected_media_seconds,
    "aspect_ratio": os.environ["ASPECT_RATIO"],
    "requested_canvas": [expected[0], expected[1]],
    "client_e2e_seconds": float(os.environ["ELAPSED"]),
    "mp4_bytes": len(payload),
    "mp4_sha256": hashlib.sha256(payload).hexdigest(),
    "ffprobe": probe,
}, allow_nan=False, indent=2, sort_keys=True))
PY
  mv -- "$output_part" "$output"
  mv -- "$metadata_part" "$metadata"
  echo "$variant $prompt_id complete: ${elapsed}s -> $output"
}

cat >"$OUTPUT_ROOT/contract.txt" <<EOF
framework=vllm-omni
model=MiniMax-H3/FL2VA + FastH3 VSA/Data-Free adapter
schedule=4 DiT forwards
attention=FlashInfer PR #4944 VSA, tile 64, top-k 162
parallelism=TP1/SP8, NCCL Ulysses
seed=1101
warmup_requests=0
qkv_wire=bfloat16
e4m3_qkv_transport=disabled
control=BF16 main DiT + full H3 VAE PP8
candidate=all-main FP8 main DiT + TAEH3 FP16 rank0
timing_use=artifact-local diagnostic only; not a performance A/B
EOF

if [[ "$PHASE" == both || "$PHASE" == control ]]; then
  start_server standard-bf16-full-vae
  request_video standard-bf16-full-vae prompt-1-tabletop "$PROMPT_TABLETOP" 768 768 1:1 6 158
  request_video standard-bf16-full-vae prompt-2-pancake "$PROMPT_PANCAKE" 768 1344 9:16 12 294
  stop_server
fi

if [[ "$PHASE" == both || "$PHASE" == candidate ]]; then
  start_server optimized-fp8-taeh3
  request_video optimized-fp8-taeh3 prompt-1-tabletop "$PROMPT_TABLETOP" 768 768 1:1 6 158
  request_video optimized-fp8-taeh3 prompt-2-pancake "$PROMPT_PANCAKE" 768 1344 9:16 12 294
  stop_server
fi

echo "$OUTPUT_ROOT"
