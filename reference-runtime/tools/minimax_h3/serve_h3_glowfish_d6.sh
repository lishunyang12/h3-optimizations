#!/usr/bin/env bash
# Serve the prompt-bound D6 stack qualified at RTF 0.976 on 2026-09-08.
set -euo pipefail

WORKSPACE=/lustre/raplab/client/sylarl/minimax-h3-native
CASE_NAME=vllm-omni-tp1-sp8-fasth3-vsa-flashinfer-topk162-rdma-qk-direct-qkv-e4m3-all-main-fp8-o-bundle-taeh3-fp16-persistent-rdma-cross-rank-audio
export VLLM_OMNI_H3_SERVICE_EVIDENCE_DIR="$WORKSPACE/results/h3-prompt-laundry-d6-repeat3-kmax280-20260908/$CASE_NAME"
export VLLM_OMNI_H3_SERVICE_DIR="${VLLM_OMNI_H3_SERVICE_DIR:-$WORKSPACE/services/minimax-h3-glowfish-d6}"
export VLLM_OMNI_H3_SERVICE_FLASHINFER_MAX_BYTES=175472640

exec "$(dirname "$0")/serve_h3_d6.sh" "$@"
