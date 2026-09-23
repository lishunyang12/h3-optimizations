# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Fail-closed MiniMax-H3 selective online-FP8 execution contracts.

This module describes four cumulative production qualification lanes, not
generic H3 FP8.  The first quantizes exactly ``fc1`` and ``fc2`` in each of the
50 main DiT blocks.  The second additionally quantizes each attention
``out_proj`` while keeping ``qkv_proj`` and ``to_gate_compress`` in BF16.  The
third additionally quantizes the fused ``qkv_proj`` while keeping its output
BF16 and keeping ``to_gate_compress`` in BF16.  The fourth additionally
quantizes exactly the 50 FastH3 VSA ``to_gate_compress`` projections; it is
named ``all-main`` because those are the last main-block token-proportional
linears, not because it quantizes AdaLN, stems, heads, or refiner blocks.  All
four use vLLM's online
E4M3 per-tensor-weight, dynamic-per-token-activation Cutlass path.  Canonical
JSON digests are shared by the launcher, QKV calibration contract, and
worker-side audit so a configuration typo cannot silently become benchmark
evidence.  QKV wire E4M3 remains an independent transport contract.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, NamedTuple

import torch
from vllm.logger import init_logger

logger = init_logger(__name__)

MINIMAX_H3_FP8_SCOPE_ENV = "VLLM_OMNI_MINIMAX_H3_FP8_SCOPE"
MINIMAX_H3_FP8_CONFIG_SHA256_ENV = (
    "VLLM_OMNI_MINIMAX_H3_FP8_CONFIG_SHA256"
)
MINIMAX_H3_FP8_CONTRACT_SHA256_ENV = (
    "VLLM_OMNI_MINIMAX_H3_FP8_CONTRACT_SHA256"
)

MINIMAX_H3_BF16_DIT_QUANT_MODE = "bf16"
MINIMAX_H3_MLP_FP8_SCOPE = "mlp"
MINIMAX_H3_MLP_FP8_DIT_QUANT_MODE = (
    "mlp-fp8-e4m3fn-w-per-tensor-a-per-token-cutlass-sm120-v1"
)
MINIMAX_H3_MLP_FP8_AUDIT_MARKER = "MINIMAX_H3_MLP_FP8_AUDIT_V1"
MINIMAX_H3_MLP_OUT_FP8_SCOPE = "mlp-out"
MINIMAX_H3_MLP_OUT_FP8_DIT_QUANT_MODE = (
    "mlp-out-fp8-e4m3fn-w-per-tensor-a-per-token-cutlass-sm120-v1"
)
MINIMAX_H3_MLP_OUT_FP8_AUDIT_MARKER = (
    "MINIMAX_H3_MLP_OUT_FP8_AUDIT_V1"
)
MINIMAX_H3_MLP_OUT_QKV_FP8_SCOPE = "mlp-out-qkv"
MINIMAX_H3_MLP_OUT_QKV_FP8_DIT_QUANT_MODE = (
    "mlp-out-qkv-fp8-e4m3fn-w-per-tensor-a-per-token-cutlass-sm120-v1"
)
MINIMAX_H3_MLP_OUT_QKV_FP8_AUDIT_MARKER = (
    "MINIMAX_H3_MLP_OUT_QKV_FP8_AUDIT_V1"
)
MINIMAX_H3_ALL_MAIN_FP8_SCOPE = "all-main"
MINIMAX_H3_ALL_MAIN_FP8_DIT_QUANT_MODE = (
    "all-main-fp8-e4m3fn-w-per-tensor-a-per-token-cutlass-sm120-v1"
)
MINIMAX_H3_ALL_MAIN_FP8_AUDIT_MARKER = (
    "MINIMAX_H3_ALL_MAIN_FP8_AUDIT_V1"
)

_NUM_LAYERS = 50
_NUM_REFINER_LAYERS = 2
_HIDDEN_SIZE = 5376
_FFN_HIDDEN_SIZE = 14336
_NUM_HEADS = 56
_HEAD_DIM = 128
_EXPECTED_WORLD_SIZE = 8
_EXPECTED_TP_SIZE = 1
_EXPECTED_SP_SIZE = 8
_EXPECTED_ULYSSES_DEGREE = 8
_EXPECTED_RING_DEGREE = 1
_EXPECTED_ALLGATHER_DEGREE = 1
_EXPECTED_CAPABILITY = (12, 0)
_EXPECTED_FP8_DTYPE = torch.float8_e4m3fn
_EXPECTED_INPUT_DTYPE = torch.bfloat16
_EXPECTED_SCALE_DTYPE = torch.float32

_MAIN_LINEAR_SUFFIXES = (
    "attn.qkv_proj",
    "attn.to_gate_compress",
    "attn.out_proj",
    "mlp.fc1",
    "mlp.fc2",
)
_MLP_TARGET_SUFFIXES = ("mlp.fc1", "mlp.fc2")
_MLP_OUT_TARGET_SUFFIXES = ("attn.out_proj", "mlp.fc1", "mlp.fc2")
_MLP_OUT_BF16_GUARD_SUFFIXES = ("attn.qkv_proj", "attn.to_gate_compress")
_MLP_OUT_QKV_TARGET_SUFFIXES = (
    "attn.qkv_proj",
    "attn.out_proj",
    "mlp.fc1",
    "mlp.fc2",
)
_MLP_OUT_QKV_BF16_GUARD_SUFFIXES = ("attn.to_gate_compress",)
_ALL_MAIN_TARGET_SUFFIXES = _MAIN_LINEAR_SUFFIXES
_REFINER_LINEAR_SUFFIXES = (
    "attn.qkv_proj",
    "attn.out_proj",
    "mlp.fc1",
    "mlp.fc2",
)


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("ascii")).hexdigest()


def minimax_h3_mlp_fp8_ignored_layers() -> tuple[str, ...]:
    """Return the exact ignored-prefix list consumed by upstream Fp8Config."""
    ignored = ["condition_proj"]
    for block in range(_NUM_REFINER_LAYERS):
        ignored.extend(
            f"token_refiner.blocks.{block}.{suffix}"
            for suffix in _REFINER_LINEAR_SUFFIXES
        )
    targets = frozenset(_MLP_TARGET_SUFFIXES)
    for block in range(_NUM_LAYERS):
        ignored.extend(
            f"blocks.{block}.{suffix}"
            for suffix in _MAIN_LINEAR_SUFFIXES
            if suffix not in targets
        )
    return tuple(ignored)


def minimax_h3_mlp_fp8_target_modules() -> tuple[str, ...]:
    return tuple(
        f"blocks.{block}.{suffix}"
        for block in range(_NUM_LAYERS)
        for suffix in _MLP_TARGET_SUFFIXES
    )


def minimax_h3_mlp_out_fp8_ignored_layers() -> tuple[str, ...]:
    """Return the exact ignored list for the cumulative MLP + out-proj rung."""
    ignored = ["condition_proj"]
    for block in range(_NUM_REFINER_LAYERS):
        ignored.extend(
            f"token_refiner.blocks.{block}.{suffix}"
            for suffix in _REFINER_LINEAR_SUFFIXES
        )
    targets = frozenset(_MLP_OUT_TARGET_SUFFIXES)
    for block in range(_NUM_LAYERS):
        ignored.extend(
            f"blocks.{block}.{suffix}"
            for suffix in _MAIN_LINEAR_SUFFIXES
            if suffix not in targets
        )
    return tuple(ignored)


def minimax_h3_mlp_out_fp8_target_modules() -> tuple[str, ...]:
    return tuple(
        f"blocks.{block}.{suffix}"
        for block in range(_NUM_LAYERS)
        for suffix in _MLP_OUT_TARGET_SUFFIXES
    )


def minimax_h3_mlp_out_fp8_bf16_guard_modules() -> tuple[str, ...]:
    return tuple(
        f"blocks.{block}.{suffix}"
        for block in range(_NUM_LAYERS)
        for suffix in _MLP_OUT_BF16_GUARD_SUFFIXES
    )


def minimax_h3_mlp_out_qkv_fp8_ignored_layers() -> tuple[str, ...]:
    """Exact ignored list for the cumulative MLP + out + QKV rung."""
    ignored = ["condition_proj"]
    for block in range(_NUM_REFINER_LAYERS):
        ignored.extend(
            f"token_refiner.blocks.{block}.{suffix}"
            for suffix in _REFINER_LINEAR_SUFFIXES
        )
    targets = frozenset(_MLP_OUT_QKV_TARGET_SUFFIXES)
    for block in range(_NUM_LAYERS):
        ignored.extend(
            f"blocks.{block}.{suffix}"
            for suffix in _MAIN_LINEAR_SUFFIXES
            if suffix not in targets
        )
    return tuple(ignored)


def minimax_h3_mlp_out_qkv_fp8_target_modules() -> tuple[str, ...]:
    return tuple(
        f"blocks.{block}.{suffix}"
        for block in range(_NUM_LAYERS)
        for suffix in _MLP_OUT_QKV_TARGET_SUFFIXES
    )


def minimax_h3_mlp_out_qkv_fp8_bf16_guard_modules() -> tuple[str, ...]:
    return tuple(
        f"blocks.{block}.{suffix}"
        for block in range(_NUM_LAYERS)
        for suffix in _MLP_OUT_QKV_BF16_GUARD_SUFFIXES
    )


def minimax_h3_all_main_fp8_ignored_layers() -> tuple[str, ...]:
    """Ignore every non-main-block linear in the all-main candidate.

    The runtime audit is the second half of this contract: it rejects a dense
    H3 model with no VSA gates and any unexpected FP8 module, so this scope
    cannot silently broaden beyond the exact 250-module inventory.
    """
    ignored = ["condition_proj"]
    for block in range(_NUM_REFINER_LAYERS):
        ignored.extend(
            f"token_refiner.blocks.{block}.{suffix}"
            for suffix in _REFINER_LINEAR_SUFFIXES
        )
    return tuple(ignored)


def minimax_h3_all_main_fp8_target_modules() -> tuple[str, ...]:
    return tuple(
        f"blocks.{block}.{suffix}"
        for block in range(_NUM_LAYERS)
        for suffix in _ALL_MAIN_TARGET_SUFFIXES
    )


def minimax_h3_mlp_fp8_quantization_config() -> dict[str, Any]:
    """Canonical command-line component quantization configuration."""
    return {
        "transformer": {
            "ignored_layers": list(minimax_h3_mlp_fp8_ignored_layers()),
            "method": "fp8",
        }
    }


def minimax_h3_mlp_fp8_quantization_config_json() -> str:
    return _canonical_json(minimax_h3_mlp_fp8_quantization_config())


def minimax_h3_mlp_fp8_config_sha256() -> str:
    return _sha256_json(minimax_h3_mlp_fp8_quantization_config())


def minimax_h3_mlp_out_fp8_quantization_config() -> dict[str, Any]:
    return {
        "transformer": {
            "ignored_layers": list(minimax_h3_mlp_out_fp8_ignored_layers()),
            "method": "fp8",
        }
    }


def minimax_h3_mlp_out_fp8_quantization_config_json() -> str:
    return _canonical_json(minimax_h3_mlp_out_fp8_quantization_config())


def minimax_h3_mlp_out_fp8_config_sha256() -> str:
    return _sha256_json(minimax_h3_mlp_out_fp8_quantization_config())


def minimax_h3_mlp_out_qkv_fp8_quantization_config() -> dict[str, Any]:
    return {
        "transformer": {
            "ignored_layers": list(
                minimax_h3_mlp_out_qkv_fp8_ignored_layers()
            ),
            "method": "fp8",
        }
    }


def minimax_h3_mlp_out_qkv_fp8_quantization_config_json() -> str:
    return _canonical_json(minimax_h3_mlp_out_qkv_fp8_quantization_config())


def minimax_h3_mlp_out_qkv_fp8_config_sha256() -> str:
    return _sha256_json(minimax_h3_mlp_out_qkv_fp8_quantization_config())


def minimax_h3_all_main_fp8_quantization_config() -> dict[str, Any]:
    return {
        "transformer": {
            "ignored_layers": list(minimax_h3_all_main_fp8_ignored_layers()),
            "method": "fp8",
        }
    }


def minimax_h3_all_main_fp8_quantization_config_json() -> str:
    return _canonical_json(minimax_h3_all_main_fp8_quantization_config())


def minimax_h3_all_main_fp8_config_sha256() -> str:
    return _sha256_json(minimax_h3_all_main_fp8_quantization_config())


def minimax_h3_mlp_fp8_contract() -> dict[str, Any]:
    """Canonical cross-process description of the qualified MLP FP8 rung."""
    targets = minimax_h3_mlp_fp8_target_modules()
    return {
        "contract_type": "minimax-h3-mlp-online-fp8-runtime",
        "format_version": "1",
        "scope": MINIMAX_H3_MLP_FP8_SCOPE,
        "dit_quant_mode": MINIMAX_H3_MLP_FP8_DIT_QUANT_MODE,
        "quantization_config": minimax_h3_mlp_fp8_quantization_config(),
        "quantization_config_sha256": minimax_h3_mlp_fp8_config_sha256(),
        "inventory": {
            "target_modules": list(targets),
            "target_module_count": 100,
            "fc1_count": 50,
            "fc2_count": 50,
            "unexpected_fp8_module_count": 0,
            "fallback_count": 0,
        },
        "linear": {
            "quant_method": "Fp8PerTensorOnlineLinearMethod",
            "kernel": "CutlassFP8ScaledMMLinearKernel",
            "weight_dtype": "float8_e4m3fn",
            "weight_quantization": "static_per_tensor_symmetric",
            "activation_dtype": "float8_e4m3fn",
            "activation_quantization": "dynamic_per_token_symmetric",
            "input_dtype": "bfloat16",
            "output_dtype": "bfloat16",
            "source_weight_dtype": "bfloat16",
            "input_scale": None,
            "weight_scale_dtype": "float32",
            "weight_scale_numel": 1,
        },
        "post_load_weight_shapes_kn": {
            "fc1": [_HIDDEN_SIZE, 2 * _FFN_HIDDEN_SIZE],
            "fc2": [_FFN_HIDDEN_SIZE, _HIDDEN_SIZE],
        },
        "topology": {
            "gpu_architecture": "sm120",
            "world_size": _EXPECTED_WORLD_SIZE,
            "tensor_parallel_size": _EXPECTED_TP_SIZE,
            "sequence_parallel_size": _EXPECTED_WORLD_SIZE,
        },
    }


def minimax_h3_mlp_fp8_contract_json() -> str:
    return _canonical_json(minimax_h3_mlp_fp8_contract())


def minimax_h3_mlp_fp8_contract_sha256() -> str:
    return _sha256_json(minimax_h3_mlp_fp8_contract())


def minimax_h3_mlp_out_fp8_contract() -> dict[str, Any]:
    """Canonical contract for exactly 50 fc1 + 50 fc2 + 50 out_proj."""
    targets = minimax_h3_mlp_out_fp8_target_modules()
    bf16_guards = minimax_h3_mlp_out_fp8_bf16_guard_modules()
    return {
        "contract_type": "minimax-h3-mlp-out-online-fp8-runtime",
        "format_version": "1",
        "scope": MINIMAX_H3_MLP_OUT_FP8_SCOPE,
        "dit_quant_mode": MINIMAX_H3_MLP_OUT_FP8_DIT_QUANT_MODE,
        "quantization_config": minimax_h3_mlp_out_fp8_quantization_config(),
        "quantization_config_sha256": minimax_h3_mlp_out_fp8_config_sha256(),
        "inventory": {
            "target_modules": list(targets),
            "target_module_count": 150,
            "fc1_count": 50,
            "fc2_count": 50,
            "out_proj_count": 50,
            "bf16_guard_modules": list(bf16_guards),
            "qkv_proj_bf16_count": 50,
            "to_gate_compress_bf16_count": 50,
            "unexpected_fp8_module_count": 0,
            "fallback_count": 0,
        },
        "linear": {
            "quant_method": "Fp8PerTensorOnlineLinearMethod",
            "kernel": "CutlassFP8ScaledMMLinearKernel",
            "weight_dtype": "float8_e4m3fn",
            "weight_quantization": "static_per_tensor_symmetric",
            "activation_dtype": "float8_e4m3fn",
            "activation_quantization": "dynamic_per_token_symmetric",
            "input_dtype": "bfloat16",
            "output_dtype": "bfloat16",
            "source_weight_dtype": "bfloat16",
            "input_scale": None,
            "weight_scale_dtype": "float32",
            "weight_scale_numel": 1,
        },
        "post_load_weight_shapes_kn": {
            "out_proj": [_NUM_HEADS * _HEAD_DIM, _HIDDEN_SIZE],
            "fc1": [_HIDDEN_SIZE, 2 * _FFN_HIDDEN_SIZE],
            "fc2": [_FFN_HIDDEN_SIZE, _HIDDEN_SIZE],
        },
        "topology": {
            "gpu_architecture": "sm120",
            "world_size": _EXPECTED_WORLD_SIZE,
            "tensor_parallel_size": _EXPECTED_TP_SIZE,
            "sequence_parallel_size": _EXPECTED_SP_SIZE,
            "ulysses_degree": _EXPECTED_ULYSSES_DEGREE,
            "ring_degree": _EXPECTED_RING_DEGREE,
            "allgather_degree": _EXPECTED_ALLGATHER_DEGREE,
        },
    }


def minimax_h3_mlp_out_fp8_contract_json() -> str:
    return _canonical_json(minimax_h3_mlp_out_fp8_contract())


def minimax_h3_mlp_out_fp8_contract_sha256() -> str:
    return _sha256_json(minimax_h3_mlp_out_fp8_contract())


def minimax_h3_mlp_out_qkv_fp8_contract() -> dict[str, Any]:
    """Canonical contract for exactly 200 main-block online-FP8 linears."""
    targets = minimax_h3_mlp_out_qkv_fp8_target_modules()
    bf16_guards = minimax_h3_mlp_out_qkv_fp8_bf16_guard_modules()
    return {
        "contract_type": "minimax-h3-mlp-out-qkv-online-fp8-runtime",
        "format_version": "1",
        "scope": MINIMAX_H3_MLP_OUT_QKV_FP8_SCOPE,
        "dit_quant_mode": MINIMAX_H3_MLP_OUT_QKV_FP8_DIT_QUANT_MODE,
        "quantization_config": (
            minimax_h3_mlp_out_qkv_fp8_quantization_config()
        ),
        "quantization_config_sha256": (
            minimax_h3_mlp_out_qkv_fp8_config_sha256()
        ),
        "inventory": {
            "target_modules": list(targets),
            "target_module_count": 200,
            "fc1_count": 50,
            "fc2_count": 50,
            "out_proj_count": 50,
            "qkv_proj_count": 50,
            "qkv_proj_output_dtype": "bfloat16",
            "bf16_guard_modules": list(bf16_guards),
            "to_gate_compress_bf16_count": 50,
            "unexpected_fp8_module_count": 0,
            "fallback_count": 0,
        },
        "linear": {
            "quant_method": "Fp8PerTensorOnlineLinearMethod",
            "kernel": "CutlassFP8ScaledMMLinearKernel",
            "weight_dtype": "float8_e4m3fn",
            "weight_quantization": "static_per_tensor_symmetric",
            "activation_dtype": "float8_e4m3fn",
            "activation_quantization": "dynamic_per_token_symmetric",
            "input_dtype": "bfloat16",
            "output_dtype": "bfloat16",
            "source_weight_dtype": "bfloat16",
            "input_scale": None,
            "weight_scale_dtype": "float32",
            "weight_scale_numel": 1,
        },
        "post_load_weight_shapes_kn": {
            "qkv_proj": [
                _HIDDEN_SIZE,
                3 * _NUM_HEADS * _HEAD_DIM,
            ],
            "out_proj": [_NUM_HEADS * _HEAD_DIM, _HIDDEN_SIZE],
            "fc1": [_HIDDEN_SIZE, 2 * _FFN_HIDDEN_SIZE],
            "fc2": [_FFN_HIDDEN_SIZE, _HIDDEN_SIZE],
        },
        "topology": {
            "gpu_architecture": "sm120",
            "world_size": _EXPECTED_WORLD_SIZE,
            "tensor_parallel_size": _EXPECTED_TP_SIZE,
            "sequence_parallel_size": _EXPECTED_SP_SIZE,
            "ulysses_degree": _EXPECTED_ULYSSES_DEGREE,
            "ring_degree": _EXPECTED_RING_DEGREE,
            "allgather_degree": _EXPECTED_ALLGATHER_DEGREE,
        },
    }


def minimax_h3_mlp_out_qkv_fp8_contract_json() -> str:
    return _canonical_json(minimax_h3_mlp_out_qkv_fp8_contract())


def minimax_h3_mlp_out_qkv_fp8_contract_sha256() -> str:
    return _sha256_json(minimax_h3_mlp_out_qkv_fp8_contract())


def minimax_h3_all_main_fp8_contract() -> dict[str, Any]:
    """Canonical contract for exactly 250 main-block online-FP8 linears."""
    targets = minimax_h3_all_main_fp8_target_modules()
    return {
        "contract_type": "minimax-h3-all-main-online-fp8-runtime",
        "format_version": "1",
        "scope": MINIMAX_H3_ALL_MAIN_FP8_SCOPE,
        "dit_quant_mode": MINIMAX_H3_ALL_MAIN_FP8_DIT_QUANT_MODE,
        "quantization_config": minimax_h3_all_main_fp8_quantization_config(),
        "quantization_config_sha256": minimax_h3_all_main_fp8_config_sha256(),
        "inventory": {
            "target_modules": list(targets),
            "target_module_count": 250,
            "fc1_count": 50,
            "fc2_count": 50,
            "out_proj_count": 50,
            "qkv_proj_count": 50,
            "qkv_proj_output_dtype": "bfloat16",
            "to_gate_compress_count": 50,
            "to_gate_compress_output_dtype": "bfloat16",
            "unexpected_fp8_module_count": 0,
            "fallback_count": 0,
        },
        "linear": {
            "quant_method": "Fp8PerTensorOnlineLinearMethod",
            "kernel": "CutlassFP8ScaledMMLinearKernel",
            "weight_dtype": "float8_e4m3fn",
            "weight_quantization": "static_per_tensor_symmetric",
            "activation_dtype": "float8_e4m3fn",
            "activation_quantization": "dynamic_per_token_symmetric",
            "input_dtype": "bfloat16",
            "output_dtype": "bfloat16",
            "source_weight_dtype": "bfloat16",
            "input_scale": None,
            "weight_scale_dtype": "float32",
            "weight_scale_numel": 1,
        },
        "post_load_weight_shapes_kn": {
            "qkv_proj": [_HIDDEN_SIZE, 3 * _NUM_HEADS * _HEAD_DIM],
            "to_gate_compress": [_HIDDEN_SIZE, _NUM_HEADS * _HEAD_DIM],
            "out_proj": [_NUM_HEADS * _HEAD_DIM, _HIDDEN_SIZE],
            "fc1": [_HIDDEN_SIZE, 2 * _FFN_HIDDEN_SIZE],
            "fc2": [_FFN_HIDDEN_SIZE, _HIDDEN_SIZE],
        },
        "topology": {
            "gpu_architecture": "sm120",
            "world_size": _EXPECTED_WORLD_SIZE,
            "tensor_parallel_size": _EXPECTED_TP_SIZE,
            "sequence_parallel_size": _EXPECTED_SP_SIZE,
            "ulysses_degree": _EXPECTED_ULYSSES_DEGREE,
            "ring_degree": _EXPECTED_RING_DEGREE,
            "allgather_degree": _EXPECTED_ALLGATHER_DEGREE,
        },
    }


def minimax_h3_all_main_fp8_contract_json() -> str:
    return _canonical_json(minimax_h3_all_main_fp8_contract())


def minimax_h3_all_main_fp8_contract_sha256() -> str:
    return _sha256_json(minimax_h3_all_main_fp8_contract())


def _require_digest_env(
    name: str,
    expected: str,
    *,
    contract_name: str,
) -> None:
    actual = os.getenv(name)
    if actual != expected:
        raise ValueError(
            f"{name} must equal the canonical MiniMax-H3 {contract_name} digest "
            f"{expected}, got {actual!r}"
        )


def _validate_production_arch(arch: Any) -> None:
    expected = {
        "num_layers": _NUM_LAYERS,
        "token_refiner_num_layers": _NUM_REFINER_LAYERS,
        "hidden_size": _HIDDEN_SIZE,
        "ffn_hidden_size": _FFN_HIDDEN_SIZE,
        "num_attention_heads": _NUM_HEADS,
        "attention_head_dim": _HEAD_DIM,
    }
    mismatches = {
        name: (getattr(arch, name, None), value)
        for name, value in expected.items()
        if getattr(arch, name, None) != value
    }
    if mismatches:
        raise ValueError(
            "MiniMax-H3 MLP-FP8 is qualified only for the released 50-layer "
            f"architecture; mismatches={mismatches!r}"
        )


def _validate_exact_fp8_config(
    quant_config: Any,
    *,
    ignored_layers: tuple[str, ...],
    contract_name: str,
) -> None:
    # Local import keeps ordinary BF16 model construction independent of the
    # complete upstream FP8 implementation.
    from vllm.model_executor.layers.quantization.fp8 import Fp8Config

    if type(quant_config) is not Fp8Config:
        raise ValueError(
            f"MiniMax-H3 {contract_name} requires the exact upstream Fp8Config type"
        )
    expected_ignored = list(ignored_layers)
    fields = {
        "is_checkpoint_fp8_serialized": False,
        "activation_scheme": "dynamic",
        "weight_block_size": None,
        "store_dtype": None,
        "ignored_layers_match_mode": "exact",
        "ignored_layers": expected_ignored,
    }
    mismatches = {
        name: (getattr(quant_config, name, None), expected)
        for name, expected in fields.items()
        if getattr(quant_config, name, None) != expected
    }
    if mismatches:
        raise ValueError(
            f"MiniMax-H3 {contract_name} quantization configuration does not match the "
            f"canonical contract; mismatches={mismatches!r}"
        )


def resolve_minimax_h3_dit_quant_mode(
    quant_config: Any,
    arch: Any,
) -> str:
    """Resolve the QKV-calibration identity for this exact DiT instance."""
    from .blockwise_experiment import H3BlockwiseConfig, MODE
    from .mxfp8_experiment import MODE as MX_MODE, requested_mode

    if requested_mode() == "mxfp8":
        if quant_config is not None:
            raise ValueError("MXFP8 requires BF16 construction before post-load conversion")
        _validate_production_arch(arch)
        return MX_MODE

    if type(quant_config) is H3BlockwiseConfig:
        _validate_production_arch(arch)
        return MODE
    scope = os.getenv(MINIMAX_H3_FP8_SCOPE_ENV, "none")
    if scope not in {
        MINIMAX_H3_MLP_FP8_SCOPE,
        MINIMAX_H3_MLP_OUT_FP8_SCOPE,
        MINIMAX_H3_MLP_OUT_QKV_FP8_SCOPE,
        MINIMAX_H3_ALL_MAIN_FP8_SCOPE,
    }:
        if quant_config is None:
            return MINIMAX_H3_BF16_DIT_QUANT_MODE
        return type(quant_config).__name__.lower()

    _validate_production_arch(arch)
    if scope == MINIMAX_H3_MLP_FP8_SCOPE:
        ignored_layers = minimax_h3_mlp_fp8_ignored_layers()
        config_sha256 = minimax_h3_mlp_fp8_config_sha256()
        contract_sha256 = minimax_h3_mlp_fp8_contract_sha256()
        dit_quant_mode = MINIMAX_H3_MLP_FP8_DIT_QUANT_MODE
        contract_name = "MLP-FP8"
    elif scope == MINIMAX_H3_MLP_OUT_FP8_SCOPE:
        ignored_layers = minimax_h3_mlp_out_fp8_ignored_layers()
        config_sha256 = minimax_h3_mlp_out_fp8_config_sha256()
        contract_sha256 = minimax_h3_mlp_out_fp8_contract_sha256()
        dit_quant_mode = MINIMAX_H3_MLP_OUT_FP8_DIT_QUANT_MODE
        contract_name = "MLP-OUT-FP8"
    elif scope == MINIMAX_H3_MLP_OUT_QKV_FP8_SCOPE:
        ignored_layers = minimax_h3_mlp_out_qkv_fp8_ignored_layers()
        config_sha256 = minimax_h3_mlp_out_qkv_fp8_config_sha256()
        contract_sha256 = minimax_h3_mlp_out_qkv_fp8_contract_sha256()
        dit_quant_mode = MINIMAX_H3_MLP_OUT_QKV_FP8_DIT_QUANT_MODE
        contract_name = "MLP-OUT-QKV-FP8"
    else:
        ignored_layers = minimax_h3_all_main_fp8_ignored_layers()
        config_sha256 = minimax_h3_all_main_fp8_config_sha256()
        contract_sha256 = minimax_h3_all_main_fp8_contract_sha256()
        dit_quant_mode = MINIMAX_H3_ALL_MAIN_FP8_DIT_QUANT_MODE
        contract_name = "ALL-MAIN-FP8"
    _validate_exact_fp8_config(
        quant_config,
        ignored_layers=ignored_layers,
        contract_name=contract_name,
    )
    _require_digest_env(
        MINIMAX_H3_FP8_CONFIG_SHA256_ENV,
        config_sha256,
        contract_name=contract_name,
    )
    _require_digest_env(
        MINIMAX_H3_FP8_CONTRACT_SHA256_ENV,
        contract_sha256,
        contract_name=contract_name,
    )
    return dit_quant_mode


class _RuntimeSymbols(NamedTuple):
    method_type: type
    kernel_type: type
    weight_quant_key: object
    activation_quant_key: object


def _load_fp8_runtime_symbols() -> _RuntimeSymbols:
    from vllm.model_executor.kernels.linear import (
        CutlassFP8ScaledMMLinearKernel,
    )
    from vllm.model_executor.layers.quantization.online.fp8 import (
        Fp8PerTensorOnlineLinearMethod,
    )
    from vllm.model_executor.layers.quantization.utils.quant_utils import (
        kFp8DynamicTokenSym,
        kFp8StaticTensorSym,
    )

    return _RuntimeSymbols(
        Fp8PerTensorOnlineLinearMethod,
        CutlassFP8ScaledMMLinearKernel,
        kFp8StaticTensorSym,
        kFp8DynamicTokenSym,
    )


@dataclass(frozen=True)
class MiniMaxH3MLPFP8Audit:
    rank: int
    world_size: int
    scope: str
    dit_quant_mode: str
    audit_marker: str
    target_modules: int
    fc1: int
    fc2: int
    out_proj: int
    qkv_proj_bf16: int
    to_gate_compress_bf16: int
    sequence_parallel_size: int
    ulysses_degree: int
    ring_degree: int
    allgather_degree: int
    unexpected_fp8_modules: int
    fallback_count: int
    config_sha256: str
    contract_sha256: str
    qkv_proj: int = 0
    to_gate_compress: int = 0

    def marker(self) -> str:
        return " ".join(
            (
                self.audit_marker,
                f"rank={self.rank}",
                f"world_size={self.world_size}",
                f"scope={self.scope}",
                f"dit_quant_mode={self.dit_quant_mode}",
                f"target_modules={self.target_modules}",
                f"fc1={self.fc1}",
                f"fc2={self.fc2}",
                *(
                    (
                        f"out_proj={self.out_proj}",
                        f"qkv_proj_bf16={self.qkv_proj_bf16}",
                        "to_gate_compress_bf16="
                        f"{self.to_gate_compress_bf16}",
                        "sequence_parallel_size="
                        f"{self.sequence_parallel_size}",
                        f"ulysses_degree={self.ulysses_degree}",
                        f"ring_degree={self.ring_degree}",
                        f"allgather_degree={self.allgather_degree}",
                    )
                    if self.scope == MINIMAX_H3_MLP_OUT_FP8_SCOPE
                    else ()
                ),
                *(
                    (
                        f"out_proj={self.out_proj}",
                        f"qkv_proj={self.qkv_proj}",
                        "qkv_proj_output_dtype=bfloat16",
                        "to_gate_compress_bf16="
                        f"{self.to_gate_compress_bf16}",
                        "sequence_parallel_size="
                        f"{self.sequence_parallel_size}",
                        f"ulysses_degree={self.ulysses_degree}",
                        f"ring_degree={self.ring_degree}",
                        f"allgather_degree={self.allgather_degree}",
                    )
                    if self.scope == MINIMAX_H3_MLP_OUT_QKV_FP8_SCOPE
                    else ()
                ),
                *(
                    (
                        f"out_proj={self.out_proj}",
                        f"qkv_proj={self.qkv_proj}",
                        "qkv_proj_output_dtype=bfloat16",
                        f"to_gate_compress={self.to_gate_compress}",
                        "to_gate_compress_output_dtype=bfloat16",
                        "sequence_parallel_size="
                        f"{self.sequence_parallel_size}",
                        f"ulysses_degree={self.ulysses_degree}",
                        f"ring_degree={self.ring_degree}",
                        f"allgather_degree={self.allgather_degree}",
                    )
                    if self.scope == MINIMAX_H3_ALL_MAIN_FP8_SCOPE
                    else ()
                ),
                "quant_method=Fp8PerTensorOnlineLinearMethod",
                "linear_kernel=CutlassFP8ScaledMMLinearKernel",
                "weight_dtype=float8_e4m3fn",
                "weight_quant=static_per_tensor_symmetric",
                "activation_quant=dynamic_per_token_symmetric",
                "input_dtype=bfloat16",
                "output_dtype=bfloat16",
                "source_weight_dtype=bfloat16",
                "weight_scale_dtype=float32",
                "tp_size=1",
                "gpu_architecture=sm120",
                f"unexpected_fp8_modules={self.unexpected_fp8_modules}",
                f"fallback_count={self.fallback_count}",
                f"config_sha256={self.config_sha256}",
                f"contract_sha256={self.contract_sha256}",
                "status=ok",
            )
        )


def _distributed_identity() -> tuple[int, int]:
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        raise RuntimeError(
            "MiniMax-H3 MLP-FP8 audit requires initialized torch.distributed"
        )
    return torch.distributed.get_rank(), torch.distributed.get_world_size()


def audit_minimax_h3_mlp_fp8_runtime(
    model: torch.nn.Module,
    *,
    _runtime_symbols: _RuntimeSymbols | None = None,
    _rank_world: tuple[int, int] | None = None,
    _device_capability: tuple[int, int] | None = None,
) -> MiniMaxH3MLPFP8Audit | None:
    """Validate the loaded worker and emit one machine-checkable rank marker.

    Private overrides exist solely for CPU unit tests.  Production obtains the
    distributed identity and CUDA capability from the live worker.
    """
    scope = os.getenv(MINIMAX_H3_FP8_SCOPE_ENV, "none")
    if scope not in {
        MINIMAX_H3_MLP_FP8_SCOPE,
        MINIMAX_H3_MLP_OUT_FP8_SCOPE,
        MINIMAX_H3_MLP_OUT_QKV_FP8_SCOPE,
        MINIMAX_H3_ALL_MAIN_FP8_SCOPE,
    }:
        return None
    if getattr(model, "_minimax_h3_mlp_fp8_audited", False):
        return None

    if scope == MINIMAX_H3_MLP_FP8_SCOPE:
        expected_names = minimax_h3_mlp_fp8_target_modules()
        bf16_guard_names: tuple[str, ...] = ()
        dit_quant_mode = MINIMAX_H3_MLP_FP8_DIT_QUANT_MODE
        audit_marker = MINIMAX_H3_MLP_FP8_AUDIT_MARKER
        config_sha256 = minimax_h3_mlp_fp8_config_sha256()
        contract_sha256 = minimax_h3_mlp_fp8_contract_sha256()
    elif scope == MINIMAX_H3_MLP_OUT_FP8_SCOPE:
        expected_names = minimax_h3_mlp_out_fp8_target_modules()
        bf16_guard_names = minimax_h3_mlp_out_fp8_bf16_guard_modules()
        dit_quant_mode = MINIMAX_H3_MLP_OUT_FP8_DIT_QUANT_MODE
        audit_marker = MINIMAX_H3_MLP_OUT_FP8_AUDIT_MARKER
        config_sha256 = minimax_h3_mlp_out_fp8_config_sha256()
        contract_sha256 = minimax_h3_mlp_out_fp8_contract_sha256()
    elif scope == MINIMAX_H3_MLP_OUT_QKV_FP8_SCOPE:
        expected_names = minimax_h3_mlp_out_qkv_fp8_target_modules()
        bf16_guard_names = minimax_h3_mlp_out_qkv_fp8_bf16_guard_modules()
        dit_quant_mode = MINIMAX_H3_MLP_OUT_QKV_FP8_DIT_QUANT_MODE
        audit_marker = MINIMAX_H3_MLP_OUT_QKV_FP8_AUDIT_MARKER
        config_sha256 = minimax_h3_mlp_out_qkv_fp8_config_sha256()
        contract_sha256 = minimax_h3_mlp_out_qkv_fp8_contract_sha256()
    else:
        expected_names = minimax_h3_all_main_fp8_target_modules()
        bf16_guard_names = ()
        dit_quant_mode = MINIMAX_H3_ALL_MAIN_FP8_DIT_QUANT_MODE
        audit_marker = MINIMAX_H3_ALL_MAIN_FP8_AUDIT_MARKER
        config_sha256 = minimax_h3_all_main_fp8_config_sha256()
        contract_sha256 = minimax_h3_all_main_fp8_contract_sha256()

    resolve_minimax_h3_dit_quant_mode(
        getattr(model, "quant_config", None),
        model.arch,
    )
    symbols = _runtime_symbols or _load_fp8_runtime_symbols()
    rank, world_size = _rank_world or _distributed_identity()
    if world_size != _EXPECTED_WORLD_SIZE or rank not in range(world_size):
        raise RuntimeError(
            "MiniMax-H3 MLP-FP8 audit requires all eight workers; "
            f"rank={rank}, world_size={world_size}"
        )
    parallel = getattr(model, "parallel_config", None)
    actual_topology = (
        int(getattr(parallel, "tensor_parallel_size", 1)),
        int(getattr(parallel, "sequence_parallel_size", 1)),
        int(getattr(parallel, "ulysses_degree", 1)),
        int(getattr(parallel, "ring_degree", 1)),
        int(getattr(parallel, "allgather_degree", 1)),
    )
    expected_topology = (
        _EXPECTED_TP_SIZE,
        _EXPECTED_SP_SIZE,
        _EXPECTED_ULYSSES_DEGREE,
        _EXPECTED_RING_DEGREE,
        _EXPECTED_ALLGATHER_DEGREE,
    )
    if actual_topology != expected_topology:
        raise RuntimeError(
            "MiniMax-H3 selective FP8 requires exact TP1/SP8 pure-Ulysses "
            f"topology: expected={expected_topology}, actual={actual_topology}"
        )
    capability = _device_capability
    if capability is None:
        if not torch.cuda.is_available():
            raise RuntimeError("MiniMax-H3 MLP-FP8 audit requires CUDA")
        capability = torch.cuda.get_device_capability()
    if tuple(capability) != _EXPECTED_CAPABILITY:
        raise RuntimeError(
            "MiniMax-H3 MLP-FP8 Cutlass contract requires sm120, got "
            f"sm{capability[0]}{capability[1]}"
        )

    modules = dict(model.named_modules())
    expected_set = frozenset(expected_names)
    missing = (expected_set | frozenset(bf16_guard_names)) - modules.keys()
    if missing:
        raise RuntimeError(
            f"MiniMax-H3 MLP-FP8 target modules are missing: {sorted(missing)!r}"
        )

    actual_fp8: set[str] = set()
    fallback_count = 0
    fc1_count = 0
    fc2_count = 0
    out_proj_count = 0
    qkv_proj_count = 0
    to_gate_compress_count = 0
    expected_shapes = {
        "attn.qkv_proj": (
            _HIDDEN_SIZE,
            3 * _NUM_HEADS * _HEAD_DIM,
        ),
        "attn.out_proj": (_NUM_HEADS * _HEAD_DIM, _HIDDEN_SIZE),
        "attn.to_gate_compress": (
            _HIDDEN_SIZE,
            _NUM_HEADS * _HEAD_DIM,
        ),
        "mlp.fc1": (_HIDDEN_SIZE, 2 * _FFN_HIDDEN_SIZE),
        "mlp.fc2": (_FFN_HIDDEN_SIZE, _HIDDEN_SIZE),
    }
    for name, module in modules.items():
        method = getattr(module, "quant_method", None)
        if type(method) is symbols.method_type:
            actual_fp8.add(name)
        elif method is not None and (
            "fp8" in type(method).__name__.lower()
            or getattr(method, "fp8_linear", None) is not None
        ):
            actual_fp8.add(name)
            fallback_count += 1

    unexpected = actual_fp8 - expected_set
    unquantized = expected_set - actual_fp8
    if unexpected or unquantized:
        raise RuntimeError(
            "MiniMax-H3 MLP-FP8 module inventory mismatch: "
            f"missing_fp8={sorted(unquantized)!r}, unexpected_fp8={sorted(unexpected)!r}"
        )

    for name in expected_names:
        module = modules[name]
        method = module.quant_method
        if type(method) is not symbols.method_type:
            fallback_count += 1
            continue
        if type(getattr(method, "fp8_linear", None)) is not symbols.kernel_type:
            raise RuntimeError(
                f"{name} did not select CutlassFP8ScaledMMLinearKernel"
            )
        if method.weight_quant_key != symbols.weight_quant_key:
            raise RuntimeError(f"{name} weight quantization is not static per-tensor symmetric")
        if method.activation_quant_key != symbols.activation_quant_key:
            raise RuntimeError(f"{name} activation quantization is not dynamic per-token symmetric")
        if method.input_dtype != _EXPECTED_INPUT_DTYPE or method.out_dtype != _EXPECTED_INPUT_DTYPE:
            raise RuntimeError(
                f"{name} must consume and produce bfloat16, got "
                f"{method.input_dtype} -> {method.out_dtype}"
            )
        if getattr(module, "orig_dtype", None) != _EXPECTED_INPUT_DTYPE:
            raise RuntimeError(f"{name} source weight dtype must be bfloat16")
        if getattr(module, "params_dtype", None) != _EXPECTED_INPUT_DTYPE:
            raise RuntimeError(f"{name} params_dtype must be bfloat16")
        if getattr(module, "tp_size", None) != _EXPECTED_TP_SIZE:
            raise RuntimeError(f"{name} must run at TP1")
        if getattr(module, "input_scale", object()) is not None:
            raise RuntimeError(f"{name} must use dynamic activations without input_scale")
        if getattr(module, "_already_called_process_weights_after_loading", False) is not True:
            raise RuntimeError(f"{name} online FP8 post-load processing did not complete")

        if name.endswith("attn.qkv_proj"):
            suffix = "attn.qkv_proj"
        elif name.endswith("attn.to_gate_compress"):
            suffix = "attn.to_gate_compress"
        elif name.endswith("attn.out_proj"):
            suffix = "attn.out_proj"
        elif name.endswith("mlp.fc1"):
            suffix = "mlp.fc1"
        else:
            suffix = "mlp.fc2"
        expected_shape = expected_shapes[suffix]
        weight = getattr(module, "weight", None)
        if weight is None or tuple(weight.shape) != expected_shape:
            raise RuntimeError(
                f"{name} post-load [K,N] weight shape must be {expected_shape}, "
                f"got {None if weight is None else tuple(weight.shape)}"
            )
        if weight.dtype != _EXPECTED_FP8_DTYPE or weight.device.type != "cuda":
            raise RuntimeError(
                f"{name} weight must be CUDA float8_e4m3fn, got "
                f"{getattr(weight, 'device', None)} {getattr(weight, 'dtype', None)}"
            )
        scale = getattr(module, "weight_scale", None)
        if (
            not isinstance(scale, torch.Tensor)
            or scale.dtype != _EXPECTED_SCALE_DTYPE
            or scale.numel() != 1
            or not math.isfinite(float(scale.item()))
            or float(scale.item()) <= 0
        ):
            raise RuntimeError(
                f"{name} weight_scale must be one finite positive float32 value"
            )
        if suffix == "mlp.fc1":
            fc1_count += 1
        elif suffix == "mlp.fc2":
            fc2_count += 1
        elif suffix == "attn.out_proj":
            out_proj_count += 1
        elif suffix == "attn.to_gate_compress":
            to_gate_compress_count += 1
        else:
            qkv_proj_count += 1

    qkv_proj_bf16_count = 0
    to_gate_compress_bf16_count = 0
    guard_contract_name = (
        "MLP-OUT-FP8"
        if scope == MINIMAX_H3_MLP_OUT_FP8_SCOPE
        else "MLP-OUT-QKV-FP8"
    )
    for name in bf16_guard_names:
        module = modules[name]
        weight = getattr(module, "weight", None)
        if weight is None or weight.dtype != _EXPECTED_INPUT_DTYPE:
            raise RuntimeError(
                f"{name} must remain bfloat16 under the {guard_contract_name} contract, "
                f"got {getattr(weight, 'dtype', None)}"
            )
        if name.endswith("attn.qkv_proj"):
            qkv_proj_bf16_count += 1
        else:
            to_gate_compress_bf16_count += 1

    expected_out_proj = 50 if scope != MINIMAX_H3_MLP_FP8_SCOPE else 0
    expected_qkv_proj = (
        50
        if scope
        in {
            MINIMAX_H3_MLP_OUT_QKV_FP8_SCOPE,
            MINIMAX_H3_ALL_MAIN_FP8_SCOPE,
        }
        else 0
    )
    expected_qkv_proj_bf16 = (
        50 if scope == MINIMAX_H3_MLP_OUT_FP8_SCOPE else 0
    )
    expected_to_gate_bf16 = (
        50
        if scope
        in {
            MINIMAX_H3_MLP_OUT_FP8_SCOPE,
            MINIMAX_H3_MLP_OUT_QKV_FP8_SCOPE,
        }
        else 0
    )
    expected_to_gate = 50 if scope == MINIMAX_H3_ALL_MAIN_FP8_SCOPE else 0
    if (
        fallback_count != 0
        or fc1_count != 50
        or fc2_count != 50
        or out_proj_count != expected_out_proj
        or qkv_proj_count != expected_qkv_proj
        or qkv_proj_bf16_count != expected_qkv_proj_bf16
        or to_gate_compress_bf16_count != expected_to_gate_bf16
        or to_gate_compress_count != expected_to_gate
    ):
        qkv_target_detail = (
            f"qkv_proj={qkv_proj_count}, "
            if scope
            in {
                MINIMAX_H3_MLP_OUT_QKV_FP8_SCOPE,
                MINIMAX_H3_ALL_MAIN_FP8_SCOPE,
            }
            else ""
        )
        gate_target_detail = (
            f"to_gate_compress={to_gate_compress_count}, "
            if scope == MINIMAX_H3_ALL_MAIN_FP8_SCOPE
            else ""
        )
        raise RuntimeError(
            "MiniMax-H3 selective FP8 audit did not cover the exact target "
            f"and BF16-guard inventory: fc1={fc1_count}, fc2={fc2_count}, "
            f"out_proj={out_proj_count}, {qkv_target_detail}"
            f"{gate_target_detail}"
            f"qkv_proj_bf16={qkv_proj_bf16_count}, "
            "to_gate_compress_bf16="
            f"{to_gate_compress_bf16_count}, "
            f"fallback_count={fallback_count}"
        )

    audit = MiniMaxH3MLPFP8Audit(
        rank=rank,
        world_size=world_size,
        scope=scope,
        dit_quant_mode=dit_quant_mode,
        audit_marker=audit_marker,
        target_modules=len(expected_names),
        fc1=fc1_count,
        fc2=fc2_count,
        out_proj=out_proj_count,
        qkv_proj=qkv_proj_count,
        to_gate_compress=to_gate_compress_count,
        qkv_proj_bf16=qkv_proj_bf16_count,
        to_gate_compress_bf16=to_gate_compress_bf16_count,
        sequence_parallel_size=actual_topology[1],
        ulysses_degree=actual_topology[2],
        ring_degree=actual_topology[3],
        allgather_degree=actual_topology[4],
        unexpected_fp8_modules=len(unexpected),
        fallback_count=fallback_count,
        config_sha256=config_sha256,
        contract_sha256=contract_sha256,
    )
    logger.info(audit.marker())
    model._minimax_h3_mlp_fp8_audited = True
    return audit


def _validate_minimax_h3_fp8_run_contract(
    value: Mapping[str, Any],
    *,
    expected_sha256: str,
    scope: str,
) -> None:
    """Validate the exact canonical observer contract used for calibration."""
    canonical = _canonical_json(value)
    actual_sha256 = hashlib.sha256(canonical.encode("ascii")).hexdigest()
    if actual_sha256 != expected_sha256:
        raise ValueError(
            "MiniMax-H3 observer run-contract SHA256 mismatch: "
            f"expected={expected_sha256}, actual={actual_sha256}"
        )
    execution = value.get("execution_profile")
    if not isinstance(execution, Mapping):
        raise ValueError("MiniMax-H3 observer run contract lacks execution_profile")
    if scope == MINIMAX_H3_MLP_FP8_SCOPE:
        dit_quant_mode = MINIMAX_H3_MLP_FP8_DIT_QUANT_MODE
        config_sha256 = minimax_h3_mlp_fp8_config_sha256()
        contract_sha256 = minimax_h3_mlp_fp8_contract_sha256()
        contract = minimax_h3_mlp_fp8_contract()
        contract_name = "MLP-FP8"
    elif scope == MINIMAX_H3_MLP_OUT_FP8_SCOPE:
        dit_quant_mode = MINIMAX_H3_MLP_OUT_FP8_DIT_QUANT_MODE
        config_sha256 = minimax_h3_mlp_out_fp8_config_sha256()
        contract_sha256 = minimax_h3_mlp_out_fp8_contract_sha256()
        contract = minimax_h3_mlp_out_fp8_contract()
        contract_name = "MLP-OUT-FP8"
    elif scope == MINIMAX_H3_MLP_OUT_QKV_FP8_SCOPE:
        dit_quant_mode = MINIMAX_H3_MLP_OUT_QKV_FP8_DIT_QUANT_MODE
        config_sha256 = minimax_h3_mlp_out_qkv_fp8_config_sha256()
        contract_sha256 = minimax_h3_mlp_out_qkv_fp8_contract_sha256()
        contract = minimax_h3_mlp_out_qkv_fp8_contract()
        contract_name = "MLP-OUT-QKV-FP8"
    elif scope == MINIMAX_H3_ALL_MAIN_FP8_SCOPE:
        dit_quant_mode = MINIMAX_H3_ALL_MAIN_FP8_DIT_QUANT_MODE
        config_sha256 = minimax_h3_all_main_fp8_config_sha256()
        contract_sha256 = minimax_h3_all_main_fp8_contract_sha256()
        contract = minimax_h3_all_main_fp8_contract()
        contract_name = "ALL-MAIN-FP8"
    else:  # pragma: no cover - public wrappers own this invariant.
        raise ValueError(f"unsupported MiniMax-H3 FP8 scope: {scope!r}")
    expected_fields = {
        "dit_quant_mode": dit_quant_mode,
        "fp8_scope": scope,
        "fp8_config_sha256": config_sha256,
        "fp8_contract_sha256": contract_sha256,
        "fp8_contract": contract,
    }
    mismatches = {
        name: (execution.get(name), expected)
        for name, expected in expected_fields.items()
        if execution.get(name) != expected
    }
    if mismatches:
        raise ValueError(
            f"MiniMax-H3 {contract_name} observer run contract does not bind the exact "
            f"runtime configuration; mismatches={mismatches!r}"
        )


def validate_minimax_h3_mlp_fp8_run_contract(
    value: Mapping[str, Any],
    *,
    expected_sha256: str,
) -> None:
    _validate_minimax_h3_fp8_run_contract(
        value,
        expected_sha256=expected_sha256,
        scope=MINIMAX_H3_MLP_FP8_SCOPE,
    )


def validate_minimax_h3_mlp_out_fp8_run_contract(
    value: Mapping[str, Any],
    *,
    expected_sha256: str,
) -> None:
    _validate_minimax_h3_fp8_run_contract(
        value,
        expected_sha256=expected_sha256,
        scope=MINIMAX_H3_MLP_OUT_FP8_SCOPE,
    )


def validate_minimax_h3_mlp_out_qkv_fp8_run_contract(
    value: Mapping[str, Any],
    *,
    expected_sha256: str,
) -> None:
    _validate_minimax_h3_fp8_run_contract(
        value,
        expected_sha256=expected_sha256,
        scope=MINIMAX_H3_MLP_OUT_QKV_FP8_SCOPE,
    )


def validate_minimax_h3_all_main_fp8_run_contract(
    value: Mapping[str, Any],
    *,
    expected_sha256: str,
) -> None:
    _validate_minimax_h3_fp8_run_contract(
        value,
        expected_sha256=expected_sha256,
        scope=MINIMAX_H3_ALL_MAIN_FP8_SCOPE,
    )


__all__ = [
    "MINIMAX_H3_ALL_MAIN_FP8_AUDIT_MARKER",
    "MINIMAX_H3_ALL_MAIN_FP8_DIT_QUANT_MODE",
    "MINIMAX_H3_ALL_MAIN_FP8_SCOPE",
    "MINIMAX_H3_BF16_DIT_QUANT_MODE",
    "MINIMAX_H3_FP8_CONFIG_SHA256_ENV",
    "MINIMAX_H3_FP8_CONTRACT_SHA256_ENV",
    "MINIMAX_H3_FP8_SCOPE_ENV",
    "MINIMAX_H3_MLP_FP8_AUDIT_MARKER",
    "MINIMAX_H3_MLP_FP8_DIT_QUANT_MODE",
    "MINIMAX_H3_MLP_FP8_SCOPE",
    "MINIMAX_H3_MLP_OUT_FP8_AUDIT_MARKER",
    "MINIMAX_H3_MLP_OUT_FP8_DIT_QUANT_MODE",
    "MINIMAX_H3_MLP_OUT_FP8_SCOPE",
    "MINIMAX_H3_MLP_OUT_QKV_FP8_AUDIT_MARKER",
    "MINIMAX_H3_MLP_OUT_QKV_FP8_DIT_QUANT_MODE",
    "MINIMAX_H3_MLP_OUT_QKV_FP8_SCOPE",
    "MiniMaxH3MLPFP8Audit",
    "audit_minimax_h3_mlp_fp8_runtime",
    "minimax_h3_all_main_fp8_config_sha256",
    "minimax_h3_all_main_fp8_contract",
    "minimax_h3_all_main_fp8_contract_json",
    "minimax_h3_all_main_fp8_contract_sha256",
    "minimax_h3_all_main_fp8_ignored_layers",
    "minimax_h3_all_main_fp8_quantization_config",
    "minimax_h3_all_main_fp8_quantization_config_json",
    "minimax_h3_all_main_fp8_target_modules",
    "minimax_h3_mlp_fp8_config_sha256",
    "minimax_h3_mlp_fp8_contract",
    "minimax_h3_mlp_fp8_contract_json",
    "minimax_h3_mlp_fp8_contract_sha256",
    "minimax_h3_mlp_fp8_ignored_layers",
    "minimax_h3_mlp_fp8_quantization_config",
    "minimax_h3_mlp_fp8_quantization_config_json",
    "minimax_h3_mlp_fp8_target_modules",
    "minimax_h3_mlp_out_fp8_bf16_guard_modules",
    "minimax_h3_mlp_out_fp8_config_sha256",
    "minimax_h3_mlp_out_fp8_contract",
    "minimax_h3_mlp_out_fp8_contract_json",
    "minimax_h3_mlp_out_fp8_contract_sha256",
    "minimax_h3_mlp_out_fp8_ignored_layers",
    "minimax_h3_mlp_out_fp8_quantization_config",
    "minimax_h3_mlp_out_fp8_quantization_config_json",
    "minimax_h3_mlp_out_fp8_target_modules",
    "minimax_h3_mlp_out_qkv_fp8_bf16_guard_modules",
    "minimax_h3_mlp_out_qkv_fp8_config_sha256",
    "minimax_h3_mlp_out_qkv_fp8_contract",
    "minimax_h3_mlp_out_qkv_fp8_contract_json",
    "minimax_h3_mlp_out_qkv_fp8_contract_sha256",
    "minimax_h3_mlp_out_qkv_fp8_ignored_layers",
    "minimax_h3_mlp_out_qkv_fp8_quantization_config",
    "minimax_h3_mlp_out_qkv_fp8_quantization_config_json",
    "minimax_h3_mlp_out_qkv_fp8_target_modules",
    "resolve_minimax_h3_dit_quant_mode",
    "validate_minimax_h3_all_main_fp8_run_contract",
    "validate_minimax_h3_mlp_fp8_run_contract",
    "validate_minimax_h3_mlp_out_fp8_run_contract",
    "validate_minimax_h3_mlp_out_qkv_fp8_run_contract",
]
