# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Deferred MiniMax-H3 VSA coarse-gate helpers.

Strict Ulysses normally sends the learned VSA gate through the same full-length
sequence-to-head all-to-all as Q/K/V.  The coarse VSA output is only one vector
per tile, so an equivalent ordering is:

1. keep the gate in its original local-sequence/all-head layout;
2. reverse-Ulysses the fine attention output;
3. all-gather the compact coarse output across head owners; and
4. apply ``fine + coarse[row_tile] * gate`` on the local sequence shard.

This module owns only the row mapping and the exact BF16 pointwise operation.
The communication and feature-contract checks remain in Ulysses.  Everything
is inference-only and behind an explicit, default-off environment gate.
"""

from __future__ import annotations

import os
import weakref

import torch
from vllm.triton_utils import HAS_TRITON, tl, triton

H3_VSA_DEFERRED_GATE_SP_ENV = "VLLM_OMNI_FASTVIDEO_VSA_DEFERRED_GATE_SP"
H3_VSA_DEFERRED_ACTIVE_KEY = "vsa_h3_deferred_gate_active"
H3_VSA_DEFERRED_STATE_KEY = "vsa_h3_deferred_gate_state"

_VALIDATED_ROW_BLOCKS: dict[
    tuple[str, int | None, int],
    tuple[weakref.ReferenceType[torch.Tensor], int, int],
] = {}


def h3_vsa_deferred_gate_sp_enabled() -> bool:
    """Return whether the strict-SP deferred-gate experiment is requested."""
    raw = os.environ.get(H3_VSA_DEFERRED_GATE_SP_ENV, "0").strip()
    if raw not in {"0", "1"}:
        raise ValueError(f"{H3_VSA_DEFERRED_GATE_SP_ENV} must be exactly '0' or '1', got {raw!r}")
    return raw == "1"


def _row_blocks_key(row_blocks: torch.Tensor) -> tuple[str, int | None, int]:
    return (
        row_blocks.device.type,
        row_blocks.device.index,
        row_blocks.data_ptr(),
    )


def _register_row_blocks(
    row_blocks: torch.Tensor,
    *,
    num_blocks: int,
) -> torch.Tensor:
    key = _row_blocks_key(row_blocks)

    def cleanup(reference: weakref.ReferenceType[torch.Tensor]) -> None:
        current = _VALIDATED_ROW_BLOCKS.get(key)
        if current is not None and current[0] is reference:
            _VALIDATED_ROW_BLOCKS.pop(key, None)

    reference = weakref.ref(row_blocks, cleanup)
    _VALIDATED_ROW_BLOCKS[key] = (
        reference,
        row_blocks._version,
        num_blocks,
    )
    return row_blocks


def _row_blocks_are_registered(
    row_blocks: torch.Tensor,
    *,
    num_blocks: int,
) -> bool:
    entry = _VALIDATED_ROW_BLOCKS.get(_row_blocks_key(row_blocks))
    if entry is None:
        return False
    reference, version, registered_blocks = entry
    return reference() is row_blocks and version == row_blocks._version and registered_blocks == num_blocks


def build_h3_compact_row_blocks(
    untile: torch.Tensor,
    *,
    num_blocks: int,
) -> torch.Tensor:
    """Map each compact/global valid H3 row to its tile-64 coarse row.

    ``untile[compact_row]`` is the source row in the tile-padded VSA output.
    Integer division by 64 therefore yields the exact coarse-output row used by
    that compact row.  The builder validates once and registers the result so
    the asynchronous CUDA kernel never consumes an untrusted index tensor.
    """
    if untile.ndim != 1:
        raise ValueError(f"untile must be one-dimensional, got {tuple(untile.shape)}")
    if untile.dtype not in (torch.int32, torch.int64):
        raise TypeError(f"untile must use int32 or int64, got {untile.dtype}")
    if not untile.numel():
        raise ValueError("untile must contain at least one valid compact row")
    if num_blocks <= 0:
        raise ValueError(f"num_blocks must be positive, got {num_blocks}")
    if int(untile.min()) < 0 or int(untile.max()) >= num_blocks * 64:
        raise ValueError(f"untile contains an out-of-bounds tiled row for num_blocks={num_blocks}")
    if torch.unique(untile).numel() != untile.numel():
        raise ValueError("untile must contain unique tiled rows")

    row_blocks = torch.div(untile, 64, rounding_mode="floor").to(torch.int32)
    return _register_row_blocks(row_blocks.contiguous(), num_blocks=num_blocks)


def _validate_deferred_gate_inputs(
    fine: torch.Tensor,
    coarse: torch.Tensor,
    gate: torch.Tensor,
    row_blocks: torch.Tensor,
    *,
    global_row_start: int,
) -> None:
    if fine.ndim != 4 or coarse.ndim != 4 or gate.ndim != 4:
        raise ValueError("fine, coarse, and gate must be [B,S,H,D] tensors")
    if fine.shape != gate.shape:
        raise ValueError(f"fine and gate must have identical local shape, got {fine.shape} and {gate.shape}")
    if coarse.shape[0] != fine.shape[0] or coarse.shape[2:] != fine.shape[2:]:
        raise ValueError(f"coarse must share fine's batch/head geometry, got coarse={coarse.shape}, fine={fine.shape}")
    if fine.dtype != torch.bfloat16 or coarse.dtype != fine.dtype or gate.dtype != fine.dtype:
        raise TypeError("deferred H3 VSA gate-add requires matching BF16 fine/coarse/gate tensors")
    if fine.device != coarse.device or fine.device != gate.device or fine.device != row_blocks.device:
        raise ValueError("fine, coarse, gate, and row_blocks must share one device")
    if row_blocks.ndim != 1 or row_blocks.dtype != torch.int32:
        raise TypeError("row_blocks must be a one-dimensional int32 tensor")
    if global_row_start < 0:
        raise ValueError(f"global_row_start must be non-negative, got {global_row_start}")
    if global_row_start >= row_blocks.numel() and row_blocks.numel() != 0:
        # A trailing padding-only shard is legal only if it begins exactly at
        # the valid prefix boundary. H3 currently has at most suffix padding.
        if global_row_start != row_blocks.numel():
            raise ValueError(
                "local shard starts beyond the valid H3 prefix: "
                f"start={global_row_start}, valid_rows={row_blocks.numel()}"
            )
    if fine.shape[-1] % 2:
        raise ValueError(f"BF16x2 exact arithmetic requires an even head dimension, got {fine.shape[-1]}")
    if fine.requires_grad or coarse.requires_grad or gate.requires_grad:
        raise ValueError("deferred H3 VSA gate-add is inference-only")
    if not fine.is_contiguous() or not gate.is_contiguous():
        raise ValueError("deferred H3 VSA fine and gate tensors must be contiguous")
    if fine.untyped_storage().data_ptr() in {
        coarse.untyped_storage().data_ptr(),
        gate.untyped_storage().data_ptr(),
    }:
        raise ValueError("fine output must not share storage with coarse or gate")


def h3_vsa_deferred_gate_reference(
    fine: torch.Tensor,
    coarse: torch.Tensor,
    gate: torch.Tensor,
    row_blocks: torch.Tensor,
    *,
    global_row_start: int,
) -> torch.Tensor:
    """Reference local-shard gate add with eager BF16 operation boundaries."""
    _validate_deferred_gate_inputs(
        fine,
        coarse,
        gate,
        row_blocks,
        global_row_start=global_row_start,
    )
    output = fine.clone()
    valid_local_rows = min(
        fine.shape[1],
        max(0, row_blocks.numel() - global_row_start),
    )
    if valid_local_rows:
        local_blocks = row_blocks.narrow(0, global_row_start, valid_local_rows).to(torch.int64)
        selected = coarse.index_select(1, local_blocks)
        # Keep this as two eager tensor operations. The CUDA kernel below uses
        # matching mul.rn/add.rn BF16x2 PTX instructions.
        gated = selected * gate[:, :valid_local_rows]
        output[:, :valid_local_rows] = output[:, :valid_local_rows] + gated
    return output


def h3_vsa_deferred_gate_cuda_supported(
    fine: torch.Tensor,
    coarse: torch.Tensor,
    gate: torch.Tensor,
    row_blocks: torch.Tensor,
    *,
    global_row_start: int,
) -> bool:
    if not HAS_TRITON:
        return False
    if (
        fine.device.type != "cuda"
        or coarse.device != fine.device
        or gate.device != fine.device
        or row_blocks.device != fine.device
        or fine.ndim != 4
        or coarse.ndim != 4
        or gate.shape != fine.shape
        or coarse.shape[0] != fine.shape[0]
        or coarse.shape[2:] != fine.shape[2:]
        or fine.dtype != torch.bfloat16
        or coarse.dtype != fine.dtype
        or gate.dtype != fine.dtype
        or row_blocks.ndim != 1
        or row_blocks.dtype != torch.int32
        or global_row_start < 0
        or global_row_start > row_blocks.numel()
        or fine.shape[-1] % 2
        or fine.requires_grad
        or coarse.requires_grad
        or gate.requires_grad
        or not fine.is_contiguous()
        or not gate.is_contiguous()
    ):
        return False
    if fine.untyped_storage().data_ptr() in {
        coarse.untyped_storage().data_ptr(),
        gate.untyped_storage().data_ptr(),
    }:
        return False
    return _row_blocks_are_registered(
        row_blocks,
        num_blocks=coarse.shape[1],
    )


if HAS_TRITON:

    @triton.jit
    def _h3_vsa_deferred_gate_kernel(
        fine_ptr,
        coarse_ptr,
        gate_ptr,
        row_blocks_ptr,
        fine_stride_batch,
        fine_stride_seq,
        fine_stride_head,
        fine_stride_dim,
        coarse_stride_batch,
        coarse_stride_seq,
        coarse_stride_head,
        coarse_stride_dim,
        gate_stride_batch,
        gate_stride_seq,
        gate_stride_head,
        gate_stride_dim,
        local_rows: tl.constexpr,
        compact_rows: tl.constexpr,
        global_row_start: tl.constexpr,
        head_dim: tl.constexpr,
        row_width: tl.constexpr,
        feature_block: tl.constexpr,
    ):
        flat_row = tl.program_id(0)
        feature_block_index = tl.program_id(1)
        batch_index = flat_row // local_rows
        local_row = flat_row - batch_index * local_rows
        global_row = global_row_start + local_row
        valid_row = global_row < compact_rows
        coarse_row = tl.load(row_blocks_ptr + global_row, mask=valid_row, other=0)

        features = feature_block_index * feature_block + tl.arange(0, feature_block)
        feature_mask = features < row_width
        head_index = features // head_dim
        dim_index = features - head_index * head_dim
        fine_offsets = (
            batch_index * fine_stride_batch
            + local_row * fine_stride_seq
            + head_index * fine_stride_head
            + dim_index * fine_stride_dim
        )
        coarse_offsets = (
            batch_index * coarse_stride_batch
            + coarse_row * coarse_stride_seq
            + head_index * coarse_stride_head
            + dim_index * coarse_stride_dim
        )
        gate_offsets = (
            batch_index * gate_stride_batch
            + local_row * gate_stride_seq
            + head_index * gate_stride_head
            + dim_index * gate_stride_dim
        )
        mask = valid_row & feature_mask
        fine_values = tl.load(fine_ptr + fine_offsets, mask=mask, other=0.0)
        coarse_values = tl.load(coarse_ptr + coarse_offsets, mask=mask, other=0.0)
        gate_values = tl.load(gate_ptr + gate_offsets, mask=mask, other=0.0)
        values = tl.inline_asm_elementwise(
            """
            {
                .reg .b32 gated;
                mul.rn.bf16x2 gated, $2, $3;
                add.rn.bf16x2 $0, $1, gated;
            }
            """,
            constraints="=r,r,r,r",
            args=[fine_values, coarse_values, gate_values],
            dtype=tl.bfloat16,
            is_pure=True,
            pack=2,
        )
        tl.store(fine_ptr + fine_offsets, values, mask=mask)


@torch.compiler.disable
def h3_vsa_deferred_gate_add_(
    fine: torch.Tensor,
    coarse: torch.Tensor,
    gate: torch.Tensor,
    row_blocks: torch.Tensor,
    *,
    global_row_start: int,
) -> torch.Tensor:
    """Apply the exact BF16 deferred gate in place on one SP row shard."""
    _validate_deferred_gate_inputs(
        fine,
        coarse,
        gate,
        row_blocks,
        global_row_start=global_row_start,
    )
    if not h3_vsa_deferred_gate_cuda_supported(
        fine,
        coarse,
        gate,
        row_blocks,
        global_row_start=global_row_start,
    ):
        raise RuntimeError("deferred H3 VSA gate-add requires a registered CUDA row map and BF16 tensors")

    row_width = fine.shape[2] * fine.shape[3]
    feature_block = 1024
    _h3_vsa_deferred_gate_kernel[(fine.shape[0] * fine.shape[1], triton.cdiv(row_width, feature_block))](
        fine,
        coarse,
        gate,
        row_blocks,
        fine.stride(0),
        fine.stride(1),
        fine.stride(2),
        fine.stride(3),
        coarse.stride(0),
        coarse.stride(1),
        coarse.stride(2),
        coarse.stride(3),
        gate.stride(0),
        gate.stride(1),
        gate.stride(2),
        gate.stride(3),
        local_rows=fine.shape[1],
        compact_rows=row_blocks.numel(),
        global_row_start=global_row_start,
        head_dim=fine.shape[3],
        row_width=row_width,
        feature_block=feature_block,
        num_warps=8,
    )
    return fine


__all__ = [
    "H3_VSA_DEFERRED_ACTIVE_KEY",
    "H3_VSA_DEFERRED_GATE_SP_ENV",
    "H3_VSA_DEFERRED_STATE_KEY",
    "build_h3_compact_row_blocks",
    "h3_vsa_deferred_gate_add_",
    "h3_vsa_deferred_gate_cuda_supported",
    "h3_vsa_deferred_gate_reference",
    "h3_vsa_deferred_gate_sp_enabled",
]
