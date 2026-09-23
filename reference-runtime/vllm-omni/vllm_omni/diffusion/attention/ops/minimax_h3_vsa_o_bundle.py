# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Exact MiniMax-H3 VSA reverse-O piggyback primitives.

This module implements only the two local layout operations required to carry
compact VSA coarse output in the reverse-Ulysses O exchange:

* bundle global aligned fine output and per-owner compact coarse tiles as
  ``[fine owner chunk | Kmax coarse tail]``;
* after the reverse exchange, apply the owner's local BF16 gate from that tail
  directly into the fine view.

No collective is launched here.  Integration is explicitly opt-in and the
CUDA path accepts only an unmodified route plan produced by the validated CPU
builder.
"""

from __future__ import annotations

import os
import weakref

import torch
from vllm.triton_utils import HAS_TRITON, tl, triton

from vllm_omni.diffusion.attention.ops.minimax_h3_vsa_owner_route import (
    H3VSAOwnerRoutePlan,
    h3_vsa_owner_route_plan_is_trusted,
)

H3_VSA_O_BUNDLE_ENV = "VLLM_OMNI_FASTVIDEO_VSA_O_BUNDLE"
H3_VSA_O_BUNDLE_ACTIVE_KEY = "vsa_h3_o_bundle_active"
H3_VSA_O_BUNDLE_STATE_KEY = "vsa_h3_o_bundle_state"
_FEATURE_BLOCK = 1024

_CUDA_PLAN_MAPS: dict[
    tuple[int, int | None],
    tuple[
        weakref.ReferenceType[H3VSAOwnerRoutePlan],
        tuple[int, ...],
        torch.Tensor,
        torch.Tensor,
    ],
] = {}


def h3_vsa_o_bundle_enabled() -> bool:
    """Return whether reverse-O compact-coarse piggybacking was requested."""
    raw = os.environ.get(H3_VSA_O_BUNDLE_ENV, "0").strip()
    if raw not in {"0", "1"}:
        raise ValueError(f"{H3_VSA_O_BUNDLE_ENV} must be exactly '0' or '1', got {raw!r}")
    return raw == "1"


def _plan_versions(plan: H3VSAOwnerRoutePlan) -> tuple[int, ...]:
    return (
        plan.owner_tile_ids._version,
        plan.owner_tile_counts._version,
        plan.local_row_to_tail._version,
    )


def _cuda_plan_maps(
    plan: H3VSAOwnerRoutePlan,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Materialize trusted CPU maps once per plan and CUDA device."""
    if not h3_vsa_owner_route_plan_is_trusted(plan):
        raise ValueError("route plan must be an unmodified result of the validated owner-route builder")
    if device.type != "cuda":
        raise ValueError(f"CUDA route-map materialization requires a CUDA device, got {device}")

    key = (id(plan), device.index)
    versions = _plan_versions(plan)
    cached = _CUDA_PLAN_MAPS.get(key)
    if cached is not None:
        reference, cached_versions, owner_tile_ids, local_row_to_tail = cached
        if reference() is plan and cached_versions == versions:
            return owner_tile_ids, local_row_to_tail

    def cleanup(reference: weakref.ReferenceType[H3VSAOwnerRoutePlan]) -> None:
        current = _CUDA_PLAN_MAPS.get(key)
        if current is not None and current[0] is reference:
            _CUDA_PLAN_MAPS.pop(key, None)

    reference = weakref.ref(plan, cleanup)
    owner_tile_ids = plan.owner_tile_ids.to(device=device, non_blocking=False)
    local_row_to_tail = plan.local_row_to_tail.to(device=device, non_blocking=False)
    _CUDA_PLAN_MAPS[key] = (
        reference,
        versions,
        owner_tile_ids,
        local_row_to_tail,
    )
    return owner_tile_ids, local_row_to_tail


def _validate_plan(plan: H3VSAOwnerRoutePlan) -> None:
    if not h3_vsa_owner_route_plan_is_trusted(plan):
        raise ValueError("route plan must be an unmodified result of the validated owner-route builder")


def _shares_storage(left: torch.Tensor, right: torch.Tensor) -> bool:
    return left.untyped_storage().data_ptr() == right.untyped_storage().data_ptr()


def _validate_bundle_inputs(
    fine_aligned: torch.Tensor,
    coarse: torch.Tensor,
    plan: H3VSAOwnerRoutePlan,
) -> None:
    _validate_plan(plan)
    if fine_aligned.ndim != 4 or coarse.ndim != 4:
        raise ValueError("fine_aligned and coarse must be [B,S,H,D] tensors")
    if fine_aligned.device != coarse.device:
        raise ValueError("fine_aligned and coarse must share one device")
    if fine_aligned.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError(f"O bundling requires FP16 or BF16 activations, got {fine_aligned.dtype}")
    if coarse.dtype != fine_aligned.dtype:
        raise TypeError("fine_aligned and coarse must share one dtype")
    if fine_aligned.shape[0] != coarse.shape[0] or fine_aligned.shape[2:] != coarse.shape[2:]:
        raise ValueError(
            "fine_aligned and coarse must share batch/head geometry, got "
            f"fine={tuple(fine_aligned.shape)}, coarse={tuple(coarse.shape)}"
        )
    if fine_aligned.shape[1] != plan.sp_world_size * plan.local_rows:
        raise ValueError(
            "fine_aligned rows do not match the route geometry: "
            f"got {fine_aligned.shape[1]}, expected {plan.sp_world_size * plan.local_rows}"
        )
    if coarse.shape[1] != plan.logical_block_count:
        raise ValueError(
            f"coarse rows do not match logical_block_count: got {coarse.shape[1]}, expected {plan.logical_block_count}"
        )
    if not fine_aligned.shape[0] or not fine_aligned.shape[2] or not fine_aligned.shape[3]:
        raise ValueError("O bundling requires non-empty batch/head dimensions")
    if fine_aligned.requires_grad or coarse.requires_grad:
        raise ValueError("H3 VSA O bundling is inference-only")
    if not fine_aligned.is_contiguous():
        raise ValueError("fine_aligned must be contiguous")


def _bundle_shape(
    fine_aligned: torch.Tensor,
    plan: H3VSAOwnerRoutePlan,
) -> tuple[int, int, int, int]:
    return (
        fine_aligned.shape[0],
        plan.sp_world_size * (plan.local_rows + plan.kmax),
        fine_aligned.shape[2],
        fine_aligned.shape[3],
    )


def _validate_bundle_output(
    fine_aligned: torch.Tensor,
    coarse: torch.Tensor,
    plan: H3VSAOwnerRoutePlan,
    out: torch.Tensor,
) -> None:
    expected = _bundle_shape(fine_aligned, plan)
    if tuple(out.shape) != expected:
        raise ValueError(f"O bundle output shape mismatch: expected {expected}, got {tuple(out.shape)}")
    if out.device != fine_aligned.device or out.dtype != fine_aligned.dtype:
        raise ValueError("O bundle output must share fine_aligned's dtype and device")
    if not out.is_contiguous():
        raise ValueError("O bundle output must be contiguous")
    if out.requires_grad:
        raise ValueError("O bundle output must not require gradients")
    for name, tensor in (("fine_aligned", fine_aligned), ("coarse", coarse)):
        if _shares_storage(out, tensor):
            raise ValueError(f"O bundle output and {name} must not share storage")


def h3_vsa_o_bundle_reference(
    fine_aligned: torch.Tensor,
    coarse: torch.Tensor,
    plan: H3VSAOwnerRoutePlan,
    *,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Pure-Torch owner bundle, including exact inactive-tail zeros."""
    _validate_bundle_inputs(fine_aligned, coarse, plan)
    if out is None:
        output = torch.empty(
            _bundle_shape(fine_aligned, plan),
            dtype=fine_aligned.dtype,
            device=fine_aligned.device,
        )
    else:
        _validate_bundle_output(fine_aligned, coarse, plan, out)
        output = out
    output.zero_()

    owner_tile_ids = plan.owner_tile_ids.to(fine_aligned.device)
    output_by_owner = output.view(
        fine_aligned.shape[0],
        plan.sp_world_size,
        plan.local_rows + plan.kmax,
        fine_aligned.shape[2],
        fine_aligned.shape[3],
    )
    fine_by_owner = fine_aligned.reshape(
        fine_aligned.shape[0],
        plan.sp_world_size,
        plan.local_rows,
        fine_aligned.shape[2],
        fine_aligned.shape[3],
    )
    output_by_owner[:, :, : plan.local_rows].copy_(fine_by_owner)
    for owner in range(plan.sp_world_size):
        count = int(plan.owner_tile_counts[owner])
        if count:
            tile_ids = owner_tile_ids[owner, :count].to(torch.int64)
            output_by_owner[:, owner, plan.local_rows : plan.local_rows + count].copy_(coarse.index_select(1, tile_ids))
    return output


def h3_vsa_o_bundle_cuda_supported(
    fine_aligned: torch.Tensor,
    coarse: torch.Tensor,
    plan: H3VSAOwnerRoutePlan,
) -> bool:
    return (
        HAS_TRITON
        and h3_vsa_owner_route_plan_is_trusted(plan)
        and fine_aligned.is_cuda
        and coarse.device == fine_aligned.device
        and fine_aligned.ndim == 4
        and coarse.ndim == 4
        and fine_aligned.is_contiguous()
        and fine_aligned.dtype in (torch.float16, torch.bfloat16)
        and coarse.dtype == fine_aligned.dtype
        and fine_aligned.shape[0] > 0
        and fine_aligned.shape[1] == plan.sp_world_size * plan.local_rows
        and fine_aligned.shape[2] > 0
        and fine_aligned.shape[3] > 0
        and coarse.shape[0] == fine_aligned.shape[0]
        and coarse.shape[1] == plan.logical_block_count
        and coarse.shape[2:] == fine_aligned.shape[2:]
        and not fine_aligned.requires_grad
        and not coarse.requires_grad
    )


def _validate_local_gate_inputs(
    reversed_bundle: torch.Tensor,
    gate_local: torch.Tensor,
    plan: H3VSAOwnerRoutePlan,
    *,
    owner_rank: int,
) -> None:
    _validate_plan(plan)
    if isinstance(owner_rank, bool) or not isinstance(owner_rank, int):
        raise ValueError(f"owner_rank must be an integer, got {owner_rank!r}")
    if not 0 <= owner_rank < plan.sp_world_size:
        raise ValueError(f"owner_rank={owner_rank} is outside world size {plan.sp_world_size}")
    if reversed_bundle.ndim != 4 or gate_local.ndim != 4:
        raise ValueError("reversed_bundle and gate_local must be [B,S,H,D] tensors")
    expected_bundle_rows = plan.local_rows + plan.kmax
    if reversed_bundle.shape[1] != expected_bundle_rows:
        raise ValueError(f"reversed_bundle has {reversed_bundle.shape[1]} rows, expected {expected_bundle_rows}")
    expected_gate_shape = (
        reversed_bundle.shape[0],
        plan.local_rows,
        reversed_bundle.shape[2],
        reversed_bundle.shape[3],
    )
    if tuple(gate_local.shape) != expected_gate_shape:
        raise ValueError(f"gate_local shape mismatch: expected {expected_gate_shape}, got {tuple(gate_local.shape)}")
    if reversed_bundle.dtype != torch.bfloat16 or gate_local.dtype != torch.bfloat16:
        raise TypeError("local O-bundle gate-add requires matching BF16 tensors")
    if reversed_bundle.device != gate_local.device:
        raise ValueError("reversed_bundle and gate_local must share one device")
    if not reversed_bundle.is_contiguous():
        raise ValueError("reversed_bundle must be contiguous")
    if not gate_local.is_contiguous():
        raise ValueError("gate_local must be contiguous")
    if reversed_bundle.shape[-1] % 2:
        raise ValueError(f"BF16x2 exact arithmetic requires an even head dimension, got {reversed_bundle.shape[-1]}")
    if reversed_bundle.requires_grad or gate_local.requires_grad:
        raise ValueError("local O-bundle gate-add is inference-only")
    if _shares_storage(reversed_bundle, gate_local):
        raise ValueError("reversed_bundle and gate_local must not share storage")


def _fine_view(reversed_bundle: torch.Tensor, plan: H3VSAOwnerRoutePlan) -> torch.Tensor:
    fine = reversed_bundle[:, : plan.local_rows]
    expected_feature_strides = (
        fine.shape[2] * fine.shape[3],
        fine.shape[3],
        1,
    )
    if fine.stride()[1:] != expected_feature_strides:
        raise RuntimeError("local fine view must retain contiguous sequence/head/dim strides for the output projection")
    return fine


def h3_vsa_o_bundle_local_gate_reference(
    reversed_bundle: torch.Tensor,
    gate_local: torch.Tensor,
    plan: H3VSAOwnerRoutePlan,
    *,
    owner_rank: int,
) -> torch.Tensor:
    """Clone, apply eager BF16 gate boundaries, and return the fine view."""
    _validate_local_gate_inputs(
        reversed_bundle,
        gate_local,
        plan,
        owner_rank=owner_rank,
    )
    output = reversed_bundle.clone()
    fine = _fine_view(output, plan)
    slots = plan.local_row_to_tail[owner_rank]
    valid_rows = torch.nonzero(slots >= 0, as_tuple=False).flatten().to(output.device)
    if valid_rows.numel():
        tail_rows = slots.index_select(0, valid_rows.cpu()).to(output.device, dtype=torch.int64)
        coarse_rows = output.index_select(1, tail_rows + plan.local_rows)
        gated = coarse_rows * gate_local.index_select(1, valid_rows)
        fine[:, valid_rows] = fine.index_select(1, valid_rows) + gated
    return fine


def h3_vsa_o_bundle_local_gate_cuda_supported(
    reversed_bundle: torch.Tensor,
    gate_local: torch.Tensor,
    plan: H3VSAOwnerRoutePlan,
    *,
    owner_rank: int,
) -> bool:
    return (
        HAS_TRITON
        and h3_vsa_owner_route_plan_is_trusted(plan)
        and isinstance(owner_rank, int)
        and not isinstance(owner_rank, bool)
        and 0 <= owner_rank < plan.sp_world_size
        and reversed_bundle.is_cuda
        and reversed_bundle.ndim == 4
        and reversed_bundle.is_contiguous()
        and reversed_bundle.dtype == torch.bfloat16
        and reversed_bundle.shape[1] == plan.local_rows + plan.kmax
        and reversed_bundle.shape[0] > 0
        and reversed_bundle.shape[2] > 0
        and reversed_bundle.shape[3] > 0
        and reversed_bundle.shape[3] % 2 == 0
        and gate_local.device == reversed_bundle.device
        and gate_local.dtype == reversed_bundle.dtype
        and gate_local.is_contiguous()
        and tuple(gate_local.shape)
        == (
            reversed_bundle.shape[0],
            plan.local_rows,
            reversed_bundle.shape[2],
            reversed_bundle.shape[3],
        )
        and not reversed_bundle.requires_grad
        and not gate_local.requires_grad
        and not _shares_storage(reversed_bundle, gate_local)
    )


if HAS_TRITON:

    @triton.jit
    def _h3_vsa_o_bundle_kernel(
        fine_ptr,
        coarse_ptr,
        owner_tile_ids_ptr,
        output_ptr,
        fine_stride_batch,
        fine_stride_seq,
        fine_stride_head,
        fine_stride_dim,
        coarse_stride_batch,
        coarse_stride_seq,
        coarse_stride_head,
        coarse_stride_dim,
        owners: tl.constexpr,
        local_rows: tl.constexpr,
        kmax: tl.constexpr,
        head_dim: tl.constexpr,
        row_width: tl.constexpr,
        feature_block: tl.constexpr,
    ):
        flat_output_row = tl.program_id(0)
        feature_block_index = tl.program_id(1)
        output_rows = owners * (local_rows + kmax)
        batch_index = flat_output_row // output_rows
        output_row = flat_output_row - batch_index * output_rows
        owner = output_row // (local_rows + kmax)
        owner_row = output_row - owner * (local_rows + kmax)
        is_fine = owner_row < local_rows
        tail_slot = owner_row - local_rows
        tile_id = tl.load(
            owner_tile_ids_ptr + owner * kmax + tail_slot,
            mask=~is_fine,
            other=-1,
        )

        features = feature_block_index * feature_block + tl.arange(0, feature_block)
        feature_mask = features < row_width
        head_index = features // head_dim
        dim_index = features - head_index * head_dim
        fine_source_row = owner * local_rows + owner_row
        fine_offsets = (
            batch_index * fine_stride_batch
            + fine_source_row * fine_stride_seq
            + head_index * fine_stride_head
            + dim_index * fine_stride_dim
        )
        coarse_offsets = (
            batch_index * coarse_stride_batch
            + tile_id * coarse_stride_seq
            + head_index * coarse_stride_head
            + dim_index * coarse_stride_dim
        )
        fine_values = tl.load(
            fine_ptr + fine_offsets,
            mask=is_fine & feature_mask,
            other=0.0,
        )
        coarse_values = tl.load(
            coarse_ptr + coarse_offsets,
            mask=(~is_fine) & (tile_id >= 0) & feature_mask,
            other=0.0,
        )
        values = tl.where(is_fine, fine_values, coarse_values)
        output_offsets = flat_output_row * row_width + features
        tl.store(output_ptr + output_offsets, values, mask=feature_mask)

    @triton.jit
    def _h3_vsa_o_bundle_local_gate_kernel(
        bundle_ptr,
        gate_ptr,
        local_row_to_tail_ptr,
        bundle_stride_batch,
        bundle_stride_seq,
        bundle_stride_head,
        bundle_stride_dim,
        gate_stride_batch,
        gate_stride_seq,
        gate_stride_head,
        gate_stride_dim,
        local_rows: tl.constexpr,
        head_dim: tl.constexpr,
        row_width: tl.constexpr,
        feature_block: tl.constexpr,
    ):
        flat_local_row = tl.program_id(0)
        feature_block_index = tl.program_id(1)
        batch_index = flat_local_row // local_rows
        local_row = flat_local_row - batch_index * local_rows
        tail_slot = tl.load(local_row_to_tail_ptr + local_row)
        valid_row = tail_slot >= 0

        features = feature_block_index * feature_block + tl.arange(0, feature_block)
        feature_mask = features < row_width
        head_index = features // head_dim
        dim_index = features - head_index * head_dim
        fine_offsets = (
            batch_index * bundle_stride_batch
            + local_row * bundle_stride_seq
            + head_index * bundle_stride_head
            + dim_index * bundle_stride_dim
        )
        coarse_offsets = (
            batch_index * bundle_stride_batch
            + (local_rows + tail_slot) * bundle_stride_seq
            + head_index * bundle_stride_head
            + dim_index * bundle_stride_dim
        )
        gate_offsets = (
            batch_index * gate_stride_batch
            + local_row * gate_stride_seq
            + head_index * gate_stride_head
            + dim_index * gate_stride_dim
        )
        mask = valid_row & feature_mask
        fine_values = tl.load(bundle_ptr + fine_offsets, mask=mask, other=0.0)
        coarse_values = tl.load(bundle_ptr + coarse_offsets, mask=mask, other=0.0)
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
        tl.store(bundle_ptr + fine_offsets, values, mask=mask)


def _h3_vsa_o_bundle_cuda_impl(
    fine_aligned: torch.Tensor,
    coarse: torch.Tensor,
    plan: H3VSAOwnerRoutePlan,
    *,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    _validate_bundle_inputs(fine_aligned, coarse, plan)
    if not h3_vsa_o_bundle_cuda_supported(fine_aligned, coarse, plan):
        raise RuntimeError("H3 VSA O bundling received unsupported CUDA inputs")
    if out is None:
        output = torch.empty(
            _bundle_shape(fine_aligned, plan),
            dtype=fine_aligned.dtype,
            device=fine_aligned.device,
        )
    else:
        _validate_bundle_output(fine_aligned, coarse, plan, out)
        output = out

    owner_tile_ids, _local_row_to_tail = _cuda_plan_maps(plan, fine_aligned.device)
    row_width = fine_aligned.shape[2] * fine_aligned.shape[3]
    output_rows = output.shape[1]
    _h3_vsa_o_bundle_kernel[
        (
            fine_aligned.shape[0] * output_rows,
            triton.cdiv(row_width, _FEATURE_BLOCK),
        )
    ](
        fine_aligned,
        coarse,
        owner_tile_ids,
        output,
        fine_aligned.stride(0),
        fine_aligned.stride(1),
        fine_aligned.stride(2),
        fine_aligned.stride(3),
        coarse.stride(0),
        coarse.stride(1),
        coarse.stride(2),
        coarse.stride(3),
        owners=plan.sp_world_size,
        local_rows=plan.local_rows,
        kmax=plan.kmax,
        head_dim=fine_aligned.shape[3],
        row_width=row_width,
        feature_block=_FEATURE_BLOCK,
        num_warps=8,
    )
    return output


@torch.compiler.disable
def h3_vsa_o_bundle(
    fine_aligned: torch.Tensor,
    coarse: torch.Tensor,
    plan: H3VSAOwnerRoutePlan,
) -> torch.Tensor:
    """Bundle aligned fine rows and compact coarse tails on CUDA."""
    return _h3_vsa_o_bundle_cuda_impl(fine_aligned, coarse, plan)


@torch.compiler.disable
def h3_vsa_o_bundle_out(
    fine_aligned: torch.Tensor,
    coarse: torch.Tensor,
    plan: H3VSAOwnerRoutePlan,
    *,
    out: torch.Tensor,
) -> torch.Tensor:
    """Bundle directly into caller-owned reverse-O send storage."""
    return _h3_vsa_o_bundle_cuda_impl(fine_aligned, coarse, plan, out=out)


@torch.compiler.disable
def h3_vsa_o_bundle_local_gate_add_(
    reversed_bundle: torch.Tensor,
    gate_local: torch.Tensor,
    plan: H3VSAOwnerRoutePlan,
    *,
    owner_rank: int,
) -> torch.Tensor:
    """Apply exact BF16 local gating in place and return the fine view."""
    _validate_local_gate_inputs(
        reversed_bundle,
        gate_local,
        plan,
        owner_rank=owner_rank,
    )
    if not h3_vsa_o_bundle_local_gate_cuda_supported(
        reversed_bundle,
        gate_local,
        plan,
        owner_rank=owner_rank,
    ):
        raise RuntimeError("local H3 VSA O-bundle gate-add received unsupported CUDA inputs")

    _owner_tile_ids, local_row_to_tail = _cuda_plan_maps(plan, reversed_bundle.device)
    owner_map = local_row_to_tail[owner_rank]
    row_width = reversed_bundle.shape[2] * reversed_bundle.shape[3]
    _h3_vsa_o_bundle_local_gate_kernel[
        (
            reversed_bundle.shape[0] * plan.local_rows,
            triton.cdiv(row_width, _FEATURE_BLOCK),
        )
    ](
        reversed_bundle,
        gate_local,
        owner_map,
        reversed_bundle.stride(0),
        reversed_bundle.stride(1),
        reversed_bundle.stride(2),
        reversed_bundle.stride(3),
        gate_local.stride(0),
        gate_local.stride(1),
        gate_local.stride(2),
        gate_local.stride(3),
        local_rows=plan.local_rows,
        head_dim=reversed_bundle.shape[3],
        row_width=row_width,
        feature_block=_FEATURE_BLOCK,
        num_warps=8,
    )
    return _fine_view(reversed_bundle, plan)


__all__ = [
    "H3_VSA_O_BUNDLE_ACTIVE_KEY",
    "H3_VSA_O_BUNDLE_ENV",
    "H3_VSA_O_BUNDLE_STATE_KEY",
    "h3_vsa_o_bundle",
    "h3_vsa_o_bundle_cuda_supported",
    "h3_vsa_o_bundle_enabled",
    "h3_vsa_o_bundle_local_gate_add_",
    "h3_vsa_o_bundle_local_gate_cuda_supported",
    "h3_vsa_o_bundle_local_gate_reference",
    "h3_vsa_o_bundle_out",
    "h3_vsa_o_bundle_reference",
]
