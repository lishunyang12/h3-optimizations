"""The existing v7 BF16 tiled producer, scoped to the v16 fine pipeline."""

import torch
import triton
import triton.language as tl

from vllm_omni.diffusion.attention.ops.minimax_h3_vsa_layout import (
    _source_rows_are_registered,
)


def h3_vsa_cake_output_view_supported(x):
    return (tuple(x.shape) == (1, 105472, 7, 128)
            and tuple(x.stride()) == (94502912, 128, 13500416, 1)
            and x.dtype == torch.bfloat16 and not x.requires_grad)


def fake_cake_output_view(query: torch.Tensor) -> torch.Tensor:
    if tuple(query.shape) != (1, 105472, 7, 128) or query.dtype != torch.bfloat16:
        raise ValueError("Strided Cake output requires [1,105472,7,128] BF16")
    # Own a fresh BHSD buffer; the result does not alias any op argument.
    return torch.empty((1, 7, 105472, 128), dtype=query.dtype, device=query.device).transpose(1, 2)


def validate_tiled_source(fine: torch.Tensor, source_rows: torch.Tensor) -> None:
    if not fine.is_cuda or not h3_vsa_cake_output_view_supported(fine):
        raise ValueError("Tiled producer requires the qualified Cake BSHD view")
    if not (
        source_rows.device == fine.device
        and tuple(source_rows.shape) == (95936,)
        and source_rows.dtype == torch.int32
        and source_rows.is_contiguous()
        and _source_rows_are_registered(
            source_rows, kind="aligned_untile", source_bound=105472, valid_rows=95924,
        )
    ):
        raise ValueError("Tiled producer requires an unmodified trusted aligned inverse map")


@triton.jit
def _produce_tiled(
    fine, coarse, owners, source_rows, owner_row_order, out,
    fine_seq_stride, fine_head_stride, coarse_row_stride, coarse_head_stride,
    ROWS: tl.constexpr, OFFSET: tl.constexpr, COUNT: tl.constexpr,
    TAIL: tl.constexpr, WIDTH: tl.constexpr, BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    owner = row // (COUNT + TAIL)
    within = row % (COUNT + TAIL)
    col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    local_row = tl.load(owner_row_order + owner * ROWS + OFFSET + within, within < COUNT, other=0)
    aligned_row = owner * ROWS + local_row
    source_row = tl.load(source_rows + aligned_row, within < COUNT, other=-1)
    fine_offset = source_row * fine_seq_stride + (col // 128) * fine_head_stride + col % 128
    fine_value = tl.load(
        fine + fine_offset,
        (within < COUNT) & (source_row >= 0) & (col < WIDTH), other=0,
    )
    tile = tl.load(owners + owner * TAIL + (within - COUNT), within >= COUNT, other=-1)
    coarse_offset = tile * coarse_row_stride + (col // 128) * coarse_head_stride + col % 128
    coarse_value = tl.load(
        coarse + coarse_offset,
        (within >= COUNT) & (tile >= 0) & (col < WIDTH), other=0,
    )
    tl.store(out + row * WIDTH + col, tl.where(within < COUNT, fine_value, coarse_value), col < WIDTH)


def produce_tiled(fine, coarse, owner_tiles, landing, start, count, source_rows):
    validate_tiled_source(fine, source_rows)
    from vllm_omni.diffusion.attention.ops.h3_fine_row_order import row_order

    order = row_order(fine.device)
    if not (isinstance(start, int) and count == 2998 and start in (0, 2998, 5996, 8994)):
        raise ValueError("Tiled producer requires one of the four fixed chunks")
    if not (
        tuple(coarse.shape) == (1, 1648, 7, 128)
        and coarse.stride(-1) == 1 and all(s > 0 for s in coarse.stride())
        and coarse.dtype == fine.dtype and coarse.device == fine.device
        and not coarse.requires_grad
        and tuple(owner_tiles.shape) == (8, 270) and owner_tiles.dtype == torch.int32
        and owner_tiles.is_contiguous() and owner_tiles.device == fine.device
        and tuple(landing.shape) == (1, 8 * (count + 270), 7, 128)
        and landing.is_contiguous() and landing.dtype == fine.dtype
        and landing.device == fine.device and not landing.requires_grad
    ):
        raise ValueError("Tiled producer coarse/owner/landing contract mismatch")
    if any(landing.untyped_storage().data_ptr() == x.untyped_storage().data_ptr()
           for x in (fine, coarse, owner_tiles, source_rows)):
        raise ValueError("Tiled producer landing must not alias an input")
    _produce_tiled[(8 * (count + 270), 1)](
        fine, coarse, owner_tiles, source_rows, order, landing,
        fine.stride(1), fine.stride(2), coarse.stride(1), coarse.stride(2),
        ROWS=11992, OFFSET=start, COUNT=count, TAIL=270, WIDTH=896, BLOCK=1024, num_warps=8,
    )

