"""Fixed H3 geometry: copy raw Sage BF16 bits directly into four O landings."""
import os
import torch
import triton
import triton.language as tl

FLAG = 'VLLM_OMNI_H3_TILED_O_PRODUCER'


def enabled():
    return os.environ.get(FLAG, '0') == '1'


def fake_output(query):
    assert query.shape == (1, 105408, 7, 128) and query.dtype == torch.bfloat16
    return torch.empty((1, 7, 105408, 128), device=query.device, dtype=query.dtype).transpose(1, 2)


def validate_source(fine, source_rows):
    from .minimax_h3_vsa_layout import _source_rows_are_registered
    assert fine.is_cuda and fine.dtype == torch.bfloat16 and not fine.requires_grad
    assert fine.shape == (1, 105408, 7, 128)
    assert fine.stride() == (105408*896, 128, 105408*128, 1)
    assert source_rows.shape == (95872,) and source_rows.dtype == torch.int32
    assert source_rows.device == fine.device and source_rows.is_contiguous()
    assert _source_rows_are_registered(source_rows, kind='aligned_untile',
                                       source_bound=105408, valid_rows=95823)


@triton.jit
def _produce(fine, coarse, owners, source_rows, out,
             FINE_ROW: tl.constexpr, FINE_HEAD: tl.constexpr,
             COARSE_ROW: tl.constexpr, COARSE_HEAD: tl.constexpr,
             OFFSET: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    owner = row // 3275
    within = row % 3275
    col = tl.arange(0, BLOCK)
    aligned_row = owner * 11984 + OFFSET + within
    source_row = tl.load(source_rows + aligned_row, within < 2996, other=-1)
    fine_offset = source_row * FINE_ROW + (col // 128) * FINE_HEAD + col % 128
    fine_bits = tl.load(fine + fine_offset,
                        (within < 2996) & (source_row >= 0) & (col < 896), other=0)
    tile = tl.load(owners + owner * 279 + within - 2996, within >= 2996, other=-1)
    coarse_offset = tile * COARSE_ROW + (col // 128) * COARSE_HEAD + col % 128
    coarse_bits = tl.load(coarse + coarse_offset,
                          (within >= 2996) & (tile >= 0) & (col < 896), other=0)
    tl.store(out + row * 896 + col, tl.where(within < 2996, fine_bits, coarse_bits), col < 896)


def produce(fine, coarse, owner_tiles, landing, start, count, source_rows):
    validate_source(fine, source_rows)
    assert count == 2996 and start in (0, 2996, 5992, 8988)
    assert coarse.shape == (1, 1647, 7, 128) and coarse.stride(-1) == 1
    assert coarse.dtype == fine.dtype and coarse.device == fine.device and not coarse.requires_grad
    assert owner_tiles.shape == (8, 279) and owner_tiles.dtype == torch.int32
    assert owner_tiles.is_contiguous() and owner_tiles.device == fine.device
    assert landing.shape == (1, 26200, 7, 128) and landing.is_contiguous()
    assert landing.device == fine.device and landing.dtype == fine.dtype
    assert not landing.requires_grad
    assert all(landing.untyped_storage().data_ptr() != x.untyped_storage().data_ptr()
               for x in (fine, coarse, owner_tiles, source_rows))
    # Integer views preserve every BF16 bit, including signed zero and NaN payloads.
    _produce[(26200,)](fine.view(torch.int16), coarse.view(torch.int16), owner_tiles,
        source_rows, landing.view(torch.int16), fine.stride(1), fine.stride(2),
        coarse.stride(1), coarse.stride(2), start, 1024, num_warps=8)
