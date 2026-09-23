"""Experimental FC2 producer matching the installed vLLM BF16 SwiGLU.

CPU-prepared candidate, not numerically admitted until probe_fusion.py passes on
the installed CUDA operator. Keeps the original MXFP8 payload/scales/layout.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

from .mxfp8_kernels import _alloc, _mx_e8m0_from_amax, _mx_scale_offsets


@triton.jit
def _bf16_round(x):
    # Explicit conversion pair preserves the CUDA kernel's two BF16 roundings.
    # Unlike the integer rounding shortcut this also preserves NaN conversion.
    return tl.inline_asm_elementwise(
        "{ .reg .b16 h; cvt.rn.bf16.f32 h, $1; cvt.f32.bf16 $0, h; }",
        constraints="=f,f", args=[x], dtype=tl.float32, is_pure=True, pack=1,
    )


@triton.jit
def _cuda_silu_product(gate, up):
    # The installed vLLM SM120 SASS uses expf range reduction and corrected
    # division gate / (1 + expf(-gate)), then BF16 round, FMUL, BF16 round.
    # tl.sigmoid uses a different approximation: never assume byte identity.
    act = _bf16_round(libdevice.div_rn(gate, 1.0 + libdevice.exp(-gate)))
    # Current generic CUDA act_and_mul keeps its (up + beta), beta=+0, add.
    # Preserve its signed-zero behavior too; an ordinary +0 can be optimized out.
    up_zero = tl.inline_asm_elementwise(
        "add.rn.f32 $0, $1, 0f00000000;", constraints="=f,f", args=[up],
        dtype=tl.float32, is_pure=True, pack=1,
    )
    return _bf16_round(libdevice.mul_rn(act, up_zero))


@triton.jit
def _silu_mxfp8_cuda_kernel(
    X, Q, S, Y, rows, hidden, groups, col_blocks, stride_row,
    STORE_BF16: tl.constexpr, BLOCK_R: tl.constexpr, G: tl.constexpr,
):
    r = tl.program_id(0) * BLOCK_R + tl.arange(0, BLOCK_R)
    g = tl.program_id(1) * G + tl.arange(0, G)
    c = tl.program_id(1) * (G * 32) + tl.arange(0, G * 32)
    mask = (r < rows)[:, None] & (c < hidden)[None, :]
    base = X + r[:, None].to(tl.int64) * stride_row
    gate = tl.load(base + c[None, :], mask, other=0).to(tl.float32)
    up = tl.load(base + hidden + c[None, :], mask, other=0).to(tl.float32)
    prod = _cuda_silu_product(gate, up)
    if STORE_BF16:
        tl.store(Y + r[:, None].to(tl.int64) * hidden + c[None, :], prod, mask)
    p3 = tl.reshape(prod, [BLOCK_R, G, 32])
    amax = tl.max(tl.abs(p3), axis=2)
    scale_byte, inv = _mx_e8m0_from_amax(amax)
    quant = tl.reshape(p3 * inv[:, :, None], [BLOCK_R, G * 32])
    tl.store(Q + r[:, None].to(tl.int64) * hidden + c[None, :],
             quant.to(tl.float8e4nv), mask)
    tl.store(S + _mx_scale_offsets(r[:, None], g[None, :], col_blocks),
             scale_byte.to(tl.uint8), (r < rows)[:, None] & (g < groups)[None, :])


def silu_mul_mxfp8_cuda(x: torch.Tensor, *, return_bf16: bool = False):
    """BF16 [M, 2K] gate|up -> E4M3 [M,K], padded swizzled E8M0 bytes.

    return_bf16 is a probe-only diagnostic and must remain False for timing.
    No fallback to eager SwiGLU or standalone quantization is permitted.
    """
    if (not x.is_cuda or x.dtype != torch.bfloat16 or x.ndim != 2
            or x.stride(1) != 1 or x.shape[1] % 64
            or torch.compiler.is_compiling()):
        raise ValueError("Expected resident row-major BF16 [M,2K], K multiple of 32, outside compile")
    rows, twice = x.shape
    hidden = twice // 2
    q, s = _alloc(rows, hidden, x.device)
    y = torch.empty((rows, hidden), dtype=torch.bfloat16, device=x.device) if return_bf16 else q
    if rows:
        groups = hidden // 32
        with torch.get_device_module().device(x.device):
            _silu_mxfp8_cuda_kernel[(triton.cdiv(rows, 16), triton.cdiv(groups, 8))](
                x, q, s, y, rows, hidden, groups, triton.cdiv(groups, 4), x.stride(0),
                STORE_BF16=return_bf16, BLOCK_R=16, G=8, num_warps=4,
                enable_fp_fusion=True,
                # vLLM's installed CUDA SASS uses non-FTZ exp/div arithmetic.
                # Triton's libdevice reflection otherwise defaults to FTZ.
                enable_reflect_ftz=False,
            )
    return (q, s, y) if return_bf16 else (q, s)
