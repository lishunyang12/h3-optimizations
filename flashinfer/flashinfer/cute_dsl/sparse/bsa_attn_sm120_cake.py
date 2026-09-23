# Copyright (c) 2026 by FlashInfer team. SPDX-License-Identifier: Apache-2.0
# Isolated pinned PR4951 API; original PR4691 entry remains available.
import math
from typing import Optional
import torch
_BLOCK_SIZE = 64
def _ceil_div(a, b): return (a + b - 1) // b

def _bsa_attn_sm120_blk64_sage_fwd_cake(
    q_int8: torch.Tensor,
    k_int8: torch.Tensor,
    v_fp8: torch.Tensor,
    q_scale: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    q2k_block_index: torch.Tensor,
    block_sparse_num: int,
    block_sizes: Optional[torch.Tensor] = None,
    q2k_block_nums: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
    *,
    out: torch.Tensor,
    tma_descriptor_workspace: torch.Tensor,
    uniform_block_count: bool = False,
    contiguous_block_indices: bool = False,
    backend: str = "cake",
) -> torch.Tensor:
    """Run prequantized SM120 Sage block-sparse attention.

    All CUDA storage is caller-owned. ``out`` is contiguous BF16 BHSD and
    ``tma_descriptor_workspace`` is contiguous, 128-byte-aligned CUDA uint8
    storage. Q/K use contiguous INT8 BHSD, V uses contiguous FP8 E4M3 HDS,
    and the operation supports MHA, head dimension 128, non-causal forward
    without LSE.
    """
    import tvm_ffi  # noqa: PLC0415

    from flashinfer.jit.cake_sage_block_sparse_attention import (  # noqa: PLC0415
        load_cake_sage_block_sparse_attention_module,
    )

    if backend != "cake":
        raise ValueError(f"unsupported SM120 Sage backend: {backend!r}")
    if (
        type(uniform_block_count) is not bool
        or type(contiguous_block_indices) is not bool
    ):
        raise TypeError("uniform_block_count and contiguous_block_indices must be bool")
    if contiguous_block_indices and not uniform_block_count:
        raise ValueError("contiguous_block_indices requires uniform_block_count")
    if not isinstance(block_sparse_num, int) or isinstance(block_sparse_num, bool):
        raise TypeError("block_sparse_num must be an integer")

    tensors = {
        "q_int8": q_int8,
        "k_int8": k_int8,
        "v_fp8": v_fp8,
        "q_scale": q_scale,
        "k_scale": k_scale,
        "v_scale": v_scale,
        "q2k_block_index": q2k_block_index,
        "out": out,
        "tma_descriptor_workspace": tma_descriptor_workspace,
    }
    if not all(isinstance(value, torch.Tensor) for value in tensors.values()):
        raise TypeError("all input, output, and workspace values must be torch.Tensor")
    device = q_int8.device
    if device.type != "cuda" or device.index != torch.cuda.current_device():
        raise ValueError("q_int8 must use the current CUDA device")
    for name, tensor in tensors.items():
        if tensor.device != device or not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous on {device}")

    capability = tuple(int(value) for value in torch.cuda.get_device_capability(device))
    if capability != (12, 0):
        raise RuntimeError(
            f"SM120 Sage attention requires compute capability 12.0, got {capability}"
        )
    if q_int8.dtype != torch.int8 or k_int8.dtype != torch.int8:
        raise TypeError("q_int8 and k_int8 must use torch.int8")
    if v_fp8.dtype != torch.float8_e4m3fn:
        raise TypeError("v_fp8 must use torch.float8_e4m3fn")
    if out.dtype != torch.bfloat16:
        raise TypeError("out must use torch.bfloat16")
    for name, tensor in (
        ("q_scale", q_scale),
        ("k_scale", k_scale),
        ("v_scale", v_scale),
    ):
        if tensor.dtype != torch.float32:
            raise TypeError(f"{name} must use torch.float32")
    if tma_descriptor_workspace.dtype != torch.uint8:
        raise TypeError("tma_descriptor_workspace must use torch.uint8")
    if tma_descriptor_workspace.data_ptr() % 128:
        raise ValueError("tma_descriptor_workspace must be 128-byte aligned")

    if q_int8.ndim != 4:
        raise ValueError("q_int8 must be rank-4 BHSD")
    batch, heads, seqlen_q, head_dim = (int(value) for value in q_int8.shape)
    if batch < 1 or heads < 1 or seqlen_q < 1 or head_dim != 128:
        raise ValueError("q_int8 must have positive B/H/Sq and head dimension 128")
    if (
        k_int8.ndim != 4
        or tuple(k_int8.shape[:2]) != (batch, heads)
        or int(k_int8.shape[3]) != 128
    ):
        raise ValueError("k_int8 must be BHSD MHA with head dimension 128")
    seqlen_k = int(k_int8.shape[2])
    if seqlen_k < 1:
        raise ValueError("seqlen_k must be positive")
    q_blocks = _ceil_div(seqlen_q, _BLOCK_SIZE)
    k_blocks = _ceil_div(seqlen_k, _BLOCK_SIZE)
    padded_k = k_blocks * _BLOCK_SIZE
    if tuple(v_fp8.shape) != (batch, heads, 128, padded_k):
        raise ValueError("v_fp8 must use contiguous HDS [B,H,128,padded_Sk]")
    if tuple(out.shape) != (batch, heads, seqlen_q, 128):
        raise ValueError("out must be contiguous BF16 BHSD [B,H,Sq,128]")
    if tuple(q_scale.shape) != (batch, heads, _ceil_div(seqlen_q, 128) * 4):
        raise ValueError("q_scale has an invalid shape")
    if tuple(k_scale.shape) != (batch, heads, k_blocks):
        raise ValueError("k_scale has an invalid shape")
    if tuple(v_scale.shape) != (batch, heads, 128):
        raise ValueError("v_scale has an invalid shape")
    if q2k_block_index.ndim != 4 or tuple(q2k_block_index.shape[:3]) != (
        batch,
        heads,
        q_blocks,
    ):
        raise ValueError("q2k_block_index must be [B,H,ceil(Sq/64),capacity]")
    if q2k_block_index.dtype != torch.int32 or int(q2k_block_index.shape[-1]) < 1:
        raise ValueError("q2k_block_index must be int32 with positive capacity")
    q2k_capacity = int(q2k_block_index.shape[-1])
    if not 0 <= block_sparse_num <= q2k_capacity:
        raise ValueError("block_sparse_num must be in [0, q2k capacity]")

    if q2k_block_nums is not None:
        if (
            not isinstance(q2k_block_nums, torch.Tensor)
            or q2k_block_nums.device != device
            or not q2k_block_nums.is_contiguous()
            or q2k_block_nums.dtype != torch.int32
            or tuple(q2k_block_nums.shape) != (batch, heads, q_blocks)
        ):
            raise ValueError("q2k_block_nums must be contiguous int32 [B,H,Q64]")
    q2k_nums_arg = q2k_block_index if q2k_block_nums is None else q2k_block_nums
    has_block_nums = int(q2k_block_nums is not None and not uniform_block_count)

    block_sizes_mode = 0
    if block_sizes is not None:
        if not isinstance(block_sizes, torch.Tensor) or block_sizes.device != device:
            raise ValueError("block_sizes must be a CUDA tensor on the input device")
        if block_sizes.dtype != torch.int32 or not block_sizes.is_contiguous():
            raise ValueError("block_sizes must be contiguous int32")
        expected = {
            1: (k_blocks,),
            2: (batch, k_blocks),
            3: (batch, heads, k_blocks),
        }.get(block_sizes.ndim)
        if expected is None or tuple(block_sizes.shape) != expected:
            raise ValueError("block_sizes must have the canonical 1D, 2D, or 3D shape")
        block_sizes_mode = int(block_sizes.ndim)
    block_sizes_arg = q2k_block_index if block_sizes is None else block_sizes

    if softmax_scale is None:
        softmax_scale = head_dim**-0.5
    if isinstance(softmax_scale, bool) or not isinstance(softmax_scale, (int, float)):
        raise TypeError("softmax_scale must be a number")
    if not math.isfinite(float(softmax_scale)) or float(softmax_scale) <= 0.0:
        raise ValueError("softmax_scale must be finite and positive")

    full_k64_tiles = int(block_sizes_mode == 0 and seqlen_k % _BLOCK_SIZE == 0)
    uniform_nonempty = int(uniform_block_count and block_sparse_num > 0)
    module, record = load_cake_sage_block_sparse_attention_module(
        has_block_nums,
        block_sizes_mode,
        full_k64_tiles,
        uniform_nonempty,
        int(contiguous_block_indices),
    )
    required_workspace = int(record["tma_workspace_bytes"])
    if tma_descriptor_workspace.numel() < required_workspace:
        raise ValueError(
            f"tma_descriptor_workspace requires at least {required_workspace} bytes"
        )
    bindings = {
        "Q_map": q_int8,
        "K_map": k_int8,
        "V_map": v_fp8.view(torch.uint8),
        "O_map": out,
        "q_scale": q_scale,
        "k_scale": k_scale,
        "v_scale": v_scale,
        "q2k_block_index": q2k_block_index,
        "q2k_block_nums": q2k_nums_arg,
        "block_sizes": block_sizes_arg,
        "seqlen_q": seqlen_q,
        "seqlen_k": seqlen_k,
        "num_heads": heads,
        "q2k_capacity": q2k_capacity,
        "num_k_blocks": k_blocks,
        "block_sparse_num": block_sparse_num,
        "softmax_scale": float(softmax_scale),
        "tma_descriptor_workspace": tma_descriptor_workspace,
        "grid_x": q_blocks,
        "grid_y": heads,
        "grid_z": batch,
    }
    try:
        args = [bindings[name] for _kind, name in record["arg_plan"]]
    except KeyError as exc:
        raise RuntimeError(
            f"generated argument plan names an unknown binding: {exc}"
        ) from exc
    with tvm_ffi.use_torch_stream():
        getattr(module, str(record["ffi_entry"]))(*args)
    return out

def bsa_attn_sm120_blk64_sage_fwd(
    q_int8: torch.Tensor,
    k_int8: torch.Tensor,
    v_fp8: torch.Tensor,
    q_scale: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    q2k_block_index: torch.Tensor,
    block_sparse_num: int,
    block_sizes: Optional[torch.Tensor] = None,
    q2k_block_nums: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
    *,
    out: torch.Tensor,
    tma_descriptor_workspace: torch.Tensor,
    uniform_block_count: bool = False,
    contiguous_block_indices: bool = False,
    backend: str = "cake",
) -> torch.Tensor:
    """Run prequantized SM120 Sage block-sparse attention.

    All CUDA storage is caller-owned. ``out`` is contiguous BF16 BHSD and
    ``tma_descriptor_workspace`` is contiguous, 128-byte-aligned CUDA uint8
    storage. Q/K use contiguous INT8 BHSD, V uses contiguous FP8 E4M3 HDS,
    and the operation supports MHA, head dimension 128, non-causal forward
    without LSE.
    """
    if backend != "cake":
        raise ValueError(f"unsupported SM120 Sage backend: {backend!r}")
    return _bsa_attn_sm120_blk64_sage_fwd_cake(
        q_int8,
        k_int8,
        v_fp8,
        q_scale,
        k_scale,
        v_scale,
        q2k_block_index,
        block_sparse_num,
        block_sizes,
        q2k_block_nums,
        softmax_scale,
        out=out,
        tma_descriptor_workspace=tma_descriptor_workspace,
        uniform_block_count=uniform_block_count,
        contiguous_block_indices=contiguous_block_indices,
        backend="cake",
    )
