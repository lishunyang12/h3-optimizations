# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import functools
import importlib.util
import math
import os
from collections.abc import Mapping
from contextlib import contextmanager
from typing import Any

import torch
from vllm.logger import init_logger

from vllm_omni.diffusion.attention.backends.abstract import (
    AttentionBackend,
    AttentionImpl,
    AttentionMetadata,
)
from vllm_omni.diffusion.attention.backends.sdpa import SDPAImpl
from vllm_omni.diffusion.attention.ops.minimax_h3_vsa_deferred_gate import (
    H3_VSA_DEFERRED_ACTIVE_KEY,
    H3_VSA_DEFERRED_GATE_SP_ENV,
    H3_VSA_DEFERRED_STATE_KEY,
    build_h3_compact_row_blocks,
    h3_vsa_deferred_gate_sp_enabled,
)
from vllm_omni.diffusion.attention.ops.minimax_h3_vsa_layout import (
    H3_VSA_FP8_QKV_SCALES_KEY,
    H3_VSA_FUSED_GATE_UNTILE_ENV,
    H3_VSA_FUSED_TILE_PACK_ENV,
    H3_VSA_FUSED_UNTILE_ENV,
    build_h3_aligned_untile_source_rows,
    build_h3_tiled_source_rows,
    h3_vsa_fp8_dequant_tile_pack,
    h3_vsa_fp8_dequant_tile_pack_cuda_supported,
    h3_vsa_fused_gate_untile_enabled,
    h3_vsa_fused_tile_pack_enabled,
    h3_vsa_fused_untile_enabled,
    h3_vsa_gate_untile,
    h3_vsa_gate_untile_cuda_supported,
    h3_vsa_gate_untile_out,
    h3_vsa_tile_pack,
    h3_vsa_tile_pack_cuda_supported,
    h3_vsa_tile_untile,
    h3_vsa_tile_untile_cuda_supported,
    h3_vsa_tile_untile_out,
)
from vllm_omni.diffusion.attention.ops.minimax_h3_vsa_o_bundle import (
    H3_VSA_O_BUNDLE_ACTIVE_KEY,
    H3_VSA_O_BUNDLE_ENV,
    H3_VSA_O_BUNDLE_STATE_KEY,
    h3_vsa_o_bundle_cuda_supported,
    h3_vsa_o_bundle_enabled,
    h3_vsa_o_bundle_out,
)
from vllm_omni.diffusion.attention.ops.minimax_h3_vsa_owner_route import (
    H3VSAOwnerRoutePlan,
    build_h3_vsa_owner_route_plan,
    h3_vsa_owner_route_plan_is_trusted,
)
from vllm_omni.diffusion.forward_context import get_forward_context, is_forward_context_available
from vllm_omni.platforms import current_omni_platform

logger = init_logger(__name__)

H3_VSA_DIRECT_Q2K_ENV = "VLLM_OMNI_FASTVIDEO_VSA_DIRECT_Q2K"
H3_VSA_SKIP_SOFTMAX_THRESHOLD_ENV = (
    "VLLM_OMNI_FASTVIDEO_VSA_SKIP_SOFTMAX_THRESHOLD_SCALE_FACTOR"
)
H3_VSA_NVTX_ENV = "VLLM_OMNI_MINIMAX_H3_VSA_NVTX"
H3_VSA_NVTX_DOMAIN = "vllm_omni.minimax_h3.vsa"

# MiniMax H3's tile-64 path operates on the post-patch DiT token lattice.
# Keep this separate from ``FastVideoVSAImpl.block_size``: that field belongs
# to the generic VSA path and defaults to the 256-token ``(4, 8, 8)`` tile.
# The H3 metadata builder and every H3 kernel route below consume this exact
# effective contract instead.
H3_VSA_EFFECTIVE_TILE_SHAPE = (4, 4, 4)
H3_VSA_EFFECTIVE_TILE_SIZE = math.prod(H3_VSA_EFFECTIVE_TILE_SHAPE)


def h3_vsa_direct_q2k_enabled() -> bool:
    return os.environ.get(H3_VSA_DIRECT_Q2K_ENV, "0") == "1"


@contextmanager
def _h3_vsa_nvtx_stage(name: str):
    """Emit nested VSA phase ranges only for an explicit diagnostic run."""

    if os.environ.get(H3_VSA_NVTX_ENV, "0") != "1":
        yield
        return
    # Use NVIDIA's package so Nsight receives registered strings.  The lazy
    # import keeps the production path dependency- and overhead-free.
    import nvtx

    nvtx.push_range(name, domain=H3_VSA_NVTX_DOMAIN)
    try:
        yield
    finally:
        nvtx.pop_range(domain=H3_VSA_NVTX_DOMAIN)


def h3_vsa_skip_softmax_threshold_scale_factor() -> float:
    """Return the opt-in SM120 VSA skip-softmax threshold.

    Zero is the exact/default kernel specialization.  A positive value enables
    an approximate early-skip policy in FlashInfer's SM120 CuTeDSL kernel, so
    malformed values must fail closed instead of silently selecting a numeric
    path that was not requested.
    """

    raw = os.environ.get(H3_VSA_SKIP_SOFTMAX_THRESHOLD_ENV, "0").strip()
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(
            f"{H3_VSA_SKIP_SOFTMAX_THRESHOLD_ENV} must be a finite non-negative float, got {raw!r}"
        ) from exc
    if not math.isfinite(value) or value < 0:
        raise ValueError(
            f"{H3_VSA_SKIP_SOFTMAX_THRESHOLD_ENV} must be a finite non-negative float, got {raw!r}"
        )
    return value


# Keep the external FastVideo pybind/CUDA kernel opaque to torch.compile.
# This mirrors the SageAttention3 backend pattern: tracing the raw extension
# through Dynamo can reach Inductor scheduling with unstable internal op names
# (e.g. KeyError: "op12").  The custom op gives Dynamo a single Tensor->Tensor
# boundary and lets Inductor schedule the surrounding Wan block normally.
if not hasattr(torch.ops.vllm_omni, "fastvideo_vsa_bshd"):

    @torch.library.custom_op("vllm_omni::fastvideo_vsa_bshd", mutates_args=())
    def _fastvideo_vsa_bshd_op(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        variable_block_sizes: torch.Tensor,
        q_variable_block_sizes: torch.Tensor,
        compress_attn_weight: torch.Tensor,
        topk: int,
        block_t: int,
        block_h: int,
        block_w: int,
    ) -> torch.Tensor:
        from fastvideo_kernel import video_sparse_attn_bshd

        return video_sparse_attn_bshd(
            query,
            key,
            value,
            variable_block_sizes=variable_block_sizes,
            q_variable_block_sizes=q_variable_block_sizes,
            topk=topk,
            block_size=(block_t, block_h, block_w),
            compress_attn_weight=compress_attn_weight if compress_attn_weight.numel() else None,
        )

    @_fastvideo_vsa_bshd_op.register_fake
    def _(
        query,
        key,
        value,
        variable_block_sizes,
        q_variable_block_sizes,
        compress_attn_weight,
        topk,
        block_t,
        block_h,
        block_w,
    ):
        del (
            key,
            value,
            variable_block_sizes,
            q_variable_block_sizes,
            compress_attn_weight,
            topk,
            block_t,
            block_h,
            block_w,
        )
        return torch.empty_like(query)


_fastvideo_vsa_bshd_op = torch.ops.vllm_omni.fastvideo_vsa_bshd


# H3 needs an explicit per-query block map: prefix queries are dense, while
# video queries select prefix + top-k video tiles. The generic
# video_sparse_attn() entry point cannot express that contract.
if not hasattr(torch.ops.vllm_omni, "fastvideo_h3_vsa_bhsd"):

    @torch.library.custom_op("vllm_omni::fastvideo_h3_vsa_bhsd", mutates_args=())
    def _fastvideo_h3_vsa_bhsd_op(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        block_map: torch.Tensor,
        variable_block_sizes: torch.Tensor,
        logical_blocks: int,
    ) -> torch.Tensor:
        q = query.transpose(1, 2).contiguous()
        k = key.transpose(1, 2).contiguous()
        v = value.transpose(1, 2).contiguous()

        # Official FastVideo's measured FastH3 route opts into this native
        # Blackwell forward. Wheels without the extension (including SM103
        # builds today) retain the corrected explicit-mask Triton route.
        if os.environ.get("FASTVIDEO_VSA_SM100A", "0") == "1":
            try:
                from fastvideo_kernel import block_sparse_attn_sm100a
                from fastvideo_kernel.triton_kernels.index import map_to_index

                if block_sparse_attn_sm100a.is_supported(q, variable_block_sizes):
                    q2k_idx, q2k_num = map_to_index(block_map)
                    out, _ = block_sparse_attn_sm100a.block_sparse_attn_sm100a(
                        q,
                        k,
                        v,
                        q2k_idx.to(torch.int32).contiguous(),
                        q2k_num.to(torch.int32).contiguous(),
                        variable_block_sizes.to(torch.int32).contiguous(),
                        need_lse=False,
                    )
                    return out.transpose(1, 2).contiguous()
            except (ImportError, RuntimeError) as exc:
                # Opting in explicitly and then silently getting a different
                # numeric path is worse than the slower route it lands on.
                logger.warning_once(
                    "FASTVIDEO_VSA_SM100A=1 requested but the native Blackwell forward is "
                    "unavailable (%s); using the Triton block-sparse route instead.",
                    exc,
                )

        from fastvideo_kernel.block_sparse_attn import block_sparse_attn

        logical_len = logical_blocks * H3_VSA_EFFECTIVE_TILE_SIZE
        out, _ = block_sparse_attn(
            q[:, :, :logical_len].contiguous(),
            k[:, :, :logical_len].contiguous(),
            v[:, :, :logical_len].contiguous(),
            block_map[..., :logical_blocks, :logical_blocks].contiguous(),
            variable_block_sizes[:logical_blocks].to(torch.int32).contiguous(),
        )
        out = out.transpose(1, 2).contiguous()
        if out.shape[1] != query.shape[1]:
            out = torch.nn.functional.pad(out, (0, 0, 0, 0, 0, query.shape[1] - out.shape[1]))
        return out

    @_fastvideo_h3_vsa_bhsd_op.register_fake
    def _(query, key, value, block_map, variable_block_sizes, logical_blocks):
        del key, value, block_map, variable_block_sizes, logical_blocks
        return torch.empty_like(query)


_fastvideo_h3_vsa_bhsd_op = torch.ops.vllm_omni.fastvideo_h3_vsa_bhsd


@functools.lru_cache(maxsize=None)
def _h3_cake_descriptor_pool(device_index: int):
    with torch.cuda.device(device_index):
        return torch.cuda.MemPool()


@functools.lru_cache(maxsize=None)
def _h3_cake_tma_workspace(device_index: int, stream_id: int, binding: tuple) -> torch.Tensor:
    if os.environ.get("VLLM_OMNI_H3_CAKE_TMA_GRID_CONSTANT") == "1":
        return torch.empty(0, device=torch.device("cuda", device_index), dtype=torch.uint8)
    """Keep an immutable descriptor slot for each exact tensor binding/stream.

    A temporary allocation can be recycled for unrelated tensors while Cake's
    native address-based descriptor cache still assumes its contents survive.
    These 512-byte slots live for this benchmark worker's lifetime. A separate
    slot for each binding also avoids remapping a descriptor still in use by
    an earlier asynchronous launch on a non-default stream.
    """
    # Native descriptor initialization uses cuMemcpyHtoD, outside the PyTorch
    # current stream. A fresh workspace from the ordinary pool may still have
    # pending writes by a prior tensor on a non-blocking stream. Reserve a pool
    # used exclusively for these lifetime-retained descriptors so that the
    # initializer never races with a previous owner of the allocation.
    with torch.cuda.use_mem_pool(_h3_cake_descriptor_pool(device_index), device=device_index):
        return torch.empty(512, device=torch.device("cuda", device_index), dtype=torch.uint8)


def _flashinfer_h3_vsa_q2k_bshd_impl(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    q2k_idx: torch.Tensor,
    q2k_num: torch.Tensor,
    variable_block_sizes: torch.Tensor,
    softmax_scale: float,
    *, fine_pipeline: bool = False, tiled_output: bool = False,
) -> torch.Tensor:
    """Dispatch H3's blk64 VSA contract to the native FlashInfer kernel."""
    skip_softmax_threshold = h3_vsa_skip_softmax_threshold_scale_factor()
    if q2k_idx.dtype != torch.int32 or not q2k_idx.is_contiguous():
        raise ValueError("q2k_idx must be a contiguous int32 tensor")
    if q2k_num.dtype != torch.int32 or not q2k_num.is_contiguous():
        raise ValueError("q2k_num must be a contiguous int32 tensor")
    blocks = query.shape[1] // H3_VSA_EFFECTIVE_TILE_SIZE
    expected_idx_shape = (query.shape[0], query.shape[2], blocks, blocks)
    expected_num_shape = expected_idx_shape[:-1]
    if tuple(q2k_idx.shape) != expected_idx_shape:
        raise ValueError(f"q2k_idx shape {tuple(q2k_idx.shape)} must be {expected_idx_shape}")
    if tuple(q2k_num.shape) != expected_num_shape:
        raise ValueError(f"q2k_num shape {tuple(q2k_num.shape)} must be {expected_num_shape}")
    if q2k_idx.device != query.device or q2k_num.device != query.device:
        raise ValueError("q2k_idx and q2k_num must be on the same device as query")
    block_sizes = variable_block_sizes.to(torch.int32).contiguous()

    # Experimental A/B: preserve the VSA mask and BF16 transport; quantize
    # only the fine-attention operands at the opaque custom-op boundary.
    sage_mode = os.environ.get("VLLM_OMNI_H3_VSA_SAGE", "bf16")
    if sage_mode not in ("bf16", "pr4691", "pr4951"):
        raise ValueError(f"Unsupported VLLM_OMNI_H3_VSA_SAGE={sage_mode!r}")
    if sage_mode in ("pr4691", "pr4951"):
        if skip_softmax_threshold:
            raise ValueError("Sage A/B requires exact online softmax (skip threshold zero)")
        if torch.cuda.get_device_capability(query.device) != (12, 0):
            raise ValueError("This Sage A/B is qualified only for SM120")
        if any(t.dtype != torch.bfloat16 for t in (query, key, value)):
            raise ValueError("Sage A/B requires BF16 Q/K/V at the transport boundary")
        if sage_mode == "pr4691":
            from flashinfer.cute_dsl.sparse.bsa_attn_sm120 import bsa_attn_sm120_blk64_sage_fwd
        else:
            if os.environ.get("VLLM_OMNI_H3_CAKE_TMA_GRID_CONSTANT") == "1":
                from vllm_omni.diffusion.attention.ops.h3_cake_grid_constant import configure
                configure()
            from flashinfer.cute_dsl.sparse.bsa_attn_sm120_cake import bsa_attn_sm120_blk64_sage_fwd
        from flashinfer.cute_dsl.sparse.bsa_utils.sage_quant_sm120 import quantize_sage_qkv_sm120
        from vllm_omni.diffusion.attention.ops.h3_schedule_v3 import layout_view_enabled, quantized_q

        logger.info_once(
            f"FASTVIDEO_VSA H3 compute kernel: FlashInfer vsa_sm120_blk64_sage_{sage_mode} "
            "QK_INT8_PV_FP8; BF16 transport; unchanged sparse indices"
        )
        with _h3_vsa_nvtx_stage("vsa.sage.input_layout"):
            use_layout_view = layout_view_enabled()
            q_bhsd, k_bhsd, v_bhsd = (
                t.transpose(1, 2) if use_layout_view else t.transpose(1, 2).contiguous()
                for t in (query, key, value)
            )
        with _h3_vsa_nvtx_stage("vsa.sage.quantize"):
            ready_q = quantized_q(query)
            if ready_q is None:
                quantized = quantize_sage_qkv_sm120(q_bhsd, k_bhsd, v_bhsd)
            else:
                from flashinfer.cute_dsl.sparse.bsa_utils.sage_quant_sm120 import quantize_sage_kv_sm120

                k_int8, v_fp8, k_scale, v_scale = quantize_sage_kv_sm120(k_bhsd, v_bhsd)
                quantized = (ready_q[0], k_int8, v_fp8, ready_q[1], k_scale, v_scale)
        with _h3_vsa_nvtx_stage("vsa.sage.attention"):
            cake_storage = {}
            if sage_mode == "pr4951":
                cake_out = torch.empty_like(q_bhsd, memory_format=torch.contiguous_format)
                binding = tuple((t.data_ptr(), tuple(t.shape)) for t in (*quantized[:3], cake_out))
                cake_storage = {
                    "out": cake_out,
                    "tma_descriptor_workspace": _h3_cake_tma_workspace(
                        query.device.index, torch.cuda.current_stream(query.device).cuda_stream, binding
                    ),
                }
            if fine_pipeline:
                from vllm_omni.diffusion.attention.ops.h3_fine_pipeline import launch

                assert sage_mode == "pr4951" and ready_q is not None and use_layout_view
                return launch(quantized, q2k_idx, q2k_num, block_sizes, softmax_scale,
                              cake_storage["out"], cake_storage["tma_descriptor_workspace"])
            output = bsa_attn_sm120_blk64_sage_fwd(
                *quantized,
                q2k_idx,
                int(q2k_idx.shape[-1]),
                block_sizes=block_sizes,
                q2k_block_nums=q2k_num,
                softmax_scale=softmax_scale,
                **cake_storage,
            )
        if tiled_output:
            assert sage_mode == "pr4951" and not fine_pipeline
            assert output.shape == (1, 7, 105408, 128) and output.is_contiguous()
            return output.transpose(1, 2)
        with _h3_vsa_nvtx_stage("vsa.sage.output_layout"):
            return output.transpose(1, 2).contiguous()

    try:
        major, minor = torch.cuda.get_device_capability(query.device)
    except (AssertionError, RuntimeError, ValueError) as exc:
        raise RuntimeError("FlashInfer H3 blk64 VSA requires a CUDA tensor on a supported GPU") from exc
    if (major, minor) in ((10, 0), (10, 3)):
        if skip_softmax_threshold:
            raise RuntimeError(
                f"{H3_VSA_SKIP_SOFTMAX_THRESHOLD_ENV} is supported only by "
                "FlashInfer's SM120/SM121 blk64 CuTeDSL kernel"
            )
        if query.dtype != torch.bfloat16:
            raise ValueError(f"FlashInfer SM100/SM103 blk64 VSA requires bfloat16, got {query.dtype}")
        try:
            from flashinfer.cute_dsl.sparse.bsa_attn_sm100_blk64 import bsa_attn_sm100_blk64_fwd
        except (ImportError, AttributeError) as exc:
            raise ImportError(
                "FlashInfer H3 VSA on SM100/SM103 requires flashinfer.cute_dsl.sparse.bsa_attn_sm100_blk64"
            ) from exc

        kernel = bsa_attn_sm100_blk64_fwd
        kernel_name = "vsa_sm100_blk64_cuda"
    elif (major, minor) in ((12, 0), (12, 1)):
        try:
            from flashinfer.cute_dsl.sparse.bsa_attn_sm120 import bsa_attn_sm120_blk64_fwd
        except (ImportError, AttributeError) as exc:
            raise ImportError(
                "FlashInfer H3 VSA on SM120/SM121 requires the #4944 blk64 provider "
                "flashinfer.cute_dsl.sparse.bsa_attn_sm120"
            ) from exc

        kernel = bsa_attn_sm120_blk64_fwd
        kernel_name = "vsa_sm120_blk64_cute_dsl"
    else:
        raise RuntimeError(
            f"FlashInfer H3 blk64 VSA supports SM100/SM103 and SM120/SM121; current device is SM{major}{minor}"
        )

    logger.info_once("FASTVIDEO_VSA H3 compute kernel: FlashInfer %s", kernel_name)
    if skip_softmax_threshold:
        logger.warning_once(
            "FASTVIDEO_VSA H3 approximate skip-softmax enabled: %s=%g; "
            "this output is not part of the exact VSA baseline",
            H3_VSA_SKIP_SOFTMAX_THRESHOLD_ENV,
            skip_softmax_threshold,
        )
    kernel_kwargs: dict[str, Any] = {}
    # PR #4259 is the exact/original SM120 CuTeDSL provider and predates the
    # optional approximate skip-softmax argument.  Do not pass a placeholder
    # keyword on the exact (zero-threshold) path: apart from preserving the
    # original ABI, this makes it impossible for the profile lane to select a
    # skip-softmax specialization accidentally.  Newer providers receive the
    # keyword only when approximation was explicitly requested.
    if (major, minor) in ((12, 0), (12, 1)) and skip_softmax_threshold > 0:
        kernel_kwargs["skip_softmax_threshold_scale_factor"] = skip_softmax_threshold
    output, _ = kernel(
        query,
        key,
        value,
        q2k_idx,
        int(q2k_idx.shape[-1]),
        block_sizes=block_sizes,
        q2k_block_nums=q2k_num,
        softmax_scale=softmax_scale,
        return_lse=False,
        **kernel_kwargs,
    )
    return output


def _flashinfer_h3_vsa_bshd_impl(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    block_map: torch.Tensor,
    variable_block_sizes: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    """Legacy FlashInfer wrapper accepting FastVideo's dense bool map."""
    from fastvideo_kernel.triton_kernels.index import map_to_index

    q2k_idx, q2k_num = map_to_index(block_map)
    return _flashinfer_h3_vsa_q2k_bshd_impl(
        query,
        key,
        value,
        q2k_idx.to(torch.int32).contiguous(),
        q2k_num.to(torch.int32).contiguous(),
        variable_block_sizes,
        softmax_scale,
    )


if not hasattr(torch.ops.vllm_omni, "flashinfer_h3_vsa_bshd"):

    @torch.library.custom_op("vllm_omni::flashinfer_h3_vsa_bshd", mutates_args=())
    def _flashinfer_h3_vsa_bshd_op(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        block_map: torch.Tensor,
        variable_block_sizes: torch.Tensor,
        softmax_scale: float,
    ) -> torch.Tensor:
        return _flashinfer_h3_vsa_bshd_impl(
            query,
            key,
            value,
            block_map,
            variable_block_sizes,
            softmax_scale,
        )

    @_flashinfer_h3_vsa_bshd_op.register_fake
    def _(query, key, value, block_map, variable_block_sizes, softmax_scale):
        del key, value, block_map, variable_block_sizes, softmax_scale
        return torch.empty_like(query)


_flashinfer_h3_vsa_bshd_op = torch.ops.vllm_omni.flashinfer_h3_vsa_bshd


if not hasattr(torch.ops.vllm_omni, "flashinfer_h3_vsa_q2k_bshd"):

    @torch.library.custom_op("vllm_omni::flashinfer_h3_vsa_q2k_bshd", mutates_args=())
    def _flashinfer_h3_vsa_q2k_bshd_op(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        q2k_idx: torch.Tensor,
        q2k_num: torch.Tensor,
        variable_block_sizes: torch.Tensor,
        softmax_scale: float,
    ) -> torch.Tensor:
        return _flashinfer_h3_vsa_q2k_bshd_impl(
            query,
            key,
            value,
            q2k_idx,
            q2k_num,
            variable_block_sizes,
            softmax_scale,
        )

    @_flashinfer_h3_vsa_q2k_bshd_op.register_fake
    def _(query, key, value, q2k_idx, q2k_num, variable_block_sizes, softmax_scale):
        del key, value, q2k_idx, q2k_num, variable_block_sizes, softmax_scale
        return torch.empty_like(query)


_flashinfer_h3_vsa_q2k_bshd_op = torch.ops.vllm_omni.flashinfer_h3_vsa_q2k_bshd


if not hasattr(torch.ops.vllm_omni, "flashinfer_h3_vsa_q2k_tiled_output"):
    @torch.library.custom_op("vllm_omni::flashinfer_h3_vsa_q2k_tiled_output", mutates_args=())
    def _flashinfer_h3_vsa_q2k_tiled_output_op(
        query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
        q2k_idx: torch.Tensor, q2k_num: torch.Tensor,
        variable_block_sizes: torch.Tensor, softmax_scale: float,
    ) -> torch.Tensor:
        return _flashinfer_h3_vsa_q2k_bshd_impl(query, key, value, q2k_idx, q2k_num,
            variable_block_sizes, softmax_scale, tiled_output=True)

    @_flashinfer_h3_vsa_q2k_tiled_output_op.register_fake
    def _(query, key, value, q2k_idx, q2k_num, variable_block_sizes, softmax_scale):
        from vllm_omni.diffusion.attention.ops.h3_tiled_o_producer import fake_output
        return fake_output(query)

_flashinfer_h3_vsa_q2k_tiled_output_op = torch.ops.vllm_omni.flashinfer_h3_vsa_q2k_tiled_output


if not hasattr(torch.ops.vllm_omni, "flashinfer_h3_vsa_q2k_pipeline"):

    @torch.library.custom_op("vllm_omni::flashinfer_h3_vsa_q2k_pipeline", mutates_args=())
    def _flashinfer_h3_vsa_q2k_pipeline_op(
        query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
        q2k_idx: torch.Tensor, q2k_num: torch.Tensor,
        variable_block_sizes: torch.Tensor, softmax_scale: float,
    ) -> torch.Tensor:
        return _flashinfer_h3_vsa_q2k_bshd_impl(
            query, key, value, q2k_idx, q2k_num, variable_block_sizes,
            softmax_scale, fine_pipeline=True,
        )

    @_flashinfer_h3_vsa_q2k_pipeline_op.register_fake
    def _(query, key, value, q2k_idx, q2k_num, variable_block_sizes, softmax_scale):
        from vllm_omni.diffusion.attention.ops.h3_fine_producer import fake_cake_output_view

        return fake_cake_output_view(query)


_flashinfer_h3_vsa_q2k_pipeline_op = torch.ops.vllm_omni.flashinfer_h3_vsa_q2k_pipeline


@functools.lru_cache(maxsize=32)
def _get_tile_partition_indices(
    dit_seq_shape: tuple[int, int, int],
    tile_size: tuple[int, int, int],
    device: torch.device,
) -> torch.Tensor:
    t_size, h_size, w_size = dit_seq_shape
    tile_t, tile_h, tile_w = tile_size
    indices = torch.arange(t_size * h_size * w_size, device=device, dtype=torch.long).reshape(t_size, h_size, w_size)
    tiles = []
    for tile_t_idx in range(math.ceil(t_size / tile_t)):
        for tile_h_idx in range(math.ceil(h_size / tile_h)):
            for tile_w_idx in range(math.ceil(w_size / tile_w)):
                tiles.append(
                    indices[
                        tile_t_idx * tile_t : min((tile_t_idx + 1) * tile_t, t_size),
                        tile_h_idx * tile_h : min((tile_h_idx + 1) * tile_h, h_size),
                        tile_w_idx * tile_w : min((tile_w_idx + 1) * tile_w, w_size),
                    ].flatten()
                )
    return torch.cat(tiles, dim=0)


@functools.lru_cache(maxsize=32)
def _construct_variable_block_sizes(
    dit_seq_shape: tuple[int, int, int],
    tile_size: tuple[int, int, int],
    device: torch.device,
) -> torch.Tensor:
    num_tiles = tuple(math.ceil(seq_dim / tile_dim) for seq_dim, tile_dim in zip(dit_seq_shape, tile_size))

    def _sizes(dim_len: int, tile: int, n_tiles: int) -> torch.Tensor:
        sizes = torch.full((n_tiles,), tile, dtype=torch.int32, device=device)
        remainder = dim_len - (n_tiles - 1) * tile
        sizes[-1] = remainder if remainder > 0 else tile
        return sizes

    t_sizes = _sizes(dit_seq_shape[0], tile_size[0], num_tiles[0])
    h_sizes = _sizes(dit_seq_shape[1], tile_size[1], num_tiles[1])
    w_sizes = _sizes(dit_seq_shape[2], tile_size[2], num_tiles[2])
    return (t_sizes[:, None, None] * h_sizes[None, :, None] * w_sizes[None, None, :]).reshape(-1)


@functools.lru_cache(maxsize=32)
def _get_non_pad_index(variable_block_sizes: torch.Tensor, max_block_size: int) -> torch.Tensor:
    num_blocks = variable_block_sizes.shape[0]
    device = variable_block_sizes.device
    starts = torch.arange(num_blocks, device=device) * max_block_size
    padded_index = starts[:, None] + torch.arange(max_block_size, device=device)[None, :]
    valid = torch.arange(max_block_size, device=device)[None, :] < variable_block_sizes[:, None]
    return padded_index[valid]


@torch.compiler.disable
def _get_tile_metadata(
    dit_seq_shape: tuple[int, int, int],
    tile_size: tuple[int, int, int],
    block_elements: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    tile_partition_indices = _get_tile_partition_indices(dit_seq_shape, tile_size, device)
    variable_block_sizes = _construct_variable_block_sizes(dit_seq_shape, tile_size, device)
    non_pad_index = _get_non_pad_index(variable_block_sizes, block_elements)
    untile_combined_index = non_pad_index[torch.argsort(tile_partition_indices)]
    return tile_partition_indices, variable_block_sizes, non_pad_index, untile_combined_index


@functools.lru_cache(maxsize=32)
def _get_h3_tile_metadata(
    prefix_segments: tuple[int, ...],
    video_shape: tuple[int, int, int],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
    """Official FastVideo H3 geometry: pure prefix chunks + 3-D video tiles."""
    block_size = H3_VSA_EFFECTIVE_TILE_SHAPE
    block_elements = H3_VSA_EFFECTIVE_TILE_SIZE
    prefix_len = sum(prefix_segments)
    prefix_sizes: list[int] = []
    for segment in prefix_segments:
        full, remainder = divmod(segment, block_elements)
        prefix_sizes.extend([block_elements] * full)
        if remainder:
            prefix_sizes.append(remainder)

    video_indices = _get_tile_partition_indices(video_shape, block_size, device) + prefix_len
    video_sizes = _construct_variable_block_sizes(video_shape, block_size, device)
    partition = torch.cat([torch.arange(prefix_len, device=device, dtype=torch.long), video_indices])
    sizes = torch.cat([torch.tensor(prefix_sizes, device=device, dtype=torch.int32), video_sizes.to(torch.int32)])
    non_pad = _get_non_pad_index(sizes, block_elements)
    untile = non_pad[torch.argsort(partition)]
    total = prefix_len + math.prod(video_shape)
    if int(sizes.sum()) != total or untile.numel() != total:
        raise ValueError(
            f"invalid H3 VSA geometry: prefix={prefix_segments}, video={video_shape}, "
            f"sizes_sum={int(sizes.sum())}, total={total}"
        )
    return partition, sizes, non_pad, untile, len(prefix_sizes), int(video_sizes.numel())


@functools.lru_cache(maxsize=32)
def _get_h3_tiled_source_rows(
    prefix_segments: tuple[int, ...],
    video_shape: tuple[int, int, int],
    padded_rows: int,
    device: torch.device,
) -> torch.Tensor:
    """Cache the validated destination-to-source map for fused H3 tiling."""
    partition, _sizes, non_pad, _untile, _prefix_blocks, _video_blocks = _get_h3_tile_metadata(
        prefix_segments,
        video_shape,
        device,
    )
    return build_h3_tiled_source_rows(partition, non_pad, padded_rows)


@functools.lru_cache(maxsize=32)
def _get_h3_aligned_untile_source_rows(
    prefix_segments: tuple[int, ...],
    video_shape: tuple[int, int, int],
    aligned_rows: int,
    tiled_rows: int,
    device: torch.device,
) -> torch.Tensor:
    """Cache the validated tile-64 source map for aligned H3 output."""
    _partition, sizes, _non_pad, untile, _prefix_blocks, _video_blocks = _get_h3_tile_metadata(
        prefix_segments,
        video_shape,
        device,
    )
    expected_tiled_rows = int(sizes.numel()) * H3_VSA_EFFECTIVE_TILE_SIZE
    if tiled_rows != expected_tiled_rows:
        raise ValueError(f"H3 VSA output has {tiled_rows} logical tiled rows, expected {expected_tiled_rows}")
    return build_h3_aligned_untile_source_rows(
        untile,
        aligned_rows=aligned_rows,
        tiled_rows=tiled_rows,
    )


@functools.lru_cache(maxsize=32)
@torch.compiler.disable
def get_h3_vsa_owner_route_plan(
    prefix_segments: tuple[int, ...],
    video_shape: tuple[int, int, int],
    aligned_rows: int,
    sp_world_size: int,
) -> H3VSAOwnerRoutePlan:
    """Build and cache the exact aligned H3 row-owner route on CPU.

    Ulysses needs ``Kmax`` before the attention backend runs so it can acquire
    an expanded reverse-O landing.  Deriving the plan from the same H3 untile
    metadata used below keeps the producer and consumer on one row-order
    contract without synchronizing CUDA metadata back to the host.
    """
    if aligned_rows <= 0 or sp_world_size <= 1:
        raise ValueError(
            "H3 VSA owner routing requires positive aligned_rows and SP world_size > 1, "
            f"got aligned_rows={aligned_rows}, sp_world_size={sp_world_size}"
        )
    if aligned_rows % sp_world_size:
        raise ValueError(f"H3 VSA aligned rows must divide the SP world: rows={aligned_rows}, world={sp_world_size}")

    cpu = torch.device("cpu")
    _partition, sizes, _non_pad, untile, _prefix_blocks, _video_blocks = _get_h3_tile_metadata(
        prefix_segments,
        video_shape,
        cpu,
    )
    logical_blocks = int(sizes.numel())
    aligned_source_rows = build_h3_aligned_untile_source_rows(
        untile,
        aligned_rows=aligned_rows,
        tiled_rows=logical_blocks * H3_VSA_EFFECTIVE_TILE_SIZE,
    )
    return build_h3_vsa_owner_route_plan(
        aligned_source_rows,
        sp_world_size=sp_world_size,
        global_rows=aligned_rows,
        local_rows=aligned_rows // sp_world_size,
        logical_block_count=logical_blocks,
    )


@functools.lru_cache(maxsize=32)
def _get_h3_compact_row_blocks(
    prefix_segments: tuple[int, ...],
    video_shape: tuple[int, int, int],
    device: torch.device,
) -> torch.Tensor:
    """Cache compact/global-row -> VSA coarse-tile mapping."""
    _partition, sizes, _non_pad, untile, _prefix_blocks, _video_blocks = _get_h3_tile_metadata(
        prefix_segments,
        video_shape,
        device,
    )
    return build_h3_compact_row_blocks(untile, num_blocks=int(sizes.numel()))


def _get_h3_layout(
    attn_metadata: AttentionMetadata | None,
) -> tuple[tuple[int, ...], tuple[int, int, int], int] | None:
    if attn_metadata is None or attn_metadata.video_layout is None:
        return None
    prefix = attn_metadata.extra.get("vsa_h3_prefix_segments")
    if not isinstance(prefix, (tuple, list)):
        return None
    target = next(
        (span for span in reversed(attn_metadata.video_layout.video_spans) if span.role == "target"),
        None,
    )
    if target is None:
        return None
    return tuple(int(x) for x in prefix if int(x) > 0), target.latent_grid, target.start


def _pool_h3_tiles(x: torch.Tensor, sizes: torch.Tensor) -> torch.Tensor:
    batch, seq_len, heads, dim = x.shape
    blocks = seq_len // H3_VSA_EFFECTIVE_TILE_SIZE
    pooled = x.view(batch, blocks, H3_VSA_EFFECTIVE_TILE_SIZE, heads, dim).sum(dim=2, dtype=torch.float32)
    pooled = pooled / sizes.view(1, -1, 1, 1).clamp_min(1)
    return pooled.permute(0, 2, 1, 3)


def _build_h3_block_map(
    scores: torch.Tensor,
    num_prefix_blocks: int,
    num_video_blocks: int,
    topk: int,
) -> torch.Tensor:
    """Prefix K/V are exempt and prefix queries stay dense, as in FastVideo."""
    keep_video = min(topk, num_video_blocks)
    if keep_video == num_video_blocks:
        return torch.ones_like(scores, dtype=torch.bool)
    block_map = torch.zeros_like(scores, dtype=torch.bool)
    indices = scores[..., num_prefix_blocks:].topk(keep_video, dim=-1).indices + num_prefix_blocks
    block_map.scatter_(-1, indices, True)
    block_map[..., :num_prefix_blocks] = True
    block_map[:, :, :num_prefix_blocks, :] = True
    return block_map


def _build_h3_ordered_q2k_indices(
    scores: torch.Tensor,
    num_prefix_blocks: int,
    num_video_blocks: int,
    topk: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build FlashInfer's ordered sparse metadata without a dense bool map.

    ``map_to_index`` scans each bool-map row from left to right. Sorting the
    selected video block IDs and prepending the exempt prefix therefore emits
    exactly the same index order and ``-1`` tail, while avoiding both the bool
    map allocation and its full-width scan.
    """
    if scores.ndim != 4:
        raise ValueError(f"scores must be [B, H, Nq, Nkv], got {tuple(scores.shape)}")
    if num_prefix_blocks < 0 or num_video_blocks < 0:
        raise ValueError("num_prefix_blocks and num_video_blocks must be non-negative")
    total_blocks = num_prefix_blocks + num_video_blocks
    if scores.shape[-2:] != (total_blocks, total_blocks):
        raise ValueError(
            "scores must match the H3 prefix+video block geometry: "
            f"got {tuple(scores.shape[-2:])}, expected {(total_blocks, total_blocks)}"
        )
    if topk < 0:
        raise ValueError(f"topk must be non-negative, got {topk}")

    keep_video = min(topk, num_video_blocks)
    selected: torch.Tensor | None = None
    if keep_video < num_video_blocks and keep_video:
        # Complete the workspace-heavy selection before the full rectangular
        # q2k output becomes live.  On the production H3 geometry, overlapping
        # q2k with torch.topk's workspace costs roughly 53 MiB of peak memory.
        selected = scores[..., num_prefix_blocks:, num_prefix_blocks:].topk(keep_video, dim=-1).indices.to(torch.int32)
        selected = selected.sort(dim=-1).values

    output_shape = tuple(scores.shape)
    q2k_idx = torch.full(output_shape, -1, dtype=torch.int32, device=scores.device)
    q2k_num = torch.full(
        output_shape[:-1],
        num_prefix_blocks + keep_video,
        dtype=torch.int32,
        device=scores.device,
    )
    all_indices = torch.arange(total_blocks, dtype=torch.int32, device=scores.device)

    if keep_video == num_video_blocks:
        q2k_idx.copy_(all_indices)
        q2k_num.fill_(total_blocks)
        return q2k_idx, q2k_num

    if num_prefix_blocks:
        q2k_idx[..., num_prefix_blocks:, :num_prefix_blocks] = all_indices[:num_prefix_blocks]
    if selected is not None:
        # Prefix-query selections are omitted entirely: those rows are dense
        # and are populated below.
        q2k_idx[
            ...,
            num_prefix_blocks:,
            num_prefix_blocks : num_prefix_blocks + keep_video,
        ] = selected + num_prefix_blocks
    if num_prefix_blocks:
        q2k_idx[..., :num_prefix_blocks, :] = all_indices
        q2k_num[..., :num_prefix_blocks] = total_blocks
    return q2k_idx, q2k_num


def _get_vsa_dit_seq_shape(attn_metadata: AttentionMetadata | None) -> tuple[int, int, int] | None:
    if attn_metadata is None:
        return None
    value = attn_metadata.extra.get("vsa_dit_seq_shape")
    if value is None:
        return None
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        return None
    return (int(value[0]), int(value[1]), int(value[2]))


def _get_gate_compress(attn_metadata: AttentionMetadata | None) -> torch.Tensor | None:
    if attn_metadata is None:
        return None
    value = attn_metadata.extra.get("gate_compress")
    return value if isinstance(value, torch.Tensor) else None


def _get_h3_fp8_qkv_scales(
    attn_metadata: AttentionMetadata | None,
) -> tuple[float, float, float] | None:
    """Return trusted per-Q/K/V E4M3 dequant scales.

    Presence of the metadata key opts into a fail-closed transport contract.
    Python-float scales avoid materializing a device tensor in every block.
    """
    if attn_metadata is None:
        return None
    if H3_VSA_FP8_QKV_SCALES_KEY not in attn_metadata.extra:
        return None
    value = attn_metadata.extra[H3_VSA_FP8_QKV_SCALES_KEY]
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise TypeError(
            f"{H3_VSA_FP8_QKV_SCALES_KEY} must be a length-3 list/tuple "
            "of Python floats"
        )
    scales: list[float] = []
    for name, scale in zip(("q", "k", "v"), value, strict=True):
        if not isinstance(scale, float):
            raise TypeError(
                f"{H3_VSA_FP8_QKV_SCALES_KEY}[{name}] must be a Python "
                f"float, got {type(scale)!r}"
            )
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError(
                f"{H3_VSA_FP8_QKV_SCALES_KEY}[{name}] must be finite and "
                f"positive, got {scale}"
            )
        scales.append(scale)
    return scales[0], scales[1], scales[2]


def _preserve_vsa_all_blocks(attn_metadata: AttentionMetadata | None) -> bool:
    if attn_metadata is None:
        return False
    return attn_metadata.extra.get("preserve_vsa_all_blocks") is True


class FastVideoVSABackend(AttentionBackend):
    accept_output_buffer: bool = True

    @classmethod
    def supports_packed_mask_free(cls) -> bool:
        # FastVideo accepts variable-sized edge blocks. This lets packed
        # [real, pad] inputs run on their valid prefix without materializing an
        # attention mask; the implementation restores the ignored pad rows.
        # Only forward_cuda honours packed_padding: every other platform hands
        # the tensors straight to SDPA, which reads attn_mask and nothing else,
        # so the pad rows would be attended as real keys.
        return current_omni_platform.is_cuda()

    @classmethod
    def validate_available(cls) -> None:
        if importlib.util.find_spec("fastvideo_kernel") is None:
            raise ImportError(
                "FASTVIDEO_VSA requires the optional fastvideo-kernel package "
                "(the installation must provide the 'fastvideo_kernel' Python module)."
            )

    @staticmethod
    def get_supported_head_sizes() -> list[int]:
        # FastVideo VSA is intended for video DiT head sizes such as 64/128.
        # Keep this permissive and let the runtime fallback handle unsupported
        # cases from the installed fastvideo-kernel build.
        return []

    @staticmethod
    def get_name() -> str:
        return "FASTVIDEO_VSA"

    @staticmethod
    def get_impl_cls() -> type[FastVideoVSAImpl]:
        return FastVideoVSAImpl


class FastVideoVSAImpl(AttentionImpl):
    # Effective H3 execution geometry.  The generic ``block_size`` below is a
    # different contract and is intentionally left at its 256-token default.
    h3_effective_tile_shape = H3_VSA_EFFECTIVE_TILE_SHAPE
    h3_effective_tile_size = H3_VSA_EFFECTIVE_TILE_SIZE

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        softmax_scale: float,
        causal: bool = False,
        num_kv_heads: int | None = None,
        prefix: str = "",
        qkv_layout: str | None = None,
        backend_kwargs: Mapping[str, Any] | None = None,
        **extra_impl_args,
    ) -> None:
        backend_kwargs = backend_kwargs or {}
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.head_size = head_size
        self.softmax_scale = softmax_scale
        self.causal = causal
        self.qkv_layout = qkv_layout

        self.topk = int(backend_kwargs.get("topk", 64))
        self.block_size = self._parse_block_size(backend_kwargs.get("block_size", (4, 8, 8)))
        self.block_elements = self.block_size[0] * self.block_size[1] * self.block_size[2]
        self.min_seq_len = int(backend_kwargs.get("min_seq_len", self.block_elements * 2))
        self.fallback_on_error = bool(backend_kwargs.get("fallback_on_error", True))
        self.disable_when_sp_active = bool(backend_kwargs.get("disable_when_sp_active", True))
        self.h3_kernel_backend = str(backend_kwargs.get("h3_kernel_backend", "fastvideo")).lower()
        if self.h3_kernel_backend not in {"fastvideo", "flashinfer"}:
            raise ValueError(
                f"FASTVIDEO_VSA h3_kernel_backend must be 'fastvideo' or 'flashinfer', got {self.h3_kernel_backend!r}"
            )

        self.sdpa_fallback = SDPAImpl(
            num_heads=num_heads,
            head_size=head_size,
            softmax_scale=softmax_scale,
            causal=causal,
            num_kv_heads=num_kv_heads,
            qkv_layout=qkv_layout,
        )

        if self.block_elements != 256:
            logger.warning(
                "FASTVIDEO_VSA currently uses fastvideo_kernel.video_sparse_attn_bshd, "
                "which supports only 256-token blocks. Configured block_size=%s "
                "(product=%d) will fall back to SDPA.",
                self.block_size,
                self.block_elements,
            )

    @staticmethod
    def _parse_block_size(value: Any) -> tuple[int, int, int]:
        if isinstance(value, int):
            return (value, value, value)
        if isinstance(value, (list, tuple)) and len(value) == 3:
            return (int(value[0]), int(value[1]), int(value[2]))
        raise ValueError(f"FASTVIDEO_VSA block_size must be an int or length-3 tuple/list, got {value!r}")

    def _fallback(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata | None,
        reason: str,
    ) -> torch.Tensor:
        """Run dense SDPA, honouring the mask-free packed contract this backend claims.

        ``supports_packed_mask_free`` tells the model it may skip building the
        padding mask, so on this path there is nothing to stop SDPA attending
        the structural pad rows as real keys. Slice to the valid prefix instead
        and leave the pad rows zeroed, exactly as the VSA path does.
        """
        logger.warning_once("FASTVIDEO_VSA falling back to SDPA: %s", reason)
        packed = attn_metadata.packed_padding if attn_metadata is not None else None
        if attn_metadata is None or packed is None or attn_metadata.attn_mask is not None:
            return self.sdpa_fallback.forward(query, key, value, attn_metadata)
        q_length = min(int(packed.q_length), query.shape[1])
        kv_length = min(int(packed.kv_length), key.shape[1])
        output = self.sdpa_fallback.forward(
            query[:, :q_length], key[:, :kv_length], value[:, :kv_length], attn_metadata
        )
        if q_length == query.shape[1]:
            return output
        restored = torch.zeros(
            (output.shape[0], query.shape[1], output.shape[2], output.shape[3]),
            device=output.device,
            dtype=output.dtype,
        )
        restored[:, :q_length] = output
        return restored

    def _fallback_reason(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata | None,
    ) -> str | None:
        if self.causal:
            return "causal attention is not supported"
        if self.block_elements != 256:
            return f"block_elements must be 256, got {self.block_elements}"
        if self.topk <= 0:
            return f"topk must be positive, got {self.topk}"
        if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
            return f"expected [B, S, H, D] tensors, got {query.shape}, {key.shape}, {value.shape}"
        if query.shape[0] != key.shape[0] or query.shape[0] != value.shape[0]:
            return "batch dimensions must match"
        if query.shape[2:] != key.shape[2:] or query.shape[2:] != value.shape[2:]:
            return "head/head_dim dimensions must match"
        if query.shape[1] != key.shape[1] or query.shape[1] != value.shape[1]:
            return "initial VSA backend supports self-attention with Sq == Skv only"
        if query.shape[1] < self.min_seq_len:
            return f"sequence length {query.shape[1]} is below min_seq_len {self.min_seq_len}"
        dit_seq_shape = _get_vsa_dit_seq_shape(attn_metadata)
        if dit_seq_shape is None:
            return "vsa_dit_seq_shape metadata is required"
        if math.prod(dit_seq_shape) != query.shape[1]:
            return f"vsa_dit_seq_shape product {math.prod(dit_seq_shape)} != sequence length {query.shape[1]}"
        num_blocks = math.prod(
            math.ceil(seq_dim / tile_dim) for seq_dim, tile_dim in zip(dit_seq_shape, self.block_size)
        )
        if self.topk > num_blocks:
            return f"topk {self.topk} > num_blocks {num_blocks}"
        if query.dtype not in (torch.float16, torch.bfloat16):
            return f"dtype {query.dtype} is not supported"
        if key.dtype != query.dtype or value.dtype != query.dtype:
            return "q/k/v dtypes must match"
        if query.device.type != "cuda" or key.device.type != "cuda" or value.device.type != "cuda":
            return "q/k/v must be CUDA tensors"
        expected_scale = self.head_size**-0.5
        if abs(float(self.softmax_scale) - float(expected_scale)) > 1e-6:
            return f"softmax_scale {self.softmax_scale} differs from FastVideo VSA scale {expected_scale}"
        if attn_metadata is not None and attn_metadata.attn_mask is not None:
            return "attention masks are not supported"
        if attn_metadata is not None and attn_metadata.full_attn_spans is not None:
            return "piecewise/full attention spans are not supported"
        if self.num_heads != self.num_kv_heads:
            return "GQA/MQA is not supported"
        if self.disable_when_sp_active and is_forward_context_available():
            ctx = get_forward_context()
            if getattr(ctx, "sp_active", False):
                return "sequence parallel context is active"
        return None

    def _forward_h3(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata,
        aligned_rows: int,
    ) -> torch.Tensor:
        skip_softmax_threshold = h3_vsa_skip_softmax_threshold_scale_factor()
        if skip_softmax_threshold and self.h3_kernel_backend != "flashinfer":
            raise RuntimeError(
                f"{H3_VSA_SKIP_SOFTMAX_THRESHOLD_ENV}>0 requires "
                "FASTVIDEO_VSA h3_kernel_backend='flashinfer'"
            )
        layout = _get_h3_layout(attn_metadata)
        if layout is None:
            raise ValueError("incomplete VSA-H3 layout metadata")
        prefix_segments, video_shape, target_start = layout
        if sum(prefix_segments) != target_start:
            raise ValueError(f"VSA-H3 prefix segments sum to {sum(prefix_segments)}, target starts at {target_start}")
        expected = target_start + math.prod(video_shape)
        if query.shape[1] != expected:
            raise ValueError(f"VSA-H3 layout has {expected} rows but attention received {query.shape[1]}")
        fp8_qkv_scales = _get_h3_fp8_qkv_scales(attn_metadata)
        fp8_qkv_transport = fp8_qkv_scales is not None
        if fp8_qkv_transport:
            if any(tensor.dtype != torch.float8_e4m3fn for tensor in (query, key, value)):
                raise TypeError(
                    "H3 VSA FP8 QKV transport requires Q/K/V to use "
                    "torch.float8_e4m3fn"
                )
            if key.shape != query.shape or value.shape != query.shape:
                raise ValueError(
                    "H3 VSA FP8 QKV transport requires identical Q/K/V shapes, "
                    f"got {query.shape}, {key.shape}, and {value.shape}"
                )
            if key.device != query.device or value.device != query.device:
                raise ValueError("H3 VSA FP8 QKV transport requires Q/K/V on one device")
        elif query.dtype not in (torch.float16, torch.bfloat16):
            raise ValueError(f"VSA-H3 requires fp16/bf16 tensors, got {query.dtype}")
        deferred_gate = attn_metadata.extra.get(H3_VSA_DEFERRED_ACTIVE_KEY, False)
        if not isinstance(deferred_gate, bool):
            raise TypeError(f"{H3_VSA_DEFERRED_ACTIVE_KEY} must be a bool")
        deferred_state = attn_metadata.extra.get(H3_VSA_DEFERRED_STATE_KEY)
        o_bundle = attn_metadata.extra.get(H3_VSA_O_BUNDLE_ACTIVE_KEY, False)
        if not isinstance(o_bundle, bool):
            raise TypeError(f"{H3_VSA_O_BUNDLE_ACTIVE_KEY} must be a bool")
        o_bundle_state = attn_metadata.extra.get(H3_VSA_O_BUNDLE_STATE_KEY)
        owner_route: H3VSAOwnerRoutePlan | None = None
        if deferred_gate and o_bundle:
            raise RuntimeError("deferred H3 VSA gate and reverse-O bundling cannot both be active")
        if deferred_gate:
            if fp8_qkv_transport:
                raise RuntimeError(
                    "H3 VSA FP8 QKV transport is not qualified with deferred-gate communication"
                )
            if not h3_vsa_deferred_gate_sp_enabled():
                raise RuntimeError(f"{H3_VSA_DEFERRED_ACTIVE_KEY}=True requires {H3_VSA_DEFERRED_GATE_SP_ENV}=1")
            if query.dtype != torch.bfloat16:
                raise TypeError("deferred H3 VSA gate communication currently requires BF16")
            if not isinstance(deferred_state, dict):
                raise RuntimeError("deferred H3 VSA mutable state is absent")
            if deferred_state:
                raise RuntimeError("stale deferred H3 VSA mutable state")
        if o_bundle:
            if not h3_vsa_o_bundle_enabled():
                raise RuntimeError(f"{H3_VSA_O_BUNDLE_ACTIVE_KEY}=True requires {H3_VSA_O_BUNDLE_ENV}=1")
            if not fp8_qkv_transport and query.dtype != torch.bfloat16:
                raise TypeError("H3 VSA reverse-O bundling currently requires BF16")
            if not isinstance(o_bundle_state, dict):
                raise RuntimeError("H3 VSA reverse-O bundle mutable state is absent")
            if set(o_bundle_state) != {"plan"}:
                raise RuntimeError(f"stale H3 VSA reverse-O bundle mutable state: keys={sorted(o_bundle_state)}")
            candidate_plan = o_bundle_state["plan"]
            if not isinstance(candidate_plan, H3VSAOwnerRoutePlan) or not h3_vsa_owner_route_plan_is_trusted(
                candidate_plan
            ):
                raise RuntimeError("H3 VSA reverse-O bundle route plan is absent or untrusted")
            owner_route = candidate_plan
            if owner_route.sp_world_size * owner_route.local_rows != aligned_rows:
                raise ValueError(
                    "H3 VSA reverse-O bundle route does not match aligned attention rows: "
                    f"route={owner_route.sp_world_size}*{owner_route.local_rows}, "
                    f"attention={aligned_rows}"
                )
        gate = _get_gate_compress(attn_metadata)
        if (deferred_gate or o_bundle) and gate is not None:
            raise RuntimeError("deferred/bundled H3 VSA must retain gate_compress on the original SP shard")
        if gate is not None:
            # H3 pads the packed document to 64 rows, while VSA operates on
            # the valid prefix. Validate/slice before launching attention so a
            # metadata error can never trigger an unsafe dense fallback after
            # an asynchronous custom kernel.
            if gate.shape[0] != query.shape[0] or gate.shape[2:] != query.shape[2:] or gate.shape[1] < query.shape[1]:
                raise ValueError(f"gate_compress shape {gate.shape} cannot cover query shape {query.shape}")
            gate = gate[:, : query.shape[1]]
            if fp8_qkv_transport and gate.dtype != torch.bfloat16:
                raise TypeError(
                    "H3 VSA FP8 QKV transport keeps gate_compress in BF16, "
                    f"got {gate.dtype}"
                )

        partition, sizes, non_pad, untile, prefix_blocks, video_blocks = _get_h3_tile_metadata(
            prefix_segments, video_shape, query.device
        )
        logical_blocks = int(sizes.numel())
        if owner_route is not None and owner_route.logical_block_count != logical_blocks:
            raise ValueError(
                "H3 VSA reverse-O bundle route block count does not match the backend layout: "
                f"route={owner_route.logical_block_count}, backend={logical_blocks}"
            )
        # FastVideo's optional native sm100a kernel assigns pairs of query
        # blocks to CTAs. FlashInfer's blk64 kernels do not require the
        # transport-only partner.
        pair_pad = logical_blocks % 2 if self.h3_kernel_backend == "fastvideo" else 0
        kernel_blocks = logical_blocks + pair_pad
        target_shape = (
            query.shape[0],
            kernel_blocks * H3_VSA_EFFECTIVE_TILE_SIZE,
            query.shape[2],
            query.shape[3],
        )
        use_fused_tile_pack = h3_vsa_fused_tile_pack_enabled()
        use_fused_untile = h3_vsa_fused_untile_enabled()
        use_fused_gate_untile = h3_vsa_fused_gate_untile_enabled()
        if fp8_qkv_transport and not use_fused_tile_pack:
            raise RuntimeError(
                "H3 VSA FP8 QKV transport requires the fused E4M3 dequant + "
                f"tile-pack path ({H3_VSA_FUSED_TILE_PACK_ENV}=1)"
            )
        if (deferred_gate or o_bundle) and use_fused_gate_untile:
            raise ValueError(
                "deferred/bundled H3 VSA gate ownership is incompatible with "
                f"{H3_VSA_FUSED_GATE_UNTILE_ENV}=1: the gate is applied after "
                "reverse Ulysses"
            )
        if use_fused_gate_untile and gate is None:
            raise ValueError(f"{H3_VSA_FUSED_GATE_UNTILE_ENV}=1 requires H3 gate_compress metadata")
        if use_fused_gate_untile and not use_fused_untile:
            raise ValueError(f"{H3_VSA_FUSED_GATE_UNTILE_ENV}=1 requires {H3_VSA_FUSED_UNTILE_ENV}=1")
        qkv_source_rows: torch.Tensor | None = None
        gate_source_rows: torch.Tensor | None = None
        if use_fused_tile_pack:
            qkv_source_rows = _get_h3_tiled_source_rows(
                prefix_segments,
                video_shape,
                target_shape[1],
                query.device,
            )
            if gate is not None and pair_pad:
                gate_source_rows = _get_h3_tiled_source_rows(
                    prefix_segments,
                    video_shape,
                    logical_blocks * H3_VSA_EFFECTIVE_TILE_SIZE,
                    query.device,
                )
            else:
                gate_source_rows = qkv_source_rows

            pack_gate = gate is not None and not use_fused_gate_untile
            packed_tensor_count = 3 + int(pack_gate)
            if fp8_qkv_transport:
                qkv_pack_supported = all(
                    h3_vsa_fp8_dequant_tile_pack_cuda_supported(
                        tensor,
                        qkv_source_rows,
                    )
                    for tensor in (query, key, value)
                )
                gate_pack_supported = (
                    not pack_gate
                    or (
                        gate_source_rows is not None
                        and h3_vsa_tile_pack_cuda_supported(gate, gate_source_rows)
                    )
                )
                if not qkv_pack_supported or not gate_pack_supported:
                    raise RuntimeError(
                        "H3 VSA FP8 QKV transport reached an unsupported fused "
                        "dequant/tile-pack tensor contract"
                    )
                use_fused_tile_pack = True
                logger.info_once(
                    "FASTVIDEO_VSA H3 FP8 QKV transport receive path: fused "
                    "E4M3->BF16 dequant + tile64 pack enabled "
                    "(compact_rows=%d, padded_rows=%d)",
                    query.shape[1],
                    target_shape[1],
                    scope="process",
                )
            else:
                fused_inputs = (query, key, value, gate) if pack_gate else (query, key, value)
                fused_maps = (
                    (qkv_source_rows, qkv_source_rows, qkv_source_rows, gate_source_rows)
                    if pack_gate
                    else (qkv_source_rows,) * 3
                )
                use_fused_tile_pack = all(
                    source_rows is not None and h3_vsa_tile_pack_cuda_supported(tensor, source_rows)
                    for tensor, source_rows in zip(fused_inputs, fused_maps, strict=True)
                )
            if use_fused_tile_pack:
                logger.info_once(
                    "FASTVIDEO_VSA H3 tile pack: fused Triton compact->tile64 enabled "
                    "(compact_rows=%d, padded_rows=%d, tensors=%d)",
                    query.shape[1],
                    target_shape[1],
                    packed_tensor_count,
                    scope="process",
                )
            else:
                logger.warning_once(
                    "%s=1 requested but the H3 input shape/platform is unsupported; using the reference tile pack",
                    H3_VSA_FUSED_TILE_PACK_ENV,
                    scope="process",
                )

        from vllm_omni.diffusion.attention.ops.h3_overlap_experiment import prepared

        overlap_prepared = prepared()
        if overlap_prepared is not None:
            assert use_fused_tile_pack and fp8_qkv_scales is None and gate is None
            q_tiled, k_tiled, q_pool, k_pool, scores, q2k_idx, q2k_num = overlap_prepared
            with _h3_vsa_nvtx_stage("vsa.layout.tile_pack_v"):
                v_tiled = h3_vsa_tile_pack(value, qkv_source_rows)
        else:
            with _h3_vsa_nvtx_stage("vsa.layout.tile_pack"):
                if use_fused_tile_pack:
                    assert qkv_source_rows is not None
                    if fp8_qkv_scales is not None:
                        q_tiled = h3_vsa_fp8_dequant_tile_pack(
                            query,
                            qkv_source_rows,
                            fp8_qkv_scales[0],
                        )
                        k_tiled = h3_vsa_fp8_dequant_tile_pack(
                            key,
                            qkv_source_rows,
                            fp8_qkv_scales[1],
                        )
                        v_tiled = h3_vsa_fp8_dequant_tile_pack(
                            value,
                            qkv_source_rows,
                            fp8_qkv_scales[2],
                        )
                    else:
                        q_tiled = h3_vsa_tile_pack(query, qkv_source_rows)
                        k_tiled = h3_vsa_tile_pack(key, qkv_source_rows)
                        v_tiled = h3_vsa_tile_pack(value, qkv_source_rows)
                else:
                    assert not fp8_qkv_transport
                    q_tiled = torch.zeros(target_shape, device=query.device, dtype=query.dtype)
                    k_tiled = torch.zeros_like(q_tiled)
                    v_tiled = torch.zeros_like(q_tiled)
                    q_tiled[:, non_pad] = query[:, partition]
                    k_tiled[:, non_pad] = key[:, partition]
                    v_tiled[:, non_pad] = value[:, partition]

            with _h3_vsa_nvtx_stage("vsa.coarse.pool_qk"):
                q_pool = _pool_h3_tiles(
                    q_tiled[:, : logical_blocks * H3_VSA_EFFECTIVE_TILE_SIZE],
                    sizes,
                )
                k_pool = _pool_h3_tiles(
                    k_tiled[:, : logical_blocks * H3_VSA_EFFECTIVE_TILE_SIZE],
                    sizes,
                )
            with _h3_vsa_nvtx_stage("vsa.coarse.qk_scores"):
                scores = torch.matmul(q_pool, k_pool.transpose(-2, -1)) * self.softmax_scale
        kernel_sizes = sizes
        from vllm_omni.diffusion.attention.ops.h3_schedule_v3 import coarse_ready, coarse_scope

        coarse_ticket = (
            coarse_ready(v_tiled, scores)
            if gate is not None or deferred_gate or o_bundle else None
        )

        from vllm_omni.diffusion.attention.ops.h3_fine_pipeline import enabled as fine_pipeline_enabled

        fine_pipeline = fine_pipeline_enabled()
        from vllm_omni.diffusion.attention.ops.h3_tiled_o_producer import enabled as tiled_output_enabled
        tiled_output = tiled_output_enabled()
        if tiled_output:
            assert not fine_pipeline and o_bundle and use_fused_untile
            assert self.h3_kernel_backend == "flashinfer" and h3_vsa_direct_q2k_enabled()
            assert os.environ.get("VLLM_OMNI_H3_O_PRODUCER_LOOKAHEAD") == "1"
        if fine_pipeline:
            assert o_bundle and gate is None and not deferred_gate and overlap_prepared is not None
            assert self.h3_kernel_backend == "flashinfer" and h3_vsa_direct_q2k_enabled()
            assert coarse_ticket is not None

        def compute_coarse():
            # Preserve the original operations, dtypes and reduction order;
            # the candidate only submits this branch before the fine launch.
            with coarse_scope(coarse_ticket):
                with _h3_vsa_nvtx_stage("vsa.coarse.pool_v"):
                    v_pool = _pool_h3_tiles(
                        v_tiled[:, : logical_blocks * H3_VSA_EFFECTIVE_TILE_SIZE], sizes,
                    )
                with _h3_vsa_nvtx_stage("vsa.coarse.softmax"):
                    coarse_weights = torch.softmax(scores, dim=-1)
                with _h3_vsa_nvtx_stage("vsa.coarse.pv"):
                    coarse = torch.matmul(coarse_weights, v_pool)
                return coarse.permute(0, 2, 1, 3).to(q_tiled.dtype)

        compressed = compute_coarse() if fine_pipeline else None

        logger.info_once(
            "FASTVIDEO_VSA H3 routing: seq_len=%d, prefix_segments=%s, video_shape=%s, "
            "prefix_blocks=%d, video_blocks=%d, topk=%d, kernel_blocks=%d, compute_backend=%s",
            query.shape[1],
            prefix_segments,
            video_shape,
            prefix_blocks,
            video_blocks,
            min(self.topk, video_blocks),
            kernel_blocks,
            self.h3_kernel_backend,
        )
        if self.h3_kernel_backend == "flashinfer":
            if h3_vsa_direct_q2k_enabled():
                logger.info_once(
                    "FASTVIDEO_VSA H3 metadata: direct ordered q2k enabled by %s=1",
                    H3_VSA_DIRECT_Q2K_ENV,
                    scope="process",
                )
                with _h3_vsa_nvtx_stage("vsa.route.topk_q2k"):
                    if overlap_prepared is None:
                        q2k_idx, q2k_num = _build_h3_ordered_q2k_indices(
                            scores,
                            prefix_blocks,
                            video_blocks,
                            self.topk,
                        )
                with _h3_vsa_nvtx_stage("vsa.fine.block_sparse_attention"):
                    fine_op = _flashinfer_h3_vsa_q2k_pipeline_op if fine_pipeline else _flashinfer_h3_vsa_q2k_bshd_op
                    if tiled_output:
                        fine_op = _flashinfer_h3_vsa_q2k_tiled_output_op
                    output = fine_op(
                        q_tiled.contiguous(),
                        k_tiled.contiguous(),
                        v_tiled.contiguous(),
                        q2k_idx,
                        q2k_num,
                        kernel_sizes.contiguous(),
                        self.softmax_scale,
                    )
            else:
                logger.info_once(
                    "FASTVIDEO_VSA H3 metadata: legacy bool-map conversion active; "
                    "set %s=1 to qualify direct ordered q2k",
                    H3_VSA_DIRECT_Q2K_ENV,
                    scope="process",
                )
                with _h3_vsa_nvtx_stage("vsa.route.topk_q2k"):
                    block_map = _build_h3_block_map(scores, prefix_blocks, video_blocks, self.topk)
                with _h3_vsa_nvtx_stage("vsa.fine.block_sparse_attention"):
                    output = _flashinfer_h3_vsa_bshd_op(
                        q_tiled.contiguous(),
                        k_tiled.contiguous(),
                        v_tiled.contiguous(),
                        block_map.contiguous(),
                        kernel_sizes.contiguous(),
                        self.softmax_scale,
                    )
        else:
            with _h3_vsa_nvtx_stage("vsa.route.topk_q2k"):
                block_map = _build_h3_block_map(scores, prefix_blocks, video_blocks, self.topk)
                if pair_pad:
                    block_map = torch.nn.functional.pad(block_map, (0, 1, 0, 1), value=False)
                    kernel_sizes = torch.nn.functional.pad(sizes, (0, 1), value=0)
            with _h3_vsa_nvtx_stage("vsa.fine.block_sparse_attention"):
                output = _fastvideo_h3_vsa_bhsd_op(
                    q_tiled.contiguous(),
                    k_tiled.contiguous(),
                    v_tiled.contiguous(),
                    block_map.contiguous(),
                    kernel_sizes.contiguous(),
                    logical_blocks,
                )
        output = output[:, : logical_blocks * H3_VSA_EFFECTIVE_TILE_SIZE]

        if not fine_pipeline and (gate is not None or deferred_gate or o_bundle):
            compressed = compute_coarse()
        if deferred_gate:
            assert compressed is not None
            assert isinstance(deferred_state, dict)
            row_blocks = _get_h3_compact_row_blocks(
                prefix_segments,
                video_shape,
                output.device,
            )
            if row_blocks.numel() != query.shape[1]:
                raise ValueError(
                    "deferred H3 VSA row map must cover the exact valid sequence: "
                    f"map={row_blocks.numel()}, query={query.shape[1]}"
                )
            deferred_state["coarse"] = compressed
            deferred_state["row_blocks"] = row_blocks
            logger.info_once(
                "FASTVIDEO_VSA H3 deferred gate producer active: fine=%s, "
                "compact_coarse=%s, gate_owner=pre_Ulysses_sequence_shard",
                tuple(output.shape),
                tuple(compressed.shape),
                scope="process",
            )

        aligned_untile_source_rows: torch.Tensor | None = None
        o_input_landing = attn_metadata.extra.get("ulysses_o_input_landing")
        if o_input_landing is not None and not isinstance(
            o_input_landing,
            torch.Tensor,
        ):
            raise TypeError(f"ulysses_o_input_landing must be a torch.Tensor, got {type(o_input_landing)!r}")

        if o_bundle:
            assert compressed is not None
            assert owner_route is not None
            assert isinstance(o_bundle_state, dict)
            if fine_pipeline:
                from vllm_omni.diffusion.attention.ops.h3_overlap_experiment import publish_reverse

                source_rows = _get_h3_aligned_untile_source_rows(
                    prefix_segments, video_shape, aligned_rows, output.shape[1], output.device,
                )
                assert publish_reverse(output, compressed, owner_route, source_rows=source_rows)
                return output
            if tiled_output:
                from vllm_omni.diffusion.attention.ops.h3_tiled_o_producer import validate_source
                from vllm_omni.diffusion.attention.ops.h3_overlap_experiment import ACTIVE, publish_reverse
                source_rows = _get_h3_aligned_untile_source_rows(
                    prefix_segments, video_shape, aligned_rows, output.shape[1], output.device)
                validate_source(output, source_rows)
                active = ACTIVE.get()
                assert active is not None and "tiled_o_source_rows" not in active
                active["tiled_o_source_rows"] = source_rows
                assert publish_reverse(output, compressed, owner_route)
                return output
            with _h3_vsa_nvtx_stage("vsa.output.untile"):
                aligned_untile_source_rows = _get_h3_aligned_untile_source_rows(
                    prefix_segments,
                    video_shape,
                    aligned_rows,
                    output.shape[1],
                    output.device,
                )
                if use_fused_untile:
                    if not h3_vsa_tile_untile_cuda_supported(
                        output,
                        aligned_untile_source_rows,
                    ):
                        raise RuntimeError(
                            "H3 VSA reverse-O bundling requested with fused untile, "
                            "but the tile64 output geometry is unsupported"
                        )
                    fine_aligned = h3_vsa_tile_untile(
                        output,
                        aligned_untile_source_rows,
                    )
                    logger.info_once(
                        "FASTVIDEO_VSA H3 output untile: fused Triton tile64->aligned enabled "
                        "(valid_rows=%d, aligned_rows=%d, tiled_rows=%d, "
                        "destination=o_bundle_staging)",
                        untile.numel(),
                        aligned_rows,
                        output.shape[1],
                        scope="process",
                    )
                else:
                    fine_aligned = torch.zeros(
                        (
                            output.shape[0],
                            aligned_rows,
                            output.shape[2],
                            output.shape[3],
                        ),
                        dtype=output.dtype,
                        device=output.device,
                    )
                    fine_aligned[:, : untile.numel()].copy_(output[:, untile])

            from vllm_omni.diffusion.attention.ops.h3_overlap_experiment import publish_reverse

            if publish_reverse(fine_aligned, compressed, owner_route):
                return fine_aligned

            expected_bundle_shape = (
                fine_aligned.shape[0],
                owner_route.sp_world_size * (owner_route.local_rows + owner_route.kmax),
                fine_aligned.shape[2],
                fine_aligned.shape[3],
            )
            if o_input_landing is None:
                raise RuntimeError(
                    "H3 VSA reverse-O bundling requires a registered FlashInfer producer-direct O landing"
                )
            if tuple(o_input_landing.shape) != expected_bundle_shape:
                raise ValueError(
                    "H3 VSA reverse-O bundle landing shape mismatch: "
                    f"expected={expected_bundle_shape}, "
                    f"observed={tuple(o_input_landing.shape)}"
                )
            if not h3_vsa_o_bundle_cuda_supported(
                fine_aligned,
                compressed,
                owner_route,
            ):
                raise RuntimeError("H3 VSA reverse-O bundle reached an unsupported CUDA tensor contract")
            with _h3_vsa_nvtx_stage("vsa.output.o_bundle"):
                bundled = h3_vsa_o_bundle_out(
                    fine_aligned,
                    compressed,
                    owner_route,
                    out=o_input_landing,
                )
            o_bundle_state["published"] = True
            logger.info_once(
                "FASTVIDEO_VSA H3 reverse-O bundle producer active: fine=%s, "
                "compact_coarse=%s, bundle=%s, kmax=%d, destination=registered_reverse_o",
                tuple(fine_aligned.shape),
                tuple(compressed.shape),
                tuple(bundled.shape),
                owner_route.kmax,
                scope="process",
            )
            return bundled

        if use_fused_untile:
            aligned_untile_source_rows = _get_h3_aligned_untile_source_rows(
                prefix_segments,
                video_shape,
                aligned_rows,
                output.shape[1],
                output.device,
            )
            if (
                use_fused_gate_untile
                and gate is not None
                and compressed is not None
                and h3_vsa_gate_untile_cuda_supported(
                    output,
                    compressed,
                    gate,
                    aligned_untile_source_rows,
                )
            ):
                logger.info_once(
                    "FASTVIDEO_VSA H3 gate+untile: fused BF16 coarse-gate add "
                    "and tile64->aligned enabled (valid_rows=%d, aligned_rows=%d, "
                    "tiled_rows=%d, destination=%s)",
                    untile.numel(),
                    aligned_rows,
                    output.shape[1],
                    ("registered_reverse_o" if o_input_landing is not None else "functional"),
                    scope="process",
                )
                if o_input_landing is not None:
                    return h3_vsa_gate_untile_out(
                        output,
                        compressed,
                        gate,
                        aligned_untile_source_rows,
                        out=o_input_landing,
                    )
                return h3_vsa_gate_untile(
                    output,
                    compressed,
                    gate,
                    aligned_untile_source_rows,
                )
            if use_fused_gate_untile:
                logger.warning_once(
                    "%s=1 requested but the H3 tensor shape/platform is unsupported; "
                    "using separate gate-add and output un-tiling",
                    H3_VSA_FUSED_GATE_UNTILE_ENV,
                    scope="process",
                )

        if gate is not None:
            assert compressed is not None
            if use_fused_tile_pack:
                if gate_source_rows is None:
                    gate_source_rows = _get_h3_tiled_source_rows(
                        prefix_segments,
                        video_shape,
                        logical_blocks * H3_VSA_EFFECTIVE_TILE_SIZE,
                        query.device,
                    )
                if h3_vsa_tile_pack_cuda_supported(gate, gate_source_rows):
                    gate_tiled = h3_vsa_tile_pack(gate, gate_source_rows)
                else:
                    gate_tiled = torch.zeros_like(
                        q_tiled[:, : logical_blocks * H3_VSA_EFFECTIVE_TILE_SIZE]
                    )
                    gate_tiled[:, non_pad] = gate[:, partition]
            else:
                gate_tiled = torch.zeros_like(
                    q_tiled[:, : logical_blocks * H3_VSA_EFFECTIVE_TILE_SIZE]
                )
                gate_tiled[:, non_pad] = gate[:, partition]
            output = (
                output.view(
                    output.shape[0],
                    logical_blocks,
                    H3_VSA_EFFECTIVE_TILE_SIZE,
                    output.shape[2],
                    output.shape[3],
                )
                + compressed.unsqueeze(2)
                * gate_tiled.view(
                    gate_tiled.shape[0],
                    logical_blocks,
                    H3_VSA_EFFECTIVE_TILE_SIZE,
                    gate_tiled.shape[2],
                    gate_tiled.shape[3],
                )
            ).view_as(output)

        if use_fused_untile:
            assert aligned_untile_source_rows is not None
            if h3_vsa_tile_untile_cuda_supported(output, aligned_untile_source_rows):
                logger.info_once(
                    "FASTVIDEO_VSA H3 output untile: fused Triton tile64->aligned enabled "
                    "(valid_rows=%d, aligned_rows=%d, tiled_rows=%d, destination=%s)",
                    untile.numel(),
                    aligned_rows,
                    output.shape[1],
                    ("registered_reverse_o" if o_input_landing is not None else "functional"),
                    scope="process",
                )
                if o_input_landing is not None:
                    return h3_vsa_tile_untile_out(
                        output,
                        aligned_untile_source_rows,
                        out=o_input_landing,
                    )
                return h3_vsa_tile_untile(output, aligned_untile_source_rows)
            logger.warning_once(
                "FASTVIDEO_VSA H3 output untile fallback: %s=1 requested but the "
                "shape/platform is unsupported; using the reference "
                "tile64->compact->aligned path (valid_rows=%d, aligned_rows=%d, "
                "tiled_rows=%d)",
                H3_VSA_FUSED_UNTILE_ENV,
                untile.numel(),
                aligned_rows,
                output.shape[1],
                scope="process",
            )
        return output[:, untile].contiguous()

    def forward_cuda(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata | None = None,
    ) -> torch.Tensor:
        original_query, original_key, original_value = query, key, value
        fp8_qkv_transport_requested = (
            attn_metadata is not None
            and H3_VSA_FP8_QKV_SCALES_KEY in attn_metadata.extra
        )
        original_seq_len = query.shape[1]
        valid_seq_len = original_seq_len
        if attn_metadata is not None and attn_metadata.packed_padding is not None:
            valid_seq_len = attn_metadata.packed_padding.q_length
            if attn_metadata.packed_padding.kv_length != valid_seq_len:
                if fp8_qkv_transport_requested:
                    raise ValueError(
                        "H3 VSA FP8 QKV transport requires equal packed Q/KV lengths"
                    )
                return self._fallback(
                    original_query, original_key, original_value, attn_metadata, "packed Q/KV lengths must match"
                )
            query = query[:, :valid_seq_len]
            key = key[:, :valid_seq_len]
            value = value[:, :valid_seq_len]

        h3_layout = _get_h3_layout(attn_metadata)
        if fp8_qkv_transport_requested and h3_layout is None:
            raise ValueError(
                "H3 VSA FP8 QKV transport requires complete H3 prefix/video layout metadata"
            )
        if attn_metadata is not None and h3_layout is not None:
            try:
                output = self._forward_h3(
                    query,
                    key,
                    value,
                    attn_metadata,
                    original_seq_len,
                )
                o_bundle = attn_metadata.extra.get(
                    H3_VSA_O_BUNDLE_ACTIVE_KEY,
                    False,
                )
                if o_bundle is True:
                    state = attn_metadata.extra.get(H3_VSA_O_BUNDLE_STATE_KEY)
                    plan = state.get("plan") if isinstance(state, dict) else None
                    if not isinstance(plan, H3VSAOwnerRoutePlan):
                        raise RuntimeError("H3 VSA reverse-O bundle plan disappeared after producer execution")
                    from vllm_omni.diffusion.attention.ops.h3_overlap_experiment import ACTIVE

                    overlap_state = ACTIVE.get()
                    if overlap_state is not None:
                        fine, _coarse, owner_plan = overlap_state["reverse"]
                        expected_fine_rows = 105472 if "v16_fine" in overlap_state else original_seq_len
                        if "tiled_o_source_rows" in overlap_state:
                            from vllm_omni.diffusion.attention.ops.h3_tiled_o_producer import validate_source
                            validate_source(output, overlap_state["tiled_o_source_rows"])
                            expected_fine_rows = 105408
                        if output is not fine or owner_plan is not plan or output.shape[1] != expected_fine_rows:
                            raise RuntimeError("H3 chunked reverse-O producer lost its fine/route identity")
                        return output
                    expected_bundle_rows = plan.sp_world_size * (plan.local_rows + plan.kmax)
                    if output.shape[1] != expected_bundle_rows:
                        raise ValueError(
                            "H3 VSA reverse-O producer returned the wrong bundle rows: "
                            f"got={output.shape[1]}, expected={expected_bundle_rows}"
                        )
                    return output
                if output.shape[1] == original_seq_len:
                    return output
                if output.shape[1] != valid_seq_len:
                    raise ValueError(
                        "VSA-H3 output must contain either the valid or aligned "
                        f"sequence, got {output.shape[1]} rows for "
                        f"valid={valid_seq_len}, aligned={original_seq_len}"
                    )
                restored = torch.zeros(
                    original_query.shape,
                    dtype=output.dtype,
                    device=output.device,
                )
                restored[:, :valid_seq_len] = output
                return restored
            except Exception as exc:
                # A CUDA fault poisons the process context; attempting SDPA
                # afterwards obscures the original kernel failure and cannot
                # recover the request.
                # Deferred gate ownership is also a transactional contract:
                # once Ulysses retains the local gate, only the H3 VSA
                # producer may publish the matching compact coarse result.
                # Falling back after (or before) publication would either
                # combine a dense fine output with VSA coarse state or leave
                # post_attention without a producer acknowledgement.
                deferred_gate = attn_metadata.extra.get(
                    H3_VSA_DEFERRED_ACTIVE_KEY,
                    False,
                )
                o_bundle = attn_metadata.extra.get(
                    H3_VSA_O_BUNDLE_ACTIVE_KEY,
                    False,
                )
                if (
                    deferred_gate is True
                    or o_bundle is True
                    or fp8_qkv_transport_requested
                    or self.h3_kernel_backend == "flashinfer"
                    or isinstance(exc, torch.AcceleratorError)
                ):
                    raise
                if not self.fallback_on_error:
                    raise
                return self._fallback(
                    original_query, original_key, original_value, attn_metadata, f"VSA-H3 kernel failed: {exc}"
                )

        reason = self._fallback_reason(query, key, value, attn_metadata)
        if reason is not None:
            return self._fallback(original_query, original_key, original_value, attn_metadata, reason)

        seq_len = query.shape[1]
        dit_seq_shape = _get_vsa_dit_seq_shape(attn_metadata)
        assert dit_seq_shape is not None
        num_blocks = math.prod(
            math.ceil(seq_dim / tile_dim) for seq_dim, tile_dim in zip(dit_seq_shape, self.block_size)
        )
        preserve_all_blocks = _preserve_vsa_all_blocks(attn_metadata)
        use_native_sdpa = self.topk == num_blocks and not preserve_all_blocks
        route = "SDPA" if use_native_sdpa else "VSA_ALL_BLOCKS" if self.topk == num_blocks else "VSA"
        checkpoint_mode = "fastvideo_dmd" if preserve_all_blocks else "native"
        logger.info_once(
            "FASTVIDEO_VSA routing: seq_len=%d, dit_seq_shape=%s, block_size=%s, num_blocks=%d, "
            "topk=%d, keep_ratio=%.1f%%, checkpoint_mode=%s, route=%s",
            seq_len,
            dit_seq_shape,
            self.block_size,
            num_blocks,
            self.topk,
            100.0 * self.topk / num_blocks,
            checkpoint_mode,
            route,
        )
        if use_native_sdpa:
            return self._fallback(
                query,
                key,
                value,
                attn_metadata,
                f"topk {self.topk} selects all blocks for a native checkpoint",
            )

        try:
            tile_partition_indices, variable_block_sizes, non_pad_index, untile_combined_index = _get_tile_metadata(
                dit_seq_shape,
                self.block_size,
                self.block_elements,
                query.device,
            )

            padded_len = variable_block_sizes.numel() * self.block_elements
            target_shape = (query.shape[0], padded_len, query.shape[2], query.shape[3])
            query_tiled = torch.zeros(target_shape, device=query.device, dtype=query.dtype)
            key_tiled = torch.zeros_like(query_tiled)
            value_tiled = torch.zeros_like(query_tiled)
            query_tiled[:, non_pad_index] = query[:, tile_partition_indices]
            key_tiled[:, non_pad_index] = key[:, tile_partition_indices]
            value_tiled[:, non_pad_index] = value[:, tile_partition_indices]
            # Gate behavior is checkpoint-driven, not user-configured.
            # Wan VSA layers always provide a gate projection. Its zero
            # initialization makes checkpoints without gate weights sparse-only;
            # checkpoints containing to_gate_compress weights use the learned gate.
            gate_compress = _get_gate_compress(attn_metadata)
            if gate_compress is None:
                gate_compress = torch.zeros_like(query)
            elif valid_seq_len != original_seq_len:
                gate_compress = gate_compress[:, :valid_seq_len]
            elif gate_compress.shape != query.shape:
                raise ValueError(f"gate_compress shape {gate_compress.shape} must match query shape {query.shape}")
            gate_tiled = torch.zeros_like(query_tiled)
            gate_tiled[:, non_pad_index] = gate_compress[:, tile_partition_indices]
            compress_attn_weight = gate_tiled

            output = _fastvideo_vsa_bshd_op(
                query_tiled.contiguous(),
                key_tiled.contiguous(),
                value_tiled.contiguous(),
                variable_block_sizes,
                variable_block_sizes,
                compress_attn_weight,
                self.topk,
                self.block_size[0],
                self.block_size[1],
                self.block_size[2],
            )
            output = output[:, untile_combined_index].contiguous()
            if valid_seq_len == original_seq_len:
                return output
            restored = torch.zeros(
                (query.shape[0], original_seq_len, query.shape[2], query.shape[3]),
                device=query.device,
                dtype=query.dtype,
            )
            restored[:, :valid_seq_len] = output
            return restored
        except Exception as exc:
            if not self.fallback_on_error:
                raise
            return self._fallback(
                original_query, original_key, original_value, attn_metadata, f"VSA kernel failed: {exc}"
            )

    def forward_npu(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata | None = None,
    ) -> torch.Tensor:
        return self.sdpa_fallback.forward_npu(query, key, value, attn_metadata)

    def forward_xpu(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata | None = None,
    ) -> torch.Tensor:
        return self.sdpa_fallback.forward_xpu(query, key, value, attn_metadata)

    def forward_musa(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata | None = None,
    ) -> torch.Tensor:
        return self.sdpa_fallback.forward_musa(query, key, value, attn_metadata)
