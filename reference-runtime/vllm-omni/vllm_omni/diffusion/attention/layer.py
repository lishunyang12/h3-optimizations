# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) Microsoft Corporation and Jiarui Fang
# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team & Jiarui Fang
# Adapted from
# https://github.com/feifeibear/long-context-attention/blob/main/yunchang/attention/layer.py


import threading
from collections.abc import Callable
from dataclasses import replace

import torch
import torch.distributed as dist
import torch.nn as nn
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.logger import init_logger
from vllm.model_executor.models.utils import extract_layer_index
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheSpec

from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata
from vllm_omni.diffusion.attention.backends.sdpa import SDPABackend
from vllm_omni.diffusion.attention.parallel import build_parallel_attention_strategy
from vllm_omni.diffusion.attention.parallel.base import NoParallelAttention
from vllm_omni.diffusion.attention.parallel.ring import RingParallelAttention
from vllm_omni.diffusion.attention.selector import get_attn_backend_for_role
from vllm_omni.diffusion.config import get_current_diffusion_config_or_none
from vllm_omni.diffusion.distributed.parallel_state import get_sp_group
from vllm_omni.diffusion.forward_context import (
    get_forward_context,
    get_ulysses_mode,
    is_forward_context_available,
)
from vllm_omni.platforms import current_omni_platform

logger = init_logger(__name__)

_H3_QCHUNK_STREAMS: dict[int, torch.cuda.Stream] = {}
_H3_QCHUNK_STREAM_LOCK = threading.Lock()


def _h3_qchunk_compute_stream(device: torch.device) -> torch.cuda.Stream:
    """Share one attention stream per device across all 50 H3 blocks."""
    index = device.index
    if index is None:
        index = torch.accelerator.current_device_index()
    with _H3_QCHUNK_STREAM_LOCK:
        stream = _H3_QCHUNK_STREAMS.get(index)
        if stream is None:
            stream = torch.cuda.Stream(device=device)
            _H3_QCHUNK_STREAMS[index] = stream
        return stream


def _h3_query_chunk_major(
    query: torch.Tensor,
    *,
    world_size: int,
    chunks: int,
) -> torch.Tensor:
    """Group the same local-sequence chunk from every SP origin.

    Strict Ulysses Q arrives as ``[origin, local_sequence, local_heads, D]``
    flattened into its sequence dimension.  An inverse Ulysses exchange can
    consume an attention result one query chunk at a time only when that chunk
    contains the corresponding rows from *all* origins.  This permutation
    creates exactly that layout; a later native transport optimization can
    write it directly as the Q-scatter destination layout.
    """
    if query.ndim != 4:
        raise ValueError(f"H3 query must be [B,S,H,D], got {tuple(query.shape)}")
    if world_size <= 1 or chunks <= 1:
        raise ValueError("H3 query chunking requires world_size and chunks > 1")
    batch, global_seq, local_heads, head_dim = query.shape
    if global_seq % world_size:
        raise ValueError(f"global sequence {global_seq} is not divisible by SP world {world_size}")
    local_seq = global_seq // world_size
    if local_seq % chunks:
        raise ValueError(f"local sequence {local_seq} is not divisible by chunk count {chunks}")
    chunk_seq = local_seq // chunks
    return (
        query.view(
            batch,
            world_size,
            chunks,
            chunk_seq,
            local_heads,
            head_dim,
        )
        .permute(0, 2, 1, 3, 4, 5)
        .contiguous()
        .view(batch, global_seq, local_heads, head_dim)
    )


def _h3_query_origin_major(
    query: torch.Tensor,
    *,
    world_size: int,
    chunks: int,
) -> torch.Tensor:
    """Undo :func:`_h3_query_chunk_major` for a monolithic fallback."""
    if query.ndim != 4:
        raise ValueError(f"H3 query must be [B,S,H,D], got {tuple(query.shape)}")
    if world_size <= 1 or chunks <= 1:
        raise ValueError("H3 query unchunking requires world_size and chunks > 1")
    batch, global_seq, local_heads, head_dim = query.shape
    if global_seq % (world_size * chunks):
        raise ValueError(f"global sequence {global_seq} does not divide SP{world_size} x {chunks} chunks")
    chunk_seq = global_seq // (world_size * chunks)
    return (
        query.view(
            batch,
            chunks,
            world_size,
            chunk_seq,
            local_heads,
            head_dim,
        )
        .permute(0, 2, 1, 3, 4, 5)
        .contiguous()
        .view(batch, global_seq, local_heads, head_dim)
    )


def _try_extract_layer_index(prefix: str) -> int | None:
    if not prefix:
        return None
    try:
        return extract_layer_index(prefix)
    except (AssertionError, ValueError):
        return None


class Attention(nn.Module):
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        causal: bool,
        softmax_scale: float,
        num_kv_heads: int | None = None,
        prefix: str = "",
        # Per-role backend selection (RFC: per-role attention backend)
        role: str = "self",
        role_category: str | None = None,
        # Model-defined Q/K/V tensor layout hint for backend execution.
        qkv_layout: str | None = None,
        # ulysses attention
        scatter_idx: int = 2,
        gather_idx: int = 1,
        use_sync: bool = False,
        skip_sequence_parallel: bool = False,
        # Opt-out for KV-cache quantization at this specific attention layer.
        # Set by the model author when quant is known to degrade quality or
        # perf for this layer (e.g. Wan2.2 cross-attn has short sequences and
        # block-FP8 quant offers no win). Default False = follow global config.
        disable_kv_quant: bool = False,
        # Opt-in marker for Scheduler-managed paged KV. Unmarked diffusion
        # attention remains dense and contributes no native KVCacheSpec.
        paged_kv_cache_role: str | None = None,
        paged_kv_cache_dtype: torch.dtype | None = None,
    ):
        super().__init__()

        self.role = role
        self.role_category = role_category
        self.qkv_layout = qkv_layout
        # ``prefix`` is also the stable layer identity used by vLLM's native
        # KV-cache metadata.  Keep it on the Omni layer so the active paged
        # adapter can dispatch the already-resharded Q/K/V to the matching
        # native cache tensor without replacing this Omni execution path.
        self.prefix = prefix
        if paged_kv_cache_role == "":
            raise ValueError("paged_kv_cache_role must be non-empty when provided")
        self.paged_kv_cache_role = paged_kv_cache_role
        self.paged_kv_cache_dtype = paged_kv_cache_dtype
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.head_size = head_size

        # Resolve backend via role-aware config.
        # The global diffusion config is set during model init via
        # set_current_diffusion_config(); no env-var re-parsing needed here.
        backend_kwargs: dict | None = None
        self.backend_pref = None

        config = get_current_diffusion_config_or_none()
        attention_config = config.diffusion_attention_config if config is not None else None

        from vllm_omni.diffusion.model_metadata import get_diffusion_model_metadata

        model_class_name = getattr(config, "model_class_name", None) if config is not None else None
        allow_trtllm_default = get_diffusion_model_metadata(model_class_name).attention_mask_free

        attn_backend_cls, spec = get_attn_backend_for_role(
            role=role,
            head_size=head_size,
            attention_config=attention_config,
            role_category=role_category,
            allow_trtllm_default=allow_trtllm_default,
        )
        parallel_config = getattr(config, "parallel_config", None)
        allgather_degree = getattr(parallel_config, "allgather_degree", 1)
        # TODO: Move AllGather-KV compatibility into an AttentionBackend capability
        # so validation does not depend on backend names.
        if not skip_sequence_parallel and allgather_degree > 1 and attn_backend_cls.get_name() == "TRTLLM_ATTN":
            raise ValueError(
                "TRTLLM_ATTN does not support AllGather-KV sequence parallelism. "
                "Set --allgather-degree 1 or select another diffusion attention backend."
            )
        if spec is not None:
            backend_kwargs = spec.backend_kwargs()
            self.backend_pref = spec.backend
            logger.debug("Attention(role=%s) → backend=%s", role, spec.backend)
        else:
            logger.debug("Attention(role=%s) → platform default", role)

        self.attn_backend = attn_backend_cls
        self.attn_impl_cls = self.attn_backend.get_impl_cls()
        self.attention = self.attn_impl_cls(
            num_heads=num_heads,
            head_size=head_size,
            softmax_scale=softmax_scale,
            causal=causal,
            num_kv_heads=num_kv_heads,
            qkv_layout=qkv_layout,
            prefix=prefix,
            backend_kwargs=backend_kwargs,
            role=role,
        )
        # Instantiate fallback backend for float32 support
        self.sdpa_fallback = SDPABackend.get_impl_cls()(
            num_heads=num_heads,
            head_size=head_size,
            softmax_scale=softmax_scale,
            causal=causal,
            num_kv_heads=num_kv_heads,
            qkv_layout=qkv_layout,
        )

        self.softmax_scale = softmax_scale
        self.scatter_idx = scatter_idx
        self.gather_idx = gather_idx
        self.use_sync = use_sync
        self.causal = causal
        self.skip_sequence_parallel = skip_sequence_parallel

        self.use_ring = False
        self.ring_pg = None
        self.ring_runner = None

        if config is not None:
            if config.parallel_config.ring_degree > 1:
                self.use_ring = True
                try:
                    sp_group = get_sp_group()
                    self.ring_pg = sp_group.ring_group
                    self.ring_runner = RingParallelAttention(
                        sp_group,
                        attn_backend_pref=self.backend_pref,
                    )
                except Exception:
                    self.use_ring = False
                    self.ring_runner = None

        self.parallel_strategy = build_parallel_attention_strategy(
            scatter_idx=scatter_idx,
            gather_idx=gather_idx,
            use_sync=use_sync,
            causal=causal,
        )
        # Fallback strategy when SP is not active (outside sharded regions)
        self._no_parallel_strategy = NoParallelAttention()

        self.layer_idx: int | None = _try_extract_layer_index(prefix)

        self._kv_cache_dtype: str | None = None
        self._kv_cache_skip_steps: set[int] | None = None
        self._kv_cache_skip_layers: set[int] | None = None
        # Per-layer opt-out from KV-cache quantization (set by model author).
        self._disable_kv_quant: bool = disable_kv_quant
        self._init_kv_cache_quantization(config)

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec | None:
        """Return native rank-local geometry for an opted-in paged cache."""

        if self.paged_kv_cache_role is None:
            return None
        dtype = self.paged_kv_cache_dtype or vllm_config.model_config.dtype
        # Keep backend layout discovery under the same config context used by
        # upstream vLLM's attention-spec collector.
        with set_current_vllm_config(vllm_config):
            indexes_kv_by_block_stride = self.attn_backend.indexes_kv_by_block_stride()
        return FullAttentionSpec(
            block_size=vllm_config.cache_config.block_size,
            num_kv_heads=self.num_kv_heads,
            head_size=self.head_size,
            dtype=dtype,
            indexes_kv_by_block_stride=indexes_kv_by_block_stride,
            non_causal=not self.causal,
        )

    def _get_active_parallel_strategy(self):
        """Get the parallel strategy based on current SP active state.

        Returns NoParallelAttention if we're outside an SP sharded region
        (e.g., in noise_refiner/context_refiner before unified_prepare in Z-Image).
        This avoids unnecessary SP communication for layers not covered by _sp_plan.
        """
        if self.skip_sequence_parallel:
            return self._no_parallel_strategy
        if is_forward_context_available():
            ctx = get_forward_context()
            if not ctx.sp_active:
                return self._no_parallel_strategy
        return self.parallel_strategy

    @property
    def supports_qk_input_landing(self) -> bool:
        """Expose the static producer-direct capability to model layers."""
        if self.skip_sequence_parallel:
            return False
        return bool(getattr(self.parallel_strategy, "supports_qk_input_landing", False))

    @property
    def supports_qkv_e4m3_input_landing(self) -> bool:
        """Expose the opt-in E4M3 QKV producer-direct capability."""
        if self.skip_sequence_parallel:
            return False
        return bool(
            getattr(
                self.parallel_strategy,
                "supports_qkv_e4m3_input_landing",
                False,
            )
        )

    @torch.compiler.disable
    def prepare_qk_input_landings(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        *,
        q_chunk_major_chunks: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Acquire Q/K landing buffers only from the existing eager island."""
        strategy = self._get_active_parallel_strategy()
        prepare = getattr(strategy, "prepare_qk_input_landings", None)
        if prepare is None:
            return None
        return prepare(
            query,
            key,
            q_chunk_major_chunks=q_chunk_major_chunks,
        )

    @torch.compiler.disable
    def prepare_qkv_e4m3_input_landings(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
        """Acquire registered E4M3 sources from the existing eager island."""
        strategy = self._get_active_parallel_strategy()
        prepare = getattr(strategy, "prepare_qkv_e4m3_input_landings", None)
        if prepare is None:
            return None
        return prepare(query, key, value)

    def _init_kv_cache_quantization(self, config) -> None:
        if config is None:
            return
        dtype = getattr(config, "diffusion_kv_cache_dtype", None)
        if dtype == "auto":
            dtype = None
        parallel_config = getattr(config, "parallel_config", None)
        ring_degree = getattr(parallel_config, "ring_degree", 1)
        if dtype:
            if ring_degree > 1:
                raise ValueError(
                    "KV quantization is not compatible with ring attention "
                    "(ring_degree > 1). Ring kernels do not propagate quantization descale "
                    "factors. Use Ulysses SP instead."
                )
            platform_key = current_omni_platform.device_name
            if not self.attention.supports_kv_cache_dtype(dtype, platform_key):
                logger.warning_once(
                    "Attention backend %s does not support kv_cache_dtype='%s' on %s. "
                    "KV quantization will be disabled.",
                    self.attn_backend.get_name(),
                    dtype,
                    platform_key,
                )
                dtype = None
        self._kv_cache_dtype = dtype
        self._kv_cache_skip_steps = getattr(config, "diffusion_kv_cache_skip_step_indices", None)
        self._kv_cache_skip_layers = getattr(config, "diffusion_kv_cache_skip_layer_indices", None)

    def _should_apply_kv_cache_quant(self) -> bool:
        skip_steps = self._kv_cache_skip_steps
        skip_layers = self._kv_cache_skip_layers
        if skip_steps is not None:
            step_idx = get_forward_context().denoise_step_idx if is_forward_context_available() else None
            if step_idx is not None and step_idx in skip_steps:
                return False
        if skip_layers is not None:
            if self.layer_idx is not None and self.layer_idx in skip_layers:
                return False
        return True

    def _with_kv_cache_dtype(self, attn_metadata: AttentionMetadata | None) -> AttentionMetadata | None:
        kv_cache_dtype = self._kv_cache_dtype
        if kv_cache_dtype is None or self._disable_kv_quant or not self._should_apply_kv_cache_quant():
            if attn_metadata is None or "kv_cache_dtype" not in attn_metadata.extra:
                return attn_metadata
            extra = dict(attn_metadata.extra)
            extra.pop("kv_cache_dtype", None)
            return replace(attn_metadata, extra=extra)

        if attn_metadata is None:
            return AttentionMetadata(extra={"kv_cache_dtype": kv_cache_dtype})
        extra = dict(attn_metadata.extra)
        extra["kv_cache_dtype"] = kv_cache_dtype
        return replace(attn_metadata, extra=extra)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata | None = None,
    ) -> torch.Tensor:
        if torch.compiler.is_compiling() and is_forward_context_available():
            od_config = get_forward_context().omni_diffusion_config
            parallel_config = getattr(od_config, "parallel_config", None)
            if getattr(parallel_config, "use_hsdp", False):
                # Keep HSDP/FSDP2 parameter all-gather outside Inductor's
                # attention graph; otherwise scheduler dependency analysis can
                # fail on the fused attention region.
                return self._forward_hsdp_compile_boundary(query, key, value, attn_metadata)

        return self._forward_impl(query, key, value, attn_metadata)

    @torch.compiler.disable
    def _forward_hsdp_compile_boundary(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata | None = None,
    ) -> torch.Tensor:
        return self._forward_impl(query, key, value, attn_metadata)

    def _forward_impl(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata | None = None,
    ) -> torch.Tensor:
        # Get the appropriate parallel strategy based on SP active state
        strategy = self._get_active_parallel_strategy()
        paged_adapter = self._active_paged_kv_adapter()
        use_paged_attention = paged_adapter is not None and self.paged_kv_cache_role is not None
        if use_paged_attention and not getattr(self.attn_backend, "supports_paged_kv", False):
            raise NotImplementedError(
                f"Diffusion paged KV requires an Omni backend with paged support; "
                f"selected {self.attn_backend.get_name()}"
            )
        if use_paged_attention and strategy is not self._no_parallel_strategy:
            strategy_name = strategy.name
            if self.use_ring or strategy_name == "ring":
                raise NotImplementedError(
                    "paged Scheduler KV is not supported with Ring attention; use strict Ulysses or no SP"
                )
            if strategy_name == "allgather_kv":
                raise NotImplementedError("paged Scheduler KV is not supported with AllGather-KV sequence parallelism")
            if strategy_name == "ulysses" and get_ulysses_mode(default="strict") != "strict":
                raise NotImplementedError("paged Scheduler KV currently supports only strict Ulysses")

        # 1. Prepare inputs (Communication / Resharding)
        # For Ulysses: AllToAll Q/K/V; Slicing joint_q/k/v
        # For Ring: Concat joint_q
        query, key, value, attn_metadata, ctx = strategy.pre_attention(query, key, value, attn_metadata)

        # 2. Kernel execution stays inside the selected Omni backend.  The
        # Worker adapter only prepares the native page-table context after
        # SP has produced rank-local Q/K/V tensors.
        if use_paged_attention:
            assert paged_adapter is not None
            paged_kv_context = paged_adapter.prepare_layer_context(
                self.prefix,
                query,
                key,
                value,
                omni_attn_metadata=attn_metadata,
            )
            out = self.attention.forward_paged(paged_kv_context)
        else:
            attn_metadata = self._with_kv_cache_dtype(attn_metadata)
            if self.use_ring and strategy is not self._no_parallel_strategy:
                out = self._run_ring_attention(query, key, value, attn_metadata)
            else:
                out = self._run_local_attention(query, key, value, attn_metadata)

        # 3. Post-processing (Reverse Communication)
        # For Ulysses: AllToAll Output, and AllGather Joint Output
        out = strategy.post_attention(out, ctx)

        return out

    @torch.compiler.disable
    def forward_h3_qchunk_projected(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata,
        *,
        chunks: int,
        output_projector: Callable[[torch.Tensor], torch.Tensor],
    ) -> torch.Tensor:
        """Pipeline cuDNN query chunks with inverse Ulysses and projection.

        This is a deliberately narrow MiniMax-H3 fast path.  Its fallback
        executes the ordinary attention graph and then calls the identical
        output projector, so enabling the experiment never silently changes
        unsupported backends, SP modes, masks, batches, or dtypes.

        The FlashInfer RDMA communicator is bound to the caller stream by the
        Q/K/V collectives.  Attention chunks run on a private compute stream;
        the caller stream waits only for the next completed chunk, performs
        its O gather, and immediately projects it while later query attention
        remains queued.  Each projected chunk owns its output, avoiding reuse
        hazards from FlashInfer's single registered O slot.
        """

        def project_fallback() -> torch.Tensor:
            return output_projector(self._forward_impl(query, key, value, attn_metadata))

        strategy = self._get_active_parallel_strategy()
        static_reason = None
        if query.device.type != "cuda":
            static_reason = "CUDA tensors are required"
        elif chunks != 2:
            static_reason = "only the qualified two-chunk contract is supported"
        elif self.attn_backend.get_name() != "CUDNN_ATTN":
            static_reason = f"backend is {self.attn_backend.get_name()}, not CUDNN_ATTN"
        elif self.use_ring:
            static_reason = "Ring attention is active"
        elif strategy is self._no_parallel_strategy or strategy.name != "ulysses":
            static_reason = f"parallel strategy is {strategy.name}, not Ulysses"
        elif self._active_paged_kv_adapter() is not None:
            static_reason = "paged KV is active"
        elif get_ulysses_mode(default="strict") != "strict":
            static_reason = "Ulysses mode is not strict"
        if static_reason is not None:
            logger.warning_once(
                "MiniMax H3 two-chunk attention pipeline fell back: %s",
                static_reason,
            )
            return project_fallback()

        # Q/K/V communication remains exactly the production strict-Ulysses
        # path.  Only after it succeeds do we inspect its concrete context and
        # decide whether the chunk pipeline contract is satisfied.
        query_sp, key_sp, value_sp, metadata_sp, ctx = strategy.pre_attention(
            query,
            key,
            value,
            attn_metadata,
            q_chunk_major_chunks=chunks,
        )
        metadata_sp = self._with_kv_cache_dtype(metadata_sp)

        def project_after_pre_attention() -> torch.Tensor:
            query_fallback = query_sp
            direct_chunks = int(getattr(ctx, "q_chunk_major_chunks", 0))
            if direct_chunks:
                query_fallback = _h3_query_origin_major(
                    query_sp,
                    world_size=world_size,
                    chunks=direct_chunks,
                )
                if bool(getattr(ctx, "qchunk2_direct_experiment", False)):
                    from vllm_omni.diffusion.distributed.flashinfer_ulysses import (
                        record_qchunk2_full_permute,
                    )

                    record_qchunk2_full_permute(
                        query_fallback,
                        ctx.ulysses_pg.group_name,
                        world_size,
                    )
            out = self._run_local_attention(
                query_fallback,
                key_sp,
                value_sp,
                metadata_sp,
            )
            return output_projector(strategy.post_attention(out, ctx))

        group = getattr(ctx, "ulysses_pg", None)
        world_size = dist.get_world_size(group) if group is not None else 0
        dynamic_reason = None
        if getattr(ctx, "strict_a2a_backend", "") != "flashinfer-pcie":
            dynamic_reason = "strict Ulysses did not select flashinfer-pcie"
        elif bool(getattr(ctx, "use_uaa", False)):
            dynamic_reason = "advanced UAA context is active"
        elif int(getattr(ctx, "joint_len", 0)) != 0:
            dynamic_reason = "joint attention rows are active"
        elif world_size != 8:
            dynamic_reason = f"qualified SP world is 8, got {world_size}"
        elif metadata_sp is None or metadata_sp.attn_mask is not None:
            dynamic_reason = "mask-free cuDNN metadata is required"
        elif query_sp.dtype != torch.bfloat16:
            dynamic_reason = f"qualified dtype is BF16, got {query_sp.dtype}"
        elif query_sp.shape != key_sp.shape or query_sp.shape != value_sp.shape:
            dynamic_reason = "Q/K/V shapes differ"
        elif query_sp.ndim != 4 or query_sp.shape[0] != 1:
            dynamic_reason = f"qualified batch shape is [1,S,H,D], got {tuple(query_sp.shape)}"
        elif query_sp.shape[2:] != (7, 128):
            dynamic_reason = f"qualified local head geometry is [7,128], got {tuple(query_sp.shape[2:])}"
        elif query_sp.shape[1] % (world_size * chunks):
            dynamic_reason = (
                f"global sequence {query_sp.shape[1]} does not split evenly across SP{world_size} x {chunks} chunks"
            )
        if dynamic_reason is not None:
            logger.warning_once(
                "MiniMax H3 two-chunk attention pipeline fell back after QKV: %s",
                dynamic_reason,
            )
            return project_after_pre_attention()

        comm_stream = torch.cuda.current_stream(query_sp.device)
        input_ready = torch.cuda.Event()
        input_ready.record(comm_stream)
        compute_stream = _h3_qchunk_compute_stream(query_sp.device)

        global_seq = query_sp.shape[1]
        global_chunk = global_seq // chunks
        attention_chunks: list[torch.Tensor] = []
        ready_events: list[torch.cuda.Event] = []
        with torch.cuda.stream(compute_stream):
            compute_stream.wait_event(input_ready)
            if int(getattr(ctx, "q_chunk_major_chunks", 0)) == chunks:
                query_chunk_major = query_sp
            else:
                query_chunk_major = _h3_query_chunk_major(
                    query_sp,
                    world_size=world_size,
                    chunks=chunks,
                )
                if bool(getattr(ctx, "qchunk2_direct_experiment", False)):
                    from vllm_omni.diffusion.distributed.flashinfer_ulysses import (
                        record_qchunk2_full_permute,
                    )

                    record_qchunk2_full_permute(
                        query_chunk_major,
                        ctx.ulysses_pg.group_name,
                        world_size,
                    )
            for index in range(chunks):
                start = index * global_chunk
                query_chunk = query_chunk_major.narrow(1, start, global_chunk)
                attention_chunks.append(
                    self._run_local_attention(
                        query_chunk,
                        key_sp,
                        value_sp,
                        metadata_sp,
                    )
                )
                ready = torch.cuda.Event()
                ready.record(compute_stream)
                ready_events.append(ready)

        projected_chunks: list[torch.Tensor] = []
        for attention_chunk, ready in zip(
            attention_chunks,
            ready_events,
            strict=True,
        ):
            comm_stream.wait_event(ready)
            # The chunk is allocated on ``compute_stream`` but its inverse
            # exchange consumes it asynchronously on ``comm_stream``.  Keep
            # the caching allocator from recycling that storage after the
            # producer stream completes but before the communicator has
            # finished reading it.
            attention_chunk.record_stream(comm_stream)
            gathered = strategy.post_attention(attention_chunk, ctx)
            projected_chunks.append(output_projector(gathered))

        result = torch.cat(projected_chunks, dim=0)
        if not getattr(self, "_h3_qchunk_logged", False):
            logger.info(
                "MiniMax H3 two-chunk attention/O-gather/out-projection pipeline active: layer=%s shape=%s",
                self.prefix,
                tuple(query_sp.shape),
            )
            self._h3_qchunk_logged = True
        return result

    @staticmethod
    def _active_paged_kv_adapter():
        """Return the opaque Worker adapter installed for this forward."""

        if not is_forward_context_available():
            return None
        return getattr(get_forward_context(), "paged_kv_adapter", None)

    def is_paged_kv_active(self) -> bool:
        """Return whether this layer will use Scheduler-managed paged KV."""

        return self.paged_kv_cache_role is not None and self._active_paged_kv_adapter() is not None

    def _run_local_attention(self, query, key, value, attn_metadata):
        self._assert_piecewise_compatible(attn_metadata)

        if query.dtype == torch.float32:
            logger.warning_once(
                f"Only SDPA supports float32. Overriding user config {type(self.attention)} "
                f"attention_backend='{self.backend_pref}' to 'sdpa' for dtype={query.dtype}."
            )
            return self.sdpa_fallback.forward(query, key, value, attn_metadata)

        # Fallback to standard attention
        return self.attention.forward(query, key, value, attn_metadata)

    def _assert_piecewise_compatible(self, attn_metadata: AttentionMetadata | None) -> None:
        if attn_metadata is None or attn_metadata.full_attn_spans is None:
            return
        if attn_metadata.attn_mask is not None and attn_metadata.attn_mask.ndim == 4:
            return
        backend_name = self.attn_backend.get_name()
        if not self.attn_backend.supports_piecewise_spans:
            raise ValueError(
                f"Attention backend '{backend_name}' does not support "
                f"piecewise attention (full_attn_spans without a 4D attn_mask). "
                f"Use a Flash backend (FLASH_ATTN / FLASH_ATTN_HUB / FLASH_ATTN_3_HUB), "
                f"or provide a 4D attn_mask that encodes the mixed causal/full pattern."
            )

    def _run_ring_attention(self, query, key, value, attn_metadata):
        skip = getattr(self.attention, "skip", None)
        if skip is not None and getattr(skip, "configured", False):
            raise NotImplementedError(
                "Skip-Softmax (TRTLLM_ATTN) is not supported with ring sequence parallelism: "
                "the ring path bypasses the backend, so the skip config would be silently ignored. "
                "Use Ulysses SP instead, or remove the skip_softmax config."
            )
        # Delegate to RingParallelAttention strategy if available
        if self.ring_runner is not None:
            return self.ring_runner.run_attention(
                query, key, value, attn_metadata, softmax_scale=self.softmax_scale, causal=self.causal
            )

        raise RuntimeError("Ring attention is enabled but strategy is not RingParallelAttention")
