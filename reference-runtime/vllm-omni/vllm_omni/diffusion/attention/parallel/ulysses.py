# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import math
import os
from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.nn.functional as F
from vllm.logger import init_logger

from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata
from vllm_omni.diffusion.attention.ops.minimax_h3_vsa_gate_compute_overlap import (
    H3VSAGateComputeOverlapTicket,
)
from vllm_omni.diffusion.attention.parallel.base import ParallelAttentionContext
from vllm_omni.diffusion.distributed.comm import SeqAllToAll4D
from vllm_omni.diffusion.distributed.group_coordinator import SequenceParallelGroupCoordinator
from vllm_omni.diffusion.forward_context import get_ulysses_mode

logger = init_logger(__name__)


def _a2a_permute_enabled(enabled: bool, scatter_idx: int, gather_idx: int, world_size: int) -> bool:
    """Whether the fused permute-free all-to-all applies (strict layout only)."""
    return enabled and world_size > 1 and scatter_idx == 2 and gather_idx == 1


def _ceil_div(n: int, d: int) -> int:
    return (n + d - 1) // d


def _positive_divisors(n: int) -> list[int]:
    if n <= 0:
        return []
    divs = set()
    i = 1
    while i * i <= n:
        if n % i == 0:
            divs.add(i)
            divs.add(n // i)
        i += 1
    return sorted(divs)


@torch.compiler.disable
def _all_gather_int(pg: dist.ProcessGroup, value: int, *, device: torch.device) -> list[int]:
    """All-gather a scalar int across pg.

    Note: we use a device tensor so this works for NCCL subgroups (e.g. Ulysses/Ring).
    """
    world_size = dist.get_world_size(pg)
    if world_size == 1:
        return [int(value)]

    t = torch.tensor([int(value)], dtype=torch.int64, device=device)
    gathered = [torch.empty_like(t) for _ in range(world_size)]
    dist.all_gather(gathered, t, group=pg)
    return [int(x.item()) for x in gathered]


def _ulysses_all_to_all_any_qkv(
    pg: dist.ProcessGroup,
    x: torch.Tensor,  # (B, S_local, H, D)
    *,
    seq_lens: list[int],
    use_sync: bool,
) -> tuple[torch.Tensor, int]:
    """UAA forward all-to-all: (B, S_local, H, D) -> (B, S_global, H_local, D).

    Returns:
        (resharded, orig_head_cnt)
    """
    world_size = dist.get_world_size(pg)
    if world_size == 1:
        return x, int(x.shape[2])

    bsz, s_local, head_cnt, head_dim = x.shape
    orig_head_cnt = int(head_cnt)
    padded_head_cnt = _ceil_div(orig_head_cnt, world_size) * world_size
    head_pad = padded_head_cnt - orig_head_cnt
    if head_pad:
        x = F.pad(x, (0, 0, 0, head_pad))

    head_cnt_local = padded_head_cnt // world_size

    # (B, S_local, H, D) -> (world_size, S_local, B, H_local, D)
    x_t = x.reshape(bsz, s_local, world_size, head_cnt_local, head_dim).permute(2, 1, 0, 3, 4).contiguous()
    # (world_size, S_local, B, H_local, D) -> (world_size * S_local, B, H_local, D)
    x_t = x_t.flatten(0, 1)

    input_split_sizes = [s_local] * world_size
    output_split_sizes = seq_lens
    s_global = int(sum(output_split_sizes))

    out = torch.empty((s_global, bsz, head_cnt_local, head_dim), device=x.device, dtype=x.dtype)
    dist.all_to_all_single(
        out,
        x_t,
        output_split_sizes=output_split_sizes,
        input_split_sizes=input_split_sizes,
        group=pg,
    )
    if use_sync:
        from vllm_omni.platforms import current_omni_platform

        current_omni_platform.synchronize()

    # (S_global, B, H_local, D) -> (B, S_global, H_local, D)
    out = out.permute(1, 0, 2, 3).contiguous()
    return out, orig_head_cnt


def _ulysses_all_to_all_any_o(
    pg: dist.ProcessGroup,
    x: torch.Tensor,  # (B, S_global, H_local, D)
    *,
    seq_lens: list[int],
    local_seq_len: int,
    orig_head_cnt: int,
    use_sync: bool,
) -> torch.Tensor:
    """UAA reverse all-to-all: (B, S_global, H_local, D) -> (B, S_local, H, D)."""
    world_size = dist.get_world_size(pg)
    if world_size == 1:
        return x

    bsz, s_global, head_cnt_local, head_dim = x.shape
    s_local = int(local_seq_len)

    # (B, S_global, H_local, D) -> (S_global, B, H_local, D)
    x_t = x.permute(1, 0, 2, 3).contiguous()

    input_split_sizes = seq_lens
    output_split_sizes = [s_local] * world_size

    out = torch.empty((world_size * s_local, bsz, head_cnt_local, head_dim), device=x.device, dtype=x.dtype)
    dist.all_to_all_single(
        out,
        x_t,
        output_split_sizes=output_split_sizes,
        input_split_sizes=input_split_sizes,
        group=pg,
    )
    if use_sync:
        from vllm_omni.platforms import current_omni_platform

        current_omni_platform.synchronize()

    # (world_size * S_local, B, H_local, D) -> (B, S_local, H, D)
    out = out.reshape(world_size, s_local, bsz, head_cnt_local, head_dim).permute(2, 1, 0, 3, 4).contiguous()
    out = out.reshape(bsz, s_local, world_size * head_cnt_local, head_dim)

    if out.shape[2] != orig_head_cnt:
        out = out[:, :, :orig_head_cnt, :].contiguous()
    return out


@torch.compiler.disable
def _all_gather_h3_vsa_deferred_coarse(
    pg: dist.ProcessGroup,
    coarse: torch.Tensor,
    *,
    use_sync: bool,
) -> torch.Tensor:
    """Gather compact H3 VSA coarse rows from head owners.

    Input is ``[B, blocks, H_local, D]``.  Moving local heads to the leading
    dimension lets ``all_gather_into_tensor`` concatenate ranks directly in
    the same head order produced by reverse Ulysses.
    """
    world_size = dist.get_world_size(pg)
    if world_size <= 1:
        raise ValueError("deferred H3 VSA coarse gather requires SP world_size > 1")
    if coarse.ndim != 4:
        raise ValueError(f"deferred H3 VSA coarse tensor must be [B,N,H,D], got {coarse.shape}")
    if coarse.device.type != "cuda" or coarse.dtype != torch.bfloat16:
        raise TypeError("deferred H3 VSA coarse gather requires CUDA BF16")
    if coarse.requires_grad:
        raise ValueError("deferred H3 VSA coarse gather is inference-only")

    send = coarse.permute(2, 0, 1, 3).contiguous()
    gathered = torch.empty(
        (send.shape[0] * world_size, *send.shape[1:]),
        dtype=send.dtype,
        device=send.device,
    )
    dist.all_gather_into_tensor(gathered, send, group=pg)
    if use_sync:
        from vllm_omni.platforms import current_omni_platform

        current_omni_platform.synchronize()
    return gathered.permute(1, 2, 0, 3)


@dataclass(frozen=True, slots=True)
class _UlyssesCtx(ParallelAttentionContext):
    """Per-forward context for Ulysses sequence-parallel attention."""

    ulysses_pg: dist.ProcessGroup
    scatter_idx: int
    gather_idx: int
    use_sync: bool
    ulysses_rank: int
    joint_len: int = 0
    joint_strategy: str = "front"
    # UAA (Ulysses Anything Attention) metadata
    use_uaa: bool = False
    uaa_seq_lens: tuple[int, ...] = ()
    uaa_local_seq_len: int = 0
    orig_head_cnt: int = 0
    joint_orig_head_cnt: int = 0
    strict_a2a_backend: str = ""
    q_chunk_major_chunks: int = 0
    qchunk2_direct_experiment: bool = False
    deferred_gate_local: torch.Tensor | None = None
    deferred_gate_metadata: AttentionMetadata | None = None
    deferred_gate_state: dict[str, object] | None = None
    o_bundle_gate_local: torch.Tensor | None = None
    o_bundle_metadata: AttentionMetadata | None = None
    o_bundle_state: dict[str, object] | None = None
    o_bundle_gate_compute_overlap_ticket: H3VSAGateComputeOverlapTicket | None = None
    fp8_qkv_transport: bool = False


class UlyssesParallelAttention:
    """Ulysses sequence-parallel strategy (all-to-all over seq/head dims).

    This preserves the semantics previously implemented in
    `Attention._forward_ulysses`:
    - If `AttentionMetadata.joint_*` is provided, joint_query/key/value are
      concatenated *after* all-to-all.
    - joint_key/value are assumed to be replicated across SP ranks and are sliced
      by ulysses head rank before concatenation.
    """

    def __init__(
        self,
        sp_group: SequenceParallelGroupCoordinator,
        scatter_idx: int,
        gather_idx: int,
        use_sync: bool,
        ulysses_a2a_permute: bool = False,
    ) -> None:
        self._sp_group = sp_group
        self._ulysses_pg = sp_group.ulysses_group
        self._scatter_idx = scatter_idx
        self._gather_idx = gather_idx
        self._use_sync = use_sync
        self._ulysses_a2a_permute = ulysses_a2a_permute
        self._ulysses_a2a_backend = ""
        self._qk_input_landing_enabled = False
        self._qkv_e4m3_input_landing_enabled = False
        self._o_input_landing_enabled = False
        self._qchunk2_direct_enabled = False
        self._qchunk2_qkv_fused_enabled = False
        self._qkv_single_phase_enabled = False
        if _a2a_permute_enabled(
            ulysses_a2a_permute,
            scatter_idx,
            gather_idx,
            sp_group.ulysses_world_size,
        ):
            from vllm_omni.diffusion.distributed.flashinfer_ulysses import configured_backend

            self._ulysses_a2a_backend = configured_backend()
            if self._ulysses_a2a_backend == "flashinfer-pcie":
                from vllm_omni.diffusion.distributed.flashinfer_ulysses import (
                    ensure_flashinfer_pcie_available,
                    is_o_producer_direct_enabled,
                    is_qchunk2_direct_enabled,
                    is_qchunk2_qkv_fused_enabled,
                    is_qk_producer_direct_enabled,
                    is_qkv_e4m3_transport_enabled,
                    is_qkv_single_phase_enabled,
                )

                ensure_flashinfer_pcie_available()
                self._qk_input_landing_enabled = is_qk_producer_direct_enabled()
                self._qkv_e4m3_input_landing_enabled = is_qkv_e4m3_transport_enabled()
                self._o_input_landing_enabled = is_o_producer_direct_enabled()
                self._qchunk2_direct_enabled = is_qchunk2_direct_enabled()
                self._qchunk2_qkv_fused_enabled = is_qchunk2_qkv_fused_enabled()
                self._qkv_single_phase_enabled = is_qkv_single_phase_enabled()
            elif self._ulysses_a2a_backend == "symmetric":
                from vllm_omni.diffusion.distributed.a2a_permute import ensure_a2a_permute_available

                ensure_a2a_permute_available()
            else:
                raise ValueError(
                    "VLLM_OMNI_ULYSSES_A2A_BACKEND must be 'symmetric' or "
                    f"'flashinfer-pcie', got {self._ulysses_a2a_backend!r}"
                )

    @property
    def enabled(self) -> bool:
        return True

    @property
    def name(self) -> str:
        return "ulysses"

    @property
    def supports_qk_input_landing(self) -> bool:
        """Whether strict Ulysses can consume registered Q/K producer output."""
        return self._qk_input_landing_enabled

    @property
    def supports_qkv_e4m3_input_landing(self) -> bool:
        """Whether strict Ulysses accepts registered E4M3 Q/K/V sources."""
        return self._qkv_e4m3_input_landing_enabled

    def prepare_qk_input_landings(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        *,
        q_chunk_major_chunks: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Acquire producer-direct Q/K buffers for the opt-in PCIe route."""
        if not self.supports_qk_input_landing:
            return None
        if get_ulysses_mode(default="strict") != "strict":
            return None
        if self._sp_group.ring_world_size != 1:
            return None

        from vllm_omni.diffusion.distributed.flashinfer_ulysses import (
            flashinfer_ulysses_qk_input_landings,
        )

        return flashinfer_ulysses_qk_input_landings(
            query,
            key,
            self._ulysses_pg,
            self._sp_group.ulysses_world_size,
            q_chunk_major_chunks=q_chunk_major_chunks,
        )

    def prepare_qkv_e4m3_input_landings(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
        """Acquire registered E4M3 buffers for BF16 H3 Q/K/V producers."""
        if not self.supports_qkv_e4m3_input_landing:
            return None
        if get_ulysses_mode(default="strict") != "strict":
            raise RuntimeError("E4M3 QKV transport requires strict Ulysses")
        if self._sp_group.ring_world_size != 1:
            raise RuntimeError("E4M3 QKV transport is incompatible with Ring attention")

        from vllm_omni.diffusion.distributed.flashinfer_ulysses import (
            flashinfer_ulysses_qkv_e4m3_input_landings,
        )

        return flashinfer_ulysses_qkv_e4m3_input_landings(
            query,
            key,
            value,
            self._ulysses_pg,
            self._sp_group.ulysses_world_size,
        )

    def pre_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata | None,
        *,
        q_chunk_major_chunks: int = 0,
    ):
        mode = get_ulysses_mode(default="strict")
        ulysses_world_size = self._sp_group.ulysses_world_size
        gate_compress = None
        deferred_gate_requested = False
        o_bundle_requested = False
        fp8_qkv_transport_requested = False
        h3_vsa_attention_active = False
        deferred_gate_configured = False
        gate_compute_overlap_ticket: H3VSAGateComputeOverlapTicket | None = None
        if attn_metadata is not None:
            from vllm_omni.diffusion.attention.ops.minimax_h3_vsa_deferred_gate import (
                H3_VSA_DEFERRED_ACTIVE_KEY,
                H3_VSA_DEFERRED_STATE_KEY,
                h3_vsa_deferred_gate_sp_enabled,
            )
            from vllm_omni.diffusion.attention.ops.minimax_h3_vsa_gate_compute_overlap import (
                H3_VSA_GATE_COMPUTE_OVERLAP_ACTIVE_KEY,
                H3_VSA_GATE_COMPUTE_OVERLAP_TICKET_KEY,
                h3_vsa_gate_compute_overlap_enabled,
            )
            from vllm_omni.diffusion.attention.ops.minimax_h3_vsa_layout import (
                H3_VSA_ATTENTION_ACTIVE_KEY,
                H3_VSA_FP8_QKV_SCALES_KEY,
            )
            from vllm_omni.diffusion.attention.ops.minimax_h3_vsa_o_bundle import (
                H3_VSA_O_BUNDLE_ACTIVE_KEY,
                H3_VSA_O_BUNDLE_STATE_KEY,
                h3_vsa_o_bundle_enabled,
            )

            # Metadata objects are reused across H3 blocks. Never let a prior
            # block's registered source pointer escape into a later route that
            # did not explicitly acquire and arm it.
            attn_metadata.extra.pop("ulysses_o_input_landing", None)
            attn_metadata.extra.pop(H3_VSA_DEFERRED_ACTIVE_KEY, None)
            attn_metadata.extra.pop(H3_VSA_DEFERRED_STATE_KEY, None)
            attn_metadata.extra.pop(H3_VSA_O_BUNDLE_ACTIVE_KEY, None)
            attn_metadata.extra.pop(H3_VSA_O_BUNDLE_STATE_KEY, None)
            overlap_active = attn_metadata.extra.pop(
                H3_VSA_GATE_COMPUTE_OVERLAP_ACTIVE_KEY,
                None,
            )
            overlap_ticket = attn_metadata.extra.pop(
                H3_VSA_GATE_COMPUTE_OVERLAP_TICKET_KEY,
                None,
            )
            if (overlap_active is None) != (overlap_ticket is None):
                raise RuntimeError(
                    "MiniMax-H3 VSA gate-compute overlap active marker and ticket must appear together"
                )
            if overlap_active is None:
                pass
            elif overlap_active is not True:
                raise RuntimeError(
                    "MiniMax-H3 VSA gate-compute overlap ticket requires its exact active marker"
                )
            elif not isinstance(overlap_ticket, H3VSAGateComputeOverlapTicket):
                raise TypeError("MiniMax-H3 VSA gate-compute overlap metadata contains an invalid ticket")
            elif not h3_vsa_gate_compute_overlap_enabled():
                raise RuntimeError(
                    "MiniMax-H3 VSA gate-compute overlap metadata reached Ulysses while its env gate is off"
                )
            else:
                gate_compute_overlap_ticket = overlap_ticket
            fp8_qkv_transport_requested = H3_VSA_FP8_QKV_SCALES_KEY in attn_metadata.extra
            candidate = attn_metadata.extra.get("gate_compress")
            deferred_gate_configured = h3_vsa_deferred_gate_sp_enabled()
            o_bundle_configured = h3_vsa_o_bundle_enabled()
            h3_vsa_attention_active = attn_metadata.extra.get(
                H3_VSA_ATTENTION_ACTIVE_KEY,
                False,
            )
            if (deferred_gate_configured or o_bundle_configured) and not isinstance(
                h3_vsa_attention_active,
                bool,
            ):
                raise TypeError(f"{H3_VSA_ATTENTION_ACTIVE_KEY} must be a bool")
            if h3_vsa_attention_active and deferred_gate_configured and o_bundle_configured:
                raise RuntimeError(
                    "MiniMax-H3 VSA deferred gate and reverse-O bundle experiments are mutually exclusive"
                )
            if isinstance(candidate, torch.Tensor):
                if candidate.shape != query.shape:
                    raise ValueError(
                        f"gate_compress must match pre-Ulysses query shape, got {candidate.shape} vs {query.shape}"
                    )
                gate_compress = candidate
                deferred_gate_requested = deferred_gate_configured and h3_vsa_attention_active
                o_bundle_requested = o_bundle_configured and h3_vsa_attention_active
            elif deferred_gate_configured and h3_vsa_attention_active:
                raise RuntimeError(
                    "deferred MiniMax-H3 VSA gate communication requires a "
                    "Tensor gate_compress on every H3 VSA attention block; "
                    f"got {type(candidate).__name__}"
                )
            elif o_bundle_configured and h3_vsa_attention_active:
                raise RuntimeError(
                    "MiniMax-H3 VSA reverse-O bundling requires a Tensor "
                    "gate_compress on every H3 VSA attention block; got "
                    f"{type(candidate).__name__}"
                )

        # advanced_uaa pads non-divisible head counts before the Ulysses
        # all-to-all. Padding K/V is not valid in hybrid Ulysses+Ring: the Ring
        # GQA/MQA path would repeat the padded zero heads as real K/V heads.
        # Reject this layout until K/V replication is performed before Ulysses.
        if mode == "advanced_uaa" and self._sp_group.ring_world_size > 1:
            for name, tensor in (("key", key), ("value", value)):
                kv_head_cnt = int(tensor.shape[2])
                if kv_head_cnt % ulysses_world_size != 0:
                    raise ValueError(
                        "ulysses_mode='advanced_uaa' with hybrid Ulysses+Ring "
                        "does not support K/V head padding. "
                        f"{name}_head_cnt={kv_head_cnt}, "
                        f"ulysses_degree={ulysses_world_size}. "
                        "Use ring_degree=1, choose a compatible ulysses_degree, "
                        "or replicate K/V heads before Ulysses."
                    )

        joint_tensor_query = joint_tensor_key = joint_tensor_value = None
        joint_strategy = "front"
        joint_len = 0
        joint_orig_head_cnt = 0

        if attn_metadata is not None:
            joint_tensor_query = attn_metadata.joint_query
            joint_tensor_key = attn_metadata.joint_key
            joint_tensor_value = attn_metadata.joint_value
            joint_strategy = attn_metadata.joint_strategy

        is_joint = False
        if joint_tensor_query is not None and joint_tensor_key is not None and joint_tensor_value is not None:
            supported_joint_strategy = ["front", "rear"]
            if joint_strategy not in supported_joint_strategy:
                raise ValueError(
                    f"joint_strategy: {joint_strategy} not supported."
                    f" supported joint strategy: {supported_joint_strategy}"
                )

            # Slice joint_query for this Ulysses rank
            # joint_query is (B, S, H, D). We split H (dim 2).
            ulysses_rank = self._sp_group.ulysses_rank
            joint_head_cnt = int(joint_tensor_query.shape[-2])
            joint_orig_head_cnt = joint_head_cnt

            if mode == "advanced_uaa":
                padded_joint_head_cnt = _ceil_div(joint_head_cnt, ulysses_world_size) * ulysses_world_size
                joint_head_pad = padded_joint_head_cnt - joint_head_cnt
                if joint_head_pad:
                    joint_tensor_query = F.pad(joint_tensor_query, (0, 0, 0, joint_head_pad))
                    joint_tensor_key = F.pad(joint_tensor_key, (0, 0, 0, joint_head_pad))
                    joint_tensor_value = F.pad(joint_tensor_value, (0, 0, 0, joint_head_pad))
                joint_head_cnt = padded_joint_head_cnt
            else:
                if joint_head_cnt % ulysses_world_size != 0:
                    supported = _positive_divisors(joint_head_cnt)
                    raise ValueError(
                        "Ulysses-SP strict mode requires joint head_cnt divisible by ulysses_degree. "
                        f"joint_head_cnt={joint_head_cnt}, ulysses_degree={ulysses_world_size}. "
                        f"Try ulysses_degree in {supported}, or set ulysses_mode='advanced_uaa'."
                    )

            attn_heads_per_ulysses_rank = joint_head_cnt // ulysses_world_size

            # Note: We use the same heads for Q/K/V
            joint_tensor_query = joint_tensor_query[
                ...,
                attn_heads_per_ulysses_rank * ulysses_rank : attn_heads_per_ulysses_rank * (ulysses_rank + 1),
                :,
            ]

            joint_len = joint_tensor_query.shape[1]

            is_joint = True
        elif joint_tensor_query is None and joint_tensor_key is None and joint_tensor_value is None:
            pass
        else:
            raise ValueError("joint_query, joint_key, and joint_value should be None or not None simultaneously.")

        any_e4m3_qkv = any(tensor.dtype == torch.float8_e4m3fn for tensor in (query, key, value))
        if any_e4m3_qkv and not fp8_qkv_transport_requested:
            raise RuntimeError("E4M3 Ulysses Q/K/V requires the explicit MiniMax-H3 VSA FP8 transport scale marker")
        if fp8_qkv_transport_requested:
            assert attn_metadata is not None
            scales = attn_metadata.extra[H3_VSA_FP8_QKV_SCALES_KEY]
            reasons: list[str] = []
            if not isinstance(scales, (tuple, list)) or len(scales) != 3:
                reasons.append("dequant scales are not a length-3 tuple/list")
            elif any(not isinstance(scale, float) or not math.isfinite(scale) or scale <= 0 for scale in scales):
                reasons.append("dequant scales are not finite positive Python floats")
            if h3_vsa_attention_active is not True:
                reasons.append("FASTVIDEO_VSA H3 attention marker is absent")
            if any(tensor.dtype != torch.float8_e4m3fn for tensor in (query, key, value)):
                reasons.append("Q/K/V are not all float8_e4m3fn")
            if key.shape != query.shape or value.shape != query.shape:
                reasons.append("Q/K/V shapes differ")
            if key.device != query.device or value.device != query.device:
                reasons.append("Q/K/V devices differ")
            for name, tensor in (("Q", query), ("K", key), ("V", value)):
                if tensor.device.type != "cuda":
                    reasons.append(f"{name} is not a CUDA tensor")
                if not tensor.is_contiguous():
                    reasons.append(f"{name} is not contiguous")
                if tensor.requires_grad:
                    reasons.append(f"{name} requires gradients")
            if mode != "strict":
                reasons.append(f"Ulysses mode is {mode!r}, not 'strict'")
            if ulysses_world_size != 8:
                reasons.append(f"Ulysses world size is {ulysses_world_size}, not qualified SP8")
            if self._sp_group.ring_world_size != 1:
                reasons.append("Ring attention is active")
            if is_joint:
                reasons.append("joint attention rows are active")
            if (self._scatter_idx, self._gather_idx) != (2, 1):
                reasons.append(f"scatter/gather dimensions are {(self._scatter_idx, self._gather_idx)}, not (2, 1)")
            if q_chunk_major_chunks:
                reasons.append("query-chunk-major attention is active")
            if self._ulysses_a2a_backend != "flashinfer-pcie":
                reasons.append("FlashInfer PCIe/RDMA Ulysses is not selected")
            if not self._qk_input_landing_enabled:
                reasons.append("registered Q/K producer-direct landing is disabled")
            if not self._o_input_landing_enabled:
                reasons.append("registered reverse-O producer-direct landing is disabled")
            if os.environ.get(
                "VLLM_OMNI_FLASHINFER_ULYSSES_REQUIRE_RDMA",
                "0",
            ).strip().lower() not in {"1", "true", "yes", "on"}:
                reasons.append("all-RDMA transport is not required")
            if not os.environ.get(
                "VLLM_OMNI_FLASHINFER_ULYSSES_MAX_BYTES",
                "",
            ).strip():
                reasons.append("explicit registered-buffer capacity is absent")
            if os.environ.get(
                "VLLM_OMNI_FASTVIDEO_VSA_FUSED_TILE_PACK",
                "0",
            ).strip().lower() not in {"1", "true", "yes", "on"}:
                reasons.append("fused E4M3 dequant/tile-pack is disabled")
            prefix_segments = attn_metadata.extra.get("vsa_h3_prefix_segments")
            if not isinstance(prefix_segments, (tuple, list)):
                reasons.append("MiniMax-H3 VSA prefix metadata is absent")
            video_layout = attn_metadata.video_layout
            if video_layout is None or not any(span.role == "target" for span in video_layout.video_spans):
                reasons.append("MiniMax-H3 target-video layout is absent")
            if gate_compress is None:
                reasons.append("BF16 VSA compression gate is absent")
            else:
                if gate_compress.device.type != "cuda":
                    reasons.append("VSA compression gate is not CUDA")
                if gate_compress.dtype != torch.bfloat16:
                    reasons.append("VSA compression gate is not BF16")
                if not gate_compress.is_contiguous():
                    reasons.append("VSA compression gate is not contiguous")
                if gate_compress.requires_grad:
                    reasons.append("VSA compression gate requires gradients")
            if deferred_gate_configured:
                reasons.append("deferred-gate communication is enabled")
            if reasons:
                raise RuntimeError("MiniMax-H3 VSA FP8 QKV transport requested but unsupported: " + "; ".join(reasons))

        if gate_compute_overlap_ticket is not None:
            reasons = []
            if not fp8_qkv_transport_requested:
                reasons.append("E4M3 QKV transport is not active")
            if not o_bundle_requested:
                reasons.append("reverse-O bundling is not active")
            if deferred_gate_requested or deferred_gate_configured:
                reasons.append("deferred-gate SP is active")
            if gate_compress is not gate_compute_overlap_ticket.gate:
                reasons.append("gate tensor identity differs from the overlap ticket")
            if reasons:
                raise RuntimeError(
                    "MiniMax-H3 VSA gate-compute overlap requested but unsupported: "
                    + "; ".join(reasons)
                )

        deferred_gate_local: torch.Tensor | None = None
        deferred_gate_metadata: AttentionMetadata | None = None
        deferred_gate_state: dict[str, object] | None = None
        if deferred_gate_requested:
            assert gate_compress is not None
            assert attn_metadata is not None
            prefix_segments = attn_metadata.extra.get("vsa_h3_prefix_segments")
            video_layout = attn_metadata.video_layout
            has_target_video = bool(
                video_layout is not None and any(span.role == "target" for span in video_layout.video_spans)
            )
            reasons: list[str] = []
            if mode != "strict":
                reasons.append(f"Ulysses mode is {mode!r}, not 'strict'")
            if ulysses_world_size <= 1:
                reasons.append(f"Ulysses world size is {ulysses_world_size}, not >1")
            if self._sp_group.ring_world_size != 1:
                reasons.append("Ring attention is active")
            if is_joint:
                reasons.append("joint attention rows are active")
            if (self._scatter_idx, self._gather_idx) != (2, 1):
                reasons.append(f"scatter/gather dimensions are {(self._scatter_idx, self._gather_idx)}, not (2, 1)")
            if q_chunk_major_chunks:
                reasons.append("query-chunk-major attention is active")
            if not isinstance(prefix_segments, (tuple, list)):
                reasons.append("MiniMax-H3 VSA prefix metadata is absent")
            if not has_target_video:
                reasons.append("MiniMax-H3 target-video layout is absent")
            if gate_compress.device.type != "cuda" or gate_compress.dtype != torch.bfloat16:
                reasons.append(f"gate is not CUDA BF16 (device={gate_compress.device}, dtype={gate_compress.dtype})")
            if gate_compress.requires_grad:
                reasons.append("gate requires gradients")
            if reasons:
                raise RuntimeError("deferred MiniMax-H3 VSA gate requested but unsupported: " + "; ".join(reasons))

            deferred_gate_local = gate_compress
            deferred_gate_metadata = attn_metadata
            deferred_gate_state = {}
            gate_compress = None
            attn_metadata.extra.pop("gate_compress")
            attn_metadata.extra[H3_VSA_DEFERRED_ACTIVE_KEY] = True
            attn_metadata.extra[H3_VSA_DEFERRED_STATE_KEY] = deferred_gate_state

        o_bundle_gate_local: torch.Tensor | None = None
        o_bundle_metadata: AttentionMetadata | None = None
        o_bundle_state: dict[str, object] | None = None
        if o_bundle_requested:
            assert gate_compress is not None
            assert attn_metadata is not None
            from vllm_omni.diffusion.attention.backends.fastvideo_vsa import (
                get_h3_vsa_owner_route_plan,
            )
            from vllm_omni.diffusion.attention.ops.minimax_h3_vsa_layout import (
                H3_VSA_FUSED_GATE_UNTILE_ENV,
                h3_vsa_fused_gate_untile_enabled,
            )

            raw_prefix_segments = attn_metadata.extra.get("vsa_h3_prefix_segments")
            video_layout = attn_metadata.video_layout
            target = (
                next(
                    (span for span in reversed(video_layout.video_spans) if span.role == "target"),
                    None,
                )
                if video_layout is not None
                else None
            )
            reasons: list[str] = []
            if mode != "strict":
                reasons.append(f"Ulysses mode is {mode!r}, not 'strict'")
            if ulysses_world_size <= 1:
                reasons.append(f"Ulysses world size is {ulysses_world_size}, not >1")
            if self._sp_group.ring_world_size != 1:
                reasons.append("Ring attention is active")
            if is_joint:
                reasons.append("joint attention rows are active")
            if (self._scatter_idx, self._gather_idx) != (2, 1):
                reasons.append(f"scatter/gather dimensions are {(self._scatter_idx, self._gather_idx)}, not (2, 1)")
            if q_chunk_major_chunks:
                reasons.append("query-chunk-major attention is active")
            if not isinstance(raw_prefix_segments, (tuple, list)):
                reasons.append("MiniMax-H3 VSA prefix metadata is absent")
            if target is None:
                reasons.append("MiniMax-H3 target-video layout is absent")
            if gate_compress.device.type != "cuda":
                reasons.append(f"gate is not CUDA (device={gate_compress.device})")
            if gate_compress.dtype != torch.bfloat16:
                reasons.append(f"gate is not BF16 (dtype={gate_compress.dtype})")
            if not gate_compress.is_contiguous():
                reasons.append("gate is not contiguous")
            if gate_compress.requires_grad:
                reasons.append("gate requires gradients")
            if self._ulysses_a2a_backend != "flashinfer-pcie":
                reasons.append("FlashInfer PCIe/RDMA strict Ulysses is not selected")
            if not self._o_input_landing_enabled:
                reasons.append("reverse-O producer-direct landing is disabled")
            if h3_vsa_fused_gate_untile_enabled():
                reasons.append(f"{H3_VSA_FUSED_GATE_UNTILE_ENV}=1 is active")
            if reasons:
                raise RuntimeError("MiniMax-H3 VSA reverse-O bundling requested but unsupported: " + "; ".join(reasons))

            assert isinstance(raw_prefix_segments, (tuple, list))
            assert target is not None
            prefix_segments = tuple(int(segment) for segment in raw_prefix_segments if int(segment) > 0)
            video_shape = tuple(int(dim) for dim in target.latent_grid)
            valid_rows = sum(prefix_segments) + math.prod(video_shape)
            if int(target.start) != sum(prefix_segments):
                raise ValueError(
                    "MiniMax-H3 VSA reverse-O bundle prefix/target boundary "
                    f"mismatch: prefix={sum(prefix_segments)}, target={target.start}"
                )
            packed = attn_metadata.packed_padding
            if packed is None:
                if valid_rows != query.shape[1] * ulysses_world_size:
                    raise ValueError(
                        "MiniMax-H3 VSA reverse-O bundle requires packed-padding "
                        "metadata when the global sequence is aligned: "
                        f"valid={valid_rows}, aligned={query.shape[1] * ulysses_world_size}"
                    )
            elif packed.q_length != valid_rows or packed.kv_length != valid_rows:
                raise ValueError(
                    "MiniMax-H3 VSA reverse-O bundle packed lengths do not "
                    f"match the H3 layout: q={packed.q_length}, "
                    f"kv={packed.kv_length}, expected={valid_rows}"
                )

            aligned_rows = query.shape[1] * ulysses_world_size
            owner_route = get_h3_vsa_owner_route_plan(
                prefix_segments,
                video_shape,
                aligned_rows,
                ulysses_world_size,
            )
            if owner_route.local_rows != query.shape[1]:
                raise RuntimeError(
                    "MiniMax-H3 VSA reverse-O owner route has the wrong local "
                    f"sequence: route={owner_route.local_rows}, query={query.shape[1]}"
                )

            o_bundle_gate_local = gate_compress
            o_bundle_metadata = attn_metadata
            o_bundle_state = {"plan": owner_route}
            if gate_compute_overlap_ticket is not None:
                gate_compute_overlap_ticket.claim(o_bundle_gate_local)
            gate_compress = None
            attn_metadata.extra.pop("gate_compress")
            attn_metadata.extra[H3_VSA_O_BUNDLE_ACTIVE_KEY] = True
            attn_metadata.extra[H3_VSA_O_BUNDLE_STATE_KEY] = o_bundle_state

        if is_joint:
            # Slice joint key/value heads for this ulysses rank.
            # Using same slicing logic as query
            attn_heads_per_ulysses_rank_kv = joint_tensor_key.shape[-2] // ulysses_world_size

            joint_tensor_key = joint_tensor_key[
                ...,
                attn_heads_per_ulysses_rank_kv * ulysses_rank : attn_heads_per_ulysses_rank_kv * (ulysses_rank + 1),
                :,
            ]
            joint_tensor_value = joint_tensor_value[
                ...,
                attn_heads_per_ulysses_rank_kv * ulysses_rank : attn_heads_per_ulysses_rank_kv * (ulysses_rank + 1),
                :,
            ]

            # Update metadata with sliced tensors so Ring attention can use them if needed
            if attn_metadata is not None:
                attn_metadata.joint_key = joint_tensor_key
                attn_metadata.joint_value = joint_tensor_value

        ulysses_world_size = self._sp_group.ulysses_world_size
        strict_a2a_backend = ""
        qchunk2_direct_experiment = False
        q_chunk_major_output = 0
        if mode == "advanced_uaa":
            if self._scatter_idx != 2 or self._gather_idx != 1:
                raise ValueError(
                    "ulysses_mode='advanced_uaa' currently only supports scatter_idx=2, gather_idx=1 "
                    f"(got scatter_idx={self._scatter_idx}, gather_idx={self._gather_idx})."
                )

            local_seq_len = int(query.shape[1])
            seq_lens = _all_gather_int(self._ulysses_pg, local_seq_len, device=query.device)
            s_global = int(sum(seq_lens))

            # In hybrid Ulysses+Ring, Ring attention uses P2P send/recv with fixed-shape
            # buffers. This requires all ring ranks to have the same seq_len after the
            # Ulysses all-to-all (i.e. per-ring-rank S_global must match).
            if self._sp_group.ring_world_size > 1:
                ring_s_globals = _all_gather_int(self._sp_group.ring_group, s_global, device=query.device)
                if len(set(ring_s_globals)) != 1:
                    raise ValueError(
                        "ulysses_mode='advanced_uaa' with hybrid Ulysses+Ring requires the "
                        "post-Ulysses seq_len to be equal across ring ranks, but got "
                        f"{ring_s_globals} (ring_degree={self._sp_group.ring_world_size}). "
                        "This typically means the input sequence was not evenly shardable across the ring. "
                        "Try setting ring_degree=1, or choose a sequence length divisible by ring_degree."
                    )
            query, orig_head_cnt = _ulysses_all_to_all_any_qkv(
                self._ulysses_pg, query, seq_lens=seq_lens, use_sync=self._use_sync
            )
            key, _ = _ulysses_all_to_all_any_qkv(self._ulysses_pg, key, seq_lens=seq_lens, use_sync=self._use_sync)
            value, _ = _ulysses_all_to_all_any_qkv(self._ulysses_pg, value, seq_lens=seq_lens, use_sync=self._use_sync)
            if gate_compress is not None:
                gate_compress, _ = _ulysses_all_to_all_any_qkv(
                    self._ulysses_pg,
                    gate_compress,
                    seq_lens=seq_lens,
                    use_sync=self._use_sync,
                )
        else:
            # Strict mode: fail fast with actionable errors for head divisibility.
            for name, t in (("query", query), ("key", key), ("value", value)):
                head_cnt = int(t.shape[2])
                if head_cnt % ulysses_world_size != 0:
                    supported = _positive_divisors(head_cnt)
                    raise ValueError(
                        "Ulysses-SP strict mode requires head_cnt divisible by ulysses_degree. "
                        f"{name}_head_cnt={head_cnt}, ulysses_degree={ulysses_world_size}. "
                        f"Try ulysses_degree in {supported}, or set ulysses_mode='advanced_uaa'."
                    )

            # (bs, seq_len/P, head_cnt, head_size) -> (bs, seq_len, head_cnt/P, head_size)
            if _a2a_permute_enabled(
                self._ulysses_a2a_permute,
                self._scatter_idx,
                self._gather_idx,
                ulysses_world_size,
            ):
                if self._ulysses_a2a_backend == "flashinfer-pcie":
                    from vllm_omni.diffusion.distributed.flashinfer_ulysses import (
                        flashinfer_ulysses_qchunk2_fwd,
                        flashinfer_ulysses_qchunk2_qkv_fwd,
                        flashinfer_ulysses_qkv_fwd,
                        flashinfer_ulysses_qkv_single_phase_fwd,
                        qchunk2_direct_fallback_reason,
                    )

                    group_name = self._ulysses_pg.group_name
                    qchunk2_direct_experiment = bool(self._qchunk2_direct_enabled and q_chunk_major_chunks)
                    if qchunk2_direct_experiment:
                        # A joint-query concatenation happens after this
                        # collective and therefore cannot preserve the native
                        # chunk-major byte order. Force the custom op's exact
                        # standard-scatter fallback for that layout.
                        dispatch_chunks = q_chunk_major_chunks if not is_joint else 0
                        reason = qchunk2_direct_fallback_reason(
                            query,
                            world_size=ulysses_world_size,
                            chunks=dispatch_chunks,
                        )
                        if self._qchunk2_qkv_fused_enabled:
                            query, key, value = flashinfer_ulysses_qchunk2_qkv_fwd(
                                query,
                                key,
                                value,
                                group_name,
                                ulysses_world_size,
                                dispatch_chunks,
                                self._use_sync,
                            )
                        else:
                            query = flashinfer_ulysses_qchunk2_fwd(
                                query,
                                group_name,
                                ulysses_world_size,
                                dispatch_chunks,
                                self._use_sync,
                            )
                            key = flashinfer_ulysses_qkv_fwd(
                                key,
                                group_name,
                                ulysses_world_size,
                                "k",
                                self._use_sync,
                            )
                            value = flashinfer_ulysses_qkv_fwd(
                                value,
                                group_name,
                                ulysses_world_size,
                                "v",
                                self._use_sync,
                            )
                        if reason is None:
                            q_chunk_major_output = q_chunk_major_chunks
                    else:
                        if self._qkv_single_phase_enabled:
                            query, key, value = flashinfer_ulysses_qkv_single_phase_fwd(
                                query,
                                key,
                                value,
                                group_name,
                                ulysses_world_size,
                                self._use_sync,
                            )
                        else:
                            query = flashinfer_ulysses_qkv_fwd(
                                query,
                                group_name,
                                ulysses_world_size,
                                "q",
                                self._use_sync,
                            )
                            from vllm_omni.diffusion.attention.ops.h3_schedule_v3 import after_q

                            after_q(query, attn_metadata)
                            from vllm_omni.diffusion.attention.ops.h3_vsplit_experiment import after_q as vsplit_after_q

                            vsplit_after_q()
                            key = flashinfer_ulysses_qkv_fwd(
                                key,
                                group_name,
                                ulysses_world_size,
                                "k",
                                self._use_sync,
                            )
                            from vllm_omni.diffusion.attention.ops.h3_overlap_experiment import before_v

                            value = before_v(query, key, value, attn_metadata, self._ulysses_pg)
                            value = flashinfer_ulysses_qkv_fwd(
                                value,
                                group_name,
                                ulysses_world_size,
                                "v",
                                self._use_sync,
                            )
                    if gate_compress is not None:
                        # The first correctness implementation uses its own
                        # registered RDMA slot.  A later QKVG-fused transport is
                        # a separately measurable communication optimization.
                        gate_compress = flashinfer_ulysses_qkv_fwd(
                            gate_compress,
                            group_name,
                            ulysses_world_size,
                            "g",
                            self._use_sync,
                        )
                    strict_a2a_backend = "flashinfer-pcie"
                else:
                    from vllm_omni.diffusion.distributed.a2a_permute import ulysses_qkv_fwd

                    group_name = self._ulysses_pg.group_name
                    query = ulysses_qkv_fwd(query, group_name, ulysses_world_size)
                    key = ulysses_qkv_fwd(key, group_name, ulysses_world_size)
                    value = ulysses_qkv_fwd(value, group_name, ulysses_world_size)
                    if gate_compress is not None:
                        gate_compress = ulysses_qkv_fwd(gate_compress, group_name, ulysses_world_size)
                    strict_a2a_backend = "symmetric"
            else:
                query = SeqAllToAll4D.apply(
                    self._ulysses_pg, query, self._scatter_idx, self._gather_idx, self._use_sync
                )
                key = SeqAllToAll4D.apply(self._ulysses_pg, key, self._scatter_idx, self._gather_idx, self._use_sync)
                value = SeqAllToAll4D.apply(
                    self._ulysses_pg, value, self._scatter_idx, self._gather_idx, self._use_sync
                )
                if gate_compress is not None:
                    gate_compress = SeqAllToAll4D.apply(
                        self._ulysses_pg,
                        gate_compress,
                        self._scatter_idx,
                        self._gather_idx,
                        self._use_sync,
                    )
            seq_lens = []
            local_seq_len = 0
            orig_head_cnt = 0

        if is_joint:
            # Concatenate joint query AFTER AllToAll
            # Image query is now (B, S, H/P, D). Joint query is (B, S_txt, H/P, D).
            # This is dimensionally consistent.
            if joint_strategy == "rear":
                query = torch.cat([query, joint_tensor_query], dim=1)
            else:
                query = torch.cat([joint_tensor_query, query], dim=1)

        # Check if Ring Attention is also active (Hybrid mode)
        # If Ring is active, we should NOT concatenate joint_key/value to k/v here.
        # Instead, they should remain in attn_metadata and be passed to the Ring kernel.
        use_ring = self._sp_group.ring_world_size > 1

        if is_joint and not use_ring:
            # Concatenate joint key/value after all-to-all ONLY for pure Ulysses (Local Attention).
            if joint_strategy == "front":
                key = torch.cat([joint_tensor_key, key], dim=1)
                value = torch.cat([joint_tensor_value, value], dim=1)
            else:  # "rear"
                key = torch.cat([key, joint_tensor_key], dim=1)
                value = torch.cat([value, joint_tensor_value], dim=1)

        if gate_compress is not None and attn_metadata is not None:
            # The VSA gate follows the query's S<->H all-to-all, yielding the
            # full packed sequence for this rank's local head shard.
            attn_metadata.extra["gate_compress"] = gate_compress

        if o_bundle_state is not None:
            if not (
                self._o_input_landing_enabled
                and mode == "strict"
                and strict_a2a_backend == "flashinfer-pcie"
                and self._sp_group.ring_world_size == 1
                and not is_joint
                and q_chunk_major_output == 0
                and attn_metadata is o_bundle_metadata
            ):
                raise RuntimeError(
                    "MiniMax-H3 VSA reverse-O bundle lost its qualified "
                    "FlashInfer producer-direct route before landing acquisition"
                )
            from vllm_omni.diffusion.attention.ops.minimax_h3_vsa_owner_route import (
                H3VSAOwnerRoutePlan,
                h3_vsa_owner_route_plan_is_trusted,
            )
            from vllm_omni.diffusion.distributed.flashinfer_ulysses import (
                flashinfer_ulysses_o_input_landing,
            )

            owner_route = o_bundle_state.get("plan")
            if not isinstance(owner_route, H3VSAOwnerRoutePlan) or not h3_vsa_owner_route_plan_is_trusted(owner_route):
                raise RuntimeError("MiniMax-H3 VSA reverse-O bundle owner route is absent or untrusted")
            expected_global_rows = owner_route.sp_world_size * owner_route.local_rows
            if query.shape[1] != expected_global_rows:
                raise ValueError(
                    "post-Ulysses query rows do not match the reverse-O bundle route: "
                    f"query={query.shape[1]}, route={expected_global_rows}"
                )
            expanded_shape = (
                query.shape[0],
                owner_route.sp_world_size * (owner_route.local_rows + owner_route.kmax),
                query.shape[2],
                query.shape[3],
            )
            o_landing_kwargs: dict[str, object] = {
                "input_shape": expanded_shape,
            }
            if fp8_qkv_transport_requested:
                o_landing_kwargs["output_dtype"] = torch.bfloat16
            o_landing = flashinfer_ulysses_o_input_landing(
                query,
                self._ulysses_pg,
                ulysses_world_size,
                **o_landing_kwargs,
            )
            if o_landing is None:
                raise RuntimeError(
                    "MiniMax-H3 VSA reverse-O bundling requires a registered "
                    "FlashInfer O producer-direct landing; no fallback is allowed"
                )
            if o_landing.dtype != torch.bfloat16:
                raise TypeError(
                    f"MiniMax-H3 VSA FP8 QKV transport requires a BF16 reverse-O bundle landing, got {o_landing.dtype}"
                )
            if tuple(o_landing.shape) != expanded_shape:
                raise RuntimeError(
                    "MiniMax-H3 VSA reverse-O landing returned the wrong shape: "
                    f"expected={expanded_shape}, observed={tuple(o_landing.shape)}"
                )
            assert attn_metadata is not None
            attn_metadata.extra["ulysses_o_input_landing"] = o_landing
        elif (
            self._o_input_landing_enabled
            and mode == "strict"
            and strict_a2a_backend == "flashinfer-pcie"
            and self._sp_group.ring_world_size == 1
            and not is_joint
            and q_chunk_major_output == 0
            and attn_metadata is not None
            and isinstance(
                attn_metadata.extra.get("vsa_h3_prefix_segments"),
                (tuple, list),
            )
        ):
            from vllm_omni.diffusion.distributed.flashinfer_ulysses import (
                flashinfer_ulysses_o_input_landing,
            )

            o_landing_kwargs = {}
            if fp8_qkv_transport_requested:
                o_landing_kwargs["output_dtype"] = torch.bfloat16
            o_landing = flashinfer_ulysses_o_input_landing(
                query,
                self._ulysses_pg,
                ulysses_world_size,
                **o_landing_kwargs,
            )
            if o_landing is not None:
                attn_metadata.extra["ulysses_o_input_landing"] = o_landing
            elif fp8_qkv_transport_requested:
                raise RuntimeError(
                    "MiniMax-H3 VSA FP8 QKV transport requires a registered "
                    "BF16 reverse-O producer landing; no fallback is allowed"
                )
            if fp8_qkv_transport_requested and o_landing is not None and o_landing.dtype != torch.bfloat16:
                raise TypeError(
                    f"MiniMax-H3 VSA FP8 QKV transport requires a BF16 reverse-O landing, got {o_landing.dtype}"
                )

        ctx = _UlyssesCtx(
            name=self.name,
            ulysses_pg=self._ulysses_pg,
            scatter_idx=self._scatter_idx,
            gather_idx=self._gather_idx,
            use_sync=self._use_sync,
            ulysses_rank=self._sp_group.ulysses_rank,
            joint_len=joint_len,
            joint_strategy=joint_strategy,
            use_uaa=(mode == "advanced_uaa"),
            uaa_seq_lens=tuple(int(x) for x in seq_lens) if mode == "advanced_uaa" else (),
            uaa_local_seq_len=int(local_seq_len) if mode == "advanced_uaa" else 0,
            orig_head_cnt=int(orig_head_cnt) if mode == "advanced_uaa" else 0,
            joint_orig_head_cnt=int(joint_orig_head_cnt) if mode == "advanced_uaa" else 0,
            strict_a2a_backend=strict_a2a_backend,
            q_chunk_major_chunks=q_chunk_major_output,
            qchunk2_direct_experiment=qchunk2_direct_experiment,
            deferred_gate_local=deferred_gate_local,
            deferred_gate_metadata=deferred_gate_metadata,
            deferred_gate_state=deferred_gate_state,
            o_bundle_gate_local=o_bundle_gate_local,
            o_bundle_metadata=o_bundle_metadata,
            o_bundle_state=o_bundle_state,
            o_bundle_gate_compute_overlap_ticket=gate_compute_overlap_ticket,
            fp8_qkv_transport=fp8_qkv_transport_requested,
        )
        use_2d_mask = False
        if attn_metadata is not None:
            if attn_metadata.attn_mask is not None and attn_metadata.attn_mask.ndim == 2:
                use_2d_mask = True
            if attn_metadata.joint_attn_mask is not None and attn_metadata.joint_attn_mask.ndim == 2:
                use_2d_mask = True

        if attn_metadata is not None and use_2d_mask:
            if is_joint:
                if attn_metadata.joint_attn_mask is None and attn_metadata.attn_mask is None:
                    attn_metadata.attn_mask = None
                else:
                    if attn_metadata.attn_mask is None:
                        attn_metadata.attn_mask = torch.ones(
                            [query.shape[0], query.shape[1] - attn_metadata.joint_attn_mask.shape[1]],
                            dtype=torch.bool,
                            device=query.device,
                        )
                    elif attn_metadata.joint_attn_mask is None:
                        attn_metadata.joint_attn_mask = torch.ones(
                            [query.shape[0], query.shape[1] - attn_metadata.attn_mask.shape[1]],
                            dtype=torch.bool,
                            device=query.device,
                        )
                    attn_metadata.attn_mask = (
                        torch.cat([attn_metadata.joint_attn_mask, attn_metadata.attn_mask], dim=1)
                        if joint_strategy == "front"
                        else torch.cat([attn_metadata.attn_mask, attn_metadata.joint_attn_mask], dim=1)
                    )

            if attn_metadata.attn_mask is not None:
                # the final attn_mask is ready, the length should be aligedn with query length
                assert attn_metadata.attn_mask.shape[1] == query.shape[1], (
                    f"attn_mask length: {attn_metadata.attn_mask.shape[1]} != query length: {query.shape[1]}"
                )
                attn_metadata.attn_mask = attn_metadata.attn_mask.bool().contiguous()
        return query, key, value, attn_metadata, ctx

    def _finish_deferred_gate(
        self,
        output: torch.Tensor,
        ctx: _UlyssesCtx,
    ) -> torch.Tensor:
        """Gather compact coarse heads and apply the retained local gate."""
        gate = ctx.deferred_gate_local
        metadata = ctx.deferred_gate_metadata
        state = ctx.deferred_gate_state
        if gate is None and metadata is None and state is None:
            return output
        if gate is None or metadata is None or state is None:
            raise RuntimeError("incomplete deferred H3 VSA gate context")

        from vllm_omni.diffusion.attention.ops.minimax_h3_vsa_deferred_gate import (
            H3_VSA_DEFERRED_ACTIVE_KEY,
            H3_VSA_DEFERRED_STATE_KEY,
            h3_vsa_deferred_gate_add_,
            h3_vsa_deferred_gate_cuda_supported,
        )

        active = metadata.extra.pop(H3_VSA_DEFERRED_ACTIVE_KEY, None)
        published_state = metadata.extra.pop(H3_VSA_DEFERRED_STATE_KEY, None)
        if active is not True:
            raise RuntimeError("H3 VSA backend did not acknowledge deferred gate ownership")
        if published_state is not state:
            raise RuntimeError("H3 VSA deferred state identity changed across attention")
        coarse = state.pop("coarse", None)
        row_blocks = state.pop("row_blocks", None)
        if state:
            raise RuntimeError(f"unexpected deferred H3 VSA state entries: {sorted(state)}")
        if not isinstance(coarse, torch.Tensor):
            raise RuntimeError("H3 VSA backend did not publish compact coarse output")
        if not isinstance(row_blocks, torch.Tensor):
            raise RuntimeError("H3 VSA backend did not publish compact row-to-block mapping")
        if "gate_compress" in metadata.extra:
            raise RuntimeError("deferred H3 VSA gate unexpectedly re-entered backend metadata")
        if output.shape != gate.shape:
            raise ValueError(
                f"reverse Ulysses output must match retained local gate: output={output.shape}, gate={gate.shape}"
            )
        if metadata.packed_padding is not None and row_blocks.numel() != metadata.packed_padding.q_length:
            raise ValueError(
                "deferred H3 VSA row map must match the packed valid prefix: "
                f"map={row_blocks.numel()}, valid={metadata.packed_padding.q_length}"
            )

        gathered_coarse = _all_gather_h3_vsa_deferred_coarse(
            ctx.ulysses_pg,
            coarse,
            use_sync=ctx.use_sync,
        )
        if gathered_coarse.shape[2:] != output.shape[2:]:
            raise ValueError(
                "compact coarse all-gather produced the wrong head geometry: "
                f"coarse={gathered_coarse.shape}, output={output.shape}"
            )
        rank = ctx.ulysses_rank
        global_row_start = rank * output.shape[1]
        if not h3_vsa_deferred_gate_cuda_supported(
            output,
            gathered_coarse,
            gate,
            row_blocks,
            global_row_start=global_row_start,
        ):
            raise RuntimeError("deferred H3 VSA gate contract reached an unsupported CUDA shape")
        gate_operand_bytes = gate.numel() * gate.element_size()
        gate_remote_wire_bytes = (
            gate_operand_bytes * (dist.get_world_size(ctx.ulysses_pg) - 1) // dist.get_world_size(ctx.ulysses_pg)
        )
        logger.info_once(
            "MiniMax H3 VSA deferred-gate SP active: rank=%d local_gate=%s "
            "local_coarse=%s gathered_coarse=%s "
            "elided_gate_operand_bytes=%d elided_gate_remote_wire_bytes=%d",
            rank,
            tuple(gate.shape),
            tuple(coarse.shape),
            tuple(gathered_coarse.shape),
            gate_operand_bytes,
            gate_remote_wire_bytes,
            scope="process",
        )
        return h3_vsa_deferred_gate_add_(
            output,
            gathered_coarse,
            gate,
            row_blocks,
            global_row_start=global_row_start,
        )

    def _finish_o_bundle(
        self,
        output: torch.Tensor,
        ctx: _UlyssesCtx,
    ) -> torch.Tensor:
        """Apply the retained local gate from the reverse-O coarse tail."""
        gate = ctx.o_bundle_gate_local
        metadata = ctx.o_bundle_metadata
        state = ctx.o_bundle_state
        gate_compute_overlap_ticket = ctx.o_bundle_gate_compute_overlap_ticket
        if gate is None and metadata is None and state is None and gate_compute_overlap_ticket is None:
            return output
        if gate is None or metadata is None or state is None:
            raise RuntimeError("incomplete MiniMax-H3 VSA reverse-O bundle context")
        if gate_compute_overlap_ticket is not None and not isinstance(
            gate_compute_overlap_ticket,
            H3VSAGateComputeOverlapTicket,
        ):
            raise TypeError("MiniMax-H3 VSA reverse-O bundle contains an invalid gate-compute overlap ticket")

        from vllm_omni.diffusion.attention.ops.minimax_h3_vsa_o_bundle import (
            H3_VSA_O_BUNDLE_ACTIVE_KEY,
            H3_VSA_O_BUNDLE_STATE_KEY,
            h3_vsa_o_bundle_local_gate_add_,
            h3_vsa_o_bundle_local_gate_cuda_supported,
        )
        from vllm_omni.diffusion.attention.ops.minimax_h3_vsa_owner_route import (
            H3VSAOwnerRoutePlan,
            h3_vsa_owner_route_plan_is_trusted,
        )

        active = metadata.extra.pop(H3_VSA_O_BUNDLE_ACTIVE_KEY, None)
        published_state = metadata.extra.pop(H3_VSA_O_BUNDLE_STATE_KEY, None)
        producer_landing = metadata.extra.pop("ulysses_o_input_landing", None)
        if active is not True:
            raise RuntimeError("H3 VSA backend did not acknowledge reverse-O bundle ownership")
        if published_state is not state:
            raise RuntimeError("H3 VSA reverse-O bundle state identity changed across attention")
        if not isinstance(producer_landing, torch.Tensor):
            raise RuntimeError("H3 VSA reverse-O bundle producer landing disappeared across attention")
        plan = state.pop("plan", None)
        published = state.pop("published", None)
        if state:
            raise RuntimeError(f"unexpected H3 VSA reverse-O bundle state entries: {sorted(state)}")
        if not isinstance(plan, H3VSAOwnerRoutePlan) or not h3_vsa_owner_route_plan_is_trusted(plan):
            raise RuntimeError("H3 VSA reverse-O bundle route plan disappeared or was modified")
        if published is not True:
            raise RuntimeError("H3 VSA backend did not publish the reverse-O bundled output")
        if "gate_compress" in metadata.extra:
            raise RuntimeError("H3 VSA reverse-O bundle gate unexpectedly re-entered backend metadata")
        if plan.sp_world_size != dist.get_world_size(ctx.ulysses_pg):
            raise ValueError(
                "H3 VSA reverse-O route world size changed across attention: "
                f"route={plan.sp_world_size}, group={dist.get_world_size(ctx.ulysses_pg)}"
            )
        expected_output = (
            gate.shape[0],
            plan.local_rows + plan.kmax,
            gate.shape[2],
            gate.shape[3],
        )
        if tuple(output.shape) != expected_output:
            raise ValueError(
                "reverse Ulysses output does not match the H3 VSA bundle: "
                f"output={tuple(output.shape)}, expected={expected_output}"
            )
        expected_gate = (
            output.shape[0],
            plan.local_rows,
            output.shape[2],
            output.shape[3],
        )
        if tuple(gate.shape) != expected_gate:
            raise ValueError(
                "retained H3 VSA gate does not match the owner-local fine rows: "
                f"gate={tuple(gate.shape)}, expected={expected_gate}"
            )

        rank = ctx.ulysses_rank
        if not h3_vsa_o_bundle_local_gate_cuda_supported(
            output,
            gate,
            plan,
            owner_rank=rank,
        ):
            raise RuntimeError("H3 VSA reverse-O bundle reached an unsupported local gate contract")
        world_size = plan.sp_world_size
        gate_operand_bytes = gate.numel() * gate.element_size()
        gate_remote_wire_bytes = gate_operand_bytes * (world_size - 1) // world_size
        tail_operand_bytes = output[:, plan.local_rows :].numel() * output.element_size()
        tail_remote_wire_bytes = tail_operand_bytes * (world_size - 1) // world_size
        logger.info_once(
            "MiniMax H3 VSA reverse-O bundle SP active: rank=%d local_gate=%s "
            "reversed_bundle=%s kmax=%d elided_gate_operand_bytes=%d "
            "elided_gate_remote_wire_bytes=%d piggyback_tail_operand_bytes=%d "
            "piggyback_tail_remote_wire_bytes=%d",
            rank,
            tuple(gate.shape),
            tuple(output.shape),
            plan.kmax,
            gate_operand_bytes,
            gate_remote_wire_bytes,
            tail_operand_bytes,
            tail_remote_wire_bytes,
            scope="process",
        )
        if gate_compute_overlap_ticket is not None:
            from vllm_omni.diffusion.attention.ops.minimax_h3_vsa_gate_compute_overlap import (
                wait_h3_vsa_gate_compute_overlap,
            )

            wait_h3_vsa_gate_compute_overlap(gate_compute_overlap_ticket, gate)
        result = h3_vsa_o_bundle_local_gate_add_(
            output,
            gate,
            plan,
            owner_rank=rank,
        )
        if gate_compute_overlap_ticket is not None:
            gate_compute_overlap_ticket.mark_consumed(gate)
        return result

    @staticmethod
    def _validate_o_bundle_producer_output(
        attn_output: torch.Tensor,
        ctx: _UlyssesCtx,
    ) -> None:
        """Fail before reverse-O if the backend lost producer-direct storage."""
        state = ctx.o_bundle_state
        metadata = ctx.o_bundle_metadata
        gate = ctx.o_bundle_gate_local
        gate_compute_overlap_ticket = ctx.o_bundle_gate_compute_overlap_ticket
        if state is None and metadata is None and gate is None and gate_compute_overlap_ticket is None:
            return
        if state is None or metadata is None or gate is None:
            raise RuntimeError("incomplete MiniMax-H3 VSA reverse-O bundle context")
        if gate_compute_overlap_ticket is not None and not isinstance(
            gate_compute_overlap_ticket,
            H3VSAGateComputeOverlapTicket,
        ):
            raise TypeError("MiniMax-H3 VSA reverse-O bundle contains an invalid gate-compute overlap ticket")
        landing = metadata.extra.get("ulysses_o_input_landing")
        if not isinstance(landing, torch.Tensor):
            raise RuntimeError("MiniMax-H3 VSA backend did not retain its registered reverse-O landing")
        if (
            attn_output.data_ptr() != landing.data_ptr()
            or attn_output.storage_offset() != landing.storage_offset()
            or attn_output.shape != landing.shape
            or attn_output.stride() != landing.stride()
            or attn_output.dtype != landing.dtype
            or attn_output.device != landing.device
        ):
            raise RuntimeError(
                "MiniMax-H3 VSA reverse-O bundle lost producer-direct identity; staging fallback is forbidden"
            )

    def post_attention(self, attn_output: torch.Tensor, ctx: ParallelAttentionContext | None) -> torch.Tensor:
        assert isinstance(ctx, _UlyssesCtx), f"Unexpected ctx type: {type(ctx)!r}"
        if ctx.fp8_qkv_transport and attn_output.dtype != torch.bfloat16:
            raise TypeError(
                "MiniMax-H3 VSA FP8 QKV transport requires BF16 attention "
                f"output before reverse Ulysses, got {attn_output.dtype}"
            )
        from vllm_omni.diffusion.attention.ops.h3_overlap_experiment import finish_reverse

        projected = finish_reverse(attn_output, ctx)
        if projected is not None:
            return projected
        self._validate_o_bundle_producer_output(attn_output, ctx)

        if ctx.joint_len > 0:
            joint_len = ctx.joint_len

            if ctx.joint_strategy == "front":
                output_joint = attn_output[:, :joint_len]
                output_img = attn_output[:, joint_len:]
            else:
                output_img = attn_output[:, :-joint_len]
                output_joint = attn_output[:, -joint_len:]

            # 1. Process Image part: Standard Ulysses Reverse (AllToAll)
            # (bs, seq_len, head_cnt/P, head_size) -> (bs, seq_len/P, head_cnt, head_size)
            # SeqAllToAll4D handles: Scatter gather_idx, Gather scatter_idx.
            # Forward: Scatter 2 (H), Gather 1 (S).
            # Reverse: Scatter 1 (S), Gather 2 (H).
            if ctx.use_uaa:
                output_img = _ulysses_all_to_all_any_o(
                    ctx.ulysses_pg,
                    output_img,
                    seq_lens=list(ctx.uaa_seq_lens),
                    local_seq_len=ctx.uaa_local_seq_len,
                    orig_head_cnt=ctx.orig_head_cnt,
                    use_sync=ctx.use_sync,
                )
            elif ctx.strict_a2a_backend == "flashinfer-pcie":
                from vllm_omni.diffusion.distributed.flashinfer_ulysses import (
                    flashinfer_ulysses_o_rev,
                )

                output_img = flashinfer_ulysses_o_rev(
                    output_img,
                    ctx.ulysses_pg.group_name,
                    dist.get_world_size(ctx.ulysses_pg),
                    ctx.use_sync,
                )
            elif ctx.strict_a2a_backend == "symmetric":
                from vllm_omni.diffusion.distributed.a2a_permute import ulysses_o_rev

                output_img = ulysses_o_rev(
                    output_img,
                    ctx.ulysses_pg.group_name,
                    dist.get_world_size(ctx.ulysses_pg),
                )
            else:
                output_img = SeqAllToAll4D.apply(
                    ctx.ulysses_pg, output_img, ctx.gather_idx, ctx.scatter_idx, ctx.use_sync
                )

            # 2. Process Joint part: AllGather on Heads
            # Input: (B, JointLen, H/P, D). Output: (B, JointLen, H, D).
            # AllGather along dim 2.
            # Ensure tensor is contiguous for all_gather (slicing may create non-contiguous views)
            output_joint = output_joint.contiguous()
            gathered_joint = [torch.zeros_like(output_joint) for _ in range(dist.get_world_size(ctx.ulysses_pg))]
            dist.all_gather(gathered_joint, output_joint, group=ctx.ulysses_pg)
            output_joint = torch.cat(gathered_joint, dim=2)
            if ctx.use_uaa and ctx.joint_orig_head_cnt > 0 and output_joint.shape[2] != ctx.joint_orig_head_cnt:
                output_joint = output_joint[:, :, : ctx.joint_orig_head_cnt, :].contiguous()

            # 3. Recombine
            if ctx.joint_strategy == "front":
                return torch.cat([output_joint, output_img], dim=1)
            else:
                return torch.cat([output_img, output_joint], dim=1)

        # Standard Ulysses Reverse
        if ctx.use_uaa:
            output = _ulysses_all_to_all_any_o(
                ctx.ulysses_pg,
                attn_output,
                seq_lens=list(ctx.uaa_seq_lens),
                local_seq_len=ctx.uaa_local_seq_len,
                orig_head_cnt=ctx.orig_head_cnt,
                use_sync=ctx.use_sync,
            )
        elif ctx.strict_a2a_backend == "flashinfer-pcie":
            from vllm_omni.diffusion.distributed.flashinfer_ulysses import (
                flashinfer_ulysses_o_rev,
            )

            output = flashinfer_ulysses_o_rev(
                attn_output,
                ctx.ulysses_pg.group_name,
                dist.get_world_size(ctx.ulysses_pg),
                ctx.use_sync,
            )
        elif ctx.strict_a2a_backend == "symmetric":
            from vllm_omni.diffusion.distributed.a2a_permute import ulysses_o_rev

            output = ulysses_o_rev(
                attn_output,
                ctx.ulysses_pg.group_name,
                dist.get_world_size(ctx.ulysses_pg),
            )
        else:
            output = SeqAllToAll4D.apply(
                ctx.ulysses_pg,
                attn_output,
                ctx.gather_idx,
                ctx.scatter_idx,
                ctx.use_sync,
            )
        output = self._finish_o_bundle(output, ctx)
        output = self._finish_deferred_gate(output, ctx)
        if ctx.fp8_qkv_transport and output.dtype != torch.bfloat16:
            raise TypeError(
                f"MiniMax-H3 VSA FP8 QKV transport changed reverse-Ulysses output dtype from BF16 to {output.dtype}"
            )
        return output
