#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Validate deferred H3 VSA gating over real FlashInfer RDMA on SP2.

The optimized path writes fine O directly into FlashInfer's registered input,
performs the reverse-Ulysses gather, gathers only compact coarse rows through
NCCL, and applies the retained local BF16 gate.  A conventional NCCL
gate-before-reverse path is the bit-exact oracle.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402

from vllm_omni.diffusion.attention.backends.abstract import (  # noqa: E402
    AttentionMetadata,
    PackedPaddingMetadata,
)
from vllm_omni.diffusion.attention.ops.minimax_h3_vsa_deferred_gate import (  # noqa: E402
    H3_VSA_DEFERRED_ACTIVE_KEY,
    H3_VSA_DEFERRED_STATE_KEY,
    build_h3_compact_row_blocks,
)
from vllm_omni.diffusion.attention.parallel.ulysses import (  # noqa: E402
    UlyssesParallelAttention,
    _UlyssesCtx,
)
from vllm_omni.diffusion.distributed.comm import SeqAllToAll4D  # noqa: E402


def main() -> None:
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if world_size != 2:
        raise RuntimeError(f"this validator requires exactly two ranks, got {world_size}")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    from vllm_omni.diffusion.distributed import flashinfer_ulysses as adapter

    if not adapter.is_flashinfer_pcie_enabled():
        raise RuntimeError("VLLM_OMNI_ULYSSES_A2A_BACKEND must be flashinfer-pcie")
    if not adapter.is_o_producer_direct_enabled():
        raise RuntimeError("O producer-direct must be enabled")

    batch = 1
    local_rows = 64
    local_heads = 2
    head_dim = 128
    blocks = 3
    global_rows = world_size * local_rows
    valid_rows = global_rows - 1
    offsets = torch.arange(valid_rows, device=device, dtype=torch.int64)
    untile = (offsets % blocks) * 64 + (offsets // blocks)
    row_blocks = build_h3_compact_row_blocks(untile, num_blocks=blocks)

    generator = torch.Generator(device=device).manual_seed(1101 + rank)
    fine_global = torch.randn(
        batch,
        global_rows,
        local_heads,
        head_dim,
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )
    # Structural packed padding must remain an exact zero through both paths.
    fine_global[:, -1].zero_()
    coarse_local = torch.randn(
        batch,
        blocks,
        local_heads,
        head_dim,
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )
    gate_local = torch.randn(
        batch,
        local_rows,
        world_size * local_heads,
        head_dim,
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )

    # Oracle: communicate the full gate, apply it in global-sequence/local-head
    # layout, then perform the conventional NCCL reverse Ulysses exchange.
    gate_by_head_owner = SeqAllToAll4D.apply(
        dist.group.WORLD,
        gate_local,
        2,
        1,
        False,
    )
    baseline_global = fine_global.clone()
    selected_coarse = coarse_local.index_select(1, row_blocks.to(torch.int64))
    baseline_global[:, :valid_rows] = (
        baseline_global[:, :valid_rows] + selected_coarse * gate_by_head_owner[:, :valid_rows]
    )
    expected = SeqAllToAll4D.apply(
        dist.group.WORLD,
        baseline_global,
        1,
        2,
        False,
    )

    # Optimized transport: the fine producer targets FlashInfer's registered
    # source exactly, proving the O producer-direct path rather than merely the
    # functional RDMA gather.
    landing = adapter.flashinfer_ulysses_o_input_landing(
        fine_global,
        dist.group.WORLD,
        world_size,
    )
    if landing is None:
        raise RuntimeError("required FlashInfer RDMA O landing was declined")
    landing.copy_(fine_global)
    fine_local = adapter.flashinfer_ulysses_o_rev(
        landing,
        dist.group.WORLD.group_name,
        world_size,
        False,
    )

    state: dict[str, object] = {
        "coarse": coarse_local,
        "row_blocks": row_blocks,
    }
    cu_seqlens = torch.tensor([0, valid_rows], dtype=torch.int32, device=device)
    metadata = AttentionMetadata(
        packed_padding=PackedPaddingMetadata(
            q_length=valid_rows,
            kv_length=valid_rows,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
        ),
        extra={
            H3_VSA_DEFERRED_ACTIVE_KEY: True,
            H3_VSA_DEFERRED_STATE_KEY: state,
        },
    )
    context = _UlyssesCtx(
        name="ulysses",
        ulysses_pg=dist.group.WORLD,
        scatter_idx=2,
        gather_idx=1,
        use_sync=False,
        ulysses_rank=rank,
        strict_a2a_backend="flashinfer-pcie",
        deferred_gate_local=gate_local,
        deferred_gate_metadata=metadata,
        deferred_gate_state=state,
    )
    strategy = object.__new__(UlyssesParallelAttention)
    actual = strategy._finish_deferred_gate(fine_local, context)
    torch.accelerator.synchronize(device)

    adapter_states = list(adapter._STATES.values())
    transport = (
        adapter_states[0].communicator.transport
        if len(adapter_states) == 1 and adapter_states[0].communicator is not None
        else None
    )
    producer_direct = bool(
        len(adapter_states) == 1
        and adapter_states[0].o_producer_direct_logged
        and adapter_states[0].o_producer_direct_expected is None
    )
    output_storage_bytes = (
        next(iter(adapter_states[0].outputs.values())).untyped_storage().nbytes()
        if len(adapter_states) == 1 and adapter_states[0].outputs
        else 0
    )
    local_mismatches = int(torch.count_nonzero(actual.view(torch.int16) != expected.view(torch.int16)))
    report = {
        "rank": rank,
        "transport": transport,
        "o_producer_direct": producer_direct,
        "bit_exact": local_mismatches == 0,
        "mismatches": local_mismatches,
        "state_consumed": not state and not metadata.extra,
        "registered_output_storage_bytes": output_storage_bytes,
        "registered_input_storage_bytes": landing.untyped_storage().nbytes(),
    }
    reports: list[dict[str, object] | None] = [None] * world_size
    dist.all_gather_object(reports, report)
    failures = [
        item
        for item in reports
        if item is None
        or item["transport"] != "rdma"
        or item["o_producer_direct"] is not True
        or item["bit_exact"] is not True
        or item["state_consumed"] is not True
    ]
    if rank == 0:
        print(
            "H3_VSA_DEFERRED_GATE_RDMA_SP2="
            + json.dumps(
                {
                    "status": "pass" if not failures else "failed",
                    "world_size": world_size,
                    "shape": [batch, global_rows, local_heads, head_dim],
                    "reports": reports,
                },
                sort_keys=True,
            ),
            flush=True,
        )

    adapter.clear_flashinfer_ulysses_communicators()
    dist.barrier()
    dist.destroy_process_group()
    if failures:
        raise SystemExit(f"deferred-gate RDMA SP2 validation failed: {failures}")


if __name__ == "__main__":
    main()
