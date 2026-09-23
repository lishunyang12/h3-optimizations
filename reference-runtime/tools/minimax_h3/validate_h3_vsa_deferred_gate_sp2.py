#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Validate MiniMax-H3 deferred VSA gate on a real two-rank NCCL group.

Example:
    CUDA_VISIBLE_DEVICES=4,5 torchrun --standalone --nproc-per-node=2 \
        tools/minimax_h3/validate_h3_vsa_deferred_gate_sp2.py
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
    _all_gather_h3_vsa_deferred_coarse,
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

    batch = 1
    local_rows = 17
    local_heads = 2
    head_dim = 128
    blocks = 3
    valid_rows = world_size * local_rows - 1
    # Every compact row gets a unique tiled source position while cycling over
    # all three coarse tiles. The last aligned row is structural suffix padding.
    offsets = torch.arange(valid_rows, device=device, dtype=torch.int64)
    untile = (offsets % 3) * 64 + (offsets // 3)
    row_blocks = build_h3_compact_row_blocks(untile, num_blocks=blocks)

    generator = torch.Generator(device=device).manual_seed(1101 + rank)
    coarse_local = torch.randn(
        batch,
        blocks,
        local_heads,
        head_dim,
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )
    gathered = _all_gather_h3_vsa_deferred_coarse(
        dist.group.WORLD,
        coarse_local,
        use_sync=False,
    )

    # Verify rank-major head ownership independently of the gate arithmetic.
    gathered_parts = [torch.empty_like(coarse_local) for _ in range(world_size)]
    dist.all_gather(gathered_parts, coarse_local)
    expected_gather = torch.cat(gathered_parts, dim=2)
    if not torch.equal(gathered, expected_gather):
        raise AssertionError("compact coarse all-gather changed rank/head order")

    global_rows = world_size * local_rows
    fine_by_head_owner = torch.randn(
        batch,
        global_rows,
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

    # Old route: all-to-all the full gate, gate in global-sequence/local-head
    # layout, then reverse all-to-all the combined output.
    gate_by_head_owner = SeqAllToAll4D.apply(
        dist.group.WORLD,
        gate_local,
        2,
        1,
        False,
    )
    selected_coarse = coarse_local.index_select(1, row_blocks.to(torch.int64))
    baseline_by_head_owner = fine_by_head_owner.clone()
    baseline_by_head_owner[:, :valid_rows] = (
        baseline_by_head_owner[:, :valid_rows]
        + selected_coarse * gate_by_head_owner[:, :valid_rows]
    )
    expected = SeqAllToAll4D.apply(
        dist.group.WORLD,
        baseline_by_head_owner,
        1,
        2,
        False,
    )

    # New route: reverse only fine O, retain the original local gate, and let
    # _finish_deferred_gate all-gather compact coarse state before the exact
    # local BF16 multiply/add.
    fine_local = SeqAllToAll4D.apply(
        dist.group.WORLD,
        fine_by_head_owner,
        1,
        2,
        False,
    )
    actual = fine_local.clone()
    state: dict[str, object] = {
        "coarse": coarse_local,
        "row_blocks": row_blocks,
    }
    cu_seqlens = torch.tensor(
        [0, valid_rows],
        dtype=torch.int32,
        device=device,
    )
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
        }
    )
    context = _UlyssesCtx(
        name="ulysses",
        ulysses_pg=dist.group.WORLD,
        scatter_idx=2,
        gather_idx=1,
        use_sync=False,
        ulysses_rank=rank,
        deferred_gate_local=gate_local,
        deferred_gate_metadata=metadata,
        deferred_gate_state=state,
    )
    strategy = object.__new__(UlyssesParallelAttention)
    result = strategy._finish_deferred_gate(
        actual,
        context,
    )
    if result.data_ptr() != actual.data_ptr():
        raise AssertionError("deferred gate did not mutate reverse-O storage in place")
    if state or metadata.extra:
        raise AssertionError("deferred gate state was not consumed exactly once")
    mismatches = torch.count_nonzero(actual.view(torch.int16) != expected.view(torch.int16))
    mismatch_count = torch.tensor([int(mismatches)], dtype=torch.int64, device=device)
    dist.all_reduce(mismatch_count, op=dist.ReduceOp.SUM)
    if int(mismatch_count.item()) != 0:
        raise AssertionError(f"deferred gate differs by {int(mismatch_count.item())} BF16 values")

    if rank == 0:
        full_gate_a2a_bytes = gate_local.numel() * gate_local.element_size()
        compact_contribution_bytes = coarse_local.numel() * coarse_local.element_size()
        gathered_coarse_bytes = gathered.numel() * gathered.element_size()
        print(
            json.dumps(
                {
                    "world_size": world_size,
                    "bit_exact": True,
                    "full_gate_a2a_bytes_per_rank": full_gate_a2a_bytes,
                    "compact_coarse_contribution_bytes_per_rank": compact_contribution_bytes,
                    "gathered_coarse_bytes_per_rank": gathered_coarse_bytes,
                    "head_order_exact": True,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
