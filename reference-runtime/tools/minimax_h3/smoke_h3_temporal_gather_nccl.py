#!/usr/bin/env python3
"""Exercise the H3 temporal-PP destination gather with a real NCCL group."""

from __future__ import annotations

import os
from types import SimpleNamespace

import torch
import torch.distributed as dist

from vllm_omni.diffusion.models.minimax_h3.temporal_chunk_parallel import (
    _gather_stack_to_rank_zero,
)


def main() -> None:
    dist.init_process_group("nccl")
    try:
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        group = SimpleNamespace(
            world_size=world_size,
            rank_in_group=rank,
            ranks=list(range(world_size)),
            device_group=dist.group.WORLD,
        )
        source = torch.full((1, 3, 2, 4, 4), rank, dtype=torch.float16, device=device)
        gathered = _gather_stack_to_rank_zero(source, group)
        if rank == 0:
            assert gathered is not None
            expected = torch.stack(
                [torch.full_like(source, peer) for peer in range(world_size)]
            )
            torch.testing.assert_close(gathered, expected, rtol=0, atol=0)
            print(
                "temporal_gather_nccl=pass "
                f"world_size={world_size} shape={tuple(gathered.shape)}"
            )
        else:
            assert gathered is None
        dist.barrier()
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
