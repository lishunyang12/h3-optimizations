#!/usr/bin/env python3
"""Compare regular NCCL Ulysses with the symmetric-memory permute-free path.

The default shape models one TP rank of MiniMax-H3 at TP2 x SP4 for a
15-second request: batch 1, 9,472 local tokens, 28 local attention heads, and
head dimension 128.  Q/K/V are strided views of one fused projection output,
matching the layout that makes the regular transpose/contiguous path costly.
"""

from __future__ import annotations

import argparse
import json
import os

import torch
import torch.distributed as dist

from vllm_omni.diffusion.distributed.a2a_permute import (
    clear_a2a_permute_workspaces,
    ensure_a2a_permute_available,
    ulysses_o_rev,
    ulysses_qkv_fwd,
)
from vllm_omni.diffusion.distributed.comm import all_to_all_4D


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-seq-len", type=int, default=9472)
    parser.add_argument("--heads", type=int, default=28)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    return parser.parse_args()


def max_rank_ms(fn, *, warmups: int, iterations: int, device: torch.device) -> float:
    for _ in range(warmups):
        fn()
    torch.cuda.synchronize(device)
    dist.barrier()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        fn()
    end.record()
    end.synchronize()

    value = torch.tensor(start.elapsed_time(end) / iterations, dtype=torch.float64, device=device)
    dist.all_reduce(value, op=dist.ReduceOp.MAX)
    return float(value.item())


def main() -> None:
    args = parse_args()
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)

    if args.heads % world_size:
        raise ValueError(f"heads={args.heads} must be divisible by world_size={world_size}")

    ensure_a2a_permute_available()
    group = dist.group.WORLD
    group_name = group.group_name

    generator = torch.Generator(device=device).manual_seed(1101 + rank)
    fused_qkv = torch.randn(
        1,
        args.local_seq_len,
        3,
        args.heads,
        args.head_dim,
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )
    query, key, value = fused_qkv.unbind(dim=2)

    def regular_qkv(x: torch.Tensor) -> torch.Tensor:
        return all_to_all_4D(x, 2, 1, group=group)

    def regular_o(x: torch.Tensor) -> torch.Tensor:
        return all_to_all_4D(x, 1, 2, group=group)

    def lsa_qkv(x: torch.Tensor) -> torch.Tensor:
        return ulysses_qkv_fwd(x, group_name, world_size)

    def lsa_o(x: torch.Tensor) -> torch.Tensor:
        return ulysses_o_rev(x, group_name, world_size)

    regular_query = regular_qkv(query)
    lsa_query = lsa_qkv(query)
    if not torch.equal(regular_query, lsa_query):
        max_diff = (regular_query.float() - lsa_query.float()).abs().max().item()
        raise AssertionError(f"forward mismatch on rank {rank}: max_abs_diff={max_diff}")

    regular_roundtrip = regular_o(regular_query)
    lsa_roundtrip = lsa_o(lsa_query)
    if not torch.equal(regular_roundtrip, lsa_roundtrip) or not torch.equal(query, lsa_roundtrip):
        max_diff = (regular_roundtrip.float() - lsa_roundtrip.float()).abs().max().item()
        raise AssertionError(f"reverse mismatch on rank {rank}: max_abs_diff={max_diff}")

    regular_fwd_ms = max_rank_ms(
        lambda: regular_qkv(query), warmups=args.warmups, iterations=args.iterations, device=device
    )
    lsa_fwd_ms = max_rank_ms(
        lambda: lsa_qkv(query), warmups=args.warmups, iterations=args.iterations, device=device
    )
    regular_rev_ms = max_rank_ms(
        lambda: regular_o(regular_query), warmups=args.warmups, iterations=args.iterations, device=device
    )
    lsa_rev_ms = max_rank_ms(
        lambda: lsa_o(lsa_query), warmups=args.warmups, iterations=args.iterations, device=device
    )

    def regular_layer() -> None:
        regular_qkv(query)
        regular_qkv(key)
        regular_qkv(value)
        regular_o(regular_query)

    def lsa_layer() -> None:
        lsa_qkv(query)
        lsa_qkv(key)
        lsa_qkv(value)
        lsa_o(lsa_query)

    regular_layer_ms = max_rank_ms(
        regular_layer, warmups=args.warmups, iterations=args.iterations, device=device
    )
    lsa_layer_ms = max_rank_ms(
        lsa_layer, warmups=args.warmups, iterations=args.iterations, device=device
    )

    if rank == 0:
        payload = {
            "world_size": world_size,
            "shape": [1, args.local_seq_len, args.heads, args.head_dim],
            "dtype": "bfloat16",
            "qkv_layout": "strided_fused_projection_views",
            "correctness": "bit_exact",
            "regular_forward_ms": regular_fwd_ms,
            "lsa_forward_ms": lsa_fwd_ms,
            "forward_speedup": regular_fwd_ms / lsa_fwd_ms,
            "regular_reverse_ms": regular_rev_ms,
            "lsa_reverse_ms": lsa_rev_ms,
            "reverse_speedup": regular_rev_ms / lsa_rev_ms,
            "regular_qkv_layer_ms": regular_layer_ms,
            "lsa_qkv_layer_ms": lsa_layer_ms,
            "qkv_layer_speedup": regular_layer_ms / lsa_layer_ms,
        }
        print("A2A_BENCHMARK=" + json.dumps(payload, sort_keys=True))

    clear_a2a_permute_workspaces()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
