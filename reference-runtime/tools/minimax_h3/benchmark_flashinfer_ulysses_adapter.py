#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Eight-rank correctness/compile smoke for the FlashInfer PCIe adapter.

Launch with torchrun.  This intentionally uses the MiniMax-H3 TP1/SP8 tensor
geometry and Q/K/V views from one fused projection allocation.  The views are
row-strided, matching the case where the adapter builds directly into an RDMA
landing buffer instead of first materializing ``contiguous()``.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any

import torch
import torch.distributed as dist


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--local-seq-len", type=int, default=4736)
    parser.add_argument("--heads", type=int, default=56)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--seed", type=int, default=1101)
    return parser.parse_args()


def conventional_scatter(
    x: torch.Tensor, group: dist.ProcessGroup, world_size: int
) -> torch.Tensor:
    batch, local_seq, heads, head_dim = x.shape
    local_heads = heads // world_size
    send = (
        x.reshape(batch, local_seq, world_size, local_heads, head_dim)
        .transpose(0, 2)
        .contiguous()
    )
    recv = torch.empty_like(send)
    dist.all_to_all_single(recv, send, group=group)
    return (
        recv.reshape(local_seq * world_size, batch, local_heads, head_dim)
        .transpose(0, 1)
        .contiguous()
        .reshape(batch, local_seq * world_size, local_heads, head_dim)
    )


def conventional_gather(
    x: torch.Tensor, group: dist.ProcessGroup, world_size: int
) -> torch.Tensor:
    batch, global_seq, local_heads, head_dim = x.shape
    local_seq = global_seq // world_size
    heads = local_heads * world_size
    send = (
        x.reshape(batch, world_size, local_seq, local_heads, head_dim)
        .transpose(0, 3)
        .transpose(0, 1)
        .contiguous()
        .reshape(world_size, local_heads, local_seq, batch, head_dim)
    )
    recv = torch.empty_like(send)
    dist.all_to_all_single(recv, send, group=group)
    return (
        recv.reshape(heads, local_seq, batch, head_dim)
        .transpose(0, 2)
        .contiguous()
        .reshape(batch, local_seq, heads, head_dim)
    )


def main() -> None:
    args = parse_args()
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size != 8:
        raise SystemExit(f"this H3 adapter smoke requires world_size=8, got {world_size}")
    if args.heads % world_size:
        raise SystemExit(f"heads={args.heads} must be divisible by world_size={world_size}")

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    group = dist.group.WORLD
    device = torch.device("cuda", local_rank)

    from vllm_omni.diffusion.distributed import flashinfer_ulysses as adapter

    group_name = group.group_name
    torch.manual_seed(args.seed + rank)
    # Unbinding the QKV axis leaves three non-contiguous [B,S,H,D] views.
    fused_qkv = torch.randn(
        (1, args.local_seq_len, 3, args.heads, args.head_dim),
        dtype=torch.bfloat16,
        device=device,
    )
    q, k, v = fused_qkv.unbind(dim=2)
    if any(tensor.is_contiguous() for tensor in (q, k, v)):
        raise RuntimeError("the fused-QKV smoke must exercise non-contiguous projection views")

    refs = tuple(conventional_scatter(x, group, world_size) for x in (q, k, v))
    ref_o = conventional_gather(refs[0], group, world_size)

    optimized = tuple(
        adapter.flashinfer_ulysses_qkv_fwd(
            x, group_name, world_size, slot, False
        )
        for x, slot in zip((q, k, v), ("q", "k", "v"), strict=True)
    )
    optimized_o = adapter.flashinfer_ulysses_o_rev(
        optimized[0], group_name, world_size, False
    )
    eager_exact = all(torch.equal(actual, expected) for actual, expected in zip(optimized, refs, strict=True))
    eager_exact = eager_exact and torch.equal(optimized_o, ref_o)

    def compiled_roundtrip(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        query_sp = adapter.flashinfer_ulysses_qkv_fwd(
            query, group_name, world_size, "q", False
        )
        key_sp = adapter.flashinfer_ulysses_qkv_fwd(
            key, group_name, world_size, "k", False
        )
        value_sp = adapter.flashinfer_ulysses_qkv_fwd(
            value, group_name, world_size, "v", False
        )
        output = adapter.flashinfer_ulysses_o_rev(
            query_sp, group_name, world_size, False
        )
        return query_sp, key_sp, value_sp, output

    compiled = torch.compile(compiled_roundtrip, fullgraph=True)
    compiled_outputs = compiled(q, k, v)
    compiled_exact = all(
        torch.equal(actual, expected)
        for actual, expected in zip(compiled_outputs, (*refs, ref_o), strict=True)
    )
    # A second call proves the warm registered Q/K/V/O slots are reusable.
    compiled_outputs_2 = compiled(q, k, v)
    compiled_reuse_exact = all(
        torch.equal(actual, expected)
        for actual, expected in zip(compiled_outputs_2, (*refs, ref_o), strict=True)
    )
    torch.cuda.synchronize(device)

    states = list(adapter._STATES.values())
    transport = (
        states[0].communicator.transport
        if len(states) == 1 and states[0].communicator is not None
        else None
    )
    local: dict[str, Any] = {
        "rank": rank,
        "transport": transport,
        "state_count": len(states),
        "output_slots": len(states[0].outputs) if len(states) == 1 else -1,
        "input_contiguous": [q.is_contiguous(), k.is_contiguous(), v.is_contiguous()],
        "eager_bit_exact": eager_exact,
        "compiled_bit_exact": compiled_exact,
        "compiled_reuse_bit_exact": compiled_reuse_exact,
    }
    reports: list[dict[str, Any] | None] = [None] * world_size
    dist.all_gather_object(reports, local)

    failures = [
        report
        for report in reports
        if report is None
        or report["transport"] != "rdma"
        or report["state_count"] != 1
        or report["output_slots"] != 4
        or any(report["input_contiguous"])
        or not report["eager_bit_exact"]
        or not report["compiled_bit_exact"]
        or not report["compiled_reuse_bit_exact"]
    ]
    if rank == 0:
        payload = {
            "world_size": world_size,
            "shape": [1, args.local_seq_len, args.heads, args.head_dim],
            "dtype": "bfloat16",
            "transport": "rdma" if not failures else "invalid",
            "fused_qkv_views": "noncontiguous",
            "eager_correctness": "bit_exact" if not failures else "failed",
            "torch_compile_fullgraph": "pass" if not failures else "failed",
            "warm_slot_reuse": "pass" if not failures else "failed",
            "rank_reports": reports,
        }
        print("FLASHINFER_ADAPTER_SMOKE=" + json.dumps(payload, sort_keys=True), flush=True)

    # The communicator close is collective, so every rank reaches it before a
    # correctness failure is raised and before the process group is destroyed.
    adapter.clear_flashinfer_ulysses_communicators()
    dist.barrier()
    dist.destroy_process_group()
    if failures:
        raise SystemExit(f"FlashInfer adapter smoke failed: {failures}")


if __name__ == "__main__":
    main()
