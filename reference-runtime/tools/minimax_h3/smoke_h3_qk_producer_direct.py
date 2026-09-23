#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Eight-rank exact-shape smoke for H3 Q/K producer-direct RDMA landing."""

from __future__ import annotations

import argparse
import json
import os
from typing import Any

import torch
import torch.distributed as dist

from vllm_omni.diffusion.distributed import flashinfer_ulysses as adapter
from vllm_omni.diffusion.layers.fused_qk_norm_rope import (
    fused_qk_norm_rope,
    fused_qk_norm_rope_out,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--local-seq-len", type=int, default=11936)
    parser.add_argument("--heads", type=int, default=56)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--rotary-dim", type=int, default=96)
    parser.add_argument("--seed", type=int, default=1101)
    return parser.parse_args()


def conventional_scatter(
    x: torch.Tensor,
    group: dist.ProcessGroup,
    world_size: int,
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


def main() -> None:
    args = parse_args()
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size != 8:
        raise SystemExit(f"this smoke requires world_size=8, got {world_size}")
    if args.heads % world_size:
        raise SystemExit("heads must be divisible by world size")
    if args.head_dim != 128 or args.rotary_dim != 96:
        raise SystemExit("the H3 Triton producer requires head_dim=128, rotary_dim=96")

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    group = dist.group.WORLD
    device = torch.device("cuda", local_rank)
    torch.manual_seed(args.seed + rank)

    # Match Q/K projection layout: each is a row-strided view of one packed QKV
    # allocation, while the producer's registered destinations are contiguous.
    qkv = torch.randn(
        args.local_seq_len,
        args.heads * args.head_dim * 3,
        dtype=torch.bfloat16,
        device=device,
    )
    width = args.heads * args.head_dim
    q, k, _v = qkv.split(width, dim=-1)
    q = q.view(args.local_seq_len, args.heads, args.head_dim)
    k = k.view(args.local_seq_len, args.heads, args.head_dim)
    if q.is_contiguous() or k.is_contiguous():
        raise RuntimeError("Q/K must remain row-strided projection views")

    q_weight = torch.randn(args.head_dim, dtype=torch.bfloat16, device=device)
    k_weight = torch.randn(args.head_dim, dtype=torch.bfloat16, device=device)
    freqs = torch.randn(
        args.local_seq_len,
        args.rotary_dim // 2,
        dtype=torch.float32,
        device=device,
    )
    rope_table = torch.cat((torch.cos(freqs), torch.sin(freqs)), dim=-1).to(torch.bfloat16)

    expected_q, expected_k = fused_qk_norm_rope(
        q,
        k,
        q_weight,
        k_weight,
        rope_table,
        1e-5,
    )
    expected_q_sp = conventional_scatter(expected_q.unsqueeze(0), group, world_size)
    expected_k_sp = conventional_scatter(expected_k.unsqueeze(0), group, world_size)

    landings = adapter.flashinfer_ulysses_qk_input_landings(
        q.unsqueeze(0),
        k.unsqueeze(0),
        group,
        world_size,
    )
    if landings is None:
        raise RuntimeError("the forced all-RDMA smoke unexpectedly declined input landing")
    q_landing, k_landing = landings
    actual_q, actual_k = fused_qk_norm_rope_out(
        q,
        k,
        q_weight,
        k_weight,
        rope_table,
        1e-5,
        q_out=q_landing.squeeze(0),
        k_out=k_landing.squeeze(0),
    )
    producer_ptr_exact = (
        actual_q.data_ptr() == q_landing.data_ptr()
        and actual_k.data_ptr() == k_landing.data_ptr()
    )
    actual_q_sp = adapter.flashinfer_ulysses_qkv_fwd(
        actual_q.unsqueeze(0), group.group_name, world_size, "q", False
    )
    actual_k_sp = adapter.flashinfer_ulysses_qkv_fwd(
        actual_k.unsqueeze(0), group.group_name, world_size, "k", False
    )
    torch.cuda.synchronize(device)

    state = next(iter(adapter._STATES.values()))
    local: dict[str, Any] = {
        "rank": rank,
        "transport": state.communicator.transport,
        "slots": len(state.outputs),
        "producer_ptr_exact": producer_ptr_exact,
        "producer_bit_exact": torch.equal(actual_q, expected_q)
        and torch.equal(actual_k, expected_k),
        "scatter_bit_exact": torch.equal(actual_q_sp, expected_q_sp)
        and torch.equal(actual_k_sp, expected_k_sp),
    }
    reports: list[dict[str, Any] | None] = [None] * world_size
    dist.all_gather_object(reports, local)
    failures = [
        report
        for report in reports
        if report is None
        or report["transport"] != "rdma"
        or report["slots"] != 2
        or not report["producer_ptr_exact"]
        or not report["producer_bit_exact"]
        or not report["scatter_bit_exact"]
    ]
    if rank == 0:
        print(
            "H3_QK_PRODUCER_DIRECT_SMOKE="
            + json.dumps(
                {
                    "shape": [1, args.local_seq_len, args.heads, args.head_dim],
                    "world_size": world_size,
                    "status": "pass" if not failures else "failed",
                    "rank_reports": reports,
                },
                sort_keys=True,
            ),
            flush=True,
        )

    adapter.clear_flashinfer_ulysses_communicators()
    dist.barrier()
    dist.destroy_process_group()
    if failures:
        raise SystemExit(f"H3 Q/K producer-direct smoke failed: {failures}")


if __name__ == "__main__":
    main()
