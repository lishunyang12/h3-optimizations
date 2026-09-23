#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Bounded production-shape microbenchmark for H3 VSA O piggyback ops."""

from __future__ import annotations

import argparse
import json
import statistics
import time

import torch

from vllm_omni.diffusion.attention.backends.fastvideo_vsa import (
    _get_h3_tile_metadata,
)
from vllm_omni.diffusion.attention.ops.minimax_h3_vsa_layout import (
    build_h3_aligned_untile_source_rows,
)
from vllm_omni.diffusion.attention.ops.minimax_h3_vsa_o_bundle import (
    h3_vsa_o_bundle_local_gate_add_,
    h3_vsa_o_bundle_out,
)
from vllm_omni.diffusion.attention.ops.minimax_h3_vsa_owner_route import (
    build_h3_vsa_owner_route_plan,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1101)
    return parser.parse_args()


def _median_cuda_ms(operation, *, warmup: int, iterations: int) -> float:
    for _ in range(warmup):
        operation()
    torch.accelerator.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
    for start, end in zip(starts, ends):
        start.record()
        operation()
        end.record()
    torch.accelerator.synchronize()
    return statistics.median(start.elapsed_time(end) for start, end in zip(starts, ends))


def main() -> None:
    args = _parse_args()
    if args.warmup < 0 or args.iterations <= 0:
        raise ValueError("warmup must be non-negative and iterations must be positive")
    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("the production-shape performance benchmark requires CUDA")

    prefix_segments = (121, 1206)
    video_shape = (107, 22, 40)
    sp_world_size = 8
    global_rows = 95_488
    local_rows = global_rows // sp_world_size
    local_heads = 7
    head_dim = 128

    plan_start = time.perf_counter()
    _partition, sizes, _non_pad, untile, _prefix_blocks, _video_blocks = _get_h3_tile_metadata(
        prefix_segments,
        video_shape,
        torch.device("cpu"),
    )
    source_rows = build_h3_aligned_untile_source_rows(
        untile,
        aligned_rows=global_rows,
        tiled_rows=sizes.numel() * 64,
    )
    plan = build_h3_vsa_owner_route_plan(
        source_rows,
        sp_world_size=sp_world_size,
        global_rows=global_rows,
        local_rows=local_rows,
        logical_block_count=sizes.numel(),
    )
    plan_ms = (time.perf_counter() - plan_start) * 1000

    torch.cuda.set_device(device)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    fine = torch.randn(
        1,
        global_rows,
        local_heads,
        head_dim,
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    ).mul_(0.1)
    coarse = torch.randn(
        1,
        plan.logical_block_count,
        local_heads,
        head_dim,
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    ).mul_(0.1)
    bundle = torch.empty(
        1,
        sp_world_size * (local_rows + plan.kmax),
        local_heads,
        head_dim,
        dtype=torch.bfloat16,
        device=device,
    )

    h3_vsa_o_bundle_out(fine, coarse, plan, out=bundle)
    torch.accelerator.synchronize()
    bundle_by_owner = bundle.view(
        1,
        sp_world_size,
        local_rows + plan.kmax,
        local_heads,
        head_dim,
    )
    for owner in range(sp_world_size):
        count = int(plan.owner_tile_counts[owner])
        ids = plan.owner_tile_ids[owner, :count].to(device=device, dtype=torch.int64)
        if not torch.equal(
            bundle_by_owner[:, owner, local_rows : local_rows + count],
            coarse.index_select(1, ids),
        ):
            raise RuntimeError(f"active coarse tail verification failed for owner {owner}")
        if count < plan.kmax and torch.count_nonzero(bundle_by_owner[:, owner, local_rows + count :]):
            raise RuntimeError(f"inactive coarse tail is nonzero for owner {owner}")

    bundle_ms = _median_cuda_ms(
        lambda: h3_vsa_o_bundle_out(fine, coarse, plan, out=bundle),
        warmup=args.warmup,
        iterations=args.iterations,
    )

    # A reverse all-to-all preserves the element count and produces this local
    # shape. Reuse the allocated bundle storage to keep peak memory bounded.
    reversed_bundle = bundle.view(
        1,
        local_rows + plan.kmax,
        sp_world_size * local_heads,
        head_dim,
    )
    gate = torch.randn(
        1,
        local_rows,
        sp_world_size * local_heads,
        head_dim,
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    ).mul_(0.01)
    gate_ms = _median_cuda_ms(
        lambda: h3_vsa_o_bundle_local_gate_add_(
            reversed_bundle,
            gate,
            plan,
            owner_rank=0,
        ),
        warmup=args.warmup,
        iterations=args.iterations,
    )

    report = {
        "device": torch.cuda.get_device_name(device),
        "geometry": {
            "global_rows": global_rows,
            "local_rows": local_rows,
            "logical_blocks": plan.logical_block_count,
            "sp_world_size": sp_world_size,
            "local_heads": local_heads,
            "head_dim": head_dim,
        },
        "route": {
            "owner_tile_counts": plan.owner_tile_counts.tolist(),
            "kmax": plan.kmax,
            "owner_tile_entries": plan.owner_tile_entries,
            "cross_owner_tiles": plan.cross_owner_tile_count,
            "plan_build_ms": plan_ms,
        },
        "latency_ms_p50": {
            "bundle_into_preallocated_send": bundle_ms,
            "local_exact_bf16_gate_add": gate_ms,
        },
        "allocated_mib": torch.accelerator.memory_allocated(device) / (1024**2),
        "peak_allocated_mib": torch.accelerator.max_memory_allocated(device) / (1024**2),
        "warmup": args.warmup,
        "iterations": args.iterations,
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
