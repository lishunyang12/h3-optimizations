#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Interleaved cuDNN SDPA head-count sweep for H3 rank balancing.

Each CUDA device is owned by an independent spawned process.  Shapes are
allocated separately so every BSHD input is contiguous; slicing an 8-head
allocation would leave the 6/7-head cases with production-incompatible row
strides.  Conditions are rotated each round to distribute clock/thermal drift.
There is no torch.distributed/NCCL initialization or GPU collective.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import queue
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

from benchmark_h3_cudnn_sdpa_balance import (
    H3AttentionShape,
    NvidiaSmiSampler,
    _physical_inventory,
    parse_visible_device_order,
    query_nvidia_smi,
    summarize_telemetry,
    summarize_timings,
)


def parse_head_counts(value: str) -> list[int]:
    heads = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not heads or any(head <= 0 for head in heads):
        raise ValueError("head counts must be a nonempty list of positive integers")
    if len(set(heads)) != len(heads):
        raise ValueError("head counts must be unique")
    return heads


def interleaved_schedule(
    heads: Sequence[int], rounds: int, rank_offset: int = 0
) -> list[int]:
    if not heads or rounds <= 0:
        raise ValueError("heads and rounds must be nonempty/positive")
    result: list[int] = []
    for round_index in range(rounds):
        offset = (round_index + rank_offset) % len(heads)
        result.extend([*heads[offset:], *heads[:offset]])
    return result


def scaled_candidate_projection(
    baseline_medians_ms: Sequence[float],
    *,
    donor_rank: int,
    donor_ratio: float,
    recipient_rank: int,
    recipient_ratio: float,
    layers: int = 50,
) -> dict[str, Any]:
    if len(baseline_medians_ms) != 8:
        raise ValueError("the H3 SP8 baseline must contain eight medians")
    if donor_rank == recipient_rank:
        raise ValueError("donor and recipient ranks must differ")
    projected = [float(value) for value in baseline_medians_ms]
    projected[donor_rank] *= donor_ratio
    projected[recipient_rank] *= recipient_ratio
    old_bottleneck = max(float(value) for value in baseline_medians_ms)
    new_bottleneck = max(projected)
    return {
        "projected_rank_medians_ms": projected,
        "uniform_bottleneck_ms_per_layer": old_bottleneck,
        "candidate_bottleneck_ms_per_layer": new_bottleneck,
        "saved_ms_per_layer": old_bottleneck - new_bottleneck,
        "saved_ms_per_step": (old_bottleneck - new_bottleneck) * layers,
        "speedup_attention_bottleneck": old_bottleneck / new_bottleneck,
    }


def _worker(
    logical_rank: int,
    sequence: int,
    head_dim: int,
    heads: list[int],
    warmup: int,
    rounds: int,
    iterations: int,
    seed: int,
    start_gate: Any,
    messages: Any,
) -> None:
    try:
        import torch
        import torch.nn.functional as F
        from torch.nn.attention import SDPBackend, sdpa_kernel

        torch.cuda.set_device(logical_rank)
        device = torch.device("cuda", logical_rank)
        calls: dict[int, Any] = {}
        outputs: dict[int, torch.Tensor | None] = {}
        tensors_by_head: dict[int, list[torch.Tensor]] = {}
        for head_count in heads:
            shape = H3AttentionShape(
                sequence=sequence, heads=head_count, head_dim=head_dim
            ).validate()
            generator = torch.Generator(device=device)
            generator.manual_seed(seed + head_count)
            tensors = []
            for _ in range(3):
                tensor = torch.empty(
                    shape.tensor_shape, dtype=torch.bfloat16, device=device
                )
                tensor.normal_(generator=generator)
                tensors.append(tensor)
            if not all(tensor.is_contiguous() for tensor in tensors):
                raise RuntimeError("head-sweep BSHD inputs must be contiguous")
            tensors_by_head[head_count] = tensors
            q_bhsd, k_bhsd, v_bhsd = (
                tensor.permute(0, 2, 1, 3) for tensor in tensors
            )

            def attention(
                q: torch.Tensor = q_bhsd,
                k: torch.Tensor = k_bhsd,
                v: torch.Tensor = v_bhsd,
                scale: float = shape.softmax_scale,
            ) -> torch.Tensor:
                with sdpa_kernel([SDPBackend.CUDNN_ATTENTION]):
                    return F.scaled_dot_product_attention(
                        q,
                        k,
                        v,
                        attn_mask=None,
                        dropout_p=0.0,
                        is_causal=False,
                        scale=scale,
                        enable_gqa=False,
                    )

            calls[head_count] = attention
            outputs[head_count] = None

        warmup_schedule = interleaved_schedule(heads, warmup, logical_rank)
        for head_count in warmup_schedule:
            outputs[head_count] = calls[head_count]()
        torch.cuda.synchronize(device)
        messages.put({"kind": "ready", "logical_rank": logical_rank})
        if not start_gate.wait(timeout=180.0):
            raise TimeoutError("timed out waiting for the host start gate")

        samples: dict[int, list[float]] = {head_count: [] for head_count in heads}
        schedule = interleaved_schedule(heads, rounds, logical_rank)
        for head_count in schedule:
            for _ in range(iterations):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                outputs[head_count] = calls[head_count]()
                end.record()
                end.synchronize()
                samples[head_count].append(float(start.elapsed_time(end)))

        checksums = {
            str(head_count): float(outputs[head_count][0, 0, 0, 0].float().item())
            for head_count in heads
            if outputs[head_count] is not None
        }
        messages.put(
            {
                "kind": "result",
                "status": "ok",
                "logical_rank": logical_rank,
                "samples": samples,
                "checksums": checksums,
                "memory_allocated_bytes": int(torch.cuda.memory_allocated(device)),
                "memory_reserved_bytes": int(torch.cuda.memory_reserved(device)),
            }
        )
    except BaseException as exc:
        status = "oom" if type(exc).__name__ == "OutOfMemoryError" else "error"
        messages.put(
            {
                "kind": "result",
                "status": status,
                "logical_rank": logical_rank,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--devices", type=int, default=2)
    parser.add_argument("--heads", default="6,7,8")
    parser.add_argument("--sequence", type=int, default=95_488)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1101)
    parser.add_argument("--worker-timeout", type=float, default=300.0)
    parser.add_argument("--smi-interval", type=float, default=0.25)
    parser.add_argument("--json-out", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    heads = parse_head_counts(args.heads)
    visible = parse_visible_device_order(
        os.environ.get("CUDA_VISIBLE_DEVICES"), args.devices
    )
    static_smi = query_nvidia_smi()
    inventory = _physical_inventory(static_smi)

    context = mp.get_context("spawn")
    start_gate = context.Event()
    messages = context.Queue()
    workers = [
        context.Process(
            target=_worker,
            args=(
                rank,
                args.sequence,
                args.head_dim,
                heads,
                args.warmup,
                args.rounds,
                args.iterations,
                args.seed,
                start_gate,
                messages,
            ),
        )
        for rank in range(args.devices)
    ]
    for worker in workers:
        worker.start()

    ready: set[int] = set()
    results_by_rank: dict[int, dict[str, Any]] = {}
    while len(ready) + len(results_by_rank) < args.devices:
        try:
            message = messages.get(timeout=args.worker_timeout)
        except queue.Empty:
            break
        rank = int(message["logical_rank"])
        if message["kind"] == "ready":
            ready.add(rank)
        else:
            results_by_rank[rank] = message

    sampler = NvidiaSmiSampler(args.smi_interval)
    sampler.start()
    start_gate.set()
    while len(results_by_rank) < args.devices:
        try:
            message = messages.get(timeout=args.worker_timeout)
        except queue.Empty:
            break
        if message["kind"] == "result":
            results_by_rank[int(message["logical_rank"])] = message
    sampler.stop()

    for worker in workers:
        worker.join(timeout=5.0)
        if worker.is_alive():
            worker.terminate()
            worker.join(timeout=5.0)

    results: list[dict[str, Any]] = []
    for rank in range(args.devices):
        result = results_by_rank.get(
            rank,
            {
                "kind": "result",
                "status": "timeout",
                "logical_rank": rank,
                "error": "worker did not report a result",
            },
        )
        physical_id = visible[rank]
        result["physical_id"] = physical_id
        result["pci_bus_id"] = inventory.get(physical_id, {}).get("pci.bus_id")
        if result.get("status") == "ok":
            result["timings_by_head"] = {
                str(head_count): asdict(summarize_timings(samples))
                for head_count, samples in result.pop("samples").items()
            }
        result["telemetry"] = summarize_telemetry(sampler.samples, physical_id)
        results.append(result)

    payload = {
        "contract": {
            "independent_workers": True,
            "gpu_collectives": False,
            "host_start_gate_only": True,
            "separately_allocated_contiguous_bshd_per_head_count": True,
            "backend": "SDPBackend.CUDNN_ATTENTION",
            "sequence": args.sequence,
            "head_dim": args.head_dim,
            "attention_mask": None,
            "is_causal": False,
        },
        "configuration": {
            "visible_device_order": visible,
            "heads": heads,
            "warmup_rotations": args.warmup,
            "rounds": args.rounds,
            "iterations_per_condition_per_round": args.iterations,
            "samples_per_head_per_gpu": args.rounds * args.iterations,
            "smi_interval_seconds": args.smi_interval,
        },
        "results": results,
        "sampler_errors": sampler.errors,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    return 0 if all(result.get("status") == "ok" for result in results) else 2


if __name__ == "__main__":
    raise SystemExit(main())
