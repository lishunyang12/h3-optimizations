#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Validate mixed-dtype FlashInfer Ulysses on a real NCCL process group.

The forward exchange carries raw E4M3 Q/K/V bytes from registered producer
landings. The reverse exchange carries BF16 O through a distinct registered
slot owned by the same FlashInfer PCIe/RDMA communicator.

Example:
    CUDA_VISIBLE_DEVICES=4,5 torchrun --standalone --nproc-per-node=2 \
        tools/minimax_h3/validate_flashinfer_ulysses_fp8_qkv.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--local-seq-len", type=int, default=256)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--dequant-scale", type=float, default=0.125)
    parser.add_argument("--seed", type=int, default=1101)
    parser.add_argument(
        "--iterations",
        type=int,
        default=8,
        help="persistent-communicator iterations (iteration zero is cold)",
    )
    parser.add_argument(
        "--sync-after-producer",
        action="store_true",
        help="diagnostic device sync after Q/K/V producers",
    )
    parser.add_argument(
        "--sync-after-each-collective",
        action="store_true",
        help="diagnostic device sync after every forward/reverse exchange",
    )
    parser.add_argument(
        "--hash-iterations",
        choices=("none", "cold", "all"),
        default="cold",
        help="SHA256 tensor hashing policy; cold hashes iteration zero only",
    )
    return parser.parse_args()


_FP8_POISON_BYTE = 0x7F  # positive E4M3FN NaN; finite producers cannot emit it


def _poison_fp8(tensor: torch.Tensor) -> None:
    if tensor.dtype != torch.float8_e4m3fn:
        raise TypeError(f"FP8 poison requires E4M3FN, got {tensor.dtype}")
    tensor.view(torch.uint8).fill_(_FP8_POISON_BYTE)


def _poison_bf16(tensor: torch.Tensor) -> None:
    if tensor.dtype != torch.bfloat16:
        raise TypeError(f"BF16 poison requires BF16, got {tensor.dtype}")
    tensor.fill_(math.nan)


def _byte_mismatches(actual: torch.Tensor, expected: torch.Tensor) -> int:
    return int(torch.count_nonzero(actual.view(torch.uint8) != expected.view(torch.uint8)).item())


def _poison_bytes_remaining(tensor: torch.Tensor) -> int:
    return int(torch.count_nonzero(tensor.view(torch.uint8) == _FP8_POISON_BYTE).item())


def _bf16_mismatches(actual: torch.Tensor, expected: torch.Tensor) -> int:
    return int(torch.count_nonzero(actual != expected).item())


def _bf16_poison_remaining(tensor: torch.Tensor) -> int:
    return int(torch.count_nonzero(torch.isnan(tensor)).item())


def _max_abs_error(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    scale: float = 1.0,
) -> tuple[float | None, int]:
    error = (actual.float() * scale - expected.float() * scale).abs()
    finite = torch.isfinite(error)
    nonfinite = int(torch.count_nonzero(~finite).item())
    if nonfinite:
        return None, nonfinite
    return float(error.max().item()), 0


def _tensor_sha256(tensor: torch.Tensor) -> str:
    """Hash exact tensor bytes without a second Python bytes copy."""

    cpu_bytes = tensor.detach().contiguous().view(torch.uint8).cpu()
    return hashlib.sha256(cpu_bytes.numpy()).hexdigest()


def _conventional_scatter_bytes(
    x: torch.Tensor,
    group: dist.ProcessGroup,
    world_size: int,
) -> torch.Tensor:
    """Reference Ulysses scatter using NCCL uint8 payloads."""
    if x.dtype != torch.uint8 or not x.is_contiguous():
        raise TypeError("byte reference requires contiguous uint8 input")
    batch, local_seq, heads, head_dim = x.shape
    local_heads = heads // world_size
    send = x.reshape(batch, local_seq, world_size, local_heads, head_dim).transpose(0, 2).contiguous()
    recv = torch.empty_like(send)
    dist.all_to_all_single(recv, send, group=group)
    return (
        recv.reshape(local_seq * world_size, batch, local_heads, head_dim)
        .transpose(0, 1)
        .contiguous()
        .reshape(batch, local_seq * world_size, local_heads, head_dim)
    )


def _conventional_gather(
    x: torch.Tensor,
    group: dist.ProcessGroup,
    world_size: int,
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
    args = _parse_args()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if world_size <= 1:
        raise RuntimeError("mixed-dtype Ulysses validation requires at least two ranks")
    if args.heads % world_size:
        raise ValueError(f"heads={args.heads} must divide world_size={world_size}")
    if args.local_seq_len <= 0 or args.head_dim <= 0:
        raise ValueError("tensor dimensions must be positive")
    if not math.isfinite(args.dequant_scale) or args.dequant_scale <= 0:
        raise ValueError("dequant scale must be finite and positive")
    if args.iterations <= 0:
        raise ValueError("iterations must be positive")

    from vllm_omni.diffusion.distributed import flashinfer_ulysses as adapter
    from vllm_omni.diffusion.layers.fused_qk_norm_rope import (
        fused_qk_norm_rope,
        fused_qk_norm_rope_e4m3_out,
        quantize_v_e4m3_out,
    )

    group = dist.group.WORLD
    iteration_reports: list[dict[str, Any]] = []
    first_landing_ptrs: tuple[int, ...] | None = None
    first_forward_output_ptrs: tuple[int, ...] | None = None
    first_reverse_output_ptr: int | None = None

    for iteration in range(args.iterations):
        # Change every iteration's values so stale buffers cannot accidentally
        # satisfy exactness checks.  Rank-major seeding keeps reruns reproducible.
        generator = torch.Generator(device=device).manual_seed(args.seed + rank + iteration * world_size)
        # Match the real packed qkv projection: unbind produces three
        # row-strided BF16 views, while the quantizing producer targets
        # contiguous registered FP8 landings.
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
        q, k, v = fused_qkv.unbind(dim=2)
        if any(tensor.is_contiguous() for tensor in (q, k, v)):
            raise AssertionError("validator did not construct strided Q/K/V inputs")

        q_3d = q.squeeze(0)
        k_3d = k.squeeze(0)
        v_3d = v.squeeze(0)
        q_weight = torch.randn(
            args.head_dim,
            dtype=torch.bfloat16,
            device=device,
            generator=generator,
        )
        k_weight = torch.randn(
            args.head_dim,
            dtype=torch.bfloat16,
            device=device,
            generator=generator,
        )
        angles = torch.randn(
            args.local_seq_len,
            48,
            dtype=torch.float32,
            device=device,
            generator=generator,
        )
        rope_table = torch.cat((angles.cos(), angles.sin()), dim=-1).to(torch.bfloat16)
        q_ref, k_ref = fused_qk_norm_rope(
            q_3d,
            k_3d,
            q_weight,
            k_weight,
            rope_table,
            1.0e-5,
        )
        expected_local = (
            (q_ref.float() / args.dequant_scale).to(torch.float8_e4m3fn).unsqueeze(0),
            (k_ref.float() / args.dequant_scale).to(torch.float8_e4m3fn).unsqueeze(0),
            (v_3d.float() / args.dequant_scale).to(torch.float8_e4m3fn).unsqueeze(0),
        )
        expected_local_poison = [_poison_bytes_remaining(expected) for expected in expected_local]
        expected_forward = tuple(
            _conventional_scatter_bytes(
                expected.view(torch.uint8),
                group,
                world_size,
            ).view(torch.float8_e4m3fn)
            for expected in expected_local
        )
        expected_forward_poison = [_poison_bytes_remaining(expected) for expected in expected_forward]
        bf16_o = (expected_forward[0].float() * args.dequant_scale).to(torch.bfloat16)
        expected_reverse = _conventional_gather(bf16_o, group, world_size)

        landings = adapter.flashinfer_ulysses_qkv_e4m3_input_landings(
            q,
            k,
            v,
            group,
            world_size,
        )
        landing_ptrs = tuple(landing.data_ptr() for landing in landings)
        if first_landing_ptrs is None:
            first_landing_ptrs = landing_ptrs
        landing_ptrs_reused = landing_ptrs == first_landing_ptrs
        for landing in landings:
            _poison_fp8(landing)

        fused_qk_norm_rope_e4m3_out(
            q_3d,
            k_3d,
            q_weight,
            k_weight,
            rope_table,
            1.0e-5,
            q_dequant_scale=args.dequant_scale,
            k_dequant_scale=args.dequant_scale,
            q_out=landings[0].squeeze(0),
            k_out=landings[1].squeeze(0),
        )
        quantize_v_e4m3_out(
            v_3d,
            dequant_scale=args.dequant_scale,
            out=landings[2].squeeze(0),
        )
        if args.sync_after_producer:
            torch.accelerator.synchronize(device)

        states = list(adapter._STATES.values())
        if len(states) != 1:
            raise RuntimeError(f"expected one FlashInfer Ulysses state, observed {len(states)}")
        state = states[0]
        forward_output_views = tuple(
            state.output(landing, op="scatter_heads", slot=slot)
            for landing, slot in zip(landings, ("q", "k", "v"), strict=True)
        )
        forward_output_ptrs = tuple(output.data_ptr() for output in forward_output_views)
        if first_forward_output_ptrs is None:
            first_forward_output_ptrs = forward_output_ptrs
        forward_output_ptrs_reused = forward_output_ptrs == first_forward_output_ptrs
        for output in forward_output_views:
            _poison_fp8(output)

        actual_forward_parts = []
        for landing, slot in zip(landings, ("q", "k", "v"), strict=True):
            actual_forward_parts.append(
                adapter.flashinfer_ulysses_qkv_fwd(
                    landing,
                    group.group_name,
                    world_size,
                    slot,
                    False,
                )
            )
            if args.sync_after_each_collective:
                torch.accelerator.synchronize(device)
        actual_forward = tuple(actual_forward_parts)

        o_landing = adapter.flashinfer_ulysses_o_input_landing(
            expected_forward[0],
            group,
            world_size,
            output_dtype=torch.bfloat16,
        )
        if o_landing is None:
            raise RuntimeError("BF16 reverse-O registered landing was not acquired")
        _poison_bf16(o_landing)
        o_landing.copy_(bf16_o)
        reverse_output_view = state.output(
            o_landing,
            op="gather_heads",
            slot="o",
        )
        reverse_output_ptr = reverse_output_view.data_ptr()
        if first_reverse_output_ptr is None:
            first_reverse_output_ptr = reverse_output_ptr
        reverse_output_ptr_reused = reverse_output_ptr == first_reverse_output_ptr
        _poison_bf16(reverse_output_view)
        actual_reverse = adapter.flashinfer_ulysses_o_rev(
            o_landing,
            group.group_name,
            world_size,
            False,
        )
        if args.sync_after_each_collective:
            torch.accelerator.synchronize(device)

        # Diagnostics intentionally happen only after the full iteration.
        # Their scalar reads synchronize eventually, but do not insert a
        # producer/collective boundary unless a --sync-* flag requests one.
        producer_mismatches = [
            _byte_mismatches(landing, expected) for landing, expected in zip(landings, expected_local, strict=True)
        ]
        producer_poison_remaining = [_poison_bytes_remaining(landing) for landing in landings]
        producer_errors = [
            _max_abs_error(
                landing,
                expected,
                scale=args.dequant_scale,
            )
            for landing, expected in zip(landings, expected_local, strict=True)
        ]
        forward_mismatches = [
            _byte_mismatches(actual, expected)
            for actual, expected in zip(
                actual_forward,
                expected_forward,
                strict=True,
            )
        ]
        forward_poison_remaining = [_poison_bytes_remaining(actual) for actual in actual_forward]
        o_producer_mismatches = _bf16_mismatches(o_landing, bf16_o)
        o_producer_poison_remaining = _bf16_poison_remaining(o_landing)
        reverse_mismatches = _bf16_mismatches(actual_reverse, expected_reverse)
        reverse_poison_remaining = _bf16_poison_remaining(actual_reverse)
        reverse_max_abs_error, reverse_nonfinite_errors = _max_abs_error(
            actual_reverse,
            expected_reverse,
        )
        hash_this_iteration = args.hash_iterations == "all" or (args.hash_iterations == "cold" and iteration == 0)
        producer_sha256_qkv = [_tensor_sha256(tensor) for tensor in landings] if hash_this_iteration else None
        expected_local_sha256_qkv = (
            [_tensor_sha256(tensor) for tensor in expected_local] if hash_this_iteration else None
        )
        forward_sha256_qkv = [_tensor_sha256(tensor) for tensor in actual_forward] if hash_this_iteration else None
        expected_forward_sha256_qkv = (
            [_tensor_sha256(tensor) for tensor in expected_forward] if hash_this_iteration else None
        )
        o_producer_sha256 = _tensor_sha256(o_landing) if hash_this_iteration else None
        expected_o_sha256 = _tensor_sha256(bf16_o) if hash_this_iteration else None
        reverse_sha256 = _tensor_sha256(actual_reverse) if hash_this_iteration else None
        expected_reverse_sha256 = _tensor_sha256(expected_reverse) if hash_this_iteration else None
        producer_exact_qkv = [value == 0 for value in producer_mismatches]
        forward_exact_qkv = [value == 0 for value in forward_mismatches]
        iteration_passed = (
            not any(expected_local_poison)
            and not any(expected_forward_poison)
            and all(producer_exact_qkv)
            and not any(producer_poison_remaining)
            and all(forward_exact_qkv)
            and not any(forward_poison_remaining)
            and o_producer_mismatches == 0
            and o_producer_poison_remaining == 0
            and reverse_mismatches == 0
            and reverse_poison_remaining == 0
            and landing_ptrs_reused
            and forward_output_ptrs_reused
            and reverse_output_ptr_reused
            and all(tensor.dtype == torch.float8_e4m3fn for tensor in actual_forward)
            and actual_reverse.dtype == torch.bfloat16
        )
        iteration_reports.append(
            {
                "iteration": iteration,
                "phase": "cold" if iteration == 0 else "warm",
                "seed": args.seed + rank + iteration * world_size,
                "producer_byte_exact_qkv": producer_exact_qkv,
                "producer_byte_mismatches_qkv": producer_mismatches,
                "producer_poison_remaining_qkv": producer_poison_remaining,
                "producer_max_abs_error_qkv": [value for value, _ in producer_errors],
                "producer_nonfinite_errors_qkv": [value for _, value in producer_errors],
                "producer_sha256_qkv": producer_sha256_qkv,
                "expected_local_sha256_qkv": expected_local_sha256_qkv,
                "forward_byte_exact_qkv": forward_exact_qkv,
                "forward_byte_mismatches_qkv": forward_mismatches,
                "forward_poison_remaining_qkv": forward_poison_remaining,
                "forward_sha256_qkv": forward_sha256_qkv,
                "expected_forward_sha256_qkv": expected_forward_sha256_qkv,
                "expected_local_poison_qkv": expected_local_poison,
                "expected_forward_poison_qkv": expected_forward_poison,
                "o_producer_bf16_exact": o_producer_mismatches == 0,
                "o_producer_bf16_mismatches": o_producer_mismatches,
                "o_producer_poison_remaining": o_producer_poison_remaining,
                "o_producer_sha256": o_producer_sha256,
                "expected_o_sha256": expected_o_sha256,
                "reverse_bf16_exact": reverse_mismatches == 0,
                "reverse_bf16_mismatches": reverse_mismatches,
                "reverse_poison_remaining": reverse_poison_remaining,
                "reverse_max_abs_error": reverse_max_abs_error,
                "reverse_nonfinite_errors": reverse_nonfinite_errors,
                "reverse_sha256": reverse_sha256,
                "expected_reverse_sha256": expected_reverse_sha256,
                "source_absmax_qkv": [float(source.float().abs().max().item()) for source in (q_ref, k_ref, v_3d)],
                "landing_ptrs_reused": landing_ptrs_reused,
                "forward_output_ptrs_reused": forward_output_ptrs_reused,
                "reverse_output_ptr_reused": reverse_output_ptr_reused,
                "registered_slots": len(state.outputs),
                "passed": iteration_passed,
            }
        )

    states = list(adapter._STATES.values())
    state = states[0] if len(states) == 1 else None
    communicator = state.communicator if state is not None else None
    all_iterations_passed = all(report["passed"] for report in iteration_reports)
    local: dict[str, Any] = {
        "rank": rank,
        "transport": getattr(communicator, "transport", None),
        "state_count": len(states),
        "forward_dtype": "torch.float8_e4m3fn",
        "reverse_dtype": "torch.bfloat16",
        "producer_byte_exact": all(all(report["producer_byte_exact_qkv"]) for report in iteration_reports),
        "forward_byte_exact": all(all(report["forward_byte_exact_qkv"]) for report in iteration_reports),
        "reverse_bf16_exact": all(report["reverse_bf16_exact"] for report in iteration_reports),
        "registered_slots": len(state.outputs) if state is not None else -1,
        "all_iterations_passed": all_iterations_passed,
        "iteration_reports": iteration_reports,
    }
    reports: list[dict[str, Any] | None] = [None] * world_size
    dist.all_gather_object(reports, local)
    failures = [
        report
        for report in reports
        if report is None
        or report["transport"] != "rdma"
        or report["state_count"] != 1
        or report["forward_dtype"] != "torch.float8_e4m3fn"
        or report["reverse_dtype"] != "torch.bfloat16"
        or not report["all_iterations_passed"]
        or report["registered_slots"] != 4
    ]
    if rank == 0:
        print(
            "FLASHINFER_FP8_QKV_VALIDATION="
            + json.dumps(
                {
                    "world_size": world_size,
                    "local_shape": [
                        1,
                        args.local_seq_len,
                        args.heads,
                        args.head_dim,
                    ],
                    "qkv_wire_dtype": "float8_e4m3fn",
                    "reverse_o_dtype": "bfloat16",
                    "dequant_scale": args.dequant_scale,
                    "iterations": args.iterations,
                    "diagnostic_sync": {
                        "after_producer": args.sync_after_producer,
                        "after_each_collective": args.sync_after_each_collective,
                    },
                    "hash_iterations": args.hash_iterations,
                    "status": "pass" if not failures else "fail",
                    "rank_reports": reports,
                },
                sort_keys=True,
            ),
            flush=True,
        )

    adapter.clear_flashinfer_ulysses_communicators()
    dist.barrier(device_ids=[local_rank])
    dist.destroy_process_group()
    if failures:
        raise SystemExit(f"mixed-dtype Ulysses validation failed: {failures}")


if __name__ == "__main__":
    main()
