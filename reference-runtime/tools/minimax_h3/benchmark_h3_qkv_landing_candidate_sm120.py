#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Interleaved correctness/performance probe for the isolated QKV candidate.

The incumbent boundary is one BF16 ``F.linear`` followed by the production
H3 Q/K RMSNorm+RoPE kernels writing to caller-owned buffers and one V copy.
The candidate writes Q/K/V to those buffers from its GEMM epilogue.  No model
dispatch imports this file or the candidate.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn.functional as F

from tools.minimax_h3.h3_qkv_landing_sm120_candidate import build_candidate
from vllm_omni.diffusion.layers.fused_qk_norm_rope import fused_qk_norm_rope_out


@dataclass(frozen=True)
class Case:
    m: int
    k: int
    heads: int
    head_dim: int = 128
    rotary_dim: int = 96
    eps: float = 1e-5

    @property
    def width(self) -> int:
        return self.heads * self.head_dim


def _timed(call: Callable[[], Any]) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    call()
    end.record()
    end.synchronize()
    return float(start.elapsed_time(end))


def _telemetry(device_index: int) -> dict[str, float | None]:
    command = [
        "nvidia-smi",
        f"--id={device_index}",
        "--query-gpu=clocks.current.sm,power.draw,temperature.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        clock, power, temperature = (
            float(part.strip()) for part in result.stdout.strip().split(",")
        )
        return {
            "sm_clock_mhz": clock,
            "power_w": power,
            "temperature_c": temperature,
        }
    except (OSError, ValueError, subprocess.SubprocessError):
        return {"sm_clock_mhz": None, "power_w": None, "temperature_c": None}


def _summary(values: list[float]) -> dict[str, float]:
    return {
        "median_ms": statistics.median(values),
        "min_ms": min(values),
        "max_ms": max(values),
        "mean_ms": statistics.fmean(values),
    }


@torch.inference_mode()
def run(case: Case, device_index: int, warmup: int, iterations: int) -> dict[str, Any]:
    torch.cuda.set_device(device_index)
    device = torch.device("cuda", device_index)
    props = torch.cuda.get_device_properties(device)
    if (props.major, props.minor) != (12, 0):
        raise RuntimeError(f"SM120 required, got SM{props.major}{props.minor}")

    torch.manual_seed(1101)
    x = torch.randn((case.m, case.k), device=device, dtype=torch.bfloat16)
    weight = torch.randn(
        (3 * case.width, case.k), device=device, dtype=torch.bfloat16
    )
    q_weight = torch.randn(
        (case.head_dim,), device=device, dtype=torch.bfloat16
    )
    k_weight = torch.randn(
        (case.head_dim,), device=device, dtype=torch.bfloat16
    )
    freqs = torch.randn(
        (case.m, case.rotary_dim // 2), device=device, dtype=torch.float32
    )
    rope = torch.cat((torch.cos(freqs), torch.sin(freqs)), dim=-1).to(
        torch.bfloat16
    )

    incumbent_outputs = tuple(
        torch.empty((case.m, case.width), device=device, dtype=torch.bfloat16)
        for _ in range(3)
    )
    candidate = build_candidate(
        weight=weight,
        q_weight=q_weight,
        k_weight=k_weight,
        rope_table=rope,
        heads=case.heads,
        head_dim=case.head_dim,
        rotary_dim=case.rotary_dim,
        eps=case.eps,
    )

    def incumbent() -> tuple[torch.Tensor, ...]:
        packed = F.linear(x, weight)
        q, k, v = packed.split(case.width, dim=-1)
        q = q.view(case.m, case.heads, case.head_dim)
        k = k.view(case.m, case.heads, case.head_dim)
        fused_qk_norm_rope_out(
            q,
            k,
            q_weight,
            k_weight,
            rope,
            case.eps,
            q_out=incumbent_outputs[0].view(
                case.m, case.heads, case.head_dim
            ),
            k_out=incumbent_outputs[1].view(
                case.m, case.heads, case.head_dim
            ),
        )
        incumbent_outputs[2].copy_(v)
        return incumbent_outputs

    def fused_candidate() -> tuple[torch.Tensor, ...]:
        return candidate(x)

    for index in range(warmup):
        if index % 2:
            fused_candidate()
            incumbent()
        else:
            incumbent()
            fused_candidate()
    torch.cuda.synchronize(device)

    expected = tuple(t.clone() for t in incumbent())
    actual = fused_candidate()
    torch.cuda.synchronize(device)
    correctness = []
    for name, expected_tensor, actual_tensor in zip(
        ("q", "k", "v"), expected, actual, strict=True
    ):
        mismatch = expected_tensor != actual_tensor
        correctness.append(
            {
                "name": name,
                "bit_exact": bool(torch.equal(expected_tensor, actual_tensor)),
                "mismatch_elements": int(mismatch.sum()),
                "total_elements": expected_tensor.numel(),
                "max_abs": float(
                    (expected_tensor.float() - actual_tensor.float()).abs().max()
                ),
            }
        )

    incumbent_ms: list[float] = []
    candidate_ms: list[float] = []
    samples: list[dict[str, Any]] = []
    for index in range(iterations):
        if index % 2:
            candidate_value = _timed(fused_candidate)
            incumbent_value = _timed(incumbent)
            order = "candidate,incumbent"
        else:
            incumbent_value = _timed(incumbent)
            candidate_value = _timed(fused_candidate)
            order = "incumbent,candidate"
        incumbent_ms.append(incumbent_value)
        candidate_ms.append(candidate_value)
        samples.append(
            {
                "iteration": index,
                "order": order,
                "incumbent_ms": incumbent_value,
                "candidate_ms": candidate_value,
                **_telemetry(device_index),
            }
        )

    incumbent_summary = _summary(incumbent_ms)
    candidate_summary = _summary(candidate_ms)
    delta = incumbent_summary["median_ms"] - candidate_summary["median_ms"]
    return {
        "status": "isolated-candidate-only",
        "production_dispatch_modified": False,
        "device": {
            "index": device_index,
            "name": props.name,
            "compute_capability": [props.major, props.minor],
        },
        "case": asdict(case),
        "warmup": warmup,
        "iterations": iterations,
        "correctness": correctness,
        "all_outputs_bit_exact": all(item["bit_exact"] for item in correctness),
        "incumbent": incumbent_summary,
        "candidate": candidate_summary,
        "candidate_win_ms_per_layer": delta,
        "passes_required_0p2ms_gate": delta >= 0.2,
        "samples": samples,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, default=11_936)
    parser.add_argument("--k", type=int, default=5_376)
    parser.add_argument("--heads", type=int, default=56)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--json-out", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run(
        Case(m=args.m, k=args.k, heads=args.heads),
        args.device,
        args.warmup,
        args.iterations,
    )
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.json_out is not None:
        args.json_out.write_text(rendered + "\n")


if __name__ == "__main__":
    main()
