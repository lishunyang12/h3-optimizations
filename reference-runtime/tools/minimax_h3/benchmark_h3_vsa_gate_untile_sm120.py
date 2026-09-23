#!/usr/bin/env python3
"""Qualify H3's exact fused VSA coarse-gate add and O landing on SM120."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from vllm_omni.diffusion.attention.backends.fastvideo_vsa import (  # noqa: E402
    _get_h3_tile_metadata,
)
from vllm_omni.diffusion.attention.ops.minimax_h3_vsa_layout import (  # noqa: E402
    build_h3_aligned_untile_source_rows,
    build_h3_tiled_source_rows,
    h3_vsa_gate_untile_out,
    h3_vsa_tile_pack,
    h3_vsa_tile_untile_out,
)

PREFIX_SEGMENTS = (121, 1206)
VIDEO_SHAPE = (107, 22, 40)
ALIGNED_ROWS = 95_488
HEADS = 7
HEAD_DIM = 128
FORMAL_E2E_SECONDS = 29.873
CALLS_PER_REQUEST = 4 * 50


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--samples", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1101)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.warmups < 1 or args.samples < 1:
        parser.error("warmups and samples must be positive")
    return args


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    low, high = math.floor(position), math.ceil(position)
    if low == high:
        return ordered[low]
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def summarize(values: list[float]) -> dict[str, float | int | list[float]]:
    return {
        "samples_ms": values,
        "sample_count": len(values),
        "p10_ms": percentile(values, 0.10),
        "p50_ms": percentile(values, 0.50),
        "p90_ms": percentile(values, 0.90),
        "mean_ms": statistics.fmean(values),
    }


def benchmark(fn: Callable[[], torch.Tensor], warmups: int, samples: int) -> dict[str, object]:
    for _ in range(warmups):
        fn()
    torch.accelerator.synchronize()
    baseline_allocated = torch.accelerator.memory_allocated()
    torch.accelerator.reset_peak_memory_stats()
    elapsed: list[float] = []
    for _ in range(samples):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        elapsed.append(float(start.elapsed_time(end)))
    result: dict[str, object] = summarize(elapsed)
    result["peak_extra_allocated_mib"] = (
        torch.accelerator.max_memory_allocated() - baseline_allocated
    ) / (1024**2)
    return result


def write_report(output_dir: Path, payload: dict[str, object]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "results.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    legacy = payload["legacy"]
    fused = payload["fused"]
    comparison = payload["comparison"]
    lines = [
        "# MiniMax-H3 VSA gate + untile SM120 qualification",
        "",
        f"Status: **{payload['status']}**",
        "",
        "Production per-rank geometry: `B1`, 95,487 valid / 95,488 aligned rows, "
        "105,024 tile-padded rows, 1,641 tile64 blocks, 7 heads × 128 BF16.",
        "",
        "| Arm | p10 (ms) | p50 (ms) | p90 (ms) | Peak extra alloc (MiB) |",
        "|---|---:|---:|---:|---:|",
        f"| Current: gate pack + mul/add + untile/O landing | {legacy['p10_ms']:.4f} | "
        f"{legacy['p50_ms']:.4f} | {legacy['p90_ms']:.4f} | "
        f"{legacy['peak_extra_allocated_mib']:.2f} |",
        f"| Fused: BF16 mul.rn/add.rn + untile/O landing | {fused['p10_ms']:.4f} | "
        f"{fused['p50_ms']:.4f} | {fused['p90_ms']:.4f} | "
        f"{fused['peak_extra_allocated_mib']:.2f} |",
        "",
        f"Bitwise equal: **{payload['bitwise_equal']}**. p50 speedup: "
        f"**{comparison['speedup_p50']:.3f}×**; projected 200-call request saving: "
        f"**{comparison['projected_request_saving_seconds']:.3f} s**.",
        "",
        "The projection is arithmetic only; a formal eight-GPU cumulative E2E run "
        "is required before promoting it into the baseline.",
        "",
    ]
    (output_dir / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda:0")
    capability = torch.cuda.get_device_capability(device)
    if capability != (12, 0):
        raise RuntimeError(f"exact qualification requires SM120, got SM{capability[0]}{capability[1]}")

    partition, sizes, non_pad, untile, _, _ = _get_h3_tile_metadata(
        PREFIX_SEGMENTS,
        VIDEO_SHAPE,
        device,
    )
    valid_rows = sum(PREFIX_SEGMENTS) + math.prod(VIDEO_SHAPE)
    tiled_rows = int(sizes.numel()) * 64
    if valid_rows != 95_487 or tiled_rows != 105_024 or ALIGNED_ROWS != 95_488:
        raise RuntimeError("production H3 geometry constants drifted")
    tiled_source_rows = build_h3_tiled_source_rows(partition, non_pad, tiled_rows)
    aligned_source_rows = build_h3_aligned_untile_source_rows(
        untile,
        aligned_rows=ALIGNED_ROWS,
        tiled_rows=tiled_rows,
    )

    generator = torch.Generator(device=device).manual_seed(args.seed)
    fine = torch.randn(
        1,
        tiled_rows,
        HEADS,
        HEAD_DIM,
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )
    compressed = torch.randn(
        1,
        HEADS,
        sizes.numel(),
        HEAD_DIM,
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    ).permute(0, 2, 1, 3)
    if compressed.is_contiguous():
        raise RuntimeError("benchmark must preserve the coarse matmul's noncontiguous permute")
    gate = torch.randn(
        1,
        valid_rows,
        HEADS,
        HEAD_DIM,
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )
    legacy_landing = torch.empty(
        1, ALIGNED_ROWS, HEADS, HEAD_DIM, dtype=torch.bfloat16, device=device
    )
    fused_landing = torch.empty_like(legacy_landing)

    def legacy() -> torch.Tensor:
        gate_tiled = h3_vsa_tile_pack(gate, tiled_source_rows)
        combined = (
            fine.view(1, sizes.numel(), 64, HEADS, HEAD_DIM)
            + compressed.unsqueeze(2)
            * gate_tiled.view(1, sizes.numel(), 64, HEADS, HEAD_DIM)
        ).view_as(fine)
        return h3_vsa_tile_untile_out(
            combined,
            aligned_source_rows,
            out=legacy_landing,
        )

    def fused() -> torch.Tensor:
        return h3_vsa_gate_untile_out(
            fine,
            compressed,
            gate,
            aligned_source_rows,
            out=fused_landing,
        )

    legacy_result = legacy()
    fused_result = fused()
    torch.accelerator.synchronize()
    bitwise_equal = bool(torch.equal(legacy_result, fused_result))

    legacy_stats = benchmark(legacy, args.warmups, args.samples)
    fused_stats = benchmark(fused, args.warmups, args.samples)
    speedup = float(legacy_stats["p50_ms"]) / float(fused_stats["p50_ms"])
    saving_seconds = (
        float(legacy_stats["p50_ms"]) - float(fused_stats["p50_ms"])
    ) * CALLS_PER_REQUEST / 1000
    payload: dict[str, object] = {
        "status": "pass" if bitwise_equal and speedup > 1 else "reject",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "device": torch.cuda.get_device_name(device),
        "capability": list(capability),
        "torch": torch.__version__,
        "geometry": {
            "prefix_segments": list(PREFIX_SEGMENTS),
            "video_shape": list(VIDEO_SHAPE),
            "valid_rows": valid_rows,
            "aligned_rows": ALIGNED_ROWS,
            "tiled_rows": tiled_rows,
            "tile_blocks": int(sizes.numel()),
            "heads": HEADS,
            "head_dim": HEAD_DIM,
            "dtype": "bfloat16",
        },
        "bitwise_equal": bitwise_equal,
        "legacy": legacy_stats,
        "fused": fused_stats,
        "comparison": {
            "speedup_p50": speedup,
            "latency_reduction_fraction": 1 - 1 / speedup,
            "calls_per_request": CALLS_PER_REQUEST,
            "projected_request_saving_seconds": saving_seconds,
            "projected_formal_e2e_seconds": FORMAL_E2E_SECONDS - saving_seconds,
        },
    }
    output_dir = args.output_dir
    if output_dir is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output_dir = (
            Path(__file__).resolve().parents[3]
            / "results"
            / "profile"
            / f"h3-vsa-gate-untile-sm120-{stamp}"
        )
    write_report(output_dir, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    print(f"artifact={output_dir}")
    return 0 if payload["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
