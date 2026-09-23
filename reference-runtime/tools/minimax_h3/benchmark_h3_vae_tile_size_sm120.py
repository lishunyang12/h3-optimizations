#!/usr/bin/env python3
"""Benchmark MiniMax-H3 video-VAE decoder tile sizes on one SM120 GPU.

This intentionally exercises the production ``MiniMaxH3VideoVAE.decode_latent``
adapter with the real checkpoint.  It is a rank-local, single temporal-window
microbenchmark; it does not model PP8 collectives or a complete-video decode.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = REPO_ROOT.parents[1]
DEFAULT_COMPONENT = WORKSPACE_ROOT / "MiniMax-H3" / "FL2VA" / "video_vae"
DEFAULT_PROFILE_ROOT = WORKSPACE_ROOT / "results" / "profile"
TILE_SIZES = (256, 384, 224)
LATENT_SHAPE = (1, 24, 7, 44, 80)  # one temporal window, 704 x 1280 canvas
CANVAS = (704, 1280)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--component", type=Path, default=DEFAULT_COMPONENT)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--physical-gpu", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1101)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--screen-samples", type=int, default=3)
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument(
        "--critical-rank-only",
        action="store_true",
        help=(
            "Benchmark artificial stacked decoder batches (not the current "
            "stack_tiling=False PP8 code path)."
        ),
    )
    parser.add_argument(
        "--actual-b1-only",
        action="store_true",
        help="Benchmark the actual B1 direct decoder calls used by spatial PP8.",
    )
    parser.add_argument(
        "--expand-threshold",
        type=float,
        default=0.03,
        help="Expand to --samples only if tile384 beats tile256 screen p50 by this fraction.",
    )
    args = parser.parse_args()
    if args.warmups < 1 or args.screen_samples < 2:
        parser.error("--warmups must be >=1 and --screen-samples must be >=2")
    if args.samples < args.screen_samples:
        parser.error("--samples must be >= --screen-samples")
    if args.critical_rank_only and args.actual_b1_only:
        parser.error("--critical-rank-only and --actual-b1-only are mutually exclusive")
    return args


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def nvidia_compute_processes(physical_gpu: int) -> list[dict[str, Any]]:
    command = [
        "nvidia-smi",
        "-i",
        str(physical_gpu),
        "--query-compute-apps=pid,process_name,used_gpu_memory",
        "--format=csv,noheader,nounits",
    ]
    completed = subprocess.run(command, check=True, text=True, capture_output=True)
    processes = []
    for raw in completed.stdout.splitlines():
        raw = raw.strip()
        if not raw:
            continue
        fields = [field.strip() for field in raw.split(",", 2)]
        if len(fields) != 3:
            raise RuntimeError(f"unparseable nvidia-smi process row: {raw!r}")
        processes.append(
            {"pid": int(fields[0]), "process_name": fields[1], "used_gpu_memory_mib": fields[2]}
        )
    return processes


def assert_no_foreign_process(physical_gpu: int, *, allow_self: bool) -> list[dict[str, Any]]:
    processes = nvidia_compute_processes(physical_gpu)
    allowed = {os.getpid()} if allow_self else set()
    foreign = [item for item in processes if item["pid"] not in allowed]
    if foreign:
        raise RuntimeError(f"foreign compute process appeared on physical GPU{physical_gpu}: {foreign}")
    return processes


def percentile(values: list[float], q: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def tensor_error(actual: torch.Tensor, reference: torch.Tensor) -> dict[str, Any]:
    if actual.shape != reference.shape:
        return {
            "shape_match": False,
            "actual_shape": list(actual.shape),
            "reference_shape": list(reference.shape),
        }
    a = actual.float()
    r = reference.float()
    diff = a - r
    mse = float(torch.mean(diff.double().square()).item())
    reference_l2 = float(torch.linalg.vector_norm(r.double()).item())
    diff_l2 = float(torch.linalg.vector_norm(diff.double()).item())
    peak = 2.0  # decoded H3 frames use the nominal [-1, 1] signal range
    return {
        "shape_match": True,
        "bitwise_equal": bool(torch.equal(actual, reference)),
        "max_abs": float(diff.abs().max().item()),
        "mean_abs": float(diff.abs().mean().item()),
        "rel_l2": diff_l2 / reference_l2 if reference_l2 else math.inf,
        "mse": mse,
        # JSON has no representation for infinity. ``None`` denotes exact
        # equality (infinite PSNR) and keeps results.json standards-compliant.
        "psnr_db_data_range_2": None if mse == 0 else 10.0 * math.log10((peak * peak) / mse),
    }


def balanced_orders(round_count: int) -> list[tuple[int, int, int]]:
    orders = (
        (256, 384, 224),
        (384, 224, 256),
        (224, 256, 384),
        (224, 384, 256),
        (384, 256, 224),
        (256, 224, 384),
    )
    return [orders[index % len(orders)] for index in range(round_count)]


def git_revision() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, check=True, text=True, capture_output=True
    )
    return completed.stdout.strip()


def write_report(output_dir: Path, payload: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "results.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )

    lines = [
        "# MiniMax-H3 SM120 VAE decoder tile-size microbenchmark",
        "",
        f"Status: **{payload['status']}**",
        "",
        "This is a **single-GPU, rank-local, one-window microbenchmark**. It measures one "
        "`[1,24,7,44,80]` latent window (704×1280 canvas) and does not include temporal PP8 "
        "collectives, full 15-second temporal assembly, transport, or MP4 encoding.",
        "",
    ]
    if payload.get("error"):
        lines.extend(["## Failure", "", f"`{payload['error']}`", ""])
    if payload.get("gpu"):
        lines.extend(
            [
                "## Runtime",
                "",
                f"- GPU: {payload['gpu']['name']} (SM {payload['gpu']['compute_capability']})",
                f"- Model load: {payload.get('model_load_seconds', math.nan):.3f} s",
                f"- vLLM H3 VAE exact custom ops installed: {payload.get('vae_optimized_ops_installed')}",
                f"- Checkpoint: `{payload['component_path']}`",
                "",
            ]
        )
    if payload.get("arms"):
        lines.extend(
            [
                "## Timing and memory",
                "",
                "| Tile | Grid | Samples | p10 (ms) | p50 (ms) | p90 (ms) | Peak alloc (GiB) | Peak reserved (GiB) |",
                "|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for tile in TILE_SIZES:
            arm = payload["arms"].get(str(tile))
            if not arm:
                continue
            lines.append(
                f"| {tile} | {arm['tile_rows']}×{arm['tile_cols']}={arm['tile_count']} | "
                f"{arm['sample_count']} | {arm['p10_ms']:.3f} | {arm['p50_ms']:.3f} | "
                f"{arm['p90_ms']:.3f} | {arm['clean_peak_allocated_gib']:.3f} | "
                f"{arm['clean_peak_reserved_gib']:.3f} |"
            )
        lines.extend(
            [
                "",
                "CUDA events bracket only `decode_latent`; CPU copies, metric calculation, "
                "foreign-process checks, and clean-memory probes are outside timing.",
                "",
                "## Numerical comparison against tile 256",
                "",
                "| Tile | Bitwise | max abs | mean abs | rel L2 | PSNR (dB, range=2) |",
                "|---:|:---:|---:|---:|---:|---:|",
            ]
        )
        for tile in TILE_SIZES:
            comparison = payload.get("comparisons_vs_256", {}).get(str(tile))
            if not comparison or not comparison.get("shape_match"):
                continue
            psnr = comparison["psnr_db_data_range_2"]
            psnr_text = "inf" if psnr is None else f"{psnr:.3f}"
            lines.append(
                f"| {tile} | {comparison['bitwise_equal']} | {comparison['max_abs']:.8g} | "
                f"{comparison['mean_abs']:.8g} | {comparison['rel_l2']:.8g} | {psnr_text} |"
            )
        lines.extend(["", "## Repeat determinism", ""])
        for tile in TILE_SIZES:
            repeat = payload.get("repeat_determinism", {}).get(str(tile))
            if repeat:
                lines.append(
                    f"- Tile {tile}: bitwise={repeat.get('bitwise_equal')}, "
                    f"max_abs={repeat.get('max_abs')}, mean_abs={repeat.get('mean_abs')}"
                )
        lines.extend(
            [
                "",
                "## Screening decision",
                "",
                payload.get("screening_decision", "not available"),
                "",
            ]
        )
    if payload.get("critical_rank_arms"):
        lines.extend(
            [
                "## Artificial stacked-batch upper bound",
                "",
                "This table batches a rank's tiles into one `remote.model.decode` call. The "
                "current checkpoint has `stack_tiling=False`, so this is **not the actual PP8 "
                "code path**; it is only an artificial stacked-throughput upper bound. It excludes "
                "spatial stitch/blend, synchronization, all-gather, and all other ranks.",
                "",
                "| Tile | Decoder input | Samples | p10 (ms) | p50 (ms) | p90 (ms) | Peak alloc (GiB) | Peak reserved (GiB) |",
                "|---:|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for tile in TILE_SIZES:
            arm = payload["critical_rank_arms"][str(tile)]
            lines.append(
                f"| {tile} | `{arm['input_shape']}` | {arm['sample_count']} | "
                f"{arm['p10_ms']:.3f} | {arm['p50_ms']:.3f} | {arm['p90_ms']:.3f} | "
                f"{arm['clean_peak_allocated_gib']:.3f} | {arm['clean_peak_reserved_gib']:.3f} |"
            )
        lines.extend(
            [
                "",
                "Inputs are fixed FP32 unnormalized VAE latents and decode runs under CUDA FP16 "
                "autocast. CUDA events exclude Python dispatch, process checks, and clean-memory probes.",
                "",
            ]
        )
    if payload.get("actual_b1_arms"):
        lines.extend(
            [
                "## Actual spatial-PP B1 decoder calls",
                "",
                "The checkpoint uses `stack_tiling=False`; each rank invokes `model.decode` "
                "serially for each assigned tile. The projected column multiplies B1 p50 by "
                "the slowest rank's 4/1/4 calls. It still excludes stitch/blend, barrier, and all-gather.",
                "",
                "| Tile | B1 input | Samples | p10 (ms) | p50 (ms) | p90 (ms) | Critical calls | Projected serial decode (ms) | Peak alloc (GiB) |",
                "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for tile in TILE_SIZES:
            arm = payload["actual_b1_arms"][str(tile)]
            lines.append(
                f"| {tile} | `{arm['input_shape']}` | {arm['sample_count']} | "
                f"{arm['p10_ms']:.3f} | {arm['p50_ms']:.3f} | {arm['p90_ms']:.3f} | "
                f"{arm['critical_rank_serial_calls']} | {arm['projected_critical_serial_ms']:.3f} | "
                f"{arm['clean_peak_allocated_gib']:.3f} |"
            )
        lines.append("")
    (output_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir or (DEFAULT_PROFILE_ROOT / f"minimax-h3-vae-tile-sm120-{utc_stamp()}")
    output_dir = output_dir.resolve()
    payload: dict[str, Any] = {
        "status": "initializing",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "repo_root": str(REPO_ROOT),
        "git_revision": git_revision(),
        "component_path": str(args.component.resolve()),
        "physical_gpu": args.physical_gpu,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "seed": args.seed,
        "latent_shape": list(LATENT_SHAPE),
        "latent_dtype": "torch.float32",
        "canvas_height": CANVAS[0],
        "canvas_width": CANVAS[1],
        "autocast_dtype": "torch.float16",
        "tile_sizes": list(TILE_SIZES),
        "requested_warmups_per_arm": args.warmups,
        "screen_samples_per_arm": args.screen_samples,
        "requested_samples_per_arm": args.samples,
        "expand_threshold_fraction": args.expand_threshold,
        "limitation": "single GPU/rank-local, one 7-token temporal window; no PP8 or full-video E2E",
    }
    write_report(output_dir, payload)

    expected_visible = str(args.physical_gpu)
    if os.environ.get("CUDA_VISIBLE_DEVICES") != expected_visible:
        payload["status"] = "failed"
        payload["error"] = (
            f"CUDA_VISIBLE_DEVICES must be exactly {expected_visible!r}, got "
            f"{os.environ.get('CUDA_VISIBLE_DEVICES')!r}"
        )
        write_report(output_dir, payload)
        return 2

    try:
        payload["preflight_processes"] = assert_no_foreign_process(
            args.physical_gpu, allow_self=False
        )
        if not args.component.is_dir() or not (args.component / "source" / "model.safetensors").is_file():
            raise FileNotFoundError(f"real VAE component not found: {args.component}")

        # Production uses the streaming temporal assembly path. For a single
        # 7-token window this still executes exactly one spatial tiled decode.
        os.environ.setdefault("MINIMAX_H3_VAE_DECODER_STREAM_TEMPORAL_CAT", "1")
        sys.path.insert(0, str(REPO_ROOT))
        from vllm_omni.diffusion.models.minimax_h3.vae import MiniMaxH3VideoVAE

        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError(
                f"expected exactly one visible CUDA device, available={torch.cuda.is_available()} "
                f"count={torch.cuda.device_count()}"
            )
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
        properties = torch.cuda.get_device_properties(device)
        payload["torch_version"] = torch.__version__
        payload["cuda_version"] = torch.version.cuda
        payload["gpu"] = {
            "logical_index": 0,
            "name": properties.name,
            "compute_capability": f"{properties.major}.{properties.minor}",
            "total_memory_gib": properties.total_memory / (1024**3),
        }

        generator = torch.Generator(device="cpu").manual_seed(args.seed)
        latent_cpu = torch.randn(LATENT_SHAPE, generator=generator, dtype=torch.float32)
        latent = latent_cpu.to(device=device, non_blocking=False)
        payload["latent_stats"] = {
            "min": float(latent_cpu.min().item()),
            "max": float(latent_cpu.max().item()),
            "mean": float(latent_cpu.mean().item()),
            "std": float(latent_cpu.std().item()),
        }
        del latent_cpu

        load_started = time.perf_counter()
        vae = MiniMaxH3VideoVAE(str(args.component), device=device)
        vae.eval()
        torch.cuda.synchronize(device)
        payload["model_load_seconds"] = time.perf_counter() - load_started
        payload["vae_optimized_ops_installed"] = bool(
            getattr(vae.model.decoder, "_omni_h3_vae_optimizations_installed", False)
        )
        payload["checkpoint_decoder_overlap"] = int(vae.remote.model.decoder_tile_overlap_min)
        payload["checkpoint_decoder_tile_initial"] = int(vae.remote.model.decoder_tile_size)
        if payload["checkpoint_decoder_overlap"] != 64:
            raise RuntimeError(
                f"expected checkpoint decoder overlap 64, got {payload['checkpoint_decoder_overlap']}"
            )
        payload["post_load_processes"] = assert_no_foreign_process(
            args.physical_gpu, allow_self=True
        )

        if args.critical_rank_only or args.actual_b1_only:
            # ``stack_tiling`` is false in the released checkpoint. The B4
            # shapes remain useful only as an artificial throughput bound;
            # B1 is the actual direct decoder-call shape on every PP8 rank.
            critical_shapes = (
                {
                    256: (1, 24, 7, 16, 16),
                    384: (1, 24, 7, 24, 24),
                    224: (1, 24, 7, 14, 14),
                }
                if args.actual_b1_only
                else {
                    256: (4, 24, 7, 16, 16),
                    384: (1, 24, 7, 24, 24),
                    224: (4, 24, 7, 14, 14),
                }
            )
            mean = torch.tensor(
                vae.config_dict["latents_mean"], dtype=torch.float32
            ).view(1, 24, 1, 1, 1)
            std = torch.tensor(
                vae.config_dict["latents_std"], dtype=torch.float32
            ).view(1, 24, 1, 1, 1)
            decoder_inputs: dict[int, torch.Tensor] = {}
            for offset, tile in enumerate(TILE_SIZES):
                generator = torch.Generator(device="cpu").manual_seed(args.seed + offset)
                normalized = torch.randn(
                    critical_shapes[tile], generator=generator, dtype=torch.float32
                )
                decoder_inputs[tile] = (normalized * std + mean).to(device)

            critical_samples: dict[int, list[float]] = {tile: [] for tile in TILE_SIZES}
            critical_sample_peaks: dict[int, list[dict[str, int]]] = {
                tile: [] for tile in TILE_SIZES
            }
            critical_output_shapes: dict[int, list[int]] = {}
            critical_output_dtypes: dict[int, str] = {}

            def direct_decode(tile: int) -> tuple[float, dict[str, int]]:
                assert_no_foreign_process(args.physical_gpu, allow_self=True)
                torch.cuda.reset_peak_memory_stats(device)
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                with torch.inference_mode(), torch.autocast(
                    device_type="cuda", dtype=torch.float16, enabled=True
                ):
                    decoded = vae.remote.model.decode(decoder_inputs[tile])
                end.record()
                end.synchronize()
                elapsed_ms = float(start.elapsed_time(end))
                peaks = {
                    "max_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
                    "max_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
                }
                critical_output_shapes[tile] = list(decoded.shape)
                critical_output_dtypes[tile] = str(decoded.dtype)
                del decoded, start, end
                return elapsed_ms, peaks

            # Warm every dynamic shape before balanced, alternating samples.
            for warmup_round in range(args.warmups):
                for tile in balanced_orders(args.warmups)[warmup_round]:
                    direct_decode(tile)
            for order in balanced_orders(args.samples):
                for tile in order:
                    elapsed_ms, peaks = direct_decode(tile)
                    critical_samples[tile].append(elapsed_ms)
                    critical_sample_peaks[tile].append(peaks)

            critical_clean_peaks: dict[int, dict[str, int]] = {}
            for tile in TILE_SIZES:
                gc.collect()
                torch.cuda.empty_cache()
                _, peaks = direct_decode(tile)
                critical_clean_peaks[tile] = peaks

            critical_arms: dict[str, Any] = {}
            for tile in TILE_SIZES:
                values = critical_samples[tile]
                clean = critical_clean_peaks[tile]
                critical_arms[str(tile)] = {
                    "tile_size": tile,
                    "input_shape": list(critical_shapes[tile]),
                    "input_dtype": str(decoder_inputs[tile].dtype),
                    "output_shape": critical_output_shapes[tile],
                    "output_dtype": critical_output_dtypes[tile],
                    "sample_count": len(values),
                    "samples_ms": values,
                    "p10_ms": percentile(values, 0.10),
                    "p50_ms": percentile(values, 0.50),
                    "p90_ms": percentile(values, 0.90),
                    "mean_ms": statistics.fmean(values),
                    "sample_peak_allocated_bytes_max": max(
                        item["max_allocated_bytes"] for item in critical_sample_peaks[tile]
                    ),
                    "sample_peak_reserved_bytes_max": max(
                        item["max_reserved_bytes"] for item in critical_sample_peaks[tile]
                    ),
                    "clean_peak_allocated_bytes": clean["max_allocated_bytes"],
                    "clean_peak_reserved_bytes": clean["max_reserved_bytes"],
                    "clean_peak_allocated_gib": clean["max_allocated_bytes"] / (1024**3),
                    "clean_peak_reserved_gib": clean["max_reserved_bytes"] / (1024**3),
                }
            payload.update(
                {
                    "status": "complete",
                    "benchmark_mode": (
                        "actual_spatial_pp_b1_direct_decoder_calls"
                        if args.actual_b1_only
                        else "artificial_stacked_decoder_upper_bound"
                    ),
                    "critical_rank_excludes": [
                        "spatial stitch/blend",
                        "barrier",
                        "all-gather",
                        "other PP8 ranks",
                        "temporal assembly",
                    ],
                    "final_processes": assert_no_foreign_process(
                        args.physical_gpu, allow_self=True
                    ),
                }
            )
            if args.actual_b1_only:
                serial_calls = {256: 4, 384: 1, 224: 4}
                for tile in TILE_SIZES:
                    arm = critical_arms[str(tile)]
                    arm["critical_rank_serial_calls"] = serial_calls[tile]
                    arm["projected_critical_serial_ms"] = (
                        arm["p50_ms"] * serial_calls[tile]
                    )
                payload["actual_b1_arms"] = critical_arms
                payload["stack_tiling"] = False
            else:
                payload["critical_rank_arms"] = critical_arms
                payload["warning"] = (
                    "Artificial stacked upper bound only; current PP8 executes serial B1 calls."
                )
            write_report(output_dir, payload)
            print(output_dir)
            for tile in TILE_SIZES:
                arm = critical_arms[str(tile)]
                print(
                    f"direct tile={tile} input={arm['input_shape']} n={arm['sample_count']} "
                    f"p50={arm['p50_ms']:.3f}ms peak={arm['clean_peak_allocated_gib']:.3f}GiB"
                )
            return 0

        samples: dict[int, list[float]] = {tile: [] for tile in TILE_SIZES}
        sample_peaks: dict[int, list[dict[str, int]]] = {tile: [] for tile in TILE_SIZES}
        outputs: dict[int, list[torch.Tensor]] = {tile: [] for tile in TILE_SIZES}
        shapes: dict[int, list[int]] = {}
        dtypes: dict[int, str] = {}
        grid: dict[int, tuple[int, int]] = {}

        def set_tile(tile: int) -> None:
            vae.remote.model.decoder_tile_size = int(tile)
            if int(vae.remote.model.decoder_tile_overlap_min) != 64:
                raise RuntimeError("decoder overlap changed unexpectedly")
            rows = vae.remote.model.split_tiles(CANVAS[0], True)[0]
            cols = vae.remote.model.split_tiles(CANVAS[1], True)[0]
            grid[tile] = (len(rows), len(cols))
            actual_count = vae._decoder_tile_count(latent)
            if actual_count != len(rows) * len(cols):
                raise RuntimeError(
                    f"tile-count mismatch for {tile}: adapter={actual_count}, grid={len(rows) * len(cols)}"
                )

        def decode_once(tile: int, *, retain: bool) -> tuple[float, torch.Tensor | None, dict[str, int]]:
            assert_no_foreign_process(args.physical_gpu, allow_self=True)
            set_tile(tile)
            torch.cuda.reset_peak_memory_stats(device)
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            with torch.inference_mode(), torch.autocast(
                device_type="cuda", dtype=torch.float16, enabled=True
            ):
                output = vae.decode_latent(latent)
            end.record()
            end.synchronize()
            elapsed_ms = float(start.elapsed_time(end))
            peaks = {
                "max_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
                "max_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
            }
            shapes[tile] = list(output.shape)
            dtypes[tile] = str(output.dtype)
            retained = output.detach().cpu().contiguous() if retain else None
            del output, start, end
            return elapsed_ms, retained, peaks

        # One explicit warmup per arm (or more if requested), excluded from timing.
        for warmup_round in range(args.warmups):
            for tile in balanced_orders(args.warmups)[warmup_round]:
                _, _, _ = decode_once(tile, retain=False)
        torch.cuda.synchronize(device)

        def run_rounds(start_round: int, end_round: int) -> None:
            orders = balanced_orders(end_round)
            for round_index in range(start_round, end_round):
                for tile in orders[round_index]:
                    retain = len(outputs[tile]) < 2
                    elapsed_ms, retained, peaks = decode_once(tile, retain=retain)
                    samples[tile].append(elapsed_ms)
                    sample_peaks[tile].append(peaks)
                    if retained is not None:
                        outputs[tile].append(retained)

        run_rounds(0, args.screen_samples)
        screen_p50 = {tile: statistics.median(samples[tile]) for tile in TILE_SIZES}
        screen_gain = 1.0 - screen_p50[384] / screen_p50[256]
        expanded = args.samples > args.screen_samples and screen_gain >= args.expand_threshold
        if expanded:
            run_rounds(args.screen_samples, args.samples)
            screening_decision = (
                f"Expanded from {args.screen_samples} to {args.samples} samples/arm: "
                f"tile384 screen p50 was {screen_gain * 100:.2f}% faster than tile256 "
                f"(threshold {args.expand_threshold * 100:.2f}%)."
            )
        else:
            screening_decision = (
                f"Stopped after {args.screen_samples} samples/arm: tile384 screen p50 was "
                f"{screen_gain * 100:.2f}% faster than tile256, below the "
                f"{args.expand_threshold * 100:.2f}% expansion threshold or no expansion was requested."
            )

        # Clean one-pass memory probes avoid inherited allocator reservations
        # from a previously measured tile arm. They are intentionally untimed.
        clean_peaks: dict[int, dict[str, int]] = {}
        for tile in TILE_SIZES:
            gc.collect()
            torch.cuda.empty_cache()
            _, _, peaks = decode_once(tile, retain=False)
            clean_peaks[tile] = peaks

        arms: dict[str, Any] = {}
        for tile in TILE_SIZES:
            rows, cols = grid[tile]
            values = samples[tile]
            arms[str(tile)] = {
                "tile_size": tile,
                "tile_overlap": 64,
                "tile_rows": rows,
                "tile_cols": cols,
                "tile_count": rows * cols,
                "sample_count": len(values),
                "samples_ms": values,
                "p10_ms": percentile(values, 0.10),
                "p50_ms": percentile(values, 0.50),
                "p90_ms": percentile(values, 0.90),
                "mean_ms": statistics.fmean(values),
                "output_shape": shapes[tile],
                "output_dtype": dtypes[tile],
                "sample_peak_allocated_bytes_max": max(
                    item["max_allocated_bytes"] for item in sample_peaks[tile]
                ),
                "sample_peak_reserved_bytes_max": max(
                    item["max_reserved_bytes"] for item in sample_peaks[tile]
                ),
                "clean_peak_allocated_bytes": clean_peaks[tile]["max_allocated_bytes"],
                "clean_peak_reserved_bytes": clean_peaks[tile]["max_reserved_bytes"],
                "clean_peak_allocated_gib": clean_peaks[tile]["max_allocated_bytes"] / (1024**3),
                "clean_peak_reserved_gib": clean_peaks[tile]["max_reserved_bytes"] / (1024**3),
            }

        reference = outputs[256][0]
        comparisons = {
            str(tile): tensor_error(outputs[tile][0], reference) for tile in TILE_SIZES
        }
        repeats = {
            str(tile): tensor_error(outputs[tile][1], outputs[tile][0]) for tile in TILE_SIZES
        }
        payload.update(
            {
                "status": "complete",
                "expanded_after_screen": expanded,
                "screen_p50_ms": {str(key): value for key, value in screen_p50.items()},
                "screen_tile384_gain_vs_256_fraction": screen_gain,
                "screening_decision": screening_decision,
                "arms": arms,
                "comparisons_vs_256": comparisons,
                "repeat_determinism": repeats,
                "final_processes": assert_no_foreign_process(args.physical_gpu, allow_self=True),
            }
        )
        write_report(output_dir, payload)
        print(output_dir)
        print(screening_decision)
        for tile in TILE_SIZES:
            arm = arms[str(tile)]
            print(
                f"tile={tile} grid={arm['tile_rows']}x{arm['tile_cols']} "
                f"n={arm['sample_count']} p50={arm['p50_ms']:.3f}ms "
                f"peak={arm['clean_peak_allocated_gib']:.3f}GiB"
            )
        return 0
    except (RuntimeError, FileNotFoundError, subprocess.SubprocessError) as exc:
        if isinstance(exc, torch.cuda.OutOfMemoryError):
            payload["status"] = "oom_stopped"
        else:
            payload["status"] = "failed"
        payload["error"] = f"{type(exc).__name__}: {exc}"
        write_report(output_dir, payload)
        print(f"benchmark stopped: {payload['error']}", file=sys.stderr)
        print(output_dir, file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
