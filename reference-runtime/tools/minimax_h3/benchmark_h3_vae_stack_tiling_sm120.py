#!/usr/bin/env python3
"""Benchmark bounded MiniMax-H3 VAE tile batching on one SM120 GPU.

The benchmark uses the real FL2VA video VAE and calls the checkpoint's full
``_adaptive_decode`` for one canonical ``[1, 24, 7, 44, 80]`` temporal window.
It monkey-patches ``_run_tile_tasks`` only for the lifetime of this process;
no production behavior is changed.
"""

from __future__ import annotations

import argparse
import gc
import inspect
import json
import math
import os
import statistics
import subprocess
import sys
import time
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from types import MethodType
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = REPO_ROOT.parents[1]
DEFAULT_COMPONENT = WORKSPACE_ROOT / "MiniMax-H3" / "FL2VA" / "video_vae"
DEFAULT_PROFILE_ROOT = WORKSPACE_ROOT / "results" / "profile"
EXACT_OPS_ENV = "VLLM_OMNI_MINIMAX_H3_VAE_EXACT_OPS_SM120"
TILE_SIZE_ENV = "VLLM_OMNI_MINIMAX_H3_VAE_DECODER_TILE_SIZE"
FP32_NORM_ENV = "MINIMAX_H3_VAE_DECODER_VIT_FP32_NORM"
FF_COMPILE_ENV = "MINIMAX_H3_VAE_DECODER_VIT_FF_TORCH_COMPILE"
LATENT_SHAPE = (1, 24, 7, 44, 80)
EXPECTED_OUTPUT_SHAPE = (1, 3, 28, 704, 1280)
CAPS = (1, 2, 4, 7, 14, 28)
TILE_COUNT = 28


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--component", type=Path, default=DEFAULT_COMPONENT)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--physical-gpu", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1101)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--samples", type=int, default=7)
    parser.add_argument(
        "--backends",
        choices=("stock", "exact", "both"),
        default="both",
        help="Run stock first and optionally reload with the SM120 exact-op installer.",
    )
    args = parser.parse_args()
    if args.warmups < 1 or args.samples < 3:
        parser.error("--warmups must be >=1 and --samples must be >=3")
    if args.physical_gpu in (2, 3):
        parser.error("physical GPUs 2 and 3 are explicitly excluded")
    return args


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[low]
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def summarize(values: list[float]) -> dict[str, Any]:
    return {
        "samples_ms": values,
        "sample_count": len(values),
        "p10_ms": percentile(values, 0.10),
        "p50_ms": percentile(values, 0.50),
        "p90_ms": percentile(values, 0.90),
        "mean_ms": statistics.fmean(values),
    }


def batch_plan_text(values: list[int]) -> str:
    if values and all(value == values[0] for value in values):
        return f"{len(values)}xB{values[0]}"
    return "+".join(f"B{value}" for value in values)


def balanced_orders(round_count: int) -> list[tuple[int, ...]]:
    """Rotate and reverse the cap order so no arm always runs first or last."""

    orders = []
    base = CAPS
    for round_index in range(round_count):
        shift = round_index % len(base)
        order = base[shift:] + base[:shift]
        if (round_index // len(base)) % 2:
            order = tuple(reversed(order))
        orders.append(order)
    return orders


def gpu_processes(index: int) -> list[dict[str, Any]]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "-i",
            str(index),
            "--query-compute-apps=pid,process_name,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    rows = []
    for raw in result.stdout.splitlines():
        raw = raw.strip()
        if not raw:
            continue
        fields = [field.strip() for field in raw.split(",", 2)]
        if len(fields) != 3:
            raise RuntimeError(f"unparseable nvidia-smi compute-process row: {raw!r}")
        rows.append(
            {
                "pid": int(fields[0]),
                "process_name": fields[1],
                "used_gpu_memory_mib": fields[2],
            }
        )
    return rows


def gpu_snapshot(index: int) -> dict[str, Any]:
    fields = (
        "name,pstate,utilization.gpu,memory.used,clocks.sm,clocks.mem,"
        "power.draw,power.limit,temperature.gpu"
    )
    result = subprocess.run(
        [
            "nvidia-smi",
            "-i",
            str(index),
            f"--query-gpu={fields}",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    values = [value.strip() for value in result.stdout.strip().split(",")]
    names = fields.split(",")
    if len(values) != len(names):
        raise RuntimeError(f"unparseable nvidia-smi GPU row: {result.stdout!r}")
    return dict(zip(names, values))


def assert_no_foreign(index: int, *, allow_self: bool) -> list[dict[str, Any]]:
    rows = gpu_processes(index)
    allowed = {os.getpid()} if allow_self else set()
    foreign = [row for row in rows if row["pid"] not in allowed]
    if foreign:
        raise RuntimeError(f"foreign compute process on physical GPU{index}: {foreign}")
    return rows


def git_revision() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    return result.stdout.strip()


def tensor_error_chunked(
    actual: torch.Tensor,
    reference: torch.Tensor,
    *,
    chunk_elements: int = 4 * 1024 * 1024,
) -> dict[str, Any]:
    """Compute exactness/error without materializing a full FP64 frame tensor."""

    if actual.shape != reference.shape:
        return {
            "shape_match": False,
            "actual_shape": list(actual.shape),
            "reference_shape": list(reference.shape),
        }
    if actual.dtype != reference.dtype:
        dtype_match = False
    else:
        dtype_match = True
    actual_flat = actual.contiguous().view(-1)
    reference_flat = reference.contiguous().view(-1)
    equal = dtype_match
    max_abs = 0.0
    absolute_sum = 0.0
    difference_square_sum = 0.0
    reference_square_sum = 0.0
    for start in range(0, actual_flat.numel(), chunk_elements):
        stop = min(start + chunk_elements, actual_flat.numel())
        actual_raw = actual_flat[start:stop]
        reference_raw = reference_flat[start:stop]
        equal = equal and bool(torch.equal(actual_raw, reference_raw))
        actual_chunk = actual_raw.double()
        reference_chunk = reference_raw.double()
        difference = actual_chunk - reference_chunk
        max_abs = max(max_abs, float(difference.abs().max().item()))
        absolute_sum += float(difference.abs().sum().item())
        difference_square_sum += float(difference.square().sum().item())
        reference_square_sum += float(reference_chunk.square().sum().item())
        del actual_chunk, reference_chunk, difference
    return {
        "shape_match": True,
        "dtype_match": dtype_match,
        "bitwise_equal": equal,
        "max_abs": max_abs,
        "mean_abs": absolute_sum / actual_flat.numel(),
        "rel_l2": (
            math.sqrt(difference_square_sum / reference_square_sum)
            if reference_square_sum
            else None
        ),
    }


class BoundedTileBatcher:
    """Process-local ``_run_tile_tasks`` replacement with a strict batch cap."""

    def __init__(self, original: Callable[..., Any]) -> None:
        self.original = original
        self.cap = 1
        self.last_tile_indices: list[int] = []
        self.last_batch_sizes: list[int] = []

    def set_cap(self, cap: int) -> None:
        if cap not in CAPS:
            raise ValueError(f"unsupported cap {cap}")
        self.cap = cap
        self.last_tile_indices = []
        self.last_batch_sizes = []

    def run(
        self,
        _model: Any,
        tiles: list[torch.Tensor],
        tile_indices: list[int],
        forward_fn: Callable[[torch.Tensor], torch.Tensor],
        stack_tiling: bool,
        cls_agg: Any = None,
    ) -> list[torch.Tensor]:
        self.last_tile_indices = list(tile_indices)
        if self.cap == 1:
            # This is intentionally the current released exact route: no input
            # cat, one B1 decoder call per tile, and the checkpoint's own loop.
            self.last_batch_sizes = [1] * len(tile_indices)
            return self.original(tiles, tile_indices, forward_fn, False, cls_agg)
        if not stack_tiling:
            raise RuntimeError("bounded tile batching requires model.stack_tiling=True")
        if not tile_indices:
            self.last_batch_sizes = []
            return []

        sample_batch_size = int(tiles[tile_indices[0]].shape[0])
        output_tiles: list[torch.Tensor] = []
        self.last_batch_sizes = []
        for start in range(0, len(tile_indices), self.cap):
            batch_indices = tile_indices[start : start + self.cap]
            first_shape = tiles[batch_indices[0]].shape
            if any(tiles[index].shape != first_shape for index in batch_indices):
                raise RuntimeError(
                    f"cap={self.cap} cannot stack unequal tile shapes: "
                    f"{[tuple(tiles[index].shape) for index in batch_indices]}"
                )
            if len(batch_indices) == 1:
                output_batch = forward_fn(tiles[batch_indices[0]])
            else:
                tile_batch = torch.cat([tiles[index] for index in batch_indices], dim=0)
                output_batch = forward_fn(tile_batch)
                del tile_batch
            expected_batch = len(batch_indices) * sample_batch_size
            if int(output_batch.shape[0]) != expected_batch:
                raise RuntimeError(
                    f"decoder batch mismatch for cap={self.cap}: "
                    f"got {tuple(output_batch.shape)}, expected B={expected_batch}"
                )
            output_tiles.extend(
                output_batch.unflatten(
                    0,
                    (len(batch_indices), sample_batch_size),
                ).unbind(dim=0)
            )
            if cls_agg is not None:
                cls_agg.collect_stacked(len(batch_indices), sample_batch_size)
            self.last_batch_sizes.append(len(batch_indices))
        return output_tiles


def install_bounded_batcher(model: Any) -> tuple[BoundedTileBatcher, Callable[[], None]]:
    original = model._run_tile_tasks
    had_instance_override = "_run_tile_tasks" in model.__dict__
    old_instance_override = model.__dict__.get("_run_tile_tasks")
    batcher = BoundedTileBatcher(original)
    model._run_tile_tasks = MethodType(batcher.run, model)

    def restore() -> None:
        if had_instance_override:
            model.__dict__["_run_tile_tasks"] = old_instance_override
        else:
            model.__dict__.pop("_run_tile_tasks", None)

    return batcher, restore


def write_artifacts(output_dir: Path, payload: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "results.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# MiniMax-H3 bounded VAE tile-batching benchmark (SM120)",
        "",
        f"Status: **{payload['status']}**",
        "",
        "This benchmark invokes the real FL2VA VAE's full `_adaptive_decode` on one "
        "`[1,24,7,44,80]` latent window at the checkpoint's tile256/overlap64 geometry. "
        "The resulting 704×1280 canvas has a 4×7 grid (28 tiles). Batching is installed "
        "only by a temporary process-local `_run_tile_tasks` wrapper.",
        "",
    ]
    if payload.get("error"):
        lines.extend(["## Failure", "", f"`{payload['error']}`", ""])
    if payload.get("gpu"):
        gpu = payload["gpu"]
        lines.extend(
            [
                "## Contract",
                "",
                f"- GPU: {gpu['name']} (SM {gpu['compute_capability']}), physical GPU "
                f"{payload['physical_gpu']}",
                f"- Checkpoint: `{payload['component_path']}`",
                f"- Samples: {payload['samples']} per cap after {payload['warmups']} warmup(s)",
                "- cap1 calls the original serial helper with `stack_tiling=False`; it is the "
                "current exact route.",
                "- Decoder ViT FP32 normalization is enabled and FF torch.compile is disabled.",
                "- CUDA events bracket `_adaptive_decode`, including split, all 28 decoder "
                "tile calls/batches, blend, and stitch. There are no PP collectives.",
                "",
            ]
        )
    for backend in ("stock", "exact"):
        result = payload.get("backends", {}).get(backend)
        if not result:
            continue
        lines.extend(
            [
                f"## {backend} backend",
                "",
                f"Exact operators installed: **{result.get('exact_ops_installed')}**. "
                f"Model load: {result.get('model_load_seconds', 0.0):.3f} s.",
                "",
                "| Cap | Status | Batches/window | n | p10 ms | p50 ms | p90 ms | "
                "Speedup | Peak alloc GiB | Peak reserved GiB |",
                "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        cap1_p50 = result.get("caps", {}).get("1", {}).get("p50_ms")
        for cap in CAPS:
            arm = result.get("caps", {}).get(str(cap), {})
            if not arm:
                continue
            if arm.get("status") != "complete":
                lines.append(
                    f"| {cap} | {arm.get('status')} | — | {arm.get('sample_count', 0)} | "
                    "— | — | — | — | — | — |"
                )
                continue
            speedup = cap1_p50 / arm["p50_ms"] if cap1_p50 else math.nan
            batches = batch_plan_text(arm["batch_sizes"])
            lines.append(
                f"| {cap} | complete | {batches} | {arm['sample_count']} | "
                f"{arm['p10_ms']:.3f} | {arm['p50_ms']:.3f} | {arm['p90_ms']:.3f} | "
                f"{speedup:.3f}× | {arm['peak_allocated_gib']:.3f} | "
                f"{arm['peak_reserved_gib']:.3f} |"
            )
        lines.extend(
            [
                "",
                "### Numerical comparison with cap1",
                "",
                "| Cap | Shape + dtype match | Bitwise | max abs | mean abs | rel L2 |",
                "|---:|:---:|:---:|---:|---:|---:|",
            ]
        )
        for cap in CAPS:
            comparison = result.get("comparisons_vs_cap1", {}).get(str(cap))
            if not comparison or not comparison.get("shape_match"):
                continue
            lines.append(
                f"| {cap} | {comparison.get('dtype_match')} | "
                f"{comparison.get('bitwise_equal')} | {comparison.get('max_abs'):.8g} | "
                f"{comparison.get('mean_abs'):.8g} | {comparison.get('rel_l2'):.8g} |"
            )
        if cap1_p50:
            best = min(
                (
                    arm
                    for arm in result.get("caps", {}).values()
                    if arm.get("status") == "complete"
                ),
                key=lambda arm: arm["p50_ms"],
            )
            lines.extend(
                [
                    "",
                    f"Best measured cap is **{best['cap']}**, {best['p50_ms']:.3f} ms/window "
                    f"({cap1_p50 / best['p50_ms']:.3f}× vs cap1). For the canonical 21-window "
                    "temporal-PP8 schedule, the critical ranks own three windows, so its "
                    f"compute-only projection is **{3 * best['p50_ms']:.3f} ms** before "
                    "gathers, callback work, and pipeline fill/drain.",
                    "",
                ]
            )
    cross = payload.get("cross_backend_cap1")
    if cross:
        lines.extend(
            [
                "## Stock versus exact-op cap1",
                "",
                f"Bitwise={cross['bitwise_equal']}, max_abs={cross['max_abs']:.8g}, "
                f"mean_abs={cross['mean_abs']:.8g}, relL2={cross['rel_l2']:.8g}.",
                "",
            ]
        )
    stock = payload.get("backends", {}).get("stock")
    exact = payload.get("backends", {}).get("exact")
    if stock and exact:
        stock_cap1 = stock.get("caps", {}).get("1", {})
        exact_cap1 = exact.get("caps", {}).get("1", {})
        exact_complete = [
            arm
            for arm in exact.get("caps", {}).values()
            if arm.get("status") == "complete"
        ]
        if stock_cap1.get("p50_ms") and exact_cap1.get("p50_ms") and exact_complete:
            best_exact = min(exact_complete, key=lambda arm: arm["p50_ms"])
            lines.extend(
                [
                    "## Decision",
                    "",
                    "- Strict bit-exact/lossless lane: retain cap1. Exact-op cap1 is "
                    f"{stock_cap1['p50_ms'] / exact_cap1['p50_ms']:.3f}x faster than "
                    "stock cap1, subject to the cross-backend equality result above.",
                    f"- Best exact-op numerical candidate: cap{best_exact['cap']} at "
                    f"{best_exact['p50_ms']:.3f} ms/window "
                    f"({exact_cap1['p50_ms'] / best_exact['p50_ms']:.3f}x versus "
                    "exact-op cap1). It is not lossless unless its comparison row is bitwise.",
                    "- Any three-window compute saving is only an optimistic ceiling before "
                    "gathers, callbacks, and pipeline fill/drain; complete E2E must be measured.",
                    "",
                ]
            )
    lines.extend(
        [
            "## PP interpretation",
            "",
            "- Current spatial PP8 assigns the 28 tiles round-robin, so the maximum local "
            "share is **4 tiles/window** (four ranks get 4; four get 3). Consequently cap4 "
            "is the largest meaningful bounded batch there; caps 7/14/28 would all collapse "
            "to a local batch of at most 4.",
            "- Temporal PP8 assigns complete temporal windows round-robin. A rank processing "
            "one window owns all **28 tiles**, so every measured cap is relevant; over 21 "
            "windows the critical rank processes 3 windows.",
            "- These are one-GPU compute measurements, not full-clip E2E projections. "
            "Collectives, temporal blend/callback, D2H, CPU MP4, and interference are excluded.",
            "- Batching can select different GEMM/attention kernels and therefore is not "
            "assumed bit-exact. The table above is the deciding fidelity evidence; no setting "
            "is wired into production by this benchmark.",
            "",
        ]
    )
    (output_dir / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def run_backend(
    *,
    backend: str,
    args: argparse.Namespace,
    normalized_latent_cpu: torch.Tensor,
    device: torch.device,
) -> tuple[dict[str, Any], torch.Tensor | None]:
    exact = backend == "exact"
    os.environ[EXACT_OPS_ENV] = "1" if exact else "0"
    assert_no_foreign(args.physical_gpu, allow_self=True)

    from vllm_omni.diffusion.models.minimax_h3.vae import MiniMaxH3VideoVAE

    adapter_source = Path(inspect.getfile(MiniMaxH3VideoVAE)).resolve()
    if REPO_ROOT not in adapter_source.parents:
        raise RuntimeError(
            "loaded MiniMaxH3VideoVAE from the wrong checkout: "
            f"source={adapter_source}, expected under {REPO_ROOT}"
        )

    load_start = time.perf_counter()
    vae = MiniMaxH3VideoVAE(str(args.component), device=device)
    vae.eval()
    torch.accelerator.synchronize(device)
    result: dict[str, Any] = {
        "backend": backend,
        "model_load_seconds": time.perf_counter() - load_start,
        "caps": {},
        "comparisons_vs_cap1": {},
        "adapter_source": str(adapter_source),
    }
    model = vae.remote.model
    exact_installed = bool(
        getattr(vae.model.decoder, "_omni_h3_vae_optimizations_installed", False)
    )
    result["exact_ops_installed"] = exact_installed
    if exact_installed != exact:
        raise RuntimeError(
            f"backend={backend} exact installer state mismatch: installed={exact_installed}"
        )
    if int(model.decoder_tile_size) != 256 or int(model.decoder_tile_overlap_min) != 64:
        raise RuntimeError(
            "expected checkpoint tile256/overlap64, got "
            f"tile={model.decoder_tile_size}, overlap={model.decoder_tile_overlap_min}"
        )
    result["checkpoint_parallel_tiling_initial"] = bool(model.parallel_tiling)

    mean = torch.tensor(vae.config_dict["latents_mean"], dtype=torch.float32).view(
        1, 24, 1, 1, 1
    )
    std = torch.tensor(vae.config_dict["latents_std"], dtype=torch.float32).view(
        1, 24, 1, 1, 1
    )
    latent_holder = [(normalized_latent_cpu * std + mean).to(device)]
    del mean, std

    saved = {
        "stack_tiling": model.stack_tiling,
        "parallel_tiling": model.parallel_tiling,
        "decoder_tiling": model.decoder_tiling,
        "decoder_tile_size": model.decoder_tile_size,
    }
    model.stack_tiling = True
    model.parallel_tiling = False
    model.decoder_tiling = True
    model.decoder_tile_size = 256
    batcher, restore_batcher = install_bounded_batcher(model)
    disabled: set[int] = set()
    samples: dict[int, list[float]] = {cap: [] for cap in CAPS}
    last_contract: dict[int, dict[str, Any]] = {}

    def decode_once(cap: int) -> tuple[float, torch.Tensor, dict[str, int]]:
        batcher.set_cap(cap)
        torch.accelerator.reset_peak_memory_stats(device)
        baseline_allocated = int(torch.cuda.memory_allocated(device))
        baseline_reserved = int(torch.cuda.memory_reserved(device))
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        with torch.inference_mode(), torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=True,
        ):
            output = model._adaptive_decode(latent_holder[0])
        end.record()
        end.synchronize()
        elapsed_ms = float(start.elapsed_time(end))
        if tuple(output.shape) != EXPECTED_OUTPUT_SHAPE:
            raise RuntimeError(
                f"unexpected cap={cap} output shape {tuple(output.shape)}, "
                f"expected {EXPECTED_OUTPUT_SHAPE}"
            )
        if batcher.last_tile_indices != list(range(TILE_COUNT)):
            raise RuntimeError(
                f"cap={cap} tile order mismatch: {batcher.last_tile_indices}"
            )
        if sum(batcher.last_batch_sizes) != TILE_COUNT or max(
            batcher.last_batch_sizes, default=0
        ) > cap:
            raise RuntimeError(
                f"cap={cap} invalid batch plan {batcher.last_batch_sizes}"
            )
        last_contract[cap] = {
            "tile_indices": batcher.last_tile_indices,
            "batch_sizes": batcher.last_batch_sizes,
            "output_shape": list(output.shape),
            "output_dtype": str(output.dtype),
        }
        peaks = {
            "baseline_allocated_bytes": baseline_allocated,
            "baseline_reserved_bytes": baseline_reserved,
            "peak_allocated_bytes": int(torch.accelerator.max_memory_allocated(device)),
            "peak_reserved_bytes": int(torch.accelerator.max_memory_reserved(device)),
        }
        del start, end
        return elapsed_ms, output, peaks

    def mark_oom(cap: int, exc: BaseException) -> None:
        result["caps"][str(cap)] = {
            "cap": cap,
            "status": "oom",
            "sample_count": len(samples[cap]),
            "error": f"{type(exc).__name__}: {exc}",
        }
        for larger in CAPS:
            if larger >= cap:
                disabled.add(larger)
                if larger != cap:
                    result["caps"].setdefault(
                        str(larger),
                        {
                            "cap": larger,
                            "status": "skipped_after_lower_cap_oom",
                            "sample_count": 0,
                        },
                    )
        gc.collect()
        torch.accelerator.empty_cache()

    try:
        # Warm each dynamic batch shape. Larger arms are skipped after the first
        # OOM because activation memory is monotonic in this fixed geometry.
        for _ in range(args.warmups):
            for cap in CAPS:
                if cap in disabled:
                    continue
                try:
                    _, output, _ = decode_once(cap)
                    del output
                except torch.OutOfMemoryError as exc:
                    mark_oom(cap, exc)
        torch.accelerator.synchronize(device)

        # Alternate arm order within one model/process to limit clock drift.
        for order in balanced_orders(args.samples):
            assert_no_foreign(args.physical_gpu, allow_self=True)
            for cap in order:
                if cap in disabled:
                    continue
                try:
                    elapsed_ms, output, _ = decode_once(cap)
                    samples[cap].append(elapsed_ms)
                    del output
                except torch.OutOfMemoryError as exc:
                    mark_oom(cap, exc)

        # Clean one-pass memory probes and numerical outputs. Emptying the
        # allocator cache keeps peak-reserved attribution local to each cap.
        cap1_output: torch.Tensor | None = None
        for cap in CAPS:
            if cap in disabled:
                continue
            gc.collect()
            torch.accelerator.empty_cache()
            try:
                _, output, peaks = decode_once(cap)
                output_cpu = output.detach().cpu().contiguous()
                del output
            except torch.OutOfMemoryError as exc:
                mark_oom(cap, exc)
                continue
            timing = summarize(samples[cap])
            contract = last_contract[cap]
            timing.update(
                {
                    "cap": cap,
                    "status": "complete",
                    "batch_sizes": contract["batch_sizes"],
                    "batch_count": len(contract["batch_sizes"]),
                    "tile_indices": contract["tile_indices"],
                    "output_shape": contract["output_shape"],
                    "output_dtype": contract["output_dtype"],
                    **peaks,
                    "peak_allocated_gib": peaks["peak_allocated_bytes"] / (1024**3),
                    "peak_reserved_gib": peaks["peak_reserved_bytes"] / (1024**3),
                    "incremental_peak_allocated_gib": (
                        peaks["peak_allocated_bytes"] - peaks["baseline_allocated_bytes"]
                    )
                    / (1024**3),
                }
            )
            result["caps"][str(cap)] = timing
            if cap == 1:
                cap1_output = output_cpu
                result["comparisons_vs_cap1"]["1"] = tensor_error_chunked(
                    cap1_output,
                    cap1_output,
                )
            else:
                if cap1_output is None:
                    raise RuntimeError("cap1 reference was not produced before comparison")
                result["comparisons_vs_cap1"][str(cap)] = tensor_error_chunked(
                    output_cpu,
                    cap1_output,
                )
                del output_cpu

        if cap1_output is None:
            raise RuntimeError(f"backend={backend} did not produce the cap1 reference")
        cap1_p50 = result["caps"]["1"]["p50_ms"]
        for arm in result["caps"].values():
            if arm.get("status") == "complete":
                arm["speedup_vs_cap1"] = cap1_p50 / arm["p50_ms"]
                arm["temporal_pp8_critical_three_window_compute_ms"] = 3 * arm["p50_ms"]
        result["status"] = (
            "complete_with_oom" if any(cap in disabled for cap in CAPS) else "complete"
        )
        return result, cap1_output
    finally:
        restore_batcher()
        for name, value in saved.items():
            setattr(model, name, value)
        latent_holder.clear()
        del vae
        gc.collect()
        torch.accelerator.empty_cache()
        torch.accelerator.synchronize(device)
        # Nested benchmark closures still hold the model until this function's
        # frame returns. These are diagnostic pre-return allocator readings,
        # not a post-exit cleanup assertion.
        result["pre_return_allocated_bytes"] = int(torch.cuda.memory_allocated(device))
        result["pre_return_reserved_bytes"] = int(torch.cuda.memory_reserved(device))


def main() -> int:
    args = parse_args()
    output_dir = (
        args.output_dir
        or DEFAULT_PROFILE_ROOT / f"minimax-h3-vae-stack-tiling-sm120-{stamp()}"
    ).resolve()
    payload: dict[str, Any] = {
        "status": "initializing",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "benchmark_script": str(Path(__file__).resolve()),
        "repo_root": str(REPO_ROOT),
        "git_revision": git_revision(),
        "component_path": str(args.component.resolve()),
        "physical_gpu": args.physical_gpu,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "seed": args.seed,
        "latent_shape": list(LATENT_SHAPE),
        "latent_dtype_before_denormalization": "torch.float32",
        "expected_output_shape": list(EXPECTED_OUTPUT_SHAPE),
        "tile_size": 256,
        "tile_overlap": 64,
        "tile_grid": [4, 7],
        "tile_count": TILE_COUNT,
        "caps": list(CAPS),
        "warmups": args.warmups,
        "samples": args.samples,
        "requested_backends": args.backends,
        "spatial_pp8_max_local_tiles_per_window": 4,
        "temporal_pp8_local_tiles_per_owned_window": 28,
        "temporal_windows": 21,
        "temporal_pp8_max_owned_windows": 3,
        "production_code_modified": False,
        "backends": {},
    }
    write_artifacts(output_dir, payload)

    try:
        expected_visible = str(args.physical_gpu)
        if os.environ.get("CUDA_VISIBLE_DEVICES") != expected_visible:
            raise RuntimeError(
                f"CUDA_VISIBLE_DEVICES must equal {expected_visible!r}, got "
                f"{os.environ.get('CUDA_VISIBLE_DEVICES')!r}"
            )
        if not args.component.is_dir() or not (
            args.component / "source" / "model.safetensors"
        ).is_file():
            raise FileNotFoundError(f"real VAE component not found: {args.component}")
        payload["preflight_processes"] = assert_no_foreign(
            args.physical_gpu,
            allow_self=False,
        )
        payload["gpu_snapshot_before"] = gpu_snapshot(args.physical_gpu)
        sys.path.insert(0, str(REPO_ROOT))
        os.environ["MINIMAX_H3_VAE_DECODER_STREAM_TEMPORAL_CAT"] = "1"
        os.environ[TILE_SIZE_ENV] = "0"
        os.environ[FP32_NORM_ENV] = "1"
        os.environ[FF_COMPILE_ENV] = "0"
        payload["environment_contract"] = {
            "MINIMAX_H3_VAE_DECODER_STREAM_TEMPORAL_CAT": "1",
            TILE_SIZE_ENV: "0",
            FP32_NORM_ENV: "1",
            FF_COMPILE_ENV: "0",
        }

        if not torch.accelerator.is_available() or torch.accelerator.device_count() != 1:
            raise RuntimeError(
                f"expected one visible CUDA GPU, available={torch.accelerator.is_available()} "
                f"count={torch.accelerator.device_count()}"
            )
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
        properties = torch.cuda.get_device_properties(device)
        payload["gpu"] = {
            "name": properties.name,
            "compute_capability": f"{properties.major}.{properties.minor}",
            "total_memory_gib": properties.total_memory / (1024**3),
        }
        payload["torch_version"] = torch.__version__
        payload["cuda_version"] = torch.version.cuda
        if (properties.major, properties.minor) != (12, 0):
            raise RuntimeError(
                f"this benchmark requires SM120, got SM{properties.major}{properties.minor}"
            )

        generator = torch.Generator(device="cpu").manual_seed(args.seed)
        normalized_latent_cpu = torch.randn(
            LATENT_SHAPE,
            generator=generator,
            dtype=torch.float32,
        )
        payload["latent_stats"] = {
            "min": float(normalized_latent_cpu.min().item()),
            "max": float(normalized_latent_cpu.max().item()),
            "mean": float(normalized_latent_cpu.mean().item()),
            "std": float(normalized_latent_cpu.std().item()),
        }

        backends = ("stock", "exact") if args.backends == "both" else (args.backends,)
        references: dict[str, torch.Tensor] = {}
        for backend in backends:
            result, reference = run_backend(
                backend=backend,
                args=args,
                normalized_latent_cpu=normalized_latent_cpu,
                device=device,
            )
            payload["backends"][backend] = result
            if reference is not None:
                references[backend] = reference
            # ``run_backend`` has returned, so its model-holding closure frame
            # is now collectible. Release cached HBM before the next load.
            gc.collect()
            torch.accelerator.empty_cache()
            torch.accelerator.synchronize(device)
            result["after_return_allocated_bytes"] = int(
                torch.cuda.memory_allocated(device)
            )
            result["after_return_reserved_bytes"] = int(
                torch.cuda.memory_reserved(device)
            )
            write_artifacts(output_dir, payload)

        if "stock" in references and "exact" in references:
            payload["cross_backend_cap1"] = tensor_error_chunked(
                references["exact"],
                references["stock"],
            )
        payload["in_process_snapshot_before_exit"] = gpu_snapshot(args.physical_gpu)
        payload["in_process_final_processes"] = assert_no_foreign(
            args.physical_gpu,
            allow_self=True,
        )
        payload["status"] = (
            "complete_with_oom"
            if any(
                result.get("status") == "complete_with_oom"
                for result in payload["backends"].values()
            )
            else "complete"
        )
        write_artifacts(output_dir, payload)
        print(output_dir)
        for backend, result in payload["backends"].items():
            for cap in CAPS:
                arm = result["caps"].get(str(cap), {})
                if arm.get("status") == "complete":
                    comparison = result["comparisons_vs_cap1"][str(cap)]
                    print(
                        f"{backend} cap={cap:2d} p50={arm['p50_ms']:.3f}ms "
                        f"peak={arm['peak_allocated_gib']:.3f}GiB "
                        f"bitwise={comparison['bitwise_equal']} "
                        f"max_abs={comparison['max_abs']:.8g}"
                    )
                else:
                    print(f"{backend} cap={cap:2d} status={arm.get('status', 'missing')}")
        return 0
    except (RuntimeError, FileNotFoundError, subprocess.SubprocessError) as exc:
        payload["status"] = "failed"
        payload["error"] = f"{type(exc).__name__}: {exc}"
        write_artifacts(output_dir, payload)
        print(output_dir, file=sys.stderr)
        print(payload["error"], file=sys.stderr)
        return 1
    finally:
        if torch.accelerator.is_available():
            gc.collect()
            torch.accelerator.empty_cache()


if __name__ == "__main__":
    raise SystemExit(main())
