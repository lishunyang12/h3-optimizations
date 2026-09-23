#!/usr/bin/env python3
"""Qualify the opt-in MiniMax-H3 VAE exact operators on one SM120 GPU."""

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
from typing import Any, Callable

import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = REPO_ROOT.parents[1]
DEFAULT_COMPONENT = WORKSPACE_ROOT / "MiniMax-H3" / "FL2VA" / "video_vae"
DEFAULT_PROFILE_ROOT = WORKSPACE_ROOT / "results" / "profile"
LATENT_SHAPE = (1, 24, 7, 44, 80)
QK_SHAPE = (4, 1797, 32, 192)  # PP8 critical rank, tile256: QKV packed shape
RESIDUAL_SHAPE = (4, 1797, 2048)


def args_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--component", type=Path, default=DEFAULT_COMPONENT)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--physical-gpu", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1101)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--op-warmups", type=int, default=5)
    parser.add_argument("--op-samples", type=int, default=50)
    args = parser.parse_args()
    if min(args.warmups, args.samples, args.op_warmups, args.op_samples) < 1:
        parser.error("all warmup/sample counts must be positive")
    return args


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def gpu_processes(index: int) -> list[dict[str, Any]]:
    command = [
        "nvidia-smi",
        "-i",
        str(index),
        "--query-compute-apps=pid,process_name,used_gpu_memory",
        "--format=csv,noheader,nounits",
    ]
    result = subprocess.run(command, check=True, text=True, capture_output=True)
    rows = []
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        fields = [field.strip() for field in line.split(",", 2)]
        rows.append(
            {
                "pid": int(fields[0]),
                "process_name": fields[1],
                "used_gpu_memory_mib": fields[2],
            }
        )
    return rows


def assert_no_foreign(index: int, *, allow_self: bool) -> list[dict[str, Any]]:
    rows = gpu_processes(index)
    allowed = {os.getpid()} if allow_self else set()
    foreign = [row for row in rows if row["pid"] not in allowed]
    if foreign:
        raise RuntimeError(f"foreign process on physical GPU{index}: {foreign}")
    return rows


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    low, high = math.floor(position), math.ceil(position)
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


def errors(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, Any]:
    if actual.shape != expected.shape:
        return {"shape_match": False, "actual_shape": list(actual.shape), "expected_shape": list(expected.shape)}
    diff = actual.float() - expected.float()
    expected_l2 = float(torch.linalg.vector_norm(expected.double()).item())
    diff_l2 = float(torch.linalg.vector_norm(diff.double()).item())
    return {
        "shape_match": True,
        "bitwise_equal": bool(torch.equal(actual, expected)),
        "max_abs": float(diff.abs().max().item()),
        "mean_abs": float(diff.abs().mean().item()),
        "rel_l2": diff_l2 / expected_l2 if expected_l2 else None,
    }


def cuda_bench(
    fn: Callable[[], Any], *, warmups: int, samples: int
) -> tuple[dict[str, Any], Any]:
    last = None
    for _ in range(warmups):
        last = fn()
    torch.cuda.synchronize()
    elapsed = []
    for _ in range(samples):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        last = fn()
        end.record()
        end.synchronize()
        elapsed.append(float(start.elapsed_time(end)))
    return summarize(elapsed), last


def qk_reference(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    def one(x: torch.Tensor) -> torch.Tensor:
        normalized = F.rms_norm(x.float(), (64,), None, 1e-5).to(x.dtype)
        rotary = normalized[..., :48]
        first, second = rotary.chunk(2, dim=-1)
        rotated = torch.cat((-second, first), dim=-1)
        return torch.cat((rotary * cos + rotated * sin, normalized[..., 48:]), dim=-1)

    return one(q), one(k)


def write_artifacts(output_dir: Path, payload: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "results.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    lines = [
        "# MiniMax-H3 VAE exact-op SM120 qualification",
        "",
        f"Status: **{payload['status']}**",
        "",
        "This is a single-GPU qualification. The real-weight test uses one "
        "`[1,24,7,44,80]` window, tile256, and excludes PP8/full-video E2E.",
        "",
    ]
    if payload.get("error"):
        lines.extend(["## Failure", "", f"`{payload['error']}`", ""])
    if payload.get("per_op"):
        lines.extend(
            [
                "## Isolated exact operators",
                "",
                "| Operator | Shape | Eager p50 (ms) | Fused p50 (ms) | Speedup | Bitwise | max abs |",
                "|---|---|---:|---:|---:|:---:|---:|",
            ]
        )
        for key in ("qk_norm_rope", "scaled_residual"):
            item = payload["per_op"][key]
            lines.append(
                f"| {key} | `{item['shape']}` | {item['eager']['p50_ms']:.4f} | "
                f"{item['fused']['p50_ms']:.4f} | {item['speedup_p50']:.3f}× | "
                f"{item['error']['bitwise_equal']} | {item['error']['max_abs']:.8g} |"
            )
        lines.append("")
    if payload.get("real_weight"):
        lines.extend(
            [
                "## Real-weight one-window A/B",
                "",
                "| Arm | Samples | p10 (ms) | p50 (ms) | p90 (ms) | Peak alloc (GiB) | Peak reserved (GiB) |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for key in ("stock", "forced_exact_ops"):
            arm = payload["real_weight"][key]
            lines.append(
                f"| {key} | {arm['sample_count']} | {arm['p10_ms']:.3f} | "
                f"{arm['p50_ms']:.3f} | {arm['p90_ms']:.3f} | "
                f"{arm['peak_allocated_gib']:.3f} | {arm['peak_reserved_gib']:.3f} |"
            )
        comparison = payload["real_weight"]["comparison"]
        lines.extend(
            [
                "",
                f"Forced installer speedup: **{comparison['speedup_p50']:.3f}×** "
                f"({comparison['latency_reduction_fraction'] * 100:.2f}% latency reduction).",
                "",
                f"Output equality: bitwise={comparison['output_error']['bitwise_equal']}, "
                f"max_abs={comparison['output_error']['max_abs']:.8g}, "
                f"mean_abs={comparison['output_error']['mean_abs']:.8g}, "
                f"relL2={comparison['output_error']['rel_l2']:.8g}.",
                "",
                "The stock arm runs first because installation irreversibly pre-casts eligible "
                "decoder block linears to FP16. CUDA events exclude CPU copies and metrics.",
                "",
            ]
        )
    (output_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = args_parser()
    output_dir = (args.output_dir or (DEFAULT_PROFILE_ROOT / f"minimax-h3-vae-exact-ops-sm120-{stamp()}"))
    output_dir = output_dir.resolve()
    payload: dict[str, Any] = {
        "status": "initializing",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "component_path": str(args.component.resolve()),
        "physical_gpu": args.physical_gpu,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "seed": args.seed,
        "real_weight_latent_shape": list(LATENT_SHAPE),
        "tile_size": 256,
        "tile_overlap": 64,
        "limitation": "single GPU, one temporal window; no PP8 or full-video E2E",
    }
    write_artifacts(output_dir, payload)
    try:
        if os.environ.get("CUDA_VISIBLE_DEVICES") != str(args.physical_gpu):
            raise RuntimeError(
                f"CUDA_VISIBLE_DEVICES must equal {args.physical_gpu}, got "
                f"{os.environ.get('CUDA_VISIBLE_DEVICES')!r}"
            )
        payload["preflight_processes"] = assert_no_foreign(args.physical_gpu, allow_self=False)
        sys.path.insert(0, str(REPO_ROOT))
        os.environ.setdefault("MINIMAX_H3_VAE_DECODER_STREAM_TEMPORAL_CAT", "1")

        from vllm_omni.diffusion.models.minimax_h3.ops import vae as vae_ops
        from vllm_omni.diffusion.models.minimax_h3.ops.vae.dispatch import H3VAEOperatorSet
        from vllm_omni.diffusion.models.minimax_h3.ops.vae.qk_norm_rope import try_qk_norm_rope_exact
        from vllm_omni.diffusion.models.minimax_h3.ops.vae.scaled_residual import try_scaled_residual_exact
        from vllm_omni.diffusion.models.minimax_h3.vae import MiniMaxH3VideoVAE

        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError(f"expected one visible CUDA GPU, got {torch.cuda.device_count()}")
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
        props = torch.cuda.get_device_properties(device)
        payload["gpu"] = {
            "name": props.name,
            "compute_capability": f"{props.major}.{props.minor}",
            "total_memory_gib": props.total_memory / (1024**3),
        }
        if (props.major, props.minor) not in ((12, 0), (12, 1)):
            raise RuntimeError(f"this qualification requires SM120/121, got {props.major}.{props.minor}")

        torch.manual_seed(args.seed)
        qkv = torch.randn(QK_SHAPE, device=device, dtype=torch.float16)
        q, k, _ = qkv.chunk(3, dim=-1)
        cos = torch.randn(QK_SHAPE[0], QK_SHAPE[1], 1, 48, device=device, dtype=torch.float16)
        sin = torch.randn_like(cos)
        residual = torch.randn(RESIDUAL_SHAPE, device=device, dtype=torch.float32)
        branch = torch.randn(RESIDUAL_SHAPE, device=device, dtype=torch.float16)
        scale = torch.randn(RESIDUAL_SHAPE[-1], device=device, dtype=torch.float32)

        with torch.inference_mode():
            expected_q, expected_k = qk_reference(q, k, cos, sin)
            fused_qk = try_qk_norm_rope_exact(q, k, (cos, sin), 1e-5)
            if fused_qk is None:
                raise RuntimeError("SM120 qk_norm_rope exact kernel rejected production inputs")
            q_error = errors(fused_qk[0], expected_q)
            k_error = errors(fused_qk[1], expected_k)
            qk_error = {
                "shape_match": q_error["shape_match"] and k_error["shape_match"],
                "bitwise_equal": q_error["bitwise_equal"] and k_error["bitwise_equal"],
                "max_abs": max(q_error["max_abs"], k_error["max_abs"]),
                "mean_abs": (q_error["mean_abs"] + k_error["mean_abs"]) / 2,
                "q": q_error,
                "k": k_error,
            }
            expected_residual = residual + branch * scale
            fused_residual = try_scaled_residual_exact(residual, branch, scale)
            if fused_residual is None:
                raise RuntimeError("SM120 scaled_residual exact kernel rejected production inputs")
            residual_error = errors(fused_residual, expected_residual)

            qk_eager_timing, _ = cuda_bench(
                lambda: qk_reference(q, k, cos, sin),
                warmups=args.op_warmups,
                samples=args.op_samples,
            )
            qk_fused_timing, _ = cuda_bench(
                lambda: try_qk_norm_rope_exact(q, k, (cos, sin), 1e-5),
                warmups=args.op_warmups,
                samples=args.op_samples,
            )
            residual_eager_timing, _ = cuda_bench(
                lambda: residual + branch * scale,
                warmups=args.op_warmups,
                samples=args.op_samples,
            )
            residual_fused_timing, _ = cuda_bench(
                lambda: try_scaled_residual_exact(residual, branch, scale),
                warmups=args.op_warmups,
                samples=args.op_samples,
            )

        payload["per_op"] = {
            "qk_norm_rope": {
                "shape": [QK_SHAPE[0], QK_SHAPE[1], 32, 64],
                "eager": qk_eager_timing,
                "fused": qk_fused_timing,
                "speedup_p50": qk_eager_timing["p50_ms"] / qk_fused_timing["p50_ms"],
                "error": qk_error,
            },
            "scaled_residual": {
                "shape": list(RESIDUAL_SHAPE),
                "eager": residual_eager_timing,
                "fused": residual_fused_timing,
                "speedup_p50": residual_eager_timing["p50_ms"] / residual_fused_timing["p50_ms"],
                "error": residual_error,
            },
        }
        if not qk_error["bitwise_equal"] or not residual_error["bitwise_equal"]:
            payload["status"] = "per_op_exactness_failed"
            write_artifacts(output_dir, payload)
            print(output_dir)
            return 3

        del qkv, q, k, cos, sin, residual, branch, scale
        del expected_q, expected_k, fused_qk, expected_residual, fused_residual
        gc.collect()
        torch.cuda.empty_cache()

        latent_generator = torch.Generator(device="cpu").manual_seed(args.seed)
        latent = torch.randn(LATENT_SHAPE, generator=latent_generator, dtype=torch.float32).to(device)
        load_start = time.perf_counter()
        vae = MiniMaxH3VideoVAE(str(args.component), device=device)
        payload["model_load_seconds"] = time.perf_counter() - load_start
        vae.remote.model.decoder_tile_size = 256
        if int(vae.remote.model.decoder_tile_overlap_min) != 64:
            raise RuntimeError("checkpoint decoder overlap is not 64")
        if getattr(vae.model.decoder, "_omni_h3_vae_optimizations_installed", False):
            raise RuntimeError("stock SM120 arm unexpectedly installed exact VAE operators")

        def decode() -> torch.Tensor:
            with torch.inference_mode(), torch.autocast(
                device_type="cuda", dtype=torch.float16, enabled=True
            ):
                return vae.decode_latent(latent)

        def real_arm() -> tuple[dict[str, Any], torch.Tensor]:
            for _ in range(args.warmups):
                warm = decode()
                del warm
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            timing, last = cuda_bench(decode, warmups=0, samples=args.samples)
            timing.update(
                {
                    "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
                    "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
                    "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / (1024**3),
                    "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / (1024**3),
                    "output_shape": list(last.shape),
                    "output_dtype": str(last.dtype),
                }
            )
            return timing, last.detach().cpu().contiguous()

        stock, stock_output = real_arm()
        exact_set = H3VAEOperatorSet(
            supports=lambda _device: True,
            qk_norm_rope=try_qk_norm_rope_exact,
            scaled_residual=try_scaled_residual_exact,
        )
        original_resolver = vae_ops.resolve_h3_vae_operators
        try:
            vae_ops.resolve_h3_vae_operators = lambda _device: exact_set
            installed = vae_ops.install_h3_vae_optimizations(vae.model.decoder, device=device)
        finally:
            vae_ops.resolve_h3_vae_operators = original_resolver
        if not installed:
            raise RuntimeError("forced installer rejected the real SM120 decoder contract")
        forced, forced_output = real_arm()
        output_error = errors(forced_output, stock_output)
        payload["real_weight"] = {
            "stock": stock,
            "forced_exact_ops": forced,
            "comparison": {
                "speedup_p50": stock["p50_ms"] / forced["p50_ms"],
                "latency_reduction_fraction": 1.0 - forced["p50_ms"] / stock["p50_ms"],
                "peak_allocated_reduction_bytes": stock["peak_allocated_bytes"]
                - forced["peak_allocated_bytes"],
                "output_error": output_error,
            },
        }
        payload["status"] = "complete"
        payload["final_processes"] = assert_no_foreign(args.physical_gpu, allow_self=True)
        write_artifacts(output_dir, payload)
        print(output_dir)
        print(
            f"qk_exact={qk_error['bitwise_equal']} "
            f"qk_speedup={payload['per_op']['qk_norm_rope']['speedup_p50']:.3f}x"
        )
        print(
            f"residual_exact={residual_error['bitwise_equal']} "
            f"residual_speedup={payload['per_op']['scaled_residual']['speedup_p50']:.3f}x"
        )
        print(
            f"real stock={stock['p50_ms']:.3f}ms forced={forced['p50_ms']:.3f}ms "
            f"speedup={payload['real_weight']['comparison']['speedup_p50']:.3f}x "
            f"bitwise={output_error['bitwise_equal']} max_abs={output_error['max_abs']}"
        )
        return 0
    except (RuntimeError, FileNotFoundError, subprocess.SubprocessError) as exc:
        payload["status"] = "oom_stopped" if isinstance(exc, torch.cuda.OutOfMemoryError) else "failed"
        payload["error"] = f"{type(exc).__name__}: {exc}"
        write_artifacts(output_dir, payload)
        print(output_dir, file=sys.stderr)
        print(payload["error"], file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
