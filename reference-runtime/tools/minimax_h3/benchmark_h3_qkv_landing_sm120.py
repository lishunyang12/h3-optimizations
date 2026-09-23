#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Go/no-go benchmark for an H3 SM120 QKV landing epilogue.

This tool deliberately does not modify vLLM's linear dispatch.  It answers two
questions before a custom CuTe DSL epilogue is allowed into production:

* how much time and HBM traffic can direct Q/K/V landing possibly remove; and
* whether the stock SM120 CuTe DSL GEMM schedule leaves enough latency budget
  for the required per-head RMSNorm and RoPE epilogue.

The candidate timed by ``run`` is only the raw GEMM.  It is not reported as a
finished fused kernel.  A real candidate must additionally satisfy
``LANDING_EPILOGUE_CONTRACT`` and beat the production baseline by the measured
post-GEMM upper bound.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import math
import statistics
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Callable


H3_LOCAL_ROWS = 11_936
H3_HIDDEN_SIZE = 5_376
H3_HEADS = 56
H3_HEAD_DIM = 128
H3_ROTARY_DIM = 96
H3_LAYERS = 50
H3_DENOISE_UPDATES = 49
BF16_BYTES = 2
MIN_LAYER_WIN_MS = 0.2

# Values from the selected one-step Nsight Systems capture.  The trace starts
# and ends at an NVTX boundary, hence 786 instead of the ideal 800 Q/K calls.
NSYS_QK_KERNEL_CALLS = 786
NSYS_QK_KERNEL_TOTAL_MS = 231.665139
NSYS_V_COPY_CALLS = 401
NSYS_V_COPY_TOTAL_MS = 132.087697

LANDING_EPILOGUE_CONTRACT: dict[str, Any] = {
    "architecture": "sm120",
    "input_dtype": "bfloat16",
    "weight_dtype": "bfloat16",
    "output_dtype": "bfloat16",
    "cta_tile_n": 128,
    "one_complete_head_per_n_tile": True,
    "split_k": False,
    "qk_projection_accumulator_rounding": "bfloat16_before_rms_reduction",
    "qk_rms_reduction": "128-wide-float32",
    "qk_normalized_rounding": "bfloat16_before_rope",
    "rotary_layout": "neox-noninterleaved-96-of-128",
    "v_transform": "identity",
    "outputs": "three-caller-owned-registered-contiguous-landings",
    "production_dispatch_modified": False,
}


@dataclass(frozen=True)
class ShapeSpec:
    m: int = H3_LOCAL_ROWS
    k: int = H3_HIDDEN_SIZE
    heads: int = H3_HEADS
    head_dim: int = H3_HEAD_DIM
    rotary_dim: int = H3_ROTARY_DIM

    def validate(self) -> None:
        for name, value in asdict(self).items():
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        if self.rotary_dim > self.head_dim or self.rotary_dim % 2:
            raise ValueError("rotary_dim must be even and no larger than head_dim")
        if self.head_dim != 128:
            raise ValueError("the candidate contract requires one 128-wide head tile")

    @property
    def one_operand_width(self) -> int:
        return self.heads * self.head_dim

    @property
    def qkv_width(self) -> int:
        return 3 * self.one_operand_width

    @property
    def input_shape(self) -> tuple[int, int]:
        return (self.m, self.k)

    @property
    def weight_shape(self) -> tuple[int, int]:
        return (self.qkv_width, self.k)

    @property
    def qkv_shape(self) -> tuple[int, int]:
        return (self.m, self.qkv_width)

    @property
    def operand_bytes(self) -> int:
        return self.m * self.one_operand_width * BF16_BYTES

    def traffic_model(self) -> dict[str, int]:
        # Current path:
        #   GEMM writes packed QKV;
        #   Q and K are each read and rewritten by norm+RoPE;
        #   row-strided V is read and copied into its registered landing.
        # Candidate path still writes three operands once, directly to the
        # registered buffers.  Only post-GEMM reads/writes disappear.
        return {
            "one_q_or_k_or_v_operand": self.operand_bytes,
            "packed_qkv_output": 3 * self.operand_bytes,
            "qk_postprocess_read_write_avoided": 4 * self.operand_bytes,
            "v_staging_read_write_avoided": 2 * self.operand_bytes,
            "total_hbm_bytes_avoided": 6 * self.operand_bytes,
        }


def measured_post_gemm_upper_bound_ms() -> dict[str, float]:
    qk_per_kernel = NSYS_QK_KERNEL_TOTAL_MS / NSYS_QK_KERNEL_CALLS
    qk_per_layer = 2.0 * qk_per_kernel
    v_per_layer = NSYS_V_COPY_TOTAL_MS / NSYS_V_COPY_CALLS
    total = qk_per_layer + v_per_layer
    return {
        "q_plus_k_norm_rope_ms_per_layer": qk_per_layer,
        "v_registered_landing_copy_ms_per_layer": v_per_layer,
        "ideal_total_ms_per_layer": total,
        "ideal_total_ms_per_step": total * H3_LAYERS,
        "ideal_total_s_per_49_update_request": total
        * H3_LAYERS
        * H3_DENOISE_UPDATES
        / 1000.0,
    }


def percentile_nearest_rank(values: list[float], percentile: float) -> float:
    if not values:
        raise ValueError("cannot summarize an empty sample")
    if not 0.0 < percentile <= 1.0:
        raise ValueError("percentile must be in (0, 1]")
    ordered = sorted(values)
    return ordered[max(0, math.ceil(percentile * len(ordered)) - 1)]


def summarize(values: list[float]) -> dict[str, Any]:
    if not values:
        raise ValueError("cannot summarize an empty sample")
    return {
        "samples_ms": values,
        "median_ms": statistics.median(values),
        "p95_ms": percentile_nearest_rank(values, 0.95),
        "min_ms": min(values),
        "max_ms": max(values),
    }


def _reference_path() -> Path:
    relative = Path(
        "flashinfer/data/cutlass/examples/python/CuTeDSL/"
        "blackwell_geforce/dense_gemm.py"
    )
    dist = importlib.metadata.distribution("flashinfer-python")
    path = Path(dist.locate_file(relative))
    if not path.is_file():
        raise RuntimeError(f"SM120 CuTe DSL dense GEMM example not found: {path}")
    return path


def _load_reference() -> ModuleType:
    path = _reference_path()
    name = f"_h3_qkv_sm120_dense_{abs(hash(path.resolve()))}"
    existing = sys.modules.get(name)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _time_cuda(call: Callable[[], Any], iterations: int) -> list[float]:
    import torch

    values: list[float] = []
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        call()
        end.record()
        end.synchronize()
        values.append(float(start.elapsed_time(end)))
    return values


def run_raw_gemm(
    shape: ShapeSpec,
    *,
    device_name: str,
    warmup: int,
    iterations: int,
    tile: tuple[int, int, int],
) -> dict[str, Any]:
    """Compare raw CuTe GEMM with the production torch linear boundary."""
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute
    import cutlass.utils as utils
    import torch
    import torch.nn.functional as F
    from cutlass.cute.runtime import from_dlpack

    shape.validate()
    device = torch.device(device_name)
    torch.cuda.set_device(device)
    props = torch.cuda.get_device_properties(device)
    if (props.major, props.minor) != (12, 0):
        raise RuntimeError(f"SM120 required, got SM{props.major}{props.minor}")
    if tile[1] != shape.head_dim:
        raise ValueError("tile N must equal the complete 128-wide H3 head")

    torch.manual_seed(1101)
    x = torch.randn(shape.input_shape, device=device, dtype=torch.bfloat16)
    weight = torch.randn(shape.weight_shape, device=device, dtype=torch.bfloat16)
    candidate_out = torch.empty(shape.qkv_shape, device=device, dtype=torch.bfloat16)

    def as_cute(tensor: torch.Tensor):
        return from_dlpack(tensor.unsqueeze(-1), assumed_align=16)

    reference = _load_reference()
    kernel = reference.Sm120GemmKernel(cutlass.Float32, tile)
    stream = cuda.CUstream(torch.cuda.current_stream(device).cuda_stream)
    max_clusters = utils.HardwareInfo().get_max_active_clusters(1)
    compiled = cute.compile(
        kernel,
        as_cute(x),
        as_cute(weight),
        as_cute(candidate_out),
        max_clusters,
        stream,
    )

    def baseline() -> torch.Tensor:
        return F.linear(x, weight)

    def candidate() -> torch.Tensor:
        compiled(as_cute(x), as_cute(weight), as_cute(candidate_out), stream)
        return candidate_out

    # Warm both paths before alternating their measurement order.  The large
    # GEMM can change clocks/power state; alternating avoids crediting one path
    # solely for running first.
    for _ in range(warmup):
        baseline()
        candidate()
    torch.cuda.synchronize(device)

    reference_out = baseline()
    candidate()
    torch.cuda.synchronize(device)
    exact = bool(torch.equal(reference_out, candidate_out))
    max_abs = float((reference_out.float() - candidate_out.float()).abs().max())
    del reference_out

    baseline_ms: list[float] = []
    candidate_ms: list[float] = []
    for index in range(iterations):
        if index % 2:
            candidate_ms.extend(_time_cuda(candidate, 1))
            baseline_ms.extend(_time_cuda(baseline, 1))
        else:
            baseline_ms.extend(_time_cuda(baseline, 1))
            candidate_ms.extend(_time_cuda(candidate, 1))

    baseline_summary = summarize(baseline_ms)
    candidate_summary = summarize(candidate_ms)
    raw_delta = baseline_summary["median_ms"] - candidate_summary["median_ms"]
    upper = measured_post_gemm_upper_bound_ms()
    projected_best_delta = raw_delta + upper["ideal_total_ms_per_layer"]
    return {
        "status": "raw-gemm-only",
        "device": {
            "name": props.name,
            "compute_capability": [props.major, props.minor],
        },
        "shape": asdict(shape),
        "tile_shape_mnk": list(tile),
        "bit_exact_raw_gemm": exact,
        "max_abs_raw_gemm": max_abs,
        "baseline": baseline_summary,
        "cute_raw_gemm": candidate_summary,
        "raw_gemm_delta_ms": raw_delta,
        "measured_post_gemm_upper_bound": upper,
        "projected_ideal_fused_delta_ms_per_layer": projected_best_delta,
        "passes_0p2ms_projection_gate": exact
        and projected_best_delta >= MIN_LAYER_WIN_MS,
        "warning": (
            "This candidate has no RMSNorm/RoPE/direct-landing epilogue; the "
            "projected fused delta is an optimistic upper bound, not a result."
        ),
    }


def check_report(shape: ShapeSpec) -> dict[str, Any]:
    shape.validate()
    upper = measured_post_gemm_upper_bound_ms()
    return {
        "status": "check-only",
        "shape": asdict(shape),
        "traffic_bytes": shape.traffic_model(),
        "measured_post_gemm_upper_bound": upper,
        "minimum_required_win_ms_per_layer": MIN_LAYER_WIN_MS,
        "upper_bound_exceeds_gate": upper["ideal_total_ms_per_layer"]
        >= MIN_LAYER_WIN_MS,
        "landing_epilogue_contract": LANDING_EPILOGUE_CONTRACT,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("check", "run"), nargs="?", default="check")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--tile", default="64,128,64")
    parser.add_argument("--json-out", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tile = tuple(int(value) for value in args.tile.split(","))
    if len(tile) != 3:
        raise ValueError("--tile must contain M,N,K")
    shape = ShapeSpec()
    report = (
        check_report(shape)
        if args.mode == "check"
        else run_raw_gemm(
            shape,
            device_name=args.device,
            warmup=args.warmup,
            iterations=args.iterations,
            tile=tile,
        )
    )
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.json_out is not None:
        args.json_out.write_text(rendered + "\n")


if __name__ == "__main__":
    main()
