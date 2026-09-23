#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Exact-shape MiniMax-H3 FC1 + SwiGLU benchmark for SM120.

The production BF16 baseline is deliberately narrow:

1. ``torch.nn.functional.linear(x, weight)`` produces a BF16 tensor whose
   first half is ``gate`` and second half is ``up``.
2. vLLM's ``torch.ops._C.silu_and_mul`` computes ``silu(gate) * up`` and
   stores BF16.

No bundled candidate is provided yet.  A candidate is accepted only through
the explicit contract below; a two-kernel wrapper must not claim that
contract.  ``check`` is the safe default and launches no GPU work.  Use
``run`` explicitly after reviewing the candidate and available devices.
"""

from __future__ import annotations

import argparse
import gc
import importlib
import importlib.metadata
import importlib.util
import json
import math
import os
import sys
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import torch
import torch.nn.functional as F


H3_LOCAL_ROWS = 11_936
H3_HIDDEN_SIZE = 5_376
H3_FFN_HIDDEN_SIZE = 14_336
H3_LAYERS = 50
H3_DENOISE_UPDATES = 49
EXPECTED_CUTLASS_DSL_VERSION = "4.6.2"

# Metadata is intentionally redundant.  It prevents an ordinary
# linear-then-activation wrapper from being accidentally reported as the
# pair-aware epilogue this benchmark is meant to evaluate.  Runtime bitwise
# comparison remains the real correctness gate.
REQUIRED_CANDIDATE_CONTRACT: dict[str, Any] = {
    "api_version": 1,
    "architecture": "sm120",
    "implementation": "cute_dsl",
    "single_kernel": True,
    "pair_aware_epilogue": True,
    "input_dtype": "bfloat16",
    "weight_dtype": "bfloat16",
    "output_dtype": "bfloat16",
    "weight_layout": "gate_rows_then_up_rows",
    "activation": "silu_gate_mul_up",
    "pre_activation_rounding": "gate_and_up_bfloat16",
    "post_silu_rounding": "bfloat16_before_up_multiply",
    "output_rounding": "bfloat16",
}


@dataclass(frozen=True)
class ShapeSpec:
    """Dense H3 FC1 problem on one TP1/SP8 rank."""

    m: int = H3_LOCAL_ROWS
    k: int = H3_HIDDEN_SIZE
    n_pair: int = H3_FFN_HIDDEN_SIZE

    def validate(self) -> None:
        for name, value in asdict(self).items():
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")

    @property
    def fc1_weight_shape(self) -> tuple[int, int]:
        return (2 * self.n_pair, self.k)

    @property
    def fc1_output_shape(self) -> tuple[int, int]:
        return (self.m, 2 * self.n_pair)

    @property
    def activation_shape(self) -> tuple[int, int]:
        return (self.m, self.n_pair)

    @property
    def fc2_weight_shape(self) -> tuple[int, int]:
        return (self.k, self.n_pair)

    @property
    def fc2_output_shape(self) -> tuple[int, int]:
        return (self.m, self.k)

    @property
    def is_champion_exact_shape(self) -> bool:
        return self == ShapeSpec()

    def byte_model(self, *, downstream_fc2: bool) -> dict[str, int]:
        element_bytes = 2
        values = {
            "input": self.m * self.k * element_bytes,
            "fc1_weight": 2 * self.n_pair * self.k * element_bytes,
            "fc1_materialized_output": self.m * 2 * self.n_pair * element_bytes,
            "activation_output": self.m * self.n_pair * element_bytes,
            # FC1 writes gate/up, the standalone activation reads gate/up and
            # writes its result.  A fused epilogue writes only the result.
            "epilogue_fusion_hbm_bytes_avoided": self.m * self.n_pair * 8,
        }
        if downstream_fc2:
            values.update(
                {
                    "fc2_weight": self.k * self.n_pair * element_bytes,
                    "fc2_output": self.m * self.k * element_bytes,
                }
            )
        return values

    def flop_model(self, *, downstream_fc2: bool) -> dict[str, int]:
        values = {"fc1": 2 * self.m * self.k * (2 * self.n_pair)}
        if downstream_fc2:
            values["fc2"] = 2 * self.m * self.n_pair * self.k
        return values

    def candidate_shape_dict(self) -> dict[str, Any]:
        return {
            "m": self.m,
            "k": self.k,
            "n_pair": self.n_pair,
            "input_shape": [self.m, self.k],
            "weight_shape": list(self.fc1_weight_shape),
            "output_shape": list(self.activation_shape),
            "dtype": "bfloat16",
        }


@dataclass
class Timing:
    samples_ms: list[float]
    median_ms: float
    p95_ms: float
    min_ms: float
    max_ms: float

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


class ProductionBaseline:
    """The exact unquantized MiniMax-H3 MLP prefix used by vLLM-Omni."""

    def __init__(self, weight: torch.Tensor) -> None:
        self.weight = weight
        try:
            from vllm_omni.diffusion.layers.activation import SiluAndMul
        except Exception as exc:  # pragma: no cover - depends on CUDA build
            raise RuntimeError(
                "Could not import vLLM-Omni SiluAndMul; run in the champion "
                "server environment with the vLLM CUDA extension available"
            ) from exc
        self.activation = SiluAndMul()
        if not hasattr(torch.ops._C, "silu_and_mul"):
            raise RuntimeError(
                "torch.ops._C.silu_and_mul is unavailable; refusing to replace "
                "the production reference with a different implementation"
            )

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        gate_up = F.linear(x, self.weight)
        if gate_up.dtype is not torch.bfloat16:
            raise RuntimeError(f"FC1 reference did not preserve BF16: {gate_up.dtype}")
        return self.activation(gate_up)


def reference_swiglu_native(gate_up: torch.Tensor) -> torch.Tensor:
    """CPU/test oracle for ordering only; never used as the GPU baseline."""
    if gate_up.shape[-1] % 2:
        raise ValueError("gate/up width must be even")
    gate, up = gate_up.chunk(2, dim=-1)
    return F.silu(gate) * up


def percentile_nearest_rank(values: list[float], percentile: float) -> float:
    if not values:
        raise ValueError("cannot compute a percentile of an empty sample")
    if not 0.0 < percentile <= 1.0:
        raise ValueError(f"percentile must be in (0, 1], got {percentile}")
    ordered = sorted(values)
    return ordered[max(0, math.ceil(percentile * len(ordered)) - 1)]


def summarize_times(values: list[float]) -> Timing:
    ordered = sorted(values)
    return Timing(
        samples_ms=values,
        median_ms=(
            ordered[len(ordered) // 2]
            if len(ordered) % 2
            else 0.5 * (ordered[len(ordered) // 2 - 1] + ordered[len(ordered) // 2])
        ),
        p95_ms=percentile_nearest_rank(values, 0.95),
        min_ms=ordered[0],
        max_ms=ordered[-1],
    )


def validate_candidate_contract(metadata: object) -> dict[str, Any]:
    if not isinstance(metadata, Mapping):
        raise TypeError("candidate CANDIDATE_CONTRACT must be a mapping")
    missing = sorted(set(REQUIRED_CANDIDATE_CONTRACT) - set(metadata))
    if missing:
        raise ValueError(f"candidate contract is missing fields: {missing}")
    mismatches = {
        key: {"required": required, "actual": metadata[key]}
        for key, required in REQUIRED_CANDIDATE_CONTRACT.items()
        if metadata[key] != required
    }
    if mismatches:
        raise ValueError(f"candidate contract mismatch: {mismatches}")
    name = metadata.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("candidate contract needs a nonempty 'name'")
    return dict(metadata)


def _load_candidate_module(reference: str) -> ModuleType:
    path = Path(reference).expanduser()
    if path.suffix == ".py" or path.exists():
        if not path.is_file():
            raise ValueError(f"candidate path is not a Python file: {path}")
        module_name = f"_h3_fc1_swiglu_candidate_{abs(hash(path.resolve()))}"
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"could not load candidate module from {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module
    return importlib.import_module(reference)


def load_candidate(
    reference: str,
    *,
    shape: ShapeSpec,
    weight: torch.Tensor,
    device: torch.device,
) -> tuple[Callable[[torch.Tensor], torch.Tensor], dict[str, Any]]:
    module = _load_candidate_module(reference)
    metadata = validate_candidate_contract(getattr(module, "CANDIDATE_CONTRACT", None))
    builder = getattr(module, "build_candidate", None)
    if not callable(builder):
        raise TypeError("candidate module must export callable build_candidate")
    runner = builder(
        weight=weight,
        shape=shape.candidate_shape_dict(),
        device=device,
    )
    if not callable(runner):
        raise TypeError("build_candidate must return callable runner(x) -> output")
    runtime_metadata = getattr(runner, "benchmark_metadata", None)
    if runtime_metadata is not None:
        if not isinstance(runtime_metadata, Mapping):
            raise TypeError("candidate runner benchmark_metadata must be a mapping")
        metadata["runtime"] = dict(runtime_metadata)
    return runner, metadata


def _find_flashinfer_source(override: Path | None) -> Path | None:
    candidates: list[Path] = []
    if override is not None:
        candidates.append(override.expanduser())
    env_source = os.environ.get("FLASHINFER_SOURCE")
    if env_source:
        candidates.append(Path(env_source).expanduser())
    script = Path(__file__).resolve()
    if len(script.parents) > 4:
        candidates.append(script.parents[4] / "fi120")
    for candidate in candidates:
        if (candidate / "flashinfer" / "fused_moe" / "cute_dsl").is_dir():
            return candidate.resolve()
    return None


def capability_report(
    *,
    device_name: str,
    flashinfer_source: Path | None,
    probe_device: bool,
) -> dict[str, Any]:
    try:
        cutlass_version = importlib.metadata.version("nvidia-cutlass-dsl")
    except importlib.metadata.PackageNotFoundError:
        cutlass_version = None
    source = _find_flashinfer_source(flashinfer_source)
    example_paths: list[str] = []
    try:
        flashinfer_distribution = importlib.metadata.distribution("flashinfer-python")
        dense_example = Path(
            flashinfer_distribution.locate_file(
                "flashinfer/data/cutlass/examples/python/CuTeDSL/"
                "blackwell_geforce/dense_gemm.py"
            )
        )
        if dense_example.is_file():
            example_paths.append(str(dense_example.resolve()))
    except importlib.metadata.PackageNotFoundError:
        pass
    if source is not None:
        for relative in (
            "flashinfer/fused_moe/cute_dsl/blackwell/blockscaled_contiguous_gather_grouped_gemm_act_fusion.py",
            "flashinfer/fused_moe/cute_dsl/blackwell_sm12x/moe_micro_kernel.py",
            "flashinfer/fused_moe/cute_dsl/blackwell_sm12x/moe_static_kernel.py",
        ):
            if (source / relative).is_file():
                example_paths.append(str(source / relative))

    bundled_candidate = Path(__file__).with_name(
        "h3_fc1_swiglu_sm120_candidate.py"
    ).is_file()
    report: dict[str, Any] = {
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cutlass_python_importable": importlib.util.find_spec("cutlass") is not None,
        "nvidia_cutlass_dsl_version": cutlass_version,
        "expected_nvidia_cutlass_dsl_version": EXPECTED_CUTLASS_DSL_VERSION,
        "cutlass_dsl_version_matches": cutlass_version == EXPECTED_CUTLASS_DSL_VERSION,
        "flashinfer_source": str(source) if source is not None else None,
        "relevant_reference_examples": example_paths,
        "bundled_candidate": bundled_candidate,
        "candidate_status": (
            "experimental_isolated_not_in_production_dispatch"
            if bundled_candidate
            else "not_available"
        ),
        "device_probe_requested": probe_device,
    }
    if not probe_device:
        report["device"] = "not_probed_no_cuda_context_requested"
        return report

    available = torch.cuda.is_available()
    report["cuda_available"] = available
    if available:
        device = torch.device(device_name)
        props = torch.cuda.get_device_properties(device)
        report["device"] = {
            "name": props.name,
            "compute_capability": [props.major, props.minor],
            "total_memory_bytes": props.total_memory,
            "is_sm120": (props.major, props.minor) == (12, 0),
        }
    else:
        report["device"] = None
    return report


def _assert_tensor_contract(
    output: torch.Tensor,
    *,
    shape: tuple[int, int],
    device: torch.device,
    label: str,
) -> None:
    if output.shape != shape:
        raise ValueError(f"{label} output shape {tuple(output.shape)} != {shape}")
    if output.dtype is not torch.bfloat16:
        raise ValueError(f"{label} output dtype {output.dtype} != torch.bfloat16")
    if output.device != device:
        raise ValueError(f"{label} output device {output.device} != {device}")
    if not output.is_contiguous():
        raise ValueError(f"{label} output must be contiguous")


def compare_outputs(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, Any]:
    _assert_tensor_contract(
        candidate,
        shape=tuple(reference.shape),
        device=reference.device,
        label="candidate",
    )
    equal = torch.equal(reference, candidate)
    ref_float = reference.float()
    candidate_float = candidate.float()
    abs_diff = (ref_float - candidate_float).abs()
    denominator = ref_float.abs().clamp_min(torch.finfo(torch.float32).tiny)
    return {
        "torch_equal": equal,
        "max_abs": float(abs_diff.max().item()),
        "mean_abs": float(abs_diff.mean().item()),
        "max_rel": float((abs_diff / denominator).max().item()),
        "reference_nan_count": int(torch.isnan(ref_float).sum().item()),
        "candidate_nan_count": int(torch.isnan(candidate_float).sum().item()),
    }


def _warmup_pair(
    baseline: Callable[[], torch.Tensor],
    candidate: Callable[[], torch.Tensor] | None,
    warmup: int,
) -> None:
    for iteration in range(warmup):
        if candidate is None:
            baseline()
        elif iteration % 2:
            candidate()
            baseline()
        else:
            baseline()
            candidate()
    torch.accelerator.synchronize()


def _time_interleaved(
    baseline: Callable[[], torch.Tensor],
    candidate: Callable[[], torch.Tensor] | None,
    *,
    iterations: int,
) -> tuple[Timing, Timing | None]:
    events: dict[str, list[tuple[torch.cuda.Event, torch.cuda.Event]]] = {
        "baseline": [],
        "candidate": [],
    }
    last_outputs: dict[str, torch.Tensor] = {}
    for iteration in range(iterations):
        order = ("baseline",) if candidate is None else (
            ("baseline", "candidate")
            if iteration % 2 == 0
            else ("candidate", "baseline")
        )
        for label in order:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            last_outputs[label] = baseline() if label == "baseline" else candidate()  # type: ignore[misc]
            end.record()
            events[label].append((start, end))
    torch.accelerator.synchronize()
    # Keep the final outputs alive through synchronization, then consume a
    # scalar so the benchmark's dataflow is explicit.
    for output in last_outputs.values():
        _ = output.view(-1)[0].item()
    samples = {
        label: [start.elapsed_time(end) for start, end in pairs]
        for label, pairs in events.items()
    }
    return (
        summarize_times(samples["baseline"]),
        summarize_times(samples["candidate"]) if samples["candidate"] else None,
    )


def _measure_peak_extra_bytes(fn: Callable[[], torch.Tensor]) -> dict[str, int]:
    torch.accelerator.synchronize()
    gc.collect()
    before_allocated = torch.accelerator.memory_allocated()
    before_reserved = torch.accelerator.memory_reserved()
    torch.accelerator.reset_peak_memory_stats()
    output = fn()
    torch.accelerator.synchronize()
    result = {
        "allocated_before": before_allocated,
        "reserved_before": before_reserved,
        "peak_allocated": torch.accelerator.max_memory_allocated(),
        "peak_reserved": torch.accelerator.max_memory_reserved(),
        "peak_extra_allocated": max(
            0, torch.accelerator.max_memory_allocated() - before_allocated
        ),
        "output_bytes": output.numel() * output.element_size(),
    }
    del output
    torch.accelerator.synchronize()
    return result


def _make_fixture(
    shape: ShapeSpec,
    *,
    device: torch.device,
    seed: int,
    input_scale: float,
    weight_scale: float,
    downstream_fc2: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    generator = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn(
        (shape.m, shape.k),
        device=device,
        dtype=torch.bfloat16,
        generator=generator,
    ).mul_(input_scale)
    fc1_weight = torch.randn(
        shape.fc1_weight_shape,
        device=device,
        dtype=torch.bfloat16,
        generator=generator,
    ).mul_(weight_scale)
    fc2_weight = None
    if downstream_fc2:
        fc2_weight = torch.randn(
            shape.fc2_weight_shape,
            device=device,
            dtype=torch.bfloat16,
            generator=generator,
        ).mul_(weight_scale)
    return x, fc1_weight, fc2_weight


def _write_report(path: Path | None, report: dict[str, Any]) -> None:
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if path is not None:
        path = path.expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".tmp")
        temporary.write_text(encoded, encoding="utf-8")
        temporary.replace(path)
    print("H3_FC1_SWIGLU_BENCHMARK=" + json.dumps(report, sort_keys=True))


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action",
        choices=("check", "run"),
        nargs="?",
        default="check",
        help="check is non-GPU preflight; run explicitly launches CUDA work",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--m", type=int, default=H3_LOCAL_ROWS)
    parser.add_argument("--k", type=int, default=H3_HIDDEN_SIZE)
    parser.add_argument("--n-pair", type=int, default=H3_FFN_HIDDEN_SIZE)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1101)
    parser.add_argument("--input-scale", type=float, default=0.02)
    parser.add_argument("--weight-scale", type=float, default=0.02)
    parser.add_argument("--downstream-fc2", action="store_true")
    parser.add_argument(
        "--candidate-module",
        help="import name or .py file implementing the reviewed candidate API",
    )
    parser.add_argument("--flashinfer-source", type=Path)
    parser.add_argument(
        "--probe-device",
        action="store_true",
        help="with check, initialize the CUDA driver only to report device properties",
    )
    parser.add_argument("--json-out", type=Path)
    return parser.parse_args(argv)


def _validate_args(args: argparse.Namespace, shape: ShapeSpec) -> None:
    shape.validate()
    if args.warmup < 0:
        raise ValueError("warmup must be nonnegative")
    if args.iterations <= 0:
        raise ValueError("iterations must be positive")
    if not math.isfinite(args.input_scale) or args.input_scale <= 0:
        raise ValueError("input-scale must be finite and positive")
    if not math.isfinite(args.weight_scale) or args.weight_scale <= 0:
        raise ValueError("weight-scale must be finite and positive")
    if args.action == "check" and args.candidate_module:
        # Loading a CuTe module may JIT or initialize CUDA as an import side
        # effect.  Keep check genuinely GPU-idle.
        raise ValueError("candidate modules are loaded only by the explicit run action")


def run_benchmark(args: argparse.Namespace, shape: ShapeSpec) -> tuple[dict[str, Any], int]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    props = torch.cuda.get_device_properties(device)
    if (props.major, props.minor) != (12, 0):
        raise RuntimeError(
            f"this benchmark is gated to SM120, got SM{props.major}{props.minor} ({props.name})"
        )

    capabilities = capability_report(
        device_name=args.device,
        flashinfer_source=args.flashinfer_source,
        probe_device=True,
    )
    initial_allocated = torch.accelerator.memory_allocated(device)
    x, fc1_weight, fc2_weight = _make_fixture(
        shape,
        device=device,
        seed=args.seed,
        input_scale=args.input_scale,
        weight_scale=args.weight_scale,
        downstream_fc2=args.downstream_fc2,
    )
    torch.accelerator.synchronize(device)
    fixture_allocated = torch.accelerator.memory_allocated(device) - initial_allocated
    baseline = ProductionBaseline(fc1_weight)

    candidate: Callable[[torch.Tensor], torch.Tensor] | None = None
    candidate_metadata: dict[str, Any] | None = None
    candidate_build_memory: dict[str, int] | None = None
    if args.candidate_module:
        before_build = torch.accelerator.memory_allocated(device)
        torch.accelerator.reset_peak_memory_stats(device)
        candidate, candidate_metadata = load_candidate(
            args.candidate_module,
            shape=shape,
            weight=fc1_weight,
            device=device,
        )
        torch.accelerator.synchronize(device)
        after_build = torch.accelerator.memory_allocated(device)
        candidate_build_memory = {
            "persistent_allocated": max(0, after_build - before_build),
            "peak_extra_allocated": max(
                0, torch.accelerator.max_memory_allocated(device) - before_build
            ),
        }

    baseline_output = baseline(x)
    _assert_tensor_contract(
        baseline_output,
        shape=shape.activation_shape,
        device=device,
        label="baseline",
    )
    correctness: dict[str, Any] | None = None
    downstream_correctness: dict[str, Any] | None = None
    if candidate is not None:
        candidate_output = candidate(x)
        correctness = compare_outputs(baseline_output, candidate_output)
        if fc2_weight is not None:
            baseline_fc2 = F.linear(baseline_output, fc2_weight)
            candidate_fc2 = F.linear(candidate_output, fc2_weight)
            downstream_correctness = compare_outputs(baseline_fc2, candidate_fc2)
        del candidate_output
    del baseline_output
    torch.accelerator.synchronize(device)

    if correctness is not None and not correctness["torch_equal"]:
        report = {
            "status": "candidate_rejected_not_bit_exact",
            "timings_run": False,
            "shape": shape.candidate_shape_dict(),
            "is_champion_exact_shape": shape.is_champion_exact_shape,
            "candidate": candidate_metadata,
            "correctness": correctness,
            "downstream_correctness": downstream_correctness,
            "capabilities": capabilities,
        }
        return report, 3

    baseline_call = lambda: baseline(x)
    candidate_call = None if candidate is None else lambda: candidate(x)
    _warmup_pair(baseline_call, candidate_call, args.warmup)
    baseline_timing, candidate_timing = _time_interleaved(
        baseline_call,
        candidate_call,
        iterations=args.iterations,
    )
    memory = {"baseline_fc1_swiglu": _measure_peak_extra_bytes(baseline_call)}
    if candidate_call is not None:
        memory["candidate_fc1_swiglu"] = _measure_peak_extra_bytes(candidate_call)

    downstream: dict[str, Any] | None = None
    if fc2_weight is not None:
        baseline_mlp = lambda: F.linear(baseline(x), fc2_weight)
        candidate_mlp = (
            None
            if candidate is None
            else lambda: F.linear(candidate(x), fc2_weight)
        )
        _warmup_pair(baseline_mlp, candidate_mlp, args.warmup)
        baseline_mlp_timing, candidate_mlp_timing = _time_interleaved(
            baseline_mlp,
            candidate_mlp,
            iterations=args.iterations,
        )
        downstream = {
            "baseline": baseline_mlp_timing.to_json(),
            "candidate": (
                candidate_mlp_timing.to_json()
                if candidate_mlp_timing is not None
                else None
            ),
        }
        memory["baseline_full_mlp"] = _measure_peak_extra_bytes(baseline_mlp)
        if candidate_mlp is not None:
            memory["candidate_full_mlp"] = _measure_peak_extra_bytes(candidate_mlp)

    projection: dict[str, float] | None = None
    if candidate_timing is not None:
        saved_ms_per_layer = baseline_timing.median_ms - candidate_timing.median_ms
        projection = {
            "saved_ms_per_layer": saved_ms_per_layer,
            "saved_ms_per_denoise_step_50_layers": saved_ms_per_layer * H3_LAYERS,
            "saved_s_per_49_update_request": (
                saved_ms_per_layer * H3_LAYERS * H3_DENOISE_UPDATES / 1_000.0
            ),
            "median_speedup": baseline_timing.median_ms / candidate_timing.median_ms,
            "integration_gate_ms_per_layer": 0.2,
            "passes_speed_gate": saved_ms_per_layer >= 0.2,
        }

    report = {
        "status": "candidate_measured_bit_exact" if candidate else "baseline_only",
        "timings_run": True,
        "shape": shape.candidate_shape_dict(),
        "is_champion_exact_shape": shape.is_champion_exact_shape,
        "dtype": "bfloat16",
        "weight_partition_order": ["gate", "up"],
        "activation_order": "silu(gate) * up",
        "reference": "torch.nn.functional.linear + torch.ops._C.silu_and_mul",
        "warmup": args.warmup,
        "iterations": args.iterations,
        "seed": args.seed,
        "input_scale": args.input_scale,
        "weight_scale": args.weight_scale,
        "candidate": candidate_metadata,
        "correctness": correctness,
        "downstream_correctness": downstream_correctness,
        "fc1_swiglu_timing": {
            "baseline": baseline_timing.to_json(),
            "candidate": candidate_timing.to_json() if candidate_timing else None,
        },
        "full_mlp_timing": downstream,
        "projection": projection,
        "memory": {
            "fixture_allocated_bytes": fixture_allocated,
            "candidate_build": candidate_build_memory,
            "operations": memory,
            "analytical_bytes": shape.byte_model(
                downstream_fc2=args.downstream_fc2
            ),
        },
        "analytical_flops": shape.flop_model(
            downstream_fc2=args.downstream_fc2
        ),
        "capabilities": capabilities,
    }
    return report, 0


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    shape = ShapeSpec(m=args.m, k=args.k, n_pair=args.n_pair)
    _validate_args(args, shape)
    if args.action == "check":
        report = {
            "status": "check_only_no_gpu_work_launched",
            "timings_run": False,
            "shape": shape.candidate_shape_dict(),
            "is_champion_exact_shape": shape.is_champion_exact_shape,
            "warmup": args.warmup,
            "iterations": args.iterations,
            "downstream_fc2": args.downstream_fc2,
            "analytical_bytes": shape.byte_model(
                downstream_fc2=args.downstream_fc2
            ),
            "analytical_flops": shape.flop_model(
                downstream_fc2=args.downstream_fc2
            ),
            "required_candidate_contract": REQUIRED_CANDIDATE_CONTRACT,
            "capabilities": capability_report(
                device_name=args.device,
                flashinfer_source=args.flashinfer_source,
                probe_device=args.probe_device,
            ),
        }
        _write_report(args.json_out, report)
        return 0

    report, status = run_benchmark(args, shape)
    _write_report(args.json_out, report)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
