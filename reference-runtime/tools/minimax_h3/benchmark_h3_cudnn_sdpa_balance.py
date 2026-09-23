#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Measure device-to-device cuDNN SDPA balance at the H3 SP8 shape.

This is deliberately *not* a distributed benchmark.  One spawned process owns
one CUDA device and no process initializes torch.distributed or NCCL.  A host
multiprocessing Event only releases the independent workers at roughly the
same time so that all eight GPUs see the same concurrent load.

The measured call mirrors ``CuDNNAttentionImpl.forward_cuda`` for mask-free
H3 attention: contiguous BSHD inputs are permuted to BHSD and PyTorch SDPA is
pinned to ``SDPBackend.CUDNN_ATTENTION``.  CUDA events enclose only the GPU
call.  Optional nvidia-smi sampling is diagnostic context, not part of the
event interval.
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import queue
import statistics
import subprocess
import threading
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence


@dataclass(frozen=True)
class H3AttentionShape:
    batch: int = 1
    sequence: int = 95_488
    heads: int = 7
    head_dim: int = 128

    def validate(self) -> "H3AttentionShape":
        for name, value in asdict(self).items():
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        return self

    @property
    def tensor_shape(self) -> tuple[int, int, int, int]:
        return (self.batch, self.sequence, self.heads, self.head_dim)

    @property
    def tensor_bytes_bf16(self) -> int:
        return math.prod(self.tensor_shape) * 2

    @property
    def softmax_scale(self) -> float:
        return 1.0 / math.sqrt(self.head_dim)

    @property
    def is_exact_h3_sp8(self) -> bool:
        return self.tensor_shape == (1, 95_488, 7, 128)


@dataclass(frozen=True)
class TimingSummary:
    count: int
    median_ms: float
    p95_ms: float
    minimum_ms: float
    maximum_ms: float


def percentile_nearest_rank(values: Sequence[float], quantile: float) -> float:
    if not values:
        raise ValueError("at least one timing sample is required")
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must be in [0, 1]")
    ordered = sorted(float(value) for value in values)
    rank = max(1, math.ceil(quantile * len(ordered)))
    return ordered[rank - 1]


def summarize_timings(values: Sequence[float]) -> TimingSummary:
    if not values:
        raise ValueError("at least one timing sample is required")
    numeric = [float(value) for value in values]
    return TimingSummary(
        count=len(numeric),
        median_ms=statistics.median(numeric),
        p95_ms=percentile_nearest_rank(numeric, 0.95),
        minimum_ms=min(numeric),
        maximum_ms=max(numeric),
    )


def parse_visible_device_order(value: str | None, device_count: int) -> list[str]:
    """Return logical-to-physical identifiers without reordering them."""
    if value is None or not value.strip():
        return [str(index) for index in range(device_count)]
    devices = [item.strip() for item in value.split(",") if item.strip()]
    if len(devices) < device_count:
        raise ValueError(
            f"CUDA_VISIBLE_DEVICES exposes {len(devices)} devices, "
            f"but {device_count} workers were requested"
        )
    return devices[:device_count]


_SMI_FIELDS = (
    "index",
    "pci.bus_id",
    "name",
    "pstate",
    "clocks.current.sm",
    "clocks.current.memory",
    "power.draw",
    "temperature.gpu",
    "utilization.gpu",
    "memory.used",
)


def _parse_optional_number(value: str) -> float | None:
    stripped = value.strip()
    if stripped in {"", "N/A", "[N/A]", "Not Supported"}:
        return None
    try:
        return float(stripped)
    except ValueError:
        return None


def parse_nvidia_smi_row(line: str) -> dict[str, Any]:
    parts = [part.strip() for part in line.split(",")]
    if len(parts) != len(_SMI_FIELDS):
        raise ValueError(
            f"expected {len(_SMI_FIELDS)} nvidia-smi columns, got {len(parts)}"
        )
    numeric_fields = {
        "clocks.current.sm",
        "clocks.current.memory",
        "power.draw",
        "temperature.gpu",
        "utilization.gpu",
        "memory.used",
    }
    result: dict[str, Any] = {}
    for field, value in zip(_SMI_FIELDS, parts, strict=True):
        result[field] = (
            _parse_optional_number(value) if field in numeric_fields else value
        )
    return result


def query_nvidia_smi() -> list[dict[str, Any]]:
    command = [
        "nvidia-smi",
        "--query-gpu=" + ",".join(_SMI_FIELDS),
        "--format=csv,noheader,nounits",
    ]
    completed = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    return [
        parse_nvidia_smi_row(line)
        for line in completed.stdout.splitlines()
        if line.strip()
    ]


def summarize_telemetry(
    samples: Sequence[dict[str, Any]], physical_id: str
) -> dict[str, Any]:
    rows = [
        row
        for sample in samples
        for row in sample.get("gpus", [])
        if str(row.get("index")) == str(physical_id)
    ]
    result: dict[str, Any] = {"sample_count": len(rows)}
    for source, label in (
        ("clocks.current.sm", "sm_clock_mhz"),
        ("clocks.current.memory", "memory_clock_mhz"),
        ("power.draw", "power_w"),
        ("temperature.gpu", "temperature_c"),
        ("utilization.gpu", "utilization_percent"),
        ("memory.used", "memory_used_mib"),
    ):
        values = [float(row[source]) for row in rows if row.get(source) is not None]
        if values:
            result[label] = {
                "minimum": min(values),
                "median": statistics.median(values),
                "maximum": max(values),
            }
    if rows:
        result["pci_bus_id"] = rows[0].get("pci.bus_id")
        result["name"] = rows[0].get("name")
        result["pstates"] = sorted({str(row.get("pstate")) for row in rows})
    return result


class NvidiaSmiSampler:
    def __init__(self, interval_seconds: float) -> None:
        self.interval_seconds = interval_seconds
        self.samples: list[dict[str, Any]] = []
        self.errors: list[str] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        def sample_loop() -> None:
            while not self._stop.is_set():
                started = time.time()
                try:
                    rows = query_nvidia_smi()
                    self.samples.append({"unix_time": started, "gpus": rows})
                except Exception as exc:  # diagnostic only
                    self.errors.append(f"{type(exc).__name__}: {exc}")
                self._stop.wait(self.interval_seconds)

        self._thread = threading.Thread(target=sample_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(2.0, self.interval_seconds * 2.0))


def _worker(
    logical_rank: int,
    shape_dict: dict[str, int],
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

        shape = H3AttentionShape(**shape_dict).validate()
        torch.cuda.set_device(logical_rank)
        device = torch.device("cuda", logical_rank)
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)

        tensors = []
        for _ in range(3):
            tensor = torch.empty(shape.tensor_shape, dtype=torch.bfloat16, device=device)
            tensor.normal_(generator=generator)
            tensors.append(tensor)
        query, key, value = tensors
        q_bhsd, k_bhsd, v_bhsd = (
            tensor.permute(0, 2, 1, 3) for tensor in (query, key, value)
        )

        def attention() -> "torch.Tensor":
            with sdpa_kernel([SDPBackend.CUDNN_ATTENTION]):
                return F.scaled_dot_product_attention(
                    q_bhsd,
                    k_bhsd,
                    v_bhsd,
                    attn_mask=None,
                    dropout_p=0.0,
                    is_causal=False,
                    scale=shape.softmax_scale,
                    enable_gqa=False,
                )

        output = None
        for _ in range(warmup):
            output = attention()
        torch.cuda.synchronize(device)
        messages.put(
            {
                "kind": "ready",
                "logical_rank": logical_rank,
                "device_name": torch.cuda.get_device_name(device),
                "torch_version": torch.__version__,
                "cudnn_version": torch.backends.cudnn.version(),
            }
        )
        if not start_gate.wait(timeout=180.0):
            raise TimeoutError("timed out waiting for the host start gate")

        samples_ms: list[float] = []
        for _round in range(rounds):
            for _iteration in range(iterations):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                output = attention()
                end.record()
                end.synchronize()
                samples_ms.append(float(start.elapsed_time(end)))

        assert output is not None
        checksum = float(output[0, 0, 0, 0].float().item())
        allocated = int(torch.cuda.memory_allocated(device))
        reserved = int(torch.cuda.memory_reserved(device))
        messages.put(
            {
                "kind": "result",
                "status": "ok",
                "logical_rank": logical_rank,
                "samples_ms": samples_ms,
                "checksum": checksum,
                "memory_allocated_bytes": allocated,
                "memory_reserved_bytes": reserved,
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


def _physical_inventory(rows: Sequence[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row["index"]): dict(row) for row in rows}


def build_balance_summary(results: Sequence[dict[str, Any]]) -> dict[str, Any]:
    successful = [result for result in results if result.get("status") == "ok"]
    if not successful:
        return {"successful_workers": 0}
    medians = {
        int(result["logical_rank"]): float(result["timing"]["median_ms"])
        for result in successful
    }
    slowest_rank = max(medians, key=medians.get)
    fastest_rank = min(medians, key=medians.get)
    return {
        "successful_workers": len(successful),
        "slowest_logical_rank": slowest_rank,
        "fastest_logical_rank": fastest_rank,
        "slowest_median_ms": medians[slowest_rank],
        "fastest_median_ms": medians[fastest_rank],
        "max_over_min_median": medians[slowest_rank] / medians[fastest_rank],
        "median_spread_percent": 100.0
        * (medians[slowest_rank] - medians[fastest_rank])
        / medians[fastest_rank],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--devices", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1101)
    parser.add_argument("--sequence", type=int, default=95_488)
    parser.add_argument("--heads", type=int, default=7)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--worker-timeout", type=float, default=300.0)
    parser.add_argument("--smi-interval", type=float, default=0.5)
    parser.add_argument("--no-smi-monitor", action="store_true")
    parser.add_argument("--json-out", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.devices <= 0 or args.warmup < 0 or args.rounds <= 0 or args.iterations <= 0:
        raise ValueError("devices/rounds/iterations must be positive and warmup nonnegative")
    shape = H3AttentionShape(
        sequence=args.sequence, heads=args.heads, head_dim=args.head_dim
    ).validate()
    visible = parse_visible_device_order(
        os.environ.get("CUDA_VISIBLE_DEVICES"), args.devices
    )

    try:
        static_smi = query_nvidia_smi()
    except Exception as exc:
        static_smi = []
        static_smi_error = f"{type(exc).__name__}: {exc}"
    else:
        static_smi_error = None
    inventory = _physical_inventory(static_smi)

    context = mp.get_context("spawn")
    start_gate = context.Event()
    messages = context.Queue()
    workers = [
        context.Process(
            target=_worker,
            args=(
                logical_rank,
                asdict(shape),
                args.warmup,
                args.rounds,
                args.iterations,
                args.seed,
                start_gate,
                messages,
            ),
            name=f"h3-cudnn-balance-rank{logical_rank}",
        )
        for logical_rank in range(args.devices)
    ]
    for worker in workers:
        worker.start()

    ready: dict[int, dict[str, Any]] = {}
    results_by_rank: dict[int, dict[str, Any]] = {}
    deadline = time.monotonic() + args.worker_timeout
    while len(ready) + len(results_by_rank) < args.devices:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        try:
            message = messages.get(timeout=min(1.0, remaining))
        except queue.Empty:
            if not any(worker.is_alive() for worker in workers):
                break
            continue
        rank = int(message["logical_rank"])
        if message["kind"] == "ready":
            ready[rank] = message
        else:
            results_by_rank[rank] = message

    sampler = NvidiaSmiSampler(args.smi_interval)
    if not args.no_smi_monitor:
        sampler.start()
    start_gate.set()

    deadline = time.monotonic() + args.worker_timeout
    while len(results_by_rank) < args.devices:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        try:
            message = messages.get(timeout=min(1.0, remaining))
        except queue.Empty:
            if not any(worker.is_alive() for worker in workers):
                break
            continue
        if message["kind"] == "result":
            results_by_rank[int(message["logical_rank"])] = message

    sampler.stop()
    for worker in workers:
        worker.join(timeout=5.0)
    for worker in workers:
        if worker.is_alive():
            worker.terminate()
            worker.join(timeout=5.0)

    results: list[dict[str, Any]] = []
    for logical_rank in range(args.devices):
        result = results_by_rank.get(
            logical_rank,
            {
                "kind": "result",
                "status": "timeout",
                "logical_rank": logical_rank,
                "error": "worker did not report a result",
            },
        )
        physical_id = visible[logical_rank]
        result["physical_id"] = physical_id
        if physical_id in inventory:
            result["pci_bus_id"] = inventory[physical_id].get("pci.bus_id")
            result["gpu_name"] = inventory[physical_id].get("name")
        if result.get("status") == "ok":
            result["timing"] = asdict(summarize_timings(result.pop("samples_ms")))
        if sampler.samples and physical_id.isdigit():
            result["telemetry"] = summarize_telemetry(
                sampler.samples, physical_id
            )
        results.append(result)

    payload = {
        "contract": {
            "independent_workers": True,
            "torch_distributed_initialized": False,
            "gpu_collectives": False,
            "host_start_gate_only": True,
            "input_layout": "contiguous BSHD",
            "sdpa_layout": "BHSD view via permute(0,2,1,3)",
            "backend": "SDPBackend.CUDNN_ATTENTION",
            "attention_mask": None,
            "is_causal": False,
            "dropout_p": 0.0,
        },
        "shape": {
            **asdict(shape),
            "tensor_shape": list(shape.tensor_shape),
            "tensor_bytes_bf16": shape.tensor_bytes_bf16,
            "softmax_scale": shape.softmax_scale,
            "is_exact_h3_sp8": shape.is_exact_h3_sp8,
        },
        "configuration": {
            "visible_device_order": visible,
            "warmup": args.warmup,
            "rounds": args.rounds,
            "iterations": args.iterations,
            "samples_per_gpu": args.rounds * args.iterations,
            "seed": args.seed,
            "smi_interval_seconds": None
            if args.no_smi_monitor
            else args.smi_interval,
        },
        "static_nvidia_smi_error": static_smi_error,
        "sampler_errors": sampler.errors,
        "results": results,
    }
    payload["balance"] = build_balance_summary(results)

    print(json.dumps(payload, indent=2, sort_keys=True))
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    return 0 if all(result.get("status") == "ok" for result in results) else 2


if __name__ == "__main__":
    raise SystemExit(main())
