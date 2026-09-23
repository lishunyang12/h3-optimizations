#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Screen FlashInfer SM120 CuTeDSL VSA skip-softmax at the H3 shape.

This is a deterministic, single-GPU operator screen.  It calls the combined
FlashInfer ``bsa_attn_sm120_blk64_fwd`` entry directly with the exact FastH3
15-second SP8 per-rank geometry:

* 1,641 logical tile-64 blocks (21 prefix + 1,620 target-video);
* seven local attention heads with head dimension 128; and
* 162 selected video blocks, plus all 21 exempt prefix blocks.

Threshold zero is the exact VSA kernel specialization.  Positive thresholds
are approximate and are compared with a separately retained threshold-zero
output.  This synthetic-QKV screen is not a video-quality qualification.

The tool never sets clocks, application clocks, or power limits.  It requires
the independent balance benchmark to be explicitly disabled and labels every
result ``node-default/unlocked clocks``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import math
import os
import statistics
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = REPO_ROOT.parents[1]
DEFAULT_FLASHINFER_ROOT = WORKSPACE_ROOT / "fi-pr4944-pr4876-combined"
DEFAULT_RESULTS_ROOT = WORKSPACE_ROOT / "results" / "profile"

BALANCE_GATE_ENV = "VLLM_OMNI_GPU_BALANCE_HEALTH_GATE"
PRODUCTION_THRESHOLD_ENV = "VLLM_OMNI_FASTVIDEO_VSA_SKIP_SOFTMAX_THRESHOLD_SCALE_FACTOR"
GPU_CLOCK_POLICY = "node-default/unlocked clocks"
GPU_CLOCK_POLICY_ID = "node_default_dynamic_unlocked"

BATCH = 1
PREFIX_SEGMENTS = (121, 1206)
VIDEO_SHAPE = (107, 22, 40)
BLOCK_SIZE = 64
PREFIX_BLOCKS = 21
VIDEO_BLOCKS = 1620
LOGICAL_BLOCKS = 1641
TILED_ROWS = LOGICAL_BLOCKS * BLOCK_SIZE
VALID_ROWS = sum(PREFIX_SEGMENTS) + math.prod(VIDEO_SHAPE)
SP8_ALIGNED_ROWS = 95_488
LOCAL_HEADS = 7
HEAD_DIM = 128
VIDEO_TOPK = 162
VIDEO_ACTIVE_KV_BLOCKS = PREFIX_BLOCKS + VIDEO_TOPK
SOFTMAX_SCALE = HEAD_DIM**-0.5
DTYPE = torch.bfloat16
REQUIRED_THRESHOLDS = (0.0, 1e-3, 1e-2, 0.1, 0.25, 0.5, 1.0)

_GPU_QUERY_FIELDS = (
    "index",
    "uuid",
    "name",
    "pci.bus_id",
    "driver_version",
    "pstate",
    "memory.used",
    "memory.total",
    "utilization.gpu",
    "temperature.gpu",
    "clocks.current.sm",
    "clocks.max.sm",
    "power.draw",
    "gpu.power.current_limit",
    "gpu.power.requested_limit",
    "gpu.power.default_limit",
    "clocks_event_reasons.applications_clocks_setting",
    "clocks_event_reasons.hw_slowdown",
    "clocks_event_reasons.hw_thermal_slowdown",
    "clocks_event_reasons.hw_power_brake_slowdown",
    "clocks_event_reasons.sw_thermal_slowdown",
    "ecc.errors.uncorrected.volatile.total",
)

_NUMERIC_GPU_FIELDS = {
    "memory.used",
    "memory.total",
    "utilization.gpu",
    "temperature.gpu",
    "clocks.current.sm",
    "clocks.max.sm",
    "power.draw",
    "gpu.power.current_limit",
    "gpu.power.requested_limit",
    "gpu.power.default_limit",
    "ecc.errors.uncorrected.volatile.total",
}

_HARD_THROTTLE_FIELDS = (
    "clocks_event_reasons.hw_slowdown",
    "clocks_event_reasons.hw_thermal_slowdown",
    "clocks_event_reasons.hw_power_brake_slowdown",
    "clocks_event_reasons.sw_thermal_slowdown",
)


class QualificationError(RuntimeError):
    """A fail-closed screening-contract violation."""


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flashinfer-root", type=Path, default=DEFAULT_FLASHINFER_ROOT)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--physical-gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1101)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument(
        "--extra-threshold",
        type=float,
        action="append",
        default=[],
        help="additional finite non-negative threshold (the required sweep is always retained)",
    )
    parser.add_argument("--metric-chunk-rows", type=int, default=4096)
    parser.add_argument("--hash-chunk-elements", type=int, default=4 * 1024 * 1024)
    parser.add_argument("--idle-samples", type=int, default=3)
    parser.add_argument("--idle-interval-seconds", type=float, default=1.0)
    parser.add_argument("--idle-max-memory-mib", type=float, default=256.0)
    parser.add_argument("--idle-max-utilization-percent", type=float, default=5.0)
    parser.add_argument("--idle-max-temperature-c", type=float, default=70.0)
    parser.add_argument("--power-limit-tolerance-w", type=float, default=1.0)
    parser.add_argument("--nvidia-smi", default="nvidia-smi")
    args = parser.parse_args(argv)

    if args.physical_gpu < 0:
        parser.error("--physical-gpu must be non-negative")
    if args.warmups < 0:
        parser.error("--warmups must be non-negative")
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    if args.metric_chunk_rows < 1 or args.hash_chunk_elements < 1:
        parser.error("metric/hash chunk sizes must be positive")
    if args.idle_samples < 1:
        parser.error("--idle-samples must be positive")
    if args.idle_interval_seconds < 0:
        parser.error("--idle-interval-seconds must be non-negative")
    idle_limits = (
        args.idle_max_memory_mib,
        args.idle_max_utilization_percent,
        args.idle_max_temperature_c,
        args.power_limit_tolerance_w,
    )
    if any(not math.isfinite(value) or value < 0 for value in idle_limits):
        parser.error("idle thresholds must be finite and non-negative")
    if any(not math.isfinite(value) or value < 0 for value in args.extra_threshold):
        parser.error("--extra-threshold must be finite and non-negative")
    args.thresholds = normalized_thresholds(args.extra_threshold)
    return args


def normalized_thresholds(extra: Sequence[float] = ()) -> tuple[float, ...]:
    values = {*REQUIRED_THRESHOLDS}
    for raw in extra:
        value = float(raw)
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"threshold must be finite and non-negative, got {raw!r}")
        values.add(value)
    return tuple(sorted(values))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run(command: Sequence[str], *, timeout: float = 15.0) -> subprocess.CompletedProcess[str]:
    try:
        completed = subprocess.run(
            list(command),
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise QualificationError(f"command failed: {command!r}: {type(exc).__name__}: {exc}") from exc
    if completed.returncode:
        raise QualificationError(
            f"command returned {completed.returncode}: {command!r}: "
            f"{completed.stderr.strip() or '<empty stderr>'}"
        )
    return completed


def _optional_number(raw: str) -> float | None:
    value = raw.strip()
    if value in {"", "N/A", "[N/A]", "Not Supported", "[Not Supported]"}:
        return None
    try:
        number = float(value)
    except ValueError as exc:
        raise QualificationError(f"invalid numeric nvidia-smi field: {raw!r}") from exc
    if not math.isfinite(number):
        raise QualificationError(f"non-finite nvidia-smi field: {raw!r}")
    return number


def parse_gpu_query(text: str) -> dict[str, Any]:
    rows = list(csv.reader(line for line in text.splitlines() if line.strip()))
    if len(rows) != 1:
        raise QualificationError(f"expected one nvidia-smi GPU row, got {len(rows)}")
    values = rows[0]
    if len(values) != len(_GPU_QUERY_FIELDS):
        raise QualificationError(
            f"expected {len(_GPU_QUERY_FIELDS)} nvidia-smi columns, got {len(values)}: {values!r}"
        )
    parsed: dict[str, Any] = {}
    for name, raw in zip(_GPU_QUERY_FIELDS, values, strict=True):
        value = raw.strip()
        parsed[name] = _optional_number(value) if name in _NUMERIC_GPU_FIELDS else value
    return parsed


def query_gpu(nvidia_smi: str, physical_gpu: int) -> dict[str, Any]:
    completed = _run(
        (
            nvidia_smi,
            "-i",
            str(physical_gpu),
            "--query-gpu=" + ",".join(_GPU_QUERY_FIELDS),
            "--format=csv,noheader,nounits",
        )
    )
    return parse_gpu_query(completed.stdout)


def query_compute_apps(nvidia_smi: str, gpu_uuid: str) -> list[dict[str, Any]]:
    completed = _run(
        (
            nvidia_smi,
            "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory",
            "--format=csv,noheader,nounits",
        )
    )
    result = []
    for row in csv.reader(line for line in completed.stdout.splitlines() if line.strip()):
        if len(row) != 4:
            raise QualificationError(f"invalid nvidia-smi compute-app row: {row!r}")
        uuid, pid, process_name, memory = (item.strip() for item in row)
        if uuid != gpu_uuid:
            continue
        try:
            parsed_pid = int(pid)
        except ValueError as exc:
            raise QualificationError(f"invalid compute-app pid: {pid!r}") from exc
        result.append(
            {
                "gpu_uuid": uuid,
                "pid": parsed_pid,
                "process_name": process_name,
                "used_gpu_memory_mib": _optional_number(memory),
            }
        )
    return result


def validate_runtime_policy(environ: Mapping[str, str], physical_gpu: int) -> dict[str, Any]:
    balance_gate = environ.get(BALANCE_GATE_ENV)
    if balance_gate != "0":
        raise QualificationError(f"{BALANCE_GATE_ENV} must be explicitly 0, got {balance_gate!r}")
    expected_visible = str(physical_gpu)
    visible = environ.get("CUDA_VISIBLE_DEVICES")
    if visible != expected_visible:
        raise QualificationError(f"CUDA_VISIBLE_DEVICES must be exactly {expected_visible!r}, got {visible!r}")
    production_threshold = environ.get(PRODUCTION_THRESHOLD_ENV)
    if production_threshold not in (None, "", "0", "0.0"):
        raise QualificationError(
            f"{PRODUCTION_THRESHOLD_ENV} must be unset or zero because this tool passes each threshold directly, "
            f"got {production_threshold!r}"
        )
    launch_blocking = environ.get("CUDA_LAUNCH_BLOCKING")
    if launch_blocking not in (None, "", "0"):
        raise QualificationError(f"CUDA_LAUNCH_BLOCKING must be unset or zero, got {launch_blocking!r}")
    return {
        "gpu_clock_policy": GPU_CLOCK_POLICY,
        "gpu_clock_policy_id": GPU_CLOCK_POLICY_ID,
        "gpu_clock_mutation": False,
        "gpu_balance_health_gate": balance_gate,
        "cuda_visible_devices": visible,
        "production_threshold_env": production_threshold,
        "cuda_launch_blocking": launch_blocking,
    }


def validate_gpu_sample(
    row: Mapping[str, Any],
    apps: Sequence[Mapping[str, Any]],
    *,
    expected_physical_gpu: int,
    allowed_pids: set[int],
    max_memory_mib: float | None,
    max_utilization_percent: float | None,
    max_temperature_c: float,
    power_tolerance_w: float,
) -> None:
    if str(row.get("index")) != str(expected_physical_gpu):
        raise QualificationError(
            f"nvidia-smi returned GPU index {row.get('index')!r}, expected {expected_physical_gpu}"
        )
    foreign = [item for item in apps if int(item.get("pid", -1)) not in allowed_pids]
    if foreign:
        raise QualificationError(f"foreign compute applications are present on GPU{expected_physical_gpu}: {foreign}")
    if max_memory_mib is not None:
        memory = row.get("memory.used")
        if memory is None or float(memory) > max_memory_mib:
            raise QualificationError(
                f"GPU{expected_physical_gpu} idle memory is {memory!r} MiB, limit={max_memory_mib:g}"
            )
    if max_utilization_percent is not None:
        utilization = row.get("utilization.gpu")
        if utilization is None or float(utilization) > max_utilization_percent:
            raise QualificationError(
                f"GPU{expected_physical_gpu} utilization is {utilization!r}%, "
                f"idle limit={max_utilization_percent:g}%"
            )
    temperature = row.get("temperature.gpu")
    if temperature is None or float(temperature) > max_temperature_c:
        raise QualificationError(
            f"GPU{expected_physical_gpu} temperature is {temperature!r} C, limit={max_temperature_c:g}"
        )

    app_clock = str(row.get("clocks_event_reasons.applications_clocks_setting", "")).strip().lower()
    if app_clock != "not active":
        raise QualificationError(f"application-clock override is not inactive: {app_clock!r}")
    for field_name in _HARD_THROTTLE_FIELDS:
        state = str(row.get(field_name, "")).strip().lower()
        if state != "not active":
            raise QualificationError(f"GPU hard-throttle signal {field_name} is {state!r}")

    current_limit = row.get("gpu.power.current_limit")
    requested_limit = row.get("gpu.power.requested_limit")
    default_limit = row.get("gpu.power.default_limit")
    if None in (current_limit, requested_limit, default_limit):
        raise QualificationError("nvidia-smi did not expose current/requested/default power limits")
    if abs(float(current_limit) - float(default_limit)) > power_tolerance_w:
        raise QualificationError(
            f"GPU power limit is not the node default: current={current_limit}, default={default_limit}"
        )
    if float(requested_limit) + power_tolerance_w < float(current_limit):
        raise QualificationError(
            f"GPU requested power limit is below current: requested={requested_limit}, current={current_limit}"
        )
    ecc = row.get("ecc.errors.uncorrected.volatile.total")
    if ecc is not None and float(ecc) != 0.0:
        raise QualificationError(f"GPU has uncorrected volatile ECC errors: {ecc}")


def collect_idle_samples(args: argparse.Namespace) -> list[dict[str, Any]]:
    samples = []
    for sample_index in range(args.idle_samples):
        row = query_gpu(args.nvidia_smi, args.physical_gpu)
        apps = query_compute_apps(args.nvidia_smi, str(row["uuid"]))
        validate_gpu_sample(
            row,
            apps,
            expected_physical_gpu=args.physical_gpu,
            allowed_pids=set(),
            max_memory_mib=args.idle_max_memory_mib,
            max_utilization_percent=args.idle_max_utilization_percent,
            max_temperature_c=args.idle_max_temperature_c,
            power_tolerance_w=args.power_limit_tolerance_w,
        )
        samples.append({"gpu": row, "compute_apps": apps, "unix_time": time.time()})
        if sample_index + 1 < args.idle_samples and args.idle_interval_seconds:
            time.sleep(args.idle_interval_seconds)
    return samples


def production_block_sizes() -> torch.Tensor:
    """Return the production H3 prefix-segment and 4x4x4 video tile sizes."""
    prefix_sizes: list[int] = []
    for segment in PREFIX_SEGMENTS:
        full, remainder = divmod(segment, BLOCK_SIZE)
        prefix_sizes.extend([BLOCK_SIZE] * full)
        if remainder:
            prefix_sizes.append(remainder)

    video_sizes = []
    tile_t = tile_h = tile_w = 4
    for t0 in range(0, VIDEO_SHAPE[0], tile_t):
        for h0 in range(0, VIDEO_SHAPE[1], tile_h):
            for w0 in range(0, VIDEO_SHAPE[2], tile_w):
                video_sizes.append(
                    min(tile_t, VIDEO_SHAPE[0] - t0)
                    * min(tile_h, VIDEO_SHAPE[1] - h0)
                    * min(tile_w, VIDEO_SHAPE[2] - w0)
                )
    sizes = torch.tensor(prefix_sizes + video_sizes, dtype=torch.int32)
    if len(prefix_sizes) != PREFIX_BLOCKS or len(video_sizes) != VIDEO_BLOCKS:
        raise QualificationError(
            f"internal H3 block-count mismatch: prefix={len(prefix_sizes)}, video={len(video_sizes)}"
        )
    if sizes.numel() != LOGICAL_BLOCKS or int(sizes.sum().item()) != VALID_ROWS:
        raise QualificationError(
            f"internal H3 block-size mismatch: blocks={sizes.numel()}, rows={int(sizes.sum().item())}"
        )
    return sizes


def build_ordered_q2k_indices(
    scores: torch.Tensor,
    *,
    prefix_blocks: int,
    video_blocks: int,
    topk: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Match the production direct-q2k route without a dense bool map."""
    if scores.ndim != 4:
        raise ValueError(f"scores must be [B,H,Nq,Nkv], got {tuple(scores.shape)}")
    if prefix_blocks < 0 or video_blocks < 0 or topk < 0:
        raise ValueError("prefix_blocks, video_blocks, and topk must be non-negative")
    total_blocks = prefix_blocks + video_blocks
    if tuple(scores.shape[-2:]) != (total_blocks, total_blocks):
        raise ValueError(
            f"scores tail must be {(total_blocks, total_blocks)}, got {tuple(scores.shape[-2:])}"
        )
    keep_video = min(topk, video_blocks)
    selected: torch.Tensor | None = None
    if 0 < keep_video < video_blocks:
        selected = scores[..., prefix_blocks:, prefix_blocks:].topk(keep_video, dim=-1).indices.to(torch.int32)
        selected = selected.sort(dim=-1).values

    q2k_idx = torch.full_like(scores, -1, dtype=torch.int32)
    q2k_num = torch.full(
        scores.shape[:-1],
        prefix_blocks + keep_video,
        dtype=torch.int32,
        device=scores.device,
    )
    all_indices = torch.arange(total_blocks, dtype=torch.int32, device=scores.device)
    if keep_video == video_blocks:
        q2k_idx.copy_(all_indices)
        q2k_num.fill_(total_blocks)
        return q2k_idx.contiguous(), q2k_num.contiguous()
    if prefix_blocks:
        q2k_idx[..., prefix_blocks:, :prefix_blocks] = all_indices[:prefix_blocks]
    if selected is not None:
        q2k_idx[
            ...,
            prefix_blocks:,
            prefix_blocks : prefix_blocks + keep_video,
        ] = selected + prefix_blocks
    if prefix_blocks:
        q2k_idx[..., :prefix_blocks, :] = all_indices
        q2k_num[..., :prefix_blocks] = total_blocks
    return q2k_idx.contiguous(), q2k_num.contiguous()


def validate_metadata_contract(
    block_sizes: torch.Tensor,
    q2k_idx: torch.Tensor,
    q2k_num: torch.Tensor,
) -> None:
    if tuple(block_sizes.shape) != (LOGICAL_BLOCKS,) or block_sizes.dtype != torch.int32:
        raise QualificationError("block_sizes does not match the 1,641-tile int32 contract")
    expected_idx = (BATCH, LOCAL_HEADS, LOGICAL_BLOCKS, LOGICAL_BLOCKS)
    expected_num = expected_idx[:-1]
    if tuple(q2k_idx.shape) != expected_idx or q2k_idx.dtype != torch.int32 or not q2k_idx.is_contiguous():
        raise QualificationError(f"q2k_idx contract mismatch: {tuple(q2k_idx.shape)}, {q2k_idx.dtype}")
    if tuple(q2k_num.shape) != expected_num or q2k_num.dtype != torch.int32 or not q2k_num.is_contiguous():
        raise QualificationError(f"q2k_num contract mismatch: {tuple(q2k_num.shape)}, {q2k_num.dtype}")
    if not bool((q2k_num[..., :PREFIX_BLOCKS] == LOGICAL_BLOCKS).all().item()):
        raise QualificationError("prefix query blocks are not fully dense")
    if not bool((q2k_num[..., PREFIX_BLOCKS:] == VIDEO_ACTIVE_KV_BLOCKS).all().item()):
        raise QualificationError("video query blocks do not retain prefix+topk=183 KV blocks")


def _valid_token_mask(block_sizes: torch.Tensor, device: torch.device) -> torch.Tensor:
    positions = torch.arange(BLOCK_SIZE, device=device)
    return (positions[None, :] < block_sizes.to(device=device)[:, None]).reshape(TILED_ROWS)


def create_inputs(
    device: torch.device,
    *,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device=device).manual_seed(seed)
    shape = (BATCH, TILED_ROWS, LOCAL_HEADS, HEAD_DIM)
    q = torch.randn(shape, generator=generator, dtype=DTYPE, device=device)
    k = torch.randn(shape, generator=generator, dtype=DTYPE, device=device)
    v = torch.randn(shape, generator=generator, dtype=DTYPE, device=device)
    block_sizes = production_block_sizes().to(device=device)
    valid = _valid_token_mask(block_sizes, device).view(1, TILED_ROWS, 1, 1)
    q.masked_fill_(~valid, 0)
    k.masked_fill_(~valid, 0)
    v.masked_fill_(~valid, 0)
    return q.contiguous(), k.contiguous(), v.contiguous(), block_sizes.contiguous()


def create_production_q2k(
    q: torch.Tensor,
    k: torch.Tensor,
    block_sizes: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    q_pool = q.view(BATCH, LOGICAL_BLOCKS, BLOCK_SIZE, LOCAL_HEADS, HEAD_DIM).sum(
        dim=2,
        dtype=torch.float32,
    )
    k_pool = k.view(BATCH, LOGICAL_BLOCKS, BLOCK_SIZE, LOCAL_HEADS, HEAD_DIM).sum(
        dim=2,
        dtype=torch.float32,
    )
    divisor = block_sizes.view(1, -1, 1, 1).clamp_min(1)
    q_pool = (q_pool / divisor).permute(0, 2, 1, 3)
    k_pool = (k_pool / divisor).permute(0, 2, 1, 3)
    scores = torch.matmul(q_pool, k_pool.transpose(-2, -1)) * SOFTMAX_SCALE
    return build_ordered_q2k_indices(
        scores,
        prefix_blocks=PREFIX_BLOCKS,
        video_blocks=VIDEO_BLOCKS,
        topk=VIDEO_TOPK,
    )


def _raw_cpu_bytes(value: torch.Tensor) -> bytes:
    contiguous = value.detach().cpu().contiguous()
    if contiguous.dtype == torch.bfloat16:
        return contiguous.view(torch.uint16).numpy().tobytes(order="C")
    return contiguous.numpy().tobytes(order="C")


def tensor_sha256(value: torch.Tensor, *, chunk_elements: int) -> str:
    if chunk_elements < 1:
        raise ValueError("chunk_elements must be positive")
    flat = value.detach().contiguous().view(-1)
    digest = hashlib.sha256()
    for first in range(0, flat.numel(), chunk_elements):
        digest.update(_raw_cpu_bytes(flat[first : first + chunk_elements]))
    return digest.hexdigest()


def relative_exact_metrics(
    exact: torch.Tensor,
    candidate: torch.Tensor,
    *,
    chunk_rows: int,
    valid_rows: torch.Tensor | None = None,
) -> dict[str, Any]:
    if tuple(exact.shape) != tuple(candidate.shape):
        raise ValueError(f"shape mismatch: exact={tuple(exact.shape)}, candidate={tuple(candidate.shape)}")
    if exact.dtype != candidate.dtype or exact.device != candidate.device:
        raise ValueError("exact and candidate must have identical dtype and device")
    if exact.ndim != 4 or chunk_rows < 1:
        raise ValueError("metrics expect BSHD tensors and a positive chunk_rows")
    if valid_rows is not None:
        if valid_rows.dtype != torch.bool or tuple(valid_rows.shape) != (exact.shape[1],):
            raise ValueError(
                f"valid_rows must be bool[{exact.shape[1]}], got {valid_rows.dtype}{tuple(valid_rows.shape)}"
            )
        if valid_rows.device != exact.device:
            raise ValueError("valid_rows must be on the same device as the outputs")

    count = 0
    sum_abs = 0.0
    sum_squared = 0.0
    exact_squared = 0.0
    candidate_squared = 0.0
    dot = 0.0
    max_abs = 0.0
    exact_max_abs = 0.0
    for first in range(0, exact.shape[1], chunk_rows):
        exact_chunk = exact[:, first : first + chunk_rows]
        candidate_chunk = candidate[:, first : first + chunk_rows]
        if valid_rows is not None:
            selection = valid_rows[first : first + chunk_rows]
            exact_chunk = exact_chunk[:, selection]
            candidate_chunk = candidate_chunk[:, selection]
        if exact_chunk.numel() == 0:
            continue
        exact_chunk = exact_chunk.float()
        candidate_chunk = candidate_chunk.float()
        difference = candidate_chunk - exact_chunk
        count += difference.numel()
        sum_abs += float(difference.abs().sum().item())
        sum_squared += float(difference.square().sum().item())
        exact_squared += float(exact_chunk.square().sum().item())
        candidate_squared += float(candidate_chunk.square().sum().item())
        dot += float((exact_chunk * candidate_chunk).sum().item())
        max_abs = max(max_abs, float(difference.abs().max().item()))
        exact_max_abs = max(exact_max_abs, float(exact_chunk.abs().max().item()))
    tiny = sys.float_info.min
    return {
        "domain": "valid production rows only" if valid_rows is not None else "all tiled rows",
        "element_count": count,
        "max_abs": max_abs,
        "mean_abs": sum_abs / count,
        "rmse": math.sqrt(sum_squared / count),
        "exact_rms": math.sqrt(exact_squared / count),
        "relative_l2": math.sqrt(sum_squared / max(exact_squared, tiny)),
        "relative_max": max_abs / max(exact_max_abs, tiny),
        "cosine_similarity": dot / max(math.sqrt(exact_squared * candidate_squared), tiny),
    }


def invoke_kernel(
    kernel: Callable[..., tuple[torch.Tensor, torch.Tensor | None]],
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q2k_idx: torch.Tensor,
    q2k_num: torch.Tensor,
    block_sizes: torch.Tensor,
    *,
    threshold: float,
    out: torch.Tensor,
    lse: torch.Tensor,
) -> torch.Tensor:
    skip = threshold if threshold > 0 else None
    result, returned_lse = kernel(
        q,
        k,
        v,
        q2k_idx,
        LOGICAL_BLOCKS,
        block_sizes=block_sizes,
        q2k_block_nums=q2k_num,
        softmax_scale=SOFTMAX_SCALE,
        skip_softmax_threshold_scale_factor=skip,
        return_lse=False,
        out=out,
        lse=lse,
    )
    if result.data_ptr() != out.data_ptr():
        raise QualificationError("FlashInfer kernel did not honor the caller-provided output buffer")
    if returned_lse is not None:
        raise QualificationError("FlashInfer kernel returned LSE despite return_lse=False")
    return result


def _percentile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * fraction
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def summarize(values: Sequence[float]) -> dict[str, Any]:
    if not values:
        raise ValueError("cannot summarize an empty sample set")
    return {
        "samples": [float(value) for value in values],
        "count": len(values),
        "minimum": min(values),
        "p50": statistics.median(values),
        "p90": _percentile(values, 0.90),
        "maximum": max(values),
        "mean": statistics.fmean(values),
    }


def _git(root: Path, command: Sequence[str]) -> str | None:
    completed = subprocess.run(
        ["git", "-C", str(root), *command],
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


def source_provenance(flashinfer_root: Path, kernel_module_file: Path) -> dict[str, Any]:
    wrapper = flashinfer_root / "flashinfer/cute_dsl/sparse/bsa_attn_sm120.py"
    implementation = flashinfer_root / "flashinfer/cute_dsl/sparse/sm120_blk64/flash_fwd_sm120.py"
    core_route = REPO_ROOT / "vllm_omni/diffusion/attention/backends/fastvideo_vsa.py"
    harness = Path(__file__).resolve()
    for path in (wrapper, implementation, core_route, harness, kernel_module_file):
        if not path.is_file():
            raise QualificationError(f"required provenance source is missing: {path}")
    return {
        "vllm_omni_git_head": _git(REPO_ROOT, ("rev-parse", "HEAD")),
        "vllm_omni_git_status": _git(REPO_ROOT, ("status", "--short")),
        "flashinfer_git_head": _git(flashinfer_root, ("rev-parse", "HEAD")),
        "flashinfer_git_status": _git(flashinfer_root, ("status", "--short")),
        "flashinfer_root": str(flashinfer_root),
        "flashinfer_wrapper": str(wrapper),
        "flashinfer_wrapper_sha256": sha256_file(wrapper),
        "flashinfer_kernel": str(implementation),
        "flashinfer_kernel_sha256": sha256_file(implementation),
        "imported_kernel_module": str(kernel_module_file),
        "imported_kernel_module_sha256": sha256_file(kernel_module_file),
        "vllm_core_route": str(core_route),
        "vllm_core_route_sha256": sha256_file(core_route),
        "harness": str(harness),
        "harness_sha256": sha256_file(harness),
    }


def import_flashinfer_kernel(
    flashinfer_root: Path,
) -> tuple[Callable[..., tuple[torch.Tensor, torch.Tensor | None]], Path, str | None]:
    root = flashinfer_root.expanduser().resolve()
    expected = root / "flashinfer/cute_dsl/sparse/bsa_attn_sm120.py"
    if not expected.is_file():
        raise QualificationError(f"combined FlashInfer SM120 wrapper is missing: {expected}")
    loaded = sys.modules.get("flashinfer")
    if loaded is not None:
        loaded_file = Path(str(getattr(loaded, "__file__", ""))).resolve()
        if not loaded_file.is_relative_to(root):
            raise QualificationError(f"flashinfer was already imported from the wrong tree: {loaded_file}")
    sys.path.insert(0, str(root))
    importlib.invalidate_caches()
    module = importlib.import_module("flashinfer.cute_dsl.sparse.bsa_attn_sm120")
    module_file = Path(str(module.__file__)).resolve()
    if not module_file.is_relative_to(root):
        raise QualificationError(f"SM120 kernel imported from {module_file}, expected under {root}")
    kernel = getattr(module, "bsa_attn_sm120_blk64_fwd", None)
    if not callable(kernel):
        raise QualificationError("combined FlashInfer tree has no callable bsa_attn_sm120_blk64_fwd")
    flashinfer_module = sys.modules.get("flashinfer")
    version = getattr(flashinfer_module, "__version__", None)
    return kernel, module_file, str(version) if version is not None else None


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _tensor_record(value: torch.Tensor, *, chunk_elements: int) -> dict[str, Any]:
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype).removeprefix("torch."),
        "device": str(value.device),
        "contiguous": value.is_contiguous(),
        "sha256": tensor_sha256(value, chunk_elements=chunk_elements),
    }


def _check_active_gpu(args: argparse.Namespace, gpu_uuid: str) -> dict[str, Any]:
    row = query_gpu(args.nvidia_smi, args.physical_gpu)
    apps = query_compute_apps(args.nvidia_smi, gpu_uuid)
    validate_gpu_sample(
        row,
        apps,
        expected_physical_gpu=args.physical_gpu,
        allowed_pids={os.getpid()},
        max_memory_mib=None,
        max_utilization_percent=None,
        max_temperature_c=args.idle_max_temperature_c,
        power_tolerance_w=args.power_limit_tolerance_w,
    )
    return {"gpu": row, "compute_apps": apps, "unix_time": time.time()}


def _timed_kernel(
    kernel: Callable[..., tuple[torch.Tensor, torch.Tensor | None]],
    tensors: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    *,
    threshold: float,
    out: torch.Tensor,
    lse: torch.Tensor,
) -> tuple[float, float]:
    q, k, v, q2k_idx, q2k_num, block_sizes = tensors
    torch.accelerator.synchronize(q.device)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    wall_start = time.perf_counter()
    start.record()
    invoke_kernel(
        kernel,
        q,
        k,
        v,
        q2k_idx,
        q2k_num,
        block_sizes,
        threshold=threshold,
        out=out,
        lse=lse,
    )
    end.record()
    end.synchronize()
    return float(start.elapsed_time(end)), time.perf_counter() - wall_start


def main() -> int:
    args = parse_args()
    output_dir = (
        args.output_dir
        or DEFAULT_RESULTS_ROOT / f"minimax-h3-vsa-skip-softmax-sm120-{stamp()}"
    ).expanduser().resolve()
    report_path = output_dir / "results.json"
    payload: dict[str, Any] = {
        "schema_version": 1,
        "status": "initializing",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "single-GPU synthetic-QKV FlashInfer SM120 CuTeDSL VSA operator screen",
        "quality_contract": (
            "threshold=0 is exact VSA; thresholds>0 are approximate; operator-output error is not video quality"
        ),
        "gpu_clock_policy": GPU_CLOCK_POLICY,
        "gpu_clock_policy_id": GPU_CLOCK_POLICY_ID,
        "gpu_clock_mutation": False,
        "thresholds": list(args.thresholds),
        "warmups_per_threshold": args.warmups,
        "repeats_per_threshold": args.repeats,
        "timing_order": "alternating forward/reverse threshold order per repeat",
        "seed": args.seed,
        "geometry": {
            "batch": BATCH,
            "prefix_segments": list(PREFIX_SEGMENTS),
            "video_shape_thw": list(VIDEO_SHAPE),
            "valid_rows": VALID_ROWS,
            "sp8_aligned_rows": SP8_ALIGNED_ROWS,
            "block_size": BLOCK_SIZE,
            "prefix_blocks": PREFIX_BLOCKS,
            "video_blocks": VIDEO_BLOCKS,
            "logical_blocks": LOGICAL_BLOCKS,
            "tiled_rows": TILED_ROWS,
            "local_heads": LOCAL_HEADS,
            "head_dim": HEAD_DIM,
            "video_topk": VIDEO_TOPK,
            "video_active_kv_blocks": VIDEO_ACTIVE_KV_BLOCKS,
            "prefix_query_active_kv_blocks": LOGICAL_BLOCKS,
            "softmax_scale": SOFTMAX_SCALE,
            "dtype": "bfloat16",
        },
    }
    _write_json(report_path, payload)

    try:
        payload["runtime_policy"] = validate_runtime_policy(os.environ, args.physical_gpu)
        flashinfer_root = args.flashinfer_root.expanduser().resolve()
        payload["idle_preflight"] = collect_idle_samples(args)
        gpu_uuid = str(payload["idle_preflight"][-1]["gpu"]["uuid"])

        kernel, module_file, flashinfer_version = import_flashinfer_kernel(flashinfer_root)
        payload["source"] = source_provenance(flashinfer_root, module_file)
        if not torch.accelerator.is_available() or torch.accelerator.device_count() != 1:
            raise QualificationError(
                f"expected exactly one visible CUDA GPU, found {torch.accelerator.device_count()}"
            )
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
        properties = torch.cuda.get_device_properties(device)
        if (properties.major, properties.minor) not in {(12, 0), (12, 1)}:
            raise QualificationError(
                "combined CuTeDSL VSA screening requires SM120/SM121, "
                f"got SM{properties.major}{properties.minor}"
            )
        payload["runtime"] = {
            "python": sys.version,
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "flashinfer": flashinfer_version,
            "device_name": properties.name,
            "compute_capability": f"{properties.major}.{properties.minor}",
            "total_hbm_bytes": properties.total_memory,
            "logical_device": "cuda:0",
            "physical_gpu": args.physical_gpu,
        }

        torch.accelerator.reset_peak_memory_stats(device)
        q, k, v, block_sizes = create_inputs(device, seed=args.seed)
        valid_token_rows = _valid_token_mask(block_sizes, device)
        q2k_idx, q2k_num = create_production_q2k(q, k, block_sizes)
        validate_metadata_contract(block_sizes, q2k_idx, q2k_num)
        tensors = (q, k, v, q2k_idx, q2k_num, block_sizes)
        torch.accelerator.synchronize(device)
        payload["inputs"] = {
            "kind": "deterministic synthetic production-shape Q/K/V",
            "generator": "torch CUDA Generator.manual_seed; version-dependent stream recorded in provenance",
            "padding": "zero outside each variable tile size",
            "q": _tensor_record(q, chunk_elements=args.hash_chunk_elements),
            "k": _tensor_record(k, chunk_elements=args.hash_chunk_elements),
            "v": _tensor_record(v, chunk_elements=args.hash_chunk_elements),
            "block_sizes": _tensor_record(block_sizes, chunk_elements=args.hash_chunk_elements),
            "q2k_idx": _tensor_record(q2k_idx, chunk_elements=args.hash_chunk_elements),
            "q2k_num": _tensor_record(q2k_num, chunk_elements=args.hash_chunk_elements),
        }
        payload["setup_peak_memory"] = {
            "allocated_bytes": torch.accelerator.max_memory_allocated(device),
            "reserved_bytes": torch.accelerator.max_memory_reserved(device),
        }
        _write_json(report_path, payload)

        exact = torch.empty_like(q)
        candidate = torch.empty_like(q)
        exact_lse = torch.empty((BATCH, LOCAL_HEADS, TILED_ROWS), dtype=torch.float32, device=device)
        candidate_lse = torch.empty_like(exact_lse)

        # Compile and warm both exact and approximate specializations before
        # retaining the exact reference. Positive threshold values share one
        # compile key, but each requested value receives the same warmup count.
        for threshold in args.thresholds:
            for _ in range(args.warmups):
                invoke_kernel(
                    kernel,
                    q,
                    k,
                    v,
                    q2k_idx,
                    q2k_num,
                    block_sizes,
                    threshold=threshold,
                    out=candidate,
                    lse=candidate_lse,
                )
        torch.accelerator.synchronize(device)
        invoke_kernel(
            kernel,
            q,
            k,
            v,
            q2k_idx,
            q2k_num,
            block_sizes,
            threshold=0.0,
            out=exact,
            lse=exact_lse,
        )
        torch.accelerator.synchronize(device)
        payload["exact_reference"] = _tensor_record(exact, chunk_elements=args.hash_chunk_elements)

        timings = {
            threshold: {"event_ms": [], "wall_s": []}
            for threshold in args.thresholds
        }
        for repeat_index in range(args.repeats):
            ordered_thresholds = (
                args.thresholds
                if repeat_index % 2 == 0
                else tuple(reversed(args.thresholds))
            )
            for threshold in ordered_thresholds:
                event, wall = _timed_kernel(
                    kernel,
                    tensors,
                    threshold=threshold,
                    out=candidate,
                    lse=candidate_lse,
                )
                timings[threshold]["event_ms"].append(event)
                timings[threshold]["wall_s"].append(wall)

        results = []
        for threshold in args.thresholds:
            torch.accelerator.reset_peak_memory_stats(device)
            # Recompute one candidate outside the timed samples so error and
            # hash collection cannot perturb any threshold's latency sample.
            invoke_kernel(
                kernel,
                q,
                k,
                v,
                q2k_idx,
                q2k_num,
                block_sizes,
                threshold=threshold,
                out=candidate,
                lse=candidate_lse,
            )
            torch.accelerator.synchronize(device)
            kernel_peak_allocated = torch.accelerator.max_memory_allocated(device)
            kernel_peak_reserved = torch.accelerator.max_memory_reserved(device)
            error = relative_exact_metrics(
                exact,
                candidate,
                chunk_rows=args.metric_chunk_rows,
                valid_rows=valid_token_rows,
            )
            output_sha = tensor_sha256(candidate, chunk_elements=args.hash_chunk_elements)
            if threshold == 0 and (
                error["relative_l2"] != 0.0
                or output_sha != payload["exact_reference"]["sha256"]
            ):
                raise QualificationError(
                    "threshold-zero kernel output is not deterministic against its retained exact reference"
                )
            active_sample = _check_active_gpu(args, gpu_uuid)
            result = {
                "threshold_scale_factor": threshold,
                "kernel_mode": "exact" if threshold == 0 else "approximate_skip_softmax",
                "skip_threshold_argument": threshold if threshold > 0 else None,
                "cuda_event_ms": summarize(timings[threshold]["event_ms"]),
                "synchronized_wall_s": summarize(timings[threshold]["wall_s"]),
                "relative_exact_error": error,
                "output_sha256": output_sha,
                "matches_exact_sha256": output_sha == payload["exact_reference"]["sha256"],
                "memory": {
                    "kernel_peak_allocated_bytes": kernel_peak_allocated,
                    "kernel_peak_reserved_bytes": kernel_peak_reserved,
                    "peak_allocated_bytes": torch.accelerator.max_memory_allocated(device),
                    "peak_reserved_bytes": torch.accelerator.max_memory_reserved(device),
                },
                "gpu_after": active_sample,
            }
            results.append(result)
            payload["results"] = results
            _write_json(report_path, payload)
            print(
                "MINIMAX_H3_VSA_SKIP_SOFTMAX_SCREEN_V1 "
                f"threshold={threshold:g} "
                f"cuda_p50_ms={result['cuda_event_ms']['p50']:.6f} "
                f"relative_l2={error['relative_l2']:.9g} "
                f"output_sha256={output_sha} "
                f"clock_policy_id={GPU_CLOCK_POLICY_ID}",
                flush=True,
            )

        exact_latency = float(results[0]["cuda_event_ms"]["p50"])
        for result in results:
            candidate_latency = float(result["cuda_event_ms"]["p50"])
            result["speedup_vs_exact_p50"] = exact_latency / candidate_latency
            result["latency_reduction_vs_exact_percent"] = 100.0 * (
                1.0 - candidate_latency / exact_latency
            )
        payload["summary"] = {
            "exact_cuda_p50_ms": exact_latency,
            "minimum_cuda_p50_ms": min(float(item["cuda_event_ms"]["p50"]) for item in results),
            "maximum_relative_l2": max(float(item["relative_exact_error"]["relative_l2"]) for item in results),
            "peak_allocated_bytes": max(int(item["memory"]["peak_allocated_bytes"]) for item in results),
            "peak_reserved_bytes": max(int(item["memory"]["peak_reserved_bytes"]) for item in results),
            "interpretation": "select a threshold only after end-to-end same-seed video/audio quality validation",
        }
        payload["postflight"] = _check_active_gpu(args, gpu_uuid)
        payload["status"] = "passed"
        payload["completed_utc"] = datetime.now(timezone.utc).isoformat()
        _write_json(report_path, payload)
        print(f"Report: {report_path}")
        return 0
    except BaseException as exc:
        payload["status"] = "failed"
        payload["error"] = f"{type(exc).__name__}: {exc}"
        payload["completed_utc"] = datetime.now(timezone.utc).isoformat()
        _write_json(report_path, payload)
        print(f"VSA skip-softmax screening failed: {payload['error']}", file=sys.stderr)
        print(f"Report: {report_path}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
