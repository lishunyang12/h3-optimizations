#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Screen the pinned MiniMax-H3 TAEH3 decoder on one SM120 GPU.

This is deliberately a decoder-only qualification, not an H3 end-to-end
benchmark.  It exercises the exact 15-second normalized latent geometry twice:

* a discard-callback pass measures decoder CUDA-event and synchronized wall
  time while validating every temporal chunk; and
* the production pass pipelines TAEH3 chunks through the existing GPU-uint8,
  pinned-D2H, PyAV/libx264 MP4 sink.

The tool never changes clocks, application clocks, or power limits.  Results
are always labelled ``node-default/unlocked clocks`` (machine id
``node_default_dynamic_unlocked``) and the independent GPU
balance benchmark must be explicitly disabled with
``VLLM_OMNI_GPU_BALANCE_HEALTH_GATE=0``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import statistics
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = REPO_ROOT.parents[1]
DEFAULT_CHECKPOINT = WORKSPACE_ROOT / "cache" / "taeh3-62f7591f59dfbb4c3c02b7a621d180a9eeaba26c.safetensors"
DEFAULT_RESULTS_ROOT = WORKSPACE_ROOT / "results" / "profile"
DEFAULT_FFPROBE = WORKSPACE_ROOT / "ffmpeg-tools" / "bin" / "ffprobe"

BALANCE_GATE_ENV = "VLLM_OMNI_GPU_BALANCE_HEALTH_GATE"
VIDEO_DECODE_BACKEND_ENV = "VLLM_OMNI_MINIMAX_H3_VIDEO_DECODE_BACKEND"
TAEH3_CHECKPOINT_ENV = "VLLM_OMNI_MINIMAX_H3_TAEH3_CHECKPOINT"
TAEH3_DTYPE_ENV = "VLLM_OMNI_MINIMAX_H3_TAEH3_DTYPE"
TAEH3_CHUNK_SIZE_ENV = "VLLM_OMNI_MINIMAX_H3_TAEH3_CHUNK_SIZE"
CHUNKED_OUTPUT_ENV = "VLLM_OMNI_MINIMAX_H3_CHUNKED_CPU_MP4_OUTPUT"
CHUNKED_SLOTS_ENV = "VLLM_OMNI_MINIMAX_H3_CHUNKED_CPU_MP4_SLOTS"
NVENC_ENV = "MINIMAX_H3_VAE_NVENC_PIPELINE"

GPU_CLOCK_POLICY = "node-default/unlocked clocks"
GPU_CLOCK_POLICY_ID = "node_default_dynamic_unlocked"
LATENT_SHAPE = (1, 24, 107, 44, 80)  # BCTHW, normalized diffusion latent
OUTPUT_SHAPE = (1, 3, 362, 704, 1280)  # BCTHW
FPS = 24
AUDIO_SAMPLE_RATE = 32_000
TAEH3_DTYPE = "fp32"
TAEH3_CHUNK_SIZE = 5
TAEH3_REVISION = "62f7591f59dfbb4c3c02b7a621d180a9eeaba26c"
TAEH3_CHECKPOINT_SHA256 = "4fd022bfcab08772fe0536b17ea1a3bbb5625be11e397868d1c5d891863d4c13"
TAEH3_CHECKPOINT_SIZE_BYTES = 22_709_752

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
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--latent",
        type=Path,
        help=(
            "optional safetensors capture containing the exact normalized "
            "predecode latent; unset uses deterministic synthetic data and "
            "is performance-only"
        ),
    )
    parser.add_argument("--latent-key", default="video_latent_fp32")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--physical-gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1101)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--idle-samples", type=int, default=3)
    parser.add_argument("--idle-interval-seconds", type=float, default=1.0)
    parser.add_argument("--idle-max-memory-mib", type=float, default=256.0)
    parser.add_argument("--idle-max-temperature-c", type=float, default=70.0)
    parser.add_argument("--power-limit-tolerance-w", type=float, default=1.0)
    parser.add_argument("--nvidia-smi", default="nvidia-smi")
    parser.add_argument("--ffprobe", type=Path, default=DEFAULT_FFPROBE)
    args = parser.parse_args(argv)
    if args.physical_gpu < 0:
        parser.error("--physical-gpu must be non-negative")
    if args.warmups < 0:
        parser.error("--warmups must be non-negative")
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    if args.idle_samples < 1:
        parser.error("--idle-samples must be positive")
    if args.idle_interval_seconds < 0:
        parser.error("--idle-interval-seconds must be non-negative")
    if (
        min(
            args.idle_max_memory_mib,
            args.idle_max_temperature_c,
            args.power_limit_tolerance_w,
        )
        < 0
    ):
        parser.error("idle thresholds must be non-negative")
    if not args.latent_key:
        parser.error("--latent-key must be non-empty")
    return args


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
            f"command returned {completed.returncode}: {command!r}: {completed.stderr.strip() or '<empty stderr>'}"
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
        raise QualificationError(f"expected {len(_GPU_QUERY_FIELDS)} nvidia-smi columns, got {len(values)}: {values!r}")
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


def validate_runtime_policy(environ: Mapping[str, str], physical_gpu: int) -> dict[str, str]:
    balance_gate = environ.get(BALANCE_GATE_ENV)
    if balance_gate != "0":
        raise QualificationError(f"{BALANCE_GATE_ENV} must be explicitly 0, got {balance_gate!r}")
    expected_visible = str(physical_gpu)
    visible = environ.get("CUDA_VISIBLE_DEVICES")
    if visible != expected_visible:
        raise QualificationError(f"CUDA_VISIBLE_DEVICES must be exactly {expected_visible!r}, got {visible!r}")
    expected = {
        VIDEO_DECODE_BACKEND_ENV: "taeh3",
        TAEH3_DTYPE_ENV: TAEH3_DTYPE,
        TAEH3_CHUNK_SIZE_ENV: str(TAEH3_CHUNK_SIZE),
        CHUNKED_OUTPUT_ENV: "1",
        CHUNKED_SLOTS_ENV: "2",
        NVENC_ENV: "0",
    }
    for name, required in expected.items():
        actual = environ.get(name)
        if actual != required:
            raise QualificationError(f"{name} must be explicitly {required!r}, got {actual!r}")
    checkpoint = environ.get(TAEH3_CHECKPOINT_ENV)
    if not checkpoint:
        raise QualificationError(f"{TAEH3_CHECKPOINT_ENV} must name the explicit pinned local checkpoint")
    return {
        "gpu_clock_policy": GPU_CLOCK_POLICY,
        "gpu_clock_policy_id": GPU_CLOCK_POLICY_ID,
        "gpu_balance_health_gate": balance_gate,
        "cuda_visible_devices": visible,
    }


def validate_idle_gpu_sample(
    row: Mapping[str, Any],
    apps: Sequence[Mapping[str, Any]],
    *,
    expected_physical_gpu: int,
    max_memory_mib: float,
    max_temperature_c: float,
    power_tolerance_w: float,
) -> None:
    if str(row.get("index")) != str(expected_physical_gpu):
        raise QualificationError(
            f"nvidia-smi returned GPU index {row.get('index')!r}, expected {expected_physical_gpu}"
        )
    if apps:
        raise QualificationError(f"foreign compute applications are present on GPU{expected_physical_gpu}: {apps}")
    memory = row.get("memory.used")
    temperature = row.get("temperature.gpu")
    if memory is None or float(memory) > max_memory_mib:
        raise QualificationError(f"GPU{expected_physical_gpu} idle memory is {memory!r} MiB, limit={max_memory_mib:g}")
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

    # Workstation SKUs can report N/A when ECC is unsupported.  When an ECC
    # counter exists, any non-zero uncorrected volatile count is a hard stop.
    ecc = row.get("ecc.errors.uncorrected.volatile.total")
    if ecc is not None and float(ecc) != 0.0:
        raise QualificationError(f"GPU has uncorrected volatile ECC errors: {ecc}")


def collect_idle_samples(args: argparse.Namespace) -> list[dict[str, Any]]:
    samples = []
    for sample_index in range(args.idle_samples):
        row = query_gpu(args.nvidia_smi, args.physical_gpu)
        apps = query_compute_apps(args.nvidia_smi, str(row["uuid"]))
        validate_idle_gpu_sample(
            row,
            apps,
            expected_physical_gpu=args.physical_gpu,
            max_memory_mib=args.idle_max_memory_mib,
            max_temperature_c=args.idle_max_temperature_c,
            power_tolerance_w=args.power_limit_tolerance_w,
        )
        samples.append({"gpu": row, "compute_apps": apps, "unix_time": time.time()})
        if sample_index + 1 < args.idle_samples and args.idle_interval_seconds:
            time.sleep(args.idle_interval_seconds)
    return samples


def validate_checkpoint(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise QualificationError(f"TAEH3 checkpoint is missing: {resolved}")
    size = resolved.stat().st_size
    digest = sha256_file(resolved)
    if size != TAEH3_CHECKPOINT_SIZE_BYTES or digest != TAEH3_CHECKPOINT_SHA256:
        raise QualificationError(
            "TAEH3 checkpoint identity mismatch: "
            f"size={size}/{TAEH3_CHECKPOINT_SIZE_BYTES}, "
            f"sha256={digest}/{TAEH3_CHECKPOINT_SHA256}"
        )
    return {
        "path": str(resolved),
        "size_bytes": size,
        "sha256": digest,
        "upstream_revision": TAEH3_REVISION,
    }


def _tensor_sha256(value: torch.Tensor) -> str:
    contiguous = value.detach().cpu().contiguous()
    return hashlib.sha256(contiguous.numpy().tobytes(order="C")).hexdigest()


def load_or_create_latent(args: argparse.Namespace) -> tuple[torch.Tensor, dict[str, Any]]:
    if args.latent is None:
        generator = torch.Generator(device="cpu").manual_seed(args.seed)
        latent = torch.randn(LATENT_SHAPE, generator=generator, dtype=torch.float32)
        source = {
            "kind": "deterministic_synthetic_performance_only",
            "seed": args.seed,
            "file": None,
            "file_sha256": None,
            "key": None,
        }
    else:
        latent_path = args.latent.expanduser().resolve()
        if not latent_path.is_file() or latent_path.suffix != ".safetensors":
            raise QualificationError(f"--latent must be an existing safetensors file: {latent_path}")
        from safetensors.torch import load_file

        tensors = load_file(str(latent_path), device="cpu")
        if args.latent_key not in tensors:
            raise QualificationError(f"latent key {args.latent_key!r} is absent; available={sorted(tensors)}")
        latent = tensors[args.latent_key]
        source = {
            "kind": "captured_normalized_predecode_latent",
            "seed": None,
            "file": str(latent_path),
            "file_sha256": sha256_file(latent_path),
            "key": args.latent_key,
        }
    if tuple(latent.shape) != LATENT_SHAPE:
        raise QualificationError(f"latent shape must be {LATENT_SHAPE}, got {tuple(latent.shape)}")
    if latent.dtype != torch.float32:
        raise QualificationError(f"latent dtype must be torch.float32, got {latent.dtype}")
    if not bool(torch.isfinite(latent).all().item()):
        raise QualificationError("latent contains NaN or infinity")
    source.update(
        {
            "normalized_diffusion_latent": True,
            "shape": list(latent.shape),
            "dtype": str(latent.dtype).removeprefix("torch."),
            "tensor_sha256": _tensor_sha256(latent),
            "minimum": float(latent.min().item()),
            "maximum": float(latent.max().item()),
            "mean": float(latent.double().mean().item()),
            "std": float(latent.double().std().item()),
        }
    )
    return latent.contiguous(), source


@dataclass
class ChunkOrderValidator:
    """Validate decoder callback metadata without synchronizing its tensors."""

    expected_device_type: str = "cuda"
    expected_total_frames: int = OUTPUT_SHAPE[2]
    next_frame: int = 0
    chunks: list[dict[str, Any]] = field(default_factory=list)

    def observe(
        self,
        shape: Sequence[int],
        *,
        device_type: str,
        dtype: torch.dtype,
        first_frame: int,
        total_frames: int,
    ) -> None:
        dimensions = tuple(int(value) for value in shape)
        if len(dimensions) != 5:
            raise QualificationError(f"TAEH3 callback must be BCTHW, got {dimensions}")
        batch, channels, frames, height, width = dimensions
        if (batch, channels, height, width) != (
            OUTPUT_SHAPE[0],
            OUTPUT_SHAPE[1],
            OUTPUT_SHAPE[3],
            OUTPUT_SHAPE[4],
        ):
            raise QualificationError(f"TAEH3 callback geometry mismatch: {dimensions}")
        if frames < 1:
            raise QualificationError("TAEH3 callback emitted an empty chunk")
        if device_type != self.expected_device_type:
            raise QualificationError(f"TAEH3 callback device must be {self.expected_device_type}, got {device_type}")
        if dtype != torch.float32:
            raise QualificationError(f"TAEH3 callback dtype must be float32, got {dtype}")
        if int(total_frames) != self.expected_total_frames:
            raise QualificationError(f"TAEH3 callback total changed: {total_frames}/{self.expected_total_frames}")
        if int(first_frame) != self.next_frame:
            raise QualificationError(f"TAEH3 callback is out of order: first={first_frame}, expected={self.next_frame}")
        if self.next_frame + frames > self.expected_total_frames:
            raise QualificationError("TAEH3 callback exceeded the advertised frame count")
        self.chunks.append(
            {
                "first_frame": self.next_frame,
                "frame_count": frames,
                "shape": list(dimensions),
                "device_type": device_type,
                "dtype": str(dtype).removeprefix("torch."),
            }
        )
        self.next_frame += frames

    def __call__(self, frames: torch.Tensor, first_frame: int, total_frames: int) -> None:
        self.observe(
            frames.shape,
            device_type=frames.device.type,
            dtype=frames.dtype,
            first_frame=first_frame,
            total_frames=total_frames,
        )

    def finish(self) -> dict[str, Any]:
        if self.next_frame != self.expected_total_frames:
            raise QualificationError(f"TAEH3 callback emitted {self.next_frame}/{self.expected_total_frames} frames")
        if len(self.chunks) != 22:
            raise QualificationError(f"TAEH3 chunk_size=5 must emit 22 chunks, got {len(self.chunks)}")
        expected_counts = [17] * 21 + [5]
        actual_counts = [int(item["frame_count"]) for item in self.chunks]
        if actual_counts != expected_counts:
            raise QualificationError(
                f"TAEH3 chunk frame contract mismatch: actual={actual_counts}, expected={expected_counts}"
            )
        return {
            "status": "passed",
            "chunk_count": len(self.chunks),
            "frame_count": self.next_frame,
            "frame_counts": actual_counts,
        }


def _dummy_audio() -> torch.Tensor:
    samples = round(OUTPUT_SHAPE[2] / FPS * AUDIO_SAMPLE_RATE)
    timeline = torch.arange(samples, dtype=torch.float32) / AUDIO_SAMPLE_RATE
    left = 0.01 * torch.sin(2.0 * math.pi * 220.0 * timeline)
    right = 0.01 * torch.sin(2.0 * math.pi * 330.0 * timeline)
    return torch.stack((left, right)).contiguous()


def _decoder_only_pass(decoder: Any, latent: torch.Tensor) -> dict[str, Any]:
    validator = ChunkOrderValidator()
    ntchw = latent.permute(0, 2, 1, 3, 4).contiguous()
    torch.accelerator.synchronize(latent.device)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    wall_start = time.perf_counter()
    start.record()
    result = decoder.decode_ntchw(ntchw, output_callback=validator)
    end.record()
    end.synchronize()
    wall_seconds = time.perf_counter() - wall_start
    if result is not None:
        raise QualificationError("discard-callback decoder unexpectedly returned a full RGB tensor")
    return {
        "cuda_event_ms": float(start.elapsed_time(end)),
        "synchronized_wall_s": wall_seconds,
        "callback": validator.finish(),
    }


def _streaming_pass(
    decoder: Any,
    latent: torch.Tensor,
    audio: torch.Tensor,
) -> tuple[bytes, dict[str, Any]]:
    torch.accelerator.synchronize(latent.device)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    wall_start = time.perf_counter()
    start.record()
    sink = decoder.decode_latent_to_chunked_cpu_mp4(
        latent,
        height=OUTPUT_SHAPE[3],
        width=OUTPUT_SHAPE[4],
        fps=FPS,
        audio_sample_rate=AUDIO_SAMPLE_RATE,
        video_codec_options=None,
        return_output=True,
    )
    end.record()
    submit_wall_seconds = time.perf_counter() - wall_start
    finish_start = time.perf_counter()
    output, sink_stats = sink.finish(audio, AUDIO_SAMPLE_RATE)
    finish_wall_seconds = time.perf_counter() - finish_start
    torch.accelerator.synchronize(latent.device)
    total_wall_seconds = time.perf_counter() - wall_start
    if not isinstance(output, bytes) or not output:
        raise QualificationError("streaming TAEH3 pipeline returned no MP4 bytes")
    return output, {
        "decoder_cuda_event_ms": float(start.elapsed_time(end)),
        "submit_wall_s": submit_wall_seconds,
        "finish_wall_s": finish_wall_seconds,
        "total_wall_s": total_wall_seconds,
        "sink": {name: float(value) for name, value in sink_stats.items()},
    }


def _fraction(value: Any) -> float:
    try:
        return float(Fraction(str(value)))
    except (ValueError, ZeroDivisionError) as exc:
        raise QualificationError(f"invalid media frame rate {value!r}") from exc


def validate_ffprobe_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    streams = payload.get("streams")
    if not isinstance(streams, list):
        raise QualificationError("ffprobe payload has no stream list")
    video = [item for item in streams if item.get("codec_type") == "video"]
    audio = [item for item in streams if item.get("codec_type") == "audio"]
    errors = []
    if len(video) != 1:
        errors.append(f"video_stream_count={len(video)}")
    else:
        item = video[0]
        if item.get("codec_name") != "h264":
            errors.append(f"video_codec={item.get('codec_name')!r}")
        if (int(item.get("width", 0)), int(item.get("height", 0))) != (
            OUTPUT_SHAPE[4],
            OUTPUT_SHAPE[3],
        ):
            errors.append(f"video_geometry={item.get('width')}x{item.get('height')}")
        if abs(_fraction(item.get("avg_frame_rate", "0/1")) - FPS) > 1e-6:
            errors.append(f"video_fps={item.get('avg_frame_rate')!r}")
        try:
            frames = int(item.get("nb_read_frames", 0))
        except (TypeError, ValueError):
            frames = 0
        if frames != OUTPUT_SHAPE[2]:
            errors.append(f"video_frames={item.get('nb_read_frames')!r}")
    if len(audio) != 1:
        errors.append(f"audio_stream_count={len(audio)}")
    else:
        item = audio[0]
        if item.get("codec_name") != "aac":
            errors.append(f"audio_codec={item.get('codec_name')!r}")
        if int(item.get("sample_rate", 0)) != AUDIO_SAMPLE_RATE or int(item.get("channels", 0)) != 2:
            errors.append(f"audio_layout={item.get('sample_rate')}Hz/{item.get('channels')}ch")
    format_data = payload.get("format")
    if not isinstance(format_data, Mapping):
        errors.append("missing_format")
        duration = 0.0
    else:
        duration = float(format_data.get("duration", 0.0))
        if abs(duration - OUTPUT_SHAPE[2] / FPS) > 0.30:
            errors.append(f"duration={duration:.6f}")
    if errors:
        raise QualificationError("media structure mismatch: " + "; ".join(errors))
    return {
        "status": "passed",
        "video_frames": OUTPUT_SHAPE[2],
        "width": OUTPUT_SHAPE[4],
        "height": OUTPUT_SHAPE[3],
        "fps": FPS,
        "audio_sample_rate": AUDIO_SAMPLE_RATE,
        "audio_channels": 2,
        "duration_s": duration,
    }


def probe_media(path: Path, requested_ffprobe: Path) -> dict[str, Any]:
    ffprobe: str | None
    if requested_ffprobe.is_file() and os.access(requested_ffprobe, os.X_OK):
        ffprobe = str(requested_ffprobe)
    else:
        ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        return {"status": "unavailable", "reason": "ffprobe was not found"}
    completed = _run(
        (
            ffprobe,
            "-v",
            "error",
            "-count_frames",
            "-show_entries",
            "format=duration,size:stream=index,codec_type,codec_name,width,height,avg_frame_rate,"
            "nb_read_frames,sample_rate,channels",
            "-of",
            "json",
            str(path),
        ),
        timeout=120.0,
    )
    try:
        raw = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise QualificationError("ffprobe returned invalid JSON") from exc
    result = validate_ffprobe_payload(raw)
    result["ffprobe"] = ffprobe
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


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _git(command: Sequence[str]) -> str | None:
    completed = subprocess.run(
        ["git", "-C", str(REPO_ROOT), *command],
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


def _source_provenance() -> dict[str, Any]:
    taeh3_source = REPO_ROOT / "vllm_omni/diffusion/models/minimax_h3/taeh3.py"
    output_source = REPO_ROOT / "vllm_omni/diffusion/models/minimax_h3/chunked_cpu_output.py"
    return {
        "vllm_omni_git_head": _git(("rev-parse", "HEAD")),
        "vllm_omni_git_status": _git(("status", "--short")),
        "taeh3_source": str(taeh3_source),
        "taeh3_source_sha256": sha256_file(taeh3_source),
        "chunked_output_source": str(output_source),
        "chunked_output_source_sha256": sha256_file(output_source),
        "harness_source": str(Path(__file__).resolve()),
        "harness_source_sha256": sha256_file(Path(__file__).resolve()),
    }


def main() -> int:
    args = parse_args()
    output_dir = (
        (args.output_dir or DEFAULT_RESULTS_ROOT / f"minimax-h3-taeh3-stream-sm120-{stamp()}").expanduser().resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "results.json"
    payload: dict[str, Any] = {
        "schema_version": 1,
        "status": "initializing",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "single-GPU decoder-only; no encoder, DiT, Ulysses, or full H3 VAE",
        "quality_contract": "TAEH3 is an approximate/lossy decoder",
        "gpu_clock_policy": GPU_CLOCK_POLICY,
        "gpu_clock_policy_id": GPU_CLOCK_POLICY_ID,
        "gpu_clock_mutation": False,
        "latent_shape_bcthw": list(LATENT_SHAPE),
        "output_shape_bcthw": list(OUTPUT_SHAPE),
        "fps": FPS,
        "audio_sample_rate": AUDIO_SAMPLE_RATE,
        "warmups": args.warmups,
        "repeats": args.repeats,
        "physical_gpu": args.physical_gpu,
    }
    _write_json(report_path, payload)

    try:
        payload["runtime_policy"] = validate_runtime_policy(os.environ, args.physical_gpu)
        checkpoint = validate_checkpoint(args.checkpoint)
        configured_checkpoint = Path(os.environ[TAEH3_CHECKPOINT_ENV]).expanduser().resolve()
        if configured_checkpoint != Path(checkpoint["path"]):
            raise QualificationError(
                f"--checkpoint and {TAEH3_CHECKPOINT_ENV} differ: {checkpoint['path']} != {configured_checkpoint}"
            )
        payload["checkpoint"] = checkpoint
        payload["source"] = _source_provenance()
        payload["idle_preflight"] = collect_idle_samples(args)
        latent_cpu, latent_provenance = load_or_create_latent(args)
        payload["latent"] = latent_provenance
        _write_json(report_path, payload)

        sys.path.insert(0, str(REPO_ROOT))
        from vllm_omni.diffusion.models.minimax_h3.taeh3 import (
            MiniMaxH3TAEH3Decoder,
            taeh3_decoded_pixel_shape,
        )

        if tuple(taeh3_decoded_pixel_shape(LATENT_SHAPE)) != OUTPUT_SHAPE:
            raise QualificationError("runtime TAEH3 geometry disagrees with the pinned 15-second contract")
        if not torch.accelerator.is_available() or torch.accelerator.device_count() != 1:
            raise QualificationError(f"expected exactly one visible CUDA GPU, found {torch.accelerator.device_count()}")
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
        props = torch.cuda.get_device_properties(device)
        if (props.major, props.minor) not in {(12, 0), (12, 1)}:
            raise QualificationError(
                f"TAEH3 SM120 screening requires compute capability 12.0/12.1, got {props.major}.{props.minor}"
            )
        payload["runtime"] = {
            "python": sys.version,
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "device_name": props.name,
            "compute_capability": f"{props.major}.{props.minor}",
            "total_hbm_bytes": props.total_memory,
            "logical_device": "cuda:0",
            "physical_gpu": args.physical_gpu,
        }

        decoder = MiniMaxH3TAEH3Decoder(
            checkpoint["path"],
            device=device,
            dtype=torch.float32,
            chunk_size=TAEH3_CHUNK_SIZE,
        )
        latent = latent_cpu.to(device=device, non_blocking=False)
        audio = _dummy_audio()

        # Warm both the raw decoder and the actual chunked media path.  Warmup
        # MP4 bytes are deliberately discarded and never reported as a sample.
        for _ in range(args.warmups):
            _decoder_only_pass(decoder, latent)
            warm_output, _ = _streaming_pass(decoder, latent, audio)
            if not warm_output:
                raise QualificationError("TAEH3 warmup produced no MP4")

        samples = []
        for repeat in range(1, args.repeats + 1):
            torch.accelerator.synchronize(device)
            torch.accelerator.reset_peak_memory_stats(device)
            allocated_before = torch.accelerator.memory_allocated(device)
            reserved_before = torch.accelerator.memory_reserved(device)
            decoder_metrics = _decoder_only_pass(decoder, latent)
            output, stream_metrics = _streaming_pass(decoder, latent, audio)
            output_path = output_dir / f"taeh3_stream_run_{repeat:02d}.mp4"
            output_path.write_bytes(output)
            media = probe_media(output_path, args.ffprobe)
            current_apps = query_compute_apps(
                args.nvidia_smi,
                str(payload["idle_preflight"][-1]["gpu"]["uuid"]),
            )
            foreign = [item for item in current_apps if int(item["pid"]) != os.getpid()]
            if foreign:
                raise QualificationError(f"foreign GPU process appeared during repeat {repeat}: {foreign}")
            sample = {
                "repeat": repeat,
                "decoder": decoder_metrics,
                "stream": stream_metrics,
                "memory": {
                    "allocated_before_bytes": allocated_before,
                    "reserved_before_bytes": reserved_before,
                    "peak_allocated_bytes": torch.accelerator.max_memory_allocated(device),
                    "peak_reserved_bytes": torch.accelerator.max_memory_reserved(device),
                },
                "output": {
                    "path": str(output_path),
                    "size_bytes": len(output),
                    "sha256": hashlib.sha256(output).hexdigest(),
                },
                "media": media,
                "compute_apps_after": current_apps,
            }
            samples.append(sample)
            payload["samples"] = samples
            _write_json(report_path, payload)
            print(
                "MINIMAX_H3_TAEH3_DECODER_ONLY_V1 "
                f"repeat={repeat} chunks={decoder_metrics['callback']['chunk_count']} "
                f"frames={decoder_metrics['callback']['frame_count']} "
                f"decoder_cuda_ms={decoder_metrics['cuda_event_ms']:.3f} "
                f"decoder_wall_s={decoder_metrics['synchronized_wall_s']:.6f} "
                f"stream_total_s={stream_metrics['total_wall_s']:.6f} "
                f"peak_allocated_mib={torch.accelerator.max_memory_allocated(device) / 2**20:.2f} "
                f"clock_policy_id={GPU_CLOCK_POLICY_ID}",
                flush=True,
            )

        payload["summary"] = {
            "decoder_cuda_event_ms": summarize([item["decoder"]["cuda_event_ms"] for item in samples]),
            "decoder_synchronized_wall_s": summarize([item["decoder"]["synchronized_wall_s"] for item in samples]),
            "stream_total_wall_s": summarize([item["stream"]["total_wall_s"] for item in samples]),
            "peak_allocated_bytes": max(int(item["memory"]["peak_allocated_bytes"]) for item in samples),
            "peak_reserved_bytes": max(int(item["memory"]["peak_reserved_bytes"]) for item in samples),
        }
        payload["postflight_gpu"] = query_gpu(args.nvidia_smi, args.physical_gpu)
        post_apps = query_compute_apps(
            args.nvidia_smi,
            str(payload["postflight_gpu"]["uuid"]),
        )
        foreign = [item for item in post_apps if int(item["pid"]) != os.getpid()]
        if foreign:
            raise QualificationError(f"foreign GPU process present at postflight: {foreign}")
        payload["postflight_compute_apps"] = post_apps
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
        print(f"TAEH3 screening failed: {payload['error']}", file=sys.stderr)
        print(f"Report: {report_path}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
