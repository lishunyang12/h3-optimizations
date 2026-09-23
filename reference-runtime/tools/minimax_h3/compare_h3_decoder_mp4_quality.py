#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Compare matched MiniMax-H3 MP4 artifacts across one controlled A/B axis.

The comparison is deliberately post-codec and CPU-only. By default it requires
the exact 15-second H3 media shape (362 frames, 1280x704, 24 fps, stereo 32 kHz
audio); explicit expected-shape arguments support other H3-aligned requests.
It then reports:

* per-frame PSNR and SSIM after deterministic FFmpeg conversion to planar RGB
  (``gbrp``) and planar YUV 4:4:4 (``yuv444p``);
* transparent luma temporal-difference/flicker proxies, computed frame by
  frame without retaining the whole video in memory; and
* canonical decoded stereo-f32 PCM identity and error metrics.

This tool does not generate media, use a GPU, change clocks, impose a quality
threshold, or claim a perceptual metric such as LPIPS.  Decoder comparison is
the backward-compatible default; precision-only and combined comparisons can
declare their contracts explicitly on the command line.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path
from typing import Any

import av
import numpy as np
from numpy.typing import NDArray

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = REPO_ROOT.parents[1]
DEFAULT_FFMPEG = WORKSPACE_ROOT / "ffmpeg-tools" / "bin" / "ffmpeg"
DEFAULT_FFPROBE = WORKSPACE_ROOT / "ffmpeg-tools" / "bin" / "ffprobe"

EXPECTED_FRAMES = 362
EXPECTED_WIDTH = 1280
EXPECTED_HEIGHT = 704
EXPECTED_FPS = 24.0
EXPECTED_AUDIO_RATE = 32_000
EXPECTED_AUDIO_CHANNELS = 2
EXPECTED_AUDIO_CHANNEL_LAYOUT = "stereo"
EXPECTED_AUDIO_SAMPLE_FRAMES = 482_400
EXPECTED_PIXEL_FORMAT = "yuv420p"
EXPECTED_START_TIME = 0.0
START_TIME_TOLERANCE = 1e-6
GPU_CLOCK_POLICY = "node-default/unlocked clocks"
GPU_CLOCK_POLICY_ID = "node_default_dynamic_unlocked"

DEFAULT_COMPARISON_AXIS = "decoder"
DEFAULT_COMPARISON_ID = "full_vae_vs_taeh3"
DEFAULT_REFERENCE_CONTRACT = "matched same-latent/same-seed full H3 video VAE MP4"
DEFAULT_CANDIDATE_CONTRACT = "TAEH3 approximate/lossy video decoder MP4"

_NUMBER_RE = r"(?:[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?|[-+]?inf|nan)"
_STATS_FIELD_RE = re.compile(rf"(?P<name>[A-Za-z_]+):(?P<value>{_NUMBER_RE})")


class ComparisonError(RuntimeError):
    """A media alignment, dependency, or metric-contract failure."""


@dataclass(frozen=True)
class ExpectedMediaContract:
    """Exact decoded-media contract shared by both sides of an A/B."""

    frames: int = EXPECTED_FRAMES
    width: int = EXPECTED_WIDTH
    height: int = EXPECTED_HEIGHT
    fps: float = EXPECTED_FPS
    audio_rate: int = EXPECTED_AUDIO_RATE
    audio_channels: int = EXPECTED_AUDIO_CHANNELS
    audio_channel_layout: str = EXPECTED_AUDIO_CHANNEL_LAYOUT
    audio_sample_frames: int = EXPECTED_AUDIO_SAMPLE_FRAMES
    pixel_format: str = EXPECTED_PIXEL_FORMAT
    start_time_s: float = EXPECTED_START_TIME


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True, help="matched reference MP4")
    parser.add_argument("--candidate", type=Path, required=True, help="matched candidate MP4")
    parser.add_argument("--output", type=Path, required=True, help="JSON report to create")
    parser.add_argument("--seed", type=int, default=1101)
    parser.add_argument("--reference-label", default="full_vae")
    parser.add_argument("--candidate-label", default="taeh3")
    parser.add_argument(
        "--comparison-axis",
        choices=("decoder", "precision", "combined"),
        default=DEFAULT_COMPARISON_AXIS,
        help="single changed axis represented by this A/B (default: decoder)",
    )
    parser.add_argument(
        "--comparison-id",
        default=DEFAULT_COMPARISON_ID,
        help="stable machine-readable ID for this comparison",
    )
    parser.add_argument(
        "--reference-contract",
        default=DEFAULT_REFERENCE_CONTRACT,
        help="human-readable execution contract for the reference artifact",
    )
    parser.add_argument(
        "--candidate-contract",
        default=DEFAULT_CANDIDATE_CONTRACT,
        help="human-readable execution contract for the candidate artifact",
    )
    parser.add_argument("--prompt-file", type=Path)
    parser.add_argument("--request-provenance-json", type=Path)
    parser.add_argument("--reference-provenance-json", type=Path)
    parser.add_argument("--candidate-provenance-json", type=Path)
    parser.add_argument("--ffmpeg", type=Path, default=DEFAULT_FFMPEG)
    parser.add_argument("--ffprobe", type=Path, default=DEFAULT_FFPROBE)
    parser.add_argument(
        "--expected-frames",
        type=int,
        default=EXPECTED_FRAMES,
        help=f"exact decoded video-frame count (default: {EXPECTED_FRAMES})",
    )
    parser.add_argument(
        "--expected-width",
        type=int,
        default=EXPECTED_WIDTH,
        help=f"exact decoded video width (default: {EXPECTED_WIDTH})",
    )
    parser.add_argument(
        "--expected-height",
        type=int,
        default=EXPECTED_HEIGHT,
        help=f"exact decoded video height (default: {EXPECTED_HEIGHT})",
    )
    parser.add_argument(
        "--expected-audio-sample-frames",
        type=int,
        default=EXPECTED_AUDIO_SAMPLE_FRAMES,
        help=(
            "exact sample-frame count after canonical stereo-f32/32-kHz decode "
            f"(default: {EXPECTED_AUDIO_SAMPLE_FRAMES})"
        ),
    )
    args = parser.parse_args(argv)
    if not args.reference_label or not args.candidate_label:
        parser.error("artifact labels must be non-empty")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", args.comparison_id):
        parser.error("--comparison-id must contain only letters, digits, '.', '_' or '-'")
    if not args.reference_contract.strip() or not args.candidate_contract.strip():
        parser.error("comparison contracts must be non-empty")
    if args.reference.expanduser().resolve() == args.candidate.expanduser().resolve():
        parser.error("--reference and --candidate must be different files")
    for name in (
        "expected_frames",
        "expected_width",
        "expected_height",
        "expected_audio_sample_frames",
    ):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    return args


def expected_media_contract_from_args(args: argparse.Namespace) -> ExpectedMediaContract:
    return ExpectedMediaContract(
        frames=int(args.expected_frames),
        width=int(args.expected_width),
        height=int(args.expected_height),
        audio_sample_frames=int(args.expected_audio_sample_frames),
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_input(path: Path, label: str, *, suffix: str | None = None) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise ComparisonError(f"{label} is missing: {resolved}")
    if suffix is not None and resolved.suffix.lower() != suffix:
        raise ComparisonError(f"{label} must have suffix {suffix}: {resolved}")
    return resolved


def _resolve_executable(path: Path, label: str) -> Path:
    expanded = path.expanduser()
    candidate = expanded.resolve() if expanded.parent != Path(".") else None
    if candidate is not None and candidate.is_file() and os.access(candidate, os.X_OK):
        return candidate
    found = shutil.which(str(path))
    if found is None:
        raise ComparisonError(f"{label} executable was not found: {path}")
    return Path(found).resolve()


def _run_text(command: Sequence[str], *, timeout: float = 300.0) -> subprocess.CompletedProcess[str]:
    try:
        result = subprocess.run(
            list(command),
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ComparisonError(f"command failed: {command!r}: {type(exc).__name__}: {exc}") from exc
    if result.returncode:
        raise ComparisonError(
            f"command returned {result.returncode}: {command!r}: "
            f"{result.stderr.strip() or '<empty stderr>'}"
        )
    return result


def _run_bytes(command: Sequence[str], *, timeout: float = 300.0) -> bytes:
    try:
        result = subprocess.run(
            list(command),
            check=False,
            capture_output=True,
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ComparisonError(f"command failed: {command!r}: {type(exc).__name__}: {exc}") from exc
    if result.returncode:
        stderr = result.stderr.decode("utf-8", errors="replace").strip()
        raise ComparisonError(
            f"command returned {result.returncode}: {command!r}: {stderr or '<empty stderr>'}"
        )
    return result.stdout


def _fraction(raw: Any, label: str) -> float:
    try:
        value = float(Fraction(str(raw)))
    except (ValueError, ZeroDivisionError) as exc:
        raise ComparisonError(f"invalid {label}: {raw!r}") from exc
    if not math.isfinite(value):
        raise ComparisonError(f"non-finite {label}: {raw!r}")
    return value


def probe_media(
    path: Path,
    ffprobe: Path,
    contract: ExpectedMediaContract | None = None,
) -> dict[str, Any]:
    result = _run_text(
        (
            str(ffprobe),
            "-v",
            "error",
            "-count_frames",
            "-show_entries",
            "format=duration,size,start_time:stream=index,codec_type,codec_name,width,height,pix_fmt,"
            "avg_frame_rate,r_frame_rate,nb_frames,nb_read_frames,duration,start_time,sample_rate,channels,"
            "channel_layout",
            "-of",
            "json",
            str(path),
        )
    )
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise ComparisonError(f"ffprobe returned invalid JSON for {path}") from exc
    return validate_media_payload(
        payload,
        str(path),
        strict_extended_contract=True,
        contract=contract,
    )


def validate_media_payload(
    payload: Mapping[str, Any],
    label: str,
    *,
    strict_extended_contract: bool = False,
    contract: ExpectedMediaContract | None = None,
) -> dict[str, Any]:
    expected = contract or ExpectedMediaContract()
    streams = payload.get("streams")
    if not isinstance(streams, list):
        raise ComparisonError(f"{label}: ffprobe has no stream list")
    videos = [item for item in streams if item.get("codec_type") == "video"]
    audios = [item for item in streams if item.get("codec_type") == "audio"]
    errors: list[str] = []
    if len(videos) != 1:
        errors.append(f"video_streams={len(videos)}")
        video: Mapping[str, Any] = {}
    else:
        video = videos[0]
        if video.get("codec_name") != "h264":
            errors.append(f"video_codec={video.get('codec_name')!r}")
        if strict_extended_contract and video.get("pix_fmt") != expected.pixel_format:
            errors.append(f"video_pix_fmt={video.get('pix_fmt')!r}")
        if (int(video.get("width", 0)), int(video.get("height", 0))) != (
            expected.width,
            expected.height,
        ):
            errors.append(f"video_shape={video.get('width')}x{video.get('height')}")
        if (
            abs(_fraction(video.get("avg_frame_rate", "0/1"), "average frame rate") - expected.fps)
            > 1e-6
        ):
            errors.append(f"video_fps={video.get('avg_frame_rate')!r}")
        frame_raw = video.get("nb_read_frames") or video.get("nb_frames")
        try:
            frame_count = int(frame_raw)
        except (TypeError, ValueError):
            frame_count = 0
        if frame_count != expected.frames:
            errors.append(f"video_frames={frame_raw!r}")
        if strict_extended_contract:
            try:
                video_start_time = _fraction(video.get("start_time"), "video start time")
            except ComparisonError:
                errors.append(f"video_start_time={video.get('start_time')!r}")
            else:
                if abs(video_start_time - expected.start_time_s) > START_TIME_TOLERANCE:
                    errors.append(f"video_start_time={video_start_time!r}")
    if len(audios) != 1:
        errors.append(f"audio_streams={len(audios)}")
        audio: Mapping[str, Any] = {}
    else:
        audio = audios[0]
        if audio.get("codec_name") != "aac":
            errors.append(f"audio_codec={audio.get('codec_name')!r}")
        if int(audio.get("sample_rate", 0)) != expected.audio_rate:
            errors.append(f"audio_rate={audio.get('sample_rate')!r}")
        if int(audio.get("channels", 0)) != expected.audio_channels:
            errors.append(f"audio_channels={audio.get('channels')!r}")
        if strict_extended_contract and audio.get("channel_layout") != expected.audio_channel_layout:
            errors.append(f"audio_channel_layout={audio.get('channel_layout')!r}")
        if strict_extended_contract:
            try:
                audio_start_time = _fraction(audio.get("start_time"), "audio start time")
            except ComparisonError:
                errors.append(f"audio_start_time={audio.get('start_time')!r}")
            else:
                if abs(audio_start_time - expected.start_time_s) > START_TIME_TOLERANCE:
                    errors.append(f"audio_start_time={audio_start_time!r}")
    format_payload = payload.get("format")
    if not isinstance(format_payload, Mapping):
        errors.append("missing_format")
        format_payload = {}
    try:
        duration = float(format_payload.get("duration", 0.0))
    except (TypeError, ValueError):
        duration = 0.0
    expected_duration = expected.frames / expected.fps
    if not math.isfinite(duration) or abs(duration - expected_duration) > 0.30:
        errors.append(f"duration={duration!r}")
    if strict_extended_contract:
        try:
            format_start_time = _fraction(format_payload.get("start_time"), "container start time")
        except ComparisonError:
            errors.append(f"format_start_time={format_payload.get('start_time')!r}")
        else:
            if abs(format_start_time - expected.start_time_s) > START_TIME_TOLERANCE:
                errors.append(f"format_start_time={format_start_time!r}")
    if errors:
        raise ComparisonError(f"{label}: exact media contract failed: {'; '.join(errors)}")
    return {
        "status": "passed",
        "video": dict(video),
        "audio": dict(audio),
        "format": dict(format_payload),
        "validated": {
            "frames": expected.frames,
            "width": expected.width,
            "height": expected.height,
            "fps": expected.fps,
            "pixel_format": expected.pixel_format if strict_extended_contract else video.get("pix_fmt"),
            "audio_rate": expected.audio_rate,
            "audio_channels": expected.audio_channels,
            "audio_channel_layout": (
                expected.audio_channel_layout if strict_extended_contract else audio.get("channel_layout")
            ),
            "start_time_s": expected.start_time_s if strict_extended_contract else format_payload.get("start_time"),
            "duration_s": duration,
            "extended_contract_enforced": strict_extended_contract,
        },
    }


def parse_stats_lines(text: str, *, expected_frames: int = EXPECTED_FRAMES) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        row: dict[str, float] = {}
        for match in _STATS_FIELD_RE.finditer(line):
            raw = match.group("value").lower()
            value = float(raw)
            if math.isnan(value):
                raise ComparisonError(f"metric stats contain NaN: {line!r}")
            row[match.group("name")] = value
        if "n" not in row:
            raise ComparisonError(f"metric stats line has no frame number: {line!r}")
        rows.append(row)
    if len(rows) != expected_frames:
        raise ComparisonError(f"metric stats contain {len(rows)}/{expected_frames} frames")
    indices = [int(row["n"]) for row in rows]
    if indices != list(range(1, expected_frames + 1)):
        raise ComparisonError(f"metric frame indices are not contiguous: {indices[:8]}...{indices[-8:]}")
    return rows


def _json_float(value: float) -> float | None:
    return float(value) if math.isfinite(value) else None


def _percentile(sorted_values: Sequence[float], quantile: float) -> float:
    """Return a deterministic linearly interpolated percentile."""

    if not sorted_values:
        raise ComparisonError("cannot compute a percentile of an empty metric")
    if not 0.0 <= quantile <= 1.0:
        raise ComparisonError(f"percentile quantile is outside [0, 1]: {quantile}")
    position = (len(sorted_values) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(sorted_values[lower])
    fraction = position - lower
    return float(sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction)


def summarize_finite(values: Sequence[float]) -> dict[str, Any]:
    if not values:
        raise ComparisonError("cannot summarize an empty metric")
    finite = sorted(float(value) for value in values if math.isfinite(value))
    infinite_count = sum(math.isinf(value) for value in values)
    if not finite:
        return {
            "count": len(values),
            "finite_count": 0,
            "positive_infinity_count": infinite_count,
            "minimum": None,
            "p05": None,
            "p50": None,
            "p95": None,
            "mean": None,
            "maximum": None,
        }
    return {
        "count": len(values),
        "finite_count": len(finite),
        "positive_infinity_count": infinite_count,
        "minimum": finite[0],
        "p05": _percentile(finite, 0.05),
        "p50": statistics.median(finite),
        "p95": _percentile(finite, 0.95),
        "mean": statistics.fmean(finite),
        "maximum": None if infinite_count else finite[-1],
    }


def _worst_color_frame(
    frames: Sequence[Mapping[str, Any]],
    metric: str,
    *,
    highest_is_worst: bool,
    fps: float = EXPECTED_FPS,
) -> dict[str, Any] | None:
    eligible = [frame for frame in frames if frame.get(metric) is not None]
    if not eligible:
        return None
    if highest_is_worst:
        selected = max(
            eligible,
            key=lambda frame: (float(frame[metric]), -int(frame["frame_index"])),
        )
    else:
        selected = min(
            eligible,
            key=lambda frame: (float(frame[metric]), int(frame["frame_index"])),
        )
    return {
        "selection_metric": metric,
        "selection_rule": "maximum" if highest_is_worst else "minimum",
        "frame_index": int(selected["frame_index"]),
        "frame_number": int(selected["frame_index"]) + 1,
        "timestamp_s": int(selected["frame_index"]) / fps,
        "mse": float(selected["mse"]),
        "psnr_db": selected["psnr_db"],
        "psnr_is_infinite": bool(selected["psnr_is_infinite"]),
        "ssim": float(selected["ssim"]),
    }


def combine_color_stats(
    psnr_rows: Sequence[Mapping[str, float]],
    ssim_rows: Sequence[Mapping[str, float]],
    *,
    color_space: str,
    fps: float = EXPECTED_FPS,
) -> dict[str, Any]:
    if color_space == "rgb_gbrp":
        psnr_components = ("r", "g", "b")
        ssim_components = ("R", "G", "B")
    elif color_space == "yuv444p":
        psnr_components = ("y", "u", "v")
        ssim_components = ("Y", "U", "V")
    else:
        raise ComparisonError(f"unsupported metric color space: {color_space}")
    if len(psnr_rows) != len(ssim_rows):
        raise ComparisonError("PSNR and SSIM stats have different frame counts")

    frames: list[dict[str, Any]] = []
    for psnr, ssim in zip(psnr_rows, ssim_rows, strict=True):
        frame_number = int(psnr.get("n", 0))
        if frame_number != int(ssim.get("n", -1)):
            raise ComparisonError(f"PSNR/SSIM frame mismatch at {frame_number}")
        required_psnr = ("mse_avg", "psnr_avg", *(f"psnr_{name}" for name in psnr_components))
        required_ssim = ("All", *ssim_components)
        missing = [name for name in (*required_psnr, *required_ssim) if name not in psnr and name not in ssim]
        if missing:
            raise ComparisonError(f"{color_space} frame {frame_number} is missing stats: {missing}")
        psnr_average = float(psnr["psnr_avg"])
        frames.append(
            {
                "frame_index": frame_number - 1,
                "mse": float(psnr["mse_avg"]),
                "psnr_db": _json_float(psnr_average),
                "psnr_is_infinite": math.isinf(psnr_average),
                "psnr_components_db": {
                    name: _json_float(float(psnr[f"psnr_{name}"])) for name in psnr_components
                },
                "ssim": float(ssim["All"]),
                "ssim_components": {
                    name.lower(): float(ssim[name]) for name in ssim_components
                },
            }
        )
    mse_values = [float(row["mse_avg"]) for row in psnr_rows]
    mean_mse = statistics.fmean(mse_values)
    global_psnr = math.inf if mean_mse == 0.0 else 10.0 * math.log10((255.0**2) / mean_mse)
    psnr_values = [float(row["psnr_avg"]) for row in psnr_rows]
    ssim_values = [float(row["All"]) for row in ssim_rows]
    return {
        "color_space": color_space,
        "conversion": "FFmpeg format filter; both decoded streams converted identically before comparison",
        "frame_count": len(frames),
        "per_frame": frames,
        "summary": {
            "mean_mse": mean_mse,
            "global_psnr_from_mean_mse_db": _json_float(global_psnr),
            "global_psnr_is_infinite": math.isinf(global_psnr),
            "per_frame_psnr_db": summarize_finite(psnr_values),
            "per_frame_ssim": summarize_finite(ssim_values),
            "worst_frames": {
                "highest_mse": _worst_color_frame(
                    frames,
                    "mse",
                    highest_is_worst=True,
                    fps=fps,
                ),
                "lowest_psnr_db": _worst_color_frame(
                    frames,
                    "psnr_db",
                    highest_is_worst=False,
                    fps=fps,
                ),
                "lowest_ssim": _worst_color_frame(
                    frames,
                    "ssim",
                    highest_is_worst=False,
                    fps=fps,
                ),
            },
        },
    }


def compute_color_metrics(
    reference: Path,
    candidate: Path,
    ffmpeg: Path,
    *,
    color_space: str,
    contract: ExpectedMediaContract | None = None,
) -> dict[str, Any]:
    expected = contract or ExpectedMediaContract()
    pixel_format = "gbrp" if color_space == "rgb_gbrp" else "yuv444p"
    with tempfile.TemporaryDirectory(prefix="h3-decoder-quality-") as temporary:
        directory = Path(temporary)
        psnr_path = directory / "psnr.txt"
        ssim_path = directory / "ssim.txt"
        graph = (
            f"[0:v:0]setpts=N/({expected.fps:g}*TB),format={pixel_format},split=2[refp][refs];"
            f"[1:v:0]setpts=N/({expected.fps:g}*TB),format={pixel_format},split=2[candp][cands];"
            f"[candp][refp]psnr=stats_file={psnr_path}[po];"
            f"[cands][refs]ssim=stats_file={ssim_path}[so]"
        )
        _run_text(
            (
                str(ffmpeg),
                "-hide_banner",
                "-nostdin",
                "-loglevel",
                "error",
                "-i",
                str(reference),
                "-i",
                str(candidate),
                "-filter_complex",
                graph,
                "-map",
                "[po]",
                "-map",
                "[so]",
                "-an",
                "-f",
                "null",
                "-",
            ),
            timeout=900.0,
        )
        psnr = parse_stats_lines(
            psnr_path.read_text(encoding="utf-8"),
            expected_frames=expected.frames,
        )
        ssim = parse_stats_lines(
            ssim_path.read_text(encoding="utf-8"),
            expected_frames=expected.frames,
        )
    return combine_color_stats(
        psnr,
        ssim,
        color_space=color_space,
        fps=expected.fps,
    )


def _iter_luma(path: Path) -> Iterator[NDArray[np.uint8]]:
    try:
        with av.open(str(path), mode="r") as container:
            stream = container.streams.video[0]
            for frame in container.decode(stream):
                yuv = frame.to_ndarray(format="yuv444p")
                yield np.ascontiguousarray(yuv[0])
    except (av.error.FFmpegError, IndexError) as exc:
        raise ComparisonError(f"PyAV could not decode video from {path}: {exc}") from exc


def compute_temporal_luma_records(
    pairs: Iterable[tuple[NDArray[np.uint8], NDArray[np.uint8]]],
    *,
    expected_frames: int | None = None,
) -> dict[str, Any]:
    frame_contract = EXPECTED_FRAMES if expected_frames is None else int(expected_frames)
    previous_reference: NDArray[np.int16] | None = None
    previous_candidate: NDArray[np.int16] | None = None
    records: list[dict[str, float | int]] = []
    frame_count = 0
    for frame_index, (reference_u8, candidate_u8) in enumerate(pairs):
        if reference_u8.shape != candidate_u8.shape:
            raise ComparisonError(
                f"luma frame {frame_index} shape mismatch: {reference_u8.shape} != {candidate_u8.shape}"
            )
        if reference_u8.dtype != np.uint8 or candidate_u8.dtype != np.uint8:
            raise ComparisonError("temporal luma inputs must be uint8")
        reference = reference_u8.astype(np.int16)
        candidate = candidate_u8.astype(np.int16)
        if previous_reference is not None and previous_candidate is not None:
            reference_delta = reference - previous_reference
            candidate_delta = candidate - previous_candidate
            reference_activity = float(np.mean(np.abs(reference_delta), dtype=np.float64))
            candidate_activity = float(np.mean(np.abs(candidate_delta), dtype=np.float64))
            records.append(
                {
                    "from_frame_index": frame_index - 1,
                    "to_frame_index": frame_index,
                    "reference_luma_temporal_mae": reference_activity,
                    "candidate_luma_temporal_mae": candidate_activity,
                    "luma_temporal_activity_abs_delta": abs(candidate_activity - reference_activity),
                    "luma_temporal_derivative_error_mae": float(
                        np.mean(np.abs(candidate_delta - reference_delta), dtype=np.float64)
                    ),
                    "global_luma_step_abs_delta": abs(
                        float(np.mean(candidate_delta, dtype=np.float64))
                        - float(np.mean(reference_delta, dtype=np.float64))
                    ),
                }
            )
        previous_reference = reference
        previous_candidate = candidate
        frame_count += 1
    if frame_count != frame_contract:
        raise ComparisonError(f"temporal decode produced {frame_count}/{frame_contract} frame pairs")
    if len(records) != frame_contract - 1:
        raise ComparisonError("temporal metric did not produce one record per frame transition")
    metric_names = (
        "reference_luma_temporal_mae",
        "candidate_luma_temporal_mae",
        "luma_temporal_activity_abs_delta",
        "luma_temporal_derivative_error_mae",
        "global_luma_step_abs_delta",
    )
    return {
        "definition": {
            "luma_source": "PyAV yuv444p Y plane after decode",
            "activity_abs_delta": "abs(MAE(candidate[t]-candidate[t-1]) - MAE(reference[t]-reference[t-1]))",
            "derivative_error": "MAE((candidate[t]-candidate[t-1]) - (reference[t]-reference[t-1]))",
            "global_step_abs_delta": "absolute difference between mean luma changes for the two videos",
            "interpretation": "transparent temporal/flicker proxies; not a learned perceptual metric",
        },
        "frame_count": frame_count,
        "transition_count": len(records),
        "per_transition": records,
        "summary": {
            name: summarize_finite([float(record[name]) for record in records]) for name in metric_names
        },
    }


def compute_temporal_metrics(
    reference: Path,
    candidate: Path,
    *,
    expected_frames: int = EXPECTED_FRAMES,
) -> dict[str, Any]:
    sentinel = object()

    def paired_luma() -> Iterator[tuple[NDArray[np.uint8], NDArray[np.uint8]]]:
        # Stream in lockstep.  At most the current and previous pair remains in
        # memory; zip_longest also catches an asymmetric decoder termination.
        for frame_index, pair in enumerate(
            itertools.zip_longest(_iter_luma(reference), _iter_luma(candidate), fillvalue=sentinel)
        ):
            reference_frame, candidate_frame = pair
            if reference_frame is sentinel or candidate_frame is sentinel:
                raise ComparisonError(f"video decoder frame-count mismatch at frame {frame_index}")
            yield reference_frame, candidate_frame

    return compute_temporal_luma_records(
        paired_luma(),
        expected_frames=expected_frames,
    )


def _decode_audio_f32le(
    path: Path,
    ffmpeg: Path,
    *,
    audio_channels: int = EXPECTED_AUDIO_CHANNELS,
    audio_rate: int = EXPECTED_AUDIO_RATE,
) -> bytes:
    return _run_bytes(
        (
            str(ffmpeg),
            "-hide_banner",
            "-nostdin",
            "-loglevel",
            "error",
            "-i",
            str(path),
            "-map",
            "0:a:0",
            "-vn",
            "-ac",
            str(audio_channels),
            "-ar",
            str(audio_rate),
            "-c:a",
            "pcm_f32le",
            "-f",
            "f32le",
            "-",
        ),
        timeout=300.0,
    )


def compare_audio_pcm(
    reference_bytes: bytes,
    candidate_bytes: bytes,
    *,
    expected_sample_frames: int | None = None,
    audio_channels: int = EXPECTED_AUDIO_CHANNELS,
    audio_rate: int = EXPECTED_AUDIO_RATE,
) -> dict[str, Any]:
    stride = np.dtype("<f4").itemsize * audio_channels
    if len(reference_bytes) % stride or len(candidate_bytes) % stride:
        raise ComparisonError("canonical stereo-f32 PCM byte length is not frame-aligned")
    reference = np.frombuffer(reference_bytes, dtype="<f4").reshape(-1, audio_channels)
    candidate = np.frombuffer(candidate_bytes, dtype="<f4").reshape(-1, audio_channels)
    if not np.isfinite(reference).all() or not np.isfinite(candidate).all():
        raise ComparisonError("canonical decoded PCM contains NaN or infinity")
    if len(reference) != len(candidate):
        raise ComparisonError(
            "canonical decoded PCM sample-frame mismatch: "
            f"reference={len(reference)} candidate={len(candidate)}"
        )
    if expected_sample_frames is not None:
        if expected_sample_frames <= 0:
            raise ComparisonError(f"expected audio sample-frame count must be positive: {expected_sample_frames}")
        if len(reference) != expected_sample_frames:
            raise ComparisonError(
                "canonical decoded PCM violates the exact sample-frame contract: "
                f"reference={len(reference)} candidate={len(candidate)} expected={expected_sample_frames}"
            )
    overlap_frames = min(len(reference), len(candidate))
    if overlap_frames:
        difference = candidate[:overlap_frames].astype(np.float64) - reference[:overlap_frames]
        absolute = np.abs(difference)
        mae = float(np.mean(absolute))
        max_abs = float(np.max(absolute))
        rmse = float(np.sqrt(np.mean(np.square(difference))))
        reference_rms = float(np.sqrt(np.mean(np.square(reference[:overlap_frames], dtype=np.float64))))
        snr = math.inf if rmse == 0.0 else 20.0 * math.log10(reference_rms / rmse) if reference_rms else -math.inf
    else:
        mae = max_abs = rmse = reference_rms = 0.0
        snr = -math.inf
    return {
        "canonical_format": (
            f"{'stereo' if audio_channels == 2 else f'{audio_channels}-channel'} "
            f"interleaved little-endian float32 at {audio_rate} Hz"
        ),
        "reference_pcm_sha256": hashlib.sha256(reference_bytes).hexdigest(),
        "candidate_pcm_sha256": hashlib.sha256(candidate_bytes).hexdigest(),
        "pcm_bitwise_identical": reference_bytes == candidate_bytes,
        "reference_sample_frames": len(reference),
        "candidate_sample_frames": len(candidate),
        "expected_sample_frames": expected_sample_frames,
        "sample_frame_contract_passed": (
            len(reference) == len(candidate)
            and (expected_sample_frames is None or len(reference) == expected_sample_frames)
        ),
        "overlap_sample_frames": overlap_frames,
        "sample_frame_count_delta": len(candidate) - len(reference),
        "overlap_mae": mae,
        "overlap_rmse": rmse,
        "overlap_max_abs": max_abs,
        "reference_rms": reference_rms,
        "overlap_snr_db": _json_float(snr),
        "overlap_snr_is_positive_infinite": snr == math.inf,
    }


def compare_audio(
    reference: Path,
    candidate: Path,
    ffmpeg: Path,
    contract: ExpectedMediaContract | None = None,
) -> dict[str, Any]:
    expected = contract or ExpectedMediaContract()
    return compare_audio_pcm(
        _decode_audio_f32le(
            reference,
            ffmpeg,
            audio_channels=expected.audio_channels,
            audio_rate=expected.audio_rate,
        ),
        _decode_audio_f32le(
            candidate,
            ffmpeg,
            audio_channels=expected.audio_channels,
            audio_rate=expected.audio_rate,
        ),
        expected_sample_frames=expected.audio_sample_frames,
        audio_channels=expected.audio_channels,
        audio_rate=expected.audio_rate,
    )


def _file_identity(path: Path, label: str) -> dict[str, Any]:
    resolved = _resolve_input(path, label)
    return {
        "path": str(resolved),
        "size_bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def _optional_artifact(path: Path | None, label: str) -> dict[str, Any] | None:
    if path is None:
        return None
    identity = _file_identity(path, label)
    if Path(identity["path"]).suffix.lower() == ".json":
        try:
            parsed = json.loads(Path(identity["path"]).read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ComparisonError(f"{label} is not valid JSON: {identity['path']}") from exc
        identity["json_type"] = type(parsed).__name__
    return identity


def _tool_identity(path: Path) -> dict[str, Any]:
    version = _run_text((str(path), "-version"), timeout=30.0).stdout.splitlines()
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "version_line": version[0] if version else None,
    }


def _write_report(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    expected = expected_media_contract_from_args(args)
    output = args.output.expanduser().resolve()
    payload: dict[str, Any] = {
        "schema_version": 1,
        "status": "initializing",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": f"CPU-only post-codec MiniMax-H3 {args.comparison_axis} A/B comparison",
        "comparison": {
            "axis": args.comparison_axis,
            "id": args.comparison_id,
            "reference_contract": args.reference_contract,
            "candidate_contract": args.candidate_contract,
        },
        "gpu_clock_policy": GPU_CLOCK_POLICY,
        "gpu_clock_policy_id": GPU_CLOCK_POLICY_ID,
        "gpu_clock_mutation": False,
        "quality_contract": {
            "reference": args.reference_contract,
            "candidate": args.candidate_contract,
            "thresholds": None,
            "lpips": "not computed; this tool does not declare an LPIPS model/dependency contract",
        },
        "expected_media_contract": asdict(expected),
        "request": {"seed": args.seed},
    }
    _write_report(output, payload)
    try:
        reference = _resolve_input(args.reference, "reference MP4", suffix=".mp4")
        candidate = _resolve_input(args.candidate, "candidate MP4", suffix=".mp4")
        ffmpeg = _resolve_executable(args.ffmpeg, "ffmpeg")
        ffprobe = _resolve_executable(args.ffprobe, "ffprobe")
        payload["tools"] = {
            "ffmpeg": _tool_identity(ffmpeg),
            "ffprobe": _tool_identity(ffprobe),
            "pyav_version": av.__version__,
            "numpy_version": np.__version__,
            "script": _file_identity(Path(__file__).resolve(), "comparison script"),
            "gpu_used": False,
            "clock_or_power_mutation": False,
        }
        payload["inputs"] = {
            "reference": {
                "label": args.reference_label,
                **_file_identity(reference, "reference MP4"),
                "provenance_json": _optional_artifact(
                    args.reference_provenance_json,
                    "reference provenance JSON",
                ),
            },
            "candidate": {
                "label": args.candidate_label,
                **_file_identity(candidate, "candidate MP4"),
                "provenance_json": _optional_artifact(
                    args.candidate_provenance_json,
                    "candidate provenance JSON",
                ),
            },
        }
        payload["request"].update(
            {
                "prompt_file": _optional_artifact(args.prompt_file, "prompt file"),
                "provenance_json": _optional_artifact(
                    args.request_provenance_json,
                    "request provenance JSON",
                ),
            }
        )
        payload["media"] = {
            "reference": probe_media(reference, ffprobe, expected),
            "candidate": probe_media(candidate, ffprobe, expected),
        }
        payload["metrics"] = {
            "rgb": compute_color_metrics(
                reference,
                candidate,
                ffmpeg,
                color_space="rgb_gbrp",
                contract=expected,
            ),
            "yuv": compute_color_metrics(
                reference,
                candidate,
                ffmpeg,
                color_space="yuv444p",
                contract=expected,
            ),
            "temporal": compute_temporal_metrics(
                reference,
                candidate,
                expected_frames=expected.frames,
            ),
            "audio": compare_audio(reference, candidate, ffmpeg, expected),
        }
        payload["status"] = "passed"
        payload["completed_utc"] = datetime.now(timezone.utc).isoformat()
        _write_report(output, payload)
        marker = (
            "MINIMAX_H3_DECODER_QUALITY_COMPARE_V1"
            if args.comparison_axis == DEFAULT_COMPARISON_AXIS
            and args.comparison_id == DEFAULT_COMPARISON_ID
            else "MINIMAX_H3_MP4_QUALITY_COMPARE_V2"
        )
        print(
            f"{marker} "
            f"comparison_axis={args.comparison_axis} "
            f"comparison_id={args.comparison_id} "
            f"frames={expected.frames} "
            f"rgb_ssim={payload['metrics']['rgb']['summary']['per_frame_ssim']['mean']:.6f} "
            f"yuv_ssim={payload['metrics']['yuv']['summary']['per_frame_ssim']['mean']:.6f} "
            f"audio_identity={str(payload['metrics']['audio']['pcm_bitwise_identical']).lower()}"
        )
        print(f"Report: {output}")
        return 0
    except BaseException as exc:
        payload["status"] = "failed"
        payload["error"] = f"{type(exc).__name__}: {exc}"
        payload["completed_utc"] = datetime.now(timezone.utc).isoformat()
        _write_report(output, payload)
        print(f"H3 decoder quality comparison failed: {payload['error']}", file=sys.stderr)
        print(f"Report: {output}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
