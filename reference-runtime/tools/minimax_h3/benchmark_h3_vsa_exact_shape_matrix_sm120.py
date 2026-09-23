#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Qualify the exact SM120 VSA fine kernel across MiniMax-H3 shapes.

This benchmark intentionally keeps the VSA algorithm fixed: tile size 64,
the supplied coarse-stage top-k block set, and threshold-zero online softmax.
It varies only request geometry and execution shape.  Positive skip-softmax
thresholds are not accepted here because they are an approximation and belong
to a separate quality lane.

The presets cover real 5/10/15-second H3 grids, landscape/portrait/square and
ragged grids, prefix ratios, sparse budgets, local head counts, and batch size.
Run every kernel candidate in a fresh process, then compare the per-case output
hashes (or the optional saved outputs) against the exact baseline report.

The tool never changes clocks or power limits.  It fails closed when a foreign
compute process is present on the selected physical GPU.
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
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


BLOCK_SIZE = 64
HEAD_DIM = 128
SOFTMAX_SCALE = HEAD_DIM**-0.5
DTYPE = torch.bfloat16
PREFIX_5S = (121, 414)
PREFIX_10S = (121, 810)
PREFIX_15S = (121, 1206)
PREFIX_FASTVIDEO_15S = (121, 1150)
BALANCE_GATE_ENV = "VLLM_OMNI_GPU_BALANCE_HEALTH_GATE"
SKIP_THRESHOLD_ENV = "VLLM_OMNI_FASTVIDEO_VSA_SKIP_SOFTMAX_THRESHOLD_SCALE_FACTOR"
CUTEDSL_CACHE_ENV = "FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED"
GPU_CLOCK_POLICY = "node-default/dynamic/unlocked"

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = REPO_ROOT.parents[1]
DEFAULT_FLASHINFER_ROOT = WORKSPACE_ROOT / "fi-pr4944-exact-general"
DEFAULT_RESULTS_ROOT = WORKSPACE_ROOT / "results" / "profile"


class QualificationError(RuntimeError):
    """A fail-closed qualification-contract violation."""


@dataclass(frozen=True)
class ShapeCase:
    name: str
    video_shape_thw: tuple[int, int, int] | None
    prefix_segments: tuple[int, ...] = PREFIX_15S
    video_topk: int | None = None
    local_heads: int = 7
    batch: int = 1
    flat_video_blocks: int | None = None
    selection_pattern: str = "coarse_topk_sorted"
    note: str = ""

    def validate(self) -> None:
        if not self.name:
            raise ValueError("case name must not be empty")
        if (self.video_shape_thw is None) == (self.flat_video_blocks is None):
            raise ValueError(
                f"{self.name}: exactly one of video_shape_thw and flat_video_blocks is required"
            )
        if self.video_shape_thw is not None and any(v <= 0 for v in self.video_shape_thw):
            raise ValueError(f"{self.name}: video dimensions must be positive")
        if self.flat_video_blocks is not None and self.flat_video_blocks <= 0:
            raise ValueError(f"{self.name}: flat_video_blocks must be positive")
        if any(v <= 0 for v in self.prefix_segments):
            raise ValueError(f"{self.name}: prefix segments must be positive")
        if self.video_topk is not None and self.video_topk <= 0:
            raise ValueError(f"{self.name}: video_topk must be positive")
        if self.local_heads <= 0 or self.batch <= 0:
            raise ValueError(f"{self.name}: local_heads and batch must be positive")
        if self.selection_pattern not in {
            "coarse_topk_sorted",
            "cyclic_sorted",
            "legacy_cyclic_unsorted",
        }:
            raise ValueError(f"{self.name}: unsupported selection pattern {self.selection_pattern!r}")


def _default_topk(video_blocks: int) -> int:
    # This deliberately matches FastVideo's compute_topk(0.9, video_blocks).
    return max(1, min(math.ceil((1.0 - 0.9) * video_blocks), video_blocks))


def _case(
    name: str,
    shape: tuple[int, int, int],
    *,
    prefix: tuple[int, ...] = PREFIX_15S,
    topk: int | None = None,
    heads: int = 7,
    batch: int = 1,
    selection_pattern: str = "coarse_topk_sorted",
    note: str = "",
) -> ShapeCase:
    return ShapeCase(
        name=name,
        video_shape_thw=shape,
        prefix_segments=prefix,
        video_topk=topk,
        local_heads=heads,
        batch=batch,
        selection_pattern=selection_pattern,
        note=note,
    )


def preset_cases(preset: str) -> list[ShapeCase]:
    smoke = [
        _case(
            "tiny_dense",
            (4, 4, 4),
            prefix=(3, 65),
            topk=1,
            heads=1,
            note="small dense oracle case with segment-pure ragged prefix",
        ),
        _case(
            "tiny_ragged_sparse",
            (5, 6, 7),
            prefix=(17, 70),
            topk=3,
            heads=2,
            note="ragged T/H/W edges and variable block sizes",
        ),
        _case(
            "small_no_prefix",
            (9, 8, 12),
            prefix=(),
            topk=3,
            heads=7,
            note="pure-video path",
        ),
    ]
    core = [
        ShapeCase(
            "legacy_pr_topk64",
            None,
            prefix_segments=(512,),
            video_topk=64,
            local_heads=7,
            flat_video_blocks=1705,
            selection_pattern="legacy_cyclic_unsorted",
            note="historical PR shape: S=109632, 8 prefix blocks, top-k 64",
        ),
        _case(
            "h3_720p_5s_landscape",
            (37, 22, 40),
            prefix=PREFIX_5S,
            note="124-frame landscape request geometry",
        ),
        _case(
            "h3_720p_5s_portrait",
            (37, 40, 22),
            prefix=PREFIX_5S,
            note="124-frame portrait request geometry",
        ),
        _case(
            "h3_square_5s",
            (37, 30, 30),
            prefix=PREFIX_5S,
            note="124-frame 960x960-like square geometry",
        ),
        _case(
            "h3_720p_10s_landscape",
            (72, 22, 40),
            prefix=PREFIX_10S,
            note="243-frame landscape request geometry",
        ),
        _case(
            "h3_720p_10s_portrait",
            (72, 40, 22),
            prefix=PREFIX_10S,
            note="243-frame portrait request geometry",
        ),
        _case(
            "h3_square_10s",
            (72, 30, 30),
            prefix=PREFIX_10S,
            note="243-frame 960x960-like square geometry",
        ),
        _case(
            "h3_720p_15s_landscape",
            (107, 22, 40),
            prefix=PREFIX_15S,
            note="362-frame vLLM landscape geometry",
        ),
        _case(
            "h3_720p_15s_portrait",
            (107, 40, 22),
            prefix=PREFIX_15S,
            note="362-frame portrait transpose",
        ),
        _case(
            "h3_square_15s",
            (107, 30, 30),
            prefix=PREFIX_15S,
            note="362-frame 960x960-like square geometry",
        ),
        _case(
            "h3_768p_15s_fastvideo",
            (102, 24, 42),
            prefix=PREFIX_FASTVIDEO_15S,
            note="345-frame FastVideo geometry",
        ),
        _case(
            "h3_768p_15s_vllm",
            (107, 24, 42),
            prefix=PREFIX_15S,
            note="362-frame 1344x768 geometry",
        ),
        _case(
            "h3_720p_15s_topk64",
            (107, 22, 40),
            topk=64,
            note="scheduler sensitivity: same geometry, smaller sparse budget",
        ),
        _case(
            "h3_720p_15s_large_prefix",
            (107, 22, 40),
            prefix=(121, 1206, 4097),
            note="dense-prefix load imbalance stress",
        ),
    ]
    extended = [
        ShapeCase(
            "legacy_pr_topk64_sorted",
            None,
            prefix_segments=(512,),
            video_topk=64,
            local_heads=7,
            flat_video_blocks=1705,
            selection_pattern="cyclic_sorted",
            note="historical shape with the production ordered-q2k contract",
        ),
        _case("h3_720p_15s_no_prefix", (107, 22, 40), prefix=(), note="pure-video long path"),
        _case(
            "h3_720p_15s_topk32",
            (107, 22, 40),
            topk=32,
            note="very sparse scheduler stress; kernel qualification only",
        ),
        _case(
            "h3_720p_15s_topk324",
            (107, 22, 40),
            topk=324,
            note="80-percent sparse scheduler stress; kernel qualification only",
        ),
        _case(
            "h3_720p_15s_local_sorted",
            (107, 22, 40),
            selection_pattern="cyclic_sorted",
            note="same exact sparse budget with a locality-heavy ordered mask",
        ),
        _case(
            "h3_720p_5s_dense",
            (37, 22, 40),
            prefix=PREFIX_5S,
            topk=600,
            note="dense boundary on short production geometry",
        ),
        _case(
            "h3_720p_5s_sp4_heads",
            (37, 22, 40),
            prefix=PREFIX_5S,
            heads=14,
            note="SP4 local-head geometry",
        ),
        _case(
            "h3_720p_5s_sp2_heads",
            (37, 22, 40),
            prefix=PREFIX_5S,
            heads=28,
            note="SP2 local-head geometry",
        ),
        _case(
            "h3_720p_5s_sp1_heads",
            (37, 22, 40),
            prefix=PREFIX_5S,
            heads=56,
            note="single-rank head geometry",
        ),
        _case(
            "h3_720p_5s_batch2",
            (37, 22, 40),
            prefix=PREFIX_5S,
            batch=2,
            note="batch-two API stress",
        ),
        _case(
            "h3_ultrawide_10s",
            (72, 20, 48),
            prefix=PREFIX_10S,
            note="ultrawide spatial lattice",
        ),
        _case("h3_ragged_all_axes", (39, 23, 41), note="non-multiple-of-four T/H/W"),
    ]
    if preset == "smoke":
        cases = smoke
    elif preset == "core":
        cases = smoke + core
    elif preset == "full":
        cases = smoke + core + extended
    elif preset == "ncu":
        wanted = {
            "h3_720p_15s_landscape",
            "h3_square_10s",
            "h3_720p_5s_sp4_heads",
        }
        cases = [case for case in core + extended if case.name in wanted]
    else:
        raise ValueError(f"unknown preset: {preset}")
    for case in cases:
        case.validate()
    return cases


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flashinfer-root", type=Path, default=DEFAULT_FLASHINFER_ROOT)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--preset", choices=("smoke", "core", "full", "ncu"), default="core")
    parser.add_argument("--case", action="append", default=[], help="run only named preset cases")
    parser.add_argument("--reverse", action="store_true", help="reverse case order for thermal-order A/B")
    parser.add_argument("--physical-gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1101)
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--cold-l2", action="store_true")
    parser.add_argument("--l2-flush-mib", type=int, default=256)
    parser.add_argument("--oracle-max-blocks", type=int, default=32)
    parser.add_argument("--reference-report", type=Path)
    parser.add_argument("--save-output-dir", type=Path)
    parser.add_argument("--reference-output-dir", type=Path)
    parser.add_argument("--allow-non-bitwise", action="store_true")
    parser.add_argument(
        "--skip-pad-poison",
        action="store_true",
        help="skip the exact ragged-padding isolation check",
    )
    parser.add_argument("--nvidia-smi", default="nvidia-smi")
    args = parser.parse_args(argv)
    if args.physical_gpu < 0 or args.warmups < 0 or args.repeats < 1:
        parser.error("GPU index/warmups/repeats are invalid")
    if args.l2_flush_mib < 1 or args.oracle_max_blocks < 0:
        parser.error("L2 flush size must be positive and oracle limit non-negative")
    cases = preset_cases(args.preset)
    if args.case:
        requested = set(args.case)
        known = {case.name for case in cases}
        unknown = requested - known
        if unknown:
            parser.error(f"case(s) not in preset {args.preset}: {sorted(unknown)}")
        cases = [case for case in cases if case.name in requested]
    if args.reverse:
        cases.reverse()
    args.cases = cases
    return args


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_case_seed(base_seed: int, name: str) -> int:
    suffix = int.from_bytes(hashlib.sha256(name.encode()).digest()[:4], "little")
    return (base_seed + suffix) % (2**63 - 1)


def split_prefix_sizes(prefix_segments: Sequence[int]) -> list[int]:
    result: list[int] = []
    for segment in prefix_segments:
        full, remainder = divmod(segment, BLOCK_SIZE)
        result.extend([BLOCK_SIZE] * full)
        if remainder:
            result.append(remainder)
    return result


def video_block_sizes(case: ShapeCase) -> list[int]:
    if case.flat_video_blocks is not None:
        return [BLOCK_SIZE] * case.flat_video_blocks
    assert case.video_shape_thw is not None
    t, h, w = case.video_shape_thw
    result = []
    for t0 in range(0, t, 4):
        for h0 in range(0, h, 4):
            for w0 in range(0, w, 4):
                result.append(min(4, t - t0) * min(4, h - h0) * min(4, w - w0))
    return result


def case_geometry(case: ShapeCase) -> dict[str, Any]:
    prefix_sizes = split_prefix_sizes(case.prefix_segments)
    video_sizes = video_block_sizes(case)
    topk = min(case.video_topk or _default_topk(len(video_sizes)), len(video_sizes))
    sizes = prefix_sizes + video_sizes
    logical_blocks = len(sizes)
    valid_rows = sum(sizes)
    active_per_head = len(prefix_sizes) * logical_blocks + len(video_sizes) * (len(prefix_sizes) + topk)
    return {
        "batch": case.batch,
        "prefix_segments": list(case.prefix_segments),
        "video_shape_thw": list(case.video_shape_thw) if case.video_shape_thw else None,
        "flat_video_blocks": case.flat_video_blocks,
        "selection_pattern": case.selection_pattern,
        "block_size": BLOCK_SIZE,
        "head_dim": HEAD_DIM,
        "local_heads": case.local_heads,
        "prefix_blocks": len(prefix_sizes),
        "video_blocks": len(video_sizes),
        "logical_blocks": logical_blocks,
        "video_topk": topk,
        "video_active_kv_blocks": len(prefix_sizes) + topk,
        "tiled_rows": logical_blocks * BLOCK_SIZE,
        "valid_rows": valid_rows,
        "sp8_aligned_rows": math.ceil(valid_rows / 8) * 8,
        "ragged_blocks": sum(size != BLOCK_SIZE for size in sizes),
        "active_block_pairs_per_head": active_per_head,
        "scheduled_density": active_per_head / (logical_blocks * logical_blocks),
        "scheduled_attention_flops": (
            4 * active_per_head * BLOCK_SIZE * BLOCK_SIZE * case.local_heads * HEAD_DIM * case.batch
        ),
        "dtype": "bfloat16",
        "softmax_scale": SOFTMAX_SCALE,
    }


def make_inputs(
    case: ShapeCase,
    geometry: Mapping[str, Any],
    device: torch.device,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    shape = (
        case.batch,
        int(geometry["tiled_rows"]),
        case.local_heads,
        HEAD_DIM,
    )
    generator = torch.Generator(device=device).manual_seed(seed)
    q = torch.randn(shape, generator=generator, dtype=DTYPE, device=device)
    k = torch.randn(shape, generator=generator, dtype=DTYPE, device=device)
    v = torch.randn(shape, generator=generator, dtype=DTYPE, device=device)
    block_sizes = torch.tensor(
        split_prefix_sizes(case.prefix_segments) + video_block_sizes(case),
        dtype=torch.int32,
        device=device,
    )
    positions = torch.arange(BLOCK_SIZE, device=device)
    valid = (positions[None, :] < block_sizes[:, None]).reshape(-1)
    valid_bshd = valid.view(1, -1, 1, 1)
    q.masked_fill_(~valid_bshd, 0)
    k.masked_fill_(~valid_bshd, 0)
    v.masked_fill_(~valid_bshd, 0)
    return q.contiguous(), k.contiguous(), v.contiguous(), block_sizes.contiguous()


def make_q2k(
    q: torch.Tensor,
    k: torch.Tensor,
    block_sizes: torch.Tensor,
    geometry: Mapping[str, Any],
) -> tuple[torch.Tensor, torch.Tensor]:
    batch, _rows, heads, dim = q.shape
    n_blocks = int(geometry["logical_blocks"])
    prefix = int(geometry["prefix_blocks"])
    video = int(geometry["video_blocks"])
    topk = int(geometry["video_topk"])
    q2k_idx = torch.full(
        (batch, heads, n_blocks, n_blocks),
        -1,
        dtype=torch.int32,
        device=q.device,
    )
    q2k_num = torch.full(
        (batch, heads, n_blocks),
        prefix + topk,
        dtype=torch.int32,
        device=q.device,
    )
    all_indices = torch.arange(n_blocks, dtype=torch.int32, device=q.device)
    if topk == video:
        q2k_idx.copy_(all_indices)
        q2k_num.fill_(n_blocks)
        return q2k_idx.contiguous(), q2k_num.contiguous()
    if prefix:
        q2k_idx[..., :prefix, :] = all_indices
        q2k_num[..., :prefix] = n_blocks
        q2k_idx[..., prefix:, :prefix] = all_indices[:prefix]
    selection_pattern = str(geometry["selection_pattern"])
    if selection_pattern == "coarse_topk_sorted":
        q_pool = q.view(batch, n_blocks, BLOCK_SIZE, heads, dim).sum(dim=2, dtype=torch.float32)
        k_pool = k.view(batch, n_blocks, BLOCK_SIZE, heads, dim).sum(dim=2, dtype=torch.float32)
        divisor = block_sizes.view(1, -1, 1, 1).clamp_min(1)
        q_pool = (q_pool / divisor).permute(0, 2, 1, 3)
        k_pool = (k_pool / divisor).permute(0, 2, 1, 3)
        video_scores = torch.matmul(
            q_pool[..., prefix:, :],
            k_pool[..., prefix:, :].transpose(-2, -1),
        ) * SOFTMAX_SCALE
        selected = video_scores.topk(topk, dim=-1).indices.to(torch.int32).sort(dim=-1).values
    else:
        starts = torch.arange(video, dtype=torch.int32, device=q.device).unsqueeze(-1)
        offsets = torch.arange(topk, dtype=torch.int32, device=q.device).unsqueeze(0)
        selected = ((starts + offsets) % video).view(1, 1, video, topk)
        if selection_pattern == "cyclic_sorted":
            selected = selected.sort(dim=-1).values
        selected = selected.expand(batch, heads, -1, -1)
    q2k_idx[..., prefix:, prefix : prefix + topk] = selected + prefix
    return q2k_idx.contiguous(), q2k_num.contiguous()


def validate_metadata(
    q2k_idx: torch.Tensor,
    q2k_num: torch.Tensor,
    block_sizes: torch.Tensor,
    geometry: Mapping[str, Any],
) -> None:
    n = int(geometry["logical_blocks"])
    p = int(geometry["prefix_blocks"])
    active = int(geometry["video_active_kv_blocks"])
    expected_idx = (int(geometry["batch"]), int(geometry["local_heads"]), n, n)
    if tuple(q2k_idx.shape) != expected_idx or q2k_idx.dtype != torch.int32:
        raise QualificationError(f"q2k_idx mismatch: {q2k_idx.shape}, {q2k_idx.dtype}")
    if tuple(q2k_num.shape) != expected_idx[:-1] or q2k_num.dtype != torch.int32:
        raise QualificationError(f"q2k_num mismatch: {q2k_num.shape}, {q2k_num.dtype}")
    if block_sizes.numel() != n or int(block_sizes.sum()) != int(geometry["valid_rows"]):
        raise QualificationError("variable block sizes do not match geometry")
    if p and not bool((q2k_num[..., :p] == n).all()):
        raise QualificationError("prefix queries must remain dense")
    if not bool((q2k_num[..., p:] == active).all()):
        raise QualificationError("video queries must retain prefix plus exact top-k")
    counts = q2k_num.unsqueeze(-1)
    slots = torch.arange(n, device=q2k_idx.device).view(1, 1, 1, n)
    used = slots < counts
    if not bool(((q2k_idx >= 0) & (q2k_idx < n) | ~used).all()):
        raise QualificationError("active q2k index is out of range")
    active_idx = q2k_idx.masked_fill(~used, n)
    ordered = geometry["selection_pattern"] != "legacy_cyclic_unsorted"
    if ordered and n > 1 and not bool((active_idx[..., 1:] >= active_idx[..., :-1]).all()):
        raise QualificationError("q2k indices are not monotonically ordered")


def import_kernel(root: Path) -> tuple[Callable[..., Any], Path]:
    root = root.expanduser().resolve()
    wrapper = root / "flashinfer/cute_dsl/sparse/bsa_attn_sm120.py"
    if not wrapper.is_file():
        raise QualificationError(f"missing SM120 VSA wrapper: {wrapper}")
    sys.path.insert(0, str(root))
    importlib.invalidate_caches()
    module = importlib.import_module("flashinfer.cute_dsl.sparse.bsa_attn_sm120")
    module_file = Path(str(module.__file__)).resolve()
    if not module_file.is_relative_to(root):
        raise QualificationError(f"imported FlashInfer from wrong tree: {module_file}")
    kernel = getattr(module, "bsa_attn_sm120_blk64_fwd", None)
    if not callable(kernel):
        raise QualificationError("bsa_attn_sm120_blk64_fwd is unavailable")
    return kernel, module_file


def invoke_kernel(
    kernel: Callable[..., Any],
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q2k_idx: torch.Tensor,
    q2k_num: torch.Tensor,
    block_sizes: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
) -> torch.Tensor:
    result, returned_lse = kernel(
        q,
        k,
        v,
        q2k_idx,
        1,
        block_sizes=block_sizes,
        q2k_block_nums=q2k_num,
        softmax_scale=SOFTMAX_SCALE,
        skip_softmax_threshold_scale_factor=None,
        return_lse=False,
        out=out,
        lse=lse,
    )
    if result.data_ptr() != out.data_ptr() or returned_lse is not None:
        raise QualificationError("output-buffer/no-LSE contract was not honored")
    return result


def _percentile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * fraction
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def summarize(values: Sequence[float]) -> dict[str, Any]:
    mean = statistics.fmean(values)
    stdev = statistics.stdev(values) if len(values) > 1 else 0.0
    return {
        "samples_ms": [float(value) for value in values],
        "count": len(values),
        "min_ms": min(values),
        "p50_ms": statistics.median(values),
        "p90_ms": _percentile(values, 0.9),
        "max_ms": max(values),
        "mean_ms": mean,
        "stdev_ms": stdev,
        "cv": stdev / mean if mean else 0.0,
    }


def canonical_output_cpu(out: torch.Tensor, block_sizes: torch.Tensor) -> torch.Tensor:
    valid = (
        torch.arange(BLOCK_SIZE, device=out.device)[None, :]
        < block_sizes[:, None]
    ).reshape(-1)
    return out[:, valid].detach().cpu().contiguous()


def tensor_sha256(value: torch.Tensor) -> str:
    raw = value.view(torch.uint16).numpy().tobytes(order="C")
    return hashlib.sha256(raw).hexdigest()


def compare_cpu(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, float | bool]:
    if reference.shape != candidate.shape or reference.dtype != candidate.dtype:
        raise QualificationError(
            f"reference output contract differs: {reference.shape}/{reference.dtype} vs "
            f"{candidate.shape}/{candidate.dtype}"
        )
    ref = reference.float()
    cand = candidate.float()
    diff = cand - ref
    ref_l2 = float(ref.square().sum())
    diff_l2 = float(diff.square().sum())
    return {
        "bitwise_equal": bool(torch.equal(reference, candidate)),
        "max_abs": float(diff.abs().max()) if diff.numel() else 0.0,
        "mean_abs": float(diff.abs().mean()) if diff.numel() else 0.0,
        "relative_l2": math.sqrt(diff_l2 / max(ref_l2, sys.float_info.min)),
    }


def pytorch_oracle(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q2k_idx: torch.Tensor,
    q2k_num: torch.Tensor,
    block_sizes: torch.Tensor,
) -> torch.Tensor:
    """Small exact block-sparse reference over valid rows only."""
    batch, _seq, heads, _dim = q.shape
    n_blocks = block_sizes.numel()
    result = torch.zeros_like(q)
    for b in range(batch):
        for h in range(heads):
            for qb in range(n_blocks):
                qn = int(block_sizes[qb])
                count = int(q2k_num[b, h, qb])
                selected = q2k_idx[b, h, qb, :count].tolist()
                qs = q[b, qb * BLOCK_SIZE : qb * BLOCK_SIZE + qn, h].float()
                ks = torch.cat(
                    [k[b, kb * BLOCK_SIZE : kb * BLOCK_SIZE + int(block_sizes[kb]), h] for kb in selected]
                ).float()
                vs = torch.cat(
                    [v[b, kb * BLOCK_SIZE : kb * BLOCK_SIZE + int(block_sizes[kb]), h] for kb in selected]
                ).float()
                probs = torch.softmax(qs @ ks.transpose(0, 1) * SOFTMAX_SCALE, dim=-1)
                result[b, qb * BLOCK_SIZE : qb * BLOCK_SIZE + qn, h] = (probs @ vs).to(DTYPE)
    return result


def _run(command: Sequence[str]) -> str:
    completed = subprocess.run(command, check=False, capture_output=True, text=True, timeout=20)
    if completed.returncode:
        raise QualificationError(
            f"command failed ({completed.returncode}): {command!r}: {completed.stderr.strip()}"
        )
    return completed.stdout


def gpu_snapshot(nvidia_smi: str, physical_gpu: int, allowed_pids: set[int]) -> dict[str, Any]:
    fields = (
        "index,uuid,name,pstate,memory.used,memory.total,utilization.gpu,temperature.gpu,"
        "clocks.current.sm,power.draw,power.limit"
    )
    raw = _run(
        [nvidia_smi, "-i", str(physical_gpu), "--query-gpu=" + fields, "--format=csv,noheader,nounits"]
    )
    rows = list(csv.reader(line for line in raw.splitlines() if line.strip()))
    if len(rows) != 1 or len(rows[0]) != 11:
        raise QualificationError(f"unexpected nvidia-smi GPU row: {rows!r}")
    names = fields.split(",")
    gpu = {name: value.strip() for name, value in zip(names, rows[0], strict=True)}
    app_raw = _run(
        [
            nvidia_smi,
            "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ]
    )
    apps = []
    for row in csv.reader(line for line in app_raw.splitlines() if line.strip()):
        if len(row) != 4 or row[0].strip() != gpu["uuid"]:
            continue
        app = {
            "gpu_uuid": row[0].strip(),
            "pid": int(row[1].strip()),
            "process_name": row[2].strip(),
            "used_gpu_memory_mib": row[3].strip(),
        }
        apps.append(app)
    foreign = [app for app in apps if app["pid"] not in allowed_pids]
    if foreign:
        raise QualificationError(f"foreign compute process present on GPU {physical_gpu}: {foreign}")
    return {"gpu": gpu, "compute_apps": apps, "unix_time": time.time()}


def validate_environment(args: argparse.Namespace) -> None:
    expected_visible = str(args.physical_gpu)
    if os.environ.get("CUDA_VISIBLE_DEVICES") != expected_visible:
        raise QualificationError(
            f"CUDA_VISIBLE_DEVICES must be exactly {expected_visible!r}, got "
            f"{os.environ.get('CUDA_VISIBLE_DEVICES')!r}"
        )
    if os.environ.get(BALANCE_GATE_ENV) != "0":
        raise QualificationError(f"{BALANCE_GATE_ENV} must be explicitly 0")
    if os.environ.get(CUTEDSL_CACHE_ENV) != "0":
        raise QualificationError(f"{CUTEDSL_CACHE_ENV} must be explicitly 0 for source-isolated A/B")
    if os.environ.get(SKIP_THRESHOLD_ENV) not in (None, "", "0", "0.0"):
        raise QualificationError(f"{SKIP_THRESHOLD_ENV} must be unset or zero in the exact lane")


def git_output(root: Path, *args: str) -> str | None:
    completed = subprocess.run(
        ["git", "-C", str(root), *args],
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


def source_provenance(root: Path, module_file: Path) -> dict[str, Any]:
    paths = {
        "wrapper": root / "flashinfer/cute_dsl/sparse/bsa_attn_sm120.py",
        "kernel": root / "flashinfer/cute_dsl/sparse/sm120_blk64/flash_fwd_sm120.py",
        "scheduler": root / "flashinfer/cute_dsl/sparse/sm120_blk64/batched_static_scheduler.py",
        "harness": Path(__file__).resolve(),
    }
    result: dict[str, Any] = {
        "flashinfer_root": str(root),
        "git_head": git_output(root, "rev-parse", "HEAD"),
        "git_status": git_output(root, "status", "--short"),
        "imported_module": str(module_file),
    }
    for name, path in paths.items():
        if not path.is_file():
            raise QualificationError(f"missing provenance file: {path}")
        result[f"{name}_path"] = str(path)
        result[f"{name}_sha256"] = sha256_file(path)
    return result


def load_reference_report(path: Path | None) -> dict[str, Mapping[str, Any]]:
    if path is None:
        return {}
    payload = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    if payload.get("quality_contract", {}).get("skip_softmax_threshold") not in (None, 0, 0.0):
        raise QualificationError("reference report is not an exact threshold-zero report")
    return {str(row["name"]): row for row in payload.get("cases", [])}


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def benchmark_case(
    case: ShapeCase,
    args: argparse.Namespace,
    kernel: Callable[..., Any],
    device: torch.device,
    l2_flush: torch.Tensor | None,
    reference_rows: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    geometry = case_geometry(case)
    case_seed = stable_case_seed(args.seed, case.name)
    q, k, v, block_sizes = make_inputs(case, geometry, device, case_seed)
    q2k_idx, q2k_num = make_q2k(q, k, block_sizes, geometry)
    validate_metadata(q2k_idx, q2k_num, block_sizes, geometry)
    out = torch.empty_like(q)
    lse = torch.empty(
        (case.batch, case.local_heads, int(geometry["tiled_rows"])),
        dtype=torch.float32,
        device=device,
    )

    for _ in range(args.warmups):
        invoke_kernel(kernel, q, k, v, q2k_idx, q2k_num, block_sizes, out, lse)
    torch.cuda.synchronize(device)

    samples = []
    for _ in range(args.repeats):
        if l2_flush is not None:
            l2_flush.zero_()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        invoke_kernel(kernel, q, k, v, q2k_idx, q2k_num, block_sizes, out, lse)
        end.record()
        end.synchronize()
        samples.append(float(start.elapsed_time(end)))

    lse_sentinel = 12345.0
    lse.fill_(lse_sentinel)
    invoke_kernel(kernel, q, k, v, q2k_idx, q2k_num, block_sizes, out, lse)
    torch.cuda.synchronize(device)
    lse_buffer_unchanged = bool((lse == lse_sentinel).all())
    output_cpu = canonical_output_cpu(out, block_sizes)
    output_hash = tensor_sha256(output_cpu)
    timing = summarize(samples)
    row: dict[str, Any] = {
        "name": case.name,
        "note": case.note,
        "case": asdict(case),
        "geometry": geometry,
        "seed": case_seed,
        "timing": timing,
        "scheduled_tflops_at_p50": float(geometry["scheduled_attention_flops"])
        / (float(timing["p50_ms"]) * 1e9),
        "output_shape_valid": list(output_cpu.shape),
        "output_sha256": output_hash,
        "return_lse_false_buffer_unchanged": lse_buffer_unchanged,
    }

    if int(geometry["logical_blocks"]) <= args.oracle_max_blocks:
        oracle = pytorch_oracle(q, k, v, q2k_idx, q2k_num, block_sizes)
        oracle_cpu = canonical_output_cpu(oracle, block_sizes)
        oracle_metrics = compare_cpu(oracle_cpu, output_cpu)
        row["pytorch_oracle"] = oracle_metrics
        if float(oracle_metrics["max_abs"]) > 0.02 or float(oracle_metrics["relative_l2"]) > 0.02:
            raise QualificationError(f"{case.name}: kernel failed PyTorch oracle: {oracle_metrics}")

    reference_row = reference_rows.get(case.name)
    if reference_row is not None:
        reference_hash = reference_row.get("output_sha256")
        matches = output_hash == reference_hash
        row["reference_report"] = {
            "output_sha256": reference_hash,
            "bitwise_equal": matches,
        }
        if not matches and not args.allow_non_bitwise and args.reference_output_dir is None:
            raise QualificationError(f"{case.name}: output hash differs from exact baseline")

    if args.reference_output_dir is not None:
        reference_path = args.reference_output_dir.expanduser().resolve() / f"{case.name}.pt"
        if not reference_path.is_file():
            raise QualificationError(f"missing reference output: {reference_path}")
        reference_output = torch.load(reference_path, map_location="cpu", weights_only=True)
        metrics = compare_cpu(reference_output, output_cpu)
        row["reference_output"] = metrics
        if not bool(metrics["bitwise_equal"]) and not args.allow_non_bitwise:
            raise QualificationError(f"{case.name}: output is not bitwise equal to exact baseline")

    if args.save_output_dir is not None:
        save_path = args.save_output_dir.expanduser().resolve() / f"{case.name}.pt"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(output_cpu, save_path)
        row["saved_output"] = str(save_path)

    if not args.skip_pad_poison and int(geometry["ragged_blocks"]) > 0:
        valid = (
            torch.arange(BLOCK_SIZE, device=device)[None, :]
            < block_sizes[:, None]
        ).reshape(-1)
        invalid = ~valid.view(1, -1, 1, 1)
        for tensor in (q, k, v):
            tensor.masked_fill_(invalid, 7.0)
        invoke_kernel(kernel, q, k, v, q2k_idx, q2k_num, block_sizes, out, lse)
        torch.cuda.synchronize(device)
        poison_cpu = canonical_output_cpu(out, block_sizes)
        poison_metrics = compare_cpu(output_cpu, poison_cpu)
        row["padding_poison"] = poison_metrics
        if not bool(poison_metrics["bitwise_equal"]):
            raise QualificationError(
                f"{case.name}: invalid padded K/V values changed valid output: {poison_metrics}"
            )
        del poison_cpu

    del q, k, v, block_sizes, q2k_idx, q2k_num, out, lse, output_cpu
    torch.cuda.empty_cache()
    return row


def main() -> int:
    args = parse_args()
    output_dir = (
        args.output_dir
        or DEFAULT_RESULTS_ROOT / f"h3-vsa-exact-shape-matrix-sm120-{stamp()}"
    ).expanduser().resolve()
    report_path = output_dir / "results.json"
    payload: dict[str, Any] = {
        "schema_version": 1,
        "status": "initializing",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "single-GPU synthetic-QKV exact FlashInfer SM120 VSA fine-kernel shape matrix",
        "quality_contract": {
            "algorithm": "paper-faithful VSA fine attention over every supplied top-k block",
            "skip_softmax_threshold": None,
            "tile_size": BLOCK_SIZE,
            "dtype": "bfloat16",
            "head_dim": HEAD_DIM,
            "topk_policy": "fixed by each case; never changed by the candidate kernel",
        },
        "gpu_clock_policy": GPU_CLOCK_POLICY,
        "preset": args.preset,
        "reverse_order": args.reverse,
        "cold_l2": args.cold_l2,
        "warmups_per_case": args.warmups,
        "repeats_per_case": args.repeats,
        "case_order": [case.name for case in args.cases],
        "cases": [],
    }
    write_json(report_path, payload)
    try:
        validate_environment(args)
        payload["preflight"] = gpu_snapshot(args.nvidia_smi, args.physical_gpu, set())
        flashinfer_root = args.flashinfer_root.expanduser().resolve()
        kernel, module_file = import_kernel(flashinfer_root)
        payload["source"] = source_provenance(flashinfer_root, module_file)
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise QualificationError(f"expected one visible CUDA GPU, found {torch.cuda.device_count()}")
        device = torch.device("cuda:0")
        properties = torch.cuda.get_device_properties(device)
        if (properties.major, properties.minor) not in {(12, 0), (12, 1)}:
            raise QualificationError(f"SM120/SM121 required, got SM{properties.major}{properties.minor}")
        payload["runtime"] = {
            "python": sys.version,
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "device": properties.name,
            "compute_capability": f"{properties.major}.{properties.minor}",
            "physical_gpu": args.physical_gpu,
            "logical_gpu": 0,
            "total_hbm_bytes": properties.total_memory,
        }
        reference_rows = load_reference_report(args.reference_report)
        l2_flush = None
        if args.cold_l2:
            l2_flush = torch.empty(args.l2_flush_mib * 1024 * 1024, dtype=torch.uint8, device=device)
        for case in args.cases:
            before = gpu_snapshot(args.nvidia_smi, args.physical_gpu, {os.getpid()})
            row = benchmark_case(case, args, kernel, device, l2_flush, reference_rows)
            row["gpu_before"] = before
            row["gpu_after"] = gpu_snapshot(args.nvidia_smi, args.physical_gpu, {os.getpid()})
            payload["cases"].append(row)
            write_json(report_path, payload)
            print(
                "H3_VSA_EXACT_SHAPE_V1 "
                f"case={case.name} p50_ms={row['timing']['p50_ms']:.6f} "
                f"p90_ms={row['timing']['p90_ms']:.6f} cv={row['timing']['cv']:.6f} "
                f"sha256={row['output_sha256']}",
                flush=True,
            )
        payload["postflight"] = gpu_snapshot(args.nvidia_smi, args.physical_gpu, {os.getpid()})
        payload["summary"] = {
            "case_count": len(payload["cases"]),
            "geomean_p50_ms": math.exp(
                statistics.fmean(math.log(float(row["timing"]["p50_ms"])) for row in payload["cases"])
            ),
            "maximum_cv": max(float(row["timing"]["cv"]) for row in payload["cases"]),
            "all_reference_hashes_match": all(
                row.get("reference_report", {}).get("bitwise_equal", True) for row in payload["cases"]
            ),
        }
        payload["status"] = "passed"
        payload["completed_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(report_path, payload)
        print(f"Report: {report_path}")
        return 0
    except BaseException as exc:
        payload["status"] = "failed"
        payload["error"] = f"{type(exc).__name__}: {exc}"
        payload["completed_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(report_path, payload)
        print(f"VSA exact shape qualification failed: {payload['error']}", file=sys.stderr)
        print(f"Report: {report_path}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
