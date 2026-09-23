#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Single-GPU feasibility benchmark for H3 O producer-direct landing.

The production SP8 path presents cuDNN with ``[B, S, H, D]`` tensors after
the forward Ulysses exchange.  PyTorch SDPA permutes them to ``[B, H, S, D]``
and allocates its own output.  cuDNN preserves token-major physical strides,
so permuting the result back produces a contiguous ``[B, S, H, D]`` tensor;
the reverse FlashInfer RDMA exchange still has to copy that tensor into its
separate registered input landing.

FlashInfer's cuDNN frontend accepts a caller-owned ``out=`` tensor in packed
``[T, H, D]`` layout.  This benchmark checks whether that frontend can write
the attention result directly into an O-landing-shaped buffer while retaining
the current H3 attention semantics.  A small numerical gate always runs before
the exact 15-second T2VA geometry.

The one-GPU output buffer is an allocation-compatible proxy for the registered
RDMA landing: registration changes ownership/lifetime but not the CUDA virtual
memory layout seen by cuDNN.  Pointer identity is checked here; an eight-rank
smoke is still required before production integration.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn.functional as F
import cudnn
from torch.nn.attention import SDPBackend, sdpa_kernel

from flashinfer.cudnn.prefill import (
    UIDs,
    _create_cudnn_handle,
    cudnn_batch_prefill_with_kv_cache,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=1101)
    parser.add_argument("--heads", type=int, default=7)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--small-q-len", type=int, default=512)
    parser.add_argument("--small-kv-len", type=int, default=509)
    parser.add_argument("--exact-q-len", type=int, default=95488)
    parser.add_argument("--exact-kv-len", type=int, default=95463)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--layers-per-step", type=int, default=50)
    parser.add_argument("--workspace-mib", type=int, default=128)
    parser.add_argument(
        "--direct-mode",
        choices=("masked", "mask-free"),
        default="masked",
        help=(
            "masked uses FlashInfer's public batch-prefill API; mask-free uses "
            "the same cuDNN frontend after H3 has physically narrowed K/V"
        ),
    )
    parser.add_argument(
        "--heur-mode",
        choices=("A", "B", "FALLBACK"),
        default="A",
        help="cuDNN frontend heuristic used by the mask-free direct graph",
    )
    parser.add_argument("--small-only", action="store_true")
    parser.add_argument("--json-out", type=Path)
    return parser.parse_args()


def _current_torch_cudnn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    scale: float,
) -> torch.Tensor:
    """Mirror CuDNNAttentionImpl: BSND -> BHSD SDPA -> BSND view."""
    q_bhsd = q.unsqueeze(0).permute(0, 2, 1, 3)
    k_bhsd = k.unsqueeze(0).permute(0, 2, 1, 3)
    v_bhsd = v.unsqueeze(0).permute(0, 2, 1, 3)
    with sdpa_kernel([SDPBackend.CUDNN_ATTENTION]):
        out = F.scaled_dot_product_attention(
            q_bhsd,
            k_bhsd,
            v_bhsd,
            dropout_p=0.0,
            is_causal=False,
            scale=scale,
        )
    return out.permute(0, 2, 1, 3).squeeze(0)


class FlashInferCudnnDirect:
    """Reusable FlashInfer cuDNN call with a caller-owned packed output."""

    def __init__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        out: torch.Tensor,
        workspace: torch.Tensor,
        *,
        scale: float,
    ) -> None:
        self.q = q
        self.k = k
        self.v = v
        self.out = out
        self.workspace = workspace
        self.scale = scale
        self.q_len = int(q.shape[0])
        self.kv_len = int(k.shape[0])
        self.actual_q = torch.tensor(
            [self.q_len], dtype=torch.int32, device=q.device
        ).view(1, 1, 1, 1)
        self.actual_kv = torch.tensor(
            [self.kv_len], dtype=torch.int32, device=q.device
        ).view(1, 1, 1, 1)

    def __call__(self) -> torch.Tensor:
        result, lse = cudnn_batch_prefill_with_kv_cache(
            self.q,
            self.k,
            self.v,
            self.scale,
            self.workspace,
            max_token_per_sequence=self.q_len,
            max_sequence_kv=self.kv_len,
            actual_seq_lens_q=self.actual_q,
            actual_seq_lens_kv=self.actual_kv,
            causal=False,
            return_lse=False,
            out=self.out,
        )
        if lse is not None:
            raise RuntimeError("return_lse=False unexpectedly produced LSE")
        if result.data_ptr() != self.out.data_ptr():
            raise RuntimeError("FlashInfer cuDNN did not return the supplied out buffer")
        return result


def _mask_free_graph_key(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    scale: float,
) -> tuple[Any, ...]:
    return (
        q.dtype,
        tuple(q.shape),
        tuple(q.stride()),
        k.dtype,
        tuple(k.shape),
        tuple(k.stride()),
        v.dtype,
        tuple(v.shape),
        tuple(v.stride()),
        out.dtype,
        tuple(out.shape),
        tuple(out.stride()),
        scale,
    )


@cudnn.jit(heur_modes=[cudnn.heur_mode.A])
@cudnn.graph_cache(key_fn=_mask_free_graph_key)
def _build_mask_free_direct_graph(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    scale: float,
):
    """Build the no-mask subset of FlashInfer's cuDNN prefill graph."""
    handle = _create_cudnn_handle(torch.cuda.current_stream(q.device))
    q_dtype = cudnn.datatypes._torch_to_cudnn_data_type(q.dtype)
    k_dtype = cudnn.datatypes._torch_to_cudnn_data_type(k.dtype)
    v_dtype = cudnn.datatypes._torch_to_cudnn_data_type(v.dtype)
    o_dtype = cudnn.datatypes._torch_to_cudnn_data_type(out.dtype)
    q_len, q_heads, q_dim = q.shape
    kv_len, kv_heads, kv_dim = k.shape
    if q_dim != kv_dim or v.shape != k.shape or q_heads != kv_heads:
        raise ValueError("mask-free probe requires equal Q/K/V heads and head dims")

    with cudnn.graph(handle) as (graph, _):
        cudnn_q = graph.tensor(
            name="q",
            dim=(1, q_heads, q_len, q_dim),
            stride=(q_heads * q_dim, q.stride(1), q.stride(0), q.stride(2)),
            data_type=q_dtype,
        )
        cudnn_k = graph.tensor(
            name="k",
            dim=(1, kv_heads, kv_len, kv_dim),
            stride=(kv_heads * kv_dim * kv_len, k.stride(1), k.stride(0), k.stride(2)),
            data_type=k_dtype,
        )
        cudnn_v = graph.tensor(
            name="v",
            dim=(1, kv_heads, kv_len, kv_dim),
            stride=(kv_heads * kv_dim * kv_len, v.stride(1), v.stride(0), v.stride(2)),
            data_type=v_dtype,
        )
        cudnn_q.set_uid(UIDs.Q_UID.value)
        cudnn_k.set_uid(UIDs.K_UID.value)
        cudnn_v.set_uid(UIDs.V_UID.value)
        cudnn_o, _stats = graph.sdpa(
            name="sdpa_mask_free_direct",
            q=cudnn_q,
            k=cudnn_k,
            v=cudnn_v,
            use_padding_mask=False,
            attn_scale=scale,
            generate_stats=False,
            use_causal_mask_bottom_right=False,
            compute_data_type=cudnn.data_type.FLOAT,
        )
        cudnn_o.set_uid(UIDs.O_UID.value).set_output(True).set_dim(
            [1, q_heads, q_len, q_dim]
        ).set_stride(
            [q_len * q_dim * q_heads, q_dim, q_dim * q_heads, 1]
        ).set_data_type(o_dtype)
        return graph, [cudnn_q, cudnn_k, cudnn_v, cudnn_o]


# Keep separate graph caches because the cuDNN frontend cache key deliberately
# describes tensor semantics, while the heuristic is a build policy rather
# than an operator input.
_mask_free_graph_definition = _build_mask_free_direct_graph.__wrapped__.__wrapped__


@cudnn.jit(heur_modes=[cudnn.heur_mode.B])
@cudnn.graph_cache(key_fn=_mask_free_graph_key)
def _build_mask_free_direct_graph_b(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    scale: float,
):
    return _mask_free_graph_definition(q, k, v, out, scale)


@cudnn.jit(heur_modes=[cudnn.heur_mode.FALLBACK])
@cudnn.graph_cache(key_fn=_mask_free_graph_key)
def _build_mask_free_direct_graph_fallback(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    scale: float,
):
    return _mask_free_graph_definition(q, k, v, out, scale)


class FlashInferCudnnMaskFreeDirect(FlashInferCudnnDirect):
    """Same frontend graph with physically sliced KV and no padding mask.

    This matches H3's current CuDNNAttentionImpl more closely: it narrows K/V
    to ``valid_kv_length`` before SDPA, so no runtime padding mask is present.
    FlashInfer's public batch-prefill wrapper currently insists on both actual
    length tensors; the internal executor already supports the required
    one-sided form and still supplies a caller-owned ``out``.
    """

    def __init__(self, *args, heur_mode: str = "A", **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.heur_mode = heur_mode

    def __call__(self) -> torch.Tensor:
        builder = {
            "A": _build_mask_free_direct_graph,
            "B": _build_mask_free_direct_graph_b,
            "FALLBACK": _build_mask_free_direct_graph_fallback,
        }[self.heur_mode]
        graph, _tensors = builder(
            self.q, self.k, self.v, self.out, self.scale
        )
        handle = _create_cudnn_handle(torch.cuda.current_stream(self.q.device))
        graph.execute(
            {
                UIDs.Q_UID.value: self.q,
                UIDs.K_UID.value: self.k,
                UIDs.V_UID.value: self.v,
                UIDs.O_UID.value: self.out,
            },
            workspace=self.workspace,
            handle=handle,
        )
        return self.out


def _event_samples_ms(
    fn: Callable[[], torch.Tensor],
    *,
    device: torch.device,
    warmup: int,
    iterations: int,
) -> list[float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize(device)

    samples: list[float] = []
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        samples.append(float(start.elapsed_time(end)))
    return samples


def _summarize(samples: list[float]) -> dict[str, Any]:
    ordered = sorted(samples)
    p95_index = min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1)
    return {
        "samples_ms": samples,
        "mean_ms": statistics.fmean(samples),
        "median_ms": statistics.median(samples),
        "min_ms": min(samples),
        "p95_ms": ordered[p95_index],
        "max_ms": max(samples),
    }


def _numeric_report(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, Any]:
    delta = (reference.float() - candidate.float()).abs()
    return {
        "bit_exact": bool(torch.equal(reference, candidate)),
        "allclose_atol_1e-2_rtol_1e-2": bool(
            torch.allclose(reference, candidate, atol=1e-2, rtol=1e-2)
        ),
        "max_abs": float(delta.max().item()),
        "mean_abs": float(delta.mean().item()),
        "different_elements": int(torch.count_nonzero(reference != candidate).item()),
        "elements": reference.numel(),
    }


def _make_inputs(
    q_len: int,
    kv_len: int,
    heads: int,
    head_dim: int,
    *,
    device: torch.device,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    q = torch.randn(
        q_len,
        heads,
        head_dim,
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )
    k = torch.randn(
        kv_len,
        heads,
        head_dim,
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )
    v = torch.randn(
        kv_len,
        heads,
        head_dim,
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )
    return q, k, v


def _run_geometry(
    name: str,
    *,
    q_len: int,
    kv_len: int,
    args: argparse.Namespace,
    device: torch.device,
    workspace: torch.Tensor,
    benchmark: bool,
) -> dict[str, Any]:
    q, k, v = _make_inputs(
        q_len,
        kv_len,
        args.heads,
        args.head_dim,
        device=device,
        seed=args.seed,
    )
    scale = args.head_dim**-0.5
    direct_out = torch.empty_like(q)
    staging = torch.empty_like(q)
    direct: Callable[[], torch.Tensor]
    if args.direct_mode == "masked":
        direct = FlashInferCudnnDirect(
            q, k, v, direct_out, workspace, scale=scale
        )
    else:
        direct = FlashInferCudnnMaskFreeDirect(
            q,
            k,
            v,
            direct_out,
            workspace,
            scale=scale,
            heur_mode=args.heur_mode,
        )

    # Both paths are materialized before any timing. This is the numerical gate
    # and also keeps graph/plan construction out of steady-state measurements.
    current_out = _current_torch_cudnn(q, k, v, scale=scale)
    candidate_out = direct()
    torch.cuda.synchronize(device)
    numeric = _numeric_report(current_out, candidate_out)

    report: dict[str, Any] = {
        "name": name,
        "q_shape_thd": list(q.shape),
        "kv_shape_thd": list(k.shape),
        "dtype": str(q.dtype),
        "causal": False,
        "softmax_scale": scale,
        "current_output_shape": list(current_out.shape),
        "current_output_stride": list(current_out.stride()),
        "current_output_contiguous": current_out.is_contiguous(),
        "direct_output_stride": list(candidate_out.stride()),
        "direct_output_contiguous": candidate_out.is_contiguous(),
        "direct_output_pointer_exact": candidate_out.data_ptr() == direct_out.data_ptr(),
        "direct_mode": args.direct_mode,
        "heur_mode": args.heur_mode if args.direct_mode == "mask-free" else None,
        "numeric": numeric,
    }
    if not numeric["allclose_atol_1e-2_rtol_1e-2"]:
        report["status"] = "failed_numeric_gate"
        return report
    report["status"] = "pass"

    if not benchmark:
        return report

    # Production baseline: cuDNN returns its own contiguous BSND allocation,
    # then gather() copies it into the separate registered O input landing.
    def current_plus_staging() -> torch.Tensor:
        out = _current_torch_cudnn(q, k, v, scale=scale)
        staging.copy_(out)
        return staging

    # This is a strict upper bound on producer-direct's removable cost while
    # retaining the exact current PyTorch/cuDNN kernel.
    frozen_current = current_out

    def staging_only() -> torch.Tensor:
        staging.copy_(frozen_current)
        return staging

    current_samples = _event_samples_ms(
        current_plus_staging,
        device=device,
        warmup=args.warmup,
        iterations=args.iterations,
    )
    direct_samples = _event_samples_ms(
        direct,
        device=device,
        warmup=args.warmup,
        iterations=args.iterations,
    )
    copy_samples = _event_samples_ms(
        staging_only,
        device=device,
        warmup=args.warmup,
        iterations=args.iterations,
    )
    current_stats = _summarize(current_samples)
    direct_stats = _summarize(direct_samples)
    copy_stats = _summarize(copy_samples)
    delta_per_layer = current_stats["median_ms"] - direct_stats["median_ms"]
    direct_label = f"flashinfer_cudnn_{args.direct_mode.replace('-', '_')}_direct_out"
    report["timing"] = {
        "current_torch_cudnn_plus_o_staging": current_stats,
        direct_label: direct_stats,
        "o_staging_copy_only": copy_stats,
        "candidate_delta_ms_per_layer_median": delta_per_layer,
        "candidate_delta_ms_per_step_median": delta_per_layer
        * args.layers_per_step,
        "copy_upper_bound_ms_per_step_median": copy_stats["median_ms"]
        * args.layers_per_step,
        "layers_per_step": args.layers_per_step,
    }
    return report


def main() -> None:
    args = parse_args()
    if args.small_q_len <= 1 or not 0 < args.small_kv_len <= args.small_q_len:
        raise SystemExit("small lengths must satisfy q_len > 1 and 0 < kv_len <= q_len")
    if args.exact_q_len <= 1 or not 0 < args.exact_kv_len <= args.exact_q_len:
        raise SystemExit("exact lengths must satisfy q_len > 1 and 0 < kv_len <= q_len")
    if args.heads <= 0 or args.head_dim <= 0:
        raise SystemExit("heads and head_dim must be positive")
    if args.warmup < 1 or args.iterations < 1:
        raise SystemExit("warmup and iterations must be positive")

    device = torch.device(args.device)
    if device.type != "cuda":
        raise SystemExit("this benchmark requires a CUDA device")
    torch.cuda.set_device(device)
    props = torch.cuda.get_device_properties(device)
    if props.major != 12 or props.minor != 0:
        raise SystemExit(
            f"this H3 benchmark targets SM120, got SM{props.major}{props.minor}"
        )

    started = time.time()
    workspace = torch.empty(
        args.workspace_mib * 1024 * 1024,
        dtype=torch.uint8,
        device=device,
    )
    result: dict[str, Any] = {
        "schema_version": 1,
        "device": {
            "name": props.name,
            "compute_capability": [props.major, props.minor],
            "visible_ordinal": device.index,
        },
        "h3_metadata": {
            "task": "t2va",
            "duration_seconds": 15,
            "decoded_frames": 362,
            "decoded_height": 704,
            "decoded_width": 1280,
            "text_tokens": 121,
            "latent_grid": [27, 88, 160],
            "video_rows": 27 * (88 // 2) * (160 // 2),
            "audio_t": 151,
            "audio_channels": 2,
            "used_kv_length": args.exact_kv_len,
            "aligned_query_length": args.exact_q_len,
            "padding_rows": args.exact_q_len - args.exact_kv_len,
            "post_sp8_shape_bshd": [1, args.exact_q_len, args.heads, args.head_dim],
        },
        "settings": {
            "seed": args.seed,
            "warmup": args.warmup,
            "iterations": args.iterations,
            "workspace_mib": args.workspace_mib,
            "layers_per_step": args.layers_per_step,
            "direct_mode": args.direct_mode,
            "heur_mode": args.heur_mode,
        },
    }

    small = _run_geometry(
        "small_gate",
        q_len=args.small_q_len,
        kv_len=args.small_kv_len,
        args=args,
        device=device,
        workspace=workspace,
        benchmark=True,
    )
    result["small"] = small
    if small["status"] != "pass":
        result["status"] = "failed_small_numeric_gate"
    elif args.small_only:
        result["status"] = "small_only_pass"
    else:
        exact = _run_geometry(
            "h3_exact",
            q_len=args.exact_q_len,
            kv_len=args.exact_kv_len,
            args=args,
            device=device,
            workspace=workspace,
            benchmark=True,
        )
        result["exact"] = exact
        result["status"] = "pass" if exact["status"] == "pass" else "failed_exact_numeric_gate"

    result["elapsed_seconds"] = time.time() - started
    encoded = json.dumps(result, indent=2, sort_keys=True)
    print("H3_O_PRODUCER_DIRECT_BENCHMARK=" + encoded, flush=True)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(encoded + "\n", encoding="utf-8")
    if result["status"].startswith("failed"):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
