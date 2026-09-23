"""Opt-in v24-qualified full-shape BF16 FC1, with per-step dispatch evidence.

Only executable code and tensor signatures are cached. Actual input, weight,
output and current CUDA stream are passed anew for every invocation.
"""
from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
from contextvars import ContextVar
from functools import lru_cache
import importlib.metadata
import json
import os
import re
import threading

import torch
from vllm.logger import init_logger

logger = init_logger(__name__)
ENV = "VLLM_OMNI_H3_EXACT_FC1"
CONFIG = "m128n128_e8"
M, N, K = 11992, 28672, 5376
_STEP = ContextVar("h3_exact_fc1_step", default=None)
_COMPILED = {}
_COMPILE_LOCK = threading.Lock()


def configured_enabled() -> bool:
    flag = os.environ.get(ENV, "0")
    if flag not in ("0", "1"):
        raise ValueError(f"{ENV} must be 0 or 1")
    return flag == "1"


def configured_layer(prefix: str) -> int | None:
    if not configured_enabled():
        return None
    matched = re.fullmatch(r"blocks\.(\d+)\.mlp", prefix)
    if matched is None:
        return None  # Token refiner and every other MLP retain their original path.
    layer = int(matched.group(1))
    if not 0 <= layer < 50:
        raise ValueError("Exact FC1 is qualified only for the 50 main H3 blocks")
    return layer


@contextmanager
def exact_fc1_step(enabled: bool, layer_count: int):
    if not enabled:
        yield
        return
    if layer_count != 50 or _STEP.get() is not None:
        raise RuntimeError("Exact FC1 requires one non-nested 50-block main DiT forward")
    state = {"layers": Counter(), "fallback_layers": Counter(), "fallback_reasons": Counter(),
             "attempted_layers": Counter()}
    token = _STEP.set(state)
    try:
        yield
    except BaseException:
        # A failed invocation must never be presented as a completed step.
        raise
    else:
        import torch.distributed as dist

        attempted = state["attempted_layers"]
        if attempted != Counter({layer: 1 for layer in range(50)}):
            raise RuntimeError(f"Exact FC1 did not observe the 50 real main-layer calls: {dict(attempted)}")
        record = dict(schema=1, rank=dist.get_rank() if dist.is_initialized() else 0,
                      config=CONFIG, shape=[M, N, K], calls=sum(state["layers"].values()),
                      layers=dict(sorted(state["layers"].items())),
                      fallback=sum(state["fallback_layers"].values()),
                      fallback_layers=dict(sorted(state["fallback_layers"].items())),
                      fallback_reasons=dict(state["fallback_reasons"]),
                      original_fc2=True, original_activation=True)
        logger.info("H3_EXACT_FC1_STEP %s", json.dumps(record, sort_keys=True))
    finally:
        _STEP.reset(token)


def _fallback(state, layer, reason):
    state["fallback_layers"][layer] += 1
    state["fallback_reasons"][reason] += 1
    return None


@lru_cache(None)
def _runtime_support(device_index: int) -> str | None:
    from .fc1_kernel.qualification import QUALIFICATION

    expected = QUALIFICATION["flags"]
    if torch.__version__ != expected["torch_version"] or torch.version.git_version != expected["torch_git_version"]:
        return "torch_version"
    if importlib.metadata.version("nvidia-cutlass-dsl") != expected["cutlass_dsl_version"]:
        return "cutlass_version"
    if os.environ.get("CUTE_DSL_ARCH") != "sm_120a":
        return "cute_arch_environment"
    props = torch.cuda.get_device_properties(device_index)
    if (props.major, props.minor, props.multi_processor_count) != (12, 0, 110):
        return "device_geometry"
    return None


def _tensor_view(tensor):
    from cutlass.cute.runtime import from_dlpack

    return from_dlpack(tensor.unsqueeze(-1), assumed_align=16)


@torch.compiler.disable
def maybe_exact_fc1(x: torch.Tensor, linear, layer: int) -> torch.Tensor | None:
    """Return the actual candidate output or None for a counted original path.

    Shape/dtype/module exclusions use the original fc1 call in the caller. A
    compilation or launch failure raises; it is never hidden as a fallback.
    """
    state = _STEP.get()
    if state is None:
        raise RuntimeError("Exact FC1 called outside its main-DiT step counter scope")
    state["attempted_layers"][layer] += 1
    if x.device.type != "cuda" or tuple(x.shape) != (M, K) or x.dtype != torch.bfloat16:
        return _fallback(state, layer, "input_shape_device_dtype")
    if x.stride() != (K, 1) or x.data_ptr() % 16:
        return _fallback(state, layer, "input_stride_alignment")
    weight = getattr(linear, "weight", None)
    if (not isinstance(weight, torch.Tensor) or tuple(weight.shape) != (N, K)
            or weight.dtype != torch.bfloat16 or weight.device != x.device):
        return _fallback(state, layer, "weight_shape_device_dtype")
    if weight.stride() != (K, 1) or weight.data_ptr() % 16:
        return _fallback(state, layer, "weight_stride_alignment")
    from vllm.distributed import get_tensor_model_parallel_world_size
    from vllm.model_executor.layers.linear import MergedColumnParallelLinear, UnquantizedLinearMethod
    from vllm import envs as vllm_envs

    if vllm_envs.VLLM_BATCH_INVARIANT:
        return _fallback(state, layer, "batch_invariant")
    if (type(linear) is not MergedColumnParallelLinear
            or type(linear.quant_method) is not UnquantizedLinearMethod
            or getattr(linear, "bias", None) is not None
            or get_tensor_model_parallel_world_size() != 1
            or linear._forward_hooks or linear._forward_pre_hooks):
        return _fallback(state, layer, "linear_semantics")
    if torch.is_grad_enabled() and (x.requires_grad or weight.requires_grad):
        return _fallback(state, layer, "autograd")
    if (torch.backends.cuda.matmul.allow_tf32 is not False
            or torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction is not True):
        return _fallback(state, layer, "matmul_flags")
    device_index = x.device.index
    if torch.cuda.current_device() != device_index:
        return _fallback(state, layer, "current_device")
    reason = _runtime_support(device_index)
    if reason is not None:
        return _fallback(state, layer, reason)

    import cuda.bindings.driver as cuda
    import cutlass.cute as cute

    # This is the real FC1 result allocation. No full-shape compile placeholder,
    # output arena, weight copy or prepacking is retained by this adapter.
    output = torch.empty((M, N), dtype=torch.bfloat16, device=x.device)
    key = (device_index, tuple(x.shape), x.stride(), tuple(weight.shape), weight.stride(), output.stride())
    compiled = _COMPILED.get(key)
    if compiled is None:
        with _COMPILE_LOCK:
            compiled = _COMPILED.get(key)
            if compiled is None:
                if torch.cuda.is_current_stream_capturing():
                    raise RuntimeError("Exact FC1 must compile on a real eager invocation before CUDA capture")
                from .fc1_kernel.candidate import ConfiguredGemm

                kernel = ConfiguredGemm(CONFIG)
                compile_stream = cuda.CUstream(torch.cuda.current_stream(x.device).cuda_stream)
                compiled = cute.compile(kernel, _tensor_view(x), _tensor_view(weight), _tensor_view(output),
                                        110, compile_stream)
                if (kernel.ab_stage, kernel.epi_stage) != (2, 8):
                    raise RuntimeError("Qualified FC1 stage configuration changed")
                _COMPILED[key] = compiled
    # In particular, do not capture the compile-time stream or first layer's
    # tensors in a closure. Parameter storage can change after model offload.
    current_stream = cuda.CUstream(torch.cuda.current_stream(x.device).cuda_stream)
    compiled(_tensor_view(x), _tensor_view(weight), _tensor_view(output), current_stream)
    state["layers"][layer] += 1
    return output
