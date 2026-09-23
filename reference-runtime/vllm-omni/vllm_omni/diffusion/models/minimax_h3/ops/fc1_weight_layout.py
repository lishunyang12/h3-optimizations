"""Load-only native FC1 storage layout for the fixed resident H3 experiment.

No forward wrapper, hook, custom operator, precision change or per-call copy.
The disabled path reads only the environment and does not import Torch.
"""
from __future__ import annotations

import json
import os

ENV = "VLLM_OMNI_H3_FC1_NATIVE_NN"
STATE = "_h3_fc1_native_nn_layout_state"
M, N, K, LAYERS = 11992, 28672, 5376, 50
WEIGHT_BYTES = N * K * 2


def configured_enabled() -> bool:
    flag = os.environ.get(ENV, "0")
    if flag not in ("0", "1"):
        raise ValueError(f"{ENV} must be 0 or 1")
    return flag == "1"


def _runtime_config(model):
    config = model.od_config
    parallel = model.parallel_config
    runtime = dict(
        tp=int(getattr(parallel, "tensor_parallel_size", 1)),
        sp=int(getattr(parallel, "sequence_parallel_size", 1)),
        host_weight_restore=getattr(config, "host_weight_runtime_mode", "disabled"),
        cpu_offload=bool(getattr(config, "enable_cpu_offload", False)),
        layerwise_offload=bool(getattr(config, "enable_layerwise_offload", False)),
        dit_layerwise_offload=bool(getattr(config, "enable_distributed_layerwise_offload", False)),
        quantization=None if model.quant_config is None else type(model.quant_config).__name__,
        old_cute=bool(getattr(model, "_h3_exact_fc1_enabled", False)),
    )
    expected = dict(tp=1, sp=8, host_weight_restore="disabled", cpu_offload=False,
                    layerwise_offload=False, dit_layerwise_offload=False,
                    quantization=None, old_cute=False)
    if runtime != expected or bool(getattr(parallel, "use_hsdp", False)):
        raise RuntimeError(f"Native NN FC1 is qualified only for resident BF16 TP1/SP8 with HWR disabled: {runtime}")
    if os.environ.get("VLLM_OMNI_H3_EXACT_FC1", "0") != "0":
        raise RuntimeError("Native NN FC1 cannot combine with the CuTe FC1 adapter")
    if (model.arch.num_layers, model.arch.hidden_size, model.arch.ffn_hidden_size) != (LAYERS, K, N // 2):
        raise RuntimeError("Native NN FC1 requires the exact 50-block main H3 architecture")
    return runtime


def _attributes(parameter):
    # Preserve loader, input/output dimension and every other Python attribute.
    return {key: id(value) for key, value in parameter.__dict__.items()}


def _fingerprint(parameter):
    return dict(parameter_id=id(parameter), parameter_type=type(parameter),
                shape=tuple(parameter.shape), stride=tuple(parameter.stride()),
                dtype=parameter.dtype, device=parameter.device, pointer=parameter.data_ptr(),
                offset=parameter.storage_offset(), bytes=parameter.untyped_storage().nbytes(),
                attributes=_attributes(parameter))


def _collect_parameters(model, torch):
    from vllm.model_executor.layers.linear import MergedColumnParallelLinear, UnquantizedLinearMethod
    from vllm.model_executor.parameter import ModelWeightParameter

    if len(model.blocks) != LAYERS:
        raise RuntimeError("Native NN FC1 must observe all 50 actual main blocks")
    parameters = []
    for layer, block in enumerate(model.blocks):
        linear = block.mlp.fc1
        parameter = linear.weight
        if (type(linear) is not MergedColumnParallelLinear
                or type(linear.quant_method) is not UnquantizedLinearMethod
                or type(parameter) is not ModelWeightParameter
                or linear.bias is not None or linear._forward_hooks or linear._forward_pre_hooks
                or getattr(block.mlp, "_h3_exact_fc1_layer", None) is not None):
            raise RuntimeError(f"Layer {layer}: native linear semantics differ")
        if (parameter.device.type != "cuda" or parameter.dtype != torch.bfloat16
                or tuple(parameter.shape) != (N, K)
                or tuple(parameter.stride()) not in ((K, 1), (1, N))
                or parameter.storage_offset() != 0
                or parameter.untyped_storage().nbytes() != WEIGHT_BYTES
                or parameter.data_ptr() % 16):
            raise RuntimeError(f"Layer {layer}: FC1 physical storage is outside the qualified contract")
        parameters.append(parameter)
    if len({id(parameter) for parameter in parameters}) != LAYERS:
        raise RuntimeError("Native NN FC1 requires 50 distinct weight Parameter objects")
    return parameters


def _all_bytes_equal(torch, converted, original):
    # Inspect every byte, keeping the temporary materialization bounded to
    # 1024 rows (~11 MiB) instead of allocating another complete FC1 matrix.
    for start in range(0, N, 1024):
        actual = converted[start:start + 1024].contiguous().view(torch.uint8)
        expected = original[start:start + 1024].view(torch.uint8)
        if not torch.equal(actual, expected):
            return False
    return True


def prepare_fc1_native_nn(model):
    """Validate/convert at post-load, then emit exactly one actual rank record."""
    if not configured_enabled():
        return None
    # Reject incompatible leases/offload before importing Torch, inspecting
    # tensor storage or changing any Parameter. This path is load-only.
    runtime = _runtime_config(model)
    import torch
    import torch.distributed as dist
    from vllm.distributed import get_tensor_model_parallel_world_size
    from vllm_omni.diffusion.distributed.parallel_state import get_sequence_parallel_world_size

    if (not dist.is_initialized() or dist.get_world_size() != 8
            or get_tensor_model_parallel_world_size() != 1
            or get_sequence_parallel_world_size() != 8):
        raise RuntimeError("Native NN FC1 requires initialized actual TP1/SP8 groups")
    parameters = _collect_parameters(model, torch)
    existing = getattr(model, STATE, None)
    if existing is not None:
        if existing["runtime"] != runtime or len(existing["fingerprints"]) != LAYERS:
            raise RuntimeError("Native NN FC1 repeated post-load runtime changed")
        for layer, (parameter, expected) in enumerate(zip(parameters, existing["fingerprints"])):
            if _fingerprint(parameter) != expected:
                raise RuntimeError(f"Layer {layer}: prepared native NN storage or Parameter metadata changed")
        existing["repeat_checks"] += 1
        return existing["record"]
    # Prevalidate the entire cohort before changing any weight. A model that
    # arrives already strided without this helper's byte receipt is not trusted.
    if any(tuple(parameter.stride()) != (K, 1) for parameter in parameters):
        raise RuntimeError("Unrecorded preconverted FC1 weights cannot claim native NN qualification")
    record = dict(schema=1, enabled=True, rank=dist.get_rank(), converted=0,
                  already_converted=0, layers=[], runtime=runtime,
                  original_forward=True, original_fc2=True)
    fingerprints = []
    with torch.no_grad():
        for layer, parameter in enumerate(parameters):
            identity, cls, attributes = id(parameter), type(parameter), _attributes(parameter)
            old_data = parameter.detach()  # Keep original storage alive for the byte oracle.
            before_stride = list(parameter.stride())
            converted = old_data.t().contiguous().t()
            if (tuple(converted.shape) != (N, K) or tuple(converted.stride()) != (1, N)
                    or converted.storage_offset() != 0
                    or converted.untyped_storage().nbytes() != WEIGHT_BYTES
                    or not _all_bytes_equal(torch, converted, old_data)):
                raise RuntimeError(f"Layer {layer}: native NN storage conversion failed the full-byte contract")
            parameter.data = converted
            del converted
            if id(parameter) != identity or type(parameter) is not cls or _attributes(parameter) != attributes:
                raise RuntimeError(f"Layer {layer}: native NN conversion changed Parameter metadata")
            after = _fingerprint(parameter)
            if after["pointer"] == old_data.data_ptr():
                raise RuntimeError(f"Layer {layer}: native NN did not create the requested physical storage")
            fingerprints.append(after)
            record["layers"].append(dict(layer=layer, name=f"blocks.{layer}.mlp.fc1.weight",
                shape=list(parameter.shape), before_stride=before_stride, after_stride=list(parameter.stride()),
                dtype=str(parameter.dtype), bf16_bytes_equal=True, parameter_id_preserved=True,
                attributes_preserved=True, storage_offset=parameter.storage_offset(),
                storage_bytes=parameter.untyped_storage().nbytes()))
            record["converted"] += 1
            del old_data
    state = dict(runtime=runtime, fingerprints=fingerprints, repeat_checks=0, record=record)
    setattr(model, STATE, state)
    # Exercise the actual repeated helper path. It reads metadata only and
    # returns above, without a tensor conversion or a second log record.
    repeated = prepare_fc1_native_nn(model)
    if repeated is not record or state["repeat_checks"] != 1 or record["converted"] != LAYERS:
        raise RuntimeError("Native NN FC1 idempotent post-load verification failed")
    record["idempotence"] = dict(actual_repeat_check=True, repeat_conversions=0, repeat_logs=0)
    from vllm.logger import init_logger

    init_logger(__name__).info("H3_FC1_NATIVE_NN_LAYOUT %s", json.dumps(record, sort_keys=True))
    return record
