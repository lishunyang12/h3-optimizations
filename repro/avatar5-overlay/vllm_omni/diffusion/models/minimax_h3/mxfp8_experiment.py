"""Isolated FastH3 MXFP8 / original blockwise-FP8 projection comparison.

Both arms keep the same opaque projection boundary inside regional compilation.
The MX producer and scaled_mm recipes are the previously measured PR37903 path;
the blockwise arm calls the unchanged vLLM online method and Cutlass kernel.
"""
from __future__ import annotations

import hashlib
import inspect
import json
import os
from pathlib import Path
from types import MethodType

import torch
from torch.nn.functional import ScalingType, SwizzleType, scaled_mm
from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod

from .mxfp8_kernels import _scale_numel, mxfp8_quantize_swizzled

ENV = "VLLM_OMNI_H3_EXPERIMENT_QUANT_MODE"
MODE = "qkv-o-mlp-mxfp8-e4m3fn-a1x32-w1x32-e8m0-cublaslt-sm120-v1"
SUFFIXES = ("attn.qkv_proj", "attn.out_proj", "mlp.fc1", "mlp.fc2")
SHAPES = dict(zip(SUFFIXES, ((21504, 5376), (5376, 7168), (28672, 5376), (5376, 14336))))
_REGISTRY = {}


def requested_mode():
    mode = os.getenv(ENV)
    if mode not in (None, "mxfp8", "blockwise"):
        raise ValueError("Unknown FastH3 experiment quantization mode")
    legacy = os.getenv("VLLM_OMNI_H3_EXPERIMENT_BLOCKWISE", "0")
    if legacy not in ("0", "1") or (legacy == "1" and mode == "mxfp8"):
        raise ValueError("Conflicting blockwise and MXFP8 experiment switches")
    return mode or ("blockwise" if legacy == "1" else None)



def requested_fusion(mode):
    flag = os.getenv("VLLM_OMNI_H3_SWIGLU_MXFP8_FUSION", "0")
    if flag not in ("0", "1") or (flag == "1" and mode != "mxfp8"):
        raise ValueError("FC2 SwiGLU fusion requires the MXFP8 experiment mode")
    return flag == "1"


def target_names():
    return tuple(f"blocks.{i}.{suffix}" for i in range(50) for suffix in SUFFIXES)


def _rank():
    return torch.distributed.get_rank()


def _mx_dense(layer, x, bias=None):
    if bias is not None or x.dtype != torch.bfloat16 or not x.is_cuda:
        raise ValueError("MXFP8 requires bias-free CUDA BF16 activations")
    lead = x.shape[:-1]
    q, scale = mxfp8_quantize_swizzled(x.reshape(-1, x.shape[-1]).contiguous())
    out = scaled_mm(
        q, layer.weight.t(),
        scale_a=scale.view(torch.float8_e8m0fnu),
        scale_recipe_a=ScalingType.BlockWise1x32,
        swizzle_a=SwizzleType.SWIZZLE_32_4_4,
        scale_b=layer.mxfp8_weight_scale,
        scale_recipe_b=ScalingType.BlockWise1x32,
        swizzle_b=SwizzleType.SWIZZLE_32_4_4,
        bias=None, output_dtype=torch.bfloat16,
    )
    return out.view(*lead, out.shape[-1])


@torch.library.custom_op("h3_quant_ab::projection", mutates_args=())
def _projection(x: torch.Tensor, weight: torch.Tensor, scale: torch.Tensor, token: int) -> torch.Tensor:
    record = _REGISTRY[token]
    layer = record["layer"]
    if (weight.data_ptr() != layer.weight.data_ptr() or scale.data_ptr() != record["scale"].data_ptr()
            or weight.data_ptr() != record["weight_ptr"] or scale.data_ptr() != record["scale_ptr"]
            or tuple(weight.shape) != record["weight_shape"] or weight.dtype != torch.float8_e4m3fn
            or tuple(scale.shape) != record["scale_shape"] or scale.dtype != record["scale_dtype"]
            or weight.device != record["device"] or scale.device != record["device"]):
        raise RuntimeError("Quantized projection storage changed after installation")
    if x.dtype != torch.bfloat16 or not x.is_cuda or x.device != weight.device:
        raise ValueError("Both comparison arms require resident CUDA BF16 activations")
    if record['name'].endswith('.attn.qkv_proj') and not record.get('dense_ref2va'):
        raise RuntimeError('MXFP8 split QKV unexpectedly used the ordinary full projection')
    if record["fused_required"]:
        raise RuntimeError("Fused FC2 unexpectedly used standalone activation quantization")
    from . import ref2va_bf16_weights as bf
    if bf.use_bf16(record):
        out = torch.nn.functional.linear(x, bf.weight_for(record))
        record["bf16_calls"] += 1
    elif record["mode"] == "mxfp8":
        if type(layer.quant_method) is not H3MXFP8Method:
            raise RuntimeError("MXFP8 method was replaced")
        out = _mx_dense(layer, x)
    else:
        if type(layer.quant_method) is not record["method_type"] or type(layer.quant_method.fp8_linear).__name__ != "CutlassFp8BlockScaledMMKernel":
            raise RuntimeError("Original blockwise Cutlass method was replaced")
        out = record["original_apply"](layer, x, None)
    if out.dtype != torch.bfloat16 or out.shape != (*x.shape[:-1], weight.shape[0]):
        raise RuntimeError("Projection output contract differs")
    record["calls"] += 1
    return out


@_projection.register_fake
def _projection_fake(x, weight, scale, token):
    return x.new_empty((*x.shape[:-1], weight.shape[0]))



@torch.library.custom_op("h3_quant_ab::swiglu_fc2", mutates_args=())
def _swiglu_fc2(x: torch.Tensor, weight: torch.Tensor, scale: torch.Tensor, token: int) -> torch.Tensor:
    record = _REGISTRY[token]
    layer = record["layer"]
    if (weight.data_ptr() != layer.weight.data_ptr() or scale.data_ptr() != record["scale"].data_ptr()
            or weight.data_ptr() != record["weight_ptr"] or scale.data_ptr() != record["scale_ptr"]
            or tuple(weight.shape) != record["weight_shape"] or weight.dtype != torch.float8_e4m3fn
            or tuple(scale.shape) != record["scale_shape"] or scale.dtype != record["scale_dtype"]
            or weight.device != record["device"] or scale.device != record["device"]):
        raise RuntimeError("Quantized projection storage changed after installation")
    if x.dtype != torch.bfloat16 or not x.is_cuda or x.device != weight.device:
        raise ValueError("Both comparison arms require resident CUDA BF16 activations")
    if (record["mode"] != "mxfp8" or type(layer.quant_method) is not H3MXFP8Method
            or not record["name"].endswith(".mlp.fc2") or not record["fused_required"]
            or layer.tp_size != 1 or layer.bias is not None or x.shape[-1] != 2 * weight.shape[1]):
        raise RuntimeError("Only admitted resident TP1 MXFP8 FC2 may consume fused gate|up")
    from . import ref2va_bf16_weights as bf
    if bf.use_bf16(record):
        from vllm.model_executor.layers.activation import SiluAndMul
        activated=SiluAndMul()(x)
        out=torch.nn.functional.linear(activated,bf.weight_for(record))
        record["calls"]+=1
        record["bf16_calls"]+=1
        bf.finished_fc2(record)
        return out
    from .silu_mxfp8_fusion import silu_mul_mxfp8_cuda
    q, activation_scale = silu_mul_mxfp8_cuda(x.reshape(-1, x.shape[-1]).contiguous())
    out = scaled_mm(q, weight.t(),
        scale_a=activation_scale.view(torch.float8_e8m0fnu),
        scale_recipe_a=ScalingType.BlockWise1x32, swizzle_a=SwizzleType.SWIZZLE_32_4_4,
        scale_b=scale, scale_recipe_b=ScalingType.BlockWise1x32,
        swizzle_b=SwizzleType.SWIZZLE_32_4_4, bias=None, output_dtype=torch.bfloat16)
    out = out.view(*x.shape[:-1], weight.shape[0])
    if out.dtype != torch.bfloat16 or out.shape != (*x.shape[:-1], weight.shape[0]):
        raise RuntimeError("Fused FC2 output contract differs")
    record["calls"] += 1
    record["fused_calls"] += 1
    return out


@_swiglu_fc2.register_fake
def _swiglu_fc2_fake(x, weight, scale, token):
    return x.new_empty((*x.shape[:-1], weight.shape[0]))


def fused_swiglu_fc2(layer, hidden):
    record = _REGISTRY[layer._h3_quant_token]
    return _swiglu_fc2(hidden, layer.weight, record["scale"], layer._h3_quant_token)


def _registered_apply(layer, x, bias):
    if bias is not None:
        raise ValueError("The 200 comparison projections must be bias-free")
    record = _REGISTRY[layer._h3_quant_token]
    return _projection(x, layer.weight, record["scale"], layer._h3_quant_token)


class H3MXFP8Method(UnquantizedLinearMethod):
    def apply(self, layer, x, bias=None):
        if hasattr(layer, "_h3_quant_token"):
            return _registered_apply(layer, x, bias)
        # Standalone real-shape numerical smoke uses the identical direct path.
        return _mx_dense(layer, x, bias)


def quantize_layer(layer):
    """Quantize one already-fused BF16 layer; also used by the isolated smoke."""
    weight = layer.weight
    if (weight.dtype != torch.bfloat16 or not weight.is_cuda or weight.ndim != 2
            or weight.shape[0] % 16 or weight.shape[1] % 32 or layer.bias is not None
            or type(layer.quant_method) is not UnquantizedLinearMethod):
        raise ValueError("MXFP8 source must be a loaded CUDA BF16 unquantized projection")
    if hasattr(layer, "mxfp8_weight_scale"):
        raise ValueError("Duplicate MXFP8 weight quantization")
    q, scale = mxfp8_quantize_swizzled(weight.detach().contiguous())
    if q.dtype != torch.float8_e4m3fn or q.shape != weight.shape or scale.numel() != _scale_numel(*weight.shape):
        raise RuntimeError("MXFP8 producer changed payload or padded scale geometry")
    layer.weight = torch.nn.Parameter(q, requires_grad=False)
    layer.register_buffer("mxfp8_weight_scale", scale.view(torch.float8_e8m0fnu))
    layer.quant_method = H3MXFP8Method()
    return layer


def _blockwise_apply(self, layer, x, bias=None):
    return _registered_apply(layer, x, bias)


def register_layer(layer, mode, name="smoke"):
    """Bind the identical compiled call boundary for model and single-layer smoke."""
    if mode not in ("mxfp8", "blockwise") or hasattr(layer, "_h3_quant_token"):
        raise ValueError("Unknown mode or duplicate projection registration")
    from vllm.model_executor.layers.quantization.online.fp8 import Fp8PerBlockOnlineLinearMethod
    expected = H3MXFP8Method if mode == "mxfp8" else Fp8PerBlockOnlineLinearMethod
    if type(layer.quant_method) is not expected or layer.weight.dtype != torch.float8_e4m3fn or layer.bias is not None:
        raise ValueError("Register only the actual bias-free quantized method")
    scale = layer.mxfp8_weight_scale if mode == "mxfp8" else layer.weight_scale_inv
    token = len(_REGISTRY)
    record = dict(name=name, layer=layer, mode=mode, scale=scale,
                  weight_ptr=layer.weight.data_ptr(), scale_ptr=scale.data_ptr(), calls=0, fused_calls=0, fused_required=False,
                  weight_shape=tuple(layer.weight.shape), scale_shape=tuple(scale.shape),
                  scale_dtype=scale.dtype, device=layer.weight.device,
                  calls_per_nfe=4 if name.endswith("attn.out_proj") else 1,
                  method_type=type(layer.quant_method), original_apply=layer.quant_method.apply)
    _REGISTRY[token] = record
    layer._h3_quant_token = token
    if mode == "blockwise":
        layer.quant_method.apply = MethodType(_blockwise_apply, layer.quant_method)
    return record


def _check_model(model, mode):
    if not getattr(model, "_h3_student_fusion_complete", False):
        raise ValueError("Quantization admission requires complete student fusion validation")
    if getattr(model, "adaln_cache", None) is None or not model.adaln_cache._loaded:
        raise ValueError("The same loaded exact AdaLN sidecar is required in both arms")
    if len(model.blocks) != 50 or torch.cuda.get_device_capability() != (12, 0):
        raise ValueError("Only the SM120 50-block FastH3 model is admitted")
    from vllm.distributed import get_tensor_model_parallel_world_size
    from vllm_omni.diffusion.distributed.parallel_state import get_sp_group
    if get_tensor_model_parallel_world_size() != 1 or get_sp_group().world_size != 8:
        raise ValueError("Only TP1/SP8 is admitted")
    if any(os.getenv(key, "0") != "0" for key in (
            "VLLM_OMNI_H3_EXACT_FC1", "VLLM_OMNI_H3_FC1_NATIVE_NN",
            "VLLM_OMNI_H3_FINE_PIPELINE")):
        raise ValueError("BF16 direct-GEMM producer bypasses cannot coexist with FP8")
    split = os.environ.get("VLLM_OMNI_H3_MXFP8_VSPLIT", "0") == "1"
    if os.environ.get("VLLM_OMNI_H3_VSPLIT", "0") != str(int(split)):
        raise RuntimeError("Only the MXFP8-aware split route is admitted")
    if split and (mode != "mxfp8" or os.environ.get("VLLM_OMNI_H3_VSPLIT_MODE") != "split"):
        raise RuntimeError("Invalid MXFP8 split configuration")
    modules = dict(model.named_modules())
    names = target_names()
    from vllm.model_executor.layers.quantization.online.fp8 import Fp8PerBlockOnlineLinearMethod
    for name, layer in modules.items():
        if name not in names:
            if isinstance(layer, LinearBase) and type(layer.quant_method) is not UnquantizedLinearMethod:
                raise ValueError("An excluded projection was quantized: " + name)
            continue
        suffix = name.split(".", 2)[2]
        if tuple(layer.weight.shape) != SHAPES[suffix] or layer.bias is not None or layer.tp_size != 1:
            raise ValueError("Main projection inventory/shape differs: " + name)
        expected = UnquantizedLinearMethod if mode == "mxfp8" else Fp8PerBlockOnlineLinearMethod
        if type(layer.quant_method) is not expected or not layer.weight.is_cuda:
            raise ValueError("Wrong source projection method/device: " + name)
    if any(name not in modules for name in names):
        raise ValueError("Missing main projections")
    for block in model.blocks:
        gate = block.attn.to_gate_compress
        if gate is None or gate.weight.dtype != torch.bfloat16 or block.adaln_proj is not None:
            raise ValueError("VSA gate must remain BF16 and AdaLN must be precomputed")
    if model.final_layer.adaln_proj is not None:
        raise ValueError("Final AdaLN must be precomputed")
    return [(name, modules[name]) for name in names]


def install_and_audit(model):
    if getattr(model, "_h3_ref2va_fusion_complete", False) and requested_mode() is not None:
        from .ref2va_mxfp8 import install
        return install(model)
    mode = requested_mode()
    fusion = requested_fusion(mode)
    if mode is None:
        if os.getenv("H3_PROFILE_PRECISION") == "bf16":
            from .profile_bf16_audit import install
            install(model)
        return
    if hasattr(model, "_h3_quant_experiment"):
        if model._h3_quant_experiment["mode"] != mode or model._h3_quant_experiment["fusion"] != fusion:
            raise ValueError("Experiment mode changed after loading")
        return
    layers = _check_model(model, mode)
    if mode == "blockwise":
        from .blockwise_experiment import audit
        audit(model)
    entries = []
    for name, layer in layers:
        if mode == "mxfp8":
            quantize_layer(layer)
            scale = layer.mxfp8_weight_scale
        else:
            scale = layer.weight_scale_inv
        entry = register_layer(layer, mode, name)
        if fusion and name.endswith(".mlp.fc2"):
            entry["fused_required"] = True
            model.blocks[int(name.split(".")[1])].mlp._h3_swiglu_mxfp8_fused = True
        entries.append(entry)
    state = dict(mode=mode, fusion=fusion, forwards=0, entries=entries)
    model._h3_quant_experiment = state

    def completed(module, args, output):
        state["forwards"] += 1
        nfe = state["forwards"]
        audit_split_nfe(entries,nfe)
        if len(entries) != 200 or any(item["calls"] != nfe * item["calls_per_nfe"] for item in entries):
            raise RuntimeError("Each actual NFE must execute all 200 real FP8 projection kernels")
        if any(item["fused_calls"] != (nfe if item["fused_required"] else 0) for item in entries):
            raise RuntimeError("Each fused FC2 must execute exactly once per NFE")
        print("H3_EXPERIMENT_QUANT_NFE " + json.dumps(dict(rank=_rank(), mode=mode, forwards=nfe,
              modules=200, fused_fc2_calls=50*nfe if fusion else 0, fc2_compiled_boundary="h3_quant_ab::swiglu_fc2" if fusion else "h3_quant_ab::projection",
              calls_per_qkv=nfe, calls_per_out_proj=4*nfe,
              calls_per_fc1=nfe, calls_per_fc2=nfe, total_projection_calls=350*nfe,
              fallback_count=0, bf16_gate=True, wire="bfloat16", compiled_boundary="h3_quant_ab::projection")), flush=True)

    model.register_forward_hook(completed)
    receipts = [dict(name=item["name"], shape=list(item["layer"].weight.shape),
                    dtype=str(item["layer"].weight.dtype), scale_dtype=str(item["scale"].dtype),
                    scale_shape=list(item["scale"].shape), method=item["method_type"].__name__)
                for item in entries]
    sources = {str(Path(inspect.getfile(fn)).resolve()): hashlib.sha256(Path(inspect.getfile(fn)).read_bytes()).hexdigest()
               for fn in (H3MXFP8Method, mxfp8_quantize_swizzled)}
    if fusion:
        from .silu_mxfp8_fusion import silu_mul_mxfp8_cuda
        fusion_source = Path(inspect.getfile(silu_mul_mxfp8_cuda)).resolve()
        sources[str(fusion_source)] = hashlib.sha256(fusion_source.read_bytes()).hexdigest()
    print("H3_EXPERIMENT_QUANT_READY " + json.dumps(dict(rank=_rank(), status="PASS", mode=mode,
          modules=200, inventory=receipts, source_sha256=sources, student_fusion_complete=True,
          exact_adaln_loaded=True, projection_calls_per_nfe=350, out_projection_segments=4, kernel=("torch.nn.functional.scaled_mm/cuBLASLt" if mode=="mxfp8" else "CutlassFp8BlockScaledMMKernel"),
          compiled_boundary="h3_quant_ab::projection", fc2_compiled_boundary="h3_quant_ab::swiglu_fc2" if fusion else "h3_quant_ab::projection",
          fused_fc2_modules=50 if fusion else 0, activation_producer="silu-fc2-fused" if fusion else "standalone", fallback_count=0,
          bf16_gate=True, wire="bfloat16", full_model_quality_claim=False)), flush=True)


# Appended to the private mxfp8_experiment module by prepare_model.py.
def _split_record(token, weight, scale):
    record = _REGISTRY[token]
    layer = record['layer']
    if (os.environ.get('VLLM_OMNI_H3_MXFP8_VSPLIT') != '1'
            or not record['name'].endswith('.attn.qkv_proj')
            or record['mode'] != 'mxfp8' or record['fused_required']
            or type(layer.quant_method) is not H3MXFP8Method
            or layer.bias is not None or layer.tp_size != 1
            or tuple(weight.shape) != (21504,5376)
            or weight.data_ptr() != record['weight_ptr']
            or scale.data_ptr() != record['scale_ptr']
            or weight.dtype != torch.float8_e4m3fn
            or scale.dtype != torch.float8_e8m0fnu
            or not weight.is_contiguous() or not scale.is_contiguous()):
        raise RuntimeError('MXFP8 split projection identity or recipe changed')
    return record


def _split_mm(q, activation_scale, weight, weight_scale):
    return scaled_mm(q,weight.t(),
        scale_a=activation_scale.view(torch.float8_e8m0fnu), scale_b=weight_scale,
        scale_recipe_a=ScalingType.BlockWise1x32,scale_recipe_b=ScalingType.BlockWise1x32,
        swizzle_a=SwizzleType.SWIZZLE_32_4_4,swizzle_b=SwizzleType.SWIZZLE_32_4_4,
        bias=None,output_dtype=torch.bfloat16)


@torch.library.custom_op('h3_quant_ab::split_qk',mutates_args=())
def _split_qk_op(x: torch.Tensor, weight: torch.Tensor, scale: torch.Tensor,
                 token: int) -> tuple[torch.Tensor,torch.Tensor,torch.Tensor]:
    record = _split_record(token,weight,scale)
    if (tuple(x.shape) != (11984,5376) or x.dtype != torch.bfloat16
            or x.device != weight.device or not x.is_contiguous()):
        raise RuntimeError('MXFP8 split requires the fixed BF16 local input')
    q, activation_scale = mxfp8_quantize_swizzled(x)
    cut = _scale_numel(14336,5376)
    qk = _split_mm(q,activation_scale,weight[:14336],scale[:cut])
    record['split_qk_calls'] = record.get('split_qk_calls',0)+1
    # All temporary tensors needed by V cross the compile boundary explicitly.
    # No tensor is retained in a Python registry between the two projections.
    return qk,q,activation_scale


@_split_qk_op.register_fake
def _split_qk_fake(x,weight,scale,token):
    return (x.new_empty((11984,14336)),
            x.new_empty((11984,5376),dtype=torch.float8_e4m3fn),
            x.new_empty((_scale_numel(11984,5376),),dtype=torch.uint8))


def project_split_qk(projection,x):
    token = projection._h3_quant_token
    return _split_qk_op(x,projection.weight,projection.mxfp8_weight_scale,token)


def project_split_v(projection,q,activation_scale,qk):
    record = _split_record(projection._h3_quant_token,projection.weight,projection.mxfp8_weight_scale)
    cut = _scale_numel(14336,5376)
    value = _split_mm(q,activation_scale,projection.weight[14336:],projection.mxfp8_weight_scale[cut:])
    if record.get('split_v_calls',0)+1 != record['split_qk_calls']:
        raise RuntimeError('MXFP8 split QK/V call sequence changed')
    if not record.get('split_raw_gate_passed',False):
        if record['split_qk_calls'] != 1:
            raise RuntimeError('MXFP8 raw gate must precede all measured requests')
        reference = _split_mm(q,activation_scale,projection.weight,projection.mxfp8_weight_scale)
        qk_equal = torch.equal(reference[:,:14336].contiguous().view(torch.uint8),qk.view(torch.uint8))
        v_equal = torch.equal(reference[:,14336:].contiguous().view(torch.uint8),value.view(torch.uint8))
        print('H3_MXSPLIT_RAW_GATE '+json.dumps(dict(rank=_rank(),name=record['name'],
            qk_byteexact=qk_equal,v_byteexact=v_equal,call_index=1,
            status='PASS' if qk_equal and v_equal else 'FAIL')),flush=True)
        if not qk_equal or not v_equal:
            raise RuntimeError('Actual model Q/K/V bytes differ after MXFP8 split')
        record['split_raw_gate_passed'] = True
    record['split_v_calls'] = record.get('split_v_calls',0)+1
    record['calls'] += 1  # One logical QKV projection, two actual GEMMs.
    return value


def audit_split_nfe(entries,nfe):
    if os.environ.get('VLLM_OMNI_H3_MXFP8_VSPLIT') != '1':
        return
    qkv = [e for e in entries if e['name'].endswith('.attn.qkv_proj')]
    assert len(qkv) == 50
    if any(e.get('split_qk_calls') != nfe or e.get('split_v_calls') != nfe
           or not e.get('split_raw_gate_passed') for e in qkv):
        raise RuntimeError('Every split QK and V must execute once per actual NFE')
    print('H3_MXSPLIT_NFE '+json.dumps(dict(rank=_rank(),forwards=nfe,modules=50,
        qk_calls=50*nfe,v_calls=50*nfe,raw_gates_passed=50,
        logical_projection_calls=350*nfe,actual_projection_gemm_calls=400*nfe,
        raw_gate_extra_gemms=50,quantizations_per_qkv=1,
        qk_compiled_boundary='h3_quant_ab::split_qk',v_execution='eager_dedicated_stream',
        status='PASS')),flush=True)
