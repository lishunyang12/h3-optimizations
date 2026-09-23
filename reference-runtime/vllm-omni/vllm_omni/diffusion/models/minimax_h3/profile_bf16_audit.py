"""Read-only BF16 inventory and NFE admission for the paired Nsight capture."""
import hashlib
import json
from pathlib import Path

import torch
from vllm.model_executor.layers.linear import UnquantizedLinearMethod


def install(model):
    from .mxfp8_experiment import _check_model

    if hasattr(model, "_h3_profile_bf16_audit"):
        raise RuntimeError("Duplicate BF16 profiling audit")
    layers = _check_model(model, "mxfp8")  # Validates the pre-quantization BF16 inventory.
    pointers = {name: layer.weight.data_ptr() for name, layer in layers}
    rank = torch.distributed.get_rank()
    count = 0
    model._h3_profile_bf16_audit = True

    def check():
        for name, layer in layers:
            if (type(layer.quant_method) is not UnquantizedLinearMethod
                    or layer.weight.dtype != torch.bfloat16
                    or layer.weight.data_ptr() != pointers[name]):
                raise RuntimeError("BF16 projection changed: " + name)
        if any(getattr(block.mlp, "_h3_swiglu_mxfp8_fused", False) for block in model.blocks):
            raise RuntimeError("SwiGLU fusion must be disabled in the BF16 arm")

    check()
    source = Path(__file__)
    print("H3_PROFILE_BF16_READY " + json.dumps({
        "rank": rank, "status": "PASS", "modules": len(layers),
        "inventory": [{"name": name, "shape": list(layer.weight.shape),
                       "dtype": str(layer.weight.dtype), "method": type(layer.quant_method).__name__}
                      for name, layer in layers],
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "student_fusion_complete": True, "exact_adaln_loaded": True,
    }), flush=True)

    def complete(module, args, output):
        nonlocal count
        check()
        count += 1
        print("H3_PROFILE_BF16_NFE " + json.dumps({
            "rank": rank, "forwards": count, "modules": len(layers),
            "dtype": "torch.bfloat16", "swiglu_fusion": False,
        }), flush=True)

    model.register_forward_hook(complete)
