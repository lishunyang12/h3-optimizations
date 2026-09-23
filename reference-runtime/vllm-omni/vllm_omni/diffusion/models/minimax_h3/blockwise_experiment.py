"""Isolated H3 online E4M3 blockwise experiment; BF16 checkpoint and wire.

Quantize the fused FastH3 weights after loading, in exactly 200 main-block
linears. This is a distinct contract from the existing per-tensor FP8 lane.
"""
import os
import re
import json
from collections import Counter
import torch
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
from vllm.model_executor.layers.quantization.fp8 import Fp8Config
from vllm.model_executor.layers.quantization.online.fp8 import Fp8PerBlockOnlineLinearMethod
from vllm.model_executor.layers.quantization.utils.quant_utils import kFp8Dynamic128Sym, kFp8Static128BlockSym

ENV = 'VLLM_OMNI_H3_EXPERIMENT_BLOCKWISE'
MODE = 'qkv-o-mlp-fp8-e4m3fn-a1x128-w128x128-online-v1'
TARGET = re.compile(r'^blocks\.(\d+)\.(attn\.(qkv_proj|out_proj)|mlp\.(fc1|fc2))$')
logger = init_logger(__name__)


class H3BlockwiseConfig(Fp8Config):
    def __init__(self):
        super().__init__(is_checkpoint_fp8_serialized=False, activation_scheme='dynamic')

    def get_quant_method(self, layer, prefix):
        if isinstance(layer, LinearBase):
            match = TARGET.fullmatch(prefix)
            if match and int(match[1]) < 50:
                return Fp8PerBlockOnlineLinearMethod()
            return UnquantizedLinearMethod()
        return None


def configure(quant_config):
    from .mxfp8_experiment import requested_mode
    mode = requested_mode()
    if mode is None:
        return quant_config
    if quant_config is not None:
        raise ValueError('Blockwise experiment requires an otherwise BF16 DiT config')
    if os.getenv('VLLM_OMNI_MINIMAX_H3_FP8_SCOPE', 'none') != 'none':
        raise ValueError('Blockwise experiment cannot reuse the per-tensor FP8 contract')
    if os.getenv('VLLM_OMNI_FLASHINFER_ULYSSES_QKV_E4M3_TRANSPORT', '0') != '0':
        raise ValueError('Blockwise compute A/B requires unchanged BF16 QKV wire')
    return H3BlockwiseConfig() if mode == "blockwise" else None


def audit(model):
    if type(model.quant_config) is not H3BlockwiseConfig:
        return
    if getattr(model, '_blockwise_experiment_audited', False):
        return
    assert torch.cuda.get_device_capability() == (12, 0)
    assert torch.distributed.get_world_size() == 8
    expected = {f'blocks.{i}.{suffix}' for i in range(50) for suffix in ('attn.qkv_proj','attn.out_proj','mlp.fc1','mlp.fc2')}
    found = set()
    kernels = Counter()
    for name, module in model.named_modules():
        method = getattr(module, 'quant_method', None)
        if name not in expected:
            if isinstance(module, LinearBase):
                assert type(method) is UnquantizedLinearMethod, (name, type(method))
            continue
        assert type(method) is Fp8PerBlockOnlineLinearMethod, (name, type(method))
        assert method.activation_quant_key == kFp8Dynamic128Sym
        assert method.weight_quant_key == kFp8Static128BlockSym
        assert method.input_dtype == method.out_dtype == torch.bfloat16
        assert module.tp_size == 1 and module.weight_block_size == [128,128]
        assert module.weight.dtype == torch.float8_e4m3fn
        assert module.weight.device.type == 'cuda'
        assert module._already_called_process_weights_after_loading
        assert module.input_scale is None
        n,k = module.weight.shape
        scales = module.weight_scale_inv
        assert scales.shape == ((n+127)//128,(k+127)//128), (name,scales.shape)
        assert scales.dtype == torch.float32 and torch.isfinite(scales).all() and (scales > 0).all()
        kernel = type(method.fp8_linear).__name__
        assert kernel == 'CutlassFp8BlockScaledMMKernel', (name, kernel)
        kernels[kernel] += 1
        found.add(name)
    assert found == expected, sorted(expected-found)
    marker = dict(rank=torch.distributed.get_rank(),status='ok',mode=MODE,modules=len(found),qkv=50,o=50,mlp_fc1=50,mlp_fc2=50,activation_group=[1,128],weight_group=[128,128],kernels=dict(kernels),wire='bfloat16',gate='bfloat16',fallbacks=0)
    logger.info('H3_BLOCKWISE_AUDIT %s',json.dumps(marker,sort_keys=True))
    model._blockwise_experiment_audited = True
