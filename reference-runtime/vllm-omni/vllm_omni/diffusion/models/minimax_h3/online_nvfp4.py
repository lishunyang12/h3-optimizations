"""Experimental H3 VAE NVFP4 A4W4; explicit CUDA quantizer and cuBLASLt.

Original FP32 weights, dynamic FP32 tensor scales, E4M3 scales per 16
elements, E2M1 packed data. FP16 activation/output boundaries remain intact.
"""
import collections
import json
import os

import torch
from torch import nn

FLAG = 'VLLM_OMNI_H3_VAE_NVFP4'


def enabled():
    value = os.environ.get(FLAG, '0')
    if value not in ('0', '1'):
        raise ValueError(f'{FLAG} must be 0 or 1')
    if value == '1' and any(os.environ.get(k, '0') != '0' for k in
                           ('VLLM_OMNI_H3_VAE_MXFP8', 'VLLM_OMNI_H3_VAE_INT8_CONVROT')):
        raise ValueError('Select exactly one VAE quantization method')
    return value == '1'


class OnlineNVFP4Linear(nn.Module):
    def __init__(self, linear, name):
        super().__init__()
        from comfy_kitchen.backends.cuda import quantize_nvfp4, scaled_mm_nvfp4
        from comfy_kitchen.backends.eager.quantization import quantize_nvfp4 as quantize_weight
        if linear.weight.dtype != torch.float32 or not linear.weight.is_cuda:
            raise RuntimeError('NVFP4 requires original CUDA FP32 weights')
        self.in_features, self.out_features = linear.in_features, linear.out_features
        assert self.in_features % 32 == 0 and self.out_features % 32 == 0
        self.name, self.calls = name, 0
        self._quantize, self._mm = quantize_nvfp4, scaled_mm_nvfp4
        with torch.no_grad(), torch.autocast('cuda', enabled=False):
            w = linear.weight.detach().contiguous()
            tensor_scale = (w.abs().amax() / (448 * 6)).clamp_min(1e-12).reshape(1)
            # CUDA FP4 uses the low nibble for the first matrix element.
            q, scale = quantize_weight(w, tensor_scale, pad_16x=False, hi_first=False)
        assert q.dtype == torch.uint8 and scale.dtype == torch.float8_e4m3fn
        self.register_buffer('weight', q)
        self.register_buffer('weight_scale', scale)
        self.register_buffer('weight_tensor_scale', tensor_scale)
        self.register_buffer('bias', None if linear.bias is None else linear.bias.detach().half())

    def forward(self, x):
        if not x.is_cuda or x.shape[-1] != self.in_features:
            raise RuntimeError('Unsupported H3 VAE NVFP4 input')
        if not torch.is_autocast_enabled('cuda') or torch.get_autocast_dtype('cuda') != torch.float16:
            raise RuntimeError('NVFP4 qualified only under FP16 VAE autocast')
        shape = x.shape
        with torch.autocast('cuda', enabled=False):
            x2d = x.half().reshape(-1, self.in_features).contiguous()
            # Device-side dynamic scale, no host scalar read or calibration reuse.
            tensor_scale = (x2d.abs().amax().float() / (448 * 6)).clamp_min(1e-12).reshape(1)
            q, scale = self._quantize(x2d, tensor_scale, pad_16x=True, hi_first=False)
            assert q.dtype == torch.uint8 and scale.dtype == torch.float8_e4m3fn
            out = self._mm(q, self.weight, tensor_scale, self.weight_tensor_scale,
                           scale, self.weight_scale, bias=self.bias, out_dtype=torch.float16)
            assert out.dtype == torch.float16 and out.shape == (q.shape[0], self.out_features)
            self.calls += 1
            return out[:x2d.shape[0]].reshape(*shape[:-1], self.out_features)


def install(decoder):
    from importlib.metadata import version
    assert version('comfy-kitchen') == '0.2.33'
    assert len(decoder.transformer_blocks) == 36
    modules = []
    for i, block in enumerate(decoder.transformer_blocks):
        for owner, key, kind in ((block.attn, 'to_qkv', 'qkv'), (block.attn, 'to_out', 'out'),
                                 (block.ff, 'w1', 'fc1'), (block.ff, 'w2', 'fc2')):
            old = getattr(owner, key)
            assert isinstance(old, nn.Linear)
            name = f'transformer_blocks.{i}.{"attn" if kind in ("qkv", "out") else "ff"}.{key}'
            module = OnlineNVFP4Linear(old, name)
            module.kind = kind
            setattr(owner, key, module)
            modules.append(module)
    decoder._h3_online_nvfp4_modules = tuple(modules)
    return dict(rank=torch.distributed.get_rank() if torch.distributed.is_initialized() else 0,
                layers=144, group_size=16, weight_source='original_fp32',
                payload='E2M1_low_nibble_first', block_scale='E4M3', tensor_scale='FP32_dynamic',
                output_dtype='float16', backend='comfy_kitchen_cuda_cublaslt', version='0.2.33',
                approximate=True, fallback=0)


def complete(decoder, rank, logger, *, expected_decoder_calls=None, expected_out_calls=None):
    modules = getattr(decoder, '_h3_online_nvfp4_modules', ())
    if not modules:
        return
    counts = collections.Counter()
    for module in modules:
        if expected_decoder_calls is None or expected_out_calls is None:
            raise RuntimeError('NVFP4 requires a concrete request-derived dispatch plan')
        expected = expected_out_calls if module.kind == 'out' else expected_decoder_calls
        if module.calls != expected:
            raise RuntimeError(f'NVFP4 actual dispatch mismatch: {module.name}: {module.calls} != {expected}')
        counts[module.kind] += module.calls
    logger.info('H3_VAE_NVFP4_COMPLETE %s', json.dumps(dict(rank=rank, layers=144,
        calls=dict(counts), group_size=16, fallback=0, status='PASS'), sort_keys=True))
    for module in modules:
        module.calls = 0
