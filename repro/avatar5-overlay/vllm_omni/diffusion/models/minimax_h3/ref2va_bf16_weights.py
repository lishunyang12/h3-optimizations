"""Audited original-BF16 first-step Ref2VA projections with two staged blocks.

Opt-in experiment; CPU weights are retained before MXFP8 conversion, never
reconstructed from quantized weights. No generic layer offload is enabled.
"""
import json
import os
import torch

_HOST = {}
_BUFFERS = []
_STAGED = [None, None]
_READY = [None, None]
_CONSUMED = [None, None]
_TRANSFER = None
_POLICY = 'mxfp8'
_BF16_STEPS = {'mxfp8': 0, 'first_bf16': 1, 'first_two_bf16': 2, 'all_bf16': 4}


def enabled():
    flag=os.environ.get('H3_REF2VA_BF16_WEIGHT_TEST','0')
    if flag not in ('0','1'):raise ValueError('Invalid BF16 weight experiment flag')
    return flag=='1'


def begin(policy):
    global _POLICY
    if policy not in _BF16_STEPS or (_BF16_STEPS[policy] and not enabled()):
        raise ValueError('Requested Ref2VA weight precision was not loaded')
    _POLICY=policy


def use_bf16(record):
    if not record.get('bf16_retained') or not _BF16_STEPS[_POLICY]:return False
    from vllm_omni.diffusion.forward_context import get_forward_context
    ctx=get_forward_context()
    if ctx.total_denoise_steps!=4 or ctx.denoise_step_idx not in range(4):
        raise RuntimeError('BF16 weight test requires four explicit denoise steps')
    return ctx.denoise_step_idx < _BF16_STEPS[_POLICY]


def preserve(layer,name):
    weight=layer.weight
    if weight.dtype!=torch.bfloat16 or not weight.is_cuda or not weight.is_contiguous():
        raise RuntimeError('Expected original fused BF16 weight')
    index=int(name.split('.')[1]);suffix=name.split('.',2)[2]
    host=torch.empty_like(weight,device='cpu',pin_memory=True)
    host.copy_(weight)
    _HOST.setdefault(index,{})[suffix]=host
    layer._h3_original_bf16=host
    layer._h3_original_bf16_ptr=host.data_ptr()


def install(device):
    global _TRANSFER
    if set(_HOST)!=set(range(50)) or any(len(block)!=4 for block in _HOST.values()):
        raise RuntimeError('Expected 50 complete original BF16 blocks')
    torch.cuda.empty_cache()
    _TRANSFER=torch.cuda.Stream(device=device)
    for _ in range(2):
        _BUFFERS.append({suffix:torch.empty_like(w,device=device) for suffix,w in _HOST[0].items()})
    print('REF2VA_BF16_WEIGHTS_READY '+json.dumps(dict(rank=torch.distributed.get_rank(),
        original_modules=200,source='original_turbo_fused_before_quantization',host_pinned_bytes=sum(w.numel()*w.element_size() for b in _HOST.values() for w in b.values()),
        staging_buffers=2,staging_bytes=sum(w.numel()*w.element_size() for b in _BUFFERS for w in b.values()),memory_allocated=torch.cuda.memory_allocated(),memory_reserved=torch.cuda.memory_reserved())),flush=True)


def prefetch(index):
    if index not in _HOST:return
    slot=index%2
    # A slot can be reused by the next block or by block 0/1 of the next
    # BF16 step. Keep the final block's event even when there is no next
    # block inside this step; otherwise step-boundary prefetch can overwrite
    # block 48/49 weights while their GEMMs are still queued.
    if _CONSUMED[slot] is not None:_TRANSFER.wait_event(_CONSUMED[slot])
    with torch.cuda.stream(_TRANSFER):
        for suffix,source in _HOST[index].items():_BUFFERS[slot][suffix].copy_(source,non_blocking=True)
        ready=torch.cuda.Event();ready.record(_TRANSFER)
    _STAGED[slot]=index;_READY[slot]=ready


def start_step(record):
    if use_bf16(record):
        prefetch(0);prefetch(1)


def weight_for(record):
    index=int(record['name'].split('.')[1]);suffix=record['name'].split('.',2)[2];slot=index%2
    source=record['layer']._h3_original_bf16
    if (source.data_ptr()!=record['layer']._h3_original_bf16_ptr or not source.is_pinned()
            or source.dtype!=torch.bfloat16 or tuple(source.shape)!=record['weight_shape']
            or _STAGED[slot]!=index or _READY[slot] is None):
        raise RuntimeError('BF16 source or prefetch state changed')
    torch.cuda.current_stream(record['device']).wait_event(_READY[slot])
    return _BUFFERS[slot][suffix]


def finished_fc2(record):
    index=int(record['name'].split('.')[1])
    consumed=torch.cuda.Event();consumed.record(torch.cuda.current_stream(record['device']))
    _CONSUMED[index%2]=consumed
    prefetch(index+2)
