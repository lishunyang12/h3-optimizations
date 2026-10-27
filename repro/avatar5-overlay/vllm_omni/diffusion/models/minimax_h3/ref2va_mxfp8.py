"""Dense Ref2VA admission for existing MXFP8 and fused SwiGLU kernels."""
import json
import os
import torch
from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
from . import mxfp8_experiment as mx
from . import ref2va_bf16_weights as bf

def install(model):
    if hasattr(model, '_h3_quant_experiment'):return
    if mx.requested_mode() != 'mxfp8' or not model._h3_ref2va_fusion_complete:
        raise ValueError('Dense Ref2VA requires validated Turbo fusion and MXFP8')
    from vllm.distributed import get_tensor_model_parallel_world_size
    from vllm_omni.diffusion.distributed.parallel_state import get_sp_group
    if (torch.cuda.get_device_capability() != (12,0) or len(model.blocks)!=50
        or get_tensor_model_parallel_world_size()!=1 or get_sp_group().world_size!=8
        or model.adaln_cache is None or not model.adaln_cache._loaded):
        raise ValueError('Ref2VA MXFP8 requires resident SM120 TP1/SP8 and exact AdaLN')
    if any(os.getenv(k,'0')!='0' for k in ('VLLM_OMNI_H3_EXPERIMENT_OVERLAP',
        'VLLM_OMNI_H3_MXFP8_VSPLIT','VLLM_OMNI_H3_VSPLIT')):
        raise ValueError('Dense Ref2VA cannot use the VSA-specific split schedule')
    modules=dict(model.named_modules());targets=set(mx.target_names())
    if not targets.issubset(modules):raise ValueError('Missing Ref2VA projections')
    for name,layer in modules.items():
        if isinstance(layer,LinearBase) and type(layer.quant_method) is not UnquantizedLinearMethod:
            raise ValueError('Ref2VA must be loaded as BF16 before MXFP8: '+name)
        if name in targets:
            suffix=name.split('.',2)[2]
            if tuple(layer.weight.shape)!=mx.SHAPES[suffix] or layer.tp_size!=1 or layer.bias is not None:
                raise ValueError('Ref2VA projection geometry changed: '+name)
    if any(b.attn.to_gate_compress is not None or b.adaln_proj is not None for b in model.blocks):
        raise ValueError('Dense Ref2VA must have no VSA gate and must use cached AdaLN')
    if model.final_layer.adaln_proj is not None:raise ValueError('Final AdaLN must be cached')
    fused=mx.requested_fusion('mxfp8');entries=[]
    if bf.enabled() and not fused:
        raise ValueError('Original BF16 staging requires fused FC2 completion scheduling')
    for name in mx.target_names():
        if bf.enabled():bf.preserve(modules[name],name)
        layer=mx.quantize_layer(modules[name]);entry=mx.register_layer(layer,'mxfp8',name)
        entry.update(dense_ref2va=True,calls_per_nfe=1,bf16_retained=bf.enabled(),bf16_calls=0)
        if fused and name.endswith('.mlp.fc2'):
            entry['fused_required']=True
            model.blocks[int(name.split('.')[1])].mlp._h3_swiglu_mxfp8_fused=True
        entries.append(entry)
    if bf.enabled():bf.install(entries[0]['device'])
    state=dict(mode='mxfp8',fusion=fused,forwards=0,bf16_forwards=0,entries=entries)
    def before(module,args):
        if bf.enabled():bf.start_step(entries[0])
    model.register_forward_pre_hook(before)
    model._h3_quant_experiment=state
    def completed(module,args,output):
        state['forwards']+=1;n=state['forwards']
        if len(entries)!=200 or any(e['calls']!=n for e in entries):
            raise RuntimeError('Each Ref2VA step must execute every real MXFP8 projection once')
        if bf.use_bf16(entries[0]):state['bf16_forwards']+=1
        b=state['bf16_forwards']
        if any(e['bf16_calls']!=b for e in entries):
            raise RuntimeError('Actual BF16 projection count differs')
        if any(e['fused_calls']!=((n-b) if e['fused_required'] else 0) for e in entries):
            raise RuntimeError('Ref2VA fused FC2 dispatch count mismatch')
        print('REF2VA_MXFP8_NFE '+json.dumps(dict(rank=mx._rank(),forwards=n,modules=200,
            calls=200*n,bf16_projection_calls=200*b,mxfp8_projection_calls=200*(n-b),fused_fc2_calls=50*(n-b) if fused else 0,dense_attention=True,fallback=0)),flush=True)
    model.register_forward_hook(completed)
    print('REF2VA_MXFP8_READY '+json.dumps(dict(rank=mx._rank(),modules=200,
        fused_fc2=fused,source='verified Ref2VA Turbo BF16 fusion',VSA=False)),flush=True)
