"""Reuse FastH3's Sage compute kernel with every KV block retained (dense).

No learned VSA gate, top-k selection or compression. QK INT8 / PV FP8 changes
precision; all real reference and generated rows remain in attention.
"""
from functools import lru_cache
import torch
from vllm.logger import init_logger

@lru_cache(maxsize=16)
def plan(device, batch, heads, qrows, krows):
    nq=(qrows+63)//64;nk=(krows+63)//64
    indices=torch.arange(nk,device=device,dtype=torch.int32).view(1,1,1,nk).expand(batch,heads,nq,nk).contiguous()
    counts=torch.full((batch,heads,nq),nk,device=device,dtype=torch.int32)
    sizes=torch.full((nk,),64,device=device,dtype=torch.int32)
    sizes[-1]=krows-64*(nk-1)
    event=torch.cuda.Event();event.record()
    workspace=torch.empty((512,),device=device,dtype=torch.uint8)
    return indices,counts,sizes,event,workspace

@torch.library.custom_op('h3_ref2va::dense_sage',mutates_args=())
def dense_sage(q:torch.Tensor,k:torch.Tensor,v:torch.Tensor,scale:float)->torch.Tensor:
    if (q.ndim!=4 or k.ndim!=4 or v.shape!=k.shape or q.shape[0]!=1
        or q.shape[0]!=k.shape[0] or q.shape[2:]!=k.shape[2:] or q.shape[-1]!=128
        or any(t.dtype!=torch.bfloat16 or not t.is_cuda for t in (q,k,v))):
        raise ValueError('Dense Sage requires B1 BF16 MHA NHD128 on CUDA')
    from vllm_omni.diffusion.models.minimax_h3.ref2va_precision import attention_mode
    if attention_mode() == 'bf16':
        from torch.nn.attention import sdpa_kernel, SDPBackend
        with sdpa_kernel([SDPBackend.CUDNN_ATTENTION]):
            out=torch.nn.functional.scaled_dot_product_attention(
                q.transpose(1,2),k.transpose(1,2),v.transpose(1,2),
                dropout_p=0.0,is_causal=False,scale=scale)
        return out.transpose(1,2).contiguous()
    from vllm_omni.diffusion.attention.ops.h3_cake_grid_constant import configure
    from flashinfer.cute_dsl.sparse.bsa_utils.sage_quant_sm120 import quantize_sage_qkv_sm120
    from flashinfer.cute_dsl.sparse.bsa_attn_sm120_cake import bsa_attn_sm120_blk64_sage_fwd
    configure()
    indices,counts,sizes,ready,workspace=plan(q.device,q.shape[0],q.shape[2],q.shape[1],k.shape[1])
    stream=torch.cuda.current_stream(q.device);stream.wait_event(ready)
    for t in (indices,counts,sizes,workspace):t.record_stream(stream)
    quant=quantize_sage_qkv_sm120(q.transpose(1,2),k.transpose(1,2),v.transpose(1,2))
    out=torch.empty((q.shape[0],q.shape[2],q.shape[1],128),device=q.device,dtype=q.dtype)
    bsa_attn_sm120_blk64_sage_fwd(*quant,indices,indices.shape[-1],block_sizes=sizes,
        q2k_block_nums=counts,softmax_scale=scale,out=out,tma_descriptor_workspace=workspace)
    init_logger(__name__).info_once('REF2VA_DENSE_SAGE_ACTIVE all_kv_blocks=1 topk_pruning=0 learned_vsa_gate=0 QK=INT8 PV=FP8 wire=BF16')
    return out.transpose(1,2).contiguous()

@dense_sage.register_fake
def _fake(q,k,v,scale):return torch.empty_like(q,memory_format=torch.contiguous_format)
