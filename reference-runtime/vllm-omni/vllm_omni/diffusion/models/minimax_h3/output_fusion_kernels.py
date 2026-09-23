"""Isolated H3 output-fusion candidates, not enabled in any service.

The CPU interpreter checks indexing/math; only CUDA qualification can establish
GPU equivalence and performance. Compile with enable_fp_fusion=False: separate
FP32 multiplication/addition and precise division are part of the contract.
"""
from __future__ import annotations

import os
import torch
import triton
import triton.language as tl


@triton.jit
def blend_kernel(A, B, WA, WB, OUT,
                 SHAPE: tl.constexpr, AS: tl.constexpr, BS: tl.constexpr,
                 A_EXTENT: tl.constexpr, AXIS: tl.constexpr,
                 EXTENT: tl.constexpr, N: tl.constexpr, BLOCK: tl.constexpr):
    i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    mask=i<N
    q=i
    ao=tl.full((BLOCK,),0,tl.int32)
    bo=tl.full((BLOCK,),0,tl.int32)
    position=tl.full((BLOCK,),0,tl.int32)
    for dim in tl.static_range(4,-1,-1):
        coordinate=q%SHAPE[dim]
        q=q//SHAPE[dim]
        bo+=coordinate*BS[dim]
        if dim==AXIS:
            position=coordinate
            coordinate=coordinate+A_EXTENT-EXTENT
        ao+=coordinate*AS[dim]
    b=tl.load(B+bo,mask,other=0.)
    overlap=mask & (position<EXTENT)
    a=tl.load(A+ao,overlap,other=0.)
    wa=tl.load(WA+position,overlap,other=0.)
    wb=tl.load(WB+position,overlap,other=0.)
    # enable_fp_fusion=False retains two rounded products then the sum.
    blended=(a*wa)+(b*wb)
    tl.store(OUT+i,tl.where(position<EXTENT,blended,b),mask)


@triton.jit
def rgb_kernel(X, MEAN, STD, OUT,
               STRIDES: tl.constexpr, HEIGHT: tl.constexpr, WIDTH: tl.constexpr,
               N: tl.constexpr, BLOCK: tl.constexpr):
    i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    valid=i<N
    c=i%3
    pixel=i//3
    w=pixel%WIDTH
    h=(pixel//WIDTH)%HEIGHT
    t=pixel//(WIDTH*HEIGHT)
    x=tl.load(X+c*STRIDES[1]+t*STRIDES[2]+h*STRIDES[3]+w*STRIDES[4],valid,other=0.)
    mean=tl.load(MEAN+c)
    std=tl.load(STD+c)
    value=tl.div_rn(x-mean,std)
    # The baseline clamps twice around layout conversion; clamp is idempotent.
    # The final uint8 NaN result is zero in the reference CPU implementation;
    # CUDA qualification explicitly checks the same case before admission.
    value=tl.where(value!=value,0.,tl.minimum(tl.maximum(value,0.),1.))
    scaled=value*255.
    lower=tl.floor(scaled)
    frac=scaled-lower
    # Exact ties-to-even on the bounded [0,255] input; no approximate division.
    increment=(frac>0.5) | ((frac==0.5) & ((lower.to(tl.int32)&1)!=0))
    result=(lower+increment.to(tl.float32)).to(tl.uint8)
    tl.store(OUT+i,result,valid)


def supported_device(tensor):
    return tensor.is_cuda or (tensor.device.type=='cpu' and os.environ.get('TRITON_INTERPRET')=='1')


class BlendCandidate:
    """Cache the original Torch weights, fuse multiply/add/cat for FP32 BCTHW.

    Cache readiness is explicit so a later call on another CUDA stream is safe.
    The cache is object-local; discard the object when its owning model unloads.
    Unsupported shapes/dtypes use the original function supplied by the caller.
    """
    def __init__(self,reference,block=512):
        self.reference=reference
        self.block=block
        self.weights={}
        self.calls=0
        self.fallbacks=0

    def __call__(self,a,b,blend_extent,dim):
        axis=dim%5
        admissible=(a.ndim==b.ndim==5 and a.dtype==b.dtype==torch.float32
                    and a.device==b.device and supported_device(a)
                    and -5<=dim<5 and not torch.is_grad_enabled()
                    and all(a.shape[k]==b.shape[k] for k in range(5) if k!=axis)
                    and all(s>0 for s in (*a.stride(),*b.stride()))
                    and a.numel()<2**31 and b.numel()<2**31
                    and sum((n-1)*s for n,s in zip(a.shape,a.stride()))<2**31
                    and sum((n-1)*s for n,s in zip(b.shape,b.stride()))<2**31)
        if not admissible:
            self.fallbacks+=1
            return self.reference(a,b,blend_extent,dim)
        extent=min(a.shape[axis],b.shape[axis],blend_extent)
        if extent<=0:
            self.fallbacks+=1
            return self.reference(a,b,blend_extent,dim)
        key=(a.device,extent)
        if key not in self.weights:
            positions=torch.arange(extent,device=b.device,dtype=b.dtype)
            wa=1-positions/extent
            wb=positions/extent
            ready=None
            if b.is_cuda:
                ready=torch.cuda.Event();ready.record(torch.cuda.current_stream(b.device))
            self.weights[key]=(wa,wb,ready)
        wa,wb,ready=self.weights[key]
        if ready is not None:
            stream=torch.cuda.current_stream(b.device)
            stream.wait_event(ready)
            wa.record_stream(stream);wb.record_stream(stream)
        out=torch.empty(b.shape,device=b.device,dtype=b.dtype)
        blend_kernel[(triton.cdiv(b.numel(),self.block),)](
            a,b,wa,wb,out,tuple(b.shape),tuple(a.stride()),tuple(b.stride()),
            a.shape[axis],axis,extent,b.numel(),self.block,
            num_warps=4,enable_fp_fusion=False)
        self.calls+=1
        return out


class RGBCandidate:
    """Fuse Normalize inverse + clamp + BCTHW->THWC + round/u8 for FP32.

    mean/std are the actual inverse Normalize constants (not the forward image
    mean/std). Conversion to FP32 happens exactly as in torchvision Normalize.
    """
    def __init__(self,mean,std,device,block=512):
        self.device=torch.device(device)
        self.mean_values=tuple(mean);self.std_values=tuple(std)
        mean_cpu=torch.as_tensor(mean,dtype=torch.float32)
        std_cpu=torch.as_tensor(std,dtype=torch.float32)
        if mean_cpu.shape!=(3,) or std_cpu.shape!=(3,) or (std_cpu==0).any():
            raise ValueError('Expected three nonzero inverse Normalize std values')
        self.mean=mean_cpu.to(self.device);self.std=std_cpu.to(self.device)
        self.device=self.mean.device
        self.ready=None
        if self.device.type=='cuda':
            self.ready=torch.cuda.Event();self.ready.record(torch.cuda.current_stream(self.device))
        self.block=block

    def __call__(self,x,*,height,width):
        if not (x.ndim==5 and x.shape[:2]==(1,3) and x.dtype==torch.float32
                and x.device==self.device and supported_device(x)
                and not torch.is_grad_enabled() and all(s>0 for s in x.stride())
                and 0<height<=x.shape[-2] and 0<width<=x.shape[-1]
                and x.shape[2]>0 and x.numel()<2**31
                and sum((n-1)*s for n,s in zip(x.shape,x.stride()))<2**31):
            raise ValueError('RGB fusion requires inference FP32 B1C3THW and a valid top-left crop')
        if self.ready is not None:
            stream=torch.cuda.current_stream(x.device)
            stream.wait_event(self.ready)
            self.mean.record_stream(stream);self.std.record_stream(stream)
        out=torch.empty((x.shape[2],height,width,3),dtype=torch.uint8,device=x.device)
        rgb_kernel[(triton.cdiv(out.numel(),self.block),)](
            x,self.mean,self.std,out,tuple(x.stride()),height,width,out.numel(),self.block,
            num_warps=4,enable_fp_fusion=False)
        return out
