"""Cache immutable Normalize constants while retaining its exact eager math."""
import copy
import json
import os

import torch
from torchvision.transforms import Normalize
from vllm.logger import init_logger


ENV = "VLLM_OMNI_H3_VAE_CACHED_NORMALIZE"


class CachedNormalize:
    def __init__(self, original, device, dtype):
        if type(original) is not Normalize or original.inplace:
            raise ValueError("Cached output normalization requires the original non-inplace Normalize")
        if dtype != torch.float32 or device.type != "cuda":
            raise ValueError("Cached output normalization is admitted only for FP32 CUDA output")
        self.device = torch.device(device)
        self.dtype = dtype
        # Match torchvision's Python-float -> actual tensor dtype conversion.
        mean = torch.as_tensor(original.mean, dtype=dtype, device="cpu")
        std = torch.as_tensor(original.std, dtype=dtype, device="cpu")
        if mean.shape != (3,) or std.shape != (3,) or bool((std == 0).any()):
            raise ValueError("Invalid RGB normalization constants")
        self.mean = mean.to(device).view(-1, 1, 1)
        self.std = std.to(device).view(-1, 1, 1)
        self.ready = torch.cuda.Event()
        self.ready.record(torch.cuda.current_stream(device))

    def __call__(self, tensor):
        if tensor.dtype != self.dtype or tensor.device != self.mean.device:
            raise ValueError("Cached output normalization device/dtype changed")
        if tensor.ndim < 3 or tensor.shape[-3] != 3:
            raise ValueError("Cached output normalization requires RGB channel dimension")
        stream = torch.cuda.current_stream(tensor.device)
        stream.wait_event(self.ready)
        self.mean.record_stream(stream)
        self.std.record_stream(stream)
        # Same clone and separate elementwise kernels as torchvision normalize.
        # Do not rewrite this as multiplication/addition or fuse it.
        return tensor.clone().sub_(self.mean).div_(self.std)


class CachedOutputProcessor:
    def __init__(self, processor, device):
        self.original = processor
        self.mean_values = tuple(processor.transform_rev.mean)
        self.std_values = tuple(processor.transform_rev.std)
        self.processor = copy.copy(processor)
        self.processor.transform_rev = CachedNormalize(processor.transform_rev, device, torch.float32)
        self.request = 0
        self.calls = 0
        self.raw_checks = 0

    def begin(self):
        if (tuple(self.original.transform_rev.mean) != self.mean_values
                or tuple(self.original.transform_rev.std) != self.std_values):
            raise RuntimeError("Output normalization constants changed after cache creation")
        self.request += 1
        self.calls = 0
        self.raw_checks = 0

    def revert_tensor(self, tensor):
        output = self.processor.revert_tensor(tensor)
        self.calls += 1
        if self.request == 1:
            # Actual model warmup only. No reference computation or scalar GPU
            # reads in any formal request; compare float bytes before uint8.
            reference = self.original.revert_tensor(tensor)
            if (output.shape != reference.shape or output.dtype != reference.dtype
                    or not torch.equal(output.contiguous().view(torch.uint8),
                                       reference.contiguous().view(torch.uint8))):
                raise RuntimeError("Cached VAE output normalization is not byteexact")
            self.raw_checks += 1
        return output

    def finish(self):
        if self.calls != 22 or self.raw_checks != (22 if self.request == 1 else 0):
            raise RuntimeError("Cached VAE output normalization call-count audit failed")
        init_logger(__name__).info("H3_VAE_CACHED_NORMALIZE %s", json.dumps(dict(
            request=self.request, calls=self.calls, raw_float_byteexact_checks=self.raw_checks,
            dtype="float32", arithmetic="clone_sub_div_original_clamp_layout",
            per_chunk_constant_h2d=0, per_chunk_std_scalar_read=0, status="PASS"), sort_keys=True))


def prepare(owner, processor, device, *, latent=None):
    flag = os.environ.get(ENV, "0")
    if flag not in ("0", "1"):
        raise ValueError(f"{ENV} must be 0 or 1")
    if flag == "0":
        return processor
    if latent is None:raise ValueError("Output fusion needs latent geometry")
    key = (id(processor), str(device), tuple(latent.shape))
    cache = getattr(owner, "_h3_cached_output_processors", None)
    if cache is None:
        cache = {}
        owner._h3_cached_output_processors = cache
    if key not in cache:
        if os.environ.get('VLLM_OMNI_H3_OUTPUT_FUSION') != '1':
            raise RuntimeError('Output-fusion experiment flag is required')
        from .output_fusion import FusedOutputProcessor
        cache[key] = FusedOutputProcessor(processor, device, owner.model)
    wrapper = cache[key]
    if latent is None:raise ValueError('Output fusion requires the real latent geometry')
    yi,_,_=owner.model.split_tiles(int(latent.shape[-2])*16,True)
    xi,_,_=owner.model.split_tiles(int(latent.shape[-1])*16,True)
    windows=(int(latent.shape[2])+int(owner.model.token_drop))//int(owner.model.tokens_chunk_size)-1
    if (int(latent.shape[2]),windows) not in ((37,7),(42,8),(107,21)):
        raise ValueError('Optimized output requires a qualified short or 362-frame window')
    wrapper.expected_rgb_calls=windows+1
    wrapper.blend.expected_calls=windows*((len(yi)-1)*len(xi)+len(yi)*(len(xi)-1))+(windows-1)
    wrapper.begin()
    return wrapper


def finish(processor):
    if isinstance(processor, CachedOutputProcessor):
        processor.finish()
