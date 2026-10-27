"""Warmup-qualified integration of the independently qualified output kernels."""
import json

import torch
from vllm.logger import init_logger

from .cached_output_normalize import CachedOutputProcessor
from .output_fusion_kernels import BlendCandidate, RGBCandidate


def same_bytes(left, right):
    return (left.shape == right.shape and left.dtype == right.dtype
            and torch.equal(left.contiguous().view(torch.uint8),
                            right.contiguous().view(torch.uint8)))


class AuditedBlend:
    def __init__(self, original):
        self.original = original
        self.candidate = BlendCandidate(original)
        self.request = 0

    def begin(self):
        self.request += 1
        self.calls = 0
        self.checks = 0
        self.start_calls = self.candidate.calls
        self.start_fallbacks = self.candidate.fallbacks

    def __call__(self, a, b, blend_extent, dim):
        result = self.candidate(a, b, blend_extent, dim)
        self.calls += 1
        if self.request == 1:
            reference = self.original(a, b, blend_extent, dim)
            if not same_bytes(result, reference):
                raise RuntimeError("Actual VAE blend FP32 bytes changed")
            self.checks += 1
        return result

    def finish(self):
        # 21 windows * (3*7 vertical + 4*6 horizontal) + 20 temporal blends.
        if (self.calls != 965 or self.candidate.calls - self.start_calls != 965
                or self.candidate.fallbacks != self.start_fallbacks
                or self.checks != (965 if self.request == 1 else 0)):
            raise RuntimeError("Actual VAE fused blend call-count or qualification failed")
        init_logger(__name__).info("H3_VAE_FUSED_BLEND %s", json.dumps(dict(
            request=self.request, calls=self.calls, raw_float_byteexact_checks=self.checks,
            fallback=0, status="PASS"), sort_keys=True))


class FusedOutputProcessor(CachedOutputProcessor):
    def __init__(self, processor, device, model):
        super().__init__(processor, device)
        self.rgb = RGBCandidate(self.mean_values, self.std_values, device)
        self.blend = AuditedBlend(model.blend)
        model.blend = self.blend

    def begin(self):
        super().begin()
        self.blend.begin()

    def revert_tensor(self, tensor):
        raise RuntimeError("Fused RGB processor must consume the complete output boundary")

    def rgb_u8(self, tensor, *, height, width):
        from .chunked_cpu_output import exact_rgb_u8_bthwc
        output = self.rgb(tensor, height=height, width=width)
        self.calls += 1
        if self.request == 1:
            reference = exact_rgb_u8_bthwc(self.original.revert_tensor(tensor),
                                           height=height, width=width)
            if not same_bytes(output, reference):
                raise RuntimeError("Actual VAE fused RGB bytes changed")
            self.raw_checks += 1
        return output

    def finish(self):
        if self.calls != 22 or self.raw_checks != (22 if self.request == 1 else 0):
            raise RuntimeError("Actual VAE fused RGB call-count or qualification failed")
        self.blend.finish()
        # This validates final RGB bytes, not the removed intermediate FP32
        # output. Do not emit the old cached-normalization FP32 receipt.
        init_logger(__name__).info("H3_VAE_FUSED_RGB %s", json.dumps(dict(
            request=self.request, calls=self.calls, raw_rgb_byteexact_checks=self.raw_checks,
            per_chunk_constant_h2d=0, per_chunk_std_scalar_read=0,
            output_dtype="uint8", status="PASS"), sort_keys=True))
