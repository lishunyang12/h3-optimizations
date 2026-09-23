"""Default-off direct chunk consumption for the fixed H3 overlap experiment."""
import os
import json
from functools import lru_cache

import torch
import torch.distributed as dist
from vllm.logger import init_logger

from vllm_omni.diffusion.attention.ops.h3_chunk_norm import norm_chunk_into
from vllm_omni.diffusion.attention.ops.minimax_h3_modulation import _indexed_gate_rms_norm_scale_shift_kernel as _norm_kernel

logger = init_logger(__name__)
_LOGGED = set()
_COMPLETED = {}


def post_attention_mode():
    mode = os.environ.get("VLLM_OMNI_H3_POST_ATTN_NORM", "off")
    if mode not in ("off", "side", "consumer"):
        raise ValueError(f"Unsupported post-attention norm mode: {mode!r}")
    return mode


@lru_cache(None)
def consumer_stream(device):
    return torch.cuda.Stream(device=device)


def _contract(tensor):
    return (tensor.data_ptr(), tuple(tensor.shape), tuple(tensor.stride()), tensor.dtype, tensor.device)


class PostAttentionTicket:
    """Own four projected inputs and two final outputs until stream joins.

    The ticket is created and consumed inside the existing eager attention
    island. Only the two normal tensors escape back into the compiled block.
    """
    def __init__(self, residual, gate, weight, shift, scale, indices, eps, *, mode=None):
        self.mode = post_attention_mode() if mode is None else mode
        if self.mode not in ("side", "consumer"):
            raise ValueError("A post-attention ticket requires side or consumer mode")
        if not (residual.is_cuda and tuple(residual.shape) == (11984, 5376)
                and residual.dtype == torch.bfloat16 and residual.is_contiguous()):
            raise ValueError("Post-attention residual must be CUDA [11984,5376] contiguous BF16")
        self.inputs = (residual, gate, weight, shift, scale, indices)
        if torch.is_grad_enabled() or not all(t.device == residual.device for t in self.inputs):
            raise ValueError("Post-attention inputs must be inference-only on one device")
        if not (weight.shape == (5376,) and weight.is_contiguous()
                and all(t.ndim == 2 and t.shape == gate.shape and t.shape[1] == 5376 and t.stride(1) == 1
                        for t in (gate, shift, scale))
                and all(t.dtype in (torch.bfloat16, torch.float32) for t in (weight, gate, shift, scale))
                and indices.shape == (11984,) and indices.dtype in (torch.int32, torch.int64)):
            raise ValueError("Post-attention parameter/index contract mismatch")
        self.contracts = tuple(_contract(t) for t in self.inputs)
        self.eps = eps
        self.residual_out = torch.empty_like(residual)
        self.modulated_out = torch.empty_like(residual)
        self.next_chunk = 0
        self.side = None
        self.consumer = None
        self.projected = []
        self.ready_events = []
        self.done = None
        self.used = False
        self.audit = os.environ.get("H3_CHUNKNORM_LIVE_AUDIT", "1") == "1"
        self.ordinal = _COMPLETED.get(self.mode, 0)
        fast = os.environ.get("VLLM_OMNI_H3_NORM_FAST_SUBMIT", "0")
        if fast not in ("0", "1"):
            raise ValueError("NORM_FAST_SUBMIT must be 0 or 1")
        self.fast_submit = fast == "1"
        if self.fast_submit:
            # Validate stable inputs/outputs once, before O communication starts.
            # Per-submit identity/layout checks below still reject replacement.
            self.output_contracts = (_contract(self.residual_out), _contract(self.modulated_out))
            self.output_storages = (self.residual_out.untyped_storage().data_ptr(),
                                    self.modulated_out.untyped_storage().data_ptr())
            input_storages = {t.untyped_storage().data_ptr() for t in self.inputs}
            if self.output_storages[0] == self.output_storages[1] or any(p in input_storages for p in self.output_storages):
                raise ValueError("Prepared outputs must not alias inputs or one another")
            self.slices = tuple((self.residual_out[i*2996:(i+1)*2996],
                                 self.modulated_out[i*2996:(i+1)*2996],
                                 residual[i*2996:(i+1)*2996], indices[i*2996:(i+1)*2996]) for i in range(4))
            self.strides = (residual.stride(0), gate.stride(0), shift.stride(0),
                            scale.stride(0), indices.stride(0))

    def bind(self, side, caller):
        if self.side is not None:
            raise RuntimeError("Post-attention ticket cannot be reused")
        self.side = side
        self.consumer = side if self.mode == "side" else consumer_stream(self.residual_out.device)
        # Original residual/AdaLN inputs and allocations precede this event.
        self.consumer.wait_stream(caller)
        for tensor in (*self.inputs, self.residual_out, self.modulated_out):
            tensor.record_stream(self.consumer)
        rank = dist.get_rank() if dist.is_initialized() else 0
        key = (rank, self.mode)
        if key not in _LOGGED:
            logger.info("H3_POST_ATTN_NORM rank=%d mode=%s chunks=4 rows=2996 assembly=0", rank, self.mode)
            _LOGGED.add(key)

    def submit(self, index, projected):
        if self.consumer is None or self.done is not None or index != self.next_chunk:
            raise RuntimeError("Post-attention chunks must be submitted exactly once in order")
        if torch.cuda.current_stream(projected.device) != self.side:
            raise RuntimeError("Projection must publish on the original O side stream")
        if tuple(_contract(t) for t in self.inputs) != self.contracts:
            raise RuntimeError("Post-attention input identity/layout changed")
        if projected.shape != (2996, 5376) or projected.device != self.residual_out.device:
            raise ValueError("Post-attention projection shape/device changed")
        if self.fast_submit:
            self._submit_prepared(index, projected)
            self.next_chunk += 1
            return
        ready = torch.cuda.Event()
        ready.record(self.side)
        self.ready_events.append(ready)
        self.projected.append(projected)
        self.consumer.wait_event(ready)
        residual, gate, weight, shift, scale, indices = self.inputs
        rows = slice(index * 2996, (index + 1) * 2996)
        with torch.cuda.stream(self.consumer), torch.cuda.nvtx.range(f"h3.output.norm_chunk.{index}"):
            projected.record_stream(self.consumer)
            norm_chunk_into(self.residual_out[rows], self.modulated_out[rows], residual[rows],
                            gate, projected, weight, shift, scale, indices[rows], self.eps)
        self.next_chunk += 1

    def _submit_prepared(self, index, projected):
        if torch.is_grad_enabled():
            raise ValueError("Prepared norm is inference-only")
        if (_contract(self.residual_out), _contract(self.modulated_out)) != self.output_contracts:
            raise RuntimeError("Prepared norm output identity/layout changed")
        if projected.dtype != torch.bfloat16 or not projected.is_contiguous():
            raise ValueError("Prepared projection must be contiguous BF16")
        if projected.untyped_storage().data_ptr() in self.output_storages:
            raise ValueError("Prepared projection must not alias norm outputs")
        self.projected.append(projected)
        if self.consumer == self.side:
            # submit already verifies that side is current. Stream ordering is
            # sufficient; avoid an event waited by the stream that recorded it.
            self._launch_prepared(index, projected)
        else:
            ready = torch.cuda.Event()
            ready.record(self.side)
            self.ready_events.append(ready)
            self.consumer.wait_event(ready)
            with torch.cuda.stream(self.consumer):
                self._launch_prepared(index, projected)

    def _launch_prepared(self, index, projected):
        # Only _submit_prepared may call this, after the original submit guards
        # and prepared-output/projection validation. Numerical kernel unchanged.
        projected.record_stream(self.consumer)
        residual_out, modulated_out, residual, indices = self.slices[index]
        _, gate, weight, shift, scale, _ = self.inputs
        sr, sg, ss, sc, si = self.strides
        with torch.cuda.nvtx.range(f"h3.output.norm_chunk.{index}"):
            _norm_kernel[(2996,)](residual_out, modulated_out, residual, gate, projected,
                weight, shift, scale, indices, 5376, self.eps, sr, sg, projected.stride(0),
                ss, sc, si, block_n=8192, num_warps=8)

    def finish(self, comm):
        if self.next_chunk != 4 or self.done is not None:
            raise RuntimeError("All four norm chunks must finish submission exactly once")
        self.done = torch.cuda.Event()
        self.done.record(self.consumer)
        comm.wait_event(self.done)
        return self.modulated_out.unsqueeze(0)

    def take(self, observed, caller):
        if self.done is None or self.used or _contract(observed) != _contract(self.modulated_out):
            raise RuntimeError("Missing, reused or substituted post-attention result")
        caller.wait_event(self.done)
        self.residual_out.record_stream(caller)
        self.modulated_out.record_stream(caller)
        if self.audit and self.ordinal < 50:
            from .minimax_h3_modulation import indexed_gate_rms_norm_scale_shift
            residual, gate, weight, shift, scale, indices = self.inputs
            branch = torch.cat(self.projected, dim=0)
            reference = indexed_gate_rms_norm_scale_shift(residual, gate, branch, weight, shift, scale, indices, self.eps)
            checks = [torch.equal(a.view(torch.uint8), b.view(torch.uint8))
                      for a, b in zip(reference, (self.residual_out, self.modulated_out))]
            assert all(checks), ('chunk norm byte mismatch', self.ordinal, checks)
            logger.info('H3_CHUNKNORM_RAW_GATE %s', json.dumps(dict(rank=dist.get_rank(),
                layer=self.ordinal, mode=self.mode, residual_byteexact=checks[0], norm_byteexact=checks[1], status='PASS')))
        if self.audit:
            assert _COMPLETED.get(self.mode, 0) == self.ordinal
            _COMPLETED[self.mode] = self.ordinal + 1
            if (self.ordinal + 1) % 50 == 0:
                logger.info('H3_CHUNKNORM_NFE %s', json.dumps(dict(rank=dist.get_rank(),
                    mode=self.mode, forwards=(self.ordinal+1)//50, layers=self.ordinal+1,
                    chunks=4*(self.ordinal+1), raw_layers_passed=50, assembly=0,
                    fast_submit=self.fast_submit, ready_events_per_layer=len(self.ready_events),
                    prepared_chunks=4 if self.fast_submit else 0, status='PASS' )))
        self.used = True
        return self.residual_out, self.modulated_out

    def join_on_error(self, caller):
        if self.fast_submit and self.used:
            # take() has already joined done on caller. Failed/partial calls
            # still use the full consumer join below.
            return
        if self.consumer is not None:
            caller.wait_stream(self.consumer)
