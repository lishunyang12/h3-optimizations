"""Explicit, audited four-step precision policies for the resident TP1/SP8 run.

All policy reads happen in eager code or opaque custom operators, never in
traced Python branches. BF16 weights are the original loaded/student-fused
tensors retained before quantization, with no dequantization. The first-step
arm stages its last 26 BF16 blocks from pinned memory with two GPU buffers.
"""
import hashlib
import json
import os
from pathlib import Path

import torch

ARM = os.environ.get("VLLM_OMNI_H3_PRECISION_SCHEDULE")
if ARM not in ("all_mxfp8", "first_step_bf16", "o_only_bf16"):
    raise RuntimeError("Select an explicit admitted precision schedule")
_ATTENTION = {"bf16": 0, "pr4951": 0}
_NFE = 0
RESIDENT_BLOCKS = 24
_HOST = {}
_BUFFERS = []
_TRANSFER = None
_STAGED = [None, None]
_READY = [None, None]
_COPIES = 0


def step():
    from vllm_omni.diffusion.forward_context import get_forward_context
    context = get_forward_context()
    if context.total_denoise_steps != 4 or context.denoise_step_idx not in range(4):
        raise RuntimeError("Precision schedule requires explicit four-step denoise progress")
    return context.denoise_step_idx


def sage_enabled():
    return not (ARM == "first_step_bf16" and step() == 0)


def attention_mode():
    if os.environ.get("VLLM_OMNI_H3_VSA_SAGE") != "pr4951":
        raise RuntimeError("Scheduled attention requires the admitted Sage provider")
    mode = "pr4951" if sage_enabled() else "bf16"
    _ATTENTION[mode] += 1
    return mode


def use_bf16(record):
    return ((ARM == "first_step_bf16" and step() == 0)
            or (ARM == "o_only_bf16" and record["name"].endswith("attn.out_proj")))


def preserve(layer, name):
    if ARM == "first_step_bf16" or (ARM == "o_only_bf16" and name.endswith("attn.out_proj")):
        weight = layer.weight
        if weight.dtype != torch.bfloat16 or not weight.is_cuda or not weight.is_contiguous():
            raise RuntimeError("Original BF16 source weight contract changed")
        # Keep the original tensor, without cloning it or registering a duplicate
        # parameter that could affect student fusion or online quantization.
        index = int(name.split(".")[1])
        if ARM == "first_step_bf16" and index >= RESIDENT_BLOCKS:
            saved = torch.empty_like(weight, device="cpu", pin_memory=True)
            saved.copy_(weight)
            _HOST.setdefault(index, {})[name.split(".", 2)[2]] = saved
        else:
            saved = weight.detach()
        layer._h3_original_bf16 = saved
        layer._h3_original_bf16_ptr = saved.data_ptr()


def bf16_weight(record):
    layer = record["layer"]
    weight = layer._h3_original_bf16
    if (weight.dtype != torch.bfloat16
            or weight.data_ptr() != layer._h3_original_bf16_ptr
            or tuple(weight.shape) != record["weight_shape"]):
        raise RuntimeError("Original BF16 weight was replaced or offloaded")
    if weight.device.type == "cpu":
        index = int(record["name"].split(".")[1])
        slot = (index-RESIDENT_BLOCKS) % 2
        if not weight.is_pinned() or _STAGED[slot] != index or _READY[slot] is None:
            raise RuntimeError("BF16 prefetch buffer is not ready for this layer")
        torch.cuda.current_stream(record["device"]).wait_event(_READY[slot])
        weight = _BUFFERS[slot][record["name"].split(".", 2)[2]]
    if weight.device != record["device"]:
        raise RuntimeError("BF16 compute weight is not resident on its original GPU")
    return weight


def prefetch(index, consumed=None):
    global _COPIES
    if index not in _HOST:
        return
    slot = (index-RESIDENT_BLOCKS) % 2
    if consumed is not None:
        _TRANSFER.wait_event(consumed)
    with torch.cuda.stream(_TRANSFER):
        for suffix, source in _HOST[index].items():
            _BUFFERS[slot][suffix].copy_(source, non_blocking=True)
            _COPIES += 1
        ready = torch.cuda.Event()
        ready.record(_TRANSFER)
    _STAGED[slot], _READY[slot] = index, ready


def finish_block(record):
    if ARM != "first_step_bf16" or step() != 0:
        return
    index = int(record["name"].split(".")[1])
    if index in _HOST:
        # FC2 is after the attention stream join and the last BF16 weight
        # consumer in this block. Reuse the slot only after all those reads.
        consumed = torch.cuda.Event()
        consumed.record(torch.cuda.current_stream(record["device"]))
        prefetch(index+2, consumed)


def count(record):
    record["bf16_calls" if use_bf16(record) else "mxfp8_calls"] += 1


def source_hashes():
    root = Path(__file__).resolve().parents[2]
    names = ("models/minimax_h3/precision_schedule.py",
             "models/minimax_h3/mxfp8_experiment.py",
             "models/minimax_h3/mxfp8_kernels.py",
             "models/minimax_h3/silu_mxfp8_fusion.py",
             "models/minimax_h3/minimax_h3_transformer.py",
             "attention/ops/h3_vsplit_experiment.py",
             "attention/ops/h3_schedule_v3.py",
             "attention/backends/fastvideo_vsa.py")
    return {str(root / name): hashlib.sha256((root / name).read_bytes()).hexdigest()
            for name in names}


def install_audit(model, state):
    global _TRANSFER
    entries = state["entries"]
    rank = torch.distributed.get_rank()
    if len(entries) != 200 or not state["fusion"] or state["mode"] != "mxfp8":
        raise RuntimeError("Unexpected base projection inventory")
    bf16_entries = [e for e in entries if hasattr(e["layer"], "_h3_original_bf16")]
    if len(bf16_entries) != (200 if ARM == "first_step_bf16" else 50 if ARM == "o_only_bf16" else 0):
        raise RuntimeError("BF16 retention inventory differs")
    if _HOST:
        _TRANSFER = torch.cuda.Stream()
        for slot in range(2):
            _BUFFERS.append({suffix: torch.empty_like(value, device=entries[0]["device"])
                             for suffix, value in _HOST[RESIDENT_BLOCKS].items()})
    inventory = [dict(name=e["name"], shape=e["weight_shape"],
                      packed_dtype=str(e["layer"].weight.dtype),
                      scale_dtype=str(e["scale"].dtype),
                      method=type(e["layer"].quant_method).__name__,
                      original_bf16_retained=e in bf16_entries)
                 for e in entries]
    # Allocator caches from model loading are not needed by the steady-state
    # run. This is outside every timed request and does not offload weights.
    torch.cuda.empty_cache()
    print("H3_PRECISION_SCHEDULE_READY " + json.dumps(dict(
        status="PASS", rank=rank, arm=ARM, inventory=inventory,
        modules=200, bf16_modules=len(bf16_entries),
        bf16_weight_bytes=sum(e["layer"]._h3_original_bf16.numel()*2 for e in bf16_entries),
        bf16_source="original_student_fused_before_quantization",
        residency="cuda+pinned_cpu_double_buffer" if _HOST else "cuda",
        host_bf16_modules=sum(len(block) for block in _HOST.values()),
        host_bf16_bytes=sum(w.numel()*2 for block in _HOST.values() for w in block.values()),
        resident_bf16_blocks=RESIDENT_BLOCKS if _HOST else None,
        source_sha256=source_hashes(), fallback_count=0,
        memory_allocated=torch.cuda.memory_allocated(),
        memory_reserved=torch.cuda.memory_reserved())), flush=True)

    previous = {id(e): (0, 0, 0) for e in entries}
    previous_attention = _ATTENTION.copy()

    def before(module, args):
        if step() != state["forwards"] % 4:
            raise RuntimeError("Pipeline step and real model forward sequence disagree")
        if ARM == "first_step_bf16" and step() == 0:
            prefetch(RESIDENT_BLOCKS)
            prefetch(RESIDENT_BLOCKS+1)

    def completed(module, args, output):
        state["forwards"] += 1
        nfe = state["forwards"]
        current_step = step()
        counts = {"bf16": 0, "mxfp8": 0}
        fused_calls = 0
        for e in entries:
            expected = e["calls_per_nfe"]
            old_bf, old_mx, old_fused = previous[id(e)]
            bf, mx = e["bf16_calls"]-old_bf, e["mxfp8_calls"]-old_mx
            if (bf, mx) != ((expected, 0) if use_bf16(e) else (0, expected)):
                raise RuntimeError("Actual projection precision calls differ: " + e["name"])
            if e["calls"] != nfe*expected:
                raise RuntimeError("Logical projection count differs")
            fused = e["fused_calls"]-old_fused
            if fused != int(e["fused_required"] and not use_bf16(e)):
                raise RuntimeError("FC2 fusion executed at the wrong precision")
            counts["bf16"] += bf
            counts["mxfp8"] += mx
            fused_calls += fused
            previous[id(e)] = (e["bf16_calls"], e["mxfp8_calls"], e["fused_calls"])
            if e["name"].endswith("attn.qkv_proj"):
                if e.get("split_qk_calls") != nfe or e.get("split_v_calls") != nfe:
                    raise RuntimeError("QK/V split count differs")
                if ("bf16" if use_bf16(e) else "mxfp8") not in e["split_raw_precisions"]:
                    raise RuntimeError("QK/V split has no byte-exact raw gate")
        attention = {key: _ATTENTION[key]-previous_attention[key] for key in _ATTENTION}
        expected_attention = {"bf16": 0, "pr4951": 50} if sage_enabled() else {"bf16": 50, "pr4951": 0}
        if attention != expected_attention:
            raise RuntimeError("Actual attention kernel precision differs")
        previous_attention.update(_ATTENTION)
        print("H3_PRECISION_SCHEDULE_NFE " + json.dumps(dict(
            status="PASS", rank=rank, arm=ARM, forwards=nfe,
            request=(nfe-1)//4, step=current_step+1, projection_calls=counts,
            attention_calls=attention, fused_mxfp8_fc2_calls=fused_calls,
            split_qk_calls=50, split_v_calls=50, modules=200,
            cumulative_bf16_h2d_copies=_COPIES,
            fallback_count=0, wire="bfloat16", bf16_gate=True,
            memory_allocated=torch.cuda.memory_allocated(),
            memory_reserved=torch.cuda.memory_reserved())), flush=True)

    model.register_forward_pre_hook(before)
    model.register_forward_hook(completed)
