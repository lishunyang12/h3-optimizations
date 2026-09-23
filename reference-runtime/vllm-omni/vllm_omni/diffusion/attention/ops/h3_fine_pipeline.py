"""Four fine query groups release the unchanged four reverse-O chunks."""
import os
from functools import lru_cache

import torch
import torch.distributed as dist
from vllm.logger import init_logger

logger = init_logger(__name__)
_LOGGED = False


def enabled():
    value = os.environ.get('VLLM_OMNI_H3_FINE_PIPELINE', '0')
    assert value in ('0', '1'), value
    return value == '1'


@lru_cache(None)
def fine_stream(device):
    return torch.cuda.Stream(device=device)


def launch(quantized, indices, nums, sizes, scale, out, workspace):
    from vllm_omni.diffusion.attention.ops.h3_overlap_experiment import ACTIVE
    from vllm_omni.diffusion.attention.ops.h3_fine_tiles import launch_subset, tile_plan

    active = ACTIVE.get()
    assert enabled() and active is not None and 'v16_fine' not in active
    assert torch.cuda.current_stream(out.device) == active['comm']
    assert active.get('v3_coarse_done') is not None, 'Submit coarse before the fine pipeline'
    assert not torch.cuda.is_current_stream_capturing(), 'v16 qualifies the eager model path'
    tiles = tile_plan(out.device.index)
    stream = fine_stream(out.device)
    stream.wait_stream(active['comm'])
    inputs = (*quantized, indices, nums, sizes, out, workspace, *tiles)
    ticket = dict(stream=stream, events=[], refs=inputs, out=out, consumed=0)
    active['v16_fine'] = ticket
    for tensor in inputs:
        tensor.record_stream(stream)
    with torch.cuda.stream(stream):
        for i, group in enumerate(tiles):
            with torch.cuda.nvtx.range(f'h3.fine_pipeline.compute.{i}'):
                launch_subset(quantized, indices, nums, sizes, scale, out, workspace, group)
            event = torch.cuda.Event(enable_timing=False)
            event.record(stream)
            ticket['events'].append(event)
    global _LOGGED
    if not _LOGGED:
        rank = dist.get_rank() if dist.is_initialized() else 0
        logger.info('H3_FINE_PIPELINE rank=%d groups=400,415,406,427 chunks=4 row_math=unchanged coarse=early graph=0', rank)
        _LOGGED = True
    return out.transpose(1, 2)


def bind_output(active, fine, source_rows):
    from vllm_omni.diffusion.attention.ops.h3_fine_producer import validate_tiled_source

    validate_tiled_source(fine, source_rows)
    ticket = active['v16_fine']
    assert fine.data_ptr() == ticket['out'].data_ptr()
    assert len(ticket['events']) == 4 and ticket['consumed'] == 0
    assert 'fine' not in ticket
    ticket.update(fine=fine, source_rows=source_rows)


def wait_chunk(active, fine, index):
    ticket = active['v16_fine']
    assert fine is ticket['fine'] and index == ticket['consumed']
    assert torch.cuda.current_stream(fine.device) == active['comm']
    with torch.cuda.nvtx.range(f'h3.fine_pipeline.ready.{index}'):
        active['comm'].wait_event(ticket['events'][index])
    ticket['consumed'] += 1
    return ticket['source_rows']


def cleanup(active, caller):
    ticket = active.get('v16_fine')
    if ticket is not None:
        # Includes exceptions before publish/reverse: no input or output may
        # return to the allocator while the independent stream still uses it.
        caller.wait_stream(ticket['stream'])

