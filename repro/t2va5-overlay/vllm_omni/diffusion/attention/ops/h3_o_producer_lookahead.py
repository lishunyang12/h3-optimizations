"""Fixed-shape O producers queued before the first blocking RDMA exchange."""
from functools import lru_cache
import json
import os

import torch
import torch.distributed as dist
from vllm.logger import init_logger

logger=init_logger(__name__)
FLAG='VLLM_OMNI_H3_O_PRODUCER_LOOKAHEAD'
_COMPLETED=0
_RAW_PASSES=0


@lru_cache(None)
def producer_stream(device):
    return torch.cuda.Stream(device=device,priority=-1)


def prepare(active,fine,coarse,tiles,state):
    from .h3_overlap_chunks import produce
    if os.environ.get(FLAG)!='1' or 'v16_fine' in active or 'o_lookahead' in active:
        raise RuntimeError('This arm requires the fixed four-chunk O lookahead route')
    assert torch.cuda.current_stream(fine.device)==active['comm']
    assert fine.dtype==coarse.dtype==torch.bfloat16
    source_rows=active.get('tiled_o_source_rows')
    if source_rows is None:
        assert tuple(fine.shape)==(1,38208,7,128)
        produce_live=produce
    else:
        from .h3_tiled_o_producer import produce as produce_tiled, validate_source
        validate_source(fine,source_rows)
        def produce_live(f,c,t,l,start,count):
            produce_tiled(f,c,t,l,start,count,source_rows)
        if _COMPLETED==0:
            logger.info('H3_TILED_O_PRODUCER rank=%d source_rows=43200 aligned_rows=38208 chunks=4 rows=1194 tail=132 raw_bit_copy=1',dist.get_rank())
    assert tuple(coarse.shape)==(1,675,7,128)
    assert dist.get_world_size()==8 and state.communicator is not None
    count=1194;shape=(1,8*(count+132),7,128)
    entries=[]
    for index in range(4):
        prototype=state.registered_shape_prototype(shape,dtype=fine.dtype,device=fine.device)
        out=state.output(prototype,op='gather_heads',slot=f'h3_o_chunk4_{index}')
        landing=state.communicator.input_buffer(out,shape)
        assert landing.is_contiguous() and out.is_contiguous()
        entries.append(dict(out=out,landing=landing,ready=None))
    # Concurrent writes must use separate registered source slots and must not
    # overlap any receive buffer consumed by gate/projection on the side stream.
    ranges=sorted((t.data_ptr(),t.data_ptr()+t.numel()*t.element_size())
        for e in entries for t in (e['landing'],e['out']))
    assert all(left[1]<=right[0] for left,right in zip(ranges,ranges[1:])), 'O registered ranges overlap'
    stream=producer_stream(fine.device)
    ticket=dict(stream=stream,entries=entries,call=_COMPLETED+1,consumed=0,raw_pass=False)
    active['o_lookahead']=ticket  # Install before submitting anything: cleanup also covers exceptions.
    stream.wait_stream(active['comm'])  # fine and the original coarse join precede this event.
    for tensor in (fine,coarse,tiles):tensor.record_stream(stream)
    if source_rows is not None:source_rows.record_stream(stream)
    with torch.cuda.stream(stream):
        for index,e in enumerate(entries):
            e['landing'].record_stream(stream)
            with torch.cuda.nvtx.range(f'h3.oproducer.lookahead.produce.{index}'):
                produce_live(fine,coarse,tiles,e['landing'],index*count,count)
            ready=torch.cuda.Event(enable_timing=False)
            ready.record(stream);e['ready']=ready
    if ticket['call']<=50:
        # Actual first-NFE input byte gates, excluded from formal timing.
        # Use the unchanged producer into independent scratch, never the live
        # transport slots. The later five requests execute no reference copies.
        reference_fine=fine
        if source_rows is not None:
            from .minimax_h3_vsa_layout import h3_vsa_tile_untile
            reference_fine=h3_vsa_tile_untile(fine.contiguous(),source_rows)
        scratch=torch.empty_like(entries[0]['landing'])
        for index,e in enumerate(entries):
            active['comm'].wait_event(e['ready'])
            produce(reference_fine,coarse,tiles,scratch,index*count,count)
            assert torch.equal(scratch.view(torch.uint8),e['landing'].view(torch.uint8)), 'O producer raw bytes differ'
        ticket['raw_pass']=True
        logger.info('H3_OPRODUCER_RAW_GATE %s',json.dumps(dict(rank=dist.get_rank(),layer=ticket['call']-1,
            chunks=4,rows=1194,tail=132,byteexact=True,status='PASS'),sort_keys=True))
    return ticket


def consume(active,index):
    ticket=active['o_lookahead']
    assert index==ticket['consumed'] and 0<=index<4
    entry=ticket['entries'][index]
    assert entry['ready'] is not None
    active['comm'].wait_event(entry['ready'])
    entry['landing'].record_stream(active['comm'])
    ticket['consumed']+=1
    return entry['out'],entry['landing']


def complete(active):
    global _COMPLETED,_RAW_PASSES
    ticket=active['o_lookahead']
    assert ticket['consumed']==4 and ticket['call']==_COMPLETED+1
    if ticket['call']<=50:
        assert ticket['raw_pass'];_RAW_PASSES+=1
    _COMPLETED+=1
    if _COMPLETED%50==0:
        assert _RAW_PASSES==50
        logger.info('H3_OPRODUCER_NFE %s',json.dumps(dict(rank=dist.get_rank(),forwards=_COMPLETED//50,
            layers_completed=_COMPLETED,producer_calls=4*_COMPLETED,ready_waits=4*_COMPLETED,
            raw_layers_passed=50,chunks=4,rows=1194,tail=132,extra_stream=True,tiled_source=('tiled_o_source_rows' in active),status='PASS'),sort_keys=True))


def cleanup(active,caller):
    ticket=active.get('o_lookahead')
    if ticket is not None:
        caller.wait_stream(ticket['stream'])
        for entry in ticket['entries']:
            entry['landing'].record_stream(caller)
