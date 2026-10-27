"""Opt-in, geometry-bound real-model integration of the validated RDMA schedule."""
from contextvars import ContextVar
from functools import lru_cache
import torch
import os
import torch.distributed as dist
from vllm.logger import init_logger

logger = init_logger(__name__)
_BUNDLE_LOGGED = False
_GATE_LOGGED = False

ACTIVE = ContextVar('h3_overlap_experiment', default=None)


@lru_cache(None)
def streams(device):
    return torch.cuda.Stream(device=device,priority=-1), torch.cuda.Stream(device=device)


def run(attention, query, key, value, metadata, projector, *, gate_input=None, gate_projector=None,
        vsplit_input=None, vsplit_projection=None, vsplit_layer_index=None):
    assert ACTIVE.get() is None, 'overlap forward cannot be nested'
    impl = attention.attention
    assert impl.h3_kernel_backend == 'flashinfer' and impl.topk == 66
    assert query.shape == key.shape == value.shape == (1,4776,56,128)
    assert query.dtype == key.dtype == value.dtype == torch.bfloat16
    caller = torch.cuda.current_stream()
    comm,side = streams(query.device)
    comm.wait_stream(caller)
    state = dict(comm=comm,side=side,projector=projector,impl=impl,prepared=None,reverse=None,projected=False)
    token = ACTIVE.set(state)
    try:
        with torch.cuda.stream(comm), torch.cuda.nvtx.range('h3.overlap.attention'):
            if vsplit_input is not None:
                from vllm_omni.diffusion.attention.ops.h3_vsplit_experiment import begin

                begin(state, attention, vsplit_projection, vsplit_input, value,
                      vsplit_layer_index, metadata)
            if gate_input is not None:
                assert gate_input.shape == (4776,5376) and gate_input.dtype == torch.bfloat16
                assert not gate_input.requires_grad and gate_input.is_contiguous()
                assert gate_projector is not None and 'gate_compress' not in metadata.extra
                assert 'vsa_h3_prefix_segments' in metadata.extra
                # Q/K norm+RoPE have already been submitted to caller. Gate
                # uses exactly the original GEMM on the same side stream as
                # QK preparation and reverse gate/O. It has no dependency on
                # received Q/K; preprocessing only inspects its shape/dtype.
                side.wait_stream(comm)
                ticket = state.get('vsplit_ticket')
                if ticket is not None:
                    assert ticket.launched and ticket.done is not None
                    with torch.cuda.nvtx.range('h3.vsplit.gate_after_v_ready'):
                        side.wait_event(ticket.done)
                with torch.cuda.stream(side), torch.cuda.nvtx.range('h3.lossless.gate_gemm'):
                    gate_input.record_stream(side)
                    gate_result = gate_projector(gate_input)
                    gate = gate_result[0] if isinstance(gate_result,tuple) else gate_result
                    assert gate.shape == (4776,7168) and gate.dtype == torch.bfloat16
                    metadata.extra['gate_compress'] = gate.view(1,4776,56,128)
                global _GATE_LOGGED
                if not _GATE_LOGGED:
                    logger.info('H3_LOSSLESS_GATE rank=%d M=4776 N=7168 K=5376 dtype=bf16 shape_unchanged=1 stream=overlap_side',dist.get_rank())
                    _GATE_LOGGED = True
            out = attention(query,key,value,metadata)
            assert state['projected'], 'chunked reverse/projector did not run'
            if vsplit_input is not None:
                from vllm_omni.diffusion.attention.ops.h3_vsplit_experiment import finish

                finish(state)
        caller.wait_stream(comm)
        out.record_stream(caller)
        return out
    finally:
        # Also cover exceptions before the normal preparation/reverse joins.
        if vsplit_input is not None:
            from vllm_omni.diffusion.attention.ops.h3_vsplit_experiment import cleanup

            cleanup(state, caller)
        from vllm_omni.diffusion.attention.ops.h3_fine_pipeline import cleanup as fine_cleanup

        fine_cleanup(state, caller)
        from .h3_o_producer_lookahead import cleanup as o_cleanup
        o_cleanup(state, caller)
        caller.wait_stream(side)
        caller.wait_stream(comm)
        ACTIVE.reset(token)


def before_v(query,key,value,metadata,group):
    active = ACTIVE.get()
    if active is None:
        return value
    if active.get('vsplit_ticket') is not None:
        from vllm_omni.diffusion.attention.ops.h3_vsplit_experiment import before_v as split_before_v

        return split_before_v(active, query, key, value, metadata, group)
    from vllm_omni.diffusion.distributed.flashinfer_ulysses import _state_for
    from vllm_omni.diffusion.attention.backends.fastvideo_vsa import (
        _get_h3_layout, _get_h3_tile_metadata, _get_h3_tiled_source_rows,
        _pool_h3_tiles, _build_h3_ordered_q2k_indices, h3_vsa_tile_pack,
    )
    state = _state_for(value,group,8)
    assert state.communicator is not None
    out = state.output(value,op='scatter_heads',slot='v')
    landing = state.communicator.input_buffer(out,value.shape)
    landing.copy_(value)
    layout = _get_h3_layout(metadata)
    assert layout == ((457, 414), (37, 24, 42), 871), layout
    prefix,shape,_ = layout
    active['side'].wait_stream(active['comm'])
    with torch.cuda.stream(active['side']), torch.cuda.nvtx.range('h3.overlap.qk_prepare'):
        from vllm_omni.diffusion.attention.ops.h3_schedule_v3 import prepared_q

        _,sizes,_,_,prefix_blocks,video_blocks = _get_h3_tile_metadata(prefix,shape,query.device)
        maps = _get_h3_tiled_source_rows(prefix,shape,43200,query.device)
        q_ready = prepared_q()
        q = h3_vsa_tile_pack(query[:,:38167],maps) if q_ready is None else q_ready[0]
        k = h3_vsa_tile_pack(key[:,:38167],maps)
        qp = _pool_h3_tiles(q,sizes) if q_ready is None else q_ready[1]
        kp = _pool_h3_tiles(k,sizes)
        scores = torch.matmul(qp,kp.transpose(-2,-1))*active['impl'].softmax_scale
        idx,num = _build_h3_ordered_q2k_indices(scores,prefix_blocks,video_blocks,66)
        active['prepared'] = (q,k,qp,kp,scores,idx,num)
    return landing


def prepared():
    active = ACTIVE.get()
    if active is None:
        return None
    result = active['prepared']
    assert result is not None, 'forward QK preparation hook did not run'
    active['comm'].wait_stream(active['side'])
    for tensor in result:
        tensor.record_stream(active['comm'])
    return result


def publish_reverse(fine,coarse,plan, *, source_rows=None):
    active = ACTIVE.get()
    if active is None:
        return False
    from vllm_omni.diffusion.attention.ops.h3_schedule_v3 import join_coarse

    join_coarse(coarse)
    assert active['reverse'] is None
    if source_rows is not None:
        from vllm_omni.diffusion.attention.ops.h3_fine_pipeline import bind_output

        bind_output(active, fine, source_rows)
    else:
        assert "v16_fine" not in active
    active['reverse'] = (fine,coarse,plan)
    return True


def finish_reverse(attn_output,ctx):
    if os.environ.get("VLLM_OMNI_H3_O_PRODUCER_LOOKAHEAD") == "1":
        return _finish_reverse_lookahead(attn_output,ctx)
    active = ACTIVE.get()
    if active is None:
        return None
    from vllm_omni.diffusion.distributed.flashinfer_ulysses import _state_for
    from vllm_omni.diffusion.attention.ops.h3_overlap_chunks import produce,gate_chunk
    from vllm_omni.diffusion.attention.ops.minimax_h3_vsa_owner_route import h3_vsa_owner_route_plan_is_trusted
    fine,coarse,plan = active['reverse']
    assert fine is attn_output
    assert set(ctx.o_bundle_state) == {'plan'} and ctx.o_bundle_state['plan'] is plan
    assert ctx.strict_a2a_backend == 'flashinfer-pcie' and ctx.joint_len == 0 and not ctx.use_uaa
    assert h3_vsa_owner_route_plan_is_trusted(plan)
    assert plan.local_rows == 4776 and plan.kmax == 132 and plan.sp_world_size == 8
    gate = ctx.o_bundle_gate_local
    assert gate is not None and gate.shape == (1,4776,56,128)
    rank = dist.get_rank(ctx.ulysses_pg)
    state = _state_for(fine,ctx.ulysses_pg,8)
    comm = state.communicator
    assert comm is not None
    # Cache these maps on the trusted, geometry-bound route's CPU tensors.
    tiles,row_map = route_maps(plan,rank,fine.device)
    row_ids = None
    if 'v16_fine' in active:
        from vllm_omni.diffusion.attention.ops.h3_fine_row_order import row_order

        row_ids = row_order(fine.device)[rank]
        # Gate was produced on this same side stream. Permuting complete rows
        # changes no arithmetic and leaves the original gate kernel intact.
        with torch.cuda.stream(active['side']), torch.cuda.nvtx.range('h3.fine_pipeline.gate_rows'):
            row_ids.record_stream(active['side'])
            gate = gate.index_select(1, row_ids)
    count = 1194
    result = torch.empty(1,4776,5376,device=fine.device,dtype=torch.bfloat16)
    for index in range(4):
        start = index*count
        shape = (1,8*(count+132),7,128)
        prototype = state.registered_shape_prototype(shape,dtype=fine.dtype,device=fine.device)
        out = state.output(prototype,op='gather_heads',slot=f'h3_o_chunk4_{index}')
        landing = comm.input_buffer(out,shape)
        with torch.cuda.nvtx.range(f'h3.overlap.o_produce.{index}'):
            if 'v16_fine' in active:
                from vllm_omni.diffusion.attention.ops.h3_fine_pipeline import wait_chunk
                from vllm_omni.diffusion.attention.ops.h3_fine_producer import produce_tiled

                source_rows = wait_chunk(active, fine, index)
                produce_tiled(fine,coarse,tiles,landing,start,count,source_rows)
            else:
                produce(fine,coarse,tiles,landing,start,count)
        with torch.cuda.nvtx.range(f'h3.overlap.o_rdma.{index}'):
            comm.gather_heads(landing,out=out)
        active['side'].wait_stream(active['comm'])
        with torch.cuda.stream(active['side']), torch.cuda.nvtx.range(f'h3.overlap.o_gate_project.{index}'):
            view = gate_chunk(out,gate[:,start:start+count],row_map[start:start+count],count)
            projected = active['projector'](view)
            if row_ids is None:
                result[:,start:start+count].copy_(projected)
            else:
                # CPU validation proves row IDs are disjoint and cover every
                # original row exactly once, so this write has no reduction.
                assert projected.shape == (count, 5376)
                result[0].index_copy_(0, row_ids[start:start+count], projected)
    active['comm'].wait_stream(active['side'])
    result.record_stream(active['side'])
    gate.record_stream(active['side'])
    if 'v16_fine' in active:
        assert active['v16_fine']['consumed'] == 4
    active['projected'] = True
    # Separate markers bind the validator to the chunked protocol and its
    # actual volume; they do not impersonate the original whole-bundle path.
    if not state.o_producer_direct_logged:
        logger.info('H3_OVERLAP_EXCHANGE rank=%d group=%s chunks=4 rows=1194 tail=132 remote_bytes=164326400 direct=1 completed=1',
                    rank,ctx.ulysses_pg.group_name)
        state.o_producer_direct_logged = True
    state.o_producer_direct_expected = None
    global _BUNDLE_LOGGED
    if not _BUNDLE_LOGGED:
        logger.info('H3_OVERLAP_BUNDLE rank=%d fine_rows=%d local_rows=4776 coarse_rows=675 chunks=4 tail=132 gate=bf16 projector=active',rank,fine.shape[1])
        _BUNDLE_LOGGED = True
    return result


_ROUTE_MAPS = {}


def route_maps(plan,rank,device):
    from vllm_omni.diffusion.attention.ops.h3_fine_pipeline import enabled

    permuted = enabled()
    key = (id(plan),rank,device,permuted)
    if key not in _ROUTE_MAPS:
        rows = plan.local_row_to_tail[rank]
        if permuted:
            from vllm_omni.diffusion.attention.ops.h3_fine_row_order import plan as row_plan

            rows = rows.index_select(0, row_plan()[0][rank])
        _ROUTE_MAPS[key] = (plan,plan.owner_tile_ids.to(device),rows.to(device))
    saved,tiles,row_map = _ROUTE_MAPS[key]
    assert saved is plan
    return tiles,row_map


def _finish_reverse_lookahead(attn_output,ctx):
    active = ACTIVE.get()
    if active is None:
        return None
    from vllm_omni.diffusion.distributed.flashinfer_ulysses import _state_for
    from vllm_omni.diffusion.attention.ops.h3_overlap_chunks import produce,gate_chunk
    from vllm_omni.diffusion.attention.ops.minimax_h3_vsa_owner_route import h3_vsa_owner_route_plan_is_trusted
    fine,coarse,plan = active['reverse']
    assert fine is attn_output
    assert set(ctx.o_bundle_state) == {'plan'} and ctx.o_bundle_state['plan'] is plan
    assert ctx.strict_a2a_backend == 'flashinfer-pcie' and ctx.joint_len == 0 and not ctx.use_uaa
    assert h3_vsa_owner_route_plan_is_trusted(plan)
    assert plan.local_rows == 4776 and plan.kmax == 132 and plan.sp_world_size == 8
    gate = ctx.o_bundle_gate_local
    assert gate is not None and gate.shape == (1,4776,56,128)
    rank = dist.get_rank(ctx.ulysses_pg)
    state = _state_for(fine,ctx.ulysses_pg,8)
    comm = state.communicator
    assert comm is not None
    # Cache these maps on the trusted, geometry-bound route's CPU tensors.
    tiles,row_map = route_maps(plan,rank,fine.device)
    row_ids = None
    if 'v16_fine' in active:
        from vllm_omni.diffusion.attention.ops.h3_fine_row_order import row_order

        row_ids = row_order(fine.device)[rank]
        # Gate was produced on this same side stream. Permuting complete rows
        # changes no arithmetic and leaves the original gate kernel intact.
        with torch.cuda.stream(active['side']), torch.cuda.nvtx.range('h3.fine_pipeline.gate_rows'):
            row_ids.record_stream(active['side'])
            gate = gate.index_select(1, row_ids)
    count = 1194
    result = torch.empty(1,4776,5376,device=fine.device,dtype=torch.bfloat16)
    from .h3_o_producer_lookahead import prepare as prepare_o, consume as consume_o, complete as complete_o
    prepare_o(active,fine,coarse,tiles,state)
    for index in range(4):
        start=index*count
        out,landing=consume_o(active,index)
        with torch.cuda.nvtx.range(f'h3.overlap.o_rdma.{index}'):
            comm.gather_heads(landing,out=out)
        active['side'].wait_stream(active['comm'])
        with torch.cuda.stream(active['side']), torch.cuda.nvtx.range(f'h3.overlap.o_gate_project.{index}'):
            view = gate_chunk(out,gate[:,start:start+count],row_map[start:start+count],count)
            projected = active['projector'](view)
            if row_ids is None:
                result[:,start:start+count].copy_(projected)
            else:
                # CPU validation proves row IDs are disjoint and cover every
                # original row exactly once, so this write has no reduction.
                assert projected.shape == (count, 5376)
                result[0].index_copy_(0, row_ids[start:start+count], projected)
    active['comm'].wait_stream(active['side'])
    result.record_stream(active['side'])
    gate.record_stream(active['side'])
    if 'v16_fine' in active:
        assert active['v16_fine']['consumed'] == 4
    complete_o(active)
    active['projected'] = True
    # Separate markers bind the validator to the chunked protocol and its
    # actual volume; they do not impersonate the original whole-bundle path.
    if not state.o_producer_direct_logged:
        logger.info('H3_OVERLAP_EXCHANGE rank=%d group=%s chunks=4 rows=1194 tail=132 remote_bytes=164326400 direct=1 completed=1',
                    rank,ctx.ulysses_pg.group_name)
        state.o_producer_direct_logged = True
    state.o_producer_direct_expected = None
    global _BUNDLE_LOGGED
    if not _BUNDLE_LOGGED:
        logger.info('H3_OVERLAP_BUNDLE rank=%d fine_rows=%d local_rows=4776 coarse_rows=675 chunks=4 tail=132 gate=bf16 projector=active',rank,fine.shape[1])
        _BUNDLE_LOGGED = True
    return result
