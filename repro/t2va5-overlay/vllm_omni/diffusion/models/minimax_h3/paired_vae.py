"""Experimental temporal/spatial VAE scheduling; original tile and blend math."""
from functools import lru_cache
import hashlib
import inspect
import math
import os

import torch
import torch.distributed as dist


@lru_cache(None)
def jobs(windows=21, tiles=28, world=8):
    assert windows > 0 and tiles > 0 and world == 8
    rounds=[]
    for first in range(0,windows,2):
        count=min(2,windows-first)*tiles
        rounds.append(tuple(tuple((first+i//tiles,i%tiles) for i in range(r,count,world)) for r in range(world)))
    counts=tuple(sum(len(row[r]) for row in rounds) for r in range(world))
    assert sum(counts) == windows * tiles
    flat=[job for row in rounds for rank in row for job in rank]
    assert len(flat)==len(set(flat))==windows*tiles
    assert set(flat)=={(w,t) for w in range(windows) for t in range(tiles)}
    return tuple(rounds)


def geometry(model,latent):
    from .short_window_experiment import latents
    assert latent.ndim==5 and tuple(latent.shape[:2])==(1,24) and latent.shape[2] in latents() and latent.dtype==torch.float32
    assert model.vae_ratio==16 and model.vae_ratio_t==4
    assert model.tokens_chunk_size==5 and model.token_overlap==2
    assert model.frame_pre_padding==3 and model.frame_overlap==5 and model.token_drop==3
    assert not model.training and not model.stack_tiling and not model.decoder_parallel
    assert model.decoder_tiling and model.decoder_tile_size==256
    yi,yl,yo=model.split_tiles(latent.shape[-2]*16,True);xi,xl,xo=model.split_tiles(latent.shape[-1]*16,True)
    assert len(yi)>0 and len(xi)>0 and set(yl+xl)=={256}
    slices=tuple((slice(y//16,(y+l)//16),slice(x//16,(x+k)//16))
                 for y,l in zip(yi,yl) for x,k in zip(xi,xl))
    return slices,tuple(yo),tuple(xo)


def assemble_spatial(model,tiles,y_overlap,x_overlap):
    # Same sequence as checkpoint AutoencoderKL.tiled_decode after gathering:
    # vertical blend, horizontal blend, trims, horizontal cats, vertical cat.
    nx=len(x_overlap)+1;ny=len(y_overlap)+1
    assert len(tiles)==nx*ny
    rows=[list(tiles[i*nx:(i+1)*nx]) for i in range(ny)]
    result_rows=[]
    for i,row in enumerate(rows):
        result_row=[]
        for j,tile in enumerate(row):
            if i>0:tile=model.blend(rows[i-1][j],tile,y_overlap[i-1],dim=-2)
            if j>0:tile=model.blend(row[j-1],tile,x_overlap[j-1],dim=-1)
            if i<len(rows)-1:tile=tile[...,:-y_overlap[i],:]
            if j<len(row)-1:tile=tile[...,:,:-x_overlap[j]]
            result_row.append(tile)
        result_rows.append(torch.cat(result_row,dim=-1))
    return torch.cat(result_rows,dim=-2)


def decode_pairs(model,latent,group,callback, *, temporal_cat_dtype=None, gather_stream=None):
    """Group wrapper requires world_size/rank_in_group/ranks/device_group/all_reduce."""
    from vllm_omni.diffusion.models.minimax_h3.temporal_chunk_parallel import (
        _LeaderAssembler,_gather_stack_to_rank_zero,_agree_on_failure,
    )
    assert group.world_size==8 and (callback is not None)==(group.rank_in_group==0)
    spatial,y_overlap,x_overlap=geometry(model,latent)
    windows=(int(latent.shape[2])-2)//5
    frames=windows*17+5
    schedule=jobs(windows,len(spatial),8);rank=group.rank_in_group
    assembler=(_LeaderAssembler(model,callback,main_frames=17,overlap_frames=5,output_frames=frames,device=latent.device)
               if callback is not None else None)
    early = os.environ.get("VLLM_OMNI_H3_VAE_EARLY_GATHER", "0")
    if early not in ("0", "1"):
        raise ValueError("VLLM_OMNI_H3_VAE_EARLY_GATHER must be 0 or 1")
    early = early == "1"
    if early and (gather_stream is None or os.environ.get("VLLM_OMNI_H3_VAE_BATCH4") != "1"):
        raise ValueError("Early gather requires separate gather stream and B4 decoder")
    gather_count=0
    early_count=0
    local_error=None
    tile_shape=(1,3,22,256,256)
    tile_dtype=torch.float32
    tile_count=0
    try:
        for round_id,round_jobs in enumerate(schedule):
            width=max(map(len,round_jobs))
            # Uniform collective shape, including four padding tiles in the
            # final round. Padding is never decoded or consumed by assembly.
            # B4 then B3. Send the first batch before computing the second.
            # Each bucket owns its allocation: never send a view that aliases
            # a buffer still being written by the decoder stream.
            splits = (4, width-4) if early and width > 4 else (width,)
            received=[]
            offset=0
            for bucket_id,bucket_width in enumerate(splits):
                packed=torch.empty((bucket_width,*tile_shape),device=latent.device,dtype=tile_dtype)
                for local_slot in range(bucket_width):
                    slot=offset+local_slot
                    if slot >= len(round_jobs[rank]) or local_error is not None:
                        packed[local_slot].zero_();continue
                    window,tile=round_jobs[rank][slot]
                    try:
                        sy,sx=spatial[tile]
                        clip=latent[:,:,window*5:window*5+7,sy,sx]
                        with torch.cuda.nvtx.range(f'h3.vae.pair.tile.{window}.{tile}'):
                            decoded=model.decode(clip)
                        assert decoded.shape==(1,3,28,256,256) and decoded.dtype==tile_dtype
                        packed[local_slot,:,:,:17].copy_(decoded[:,:,3:20])
                        packed[local_slot,:,:,17:].copy_(decoded[:,:,23:28])
                        tile_count+=1
                    except BaseException as e:
                        e.__traceback__=None;local_error=e;packed[local_slot].zero_()
                if bucket_id == 0 and assembler is not None:assembler.drain_previous_round()
                with torch.cuda.nvtx.range(f'h3.vae.pair.gather.{round_id}.{bucket_id}'):
                    if gather_stream is None:
                        gathered=_gather_stack_to_rank_zero(packed,group)
                    else:
                        from .vae_gather_overlap import gather_on_stream
                        gathered=gather_on_stream(packed,group,gather_stream)
                received.append(gathered)
                gather_count+=1
                if early and bucket_id==0 and len(splits)==2:early_count+=1
                offset+=bucket_width
                del packed
            gathered=received[0]
            if assembler is not None and gathered is not None:
                # Assemble on the dedicated output stream, allowing the next
                # pair's decoder to run concurrently. The original temporal
                # assembler preserves callback order and the five-frame tail.
                stream=assembler._stream
                stream.wait_stream(gather_stream if gather_stream is not None else
                                   torch.cuda.current_stream(latent.device))
                for part in received:part.record_stream(stream)
                try:
                    with torch.cuda.stream(stream):
                        for window in range(round_id*2,min(round_id*2+2,windows)):
                            parts=[None]*len(spatial)
                            for owner,owner_jobs in enumerate(round_jobs):
                                for slot,(w,t) in enumerate(owner_jobs):
                                    if w==window:
                                        bucket=0 if slot < splits[0] else 1
                                        within=slot if bucket==0 else slot-splits[0]
                                        parts[t]=received[bucket][owner,within]
                            assert all(t is not None for t in parts)
                            segment=assemble_spatial(model,parts,y_overlap,x_overlap)
                            if temporal_cat_dtype is not None:segment=segment.to(temporal_cat_dtype)
                            assembler.push(segment)
                        assembler.finish_round()
                except BaseException as e:
                    assembler._remember_error(e)
            del gathered,received
        if assembler is not None:assembler.finalize()
    finally:
        # Include non-output ranks and exceptional exits. No gather may still
        # access a send/receive allocation when this request is released or
        # the all-rank failure agreement enters the original caller stream.
        try:
            if gather_stream is not None:gather_stream.synchronize()
        finally:
            if assembler is not None:assembler.synchronize()
    callback_error=assembler.error if assembler is not None else None
    if _agree_on_failure(group,latent.device,local_error is not None or callback_error is not None):
        raise RuntimeError('Paired VAE failed after draining collectives') from (local_error or callback_error)
    assert tile_count==sum(len(row[rank]) for row in schedule)
    if gather_stream is not None:
        from vllm.logger import init_logger

        init_logger(__name__).info(
            'H3_VAE_GATHER_OVERLAP rank=%d gathers=%d separate_stream=1 '
            'send_record_stream=1 tiles=%d full_vae=1 '
            'mixed=1 gather_drained=1 frames=%d complete=1',rank,gather_count,tile_count,frames)
    if early:
        expected_early=sum(max(map(len,row))>4 for row in schedule)
        assert gather_count==len(schedule)+expected_early and early_count==expected_early
        init_logger(__name__).info('H3_VAE_EARLY_GATHER %s',
            __import__('json').dumps(dict(rank=rank,gathers=gather_count,early_batches=early_count,
                decoder_calls=len(__import__(__package__+'.mixed_vae',fromlist=['batch_plan']).batch_plan(rank,len(spatial),windows)),tiles=tile_count,payload_dtype='float32',fallback=0,status='PASS'),sort_keys=True))
    return dict(rank=rank,tile_calls=tile_count,gathers=gather_count,windows=windows,padding_tiles=sum(max(map(len,row))*8-sum(map(len,row)) for row in schedule),frames=frames)
