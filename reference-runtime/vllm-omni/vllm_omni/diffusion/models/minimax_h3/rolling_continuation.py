# SPDX-License-Identifier: Apache-2.0
"""One bounded continuation window per request, with per-worker AV tail state.

Guide placement follows vllm-omni PR #7838 (1ac197ab7). We keep only a 22-frame
latent tail, not all past latents. No crossfade or synthetic transition is used.
"""
from dataclasses import dataclass
import json
import re
import torch
from vllm.logger import init_logger
from vllm_omni.errors import OmniClientError
from .packed_tokens import minimax_h3_pack_audio_latent, minimax_h3_patchify_video_latent

log = init_logger(__name__)
WINDOW, OVERLAP, TAIL_VIDEO, TAIL_AUDIO = 362, 22, 7, 37

def audio_boundary(frame):
    return round(frame * 5 / 3)

@dataclass
class Tail:
    session: str
    ordinal: int
    end: int
    media_origin: int
    video: torch.Tensor
    audio: torch.Tensor
    overlap_frames: int = OVERLAP
    position_mode: str = 'global'
    window_frames: int = WINDOW

@dataclass
class Pending:
    session: str
    ordinal: int
    start: int
    end: int
    overlap_frames: int
    overlap_audio_tokens: int
    media_origin: int
    kwargs: dict
    configured_overlap: int = OVERLAP
    position_mode: str = 'global'

def prepare(previous, options, kwargs, *, num_outputs):
    if (not isinstance(options, dict) or not {'session', 'ordinal'} <= set(options)
            or set(options) - {'session', 'ordinal', 'overlap_frames', 'position_mode', 'window_frames'}):
        raise OmniClientError('rolling_continuation requires session and ordinal')
    window_frames = options.get('window_frames', WINDOW)
    if type(window_frames) is not int or window_frames not in (124, 141, WINDOW):
        raise OmniClientError('Qualified rolling windows are 124, 141 or 362 frames')
    latent_t = (window_frames - 5) // 17 * 5 + 2
    session, ordinal = options['session'], options['ordinal']
    configured_overlap = options.get('overlap_frames', OVERLAP)
    position_mode = options.get('position_mode', 'global')
    if type(configured_overlap) is not int or configured_overlap not in (5, OVERLAP):
        raise OmniClientError('Rolling overlap must be 5 or 22 native frames')
    if position_mode not in ('global', 'window'):
        raise OmniClientError('Rolling position_mode must be global or window')
    if not isinstance(session, str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,64}', session):
        raise OmniClientError('Invalid rolling continuation session')
    if isinstance(ordinal, bool) or not isinstance(ordinal, int) or ordinal < 0:
        raise OmniClientError('Rolling ordinal must be a nonnegative integer')
    if num_outputs != 1 or kwargs['task'] != 'ref2va' or kwargs['num_frames'] != window_frames or kwargs['latent_t'] != latent_t:
        raise OmniClientError('Rolling continuation requires one Ref2VA output matching its qualified window')
    if ordinal == 0:
        start, overlap, overlap_a = 0, 0, 0
        media_origin = int(kwargs['text_embeddings'].shape[0]) + 128
    else:
        if previous is None or previous.session != session or previous.ordinal + 1 != ordinal:
            raise OmniClientError('Missing/out-of-order continuation state; start a new session at ordinal 0')
        if previous.window_frames != window_frames:
            raise OmniClientError('Continuation window length changed inside a session')
        if previous.video.shape[-2:] != (kwargs['latent_h'], kwargs['latent_w']):
            raise OmniClientError('Continuation canvas changed')
        if (previous.overlap_frames, previous.position_mode) != (configured_overlap, position_mode):
            raise OmniClientError('Continuation overlap/position policy changed inside a session')
        start, overlap = previous.end - configured_overlap, configured_overlap
        overlap_a = audio_boundary(previous.end) - audio_boundary(start)
        media_origin = previous.media_origin
    if kwargs['text_embeddings'].shape[0] > media_origin:
        raise OmniClientError('Prompt exceeds the fixed continuation media origin; restart the session')
    end = start + window_frames
    args = {**kwargs, 'audio_t': audio_boundary(end) - audio_boundary(start),
            'temporal_offset': start * (40.0 / 24.0) if position_mode == 'global' else 0.0,
            'media_time_origin': media_origin}
    if ordinal:
        guide_video_t = (overlap - 5) // 17 * 5 + 2
        gv = minimax_h3_patchify_video_latent(previous.video[:, :, -guide_video_t:], patch_size=(1,2,2))
        ga = minimax_h3_pack_audio_latent(previous.audio[..., -overlap_a:])
        for name, guide in (('visual_condition', gv), ('audio_condition', ga)):
            old = kwargs.get(name)
            args[name] = guide if old is None else torch.cat((old, guide), dim=0)
        args['visual_condition_shapes'] = [*(kwargs.get('visual_condition_shapes') or []),
            (guide_video_t, kwargs['latent_h'], kwargs['latent_w'])]
        args['audio_condition_lengths'] = [*(kwargs.get('audio_condition_lengths') or []), overlap_a]
        args['ref_blocks'] = [*(kwargs.get('ref_blocks') or []), dict(kind='latent_guide',
            latent_t=guide_video_t, latent_h=kwargs['latent_h'], latent_w=kwargs['latent_w'], ref_audio_t=overlap_a)]
    return Pending(session, ordinal, start, end, overlap, overlap_a, media_origin, args,
                   configured_overlap, position_mode)

def commit(pending, video, audio):
    if video.shape[2] != pending.kwargs['latent_t'] or audio.shape[-1] != pending.kwargs['audio_t']:
        raise RuntimeError('Rolling sampler produced unexpected AV geometry')
    # clone is essential: a slice would retain the entire old GPU allocation.
    result = Tail(pending.session, pending.ordinal, pending.end, pending.media_origin,
                  video[:, :, -TAIL_VIDEO:].detach().clone(), audio[..., -TAIL_AUDIO:].detach().clone(),
                  pending.configured_overlap, pending.position_mode, pending.kwargs['num_frames'])
    log.info('H3_ROLLING_CONTINUATION %s', json.dumps(dict(session=pending.session,
        ordinal=pending.ordinal, window_start=pending.start, window_end=pending.end,
        overlap_frames=pending.overlap_frames, overlap_audio_tokens=pending.overlap_audio_tokens,
        audio_tokens=audio.shape[-1], media_origin=pending.media_origin,
        configured_overlap=pending.configured_overlap, position_mode=pending.position_mode,
        temporal_offset=pending.kwargs['temporal_offset'],
        saved_video_tokens=result.video.shape[2], saved_audio_tokens=result.audio.shape[-1],
        retained_bytes=result.video.numel()*result.video.element_size()+result.audio.numel()*result.audio.element_size(),
        motion_context=bool(pending.ordinal), accumulated_full_latents=False)))
    return result
