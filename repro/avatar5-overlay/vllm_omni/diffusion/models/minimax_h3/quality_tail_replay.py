"""Opt-in local diagnostic: restore an authored tail while holding clocks fixed."""
import os,re
from pathlib import Path
import torch
from vllm_omni.errors import OmniClientError

def restore(key,session,kwargs):
    from .rolling_continuation import Tail
    directory=os.environ.get('H3_QUALITY_TAIL_REPLAY_DIR')
    if not directory or not isinstance(key,str) or not re.fullmatch(r'[a-z0-9_-]{1,48}',key):
        raise OmniClientError('Quality tail replay is disabled or key is invalid')
    root=Path(directory).resolve();path=(root/(key+'.pt')).resolve()
    if path.parent!=root:raise OmniClientError('Quality tail must remain in the configured diagnostic directory')
    data=torch.load(path,map_location='cpu',weights_only=True)
    if set(data)!={'video','audio','ordinal','end','media_origin','window_frames','overlap_frames','position_mode'}:
        raise OmniClientError('Invalid diagnostic tail contract')
    h,w=kwargs['latent_h'],kwargs['latent_w']
    if (tuple(data['video'].shape)!=(1,24,7,h,w) or tuple(data['audio'].shape)!=(2,32,37)
        or data['video'].dtype!=torch.float32 or data['audio'].dtype!=torch.float32
        or not torch.isfinite(data['video']).all() or not torch.isfinite(data['audio']).all()):
        raise OmniClientError('Invalid diagnostic tail tensors')
    from .short_window_experiment import windows
    frames = data['window_frames']
    if (type(frames) is not int or frames not in (56,73,124) or frames not in windows() or data['overlap_frames']!=5 or data['position_mode']!='window'
        or type(data['ordinal']) is not int or data['ordinal']<0
        or data['end']!=frames+(frames-5)*data['ordinal'] or type(data['media_origin']) is not int):
        raise OmniClientError('Invalid diagnostic tail clock')
    device=kwargs['text_embeddings'].device
    return Tail(session,data['ordinal'],data['end'],data['media_origin'],data['video'].to(device),data['audio'].to(device),5,'window',frames,'none')
