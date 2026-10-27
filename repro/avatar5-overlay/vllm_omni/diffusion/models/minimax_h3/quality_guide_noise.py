"""Experimental RF-noised visual continuation anchors, opt-in only.

Reference image/audio and raw committed tails are unchanged. Guide values and
guide timestep labels must change together. No output-frame filtering is used.
"""
import os
import torch
from vllm_omni.errors import OmniClientError
from .condition_noise import minimax_h3_imgvid_cond_noise_aug_rows

GUIDE_TIMESTEPS = (0.999, 0.98, 0.95, 0.9, 0.8, 0.5, 0.25)

def validate_timestep(value):
    if type(value) not in (int, float) or value not in GUIDE_TIMESTEPS:
        raise OmniClientError(f'Guide timestep must be one of {GUIDE_TIMESTEPS}')
    if value != 0.999 and os.environ.get('H3_QUALITY_GUIDE_NOISE') != '1':
        raise OmniClientError('Experimental guide noise is disabled')
    return float(value)

def apply_guide_noise(branch, clean_rows, anchor, *, ref_blocks,
                      condition_shapes, target_latent_t, seed):
    guides = [block for block in (ref_blocks or []) if block.get('kind') == 'latent_guide']
    if not guides:
        return anchor
    if len(guides) != 1 or ref_blocks[-1] is not guides[0]:
        raise OmniClientError('Visual guide noise requires exactly one final guide')
    tau = validate_timestep(guides[0].get('quality_guide_timestep', 0.999))
    if tau == 0.999:
        return anchor
    t, h, w = condition_shapes[-1]
    count = t * (h // 2) * (w // 2)
    if (t,h,w) != tuple(guides[0][k] for k in ('latent_t','latent_h','latent_w')):
        raise OmniClientError('Visual guide geometry does not match final condition')
    # Preserve the original dependent-noise template length and CPU FP32 recipe.
    guide_anchor = minimax_h3_imgvid_cond_noise_aug_rows(
        clean_rows[-count:], condition_shapes=[condition_shapes[-1]],
        target_latent_t=target_latent_t,
        imgvid_cond_num_frames=len(condition_shapes), seed=seed, noise_aug=tau)
    out = anchor.clone()
    out[-count:] = guide_anchor
    positions = branch.img_pos_dev[~branch.update_mask_dev][-count:]
    branch.quality_guide_timestep = (positions, tau)
    return out

def apply_guide_timesteps(branch, timesteps, *, t_video):
    guide = getattr(branch, 'quality_guide_timestep', None)
    if guide is not None:
        positions, tau = guide
        # Qualified four-step Ref2VA schedules stay below every tested tau.
        # Reject other schedules instead of labelling a fixed anchor incorrectly.
        if t_video > tau:
            raise OmniClientError('Guide timestep is below the target timestep')
        timesteps[positions] = tau
