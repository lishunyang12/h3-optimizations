"""Experimental exact target-prefix constraints; retain the full output geometry."""
import torch


def install(branch, video=None, audio=None):
    for kind, values, mask in (('video', video, branch.update_mask),
                               ('audio', audio, branch.audio_update_mask)):
        if values is None:
            continue
        target = torch.nonzero(mask, as_tuple=False).flatten()
        width = 96 if kind == 'video' else 32
        if values.ndim != 2 or values.shape[1] != width or not 0 < len(values) < len(target):
            raise ValueError('Invalid rolling target prefix geometry')
        if values.dtype != torch.float32:
            raise ValueError('Target prefix must use the original FP32 latent')
        if kind == 'audio':
            # Native stereo rows are channel-major: [L0..LT, R0..RT].
            # Each channel needs its own opening; the first 2*N rows would
            # incorrectly constrain only the left channel.
            if len(values) % 2 or len(target) % 2:
                raise ValueError('Stereo prefix and target need two equal channels')
            ticks = len(values) // 2
            target_ticks = len(target) // 2
            indices = torch.cat((target[:ticks], target[target_ticks:target_ticks + ticks]))
        else:
            indices = target[:len(values)]
        # Output masks continue to select the entire target, including its
        # disposable overlap. Only scheduler updates for these rows are reset.
        setattr(branch, '_quality_' + kind + '_prefix',
                (indices.to(branch.device), values.detach().to(branch.device).clone()))


def apply_timesteps(branch, timesteps):
    for kind, positions in (('video', branch.img_pos_dev), ('audio', branch.audio_pos_dev)):
        prefix = getattr(branch, '_quality_' + kind + '_prefix', None)
        if prefix is not None:
            indices, _ = prefix
            # RF convention: t=1 is clean. This timestep already exists in the
            # exact cache plan because Ref2VA has a clean reference-audio block.
            timesteps[positions[indices]] = 1.0


def hold(branch, video_rows, audio_rows):
    for kind, rows in (('video', video_rows), ('audio', audio_rows)):
        prefix = getattr(branch, '_quality_' + kind + '_prefix', None)
        if prefix is not None:
            indices, values = prefix
            rows.index_copy_(0, indices, values)
