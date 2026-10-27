"""Capture original decoded FP32 samples before AAC; never alter model audio."""
import hashlib
import json
import os
from pathlib import Path
import re

import numpy as np


def destination(enabled, rolling):
    if type(enabled) is not bool:
        raise ValueError('capture_quality_pcm must be boolean')
    if not enabled:
        return None
    directory = os.environ.get('H3_QUALITY_PCM_DIR')
    if not directory or rolling is None or not re.fullmatch(r'[A-Za-z0-9_-]{1,64}', rolling.session):
        raise ValueError('PCM capture requires a configured directory and valid rolling session')
    root = Path(directory).resolve()
    root.mkdir(parents=True, exist_ok=True)
    result = root / f'{rolling.session}-{rolling.ordinal:04d}.npy'
    if result.exists() or result.is_symlink() or result.with_suffix('.json').exists():
        raise RuntimeError('Refusing to overwrite captured PCM')
    return result


def save(path, host, sample_rate, gain_db):
    if path is None:
        return
    from .chunked_cpu_output import _audio_to_numpy
    samples = _audio_to_numpy(host)
    if not np.isfinite(samples).all():
        raise RuntimeError('Nonfinite raw audio')
    with path.open('xb') as output:
        np.save(output, samples, allow_pickle=False)
    metadata = dict(format='h3-fp32-pcm-v1', sample_rate=int(sample_rate), channels=2,
        samples=samples.shape[1], gain_db=float(gain_db), gain_applied=False,
        dtype='float32', pcm_sha256=hashlib.sha256(samples.tobytes()).hexdigest(),
        file_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    with path.with_suffix('.json').open('x') as output:
        json.dump(metadata, output, indent=2)
        output.write('\n')
