#!/usr/bin/env python3
"""Build isolated single-GPU H3 experiments; print the command unless --run."""
# SPDX-License-Identifier: Apache-2.0
import argparse
import json
import os
from pathlib import Path
import shlex
import sys

ROOT = Path(__file__).resolve().parents[1]
PROFILES = ('baseline', 'adaln', 'mxfp8', 'audio-overlap')


def configuration(model, profile, port):
    if profile not in PROFILES:
        raise ValueError(f'Unknown profile: {profile}')
    # Set each component explicitly: global FP8 defaults change across revisions.
    quant = {
        'transformer': {'method': 'mxfp8' if profile == 'mxfp8' else 'fp8'},
        'text_encoder': None, 'video_vae': None, 'audio_vae': None,
    }
    environment = {
        'CUDA_VISIBLE_DEVICES': '0', 'VLLM_WORKER_MULTIPROC_METHOD': 'spawn',
        'VLLM_OMNI_VIDEO_SYNC_TIMEOUT': '14400',
        'VLLM_OMNI_H3_VAE_AUDIO_OVERLAP': '1' if profile == 'audio-overlap' else '0',
    }
    command = [
        sys.executable, '-m', 'vllm_omni.entrypoints.cli.main', 'serve', model,
        '--omni', '--trust-remote-code', '--host', '127.0.0.1', '--port', str(port),
        '--init-timeout', '3600', '--num-gpus', '1', '--tensor-parallel-size', '1',
        '--text-encoder-tp-size', '1', '--usp', '1', '--ring', '1',
        '--vae-patch-parallel-size', '1', '--vae-parallel-mode', 'tile',
        '--vae-use-tiling', '--enforce-eager',
        '--diffusion-attention-backend', 'CUDNN_ATTN',
        '--diffusion-quantization-config', json.dumps(quant, separators=(',', ':')),
        '--cache-config', json.dumps({'minimax_h3_adaln_cache': profile != 'baseline'}),
        '--enable-diffusion-pipeline-profiler',
    ]
    return environment, command


def isolated_environment(updates):
    # An SP8 experiment shell can carry dozens of flags. Remove all H3/transport
    # tuning first so the selected profile actually controls this experiment.
    prefixes = ('VLLM_OMNI_H3_', 'VLLM_OMNI_MINIMAX_H3_', 'VLLM_OMNI_FLASHINFER_ULYSSES_',
                'VLLM_OMNI_ULYSSES_', 'FLASHINFER_ULYSSES_', 'H3_', 'MINIMAX_H3_')
    exact = {'TORCH_CUDA_ARCH_LIST', 'CUTE_DSL_ARCH', 'FLASHINFER_CUDA_ARCH_LIST'}
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(prefixes) and k not in exact}
    # Use the bundled integration and the installed platform-compatible
    # FlashInfer. Opting into experimental FlashInfer is documented separately.
    env['PYTHONPATH'] = str(ROOT / 'vllm-omni')
    env.update(updates)
    return env


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True, help='Local FL2VA or Ref2VA partition')
    parser.add_argument('--profile', choices=PROFILES, default='adaln')
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--run', action='store_true', help='Actually start the server')
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error('--port must be between 1 and 65535')
    env, command = configuration(args.model, args.profile, args.port)
    print(json.dumps({'profile': args.profile, 'environment': env,
                      'pythonpath': str(ROOT / 'vllm-omni'),
                      'command': shlex.join(command), 'run': args.run}, indent=2), flush=True)
    if args.run:
        if not Path(args.model).is_dir():
            parser.error('--model must be an existing local model partition directory')
        os.execvpe(command[0], command, isolated_environment(env))


if __name__ == '__main__':
    main()
