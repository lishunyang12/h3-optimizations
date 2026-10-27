#!/usr/bin/env python3
"""Two-workload container entry point. Downloads and cache preparation are explicit."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
CASES = ('5s-avatar-ref2va-4step', '15s-4step-mxfp8-sage-nvfp4')
RUNTIMES = ('/opt/h3-runtime-avatar5', '/opt/h3-runtime-15s')
DIRECTORIES = ('avatar-assets', 'fasth3-assets')


def run(*args, env=None):
    subprocess.run([str(x) for x in args], check=True, env=env, cwd=ROOT)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=('check', 'download', 'caches', 'run', 'exec'))
    p.add_argument('args', nargs=argparse.REMAINDER)
    a = p.parse_args()
    py = sys.executable
    if a.command == 'exec':
        if not a.args:
            p.error('exec requires a command')
        os.execvp(a.args[0], a.args)
    if a.command == 'check':
        sys.path.insert(0, str(ROOT / 'scripts'))
        from sku_benchmark import get_case
        for name in CASES:
            case = get_case(name)
            for package, expected in case['dependencies'].items():
                actual = importlib.metadata.version(package)
                if actual != expected:
                    raise RuntimeError(f'{package}: {actual} != {expected}')
        import torch
        import torchvision
        import diffusers
        import comfy_kitchen
        run('/opt/h3-cuda/nvidia/cu13/bin/nvcc', '--version')
        run(py, '-m', 'unittest', 'discover', '-s', 'tests', '-p', 'test_sku_benchmark.py')
        run(py, 'scripts/export_pro5000_reference.py', '--check')
        run(py, 'scripts/export_sku_tradeoffs.py', '--check')
        for runtime in RUNTIMES:
            env = dict(os.environ, PYTHONPATH=':'.join([runtime, str(ROOT / 'flashinfer'),
                       str(ROOT / 'fastvideo-kernel/python')]), VLLM_PLUGINS='', VLLM_USE_DEEP_GEMM='0')
            result = subprocess.run([py, str(Path(py).parent / 'vllm'), 'serve', '--omni', '--help'],
                                    env=env, capture_output=True, text=True)
            if result.returncode:
                raise RuntimeError(result.stdout + result.stderr)
            print(f'Runtime CLI import passed: {runtime}', flush=True)
        print('CPU/package checks passed; no GPU inference was performed.', flush=True)
        return
    if a.command == 'download':
        if a.args:
            p.error('download takes no arguments; mount a writable /models directory')
        for name, directory in zip(CASES, DIRECTORIES):
            run(py, 'scripts/sku_models.py', '--case', name, '--download',
                '--output', Path('/models') / directory)
        return
    if a.command == 'caches':
        if a.args:
            p.error('caches takes no arguments; mount the downloaded /models directory')
        for index, directory in enumerate(DIRECTORIES):
            assets = json.loads((Path('/models') / directory / 'model-assets.json').read_text())
            env = dict(os.environ, PYTHONPATH=RUNTIMES[index])
            if index == 0:
                run(py, 'scripts/sku_adaln.py', '--runtime', RUNTIMES[index],
                    '--transformer-path', Path(assets['model']) / 'transformer',
                    '--output', '/models/guide-adaln.safetensors', env=env)
            else:
                run(py, 'reference-runtime/tools/minimax_h3/build_adaln_cache.py',
                    '--transformer-path', Path(assets['model']) / 'transformer',
                    '--model-variant', 'fl2va', '--mode', 't2va',
                    '--fasth3-adapter', assets['adapter'],
                    '--base-schedule', '0.999', '0.749', '0.5', '0.25', '0.0',
                    '--flow-shift', '12', '--audio-flow-shift', '3',
                    '--output', '/models/fasth3-adaln.safetensors', env=env)
        return
    assets = {}
    for index, (name, directory) in enumerate(zip(CASES, DIRECTORIES)):
        source = json.loads((Path('/models') / directory / 'model-assets.json').read_text())
        assets[name] = dict(source, runtime=RUNTIMES[index], python=py,
                            cuda='/opt/h3-cuda/nvidia/cu13',
                            adaln='/models/' + ('guide' if index == 0 else 'fasth3') + '-adaln.safetensors')
    path = Path('/tmp/h3-container-assets.json')
    path.write_text(json.dumps(assets, indent=2) + '\n')
    os.execv(py, [py, str(ROOT / 'scripts/sku_benchmark.py'), 'suite', '--cases', 'main',
                 '--assets', str(path), *a.args])


if __name__ == '__main__':
    main()
