#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Build the deployed avatar's AdaLN sidecar from its saved 34 timestep plans.

Run in the pinned H3 environment. Build once and copy the same sidecar to both
machines: independent GPU builds may differ in floating-point rounding.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime', type=Path, required=True)
    parser.add_argument('--transformer-path', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a new output path; refusing to replace a sidecar')
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(args.runtime.resolve()))
    import torch
    spec = importlib.util.spec_from_file_location(
        'native_builder', root / 'reference-runtime/tools/minimax_h3/build_adaln_cache.py')
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    plans = json.loads((root / 'repro/avatar5-adaln-plans.json').read_text())
    builder._cache_timestep_plans = lambda _: [torch.tensor(p, dtype=torch.float32) for p in plans]
    torch.set_num_threads(8)
    sys.argv = [str(spec.origin), '--transformer-path', str(args.transformer_path),
                '--output', str(args.output), '--model-variant', 'ref2va',
                '--mode', 'ref2va-mixed', '--num-inference-steps', '4', '--device', args.device]
    builder.main()


if __name__ == '__main__':
    main()
