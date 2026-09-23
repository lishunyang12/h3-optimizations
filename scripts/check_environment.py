#!/usr/bin/env python3
"""Report architecture and installed versions without loading H3 weights."""
# SPDX-License-Identifier: Apache-2.0
import argparse
import importlib.metadata
import json
import platform


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu', action='store_true', help='Query the CUDA device via PyTorch')
    args = parser.parse_args()
    result = {'machine': platform.machine(), 'python': platform.python_version(), 'packages': {}}
    for package in ['torch', 'vllm', 'vllm-omni', 'flashinfer-python', 'triton', 'nvidia-cutlass-dsl']:
        try:
            result['packages'][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result['packages'][package] = None
    if args.gpu:
        import torch
        result['cuda_available'] = torch.cuda.is_available()
        result['torch_cuda'] = torch.version.cuda
        result['block_scaled_mm_api'] = hasattr(torch.nn.functional, 'scaled_mm')
        result['gpus'] = [{'name': torch.cuda.get_device_name(i),
                           'capability': list(torch.cuda.get_device_capability(i))}
                          for i in range(torch.cuda.device_count())]
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
