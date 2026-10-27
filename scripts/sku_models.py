#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Download pinned legacy model assets, or prepare a view of existing weights.

No GPU work. Does not download unless --download is specified. Obtain weights
under their upstream terms; the reproducibility archive contains no weights.
"""
import argparse
import json
from pathlib import Path
import shutil
from sku_benchmark import get_case, digest, save


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--case', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--download', action='store_true')
    p.add_argument('--base', type=Path, help='Existing MiniMax-H3 directory containing FL2VA or Ref2VA')
    p.add_argument('--adapter', type=Path)
    args = p.parse_args()
    case = get_case(args.case)
    if case['family'] not in ('legacy15', 'avatar5'):
        p.error('This package supports the 5s avatar and 15s FastH3 reference only')
    if args.output.exists():
        p.error('Output must be new; refusing to replace existing assets')
    model = case['model']
    partition = model['partition']
    if args.download:
        if args.base or args.adapter:
            p.error('Use --download or --base plus --adapter')
        from huggingface_hub import snapshot_download, hf_hub_download
        args.output.mkdir(parents=True)
        base = Path(snapshot_download(model.get('base_repository', model.get('base_repo')),
                    revision=model['base_revision'], local_dir=args.output / 'MiniMax-H3',
                    allow_patterns=[partition + '/*', 'LICENSE*', 'README.md']))
        adapter = Path(hf_hub_download(model.get('adapter_repository', model.get('adapter_repo')),
                       model['adapter_file'], revision=model['adapter_revision'],
                       local_dir=args.output / 'adapter'))
    else:
        if not args.base or not args.adapter:
            p.error('Supply --download, or both --base and --adapter')
        base, adapter = args.base.resolve(), args.adapter.resolve()
    if digest(adapter) != model['adapter_sha256']:
        raise ValueError('Adapter does not match the historical SHA-256')
    checkpoint = base / partition
    if not (checkpoint / 'transformer/config.json').is_file():
        raise ValueError(f'Missing checkpoint partition: {checkpoint}')
    args.output.mkdir(parents=True, exist_ok=True)
    if case['family'] == 'avatar5':
        view = args.output / 'ref2va-fourstep/Ref2VA'
        view.mkdir(parents=True)
        for name in ('audio_vae', 'video_vae', 'text_encoder', 'tokenizer', 'processor', 'transformer'):
            source = (checkpoint / name).resolve(strict=True)
            (view / name).symlink_to(source, target_is_directory=True)
        shutil.copy2(Path(__file__).resolve().parents[1] / 'repro/avatar5-model-index.json', view / 'model_index.json')
        checkpoint = view
    result = {'model': str(checkpoint.resolve()), 'adapter': str(adapter.resolve())}
    save(args.output / 'model-assets.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
