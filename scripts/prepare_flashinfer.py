# SPDX-License-Identifier: Apache-2.0
"""Fetch the exact optional C++ header dependencies recorded by this snapshot."""
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
modules = json.loads((ROOT / 'configs/flashinfer-third-party.json').read_text())
for name, spec in modules.items():
    target = ROOT / 'flashinfer/3rdparty' / name
    if target.exists() and any(target.iterdir()):
        actual = subprocess.check_output(['git', '-C', str(target), 'rev-parse', 'HEAD'], text=True).strip()
        if actual != spec['revision']:
            raise SystemExit(f'{name}: existing checkout differs; refusing to replace it')
        print(f'{name}: pinned checkout already present')
        continue
    target.mkdir(parents=True, exist_ok=True)
    subprocess.run(['git', 'init', str(target)], check=True)
    subprocess.run(['git', '-C', str(target), 'fetch', '--depth=1', spec['url'], spec['revision']], check=True)
    subprocess.run(['git', '-C', str(target), 'checkout', '--detach', 'FETCH_HEAD'], check=True)
    print(f'{name}: ready at {spec["revision"]}')
