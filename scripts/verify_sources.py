#!/usr/bin/env python3
"""Check bundled source hashes and packaging changes, without importing CUDA."""
# SPDX-License-Identifier: Apache-2.0
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    manifest = json.loads((ROOT / 'evidence/source-sha256.json').read_text())
    changes_file = ROOT / 'evidence/packaging-changes.json'
    changes = json.loads(changes_file.read_text()) if changes_file.exists() else {}
    failures = []
    for name, original in manifest.items():
        expected = changes.get(name, {}).get('after', original)
        path = ROOT / name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            failures.append(name)
    print(json.dumps({'checked': len(manifest), 'packaging_changes': len(changes),
                      'failures': failures}, indent=2))
    raise SystemExit(bool(failures))


if __name__ == '__main__':
    main()
