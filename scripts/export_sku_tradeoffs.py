#!/usr/bin/env python3
"""Validate retained stage measurements and export their per-request timings."""
import argparse
import csv
import io
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]


def render():
    data = json.loads((ROOT / 'evidence/sku-tradeoff-history.json').read_text())
    sources = {r['artifact'] for r in data['original_sources']}
    buffer = io.StringIO(newline='')
    writer = csv.writer(buffer, lineterminator='\n')
    writer.writerow(['comparison_group', 'arm', 'gpu_count', 'width', 'height',
                     'frames', 'fps', 'formal_sample', 'http_seconds'])
    for group in data['groups']:
        for arm in group['arms']:
            samples = arm['http_samples_s']
            assert arm['source_artifact'] in sources
            assert arm['measured_requests'] == len(samples)
            assert arm['http_median_s'] == statistics.median(samples)
            if group['id'].startswith('avatar-bf16-'):
                assert samples == [r['http_seconds'] for r in arm['records'] if r['phase'] == 'formal']
                assert all(r['full_av_decode'] for r in arm['records'])
            for index, value in enumerate(samples, 1):
                writer.writerow([group['id'], arm['id'], group['gpu_count'],
                                 group['width'], group['height'], group['frames'],
                                 group['fps'], index, value])
    return buffer.getvalue()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    target = ROOT / 'benchmark/SM120/tradeoff-timings.csv'
    content = render()
    if args.check:
        if not target.exists() or target.read_text() != content:
            raise SystemExit('Stale tradeoff CSV; rerun scripts/export_sku_tradeoffs.py')
        print('All 14 historical arms and their CSV samples verified')
    else:
        target.write_text(content)


if __name__ == '__main__':
    main()
