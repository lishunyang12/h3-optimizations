#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Export the two historical PRO 5000 references from checked receipts.

No GPU access. --check detects stale or edited published JSON/CSV files.
"""
import argparse
import csv
import io
import json
from pathlib import Path
import statistics

from sku_benchmark import ROOT, digest, get_case, identity, load

CASES = ('5s-avatar-ref2va-4step', '15s-4step-mxfp8-sage-nvfp4')


def render():
    audit = load(ROOT / 'evidence/sku-evidence-audit.json')
    audited = {row['case']: row for row in audit['cases']}
    avatar = load(ROOT / 'evidence/sku-avatar5-history.json')
    fasth3 = load(ROOT / 'evidence/sku-15s-history.json')
    cases = []
    csv_buffer = io.StringIO(newline='')
    writer = csv.writer(csv_buffer, lineterminator='\n')
    writer.writerow(['case', 'request_ordinal', 'phase', 'http_seconds',
                     'application_ready_seconds', 'video_frames', 'fps', 'video_seconds'])
    for name in CASES:
        case = get_case(name)
        history = case['historical']
        if name == CASES[0]:
            receipt = 'evidence/sku-avatar5-history.json'
            measured = [row for row in avatar['records'] if row['phase'] == 'measured']
            samples = [row['http_seconds'] for row in measured]
            for row in avatar['records']:
                writer.writerow([name, row['ordinal'], row['phase'], row['http_seconds'],
                                 row['ready_seconds'], row['frames'], 24, row['media_seconds']])
        else:
            receipt = 'evidence/sku-15s-history.json'
            samples = fasth3['e2e_seconds']
            for index, seconds in enumerate(samples, 1):
                writer.writerow([name, index, 'measured', seconds, '',
                                 case['media']['frames'], 24, case['media']['duration_seconds']])
        if samples != history.get('http_samples_s', history.get('samples_s')):
            raise ValueError(f'Profile samples differ from receipt: {name}')
        if samples != audited[name]['samples_s']:
            raise ValueError(f'Samples differ from artifact audit: {name}')
        median = statistics.median(samples)
        if median != audited[name]['median_s']:
            raise ValueError(f'Median differs from artifact audit: {name}')
        rows = {
            'case': name, 'status': 'historical_measurement',
            'measurement_date': history.get('date', history.get('measured_date')),
            'gpu_count': case['topology']['num_gpus'], 'steps': 4,
            'media': case['media'], 'topology': case['topology'],
            'warmup_requests': case['protocol']['warmups'], 'measured_requests': len(samples),
            'http_samples_s': samples, 'http_median_s': median,
            'http_mean_s': statistics.mean(samples),
            'model': case['model'], 'runtime_variant': case['runtime_variant'],
            'reproduction_case_identity': identity(case),
            'receipt': receipt, 'receipt_sha256': digest(ROOT / receipt),
            'original_artifact': audited[name]['source_artifact'],
            'original_artifact_sha256': audited[name]['source_sha256'],
        }
        if name == CASES[0]:
            rows.update(first_output_frames=case['first_media_frames'],
                        application_ready_samples_s=history['ready_samples_s'],
                        application_ready_median_s=statistics.median(history['ready_samples_s']),
                        fixed_requests='repro/avatar5-requests.json',
                        fixed_requests_sha256=digest(ROOT / 'repro/avatar5-requests.json'),
                        reference_image_sha256=case['reference_sha256'])
        cases.append(rows)
    data = {
        'schema': 1, 'status': 'historical_reference_only',
        'hardware': audit['hardware_identification'],
        'timing': 'Historical HTTP request through completed MP4; warmup excluded from medians. Avatar application-ready time is separate.',
        'comparison': 'Run the same new launcher on both systems before claiming a strict cross-SKU speedup.',
        'target_pro5500': {'hardware_verified': False, 'measurements': None},
        'cases': cases,
    }
    return {'results.json': json.dumps(data, ensure_ascii=True, indent=2) + '\n',
            'timings.csv': csv_buffer.getvalue()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'benchmarks/pro5000')
    parser.add_argument('--check', action='store_true', help='Verify existing exports without writing')
    args = parser.parse_args()
    contents = render()
    if args.check:
        for name, expected in contents.items():
            path = args.output / name
            if not path.is_file() or path.read_text() != expected:
                raise SystemExit(f'Stale export: {path}; regenerate from the checked receipts')
        print('PRO 5000 JSON/CSV match both historical receipts and the artifact audit')
    else:
        args.output.mkdir(parents=True, exist_ok=True)
        for name, text in contents.items():
            (args.output / name).write_text(text)
        print(f'Exported two historical references to {args.output}')


if __name__ == '__main__':
    main()
