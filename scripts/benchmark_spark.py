#!/usr/bin/env python3
"""Measure serial local H3 HTTP requests and retain outputs and response headers."""
# SPDX-License-Identifier: Apache-2.0
import argparse
import json
import mimetypes
from pathlib import Path
import shutil
import statistics
import subprocess
import time
import urllib.request
import uuid


def multipart(fields, reference=None):
    boundary = uuid.uuid4().hex
    parts = []
    for key, value in fields.items():
        parts.append((f'--{boundary}\r\nContent-Disposition: form-data; name="{key}"'
                      f'\r\n\r\n{value}\r\n').encode())
    if reference:
        content_type = mimetypes.guess_type(reference.name)[0] or 'application/octet-stream'
        # A fixed filename avoids embedding client-local paths in the request.
        parts.append((f'--{boundary}\r\nContent-Disposition: form-data; '
                      'name="input_reference"; filename="reference"\r\n'
                      f'Content-Type: {content_type}\r\n\r\n').encode())
        parts.extend([reference.read_bytes(), b'\r\n'])
    parts.append(f'--{boundary}--\r\n'.encode())
    return b''.join(parts), f'multipart/form-data; boundary={boundary}'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='http://127.0.0.1:8000/v1/videos/sync')
    parser.add_argument('--output', type=Path, required=True, help='New result directory')
    parser.add_argument('--prompt', default='Three cats walk through a quiet room, with soft footsteps.')
    parser.add_argument('--reference', type=Path, help='Image for a Ref2VA server; otherwise T2VA')
    parser.add_argument('--steps', type=int, default=10)
    parser.add_argument('--duration', type=float, default=4.0)
    parser.add_argument('--width', type=int, default=960)
    parser.add_argument('--height', type=int, default=576)
    parser.add_argument('--seed', type=int, default=1101)
    parser.add_argument('--warmup', type=int, default=1)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--timeout', type=float, default=14400)
    args = parser.parse_args()
    if min(args.steps, args.duration, args.width, args.height, args.repeats, args.timeout) <= 0 or args.warmup < 0:
        parser.error('Positive dimensions/counts/timeouts and nonnegative warmup are required')
    if args.reference and not args.reference.is_file():
        parser.error('--reference must be an existing file')
    args.output.mkdir(parents=True, exist_ok=False)
    fields = {'prompt': args.prompt, 'width': args.width, 'height': args.height,
              'aspect_ratio': '16:9', 'fps': 24, 'num_inference_steps': args.steps,
              'flow_shift': 12, 'seed': args.seed,
              'extra_params': json.dumps({'task': 'ref2va' if args.reference else 't2va',
                                           'duration': args.duration, 'audio_flow_shift': 3.0})}
    payload, content_type = multipart(fields, args.reference)
    report = {'request': fields, 'warmup': args.warmup, 'samples': [],
              'timing': 'HTTP POST to complete response bytes saved; ffprobe excluded',
              'server_configuration': 'Record the spark.py launch JSON alongside this report.'}
    report_file = args.output / 'results.json'
    report_file.write_text(json.dumps(report, indent=2) + '\n')
    for i in range(args.warmup + args.repeats):
        path = args.output / f'{i:03d}.mp4'
        request = urllib.request.Request(args.url, data=payload, headers={'Content-Type': content_type})
        start = time.perf_counter()
        with urllib.request.urlopen(request, timeout=args.timeout) as response, path.open('xb') as file:
            shutil.copyfileobj(response, file)
            headers = {k: v for k, v in response.headers.items()
                       if k.lower() in {'content-type', 'x-stage-durations', 'x-peak-memory-mb'}}
        elapsed = time.perf_counter() - start
        sample = {'index': i, 'warmup': i < args.warmup, 'e2e_seconds': elapsed,
                  'bytes': path.stat().st_size, 'headers': headers,
                  'ffprobe_available': shutil.which('ffprobe') is not None}
        report['samples'].append(sample)
        # Save timing even if media inspection fails; never discard slow samples.
        report_file.write_text(json.dumps(report, indent=2) + '\n')
        if sample['ffprobe_available']:
            result = subprocess.run(['ffprobe', '-v', 'error', '-show_streams', '-show_format',
                                     '-of', 'json', str(path)], capture_output=True, text=True, check=True)
            (args.output / f'{i:03d}.ffprobe.json').write_text(result.stdout)
        print(json.dumps(sample), flush=True)
    timings = [row['e2e_seconds'] for row in report['samples'] if not row['warmup']]
    report['summary'] = {'n': len(timings), 'median_seconds': statistics.median(timings),
                         'min_seconds': min(timings), 'max_seconds': max(timings)}
    report_file.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
