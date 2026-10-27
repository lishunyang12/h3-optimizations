#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Build the English report for the two pinned historical reproductions."""
import argparse
import csv
import html
import json
import re
import shutil
import statistics
import subprocess
from pathlib import Path
from sku_benchmark import ROOT, CATALOG, get_case, load, save


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--site', type=Path, required=True)
    args = parser.parse_args()
    site = args.site
    assets = site / 'benchmarks/sku-reproduction'
    assets.mkdir(parents=True, exist_ok=True)
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    rows = []
    for name in load(CATALOG)['cases']:
        case = get_case(name)
        history = case['historical']
        samples = history.get('http_samples_s', history.get('samples_s', []))
        rows.append({
            'id': name, 'family': case['family'], 'gpu_count': case['topology']['num_gpus'],
            'gpu': 'RTX PRO 5000-class Blackwell (historical project designation)',
            'gpu_label_scope': 'Project records; some driver inventories report NVIDIA Graphics Device',
            **{key: case['media'][key] for key in ('width', 'height', 'frames', 'duration_seconds')},
            'steps': 4, 'samples_s': samples, 'median_s': statistics.median(samples),
            'mean_s': statistics.mean(samples), 'n': len(samples), 'historical': history,
            'request': case['request'], 'protocol': case['protocol'],
            'reproduction_status': 'packaged; fresh GPU rerun pending', 'source_commit': commit,
        })
    assert [r['id'] for r in rows] == ['5s-avatar-ref2va-4step', '15s-4step-mxfp8-sage-nvfp4']
    data = {
        'schema': 1, 'publication_date': '2026-10-27',
        'source_repository': 'https://github.com/lishunyang12/h3-optimizations',
        'source_commit': commit,
        'status': 'Historical measurements; new launcher CPU/source/log checks passed; no fresh GPU measurements',
        'target_sku_status': '5500 exact hardware and results pending', 'cases': rows,
    }
    save(assets / 'results.json', data)
    with (assets / 'timings.csv').open('w', newline='') as stream:
        writer = csv.writer(stream, lineterminator='\n')
        writer.writerow(['case', 'gpu_count', 'gpu', 'sample', 'historical_seconds', 'boundary'])
        for row in rows:
            for index, seconds in enumerate(row['samples_s'], 1):
                writer.writerow([row['id'], row['gpu_count'], row['gpu'], index, seconds,
                                 'historical HTTP; see case protocol'])
            for index, seconds in enumerate(row['historical'].get('ready_samples_s', []), 1):
                writer.writerow([row['id'], row['gpu_count'], row['gpu'], index, seconds,
                                 'avatar application ready, includes checks'])
    for source, target in {
        'evidence/sku-avatar5-history.json': 'avatar5-history.json',
        'evidence/sku-15s-history.json': 'fasth3-15s-history.json',
        'evidence/sku-evidence-audit.json': 'evidence-audit.json',
        'evidence/sku-package-validation.json': 'validation.json',
        'docs/VERIFIED_CONTEXT.md': 'verified-context.md',
    }.items():
        shutil.copy2(ROOT / source, assets / target)
    body = []
    for row in rows:
        body.append('<tr><th scope="row"><code>' + html.escape(row['id']) + '</code></th>'
                    f"<td>{row['gpu_count']}</td><td>{row['width']} × {row['height']} / {row['frames']}f</td>"
                    f"<td>4</td><td>{row['median_s']:.6f}</td><td>{row['n']}</td><td>Pending</td></tr>")
    chart = {'cases': [{key: row[key] for key in
                       ('id', 'family', 'gpu_count', 'duration_seconds', 'samples_s', 'median_s', 'n')}
                      for row in rows]}
    payload = json.dumps(chart, ensure_ascii=True).replace('<', '\\u003c')
    url = data['source_repository']
    replacements = {
        'ROWS': ''.join(body), 'DATA': payload, 'AVATAR_MEDIAN': f"{rows[0]['median_s']:.3f}",
        'FASTH3_MEDIAN': f"{rows[1]['median_s']:.3f}", 'COMMIT': commit, 'SHORT_COMMIT': commit[:12],
        'COMMIT_URL': url + '/commit/' + commit,
        'GUIDE': url + '/blob/' + commit + '/docs/CROSS_SKU_REPRO.md',
        'CONTEXT': url + '/blob/' + commit + '/docs/VERIFIED_CONTEXT.md',
        'SOURCE': url + '/blob/' + commit + '/scripts/sku_benchmark.py',
        'CASES': url + '/blob/' + commit + '/configs/sku-cases.json',
    }
    doc = (ROOT / 'scripts/sku_report.html').read_text()
    for key, value in replacements.items():
        doc = doc.replace('__' + key + '__', value)
    assert not re.search(r'__[A-Z_]+__', doc), 'Unresolved report placeholder'
    assert not re.search(r'[\u4e00-\u9fff]', doc), 'Report prose must be English'
    (site / 'sku-reproduction.html').write_text(doc)
    index = site / 'index.html'
    if index.exists():
        text = index.read_text()
        link = '<a href="sku-reproduction.html">5s / 15s Reproduction &amp; Data</a>'
        if 'href="sku-reproduction.html"' in text:
            text = re.sub(r'<a href="sku-reproduction.html">.*?</a>', lambda _: link, text, count=1)
        else:
            needle = '<div class="nav-links">'
            assert needle in text
            text = text.replace(needle, needle + link, 1)
        index.write_text(text)
    print(json.dumps({'cases': len(rows), 'site': str(site), 'source_commit': commit}))


if __name__ == '__main__':
    main()
