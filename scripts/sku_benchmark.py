#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Pinned H3 cases: prepare, plan, run, and compare across GPU SKUs.

Only `run` starts a model. This launcher never changes clocks or stops foreign jobs.
Python standard library only; the selected model environment supplies vLLM.
"""
import argparse
import collections
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import signal
import socket
import statistics
import subprocess
import sys
import threading
import time
from urllib.request import urlopen
from urllib.error import URLError

ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / 'configs/sku-cases.json'


def load(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True,
                                    separators=(',', ':')).encode()).hexdigest()


def command(argv, **kwargs):
    return subprocess.check_output([str(x) for x in argv], text=True, **kwargs).strip()


def tree_identity(root):
    root = Path(root)
    if not root.is_dir():
        raise ValueError(f'Missing source directory: {root}')
    suffixes = {'.py', '.cu', '.cuh', '.c', '.cc', '.cpp', '.h', '.hpp', '.json', '.toml', '.yaml', '.yml', '.so'}
    files = {}
    for current, dirs, names in os.walk(root):
        dirs[:] = sorted(d for d in dirs if d not in
                         {'.git', '.cache', '__pycache__', '.venv', 'build', 'dist'})
        for name in sorted(names):
            p = Path(current) / name
            if p.suffix in suffixes and name != 'sku-source.json':
                files[str(p.relative_to(root))] = digest(p)
    if not files:
        raise ValueError(f'No source files found: {root}')
    return {'sha256': identity(files), 'files': files}


def asset_identity(path):
    """Hash actual checkpoint bytes, not an unverified model name or revision label."""
    p = Path(path)
    if p.is_file():
        return {'sha256': digest(p), 'bytes': p.stat().st_size}
    if not p.is_dir():
        raise ValueError(f'Missing asset: {p}')
    files = {}
    for current, dirs, names in os.walk(p, followlinks=True):
        dirs[:] = sorted(d for d in dirs if not d.startswith('.'))
        for name in sorted(names):
            f = Path(current) / name
            if name.startswith('.') or f.suffix in {'.lock', '.incomplete', '.log', '.md'}:
                continue
            files[str(f.relative_to(p))] = {'sha256': digest(f), 'bytes': f.stat().st_size}
    if not files or not any(x.endswith(('.safetensors', '.bin', '.pt')) for x in files):
        raise ValueError(f'No checkpoint weights found: {p}')
    return {'sha256': identity(files), 'files': files}


def get_case(name):
    cases = load(CATALOG)['cases']
    if name not in cases:
        raise ValueError(f'Unknown case {name}; use list')
    case = dict(cases[name])
    if 'profile_file' in case:
        case = {**load(ROOT / case.pop('profile_file')), **case}
    if 'requests_file' in case:
        case['requests'] = load(ROOT / case['requests_file'])
    case['environment'] = {**case['environment'], **case.get('environment_overrides', {})}
    return case


def prepare(destination, variant='reference'):
    """Reconstruct the published 15s runtime; change only the CAKE source location."""
    destination = Path(destination).resolve()
    if destination.exists():
        raise ValueError('Prepare requires a new directory; refusing to overwrite sources')
    profile = load(ROOT / 'configs/fasth3-15s-4step-reference.json')
    patch = ROOT / 'patches/ref2va-to-full-length-runtime.patch'
    if digest(patch) != profile['patch_sha256']:
        raise ValueError('Historical runtime patch changed')
    # Verify all released source bytes before reconstructing the reference.
    expected = load(ROOT / 'evidence/source-sha256.json')
    for rel, checksum in expected.items():
        if digest(ROOT / rel) != checksum:
            raise ValueError(f'Released source differs: {rel}')
    shutil.copytree(ROOT / 'reference-runtime/vllm-omni', destination,
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    if variant in ('reference', 'output-fusion'):
        subprocess.run(['git', 'apply', '--check', str(patch)], cwd=destination, check=True)
        subprocess.run(['git', 'apply', str(patch)], cwd=destination, check=True)
    if variant != 'reference':
        overlay = load(ROOT / 'repro/variants.json')[variant]
        for rel, checksum in overlay['files'].items():
            source = ROOT / overlay['overlay'] / rel
            if digest(source) != checksum:
                raise ValueError(f'Historical overlay changed: {rel}')
            (destination / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination / rel)
    rel = 'vllm_omni/diffusion/attention/ops/h3_cake_grid_constant.py'
    shutil.copy2(ROOT / 'reference-runtime/vllm-omni' / rel, destination / rel)
    save(destination / 'sku-source.json', {
        'kind': 'historical-reconstructed', 'variant': variant, 'base_commit': profile['snapshot_commit'],
        'patch_sha256': digest(patch), 'relocation': rel,
        'source': tree_identity(destination)})
    print(f'Prepared {destination}')


def resolve_case(case, assets, out, gpus, port):
    """Resolve machine paths separately; hardware paths never enter the workload identity."""
    paths = {k: str(Path(v).expanduser().resolve()) for k, v in assets.items()}
    if case.get('reference_asset'):
        paths.setdefault('reference', str(ROOT / case['reference_asset']))
    required = set(case.get('required_assets', ['runtime', 'python', 'model', 'adapter', 'adaln', 'cuda']))
    missing = sorted(required - paths.keys())
    if missing:
        raise ValueError('Missing assets: ' + ', '.join(missing))
    paths.update(repo=str(ROOT), out=str(out), port=str(port))
    if case['family'] in ('legacy15', 'avatar5', 't2va5'):
        paths.setdefault('flashinfer', str(ROOT / 'flashinfer'))
        paths.setdefault('fastvideo', str(ROOT / 'fastvideo-kernel/python'))
    def substitute(value):
        if isinstance(value, str):
            for k, v in paths.items():
                value = value.replace('{' + k + '}', v)
            if re.search(r'\{[a-z_]+\}', value):
                raise ValueError(f'Unresolved path placeholder: {value}')
        elif isinstance(value, dict):
            return {k: substitute(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [substitute(v) for v in value]
        return value
    # Avoid leaking arbitrary inherited tuning flags into either SKU's measurement.
    env = {k: v for k, v in os.environ.items() if k in
           {'PATH', 'HOME', 'USER', 'LANG', 'LC_ALL', 'SYSTEMROOT', 'TMPDIR'}}
    env.update(substitute(case['environment']))
    env.update(CUDA_VISIBLE_DEVICES=gpus, PYTHONDONTWRITEBYTECODE='1')
    import_roots = [paths['runtime']]
    for k in ('flashinfer', 'fastvideo', 'kitchen'):
        if k in paths:
            import_roots.append(paths[k])
    env['PYTHONPATH'] = os.pathsep.join(import_roots)
    for k, dirname in {'HF_MODULES_CACHE': 'hf-modules', 'XDG_CACHE_HOME': 'xdg',
                       'TRITON_CACHE_DIR': 'triton', 'TORCHINDUCTOR_CACHE_DIR': 'inductor',
                       'VLLM_CACHE_ROOT': 'vllm', 'CUDA_CACHE_PATH': 'cuda',
                       'CUTE_DSL_CACHE_DIR': 'cute', 'FLASHINFER_WORKSPACE_BASE': 'flashinfer'}.items():
        env[k] = str(out / 'cache' / dirname)
    if 'cuda' in paths:
        env.update(CUDA_HOME=paths['cuda'], FLASHINFER_NVCC=paths['cuda'] + '/bin/nvcc',
                   FLASHINFER_EXTRA_LDFLAGS='-L' + paths['cuda'] + '/lib',
                   LD_LIBRARY_PATH=paths['cuda'] + '/lib:' + paths['cuda'] + '/lib64')
        env['PATH'] = paths['cuda'] + '/bin:' + env['PATH']
    server = [paths['python'], str(Path(paths['python']).parent / 'vllm')]
    if case['family'] in ('legacy15', 't2va5'):
        server = ['numactl', '--interleave=all'] + server
    server += substitute(case['server_args'])
    config = substitute(case.get('server_config'))
    return {'paths': paths, 'command': server, 'environment': env, 'server_config': config}


def gpu_snapshot():
    fields = ['index', 'uuid', 'name', 'pci.bus_id', 'compute_cap', 'memory.total',
              'memory.used', 'utilization.gpu', 'temperature.gpu', 'power.limit',
              'clocks.sm', 'clocks.mem', 'driver_version']
    raw = command(['nvidia-smi', '--query-gpu=' + ','.join(fields), '--format=csv,noheader,nounits'])
    return [dict(zip(fields, (x.strip() for x in row))) for row in csv.reader(raw.splitlines())]


def select_gpus(snapshot, selection, expected):
    values = selection.split(',')
    if len(values) != expected or len(set(values)) != expected:
        raise ValueError(f'This case requires exactly {expected} different GPUs')
    selected = []
    for value in values:
        matches = [g for g in snapshot if value in (g['index'], g['uuid'])]
        if len(matches) != 1:
            raise ValueError(f'GPU not found or ambiguous: {value}')
        selected += matches
    if len({g['name'] for g in selected}) != 1:
        raise ValueError('Mixed GPU SKUs in one run are not a SKU comparison')
    return selected


def foreign_processes(uuids, server_pid=None):
    raw = command(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader,nounits'])
    foreign = []
    for row in csv.reader(raw.splitlines()):
        if len(row) != 2 or row[0].strip() not in uuids:
            continue
        pid = int(row[1])
        try:
            own = server_pid is not None and os.getpgid(pid) == server_pid
        except ProcessLookupError:
            continue
        if not own:
            foreign.append({'gpu_uuid': row[0].strip(), 'pid': pid})
    return foreign


def runtime_versions(python, env, packages):
    program = ('import json,sys,importlib.metadata as m; '\
               'print(json.dumps({p:m.version(p) for p in json.loads(sys.argv[1])}))')
    return json.loads(command([python, '-c', program, json.dumps(packages)], env=env))


def records(text, marker):
    decoder = json.JSONDecoder()
    result = []
    for m in re.finditer(re.escape(marker) + r'\s+', text):
        try:
            result.append(decoder.raw_decode(text, m.end())[0])
        except (ValueError, TypeError) as exc:
            raise ValueError(f'Corrupted runtime marker {marker}') from exc
    return result


def audit_legacy_request(text, request_index, case):
    expected = set(range(8))
    avatar = case['family'] == 'avatar5'
    t2va5 = case['family'] == 't2va5'
    nfe = records(text, 'REF2VA_MXFP8_NFE' if avatar else 'H3_EXPERIMENT_QUANT_NFE')
    if t2va5:
        nfe = records(text, 'H3_PRECISION_SCHEDULE_NFE')
    for rank in expected:
        rr = [r for r in nfe if r['rank'] == rank]
        wanted = list(range(1, 5)) if t2va5 else list(range(request_index * 4 + 1, request_index * 4 + 5))
        if [r['step'] if t2va5 else r['forwards'] for r in rr] != wanted:
            raise ValueError(f'Expected four real forwards on rank {rank}')
        if avatar:
            ok = all(r['modules'] == 200 and r['fallback'] == 0 and
                     r['dense_attention'] and r['bf16_projection_calls'] == 0 and
                     r['mxfp8_projection_calls'] == r['forwards'] * 200 for r in rr)
        elif t2va5:
            ok = True
            for r in rr:
                bf = case['environment']['VLLM_OMNI_H3_PRECISION_SCHEDULE'] == 'first_step_bf16' and r['step'] == 1
                ok &= (r['status'] == 'PASS' and r['fallback_count'] == 0 and r['modules'] == 200 and
                       r['attention_calls'] == ({'bf16': 50, 'pr4951': 0} if bf else {'bf16': 0, 'pr4951': 50}) and
                       r['projection_calls'] == ({'bf16': 350, 'mxfp8': 0} if bf else {'bf16': 0, 'mxfp8': 350}))
        else:
            ok = all(r['mode'] == 'mxfp8' and r['modules'] == 200 and
                     r['fallback_count'] == 0 and r['wire'] == 'bfloat16' and r['bf16_gate'] for r in rr)
        if not ok:
            raise ValueError('MXFP8 dispatch or transport changed')
    vae = records(text, 'H3_VAE_NVFP4_COMPLETE')
    if collections.Counter(r['rank'] for r in vae) != collections.Counter(expected):
        raise ValueError('Missing full NVFP4 VAE completion records')
    if not all(r['layers'] == 144 and r['fallback'] == 0 and r['status'] == 'PASS' for r in vae):
        raise ValueError('NVFP4 VAE fell back or changed')
    pairs = records(text, 'H3_VAE_PAIR_COMPLETE')
    frames = 124 if avatar else case['media']['frames']
    if collections.Counter(r['rank'] for r in pairs) != collections.Counter(expected) or not all(r['frames'] == frames for r in pairs):
        raise ValueError('VAE geometry or gather schedule changed')
    if case['family'] == 'legacy15' and any(r['gathers'] != 21 for r in pairs):
        raise ValueError('15s gather schedule changed')
    return {'actual_forwards_per_rank': 4, 'ranks': 8, 'quantization_fallbacks': 0}


def validate_media(path, expected, ffprobe='ffprobe', ffmpeg='ffmpeg'):
    probe = json.loads(command([ffprobe, '-v', 'error', '-count_frames', '-show_streams',
                                '-show_format', '-of', 'json', path]))
    v = next(s for s in probe['streams'] if s['codec_type'] == 'video')
    a = next(s for s in probe['streams'] if s['codec_type'] == 'audio')
    actual = {'width': v['width'], 'height': v['height'], 'frames': int(v['nb_read_frames']),
              'fps': v['r_frame_rate'], 'video_codec': v['codec_name'],
              'audio_codec': a['codec_name'], 'sample_rate': int(a['sample_rate']),
              'channels': a['channels']}
    for k, value in actual.items():
        if k in expected and value != expected[k]:
            raise ValueError(f'Media mismatch {k}: {value} != {expected[k]}')
    if abs(float(v['duration']) - actual['frames'] / 24) > 0.05 or abs(float(a['duration']) - float(v['duration'])) > 0.05:
        raise ValueError('Audio/video duration differs from the recorded frame contract')
    subprocess.run([ffmpeg, '-v', 'error', '-xerror', '-threads', '2', '-i', str(path),
                    '-map', '0:v:0', '-map', '0:a:0', '-f', 'null', '-'], check=True)
    return {**actual, 'sha256': digest(path), 'bytes': Path(path).stat().st_size,
            'full_decode': 'pass'}


def one_request(case, url, path, resolved):
    argv = ['curl', '--noproxy', '*', '--fail-with-body', '--silent', '--show-error', '--max-time', '7200',
            '-X', 'POST', url + '/v1/videos/sync']
    for k, v in case['request'].items():
        argv += ['--form-string', f'{k}={v}']
    if 'reference' in resolved['paths']:
        for _ in range(case.get('reference_copies', 1)):
            argv += ['-F', case.get('reference_form_field', 'input_reference') + '=@' + resolved['paths']['reference']]
    argv += ['-D', str(path.with_suffix('.headers')), '-o', str(path), '-w', '%{time_total}']
    started_at = time.time()
    start = time.perf_counter()
    response = subprocess.run(argv, capture_output=True, text=True)
    elapsed = time.perf_counter() - start
    if response.returncode:
        raise RuntimeError(f'HTTP request failed: {response.stderr[-2000:]}')
    headers = {}
    for line in path.with_suffix('.headers').read_text().splitlines():
        if ':' in line:
            k, v = line.split(':', 1)
            headers[k.lower()] = v.strip()
    return {'started_at': started_at, 'ended_at': time.time(),
            'client_seconds': elapsed, 'curl_seconds': float(response.stdout),
            'server_seconds': float(headers['x-inference-time-s']) if 'x-inference-time-s' in headers else None,
            'stage_durations_raw': json.loads(headers.get('x-stage-durations', '{}'))}


def summary(samples):
    values = [r['client_seconds'] for r in samples if r['phase'] == 'measured']
    if len(values) < 3:
        raise ValueError('At least three successful measured requests required')
    return {'n': len(values), 'samples_s': values, 'median_s': statistics.median(values),
            'mean_s': statistics.mean(values), 'min_s': min(values), 'max_s': max(values)}


def run_case(args):
    case = get_case(args.case)
    if case.get('availability') != 'runnable':
        raise ValueError(case.get('unavailable_reason', 'This case is evidence only'))
    out = Path(args.output).resolve()
    if out.exists():
        raise ValueError('Output directory already exists; use a fresh run directory')
    assets = load(args.assets)
    assets = assets.get(args.case, assets)
    resolved = resolve_case(case, assets, out, args.gpus, args.port)
    if args.dry_run:
        print(json.dumps({'case': args.case, 'command': resolved['command'],
                          'server_config': resolved['server_config'], 'environment': resolved['environment'],
                          'required_assets': case.get('required_assets'), 'request': case['request']}, indent=2))
        return
    selected = select_gpus(gpu_snapshot(), args.gpus, case['topology']['num_gpus'])
    uuids = {g['uuid'] for g in selected}
    resolved['environment']['CUDA_VISIBLE_DEVICES'] = ','.join(g['uuid'] for g in selected)
    if any(g['compute_cap'] != case['topology']['compute_capability'] for g in selected):
        raise ValueError('This case needs SM120; port/qualify another architecture before comparing')
    minimum = case['protocol'].get('minimum_memory_mib', 0)
    if any(float(g['memory.total']) < minimum for g in selected):
        raise ValueError('GPU memory is below the historical requirement; no automatic offload')
    host_memory = int(re.search(r'MemAvailable:\s+(\d+)', Path('/proc/meminfo').read_text()).group(1)) / 2**20
    if host_memory < case['protocol'].get('minimum_host_memory_gib', 0):
        raise ValueError('Insufficient available host RAM for the recorded pinned BF16 weights')
    if foreign_processes(uuids) or any(float(g['utilization.gpu']) > 5 or
                                     float(g['memory.used']) > 256 for g in selected):
        raise RuntimeError('Selected GPUs are occupied; no workload started')
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', args.port))
    out.mkdir(parents=True)
    report = {'schema': 1, 'case': args.case, 'status': 'preflight', 'requests': [],
              'case_identity': identity(case), 'gpu_before': selected,
              'host': {'platform': platform.platform(), 'cpu_count': os.cpu_count(),
                       'lscpu': command(['lscpu']), 'topology': command(['nvidia-smi', 'topo', '-m'])},
              'configuration': case, 'started_at': time.time(),
              'command': resolved['command']}
    save(out / 'result.json', report)
    server = None
    stopped = threading.Event()
    monitor_errors = []
    try:
        paths, env = resolved['paths'], resolved['environment']
        for tool in ('curl', 'ffmpeg', 'ffprobe'):
            if not shutil.which(tool):
                raise ValueError(f'Missing tool: {tool}')
        versions = runtime_versions(paths['python'], env, list(case['dependencies']))
        for name, expected in case['dependencies'].items():
            if versions[name] != expected:
                raise ValueError(f'{name}={versions[name]}, expected {expected}; use the pinned environment')
        report['software'] = versions
        if case.get('expected_runtime_commit'):
            revision = command(['git', '-C', paths['runtime'], 'rev-parse', 'HEAD'])
            if revision != case['expected_runtime_commit']:
                raise ValueError('Runtime git revision differs from the recorded case')
        if case.get('runtime_variant'):
            marker = load(Path(paths['runtime']) / 'sku-source.json')
            if marker['variant'] != case['runtime_variant'] or marker['source'] != tree_identity(paths['runtime']):
                raise ValueError('Prepared runtime variant or source hashes differ; run prepare again')
        sources = {k: tree_identity(paths[k]) for k in ('runtime', 'flashinfer', 'fastvideo', 'kitchen') if k in paths}
        report['source_identity'] = identity(sources)
        save(out / 'sources.json', sources)
        print('Hashing checkpoints and inputs before model startup...', flush=True)
        fingerprints = {k: asset_identity(paths[k]) for k in
                        ('model', 'adapter', 'adaln', 'decoder', 'reference') if k in paths}
        if 'adapter' in fingerprints and fingerprints['adapter']['sha256'] != case['model']['adapter_sha256']:
            raise ValueError('Adapter differs from the historical 4-step checkpoint')
        if case.get('reference_sha256') and fingerprints['reference']['sha256'] != case['reference_sha256']:
            raise ValueError('Reference image differs from the recorded case')
        for relative, key in [('model_index.json', 'model_index_sha256'), ('transformer/config.json', 'transformer_config_sha256')]:
            expected = case.get('model', {}).get(key)
            if expected and digest(Path(paths['model']) / relative) != expected:
                raise ValueError(f'Model metadata differs: {relative}')
        report['asset_identity'] = identity(fingerprints)
        save(out / 'assets.json', fingerprints)
        # Re-check after slow fingerprinting, immediately before taking GPU resources.
        if foreign_processes(uuids):
            raise RuntimeError('GPU became occupied during preparation')
        if resolved['server_config'] is not None:
            save(out / 'server-config.json', resolved['server_config'])  # JSON is valid YAML.
        for k, value in env.items():
            if k.endswith(('_DIR', '_HOME', '_PATH', '_ROOT', '_BASE', '_CACHE')) and str(out / 'cache') in value:
                Path(value).mkdir(parents=True, exist_ok=True)
        save(out / 'environment.json', env)
        if case['family'] == 'avatar5':
            for key in ('H3_QUALITY_LATENT_DIR', 'H3_QUALITY_TAIL_REPLAY_DIR', 'H3_QUALITY_PCM_DIR', 'VLLM_OMNI_OUTPUT_DIR'):
                Path(env[key]).mkdir(parents=True, exist_ok=True)
        log_path = out / 'server.log'
        startup_started = time.monotonic()
        with log_path.open('w') as log:
            server = subprocess.Popen(resolved['command'], cwd=paths['runtime'], env=env,
                                      stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        def watch():
            with (out / 'telemetry.jsonl').open('w') as stream:
                while not stopped.is_set():
                    try:
                        foreign = foreign_processes(uuids, server.pid)
                        if foreign:
                            monitor_errors.append({'foreign_processes': foreign})
                        sample = {'time': time.time(), 'gpus': [g for g in gpu_snapshot() if g['uuid'] in uuids]}
                        stream.write(json.dumps(sample) + '\n'); stream.flush()
                    except Exception as exc:
                        monitor_errors.append({'monitor_error': str(exc)})
                    stopped.wait(0.5)
        watcher = threading.Thread(target=watch, daemon=True)
        watcher.start()
        base = f'http://127.0.0.1:{args.port}'
        deadline = time.monotonic() + 7200
        while True:
            if server.poll() is not None:
                raise RuntimeError(f'Server exited {server.returncode}; see server.log')
            if monitor_errors:
                raise RuntimeError(f'Telemetry/ownership check failed: {monitor_errors[:2]}')
            try:
                with urlopen(base + '/health', timeout=2) as response:
                    if response.status == 200:
                        break
            except (URLError, TimeoutError):
                pass
            if time.monotonic() > deadline:
                raise TimeoutError('Server startup timed out')
            time.sleep(2)
        report['cold_start_seconds'] = time.monotonic() - startup_started
        protocol = case['protocol']
        for i in range(protocol['warmups'] + protocol['repeats']):
            phase = 'warmup' if i < protocol['warmups'] else 'measured'
            if monitor_errors or foreign_processes(uuids, server.pid):
                raise RuntimeError('Foreign GPU activity detected; refusing measurement')
            offset = log_path.stat().st_size
            output = out / f'{i:02d}-{phase}.mp4'
            request_case = {**case, 'request': case['requests'][i]} if 'requests' in case else case
            save(out / f'{i:02d}-{phase}-request.json', request_case['request'])
            row = one_request(request_case, base, output, resolved)
            row.update(phase=phase, output=output.name)
            media = dict(case['media'])
            if i == 0 and case.get('first_media_frames'):
                media['frames'] = case['first_media_frames']
            row['media'] = validate_media(output, media)
            time.sleep(1)  # Same inter-request cooldown as the historical 15s runner.
            with log_path.open('rb') as stream:
                stream.seek(offset)
                request_log = stream.read().decode(errors='replace')
            (out / f'{i:02d}-{phase}.log').write_text(request_log)
            if case['family'] in ('legacy15', 'avatar5', 't2va5'):
                row['dispatch'] = audit_legacy_request(request_log, i, case)
                if case.get('runtime_variant') == 'output-fusion':
                    for marker, calls in [('H3_VAE_FUSED_BLEND', 965), ('H3_VAE_FUSED_RGB', 22)]:
                        values = records(request_log, marker)
                        if len(values) != 1 or values[0]['status'] != 'PASS' or values[0]['calls'] != calls:
                            raise ValueError('Output fusion dispatch evidence missing')
                if i == 0:
                    full_log = log_path.read_text(errors='replace')
                    if len(records(full_log, 'H3_CAKE_GRID_CONSTANT_READY')) != 8:
                        raise ValueError('Sage/CAKE startup evidence missing')
                    armed = re.findall(r'FlashInfer PCIe Ulysses armed:.*', full_log)
                    if len(armed) != 8 or any('transport=rdma' not in x for x in armed):
                        raise ValueError('Expected eight all-RDMA communicators')
            if monitor_errors:
                raise RuntimeError('GPU ownership/telemetry failed during request')
            report['requests'].append(row)
            save(out / 'result.json', report)
            print(json.dumps({'phase': phase, 'seconds': row['client_seconds']}), flush=True)
        if identity({k: tree_identity(paths[k]) for k in sources}) != report['source_identity']:
            raise RuntimeError('Source files changed during the run')
        report.update(status='complete', summary=summary(report['requests']))
        report['summary']['rtf'] = report['summary']['median_s'] / case['media']['duration_seconds']
        samples = [json.loads(line) for line in (out / 'telemetry.jsonl').read_text().splitlines()]
        measured = [r for r in report['requests'] if r['phase'] == 'measured']
        active = [s for s in samples if any(r['started_at'] <= s['time'] <= r['ended_at'] for r in measured)]
        if not active:
            raise RuntimeError('No telemetry samples during measured requests')
        report['measured_peak_gpu_memory_mib'] = {
            u: max(float(g['memory.used']) for s in active for g in s['gpus'] if g['uuid'] == u) for u in sorted(uuids)}
    except BaseException as exc:
        report.update(status='failed', error=str(exc))
        raise
    finally:
        stopped.set()
        if 'watcher' in locals():
            watcher.join(timeout=10)
        # Only the process group created by this invocation is eligible for cleanup.
        if server is not None:
            try:
                os.killpg(server.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                server.wait(timeout=20)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(server.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                server.wait(timeout=10)
        report.update(ended_at=time.time(), telemetry_errors=monitor_errors)
        if monitor_errors:
            report['status'] = 'failed'
        save(out / 'result.json', report)
    if report['status'] != 'complete':
        raise RuntimeError('Run failed ownership/telemetry checks; inspect result.json')


def compare(left, right):
    for result in (left, right):
        if result.get('status') != 'complete' or result.get('telemetry_errors'):
            raise ValueError('Only complete, uncontaminated runs are comparable')
    keys = ('case_identity', 'asset_identity', 'source_identity', 'software')
    mismatches = [k for k in keys if left.get(k) != right.get(k) or not left.get(k)]
    if mismatches:
        raise ValueError('Not apples-to-apples: mismatched ' + ', '.join(mismatches))
    a, b = summary(left['requests']), summary(right['requests'])
    return {'case': left['case'], 'status': 'same_workload_and_software',
            'reference': a, 'candidate': b, 'speedup_reference_over_candidate': a['median_s'] / b['median_s'],
            'reference_gpus': left['gpu_before'], 'candidate_gpus': right['gpu_before'],
            'host_differs': left.get('host') != right.get('host'),
            'scope': 'System latency under the same workload; CPU, PCIe/RDMA, clocks and driver can differ.',
            'output_bytes_equal': [r['media']['sha256'] for r in left['requests'] if r['phase'] == 'measured'] ==
                                  [r['media']['sha256'] for r in right['requests'] if r['phase'] == 'measured']}


def run_suite(args):
    names = args.cases.split(',')
    if names == ['main']:
        names = ['5s-avatar-ref2va-4step', '15s-4step-mxfp8-sage-nvfp4', '15s-4step-output-fusion']
    out = Path(args.output).resolve()
    if out.exists():
        raise ValueError('Suite requires a new output directory')
    cases = [get_case(name) for name in names]
    if any(c['topology']['num_gpus'] != len(args.gpus.split(',')) for c in cases):
        raise ValueError('Suite GPU count must match every selected case; run the 4-GPU cases separately')
    for name in names:
        item = argparse.Namespace(**vars(args))
        item.case, item.output = name, str(out / name)
        run_case(item)
    if not args.dry_run:
        save(out / 'suite.json', {'status': 'complete', 'cases': names})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='action', required=True)
    sub.add_parser('list')
    prep = sub.add_parser('prepare', aliases=['prepare-15s']); prep.add_argument('--output', required=True)
    prep.add_argument('--variant', choices=['reference', 'output-fusion', 'avatar5', 't2va5'], default='reference')
    run = sub.add_parser('run')
    run.add_argument('--case', required=True); run.add_argument('--assets', required=True)
    run.add_argument('--gpus', required=True); run.add_argument('--output', required=True)
    run.add_argument('--port', type=int, default=18126); run.add_argument('--dry-run', action='store_true')
    suite = sub.add_parser('suite')
    suite.add_argument('--cases', default='main'); suite.add_argument('--assets', required=True)
    suite.add_argument('--gpus', required=True); suite.add_argument('--output', required=True)
    suite.add_argument('--port', type=int, default=18126); suite.add_argument('--dry-run', action='store_true')
    comp = sub.add_parser('compare')
    comp.add_argument('reference'); comp.add_argument('candidate'); comp.add_argument('--output')
    args = p.parse_args()
    if args.action == 'list':
        for name, case in load(CATALOG)['cases'].items():
            print(f"{name:34} {case['availability']:12} {case.get('description', '')}")
    elif args.action in ('prepare', 'prepare-15s'):
        prepare(args.output, args.variant)
    elif args.action == 'run':
        run_case(args)
    elif args.action == 'suite':
        run_suite(args)
    else:
        result = compare(load(args.reference), load(args.candidate))
        if args.output:
            save(args.output, result)
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    try:
        main()
    except (ValueError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(2)
