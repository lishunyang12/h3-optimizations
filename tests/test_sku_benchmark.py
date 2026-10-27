"""CPU checks for provenance, dispatch auditing and comparison boundaries."""
import copy
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import sku_benchmark as b


class ReproductionTests(unittest.TestCase):
    def test_recorded_dispatch_receipts(self):
        for fixture, name, index in [
            ('15s', '15s-4step-mxfp8-sage-nvfp4', 1),
            ('avatar5', '5s-avatar-ref2va-4step', 0),
            ('t2va5', '5s-t2va-4step-all-mxfp8', 2),
            ('t2va5-bf16', '5s-t2va-4step-first-step-bf16', 2),
        ]:
            with self.subTest(case=name):
                text = (Path(__file__).parent / 'sku-fixtures' / (fixture + '.log')).read_text()
                self.assertEqual(b.audit_legacy_request(text, index, b.get_case(name))['ranks'], 8)
                with self.assertRaises(ValueError):
                    b.audit_legacy_request(text.replace('"modules": 200', '"modules": 199', 1), index, b.get_case(name))
                with self.assertRaises(ValueError):
                    b.audit_legacy_request(text.replace('H3_VAE_NVFP4_COMPLETE', 'MISSING', 1), index, b.get_case(name))

    def test_all_profiles_resolve_without_host_paths_or_inherited_tuning(self):
        assets = b.load(b.ROOT / 'configs/sku-assets.example.json')
        for name in b.load(b.CATALOG)['cases']:
            c = b.get_case(name)
            with self.subTest(case=name), patch.dict(os.environ, {'H3_UNREVIEWED_OPTIMIZATION': '1'}):
                r = b.resolve_case(c, assets[name], Path('/tmp/sku-test'), '0,1,2,3,4,5,6,7', 18126)
                self.assertNotIn('H3_UNREVIEWED_OPTIMIZATION', r['environment'])
                self.assertFalse(any('{adapter}' in v or '{out}' in v for v in r['environment'].values()))
                self.assertNotIn('/lustre/', ' '.join(r['command']))
                if c.get('reference_asset'):
                    self.assertEqual(b.digest(r['paths']['reference']), c['reference_sha256'])
        fusion = b.get_case('15s-4step-output-fusion')
        self.assertEqual(fusion['environment']['VLLM_OMNI_H3_OUTPUT_FUSION'], '1')

    def test_avatar_preserves_all_original_continuation_requests(self):
        case = b.get_case('5s-avatar-ref2va-4step')
        self.assertEqual(len(case['requests']), 16)
        self.assertEqual([int(r['seed']) for r in case['requests']], list(range(8300, 8316)))
        seq = [json.loads(r['extra_params'])['rolling_continuation'] for r in case['requests']]
        self.assertEqual([s['ordinal'] for s in seq], list(range(16)))
        self.assertEqual(len({s['session'] for s in seq}), 1)
        self.assertTrue(all(s['window_frames'] == 124 and s['overlap_frames'] == 5 for s in seq))
        self.assertEqual(case['reference_copies'], 2)

    def test_model_fingerprint_follows_component_symlinks(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); (root / 'weights').mkdir(); (root / 'view').mkdir()
            weight = root / 'weights/model.safetensors'; weight.write_bytes(b'first model')
            (root / 'view/transformer').symlink_to(root / 'weights', target_is_directory=True)
            a = b.asset_identity(root / 'view')
            weight.write_bytes(b'different model')
            self.assertNotEqual(a['sha256'], b.asset_identity(root / 'view')['sha256'])

    def test_comparator_excludes_warmup_and_rejects_incompatible_runs(self):
        a = {'status': 'complete', 'case': 'test', 'case_identity': 'case', 'asset_identity': 'weights',
             'source_identity': 'source', 'software': {'torch': 'pinned'}, 'gpu_before': [],
             'requests': [{'phase': p, 'client_seconds': t, 'media': {'sha256': 'x'}}
                          for p, t in [('warmup', 999), ('measured', 3), ('measured', 2), ('measured', 4)]]}
        self.assertEqual(b.compare(a, a)['reference']['median_s'], 3)
        for key, value in [('asset_identity', 'other weights'), ('source_identity', 'other source'),
                           ('case_identity', 'other case'), ('status', 'failed'),
                           ('telemetry_errors', ['foreign GPU activity'])]:
            other = copy.deepcopy(a); other[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                b.compare(a, other)

    def test_gpu_selection_requires_same_sku_and_exact_count(self):
        cards = [{'index': str(i), 'uuid': f'GPU-{i}', 'name': 'same'} for i in range(8)]
        self.assertEqual(len(b.select_gpus(cards, ','.join(str(i) for i in range(8)), 8)), 8)
        with self.assertRaises(ValueError): b.select_gpus(cards, '0,0', 2)
        cards[1]['name'] = 'different'
        with self.assertRaises(ValueError): b.select_gpus(cards, '0,1', 2)


if __name__ == '__main__':
    unittest.main()
