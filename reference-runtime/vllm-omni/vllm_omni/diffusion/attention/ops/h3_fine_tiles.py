"""Fixed-shape query-tile launcher, preserving the qualified Cake row math."""
from functools import lru_cache
import hashlib
import json
from pathlib import Path

import torch


@lru_cache(None)
def cpu_plan():
    from vllm_omni.diffusion.attention.ops.h3_fine_row_order import plan

    _order, groups, needs = plan()
    return groups, needs


@lru_cache(None)
def tile_plan(device_index):
    groups, _needs = cpu_plan()
    return tuple(torch.tensor(g, dtype=torch.int32, device=f'cuda:{device_index}') for g in groups)


@lru_cache(None)
def load_module():
    from flashinfer.jit.core import gen_jit_spec, sm120a_nvcc_flags
    from flashinfer.jit.cake_sage_block_sparse_attention import _get_csrc_dir, _get_include_dir

    root = Path(__file__).parent/'h3_fine_pipeline_kernel'
    receipt = json.loads((root/'receipt.json').read_text())
    paths = []
    for kind in ('kernel', 'binding'):
        r = receipt['sources'][kind]
        p = root/Path(r['output']).name
        assert hashlib.sha256(p.read_bytes()).hexdigest() == r['output_sha256']
        paths.append(p)
    digest = hashlib.sha256(b''.join(p.read_bytes() for p in paths)).hexdigest()[:16]
    spec = gen_jit_spec(name=f'h3_v16_fine_tiles_{digest}', sources=paths,
        extra_cuda_cflags=[*sm120a_nvcc_flags, *receipt['compile_flags']],
        extra_ldflags=['-lcuda'], extra_include_paths=[_get_csrc_dir().parent, _get_include_dir()])
    return spec.build_and_load()


def launch_subset(quantized, indices, nums, sizes, scale, out, workspace, tiles):
    """Private ABI: tiles must be a unique, in-range, CPU-validated query list."""
    import tvm_ffi

    q, k, v, qs, ks, vs = quantized
    assert q.shape == k.shape == (1, 7, 105472, 128)
    assert v.shape == (1, 7, 128, 105472)
    assert out.shape == q.shape and out.dtype == torch.bfloat16
    assert q.dtype == k.dtype == torch.int8 and v.dtype == torch.float8_e4m3fn
    assert indices.shape == (1, 7, 1648, 1648) and nums.shape == (1, 7, 1648)
    assert sizes.shape == (1648,) and tiles.ndim == 1 and tiles.dtype == torch.int32
    assert 0 < tiles.numel() <= 1648
    assert qs.shape == (1,7,3296) and ks.shape == (1,7,1648) and vs.shape == (1,7,128)
    assert all(t.is_contiguous() and t.device == q.device for t in (*quantized,indices,nums,sizes,out,workspace,tiles))
    with tvm_ffi.use_torch_stream():
        load_module().run(q,k,v.view(torch.uint8),out,qs,ks,vs,indices,nums,sizes,
            105472,105472,7,1648,1648,1648,float(scale),tiles,workspace,tiles.numel(),7,1)
