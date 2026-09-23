"""Immutable row permutation, validated against the original CPU inverse map."""
from functools import lru_cache
import json
from pathlib import Path
import torch


@lru_cache(None)
def plan():
    from vllm_omni.diffusion.attention.backends.fastvideo_vsa import _get_h3_aligned_untile_source_rows
    data=json.loads(Path(__file__).with_name('h3_fine_row_plan.json').read_text())
    order=torch.tensor(data['row_order'],dtype=torch.int64)
    assert order.shape==(8,11992)
    assert torch.equal(torch.sort(order,dim=1).values,torch.arange(11992).expand(8,-1))
    source=_get_h3_aligned_untile_source_rows((558,1206),(107,22,40),95936,105472,torch.device('cpu')).view(8,11992)
    rows=source.gather(1,order)
    groups=tuple(tuple(g) for g in data['groups']);needs=[];done=set()
    assert tuple(map(len,groups))==(400,415,406,427)
    for i,g in enumerate(groups):
        assert len(set(g))==len(g) and not (set(g)&done)
        done.update(g)
        part=rows[:,i*2998:(i+1)*2998]
        needed=set((part[part>=0]//64).tolist())
        assert needed<=done
        needs.append(tuple(sorted(needed)))
    assert done==set(range(1648))
    return order,groups,tuple(needs)


@lru_cache(None)
def row_order(device):
    return plan()[0].to(device)
