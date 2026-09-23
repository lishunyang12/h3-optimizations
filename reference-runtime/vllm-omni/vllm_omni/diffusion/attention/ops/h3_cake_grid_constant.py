"""Load only the fixed H3 descriptor-ABI backport; fail closed on mismatch."""
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path

@lru_cache(None)
def configure():
    import torch
    from flashinfer.jit import cake_sage_block_sparse_attention as jit
    from vllm.logger import init_logger
    if os.environ.get("VLLM_OMNI_H3_CAKE_TMA_GRID_CONSTANT") != "1":
        raise RuntimeError("This private H3 arm requires the grid-constant ABI")
    csrc = Path(__file__).resolve().parent / "h3_cake_grid_constant_sources"
    jit._get_csrc_dir = lambda: csrc
    jit._manifest.cache_clear()
    jit.gen_cake_sage_block_sparse_attention_module.cache_clear()
    jit.load_cake_sage_block_sparse_attention_module.cache_clear()
    module, record = jit.load_cake_sage_block_sparse_attention_module(1,1,0,0,0)
    assert int(module.h3_tma_abi_signature()) == 5127001
    assert record["tma_workspace_bytes"] == 0 and record["tma_abi"] == "grid_constant"
    assert record["kernel_symbol"].endswith("_grid_constant_r1")
    init_logger(__name__).info("H3_CAKE_GRID_CONSTANT_READY %s", json.dumps(dict(
        rank=torch.distributed.get_rank(), signature=5127001, workspace_bytes=0,
        kernel=record["kernel_symbol"], manifest_sha256=hashlib.sha256(
            (csrc/"cake_sage_block_sparse_attention_manifest.json").read_bytes()).hexdigest()), sort_keys=True))
