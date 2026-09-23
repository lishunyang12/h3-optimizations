# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch

from tests.helpers.mark import hardware_test

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion]


@hardware_test(res={"cuda": ["B200"]}, num_cards=1)
@pytest.mark.parametrize("rows,keys", [(128, 192), (129, 191), (256, 256)])
@pytest.mark.parametrize("precision", ["sage", "bf16"])
def test_sage_sparse_against_masked_sdpa(rows, keys, precision):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0):
        pytest.skip("requires SM120")
    pytest.importorskip("flashinfer.cute_dsl.sparse.bsa_attn_sm120")
    from vllm_omni.diffusion.attention.ops.sage_block_sparse_attention import flashinfer_block_sparse_attention

    generator = torch.Generator(device="cuda").manual_seed(7415)
    q = torch.randn(1, rows, 2, 128, device="cuda", dtype=torch.bfloat16, generator=generator)
    k = torch.randn(1, keys, 2, 128, device="cuda", dtype=torch.bfloat16, generator=generator)
    v = torch.randn(1, keys, 2, 128, device="cuda", dtype=torch.bfloat16, generator=generator)
    blocks = (keys + 63) // 64
    indices = torch.arange(blocks, device="cuda", dtype=torch.int32).repeat(1, 2, (rows + 63) // 64, 1)
    counts = torch.full(indices.shape[:3], blocks, device="cuda", dtype=torch.int32)
    counts[..., 1::2] -= 1
    sizes = torch.full((blocks,), 64, device="cuda", dtype=torch.int32)
    sizes[-1] = keys - (blocks - 1) * 64
    mask = torch.arange(keys, device="cuda")[None, :] < counts[0, 0].repeat_interleave(64)[:rows, None] * 64
    expected = torch.nn.functional.scaled_dot_product_attention(
        q.transpose(1, 2).float(), k.transpose(1, 2).float(), v.transpose(1, 2).float(), attn_mask=mask
    ).transpose(1, 2)
    block_map = torch.arange(blocks, device="cuda")[None, None, None, :] < counts[..., None]
    actual = flashinfer_block_sparse_attention(q, k, v, block_map, sizes, 128**-0.5, precision=precision)
    torch.accelerator.synchronize()
    assert actual.shape == q.shape and actual.dtype == q.dtype
    relative_rms = (actual.float() - expected).square().mean().sqrt() / expected.square().mean().sqrt()
    assert relative_rms.item() < (0.08 if precision == "sage" else 0.01)
    assert torch.isfinite(actual).all()
