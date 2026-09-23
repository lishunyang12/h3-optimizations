#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Validate H3 VSA reverse-O piggybacking over real FlashInfer RDMA SP2.

The optimized path keeps the learned gate on its original sequence owner,
bundles only the coarse tiles needed by each owner behind that owner's fine O,
and writes the expanded payload directly into FlashInfer's registered reverse-O
source.  The oracle communicates the full gate and uses an ordinary NCCL
reverse Ulysses exchange.  Both paths must be BF16 bit-exact, including H3's
aligned suffix padding and rank-major head order.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402

from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata  # noqa: E402
from vllm_omni.diffusion.attention.backends.fastvideo_vsa import (  # noqa: E402
    _get_h3_tile_metadata,
    get_h3_vsa_owner_route_plan,
)
from vllm_omni.diffusion.attention.ops.minimax_h3_vsa_layout import (  # noqa: E402
    build_h3_aligned_untile_source_rows,
)
from vllm_omni.diffusion.attention.ops.minimax_h3_vsa_o_bundle import (  # noqa: E402
    H3_VSA_O_BUNDLE_ACTIVE_KEY,
    H3_VSA_O_BUNDLE_STATE_KEY,
    h3_vsa_o_bundle_out,
)
from vllm_omni.diffusion.attention.parallel.ulysses import (  # noqa: E402
    UlyssesParallelAttention,
    _UlyssesCtx,
)
from vllm_omni.diffusion.distributed.comm import SeqAllToAll4D  # noqa: E402


def main() -> None:
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if world_size != 2:
        raise RuntimeError(f"this validator requires exactly two ranks, got {world_size}")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    from vllm_omni.diffusion.distributed import flashinfer_ulysses as adapter

    if not adapter.is_flashinfer_pcie_enabled():
        raise RuntimeError("VLLM_OMNI_ULYSSES_A2A_BACKEND must be flashinfer-pcie")
    if not adapter.is_o_producer_direct_enabled():
        raise RuntimeError("O producer-direct must be enabled")

    batch = 1
    local_rows = 64
    local_heads = 2
    global_heads = world_size * local_heads
    head_dim = 128
    global_rows = world_size * local_rows
    # An actual H3 tile builder supplies the row order.  The 127 valid rows
    # leave one aligned suffix row and make one video tile cross SP owners.
    prefix_segments = (63,)
    video_shape = (1, 1, 64)
    valid_rows = sum(prefix_segments) + int(torch.Size(video_shape).numel())
    if valid_rows != global_rows - 1:
        raise AssertionError("validator geometry must retain one padding row")

    (
        _partition,
        variable_block_sizes,
        _non_pad,
        untile,
        _prefix_blocks,
        _video_blocks,
    ) = _get_h3_tile_metadata(
        prefix_segments,
        video_shape,
        torch.device("cpu"),
    )
    logical_blocks = int(variable_block_sizes.numel())
    aligned_source_rows = build_h3_aligned_untile_source_rows(
        untile,
        aligned_rows=global_rows,
        tiled_rows=logical_blocks * 64,
    )
    plan = get_h3_vsa_owner_route_plan(
        prefix_segments,
        video_shape,
        global_rows,
        world_size,
    )
    if plan.logical_block_count != logical_blocks:
        raise AssertionError("cached owner route changed the H3 block count")
    row_blocks = torch.div(
        aligned_source_rows[:valid_rows].to(device=device, dtype=torch.int64),
        64,
        rounding_mode="floor",
    )

    generator = torch.Generator(device=device).manual_seed(1101 + rank)
    local_q = torch.randn(
        batch,
        local_rows,
        global_heads,
        head_dim,
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )
    local_k = torch.randn(
        local_q.shape,
        dtype=local_q.dtype,
        device=device,
        generator=generator,
    )
    local_v = torch.randn(
        local_q.shape,
        dtype=local_q.dtype,
        device=device,
        generator=generator,
    )
    group_name = dist.group.WORLD.group_name
    # Match production collective order and arm capacity-backed registered
    # Q/K/V outputs before asking for an expanded O shape-only prototype.
    fine_global = adapter.flashinfer_ulysses_qkv_fwd(
        local_q,
        group_name,
        world_size,
        "q",
        False,
    )
    adapter.flashinfer_ulysses_qkv_fwd(
        local_k,
        group_name,
        world_size,
        "k",
        False,
    )
    adapter.flashinfer_ulysses_qkv_fwd(
        local_v,
        group_name,
        world_size,
        "v",
        False,
    )
    fine_global[:, -1].zero_()
    coarse_local = torch.randn(
        batch,
        logical_blocks,
        local_heads,
        head_dim,
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )
    gate_local = torch.randn(
        batch,
        local_rows,
        global_heads,
        head_dim,
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )

    # Oracle: communicate the full gate, perform exact BF16 gate arithmetic in
    # global-sequence/local-head layout, then reverse through ordinary NCCL.
    gate_by_head_owner = SeqAllToAll4D.apply(
        dist.group.WORLD,
        gate_local,
        2,
        1,
        False,
    )
    baseline_global = fine_global.clone()
    selected_coarse = coarse_local.index_select(1, row_blocks)
    baseline_global[:, :valid_rows] = (
        baseline_global[:, :valid_rows] + selected_coarse * gate_by_head_owner[:, :valid_rows]
    )
    expected = SeqAllToAll4D.apply(
        dist.group.WORLD,
        baseline_global,
        1,
        2,
        False,
    )

    expanded_shape = (
        batch,
        world_size * (local_rows + plan.kmax),
        local_heads,
        head_dim,
    )
    landing = adapter.flashinfer_ulysses_o_input_landing(
        fine_global,
        dist.group.WORLD,
        world_size,
        input_shape=expanded_shape,
    )
    if landing is None:
        raise RuntimeError("required expanded FlashInfer O landing was declined")
    produced = h3_vsa_o_bundle_out(
        fine_global,
        coarse_local,
        plan,
        out=landing,
    )
    if produced.data_ptr() != landing.data_ptr():
        raise AssertionError("O bundle producer did not preserve landing identity")

    state: dict[str, object] = {"plan": plan, "published": True}
    metadata = AttentionMetadata(
        extra={
            H3_VSA_O_BUNDLE_ACTIVE_KEY: True,
            H3_VSA_O_BUNDLE_STATE_KEY: state,
            "ulysses_o_input_landing": landing,
        },
    )
    context = _UlyssesCtx(
        name="ulysses",
        ulysses_pg=dist.group.WORLD,
        scatter_idx=2,
        gather_idx=1,
        use_sync=False,
        ulysses_rank=rank,
        strict_a2a_backend="flashinfer-pcie",
        o_bundle_gate_local=gate_local,
        o_bundle_metadata=metadata,
        o_bundle_state=state,
    )
    strategy = object.__new__(UlyssesParallelAttention)
    strategy._validate_o_bundle_producer_output(produced, context)
    reversed_bundle = adapter.flashinfer_ulysses_o_rev(
        produced,
        group_name,
        world_size,
        False,
    )
    actual = strategy._finish_o_bundle(reversed_bundle, context)
    torch.accelerator.synchronize(device)

    adapter_states = list(adapter._STATES.values())
    adapter_state = adapter_states[0] if len(adapter_states) == 1 else None
    transport = (
        adapter_state.communicator.transport
        if adapter_state is not None and adapter_state.communicator is not None
        else None
    )
    producer_direct = bool(
        adapter_state is not None
        and adapter_state.o_producer_direct_logged
        and adapter_state.o_producer_direct_expected is None
    )
    o_output = adapter_state.outputs.get(("o", "gather_heads", torch.bfloat16)) if adapter_state is not None else None
    local_mismatches = int(torch.count_nonzero(actual.view(torch.int16) != expected.view(torch.int16)))
    report = {
        "rank": rank,
        "transport": transport,
        "o_producer_direct": producer_direct,
        "bit_exact": local_mismatches == 0,
        "mismatches": local_mismatches,
        "head_order_exact": local_mismatches == 0,
        "suffix_padding_exact": bool(torch.count_nonzero(actual[:, -1]) == 0) if rank == world_size - 1 else True,
        "state_consumed": not state and not metadata.extra,
        "fine_view_aliases_reverse_bundle": (
            actual.untyped_storage().data_ptr() == reversed_bundle.untyped_storage().data_ptr()
        ),
        "expanded_shape": list(expanded_shape),
        "kmax": plan.kmax,
        "logical_blocks": logical_blocks,
        "registered_o_output_storage_bytes": (
            o_output.untyped_storage().nbytes() if isinstance(o_output, torch.Tensor) else 0
        ),
        "registered_o_input_storage_bytes": landing.untyped_storage().nbytes(),
    }
    reports: list[dict[str, object] | None] = [None] * world_size
    dist.all_gather_object(reports, report)
    failures = [
        item
        for item in reports
        if item is None
        or item["transport"] != "rdma"
        or item["o_producer_direct"] is not True
        or item["bit_exact"] is not True
        or item["head_order_exact"] is not True
        or item["suffix_padding_exact"] is not True
        or item["state_consumed"] is not True
        or item["fine_view_aliases_reverse_bundle"] is not True
    ]
    if rank == 0:
        print(
            "H3_VSA_O_BUNDLE_RDMA_SP2="
            + json.dumps(
                {
                    "status": "pass" if not failures else "failed",
                    "world_size": world_size,
                    "valid_rows": valid_rows,
                    "aligned_rows": global_rows,
                    "reports": reports,
                },
                sort_keys=True,
            ),
            flush=True,
        )

    adapter.clear_flashinfer_ulysses_communicators()
    dist.barrier()
    dist.destroy_process_group()
    if failures:
        raise SystemExit(f"reverse-O bundle RDMA SP2 validation failed: {failures}")


if __name__ == "__main__":
    main()
