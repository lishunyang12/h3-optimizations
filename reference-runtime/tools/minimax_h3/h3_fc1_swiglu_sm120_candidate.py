# Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

"""Experimental SM120 BF16 FC1 + pair-aware SwiGLU CuTe DSL candidate.

This module is intentionally a benchmark plugin, not a production dispatch.
It derives the scheduling and TMA plumbing from CUTLASS DSL's installed
``blackwell_geforce/dense_gemm.py`` example and replaces its device kernel
with a paired epilogue kernel:

* one CUDA launch covers both halves of the checkpoint FC1 weight;
* gate rows are the first half and up rows are the second half;
* one logical output CTA executes the two GEMMs sequentially and keeps both
  accumulators in registers;
* the epilogue emits only ``silu(gate) * up`` in BF16.

There is deliberately no activation-kernel fallback in this module.  A
missing CUTLASS example or an unsupported device fails capability checking;
the benchmark harness remains responsible for using the production baseline.
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.hopper_helpers as sm90_utils
import torch
from cutlass.cute.runtime import from_dlpack


CANDIDATE_CONTRACT: dict[str, Any] = {
    "api_version": 1,
    "name": "sm120-cute-bf16-fc1-paired-swiglu-v0",
    "architecture": "sm120",
    "implementation": "cute_dsl",
    "single_kernel": True,
    "pair_aware_epilogue": True,
    "input_dtype": "bfloat16",
    "weight_dtype": "bfloat16",
    "output_dtype": "bfloat16",
    "weight_layout": "gate_rows_then_up_rows",
    "activation": "silu_gate_mul_up",
    "pre_activation_rounding": "gate_and_up_bfloat16",
    "post_silu_rounding": "bfloat16_before_up_multiply",
    "output_rounding": "bfloat16",
    "status": "experimental_benchmark_only",
}

EXPECTED_CUTLASS_DSL_VERSION = "4.6.2"
REFERENCE_RELATIVE_PATH = Path(
    "flashinfer/data/cutlass/examples/python/CuTeDSL/blackwell_geforce/dense_gemm.py"
)
SUPPORTED_TILES = (
    (64, 64, 64),
    (64, 128, 64),
    (128, 64, 64),
)
DEFAULT_TILE = (64, 64, 64)


@dataclass(frozen=True)
class Tactic:
    """Compile-time CTA tile for the paired kernel."""

    tile_m: int
    tile_n: int
    tile_k: int

    @property
    def tile_shape_mnk(self) -> tuple[int, int, int]:
        return (self.tile_m, self.tile_n, self.tile_k)


def _find_reference_example() -> Path | None:
    """Find the example shipped by the installed FlashInfer wheel."""
    candidates: list[Path] = []
    override = os.environ.get("H3_CUTLASS_SM120_DENSE_EXAMPLE")
    if override:
        candidates.append(Path(override).expanduser())
    try:
        distribution = importlib.metadata.distribution("flashinfer-python")
        candidates.append(Path(distribution.locate_file(REFERENCE_RELATIVE_PATH)))
    except importlib.metadata.PackageNotFoundError:
        pass
    candidates.extend(Path(entry) / REFERENCE_RELATIVE_PATH for entry in sys.path)
    return next((path.resolve() for path in candidates if path.is_file()), None)


def candidate_capability_report(*, probe_device: bool = False) -> dict[str, Any]:
    """Return dependency/device facts without compiling or launching a kernel."""
    try:
        dsl_version = importlib.metadata.version("nvidia-cutlass-dsl")
    except importlib.metadata.PackageNotFoundError:
        dsl_version = None
    reference = _find_reference_example()
    report: dict[str, Any] = {
        "nvidia_cutlass_dsl_version": dsl_version,
        "expected_nvidia_cutlass_dsl_version": EXPECTED_CUTLASS_DSL_VERSION,
        "cutlass_dsl_version_matches": dsl_version == EXPECTED_CUTLASS_DSL_VERSION,
        "reference_example": str(reference) if reference else None,
        "reference_example_found": reference is not None,
        "supported_tiles": [list(tile) for tile in SUPPORTED_TILES],
        "single_launch": True,
        "production_dispatch_modified": False,
    }
    if probe_device:
        available = torch.cuda.is_available()
        report["cuda_available"] = available
        if available:
            props = torch.cuda.get_device_properties(torch.cuda.current_device())
            report["device"] = {
                "name": props.name,
                "compute_capability": [props.major, props.minor],
                "is_sm120": (props.major, props.minor) == (12, 0),
            }
    return report


def _load_reference_module() -> ModuleType:
    path = _find_reference_example()
    if path is None:
        raise RuntimeError(
            "CUTLASS DSL SM120 dense GEMM example was not found; install the "
            "FlashInfer wheel containing " + str(REFERENCE_RELATIVE_PATH)
        )
    name = f"_h3_cutlass_sm120_dense_reference_{abs(hash(path))}"
    module = sys.modules.get(name)
    if module is not None:
        return module
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load CUTLASS reference example at {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_REFERENCE = _load_reference_module()
_Sm120GemmKernel = _REFERENCE.Sm120GemmKernel


class Sm120Fc1SwiGLUKernel(_Sm120GemmKernel):
    """Two BF16 GEMMs plus pair-aware SwiGLU in one SM120 kernel launch."""

    def __init__(self, tactic: Tactic) -> None:
        super().__init__(cutlass.Float32, tactic.tile_shape_mnk)
        # Two accumulator fragments stay live through the epilogue.  Match the
        # reviewed SM12x gated-MoE split (32 DMA / 232 MMA) and keep the tactic
        # space below 128x128. PTXAS still reports the actual register count.
        self.load_register_requirement = 32
        self.mma_register_requirement = 232

    @cute.kernel
    def kernel(
        self,
        tma_atom_a: cute.CopyAtom,
        mA_mkl: cute.Tensor,
        tma_atom_b: cute.CopyAtom,
        mB_nkl: cute.Tensor,
        tma_atom_c: cute.CopyAtom,
        mC_mnl: cute.Tensor,
        tiled_mma: cute.TiledMma,
        cta_layout_mnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        epi_smem_layout_staged: cute.ComposedLayout,
        tile_sched_params: utils.PersistentTileSchedulerParams,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

        if warp_idx == 0:
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_a)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_b)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_c)

        cta_rank = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        cluster_coord_mnk = cta_layout_mnk.get_flat_coord(cta_rank)
        a_mcast_mask = cute.make_layout_image_mask(
            cta_layout_mnk, cluster_coord_mnk, mode=1
        )
        b_mcast_mask = cute.make_layout_image_mask(
            cta_layout_mnk, cluster_coord_mnk, mode=0
        )
        a_mcast_mask = a_mcast_mask if self.is_a_mcast else 0
        b_mcast_mask = b_mcast_mask if self.is_b_mcast else 0

        a_smem_layout = cute.slice_(a_smem_layout_staged, (None, None, 0))
        b_smem_layout = cute.slice_(b_smem_layout_staged, (None, None, 0))
        tma_copy_bytes = cute.size_in_bytes(
            self.a_dtype, a_smem_layout
        ) + cute.size_in_bytes(self.b_dtype, b_smem_layout)

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)
        mainloop_pipeline = pipeline.PipelineTmaAsync.create(
            num_stages=self.ab_stage,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                (self.num_mcast_ctas_a + self.num_mcast_ctas_b - 1)
                * self.num_mma_warps,
            ),
            tx_count=tma_copy_bytes,
            barrier_storage=storage.mainloop_pipeline_array_ptr.data_ptr(),
            cta_layout_vmnk=cute.make_layout((1, *cta_layout_mnk.shape)),
        )

        if cute.size(self.cluster_shape_mnk) > 1:
            cute.arch.cluster_arrive_relaxed()

        sA = storage.sA.get_tensor(
            a_smem_layout_staged.outer, swizzle=a_smem_layout_staged.inner
        )
        sB = storage.sB.get_tensor(
            b_smem_layout_staged.outer, swizzle=b_smem_layout_staged.inner
        )
        sC = storage.sC.get_tensor(
            epi_smem_layout_staged.outer, swizzle=epi_smem_layout_staged.inner
        )

        gA_mkl = cute.local_tile(
            mA_mkl,
            cute.slice_(self.tile_shape_mnk, (None, 0, None)),
            (None, None, None),
        )
        gB_nkl = cute.local_tile(
            mB_nkl,
            cute.slice_(self.tile_shape_mnk, (0, None, None)),
            (None, None, None),
        )
        gC_mnl = cute.local_tile(
            mC_mnl,
            cute.slice_(self.tile_shape_mnk, (None, None, 0)),
            (None, None, None),
        )
        n_pair_tile_cnt = cute.size(gC_mnl, mode=[3])

        thr_mma = tiled_mma.get_slice(tidx)
        a_cta_layout = cute.make_layout(
            cute.slice_(cta_layout_mnk, (0, None, 0)).shape
        )
        tAsA, tAgA = cute.nvgpu.cpasync.tma_partition(
            tma_atom_a,
            cluster_coord_mnk[1],
            a_cta_layout,
            cute.group_modes(sA, 0, 2),
            cute.group_modes(gA_mkl, 0, 2),
        )
        b_cta_layout = cute.make_layout(
            cute.slice_(cta_layout_mnk, (None, 0, 0)).shape
        )
        tBsB, tBgB = cute.nvgpu.cpasync.tma_partition(
            tma_atom_b,
            cluster_coord_mnk[0],
            b_cta_layout,
            cute.group_modes(sB, 0, 2),
            cute.group_modes(gB_nkl, 0, 2),
        )

        tCsA = thr_mma.partition_A(sA)
        tCsB = thr_mma.partition_B(sB)
        tCrA = tiled_mma.make_fragment_A(tCsA[None, None, None, 0])
        tCrB = tiled_mma.make_fragment_B(tCsB[None, None, None, 0])
        tCgC = thr_mma.partition_C(gC_mnl)
        acc_shape = tCgC.shape[:3]
        gate_acc = cute.make_rmem_tensor(acc_shape, self.acc_dtype)
        up_acc = cute.make_rmem_tensor(acc_shape, self.acc_dtype)

        if cute.size(self.cluster_shape_mnk) > 1:
            cute.arch.cluster_wait()
        else:
            pipeline.sync(barrier_id=1)

        k_tile_cnt = cute.size(gA_mkl, mode=[3])
        tile_sched = utils.StaticPersistentTileScheduler.create(
            tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
        )
        work_tile = tile_sched.initial_work_tile_info()
        producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.ab_stage
        )
        consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.ab_stage
        )

        if warp_idx < self.num_mma_warps:
            cute.arch.setmaxregister_increase(self.mma_register_requirement)
            num_k_blocks = cute.size(tCrA, mode=[2])
            copy_a = cute.make_tiled_copy_A(
                cute.make_copy_atom(
                    cute.nvgpu.warp.LdMatrix8x8x16bOp(
                        self.a_layout.is_m_major_a(), 4
                    ),
                    self.a_dtype,
                ),
                tiled_mma,
            )
            copy_b = cute.make_tiled_copy_B(
                cute.make_copy_atom(
                    cute.nvgpu.warp.LdMatrix8x8x16bOp(
                        self.b_layout.is_n_major_b(), 4
                    ),
                    self.b_dtype,
                ),
                tiled_mma,
            )
            copy_a_thr = copy_a.get_slice(tidx)
            copy_b_thr = copy_b.get_slice(tidx)
            tCsA_view = copy_a_thr.partition_S(sA)
            tCsB_view = copy_b_thr.partition_S(sB)
            tCrA_view = copy_a_thr.retile(tCrA)
            tCrB_view = copy_b_thr.retile(tCrB)

            while work_tile.is_valid_tile:
                tile_coord_mnl = work_tile.tile_idx
                gC_slice = gC_mnl[(None, None, *tile_coord_mnl)]

                # The producer places gate then up K streams into the same AB
                # pipeline.  A constexpr loop keeps both accumulation passes
                # inside this one resident CTA/kernel launch.
                accumulators = gate_acc
                for fc1_part in cutlass.range_constexpr(2):
                    if fc1_part == 0:
                        accumulators = gate_acc
                    else:
                        accumulators = up_acc
                    accumulators.fill(0.0)
                    consumer_state.reset_count()

                    peek = cutlass.Boolean(1)
                    if consumer_state.count < k_tile_cnt:
                        peek = mainloop_pipeline.consumer_try_wait(consumer_state)
                    mainloop_pipeline.consumer_wait(consumer_state, peek)
                    tCsA_p = tCsA_view[None, None, None, consumer_state.index]
                    tCsB_p = tCsB_view[None, None, None, consumer_state.index]
                    cute.copy(
                        copy_a, tCsA_p[None, None, 0], tCrA_view[None, None, 0]
                    )
                    cute.copy(
                        copy_b, tCsB_p[None, None, 0], tCrB_view[None, None, 0]
                    )

                    for _ in range(0, k_tile_cnt - 1, 1, unroll=1):
                        for k_block in cutlass.range_constexpr(num_k_blocks):
                            k_next = (
                                0 if k_block + 1 == num_k_blocks else k_block + 1
                            )
                            if k_block == num_k_blocks - 1:
                                mainloop_pipeline.consumer_release(consumer_state)
                                consumer_state.advance()
                                peek = mainloop_pipeline.consumer_try_wait(
                                    consumer_state
                                )
                                tCsA_p = tCsA_view[
                                    None, None, None, consumer_state.index
                                ]
                                tCsB_p = tCsB_view[
                                    None, None, None, consumer_state.index
                                ]
                                mainloop_pipeline.consumer_wait(consumer_state, peek)
                            cute.copy(
                                copy_a,
                                tCsA_p[None, None, k_next],
                                tCrA_view[None, None, k_next],
                            )
                            cute.copy(
                                copy_b,
                                tCsB_p[None, None, k_next],
                                tCrB_view[None, None, k_next],
                            )
                            cute.gemm(
                                tiled_mma,
                                accumulators,
                                tCrA[None, None, k_block],
                                tCrB[None, None, k_block],
                                accumulators,
                            )

                    for k_block in cutlass.range_constexpr(num_k_blocks):
                        k_next = 0 if k_block + 1 == num_k_blocks else k_block + 1
                        if k_block == num_k_blocks - 1:
                            mainloop_pipeline.consumer_release(consumer_state)
                            consumer_state.advance()
                        if k_next > 0:
                            cute.copy(
                                copy_a,
                                tCsA_p[None, None, k_next],
                                tCrA_view[None, None, k_next],
                            )
                            cute.copy(
                                copy_b,
                                tCsB_p[None, None, k_next],
                                tCrB_view[None, None, k_next],
                            )
                        cute.gemm(
                            tiled_mma,
                            accumulators,
                            tCrA[None, None, k_block],
                            tCrB[None, None, k_block],
                            accumulators,
                        )

                copy_r2s_atom = sm90_utils.sm90_get_smem_store_op(
                    self.c_layout,
                    elem_ty_d=self.c_dtype,
                    elem_ty_acc=self.acc_dtype,
                )
                copy_c_atom = cute.make_copy_atom(
                    cute.nvgpu.warp.StMatrix8x8x16bOp(
                        self.c_layout.is_m_major_c(), 4
                    ),
                    self.c_dtype,
                )
                copy_c = cute.make_tiled_copy_C_atom(copy_c_atom, tiled_mma)
                copy_r2s = cute.make_tiled_copy_S(copy_r2s_atom, copy_c)
                copy_r2s_thr = copy_r2s.get_slice(tidx)
                tRS_sD = copy_r2s_thr.partition_D(sC)
                tRS_rGate = copy_r2s.retile(gate_acc)
                tRS_rUp = copy_r2s.retile(up_acc)

                rD_shape = cute.shape(copy_r2s_thr.partition_S(sC))
                rD_layout = cute.make_layout(rD_shape[:3])
                rD = cute.make_rmem_tensor(rD_layout.shape, self.c_dtype)
                rD_size = cute.size(rD)
                sC_grouped = cute.group_modes(sC, 0, 2)
                gC_divided = cute.zipped_divide(gC_slice, self.epi_tile)
                bSG_sD, bSG_gD = cute.nvgpu.cpasync.tma_partition(
                    tma_atom_c,
                    0,
                    cute.make_layout(1),
                    sC_grouped,
                    gC_divided,
                )
                epi_tile_num = cute.size(gC_divided, mode=[1])
                epi_tile_shape = gC_divided.shape[1]
                epi_tile_layout = cute.make_layout(
                    epi_tile_shape, stride=(1, epi_tile_shape[0])
                )
                store_pipeline = pipeline.PipelineTmaStore.create(
                    num_stages=self.epi_stage,
                    producer_group=pipeline.CooperativeGroup(
                        pipeline.Agent.Thread,
                        self.num_mma_warps * self.num_threads_per_warp,
                    ),
                )

                for epi_idx in cutlass.range_constexpr(epi_tile_num):
                    for epi_v in cutlass.range_constexpr(rD_size):
                        fragment_idx = epi_idx * rD_size + epi_v
                        # Match the production packed activation's observable
                        # BF16 boundaries, including its BF16 SiLU temporary.
                        gate = (
                            tRS_rGate[fragment_idx]
                            .to(cutlass.BFloat16)
                            .to(cutlass.Float32)
                        )
                        up = (
                            tRS_rUp[fragment_idx]
                            .to(cutlass.BFloat16)
                            .to(cutlass.Float32)
                        )
                        silu = gate / (
                            cutlass.Float32(1.0)
                            + cute.math.exp(-gate, fastmath=False)
                        )
                        silu_bf16 = silu.to(cutlass.BFloat16).to(cutlass.Float32)
                        rD[epi_v] = (silu_bf16 * up).to(cutlass.BFloat16)

                    epi_buffer = epi_idx % cute.size(tRS_sD, mode=[3])
                    cute.copy(
                        copy_r2s,
                        rD,
                        tRS_sD[(None, None, None, epi_buffer)],
                    )
                    cute.arch.fence_proxy("async.shared", space="cta")
                    self.epilog_sync_barrier.arrive_and_wait()
                    gmem_coord = epi_tile_layout.get_hier_coord(epi_idx)
                    if warp_idx == 0:
                        cute.copy(
                            tma_atom_c,
                            bSG_sD[(None, epi_buffer)],
                            bSG_gD[(None, gmem_coord)],
                        )
                        store_pipeline.producer_commit()
                        store_pipeline.producer_acquire()

                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()
                store_pipeline.producer_tail()

        elif warp_idx == self.num_mma_warps:
            cute.arch.setmaxregister_decrease(self.load_register_requirement)
            while work_tile.is_valid_tile:
                tile_coord_mnl = work_tile.tile_idx
                tAgA_tile = tAgA[
                    (None, tile_coord_mnl[0], None, tile_coord_mnl[2])
                ]

                for fc1_part in cutlass.range_constexpr(2):
                    b_tile = tile_coord_mnl[1] + fc1_part * n_pair_tile_cnt
                    tBgB_tile = tBgB[(None, b_tile, None, tile_coord_mnl[2])]
                    producer_state.reset_count()
                    for _ in range(0, k_tile_cnt, 1, unroll=1):
                        mainloop_pipeline.producer_acquire(producer_state)
                        tAgA_k = tAgA_tile[(None, producer_state.count)]
                        tBgB_k = tBgB_tile[(None, producer_state.count)]
                        tAsA_pipe = tAsA[(None, producer_state.index)]
                        tBsB_pipe = tBsB[(None, producer_state.index)]
                        barrier = mainloop_pipeline.producer_get_barrier(producer_state)
                        cute.copy(
                            tma_atom_a,
                            tAgA_k,
                            tAsA_pipe,
                            tma_bar_ptr=barrier,
                            mcast_mask=a_mcast_mask,
                        )
                        cute.copy(
                            tma_atom_b,
                            tBgB_k,
                            tBsB_pipe,
                            tma_bar_ptr=barrier,
                            mcast_mask=b_mcast_mask,
                        )
                        mainloop_pipeline.producer_commit(producer_state)
                        producer_state.advance()

                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()
            mainloop_pipeline.producer_tail(producer_state)
        return


def _parse_tactic() -> Tactic:
    raw = os.environ.get("H3_FC1_SWIGLU_TILE_MNK")
    if raw is None:
        tile = DEFAULT_TILE
    else:
        try:
            tile = tuple(int(value.strip()) for value in raw.split(","))
        except ValueError as exc:
            raise ValueError(
                "H3_FC1_SWIGLU_TILE_MNK must be three comma-separated integers"
            ) from exc
    if tile not in SUPPORTED_TILES:
        raise ValueError(f"unsupported tile {tile}; choose one of {SUPPORTED_TILES}")
    return Tactic(*tile)


def _validate_torch_contract(
    *, weight: torch.Tensor, shape: dict[str, Any], device: torch.device
) -> tuple[int, int, int]:
    m, k, n_pair = int(shape["m"]), int(shape["k"]), int(shape["n_pair"])
    if weight.shape != (2 * n_pair, k):
        raise ValueError(
            f"weight shape {tuple(weight.shape)} != expected {(2 * n_pair, k)}"
        )
    if weight.dtype is not torch.bfloat16:
        raise TypeError(f"weight must be BF16, got {weight.dtype}")
    if weight.device != device or device.type != "cuda":
        raise ValueError(f"weight/device mismatch: {weight.device} vs {device}")
    if not weight.is_contiguous():
        raise ValueError("weight must be contiguous gate rows followed by up rows")
    props = torch.cuda.get_device_properties(device)
    if (props.major, props.minor) != (12, 0):
        raise RuntimeError(f"candidate requires SM120, got SM{props.major}{props.minor}")
    version = importlib.metadata.version("nvidia-cutlass-dsl")
    if version != EXPECTED_CUTLASS_DSL_VERSION:
        raise RuntimeError(
            f"candidate was reviewed for CUTLASS DSL {EXPECTED_CUTLASS_DSL_VERSION}, "
            f"got {version}"
        )
    # TMA's contiguous BF16 dimension must be at least 16-byte aligned.
    if k % 8 or n_pair % 8:
        raise ValueError("K and Npair must be multiples of 8 BF16 values")
    return m, k, n_pair


def _as_cute_3d(tensor: torch.Tensor) -> cute.Tensor:
    return from_dlpack(tensor.unsqueeze(-1), assumed_align=16)


class _Runner:
    def __init__(
        self,
        *,
        weight: torch.Tensor,
        shape: dict[str, Any],
        device: torch.device,
        tactic: Tactic,
    ) -> None:
        self.weight = weight
        self.device = device
        self.m, self.k, self.n_pair = _validate_torch_contract(
            weight=weight, shape=shape, device=device
        )
        self.tactic = tactic
        self.benchmark_metadata = {
            "tile_shape_mnk": list(tactic.tile_shape_mnk),
            "fc1_passes_per_cta": ["gate", "up"],
            "a_tile_reloaded_between_passes": True,
            "weight_prepacked": False,
        }
        self.kernel = Sm120Fc1SwiGLUKernel(tactic)

        # Compile against the real static shape.  No kernel is launched here.
        compile_input = torch.empty(
            (self.m, self.k), dtype=torch.bfloat16, device=device
        )
        compile_output = torch.empty(
            (self.m, self.n_pair), dtype=torch.bfloat16, device=device
        )
        stream = cuda.CUstream(torch.cuda.current_stream(device).cuda_stream)
        max_clusters = utils.HardwareInfo().get_max_active_clusters(1)
        self.compiled = cute.compile(
            self.kernel,
            _as_cute_3d(compile_input),
            _as_cute_3d(self.weight),
            _as_cute_3d(compile_output),
            max_clusters,
            stream,
        )
        del compile_input, compile_output

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape != (self.m, self.k):
            raise ValueError(f"input shape {tuple(x.shape)} != {(self.m, self.k)}")
        if x.dtype is not torch.bfloat16 or x.device != self.device:
            raise TypeError("input must be BF16 on the candidate build device")
        if not x.is_contiguous():
            raise ValueError("input must be contiguous")
        output = torch.empty(
            (self.m, self.n_pair), dtype=torch.bfloat16, device=self.device
        )
        stream = cuda.CUstream(torch.cuda.current_stream(self.device).cuda_stream)
        self.compiled(
            _as_cute_3d(x),
            _as_cute_3d(self.weight),
            _as_cute_3d(output),
            stream,
        )
        return output


def build_candidate(
    *, weight: torch.Tensor, shape: dict[str, Any], device: torch.device
) -> _Runner:
    """Build the isolated exact-shape candidate; compilation launches no GPU work."""
    return _Runner(weight=weight, shape=shape, device=device, tactic=_parse_tactic())
