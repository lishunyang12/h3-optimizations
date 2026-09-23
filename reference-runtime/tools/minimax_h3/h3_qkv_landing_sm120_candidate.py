# Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Experimental MiniMax-H3 QKV-to-Ulysses-landing CuTe DSL kernel.

This is an isolated benchmark candidate.  It is intentionally not imported by
the model or attention dispatch.  One CTA owns exactly one 128-wide Q, K, or V
head tile.  Q/K projection values are rounded to BF16 before their 128-wide
FP32 RMS reduction, rounded again after normalization, and then receive H3's
96-wide non-interleaved RoPE.  V is the BF16-rounded projection unchanged.

The three outputs are caller-owned contiguous tensors.  A benchmark may use
ordinary allocations; production would pass FlashInfer's registered input
landing views without changing the generated kernel.
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
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
    "name": "sm120-cute-bf16-h3-qkv-qknorm-rope-landing-v0",
    "architecture": "sm120",
    "implementation": "cute_dsl",
    "single_kernel": True,
    "input_dtype": "bfloat16",
    "weight_dtype": "bfloat16",
    "output_dtype": "bfloat16",
    "cta_tile_n": 128,
    "one_complete_head_per_n_tile": True,
    "split_k": False,
    "qk_projection_accumulator_rounding": "bfloat16_before_rms_reduction",
    "qk_rms_reduction": "128-wide-float32",
    "qk_normalized_rounding": "bfloat16_before_rope",
    "rotary_layout": "neox-noninterleaved-96-of-128",
    "v_transform": "identity",
    "outputs": "three-caller-owned-registered-contiguous-landings",
    "production_dispatch_modified": False,
    "status": "rejected_benchmark_only",
    "rejection": (
        "The stock [64,32,>=4]-stage epilogue candidate was 2.098 ms/layer "
        "slower than the incumbent and Q/K were not bit-exact."
    ),
}

EXPECTED_CUTLASS_DSL_VERSION = "4.6.2"
REFERENCE_RELATIVE_PATH = Path(
    "flashinfer/data/cutlass/examples/python/CuTeDSL/"
    "blackwell_geforce/dense_gemm.py"
)


@dataclass(frozen=True)
class Tactic:
    tile_m: int = 64
    tile_n: int = 128
    tile_k: int = 64

    @property
    def tile_shape_mnk(self) -> tuple[int, int, int]:
        return (self.tile_m, self.tile_n, self.tile_k)

    def validate(self, head_dim: int) -> None:
        if self.tile_m not in (64, 128):
            raise ValueError("SM120 tile M must be 64 or 128")
        if self.tile_n != head_dim or head_dim != 128:
            raise ValueError("one complete 128-wide head must occupy each N tile")
        if self.tile_k != 64:
            raise ValueError("the reviewed SM120 BF16 mainloop requires tile K=64")


def _find_reference_example() -> Path:
    dist = importlib.metadata.distribution("flashinfer-python")
    path = Path(dist.locate_file(REFERENCE_RELATIVE_PATH))
    if not path.is_file():
        raise RuntimeError(f"CUTLASS DSL SM120 dense GEMM example not found: {path}")
    return path


def _load_reference_module() -> ModuleType:
    path = _find_reference_example()
    name = f"_h3_qkv_landing_dense_reference_{abs(hash(path.resolve()))}"
    existing = sys.modules.get(name)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_REFERENCE = _load_reference_module()
_Sm120GemmKernel = _REFERENCE.Sm120GemmKernel


class Sm120H3QkvLandingKernel(_Sm120GemmKernel):
    """BF16 QKV GEMM with H3 Q/K normalization/RoPE landing epilogue."""

    def __init__(
        self,
        *,
        heads: int,
        head_dim: int,
        rotary_dim: int,
        eps: float,
        tactic: Tactic,
    ) -> None:
        tactic.validate(head_dim)
        if heads <= 0:
            raise ValueError("heads must be positive")
        if rotary_dim != 96:
            raise ValueError("H3 requires rotary_dim=96")
        super().__init__(cutlass.Float32, tactic.tile_shape_mnk)
        self.heads = heads
        self.head_dim = head_dim
        self.rotary_dim = rotary_dim
        self.eps = eps
        self.tactic = tactic

    def _setup_landing_attributes(self) -> None:
        """Keep the stock four-fragment epilogue layout for one head.

        For a 64x128 CTA, CUTLASS chooses a logical ``[64, 32, 8]`` BF16
        shared tensor.  Delaying TMA stores lets stages 0..3 retain all four
        32-wide fragments of one complete head.  This preserves the stock
        GEMM schedule and avoids the severe mainloop regression caused by a
        custom 64x128 single-stage epilogue layout.
        """
        self._setup_attributes()
        if self.epi_tile != (self.tactic.tile_m, 32) or self.epi_stage < 4:
            raise RuntimeError(
                "candidate requires stock logical [64,32,>=4] epilogue storage"
            )

    @cute.jit
    def __call__(
        self,
        a: cute.Tensor,
        b: cute.Tensor,
        q_weight: cute.Tensor,
        k_weight: cute.Tensor,
        rope_table: cute.Tensor,
        q_out: cute.Tensor,
        k_out: cute.Tensor,
        v_out: cute.Tensor,
        max_active_clusters: cutlass.Constexpr,
        stream: cuda.CUstream,
    ):
        self.a_dtype = a.element_type
        self.b_dtype = b.element_type
        self.c_dtype = q_out.element_type
        self.a_layout = utils.LayoutEnum.from_tensor(a)
        self.b_layout = utils.LayoutEnum.from_tensor(b)
        self.c_layout = utils.LayoutEnum.from_tensor(q_out)
        if cutlass.const_expr(
            self.a_dtype != cutlass.BFloat16
            or self.b_dtype != cutlass.BFloat16
            or self.c_dtype != cutlass.BFloat16
        ):
            raise TypeError("candidate accepts BF16 A, B, and outputs only")
        self._setup_landing_attributes()

        tma_atom_a, tma_tensor_a = self._make_tma_atoms_and_tensors(
            a,
            self.a_smem_layout_staged,
            (self.tile_shape_mnk[0], self.tile_shape_mnk[2]),
            1,
        )
        tma_atom_b, tma_tensor_b = self._make_tma_atoms_and_tensors(
            b,
            self.b_smem_layout_staged,
            (self.tile_shape_mnk[1], self.tile_shape_mnk[2]),
            1,
        )
        tma_atom_q, tma_tensor_q = self._make_tma_store_atoms_and_tensors(
            q_out,
            self.epi_smem_layout_staged,
            self.epi_tile,
        )
        tma_atom_k, tma_tensor_k = self._make_tma_store_atoms_and_tensors(
            k_out,
            self.epi_smem_layout_staged,
            self.epi_tile,
        )
        tma_atom_v, tma_tensor_v = self._make_tma_store_atoms_and_tensors(
            v_out,
            self.epi_smem_layout_staged,
            self.epi_tile,
        )

        m_extent = cute.size(q_out, mode=[0])
        num_ctas_mnl = (
            cute.ceil_div(m_extent, self.tile_shape_mnk[0]),
            3 * self.heads,
            1,
        )
        tile_sched_params = utils.PersistentTileSchedulerParams(
            num_ctas_mnl, (1, 1, 1)
        )
        grid = utils.StaticPersistentTileScheduler.get_grid_shape(
            tile_sched_params, max_active_clusters
        )

        @cute.struct
        class SharedStorage:
            mainloop_pipeline_array_ptr: cute.struct.MemRange[
                cutlass.Int64, self.ab_stage * 2
            ]
            sA: cute.struct.Align[
                cute.struct.MemRange[
                    self.a_dtype, cute.cosize(self.a_smem_layout_staged)
                ],
                self.buffer_align_bytes,
            ]
            sB: cute.struct.Align[
                cute.struct.MemRange[
                    self.b_dtype, cute.cosize(self.b_smem_layout_staged)
                ],
                self.buffer_align_bytes,
            ]
            sC: cute.struct.Align[
                cute.struct.MemRange[
                    self.c_dtype, cute.cosize(self.epi_smem_layout_staged)
                ],
                self.buffer_align_bytes,
            ]

        self.shared_storage = SharedStorage
        self.kernel(
            tma_atom_a,
            tma_tensor_a,
            tma_atom_b,
            tma_tensor_b,
            tma_atom_q,
            tma_tensor_q,
            tma_atom_k,
            tma_tensor_k,
            tma_atom_v,
            tma_tensor_v,
            q_weight,
            k_weight,
            rope_table,
            self.tiled_mma,
            self.cta_layout_mnk,
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.epi_smem_layout_staged,
            tile_sched_params,
        ).launch(
            grid=grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=[1, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        tma_atom_a: cute.CopyAtom,
        mA_mkl: cute.Tensor,
        tma_atom_b: cute.CopyAtom,
        mB_nkl: cute.Tensor,
        tma_atom_q: cute.CopyAtom,
        mQ_mnl: cute.Tensor,
        tma_atom_k: cute.CopyAtom,
        mK_mnl: cute.Tensor,
        tma_atom_v: cute.CopyAtom,
        mV_mnl: cute.Tensor,
        q_weight: cute.Tensor,
        k_weight: cute.Tensor,
        rope_table: cute.Tensor,
        tiled_mma: cute.TiledMma,
        cta_layout_mnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        epi_smem_layout_staged: cute.ComposedLayout,
        tile_sched_params: utils.PersistentTileSchedulerParams,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        lane_idx = tidx % cute.arch.WARP_SIZE

        if warp_idx == 0:
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_a)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_b)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_q)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_k)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_v)

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
        output_tile = cute.slice_(self.tile_shape_mnk, (None, None, 0))
        gQ_mnl = cute.local_tile(mQ_mnl, output_tile, (None, None, None))
        gK_mnl = cute.local_tile(mK_mnl, output_tile, (None, None, None))
        gV_mnl = cute.local_tile(mV_mnl, output_tile, (None, None, None))

        thr_mma = tiled_mma.get_slice(tidx)
        a_cta_layout = cute.make_layout(
            cute.slice_(cta_layout_mnk, (0, None, 0)).shape
        )
        b_cta_layout = cute.make_layout(
            cute.slice_(cta_layout_mnk, (None, 0, 0)).shape
        )
        tAsA, tAgA = cute.nvgpu.cpasync.tma_partition(
            tma_atom_a,
            cluster_coord_mnk[1],
            a_cta_layout,
            cute.group_modes(sA, 0, 2),
            cute.group_modes(gA_mkl, 0, 2),
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
        acc_shape = thr_mma.partition_shape_C(
            (self.tactic.tile_m, self.tactic.tile_n)
        )
        accumulators = thr_mma.make_fragment_C(acc_shape)

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
            num_k_blocks = cute.size(tCrA, mode=[2])

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
            tRS_rAcc = copy_r2s.retile(accumulators)
            rD_shape = cute.shape(copy_r2s_thr.partition_S(sC))
            rD_layout = cute.make_layout(rD_shape[:3])
            rD = cute.make_rmem_tensor(rD_layout.shape, self.c_dtype)
            rD_size = cute.size(rD)
            transformed = cute.make_rmem_tensor((4,), self.c_dtype)

            while work_tile.is_valid_tile:
                tile_coord_mnl = work_tile.tile_idx
                accumulators.fill(0.0)
                consumer_state.reset_count()
                peek = cutlass.Boolean(1)
                if consumer_state.count < k_tile_cnt:
                    peek = mainloop_pipeline.consumer_try_wait(consumer_state)
                mainloop_pipeline.consumer_wait(consumer_state, peek)
                tCsA_p = tCsA_view[None, None, None, consumer_state.index]
                tCsB_p = tCsB_view[None, None, None, consumer_state.index]
                cute.copy(copy_a, tCsA_p[None, None, 0], tCrA_view[None, None, 0])
                cute.copy(copy_b, tCsB_p[None, None, 0], tCrB_view[None, None, 0])

                for _ in range(0, k_tile_cnt - 1, 1, unroll=1):
                    for k_block in cutlass.range_constexpr(num_k_blocks):
                        k_next = 0 if k_block + 1 == num_k_blocks else k_block + 1
                        if k_block == num_k_blocks - 1:
                            mainloop_pipeline.consumer_release(consumer_state)
                            consumer_state.advance()
                            peek = mainloop_pipeline.consumer_try_wait(consumer_state)
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

                # Retain all four 32-wide accumulator fragments in distinct
                # stock epilogue stages.  The conversion is the incumbent
                # GEMM's observable BF16 boundary and must precede RMSNorm.
                for epi_idx in cutlass.range_constexpr(4):
                    for epi_v in cutlass.range_constexpr(rD_size):
                        rD[epi_v] = tRS_rAcc[
                            epi_idx * rD_size + epi_v
                        ].to(cutlass.BFloat16)
                    cute.copy(
                        copy_r2s,
                        rD,
                        tRS_sD[(None, None, None, epi_idx)],
                    )
                cute.arch.fence_proxy("async.shared", space="cta")
                self.epilog_sync_barrier.arrive_and_wait()

                tile_m = tile_coord_mnl[0]
                qkv_head = tile_coord_mnl[1]
                modality = qkv_head // self.heads
                head = qkv_head - modality * self.heads
                m_extent = cute.size(mQ_mnl, mode=[0])
                # ``warp_idx`` is a runtime value, so keep the loop bounds
                # constexpr and predicate ownership instead of using it as a
                # Python/CuTe range start.  Each of the four MMA warps owns 16
                # complete rows of the 64-row CTA tile.
                for local_row in cutlass.range_constexpr(self.tactic.tile_m):
                    owns_row = local_row % self.num_mma_warps == warp_idx
                    global_row = tile_m * self.tactic.tile_m + local_row
                    if owns_row and global_row < m_extent:
                        if modality < 2:
                            sum_sq = cutlass.Float32(0.0)
                            for vec in cutlass.range_constexpr(4):
                                dim = lane_idx + vec * cute.arch.WARP_SIZE
                                projected = sC[local_row, lane_idx, vec].to(
                                    cutlass.Float32
                                )
                                sum_sq += projected * projected
                            sum_sq = cute.arch.warp_reduction_sum(sum_sq)
                            inv_rms = cute.math.rsqrt(
                                sum_sq / cutlass.Float32(self.head_dim)
                                + cutlass.Float32(self.eps),
                                fastmath=True,
                            )
                            for vec in cutlass.range_constexpr(4):
                                dim = lane_idx + vec * cute.arch.WARP_SIZE
                                projected = sC[local_row, lane_idx, vec].to(
                                    cutlass.Float32
                                )
                                weight = (
                                    q_weight[dim]
                                    if modality == 0
                                    else k_weight[dim]
                                ).to(cutlass.Float32)
                                normalized = (
                                    projected * inv_rms * weight
                                ).to(cutlass.BFloat16).to(cutlass.Float32)
                                value = normalized
                                if dim < self.rotary_dim:
                                    pair_dim = (
                                        dim + self.rotary_dim // 2
                                        if dim < self.rotary_dim // 2
                                        else dim - self.rotary_dim // 2
                                    )
                                    pair_projected = sC[
                                        local_row,
                                        pair_dim % cute.arch.WARP_SIZE,
                                        pair_dim // cute.arch.WARP_SIZE,
                                    ].to(cutlass.Float32)
                                    pair_weight = (
                                        q_weight[pair_dim]
                                        if modality == 0
                                        else k_weight[pair_dim]
                                    ).to(cutlass.Float32)
                                    pair_normalized = (
                                        pair_projected * inv_rms * pair_weight
                                    ).to(cutlass.BFloat16).to(cutlass.Float32)
                                    freq_dim = (
                                        dim
                                        if dim < self.rotary_dim // 2
                                        else dim - self.rotary_dim // 2
                                    )
                                    cos = rope_table[global_row, freq_dim].to(
                                        cutlass.Float32
                                    )
                                    sin = rope_table[
                                        global_row,
                                        freq_dim + self.rotary_dim // 2,
                                    ].to(cutlass.Float32)
                                    value = (
                                        normalized * cos - pair_normalized * sin
                                        if dim < self.rotary_dim // 2
                                        else normalized * cos + pair_normalized * sin
                                    )
                                transformed[vec] = value.to(cutlass.BFloat16)
                            # Keep every source projection intact until all
                            # four pair reads finish.  Writing each rotary
                            # half in-place as it is computed corrupts the
                            # later half's d<->d+48 source.
                            for vec in cutlass.range_constexpr(4):
                                sC[local_row, lane_idx, vec] = transformed[vec]

                # Q/K warps modified shared memory; V can proceed directly
                # from the initial accumulator-to-shared barrier.
                if modality < 2:
                    cute.arch.fence_proxy("async.shared", space="cta")
                    self.epilog_sync_barrier.arrive_and_wait()

                sepi = cute.group_modes(sC, 0, 2)
                store_group = pipeline.CooperativeGroup(
                    pipeline.Agent.Thread,
                    self.num_mma_warps * self.num_threads_per_warp,
                )
                store_pipeline = pipeline.PipelineTmaStore.create(
                    num_stages=self.epi_stage,
                    producer_group=store_group,
                )

                # Keep three explicit branches: CuTe's TMA CopyAtom/Tensor
                # types remain compile-time values even though ``modality``
                # is a uniform runtime selector.
                if modality == 0:
                    g_slice = gQ_mnl[(None, None, tile_m, head, 0)]
                    tcgc = cute.zipped_divide(g_slice, self.epi_tile)
                    bSG_sD, bSG_gD = cute.nvgpu.cpasync.tma_partition(
                        tma_atom_q,
                        0,
                        cute.make_layout(1),
                        sepi,
                        tcgc,
                    )
                    epi_shape = tcgc.shape[1]
                    epi_layout = cute.make_layout(
                        epi_shape, stride=(1, epi_shape[0])
                    )
                    for epi_idx in cutlass.range_constexpr(4):
                        coord = epi_layout.get_hier_coord(epi_idx)
                        if warp_idx == 0:
                            cute.copy(
                                tma_atom_q,
                                bSG_sD[(None, epi_idx)],
                                bSG_gD[(None, coord)],
                            )
                        store_pipeline.producer_commit()
                        store_pipeline.producer_acquire()
                elif modality == 1:
                    g_slice = gK_mnl[(None, None, tile_m, head, 0)]
                    tcgc = cute.zipped_divide(g_slice, self.epi_tile)
                    bSG_sD, bSG_gD = cute.nvgpu.cpasync.tma_partition(
                        tma_atom_k,
                        0,
                        cute.make_layout(1),
                        sepi,
                        tcgc,
                    )
                    epi_shape = tcgc.shape[1]
                    epi_layout = cute.make_layout(
                        epi_shape, stride=(1, epi_shape[0])
                    )
                    for epi_idx in cutlass.range_constexpr(4):
                        coord = epi_layout.get_hier_coord(epi_idx)
                        if warp_idx == 0:
                            cute.copy(
                                tma_atom_k,
                                bSG_sD[(None, epi_idx)],
                                bSG_gD[(None, coord)],
                            )
                        store_pipeline.producer_commit()
                        store_pipeline.producer_acquire()
                else:
                    g_slice = gV_mnl[(None, None, tile_m, head, 0)]
                    tcgc = cute.zipped_divide(g_slice, self.epi_tile)
                    bSG_sD, bSG_gD = cute.nvgpu.cpasync.tma_partition(
                        tma_atom_v,
                        0,
                        cute.make_layout(1),
                        sepi,
                        tcgc,
                    )
                    epi_shape = tcgc.shape[1]
                    epi_layout = cute.make_layout(
                        epi_shape, stride=(1, epi_shape[0])
                    )
                    for epi_idx in cutlass.range_constexpr(4):
                        coord = epi_layout.get_hier_coord(epi_idx)
                        if warp_idx == 0:
                            cute.copy(
                                tma_atom_v,
                                bSG_sD[(None, epi_idx)],
                                bSG_gD[(None, coord)],
                            )
                        store_pipeline.producer_commit()
                        store_pipeline.producer_acquire()

                store_pipeline.producer_tail()
                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()

        elif warp_idx == self.num_mma_warps:
            cute.arch.setmaxregister_decrease(self.load_register_requirement)
            while work_tile.is_valid_tile:
                tile_coord_mnl = work_tile.tile_idx
                tAgA_tile = tAgA[
                    (None, tile_coord_mnl[0], None, tile_coord_mnl[2])
                ]
                tBgB_tile = tBgB[
                    (None, tile_coord_mnl[1], None, tile_coord_mnl[2])
                ]
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


def _as_cute_3d(tensor: torch.Tensor) -> cute.Tensor:
    return from_dlpack(tensor.unsqueeze(-1), assumed_align=16)


class CandidateRunner:
    def __init__(
        self,
        *,
        weight: torch.Tensor,
        q_weight: torch.Tensor,
        k_weight: torch.Tensor,
        rope_table: torch.Tensor,
        heads: int,
        head_dim: int,
        rotary_dim: int,
        eps: float,
        tactic: Tactic,
    ) -> None:
        if importlib.metadata.version("nvidia-cutlass-dsl") != EXPECTED_CUTLASS_DSL_VERSION:
            raise RuntimeError("candidate requires nvidia-cutlass-dsl 4.6.2")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required")
        self.device = weight.device
        props = torch.cuda.get_device_properties(self.device)
        if (props.major, props.minor) != (12, 0):
            raise RuntimeError(f"candidate requires SM120, got SM{props.major}{props.minor}")
        m = rope_table.shape[0]
        k = weight.shape[1]
        expected_n = 3 * heads * head_dim
        if weight.shape != (expected_n, k):
            raise ValueError("weight must contain contiguous Q, K, V head rows")
        if q_weight.shape != (head_dim,) or k_weight.shape != (head_dim,):
            raise ValueError("Q/K RMSNorm weights must have head_dim elements")
        if rope_table.shape != (m, rotary_dim):
            raise ValueError("rope_table shape mismatch")
        tensors = (weight, q_weight, k_weight, rope_table)
        if any(t.dtype is not torch.bfloat16 for t in tensors):
            raise TypeError("weight, norm weights, and rope table must be BF16")
        if any(t.device != self.device for t in tensors):
            raise ValueError("all tensors must share one CUDA device")
        if any(not t.is_contiguous() for t in tensors):
            raise ValueError("all tensors must be contiguous")
        self.weight = weight
        self.q_weight = q_weight
        self.k_weight = k_weight
        self.rope_table = rope_table
        self.m = m
        self.k = k
        self.heads = heads
        self.head_dim = head_dim
        self.rotary_dim = rotary_dim
        self.tactic = tactic
        self.outputs = tuple(
            torch.empty(
                (m, heads * head_dim),
                dtype=torch.bfloat16,
                device=self.device,
            )
            for _ in range(3)
        )
        self.kernel = Sm120H3QkvLandingKernel(
            heads=heads,
            head_dim=head_dim,
            rotary_dim=rotary_dim,
            eps=eps,
            tactic=tactic,
        )
        compile_input = torch.empty(
            (m, k), dtype=torch.bfloat16, device=self.device
        )
        max_clusters = utils.HardwareInfo().get_max_active_clusters(1)
        stream = cuda.CUstream(torch.cuda.current_stream(self.device).cuda_stream)
        self.compiled = cute.compile(
            self.kernel,
            _as_cute_3d(compile_input),
            _as_cute_3d(self.weight),
            from_dlpack(self.q_weight, assumed_align=16),
            from_dlpack(self.k_weight, assumed_align=16),
            from_dlpack(self.rope_table, assumed_align=16),
            _as_cute_3d(self.outputs[0]),
            _as_cute_3d(self.outputs[1]),
            _as_cute_3d(self.outputs[2]),
            max_clusters,
            stream,
        )
        del compile_input

    def __call__(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        if x.shape != (self.m, self.k):
            raise ValueError(f"input shape must be {(self.m, self.k)}")
        if x.dtype is not torch.bfloat16 or x.device != self.device:
            raise TypeError("input must be BF16 on the build device")
        if not x.is_contiguous():
            raise ValueError("input must be contiguous")
        stream = cuda.CUstream(torch.cuda.current_stream(self.device).cuda_stream)
        self.compiled(
            _as_cute_3d(x),
            _as_cute_3d(self.weight),
            from_dlpack(self.q_weight, assumed_align=16),
            from_dlpack(self.k_weight, assumed_align=16),
            from_dlpack(self.rope_table, assumed_align=16),
            _as_cute_3d(self.outputs[0]),
            _as_cute_3d(self.outputs[1]),
            _as_cute_3d(self.outputs[2]),
            stream,
        )
        return self.outputs


def build_candidate(
    *,
    weight: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    rope_table: torch.Tensor,
    heads: int,
    head_dim: int = 128,
    rotary_dim: int = 96,
    eps: float = 1e-5,
    tactic: Tactic | None = None,
) -> CandidateRunner:
    return CandidateRunner(
        weight=weight,
        q_weight=q_weight,
        k_weight=k_weight,
        rope_table=rope_table,
        heads=heads,
        head_dim=head_dim,
        rotary_dim=rotary_dim,
        eps=eps,
        tactic=tactic or Tactic(),
    )


__all__ = [
    "CANDIDATE_CONTRACT",
    "CandidateRunner",
    "Sm120H3QkvLandingKernel",
    "Tactic",
    "build_candidate",
]
