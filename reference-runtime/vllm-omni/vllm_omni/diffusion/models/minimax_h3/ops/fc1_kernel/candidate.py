"""Isolated plain BF16 GEMM; original K-loop, MMA and epilogue unchanged."""
import hashlib
import json
from pathlib import Path

import cutlass
import cutlass.cute as cute

from .configs import CONFIGS
from .dense_reference import Sm120GemmKernel

ROOT = Path(__file__).resolve().parent


def verify_reference():
    from .qualification import DENSE_REFERENCE_SHA256, CONFIGS_SHA256
    for name, digest in (("dense_reference.py", DENSE_REFERENCE_SHA256), ("configs.py", CONFIGS_SHA256)):
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest:
            raise RuntimeError("Private qualified FC1 source changed: " + name)


class ConfiguredGemm(Sm120GemmKernel):
    def __init__(self, name):
        verify_reference()
        config = CONFIGS[name]
        super().__init__(cutlass.Float32, config.tile)
        self.config_name = name
        self.epilogue_stage_override = config.epilogue_stages

    def _compute_stages(self, tile_shape_mnk, a_dtype, b_dtype, epi_tile,
                        c_dtype, smem_capacity, occupancy):
        # Same capacity formula and alignment reserves as the vendor parent.
        # The e8 variants return exactly the original stage choices.
        epi_stage = self.epilogue_stage_override
        epi_bytes = cute.size(epi_tile) * c_dtype.width // 8 * epi_stage
        a_shape = cute.slice_(tile_shape_mnk, (None, 0, None))
        b_shape = cute.slice_(tile_shape_mnk, (0, None, None))
        ab_bytes_per_stage = (cute.size(a_shape) * a_dtype.width // 8
                              + cute.size(b_shape) * b_dtype.width // 8)
        ab_stage = ((smem_capacity - occupancy * 1024) // occupancy
                    - 1024 - epi_bytes) // ab_bytes_per_stage
        if ab_stage < 2:
            raise ValueError('v24 requires at least two AB stages')
        return ab_stage, epi_stage
