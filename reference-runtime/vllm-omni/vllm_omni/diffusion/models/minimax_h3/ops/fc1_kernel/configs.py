"""Bounded BF16 dense GEMM configuration space; standard-library CPU only."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    tile: tuple[int, int, int]
    epilogue_stages: int = 8


CONFIGS = {
    'm64n128_e8': Config((64, 128, 64)),
    'm128n64_e8': Config((128, 64, 64)),
    'm128n128_e8': Config((128, 128, 64)),
    'm64n128_e4': Config((64, 128, 64), 4),
    'm128n64_e4': Config((128, 64, 64), 4),
}
SHAPES = {'fc1': (11992, 28672, 5376), 'fc2': (11992, 5376, 14336)}
SEEDS = (1101, 2027, 3037)
MAX_ACTIVE_CLUSTERS = 110  # Previously read GB202 SM count; no query during cross compilation.
SMEM_CAPACITY = 102400


def stage_plan(config):
    m, n, k = config.tile
    epi_tile = min(64, m), min(32, n)
    epi_bytes = epi_tile[0] * epi_tile[1] * 2 * config.epilogue_stages
    ab_bytes = (m + n) * k * 2
    ab_stages = (SMEM_CAPACITY - 1024 - 1024 - epi_bytes) // ab_bytes
    assert ab_stages >= 2
    return dict(ab_stages=ab_stages, epilogue_stages=config.epilogue_stages,
                epilogue_tile=list(epi_tile), ab_bytes_per_stage=ab_bytes,
                staging_bytes=ab_bytes * ab_stages + epi_bytes,
                accumulators_per_mma_thread=m * n // 128)
