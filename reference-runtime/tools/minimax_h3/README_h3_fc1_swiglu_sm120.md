# MiniMax-H3 SM120 FC1 + SwiGLU probe

This directory contains a safe-by-default, exact-shape benchmark scaffold for
the original BF16 MiniMax-H3 TP1/SP8 path and an isolated experimental CuTe DSL
candidate. The candidate is not wired into production. The default `check`
action does not initialize a CUDA context or launch GPU work.

The production shape on each SP rank is:

- input `A`: `[11936, 5376]` BF16;
- FC1 weight: `[28672, 5376]` BF16, laid out as all 14336 gate rows followed by
  all 14336 up rows;
- materialized FC1 output: `[11936, 28672]` BF16;
- SwiGLU output: `[11936, 14336]` BF16;
- optional FC2 weight/output: `[5376, 14336]` and `[11936, 5376]` BF16.

The standalone path writes 652.75 MiB of gate/up data, reads it again, then
writes 326.375 MiB of activation data. A real epilogue fusion avoids 1.275 GiB
of HBM traffic per layer and one kernel launch.

## Safe preflight

```bash
python tools/minimax_h3/benchmark_h3_fc1_swiglu_sm120.py check \
  --json-out /tmp/h3-fc1-swiglu-check.json
```

`--probe-device` may be added to inspect compute capability, but it can
initialize the CUDA driver. It still launches no kernels.

After GPU ownership and the candidate have been reviewed, run the baseline:

```bash
python tools/minimax_h3/benchmark_h3_fc1_swiglu_sm120.py run \
  --warmup 20 --iterations 100 --downstream-fc2 \
  --json-out /tmp/h3-fc1-swiglu-baseline.json
```

The script refuses non-SM120 devices. It reports per-call median and
nearest-rank p95, peak allocated/reserved memory, analytical bytes/FLOPs, and
the projected 50-layer/49-update saving. The optional FC2 measurement catches
an apparent FC1 win that damages the following GEMM through layout or cache
effects.

## Candidate API

A reviewed candidate is supplied using `--candidate-module package.module` or
`--candidate-module /absolute/path/candidate.py`. The module must export:

```python
CANDIDATE_CONTRACT = {
    "api_version": 1,
    "name": "descriptive-kernel-name",
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
}


def build_candidate(*, weight, shape, device):
    # Compilation and one-time weight packing happen here, outside timing.
    # Return callable runner(x) -> contiguous BF16 [M, Npair].
    ...
```

Metadata is only a review guard. The harness runs `torch.equal` against the
actual vLLM `_C.silu_and_mul` baseline before timing. A mismatch writes a JSON
report, skips timing, and exits with status 3. Max/mean absolute and max
relative errors are recorded for diagnosis, but tolerance does not qualify as
strictly lossless.

The bundled experimental module is selected explicitly. Importing or running
the normal server does not load it:

```bash
H3_FC1_SWIGLU_TILE_MNK=64,64,64 \
python tools/minimax_h3/benchmark_h3_fc1_swiglu_sm120.py run \
  --m 256 --k 256 --n-pair 256 --warmup 20 --iterations 100 \
  --candidate-module tools/minimax_h3/h3_fc1_swiglu_sm120_candidate.py \
  --json-out /tmp/h3-fc1-swiglu-small.json
```

Start with the small gate above, then remove the three shape overrides for the
real `[11936,5376] x [28672,5376]` case. Supported exploratory CTA tiles are
`64,64,64`, `64,128,64`, and `128,64,64`. Every variant still has to pass
bitwise equality before the harness will time it. Compilation is outside the
timed region. Use Nsight Systems to confirm that each candidate call contains
exactly one CUDA kernel launch.

## CuTe implementation boundary

The candidate must be a paired FC1 epilogue, not a wrapper around two kernels:

1. Preserve the checkpoint's `[gate rows][up rows]` layout; optional prepacking
   must happen once, never in the measured call.
2. For a given output tile, accumulate the corresponding gate and up tiles.
3. Convert each accumulator value to BF16 before activation, matching the
   existing FC1 output boundary. Re-promote those BF16 values as required by
   the scalar activation implementation.
4. Round the SiLU result itself to BF16 before multiplying by BF16 `up`, matching
   vLLM's packed BF16 activation temporary, then round/store a contiguous BF16
   `[M, Npair]` result.
5. Confirm one CUDA kernel per call with Nsight Systems; the Python metadata is
   not proof of fusion.

The installed `nvidia-cutlass-dsl==4.6.2` and FlashInfer wheel provide the
SM120 GeForce dense GEMM reference used by the candidate:

- `blackwell_geforce/dense_gemm.py` for SM120 warp MMA, TMA and persistent
  scheduling. This is the relevant path for SM120; it does not require the
  SM100 tcgen05/TMEM machinery.

The local FlashInfer source also provides useful pair-aware structure:

- `blackwell/blockscaled_contiguous_gather_grouped_gemm_act_fusion.py` for a
  pair-aware activation epilogue;
- `blackwell_sm12x/moe_micro_kernel.py` and `moe_static_kernel.py` for SM120
  warp-MMA scheduling and paired accumulator patterns.

The MoE kernels use quantized operands and small routed working sets. They
must not be exposed as a BF16 dense H3 candidate without replacing their
operand, accumulation, rounding, and output contracts.

## Production patch plan

Only after both the exact-shape correctness and speed gates pass:

1. Put the CuTe implementation behind a new isolated diffusion custom op with
   an eager fallback to the current FC1 plus `SiluAndMul` path.
2. Prepack each block's FC1 weight once after checkpoint loading; never prepack
   inside `forward`.
3. Dispatch only for CUDA SM120, BF16, TP1, no FC1 bias, contiguous input, and
   supported fixed shapes. Fail closed to the existing path otherwise.
4. Change only `MiniMaxH3MLP.forward` at first; do not combine FC2 until this
   boundary independently passes.
5. Gate integration on at least 0.2 ms/layer median saving, no downstream-FC2
   regression, and no material increase in persistent/peak memory.
6. Re-run correctness at activation, whole block, single denoise update, and
   decoded RGB/PCM levels before the 50-step benchmark.

Full FC1 + SwiGLU + FC2 fusion is intentionally out of scope. At H3's large M,
an activation K-slice must feed roughly 42 FC2 N tiles; retaining all FC2
accumulators is infeasible, while recomputing FC1 or using global atomic
partials defeats the strict, lossless objective.

## SM120 exact-shape result (2026-09-05)

The isolated candidate passed strict bitwise correctness on physical GPU 0 at
the full H3 shape for every supported tile: `torch.equal=True`, with zero
maximum, mean, and relative error. It also reduced peak extra allocation from
1,026,686,976 bytes to 342,228,992 bytes, a saving of 684,457,984 bytes
(652.75 MiB). Candidate compilation temporarily peaked 470,564,864 bytes above
its starting allocation and retained no persistent allocation.

The speed gate failed, so this version is rejected for production:

| CTA tile | Baseline median | Candidate median | Candidate delta |
| --- | ---: | ---: | ---: |
| `64x64x64` | 20.225 ms | 23.415 ms | +3.190 ms |
| `64x128x64` | 20.178 ms | 21.130 ms | +0.952 ms |
| `128x64x64` | 19.975 ms | 23.399 ms | +3.424 ms |

Each row used three warmups and ten interleaved measurements. The best tile was
still about 4.7% slower than its same-run baseline. Per the early-stop rule, no
Nsight trace or production dispatch was added. The likely next design, if this
experiment is revisited, is not another tile sweep: it must remove the second A
tile load and/or reduce the two live accumulator fragments enough to enable a
larger output tile without register spilling. Those are new kernel designs and
must repeat the strict correctness gate.
