# Optimization map

Paths below are relative to `vllm-omni/` unless marked `reference-runtime`.
“Candidate” means code is available to test; it does not assert an SM121 speedup.

| Optimization | Source entry | Spark applicability |
| --- | --- | --- |
| Exact AdaLN projection reuse | `vllm_omni/diffusion/models/minimax_h3/adaln_cache.py` | First single-GPU candidate; enabled by default, baseline profile disables it |
| DiT CUDA MXFP8 | `vllm_omni/diffusion/layers/mxfp8.py`, `models/minimax_h3/quantization.py` under `diffusion/` | Candidate; requires Blackwell block-scaled GEMM support in PyTorch; approximate quantization |
| SwiGLU → MXFP8 producer fusion | `vllm_omni/diffusion/layers/mxfp8.py` | TP1 path; compare fused and unfused byte outputs on the target device |
| Sage/VSA attention and quantization | `vllm_omni/diffusion/attention/ops/sage_block_sparse_attention.py`, `sage_quantization.py` | Bundled custom kernel targets SM120; Spark SM121 requires qualification/porting |
| VAE MXFP8 projections | `vllm_omni/diffusion/models/minimax_h3/quantization.py` | Separate approximate option; not enabled by the launcher |
| Paired/mixed VAE decode | `vllm_omni/diffusion/models/minimax_h3/vae_batching.py` | Fixed 1280×704 / 362-frame geometry, distributed VAE; not a general Spark option |
| Video/audio decode overlap | `vllm_omni/diffusion/models/minimax_h3/pipeline_minimax_h3.py` | Single-GPU candidate; additional concurrent memory usage must be measured |
| Ulysses/RDMA transport | `vllm_omni/diffusion/distributed/flashinfer_ulysses.py` | Multi-GPU/multi-rank only; unused by Spark single-GPU profiles |

Additional experimental implementations are in
`reference-runtime/vllm-omni/vllm_omni/diffusion/`:

- `models/minimax_h3/cached_output_normalize.py`: cached VAE output constants.
- `models/minimax_h3/early_audio_encode.py`: early AAC encoding.
- `models/minimax_h3/chunked_cpu_output.py`: chunked CPU MP4 output.
- `models/minimax_h3/online_nvfp4.py`: experimental NVFP4 VAE linear conversions.
- `models/minimax_h3/{mixed_vae,paired_vae,mxfp8_experiment}.py`: original experiment paths.
- `attention/ops/`: CAKE descriptor ABI, norm/RoPE and output producer kernels,
  fine attention pipeline, communication overlap and scheduling experiments.

These modules are research source, with hardware/shape guards and historical
assumptions. Read their guards before selecting a candidate; copying a kernel
without its quantization, stream and layout contracts is insufficient.

AdaLN reuse preserves repeated projection results under its cache identity
checks. MXFP8/NVFP4/Sage are precision changes; distillation and VSA masking also
change the computation. Fusion parity versus an unfused quantized path does not
establish equivalence to dense BF16 model output.
