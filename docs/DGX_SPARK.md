# DGX Spark: single-GPU experiments

This guide is an experiment entry point, not a claim that the bundled integration
has been run on Spark. NVIDIA lists GB10 as compute capability **12.1**; compile
for the actual device, not an inherited SM120 setting. See the
[NVIDIA capability table](https://developer.nvidia.com/cuda/gpus) and
[Spark build guide](https://docs.nvidia.com/dgx/dgx-spark-porting-guide/porting/compilation.html).

The upstream [H3 Spark recipe](../vllm-omni/recipes/MiniMaxAI/MiniMax-H3-Spark-GB10.md)
provides a separately pinned reference configuration and historical measurements.
Its unified-memory guidance applies here: keep one model partition resident,
quantize to fit, and do not use CPU or distributed layerwise offload to try to
gain capacity in the shared CPU/GPU memory pool. Our launcher fixes all
parallelism degrees to one and uses cuDNN attention.

## 1. Prepare the environment

Use an isolated ARM64 environment with CUDA-enabled PyTorch, cuDNN, vLLM and
FlashInfer built for GB10. The integration was checked locally with Python 3.12,
vLLM 0.29.0 and PyTorch 2.13.0+cu132. Those are **observed x86_64 test versions**,
not a verified ARM64 wheel installation recipe. Consult the bundled
[installation documentation](../vllm-omni/docs/getting_started/installation/README.md)
for platform-specific dependencies. FFmpeg/ffprobe are needed for output checks.

Once the matching platform stack is installed, from this repository root:

```bash
python scripts/verify_sources.py
python -m pip install setuptools wheel setuptools-scm
# --no-deps preserves the platform-specific stack you prepared above.
# It does not install missing dependencies: check the requirements/ directory.
VLLM_OMNI_VERSION_OVERRIDE=0.29.0.dev0+h3snapshot \
  python -m pip install --no-deps --no-build-isolation -e ./vllm-omni
python -m pip check
python scripts/check_environment.py --gpu
```

Resolve missing dependencies and ABI errors before model loading. An editable
install may regenerate `vllm_omni/_version.py`, causing its source hash to change.
The runtime entry point always selects this repository's `vllm-omni/` source.

Download or reuse MiniMax-H3 weights separately. `--model` must point directly
at a local `FL2VA` or `Ref2VA` partition, not its parent. This guide starts with
base H3; it does not silently select a Turbo or FastH3 adapter.

## 2. Establish an FP8 baseline

Preview the command, then explicitly start the server:

```bash
python scripts/spark.py --model /path/to/MiniMax-H3/FL2VA --profile baseline
python scripts/spark.py --model /path/to/MiniMax-H3/FL2VA --profile baseline --run
```

Wait for server startup and a successful `http://127.0.0.1:8000/health` response.
From another terminal, use a new results directory for every run:

```bash
python scripts/benchmark_spark.py --output results/baseline-r1
```

The benchmark uses a fixed prompt/seed, 960×576, 4 seconds, 10 steps, one warmup
and three measured requests. It saves MP4s, selected timing/memory headers,
ffprobe metadata when available, and all measured times. Timing ends when the
complete HTTP body is saved; ffprobe runs afterward. A short-step request is a
runtime smoke test, not a full-quality model evaluation.

The baseline explicitly uses **DiT FP8**, with text encoder and VAEs unquantized,
and turns AdaLN reuse off. Thus precision stays constant in the AdaLN A/B test.
Save the launch JSON and environment report with the benchmark results.

## 3. Test one change at a time

Stop your baseline server before starting another model instance.

| Profile | Difference | Comparison |
| --- | --- | --- |
| `baseline` | DiT FP8; AdaLN reuse disabled | Reference |
| `adaln` | Enable exact AdaLN runtime cache | Compare against baseline at identical precision |
| `mxfp8` | AdaLN enabled; replace DiT FP8 with MXFP8, including TP1 SwiGLU fusion | Compare against adaln; check output quality as precision changes |
| `audio-overlap` | AdaLN enabled; DiT FP8; overlap audio and video VAE decode | Compare against adaln; track shared-memory use |

For example:

```bash
python scripts/spark.py --model /path/to/MiniMax-H3/FL2VA --profile adaln --run
# In another terminal:
python scripts/benchmark_spark.py --output results/adaln-r1
```

Run Ref2VA by selecting the `Ref2VA` partition and passing
`--reference /path/to/reference.jpg` to the benchmark. Stop FL2VA first. Reuse the
same reference bytes for every comparison. Check visual/audio quality as well as
latency, and observe `/proc/meminfo` on Spark because allocator peaks alone do not
account for the whole shared memory pool. Repeat A/B in alternating order to
separate sustained-load/thermal effects from the optimization.

The launcher clears inherited H3/Ulysses tuning and architecture overrides,
then applies only the chosen profile. It does not select experimental Sage,
RDMA, NVFP4 VAE, fixed-shape paired VAE or offload options.

## 4. Start with a standalone non-communication kernel

MXFP8 needs a PyTorch build providing
`torch.nn.functional.scaled_mm` with block-scaled recipes, plus a compatible CUDA
library implementation. The environment report only checks API presence; it
cannot establish runtime compatibility. This microbenchmark requires no weights:

```bash
PYTHONPATH="$PWD/vllm-omni" python vllm-omni/benchmarks/diffusion/bench_mxfp8_linear.py \
  --rows 128 --hidden 1024 --output 512 --repeats 30
```

It verifies fused/unfused equality, warms up, and reports CUDA-event timings.
If this small check passes, repeat with representative H3 shapes. Equality here
is with the unfused quantized path, not dense BF16. Do not use a small synthetic
GEMM result as a full-model speedup claim.

## 5. Architecture-specific sources

The custom FlashInfer source is bundled for completeness. Default Spark profiles
use the compatible FlashInfer installation in your environment. To work on the
bundled experimental kernels, fetch the pinned header dependencies with:

```bash
python scripts/prepare_flashinfer.py
# For a standalone development command, opt in via:
# PYTHONPATH="$PWD/vllm-omni:$PWD/flashinfer" python your_kernel_test.py
```

The generated CAKE JIT manifest selects `sm_120a` and SM120 compiler flags. An API
accepting capability family 12 does not prove its generated binary supports
SM121. Port and validate that path before selecting Sage on Spark. Similarly,
`fastvideo-kernel/` contains architecture-specific sources; it is not installed
by default. See [optimization map](OPTIMIZATIONS.md) for other restrictions.

The original full-length and short-window research runtimes are available in
`reference-runtime/`; their historical scripts are not the Spark launch path.
