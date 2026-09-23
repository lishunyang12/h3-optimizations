# H3 Optimizations

H3 / FastH3 inference optimizations collected in one repository: **AdaLN reuse,
MXFP8 projections and SwiGLU fusion, Sage/VSA attention, VAE kernels and scheduling,
audio/video output overlap, and Ulysses/RDMA communication**.

For DGX Spark single-GPU experiments, start with the
[Spark guide](docs/DGX_SPARK.md) and [optimization map](docs/OPTIMIZATIONS.md).
The launcher provides separate baseline, AdaLN, MXFP8 and audio-overlap profiles.
**This combined snapshot has not been validated on DGX Spark.** Original SM120
multi-GPU timings are not Spark performance claims; several experimental kernels
and VAE schedules have explicit architecture or shape restrictions.

中文：这里集中公开 H3 优化源码、定制依赖、分项补丁和测试。Spark 单卡优先试
AdaLN 缓存，再测试 MXFP8/SwiGLU 和音视频解码重叠；SM120 专用注意力内核需要
先做 SM121 移植验证。单卡入口关闭多卡通信，不启用 CPU/DLO offload。

## Contents

| Directory | Contents |
| --- | --- |
| [`vllm-omni/`](vllm-omni) | Complete vLLM-Omni integration, including H3 optimization ports, tests, benchmarks and recipes |
| [`flashinfer/`](flashinfer) | Actual custom FlashInfer Python, CUDA and C++ sources, including sparse attention and RDMA |
| [`fastvideo-kernel/`](fastvideo-kernel) | Optional FastVideo sparse-attention kernel sources |
| [`reference-runtime/`](reference-runtime/README.md) | Full experimental runtime, additional kernels and original microbenchmarks |
| [`patches/`](patches/README.md) | Individual AdaLN/MXFP8/Sage/VAE/RDMA diffs and full-length runtime variant |
| [`scripts/`](scripts) | Source verification, environment report, Spark launcher and HTTP benchmark |
| [`configs/sources.json`](configs/sources.json) | Upstream identities, exact port revisions and snapshot provenance |

The portable ports are already integrated in `vllm-omni/`; do not apply their
patches again. Experimental runtime code is retained separately because it has
additional features and different APIs. Do not mix modules from the two runtimes.

## Quick start

```bash
git clone https://github.com/lishunyang12/h3-optimizations.git
cd h3-optimizations
python3 scripts/verify_sources.py
python3 scripts/check_environment.py
# Preview only; no model loading or GPU access:
python3 scripts/spark.py --model /path/to/MiniMax-H3/FL2VA --profile adaln
```

Follow [installation and experiments](docs/DGX_SPARK.md) before adding `--run`.
The H3 integration requires a matching vLLM/PyTorch environment; this repository
does not supply binary wheels, model weights or a universal environment lockfile.
See [validation](docs/VALIDATION.md) for checks performed for this release.

## License

Apache-2.0 for this project's packaging and optimization contributions, with
original third-party notices and licenses preserved. See [LICENSE](LICENSE),
[NOTICE](NOTICE) and [third-party provenance](docs/THIRD_PARTY.md).
Model weights, private reference media, generated videos and credentials are not
distributed. Obtain model weights separately under their source terms.
