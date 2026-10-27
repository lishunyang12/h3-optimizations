# H3 Optimizations

H3 / FastH3 inference optimizations collected in one repository: **AdaLN reuse,
MXFP8 projections and SwiGLU fusion, Sage/VSA attention, VAE kernels and scheduling,
audio/video output overlap, and Ulysses/RDMA communication**.

For the deployed **5s avatar** and **15s FastH3** cross-SKU reproduction, start
with the [reproduction guide](docs/CROSS_SKU_REPRO.md),
[pinned cases](configs/sku-cases.json), and
[public test data](https://lishunyang12.github.io/fasth3-lossy-ablation/sku-reproduction.html).
Both references used **8 GPUs**, described in the historical project records as
RTX PRO 5000-class Blackwell. See the [verified context](docs/VERIFIED_CONTEXT.md)
for measurement boundaries and hardware identification limits. The new launcher has CPU and
historical-log validation; fresh 5000/5500 GPU runs remain pending.

For DGX Spark single-GPU experiments, start with the
[Spark guide](docs/DGX_SPARK.md) and [optimization map](docs/OPTIMIZATIONS.md).
The launcher provides separate baseline, AdaLN, MXFP8 and audio-overlap profiles.
**This combined snapshot has not been validated on DGX Spark.** Original SM120
multi-GPU timings are not Spark performance claims; several experimental kernels
and VAE schedules have explicit architecture or shape restrictions.

For Spark, test AdaLN caching before MXFP8/SwiGLU and audiovisual decode overlap.
SM120-specific attention kernels require a separate SM121 port and validation.
The single-GPU launcher disables multi-GPU communication and CPU/DLO offload.

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
Model weights, generated videos and credentials are not distributed here.
The fixed avatar benchmark reference image is included under `repro/inputs/`;
obtain model weights separately under their source terms.
