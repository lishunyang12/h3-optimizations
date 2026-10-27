# H3 Optimizations

H3 / FastH3 inference optimizations collected in one repository: **AdaLN reuse,
MXFP8 projections and SwiGLU fusion, Sage/VSA attention, VAE kernels and scheduling,
audio/video output overlap, and Ulysses/RDMA communication**.

[PRO 5000 data](benchmark/SM120/README.md) ·
[Run on PRO 5500](docs/PRO5500.md) ·
[Docker setup](docs/SM120_DOCKER.md) ·
[Intermediate stages](benchmark/SM120/TRADEOFFS.md) ·
[Setup](docs/CROSS_SKU_REPRO.md) ·
[Validation](docs/VERIFIED_CONTEXT.md)

## PRO 5000 measurements and PRO 5500 reproduction

The reproduction handoff covers **the deployed 5s avatar and the 15s FastH3
reference**. Both historical runs used eight GPUs, described in project records
as RTX PRO 5000-class Blackwell. Individual samples, model pins and provenance
are included in this repository.

| Workload | GPUs | Output at 24 fps | Steps | Historical HTTP median | Formal samples | PRO 5500 |
|---|---:|---|---:|---:|---:|---|
| Deployed 5s avatar | 8 | 960×544, 119 frames per continuation | 4 | **3.589779 s** | 13 | Pending |
| 15s FastH3 reference | 8 | 1280×704, 362 frames / 15.083333 s | 4 | **13.304 s** | 3 | Pending |

Avatar windows contain 124 frames with 5-frame overlap; the initial warmup output
has 124 frames. The table uses HTTP latency, not application-ready or playback
latency. Warmups are excluded. Some historical driver records report the generic
name `NVIDIA Graphics Device`; the [verified context](docs/VERIFIED_CONTEXT.md)
explains hardware attribution and timing boundaries.

| Deliverable | Repository entry |
|---|---|
| PRO 5000 reference measurements | [JSON](benchmark/SM120/results.json) · [CSV](benchmark/SM120/timings.csv) · [Explanation](benchmark/SM120/README.md) |
| PRO 5500 test entry point | [Docker walkthrough](docs/PRO5500.md#docker-image-recommended) · [Manual shell launcher](scripts/run_pro5500.sh) |
| Pinned workload settings | [`configs/sku-cases.json`](configs/sku-cases.json) |
| Machine-specific asset paths | Docker generates these automatically; [example JSON for manual setup](configs/sku-assets.example.json) |
| Prepare, run and compare | [Target-machine guide](docs/PRO5500.md) · [Full setup](docs/CROSS_SKU_REPRO.md) |
| Original receipts and checks | [Artifact audit](evidence/sku-evidence-audit.json) · [Package validation](evidence/sku-package-validation.json) |

With the [prebuilt Docker image](docs/SM120_DOCKER.md#get-the-prebuilt-image), follow
the [PRO 5500 Docker walkthrough](docs/PRO5500.md#docker-image-recommended). The
image includes both prepared runtimes and the Python/CUDA environment, and it
generates asset paths automatically. Prepare weights and caches once, mount them,
and use the container's `run` command. No host repository clone, runtime
preparation or manual asset JSON is needed.

For the **manual host-Python setup**, prepare the environment, weights and caches
using the [setup guide](docs/CROSS_SKU_REPRO.md), then edit the asset JSON:

```bash
cp configs/sku-assets.example.json /tmp/h3-assets.json
# Fill in the prepared runtime, interpreter, model, adapter, AdaLN and CUDA paths.
bash scripts/run_pro5500.sh \
  --assets /tmp/h3-assets.json --gpus 0,1,2,3,4,5,6,7 \
  --output /results/pro5500
```

Add `--dry-run` to preview without accessing GPUs. The target must satisfy the
8-GPU SM120, memory and RDMA requirements in the guide. The exact target hardware
and its results remain unverified. CPU and packaging validation have passed;
**fresh GPU runs of this launcher are pending on both systems**. Run the same
revision on PRO 5000 and PRO 5500 for a strict comparison.

The presentation follows [Sana SOL Engine](https://github.com/NVlabs/Sana/tree/sol-engine)
and [PR #511](https://github.com/NVlabs/Sana/pull/511): retained measurements,
fixed recipes, execution commands and validation scope. The runtime and
measurements here come from this repository's historical H3 experiments.
The [public data page](https://lishunyang12.github.io/fasth3-lossy-ablation/sku-reproduction.html)
also shows the two historical references.

## Other optimization experiments

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
| [`benchmark/SM120/`](benchmark/SM120) | Historical 5s avatar and 15s FastH3 samples, statistics and receipt provenance |
| [`scripts/`](scripts) | PRO 5500 handoff, shared cross-SKU harness, data export and source preparation |
| [`configs/sources.json`](configs/sources.json) | Upstream identities, exact port revisions and snapshot provenance |

The portable ports are already integrated in `vllm-omni/`; do not apply their
patches again. Experimental runtime code is retained separately because it has
additional features and different APIs. Do not mix modules from the two runtimes.

## DGX Spark quick start

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
