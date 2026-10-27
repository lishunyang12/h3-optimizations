# PRO 5000 reference data

[Run on PRO 5500](../../docs/PRO5500.md) · [JSON data](results.json) ·
[Individual samples CSV](timings.csv) · [Original artifact audit](../../evidence/sku-evidence-audit.json)

Two historical measurements from the system described in project records as
RTX PRO 5000-class Blackwell. Each workload used **8 GPUs**. Some raw driver
inventories use the generic name `NVIDIA Graphics Device`; see the
[hardware attribution](../../docs/VERIFIED_CONTEXT.md#hardware-attribution-and-comparison-limits).
These data are retained measurements, not fresh runs of the packaged launcher.

## Measurements

| Case | Output | Actual steps | Warmup / measured | Historical HTTP median | Target PRO 5500 |
|---|---|---:|---:|---:|---|
| Deployed 5s avatar | 960×544, 24 fps; 119 frames per measured continuation | 4 | 3 / 13 | **3.589779 s** | Pending |
| 15s FastH3 reference | 1280×704, 24 fps; 362 frames / 15.083333 s | 4 | 1 / 3 | **13.304 s** | Pending |

The avatar uses 124-frame windows with a 5-frame overlap. Its first, warmup output
has 124 frames; later segments have 119 frames / 4.958333 seconds. The saved
16-request sequence contains seeds 8300–8315 and is replayed without changing the
prompts or reference image. Its separate application-ready median is 3.706288 s.

The 15s request asks for 1280×720; the actual decoded output is 1280×704.
Its three formal samples are 13.281, 13.304 and 13.348 s. **13.311 s is the mean**;
13.304 s is the median. This is the selected published reference, not a claim
about every optimization branch or every prompt.

## Fixed recipes

| Component | Deployed 5s avatar | 15s FastH3 |
|---|---|---|
| Base partition | MiniMax-H3 Ref2VA | MiniMax-H3 FL2VA |
| Adapter | LightX2V Turbo 4-step | FastVideo FastH3 4-step Preview LoRA |
| DiT / attention | MXFP8 / dense Sage | MXFP8 / Sage-VSA |
| Video / audio decoder | Full NVFP4 video VAE / FP32 audio VAE | Same |
| Parallelism | TP1/SP8, text encoder TP8, VAE PP8 | Same |
| Communication | BF16 RDMA | Same |
| Seed | 8300–8315 | 1101 |
| Runtime preparation | `--variant avatar5` | `--variant reference` |

[Case profiles](../../configs/sku-cases.json) and the
[15s profile](../../configs/fasth3-15s-4step-reference.json) contain the exact
source/model revisions, schedules, launch flags and dependency versions.
The [setup guide](../../docs/CROSS_SKU_REPRO.md) prepares the weights and caches.

## What is in the data files

- `results.json`: both sample arrays, recomputed means/medians, model pins,
  topology, media contract, receipt hashes and original artifact hashes.
- `timings.csv`: all 16 avatar requests, including the 3 labeled warmups, and
  the 3 formal FastH3 samples. Avatar application-ready timing is a separate column.
  The retained FastH3 qualification receipt does not supply a warmup latency;
  the CSV does not invent one.
- [Avatar receipt](../../evidence/sku-avatar5-history.json): per-request timing,
  recorded frames and MP4 hashes.
- [FastH3 receipt](../../evidence/sku-15s-history.json): full-decode status,
  output hashes, execution checks and equality to the accepted quantized parent.

The files are regenerated from the receipts, with checks against the independent
artifact audit. To verify or export them without accessing a GPU:

```bash
python3 scripts/export_pro5000_reference.py --check
python3 scripts/export_pro5000_reference.py --output /tmp/pro5000-reference
```

## Validation and comparison

The existing package has six passing CPU tests, checked historical dispatch
records, source reconstruction and local HTTP/media validation. The export and
target launcher are checked without model execution. Packaging checks do not
constitute a new GPU benchmark. See the
[validation record](../../evidence/sku-package-validation.json) and
[verified context](../../docs/VERIFIED_CONTEXT.md).

For an apples-to-apples comparison, rerun both workloads on the reference and
target systems with the same launcher revision, model/cache bytes and GPU count.
The historical JSON is deliberately labeled `historical_reference_only`; it
cannot stand in for a new `result.json` in the strict comparator.
