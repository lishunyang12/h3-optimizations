# Intermediate stages, checkpoints and tradeoffs

[Selected two-case reference](README.md) ·
[Historical receipts and source hashes](../../evidence/sku-tradeoff-history.json) ·
[Individual formal samples](tradeoff-timings.csv)

These are retained experiments on the historical eight-GPU reference system.
Only the selected 5s avatar and 15s FastH3 recipes are packaged for cross-SKU
reproduction. Intermediate arms below are historical evidence, not additional
qualified launcher profiles. No new GPU measurements are reported.

## 5s avatar: precision and residency

The September 21 trial used the same Ref2VA Turbo four-step model family,
124-frame windows, five-frame overlap, guide tau 0.8, NVFP4 video VAE and FP32
audio VAE. Each arm had three warmups and nine formal requests. Continuations
contain 119 frames / 4.958333 s at 24 fps. Every arm generates its own tails.

| Trial output | DiT projection weights | Attention | Historical HTTP median |
|---|---|---|---:|
| 1088×608 | First step BF16 staged from CPU; later MXFP8 | First step BF16; later Sage | 5.583326 s |
| 1088×608 | GPU-resident BF16 source; first step BF16, later MXFP8 | First step BF16; later Sage | 5.699592 s |
| 1088×608 | All steps BF16, GPU resident | First step BF16; later Sage | 6.568190 s |
| 1088×608 | All steps BF16, GPU resident | All steps BF16 | 7.343403 s |
| 960×544 | GPU-resident BF16 source; first step BF16, later MXFP8 | First step BF16; later Sage | 4.426763 s |
| 960×544 | All steps BF16, GPU resident | First step BF16; later Sage | 5.059722 s |

The resident path keeps **38,535,168,000 bytes of BF16 projection weights per
rank** on GPU and removes CPU weight staging. That is weight storage, not total
peak device memory. The retained peak-memory audit spans multiple arms, so it
does not establish a separate peak for each row.

These trials use a **different reference image and external reference audio**
from the selected deployed avatar. The deployed September 22 run uses prompt-only
native speech, all-step MXFP8/dense Sage and 960×544 output; its 13-formal-request
median is **3.589779 s**. Do not claim a matched speedup by dividing earlier BF16
timings by that number, even for rows with the same resolution.

All recorded trial outputs passed full audiovisual decode and projection
dispatch checks. Those checks do not establish perceptual quality, lip-sync or
long-term stability. The initial `probe-124` failed in the application controller
and is excluded from successful results. The public avatar quality comparison
uses a separate 10-second clip and is not substituted for this 5s workload.

## 15s FastH3: precision protection

Same fixed moon-teahouse prompt, seed 1101, four steps, TP1/SP8, NVFP4 video VAE,
1280×704 decoded output, 362 frames at 24 fps. Each arm has one warmup and three
formal complete-HTTP-to-MP4 samples.

| Precision schedule | Formal samples / s | Median / s |
|---|---|---:|
| All steps MXFP8 + Sage | 13.506 / 13.545 / 13.576 | 13.545 |
| First step BF16 DiT + BF16 VSA; later MXFP8 + Sage | 15.215 / 15.255 / 15.337 | 15.255 |
| BF16 output projections; other projections MXFP8 + Sage | 13.859 / 13.934 / 13.965 | 13.934 |

The BF16-first-step arm holds 24 blocks on GPU and stages 26 blocks from pinned
host memory; the transfer cost is included. Its recorded formal device peak is
71849 MiB, and its pinned BF16 host weights occupy 20,038,287,360 bytes per rank.
The output-projection-only arm keeps those BF16 weights on GPU. No uniform
per-arm peak-memory export is available here for a matched memory comparison.

Each arm repeats its own formal MP4 bytes. Precision protection changes the
outputs; these timing/dispatch receipts do not rank perceptual quality.

## 15s FastH3: changes that preserved the accepted output

| Matched test round | Parent median | Candidate median | Change | Output evidence |
|---|---:|---:|---|---|
| Cached normalization + early AAC | 13.551 s | **13.304 s** | 247 ms reduction | All four MP4s per arm equal the accepted quantized parent |
| Later output conversion fusion | 13.334 s | 13.261 s | 73 ms reduction | All four MP4s per arm equal that round's parent |

The selected packaged reference is **13.304 s**. The later 13.261 s result belongs
to a separate retained branch and is not a runnable third case in this package.
The 13.545 s precision baseline is another round; it is not used as the parent
for either speedup above. Byte equality to a quantized parent does not establish
equality to original BF16 inference. Full decode passed for these recorded clips.

## Checkpoint and intermediate-path map

| Stage | Deployed 5s avatar | Selected 15s FastH3 |
|---|---|---|
| Base weights | [Verified original-compatible Ref2VA download, revision 42ed227](https://huggingface.co/MiniMaxAI/MiniMax-H3/tree/42ed227ee7df40d41602854ae760620d6eb651fe/Ref2VA); historical revision `73372e6` is unavailable | [MiniMax-H3 FL2VA, revision 9bfb669](https://huggingface.co/MiniMaxAI/MiniMax-H3/tree/9bfb6693f2cf6de171db46d1aa586f67d773a1da/FL2VA) |
| Four-step adapter | [LightX2V Turbo v0.1](https://huggingface.co/lightx2v/Minimax-h3-Turbo/tree/3ec17a324ced54151364f24f8b5fb6bf7e26414f) | [FastH3 Preview LoRA, vsa-datafree](https://huggingface.co/FastVideo/FastVideo-FastH3-4-step-Preview-v1-LoRA/tree/c0e8ec2ac460e801094a0abe74bfb57347388ce3/vsa-datafree) |
| Prepared runtime | `/opt/h3-runtime-avatar5` | `/opt/h3-runtime-15s` |
| AdaLN sidecar | `/models/guide-adaln.safetensors`; 34 recorded plans | `/models/fasth3-adaln.safetensors`; adapter-bound v3 |
| Precision changes | Same base and adapter; runtime BF16/MXFP8 dispatch | Same base and adapter; runtime BF16/MXFP8 dispatch |
| Decode/output changes | NVFP4 conversion and scheduling in runtime | NVFP4 conversion, cached normalization and AAC scheduling in runtime |

Exact adapter filenames, revisions and SHA-256 values are in the
[case profiles](../../configs/sku-cases.json) and
[15s profile](../../configs/fasth3-15s-4step-reference.json). The
[setup guide](../../docs/CROSS_SKU_REPRO.md) creates the model views and sidecars.
The [avatar download instructions](../../docs/CROSS_SKU_REPRO.md#5s-avatar-verified-download-instructions)
explain the verified retrieval revision and required revision pins. The prepared
metadata paths are `/models/avatar-assets/model-assets.json` and
`/models/fasth3-assets/model-assets.json`.

The retained intermediate precision modes do **not** establish separately
trained checkpoint files to download. Quantization occurs in the runtime.
Avatar rolling-continuation checkpoints store session state; they are not model
weights or portable quality baselines. Only paths with retained evidence are
listed; private local snapshot names are not presented as public downloads.

## Cost interpretation

At the same eight-GPU hourly price, request GPU-seconds are `8 × HTTP latency`.
If each GPU costs `P` per hour, the request-only cost estimate is
`8 × latency_seconds × P / 3600`. This excludes model loading, warmup, idle time,
CPU/RAM, storage and networking. No price or throughput measurement was recorded,
so these data do not establish a monetary winner across SKUs.

Compare latency within each matched group, retain the output-quality constraints,
and measure both target SKUs with the same packaged recipe. Recompute the
historical CSV and check all 14 arms without accessing GPUs:

```bash
python3 scripts/export_sku_tradeoffs.py
python3 scripts/export_sku_tradeoffs.py --check
```
