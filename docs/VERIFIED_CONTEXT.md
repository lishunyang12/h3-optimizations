# Verified context: the deployed 5s avatar and the 15s FastH3 reference

Recorded on 2026-10-27. This document covers the two requested reproductions.
It distinguishes saved measurements from package validation and unperformed tests.
The supporting [artifact audit](../evidence/sku-evidence-audit.json) records the
original result-file hashes and checked samples. It is not a claim that every
historical session or every optimization experiment has been reviewed.

## Scope and status

The requested deliverable is an English reproduction package and public data
page for the deployed 5s avatar and the 15s FastH3 reference, so a colleague can
run the same workload on another GPU SKU. "Cross-SKU" means separate runs on
homogeneous machines with the same GPU count. Both selected cases use eight GPUs.

The source reconstruction, CPU checks and historical dispatch checks have
passed. No new GPU latency measurements have been produced by this launcher.
The target called "5500" has no verified hardware inventory or benchmark result
in this release. Both the reference machine and target machine need a fresh run
of the same revision. See the [execution guide](CROSS_SKU_REPRO.md).

## Two distinct workloads

| Property | Deployed 5s avatar | 15s FastH3 reference |
|---|---|---|
| Case | `5s-avatar-ref2va-4step` | `15s-4step-mxfp8-sage-nvfp4` |
| Historical run | September 22, `probe-v4-124` | September 18, `aac-r1` |
| Model route | MiniMax-H3 Ref2VA + LightX2V Turbo 4-step adapter | MiniMax-H3 FL2VA + FastVideo FastH3 4-step Preview LoRA |
| Task | Stateful reference-conditioned avatar continuation | Text-to-video-and-audio, independent requests |
| Output | 960×544, 24 fps | 1280×704, 24 fps |
| Request duration | 124-frame window; 5-frame overlap | 15 seconds requested |
| Actual output duration | Initial 124 frames / 5.166667 s; later 119 frames / 4.958333 s | 362 frames / 15.083333 s |
| Actual denoising steps | 4 | 4 |
| DiT and attention | MXFP8 projections, dense Sage | MXFP8 projections, Sage/VSA |
| Video decoder | Full NVFP4 VAE | Full NVFP4 VAE |
| Audio decoder | FP32 audio VAE | FP32 audio VAE |
| Parallelism | 8 GPUs; TP1/SP8, encoder TP8, VAE PP8 | Same |
| Transport | BF16 RDMA | BF16 RDMA |
| Warmup / measured requests | 3 / 13, one saved sequence | 1 / 3 |
| Historical HTTP median | **3.5897789399605244 s** | **13.304 s** |
| Seed | 8300 through 8315 | 1101 |

The 15s request asks for 1280×720; the historical model produces 1280×704.
The 5s label denotes the avatar window. Its measured continuation segments are
119 frames, not 124 frames. Preserve these dimensions and durations when comparing.

## Deployed avatar evidence

The saved runtime is `short-window-20260922/native-v3`. Its 17 changed Python
files are preserved in [the avatar overlay](../repro/avatar5-overlay), with
[checksums](../repro/variants.json). All sixteen packaged
[requests](../repro/avatar5-requests.json) were compared with the original request
JSON files, including prompts, seeds and continuation parameters.

The original reference image shows the open-window study. It is uploaded twice
per request. Its SHA-256 is
`41f1f15af74b88d9b11117bc73b0af560f9128aeb32c0093f0b92e019e826376`.
The sequence contains reading, speaking and page-turn actions, uses one
continuation session and preserves all warmup requests. Every saved request in
this run reports one generation attempt. Native H3 audio is conditioned by the
prompt; there is no external voice-reference audio in these requests.

The [saved timing records](../evidence/sku-avatar5-history.json) include all 16
outputs and identify warmup explicitly. Recomputing the thirteen measured
`http_seconds` values gives **3.5897789399605244 s**. Recomputing `ready_seconds`
gives **3.7062881940510124 s**. The latter includes application media/seam checks.
The new reproduction measures HTTP generation and complete MP4 delivery, not
application-ready time, chat response latency, RTMP or playback delay.

The user-supplied [avatar quality page](https://lishunyang12.github.io/fasth3-lossy-ablation/avatar-quality.html)
belongs to the same deployment family but is a separate experiment: it requests
10 seconds and produces 243 frames / 10.125 seconds, with a different reference
image. Its deployed-model measurement of **6.993 s** must not be presented as the
5s avatar latency. That page remains a quality reference; it is not a third case
in this release.

## 15s FastH3 evidence

The selected reference is the full NVFP4 VAE configuration with cached output
normalization and early AAC. It is reconstructed from the September 23 public
snapshot `090f41b83156ae61e6977e7d6069fd36b8fe584c`, using
`reference-runtime/` and `patches/ref2va-to-full-length-runtime.patch`.
The newer general integration directory is a different code path.

The original [qualification receipt](../evidence/sku-15s-history.json) records
formal HTTP times of **13.281, 13.304 and 13.348 s**. Their median is **13.304 s**;
their mean is **13.311 s**. Both are below the 15.083333-second video playback
length for this particular workload and eight-GPU setup. This does not establish
single-GPU performance, arbitrary-prompt latency or production tail latency.

The receipt reports all four outputs, including warmup, passing full audiovisual
decoding. Its `outputs_byteexact_to_fastest_baseline` entries are all true:
this compares against the accepted quantized parent configuration. Other legacy
`baseline_mp4_byteexact` fields refer to a different baseline and are false.
Neither field establishes equality with the original dense BF16 model.

The quantized route uses a distilled four-step student, 200 MXFP8 projection
modules, BF16 gate computation, and a full NVFP4 video VAE with 144 converted
layers per rank. Communication uses BF16, not FP8 transport. Loading, JIT and
warmup are outside formal request timing. The run used unlocked node clocks and
an exclusive measurement barrier. The original summary's 63099 MiB device-memory
peak is about **61.62 GiB**, not 63.1 GiB; it is a sampled historical peak, not a
portable minimum-capacity guarantee.

This is the selected published reference, not a claim of the fastest result in
all historical branches. Additional optimization variants are outside this
reproduction's scope.

## Exact model and schedule identities

| Component | Avatar | FastH3 |
|---|---|---|
| Base repository | `MiniMaxAI/MiniMax-H3` | `MiniMaxAI/MiniMax-H3` |
| Base revision | `73372e6cf53e414edd3ab03e357717fb0602e758` | `9bfb6693f2cf6de171db46d1aa586f67d773a1da` |
| Partition | `Ref2VA` | `FL2VA` |
| Adapter repository | `lightx2v/Minimax-h3-Turbo` | `FastVideo/FastVideo-FastH3-4-step-Preview-v1-LoRA` |
| Adapter revision | `3ec17a324ced54151364f24f8b5fb6bf7e26414f` | `c0e8ec2ac460e801094a0abe74bfb57347388ce3` |
| Adapter file | `minimax_h3_ref2v_turbo_4step_v0.1_bf16.safetensors` | `vsa-datafree/adapter_model.safetensors` |
| Base sigma schedule | `[1, 0.75, 0.5, 0.25, 0]` | `[0.999, 0.749, 0.5, 0.25, 0]` |
| Video / audio flow shifts | 12 / 3 | 12 / 3 |

Avatar adapter SHA-256:
`9e642fc8749c74f8da5e2382877ab5c7aa37b9a73b7fd0d6d457bd1b3cb1ae99`.

FastH3 adapter SHA-256:
`42dc502a2078f166c396a1fa75f29728d1844363652d345d5ef3e2b444ed6470`.

The avatar has 34 saved guide timestep plans and guide timestep 0.8. The FastH3
sidecar is v3 and bound to its adapter and schedule. Generate each sidecar once
and copy the same bytes to both machines. The launcher verifies input and adapter
hashes and compares actual model/cache fingerprints between runs.

## Hardware attribution and comparison limits

Historical project records identify the source system as RTX PRO 5000-class
Blackwell. Some raw driver inventories use the generic name
`NVIDIA Graphics Device`. A project label is not the same as a driver-verified
marketing SKU. The launcher therefore records PCI device and subsystem IDs,
architecture and VRAM in addition to the name, and rejects inconsistent cards
within a run. The new identity checks have CPU test coverage; live driver
compatibility still belongs to the pending GPU qualification.

Do not infer the target card's architecture, VRAM or performance from the shorthand
"5500". These runtimes require SM120 and their configured memory guards. Another
architecture needs a separate port and qualification rather than a silent change
to precision, GPU count or offload.

Identical workload and software do not make the hosts identical. CPU, PCIe/RDMA,
driver, clocks and power limits can affect end-to-end latency. The comparison is
therefore a same-workload system comparison. Output-byte equality is reported
separately; subjective quality requires its own evaluation.

## Validation and remaining work

The [package validation record](../evidence/sku-package-validation.json) covers
CPU tests, dry-run command resolution, syntax checks, reconstruction against the
saved source trees, original dispatch receipts, and local multipart HTTP / MP4
decode checks. The [artifact audit](../evidence/sku-evidence-audit.json) checks the
original sample arrays and recorded media metadata. These checks do not replace
an end-to-end GPU run.

The avatar runtime Python files match the saved runtime. The 15s reconstruction
has the documented CAKE source-location change for portability. Benchmark prompts
remain unchanged, including non-English dialogue: translating them would change
the workload. The new documentation and report are in English.

Outstanding work is to run these same two profiles on an available reference
system and the identified target system, retain all outputs and manifests, then
run `compare` separately for each case. No target measurements or estimated
speedups are supplied in this release.

Accuracy corrections carried into this release: the avatar median is 3.589779 s
(the value 3.602356 s is one sample, not the median); 13.311 s is the 15s mean,
not its median; and the quality page's 6.993 s belongs to a 10-second experiment.
