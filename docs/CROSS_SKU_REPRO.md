# Reproduce the deployed 5s avatar and the 15s FastH3 reference

[PRO 5000 data](../benchmark/SM120/README.md) · [PRO 5500 launcher guide](PRO5500.md) ·
[Public data and downloads](https://lishunyang12.github.io/fasth3-lossy-ablation/sku-reproduction.html) · [Verified context](VERIFIED_CONTEXT.md)

This package contains **two benchmark cases**. Its configuration, preparation and
result layout follows the organization of Sana SOL Engine documentation; the
runtime is the saved H3 implementation.

**The measurements below are historical. The new launcher has passed CPU,
source-reconstruction and historical-log checks, but has not been rerun on GPUs.**
Run the same published revision on both machines for an apples-to-apples
comparison. Historical numbers are references, not newly measured target results.

| Case | GPUs | Workload | Historical HTTP median |
|---|---:|---|---:|
| `5s-avatar-ref2va-4step` | 8 | Ref2VA Turbo, 960×544, 4 steps; 124-frame window with 5-frame overlap | 3.589779 s; 13 measured continuations |
| `15s-4step-mxfp8-sage-nvfp4` | 8 | FastH3, 1280×704 output, 362 frames, 4 steps | 13.304 s; 3 measured requests |

The avatar replays all 16 saved requests, seeds 8300–8315, with the same reference
image uploaded twice and one continuation session. The first 3 requests are
warmup. The first output has 124 frames / 5.166667 s; subsequent outputs have
119 frames / 4.958333 s. Its historical application-ready median is 3.706288 s,
which includes additional application checks and is a different timing boundary.
The [avatar quality page](https://lishunyang12.github.io/fasth3-lossy-ablation/avatar-quality.html)
is a separate 10-second / 243-frame comparison: its 6.993 s result is not the
5-second performance reference.

The numbered setup steps below describe the **manual host-Python path**.
For the prebuilt image, follow the [Docker guide](SM120_DOCKER.md) or the
[PRO 5500 Docker walkthrough](PRO5500.md#docker-image-recommended).
The image already contains the dependencies and both prepared runtimes; Docker
users do not repeat the host environment setup, runtime reconstruction or asset
JSON editing below. Model weights and AdaLN caches still need to be prepared or
copied from the reference machine.

## 1. Environment and hardware

Each case requires **8 idle GPUs of the same SKU and SM120 architecture**, with
TP1/SP8, text-encoder TP8, VAE PP8 and the recorded BF16 RDMA path. The launcher
records the driver-reported GPU name, PCI device/subsystem IDs, UUID, memory,
driver, clocks, power limit, CPU and GPU interconnect topology. It rejects
incomplete or inconsistent GPU identities within one run. Some historical logs
report only `NVIDIA Graphics Device`; the RTX PRO 5000 designation comes from
project hardware records. The exact hardware meant by "5500" remains unverified.

Without an RDMA-capable NIC/path, these two profiles cannot run as released.
A no-NIC NCCL/P2P variant needs separate qualification.

The launcher does not reduce resolution, change precision or enable offload to
fit a different card. The 15s historical summary reports 63099 MiB of peak device
memory, approximately 61.62 GiB. The profile uses a conservative 65147 MiB total
VRAM guard; this is not a measured minimum. The avatar uses a 60000 MiB VRAM guard
and a 400 GiB available-host-memory guard. Its experimental path retains BF16
projection weights, with approximately 38.5 GB of pinned host memory per rank.
These guards are operational requirements of this package, not capacity studies.

Use Python 3.12, a compatible CUDA 13.2 environment, and `curl`, `ffmpeg`,
`ffprobe`, `git`, `patch` and `numactl`. The recorded key package versions are:

```text
torch==2.13.0+cu132
torchvision==0.28.0+cu132
vllm==0.28.0
transformers==5.14.1
diffusers==0.40.0
flashinfer-python==0.6.16.post3
nvidia-cutlass-dsl==4.6.2
comfy-kitchen==0.2.33
safetensors==0.8.0
huggingface-hub==1.29.0
```

These are version constraints, not a universal pip installation command. Obtain
compatible builds from their distribution channels. The repository's
`flashinfer/` supplies the custom Python/CUDA sources. Run
`python3 scripts/prepare_flashinfer.py` to obtain the pinned third-party headers.
If comfy-kitchen is installed in a separate directory, add its `kitchen` path to
the asset entries; the directory must include distribution metadata. The
launcher checks dependency versions and fingerprints source and binary files.

## 2. Reconstruct both historical runtimes

```bash
git clone https://github.com/lishunyang12/h3-optimizations.git
cd h3-optimizations
# Check out the source revision shown on the public data page.
python3 scripts/sku_benchmark.py list
python3 scripts/sku_benchmark.py prepare --variant avatar5 --output /opt/h3-runtime-avatar5
python3 scripts/sku_benchmark.py prepare --variant reference --output /opt/h3-runtime-15s
```

Use writable, previously nonexistent output directories. `prepare` verifies the
September 23 source snapshot hashes, applies the exact historical patch or
avatar overlay, and uses the bundled relative CAKE source location. Each runtime
receives a `sku-source.json` fingerprint checked before execution. Keep the two
copies separate. The general integration under `vllm-omni/` is not a substitute
for these historical runtimes.

## 3. Model weights and AdaLN caches

Use an environment with huggingface-hub installed. These commands download large
model weights; the reproduction ZIP itself contains no model weights.

### 5s avatar: verified download instructions

As checked on **2026-10-28**, the historical MiniMax-H3 revision
`73372e6cf53e414edd3ab03e357717fb0602e758` returns `RevisionNotFound`.
The official repository still serves the original-compatible
[Ref2VA partition at revision 42ed227](https://huggingface.co/MiniMaxAI/MiniMax-H3/tree/42ed227ee7df40d41602854ae760620d6eb651fe/Ref2VA).
All **81 files**, including **29 weight files**, have server-reported sizes and
hashes matching the verified retained original partition. This verification
covers `Ref2VA`, not the whole repository or the 15s `FL2VA` checkpoint.

| Asset | Size (decimal GB) | How to obtain it |
|---|---:|---|
| Ref2VA base partition | 144.05 | Official pinned download below |
| LightX2V Ref2VA 4-step adapter | 1.38 | Official pinned download below |
| Original `guide-adaln.safetensors` | 1.65 | Supplied separately by the experiment owner; no public download URL as of 2026-10-28 |

Run from the repository root. `/models` must be writable; adjust `H3_MODELS` if
needed. The prepared output directory `avatar-assets` must not already exist.

```bash
python3 -m pip install "huggingface_hub==1.29.0"
H3_MODELS=/models

hf download MiniMaxAI/MiniMax-H3 \
  --revision 42ed227ee7df40d41602854ae760620d6eb651fe \
  --include "Ref2VA/*" \
  --local-dir "$H3_MODELS/avatar-downloads/MiniMax-H3"

hf download lightx2v/Minimax-h3-Turbo \
  minimax_h3_ref2v_turbo_4step_v0.1_bf16.safetensors \
  --revision 3ec17a324ced54151364f24f8b5fb6bf7e26414f \
  --local-dir "$H3_MODELS/avatar-downloads/adapter"

python3 scripts/sku_models.py --case 5s-avatar-ref2va-4step \
  --base "$H3_MODELS/avatar-downloads/MiniMax-H3" \
  --adapter "$H3_MODELS/avatar-downloads/adapter/minimax_h3_ref2v_turbo_4step_v0.1_bf16.safetensors" \
  --output "$H3_MODELS/avatar-assets"
```

**Keep both `--revision` arguments for reproduction.** Without them, Hugging Face
uses the repository's moving `main` branch. The LightX2V revision is accessible
and is separate from the unavailable historical MiniMax base revision.
Use the exact BF16 Ref2VA adapter filename above.

The existing avatar `sku_models.py --download` command and the prebuilt image's
`download` command still request the unavailable base revision. Use the explicit
downloads above followed by `--base` and `--adapter`; no inference-code change is
required. Docker users should use the equivalent
[container commands](SM120_DOCKER.md#download-models-and-prepare-caches).

The preparation script verifies the adapter SHA-256 and creates the four-step
model view. Copy `model` and `adapter` from
`$H3_MODELS/avatar-assets/model-assets.json` into the asset configuration. Keep
`avatar-downloads` alongside `avatar-assets`: the prepared view links to those
downloaded files. For existing weights, pass their paths to `--base` and
`--adapter`; the base directory must contain `Ref2VA`.

After receiving the original AdaLN file, place it at
`$H3_MODELS/guide-adaln.safetensors`, use that path for `adaln`, and verify:

```bash
printf '%s  %s\n' \
  000ba165406e6b64e0ae3a952f89f55833f45b37dd9a41eae327bac6bfe42301 \
  "$H3_MODELS/guide-adaln.safetensors" | sha256sum --check -
```

This is the SHA-256 recorded from the retained original sidecar on 2026-10-28.
Use identical file bytes on both machines. Generating a new sidecar is an
alternative for fresh paired runs, but does not establish byte identity with
the historical cache.

### 15s FastH3 assets

```bash
python scripts/sku_models.py --case 15s-4step-mxfp8-sage-nvfp4 --download --output /models/fasth3-assets
```

For existing weights, replace `--download` with
`--base /path/to/MiniMax-H3 --adapter /path/to/adapter.safetensors`. The tool
verifies the adapter SHA-256.
Copy the resulting `model` and `adapter` paths from `model-assets.json` into your
asset configuration. Base directories must contain the relevant model partition.

The avatar's historical case record retains MiniMax-H3 revision
`73372e6cf53e414edd3ab03e357717fb0602e758`; the verified retrieval revision above
does not rewrite that provenance. The 15s case pins MiniMax-H3 revision
`9bfb6693f2cf6de171db46d1aa586f67d773a1da` and FastVideo Preview LoRA revision
`c0e8ec2ac460e801094a0abe74bfb57347388ce3`. Exact filenames and adapter hashes
are in the case profiles. The avatar reference image is bundled and hash-checked.

### Generate sidecars when no retained cache is available

Skip the avatar generation command if using the supplied original cache.
Otherwise, **generate each needed sidecar once on an idle GPU, then copy that
exact file to both machines**. This preparation is outside request timing.
Independently generated sidecars should not be assumed to contain identical
bytes. The paths below assume `/models`; adjust them if using another location.

```bash
PYTHONPATH=/opt/h3-runtime-avatar5 python scripts/sku_adaln.py \
  --runtime /opt/h3-runtime-avatar5 \
  --transformer-path /models/avatar-assets/ref2va-fourstep/Ref2VA/transformer \
  --output /models/guide-adaln.safetensors

PYTHONPATH=/opt/h3-runtime-15s python reference-runtime/tools/minimax_h3/build_adaln_cache.py \
  --transformer-path /models/fasth3-assets/MiniMax-H3/FL2VA/transformer \
  --model-variant fl2va --mode t2va \
  --fasth3-adapter /models/fasth3-assets/adapter/vsa-datafree/adapter_model.safetensors \
  --base-schedule 0.999 0.749 0.5 0.25 0.0 --flow-shift 12 --audio-flow-shift 3 \
  --output /models/fasth3-adaln.safetensors
```

The avatar uses its original 34 guide timestep plans; the Turbo adapter does not
modify those AdaLN weights. FastH3 uses an adapter-bound v3 sidecar. These caches
are not interchangeable. The comparison checks actual weight, input and cache
byte hashes, including weights reached through model-directory symlinks.

## 4. Run the same two cases on both machines

Edit only the machine-specific asset paths:

```bash
cp configs/sku-assets.example.json /tmp/h3-assets.json
# Edit runtime, python, model, adapter, adaln and cuda paths in both entries.
# Preview commands without model loading or GPU access:
python3 scripts/sku_benchmark.py suite --cases main --assets /tmp/h3-assets.json \
  --gpus 0,1,2,3,4,5,6,7 --output /results/pro5000-plan --dry-run
# Run the two cases sequentially, with a separate server for each:
python3 scripts/sku_benchmark.py suite --cases main --assets /tmp/h3-assets.json \
  --gpus 0,1,2,3,4,5,6,7 --output /results/pro5000
```

On the target machine, use the same source revision and workload files, adjust
only asset paths and GPU selectors, and use `/results/target-sku` as the output
directory. A single case can also be launched with `run --case CASE` instead of
`suite --cases main`. Copy both result directories to the comparison machine.

```bash
python3 scripts/sku_benchmark.py compare \
  /results/pro5000/5s-avatar-ref2va-4step/result.json \
  /results/target-sku/5s-avatar-ref2va-4step/result.json \
  --output /results/avatar-comparison.json

python3 scripts/sku_benchmark.py compare \
  /results/pro5000/15s-4step-mxfp8-sage-nvfp4/result.json \
  /results/target-sku/15s-4step-mxfp8-sage-nvfp4/result.json \
  --output /results/fasth3-comparison.json
```

## 5. Results and interpretation

Each case writes `result.json`, `sources.json`, `assets.json`, `environment.json`,
`server.log`, per-request JSON and logs, MP4 files and `telemetry.jsonl`.
The new launcher times local curl process launch through the complete MP4 being
written, and also records curl's own timing. Full media decoding, probing and
hashing occur outside this interval. Historical clients used their own timing
code, so rerun the 5000 reference with this launcher before comparing a new SKU.

Both cases audit actual four-step DiT execution on every rank, quantization
fallbacks, full NVFP4 VAE execution, and CAKE/RDMA evidence. The avatar replays
continuation requests; it does not reproduce chat, RTMP, playback delay or
application-level quality-screening latency. Original benchmark prompts remain
verbatim, including Chinese dialogue, to preserve the workload. Documentation
and the data page are in English.

The launcher exits if the selected GPUs are occupied, records foreign compute
activity as a failure, and cleans up only the service process group it starts.
It does not lock clocks. CPU, interconnect, driver, power and clock differences
remain part of the measured system; a speedup cannot automatically be attributed
to GPU compute alone.

`compare` rejects different configurations, sources, weights, caches, dependency
versions, failed runs and detected interference. It separately reports whether
MP4 bytes match; different bytes across GPUs are not automatically a perceptual
quality regression. Both quantized configurations are lossy relative to their
original full-precision models.

CPU checks: `python3 -m unittest discover -s tests -p test_sku_benchmark.py -v`.
The fixtures contain structured historical dispatch records, not private launch
manifests. Fresh GPU execution remains the outstanding qualification step.
