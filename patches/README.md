# Individual optimization patches

The main `vllm-omni/` tree already includes all five optimization ports. These
patches are provided for review and cherry-picking into separate checkouts.
They are `git diff --binary --full-index` patches for `git apply`, not `git am`.

Exact base and head IDs for each patch are in
[`configs/sources.json`](../configs/sources.json). Use the matching base, then
`git apply --check /path/to/patch` before applying. In particular, `vae.patch`
depends on the MXFP8 head; the patches are not an arbitrary-order series.

| Patch | Content |
| --- | --- |
| `adaln.patch` | Exact runtime AdaLN reuse, optional sidecars, builder and tests |
| `mxfp8.patch` | CUDA MXFP8 DiT projections, SwiGLU fusion and microbenchmark |
| `sage.patch` | Sparse Sage adapter, quantization, BF16 API contract fix and tests |
| `vae.patch` | Video VAE MXFP8, paired/mixed scheduling, stream/output ownership fixes |
| `rdma.patch` | Optional Ulysses FlashInfer transport and tests |

`ref2va-to-full-length-runtime.patch` is different: apply it **inside a copy of**
`reference-runtime/vllm-omni/` to reconstruct the earlier selected 15-second
FastH3 runtime's `vllm_omni/` source. It must not be applied to the main integration.
The reconstruction restores historical absolute overlay paths and is for source
review; it is not a portable installation or a Spark configuration.
