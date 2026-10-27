# Historical runtime and fixed inputs

These files support exactly two cases in the [reproduction guide](../docs/CROSS_SKU_REPRO.md):
the deployed 5s avatar and the published 15s FastH3 reference.

- `avatar5-overlay/` contains all 17 changed Python files from the saved
  September 22 `native-v3` avatar runtime, copied byte-for-byte.
  `variants.json` records their SHA-256 hashes.
- `avatar5-requests.json` preserves all 16 requests, seeds and continuation
  ordinals. Three warmup segments precede thirteen measured segments.
- `avatar5-adaln-plans.json` preserves 34 FP32 timestep plans, without weights.
- `avatar5-model-index.json` supplies the historical four-step model view.
- `inputs/avatar5-reference.png` is the generated Zhiwei study reference used
  by those requests. It differs from the closed-window picture in the later
  10-second quality comparison.

Both reconstructions start from `reference-runtime/vllm-omni`. The 15s case
applies `patches/ref2va-to-full-length-runtime.patch`. The launcher uses the
bundled CAKE source location. See [verified context](../docs/VERIFIED_CONTEXT.md).

Original source notices are preserved. No model checkpoint, AdaLN tensor
sidecar, private launch environment or credential is included. The wrapper has
CPU and historical-log validation; fresh GPU qualification is still required.
