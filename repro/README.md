# Historical runtime overlays and fixed inputs

These files support the [cross-SKU reproduction guide](../docs/CROSS_SKU_REPRO.md).
`variants.json` records the complete overlay checksums; source reconstruction
starts from the already published `reference-runtime/vllm-omni` snapshot.

- `avatar5-overlay`: the saved `native-v3` runtime behind the September 22
  deployed avatar. All 17 changed Python files are copied byte-for-byte.
- `t2va5-overlay`: the September 23 1344×768 / 124-frame precision experiment.
  The launcher relocates CAKE's old absolute build path to bundled sources.
- `15s-output-fusion-overlay`: four files from the September 18 full NVFP4 VAE
  output-fusion experiment. This is separate from the public 13.304s reference.
- `avatar5-requests.json`: all 16 original fixed requests, with seeds and
  continuation ordinals. Three warmup segments precede thirteen measured ones.
- `avatar5-adaln-plans.json`: original 34 exact FP32 timestep plans; no weights.
- `inputs/avatar5-reference.png`: the fixed generated Zhiwei study reference
  used by that 16-request sequence. It differs from the later closed-window
  picture used for the separate 10-second quality comparison.
- `inputs/omni5-reference.png`: the fixed reference from the October 21
  four-GPU FastH3-Omni test. This is a separate workload and model family.

The source overlays preserve their original notices. No model checkpoint,
AdaLN tensor sidecar, private launch environment or credential is included.
The new benchmark wrapper has CPU and historical-log validation only; fresh
GPU qualification is still required on both machines.
