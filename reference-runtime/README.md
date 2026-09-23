# Original experimental runtime

`vllm-omni/` contains the actual short-window Ref2VA runtime source, including
optimizations not yet present in the portable integration. The earlier full-length
FastH3 runtime can be reconstructed using
[`ref2va-to-full-length-runtime.patch`](../patches/ref2va-to-full-length-runtime.patch).
Both variants retain the original computation code; generated caches and weights
are excluded.

`tools/minimax_h3/` contains the full-length runtime's additional kernel
microbenchmarks, validation utilities and historical launch/profiling scripts.
Some scripts contain original filesystem paths, eight-GPU placement and fixed
experiment assumptions. Read and adapt them before use. They are not an automatic
installer or the entry point for DGX Spark.

This runtime uses different experimental environment switches from the main
`vllm-omni/` ports. Do not copy modules between them or place both on `PYTHONPATH`.
`gpu-observed.txt` records the original software environment, not a clean-install
lockfile, and it does not establish ARM64 binary availability.

The packaged grid-constant attention loader points at its adjacent bundled
sources. Additional unused kernel qualification metadata may retain original
paths as historical provenance. See the root [source manifest](../configs/sources.json)
and [license notes](../docs/THIRD_PARTY.md).

The selected full-length experiment recorded 13.281 / 13.304 / 13.348 seconds for
three measured requests after one warmup on eight SM120 devices, TP1/SP8, four
FastH3 steps, MXFP8 DiT, Sage attention and NVFP4 VAE, at 1280×704 / 362 frames.
This is a historical single-input observation. It is not a measurement of the
new integration, Spark performance, or dense BF16 output equivalence. The new
repository does not include the original generated media to replay that quality
comparison.
