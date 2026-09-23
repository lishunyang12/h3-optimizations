# Release validation — 2026-09-23

The integrated source passed **761 CPU-marked tests**, with **2 skipped** and
**61 deselected**. The test command, from `vllm-omni/`, was:

```bash
python -m pytest -q \
  tests/diffusion/models/minimax_h3 \
  tests/diffusion/attention/test_sage_block_sparse_interface.py \
  tests/diffusion/attention/test_ulysses_transport_interface.py \
  tests/diffusion/quantization/test_cuda_mxfp8.py \
  -m cpu --run-level=core_model
```

Environment: Linux x86_64, Python 3.12.13, PyTorch 2.13.0+cu132, vLLM 0.29.0.
The test process imported the bundled `vllm-omni/` source. Existing installed
Omni metadata still reported an older 0.28 development version and emitted a
version warning; it was not used as the identity of the tested source.

The passing run allowed localhost/Gloo sockets and CUDA runtime initialization
for CPU pinned-memory tests. An initial restricted-sandbox run with GPUs hidden
failed on those environment requirements and the platform cache API. It was
rerun in the compatible environment above without changing computation code.
No checkpoint, inference server or full-model workload was loaded for testing.

Additional release checks:

- SHA-256 verification of 8,108 bundled regular source/support files.
- Python syntax parsing of 5,607 files, including both runtimes.
- The full-length reconstruction patch applies cleanly to the reference runtime.
- New Spark launch commands are previewable with only the Python standard library.
- Source credential-pattern scan and preservation of third-party license files.

This release has **not** been tested on ARM64/GB10, with a clean GPU environment
installation, or through full-model numerical/quality/latency comparisons of the
combined ports. CPU tests do not validate CUDA compilation, quantized output,
custom Sage architecture support, RDMA or a performance improvement. GPU-specific
tests and microbenchmarks are included for target-device qualification.
