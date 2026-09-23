# Source and license provenance

This is a fresh source export; it contains no history from private working
repositories. Original third-party license and copyright files remain in place.
The root Apache-2.0 license applies to new project contributions and does not
replace other licenses embedded in third-party sources.

| Tree | Source | License |
| --- | --- | --- |
| `vllm-omni/` | [vLLM-Omni](https://github.com/vllm-project/vllm-omni), integration of the five local optimization ports | Apache-2.0, plus embedded notices |
| `flashinfer/` | Custom [FlashInfer](https://github.com/lishunyang12/flashinfer) source snapshot | Apache-2.0; also `NOTICE` and `licenses/` |
| `fastvideo-kernel/` | Kernel package from [FastVideo](https://github.com/hao-ai-lab/FastVideo) | Apache-2.0 |
| `reference-runtime/vllm-omni/` | Actual modified H3 Ref2VA runtime source | Apache-2.0, plus embedded notices |

Exact port bases and heads are in [sources.json](../configs/sources.json). The
FlashInfer Python snapshot and effective `include/` / `csrc/` came from different
working trees with local changes; neither Git HEAD alone reconstructs this
bundle. The experimental runtime had no exact upstream Git identity. Its package
version is metadata, not proof of source identity.

`evidence/source-sha256.json` records every regular bundled source file;
`evidence/packaging-changes.json` records any subsequent packaging edits. Verify
with `python scripts/verify_sources.py`. Editable installation may rewrite
`_version.py`; run verification before installation or inspect that difference.

FlashInfer's `flashinfer/data/{csrc,include}` links resolve inside this repository.
Its CUTLASS, CCCL and spdlog links resolve after running
`python scripts/prepare_flashinfer.py`, which retrieves the exact public commits
listed in `configs/flashinfer-third-party.json`. These dependencies retain their
own licenses and are deliberately not committed a second time.

The descriptor-ABI overlay needed by the experimental runtime is included in
`reference-runtime/vllm-omni/vllm_omni/diffusion/attention/ops/h3_cake_grid_constant_sources/`.
The full-length variant patch restores the original runtime, including historical
absolute paths; it is an archival reconstruction, not a portable launcher.

Public upstream test/example assets remain in the vLLM-Omni source export. No H3
weights, personal reference inputs, service credentials, user conversation logs,
generated videos or profiler captures from local experiments are included.
