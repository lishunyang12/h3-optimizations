# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Build the two-stage MiniMax H3 E4M3 QKV transport-scale sidecar.

This tool is intentionally offline.  The ``derive`` stage merges exactly one
calibration snapshot from each SP8 rank and emits the scale-only artifact read
by the validation observer.  The ``finalize`` stage merges the corresponding
validation snapshots, requires a clean audit, binds the scales to the adapter
identity carried by a FastH3 AdaLN-v3 sidecar, and atomically publishes the
runtime sidecar.

Neither stage performs a distributed collective.  Both refuse to replace an
existing output unless ``--draft-overwrite`` is explicitly supplied for a
filename ending in ``.draft.safetensors``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import safe_open, save_file

from vllm_omni.diffusion.models.minimax_h3.adaln_cache import (
    MiniMaxH3AdalnCacheBinding,
    read_minimax_h3_adaln_cache_binding,
)
from vllm_omni.diffusion.models.minimax_h3.fp8_contract import (
    MINIMAX_H3_ALL_MAIN_FP8_DIT_QUANT_MODE,
    MINIMAX_H3_BF16_DIT_QUANT_MODE,
    MINIMAX_H3_MLP_FP8_DIT_QUANT_MODE,
    MINIMAX_H3_MLP_OUT_FP8_DIT_QUANT_MODE,
    MINIMAX_H3_MLP_OUT_QKV_FP8_DIT_QUANT_MODE,
    validate_minimax_h3_all_main_fp8_run_contract,
    validate_minimax_h3_mlp_fp8_run_contract,
    validate_minimax_h3_mlp_out_fp8_run_contract,
    validate_minimax_h3_mlp_out_qkv_fp8_run_contract,
)
from vllm_omni.diffusion.models.minimax_h3.qkv_transport_calibration import (
    MiniMaxH3QKVTransportCalibrationSnapshot,
    audit_qkv_e4m3_scales,
    derive_qkv_e4m3_scales,
    merge_qkv_transport_calibration_files,
    qkv_transport_calibration_provenance_from_metadata,
    qkv_transport_validation_scale_draft_metadata,
)
from vllm_omni.diffusion.models.minimax_h3.qkv_transport_scales import (
    MINIMAX_H3_QKV_RUNTIME_CHECKPOINT_MANIFEST_SCOPE,
    MINIMAX_H3_QKV_TRANSPORT_KERNEL_ABI,
    MINIMAX_H3_QKV_TRANSPORT_KERNEL_PROVIDER,
    MiniMaxH3QKVTransportCalibrationProvenance,
    MiniMaxH3QKVTransportMediaProfile,
    MiniMaxH3QKVTransportScaleBinding,
    canonical_minimax_h3_runtime_checkpoint_manifest,
    digest_minimax_h3_qkv_transport_scale_payload,
    digest_minimax_h3_runtime_checkpoint_manifest,
    prevalidate_minimax_h3_qkv_transport_scale_sidecar,
    read_minimax_h3_runtime_checkpoint_manifest,
)

_NUM_RANKS = 8
_NUM_LAYERS = 50
_NUM_COMPONENTS = 3
_EXPECTED_SHAPE = (_NUM_LAYERS, _NUM_COMPONENTS)

_BASE_SCHEDULE = (0.999, 0.749, 0.5, 0.25, 0.0)
_FLOW_SHIFT = 12.0
_AUDIO_FLOW_SHIFT = 3.0
_KERNEL_PROVIDER = MINIMAX_H3_QKV_TRANSPORT_KERNEL_PROVIDER
_KERNEL_ABI = MINIMAX_H3_QKV_TRANSPORT_KERNEL_ABI
_PROMPT_SHA256 = "613e0a1beccf83566e3531852c52c2e11daffa3b255222cd67726cdf73c1a563"

_DERIVED_METADATA_KEYS = frozenset(
    {
        "format",
        "format_version",
        "source_snapshot_count",
        "scale_formula",
        "payload_sha256",
        "source_model_revision",
        "runtime_checkpoint_revision",
        "runtime_checkpoint_manifest_sha256",
        "campaign_uuid",
        "run_contract_sha256",
        "expected_observation_count",
    }
)
_MANIFEST_ROOT_FILE = "model_index.json"
_MANIFEST_ROOT_DIRECTORIES = ("processor", "tokenizer", "text_encoder", "transformer")
_OUTPUT_MARKER = "MINIMAX_H3_QKV_SCALE_BUILDER="


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _resolve_snapshot_paths(
    paths: Sequence[str | os.PathLike[str]],
    *,
    kind: str,
) -> tuple[Path, ...]:
    if len(paths) != _NUM_RANKS:
        raise ValueError(f"{kind} requires exactly {_NUM_RANKS} rank snapshots, got {len(paths)}")
    resolved: list[Path] = []
    for raw_path in paths:
        path = Path(raw_path).expanduser()
        if not path.is_file():
            raise ValueError(f"{kind} snapshot does not exist: {path}")
        if path.suffix != ".safetensors":
            raise ValueError(f"{kind} snapshot is not a safetensors file: {path}")
        resolved.append(path.resolve())
    if len(set(resolved)) != _NUM_RANKS:
        raise ValueError(f"{kind} requires {_NUM_RANKS} distinct rank snapshots")
    return tuple(resolved)


def _validate_complete_snapshot(
    snapshot: MiniMaxH3QKVTransportCalibrationSnapshot,
    *,
    kind: str,
) -> None:
    if snapshot.num_layers != _NUM_LAYERS:
        raise ValueError(f"{kind} must cover {_NUM_LAYERS} layers, got {snapshot.num_layers}")
    if tuple(snapshot.absmax.shape) != _EXPECTED_SHAPE:
        raise ValueError(f"{kind} must have shape {_EXPECTED_SHAPE}")
    if bool((snapshot.observation_count <= 0).any()):
        missing = int((snapshot.observation_count <= 0).sum().item())
        raise ValueError(f"{kind} does not cover all 50x3 cells: {missing} are missing")
    if snapshot.provenance is None:
        raise ValueError(f"{kind} must carry calibration provenance")
    expected_observations = snapshot.provenance.expected_observation_count * _NUM_RANKS
    if not bool((snapshot.observation_count == expected_observations).all()):
        raise ValueError(f"{kind} observation_count must be exactly {expected_observations} for every layer/component")
    if bool((snapshot.numel <= 0).any()):
        raise ValueError(f"{kind} must contain samples for every 50x3 cell")
    nonfinite = int(snapshot.nonfinite_count.sum().item())
    if nonfinite:
        raise ValueError(f"{kind} contains {nonfinite} non-finite values")
    if not torch.equal(snapshot.finite_count, snapshot.numel):
        raise ValueError(f"{kind} finite_count must equal numel")


def _validate_scale_tensor(scales: torch.Tensor, *, kind: str) -> torch.Tensor:
    if not isinstance(scales, torch.Tensor):
        raise TypeError(f"{kind} must be a tensor")
    if scales.device.type != "cpu" or scales.dtype != torch.float32:
        raise ValueError(f"{kind} must be CPU FP32")
    if tuple(scales.shape) != _EXPECTED_SHAPE:
        raise ValueError(f"{kind} must have shape [50, 3]")
    if not scales.is_contiguous():
        raise ValueError(f"{kind} must be contiguous")
    if not bool(torch.isfinite(scales).all()) or not bool((scales > 0).all()):
        raise ValueError(f"{kind} must be finite and positive")
    if not bool(torch.isfinite(torch.reciprocal(scales)).all()):
        raise ValueError(f"{kind} must have finite FP32 reciprocals")
    return scales.detach().contiguous().clone()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        before = os.fstat(handle.fileno())
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
        after = os.fstat(handle.fileno())
    identity_before = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    identity_after = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    if identity_before != identity_after:
        raise ValueError(f"artifact changed while hashing: {path}")
    return digest.hexdigest()


def _runtime_checkpoint_manifest_entries(
    runtime_root: str | os.PathLike[str],
) -> list[dict[str, Any]]:
    """Hash the exact qkv-input-conditioning-v1 file scope once, offline."""

    root = Path(runtime_root).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"runtime checkpoint root is not a directory: {root}")
    selected: list[Path] = []
    root_file = root / _MANIFEST_ROOT_FILE
    if not root_file.is_file():
        raise ValueError(f"runtime checkpoint is missing {_MANIFEST_ROOT_FILE}: {root}")
    selected.append(root_file)
    for directory_name in _MANIFEST_ROOT_DIRECTORIES:
        directory = root / directory_name
        if not directory.is_dir():
            raise ValueError(f"runtime checkpoint is missing required {directory_name}/ directory: {root}")
        selected.extend(
            path for path in directory.rglob("*") if path.is_file() and ".cache" not in path.relative_to(root).parts
        )
    entries: list[dict[str, Any]] = []
    for path in sorted(selected, key=lambda item: item.relative_to(root).as_posix()):
        relative = path.relative_to(root).as_posix()
        before = path.stat()
        sha256 = _file_sha256(path)
        after = path.stat()
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ):
            raise ValueError(f"runtime checkpoint file changed while building manifest: {path}")
        entries.append(
            {
                "path": relative,
                "size_bytes": before.st_size,
                "sha256": sha256,
            }
        )
    return entries


def build_runtime_checkpoint_manifest(
    runtime_root: str | os.PathLike[str],
    output: str | os.PathLike[str],
) -> dict[str, Any]:
    """Create the no-clobber canonical manifest consumed by launchers."""

    target = Path(output).expanduser().absolute()
    if target.suffix != ".json":
        raise ValueError("runtime checkpoint manifest output must end in .json")
    if target.exists():
        raise FileExistsError(f"refusing to overwrite existing artifact: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    entries = _runtime_checkpoint_manifest_entries(runtime_root)
    content = canonical_minimax_h3_runtime_checkpoint_manifest(entries)
    expected_digest = digest_minimax_h3_runtime_checkpoint_manifest(entries)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    fd = os.open(target, flags, 0o644)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        target.unlink(missing_ok=True)
        raise
    _fsync_directory(target.parent)
    actual_digest = read_minimax_h3_runtime_checkpoint_manifest(target)
    if actual_digest != expected_digest:
        raise AssertionError("runtime checkpoint manifest did not round-trip exactly")
    return {
        "stage": "manifest",
        "output": str(target),
        "scope": MINIMAX_H3_QKV_RUNTIME_CHECKPOINT_MANIFEST_SCOPE,
        "file_count": len(entries),
        "runtime_checkpoint_manifest_sha256": actual_digest,
        "whole_sha256": _file_sha256(target),
    }


def _is_draft_path(path: Path) -> bool:
    return path.name.endswith(".draft.safetensors")


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    directory_fd = os.open(path, flags)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _atomic_save_safetensors(
    path: str | os.PathLike[str],
    tensors: Mapping[str, torch.Tensor],
    metadata: Mapping[str, str],
    *,
    draft_overwrite: bool,
) -> Path:
    target = Path(path).expanduser().absolute()
    if target.suffix != ".safetensors":
        raise ValueError(f"output must end in .safetensors: {target}")
    if draft_overwrite and not _is_draft_path(target):
        raise ValueError("--draft-overwrite is allowed only for *.draft.safetensors outputs")
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and not draft_overwrite:
        raise FileExistsError(f"refusing to overwrite existing artifact: {target}")

    temporary_fd, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.",
        suffix=".tmp",
        dir=target.parent,
    )
    os.close(temporary_fd)
    temporary = Path(temporary_name)
    try:
        save_file(dict(tensors), str(temporary), metadata=dict(metadata))
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        if draft_overwrite:
            os.replace(temporary, target)
        else:
            # Hard-link publication is an atomic no-clobber operation.  Unlike
            # os.replace(), it cannot overwrite a file created after the early
            # existence check above.
            try:
                os.link(temporary, target)
            except FileExistsError as exc:
                raise FileExistsError(f"refusing to overwrite existing artifact: {target}") from exc
            temporary.unlink()
        _fsync_directory(target.parent)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def _scale_summary(scales: torch.Tensor) -> dict[str, Any]:
    return {
        "shape": list(scales.shape),
        "dtype": "float32",
        "min": float(scales.min().item()),
        "max": float(scales.max().item()),
        "mean": float(scales.mean().item()),
    }


def _snapshot_summary(snapshot: MiniMaxH3QKVTransportCalibrationSnapshot) -> dict[str, Any]:
    return {
        "coverage_cells": int((snapshot.observation_count > 0).sum().item()),
        "expected_coverage_cells": _NUM_LAYERS * _NUM_COMPONENTS,
        "observation_count_min": int(snapshot.observation_count.min().item()),
        "observation_count_max": int(snapshot.observation_count.max().item()),
        "finite_count": int(snapshot.finite_count.sum().item()),
        "nonfinite_count": int(snapshot.nonfinite_count.sum().item()),
        "numel": int(snapshot.numel.sum().item()),
        "observed_absmax": float(snapshot.absmax.max().item()),
    }


def _derived_metadata(
    provenance: MiniMaxH3QKVTransportCalibrationProvenance,
    payload_sha256: str,
) -> dict[str, str]:
    return qkv_transport_validation_scale_draft_metadata(
        provenance=provenance,
        source_snapshot_count=_NUM_RANKS,
        payload_sha256=payload_sha256,
    )


def load_derived_scales(
    path: str | os.PathLike[str],
) -> tuple[torch.Tensor, str, MiniMaxH3QKVTransportCalibrationProvenance]:
    """Load and fully validate a scale-only stage-one artifact."""

    candidate = Path(path).expanduser()
    if not candidate.is_file():
        raise ValueError(f"derived scale artifact does not exist: {candidate}")
    before = candidate.stat()
    with safe_open(candidate, framework="pt", device="cpu") as handle:
        metadata = handle.metadata() or {}
        if set(metadata) != _DERIVED_METADATA_KEYS:
            raise ValueError(
                "invalid derived scale metadata: "
                f"missing={sorted(_DERIVED_METADATA_KEYS - set(metadata))}, "
                f"unknown={sorted(set(metadata) - _DERIVED_METADATA_KEYS)}"
            )
        provenance = qkv_transport_calibration_provenance_from_metadata(metadata)
        expected_metadata = _derived_metadata(provenance, metadata.get("payload_sha256", ""))
        if metadata != expected_metadata:
            raise ValueError("derived scale artifact metadata does not match the v2 contract")
        if set(handle.keys()) != {"dequant_scales"}:
            raise ValueError("derived scale artifact tensor set must be exactly {dequant_scales}")
        scales = _validate_scale_tensor(handle.get_tensor("dequant_scales"), kind="dequant_scales")
    after = candidate.stat()
    identity_before = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    identity_after = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    if identity_before != identity_after:
        raise ValueError("derived scale artifact changed while loading")

    payload_sha256 = digest_minimax_h3_qkv_transport_scale_payload({"dequant_scales": scales})
    if payload_sha256 != metadata["payload_sha256"]:
        raise ValueError(
            f"derived scale payload digest mismatch: expected={metadata['payload_sha256']}, actual={payload_sha256}"
        )
    return scales, payload_sha256, provenance


def derive_scale_artifact(
    calibration_snapshot_paths: Sequence[str | os.PathLike[str]],
    output: str | os.PathLike[str],
    *,
    draft_overwrite: bool = False,
) -> dict[str, Any]:
    """Merge eight calibration ranks and publish validation-observer scales."""

    paths = _resolve_snapshot_paths(calibration_snapshot_paths, kind="calibration")
    merged = merge_qkv_transport_calibration_files(
        paths,
        expected_source_ranks=range(_NUM_RANKS),
        expected_observer_mode="calibration",
    )
    if merged.is_validation:
        raise ValueError("derive requires calibration snapshots, not validation snapshots")
    _validate_complete_snapshot(merged, kind="calibration")
    if merged.provenance is None:
        raise ValueError("calibration merge did not retain campaign provenance")
    scales = _validate_scale_tensor(derive_qkv_e4m3_scales(merged), kind="derived dequant_scales")
    tensors = {"dequant_scales": scales}
    payload_sha256 = digest_minimax_h3_qkv_transport_scale_payload(tensors)
    target = _atomic_save_safetensors(
        output,
        tensors,
        _derived_metadata(merged.provenance, payload_sha256),
        draft_overwrite=draft_overwrite,
    )
    loaded_scales, loaded_digest, loaded_provenance = load_derived_scales(target)
    if loaded_digest != payload_sha256 or not torch.equal(loaded_scales, scales):
        raise AssertionError("published derived scale artifact did not round-trip exactly")
    if loaded_provenance != merged.provenance:
        raise AssertionError("published derived scale artifact lost calibration provenance")
    return {
        "stage": "derive",
        "output": str(target),
        "whole_sha256": _file_sha256(target),
        "payload_sha256": payload_sha256,
        "scale_summary": _scale_summary(scales),
        "calibration_audit": _snapshot_summary(merged),
        "provenance": merged.provenance.to_dict(),
    }


def _fixed_media_profile() -> MiniMaxH3QKVTransportMediaProfile:
    return MiniMaxH3QKVTransportMediaProfile(
        width=1280,
        height=720,
        fps=24,
        num_frames=362,
        requested_duration_seconds=15.0,
        seed=1101,
        prompt_sha256=_PROMPT_SHA256,
        effective_width=1280,
        effective_height=704,
        video_shape=(107, 22, 40),
        video_tile_count=1620,
        prefix_tile_count=21,
        compact_row_count=95487,
        padded_row_count=105024,
    )


def _load_media_profile(
    path: str | os.PathLike[str] | None,
) -> MiniMaxH3QKVTransportMediaProfile:
    """Load an exact request/packing profile, retaining the legacy default.

    A prompt-specific scale campaign must bind the sidecar to the prompt and
    packed geometry it actually observed.  The optional file keeps existing
    benchmark callers byte-for-byte compatible while allowing a new campaign
    to supply a canonical JSON object produced before calibration.
    """

    if path is None:
        return _fixed_media_profile()
    candidate = Path(path).expanduser()
    if not candidate.is_file():
        raise ValueError(f"media profile does not exist: {candidate}")
    raw = candidate.read_bytes()
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, ValueError) as exc:
        raise ValueError("media profile must be valid UTF-8 JSON") from exc
    if not isinstance(value, Mapping):
        raise ValueError("media profile must be a JSON object")
    canonical = _canonical_json(value).encode("ascii")
    if raw not in {canonical, canonical + b"\n"}:
        raise ValueError("media profile must use canonical JSON encoding")
    return MiniMaxH3QKVTransportMediaProfile.from_dict(value)


def _read_exact_adaln_binding(path: str | os.PathLike[str]) -> MiniMaxH3AdalnCacheBinding:
    binding = read_minimax_h3_adaln_cache_binding(path)
    if binding.format_version != "3" or binding.fasth3_adapter is None:
        raise ValueError("QKV finalization requires an adapter-bound AdaLN v3 sidecar")
    expected = MiniMaxH3AdalnCacheBinding.for_fasth3(
        adapter=binding.fasth3_adapter,
        model_variant="fl2va",
        mode="t2va",
        base_schedule=_BASE_SCHEDULE,
        flow_shift=_FLOW_SHIFT,
        audio_flow_shift=_AUDIO_FLOW_SHIFT,
    )
    if binding != expected:
        raise ValueError("AdaLN v3 sidecar does not match the fixed QKV v2 FastH3 schedule/shift contract")
    return binding


def finalize_scale_sidecar(
    derived_scale_path: str | os.PathLike[str],
    validation_snapshot_paths: Sequence[str | os.PathLike[str]],
    adaln_sidecar_path: str | os.PathLike[str],
    output: str | os.PathLike[str],
    *,
    draft_overwrite: bool = False,
    dit_quant_mode: str = MINIMAX_H3_BF16_DIT_QUANT_MODE,
    run_contract_path: str | os.PathLike[str] | None = None,
    media_profile_path: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Audit eight validation ranks and atomically publish the bound sidecar."""

    scales, derived_payload_sha256, derived_provenance = load_derived_scales(derived_scale_path)
    paths = _resolve_snapshot_paths(validation_snapshot_paths, kind="validation")
    merged = merge_qkv_transport_calibration_files(
        paths,
        expected_source_ranks=range(_NUM_RANKS),
        expected_observer_mode="validation",
    )
    if not merged.is_validation:
        raise ValueError("finalize requires scale-validation snapshots")
    _validate_complete_snapshot(merged, kind="validation")
    if merged.provenance != derived_provenance:
        raise ValueError("validation campaign provenance does not exactly match the calibration draft")
    assert merged.validation_scales is not None
    if not torch.equal(merged.validation_scales, scales):
        raise ValueError(
            "validation scales do not exactly match the derived scale artifact: "
            f"derived_payload_sha256={derived_payload_sha256}"
        )

    audit = audit_qkv_e4m3_scales(merged)
    if not audit.coverage_complete:
        raise ValueError("validation does not cover all 50x3 QKV cells")
    nonfinite_count = int(audit.nonfinite_count.sum().item())
    if nonfinite_count:
        raise ValueError(f"validation contains {nonfinite_count} non-finite values")
    saturation_count = int(audit.saturation_count.sum().item())
    if saturation_count:
        raise ValueError(f"validation contains {saturation_count} E4M3 saturations")
    if not audit.passed:
        raise ValueError("validation audit did not pass")

    if dit_quant_mode in {
        MINIMAX_H3_ALL_MAIN_FP8_DIT_QUANT_MODE,
        MINIMAX_H3_MLP_FP8_DIT_QUANT_MODE,
        MINIMAX_H3_MLP_OUT_FP8_DIT_QUANT_MODE,
        MINIMAX_H3_MLP_OUT_QKV_FP8_DIT_QUANT_MODE,
    }:
        if run_contract_path is None:
            raise ValueError(
                "selective-FP8 QKV finalization requires --run-contract from the "
                "calibration campaign"
            )
        contract_path = Path(run_contract_path).expanduser()
        if not contract_path.is_file():
            raise ValueError(f"observer run contract does not exist: {contract_path}")
        raw_contract = contract_path.read_bytes()
        try:
            decoded_contract = json.loads(raw_contract)
        except (UnicodeDecodeError, ValueError) as exc:
            raise ValueError("observer run contract must be canonical JSON") from exc
        if not isinstance(decoded_contract, Mapping):
            raise ValueError("observer run contract must be a JSON object")
        canonical_contract = _canonical_json(decoded_contract).encode("ascii")
        if raw_contract != canonical_contract:
            raise ValueError("observer run contract must use canonical JSON encoding")
        validators = {
            MINIMAX_H3_ALL_MAIN_FP8_DIT_QUANT_MODE: (
                validate_minimax_h3_all_main_fp8_run_contract
            ),
            MINIMAX_H3_MLP_FP8_DIT_QUANT_MODE: (
                validate_minimax_h3_mlp_fp8_run_contract
            ),
            MINIMAX_H3_MLP_OUT_FP8_DIT_QUANT_MODE: (
                validate_minimax_h3_mlp_out_fp8_run_contract
            ),
            MINIMAX_H3_MLP_OUT_QKV_FP8_DIT_QUANT_MODE: (
                validate_minimax_h3_mlp_out_qkv_fp8_run_contract
            ),
        }
        validator = validators[dit_quant_mode]
        validator(
            decoded_contract,
            expected_sha256=derived_provenance.run_contract_sha256,
        )
    elif dit_quant_mode != MINIMAX_H3_BF16_DIT_QUANT_MODE:
        raise ValueError(f"unsupported DiT quantization mode: {dit_quant_mode!r}")
    elif run_contract_path is not None:
        raise ValueError(
            "--run-contract is reserved for explicit selective-FP8 QKV lanes"
        )

    adaln_binding = _read_exact_adaln_binding(adaln_sidecar_path)
    media_profile = _load_media_profile(media_profile_path)
    assert adaln_binding.fasth3_adapter is not None
    tensors = {
        "dequant_scales": scales,
        "observed_absmax": audit.observed_absmax,
        "finite_count": audit.finite_count,
        "nonfinite_count": audit.nonfinite_count,
        "numel": audit.numel,
        "observation_count": audit.observation_count,
        "saturation_count": audit.saturation_count,
    }
    payload_sha256 = digest_minimax_h3_qkv_transport_scale_payload(tensors)
    binding = MiniMaxH3QKVTransportScaleBinding.for_fasth3(
        source_model_revision=derived_provenance.source_model_revision,
        runtime_checkpoint_revision=derived_provenance.runtime_checkpoint_revision,
        runtime_checkpoint_manifest_sha256=(derived_provenance.runtime_checkpoint_manifest_sha256),
        campaign_uuid=derived_provenance.campaign_uuid,
        run_contract_sha256=derived_provenance.run_contract_sha256,
        expected_observation_count=derived_provenance.expected_observation_count,
        adapter=adaln_binding.fasth3_adapter,
        base_schedule=_BASE_SCHEDULE,
        flow_shift=_FLOW_SHIFT,
        audio_flow_shift=_AUDIO_FLOW_SHIFT,
        kernel_provider=_KERNEL_PROVIDER,
        kernel_abi=_KERNEL_ABI,
        media_profile=media_profile,
        payload_sha256=payload_sha256,
        dit_quant_mode=dit_quant_mode,
    )
    target = _atomic_save_safetensors(
        output,
        tensors,
        binding.to_metadata(),
        draft_overwrite=draft_overwrite,
    )
    round_trip = prevalidate_minimax_h3_qkv_transport_scale_sidecar(
        target,
        source_model_revision=derived_provenance.source_model_revision,
        runtime_checkpoint_revision=derived_provenance.runtime_checkpoint_revision,
        runtime_checkpoint_manifest_sha256=(derived_provenance.runtime_checkpoint_manifest_sha256),
        campaign_uuid=derived_provenance.campaign_uuid,
        run_contract_sha256=derived_provenance.run_contract_sha256,
        expected_observation_count=derived_provenance.expected_observation_count,
        adapter=adaln_binding.fasth3_adapter,
        base_schedule=_BASE_SCHEDULE,
        flow_shift=_FLOW_SHIFT,
        audio_flow_shift=_AUDIO_FLOW_SHIFT,
        kernel_provider=_KERNEL_PROVIDER,
        kernel_abi=_KERNEL_ABI,
        media_profile=media_profile,
        expected_payload_sha256=payload_sha256,
        dit_quant_mode=dit_quant_mode,
    )
    if round_trip != binding:
        raise AssertionError("published QKV scale binding did not round-trip exactly")

    return {
        "stage": "finalize",
        "output": str(target),
        "whole_sha256": _file_sha256(target),
        "payload_sha256": payload_sha256,
        "derived_payload_sha256": derived_payload_sha256,
        "scale_summary": _scale_summary(scales),
        "validation_audit": {
            **_snapshot_summary(merged),
            "coverage_complete": audit.coverage_complete,
            "saturation_count": saturation_count,
            "saturation_fraction_max": float(audit.saturation_fraction.max().item()),
            "peak_utilization_max": float(audit.peak_utilization.max().item()),
            "passed": audit.passed,
        },
        "binding_summary": {
            "source_model_revision": binding.source_model_revision,
            "runtime_checkpoint_revision": binding.runtime_checkpoint_revision,
            "runtime_checkpoint_manifest_scope": binding.runtime_checkpoint_manifest_scope,
            "runtime_checkpoint_manifest_sha256": binding.runtime_checkpoint_manifest_sha256,
            "campaign_uuid": binding.campaign_uuid,
            "run_contract_sha256": binding.run_contract_sha256,
            "expected_observation_count": binding.expected_observation_count,
            "fasth3_adapter_sha256": binding.fasth3_adapter.sha256,
            "base_schedule": list(binding.base_schedule),
            "flow_shift": binding.flow_shift,
            "audio_flow_shift": binding.audio_flow_shift,
            "kernel_provider": binding.kernel_provider,
            "kernel_abi": binding.kernel_abi,
            "dit_quant_mode": binding.dit_quant_mode,
            "dit_quant_scope": binding.dit_quant_scope,
            "dit_quant_config_sha256": binding.dit_quant_config_sha256,
            "dit_quant_contract_sha256": binding.dit_quant_contract_sha256,
            "prompt_sha256": binding.media_profile.prompt_sha256,
            "width": binding.media_profile.width,
            "height": binding.media_profile.height,
            "fps": binding.media_profile.fps,
            "num_frames": binding.media_profile.num_frames,
            "requested_duration_seconds": binding.media_profile.requested_duration_seconds,
            "seed": binding.media_profile.seed,
        },
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build and validate MiniMax H3 static E4M3 QKV transport scales.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    manifest = subparsers.add_parser(
        "manifest",
        help="hash the exact runtime QKV input-conditioning checkpoint once",
    )
    manifest.add_argument("--runtime-root", type=Path, required=True)
    manifest.add_argument("--output", type=Path, required=True)

    derive = subparsers.add_parser(
        "derive",
        help="merge eight calibration snapshots and emit observer scales",
    )
    derive.add_argument(
        "--calibration-snapshot",
        type=Path,
        nargs="+",
        required=True,
        help="exactly eight rank-local calibration snapshot files",
    )
    derive.add_argument("--output", type=Path, required=True)
    derive.add_argument("--draft-overwrite", action="store_true")

    finalize = subparsers.add_parser(
        "finalize",
        help="audit eight validation snapshots and emit the bound v2 sidecar",
    )
    finalize.add_argument("--derived-scales", type=Path, required=True)
    finalize.add_argument(
        "--validation-snapshot",
        type=Path,
        nargs="+",
        required=True,
        help="exactly eight rank-local validation snapshot files",
    )
    finalize.add_argument("--adaln-sidecar", type=Path, required=True)
    finalize.add_argument(
        "--dit-quant-mode",
        choices=(
            MINIMAX_H3_ALL_MAIN_FP8_DIT_QUANT_MODE,
            MINIMAX_H3_BF16_DIT_QUANT_MODE,
            MINIMAX_H3_MLP_FP8_DIT_QUANT_MODE,
            MINIMAX_H3_MLP_OUT_FP8_DIT_QUANT_MODE,
            MINIMAX_H3_MLP_OUT_QKV_FP8_DIT_QUANT_MODE,
        ),
        default=MINIMAX_H3_BF16_DIT_QUANT_MODE,
    )
    finalize.add_argument(
        "--run-contract",
        type=Path,
        help=(
            "canonical calibration run contract; required only for the "
            "selective-FP8 DiT quantization modes"
        ),
    )
    finalize.add_argument(
        "--media-profile",
        type=Path,
        help=(
            "canonical JSON MiniMax-H3 request/packing profile; omit only for "
            "the legacy kitchen-prompt campaign"
        ),
    )
    finalize.add_argument("--output", type=Path, required=True)
    finalize.add_argument("--draft-overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.command == "manifest":
        result = build_runtime_checkpoint_manifest(args.runtime_root, args.output)
    elif args.command == "derive":
        result = derive_scale_artifact(
            args.calibration_snapshot,
            args.output,
            draft_overwrite=args.draft_overwrite,
        )
    elif args.command == "finalize":
        result = finalize_scale_sidecar(
            args.derived_scales,
            args.validation_snapshot,
            args.adaln_sidecar,
            args.output,
            draft_overwrite=args.draft_overwrite,
            dit_quant_mode=args.dit_quant_mode,
            run_contract_path=args.run_contract,
            media_profile_path=args.media_profile,
        )
    else:  # pragma: no cover - argparse owns this invariant.
        raise AssertionError(f"unexpected command: {args.command}")
    print(f"{_OUTPUT_MARKER}{_canonical_json(result)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
