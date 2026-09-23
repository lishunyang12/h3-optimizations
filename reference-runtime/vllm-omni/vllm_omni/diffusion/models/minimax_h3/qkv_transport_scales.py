# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Bound static E4M3 QKV transport scales for the MiniMax H3 DiT.

The scale file is deliberately a separate, small safetensors artifact.  Static
transport quantization is only safe for the exact student, sampling schedule,
VSA implementation, distributed topology, and media profile used during scale
calibration.  Version 2 therefore fails closed unless every one of those
identities matches a runner-prevalidated runtime binding.

The runner reads and hashes the complete sidecar once, then serializes
``MiniMaxH3QKVTransportScaleBinding.to_dict()`` to diffusion workers.  A worker
validates the header against that binding and recomputes the tensor-payload
digest while loading the tiny CPU FP32 scale table.  No GPU allocation occurs
in this module.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import safe_open

from vllm_omni.diffusion.models.minimax_h3.adaln_cache import (
    MiniMaxH3FastH3AdapterIdentity,
)
from vllm_omni.diffusion.models.minimax_h3.fp8_contract import (
    MINIMAX_H3_ALL_MAIN_FP8_DIT_QUANT_MODE,
    MINIMAX_H3_ALL_MAIN_FP8_SCOPE,
    MINIMAX_H3_BF16_DIT_QUANT_MODE,
    MINIMAX_H3_MLP_FP8_DIT_QUANT_MODE,
    MINIMAX_H3_MLP_FP8_SCOPE,
    MINIMAX_H3_MLP_OUT_FP8_DIT_QUANT_MODE,
    MINIMAX_H3_MLP_OUT_FP8_SCOPE,
    MINIMAX_H3_MLP_OUT_QKV_FP8_DIT_QUANT_MODE,
    MINIMAX_H3_MLP_OUT_QKV_FP8_SCOPE,
    minimax_h3_all_main_fp8_config_sha256,
    minimax_h3_all_main_fp8_contract_sha256,
    minimax_h3_mlp_fp8_config_sha256,
    minimax_h3_mlp_fp8_contract_sha256,
    minimax_h3_mlp_out_fp8_config_sha256,
    minimax_h3_mlp_out_fp8_contract_sha256,
    minimax_h3_mlp_out_qkv_fp8_config_sha256,
    minimax_h3_mlp_out_qkv_fp8_contract_sha256,
)
from vllm_omni.diffusion.sched.sigma_schedule import DMD2SigmaSchedule

_ARTIFACT_TYPE = "minimax-h3-qkv-e4m3-transport-scales"
_FORMAT_VERSION = "2"
_BASE_MODEL = "MiniMaxAI/MiniMax-H3"
_MODEL_VARIANT = "fl2va"
_MODE = "t2va"
_FP8_DTYPE = "float8_e4m3fn"
_SCALE_SEMANTICS = "bf16_value=fp8_code*dequant_scale"
_NUM_LAYERS = 50
_NUM_QKV = 3
_VSA_TOPK = 162
_VSA_TILE_SIZE = 64
_TENSOR_PARALLEL_SIZE = 1
_SEQUENCE_PARALLEL_SIZE = 8
_GPU_ARCHITECTURE = "sm120"
_DIT_QUANT_MODE = MINIMAX_H3_BF16_DIT_QUANT_MODE
_SHA256_HEX_LENGTH = 64

# Exact VSA implementation calibrated by this transport ABI.
MINIMAX_H3_QKV_TRANSPORT_KERNEL_PROVIDER = "flashinfer-pr4944-cutedsl-sm120"
MINIMAX_H3_QKV_TRANSPORT_KERNEL_ABI = "minimax-h3-vsa-qkv-e4m3-bf16-o-v1"
MINIMAX_H3_QKV_RUNTIME_CHECKPOINT_MANIFEST_SCOPE = "qkv-input-conditioning-v1"

_BINDING_METADATA_KEY = "minimax_h3_qkv_transport_binding"
_PAYLOAD_DIGEST_METADATA_KEY = "minimax_h3_qkv_transport_payload_sha256"
_METADATA_KEYS = frozenset({_BINDING_METADATA_KEY, _PAYLOAD_DIGEST_METADATA_KEY})
_DEQUANT_SCALES_KEY = "dequant_scales"
_AUDIT_TENSOR_SPECS = {
    "observed_absmax": torch.float32,
    "finite_count": torch.int64,
    "nonfinite_count": torch.int64,
    "numel": torch.int64,
    "observation_count": torch.int64,
    "saturation_count": torch.int64,
}
_PAYLOAD_DIGEST_DOMAIN = b"minimax-h3-qkv-e4m3-transport-scales-v2\0"
_RUNTIME_MANIFEST_DIGEST_DOMAIN = b"minimax-h3-runtime-checkpoint-manifest-v1\0"


def _dit_quant_identity(
    dit_quant_mode: str,
) -> tuple[str, str, str] | None:
    if dit_quant_mode == MINIMAX_H3_BF16_DIT_QUANT_MODE:
        return None
    if dit_quant_mode == MINIMAX_H3_MLP_FP8_DIT_QUANT_MODE:
        return (
            MINIMAX_H3_MLP_FP8_SCOPE,
            minimax_h3_mlp_fp8_config_sha256(),
            minimax_h3_mlp_fp8_contract_sha256(),
        )
    if dit_quant_mode == MINIMAX_H3_MLP_OUT_FP8_DIT_QUANT_MODE:
        return (
            MINIMAX_H3_MLP_OUT_FP8_SCOPE,
            minimax_h3_mlp_out_fp8_config_sha256(),
            minimax_h3_mlp_out_fp8_contract_sha256(),
        )
    if dit_quant_mode == MINIMAX_H3_MLP_OUT_QKV_FP8_DIT_QUANT_MODE:
        return (
            MINIMAX_H3_MLP_OUT_QKV_FP8_SCOPE,
            minimax_h3_mlp_out_qkv_fp8_config_sha256(),
            minimax_h3_mlp_out_qkv_fp8_contract_sha256(),
        )
    if dit_quant_mode == MINIMAX_H3_ALL_MAIN_FP8_DIT_QUANT_MODE:
        return (
            MINIMAX_H3_ALL_MAIN_FP8_SCOPE,
            minimax_h3_all_main_fp8_config_sha256(),
            minimax_h3_all_main_fp8_contract_sha256(),
        )
    raise ValueError(
        "dit_quant_mode must identify the BF16 DiT or an exact qualified "
        "SM120 Cutlass selective-FP8 contract"
    )


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _validate_sha256(value: str, *, field: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field} must be a string")
    normalized = value.lower()
    if len(normalized) != _SHA256_HEX_LENGTH:
        raise ValueError(f"{field} must be a 64-character SHA256 digest")
    try:
        int(normalized, 16)
    except ValueError as exc:
        raise ValueError(f"{field} must be a hexadecimal SHA256 digest") from exc
    return normalized


def _validate_git_commit(value: str, *, field: str) -> str:
    revision = _require_string(value, field=field)
    if len(revision) != 40:
        raise ValueError(f"{field} must be a 40-character git commit")
    try:
        int(revision, 16)
    except ValueError as exc:
        raise ValueError(f"{field} must be a hexadecimal git commit") from exc
    return revision.lower()


def _require_string(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError(f"{field} must be a non-empty canonical string")
    return value


def _require_int(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field} must be an integer")
    return value


def _require_float(value: Any, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field} must be a JSON number")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{field} must be finite")
    return normalized


def _require_exact_fields(
    value: Mapping[str, Any],
    expected: set[str],
    *,
    kind: str,
) -> None:
    unknown = set(value) - expected
    missing = expected - set(value)
    if missing or unknown:
        raise ValueError(f"invalid MiniMax H3 QKV {kind} fields: missing={sorted(missing)}, unknown={sorted(unknown)}")


@dataclass(frozen=True)
class MiniMaxH3QKVTransportCalibrationProvenance:
    """Immutable identity shared by calibration, validation, and serving.

    ``campaign_uuid`` is intentionally opaque.  A launcher generates it once
    for the complete calibration campaign and carries it through validation;
    per-process execution IDs belong in launcher logs and are not lineage.
    """

    source_model_revision: str
    runtime_checkpoint_revision: str
    runtime_checkpoint_manifest_sha256: str
    campaign_uuid: str
    run_contract_sha256: str
    expected_observation_count: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "source_model_revision",
            _validate_git_commit(self.source_model_revision, field="source_model_revision"),
        )
        object.__setattr__(
            self,
            "runtime_checkpoint_revision",
            _validate_git_commit(
                self.runtime_checkpoint_revision,
                field="runtime_checkpoint_revision",
            ),
        )
        object.__setattr__(
            self,
            "runtime_checkpoint_manifest_sha256",
            _validate_sha256(
                self.runtime_checkpoint_manifest_sha256,
                field="runtime_checkpoint_manifest_sha256",
            ),
        )
        object.__setattr__(
            self,
            "campaign_uuid",
            _require_string(self.campaign_uuid, field="campaign_uuid"),
        )
        object.__setattr__(
            self,
            "run_contract_sha256",
            _validate_sha256(self.run_contract_sha256, field="run_contract_sha256"),
        )
        count = _require_int(
            self.expected_observation_count,
            field="expected_observation_count",
        )
        if count <= 0:
            raise ValueError("expected_observation_count must be positive")

    def to_dict(self) -> dict[str, str | int]:
        return {
            "source_model_revision": self.source_model_revision,
            "runtime_checkpoint_revision": self.runtime_checkpoint_revision,
            "runtime_checkpoint_manifest_sha256": self.runtime_checkpoint_manifest_sha256,
            "campaign_uuid": self.campaign_uuid,
            "run_contract_sha256": self.run_contract_sha256,
            "expected_observation_count": self.expected_observation_count,
        }

    @classmethod
    def from_dict(
        cls,
        value: Mapping[str, Any],
    ) -> MiniMaxH3QKVTransportCalibrationProvenance:
        expected = {
            "source_model_revision",
            "runtime_checkpoint_revision",
            "runtime_checkpoint_manifest_sha256",
            "campaign_uuid",
            "run_contract_sha256",
            "expected_observation_count",
        }
        _require_exact_fields(value, expected, kind="calibration provenance")
        return cls(
            source_model_revision=_require_string(value["source_model_revision"], field="source_model_revision"),
            runtime_checkpoint_revision=_require_string(
                value["runtime_checkpoint_revision"],
                field="runtime_checkpoint_revision",
            ),
            runtime_checkpoint_manifest_sha256=_require_string(
                value["runtime_checkpoint_manifest_sha256"],
                field="runtime_checkpoint_manifest_sha256",
            ),
            campaign_uuid=_require_string(value["campaign_uuid"], field="campaign_uuid"),
            run_contract_sha256=_require_string(value["run_contract_sha256"], field="run_contract_sha256"),
            expected_observation_count=_require_int(
                value["expected_observation_count"],
                field="expected_observation_count",
            ),
        )


def canonical_minimax_h3_runtime_checkpoint_manifest(
    entries: Sequence[Mapping[str, Any]],
) -> bytes:
    """Return canonical bytes for a precomputed runtime-file manifest.

    This helper is intended for a one-shot artifact-preparation command.  DiT
    workers consume only its SHA256 digest and never rehash multi-GB shards.
    """

    canonical_entries: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in entries:
        expected = {"path", "size_bytes", "sha256"}
        _require_exact_fields(raw, expected, kind="runtime manifest entry")
        path = _require_string(raw["path"], field="runtime manifest entry path")
        path_parts = path.split("/")
        if path.startswith("/") or "\\" in path or any(part in {"", ".", ".."} for part in path_parts):
            raise ValueError("runtime manifest paths must be normalized relative POSIX paths")
        if path in seen:
            raise ValueError(f"duplicate runtime manifest path: {path!r}")
        seen.add(path)
        size = _require_int(raw["size_bytes"], field=f"runtime manifest {path!r} size_bytes")
        if size < 0:
            raise ValueError("runtime manifest size_bytes must be nonnegative")
        canonical_entries.append(
            {
                "path": path,
                "size_bytes": size,
                "sha256": _validate_sha256(
                    _require_string(raw["sha256"], field=f"runtime manifest {path!r} sha256"),
                    field=f"runtime manifest {path!r} sha256",
                ),
            }
        )
    if not canonical_entries:
        raise ValueError("runtime checkpoint manifest must contain at least one file")
    canonical_entries.sort(key=lambda entry: entry["path"])
    envelope = {
        "artifact_type": "minimax-h3-runtime-checkpoint-manifest",
        "format_version": "1",
        "scope": MINIMAX_H3_QKV_RUNTIME_CHECKPOINT_MANIFEST_SCOPE,
        "files": canonical_entries,
    }
    return _canonical_json(envelope).encode("ascii")


def digest_minimax_h3_runtime_checkpoint_manifest(
    entries: Sequence[Mapping[str, Any]],
) -> str:
    """Digest canonical precomputed file identities without reading the files."""

    digest = hashlib.sha256(_RUNTIME_MANIFEST_DIGEST_DOMAIN)
    digest.update(canonical_minimax_h3_runtime_checkpoint_manifest(entries))
    return digest.hexdigest()


def read_minimax_h3_runtime_checkpoint_manifest(
    path: str | os.PathLike[str],
) -> str:
    """Validate a precomputed canonical manifest and return its bound digest."""

    candidate = Path(path).expanduser()
    if not candidate.is_file() or candidate.suffix != ".json":
        raise ValueError(f"runtime checkpoint manifest must be an existing JSON file: {candidate}")
    raw = candidate.read_bytes()
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, ValueError) as exc:
        raise ValueError("runtime checkpoint manifest must be canonical ASCII JSON") from exc
    if not isinstance(value, dict) or set(value) != {
        "artifact_type",
        "format_version",
        "scope",
        "files",
    }:
        raise ValueError("runtime checkpoint manifest envelope is invalid")
    if value["artifact_type"] != "minimax-h3-runtime-checkpoint-manifest":
        raise ValueError("runtime checkpoint manifest artifact_type is invalid")
    if value["format_version"] != "1":
        raise ValueError("runtime checkpoint manifest format_version is invalid")
    if value["scope"] != MINIMAX_H3_QKV_RUNTIME_CHECKPOINT_MANIFEST_SCOPE:
        raise ValueError("runtime checkpoint manifest scope is invalid")
    entries = value["files"]
    if not isinstance(entries, list):
        raise TypeError("runtime checkpoint manifest files must be a JSON list")
    canonical = canonical_minimax_h3_runtime_checkpoint_manifest(entries)
    if raw != canonical:
        raise ValueError("runtime checkpoint manifest must use exact canonical encoding")
    return digest_minimax_h3_runtime_checkpoint_manifest(entries)


@dataclass(frozen=True)
class MiniMaxH3QKVTransportMediaProfile:
    """Exact 15-second benchmark media identity used for scale calibration."""

    width: int
    height: int
    fps: int
    num_frames: int
    requested_duration_seconds: float
    seed: int
    prompt_sha256: str
    effective_width: int
    effective_height: int
    video_shape: tuple[int, int, int]
    video_tile_count: int
    prefix_tile_count: int
    compact_row_count: int
    padded_row_count: int

    def __post_init__(self) -> None:
        for field, expected in (
            ("width", 1280),
            ("height", 720),
            ("fps", 24),
            ("num_frames", 362),
            ("seed", 1101),
        ):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"media_profile.{field} must be an integer")
            if value != expected:
                raise ValueError(f"media_profile.{field} must be {expected} for the v2 profile")
        duration = float(self.requested_duration_seconds)
        if not math.isfinite(duration) or duration != 15.0:
            raise ValueError("media_profile.requested_duration_seconds must be 15.0 for v2")
        object.__setattr__(self, "requested_duration_seconds", duration)
        object.__setattr__(
            self,
            "prompt_sha256",
            _validate_sha256(
                self.prompt_sha256,
                field="media_profile.prompt_sha256",
            ),
        )
        fixed_geometry = {
            "effective_width": 1280,
            "effective_height": 704,
            "video_tile_count": 1620,
        }
        for field, expected in fixed_geometry.items():
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"media_profile.{field} must be an integer")
            if value != expected:
                raise ValueError(f"media_profile.{field} must be {expected} for the v2 profile")
        shape = tuple(self.video_shape)
        if shape != (107, 22, 40):
            raise ValueError("media_profile.video_shape must be (107, 22, 40) for the v2 profile")
        object.__setattr__(self, "video_shape", shape)
        for field in (
            "prefix_tile_count",
            "compact_row_count",
            "padded_row_count",
        ):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"media_profile.{field} must be an integer")
            if value <= 0:
                raise ValueError(f"media_profile.{field} must be positive")
        if self.video_tile_count * _VSA_TILE_SIZE != 103680:
            raise AssertionError("fixed MiniMax H3 VSA video tile geometry is inconsistent")
        if (self.video_tile_count + self.prefix_tile_count) * _VSA_TILE_SIZE != self.padded_row_count:
            raise ValueError("media_profile tile counts do not match padded_row_count")
        if math.prod(self.video_shape) >= self.compact_row_count:
            raise ValueError("media_profile compact_row_count must include a non-video prefix")
        if self.compact_row_count > math.prod(self.video_shape) + self.prefix_tile_count * _VSA_TILE_SIZE:
            raise ValueError("media_profile compact prefix exceeds prefix_tile_count capacity")
        if self.compact_row_count > self.padded_row_count:
            raise ValueError("media_profile compact_row_count exceeds padded_row_count")

    def to_dict(self) -> dict[str, Any]:
        return {
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "num_frames": self.num_frames,
            "requested_duration_seconds": self.requested_duration_seconds,
            "seed": self.seed,
            "prompt_sha256": self.prompt_sha256,
            "effective_width": self.effective_width,
            "effective_height": self.effective_height,
            "video_shape": list(self.video_shape),
            "video_tile_count": self.video_tile_count,
            "prefix_tile_count": self.prefix_tile_count,
            "compact_row_count": self.compact_row_count,
            "padded_row_count": self.padded_row_count,
        }

    @classmethod
    def from_dict(
        cls,
        value: Mapping[str, Any],
    ) -> MiniMaxH3QKVTransportMediaProfile:
        expected = {
            "width",
            "height",
            "fps",
            "num_frames",
            "requested_duration_seconds",
            "seed",
            "prompt_sha256",
            "effective_width",
            "effective_height",
            "video_shape",
            "video_tile_count",
            "prefix_tile_count",
            "compact_row_count",
            "padded_row_count",
        }
        _require_exact_fields(value, expected, kind="media profile")
        raw_video_shape = value["video_shape"]
        if not isinstance(raw_video_shape, list) or len(raw_video_shape) != 3:
            raise TypeError("media_profile.video_shape must be a JSON list of length 3")
        return cls(
            width=_require_int(value["width"], field="media_profile.width"),
            height=_require_int(value["height"], field="media_profile.height"),
            fps=_require_int(value["fps"], field="media_profile.fps"),
            num_frames=_require_int(value["num_frames"], field="media_profile.num_frames"),
            requested_duration_seconds=_require_float(
                value["requested_duration_seconds"],
                field="media_profile.requested_duration_seconds",
            ),
            seed=_require_int(value["seed"], field="media_profile.seed"),
            prompt_sha256=_require_string(value["prompt_sha256"], field="media_profile.prompt_sha256"),
            effective_width=_require_int(value["effective_width"], field="media_profile.effective_width"),
            effective_height=_require_int(value["effective_height"], field="media_profile.effective_height"),
            video_shape=tuple(_require_int(item, field="media_profile.video_shape entry") for item in raw_video_shape),
            video_tile_count=_require_int(value["video_tile_count"], field="media_profile.video_tile_count"),
            prefix_tile_count=_require_int(value["prefix_tile_count"], field="media_profile.prefix_tile_count"),
            compact_row_count=_require_int(value["compact_row_count"], field="media_profile.compact_row_count"),
            padded_row_count=_require_int(value["padded_row_count"], field="media_profile.padded_row_count"),
        )


@dataclass(frozen=True)
class MiniMaxH3QKVTransportScaleBinding:
    """Serializable identity of one calibrated QKV transport-scale payload."""

    format_version: str
    artifact_type: str
    base_model: str
    source_model_revision: str
    runtime_checkpoint_revision: str
    runtime_checkpoint_manifest_scope: str
    runtime_checkpoint_manifest_sha256: str
    campaign_uuid: str
    run_contract_sha256: str
    expected_observation_count: int
    model_variant: str
    mode: str
    fasth3_adapter: MiniMaxH3FastH3AdapterIdentity
    base_schedule: tuple[float, ...]
    flow_shift: float
    audio_flow_shift: float
    fp8_dtype: str
    scale_semantics: str
    vsa_topk: int
    vsa_tile_size: int
    kernel_provider: str
    kernel_abi: str
    tensor_parallel_size: int
    sequence_parallel_size: int
    gpu_architecture: str
    dit_quant_mode: str
    media_profile: MiniMaxH3QKVTransportMediaProfile
    payload_sha256: str
    dit_quant_scope: str | None = None
    dit_quant_config_sha256: str | None = None
    dit_quant_contract_sha256: str | None = None

    def __post_init__(self) -> None:
        fixed_strings = {
            "format_version": _FORMAT_VERSION,
            "artifact_type": _ARTIFACT_TYPE,
            "base_model": _BASE_MODEL,
            "model_variant": _MODEL_VARIANT,
            "mode": _MODE,
            "fp8_dtype": _FP8_DTYPE,
            "scale_semantics": _SCALE_SEMANTICS,
            "gpu_architecture": _GPU_ARCHITECTURE,
            "runtime_checkpoint_manifest_scope": MINIMAX_H3_QKV_RUNTIME_CHECKPOINT_MANIFEST_SCOPE,
        }
        for field, expected in fixed_strings.items():
            if getattr(self, field) != expected:
                raise ValueError(f"{field} must be {expected!r} for the v2 scale ABI")

        expected_quant_identity = _dit_quant_identity(self.dit_quant_mode)
        if expected_quant_identity is None:
            extra_quant_identity = (
                self.dit_quant_scope,
                self.dit_quant_config_sha256,
                self.dit_quant_contract_sha256,
            )
            if extra_quant_identity != (None, None, None):
                raise ValueError(
                    "BF16 QKV scale bindings must not carry a selective-FP8 identity"
                )
        else:
            expected_fields = dict(
                zip(
                    (
                        "dit_quant_scope",
                        "dit_quant_config_sha256",
                        "dit_quant_contract_sha256",
                    ),
                    expected_quant_identity,
                    strict=True,
                )
            )
            for field, expected in expected_fields.items():
                actual = getattr(self, field)
                if actual != expected:
                    raise ValueError(
                        f"{field} must be {expected!r} for the selective-FP8 QKV scale ABI"
                    )

        provenance = MiniMaxH3QKVTransportCalibrationProvenance(
            source_model_revision=self.source_model_revision,
            runtime_checkpoint_revision=self.runtime_checkpoint_revision,
            runtime_checkpoint_manifest_sha256=self.runtime_checkpoint_manifest_sha256,
            campaign_uuid=self.campaign_uuid,
            run_contract_sha256=self.run_contract_sha256,
            expected_observation_count=self.expected_observation_count,
        )
        for field, value in provenance.to_dict().items():
            object.__setattr__(self, field, value)

        if not isinstance(self.fasth3_adapter, MiniMaxH3FastH3AdapterIdentity):
            raise TypeError("fasth3_adapter must be a MiniMaxH3FastH3AdapterIdentity")

        schedule = DMD2SigmaSchedule.from_positions(self.base_schedule)
        if schedule.num_inference_steps != 4:
            raise ValueError("QKV transport scales v2 require the four-forward FastH3 schedule")
        object.__setattr__(self, "base_schedule", schedule.base_schedule)

        for field in ("flow_shift", "audio_flow_shift"):
            value = float(getattr(self, field))
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{field} must be finite and positive")
            object.__setattr__(self, field, value)

        fixed_ints = {
            "vsa_topk": _VSA_TOPK,
            "vsa_tile_size": _VSA_TILE_SIZE,
            "tensor_parallel_size": _TENSOR_PARALLEL_SIZE,
            "sequence_parallel_size": _SEQUENCE_PARALLEL_SIZE,
        }
        for field, expected in fixed_ints.items():
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{field} must be an integer")
            if value != expected:
                raise ValueError(f"{field} must be {expected} for the v2 scale ABI")

        object.__setattr__(
            self,
            "kernel_provider",
            _require_string(self.kernel_provider, field="kernel_provider"),
        )
        object.__setattr__(
            self,
            "kernel_abi",
            _require_string(self.kernel_abi, field="kernel_abi"),
        )
        if self.kernel_provider != MINIMAX_H3_QKV_TRANSPORT_KERNEL_PROVIDER:
            raise ValueError("kernel_provider must identify the calibrated FlashInfer PR #4944 CuTeDSL path")
        if self.kernel_abi != MINIMAX_H3_QKV_TRANSPORT_KERNEL_ABI:
            raise ValueError("kernel_abi does not match the calibrated QKV transport ABI")
        if not isinstance(self.media_profile, MiniMaxH3QKVTransportMediaProfile):
            raise TypeError("media_profile must be a MiniMaxH3QKVTransportMediaProfile")
        object.__setattr__(
            self,
            "payload_sha256",
            _validate_sha256(self.payload_sha256, field="payload_sha256"),
        )

    @property
    def calibration_provenance(self) -> MiniMaxH3QKVTransportCalibrationProvenance:
        return MiniMaxH3QKVTransportCalibrationProvenance(
            source_model_revision=self.source_model_revision,
            runtime_checkpoint_revision=self.runtime_checkpoint_revision,
            runtime_checkpoint_manifest_sha256=self.runtime_checkpoint_manifest_sha256,
            campaign_uuid=self.campaign_uuid,
            run_contract_sha256=self.run_contract_sha256,
            expected_observation_count=self.expected_observation_count,
        )

    @classmethod
    def for_fasth3(
        cls,
        *,
        source_model_revision: str,
        runtime_checkpoint_revision: str,
        runtime_checkpoint_manifest_sha256: str,
        campaign_uuid: str,
        run_contract_sha256: str,
        expected_observation_count: int,
        adapter: MiniMaxH3FastH3AdapterIdentity,
        base_schedule: Sequence[float],
        flow_shift: float,
        audio_flow_shift: float,
        kernel_provider: str,
        kernel_abi: str,
        media_profile: MiniMaxH3QKVTransportMediaProfile,
        payload_sha256: str,
        dit_quant_mode: str = _DIT_QUANT_MODE,
    ) -> MiniMaxH3QKVTransportScaleBinding:
        quant_identity = _dit_quant_identity(dit_quant_mode)
        return cls(
            format_version=_FORMAT_VERSION,
            artifact_type=_ARTIFACT_TYPE,
            base_model=_BASE_MODEL,
            source_model_revision=source_model_revision,
            runtime_checkpoint_revision=runtime_checkpoint_revision,
            runtime_checkpoint_manifest_scope=MINIMAX_H3_QKV_RUNTIME_CHECKPOINT_MANIFEST_SCOPE,
            runtime_checkpoint_manifest_sha256=runtime_checkpoint_manifest_sha256,
            campaign_uuid=campaign_uuid,
            run_contract_sha256=run_contract_sha256,
            expected_observation_count=expected_observation_count,
            model_variant=_MODEL_VARIANT,
            mode=_MODE,
            fasth3_adapter=adapter,
            base_schedule=tuple(float(value) for value in base_schedule),
            flow_shift=float(flow_shift),
            audio_flow_shift=float(audio_flow_shift),
            fp8_dtype=_FP8_DTYPE,
            scale_semantics=_SCALE_SEMANTICS,
            vsa_topk=_VSA_TOPK,
            vsa_tile_size=_VSA_TILE_SIZE,
            kernel_provider=kernel_provider,
            kernel_abi=kernel_abi,
            tensor_parallel_size=_TENSOR_PARALLEL_SIZE,
            sequence_parallel_size=_SEQUENCE_PARALLEL_SIZE,
            gpu_architecture=_GPU_ARCHITECTURE,
            dit_quant_mode=dit_quant_mode,
            media_profile=media_profile,
            payload_sha256=payload_sha256,
            dit_quant_scope=(None if quant_identity is None else quant_identity[0]),
            dit_quant_config_sha256=(
                None if quant_identity is None else quant_identity[1]
            ),
            dit_quant_contract_sha256=(
                None if quant_identity is None else quant_identity[2]
            ),
        )

    def _contract_dict(self) -> dict[str, Any]:
        contract = {
            "format_version": self.format_version,
            "artifact_type": self.artifact_type,
            "base_model": self.base_model,
            "source_model_revision": self.source_model_revision,
            "runtime_checkpoint_revision": self.runtime_checkpoint_revision,
            "runtime_checkpoint_manifest_scope": self.runtime_checkpoint_manifest_scope,
            "runtime_checkpoint_manifest_sha256": self.runtime_checkpoint_manifest_sha256,
            "campaign_uuid": self.campaign_uuid,
            "run_contract_sha256": self.run_contract_sha256,
            "expected_observation_count": self.expected_observation_count,
            "model_variant": self.model_variant,
            "mode": self.mode,
            "fasth3_adapter": self.fasth3_adapter.to_dict(),
            "base_schedule": list(self.base_schedule),
            "flow_shift": self.flow_shift,
            "audio_flow_shift": self.audio_flow_shift,
            "fp8_dtype": self.fp8_dtype,
            "scale_semantics": self.scale_semantics,
            "vsa_topk": self.vsa_topk,
            "vsa_tile_size": self.vsa_tile_size,
            "kernel_provider": self.kernel_provider,
            "kernel_abi": self.kernel_abi,
            "tensor_parallel_size": self.tensor_parallel_size,
            "sequence_parallel_size": self.sequence_parallel_size,
            "gpu_architecture": self.gpu_architecture,
            "dit_quant_mode": self.dit_quant_mode,
            "media_profile": self.media_profile.to_dict(),
        }
        if _dit_quant_identity(self.dit_quant_mode) is not None:
            contract.update(
                {
                    "dit_quant_scope": self.dit_quant_scope,
                    "dit_quant_config_sha256": self.dit_quant_config_sha256,
                    "dit_quant_contract_sha256": self.dit_quant_contract_sha256,
                }
            )
        return contract

    def to_dict(self) -> dict[str, Any]:
        return {**self._contract_dict(), "payload_sha256": self.payload_sha256}

    def to_metadata(self) -> dict[str, str]:
        return {
            _BINDING_METADATA_KEY: _canonical_json(self._contract_dict()),
            _PAYLOAD_DIGEST_METADATA_KEY: self.payload_sha256,
        }

    @classmethod
    def from_dict(
        cls,
        value: Mapping[str, Any],
    ) -> MiniMaxH3QKVTransportScaleBinding:
        expected = {
            "format_version",
            "artifact_type",
            "base_model",
            "source_model_revision",
            "runtime_checkpoint_revision",
            "runtime_checkpoint_manifest_scope",
            "runtime_checkpoint_manifest_sha256",
            "campaign_uuid",
            "run_contract_sha256",
            "expected_observation_count",
            "model_variant",
            "mode",
            "fasth3_adapter",
            "base_schedule",
            "flow_shift",
            "audio_flow_shift",
            "fp8_dtype",
            "scale_semantics",
            "vsa_topk",
            "vsa_tile_size",
            "kernel_provider",
            "kernel_abi",
            "tensor_parallel_size",
            "sequence_parallel_size",
            "gpu_architecture",
            "dit_quant_mode",
            "media_profile",
            "payload_sha256",
        }
        raw_dit_quant_mode = value.get("dit_quant_mode")
        if raw_dit_quant_mode in {
            MINIMAX_H3_MLP_FP8_DIT_QUANT_MODE,
            MINIMAX_H3_MLP_OUT_FP8_DIT_QUANT_MODE,
            MINIMAX_H3_MLP_OUT_QKV_FP8_DIT_QUANT_MODE,
            MINIMAX_H3_ALL_MAIN_FP8_DIT_QUANT_MODE,
        }:
            expected.update(
                {
                    "dit_quant_scope",
                    "dit_quant_config_sha256",
                    "dit_quant_contract_sha256",
                }
            )
        _require_exact_fields(value, expected, kind="scale binding")
        raw_adapter = value["fasth3_adapter"]
        if not isinstance(raw_adapter, Mapping):
            raise TypeError("fasth3_adapter must be an object")
        raw_schedule = value["base_schedule"]
        if not isinstance(raw_schedule, list):
            raise TypeError("base_schedule must be a JSON list")
        raw_media_profile = value["media_profile"]
        if not isinstance(raw_media_profile, Mapping):
            raise TypeError("media_profile must be an object")
        return cls(
            format_version=_require_string(value["format_version"], field="format_version"),
            artifact_type=_require_string(value["artifact_type"], field="artifact_type"),
            base_model=_require_string(value["base_model"], field="base_model"),
            source_model_revision=_require_string(value["source_model_revision"], field="source_model_revision"),
            runtime_checkpoint_revision=_require_string(
                value["runtime_checkpoint_revision"], field="runtime_checkpoint_revision"
            ),
            runtime_checkpoint_manifest_scope=_require_string(
                value["runtime_checkpoint_manifest_scope"], field="runtime_checkpoint_manifest_scope"
            ),
            runtime_checkpoint_manifest_sha256=_require_string(
                value["runtime_checkpoint_manifest_sha256"],
                field="runtime_checkpoint_manifest_sha256",
            ),
            campaign_uuid=_require_string(value["campaign_uuid"], field="campaign_uuid"),
            run_contract_sha256=_require_string(value["run_contract_sha256"], field="run_contract_sha256"),
            expected_observation_count=_require_int(
                value["expected_observation_count"], field="expected_observation_count"
            ),
            model_variant=_require_string(value["model_variant"], field="model_variant"),
            mode=_require_string(value["mode"], field="mode"),
            fasth3_adapter=MiniMaxH3FastH3AdapterIdentity.from_dict(raw_adapter),
            base_schedule=tuple(_require_float(item, field="base_schedule entry") for item in raw_schedule),
            flow_shift=_require_float(value["flow_shift"], field="flow_shift"),
            audio_flow_shift=_require_float(value["audio_flow_shift"], field="audio_flow_shift"),
            fp8_dtype=_require_string(value["fp8_dtype"], field="fp8_dtype"),
            scale_semantics=_require_string(value["scale_semantics"], field="scale_semantics"),
            vsa_topk=_require_int(value["vsa_topk"], field="vsa_topk"),
            vsa_tile_size=_require_int(value["vsa_tile_size"], field="vsa_tile_size"),
            kernel_provider=_require_string(value["kernel_provider"], field="kernel_provider"),
            kernel_abi=_require_string(value["kernel_abi"], field="kernel_abi"),
            tensor_parallel_size=_require_int(value["tensor_parallel_size"], field="tensor_parallel_size"),
            sequence_parallel_size=_require_int(value["sequence_parallel_size"], field="sequence_parallel_size"),
            gpu_architecture=_require_string(value["gpu_architecture"], field="gpu_architecture"),
            dit_quant_mode=_require_string(value["dit_quant_mode"], field="dit_quant_mode"),
            media_profile=MiniMaxH3QKVTransportMediaProfile.from_dict(raw_media_profile),
            payload_sha256=_require_string(value["payload_sha256"], field="payload_sha256"),
            dit_quant_scope=(
                _require_string(value["dit_quant_scope"], field="dit_quant_scope")
                if "dit_quant_scope" in value
                else None
            ),
            dit_quant_config_sha256=(
                _require_string(
                    value["dit_quant_config_sha256"],
                    field="dit_quant_config_sha256",
                )
                if "dit_quant_config_sha256" in value
                else None
            ),
            dit_quant_contract_sha256=(
                _require_string(
                    value["dit_quant_contract_sha256"],
                    field="dit_quant_contract_sha256",
                )
                if "dit_quant_contract_sha256" in value
                else None
            ),
        )

    @classmethod
    def from_metadata(
        cls,
        metadata: Mapping[str, str],
    ) -> MiniMaxH3QKVTransportScaleBinding:
        unknown = set(metadata) - _METADATA_KEYS
        missing = _METADATA_KEYS - set(metadata)
        if missing or unknown:
            raise ValueError(
                f"invalid MiniMax H3 QKV scale metadata: missing={sorted(missing)}, unknown={sorted(unknown)}"
            )
        encoded = metadata[_BINDING_METADATA_KEY]
        try:
            contract = json.loads(encoded)
        except (TypeError, ValueError) as exc:
            raise ValueError("MiniMax H3 QKV scale binding must be canonical JSON") from exc
        if not isinstance(contract, dict):
            raise ValueError("MiniMax H3 QKV scale binding must be a JSON object")
        if _canonical_json(contract) != encoded:
            raise ValueError("MiniMax H3 QKV scale binding must use canonical JSON encoding")
        return cls.from_dict(
            {
                **contract,
                "payload_sha256": metadata[_PAYLOAD_DIGEST_METADATA_KEY],
            }
        )


def _tensor_payload_bytes(tensor: torch.Tensor) -> bytes:
    cpu_tensor = tensor.detach().to(device="cpu").contiguous()
    return cpu_tensor.view(torch.uint8).numpy().tobytes()


def digest_minimax_h3_qkv_transport_scale_payload(
    tensors: Mapping[str, torch.Tensor],
) -> str:
    """Return the canonical digest over tensor names, schemas, and raw bytes."""
    digest = hashlib.sha256(_PAYLOAD_DIGEST_DOMAIN)
    for name in sorted(tensors):
        tensor = tensors[name]
        if not isinstance(name, str) or not isinstance(tensor, torch.Tensor):
            raise TypeError("QKV scale payload must map string names to tensors")
        if tensor.layout != torch.strided:
            raise ValueError(f"QKV scale tensor {name!r} must use strided layout")
        schema = _canonical_json(
            {
                "dtype": str(tensor.dtype).removeprefix("torch."),
                "name": name,
                "shape": list(tensor.shape),
            }
        ).encode("ascii")
        payload = _tensor_payload_bytes(tensor)
        digest.update(len(schema).to_bytes(8, byteorder="little", signed=False))
        digest.update(schema)
        digest.update(len(payload).to_bytes(8, byteorder="little", signed=False))
        digest.update(payload)
    return digest.hexdigest()


def _coerce_binding(
    binding: MiniMaxH3QKVTransportScaleBinding | Mapping[str, Any],
) -> MiniMaxH3QKVTransportScaleBinding:
    if isinstance(binding, MiniMaxH3QKVTransportScaleBinding):
        return binding
    if isinstance(binding, Mapping):
        return MiniMaxH3QKVTransportScaleBinding.from_dict(binding)
    raise TypeError("MiniMax H3 QKV runtime binding must be a binding or mapping")


def _sidecar_path(path: str | os.PathLike[str]) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_file():
        raise ValueError(f"MiniMax H3 QKV scale sidecar does not exist: {candidate}")
    if candidate.suffix != ".safetensors":
        raise ValueError(f"MiniMax H3 QKV scale sidecar is not a safetensors file: {candidate}")
    return candidate.resolve()


def read_minimax_h3_qkv_transport_scale_binding(
    path: str | os.PathLike[str],
) -> MiniMaxH3QKVTransportScaleBinding:
    """Read and validate the exact, canonical sidecar metadata only."""
    candidate = _sidecar_path(path)
    with safe_open(candidate, framework="pt", device="cpu") as scale_file:
        return MiniMaxH3QKVTransportScaleBinding.from_metadata(scale_file.metadata() or {})


def validate_minimax_h3_qkv_transport_scale_sidecar_binding(
    path: str | os.PathLike[str],
    binding: MiniMaxH3QKVTransportScaleBinding | Mapping[str, Any],
) -> MiniMaxH3QKVTransportScaleBinding:
    """Require the sidecar header to match a runner-owned runtime binding."""
    expected = _coerce_binding(binding)
    actual = read_minimax_h3_qkv_transport_scale_binding(path)
    if actual != expected:
        raise ValueError(
            f"MiniMax H3 QKV scale binding mismatch: expected={expected.to_dict()!r}, actual={actual.to_dict()!r}"
        )
    return actual


def _validate_payload_tensors(
    tensors: Mapping[str, torch.Tensor],
    *,
    binding: MiniMaxH3QKVTransportScaleBinding,
) -> None:
    names = set(tensors)
    required = {_DEQUANT_SCALES_KEY, *_AUDIT_TENSOR_SPECS}
    unknown = names - required
    missing = required - names
    if missing or unknown:
        raise ValueError(f"invalid MiniMax H3 QKV scale tensors: missing={sorted(missing)}, unknown={sorted(unknown)}")
    scales = tensors[_DEQUANT_SCALES_KEY]
    expected_shape = (_NUM_LAYERS, _NUM_QKV)
    if scales.device.type != "cpu" or scales.dtype != torch.float32 or scales.shape != expected_shape:
        raise ValueError("MiniMax H3 QKV dequant_scales must be CPU FP32 with shape [50, 3]")
    if not bool(torch.isfinite(scales).all()) or not bool((scales > 0).all()):
        raise ValueError("MiniMax H3 QKV dequant_scales must be finite and positive")
    # The producer multiplies by the FP32 reciprocal before casting to E4M3.
    # A positive subnormal scale can pass the check above but still overflow
    # its reciprocal, so validate the actual arithmetic contract as well.
    if not bool(torch.isfinite(torch.reciprocal(scales)).all()):
        raise ValueError("MiniMax H3 QKV dequant_scales must have finite FP32 reciprocals")

    for name, dtype in _AUDIT_TENSOR_SPECS.items():
        tensor = tensors.get(name)
        if tensor is None:  # guarded by the exact required-name check above
            raise AssertionError("validated final QKV sidecar lost an audit tensor")
        if tensor.device.type != "cpu" or tensor.dtype != dtype or tensor.shape != expected_shape:
            raise ValueError(f"MiniMax H3 QKV audit tensor {name!r} must be CPU {dtype} with shape [50, 3]")
        if dtype.is_floating_point and not bool(torch.isfinite(tensor).all()):
            raise ValueError(f"MiniMax H3 QKV audit tensor {name!r} must be finite")
        if not bool((tensor >= 0).all()):
            raise ValueError(f"MiniMax H3 QKV audit tensor {name!r} must be nonnegative")

    observed_absmax = tensors.get("observed_absmax")
    finite = tensors.get("finite_count")
    nonfinite = tensors.get("nonfinite_count")
    numel = tensors.get("numel")
    observations = tensors.get("observation_count")
    saturation = tensors.get("saturation_count")
    if observations is not None and not bool((observations > 0).all()):
        raise ValueError("MiniMax H3 QKV observation_count must be positive")
    expected_observations = binding.expected_observation_count * binding.sequence_parallel_size
    if observations is not None and not bool((observations == expected_observations).all()):
        raise ValueError(
            "MiniMax H3 QKV observation_count must equal the campaign contract "
            f"({expected_observations} merged observations per layer/component)"
        )
    if numel is not None and not bool((numel > 0).all()):
        raise ValueError("MiniMax H3 QKV numel must be positive")
    if finite is not None and nonfinite is not None and numel is not None:
        if not torch.equal(finite + nonfinite, numel):
            raise ValueError("MiniMax H3 QKV finite_count + nonfinite_count must equal numel")
    if observed_absmax is not None and finite is not None:
        if bool(((finite == 0) & (observed_absmax != 0)).any()):
            raise ValueError("MiniMax H3 QKV observed_absmax must be zero when finite_count is zero")
    if saturation is not None and finite is not None and not bool((saturation <= finite).all()):
        raise ValueError("MiniMax H3 QKV saturation_count cannot exceed finite_count")
    if nonfinite is not None and bool((nonfinite != 0).any()):
        raise ValueError("MiniMax H3 QKV final sidecar cannot contain nonfinite observations")
    if saturation is not None and bool((saturation != 0).any()):
        raise ValueError("MiniMax H3 QKV final sidecar cannot contain saturated E4M3 values")


class MiniMaxH3QKVTransportScales:
    """CPU loader whose scale rows are safe to hand to the 50 DiT blocks."""

    def __init__(
        self,
        path: str | os.PathLike[str],
        *,
        runtime_binding: MiniMaxH3QKVTransportScaleBinding | Mapping[str, Any] | None = None,
    ) -> None:
        self.path = _sidecar_path(path)
        self._runtime_binding: MiniMaxH3QKVTransportScaleBinding | None = None
        self._loaded_binding: MiniMaxH3QKVTransportScaleBinding | None = None
        self._scale_rows: tuple[tuple[float, float, float], ...] | None = None
        self._audit_tensors: dict[str, torch.Tensor] = {}
        if runtime_binding is not None:
            self.bind_runtime_contract(runtime_binding)

    def bind_runtime_contract(
        self,
        binding: MiniMaxH3QKVTransportScaleBinding | Mapping[str, Any],
    ) -> None:
        if self._scale_rows is not None:
            raise RuntimeError("MiniMax H3 QKV runtime contract must be bound before load")
        expected = _coerce_binding(binding)
        validate_minimax_h3_qkv_transport_scale_sidecar_binding(
            self.path,
            expected,
        )
        if self._runtime_binding is not None and self._runtime_binding != expected:
            raise ValueError("MiniMax H3 QKV scales are already bound to another runtime contract")
        self._runtime_binding = expected

    def load(self) -> tuple[tuple[float, float, float], ...]:
        if self._scale_rows is not None:
            return self._scale_rows
        if self._runtime_binding is None:
            raise ValueError("MiniMax H3 QKV scales require bind_runtime_contract() before load")

        before = self.path.stat()
        with safe_open(self.path, framework="pt", device="cpu") as scale_file:
            binding = MiniMaxH3QKVTransportScaleBinding.from_metadata(scale_file.metadata() or {})
            tensors = {name: scale_file.get_tensor(name) for name in scale_file.keys()}
        after = self.path.stat()
        before_identity = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        )
        after_identity = (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        )
        if before_identity != after_identity:
            raise ValueError("MiniMax H3 QKV scale sidecar changed while loading")
        if binding != self._runtime_binding:
            raise ValueError("MiniMax H3 QKV scale binding changed after runtime prevalidation")
        _validate_payload_tensors(tensors, binding=binding)
        actual_digest = digest_minimax_h3_qkv_transport_scale_payload(tensors)
        if actual_digest != binding.payload_sha256:
            raise ValueError(
                "MiniMax H3 QKV scale payload digest mismatch: "
                f"expected={binding.payload_sha256}, actual={actual_digest}"
            )

        scales = tensors[_DEQUANT_SCALES_KEY]
        rows = tuple((float(row[0]), float(row[1]), float(row[2])) for row in scales.tolist())
        if len(rows) != _NUM_LAYERS:
            raise AssertionError("validated QKV scale table lost DiT rows")
        self._scale_rows = rows
        self._audit_tensors = {name: tensor.clone() for name, tensor in tensors.items() if name in _AUDIT_TENSOR_SPECS}
        self._loaded_binding = binding
        return rows

    @property
    def scale_rows(self) -> tuple[tuple[float, float, float], ...]:
        if self._scale_rows is None:
            raise RuntimeError("MiniMax H3 QKV transport scales have not been loaded")
        return self._scale_rows

    @property
    def audit_tensors(self) -> Mapping[str, torch.Tensor]:
        if self._scale_rows is None:
            raise RuntimeError("MiniMax H3 QKV transport scales have not been loaded")
        return self._audit_tensors

    @property
    def binding(self) -> MiniMaxH3QKVTransportScaleBinding:
        if self._loaded_binding is None:
            raise RuntimeError("MiniMax H3 QKV transport scales have not been loaded")
        return self._loaded_binding


def load_minimax_h3_qkv_transport_scale_rows(
    path: str | os.PathLike[str],
    binding: MiniMaxH3QKVTransportScaleBinding | Mapping[str, Any],
) -> tuple[tuple[float, float, float], ...]:
    """Validate and return 50 Python ``(q, k, v)`` dequant-scale rows."""
    return MiniMaxH3QKVTransportScales(
        path,
        runtime_binding=binding,
    ).load()


def prevalidate_minimax_h3_qkv_transport_scale_sidecar(
    path: str | os.PathLike[str],
    *,
    source_model_revision: str,
    runtime_checkpoint_revision: str,
    runtime_checkpoint_manifest_sha256: str,
    campaign_uuid: str,
    run_contract_sha256: str,
    expected_observation_count: int,
    adapter: MiniMaxH3FastH3AdapterIdentity,
    base_schedule: Sequence[float],
    flow_shift: float,
    audio_flow_shift: float,
    kernel_provider: str,
    kernel_abi: str,
    media_profile: MiniMaxH3QKVTransportMediaProfile,
    expected_payload_sha256: str | None = None,
    dit_quant_mode: str = _DIT_QUANT_MODE,
) -> MiniMaxH3QKVTransportScaleBinding:
    """Runner-side full validation, returning the worker runtime binding."""
    actual = read_minimax_h3_qkv_transport_scale_binding(path)
    payload_sha256 = (
        _validate_sha256(
            expected_payload_sha256,
            field="expected_payload_sha256",
        )
        if expected_payload_sha256 is not None
        else actual.payload_sha256
    )
    expected = MiniMaxH3QKVTransportScaleBinding.for_fasth3(
        source_model_revision=source_model_revision,
        runtime_checkpoint_revision=runtime_checkpoint_revision,
        runtime_checkpoint_manifest_sha256=runtime_checkpoint_manifest_sha256,
        campaign_uuid=campaign_uuid,
        run_contract_sha256=run_contract_sha256,
        expected_observation_count=expected_observation_count,
        adapter=adapter,
        base_schedule=base_schedule,
        flow_shift=flow_shift,
        audio_flow_shift=audio_flow_shift,
        kernel_provider=kernel_provider,
        kernel_abi=kernel_abi,
        media_profile=media_profile,
        payload_sha256=payload_sha256,
        dit_quant_mode=dit_quant_mode,
    )
    if actual != expected:
        raise ValueError(
            f"MiniMax H3 QKV scale binding mismatch: expected={expected.to_dict()!r}, actual={actual.to_dict()!r}"
        )
    MiniMaxH3QKVTransportScales(path, runtime_binding=expected).load()
    return expected


__all__ = [
    "MINIMAX_H3_ALL_MAIN_FP8_DIT_QUANT_MODE",
    "MINIMAX_H3_BF16_DIT_QUANT_MODE",
    "MINIMAX_H3_MLP_FP8_DIT_QUANT_MODE",
    "MINIMAX_H3_MLP_OUT_FP8_DIT_QUANT_MODE",
    "MINIMAX_H3_MLP_OUT_QKV_FP8_DIT_QUANT_MODE",
    "MINIMAX_H3_QKV_TRANSPORT_KERNEL_ABI",
    "MINIMAX_H3_QKV_TRANSPORT_KERNEL_PROVIDER",
    "MINIMAX_H3_QKV_RUNTIME_CHECKPOINT_MANIFEST_SCOPE",
    "MiniMaxH3QKVTransportCalibrationProvenance",
    "MiniMaxH3QKVTransportMediaProfile",
    "MiniMaxH3QKVTransportScaleBinding",
    "MiniMaxH3QKVTransportScales",
    "digest_minimax_h3_qkv_transport_scale_payload",
    "canonical_minimax_h3_runtime_checkpoint_manifest",
    "digest_minimax_h3_runtime_checkpoint_manifest",
    "read_minimax_h3_runtime_checkpoint_manifest",
    "load_minimax_h3_qkv_transport_scale_rows",
    "prevalidate_minimax_h3_qkv_transport_scale_sidecar",
    "read_minimax_h3_qkv_transport_scale_binding",
    "validate_minimax_h3_qkv_transport_scale_sidecar_binding",
]
