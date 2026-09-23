# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Offline calibration helpers for MiniMax H3 E4M3 QKV transport.

The transport quantizer sees Q and K *after* BF16 QK normalization and RoPE,
and V directly at the BF16 projection boundary.  This module deliberately
names those two observation sites so an integration cannot silently calibrate
the wrong tensors.

Collection is rank-local and contains no distributed operations.  A worker can
write one snapshot per request (or one accumulated snapshot per rank), and a
coordinator can merge those CPU snapshots after inference has completed.
This keeps calibration off the latency-critical Ulysses collective path.
"""

from __future__ import annotations

import math
import os
import tempfile
import threading
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors.torch import safe_open, save_file

from vllm_omni.diffusion.models.minimax_h3.qkv_transport_scales import (
    MiniMaxH3QKVTransportCalibrationProvenance,
    digest_minimax_h3_qkv_transport_scale_payload,
)

QKV_COMPONENTS = ("q", "k", "v")
QKV_COMPONENT_COUNT = len(QKV_COMPONENTS)
E4M3_FINITE_ABSMAX = 448.0
QKV_E4M3_HEADROOM = 1.125

QKV_TRANSPORT_OBSERVER_MODE_ENV = "VLLM_OMNI_MINIMAX_H3_QKV_TRANSPORT_OBSERVER_MODE"
QKV_TRANSPORT_OBSERVER_SNAPSHOT_DIR_ENV = "VLLM_OMNI_MINIMAX_H3_QKV_TRANSPORT_OBSERVER_SNAPSHOT_DIR"
QKV_TRANSPORT_OBSERVER_SCALES_PATH_ENV = "VLLM_OMNI_MINIMAX_H3_QKV_TRANSPORT_OBSERVER_SCALES_PATH"
QKV_TRANSPORT_OBSERVER_SOURCE_MODEL_REVISION_ENV = "VLLM_OMNI_MINIMAX_H3_QKV_TRANSPORT_OBSERVER_SOURCE_MODEL_REVISION"
QKV_TRANSPORT_OBSERVER_RUNTIME_CHECKPOINT_REVISION_ENV = (
    "VLLM_OMNI_MINIMAX_H3_QKV_TRANSPORT_OBSERVER_RUNTIME_CHECKPOINT_REVISION"
)
QKV_TRANSPORT_OBSERVER_RUNTIME_CHECKPOINT_MANIFEST_SHA256_ENV = (
    "VLLM_OMNI_MINIMAX_H3_QKV_TRANSPORT_OBSERVER_RUNTIME_CHECKPOINT_MANIFEST_SHA256"
)
QKV_TRANSPORT_OBSERVER_CAMPAIGN_UUID_ENV = "VLLM_OMNI_MINIMAX_H3_QKV_TRANSPORT_OBSERVER_CAMPAIGN_UUID"
QKV_TRANSPORT_OBSERVER_RUN_CONTRACT_SHA256_ENV = "VLLM_OMNI_MINIMAX_H3_QKV_TRANSPORT_OBSERVER_RUN_CONTRACT_SHA256"
QKV_TRANSPORT_OBSERVER_EXPECTED_OBSERVATION_COUNT_ENV = (
    "VLLM_OMNI_MINIMAX_H3_QKV_TRANSPORT_OBSERVER_EXPECTED_OBSERVATION_COUNT"
)
QKV_E4M3_TRANSPORT_ENV = "VLLM_OMNI_FLASHINFER_ULYSSES_QKV_E4M3_TRANSPORT"
QKV_TRANSPORT_OBSERVER_ACTIVE_MARKER = "MINIMAX_H3_QKV_TRANSPORT_OBSERVER_ACTIVE"
QKV_TRANSPORT_OBSERVER_SNAPSHOT_MARKER = "MINIMAX_H3_QKV_TRANSPORT_OBSERVER_SNAPSHOT_UPDATED"

_OBSERVER_MODES = frozenset({"calibration", "validation"})
_VALIDATION_SCALE_KEY = "dequant_scales"
_MINIMAX_H3_NUM_LAYERS = 50

_SNAPSHOT_FORMAT = "minimax-h3-qkv-transport-calibration"
_SNAPSHOT_VERSION = "3"
QKV_TRANSPORT_DRAFT_FORMAT = "minimax-h3-qkv-e4m3-transport-scale-draft"
QKV_TRANSPORT_DRAFT_VERSION = "2"
_PROVENANCE_METADATA_KEYS = frozenset(
    {
        "source_model_revision",
        "runtime_checkpoint_revision",
        "runtime_checkpoint_manifest_sha256",
        "campaign_uuid",
        "run_contract_sha256",
        "expected_observation_count",
    }
)
_DRAFT_METADATA_KEYS = frozenset(
    {
        "format",
        "format_version",
        "source_snapshot_count",
        "scale_formula",
        "payload_sha256",
        *_PROVENANCE_METADATA_KEYS,
    }
)
_SNAPSHOT_TENSOR_KEYS = frozenset(
    {
        "absmax",
        "finite_count",
        "nonfinite_count",
        "numel",
        "observation_count",
    }
)
_VALIDATION_TENSOR_KEYS = frozenset(
    {
        "saturation_count",
        "validation_scales",
    }
)


def qkv_transport_calibration_provenance_to_metadata(
    provenance: MiniMaxH3QKVTransportCalibrationProvenance,
) -> dict[str, str]:
    if not isinstance(provenance, MiniMaxH3QKVTransportCalibrationProvenance):
        raise TypeError("provenance must be MiniMaxH3QKVTransportCalibrationProvenance")
    return {key: str(value) for key, value in provenance.to_dict().items()}


def qkv_transport_calibration_provenance_from_metadata(
    metadata: dict[str, str],
) -> MiniMaxH3QKVTransportCalibrationProvenance:
    missing = _PROVENANCE_METADATA_KEYS - set(metadata)
    if missing:
        raise ValueError(f"QKV transport provenance metadata is missing {sorted(missing)}")
    raw_count = metadata["expected_observation_count"]
    if not raw_count.isdecimal():
        raise ValueError("QKV transport expected_observation_count metadata must be a positive integer")
    return MiniMaxH3QKVTransportCalibrationProvenance.from_dict(
        {
            key: (int(metadata[key]) if key == "expected_observation_count" else metadata[key])
            for key in _PROVENANCE_METADATA_KEYS
        }
    )


def qkv_transport_validation_scale_draft_metadata(
    *,
    provenance: MiniMaxH3QKVTransportCalibrationProvenance,
    source_snapshot_count: int,
    payload_sha256: str,
) -> dict[str, str]:
    if isinstance(source_snapshot_count, bool) or not isinstance(source_snapshot_count, int):
        raise TypeError("source_snapshot_count must be an integer")
    if source_snapshot_count <= 0:
        raise ValueError("source_snapshot_count must be positive")
    if not isinstance(payload_sha256, str) or len(payload_sha256) != 64:
        raise ValueError("payload_sha256 must be a 64-character SHA256 digest")
    try:
        int(payload_sha256, 16)
    except ValueError as exc:
        raise ValueError("payload_sha256 must be hexadecimal") from exc
    payload_digest = payload_sha256.lower()
    return {
        "format": QKV_TRANSPORT_DRAFT_FORMAT,
        "format_version": QKV_TRANSPORT_DRAFT_VERSION,
        "source_snapshot_count": str(source_snapshot_count),
        "scale_formula": "nextafter(absmax*1.125/448,+inf)-fp32",
        "payload_sha256": payload_digest,
        **qkv_transport_calibration_provenance_to_metadata(provenance),
    }


def _cpu_float32_matrix(
    value: torch.Tensor | Sequence[Sequence[float]],
    *,
    name: str,
    num_layers: int | None = None,
    positive: bool = False,
) -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=torch.float32, device="cpu")
    if tensor.ndim != 2 or tensor.shape[1] != QKV_COMPONENT_COUNT:
        raise ValueError(f"{name} must have shape [num_layers, 3], got {tuple(tensor.shape)}")
    if num_layers is not None and tensor.shape[0] != num_layers:
        raise ValueError(f"{name} has {tensor.shape[0]} layers, expected {num_layers}")
    if tensor.shape[0] <= 0:
        raise ValueError(f"{name} must contain at least one layer")
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"{name} must contain only finite values")
    if positive and not bool((tensor > 0).all()):
        raise ValueError(f"{name} must contain only positive values")
    if positive and not bool(torch.isfinite(torch.reciprocal(tensor)).all()):
        raise ValueError(f"{name} must have finite FP32 reciprocals")
    return tensor.detach().contiguous().clone()


def _cpu_int64_matrix(
    value: torch.Tensor,
    *,
    name: str,
    num_layers: int,
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.dtype != torch.int64:
        raise TypeError(f"{name} must have dtype torch.int64, got {value.dtype}")
    if value.shape != (num_layers, QKV_COMPONENT_COUNT):
        raise ValueError(f"{name} must have shape [{num_layers}, 3], got {tuple(value.shape)}")
    tensor = value.detach().to(device="cpu").contiguous().clone()
    if bool((tensor < 0).any()):
        raise ValueError(f"{name} must contain only non-negative values")
    return tensor


@dataclass(frozen=True)
class MiniMaxH3QKVTransportCalibrationSnapshot:
    """Rank-local or merged QKV statistics.

    All tensors are normalized to contiguous CPU storage.  Counts describe
    scalar elements, not tokens.  ``saturation_count`` is present only for a
    validation run constructed with fixed ``validation_scales``.
    """

    absmax: torch.Tensor
    finite_count: torch.Tensor
    nonfinite_count: torch.Tensor
    numel: torch.Tensor
    observation_count: torch.Tensor
    saturation_count: torch.Tensor | None = None
    validation_scales: torch.Tensor | None = None
    source_rank: int | None = None
    observer_mode: str | None = None
    provenance: MiniMaxH3QKVTransportCalibrationProvenance | None = None

    def __post_init__(self) -> None:
        absmax = _cpu_float32_matrix(self.absmax, name="absmax")
        num_layers = absmax.shape[0]
        if bool((absmax < 0).any()):
            raise ValueError("absmax must contain only non-negative values")

        finite_count = _cpu_int64_matrix(
            self.finite_count,
            name="finite_count",
            num_layers=num_layers,
        )
        nonfinite_count = _cpu_int64_matrix(
            self.nonfinite_count,
            name="nonfinite_count",
            num_layers=num_layers,
        )
        numel = _cpu_int64_matrix(
            self.numel,
            name="numel",
            num_layers=num_layers,
        )
        observation_count = _cpu_int64_matrix(
            self.observation_count,
            name="observation_count",
            num_layers=num_layers,
        )
        if not torch.equal(finite_count + nonfinite_count, numel):
            raise ValueError("finite_count + nonfinite_count must equal numel")
        if bool(((finite_count == 0) & (absmax != 0)).any()):
            raise ValueError("absmax must be zero when no finite values were observed")
        if bool(((observation_count == 0) != (numel == 0)).any()):
            raise ValueError("observation_count and numel must either both be zero or both be nonzero")

        if (self.saturation_count is None) != (self.validation_scales is None):
            raise ValueError("saturation_count and validation_scales must be provided together")
        saturation_count = None
        validation_scales = None
        if self.saturation_count is not None:
            saturation_count = _cpu_int64_matrix(
                self.saturation_count,
                name="saturation_count",
                num_layers=num_layers,
            )
            if bool((saturation_count > finite_count).any()):
                raise ValueError("saturation_count cannot exceed finite_count")
            assert self.validation_scales is not None
            validation_scales = _cpu_float32_matrix(
                self.validation_scales,
                name="validation_scales",
                num_layers=num_layers,
                positive=True,
            )

        if (self.source_rank is None) != (self.observer_mode is None):
            raise ValueError("source_rank and observer_mode must either both be set or both be absent")
        if self.source_rank is not None:
            if isinstance(self.source_rank, bool) or not isinstance(self.source_rank, int) or self.source_rank < 0:
                raise ValueError("source_rank must be a non-negative integer")
            if self.observer_mode not in _OBSERVER_MODES:
                raise ValueError(f"observer_mode must be one of {sorted(_OBSERVER_MODES)}")
            if (self.observer_mode == "validation") != (validation_scales is not None):
                raise ValueError("observer_mode must match whether validation scales are present")
            if not isinstance(self.provenance, MiniMaxH3QKVTransportCalibrationProvenance):
                raise ValueError("rank-bound observer snapshots require calibration provenance")
        elif self.provenance is not None and not isinstance(
            self.provenance, MiniMaxH3QKVTransportCalibrationProvenance
        ):
            raise TypeError("provenance must be MiniMaxH3QKVTransportCalibrationProvenance or None")

        for name, value in (
            ("absmax", absmax),
            ("finite_count", finite_count),
            ("nonfinite_count", nonfinite_count),
            ("numel", numel),
            ("observation_count", observation_count),
            ("saturation_count", saturation_count),
            ("validation_scales", validation_scales),
        ):
            object.__setattr__(self, name, value)

    @property
    def num_layers(self) -> int:
        return int(self.absmax.shape[0])

    @property
    def is_validation(self) -> bool:
        return self.validation_scales is not None


@dataclass(frozen=True)
class MiniMaxH3QKVTransportScaleAudit:
    """Exact validation statistics for one fixed E4M3 scale matrix."""

    scales: torch.Tensor
    representable_absmax: torch.Tensor
    observed_absmax: torch.Tensor
    peak_utilization: torch.Tensor
    finite_count: torch.Tensor
    nonfinite_count: torch.Tensor
    numel: torch.Tensor
    observation_count: torch.Tensor
    saturation_count: torch.Tensor
    saturation_fraction: torch.Tensor
    coverage_complete: bool
    passed: bool


@dataclass(frozen=True)
class _LocalTensorStats:
    absmax: float
    finite_count: int
    nonfinite_count: int
    numel: int
    saturation_count: int | None


class MiniMaxH3QKVTransportCalibrationCollector:
    """Collect rank-local QKV statistics without any distributed collective.

    Args:
        num_layers: Number of MiniMax H3 DiT blocks.
        validation_scales: Optional fixed CPU FP32 ``[num_layers, 3]`` scale
            matrix.  When present, the collector additionally counts finite
            values whose magnitude exceeds the E4M3 representable range.

    This object is inference-only: observed tensors must be BF16 and must not
    require gradients.  The object never modifies an observed tensor.
    """

    def __init__(
        self,
        num_layers: int,
        *,
        validation_scales: torch.Tensor | Sequence[Sequence[float]] | None = None,
    ) -> None:
        if isinstance(num_layers, bool) or not isinstance(num_layers, int):
            raise TypeError("num_layers must be an integer")
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")
        self._num_layers = num_layers
        self._validation_scales = (
            None
            if validation_scales is None
            else _cpu_float32_matrix(
                validation_scales,
                name="validation_scales",
                num_layers=num_layers,
                positive=True,
            )
        )
        shape = (num_layers, QKV_COMPONENT_COUNT)
        self._absmax = torch.zeros(shape, dtype=torch.float32)
        self._finite_count = torch.zeros(shape, dtype=torch.int64)
        self._nonfinite_count = torch.zeros(shape, dtype=torch.int64)
        self._numel = torch.zeros(shape, dtype=torch.int64)
        self._observation_count = torch.zeros(shape, dtype=torch.int64)
        self._saturation_count = torch.zeros(shape, dtype=torch.int64) if self._validation_scales is not None else None
        self._lock = threading.Lock()

    @property
    def num_layers(self) -> int:
        return self._num_layers

    def _validate_layer(self, layer_index: int) -> int:
        if isinstance(layer_index, bool) or not isinstance(layer_index, int):
            raise TypeError("layer_index must be an integer")
        if not 0 <= layer_index < self._num_layers:
            raise IndexError(f"layer_index {layer_index} is outside [0, {self._num_layers})")
        return layer_index

    def _observe_tensor(
        self,
        tensor: torch.Tensor,
        *,
        layer_index: int,
        component_index: int,
        boundary: str,
    ) -> _LocalTensorStats:
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{boundary} {QKV_COMPONENTS[component_index]} must be a tensor")
        if tensor.dtype != torch.bfloat16:
            raise TypeError(f"{boundary} {QKV_COMPONENTS[component_index]} must be BF16, got {tensor.dtype}")
        if tensor.requires_grad:
            raise ValueError(
                "MiniMax H3 QKV transport calibration is inference-only; "
                f"{boundary} {QKV_COMPONENTS[component_index]} requires gradients"
            )
        if tensor.numel() == 0:
            raise ValueError(f"{boundary} {QKV_COMPONENTS[component_index]} must be non-empty")

        with torch.no_grad():
            fp32 = tensor.detach().to(dtype=torch.float32)
            finite = torch.isfinite(fp32)
            finite_count = int(finite.sum(dtype=torch.int64).item())
            numel = tensor.numel()
            nonfinite_count = numel - finite_count
            absmax = float(torch.where(finite, fp32.abs(), torch.zeros_like(fp32)).amax().item())

            saturation_count = None
            if self._validation_scales is not None:
                scale = float(self._validation_scales[layer_index, component_index])
                saturation_count = int(
                    (finite & (fp32.abs() > scale * E4M3_FINITE_ABSMAX)).sum(dtype=torch.int64).item()
                )

        return _LocalTensorStats(
            absmax=absmax,
            finite_count=finite_count,
            nonfinite_count=nonfinite_count,
            numel=numel,
            saturation_count=saturation_count,
        )

    def _commit(
        self,
        layer_index: int,
        component_stats: Sequence[tuple[int, _LocalTensorStats]],
    ) -> None:
        with self._lock:
            for component_index, stats in component_stats:
                self._absmax[layer_index, component_index] = max(
                    float(self._absmax[layer_index, component_index]),
                    stats.absmax,
                )
                self._finite_count[layer_index, component_index] += stats.finite_count
                self._nonfinite_count[layer_index, component_index] += stats.nonfinite_count
                self._numel[layer_index, component_index] += stats.numel
                self._observation_count[layer_index, component_index] += 1
                if self._saturation_count is not None:
                    assert stats.saturation_count is not None
                    self._saturation_count[layer_index, component_index] += stats.saturation_count

    def record_post_qknorm_rope(
        self,
        layer_index: int,
        q: torch.Tensor,
        k: torch.Tensor,
    ) -> None:
        """Record Q/K at the post-QKNorm+RoPE BF16 transport boundary."""

        layer_index = self._validate_layer(layer_index)
        if q.shape != k.shape:
            raise ValueError(f"post-QKNorm+RoPE Q/K shapes differ: {tuple(q.shape)} vs {tuple(k.shape)}")
        if q.device != k.device:
            raise ValueError(f"post-QKNorm+RoPE Q/K devices differ: {q.device} vs {k.device}")
        q_stats = self._observe_tensor(
            q,
            layer_index=layer_index,
            component_index=0,
            boundary="post-QKNorm+RoPE",
        )
        k_stats = self._observe_tensor(
            k,
            layer_index=layer_index,
            component_index=1,
            boundary="post-QKNorm+RoPE",
        )
        self._commit(layer_index, ((0, q_stats), (1, k_stats)))

    def record_projected_v(
        self,
        layer_index: int,
        v: torch.Tensor,
    ) -> None:
        """Record V at the BF16 QKV-projection output boundary."""

        layer_index = self._validate_layer(layer_index)
        v_stats = self._observe_tensor(
            v,
            layer_index=layer_index,
            component_index=2,
            boundary="projected-V",
        )
        self._commit(layer_index, ((2, v_stats),))

    def record_qkv(
        self,
        layer_index: int,
        q_post_qknorm_rope: torch.Tensor,
        k_post_qknorm_rope: torch.Tensor,
        v_projection_bf16: torch.Tensor,
    ) -> None:
        """Convenience wrapper for both explicitly named observation sites."""

        self.record_post_qknorm_rope(
            layer_index,
            q_post_qknorm_rope,
            k_post_qknorm_rope,
        )
        self.record_projected_v(layer_index, v_projection_bf16)

    def snapshot(self) -> MiniMaxH3QKVTransportCalibrationSnapshot:
        """Return an immutable CPU copy of the current rank-local statistics."""

        with self._lock:
            return MiniMaxH3QKVTransportCalibrationSnapshot(
                absmax=self._absmax.clone(),
                finite_count=self._finite_count.clone(),
                nonfinite_count=self._nonfinite_count.clone(),
                numel=self._numel.clone(),
                observation_count=self._observation_count.clone(),
                saturation_count=(None if self._saturation_count is None else self._saturation_count.clone()),
                validation_scales=(None if self._validation_scales is None else self._validation_scales.clone()),
            )


class MiniMaxH3QKVTransportCalibrationObserver:
    """Rank-local, eager-only observer for the exact QKV transport boundary.

    The observer owns a transactional collector for each forward and merges it
    into a cumulative CPU snapshot only after block 49 returns.  A completed
    forward must have contributed exactly one Q/K/V observation for every
    layer.  This catches Cache-DiT/skipped-layer configurations and prevents a
    failed partial forward from contaminating a later published snapshot.
    """

    def __init__(
        self,
        *,
        mode: str,
        rank: int,
        snapshot_path: str | Path,
        num_layers: int,
        provenance: MiniMaxH3QKVTransportCalibrationProvenance,
        validation_scales: torch.Tensor | None = None,
    ) -> None:
        if mode not in _OBSERVER_MODES:
            raise ValueError(f"QKV transport observer mode must be one of {sorted(_OBSERVER_MODES)}, got {mode!r}")
        if isinstance(rank, bool) or not isinstance(rank, int) or rank < 0:
            raise ValueError("QKV transport observer rank must be a non-negative integer")
        if mode == "calibration" and validation_scales is not None:
            raise ValueError("calibration mode must not receive validation scales")
        if mode == "validation" and validation_scales is None:
            raise ValueError("validation mode requires fixed dequant_scales")
        if not isinstance(provenance, MiniMaxH3QKVTransportCalibrationProvenance):
            raise TypeError("observer provenance must be MiniMaxH3QKVTransportCalibrationProvenance")

        self.mode = mode
        self.rank = rank
        self.snapshot_path = Path(snapshot_path)
        self._num_layers = num_layers
        self.provenance = provenance
        self._validation_scales = (
            None
            if validation_scales is None
            else _cpu_float32_matrix(
                validation_scales,
                name="validation_scales",
                num_layers=num_layers,
                positive=True,
            )
        )
        self._cumulative_snapshot: MiniMaxH3QKVTransportCalibrationSnapshot | None = None
        self._active_collector: MiniMaxH3QKVTransportCalibrationCollector | None = None
        self._active_token: int | None = None
        self._next_token = 0
        self._state_lock = threading.Lock()

    @property
    def num_layers(self) -> int:
        return self._num_layers

    @property
    def is_validation(self) -> bool:
        return self._validation_scales is not None

    def begin_forward(self) -> int:
        """Start a transactional per-forward collection and return its token."""

        with self._state_lock:
            if self._active_collector is not None:
                raise RuntimeError("MiniMax H3 QKV transport observer does not support overlapping model forwards")
            token = self._next_token
            self._next_token += 1
            self._active_token = token
            self._active_collector = MiniMaxH3QKVTransportCalibrationCollector(
                self.num_layers,
                validation_scales=self._validation_scales,
            )
            return token

    def record_qkv(
        self,
        layer_index: int,
        q_post_qknorm_rope: torch.Tensor,
        k_post_qknorm_rope: torch.Tensor,
        v_projection_bf16: torch.Tensor,
    ) -> None:
        """Observe one local layer without invoking any distributed API."""

        if torch.compiler.is_compiling():
            raise RuntimeError(
                "MiniMax H3 QKV transport observer does not support torch.compile; run with enforce_eager=True"
            )
        tensors = (
            q_post_qknorm_rope,
            k_post_qknorm_rope,
            v_projection_bf16,
        )
        if any(tensor.device.type == "cuda" and torch.cuda.is_current_stream_capturing() for tensor in tensors):
            raise RuntimeError("MiniMax H3 QKV transport observer does not support CUDA graph capture")
        with self._state_lock:
            if self._active_collector is None:
                raise RuntimeError("MiniMax H3 QKV transport observer record_qkv requires begin_forward()")
            self._active_collector.record_qkv(layer_index, *tensors)

    def active_snapshot(self) -> MiniMaxH3QKVTransportCalibrationSnapshot:
        """Return a CPU copy of the in-flight transaction for diagnostics."""

        with self._state_lock:
            if self._active_collector is None:
                raise RuntimeError("QKV transport observer has no active forward")
            return self._active_collector.snapshot()

    def abort_forward(self, token: int) -> None:
        """Discard observations from a failed block-stack execution."""

        with self._state_lock:
            if self._active_collector is None or token != self._active_token:
                raise RuntimeError("QKV transport observer abort token is stale")
            self._active_collector = None
            self._active_token = None

    def complete_forward(
        self,
        token: int,
    ) -> MiniMaxH3QKVTransportCalibrationSnapshot:
        """Verify all layers ran once, then atomically replace this rank's file."""

        with self._state_lock:
            if self._active_collector is None or token != self._active_token:
                raise RuntimeError("QKV transport observer completion token is stale")
            try:
                forward_snapshot = self._active_collector.snapshot()
                expected = torch.ones_like(forward_snapshot.observation_count)
                if not torch.equal(forward_snapshot.observation_count, expected):
                    bad = torch.nonzero(
                        forward_snapshot.observation_count != expected,
                        as_tuple=False,
                    ).tolist()
                    raise RuntimeError(
                        "MiniMax H3 QKV transport observer requires exactly one "
                        "Q/K/V observation from every block in a completed forward; "
                        f"bad layer/component indices={bad[:12]}"
                    )
                cumulative = (
                    forward_snapshot
                    if self._cumulative_snapshot is None
                    else merge_qkv_transport_calibration_snapshots((self._cumulative_snapshot, forward_snapshot))
                )
                if bool((cumulative.observation_count > self.provenance.expected_observation_count).any()):
                    raise RuntimeError(
                        "MiniMax H3 QKV transport observer exceeded the exact campaign observation count"
                    )
                published = _with_qkv_transport_snapshot_identity(
                    cumulative,
                    source_rank=self.rank,
                    observer_mode=self.mode,
                    provenance=self.provenance,
                )
                save_qkv_transport_calibration_snapshot_atomic(
                    published,
                    self.snapshot_path,
                )
                self._cumulative_snapshot = cumulative
                return published
            finally:
                self._active_collector = None
                self._active_token = None


def _with_qkv_transport_snapshot_identity(
    snapshot: MiniMaxH3QKVTransportCalibrationSnapshot,
    *,
    source_rank: int,
    observer_mode: str,
    provenance: MiniMaxH3QKVTransportCalibrationProvenance,
) -> MiniMaxH3QKVTransportCalibrationSnapshot:
    return MiniMaxH3QKVTransportCalibrationSnapshot(
        absmax=snapshot.absmax,
        finite_count=snapshot.finite_count,
        nonfinite_count=snapshot.nonfinite_count,
        numel=snapshot.numel,
        observation_count=snapshot.observation_count,
        saturation_count=snapshot.saturation_count,
        validation_scales=snapshot.validation_scales,
        source_rank=source_rank,
        observer_mode=observer_mode,
        provenance=provenance,
    )


def merge_qkv_transport_calibration_snapshots(
    snapshots: Iterable[MiniMaxH3QKVTransportCalibrationSnapshot],
    *,
    expected_source_ranks: Iterable[int] | None = None,
    expected_observer_mode: str | None = None,
) -> MiniMaxH3QKVTransportCalibrationSnapshot:
    """Merge snapshots and optionally prove an exact worker-rank set."""

    snapshots = tuple(snapshots)
    if not snapshots:
        raise ValueError("at least one QKV transport calibration snapshot is required")
    if any(not isinstance(snapshot, MiniMaxH3QKVTransportCalibrationSnapshot) for snapshot in snapshots):
        raise TypeError("all inputs must be MiniMaxH3QKVTransportCalibrationSnapshot instances")

    bound = [snapshot.source_rank is not None for snapshot in snapshots]
    if any(bound) and not all(bound):
        raise ValueError("cannot merge bound and unbound QKV calibration snapshots")
    merged_source_rank = None
    merged_observer_mode = None
    provenance_bound = [snapshot.provenance is not None for snapshot in snapshots]
    if any(provenance_bound) and not all(provenance_bound):
        raise ValueError("cannot merge QKV calibration snapshots with different provenance presence")
    merged_provenance = snapshots[0].provenance if all(provenance_bound) else None
    if merged_provenance is not None and any(snapshot.provenance != merged_provenance for snapshot in snapshots[1:]):
        raise ValueError("QKV calibration snapshot provenance must match bit-exactly")
    if all(bound):
        ranks = [snapshot.source_rank for snapshot in snapshots]
        assert all(rank is not None for rank in ranks)
        concrete_ranks = [int(rank) for rank in ranks]
        if len(set(concrete_ranks)) != len(concrete_ranks):
            raise ValueError("QKV calibration snapshot source ranks must be unique")
        modes = {snapshot.observer_mode for snapshot in snapshots}
        if len(modes) != 1:
            raise ValueError("QKV calibration snapshot observer modes must match")
        only_mode = next(iter(modes))
        assert only_mode is not None
        if len(snapshots) == 1:
            merged_source_rank = concrete_ranks[0]
            merged_observer_mode = only_mode
    else:
        concrete_ranks = []
        only_mode = None

    if expected_source_ranks is not None:
        expected_rank_tuple = tuple(expected_source_ranks)
        if not expected_rank_tuple:
            raise ValueError("expected_source_ranks must not be empty")
        if any(isinstance(rank, bool) or not isinstance(rank, int) or rank < 0 for rank in expected_rank_tuple):
            raise ValueError("expected_source_ranks must contain non-negative integers")
        if len(set(expected_rank_tuple)) != len(expected_rank_tuple):
            raise ValueError("expected_source_ranks must not contain duplicates")
        if not all(bound):
            raise ValueError("expected_source_ranks requires rank-bound observer snapshots")
        if set(concrete_ranks) != set(expected_rank_tuple):
            raise ValueError(
                "QKV calibration snapshot rank set mismatch: "
                f"expected={sorted(expected_rank_tuple)}, actual={sorted(concrete_ranks)}"
            )
        if merged_provenance is None:
            raise ValueError("rank-complete QKV calibration merge requires provenance")
        expected_count = torch.full_like(
            snapshots[0].observation_count,
            merged_provenance.expected_observation_count,
        )
        for snapshot in snapshots:
            if not torch.equal(snapshot.observation_count, expected_count):
                raise ValueError(
                    "QKV calibration snapshot observation_count does not exactly match the campaign contract"
                )

    if expected_observer_mode is not None:
        if expected_observer_mode not in _OBSERVER_MODES:
            raise ValueError(f"expected_observer_mode must be one of {sorted(_OBSERVER_MODES)}")
        if only_mode is None:
            raise ValueError("expected_observer_mode requires rank-bound observer snapshots")
        if only_mode != expected_observer_mode:
            raise ValueError(
                "QKV calibration snapshot observer mode mismatch: "
                f"expected={expected_observer_mode}, actual={only_mode}"
            )

    first = snapshots[0]
    for snapshot in snapshots[1:]:
        if snapshot.num_layers != first.num_layers:
            raise ValueError("cannot merge QKV calibration snapshots with different layer counts")
        if snapshot.is_validation != first.is_validation:
            raise ValueError("cannot merge calibration and scale-validation snapshots")
        if first.validation_scales is not None and not torch.equal(
            snapshot.validation_scales,
            first.validation_scales,
        ):
            raise ValueError("cannot merge QKV validation snapshots collected with different scales")

    absmax = torch.stack([snapshot.absmax for snapshot in snapshots]).amax(dim=0)

    def sum_field(name: str) -> torch.Tensor:
        return torch.stack([getattr(snapshot, name) for snapshot in snapshots]).sum(
            dim=0,
            dtype=torch.int64,
        )

    return MiniMaxH3QKVTransportCalibrationSnapshot(
        absmax=absmax,
        finite_count=sum_field("finite_count"),
        nonfinite_count=sum_field("nonfinite_count"),
        numel=sum_field("numel"),
        observation_count=sum_field("observation_count"),
        saturation_count=(sum_field("saturation_count") if first.saturation_count is not None else None),
        validation_scales=first.validation_scales,
        source_rank=merged_source_rank,
        observer_mode=merged_observer_mode,
        provenance=merged_provenance,
    )


def derive_qkv_e4m3_scales(
    snapshot: MiniMaxH3QKVTransportCalibrationSnapshot,
) -> torch.Tensor:
    """Derive CPU FP32 E4M3 dequant scales with 12.5% magnitude headroom.

    The operation order is intentionally fixed to the serving contract:
    ``nextafter(absmax * 1.125 / 448, +inf)`` in CPU FP32.
    """

    if not isinstance(snapshot, MiniMaxH3QKVTransportCalibrationSnapshot):
        raise TypeError("snapshot must be a MiniMaxH3QKVTransportCalibrationSnapshot")
    if bool((snapshot.observation_count == 0).any()):
        raise ValueError("every layer and Q/K/V component must be observed")
    if bool((snapshot.nonfinite_count != 0).any()):
        raise ValueError("cannot derive E4M3 scales from non-finite observations")
    if not torch.equal(snapshot.finite_count, snapshot.numel):
        raise ValueError("all calibration observations must be finite")
    if bool((snapshot.absmax <= 0).any()):
        raise ValueError("every calibrated Q/K/V absmax must be positive before deriving E4M3 scales")

    scaled = snapshot.absmax.to(device="cpu", dtype=torch.float32).clone()
    scaled.mul_(QKV_E4M3_HEADROOM).div_(E4M3_FINITE_ABSMAX)
    scales = torch.nextafter(scaled, torch.full_like(scaled, math.inf))
    if not bool(torch.isfinite(scales).all()) or not bool((scales > 0).all()):
        raise ValueError("derived E4M3 scales are not finite and positive")
    quant_scales = (1.0 / scales).float()
    if not bool(torch.isfinite(quant_scales).all()):
        raise ValueError("derived E4M3 quant scales (1 / dequant_scales) are not finite")
    return scales.contiguous()


def audit_qkv_e4m3_scales(
    validation_snapshot: MiniMaxH3QKVTransportCalibrationSnapshot,
) -> MiniMaxH3QKVTransportScaleAudit:
    """Build utilization and exact saturation statistics for a validation run."""

    if not isinstance(
        validation_snapshot,
        MiniMaxH3QKVTransportCalibrationSnapshot,
    ):
        raise TypeError("validation_snapshot must be a MiniMaxH3QKVTransportCalibrationSnapshot")
    if validation_snapshot.validation_scales is None or validation_snapshot.saturation_count is None:
        raise ValueError("validation snapshot must be collected with fixed validation_scales")

    scales = validation_snapshot.validation_scales.clone()
    representable_absmax = scales * E4M3_FINITE_ABSMAX
    peak_utilization = validation_snapshot.absmax / representable_absmax
    finite_denominator = validation_snapshot.finite_count.clamp_min(1).to(torch.float64)
    saturation_fraction = validation_snapshot.saturation_count.to(torch.float64) / finite_denominator
    coverage_complete = bool((validation_snapshot.observation_count > 0).all())
    passed = (
        coverage_complete
        and not bool((validation_snapshot.nonfinite_count != 0).any())
        and not bool((validation_snapshot.saturation_count != 0).any())
    )
    return MiniMaxH3QKVTransportScaleAudit(
        scales=scales,
        representable_absmax=representable_absmax,
        observed_absmax=validation_snapshot.absmax.clone(),
        peak_utilization=peak_utilization,
        finite_count=validation_snapshot.finite_count.clone(),
        nonfinite_count=validation_snapshot.nonfinite_count.clone(),
        numel=validation_snapshot.numel.clone(),
        observation_count=validation_snapshot.observation_count.clone(),
        saturation_count=validation_snapshot.saturation_count.clone(),
        saturation_fraction=saturation_fraction,
        coverage_complete=coverage_complete,
        passed=passed,
    )


def save_qkv_transport_calibration_snapshot(
    snapshot: MiniMaxH3QKVTransportCalibrationSnapshot,
    path: str | Path,
    *,
    source_rank: int | None = None,
    observer_mode: str | None = None,
    provenance: MiniMaxH3QKVTransportCalibrationProvenance | None = None,
) -> None:
    """Save one rank/request snapshot as a self-describing safetensors file."""

    if not isinstance(snapshot, MiniMaxH3QKVTransportCalibrationSnapshot):
        raise TypeError("snapshot must be a MiniMaxH3QKVTransportCalibrationSnapshot")
    overrides = (source_rank, observer_mode, provenance)
    if any(value is not None for value in overrides) and not all(value is not None for value in overrides):
        raise ValueError("source_rank, observer_mode, and provenance overrides must be provided together")
    if source_rank is not None:
        assert observer_mode is not None
        if snapshot.source_rank is not None and (
            snapshot.source_rank != source_rank or snapshot.observer_mode != observer_mode
        ):
            raise ValueError("QKV transport snapshot identity override conflicts with snapshot")
        snapshot = _with_qkv_transport_snapshot_identity(
            snapshot,
            source_rank=source_rank,
            observer_mode=observer_mode,
            provenance=provenance,
        )
    if snapshot.source_rank is None or snapshot.observer_mode is None or snapshot.provenance is None:
        raise ValueError("persisted QKV transport snapshots must include source_rank, observer_mode, and provenance")
    tensors = {
        "absmax": snapshot.absmax,
        "finite_count": snapshot.finite_count,
        "nonfinite_count": snapshot.nonfinite_count,
        "numel": snapshot.numel,
        "observation_count": snapshot.observation_count,
    }
    if snapshot.is_validation:
        assert snapshot.saturation_count is not None
        assert snapshot.validation_scales is not None
        tensors["saturation_count"] = snapshot.saturation_count
        tensors["validation_scales"] = snapshot.validation_scales
    save_file(
        tensors,
        str(path),
        metadata={
            "format": _SNAPSHOT_FORMAT,
            "format_version": _SNAPSHOT_VERSION,
            "components": ",".join(QKV_COMPONENTS),
            "qk_boundary": "post_qknorm_rope_bf16",
            "v_boundary": "projection_bf16",
            "validation": "1" if snapshot.is_validation else "0",
            "source_rank": str(snapshot.source_rank),
            "observer_mode": snapshot.observer_mode,
            **qkv_transport_calibration_provenance_to_metadata(snapshot.provenance),
        },
    )


def save_qkv_transport_calibration_snapshot_atomic(
    snapshot: MiniMaxH3QKVTransportCalibrationSnapshot,
    path: str | Path,
) -> None:
    """Atomically replace one rank-local cumulative snapshot.

    The temporary file is created in the destination directory so
    :func:`os.replace` cannot cross filesystems.  No other rank shares the
    destination name generated by the observer factory.
    """

    destination = Path(path)
    parent = destination.parent
    if not parent.is_dir():
        raise ValueError(f"QKV transport observer snapshot directory does not exist: {parent}")
    fd, temporary_name = tempfile.mkstemp(
        dir=parent,
        prefix=f".{destination.name}.",
        suffix=".safetensors",
    )
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        save_qkv_transport_calibration_snapshot(snapshot, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def load_qkv_transport_calibration_snapshot(
    path: str | Path,
) -> MiniMaxH3QKVTransportCalibrationSnapshot:
    """Load and fail closed on malformed or incompatible snapshot files."""

    with safe_open(str(path), framework="pt", device="cpu") as handle:
        metadata = handle.metadata() or {}
        expected_metadata_keys = {
            "format",
            "format_version",
            "components",
            "qk_boundary",
            "v_boundary",
            "validation",
            "source_rank",
            "observer_mode",
            *_PROVENANCE_METADATA_KEYS,
        }
        if set(metadata) != expected_metadata_keys:
            raise ValueError(
                "QKV calibration snapshot metadata set is invalid: "
                f"missing={sorted(expected_metadata_keys - set(metadata))}, "
                f"unknown={sorted(set(metadata) - expected_metadata_keys)}"
            )
        if metadata.get("format") != _SNAPSHOT_FORMAT:
            raise ValueError("not a MiniMax H3 QKV transport calibration snapshot")
        if metadata.get("format_version") != _SNAPSHOT_VERSION:
            raise ValueError("unsupported MiniMax H3 QKV transport calibration snapshot version")
        if metadata.get("components") != ",".join(QKV_COMPONENTS):
            raise ValueError("QKV calibration snapshot component order is invalid")
        if metadata.get("qk_boundary") != "post_qknorm_rope_bf16":
            raise ValueError("QKV calibration snapshot Q/K boundary is invalid")
        if metadata.get("v_boundary") != "projection_bf16":
            raise ValueError("QKV calibration snapshot V boundary is invalid")
        validation_marker = metadata.get("validation")
        if validation_marker not in {"0", "1"}:
            raise ValueError("QKV calibration snapshot validation marker is invalid")
        source_rank_raw = metadata.get("source_rank")
        if source_rank_raw is None or not source_rank_raw.isdecimal():
            raise ValueError("QKV calibration snapshot source_rank is invalid")
        source_rank = int(source_rank_raw)
        observer_mode = metadata.get("observer_mode")
        if observer_mode not in _OBSERVER_MODES:
            raise ValueError("QKV calibration snapshot observer_mode is invalid")
        if (observer_mode == "validation") != (validation_marker == "1"):
            raise ValueError("QKV calibration snapshot observer_mode/validation markers disagree")
        provenance = qkv_transport_calibration_provenance_from_metadata(metadata)
        expected_keys = set(_SNAPSHOT_TENSOR_KEYS)
        if validation_marker == "1":
            expected_keys.update(_VALIDATION_TENSOR_KEYS)
        actual_keys = set(handle.keys())
        if actual_keys != expected_keys:
            raise ValueError(
                "QKV calibration snapshot tensor set is invalid: "
                f"missing={sorted(expected_keys - actual_keys)}, "
                f"unknown={sorted(actual_keys - expected_keys)}"
            )
        tensors = {name: handle.get_tensor(name) for name in expected_keys}

    return MiniMaxH3QKVTransportCalibrationSnapshot(
        absmax=tensors["absmax"],
        finite_count=tensors["finite_count"],
        nonfinite_count=tensors["nonfinite_count"],
        numel=tensors["numel"],
        observation_count=tensors["observation_count"],
        saturation_count=tensors.get("saturation_count"),
        validation_scales=tensors.get("validation_scales"),
        source_rank=source_rank,
        observer_mode=observer_mode,
        provenance=provenance,
    )


def merge_qkv_transport_calibration_files(
    paths: Iterable[str | Path],
    *,
    expected_source_ranks: Iterable[int] | None = None,
    expected_observer_mode: str | None = None,
) -> MiniMaxH3QKVTransportCalibrationSnapshot:
    """Load and merge rank/request snapshot files entirely on CPU."""

    return merge_qkv_transport_calibration_snapshots(
        (load_qkv_transport_calibration_snapshot(path) for path in paths),
        expected_source_ranks=expected_source_ranks,
        expected_observer_mode=expected_observer_mode,
    )


def load_qkv_transport_validation_scales(
    path: str | Path,
    *,
    num_layers: int = _MINIMAX_H3_NUM_LAYERS,
    expected_provenance: MiniMaxH3QKVTransportCalibrationProvenance | None = None,
) -> torch.Tensor:
    """Load the strict draft scale artifact consumed by validation mode.

    This is intentionally not the final identity-bound serving sidecar.  The
    draft artifact has exactly one tensor, ``dequant_scales``; accepting audit
    tensors or unrelated payloads here would make the two-stage builder
    contract ambiguous.
    """

    candidate = Path(path)
    if not candidate.is_absolute():
        raise ValueError("QKV validation scale path must be absolute")
    if candidate.suffix != ".safetensors" or not candidate.is_file():
        raise ValueError(f"QKV validation scale path must name an existing .safetensors file: {candidate}")
    with safe_open(candidate, framework="pt", device="cpu") as handle:
        metadata = handle.metadata() or {}
        if set(metadata) != _DRAFT_METADATA_KEYS:
            raise ValueError(
                "QKV validation scale draft metadata set is invalid: "
                f"missing={sorted(_DRAFT_METADATA_KEYS - set(metadata))}, "
                f"unknown={sorted(set(metadata) - _DRAFT_METADATA_KEYS)}"
            )
        if metadata["format"] != QKV_TRANSPORT_DRAFT_FORMAT:
            raise ValueError("QKV validation scale draft format is invalid")
        if metadata["format_version"] != QKV_TRANSPORT_DRAFT_VERSION:
            raise ValueError("QKV validation scale draft version is invalid")
        if metadata["scale_formula"] != "nextafter(absmax*1.125/448,+inf)-fp32":
            raise ValueError("QKV validation scale draft formula is invalid")
        if metadata["source_snapshot_count"] != "8":
            raise ValueError("QKV validation scale draft must be derived from exactly eight ranks")
        provenance = qkv_transport_calibration_provenance_from_metadata(metadata)
        if expected_provenance is not None and provenance != expected_provenance:
            raise ValueError("QKV validation scale draft provenance mismatch")
        keys = set(handle.keys())
        if keys != {_VALIDATION_SCALE_KEY}:
            raise ValueError(
                "QKV validation scale artifact tensor set must be exactly "
                f"{{{_VALIDATION_SCALE_KEY!r}}}, got {sorted(keys)}"
            )
        scales = handle.get_tensor(_VALIDATION_SCALE_KEY)
        payload_sha256 = digest_minimax_h3_qkv_transport_scale_payload({_VALIDATION_SCALE_KEY: scales})
        if payload_sha256 != metadata["payload_sha256"]:
            raise ValueError("QKV validation scale draft payload digest mismatch")
    expected_shape = (num_layers, QKV_COMPONENT_COUNT)
    if scales.device.type != "cpu" or scales.dtype != torch.float32:
        raise ValueError("QKV validation dequant_scales must be CPU FP32")
    if scales.shape != expected_shape:
        raise ValueError(f"QKV validation dequant_scales must have shape {expected_shape}, got {tuple(scales.shape)}")
    if not scales.is_contiguous():
        raise ValueError("QKV validation dequant_scales must be contiguous")
    if not bool(torch.isfinite(scales).all()) or not bool((scales > 0).all()):
        raise ValueError("QKV validation dequant_scales must contain only finite positive values")
    if not bool(torch.isfinite(torch.reciprocal(scales)).all()):
        raise ValueError("QKV validation dequant_scales must have finite FP32 reciprocals")
    return scales.detach().clone()


def read_qkv_transport_validation_scale_provenance(
    path: str | Path,
) -> MiniMaxH3QKVTransportCalibrationProvenance:
    """Read the lineage carried by a strict validation-scale draft."""

    candidate = Path(path)
    if not candidate.is_absolute() or candidate.suffix != ".safetensors" or not candidate.is_file():
        raise ValueError("QKV validation scale path must name an existing absolute .safetensors file")
    with safe_open(candidate, framework="pt", device="cpu") as handle:
        metadata = handle.metadata() or {}
    if set(metadata) != _DRAFT_METADATA_KEYS:
        raise ValueError("QKV validation scale draft metadata set is invalid")
    return qkv_transport_calibration_provenance_from_metadata(metadata)


def _strict_env_flag(name: str, *, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    value = raw.strip().lower()
    if value in {"1", "true"}:
        return True
    if value in {"0", "false"}:
        return False
    raise ValueError(f"{name} must be 0/1/false/true, got {raw!r}")


def _rank_without_collective() -> int:
    """Resolve process rank from initialized state or launcher metadata only."""

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        rank = int(torch.distributed.get_rank())
    else:
        raw = os.getenv("RANK", "0")
        try:
            rank = int(raw)
        except ValueError as exc:
            raise ValueError(f"RANK must be a non-negative integer, got {raw!r}") from exc
    if rank < 0:
        raise ValueError(f"RANK must be a non-negative integer, got {rank}")
    return rank


def build_qkv_transport_calibration_observer_from_env(
    *,
    num_layers: int,
    rank: int | None = None,
) -> MiniMaxH3QKVTransportCalibrationObserver | None:
    """Build the strictly opt-in rank-local observer from process env.

    Disabled mode performs no allocations and returns ``None``.  Active mode
    is mutually exclusive with live E4M3 transport because it must observe the
    original BF16 producer values rather than already-quantized payloads.
    """

    raw_mode = os.getenv(QKV_TRANSPORT_OBSERVER_MODE_ENV, "").strip().lower()
    if raw_mode in {"", "0", "false", "off"}:
        stale = [
            name
            for name in (
                QKV_TRANSPORT_OBSERVER_SNAPSHOT_DIR_ENV,
                QKV_TRANSPORT_OBSERVER_SCALES_PATH_ENV,
                QKV_TRANSPORT_OBSERVER_SOURCE_MODEL_REVISION_ENV,
                QKV_TRANSPORT_OBSERVER_RUNTIME_CHECKPOINT_REVISION_ENV,
                QKV_TRANSPORT_OBSERVER_RUNTIME_CHECKPOINT_MANIFEST_SHA256_ENV,
                QKV_TRANSPORT_OBSERVER_CAMPAIGN_UUID_ENV,
                QKV_TRANSPORT_OBSERVER_RUN_CONTRACT_SHA256_ENV,
                QKV_TRANSPORT_OBSERVER_EXPECTED_OBSERVATION_COUNT_ENV,
            )
            if os.getenv(name) not in (None, "")
        ]
        if stale:
            raise ValueError(f"{', '.join(stale)} require {QKV_TRANSPORT_OBSERVER_MODE_ENV}=calibration|validation")
        return None
    if raw_mode not in _OBSERVER_MODES:
        raise ValueError(
            f"{QKV_TRANSPORT_OBSERVER_MODE_ENV} must be disabled, calibration, or validation, got {raw_mode!r}"
        )
    if num_layers != _MINIMAX_H3_NUM_LAYERS:
        raise ValueError(
            "MiniMax H3 QKV transport observer is qualified only for exactly "
            f"{_MINIMAX_H3_NUM_LAYERS} DiT blocks, got {num_layers}"
        )
    if _strict_env_flag(QKV_E4M3_TRANSPORT_ENV):
        raise ValueError(
            f"{QKV_TRANSPORT_OBSERVER_MODE_ENV}={raw_mode} is mutually exclusive with {QKV_E4M3_TRANSPORT_ENV}=1"
        )

    required_provenance_env = {
        "source_model_revision": QKV_TRANSPORT_OBSERVER_SOURCE_MODEL_REVISION_ENV,
        "runtime_checkpoint_revision": QKV_TRANSPORT_OBSERVER_RUNTIME_CHECKPOINT_REVISION_ENV,
        "runtime_checkpoint_manifest_sha256": QKV_TRANSPORT_OBSERVER_RUNTIME_CHECKPOINT_MANIFEST_SHA256_ENV,
        "campaign_uuid": QKV_TRANSPORT_OBSERVER_CAMPAIGN_UUID_ENV,
        "run_contract_sha256": QKV_TRANSPORT_OBSERVER_RUN_CONTRACT_SHA256_ENV,
        "expected_observation_count": QKV_TRANSPORT_OBSERVER_EXPECTED_OBSERVATION_COUNT_ENV,
    }
    provenance_values = {field: os.getenv(env_name, "") for field, env_name in required_provenance_env.items()}
    missing_provenance = [required_provenance_env[field] for field, value in provenance_values.items() if not value]
    if missing_provenance:
        raise ValueError(f"active QKV transport observer requires provenance env: {', '.join(missing_provenance)}")
    raw_expected_count = provenance_values["expected_observation_count"]
    if not raw_expected_count.isdecimal():
        raise ValueError(f"{QKV_TRANSPORT_OBSERVER_EXPECTED_OBSERVATION_COUNT_ENV} must be a positive integer")
    provenance = MiniMaxH3QKVTransportCalibrationProvenance(
        source_model_revision=provenance_values["source_model_revision"],
        runtime_checkpoint_revision=provenance_values["runtime_checkpoint_revision"],
        runtime_checkpoint_manifest_sha256=provenance_values["runtime_checkpoint_manifest_sha256"],
        campaign_uuid=provenance_values["campaign_uuid"],
        run_contract_sha256=provenance_values["run_contract_sha256"],
        expected_observation_count=int(raw_expected_count),
    )

    snapshot_dir_raw = os.getenv(QKV_TRANSPORT_OBSERVER_SNAPSHOT_DIR_ENV, "")
    snapshot_dir = Path(snapshot_dir_raw)
    if not snapshot_dir_raw or not snapshot_dir.is_absolute() or not snapshot_dir.is_dir():
        raise ValueError(f"{QKV_TRANSPORT_OBSERVER_SNAPSHOT_DIR_ENV} must be an existing absolute directory")
    resolved_rank = _rank_without_collective() if rank is None else rank
    if isinstance(resolved_rank, bool) or not isinstance(resolved_rank, int) or resolved_rank < 0:
        raise ValueError("QKV transport observer rank must be a non-negative integer")
    snapshot_path = snapshot_dir / (f"minimax-h3-qkv-transport-{raw_mode}-rank-{resolved_rank:05d}.safetensors")

    scales_path_raw = os.getenv(QKV_TRANSPORT_OBSERVER_SCALES_PATH_ENV, "")
    if raw_mode == "calibration":
        if scales_path_raw:
            raise ValueError(f"{QKV_TRANSPORT_OBSERVER_SCALES_PATH_ENV} is valid only in validation mode")
        validation_scales = None
    else:
        if not scales_path_raw:
            raise ValueError(
                f"{QKV_TRANSPORT_OBSERVER_MODE_ENV}=validation requires {QKV_TRANSPORT_OBSERVER_SCALES_PATH_ENV}"
            )
        validation_scales = load_qkv_transport_validation_scales(
            scales_path_raw,
            num_layers=num_layers,
            expected_provenance=provenance,
        )

    return MiniMaxH3QKVTransportCalibrationObserver(
        mode=raw_mode,
        rank=resolved_rank,
        snapshot_path=snapshot_path,
        num_layers=num_layers,
        provenance=provenance,
        validation_scales=validation_scales,
    )


__all__ = [
    "E4M3_FINITE_ABSMAX",
    "MiniMaxH3QKVTransportCalibrationCollector",
    "MiniMaxH3QKVTransportCalibrationSnapshot",
    "MiniMaxH3QKVTransportCalibrationObserver",
    "MiniMaxH3QKVTransportScaleAudit",
    "QKV_COMPONENTS",
    "QKV_E4M3_HEADROOM",
    "QKV_E4M3_TRANSPORT_ENV",
    "QKV_TRANSPORT_DRAFT_FORMAT",
    "QKV_TRANSPORT_DRAFT_VERSION",
    "QKV_TRANSPORT_OBSERVER_ACTIVE_MARKER",
    "QKV_TRANSPORT_OBSERVER_CAMPAIGN_UUID_ENV",
    "QKV_TRANSPORT_OBSERVER_EXPECTED_OBSERVATION_COUNT_ENV",
    "QKV_TRANSPORT_OBSERVER_MODE_ENV",
    "QKV_TRANSPORT_OBSERVER_RUN_CONTRACT_SHA256_ENV",
    "QKV_TRANSPORT_OBSERVER_RUNTIME_CHECKPOINT_MANIFEST_SHA256_ENV",
    "QKV_TRANSPORT_OBSERVER_RUNTIME_CHECKPOINT_REVISION_ENV",
    "QKV_TRANSPORT_OBSERVER_SCALES_PATH_ENV",
    "QKV_TRANSPORT_OBSERVER_SOURCE_MODEL_REVISION_ENV",
    "QKV_TRANSPORT_OBSERVER_SNAPSHOT_DIR_ENV",
    "QKV_TRANSPORT_OBSERVER_SNAPSHOT_MARKER",
    "audit_qkv_e4m3_scales",
    "build_qkv_transport_calibration_observer_from_env",
    "derive_qkv_e4m3_scales",
    "load_qkv_transport_calibration_snapshot",
    "load_qkv_transport_validation_scales",
    "read_qkv_transport_validation_scale_provenance",
    "merge_qkv_transport_calibration_files",
    "merge_qkv_transport_calibration_snapshots",
    "save_qkv_transport_calibration_snapshot",
    "save_qkv_transport_calibration_snapshot_atomic",
    "qkv_transport_calibration_provenance_from_metadata",
    "qkv_transport_calibration_provenance_to_metadata",
    "qkv_transport_validation_scale_draft_metadata",
]
