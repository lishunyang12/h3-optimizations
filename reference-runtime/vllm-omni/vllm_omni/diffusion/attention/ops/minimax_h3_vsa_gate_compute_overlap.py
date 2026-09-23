# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CUDA-stream lifetime contract for overlapping FastH3's VSA gate GEMM.

The learned compression gate is owner-local in the reverse-O bundle path and
is not consumed until reverse Ulysses finishes.  This module lets the original
BF16 projection run on one persistent side stream while the main stream runs
QKV E4M3 packing, Ulysses, and VSA.  The ticket is deliberately a Python
object: it is created and consumed inside the existing eager attention island
and must never cross a ``torch.compile`` graph boundary.
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn
from vllm.logger import init_logger

logger = init_logger(__name__)

H3_VSA_GATE_COMPUTE_OVERLAP_ENV = "VLLM_OMNI_FASTVIDEO_VSA_GATE_COMPUTE_OVERLAP"
H3_VSA_GATE_COMPUTE_OVERLAP_ACTIVE_KEY = "vsa_h3_gate_compute_overlap_active"
H3_VSA_GATE_COMPUTE_OVERLAP_TICKET_KEY = "vsa_h3_gate_compute_overlap_ticket"
H3_VSA_GATE_COMPUTE_OVERLAP_MARKER = "MINIMAX_H3_VSA_GATE_COMPUTE_OVERLAP_ACTIVE_V1"

_STREAMS: dict[int, torch.cuda.Stream] = {}
_STREAMS_LOCK = threading.Lock()


def h3_vsa_gate_compute_overlap_enabled() -> bool:
    """Return the strictly opt-in gate-compute overlap setting."""
    raw = os.environ.get(H3_VSA_GATE_COMPUTE_OVERLAP_ENV, "0").strip()
    if raw not in {"0", "1"}:
        raise ValueError(
            f"{H3_VSA_GATE_COMPUTE_OVERLAP_ENV} must be exactly '0' or '1', got {raw!r}"
        )
    return raw == "1"


def _device_index(device: torch.device) -> int:
    if device.type != "cuda":
        raise ValueError(f"H3 VSA gate-compute overlap requires CUDA, got {device}")
    return torch.accelerator.current_device_index() if device.index is None else int(device.index)


def _gate_stream(device: torch.device) -> torch.cuda.Stream:
    index = _device_index(device)
    stream = _STREAMS.get(index)
    if stream is not None:
        return stream
    with _STREAMS_LOCK:
        stream = _STREAMS.get(index)
        if stream is None:
            with torch.cuda.device(index):
                stream = torch.cuda.Stream(device=index)
            _STREAMS[index] = stream
    return stream


@dataclass(slots=True)
class H3VSAGateComputeOverlapTicket:
    """Single-use proof that one exact gate is produced on a side stream."""

    gate: torch.Tensor
    ready_event: Any
    producer_stream: Any
    label: str
    _state: str = field(default="launched", init=False)
    _tensor_id: int = field(init=False)
    _data_ptr: int = field(init=False)
    _storage_ptr: int = field(init=False)
    _storage_offset: int = field(init=False)
    _shape: tuple[int, ...] = field(init=False)
    _stride: tuple[int, ...] = field(init=False)
    _dtype: torch.dtype = field(init=False)
    _device: torch.device = field(init=False)
    _is_inference: bool = field(init=False)
    _version: int | None = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.gate, torch.Tensor):
            raise TypeError("gate-overlap ticket requires a Tensor gate")
        self._tensor_id = id(self.gate)
        self._data_ptr = self.gate.data_ptr()
        self._storage_ptr = self.gate.untyped_storage().data_ptr()
        self._storage_offset = self.gate.storage_offset()
        self._shape = tuple(self.gate.shape)
        self._stride = tuple(self.gate.stride())
        self._dtype = self.gate.dtype
        self._device = self.gate.device
        self._is_inference = torch.is_inference(self.gate)
        # Inference tensors deliberately have no version counter.  The ticket
        # itself keeps the exact Tensor and its storage alive, while identity,
        # storage, metadata, and inference-state checks below prevent a view or
        # replacement Tensor from being substituted at the consumer boundary.
        self._version = None if self._is_inference else self.gate._version

    @property
    def state(self) -> str:
        return self._state

    def _validate_gate(self, gate: torch.Tensor) -> None:
        if not isinstance(gate, torch.Tensor):
            raise TypeError("gate-overlap ticket can only authenticate a Tensor")
        if gate is not self.gate or id(gate) != self._tensor_id:
            raise RuntimeError("H3 VSA gate-overlap Tensor identity changed")
        is_inference = torch.is_inference(gate)
        actual = (
            gate.data_ptr(),
            gate.untyped_storage().data_ptr(),
            gate.storage_offset(),
            tuple(gate.shape),
            tuple(gate.stride()),
            gate.dtype,
            gate.device,
            is_inference,
            None if is_inference else gate._version,
        )
        expected = (
            self._data_ptr,
            self._storage_ptr,
            self._storage_offset,
            self._shape,
            self._stride,
            self._dtype,
            self._device,
            self._is_inference,
            self._version,
        )
        if actual != expected:
            raise RuntimeError("H3 VSA gate-overlap Tensor storage or version changed")

    def claim(self, gate: torch.Tensor) -> None:
        self._validate_gate(gate)
        if self._state != "launched":
            raise RuntimeError(f"H3 VSA gate-overlap ticket cannot be claimed from state {self._state!r}")
        self._state = "claimed"

    def mark_wait_enqueued(self, gate: torch.Tensor) -> None:
        self._validate_gate(gate)
        if self._state != "claimed":
            raise RuntimeError(f"H3 VSA gate-overlap wait cannot be enqueued from state {self._state!r}")
        self._state = "waited"

    def mark_consumed(self, gate: torch.Tensor) -> None:
        self._validate_gate(gate)
        if self._state != "waited":
            raise RuntimeError(f"H3 VSA gate-overlap ticket cannot be consumed from state {self._state!r}")
        self._state = "consumed"


def launch_h3_vsa_gate_compute_overlap(
    projector: nn.Module,
    gate_input: torch.Tensor,
    *,
    output_shape: tuple[int, int, int, int],
    label: str,
) -> tuple[torch.Tensor, H3VSAGateComputeOverlapTicket]:
    """Launch the unchanged BF16 gate projection after prior main-stream work."""
    if not h3_vsa_gate_compute_overlap_enabled():
        raise RuntimeError("H3 VSA gate-compute overlap launch reached while its env gate is disabled")
    if gate_input.device.type != "cuda" or gate_input.dtype != torch.bfloat16:
        raise TypeError(
            "H3 VSA gate-compute overlap requires a CUDA BF16 input, "
            f"got device={gate_input.device}, dtype={gate_input.dtype}"
        )
    if gate_input.requires_grad:
        raise RuntimeError("H3 VSA gate-compute overlap is inference-only")
    if torch.is_grad_enabled():
        raise RuntimeError("H3 VSA gate-compute overlap requires gradients to be disabled")
    if not gate_input.is_contiguous():
        raise ValueError("H3 VSA gate-compute overlap requires a contiguous input")
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("H3 VSA gate-compute overlap does not support CUDA graph capture")

    parameters = tuple(projector.parameters())
    if not parameters:
        raise RuntimeError("H3 VSA gate-compute overlap projector has no parameters")
    for parameter in parameters:
        if parameter.device != gate_input.device or parameter.dtype != torch.bfloat16:
            raise TypeError(
                "H3 VSA gate-compute overlap requires CUDA BF16 projector parameters on the input device"
            )

    main_stream = torch.cuda.current_stream(gate_input.device)
    side_stream = _gate_stream(gate_input.device)
    side_stream.wait_stream(main_stream)
    with torch.cuda.stream(side_stream), torch.cuda.nvtx.range(
        f"minimax_h3.gate_compress.side:{label}"
    ):
        gate_input.record_stream(side_stream)
        for parameter in parameters:
            parameter.record_stream(side_stream)
        result = projector(gate_input)
        gate = result[0] if isinstance(result, tuple) else result
        if not isinstance(gate, torch.Tensor):
            raise TypeError("H3 VSA side-stream gate projection did not return a Tensor")
        gate = gate.view(output_shape)
        ready_event = torch.cuda.Event(enable_timing=False)
        ready_event.record(side_stream)

    if gate.device != gate_input.device or gate.dtype != torch.bfloat16:
        raise TypeError("H3 VSA side-stream gate projection did not preserve CUDA BF16 output")
    if not gate.is_contiguous() or gate.requires_grad:
        raise RuntimeError("H3 VSA side-stream gate output must be contiguous and inference-only")
    if gate.untyped_storage().data_ptr() == gate_input.untyped_storage().data_ptr():
        raise RuntimeError("H3 VSA side-stream gate output must not alias its input")
    ticket = H3VSAGateComputeOverlapTicket(
        gate=gate,
        ready_event=ready_event,
        producer_stream=side_stream,
        label=label,
    )
    logger.info_once(
        "%s route=post_qkv_to_reverse_o gate=%s dtype=%s device=%s",
        H3_VSA_GATE_COMPUTE_OVERLAP_MARKER,
        tuple(gate.shape),
        gate.dtype,
        gate.device,
        scope="process",
    )
    return gate, ticket


def wait_h3_vsa_gate_compute_overlap(
    ticket: H3VSAGateComputeOverlapTicket,
    gate: torch.Tensor,
) -> None:
    """Put the exact gate-ready dependency on its current consumer stream."""
    if not isinstance(ticket, H3VSAGateComputeOverlapTicket):
        raise TypeError("H3 VSA gate-compute overlap context contains an invalid ticket")
    ticket._validate_gate(gate)
    if ticket.state != "claimed":
        raise RuntimeError(f"H3 VSA gate-overlap wait reached ticket state {ticket.state!r}")
    consumer_stream = torch.cuda.current_stream(gate.device)
    with torch.cuda.nvtx.range(f"minimax_h3.gate_compress.wait:{ticket.label}"):
        consumer_stream.wait_event(ticket.ready_event)
        gate.record_stream(consumer_stream)
    ticket.mark_wait_enqueued(gate)


__all__ = [
    "H3VSAGateComputeOverlapTicket",
    "H3_VSA_GATE_COMPUTE_OVERLAP_ACTIVE_KEY",
    "H3_VSA_GATE_COMPUTE_OVERLAP_ENV",
    "H3_VSA_GATE_COMPUTE_OVERLAP_MARKER",
    "H3_VSA_GATE_COMPUTE_OVERLAP_TICKET_KEY",
    "h3_vsa_gate_compute_overlap_enabled",
    "launch_h3_vsa_gate_compute_overlap",
    "wait_h3_vsa_gate_compute_overlap",
]
