#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Read-only analysis of a MiniMax-H3 Nsight Systems capture.

The analyzer exports an ``.nsys-rep`` into a temporary SQLite database (or
opens an existing SQLite export with ``mode=ro``), invokes the canonical
``nsys stats`` reports as a cross-check, and produces a conservative H3-aware
breakdown.  Classification is name-evidence based: generic GEMMs are never
guessed to be FC1/FC2/QKV/out projections, and generic CUDA/OS runtime stalls
are never presented as RDMA time.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sqlite3
import statistics
import subprocess
import sys
import tempfile
from collections import defaultdict
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_NVTX_DOMAIN = "vllm_omni.minimax_h3"
DEFAULT_STEP_PREFIX = "minimax_h3.denoise.step_"

KERNEL_CATEGORY_DESCRIPTIONS = {
    "vsa_attention": (
        "Explicit block-sparse/VSA attention kernel, including FlashInfer CuTeDSL SM120 BlockSparseAttnForward"
    ),
    "vsa_indexing": "Explicit VSA block-map-to-index compaction kernel",
    "cudnn_attention": "cuDNN kernel with explicit SDPA/attention/FMHA evidence",
    "cudnn_other": "cuDNN kernel without explicit attention evidence",
    "gemm_fc1": "GEMM with an explicit FC1/gate/up-projection token",
    "gemm_fc2": "GEMM with an explicit FC2/down-projection token",
    "gemm_qkv": "GEMM with an explicit Q/K/V/QKV-projection token",
    "gemm_out": "GEMM with an explicit attention output-projection token",
    "gemm_unknown": "Recognizable GEMM, but its H3 layer role is not encoded in the kernel name",
    "flashinfer_rdma_barrier": (
        "Explicit FlashInfer Ulysses PCIe/RDMA barrier; duration is synchronization residency/wait"
    ),
    "flashinfer_rdma_transfer": ("Explicit FlashInfer Ulysses PCIe/RDMA data-movement kernel"),
    "flashinfer_rdma_other": ("Explicit FlashInfer Ulysses PCIe/RDMA kernel whose role is not distinguishable by name"),
    "flashinfer_other": "FlashInfer kernel without explicit PCIe/RDMA evidence",
    "nccl": "NCCL device kernel",
    "triton_activation": "Triton kernel with an explicit activation token",
    "triton_pointwise": "Triton pointwise kernel without a more specific activation token",
    "triton_reduction": "Triton reduction kernel",
    "triton_other": "Other Triton kernel",
    "h3_pointwise": ("Explicitly named H3 fused modulation, gated-residual, Q/K norm+RoPE, or SwiGLU kernel"),
    "vsa_layout": "Explicit H3 VSA compact/tile layout kernel",
    "attention_other": "Non-cuDNN kernel with explicit attention/FMHA/flash-attention evidence",
    "copy_kernel": "Device-side copy/cat/transpose-like kernel (not a DMA memcpy record)",
    "unknown": "No conservative category matched",
}

KERNEL_WALL_PHASE_DESCRIPTIONS = {
    "linear_gemm": "Recognizable GEMM; individual H3 projection role may remain unknown",
    "vsa": "Explicit VSA attention or VSA index-compaction kernel",
    "rdma_barrier": "Explicit FlashInfer Ulysses PCIe barrier residency",
    "rdma_other": "Other explicitly named FlashInfer PCIe/RDMA device work",
    "pointwise": "Explicit Triton or H3 fused pointwise/reduction work",
    "attention_other": "Dense/non-VSA attention work",
    "communication_other": "Non-RDMA device communication work such as NCCL",
    "layout_copy": "VSA layout or other explicitly named device layout/copy work",
    "memory_ops": "CUDA DMA memcpy or memset work",
    "other": "Kernel work not conservatively assigned above",
    "concurrent_mixed": "Two or more different phases active on this device",
    "idle": "No traced CUDA kernel, memcpy, or memset active on this device",
}

CUDA_API_CATEGORY_DESCRIPTIONS = {
    "synchronization_wait": "CUDA synchronize calls; duration includes waiting and is not CPU overhead alone",
    "kernel_launch": "CUDA kernel/graph launch APIs",
    "memcpy": "CUDA memcpy APIs",
    "memset": "CUDA memset APIs",
    "external_memory_possible_rdma_setup": (
        "CUDA external/shareable-memory APIs; possible registration setup, not measured RDMA transfer time"
    ),
    "memory_management": "CUDA allocation, mapping, and free APIs",
    "event_stream": "CUDA event and stream bookkeeping APIs",
    "other": "Other CUDA APIs",
}

_ACTIVATION_TOKENS = ("silu", "swiglu", "gelu", "relu", "sigmoid", "tanh")
_ATTENTION_TOKENS = ("sdpa", "attention", "fmha", "flash_fwd", "flash_attention")
_RDMA_TOKENS = (
    "ulysses_pcie",
    "ulyssespcie",
    "ulysses::pcie",
    "ulysses_rdma",
    "flashinfer::comm",
    "flashinfer_comm",
    "ibgda",
    "nvshmem",
    "rdma_scatter",
    "rdma_gather",
)
_RDMA_BARRIER_TOKENS = ("barrier", "wait", "signal", "fence")
_RDMA_TRANSFER_TOKENS = ("scatter", "gather", "exchange", "copy", "write", "read")
_GEMM_TOKENS = ("gemm", "cublas", "matmul", "wgmma")
_VSA_ATTENTION_TOKENS = (
    "blocksparseattn",
    "block_sparse_attn",
    "block_sparse_attention",
    "bsa_attn",
)
_VSA_INDEX_TOKENS = ("map_to_index_kernel",)
_H3_POINTWISE_TOKENS = (
    "_indexed_scale_shift_kernel",
    "_indexed_gate_kernel",
    "_rms_norm_indexed_scale_shift_kernel",
    "_indexed_gate_rms_norm_scale_shift_kernel",
    "_rmsnorm_modulate_kernel",
    "_residual_gate_rmsnorm_modulate_kernel",
    "_qknorm_partial_rope_kernel",
    "_rms_norm_rope_kernel",
    "_minimax_h3_swiglu_kernel",
)
_VSA_LAYOUT_TOKENS = (
    "_h3_vsa_tile_pack_kernel",
    "_h3_vsa_tile_untile_kernel",
    "_h3_vsa_gate_untile_kernel",
    "_h3_vsa_deferred_gate_kernel",
    "_h3_vsa_o_bundle_kernel",
    "_h3_vsa_o_bundle_local_gate_kernel",
)
_FC1_RE = re.compile(r"(?:^|[._:/])(?:fc1|gate_proj|up_proj|gate_up)(?:$|[._:/<])")
_FC2_RE = re.compile(r"(?:^|[._:/])(?:fc2|down_proj)(?:$|[._:/<])")
_QKV_RE = re.compile(r"(?:^|[._:/])(?:qkv|q_proj|k_proj|v_proj|query_key_value)(?:$|[._:/<])")
_OUT_RE = re.compile(r"(?:^|[._:/])(?:out_proj|o_proj|proj_out|attention_output)(?:$|[._:/<])")
_NCCL_DEVICE_RE = re.compile(
    r"(?P<pid>\d+):\d+\s+\[(?P<logger_rank>\d+)\]\s+NCCL INFO\s+"
    r".*?\bncclCommInitRankConfig\b.*?\brank\s+(?P<rank>\d+)\s+"
    r"nranks\s+\d+\s+cudaDev\s+(?P<cuda_device>\d+)\s+"
    r"nvmlDev\s+(?P<nvml_device>\d+)\s+busId\s+(?P<bus_id>[0-9a-fA-F]+)\b"
)


@dataclass(frozen=True)
class AnalysisScope:
    start_ns: int
    end_ns: int
    kind: str
    name: str
    domain: str | None
    pid: int | None

    @property
    def duration_ns(self) -> int:
        return self.end_ns - self.start_ns


def _process_key(serialized_id: int | None) -> int | None:
    if serialized_id is None:
        return None
    return int(serialized_id) & ~0xFFFFFF


def _linux_pid(serialized_id: int | None) -> int | None:
    if serialized_id is None:
        return None
    return (int(serialized_id) >> 24) & 0xFFFFFF


def _overlap_ns(start: int, end: int, scope: AnalysisScope) -> int:
    return max(0, min(int(end), scope.end_ns) - max(int(start), scope.start_ns))


def _percentile_nearest_rank(values: Sequence[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int((len(ordered) * percentile) + 0.999999999) - 1))
    return float(ordered[index])


def classify_kernel(name: str) -> str:
    """Classify only semantics explicitly visible in a kernel name."""
    lowered = name.lower()
    if "cudnn" in lowered:
        if any(token in lowered for token in _ATTENTION_TOKENS):
            return "cudnn_attention"
        return "cudnn_other"
    if any(token in lowered for token in _RDMA_TOKENS):
        if any(token in lowered for token in _RDMA_BARRIER_TOKENS):
            return "flashinfer_rdma_barrier"
        if any(token in lowered for token in _RDMA_TRANSFER_TOKENS):
            return "flashinfer_rdma_transfer"
        return "flashinfer_rdma_other"
    # FlashInfer #4944's CuTeDSL symbol begins with ``kernel_cutlass_kernel``
    # and also contains ``flashinfer``.  Match its explicit block-sparse
    # attention evidence before the generic FlashInfer and GEMM branches.
    if any(token in lowered for token in _VSA_ATTENTION_TOKENS):
        return "vsa_attention"
    if any(token in lowered for token in _VSA_INDEX_TOKENS):
        return "vsa_indexing"
    if any(token in lowered for token in _VSA_LAYOUT_TOKENS):
        return "vsa_layout"
    if any(token in lowered for token in _H3_POINTWISE_TOKENS):
        return "h3_pointwise"
    if "flashinfer" in lowered:
        return "flashinfer_other"
    if "nccl" in lowered:
        return "nccl"
    if "triton" in lowered:
        if any(token in lowered for token in _ACTIVATION_TOKENS):
            return "triton_activation"
        if any(token in lowered for token in _GEMM_TOKENS):
            return _classify_gemm_role(lowered)
        if "triton_poi" in lowered or "pointwise" in lowered:
            return "triton_pointwise"
        if "triton_red" in lowered or "reduction" in lowered:
            return "triton_reduction"
        return "triton_other"
    if any(token in lowered for token in _ATTENTION_TOKENS):
        return "attention_other"
    if any(token in lowered for token in _GEMM_TOKENS) or ("cutlass::kernel" in lowered and "mma" in lowered):
        return _classify_gemm_role(lowered)
    if any(token in lowered for token in ("copy_kernel", "catarray", "transpose", "permute", "index_select")):
        return "copy_kernel"
    return "unknown"


def _classify_gemm_role(lowered_name: str) -> str:
    if _FC1_RE.search(lowered_name):
        return "gemm_fc1"
    if _FC2_RE.search(lowered_name):
        return "gemm_fc2"
    if _QKV_RE.search(lowered_name):
        return "gemm_qkv"
    if _OUT_RE.search(lowered_name):
        return "gemm_out"
    return "gemm_unknown"


def classify_cuda_api(name: str) -> str:
    lowered = name.lower()
    if "synchronize" in lowered:
        return "synchronization_wait"
    if "launch" in lowered:
        return "kernel_launch"
    if "memcpy" in lowered:
        return "memcpy"
    if "memset" in lowered:
        return "memset"
    if any(token in lowered for token in ("shareablehandle", "externalmemory", "memexport", "memimport")):
        return "external_memory_possible_rdma_setup"
    if any(token in lowered for token in ("malloc", "free", "memcreate", "memmap", "memunmap", "addressreserve")):
        return "memory_management"
    if "event" in lowered or "stream" in lowered:
        return "event_stream"
    return "other"


def _table_exists(connection: sqlite3.Connection, name: str) -> bool:
    return (
        connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
            (name,),
        ).fetchone()
        is not None
    )


def _open_read_only(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True)
    connection.execute("PRAGMA query_only=ON")
    connection.execute("PRAGMA temp_store=MEMORY")
    return connection


@contextmanager
def materialize_sqlite(
    source: Path,
    *,
    nsys: str,
    sqlite_override: Path | None = None,
) -> Iterator[tuple[Path, str]]:
    """Yield a SQLite export without modifying the input report directory."""
    if sqlite_override is not None:
        if not sqlite_override.is_file():
            raise FileNotFoundError(f"SQLite override does not exist: {sqlite_override}")
        yield sqlite_override, "explicit SQLite override"
        return
    if source.suffix == ".sqlite":
        yield source, "input SQLite"
        return
    if source.suffix != ".nsys-rep":
        raise ValueError("input must end in .nsys-rep or .sqlite")
    sibling = source.with_suffix(".sqlite")
    if sibling.is_file() and sibling.stat().st_mtime_ns >= source.stat().st_mtime_ns:
        yield sibling, "up-to-date sibling SQLite"
        return

    with tempfile.TemporaryDirectory(prefix="h3-nsys-analysis-") as temp_dir:
        exported = Path(temp_dir) / "report.sqlite"
        command = [
            nsys,
            "export",
            "--type=sqlite",
            f"--output={exported}",
            "--force-overwrite=true",
            "--quiet=true",
            str(source),
        ]
        completed = subprocess.run(command, check=False, capture_output=True, text=True)
        if completed.returncode != 0 or not exported.is_file():
            detail = completed.stderr.strip() or completed.stdout.strip()
            raise RuntimeError(f"nsys export failed ({completed.returncode}): {detail}")
        yield exported, "temporary nsys export"


def _nvtx_ranges(connection: sqlite3.Connection) -> list[dict[str, Any]]:
    if not _table_exists(connection, "NVTX_EVENTS"):
        return []
    domains: dict[tuple[int | None, int], str] = {}
    for global_tid, domain_id, text in connection.execute(
        """
        SELECT globalTid, domainId, text
        FROM NVTX_EVENTS
        WHERE end IS NULL AND domainId IS NOT NULL AND text IS NOT NULL
        """
    ):
        domains[(_process_key(global_tid), int(domain_id))] = str(text)

    ranges = []
    for start, end, global_tid, domain_id, name in connection.execute(
        """
        SELECT e.start, e.end, e.globalTid, e.domainId,
               COALESCE(e.text, strings.value)
        FROM NVTX_EVENTS AS e
        LEFT JOIN StringIds AS strings ON strings.id = e.textId
        WHERE e.end IS NOT NULL AND COALESCE(e.text, strings.value) IS NOT NULL
        ORDER BY e.start
        """
    ):
        process_key = _process_key(global_tid)
        domain = domains.get((process_key, int(domain_id))) if domain_id is not None else None
        ranges.append(
            {
                "start_ns": int(start),
                "end_ns": int(end),
                "name": str(name),
                "domain": domain,
                "pid": _linux_pid(global_tid),
            }
        )
    return ranges


def select_scope(
    connection: sqlite3.Connection,
    *,
    requested_range: str,
    requested_domain: str | None,
) -> tuple[AnalysisScope, list[str]]:
    warnings: list[str] = []
    ranges = _nvtx_ranges(connection)
    candidates: list[dict[str, Any]] = []
    kind = "nvtx_range"
    if requested_range == "none":
        candidates = []
    elif requested_range != "auto":
        candidates = [item for item in ranges if item["name"] == requested_range]
        if requested_domain is not None:
            candidates = [item for item in candidates if item["domain"] == requested_domain]
        if not candidates:
            suffix = f"@{requested_domain}" if requested_domain else ""
            raise ValueError(f"NVTX range not found: {requested_range}{suffix}")
    else:
        candidates = [item for item in ranges if item["name"].startswith(DEFAULT_STEP_PREFIX)]
        if candidates:
            kind = "denoise_step_nvtx"
        else:
            candidates = [item for item in ranges if item["name"] == "minimax_h3_task:t2va"]
            if candidates:
                kind = "t2va_task_nvtx_fallback"
                warnings.append(
                    "No denoise-step NVTX range was present; results cover the full T2VA task range, not one step."
                )
    if candidates:
        candidates.sort(key=lambda item: (item["start_ns"], item["end_ns"]))
        selected = candidates[0]
        if len(candidates) > 1:
            warnings.append(
                f"Found {len(candidates)} matching NVTX ranges; selected the earliest and ignored the rest."
            )
        return (
            AnalysisScope(
                start_ns=selected["start_ns"],
                end_ns=selected["end_ns"],
                kind=kind,
                name=selected["name"],
                domain=selected["domain"],
                pid=selected["pid"],
            ),
            warnings,
        )

    bounds = []
    for table in (
        "CUPTI_ACTIVITY_KIND_KERNEL",
        "CUPTI_ACTIVITY_KIND_MEMCPY",
        "CUPTI_ACTIVITY_KIND_MEMSET",
    ):
        if _table_exists(connection, table):
            row = connection.execute(f"SELECT MIN(start), MAX(end) FROM {table}").fetchone()
            if row and row[0] is not None and row[1] is not None:
                bounds.append((int(row[0]), int(row[1])))
    if not bounds:
        raise ValueError("report contains no CUDA GPU kernel or memory-operation timeline")
    warnings.append(
        "No usable NVTX range was selected; results cover the complete CUDA GPU trace and may include startup."
    )
    return (
        AnalysisScope(
            start_ns=min(item[0] for item in bounds),
            end_ns=max(item[1] for item in bounds),
            kind="full_cuda_trace",
            name="full CUDA trace",
            domain=None,
            pid=None,
        ),
        warnings,
    )


def _gpu_metadata(connection: sqlite3.Connection) -> dict[int, dict[str, Any]]:
    if not _table_exists(connection, "TARGET_INFO_GPU"):
        return {}
    metadata = {}
    for device_id, name, bus, uuid in connection.execute("SELECT id, name, busLocation, uuid FROM TARGET_INFO_GPU"):
        metadata[int(device_id)] = {
            "name": name,
            "bus_location": bus,
            "uuid": uuid,
        }
    return metadata


def _normalize_pci_bus(bus: str | None) -> str | None:
    if bus is None:
        return None
    match = re.fullmatch(
        r"(?P<domain>[0-9a-fA-F]{4,8}):(?P<bus>[0-9a-fA-F]{2}):"
        r"(?P<device>[0-9a-fA-F]{2})\.(?P<function>[0-9a-fA-F])",
        str(bus).strip(),
    )
    if match is None:
        return str(bus).strip().lower()
    return (
        f"{int(match.group('domain'), 16):04x}:"
        f"{int(match.group('bus'), 16):02x}:"
        f"{int(match.group('device'), 16):02x}."
        f"{int(match.group('function'), 16):x}"
    )


def _nccl_bus_id_to_location(bus_id: str) -> str:
    encoded = int(bus_id, 16)
    domain = encoded >> 20
    bus = (encoded >> 12) & 0xFF
    device = (encoded >> 4) & 0x1F
    function = encoded & 0xF
    return f"{domain:04x}:{bus:02x}:{device:02x}.{function:x}"


def _parse_server_nccl_device_map(
    server_log: Path,
) -> tuple[dict[int, dict[str, Any]], list[str]]:
    """Return authoritative PID-to-device identities from NCCL startup logs.

    CUPTI ``deviceId`` and ``TARGET_INFO_GPU.id`` are different namespaces in
    captures launched with a reordered ``CUDA_VISIBLE_DEVICES``. The worker
    PID recorded on each kernel can instead be joined to NCCL's rank/cudaDev/
    nvmlDev/busId line in the server log.
    """
    warnings: list[str] = []
    records: dict[int, dict[str, Any]] = {}
    conflicted_pids: set[int] = set()
    try:
        lines = server_log.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError as error:
        return {}, [f"Could not read server log for device mapping: {server_log}: {error}"]

    for line in lines:
        for match in _NCCL_DEVICE_RE.finditer(line):
            pid = int(match.group("pid"))
            if pid in conflicted_pids:
                continue
            record = {
                "pid": pid,
                "rank": int(match.group("rank")),
                "cuda_device": int(match.group("cuda_device")),
                "physical_gpu": int(match.group("nvml_device")),
                "bus_location": _nccl_bus_id_to_location(match.group("bus_id")),
            }
            logger_rank = int(match.group("logger_rank"))
            if logger_rank != record["rank"]:
                warnings.append(
                    f"NCCL logger rank {logger_rank} disagrees with communicator rank "
                    f"{record['rank']} for PID {pid}; ignored that line."
                )
                continue
            previous = records.get(pid)
            if previous is not None and previous != record:
                warnings.append(f"Conflicting NCCL device mappings for PID {pid}; device identity is unresolved.")
                records.pop(pid, None)
                conflicted_pids.add(pid)
                continue
            records[pid] = record
    if not records:
        warnings.append(f"No NCCL rank/cudaDev/nvmlDev/busId mappings found in {server_log}.")
    return records, warnings


def _resolve_gpu_identities(
    gpu_data: dict[int, dict[str, Any]],
    target_metadata: dict[int, dict[str, Any]],
    *,
    server_log: Path | None,
) -> tuple[dict[int, dict[str, Any]], list[str]]:
    warnings: list[str] = []
    pid_map: dict[int, dict[str, Any]] = {}
    if server_log is not None:
        pid_map, parse_warnings = _parse_server_nccl_device_map(server_log)
        warnings.extend(parse_warnings)

    metadata_by_bus = {
        normalized: info
        for info in target_metadata.values()
        if (normalized := _normalize_pci_bus(info.get("bus_location"))) is not None
    }
    identities: dict[int, dict[str, Any]] = {}
    for gpu in sorted(gpu_data):
        pids = sorted(gpu_data[gpu].get("pids", set()))
        candidates = [pid_map[pid] for pid in pids if pid in pid_map]
        unique = {
            (
                item["rank"],
                item["cuda_device"],
                item["physical_gpu"],
                item["bus_location"],
            ): item
            for item in candidates
        }
        if len(unique) != 1:
            identities[gpu] = {
                "rank": None,
                "cuda_device": None,
                "physical_gpu": None,
                "bus_location": None,
                "uuid": None,
                "name": None,
                "mapping_source": "unresolved",
            }
            if server_log is not None:
                warnings.append(
                    f"CUPTI trace device {gpu} with PID(s) {pids or ['n/a']} did not resolve "
                    "to exactly one NCCL device mapping."
                )
            continue

        record = next(iter(unique.values()))
        if record["cuda_device"] != gpu:
            warnings.append(
                f"CUPTI trace device {gpu} maps by PID to NCCL cudaDev "
                f"{record['cuda_device']}; retaining the PID-derived physical identity."
            )
        info = metadata_by_bus.get(_normalize_pci_bus(record["bus_location"]), {})
        identities[gpu] = {
            **record,
            "uuid": info.get("uuid"),
            "name": info.get("name"),
            "mapping_source": "kernel PID -> server NCCL rank/nvmlDev/busId",
        }

    if server_log is None and target_metadata and gpu_data:
        warnings.append(
            "PCI bus/UUID metadata is intentionally unresolved: CUPTI deviceId is not "
            "assumed equal to TARGET_INFO_GPU.id after CUDA_VISIBLE_DEVICES reordering. "
            "Provide --server-log (or place server-nsys.log beside the report) for a "
            "PID-to-NCCL rank/nvmlDev/bus mapping."
        )
    return identities, warnings


def _kernel_analysis(
    connection: sqlite3.Connection,
    scope: AnalysisScope,
    *,
    top: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[int, dict[str, Any]], dict[Any, list[int]]]:
    if not _table_exists(connection, "CUPTI_ACTIVITY_KIND_KERNEL"):
        return [], [], {}, {}
    categories: dict[str, dict[str, Any]] = {
        name: {"count": 0, "duration_ns": 0, "per_gpu_ns": defaultdict(int)} for name in KERNEL_CATEGORY_DESCRIPTIONS
    }
    kernel_names: dict[tuple[str, str], dict[str, int]] = defaultdict(lambda: {"count": 0, "duration_ns": 0})
    gpu_data: dict[int, dict[str, Any]] = defaultdict(lambda: {"kernel_count": 0, "kernel_sum_ns": 0, "pids": set()})
    rdma_correlations: dict[Any, list[int]] = defaultdict(list)
    cursor = connection.execute(
        """
        SELECT kernels.start, kernels.end, kernels.deviceId, kernels.globalPid,
               kernels.correlationId, strings.value
        FROM CUPTI_ACTIVITY_KIND_KERNEL AS kernels
        JOIN StringIds AS strings ON strings.id = kernels.demangledName
        WHERE kernels.end > ? AND kernels.start < ?
        """,
        (scope.start_ns, scope.end_ns),
    )
    for start, end, device_id, global_pid, correlation_id, kernel_name in cursor:
        duration_ns = _overlap_ns(start, end, scope)
        if duration_ns <= 0:
            continue
        gpu = int(device_id)
        name = str(kernel_name)
        category = classify_kernel(name)
        category_data = categories[category]
        category_data["count"] += 1
        category_data["duration_ns"] += duration_ns
        category_data["per_gpu_ns"][gpu] += duration_ns
        name_data = kernel_names[(category, name)]
        name_data["count"] += 1
        name_data["duration_ns"] += duration_ns
        gpu_data[gpu]["kernel_count"] += 1
        gpu_data[gpu]["kernel_sum_ns"] += duration_ns
        pid = _linux_pid(global_pid)
        if pid is not None:
            gpu_data[gpu]["pids"].add(pid)
        if category.startswith("flashinfer_rdma_") and correlation_id is not None:
            rdma_correlations[(_process_key(global_pid), int(correlation_id))].append(int(start))

    all_gpus = sorted(gpu_data)
    total_kernel_ns = sum(item["duration_ns"] for item in categories.values())
    category_rows = []
    for category, data in categories.items():
        per_gpu_ms = {str(gpu): data["per_gpu_ns"].get(gpu, 0) / 1e6 for gpu in all_gpus}
        values = list(per_gpu_ms.values())
        slowest_gpu = max(all_gpus, key=lambda gpu: data["per_gpu_ns"].get(gpu, 0)) if all_gpus else None
        category_rows.append(
            {
                "category": category,
                "description": KERNEL_CATEGORY_DESCRIPTIONS[category],
                "count": data["count"],
                "total_gpu_ms": data["duration_ns"] / 1e6,
                "percent_of_kernel_gpu_time": (
                    100.0 * data["duration_ns"] / total_kernel_ns if total_kernel_ns else 0.0
                ),
                "median_gpu_ms": statistics.median(values) if values else None,
                "slowest_gpu": slowest_gpu,
                "slowest_gpu_ms": (data["per_gpu_ns"].get(slowest_gpu, 0) / 1e6 if slowest_gpu is not None else None),
                "per_gpu_ms": per_gpu_ms,
            }
        )
    category_rows.sort(key=lambda item: (-item["total_gpu_ms"], item["category"]))

    top_rows = []
    for (category, name), data in sorted(kernel_names.items(), key=lambda item: (-item[1]["duration_ns"], item[0][1]))[
        :top
    ]:
        top_rows.append(
            {
                "category": category,
                "name": name,
                "count": data["count"],
                "total_gpu_ms": data["duration_ns"] / 1e6,
                "average_us": data["duration_ns"] / max(1, data["count"]) / 1e3,
                "percent_of_kernel_gpu_time": (
                    100.0 * data["duration_ns"] / total_kernel_ns if total_kernel_ns else 0.0
                ),
            }
        )
    return category_rows, top_rows, gpu_data, rdma_correlations


def _kernel_wall_phase(category: str) -> str:
    """Collapse a name-evidence category into a coarse wall-time phase."""
    if category.startswith("gemm_"):
        return "linear_gemm"
    if category in {"vsa_attention", "vsa_indexing"}:
        return "vsa"
    if category == "flashinfer_rdma_barrier":
        return "rdma_barrier"
    if category in {"flashinfer_rdma_transfer", "flashinfer_rdma_other"}:
        return "rdma_other"
    if category in {
        "triton_activation",
        "triton_pointwise",
        "triton_reduction",
        "h3_pointwise",
    }:
        return "pointwise"
    if category in {"cudnn_attention", "attention_other"}:
        return "attention_other"
    if category == "nccl":
        return "communication_other"
    if category in {"copy_kernel", "vsa_layout"}:
        return "layout_copy"
    return "other"


def _kernel_phase_wall_analysis(
    connection: sqlite3.Connection,
    scope: AnalysisScope,
    *,
    expected_devices: Sequence[int],
) -> dict[str, Any]:
    """Partition each device's selected wall window without double-counting.

    Summed kernel duration is useful for work accounting but can double-count
    kernels on concurrent streams.  This sweep instead splits each trace
    device's wall clock into disjoint segments.  A segment covered by one phase
    (even by multiple same-phase events) is charged once to that phase; a
    segment with different phases active together is charged once to
    ``concurrent_mixed``.  Idle plus every returned phase therefore equals the
    selected wall window exactly for every device.
    """
    changes: dict[int, dict[int, dict[str, int]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    event_counts: dict[int, int] = defaultdict(int)

    if _table_exists(connection, "CUPTI_ACTIVITY_KIND_KERNEL"):
        cursor = connection.execute(
            """
            SELECT kernels.deviceId, kernels.start, kernels.end, strings.value
            FROM CUPTI_ACTIVITY_KIND_KERNEL AS kernels
            JOIN StringIds AS strings ON strings.id = kernels.demangledName
            WHERE kernels.end > ? AND kernels.start < ?
            """,
            (scope.start_ns, scope.end_ns),
        )
        for device_id, raw_start, raw_end, name in cursor:
            gpu = int(device_id)
            start = max(int(raw_start), scope.start_ns)
            end = min(int(raw_end), scope.end_ns)
            if end <= start:
                continue
            phase = _kernel_wall_phase(classify_kernel(str(name)))
            changes[gpu][start][phase] += 1
            changes[gpu][end][phase] -= 1
            event_counts[gpu] += 1

    for table in ("CUPTI_ACTIVITY_KIND_MEMCPY", "CUPTI_ACTIVITY_KIND_MEMSET"):
        if not _table_exists(connection, table):
            continue
        for device_id, raw_start, raw_end in connection.execute(
            f"SELECT deviceId, start, end FROM {table} WHERE end > ? AND start < ?",
            (scope.start_ns, scope.end_ns),
        ):
            gpu = int(device_id)
            start = max(int(raw_start), scope.start_ns)
            end = min(int(raw_end), scope.end_ns)
            if end <= start:
                continue
            changes[gpu][start]["memory_ops"] += 1
            changes[gpu][end]["memory_ops"] -= 1
            event_counts[gpu] += 1

    devices = sorted(set(expected_devices) | set(changes))
    per_device = []
    phase_names = list(KERNEL_WALL_PHASE_DESCRIPTIONS)
    for gpu in devices:
        durations_ns = {phase: 0 for phase in phase_names}
        active: dict[str, int] = defaultdict(int)
        previous = scope.start_ns
        for timestamp in sorted(changes[gpu]):
            if timestamp > previous:
                active_phases = [phase for phase, count in active.items() if count > 0]
                if not active_phases:
                    phase = "idle"
                elif len(active_phases) == 1:
                    phase = active_phases[0]
                else:
                    phase = "concurrent_mixed"
                durations_ns[phase] += timestamp - previous
            for phase, delta in changes[gpu][timestamp].items():
                active[phase] += delta
                if active[phase] == 0:
                    del active[phase]
            previous = timestamp
        if previous < scope.end_ns:
            active_phases = [phase for phase, count in active.items() if count > 0]
            if not active_phases:
                phase = "idle"
            elif len(active_phases) == 1:
                phase = active_phases[0]
            else:
                phase = "concurrent_mixed"
            durations_ns[phase] += scope.end_ns - previous

        accounted_ns = sum(durations_ns.values())
        per_device.append(
            {
                "gpu": gpu,
                "event_count": event_counts[gpu],
                "scope_wall_ms": scope.duration_ns / 1e6,
                "busy_wall_ms": (scope.duration_ns - durations_ns["idle"]) / 1e6,
                "phase_wall_ms": {phase: durations_ns[phase] / 1e6 for phase in phase_names},
                "phase_percent_of_scope_wall": {
                    phase: (100.0 * durations_ns[phase] / scope.duration_ns if scope.duration_ns else 0.0)
                    for phase in phase_names
                },
                "accounting_error_ns": accounted_ns - scope.duration_ns,
            }
        )

    reference = max(per_device, key=lambda row: (row["busy_wall_ms"], -row["gpu"])) if per_device else None
    return {
        "phase_descriptions": KERNEL_WALL_PHASE_DESCRIPTIONS,
        "per_device": per_device,
        "reference_device": reference["gpu"] if reference is not None else None,
        "reference_phase_wall_ms": reference["phase_wall_ms"] if reference is not None else {},
        "method": (
            "Per-device disjoint interval sweep. Different simultaneously active phases are charged "
            "once to concurrent_mixed; no values are summed across devices."
        ),
    }


def _gpu_timeline_analysis(
    connection: sqlite3.Connection,
    scope: AnalysisScope,
    gpu_data: dict[int, dict[str, Any]],
) -> None:
    parts = []
    parameters: list[int] = []
    for table in (
        "CUPTI_ACTIVITY_KIND_KERNEL",
        "CUPTI_ACTIVITY_KIND_MEMCPY",
        "CUPTI_ACTIVITY_KIND_MEMSET",
    ):
        if not _table_exists(connection, table):
            continue
        parts.append(f"SELECT deviceId, start, end FROM {table} WHERE end > ? AND start < ?")
        parameters.extend((scope.start_ns, scope.end_ns))
    if not parts:
        return
    query = "SELECT deviceId, start, end FROM (" + " UNION ALL ".join(parts) + ") ORDER BY deviceId, start, end"
    states: dict[int, dict[str, int | None]] = {}
    for device_id, raw_start, raw_end in connection.execute(query, parameters):
        gpu = int(device_id)
        start = max(int(raw_start), scope.start_ns)
        end = min(int(raw_end), scope.end_ns)
        if end <= start:
            continue
        state = states.setdefault(
            gpu,
            {
                "first": None,
                "last": None,
                "current_start": None,
                "current_end": None,
                "busy": 0,
                "largest_idle": 0,
                "event_count": 0,
            },
        )
        state["event_count"] = int(state["event_count"] or 0) + 1
        if state["current_start"] is None:
            state["first"] = start
            state["last"] = end
            state["current_start"] = start
            state["current_end"] = end
            state["largest_idle"] = max(0, start - scope.start_ns)
            continue
        state["last"] = max(int(state["last"] or end), end)
        current_end = int(state["current_end"] or end)
        if start > current_end:
            state["busy"] = int(state["busy"] or 0) + current_end - int(state["current_start"] or start)
            state["largest_idle"] = max(int(state["largest_idle"] or 0), start - current_end)
            state["current_start"] = start
            state["current_end"] = end
        else:
            state["current_end"] = max(current_end, end)

    for gpu, state in states.items():
        current_start = state["current_start"]
        current_end = state["current_end"]
        if current_start is None or current_end is None:
            continue
        busy = int(state["busy"] or 0) + int(current_end) - int(current_start)
        trailing_idle = max(0, scope.end_ns - int(current_end))
        largest_idle = max(int(state["largest_idle"] or 0), trailing_idle)
        first = int(state["first"] or current_start)
        last = int(state["last"] or current_end)
        gpu_entry = gpu_data.setdefault(gpu, {"kernel_count": 0, "kernel_sum_ns": 0, "pids": set()})
        gpu_entry.update(
            {
                "gpu_event_count": int(state["event_count"] or 0),
                "busy_union_ns": busy,
                "active_span_ns": max(0, last - first),
                "largest_idle_gap_ns": largest_idle,
                "window_idle_ns": max(0, scope.duration_ns - busy),
            }
        )


def _memory_analysis(
    connection: sqlite3.Connection,
    scope: AnalysisScope,
    gpu_data: dict[int, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    if _table_exists(connection, "CUPTI_ACTIVITY_KIND_MEMCPY"):
        query = """
            SELECT copies.deviceId,
                   COALESCE(kinds.label, kinds.name, 'copyKind=' || copies.copyKind),
                   COUNT(*), SUM(copies.bytes),
                   SUM(MIN(copies.end, ?) - MAX(copies.start, ?))
            FROM CUPTI_ACTIVITY_KIND_MEMCPY AS copies
            LEFT JOIN ENUM_CUDA_MEMCPY_OPER AS kinds ON kinds.id = copies.copyKind
            WHERE copies.end > ? AND copies.start < ?
            GROUP BY copies.deviceId, copies.copyKind
            ORDER BY copies.deviceId, 5 DESC
        """
        for gpu, kind, count, byte_count, duration_ns in connection.execute(
            query,
            (scope.end_ns, scope.start_ns, scope.start_ns, scope.end_ns),
        ):
            device_id = int(gpu)
            duration = int(duration_ns or 0)
            gpu_data.setdefault(device_id, {"kernel_count": 0, "kernel_sum_ns": 0, "pids": set()})
            gpu_data[device_id]["memcpy_sum_ns"] = gpu_data[device_id].get("memcpy_sum_ns", 0) + duration
            gpu_data[device_id]["memcpy_bytes"] = gpu_data[device_id].get("memcpy_bytes", 0) + int(byte_count or 0)
            rows.append(
                {
                    "gpu": device_id,
                    "kind": kind,
                    "count": int(count),
                    "bytes_of_overlapping_ops": int(byte_count or 0),
                    "gpu_ms": duration / 1e6,
                }
            )
    if _table_exists(connection, "CUPTI_ACTIVITY_KIND_MEMSET"):
        for gpu, count, byte_count, duration_ns in connection.execute(
            """
            SELECT deviceId, COUNT(*), SUM(bytes),
                   SUM(MIN(end, ?) - MAX(start, ?))
            FROM CUPTI_ACTIVITY_KIND_MEMSET
            WHERE end > ? AND start < ?
            GROUP BY deviceId
            """,
            (scope.end_ns, scope.start_ns, scope.start_ns, scope.end_ns),
        ):
            device_id = int(gpu)
            duration = int(duration_ns or 0)
            gpu_data.setdefault(device_id, {"kernel_count": 0, "kernel_sum_ns": 0, "pids": set()})
            gpu_data[device_id]["memset_sum_ns"] = duration
            gpu_data[device_id]["memset_bytes"] = int(byte_count or 0)
            gpu_data[device_id]["memset_count"] = int(count)
    return rows


def _cuda_api_analysis(connection: sqlite3.Connection, scope: AnalysisScope) -> list[dict[str, Any]]:
    if not _table_exists(connection, "CUPTI_ACTIVITY_KIND_RUNTIME"):
        return []
    categories = {category: {"count": 0, "thread_time_ns": 0, "top": []} for category in CUDA_API_CATEGORY_DESCRIPTIONS}
    cursor = connection.execute(
        """
        SELECT strings.value, COUNT(*),
               SUM(MIN(runtime.end, ?) - MAX(runtime.start, ?)) AS duration_ns
        FROM CUPTI_ACTIVITY_KIND_RUNTIME AS runtime
        JOIN StringIds AS strings ON strings.id = runtime.nameId
        WHERE runtime.end > ? AND runtime.start < ?
        GROUP BY runtime.nameId
        ORDER BY duration_ns DESC
        """,
        (scope.end_ns, scope.start_ns, scope.start_ns, scope.end_ns),
    )
    for api_name, count, duration_ns in cursor:
        category = classify_cuda_api(str(api_name))
        categories[category]["count"] += int(count)
        categories[category]["thread_time_ns"] += int(duration_ns or 0)
        categories[category]["top"].append(
            {"name": str(api_name), "count": int(count), "thread_time_ms": int(duration_ns or 0) / 1e6}
        )
    rows = []
    for category, data in categories.items():
        rows.append(
            {
                "category": category,
                "description": CUDA_API_CATEGORY_DESCRIPTIONS[category],
                "count": data["count"],
                "summed_thread_time_ms": data["thread_time_ns"] / 1e6,
                "top_apis": data["top"][:5],
            }
        )
    rows.sort(key=lambda item: (-item["summed_thread_time_ms"], item["category"]))
    return rows


def _rdma_launch_gap_analysis(
    connection: sqlite3.Connection,
    scope: AnalysisScope,
    correlations: dict[Any, list[int]],
) -> dict[str, Any]:
    result = {
        "explicit_kernel_count": sum(len(starts) for starts in correlations.values()),
        "matched_launches": 0,
        "unmatched_launches": sum(len(starts) for starts in correlations.values()),
        "median_us": None,
        "p95_us": None,
        "max_us": None,
        "note": (
            "The CUDA launch API return-to-GPU-start gap includes stream queueing and backpressure; "
            "it is neither pure CPU dispatch overhead nor network transfer duration."
        ),
    }
    if not correlations or not _table_exists(connection, "CUPTI_ACTIVITY_KIND_RUNTIME"):
        return result
    launch_ends: dict[Any, int] = {}
    cursor = connection.execute(
        """
        SELECT runtime.end, runtime.globalTid, runtime.correlationId, strings.value
        FROM CUPTI_ACTIVITY_KIND_RUNTIME AS runtime
        JOIN StringIds AS strings ON strings.id = runtime.nameId
        WHERE runtime.correlationId IS NOT NULL
          AND runtime.end > ? AND runtime.start < ?
        """,
        (scope.start_ns, scope.end_ns),
    )
    for end, global_tid, correlation_id, api_name in cursor:
        key = (_process_key(global_tid), int(correlation_id))
        if key not in correlations or "launch" not in str(api_name).lower():
            continue
        launch_ends[key] = max(launch_ends.get(key, 0), int(end))
    gaps_us = []
    for key, starts in correlations.items():
        launch_end = launch_ends.get(key)
        if launch_end is None:
            continue
        for start in starts:
            if start >= launch_end:
                gaps_us.append((start - launch_end) / 1e3)
    result["matched_launches"] = len(gaps_us)
    result["unmatched_launches"] = result["explicit_kernel_count"] - len(gaps_us)
    if gaps_us:
        result["median_us"] = statistics.median(gaps_us)
        result["p95_us"] = _percentile_nearest_rank(gaps_us, 0.95)
        result["max_us"] = max(gaps_us)
    return result


def _summary_us(values_ns: Sequence[int]) -> dict[str, float | None]:
    values_us = [value / 1e3 for value in values_ns]
    return {
        "median_us": statistics.median(values_us) if values_us else None,
        "p95_us": _percentile_nearest_rank(values_us, 0.95),
        "max_us": max(values_us) if values_us else None,
    }


def _interval_union_ns(intervals: Sequence[tuple[int, int]]) -> int:
    if not intervals:
        return 0
    ordered = sorted(intervals)
    total = 0
    current_start, current_end = ordered[0]
    for start, end in ordered[1:]:
        if start > current_end:
            total += current_end - current_start
            current_start, current_end = start, end
        else:
            current_end = max(current_end, end)
    return total + current_end - current_start


def _merged_intervals(intervals: Sequence[tuple[int, int]]) -> list[tuple[int, int]]:
    ordered = sorted((start, end) for start, end in intervals if end > start)
    if not ordered:
        return []
    merged = [ordered[0]]
    for start, end in ordered[1:]:
        previous_start, previous_end = merged[-1]
        if start > previous_end:
            merged.append((start, end))
        else:
            merged[-1] = (previous_start, max(previous_end, end))
    return merged


def _interval_set_overlap_ns(
    left: Sequence[tuple[int, int]],
    right: Sequence[tuple[int, int]],
) -> int:
    left_merged = _merged_intervals(left)
    right_merged = _merged_intervals(right)
    left_index = 0
    right_index = 0
    total = 0
    while left_index < len(left_merged) and right_index < len(right_merged):
        left_start, left_end = left_merged[left_index]
        right_start, right_end = right_merged[right_index]
        total += max(0, min(left_end, right_end) - max(left_start, right_start))
        if left_end <= right_end:
            left_index += 1
        else:
            right_index += 1
    return total


def _infer_barriers_per_layer(
    category_rows: Sequence[dict[str, Any]],
) -> tuple[int, str]:
    vsa_calls = sum(int(row["count"]) for row in category_rows if row["category"] in {"vsa_attention", "vsa_indexing"})
    if vsa_calls:
        return (
            10,
            "explicit VSA evidence: Q/K/V/gate scatters plus O gather, each with opening and closing barriers",
        )
    return (
        8,
        "no explicit VSA kernel evidence: dense Q/K/V scatters plus O gather, each with opening and closing barriers",
    )


def _barrier_analysis(
    connection: sqlite3.Connection,
    scope: AnalysisScope,
    *,
    expected_devices: Sequence[int],
    barriers_per_layer: int = 8,
    release_cluster_tolerance_ns: int = 20_000,
) -> dict[str, Any]:
    """Analyze explicit Ulysses barriers without treating spin time as transfer.

    A barrier group is reconstructed from kernel release timestamps.  A group
    cannot contain the same device twice, and adjacent releases must be within
    ``release_cluster_tolerance_ns``. Slots are deliberately named B0..Bn
    because the trace carries no semantic exchange labels. Dense H3 has eight
    barriers per layer; the current VSA path adds a gate all-to-all and has ten.
    """
    empty = {
        "available": False,
        "barriers_per_layer": barriers_per_layer,
        "release_cluster_tolerance_us": release_cluster_tolerance_ns / 1e3,
        "observed_calls": 0,
        "observed_groups": 0,
        "full_device_groups": 0,
        "partial_groups": 0,
        "observed_layer_cycles": None,
        "per_device": [],
        "per_slot": [],
        "per_device_slot": [],
        "critical_path": None,
        "limitations": ["No explicitly named FlashInfer Ulysses barrier kernel was found."],
    }
    if not _table_exists(connection, "CUPTI_ACTIVITY_KIND_KERNEL"):
        return empty

    events: list[dict[str, Any]] = []
    cursor = connection.execute(
        """
        SELECT kernels.deviceId, kernels.start, kernels.end, strings.value
        FROM CUPTI_ACTIVITY_KIND_KERNEL AS kernels
        JOIN StringIds AS strings ON strings.id = kernels.demangledName
        WHERE kernels.end > ? AND kernels.start < ?
          AND (
              LOWER(strings.value) LIKE '%ulysses%barrier%'
              OR LOWER(strings.value) LIKE '%rdma%barrier%'
              OR LOWER(strings.value) LIKE '%flashinfer::comm%barrier%'
          )
        ORDER BY kernels.end, kernels.deviceId
        """,
        (scope.start_ns, scope.end_ns),
    )
    for device_id, start, end, name in cursor:
        if classify_kernel(str(name)) != "flashinfer_rdma_barrier":
            continue
        events.append(
            {
                "gpu": int(device_id),
                "start_ns": int(start),
                "end_ns": int(end),
                "duration_ns": _overlap_ns(start, end, scope),
                "name": str(name),
            }
        )
    if not events:
        return empty

    kernel_timeline: dict[int, list[tuple[int, int, str]]] = defaultdict(list)
    for device_id, start, end, name in connection.execute(
        """
        SELECT kernels.deviceId, kernels.start, kernels.end, strings.value
        FROM CUPTI_ACTIVITY_KIND_KERNEL AS kernels
        JOIN StringIds AS strings ON strings.id = kernels.demangledName
        WHERE kernels.end > ? AND kernels.start < ?
        ORDER BY kernels.deviceId, kernels.start
        """,
        (scope.start_ns, scope.end_ns),
    ):
        kernel_timeline[int(device_id)].append((int(start), int(end), classify_kernel(str(name))))

    groups: list[list[dict[str, Any]]] = []
    for event in events:
        if not groups:
            groups.append([event])
            continue
        previous = groups[-1]
        duplicate_device = any(item["gpu"] == event["gpu"] for item in previous)
        release_gap = event["end_ns"] - previous[-1]["end_ns"]
        if duplicate_device or release_gap > release_cluster_tolerance_ns:
            groups.append([event])
        else:
            previous.append(event)

    devices = sorted(set(expected_devices) or {event["gpu"] for event in events})
    expected_device_set = set(devices)
    full_groups: list[tuple[int, list[dict[str, Any]]]] = []
    partial_details = []
    for group_index, group in enumerate(groups):
        present = {event["gpu"] for event in group}
        if present == expected_device_set:
            full_groups.append((group_index, group))
        else:
            partial_details.append(
                {
                    "group": group_index,
                    "slot": group_index % barriers_per_layer,
                    "present_devices": sorted(present),
                    "missing_devices": sorted(expected_device_set - present),
                }
            )

    device_accumulators: dict[int, dict[str, Any]] = {
        gpu: {
            "calls": 0,
            "residency_ns": [],
            "arrival_lag_ns": [],
            "latest_arrival_groups": 0,
        }
        for gpu in devices
    }
    device_slot_accumulators: dict[tuple[int, int], dict[str, Any]] = {
        (gpu, slot): {
            "calls": 0,
            "residency_ns": [],
            "arrival_lag_ns": [],
            "latest_arrival_groups": 0,
        }
        for gpu in devices
        for slot in range(barriers_per_layer)
    }
    for group_index, group in enumerate(groups):
        slot = group_index % barriers_per_layer
        for event in group:
            for accumulator in (
                device_accumulators[event["gpu"]],
                device_slot_accumulators[(event["gpu"], slot)],
            ):
                accumulator["calls"] += 1
                accumulator["residency_ns"].append(event["duration_ns"])

    slot_accumulators: dict[int, dict[str, Any]] = {
        slot: {
            "groups": 0,
            "full_groups": 0,
            "arrival_skew_ns": [],
            "release_skew_ns": [],
            "latest_by_gpu": defaultdict(int),
            "late_path_kernel_ns": defaultdict(int),
            "late_path_covered_union_ns": 0,
        }
        for slot in range(barriers_per_layer)
    }
    for group_index, group in enumerate(groups):
        slot_accumulators[group_index % barriers_per_layer]["groups"] += 1
    for group_index, group in full_groups:
        slot = group_index % barriers_per_layer
        starts = [event["start_ns"] for event in group]
        ends = [event["end_ns"] for event in group]
        earliest = min(starts)
        latest_event = max(group, key=lambda event: (event["start_ns"], event["gpu"]))
        slot_data = slot_accumulators[slot]
        slot_data["full_groups"] += 1
        slot_data["arrival_skew_ns"].append(max(starts) - earliest)
        slot_data["release_skew_ns"].append(max(ends) - min(ends))
        slot_data["latest_by_gpu"][latest_event["gpu"]] += 1
        device_accumulators[latest_event["gpu"]]["latest_arrival_groups"] += 1
        device_slot_accumulators[(latest_event["gpu"], slot)]["latest_arrival_groups"] += 1
        for event in group:
            lag = event["start_ns"] - earliest
            device_accumulators[event["gpu"]]["arrival_lag_ns"].append(lag)
            device_slot_accumulators[(event["gpu"], slot)]["arrival_lag_ns"].append(lag)
        late_path_intervals = []
        for kernel_start, kernel_end, category in kernel_timeline[latest_event["gpu"]]:
            overlap_start = max(kernel_start, earliest)
            overlap_end = min(kernel_end, latest_event["start_ns"])
            if overlap_end <= overlap_start:
                continue
            slot_data["late_path_kernel_ns"][category] += overlap_end - overlap_start
            late_path_intervals.append((overlap_start, overlap_end))
        slot_data["late_path_covered_union_ns"] += _interval_union_ns(late_path_intervals)

    per_device = []
    for gpu in devices:
        data = device_accumulators[gpu]
        residency = _summary_us(data["residency_ns"])
        arrival = _summary_us(data["arrival_lag_ns"])
        per_device.append(
            {
                "gpu": gpu,
                "calls": data["calls"],
                "summed_residency_ms": sum(data["residency_ns"]) / 1e6,
                "median_residency_us": residency["median_us"],
                "p95_residency_us": residency["p95_us"],
                "max_residency_us": residency["max_us"],
                "median_arrival_lag_us": arrival["median_us"],
                "p95_arrival_lag_us": arrival["p95_us"],
                "latest_arrival_groups": data["latest_arrival_groups"],
                "latest_arrival_percent": (
                    100.0 * data["latest_arrival_groups"] / len(full_groups) if full_groups else None
                ),
            }
        )

    per_slot = []
    for slot in range(barriers_per_layer):
        data = slot_accumulators[slot]
        arrival = _summary_us(data["arrival_skew_ns"])
        release = _summary_us(data["release_skew_ns"])
        latest_counts = dict(sorted(data["latest_by_gpu"].items()))
        dominant_gpu = max(latest_counts, key=lambda gpu: (latest_counts[gpu], -gpu)) if latest_counts else None
        per_slot.append(
            {
                "slot": slot,
                "groups": data["groups"],
                "full_groups": data["full_groups"],
                "arrival_skew_sum_ms": sum(data["arrival_skew_ns"]) / 1e6,
                "median_arrival_skew_us": arrival["median_us"],
                "p95_arrival_skew_us": arrival["p95_us"],
                "max_arrival_skew_us": arrival["max_us"],
                "median_release_skew_us": release["median_us"],
                "p95_release_skew_us": release["p95_us"],
                "max_release_skew_us": release["max_us"],
                "dominant_latest_gpu": dominant_gpu,
                "dominant_latest_count": latest_counts.get(dominant_gpu, 0),
                "latest_arrival_counts": {str(gpu): count for gpu, count in latest_counts.items()},
                "late_path_kernel_gpu_ms": {
                    category: duration_ns / 1e6
                    for category, duration_ns in sorted(
                        data["late_path_kernel_ns"].items(),
                        key=lambda item: (-item[1], item[0]),
                    )
                },
                "late_path_kernel_covered_union_ms": (data["late_path_covered_union_ns"] / 1e6),
                "uncovered_arrival_skew_ms": max(
                    0,
                    (sum(data["arrival_skew_ns"]) - data["late_path_covered_union_ns"]) / 1e6,
                ),
            }
        )

    per_device_slot = []
    for slot in range(barriers_per_layer):
        for gpu in devices:
            data = device_slot_accumulators[(gpu, slot)]
            residency = _summary_us(data["residency_ns"])
            arrival = _summary_us(data["arrival_lag_ns"])
            per_device_slot.append(
                {
                    "slot": slot,
                    "gpu": gpu,
                    "calls": data["calls"],
                    "full_group_observations": len(data["arrival_lag_ns"]),
                    "summed_residency_ms": sum(data["residency_ns"]) / 1e6,
                    "median_residency_us": residency["median_us"],
                    "p95_residency_us": residency["p95_us"],
                    "max_residency_us": residency["max_us"],
                    "median_arrival_lag_us": arrival["median_us"],
                    "p95_arrival_lag_us": arrival["p95_us"],
                    "latest_arrival_groups": data["latest_arrival_groups"],
                }
            )

    latest_totals = {row["gpu"]: row["latest_arrival_groups"] for row in per_device}
    dominant_gpu = max(latest_totals, key=lambda gpu: (latest_totals[gpu], -gpu)) if full_groups else None
    skew_sum_ns = sum(value for data in slot_accumulators.values() for value in data["arrival_skew_ns"])
    critical_kernel_ns: dict[str, int] = defaultdict(int)
    critical_kernel_covered_ns = 0
    for data in slot_accumulators.values():
        critical_kernel_covered_ns += data["late_path_covered_union_ns"]
        for category, duration_ns in data["late_path_kernel_ns"].items():
            critical_kernel_ns[category] += duration_ns
    limitations = [
        f"Barrier slots B0..B{barriers_per_layer - 1} are chronological "
        f"modulo-{barriers_per_layer} positions; the trace has no semantic exchange labels.",
        "Barrier kernel residency is synchronization/spin time, not RDMA byte-transfer time.",
        "The latest-arriving device constrains that barrier release, but the trace alone does not "
        "identify why its preceding work was late.",
        "Late-path kernel evidence is overlap on the latest-arriving device during the arrival-skew "
        "window; uncovered time may be CPU, transport, DMA, or untraced work.",
    ]
    if partial_details:
        limitations.append(
            f"{len(partial_details)} boundary/partial group(s) omit devices, commonly because "
            "NVTX-triggered capture began mid-kernel."
        )
    observed_layer_cycles = len(groups) // barriers_per_layer if len(groups) % barriers_per_layer == 0 else None
    return {
        "available": True,
        "barriers_per_layer": barriers_per_layer,
        "release_cluster_tolerance_us": release_cluster_tolerance_ns / 1e3,
        "devices": devices,
        "observed_calls": len(events),
        "observed_groups": len(groups),
        "full_device_groups": len(full_groups),
        "partial_groups": len(partial_details),
        "partial_group_details": partial_details,
        "observed_layer_cycles": observed_layer_cycles,
        "per_device": per_device,
        "per_slot": per_slot,
        "per_device_slot": per_device_slot,
        "critical_path": {
            "arrival_skew_sum_ms": skew_sum_ns / 1e6,
            "arrival_skew_percent_of_scope_wall": (
                100.0 * skew_sum_ns / scope.duration_ns if scope.duration_ns else None
            ),
            "dominant_latest_gpu": dominant_gpu,
            "dominant_latest_groups": latest_totals.get(dominant_gpu, 0),
            "dominant_latest_percent": (
                100.0 * latest_totals[dominant_gpu] / len(full_groups)
                if dominant_gpu is not None and full_groups
                else None
            ),
            "late_path_kernel_gpu_ms": {
                category: duration_ns / 1e6
                for category, duration_ns in sorted(critical_kernel_ns.items(), key=lambda item: (-item[1], item[0]))
            },
            "late_path_kernel_covered_union_ms": critical_kernel_covered_ns / 1e6,
            "uncovered_arrival_skew_ms": max(0, (skew_sum_ns - critical_kernel_covered_ns) / 1e6),
            "interpretation": (
                "Arrival-skew sums cover sequential full-device barriers and are already inside step wall time; "
                "they indicate load-imbalance exposure, not additional time or guaranteed removable time."
            ),
        },
        "limitations": limitations,
    }


def _nvtx_communication_ranges(connection: sqlite3.Connection, scope: AnalysisScope) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], dict[str, int]] = defaultdict(lambda: {"count": 0, "duration_ns": 0})
    for item in _nvtx_ranges(connection):
        duration = _overlap_ns(item["start_ns"], item["end_ns"], scope)
        if duration <= 0:
            continue
        lowered = item["name"].lower()
        if "nccl" in lowered:
            evidence = "nccl"
        elif any(token in lowered for token in _RDMA_TOKENS):
            evidence = "flashinfer_rdma"
        else:
            continue
        data = grouped[(evidence, item["name"])]
        data["count"] += 1
        data["duration_ns"] += duration
    return [
        {
            "evidence": evidence,
            "name": name,
            "count": data["count"],
            "summed_cpu_range_ms": data["duration_ns"] / 1e6,
        }
        for (evidence, name), data in sorted(grouped.items(), key=lambda item: (-item[1]["duration_ns"], item[0]))
    ]


_DECODE_STAGE_PREFIX = "minimax_h3.decode."
_DECODE_REQUEST_PREFIX = f"{_DECODE_STAGE_PREFIX}request_"


def _decode_nvtx_stage_analysis(
    connection: sqlite3.Connection,
    scope: AnalysisScope,
    *,
    server_log: Path | None,
) -> dict[str, Any]:
    """Summarize explicit cross-rank decode NVTX ranges on one wall clock.

    These are CPU push/pop ranges. They can establish that rank-0 video and
    rank-1 audio work overlap, but they do not assign enclosed CUDA kernels to
    a stage or measure codec CPU utilization.
    """
    pid_map: dict[int, dict[str, Any]] = {}
    if server_log is not None:
        pid_map, _ = _parse_server_nccl_device_map(server_log)

    ranges: list[dict[str, Any]] = []
    stage_intervals: dict[str, list[tuple[int, int]]] = defaultdict(list)
    stage_summed_ns: dict[str, int] = defaultdict(int)
    stage_ranks: dict[str, set[int]] = defaultdict(set)
    stage_pids: dict[str, set[int]] = defaultdict(set)
    for item in _nvtx_ranges(connection):
        name = item["name"]
        if not name.startswith(_DECODE_STAGE_PREFIX) or name.startswith(_DECODE_REQUEST_PREFIX):
            continue
        start = max(int(item["start_ns"]), scope.start_ns)
        end = min(int(item["end_ns"]), scope.end_ns)
        if end <= start:
            continue
        stage = name.removeprefix(_DECODE_STAGE_PREFIX)
        pid = item["pid"]
        rank = pid_map.get(pid, {}).get("rank") if pid is not None else None
        duration_ns = end - start
        stage_intervals[stage].append((start, end))
        stage_summed_ns[stage] += duration_ns
        if pid is not None:
            stage_pids[stage].add(pid)
        if rank is not None:
            stage_ranks[stage].add(int(rank))
        ranges.append(
            {
                "stage": stage,
                "name": name,
                "pid": pid,
                "rank": rank,
                "start_offset_ms": (start - scope.start_ns) / 1e6,
                "end_offset_ms": (end - scope.start_ns) / 1e6,
                "wall_ms": duration_ns / 1e6,
                "clipped_to_scope": start != item["start_ns"] or end != item["end_ns"],
            }
        )

    per_stage = []
    for stage in sorted(stage_intervals):
        per_stage.append(
            {
                "stage": stage,
                "count": len(stage_intervals[stage]),
                "ranks": sorted(stage_ranks[stage]),
                "pids": sorted(stage_pids[stage]),
                "wall_union_ms": _interval_union_ns(stage_intervals[stage]) / 1e6,
                "summed_cpu_range_ms": stage_summed_ns[stage] / 1e6,
            }
        )

    def overlap_row(label: str, left_stage: str, right_stages: Sequence[str]) -> dict[str, Any]:
        left = stage_intervals.get(left_stage, [])
        right = [interval for stage in right_stages for interval in stage_intervals.get(stage, [])]
        left_ns = _interval_union_ns(left)
        right_ns = _interval_union_ns(right)
        overlap_ns = _interval_set_overlap_ns(left, right)
        return {
            "comparison": label,
            "left_stage": left_stage,
            "right_stages": list(right_stages),
            "available": bool(left and right),
            "left_union_ms": left_ns / 1e6,
            "right_union_ms": right_ns / 1e6,
            "overlap_ms": overlap_ns / 1e6,
            "right_hidden_percent": 100.0 * overlap_ns / right_ns if right_ns else None,
            "left_overlap_percent": 100.0 * overlap_ns / left_ns if left_ns else None,
        }

    return {
        "available": bool(ranges),
        "ranges": sorted(ranges, key=lambda row: (row["start_offset_ms"], row["pid"] or -1, row["stage"])),
        "per_stage": per_stage,
        "overlaps": [
            overlap_row("TAEH3 video vs Audio VAE", "taeh3_video", ["audio_vae"]),
            overlap_row("TAEH3 video vs AAC encode", "taeh3_video", ["audio_aac_encode"]),
            overlap_row(
                "TAEH3 video vs rank-1 audio pipeline",
                "taeh3_video",
                ["audio_vae", "audio_aac_encode"],
            ),
        ],
        "rank_mapping": (
            "NVTX PID -> server NCCL rank mapping"
            if server_log is not None and pid_map
            else "unresolved; provide --server-log with NCCL startup mappings"
        ),
        "limitations": [
            "Stage durations are CPU NVTX wall ranges, not summed CPU utilization or GPU kernel attribution.",
            "Ranges on different ranks share a timestamp axis; per-stage summed time can double-count overlap.",
            "The hidden percentages describe observed interval overlap, not guaranteed removable latency.",
        ],
    }


def _osrt_top_calls(connection: sqlite3.Connection, scope: AnalysisScope, top: int) -> list[dict[str, Any]]:
    if not _table_exists(connection, "OSRT_API"):
        return []
    rows = []
    cursor = connection.execute(
        """
        SELECT strings.value, COUNT(*),
               SUM(MIN(calls.end, ?) - MAX(calls.start, ?)) AS duration_ns
        FROM OSRT_API AS calls
        JOIN StringIds AS strings ON strings.id = calls.nameId
        WHERE calls.end > ? AND calls.start < ?
        GROUP BY calls.nameId
        ORDER BY duration_ns DESC
        LIMIT ?
        """,
        (scope.end_ns, scope.start_ns, scope.start_ns, scope.end_ns, top),
    )
    for name, count, duration_ns in cursor:
        rows.append(
            {
                "name": str(name),
                "count": int(count),
                "summed_thread_time_ms": int(duration_ns or 0) / 1e6,
            }
        )
    return rows


def analyze_sqlite(
    sqlite_path: Path,
    *,
    requested_range: str = "auto",
    requested_domain: str | None = None,
    top: int = 20,
    server_log: Path | None = None,
    barriers_per_layer: int | None = None,
) -> dict[str, Any]:
    connection = _open_read_only(sqlite_path)
    try:
        scope, warnings = select_scope(
            connection,
            requested_range=requested_range,
            requested_domain=requested_domain,
        )
        category_rows, top_kernels, gpu_data, rdma_correlations = _kernel_analysis(connection, scope, top=top)
        memory_rows = _memory_analysis(connection, scope, gpu_data)
        _gpu_timeline_analysis(connection, scope, gpu_data)
        kernel_phase_wall = _kernel_phase_wall_analysis(
            connection,
            scope,
            expected_devices=sorted(gpu_data),
        )
        if barriers_per_layer is None:
            selected_barriers_per_layer, barrier_layout_basis = _infer_barriers_per_layer(category_rows)
        else:
            if barriers_per_layer < 1:
                raise ValueError("barriers_per_layer must be positive")
            selected_barriers_per_layer = barriers_per_layer
            barrier_layout_basis = "explicit analyzer override"
        barrier_analysis = _barrier_analysis(
            connection,
            scope,
            expected_devices=sorted(gpu_data),
            barriers_per_layer=selected_barriers_per_layer,
        )
        barrier_analysis["layout_basis"] = barrier_layout_basis
        metadata = _gpu_metadata(connection)
        identities, identity_warnings = _resolve_gpu_identities(
            gpu_data,
            metadata,
            server_log=server_log,
        )
        warnings.extend(identity_warnings)
        barrier_per_gpu_ms = next(
            (item["per_gpu_ms"] for item in category_rows if item["category"] == "flashinfer_rdma_barrier"),
            {},
        )
        gpu_rows = []
        for gpu in sorted(gpu_data):
            data = gpu_data[gpu]
            identity = identities[gpu]
            kernel_sum_ms = data.get("kernel_sum_ns", 0) / 1e6
            barrier_kernel_ms = float(barrier_per_gpu_ms.get(str(gpu), 0.0))
            gpu_rows.append(
                {
                    "gpu": gpu,
                    "rank": identity.get("rank"),
                    "cuda_device": identity.get("cuda_device"),
                    "physical_gpu": identity.get("physical_gpu"),
                    "bus_location": identity.get("bus_location"),
                    "uuid": identity.get("uuid"),
                    "name": identity.get("name"),
                    "mapping_source": identity.get("mapping_source"),
                    "pids": sorted(data.get("pids", set())),
                    "kernel_count": data.get("kernel_count", 0),
                    "kernel_sum_ms": kernel_sum_ms,
                    "barrier_kernel_ms": barrier_kernel_ms,
                    "non_barrier_kernel_sum_ms": kernel_sum_ms - barrier_kernel_ms,
                    "memcpy_sum_ms": data.get("memcpy_sum_ns", 0) / 1e6,
                    "memcpy_mib": data.get("memcpy_bytes", 0) / (1024 * 1024),
                    "memset_sum_ms": data.get("memset_sum_ns", 0) / 1e6,
                    "busy_union_ms": data.get("busy_union_ns", 0) / 1e6,
                    "active_span_ms": data.get("active_span_ns", 0) / 1e6,
                    "window_idle_ms": data.get("window_idle_ns", scope.duration_ns) / 1e6,
                    "largest_idle_gap_ms": data.get("largest_idle_gap_ns", scope.duration_ns) / 1e6,
                }
            )
        active_spans = [item["active_span_ms"] for item in gpu_rows]
        slowest = max(gpu_rows, key=lambda item: item["active_span_ms"]) if gpu_rows else None
        compute_slowest = max(gpu_rows, key=lambda item: item["non_barrier_kernel_sum_ms"]) if gpu_rows else None
        unknown_row = next((item for item in category_rows if item["category"] == "unknown"), None)
        if unknown_row and unknown_row["total_gpu_ms"] > 0:
            warnings.append(
                f"{unknown_row['percent_of_kernel_gpu_time']:.2f}% of summed kernel GPU-time is "
                "unknown; it was not force-fit into a named H3 stage."
            )
        semantic_gemm_ms = sum(
            item["total_gpu_ms"]
            for item in category_rows
            if item["category"] in {"gemm_fc1", "gemm_fc2", "gemm_qkv", "gemm_out"}
        )
        generic_gemm_ms = next(
            (item["total_gpu_ms"] for item in category_rows if item["category"] == "gemm_unknown"),
            0.0,
        )
        if generic_gemm_ms > 0 and semantic_gemm_ms == 0:
            warnings.append(
                "GEMM kernel names do not expose FC1/FC2/QKV/out roles; all recognizable GEMMs remain gemm_unknown."
            )
        barrier_row = next(
            (item for item in category_rows if item["category"] == "flashinfer_rdma_barrier"),
            None,
        )
        if barrier_row and barrier_row["total_gpu_ms"] > 0:
            warnings.append(
                "FlashInfer Ulysses barrier GPU-time is synchronization/spin residency; it must "
                "not be read as RDMA wire time."
            )
        return {
            "sqlite": str(sqlite_path),
            "scope": {
                "kind": scope.kind,
                "name": scope.name,
                "domain": scope.domain,
                "pid": scope.pid,
                "start_ns": scope.start_ns,
                "end_ns": scope.end_ns,
                "wall_ms": scope.duration_ns / 1e6,
            },
            "per_gpu": gpu_rows,
            "per_gpu_active_span_summary": {
                "gpu_count": len(gpu_rows),
                "median_ms": statistics.median(active_spans) if active_spans else None,
                "slowest_gpu": slowest["gpu"] if slowest else None,
                "slowest_rank": slowest["rank"] if slowest else None,
                "slowest_physical_gpu": slowest["physical_gpu"] if slowest else None,
                "slowest_ms": slowest["active_span_ms"] if slowest else None,
                "largest_non_barrier_kernel_gpu": compute_slowest["gpu"] if compute_slowest else None,
                "largest_non_barrier_kernel_rank": compute_slowest["rank"] if compute_slowest else None,
                "largest_non_barrier_kernel_physical_gpu": (
                    compute_slowest["physical_gpu"] if compute_slowest else None
                ),
                "largest_non_barrier_kernel_ms": (
                    compute_slowest["non_barrier_kernel_sum_ms"] if compute_slowest else None
                ),
                "metric": "first-to-last GPU event span inside the selected CPU NVTX window",
            },
            "device_mapping": {
                "server_log": str(server_log) if server_log is not None else None,
                "method": (
                    "kernel PID -> server NCCL rank/cudaDev/nvmlDev/busId"
                    if server_log is not None
                    else "unresolved (TARGET_INFO_GPU.id is not treated as CUPTI deviceId)"
                ),
            },
            "kernel_categories": category_rows,
            "kernel_phase_wall": kernel_phase_wall,
            "top_kernels": top_kernels,
            "memory_copies": memory_rows,
            "cuda_api_categories": _cuda_api_analysis(connection, scope),
            "flashinfer_rdma_launch_gaps": _rdma_launch_gap_analysis(connection, scope, rdma_correlations),
            "flashinfer_ulysses_barriers": barrier_analysis,
            "decode_nvtx_stages": _decode_nvtx_stage_analysis(
                connection,
                scope,
                server_log=server_log,
            ),
            "communication_nvtx_ranges": _nvtx_communication_ranges(connection, scope),
            "top_os_runtime_calls": _osrt_top_calls(connection, scope, top=min(top, 20)),
            "warnings": warnings,
            "methodology": {
                "kernel_totals": (
                    "Summed event duration across GPUs/streams; concurrent work double-counts and is not wall time."
                ),
                "busy_union": "Union of kernel, memcpy, and memset intervals independently for each GPU.",
                "phase_wall": (
                    "A disjoint interval sweep per GPU. Same-phase overlap is counted once; "
                    "cross-phase overlap is counted once as concurrent_mixed; devices are never summed."
                ),
                "api_totals": (
                    "Summed CPU thread duration; calls on different ranks/threads and nested ranges can overlap."
                ),
                "classification": (
                    "Kernel/API names only. No temporal inference is used to assign ambiguous GEMMs or stalls."
                ),
                "rdma": (
                    "Only explicit Ulysses PCIe/RDMA names are attributed. Generic ioctl/poll/CUDA waits are not RDMA."
                ),
                "capture_range": (
                    "An NVTX-triggered capture can clip kernels already in flight on non-trigger "
                    "ranks and can perturb the trigger rank; compare absolute step time with an "
                    "unprofiled run."
                ),
                "device_identity": (
                    "CUPTI device IDs are trace-local. Physical GPU and PCI bus identity is joined "
                    "through each kernel's PID and NCCL startup mapping; TARGET_INFO_GPU.id is not "
                    "directly joined after CUDA_VISIBLE_DEVICES reordering."
                ),
                "decode_nvtx": (
                    "Explicit CPU NVTX intervals only. Cross-rank overlap is computed on the shared trace clock; "
                    "it is not CUDA-kernel attribution or a claim of removable latency."
                ),
            },
        }
    finally:
        connection.close()


def run_canonical_nsys_stats(
    sqlite_path: Path,
    scope: dict[str, Any],
    *,
    nsys: str,
    timeout: int,
) -> tuple[dict[str, Any], str]:
    command = [
        nsys,
        "stats",
        "--report",
        "nvtx_pushpop_sum,cuda_gpu_kern_sum,cuda_gpu_mem_time_sum,cuda_api_sum",
        "--filter-time",
        f"{scope['start_ns']}ns/{scope['end_ns']}ns",
        str(sqlite_path),
    ]
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    raw = completed.stdout
    if completed.stderr:
        raw += ("\n" if raw else "") + completed.stderr
    expected_headers = {
        "nvtx": "NVTX Push/Pop Range Summary" in raw,
        "kernels": "CUDA GPU Kernel Summary" in raw,
        "memory": "CUDA GPU MemOps Summary" in raw,
        "cuda_api": "CUDA API Summary" in raw,
    }
    return (
        {
            "command": command,
            "returncode": completed.returncode,
            "output_sha256": hashlib.sha256(raw.encode("utf-8")).hexdigest(),
            "output_lines": len(raw.splitlines()),
            "reports_detected": expected_headers,
        },
        raw,
    )


def _fmt(value: float | None, digits: int = 3) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def _escape_markdown(value: Any, limit: int | None = None) -> str:
    text = str(value).replace("|", "\\|").replace("\n", " ")
    if limit is not None and len(text) > limit:
        return text[: max(0, limit - 1)] + "…"
    return text


def render_markdown(result: dict[str, Any], *, source: Path) -> str:
    scope = result["scope"]
    summary = result["per_gpu_active_span_summary"]
    lines = [
        "# MiniMax-H3 Nsight analysis",
        "",
        f"- Source: `{source}`",
        f"- SQLite: `{result['sqlite']}`",
        f"- Scope: `{scope['kind']}` / `{scope['name']}`" + (f" @ `{scope['domain']}`" if scope["domain"] else ""),
        f"- Window wall time: **{scope['wall_ms']:.3f} ms**",
        f"- GPU active-span median: **{_fmt(summary['median_ms'])} ms**",
        f"- Slowest trace device by active span: **device {summary['slowest_gpu']}"
        + (f" / rank {summary['slowest_rank']}" if summary.get("slowest_rank") is not None else "")
        + (
            f" / physical GPU {summary['slowest_physical_gpu']}"
            if summary.get("slowest_physical_gpu") is not None
            else ""
        )
        + f" / {_fmt(summary['slowest_ms'])} ms**",
        f"- Largest non-barrier kernel sum: **trace device "
        f"{summary['largest_non_barrier_kernel_gpu']}"
        + (
            f" / rank {summary['largest_non_barrier_kernel_rank']}"
            if summary.get("largest_non_barrier_kernel_rank") is not None
            else ""
        )
        + (
            f" / physical GPU {summary['largest_non_barrier_kernel_physical_gpu']}"
            if summary.get("largest_non_barrier_kernel_physical_gpu") is not None
            else ""
        )
        + f" / {_fmt(summary['largest_non_barrier_kernel_ms'])} ms**",
        f"- Device identity: **{_escape_markdown(result['device_mapping']['method'])}**",
        "",
        "> GPU-time below is summed across devices and streams; it is not additive wall time.",
        "",
    ]

    decode_stages = result["decode_nvtx_stages"]
    lines.extend(["## Cross-rank decode NVTX timeline", ""])
    if not decode_stages["available"]:
        lines.append("No explicit nested `minimax_h3.decode.*` stage range was found in scope.")
    else:
        lines.extend(
            [
                f"Rank mapping: {_escape_markdown(decode_stages['rank_mapping'])}.",
                "",
                "| Stage | Rank | PID | Start / end offset ms | Wall ms | Clipped |",
                "|---|---:|---:|---:|---:|---|",
            ]
        )
        for row in decode_stages["ranges"]:
            lines.append(
                f"| `{_escape_markdown(row['stage'])}` | "
                f"{row['rank'] if row['rank'] is not None else 'n/a'} | "
                f"{row['pid'] if row['pid'] is not None else 'n/a'} | "
                f"{row['start_offset_ms']:.3f} / {row['end_offset_ms']:.3f} | "
                f"{row['wall_ms']:.3f} | {'yes' if row['clipped_to_scope'] else 'no'} |"
            )
        lines.extend(
            [
                "",
                "| Overlap comparison | Video ms | Audio ms | Overlap ms | Audio hidden | Video overlap |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for row in decode_stages["overlaps"]:
            if not row["available"]:
                lines.append(f"| {_escape_markdown(row['comparison'])} | n/a | n/a | n/a | n/a | n/a |")
                continue
            lines.append(
                f"| {_escape_markdown(row['comparison'])} | {row['left_union_ms']:.3f} | "
                f"{row['right_union_ms']:.3f} | {row['overlap_ms']:.3f} | "
                f"{_fmt(row['right_hidden_percent'], 2)}% | {_fmt(row['left_overlap_percent'], 2)}% |"
            )
        lines.extend(["", "Attribution limits:", ""])
        for limitation in decode_stages["limitations"]:
            lines.append(f"- {limitation}")

    lines.extend(
        [
            "",
            "## Per-GPU timeline",
            "",
            (
                "| Trace dev | Rank | Physical GPU | PCI bus | PID(s) | Active span ms | Busy union ms | "
                "Kernel-sum ms | Non-barrier / barrier kernel ms | Memcpy ms / MiB | Window idle ms | "
                "Largest idle gap ms |"
            ),
            "|---:|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in result["per_gpu"]:
        lines.append(
            f"| {row['gpu']} | {row['rank'] if row['rank'] is not None else 'n/a'} | "
            f"{row['physical_gpu'] if row['physical_gpu'] is not None else 'n/a'} | "
            f"{_escape_markdown(row['bus_location'] or 'n/a')} | "
            f"{','.join(map(str, row['pids'])) or 'n/a'} | {row['active_span_ms']:.3f} | "
            f"{row['busy_union_ms']:.3f} | {row['kernel_sum_ms']:.3f} | "
            f"{row['non_barrier_kernel_sum_ms']:.3f} / {row['barrier_kernel_ms']:.3f} | "
            f"{row['memcpy_sum_ms']:.3f} / {row['memcpy_mib']:.2f} | "
            f"{row['window_idle_ms']:.3f} | {row['largest_idle_gap_ms']:.3f} |"
        )

    phase_wall = result["kernel_phase_wall"]
    lines.extend(
        [
            "",
            "## Per-device wall decomposition (no double-counting)",
            "",
            f"{phase_wall['method']} Reference device (largest traced busy union): {phase_wall['reference_device']}.",
            "",
            (
                "| Trace dev | Linear GEMM ms | VSA ms | RDMA barrier ms | RDMA other ms | Pointwise ms | "
                "Other single-phase ms | Mixed-concurrent ms | Idle ms | Account error ns |"
            ),
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    requested_phases = {
        "linear_gemm",
        "vsa",
        "rdma_barrier",
        "rdma_other",
        "pointwise",
        "concurrent_mixed",
        "idle",
    }
    for row in phase_wall["per_device"]:
        phases = row["phase_wall_ms"]
        other_single_phase_ms = sum(value for phase, value in phases.items() if phase not in requested_phases)
        lines.append(
            f"| {row['gpu']} | {phases['linear_gemm']:.3f} | {phases['vsa']:.3f} | "
            f"{phases['rdma_barrier']:.3f} | {phases['rdma_other']:.3f} | "
            f"{phases['pointwise']:.3f} | {other_single_phase_ms:.3f} | "
            f"{phases['concurrent_mixed']:.3f} | {phases['idle']:.3f} | "
            f"{row['accounting_error_ns']} |"
        )

    lines.extend(
        [
            "",
            "## Kernel categories",
            "",
            (
                "| Category | Calls | Total GPU-ms | % kernel GPU-time | Median GPU-ms | "
                "Slowest trace dev / ms | Evidence rule |"
            ),
            "|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in result["kernel_categories"]:
        lines.append(
            f"| `{row['category']}` | {row['count']} | {row['total_gpu_ms']:.3f} | "
            f"{row['percent_of_kernel_gpu_time']:.2f}% | {_fmt(row['median_gpu_ms'])} | "
            f"{row['slowest_gpu']} / {_fmt(row['slowest_gpu_ms'])} | "
            f"{_escape_markdown(row['description'])} |"
        )

    barriers = result["flashinfer_ulysses_barriers"]
    lines.extend(["", "## FlashInfer Ulysses barrier/rank-skew analysis", ""])
    if not barriers["available"]:
        lines.append("No explicitly named Ulysses barrier kernel was found in scope.")
    else:
        critical = barriers["critical_path"]
        layer_cycles = barriers["observed_layer_cycles"]
        critical_kernel_evidence = (
            ", ".join(
                f"{category}={duration_ms:.3f} GPU-ms"
                for category, duration_ms in critical["late_path_kernel_gpu_ms"].items()
            )
            or "none"
        )
        lines.extend(
            [
                f"- Observed {barriers['observed_calls']} calls reconstructed as "
                f"{barriers['observed_groups']} chronological groups: "
                f"{barriers['full_device_groups']} full-device and {barriers['partial_groups']} partial.",
                f"- Layout: {barriers['barriers_per_layer']} barrier slots/layer "
                f"(B0..B{barriers['barriers_per_layer'] - 1}); "
                + (
                    f"the group count is consistent with {layer_cycles} layer cycles."
                    if layer_cycles is not None
                    else "the captured group count is not an exact whole-layer multiple."
                ),
                f"- Layout basis: {barriers['layout_basis']}.",
                f"- Full-group arrival-skew sum: {critical['arrival_skew_sum_ms']:.3f} ms "
                f"({critical['arrival_skew_percent_of_scope_wall']:.2f}% of the selected wall window).",
                f"- Dominant latest arrival: trace device {critical['dominant_latest_gpu']} in "
                f"{critical['dominant_latest_groups']}/{barriers['full_device_groups']} groups "
                f"({critical['dominant_latest_percent']:.2f}%).",
                f"- Latest-device work overlapping the skew windows: {critical_kernel_evidence}; "
                f"covered union {critical['late_path_kernel_covered_union_ms']:.3f} ms, "
                f"uncovered {critical['uncovered_arrival_skew_ms']:.3f} ms.",
                f"- {critical['interpretation']}",
                "",
                "### Per-device barrier residency and arrival lag",
                "",
                (
                    "| Trace dev | Calls | Residency sum ms | Residency median / p95 / max µs | "
                    "Arrival-lag median / p95 µs | Latest arrival groups / % |"
                ),
                "|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in barriers["per_device"]:
            lines.append(
                f"| {row['gpu']} | {row['calls']} | {row['summed_residency_ms']:.3f} | "
                f"{_fmt(row['median_residency_us'])} / {_fmt(row['p95_residency_us'])} / "
                f"{_fmt(row['max_residency_us'])} | "
                f"{_fmt(row['median_arrival_lag_us'])} / {_fmt(row['p95_arrival_lag_us'])} | "
                f"{row['latest_arrival_groups']} / {_fmt(row['latest_arrival_percent'], 2)}% |"
            )
        lines.extend(
            [
                "",
                "### Per-layer chronological barrier slots",
                "",
                (
                    "| Slot | Groups (full) | Arrival-skew sum ms | Arrival skew median / p95 / max µs | "
                    "Release skew median / p95 / max µs | Dominant latest GPU / count | "
                    "Late-device kernel evidence (GPU-ms) |"
                ),
                "|---:|---:|---:|---:|---:|---:|---|",
            ]
        )
        for row in barriers["per_slot"]:
            slot_kernel_evidence = (
                ", ".join(
                    f"{category}={duration_ms:.3f}" for category, duration_ms in row["late_path_kernel_gpu_ms"].items()
                )
                or f"uncovered={row['uncovered_arrival_skew_ms']:.3f}"
            )
            lines.append(
                f"| B{row['slot']} | {row['groups']} ({row['full_groups']}) | "
                f"{row['arrival_skew_sum_ms']:.3f} | "
                f"{_fmt(row['median_arrival_skew_us'])} / {_fmt(row['p95_arrival_skew_us'])} / "
                f"{_fmt(row['max_arrival_skew_us'])} | "
                f"{_fmt(row['median_release_skew_us'])} / {_fmt(row['p95_release_skew_us'])} / "
                f"{_fmt(row['max_release_skew_us'])} | "
                f"{row['dominant_latest_gpu']} / {row['dominant_latest_count']} | "
                f"{_escape_markdown(slot_kernel_evidence)} |"
            )
        lines.extend(
            [
                "",
                "### Per-device × barrier slot",
                "",
                (
                    "| Slot | Trace dev | Calls | Residency sum ms | Residency median / p95 µs | "
                    "Arrival-lag median / p95 µs | Latest groups |"
                ),
                "|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in barriers["per_device_slot"]:
            lines.append(
                f"| B{row['slot']} | {row['gpu']} | {row['calls']} | "
                f"{row['summed_residency_ms']:.3f} | "
                f"{_fmt(row['median_residency_us'])} / {_fmt(row['p95_residency_us'])} | "
                f"{_fmt(row['median_arrival_lag_us'])} / {_fmt(row['p95_arrival_lag_us'])} | "
                f"{row['latest_arrival_groups']} |"
            )
        lines.extend(["", "Attribution limits:", ""])
        for limitation in barriers["limitations"]:
            lines.append(f"- {limitation}")

    gap = result["flashinfer_rdma_launch_gaps"]
    lines.extend(
        [
            "",
            "## FlashInfer/RDMA evidence and launch gaps",
            "",
            f"- Explicitly named kernels: {gap['explicit_kernel_count']}",
            f"- Matched CUDA launches: {gap['matched_launches']}; unmatched: {gap['unmatched_launches']}",
            f"- Launch→GPU gap median/p95/max: {_fmt(gap['median_us'])} / "
            f"{_fmt(gap['p95_us'])} / {_fmt(gap['max_us'])} µs",
            f"- {gap['note']}",
            "",
            "## CUDA API categories",
            "",
            "| Category | Calls | Summed thread-ms | Interpretation |",
            "|---|---:|---:|---|",
        ]
    )
    for row in result["cuda_api_categories"]:
        lines.append(
            f"| `{row['category']}` | {row['count']} | {row['summed_thread_time_ms']:.3f} | "
            f"{_escape_markdown(row['description'])} |"
        )

    lines.extend(
        [
            "",
            "## Memcpy operations",
            "",
            "| Trace dev | Kind | Calls | MiB (overlapping ops) | GPU-ms |",
            "|---:|---|---:|---:|---:|",
        ]
    )
    for row in result["memory_copies"]:
        lines.append(
            f"| {row['gpu']} | {_escape_markdown(row['kind'])} | {row['count']} | "
            f"{row['bytes_of_overlapping_ops'] / (1024 * 1024):.3f} | {row['gpu_ms']:.3f} |"
        )
    if not result["memory_copies"]:
        lines.append("| - | No DMA memcpy records in scope | 0 | 0 | 0 |")

    lines.extend(
        [
            "",
            "## Communication NVTX evidence",
            "",
            "These CPU-range totals can overlap because NCCL group/collective ranges may be nested.",
            "",
            "| Evidence | Range | Calls | Summed CPU-range ms |",
            "|---|---|---:|---:|",
        ]
    )
    for row in result["communication_nvtx_ranges"]:
        lines.append(
            f"| `{row['evidence']}` | `{_escape_markdown(row['name'])}` | {row['count']} | "
            f"{row['summed_cpu_range_ms']:.3f} |"
        )
    if not result["communication_nvtx_ranges"]:
        lines.append("| unknown | No explicit communication NVTX range | 0 | 0 |")

    lines.extend(
        [
            "",
            "## Top kernels",
            "",
            "| # | Category | Calls | Total GPU-ms | Avg µs | % kernel GPU-time | Kernel |",
            "|---:|---|---:|---:|---:|---:|---|",
        ]
    )
    for index, row in enumerate(result["top_kernels"], 1):
        lines.append(
            f"| {index} | `{row['category']}` | {row['count']} | {row['total_gpu_ms']:.3f} | "
            f"{row['average_us']:.3f} | {row['percent_of_kernel_gpu_time']:.2f}% | "
            f"`{_escape_markdown(row['name'], 180)}` |"
        )

    lines.extend(["", "## Top OS runtime calls", ""])
    lines.append("OSRT time is summed across threads and is never assigned to RDMA without explicit naming evidence.")
    lines.extend(["", "| Call | Count | Summed thread-ms |", "|---|---:|---:|"])
    for row in result["top_os_runtime_calls"]:
        lines.append(f"| `{_escape_markdown(row['name'])}` | {row['count']} | {row['summed_thread_time_ms']:.3f} |")
    if not result["top_os_runtime_calls"]:
        lines.append("| No OSRT trace data | 0 | 0 |")

    lines.extend(["", "## Warnings and attribution limits", ""])
    for warning in result["warnings"]:
        lines.append(f"- {warning}")
    for key, value in result["methodology"].items():
        lines.append(f"- `{key}`: {value}")
    canonical = result.get("canonical_nsys_stats")
    if canonical:
        lines.extend(
            [
                "",
                "## Canonical `nsys stats` cross-check",
                "",
                f"- Return code: {canonical['returncode']}",
                f"- Output lines: {canonical['output_lines']}",
                f"- Output SHA256: `{canonical['output_sha256']}`",
                f"- Reports detected: `{json.dumps(canonical['reports_detected'], sort_keys=True)}`",
            ]
        )
    return "\n".join(lines) + "\n"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path, help="Input .nsys-rep or .sqlite")
    parser.add_argument("--sqlite", type=Path, help="Explicit existing SQLite export")
    parser.add_argument(
        "--nvtx-range",
        default="auto",
        help="Exact NVTX range, 'auto' (step then T2VA fallback), or 'none'",
    )
    parser.add_argument("--nvtx-domain", help="Optional exact NVTX domain for --nvtx-range")
    parser.add_argument("--top", type=int, default=20, help="Number of top kernels/OSRT calls")
    parser.add_argument(
        "--barriers-per-layer",
        type=int,
        help=(
            "Override chronological Ulysses barrier slots per H3 layer "
            "(auto: 10 with explicit VSA evidence, otherwise 8)"
        ),
    )
    parser.add_argument("--output", type=Path, help="Markdown output (default: stdout)")
    parser.add_argument("--json-output", type=Path, help="Optional machine-readable JSON output")
    parser.add_argument("--raw-nsys-stats", type=Path, help="Optional raw canonical nsys stats output")
    parser.add_argument(
        "--server-log",
        type=Path,
        help=(
            "Server log containing NCCL rank/cudaDev/nvmlDev/busId mappings "
            "(default: auto-detect server-nsys.log beside report/SQLite)"
        ),
    )
    parser.add_argument("--nsys", default=shutil.which("nsys") or "nsys")
    parser.add_argument("--nsys-timeout", type=int, default=900)
    parser.add_argument(
        "--skip-nsys-stats",
        action="store_true",
        help="Skip canonical nsys stats cross-check (intended only for unit tests/minimal exports)",
    )
    args = parser.parse_args(argv)
    if args.top < 1:
        parser.error("--top must be positive")
    if args.nsys_timeout < 1:
        parser.error("--nsys-timeout must be positive")
    if args.barriers_per_layer is not None and args.barriers_per_layer < 1:
        parser.error("--barriers-per-layer must be positive")
    if not args.report.is_file():
        parser.error(f"report does not exist: {args.report}")
    if args.server_log is not None and not args.server_log.is_file():
        parser.error(f"server log does not exist: {args.server_log}")
    return args


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.skip_nsys_stats and shutil.which(args.nsys) is None and not Path(args.nsys).is_file():
        raise SystemExit(f"nsys executable not found: {args.nsys}")
    with materialize_sqlite(args.report, nsys=args.nsys, sqlite_override=args.sqlite) as (
        sqlite_path,
        sqlite_provenance,
    ):
        server_log = args.server_log
        if server_log is None:
            for candidate in (
                args.report.parent / "server-nsys.log",
                sqlite_path.parent / "server-nsys.log",
            ):
                if candidate.is_file():
                    server_log = candidate
                    break
        result = analyze_sqlite(
            sqlite_path,
            requested_range=args.nvtx_range,
            requested_domain=args.nvtx_domain,
            top=args.top,
            server_log=server_log,
            barriers_per_layer=args.barriers_per_layer,
        )
        result["source_report"] = str(args.report)
        result["sqlite_provenance"] = sqlite_provenance
        if not args.skip_nsys_stats:
            canonical, raw_stats = run_canonical_nsys_stats(
                sqlite_path,
                result["scope"],
                nsys=args.nsys,
                timeout=args.nsys_timeout,
            )
            result["canonical_nsys_stats"] = canonical
            if canonical["returncode"] != 0:
                result["warnings"].append(
                    f"canonical nsys stats returned {canonical['returncode']}; inspect raw output"
                )
            if args.raw_nsys_stats:
                _write_text(args.raw_nsys_stats, raw_stats)

        markdown = render_markdown(result, source=args.report)
        if args.output:
            _write_text(args.output, markdown)
        else:
            sys.stdout.write(markdown)
        if args.json_output:
            _write_text(args.json_output, json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
