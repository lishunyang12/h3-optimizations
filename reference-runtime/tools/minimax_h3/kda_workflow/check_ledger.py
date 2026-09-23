#!/usr/bin/env python3
"""Validate the H3 kernel ledger and its NCU promotion gate using stdlib only."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
ALLOWED_STATUS = {"draft", "measured", "ncu_pending", "qualified", "rejected"}
ALLOWED_NCU = {"pending", "pass", "fail", "not_run_early_stop"}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load_records(path: Path = HERE / "candidates.jsonl") -> list[dict]:
    records = []
    for line_no, line in enumerate(path.read_text().splitlines(), 1):
        if line.strip():
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
    return records


def load_benchmarks(path: Path = HERE / "benchmarks.csv") -> dict[str, dict[str, str]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    result = {row["benchmark_id"]: row for row in rows}
    _require(len(result) == len(rows), "duplicate benchmark_id in benchmarks.csv")
    return result


def validate_record(record: dict, benchmarks: dict[str, dict[str, str]]) -> None:
    candidate_id = record.get("candidate_id", "<missing>")
    _require(record.get("schema_version") == 1, f"{candidate_id}: schema_version must be 1")
    _require(record.get("status") in ALLOWED_STATUS, f"{candidate_id}: invalid status")
    _require(record.get("target", {}).get("model") == "MiniMax-H3", f"{candidate_id}: wrong model")
    _require(record.get("target", {}).get("architecture") == "sm120", f"{candidate_id}: wrong architecture")

    artifacts = record.get("artifacts", {})
    for key in ("candidate", "benchmark"):
        path = ROOT / artifacts.get(key, "")
        _require(path.is_file(), f"{candidate_id}: missing {key} artifact {path}")
    for raw in artifacts.get("raw_results", []):
        _require((ROOT / raw).is_file() or Path(raw).is_file(), f"{candidate_id}: missing raw result {raw}")

    ids = record.get("benchmark_ids", [])
    _require(ids, f"{candidate_id}: at least one benchmark_id is required")
    for benchmark_id in ids:
        _require(benchmark_id in benchmarks, f"{candidate_id}: unknown benchmark {benchmark_id}")
        row = benchmarks[benchmark_id]
        _require(row["candidate_id"] == candidate_id, f"{benchmark_id}: candidate_id mismatch")
        baseline = float(row["baseline_median_ms"])
        candidate = float(row["candidate_median_ms"])
        observed = float(row["saving_ms_per_layer"])
        _require(math.isclose(observed, baseline - candidate, abs_tol=1e-9), f"{benchmark_id}: saving arithmetic mismatch")

    gates = record.get("gates", {})
    correctness = gates.get("correctness", {})
    performance = gates.get("performance", {})
    ncu = gates.get("ncu", {})
    threshold = float(performance.get("minimum_saving_ms_per_layer", -1))
    saving = float(performance.get("observed_saving_ms_per_layer", 0))
    _require(performance.get("passed") is (saving >= threshold), f"{candidate_id}: inconsistent performance gate")
    if correctness.get("mode") == "bitwise":
        expected = bool(correctness.get("bit_exact")) and correctness.get("mismatch_elements") == 0
        _require(correctness.get("passed") is expected, f"{candidate_id}: inconsistent bitwise gate")

    _require(ncu.get("required_before_production") is True, f"{candidate_id}: NCU gate must be required")
    _require(ncu.get("status") in ALLOWED_NCU, f"{candidate_id}: invalid NCU status")
    if ncu.get("status") == "pass":
        _require(ncu.get("reports"), f"{candidate_id}: passing NCU gate needs a report")

    decision = record.get("decision", {})
    if record.get("status") == "qualified" or decision.get("verdict") == "qualify":
        _require(correctness.get("passed") is True, f"{candidate_id}: qualified without correctness")
        _require(performance.get("passed") is True, f"{candidate_id}: qualified without speed")
        _require(ncu.get("status") == "pass", f"{candidate_id}: qualified without NCU pass")
    if ncu.get("status") == "not_run_early_stop":
        _require(not correctness.get("passed") or not performance.get("passed"), f"{candidate_id}: invalid NCU early-stop")
    if record.get("status") == "rejected":
        _require(decision.get("verdict") == "reject", f"{candidate_id}: rejected status/verdict mismatch")
        _require(decision.get("production_dispatch_modified") is False, f"{candidate_id}: rejected candidate changed production dispatch")


def validate() -> tuple[int, int]:
    benchmarks = load_benchmarks()
    records = load_records()
    ids = [record.get("candidate_id") for record in records]
    _require(len(ids) == len(set(ids)), "duplicate candidate_id in candidates.jsonl")
    for record in records:
        validate_record(record, benchmarks)
    referenced = {item for record in records for item in record["benchmark_ids"]}
    _require(referenced == set(benchmarks), "orphan benchmark row or missing CSV row")
    return len(records), len(benchmarks)


if __name__ == "__main__":
    candidate_count, benchmark_count = validate()
    print(f"ledger valid: {candidate_count} candidates, {benchmark_count} benchmarks")
