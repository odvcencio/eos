#!/usr/bin/env python3
"""Summarize an EOS embedder deployment envelope.

The report is intentionally conservative: measured evidence is copied from
files, vector storage envelopes are labelled as payload-only projections, and
release gates come only from explicit legal manifests.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA = "eos.embedder_deployment_envelope.v1"
TURBOQUANT_RETRIEVAL_SCHEMA = "manta.embedding_turboquant_retrieval_metrics.v1"
ENERGY_SCHEMA = "eos.default_embedder_serving_energy.v1"
THROUGHPUT_GATE_SCHEMA = "eos.embedder1_startup_load_encode_throughput_gate.v1"
TSV_COLUMNS = ["section", "evidence_state", "item", "metric", "value", "unit", "source", "notes"]
DEFAULT_BITS = [1, 2, 3, 4, 5, 8]
DEFAULT_CORPUS_SIZES = [1_000_000, 10_000_000]
QUALITY_METRICS = ["ndcg_at_10", "mrr_at_10", "recall_at_10", "recall_at_100"]


class EnvelopeError(ValueError):
    """Raised when inputs cannot produce a trustworthy envelope."""


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_json_object(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise EnvelopeError(f"JSON evidence not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise EnvelopeError(f"{path}: invalid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise EnvelopeError(f"{path}: expected top-level JSON object")
    return data


def parse_int_list(value: str, *, label: str, minimum: int, maximum: int | None = None) -> list[int]:
    out: list[int] = []
    for raw in value.split(","):
        raw = raw.strip()
        if not raw:
            continue
        try:
            parsed = int(raw)
        except ValueError as exc:
            raise EnvelopeError(f"{label} contains non-integer value {raw!r}") from exc
        if parsed < minimum or (maximum is not None and parsed > maximum):
            if maximum is None:
                raise EnvelopeError(f"{label} values must be >= {minimum}: {parsed}")
            raise EnvelopeError(f"{label} values must be between {minimum} and {maximum}: {parsed}")
        out.append(parsed)
    if not out:
        raise EnvelopeError(f"{label} must contain at least one value")
    return sorted(dict.fromkeys(out))


def require_existing_file(path: Path, *, label: str) -> Path:
    if not path.is_file():
        raise EnvelopeError(f"{label} not found: {path}")
    return path


def optional_file_record(path: Path | None, *, label: str) -> dict[str, Any]:
    if path is None:
        return {"label": label, "state": "missing", "path": None, "size_bytes": None}
    if not path.is_file():
        raise EnvelopeError(f"{label} not found: {path}")
    return {"label": label, "state": "measured", "path": str(path), "size_bytes": path.stat().st_size}


def file_sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_footprint(
    *,
    artifact: Path,
    tokenizer: Path | None,
    weights: Path | None,
    checkpoint: Path | None,
) -> dict[str, Any]:
    artifact = require_existing_file(artifact, label="artifact")
    files = {
        "artifact": {
            "label": "artifact",
            "state": "measured",
            "path": str(artifact),
            "size_bytes": artifact.stat().st_size,
            "sha256": file_sha256(artifact),
        },
        "tokenizer": optional_file_record(tokenizer, label="tokenizer"),
        "weights": optional_file_record(weights, label="weights"),
        "checkpoint": optional_file_record(checkpoint, label="checkpoint"),
    }
    supplied_files_total = sum(int(row["size_bytes"] or 0) for row in files.values())
    return {
        "state": "measured",
        "files": files,
        "deployable_artifact_bytes": files["artifact"]["size_bytes"],
        "supplied_files_total_bytes": supplied_files_total,
        "notes": (
            "deployable_artifact_bytes is only the selected artifact. "
            "supplied_files_total_bytes sums every supplied evidence/support file and is not a deploy footprint."
        ),
    }


def ceil_div(num: int, den: int) -> int:
    return (num + den - 1) // den


def turboquant_ip_bytes_per_vector(dimension: int, bits: int) -> dict[str, int]:
    mse_bits = dimension * max(bits - 1, 0)
    sign_bits = dimension
    mse_bytes = ceil_div(mse_bits, 8)
    sign_bytes = ceil_div(sign_bits, 8)
    packed_bytes = mse_bytes + sign_bytes
    norm_resnorm_bytes = 8
    return {
        "mse_bits": mse_bits,
        "sign_bits": sign_bits,
        "mse_bytes": mse_bytes,
        "sign_bytes": sign_bytes,
        "packed_bytes": packed_bytes,
        "norm_resnorm_bytes": norm_resnorm_bytes,
        "bytes_per_vector": packed_bytes + norm_resnorm_bytes,
    }


def projected_vector_payloads(dimension: int, corpus_sizes: list[int], bits: list[int]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    dense_bpv = dimension * 4
    fp16_bpv = dimension * 2
    for corpus_size in corpus_sizes:
        rows.append(
            {
                "state": "projected",
                "corpus_vectors": corpus_size,
                "method": "dense_f32",
                "dimension": dimension,
                "bytes_per_vector": dense_bpv,
                "payload_bytes": corpus_size * dense_bpv,
                "compression_vs_dense_f32": 1.0,
                "notes": "Payload only; excludes ANN graph, ids, metadata, mmap/page overhead, and indexes.",
            }
        )
        rows.append(
            {
                "state": "projected",
                "corpus_vectors": corpus_size,
                "method": "fp16",
                "dimension": dimension,
                "bytes_per_vector": fp16_bpv,
                "payload_bytes": corpus_size * fp16_bpv,
                "compression_vs_dense_f32": dense_bpv / fp16_bpv,
                "notes": "Payload only; excludes ANN graph, ids, metadata, mmap/page overhead, and indexes.",
            }
        )
        for bit in bits:
            tq = turboquant_ip_bytes_per_vector(dimension, bit)
            bytes_per_vector = tq["bytes_per_vector"]
            rows.append(
                {
                    "state": "projected",
                    "corpus_vectors": corpus_size,
                    "method": f"turboquant_ip_b{bit}",
                    "dimension": dimension,
                    **tq,
                    "payload_bytes": corpus_size * bytes_per_vector,
                    "compression_vs_dense_f32": dense_bpv / bytes_per_vector,
                    "notes": (
                        "TurboQuant IP projection: separate byte ceilings for packed MSE bits at bits-1 "
                        "and sign bits, plus 8 bytes for Norm/ResNorm; payload only, excluding "
                        "index/id/metadata overhead."
                    ),
                }
            )
    return rows


def as_float(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        return None
    try:
        parsed = float(value)
    except ValueError:
        return None
    return parsed if math.isfinite(parsed) else None


def as_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def require_list(value: Any, *, path: Path, field: str) -> list[Any]:
    if not isinstance(value, list):
        raise EnvelopeError(f"{path}: {field} must be a list")
    return value


def require_dict(value: Any, *, path: Path, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise EnvelopeError(f"{path}: {field} must be an object")
    return value


def require_nonempty_string(value: Any, *, path: Path, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise EnvelopeError(f"{path}: {field} must be a non-empty string")
    return value


def require_positive_number(value: Any, *, path: Path, field: str, allow_zero: bool = False) -> float:
    parsed = as_float(value)
    if parsed is None or parsed < 0.0 or (parsed == 0.0 and not allow_zero):
        comparator = ">= 0" if allow_zero else "> 0"
        raise EnvelopeError(f"{path}: {field} must be numeric and {comparator}")
    return parsed


def optional_number(value: Any, *, path: Path, field: str) -> float | None:
    if value is None:
        return None
    parsed = as_float(value)
    if parsed is None:
        raise EnvelopeError(f"{path}: {field} must be numeric when present")
    return parsed


def optional_positive_number(value: Any, *, path: Path, field: str, allow_zero: bool = False) -> float | None:
    if value is None:
        return None
    return require_positive_number(value, path=path, field=field, allow_zero=allow_zero)


def require_int_range(value: Any, *, path: Path, field: str, minimum: int, maximum: int | None = None) -> int:
    parsed = as_int(value)
    if parsed is None or parsed < minimum or (maximum is not None and parsed > maximum):
        if maximum is None:
            raise EnvelopeError(f"{path}: {field} must be an integer >= {minimum}")
        raise EnvelopeError(f"{path}: {field} must be an integer between {minimum} and {maximum}")
    return parsed


def parse_leaderboard(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {"state": "missing", "path": None, "rows": [], "macro": {}, "dense_vs_bm25": []}
    require_existing_file(path, label="leaderboard TSV")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if not reader.fieldnames:
            raise EnvelopeError(f"{path}: empty leaderboard TSV")
        required = {"dataset", "backend", "ndcg_at_10"}
        missing = required - set(reader.fieldnames)
        if missing:
            raise EnvelopeError(f"{path}: leaderboard TSV missing columns: {', '.join(sorted(missing))}")
        for raw in reader:
            row: dict[str, Any] = {"dataset": raw.get("dataset"), "backend": raw.get("backend")}
            for key, value in raw.items():
                if key in ("dataset", "backend"):
                    continue
                number = as_float(value)
                row[key] = number if number is not None else value
            rows.append(row)
    macro: dict[str, dict[str, float]] = {}
    by_backend: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        backend = str(row.get("backend") or "")
        by_backend.setdefault(backend, []).append(row)
    for backend, backend_rows in by_backend.items():
        macro[backend] = {}
        for metric in QUALITY_METRICS:
            values = [as_float(row.get(metric)) for row in backend_rows]
            values = [value for value in values if value is not None]
            if values:
                macro[backend][metric] = sum(values) / len(values)
    bm25_by_dataset = {row.get("dataset"): row for row in rows if row.get("backend") == "bm25"}
    dense_vs_bm25: list[dict[str, Any]] = []
    for row in rows:
        backend = row.get("backend")
        dataset = row.get("dataset")
        if backend == "bm25" or dataset not in bm25_by_dataset:
            continue
        comparison = {"dataset": dataset, "backend": backend, "baseline_backend": "bm25"}
        for metric in QUALITY_METRICS:
            value = as_float(row.get(metric))
            base = as_float(bm25_by_dataset[dataset].get(metric))
            if value is not None and base is not None:
                comparison[f"{metric}_delta"] = value - base
        dense_vs_bm25.append(comparison)
    return {"state": "measured", "path": str(path), "rows": rows, "macro": macro, "dense_vs_bm25": dense_vs_bm25}


def summarize_turboquant_metrics(paths: list[Path]) -> dict[str, Any]:
    if not paths:
        return {"state": "missing", "files": [], "rows": []}
    files: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for path in paths:
        data = load_json_object(path)
        schema = data.get("schema")
        if schema != TURBOQUANT_RETRIEVAL_SCHEMA:
            raise EnvelopeError(f"{path}: expected schema {TURBOQUANT_RETRIEVAL_SCHEMA}, got {schema!r}")
        metric_rows = require_list(data.get("rows"), path=path, field="rows")
        if not metric_rows:
            raise EnvelopeError(f"{path}: rows must not be empty")
        config = data.get("config") if isinstance(data.get("config"), dict) else {}
        file_record = {
            "path": str(path),
            "state": "measured",
            "dataset": data.get("dataset"),
            "schema": schema,
            "quantizer_seed": config.get("quantizer_seed"),
        }
        dense = require_dict(data.get("dense"), path=path, field="dense")
        file_record["dense_vector_bytes"] = require_positive_number(
            dense.get("vector_bytes"), path=path, field="dense.vector_bytes"
        )
        if dense.get("scores_per_second") is not None:
            file_record["dense_scores_per_second"] = require_positive_number(
                dense.get("scores_per_second"), path=path, field="dense.scores_per_second"
            )
        files.append(file_record)
        for index, row_value in enumerate(metric_rows):
            row = require_dict(row_value, path=path, field=f"rows[{index}]")
            method = require_nonempty_string(row.get("method"), path=path, field=f"rows[{index}].method")
            bits = require_int_range(row.get("bits"), path=path, field=f"rows[{index}].bits", minimum=1, maximum=8)
            vector_bytes = require_positive_number(
                row.get("vector_bytes"), path=path, field=f"rows[{index}].vector_bytes"
            )
            dense_vector_bytes = require_positive_number(
                row.get("dense_vector_bytes"), path=path, field=f"rows[{index}].dense_vector_bytes"
            )
            compression_ratio = require_positive_number(
                row.get("compression_ratio"), path=path, field=f"rows[{index}].compression_ratio"
            )
            scores_per_second = require_positive_number(
                row.get("scores_per_second"), path=path, field=f"rows[{index}].scores_per_second"
            )
            quality = require_dict(row.get("quality"), path=path, field=f"rows[{index}].quality")
            query_latency = require_dict(row.get("query_latency"), path=path, field=f"rows[{index}].query_latency")
            total_vector_bytes = optional_positive_number(
                row.get("total_vector_bytes"), path=path, field=f"rows[{index}].total_vector_bytes"
            )
            total_compression_ratio = optional_positive_number(
                row.get("total_compression_ratio"), path=path, field=f"rows[{index}].total_compression_ratio"
            )
            ndcg_at_10 = optional_number(
                quality.get("ndcg_at_10"), path=path, field=f"rows[{index}].quality.ndcg_at_10"
            )
            ndcg_at_10_delta = optional_number(
                row.get("ndcg_at_10_delta"), path=path, field=f"rows[{index}].ndcg_at_10_delta"
            )
            recall_at_100 = optional_number(
                quality.get("recall_at_100"), path=path, field=f"rows[{index}].quality.recall_at_100"
            )
            recall_at_100_delta = optional_number(
                row.get("recall_at_100_delta"), path=path, field=f"rows[{index}].recall_at_100_delta"
            )
            p95_ms = require_positive_number(
                query_latency.get("p95_ms"), path=path, field=f"rows[{index}].query_latency.p95_ms", allow_zero=True
            )
            rows.append(
                {
                    "state": "measured",
                    "source": str(path),
                    "dataset": data.get("dataset"),
                    "method": method,
                    "bits": bits,
                    "turboquant_version": row.get("turboquant_version"),
                    "codebook_version": row.get("codebook_version"),
                    "quantizer_seed": config.get("quantizer_seed"),
                    "rerank_overfetch": row.get("rerank_overfetch"),
                    "rerank_storage": row.get("rerank_storage"),
                    "rerank_bits": row.get("rerank_bits"),
                    "vector_bytes": vector_bytes,
                    "dense_vector_bytes": dense_vector_bytes,
                    "compression_ratio": compression_ratio,
                    "total_vector_bytes": total_vector_bytes,
                    "total_compression_ratio": total_compression_ratio,
                    "ndcg_at_10": ndcg_at_10,
                    "ndcg_at_10_delta": ndcg_at_10_delta,
                    "recall_at_100": recall_at_100,
                    "recall_at_100_delta": recall_at_100_delta,
                    "scores_per_second": scores_per_second,
                    "p95_ms": p95_ms,
                }
            )
    return {"state": "measured", "files": files, "rows": rows}


def summarize_energy(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {"state": "missing", "path": None, "phases": []}
    data = load_json_object(path)
    schema = data.get("schema")
    if schema != ENERGY_SCHEMA:
        raise EnvelopeError(f"{path}: expected schema {ENERGY_SCHEMA}, got {schema!r}")
    overall = data.get("overall_status")
    state = "measured" if overall == "measured" else "missing" if overall in (None, "") else str(overall)
    raw_phases = require_list(data.get("phases"), path=path, field="phases")
    if overall == "measured" and not raw_phases:
        raise EnvelopeError(f"{path}: measured energy manifest must include at least one phase")
    phases: list[dict[str, Any]] = []
    for index, phase_value in enumerate(raw_phases):
        phase = require_dict(phase_value, path=path, field=f"phases[{index}]")
        phase_name = require_nonempty_string(phase.get("phase"), path=path, field=f"phases[{index}].phase")
        telemetry_status = require_nonempty_string(
            phase.get("telemetry_status"), path=path, field=f"phases[{index}].telemetry_status"
        )
        wall_seconds = require_positive_number(
            phase.get("wall_seconds"), path=path, field=f"phases[{index}].wall_seconds"
        )
        if telemetry_status == "measured":
            require_positive_number(
                phase.get("energy_joules"), path=path, field=f"phases[{index}].energy_joules", allow_zero=True
            )
        phases.append(
            {
                "phase": phase_name,
                "telemetry_status": telemetry_status,
                "telemetry_source": phase.get("telemetry_source"),
                "wall_seconds": wall_seconds,
                "query_count": phase.get("query_count"),
                "document_count": phase.get("document_count"),
                "workload_item_count": phase.get("workload_item_count"),
                "energy_joules": phase.get("energy_joules"),
                "energy_joules_per_query": phase.get("energy_joules_per_query"),
                "energy_joules_per_workload_item": phase.get("energy_joules_per_workload_item"),
                "incremental_joules_per_query": phase.get("incremental_joules_per_query"),
                "incremental_joules_per_workload_item": phase.get("incremental_joules_per_workload_item"),
            }
        )
    return {
        "state": state,
        "path": str(path),
        "overall_status": overall,
        "unsupported_reason": data.get("unsupported_reason"),
        "workload_mode": data.get("workload_mode"),
        "phases": phases,
    }


def summarize_throughput_gate(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {"state": "missing", "path": None}
    data = load_json_object(path)
    schema = data.get("schema")
    if schema != THROUGHPUT_GATE_SCHEMA:
        raise EnvelopeError(f"{path}: expected schema {THROUGHPUT_GATE_SCHEMA}, got {schema!r}")
    cold_load_ms = require_positive_number(data.get("cold_load_ms"), path=path, field="cold_load_ms")
    first_query_encode_ms = require_positive_number(
        data.get("first_query_encode_ms"), path=path, field="first_query_encode_ms"
    )
    warm_batch64_docs_per_second = require_positive_number(
        data.get("warm_batch64_docs_per_second"), path=path, field="warm_batch64_docs_per_second"
    )
    peak_rss_mb = require_positive_number(data.get("peak_rss_mb"), path=path, field="peak_rss_mb")
    return {
        "state": "measured",
        "path": str(path),
        "cold_load_ms": cold_load_ms,
        "first_query_encode_ms": first_query_encode_ms,
        "warm_batch64_docs_per_second": warm_batch64_docs_per_second,
        "peak_rss_mb": peak_rss_mb,
        "explicit_owner_exception": data.get("explicit_owner_exception") is True,
        "notes": "This is startup/load/encode evidence only, not end-to-end retrieval service throughput.",
    }


def summarize_legal(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {
            "state": "unknown",
            "path": None,
            "commercial_release": {"state": "unknown", "source_field": "commercial_use_allowed"},
            "open_or_free_release": {"state": "unknown", "source_field": "release_train_allowed"},
            "research_training": {"state": "unknown", "source_field": "train_allowed_for_research"},
            "blockers": ["legal manifest not supplied"],
        }
    data = load_json_object(path)
    gates = data.get("legal_gates")
    if not isinstance(gates, dict):
        return {
            "state": "unknown",
            "path": str(path),
            "commercial_release": {"state": "unknown", "source_field": "commercial_use_allowed"},
            "open_or_free_release": {"state": "unknown", "source_field": "release_train_allowed"},
            "research_training": {"state": "unknown", "source_field": "train_allowed_for_research"},
            "blockers": ["legal_gates object missing"],
        }

    def gate(field: str) -> dict[str, Any]:
        value = gates.get(field)
        if value is True:
            state = "allowed"
        elif value is False:
            state = "blocked"
        else:
            state = "unknown"
        return {"state": state, "allowed": value if isinstance(value, bool) else None, "source_field": field}

    commercial = gate("commercial_use_allowed")
    open_release = gate("release_train_allowed")
    research = gate("train_allowed_for_research")
    blockers: list[str] = []
    if commercial["state"] == "blocked":
        blockers.append("commercial_use_allowed=false")
    if open_release["state"] == "blocked":
        blockers.append("release_train_allowed=false")
    if commercial["state"] == "unknown":
        blockers.append("commercial_use_allowed unknown")
    if open_release["state"] == "unknown":
        blockers.append("release_train_allowed unknown")
    return {
        "state": "blocked" if blockers else "allowed",
        "path": str(path),
        "commercial_release": commercial,
        "open_or_free_release": open_release,
        "research_training": research,
        "raw_legal_gates": gates,
        "blockers": blockers,
        "notes": "Release gates are legal/provenance gates only; model quality never overrides them.",
    }


def build_summary(
    *,
    artifact: Path,
    dimension: int,
    tokenizer: Path | None = None,
    weights: Path | None = None,
    checkpoint: Path | None = None,
    leaderboard_tsv: Path | None = None,
    turboquant_metrics_json: list[Path] | None = None,
    energy_json: Path | None = None,
    throughput_gate_json: Path | None = None,
    legal_manifest_json: Path | None = None,
    corpus_sizes: list[int] | None = None,
    bits: list[int] | None = None,
    model_family: str = "unspecified",
    clock: Any = utc_now,
) -> dict[str, Any]:
    if dimension <= 0:
        raise EnvelopeError("--dimension must be > 0")
    corpus_sizes = corpus_sizes or DEFAULT_CORPUS_SIZES
    bits = bits or DEFAULT_BITS
    footprint = artifact_footprint(artifact=artifact, tokenizer=tokenizer, weights=weights, checkpoint=checkpoint)
    legal = summarize_legal(legal_manifest_json)
    leaderboard = parse_leaderboard(leaderboard_tsv)
    projections = projected_vector_payloads(dimension, corpus_sizes, bits)
    tq = summarize_turboquant_metrics(turboquant_metrics_json or [])
    energy = summarize_energy(energy_json)
    throughput = summarize_throughput_gate(throughput_gate_json)
    measured_sections = ["artifact_footprint"]
    for name, section in {
        "leaderboard": leaderboard,
        "turboquant_metrics": tq,
        "energy": energy,
        "throughput_gate": throughput,
    }.items():
        if section.get("state") == "measured":
            measured_sections.append(name)
    missing_sections = [
        name
        for name, section in {
            "leaderboard": leaderboard,
            "turboquant_metrics": tq,
            "energy": energy,
            "throughput_gate": throughput,
        }.items()
        if section.get("state") == "missing"
    ]
    return {
        "schema": SCHEMA,
        "generated_at": clock(),
        "identity": {
            "model_family": model_family,
            "dimension": dimension,
            "artifact_path": str(artifact),
            "artifact_sha256": footprint["files"]["artifact"]["sha256"],
        },
        "artifact_footprint": footprint,
        "vector_payload_projections": {
            "state": "projected",
            "dimension": dimension,
            "corpus_sizes": corpus_sizes,
            "bits": bits,
            "accounting_scope": "payload_only_excludes_index_ids_metadata",
            "rows": projections,
        },
        "retrieval_leaderboard": leaderboard,
        "turboquant_metrics": tq,
        "serving_energy": energy,
        "startup_load_encode_throughput_gate": throughput,
        "legal_release_gates": legal,
        "summary": {
            "commercial_release_state": legal["commercial_release"]["state"],
            "open_or_free_release_state": legal["open_or_free_release"]["state"],
            "legal_blockers": legal.get("blockers", []),
            "missing_evidence_sections": missing_sections,
            "measured_evidence_sections": measured_sections,
            "notes": [
                "Vector storage projections are not measured service or database footprints.",
                "Retrieval score throughput is vector scoring evidence, not end-to-end serving throughput.",
                "Commercial/open release readiness is blocked or allowed only by explicit legal gates.",
            ],
        },
    }


def add_tsv(rows: list[dict[str, Any]], section: str, state: str, item: Any, metric: str, value: Any, unit: str, source: str, notes: str = "") -> None:
    rows.append(
        {
            "section": section,
            "evidence_state": state,
            "item": str(item),
            "metric": metric,
            "value": "" if value is None else str(value),
            "unit": unit,
            "source": source,
            "notes": notes,
        }
    )


def flatten_tsv(summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key, record in summary["artifact_footprint"]["files"].items():
        add_tsv(rows, "artifact_footprint", record["state"], key, "size_bytes", record.get("size_bytes"), "bytes", record.get("path") or "")
    add_tsv(
        rows,
        "artifact_footprint",
        "measured",
        "artifact",
        "deployable_artifact_bytes",
        summary["artifact_footprint"].get("deployable_artifact_bytes"),
        "bytes",
        summary["artifact_footprint"]["files"]["artifact"].get("path") or "",
        summary["artifact_footprint"].get("notes", ""),
    )
    add_tsv(
        rows,
        "artifact_footprint",
        "measured",
        "supplied_files",
        "supplied_files_total_bytes",
        summary["artifact_footprint"].get("supplied_files_total_bytes"),
        "bytes",
        "",
        summary["artifact_footprint"].get("notes", ""),
    )
    for row in summary["vector_payload_projections"]["rows"]:
        add_tsv(
            rows,
            "vector_payload_projection",
            row["state"],
            f"{row['corpus_vectors']}:{row['method']}",
            "payload_bytes",
            row["payload_bytes"],
            "bytes",
            "formula",
            row["notes"],
        )
        add_tsv(
            rows,
            "vector_payload_projection",
            row["state"],
            f"{row['corpus_vectors']}:{row['method']}",
            "compression_vs_dense_f32",
            row["compression_vs_dense_f32"],
            "ratio",
            "formula",
            row["notes"],
        )
    leaderboard = summary["retrieval_leaderboard"]
    for row in leaderboard.get("rows", []):
        for metric in QUALITY_METRICS + ["scores_per_second"]:
            if metric in row:
                add_tsv(rows, "retrieval_leaderboard", leaderboard["state"], f"{row.get('dataset')}:{row.get('backend')}", metric, row.get(metric), "", leaderboard.get("path") or "")
    if leaderboard.get("state") == "missing":
        add_tsv(rows, "retrieval_leaderboard", "missing", "leaderboard", "status", "missing", "", "", "No leaderboard TSV supplied.")
    for row in summary["turboquant_metrics"].get("rows", []):
        for metric in ("compression_ratio", "total_compression_ratio", "ndcg_at_10_delta", "recall_at_100_delta", "scores_per_second", "p95_ms"):
            add_tsv(rows, "turboquant_metrics", row["state"], f"{row.get('dataset')}:{row.get('method')}", metric, row.get(metric), "", row.get("source") or "")
    if summary["turboquant_metrics"].get("state") == "missing":
        add_tsv(rows, "turboquant_metrics", "missing", "turboquant", "status", "missing", "", "", "No TurboQuant metrics JSON supplied.")
    energy = summary["serving_energy"]
    for phase in energy.get("phases", []):
        for metric in ("wall_seconds", "energy_joules", "energy_joules_per_query", "energy_joules_per_workload_item", "incremental_joules_per_query", "incremental_joules_per_workload_item"):
            add_tsv(rows, "serving_energy", energy["state"], phase.get("phase"), metric, phase.get(metric), "", energy.get("path") or "")
    if energy.get("state") == "missing":
        add_tsv(rows, "serving_energy", "missing", "energy", "status", "missing", "", "", "No serving energy manifest supplied.")
    throughput = summary["startup_load_encode_throughput_gate"]
    for metric in ("cold_load_ms", "first_query_encode_ms", "warm_batch64_docs_per_second", "peak_rss_mb"):
        if metric in throughput:
            add_tsv(rows, "throughput_gate", throughput["state"], "startup_load_encode", metric, throughput.get(metric), "", throughput.get("path") or "", throughput.get("notes", ""))
    if throughput.get("state") == "missing":
        add_tsv(rows, "throughput_gate", "missing", "startup_load_encode", "status", "missing", "", "", "No throughput gate JSON supplied.")
    legal = summary["legal_release_gates"]
    for item in ("commercial_release", "open_or_free_release", "research_training"):
        gate = legal[item]
        add_tsv(rows, "legal_release_gates", gate["state"], item, gate["source_field"], gate.get("allowed"), "bool", legal.get("path") or "", legal.get("notes", ""))
    return rows


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=TSV_COLUMNS, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", required=True, type=Path)
    parser.add_argument("--dimension", required=True, type=int)
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--leaderboard-tsv", type=Path)
    parser.add_argument("--turboquant-metrics-json", type=Path, action="append", default=[])
    parser.add_argument("--energy-json", type=Path)
    parser.add_argument("--throughput-gate-json", type=Path)
    parser.add_argument("--legal-manifest-json", type=Path)
    parser.add_argument("--corpus-size", default=",".join(str(value) for value in DEFAULT_CORPUS_SIZES))
    parser.add_argument("--bits", default=",".join(str(value) for value in DEFAULT_BITS))
    parser.add_argument("--model-family", default="unspecified")
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-tsv", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    try:
        summary = build_summary(
            artifact=args.artifact,
            dimension=args.dimension,
            tokenizer=args.tokenizer,
            weights=args.weights,
            checkpoint=args.checkpoint,
            leaderboard_tsv=args.leaderboard_tsv,
            turboquant_metrics_json=args.turboquant_metrics_json,
            energy_json=args.energy_json,
            throughput_gate_json=args.throughput_gate_json,
            legal_manifest_json=args.legal_manifest_json,
            corpus_sizes=parse_int_list(args.corpus_size, label="--corpus-size", minimum=1),
            bits=parse_int_list(args.bits, label="--bits", minimum=1, maximum=8),
            model_family=args.model_family,
        )
        rendered = json.dumps(summary, indent=2, sort_keys=True) + "\n"
        if args.output_json:
            args.output_json.parent.mkdir(parents=True, exist_ok=True)
            args.output_json.write_text(rendered, encoding="utf-8")
        else:
            sys.stdout.write(rendered)
        if args.output_tsv:
            args.output_tsv.parent.mkdir(parents=True, exist_ok=True)
            write_tsv(args.output_tsv, flatten_tsv(summary))
    except EnvelopeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
