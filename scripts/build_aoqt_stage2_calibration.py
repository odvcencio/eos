#!/usr/bin/env python3
"""Build a plan-only AOQT Stage 2 calibration manifest.

This tool intentionally emits metadata and workload only.  It validates frozen
anchor, exclusion, vector-cache, and score-cache manifests, then writes a
deterministic calibration plan with row/pair identities.  It never materializes
vectors, numeric scores, model outputs, training caches, or evaluation results.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

SCHEMA = "eos.aoqt_stage2_calibration_plan.v1"
ANCHOR_SCHEMA = "eos.aoqt_stage2.anchor_manifest.v1"
EXCLUSION_SCHEMA = "eos.aoqt_stage2.exclusion_qids.v1"
VECTOR_CACHE_SCHEMA = "eos.aoqt_stage2.vector_cache_manifest.v1"
SCORE_CACHE_SCHEMA = "eos.aoqt_stage2.score_cache_manifest.v1"
ALLOWED_DATASETS = ("fiqa", "nfcorpus", "scifact")
FORBIDDEN_SPLITS = ("dev", "reserve", "official", "test", "eval", "heldout")
REQUIRED_EXCLUSION_NAMES = ("dev4", "reserve4", "official-test")
TOPOLOGY = {
    "id": "aoqt_givens_v1",
    "dim": 384,
    "stages": 8,
    "pairs_per_stage": 192,
    "angle_count": 1536,
    "angle_cap_default": 0.04,
    "angle_cap_hard": 0.08,
}
TOPOLOGY_SHA256 = sha256_json(TOPOLOGY) if "sha256_json" in globals() else ""
LEGAL_SCOPE = {
    "train_allowed_for_research": True,
    "release_train_allowed": False,
    "commercial_use_allowed": False,
    "free_open_release_allowed": False,
    "quality_claim": False,
}


class PlanError(ValueError):
    """Raised when a manifest cannot be trusted for AOQT calibration planning."""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchor-manifest", type=Path, required=True)
    parser.add_argument("--exclusion-qids", type=Path, action="append", default=[])
    parser.add_argument("--vector-cache", type=Path, action="append", default=[])
    parser.add_argument("--score-cache", type=Path, action="append", default=[])
    parser.add_argument("--output-plan", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=191)
    parser.add_argument("--top10-limit", type=int, default=10)
    parser.add_argument("--nf-start", type=int, default=80)
    parser.add_argument("--nf-end", type=int, default=120)
    parser.add_argument("--expected-anchor-sha256")
    return parser.parse_args(argv)


def repo_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def display_path(path: Path) -> str:
    resolved = repo_path(path).resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        return str(resolved)


def utc_stamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def canonical_bytes(payload: Any) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha256_json(payload: Any) -> str:
    return hashlib.sha256(canonical_bytes(payload)).hexdigest()


TOPOLOGY_SHA256 = sha256_json(TOPOLOGY)
LEGAL_SCOPE_SHA256 = sha256_json(LEGAL_SCOPE)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with repo_path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(repo_path(path).read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise PlanError(f"{display_path(path)}: invalid JSON") from exc
    if not isinstance(payload, dict):
        raise PlanError(f"{display_path(path)}: expected JSON object")
    return payload


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    target = repo_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(target)


def require_schema(payload: dict[str, Any], schema: str, label: str) -> None:
    if payload.get("schema") != schema:
        raise PlanError(f"{label}: schema mismatch")


def require_train_split(value: Any, label: str) -> None:
    if value != "train":
        raise PlanError(f"{label}: split must be train")


def require_dataset(value: Any, label: str) -> str:
    dataset = str(value).lower()
    if dataset not in ALLOWED_DATASETS:
        raise PlanError(f"{label}: unsupported dataset {value!r}")
    return dataset


def require_sha256(value: Any, label: str, field: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise PlanError(f"{label}: {field} must be lowercase sha256")
    return value


def require_sha256_mapping(value: Any, label: str, field: str) -> dict[str, str] | None:
    if value is None:
        return None
    if not isinstance(value, dict) or not value:
        raise PlanError(f"{label}: {field} must be a non-empty sha256 mapping")
    out: dict[str, str] = {}
    for key, item in value.items():
        if not isinstance(key, str) or not key:
            raise PlanError(f"{label}: {field} keys must be non-empty strings")
        out[key] = require_sha256(item, label, f"{field}.{key}")
    return dict(sorted(out.items()))


def require_provenance_hashes(payload: dict[str, Any], label: str) -> dict[str, Any]:
    source_sha = payload.get("source_sha256")
    source_map = require_sha256_mapping(payload.get("source_sha256_by_file"), label, "source_sha256_by_file")
    if source_sha is None and source_map is None:
        raise PlanError(f"{label}: source hash provenance is required")
    provenance = {}
    if source_sha is not None:
        provenance["source_sha256"] = require_sha256(source_sha, label, "source_sha256")
    if source_map is not None:
        provenance["source_sha256_by_file"] = source_map
    qrels_sha = payload.get("qrels_sha256")
    if qrels_sha is not None:
        provenance["qrels_sha256"] = require_sha256(qrels_sha, label, "qrels_sha256")
    return provenance


def require_anchor_topology_legal_binding(payload: dict[str, Any], label: str, anchor: dict[str, Any]) -> dict[str, Any]:
    anchor_binding = payload.get("anchor")
    if not isinstance(anchor_binding, dict):
        raise PlanError(f"{label}: anchor binding object is required")
    if anchor_binding.get("package_sha256") != anchor["package_sha256"]:
        raise PlanError(f"{label}: anchor package_sha256 binding mismatch")
    if anchor_binding.get("manifest_sha256") != anchor["manifest_sha256"]:
        raise PlanError(f"{label}: anchor manifest_sha256 binding mismatch")

    topology_binding = payload.get("topology")
    if not isinstance(topology_binding, dict):
        raise PlanError(f"{label}: topology binding object is required")
    if topology_binding.get("id") != TOPOLOGY["id"] or topology_binding.get("sha256") != TOPOLOGY_SHA256:
        raise PlanError(f"{label}: topology digest/version binding mismatch")

    legal_binding = payload.get("legal_scope")
    if legal_binding != LEGAL_SCOPE:
        raise PlanError(f"{label}: legal scope binding mismatch")

    return {
        "anchor": {
            "package_sha256": anchor["package_sha256"],
            "manifest_sha256": anchor["manifest_sha256"],
        },
        "topology": {"id": TOPOLOGY["id"], "sha256": TOPOLOGY_SHA256},
        "legal_scope_sha256": LEGAL_SCOPE_SHA256,
    }


def lower_pathish_values(node: Any) -> Iterable[str]:
    if isinstance(node, dict):
        for value in node.values():
            yield from lower_pathish_values(value)
    elif isinstance(node, list):
        for value in node:
            yield from lower_pathish_values(value)
    elif isinstance(node, str):
        yield node.lower()


def require_no_forbidden_split_tokens(payload: Any, label: str) -> None:
    for value in lower_pathish_values(payload):
        parts = [part for part in re.split(r"[^a-z0-9]+", value) if part]
        if any(part in FORBIDDEN_SPLITS for part in parts):
            raise PlanError(f"{label}: forbidden split token in {value!r}")


def require_no_vectors(node: Any, label: str) -> None:
    if isinstance(node, dict):
        for key, value in node.items():
            low = str(key).lower()
            if low in {"vector", "vectors", "embedding", "embeddings", "values", "float32"}:
                raise PlanError(f"{label}: raw vector-like field {key!r} is forbidden")
            require_no_vectors(value, label)
    elif isinstance(node, list):
        for value in node:
            require_no_vectors(value, label)


def require_output_has_no_vectors_or_numeric_scores(node: Any, label: str = "output") -> None:
    if isinstance(node, dict):
        for key, value in node.items():
            low = str(key).lower()
            if low in {"vector", "vectors", "embedding", "embeddings", "values", "float32"}:
                raise PlanError(f"{label}: vector-like field {key!r} is forbidden")
            if "score" in low and isinstance(value, (int, float)):
                raise PlanError(f"{label}: numeric score field {key!r} is forbidden")
            require_output_has_no_vectors_or_numeric_scores(value, label)
    elif isinstance(node, list):
        for value in node:
            require_output_has_no_vectors_or_numeric_scores(value, label)


def validate_anchor(path: Path, expected_sha256: str | None) -> dict[str, Any]:
    payload = load_json(path)
    require_schema(payload, ANCHOR_SCHEMA, display_path(path))
    require_no_forbidden_split_tokens(payload, display_path(path))
    require_no_vectors(payload, display_path(path))
    if payload.get("embedding_dim") != 384:
        raise PlanError("anchor: embedding_dim must be 384")
    if payload.get("topology") != TOPOLOGY:
        raise PlanError("anchor: topology mismatch")
    if payload.get("legal_scope") != LEGAL_SCOPE:
        raise PlanError("anchor: legal scope mismatch")
    package_sha = payload.get("package_sha256")
    if not isinstance(package_sha, str) or not re.fullmatch(r"[0-9a-f]{64}", package_sha):
        raise PlanError("anchor: package_sha256 must be lowercase sha256")
    if expected_sha256 is not None and package_sha != expected_sha256:
        raise PlanError("anchor: package_sha256 does not match expected pin")
    return {
        "path": display_path(path),
        "manifest_sha256": sha256_file(path),
        "package_path": str(payload.get("package_path", "")),
        "package_sha256": package_sha,
        "embedding_dim": 384,
        "topology": TOPOLOGY,
        "topology_sha256": TOPOLOGY_SHA256,
        "legal_scope": LEGAL_SCOPE,
        "legal_scope_sha256": LEGAL_SCOPE_SHA256,
    }


def validate_exclusions(paths: list[Path]) -> dict[str, Any]:
    if not paths:
        raise PlanError("exclusions: at least one qid-only exclusion manifest is required")
    names: dict[str, set[str]] = {}
    audit = []
    for path in paths:
        payload = load_json(path)
        label = display_path(path)
        require_schema(payload, EXCLUSION_SCHEMA, label)
        allowed = {"schema", "name", "qids", "source_sha256", "source_sha256_by_file"}
        unexpected = sorted(set(payload) - allowed)
        if unexpected:
            raise PlanError(f"{label}: exclusion manifest must be qid-only; unexpected fields {unexpected}")
        name = str(payload.get("name", ""))
        qids = payload.get("qids")
        if not name or not isinstance(qids, list) or not all(isinstance(qid, str) and qid for qid in qids):
            raise PlanError(f"{label}: invalid exclusion name/qids")
        if not qids:
            raise PlanError(f"{label}: qids must be non-empty")
        if name not in REQUIRED_EXCLUSION_NAMES:
            require_no_forbidden_split_tokens({"name": name}, label)
        provenance = require_provenance_hashes(payload, label)
        if len(qids) != len(set(qids)):
            raise PlanError(f"{label}: duplicate exclusion qids")
        if name in names:
            raise PlanError(f"exclusions: duplicate qid-only set {name!r}")
        names[name] = set(qids)
        audit.append({"name": name, "path": label, "qid_count": len(qids), "manifest_sha256": sha256_file(path), "provenance": provenance})
    missing = [name for name in REQUIRED_EXCLUSION_NAMES if name not in names]
    if missing:
        raise PlanError(f"exclusions: missing required qid-only sets {missing}")
    all_qids: set[str] = set()
    for qids in names.values():
        all_qids.update(qids)
    return {"sets": sorted(audit, key=lambda item: item["name"]), "excluded_qids": sorted(all_qids)}


def validate_vector_caches(paths: list[Path], anchor: dict[str, Any]) -> dict[str, dict[str, Any]]:
    by_key: dict[str, dict[str, Any]] = {}
    for path in paths:
        payload = load_json(path)
        label = display_path(path)
        require_schema(payload, VECTOR_CACHE_SCHEMA, label)
        require_train_split(payload.get("split"), label)
        require_no_forbidden_split_tokens(payload, label)
        require_no_vectors(payload, label)
        binding = require_anchor_topology_legal_binding(payload, label, anchor)
        provenance = require_provenance_hashes(payload, label)
        if "qrels_sha256" not in provenance:
            raise PlanError(f"{label}: qrels_sha256 provenance is required")
        dataset = require_dataset(payload.get("dataset"), label)
        role = str(payload.get("role", ""))
        if role not in {"query", "doc"}:
            raise PlanError(f"{label}: role must be query or doc")
        ids = payload.get("ids")
        if not isinstance(ids, list) or not all(isinstance(item, str) and item for item in ids):
            raise PlanError(f"{label}: ids must be non-empty strings")
        if len(ids) != len(set(ids)):
            raise PlanError(f"{label}: duplicate vector ids")
        cache_sha = require_sha256(payload.get("cache_sha256"), label, "cache_sha256")
        key = f"{dataset}:{role}"
        if key in by_key:
            raise PlanError(f"vector caches: duplicate {key}")
        by_key[key] = {
            "path": label,
            "manifest_sha256": sha256_file(path),
            "cache_sha256": cache_sha,
            "dataset": dataset,
            "role": role,
            "id_count": len(ids),
            "ids": set(ids),
            "binding": binding,
            "provenance": provenance,
        }
    for dataset in ALLOWED_DATASETS:
        for role in ("query", "doc"):
            if f"{dataset}:{role}" not in by_key:
                raise PlanError(f"vector caches: missing train {dataset}:{role}")
    return by_key


def parse_int_field(value: Any, label: str, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise PlanError(f"{label}: {field} must be an integer")
    return value


def qrels_sha256(item: dict[str, Any], label: str) -> str:
    provenance = item.get("provenance")
    if not isinstance(provenance, dict):
        raise PlanError(f"{label}: provenance is required")
    value = provenance.get("qrels_sha256")
    if not isinstance(value, str):
        raise PlanError(f"{label}: qrels_sha256 provenance is required")
    return value


def source_provenance_signature(item: dict[str, Any], label: str) -> str:
    provenance = item.get("provenance")
    if not isinstance(provenance, dict):
        raise PlanError(f"{label}: provenance is required")
    source_sha = provenance.get("source_sha256")
    source_map = provenance.get("source_sha256_by_file")
    return sha256_json({"source_sha256": source_sha, "source_sha256_by_file": source_map})


def require_provenance_consistency(vector_caches: dict[str, dict[str, Any]], score_caches: list[dict[str, Any]]) -> dict[str, str]:
    by_dataset: dict[str, str] = {}
    for dataset in ALLOWED_DATASETS:
        labels_and_hashes = [
            (f"vector cache {dataset}:query", qrels_sha256(vector_caches[f"{dataset}:query"], f"vector cache {dataset}:query")),
            (f"vector cache {dataset}:doc", qrels_sha256(vector_caches[f"{dataset}:doc"], f"vector cache {dataset}:doc")),
        ]
        for cache in score_caches:
            if cache["dataset"] == dataset:
                labels_and_hashes.append((f"score cache {dataset} q{cache['bits']}", qrels_sha256(cache, f"score cache {dataset} q{cache['bits']}")))
        observed = {value for _, value in labels_and_hashes}
        if len(observed) != 1:
            details = ", ".join(f"{label}={value}" for label, value in labels_and_hashes)
            raise PlanError(f"qrels provenance mismatch for dataset {dataset}: {details}")
        by_dataset[dataset] = labels_and_hashes[0][1]
        labels_and_sources = [
            (f"vector cache {dataset}:query", source_provenance_signature(vector_caches[f"{dataset}:query"], f"vector cache {dataset}:query")),
            (f"vector cache {dataset}:doc", source_provenance_signature(vector_caches[f"{dataset}:doc"], f"vector cache {dataset}:doc")),
        ]
        for cache in score_caches:
            if cache["dataset"] == dataset:
                labels_and_sources.append((f"score cache {dataset} q{cache['bits']}", source_provenance_signature(cache, f"score cache {dataset} q{cache['bits']}")))
        source_observed = {value for _, value in labels_and_sources}
        if len(source_observed) != 1:
            details = ", ".join(f"{label}={value}" for label, value in labels_and_sources)
            raise PlanError(f"source provenance mismatch for dataset {dataset}: {details}")
    return by_dataset


def validate_score_cache(path: Path, excluded_qids: set[str], vector_caches: dict[str, dict[str, Any]], anchor: dict[str, Any], seed: int) -> dict[str, Any]:
    payload = load_json(path)
    label = display_path(path)
    require_schema(payload, SCORE_CACHE_SCHEMA, label)
    require_train_split(payload.get("split"), label)
    require_no_forbidden_split_tokens(payload, label)
    require_no_vectors(payload, label)
    binding = require_anchor_topology_legal_binding(payload, label, anchor)
    provenance = require_provenance_hashes(payload, label)
    if "qrels_sha256" not in provenance:
        raise PlanError(f"{label}: qrels_sha256 provenance is required")
    dataset = require_dataset(payload.get("dataset"), label)
    bits = parse_int_field(payload.get("bits", -1), label, "bits")
    if bits not in {3, 5}:
        raise PlanError(f"{label}: bits must be 3 or 5")
    if payload.get("score_mode") != "prepared_ip":
        raise PlanError(f"{label}: score_mode must be prepared_ip")
    top_k = parse_int_field(payload.get("top_k", 0), label, "top_k")
    if top_k < 120:
        raise PlanError(f"{label}: top_k must be at least 120")
    tq = payload.get("turboquant")
    if not isinstance(tq, dict):
        raise PlanError(f"{label}: turboquant binding object is required")
    if tq.get("score_mode") != "prepared_ip" or tq.get("bits") != bits or tq.get("seed") != seed:
        raise PlanError(f"{label}: TurboQuant prepared-IP config binding mismatch")
    query_cache = vector_caches[f"{dataset}:query"]
    doc_cache = vector_caches[f"{dataset}:doc"]
    vector_binding = payload.get("vector_cache")
    expected_vector_binding = {
        "query_manifest_sha256": query_cache["manifest_sha256"],
        "query_cache_sha256": query_cache["cache_sha256"],
        "doc_manifest_sha256": doc_cache["manifest_sha256"],
        "doc_cache_sha256": doc_cache["cache_sha256"],
    }
    if vector_binding != expected_vector_binding:
        raise PlanError(f"{label}: vector cache binding mismatch")
    rows = payload.get("rows")
    if not isinstance(rows, list) or not rows:
        raise PlanError(f"{label}: rows must be non-empty")
    query_ids = query_cache["ids"]
    doc_ids = doc_cache["ids"]
    normalized_rows = []
    seen_qids: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            raise PlanError(f"{label}: row must be object")
        qid = row.get("qid")
        if not isinstance(qid, str) or not qid:
            raise PlanError(f"{label}: invalid qid")
        if qid in excluded_qids:
            raise PlanError(f"{label}: qid {qid!r} is excluded")
        if qid not in query_ids:
            raise PlanError(f"{label}: qid {qid!r} missing query vector cache")
        if qid in seen_qids:
            raise PlanError(f"{label}: duplicate qid {qid!r}")
        seen_qids.add(qid)
        docs = row.get("docs")
        if not isinstance(docs, list) or len(docs) < 120:
            raise PlanError(f"{label}: qid {qid!r} needs at least 120 ranked docs")
        seen_docs: set[str] = set()
        normalized_docs = []
        for index, doc in enumerate(docs, start=1):
            if not isinstance(doc, dict):
                raise PlanError(f"{label}: doc row must be object")
            if set(doc) - {"doc_id", "rank", "gain"}:
                raise PlanError(f"{label}: score cache docs may contain only doc_id/rank/gain metadata")
            doc_id = doc.get("doc_id")
            if not isinstance(doc_id, str) or not doc_id:
                raise PlanError(f"{label}: invalid doc_id")
            if doc_id not in doc_ids:
                raise PlanError(f"{label}: doc_id {doc_id!r} missing doc vector cache")
            if doc_id in seen_docs:
                raise PlanError(f"{label}: duplicate doc_id {doc_id!r} for qid {qid!r}")
            rank = parse_int_field(doc.get("rank"), label, "rank")
            if rank != index:
                raise PlanError(f"{label}: ranks must be contiguous from 1 for qid {qid!r}")
            gain = parse_int_field(doc.get("gain", 0), label, "gain")
            if gain < 0:
                raise PlanError(f"{label}: gain must be non-negative integer")
            seen_docs.add(doc_id)
            normalized_docs.append({"doc_id": doc_id, "rank": rank, "gain": gain})
        if not any(doc["gain"] > 0 for doc in normalized_docs[:120]):
            raise PlanError(f"{label}: qid {qid!r} has no positive in top120")
        normalized_rows.append({"qid": qid, "docs": normalized_docs})
    return {
        "path": label,
        "manifest_sha256": sha256_file(path),
        "dataset": dataset,
        "bits": bits,
        "score_mode": "prepared_ip",
        "top_k": top_k,
        "binding": binding,
        "provenance": provenance,
        "turboquant": {"score_mode": "prepared_ip", "bits": bits, "seed": seed},
        "vector_cache": expected_vector_binding,
        "rows": sorted(normalized_rows, key=lambda row: row["qid"]),
    }


def pair_ids(row_id: str, docs: list[dict[str, Any]]) -> list[dict[str, str]]:
    pairs = []
    positives = [doc for doc in docs if int(doc["gain"]) > 0]
    negatives = [doc for doc in docs if int(doc["gain"]) == 0]
    for pos in positives:
        for neg in negatives:
            pairs.append({"pair_id": f"{row_id}:{pos['doc_id']}:{neg['doc_id']}", "positive_doc_id": pos["doc_id"], "negative_doc_id": neg["doc_id"]})
    return sorted(pairs, key=lambda item: item["pair_id"])


def top10_row(dataset: str, bits: int, row: dict[str, Any], limit: int) -> dict[str, Any] | None:
    docs = row["docs"][:limit]
    if not any(doc["gain"] > 0 for doc in docs):
        return None
    row_id = f"{dataset}.q{bits}.top10.{row['qid']}"
    return {
        "row_id": row_id,
        "dataset": dataset,
        "bits": bits,
        "bucket": "top10_guard",
        "qid": row["qid"],
        "rank_window": [1, limit],
        "candidate_doc_ids": [doc["doc_id"] for doc in docs],
        "positive_doc_ids": [doc["doc_id"] for doc in docs if doc["gain"] > 0],
        "negative_doc_ids": [doc["doc_id"] for doc in docs if doc["gain"] == 0],
        "pair_ids": pair_ids(row_id, docs),
    }


def nf_boundary_row(dataset: str, bits: int, row: dict[str, Any], start: int, end: int) -> dict[str, Any] | None:
    if dataset != "nfcorpus" or bits != 3:
        return None
    docs = row["docs"][start - 1 : end]
    if not any(doc["gain"] > 0 for doc in docs):
        return None
    row_id = f"{dataset}.q{bits}.nf80_120.{row['qid']}"
    return {
        "row_id": row_id,
        "dataset": dataset,
        "bits": bits,
        "bucket": "nf_boundary80_120_guard",
        "qid": row["qid"],
        "rank_window": [start, end],
        "candidate_doc_ids": [doc["doc_id"] for doc in docs],
        "positive_doc_ids": [doc["doc_id"] for doc in docs if doc["gain"] > 0],
        "negative_doc_ids": [doc["doc_id"] for doc in docs if doc["gain"] == 0],
        "pair_ids": pair_ids(row_id, docs),
    }


def build_rows(score_caches: list[dict[str, Any]], top10_limit: int, nf_start: int, nf_end: int) -> list[dict[str, Any]]:
    rows = []
    for cache in sorted(score_caches, key=lambda item: (item["dataset"], item["bits"], item["path"])):
        for source_row in cache["rows"]:
            for built in (
                top10_row(cache["dataset"], cache["bits"], source_row, top10_limit),
                nf_boundary_row(cache["dataset"], cache["bits"], source_row, nf_start, nf_end),
            ):
                if built is not None:
                    rows.append(built)
    row_ids = [row["row_id"] for row in rows]
    if len(row_ids) != len(set(row_ids)):
        raise PlanError("plan: duplicate row ids")
    return sorted(rows, key=lambda row: row["row_id"])


def require_guard_coverage(rows: list[dict[str, Any]]) -> None:
    covered = {(row["dataset"], row["bits"], row["bucket"]) for row in rows}
    missing_top10 = [
        f"{dataset} q{bits}"
        for dataset in ALLOWED_DATASETS
        for bits in (3, 5)
        if (dataset, bits, "top10_guard") not in covered
    ]
    if missing_top10:
        raise PlanError(f"plan: missing top10 guard coverage for {missing_top10}")
    missing_nf_q3 = [
        dataset
        for dataset in ("nfcorpus",)
        if (dataset, 3, "nf_boundary80_120_guard") not in covered
    ]
    if missing_nf_q3:
        raise PlanError("plan: missing NFCorpus q3 boundary rank 80..120 coverage")
    if ("nfcorpus", 5, "nf_boundary80_120_guard") in covered:
        raise PlanError("plan: NFCorpus q5 boundary coverage is not allowed")


def workload(rows: list[dict[str, Any]]) -> dict[str, Any]:
    bucket_counts = Counter(str(row["bucket"]) for row in rows)
    domain_counts = Counter(str(row["dataset"]) for row in rows)
    bit_counts = Counter(str(row["bits"]) for row in rows)
    pair_count = sum(len(row["pair_ids"]) for row in rows)
    candidate_count = sum(len(row["candidate_doc_ids"]) for row in rows)
    return {
        "row_count": len(rows),
        "pair_count": pair_count,
        "candidate_reference_count": candidate_count,
        "bucket_counts": dict(sorted(bucket_counts.items())),
        "dataset_counts": dict(sorted(domain_counts.items())),
        "bit_counts": {str(k): bit_counts[k] for k in sorted(bit_counts)},
        "angle_count": TOPOLOGY["angle_count"],
        "planned_work_units": pair_count * TOPOLOGY["angle_count"],
        "actual_train_pairs": 0,
        "actual_vectors_written": 0,
        "actual_similarity_values_written": 0,
    }


def stripped_vector_cache_audit(vector_caches: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for key in sorted(vector_caches):
        item = vector_caches[key]
        out.append(
            {
                "dataset": item["dataset"],
                "role": item["role"],
                "path": item["path"],
                "manifest_sha256": item["manifest_sha256"],
                "cache_sha256": item["cache_sha256"],
                "id_count": item["id_count"],
                "binding": item["binding"],
                "provenance": item["provenance"],
            }
        )
    return out


def stripped_score_cache_audit(score_caches: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "dataset": item["dataset"],
            "bits": item["bits"],
            "path": item["path"],
            "manifest_sha256": item["manifest_sha256"],
            "score_mode": item["score_mode"],
            "top_k": item["top_k"],
            "binding": item["binding"],
            "provenance": item["provenance"],
            "turboquant": item["turboquant"],
            "vector_cache": item["vector_cache"],
            "qid_count": len(item["rows"]),
        }
        for item in sorted(score_caches, key=lambda value: (value["dataset"], value["bits"], value["path"]))
    ]


def build_plan(args: argparse.Namespace) -> dict[str, Any]:
    anchor = validate_anchor(args.anchor_manifest, args.expected_anchor_sha256)
    exclusions = validate_exclusions(args.exclusion_qids)
    excluded_qids = set(exclusions["excluded_qids"])
    vector_caches = validate_vector_caches(args.vector_cache, anchor)
    score_caches = [validate_score_cache(path, excluded_qids, vector_caches, anchor, args.seed) for path in args.score_cache]
    seen_score_keys: set[tuple[str, int]] = set()
    for cache in score_caches:
        key = (cache["dataset"], cache["bits"])
        if key in seen_score_keys:
            raise PlanError(f"score caches: duplicate {cache['dataset']} q{cache['bits']}")
        seen_score_keys.add(key)
    for dataset in ALLOWED_DATASETS:
        for bits in (3, 5):
            if (dataset, bits) not in seen_score_keys:
                raise PlanError(f"score caches: missing train {dataset} q{bits}")
    qrels_sha256_by_dataset = require_provenance_consistency(vector_caches, score_caches)
    rows = build_rows(score_caches, args.top10_limit, args.nf_start, args.nf_end)
    require_guard_coverage(rows)
    if not rows:
        raise PlanError("plan: no eligible guard rows")
    rows_sha = sha256_json(rows)
    plan = {
        "schema": SCHEMA,
        "created_utc": utc_stamp(),
        "mode": "plan_only",
        "actual_cache_generation_ran": False,
        "actual_training_ran": False,
        "actual_eval_ran": False,
        "seed": args.seed,
        "topology": TOPOLOGY,
        "legal_scope": LEGAL_SCOPE,
        "anchor": anchor,
        "exclusions": {k: v for k, v in exclusions.items() if k != "excluded_qids"},
        "input_manifests": {
            "vector_caches": stripped_vector_cache_audit(vector_caches),
            "score_caches": stripped_score_cache_audit(score_caches),
        },
        "selection_policy": {
            "split": "train",
            "datasets": list(ALLOWED_DATASETS),
            "required_score_bits": [3, 5],
            "score_mode": "prepared_ip",
            "top10_guard_limit": args.top10_limit,
            "nf_boundary_window": [args.nf_start, args.nf_end],
            "official_exclusion_mode": "qid_only",
            "forbidden_splits": list(FORBIDDEN_SPLITS),
        },
        "rows": rows,
        "row_ids_sha256": sha256_json([row["row_id"] for row in rows]),
        "rows_sha256": rows_sha,
        "workload": workload(rows),
        "provenance": {
            "builder": display_path(Path(__file__)),
            "builder_sha256": sha256_file(Path(__file__)),
            "qrels_sha256_by_dataset": qrels_sha256_by_dataset,
            "self_hash_excludes": ["created_utc", "provenance.plan_sha256"],
        },
    }
    require_output_has_no_vectors_or_numeric_scores(plan)
    plan["provenance"]["plan_sha256"] = sha256_json({k: v for k, v in plan.items() if k != "created_utc"})
    return plan


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        plan = build_plan(args)
        write_json_atomic(args.output_plan, plan)
    except PlanError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
