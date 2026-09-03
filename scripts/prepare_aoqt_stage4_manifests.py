#!/usr/bin/env python3
"""Prepare skinny AOQT Stage 4 manifests from train-only fixture artifacts.

The subcommands in this file are deliberately metadata adapters.  They may read
raw vector JSONL or per-query score JSONL to validate ids, dimensions, ranks,
hashes, and provenance, but they never copy vectors or numeric similarity
scores into the emitted AOQT manifests.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Iterable


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import build_aoqt_stage2_calibration as calibration  # noqa: E402

RETRIEVAL_EXPORT_SCHEMA = "eos.aoqt_stage4.retrieval_export_manifest.v1"
QID_ONLY_JSON_FIELDS = {
    "selected_qids",
    "qids",
    "qids_by_dataset",
    "name",
    "schema",
    "dataset",
    "split",
    "source_selected_qids",
    "source_sha256",
    "source_sha256_by_file",
}
QID_CARRIER_FIELDS = ("qids_by_dataset", "selected_qids", "qids")
LEGACY_WRAPPER_SPLITS = {"dev4", "reserve4"}


class ManifestError(ValueError):
    """Raised when a source artifact cannot be safely adapted."""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    exclusions = sub.add_parser("normalize-exclusions")
    exclusions.add_argument("--name", required=True)
    exclusions.add_argument("--legacy-selected-qids", type=Path, action="append", default=[])
    exclusions.add_argument("--official-test-qrels", type=Path, action="append", default=[])
    exclusions.add_argument("--output", type=Path, required=True)

    vectors = sub.add_parser("adapt-vector-cache")
    vectors.add_argument("--retrieval-manifest", type=Path, required=True)
    vectors.add_argument("--vector-jsonl", type=Path, required=True)
    vectors.add_argument("--anchor-manifest", type=Path, required=True)
    vectors.add_argument("--dataset", choices=calibration.ALLOWED_DATASETS, required=True)
    vectors.add_argument("--role", choices=("query", "doc"), required=True)
    vectors.add_argument("--qrels-sha256", required=True)
    vectors.add_argument("--output", type=Path, required=True)

    scores = sub.add_parser("convert-score-cache")
    scores.add_argument("--per-query-jsonl", type=Path, required=True)
    scores.add_argument("--query-vector-manifest", type=Path, required=True)
    scores.add_argument("--doc-vector-manifest", type=Path, required=True)
    scores.add_argument("--anchor-manifest", type=Path, required=True)
    scores.add_argument("--dataset", choices=calibration.ALLOWED_DATASETS, required=True)
    scores.add_argument("--bits", type=int, choices=(3, 5), required=True)
    scores.add_argument("--top-k", type=int, default=120)
    scores.add_argument("--turboquant-seed", type=int, default=calibration.DEFAULT_TURBOQUANT_SEED)
    scores.add_argument("--qrels-sha256", required=True)
    scores.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def repo_path(path: Path) -> Path:
    return path if path.is_absolute() else calibration.REPO_ROOT / path


def display_path(path: Path) -> str:
    return calibration.display_path(path)


def load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(repo_path(path).read_text(encoding="utf-8"))
    except OSError as exc:
        raise ManifestError(f"{display_path(path)}: cannot read JSON") from exc
    except json.JSONDecodeError as exc:
        raise ManifestError(f"{display_path(path)}: invalid JSON") from exc
    if not isinstance(payload, dict):
        raise ManifestError(f"{display_path(path)}: expected JSON object")
    return payload


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    target = repo_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(target)


def sha256_file(path: Path) -> str:
    return calibration.sha256_file(path)


def require_sha256(value: Any, label: str, field: str) -> str:
    return calibration.require_sha256(value, label, field)


def source_hashes(paths: Iterable[Path]) -> dict[str, Any]:
    by_file = {display_path(path): sha256_file(path) for path in paths}
    return {
        "source_sha256": calibration.sha256_json(by_file),
        "source_sha256_by_file": dict(sorted(by_file.items())),
    }


def dataset_source_provenance(component: str, source_sha256: str) -> dict[str, Any]:
    mapping = {component: source_sha256}
    return {
        "dataset_source_provenance": mapping,
        "dataset_source_provenance_sha256": calibration.sha256_json(mapping),
    }


def merged_dataset_source_provenance(*maps: dict[str, str]) -> dict[str, Any]:
    merged: dict[str, str] = {}
    for source_map in maps:
        for component, value in source_map.items():
            prior = merged.get(component)
            if prior is not None and prior != value:
                raise ManifestError(f"dataset source provenance component {component!r} mismatch")
            merged[component] = value
    return {
        "dataset_source_provenance": dict(sorted(merged.items())),
        "dataset_source_provenance_sha256": calibration.sha256_json(dict(sorted(merged.items()))),
    }


def reject_forbidden_payload_keys(node: Any, label: str) -> None:
    forbidden = {
        "doc",
        "docs",
        "docid",
        "docids",
        "doc_id",
        "doc_ids",
        "document",
        "documents",
        "corpus",
        "qrel",
        "qrels",
        "score",
        "scores",
        "text",
        "gain",
        "gains",
        "relevance",
        "top_k",
        "embedding",
        "embeddings",
        "vector",
        "vectors",
    }
    if isinstance(node, dict):
        for key, value in node.items():
            if str(key).lower() in forbidden:
                raise ManifestError(f"{label}: qid-only input contains forbidden field {key!r}")
            reject_forbidden_payload_keys(value, label)
    elif isinstance(node, list):
        for value in node:
            reject_forbidden_payload_keys(value, label)


def require_qid(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or re.search(r"\s", value):
        raise ManifestError(f"{label}: qids must be non-empty strings without whitespace")
    return value


def require_dataset(value: Any, label: str) -> str:
    try:
        return calibration.require_dataset(value, label)
    except calibration.PlanError as exc:
        raise ManifestError(str(exc)) from exc


def require_wrapper_text(value: Any, label: str, field: str) -> str:
    if not isinstance(value, str) or not value.strip() or re.search(r"[\x00-\x1f]", value):
        raise ManifestError(f"{label}: {field} must be a non-empty string metadata value")
    return value


def require_wrapper_schema(value: Any, label: str) -> None:
    schema = require_wrapper_text(value, label, "schema")
    if not re.fullmatch(r"[A-Za-z0-9_.:-]+", schema):
        raise ManifestError(f"{label}: schema must be a simple string identifier")


def require_wrapper_name(value: Any, label: str) -> None:
    name = require_wrapper_text(value, label, "name")
    if re.search(r"\s", name):
        raise ManifestError(f"{label}: name must not contain whitespace")


def require_wrapper_split(value: Any, label: str, expected_name: str) -> None:
    split = require_wrapper_text(value, label, "split")
    if split not in LEGACY_WRAPPER_SPLITS:
        raise ManifestError(f"{label}: split must be one of {sorted(LEGACY_WRAPPER_SPLITS)}")
    if split != expected_name:
        raise ManifestError(f"{label}: split must exactly match requested name {expected_name!r}")


def require_source_selected_qids(value: Any, label: str) -> None:
    if isinstance(value, str):
        require_pathish_metadata_key(value, label, "source_selected_qids")
        return
    if not isinstance(value, dict):
        raise ManifestError(f"{label}: source_selected_qids must be a string or dataset-keyed string mapping")
    unexpected = sorted(set(value) - set(calibration.ALLOWED_DATASETS))
    missing = [dataset for dataset in calibration.ALLOWED_DATASETS if dataset not in value]
    if unexpected:
        raise ManifestError(f"{label}: source_selected_qids has unsupported datasets {unexpected}")
    if missing:
        raise ManifestError(f"{label}: source_selected_qids missing required dataset coverage {missing}")
    for dataset in calibration.ALLOWED_DATASETS:
        require_pathish_metadata_key(value[dataset], label, f"source_selected_qids.{dataset}")


def require_pathish_metadata_key(value: Any, label: str, field: str) -> str:
    key = require_wrapper_text(value, label, f"{field} key")
    parts = [part for part in key.split("/") if part]
    if "\\" in key or re.search(r"\s", key) or not parts or any(part in {".", ".."} for part in parts):
        raise ManifestError(f"{label}: {field} keys must be path/name strings")
    if "/" not in key and "." not in key:
        raise ManifestError(f"{label}: {field} keys must be path/name strings, not arbitrary fields")
    return key


def require_sha256_mapping(value: Any, label: str, field: str) -> None:
    if not isinstance(value, dict) or not value:
        raise ManifestError(f"{label}: {field} must be a non-empty sha256 mapping")
    for key, item in value.items():
        clean_key = require_pathish_metadata_key(key, label, field)
        try:
            calibration.require_sha256(item, label, f"{field}.{clean_key}")
        except calibration.PlanError as exc:
            raise ManifestError(str(exc)) from exc


def validate_legacy_qid_wrapper(payload: dict[str, Any], label: str, expected_name: str) -> list[dict[str, list[str]]]:
    reject_forbidden_payload_keys(payload, label)
    unexpected = sorted(set(payload) - QID_ONLY_JSON_FIELDS)
    if unexpected:
        raise ManifestError(f"{label}: qid-only JSON has unexpected fields {unexpected}")
    if expected_name not in LEGACY_WRAPPER_SPLITS:
        raise ManifestError(f"{label}: legacy selected_qids inputs are only allowed for {sorted(LEGACY_WRAPPER_SPLITS)}")
    if "split" not in payload:
        raise ManifestError(f"{label}: legacy selected_qids input must declare split {expected_name!r}")

    if "schema" in payload:
        require_wrapper_schema(payload["schema"], label)
    if "name" in payload:
        require_wrapper_name(payload["name"], label)
    if "dataset" in payload:
        require_dataset(payload["dataset"], label)
    require_wrapper_split(payload["split"], label, expected_name)
    if "source_selected_qids" in payload:
        require_source_selected_qids(payload["source_selected_qids"], label)
    if "source_sha256" in payload:
        try:
            calibration.require_sha256(payload["source_sha256"], label, "source_sha256")
        except calibration.PlanError as exc:
            raise ManifestError(str(exc)) from exc
    if "source_sha256_by_file" in payload:
        require_sha256_mapping(payload["source_sha256_by_file"], label, "source_sha256_by_file")

    present = [field for field in QID_CARRIER_FIELDS if field in payload]
    if not present:
        raise ManifestError(f"{label}: expected selected_qids/qids mapping by dataset")
    mappings = [qid_mapping_from_value(payload[field], f"{label}: {field}") for field in present]
    first = {dataset: sorted(qids) for dataset, qids in mappings[0].items()}
    for field, mapping in zip(present[1:], mappings[1:]):
        comparable = {dataset: sorted(qids) for dataset, qids in mapping.items()}
        if comparable != first:
            raise ManifestError(f"{label}: qid carrier {field!r} does not match other qid carriers")
    return mappings


def infer_dataset_from_path(path: Path) -> str:
    label = display_path(path)
    compact = re.sub(r"[^a-z0-9]+", "", label.lower())
    matches = [dataset for dataset in calibration.ALLOWED_DATASETS if dataset in compact]
    if len(matches) != 1:
        raise ManifestError(f"{label}: official qrels path must identify exactly one dataset")
    return matches[0]


def qid_mapping_from_value(value: Any, label: str) -> dict[str, list[str]]:
    if isinstance(value, dict):
        unexpected = sorted(set(value) - set(calibration.ALLOWED_DATASETS))
        if unexpected:
            raise ManifestError(f"{label}: unsupported qid datasets {unexpected}")
        missing = [dataset for dataset in calibration.ALLOWED_DATASETS if dataset not in value]
        if missing:
            raise ManifestError(f"{label}: missing required dataset coverage {missing}")
        out: dict[str, list[str]] = {}
        for key in calibration.ALLOWED_DATASETS:
            qids = value[key]
            qid_list = [require_qid(item, f"{label}: {key}") for item in qids] if isinstance(qids, list) else None
            if qid_list is None or not qid_list:
                raise ManifestError(f"{label}: {key} qids must be a non-empty list")
            if len(qid_list) != len(set(qid_list)):
                raise ManifestError(f"{label}: duplicate qids for dataset {key}")
            out[require_dataset(key, label)] = qid_list
        return out
    if isinstance(value, list):
        raise ManifestError(f"{label}: ambiguous legacy global qids require an explicit dataset mapping")
    raise ManifestError(f"{label}: expected selected_qids/qids mapping by dataset")


def selected_qids_from_json(path: Path, expected_name: str) -> dict[str, list[str]]:
    payload = load_json(path)
    label = display_path(path)
    mappings = validate_legacy_qid_wrapper(payload, label, expected_name)
    return mappings[0]


def is_qrels_header(tokens: list[str]) -> bool:
    return len(tokens) >= 2 and tokens[0] == "query-id" and any(token in {"corpus-id", "doc-id", "score", "relevance"} for token in tokens[1:])


def selected_qids_from_qrels(path: Path) -> dict[str, list[str]]:
    qids: list[str] = []
    seen: set[str] = set()
    label = display_path(path)
    dataset = infer_dataset_from_path(path)
    with repo_path(path).open("r", encoding="utf-8") as handle:
        data_row_index = 0
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text or text.startswith("#"):
                continue
            tokens = text.split()
            if data_row_index == 0 and is_qrels_header(tokens):
                data_row_index += 1
                continue
            data_row_index += 1
            qid = tokens[0]
            qid = require_qid(qid, f"{label}:{line_number}")
            if qid not in seen:
                seen.add(qid)
                qids.append(qid)
    if not qids:
        raise ManifestError(f"{label}: no qids found in qrels first column")
    return {dataset: qids}


def validate_exclusion_source_binding(args: argparse.Namespace) -> None:
    has_legacy = bool(args.legacy_selected_qids)
    has_official = bool(args.official_test_qrels)
    if not has_legacy and not has_official:
        raise ManifestError("normalize-exclusions: at least one qid source is required")
    if has_legacy and has_official:
        raise ManifestError("normalize-exclusions: legacy_selected_qids and official_test_qrels source classes must not be mixed")
    if has_official:
        if args.name != "official-test":
            raise ManifestError("normalize-exclusions: official_test_qrels inputs are only allowed for name 'official-test'")
        expected_count = len(calibration.ALLOWED_DATASETS)
        if len(args.official_test_qrels) != expected_count:
            raise ManifestError(f"normalize-exclusions: official-test requires exactly {expected_count} official qrels inputs")
    if has_legacy:
        if args.name not in LEGACY_WRAPPER_SPLITS:
            raise ManifestError(f"normalize-exclusions: legacy_selected_qids inputs are only allowed for {sorted(LEGACY_WRAPPER_SPLITS)}")
        if len(args.legacy_selected_qids) != 1:
            raise ManifestError("normalize-exclusions: dev4/reserve4 require exactly one legacy selected_qids wrapper")


def merge_qids_by_dataset(sources: Iterable[dict[str, list[str]]]) -> dict[str, list[str]]:
    merged: dict[str, list[str]] = {dataset: [] for dataset in calibration.ALLOWED_DATASETS}
    seen: dict[str, set[str]] = {dataset: set() for dataset in calibration.ALLOWED_DATASETS}
    for source in sources:
        for dataset, qids in source.items():
            dataset_name = require_dataset(dataset, "normalize-exclusions")
            for qid in qids:
                if qid in seen[dataset_name]:
                    raise ManifestError(f"normalize-exclusions: duplicate qids for dataset {dataset_name}")
                seen[dataset_name].add(qid)
                merged[dataset_name].append(qid)
    missing = [dataset for dataset in calibration.ALLOWED_DATASETS if not merged[dataset]]
    if missing:
        raise ManifestError(f"normalize-exclusions: missing required dataset coverage {missing}")
    return {dataset: sorted(merged[dataset]) for dataset in calibration.ALLOWED_DATASETS}


def normalize_exclusions(args: argparse.Namespace) -> dict[str, Any]:
    validate_exclusion_source_binding(args)
    sources: list[dict[str, list[str]]] = []
    for path in args.legacy_selected_qids:
        sources.append(selected_qids_from_json(path, args.name))
    for path in args.official_test_qrels:
        sources.append(selected_qids_from_qrels(path))
    qids_by_dataset = merge_qids_by_dataset(sources)
    payload = {
        "schema": calibration.EXCLUSION_SCHEMA,
        "name": args.name,
        "qids_by_dataset": qids_by_dataset,
        **source_hashes([*args.legacy_selected_qids, *args.official_test_qrels]),
    }
    return payload


def anchor_binding_from(anchor: dict[str, Any]) -> dict[str, Any]:
    return {
        "anchor": {
            "package_sha256": anchor["package_sha256"],
            "manifest_sha256": anchor["manifest_sha256"],
        },
        "topology": {"id": calibration.TOPOLOGY["id"], "sha256": calibration.TOPOLOGY_SHA256},
        "legal_scope": calibration.LEGAL_SCOPE,
    }


def anchor_binding(anchor_path: Path) -> dict[str, Any]:
    return anchor_binding_from(calibration.validate_anchor(anchor_path, None))


def scalar_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def vector_row_id(row: dict[str, Any], role: str, label: str) -> str:
    keys = ("id", "_id", "query_id") if role == "query" else ("id", "_id", "doc_id", "document_id")
    for key in keys:
        value = row.get(key)
        if isinstance(value, str) and value:
            return value
    raise ManifestError(f"{label}: missing vector id")


def vector_values(row: dict[str, Any], label: str) -> list[Any]:
    for key in ("embedding", "vector", "values"):
        value = row.get(key)
        if isinstance(value, list):
            return value
    raise ManifestError(f"{label}: missing vector values")


def read_vector_jsonl(path: Path, role: str, expected_dim: int = 384) -> list[str]:
    ids: list[str] = []
    seen: set[str] = set()
    label = display_path(path)
    with repo_path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ManifestError(f"{label}:{line_number}: invalid JSONL row") from exc
            if not isinstance(row, dict):
                raise ManifestError(f"{label}:{line_number}: expected object row")
            item_id = vector_row_id(row, role, f"{label}:{line_number}")
            values = vector_values(row, f"{label}:{line_number}")
            if len(values) != expected_dim or not all(scalar_number(value) for value in values):
                raise ManifestError(f"{label}:{line_number}: vector dimension/value check failed")
            if item_id in seen:
                raise ManifestError(f"{label}:{line_number}: duplicate vector id {item_id!r}")
            seen.add(item_id)
            ids.append(item_id)
    if not ids:
        raise ManifestError(f"{label}: vector JSONL must contain at least one row")
    return ids


def require_retrieval_manifest(payload: dict[str, Any], args: argparse.Namespace, anchor: dict[str, Any]) -> None:
    label = display_path(args.retrieval_manifest)
    calibration.require_schema(payload, RETRIEVAL_EXPORT_SCHEMA, label)
    calibration.require_no_forbidden_split_tokens(payload, label)
    calibration.require_train_split(require_field(payload, "split", label), label)
    dataset = str(require_field(payload, "dataset", label)).lower()
    if dataset != args.dataset:
        raise ManifestError(f"{label}: dataset mismatch")
    dim = require_field(payload, "dimension", label)
    if dim != calibration.TOPOLOGY["dim"]:
        raise ManifestError(f"{label}: embedding dimension must be {calibration.TOPOLOGY['dim']}")
    anchor_node = require_field(payload, "anchor", label)
    if not isinstance(anchor_node, dict):
        raise ManifestError(f"{label}: anchor identity object is required")
    if anchor_node.get("package_sha256") != anchor["package_sha256"]:
        raise ManifestError(f"{label}: anchor package_sha256 identity mismatch")
    if anchor_node.get("manifest_sha256") != anchor["manifest_sha256"]:
        raise ManifestError(f"{label}: anchor manifest_sha256 identity mismatch")


def adapt_vector_cache(args: argparse.Namespace) -> dict[str, Any]:
    require_sha256(args.qrels_sha256, "adapt-vector-cache", "qrels_sha256")
    anchor = calibration.validate_anchor(args.anchor_manifest, None)
    retrieval_manifest = load_json(args.retrieval_manifest)
    require_retrieval_manifest(retrieval_manifest, args, anchor)
    ids = read_vector_jsonl(args.vector_jsonl, args.role, calibration.TOPOLOGY["dim"])
    sources = source_hashes([args.retrieval_manifest, args.vector_jsonl])
    payload = {
        "schema": calibration.VECTOR_CACHE_SCHEMA,
        "dataset": args.dataset,
        "split": "train",
        "role": args.role,
        "cache_sha256": sha256_file(args.vector_jsonl),
        "ids": ids,
        "qrels_sha256": args.qrels_sha256,
        **sources,
        **dataset_source_provenance(f"{args.role}_vector", sources["source_sha256"]),
        **anchor_binding_from(anchor),
    }
    calibration.require_output_has_no_vectors_or_numeric_scores(payload)
    return payload


def load_vector_manifest(path: Path, role: str, dataset: str, anchor: dict[str, Any]) -> dict[str, Any]:
    payload = load_json(path)
    label = display_path(path)
    calibration.require_schema(payload, calibration.VECTOR_CACHE_SCHEMA, label)
    if payload.get("dataset") != dataset or payload.get("role") != role:
        raise ManifestError(f"{label}: expected {dataset}:{role} vector manifest")
    calibration.require_train_split(payload.get("split"), label)
    calibration.require_anchor_topology_legal_binding(payload, label, anchor)
    ids = payload.get("ids")
    if not isinstance(ids, list) or not all(isinstance(item, str) and item for item in ids):
        raise ManifestError(f"{label}: ids must be non-empty strings")
    provenance = calibration.require_provenance_hashes(payload, label)
    component = f"{role}_vector"
    dataset_sources = provenance.get("dataset_source_provenance")
    if not isinstance(dataset_sources, dict) or dataset_sources.get(component) != provenance.get("source_sha256"):
        raise ManifestError(f"{label}: {component} dataset source provenance mismatch")
    return {
        "manifest_sha256": sha256_file(path),
        "cache_sha256": require_sha256(payload.get("cache_sha256"), label, "cache_sha256"),
        "ids": set(ids),
        "provenance": provenance,
    }


def candidate_doc(item: dict[str, Any], label: str, index: int) -> dict[str, int | str]:
    unexpected = set(item) - {"doc_id", "document_id", "id", "rank", "gain", "relevance", "score", "similarity", "prepared_ip_score"}
    if unexpected:
        raise ManifestError(f"{label}: unexpected candidate fields {sorted(unexpected)}")
    doc_id = item.get("doc_id", item.get("document_id", item.get("id")))
    if not isinstance(doc_id, str) or not doc_id:
        raise ManifestError(f"{label}: invalid doc_id")
    rank = item.get("rank", index)
    if rank != index:
        raise ManifestError(f"{label}: ranks must be contiguous from 1")
    gain = item.get("gain", item.get("relevance", 0))
    if isinstance(gain, bool) or not isinstance(gain, int) or gain < 0:
        raise ManifestError(f"{label}: gain/relevance must be a non-negative integer")
    return {"doc_id": doc_id, "rank": index, "gain": gain}


def require_optional_train_split(row: dict[str, Any], label: str) -> None:
    if "split" in row:
        calibration.require_train_split(row.get("split"), label)


def require_field(row: dict[str, Any], field: str, label: str) -> Any:
    if field not in row:
        raise ManifestError(f"{label}: missing required source field {field!r}")
    return row[field]


def require_source_row_metadata(row: dict[str, Any], label: str, dataset: str, bits: int, turboquant_seed: int, top_k: int) -> None:
    source_dataset = require_field(row, "dataset", label)
    if source_dataset != dataset:
        raise ManifestError(f"{label}: source dataset mismatch")
    calibration.require_train_split(require_field(row, "split", label), label)
    source_bits = require_field(row, "bits", label)
    if source_bits != bits:
        raise ManifestError(f"{label}: source bits mismatch")
    source_seed = require_field(row, "quantizer_seed", label)
    if source_seed != turboquant_seed:
        raise ManifestError(f"{label}: source quantizer seed mismatch")
    surface = require_field(row, "scoring_surface", label)
    if surface != "turboquant_ip_prepared":
        raise ManifestError(f"{label}: scoring_surface must be turboquant_ip_prepared")
    if require_optional_zero_int(row, label, "rerank_overfetch") != 0 or require_optional_zero_int(row, label, "rerank_bits") != 0 or str(row.get("rerank_storage", "") or ""):
        raise ManifestError(f"{label}: rerank source rows are not allowed")
    per_query_top_k = require_field(row, "top_k_limit", label)
    if per_query_top_k != top_k:
        raise ManifestError(f"{label}: source per-query top_k must be {top_k}")


def require_optional_zero_int(row: dict[str, Any], label: str, field: str) -> int:
    value = row.get(field, 0)
    if value is None:
        return 0
    if isinstance(value, bool) or not isinstance(value, int):
        raise ManifestError(f"{label}: {field} must be integer 0 when present")
    return value


def read_score_rows(path: Path, top_k: int, query_ids: set[str], doc_ids: set[str], dataset: str, bits: int, turboquant_seed: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen_qids: set[str] = set()
    label = display_path(path)
    with repo_path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ManifestError(f"{label}:{line_number}: invalid JSONL row") from exc
            if not isinstance(row, dict):
                raise ManifestError(f"{label}:{line_number}: expected object row")
            require_source_row_metadata(row, f"{label}:{line_number}", dataset, bits, turboquant_seed, top_k)
            qid = row.get("qid", row.get("query_id"))
            if not isinstance(qid, str) or not qid:
                raise ManifestError(f"{label}:{line_number}: invalid qid")
            if qid not in query_ids:
                raise ManifestError(f"{label}:{line_number}: qid {qid!r} missing query vector manifest")
            if qid in seen_qids:
                raise ManifestError(f"{label}:{line_number}: duplicate qid {qid!r}")
            if "docs" in row:
                raise ManifestError(f"{label}:{line_number}: source rows must use direct top_k candidate shape")
            candidates = row.get("top_k")
            if not isinstance(candidates, list) or len(candidates) != top_k:
                raise ManifestError(f"{label}:{line_number}: expected exactly {top_k} top_k candidates")
            docs = [candidate_doc(item, f"{label}:{line_number}", index) for index, item in enumerate(candidates, start=1) if isinstance(item, dict)]
            if len(docs) != top_k:
                raise ManifestError(f"{label}:{line_number}: top_k candidates must be objects")
            if len({doc["doc_id"] for doc in docs}) != len(docs):
                raise ManifestError(f"{label}:{line_number}: duplicate doc_id in top_k")
            missing = [doc["doc_id"] for doc in docs if doc["doc_id"] not in doc_ids]
            if missing:
                raise ManifestError(f"{label}:{line_number}: doc_id {missing[0]!r} missing doc vector manifest")
            seen_qids.add(qid)
            rows.append({"qid": qid, "docs": docs})
    if not rows:
        raise ManifestError(f"{label}: score JSONL must contain at least one row")
    return sorted(rows, key=lambda item: item["qid"])


def convert_score_cache(args: argparse.Namespace) -> dict[str, Any]:
    if args.top_k != 120:
        raise ManifestError("convert-score-cache: top_k must be exactly 120 for AOQT Stage4")
    require_sha256(args.qrels_sha256, "convert-score-cache", "qrels_sha256")
    anchor = calibration.validate_anchor(args.anchor_manifest, None)
    query_cache = load_vector_manifest(args.query_vector_manifest, "query", args.dataset, anchor)
    doc_cache = load_vector_manifest(args.doc_vector_manifest, "doc", args.dataset, anchor)
    for label, cache in (("query", query_cache), ("doc", doc_cache)):
        qrels = cache["provenance"].get("qrels_sha256")
        if qrels != args.qrels_sha256:
            raise ManifestError(f"convert-score-cache: {label} vector qrels provenance mismatch")
    rows = read_score_rows(args.per_query_jsonl, args.top_k, query_cache["ids"], doc_cache["ids"], args.dataset, args.bits, args.turboquant_seed)
    sources = source_hashes([args.per_query_jsonl])
    query_sources = query_cache["provenance"]["dataset_source_provenance"]
    doc_sources = doc_cache["provenance"]["dataset_source_provenance"]
    score_sources = dataset_source_provenance(f"q{args.bits}_score", sources["source_sha256"])["dataset_source_provenance"]
    payload = {
        "schema": calibration.SCORE_CACHE_SCHEMA,
        "dataset": args.dataset,
        "split": "train",
        "bits": args.bits,
        "score_mode": "prepared_ip",
        "top_k": args.top_k,
        "turboquant": {"score_mode": "prepared_ip", "bits": args.bits, "seed": args.turboquant_seed},
        "vector_cache": {
            "query_manifest_sha256": query_cache["manifest_sha256"],
            "query_cache_sha256": query_cache["cache_sha256"],
            "doc_manifest_sha256": doc_cache["manifest_sha256"],
            "doc_cache_sha256": doc_cache["cache_sha256"],
        },
        "qrels_sha256": args.qrels_sha256,
        **sources,
        **merged_dataset_source_provenance(query_sources, doc_sources, score_sources),
        **anchor_binding(args.anchor_manifest),
        "rows": rows,
    }
    calibration.require_output_has_no_vectors_or_numeric_scores(payload)
    return payload


def build_manifest(args: argparse.Namespace) -> dict[str, Any]:
    if args.command == "normalize-exclusions":
        return normalize_exclusions(args)
    if args.command == "adapt-vector-cache":
        return adapt_vector_cache(args)
    if args.command == "convert-score-cache":
        return convert_score_cache(args)
    raise ManifestError(f"unknown command {args.command!r}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        payload = build_manifest(args)
        calibration.require_output_has_no_vectors_or_numeric_scores(payload)
        write_json_atomic(args.output, payload)
    except ManifestError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except calibration.PlanError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
