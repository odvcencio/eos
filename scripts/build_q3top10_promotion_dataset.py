#!/usr/bin/env python3
"""Derive a q3 top10 promotion+guard research dataset from reviewed rows."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


SCHEMA = "eos.q3top10_promotion_guard_dataset.v1"
LEGAL_GATES = {
    "train_allowed_for_research": True,
    "release_train_allowed": False,
    "commercial_use_allowed": False,
}
Q3_METHOD = "turboquant_ip_b3"
Q3_BITS = 3
Q3_SEED = 5581486560434873699
Q3_SURFACE = "turboquant_ip_prepared"
ALIGNED_CANDIDATE_FIELDS = (
    "candidate_doc_ids",
    "candidate_texts",
    "candidate_sources",
    "qrel_gains",
    "hard_negative_eligible",
    "target_probabilities",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-train-jsonl", required=True, type=Path)
    parser.add_argument("--parent-manifest", required=True, type=Path)
    parser.add_argument("--q3-per-query-jsonl", action="append", required=True, type=Path)
    parser.add_argument("--exclude-qids-json", action="append", default=[], type=Path)
    parser.add_argument("--test-qrels", action="append", default=[], type=Path)
    parser.add_argument("--output-jsonl", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--coverage-report", required=True, type=Path)
    parser.add_argument("--promotion-max-score-gap", type=float, default=0.03)
    parser.add_argument("--guard-count-per-dataset", type=int, default=6)
    parser.add_argument("--guard-max-total", type=int, default=12)
    parser.add_argument("--q3-negative-primary-rank-cutoff", type=int, default=12)
    parser.add_argument("--q3-negative-fill-rank-cutoff", type=int, default=20)
    parser.add_argument("--min-q3-negatives", type=int, default=4)
    parser.add_argument("--expected-promotion-rows", type=int, default=0)
    parser.add_argument("--expected-rows", type=int, default=0)
    parser.add_argument("--promotion-turboquant-topk-loss-weight", type=float, default=None)
    parser.add_argument("--guard-turboquant-topk-loss-weight", type=float, default=None)
    return parser.parse_args()


def utc_stamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_strings(values: list[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(value.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                yield line_number, json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")


def doc_id_for_dataset(dataset: str, candidate_doc_id: str) -> str:
    prefix = f"{dataset}:"
    if candidate_doc_id.startswith(prefix):
        return candidate_doc_id[len(prefix) :]
    return candidate_doc_id


def dataset_from_qrels_path(path: Path) -> str:
    parts = path.parts
    if "qrels" not in parts:
        raise ValueError(f"cannot infer dataset from qrels path: {path}")
    idx = parts.index("qrels")
    if idx < 1:
        raise ValueError(f"cannot infer dataset from qrels path: {path}")
    return parts[idx - 1].lower()


def read_qrel_qids(path: Path) -> tuple[str, set[str]]:
    dataset = dataset_from_qrels_path(path)
    qids: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        header = next(handle, "")
        if not header.lower().startswith("query-id\t"):
            raise ValueError(f"{path}: expected BEIR qrels TSV header")
        for raw in handle:
            fields = raw.rstrip("\n").split("\t")
            if len(fields) >= 3 and fields[0]:
                qids.add(fields[0])
    return dataset, qids


def read_excluded_qids(paths: list[Path]) -> dict[str, set[str]]:
    excluded: dict[str, set[str]] = defaultdict(set)
    for path in paths:
        payload = read_json(path)
        selected = payload.get("selected_qids", payload)
        if not isinstance(selected, dict):
            raise ValueError(f"{path}: expected selected_qids object")
        for dataset, qids in selected.items():
            if not isinstance(qids, list):
                raise ValueError(f"{path}: selected_qids.{dataset} must be a list")
            excluded[str(dataset).lower()].update(str(qid) for qid in qids)
    return excluded


def parse_manifest_q3_hashes(parent_manifest: dict[str, Any]) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for value in parent_manifest.get("q3_source_hashes", []):
        if not isinstance(value, str) or ":" not in value:
            raise ValueError("parent manifest q3_source_hashes must be path:sha256 strings")
        path, digest = value.rsplit(":", 1)
        hashes[path] = digest
    return hashes


def source_identity(path: Path, digest: str) -> dict[str, str]:
    return {
        "supplied_path": str(path),
        "absolute_path": str(path.resolve()),
        "sha256": digest,
    }


def validate_parent_inputs(parent_train: Path, parent_manifest_path: Path) -> dict[str, Any]:
    if not parent_train.is_file():
        raise ValueError(f"missing parent train JSONL: {parent_train}")
    if not parent_manifest_path.is_file():
        raise ValueError(f"missing parent manifest: {parent_manifest_path}")
    manifest = read_json(parent_manifest_path)
    expected_train_sha = manifest.get("sha256", {}).get("train_score_spectrum")
    actual_train_sha = sha256_file(parent_train)
    if expected_train_sha and expected_train_sha != actual_train_sha:
        raise ValueError(
            f"parent train sha mismatch: manifest={expected_train_sha} actual={actual_train_sha}"
        )
    if manifest.get("legal_gates") != LEGAL_GATES:
        raise ValueError("parent manifest legal gates are not fail-closed research-only gates")
    return manifest


def load_q3_lookup(
    paths: list[Path],
    parent_manifest: dict[str, Any],
) -> tuple[dict[tuple[str, str], dict[str, dict[str, Any]]], dict[str, Any]]:
    expected_hashes_by_path = parse_manifest_q3_hashes(parent_manifest)
    expected_digests = list(expected_hashes_by_path.values())
    if len(expected_digests) != len(set(expected_digests)):
        raise ValueError("parent manifest q3 source hashes contain duplicate digests")
    actual_hashes = [sha256_file(path) for path in paths]
    if len(actual_hashes) != len(set(actual_hashes)):
        raise ValueError("supplied q3 source hashes contain duplicate digests")
    if expected_digests and set(actual_hashes) != set(expected_digests):
        raise ValueError(
            "q3 source digest set does not match parent manifest: "
            f"supplied={sorted(actual_hashes)} parent={sorted(expected_digests)}"
        )

    lookup: dict[tuple[str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    rows_by_path: dict[str, Any] = {
        "parent_sources": [
            {"parent_path": path, "sha256": digest} for path, digest in sorted(expected_hashes_by_path.items())
        ],
        "supplied_sources": [],
    }
    for path in paths:
        path_digest = sha256_file(path)
        count = 0
        datasets = Counter()
        for line_number, row in iter_jsonl(path):
            if row.get("method") != Q3_METHOD:
                raise ValueError(f"{path}:{line_number}: unexpected q3 method")
            if int(row.get("bits") or 0) != Q3_BITS:
                raise ValueError(f"{path}:{line_number}: unexpected q3 bits")
            if int(row.get("quantizer_seed") or 0) != Q3_SEED:
                raise ValueError(f"{path}:{line_number}: unexpected q3 seed")
            if str(row.get("scoring_surface", "")) != Q3_SURFACE:
                raise ValueError(f"{path}:{line_number}: unexpected q3 scoring surface")
            dataset = str(row.get("dataset", "")).lower()
            query_id = str(row.get("query_id", ""))
            if not dataset or not query_id:
                raise ValueError(f"{path}:{line_number}: missing dataset/query_id")
            for candidate in row.get("top_k", []):
                doc_id = str(candidate.get("doc_id", ""))
                rank = int(candidate.get("rank") or 0)
                if not doc_id or rank <= 0:
                    continue
                if doc_id not in lookup[(dataset, query_id)]:
                    lookup[(dataset, query_id)][doc_id] = {
                        "doc_id": doc_id,
                        "q3_rank": rank,
                        "q3_score": candidate.get("score"),
                        "relevance": int(candidate.get("relevance") or 0),
                        "dense_rank": candidate.get("dense_rank"),
                        "dense_score": candidate.get("dense_score"),
                        "compact_rank": candidate.get("compact_rank"),
                        "compact_score": candidate.get("compact_score"),
                        "source_path": str(path),
                    }
            datasets[dataset] += 1
            count += 1
        identity = source_identity(path, path_digest)
        identity.update({"rows": count, "datasets": dict(datasets)})
        rows_by_path["supplied_sources"].append(identity)
    rows_by_path["supplied_sources"].sort(key=lambda item: (item["sha256"], item["absolute_path"]))
    return lookup, rows_by_path


def validate_parent_row(row: dict[str, Any], line_number: int) -> None:
    sizes = {}
    for field in ALIGNED_CANDIDATE_FIELDS:
        value = row.get(field)
        if not isinstance(value, list):
            raise ValueError(f"parent line {line_number}: {field} must be a list")
        sizes[field] = len(value)
    if len(set(sizes.values())) != 1:
        raise ValueError(f"parent line {line_number}: aligned field length mismatch {sizes}")
    if row.get("legal_gates") != LEGAL_GATES:
        raise ValueError(f"parent line {line_number}: row legal gates are not fail-closed")
    if row.get("train_allowed_for_research") is not True:
        raise ValueError(f"parent line {line_number}: train_allowed_for_research must be true")
    if row.get("release_train_allowed") is not False or row.get("commercial_use_allowed") is not False:
        raise ValueError(f"parent line {line_number}: release/commercial gates must be false")


def q3_meta(lookup: dict[tuple[str, str], dict[str, dict[str, Any]]], dataset: str, qid: str, doc_id: str) -> dict[str, Any] | None:
    return lookup.get((dataset, qid), {}).get(doc_id)


def aligned_parent_hash(row: dict[str, Any]) -> str:
    payload = json.dumps(row, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_derived_row(
    row: dict[str, Any],
    line_number: int,
    lookup: dict[tuple[str, str], dict[str, dict[str, Any]]],
    bucket: str,
    guard_margin: float | None,
    retained_negative_doc_ids: set[str],
    turboquant_topk_loss_weight: float | None,
) -> dict[str, Any]:
    dataset = str(row["source_dataset"]).lower()
    qid = str(row["source_query_id"])
    retained_indexes: list[int] = []
    retained_q3_evidence: list[dict[str, Any]] = []
    for index, candidate_doc_id in enumerate(row["candidate_doc_ids"]):
        source = row["candidate_sources"][index]
        gain = float(row["qrel_gains"][index])
        raw_doc_id = doc_id_for_dataset(dataset, str(candidate_doc_id))
        evidence = q3_meta(lookup, dataset, qid, raw_doc_id)
        rank = int((evidence or {}).get("q3_rank") or 0)
        keep_positive = source == "qrel" and gain > 0.0 and 1 <= rank <= 20
        keep_negative = source == "q3" and gain == 0.0 and raw_doc_id in retained_negative_doc_ids
        if keep_positive and int((evidence or {}).get("relevance") or 0) <= 0:
            raise ValueError(f"{row['row_id']}: retained qrel positive {raw_doc_id} has nonpositive q3 relevance")
        if keep_negative and int((evidence or {}).get("relevance") or 0) > 0:
            raise ValueError(f"{row['row_id']}: retained q3 negative {raw_doc_id} has positive q3 relevance")
        if not (keep_positive or keep_negative):
            continue
        retained_indexes.append(index)
        retained_q3_evidence.append(dict(evidence or {}))

    if not retained_indexes:
        raise ValueError(f"{row['row_id']}: no candidates retained")
    old_to_new = {old: new for new, old in enumerate(retained_indexes)}
    derived = dict(row)
    for field in ALIGNED_CANDIDATE_FIELDS:
        derived[field] = [row[field][old] for old in retained_indexes]
    positive_indexes = [
        old_to_new[old]
        for old in retained_indexes
        if row["candidate_sources"][old] == "qrel" and float(row["qrel_gains"][old]) > 0.0
    ]
    negative_indexes = [
        old_to_new[old]
        for old in retained_indexes
        if row["candidate_sources"][old] == "q3" and float(row["qrel_gains"][old]) == 0.0
    ]
    if not positive_indexes or not negative_indexes:
        raise ValueError(f"{row['row_id']}: retained row requires at least one positive and one q3 negative")
    gain_sum = sum(float(derived["qrel_gains"][index]) for index in positive_indexes)
    derived["target_probabilities"] = [0.0] * len(retained_indexes)
    for index in positive_indexes:
        derived["target_probabilities"][index] = (
            float(derived["qrel_gains"][index]) / gain_sum if gain_sum > 0.0 else 1.0 / len(positive_indexes)
        )
    derived["positive_indexes"] = positive_indexes
    derived["selected_positive_index"] = positive_indexes[0]
    derived["source_positive_doc_ids"] = [
        doc_id_for_dataset(dataset, str(derived["candidate_doc_ids"][index])) for index in positive_indexes
    ]
    derived["candidate_positive_doc_ids_pre_dedupe"] = list(derived["source_positive_doc_ids"])
    derived["positive_count"] = len(positive_indexes)
    derived["q3_negative_count"] = len(negative_indexes)
    derived["bm25_negative_count"] = 0
    derived["extra_negative_count"] = 0
    derived["q3_evidence"] = [
        retained_q3_evidence[i]
        for i, old in enumerate(retained_indexes)
        if row["candidate_sources"][old] == "q3" and float(row["qrel_gains"][old]) == 0.0
    ]
    derived["anchor_q3_candidate_evidence"] = retained_q3_evidence
    derived["parent_audit_provenance"] = {
        "parent_line_number": line_number,
        "parent_row_sha256": aligned_parent_hash(row),
        "parent_candidate_count": len(row["candidate_doc_ids"]),
        "parent_positive_indexes": list(row.get("positive_indexes", [])),
        "parent_source_positive_doc_ids": list(row.get("source_positive_doc_ids", [])),
        "parent_candidate_positive_doc_ids_pre_dedupe": list(row.get("candidate_positive_doc_ids_pre_dedupe", [])),
        "parent_all_train_qrel_positive_doc_ids": list(row.get("all_train_qrel_positive_doc_ids", [])),
    }
    derived["source"] = f"{dataset}:beir-train:q3top10-promotion-guard-derived-v1"
    derived["train_policy"] = "q3top10_promotion_guard_research_only_parent_reviewed"
    derived["selection_bucket"] = bucket
    derived["guard_margin"] = guard_margin
    if turboquant_topk_loss_weight is not None:
        if turboquant_topk_loss_weight < 0.0 or not math.isfinite(turboquant_topk_loss_weight):
            raise ValueError("turboquant_topk_loss_weight must be finite and nonnegative")
        derived["turboquant_topk_loss_weight"] = turboquant_topk_loss_weight
    derived["legal_gates"] = dict(LEGAL_GATES)
    derived.update(LEGAL_GATES)
    return derived


def row_features(
    row: dict[str, Any],
    lookup: dict[tuple[str, str], dict[str, dict[str, Any]]],
    promotion_max_score_gap: float,
    q3_negative_primary_rank_cutoff: int,
    q3_negative_fill_rank_cutoff: int,
    min_q3_negatives: int,
) -> dict[str, Any]:
    dataset = str(row["source_dataset"]).lower()
    qid = str(row["source_query_id"])
    positives: list[dict[str, Any]] = []
    negatives: list[dict[str, Any]] = []
    for index, candidate_doc_id in enumerate(row["candidate_doc_ids"]):
        raw_doc_id = doc_id_for_dataset(dataset, str(candidate_doc_id))
        evidence = q3_meta(lookup, dataset, qid, raw_doc_id)
        rank = int((evidence or {}).get("q3_rank") or 0)
        score = evidence.get("q3_score") if evidence else None
        item = {"index": index, "doc_id": raw_doc_id, "rank": rank, "score": float(score) if score is not None else None}
        if row["candidate_sources"][index] == "qrel" and float(row["qrel_gains"][index]) > 0.0:
            positives.append(item)
        elif row["candidate_sources"][index] == "q3" and float(row["qrel_gains"][index]) == 0.0:
            negatives.append(item)

    promotion_positive_scores = [
        item["score"] for item in positives if 11 <= item["rank"] <= 20 and item["score"] is not None
    ]
    promotion_positive_ranks = [item["rank"] for item in positives if 11 <= item["rank"] <= 20]
    top10_positive_scores = [item["score"] for item in positives if 1 <= item["rank"] <= 10 and item["score"] is not None]
    top10_negatives = [item for item in negatives if 1 <= item["rank"] <= 10 and item["score"] is not None]
    top10_negative_scores = [item["score"] for item in top10_negatives]
    promotion_gap = None
    if promotion_positive_scores and top10_negative_scores:
        promotion_gap = min(neg_score - pos_score for pos_score in promotion_positive_scores for neg_score in top10_negative_scores)
    promotion_before_gap = bool(promotion_positive_ranks and top10_negative_scores and promotion_gap is not None)
    promotion = bool(promotion_before_gap and promotion_gap <= promotion_max_score_gap)
    guard = bool(top10_positive_scores and top10_negative_scores)
    margin = None
    if guard:
        margin = min(pos_score - neg_score for pos_score in top10_positive_scores for neg_score in top10_negative_scores)
    primary_negatives = sorted(
        [item for item in negatives if 1 <= item["rank"] <= q3_negative_primary_rank_cutoff],
        key=lambda item: (item["rank"], item["doc_id"]),
    )
    fill_negatives = sorted(
        [
            item
            for item in negatives
            if q3_negative_primary_rank_cutoff < item["rank"] <= q3_negative_fill_rank_cutoff
            and item["doc_id"] not in {primary["doc_id"] for primary in primary_negatives}
        ],
        key=lambda item: (item["rank"], item["doc_id"]),
    )
    retained_negatives = list(primary_negatives)
    if len(retained_negatives) < min_q3_negatives:
        retained_negatives.extend(fill_negatives[: min_q3_negatives - len(retained_negatives)])
    return {
        "dataset": dataset,
        "query_id": qid,
        "row_id": row["row_id"],
        "promotion": promotion,
        "promotion_before_gap": promotion_before_gap,
        "guard": guard,
        "promotion_positive_ranks_11_20": sorted(promotion_positive_ranks),
        "promotion_score_gap": promotion_gap,
        "q3_nonrelevant_top10_count": len(top10_negative_scores),
        "guard_margin": margin,
        "positive_top10_count": len(top10_positive_scores),
        "retained_negative_doc_ids": {item["doc_id"] for item in retained_negatives},
        "retained_negative_ranks": [item["rank"] for item in retained_negatives],
        "retained_positive_ranks": [item["rank"] for item in positives if 1 <= item["rank"] <= 20],
        "primary_q3_negative_count": len(primary_negatives),
        "filled_q3_negative_count": max(0, len(retained_negatives) - len(primary_negatives)),
    }


def select_rows(
    parent_rows: list[tuple[int, dict[str, Any]]],
    lookup: dict[tuple[str, str], dict[str, dict[str, Any]]],
    expected_promotion_rows: int,
    guard_count_per_dataset: int,
    guard_max_total: int,
    expected_rows: int,
    promotion_max_score_gap: float,
    q3_negative_primary_rank_cutoff: int,
    q3_negative_fill_rank_cutoff: int,
    min_q3_negatives: int,
    promotion_turboquant_topk_loss_weight: float | None,
    guard_turboquant_topk_loss_weight: float | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    promotion_rows: list[tuple[dict[str, Any], int, dict[str, Any]]] = []
    guard_candidates: dict[str, list[tuple[dict[str, Any], int, dict[str, Any]]]] = defaultdict(list)
    rejected = Counter()
    feature_records: list[dict[str, Any]] = []
    broad_promotion_rows: list[dict[str, Any]] = []
    rejected_high_gap: list[dict[str, Any]] = []
    for line_number, row in parent_rows:
        features = row_features(
            row,
            lookup,
            promotion_max_score_gap,
            q3_negative_primary_rank_cutoff,
            q3_negative_fill_rank_cutoff,
            min_q3_negatives,
        )
        feature_records.append({key: value for key, value in features.items() if key != "retained_negative_doc_ids"})
        if len(features["retained_negative_doc_ids"]) < min_q3_negatives:
            rejected["insufficient_retained_q3_negatives"] += 1
            continue
        if features["promotion_before_gap"]:
            broad_promotion_rows.append(features)
        if features["promotion"]:
            promotion_rows.append((row, line_number, features))
        elif features["promotion_before_gap"]:
            rejected_high_gap.append(
                {
                    "dataset": features["dataset"],
                    "query_id": features["query_id"],
                    "row_id": features["row_id"],
                    "promotion_score_gap": features["promotion_score_gap"],
                    "promotion_positive_ranks_11_20": features["promotion_positive_ranks_11_20"],
                }
            )
            rejected["promotion_high_gap"] += 1
        elif features["guard"]:
            guard_candidates[features["dataset"]].append((row, line_number, features))
        else:
            rejected["not_promotion_or_guard"] += 1

    if expected_promotion_rows > 0 and len(promotion_rows) != expected_promotion_rows:
        raise ValueError(f"selected {len(promotion_rows)} promotion rows, expected {expected_promotion_rows}")

    selected: list[dict[str, Any]] = []
    for row, line_number, features in sorted(promotion_rows, key=lambda item: item[2]["row_id"]):
        selected.append(
            build_derived_row(
                row,
                line_number,
                lookup,
                "promotion_11_20_to_top10_near_gap",
                features["promotion_score_gap"],
                features["retained_negative_doc_ids"],
                promotion_turboquant_topk_loss_weight,
            )
        )

    guard_summary: dict[str, Any] = {}
    for dataset in ("fiqa", "scifact"):
        candidates = sorted(
            guard_candidates.get(dataset, []),
            key=lambda item: (
                float(item[2]["guard_margin"]),
                item[2]["row_id"],
            ),
        )
        if len(candidates) < guard_count_per_dataset:
            raise ValueError(f"{dataset}: only {len(candidates)} guard candidates, expected {guard_count_per_dataset}")
        chosen = candidates[:guard_count_per_dataset]
        guard_summary[dataset] = {
            "available": len(candidates),
            "selected": len(chosen),
            "selected_qids": [item[2]["query_id"] for item in chosen],
            "selected_margins": [item[2]["guard_margin"] for item in chosen],
        }
        for row, line_number, features in chosen:
            selected.append(
                build_derived_row(
                    row,
                    line_number,
                    lookup,
                    "guard_top10_margin",
                    features["guard_margin"],
                    features["retained_negative_doc_ids"],
                    guard_turboquant_topk_loss_weight,
                )
            )

    if any(row["selection_bucket"] == "guard_top10_margin" and row["source_dataset"] == "nfcorpus" for row in selected):
        raise ValueError("nfcorpus guards are intentionally disabled for this artifact")
    if sum(len(summary["selected_qids"]) for summary in guard_summary.values()) > guard_max_total:
        raise ValueError(f"selected more than {guard_max_total} guard rows")
    selected.sort(key=lambda row: (row["selection_bucket"] != "promotion_11_20_to_top10_near_gap", row["source_dataset"], row["source_query_id"]))
    if expected_rows > 0 and len(selected) != expected_rows:
        raise ValueError(f"selected {len(selected)} rows, expected {expected_rows}")
    if len({row["row_id"] for row in selected}) != len(selected):
        raise ValueError("duplicate selected row_id")
    return selected, {
        "promotion_available": len(promotion_rows),
        "promotion_before_gap_available": len(broad_promotion_rows),
        "promotion_rejected_high_gap_count": len(rejected_high_gap),
        "promotion_rejected_high_gap_qids": [
            {
                "dataset": item["dataset"],
                "query_id": item["query_id"],
                "promotion_score_gap": item["promotion_score_gap"],
            }
            for item in sorted(rejected_high_gap, key=lambda item: (item["dataset"], item["query_id"]))
        ],
        "promotion_score_gaps": sorted(features["promotion_score_gap"] for _row, _line, features in promotion_rows),
        "guard_available": {dataset: len(rows) for dataset, rows in sorted(guard_candidates.items())},
        "guards": guard_summary,
        "rejected": dict(rejected),
        "feature_records_sha256": sha256_strings(
            [json.dumps(record, sort_keys=True, separators=(",", ":")) for record in feature_records]
        ),
    }


def validate_output_rows(rows: list[dict[str, Any]], min_q3_negatives: int) -> dict[str, Any]:
    source_counts = Counter()
    bucket_counts = Counter()
    dataset_counts = Counter()
    positive_rank_counts = Counter()
    negative_rank_counts = Counter()
    fill_counts = Counter()
    topk_weight_counts = Counter()
    alignment_checks = {
        "aligned_candidate_fields": True,
        "target_probability_sum_1": True,
        "positive_indexes_match_qrel_gains": True,
        "legal_gates_fail_closed": True,
        "min_q3_negative_count": None,
    }
    for row in rows:
        sizes = {field: len(row[field]) for field in ALIGNED_CANDIDATE_FIELDS}
        if len(set(sizes.values())) != 1:
            raise ValueError(f"{row['row_id']}: aligned field length mismatch {sizes}")
        if not row["positive_indexes"]:
            raise ValueError(f"{row['row_id']}: missing positives")
        if not any(row["candidate_sources"][index] == "q3" for index in range(len(row["candidate_doc_ids"])) if index not in row["positive_indexes"]):
            raise ValueError(f"{row['row_id']}: missing q3 negative")
        if sum(row["target_probabilities"]) < 0.999999 or sum(row["target_probabilities"]) > 1.000001:
            raise ValueError(f"{row['row_id']}: target probabilities do not sum to 1")
        if row.get("legal_gates") != LEGAL_GATES:
            raise ValueError(f"{row['row_id']}: legal gates are not fail-closed")
        if "turboquant_topk_loss_weight" in row:
            topk_weight = float(row["turboquant_topk_loss_weight"])
            if topk_weight < 0.0 or not math.isfinite(topk_weight):
                raise ValueError(f"{row['row_id']}: turboquant_topk_loss_weight must be finite and nonnegative")
            topk_weight_counts[str(topk_weight)] += 1
        else:
            topk_weight_counts["omitted"] += 1
        dataset_counts[row["source_dataset"]] += 1
        bucket_counts[row["selection_bucket"]] += 1
        source_counts.update(row["candidate_sources"])
        negative_count = 0
        for index, evidence in enumerate(row["anchor_q3_candidate_evidence"]):
            rank = int(evidence["q3_rank"])
            if row["candidate_sources"][index] == "qrel":
                positive_rank_counts[str(rank)] += 1
            elif row["candidate_sources"][index] == "q3":
                negative_count += 1
                negative_rank_counts[str(rank)] += 1
        fill_counts[
            str(
                max(
                    0,
                    min_q3_negatives
                    - sum(
                        1
                        for index, evidence in enumerate(row["anchor_q3_candidate_evidence"])
                        if row["candidate_sources"][index] == "q3" and int(evidence["q3_rank"]) <= 12
                    ),
                )
            )
        ] += 1
        current_min = alignment_checks["min_q3_negative_count"]
        alignment_checks["min_q3_negative_count"] = negative_count if current_min is None else min(int(current_min), negative_count)
    return {
        "dataset_counts": dict(dataset_counts),
        "bucket_counts": dict(bucket_counts),
        "candidate_source_counts": dict(source_counts),
        "retained_positive_rank_counts": dict(sorted(positive_rank_counts.items(), key=lambda item: int(item[0]))),
        "retained_q3_negative_rank_counts": dict(sorted(negative_rank_counts.items(), key=lambda item: int(item[0]))),
        "min_negative_fill_rows": dict(sorted(fill_counts.items(), key=lambda item: int(item[0]))),
        "turboquant_topk_loss_weight_counts": dict(sorted(topk_weight_counts.items())),
        "alignment_checks": alignment_checks,
        "min_candidate_count": min(len(row["candidate_doc_ids"]) for row in rows),
        "max_candidate_count": max(len(row["candidate_doc_ids"]) for row in rows),
        "min_positive_count": min(len(row["positive_indexes"]) for row in rows),
        "max_positive_count": max(len(row["positive_indexes"]) for row in rows),
    }


def overlap_report(
    rows: list[dict[str, Any]],
    excluded_qids: dict[str, set[str]],
    test_qids: dict[str, set[str]],
) -> dict[str, Any]:
    report: dict[str, Any] = {}
    selected: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        selected[row["source_dataset"]].add(str(row["source_query_id"]))
    for dataset in sorted(selected):
        frozen = sorted(selected[dataset] & excluded_qids.get(dataset, set()))
        test = sorted(selected[dataset] & test_qids.get(dataset, set()))
        report[dataset] = {
            "selected_count": len(selected[dataset]),
            "selected_x_frozen_dev_reserve": len(frozen),
            "selected_x_frozen_dev_reserve_qids": frozen,
            "selected_x_official_test": len(test),
            "selected_x_official_test_qids": test,
        }
        if frozen or test:
            raise ValueError(f"{dataset}: selected qids overlap frozen/test qids")
    return report


def main() -> None:
    args = parse_args()
    parent_manifest = validate_parent_inputs(args.parent_train_jsonl, args.parent_manifest)
    q3_lookup, q3_meta_by_path = load_q3_lookup(args.q3_per_query_jsonl, parent_manifest)
    parent_rows: list[tuple[int, dict[str, Any]]] = []
    for line_number, row in iter_jsonl(args.parent_train_jsonl):
        validate_parent_row(row, line_number)
        parent_rows.append((line_number, row))
    if not parent_rows:
        raise SystemExit("parent train JSONL has no rows")

    rows, selection_audit = select_rows(
        parent_rows,
        q3_lookup,
        args.expected_promotion_rows,
        args.guard_count_per_dataset,
        args.guard_max_total,
        args.expected_rows,
        args.promotion_max_score_gap,
        args.q3_negative_primary_rank_cutoff,
        args.q3_negative_fill_rank_cutoff,
        args.min_q3_negatives,
        args.promotion_turboquant_topk_loss_weight,
        args.guard_turboquant_topk_loss_weight,
    )
    excluded_qids = read_excluded_qids(args.exclude_qids_json)
    test_qids = dict(read_qrel_qids(path) for path in args.test_qrels)
    overlap = overlap_report(rows, excluded_qids, test_qids)
    validation = validate_output_rows(rows, args.min_q3_negatives)

    write_jsonl(args.output_jsonl, rows)
    output_hash = sha256_file(args.output_jsonl)
    coverage = {
        "schema": f"{SCHEMA}.coverage.v1",
        "created_utc": utc_stamp(),
        "selection_audit": selection_audit,
        "validation": validation,
        "overlap": overlap,
        "selected_rows": [
            {
                "row_id": row["row_id"],
                "dataset": row["source_dataset"],
                "query_id": row["source_query_id"],
                "selection_bucket": row["selection_bucket"],
                "guard_margin": row.get("guard_margin"),
                "turboquant_topk_loss_weight": row.get("turboquant_topk_loss_weight"),
                "candidate_count": len(row["candidate_doc_ids"]),
                "positive_count": len(row["positive_indexes"]),
                "q3_negative_count": row["q3_negative_count"],
            }
            for row in rows
        ],
        "legal_gates": dict(LEGAL_GATES),
        "quality_claim": False,
    }
    write_json(args.coverage_report, coverage)

    manifest = {
        "schema": SCHEMA,
        "created_utc": utc_stamp(),
        "builder": "scripts/build_q3top10_promotion_dataset.py",
        "inputs": {
            "parent_train_jsonl": str(args.parent_train_jsonl),
            "parent_train_jsonl_sha256": sha256_file(args.parent_train_jsonl),
            "parent_manifest": str(args.parent_manifest),
            "parent_manifest_sha256": sha256_file(args.parent_manifest),
            "q3_per_query_jsonl": q3_meta_by_path,
            "exclude_qids_json": {str(path): {"sha256": sha256_file(path)} for path in args.exclude_qids_json},
            "test_qrels": {str(path): {"dataset": dataset_from_qrels_path(path), "sha256": sha256_file(path)} for path in args.test_qrels},
        },
        "outputs": {
            "train_score_spectrum": str(args.output_jsonl),
            "manifest": str(args.manifest),
            "coverage_report": str(args.coverage_report),
        },
        "policy": {
            "promotion": "retain every parent row with at least one qrel positive at anchor q3 rank 11..20, at least one q3-source nonrelevant candidate at anchor q3 rank 1..10, and top10-nonrel-minus-positive q3 score gap <= configured promotion_max_score_gap",
            "guard": "from remaining rows select FiQA and SciFact guard rows by smallest vulnerable top10-positive/top10-nonrel q3 margin; no NFCorpus guards",
            "candidate_filter": "keep qrel positives with anchor q3 rank 1..20; keep q3-source nonrelevant candidates through rank12, filling deterministically from ranks13..20 until min_q3_negatives is met; drop BM25, far, sentinel, and missing-rank candidates",
            "provenance": "derived only from reviewed parent rows, with parent audit fields preserved under parent_audit_provenance",
        },
        "config": {
            "expected_promotion_rows": args.expected_promotion_rows,
            "guard_count_per_dataset": args.guard_count_per_dataset,
            "guard_max_total": args.guard_max_total,
            "expected_rows": args.expected_rows,
            "promotion_max_score_gap": args.promotion_max_score_gap,
            "q3_negative_primary_rank_cutoff": args.q3_negative_primary_rank_cutoff,
            "q3_negative_fill_rank_cutoff": args.q3_negative_fill_rank_cutoff,
            "min_q3_negatives": args.min_q3_negatives,
            "promotion_turboquant_topk_loss_weight": args.promotion_turboquant_topk_loss_weight,
            "guard_turboquant_topk_loss_weight": args.guard_turboquant_topk_loss_weight,
            "q3_method": Q3_METHOD,
            "q3_bits": Q3_BITS,
            "q3_quantizer_seed": Q3_SEED,
            "q3_scoring_surface": Q3_SURFACE,
        },
        "counts": {
            "parent_rows": len(parent_rows),
            "rows": len(rows),
            **validation,
        },
        "selection_audit": selection_audit,
        "overlap": overlap,
        "legal_gates": dict(LEGAL_GATES),
        **LEGAL_GATES,
        "quality_claim": False,
        "sha256": {
            "builder": sha256_file(Path(__file__).resolve()),
            "train_score_spectrum": output_hash,
            "coverage_report": sha256_file(args.coverage_report),
        },
    }
    write_json(args.manifest, manifest)
    print(f"q3top10 promotion dataset rows={len(rows)} promotions={selection_audit['promotion_available']}")
    print(f"output: {args.output_jsonl}")
    print(f"manifest: {args.manifest}")
    print(f"coverage: {args.coverage_report}")


if __name__ == "__main__":
    main()
