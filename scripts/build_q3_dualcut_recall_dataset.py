#!/usr/bin/env python3
"""Build the q3-near dual-cut recall-sentinel research dataset."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


SCHEMA = "eos.q3near_dualcut_recall_sentinel_dataset.v1"
LEGAL_GATES = {
    "train_allowed_for_research": True,
    "release_train_allowed": False,
    "commercial_use_allowed": False,
}
EXPECTED_BASE_SHA256 = "8cf3c1ecadc23f79520440aa415b6511654a5be71a08783f20144b7818adabee"
Q3_METHOD = "turboquant_ip_b3"
Q3_BITS = 3
Q3_SEED = 5581486560434873699
Q3_SURFACE = "turboquant_ip_prepared"
BASE_PROMOTION_BUCKET = "promotion_11_20_to_top10_near_gap"
BASE_GUARD_BUCKET = "guard_top10_margin"
SENTINEL_BUCKET = "nfcorpus_recall_sentinel_q3top100_rank80_100"
SENTINEL_SUFFIX = "nfrecall24-q3top100"
ALIGNED_CANDIDATE_FIELDS = (
    "candidate_doc_ids",
    "candidate_texts",
    "candidate_sources",
    "qrel_gains",
    "hard_negative_eligible",
    "target_probabilities",
)


def normalize_candidate_text(text: str) -> str:
    return " ".join(str(text).lower().split())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-train-jsonl", required=True, type=Path)
    parser.add_argument("--base-manifest", required=True, type=Path)
    parser.add_argument("--base-coverage", required=True, type=Path)
    parser.add_argument("--q3-per-query-jsonl", required=True, type=Path)
    parser.add_argument("--train-qrels", required=True, type=Path)
    parser.add_argument("--queries-jsonl", required=True, type=Path)
    parser.add_argument("--corpus-jsonl", required=True, type=Path)
    parser.add_argument("--exclude-qids-json", action="append", default=[], type=Path)
    parser.add_argument("--test-qrels", required=True, type=Path)
    parser.add_argument("--output-jsonl", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--coverage-report", required=True, type=Path)
    parser.add_argument("--expected-base-sha256", default=EXPECTED_BASE_SHA256)
    parser.add_argument("--expected-base-rows", type=int, default=45)
    parser.add_argument("--sentinel-count", type=int, default=24)
    parser.add_argument("--expected-rows", type=int, default=69)
    parser.add_argument("--required-qid", action="append", default=["PLAIN-63"])
    parser.add_argument("--required-doc-id", action="append", default=["MED-5329", "MED-4523"])
    parser.add_argument("--min-recall-rank", type=int, default=80)
    parser.add_argument("--max-recall-rank", type=int, default=100)
    parser.add_argument("--expected-q3-topk", type=int, default=100)
    return parser.parse_args()


def utc_stamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


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


def text_for_doc(row: dict[str, Any]) -> str:
    title = str(row.get("title") or "").strip()
    text = str(row.get("text") or "").strip()
    if title and text:
        return f"{title}\n{text}"
    return title or text


def read_text_jsonl(path: Path, label: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for line_number, row in iter_jsonl(path):
        item_id = str(row.get("_id") or "")
        if not item_id:
            raise ValueError(f"{path}:{line_number}: missing _id")
        values[item_id] = str(row.get("text") or "") if label == "query" else text_for_doc(row)
    return values


def read_qrels(path: Path) -> dict[str, dict[str, int]]:
    qrels: dict[str, dict[str, int]] = defaultdict(dict)
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"query-id", "corpus-id", "score"}
        if not reader.fieldnames or not required.issubset(set(reader.fieldnames)):
            raise ValueError(f"{path}: expected BEIR qrels TSV header")
        for row in reader:
            qid = str(row["query-id"])
            doc_id = str(row["corpus-id"])
            gain = int(row["score"])
            if gain > 0:
                qrels[qid][doc_id] = gain
    return dict(qrels)


def read_qrel_qids(path: Path) -> set[str]:
    qids: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if not reader.fieldnames or "query-id" not in reader.fieldnames:
            raise ValueError(f"{path}: expected BEIR qrels TSV header")
        for row in reader:
            if row.get("query-id"):
                qids.add(str(row["query-id"]))
    return qids


def read_excluded_qids(paths: list[Path]) -> set[str]:
    excluded: set[str] = set()
    for path in paths:
        payload = read_json(path)
        selected = payload.get("selected_qids", payload)
        if not isinstance(selected, dict):
            raise ValueError(f"{path}: expected selected_qids object")
        qids = selected.get("nfcorpus", [])
        if not isinstance(qids, list):
            raise ValueError(f"{path}: selected_qids.nfcorpus must be a list")
        excluded.update(str(qid) for qid in qids)
    return excluded


def parse_manifest_q3_hashes(base_manifest: dict[str, Any]) -> set[str]:
    source = base_manifest.get("inputs", {}).get("q3_per_query_jsonl", {})
    hashes = set()
    for item in source.get("parent_sources", []):
        if isinstance(item, dict) and item.get("sha256"):
            hashes.add(str(item["sha256"]))
    for value in base_manifest.get("q3_source_hashes", []):
        if isinstance(value, str) and ":" in value:
            hashes.add(value.rsplit(":", 1)[1])
    return hashes


def validate_base_inputs(args: argparse.Namespace) -> tuple[list[tuple[int, dict[str, Any]]], dict[str, Any]]:
    for path in (args.base_train_jsonl, args.base_manifest, args.base_coverage):
        if not path.is_file():
            raise ValueError(f"missing base input: {path}")
    actual_base_sha = sha256_file(args.base_train_jsonl)
    if actual_base_sha != args.expected_base_sha256:
        raise ValueError(f"base train sha mismatch: expected={args.expected_base_sha256} actual={actual_base_sha}")
    manifest = read_json(args.base_manifest)
    if manifest.get("legal_gates") != LEGAL_GATES:
        raise ValueError("base manifest legal gates are not fail-closed research-only gates")
    manifest_train_sha = manifest.get("sha256", {}).get("train_score_spectrum")
    if manifest_train_sha and manifest_train_sha != actual_base_sha:
        raise ValueError(f"base manifest train sha mismatch: manifest={manifest_train_sha} actual={actual_base_sha}")
    coverage = read_json(args.base_coverage)
    if coverage.get("legal_gates") != LEGAL_GATES:
        raise ValueError("base coverage legal gates are not fail-closed research-only gates")

    rows: list[tuple[int, dict[str, Any]]] = []
    for line_number, row in iter_jsonl(args.base_train_jsonl):
        validate_row_legal_and_alignment(row, f"base line {line_number}")
        rows.append((line_number, row))
    if len(rows) != args.expected_base_rows:
        raise ValueError(f"base rows={len(rows)}, expected {args.expected_base_rows}")
    return rows, manifest


def validate_q3_source(path: Path, base_manifest: dict[str, Any], expected_topk: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not path.is_file():
        raise ValueError(f"missing q3 per-query JSONL: {path}")
    digest = sha256_file(path)
    manifest_hashes = parse_manifest_q3_hashes(base_manifest)
    if manifest_hashes and digest not in manifest_hashes:
        raise ValueError(f"q3 source digest not present in base manifest: {digest}")
    rows: list[dict[str, Any]] = []
    for line_number, row in iter_jsonl(path):
        if str(row.get("dataset", "")).lower() != "nfcorpus":
            raise ValueError(f"{path}:{line_number}: expected nfcorpus q3 row")
        if row.get("method") != Q3_METHOD:
            raise ValueError(f"{path}:{line_number}: unexpected q3 method")
        if int(row.get("bits") or 0) != Q3_BITS:
            raise ValueError(f"{path}:{line_number}: unexpected q3 bits")
        if int(row.get("quantizer_seed") or 0) != Q3_SEED:
            raise ValueError(f"{path}:{line_number}: unexpected q3 seed")
        if str(row.get("scoring_surface", "")) != Q3_SURFACE:
            raise ValueError(f"{path}:{line_number}: unexpected q3 scoring surface")
        top_k = row.get("top_k")
        if not isinstance(top_k, list):
            raise ValueError(f"{path}:{line_number}: top_k must be a list")
        if len(top_k) != expected_topk:
            raise ValueError(f"{path}:{line_number}: q3 top_k length={len(top_k)}, expected {expected_topk}")
        ranks = [int(candidate.get("rank") or 0) for candidate in top_k]
        if ranks != list(range(1, expected_topk + 1)):
            raise ValueError(f"{path}:{line_number}: q3 ranks are not exactly 1..{expected_topk}")
        rows.append(row)
    return rows, {
        "supplied_path": str(path),
        "absolute_path": str(path.resolve()),
        "sha256": digest,
        "rows": len(rows),
    }


def validate_row_legal_and_alignment(row: dict[str, Any], context: str) -> None:
    sizes = {}
    for field in ALIGNED_CANDIDATE_FIELDS:
        value = row.get(field)
        if not isinstance(value, list):
            raise ValueError(f"{context}: {field} must be a list")
        sizes[field] = len(value)
    if len(set(sizes.values())) != 1:
        raise ValueError(f"{context}: aligned field length mismatch {sizes}")
    if row.get("legal_gates") != LEGAL_GATES:
        raise ValueError(f"{context}: legal gates are not fail-closed")
    if row.get("train_allowed_for_research") is not True:
        raise ValueError(f"{context}: train_allowed_for_research must be true")
    if row.get("release_train_allowed") is not False or row.get("commercial_use_allowed") is not False:
        raise ValueError(f"{context}: release/commercial gates must be false")
    positive_set = set(int(index) for index in row.get("positive_indexes", []))
    hard_eligible = row["hard_negative_eligible"]
    qrel_gains = row["qrel_gains"]
    for index, hard in enumerate(hard_eligible):
        gain = float(qrel_gains[index])
        if index in positive_set and hard:
            raise ValueError(f"{context}: positive candidate {index} cannot be hard-negative eligible")
        if index in positive_set and gain <= 0.0:
            raise ValueError(f"{context}: positive candidate {index} must have positive qrel gain")
        if index not in positive_set and gain > 0.0:
            raise ValueError(f"{context}: non-positive candidate {index} cannot have positive qrel gain")
        if hard and gain > 0.0:
            raise ValueError(f"{context}: qrel-positive candidate {index} cannot be hard-negative eligible")
    dataset = str(row.get("source_dataset") or "").lower()
    all_positive_doc_ids = {str(doc_id) for doc_id in row.get("all_train_qrel_positive_doc_ids", [])}
    for index, candidate_doc_id in enumerate(row["candidate_doc_ids"]):
        raw_doc_id = doc_id_for_dataset(dataset, str(candidate_doc_id)) if dataset else str(candidate_doc_id)
        if raw_doc_id in all_positive_doc_ids and index not in positive_set:
            raise ValueError(f"{context}: qrel-positive doc {raw_doc_id} is not in positive_indexes")
        if raw_doc_id in all_positive_doc_ids and hard_eligible[index]:
            raise ValueError(f"{context}: qrel-positive doc {raw_doc_id} cannot be hard-negative eligible")
    indexes_by_text: dict[str, list[int]] = defaultdict(list)
    for index, text in enumerate(row["candidate_texts"]):
        key = normalize_candidate_text(str(text))
        if key:
            indexes_by_text[key].append(index)
    for indexes in indexes_by_text.values():
        if not any(index in positive_set for index in indexes):
            continue
        hard_indexes = [index for index in indexes if hard_eligible[index]]
        if hard_indexes:
            raise ValueError(
                f"{context}: positive text duplicate candidates cannot be hard-negative eligible: {hard_indexes}"
            )


def base_row_copy(row: dict[str, Any], line_number: int) -> dict[str, Any]:
    copied = dict(row)
    bucket = copied.get("selection_bucket")
    if bucket == BASE_PROMOTION_BUCKET:
        main_weight = 1.0
    elif bucket == BASE_GUARD_BUCKET:
        main_weight = 0.0
    else:
        raise ValueError(f"{row.get('row_id')}: unexpected base selection bucket {bucket!r}")
    copied["turboquant_topk_loss_weight"] = main_weight
    copied["turboquant_topk_recall_loss_weight"] = 0.0
    copied["dualcut_parent_provenance"] = {
        "base_line_number": line_number,
        "base_row_sha256": sha256_json(row),
        "base_row_id": row.get("row_id"),
        "base_train_policy": row.get("train_policy"),
        "base_selection_bucket": bucket,
    }
    copied["train_policy"] = "q3near_dualcut_research_only_base_copy"
    copied["legal_gates"] = dict(LEGAL_GATES)
    copied.update(LEGAL_GATES)
    return copied


def q3_positive_hits(row: dict[str, Any], qrels: dict[str, dict[str, int]], min_rank: int, max_rank: int) -> list[dict[str, Any]]:
    qid = str(row["query_id"])
    gains = qrels.get(qid, {})
    hits = []
    for candidate in row["top_k"]:
        rank = int(candidate["rank"])
        doc_id = str(candidate["doc_id"])
        gain = gains.get(doc_id, 0)
        if min_rank <= rank <= max_rank and gain > 0:
            hits.append({"rank": rank, "doc_id": doc_id, "gain": gain})
    return hits


def selection_key(row: dict[str, Any], hits: list[dict[str, Any]], base_qids: set[str]) -> tuple[Any, ...]:
    nearest_hit = sorted(hits, key=lambda hit: (abs(100 - hit["rank"]), -hit["rank"], hit["doc_id"]))[0]
    return (
        str(row["query_id"]) not in base_qids,
        abs(100 - nearest_hit["rank"]),
        -nearest_hit["rank"],
        nearest_hit["doc_id"],
        str(row["query_id"]),
    )


def select_sentinel_qids(
    q3_rows: list[dict[str, Any]],
    qrels: dict[str, dict[str, int]],
    excluded_qids: set[str],
    test_qids: set[str],
    base_qids: set[str],
    count: int,
    required_qids: list[str],
    min_rank: int,
    max_rank: int,
) -> tuple[list[tuple[dict[str, Any], list[dict[str, Any]]]], dict[str, Any]]:
    rejected = Counter()
    candidates: list[tuple[dict[str, Any], list[dict[str, Any]]]] = []
    for row in q3_rows:
        qid = str(row["query_id"])
        if qid in excluded_qids:
            rejected["frozen_dev_reserve"] += 1
            continue
        if qid in test_qids:
            rejected["official_test"] += 1
            continue
        if qid not in qrels:
            rejected["missing_train_qrels"] += 1
            continue
        hits = q3_positive_hits(row, qrels, min_rank, max_rank)
        if not hits:
            rejected["no_positive_rank80_100"] += 1
            continue
        candidates.append((row, hits))
    by_qid = {str(row["query_id"]): (row, hits) for row, hits in candidates}
    missing_required = [qid for qid in required_qids if qid not in by_qid]
    if missing_required:
        raise ValueError(f"required qids not eligible after exclusions: {missing_required}")
    selected = sorted(candidates, key=lambda item: selection_key(item[0], item[1], base_qids))[:count]
    selected_qids = {str(row["query_id"]) for row, _hits in selected}
    if not set(required_qids).issubset(selected_qids):
        raise ValueError(f"required qids not selected fail-closed: {sorted(set(required_qids) - selected_qids)}")
    if len(selected) != count:
        raise ValueError(f"selected {len(selected)} sentinel rows, expected {count}")
    return selected, {
        "eligible": len(candidates),
        "eligible_base_qids": sum(1 for row, _hits in candidates if str(row["query_id"]) in base_qids),
        "selected": len(selected),
        "selected_base_qids": sum(1 for row, _hits in selected if str(row["query_id"]) in base_qids),
        "selected_qids": [str(row["query_id"]) for row, _hits in selected],
        "selected_rank80_100_hits": {
            str(row["query_id"]): sorted(hits, key=lambda hit: (hit["rank"], hit["doc_id"])) for row, hits in selected
        },
        "rejected": dict(rejected),
        "policy": "train-only NFCorpus qids, frozen dev/reserve and official test excluded first; require at least one train-qrel positive at anchor q3 rank 80..100; sort by already-in-base vulnerability, nearest-to-rank100, rank, doc-id, query-id",
    }


def build_sentinel_row(
    q3_row: dict[str, Any],
    hits: list[dict[str, Any]],
    qrels: dict[str, dict[str, int]],
    queries: dict[str, str],
    corpus: dict[str, str],
    q3_source_path: Path,
) -> dict[str, Any]:
    qid = str(q3_row["query_id"])
    if qid not in queries:
        raise ValueError(f"{qid}: missing query text")
    gains = qrels[qid]
    candidate_doc_ids: list[str] = []
    candidate_texts: list[str] = []
    candidate_sources: list[str] = []
    qrel_gains: list[float] = []
    evidence: list[dict[str, Any]] = []
    for candidate in q3_row["top_k"]:
        doc_id = str(candidate["doc_id"])
        if doc_id not in corpus:
            raise ValueError(f"{qid}: missing corpus text for {doc_id}")
        gain = gains.get(doc_id, 0)
        candidate_doc_ids.append(f"nfcorpus:{doc_id}")
        candidate_texts.append(corpus[doc_id])
        candidate_sources.append("qrel" if gain > 0 else "q3")
        qrel_gains.append(float(gain))
        ev = {
            "doc_id": doc_id,
            "q3_rank": int(candidate["rank"]),
            "q3_score": candidate.get("score"),
            "relevance": int(candidate.get("relevance") or 0),
            "dense_rank": candidate.get("dense_rank"),
            "dense_score": candidate.get("dense_score"),
            "compact_rank": candidate.get("compact_rank"),
            "compact_score": candidate.get("compact_score"),
            "source_path": str(q3_source_path),
        }
        if ev["relevance"] != int(gain):
            raise ValueError(f"{qid}: q3 relevance/qrels mismatch for {doc_id}: q3={ev['relevance']} qrels={gain}")
        evidence.append(ev)
    positive_indexes = [index for index, gain in enumerate(qrel_gains) if gain > 0.0]
    if not positive_indexes:
        raise ValueError(f"{qid}: sentinel row has no positives in top100")
    positive_text_keys = {normalize_candidate_text(candidate_texts[index]) for index in positive_indexes}
    hard_negative_eligible = [
        gain <= 0.0 and normalize_candidate_text(text) not in positive_text_keys
        for text, gain in zip(candidate_texts, qrel_gains)
    ]
    positive_text_alias_indexes = [
        index
        for index, text in enumerate(candidate_texts)
        if index not in positive_indexes and normalize_candidate_text(text) in positive_text_keys
    ]
    gain_sum = sum(qrel_gains[index] for index in positive_indexes)
    target_probabilities = [0.0] * len(candidate_doc_ids)
    for index in positive_indexes:
        target_probabilities[index] = qrel_gains[index] / gain_sum
    hit_ranks = sorted(hit["rank"] for hit in hits)
    return {
        "row_id": f"nfcorpus:{qid}:{SENTINEL_SUFFIX}",
        "source": "nfcorpus:beir-train:q3near-dualcut-recall-sentinel-v1",
        "query": queries[qid],
        "candidate_doc_ids": candidate_doc_ids,
        "candidate_texts": candidate_texts,
        "candidate_sources": candidate_sources,
        "qrel_gains": qrel_gains,
        "positive_indexes": positive_indexes,
        "selected_positive_index": positive_indexes[0],
        "hard_negative_eligible": hard_negative_eligible,
        "target_probabilities": target_probabilities,
        "hard_loss_weight": 0.0,
        "soft_loss_weight": 0.0,
        "recovery_loss_weight": 0.0,
        "turboquant_topk_loss_weight": 0.0,
        "turboquant_topk_recall_loss_weight": 1.0,
        "train_policy": "q3near_dualcut_nfrecal24_research_only_recall_sentinel",
        "selection_bucket": SENTINEL_BUCKET,
        "guard_margin": None,
        "legal_gates": dict(LEGAL_GATES),
        **LEGAL_GATES,
        "source_artifact_hash": sha256_json({"q3_query": q3_row, "train_qrel_positive_doc_ids": sorted(gains)}),
        "source_dataset": "nfcorpus",
        "source_query_id": qid,
        "source_positive_doc_ids": [doc_id_for_dataset("nfcorpus", candidate_doc_ids[index]) for index in positive_indexes],
        "candidate_positive_doc_ids_pre_dedupe": [doc_id_for_dataset("nfcorpus", candidate_doc_ids[index]) for index in positive_indexes],
        "all_train_qrel_positive_doc_ids": sorted(gains),
        "positive_count": len(positive_indexes),
        "bm25_negative_count": 0,
        "q3_negative_count": sum(1 for value in hard_negative_eligible if value),
        "q3_positive_text_alias_count": len(positive_text_alias_indexes),
        "q3_positive_text_alias_candidate_indexes": positive_text_alias_indexes,
        "extra_negative_count": 0,
        "q3_evidence": [evidence[index] for index, hard in enumerate(hard_negative_eligible) if hard],
        "anchor_q3_candidate_evidence": evidence,
        "q3_scoring_metadata": {
            "method": Q3_METHOD,
            "bits": Q3_BITS,
            "scoring_surface": Q3_SURFACE,
            "quantizer_seed": Q3_SEED,
            "candidate_count": q3_row.get("candidate_count"),
            "candidates_scored": q3_row.get("candidates_scored"),
            "pruning_supported": q3_row.get("pruning_supported"),
            "pruning_used": q3_row.get("pruning_used"),
        },
        "recall_sentinel_provenance": {
            "q3_source_path": str(q3_source_path),
            "q3_query_row_sha256": sha256_json(q3_row),
            "selection_reason": "train qrel positive in anchor q3 rank80..100",
            "rank80_100_positive_hits": sorted(hits, key=lambda hit: (hit["rank"], hit["doc_id"])),
            "top100_candidate_count": len(candidate_doc_ids),
            "all_train_qrel_positive_count": len(gains),
            "positive_text_alias_candidate_indexes": positive_text_alias_indexes,
        },
    }


def validate_output_rows(rows: list[dict[str, Any]], expected_base_rows: int, sentinel_count: int, expected_topk: int) -> dict[str, Any]:
    dataset_counts = Counter()
    bucket_counts = Counter()
    candidate_source_counts = Counter()
    main_weight_counts = Counter()
    recall_weight_counts = Counter()
    candidate_count_counts = Counter()
    pair_count = 0
    positive_pair_count = 0
    hard_negative_eligible_pair_count = 0
    positive_text_alias_count = 0
    positive_text_alias_rows: dict[str, int] = {}
    sentinel_qids: list[str] = []
    sentinel_rank_buckets = Counter()
    for row in rows:
        validate_row_legal_and_alignment(row, row["row_id"])
        if sum(float(value) for value in row["target_probabilities"]) < -0.000001:
            raise ValueError(f"{row['row_id']}: target probabilities invalid")
        if row["positive_indexes"]:
            total_prob = sum(float(value) for value in row["target_probabilities"])
            if total_prob < 0.999999 or total_prob > 1.000001:
                raise ValueError(f"{row['row_id']}: target probabilities do not sum to 1")
        positives = [index for index, gain in enumerate(row["qrel_gains"]) if float(gain) > 0.0]
        if positives != list(row["positive_indexes"]):
            raise ValueError(f"{row['row_id']}: positive indexes do not match qrel gains")
        main_weight = float(row.get("turboquant_topk_loss_weight"))
        recall_weight = float(row.get("turboquant_topk_recall_loss_weight"))
        if main_weight < 0.0 or recall_weight < 0.0 or not math.isfinite(main_weight) or not math.isfinite(recall_weight):
            raise ValueError(f"{row['row_id']}: nonfinite/negative loss weight")
        dataset_counts[row["source_dataset"]] += 1
        bucket_counts[row["selection_bucket"]] += 1
        candidate_source_counts.update(row["candidate_sources"])
        main_weight_counts[str(main_weight)] += 1
        recall_weight_counts[str(recall_weight)] += 1
        candidate_count_counts[str(len(row["candidate_doc_ids"]))] += 1
        pair_count += len(row["candidate_doc_ids"])
        positive_pair_count += len(positives)
        hard_negative_eligible_pair_count += sum(1 for value in row["hard_negative_eligible"] if bool(value))
        row_alias_count = 0
        positive_text_keys = {normalize_candidate_text(row["candidate_texts"][index]) for index in positives}
        for index, text in enumerate(row["candidate_texts"]):
            if index not in positives and normalize_candidate_text(text) in positive_text_keys:
                row_alias_count += 1
        if row_alias_count:
            positive_text_alias_count += row_alias_count
            positive_text_alias_rows[row["row_id"]] = row_alias_count
        if row["selection_bucket"] == SENTINEL_BUCKET:
            sentinel_qids.append(str(row["source_query_id"]))
            if len(row["candidate_doc_ids"]) != expected_topk:
                raise ValueError(f"{row['row_id']}: sentinel candidate count is not {expected_topk}")
            ranks = [int(evidence["q3_rank"]) for evidence in row["anchor_q3_candidate_evidence"]]
            if ranks != list(range(1, expected_topk + 1)):
                raise ValueError(f"{row['row_id']}: sentinel q3 evidence ranks are not exactly 1..{expected_topk}")
            for index in positives:
                rank = int(row["anchor_q3_candidate_evidence"][index]["q3_rank"])
                if 80 <= rank <= 100:
                    sentinel_rank_buckets[str(rank)] += 1
        elif recall_weight != 0.0:
            raise ValueError(f"{row['row_id']}: non-sentinel row has recall loss weight")
    if bucket_counts[SENTINEL_BUCKET] != sentinel_count:
        raise ValueError(f"sentinel rows={bucket_counts[SENTINEL_BUCKET]}, expected {sentinel_count}")
    if len(rows) - sentinel_count != expected_base_rows:
        raise ValueError(f"base-copy rows={len(rows) - sentinel_count}, expected {expected_base_rows}")
    if len({row["row_id"] for row in rows}) != len(rows):
        raise ValueError("duplicate row_id in output")
    return {
        "dataset_counts": dict(dataset_counts),
        "bucket_counts": dict(bucket_counts),
        "candidate_source_counts": dict(candidate_source_counts),
        "turboquant_topk_loss_weight_counts": dict(sorted(main_weight_counts.items())),
        "turboquant_topk_recall_loss_weight_counts": dict(sorted(recall_weight_counts.items())),
        "candidate_count_counts": dict(sorted(candidate_count_counts.items(), key=lambda item: int(item[0]))),
        "pair_count": pair_count,
        "positive_pair_count": positive_pair_count,
        "hard_negative_eligible_pair_count": hard_negative_eligible_pair_count,
        "positive_text_alias_count": positive_text_alias_count,
        "positive_text_alias_rows": dict(sorted(positive_text_alias_rows.items())),
        "sentinel_qids": sentinel_qids,
        "sentinel_rank80_100_positive_rank_counts": dict(sorted(sentinel_rank_buckets.items(), key=lambda item: int(item[0]))),
        "alignment_checks": {
            "aligned_candidate_fields": True,
            "target_probability_sum_1": True,
            "positive_indexes_match_qrel_gains": True,
            "positive_text_duplicates_not_hard_negative_eligible": True,
            "legal_gates_fail_closed": True,
            "sentinel_candidate_count": expected_topk,
        },
    }


def overlap_report(rows: list[dict[str, Any]], excluded_qids: set[str], test_qids: set[str]) -> dict[str, Any]:
    selected = {str(row["source_query_id"]) for row in rows if row["source_dataset"] == "nfcorpus"}
    frozen = sorted(selected & excluded_qids)
    test = sorted(selected & test_qids)
    if frozen or test:
        raise ValueError(f"nfcorpus: selected qids overlap frozen/test qids: frozen={frozen} test={test}")
    return {
        "nfcorpus": {
            "selected_count": len(selected),
            "selected_x_frozen_dev_reserve": len(frozen),
            "selected_x_frozen_dev_reserve_qids": frozen,
            "selected_x_official_test": len(test),
            "selected_x_official_test_qids": test,
        }
    }


def main() -> None:
    args = parse_args()
    base_rows_with_lines, base_manifest = validate_base_inputs(args)
    q3_rows, q3_identity = validate_q3_source(args.q3_per_query_jsonl, base_manifest, args.expected_q3_topk)
    qrels = read_qrels(args.train_qrels)
    queries = read_text_jsonl(args.queries_jsonl, "query")
    corpus = read_text_jsonl(args.corpus_jsonl, "corpus")
    excluded_qids = read_excluded_qids(args.exclude_qids_json)
    test_qids = read_qrel_qids(args.test_qrels)
    base_qids = {str(row["source_query_id"]) for _line, row in base_rows_with_lines if row["source_dataset"] == "nfcorpus"}

    selected_sentinels, sentinel_audit = select_sentinel_qids(
        q3_rows,
        qrels,
        excluded_qids,
        test_qids,
        base_qids,
        args.sentinel_count,
        args.required_qid,
        args.min_recall_rank,
        args.max_recall_rank,
    )

    output_rows = [base_row_copy(row, line_number) for line_number, row in base_rows_with_lines]
    output_rows.extend(
        build_sentinel_row(row, hits, qrels, queries, corpus, args.q3_per_query_jsonl)
        for row, hits in selected_sentinels
    )
    if len(output_rows) != args.expected_rows:
        raise ValueError(f"output rows={len(output_rows)}, expected {args.expected_rows}")
    required_doc_ids = set(args.required_doc_id)
    selected_positive_doc_ids = {
        doc_id_for_dataset("nfcorpus", row["candidate_doc_ids"][index])
        for row in output_rows
        if row["selection_bucket"] == SENTINEL_BUCKET
        for index in row["positive_indexes"]
    }
    missing_required_doc_ids = sorted(required_doc_ids - selected_positive_doc_ids)
    if missing_required_doc_ids:
        raise ValueError(f"required doc ids missing from sentinel positives: {missing_required_doc_ids}")

    overlap = overlap_report(output_rows, excluded_qids, test_qids)
    validation = validate_output_rows(output_rows, args.expected_base_rows, args.sentinel_count, args.expected_q3_topk)
    write_jsonl(args.output_jsonl, output_rows)
    train_sha = sha256_file(args.output_jsonl)

    coverage = {
        "schema": f"{SCHEMA}.coverage.v1",
        "created_utc": utc_stamp(),
        "selection_audit": sentinel_audit,
        "validation": validation,
        "overlap": overlap,
        "required_qids": {qid: qid in validation["sentinel_qids"] for qid in args.required_qid},
        "required_doc_ids_in_sentinel_positives": {doc_id: doc_id in selected_positive_doc_ids for doc_id in args.required_doc_id},
        "selected_rows": [
            {
                "row_id": row["row_id"],
                "dataset": row["source_dataset"],
                "query_id": row["source_query_id"],
                "selection_bucket": row["selection_bucket"],
                "turboquant_topk_loss_weight": row["turboquant_topk_loss_weight"],
                "turboquant_topk_recall_loss_weight": row["turboquant_topk_recall_loss_weight"],
                "candidate_count": len(row["candidate_doc_ids"]),
                "positive_count": len(row["positive_indexes"]),
                "q3_negative_count": row["q3_negative_count"],
            }
            for row in output_rows
        ],
        "legal_gates": dict(LEGAL_GATES),
        "quality_claim": False,
    }
    write_json(args.coverage_report, coverage)

    manifest = {
        "schema": SCHEMA,
        "created_utc": utc_stamp(),
        "builder": "scripts/build_q3_dualcut_recall_dataset.py",
        "inputs": {
            "base_train_jsonl": str(args.base_train_jsonl),
            "base_train_jsonl_sha256": sha256_file(args.base_train_jsonl),
            "base_manifest": str(args.base_manifest),
            "base_manifest_sha256": sha256_file(args.base_manifest),
            "base_coverage": str(args.base_coverage),
            "base_coverage_sha256": sha256_file(args.base_coverage),
            "q3_per_query_jsonl": q3_identity,
            "train_qrels": {"path": str(args.train_qrels), "sha256": sha256_file(args.train_qrels)},
            "queries_jsonl": {"path": str(args.queries_jsonl), "sha256": sha256_file(args.queries_jsonl)},
            "corpus_jsonl": {"path": str(args.corpus_jsonl), "sha256": sha256_file(args.corpus_jsonl)},
            "exclude_qids_json": {str(path): {"sha256": sha256_file(path)} for path in args.exclude_qids_json},
            "test_qrels": {"path": str(args.test_qrels), "sha256": sha256_file(args.test_qrels)},
        },
        "outputs": {
            "train_score_spectrum": str(args.output_jsonl),
            "manifest": str(args.manifest),
            "coverage_report": str(args.coverage_report),
        },
        "policy": {
            "base_copy": "copy finalized 45-row q3-near promotion/guard base, assigning turboquant_topk_loss_weight=1 for promotion_11_20_to_top10_near_gap and 0 for guards, with recall loss weight 0",
            "sentinel_selection": sentinel_audit["policy"],
            "sentinel_candidates": "each sentinel row keeps the persisted anchor q3 top100 candidates exactly; train-qrel positive candidates become qrel with raw qrel gains and all others remain q3/nonrelevant",
            "legal": "research-only rows with release/commercial gates fail-closed; no quality claim",
        },
        "config": {
            "expected_base_sha256": args.expected_base_sha256,
            "expected_base_rows": args.expected_base_rows,
            "sentinel_count": args.sentinel_count,
            "expected_rows": args.expected_rows,
            "min_recall_rank": args.min_recall_rank,
            "max_recall_rank": args.max_recall_rank,
            "expected_q3_topk": args.expected_q3_topk,
            "q3_method": Q3_METHOD,
            "q3_bits": Q3_BITS,
            "q3_quantizer_seed": Q3_SEED,
            "q3_scoring_surface": Q3_SURFACE,
        },
        "counts": {
            "base_rows": args.expected_base_rows,
            "sentinel_rows": args.sentinel_count,
            "rows": len(output_rows),
            **validation,
        },
        "selection_audit": sentinel_audit,
        "overlap": overlap,
        "legal_gates": dict(LEGAL_GATES),
        **LEGAL_GATES,
        "quality_claim": False,
        "sha256": {
            "builder": sha256_file(Path(__file__).resolve()),
            "base_train_jsonl": sha256_file(args.base_train_jsonl),
            "base_manifest": sha256_file(args.base_manifest),
            "base_coverage": sha256_file(args.base_coverage),
            "q3_per_query_jsonl": q3_identity["sha256"],
            "train_qrels": sha256_file(args.train_qrels),
            "queries_jsonl": sha256_file(args.queries_jsonl),
            "corpus_jsonl": sha256_file(args.corpus_jsonl),
            "exclude_qids_json": {str(path): sha256_file(path) for path in args.exclude_qids_json},
            "test_qrels": sha256_file(args.test_qrels),
            "train_score_spectrum": train_sha,
            "coverage_report": sha256_file(args.coverage_report),
        },
    }
    write_json(args.manifest, manifest)
    print(f"q3 dualcut recall dataset rows={len(output_rows)} sentinels={args.sentinel_count}")
    print(f"output: {args.output_jsonl}")
    print(f"manifest: {args.manifest}")
    print(f"coverage: {args.coverage_report}")


if __name__ == "__main__":
    main()
