#!/usr/bin/env python3
"""Select qids for a q3 promotion+guard score-spectrum slice.

The selector consumes anchor `eval-retrieval-turboquant --per-query-jsonl`
rows and emits a selected-qids JSON suitable for
`build_retrieval_beir_score_spectrum.py --selected-qids-json`.

It is intentionally data-only: official test qids and frozen local dev/reserve
qids are excluded before ranking any candidates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


SCHEMA = "eos.q3_promotion_guard_qids.v1"
LEGAL_GATES = {
    "train_allowed_for_research": True,
    "release_train_allowed": False,
    "commercial_use_allowed": False,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-query-jsonl", action="append", required=True, type=Path)
    parser.add_argument("--train-qrels", action="append", required=True, type=Path)
    parser.add_argument("--test-qrels", action="append", required=True, type=Path)
    parser.add_argument("--exclude-qids-json", action="append", default=[], type=Path)
    parser.add_argument("--rows-per-dataset", type=int, default=16)
    parser.add_argument("--promotion-quota", type=int, default=8)
    parser.add_argument("--output-selected-qids-json", required=True, type=Path)
    parser.add_argument("--output-records-json", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
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


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


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


def read_qrels_qids(path: Path) -> tuple[str, set[str]]:
    dataset = dataset_from_qrels_path(path)
    qids: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        header = next(handle, "")
        if not header.lower().startswith("query-id\t"):
            raise ValueError(f"{path}: expected BEIR qrels TSV header")
        for raw in handle:
            parts = raw.rstrip("\n").split("\t")
            if len(parts) >= 3 and parts[0]:
                qids.add(parts[0])
    return dataset, qids


def dataset_from_qrels_path(path: Path) -> str:
    # .../raw/<dataset>/<dataset>/qrels/train.tsv or fixture equivalents.
    parts = path.parts
    if "qrels" in parts:
        idx = parts.index("qrels")
        if idx >= 1:
            return parts[idx - 1].lower()
    raise ValueError(f"cannot infer dataset from qrels path: {path}")


def read_excluded_qids(paths: list[Path]) -> dict[str, set[str]]:
    excluded: dict[str, set[str]] = defaultdict(set)
    for path in paths:
        payload = read_json(path)
        selected = payload.get("selected_qids", payload)
        if not isinstance(selected, dict):
            raise ValueError(f"{path}: expected selected_qids object")
        for dataset, qids in selected.items():
            excluded[str(dataset).lower()].update(str(qid) for qid in qids)
    return excluded


def load_per_query_rows(paths: list[Path]) -> tuple[dict[str, list[dict[str, Any]]], dict[str, dict[str, Any]]]:
    rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    input_meta: dict[str, dict[str, Any]] = {}
    for path in paths:
        count = 0
        datasets = Counter()
        for _line_number, row in iter_jsonl(path):
            dataset = str(row.get("dataset", "")).lower()
            qid = str(row.get("query_id", ""))
            if not dataset or not qid:
                raise ValueError(f"{path}: per-query row missing dataset/query_id")
            rows[dataset].append(row)
            datasets[dataset] += 1
            count += 1
        input_meta[str(path)] = {
            "sha256": sha256_file(path),
            "rows": count,
            "datasets": dict(datasets),
        }
    return rows, input_meta


def row_features(row: dict[str, Any]) -> dict[str, Any]:
    relevant_ranks = [
        int(item["rank"])
        for item in row.get("top_k", [])
        if int(item.get("relevance") or 0) > 0 and int(item.get("rank") or 0) > 0
    ]
    relevant_ranks.sort()
    quality = row.get("quality", {})
    relevant_count = int(row.get("relevant_count") or 0)
    in_top100 = len(relevant_ranks)
    first_rank = relevant_ranks[0] if relevant_ranks else 1_000_000
    promotion_ranks = [rank for rank in relevant_ranks if 11 <= rank <= 20]
    guard_ranks = [rank for rank in relevant_ranks if 1 <= rank <= 10]
    sentinel_ranks = [rank for rank in relevant_ranks if 80 <= rank <= 100]
    return {
        "dataset": str(row.get("dataset", "")).lower(),
        "query_id": str(row.get("query_id", "")),
        "relevant_count": relevant_count,
        "relevant_in_top100": in_top100,
        "missing_relevant_top100": max(0, relevant_count - in_top100),
        "first_relevant_rank": first_rank,
        "ndcg_at_10": float(quality.get("ndcg_at_10") or 0.0),
        "recall_at_100": float(quality.get("recall_at_100") or 0.0),
        "promotion_ranks_11_20": promotion_ranks,
        "guard_ranks_1_10": guard_ranks,
        "recall_sentinel_ranks_80_100": sentinel_ranks,
        "top10_relevant_count": len(guard_ranks),
        "top20_relevant_count": len([rank for rank in relevant_ranks if rank <= 20]),
        "q3_nonrelevant_top10": len(
            [
                item
                for item in row.get("top_k", [])
                if 1 <= int(item.get("rank") or 0) <= 10 and int(item.get("relevance") or 0) <= 0
            ]
        ),
    }


def promotion_key(record: dict[str, Any]) -> tuple[Any, ...]:
    ranks = record["promotion_ranks_11_20"]
    nearest = min(abs(rank - 10) for rank in ranks) if ranks else 1_000_000
    return (
        nearest,
        record["ndcg_at_10"],
        -record["top20_relevant_count"],
        -record["relevant_count"],
        record["first_relevant_rank"],
        record["query_id"],
    )


def guard_key(record: dict[str, Any]) -> tuple[Any, ...]:
    return (
        -record["missing_relevant_top100"],
        -len(record["recall_sentinel_ranks_80_100"]),
        record["ndcg_at_10"],
        -record["top10_relevant_count"],
        -record["relevant_count"],
        record["first_relevant_rank"],
        record["query_id"],
    )


def fallback_key(record: dict[str, Any]) -> tuple[Any, ...]:
    return (
        record["ndcg_at_10"],
        -record["relevant_count"],
        record["first_relevant_rank"],
        record["query_id"],
    )


def select_for_dataset(records: list[dict[str, Any]], rows_per_dataset: int, promotion_quota: int) -> list[dict[str, Any]]:
    if rows_per_dataset <= 0:
        raise ValueError("--rows-per-dataset must be positive")
    if promotion_quota < 0:
        raise ValueError("--promotion-quota must be non-negative")

    headroom = [r for r in records if r["recall_at_100"] > 0 and r["ndcg_at_10"] < 0.995]
    promotion = sorted([r for r in headroom if r["promotion_ranks_11_20"]], key=promotion_key)
    selected: list[dict[str, Any]] = []
    selected_qids: set[str] = set()

    for record in promotion[: min(promotion_quota, rows_per_dataset)]:
        selected.append({**record, "selected_bucket": "promotion_11_20"})
        selected_qids.add(record["query_id"])

    guards = sorted(
        [
            r
            for r in headroom
            if r["query_id"] not in selected_qids and (r["guard_ranks_1_10"] or r["missing_relevant_top100"] > 0)
        ],
        key=guard_key,
    )
    for record in guards:
        if len(selected) >= rows_per_dataset:
            break
        selected.append({**record, "selected_bucket": "guard_recall_quality"})
        selected_qids.add(record["query_id"])

    fallback = sorted([r for r in headroom if r["query_id"] not in selected_qids], key=fallback_key)
    for record in fallback:
        if len(selected) >= rows_per_dataset:
            break
        selected.append({**record, "selected_bucket": "fallback_headroom"})
        selected_qids.add(record["query_id"])

    if len(selected) != rows_per_dataset:
        raise ValueError(f"selected {len(selected)} rows, expected {rows_per_dataset}")
    return selected


def main() -> None:
    args = parse_args()
    per_query_rows, per_query_meta = load_per_query_rows(args.per_query_jsonl)
    train_qids = dict(read_qrels_qids(path) for path in args.train_qrels)
    test_qids = dict(read_qrels_qids(path) for path in args.test_qrels)
    frozen_qids = read_excluded_qids(args.exclude_qids_json)

    selected_by_dataset: dict[str, list[str]] = {}
    records_by_dataset: dict[str, list[dict[str, Any]]] = {}
    manifest_datasets: dict[str, Any] = {}

    for dataset in sorted(train_qids):
        records: list[dict[str, Any]] = []
        excluded_counts = Counter()
        for row in per_query_rows.get(dataset, []):
            qid = str(row.get("query_id", ""))
            if qid not in train_qids[dataset]:
                excluded_counts["not_in_train_qrels"] += 1
                continue
            if qid in test_qids.get(dataset, set()):
                excluded_counts["official_test_overlap"] += 1
                continue
            if qid in frozen_qids.get(dataset, set()):
                excluded_counts["frozen_dev_reserve"] += 1
                continue
            records.append(row_features(row))

        selected = select_for_dataset(records, args.rows_per_dataset, args.promotion_quota)
        selected_qids = [record["query_id"] for record in selected]
        if set(selected_qids) & test_qids.get(dataset, set()):
            raise SystemExit(f"{dataset}: selected qids overlap official test qrels")
        if set(selected_qids) & frozen_qids.get(dataset, set()):
            raise SystemExit(f"{dataset}: selected qids overlap frozen dev/reserve qids")

        selected_by_dataset[dataset] = selected_qids
        records_by_dataset[dataset] = selected
        bucket_counts = Counter(record["selected_bucket"] for record in selected)
        manifest_datasets[dataset] = {
            "available_per_query_rows": len(per_query_rows.get(dataset, [])),
            "eligible_after_exclusions": len(records),
            "selected_count": len(selected),
            "selected_qids_sha256": sha256_strings(selected_qids),
            "bucket_counts": dict(bucket_counts),
            "promotion_available_after_exclusions": sum(1 for record in records if record["promotion_ranks_11_20"]),
            "guard_available_after_exclusions": sum(
                1 for record in records if record["guard_ranks_1_10"] or record["missing_relevant_top100"] > 0
            ),
            "official_test_overlap_count": 0,
            "frozen_overlap_count": 0,
            "excluded_counts": dict(excluded_counts),
        }

    selected_payload = {
        "schema": SCHEMA,
        "selected_qids": selected_by_dataset,
        "policy": "q3 promotion rows first, then q3 recall/quality guards; frozen dev/reserve and official test qids excluded before ranking",
    }
    records_payload = {
        "schema": SCHEMA,
        "records": records_by_dataset,
    }
    write_json(args.output_selected_qids_json, selected_payload)
    write_json(args.output_records_json, records_payload)

    manifest = {
        "schema": SCHEMA,
        "created_utc": utc_stamp(),
        "selector": "scripts/select_q3_promotion_guard_qids.py",
        "config": {
            "rows_per_dataset": args.rows_per_dataset,
            "promotion_quota": args.promotion_quota,
        },
        "inputs": {
            "per_query_jsonl": per_query_meta,
            "train_qrels": {str(path): {"dataset": dataset_from_qrels_path(path), "sha256": sha256_file(path)} for path in args.train_qrels},
            "test_qrels": {str(path): {"dataset": dataset_from_qrels_path(path), "sha256": sha256_file(path)} for path in args.test_qrels},
            "exclude_qids_json": {str(path): {"sha256": sha256_file(path)} for path in args.exclude_qids_json},
        },
        "outputs": {
            "selected_qids_json": str(args.output_selected_qids_json),
            "records_json": str(args.output_records_json),
            "manifest": str(args.manifest),
        },
        "datasets": manifest_datasets,
        "legal_gates": dict(LEGAL_GATES),
        **LEGAL_GATES,
        "quality_claim": False,
    }
    manifest["sha256"] = {
        "selected_qids_json": sha256_file(args.output_selected_qids_json),
        "records_json": sha256_file(args.output_records_json),
    }
    write_json(args.manifest, manifest)
    print(
        "q3 promotion+guard qids: "
        + " ".join(f"{dataset}={len(qids)}" for dataset, qids in sorted(selected_by_dataset.items()))
    )
    print(f"selected: {args.output_selected_qids_json}")
    print(f"records: {args.output_records_json}")
    print(f"manifest: {args.manifest}")


if __name__ == "__main__":
    main()
