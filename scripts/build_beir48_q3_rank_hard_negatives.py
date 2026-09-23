#!/usr/bin/env python3
"""Build a tiny train-only BEIR hard-negative set for q3 rank-margin probes."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from build_retrieval_beir_score_spectrum import (  # noqa: E402
    BM25Index,
    bm25_rank,
    read_beir_jsonl_with_audit,
    read_positive_qrels,
    resolve_dataset_dir,
    sha256_file,
    stable_text,
)


SCHEMA = "eos.beir48_q3_rank_hard_negatives.v1"
LEGAL_GATES = {
    "train_allowed_for_research": True,
    "release_train_allowed": False,
    "commercial_use_allowed": False,
    "test_rows_train_allowed": False,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--datasets", default="scifact,nfcorpus,fiqa")
    parser.add_argument("--split", default="train")
    parser.add_argument("--rows-per-dataset", type=int, default=16)
    parser.add_argument("--negatives-per-row", type=int, default=7)
    parser.add_argument("--bm25-primary-negatives", type=int, default=4)
    parser.add_argument("--q3-negatives", type=int, default=3)
    parser.add_argument("--candidate-top-k", type=int, default=100)
    parser.add_argument("--seed", type=int, default=191)
    parser.add_argument("--q3-hard-negative-jsonl", action="append", default=[], help="legacy text hard-negative q3 candidate JSONL")
    parser.add_argument("--q3-per-query-jsonl", action="append", default=[], help="eval-retrieval-turboquant per-query JSONL with true q3 top-k rows")
    parser.add_argument(
        "--q3-selection-mode",
        choices=["raw_top", "disagreement", "boundary", "q3_boundary", "tiered_q3_recall"],
        default="disagreement",
    )
    parser.add_argument("--require-q3-negatives", action="store_true")
    parser.add_argument("--q3-method", default="turboquant_ip_b3")
    parser.add_argument("--q3-bits", type=int, default=3)
    parser.add_argument("--q3-dim", type=int, default=384)
    parser.add_argument("--q3-quantizer-seed", type=int, default=5581486560434873699)
    parser.add_argument("--q3-scoring-surface", default="turboquant_ip_prepared")
    parser.add_argument("--selected-qids-json", type=Path, default=None, help="reuse exact selected qids from a previous builder run")
    parser.add_argument("--emit-selected-qids-json", type=Path, default=None)
    parser.add_argument("--emit-qrels-subset-dir", type=Path, default=None)
    parser.add_argument("--output-jsonl", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    return parser.parse_args()


def utc_stamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


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


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_input_files(paths: list[str]) -> list[dict[str, str]]:
    hashed: list[dict[str, str]] = []
    for raw_path in paths:
        path = Path(raw_path)
        if path.is_file():
            hashed.append({"path": str(path), "sha256": sha256_file(path)})
    return hashed


def stable_dataset_doc_id(dataset: str, doc_id: str) -> str:
    return f"{dataset}:{stable_text(doc_id)}"


def source_dataset(source: str) -> str:
    return stable_text(source).split(":", 1)[0].lower()


def load_q3_candidates(paths: list[str]) -> dict[tuple[str, str], list[dict[str, Any]]]:
    candidates: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for raw_path in paths:
        path = Path(raw_path)
        if not path.is_file():
            raise SystemExit(f"missing q3 hard-negative JSONL: {path}")
        for line_number, row in iter_jsonl(path):
            dataset = stable_text(row.get("dataset")).lower() or source_dataset(stable_text(row.get("source")))
            query_id = stable_text(row.get("query_id"))
            if not dataset or not query_id:
                raise ValueError(f"{path}:{line_number}: q3 row missing dataset/source or query_id")
            neg_ids = [stable_text(value) for value in row.get("negative_doc_ids") or []]
            neg_texts = [stable_text(value) for value in row.get("negatives") or []]
            if neg_ids and len(neg_ids) != len(neg_texts):
                raise ValueError(f"{path}:{line_number}: negative_doc_ids length mismatch")
            for index, text in enumerate(neg_texts):
                if not text:
                    continue
                doc_id = neg_ids[index] if index < len(neg_ids) else ""
                if doc_id:
                    candidates[(dataset, query_id)].append(
                        {
                            "doc_id": doc_id,
                            "text": text,
                            "source_path": str(path),
                            "candidate_source": "q3_text_jsonl",
                        }
                    )
    return candidates


def load_selected_qids(path: Path | None) -> dict[str, list[str]]:
    if path is None:
        return {}
    if not path.is_file():
        raise SystemExit(f"missing selected qids JSON: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw = payload.get("selected_qids", payload)
    if not isinstance(raw, dict):
        raise SystemExit(f"{path}: selected qids JSON must contain an object")
    out: dict[str, list[str]] = {}
    for dataset, values in raw.items():
        clean_dataset = stable_text(dataset).lower()
        if not isinstance(values, list):
            raise SystemExit(f"{path}: selected qids for {dataset} must be a list")
        out[clean_dataset] = [stable_text(value) for value in values if stable_text(value)]
    return out


def per_query_candidate_sort_key(candidate: dict[str, Any], mode: str) -> tuple[float, int, str]:
    q3_rank = int(candidate.get("q3_rank") or 0)
    dense_rank = candidate.get("dense_rank")
    if mode == "raw_top":
        return (q3_rank, q3_rank, candidate["doc_id"])
    if mode == "boundary":
        boundary = 10
        if dense_rank:
            return (abs(int(dense_rank) - boundary), q3_rank, candidate["doc_id"])
        return (boundary + q3_rank, q3_rank, candidate["doc_id"])
    if mode == "q3_boundary":
        boundary = 10
        return (abs(q3_rank - boundary), q3_rank, candidate["doc_id"])
    if mode == "tiered_q3_recall":
        if 11 <= q3_rank <= 20:
            return (0, abs(q3_rank - 10), q3_rank, candidate["doc_id"])
        if 1 <= q3_rank <= 10:
            return (1, abs(q3_rank - 10), q3_rank, candidate["doc_id"])
        if 80 <= q3_rank <= 100:
            return (2, abs(q3_rank - 100), q3_rank, candidate["doc_id"])
        if 21 <= q3_rank <= 79:
            return (3, q3_rank, q3_rank, candidate["doc_id"])
        return (4, q3_rank, q3_rank, candidate["doc_id"])
    if dense_rank:
        promotion = int(dense_rank) - q3_rank
        if promotion > 0:
            return (-promotion, q3_rank, candidate["doc_id"])
    return (10_000 + q3_rank, q3_rank, candidate["doc_id"])


def load_q3_per_query_candidates(
    paths: list[str],
    *,
    expected_method: str,
    expected_bits: int,
    expected_seed: int,
    expected_surface: str,
    selection_mode: str,
) -> dict[tuple[str, str], list[dict[str, Any]]]:
    candidates: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for raw_path in paths:
        path = Path(raw_path)
        if not path.is_file():
            raise SystemExit(f"missing q3 per-query JSONL: {path}")
        for line_number, row in iter_jsonl(path):
            if row.get("method") != expected_method:
                continue
            if int(row.get("bits") or 0) != expected_bits:
                continue
            if int(row.get("quantizer_seed") or 0) != expected_seed:
                continue
            if stable_text(row.get("scoring_surface")) != expected_surface:
                continue
            dataset = stable_text(row.get("dataset")).lower()
            query_id = stable_text(row.get("query_id"))
            if not dataset or not query_id:
                raise ValueError(f"{path}:{line_number}: q3 per-query row missing dataset or query_id")
            top_docs = row.get("top_k") or []
            if not isinstance(top_docs, list):
                raise ValueError(f"{path}:{line_number}: top_k must be a list")
            for top_doc in top_docs:
                doc_id = stable_text(top_doc.get("doc_id"))
                rank = int(top_doc.get("rank") or 0)
                if not doc_id or rank <= 0:
                    continue
                candidates[(dataset, query_id)].append(
                    {
                        "doc_id": doc_id,
                        "text": "",
                        "source_path": str(path),
                        "candidate_source": "q3_per_query",
                        "q3_rank": rank,
                        "q3_score": top_doc.get("score"),
                        "dense_rank": top_doc.get("dense_rank"),
                        "dense_score": top_doc.get("dense_score"),
                        "compact_rank": top_doc.get("compact_rank"),
                        "compact_score": top_doc.get("compact_score"),
                    }
                )
    for key, values in list(candidates.items()):
        seen: set[str] = set()
        deduped: list[dict[str, Any]] = []
        for candidate in sorted(values, key=lambda item: per_query_candidate_sort_key(item, selection_mode)):
            doc_id = candidate["doc_id"]
            if doc_id in seen:
                continue
            seen.add(doc_id)
            deduped.append(candidate)
        candidates[key] = deduped
    return candidates


def merge_q3_candidates(
    text_candidates: dict[tuple[str, str], list[dict[str, Any]]],
    per_query_candidates: dict[tuple[str, str], list[dict[str, Any]]],
) -> dict[tuple[str, str], list[dict[str, Any]]]:
    out: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for source in (per_query_candidates, text_candidates):
        for key, values in source.items():
            out[key].extend(values)
    return out


def append_negative(
    negative_doc_ids: list[str],
    negatives: list[str],
    negative_sources: list[str],
    seen_doc_ids: set[str],
    seen_texts: set[str],
    qrel_positive_doc_ids: set[str],
    doc_id: str,
    text: str,
    source: str,
) -> bool:
    doc_id = stable_text(doc_id)
    text = stable_text(text)
    if not doc_id or not text:
        return False
    if doc_id in qrel_positive_doc_ids:
        return False
    if doc_id in seen_doc_ids or text in seen_texts:
        return False
    seen_doc_ids.add(doc_id)
    seen_texts.add(text)
    negative_doc_ids.append(doc_id)
    negatives.append(text)
    negative_sources.append(source)
    return True


def candidate_qids(qrels: dict[str, set[str]], queries: dict[str, str], docs: dict[str, str], seed: int) -> list[str]:
    qids = [
        qid
        for qid in sorted(qrels)
        if qid in queries and any(doc_id in docs for doc_id in qrels[qid])
    ]
    rng = random.Random(seed)
    rng.shuffle(qids)
    return qids


def build_dataset_rows(
    dataset_root: Path,
    dataset: str,
    split: str,
    rows_per_dataset: int,
    negatives_per_row: int,
    bm25_primary_negatives: int,
    q3_negatives: int,
    candidate_top_k: int,
    seed: int,
    q3_candidates: dict[tuple[str, str], list[dict[str, Any]]],
    require_q3_negatives: bool,
    forced_qids: list[str],
    counts: Counter[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    dataset_dir = resolve_dataset_dir(dataset_root, dataset)
    corpus_path = dataset_dir / "corpus.jsonl"
    queries_path = dataset_dir / "queries.jsonl"
    qrels_path = dataset_dir / "qrels" / f"{split}.tsv"
    if split != "train":
        raise SystemExit("this builder is train-only; --split must be train")
    if not corpus_path.is_file() or not queries_path.is_file() or not qrels_path.is_file():
        raise SystemExit(f"missing BEIR files for {dataset} under {dataset_dir}")

    docs, doc_audit = read_beir_jsonl_with_audit(corpus_path, "text")
    queries, query_audit = read_beir_jsonl_with_audit(queries_path, "text")
    qrels = read_positive_qrels(qrels_path)
    index = BM25Index(docs)
    rows: list[dict[str, Any]] = []
    shuffled_qids = forced_qids if forced_qids else candidate_qids(qrels, queries, docs, seed)

    for qid in shuffled_qids:
        resolved_positive_ids = sorted(doc_id for doc_id in qrels[qid] if doc_id in docs)
        if not resolved_positive_ids:
            continue
        positive_doc_id = resolved_positive_ids[0]
        qrel_positive_ids = set(resolved_positive_ids)
        seen_doc_ids = {positive_doc_id}
        seen_texts = {docs[positive_doc_id]}
        negative_doc_ids: list[str] = []
        negatives: list[str] = []
        negative_sources: list[str] = []

        bm25_hits = bm25_rank(queries[qid], index, qrel_positive_ids, candidate_top_k)
        for doc_id, _score in bm25_hits:
            if sum(1 for source in negative_sources if source == "bm25") >= bm25_primary_negatives:
                break
            append_negative(
                negative_doc_ids,
                negatives,
                negative_sources,
                seen_doc_ids,
                seen_texts,
                qrel_positive_ids,
                doc_id,
                docs.get(doc_id, ""),
                "bm25",
            )

        q3_evidence: list[dict[str, Any]] = []
        for candidate in q3_candidates.get((dataset, qid), []):
            if sum(1 for source in negative_sources if source == "q3") >= q3_negatives:
                break
            doc_id = stable_text(candidate.get("doc_id"))
            text = stable_text(candidate.get("text")) or docs.get(doc_id, "")
            added = append_negative(
                negative_doc_ids,
                negatives,
                negative_sources,
                seen_doc_ids,
                seen_texts,
                qrel_positive_ids,
                doc_id,
                text,
                "q3",
            )
            if added:
                source_path = stable_text(candidate.get("source_path"))
                counts[f"{dataset}_q3_source:{source_path}"] += 1
                q3_evidence.append(
                    {
                        key: candidate[key]
                        for key in (
                            "doc_id",
                            "candidate_source",
                            "source_path",
                            "q3_rank",
                            "q3_score",
                            "dense_rank",
                            "dense_score",
                            "compact_rank",
                            "compact_score",
                        )
                        if key in candidate and candidate[key] not in (None, "")
                    }
                )

        if require_q3_negatives and sum(1 for source in negative_sources if source == "q3") != q3_negatives:
            counts[f"{dataset}_skipped_insufficient_q3_negatives"] += 1
            if forced_qids:
                raise SystemExit(f"{dataset}:{qid}: insufficient q3 negatives")
            continue

        for doc_id, _score in bm25_hits:
            if len(negative_doc_ids) >= negatives_per_row:
                break
            append_negative(
                negative_doc_ids,
                negatives,
                negative_sources,
                seen_doc_ids,
                seen_texts,
                qrel_positive_ids,
                doc_id,
                docs.get(doc_id, ""),
                "bm25_fill",
            )

        if len(negative_doc_ids) != negatives_per_row:
            counts[f"{dataset}_skipped_insufficient_negatives"] += 1
            continue

        candidate_doc_ids = [positive_doc_id] + negative_doc_ids
        candidate_texts = [docs[positive_doc_id]] + negatives
        rows.append(
            {
                "row_id": f"{dataset}:{qid}",
                "source": f"{dataset}:beir-train:qrels-bm25-q3-rank-hard-negatives",
                "source_dataset": dataset,
                "query_id": qid,
                "positive_doc_id": positive_doc_id,
                "negative_doc_ids": negative_doc_ids,
                "query": queries[qid],
                "positive": docs[positive_doc_id],
                "negatives": negatives,
                "teacher_scores": [1.0] + [0.0] * negatives_per_row,
                "candidate_doc_ids": candidate_doc_ids,
                "candidate_texts": candidate_texts,
                "candidate_positive_index": 0,
                "negative_sources": negative_sources,
                "q3_evidence": q3_evidence,
                "all_train_qrel_positive_doc_ids": sorted(qrel_positive_ids),
                "train_policy": "train_qrels_only_positive_at_candidate0_bm25_primary_q3_optional_fill_research_only",
                "legal_gates": dict(LEGAL_GATES),
                **LEGAL_GATES,
            }
        )
        if len(rows) >= rows_per_dataset:
            break

    if len(rows) != rows_per_dataset:
        raise SystemExit(
            f"{dataset}: built {len(rows)} rows, expected {rows_per_dataset}; "
            f"skipped_insufficient_negatives={counts[f'{dataset}_skipped_insufficient_negatives']}"
        )

    dataset_manifest = {
        "dataset_dir": str(dataset_dir),
        "corpus": str(corpus_path),
        "queries": str(queries_path),
        "qrels": str(qrels_path),
        "corpus_sha256": sha256_file(corpus_path),
        "queries_sha256": sha256_file(queries_path),
        "qrels_sha256": sha256_file(qrels_path),
        "queries_with_train_qrels": len([qid for qid in qrels if qid in queries]),
        "empty_corpus_docs": doc_audit["empty_text"],
        "empty_queries": query_audit["empty_text"],
        "bm25_index_docs": index.tokenized_doc_count,
        "selected_qids": [row["query_id"] for row in rows],
    }
    return rows, dataset_manifest


def audit_rows(rows: list[dict[str, Any]], datasets: list[str], rows_per_dataset: int, negatives_per_row: int, bm25_primary_negatives: int, q3_negatives: int, require_q3_negatives: bool) -> dict[str, Any]:
    counts = Counter(row["source_dataset"] for row in rows)
    errors: list[str] = []
    if len(rows) != rows_per_dataset * len(datasets):
        errors.append(f"row count {len(rows)} != {rows_per_dataset * len(datasets)}")
    for dataset in datasets:
        if counts[dataset] != rows_per_dataset:
            errors.append(f"{dataset} row count {counts[dataset]} != {rows_per_dataset}")
    for row in rows:
        row_id = row["row_id"]
        if row.get("candidate_positive_index") != 0:
            errors.append(f"{row_id}: candidate_positive_index is not 0")
        candidate_ids = row.get("candidate_doc_ids") or []
        candidate_texts = row.get("candidate_texts") or []
        expected_candidates = negatives_per_row + 1
        if len(candidate_ids) != expected_candidates or len(candidate_texts) != expected_candidates:
            errors.append(f"{row_id}: candidate length mismatch")
        if candidate_ids[:1] != [row.get("positive_doc_id")]:
            errors.append(f"{row_id}: candidate[0] doc is not positive_doc_id")
        if candidate_texts[:1] != [row.get("positive")]:
            errors.append(f"{row_id}: candidate[0] text is not positive")
        if len(row.get("negative_doc_ids") or []) != negatives_per_row:
            errors.append(f"{row_id}: negative_doc_ids length mismatch")
        if len(row.get("negatives") or []) != negatives_per_row:
            errors.append(f"{row_id}: negatives length mismatch")
        sources = row.get("negative_sources") or []
        if len(sources) != negatives_per_row:
            errors.append(f"{row_id}: negative_sources length mismatch")
        if sources[:bm25_primary_negatives] != ["bm25"] * bm25_primary_negatives:
            errors.append(f"{row_id}: primary BM25 source prefix mismatch")
        if require_q3_negatives and sources.count("q3") != q3_negatives:
            errors.append(f"{row_id}: q3 source count {sources.count('q3')} != {q3_negatives}")
        if require_q3_negatives and len(row.get("q3_evidence") or []) != q3_negatives:
            errors.append(f"{row_id}: q3 evidence count mismatch")
        if len(set(candidate_ids)) != len(candidate_ids):
            errors.append(f"{row_id}: duplicate candidate doc id")
        if len(set(candidate_texts)) != len(candidate_texts):
            errors.append(f"{row_id}: duplicate candidate text")
        qrel_positive_ids = set(row.get("all_train_qrel_positive_doc_ids") or [])
        leaked = sorted(set(row.get("negative_doc_ids") or []) & qrel_positive_ids)
        if leaked:
            errors.append(f"{row_id}: qrel-positive negative leak {leaked}")
        if row.get("legal_gates") != LEGAL_GATES:
            errors.append(f"{row_id}: legal_gates mismatch")
        for key, expected in LEGAL_GATES.items():
            if row.get(key) != expected:
                errors.append(f"{row_id}: {key} mismatch")
    return {
        "passed": not errors,
        "errors": errors,
        "rows": len(rows),
        "dataset_counts": dict(counts),
        "candidates_per_row": negatives_per_row + 1,
        "positive_index": 0,
        "legal_gates": dict(LEGAL_GATES),
    }


def write_selected_qids(path: Path, selected: dict[str, list[str]], args: argparse.Namespace) -> None:
    write_json(
        path,
        {
            "schema": "eos.beir48_q3_rank_hard_negatives.selected_qids.v1",
            "created_utc": utc_stamp(),
            "selected_qids": selected,
            "seed": args.seed,
            "datasets": [stable_text(item).lower() for item in args.datasets.split(",") if stable_text(item)],
            "split": args.split,
            "rows_per_dataset": args.rows_per_dataset,
            "selection_basis": "train_qrels_only_bm25_viable_rows",
        },
    )


def write_qrels_subset(root: Path, dataset: str, qids: list[str], qrels: dict[str, set[str]], source_qrels_path: Path) -> dict[str, Any]:
    path = root / dataset / "qrels" / "train.tsv"
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = 0
    with path.open("w", encoding="utf-8") as handle:
        handle.write("query-id\tcorpus-id\tscore\n")
        for qid in qids:
            for doc_id in sorted(qrels.get(qid, set())):
                handle.write(f"{qid}\t{doc_id}\t1\n")
                rows += 1
    return {
        "path": str(path),
        "source_qrels": str(source_qrels_path),
        "sha256": sha256_file(path),
        "queries": len(qids),
        "qrel_pairs": rows,
    }


def main() -> int:
    args = parse_args()
    datasets = [stable_text(item).lower() for item in args.datasets.split(",") if stable_text(item)]
    if not datasets:
        raise SystemExit("--datasets must name at least one dataset")
    if args.rows_per_dataset <= 0:
        raise SystemExit("--rows-per-dataset must be positive")
    if args.negatives_per_row <= 0:
        raise SystemExit("--negatives-per-row must be positive")
    if args.bm25_primary_negatives < 0 or args.q3_negatives < 0:
        raise SystemExit("--bm25-primary-negatives and --q3-negatives must be non-negative")
    if args.bm25_primary_negatives > args.negatives_per_row:
        raise SystemExit("--bm25-primary-negatives cannot exceed --negatives-per-row")
    if args.candidate_top_k < args.negatives_per_row:
        raise SystemExit("--candidate-top-k must be at least --negatives-per-row")

    if args.bm25_primary_negatives + args.q3_negatives > args.negatives_per_row:
        raise SystemExit("--bm25-primary-negatives + --q3-negatives cannot exceed --negatives-per-row")
    if args.require_q3_negatives and args.q3_negatives <= 0:
        raise SystemExit("--require-q3-negatives requires --q3-negatives > 0")

    selected_qids_input = load_selected_qids(args.selected_qids_json)
    text_q3_candidates = load_q3_candidates(args.q3_hard_negative_jsonl)
    per_query_q3_candidates = load_q3_per_query_candidates(
        args.q3_per_query_jsonl,
        expected_method=args.q3_method,
        expected_bits=args.q3_bits,
        expected_seed=args.q3_quantizer_seed,
        expected_surface=args.q3_scoring_surface,
        selection_mode=args.q3_selection_mode,
    )
    q3_candidates = merge_q3_candidates(text_q3_candidates, per_query_q3_candidates)
    counts: Counter[str] = Counter()
    rows: list[dict[str, Any]] = []
    inputs: dict[str, Any] = {}
    selected_qids: dict[str, list[str]] = {}
    qrels_subsets: dict[str, Any] = {}
    for index, dataset in enumerate(datasets):
        forced_qids = selected_qids_input.get(dataset, [])
        dataset_rows, dataset_inputs = build_dataset_rows(
            args.dataset_root,
            dataset,
            args.split,
            args.rows_per_dataset,
            args.negatives_per_row,
            args.bm25_primary_negatives,
            args.q3_negatives,
            args.candidate_top_k,
            args.seed + index,
            q3_candidates,
            args.require_q3_negatives,
            forced_qids,
            counts,
        )
        rows.extend(dataset_rows)
        inputs[dataset] = dataset_inputs
        selected_qids[dataset] = [row["query_id"] for row in dataset_rows]
        if args.emit_qrels_subset_dir is not None:
            qrels_path = Path(dataset_inputs["qrels"])
            qrels_subsets[dataset] = write_qrels_subset(
                args.emit_qrels_subset_dir,
                dataset,
                selected_qids[dataset],
                read_positive_qrels(qrels_path),
                qrels_path,
            )

    rows.sort(key=lambda row: (row["source_dataset"], row["row_id"]))
    write_jsonl(args.output_jsonl, rows)
    if args.emit_selected_qids_json is not None:
        write_selected_qids(args.emit_selected_qids_json, selected_qids, args)
    audit = audit_rows(
        rows,
        datasets,
        args.rows_per_dataset,
        args.negatives_per_row,
        args.bm25_primary_negatives,
        args.q3_negatives,
        args.require_q3_negatives,
    )
    if not audit["passed"]:
        raise SystemExit(f"artifact audit failed: {audit['errors'][:8]}")
    source_counts = Counter(source for row in rows for source in row["negative_sources"])
    manifest = {
        "schema": SCHEMA,
        "created_utc": utc_stamp(),
        "builder": "scripts/build_beir48_q3_rank_hard_negatives.py",
        "seed": args.seed,
        "inputs": {
            "dataset_root": str(args.dataset_root),
            "datasets": datasets,
            "split": args.split,
            "q3_hard_negative_jsonl": [str(path) for path in args.q3_hard_negative_jsonl],
            "q3_per_query_jsonl": [str(path) for path in args.q3_per_query_jsonl],
            "q3_hard_negative_hashes": hash_input_files(args.q3_hard_negative_jsonl),
            "q3_per_query_hashes": hash_input_files(args.q3_per_query_jsonl),
            "selected_qids_json": str(args.selected_qids_json) if args.selected_qids_json else "",
            "beir": inputs,
        },
        "outputs": {
            "hard_negatives": str(args.output_jsonl),
            "manifest": str(args.manifest),
            "selected_qids_json": str(args.emit_selected_qids_json) if args.emit_selected_qids_json else "",
            "qrels_subsets": qrels_subsets,
        },
        "config": {
            "rows_per_dataset": args.rows_per_dataset,
            "negatives_per_row": args.negatives_per_row,
            "bm25_primary_negatives": args.bm25_primary_negatives,
            "q3_negatives": args.q3_negatives,
            "candidate_top_k": args.candidate_top_k,
            "q3_selection_mode": args.q3_selection_mode,
            "require_q3_negatives": args.require_q3_negatives,
            "q3_method": args.q3_method,
            "q3_bits": args.q3_bits,
            "q3_dim": args.q3_dim,
            "q3_dim_source": "caller_declared_unverified",
            "q3_dim_validated": False,
            "q3_quantizer_seed": args.q3_quantizer_seed,
            "q3_scoring_surface": args.q3_scoring_surface,
        },
        "legal_gates": dict(LEGAL_GATES),
        "legal_provenance": "BEIR train qrels only; dev/test qrels are not read; research-only training artifact, not release/commercial-cleared data.",
        "audit": audit,
        "counts": {
            **dict(counts),
            "rows": len(rows),
            "candidate_texts": sum(len(row["candidate_texts"]) for row in rows),
            "negative_source_counts": dict(source_counts),
            "q3_candidates_loaded": sum(len(values) for values in q3_candidates.values()),
            "q3_text_candidates_loaded": sum(len(values) for values in text_q3_candidates.values()),
            "q3_per_query_candidates_loaded": sum(len(values) for values in per_query_q3_candidates.values()),
            "q3_candidates_added": source_counts.get("q3", 0),
            "bm25_primary_negatives_added": source_counts.get("bm25", 0),
            "bm25_fill_negatives_added": source_counts.get("bm25_fill", 0),
        },
        "sha256": {
            "hard_negatives": sha256_file(args.output_jsonl),
        },
        "quality_claim": False,
    }
    write_json(args.manifest, manifest)
    print(
        "BEIR48 q3-rank hard negatives: "
        f"rows={len(rows)} candidates_per_row={args.negatives_per_row + 1} "
        f"datasets={dict(audit['dataset_counts'])} q3_added={source_counts.get('q3', 0)}"
    )
    print(f"output: {args.output_jsonl}")
    print(f"manifest: {args.manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
