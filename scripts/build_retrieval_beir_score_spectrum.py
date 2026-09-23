#!/usr/bin/env python3
"""Materialize BEIR qrels and BM25/model negatives as score-spectrum JSONL."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


SCHEMA = "eos.retrieval_beir_score_spectrum.v2"
LEGAL_GATES = {
    "train_allowed_for_research": True,
    "release_train_allowed": False,
    "commercial_use_allowed": False,
}
TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--datasets", default="scifact,nfcorpus,fiqa")
    parser.add_argument("--split", default="train")
    parser.add_argument("--output-full-jsonl", required=True, type=Path)
    parser.add_argument("--output-train-jsonl", required=True, type=Path)
    parser.add_argument("--output-eval-jsonl", required=True, type=Path)
    parser.add_argument("--excluded-jsonl", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--extra-hard-negative-jsonl", action="append", default=[])
    parser.add_argument("--bm25-negatives", type=int, default=12)
    parser.add_argument("--extra-negatives", type=int, default=12)
    parser.add_argument("--eval-count", type=int, default=256)
    parser.add_argument("--max-queries-per-dataset", type=int, default=0)
    parser.add_argument("--rows-per-dataset", type=int, default=0)
    parser.add_argument("--candidate-cap", type=int, default=0)
    parser.add_argument("--qrel-positive-cap", type=int, default=5)
    parser.add_argument(
        "--qrel-positive-selection",
        choices=["doc_id", "gain_q3_boundary", "tiered_q3_recall"],
        default="doc_id",
    )
    parser.add_argument("--seed", type=int, default=191)
    parser.add_argument("--selected-qids-json", type=Path, default=None)
    parser.add_argument("--q3-per-query-jsonl", action="append", default=[])
    parser.add_argument("--q3-negatives", type=int, default=8)
    parser.add_argument(
        "--q3-selection-mode",
        choices=["raw_top", "disagreement", "boundary", "q3_boundary", "tiered_q3_recall"],
        default="disagreement",
    )
    parser.add_argument("--q3-method", default="turboquant_ip_b3")
    parser.add_argument("--q3-bits", type=int, default=3)
    parser.add_argument("--q3-quantizer-seed", type=int, default=5581486560434873699)
    parser.add_argument("--q3-scoring-surface", default="turboquant_ip_prepared")
    parser.add_argument("--require-q3-negatives", action="store_true")
    parser.add_argument("--hard-loss-weight", type=float, default=1.0)
    parser.add_argument("--soft-loss-weight", type=float, default=0.1)
    parser.add_argument("--recovery-loss-weight", type=float, default=1.0)
    return parser.parse_args()


def utc_stamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def stable_text(value: Any) -> str:
    return " ".join(str(value or "").replace("\r\n", "\n").split())


def stable_id(dataset: str, prefix: str, value: str) -> str:
    value = stable_text(value)
    if value:
        return f"{dataset}:{value}"
    digest = hashlib.sha256(f"{dataset}:{prefix}:{value}".encode("utf-8")).hexdigest()[:24]
    return f"{dataset}:{prefix}-{digest}"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
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


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")


def tokens(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def read_beir_jsonl_with_audit(path: Path, text_key: str) -> tuple[dict[str, str], Counter[str]]:
    out: dict[str, str] = {}
    audit: Counter[str] = Counter()
    for line_number, row in iter_jsonl(path):
        item_id = stable_text(row.get("_id"))
        text = stable_text(row.get(text_key))
        if text_key == "text":
            text = stable_text(f"{stable_text(row.get('title'))} {text}")
        if not item_id:
            raise ValueError(f"{path}:{line_number}: missing _id")
        if not text:
            audit["empty_text"] += 1
            continue
        out[item_id] = text
    return out, audit


def read_beir_jsonl(path: Path, text_key: str) -> dict[str, str]:
    out, _audit = read_beir_jsonl_with_audit(path, text_key)
    return out


def read_qrel_gains(path: Path) -> dict[str, dict[str, float]]:
    gains: dict[str, dict[str, float]] = defaultdict(dict)
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            line = raw.strip()
            if not line:
                continue
            fields = line.split("\t")
            if line_number == 1 and fields[0].lower() in {"query-id", "query_id", "qid"}:
                continue
            if len(fields) == 3:
                query_id, doc_id, score_raw = fields
            elif len(fields) == 4:
                query_id, _, doc_id, score_raw = fields
            else:
                raise ValueError(f"{path}:{line_number}: expected 3 or 4 qrels columns")
            try:
                score = float(score_raw)
            except ValueError as exc:
                raise ValueError(f"{path}:{line_number}: invalid qrel score {score_raw!r}") from exc
            if score > 0:
                gains[stable_text(query_id)][stable_text(doc_id)] = score
    return gains


def read_positive_qrels(path: Path) -> dict[str, set[str]]:
    return {query_id: set(doc_gains) for query_id, doc_gains in read_qrel_gains(path).items()}


def resolve_dataset_dir(dataset_root: Path, dataset: str) -> Path:
    candidates = [
        dataset_root / dataset,
        dataset_root / dataset / dataset,
    ]
    for candidate in candidates:
        if (
            (candidate / "corpus.jsonl").is_file()
            and (candidate / "queries.jsonl").is_file()
            and (candidate / "qrels").is_dir()
        ):
            return candidate
    return dataset_root / dataset


class BM25Index:
    def __init__(self, docs: dict[str, str]):
        self.docs = docs
        self.doc_terms: dict[str, list[str]] = {doc_id: tokens(text) for doc_id, text in docs.items()}
        self.doc_count = max(1, len(self.doc_terms))
        self.doc_lengths: dict[str, int] = {doc_id: len(terms) or 1 for doc_id, terms in self.doc_terms.items()}
        self.avgdl = sum(len(terms) for terms in self.doc_terms.values()) / self.doc_count
        self.avgdl = self.avgdl if self.avgdl > 0 else 1.0
        df: Counter[str] = Counter()
        postings: dict[str, dict[str, int]] = defaultdict(dict)
        for doc_id, terms in self.doc_terms.items():
            term_counts = Counter(terms)
            df.update(term_counts.keys())
            for term, freq in term_counts.items():
                postings[term][doc_id] = freq
        self.df = df
        self.postings = dict(postings)
        self.idf = {
            term: math.log(1.0 + (self.doc_count - freq + 0.5) / (freq + 0.5))
            for term, freq in df.items()
        }
        self.tokenized_doc_count = len(self.doc_terms)

    def rank(self, query: str, positive_ids: set[str], limit: int) -> list[tuple[str, float]]:
        if limit <= 0:
            return []
        query_terms = tokens(query)
        if not query_terms:
            return []
        k1 = 1.2
        b = 0.75
        query_counter = Counter(query_terms)
        candidate_ids: set[str] = set()
        for term in query_counter:
            candidate_ids.update(self.postings.get(term, {}).keys())
        scored: list[tuple[str, float]] = []
        for doc_id in candidate_ids:
            if doc_id in positive_ids:
                continue
            score = 0.0
            dl = self.doc_lengths.get(doc_id, 1)
            for term, qtf in query_counter.items():
                freq = self.postings.get(term, {}).get(doc_id, 0)
                if freq <= 0:
                    continue
                idf = self.idf.get(term, 0.0)
                denom = freq + k1 * (1.0 - b + b * dl / self.avgdl)
                score += qtf * idf * (freq * (k1 + 1.0) / denom)
            if score > 0:
                scored.append((doc_id, score))
        scored.sort(key=lambda item: (-item[1], item[0]))
        return scored[:limit]


def bm25_rank_naive(query: str, docs: dict[str, str], positive_ids: set[str], limit: int) -> list[tuple[str, float]]:
    if limit <= 0:
        return []
    query_terms = tokens(query)
    if not query_terms:
        return []
    doc_terms = {doc_id: tokens(text) for doc_id, text in docs.items()}
    doc_count = max(1, len(doc_terms))
    df: Counter[str] = Counter()
    for terms in doc_terms.values():
        df.update(set(terms))
    avgdl = sum(len(terms) for terms in doc_terms.values()) / doc_count
    avgdl = avgdl if avgdl > 0 else 1.0
    k1 = 1.2
    b = 0.75
    query_counter = Counter(query_terms)
    scored: list[tuple[str, float]] = []
    for doc_id, terms in doc_terms.items():
        if doc_id in positive_ids:
            continue
        term_counts = Counter(terms)
        score = 0.0
        dl = len(terms) or 1
        for term, qtf in query_counter.items():
            freq = term_counts.get(term, 0)
            if freq <= 0:
                continue
            idf = math.log(1.0 + (doc_count - df[term] + 0.5) / (df[term] + 0.5))
            denom = freq + k1 * (1.0 - b + b * dl / avgdl)
            score += qtf * idf * (freq * (k1 + 1.0) / denom)
        if score > 0:
            scored.append((doc_id, score))
    scored.sort(key=lambda item: (-item[1], item[0]))
    return scored[:limit]


def bm25_rank(query: str, index: BM25Index, positive_ids: set[str], limit: int) -> list[tuple[str, float]]:
    return index.rank(query, positive_ids, limit)


def dataset_from_source(source: str) -> str:
    return stable_text(source).split(":", 1)[0].lower()


def load_extra_negatives(paths: list[str]) -> dict[tuple[str, str], list[tuple[str, str, str]]]:
    extra: dict[tuple[str, str], list[tuple[str, str, str]]] = defaultdict(list)
    for raw_path in paths:
        path = Path(raw_path)
        if not path.is_file():
            raise SystemExit(f"missing extra hard-negative JSONL: {path}")
        for _, row in iter_jsonl(path):
            dataset = stable_text(row.get("dataset")).lower() or dataset_from_source(stable_text(row.get("source")))
            query_id = stable_text(row.get("query_id"))
            if not dataset or not query_id:
                continue
            neg_ids = [stable_text(value) for value in row.get("negative_doc_ids") or []]
            neg_texts = [stable_text(value) for value in row.get("negatives") or []]
            source = stable_text(row.get("source"))
            for index, text in enumerate(neg_texts):
                if not text:
                    continue
                doc_id = neg_ids[index] if index < len(neg_ids) and neg_ids[index] else ""
                extra[(dataset, query_id)].append((doc_id, text, source))
    return extra


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
        if not isinstance(values, list):
            raise SystemExit(f"{path}: selected qids for {dataset} must be a list")
        out[stable_text(dataset).lower()] = [stable_text(value) for value in values if stable_text(value)]
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
) -> tuple[dict[tuple[str, str], list[dict[str, Any]]], Counter[str], list[str]]:
    candidates: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    counts: Counter[str] = Counter()
    source_hashes: list[str] = []
    for raw_path in paths:
        path = Path(raw_path)
        if not path.is_file():
            raise SystemExit(f"missing q3 per-query JSONL: {path}")
        source_hashes.append(f"{path}:{sha256_file(path)}")
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
                        "source_path": str(path),
                        "q3_rank": rank,
                        "q3_score": top_doc.get("score"),
                        "dense_rank": top_doc.get("dense_rank"),
                        "dense_score": top_doc.get("dense_score"),
                        "compact_rank": top_doc.get("compact_rank"),
                        "compact_score": top_doc.get("compact_score"),
                    }
                )
                counts["q3_per_query_candidates_loaded"] += 1
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
    return candidates, counts, sorted(source_hashes)


def append_candidate(
    doc_ids: list[str],
    texts: list[str],
    hard_eligible: list[bool],
    candidate_sources: list[str],
    qrel_gains: list[float],
    seen_ids: set[str],
    seen_texts: set[str],
    doc_id: str,
    text: str,
    eligible: bool,
    source: str,
    gain: float,
) -> bool:
    doc_id = stable_text(doc_id)
    text = stable_text(text)
    if not doc_id or not text:
        return False
    if doc_id in seen_ids or text in seen_texts:
        return False
    seen_ids.add(doc_id)
    seen_texts.add(text)
    doc_ids.append(doc_id)
    texts.append(text)
    hard_eligible.append(eligible)
    candidate_sources.append(source)
    qrel_gains.append(gain)
    return True


def select_qrel_positive_ids(
    positive_ids: list[str],
    qrel_gains: dict[str, float],
    q3_candidates: list[dict[str, Any]],
    cap: int,
    mode: str,
) -> list[str]:
    if cap < 0:
        raise SystemExit("--qrel-positive-cap must be non-negative")
    if cap == 0:
        return []
    if len(positive_ids) <= cap:
        return positive_ids
    if mode == "doc_id":
        return positive_ids[:cap]
    if mode == "tiered_q3_recall":
        return select_tiered_q3_recall_positive_ids(positive_ids, qrel_gains, q3_candidates, cap)
    if mode != "gain_q3_boundary":
        raise SystemExit(f"unsupported qrel-positive-selection {mode!r}")
    boundary = 10
    q3_rank_by_doc: dict[str, int] = {}
    for candidate in q3_candidates:
        doc_id = stable_text(candidate.get("doc_id"))
        rank = int(candidate.get("q3_rank") or 0)
        if doc_id and rank > 0 and (doc_id not in q3_rank_by_doc or rank < q3_rank_by_doc[doc_id]):
            q3_rank_by_doc[doc_id] = rank

    def key(doc_id: str) -> tuple[int, int, int, float, int, str]:
        rank = q3_rank_by_doc.get(doc_id, 0)
        if rank > 0:
            # Prefer positives closest to the q3@10 decision boundary, with
            # just-outside/at-boundary positives before already-safe positives.
            side = 0 if rank >= boundary else 1
            return (0, abs(rank - boundary), side, -float(qrel_gains.get(doc_id, 0.0)), rank, doc_id)
        return (1, 1_000_000, 1, -float(qrel_gains.get(doc_id, 0.0)), 1_000_000, doc_id)

    return sorted(positive_ids, key=key)[:cap]


def select_tiered_q3_recall_positive_ids(
    positive_ids: list[str],
    qrel_gains: dict[str, float],
    q3_candidates: list[dict[str, Any]],
    cap: int,
) -> list[str]:
    q3_rank_by_doc: dict[str, int] = {}
    for candidate in q3_candidates:
        doc_id = stable_text(candidate.get("doc_id"))
        rank = int(candidate.get("q3_rank") or 0)
        if doc_id and rank > 0 and (doc_id not in q3_rank_by_doc or rank < q3_rank_by_doc[doc_id]):
            q3_rank_by_doc[doc_id] = rank
    quotas = [
        ("boundary_out", 4),
        ("guard_top10", 3),
        ("recall_sentinel", 2),
        ("coverage", 1),
    ]

    def tier(doc_id: str) -> str:
        rank = q3_rank_by_doc.get(doc_id, 0)
        if 11 <= rank <= 20:
            return "boundary_out"
        if 1 <= rank <= 10:
            return "guard_top10"
        if 80 <= rank <= 100:
            return "recall_sentinel"
        if 21 <= rank <= 79:
            return "coverage"
        return "fallback"

    def key(doc_id: str) -> tuple[int, float, int, str]:
        rank = q3_rank_by_doc.get(doc_id, 0)
        gain = -float(qrel_gains.get(doc_id, 0.0))
        if 11 <= rank <= 20:
            return (abs(rank - 10), gain, rank, doc_id)
        if 1 <= rank <= 10:
            return (abs(rank - 10), gain, rank, doc_id)
        if 80 <= rank <= 100:
            return (abs(rank - 100), gain, rank, doc_id)
        if 21 <= rank <= 79:
            return (rank, gain, rank, doc_id)
        return (1_000_000, gain, rank, doc_id)

    selected: list[str] = []
    selected_set: set[str] = set()
    remaining = list(positive_ids)
    for tier_name, quota in quotas:
        if len(selected) >= cap:
            break
        tier_docs = sorted((doc_id for doc_id in remaining if tier(doc_id) == tier_name), key=key)
        for doc_id in tier_docs[: min(quota, cap - len(selected))]:
            selected.append(doc_id)
            selected_set.add(doc_id)
        remaining = [doc_id for doc_id in remaining if doc_id not in selected_set]
    if len(selected) < cap:
        selected.extend(sorted(remaining, key=key)[: cap - len(selected)])
    return selected


def build_rows(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]], Counter[str]]:
    rows: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    extra = load_extra_negatives(getattr(args, "extra_hard_negative_jsonl", []))
    datasets = [item.strip().lower() for item in args.datasets.split(",") if item.strip()]
    selected_qids_input = load_selected_qids(getattr(args, "selected_qids_json", None))
    q3_candidates, q3_counts, q3_source_hashes = load_q3_per_query_candidates(
        getattr(args, "q3_per_query_jsonl", []),
        expected_method=getattr(args, "q3_method", "turboquant_ip_b3"),
        expected_bits=getattr(args, "q3_bits", 3),
        expected_seed=getattr(args, "q3_quantizer_seed", 5581486560434873699),
        expected_surface=getattr(args, "q3_scoring_surface", "turboquant_ip_prepared"),
        selection_mode=getattr(args, "q3_selection_mode", "disagreement"),
    )
    counts.update(q3_counts)
    candidate_cap_arg = getattr(args, "candidate_cap", 0)
    candidate_cap = candidate_cap_arg if candidate_cap_arg > 0 else 1 + args.bm25_negatives + args.extra_negatives
    if candidate_cap < 2:
        raise SystemExit("--candidate-cap must allow at least two candidates")
    rows_per_dataset = getattr(args, "rows_per_dataset", 0)
    if rows_per_dataset > 0 and candidate_cap < 20:
        raise SystemExit("--candidate-cap must be at least 20 for balanced train-only v2 rows")
    for dataset_index, dataset in enumerate(datasets):
        dataset_dir = resolve_dataset_dir(args.dataset_root, dataset)
        corpus_path = dataset_dir / "corpus.jsonl"
        queries_path = dataset_dir / "queries.jsonl"
        qrels_path = dataset_dir / "qrels" / f"{args.split}.tsv"
        if not corpus_path.is_file() or not queries_path.is_file() or not qrels_path.is_file():
            raise SystemExit(f"missing BEIR files for {dataset} under {dataset_dir}")
        docs, doc_audit = read_beir_jsonl_with_audit(corpus_path, "text")
        queries, query_audit = read_beir_jsonl_with_audit(queries_path, "text")
        qrel_gains_by_query = read_qrel_gains(qrels_path)
        qrels = {qid: set(doc_gains) for qid, doc_gains in qrel_gains_by_query.items()}
        bm25_index = BM25Index(docs)
        counts[f"{dataset}_empty_corpus_docs"] = doc_audit["empty_text"]
        counts[f"{dataset}_empty_queries"] = query_audit["empty_text"]
        counts[f"{dataset}_bm25_index_builds"] = 1
        counts[f"{dataset}_bm25_index_docs"] = bm25_index.tokenized_doc_count
        forced_qids = selected_qids_input.get(dataset, [])
        if forced_qids:
            selected_qids = [qid for qid in forced_qids if qid in qrels and qid in queries]
        else:
            selected_qids = sorted(qid for qid in qrels if qid in queries)
            if rows_per_dataset > 0:
                rng = random.Random(args.seed + dataset_index)
                rng.shuffle(selected_qids)
        if args.max_queries_per_dataset > 0:
            selected_qids = selected_qids[: args.max_queries_per_dataset]
        counts[f"{dataset}_queries_with_qrels"] = len(selected_qids)
        built_for_dataset = 0
        for qid in selected_qids:
            q3_negatives = getattr(args, "q3_negatives", 8)
            positive_ids = sorted(doc_id for doc_id in qrels[qid] if doc_id in docs)
            if not positive_ids:
                excluded.append({"dataset": dataset, "query_id": qid, "reason": "no_resolved_positive"})
                counts["excluded_no_resolved_positive"] += 1
                continue
            qrel_positive_cap = getattr(args, "qrel_positive_cap", 5)
            if len(positive_ids) > qrel_positive_cap:
                counts[f"{dataset}_qrel_positive_cap_applied"] += 1
                positive_ids = select_qrel_positive_ids(
                    positive_ids,
                    qrel_gains_by_query[qid],
                    q3_candidates.get((dataset, qid), []),
                    qrel_positive_cap,
                    getattr(args, "qrel_positive_selection", "doc_id"),
                )
            seen_ids: set[str] = set()
            seen_texts: set[str] = set()
            candidate_ids: list[str] = []
            candidate_texts: list[str] = []
            hard_eligible: list[bool] = []
            candidate_sources: list[str] = []
            qrel_gains: list[float] = []
            for doc_id in positive_ids:
                append_candidate(
                    candidate_ids,
                    candidate_texts,
                    hard_eligible,
                    candidate_sources,
                    qrel_gains,
                    seen_ids,
                    seen_texts,
                    stable_id(dataset, "d", doc_id),
                    docs[doc_id],
                    False,
                    "qrel",
                    qrel_gains_by_query[qid][doc_id],
                )
            positive_count = len(candidate_ids)
            if positive_count == 0:
                excluded.append({"dataset": dataset, "query_id": qid, "reason": "no_resolved_positive_after_cap"})
                counts["excluded_no_resolved_positive"] += 1
                continue
            appended_positive_doc_ids = [
                candidate_ids[index].split(":", 1)[1] if ":" in candidate_ids[index] else candidate_ids[index]
                for index in range(positive_count)
            ]
            qrel_positive_ids = set(doc_id for doc_id in qrels[qid] if doc_id in docs)
            q3_added = 0
            q3_evidence: list[dict[str, Any]] = []
            for candidate in q3_candidates.get((dataset, qid), []):
                if q3_added >= q3_negatives:
                    break
                if len(candidate_ids) >= candidate_cap:
                    break
                doc_id = stable_text(candidate.get("doc_id"))
                if doc_id in qrel_positive_ids:
                    continue
                added = append_candidate(
                    candidate_ids,
                    candidate_texts,
                    hard_eligible,
                    candidate_sources,
                    qrel_gains,
                    seen_ids,
                    seen_texts,
                    stable_id(dataset, "d", doc_id),
                    docs.get(doc_id, ""),
                    True,
                    "q3",
                    0.0,
                )
                if added:
                    q3_added += 1
                    counts["q3_candidates_added"] += 1
                    q3_evidence.append(
                        {
                            key: candidate[key]
                            for key in (
                                "doc_id",
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
            if getattr(args, "require_q3_negatives", False) and q3_added < q3_negatives:
                excluded.append({"dataset": dataset, "query_id": qid, "reason": "insufficient_q3_negatives"})
                counts[f"{dataset}_excluded_insufficient_q3_negatives"] += 1
                continue
            bm25_limit = max(args.bm25_negatives, candidate_cap)
            bm25_hits = bm25_rank(queries[qid], bm25_index, qrel_positive_ids, bm25_limit)
            bm25_added = 0
            for doc_id, _score in bm25_hits:
                if len(candidate_ids) >= candidate_cap:
                    break
                added = append_candidate(
                    candidate_ids,
                    candidate_texts,
                    hard_eligible,
                    candidate_sources,
                    qrel_gains,
                    seen_ids,
                    seen_texts,
                    stable_id(dataset, "d", doc_id),
                    docs[doc_id],
                    True,
                    "bm25",
                    0.0,
                )
                if added:
                    bm25_added += 1
                    counts["bm25_negatives_added"] += 1
            extra_added = 0
            for doc_id, text, _source in extra.get((dataset, qid), []):
                if extra_added >= args.extra_negatives:
                    break
                if len(candidate_ids) >= candidate_cap:
                    break
                if stable_text(doc_id) in qrel_positive_ids:
                    continue
                added = append_candidate(
                    candidate_ids,
                    candidate_texts,
                    hard_eligible,
                    candidate_sources,
                    qrel_gains,
                    seen_ids,
                    seen_texts,
                    stable_id(dataset, "d", doc_id),
                    text,
                    True,
                    "other",
                    0.0,
                )
                if added:
                    extra_added += 1
            if len(candidate_ids) == positive_count:
                excluded.append({"dataset": dataset, "query_id": qid, "reason": "no_unique_negative"})
                counts["excluded_no_unique_negative"] += 1
                continue
            if rows_per_dataset > 0 and len(candidate_ids) < 20:
                excluded.append({"dataset": dataset, "query_id": qid, "reason": "fewer_than_20_candidates"})
                counts[f"{dataset}_excluded_fewer_than_20_candidates"] += 1
                continue
            target = [0.0] * len(candidate_ids)
            gain_sum = sum(qrel_gains[:positive_count])
            for index in range(positive_count):
                target[index] = qrel_gains[index] / gain_sum if gain_sum > 0 else 1.0 / positive_count
            rows.append(
                {
                    "row_id": f"{dataset}:{qid}",
                    "source": f"{dataset}:beir-{args.split}:qrels-q3-bm25-score-spectrum-v2",
                    "query": queries[qid],
                    "candidate_doc_ids": candidate_ids,
                    "candidate_sources": candidate_sources,
                    "qrel_gains": qrel_gains,
                    "candidate_texts": candidate_texts,
                    "positive_indexes": list(range(positive_count)),
                    "selected_positive_index": 0,
                    "hard_negative_eligible": hard_eligible,
                    "target_probabilities": target,
                    "hard_loss_weight": args.hard_loss_weight,
                    "soft_loss_weight": args.soft_loss_weight,
                    "recovery_loss_weight": args.recovery_loss_weight,
                    "train_policy": "hard_soft_recovery_qrels_bm25_positive_wins_dedup_research_only",
                    "legal_gates": dict(LEGAL_GATES),
                    **LEGAL_GATES,
                    "source_artifact_hash": sha256_file(qrels_path),
                    "source_dataset": dataset,
                    "source_query_id": qid,
                    "source_positive_doc_ids": appended_positive_doc_ids,
                    "candidate_positive_doc_ids_pre_dedupe": positive_ids,
                    "all_train_qrel_positive_doc_ids": sorted(qrel_positive_ids),
                    "positive_count": positive_count,
                    "bm25_negative_count": bm25_added,
                    "q3_negative_count": q3_added,
                    "extra_negative_count": extra_added,
                    "q3_evidence": q3_evidence,
                    "q3_scoring_metadata": {
                        "method": getattr(args, "q3_method", "turboquant_ip_b3"),
                        "bits": getattr(args, "q3_bits", 3),
                        "scoring_surface": getattr(args, "q3_scoring_surface", "turboquant_ip_prepared"),
                        "quantizer_seed": getattr(args, "q3_quantizer_seed", 5581486560434873699),
                        "selection_mode": getattr(args, "q3_selection_mode", "disagreement"),
                        "source_hashes": q3_source_hashes,
                    },
                }
            )
            built_for_dataset += 1
            if rows_per_dataset > 0 and built_for_dataset >= rows_per_dataset:
                break
        if rows_per_dataset > 0 and built_for_dataset != rows_per_dataset:
            raise SystemExit(f"{dataset}: built {built_for_dataset} rows, expected {rows_per_dataset}")
    rows.sort(key=lambda row: row["row_id"])
    return rows, excluded, counts


def split_rows(rows: list[dict[str, Any]], eval_count: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if eval_count < 0:
        raise ValueError("--eval-count must be non-negative")
    if eval_count > 0 and len(rows) < 2:
        raise ValueError("at least two score-spectrum rows are required when --eval-count is positive")
    eval_count = min(eval_count, max(0, len(rows) - 1))
    eval_rows = rows[: min(eval_count, len(rows))]
    eval_ids = {row["row_id"] for row in eval_rows}
    train_rows = [row for row in rows if row["row_id"] not in eval_ids]
    if set(row["row_id"] for row in train_rows) & eval_ids:
        raise ValueError("score-spectrum train/eval split overlap")
    return train_rows, eval_rows


def main() -> None:
    args = parse_args()
    rows, excluded, counts = build_rows(args)
    if not rows:
        raise SystemExit("no score-spectrum rows materialized")
    try:
        train_rows, eval_rows = split_rows(rows, args.eval_count)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    candidate_cap = args.candidate_cap if args.candidate_cap > 0 else 1 + args.bm25_negatives + args.extra_negatives
    write_jsonl(args.output_full_jsonl, rows)
    write_jsonl(args.output_train_jsonl, train_rows)
    write_jsonl(args.output_eval_jsonl, eval_rows)
    write_jsonl(args.excluded_jsonl, excluded)
    dataset_counts = Counter(row["source_dataset"] for row in rows)
    manifest = {
        "schema": SCHEMA,
        "created_utc": utc_stamp(),
        "builder": "scripts/build_retrieval_beir_score_spectrum.py",
        "inputs": {
            "dataset_root": str(args.dataset_root),
            "datasets": [item.strip().lower() for item in args.datasets.split(",") if item.strip()],
            "split": args.split,
            "extra_hard_negative_jsonl": [str(path) for path in args.extra_hard_negative_jsonl],
            "q3_per_query_jsonl": [str(path) for path in args.q3_per_query_jsonl],
            "selected_qids_json": str(args.selected_qids_json) if args.selected_qids_json is not None else "",
        },
        "outputs": {
            "full_score_spectrum": str(args.output_full_jsonl),
            "train_score_spectrum": str(args.output_train_jsonl),
            "eval_score_spectrum": str(args.output_eval_jsonl),
            "excluded_jsonl": str(args.excluded_jsonl),
            "manifest": str(args.manifest),
        },
        "policy": "one deterministic row per dataset/query id; all qrel positives when <= cap, otherwise deterministic qrel-positive cap by configured selection mode; q3 disagreement/boundary negatives plus BM25 negatives; unique eligible negatives; positive-wins duplicate merge",
        "config": {
            "rows_per_dataset": args.rows_per_dataset,
            "candidate_cap": candidate_cap,
            "qrel_positive_cap": args.qrel_positive_cap,
            "qrel_positive_selection": args.qrel_positive_selection,
            "bm25_negatives": args.bm25_negatives,
            "extra_negatives": args.extra_negatives,
            "eval_count": args.eval_count,
            "max_queries_per_dataset": args.max_queries_per_dataset,
            "seed": args.seed,
            "q3_negatives": args.q3_negatives,
            "q3_method": args.q3_method,
            "q3_bits": args.q3_bits,
            "q3_quantizer_seed": args.q3_quantizer_seed,
            "q3_scoring_surface": args.q3_scoring_surface,
            "q3_selection_mode": args.q3_selection_mode,
            "require_q3_negatives": args.require_q3_negatives,
            "hard_loss_weight": args.hard_loss_weight,
            "soft_loss_weight": args.soft_loss_weight,
            "recovery_loss_weight": args.recovery_loss_weight,
        },
        "loss_mode": "hard_soft_recovery",
        "legal_gates": dict(LEGAL_GATES),
        "counts": {
            **dict(counts),
            "rows": len(rows),
            "train_rows": len(train_rows),
            "eval_rows": len(eval_rows),
            "excluded_rows": len(excluded),
            "dataset_counts": dict(dataset_counts),
            "max_candidate_count": max(len(row["candidate_texts"]) for row in rows),
            "min_candidate_count": min(len(row["candidate_texts"]) for row in rows),
            "max_positive_count": max(len(row["positive_indexes"]) for row in rows),
            "candidate_source_counts": dict(Counter(source for row in rows for source in row["candidate_sources"])),
        },
        "q3_source_hashes": sorted(
            {
                source_hash
                for row in rows
                for source_hash in row["q3_scoring_metadata"]["source_hashes"]
            }
        ),
        "sha256": {
            "full_score_spectrum": sha256_file(args.output_full_jsonl),
            "train_score_spectrum": sha256_file(args.output_train_jsonl),
            "eval_score_spectrum": sha256_file(args.output_eval_jsonl),
            "excluded_jsonl": sha256_file(args.excluded_jsonl),
        },
        "quality_claim": False,
    }
    write_json(args.manifest, manifest)
    print(
        "BEIR score-spectrum rows: "
        f"rows={len(rows)} train={len(train_rows)} eval={len(eval_rows)} excluded={len(excluded)} "
        f"datasets={dict(dataset_counts)}"
    )
    print(f"train: {args.output_train_jsonl}")
    print(f"eval: {args.output_eval_jsonl}")
    print(f"manifest: {args.manifest}")


if __name__ == "__main__":
    main()
