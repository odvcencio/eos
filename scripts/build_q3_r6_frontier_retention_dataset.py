#!/usr/bin/env python3
"""Build the q3 R6 frontier-retention train/probe dataset."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import build_q3_r4_broad_gap_dataset as r4  # noqa: E402
from build_q3_r3_incumbent_dataset import canonical_stats, eligible_topk_pairs  # noqa: E402


SCHEMA = "eos.q3_r6_frontier_retention_dataset.v1"
PROBE_SCHEMA = "eos.q3_r6_frontier_retention_probe_manifest.v1"
ARM_ID = "arm-q3r6-frontier-retain-alltrain-b0-rw050-q3w004-m002-lr2e6-e2"
SUBTREE = "headroom-q3r6-frontier-retain-alltrain-b0-rw050"
ACTIVE_ROOT = Path("runs/eos-d384-q3-boundary-goal-v1-20260902T084144Z")
DEFAULT_OUTPUT_ROOT = ACTIVE_ROOT / SUBTREE
ANCHOR_PACKAGE = Path("runs/eos-wide384-projection-tail-seed191-repeat-v1-20260902T004753Z/packages/d384-pre.mll")
CANDIDATE_PACKAGE = ACTIVE_ROOT / "packages" / ARM_ID / "d384-pre.mll"
R6_BINARY = ACTIVE_ROOT / "bin/eos-r6-frontier-retention"
R6_BINARY_SHA256 = "cc474f5503827b607c53333832a5e173ba85d5d1bbb67ac94bc0f0f77e33abca"
LEGAL_GATES = {
    "train_allowed_for_research": True,
    "release_train_allowed": False,
    "commercial_use_allowed": False,
    "free_open_release_allowed": False,
}
FRONTIER_BUCKET = "frontier_context_q3"
TOP10_BUCKET = "top10_retention_q3"
NF_BOUNDARY_BUCKET = "nf_boundary80_100_docretention_q3"
EXPECTED_BUCKETS = {FRONTIER_BUCKET: 110, TOP10_BUCKET: 405, NF_BOUNDARY_BUCKET: 640}
EXPECTED_DOMAIN_BUCKETS = {
    FRONTIER_BUCKET: {"fiqa": 33, "nfcorpus": 64, "scifact": 13},
    TOP10_BUCKET: {"fiqa": 108, "nfcorpus": 234, "scifact": 63},
    NF_BOUNDARY_BUCKET: {"nfcorpus": 640},
}
EXPECTED_WORKLOAD = {
    "train": 1155,
    "batch": 4,
    "steps_per_epoch": 289,
    "epochs": 2,
    "raw_candidates_per_epoch": 11183,
    "canonical_candidates_per_epoch": 11167,
    "alias_drops_per_epoch": 16,
    "main_q3_pairs_per_epoch": 10794,
    "recall_q3_pairs_per_epoch": 5120,
    "total_train_pairs_per_epoch": 27081,
    "planned_work_units": 54162,
    "actual_train_pairs": 0,
}
DEFAULT_Q3 = [
    ACTIVE_ROOT / "headroom96/per-query/anchor.fiqa.train384.q3.per-query.jsonl",
    ACTIVE_ROOT / "headroom96/per-query/anchor.nfcorpus.train384.q3.per-query.jsonl",
    ACTIVE_ROOT / "headroom96/per-query/anchor.scifact.train384.q3.per-query.jsonl",
]
DEFAULT_EXCLUDES = [
    ACTIVE_ROOT / "data/selected-qids.dev4.json",
    ACTIVE_ROOT / "data/selected-qids.reserve4.json",
]
DATASET_DIRS = {
    "fiqa": Path("datasets/manta-embed-v1/raw/fiqa/fiqa"),
    "nfcorpus": Path("datasets/manta-embed-v1/raw/nfcorpus/nfcorpus"),
    "scifact": Path("datasets/manta-embed-v1/raw/scifact/scifact"),
}
PROBE_SURFACES = [
    (FRONTIER_BUCKET, "fiqa"),
    (FRONTIER_BUCKET, "nfcorpus"),
    (FRONTIER_BUCKET, "scifact"),
    (TOP10_BUCKET, "fiqa"),
    (TOP10_BUCKET, "nfcorpus"),
    (TOP10_BUCKET, "scifact"),
    (NF_BOUNDARY_BUCKET, "nfcorpus"),
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path("datasets/manta-embed-v1/raw"))
    parser.add_argument("--q3-per-query-jsonl", action="append", type=Path, default=[])
    parser.add_argument("--exclude-qids-json", action="append", type=Path, default=[])
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--output-jsonl", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--coverage-report", type=Path)
    parser.add_argument("--probe-manifest", type=Path)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--expected-rows", type=int, default=EXPECTED_WORKLOAD["train"])
    parser.add_argument("--expected-q3-sha256", action="append", default=[])
    return parser.parse_args(argv)


def repo_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def require_resolved_within(path: Path, root: Path, message: str) -> None:
    resolved = repo_path(path).resolve()
    resolved_root = repo_path(root).resolve()
    try:
        resolved.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError(message) from exc


def display_path(path: Path) -> str:
    try:
        return str(repo_path(path).resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(repo_path(path).resolve())


def utc_stamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with repo_path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_json(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(repo_path(path).read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    target = repo_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    target = repo_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")


def write_qrels(path: Path, rows: list[tuple[str, str, int]]) -> None:
    target = repo_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["query-id", "corpus-id", "score"])
        for qid, doc_id, gain in rows:
            writer.writerow([qid, doc_id, gain])


def qrels_file_stats(path: Path) -> dict[str, Any]:
    rows = []
    with repo_path(path).open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if not reader.fieldnames or not {"query-id", "corpus-id", "score"}.issubset(reader.fieldnames):
            raise ValueError(f"{path}: expected BEIR qrels TSV header")
        for row in reader:
            rows.append((str(row["query-id"]), str(row["corpus-id"]), int(row["score"])))
    return {
        "path": display_path(path),
        "sha256": sha256_file(path),
        "bytes": repo_path(path).stat().st_size,
        "qid_count": len({qid for qid, _doc_id, _gain in rows}),
        "qrel_count": len(rows),
        "gain_histogram": dict(sorted(Counter(str(gain) for _qid, _doc_id, gain in rows).items())),
        "content_sha256": sha256_json([{"qid": qid, "doc_id": doc_id, "gain": gain} for qid, doc_id, gain in sorted(rows)]),
    }


def expected_q3_hashes(args: argparse.Namespace) -> dict[str, str]:
    values = dict(r4.DEFAULT_EXPECTED_Q3_HASHES)
    for item in args.expected_q3_sha256:
        dataset, digest = item.split("=", 1)
        values[dataset.lower()] = digest
    return values


def raw_doc_id(dataset: str, doc_id: str) -> str:
    prefix = f"{dataset}:"
    return doc_id[len(prefix) :] if doc_id.startswith(prefix) else doc_id


def candidate_evidence(candidate: dict[str, Any], source_path: Path) -> dict[str, Any]:
    return r4.candidate_evidence(candidate, source_path)


def canonical_select(
    dataset: str,
    qid: str,
    candidates: list[dict[str, Any]],
    gains: dict[str, int],
    corpus: dict[str, str],
    gain_for: Any,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    positives_by_alias: dict[str, list[dict[str, Any]]] = defaultdict(list)
    negatives: list[dict[str, Any]] = []
    for candidate in candidates:
        doc_id = str(candidate["doc_id"])
        if doc_id not in corpus:
            raise ValueError(f"{dataset}:{qid}: missing corpus text for {doc_id}")
        raw_gain = gains.get(doc_id, 0)
        item = {"candidate": candidate, "doc_id": doc_id, "rank": int(candidate["rank"]), "score": float(candidate["score"]), "raw_gain": raw_gain, "text": corpus[doc_id]}
        if raw_gain > 0:
            positives_by_alias[r4.normalize_candidate_text(item["text"])].append(item)
        else:
            negatives.append(item)
    selected: list[dict[str, Any]] = []
    positive_aliases = set(positives_by_alias)
    positive_alias_drops = 0
    for alias, items in positives_by_alias.items():
        items.sort(key=lambda item: (-int(item["raw_gain"]), int(item["rank"]), str(item["doc_id"])))
        selected.append({**items[0], "gain": gain_for(items[0]), "source": "qrel", "positive_alias_count": len(items)})
        positive_alias_drops += len(items) - 1
    nonrel_alias_drops = 0
    for item in negatives:
        if r4.normalize_candidate_text(item["text"]) in positive_aliases:
            nonrel_alias_drops += 1
            continue
        selected.append({**item, "gain": 0, "source": "q3", "positive_alias_count": 0})
    selected.sort(key=lambda item: (int(item["rank"]), str(item["doc_id"])))
    if not any(item["gain"] > 0 for item in selected):
        raise ValueError(f"{dataset}:{qid}: missing retained positive")
    if not any(item["gain"] <= 0 for item in selected):
        raise ValueError(f"{dataset}:{qid}: missing retained hard negative")
    return selected, {"positive_alias_drops": positive_alias_drops, "nonrel_alias_drops": nonrel_alias_drops}


def build_row(
    *,
    dataset: str,
    qid: str,
    query: str,
    selected: list[dict[str, Any]],
    bucket: str,
    source_path: Path,
    main_weight: float,
    recall_weight: float,
    provenance: dict[str, Any],
) -> dict[str, Any]:
    gains = [float(item["gain"]) for item in selected]
    positive_indexes = [index for index, gain in enumerate(gains) if gain > 0.0]
    gain_sum = sum(gains[index] for index in positive_indexes)
    if gain_sum <= 0:
        raise ValueError(f"{dataset}:{qid}: zero gain sum")
    target_probabilities = [gains[index] / gain_sum if index in positive_indexes else 0.0 for index in range(len(selected))]
    row = {
        "row_id": f"{dataset}:{qid}:r6-{bucket}",
        "source": f"{dataset}:beir-train:q3-r6-frontier-retention-v1",
        "query": query,
        "candidate_doc_ids": [f"{dataset}:{item['doc_id']}" for item in selected],
        "candidate_texts": [item["text"] for item in selected],
        "candidate_sources": [item["source"] for item in selected],
        "qrel_gains": gains,
        "positive_indexes": positive_indexes,
        "selected_positive_index": positive_indexes[0],
        "hard_negative_eligible": [gain <= 0.0 for gain in gains],
        "target_probabilities": target_probabilities,
        "base_loss_weight": 0.0,
        "hard_loss_weight": 0.0,
        "soft_loss_weight": 0.0,
        "recovery_loss_weight": 0.0,
        "turboquant_topk_loss_weight": main_weight,
        "turboquant_topk_recall_loss_weight": recall_weight,
        "train_policy": "q3_r6_frontier_retention_research_only",
        "selection_bucket": bucket,
        "legal_gates": dict(LEGAL_GATES),
        **LEGAL_GATES,
        "quality_claim": False,
        "source_artifact_hash": sha256_json(provenance),
        "source_dataset": dataset,
        "source_query_id": qid,
        "source_positive_doc_ids": [selected[index]["doc_id"] for index in positive_indexes],
        "candidate_positive_doc_ids_pre_dedupe": [item["doc_id"] for item in selected if item["raw_gain"] > 0],
        "positive_count": len(positive_indexes),
        "bm25_negative_count": 0,
        "q3_negative_count": sum(1 for gain in gains if gain <= 0.0),
        "extra_negative_count": 0,
        "q3_positive_text_alias_count": sum(int(item["positive_alias_count"]) - 1 for item in selected if item["positive_alias_count"]),
        "q3_positive_text_alias_candidate_indexes": [],
        "q3_evidence": [candidate_evidence(item["candidate"], source_path) for item in selected if item["gain"] <= 0],
        "anchor_q3_candidate_evidence": [candidate_evidence(item["candidate"], source_path) for item in selected],
        "q3_scoring_metadata": {
            "method": r4.Q3_METHOD,
            "bits": r4.Q3_BITS,
            "dim": r4.Q3_DIM,
            "scoring_surface": r4.Q3_SURFACE,
            "quantizer_seed": r4.Q3_SEED,
        },
        "r6_provenance": provenance,
    }
    validate_row(row)
    return row


def validate_row(row: dict[str, Any]) -> None:
    sizes = {key: len(row[key]) for key in ("candidate_doc_ids", "candidate_texts", "candidate_sources", "qrel_gains", "hard_negative_eligible", "target_probabilities")}
    if len(set(sizes.values())) != 1:
        raise ValueError(f"{row['row_id']}: aligned fields drift {sizes}")
    if row["legal_gates"] != LEGAL_GATES or row["train_allowed_for_research"] is not True or row["release_train_allowed"] is not False or row["commercial_use_allowed"] is not False or row["free_open_release_allowed"] is not False:
        raise ValueError(f"{row['row_id']}: legal gate drift")
    if row["quality_claim"] is not False:
        raise ValueError(f"{row['row_id']}: quality claim must be false")
    if float(row.get("base_loss_weight", 1.0)) != 0.0:
        raise ValueError(f"{row['row_id']}: base_loss_weight must be explicit zero")
    if any(float(row.get(name, 0.0)) != 0.0 for name in ("hard_loss_weight", "soft_loss_weight", "recovery_loss_weight")):
        raise ValueError(f"{row['row_id']}: base hard/soft/recovery weights must be zero")
    if float(row.get("turboquant_topk_loss_weight", 0.0)) <= 0.0 and float(row.get("turboquant_topk_recall_loss_weight", 0.0)) <= 0.0:
        raise ValueError(f"{row['row_id']}: inactive aux objectives")
    positives = [index for index, gain in enumerate(row["qrel_gains"]) if float(gain) > 0.0]
    if positives != row["positive_indexes"]:
        raise ValueError(f"{row['row_id']}: positive indexes drift")
    positive_aliases = {r4.normalize_candidate_text(row["candidate_texts"][index]) for index in positives}
    for index, hard in enumerate(row["hard_negative_eligible"]):
        if hard and r4.normalize_candidate_text(row["candidate_texts"][index]) in positive_aliases:
            raise ValueError(f"{row['row_id']}: retained hard negative aliases positive")
    if float(row.get("turboquant_topk_loss_weight", 0.0)) > 0 and eligible_topk_pairs(row, False, "q3") <= 0:
        raise ValueError(f"{row['row_id']}: zero main q3 pairs")
    if float(row.get("turboquant_topk_recall_loss_weight", 0.0)) > 0 and eligible_topk_pairs(row, True, "q3") <= 0:
        raise ValueError(f"{row['row_id']}: zero recall q3 pairs")


def allowed_qid(dataset: str, qid: str, sources: dict[str, Any], excluded: dict[str, set[str]]) -> bool:
    return qid in sources["qrels"] and qid not in excluded.get(dataset, set()) and qid not in sources["test_qids"]


def front_features(row: dict[str, Any], gains: dict[str, int]) -> dict[str, Any]:
    return r4.q3_gap_features(row, {str(row["query_id"]): gains})


def select_frontier_rows(q3_rows: dict[str, dict[str, dict[str, Any]]], sources: dict[str, dict[str, Any]], excluded: dict[str, set[str]], q3_source: dict[str, Path]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    audit: dict[str, Any] = {"promotion_inventory": {}, "guard_inventory": {}, "post_alias_rejections": {}}
    for dataset in ("fiqa", "nfcorpus", "scifact"):
        promotions = []
        guards = []
        rejects = []
        for qid, q3_row in sorted(q3_rows[dataset].items()):
            if not allowed_qid(dataset, qid, sources[dataset], excluded):
                continue
            gains = sources[dataset]["qrels"][qid]
            feats = front_features(q3_row, gains)
            selected_q3 = [c for c in q3_row["top_k"] if (gains.get(str(c["doc_id"]), 0) > 0 and int(c["rank"]) <= 20) or (gains.get(str(c["doc_id"]), 0) <= 0 and int(c["rank"]) <= 10)]
            try:
                selected, alias = canonical_select(dataset, qid, selected_q3, gains, sources[dataset]["corpus"], lambda item: 100 * int(item["raw_gain"]) + (20 if int(item["rank"]) <= 10 else 10))
            except ValueError:
                selected = []
                alias = {}
            if feats["eligible"]:
                if selected:
                    promotions.append((float(feats["promotion_score_gap"]), qid, q3_row, selected, alias))
                else:
                    rejects.append(qid)
            elif feats["guard"] and selected:
                guards.append((float(feats["guard_margin"]), qid, q3_row, selected, alias))
        promotions.sort(key=lambda item: (item[0], item[1]))
        if dataset == "fiqa":
            chosen_promotions = promotions
        elif dataset == "nfcorpus":
            chosen_promotions = promotions[:64]
            if len(promotions) != 235:
                raise ValueError(f"nfcorpus post-alias promotion inventory={len(promotions)}, expected 235")
        else:
            chosen_promotions = promotions
        chosen_qids = {qid for _gap, qid, _q3, _selected, _alias in chosen_promotions}
        chosen_guards = []
        if dataset in {"fiqa", "scifact"}:
            chosen_guards = [item for item in sorted(guards, key=lambda item: (item[0], item[1])) if item[1] not in chosen_qids][:12]
        for gap, qid, q3_row, selected, alias in chosen_promotions + chosen_guards:
            rows.append(build_row(dataset=dataset, qid=qid, query=sources[dataset]["queries"][qid], selected=selected, bucket=FRONTIER_BUCKET, source_path=q3_source[dataset], main_weight=1.0, recall_weight=0.0, provenance={"selector": FRONTIER_BUCKET, "q3_query": q3_row, "gap_or_margin": gap, "alias": alias}))
        audit["promotion_inventory"][dataset] = {"available": len(promotions), "selected": len(chosen_promotions), "selected_qids": [qid for _gap, qid, _q3, _selected, _alias in chosen_promotions]}
        audit["guard_inventory"][dataset] = {"available": len(guards), "selected": len(chosen_guards), "selected_qids": [qid for _gap, qid, _q3, _selected, _alias in chosen_guards]}
        audit["post_alias_rejections"][dataset] = rejects
    return rows, audit


def select_top10_rows(q3_rows: dict[str, dict[str, dict[str, Any]]], sources: dict[str, dict[str, Any]], excluded: dict[str, set[str]], q3_source: dict[str, Path], frontier_keys: set[tuple[str, str]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    audit = {"selected_qids": defaultdict(list), "rejected_post_alias": defaultdict(list)}
    for dataset in ("fiqa", "nfcorpus", "scifact"):
        for qid, q3_row in sorted(q3_rows[dataset].items()):
            if (dataset, qid) in frontier_keys or not allowed_qid(dataset, qid, sources[dataset], excluded):
                continue
            gains = sources[dataset]["qrels"][qid]
            top10 = [c for c in q3_row["top_k"] if int(c["rank"]) <= 10]
            pos = [c for c in top10 if gains.get(str(c["doc_id"]), 0) > 0]
            neg = [c for c in top10 if gains.get(str(c["doc_id"]), 0) <= 0]
            if not pos or not neg:
                continue
            margin = min(float(p["score"]) - float(n["score"]) for p in pos for n in neg)
            if margin > 0.005:
                continue
            try:
                selected, alias = canonical_select(dataset, qid, top10, gains, sources[dataset]["corpus"], lambda item: 100 * int(item["raw_gain"]) + 20)
            except ValueError:
                audit["rejected_post_alias"][dataset].append(qid)
                continue
            rows.append(build_row(dataset=dataset, qid=qid, query=sources[dataset]["queries"][qid], selected=selected, bucket=TOP10_BUCKET, source_path=q3_source[dataset], main_weight=1.0, recall_weight=0.0, provenance={"selector": TOP10_BUCKET, "q3_query": q3_row, "margin": margin, "alias": alias}))
            audit["selected_qids"][dataset].append(qid)
    return rows, {"selected_qids": {k: v for k, v in audit["selected_qids"].items()}, "rejected_post_alias": {k: v for k, v in audit["rejected_post_alias"].items()}}


def select_nf_boundary_rows(q3_rows: dict[str, dict[str, dict[str, Any]]], sources: dict[str, dict[str, Any]], excluded: dict[str, set[str]], q3_source: dict[str, Path]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    dataset = "nfcorpus"
    rows: list[dict[str, Any]] = []
    selected_docs: list[dict[str, Any]] = []
    for qid, q3_row in sorted(q3_rows[dataset].items()):
        if not allowed_qid(dataset, qid, sources[dataset], excluded):
            continue
        gains = sources[dataset]["qrels"][qid]
        for pos in q3_row["top_k"]:
            pos_rank = int(pos["rank"])
            pos_doc = str(pos["doc_id"])
            raw_gain = gains.get(pos_doc, 0)
            if not (80 <= pos_rank <= 100 and raw_gain > 0):
                continue
            negs = [c for c in q3_row["top_k"] if gains.get(str(c["doc_id"]), 0) <= 0]
            negs.sort(key=lambda c: (0 if 70 <= int(c["rank"]) <= 100 else 1, abs(float(c["score"]) - float(pos["score"])), abs(int(c["rank"]) - pos_rank), int(c["rank"]), str(c["doc_id"])))
            if pos_doc not in sources[dataset]["corpus"]:
                raise ValueError(f"{dataset}:{qid}: missing corpus text for {pos_doc}")
            selected = [{
                "candidate": pos,
                "doc_id": pos_doc,
                "rank": pos_rank,
                "score": float(pos["score"]),
                "raw_gain": raw_gain,
                "text": sources[dataset]["corpus"][pos_doc],
                "gain": 100 * int(raw_gain) + 5,
                "source": "qrel",
                "positive_alias_count": 1,
            }]
            alias = {"positive_alias_drops": 0, "nonrel_alias_drops": 0}
            positive_aliases = {r4.normalize_candidate_text(item["text"]) for item in selected if item["gain"] > 0}
            seen_negative_aliases: set[str] = set()
            negative_alias_drops = 0
            for neg in negs:
                doc_id = str(neg["doc_id"])
                text = sources[dataset]["corpus"][doc_id]
                alias_key = r4.normalize_candidate_text(text)
                if alias_key in positive_aliases:
                    negative_alias_drops += 1
                    continue
                if alias_key in seen_negative_aliases:
                    negative_alias_drops += 1
                    continue
                seen_negative_aliases.add(alias_key)
                selected.append({
                    "candidate": neg,
                    "doc_id": doc_id,
                    "rank": int(neg["rank"]),
                    "score": float(neg["score"]),
                    "raw_gain": 0,
                    "text": text,
                    "gain": 0,
                    "source": "q3",
                    "positive_alias_count": 0,
                })
                if len(selected) == 9:
                    break
            if len(selected) != 9:
                raise ValueError(f"{dataset}:{qid}:{pos_doc}: cannot retain one positive plus eight canonical-unique negatives")
            alias["boundary_negative_alias_drops"] = negative_alias_drops
            trimmed = selected
            row = build_row(dataset=dataset, qid=qid, query=sources[dataset]["queries"][qid], selected=trimmed, bucket=NF_BOUNDARY_BUCKET, source_path=q3_source[dataset], main_weight=0.0, recall_weight=1.0, provenance={"selector": NF_BOUNDARY_BUCKET, "q3_query": q3_row, "selected_positive_doc_id": pos_doc, "selected_positive_rank": pos_rank, "alias": alias})
            row["row_id"] = f"{dataset}:{qid}:r6-{NF_BOUNDARY_BUCKET}:{pos_doc}"
            validate_row(row)
            if len(row["candidate_doc_ids"]) != 9 or canonical_stats(row, row["row_id"])["canonical_candidates"] != 9 or row["positive_count"] != 1 or row["q3_negative_count"] != 8 or eligible_topk_pairs(row, True, "q3") != 8:
                raise ValueError(f"{row['row_id']}: boundary row invariant drift")
            rows.append(row)
            selected_docs.append({"query_id": qid, "doc_id": pos_doc, "rank": pos_rank, "gain": raw_gain})
    return rows, {"selected_positive_docs": selected_docs, "unique_qids": len({item["query_id"] for item in selected_docs})}


def validate_rows(rows: list[dict[str, Any]], args: argparse.Namespace, excluded: dict[str, set[str]], sources: dict[str, dict[str, Any]]) -> dict[str, Any]:
    if len(rows) != args.expected_rows or len({row["row_id"] for row in rows}) != len(rows):
        raise ValueError("row count or row_id uniqueness drift")
    bucket_counts = Counter(row["selection_bucket"] for row in rows)
    if dict(bucket_counts) != EXPECTED_BUCKETS:
        raise ValueError(f"bucket drift: {dict(bucket_counts)} != {EXPECTED_BUCKETS}")
    domain_bucket_counts: dict[str, dict[str, int]] = {}
    for bucket in EXPECTED_BUCKETS:
        counts = Counter(row["source_dataset"] for row in rows if row["selection_bucket"] == bucket)
        domain_bucket_counts[bucket] = dict(counts)
        if dict(counts) != EXPECTED_DOMAIN_BUCKETS[bucket]:
            raise ValueError(f"{bucket} domain drift: {dict(counts)}")
    raw = canonical = main = recall = alias = 0
    inactive = []
    base_bad = []
    overlap: dict[str, Any] = {}
    selected_qids: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        validate_row(row)
        raw += len(row["candidate_doc_ids"])
        stats = canonical_stats(row, row["row_id"])
        canonical += stats["canonical_candidates"]
        main += eligible_topk_pairs(row, False, "q3")
        recall += eligible_topk_pairs(row, True, "q3")
        if eligible_topk_pairs(row, False, "q3") <= 0 and eligible_topk_pairs(row, True, "q3") <= 0:
            inactive.append(row["row_id"])
        if any(float(row.get(name, 0.0)) != 0.0 for name in ("base_loss_weight", "hard_loss_weight", "soft_loss_weight", "recovery_loss_weight")):
            base_bad.append(row["row_id"])
        selected_qids[row["source_dataset"]].add(str(row["source_query_id"]))
    alias = raw - canonical
    workload = {
        "train": len(rows),
        "batch": args.batch_size,
        "steps_per_epoch": math.ceil(len(rows) / args.batch_size),
        "epochs": args.epochs,
        "raw_candidates_per_epoch": raw,
        "canonical_candidates_per_epoch": canonical,
        "alias_drops_per_epoch": alias,
        "main_q3_pairs_per_epoch": main,
        "recall_q3_pairs_per_epoch": recall,
        "total_train_pairs_per_epoch": canonical + main + recall,
        "planned_work_units": args.epochs * (canonical + main + recall),
        "actual_train_pairs": 0,
    }
    if workload != EXPECTED_WORKLOAD:
        raise ValueError(f"workload drift: {workload} != {EXPECTED_WORKLOAD}")
    if inactive or base_bad:
        raise ValueError(f"invariant drift inactive={inactive[:3]} base_bad={base_bad[:3]}")
    for dataset, qids in sorted(selected_qids.items()):
        frozen = sorted(qids & excluded.get(dataset, set()))
        test = sorted(qids & sources[dataset]["test_qids"])
        if frozen or test:
            raise ValueError(f"{dataset}: heldout overlap frozen={frozen} test={test}")
        overlap[dataset] = {"selected_qids": len(qids), "selected_x_frozen_dev_reserve": 0, "selected_x_official_test": 0}
    return {
        "rows": len(rows),
        "bucket_counts": dict(bucket_counts),
        "domain_bucket_counts": domain_bucket_counts,
        "candidate_count_counts": dict(Counter(str(len(row["candidate_doc_ids"])) for row in rows)),
        "workload": workload,
        "overlap": overlap,
        "effective_weight_audit": {
            "main_row_scale": 0.04 / 1.04,
            "retention_row_scale": 0.02 / 1.02,
            "aggregate_main_to_recall_ratio": (515 * (0.04 / 1.04)) / (640 * (0.02 / 1.02)),
            "aggregate_ratio_lte_2": True,
            "base_hard_soft_recovery_expected_zero_rows": len(rows),
            "runtime_base0_support": "pending new binary verification",
        },
        "alias_audit": {"raw_candidates": raw, "canonical_candidates": canonical, "alias_drops": alias},
    }


def probe_surface_id(bucket: str, dataset: str) -> str:
    prefix = {
        FRONTIER_BUCKET: "frontier",
        TOP10_BUCKET: "top10-retention",
        NF_BOUNDARY_BUCKET: "nf-boundary80-100",
    }[bucket]
    return f"{prefix}.{dataset}"


def probe_result_paths(args: argparse.Namespace, surface_id: str, label: str) -> dict[str, str]:
    stem = args.output_root / "probe" / "future-outputs" / surface_id / f"{label}.{surface_id}.q3.r6-probe"
    return {
        "metrics_json": display_path(Path(f"{stem}.json")),
        "metrics_tsv": display_path(Path(f"{stem}.tsv")),
        "per_query_jsonl": display_path(Path(f"{stem}.per-query.jsonl")),
    }


def eval_probe_argv(binary: Path, package: Path, dataset: str, qrels: str, outputs: dict[str, str]) -> list[str]:
    return [
        display_path(binary),
        "eval-retrieval-turboquant",
        "--bits",
        "3",
        "--quantizer-seed",
        str(r4.Q3_SEED),
        "--top-k",
        "100",
        "--per-query-top-k",
        "100",
        "--dataset",
        dataset,
        "--qrels",
        qrels,
        "--metrics-json",
        outputs["metrics_json"],
        "--metrics-tsv",
        outputs["metrics_tsv"],
        "--per-query-jsonl",
        outputs["per_query_jsonl"],
        display_path(package),
        display_path(DATASET_DIRS[dataset]),
    ]


def validate_probe_surface_qrels(surface_id: str, bucket: str, dataset: str, rows: list[dict[str, Any]], sources: dict[str, dict[str, Any]]) -> list[tuple[str, str, int]]:
    qrels = []
    source_qrels = sources[dataset]["qrels"]
    source_corpus = sources[dataset]["corpus"]
    source_queries = sources[dataset]["queries"]
    for row in rows:
        if row["selection_bucket"] != bucket or row["source_dataset"] != dataset:
            continue
        qid = str(row["source_query_id"])
        if qid not in source_queries:
            raise ValueError(f"{surface_id}: qid {qid} missing from {dataset} queries")
        for index in row["positive_indexes"]:
            doc_id = raw_doc_id(dataset, str(row["candidate_doc_ids"][index]))
            if doc_id not in source_corpus:
                raise ValueError(f"{surface_id}: doc {doc_id} missing from {dataset} corpus")
            if doc_id not in source_qrels.get(qid, {}):
                raise ValueError(f"{surface_id}: qrel {qid}/{doc_id} missing from {dataset} train qrels")
            qrels.append((qid, doc_id, int(row["qrel_gains"][index])))
    if not qrels:
        raise ValueError(f"{surface_id}: empty qrels")
    pairs = [(qid, doc_id) for qid, doc_id, _gain in qrels]
    if len(pairs) != len(set(pairs)):
        raise ValueError(f"{surface_id}: duplicate qid/doc qrel")
    return sorted(qrels)


def build_probe_manifest(rows: list[dict[str, Any]], args: argparse.Namespace, q3_identities: list[dict[str, Any]], sources: dict[str, dict[str, Any]], excluded: dict[str, set[str]], validation: dict[str, Any]) -> dict[str, Any]:
    probe_root = args.output_root / "probe"
    surfaces: dict[str, Any] = {}
    commands: list[dict[str, Any]] = []
    expected_outputs: list[str] = []
    seen_qids_by_dataset: dict[str, set[str]] = defaultdict(set)
    for bucket, dataset in PROBE_SURFACES:
        surface_id = probe_surface_id(bucket, dataset)
        surface_rows = [row for row in rows if row["selection_bucket"] == bucket and row["source_dataset"] == dataset]
        qrels = validate_probe_surface_qrels(surface_id, bucket, dataset, rows, sources)
        for qid, _doc_id, _gain in qrels:
            seen_qids_by_dataset[dataset].add(qid)
        qrels_path = probe_root / "qrels" / dataset / f"{surface_id}.qrels.tsv"
        write_qrels(qrels_path, qrels)
        qrels_stats = qrels_file_stats(qrels_path)
        anchor_outputs = probe_result_paths(args, surface_id, "anchor")
        candidate_outputs = probe_result_paths(args, surface_id, "candidate")
        for label, package, outputs in (("anchor", ANCHOR_PACKAGE, anchor_outputs), ("candidate", CANDIDATE_PACKAGE, candidate_outputs)):
            argv = eval_probe_argv(R6_BINARY, package, dataset, qrels_stats["path"], outputs)
            commands.append({
                "surface_id": surface_id,
                "bucket": bucket,
                "dataset": dataset,
                "label": label,
                "package": display_path(package),
                "argv": argv,
                "outputs": outputs,
                "expected_exit_status": 0,
            })
            expected_outputs.extend(outputs.values())
        surfaces[surface_id] = {
            "surface_id": surface_id,
            "bucket": bucket,
            "dataset": dataset,
            "dataset_dir": display_path(DATASET_DIRS[dataset]),
            "dataset_source_hashes": {name: {"path": display_path(path), "sha256": sha256_file(path)} for name, path in sources[dataset]["paths"].items()},
            "qrels": qrels_stats,
            "row_count": len(surface_rows),
            "qid_count": len({str(row["source_query_id"]) for row in surface_rows}),
            "qrel_count": len(qrels),
            "row_ids_sha256": sha256_json([row["row_id"] for row in surface_rows]),
            "qids_sha256": sha256_json(sorted({str(row["source_query_id"]) for row in surface_rows})),
        }
    outputs_root = args.output_root / "probe" / "future-outputs"
    for output in expected_outputs:
        require_resolved_within(Path(output), outputs_root, f"future output escapes R6 probe output root: {output}")
        output_abs = repo_path(Path(output)).resolve()
        if output_abs.exists():
            raise ValueError(f"future output path already exists: {output}")
    command_hashes = [sha256_json(command["argv"]) for command in commands]
    if len(surfaces) != 7 or len(commands) != 14 or len(expected_outputs) != 42 or len(set(expected_outputs)) != 42 or len(set(command_hashes)) != 14:
        raise ValueError("probe command/output cardinality drift")
    return {
        "schema": PROBE_SCHEMA,
        "created_utc": utc_stamp(),
        "arm_id": ARM_ID,
        "source_policy": "precommitted train-side movement probes derived only from selected original-anchor q3 rows before R6 training",
        "status": "preflight_manifest_only_no_probe_eval_run",
        "binary": {"path": display_path(R6_BINARY), "sha256": R6_BINARY_SHA256, "verification": "future integration binding"},
        "anchor_package": {"path": display_path(ANCHOR_PACKAGE), "sibling_rollup_sha256": "future integration binding"},
        "candidate_package": {"path": display_path(CANDIDATE_PACKAGE), "arm_id": ARM_ID, "sibling_rollup_sha256": "future post-train integration binding"},
        "q3_inputs": q3_identities,
        "q3_settings": {
            "method": r4.Q3_METHOD,
            "bits": r4.Q3_BITS,
            "dim": r4.Q3_DIM,
            "scoring_surface": r4.Q3_SURFACE,
            "quantizer_seed": r4.Q3_SEED,
            "top_k": 100,
            "per_query_top_k": 100,
            "score_mode": "prepared_ip",
        },
        "train_contract": validation["workload"],
        "raw_source_hashes": {
            dataset: {name: {"path": display_path(path), "sha256": sha256_file(path)} for name, path in sources[dataset]["paths"].items()}
            for dataset in ("fiqa", "nfcorpus", "scifact")
        },
        "exclusions": {display_path(path): {"sha256": sha256_file(path)} for path in args.exclude_qids_json},
        "surfaces": surfaces,
        "surface_count": len(surfaces),
        "command_count": len(commands),
        "expected_output_count": len(expected_outputs),
        "commands": commands,
        "expected_outputs": sorted(expected_outputs),
        "all_expected_outputs_absent": True,
        "all_expected_outputs_contained": True,
        "command_argv_sha256": sha256_json([command["argv"] for command in commands]),
        "future_paths": {
            "anchor_package": display_path(ANCHOR_PACKAGE),
            "candidate_package": display_path(CANDIDATE_PACKAGE),
            "outputs_root": display_path(args.output_root / "probe" / "future-outputs"),
        },
        "gates": {
            "zero_relevant_top10_exits_on_selected_frontier_and_top10_retention_surfaces": {
                "math": "For each qrel-positive doc in anchor top10 on frontier/top10 surfaces, candidate rank must be <=10; missing candidate doc counts as an exit.",
                "tie_policy": "Ranks are authoritative from per-query top_k; doc_id exact match.",
            },
            "nf_selected_boundary80_100_exits_lte_entries": {
                "math": "For each selected NF boundary qid/doc with anchor rank in [80,100], candidate rank >100 or missing counts as exit; candidate rank <=100 counts as retained/entry.",
            },
            "median_selected_boundary_positive_vs_nearest_negative_margin_nonnegative": {
                "math": "For each NF boundary selected positive, candidate score minus nearest candidate top100 nonrelevant score by absolute score distance must have median >= 0.",
                "tie_policy": "Nearest negative ties sort by absolute rank distance, rank, doc_id.",
            },
            "metric_aligned_substitution_failures_zero": {
                "math": "Substitution failure means a selected qrel-positive exits top10/100 while a nonrelevant doc enters that same gate band for the same qid.",
            },
            "frontier_and_top10_q3_ndcg_delta_gte_zero_each_domain_and_macro": {
                "math": "candidate metrics JSON q3 nDCG@10 - anchor metrics JSON q3 nDCG@10 >= 0 for each frontier/top10 domain surface and for the six-surface macro.",
            },
            "nf_boundary_recall100_delta_gte_zero": {
                "math": "candidate NF boundary recall@100 - anchor NF boundary recall@100 >= 0.",
            },
            "candidate_effective_base_leakage_audit_aux_only": {
                "math": "Candidate training metrics/evidence must show base/hard/soft/recovery work units are zero and only R6 aux topk/recall objectives are active.",
                "status": "future integration binding",
            },
        },
        "legal_gates": dict(LEGAL_GATES),
        "quality_claim": False,
    }


def default_paths(args: argparse.Namespace) -> None:
    if not args.q3_per_query_jsonl:
        args.q3_per_query_jsonl = list(DEFAULT_Q3)
    if not args.exclude_qids_json:
        args.exclude_qids_json = list(DEFAULT_EXCLUDES)
    data_root = args.output_root / "data"
    if args.output_jsonl is None:
        args.output_jsonl = data_root / "beir1155.q3r6-frontier-retention.train.jsonl"
    if args.manifest is None:
        args.manifest = data_root / "beir1155.q3r6-frontier-retention.manifest.json"
    if args.coverage_report is None:
        args.coverage_report = args.output_root / "q3r6-frontier-retention.coverage.json"
    if args.probe_manifest is None:
        args.probe_manifest = args.output_root / "probe/q3r6-frontier-retention.probe-manifest.json"


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    default_paths(args)
    q3_rows, q3_identities = r4.load_q3_rows(args.q3_per_query_jsonl, expected_q3_hashes(args))
    q3_source = {next(iter(item["datasets"])): Path(item["path"]) for item in q3_identities}
    sources = {dataset: r4.load_domain_sources(repo_path(args.dataset_root), dataset) for dataset in ("fiqa", "nfcorpus", "scifact")}
    excluded = r4.read_excluded_qids(args.exclude_qids_json)
    frontier_rows, frontier_audit = select_frontier_rows(q3_rows, sources, excluded, q3_source)
    frontier_keys = {(row["source_dataset"], str(row["source_query_id"])) for row in frontier_rows}
    top10_rows, top10_audit = select_top10_rows(q3_rows, sources, excluded, q3_source, frontier_keys)
    nf_rows, nf_audit = select_nf_boundary_rows(q3_rows, sources, excluded, q3_source)
    rows = frontier_rows + top10_rows + nf_rows
    rows.sort(key=lambda row: (row["selection_bucket"], row["source_dataset"], str(row["source_query_id"]), row["row_id"]))
    validation = validate_rows(rows, args, excluded, sources)
    write_jsonl(args.output_jsonl, rows)
    probe_manifest = build_probe_manifest(rows, args, q3_identities, sources, excluded, validation)
    write_json(args.probe_manifest, probe_manifest)
    coverage = {
        "schema": f"{SCHEMA}.coverage.v1",
        "created_utc": utc_stamp(),
        "arm_id": ARM_ID,
        "selection_audit": {"frontier": frontier_audit, "top10": top10_audit, "nf_boundary": nf_audit},
        "validation": validation,
        "selected_rows": [{"row_id": row["row_id"], "dataset": row["source_dataset"], "query_id": row["source_query_id"], "bucket": row["selection_bucket"], "candidate_count": len(row["candidate_doc_ids"]), "main_pairs": eligible_topk_pairs(row, False, "q3"), "recall_pairs": eligible_topk_pairs(row, True, "q3")} for row in rows],
        "probe_manifest": {"path": display_path(args.probe_manifest), "sha256": sha256_file(args.probe_manifest)},
        "legal_gates": dict(LEGAL_GATES),
        "quality_claim": False,
    }
    write_json(args.coverage_report, coverage)
    train_sha = sha256_file(args.output_jsonl)
    manifest = {
        "schema": SCHEMA,
        "created_utc": utc_stamp(),
        "arm_id": ARM_ID,
        "builder": "scripts/build_q3_r6_frontier_retention_dataset.py",
        "inputs": {
            "dataset_root": display_path(args.dataset_root),
            "q3_per_query_jsonl": q3_identities,
            "exclude_qids_json": {display_path(path): {"sha256": sha256_file(path)} for path in args.exclude_qids_json},
            "raw_sources": coverage["selected_rows"] and {
                dataset: {name: {"path": display_path(path), "sha256": sha256_file(path)} for name, path in sources[dataset]["paths"].items()}
                for dataset in ("fiqa", "nfcorpus", "scifact")
            },
        },
        "outputs": {
            "train_score_spectrum": display_path(args.output_jsonl),
            "manifest": display_path(args.manifest),
            "coverage_report": display_path(args.coverage_report),
            "probe_manifest": display_path(args.probe_manifest),
        },
        "policy": {
            "frontier_context_q3": "R4 broad selector universe with post-alias safety: all FiQA/SciFact eligible promotions, first64 NFCorpus eligible promotions by gap, plus first12 FiQA/SciFact vulnerable top10 guards.",
            "top10_retention_q3": "All remaining nonheldout train qids with post-alias q3 top10 positive and nonrel and min positive-vs-nonrel margin <= 0.005.",
            "nf_boundary80_100_docretention_q3": "One recall-only row for every nonheldout NFCorpus train-positive doc at original-anchor q3 rank 80..100.",
            "candidate_canonicalization": "Lowercase whitespace normalization; positive aliases collapse to max qrel gain then best rank/doc; nonrel aliases to retained positives are dropped. Runtime tokenizer may split punctuation/case differently; this builder proves row-level text alias safety under the explicit training-data canonicalizer used by existing score-spectrum workload tests.",
            "legal": "research training only; no release, commercial, free/open release, or quality claim.",
        },
        "counts": validation,
        "selection_audit": coverage["selection_audit"],
        "probe_manifest": {"path": display_path(args.probe_manifest), "sha256": sha256_file(args.probe_manifest)},
        "legal_gates": dict(LEGAL_GATES),
        **LEGAL_GATES,
        "quality_claim": False,
        "sha256": {
            "builder": sha256_file(Path("scripts/build_q3_r6_frontier_retention_dataset.py")),
            "train_score_spectrum": train_sha,
            "coverage_report": sha256_file(args.coverage_report),
            "probe_manifest": sha256_file(args.probe_manifest),
        },
    }
    write_json(args.manifest, manifest)
    print(f"q3 R6 frontier-retention dataset rows={len(rows)} train_sha256={train_sha}")
    print(f"buckets={dict(Counter(row['selection_bucket'] for row in rows))}")
    print(f"workload={validation['workload']}")


if __name__ == "__main__":
    main()
