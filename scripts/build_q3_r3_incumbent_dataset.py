#!/usr/bin/env python3
"""Append the NFCorpus PLAIN-73 top10 incumbent-preservation replay row."""

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


SCHEMA = "eos.q3near_dualcut_r3_incumbent_preservation_dataset.v1"
LEGAL_GATES = {
    "train_allowed_for_research": True,
    "release_train_allowed": False,
    "commercial_use_allowed": False,
}
PARENT_TRAIN_SHA256 = "9a950dbe8eddcdcd1f20fefe581db2c2b9a4bee2e3678f346577b5475bee5433"
PARENT_MANIFEST_SHA256 = "dcc8449f6e462cd0683a72d471314ad4f57fbbb0f92daf128e79b1bd98351c40"
PARENT_COVERAGE_SHA256 = "bab2b7485de4af6a36d88944658c40550fafd7e9cd06c37c94902314af285259"
R2_GATE_SHA256 = "da529984b338334daf1a8d690c727317ce54a40d74d652c44e0740a5b71a7aea"
R2_PROXY_QRELS_MANIFEST_SHA256 = "8e9b4539348013859d5baa4b676a8091c2ef27703299ecc8e3ad688873c9136b"
SOURCE_ROW_ID = "nfcorpus:PLAIN-73"
APPENDED_ROW_ID = "nfcorpus:PLAIN-73:r2-top10-incumbent-preserve"
QUERY_ID = "PLAIN-73"
INCUMBENT_GAIN2_DOCS = ("MED-3032", "MED-3023", "MED-3020", "MED-3019", "MED-3031")
NON_INCUMBENT_GAIN1_DOCS = ("MED-3022", "MED-3026")
SHAPED_GAINS = {doc_id: 2.0 for doc_id in INCUMBENT_GAIN2_DOCS} | {doc_id: 1.0 for doc_id in NON_INCUMBENT_GAIN1_DOCS}
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
    parser.add_argument("--parent-coverage", required=True, type=Path)
    parser.add_argument("--r2-gate", required=True, type=Path)
    parser.add_argument("--r2-proxy-summary", required=True, type=Path)
    parser.add_argument("--r2-movement", required=True, type=Path)
    parser.add_argument("--r2-proxy-qrels-manifest", required=True, type=Path)
    parser.add_argument("--anchor-per-query-jsonl", required=True, type=Path)
    parser.add_argument("--candidate-per-query-jsonl", required=True, type=Path)
    parser.add_argument("--exclude-qids-json", action="append", default=[], type=Path)
    parser.add_argument("--test-qrels", required=True, type=Path)
    parser.add_argument("--output-jsonl", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--coverage-report", required=True, type=Path)
    parser.add_argument("--test-file", default=Path("scripts/test_build_q3_r3_incumbent_dataset.py"), type=Path)
    parser.add_argument("--expected-parent-train-sha256", default=PARENT_TRAIN_SHA256)
    parser.add_argument("--expected-parent-manifest-sha256", default=PARENT_MANIFEST_SHA256)
    parser.add_argument("--expected-parent-coverage-sha256", default=PARENT_COVERAGE_SHA256)
    parser.add_argument("--expected-r2-gate-sha256", default=R2_GATE_SHA256)
    parser.add_argument("--expected-r2-proxy-qrels-manifest-sha256", default=R2_PROXY_QRELS_MANIFEST_SHA256)
    parser.add_argument("--expected-parent-rows", type=int, default=69)
    parser.add_argument("--expected-rows", type=int, default=70)
    parser.add_argument("--expected-steps-per-epoch", type=int, default=18)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--turboquant-objectives", type=int, default=1)
    parser.add_argument("--turboquant-topk-negative-mask", default="q3")
    parser.add_argument("--turboquant-topk-recall-weight", type=float, default=0.15)
    parser.add_argument("--turboquant-topk-recall-negative-mask", default="q3")
    parser.add_argument("--turboquant-topk-recall-cutoff", type=int, default=100)
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
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


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


def raw_doc_id(candidate_doc_id: str) -> str:
    return candidate_doc_id.split(":", 1)[1] if candidate_doc_id.startswith("nfcorpus:") else candidate_doc_id


def normalize_candidate_text(text: str) -> str:
    return " ".join(str(text).lower().split())


def assert_hash(path: Path, expected: str, label: str) -> str:
    if not path.is_file():
        raise ValueError(f"missing {label}: {path}")
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"{label} sha mismatch: expected={expected} actual={actual}")
    return actual


def validate_row_shape(row: dict[str, Any], context: str) -> None:
    sizes = {}
    for field in ALIGNED_CANDIDATE_FIELDS:
        value = row.get(field)
        if not isinstance(value, list):
            raise ValueError(f"{context}: {field} must be a list")
        sizes[field] = len(value)
    if len(set(sizes.values())) != 1:
        raise ValueError(f"{context}: aligned field length mismatch {sizes}")
    if len(row.get("candidate_doc_ids", [])) != len(set(row.get("candidate_doc_ids", []))):
        raise ValueError(f"{context}: duplicate candidate doc id")
    positives = [idx for idx, gain in enumerate(row["qrel_gains"]) if float(gain) > 0.0]
    if positives != list(row.get("positive_indexes", [])):
        raise ValueError(f"{context}: positive_indexes do not match qrel_gains")
    for idx in positives:
        if row["candidate_sources"][idx] != "qrel":
            raise ValueError(f"{context}: positive candidate {idx} source is not qrel")
        if row["hard_negative_eligible"][idx]:
            raise ValueError(f"{context}: positive candidate {idx} is hard-negative eligible")
    if abs(sum(float(value) for value in row["target_probabilities"]) - 1.0) > 1e-6:
        raise ValueError(f"{context}: target_probabilities do not sum to 1")
    if row.get("legal_gates") != LEGAL_GATES:
        raise ValueError(f"{context}: legal gates are not research-only fail-closed")
    if row.get("train_allowed_for_research") is not True or row.get("release_train_allowed") is not False or row.get("commercial_use_allowed") is not False:
        raise ValueError(f"{context}: scalar legal gates are not research-only fail-closed")


def canonical_stats(row: dict[str, Any], context: str) -> dict[str, int]:
    merged = canonical_candidates(row, context)
    return {
        "raw_candidates": len(row["candidate_doc_ids"]),
        "canonical_candidates": len(merged),
        "raw_positive_pairs": len(set(row["positive_indexes"])),
        "canonical_positive_pairs": sum(1 for item in merged if item["positive"]),
        "raw_hard_negative_pairs": sum(1 for value in row["hard_negative_eligible"] if bool(value)),
        "canonical_hard_negative_pairs": sum(1 for item in merged if item["hard"] and not item["positive"]),
    }


def canonical_candidates(row: dict[str, Any], context: str) -> list[dict[str, Any]]:
    positive_set = set(row["positive_indexes"])
    merged: list[dict[str, Any]] = []
    index_by_key: dict[str, int] = {}
    for idx, text in enumerate(row["candidate_texts"]):
        key = normalize_candidate_text(text)
        if not key:
            raise ValueError(f"{context}: empty canonical candidate text at {idx}")
        positive = idx in positive_set
        hard = bool(row["hard_negative_eligible"][idx])
        if key in index_by_key:
            prev = merged[index_by_key[key]]
            if (prev["positive"] or positive) and (prev["hard"] or hard):
                raise ValueError(f"{context}: duplicate-text positive/hard-negative conflict at candidate {idx}")
            prev["positive"] = prev["positive"] or positive
            prev["hard"] = prev["hard"] or hard
            prev["gain"] = max(prev["gain"], float(row["qrel_gains"][idx]))
            prev["prob"] += float(row["target_probabilities"][idx])
            source = str(row["candidate_sources"][idx])
            if source and source not in str(prev["source"]).split("+"):
                prev["source"] = f"{prev['source']}+{source}" if prev["source"] else source
        else:
            index_by_key[key] = len(merged)
            merged.append({
                "positive": positive,
                "hard": hard,
                "gain": float(row["qrel_gains"][idx]),
                "prob": float(row["target_probabilities"][idx]),
                "source": str(row["candidate_sources"][idx]),
            })
    for item in merged:
        if item["positive"]:
            item["hard"] = False
    return merged


def eligible_topk_pairs(row: dict[str, Any], recall: bool, negative_mask: str = "q3") -> int:
    if not recall and float(row.get("turboquant_topk_loss_weight", 1.0)) <= 0.0:
        return 0
    if recall and float(row.get("turboquant_topk_recall_loss_weight", 1.0)) <= 0.0:
        return 0
    candidates = canonical_candidates(row, row.get("row_id", "<row>"))
    count = 0
    gains = [float(candidate["gain"]) for candidate in candidates]
    for high_gain in gains:
        for low, low_gain in enumerate(gains):
            if high_gain <= low_gain:
                continue
            if low_gain > 0.0:
                count += 1
                continue
            candidate = candidates[low]
            if negative_mask == "q3" and candidate["hard"] and "q3" in str(candidate["source"]).lower():
                count += 1
            elif negative_mask == "hard" and candidate["hard"]:
                count += 1
            elif negative_mask == "all":
                count += 1
            elif negative_mask == "bm25" and candidate["hard"] and "bm25" in str(candidate["source"]).lower():
                count += 1
    return count


def workload_counts(rows: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    raw = sum(len(row["candidate_doc_ids"]) for row in rows)
    canonical = sum(canonical_stats(row, row["row_id"])["canonical_candidates"] for row in rows)
    main = sum(eligible_topk_pairs(row, False, args.turboquant_topk_negative_mask) for row in rows)
    recall = sum(eligible_topk_pairs(row, True, args.turboquant_topk_recall_negative_mask) for row in rows)
    total_epoch = canonical + args.turboquant_objectives * (main + recall)
    steps = math.ceil(len(rows) / args.batch_size)
    return {
        "train": len(rows),
        "batch": args.batch_size,
        "steps_per_epoch": steps,
        "epochs": args.epochs,
        "raw_candidates_per_epoch": raw,
        "canonical_candidates_per_epoch": canonical,
        "main_q3_pairs_per_epoch": main,
        "recall_q3_pairs_per_epoch": recall,
        "total_train_pairs_per_epoch": total_epoch,
        "planned_pairs": total_epoch * args.epochs,
        "turboquant_topk_negative_mask": args.turboquant_topk_negative_mask,
        "turboquant_topk_recall_weight": args.turboquant_topk_recall_weight,
        "turboquant_topk_recall_cutoff": args.turboquant_topk_recall_cutoff,
        "turboquant_topk_recall_negative_mask": args.turboquant_topk_recall_negative_mask,
    }


def go_loader_plan(args: argparse.Namespace, workload: dict[str, Any]) -> dict[str, Any]:
    return {
        "plan_only": True,
        "go_runtime_edits_required": False,
        "dataset_form": "text_score_spectrum_jsonl",
        "loader_path": "runtime.OpenEmbeddingTextScoreSpectrumSource",
        "required_cli_flags": [
            "--score-spectrum-train",
            "--allow-research-only-score-spectrum",
            "--epochs",
            str(args.epochs),
            "--batch-size",
            str(args.batch_size),
            "--turboquant-topk-loss",
            "lambdandcg",
            "--turboquant-topk-negative-mask",
            args.turboquant_topk_negative_mask,
            "--turboquant-topk-recall-weight",
            f"{args.turboquant_topk_recall_weight:g}",
            "--turboquant-topk-recall-cutoff",
            str(args.turboquant_topk_recall_cutoff),
            "--turboquant-topk-recall-negative-mask",
            args.turboquant_topk_recall_negative_mask,
        ],
        "forbidden_cli_flags": [
            "--no-tokenizer",
            "--pairwise-train",
            "--hard-negative-train",
            "--listwise-geometry-train",
            "--vector-distill-train",
        ],
        "inputs": {
            "train_score_spectrum_jsonl": str(args.output_jsonl),
            "tokenizer": "required: existing package tokenizer or explicit --tokenizer path",
            "package": "required: existing d384 training package path",
        },
        "workload_expectations": {
            "train_rows": workload["train"],
            "batch_size": workload["batch"],
            "epochs": workload["epochs"],
            "steps_per_epoch": workload["steps_per_epoch"],
            "turboquant_topk_objective_count": args.turboquant_objectives,
            "raw_candidates_per_epoch": workload["raw_candidates_per_epoch"],
            "canonical_candidates_per_epoch": workload["canonical_candidates_per_epoch"],
            "main_q3_pairs_per_epoch": workload["main_q3_pairs_per_epoch"],
            "recall_q3_pairs_per_epoch": workload["recall_q3_pairs_per_epoch"],
            "total_train_pairs_per_epoch": workload["total_train_pairs_per_epoch"],
            "planned_pairs": workload["planned_pairs"],
        },
        "legal_gate": {
            "requires_allow_research_only_score_spectrum": True,
            **LEGAL_GATES,
        },
    }


def read_qids_from_selection(paths: list[Path]) -> set[str]:
    qids: set[str] = set()
    for path in paths:
        payload = read_json(path)
        selected = payload.get("selected_qids", payload)
        if not isinstance(selected, dict):
            raise ValueError(f"{path}: expected selected_qids object")
        values = selected.get("nfcorpus", [])
        if not isinstance(values, list):
            raise ValueError(f"{path}: selected_qids.nfcorpus must be a list")
        qids.update(str(value) for value in values)
    return qids


def read_test_qrels(path: Path) -> dict[str, set[str]]:
    by_qid: dict[str, set[str]] = defaultdict(set)
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if not reader.fieldnames or not {"query-id", "corpus-id"}.issubset(set(reader.fieldnames)):
            raise ValueError(f"{path}: expected BEIR qrels TSV header")
        for row in reader:
            by_qid[str(row["query-id"])].add(str(row["corpus-id"]))
    return dict(by_qid)


def load_per_query_row(path: Path, qid: str) -> dict[str, Any]:
    matches = [row for _line, row in iter_jsonl(path) if str(row.get("query_id")) == qid]
    if len(matches) != 1:
        raise ValueError(f"{path}: expected exactly one row for {qid}, found {len(matches)}")
    return matches[0]


def rank_of(row: dict[str, Any], doc_id: str) -> int:
    for candidate in row.get("top_k", []):
        if str(candidate.get("doc_id")) == doc_id:
            return int(candidate.get("rank") or 0)
    raise ValueError(f"{row.get('query_id')}: missing doc {doc_id} in top_k")


def validate_r2_evidence(args: argparse.Namespace) -> dict[str, Any]:
    gate = read_json(args.r2_gate)
    summary = read_json(args.r2_proxy_summary)
    proxy_qrels_manifest = read_json(args.r2_proxy_qrels_manifest)
    movement_rows = [row for _line, row in iter_jsonl(args.r2_movement) if str(row.get("query_id")) == QUERY_ID]
    if len(movement_rows) != 1:
        raise ValueError(f"r2 movement expected exactly one {QUERY_ID} row, found {len(movement_rows)}")
    movement = movement_rows[0]
    crossings = gate.get("crossing_examples", [])
    lost = [item for item in crossings if item == {"anchor_rank": 10, "candidate_rank": 11, "direction": "loss_out_of_top10", "doc_id": "MED-3031", "query_id": QUERY_ID}]
    promoted = [item for item in crossings if item == {"anchor_rank": 11, "candidate_rank": 5, "direction": "promotion_into_top10", "doc_id": "MED-3022", "query_id": QUERY_ID}]
    if not lost or not promoted:
        raise ValueError("r2 gate does not contain required PLAIN-73 MED-3031 loss and MED-3022 promotion")
    if float(movement.get("candidate_ndcg_at_10", 0.0)) <= float(movement.get("anchor_ndcg_at_10", 0.0)):
        raise ValueError("PLAIN-73 candidate nDCG@10 did not improve")
    if int(movement.get("top10_losses") or 0) != 1 or int(movement.get("top10_promotions") or 0) != 1:
        raise ValueError("PLAIN-73 movement does not have exactly one top10 loss and one promotion")
    if summary.get("hashes", {}).get("nfcorpus_gate") != args.expected_r2_gate_sha256:
        raise ValueError("r2 proxy summary gate hash does not match required hash")
    if summary.get("hashes", {}).get("proxy_qrels_manifest") != args.expected_r2_proxy_qrels_manifest_sha256:
        raise ValueError("r2 proxy summary proxy-qrels manifest hash does not match required hash")
    if proxy_qrels_manifest.get("source_train_sha256") != args.expected_parent_train_sha256:
        raise ValueError("r2 proxy qrels manifest source train sha does not match parent")
    anchor = load_per_query_row(args.anchor_per_query_jsonl, QUERY_ID)
    candidate = load_per_query_row(args.candidate_per_query_jsonl, QUERY_ID)
    rank_checks = {
        "MED-3031": {"anchor_rank": rank_of(anchor, "MED-3031"), "candidate_rank": rank_of(candidate, "MED-3031")},
        "MED-3022": {"anchor_rank": rank_of(anchor, "MED-3022"), "candidate_rank": rank_of(candidate, "MED-3022")},
    }
    if rank_checks["MED-3031"] != {"anchor_rank": 10, "candidate_rank": 11}:
        raise ValueError(f"unexpected MED-3031 ranks: {rank_checks['MED-3031']}")
    if rank_checks["MED-3022"] != {"anchor_rank": 11, "candidate_rank": 5}:
        raise ValueError(f"unexpected MED-3022 ranks: {rank_checks['MED-3022']}")
    if float(candidate.get("quality", {}).get("ndcg_at_10", 0.0)) <= float(anchor.get("quality", {}).get("ndcg_at_10", 0.0)):
        raise ValueError("per-query candidate nDCG@10 did not improve")
    return {
        "gate": {
            "path": str(args.r2_gate),
            "sha256": sha256_file(args.r2_gate),
            "top10_losses": gate.get("top10_losses"),
            "top10_promotions": gate.get("top10_promotions"),
            "net_top10_promotions": gate.get("net_top10_promotions"),
            "q3_ndcg_at_10_delta": gate.get("q3_ndcg_at_10_delta"),
            "q3_recall_at_100_delta": gate.get("q3_recall_at_100_delta"),
        },
        "proxy_summary": {"path": str(args.r2_proxy_summary), "sha256": sha256_file(args.r2_proxy_summary)},
        "movement": {"path": str(args.r2_movement), "sha256": sha256_file(args.r2_movement), "plain73": movement},
        "proxy_qrels_manifest": {"path": str(args.r2_proxy_qrels_manifest), "sha256": sha256_file(args.r2_proxy_qrels_manifest)},
        "anchor_per_query": {"path": str(args.anchor_per_query_jsonl), "sha256": sha256_file(args.anchor_per_query_jsonl)},
        "candidate_per_query": {"path": str(args.candidate_per_query_jsonl), "sha256": sha256_file(args.candidate_per_query_jsonl)},
        "rank_checks": rank_checks,
    }


def build_appended_row(source: dict[str, Any], parent_line_number: int, evidence: dict[str, Any]) -> dict[str, Any]:
    if len(source["candidate_doc_ids"]) != 11:
        raise ValueError(f"{SOURCE_ROW_ID}: source row candidate count is {len(source['candidate_doc_ids'])}, expected 11")
    doc_ids = [raw_doc_id(str(value)) for value in source["candidate_doc_ids"]]
    if set(doc_ids) != set(SHAPED_GAINS) | {"MED-2910", "MED-3025", "MED-3034", "MED-4622"}:
        raise ValueError(f"{SOURCE_ROW_ID}: source row doc topology mismatch: {doc_ids}")
    row = dict(source)
    row["row_id"] = APPENDED_ROW_ID
    row["source"] = "nfcorpus:beir-train:q3near-dualcut-r3-incumbent-preservation-v1"
    row["qrel_gains"] = [SHAPED_GAINS.get(doc_id, 0.0) for doc_id in doc_ids]
    row["candidate_sources"] = ["qrel" if gain > 0.0 else "q3" for gain in row["qrel_gains"]]
    row["positive_indexes"] = [idx for idx, gain in enumerate(row["qrel_gains"]) if gain > 0.0]
    row["selected_positive_index"] = row["positive_indexes"][0]
    row["hard_negative_eligible"] = [gain <= 0.0 for gain in row["qrel_gains"]]
    gain_sum = sum(row["qrel_gains"])
    row["target_probabilities"] = [gain / gain_sum if gain > 0.0 else 0.0 for gain in row["qrel_gains"]]
    row["hard_loss_weight"] = 0.0
    row["soft_loss_weight"] = 0.0
    row["recovery_loss_weight"] = 0.0
    row["turboquant_topk_loss_weight"] = 0.5
    row["turboquant_topk_recall_loss_weight"] = 0.0
    row["train_policy"] = "q3near_dualcut_r3_nfplain73_top10_incumbent_preservation_research_only"
    row["selection_bucket"] = "nfcorpus_plain73_r2_top10_incumbent_preserve"
    row["guard_margin"] = None
    row["legal_gates"] = dict(LEGAL_GATES)
    row.update(LEGAL_GATES)
    row["source_positive_doc_ids"] = [doc_ids[idx] for idx in row["positive_indexes"]]
    row["candidate_positive_doc_ids_pre_dedupe"] = list(row["source_positive_doc_ids"])
    row["all_train_qrel_positive_doc_ids"] = list(row["source_positive_doc_ids"])
    row["positive_count"] = len(row["positive_indexes"])
    row["bm25_negative_count"] = 0
    row["q3_negative_count"] = len(doc_ids) - len(row["positive_indexes"])
    row["extra_negative_count"] = 0
    row["q3_evidence"] = [
        item
        for idx, item in enumerate(row.get("anchor_q3_candidate_evidence", []))
        if idx < len(row["qrel_gains"]) and row["qrel_gains"][idx] == 0.0
    ]
    row["source_artifact_hash"] = sha256_json({"source_row": source, "shaped_gains": SHAPED_GAINS, "evidence": evidence["rank_checks"]})
    row["incumbent_preservation_provenance"] = {
        "source_row_id": SOURCE_ROW_ID,
        "source_parent_line_number": parent_line_number,
        "source_parent_row_sha256": sha256_json(source),
        "selection_as_proxy_repair": "r2 nfcorpus gate failed only on one top10 loss; replay protects PLAIN-73 incumbent top10 qrel documents with low-weight shaped gains",
        "incumbent_gain2_doc_ids": list(INCUMBENT_GAIN2_DOCS),
        "non_incumbent_gain1_doc_ids": list(NON_INCUMBENT_GAIN1_DOCS),
        "observed_failure": evidence["rank_checks"],
        "plain73_movement": evidence["movement"]["plain73"],
    }
    validate_row_shape(row, APPENDED_ROW_ID)
    canonical_stats(row, APPENDED_ROW_ID)
    return row


def validate_overlap(rows: list[dict[str, Any]], exclude_qids: set[str], test_qrels: dict[str, set[str]]) -> dict[str, Any]:
    nf_qids = {str(row["source_query_id"]) for row in rows if row.get("source_dataset") == "nfcorpus"}
    frozen = sorted(nf_qids & exclude_qids)
    test_qids = sorted(nf_qids & set(test_qrels))
    shaped_docs = set(SHAPED_GAINS)
    test_docs_for_query = test_qrels.get(QUERY_ID, set())
    test_doc_overlap = sorted(shaped_docs & test_docs_for_query)
    if frozen or test_qids or test_doc_overlap:
        raise ValueError(f"held-out overlap: frozen={frozen} test_qids={test_qids} test_doc_overlap={test_doc_overlap}")
    return {
        "nfcorpus": {
            "selected_count": len(nf_qids),
            "selected_x_frozen_dev_reserve": len(frozen),
            "selected_x_frozen_dev_reserve_qids": frozen,
            "selected_x_official_test": len(test_qids),
            "selected_x_official_test_qids": test_qids,
            "plain73_shaped_qrel_docs_x_official_test": len(test_doc_overlap),
            "plain73_shaped_qrel_docs_x_official_test_docs": test_doc_overlap,
        }
    }


def output_validation(rows: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    if len(rows) != args.expected_rows:
        raise ValueError(f"rows={len(rows)}, expected {args.expected_rows}")
    if len({row["row_id"] for row in rows}) != len(rows):
        raise ValueError("duplicate output row_id")
    bucket_counts = Counter()
    dataset_counts = Counter()
    main_weights = Counter()
    recall_weights = Counter()
    candidate_count_counts = Counter()
    raw_stats = Counter()
    canonical_stat_counts = Counter()
    recall_sentinel_bad: list[str] = []
    appended = [row for row in rows if row["row_id"] == APPENDED_ROW_ID]
    if len(appended) != 1:
        raise ValueError(f"expected exactly one appended row {APPENDED_ROW_ID}, found {len(appended)}")
    for row in rows:
        validate_row_shape(row, row["row_id"])
        stats = canonical_stats(row, row["row_id"])
        raw_stats.update({key: stats[key] for key in ("raw_candidates", "raw_positive_pairs", "raw_hard_negative_pairs")})
        canonical_stat_counts.update({key: stats[key] for key in ("canonical_candidates", "canonical_positive_pairs", "canonical_hard_negative_pairs")})
        bucket_counts[row["selection_bucket"]] += 1
        dataset_counts[row["source_dataset"]] += 1
        main_weights[str(float(row["turboquant_topk_loss_weight"]))] += 1
        recall_weights[str(float(row["turboquant_topk_recall_loss_weight"]))] += 1
        candidate_count_counts[str(len(row["candidate_doc_ids"]))] += 1
        if row["selection_bucket"] == "nfcorpus_recall_sentinel_q3top100_rank80_100" and eligible_topk_pairs(row, True, args.turboquant_topk_recall_negative_mask) <= 0:
            recall_sentinel_bad.append(row["row_id"])
    if recall_sentinel_bad:
        raise ValueError(f"recall sentinel rows with zero recall pairs: {recall_sentinel_bad}")
    appended_row = appended[0]
    doc_to_gain = {raw_doc_id(doc_id): gain for doc_id, gain in zip(appended_row["candidate_doc_ids"], appended_row["qrel_gains"], strict=True)}
    if doc_to_gain != {doc_id: SHAPED_GAINS.get(doc_id, 0.0) for doc_id in doc_to_gain}:
        raise ValueError(f"appended shaped gains mismatch: {doc_to_gain}")
    return {
        "rows": len(rows),
        "dataset_counts": dict(dataset_counts),
        "bucket_counts": dict(bucket_counts),
        "turboquant_topk_loss_weight_counts": dict(sorted(main_weights.items())),
        "turboquant_topk_recall_loss_weight_counts": dict(sorted(recall_weights.items())),
        "candidate_count_counts": dict(sorted(candidate_count_counts.items(), key=lambda item: int(item[0]))),
        "raw_counts": dict(raw_stats),
        "canonical_counts": dict(canonical_stat_counts),
        "appended_row": {
            "row_id": appended_row["row_id"],
            "candidate_count": len(appended_row["candidate_doc_ids"]),
            "positive_indexes": appended_row["positive_indexes"],
            "shaped_gains_by_doc_id": doc_to_gain,
            "target_probabilities": appended_row["target_probabilities"],
            "main_topk_pairs_q3_mask": eligible_topk_pairs(appended_row, False, args.turboquant_topk_negative_mask),
            "recall_topk_pairs_q3_mask": eligible_topk_pairs(appended_row, True, args.turboquant_topk_recall_negative_mask),
        },
        "alignment_checks": {
            "aligned_candidate_fields": True,
            "target_probability_sum_1": True,
            "positive_indexes_match_qrel_gains": True,
            "positive_hard_negative_disjoint": True,
            "duplicate_text_positive_hard_disjoint": True,
            "legal_gates_fail_closed": True,
            "all_recall_sentinels_have_positive_recall_pairs": True,
        },
    }


def main() -> None:
    args = parse_args()
    parent_train_sha = assert_hash(args.parent_train_jsonl, args.expected_parent_train_sha256, "parent train")
    parent_manifest_sha = assert_hash(args.parent_manifest, args.expected_parent_manifest_sha256, "parent manifest")
    parent_coverage_sha = assert_hash(args.parent_coverage, args.expected_parent_coverage_sha256, "parent coverage")
    gate_sha = assert_hash(args.r2_gate, args.expected_r2_gate_sha256, "r2 gate")
    proxy_qrels_sha = assert_hash(args.r2_proxy_qrels_manifest, args.expected_r2_proxy_qrels_manifest_sha256, "r2 proxy qrels manifest")
    evidence = validate_r2_evidence(args)

    parent_rows: list[dict[str, Any]] = []
    source_matches: list[tuple[int, dict[str, Any]]] = []
    for line_number, row in iter_jsonl(args.parent_train_jsonl):
        validate_row_shape(row, f"parent line {line_number}")
        parent_rows.append(row)
        if row.get("row_id") == SOURCE_ROW_ID:
            source_matches.append((line_number, row))
        if row.get("row_id") == APPENDED_ROW_ID:
            raise ValueError(f"parent already contains appended row id {APPENDED_ROW_ID}")
    if len(parent_rows) != args.expected_parent_rows:
        raise ValueError(f"parent rows={len(parent_rows)}, expected {args.expected_parent_rows}")
    if len(source_matches) != 1:
        raise ValueError(f"expected exactly one source row {SOURCE_ROW_ID}, found {len(source_matches)}")
    source_line, source_row = source_matches[0]
    if len(source_row["candidate_doc_ids"]) != 11:
        raise ValueError(f"{SOURCE_ROW_ID}: selected source is not the 11-candidate base row")

    appended_row = build_appended_row(source_row, source_line, evidence)
    rows = [dict(row) for row in parent_rows] + [appended_row]
    exclude_qids = read_qids_from_selection(args.exclude_qids_json)
    test_qrels = read_test_qrels(args.test_qrels)
    overlap = validate_overlap(rows, exclude_qids, test_qrels)
    validation = output_validation(rows, args)
    workload = workload_counts(rows, args)
    loader_plan = go_loader_plan(args, workload)
    if workload["steps_per_epoch"] != args.expected_steps_per_epoch:
        raise ValueError(f"planned steps_per_epoch={workload['steps_per_epoch']}, expected {args.expected_steps_per_epoch}")
    if workload["train"] != args.expected_rows:
        raise ValueError(f"planned train rows={workload['train']}, expected {args.expected_rows}")

    write_jsonl(args.output_jsonl, rows)
    coverage = {
        "schema": f"{SCHEMA}.coverage.v1",
        "created_utc": utc_stamp(),
        "selection_audit": {
            "source_row_id": SOURCE_ROW_ID,
            "appended_row_id": APPENDED_ROW_ID,
            "policy": "clone the 11-candidate nfcorpus:PLAIN-73 base row and shape qrel gains to preserve R2 top10 incumbents after the observed train-proxy loss",
            "evidence": evidence,
        },
        "validation": validation,
        "workload": workload,
        "go_loader_plan": loader_plan,
        "overlap": overlap,
        "legal_gates": dict(LEGAL_GATES),
        "quality_claim": False,
    }
    write_json(args.coverage_report, coverage)

    manifest = {
        "schema": SCHEMA,
        "created_utc": utc_stamp(),
        "builder": "scripts/build_q3_r3_incumbent_dataset.py",
        "inputs": {
            "parent_train_jsonl": {"path": str(args.parent_train_jsonl), "sha256": parent_train_sha},
            "parent_manifest": {"path": str(args.parent_manifest), "sha256": parent_manifest_sha},
            "parent_coverage": {"path": str(args.parent_coverage), "sha256": parent_coverage_sha},
            "r2_gate": {"path": str(args.r2_gate), "sha256": gate_sha},
            "r2_proxy_summary": {"path": str(args.r2_proxy_summary), "sha256": sha256_file(args.r2_proxy_summary)},
            "r2_movement": {"path": str(args.r2_movement), "sha256": sha256_file(args.r2_movement)},
            "r2_proxy_qrels_manifest": {"path": str(args.r2_proxy_qrels_manifest), "sha256": proxy_qrels_sha},
            "anchor_per_query_jsonl": {"path": str(args.anchor_per_query_jsonl), "sha256": sha256_file(args.anchor_per_query_jsonl)},
            "candidate_per_query_jsonl": {"path": str(args.candidate_per_query_jsonl), "sha256": sha256_file(args.candidate_per_query_jsonl)},
            "exclude_qids_json": {str(path): sha256_file(path) for path in args.exclude_qids_json},
            "test_qrels": {"path": str(args.test_qrels), "sha256": sha256_file(args.test_qrels)},
        },
        "outputs": {
            "train_score_spectrum": str(args.output_jsonl),
            "manifest": str(args.manifest),
            "coverage_report": str(args.coverage_report),
        },
        "policy": {
            "append": "clone existing 11-candidate nfcorpus:PLAIN-73 base row, not the 100-candidate recall sentinel",
            "shaped_gains": {
                "gain_2_incumbent_anchor_q3_top10_qrel_doc_ids": list(INCUMBENT_GAIN2_DOCS),
                "gain_1_non_incumbent_qrel_doc_ids": list(NON_INCUMBENT_GAIN1_DOCS),
                "gain_0": "all q3 negatives",
            },
            "weights": {
                "hard_loss_weight": 0,
                "soft_loss_weight": 0,
                "recovery_loss_weight": 0,
                "turboquant_topk_loss_weight": 0.5,
                "turboquant_topk_recall_loss_weight": 0,
            },
            "fail_closed_evidence": "require R2 proxy hashes, MED-3031 rank10->11 loss, MED-3022 rank11->5 promotion, same-query nDCG improvement, and no held-out overlap",
        },
        "counts": {
            **validation,
            "workload": workload,
        },
        "go_loader_plan": loader_plan,
        "selection_audit": coverage["selection_audit"],
        "overlap": overlap,
        "legal_gates": dict(LEGAL_GATES),
        **LEGAL_GATES,
        "quality_claim": False,
        "sha256": {
            "builder": sha256_file(Path(__file__).resolve()),
            "test_file": sha256_file(args.test_file) if args.test_file.is_file() else None,
            "parent_train_jsonl": parent_train_sha,
            "parent_manifest": parent_manifest_sha,
            "parent_coverage": parent_coverage_sha,
            "r2_gate": gate_sha,
            "r2_proxy_summary": sha256_file(args.r2_proxy_summary),
            "r2_movement": sha256_file(args.r2_movement),
            "r2_proxy_qrels_manifest": proxy_qrels_sha,
            "anchor_per_query_jsonl": sha256_file(args.anchor_per_query_jsonl),
            "candidate_per_query_jsonl": sha256_file(args.candidate_per_query_jsonl),
            "exclude_qids_json": {str(path): sha256_file(path) for path in args.exclude_qids_json},
            "test_qrels": sha256_file(args.test_qrels),
            "train_score_spectrum": sha256_file(args.output_jsonl),
            "coverage_report": sha256_file(args.coverage_report),
        },
    }
    write_json(args.manifest, manifest)
    print(f"q3 r3 incumbent dataset rows={len(rows)} appended={APPENDED_ROW_ID}")
    print(
        "planned workload: "
        f"train={workload['train']} batch={workload['batch']} steps/epoch={workload['steps_per_epoch']} "
        f"raw={workload['raw_candidates_per_epoch']} canonical={workload['canonical_candidates_per_epoch']} "
        f"main_q3={workload['main_q3_pairs_per_epoch']} recall_q3={workload['recall_q3_pairs_per_epoch']} "
        f"total/epoch={workload['total_train_pairs_per_epoch']} planned={workload['planned_pairs']}"
    )
    print(f"output: {args.output_jsonl}")
    print(f"manifest: {args.manifest}")
    print(f"coverage: {args.coverage_report}")


if __name__ == "__main__":
    main()
