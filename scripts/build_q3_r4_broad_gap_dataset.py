#!/usr/bin/env python3
"""Build the q3 R4 broad-gap promotion/guard/recall-sentinel dataset."""

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
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from build_q3_dualcut_recall_dataset import (  # noqa: E402
    build_sentinel_row,
    normalize_candidate_text,
    validate_row_legal_and_alignment,
)
from build_q3_r3_incumbent_dataset import (  # noqa: E402
    canonical_stats,
    eligible_topk_pairs,
)


SCHEMA = "eos.q3_r4_broad_gap_dualcut_dataset.v1"
LEGAL_GATES = {
    "train_allowed_for_research": True,
    "release_train_allowed": False,
    "commercial_use_allowed": False,
}
Q3_METHOD = "turboquant_ip_b3"
Q3_BITS = 3
Q3_DIM = 384
Q3_SEED = 5581486560434873699
Q3_SURFACE = "turboquant_ip_prepared"
DEFAULT_EXPECTED_Q3_HASHES = {
    "fiqa": "79d2bfaaf26abc1980487aa777e1c6a3cae9db965e2cab42ad63840702b09a3b",
    "nfcorpus": "b364ede6b3aecb605deabd95c3767d12d23fcdf7c69b061821f4b58cdf3c48eb",
    "scifact": "ab66dda3bedc7ea5886816759238e64425c08ac7c22304f632ae04c2ec5748e9",
}
PARENT_SENTINEL_TRAIN_SHA256 = "9a950dbe8eddcdcd1f20fefe581db2c2b9a4bee2e3678f346577b5475bee5433"
ANCHOR_PACKAGE = Path("runs/eos-wide384-projection-tail-seed191-repeat-v1-20260902T004753Z/packages/d384-pre.mll")
CANDIDATE_ID = "arm-q3broad-gap050-nf64-fiqa21-sf1-nfrecall24-rec005-q3w004-rw015-m002-lr2e6-e2-r4"
CANDIDATE_PACKAGE = Path(f"runs/eos-d384-q3-boundary-goal-v1-20260902T084144Z/packages/{CANDIDATE_ID}/d384-pre.mll")
METRICS_TARGET = Path(f"runs/eos-d384-q3-boundary-goal-v1-20260902T084144Z/metrics/{CANDIDATE_ID}.train.metrics.json")
PROMOTION_BUCKET = "promotion_11_20_to_top10_near_gap"
GUARD_BUCKET = "guard_top10_margin"
SENTINEL_BUCKET = "nfcorpus_recall_sentinel_q3top100_rank80_100"
REQUIRED_TRAIN_FLAGS: dict[str, Any] = {
    "plan_only": True,
    "score_spectrum_train": True,
    "allow_research_only_score_spectrum": True,
    "contrastive_loss": "grouped_infonce",
    "temperature": 0.05,
    "score_spectrum_loss_mode": "hard_soft_recovery",
    "score_spectrum_recovery_weight": 0.05,
    "score_spectrum_recovery_margin": 0.0,
    "score_spectrum_recovery_top_k": 4,
    "score_spectrum_recovery_tau": 0.05,
    "turboquant_prefix_score_mode": "prepared_ip",
    "turboquant_prefix_seed": Q3_SEED,
    "turboquant_topk_objectives": "fullDim:3=0.04",
    "turboquant_topk_loss": "lambdandcg",
    "turboquant_topk_cutoff": 10,
    "turboquant_topk_tau": 0.05,
    "turboquant_topk_margin": 0.002,
    "turboquant_topk_negative_mask": "q3",
    "turboquant_topk_recall_weight": 0.15,
    "turboquant_topk_recall_cutoff": 100,
    "turboquant_topk_recall_tau": 0.05,
    "turboquant_topk_recall_margin": 0.0,
    "turboquant_topk_recall_negative_mask": "q3",
    "lr": 0.000002,
    "epochs": 2,
    "batch_size": 4,
    "seed": 191,
    "restore_best": False,
}
EXPECTED_R4_WORKLOAD: dict[str, int] = {
    "train": 134,
    "batch": 4,
    "steps_per_epoch": 34,
    "raw_candidates_per_epoch": 3815,
    "canonical_candidates_per_epoch": 3725,
    "main_q3_pairs_per_epoch": 2701,
    "recall_q3_pairs_per_epoch": 40486,
    "total_train_pairs_per_epoch": 46912,
    "planned_pairs": 93824,
    "actual_train_pairs": 0,
}
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
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--q3-per-query-jsonl", action="append", required=True, type=Path)
    parser.add_argument("--parent-sentinel-train-jsonl", required=True, type=Path)
    parser.add_argument("--parent-sentinel-manifest", required=True, type=Path)
    parser.add_argument("--parent-sentinel-coverage", required=True, type=Path)
    parser.add_argument("--anchor-package", type=Path, default=ANCHOR_PACKAGE)
    parser.add_argument("--candidate-package", type=Path, default=CANDIDATE_PACKAGE)
    parser.add_argument("--metrics-target", type=Path, default=METRICS_TARGET)
    parser.add_argument("--expected-parent-sentinel-train-sha256", default=PARENT_SENTINEL_TRAIN_SHA256)
    parser.add_argument("--expected-q3-sha256", action="append", default=[], help="override expected q3 source digest as dataset=sha256")
    parser.add_argument("--exclude-qids-json", action="append", default=[], type=Path)
    parser.add_argument("--output-jsonl", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--coverage-report", required=True, type=Path)
    parser.add_argument("--promotion-max-score-gap", type=float, default=0.05)
    parser.add_argument("--nfcorpus-promotion-count", type=int, default=64)
    parser.add_argument("--expected-nfcorpus-eligible", type=int, default=235)
    parser.add_argument("--expected-scifact-eligible", type=int, default=1)
    parser.add_argument("--fiqa-promotion-count", type=int, default=21)
    parser.add_argument("--scifact-required-qid", default="243")
    parser.add_argument("--guard-count-per-dataset", type=int, default=12)
    parser.add_argument("--expected-guard-total", type=int, default=24)
    parser.add_argument("--sentinel-count", type=int, default=24)
    parser.add_argument("--expected-rows", type=int, default=134)
    parser.add_argument("--expected-promotion-total", type=int, default=86)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--turboquant-objectives", type=int, default=1)
    parser.add_argument("--turboquant-topk-negative-mask", default="q3")
    parser.add_argument("--turboquant-topk-recall-negative-mask", default="q3")
    parser.add_argument("--expected-steps-per-epoch", type=int, default=34)
    parser.add_argument("--test-file", default=Path("scripts/test_build_q3_r4_broad_gap_dataset.py"), type=Path)
    return parser.parse_args()


def hash_package_siblings(anchor_package: Path) -> dict[str, Any]:
    if not anchor_package.is_file():
        raise ValueError(f"missing anchor package: {anchor_package}")
    stem = anchor_package.name[:-4] if anchor_package.name.endswith(".mll") else anchor_package.stem
    siblings = sorted(path for path in anchor_package.parent.glob(f"{stem}*") if path.is_file())
    if anchor_package not in siblings:
        raise ValueError(f"anchor package missing from sibling rollup: {anchor_package}")
    sibling_hashes = [
        {"path": str(path), "name": path.name, "sha256": sha256_file(path), "bytes": path.stat().st_size}
        for path in siblings
    ]
    content_items = [{"name": item["name"], "sha256": item["sha256"], "bytes": item["bytes"]} for item in sibling_hashes]
    rollup = sha256_strings([json.dumps(item, sort_keys=True, separators=(",", ":")) for item in content_items])
    tokenizer = anchor_package.with_name(f"{stem}.tokenizer.mll")
    if not tokenizer.is_file():
        raise ValueError(f"missing required tokenizer sibling: {tokenizer}")
    return {
        "anchor_package": str(anchor_package),
        "anchor_package_absolute": str(anchor_package.resolve()),
        "anchor_package_sha256": sha256_file(anchor_package),
        "sibling_count": len(sibling_hashes),
        "sibling_hashes": sibling_hashes,
        "sibling_content_hashes": content_items,
        "sibling_rollup_sha256": rollup,
        "tokenizer_package": str(tokenizer),
        "tokenizer_package_sha256": sha256_file(tokenizer),
    }


def train_argv_tokens(args: argparse.Namespace, *, plan_only: bool) -> list[str]:
    tokens = ["train-embed"]
    if plan_only:
        tokens.append("--plan-only")
    tokens.extend(
        [
            "--score-spectrum-train",
            "--allow-research-only-score-spectrum",
            "--shuffle=false",
            "--epochs", "2",
            "--batch-size", "4",
            "--seed", "191",
            "--contrastive-loss", "grouped_infonce",
            "--temperature", "0.05",
            "--lr", "0.000002",
            "--restore-best=false",
            "--metrics-json", str(args.metrics_target),
            "--score-spectrum-loss-mode", "hard_soft_recovery",
            "--score-spectrum-recovery-weight", "0.05",
            "--score-spectrum-recovery-margin", "0",
            "--score-spectrum-recovery-top-k", "4",
            "--score-spectrum-recovery-tau", "0.05",
            "--turboquant-topk-objectives", "fullDim:3=0.04",
            "--turboquant-topk-loss", "lambdandcg",
            "--turboquant-topk-cutoff", "10",
            "--turboquant-topk-tau", "0.05",
            "--turboquant-topk-margin", "0.002",
            "--turboquant-topk-negative-mask", "q3",
            "--turboquant-topk-recall-weight", "0.15",
            "--turboquant-topk-recall-cutoff", "100",
            "--turboquant-topk-recall-tau", "0.05",
            "--turboquant-topk-recall-margin", "0",
            "--turboquant-topk-recall-negative-mask", "q3",
            "--turboquant-prefix-score-mode", "prepared_ip",
            "--turboquant-prefix-seed", str(Q3_SEED),
            str(args.candidate_package),
            str(args.output_jsonl),
        ]
    )
    return tokens


def argv_delta(plan_argv: list[str], train_argv: list[str]) -> dict[str, Any]:
    removed = [item for item in plan_argv if item not in train_argv]
    added = [item for item in train_argv if item not in plan_argv]
    return {
        "removed_for_training": removed,
        "added_for_training": added,
        "exactly_one_semantic_delta": removed == ["--plan-only"] and added == [],
        "semantic_delta": "remove --plan-only only",
    }


def go_loader_plan_contract(args: argparse.Namespace, validation: dict[str, Any], hashes: dict[str, Any]) -> dict[str, Any]:
    workload = {
        "train": validation["rows"],
        "batch": validation["batch"],
        "steps_per_epoch": validation["steps_per_epoch"],
        "raw_candidates_per_epoch": validation["raw_candidates_per_epoch"],
        "canonical_candidates_per_epoch": validation["canonical_candidates_per_epoch"],
        "main_q3_pairs_per_epoch": validation["main_q3_pairs_per_epoch"],
        "recall_q3_pairs_per_epoch": validation["recall_q3_pairs_per_epoch"],
        "total_train_pairs_per_epoch": validation["total_train_pairs_per_epoch"],
        "planned_pairs": validation["planned_pairs"],
        "actual_train_pairs": 0,
    }
    is_real_contract = args.expected_rows == EXPECTED_R4_WORKLOAD["train"]
    if is_real_contract:
        drift = {key: (EXPECTED_R4_WORKLOAD[key], workload.get(key)) for key in EXPECTED_R4_WORKLOAD if EXPECTED_R4_WORKLOAD[key] != workload.get(key)}
        if drift:
            raise ValueError(f"R4 workload contract drift: {drift}")
    plan_argv = train_argv_tokens(args, plan_only=True)
    training_argv = train_argv_tokens(args, plan_only=False)
    return {
        "schema": f"{SCHEMA}.go_loader_plan_contract.v1",
        "training_blocked_until_re_review": True,
        "successful_plan_record_required_before_training": True,
        "candidate_id": CANDIDATE_ID,
        "id_token_legend": {
            "q3w004": "literal turboquant_topk_objectives fullDim:3=0.04",
            "rw015": "literal turboquant_topk_recall_weight=0.15",
            "m002": "literal turboquant_topk_margin=0.002",
        },
        "ingestion": {
            "row_format": "text_score_spectrum",
            "tokenizer_backed": True,
            "required_package_tokenizer": hashes["anchor"]["tokenizer_package"],
            "required_package_tokenizer_sha256": hashes["anchor"]["tokenizer_package_sha256"],
            "forbidden_flags": ["--no-tokenizer"],
            "forbidden_no_tokenizer": True,
            "no_eval_rows": True,
        },
        "required_flags": dict(REQUIRED_TRAIN_FLAGS),
        "required_plan_argv_tokens": plan_argv,
        "required_training_argv_tokens": training_argv,
        "required_cli_tokens": plan_argv,
        "argv_delta": argv_delta(plan_argv, training_argv),
        "anchor": hashes["anchor"],
        "candidate": {
            "candidate_package": str(args.candidate_package),
            "candidate_package_absolute": str(args.candidate_package.resolve()),
            "candidate_dir": str(args.candidate_package.parent),
            "candidate_dir_must_be_fresh_before_clone": True,
            "candidate_target_differs_from_anchor": str(args.candidate_package.resolve()) != str(args.anchor_package.resolve()),
        },
        "metrics_target": str(args.metrics_target),
        "metrics_target_must_be_absent_before_training": True,
        "artifact_hashes": hashes["artifact_hashes"],
        "source_hashes": hashes["source_hashes"],
        "parent_hashes": hashes["parent_hashes"],
        "workload": workload,
        "legal_gates": dict(LEGAL_GATES),
        "quality_claim": False,
    }


def expected_q3_hashes(args: argparse.Namespace) -> dict[str, str]:
    values = dict(DEFAULT_EXPECTED_Q3_HASHES)
    for item in args.expected_q3_sha256:
        if "=" not in item:
            raise ValueError("--expected-q3-sha256 must be dataset=sha256")
        dataset, digest = item.split("=", 1)
        values[dataset.lower()] = digest
    return values


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
            if line:
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


def dataset_dir(root: Path, dataset: str) -> Path:
    return root / dataset / dataset


def read_text_jsonl(path: Path, *, corpus: bool) -> dict[str, str]:
    values: dict[str, str] = {}
    for line_number, row in iter_jsonl(path):
        item_id = str(row.get("_id") or "")
        if not item_id:
            raise ValueError(f"{path}:{line_number}: missing _id")
        if corpus:
            title = str(row.get("title") or "").strip()
            text = str(row.get("text") or "").strip()
            values[item_id] = f"{title}\n{text}" if title and text else title or text
        else:
            values[item_id] = str(row.get("text") or "")
    return values


def read_qrels(path: Path) -> dict[str, dict[str, int]]:
    qrels: dict[str, dict[str, int]] = defaultdict(dict)
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if not reader.fieldnames or not {"query-id", "corpus-id", "score"}.issubset(reader.fieldnames):
            raise ValueError(f"{path}: expected BEIR qrels TSV header")
        for row in reader:
            gain = int(row["score"])
            if gain > 0:
                qrels[str(row["query-id"])][str(row["corpus-id"])] = gain
    return dict(qrels)


def read_qrel_qids(path: Path) -> set[str]:
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if not reader.fieldnames or "query-id" not in reader.fieldnames:
            raise ValueError(f"{path}: expected BEIR qrels TSV header")
        return {str(row["query-id"]) for row in reader if row.get("query-id")}


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


def doc_id_for_dataset(dataset: str, candidate_doc_id: str) -> str:
    prefix = f"{dataset}:"
    return candidate_doc_id[len(prefix) :] if candidate_doc_id.startswith(prefix) else candidate_doc_id


def load_q3_rows(paths: list[Path], expected_hashes: dict[str, str]) -> tuple[dict[str, dict[str, dict[str, Any]]], list[dict[str, Any]]]:
    rows: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    identities: list[dict[str, Any]] = []
    seen_datasets: set[str] = set()
    for path in paths:
        digest = sha256_file(path)
        count = 0
        datasets = Counter()
        for line_number, row in iter_jsonl(path):
            dataset = str(row.get("dataset") or "").lower()
            qid = str(row.get("query_id") or "")
            if dataset not in expected_hashes:
                raise ValueError(f"{path}:{line_number}: unexpected dataset {dataset!r}")
            if row.get("method") != Q3_METHOD or int(row.get("bits") or 0) != Q3_BITS:
                raise ValueError(f"{path}:{line_number}: unexpected q3 method/bits")
            if int(row.get("quantizer_seed") or 0) != Q3_SEED or str(row.get("scoring_surface")) != Q3_SURFACE:
                raise ValueError(f"{path}:{line_number}: unexpected q3 seed/surface")
            top_k = row.get("top_k")
            if not isinstance(top_k, list) or len(top_k) != 100:
                raise ValueError(f"{path}:{line_number}: expected top_k length 100")
            ranks = [int(candidate.get("rank") or 0) for candidate in top_k]
            if ranks != list(range(1, 101)):
                raise ValueError(f"{path}:{line_number}: q3 ranks are not exactly 1..100")
            if qid in rows[dataset]:
                raise ValueError(f"{path}:{line_number}: duplicate q3 query {dataset}:{qid}")
            rows[dataset][qid] = row
            datasets[dataset] += 1
            count += 1
        if len(datasets) != 1:
            raise ValueError(f"{path}: expected exactly one dataset, got {dict(datasets)}")
        dataset = next(iter(datasets))
        seen_datasets.add(dataset)
        expected = expected_hashes[dataset]
        if digest != expected:
            raise ValueError(f"{dataset} q3 sha mismatch: expected={expected} actual={digest}")
        identities.append({"path": str(path), "absolute_path": str(path.resolve()), "sha256": digest, "rows": count, "datasets": dict(datasets)})
    missing = sorted(set(expected_hashes) - seen_datasets)
    if missing:
        raise ValueError(f"missing q3 datasets: {missing}")
    return {dataset: dict(by_qid) for dataset, by_qid in rows.items()}, sorted(identities, key=lambda item: item["path"])


def load_domain_sources(root: Path, dataset: str) -> dict[str, Any]:
    base = dataset_dir(root, dataset)
    paths = {
        "queries": base / "queries.jsonl",
        "corpus": base / "corpus.jsonl",
        "train_qrels": base / "qrels" / "train.tsv",
        "test_qrels": base / "qrels" / "test.tsv",
    }
    return {
        "queries": read_text_jsonl(paths["queries"], corpus=False),
        "corpus": read_text_jsonl(paths["corpus"], corpus=True),
        "qrels": read_qrels(paths["train_qrels"]),
        "test_qids": read_qrel_qids(paths["test_qrels"]),
        "paths": paths,
    }


def q3_gap_features(row: dict[str, Any], qrels: dict[str, dict[str, int]]) -> dict[str, Any]:
    qid = str(row["query_id"])
    gains = qrels.get(qid, {})
    if not gains:
        return {"eligible_before_gap": False, "eligible": False}
    positives_11_20: list[dict[str, Any]] = []
    nonrel_top10: list[dict[str, Any]] = []
    positive_top10: list[dict[str, Any]] = []
    for candidate in row["top_k"]:
        rank = int(candidate["rank"])
        doc_id = str(candidate["doc_id"])
        gain = gains.get(doc_id, 0)
        if int(candidate.get("relevance") or 0) != int(gain):
            raise ValueError(f"{row['dataset']}:{qid}: q3 relevance/qrels mismatch for {doc_id}")
        item = {"doc_id": doc_id, "rank": rank, "score": float(candidate["score"]), "gain": gain}
        if 11 <= rank <= 20 and gain > 0:
            positives_11_20.append(item)
        if 1 <= rank <= 10 and gain <= 0:
            nonrel_top10.append(item)
        if 1 <= rank <= 10 and gain > 0:
            positive_top10.append(item)
    gap = None
    if positives_11_20 and nonrel_top10:
        gap = min(neg["score"] - pos["score"] for neg in nonrel_top10 for pos in positives_11_20)
    margin = None
    if positive_top10 and nonrel_top10:
        margin = min(pos["score"] - neg["score"] for pos in positive_top10 for neg in nonrel_top10)
    return {
        "eligible_before_gap": bool(positives_11_20 and nonrel_top10),
        "eligible": bool(positives_11_20 and nonrel_top10 and gap is not None and gap <= 0.05),
        "promotion_score_gap": gap,
        "promotion_positive_ranks_11_20": sorted(item["rank"] for item in positives_11_20),
        "q3_nonrelevant_top10_count": len(nonrel_top10),
        "guard": bool(positive_top10 and nonrel_top10),
        "guard_margin": margin,
        "positive_top10_count": len(positive_top10),
    }


def candidate_evidence(candidate: dict[str, Any], source_path: Path) -> dict[str, Any]:
    return {
        "doc_id": str(candidate["doc_id"]),
        "q3_rank": int(candidate["rank"]),
        "q3_score": candidate.get("score"),
        "relevance": int(candidate.get("relevance") or 0),
        "dense_rank": candidate.get("dense_rank"),
        "dense_score": candidate.get("dense_score"),
        "compact_rank": candidate.get("compact_rank"),
        "compact_score": candidate.get("compact_score"),
        "source_path": str(source_path),
    }


def build_q3_row(
    q3_row: dict[str, Any],
    *,
    qrels: dict[str, dict[str, int]],
    queries: dict[str, str],
    corpus: dict[str, str],
    source_path: Path,
    bucket: str,
    gap: float | None,
    main_weight: float,
    recall_weight: float,
) -> dict[str, Any]:
    dataset = str(q3_row["dataset"]).lower()
    qid = str(q3_row["query_id"])
    if qid not in queries:
        raise ValueError(f"{dataset}:{qid}: missing query text")
    gains = qrels.get(qid, {})
    if not gains:
        raise ValueError(f"{dataset}:{qid}: missing train qrels")
    selected_candidates = []
    for candidate in q3_row["top_k"]:
        rank = int(candidate["rank"])
        doc_id = str(candidate["doc_id"])
        gain = gains.get(doc_id, 0)
        keep_positive = gain > 0 and 1 <= rank <= 20
        keep_negative = gain <= 0 and 1 <= rank <= 10
        if keep_positive or keep_negative:
            selected_candidates.append((candidate, gain))
    if not selected_candidates:
        raise ValueError(f"{dataset}:{qid}: no retained candidates")
    candidate_doc_ids: list[str] = []
    candidate_texts: list[str] = []
    candidate_sources: list[str] = []
    qrel_gains: list[float] = []
    hard_negative_eligible: list[bool] = []
    evidence: list[dict[str, Any]] = []
    for candidate, gain in selected_candidates:
        doc_id = str(candidate["doc_id"])
        if doc_id not in corpus:
            raise ValueError(f"{dataset}:{qid}: missing corpus text for {doc_id}")
        candidate_doc_ids.append(f"{dataset}:{doc_id}")
        candidate_texts.append(corpus[doc_id])
        candidate_sources.append("qrel" if gain > 0 else "q3")
        qrel_gains.append(float(gain))
        hard_negative_eligible.append(gain <= 0)
        evidence.append(candidate_evidence(candidate, source_path))
    positive_indexes = [index for index, gain in enumerate(qrel_gains) if gain > 0.0]
    negative_indexes = [index for index, gain in enumerate(qrel_gains) if gain <= 0.0]
    if not positive_indexes or not negative_indexes:
        raise ValueError(f"{dataset}:{qid}: requires at least one positive and one q3 negative")
    positive_texts = {normalize_candidate_text(candidate_texts[index]) for index in positive_indexes}
    positive_text_alias_indexes = [
        index for index in negative_indexes if normalize_candidate_text(candidate_texts[index]) in positive_texts
    ]
    for index in positive_text_alias_indexes:
        hard_negative_eligible[index] = False
    gain_sum = sum(qrel_gains[index] for index in positive_indexes)
    target_probabilities = [0.0] * len(candidate_doc_ids)
    for index in positive_indexes:
        target_probabilities[index] = qrel_gains[index] / gain_sum
    row = {
        "row_id": f"{dataset}:{qid}:r4-broad-gap-dualcut",
        "source": f"{dataset}:beir-train:q3-r4-broad-gap-dualcut-v1",
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
        "turboquant_topk_loss_weight": main_weight,
        "turboquant_topk_recall_loss_weight": recall_weight,
        "train_policy": "q3_r4_broad_gap_dualcut_research_only",
        "selection_bucket": bucket,
        "guard_margin": gap,
        "legal_gates": dict(LEGAL_GATES),
        **LEGAL_GATES,
        "source_artifact_hash": sha256_json({"q3_query": q3_row, "train_qrel_positive_doc_ids": sorted(gains)}),
        "source_dataset": dataset,
        "source_query_id": qid,
        "source_positive_doc_ids": [doc_id_for_dataset(dataset, candidate_doc_ids[index]) for index in positive_indexes],
        "candidate_positive_doc_ids_pre_dedupe": [doc_id_for_dataset(dataset, candidate_doc_ids[index]) for index in positive_indexes],
        "all_train_qrel_positive_doc_ids": sorted(gains),
        "positive_count": len(positive_indexes),
        "bm25_negative_count": 0,
        "q3_negative_count": sum(1 for value in hard_negative_eligible if value),
        "q3_positive_text_alias_count": len(positive_text_alias_indexes),
        "q3_positive_text_alias_candidate_indexes": positive_text_alias_indexes,
        "extra_negative_count": 0,
        "q3_evidence": [evidence[index] for index in negative_indexes],
        "anchor_q3_candidate_evidence": evidence,
        "q3_scoring_metadata": {
            "method": Q3_METHOD,
            "bits": Q3_BITS,
            "dim": Q3_DIM,
            "scoring_surface": Q3_SURFACE,
            "quantizer_seed": Q3_SEED,
            "candidate_count": q3_row.get("candidate_count"),
            "candidates_scored": q3_row.get("candidates_scored"),
            "pruning_supported": q3_row.get("pruning_supported"),
            "pruning_used": q3_row.get("pruning_used"),
        },
    }
    validate_row_legal_and_alignment(row, row["row_id"])
    return row


def select_promotions_and_guards(
    q3_rows: dict[str, dict[str, dict[str, Any]]],
    sources: dict[str, dict[str, Any]],
    excluded_qids: dict[str, set[str]],
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    selected_promotions: list[dict[str, Any]] = []
    selected_guards: list[dict[str, Any]] = []
    audit: dict[str, Any] = {"promotion_inventory": {}, "guard_inventory": {}, "rejected": {}}
    for dataset in ("fiqa", "nfcorpus", "scifact"):
        rejected = Counter()
        eligible = []
        guard_candidates = []
        source_path = next(path for path in args.q3_per_query_jsonl if dataset in path.name)
        for qid, row in sorted(q3_rows[dataset].items()):
            if qid in excluded_qids.get(dataset, set()):
                rejected["frozen_dev_reserve"] += 1
                continue
            if qid in sources[dataset]["test_qids"]:
                rejected["official_test"] += 1
                continue
            features = q3_gap_features(row, sources[dataset]["qrels"])
            if features["eligible_before_gap"] and not features["eligible"]:
                rejected["promotion_high_gap"] += 1
            if features["eligible"]:
                candidate_row = build_q3_row(
                    row,
                    qrels=sources[dataset]["qrels"],
                    queries=sources[dataset]["queries"],
                    corpus=sources[dataset]["corpus"],
                    source_path=source_path,
                    bucket=PROMOTION_BUCKET,
                    gap=float(features["promotion_score_gap"]),
                    main_weight=1.0,
                    recall_weight=0.0,
                )
                if candidate_row["q3_negative_count"] <= 0:
                    rejected["promotion_post_alias_no_q3_hard_negative"] += 1
                    continue
                eligible.append((float(features["promotion_score_gap"]), qid, row, features, candidate_row))
            elif features["guard"]:
                candidate_row = build_q3_row(
                    row,
                    qrels=sources[dataset]["qrels"],
                    queries=sources[dataset]["queries"],
                    corpus=sources[dataset]["corpus"],
                    source_path=source_path,
                    bucket=GUARD_BUCKET,
                    gap=float(features["guard_margin"]),
                    main_weight=0.0,
                    recall_weight=0.0,
                )
                if candidate_row["q3_negative_count"] <= 0:
                    rejected["guard_post_alias_no_q3_hard_negative"] += 1
                    continue
                guard_candidates.append((float(features["guard_margin"]), qid, row, features, candidate_row))
            else:
                rejected["not_promotion_or_guard"] += 1
        eligible.sort(key=lambda item: (item[0], item[1]))
        if dataset == "fiqa":
            chosen = eligible
            if len(chosen) != args.fiqa_promotion_count:
                raise ValueError(f"fiqa eligible promotions={len(chosen)}, expected {args.fiqa_promotion_count}")
        elif dataset == "nfcorpus":
            if len(eligible) != args.expected_nfcorpus_eligible:
                raise ValueError(f"nfcorpus eligible inventory={len(eligible)}, expected {args.expected_nfcorpus_eligible}")
            chosen = eligible[: args.nfcorpus_promotion_count]
        else:
            chosen = [item for item in eligible if item[1] == args.scifact_required_qid]
            if len(chosen) != 1 or len(eligible) != args.expected_scifact_eligible:
                raise ValueError(f"scifact eligible={[(item[1], item[0]) for item in eligible]}, expected only {args.scifact_required_qid}")
        for _gap, _qid, _row, _features, candidate_row in chosen:
            selected_promotions.append(candidate_row)
        chosen_qids = {qid for _gap, qid, _row, _features, _candidate_row in chosen}
        guard_candidates = [item for item in guard_candidates if item[1] not in chosen_qids]
        if dataset in ("fiqa", "scifact"):
            guard_candidates.sort(key=lambda item: (item[0], item[1]))
            if len(guard_candidates) < args.guard_count_per_dataset:
                raise ValueError(f"{dataset} guard candidates={len(guard_candidates)}, expected at least {args.guard_count_per_dataset}")
            for _margin, _qid, _row, _features, candidate_row in guard_candidates[: args.guard_count_per_dataset]:
                selected_guards.append(candidate_row)
        audit["promotion_inventory"][dataset] = {
            "eligible": len(eligible),
            "selected": len(chosen),
            "selected_qids": [qid for _gap, qid, _row, _features, _candidate_row in chosen],
            "selected_gap_min": min((gap for gap, _qid, _row, _features, _candidate_row in chosen), default=None),
            "selected_gap_max": max((gap for gap, _qid, _row, _features, _candidate_row in chosen), default=None),
            "eligible_inventory_sha256": sha256_strings([json.dumps({"gap": gap, "qid": qid}, sort_keys=True, separators=(",", ":")) for gap, qid, _row, _features, _candidate_row in eligible]),
        }
        chosen_guards = [row for row in selected_guards if row["source_dataset"] == dataset]
        audit["guard_inventory"][dataset] = {
            "available": len(guard_candidates),
            "selected": len(chosen_guards),
            "selected_qids": [row["source_query_id"] for row in chosen_guards],
            "selected_margins": [row["guard_margin"] for row in chosen_guards],
        }
        audit["rejected"][dataset] = dict(rejected)
    if len(selected_promotions) != args.expected_promotion_total:
        raise ValueError(f"promotion rows={len(selected_promotions)}, expected {args.expected_promotion_total}")
    if len(selected_guards) != args.expected_guard_total:
        raise ValueError(f"guard rows={len(selected_guards)}, expected {args.expected_guard_total}")
    return selected_promotions + selected_guards, audit


def load_parent_sentinels(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    actual = sha256_file(args.parent_sentinel_train_jsonl)
    if actual != args.expected_parent_sentinel_train_sha256:
        raise ValueError(f"parent sentinel train sha mismatch: expected={args.expected_parent_sentinel_train_sha256} actual={actual}")
    rows = [row for _line, row in iter_jsonl(args.parent_sentinel_train_jsonl)]
    sentinels = [row for row in rows if row.get("selection_bucket") == SENTINEL_BUCKET]
    if len(sentinels) != args.sentinel_count:
        raise ValueError(f"parent sentinels={len(sentinels)}, expected {args.sentinel_count}")
    required = {"PLAIN-63": {"MED-5329", "MED-4523"}}
    for qid, docs in required.items():
        matching = [row for row in sentinels if row["source_query_id"] == qid]
        if len(matching) != 1:
            raise ValueError(f"missing required sentinel qid {qid}")
        positives = {doc_id_for_dataset("nfcorpus", matching[0]["candidate_doc_ids"][index]) for index in matching[0]["positive_indexes"]}
        missing = sorted(docs - positives)
        if missing:
            raise ValueError(f"{qid}: missing required sentinel positives {missing}")
    copied = []
    for row in sentinels:
        new_row = dict(row)
        new_row["r4_parent_sentinel_provenance"] = {
            "parent_train_jsonl": str(args.parent_sentinel_train_jsonl),
            "parent_train_sha256": actual,
            "parent_row_sha256": sha256_json(row),
        }
        validate_row_legal_and_alignment(new_row, new_row["row_id"])
        copied.append(new_row)
    return copied, {
        "parent_train_sha256": actual,
        "parent_manifest_sha256": sha256_file(args.parent_sentinel_manifest),
        "parent_coverage_sha256": sha256_file(args.parent_sentinel_coverage),
        "selected_qids": [row["source_query_id"] for row in copied],
        "required_plain63": True,
    }


def overlap_report(rows: list[dict[str, Any]], excluded_qids: dict[str, set[str]], sources: dict[str, dict[str, Any]]) -> dict[str, Any]:
    selected: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        selected[row["source_dataset"]].add(str(row["source_query_id"]))
    report: dict[str, Any] = {}
    for dataset, qids in sorted(selected.items()):
        frozen = sorted(qids & excluded_qids.get(dataset, set()))
        test = sorted(qids & sources[dataset]["test_qids"])
        if frozen or test:
            raise ValueError(f"{dataset}: selected qids overlap exclusions frozen={frozen} test={test}")
        report[dataset] = {
            "selected_count": len(qids),
            "selected_x_frozen_dev_reserve": len(frozen),
            "selected_x_frozen_dev_reserve_qids": frozen,
            "selected_x_official_test": len(test),
            "selected_x_official_test_qids": test,
        }
    return report


def validate_output_rows(rows: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    bucket_counts = Counter()
    dataset_counts = Counter()
    main_weight_counts = Counter()
    recall_weight_counts = Counter()
    candidate_count_counts = Counter()
    raw = 0
    canonical = 0
    main = 0
    recall = 0
    for row in rows:
        validate_row_legal_and_alignment(row, row["row_id"])
        if row.get("legal_gates") != LEGAL_GATES or row.get("release_train_allowed") is not False or row.get("commercial_use_allowed") is not False:
            raise ValueError(f"{row['row_id']}: legal gates are not fail-closed")
        positives = [index for index, gain in enumerate(row["qrel_gains"]) if float(gain) > 0.0]
        if positives != list(row["positive_indexes"]):
            raise ValueError(f"{row['row_id']}: positive indexes do not match gains")
        positive_texts = {normalize_candidate_text(row["candidate_texts"][index]) for index in positives}
        hard_count = sum(1 for value in row["hard_negative_eligible"] if value)
        if row["selection_bucket"] in {PROMOTION_BUCKET, GUARD_BUCKET} and hard_count <= 0:
            raise ValueError(f"{row['row_id']}: {row['selection_bucket']} requires at least one post-alias hard negative")
        for index, hard in enumerate(row["hard_negative_eligible"]):
            if hard and normalize_candidate_text(row["candidate_texts"][index]) in positive_texts:
                raise ValueError(f"{row['row_id']}: normalized-text positive/hard conflict")
        row_main_pairs = eligible_topk_pairs(row, False, args.turboquant_topk_negative_mask)
        if float(row.get("turboquant_topk_loss_weight", 0.0)) > 0.0 and row_main_pairs <= 0:
            raise ValueError(f"{row['row_id']}: weighted top-k row has no eligible q3 pairs")
        row_recall_pairs = eligible_topk_pairs(row, True, args.turboquant_topk_recall_negative_mask)
        if float(row.get("turboquant_topk_recall_loss_weight", 0.0)) > 0.0 and row_recall_pairs <= 0:
            raise ValueError(f"{row['row_id']}: weighted recall row has no eligible q3 pairs")
        bucket_counts[row["selection_bucket"]] += 1
        dataset_counts[row["source_dataset"]] += 1
        main_weight_counts[str(float(row.get("turboquant_topk_loss_weight", 0.0)))] += 1
        recall_weight_counts[str(float(row.get("turboquant_topk_recall_loss_weight", 0.0)))] += 1
        candidate_count_counts[str(len(row["candidate_doc_ids"]))] += 1
        raw += len(row["candidate_doc_ids"])
        canonical += canonical_stats(row, row["row_id"])["canonical_candidates"]
        main += row_main_pairs
        recall += row_recall_pairs
    if bucket_counts[PROMOTION_BUCKET] != args.expected_promotion_total or bucket_counts[GUARD_BUCKET] != args.expected_guard_total or bucket_counts[SENTINEL_BUCKET] != args.sentinel_count:
        raise ValueError(f"wrong bucket counts: {dict(bucket_counts)}")
    steps = math.ceil(len(rows) / args.batch_size)
    if len(rows) != args.expected_rows or steps != args.expected_steps_per_epoch:
        raise ValueError(f"rows/steps mismatch rows={len(rows)} steps={steps}")
    total_epoch = canonical + args.turboquant_objectives * (main + recall)
    return {
        "dataset_counts": dict(dataset_counts),
        "bucket_counts": dict(bucket_counts),
        "turboquant_topk_loss_weight_counts": dict(sorted(main_weight_counts.items())),
        "turboquant_topk_recall_loss_weight_counts": dict(sorted(recall_weight_counts.items())),
        "candidate_count_counts": dict(sorted(candidate_count_counts.items(), key=lambda item: int(item[0]))),
        "raw_candidates_per_epoch": raw,
        "canonical_candidates_per_epoch": canonical,
        "main_q3_pairs_per_epoch": main,
        "recall_q3_pairs_per_epoch": recall,
        "total_train_pairs_per_epoch": total_epoch,
        "planned_pairs": total_epoch * args.epochs,
        "steps_per_epoch": steps,
        "epochs": args.epochs,
        "batch": args.batch_size,
        "alignment_checks": {
            "aligned_candidate_fields": True,
            "positive_indexes_match_qrel_gains": True,
            "legal_gates_fail_closed": True,
            "normalized_text_positive_hard_conflict_free": True,
        },
    }


def main() -> None:
    args = parse_args()
    if args.promotion_max_score_gap != 0.05:
        raise ValueError("R4 descriptor requires promotion-max-score-gap exactly 0.05")
    q3_rows, q3_identities = load_q3_rows(args.q3_per_query_jsonl, expected_q3_hashes(args))
    sources = {dataset: load_domain_sources(args.dataset_root, dataset) for dataset in ("fiqa", "nfcorpus", "scifact")}
    excluded_qids = read_excluded_qids(args.exclude_qids_json)
    selected, selection_audit = select_promotions_and_guards(q3_rows, sources, excluded_qids, args)
    sentinels, sentinel_audit = load_parent_sentinels(args)
    output_rows = selected + sentinels
    if len({row["row_id"] for row in output_rows}) != len(output_rows):
        raise ValueError("duplicate output row_id")
    output_rows.sort(key=lambda row: (row["selection_bucket"], row["source_dataset"], str(row["source_query_id"]), row["row_id"]))
    overlap = overlap_report(output_rows, excluded_qids, sources)
    validation = validate_output_rows(output_rows, args)
    write_jsonl(args.output_jsonl, output_rows)
    train_sha = sha256_file(args.output_jsonl)
    anchor_hashes = hash_package_siblings(args.anchor_package)
    source_hashes = {
        "q3_per_query_jsonl": q3_identities,
        "exclude_qids_json": {str(path): {"sha256": sha256_file(path)} for path in args.exclude_qids_json},
        "raw_sources": {
            dataset: {
                name: {"path": str(path), "sha256": sha256_file(path)}
                for name, path in sources[dataset]["paths"].items()
            }
            for dataset in ("fiqa", "nfcorpus", "scifact")
        },
        "builder": {"path": "scripts/build_q3_r4_broad_gap_dataset.py", "sha256": sha256_file(Path(__file__).resolve())},
        "test_file": {"path": str(args.test_file), "sha256": sha256_file(args.test_file) if args.test_file.is_file() else None},
    }
    parent_hashes = {
        "sentinel_train_jsonl": str(args.parent_sentinel_train_jsonl),
        "sentinel_train_sha256": sentinel_audit["parent_train_sha256"],
        "sentinel_manifest": str(args.parent_sentinel_manifest),
        "sentinel_manifest_sha256": sentinel_audit["parent_manifest_sha256"],
        "sentinel_coverage": str(args.parent_sentinel_coverage),
        "sentinel_coverage_sha256": sentinel_audit["parent_coverage_sha256"],
    }
    artifact_hashes = {
        "train_score_spectrum": train_sha,
        "manifest": {"path": str(args.manifest), "sha256_recorded_externally_after_write": True},
        "coverage_report": {"path": str(args.coverage_report), "sha256_recorded_in_manifest_after_write": True},
    }
    contract_hashes = {
        "anchor": anchor_hashes,
        "artifact_hashes": artifact_hashes,
        "source_hashes": source_hashes,
        "parent_hashes": parent_hashes,
    }
    plan_contract = go_loader_plan_contract(args, {"rows": len(output_rows), **validation}, contract_hashes)
    coverage = {
        "schema": f"{SCHEMA}.coverage.v1",
        "created_utc": utc_stamp(),
        "selection_audit": selection_audit,
        "sentinel_audit": sentinel_audit,
        "overlap": overlap,
        "validation": validation,
        "go_loader_plan": plan_contract,
        "selected_rows": [
            {
                "row_id": row["row_id"],
                "dataset": row["source_dataset"],
                "query_id": row["source_query_id"],
                "selection_bucket": row["selection_bucket"],
                "guard_margin": row.get("guard_margin"),
                "turboquant_topk_loss_weight": row.get("turboquant_topk_loss_weight"),
                "turboquant_topk_recall_loss_weight": row.get("turboquant_topk_recall_loss_weight"),
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
    coverage_sha = sha256_file(args.coverage_report)
    artifact_hashes["coverage_report"] = {"path": str(args.coverage_report), "sha256": coverage_sha}
    plan_contract = go_loader_plan_contract(args, {"rows": len(output_rows), **validation}, contract_hashes)
    manifest = {
        "schema": SCHEMA,
        "created_utc": utc_stamp(),
        "builder": "scripts/build_q3_r4_broad_gap_dataset.py",
        "inputs": {
            "dataset_root": str(args.dataset_root),
            "q3_per_query_jsonl": q3_identities,
            "anchor_package": anchor_hashes,
            "parent_hashes": parent_hashes,
            "exclude_qids_json": source_hashes["exclude_qids_json"],
            "raw_sources": source_hashes["raw_sources"],
            "test_file": source_hashes["test_file"],
        },
        "outputs": {
            "train_score_spectrum": str(args.output_jsonl),
            "manifest": str(args.manifest),
            "coverage_report": str(args.coverage_report),
        },
        "policy": {
            "promotion": "train qid is eligible only after frozen dev/reserve and official test exclusion, with >=1 train-qrel positive at q3 rank 11..20, >=1 nonrelevant q3 rank 1..10, and min(top10 nonrel score - eligible positive score) <= 0.05",
            "promotion_selection": "all 21 eligible FiQA, 64 smallest-gap eligible NFCorpus from post-alias inventory 235, and exactly SciFact qid 243",
            "guard": "FiQA and SciFact only, 12 each, from nonpromotion qids by smallest vulnerable top10 positive/nonrel q3 margin",
            "sentinels": "reuse the 24 verified duplicate-safe NFCorpus rank80..100 recall sentinel rows from the 69-row dualcut parent",
            "candidates": "promotion and guard rows keep train-qrel positives at q3 rank 1..20 and nonrelevant q3 negatives at rank 1..10",
            "legal": "research-only rows with release/commercial gates fail-closed; no quality claim",
        },
        "config": {
            "promotion_max_score_gap": args.promotion_max_score_gap,
            "fiqa_promotion_count": args.fiqa_promotion_count,
            "nfcorpus_promotion_count": args.nfcorpus_promotion_count,
            "scifact_required_qid": args.scifact_required_qid,
            "guard_count_per_dataset": args.guard_count_per_dataset,
            "sentinel_count": args.sentinel_count,
            "expected_rows": args.expected_rows,
            "q3_method": Q3_METHOD,
            "q3_bits": Q3_BITS,
            "q3_dim": Q3_DIM,
            "q3_quantizer_seed": Q3_SEED,
            "q3_scoring_surface": Q3_SURFACE,
        },
        "counts": {"rows": len(output_rows), **validation},
        "selection_audit": selection_audit,
        "sentinel_audit": sentinel_audit,
        "overlap": overlap,
        "workload_contract": validation,
        "go_loader_plan": plan_contract,
        "legal_gates": dict(LEGAL_GATES),
        **LEGAL_GATES,
        "quality_claim": False,
        "sha256": {
            "builder": sha256_file(Path(__file__).resolve()),
            "train_score_spectrum": train_sha,
            "coverage_report": coverage_sha,
        },
    }
    write_json(args.manifest, manifest)
    print(f"q3 R4 broad-gap dataset rows={len(output_rows)} train_sha256={train_sha}")
    print(f"promotions=86 guards=24 sentinels=24 steps_per_epoch={validation['steps_per_epoch']}")
    print(f"workload planned_pairs={validation['planned_pairs']} main_q3_pairs={validation['main_q3_pairs_per_epoch']} recall_q3_pairs={validation['recall_q3_pairs_per_epoch']}")


if __name__ == "__main__":
    main()
