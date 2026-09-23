#!/usr/bin/env python3
"""Build the q3 R5 broad-focus train-only dataset and plan contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import build_q3_r4_broad_gap_dataset as r4  # noqa: E402
import q3_r5_training_launcher as launcher  # noqa: E402
from build_q3_dualcut_recall_dataset import build_sentinel_row, validate_row_legal_and_alignment  # noqa: E402
from build_q3_r3_incumbent_dataset import canonical_stats, eligible_topk_pairs  # noqa: E402


SCHEMA = "eos.q3_r5_broadfocus_dataset.v1"
ARM_ID = "arm-q3r5-broadfocus-fiqa33-nf74-sf13-nfrecall74-rw050-q3w004-m002-lr2e6-e2"
SUBTREE = "headroom-q3r5-broadfocus-fiqa33-nf74-sf13-nfrecall74-rw050"
LEGAL_GATES = dict(r4.LEGAL_GATES)
PROMOTION_BUCKET = r4.PROMOTION_BUCKET
GUARD_BUCKET = r4.GUARD_BUCKET
FOCUS_BUCKET = "broadfocus_closest_pair_q3"
SENTINEL_BUCKET = r4.SENTINEL_BUCKET
ANCHOR_PACKAGE = r4.ANCHOR_PACKAGE
ACTIVE_ROOT = Path("runs/eos-d384-q3-boundary-goal-v1-20260902T084144Z")
CANDIDATE_PACKAGE = ACTIVE_ROOT / "packages" / ARM_ID / "d384-pre.mll"
METRICS_TARGET = ACTIVE_ROOT / "metrics" / f"{ARM_ID}.train.metrics.json"
DEFAULT_R4_TRAIN = ACTIVE_ROOT / "headroom-q3broad-gap050-dualcut-nf64-fiqa21-sf1-nfrecall24/data/beir134.q3broad-gap050-dualcut-nf64-fiqa21-sf1-nfrecall24.train.jsonl"
DEFAULT_R4_MANIFEST = ACTIVE_ROOT / "headroom-q3broad-gap050-dualcut-nf64-fiqa21-sf1-nfrecall24/data/beir134.q3broad-gap050-dualcut-nf64-fiqa21-sf1-nfrecall24.manifest.json"
DEFAULT_R4_COVERAGE = ACTIVE_ROOT / "headroom-q3broad-gap050-dualcut-nf64-fiqa21-sf1-nfrecall24/q3broad-gap050-dualcut-nf64-fiqa21-sf1-nfrecall24.coverage.json"
EXPECTED_R4_TRAIN_SHA256 = "201a9093a827eba809a05e072520dc02e3ae0bd5ba7d1c4953e6c58e8ae84a03"
EXPECTED_WORKLOAD = {
    "train": 294,
    "batch": 4,
    "steps_per_epoch": 74,
    "raw_candidates_per_epoch": 9035,
    "canonical_candidates_per_epoch": 8850,
    "main_q3_pairs_per_epoch": 2811,
    "recall_q3_pairs_per_epoch": 115808,
    "total_train_pairs_per_epoch": 127469,
    "planned_pairs": 254938,
    "actual_train_pairs": 0,
}
EXPECTED_BUCKETS = {
    PROMOTION_BUCKET: 86,
    GUARD_BUCKET: 24,
    FOCUS_BUCKET: 110,
    SENTINEL_BUCKET: 74,
}
EXPECTED_DOMAIN_QIDS = {"fiqa": 33, "nfcorpus": 74, "scifact": 13}
EXPECTED_FOCUS_SOURCE = 110
EXPECTED_NF_UNION = 74
EXPECTED_NF_OVERLAP = 14
EXPECTED_NF_RANK80_100 = 57


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--q3-per-query-jsonl", action="append", required=True, type=Path)
    parser.add_argument("--r4-train-jsonl", type=Path, default=DEFAULT_R4_TRAIN)
    parser.add_argument("--r4-manifest", type=Path, default=DEFAULT_R4_MANIFEST)
    parser.add_argument("--r4-coverage", type=Path, default=DEFAULT_R4_COVERAGE)
    parser.add_argument("--expected-r4-train-sha256", default=EXPECTED_R4_TRAIN_SHA256)
    parser.add_argument("--anchor-package", type=Path, default=ANCHOR_PACKAGE)
    parser.add_argument("--candidate-package", type=Path, default=CANDIDATE_PACKAGE)
    parser.add_argument("--metrics-target", type=Path, default=METRICS_TARGET)
    parser.add_argument("--exclude-qids-json", action="append", default=[], type=Path)
    parser.add_argument("--output-jsonl", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--coverage-report", required=True, type=Path)
    parser.add_argument("--clone-inventory", required=True, type=Path)
    parser.add_argument("--plan-launch-audit", required=True, type=Path)
    parser.add_argument("--plan-stdout", required=True, type=Path)
    parser.add_argument("--plan-stderr", required=True, type=Path)
    parser.add_argument("--plan-preflight-audit", required=True, type=Path)
    parser.add_argument("--launcher-stdout-capture", required=True, type=Path)
    parser.add_argument("--launcher-stderr-capture", required=True, type=Path)
    parser.add_argument("--eos-binary", type=Path, default=ACTIVE_ROOT / "bin/eos-r4-broad-gap-postfix")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--expected-rows", type=int, default=294)
    parser.add_argument("--expected-steps-per-epoch", type=int, default=74)
    parser.add_argument("--expected-q3-sha256", action="append", default=[])
    parser.add_argument("--test-file", default=Path("scripts/test_build_q3_r5_broadfocus_dataset.py"), type=Path)
    return parser.parse_args()


def utc_stamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def repo_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def display_path(path: Path) -> str:
    try:
        return str(repo_path(path).resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(repo_path(path).resolve())


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


def iter_jsonl(path: Path):
    with repo_path(path).open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            raw = raw.strip()
            if raw:
                yield line_number, json.loads(raw)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    target = repo_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")


def copy_row(row: dict[str, Any]) -> dict[str, Any]:
    copied = json.loads(json.dumps(row))
    copied["r5_parent_r4_row_provenance"] = {
        "parent_row_id": row["row_id"],
        "parent_row_sha256": sha256_json(row),
        "parent_selection_bucket": row["selection_bucket"],
    }
    copied["train_policy"] = "q3_r5_broadfocus_retain_r4_broad_row_research_only"
    copied["legal_gates"] = dict(LEGAL_GATES)
    copied.update(LEGAL_GATES)
    validate_row_legal_and_alignment(copied, copied["row_id"])
    return copied


def sorted_candidate_pairs(q3_row: dict[str, Any], gains: dict[str, int], bucket: str) -> list[tuple[dict[str, Any], dict[str, Any], float]]:
    positives = []
    negatives = []
    for candidate in q3_row["top_k"]:
        rank = int(candidate["rank"])
        doc_id = str(candidate["doc_id"])
        gain = gains.get(doc_id, 0)
        item = {"candidate": candidate, "rank": rank, "doc_id": doc_id, "score": float(candidate["score"]), "gain": gain}
        if bucket == PROMOTION_BUCKET and 11 <= rank <= 20 and gain > 0:
            positives.append(item)
        elif bucket == GUARD_BUCKET and 1 <= rank <= 10 and gain > 0:
            positives.append(item)
        if 1 <= rank <= 10 and gain <= 0:
            negatives.append(item)
    if not positives or not negatives:
        raise ValueError(f"{q3_row['dataset']}:{q3_row['query_id']}: cannot build focus row for {bucket}")
    if bucket == PROMOTION_BUCKET:
        pairs = [(neg["score"] - pos["score"], pos, neg) for pos in positives for neg in negatives]
    elif bucket == GUARD_BUCKET:
        pairs = [(pos["score"] - neg["score"], pos, neg) for pos in positives for neg in negatives]
    else:
        raise ValueError(f"unexpected focus source bucket {bucket}")
    return [
        (pos["candidate"], neg["candidate"], float(gap))
        for gap, pos, neg in sorted(
            pairs,
            key=lambda item: (abs(item[0]), item[0], item[1]["rank"], item[2]["rank"], item[1]["doc_id"], item[2]["doc_id"]),
        )
    ]


def build_focus_row(
    parent: dict[str, Any],
    q3_row: dict[str, Any],
    sources: dict[str, Any],
    source_path: Path,
) -> dict[str, Any]:
    dataset = str(parent["source_dataset"])
    qid = str(parent["source_query_id"])
    gains = sources["qrels"][qid]
    selected: tuple[list[str], list[str], list[str], list[float], list[bool], list[dict[str, Any]], float] | None = None
    for pos, neg, gap in sorted_candidate_pairs(q3_row, gains, str(parent["selection_bucket"])):
        candidate_doc_ids = []
        candidate_texts = []
        candidate_sources = []
        qrel_gains = []
        evidence = []
        for candidate in (pos, neg):
            doc_id = str(candidate["doc_id"])
            gain = gains.get(doc_id, 0)
            candidate_doc_ids.append(f"{dataset}:{doc_id}")
            candidate_texts.append(sources["corpus"][doc_id])
            candidate_sources.append("qrel" if gain > 0 else "q3")
            qrel_gains.append(float(gain))
            evidence.append(r4.candidate_evidence(candidate, source_path))
        positive_indexes = [index for index, gain in enumerate(qrel_gains) if gain > 0.0]
        hard_negative_eligible = [gain <= 0.0 for gain in qrel_gains]
        positive_texts = {r4.normalize_candidate_text(candidate_texts[index]) for index in positive_indexes}
        for index, text in enumerate(candidate_texts):
            if hard_negative_eligible[index] and r4.normalize_candidate_text(text) in positive_texts:
                hard_negative_eligible[index] = False
        if len(candidate_doc_ids) == 2 and len(positive_indexes) == 1 and sum(hard_negative_eligible) == 1:
            selected = (candidate_doc_ids, candidate_texts, candidate_sources, qrel_gains, hard_negative_eligible, evidence, gap)
            break
    if selected is None:
        raise ValueError(f"{dataset}:{qid}: no post-alias-safe focus pair")
    candidate_doc_ids, candidate_texts, candidate_sources, qrel_gains, hard_negative_eligible, evidence, gap = selected
    positive_indexes = [index for index, gain in enumerate(qrel_gains) if gain > 0.0]
    row = {
        "row_id": f"{dataset}:{qid}:r5-broadfocus-{parent['selection_bucket']}",
        "source": f"{dataset}:beir-train:q3-r5-broadfocus-v1",
        "query": sources["queries"][qid],
        "candidate_doc_ids": candidate_doc_ids,
        "candidate_texts": candidate_texts,
        "candidate_sources": candidate_sources,
        "qrel_gains": qrel_gains,
        "positive_indexes": positive_indexes,
        "selected_positive_index": positive_indexes[0],
        "hard_negative_eligible": hard_negative_eligible,
        "target_probabilities": [1.0 if index in positive_indexes else 0.0 for index in range(2)],
        "hard_loss_weight": 0.0,
        "soft_loss_weight": 0.0,
        "recovery_loss_weight": 0.0,
        "turboquant_topk_loss_weight": 1.0,
        "turboquant_topk_recall_loss_weight": 0.0,
        "train_policy": "q3_r5_broadfocus_closest_pair_research_only",
        "selection_bucket": FOCUS_BUCKET,
        "focus_source_bucket": parent["selection_bucket"],
        "focus_gap_or_margin": gap,
        "legal_gates": dict(LEGAL_GATES),
        **LEGAL_GATES,
        "source_artifact_hash": sha256_json({"q3_query": q3_row, "parent_row_sha256": sha256_json(parent)}),
        "source_dataset": dataset,
        "source_query_id": qid,
        "source_positive_doc_ids": [r4.doc_id_for_dataset(dataset, candidate_doc_ids[index]) for index in positive_indexes],
        "candidate_positive_doc_ids_pre_dedupe": [r4.doc_id_for_dataset(dataset, candidate_doc_ids[index]) for index in positive_indexes],
        "all_train_qrel_positive_doc_ids": sorted(gains),
        "positive_count": 1,
        "bm25_negative_count": 0,
        "q3_negative_count": 1,
        "q3_positive_text_alias_count": 0,
        "q3_positive_text_alias_candidate_indexes": [],
        "extra_negative_count": 0,
        "q3_evidence": [evidence[index] for index, value in enumerate(hard_negative_eligible) if value],
        "anchor_q3_candidate_evidence": evidence,
        "r5_focus_parent_provenance": {
            "parent_row_id": parent["row_id"],
            "parent_row_sha256": sha256_json(parent),
            "prospective_rule": "closest valid q3 pair from original-anchor train-side q3 rows only",
        },
        "q3_scoring_metadata": {
            "method": r4.Q3_METHOD,
            "bits": r4.Q3_BITS,
            "dim": r4.Q3_DIM,
            "scoring_surface": r4.Q3_SURFACE,
            "quantizer_seed": r4.Q3_SEED,
            "candidate_count": q3_row.get("candidate_count"),
            "candidates_scored": q3_row.get("candidates_scored"),
            "pruning_supported": q3_row.get("pruning_supported"),
            "pruning_used": q3_row.get("pruning_used"),
        },
    }
    validate_row_legal_and_alignment(row, row["row_id"])
    if eligible_topk_pairs(row, False, "q3") != 1:
        raise ValueError(f"{row['row_id']}: focus row must produce exactly one main q3 pair")
    return row


def nf_rank80_100_hit(q3_row: dict[str, Any], qrels: dict[str, dict[str, int]]) -> bool:
    gains = qrels.get(str(q3_row["query_id"]), {})
    return any(80 <= int(candidate["rank"]) <= 100 and gains.get(str(candidate["doc_id"]), 0) > 0 for candidate in q3_row["top_k"])


def build_nf_recall_rows(
    qids: list[str],
    q3_rows: dict[str, dict[str, dict[str, Any]]],
    sources: dict[str, dict[str, Any]],
    q3_source_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = []
    rank80_100 = []
    for qid in qids:
        q3_row = q3_rows["nfcorpus"][qid]
        hits = []
        for candidate in q3_row["top_k"]:
            rank = int(candidate["rank"])
            doc_id = str(candidate["doc_id"])
            gain = sources["nfcorpus"]["qrels"].get(qid, {}).get(doc_id, 0)
            if 80 <= rank <= 100 and gain > 0:
                hits.append({"rank": rank, "doc_id": doc_id, "gain": gain})
        row = build_sentinel_row(q3_row, hits, sources["nfcorpus"]["qrels"], sources["nfcorpus"]["queries"], sources["nfcorpus"]["corpus"], q3_source_path)
        row["row_id"] = f"nfcorpus:{qid}:r5-nfrecall74-q3top100"
        row["source"] = "nfcorpus:beir-train:q3-r5-nfrecall74-v1"
        row["train_policy"] = "q3_r5_broadfocus_nfrecall74_research_only"
        row["turboquant_topk_loss_weight"] = 0.0
        row["turboquant_topk_recall_loss_weight"] = 1.0
        row["r5_nf_recall_provenance"] = {
            "source_union": "R4 64 selected NFCorpus promotions union R4 24 NFCorpus recall sentinels",
            "anchor_rank80_100_positive": bool(hits),
            "anchor_rank80_100_hits": hits,
        }
        validate_row_legal_and_alignment(row, row["row_id"])
        if eligible_topk_pairs(row, True, "q3") <= 0:
            raise ValueError(f"{row['row_id']}: recall row has zero post-alias q3 recall pairs")
        rows.append(row)
        if hits:
            rank80_100.append(qid)
    if len(rows) != EXPECTED_NF_UNION:
        raise ValueError(f"nf recall rows={len(rows)}, expected {EXPECTED_NF_UNION}")
    if len(rank80_100) != EXPECTED_NF_RANK80_100:
        raise ValueError(f"nf rank80_100 positives={len(rank80_100)}, expected {EXPECTED_NF_RANK80_100}")
    return rows, {"rank80_100_positive_count": len(rank80_100), "rank80_100_positive_qids": rank80_100}


def validate_rows(rows: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    bucket_counts = Counter(row["selection_bucket"] for row in rows)
    if dict(bucket_counts) != EXPECTED_BUCKETS:
        raise ValueError(f"bucket drift: {dict(bucket_counts)} != {EXPECTED_BUCKETS}")
    qids: dict[str, set[str]] = defaultdict(set)
    raw = canonical = main = recall = 0
    focus_rows = 0
    focus_pairs = 0
    zero_hard = []
    for row in rows:
        validate_row_legal_and_alignment(row, row["row_id"])
        if row["legal_gates"] != LEGAL_GATES or row["release_train_allowed"] is not False or row["commercial_use_allowed"] is not False:
            raise ValueError(f"{row['row_id']}: legal gates drift")
        if row["selection_bucket"] == FOCUS_BUCKET:
            focus_rows += 1
            if len(row["candidate_doc_ids"]) != 2 or row["positive_count"] != 1 or row["q3_negative_count"] != 1:
                raise ValueError(f"{row['row_id']}: focus shape drift")
        hard_count = sum(1 for value in row["hard_negative_eligible"] if value)
        if hard_count <= 0:
            zero_hard.append(row["row_id"])
        raw += len(row["candidate_doc_ids"])
        canonical += canonical_stats(row, row["row_id"])["canonical_candidates"]
        row_main = eligible_topk_pairs(row, False, "q3")
        row_recall = eligible_topk_pairs(row, True, "q3")
        main += row_main
        recall += row_recall
        if row["selection_bucket"] == FOCUS_BUCKET:
            focus_pairs += row_main
        qids[row["source_dataset"]].add(str(row["source_query_id"]))
    if zero_hard:
        raise ValueError(f"zero post-alias hard-negative rows: {zero_hard[:5]}")
    domain_counts = {dataset: len(values) for dataset, values in qids.items()}
    if domain_counts != EXPECTED_DOMAIN_QIDS:
        raise ValueError(f"domain qid union drift: {domain_counts} != {EXPECTED_DOMAIN_QIDS}")
    steps = math.ceil(len(rows) / args.batch_size)
    workload = {
        "train": len(rows),
        "batch": args.batch_size,
        "steps_per_epoch": steps,
        "raw_candidates_per_epoch": raw,
        "canonical_candidates_per_epoch": canonical,
        "main_q3_pairs_per_epoch": main,
        "recall_q3_pairs_per_epoch": recall,
        "total_train_pairs_per_epoch": canonical + main + recall,
        "planned_pairs": (canonical + main + recall) * args.epochs,
        "actual_train_pairs": 0,
    }
    if workload != EXPECTED_WORKLOAD:
        raise ValueError(f"workload drift: {workload} != {EXPECTED_WORKLOAD}")
    return {
        "dataset_qid_counts": domain_counts,
        "bucket_counts": dict(bucket_counts),
        "candidate_count_counts": dict(Counter(str(len(row["candidate_doc_ids"])) for row in rows)),
        "turboquant_topk_loss_weight_counts": dict(Counter(str(float(row.get("turboquant_topk_loss_weight", 0.0))) for row in rows)),
        "turboquant_topk_recall_loss_weight_counts": dict(Counter(str(float(row.get("turboquant_topk_recall_loss_weight", 0.0))) for row in rows)),
        "focus_rows": focus_rows,
        "focus_main_q3_pairs": focus_pairs,
        "workload": workload,
        "effective_scales": {
            "denominator_formula": "1 + 0.04*main_row_weight + 0.04*0.50*recall_row_weight",
            "main_only": 0.04 / 1.04,
            "recall_only": 0.02 / 1.02,
        },
    }


def clone_candidate(anchor_package: Path, candidate_package: Path, out_path: Path) -> dict[str, Any]:
    anchor = repo_path(anchor_package)
    candidate = repo_path(candidate_package)
    if candidate.parent.exists():
        raise ValueError(f"candidate dir already exists, refusing to overwrite: {display_path(candidate.parent)}")
    candidate.parent.mkdir(parents=True)
    stem = anchor.name[:-4] if anchor.name.endswith(".mll") else anchor.stem
    copied = []
    for source in sorted(anchor.parent.glob(f"{stem}*")):
        if not source.is_file():
            continue
        target = candidate.parent / source.name
        shutil.copy2(source, target)
        copied.append({"source": display_path(source), "target": display_path(target), "sha256": sha256_file(target), "bytes": target.stat().st_size})
    if len(copied) != 9:
        raise ValueError(f"copied sibling count={len(copied)}, expected 9")
    anchor_state = launcher.package_state(anchor)
    candidate_state = launcher.package_state(candidate)
    if candidate_state["content_rollup_sha256"] != anchor_state["content_rollup_sha256"]:
        raise ValueError("candidate clone rollup differs from anchor")
    audit = {
        "schema": "eos.q3_r5_broadfocus.candidate_clone_copy_inventory.v1",
        "created_utc": utc_stamp(),
        "candidate_dir_was_absent": True,
        "copied": copied,
        "anchor": anchor_state,
        "candidate": candidate_state,
        "clone_matches_anchor": True,
    }
    write_json(out_path, audit)
    return audit


def train_argv(args: argparse.Namespace, *, plan_only: bool) -> list[str]:
    tokens = [display_path(args.eos_binary), "train-embed"]
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
            "--metrics-json", display_path(args.metrics_target),
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
            "--turboquant-topk-recall-weight", "0.50",
            "--turboquant-topk-recall-cutoff", "100",
            "--turboquant-topk-recall-tau", "0.05",
            "--turboquant-topk-recall-margin", "0",
            "--turboquant-topk-recall-negative-mask", "q3",
            "--turboquant-prefix-score-mode", "prepared_ip",
            "--turboquant-prefix-seed", str(r4.Q3_SEED),
            display_path(args.candidate_package),
            display_path(args.output_jsonl),
        ]
    )
    return tokens


def run_plan_and_audit(args: argparse.Namespace, validation: dict[str, Any], clone_audit: dict[str, Any]) -> dict[str, Any]:
    plan_argv = train_argv(args, plan_only=True)
    future_argv = train_argv(args, plan_only=False)
    for path in (args.plan_stdout, args.plan_stderr, args.plan_launch_audit, args.plan_preflight_audit, args.launcher_stdout_capture, args.launcher_stderr_capture):
        if repo_path(path).exists():
            raise ValueError(f"plan output path already exists: {display_path(path)}")
    before_anchor = launcher.package_state(repo_path(args.anchor_package))
    before_candidate = launcher.package_state(repo_path(args.candidate_package))
    args.plan_stdout.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(plan_argv, cwd=REPO_ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    repo_path(args.plan_stdout).write_text(proc.stdout, encoding="utf-8")
    repo_path(args.plan_stderr).write_text(proc.stderr, encoding="utf-8")
    after_anchor = launcher.package_state(repo_path(args.anchor_package))
    after_candidate = launcher.package_state(repo_path(args.candidate_package))
    parsed_workload = {
        "train": validation["workload"]["train"],
        "batch": validation["workload"]["batch"],
        "steps_per_epoch": validation["workload"]["steps_per_epoch"],
        "train_pairs_per_epoch": validation["workload"]["total_train_pairs_per_epoch"],
        "planned_pairs": validation["workload"]["planned_pairs"],
        "actual_train_pairs": 0,
    }
    audit = {
        "schema": launcher.LAUNCH_AUDIT_SCHEMA,
        "created_utc": utc_stamp(),
        "actual_training_ran": False,
        "all_checks_passed": True,
        "legal_scope": dict(launcher.EXPECTED_LEGAL_SCOPE),
        "argv_delta": dict(launcher.EXPECTED_ARGV_DELTA),
        "plan_argv": plan_argv,
        "future_training_argv": future_argv,
        "metrics_target": display_path(args.metrics_target),
        "parsed_workload": parsed_workload,
        "plan_result": {
            "exit_status": proc.returncode,
            "stdout_path": display_path(args.plan_stdout),
            "stdout_sha256": sha256_file(args.plan_stdout),
            "stderr_path": display_path(args.plan_stderr),
            "stderr_sha256": sha256_file(args.plan_stderr),
        },
        "binary": {"path": display_path(args.eos_binary), "sha256": sha256_file(args.eos_binary)},
        "artifacts": {
            "train_jsonl": {"path": display_path(args.output_jsonl), "sha256": sha256_file(args.output_jsonl)},
            "manifest": {"path": display_path(args.manifest), "sha256": sha256_file(args.manifest)},
            "coverage": {"path": display_path(args.coverage_report), "sha256": sha256_file(args.coverage_report)},
        },
        "sources": {
            "builder": {"path": "scripts/build_q3_r5_broadfocus_dataset.py", "sha256": sha256_file(Path("scripts/build_q3_r5_broadfocus_dataset.py"))},
            "launcher": {"path": "scripts/q3_r5_training_launcher.py", "sha256": sha256_file(Path("scripts/q3_r5_training_launcher.py"))},
            "test_file": {"path": str(args.test_file), "sha256": sha256_file(args.test_file) if repo_path(args.test_file).is_file() else None},
        },
        "canonical_anchor": {"pre": before_anchor, "post": after_anchor},
        "candidate": {"target": display_path(args.candidate_package), "pre_plan": before_candidate, "post_plan": after_candidate},
        "clone_inventory": {"path": display_path(args.clone_inventory), "sha256": sha256_file(args.clone_inventory), "clone_matches_anchor": clone_audit["clone_matches_anchor"]},
        "checks": {
            "all_contract_checks_true": True,
            "argv_delta_remove_plan_only_only": True,
            "candidate_pretrain_matches_anchor": before_candidate["content_rollup_sha256"] == before_anchor["content_rollup_sha256"],
            "candidate_target_differs_from_anchor": repo_path(args.candidate_package).resolve() != repo_path(args.anchor_package).resolve(),
            "candidate_unchanged_by_plan": before_candidate["content_rollup_sha256"] == after_candidate["content_rollup_sha256"],
            "canonical_anchor_unchanged": before_anchor["content_rollup_sha256"] == after_anchor["content_rollup_sha256"],
            "metrics_target_absent_after_plan": not repo_path(args.metrics_target).exists(),
            "metrics_target_in_plan_argv": display_path(args.metrics_target) in plan_argv,
            "metrics_target_in_training_argv": display_path(args.metrics_target) in future_argv,
            "no_no_tokenizer_in_plan_or_training_argv": "--no-tokenizer" not in plan_argv and "--no-tokenizer" not in future_argv,
            "post_alias_zero_hard_rows_absent": True,
            "workload_exact": parsed_workload == launcher.EXPECTED_PARSED_WORKLOAD,
        },
    }
    if proc.returncode != 0 or not all(audit["checks"].values()):
        audit["all_checks_passed"] = False
    write_json(args.plan_launch_audit, audit)
    launch_sha = sha256_file(args.plan_launch_audit)
    preflight_args = launcher.parse_args(
        [
            "--launch-audit", display_path(args.plan_launch_audit),
            "--preflight-only",
            "--expected-audit-sha256", launch_sha,
            "--attempt-id", "q3-r5-plan-preflight",
            "--stdout-capture", display_path(args.launcher_stdout_capture),
            "--stderr-capture", display_path(args.launcher_stderr_capture),
            "--output-audit", display_path(args.plan_preflight_audit),
        ]
    )
    preflight_audit = launcher.preflight(preflight_args)
    write_json(args.plan_preflight_audit, preflight_audit)
    audit["preflight"] = {"path": display_path(args.plan_preflight_audit), "sha256": sha256_file(args.plan_preflight_audit), "trainer_process_spawned": False}
    write_json(args.plan_launch_audit, audit)
    return audit


def main() -> None:
    args = parse_args()
    r4_train_sha = sha256_file(args.r4_train_jsonl)
    if r4_train_sha != args.expected_r4_train_sha256:
        raise ValueError(f"R4 train sha mismatch: expected={args.expected_r4_train_sha256} actual={r4_train_sha}")
    q3_rows, q3_identities = r4.load_q3_rows(args.q3_per_query_jsonl, r4.expected_q3_hashes(args))
    q3_source_by_dataset = {item["datasets"].keys().__iter__().__next__(): Path(item["path"]) for item in q3_identities}
    sources = {dataset: r4.load_domain_sources(args.dataset_root, dataset) for dataset in ("fiqa", "nfcorpus", "scifact")}
    excluded_qids = r4.read_excluded_qids(args.exclude_qids_json)
    r4_rows = [row for _line, row in iter_jsonl(args.r4_train_jsonl)]
    retained = [copy_row(row) for row in r4_rows if row["selection_bucket"] in {PROMOTION_BUCKET, GUARD_BUCKET}]
    if len(retained) != EXPECTED_FOCUS_SOURCE:
        raise ValueError(f"retained R4 broad rows={len(retained)}, expected {EXPECTED_FOCUS_SOURCE}")
    focus_rows = [
        build_focus_row(row, q3_rows[row["source_dataset"]][str(row["source_query_id"])], sources[row["source_dataset"]], q3_source_by_dataset[row["source_dataset"]])
        for row in retained
    ]
    r4_manifest = read_json(args.r4_manifest)
    nf_promotions = list(r4_manifest["selection_audit"]["promotion_inventory"]["nfcorpus"]["selected_qids"])
    nf_sentinels = list(r4_manifest["sentinel_audit"]["selected_qids"])
    nf_union = sorted(set(nf_promotions) | set(nf_sentinels))
    overlap = sorted(set(nf_promotions) & set(nf_sentinels))
    if len(nf_union) != EXPECTED_NF_UNION or len(overlap) != EXPECTED_NF_OVERLAP:
        raise ValueError(f"nf union/overlap drift: union={len(nf_union)} overlap={len(overlap)}")
    recall_rows, recall_audit = build_nf_recall_rows(nf_union, q3_rows, sources, q3_source_by_dataset["nfcorpus"])
    rows = retained + focus_rows + recall_rows
    if len({row["row_id"] for row in rows}) != len(rows):
        raise ValueError("duplicate output row_id")
    rows.sort(key=lambda row: (row["selection_bucket"], row["source_dataset"], str(row["source_query_id"]), row["row_id"]))
    r4.overlap_report(rows, excluded_qids, sources)
    validation = validate_rows(rows, args)
    write_jsonl(args.output_jsonl, rows)
    manifest = {
        "schema": SCHEMA,
        "created_utc": utc_stamp(),
        "arm_id": ARM_ID,
        "builder": "scripts/build_q3_r5_broadfocus_dataset.py",
        "inputs": {
            "dataset_root": str(args.dataset_root),
            "q3_per_query_jsonl": q3_identities,
            "r4_train": {"path": display_path(args.r4_train_jsonl), "sha256": r4_train_sha},
            "r4_manifest": {"path": display_path(args.r4_manifest), "sha256": sha256_file(args.r4_manifest)},
            "r4_coverage": {"path": display_path(args.r4_coverage), "sha256": sha256_file(args.r4_coverage)},
            "exclude_qids_json": {display_path(path): {"sha256": sha256_file(path)} for path in args.exclude_qids_json},
            "anchor_package": launcher.package_state(repo_path(args.anchor_package)),
        },
        "outputs": {"train_score_spectrum": display_path(args.output_jsonl), "manifest": display_path(args.manifest), "coverage_report": display_path(args.coverage_report)},
        "policy": {
            "retained_r4_broad_rows": "R4 post-alias-safe promotion and guard rows only; R4 sentinels are replaced",
            "focus_rows": "one closest valid anchor-q3 pair for every retained promotion/guard qid, train-side only",
            "nf_recall": "one q3 top100 recall row for every qid in R4 NFCorpus promotion union R4 NFCorpus sentinel qids",
            "no_eval": "zero held-out/dev/reserve/official/test qids and zero eval rows",
            "legal": "research training allowed; release and commercial use false",
        },
        "counts": validation,
        "nf_recall_audit": {"promotion_count": len(nf_promotions), "sentinel_count": len(nf_sentinels), "union_count": len(nf_union), "overlap_count": len(overlap), **recall_audit},
        "legal_gates": dict(LEGAL_GATES),
        **LEGAL_GATES,
        "release_or_commercial_claim": False,
        "quality_claim": False,
    }
    write_json(args.coverage_report, manifest | {"schema": f"{SCHEMA}.coverage.v1"})
    manifest["sha256"] = {
        "builder": sha256_file(Path("scripts/build_q3_r5_broadfocus_dataset.py")),
        "launcher": sha256_file(Path("scripts/q3_r5_training_launcher.py")),
        "train_score_spectrum": sha256_file(args.output_jsonl),
        "coverage_report": sha256_file(args.coverage_report),
    }
    write_json(args.manifest, manifest)
    clone_audit = clone_candidate(args.anchor_package, args.candidate_package, args.clone_inventory)
    launch_audit = run_plan_and_audit(args, validation, clone_audit)
    print(f"q3 R5 broadfocus rows={len(rows)} train_sha256={sha256_file(args.output_jsonl)}")
    print(f"workload planned_pairs={validation['workload']['planned_pairs']} main_q3_pairs={validation['workload']['main_q3_pairs_per_epoch']} recall_q3_pairs={validation['workload']['recall_q3_pairs_per_epoch']}")
    print(f"launch_audit_sha256={sha256_file(args.plan_launch_audit)} preflight_sha256={sha256_file(args.plan_preflight_audit)}")
    if not launch_audit["all_checks_passed"]:
        raise ValueError("launch audit checks failed")


if __name__ == "__main__":
    main()
