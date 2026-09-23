#!/usr/bin/env python3
"""Materialize and gate the q3 R4 train-proxy retrieval preflight."""

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


SCHEMA = "eos.q3_r4_proxy_preflight.v2"
POLICY_SCHEMA = "eos.q3_r4_proxy_metric_aligned_policy.v2"
SUMMARY_SCHEMA = "eos.q3_r4_proxy_gate_summary.v1"

RUN_ROOT = Path("runs/eos-d384-q3-boundary-goal-v1-20260902T084144Z")
R4_ROOT = RUN_ROOT / "headroom-q3broad-gap050-dualcut-nf64-fiqa21-sf1-nfrecall24"
CANDIDATE_ID = "arm-q3broad-gap050-nf64-fiqa21-sf1-nfrecall24-rec005-q3w004-rw015-m002-lr2e6-e2-r4"
ANCHOR_PACKAGE = Path("runs/eos-wide384-projection-tail-seed191-repeat-v1-20260902T004753Z/packages/d384-pre.mll")
CANDIDATE_PACKAGE = RUN_ROOT / "packages" / CANDIDATE_ID / "d384-pre.mll"
BINARY = RUN_ROOT / "bin/eos-r4-broad-gap-postfix"
TRAIN_JSONL = R4_ROOT / "data/beir134.q3broad-gap050-dualcut-nf64-fiqa21-sf1-nfrecall24.train.jsonl"
TRAIN_MANIFEST = R4_ROOT / "data/beir134.q3broad-gap050-dualcut-nf64-fiqa21-sf1-nfrecall24.manifest.json"
COVERAGE = R4_ROOT / "q3broad-gap050-dualcut-nf64-fiqa21-sf1-nfrecall24.coverage.json"
TRAIN_METRICS = RUN_ROOT / "metrics" / f"{CANDIDATE_ID}.train.metrics.json"
ATTEMPT2_AUDIT = R4_ROOT / "reports/q3-r4-training-postfix-attempt2-execute-audit.json"

EXPECTED_HASHES = {
    "train_jsonl": "201a9093a827eba809a05e072520dc02e3ae0bd5ba7d1c4953e6c58e8ae84a03",
    "train_manifest": "d0f778bc4f635b0b0200ab88c567306106ac5b401a7968d1bd76ee21654d7ec2",
    "coverage": "ddfa88b8fe0fde593e3bb40a76adf8347005ac1d4be001e875fdc9d376f8b389",
    "train_metrics": "d6691275c4b3f40b5164190f316fc46ec9adfa38cf562bcc09b8215c48431f55",
    "attempt2_audit": "b4dcd916cf34d39cd7c4a44cb31d15f0d79512d8ae3faf6eb04dc8d63c466300",
    "binary": "c03c60696cf6fb89efc020eb14244188f98ea0b0ec628680320bd72351229f3d",
}
EXPECTED_CANONICAL_PACKAGE_ROLLUPS = {
    "anchor": "17335593ec850a1c01c8b31b17a32fe002368a508c06d431b761d45f1d9a76b0",
    "candidate": "5f5363253530171b88a70fca6341dfa6efb8bbd1e9ef7e53524bf912e6365d5f",
}
HISTORICAL_NON_AUTHORIZING_ARTIFACTS = {
    "note": "Earlier q3 R4 proxy preflight policies/plans are retained only as historical review inputs; they do not authorize execution or gate decisions after this policy version.",
    "policy_sha256": [
        "da369c9b21fb87f3e6a2d18dff6b8bfafc7540e6c8a264946c27dbd13ebde4b5",
        "524941c1afc48cb3234577e665298e64a568f87908a793bb5ff24aac732dfb7f",
    ],
    "execution_plan_sha256": [
        "3525b1d0a4c5caf5a082857c130bcad8e5ad5cf4c7c21c3648773bd5daf25d34",
    ],
    "proxy_qrels_manifest_sha256": [
        "6f774519a3ec9881b5a3c0e8d5ca0efeddac5b52fef29da85708ace1763c798d",
    ],
}
Q3 = {
    "method": "turboquant_ip_b3",
    "bits": 3,
    "seed": 5581486560434873699,
    "surface": "turboquant_ip_prepared",
    "top_k": 100,
    "per_query_top_k": 100,
}
DOMAINS = ("fiqa", "nfcorpus", "scifact")
RAW_QRELS = {
    "fiqa": Path("datasets/manta-embed-v1/raw/fiqa/fiqa/qrels/train.tsv"),
    "nfcorpus": Path("datasets/manta-embed-v1/raw/nfcorpus/nfcorpus/qrels/train.tsv"),
    "scifact": Path("datasets/manta-embed-v1/raw/scifact/scifact/qrels/train.tsv"),
}
DATASET_DIRS = {
    "fiqa": Path("datasets/manta-embed-v1/raw/fiqa/fiqa"),
    "nfcorpus": Path("datasets/manta-embed-v1/raw/nfcorpus/nfcorpus"),
    "scifact": Path("datasets/manta-embed-v1/raw/scifact/scifact"),
}
HELDOUT_QREL_PATHS = {
    "fiqa": [Path("datasets/manta-embed-v1/raw/fiqa/fiqa/qrels/test.tsv")],
    "nfcorpus": [Path("datasets/manta-embed-v1/raw/nfcorpus/nfcorpus/qrels/test.tsv")],
    "scifact": [Path("datasets/manta-embed-v1/raw/scifact/scifact/qrels/test.tsv")],
}
EXCLUSION_JSONS = [
    RUN_ROOT / "data/selected-qids.dev4.json",
    RUN_ROOT / "data/selected-qids.reserve4.json",
]
QID_SPLIT_MANIFESTS = [RUN_ROOT / "data/qid-splits.manifest.json"]
BUCKETS = {
    "promotion": "promotion_11_20_to_top10_near_gap",
    "guard": "guard_top10_margin",
    "sentinel": "nfcorpus_recall_sentinel_q3top100_rank80_100",
}


class PreflightError(RuntimeError):
    pass


def utc_stamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def compact_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def canonical_sibling_rollup(items: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for item in items:
        digest.update(compact_json({"bytes": item["bytes"], "name": item["name"], "sha256": item["sha256"]}).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if line:
                try:
                    yield line_no, json.loads(line)
                except json.JSONDecodeError as exc:
                    raise PreflightError(f"{path}:{line_no}: invalid JSON: {exc}") from exc


def read_train_rows(path: Path) -> list[dict[str, Any]]:
    return [row for _line, row in iter_jsonl(path)]


def read_qrels(path: Path) -> dict[str, dict[str, int]]:
    qrels: dict[str, dict[str, int]] = defaultdict(dict)
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if not reader.fieldnames or not {"query-id", "corpus-id", "score"}.issubset(reader.fieldnames):
            raise PreflightError(f"{path}: expected BEIR qrels header")
        for row in reader:
            gain = int(row["score"])
            if gain > 0:
                qrels[str(row["query-id"])][str(row["corpus-id"])] = gain
    return dict(qrels)


def qrels_stats(path: Path) -> dict[str, Any]:
    qrels = read_qrels(path)
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "qid_count": len(qrels),
        "relevant_rows": sum(len(docs) for docs in qrels.values()),
        "qids": sorted(qrels),
        "gain_histogram": dict(sorted(Counter(str(gain) for docs in qrels.values() for gain in docs.values()).items())),
        "content_sha256": sha256_json({qid: dict(sorted(docs.items())) for qid, docs in sorted(qrels.items())}),
    }


def qrels_all_stats(path: Path) -> dict[str, Any]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if not reader.fieldnames or not {"query-id", "corpus-id", "score"}.issubset(reader.fieldnames):
            raise PreflightError(f"{path}: expected BEIR qrels header")
        for row in reader:
            rows.append((str(row["query-id"]), str(row["corpus-id"]), int(row["score"])))
    qids = sorted({qid for qid, _doc, _gain in rows})
    docs = sorted({doc for _qid, doc, _gain in rows})
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "qrel_rows": len(rows),
        "qid_count": len(qids),
        "doc_count": len(docs),
        "qids": qids,
        "qid_set_sha256": sha256_json(qids),
        "gain_histogram": dict(sorted(Counter(str(gain) for _qid, _doc, gain in rows).items())),
        "qrel_content_sha256": sha256_json([{"qid": qid, "doc_id": doc, "gain": gain} for qid, doc, gain in sorted(rows)]),
    }


def qids_from_qrels(path: Path) -> set[str]:
    qids: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if not reader.fieldnames or "query-id" not in reader.fieldnames:
            raise PreflightError(f"{path}: expected BEIR qrels header")
        for row in reader:
            if row.get("query-id"):
                qids.add(str(row["query-id"]))
    return qids


def qids_from_exclusion_json(path: Path) -> dict[str, set[str]]:
    payload = read_json(path)
    selected = payload.get("selected_qids", payload)
    if not isinstance(selected, dict):
        raise PreflightError(f"{path}: expected selected_qids object")
    out: dict[str, set[str]] = defaultdict(set)
    for dataset, values in selected.items():
        if not isinstance(values, list):
            raise PreflightError(f"{path}: selected_qids.{dataset} must be a list")
        out[str(dataset).lower()].update(str(value) for value in values)
    return out


def exclusion_source_record(path: Path) -> dict[str, Any]:
    selected = {dataset: sorted(qids) for dataset, qids in qids_from_exclusion_json(path).items()}
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "split": str(read_json(path).get("split") or path.stem),
        "selected_qids": selected,
        "qid_counts": {dataset: len(qids) for dataset, qids in selected.items()},
        "qid_set_sha256": sha256_json(selected),
    }


def qid_split_manifest_record(path: Path) -> dict[str, Any]:
    payload = read_json(path)
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "schema": payload.get("schema"),
        "splits": payload.get("splits"),
        "overlap": payload.get("overlap"),
        "content_sha256": sha256_json(payload),
    }


def source_input_records() -> dict[str, Any]:
    return {
        "exclusion_qid_json": [exclusion_source_record(path) for path in EXCLUSION_JSONS],
        "qid_split_manifests": [qid_split_manifest_record(path) for path in QID_SPLIT_MANIFESTS],
        "heldout_test_qrels": {
            dataset: [qrels_all_stats(path) for path in HELDOUT_QREL_PATHS[dataset]]
            for dataset in DOMAINS
        },
    }


def verify_source_inputs(policy: dict[str, Any]) -> tuple[dict[str, set[str]], dict[str, set[str]], dict[str, Any]]:
    source_inputs = policy.get("source_inputs")
    if not isinstance(source_inputs, dict):
        raise PreflightError("policy missing source_inputs")
    if len(source_inputs.get("exclusion_qid_json", [])) != len(EXCLUSION_JSONS):
        raise PreflightError("policy pinned exclusion source count mismatch")
    if len(source_inputs.get("qid_split_manifests", [])) != len(QID_SPLIT_MANIFESTS):
        raise PreflightError("policy pinned qid split manifest count mismatch")
    heldout_inputs = source_inputs.get("heldout_test_qrels", {})
    if not isinstance(heldout_inputs, dict):
        raise PreflightError("policy heldout qrels source pins missing")
    for dataset in DOMAINS:
        if len(heldout_inputs.get(dataset, [])) != len(HELDOUT_QREL_PATHS[dataset]):
            raise PreflightError(f"policy pinned heldout qrels count mismatch: {dataset}")
    excluded: dict[str, set[str]] = defaultdict(set)
    verified_exclusions = []
    seen_paths: set[str] = set()
    for pinned in source_inputs.get("exclusion_qid_json", []):
        path = Path(str(pinned.get("path") or ""))
        if not path.is_file():
            raise PreflightError(f"missing pinned exclusion source: {path}")
        if str(path) in seen_paths:
            raise PreflightError(f"duplicate pinned exclusion source: {path}")
        seen_paths.add(str(path))
        current = exclusion_source_record(path)
        for key in ("path", "sha256", "bytes", "split", "selected_qids", "qid_counts", "qid_set_sha256"):
            if current.get(key) != pinned.get(key):
                raise PreflightError(f"exclusion source {path} {key} drift")
        for dataset, qids in current["selected_qids"].items():
            excluded[dataset].update(qids)
        verified_exclusions.append(current)
    verified_manifests = []
    seen_manifest_paths: set[str] = set()
    for pinned in source_inputs.get("qid_split_manifests", []):
        path = Path(str(pinned.get("path") or ""))
        if not path.is_file():
            raise PreflightError(f"missing pinned qid split manifest: {path}")
        if str(path) in seen_manifest_paths:
            raise PreflightError(f"duplicate pinned qid split manifest: {path}")
        seen_manifest_paths.add(str(path))
        current = qid_split_manifest_record(path)
        for key in ("path", "sha256", "bytes", "schema", "splits", "overlap", "content_sha256"):
            if current.get(key) != pinned.get(key):
                raise PreflightError(f"qid split manifest {path} {key} drift")
        verified_manifests.append(current)
    heldout: dict[str, set[str]] = defaultdict(set)
    verified_heldout: dict[str, Any] = {}
    for dataset in DOMAINS:
        verified_heldout[dataset] = []
        seen_heldout_paths: set[str] = set()
        for pinned in source_inputs.get("heldout_test_qrels", {}).get(dataset, []):
            path = Path(str(pinned.get("path") or ""))
            if not path.is_file():
                raise PreflightError(f"missing pinned heldout qrels source: {dataset}:{path}")
            if str(path) in seen_heldout_paths:
                raise PreflightError(f"duplicate pinned heldout qrels source: {dataset}:{path}")
            seen_heldout_paths.add(str(path))
            current = qrels_all_stats(path)
            for key in ("path", "sha256", "bytes", "qrel_rows", "qid_count", "doc_count", "qids", "qid_set_sha256", "gain_histogram", "qrel_content_sha256"):
                if current.get(key) != pinned.get(key):
                    raise PreflightError(f"heldout qrels {dataset}:{path} {key} drift")
            heldout[dataset].update(current["qids"])
            verified_heldout[dataset].append(current)
    return (
        {dataset: set(excluded.get(dataset, set())) for dataset in DOMAINS},
        {dataset: set(heldout.get(dataset, set())) for dataset in DOMAINS},
        {"exclusion_qid_json": verified_exclusions, "qid_split_manifests": verified_manifests, "heldout_test_qrels": verified_heldout},
    )


def merge_excluded(paths: list[Path]) -> dict[str, set[str]]:
    out: dict[str, set[str]] = defaultdict(set)
    for path in paths:
        for dataset, qids in qids_from_exclusion_json(path).items():
            out[dataset].update(qids)
    return out


def train_qid_inventory(rows: list[dict[str, Any]]) -> dict[str, set[str]]:
    qids: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        dataset = str(row.get("source_dataset") or "").lower()
        qid = str(row.get("source_query_id") or "")
        if dataset not in DOMAINS or not qid:
            raise PreflightError(f"bad train row identity: {row.get('row_id')}")
        qids[dataset].add(qid)
    return {dataset: set(qids.get(dataset, set())) for dataset in DOMAINS}


def write_subset_qrels(raw_qrels: Path, selected_qids: set[str], output: Path) -> dict[str, Any]:
    output.parent.mkdir(parents=True, exist_ok=True)
    seen_pairs: set[tuple[str, str]] = set()
    rows: list[tuple[str, str, int]] = []
    with raw_qrels.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if not reader.fieldnames or not {"query-id", "corpus-id", "score"}.issubset(reader.fieldnames):
            raise PreflightError(f"{raw_qrels}: expected BEIR qrels header")
        for row in reader:
            qid = str(row["query-id"])
            doc_id = str(row["corpus-id"])
            gain = int(row["score"])
            if qid not in selected_qids or gain <= 0:
                continue
            key = (qid, doc_id)
            if key in seen_pairs:
                raise PreflightError(f"{raw_qrels}: duplicate positive qrel {qid}/{doc_id}")
            seen_pairs.add(key)
            rows.append((qid, doc_id, gain))
    missing = sorted(qid for qid in selected_qids if not any(row[0] == qid for row in rows))
    if missing:
        raise PreflightError(f"{raw_qrels}: selected qids missing train qrels: {missing[:10]}")
    rows.sort(key=lambda item: (item[0], item[1]))
    with output.open("w", encoding="utf-8") as handle:
        handle.write("query-id\tcorpus-id\tscore\n")
        for qid, doc_id, gain in rows:
            handle.write(f"{qid}\t{doc_id}\t{gain}\n")
    stats = qrels_stats(output)
    stats["source_qrels"] = str(raw_qrels)
    stats["source_qrels_sha256"] = sha256_file(raw_qrels)
    stats["source_qrels_bytes"] = raw_qrels.stat().st_size
    return stats


def package_sibling_hashes(package: Path) -> dict[str, Any]:
    if not package.is_file():
        raise PreflightError(f"missing package {package}")
    stem = package.name[:-4] if package.name.endswith(".mll") else package.stem
    siblings = sorted(path for path in package.parent.glob(f"{stem}*") if path.is_file())
    items = [{"name": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)} for path in siblings]
    return {
        "target": str(package),
        "target_sha256": sha256_file(package),
        "sibling_count": len(items),
        "sibling_hashes": items,
        "canonical_content_rollup_sha256": canonical_sibling_rollup(items),
        "noncanonical_json_array_rollup_sha256": sha256_json(items),
        "rollup_algorithm": "canonical: sha256 of sorted pathless sibling records; each compact key-sorted JSON object followed by LF, including final record",
    }


def verify_fixed_inputs() -> dict[str, Any]:
    paths = {
        "train_jsonl": TRAIN_JSONL,
        "train_manifest": TRAIN_MANIFEST,
        "coverage": COVERAGE,
        "train_metrics": TRAIN_METRICS,
        "attempt2_audit": ATTEMPT2_AUDIT,
        "binary": BINARY,
    }
    actual = {}
    for key, path in paths.items():
        if not path.is_file():
            raise PreflightError(f"missing required input {key}: {path}")
        digest = sha256_file(path)
        want = EXPECTED_HASHES[key]
        if digest != want:
            raise PreflightError(f"{key} sha mismatch: expected {want}, actual {digest}")
        actual[key] = {"path": str(path), "sha256": digest, "bytes": path.stat().st_size}
    return actual


def proxy_result_paths(proxy_eval: Path, dataset: str, label: str) -> dict[str, str]:
    stem = f"{label}.{dataset}.q3.r4-proxy"
    return {
        "metrics_json": str(proxy_eval / f"{stem}.json"),
        "metrics_tsv": str(proxy_eval / f"{stem}.tsv"),
        "per_query_jsonl": str(proxy_eval / f"{stem}.per-query.jsonl"),
    }


def eval_argv(binary: Path, package: Path, dataset: str, qrels: str, outputs: dict[str, str]) -> list[str]:
    return [
        str(binary),
        "eval-retrieval-turboquant",
        "--bits",
        "3",
        "--quantizer-seed",
        str(Q3["seed"]),
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
        str(package),
        str(DATASET_DIRS[dataset]),
    ]


def build_preflight() -> tuple[dict[str, Any], dict[str, Any]]:
    fixed = verify_fixed_inputs()
    rows = read_train_rows(TRAIN_JSONL)
    qids = train_qid_inventory(rows)
    excluded = merge_excluded(EXCLUSION_JSONS)
    sources = source_input_records()
    row_counts = Counter(str(row["source_dataset"]).lower() for row in rows)
    bucket_counts = Counter(str(row["selection_bucket"]) for row in rows)
    bucket_qids: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        bucket_qids[str(row["selection_bucket"])][str(row["source_dataset"]).lower()].append(str(row["source_query_id"]))

    proxy_qrels_root = R4_ROOT / "proxy-qrels-r4"
    qrel_outputs: dict[str, Any] = {}
    heldout_report: dict[str, Any] = {}
    for dataset in DOMAINS:
        test_qids: set[str] = set()
        for heldout_path in HELDOUT_QREL_PATHS[dataset]:
            test_qids.update(qids_from_qrels(heldout_path))
        frozen = qids[dataset] & excluded.get(dataset, set())
        heldout = qids[dataset] & test_qids
        if frozen or heldout:
            raise PreflightError(f"{dataset}: qid leakage frozen={sorted(frozen)} heldout={sorted(heldout)}")
        out = proxy_qrels_root / dataset / "qrels" / "r4-train-proxy.tsv"
        qrel_outputs[dataset] = write_subset_qrels(RAW_QRELS[dataset], qids[dataset], out)
        heldout_report[dataset] = {
            "selected_qids": len(qids[dataset]),
            "frozen_dev_reserve_overlap": len(frozen),
            "official_test_overlap": len(heldout),
            "frozen_dev_reserve_overlap_qids": sorted(frozen),
            "official_test_overlap_qids": sorted(heldout),
        }

    proxy_eval = R4_ROOT / "proxy-eval"
    commands = []
    expected_outputs: dict[str, Any] = defaultdict(dict)
    absent_outputs: dict[str, bool] = {}
    for dataset in DOMAINS:
        for label, package in (("anchor", ANCHOR_PACKAGE), ("candidate", CANDIDATE_PACKAGE)):
            outputs = proxy_result_paths(proxy_eval, dataset, label)
            expected_outputs[label][dataset] = outputs
            absent_outputs[f"{label}.{dataset}"] = not any(Path(path).exists() for path in outputs.values())
            commands.append(
                {
                    "label": label,
                    "dataset": dataset,
                    "argv": eval_argv(BINARY, package, dataset, qrel_outputs[dataset]["path"], outputs),
                    "outputs": outputs,
                }
            )
    if not all(absent_outputs.values()):
        raise PreflightError(f"proxy outputs already exist: {absent_outputs}")

    anchor_hashes = package_sibling_hashes(ANCHOR_PACKAGE)
    candidate_hashes = package_sibling_hashes(CANDIDATE_PACKAGE)
    for label, actual in (("anchor", anchor_hashes), ("candidate", candidate_hashes)):
        expected = EXPECTED_CANONICAL_PACKAGE_ROLLUPS[label]
        if actual["canonical_content_rollup_sha256"] != expected:
            raise PreflightError(f"{label} canonical package rollup mismatch: expected {expected}, actual {actual['canonical_content_rollup_sha256']}")
    policy = {
        "schema": POLICY_SCHEMA,
        "created_utc": utc_stamp(),
        "status": "no_results_yet",
        "evaluation_executed": False,
        "candidate_id": CANDIDATE_ID,
        "execution_authorization": {
            "required_marker": "q3-r4-proxy-independent-review-complete",
            "review_required_before_eval": True,
            "policy_mutation_after_eval_forbidden": True,
        },
        "immutable_inputs": fixed,
        "source_inputs": sources,
        "packages": {"anchor": anchor_hashes, "candidate": candidate_hashes},
        "q3_recipe": Q3,
        "datasets": DOMAINS,
        "forbidden_eval_surfaces": ["dense", "q5", "dev", "reserve", "official"],
        "historical_non_authorizing_artifacts": HISTORICAL_NON_AUTHORIZING_ARTIFACTS,
        "proxy_qrels": qrel_outputs,
        "proxy_qrels_manifest": {
            "path": str(proxy_qrels_root / "manifest.json"),
            "sha256": None,
            "bytes": None,
        },
        "expected_outputs": expected_outputs,
        "future_eval_commands": commands,
        "gates": {
            "macro_q3_ndcg_delta_min_exclusive": 0.0,
            "macro_q3_recall_at_100_delta_min": 0.0,
            "domain_q3_ndcg_delta_min": {"nfcorpus": 0.006, "fiqa": -0.0005, "scifact": -0.0005},
            "domain_recall_at_100_delta_min": {"nfcorpus": 0.0, "fiqa": 0.0, "scifact": 0.0},
            "metric_aligned_top10_substitutions": {
                "same_query_one_to_one": True,
                "sort_by_qrel_gain_desc": True,
                "replacement_gain_must_be_at_least_lost_gain": True,
                "affected_query_top10_gain_mass_nondecreasing": True,
                "affected_query_ndcg_at_10_nondecreasing": True,
                "affected_query_recall_at_10_nondecreasing": True,
                "lower_gain_or_nonrelevant_replacement": "fail",
            },
            "diagnostics": {
                "worst_per_query_ndcg_delta_min": -0.01,
                "recall_at_100_per_query_loss_at_most_relevant_docs": 1,
                "recall_at_100_per_query_loss_delta_min": -0.03,
                "aggregate_domain_recall_gates_decisive": True,
            },
            "fail_closed": [
                "missing_outputs",
                "duplicate_outputs",
                "asymmetric_outputs",
                "stale_outputs",
                "anchor_drift",
                "seed_bits_surface_mismatch",
                "package_binary_input_hash_drift",
                "nonfinite_metrics",
                "qrel_query_mismatch",
            ],
        },
    }
    policy_path = proxy_eval / "q3-r4-proxy-metric-aligned-policy.json"
    write_json(policy_path, policy)

    manifest = {
        "schema": f"{SCHEMA}.proxy_qrels_manifest.v1",
        "created_utc": utc_stamp(),
        "source_train_jsonl": str(TRAIN_JSONL),
        "source_train_sha256": fixed["train_jsonl"]["sha256"],
        "source_inputs": sources,
        "row_counts": dict(row_counts),
        "bucket_counts": dict(bucket_counts),
        "bucket_qids": {bucket: {ds: sorted(set(vals)) for ds, vals in by_ds.items()} for bucket, by_ds in bucket_qids.items()},
        "datasets": qrel_outputs,
        "heldout_overlap": heldout_report,
        "dedupe_policy": "proxy qrels dedupe to unique dataset/query IDs; promotion, guard, and sentinel training rows may reference the same qid but eval qrels include each query once with all positive train qrels",
        "sentinel_guard_promotion_semantics": "all qids referenced by any R4 training row are included once in the train-proxy qrels; qrel gains come only from raw train qrels",
        "policy_path": str(policy_path),
        "policy_authority_note": "policy pins this manifest hash; this manifest is provenance, not a result authorization artifact",
    }
    manifest_path = proxy_qrels_root / "manifest.json"
    write_json(manifest_path, manifest)
    manifest_sha = sha256_file(manifest_path)
    write_json(manifest_path.with_suffix(manifest_path.suffix + ".sha256"), {"path": str(manifest_path), "sha256": manifest_sha})
    policy["proxy_qrels_manifest"] = {
        "path": str(manifest_path),
        "sha256": manifest_sha,
        "bytes": manifest_path.stat().st_size,
    }
    write_json(policy_path, policy)
    policy_sha = sha256_file(policy_path)
    write_json(policy_path.with_suffix(policy_path.suffix + ".sha256"), {"path": str(policy_path), "sha256": policy_sha})
    audit = {
        "schema": f"{SCHEMA}.execution_plan_audit.v1",
        "created_utc": utc_stamp(),
        "created_epoch_ns": time.time_ns(),
        "evaluation_executed": False,
        "execution_session": {
            "authorization_marker_required": policy["execution_authorization"]["required_marker"],
            "authorized": False,
        },
        "input_checks": {
            "fixed_hashes_match": True,
            "heldout_overlap_zero": all(
                item["frozen_dev_reserve_overlap"] == 0 and item["official_test_overlap"] == 0
                for item in heldout_report.values()
            ),
            "outputs_absent": all(absent_outputs.values()),
            "policy_written_before_eval": True,
            "same_binary_for_anchor_candidate": True,
            "same_q3_recipe_for_anchor_candidate": True,
        },
        "outputs_absent": absent_outputs,
        "policy_path": str(policy_path),
        "policy_sha256": policy_sha,
        "proxy_qrels_manifest": str(manifest_path),
        "proxy_qrels_manifest_sha256": manifest_sha,
        "historical_non_authorizing_artifacts": HISTORICAL_NON_AUTHORIZING_ARTIFACTS,
        "future_eval_commands": commands,
        "expected_outputs": expected_outputs,
    }
    audit_path = proxy_eval / "q3-r4-proxy-execution-plan-audit.json"
    write_json(audit_path, audit)
    audit_sha = sha256_file(audit_path)
    write_json(audit_path.with_suffix(audit_path.suffix + ".sha256"), {"path": str(audit_path), "sha256": audit_sha})
    return manifest, {"policy": policy, "audit": audit, "hashes": {"manifest": manifest_sha, "policy": policy_sha, "audit": audit_sha}}


def metric_row(metrics: dict[str, Any]) -> dict[str, Any]:
    rows = metrics.get("rows", [])
    if not isinstance(rows, list):
        raise PreflightError("metrics rows missing")
    matches = []
    for row in rows:
        if row.get("method") == Q3["method"] and int(row.get("bits", 0)) == Q3["bits"]:
            matches.append(row)
    if len(matches) != 1:
        raise PreflightError(f"expected exactly one q3 metric row, got {len(matches)}")
    return matches[0]


def finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def require_finite(value: Any, label: str) -> float:
    if not finite(value):
        raise PreflightError(f"nonfinite metric {label}: {value!r}")
    return float(value)


def verify_proxy_qrels(policy: dict[str, Any], policy_path: Path) -> dict[str, Any]:
    manifest_pin = policy.get("proxy_qrels_manifest")
    if not isinstance(manifest_pin, dict):
        raise PreflightError("policy missing proxy_qrels_manifest")
    manifest_path = Path(str(manifest_pin.get("path") or ""))
    if not manifest_path.is_file():
        raise PreflightError(f"missing proxy qrels manifest: {manifest_path}")
    manifest_sha = sha256_file(manifest_path)
    if manifest_sha != manifest_pin.get("sha256"):
        raise PreflightError(f"proxy qrels manifest hash drift: expected {manifest_pin.get('sha256')} actual {manifest_sha}")
    if manifest_path.stat().st_size != manifest_pin.get("bytes"):
        raise PreflightError("proxy qrels manifest byte count drift")
    manifest = read_json(manifest_path)
    if manifest.get("policy_path") != str(policy_path):
        raise PreflightError("proxy qrels manifest policy path mismatch")
    if manifest.get("source_inputs") != policy.get("source_inputs"):
        raise PreflightError("proxy qrels manifest source_inputs mismatch")
    excluded, heldout, source_audit = verify_source_inputs(policy)
    rows = read_train_rows(TRAIN_JSONL)
    expected_qids = train_qid_inventory(rows)
    current: dict[str, Any] = {}
    for dataset in DOMAINS:
        pinned = policy["proxy_qrels"][dataset]
        current_stats = qrels_stats(Path(pinned["path"]))
        for key in ("path", "sha256", "bytes", "qid_count", "relevant_rows", "qids", "gain_histogram", "content_sha256"):
            if current_stats.get(key) != pinned.get(key):
                raise PreflightError(f"{dataset} proxy qrels {key} drift")
        if set(current_stats["qids"]) != expected_qids[dataset]:
            raise PreflightError(f"{dataset} proxy qrels qid set no longer equals train-only R4 qids")
        source_path = Path(str(pinned.get("source_qrels") or ""))
        if not source_path.is_file():
            raise PreflightError(f"{dataset} source qrels missing")
        if sha256_file(source_path) != pinned.get("source_qrels_sha256") or source_path.stat().st_size != pinned.get("source_qrels_bytes"):
            raise PreflightError(f"{dataset} source qrels hash/byte drift")
        frozen = expected_qids[dataset] & excluded.get(dataset, set())
        heldout_overlap = expected_qids[dataset] & heldout.get(dataset, set())
        if frozen or heldout_overlap:
            raise PreflightError(f"{dataset}: qid leakage frozen={sorted(frozen)} heldout={sorted(heldout_overlap)}")
        current[dataset] = current_stats
    return {"manifest_path": str(manifest_path), "manifest_sha256": manifest_sha, "datasets": current, "source_inputs": source_audit}


def expected_output_paths(policy: dict[str, Any]) -> set[str]:
    paths = set()
    for label in ("anchor", "candidate"):
        for dataset in DOMAINS:
            for value in policy["expected_outputs"][label][dataset].values():
                paths.add(str(value))
    return paths


def verify_output_set_and_freshness(policy: dict[str, Any], policy_path: Path) -> dict[str, Any]:
    expected = expected_output_paths(policy)
    proxy_eval = policy_path.parent
    extras = [
        str(path)
        for path in proxy_eval.glob("*.q3.r4-proxy*")
        if str(path) not in expected
    ]
    if extras:
        raise PreflightError(f"unexpected proxy output extras: {extras}")
    plan_path = proxy_eval / "q3-r4-proxy-execution-plan-audit.json"
    if not plan_path.is_file():
        raise PreflightError(f"missing proxy execution plan: {plan_path}")
    plan = read_json(plan_path)
    if plan.get("policy_sha256") != sha256_file(policy_path):
        raise PreflightError("execution plan policy hash mismatch")
    minimum_mtime_ns = max(policy_path.stat().st_mtime_ns, plan_path.stat().st_mtime_ns)
    records: dict[str, Any] = {}
    for path_text in sorted(expected):
        path = Path(path_text)
        if not path.is_file():
            raise PreflightError(f"missing expected proxy output: {path}")
        if path.stat().st_mtime_ns <= minimum_mtime_ns:
            raise PreflightError(f"stale proxy output before policy/plan: {path}")
        records[path_text] = {"sha256": sha256_file(path), "bytes": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns}
    return {"plan_path": str(plan_path), "plan_sha256": sha256_file(plan_path), "outputs": records}


def validate_aggregate_metrics(
    metrics: dict[str, Any],
    *,
    dataset: str,
    label: str,
    package: Path,
    qrels: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, float]]:
    if metrics.get("schema") != "manta.embedding_turboquant_retrieval_metrics.v1":
        raise PreflightError(f"{label}.{dataset}: aggregate schema mismatch")
    if metrics.get("dataset") != dataset:
        raise PreflightError(f"{label}.{dataset}: aggregate dataset mismatch")
    if metrics.get("artifact") != str(package):
        raise PreflightError(f"{label}.{dataset}: aggregate artifact mismatch")
    config = metrics.get("config")
    if not isinstance(config, dict):
        raise PreflightError(f"{label}.{dataset}: aggregate config missing")
    if config.get("bits") != [Q3["bits"]] or int(config.get("quantizer_seed", 0)) != Q3["seed"]:
        raise PreflightError(f"{label}.{dataset}: aggregate bits/seed mismatch")
    if int(config.get("top_k", 0)) != Q3["top_k"]:
        raise PreflightError(f"{label}.{dataset}: aggregate top_k mismatch")
    inputs = metrics.get("inputs")
    if not isinstance(inputs, dict):
        raise PreflightError(f"{label}.{dataset}: aggregate inputs missing")
    if inputs.get("qrels_path") != qrels["path"]:
        raise PreflightError(f"{label}.{dataset}: aggregate qrels path mismatch")
    if inputs.get("qrels_sha256") is not None and inputs.get("qrels_sha256") != qrels["sha256"]:
        raise PreflightError(f"{label}.{dataset}: aggregate qrels hash mismatch")
    if int(inputs.get("queries", -1)) != int(qrels["qid_count"]):
        raise PreflightError(f"{label}.{dataset}: aggregate query count mismatch")
    if int(inputs.get("relevant_pairs", -1)) != int(qrels["relevant_rows"]):
        raise PreflightError(f"{label}.{dataset}: aggregate qrel count mismatch")
    row = metric_row(metrics)
    if int(row.get("candidate_count", 0)) <= 0 or int(row.get("candidates_scored", 0)) <= 0:
        raise PreflightError(f"{label}.{dataset}: aggregate candidate counts missing")
    if row.get("scoring_surface") is not None and row.get("scoring_surface") != Q3["surface"]:
        raise PreflightError(f"{label}.{dataset}: aggregate scoring surface mismatch")
    quality = row.get("quality")
    if not isinstance(quality, dict):
        raise PreflightError(f"{label}.{dataset}: aggregate quality missing")
    values = {
        "ndcg_at_10": require_finite(quality.get("ndcg_at_10"), f"{label}.{dataset}.ndcg_at_10"),
        "recall_at_100": require_finite(quality.get("recall_at_100"), f"{label}.{dataset}.recall_at_100"),
    }
    return row, values


def parse_tsv_float(value: str, label: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise PreflightError(f"nonfinite or nonnumeric TSV metric {label}: {value!r}") from exc
    if not math.isfinite(parsed):
        raise PreflightError(f"nonfinite or nonnumeric TSV metric {label}: {value!r}")
    return parsed


def validate_metrics_tsv(path: Path, *, dataset: str, label: str, json_quality: dict[str, float]) -> dict[str, Any]:
    required = {"dataset", "row", "bits", "method", "ndcg_at_10", "recall_at_100"}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise PreflightError(f"{label}.{dataset}: TSV schema/header mismatch")
        rows = list(reader)
    if len(rows) != 2:
        raise PreflightError(f"{label}.{dataset}: TSV expected dense+q3 rows, got {len(rows)}")
    dense_rows = [row for row in rows if row.get("row") == "dense"]
    q3_rows = [row for row in rows if row.get("row") == "quantized" and row.get("method") == Q3["method"] and row.get("bits") == str(Q3["bits"])]
    if len(dense_rows) != 1 or len(q3_rows) != 1:
        raise PreflightError(f"{label}.{dataset}: TSV missing/duplicate expected dense or q3 row")
    dense = dense_rows[0]
    if dense.get("dataset") != dataset or dense.get("method") != "float32" or dense.get("bits") not in {"", None}:
        raise PreflightError(f"{label}.{dataset}: TSV dense row identity mismatch")
    q3 = q3_rows[0]
    if q3.get("dataset") != dataset:
        raise PreflightError(f"{label}.{dataset}: TSV q3 dataset mismatch")
    ndcg = parse_tsv_float(q3.get("ndcg_at_10", ""), f"{label}.{dataset}.ndcg_at_10")
    recall = parse_tsv_float(q3.get("recall_at_100", ""), f"{label}.{dataset}.recall_at_100")
    if abs(ndcg - json_quality["ndcg_at_10"]) > 0.000001:
        raise PreflightError(f"{label}.{dataset}: TSV-vs-JSON ndcg_at_10 mismatch")
    if abs(recall - json_quality["recall_at_100"]) > 0.000001:
        raise PreflightError(f"{label}.{dataset}: TSV-vs-JSON recall_at_100 mismatch")
    for field in ("candidate_count", "candidates_scored"):
        if q3.get(field) not in {"", None}:
            try:
                int(q3[field])
            except ValueError as exc:
                raise PreflightError(f"{label}.{dataset}: TSV {field} is not an integer") from exc
    return {"path": str(path), "row_count": len(rows), "q3": {"ndcg_at_10": ndcg, "recall_at_100": recall}}


def validate_per_query_rows(
    path: Path,
    *,
    dataset: str,
    label: str,
    qrels: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    rows = load_per_query(path)
    expected = set(qrels["qids"])
    qrel_map = read_qrels(Path(qrels["path"]))
    if set(rows) != expected:
        raise PreflightError(f"{label}.{dataset}: per-query qid set mismatch")
    for qid, row in rows.items():
        if row.get("dataset") != dataset:
            raise PreflightError(f"{label}.{dataset}: per-query dataset mismatch at {qid}")
        if int(row.get("relevant_count", -1)) != len(qrel_map[qid]):
            raise PreflightError(f"{label}.{dataset}: per-query relevant_count mismatch at {qid}")
        quality = row.get("quality")
        if not isinstance(quality, dict):
            raise PreflightError(f"{label}.{dataset}: per-query quality missing at {qid}")
        for metric in ("ndcg_at_10", "recall_at_10", "recall_at_100"):
            require_finite(quality.get(metric), f"{label}.{dataset}.{qid}.{metric}")
        qrel_docs = qrel_map[qid]
        for item in row["top_k"]:
            gain = int(item.get("relevance") or 0)
            expected_gain = int(qrel_docs.get(str(item.get("doc_id")), 0))
            if gain != expected_gain:
                raise PreflightError(f"{label}.{dataset}: per-query qrel/query mismatch at {qid}/{item.get('doc_id')}")
    return rows


def load_per_query(path: Path) -> dict[str, dict[str, Any]]:
    rows = {}
    for _line, row in iter_jsonl(path):
        qid = str(row.get("query_id") or "")
        if not qid or qid in rows:
            raise PreflightError(f"{path}: missing or duplicate qid {qid!r}")
        if row.get("method") != Q3["method"] or int(row.get("bits", 0)) != Q3["bits"]:
            raise PreflightError(f"{path}: seed/bits/method mismatch at {qid}")
        if int(row.get("quantizer_seed", 0)) != Q3["seed"] or row.get("scoring_surface") != Q3["surface"]:
            raise PreflightError(f"{path}: seed/surface mismatch at {qid}")
        top_k = row.get("top_k")
        if not isinstance(top_k, list) or len(top_k) != 100:
            raise PreflightError(f"{path}: qid {qid} missing top100 diagnostics")
        rows[qid] = row
    return rows


def top10_gain(row: dict[str, Any]) -> dict[str, int]:
    return {str(item["doc_id"]): int(item.get("relevance") or 0) for item in row["top_k"][:10]}


def substitution_audit(anchor: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    lost = [(doc, gain) for doc, gain in top10_gain(anchor).items() if gain > 0 and doc not in top10_gain(candidate)]
    entered = [(doc, gain) for doc, gain in top10_gain(candidate).items() if gain > 0 and doc not in top10_gain(anchor)]
    lost.sort(key=lambda item: (-item[1], item[0]))
    entered.sort(key=lambda item: (-item[1], item[0]))
    matches = []
    ok = len(entered) >= len(lost)
    for index, lost_item in enumerate(lost):
        entered_item = entered[index] if index < len(entered) else (None, 0)
        match_ok = entered_item[1] >= lost_item[1]
        ok = ok and match_ok
        matches.append({"lost": lost_item, "entered": entered_item, "ok": match_ok})
    gain_delta = sum(top10_gain(candidate).values()) - sum(top10_gain(anchor).values())
    ndcg_delta = float(candidate["quality"]["ndcg_at_10"]) - float(anchor["quality"]["ndcg_at_10"])
    recall10_delta = float(candidate["quality"]["recall_at_10"]) - float(anchor["quality"]["recall_at_10"])
    affected = bool(lost or entered)
    metric_ok = (not affected) or (gain_delta >= -1e-12 and ndcg_delta >= -1e-12 and recall10_delta >= -1e-12)
    return {
        "query_id": str(anchor["query_id"]),
        "affected_top10": affected,
        "lost_relevant_top10_count": len(lost),
        "entered_relevant_top10_count": len(entered),
        "matches": matches,
        "matching_ok": ok,
        "top10_gain_delta": gain_delta,
        "ndcg_at_10_delta": ndcg_delta,
        "recall_at_10_delta": recall10_delta,
        "affected_metric_ok": metric_ok,
    }


def gate(policy_path: Path, output_path: Path) -> dict[str, Any]:
    policy = read_json(policy_path)
    if policy.get("schema") != POLICY_SCHEMA:
        raise PreflightError("policy schema mismatch")
    verify_fixed_inputs()
    if policy.get("evaluation_executed") is not False or policy.get("status") != "no_results_yet":
        raise PreflightError("policy is not in no-results-yet state")
    qrels_audit = verify_proxy_qrels(policy, policy_path)
    output_audit = verify_output_set_and_freshness(policy, policy_path)
    for label, package in (("anchor", ANCHOR_PACKAGE), ("candidate", CANDIDATE_PACKAGE)):
        current = package_sibling_hashes(package)
        pinned = policy["packages"][label]
        if current["target_sha256"] != pinned["target_sha256"]:
            raise PreflightError(f"{label} package target hash drift")
        if current["canonical_content_rollup_sha256"] != pinned["canonical_content_rollup_sha256"]:
            raise PreflightError(f"{label} package sibling rollup drift")
        expected_rollup = EXPECTED_CANONICAL_PACKAGE_ROLLUPS[label]
        if current["canonical_content_rollup_sha256"] != expected_rollup:
            raise PreflightError(f"{label} package canonical rollup no longer equals expected R4 evidence")
    domains: dict[str, Any] = {}
    failures: list[str] = []
    macro_ndcg = 0.0
    macro_recall = 0.0
    for dataset in DOMAINS:
        paths = policy["expected_outputs"]
        anchor_paths = paths["anchor"][dataset]
        candidate_paths = paths["candidate"][dataset]
        anchor_metrics = read_json(Path(anchor_paths["metrics_json"]))
        candidate_metrics = read_json(Path(candidate_paths["metrics_json"]))
        qrels = policy["proxy_qrels"][dataset]
        _ar, anchor_quality = validate_aggregate_metrics(
            anchor_metrics, dataset=dataset, label="anchor", package=ANCHOR_PACKAGE, qrels=qrels
        )
        _cr, candidate_quality = validate_aggregate_metrics(
            candidate_metrics, dataset=dataset, label="candidate", package=CANDIDATE_PACKAGE, qrels=qrels
        )
        anchor_tsv = validate_metrics_tsv(Path(anchor_paths["metrics_tsv"]), dataset=dataset, label="anchor", json_quality=anchor_quality)
        candidate_tsv = validate_metrics_tsv(Path(candidate_paths["metrics_tsv"]), dataset=dataset, label="candidate", json_quality=candidate_quality)
        for key in ("config", "inputs"):
            anchor_copy = dict(anchor_metrics[key])
            candidate_copy = dict(candidate_metrics[key])
            if key == "inputs":
                anchor_copy.pop("scored_pairs", None)
                candidate_copy.pop("scored_pairs", None)
            if anchor_copy != candidate_copy:
                raise PreflightError(f"{dataset}: anchor/candidate {key} asymmetry")
        ndcg_delta = candidate_quality["ndcg_at_10"] - anchor_quality["ndcg_at_10"]
        recall_delta = candidate_quality["recall_at_100"] - anchor_quality["recall_at_100"]
        macro_ndcg += ndcg_delta
        macro_recall += recall_delta
        anchor_pq = validate_per_query_rows(Path(anchor_paths["per_query_jsonl"]), dataset=dataset, label="anchor", qrels=qrels)
        candidate_pq = validate_per_query_rows(Path(candidate_paths["per_query_jsonl"]), dataset=dataset, label="candidate", qrels=qrels)
        if set(anchor_pq) != set(candidate_pq):
            raise PreflightError(f"{dataset}: asymmetric per-query qids")
        audits = [substitution_audit(anchor_pq[qid], candidate_pq[qid]) for qid in sorted(anchor_pq)]
        worst_ndcg = min((item["ndcg_at_10_delta"] for item in audits), default=0.0)
        recall_losses = []
        for qid in sorted(anchor_pq):
            delta = float(candidate_pq[qid]["quality"]["recall_at_100"]) - float(anchor_pq[qid]["quality"]["recall_at_100"])
            if delta < -1e-12:
                relevant = int(anchor_pq[qid].get("relevant_count") or 0)
                recall_losses.append(
                    {
                        "query_id": qid,
                        "recall_at_100_delta": delta,
                        "estimated_relevant_doc_loss": round(abs(delta) * relevant),
                        "ok": round(abs(delta) * relevant) <= 1 and delta >= -0.03 - 1e-12,
                    }
                )
        lower = [item for item in audits if not item["matching_ok"] or not item["affected_metric_ok"]]
        domain_pass = (
            ndcg_delta >= float(policy["gates"]["domain_q3_ndcg_delta_min"][dataset]) - 1e-12
            and recall_delta >= float(policy["gates"]["domain_recall_at_100_delta_min"][dataset]) - 1e-12
            and not lower
            and worst_ndcg >= -0.01 - 1e-12
            and all(item["ok"] for item in recall_losses)
        )
        if not domain_pass:
            failures.append(dataset)
        domains[dataset] = {
            "q3_ndcg_at_10_delta": ndcg_delta,
            "recall_at_100_delta": recall_delta,
            "tsv": {"anchor": anchor_tsv, "candidate": candidate_tsv},
            "worst_per_query_ndcg_delta": worst_ndcg,
            "recall100_losses": recall_losses,
            "substitution_failures": lower,
            "pass": domain_pass,
        }
    macro_ndcg /= len(DOMAINS)
    macro_recall /= len(DOMAINS)
    macro_pass = macro_ndcg > 0.0 and macro_recall >= -1e-12
    summary = {
        "schema": SUMMARY_SCHEMA,
        "created_utc": utc_stamp(),
        "policy": str(policy_path),
        "policy_sha256": sha256_file(policy_path),
        "proxy_qrels_audit": qrels_audit,
        "output_hashes": output_audit["outputs"],
        "execution_plan": {"path": output_audit["plan_path"], "sha256": output_audit["plan_sha256"]},
        "evaluation_executed": True,
        "domains": domains,
        "macro": {"q3_ndcg_at_10_delta": macro_ndcg, "recall_at_100_delta": macro_recall, "pass": macro_pass},
        "gate_pass": macro_pass and not failures,
        "failures": failures + ([] if macro_pass else ["macro"]),
    }
    write_json(output_path, summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("preflight")
    gate_cmd = sub.add_parser("gate")
    gate_cmd.add_argument("--policy", required=True, type=Path)
    gate_cmd.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.cmd == "preflight":
        manifest, bundle = build_preflight()
        print(
            "q3 R4 proxy preflight ready "
            f"qids={sum(item['qid_count'] for item in manifest['datasets'].values())} "
            f"qrels={sum(item['relevant_rows'] for item in manifest['datasets'].values())} "
            f"policy_sha256={bundle['hashes']['policy']}"
        )
    elif args.cmd == "gate":
        summary = gate(args.policy, args.output)
        print(f"q3 R4 proxy gate {'PASS' if summary['gate_pass'] else 'FAIL'}")
        if not summary["gate_pass"]:
            raise SystemExit(1)


if __name__ == "__main__":
    try:
        main()
    except PreflightError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
