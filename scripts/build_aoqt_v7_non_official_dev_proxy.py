#!/usr/bin/env python3
"""Build a non-official AOQT V7 full-retrieval dev proxy report.

This producer is intentionally separate from the official heldout gate. It
materializes dev-only qrels/queries from a V7 split fold, uses full corpus
vector caches, optionally runs the native vector-cache evaluators, and emits a
hash-bound report consumable by the V7 movement canary.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import struct
import subprocess
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "eos.aoqt.v7_non_official_dev_proxy.v1"
SPLIT_SCHEMA = "eos.aoqt.v7_dev_split_manifest.v1"
SPLIT_VALIDATION_SCHEMA = "eos.aoqt.v7_dev_split_validation.v1"
DEV_PROXY_SPLIT_BINDING_SCHEMA = "eos.aoqt.v7_dev_proxy_split_binding.v1"
OFFICIAL_EXCLUSION_SCHEMAS = {"eos.aoqt_stage2.exclusion_qids.v1", "eos.aoqt.exclusion_qids.v1", "eos.aoqt.official_qids.v1"}
TRANSFORM_BINDING_SCHEMA = "eos.aoqt.vector_cache_transform_binding.v1"
EVIDENCE_LABEL = "non_official_dev_proxy"
DOMAINS = ("fiqa", "nfcorpus", "scifact")
SURFACES = ("dense", "q3", "q5")
DEFAULT_QUANTIZER_SEED = 5581486560434873699
CANONICAL_D384_SEED191_PAIRINGS_SHA256 = "dcac29b5b7ab30291bb6791b1e88178e1c755d5d3f995a06efe29a4f430bc6f8"
FORBIDDEN_TRUE_CLAIM_KEYS = (
    "official_heldout_gate",
    "quality_claim",
    "official_claim",
    "official_quality_claim",
    "heldout_quality_claim",
    "promotion_quality_claim",
    "production_quality_claim",
    "release_claim",
    "release_train_allowed",
    "quality_release_authorized",
    "commercial_use_allowed",
    "commercial_release_authorized",
    "commercial_claim",
)


class DevProxyError(RuntimeError):
    pass


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-manifest", required=True, type=Path)
    parser.add_argument("--split-validation", required=True, type=Path)
    parser.add_argument("--fold-id", required=True)
    parser.add_argument("--official-qids", required=True, type=Path)
    parser.add_argument("--sidecar", required=True, type=Path)
    parser.add_argument("--train-metrics", type=Path)
    parser.add_argument("--dev-evidence", type=Path)
    parser.add_argument("--preflight", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--eos-bin", default="./eos")
    parser.add_argument("--quantizer-seed", type=int, default=DEFAULT_QUANTIZER_SEED)
    parser.add_argument("--dataset-dir", action="append", default=[], metavar="DOMAIN=PATH")
    parser.add_argument("--anchor-doc-vectors", action="append", default=[], metavar="DOMAIN=PATH")
    parser.add_argument("--anchor-query-vectors", action="append", default=[], metavar="DOMAIN=PATH")
    parser.add_argument("--candidate-doc-vectors", action="append", default=[], metavar="DOMAIN=PATH")
    parser.add_argument("--candidate-query-vectors", action="append", default=[], metavar="DOMAIN=PATH")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--allow-existing-work", action="store_true")
    return parser.parse_args(argv)


def repo_path(path: Path | str) -> Path:
    parsed = Path(path)
    return parsed if parsed.is_absolute() else REPO_ROOT / parsed


def display_path(path: Path | str) -> str:
    resolved = repo_path(path).resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        return str(resolved)


def sha256_file(path: Path | str) -> str:
    h = hashlib.sha256()
    with repo_path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_json(payload: Any) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def sha256_json(payload: Any) -> str:
    return hashlib.sha256(canonical_json(payload)).hexdigest()


def load_json(path: Path | str, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(repo_path(path).read_text(encoding="utf-8"))
    except OSError as exc:
        raise DevProxyError(f"{label}: cannot read {display_path(path)}") from exc
    except json.JSONDecodeError as exc:
        raise DevProxyError(f"{label}: invalid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise DevProxyError(f"{label}: expected JSON object")
    return payload


def write_json_new(path: Path | str, payload: dict[str, Any]) -> None:
    out = repo_path(path)
    if out.exists():
        raise DevProxyError(f"refusing to overwrite output: {display_path(out)}")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_domain_paths(values: list[str], label: str) -> dict[str, Path]:
    parsed: dict[str, Path] = {}
    for item in values:
        if "=" not in item:
            raise DevProxyError(f"{label}: expected DOMAIN=PATH, got {item!r}")
        domain, raw_path = item.split("=", 1)
        if domain not in DOMAINS:
            raise DevProxyError(f"{label}: unsupported domain {domain!r}")
        if domain in parsed:
            raise DevProxyError(f"{label}: duplicate domain {domain!r}")
        parsed[domain] = Path(raw_path)
    missing = [domain for domain in DOMAINS if domain not in parsed]
    if missing:
        raise DevProxyError(f"{label}: missing domains {missing}")
    return parsed


def require_sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise DevProxyError(f"{label}: expected lowercase sha256")
    return value


def require_false_or_absent_claims(payload: dict[str, Any], label: str) -> None:
    for key in FORBIDDEN_TRUE_CLAIM_KEYS:
        if payload.get(key) is True:
            raise DevProxyError(f"{label}.{key}: dev proxy inputs must not claim quality, official, release, or commercial status")


def validate_split(split_path: Path, validation_path: Path, fold_id: str) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    split = load_json(split_path, "split manifest")
    if split.get("schema") != SPLIT_SCHEMA:
        raise DevProxyError(f"split manifest schema mismatch: {split.get('schema')!r}")
    if split.get("actual_training_ran") is not False or split.get("actual_eval_ran") is not False or split.get("actual_official_data_eval_ran") is not False:
        raise DevProxyError("split manifest must be split-only with no training/eval")
    proof = split.get("leakage_proof")
    if not isinstance(proof, dict) or proof.get("passed") is not True or proof.get("official_overlap_count") != 0:
        raise DevProxyError("split manifest leakage proof did not pass or has official overlap")
    provenance = split.get("provenance")
    if not isinstance(provenance, dict):
        raise DevProxyError("split manifest provenance missing")
    payload_sha = require_sha(provenance.get("manifest_sha256"), "split manifest provenance.manifest_sha256")
    comparable = json.loads(json.dumps(split))
    comparable["provenance"].pop("manifest_sha256", None)
    if sha256_json(comparable) != payload_sha:
        raise DevProxyError("split manifest payload hash mismatch")
    validation = load_json(validation_path, "split validation")
    if validation.get("schema") != SPLIT_VALIDATION_SCHEMA:
        raise DevProxyError("split validation schema mismatch")
    if validation.get("passed") is not True:
        raise DevProxyError("split validation did not pass")
    if validation.get("manifest_file_sha256") != sha256_file(split_path):
        raise DevProxyError("split validation does not bind split manifest file")
    if validation.get("manifest_payload_sha256") != payload_sha:
        raise DevProxyError("split validation does not bind split payload")
    if validation.get("source_plan_sha256") != require_sha(provenance_parent(split, "source_plan").get("sha256"), "split source plan sha256"):
        raise DevProxyError("split validation does not bind split source plan")
    if validation.get("official_qid_registry_sha256") != require_sha(provenance_parent(split, "official_qid_registry").get("manifest_sha256"), "split official registry sha256"):
        raise DevProxyError("split validation does not bind split official registry")
    validation_meta = {
        "schema": SPLIT_VALIDATION_SCHEMA,
        "path": display_path(validation_path),
        "file_sha256": sha256_file(validation_path),
        "payload_sha256": sha256_json(validation),
        "manifest_file_sha256": validation["manifest_file_sha256"],
        "manifest_payload_sha256": validation["manifest_payload_sha256"],
        "source_plan_sha256": validation["source_plan_sha256"],
        "official_qid_registry_sha256": validation["official_qid_registry_sha256"],
        "passed": True,
    }
    folds = split.get("folds")
    if not isinstance(folds, list):
        raise DevProxyError("split manifest folds missing")
    for fold in folds:
        if isinstance(fold, dict) and fold.get("name") == fold_id:
            dev = fold.get("dev")
            if not isinstance(dev, dict):
                raise DevProxyError(f"{fold_id}: dev split missing")
            return split, dev, validation_meta
    raise DevProxyError(f"unknown fold id {fold_id!r}")


def provenance_parent(split: dict[str, Any], key: str) -> dict[str, Any]:
    value = split.get(key)
    if not isinstance(value, dict):
        raise DevProxyError(f"split manifest {key} missing")
    return value


def split_binding(split: dict[str, Any], split_path: Path, fold_id: str, dev: dict[str, Any], validation_meta: dict[str, Any], official_meta: dict[str, Any]) -> dict[str, Any]:
    source_plan = split.get("source_plan", {})
    official = split.get("official_qid_registry", {})
    qids_by_dataset = dev.get("qids_by_dataset")
    counts = dev.get("qid_count_by_dataset")
    if not isinstance(qids_by_dataset, dict) or not isinstance(counts, dict):
        raise DevProxyError("split dev qids/counts missing")
    qid_hashes = {}
    for domain in DOMAINS:
        qids = qids_by_dataset.get(domain)
        if not isinstance(qids, list) or qids != sorted(qids) or any(not isinstance(qid, str) or not qid for qid in qids):
            raise DevProxyError(f"split dev qids for {domain} must be sorted strings")
        if counts.get(domain) != len(qids):
            raise DevProxyError(f"split dev qid count mismatch for {domain}")
        qid_hashes[domain] = sha256_json(qids)
    qids_sha = require_sha(dev.get("qids_by_dataset_sha256"), "split dev qids_by_dataset_sha256")
    if qids_sha != sha256_json(qids_by_dataset):
        raise DevProxyError("split dev qids hash mismatch")
    return {
        "schema": DEV_PROXY_SPLIT_BINDING_SCHEMA,
        "split_manifest_schema": SPLIT_SCHEMA,
        "split_manifest_file_sha256": sha256_file(split_path),
        "split_manifest_payload_sha256": require_sha(split.get("provenance", {}).get("manifest_sha256"), "split payload sha"),
        "split_validation_schema": SPLIT_VALIDATION_SCHEMA,
        "split_validation_file_sha256": require_sha(validation_meta.get("file_sha256"), "split validation file sha"),
        "split_validation_payload_sha256": require_sha(validation_meta.get("payload_sha256"), "split validation payload sha"),
        "source_plan_sha256": require_sha(source_plan.get("sha256"), "split source plan sha256"),
        "official_qid_registry_sha256": require_sha(official.get("manifest_sha256"), "split official registry sha256"),
        "official_registry_source_sha256": require_sha(official.get("source_sha256"), "split official registry source sha256"),
        "official_qids_file_sha256": require_sha(official_meta.get("file_sha256"), "official qids file sha"),
        "fold_id": fold_id,
        "dev_qid_count_by_dataset": {domain: counts[domain] for domain in DOMAINS},
        "dev_qid_set_sha256_by_dataset": qid_hashes,
        "dev_qids_by_dataset_sha256": qids_sha,
    }


def load_official_qids(path: Path, split: dict[str, Any]) -> tuple[dict[str, set[str]], dict[str, Any]]:
    official = load_json(path, "official qids")
    if official.get("schema") not in OFFICIAL_EXCLUSION_SCHEMAS:
        raise DevProxyError("official qid registry schema mismatch")
    split_official = provenance_parent(split, "official_qid_registry")
    official_file_sha = sha256_file(path)
    if official_file_sha != require_sha(split_official.get("manifest_sha256"), "split official registry sha256"):
        raise DevProxyError("official qids file sha does not match split official registry")
    if official.get("source_sha256") is not None and official.get("source_sha256") != require_sha(split_official.get("source_sha256"), "split official registry source sha256"):
        raise DevProxyError("official qids source sha does not match split official registry")
    qids_by_dataset = official.get("qids_by_dataset")
    if not isinstance(qids_by_dataset, dict):
        raise DevProxyError("official qids missing qids_by_dataset")
    out: dict[str, set[str]] = {}
    for domain in DOMAINS:
        qids = qids_by_dataset.get(domain)
        if not isinstance(qids, list) or any(not isinstance(qid, str) for qid in qids):
            raise DevProxyError(f"official qids for {domain} must be array")
        out[domain] = set(qids)
    meta = {
        "path": display_path(path),
        "schema": official["schema"],
        "sha256": official_file_sha,
        "file_sha256": official_file_sha,
        "payload_sha256": sha256_json(official),
        "source_sha256": require_sha(official.get("source_sha256"), "official qids source_sha256") if official.get("source_sha256") is not None else require_sha(split_official.get("source_sha256"), "split official registry source sha256"),
        "qids_by_dataset_sha256": sha256_json({domain: qids_by_dataset[domain] for domain in DOMAINS}),
    }
    return out, meta


def read_queries(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, start=1):
            if not raw.strip():
                continue
            row = json.loads(raw)
            qid = str(row.get("_id") or row.get("id") or "")
            text = "\n".join(str(row.get(key, "")).strip() for key in ("title", "text") if str(row.get(key, "")).strip())
            if not qid:
                raise DevProxyError(f"{path}:{line_no}: missing query id")
            result[qid] = text
    return result


def read_qrels(path: Path) -> list[tuple[str, str, int]]:
    rows: list[tuple[str, str, int]] = []
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if not reader.fieldnames or not {"query-id", "corpus-id", "score"}.issubset(reader.fieldnames):
            raise DevProxyError(f"{path}: expected BEIR qrels TSV header")
        for row in reader:
            qid = str(row["query-id"])
            doc = str(row["corpus-id"])
            score = int(row["score"])
            if score > 0:
                rows.append((qid, doc, score))
    return rows


def vector_ids(path: Path) -> set[str]:
    ids: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, start=1):
            if not raw.strip():
                continue
            row = json.loads(raw)
            rid = str(row.get("id") or row.get("_id") or "")
            if not rid:
                raise DevProxyError(f"{path}:{line_no}: vector row missing id")
            ids.add(rid)
    return ids


def materialize_dev_dataset(domain: str, source_dir: Path, qids: list[str], official_qids: set[str], work_dir: Path) -> dict[str, Any]:
    overlap = sorted(set(qids) & official_qids)
    if overlap:
        raise DevProxyError(f"{domain}: dev fold contains official qids: {overlap[:5]}")
    source_dir = repo_path(source_dir)
    corpus_src = source_dir / "corpus.jsonl"
    queries_src = source_dir / "queries.jsonl"
    qrels_src = source_dir / "qrels" / "train.tsv"
    if not qrels_src.exists():
        qrels_src = source_dir / "qrels" / "dev.tsv"
    if not qrels_src.exists():
        qrels_src = source_dir / "qrels" / "test.tsv"
    if not corpus_src.exists() or not queries_src.exists() or not qrels_src.exists():
        raise DevProxyError(f"{domain}: dataset dir must contain corpus.jsonl, queries.jsonl, and qrels TSV")
    target = work_dir / "dev-datasets" / domain
    qrels_target = target / "qrels" / "dev.tsv"
    if target.exists():
        raise DevProxyError(f"{domain}: work dataset already exists: {display_path(target)}")
    (target / "qrels").mkdir(parents=True, exist_ok=True)
    shutil.copyfile(corpus_src, target / "corpus.jsonl")
    queries = read_queries(queries_src)
    missing_queries = [qid for qid in qids if qid not in queries]
    if missing_queries:
        raise DevProxyError(f"{domain}: missing query text for dev qids {missing_queries[:5]}")
    with (target / "queries.jsonl").open("w", encoding="utf-8") as handle:
        with queries_src.open("r", encoding="utf-8") as source:
            for raw in source:
                if not raw.strip():
                    continue
                row = json.loads(raw)
                qid = str(row.get("_id") or row.get("id") or "")
                if qid in qids:
                    handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    all_qrels = read_qrels(qrels_src)
    qid_set = set(qids)
    dev_qrels = [(qid, doc, score) for qid, doc, score in all_qrels if qid in qid_set]
    with qrels_target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["query-id", "corpus-id", "score"])
        writer.writerows(dev_qrels)
    nonempty_qids = sorted({qid for qid, _doc, _score in dev_qrels})
    empty_qids = sorted(qid_set - set(nonempty_qids))
    return {
        "path": display_path(target),
        "source_dataset_dir": display_path(source_dir),
        "corpus_path": display_path(target / "corpus.jsonl"),
        "queries_path": display_path(target / "queries.jsonl"),
        "qrels_path": display_path(qrels_target),
        "corpus_sha256": sha256_file(target / "corpus.jsonl"),
        "queries_sha256": sha256_file(target / "queries.jsonl"),
        "qrels_sha256": sha256_file(qrels_target),
        "dev_qid_count": len(qids),
        "evaluable_qid_count": len(nonempty_qids),
        "empty_relevance_qid_count": len(empty_qids),
        "empty_relevance_qids_sha256": sha256_json(empty_qids),
        "relevant_doc_ids": sorted({doc for _qid, doc, _score in dev_qrels}),
        "dev_qids": sorted(qids),
        "evaluable_qids": nonempty_qids,
        "qrel_count": len(dev_qrels),
        "fiqa_empty_relevant_doc_vector_misses_audited": domain == "fiqa" and len(empty_qids) > 0,
    }


def validate_vector_coverage(domain: str, dataset_meta: dict[str, Any], doc_vectors: Path, query_vectors: Path) -> dict[str, Any]:
    doc_ids = vector_ids(repo_path(doc_vectors))
    query_ids = vector_ids(repo_path(query_vectors))
    missing_docs = sorted(set(dataset_meta["relevant_doc_ids"]) - doc_ids)
    missing_queries = sorted(set(dataset_meta["dev_qids"]) - query_ids)
    if missing_docs:
        raise DevProxyError(f"{domain}: missing vectors for non-empty relevant docs {missing_docs[:5]}")
    if missing_queries:
        raise DevProxyError(f"{domain}: missing query vectors {missing_queries[:5]}")
    return {
        "doc_vector_path": display_path(doc_vectors),
        "doc_vector_sha256": sha256_file(doc_vectors),
        "doc_vector_id_count": len(doc_ids),
        "query_vector_path": display_path(query_vectors),
        "query_vector_sha256": sha256_file(query_vectors),
        "query_vector_id_count": len(query_ids),
        "missing_non_empty_relevant_doc_vectors": 0,
        "missing_dev_query_vectors": 0,
        "empty_relevance_qids_audited": dataset_meta["empty_relevance_qid_count"],
    }


def command_paths(work_dir: Path, domain: str, label: str, surface: str) -> dict[str, Path]:
    root = work_dir / "eval" / domain
    return {
        "metrics_json": root / f"{label}.{surface}.metrics.json",
        "metrics_tsv": root / f"{label}.{surface}.metrics.tsv",
        "per_query_jsonl": root / f"{label}.{surface}.per-query.jsonl",
    }


def run_command(argv: list[str], execute: bool) -> None:
    if not execute:
        return
    completed = subprocess.run(argv, cwd=REPO_ROOT, text=True, capture_output=True, check=False)
    if completed.returncode != 0:
        raise DevProxyError(f"command failed ({completed.returncode}): {' '.join(argv)}\nstdout={completed.stdout}\nstderr={completed.stderr}")


def build_commands(args: argparse.Namespace, domain: str, dataset_path: str, anchor_doc: Path, anchor_query: Path, candidate_doc: Path, candidate_query: Path, work_dir: Path) -> list[dict[str, Any]]:
    commands: list[dict[str, Any]] = []
    for label, doc_path, query_path in (("anchor", anchor_doc, anchor_query), ("candidate", candidate_doc, candidate_query)):
        dense = command_paths(work_dir, domain, label, "dense")
        dense["metrics_json"].parent.mkdir(parents=True, exist_ok=True)
        commands.append({
            "domain": domain,
            "label": label,
            "surface": "dense",
            "outputs": {key: display_path(value) for key, value in dense.items() if key != "metrics_tsv"},
            "argv": [args.eos_bin, "eval-retrieval-vectors", "--dataset", domain, "--backend", f"{label}-full-cache", "--doc-vectors", str(doc_path), "--query-vectors", str(query_path), "--metrics-json", str(dense["metrics_json"]), "--per-query-jsonl", str(dense["per_query_jsonl"]), dataset_path],
        })
        for surface, bits in (("q3", 3), ("q5", 5)):
            out = command_paths(work_dir, domain, label, surface)
            commands.append({
                "domain": domain,
                "label": label,
                "surface": surface,
                "outputs": {key: display_path(value) for key, value in out.items()},
                "argv": [args.eos_bin, "eval-retrieval-vectors-turboquant", "--dataset", domain, "--backend", f"{label}-full-cache", "--bits", str(bits), "--quantizer-seed", str(args.quantizer_seed), "--top-k", "100", "--per-query-top-k", "10", "--doc-vectors", str(doc_path), "--query-vectors", str(query_path), "--metrics-json", str(out["metrics_json"]), "--metrics-tsv", str(out["metrics_tsv"]), "--per-query-jsonl", str(out["per_query_jsonl"]), dataset_path],
            })
    return commands


def load_metric(path: Path, surface: str) -> dict[str, Any]:
    metric = load_json(path, f"{surface} metrics")
    if surface == "dense":
        return metric["quality"]
    bits = 3 if surface == "q3" else 5
    for row in metric.get("rows", []):
        if isinstance(row, dict) and row.get("bits") == bits and int(row.get("rerank_overfetch") or 0) == 0:
            return row["quality"]
    raise DevProxyError(f"{path}: missing q{bits} metric row")


def load_top10(path: Path) -> dict[str, list[str]]:
    rows: dict[str, list[str]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, start=1):
            if not raw.strip():
                continue
            row = json.loads(raw)
            qid = str(row.get("query_id") or "")
            if not qid:
                raise DevProxyError(f"{path}:{line_no}: missing query_id")
            rows[qid] = [str(item.get("doc_id")) for item in row.get("top_k", [])[:10]]
    return rows


def top10_churn(anchor_path: Path, candidate_path: Path) -> int:
    anchor = load_top10(anchor_path)
    candidate = load_top10(candidate_path)
    if set(anchor) != set(candidate):
        raise DevProxyError("q3 per-query anchor/candidate qid sets differ")
    churn = 0
    for qid in sorted(anchor):
        if anchor[qid] != candidate[qid]:
            churn += 1
    return churn


def metric_delta(candidate: dict[str, Any], anchor: dict[str, Any], key: str) -> float:
    return float(candidate[key]) - float(anchor[key])


def validate_optional_bindings(args: argparse.Namespace) -> dict[str, Any]:
    evidence: dict[str, Any] = {}
    for label, path in (("train_metrics", args.train_metrics), ("dev_evidence", args.dev_evidence), ("preflight", args.preflight), ("sidecar", args.sidecar)):
        if path is not None:
            evidence[label] = {"path": display_path(path), "sha256": sha256_file(path)}
    sidecar = load_json(args.sidecar, "sidecar")
    if sidecar.get("dim") != 384:
        raise DevProxyError("sidecar dim must be 384")
    if sidecar.get("kind") != "aoqt_givens_v1":
        raise DevProxyError("sidecar kind must be aoqt_givens_v1")
    if sidecar.get("seed") != 191:
        raise DevProxyError("sidecar seed must be canonical seed191")
    if sidecar.get("version") not in (None, "eos/aoqt-givens-transform/v1"):
        raise DevProxyError("sidecar version must be eos/aoqt-givens-transform/v1")
    stages = sidecar.get("stages")
    if not isinstance(stages, list) or len(stages) != 8:
        raise DevProxyError("sidecar topology must have 8 stages")
    angle_count = 0
    for stage_index, stage in enumerate(stages):
        if not isinstance(stage, dict):
            raise DevProxyError(f"sidecar stage {stage_index} must be object")
        pairs = stage.get("pairs")
        angles = stage.get("angles")
        if not isinstance(pairs, list) or len(pairs) != 192:
            raise DevProxyError(f"sidecar stage {stage_index} must have 192 pairs")
        if not isinstance(angles, list) or len(angles) != len(pairs):
            raise DevProxyError(f"sidecar stage {stage_index} angle count mismatch")
        seen_coordinates: set[int] = set()
        for pair_index, pair in enumerate(pairs):
            if not isinstance(pair, list) or len(pair) != 2 or any(isinstance(item, bool) or not isinstance(item, int) for item in pair):
                raise DevProxyError(f"sidecar stage {stage_index} pair {pair_index} must be integer pair")
            left, right = pair
            if left < 0 or right < 0 or left >= 384 or right >= 384 or left == right or left in seen_coordinates or right in seen_coordinates:
                raise DevProxyError(f"sidecar stage {stage_index} pair {pair_index} is not a disjoint D384 pairing")
            seen_coordinates.update((left, right))
        angle_count += len(angles)
    if angle_count != 1536:
        raise DevProxyError("sidecar topology must have 1536 angles")
    audit = sidecar.get("audit")
    if not isinstance(audit, dict) or not audit.get("pairings_sha256") or not audit.get("angles_sha256") or audit.get("orthogonality_frobenius_per_dim") is None:
        raise DevProxyError("sidecar audit must bind pairings, angles, and orthogonality")
    if audit["pairings_sha256"] != transform_pairings_sha256(stages):
        raise DevProxyError("sidecar audit pairings_sha256 mismatch")
    if audit["pairings_sha256"] != CANONICAL_D384_SEED191_PAIRINGS_SHA256:
        raise DevProxyError("sidecar audit pairings_sha256 is not canonical D384 seed191")
    if audit["angles_sha256"] != transform_angles_sha256(stages):
        raise DevProxyError("sidecar audit angles_sha256 mismatch")
    orthogonality = audit["orthogonality_frobenius_per_dim"]
    if isinstance(orthogonality, bool) or not isinstance(orthogonality, (int, float)) or not math.isfinite(float(orthogonality)) or float(orthogonality) < 0.0:
        raise DevProxyError("sidecar audit orthogonality_frobenius_per_dim must be finite and non-negative")
    if args.train_metrics is not None:
        train = load_json(args.train_metrics, "train metrics")
        if train.get("schema") != "eos.q3_aoqt_sidecar_metrics.v2":
            raise DevProxyError("train metrics schema mismatch")
        require_false_or_absent_claims(train, "train metrics")
        if isinstance(train.get("summary"), dict):
            require_false_or_absent_claims(train["summary"], "train metrics.summary")
    if args.dev_evidence is not None:
        dev = load_json(args.dev_evidence, "dev evidence")
        if dev.get("dev_only") is not True:
            raise DevProxyError("dev evidence must be dev_only")
        require_false_or_absent_claims(dev, "dev evidence")
    if args.preflight is not None:
        preflight = load_json(args.preflight, "preflight")
        require_false_or_absent_claims(preflight, "preflight")
        if preflight.get("actual_eval_ran") is True:
            raise DevProxyError("preflight must not bind official eval")
    return evidence


def transform_pairings_sha256(stages: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for stage in stages:
        for pair in stage["pairs"]:
            if not isinstance(pair, list) or len(pair) != 2 or any(isinstance(item, bool) or not isinstance(item, int) for item in pair):
                raise DevProxyError("sidecar pairings must be integer pairs")
            digest.update(f"{pair[0]},{pair[1]}\n".encode("ascii"))
    return digest.hexdigest()


def transform_angles_sha256(stages: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for stage in stages:
        for angle in stage["angles"]:
            if isinstance(angle, bool) or not isinstance(angle, (int, float)) or not math.isfinite(float(angle)):
                raise DevProxyError("sidecar angles must be finite numbers")
            try:
                number = struct.unpack("<f", struct.pack("<f", float(angle)))[0]
            except (OverflowError, struct.error) as exc:
                raise DevProxyError("sidecar angles must be representable as float32") from exc
            digest.update(f"{number:.9g}\n".encode("ascii"))
    return digest.hexdigest()


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    if args.execute and args.plan_only:
        raise DevProxyError("--execute and --plan-only are mutually exclusive")
    split, dev, validation_meta = validate_split(args.split_manifest, args.split_validation, args.fold_id)
    official, official_meta = load_official_qids(args.official_qids, split)
    split_meta = split_binding(split, args.split_manifest, args.fold_id, dev, validation_meta, official_meta)
    evidence = validate_optional_bindings(args)
    dataset_dirs = parse_domain_paths(args.dataset_dir, "--dataset-dir")
    anchor_docs = parse_domain_paths(args.anchor_doc_vectors, "--anchor-doc-vectors")
    anchor_queries = parse_domain_paths(args.anchor_query_vectors, "--anchor-query-vectors")
    candidate_docs = parse_domain_paths(args.candidate_doc_vectors, "--candidate-doc-vectors") if args.candidate_doc_vectors else {}
    candidate_queries = parse_domain_paths(args.candidate_query_vectors, "--candidate-query-vectors") if args.candidate_query_vectors else {}
    output = repo_path(args.output)
    work_dir = repo_path(args.work_dir) if args.work_dir else output.with_suffix("")
    if work_dir.exists() and not args.allow_existing_work:
        raise DevProxyError(f"work dir already exists: {display_path(work_dir)}")
    work_dir.mkdir(parents=True, exist_ok=True)

    domains: dict[str, Any] = {}
    commands: list[dict[str, Any]] = []
    transform_bindings: dict[str, Any] = {}
    vector_coverage: dict[str, Any] = {}
    materialized: dict[str, Any] = {}
    for domain in DOMAINS:
        qids = dev["qids_by_dataset"][domain]
        dataset_meta = materialize_dev_dataset(domain, dataset_dirs[domain], qids, official[domain], work_dir)
        materialized[domain] = {key: value for key, value in dataset_meta.items() if key != "relevant_doc_ids" and key != "evaluable_qids"}
        vector_coverage[domain] = validate_vector_coverage(domain, dataset_meta, anchor_docs[domain], anchor_queries[domain])
        cand_doc = candidate_docs.get(domain)
        cand_query = candidate_queries.get(domain)
        if cand_doc is None or cand_query is None:
            cand_doc = work_dir / "transformed-vectors" / domain / "doc-vectors.jsonl"
            cand_query = work_dir / "transformed-vectors" / domain / "query-vectors.jsonl"
            sidecar_out = work_dir / "transformed-vectors" / domain / "binding.json"
            transform_argv = [
                args.eos_bin,
                "transform-aoqt-vectors",
                "--transform", str(args.sidecar),
                "--doc-vectors", str(anchor_docs[domain]),
                "--query-vectors", str(anchor_queries[domain]),
                "--out-doc-vectors", str(cand_doc),
                "--out-query-vectors", str(cand_query),
                "--sidecar-json", str(sidecar_out),
                "--dataset", domain,
                "--artifact", "aoqt-v7-candidate",
            ]
            commands.append({"domain": domain, "label": "candidate", "surface": "transform", "outputs": {"sidecar_json": display_path(sidecar_out), "doc_vectors": display_path(cand_doc), "query_vectors": display_path(cand_query)}, "argv": transform_argv})
            run_command(transform_argv, args.execute)
            if args.execute:
                binding = load_json(sidecar_out, f"{domain} transform binding")
                if binding.get("schema") != TRANSFORM_BINDING_SCHEMA:
                    raise DevProxyError(f"{domain}: transform binding schema mismatch")
                transform_bindings[domain] = binding
        vector_coverage[f"{domain}.candidate"] = validate_vector_coverage(domain, dataset_meta, cand_doc, cand_query) if args.execute or (repo_path(cand_doc).exists() and repo_path(cand_query).exists()) else {"planned_doc_vector_path": display_path(cand_doc), "planned_query_vector_path": display_path(cand_query)}
        eval_commands = build_commands(args, domain, materialized[domain]["path"], anchor_docs[domain], anchor_queries[domain], cand_doc, cand_query, work_dir)
        commands.extend(eval_commands)
        for command in eval_commands:
            run_command(command["argv"], args.execute)

        if args.execute:
            metrics: dict[tuple[str, str], dict[str, Any]] = {}
            per_query: dict[tuple[str, str], Path] = {}
            for label in ("anchor", "candidate"):
                for surface in SURFACES:
                    out = command_paths(work_dir, domain, label, surface)
                    metrics[(label, surface)] = load_metric(out["metrics_json"], surface)
                    per_query[(label, surface)] = out["per_query_jsonl"]
            q3_churn = top10_churn(per_query[("anchor", "q3")], per_query[("candidate", "q3")])
            domains[domain] = {
                "dense": {"ndcg_at_10_delta": metric_delta(metrics[("candidate", "dense")], metrics[("anchor", "dense")], "ndcg_at_10")},
                "q3": {
                    "ndcg_at_10_delta": metric_delta(metrics[("candidate", "q3")], metrics[("anchor", "q3")], "ndcg_at_10"),
                    "recall_at_100_delta": metric_delta(metrics[("candidate", "q3")], metrics[("anchor", "q3")], "recall_at_100"),
                    "top10_churn": q3_churn,
                },
                "q5": {"ndcg_at_10_delta": metric_delta(metrics[("candidate", "q5")], metrics[("anchor", "q5")], "ndcg_at_10")},
                "q3_top10_churn": q3_churn,
            }
        else:
            domains[domain] = {"dense": {"ndcg_at_10_delta": 0.0}, "q3": {"ndcg_at_10_delta": 0.0, "recall_at_100_delta": 0.0, "top10_churn": 0}, "q5": {"ndcg_at_10_delta": 0.0}, "q3_top10_churn": 0}

    q3_macro = math.fsum(float(domains[domain]["q3"]["ndcg_at_10_delta"]) for domain in DOMAINS) / len(DOMAINS)
    report = {
        "schema": SCHEMA,
        "evidence_label": EVIDENCE_LABEL,
        "dev_only": True,
        "official_heldout_gate": False,
        "quality_claim": False,
        "official_claim": False,
        "release_claim": False,
        "commercial_claim": False,
        "split_manifest": split_meta,
        "split_validation": validation_meta,
        "evidence": evidence,
        "official_qids": official_meta,
        "materialized_dev_datasets": materialized,
        "vector_coverage": vector_coverage,
        "transform_bindings": transform_bindings,
        "commands": commands,
        "execution": {"executed": args.execute, "plan_only": args.plan_only, "full_real_proxy_ran": False},
        "domains": domains,
        "macro": {
            "q3_ndcg_at_10_delta": q3_macro,
            "q3_top10_churn": int(math.fsum(int(domains[domain]["q3_top10_churn"]) for domain in DOMAINS)),
            "recomputed_from_domains": True,
        },
        "provenance": {
            "builder": display_path(Path(__file__)),
            "builder_sha256": sha256_file(Path(__file__)),
            "report_hash_excludes": ["provenance.report_sha256"],
        },
    }
    comparable = json.loads(json.dumps(report))
    comparable["provenance"].pop("report_sha256", None)
    report["provenance"]["report_sha256"] = sha256_json(comparable)
    return report


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        report = build_report(args)
        write_json_new(args.output, report)
    except DevProxyError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
