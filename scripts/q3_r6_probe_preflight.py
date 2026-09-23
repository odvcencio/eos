#!/usr/bin/env python3
"""Preflight and future gate validation for q3 R6 probe outputs."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import median
from typing import Any
from hashlib import sha256


PROBE_SCHEMA = "eos.q3_r6_frontier_retention_probe_manifest.v1"
PER_QUERY_SCHEMA = "manta.embedding_turboquant_retrieval_per_query.v1"
METRICS_SCHEMA = "manta.embedding_turboquant_retrieval_metrics.v1"
Q3_METHOD = "turboquant_ip_b3"
Q3_BITS = 3
Q3_SEED = 5581486560434873699
Q3_SURFACE = "turboquant_ip_prepared"
EXPECTED_BINARY_SHA256 = "cc474f5503827b607c53333832a5e173ba85d5d1bbb67ac94bc0f0f77e33abca"
EXPECTED_SURFACES = {
    "frontier.fiqa": ("frontier_context_q3", "fiqa"),
    "frontier.nfcorpus": ("frontier_context_q3", "nfcorpus"),
    "frontier.scifact": ("frontier_context_q3", "scifact"),
    "top10-retention.fiqa": ("top10_retention_q3", "fiqa"),
    "top10-retention.nfcorpus": ("top10_retention_q3", "nfcorpus"),
    "top10-retention.scifact": ("top10_retention_q3", "scifact"),
    "nf-boundary80-100.nfcorpus": ("nf_boundary80_100_docretention_q3", "nfcorpus"),
}
REPO_ROOT = Path(__file__).resolve().parent.parent


class ProbeContractError(RuntimeError):
    pass


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-manifest", required=True, type=Path)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--allow-pending-runtime-bindings", action="store_true")
    return parser.parse_args(argv)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    h = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, start=1):
            raw = raw.strip()
            if raw:
                try:
                    yield line_no, json.loads(raw)
                except json.JSONDecodeError as exc:
                    raise ProbeContractError(f"{path}:{line_no}: invalid JSON: {exc}") from exc


def read_qrels(path: Path) -> dict[str, dict[str, int]]:
    qrels: dict[str, dict[str, int]] = {}
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if not reader.fieldnames or not {"query-id", "corpus-id", "score"}.issubset(reader.fieldnames):
            raise ProbeContractError(f"{path}: expected BEIR qrels TSV header")
        for row in reader:
            qid = str(row["query-id"])
            doc_id = str(row["corpus-id"])
            try:
                gain = int(row["score"])
            except (TypeError, ValueError) as exc:
                raise ProbeContractError(f"{path}: nonnumeric qrel gain {qid}/{doc_id}") from exc
            if gain <= 0:
                raise ProbeContractError(f"{path}: nonpositive qrel gain {qid}/{doc_id}")
            if qid in qrels and doc_id in qrels[qid]:
                raise ProbeContractError(f"{path}: duplicate qrel {qid}/{doc_id}")
            qrels.setdefault(qid, {})[doc_id] = gain
    if not qrels:
        raise ProbeContractError(f"{path}: empty qrels")
    return qrels


def repo_path(path: str | Path, base: Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else base / p


def require_resolved_within(path: str | Path, root: str | Path, base: Path, message: str) -> Path:
    resolved = repo_path(path, base).resolve()
    resolved_root = repo_path(root, base).resolve()
    try:
        resolved.relative_to(resolved_root)
    except ValueError as exc:
        raise ProbeContractError(message) from exc
    return resolved


def finite_float(value: Any, label: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ProbeContractError(f"{label}: nonnumeric value") from exc
    if not math.isfinite(parsed):
        raise ProbeContractError(f"{label}: nonfinite value")
    return parsed


def exact_int(value: Any, label: str) -> int:
    try:
        if isinstance(value, float) and not value.is_integer():
            raise ValueError(value)
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ProbeContractError(f"{label}: nonnumeric integer") from exc


def validate_manifest(manifest_path: Path, *, require_outputs_absent: bool = True, allow_pending_runtime_bindings: bool = True) -> dict[str, Any]:
    base = REPO_ROOT
    manifest = read_json(manifest_path)
    if manifest.get("schema") != PROBE_SCHEMA:
        raise ProbeContractError("probe manifest schema mismatch")
    surfaces = manifest.get("surfaces")
    if not isinstance(surfaces, dict) or set(surfaces) != set(EXPECTED_SURFACES):
        raise ProbeContractError("probe surface set mismatch")
    if manifest.get("surface_count") != 7 or manifest.get("command_count") != 14 or manifest.get("expected_output_count") != 42:
        raise ProbeContractError("probe command/output count mismatch")
    binary = manifest.get("binary", {})
    if binary.get("sha256") != EXPECTED_BINARY_SHA256:
        raise ProbeContractError("binary sha mismatch")
    if not allow_pending_runtime_bindings:
        candidate = manifest.get("candidate_package", {})
        if "future" in str(candidate.get("sibling_rollup_sha256", "")):
            raise ProbeContractError("candidate package rollup is still a future binding")
        audit = manifest.get("candidate_effective_base_leakage_audit")
        if not isinstance(audit, dict) or audit.get("base_hard_soft_recovery_work_units") != 0 or audit.get("aux_only") is not True:
            raise ProbeContractError("candidate base leakage audit missing or non-aux-only")
    expected_outputs = manifest.get("expected_outputs", [])
    if len(expected_outputs) != 42 or len(set(expected_outputs)) != 42:
        raise ProbeContractError("expected output set mismatch")
    outputs_root = repo_path(manifest["future_paths"]["outputs_root"], base).resolve()
    for output in expected_outputs:
        output_path = require_resolved_within(output, outputs_root, base, f"future output path escapes outputs root: {output}")
        if require_outputs_absent and output_path.exists():
            raise ProbeContractError(f"future output path already exists: {output}")
    commands = manifest.get("commands", [])
    if len(commands) != 14:
        raise ProbeContractError("command count mismatch")
    seen = set()
    command_outputs = []
    for command in commands:
        surface_id = command.get("surface_id")
        label = command.get("label")
        if surface_id not in EXPECTED_SURFACES or label not in {"anchor", "candidate"}:
            raise ProbeContractError("command surface/label mismatch")
        key = (surface_id, label)
        if key in seen:
            raise ProbeContractError("duplicate command surface/label")
        seen.add(key)
        bucket, dataset = EXPECTED_SURFACES[surface_id]
        if command.get("bucket") != bucket or command.get("dataset") != dataset:
            raise ProbeContractError("command bucket/dataset mismatch")
        outputs = command.get("outputs", {})
        if set(outputs) != {"metrics_json", "metrics_tsv", "per_query_jsonl"}:
            raise ProbeContractError("command outputs shape mismatch")
        command_outputs.extend(outputs.values())
        argv = command.get("argv")
        expected_argv = [
            binary["path"],
            "eval-retrieval-turboquant",
            "--bits",
            "3",
            "--quantizer-seed",
            str(Q3_SEED),
            "--top-k",
            "100",
            "--per-query-top-k",
            "100",
            "--dataset",
            dataset,
            "--qrels",
            surfaces[surface_id]["qrels"]["path"],
            "--metrics-json",
            outputs["metrics_json"],
            "--metrics-tsv",
            outputs["metrics_tsv"],
            "--per-query-jsonl",
            outputs["per_query_jsonl"],
            command["package"],
            surfaces[surface_id]["dataset_dir"],
        ]
        if argv != expected_argv:
            raise ProbeContractError(f"{surface_id}.{label}: argv shape mismatch")
    if sorted(command_outputs) != sorted(expected_outputs):
        raise ProbeContractError("command outputs do not equal expected_outputs")
    for surface_id, surface in surfaces.items():
        bucket, dataset = EXPECTED_SURFACES[surface_id]
        if surface.get("bucket") != bucket or surface.get("dataset") != dataset:
            raise ProbeContractError(f"{surface_id}: surface bucket/dataset mismatch")
        qrels_meta = surface.get("qrels", {})
        qrels_rel = Path(qrels_meta.get("path", ""))
        if qrels_rel.is_absolute():
            raise ProbeContractError(f"{surface_id}: qrels path must be repo-relative")
        qrels_parts = qrels_rel.parts
        if "probe" not in qrels_parts or "qrels" not in qrels_parts or dataset not in qrels_parts:
            raise ProbeContractError(f"{surface_id}: qrels path is not namespaced by dataset")
        if qrels_rel.name != f"{surface_id}.qrels.tsv":
            raise ProbeContractError(f"{surface_id}: qrels filename mismatch")
        qrels_path = repo_path(qrels_meta["path"], base)
        if sha256_file(qrels_path) != qrels_meta.get("sha256"):
            raise ProbeContractError(f"{surface_id}: qrels sha mismatch")
        qrels = read_qrels(qrels_path)
        if len(qrels) != surface["qrels"]["qid_count"] or sum(len(docs) for docs in qrels.values()) != surface["qrels"]["qrel_count"]:
            raise ProbeContractError(f"{surface_id}: qrels count mismatch")
    return manifest


def load_per_query(path: Path, surface_id: str, dataset: str, qrels: dict[str, dict[str, int]], expected_package: str) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for _line, row in iter_jsonl(path):
        if row.get("schema") != PER_QUERY_SCHEMA:
            raise ProbeContractError(f"{surface_id}: per-query schema mismatch")
        if row.get("dataset") != dataset or row.get("method") != Q3_METHOD or int(row.get("bits") or 0) != Q3_BITS:
            raise ProbeContractError(f"{surface_id}: per-query dataset/method/bits mismatch")
        if int(row.get("quantizer_seed") or 0) != Q3_SEED or row.get("scoring_surface") != Q3_SURFACE:
            raise ProbeContractError(f"{surface_id}: per-query seed/surface mismatch")
        if row.get("artifact") is not None and row.get("artifact") != expected_package:
            raise ProbeContractError(f"{surface_id}: per-query artifact mismatch")
        qid = str(row.get("query_id"))
        if qid in rows:
            raise ProbeContractError(f"{surface_id}: duplicate per-query qid {qid}")
        if qid not in qrels:
            raise ProbeContractError(f"{surface_id}: per-query qid outside qrels {qid}")
        if exact_int(row.get("relevant_count"), f"{surface_id}: relevant_count") != len(qrels[qid]):
            raise ProbeContractError(f"{surface_id}: relevant_count mismatch for {qid}")
        top_k = row.get("top_k")
        if not isinstance(top_k, list) or len(top_k) != 100:
            raise ProbeContractError(f"{surface_id}: per-query top_k length mismatch")
        ranks = [int(item.get("rank") or 0) for item in top_k]
        if ranks != list(range(1, 101)):
            raise ProbeContractError(f"{surface_id}: per-query ranks mismatch")
        seen_docs: set[str] = set()
        for item in top_k:
            doc_id = str(item.get("doc_id"))
            if not doc_id or doc_id == "None":
                raise ProbeContractError(f"{surface_id}: top_k missing doc_id")
            if doc_id in seen_docs:
                raise ProbeContractError(f"{surface_id}: duplicate top_k doc {qid}/{doc_id}")
            seen_docs.add(doc_id)
            finite_float(item.get("score"), f"{surface_id}: top_k score {qid}/{doc_id}")
            expected_gain = qrels[qid].get(doc_id, 0)
            actual_gain = exact_int(item.get("relevance"), f"{surface_id}: top_k relevance {qid}/{doc_id}")
            if actual_gain != expected_gain:
                raise ProbeContractError(f"{surface_id}: top_k relevance mismatch for {qid}/{doc_id}")
        rows[qid] = row
    if set(rows) != set(qrels):
        raise ProbeContractError(f"{surface_id}: per-query qid set mismatch")
    return rows


def validate_metrics_json(path: Path, surface_id: str, dataset: str, qrels_meta: dict[str, Any], expected_package: str) -> dict[str, float]:
    payload = read_json(path)
    if payload.get("schema") != METRICS_SCHEMA or payload.get("dataset") != dataset:
        raise ProbeContractError(f"{surface_id}: metrics schema/dataset mismatch")
    if payload.get("artifact") != expected_package:
        raise ProbeContractError(f"{surface_id}: metrics artifact mismatch")
    cfg = payload.get("config", {})
    if cfg.get("bits") != [3] or int(cfg.get("quantizer_seed") or 0) != Q3_SEED or int(cfg.get("top_k") or 0) != 100:
        raise ProbeContractError(f"{surface_id}: metrics config mismatch")
    inputs = payload.get("inputs", {})
    if inputs.get("qrels_path") != qrels_meta["path"] or inputs.get("qrels_sha256") != qrels_meta["sha256"]:
        raise ProbeContractError(f"{surface_id}: metrics qrels binding mismatch")
    if int(inputs.get("queries") or 0) != qrels_meta["qid_count"] or int(inputs.get("relevant_pairs") or 0) != qrels_meta["qrel_count"]:
        raise ProbeContractError(f"{surface_id}: metrics qrels counts mismatch")
    rows = payload.get("rows", [])
    q3 = [row for row in rows if row.get("method") == Q3_METHOD and int(row.get("bits") or 0) == 3]
    if len(q3) != 1:
        raise ProbeContractError(f"{surface_id}: missing/duplicate q3 metrics row")
    quality = q3[0].get("quality", {})
    return {
        "ndcg_at_10": finite_float(quality.get("ndcg_at_10"), f"{surface_id}: ndcg_at_10"),
        "recall_at_100": finite_float(quality.get("recall_at_100"), f"{surface_id}: recall_at_100"),
    }


def validate_metrics_tsv(path: Path, surface_id: str, dataset: str, expected: dict[str, float], expected_package: str) -> None:
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        rows = list(reader)
    q3 = [row for row in rows if row.get("dataset") == dataset and row.get("method") == Q3_METHOD and row.get("bits") == "3"]
    if len(q3) != 1:
        raise ProbeContractError(f"{surface_id}: metrics TSV missing/duplicate q3 row")
    if "artifact" in q3[0] and q3[0].get("artifact") != expected_package:
        raise ProbeContractError(f"{surface_id}: metrics TSV artifact mismatch")
    for key in ("ndcg_at_10", "recall_at_100"):
        value = finite_float(q3[0][key], f"{surface_id}: metrics TSV {key}")
        if not math.isclose(value, expected[key], rel_tol=1e-6, abs_tol=1e-6):
            raise ProbeContractError(f"{surface_id}: metrics TSV {key} mismatch")


def rank_score(row: dict[str, Any], doc_id: str) -> tuple[int | None, float | None]:
    for item in row["top_k"]:
        if str(item.get("doc_id")) == doc_id:
            return int(item.get("rank") or 0), float(item.get("score") or 0.0)
    return None, None


def validate_gate(manifest_path: Path) -> dict[str, Any]:
    base = REPO_ROOT
    manifest = validate_manifest(manifest_path, require_outputs_absent=False, allow_pending_runtime_bindings=False)
    metrics_by_surface: dict[str, dict[str, dict[str, float]]] = {}
    per_query_by_surface: dict[str, dict[str, dict[str, dict[str, Any]]]] = {}
    for command in manifest["commands"]:
        surface_id = command["surface_id"]
        label = command["label"]
        dataset = command["dataset"]
        surface = manifest["surfaces"][surface_id]
        qrels = read_qrels(repo_path(surface["qrels"]["path"], base))
        outputs = {key: repo_path(value, base) for key, value in command["outputs"].items()}
        missing = [str(path) for path in outputs.values() if not path.is_file()]
        if missing:
            raise ProbeContractError(f"{surface_id}.{label}: missing outputs {missing}")
        expected_package = command["package"]
        metrics = validate_metrics_json(outputs["metrics_json"], surface_id, dataset, surface["qrels"], expected_package)
        validate_metrics_tsv(outputs["metrics_tsv"], surface_id, dataset, metrics, expected_package)
        per_query = load_per_query(outputs["per_query_jsonl"], surface_id, dataset, qrels, expected_package)
        metrics_by_surface.setdefault(surface_id, {})[label] = metrics
        per_query_by_surface.setdefault(surface_id, {})[label] = per_query
    ndcg_deltas = []
    exits = 0
    boundary_exits = 0
    boundary_entries = 0
    boundary_margins = []
    substitutions = 0
    for surface_id, surface in manifest["surfaces"].items():
        bucket, dataset = EXPECTED_SURFACES[surface_id]
        qrels = read_qrels(repo_path(surface["qrels"]["path"], base))
        anchor = per_query_by_surface[surface_id]["anchor"]
        candidate = per_query_by_surface[surface_id]["candidate"]
        ndcg_delta = metrics_by_surface[surface_id]["candidate"]["ndcg_at_10"] - metrics_by_surface[surface_id]["anchor"]["ndcg_at_10"]
        if bucket in {"frontier_context_q3", "top10_retention_q3"}:
            ndcg_deltas.append(ndcg_delta)
            if ndcg_delta < 0:
                raise ProbeContractError(f"{surface_id}: q3 nDCG delta negative")
            for qid, docs in qrels.items():
                for doc_id in docs:
                    anchor_rank, _anchor_score = rank_score(anchor[qid], doc_id)
                    candidate_rank, _candidate_score = rank_score(candidate[qid], doc_id)
                    if anchor_rank is not None and anchor_rank <= 10 and (candidate_rank is None or candidate_rank > 10):
                        exits += 1
                        entered_nonrel = any(int(item.get("rank") or 0) <= 10 and str(item.get("doc_id")) not in docs for item in candidate[qid]["top_k"])
                        if entered_nonrel:
                            substitutions += 1
        elif bucket == "nf_boundary80_100_docretention_q3":
            recall_delta = metrics_by_surface[surface_id]["candidate"]["recall_at_100"] - metrics_by_surface[surface_id]["anchor"]["recall_at_100"]
            if recall_delta < 0:
                raise ProbeContractError(f"{surface_id}: recall@100 delta negative")
            for qid, docs in qrels.items():
                for doc_id in docs:
                    anchor_rank, _anchor_score = rank_score(anchor[qid], doc_id)
                    candidate_rank, candidate_score = rank_score(candidate[qid], doc_id)
                    if anchor_rank is None or not (80 <= anchor_rank <= 100):
                        raise ProbeContractError(f"{surface_id}: boundary anchor rank outside 80..100 for {qid}/{doc_id}")
                    if candidate_rank is None or candidate_rank > 100:
                        boundary_exits += 1
                    else:
                        boundary_entries += 1
                    neg_scores = [
                        (abs(float(item.get("score") or 0.0) - float(candidate_score or 0.0)), abs(int(item.get("rank") or 0) - int(candidate_rank or 101)), int(item.get("rank") or 0), str(item.get("doc_id")), float(item.get("score") or 0.0))
                        for item in candidate[qid]["top_k"]
                        if str(item.get("doc_id")) not in docs
                    ]
                    if not neg_scores:
                        raise ProbeContractError(f"{surface_id}: no candidate negatives for margin")
                    nearest = sorted(neg_scores)[0]
                    boundary_margins.append(float(candidate_score or 0.0) - nearest[-1])
    if substitutions != 0:
        raise ProbeContractError("metric-aligned substitution failures nonzero")
    if boundary_exits > boundary_entries:
        raise ProbeContractError("NF boundary exits exceed entries")
    if boundary_margins and median(boundary_margins) < 0:
        raise ProbeContractError("median boundary margin negative")
    macro = sum(ndcg_deltas) / len(ndcg_deltas)
    if macro < 0:
        raise ProbeContractError("frontier/top10 macro q3 nDCG delta negative")
    return {
        "schema": "eos.q3_r6_probe_gate_summary.v1",
        "surface_count": 7,
        "command_count": 14,
        "output_count": 42,
        "frontier_top10_macro_ndcg_delta": macro,
        "relevant_top10_exits": exits,
        "nf_boundary_exits": boundary_exits,
        "nf_boundary_entries": boundary_entries,
        "median_boundary_margin": median(boundary_margins) if boundary_margins else None,
        "substitution_failures": substitutions,
        "passed": True,
    }


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.preflight_only:
        validate_manifest(args.probe_manifest, require_outputs_absent=True, allow_pending_runtime_bindings=args.allow_pending_runtime_bindings)
        print("q3 R6 probe preflight: OK")
        return
    summary = validate_gate(args.probe_manifest)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
