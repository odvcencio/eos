"""Synthetic, local-only tests for :mod:`aoqt_heldout_gate`.

The evaluator used here is a tiny executable written into a temporary
directory.  It is never the EOS binary and it consumes only synthetic qrels,
corpus, and query files.  No official qrels result or official evaluator is
read by this test module.
"""

from __future__ import annotations

import copy
import json
import math
import os
import stat
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))

import aoqt_heldout_gate as gate  # noqa: E402


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_raw(path: Path, text: str, *, executable: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    if executable:
        path.chmod(path.stat().st_mode | stat.S_IXUSR)


def record(path: Path) -> dict[str, object]:
    return {"path": str(path.resolve()), "sha256": gate.sha256_file(path), "bytes": path.stat().st_size}


MOCK_EVALUATOR = r'''#!/usr/bin/env python3
import hashlib
import json
import math
import os
import shutil
import sys
from pathlib import Path

MODE = os.environ.get("MOCK_MODE", "normal")
RUNTIME_SCHEMA = "eos.aoqt.native_eval_runtime_binding.v1"
RUNTIME_PRODUCER = "eos-native-runtime-resolved-inputs"

def arg(name):
    try:
        return sys.argv[sys.argv.index(name) + 1]
    except (ValueError, IndexError):
        raise SystemExit("missing " + name)

def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()

def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()

def file_digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def qrels(path):
    out = {}
    for raw in path.read_text().splitlines():
        if not raw.strip() or raw.lower().startswith("query-id"):
            continue
        p = raw.split("\t")
        if len(p) == 3:
            q, d, rel = p
        else:
            q, _, d, rel = p
        out.setdefault(q, {})[d] = float(rel)
    return out

def quality(ranking, rels):
    # EOS retrieval_eval.go uses linear positive relevance gains.
    positive = [v for v in rels.values() if v > 0]
    ideal = sorted(positive, reverse=True)[:10]
    idcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(ideal))
    dcg = sum(rel / math.log2(rank + 1) for rank, _d, rel in ranking if rank <= 10 and rel > 0)
    hit = sum(1 for rank, d, rel in ranking if rank <= 100 and rel > 0 and d in rels)
    return {"ndcg_at_10": dcg / idcg if idcg else 0.0, "recall_at_100": hit / len(positive) if positive else 0.0}

def ranking(qid, rels, relevant_rank, short=False):
    n = 119 if short else 120
    docs = [f"doc-{qid}-{i}" for i in range(1, n + 1)]
    rel_doc = next(d for d, v in rels.items() if v > 0)
    if rel_doc in docs:
        docs.remove(rel_doc)
        docs.append(f"filler-{qid}")
    if len(docs) != n or not 1 <= relevant_rank <= n:
        raise SystemExit("fixture ranking length/rank drift")
    docs[relevant_rank - 1] = rel_doc
    return [(i + 1, d, float(rels.get(d, 0.0))) for i, d in enumerate(docs)]

def runtime_binding(binding, frozen, package, dataset_dir, qpath, outputs):
    role = binding["role"]
    domain = binding["domain"]
    dataset_dir = dataset_dir.resolve()
    package = package.resolve()
    qpath = qpath.resolve()
    rels_by_qid = qrels(qpath)
    frozen_workload = frozen["workload"]
    frozen_approved = frozen["approved_workload"]
    descriptor = frozen_approved["descriptor"]
    corpus = dataset_dir / "corpus.jsonl"
    queries = dataset_dir / "queries.jsonl"
    runtime = {
        "schema": RUNTIME_SCHEMA,
        "producer": RUNTIME_PRODUCER,
        "gate_id": binding["gate_id"],
        "role": role,
        "domain": domain,
        "split": "test",
        "nonce": os.environ["EOS_AOQT_GATE_NONCE"],
        "frozen_manifest_sha256": os.environ["EOS_AOQT_FROZEN_MANIFEST_SHA256"],
        "binary_path": Path(sys.argv[0]).resolve().as_posix(),
        "binary_sha256": file_digest(Path(sys.argv[0]).resolve()),
        "argv": [str(Path(sys.argv[0]).resolve()), *sys.argv[1:]],
        "argv_sha256": digest([str(Path(sys.argv[0]).resolve()), *sys.argv[1:]]),
        "cwd": Path.cwd().resolve().as_posix(),
        "dimension": 384,
        "score_mode": "turboquant_ip_prepared",
        "package_mode": "native_mll_sibling",
        "package": {"artifact_path": package.as_posix(), "artifact_sha256": file_digest(package), "package_manifest_path": None, "package_manifest_sha256": None, "roles": [], "roles_sha256": digest([])},
        "dataset": {"dataset_id": domain, "dataset_dir": dataset_dir.as_posix(), "manifest_path": (dataset_dir / "manifest.json").as_posix(), "manifest_sha256": file_digest(dataset_dir / "manifest.json"), "corpus_path": corpus.as_posix(), "corpus_sha256": file_digest(corpus), "queries_path": queries.as_posix(), "queries_sha256": file_digest(queries)},
        "qrels": {"path": qpath.as_posix(), "sha256": file_digest(qpath), "qid_set_sha256": digest(sorted(rels_by_qid)), "query_count": len(rels_by_qid), "qrels_pair_count": sum(len(rels) for rels in rels_by_qid.values()), "relevant_pair_count": sum(1 for rels in rels_by_qid.values() for value in rels.values() if value > 0)},
        "workload": {"path": str(Path(frozen_workload["path"]).resolve()), "sha256": frozen_workload["sha256"], "descriptor_sha256": frozen_approved["sha256"], "qid_set_sha256_by_domain": descriptor["qid_set_sha256_by_domain"], "query_count_by_domain": descriptor["query_count_by_domain"]},
        "config": {"dimension": 384, "bits": [3, 5], "seed": 5581486560434873699, "top_k": 120, "per_query_top_k": 120, "batch_size": 64, "max_docs": 0, "max_queries": 0, "split": "test", "score_mode": "turboquant_ip_prepared", "package_mode": "native_mll_sibling", "rerank_overfetch": [], "rerank_bits": 0, "allow_research_only_aoqt": role == "candidate"},
        "outputs": {key: str(outputs[key].resolve()) for key in ("metrics", "metrics_tsv", "per_query")},
    }
    runtime["binding_sha256"] = digest(runtime)
    return runtime

def main():
    metrics = Path(arg("--metrics-json"))
    tsv = Path(arg("--metrics-tsv"))
    perq = Path(arg("--per-query-jsonl"))
    qpath = Path(arg("--qrels"))
    domain = arg("--dataset")
    package = Path(sys.argv[-2])
    dataset_dir = Path(sys.argv[-1])
    binding = json.loads(os.environ["EOS_AOQT_GATE_BINDING_JSON"])
    frozen = json.loads(Path(os.environ["EOS_AOQT_FROZEN_MANIFEST_PATH"]).read_text())
    rels_by_qid = qrels(qpath)
    if MODE == "copy-old":
        old = Path(os.environ["MOCK_COPY_FROM"])
        stem = f"{binding['role']}.{domain}"
        shutil.copy2(old / f"{stem}.metrics.json", metrics)
        shutil.copy2(old / f"{stem}.metrics.tsv", tsv)
        shutil.copy2(old / f"{stem}.per-query.jsonl", perq)
        return
    if MODE == "forged":
        metrics.write_text(json.dumps({"schema": "manta.embedding_turboquant_retrieval_metrics.v1", "dataset": domain}) + "\n")
        tsv.write_text("not-native\n")
        perq.write_text("{}\n")
        return
    bad_nf = MODE == "missing-nf" and domain == "nfcorpus"
    candidate = binding["role"] == "candidate"
    regression = MODE == "regression" and candidate
    boundary_regression = MODE == "boundary-regression" and candidate and domain == "nfcorpus"
    outputs = {"metrics": metrics, "metrics_tsv": tsv, "per_query": perq}
    runtime = runtime_binding(binding, frozen, package, dataset_dir, qpath, outputs)
    if MODE == "echo-gate-binding":
        runtime = binding
    elif MODE == "bad-runtime-binding":
        runtime["binary_sha256"] = "0" * 64
    elif MODE == "drop-research-loader" and candidate:
        runtime["config"].pop("allow_research_only_aoqt", None)
        runtime["binding_sha256"] = digest(runtime)
    elif MODE == "alter-research-loader" and candidate:
        runtime["config"]["allow_research_only_aoqt"] = not runtime["config"]["allow_research_only_aoqt"]
        runtime["binding_sha256"] = digest(runtime)
    rows = []
    aggregates = {3: [], 5: []}
    dense_values = []
    for qid, rels in sorted(rels_by_qid.items()):
        if domain == "nfcorpus" and qid.endswith("-q0"):
            q3_rank = 1 if candidate else 90
            if boundary_regression:
                q3_rank = 110
        else:
            q3_rank = 1 if candidate and qid.endswith("-q0") else 2
        if regression:
            q3_rank = 3
        q5_rank = 1
        dense_rank = 1
        compact3 = ranking(qid, rels, q3_rank, short=bad_nf)
        compact5 = ranking(qid, rels, q5_rank, short=bad_nf)
        dense = ranking(qid, rels, dense_rank, short=bad_nf)
        q3 = quality(compact3, rels)
        q5 = quality(compact5, rels)
        dq = quality(dense, rels)
        relevant_count = sum(1 for v in rels.values() if v > 0)
        if MODE == "full-per-query-relevant-count":
            relevant_count = len(rels)
        dense_values.append(dq)
        aggregates[3].append(q3)
        aggregates[5].append(q5)
        for bits, compact, q in ((3, compact3, q3), (5, compact5, q5)):
            row = {
                "schema": "manta.embedding_turboquant_retrieval_per_query.v1", "dataset": domain, "query_id": qid, "method": f"turboquant_ip_b{bits}", "bits": bits, "scoring_surface": "turboquant_ip_prepared", "quantizer_seed": 5581486560434873699, "relevant_count": relevant_count, "first_relevant_rank": next(rank for rank, d, rel in compact if d in rels and rel > 0), "quality": q, "dense_quality": dq,
                "top_k": [{"rank": r, "doc_id": d, "score": float(1.0 / r), "relevance": rel} for r, d, rel in compact], "dense_top_k": [{"rank": r, "doc_id": d, "score": float(1.0 / r), "relevance": rel} for r, d, rel in dense], "gate_binding": binding, "runtime_binding": runtime,
            }
            if MODE != "omit-boundary" and not bad_nf and domain == "nfcorpus" and qid in frozen["approved_workload"]["descriptor"]["nfcorpus_boundary_qids"]:
                compact_fingerprint = [{"rank": r, "doc_id": d, "score": float(1.0 / r)} for r, d, _rel in compact]
                dense_fingerprint = [{"rank": r, "doc_id": d, "score": float(1.0 / r)} for r, d, _rel in dense]
                row["boundary_evidence"] = {"schema": "eos.aoqt.nfcorpus_boundary_evidence.v1", "qid": qid, "rank_window": [80, 120], "candidate_count": len(compact), "exit_rank": len(compact), "substitution_count": 0, "margin_to_exit": 1.0, "compact_top_k_sha256": digest(compact_fingerprint), "dense_top_k_sha256": digest(dense_fingerprint), "window": [{"rank": rank, "doc_id": compact[rank - 1][1], "score": float(1.0 / rank)} for rank in (80, 120)], "boundary_complete": True}
            if MODE == "omit-binding":
                row.pop("gate_binding")
                row.pop("runtime_binding")
            rows.append(row)
    def avg(values, key):
        return sum(item[key] for item in values) / len(values)
    dense = {"ndcg_at_10": avg(dense_values, "ndcg_at_10"), "recall_at_100": avg(dense_values, "recall_at_100")}
    native_rows = []
    for bits in (3, 5):
        q = {"ndcg_at_10": avg(aggregates[bits], "ndcg_at_10"), "recall_at_100": avg(aggregates[bits], "recall_at_100")}
        native_rows.append({"bits": bits, "method": f"turboquant_ip_b{bits}", "quality": q, "ndcg_at_10_delta": 0.0, "recall_at_100_delta": 0.0})
    corpus_count = sum(1 for line in (dataset_dir / "corpus.jsonl").read_text().splitlines() if line.strip())
    relevant_pairs = sum(1 for rels in rels_by_qid.values() for value in rels.values() if value > 0)
    if MODE == "full-metric-relevant-pairs":
        relevant_pairs = sum(len(rels) for rels in rels_by_qid.values())
    payload = {
        "schema": "manta.embedding_turboquant_retrieval_metrics.v1", "dataset": domain, "artifact": str(package.resolve()), "backend": "mock-native",
        "inputs": {"corpus_path": str((dataset_dir / "corpus.jsonl").resolve()), "corpus_sha256": file_digest(dataset_dir / "corpus.jsonl"), "queries_path": str((dataset_dir / "queries.jsonl").resolve()), "queries_sha256": file_digest(dataset_dir / "queries.jsonl"), "qrels_path": str(qpath.resolve()), "qrels_sha256": file_digest(qpath), "workload_sha256": frozen["workload"]["sha256"], "approved_workload_sha256": frozen["approved_workload"]["sha256"], "documents": corpus_count, "queries": len(rels_by_qid), "relevant_pairs": relevant_pairs, "scored_pairs": 120 * len(rels_by_qid)},
        "config": {"dimension": 384, "split": "test", "score_mode": "turboquant_ip_prepared", "package_mode": "native_mll_sibling", "batch_size": 64, "top_k": 120, "per_query_top_k": 120, "bits": [3, 5], "quantizer_seed": 5581486560434873699, "max_docs": 0, "max_queries": 0, "rerank_overfetch": [], "rerank_bits": 0, "allow_research_only_aoqt": candidate},
        "dense": {"quality": dense}, "rows": native_rows, "gate_binding": binding, "runtime_binding": runtime,
    }
    if MODE == "bad-qrels-binding":
        payload["inputs"]["qrels_sha256"] = "0" * 64
    if MODE == "omit-binding":
        payload.pop("gate_binding")
        payload.pop("runtime_binding")
    metrics.write_text(json.dumps(payload, sort_keys=True) + "\n")
    tsv.write_text("dataset\trow\tbits\tmethod\tplaceholder\n" + f"{domain}\tdense\t\tfloat32\n" + f"{domain}\tquantized\t3\tturboquant_ip_b3\n" + f"{domain}\tquantized\t5\tturboquant_ip_b5\n")
    perq.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n")
    if MODE == "mutate-qrels":
        with qpath.open("a") as handle:
            handle.write("mutated-qid\tmutated-doc\t1\n")

if __name__ == "__main__":
    main()
'''


class Fixture:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.inputs = self.root / "inputs"
        self.workload_root = self.root / "workload-root"
        self.output_root = self.root / "gate-runs"
        self.inputs.mkdir(parents=True)
        self.workload_root.mkdir()
        self.output_root.mkdir()
        self.plan_path = self.root / "gate-plan.json"
        self.frozen_path = self.root / "frozen.json"
        self.binary = self.inputs / "mock-eos"
        self.source_manifest = self.inputs / "source.json"
        self.approved = self.inputs / "approved-workload.json"
        self.workload = self.inputs / "workload.json"
        self.qrels = {domain: self.inputs / domain / "qrels.tsv" for domain in gate.DOMAINS}
        self.datasets = {domain: self.inputs / domain for domain in gate.DOMAINS}
        self.exclusions = {name: self.inputs / f"{name}.json" for name in gate.EXCLUSION_NAMES}
        self.exclusion_sources = {name: self.inputs / f"{name}.source.json" for name in gate.EXCLUSION_NAMES}
        self.anchor_package = self.inputs / "anchor" / "d384-anchor.mll"
        self.anchor_manifest = self.inputs / "anchor" / "d384-anchor.package.json"
        self.anchor_attestation = self.inputs / "anchor" / "d384-anchor.attestation.json"
        self.candidate_package = self.inputs / "candidate" / "d384-candidate.mll"
        self.candidate_manifest = self.inputs / "candidate" / "d384-candidate.package.json"
        self.candidate_attestation = self.inputs / "candidate" / "d384-candidate.attestation.json"
        self.candidate_sidecar = self.inputs / "candidate" / "d384-candidate.sidecar.json"
        self.plan: dict = {}
        self.qids = {domain: [f"{domain}-q0", f"{domain}-q1"] for domain in gate.DOMAINS}

    def transform(self) -> dict:
        return {"id": gate.AOQT_TOPOLOGY["id"], "dim": gate.DIMENSION, "stages": 8, "pairs_per_stage": 192, "angle_count": 1536, "pairings_sha256": gate.sha256_bytes(b"pairings"), "angles_sha256": gate.sha256_bytes(b"angles"), "angle_cap": 0.04, "max_angle_cap": 0.08}

    def package_manifest(self, role: str, package: Path, *, anchor: dict | None, sidecar_sha: str | None, transform: dict | None, embedding_space: str) -> dict:
        anchor_view = None if anchor is None else {"package_sha256": anchor["artifact_sha256"], "manifest_sha256": anchor["package_manifest_sha256"], "embedding_space_id": anchor["embedding_space_id"]}
        return {"schema": gate.PACKAGE_MANIFEST_SCHEMA, "role": role, "package_sha256": gate.sha256_file(package), "dimension": gate.DIMENSION, "embedding_space_id": embedding_space, "post_pool_transform": "none" if role == "anchor" else gate.AOQT_TOPOLOGY["id"], "research_only": role == "candidate", "topology": copy.deepcopy(gate.AOQT_TOPOLOGY), "transform": copy.deepcopy(transform), "anchor": anchor_view, "legal_scope": copy.deepcopy(gate.AOQT_LEGAL_SCOPE), "sidecar_sha256": sidecar_sha}

    def entries(self, paths: list[Path]) -> list[dict]:
        return sorted(({"name": path.name, **record(path)} for path in paths), key=lambda item: item["name"])

    def attestation(self, role: str, package: Path, manifest: Path, *, embedding_space: str, anchor: dict, sidecar: Path | None, transform: dict | None, dataset_binding: dict) -> dict:
        sidecar_record = record(sidecar) if sidecar else None
        siblings = self.entries([package, manifest] + ([sidecar] if sidecar else []))
        anchor_view = {"package_sha256": anchor["artifact_sha256"], "manifest_sha256": anchor["package_manifest_sha256"], "embedding_space_id": anchor["embedding_space_id"]}
        return {"schema": gate.PACKAGE_ATTESTATION_SCHEMA, "role": role, "package_sha256": gate.sha256_file(package), "manifest_sha256": gate.sha256_file(manifest), "dimension": gate.DIMENSION, "embedding_space_id": embedding_space, "post_pool_transform": "none" if role == "anchor" else gate.AOQT_TOPOLOGY["id"], "research_only": role == "candidate", "topology": copy.deepcopy(gate.AOQT_TOPOLOGY), "transform": copy.deepcopy(transform), "anchor": anchor_view, "legal_scope": copy.deepcopy(gate.AOQT_LEGAL_SCOPE), "sidecar": sidecar_record, "sibling_rollup": {"entries": siblings, "sha256": gate.sha256_json(siblings)}, "dataset_binding": copy.deepcopy(dataset_binding), "compatibility_digest": gate.sha256_bytes(b"fixture-compatibility"), "identity": {"artifact_sha256": gate.sha256_file(package), "package_manifest_sha256": gate.sha256_file(manifest), "embedding_space_id": embedding_space}}

    def build(self) -> "Fixture":
        write_raw(self.binary, MOCK_EVALUATOR, executable=True)
        write_json(self.source_manifest, {"schema": "fixture.source.v1", "source_id": "mock-native"})
        for domain in gate.DOMAINS:
            dataset = self.datasets[domain]
            corpus_ids = []
            for qid in self.qids[domain]:
                corpus_ids.extend([f"doc-{qid}-{index}" for index in range(1, 121)])
                corpus_ids.append(f"filler-{qid}")
                corpus_ids.append(f"rel-{qid}")
            write_raw(dataset / "corpus.jsonl", "".join(json.dumps({"_id": doc_id}) + "\n" for doc_id in corpus_ids))
            write_raw(dataset / "queries.jsonl", '{"_id":"q"}\n')
            write_json(dataset / "manifest.json", {"schema": gate.DATASET_SCHEMA, "domain": domain, "dataset_id": domain, "split": gate.HELDOUT_SPLIT, "corpus_sha256": gate.sha256_file(dataset / "corpus.jsonl"), "queries_sha256": gate.sha256_file(dataset / "queries.jsonl")})
            qrel_lines = []
            for qid in self.qids[domain]:
                qrel_lines.append(f"{qid}\trel-{qid}\t1")
                qrel_lines.append(f"{qid}\tdoc-{qid}-120\t0")
            write_raw(self.qrels[domain], "query-id\tcorpus-id\tscore\n" + "\n".join(qrel_lines) + "\n")
        qrels_records = {domain: record(path) for domain, path in self.qrels.items()}
        dataset_records = {domain: {"dataset_id": domain, "dataset_dir": str(self.datasets[domain]), "corpus": record(self.datasets[domain] / "corpus.jsonl"), "queries": record(self.datasets[domain] / "queries.jsonl"), "manifest": record(self.datasets[domain] / "manifest.json")} for domain in gate.DOMAINS}
        compatibility = gate.sha256_bytes(b"fixture-compatibility")
        approved_payload = {"schema": gate.APPROVED_WORKLOAD_SCHEMA, "gate_id": "fixture-gate", "descriptor_id": "fixture-approved", "split": gate.HELDOUT_SPLIT, "dimension": gate.DIMENSION, "domains": list(gate.DOMAINS), "query_ids_by_domain": copy.deepcopy(self.qids), "qid_set_sha256_by_domain": gate.qids_sha256_by_domain(self.qids), "query_count_by_domain": {domain: 2 for domain in gate.DOMAINS}, "qrels_sha256_by_domain": {domain: qrels_records[domain]["sha256"] for domain in gate.DOMAINS}, "qrels_query_count_by_domain": {domain: 2 for domain in gate.DOMAINS}, "relevant_pair_count_by_domain": {domain: 2 for domain in gate.DOMAINS}, "dataset_manifest_sha256_by_domain": {domain: dataset_records[domain]["manifest"]["sha256"] for domain in gate.DOMAINS}, "corpus_sha256_by_domain": {domain: dataset_records[domain]["corpus"]["sha256"] for domain in gate.DOMAINS}, "queries_sha256_by_domain": {domain: dataset_records[domain]["queries"]["sha256"] for domain in gate.DOMAINS}, "compatibility_digest": compatibility, "nfcorpus_boundary_qids": sorted(self.qids["nfcorpus"]), "nfcorpus_boundary_qids_sha256": gate.sha256_json(sorted(self.qids["nfcorpus"])), "nfcorpus_boundary_rank_window": [80, 120], "metric_surfaces": list(gate.SURFACES), "cutoffs": {"ndcg_at_10": 10, "recall_at_100": 100}, "turboquant": {"q3_bits": 3, "q5_bits": 5, "seed": gate.TURBOQUANT_SEED, "top_k": 120, "score_mode": "turboquant_ip_prepared", "package_mode": "native_mll_sibling"}}
        write_json(self.approved, approved_payload)
        approved_record = record(self.approved)
        workload_payload = {"schema": gate.WORKLOAD_SCHEMA, "gate_id": "fixture-gate", "workload_id": "fixture-workload", "descriptor_sha256": approved_record["sha256"], "split": gate.HELDOUT_SPLIT, "dimension": gate.DIMENSION, "query_ids_by_domain": copy.deepcopy(self.qids), "qid_set_sha256_by_domain": gate.qids_sha256_by_domain(self.qids), "query_count_by_domain": {domain: 2 for domain in gate.DOMAINS}, "qrels_sha256_by_domain": {domain: qrels_records[domain]["sha256"] for domain in gate.DOMAINS}, "dataset_manifest_sha256_by_domain": {domain: dataset_records[domain]["manifest"]["sha256"] for domain in gate.DOMAINS}, "corpus_sha256_by_domain": {domain: dataset_records[domain]["corpus"]["sha256"] for domain in gate.DOMAINS}, "queries_sha256_by_domain": {domain: dataset_records[domain]["queries"]["sha256"] for domain in gate.DOMAINS}, "compatibility_digest": compatibility}
        write_json(self.workload, workload_payload)
        workload_record = record(self.workload)
        for name in gate.EXCLUSION_NAMES:
            source = self.exclusion_sources[name]
            write_json(source, {"source": name, "fixture": True})
            source_map = {str(source.resolve()): gate.sha256_file(source)}
            qids_by_domain = copy.deepcopy(self.qids) if name == "official-test" else {domain: [f"excluded-{name}-{domain}"] for domain in gate.DOMAINS}
            write_json(self.exclusions[name], {"schema": "eos.aoqt_stage2.exclusion_qids.v1", "name": name, "qids_by_dataset": qids_by_domain, "source_sha256": gate.sha256_json(source_map), "source_sha256_by_file": source_map})
        anchor_space = gate.sha256_bytes(b"fixture-anchor-space")
        candidate_space = gate.sha256_bytes(b"fixture-candidate-space")
        self.anchor_package.parent.mkdir(parents=True, exist_ok=True)
        self.candidate_package.parent.mkdir(parents=True, exist_ok=True)
        write_raw(self.anchor_package, "synthetic-anchor-package")
        write_raw(self.candidate_package, "synthetic-candidate-package")
        anchor_manifest = self.package_manifest("anchor", self.anchor_package, anchor=None, sidecar_sha=None, transform=None, embedding_space=anchor_space)
        write_json(self.anchor_manifest, anchor_manifest)
        anchor_identity = {"artifact_sha256": gate.sha256_file(self.anchor_package), "package_manifest_sha256": gate.sha256_file(self.anchor_manifest), "embedding_space_id": anchor_space}
        transform = self.transform()
        write_json(self.candidate_sidecar, {"schema": gate.SIDECAR_SCHEMA, "kind": gate.AOQT_TOPOLOGY["id"], "dim": 384, "stages": 8, "pairs_per_stage": 192, "angle_count": 1536, "angle_cap": 0.04, "max_angle_cap": 0.08, "pairings_sha256": transform["pairings_sha256"], "angles_sha256": transform["angles_sha256"], "anchor_package_sha256": anchor_identity["artifact_sha256"], "anchor_manifest_sha256": anchor_identity["package_manifest_sha256"], "anchor_embedding_space_id": anchor_space, "legal_scope": copy.deepcopy(gate.AOQT_LEGAL_SCOPE)})
        write_json(self.candidate_manifest, self.package_manifest("candidate", self.candidate_package, anchor=anchor_identity, sidecar_sha=gate.sha256_file(self.candidate_sidecar), transform=transform, embedding_space=candidate_space))
        data_binding = {"dataset_manifest_sha256_by_domain": {domain: dataset_records[domain]["manifest"]["sha256"] for domain in gate.DOMAINS}, "corpus_sha256_by_domain": {domain: dataset_records[domain]["corpus"]["sha256"] for domain in gate.DOMAINS}, "queries_sha256_by_domain": {domain: dataset_records[domain]["queries"]["sha256"] for domain in gate.DOMAINS}, "qrels_sha256_by_domain": {domain: qrels_records[domain]["sha256"] for domain in gate.DOMAINS}}
        write_json(self.anchor_attestation, self.attestation("anchor", self.anchor_package, self.anchor_manifest, embedding_space=anchor_space, anchor=anchor_identity, sidecar=None, transform=None, dataset_binding=data_binding))
        write_json(self.candidate_attestation, self.attestation("candidate", self.candidate_package, self.candidate_manifest, embedding_space=candidate_space, anchor=anchor_identity, sidecar=self.candidate_sidecar, transform=transform, dataset_binding=data_binding))
        anchor_siblings = self.entries([self.anchor_package, self.anchor_manifest])
        candidate_siblings = self.entries([self.candidate_package, self.candidate_manifest, self.candidate_sidecar])
        anchor_record = {**record(self.anchor_package), "manifest": record(self.anchor_manifest), "attestation": record(self.anchor_attestation), "sibling_rollup_sha256": gate.sha256_json(anchor_siblings)}
        candidate_record = {**record(self.candidate_package), "manifest": record(self.candidate_manifest), "attestation": record(self.candidate_attestation), "sibling_rollup_sha256": gate.sha256_json(candidate_siblings)}
        self.plan = {"schema": gate.PLAN_SCHEMA, "gate_id": "fixture-gate", "candidate_id": "fixture-candidate", "dimension": gate.DIMENSION, "thresholds": copy.deepcopy(gate.THRESHOLDS), "evaluation": {"executed": False, "official": False}, "anchor": {"id": "fixture-anchor", "package": anchor_record}, "candidate": {"id": "fixture-candidate", "package": candidate_record}, "source": {"manifest": record(self.source_manifest), "binary": record(self.binary), "cwd": str(self.root), "dataset_by_domain": dataset_records, "workload": workload_record, "approved_workload": approved_record}, "approved_workload": approved_record, "qrels": qrels_records, "exclusions": [{"kind": "qid_only", "name": name, "manifest": record(self.exclusions[name])} for name in gate.EXCLUSION_NAMES], "workload_root": str(self.workload_root), "output_root": str(self.output_root)}
        write_json(self.plan_path, self.plan)
        return self

    def freeze(self) -> dict:
        return gate._freeze_manifest_for_tests(self.plan_path, self.frozen_path)

    def run_kwargs(self, frozen: dict) -> dict:
        return {"expected_frozen_manifest_sha256": frozen["file_sha256"], "expected_workload_manifest_sha256": frozen["manifest"]["approved_workload"]["sha256"], "expected_binary_sha256": frozen["manifest"]["provenance"]["binary_sha256"], "expected_anchor_package_sha256": frozen["manifest"]["expected_anchor_identity"]["artifact_sha256"], "expected_anchor_manifest_sha256": frozen["manifest"]["expected_anchor_identity"]["package_manifest_sha256"], "expected_anchor_embedding_space_id": frozen["manifest"]["expected_anchor_identity"]["embedding_space_id"], "test_mode": True, "timeout_seconds": 30}


class AOQTHeldoutGateTest(unittest.TestCase):
    def fixture(self) -> Fixture:
        return Fixture(Path(tempfile.mkdtemp())).build()

    def test_corpus_metadata_is_accepted_and_ignored_for_ids(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "corpus.jsonl"
            write_raw(
                path,
                "\n".join(
                    [
                        json.dumps({"_id": "doc-a", "title": "A", "text": "alpha", "metadata": {"source": "fixture"}}),
                        json.dumps({"_id": "doc-b", "metadata": {"nested": {"ignored": True}}}),
                    ]
                )
                + "\n",
            )
            self.assertEqual(gate.parse_corpus_ids(path, "fixture.corpus"), {"doc-a", "doc-b"})

    def test_corpus_unknown_extra_field_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "corpus.jsonl"
            write_raw(path, json.dumps({"_id": "doc-a", "metadata": {}, "unexpected": True}) + "\n")
            with self.assertRaisesRegex(gate.GateError, "corpus rows must be native"):
                gate.parse_corpus_ids(path, "fixture.corpus")

    def test_corpus_bad_and_duplicate_ids_are_still_rejected(self) -> None:
        cases = {
            "nonobject": "[\"doc-a\"]\n",
            "missing": json.dumps({"metadata": {}}) + "\n",
            "nonstring": json.dumps({"_id": 7, "metadata": {}}) + "\n",
            "empty": json.dumps({"_id": "", "metadata": {}}) + "\n",
            "whitespace": json.dumps({"_id": "doc a", "metadata": {}}) + "\n",
            "duplicate": "\n".join([json.dumps({"_id": "doc-a", "metadata": {}}), json.dumps({"_id": "doc-a", "metadata": {}})]) + "\n",
        }
        for name, contents in cases.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as raw:
                path = Path(raw) / "corpus.jsonl"
                write_raw(path, contents)
                with self.assertRaisesRegex(gate.GateError, "corpus rows|document identity|_id"):
                    gate.parse_corpus_ids(path, f"fixture.{name}.corpus")

    def test_qrels_document_membership_failure_remains_after_metadata_acceptance(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            corpus = root / "corpus.jsonl"
            queries = root / "queries.jsonl"
            qrels_path = root / "qrels.tsv"
            package = root / "package.mll"
            metrics = root / "metrics.json"
            tsv = root / "metrics.tsv"
            per_query = root / "per-query.jsonl"
            write_raw(corpus, json.dumps({"_id": "doc-present", "metadata": {"accepted": True}}) + "\n")
            write_raw(queries, json.dumps({"_id": "query-a"}) + "\n")
            write_raw(qrels_path, "query-id\tcorpus-id\tscore\nquery-a\tdoc-missing\t1\n")
            write_raw(package, "package")
            write_raw(tsv, "dataset\trow\nfixture\tdense\n")
            write_raw(per_query, "{}\n")
            qrels = gate.parse_qrels(qrels_path, "fixture.qrels")
            gate_binding = {"schema": gate.GATE_BINDING_SCHEMA, "gate_id": "fixture-gate"}
            gate_binding["binding_sha256"] = gate.digest_without(gate_binding, "binding_sha256")
            runtime_binding = {"schema": gate.RUNTIME_BINDING_SCHEMA, "producer": "eos-native-runtime-resolved-inputs"}
            runtime_binding["binding_sha256"] = gate.digest_without(runtime_binding, "binding_sha256")
            frozen_info = {
                "source": {
                    "dataset_by_domain": {
                        "fiqa": {
                            "corpus": {"path": str(corpus.resolve()), "sha256": gate.sha256_file(corpus)},
                            "queries": {"path": str(queries.resolve()), "sha256": gate.sha256_file(queries)},
                        }
                    }
                },
                "qrels": {"fiqa": {**qrels, "path": str(qrels_path.resolve())}},
                "workload": {"sha256": "a" * 64},
                "approved_workload": {"sha256": "b" * 64},
                "anchor": {"path": str(package.resolve())},
            }
            write_json(
                metrics,
                {
                    "schema": gate.NATIVE_METRICS_SCHEMA,
                    "dataset": "fiqa",
                    "artifact": str(package.resolve()),
                    "backend": "fixture",
                    "inputs": {
                        "corpus_path": str(corpus.resolve()),
                        "corpus_sha256": gate.sha256_file(corpus),
                        "queries_path": str(queries.resolve()),
                        "queries_sha256": gate.sha256_file(queries),
                        "qrels_path": str(qrels_path.resolve()),
                        "qrels_sha256": qrels["sha256"],
                        "workload_sha256": frozen_info["workload"]["sha256"],
                        "approved_workload_sha256": frozen_info["approved_workload"]["sha256"],
                        "queries": qrels["query_count"],
                        "relevant_pairs": qrels["qrels_pair_count"],
                    },
                    "config": {
                        "dimension": gate.DIMENSION,
                        "split": gate.HELDOUT_SPLIT,
                        "score_mode": "turboquant_ip_prepared",
                        "package_mode": "native_mll_sibling",
                        "batch_size": gate.BATCH_SIZE,
                        "top_k": gate.TOP_K,
                        "per_query_top_k": gate.TOP_K,
                        "bits": [gate.Q3_BITS, gate.Q5_BITS],
                        "quantizer_seed": gate.TURBOQUANT_SEED,
                        "max_docs": 0,
                        "max_queries": 0,
                        "rerank_overfetch": [],
                        "rerank_bits": 0,
                        "allow_research_only_aoqt": False,
                    },
                    "dense": {"quality": {"ndcg_at_10": 0.0, "recall_at_100": 0.0}},
                    "rows": [
                        {"bits": gate.Q3_BITS, "method": "turboquant_ip_b3"},
                        {"bits": gate.Q5_BITS, "method": "turboquant_ip_b5"},
                    ],
                    "gate_binding": gate_binding,
                    "runtime_binding": runtime_binding,
                },
            )
            with self.assertRaisesRegex(gate.GateError, "frozen qrels contain documents absent from the frozen corpus"):
                gate._parse_native_result(
                    metrics,
                    tsv,
                    per_query,
                    frozen_info=frozen_info,
                    role="anchor",
                    domain="fiqa",
                    gate_binding=gate_binding,
                    runtime_binding=runtime_binding,
                    production=False,
                )

    def test_run_executes_fresh_native_harness_and_passes_thresholds(self) -> None:
        f = self.fixture()
        frozen = f.freeze()
        with mock.patch.dict(os.environ, {"MOCK_MODE": "normal"}, clear=False):
            report = gate.run_harness(f.frozen_path, **f.run_kwargs(frozen))
        self.assertTrue(report["gate_pass"])
        self.assertTrue(report["evaluation_invoked_by_gate"])
        self.assertEqual(report["nonce"], Path(report["run_dir"]).name.removeprefix("aoqt-"))
        self.assertGreaterEqual(report["macro"]["q3_ndcg_at_10_delta"], gate.THRESHOLDS["q3_macro_ndcg_at_10_delta_min"])
        self.assertEqual(len(report["commands"]), gate.RESULT_OUTPUT_COUNT)
        self.assertTrue(Path(report["attestation_path"]).is_file())

    def test_native_positive_relevance_counts_are_distinct_from_full_qrels_pairs(self) -> None:
        f = self.fixture()
        parsed = gate.parse_qrels(f.qrels["fiqa"], "fixture.fiqa.qrels")
        self.assertEqual(parsed["qrels_pair_count"], 4)
        self.assertEqual(parsed["relevant_pair_count"], 2)
        frozen = f.freeze()
        with mock.patch.dict(os.environ, {"MOCK_MODE": "normal"}, clear=False):
            report = gate.run_harness(f.frozen_path, **f.run_kwargs(frozen))
        metrics_path = Path(report["run_dir"]) / "anchor.fiqa.metrics.json"
        per_query_path = Path(report["run_dir"]) / "anchor.fiqa.per-query.jsonl"
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        self.assertEqual(metrics["inputs"]["relevant_pairs"], 2)
        rows = [json.loads(line) for line in per_query_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        self.assertEqual({row["relevant_count"] for row in rows}, {1})

    def test_native_metric_relevant_pairs_must_be_positive_only(self) -> None:
        f = self.fixture()
        frozen = f.freeze()
        with mock.patch.dict(os.environ, {"MOCK_MODE": "full-metric-relevant-pairs"}, clear=False):
            with self.assertRaisesRegex(gate.GateError, "native metrics workload counts mismatch"):
                gate.run_harness(f.frozen_path, **f.run_kwargs(frozen))

    def test_native_per_query_relevant_count_must_be_positive_only(self) -> None:
        f = self.fixture()
        frozen = f.freeze()
        with mock.patch.dict(os.environ, {"MOCK_MODE": "full-per-query-relevant-count"}, clear=False):
            with self.assertRaisesRegex(gate.GateError, "relevant-count mismatch"):
                gate.run_harness(f.frozen_path, **f.run_kwargs(frozen))

    def test_legacy_attest_import_is_disabled(self) -> None:
        with self.assertRaisesRegex(gate.GateError, "receipt import is disabled"):
            gate.attest_manifest(Path("unused"), Path("unused"), expected_frozen_manifest_sha256="0" * 64)

    def test_external_frozen_hash_rejects_rewrite_and_rehash(self) -> None:
        f = self.fixture()
        frozen = f.freeze()
        payload = json.loads(f.frozen_path.read_text())
        payload["candidate_id"] = "rewritten"
        payload["manifest_sha256"] = gate.digest_without(payload, "manifest_sha256")
        write_json(f.frozen_path, payload)
        with self.assertRaisesRegex(gate.GateError, "external frozen manifest sha256 trust-anchor mismatch"):
            gate.run_harness(f.frozen_path, **f.run_kwargs(frozen))

    def test_production_requires_known_anchor_identity(self) -> None:
        f = self.fixture()
        frozen = f.freeze()
        kwargs = f.run_kwargs(frozen)
        kwargs.update({"expected_anchor_package_sha256": "0" * 64, "test_mode": False})
        with self.assertRaisesRegex(gate.GateError, "not the pinned current D384 anchor"):
            gate.run_harness(f.frozen_path, **kwargs)

    def test_production_rejects_coherent_alternate_workload_against_fixed_registry(self) -> None:
        f = self.fixture()
        approved = json.loads(f.approved.read_text())
        workload = json.loads(f.workload.read_text())
        qrel_records = {}
        for domain, path in f.qrels.items():
            with path.open("a", encoding="utf-8") as handle:
                handle.write(f"{domain}-q-extra\trel-{domain}-q-extra\t1\n")
            parsed = gate.parse_qrels(path, f"test.{domain}.qrels")
            qrel_records[domain] = record(path)
            approved["query_ids_by_domain"][domain] = parsed["qids"]
            approved["qid_set_sha256_by_domain"][domain] = parsed["qid_set_sha256"]
            approved["query_count_by_domain"][domain] = parsed["query_count"]
            approved["qrels_sha256_by_domain"][domain] = parsed["sha256"]
            approved["qrels_query_count_by_domain"][domain] = parsed["query_count"]
            approved["relevant_pair_count_by_domain"][domain] = parsed["relevant_pair_count"]
            workload["query_ids_by_domain"][domain] = parsed["qids"]
            workload["qid_set_sha256_by_domain"][domain] = parsed["qid_set_sha256"]
            workload["query_count_by_domain"][domain] = parsed["query_count"]
            workload["qrels_sha256_by_domain"][domain] = parsed["sha256"]
        write_json(f.approved, approved)
        approved_record = record(f.approved)
        workload["descriptor_sha256"] = approved_record["sha256"]
        write_json(f.workload, workload)
        f.plan["qrels"] = qrel_records
        f.plan["approved_workload"] = approved_record
        f.plan["source"]["approved_workload"] = approved_record
        f.plan["source"]["workload"] = record(f.workload)
        write_json(f.plan_path, f.plan)
        with mock.patch.object(gate, "require_elf_executable"):
            with self.assertRaisesRegex(gate.GateError, "fixed trusted registry artifact|registry qrels identity/count mismatch|official-test qids must equal"):
                gate.validate_plan(f.plan_path, production=True)

    def test_synthetic_production_registry_accepts_exact_official_population(self) -> None:
        f = self.fixture()
        qrels = {domain: gate.parse_qrels(path, f"synthetic.{domain}.qrels") for domain, path in f.qrels.items()}
        datasets = {
            domain: gate.normalize_dataset_record(
                {
                    "dataset_id": domain,
                    "dataset_dir": str(f.datasets[domain]),
                    "corpus": record(f.datasets[domain] / "corpus.jsonl"),
                    "queries": record(f.datasets[domain] / "queries.jsonl"),
                    "manifest": record(f.datasets[domain] / "manifest.json"),
                },
                f.root,
                domain,
                f"synthetic.{domain}.dataset",
            )
            for domain in gate.DOMAINS
        }
        approved = gate.validate_approved_workload(f.approved, gate_id="fixture-gate", qrels=qrels, datasets=datasets, label="synthetic.approved")
        exclusions = {
            name: gate.normalize_exclusion_record(
                {"kind": "qid_only", "name": name, "manifest": record(f.exclusions[name])},
                f.root,
                name,
                approved["query_ids_by_domain"],
                f"synthetic.exclusions.{name}",
            )
            for name in gate.EXCLUSION_NAMES
        }
        registry = {
            "schema": gate.TRUSTED_WORKLOAD_REGISTRY_SCHEMA,
            "status": "active",
            "domains": {
                domain: {
                    "qrels_path_suffix": str(f.qrels[domain].resolve()).lstrip("/"),
                    "qrels_sha256": qrels[domain]["sha256"],
                    "qid_set_sha256": qrels[domain]["qid_set_sha256"],
                    "query_count": qrels[domain]["query_count"],
                    "qrels_pair_count": qrels[domain]["qrels_pair_count"],
                    "relevant_pair_count": qrels[domain]["relevant_pair_count"],
                    "corpus_path_suffix": str((f.datasets[domain] / "corpus.jsonl").resolve()).lstrip("/"),
                    "corpus_sha256": datasets[domain]["corpus"]["sha256"],
                    "queries_path_suffix": str((f.datasets[domain] / "queries.jsonl").resolve()).lstrip("/"),
                    "queries_sha256": datasets[domain]["queries"]["sha256"],
                }
                for domain in gate.DOMAINS
            },
            "nfcorpus_boundary": {
                "source_path_suffix": str(f.exclusions["official-test"].resolve()).lstrip("/"),
                "source_sha256": exclusions["official-test"]["manifest_sha256"],
                "qid_set_sha256": gate.sha256_json(sorted(f.qids["nfcorpus"])),
                "qid_count": len(f.qids["nfcorpus"]),
                "rank_window": [80, 120],
            },
        }
        self.assertEqual({domain: qrels[domain]["qrels_pair_count"] for domain in gate.DOMAINS}, {domain: 4 for domain in gate.DOMAINS})
        self.assertEqual(approved["relevant_pair_count_by_domain"], {domain: 2 for domain in gate.DOMAINS})
        with mock.patch.object(gate, "PRODUCTION_TRUSTED_WORKLOAD_REGISTRY", registry):
            gate._validate_production_registry_binding(approved, qrels, datasets, exclusions, label="synthetic.production")

    def test_production_rejects_synthetic_json_package_fallback(self) -> None:
        f = self.fixture()
        frozen = f.freeze()
        with self.assertRaisesRegex(gate.GateError, r"native MLL artifacts|JSON fallback|missing \['package_manifest'\]"):
            gate.normalize_package_record(
                frozen["manifest"]["anchor"],
                f.frozen_path.parent,
                "frozen.anchor",
                expected_role="anchor",
                expected_anchor=frozen["manifest"]["expected_anchor_identity"],
                data_binding=gate._data_binding(frozen["manifest"]["approved_workload"]["descriptor"]),
                production=True,
            )

    def test_complete_qrels_rejects_self_approved_small_workload(self) -> None:
        f = self.fixture()
        approved = json.loads(f.approved.read_text())
        approved["query_ids_by_domain"]["fiqa"] = [f.qids["fiqa"][0]]
        approved["qid_set_sha256_by_domain"]["fiqa"] = gate.sha256_json(approved["query_ids_by_domain"]["fiqa"])
        approved["query_count_by_domain"]["fiqa"] = 1
        approved["qrels_query_count_by_domain"]["fiqa"] = 1
        write_json(f.approved, approved)
        f.plan["approved_workload"] = record(f.approved)
        f.plan["source"]["approved_workload"] = record(f.approved)
        write_json(f.plan_path, f.plan)
        with self.assertRaisesRegex(gate.GateError, "approved qid allowlist does not equal complete"):
            f.freeze()

    def test_qrels_changed_after_freeze_is_rejected(self) -> None:
        f = self.fixture()
        frozen = f.freeze()
        with f.qrels["fiqa"].open("a") as handle:
            handle.write("fiqa-q-extra\trel-fiqa-q-extra\t1\n")
        with self.assertRaisesRegex(gate.GateError, "frozen qrels binding changed|sha256 mismatch"):
            gate.run_harness(f.frozen_path, **f.run_kwargs(frozen))

    def test_frozen_plan_provenance_is_revalidated(self) -> None:
        f = self.fixture()
        frozen = f.freeze()
        with f.plan_path.open("a") as handle:
            handle.write("\n")
        with self.assertRaisesRegex(gate.GateError, "provenance plan sha256 mismatch"):
            gate.run_harness(f.frozen_path, **f.run_kwargs(frozen))

    def test_package_compatibility_digest_must_equal_approved_workload(self) -> None:
        f = self.fixture()
        payload = json.loads(f.candidate_attestation.read_text())
        payload["compatibility_digest"] = gate.sha256_bytes(b"wrong-compatibility")
        write_json(f.candidate_attestation, payload)
        f.plan["candidate"]["package"]["attestation"] = record(f.candidate_attestation)
        write_json(f.plan_path, f.plan)
        with self.assertRaisesRegex(gate.GateError, "compatibility digest mismatch"):
            f.freeze()

    def test_package_pair_allows_shared_anchor_artifact_with_distinct_xpkg(self) -> None:
        anchor_identity = {
            "artifact_sha256": "a" * 64,
            "package_manifest_sha256": "b" * 64,
            "embedding_space_id": "c" * 64,
        }
        candidate_identity = {
            "artifact_sha256": "a" * 64,
            "package_manifest_sha256": "d" * 64,
            "embedding_space_id": "c" * 64,
        }
        gate.validate_package_pair(
            {"package_identity": anchor_identity, "embedding_space_id": "c" * 64},
            {"package_identity": candidate_identity, "anchor_identity": anchor_identity, "embedding_space_id": "c" * 64},
        )

    def test_package_pair_rejects_reused_xpkg_identity(self) -> None:
        identity = {
            "artifact_sha256": "a" * 64,
            "package_manifest_sha256": "b" * 64,
            "embedding_space_id": "c" * 64,
        }
        with self.assertRaisesRegex(gate.GateError, "package identities must be distinct"):
            gate.validate_package_pair(
                {"package_identity": identity, "embedding_space_id": "c" * 64},
                {"package_identity": dict(identity), "anchor_identity": identity, "embedding_space_id": "c" * 64},
            )

    def test_exclusion_source_rollup_digest_must_bind_files(self) -> None:
        f = self.fixture()
        payload = json.loads(f.exclusions["dev4"].read_text())
        payload["source_sha256"] = "0" * 64
        write_json(f.exclusions["dev4"], payload)
        f.plan["exclusions"][0]["manifest"] = record(f.exclusions["dev4"])
        write_json(f.plan_path, f.plan)
        with self.assertRaisesRegex(gate.GateError, "source_sha256 does not bind"):
            f.freeze()

    def test_official_test_population_equals_complete_workload(self) -> None:
        f = self.fixture()
        # The fixture now uses a real-shaped official-test exclusion: all
        # workload qids are included, while dev4/reserve4 remain separate.
        frozen = f.freeze()
        official = next(item for item in frozen["manifest"]["exclusions"] if item["name"] == "official-test")
        self.assertEqual(official["qids_by_domain"], f.qids)

    def test_partial_official_test_population_is_rejected(self) -> None:
        f = self.fixture()
        payload = json.loads(f.exclusions["official-test"].read_text())
        payload["qids_by_dataset"]["fiqa"] = [f.qids["fiqa"][0]]
        write_json(f.exclusions["official-test"], payload)
        f.plan["exclusions"][2]["manifest"] = record(f.exclusions["official-test"])
        write_json(f.plan_path, f.plan)
        with self.assertRaisesRegex(gate.GateError, "official-test qids must equal"):
            f.freeze()

    def test_outside_official_test_population_is_rejected(self) -> None:
        f = self.fixture()
        payload = json.loads(f.exclusions["official-test"].read_text())
        payload["qids_by_dataset"]["nfcorpus"] = ["outside-official-qid"]
        write_json(f.exclusions["official-test"], payload)
        f.plan["exclusions"][2]["manifest"] = record(f.exclusions["official-test"])
        write_json(f.plan_path, f.plan)
        with self.assertRaisesRegex(gate.GateError, "official-test qids must equal"):
            f.freeze()

    def test_exclusion_identity_source_hash_and_intersection_firewall(self) -> None:
        f = self.fixture()
        payload = json.loads(f.exclusions["dev4"].read_text())
        payload["qids_by_dataset"]["fiqa"] = [f.qids["fiqa"][0]]
        write_json(f.exclusions["dev4"], payload)
        f.plan["exclusions"][0]["manifest"] = record(f.exclusions["dev4"])
        write_json(f.plan_path, f.plan)
        with self.assertRaisesRegex(gate.GateError, "workload/exclusion qid intersection"):
            f.freeze()

    def test_forged_metrics_without_harness_binding_are_rejected(self) -> None:
        f = self.fixture()
        frozen = f.freeze()
        with mock.patch.dict(os.environ, {"MOCK_MODE": "forged"}, clear=False):
            with self.assertRaisesRegex(gate.GateError, "native evaluator fields missing|native metrics TSV"):
                gate.run_harness(f.frozen_path, **f.run_kwargs(frozen))

    def test_native_outputs_without_binding_are_rejected(self) -> None:
        f = self.fixture()
        frozen = f.freeze()
        with mock.patch.dict(os.environ, {"MOCK_MODE": "omit-binding"}, clear=False):
            with self.assertRaisesRegex(gate.GateError, "native evaluator fields missing|gate binding"):
                gate.run_harness(f.frozen_path, **f.run_kwargs(frozen))

    def test_runtime_binding_cannot_echo_gate_binding(self) -> None:
        f = self.fixture()
        frozen = f.freeze()
        with mock.patch.dict(os.environ, {"MOCK_MODE": "echo-gate-binding"}, clear=False):
            with self.assertRaisesRegex(gate.GateError, "native runtime binding mismatch"):
                gate.run_harness(f.frozen_path, **f.run_kwargs(frozen))

    def test_runtime_binding_hash_must_match_resolved_binary(self) -> None:
        f = self.fixture()
        frozen = f.freeze()
        with mock.patch.dict(os.environ, {"MOCK_MODE": "bad-runtime-binding"}, clear=False):
            with self.assertRaisesRegex(gate.GateError, "native runtime binding mismatch"):
                gate.run_harness(f.frozen_path, **f.run_kwargs(frozen))

    def test_research_only_loader_opt_in_is_candidate_only_and_bound(self) -> None:
        f = self.fixture()
        frozen = f.freeze()
        info = gate.validate_frozen_manifest(f.frozen_path, expected_frozen_manifest_sha256=frozen["file_sha256"], production=False)
        outputs = {"metrics": f.root / "m.json", "metrics_tsv": f.root / "m.tsv", "per_query": f.root / "m.jsonl"}
        for role, expected in (("anchor", False), ("candidate", True)):
            argv = gate.build_evaluator_argv(info, role, "nfcorpus", outputs)
            self.assertEqual("--allow-research-only-aoqt" in argv, expected)
            binding = gate._binding_for_run(info, role, "nfcorpus", "a" * 32, argv, outputs)
            runtime = gate._runtime_binding_for_run(info, role, "nfcorpus", "a" * 32, argv, outputs)
            self.assertEqual(binding["config"]["allow_research_only_aoqt"], expected)
            self.assertEqual(runtime["config"]["allow_research_only_aoqt"], expected)

    def test_runtime_evidence_cannot_drop_or_alter_research_loader_binding(self) -> None:
        for mode in ("drop-research-loader", "alter-research-loader"):
            f = self.fixture()
            frozen = f.freeze()
            with self.subTest(mode=mode), mock.patch.dict(os.environ, {"MOCK_MODE": mode}, clear=False):
                with self.assertRaisesRegex(gate.GateError, "native runtime binding mismatch"):
                    gate.run_harness(f.frozen_path, **f.run_kwargs(frozen))

    def test_native_gate_binding_environment_uses_self_digest(self) -> None:
        f = self.fixture()
        frozen = f.freeze()
        info = gate.validate_frozen_manifest(f.frozen_path, expected_frozen_manifest_sha256=frozen["file_sha256"], production=False)
        outputs = {"metrics": f.root / "m.json", "metrics_tsv": f.root / "m.tsv", "per_query": f.root / "m.jsonl"}
        argv = gate.build_evaluator_argv(info, "candidate", "nfcorpus", outputs)
        binding = gate._binding_for_run(info, "candidate", "nfcorpus", "a" * 32, argv, outputs)
        env = gate._harness_environment(info, "a" * 32, binding, test_mode=True)
        self.assertEqual(env["EOS_AOQT_GATE_BINDING_JSON"], gate.canonical_json(binding).decode("utf-8"))
        self.assertEqual(env["EOS_AOQT_GATE_BINDING_SHA256"], binding["binding_sha256"])
        self.assertNotEqual(env["EOS_AOQT_GATE_BINDING_SHA256"], gate.sha256_bytes(env["EOS_AOQT_GATE_BINDING_JSON"].encode("utf-8")))

    def test_native_qrels_binding_mismatch_is_rejected(self) -> None:
        f = self.fixture()
        frozen = f.freeze()
        with mock.patch.dict(os.environ, {"MOCK_MODE": "bad-qrels-binding"}, clear=False):
            with self.assertRaisesRegex(gate.GateError, "dataset/qrels binding mismatch"):
                gate.run_harness(f.frozen_path, **f.run_kwargs(frozen))

    def test_copied_old_outputs_are_rejected_by_nonce_binding(self) -> None:
        f = self.fixture()
        frozen = f.freeze()
        with mock.patch.dict(os.environ, {"MOCK_MODE": "normal"}, clear=False):
            first = gate.run_harness(f.frozen_path, **f.run_kwargs(frozen))
            old_dir = first["run_dir"]
            with mock.patch.dict(os.environ, {"MOCK_MODE": "copy-old", "MOCK_COPY_FROM": old_dir}, clear=False):
                with self.assertRaisesRegex(gate.GateError, "gate binding mismatch"):
                    gate.run_harness(f.frozen_path, **f.run_kwargs(frozen))

    def test_evaluator_mutating_frozen_input_is_rejected_after_subprocess(self) -> None:
        f = self.fixture()
        frozen = f.freeze()
        with mock.patch.dict(os.environ, {"MOCK_MODE": "mutate-qrels"}, clear=False):
            with self.assertRaisesRegex(gate.GateError, "sha256 mismatch|frozen qrels binding changed"):
                gate.run_harness(f.frozen_path, **f.run_kwargs(frozen))

    def test_missing_nf_boundary_evidence_fails_closed(self) -> None:
        f = self.fixture()
        frozen = f.freeze()
        with mock.patch.dict(os.environ, {"MOCK_MODE": "missing-nf"}, clear=False):
            with self.assertRaisesRegex(gate.GateError, "boundary coverage|boundary ranks|exactly top-k=120"):
                gate.run_harness(f.frozen_path, **f.run_kwargs(frozen))

    def test_nf_boundary_is_derived_from_bound_native_top120_when_receipt_omitted(self) -> None:
        f = self.fixture()
        frozen = f.freeze()
        with mock.patch.dict(os.environ, {"MOCK_MODE": "omit-boundary"}, clear=False):
            report = gate.run_harness(f.frozen_path, **f.run_kwargs(frozen))
        self.assertTrue(report["gate_pass"])
        self.assertEqual(report["domains"]["nfcorpus"]["boundary"]["rank_window"], [80, 120])

    def test_native_per_query_regression_fails_quality_gate(self) -> None:
        f = self.fixture()
        frozen = f.freeze()
        with mock.patch.dict(os.environ, {"MOCK_MODE": "regression"}, clear=False):
            report = gate.run_harness(f.frozen_path, **f.run_kwargs(frozen))
        self.assertFalse(report["gate_pass"])
        self.assertTrue(any(item.startswith("domain:fiqa") or item.startswith("macro:") for item in report["failures"]))

    def test_fixed_nf_boundary_safety_is_not_an_unconditional_pass(self) -> None:
        f = self.fixture()
        frozen = f.freeze()
        with mock.patch.dict(os.environ, {"MOCK_MODE": "boundary-regression"}, clear=False):
            report = gate.run_harness(f.frozen_path, **f.run_kwargs(frozen))
        self.assertFalse(report["gate_pass"])
        self.assertIn("domain:nfcorpus:nfcorpus_boundary_safety", report["failures"])
        boundary = report["domains"]["nfcorpus"]["boundary"]
        self.assertEqual(boundary["rank_window"], [80, 120])
        self.assertEqual(boundary["evaluable_qid_count"], 1)
        self.assertLess(boundary["candidate_recall_at_100"], boundary["anchor_recall_at_100"])

    def test_linear_graded_ndcg_matches_eos_convention(self) -> None:
        rels = {"graded-a": 3.0, "graded-b": 1.0}
        top = [{"rank": 1, "doc_id": "graded-b", "score": 1.0, "relevance": 1.0}, {"rank": 2, "doc_id": "graded-a", "score": 0.5, "relevance": 3.0}]
        for rank in range(3, gate.TOP_K + 1):
            top.append({"rank": rank, "doc_id": f"graded-filler-{rank}", "score": 1.0 / rank, "relevance": 0.0})
        metrics = gate._rank_metrics(top, rels, {item["doc_id"] for item in top}, "graded")
        expected = (1.0 / 1.0 + 3.0 / math.log2(3.0)) / (3.0 / 1.0 + 1.0 / math.log2(3.0))
        self.assertAlmostEqual(metrics["ndcg_at_10"], expected)

    def test_native_aoqt_json_uses_xpkg_policy_and_allows_optional_orthogonality(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "candidate.aoqt.json"
            stages = []
            for _stage in range(gate.AOQT_TOPOLOGY["stages"]):
                stages.append(
                    {
                        "pairs": [[2 * index, 2 * index + 1] for index in range(gate.AOQT_TOPOLOGY["pairs_per_stage"])],
                        "angles": [0.001] * gate.AOQT_TOPOLOGY["pairs_per_stage"],
                    }
                )
            payload = {"version": "eos/aoqt-givens-transform/v1", "kind": gate.AOQT_TOPOLOGY["id"], "dim": gate.DIMENSION, "seed": 191, "angle_cap": 0.04, "stages": stages, "audit": {}}
            payload["audit"] = {"pairings_sha256": gate._transform_pairings_sha256(stages, "fixture"), "angles_sha256": gate._transform_angles_sha256(stages, "fixture"), "orthogonality_frobenius_per_dim": 0.0}
            write_json(path, payload)
            expected_policy = {
                "transform_sha256": gate.sha256_file(path),
                "pairings_sha256": payload["audit"]["pairings_sha256"],
                "angles_sha256": payload["audit"]["angles_sha256"],
                "anchor_artifact_sha256": "a" * 64,
                "anchor_package_manifest_sha256": "b" * 64,
                "anchor_embedding_space_id": "c" * 64,
            }
            parsed = gate._parse_actual_aoqt_transform(path, expected_anchor={"artifact_sha256": "a" * 64, "package_manifest_sha256": "b" * 64, "embedding_space_id": "c" * 64}, expected_policy=expected_policy, label="fixture.sidecar")
            self.assertEqual(parsed["angle_count"], 1536)
            self.assertEqual(parsed["angles_sha256"], payload["audit"]["angles_sha256"])
            self.assertEqual(parsed["orthogonality_frobenius_per_dim"], 0.0)

    def test_native_aoqt_json_rejects_training_policy_hash_substitution(self) -> None:
        head = {
            "aoqt_transform_enabled": True,
            "aoqt_transform_research_only": True,
            "aoqt_transform_research_train_allowed": True,
            "aoqt_transform_release_train_allowed": False,
            "aoqt_transform_commercial_use_allowed": False,
            "aoqt_transform_free_open_release_allowed": False,
            "aoqt_transform_quality_claim": False,
            "aoqt_transform_schema": "eos.aoqt_transform_policy.v1",
            "aoqt_transform_anchor_artifact_sha256": "a" * 64,
            "aoqt_transform_anchor_package_manifest_sha256": "b" * 64,
            "aoqt_transform_anchor_embedding_space_id": "c" * 64,
            "aoqt_transform_dataset_manifest_sha256": gate.AOQT_TRAIN_PROVENANCE["dataset_manifest_sha256"],
            "aoqt_transform_qrels_sha256_by_dataset": "\n".join(f"{domain}={digest}" for domain, digest in gate.AOQT_TRAIN_PROVENANCE["qrels_sha256_by_dataset"].items()),
            "aoqt_transform_compatibility_digest": gate.AOQT_TRAIN_PROVENANCE["compatibility_digest"],
            "aoqt_transform_transform_sha256": "d" * 64,
            "aoqt_transform_pairings_sha256": "e" * 64,
            "aoqt_transform_angles_sha256": "f" * 64,
        }
        policy = gate._parse_xpkg_aoqt_policy(head, "candidate", "fixture")
        self.assertIsNotNone(policy)
        self.assertEqual(policy["train_provenance_id"], "raw-v4")
        self.assertEqual(policy["training_contract_id"], "raw-v4")
        self.assertNotIn("aggregate_score_distill_budgets", policy)
        head["aoqt_transform_qrels_sha256_by_dataset"] = "\n".join(f"{domain}={'0' * 64 if domain == 'fiqa' else digest}" for domain, digest in gate.AOQT_TRAIN_PROVENANCE["qrels_sha256_by_dataset"].items())
        with self.assertRaisesRegex(gate.GateError, "raw-v4 provenance"):
            gate._parse_xpkg_aoqt_policy(head, "candidate", "fixture")

    def test_native_aoqt_json_accepts_registered_lane_a_contract(self) -> None:
        entry = gate.AOQT_TRAIN_PROVENANCE_REGISTRY["laneA"]
        head = {
            "aoqt_transform_enabled": True,
            "aoqt_transform_research_only": True,
            "aoqt_transform_research_train_allowed": True,
            "aoqt_transform_release_train_allowed": False,
            "aoqt_transform_commercial_use_allowed": False,
            "aoqt_transform_free_open_release_allowed": False,
            "aoqt_transform_quality_claim": False,
            "aoqt_transform_schema": "eos.aoqt_transform_policy.v1",
            "aoqt_transform_anchor_artifact_sha256": "a" * 64,
            "aoqt_transform_anchor_package_manifest_sha256": "b" * 64,
            "aoqt_transform_anchor_embedding_space_id": "c" * 64,
            "aoqt_transform_dataset_manifest_sha256": entry["dataset_manifest_sha256"],
            "aoqt_transform_qrels_sha256_by_dataset": "\n".join(f"{domain}={digest}" for domain, digest in entry["qrels_sha256_by_dataset"].items()),
            "aoqt_transform_compatibility_digest": entry["compatibility_digest"],
            "aoqt_transform_transform_sha256": "d" * 64,
            "aoqt_transform_pairings_sha256": "e" * 64,
            "aoqt_transform_angles_sha256": "f" * 64,
        }
        policy = gate._parse_xpkg_aoqt_policy(head, "candidate", "laneA")
        self.assertIsNotNone(policy)
        self.assertEqual(policy["dataset_manifest_sha256"], "f58f42acf30874f0d2c8773d2508672534348f9b20279da29719fe3e7737d39f")
        self.assertEqual(policy["train_provenance_id"], "raw-v4-score-distill-budget-laneA-v1")
        self.assertEqual(policy["training_contract"], gate.AOQT_TRAIN_LANE_A_CONTRACT)
        self.assertEqual(policy["training_contract_id"], "aoqt-trainonly-score-distill-budget-laneA-v1/q3q5")
        self.assertEqual(policy["aggregate_score_distill_budget"], 1e-5)
        self.assertNotIn("aggregate_score_distill_budgets", policy)
        self.assertEqual(policy["candidate_eligibility_policy"], gate.AOQT_TRAIN_LANE_A_POLICY)
        self.assertEqual(policy["training_provenance_source"], "closed_registry_by_manifest_sha256")

    def test_joint_budget_contract_is_explicitly_asymmetric_and_not_aggregate(self) -> None:
        self.assertEqual(gate.AOQT_TRAIN_JOINT_BUDGET_CONTRACT_ID, "aoqt-trainonly-score-distill-joint-budget-v1/q3q5")
        self.assertEqual(gate.AOQT_TRAIN_JOINT_BUDGET_PROVENANCE_ID, "raw-v4-score-distill-joint-budget-v1")
        self.assertEqual(gate.AOQT_TRAIN_JOINT_BUDGET_COMPONENTS, {"q3_score_distill": 1e-4, "q5_score_distill": 2e-5})
        self.assertNotIn("aggregate_score_distill_budget", gate.AOQT_TRAIN_JOINT_BUDGET_TEMPLATE)
        self.assertEqual(gate.AOQT_TRAIN_JOINT_BUDGET_TEMPLATE["aggregate_score_distill_budgets"], gate.AOQT_TRAIN_JOINT_BUDGET_COMPONENTS)
        self.assertEqual(gate.AOQT_TRAIN_JOINT_BUDGET_POLICY["q3_score_distill_allowed_loss_increase"], 1e-4)
        self.assertEqual(gate.AOQT_TRAIN_JOINT_BUDGET_POLICY["q5_score_distill_allowed_loss_increase"], 2e-5)
        self.assertEqual(gate.AOQT_TRAIN_JOINT_BUDGET_POLICY["q3_order_guard_allowed_loss_increase"], 0.0)
        self.assertEqual(gate.AOQT_TRAIN_JOINT_BUDGET_POLICY["q5_order_guard_allowed_loss_increase"], 0.0)
        self.assertEqual(gate.AOQT_TRAIN_JOINT_BUDGET_POLICY["nf_boundary_guard_allowed_loss_increase"], 0.0)
        self.assertEqual(gate.AOQT_TRAIN_JOINT_BUDGET_MANIFEST_SHA256, "ea05d195813f22a7829836f0bf1227b5522aa4f460cd6b4a066b55ee7444f269")
        self.assertEqual(gate.AOQT_TRAIN_PROVENANCE_REGISTRY["joint-budget-v1"]["dataset_manifest_sha256"], gate.AOQT_TRAIN_JOINT_BUDGET_MANIFEST_SHA256)

    def test_joint_budget_registry_resolves_map_and_rejects_tampering(self) -> None:
        manifest_sha = "9" * 64
        entry = copy.deepcopy(gate.AOQT_TRAIN_JOINT_BUDGET_TEMPLATE)
        entry["dataset_manifest_sha256"] = manifest_sha
        common = {
            "dataset_manifest_sha256": manifest_sha,
            "qrels_sha256_by_dataset": entry["qrels_sha256_by_dataset"],
            "compatibility_digest": entry["compatibility_digest"],
            "label": "joint.registration",
        }
        with mock.patch.dict(gate.AOQT_TRAIN_PROVENANCE_BY_MANIFEST_SHA256, {manifest_sha: entry}, clear=False):
            resolved = gate._resolve_registered_aoqt_train_provenance(
                **common,
                training_contract=gate.AOQT_TRAIN_JOINT_BUDGET_CONTRACT,
                contract_id=gate.AOQT_TRAIN_JOINT_BUDGET_CONTRACT_ID,
                aggregate_score_distill_budgets=gate.AOQT_TRAIN_JOINT_BUDGET_COMPONENTS,
            )
            self.assertEqual(resolved["provenance_id"], gate.AOQT_TRAIN_JOINT_BUDGET_PROVENANCE_ID)
            self.assertNotIn("aggregate_score_distill_budget", resolved)
            self.assertEqual(resolved["aggregate_score_distill_budgets"], gate.AOQT_TRAIN_JOINT_BUDGET_COMPONENTS)
            with self.assertRaisesRegex(gate.GateError, "per-component score-distill budgets"):
                gate._resolve_registered_aoqt_train_provenance(**common, aggregate_score_distill_budgets={"q3_score_distill": 1e-4, "q5_score_distill": 1e-5})
            with self.assertRaisesRegex(gate.GateError, "aggregate score-distill budget is not valid"):
                gate._resolve_registered_aoqt_train_provenance(**common, aggregate_score_distill_budget=1e-4)
            with self.assertRaisesRegex(gate.GateError, "training contract"):
                gate._resolve_registered_aoqt_train_provenance(**common, training_contract=gate.AOQT_TRAIN_LANE_A_CONTRACT)
        missing_map_sha = "7" * 64
        missing_map_entry = copy.deepcopy(entry)
        missing_map_entry.pop("aggregate_score_distill_budgets")
        with mock.patch.dict(gate.AOQT_TRAIN_PROVENANCE_BY_MANIFEST_SHA256, {missing_map_sha: missing_map_entry}, clear=False):
            with self.assertRaisesRegex(gate.GateError, "missing its closed per-component budget map"):
                gate._resolve_registered_aoqt_train_provenance(
                    dataset_manifest_sha256=missing_map_sha,
                    qrels_sha256_by_dataset=missing_map_entry["qrels_sha256_by_dataset"],
                    compatibility_digest=missing_map_entry["compatibility_digest"],
                    label="joint.missing-map",
                )
        with self.assertRaisesRegex(gate.GateError, "not a preregistered train-only provenance"):
            gate._resolve_registered_aoqt_train_provenance(
                dataset_manifest_sha256="0" * 64,
                qrels_sha256_by_dataset=gate.AOQT_TRAIN_PROVENANCE["qrels_sha256_by_dataset"],
                compatibility_digest=gate.AOQT_TRAIN_PROVENANCE["compatibility_digest"],
                label="joint.unregistered",
            )

    def test_joint_budget_policy_parse_exposes_closed_component_map(self) -> None:
        manifest_sha = "8" * 64
        entry = copy.deepcopy(gate.AOQT_TRAIN_JOINT_BUDGET_TEMPLATE)
        entry["dataset_manifest_sha256"] = manifest_sha
        head = {
            "aoqt_transform_enabled": True,
            "aoqt_transform_research_only": True,
            "aoqt_transform_research_train_allowed": True,
            "aoqt_transform_release_train_allowed": False,
            "aoqt_transform_commercial_use_allowed": False,
            "aoqt_transform_free_open_release_allowed": False,
            "aoqt_transform_quality_claim": False,
            "aoqt_transform_schema": "eos.aoqt_transform_policy.v1",
            "aoqt_transform_anchor_artifact_sha256": "a" * 64,
            "aoqt_transform_anchor_package_manifest_sha256": "b" * 64,
            "aoqt_transform_anchor_embedding_space_id": "c" * 64,
            "aoqt_transform_dataset_manifest_sha256": manifest_sha,
            "aoqt_transform_qrels_sha256_by_dataset": "\n".join(f"{domain}={digest}" for domain, digest in entry["qrels_sha256_by_dataset"].items()),
            "aoqt_transform_compatibility_digest": entry["compatibility_digest"],
            "aoqt_transform_transform_sha256": "d" * 64,
            "aoqt_transform_pairings_sha256": "e" * 64,
            "aoqt_transform_angles_sha256": "f" * 64,
        }
        with mock.patch.dict(gate.AOQT_TRAIN_PROVENANCE_BY_MANIFEST_SHA256, {manifest_sha: entry}, clear=False):
            policy = gate._parse_xpkg_aoqt_policy(head, "candidate", "joint")
        self.assertIsNotNone(policy)
        self.assertEqual(policy["training_contract"], gate.AOQT_TRAIN_JOINT_BUDGET_CONTRACT)
        self.assertEqual(policy["training_contract_id"], gate.AOQT_TRAIN_JOINT_BUDGET_CONTRACT_ID)
        self.assertEqual(policy["aggregate_score_distill_budgets"], gate.AOQT_TRAIN_JOINT_BUDGET_COMPONENTS)
        self.assertNotIn("aggregate_score_distill_budget", policy)

    def test_registered_aoqt_train_provenance_rejects_unknown_manifest_and_contract(self) -> None:
        entry = gate.AOQT_TRAIN_PROVENANCE_REGISTRY["laneA"]
        common = {
            "qrels_sha256_by_dataset": entry["qrels_sha256_by_dataset"],
            "compatibility_digest": entry["compatibility_digest"],
            "label": "laneA.registration",
        }
        with self.assertRaisesRegex(gate.GateError, "not a preregistered train-only provenance"):
            gate._resolve_registered_aoqt_train_provenance(dataset_manifest_sha256="0" * 64, **common)
        with self.assertRaisesRegex(gate.GateError, "does not match the preregistered"):
            gate._resolve_registered_aoqt_train_provenance(dataset_manifest_sha256=entry["dataset_manifest_sha256"], training_contract="unregistered-contract", **common)
        with self.assertRaisesRegex(gate.GateError, "aggregate score-distill budget"):
            gate._resolve_registered_aoqt_train_provenance(dataset_manifest_sha256=entry["dataset_manifest_sha256"], aggregate_score_distill_budget=2e-5, **common)

    def test_workload_output_root_collision_is_rejected(self) -> None:
        f = self.fixture()
        f.plan["output_root"] = str(f.datasets["fiqa"])
        write_json(f.plan_path, f.plan)
        with self.assertRaisesRegex(gate.GateError, "dataset directories|input/package artifacts"):
            f.freeze()

    def test_argv_is_constructed_with_all_pinned_semantics(self) -> None:
        f = self.fixture()
        frozen = f.freeze()
        info = gate.validate_frozen_manifest(f.frozen_path, expected_frozen_manifest_sha256=frozen["file_sha256"], production=False)
        outputs = {"metrics": f.root / "m.json", "metrics_tsv": f.root / "m.tsv", "per_query": f.root / "m.jsonl"}
        argv = gate.build_evaluator_argv(info, "candidate", "nfcorpus", outputs)
        self.assertEqual(argv[0], str(f.binary.resolve()))
        self.assertEqual(argv[1], gate.NATIVE_EVAL_SUBCOMMAND)
        self.assertEqual(argv[argv.index("--bits") + 1], "3,5")
        self.assertEqual(argv[argv.index("--quantizer-seed") + 1], str(gate.TURBOQUANT_SEED))
        self.assertEqual(argv[argv.index("--batch-size") + 1], str(gate.BATCH_SIZE))
        self.assertEqual(argv[argv.index("--top-k") + 1], "120")
        self.assertEqual(argv[-1], str(f.datasets["nfcorpus"].resolve()))

    def test_harness_rejects_internal_argv_semantics_drift(self) -> None:
        f = self.fixture()
        frozen = f.freeze()
        original = gate.build_evaluator_argv

        def drifted_builder(info, role, domain, outputs):
            argv = original(info, role, domain, outputs)
            argv[argv.index("--max-queries") + 1] = "1"
            return argv

        with mock.patch.object(gate, "build_evaluator_argv", side_effect=drifted_builder):
            with self.assertRaisesRegex(gate.GateError, "exact pinned command"):
                gate.run_harness(f.frozen_path, **f.run_kwargs(frozen))

    def test_duplicate_json_keys_are_rejected(self) -> None:
        f = self.fixture()
        raw = f.workload.read_text()
        f.workload.write_text(raw.replace('"schema": "eos.aoqt.heldout_workload.v3"', '"schema": "eos.aoqt.heldout_workload.v3", "schema": "forged"', 1))
        f.plan["source"]["workload"] = record(f.workload)
        write_json(f.plan_path, f.plan)
        with self.assertRaisesRegex(gate.GateError, "duplicate JSON key"):
            f.freeze()


if __name__ == "__main__":
    unittest.main()
