"""Synthetic, local-only tests for :mod:`aoqt_heldout_gate`.

The evaluator used here is a tiny executable written into a temporary
directory.  It is never the EOS binary and it consumes only synthetic qrels,
corpus, and query files.  No official qrels result or official evaluator is
read by this test module.
"""

from __future__ import annotations

import copy
import json
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
import json
import math
import os
import shutil
import sys
from pathlib import Path

MODE = os.environ.get("MOCK_MODE", "normal")

def arg(name):
    try:
        return sys.argv[sys.argv.index(name) + 1]
    except (ValueError, IndexError):
        raise SystemExit("missing " + name)

def qrels(path):
    out = {}
    for raw in Path(path).read_text().splitlines():
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
    positive = [v for v in rels.values() if v > 0]
    ideal = sorted(positive, reverse=True)[:10]
    idcg = sum((2.0 ** rel - 1.0) / math.log2(i + 2) for i, rel in enumerate(ideal))
    dcg = sum((2.0 ** rel - 1.0) / math.log2(rank + 1) for rank, _d, rel in ranking if rank <= 10 and rel > 0)
    hit = sum(1 for rank, d, rel in ranking if rank <= 100 and rel > 0 and d in rels)
    return {"ndcg_at_10": dcg / idcg if idcg else 0.0, "recall_at_100": hit / len(positive) if positive else 0.0}

def ranking(qid, rels, relevant_rank, short=False):
    n = 119 if short else 120
    docs = [f"doc-{qid}-{i}" for i in range(1, n + 1)]
    rel_doc = next(d for d, v in rels.items() if v > 0)
    # Replace a slot rather than inserting a new document.  This keeps the
    # normal fixture exactly top-k=120 and makes the short NF fixture truly
    # top-k=119, so the production parser must exercise its boundary guard.
    if rel_doc in docs:
        docs.remove(rel_doc)
        docs.append(f"filler-{qid}")
    if len(docs) != n:
        raise SystemExit("fixture ranking length drift")
    if not 1 <= relevant_rank <= n:
        raise SystemExit("fixture relevant rank outside top-k")
    docs[relevant_rank - 1] = rel_doc
    return [(i + 1, d, float(rels.get(d, 0.0))) for i, d in enumerate(docs)]

def main():
    metrics = Path(arg("--metrics-json"))
    tsv = Path(arg("--metrics-tsv"))
    perq = Path(arg("--per-query-jsonl"))
    qpath = Path(arg("--qrels"))
    domain = arg("--dataset")
    package = Path(sys.argv[-2])
    dataset_dir = Path(sys.argv[-1])
    binding = json.loads(os.environ["EOS_AOQT_GATE_BINDING_JSON"])
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
    rows = []
    aggregates = {3: [], 5: []}
    dense_values = []
    for qid, rels in sorted(rels_by_qid.items()):
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
        dense_values.append(dq)
        aggregates[3].append(q3)
        aggregates[5].append(q5)
        for bits, compact, q in ((3, compact3, q3), (5, compact5, q5)):
            row = {
                "schema": "manta.embedding_turboquant_retrieval_per_query.v1",
                "dataset": domain,
                "query_id": qid,
                "method": f"turboquant_ip_b{bits}",
                "bits": bits,
                "scoring_surface": "turboquant_ip_prepared",
                "quantizer_seed": 5581486560434873699,
                "relevant_count": sum(1 for v in rels.values() if v > 0),
                "first_relevant_rank": next(rank for rank, d, _r in compact if d in rels),
                "quality": q,
                "dense_quality": dq,
                "top_k": [{"rank": r, "doc_id": d, "score": float(1.0 / r), "relevance": rel} for r, d, rel in compact],
                "dense_top_k": [{"rank": r, "doc_id": d, "score": float(1.0 / r), "relevance": rel} for r, d, rel in dense],
                "gate_binding": binding,
            }
            if MODE == "omit-binding":
                row.pop("gate_binding")
            rows.append(row)
    def avg(values, key):
        return sum(item[key] for item in values) / len(values)
    dense = {"ndcg_at_10": avg(dense_values, "ndcg_at_10"), "recall_at_100": avg(dense_values, "recall_at_100")}
    native_rows = []
    for bits in (3, 5):
        q = {"ndcg_at_10": avg(aggregates[bits], "ndcg_at_10"), "recall_at_100": avg(aggregates[bits], "recall_at_100")}
        native_rows.append({"bits": bits, "method": f"turboquant_ip_b{bits}", "quality": q, "ndcg_at_10_delta": 0.0, "recall_at_100_delta": 0.0})
    payload = {
        "schema": "manta.embedding_turboquant_retrieval_metrics.v1",
        "dataset": domain,
        "artifact": str(package),
        "backend": "mock-native",
        "inputs": {
            "corpus_path": str((dataset_dir / "corpus.jsonl").resolve()),
            "queries_path": str((dataset_dir / "queries.jsonl").resolve()),
            "qrels_path": str(qpath.resolve()),
            "qrels_sha256": __import__("hashlib").sha256(qpath.read_bytes()).hexdigest(),
            "documents": 120,
            "queries": len(rels_by_qid),
            "relevant_pairs": sum(1 for rels in rels_by_qid.values() for v in rels.values() if v > 0),
            "scored_pairs": 120 * len(rels_by_qid),
        },
        "config": {"batch_size": 64, "top_k": 120, "bits": [3, 5], "quantizer_seed": 5581486560434873699},
        "dense": {"quality": dense},
        "rows": native_rows,
        "gate_binding": binding,
    }
    if MODE == "bad-qrels-binding":
        payload["inputs"]["qrels_sha256"] = "0" * 64
    if MODE == "omit-binding":
        payload.pop("gate_binding")
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
        self.output_root = self.root / "gate-runs"
        self.inputs.mkdir(parents=True)
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
            write_raw(dataset / "corpus.jsonl", '{"_id":"doc"}\n')
            write_raw(dataset / "queries.jsonl", '{"_id":"q"}\n')
            write_json(dataset / "manifest.json", {"schema": gate.DATASET_SCHEMA, "domain": domain, "dataset_id": f"fixture-{domain}", "split": gate.HELDOUT_SPLIT, "corpus_sha256": gate.sha256_file(dataset / "corpus.jsonl"), "queries_sha256": gate.sha256_file(dataset / "queries.jsonl")})
            write_raw(self.qrels[domain], "query-id\tcorpus-id\tscore\n" + "\n".join(f"{qid}\trel-{qid}\t1" for qid in self.qids[domain]) + "\n")
        qrels_records = {domain: record(path) for domain, path in self.qrels.items()}
        dataset_records = {domain: {"dataset_id": f"fixture-{domain}", "dataset_dir": str(self.datasets[domain]), "corpus": record(self.datasets[domain] / "corpus.jsonl"), "queries": record(self.datasets[domain] / "queries.jsonl"), "manifest": record(self.datasets[domain] / "manifest.json")} for domain in gate.DOMAINS}
        compatibility = gate.sha256_bytes(b"fixture-compatibility")
        approved_payload = {"schema": gate.APPROVED_WORKLOAD_SCHEMA, "gate_id": "fixture-gate", "descriptor_id": "fixture-approved", "split": gate.HELDOUT_SPLIT, "dimension": gate.DIMENSION, "domains": list(gate.DOMAINS), "query_ids_by_domain": copy.deepcopy(self.qids), "qid_set_sha256_by_domain": gate.qids_sha256_by_domain(self.qids), "query_count_by_domain": {domain: 2 for domain in gate.DOMAINS}, "qrels_sha256_by_domain": {domain: qrels_records[domain]["sha256"] for domain in gate.DOMAINS}, "qrels_query_count_by_domain": {domain: 2 for domain in gate.DOMAINS}, "relevant_pair_count_by_domain": {domain: 2 for domain in gate.DOMAINS}, "dataset_manifest_sha256_by_domain": {domain: dataset_records[domain]["manifest"]["sha256"] for domain in gate.DOMAINS}, "corpus_sha256_by_domain": {domain: dataset_records[domain]["corpus"]["sha256"] for domain in gate.DOMAINS}, "queries_sha256_by_domain": {domain: dataset_records[domain]["queries"]["sha256"] for domain in gate.DOMAINS}, "compatibility_digest": compatibility, "nfcorpus_boundary_qids": [self.qids["nfcorpus"][0]], "nfcorpus_boundary_qids_sha256": gate.sha256_json([self.qids["nfcorpus"][0]]), "nfcorpus_boundary_rank_window": [80, 120], "metric_surfaces": list(gate.SURFACES), "cutoffs": {"ndcg_at_10": 10, "recall_at_100": 100}, "turboquant": {"q3_bits": 3, "q5_bits": 5, "seed": gate.TURBOQUANT_SEED, "top_k": 120, "score_mode": "turboquant_ip_prepared", "package_mode": "native_mll_sibling"}}
        write_json(self.approved, approved_payload)
        approved_record = record(self.approved)
        workload_payload = {"schema": gate.WORKLOAD_SCHEMA, "gate_id": "fixture-gate", "workload_id": "fixture-workload", "descriptor_sha256": approved_record["sha256"], "split": gate.HELDOUT_SPLIT, "dimension": gate.DIMENSION, "query_ids_by_domain": copy.deepcopy(self.qids), "qid_set_sha256_by_domain": gate.qids_sha256_by_domain(self.qids), "query_count_by_domain": {domain: 2 for domain in gate.DOMAINS}, "qrels_sha256_by_domain": {domain: qrels_records[domain]["sha256"] for domain in gate.DOMAINS}, "dataset_manifest_sha256_by_domain": {domain: dataset_records[domain]["manifest"]["sha256"] for domain in gate.DOMAINS}, "corpus_sha256_by_domain": {domain: dataset_records[domain]["corpus"]["sha256"] for domain in gate.DOMAINS}, "queries_sha256_by_domain": {domain: dataset_records[domain]["queries"]["sha256"] for domain in gate.DOMAINS}, "compatibility_digest": compatibility}
        write_json(self.workload, workload_payload)
        workload_record = record(self.workload)
        for name in gate.EXCLUSION_NAMES:
            source = self.exclusion_sources[name]
            write_json(source, {"source": name, "fixture": True})
            source_map = {str(source.resolve()): gate.sha256_file(source)}
            write_json(self.exclusions[name], {"schema": "eos.aoqt_stage2.exclusion_qids.v1", "name": name, "qids_by_dataset": {domain: [f"excluded-{name}-{domain}"] for domain in gate.DOMAINS}, "source_sha256": gate.sha256_json(source_map), "source_sha256_by_file": source_map})
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
        self.plan = {"schema": gate.PLAN_SCHEMA, "gate_id": "fixture-gate", "candidate_id": "fixture-candidate", "dimension": gate.DIMENSION, "thresholds": copy.deepcopy(gate.THRESHOLDS), "evaluation": {"executed": False, "official": False}, "anchor": {"id": "fixture-anchor", "package": anchor_record}, "candidate": {"id": "fixture-candidate", "package": candidate_record}, "source": {"manifest": record(self.source_manifest), "binary": record(self.binary), "cwd": str(self.root), "dataset_by_domain": dataset_records, "workload": workload_record, "approved_workload": approved_record}, "approved_workload": approved_record, "qrels": qrels_records, "exclusions": [{"kind": "qid_only", "name": name, "manifest": record(self.exclusions[name])} for name in gate.EXCLUSION_NAMES], "workload_root": str(self.inputs), "output_root": str(self.output_root)}
        write_json(self.plan_path, self.plan)
        return self

    def freeze(self) -> dict:
        return gate.freeze_manifest(self.plan_path, self.frozen_path, test_mode=True)

    def run_kwargs(self, frozen: dict) -> dict:
        return {"expected_frozen_manifest_sha256": frozen["file_sha256"], "expected_workload_manifest_sha256": frozen["manifest"]["approved_workload"]["sha256"], "expected_binary_sha256": frozen["manifest"]["provenance"]["binary_sha256"], "expected_anchor_package_sha256": frozen["manifest"]["expected_anchor_identity"]["artifact_sha256"], "expected_anchor_manifest_sha256": frozen["manifest"]["expected_anchor_identity"]["package_manifest_sha256"], "expected_anchor_embedding_space_id": frozen["manifest"]["expected_anchor_identity"]["embedding_space_id"], "test_mode": True, "timeout_seconds": 30}


class AOQTHeldoutGateTest(unittest.TestCase):
    def fixture(self) -> Fixture:
        return Fixture(Path(tempfile.mkdtemp())).build()

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

    def test_production_rejects_synthetic_json_package_fallback(self) -> None:
        f = self.fixture()
        frozen = f.freeze()
        with self.assertRaisesRegex(gate.GateError, "native MLL artifacts|JSON fallback"):
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

    def test_exclusion_source_rollup_digest_must_bind_files(self) -> None:
        f = self.fixture()
        payload = json.loads(f.exclusions["dev4"].read_text())
        payload["source_sha256"] = "0" * 64
        write_json(f.exclusions["dev4"], payload)
        f.plan["exclusions"][0]["manifest"] = record(f.exclusions["dev4"])
        write_json(f.plan_path, f.plan)
        with self.assertRaisesRegex(gate.GateError, "source_sha256 does not bind"):
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

    def test_native_per_query_regression_fails_quality_gate(self) -> None:
        f = self.fixture()
        frozen = f.freeze()
        with mock.patch.dict(os.environ, {"MOCK_MODE": "regression"}, clear=False):
            report = gate.run_harness(f.frozen_path, **f.run_kwargs(frozen))
        self.assertFalse(report["gate_pass"])
        self.assertTrue(any(item.startswith("query:fiqa") for item in report["failures"]))

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
