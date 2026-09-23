#!/usr/bin/env python3
"""Tests for AOQT V7 non-official dev proxy producer."""

from __future__ import annotations

import importlib.util
import json
import os
import stat
import subprocess
import tempfile
import textwrap
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parent / "build_aoqt_v7_non_official_dev_proxy.py"
spec = importlib.util.spec_from_file_location("dev_proxy", SCRIPT)
dev_proxy = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(dev_proxy)


HEX = "a" * 64
OFFICIAL_SOURCE_SHA = "c" * 64
_CANONICAL_SEED191_PAIRINGS: list[list[list[int]]] | None = None


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def make_split(path: Path, official_manifest_sha: str, official_source_sha: str, *, official_overlap: bool = False) -> dict:
    qids = {
        "fiqa": ["fiqa-empty", "fiqa-q1"],
        "nfcorpus": ["nf-q1"],
        "scifact": ["sci-q1"],
    }
    if official_overlap:
        qids["fiqa"] = ["official-fiqa"]
    dev = {
        "qid_count_by_dataset": {domain: len(values) for domain, values in qids.items()},
        "qids_by_dataset": qids,
        "qids_by_dataset_sha256": dev_proxy.sha256_json(qids),
    }
    split = {
        "schema": dev_proxy.SPLIT_SCHEMA,
        "actual_training_ran": False,
        "actual_eval_ran": False,
        "actual_official_data_eval_ran": False,
        "source_plan": {"sha256": HEX},
        "official_qid_registry": {"manifest_sha256": official_manifest_sha, "source_sha256": official_source_sha},
        "folds": [{"name": "fold-0", "dev": dev}],
        "leakage_proof": {"passed": True, "official_overlap_count": 0},
        "provenance": {},
    }
    comparable = json.loads(json.dumps(split))
    split["provenance"]["manifest_sha256"] = dev_proxy.sha256_json(comparable)
    write_json(path, split)
    return split


def canonical_seed191_pairings() -> list[list[list[int]]]:
    global _CANONICAL_SEED191_PAIRINGS
    if _CANONICAL_SEED191_PAIRINGS is not None:
        return json.loads(json.dumps(_CANONICAL_SEED191_PAIRINGS))
    program = textwrap.dedent(
        """
        package main

        import (
            "crypto/sha256"
            "encoding/hex"
            "encoding/json"
            "fmt"
            "math/rand"
        )

        func main() {
            rng := rand.New(rand.NewSource(191))
            stages := make([][][2]int, 8)
            digest := sha256.New()
            for stage := range stages {
                perm := rng.Perm(384)
                pairs := make([][2]int, 192)
                for i := range pairs {
                    a, b := perm[2*i], perm[2*i+1]
                    if a > b {
                        a, b = b, a
                    }
                    pairs[i] = [2]int{a, b}
                    fmt.Fprintf(digest, "%d,%d\\n", a, b)
                }
                stages[stage] = pairs
            }
            data, err := json.Marshal(stages)
            if err != nil {
                panic(err)
            }
            fmt.Println(hex.EncodeToString(digest.Sum(nil)))
            fmt.Println(string(data))
        }
        """
    )
    with tempfile.TemporaryDirectory() as temp:
        source = Path(temp) / "main.go"
        source.write_text(program, encoding="utf-8")
        proc = subprocess.run(["go", "run", str(source)], cwd=SCRIPT.parent, check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    observed_sha, pairings_json = proc.stdout.splitlines()
    if observed_sha != dev_proxy.CANONICAL_D384_SEED191_PAIRINGS_SHA256:
        raise AssertionError(f"canonical pairings sha = {observed_sha}")
    _CANONICAL_SEED191_PAIRINGS = json.loads(pairings_json)
    return json.loads(json.dumps(_CANONICAL_SEED191_PAIRINGS))


def make_official(path: Path, *, include_dev_qid: bool = False, source_sha: str = OFFICIAL_SOURCE_SHA) -> None:
    qids = {"fiqa": ["official-fiqa"], "nfcorpus": ["official-nf"], "scifact": ["official-sci"]}
    if include_dev_qid:
        qids["fiqa"].append("fiqa-q1")
    write_json(path, {"schema": "eos.aoqt_stage2.exclusion_qids.v1", "name": "official-test", "qids_by_dataset": qids, "source_sha256": source_sha})


def make_sidecar(path: Path, *, canonical_pairings: bool = True, orthogonality: float | str = 0.0) -> None:
    pairings = canonical_seed191_pairings() if canonical_pairings else [[[i, i + 192] for i in range(192)] for _ in range(8)]
    stages = [{"pairs": pairings[stage], "angles": [0.0] * 192} for stage in range(8)]
    write_json(
        path,
        {
            "version": "eos/aoqt-givens-transform/v1",
            "kind": "aoqt_givens_v1",
            "dim": 384,
            "seed": 191,
            "stages": stages,
            "audit": {
                "pairings_sha256": dev_proxy.transform_pairings_sha256(stages),
                "angles_sha256": dev_proxy.transform_angles_sha256(stages),
                "orthogonality_frobenius_per_dim": orthogonality,
            },
        },
    )


def make_dataset(root: Path, domain: str) -> Path:
    dataset = root / domain
    write_jsonl(dataset / "corpus.jsonl", [{"_id": f"{domain}-d1", "text": "alpha"}, {"_id": f"{domain}-d2", "text": "beta"}])
    qids = {
        "fiqa": ["fiqa-empty", "fiqa-q1"],
        "nfcorpus": ["nf-q1"],
        "scifact": ["sci-q1"],
    }[domain]
    write_jsonl(dataset / "queries.jsonl", [{"_id": qid, "text": qid} for qid in qids])
    rows = {
        "fiqa": [("fiqa-q1", "fiqa-d1", 1)],
        "nfcorpus": [("nf-q1", "nfcorpus-d1", 1)],
        "scifact": [("sci-q1", "scifact-d1", 1)],
    }[domain]
    qrels = dataset / "qrels" / "train.tsv"
    qrels.parent.mkdir(parents=True, exist_ok=True)
    qrels.write_text("query-id\tcorpus-id\tscore\n" + "".join(f"{qid}\t{doc}\t{score}\n" for qid, doc, score in rows), encoding="utf-8")
    return dataset


def make_vectors(path: Path, ids: list[str]) -> None:
    write_jsonl(path, [{"_id": rid, "role": "query" if "q" in rid or "empty" in rid else "document", "embedding": [1.0, 0.0]} for rid in ids])


def make_fake_eos(path: Path, *, churn: bool = False, q3_delta: float = 0.0) -> None:
    script = f"""#!/usr/bin/env python3
import json, os, shutil, sys
from pathlib import Path
churn = {str(churn)}
q3_delta = {q3_delta!r}
args = sys.argv[1:]
cmd = args[0]
def flag(name):
    i = args.index(name)
    return args[i+1]
def write_json(path, payload):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, sort_keys=True) + "\\n", encoding="utf-8")
def write_per(path, candidate, bits):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    docs = ["d1","d2","d3"] if (candidate and churn and bits == 3) else ["d2","d1","d3"]
    rows = []
    for qid in ["fiqa-q1"] if "fiqa" in str(path) else (["nf-q1"] if "nfcorpus" in str(path) else ["sci-q1"]):
        rows.append({{"schema":"manta.embedding_turboquant_retrieval_per_query.v1","dataset":flag("--dataset"),"query_id":qid,"method":f"turboquant_ip_b{{bits}}","bits":bits,"scoring_surface":"turboquant_ip_prepared","top_k":[{{"rank":i+1,"doc_id":doc,"score":1.0/(i+1),"relevance":0}} for i, doc in enumerate(docs)]}})
    Path(path).write_text("".join(json.dumps(row, sort_keys=True)+"\\n" for row in rows), encoding="utf-8")
if cmd == "transform-aoqt-vectors":
    Path(flag("--out-doc-vectors")).parent.mkdir(parents=True, exist_ok=True)
    Path(flag("--out-query-vectors")).parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(flag("--doc-vectors"), flag("--out-doc-vectors"))
    shutil.copyfile(flag("--query-vectors"), flag("--out-query-vectors"))
    write_json(flag("--sidecar-json"), {{"schema":"eos.aoqt.vector_cache_transform_binding.v1"}})
    sys.exit(0)
candidate = "candidate" in flag("--backend")
metrics = flag("--metrics-json")
perq = flag("--per-query-jsonl")
dataset = flag("--dataset")
if cmd == "eval-retrieval-vectors":
    write_json(metrics, {{"schema":"manta.embedding_retrieval_metrics.v1","dataset":dataset,"quality":{{"ndcg_at_10":0.5,"recall_at_100":1.0}}}})
    Path(perq).parent.mkdir(parents=True, exist_ok=True)
    Path(perq).write_text("", encoding="utf-8")
    sys.exit(0)
if cmd == "eval-retrieval-vectors-turboquant":
    bits = int(flag("--bits"))
    ndcg = 0.5 + (q3_delta if candidate and bits == 3 else 0.0)
    write_json(metrics, {{"schema":"manta.embedding_turboquant_retrieval_metrics.v1","dataset":dataset,"rows":[{{"bits":bits,"method":f"turboquant_ip_b{{bits}}","quality":{{"ndcg_at_10":ndcg,"recall_at_100":1.0}}}}]}})
    Path(flag("--metrics-tsv")).write_text("dataset\\trow\\n", encoding="utf-8")
    write_per(perq, candidate, bits)
    sys.exit(0)
sys.exit(9)
"""
    path.write_text(script, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


class Fixture:
    def __init__(self, root: Path):
        self.root = root
        self.official = root / "official.json"
        self.official_source_sha = OFFICIAL_SOURCE_SHA
        make_official(self.official, source_sha=self.official_source_sha)
        self.official_sha = dev_proxy.sha256_file(self.official)
        self.split = root / "split.json"
        make_split(self.split, self.official_sha, self.official_source_sha)
        self.validation = root / "validation.json"
        write_json(self.validation, {"schema": dev_proxy.SPLIT_VALIDATION_SCHEMA, "manifest_file_sha256": dev_proxy.sha256_file(self.split), "manifest_payload_sha256": json.loads(self.split.read_text())["provenance"]["manifest_sha256"], "source_plan_sha256": HEX, "official_qid_registry_sha256": self.official_sha, "passed": True})
        self.sidecar = root / "sidecar.json"
        make_sidecar(self.sidecar)
        self.data_root = root / "datasets"
        self.datasets = {domain: make_dataset(self.data_root, domain) for domain in dev_proxy.DOMAINS}
        self.anchor_docs = {}
        self.anchor_queries = {}
        for domain in dev_proxy.DOMAINS:
            self.anchor_docs[domain] = root / "vectors" / domain / "docs.jsonl"
            self.anchor_queries[domain] = root / "vectors" / domain / "queries.jsonl"
            make_vectors(self.anchor_docs[domain], [f"{domain}-d1", f"{domain}-d2"])
            qids = {"fiqa": ["fiqa-empty", "fiqa-q1"], "nfcorpus": ["nf-q1"], "scifact": ["sci-q1"]}[domain]
            make_vectors(self.anchor_queries[domain], qids)
        self.fake_eos = root / "fake-eos.py"

    def argv(self, output: Path) -> list[str]:
        args = [
            "--split-manifest", str(self.split),
            "--split-validation", str(self.validation),
            "--fold-id", "fold-0",
            "--official-qids", str(self.official),
            "--sidecar", str(self.sidecar),
            "--output", str(output),
            "--work-dir", str(output.with_suffix("")),
            "--eos-bin", str(self.fake_eos),
            "--execute",
        ]
        for domain in dev_proxy.DOMAINS:
            args += ["--dataset-dir", f"{domain}={self.datasets[domain]}"]
            args += ["--anchor-doc-vectors", f"{domain}={self.anchor_docs[domain]}"]
            args += ["--anchor-query-vectors", f"{domain}={self.anchor_queries[domain]}"]
        return args


class DevProxyTests(unittest.TestCase):
    def test_identity_zero_deltas_and_hash_binding(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            fixture = Fixture(Path(td))
            make_fake_eos(fixture.fake_eos)
            output = Path(td) / "report.json"
            self.assertEqual(dev_proxy.main(fixture.argv(output)), 0)
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(report["schema"], dev_proxy.SCHEMA)
            self.assertEqual(report["macro"]["q3_ndcg_at_10_delta"], 0.0)
            self.assertEqual(report["macro"]["q3_top10_churn"], 0)
            self.assertEqual(report["official_qids"]["sha256"], fixture.official_sha)
            self.assertEqual(report["official_qids"]["file_sha256"], fixture.official_sha)
            self.assertEqual(report["split_validation"]["file_sha256"], dev_proxy.sha256_file(fixture.validation))
            self.assertEqual(report["split_manifest"]["split_validation_file_sha256"], dev_proxy.sha256_file(fixture.validation))
            self.assertEqual(report["split_manifest"]["official_qids_file_sha256"], fixture.official_sha)
            self.assertEqual(report["split_manifest"]["split_manifest_file_sha256"], dev_proxy.sha256_file(fixture.split))
            self.assertTrue(report["materialized_dev_datasets"]["fiqa"]["fiqa_empty_relevant_doc_vector_misses_audited"])

    def test_transformed_toy_churn_and_macro_recompute(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            fixture = Fixture(Path(td))
            make_fake_eos(fixture.fake_eos, churn=True, q3_delta=0.003)
            output = Path(td) / "report.json"
            self.assertEqual(dev_proxy.main(fixture.argv(output)), 0)
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertAlmostEqual(report["macro"]["q3_ndcg_at_10_delta"], 0.003)
            self.assertEqual(report["macro"]["q3_top10_churn"], 3)
            for domain in dev_proxy.DOMAINS:
                self.assertEqual(report["domains"][domain]["q3_top10_churn"], 1)

    def test_official_qid_rejection(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            fixture = Fixture(Path(td))
            make_official(fixture.official, include_dev_qid=True)
            fixture.official_sha = dev_proxy.sha256_file(fixture.official)
            make_split(fixture.split, fixture.official_sha, fixture.official_source_sha)
            write_json(fixture.validation, {"schema": dev_proxy.SPLIT_VALIDATION_SCHEMA, "manifest_file_sha256": dev_proxy.sha256_file(fixture.split), "manifest_payload_sha256": json.loads(fixture.split.read_text())["provenance"]["manifest_sha256"], "source_plan_sha256": HEX, "official_qid_registry_sha256": fixture.official_sha, "passed": True})
            make_fake_eos(fixture.fake_eos)
            with self.assertRaisesRegex(dev_proxy.DevProxyError, "official qids"):
                dev_proxy.build_report(dev_proxy.parse_args(fixture.argv(Path(td) / "report.json")))

    def test_missing_vectors_fail_for_non_empty_relevant_doc_and_query(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            fixture = Fixture(Path(td))
            make_fake_eos(fixture.fake_eos)
            make_vectors(fixture.anchor_docs["fiqa"], ["fiqa-d2"])
            with self.assertRaisesRegex(dev_proxy.DevProxyError, "missing vectors for non-empty relevant docs"):
                dev_proxy.build_report(dev_proxy.parse_args(fixture.argv(Path(td) / "report.json")))
            fixture = Fixture(Path(td) / "q")
            make_fake_eos(fixture.fake_eos)
            make_vectors(fixture.anchor_queries["fiqa"], ["fiqa-q1"])
            with self.assertRaisesRegex(dev_proxy.DevProxyError, "missing query vectors"):
                dev_proxy.build_report(dev_proxy.parse_args(fixture.argv(Path(td) / "q-report.json")))

    def test_artifact_hash_binding_rejects_stale_validation(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            fixture = Fixture(Path(td))
            make_fake_eos(fixture.fake_eos)
            write_json(fixture.validation, {"schema": dev_proxy.SPLIT_VALIDATION_SCHEMA, "manifest_file_sha256": "0" * 64, "manifest_payload_sha256": "1" * 64, "source_plan_sha256": HEX, "official_qid_registry_sha256": fixture.official_sha, "passed": True})
            with self.assertRaisesRegex(dev_proxy.DevProxyError, "split validation"):
                dev_proxy.build_report(dev_proxy.parse_args(fixture.argv(Path(td) / "report.json")))

    def test_requires_split_validation_passed_true(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            fixture = Fixture(Path(td))
            make_fake_eos(fixture.fake_eos)
            receipt = json.loads(fixture.validation.read_text(encoding="utf-8"))
            receipt["passed"] = False
            write_json(fixture.validation, receipt)
            with self.assertRaisesRegex(dev_proxy.DevProxyError, "split validation did not pass"):
                dev_proxy.build_report(dev_proxy.parse_args(fixture.argv(Path(td) / "report.json")))

    def test_rejects_official_registry_file_or_source_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            fixture = Fixture(Path(td))
            make_fake_eos(fixture.fake_eos)
            make_official(fixture.official, source_sha="d" * 64)
            with self.assertRaisesRegex(dev_proxy.DevProxyError, "official qids file sha"):
                dev_proxy.build_report(dev_proxy.parse_args(fixture.argv(Path(td) / "report.json")))

    def test_rejects_forbidden_claim_flags(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            fixture = Fixture(Path(td))
            make_fake_eos(fixture.fake_eos)
            preflight = Path(td) / "preflight.json"
            write_json(preflight, {"commercial_use_allowed": True})
            args = fixture.argv(Path(td) / "report.json") + ["--preflight", str(preflight)]
            with self.assertRaisesRegex(dev_proxy.DevProxyError, "commercial_use_allowed"):
                dev_proxy.build_report(dev_proxy.parse_args(args))

    def test_rejects_noncanonical_or_bad_orthogonality_sidecar(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            fixture = Fixture(Path(td))
            make_fake_eos(fixture.fake_eos)
            make_sidecar(fixture.sidecar, canonical_pairings=False)
            with self.assertRaisesRegex(dev_proxy.DevProxyError, "canonical D384 seed191"):
                dev_proxy.build_report(dev_proxy.parse_args(fixture.argv(Path(td) / "report.json")))
        with tempfile.TemporaryDirectory() as td:
            fixture = Fixture(Path(td))
            make_fake_eos(fixture.fake_eos)
            make_sidecar(fixture.sidecar, orthogonality=-1.0)
            with self.assertRaisesRegex(dev_proxy.DevProxyError, "orthogonality"):
                dev_proxy.build_report(dev_proxy.parse_args(fixture.argv(Path(td) / "report.json")))


if __name__ == "__main__":
    unittest.main()
