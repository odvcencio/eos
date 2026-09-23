#!/usr/bin/env python3
"""Tests for q3 top10 promotion+guard dataset derivation."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPT = SCRIPT_DIR / "build_q3top10_promotion_dataset.py"
LEGAL_GATES = {
    "train_allowed_for_research": True,
    "release_train_allowed": False,
    "commercial_use_allowed": False,
}


def sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def base_row(dataset: str, qid: str, positives: list[str], q3_negs: list[str], bm25_negs: list[str] | None = None) -> dict:
    candidate_doc_ids = [f"{dataset}:{doc_id}" for doc_id in positives + q3_negs + (bm25_negs or ["bm25"])]
    candidate_sources = ["qrel"] * len(positives) + ["q3"] * len(q3_negs) + ["bm25"] * len(bm25_negs or ["bm25"])
    qrel_gains = [1.0] * len(positives) + [0.0] * (len(q3_negs) + len(bm25_negs or ["bm25"]))
    hard = [False] * len(positives) + [True] * (len(q3_negs) + len(bm25_negs or ["bm25"]))
    target = [1.0 / len(positives)] * len(positives) + [0.0] * (len(q3_negs) + len(bm25_negs or ["bm25"]))
    return {
        "row_id": f"{dataset}:{qid}",
        "source": f"{dataset}:parent",
        "query": f"query {qid}",
        "candidate_doc_ids": candidate_doc_ids,
        "candidate_texts": [f"text {doc_id}" for doc_id in candidate_doc_ids],
        "candidate_sources": candidate_sources,
        "qrel_gains": qrel_gains,
        "positive_indexes": list(range(len(positives))),
        "selected_positive_index": 0,
        "hard_negative_eligible": hard,
        "target_probabilities": target,
        "hard_loss_weight": 0.02,
        "soft_loss_weight": 0.05,
        "recovery_loss_weight": 0.1,
        "train_policy": "parent",
        "legal_gates": dict(LEGAL_GATES),
        **LEGAL_GATES,
        "source_artifact_hash": "qrels-sha",
        "source_dataset": dataset,
        "source_query_id": qid,
        "source_positive_doc_ids": list(positives),
        "candidate_positive_doc_ids_pre_dedupe": list(positives),
        "all_train_qrel_positive_doc_ids": list(positives),
        "positive_count": len(positives),
        "bm25_negative_count": len(bm25_negs or ["bm25"]),
        "q3_negative_count": len(q3_negs),
        "extra_negative_count": 0,
        "q3_evidence": [],
        "q3_scoring_metadata": {
            "method": "turboquant_ip_b3",
            "bits": 3,
            "scoring_surface": "turboquant_ip_prepared",
            "quantizer_seed": 5581486560434873699,
        },
    }


def q3_row(dataset: str, qid: str, ranked: list[tuple[int, str, int, float]]) -> dict:
    return {
        "schema": "manta.embedding_turboquant_retrieval_per_query.v1",
        "dataset": dataset,
        "query_id": qid,
        "method": "turboquant_ip_b3",
        "bits": 3,
        "scoring_surface": "turboquant_ip_prepared",
        "quantizer_seed": 5581486560434873699,
        "top_k": [
            {
                "rank": rank,
                "doc_id": doc_id,
                "relevance": relevance,
                "score": score,
                "dense_rank": rank,
                "dense_score": score,
                "compact_rank": rank,
                "compact_score": score,
            }
            for rank, doc_id, relevance, score in ranked
        ],
    }


def write_qrels(path: Path, qids: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("query-id\tcorpus-id\tscore\n" + "".join(f"{qid}\tdoc\t1\n" for qid in qids), encoding="utf-8")


class BuildQ3Top10PromotionDatasetTest(unittest.TestCase):
    def fixture(self, root: Path) -> dict[str, Path]:
        parent = root / "parent.train.jsonl"
        q3 = root / "q3.per-query.jsonl"
        rows = [
            base_row("fiqa", "p1", ["pos11", "pos30"], ["neg4", "neg11", "neg12", "neg13", "neg21"]),
            base_row("fiqa", "g1", ["pos4"], ["neg6", "neg11", "neg13", "neg15"]),
            base_row("fiqa", "g2", ["pos5"], ["neg7", "neg11", "neg12", "neg13"]),
            base_row("scifact", "g1", ["pos3"], ["neg5", "neg11", "neg12", "neg13"]),
            base_row("scifact", "g2", ["pos6"], ["neg8", "neg11", "neg12", "neg13"]),
            base_row("nfcorpus", "g1", ["pos3"], ["neg5", "neg11", "neg12", "neg13"]),
            base_row("scifact", "skip", ["pos30"], ["neg5", "neg11", "neg12", "neg13"]),
        ]
        q3_rows = [
            q3_row("fiqa", "p1", [(4, "neg4", 0, 0.90), (11, "neg11", 0, 0.885), (12, "neg12", 0, 0.883), (13, "neg13", 0, 0.881), (14, "pos11", 1, 0.88), (21, "neg21", 0, 0.50), (30, "pos30", 1, 0.40)]),
            q3_row("fiqa", "g1", [(4, "pos4", 1, 0.80), (6, "neg6", 0, 0.81), (11, "neg11", 0, 0.79), (13, "neg13", 0, 0.70), (15, "neg15", 0, 0.69)]),
            q3_row("fiqa", "g2", [(5, "pos5", 1, 0.83), (7, "neg7", 0, 0.70), (11, "neg11", 0, 0.69), (12, "neg12", 0, 0.68), (13, "neg13", 0, 0.67)]),
            q3_row("scifact", "g1", [(3, "pos3", 1, 0.72), (5, "neg5", 0, 0.71), (11, "neg11", 0, 0.70), (12, "neg12", 0, 0.69), (13, "neg13", 0, 0.68)]),
            q3_row("scifact", "g2", [(6, "pos6", 1, 0.90), (8, "neg8", 0, 0.70), (11, "neg11", 0, 0.69), (12, "neg12", 0, 0.68), (13, "neg13", 0, 0.67)]),
            q3_row("nfcorpus", "g1", [(3, "pos3", 1, 0.60), (5, "neg5", 0, 0.61), (11, "neg11", 0, 0.59), (12, "neg12", 0, 0.58), (13, "neg13", 0, 0.57)]),
            q3_row("scifact", "skip", [(5, "neg5", 0, 0.90), (11, "neg11", 0, 0.80), (12, "neg12", 0, 0.79), (13, "neg13", 0, 0.78), (30, "pos30", 1, 0.40)]),
        ]
        write_jsonl(parent, rows)
        write_jsonl(q3, q3_rows)
        manifest = root / "parent.manifest.json"
        write_json(
            manifest,
            {
                "legal_gates": LEGAL_GATES,
                "q3_source_hashes": [f"{q3}:{sha256_file(q3)}"],
                "sha256": {"train_score_spectrum": sha256_file(parent)},
            },
        )
        dev = root / "dev.json"
        reserve = root / "reserve.json"
        write_json(dev, {"selected_qids": {"fiqa": ["dev"], "scifact": ["dev"], "nfcorpus": ["dev"]}})
        write_json(reserve, {"selected_qids": {"fiqa": ["reserve"], "scifact": ["reserve"], "nfcorpus": ["reserve"]}})
        test_fiqa = root / "raw" / "fiqa" / "fiqa" / "qrels" / "test.tsv"
        test_scifact = root / "raw" / "scifact" / "scifact" / "qrels" / "test.tsv"
        test_nf = root / "raw" / "nfcorpus" / "nfcorpus" / "qrels" / "test.tsv"
        write_qrels(test_fiqa, ["test"])
        write_qrels(test_scifact, ["test"])
        write_qrels(test_nf, ["test"])
        return {
            "parent": parent,
            "manifest": manifest,
            "q3": q3,
            "dev": dev,
            "reserve": reserve,
            "test_fiqa": test_fiqa,
            "test_scifact": test_scifact,
            "test_nf": test_nf,
            "out": root / "out.jsonl",
            "out_manifest": root / "manifest.json",
            "coverage": root / "coverage.json",
        }

    def refresh_parent_manifest_q3_hash(self, paths: dict[str, Path]) -> None:
        manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
        manifest["q3_source_hashes"] = [f"{paths['q3']}:{sha256_file(paths['q3'])}"]
        write_json(paths["manifest"], manifest)

    def run_builder(self, paths: dict[str, Path], *extra: str, check: bool = True) -> subprocess.CompletedProcess:
        return subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--parent-train-jsonl",
                str(paths["parent"]),
                "--parent-manifest",
                str(paths["manifest"]),
                "--q3-per-query-jsonl",
                str(paths["q3"]),
                "--exclude-qids-json",
                str(paths["dev"]),
                "--exclude-qids-json",
                str(paths["reserve"]),
                "--test-qrels",
                str(paths["test_fiqa"]),
                "--test-qrels",
                str(paths["test_scifact"]),
                "--test-qrels",
                str(paths["test_nf"]),
                "--output-jsonl",
                str(paths["out"]),
                "--manifest",
                str(paths["out_manifest"]),
                "--coverage-report",
                str(paths["coverage"]),
                "--expected-promotion-rows",
                "1",
                "--guard-count-per-dataset",
                "1",
                "--expected-rows",
                "3",
                "--min-q3-negatives",
                "2",
                *extra,
            ],
            text=True,
            capture_output=True,
            check=check,
        )

    def test_selects_promotions_and_top_margin_guards_without_nfcorpus_guards(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = self.fixture(Path(tmp))
            self.run_builder(paths)
            rows = read_jsonl(paths["out"])
            manifest = json.loads(paths["out_manifest"].read_text(encoding="utf-8"))

        self.assertEqual([row["row_id"] for row in rows], ["fiqa:p1", "fiqa:g1", "scifact:g1"])
        self.assertEqual(manifest["counts"]["rows"], 3)
        self.assertEqual(manifest["selection_audit"]["promotion_available"], 1)
        self.assertEqual(manifest["selection_audit"]["guards"]["fiqa"]["selected_qids"], ["g1"])
        self.assertEqual(manifest["selection_audit"]["guards"]["scifact"]["selected_qids"], ["g1"])
        self.assertEqual(manifest["legal_gates"], LEGAL_GATES)
        self.assertFalse(manifest["commercial_use_allowed"])

    def test_reindexes_alignment_and_preserves_parent_audit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = self.fixture(Path(tmp))
            self.run_builder(paths)
            promo = read_jsonl(paths["out"])[0]

        self.assertEqual(promo["candidate_doc_ids"], ["fiqa:pos11", "fiqa:neg4", "fiqa:neg11", "fiqa:neg12"])
        self.assertEqual(promo["candidate_sources"], ["qrel", "q3", "q3", "q3"])
        self.assertEqual(promo["positive_indexes"], [0])
        self.assertEqual(promo["selected_positive_index"], 0)
        self.assertEqual(promo["hard_negative_eligible"], [False, True, True, True])
        self.assertEqual(promo["target_probabilities"], [1.0, 0.0, 0.0, 0.0])
        self.assertEqual(promo["source_positive_doc_ids"], ["pos11"])
        self.assertEqual(promo["parent_audit_provenance"]["parent_source_positive_doc_ids"], ["pos11", "pos30"])
        self.assertEqual([item["q3_rank"] for item in promo["anchor_q3_candidate_evidence"]], [14, 4, 11, 12])

    def test_assigns_row_local_turboquant_topk_weights_when_requested(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = self.fixture(Path(tmp))
            self.run_builder(
                paths,
                "--promotion-turboquant-topk-loss-weight",
                "1",
                "--guard-turboquant-topk-loss-weight",
                "0",
            )
            rows = read_jsonl(paths["out"])
            manifest = json.loads(paths["out_manifest"].read_text(encoding="utf-8"))
            coverage = json.loads(paths["coverage"].read_text(encoding="utf-8"))

        weights_by_bucket = {row["selection_bucket"]: row["turboquant_topk_loss_weight"] for row in rows}
        self.assertEqual(weights_by_bucket["promotion_11_20_to_top10_near_gap"], 1.0)
        self.assertEqual(weights_by_bucket["guard_top10_margin"], 0.0)
        self.assertEqual(manifest["config"]["promotion_turboquant_topk_loss_weight"], 1.0)
        self.assertEqual(manifest["config"]["guard_turboquant_topk_loss_weight"], 0.0)
        self.assertEqual(coverage["validation"]["turboquant_topk_loss_weight_counts"], {"0.0": 2, "1.0": 1})

    def test_hash_provenance_failure_is_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = self.fixture(Path(tmp))
            with paths["q3"].open("a", encoding="utf-8") as handle:
                handle.write("\n")
            completed = self.run_builder(paths, check=False)

        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("q3 source digest set does not match", completed.stderr)

    def test_absolute_q3_source_path_replays_by_digest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = self.fixture(Path(tmp))
            paths["q3"] = paths["q3"].resolve()
            self.run_builder(paths)
            manifest = json.loads(paths["out_manifest"].read_text(encoding="utf-8"))

        supplied = manifest["inputs"]["q3_per_query_jsonl"]["supplied_sources"][0]
        parent = manifest["inputs"]["q3_per_query_jsonl"]["parent_sources"][0]
        self.assertEqual(supplied["sha256"], parent["sha256"])
        self.assertEqual(supplied["absolute_path"], str(paths["q3"]))

    def test_qrel_positive_q3_relevance_mismatch_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = self.fixture(Path(tmp))
            q3_rows = read_jsonl(paths["q3"])
            q3_rows[0]["top_k"][4]["relevance"] = 0
            write_jsonl(paths["q3"], q3_rows)
            self.refresh_parent_manifest_q3_hash(paths)
            completed = self.run_builder(paths, check=False)

        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("retained qrel positive", completed.stderr)

    def test_q3_negative_relevance_mismatch_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = self.fixture(Path(tmp))
            q3_rows = read_jsonl(paths["q3"])
            q3_rows[0]["top_k"][0]["relevance"] = 1
            write_jsonl(paths["q3"], q3_rows)
            self.refresh_parent_manifest_q3_hash(paths)
            completed = self.run_builder(paths, check=False)

        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("retained q3 negative", completed.stderr)

    def test_overlap_failure_is_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = self.fixture(Path(tmp))
            write_json(paths["dev"], {"selected_qids": {"fiqa": ["p1"]}})
            completed = self.run_builder(paths, check=False)

        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("overlap frozen/test", completed.stderr)

    def test_unexpected_legal_gates_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = self.fixture(Path(tmp))
            manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
            manifest["legal_gates"]["commercial_use_allowed"] = True
            write_json(paths["manifest"], manifest)
            completed = self.run_builder(paths, check=False)

        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("legal gates", completed.stderr)


if __name__ == "__main__":
    unittest.main()
