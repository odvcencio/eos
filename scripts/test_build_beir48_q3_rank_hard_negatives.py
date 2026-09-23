#!/usr/bin/env python3
"""Dependency-free tests for the BEIR48 q3 rank hard-negative builder."""

from __future__ import annotations

import json
import hashlib
import importlib.util
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPT = SCRIPT_DIR / "build_beir48_q3_rank_hard_negatives.py"


def load_builder_module():
    spec = importlib.util.spec_from_file_location("build_beir48_q3_rank_hard_negatives", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_tsv(path: Path, rows: list[tuple[str, str, int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("query-id\tcorpus-id\tscore\n")
        for qid, docid, score in rows:
            handle.write(f"{qid}\t{docid}\t{score}\n")


def make_dataset(root: Path, dataset: str, queries: int = 4) -> None:
    dataset_dir = root / dataset / dataset
    docs = []
    qrels = []
    query_rows = []
    for qi in range(queries):
        qid = f"q{qi}"
        positive_id = f"p{qi}"
        leak_id = f"p{qi}b"
        query_rows.append({"_id": qid, "text": f"{dataset} topic {qi} money health science"})
        docs.append({"_id": positive_id, "title": "", "text": f"{dataset} positive {qi} exact topic"})
        docs.append({"_id": leak_id, "title": "", "text": f"{dataset} second positive {qi} exact topic"})
        qrels.append((qid, positive_id, 1))
        qrels.append((qid, leak_id, 1))
        for ni in range(12):
            docs.append(
                {
                    "_id": f"n{qi}_{ni}",
                    "title": "",
                    "text": f"{dataset} topic {qi} negative {ni} money health science extra",
                }
            )
    write_jsonl(dataset_dir / "corpus.jsonl", docs)
    write_jsonl(dataset_dir / "queries.jsonl", query_rows)
    write_tsv(dataset_dir / "qrels" / "train.tsv", qrels)
    write_tsv(dataset_dir / "qrels" / "test.tsv", [("q0", "n0_0", 1)])


class BuildBEIR48Q3RankHardNegativesTest(unittest.TestCase):
    def run_builder(
        self,
        root: Path,
        out: Path,
        manifest: Path,
        *extra: str,
        rows_per_dataset: int = 2,
        check: bool = True,
    ) -> subprocess.CompletedProcess:
        return subprocess.run(
            [
                sys.executable,
                str(SCRIPT_DIR / "build_beir48_q3_rank_hard_negatives.py"),
                "--dataset-root",
                str(root),
                "--datasets",
                "scifact,nfcorpus,fiqa",
                "--rows-per-dataset",
                str(rows_per_dataset),
                "--negatives-per-row",
                "7",
                "--bm25-primary-negatives",
                "4",
                "--q3-negatives",
                "3",
                "--candidate-top-k",
                "20",
                "--seed",
                "191",
                "--output-jsonl",
                str(out),
                "--manifest",
                str(manifest),
                *extra,
            ],
            text=True,
            capture_output=True,
            check=check,
        )

    def test_builds_balanced_train_only_rows_with_bm25_fill(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for dataset in ["scifact", "nfcorpus", "fiqa"]:
                make_dataset(root, dataset)
            out = root / "out" / "hard.jsonl"
            manifest = root / "out" / "manifest.json"

            self.run_builder(root, out, manifest)
            rows = read_jsonl(out)
            summary = json.loads(manifest.read_text(encoding="utf-8"))

        self.assertEqual(len(rows), 6)
        self.assertEqual(summary["audit"]["passed"], True)
        self.assertEqual(summary["audit"]["dataset_counts"], {"fiqa": 2, "nfcorpus": 2, "scifact": 2})
        self.assertEqual(summary["counts"]["negative_source_counts"], {"bm25": 24, "bm25_fill": 18})
        self.assertEqual(summary["counts"]["q3_candidates_added"], 0)
        self.assertIn("train qrels only", summary["legal_provenance"].lower())
        for row in rows:
            self.assertEqual(row["candidate_positive_index"], 0)
            self.assertEqual(row["candidate_doc_ids"][0], row["positive_doc_id"])
            self.assertEqual(len(row["candidate_doc_ids"]), 8)
            self.assertEqual(len(row["negatives"]), 7)
            self.assertFalse(set(row["negative_doc_ids"]) & set(row["all_train_qrel_positive_doc_ids"]))
            self.assertEqual(len(set(row["candidate_doc_ids"])), 8)
            self.assertEqual(len(set(row["candidate_texts"])), 8)
            self.assertEqual(row["legal_gates"]["test_rows_train_allowed"], False)

    def test_q3_boundary_selection_uses_q3_rank_not_dense_rank(self) -> None:
        builder = load_builder_module()
        candidates = [
            {"doc_id": "dense-boundary", "q3_rank": 41, "dense_rank": 10},
            {"doc_id": "q3-boundary-low", "q3_rank": 9, "dense_rank": 80},
            {"doc_id": "q3-boundary-high", "q3_rank": 11, "dense_rank": 1},
        ]

        ordered = sorted(candidates, key=lambda item: builder.per_query_candidate_sort_key(item, "q3_boundary"))

        self.assertEqual([item["doc_id"] for item in ordered[:2]], ["q3-boundary-low", "q3-boundary-high"])

    def test_legacy_boundary_selection_still_uses_dense_rank(self) -> None:
        builder = load_builder_module()
        candidates = [
            {"doc_id": "dense-boundary", "q3_rank": 41, "dense_rank": 10},
            {"doc_id": "q3-boundary-low", "q3_rank": 9, "dense_rank": 80},
        ]

        ordered = sorted(candidates, key=lambda item: builder.per_query_candidate_sort_key(item, "boundary"))

        self.assertEqual(ordered[0]["doc_id"], "dense-boundary")

    def test_tiered_q3_recall_selection_prioritizes_boundary_then_guards_and_sentinels(self) -> None:
        builder = load_builder_module()
        candidates = [
            {"doc_id": "guard-r10", "q3_rank": 10, "dense_rank": 1},
            {"doc_id": "coverage-r21", "q3_rank": 21, "dense_rank": 2},
            {"doc_id": "sentinel-r100", "q3_rank": 100, "dense_rank": 3},
            {"doc_id": "outside-r11", "q3_rank": 11, "dense_rank": 4},
            {"doc_id": "fallback-missing", "q3_rank": 0, "dense_rank": 5},
            {"doc_id": "outside-r20", "q3_rank": 20, "dense_rank": 6},
            {"doc_id": "guard-r1", "q3_rank": 1, "dense_rank": 7},
        ]

        ordered = sorted(candidates, key=lambda item: builder.per_query_candidate_sort_key(item, "tiered_q3_recall"))

        self.assertEqual(
            [item["doc_id"] for item in ordered],
            [
                "outside-r11",
                "outside-r20",
                "guard-r10",
                "guard-r1",
                "sentinel-r100",
                "coverage-r21",
                "fallback-missing",
            ],
        )

    def test_uses_q3_candidates_after_primary_bm25_and_skips_qrel_leaks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for dataset in ["scifact", "nfcorpus", "fiqa"]:
                make_dataset(root, dataset)
            q3 = root / "q3.jsonl"
            write_jsonl(
                q3,
                [
                    {
                        "source": "scifact:q3-hard",
                        "query_id": "q0",
                        "negative_doc_ids": ["p0b", "q3a", "q3b", "q3c"],
                        "negatives": [
                            "scifact second positive 0 exact topic",
                            "q3 unique a",
                            "q3 unique b",
                            "q3 unique c",
                        ],
                    }
                ],
            )
            out = root / "out" / "hard.jsonl"
            manifest = root / "out" / "manifest.json"

            self.run_builder(
                root,
                out,
                manifest,
                "--q3-hard-negative-jsonl",
                str(q3),
                rows_per_dataset=4,
            )
            rows = read_jsonl(out)
            summary = json.loads(manifest.read_text(encoding="utf-8"))

        scifact_q0 = next(row for row in rows if row["row_id"] == "scifact:q0")
        self.assertEqual(scifact_q0["negative_sources"][:4], ["bm25", "bm25", "bm25", "bm25"])
        self.assertEqual(scifact_q0["negative_sources"][4:7], ["q3", "q3", "q3"])
        self.assertNotIn("p0b", scifact_q0["negative_doc_ids"])
        self.assertEqual(summary["counts"]["q3_candidates_added"], 3)

    def test_requires_matching_true_q3_per_query_candidates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for dataset in ["scifact", "nfcorpus", "fiqa"]:
                make_dataset(root, dataset)
            selected = root / "selected.json"
            selected.write_text(
                json.dumps({"selected_qids": {"scifact": ["q0"], "nfcorpus": ["q0"], "fiqa": ["q0"]}}),
                encoding="utf-8",
            )
            q3 = root / "q3.per-query.jsonl"
            rows = []
            for dataset in ["scifact", "nfcorpus", "fiqa"]:
                rows.append(
                    {
                        "schema": "manta.embedding_turboquant_retrieval_per_query.v1",
                        "dataset": dataset,
                        "query_id": "q0",
                        "method": "turboquant_ip_b3",
                        "bits": 3,
                        "scoring_surface": "turboquant_ip_prepared",
                        "quantizer_seed": 5581486560434873699,
                        "top_k": [
                            {"rank": 1, "doc_id": "p0b", "score": 0.9, "dense_rank": 1},
                            {"rank": 2, "doc_id": "n0_5", "score": 0.8, "dense_rank": 9},
                            {"rank": 3, "doc_id": "n0_6", "score": 0.7, "dense_rank": 20},
                            {"rank": 4, "doc_id": "n0_7", "score": 0.6, "dense_rank": 4},
                        ],
                    }
                )
            write_jsonl(q3, rows)
            q3_hash = hashlib.sha256(q3.read_bytes()).hexdigest()
            out = root / "out" / "hard.jsonl"
            manifest = root / "out" / "manifest.json"

            self.run_builder(
                root,
                out,
                manifest,
                "--selected-qids-json",
                str(selected),
                "--q3-per-query-jsonl",
                str(q3),
                "--require-q3-negatives",
                rows_per_dataset=1,
            )
            built_rows = read_jsonl(out)
            summary = json.loads(manifest.read_text(encoding="utf-8"))

        self.assertEqual(summary["counts"]["q3_per_query_candidates_loaded"], 12)
        self.assertEqual(summary["counts"]["q3_candidates_added"], 9)
        self.assertEqual(
            summary["inputs"]["q3_per_query_hashes"],
            [{"path": str(q3), "sha256": q3_hash}],
        )
        self.assertEqual(summary["inputs"]["q3_hard_negative_hashes"], [])
        self.assertEqual(summary["config"]["q3_dim"], 384)
        self.assertEqual(summary["config"]["q3_dim_source"], "caller_declared_unverified")
        self.assertEqual(summary["config"]["q3_dim_validated"], False)
        for row in built_rows:
            self.assertEqual(row["negative_sources"], ["bm25", "bm25", "bm25", "bm25", "q3", "q3", "q3"])
            self.assertNotIn("p0b", row["negative_doc_ids"])
            self.assertEqual(len(row["q3_evidence"]), 3)
            self.assertEqual(row["q3_evidence"][0]["candidate_source"], "q3_per_query")

    def test_require_q3_fails_when_matching_per_query_candidates_are_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for dataset in ["scifact", "nfcorpus", "fiqa"]:
                make_dataset(root, dataset)
            selected = root / "selected.json"
            selected.write_text(
                json.dumps({"selected_qids": {"scifact": ["q0"], "nfcorpus": ["q0"], "fiqa": ["q0"]}}),
                encoding="utf-8",
            )
            q3 = root / "q3.per-query.jsonl"
            write_jsonl(
                q3,
                [
                    {
                        "schema": "manta.embedding_turboquant_retrieval_per_query.v1",
                        "dataset": "scifact",
                        "query_id": "q0",
                        "method": "turboquant_ip_b2",
                        "bits": 2,
                        "scoring_surface": "turboquant_ip_prepared",
                        "quantizer_seed": 5581486560434873699,
                        "top_k": [{"rank": 1, "doc_id": "n0_5", "score": 0.8}],
                    }
                ],
            )
            out = root / "out" / "hard.jsonl"
            manifest = root / "out" / "manifest.json"

            result = self.run_builder(
                root,
                out,
                manifest,
                "--selected-qids-json",
                str(selected),
                "--q3-per-query-jsonl",
                str(q3),
                "--require-q3-negatives",
                rows_per_dataset=1,
                check=False,
            )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("insufficient q3 negatives", result.stderr)

    def test_rejects_non_train_split(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for dataset in ["scifact", "nfcorpus", "fiqa"]:
                make_dataset(root, dataset)
            out = root / "out" / "hard.jsonl"
            manifest = root / "out" / "manifest.json"
            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT_DIR / "build_beir48_q3_rank_hard_negatives.py"),
                    "--dataset-root",
                    str(root),
                    "--split",
                    "test",
                    "--output-jsonl",
                    str(out),
                    "--manifest",
                    str(manifest),
                ],
                text=True,
                capture_output=True,
                check=False,
            )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("train-only", result.stderr)


if __name__ == "__main__":
    unittest.main()
