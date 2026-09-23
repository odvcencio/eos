#!/usr/bin/env python3
"""Tests for BEIR qrels/BM25 score-spectrum materialization."""

from __future__ import annotations

import json
import importlib.util
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPT = SCRIPT_DIR / "build_retrieval_beir_score_spectrum.py"


def load_builder_module():
    spec = importlib.util.spec_from_file_location("build_retrieval_beir_score_spectrum", SCRIPT)
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


def write_fixture_dataset(root: Path) -> None:
    dataset = root / "raw" / "scifact"
    write_jsonl(
        dataset / "corpus.jsonl",
        [
            {"_id": "p1", "title": "alpha", "text": "needle positive immune retrieval"},
            {"_id": "p2", "title": "beta", "text": "second positive immune retrieval"},
            {"_id": "p3", "title": "zeta", "text": "resolved positive finance retrieval"},
            {"_id": "n1", "title": "gamma", "text": "immune retrieval nearby negative"},
            {"_id": "n2", "title": "delta", "text": "unrelated finance cooking"},
            {"_id": "n3", "title": "epsilon", "text": "immune retrieval model extra"},
            {"_id": "n4", "title": "eta", "text": "finance retrieval nearby negative"},
        ],
    )
    write_jsonl(
        dataset / "queries.jsonl",
        [
            {"_id": "q1", "text": "immune retrieval"},
            {"_id": "q2", "text": "finance retrieval"},
            {"_id": "q3", "text": "missing positive"},
        ],
    )
    qrels = dataset / "qrels"
    qrels.mkdir(parents=True)
    (qrels / "train.tsv").write_text(
        "query-id\tcorpus-id\tscore\n"
        "q1\tp1\t1\n"
        "q1\tp2\t2\n"
        "q1\tn2\t0\n"
        "q2\tp3\t1\n"
        "q3\tmissing\t1\n",
        encoding="utf-8",
    )


def write_empty_text_fixture_dataset(root: Path) -> None:
    dataset = root / "raw" / "scifact"
    write_jsonl(
        dataset / "corpus.jsonl",
        [
            {"_id": "title_only_positive", "title": "title only positive", "text": ""},
            {"_id": "empty_positive", "title": "", "text": ""},
            {"_id": "n1", "title": "title only", "text": "nearby negative"},
        ],
    )
    write_jsonl(
        dataset / "queries.jsonl",
        [
            {"_id": "q_title_only", "text": "title only"},
            {"_id": "q_empty", "text": ""},
            {"_id": "q_empty_doc", "text": "empty doc positive"},
        ],
    )
    qrels = dataset / "qrels"
    qrels.mkdir(parents=True)
    (qrels / "train.tsv").write_text(
        "query-id\tcorpus-id\tscore\n"
        "q_title_only\ttitle_only_positive\t1\n"
        "q_empty\tn1\t1\n"
        "q_empty_doc\tempty_positive\t1\n",
        encoding="utf-8",
    )


def write_v2_fixture_dataset(root: Path) -> tuple[Path, Path]:
    dataset = root / "raw" / "scifact"
    corpus_rows = []
    for i in range(1, 31):
        corpus_rows.append({"_id": f"d{i}", "title": f"doc {i}", "text": f"alpha beta retrieval candidate {i}"})
    write_jsonl(dataset / "corpus.jsonl", corpus_rows)
    write_jsonl(
        dataset / "queries.jsonl",
        [
            {"_id": "q1", "text": "alpha beta retrieval"},
            {"_id": "q2", "text": "alpha beta retrieval"},
        ],
    )
    qrels = dataset / "qrels"
    qrels.mkdir(parents=True)
    (qrels / "train.tsv").write_text(
        "query-id\tcorpus-id\tscore\n"
        "q1\td1\t3\n"
        "q1\td2\t2\n"
        "q1\td3\t1\n"
        "q1\td4\t1\n"
        "q1\td5\t1\n"
        "q1\td6\t1\n"
        "q2\td7\t2\n"
        "q2\td8\t1\n",
        encoding="utf-8",
    )
    selected = root / "selected.json"
    selected.write_text(json.dumps({"selected_qids": {"scifact": ["q1", "q2"]}}, sort_keys=True), encoding="utf-8")
    q3 = root / "q3.per-query.jsonl"
    write_jsonl(
        q3,
        [
            {
                "schema": "manta.embedding_turboquant_retrieval_per_query.v1",
                "dataset": "scifact",
                "query_id": query_id,
                "method": "turboquant_ip_b3",
                "bits": 3,
                "scoring_surface": "turboquant_ip_prepared",
                "quantizer_seed": 5581486560434873699,
                "top_k": [
                    {"rank": 1, "doc_id": "d20", "score": 0.9, "dense_rank": 25},
                    {"rank": 2, "doc_id": "d21", "score": 0.8, "dense_rank": 22},
                    {"rank": 3, "doc_id": "d22", "score": 0.7, "dense_rank": 21},
                    {"rank": 4, "doc_id": "d1", "score": 0.6, "dense_rank": 1},
                ],
            }
            for query_id in ("q1", "q2")
        ],
    )
    return selected, q3


class BEIRScoreSpectrumTest(unittest.TestCase):
    def run_builder(self, root: Path, extra: Path | None = None, eval_count: int = 1) -> tuple[Path, Path, Path, Path, subprocess.CompletedProcess]:
        full = root / "full.jsonl"
        train = root / "train.jsonl"
        eval_path = root / "eval.jsonl"
        excluded = root / "excluded.jsonl"
        manifest = root / "manifest.json"
        cmd = [
            sys.executable,
            str(SCRIPT),
            "--dataset-root",
            str(root / "raw"),
            "--datasets",
            "scifact",
            "--split",
            "train",
            "--output-full-jsonl",
            str(full),
            "--output-train-jsonl",
            str(train),
            "--output-eval-jsonl",
            str(eval_path),
            "--excluded-jsonl",
            str(excluded),
            "--manifest",
            str(manifest),
            "--bm25-negatives",
            "2",
            "--extra-negatives",
            "2",
            "--eval-count",
            str(eval_count),
        ]
        if extra is not None:
            cmd.extend(["--extra-hard-negative-jsonl", str(extra)])
        completed = subprocess.run(cmd, check=True, text=True, capture_output=True)
        return full, train, eval_path, manifest, completed

    def test_materializes_all_qrel_positives_unique_negatives_and_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_fixture_dataset(root)
            extra = root / "extra.jsonl"
            write_jsonl(
                extra,
                [
                    {
                        "dataset": "scifact",
                        "source": "scifact:model-hard",
                        "query_id": "q1",
                        "negative_doc_ids": ["p1", "n3"],
                        "negatives": ["needle positive immune retrieval", "immune retrieval model extra"],
                    }
                ],
            )
            full, train, eval_path, manifest, completed = self.run_builder(root, extra)
            rows = read_jsonl(full)
            train_rows = read_jsonl(train)
            eval_rows = read_jsonl(eval_path)
            summary = json.loads(manifest.read_text(encoding="utf-8"))

        self.assertIn("rows=2", completed.stdout)
        self.assertEqual(len(rows), 2)
        row = rows[0]
        self.assertEqual(row["row_id"], "scifact:q1")
        self.assertEqual(row["positive_indexes"], [0, 1])
        self.assertEqual(row["selected_positive_index"], 0)
        self.assertEqual(row["hard_negative_eligible"][:2], [False, False])
        self.assertTrue(all(row["hard_negative_eligible"][2:]))
        self.assertEqual(sum(row["target_probabilities"]), 1.0)
        self.assertEqual(row["qrel_gains"][:2], [1.0, 2.0])
        self.assertEqual(row["candidate_sources"][:2], ["qrel", "qrel"])
        self.assertEqual(row["source_positive_doc_ids"], ["p1", "p2"])
        self.assertEqual(row["candidate_positive_doc_ids_pre_dedupe"], ["p1", "p2"])
        self.assertAlmostEqual(row["target_probabilities"][0], 1.0 / 3.0)
        self.assertAlmostEqual(row["target_probabilities"][1], 2.0 / 3.0)
        self.assertEqual(len(row["candidate_doc_ids"]), len(set(row["candidate_doc_ids"])))
        self.assertEqual(len(row["candidate_texts"]), len(set(row["candidate_texts"])))
        self.assertNotIn("scifact:p1", row["candidate_doc_ids"][2:])
        self.assertIn("scifact:n3", row["candidate_doc_ids"])
        model_negative_index = row["candidate_doc_ids"].index("scifact:n3")
        self.assertGreaterEqual(model_negative_index, len(row["positive_indexes"]))
        self.assertTrue(row["hard_negative_eligible"][model_negative_index])
        self.assertEqual(row["qrel_gains"][model_negative_index], 0.0)
        self.assertEqual(row["candidate_sources"][model_negative_index], "bm25")
        self.assertEqual(len(eval_rows), 1)
        self.assertEqual(len(train_rows), 1)
        self.assertFalse({row["row_id"] for row in train_rows} & {row["row_id"] for row in eval_rows})
        self.assertEqual(summary["schema"], "eos.retrieval_beir_score_spectrum.v2")
        self.assertEqual(summary["counts"]["excluded_no_resolved_positive"], 1)
        self.assertEqual(summary["counts"]["dataset_counts"], {"scifact": 2})
        self.assertEqual(summary["counts"]["scifact_bm25_index_builds"], 1)
        self.assertEqual(summary["counts"]["scifact_bm25_index_docs"], 7)
        self.assertEqual(summary["counts"]["scifact_empty_corpus_docs"], 0)
        self.assertEqual(summary["counts"]["scifact_empty_queries"], 0)
        self.assertIn("all qrel positives when <= cap", summary["policy"])

    def test_source_positive_doc_ids_reflect_actual_appended_deduped_qrels(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "raw" / "scifact"
            write_jsonl(
                dataset / "corpus.jsonl",
                [
                    {"_id": "p1", "title": "", "text": "duplicate positive text"},
                    {"_id": "p2", "title": "", "text": "duplicate positive text"},
                    {"_id": "n1", "title": "", "text": "duplicate negative one"},
                    {"_id": "n2", "title": "", "text": "duplicate negative two"},
                ],
            )
            write_jsonl(dataset / "queries.jsonl", [{"_id": "q1", "text": "duplicate"}])
            (dataset / "qrels").mkdir(parents=True)
            (dataset / "qrels" / "train.tsv").write_text(
                "query-id\tcorpus-id\tscore\n"
                "q1\tp1\t1\n"
                "q1\tp2\t2\n",
                encoding="utf-8",
            )

            full, _train, _eval, _manifest, _completed = self.run_builder(root, eval_count=0)
            row = read_jsonl(full)[0]

        self.assertEqual(row["positive_indexes"], [0])
        self.assertEqual(row["source_positive_doc_ids"], ["p1"])
        self.assertEqual(row["candidate_positive_doc_ids_pre_dedupe"], ["p1", "p2"])
        self.assertEqual(row["all_train_qrel_positive_doc_ids"], ["p1", "p2"])

    def test_title_only_docs_survive_empty_docs_and_queries_are_audited(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_empty_text_fixture_dataset(root)
            full, train, eval_path, manifest, completed = self.run_builder(root, eval_count=0)
            rows = read_jsonl(full)
            train_rows = read_jsonl(train)
            eval_rows = read_jsonl(eval_path)
            excluded = read_jsonl(root / "excluded.jsonl")
            summary = json.loads(manifest.read_text(encoding="utf-8"))

        self.assertIn("rows=1", completed.stdout)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["row_id"], "scifact:q_title_only")
        self.assertEqual(rows[0]["candidate_doc_ids"][0], "scifact:title_only_positive")
        self.assertEqual(rows[0]["candidate_texts"][0], "title only positive")
        self.assertEqual(len(train_rows), 1)
        self.assertEqual(eval_rows, [])
        self.assertEqual(excluded, [{"dataset": "scifact", "query_id": "q_empty_doc", "reason": "no_resolved_positive"}])
        self.assertEqual(summary["counts"]["scifact_empty_corpus_docs"], 1)
        self.assertEqual(summary["counts"]["scifact_empty_queries"], 1)
        self.assertEqual(summary["counts"]["scifact_bm25_index_docs"], 2)
        self.assertEqual(summary["counts"]["excluded_no_resolved_positive"], 1)

    def test_missing_id_remains_hard_error(self) -> None:
        builder = load_builder_module()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "broken.jsonl"
            write_jsonl(path, [{"text": "has text but no id"}])
            with self.assertRaisesRegex(ValueError, "missing _id"):
                builder.read_beir_jsonl(path, "text")

    def test_indexed_bm25_matches_naive_reference_ranking(self) -> None:
        builder = load_builder_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_fixture_dataset(root)
            dataset_dir = root / "raw" / "scifact"
            docs = builder.read_beir_jsonl(dataset_dir / "corpus.jsonl", "text")
            queries = builder.read_beir_jsonl(dataset_dir / "queries.jsonl", "text")
            qrels = builder.read_positive_qrels(dataset_dir / "qrels" / "train.tsv")
            index = builder.BM25Index(docs)

        for qid in ("q1", "q2"):
            positive_ids = {doc_id for doc_id in qrels[qid] if doc_id in docs}
            self.assertEqual(
                builder.bm25_rank(queries[qid], index, positive_ids, 10),
                builder.bm25_rank_naive(queries[qid], docs, positive_ids, 10),
            )

    def test_build_rows_builds_one_bm25_index_per_dataset(self) -> None:
        builder = load_builder_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_fixture_dataset(root)
            args = type(
                "Args",
                (),
                {
                    "dataset_root": root / "raw",
                    "datasets": "scifact",
                    "split": "train",
                    "extra_hard_negative_jsonl": [],
                    "max_queries_per_dataset": 0,
                    "bm25_negatives": 2,
                    "extra_negatives": 0,
                    "hard_loss_weight": 1.0,
                    "soft_loss_weight": 0.1,
                    "recovery_loss_weight": 1.0,
                },
            )()
            rows, _excluded, counts = builder.build_rows(args)

        self.assertEqual(len(rows), 2)
        self.assertEqual(counts["scifact_bm25_index_builds"], 1)
        self.assertEqual(counts["scifact_bm25_index_docs"], 7)

    def test_clamps_eval_count_to_preserve_training_row(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_fixture_dataset(root)
            full, train, eval_path, manifest, _completed = self.run_builder(root)
            train_rows = read_jsonl(train)
            eval_rows = read_jsonl(eval_path)
            summary = json.loads(manifest.read_text(encoding="utf-8"))
            full_rows = read_jsonl(full)

        self.assertEqual(len(full_rows), 2)
        self.assertEqual(len(train_rows), 1)
        self.assertEqual(len(eval_rows), 1)
        self.assertFalse({row["row_id"] for row in train_rows} & {row["row_id"] for row in eval_rows})
        self.assertEqual(summary["counts"]["train_rows"], 1)
        self.assertEqual(summary["counts"]["eval_rows"], 1)

    def test_one_row_with_eval_count_fails_instead_of_overlapping(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_fixture_dataset(root)
            qrels = root / "raw" / "scifact" / "qrels" / "train.tsv"
            qrels.write_text(
                "query-id\tcorpus-id\tscore\n"
                "q1\tp1\t1\n",
                encoding="utf-8",
            )
            full = root / "full.jsonl"
            train = root / "train.jsonl"
            eval_path = root / "eval.jsonl"
            excluded = root / "excluded.jsonl"
            manifest = root / "manifest.json"
            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--dataset-root",
                    str(root / "raw"),
                    "--datasets",
                    "scifact",
                    "--split",
                    "train",
                    "--output-full-jsonl",
                    str(full),
                    "--output-train-jsonl",
                    str(train),
                    "--output-eval-jsonl",
                    str(eval_path),
                    "--excluded-jsonl",
                    str(excluded),
                    "--manifest",
                    str(manifest),
                    "--bm25-negatives",
                    "1",
                    "--eval-count",
                    "1",
                ],
                check=False,
                text=True,
                capture_output=True,
            )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("at least two score-spectrum rows", result.stderr)

    def test_output_is_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_fixture_dataset(root)
            full_a, _train_a, _eval_a, manifest_a, _ = self.run_builder(root)
            payload_a = full_a.read_text(encoding="utf-8")
            summary_a = json.loads(manifest_a.read_text(encoding="utf-8"))
            full_b, _train_b, _eval_b, manifest_b, _ = self.run_builder(root)
            payload_b = full_b.read_text(encoding="utf-8")
            summary_b = json.loads(manifest_b.read_text(encoding="utf-8"))

        self.assertEqual(payload_a, payload_b)
        self.assertEqual(summary_a["sha256"], summary_b["sha256"])

    def test_q3_boundary_selection_uses_q3_rank_not_dense_rank(self) -> None:
        builder = load_builder_module()
        candidates = [
            {"doc_id": "dense-boundary", "q3_rank": 41, "dense_rank": 10},
            {"doc_id": "q3-boundary-low", "q3_rank": 9, "dense_rank": 80},
            {"doc_id": "q3-boundary-high", "q3_rank": 11, "dense_rank": 1},
            {"doc_id": "q3-far-top", "q3_rank": 1, "dense_rank": 99},
        ]

        ordered = sorted(candidates, key=lambda item: builder.per_query_candidate_sort_key(item, "q3_boundary"))

        self.assertEqual([item["doc_id"] for item in ordered[:2]], ["q3-boundary-low", "q3-boundary-high"])
        self.assertLess(
            builder.per_query_candidate_sort_key({"doc_id": "q3-r10", "q3_rank": 10, "dense_rank": 100}, "q3_boundary"),
            builder.per_query_candidate_sort_key({"doc_id": "dense-r10", "q3_rank": 41, "dense_rank": 10}, "q3_boundary"),
        )

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

    def test_tiered_q3_recall_positive_selection_uses_exact_cap12_quotas_rollover_and_fallback(self) -> None:
        builder = load_builder_module()
        positives = [
            "r1",
            "r9",
            "r10",
            "r11",
            "r12",
            "r13",
            "r14",
            "r20",
            "r21",
            "r22",
            "r99",
            "r100",
            "missing",
            "extra",
        ]
        q3_candidates = [{"doc_id": doc_id, "q3_rank": int(doc_id[1:])} for doc_id in positives if doc_id.startswith("r")]
        gains = {doc_id: float(index) for index, doc_id in enumerate(positives)}

        selected = builder.select_qrel_positive_ids(positives, gains, q3_candidates, 12, "tiered_q3_recall")

        self.assertEqual(
            selected,
            ["r11", "r12", "r13", "r14", "r10", "r9", "r1", "r100", "r99", "r21", "r20", "r22"],
        )

    def test_gain_q3_boundary_positive_selection_prefers_q3_boundary_relevant_docs(self) -> None:
        builder = load_builder_module()
        positives = ["p1", "p2", "p3", "p4"]
        gains = {"p1": 9.0, "p2": 1.0, "p3": 2.0, "p4": 3.0}
        q3_candidates = [
            {"doc_id": "p1", "q3_rank": 1},
            {"doc_id": "p2", "q3_rank": 11},
            {"doc_id": "p3", "q3_rank": 9},
        ]

        selected = builder.select_qrel_positive_ids(positives, gains, q3_candidates, 2, "gain_q3_boundary")

        self.assertEqual(selected, ["p2", "p3"])

    def test_default_positive_selection_preserves_doc_id_cap(self) -> None:
        builder = load_builder_module()

        selected = builder.select_qrel_positive_ids(["p1", "p2", "p3"], {"p3": 9.0}, [], 2, "doc_id")

        self.assertEqual(selected, ["p1", "p2"])

    def test_v2_balanced_rows_emit_aligned_gains_sources_and_q3_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            selected, q3 = write_v2_fixture_dataset(root)
            full = root / "full.jsonl"
            train = root / "train.jsonl"
            eval_path = root / "eval.jsonl"
            excluded = root / "excluded.jsonl"
            manifest = root / "manifest.json"
            subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--dataset-root",
                    str(root / "raw"),
                    "--datasets",
                    "scifact",
                    "--split",
                    "train",
                    "--output-full-jsonl",
                    str(full),
                    "--output-train-jsonl",
                    str(train),
                    "--output-eval-jsonl",
                    str(eval_path),
                    "--excluded-jsonl",
                    str(excluded),
                    "--manifest",
                    str(manifest),
                    "--rows-per-dataset",
                    "2",
                    "--candidate-cap",
                    "20",
                    "--q3-negatives",
                    "3",
                    "--bm25-negatives",
                    "24",
                    "--eval-count",
                    "0",
                    "--require-q3-negatives",
                    "--selected-qids-json",
                    str(selected),
                    "--q3-per-query-jsonl",
                    str(q3),
                ],
                check=True,
                text=True,
                capture_output=True,
            )
            rows = read_jsonl(train)
            summary = json.loads(manifest.read_text(encoding="utf-8"))

        self.assertEqual(len(rows), 2)
        self.assertEqual(summary["schema"], "eos.retrieval_beir_score_spectrum.v2")
        self.assertEqual(summary["counts"]["rows"], 2)
        self.assertEqual(summary["counts"]["train_rows"], 2)
        self.assertEqual(summary["counts"]["eval_rows"], 0)
        self.assertEqual(summary["counts"]["min_candidate_count"], 20)
        self.assertEqual(summary["counts"]["max_candidate_count"], 20)
        self.assertEqual(summary["counts"]["candidate_source_counts"]["q3"], 6)
        self.assertEqual(summary["counts"]["scifact_qrel_positive_cap_applied"], 1)
        self.assertEqual(summary["config"]["qrel_positive_cap"], 5)
        self.assertEqual(summary["config"]["qrel_positive_selection"], "doc_id")
        self.assertEqual(summary["config"]["bm25_negatives"], 24)
        self.assertEqual(summary["config"]["extra_negatives"], 12)
        self.assertEqual(summary["config"]["eval_count"], 0)
        self.assertEqual(summary["config"]["max_queries_per_dataset"], 0)
        self.assertEqual(summary["config"]["hard_loss_weight"], 1.0)
        self.assertEqual(summary["config"]["soft_loss_weight"], 0.1)
        self.assertEqual(summary["config"]["recovery_loss_weight"], 1.0)
        for row in rows:
            self.assertEqual(len(row["candidate_doc_ids"]), 20)
            self.assertEqual(len(row["candidate_sources"]), 20)
            self.assertEqual(len(row["qrel_gains"]), 20)
            self.assertEqual(len(row["hard_negative_eligible"]), 20)
            self.assertEqual(row["candidate_sources"][: len(row["positive_indexes"])], ["qrel"] * len(row["positive_indexes"]))
            self.assertTrue(all(gain > 0 for gain in row["qrel_gains"][: len(row["positive_indexes"])]))
            self.assertTrue(all(gain == 0 for gain in row["qrel_gains"][len(row["positive_indexes"]) :]))
            self.assertEqual(row["candidate_sources"][len(row["positive_indexes"]) : len(row["positive_indexes"]) + 3], ["q3", "q3", "q3"])
            self.assertFalse(set(row["candidate_doc_ids"][len(row["positive_indexes"]) :]) & {f"scifact:{doc_id}" for doc_id in row["all_train_qrel_positive_doc_ids"]})
            self.assertEqual(row["q3_scoring_metadata"]["method"], "turboquant_ip_b3")
            self.assertTrue(row["q3_scoring_metadata"]["source_hashes"])


if __name__ == "__main__":
    unittest.main()
