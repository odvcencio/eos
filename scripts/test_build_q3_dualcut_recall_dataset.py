#!/usr/bin/env python3
"""Tests for q3-near dual-cut recall-sentinel dataset derivation."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPT = SCRIPT_DIR / "build_q3_dualcut_recall_dataset.py"
LEGAL_GATES = {
    "train_allowed_for_research": True,
    "release_train_allowed": False,
    "commercial_use_allowed": False,
}


def sha256_file(path: Path) -> str:
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


def base_row(dataset: str, qid: str, bucket: str) -> dict:
    candidate_doc_ids = [f"{dataset}:pos-{qid}", f"{dataset}:neg-{qid}"]
    return {
        "row_id": f"{dataset}:{qid}",
        "source": f"{dataset}:base",
        "query": f"query {qid}",
        "candidate_doc_ids": candidate_doc_ids,
        "candidate_texts": [f"text {doc_id}" for doc_id in candidate_doc_ids],
        "candidate_sources": ["qrel", "q3"],
        "qrel_gains": [1.0, 0.0],
        "positive_indexes": [0],
        "selected_positive_index": 0,
        "hard_negative_eligible": [False, True],
        "target_probabilities": [1.0, 0.0],
        "hard_loss_weight": 0.02,
        "soft_loss_weight": 0.05,
        "recovery_loss_weight": 0.1,
        "train_policy": "q3top10_promotion_guard_research_only_parent_reviewed",
        "selection_bucket": bucket,
        "legal_gates": dict(LEGAL_GATES),
        **LEGAL_GATES,
        "source_artifact_hash": "base-source-sha",
        "source_dataset": dataset,
        "source_query_id": qid,
        "source_positive_doc_ids": [f"pos-{qid}"],
        "candidate_positive_doc_ids_pre_dedupe": [f"pos-{qid}"],
        "all_train_qrel_positive_doc_ids": [f"pos-{qid}"],
        "positive_count": 1,
        "bm25_negative_count": 0,
        "q3_negative_count": 1,
        "extra_negative_count": 0,
        "q3_evidence": [{"doc_id": f"neg-{qid}", "q3_rank": 1, "q3_score": 0.5, "relevance": 0}],
        "anchor_q3_candidate_evidence": [
            {"doc_id": f"pos-{qid}", "q3_rank": 11, "q3_score": 0.49, "relevance": 1},
            {"doc_id": f"neg-{qid}", "q3_rank": 1, "q3_score": 0.5, "relevance": 0},
        ],
        "q3_scoring_metadata": {
            "method": "turboquant_ip_b3",
            "bits": 3,
            "scoring_surface": "turboquant_ip_prepared",
            "quantizer_seed": 5581486560434873699,
        },
    }


def q3_row(qid: str, late_hits: dict[int, tuple[str, int]] | None = None, early_positive: str | None = None) -> dict:
    late_hits = late_hits or {}
    top_k = []
    for rank in range(1, 101):
        doc_id = f"DOC-{qid}-{rank}"
        relevance = 0
        if rank in late_hits:
            doc_id, relevance = late_hits[rank]
        elif early_positive and rank == 3:
            doc_id, relevance = early_positive, 1
        top_k.append(
            {
                "rank": rank,
                "doc_id": doc_id,
                "score": round(1.0 - rank / 1000.0, 6),
                "relevance": relevance,
                "dense_rank": rank,
                "dense_score": round(1.0 - rank / 900.0, 6),
                "compact_rank": rank,
                "compact_score": round(1.0 - rank / 1000.0, 6),
            }
        )
    return {
        "schema": "manta.embedding_turboquant_retrieval_per_query.v1",
        "dataset": "nfcorpus",
        "query_id": qid,
        "method": "turboquant_ip_b3",
        "bits": 3,
        "scoring_surface": "turboquant_ip_prepared",
        "quantizer_seed": 5581486560434873699,
        "relevant_count": 1,
        "first_relevant_rank": min(late_hits) if late_hits else 3,
        "quality": {},
        "candidate_count": 1000,
        "candidates_scored": 1000,
        "pruning_supported": True,
        "pruning_used": True,
        "top_k": top_k,
    }


def normalize_candidate_text(text: str) -> str:
    return " ".join(str(text).lower().split())


def write_qrels(path: Path, qrels: dict[str, dict[str, int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["query-id\tcorpus-id\tscore\n"]
    for qid in sorted(qrels):
        for doc_id, gain in sorted(qrels[qid].items()):
            lines.append(f"{qid}\t{doc_id}\t{gain}\n")
    path.write_text("".join(lines), encoding="utf-8")


class BuildQ3DualcutRecallDatasetTest(unittest.TestCase):
    def fixture(self, root: Path, sentinel_count: int = 4) -> dict[str, Path | str | int]:
        base = root / "base.train.jsonl"
        base_rows = [
            base_row("nfcorpus", "PLAIN-63", "promotion_11_20_to_top10_near_gap"),
            base_row("nfcorpus", "PLAIN-72", "guard_top10_margin"),
            base_row("fiqa", "25", "promotion_11_20_to_top10_near_gap"),
        ]
        write_jsonl(base, base_rows)
        q3 = root / "anchor.nfcorpus.train384.q3.per-query.jsonl"
        q3_rows = [
            q3_row("PLAIN-63", {84: ("MED-5329", 2), 88: ("MED-4523", 1), 100: ("MED-5328", 2)}),
            q3_row("PLAIN-72", {99: ("MED-5328", 2)}),
            q3_row("PLAIN-80", {100: ("MED-8000", 1)}),
            q3_row("PLAIN-90", {98: ("MED-9000", 1)}),
            q3_row("PLAIN-12", {96: ("MED-1200", 1)}),
            q3_row("PLAIN-early", {}, "MED-early"),
        ]
        q3_rows[1]["top_k"][3]["doc_id"] = "MED-2921"
        q3_rows[1]["top_k"][4]["doc_id"] = "MED-3019"
        q3_rows[1]["top_k"][4]["relevance"] = 1
        write_jsonl(q3, q3_rows)
        manifest = root / "base.manifest.json"
        write_json(
            manifest,
            {
                "legal_gates": LEGAL_GATES,
                "sha256": {"train_score_spectrum": sha256_file(base)},
                "inputs": {
                    "q3_per_query_jsonl": {
                        "parent_sources": [{"parent_path": str(q3), "sha256": sha256_file(q3)}]
                    }
                },
            },
        )
        coverage = root / "base.coverage.json"
        write_json(coverage, {"legal_gates": LEGAL_GATES, "schema": "base.coverage"})
        query_rows = [{"_id": row["query_id"], "text": f"query text {row['query_id']}"} for row in q3_rows]
        queries = root / "queries.jsonl"
        write_jsonl(queries, query_rows)
        doc_ids = {candidate["doc_id"] for row in q3_rows for candidate in row["top_k"]}
        corpus = root / "corpus.jsonl"
        duplicate_mercury_text = "Evidence on the Human Health Effects of Low-Level Methylmercury Exposure"
        corpus_rows = []
        for doc_id in sorted(doc_ids):
            if doc_id in {"MED-2921", "MED-3019"}:
                corpus_rows.append({"_id": doc_id, "title": "", "text": duplicate_mercury_text})
            else:
                corpus_rows.append({"_id": doc_id, "title": f"title {doc_id}", "text": f"body {doc_id}"})
        write_jsonl(corpus, corpus_rows)
        qrels = {
            "PLAIN-63": {"MED-5329": 2, "MED-4523": 1, "MED-5328": 2},
            "PLAIN-72": {"MED-3019": 1, "MED-5328": 2},
            "PLAIN-80": {"MED-8000": 1},
            "PLAIN-90": {"MED-9000": 1},
            "PLAIN-12": {"MED-1200": 1},
            "PLAIN-early": {"MED-early": 1},
        }
        train_qrels = root / "train.tsv"
        test_qrels = root / "test.tsv"
        write_qrels(train_qrels, qrels)
        write_qrels(test_qrels, {"PLAIN-test": {"DOC-test": 1}})
        dev = root / "dev.json"
        reserve = root / "reserve.json"
        write_json(dev, {"selected_qids": {"nfcorpus": ["PLAIN-dev"]}})
        write_json(reserve, {"selected_qids": {"nfcorpus": ["PLAIN-reserve"]}})
        return {
            "base": base,
            "base_sha": sha256_file(base),
            "manifest": manifest,
            "coverage": coverage,
            "q3": q3,
            "train_qrels": train_qrels,
            "queries": queries,
            "corpus": corpus,
            "dev": dev,
            "reserve": reserve,
            "test_qrels": test_qrels,
            "out": root / "out.jsonl",
            "out_manifest": root / "manifest.json",
            "out_coverage": root / "coverage.json",
            "base_rows": len(base_rows),
            "sentinel_count": sentinel_count,
        }

    def run_builder(self, paths: dict[str, Path | str | int], *extra: str, check: bool = True) -> subprocess.CompletedProcess:
        expected_rows = int(paths["base_rows"]) + int(paths["sentinel_count"])
        return subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--base-train-jsonl",
                str(paths["base"]),
                "--base-manifest",
                str(paths["manifest"]),
                "--base-coverage",
                str(paths["coverage"]),
                "--q3-per-query-jsonl",
                str(paths["q3"]),
                "--train-qrels",
                str(paths["train_qrels"]),
                "--queries-jsonl",
                str(paths["queries"]),
                "--corpus-jsonl",
                str(paths["corpus"]),
                "--exclude-qids-json",
                str(paths["dev"]),
                "--exclude-qids-json",
                str(paths["reserve"]),
                "--test-qrels",
                str(paths["test_qrels"]),
                "--output-jsonl",
                str(paths["out"]),
                "--manifest",
                str(paths["out_manifest"]),
                "--coverage-report",
                str(paths["out_coverage"]),
                "--expected-base-sha256",
                str(paths["base_sha"]),
                "--expected-base-rows",
                str(paths["base_rows"]),
                "--sentinel-count",
                str(paths["sentinel_count"]),
                "--expected-rows",
                str(expected_rows),
                *extra,
            ],
            text=True,
            capture_output=True,
            check=check,
        )

    def test_deterministic_selection_exact_counts_weights_and_required_inclusion(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = self.fixture(Path(tmp))
            self.run_builder(paths)
            rows = read_jsonl(paths["out"])  # type: ignore[arg-type]
            manifest = json.loads(Path(paths["out_manifest"]).read_text(encoding="utf-8"))
            coverage = json.loads(Path(paths["out_coverage"]).read_text(encoding="utf-8"))

        sentinels = [row for row in rows if row["selection_bucket"] == "nfcorpus_recall_sentinel_q3top100_rank80_100"]
        self.assertEqual(len(rows), 7)
        self.assertEqual([row["source_query_id"] for row in sentinels], ["PLAIN-63", "PLAIN-72", "PLAIN-80", "PLAIN-90"])
        self.assertEqual(manifest["counts"]["turboquant_topk_loss_weight_counts"], {"0.0": 5, "1.0": 2})
        self.assertEqual(manifest["counts"]["turboquant_topk_recall_loss_weight_counts"], {"0.0": 3, "1.0": 4})
        self.assertEqual(manifest["counts"]["positive_text_alias_count"], 1)
        self.assertEqual(manifest["counts"]["positive_text_alias_rows"], {"nfcorpus:PLAIN-72:nfrecall24-q3top100": 1})
        plain63 = next(row for row in sentinels if row["source_query_id"] == "PLAIN-63")
        self.assertEqual(len(plain63["candidate_doc_ids"]), 100)
        self.assertIn("MED-5329", plain63["source_positive_doc_ids"])
        self.assertIn("MED-4523", plain63["source_positive_doc_ids"])
        self.assertTrue(coverage["required_qids"]["PLAIN-63"])
        self.assertTrue(coverage["required_doc_ids_in_sentinel_positives"]["MED-5329"])
        self.assertTrue(coverage["required_doc_ids_in_sentinel_positives"]["MED-4523"])

    def test_alignment_and_old_new_provenance_are_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = self.fixture(Path(tmp))
            self.run_builder(paths)
            rows = read_jsonl(paths["out"])  # type: ignore[arg-type]

        base_copy = rows[0]
        sentinel = next(row for row in rows if row["source_query_id"] == "PLAIN-63" and row["selection_bucket"].startswith("nfcorpus_recall"))
        self.assertEqual(base_copy["dualcut_parent_provenance"]["base_row_id"], "nfcorpus:PLAIN-63")
        self.assertEqual(base_copy["train_policy"], "q3near_dualcut_research_only_base_copy")
        self.assertIn("recall_sentinel_provenance", sentinel)
        self.assertEqual(sentinel["recall_sentinel_provenance"]["rank80_100_positive_hits"][0]["doc_id"], "MED-5329")
        aligned = ["candidate_doc_ids", "candidate_texts", "candidate_sources", "qrel_gains", "hard_negative_eligible", "target_probabilities"]
        self.assertEqual({len(sentinel[field]) for field in aligned}, {100})
        self.assertEqual(sentinel["positive_indexes"], [83, 87, 99])
        self.assertAlmostEqual(sum(sentinel["target_probabilities"]), 1.0)
        self.assertEqual([sentinel["anchor_q3_candidate_evidence"][idx]["q3_rank"] for idx in sentinel["positive_indexes"]], [84, 88, 100])

    def test_positive_text_duplicate_alias_is_never_hard_negative_eligible(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = self.fixture(Path(tmp))
            self.run_builder(paths)
            rows = read_jsonl(paths["out"])  # type: ignore[arg-type]

        row = next(row for row in rows if row["source_query_id"] == "PLAIN-72" and row["selection_bucket"].startswith("nfcorpus_recall"))
        self.assertEqual(row["candidate_doc_ids"][3], "nfcorpus:MED-2921")
        self.assertEqual(row["candidate_doc_ids"][4], "nfcorpus:MED-3019")
        self.assertEqual(normalize_candidate_text(row["candidate_texts"][3]), normalize_candidate_text(row["candidate_texts"][4]))
        self.assertNotIn(3, row["positive_indexes"])
        self.assertIn(4, row["positive_indexes"])
        self.assertFalse(row["hard_negative_eligible"][3])
        self.assertFalse(row["hard_negative_eligible"][4])
        self.assertEqual(row["qrel_gains"][3], 0.0)
        self.assertEqual(row["qrel_gains"][4], 1.0)
        self.assertEqual(row["q3_positive_text_alias_candidate_indexes"], [3])
        self.assertNotIn("MED-2921", [item["doc_id"] for item in row["q3_evidence"]])
        self.assertEqual(len(row["candidate_doc_ids"]), 100)

    def test_every_generated_row_has_runtime_merge_safe_labels(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = self.fixture(Path(tmp))
            self.run_builder(paths)
            rows = read_jsonl(paths["out"])  # type: ignore[arg-type]

        for row in rows:
            positive_indexes = set(row["positive_indexes"])
            self.assertEqual(
                positive_indexes,
                {index for index, gain in enumerate(row["qrel_gains"]) if float(gain) > 0.0},
                row["row_id"],
            )
            by_text: dict[str, list[int]] = {}
            for index, text in enumerate(row["candidate_texts"]):
                by_text.setdefault(normalize_candidate_text(text), []).append(index)
            for indexes in by_text.values():
                if not positive_indexes.intersection(indexes):
                    continue
                self.assertFalse(
                    any(row["hard_negative_eligible"][index] for index in indexes),
                    f"{row['row_id']} duplicate text indexes {indexes}",
                )

    def test_hash_mismatch_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = self.fixture(Path(tmp))
            paths["base_sha"] = "0" * 64
            completed = self.run_builder(paths, check=False)

        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("base train sha mismatch", completed.stderr)

    def test_missing_input_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = self.fixture(Path(tmp))
            Path(paths["coverage"]).unlink()
            completed = self.run_builder(paths, check=False)

        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("missing base input", completed.stderr)

    def test_overlap_failure_is_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = self.fixture(Path(tmp))
            write_json(Path(paths["dev"]), {"selected_qids": {"nfcorpus": ["PLAIN-63"]}})
            completed = self.run_builder(paths, check=False)

        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("required qids not eligible after exclusions", completed.stderr)

    def test_legal_gate_failure_is_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = self.fixture(Path(tmp))
            manifest = json.loads(Path(paths["manifest"]).read_text(encoding="utf-8"))
            manifest["legal_gates"]["commercial_use_allowed"] = True
            write_json(Path(paths["manifest"]), manifest)
            completed = self.run_builder(paths, check=False)

        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("legal gates", completed.stderr)

    def test_q3_relevance_mismatch_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = self.fixture(Path(tmp))
            q3_rows = read_jsonl(Path(paths["q3"]))
            q3_rows[0]["top_k"][83]["relevance"] = 0
            write_jsonl(Path(paths["q3"]), q3_rows)
            manifest = json.loads(Path(paths["manifest"]).read_text(encoding="utf-8"))
            manifest["inputs"]["q3_per_query_jsonl"]["parent_sources"][0]["sha256"] = sha256_file(Path(paths["q3"]))
            write_json(Path(paths["manifest"]), manifest)
            completed = self.run_builder(paths, check=False)

        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("q3 relevance/qrels mismatch", completed.stderr)

    def test_q3_source_digest_mismatch_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = self.fixture(Path(tmp))
            q3_rows = read_jsonl(Path(paths["q3"]))
            q3_rows.append(q3_row("PLAIN-extra", {100: ("MED-extra", 1)}))
            write_jsonl(Path(paths["q3"]), q3_rows)
            completed = self.run_builder(paths, check=False)

        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("q3 source digest not present", completed.stderr)


if __name__ == "__main__":
    unittest.main()
