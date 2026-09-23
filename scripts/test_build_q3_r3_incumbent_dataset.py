#!/usr/bin/env python3
"""Tests for the q3 R3 PLAIN-73 incumbent-preservation dataset builder."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPT = SCRIPT_DIR / "build_q3_r3_incumbent_dataset.py"
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


def plain73_row(row_id: str = "nfcorpus:PLAIN-73") -> dict:
    docs = ["MED-3022", "MED-3026", "MED-3031", "MED-3019", "MED-3020", "MED-3023", "MED-3032", "MED-2910", "MED-3025", "MED-3034", "MED-4622"]
    return {
        "row_id": row_id,
        "source": "nfcorpus:base",
        "query": "PLAIN-73 query",
        "candidate_doc_ids": [f"nfcorpus:{doc}" for doc in docs],
        "candidate_texts": [f"text {doc}" for doc in docs],
        "candidate_sources": ["qrel"] * 7 + ["q3"] * 4,
        "qrel_gains": [1.0] * 7 + [0.0] * 4,
        "positive_indexes": list(range(7)),
        "selected_positive_index": 0,
        "hard_negative_eligible": [False] * 7 + [True] * 4,
        "target_probabilities": [1 / 7] * 7 + [0.0] * 4,
        "hard_loss_weight": 0.1,
        "soft_loss_weight": 0.2,
        "recovery_loss_weight": 0.3,
        "turboquant_topk_loss_weight": 1.0,
        "turboquant_topk_recall_loss_weight": 0.0,
        "train_policy": "q3near_dualcut_research_only_base_copy",
        "selection_bucket": "promotion_11_20_to_top10_near_gap",
        "guard_margin": 0.0,
        "legal_gates": dict(LEGAL_GATES),
        **LEGAL_GATES,
        "source_artifact_hash": "parent",
        "source_dataset": "nfcorpus",
        "source_query_id": "PLAIN-73",
        "source_positive_doc_ids": docs[:7],
        "candidate_positive_doc_ids_pre_dedupe": docs[:7],
        "all_train_qrel_positive_doc_ids": docs[:7],
        "positive_count": 7,
        "bm25_negative_count": 0,
        "q3_negative_count": 4,
        "extra_negative_count": 0,
        "q3_evidence": [{"doc_id": doc, "q3_rank": idx + 1, "q3_score": 1 - idx / 100, "relevance": 0} for idx, doc in enumerate(docs[7:])],
        "anchor_q3_candidate_evidence": [
            {"doc_id": doc, "q3_rank": idx + 1, "q3_score": 1 - idx / 100, "relevance": 1 if idx < 7 else 0}
            for idx, doc in enumerate(docs)
        ],
    }


def other_row() -> dict:
    row = plain73_row("nfcorpus:PLAIN-1")
    row["source_query_id"] = "PLAIN-1"
    row["query"] = "other query"
    row["candidate_doc_ids"] = ["nfcorpus:OTHER-P", "nfcorpus:OTHER-N"]
    row["candidate_texts"] = ["other positive", "other negative"]
    row["candidate_sources"] = ["qrel", "q3"]
    row["qrel_gains"] = [1.0, 0.0]
    row["positive_indexes"] = [0]
    row["hard_negative_eligible"] = [False, True]
    row["target_probabilities"] = [1.0, 0.0]
    row["source_positive_doc_ids"] = ["OTHER-P"]
    row["candidate_positive_doc_ids_pre_dedupe"] = ["OTHER-P"]
    row["all_train_qrel_positive_doc_ids"] = ["OTHER-P"]
    row["positive_count"] = 1
    row["q3_negative_count"] = 1
    return row


def per_query_row(qid: str, doc_ranks: dict[str, int], ndcg: float) -> dict:
    top_k = []
    by_rank = {rank: doc for doc, rank in doc_ranks.items()}
    for rank in range(1, 101):
        doc_id = by_rank.get(rank, f"DOC-{rank}")
        top_k.append({"rank": rank, "doc_id": doc_id, "score": 1 - rank / 1000, "relevance": 1 if doc_id.startswith("MED-30") else 0})
    return {"query_id": qid, "quality": {"ndcg_at_10": ndcg, "recall_at_100": 0.5}, "top_k": top_k}


class BuildQ3R3IncumbentDatasetTest(unittest.TestCase):
    def fixture(self, root: Path, duplicate_source: bool = False) -> dict[str, Path | str | int]:
        parent = root / "parent.jsonl"
        rows = [other_row(), plain73_row()]
        if duplicate_source:
            rows.append(plain73_row())
        write_jsonl(parent, rows)
        parent_manifest = root / "parent.manifest.json"
        parent_coverage = root / "parent.coverage.json"
        write_json(parent_manifest, {"schema": "parent", "legal_gates": LEGAL_GATES})
        write_json(parent_coverage, {"schema": "parent.coverage", "legal_gates": LEGAL_GATES})
        gate = root / "gate.json"
        gate_payload = {
            "schema": "gate",
            "top10_losses": 1,
            "top10_promotions": 4,
            "net_top10_promotions": 3,
            "q3_ndcg_at_10_delta": 0.01,
            "q3_recall_at_100_delta": 0.0,
            "crossing_examples": [
                {"anchor_rank": 11, "candidate_rank": 5, "direction": "promotion_into_top10", "doc_id": "MED-3022", "query_id": "PLAIN-73"},
                {"anchor_rank": 10, "candidate_rank": 11, "direction": "loss_out_of_top10", "doc_id": "MED-3031", "query_id": "PLAIN-73"},
            ],
        }
        write_json(gate, gate_payload)
        proxy_qrels = root / "proxy-qrels.manifest.json"
        write_json(proxy_qrels, {"schema": "proxy-qrels", "source_train_sha256": sha256_file(parent)})
        proxy_summary = root / "proxy-summary.json"
        write_json(proxy_summary, {"hashes": {"nfcorpus_gate": sha256_file(gate), "proxy_qrels_manifest": sha256_file(proxy_qrels)}})
        movement = root / "movement.jsonl"
        write_jsonl(
            movement,
            [
                {
                    "schema": "movement",
                    "dataset": "nfcorpus",
                    "query_id": "PLAIN-73",
                    "anchor_ndcg_at_10": 0.4,
                    "candidate_ndcg_at_10": 0.43,
                    "anchor_recall_at_100": 0.7,
                    "candidate_recall_at_100": 0.7,
                    "ndcg_delta": 0.03,
                    "recall_delta": 0.0,
                    "top10_losses": 1,
                    "top10_promotions": 1,
                }
            ],
        )
        anchor = root / "anchor.per-query.jsonl"
        candidate = root / "candidate.per-query.jsonl"
        write_jsonl(anchor, [per_query_row("PLAIN-73", {"MED-3031": 10, "MED-3022": 11}, 0.4)])
        write_jsonl(candidate, [per_query_row("PLAIN-73", {"MED-3022": 5, "MED-3031": 11}, 0.43)])
        dev = root / "dev.json"
        reserve = root / "reserve.json"
        write_json(dev, {"selected_qids": {"nfcorpus": ["PLAIN-dev"]}})
        write_json(reserve, {"selected_qids": {"nfcorpus": ["PLAIN-reserve"]}})
        test_qrels = root / "test.tsv"
        test_qrels.write_text("query-id\tcorpus-id\tscore\nPLAIN-test\tMED-3031\t1\n", encoding="utf-8")
        return {
            "parent": parent,
            "parent_manifest": parent_manifest,
            "parent_coverage": parent_coverage,
            "gate": gate,
            "proxy_summary": proxy_summary,
            "movement": movement,
            "proxy_qrels": proxy_qrels,
            "anchor": anchor,
            "candidate": candidate,
            "dev": dev,
            "reserve": reserve,
            "test_qrels": test_qrels,
            "out": root / "out.jsonl",
            "manifest": root / "manifest.json",
            "coverage": root / "coverage.json",
            "parent_rows": len(rows),
        }

    def run_builder(self, paths: dict[str, Path | str | int], *extra: str, check: bool = True) -> subprocess.CompletedProcess:
        return subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--parent-train-jsonl",
                str(paths["parent"]),
                "--parent-manifest",
                str(paths["parent_manifest"]),
                "--parent-coverage",
                str(paths["parent_coverage"]),
                "--r2-gate",
                str(paths["gate"]),
                "--r2-proxy-summary",
                str(paths["proxy_summary"]),
                "--r2-movement",
                str(paths["movement"]),
                "--r2-proxy-qrels-manifest",
                str(paths["proxy_qrels"]),
                "--anchor-per-query-jsonl",
                str(paths["anchor"]),
                "--candidate-per-query-jsonl",
                str(paths["candidate"]),
                "--exclude-qids-json",
                str(paths["dev"]),
                "--exclude-qids-json",
                str(paths["reserve"]),
                "--test-qrels",
                str(paths["test_qrels"]),
                "--output-jsonl",
                str(paths["out"]),
                "--manifest",
                str(paths["manifest"]),
                "--coverage-report",
                str(paths["coverage"]),
                "--expected-parent-train-sha256",
                sha256_file(Path(paths["parent"])),
                "--expected-parent-manifest-sha256",
                sha256_file(Path(paths["parent_manifest"])),
                "--expected-parent-coverage-sha256",
                sha256_file(Path(paths["parent_coverage"])),
                "--expected-r2-gate-sha256",
                sha256_file(Path(paths["gate"])),
                "--expected-r2-proxy-qrels-manifest-sha256",
                sha256_file(Path(paths["proxy_qrels"])),
                "--expected-parent-rows",
                str(paths["parent_rows"]),
                "--expected-rows",
                str(int(paths["parent_rows"]) + 1),
                "--expected-steps-per-epoch",
                str((int(paths["parent_rows"]) + 4) // 4),
                *extra,
            ],
            text=True,
            capture_output=True,
            check=check,
        )

    def test_appends_exact_incumbent_row_gains_weights_probs_and_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = self.fixture(Path(tmp))
            self.run_builder(paths)
            rows = read_jsonl(Path(paths["out"]))
            manifest = json.loads(Path(paths["manifest"]).read_text(encoding="utf-8"))
            coverage = json.loads(Path(paths["coverage"]).read_text(encoding="utf-8"))

        appended = rows[-1]
        self.assertEqual(appended["row_id"], "nfcorpus:PLAIN-73:r2-top10-incumbent-preserve")
        self.assertEqual(len(appended["candidate_doc_ids"]), 11)
        self.assertEqual(appended["candidate_doc_ids"], rows[1]["candidate_doc_ids"])
        gains = dict(zip([doc.split(":", 1)[1] for doc in appended["candidate_doc_ids"]], appended["qrel_gains"], strict=True))
        for doc in ["MED-3032", "MED-3023", "MED-3020", "MED-3019", "MED-3031"]:
            self.assertEqual(gains[doc], 2.0)
        for doc in ["MED-3022", "MED-3026"]:
            self.assertEqual(gains[doc], 1.0)
        self.assertEqual(appended["positive_indexes"], list(range(7)))
        self.assertFalse(any(appended["hard_negative_eligible"][:7]))
        self.assertTrue(all(appended["hard_negative_eligible"][7:]))
        self.assertAlmostEqual(sum(appended["target_probabilities"]), 1.0)
        self.assertEqual(appended["turboquant_topk_loss_weight"], 0.5)
        self.assertEqual(appended["turboquant_topk_recall_loss_weight"], 0.0)
        self.assertEqual(appended["hard_loss_weight"], 0.0)
        self.assertEqual(appended["soft_loss_weight"], 0.0)
        self.assertEqual(appended["recovery_loss_weight"], 0.0)
        self.assertEqual(manifest["counts"]["rows"], 3)
        self.assertEqual(manifest["counts"]["canonical_counts"]["canonical_candidates"], 24)
        self.assertEqual(manifest["counts"]["workload"]["raw_candidates_per_epoch"], 24)
        self.assertEqual(manifest["counts"]["workload"]["canonical_candidates_per_epoch"], 24)
        self.assertEqual(manifest["counts"]["workload"]["main_q3_pairs_per_epoch"], 67)
        self.assertEqual(manifest["counts"]["workload"]["recall_q3_pairs_per_epoch"], 0)
        self.assertEqual(manifest["counts"]["workload"]["total_train_pairs_per_epoch"], 91)
        self.assertEqual(manifest["go_loader_plan"]["dataset_form"], "text_score_spectrum_jsonl")
        self.assertEqual(manifest["go_loader_plan"]["loader_path"], "runtime.OpenEmbeddingTextScoreSpectrumSource")
        self.assertTrue(manifest["go_loader_plan"]["plan_only"])
        self.assertFalse(manifest["go_loader_plan"]["go_runtime_edits_required"])
        self.assertIn("--score-spectrum-train", manifest["go_loader_plan"]["required_cli_flags"])
        self.assertIn("--allow-research-only-score-spectrum", manifest["go_loader_plan"]["required_cli_flags"])
        self.assertIn("--no-tokenizer", manifest["go_loader_plan"]["forbidden_cli_flags"])
        self.assertEqual(manifest["go_loader_plan"]["workload_expectations"]["planned_pairs"], 182)
        self.assertEqual(coverage["go_loader_plan"], manifest["go_loader_plan"])
        self.assertEqual(coverage["selection_audit"]["evidence"]["rank_checks"]["MED-3031"], {"anchor_rank": 10, "candidate_rank": 11})

    def test_deterministic_replay(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = self.fixture(Path(tmp))
            self.run_builder(paths)
            first = Path(paths["out"]).read_text(encoding="utf-8")
            first_manifest_counts = json.loads(Path(paths["manifest"]).read_text(encoding="utf-8"))["counts"]
            self.run_builder(paths)
            second = Path(paths["out"]).read_text(encoding="utf-8")
            second_manifest_counts = json.loads(Path(paths["manifest"]).read_text(encoding="utf-8"))["counts"]

        self.assertEqual(first, second)
        self.assertEqual(first_manifest_counts, second_manifest_counts)

    def test_wrong_parent_hash_rejects(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = self.fixture(Path(tmp))
            completed = self.run_builder(paths, "--expected-parent-train-sha256", "0" * 64, check=False)

        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("parent train sha mismatch", completed.stderr)

    def test_stale_evidence_hash_rejects(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = self.fixture(Path(tmp))
            completed = self.run_builder(paths, "--expected-r2-gate-sha256", "1" * 64, check=False)

        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("r2 gate sha mismatch", completed.stderr)

    def test_missing_and_duplicate_source_row_reject(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = self.fixture(Path(tmp), duplicate_source=True)
            completed = self.run_builder(paths, check=False)
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("expected exactly one source row", completed.stderr)

        with tempfile.TemporaryDirectory() as tmp:
            paths = self.fixture(Path(tmp))
            rows = [other_row()]
            write_jsonl(Path(paths["parent"]), rows)
            write_json(Path(paths["proxy_qrels"]), {"schema": "proxy-qrels", "source_train_sha256": sha256_file(Path(paths["parent"]))})
            write_json(Path(paths["proxy_summary"]), {"hashes": {"nfcorpus_gate": sha256_file(Path(paths["gate"])), "proxy_qrels_manifest": sha256_file(Path(paths["proxy_qrels"]))}})
            paths["parent_rows"] = 1
            completed = self.run_builder(paths, check=False)
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("expected exactly one source row", completed.stderr)

    def test_unexpected_rank_and_qrel_evidence_reject(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = self.fixture(Path(tmp))
            write_jsonl(Path(paths["candidate"]), [per_query_row("PLAIN-73", {"MED-3022": 6, "MED-3031": 11}, 0.43)])
            completed = self.run_builder(paths, check=False)
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("unexpected MED-3022 ranks", completed.stderr)

        with tempfile.TemporaryDirectory() as tmp:
            paths = self.fixture(Path(tmp))
            gate = json.loads(Path(paths["gate"]).read_text(encoding="utf-8"))
            gate["crossing_examples"] = []
            write_json(Path(paths["gate"]), gate)
            write_json(Path(paths["proxy_summary"]), {"hashes": {"nfcorpus_gate": sha256_file(Path(paths["gate"])), "proxy_qrels_manifest": sha256_file(Path(paths["proxy_qrels"]))}})
            completed = self.run_builder(paths, check=False)
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("does not contain required", completed.stderr)

    def test_duplicate_doc_and_duplicate_text_conflict_reject(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = self.fixture(Path(tmp))
            rows = read_jsonl(Path(paths["parent"]))
            rows[1]["candidate_doc_ids"][1] = rows[1]["candidate_doc_ids"][0]
            write_jsonl(Path(paths["parent"]), rows)
            write_json(Path(paths["proxy_qrels"]), {"schema": "proxy-qrels", "source_train_sha256": sha256_file(Path(paths["parent"]))})
            write_json(Path(paths["proxy_summary"]), {"hashes": {"nfcorpus_gate": sha256_file(Path(paths["gate"])), "proxy_qrels_manifest": sha256_file(Path(paths["proxy_qrels"]))}})
            completed = self.run_builder(paths, check=False)
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("duplicate candidate doc id", completed.stderr)

        with tempfile.TemporaryDirectory() as tmp:
            paths = self.fixture(Path(tmp))
            rows = read_jsonl(Path(paths["parent"]))
            rows[1]["candidate_texts"][0] = "same text"
            rows[1]["candidate_texts"][7] = " Same   Text "
            write_jsonl(Path(paths["parent"]), rows)
            write_json(Path(paths["proxy_qrels"]), {"schema": "proxy-qrels", "source_train_sha256": sha256_file(Path(paths["parent"]))})
            write_json(Path(paths["proxy_summary"]), {"hashes": {"nfcorpus_gate": sha256_file(Path(paths["gate"])), "proxy_qrels_manifest": sha256_file(Path(paths["proxy_qrels"]))}})
            completed = self.run_builder(paths, check=False)
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("duplicate-text positive/hard-negative conflict", completed.stderr)

    def test_held_out_overlap_rejects(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = self.fixture(Path(tmp))
            write_json(Path(paths["dev"]), {"selected_qids": {"nfcorpus": ["PLAIN-73"]}})
            completed = self.run_builder(paths, check=False)
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("held-out overlap", completed.stderr)

        with tempfile.TemporaryDirectory() as tmp:
            paths = self.fixture(Path(tmp))
            Path(paths["test_qrels"]).write_text("query-id\tcorpus-id\tscore\nPLAIN-73\tMED-3031\t1\n", encoding="utf-8")
            completed = self.run_builder(paths, check=False)
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("held-out overlap", completed.stderr)


if __name__ == "__main__":
    unittest.main()
