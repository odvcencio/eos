#!/usr/bin/env python3
"""Tests for q3 promotion+guard qid selection."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPT = SCRIPT_DIR / "select_q3_promotion_guard_qids.py"


def load_module():
    spec = importlib.util.spec_from_file_location("select_q3_promotion_guard_qids", SCRIPT)
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


def per_query_row(query_id: str, relevant_ranks: list[int], relevant_count: int = 3, ndcg: float = 0.5) -> dict:
    relevant_by_rank = set(relevant_ranks)
    return {
        "schema": "manta.embedding_turboquant_retrieval_per_query.v1",
        "dataset": "scifact",
        "query_id": query_id,
        "method": "turboquant_ip_b3",
        "bits": 3,
        "scoring_surface": "turboquant_ip_prepared",
        "quantizer_seed": 5581486560434873699,
        "relevant_count": relevant_count,
        "quality": {"ndcg_at_10": ndcg, "recall_at_100": len(relevant_ranks) / relevant_count},
        "top_k": [
            {
                "rank": rank,
                "doc_id": f"d{query_id}-{rank}",
                "score": 1.0 / rank,
                "relevance": 1 if rank in relevant_by_rank else 0,
            }
            for rank in range(1, 101)
        ],
    }


class PromotionGuardSelectorTest(unittest.TestCase):
    def test_select_for_dataset_prefers_promotion_then_guard(self) -> None:
        mod = load_module()
        records = [
            mod.row_features(per_query_row("promo11", [11], ndcg=0.25)),
            mod.row_features(per_query_row("promo19", [19], ndcg=0.10)),
            mod.row_features(per_query_row("guard-risk", [2, 90], relevant_count=5, ndcg=0.40)),
            mod.row_features(per_query_row("guard-safe", [1], relevant_count=1, ndcg=0.60)),
            mod.row_features(per_query_row("fallback", [30], relevant_count=1, ndcg=0.30)),
        ]
        selected = mod.select_for_dataset(records, rows_per_dataset=4, promotion_quota=2)

        self.assertEqual([row["query_id"] for row in selected], ["promo11", "promo19", "guard-risk", "guard-safe"])
        self.assertEqual([row["selected_bucket"] for row in selected], [
            "promotion_11_20",
            "promotion_11_20",
            "guard_recall_quality",
            "guard_recall_quality",
        ])

    def test_main_excludes_frozen_and_official_test_before_selection(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            per_query = root / "anchor.scifact.per-query.jsonl"
            write_jsonl(
                per_query,
                [
                    per_query_row("frozen", [11], ndcg=0.01),
                    per_query_row("test-qid", [11], ndcg=0.02),
                    per_query_row("promo", [11], ndcg=0.03),
                    per_query_row("guard", [2, 90], relevant_count=4, ndcg=0.40),
                ],
            )
            train_qrels = root / "raw" / "scifact" / "scifact" / "qrels" / "train.tsv"
            train_qrels.parent.mkdir(parents=True)
            train_qrels.write_text(
                "query-id\tcorpus-id\tscore\n"
                "frozen\td1\t1\n"
                "test-qid\td2\t1\n"
                "promo\td3\t1\n"
                "guard\td4\t1\n",
                encoding="utf-8",
            )
            test_qrels = root / "raw" / "scifact" / "scifact" / "qrels" / "test.tsv"
            test_qrels.write_text("query-id\tcorpus-id\tscore\ntest-qid\td2\t1\n", encoding="utf-8")
            excluded = root / "excluded.json"
            excluded.write_text(json.dumps({"selected_qids": {"scifact": ["frozen"]}}), encoding="utf-8")
            selected = root / "selected.json"
            records = root / "records.json"
            manifest = root / "manifest.json"

            completed = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--per-query-jsonl",
                    str(per_query),
                    "--train-qrels",
                    str(train_qrels),
                    "--test-qrels",
                    str(test_qrels),
                    "--exclude-qids-json",
                    str(excluded),
                    "--rows-per-dataset",
                    "2",
                    "--promotion-quota",
                    "1",
                    "--output-selected-qids-json",
                    str(selected),
                    "--output-records-json",
                    str(records),
                    "--manifest",
                    str(manifest),
                ],
                check=True,
                text=True,
                capture_output=True,
            )
            selected_payload = json.loads(selected.read_text(encoding="utf-8"))
            manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))

        self.assertIn("scifact=2", completed.stdout)
        self.assertEqual(selected_payload["selected_qids"]["scifact"], ["promo", "guard"])
        self.assertEqual(manifest_payload["datasets"]["scifact"]["official_test_overlap_count"], 0)
        self.assertEqual(manifest_payload["datasets"]["scifact"]["frozen_overlap_count"], 0)
        self.assertEqual(manifest_payload["datasets"]["scifact"]["excluded_counts"]["official_test_overlap"], 1)
        self.assertEqual(manifest_payload["datasets"]["scifact"]["excluded_counts"]["frozen_dev_reserve"], 1)
        self.assertEqual(manifest_payload["legal_gates"]["commercial_use_allowed"], False)
        self.assertIn("selected_qids_json", manifest_payload["sha256"])


if __name__ == "__main__":
    unittest.main()
