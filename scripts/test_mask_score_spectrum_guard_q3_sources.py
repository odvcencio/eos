#!/usr/bin/env python3
"""Tests for score-spectrum guard q3 source masking."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPT = SCRIPT_DIR / "mask_score_spectrum_guard_q3_sources.py"


def load_module():
    spec = importlib.util.spec_from_file_location("mask_score_spectrum_guard_q3_sources", SCRIPT)
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


class GuardQ3SourceMaskTest(unittest.TestCase):
    def test_transform_masks_only_non_promotion_q3_sources(self) -> None:
        mod = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows_path = root / "rows.jsonl"
            write_jsonl(
                rows_path,
                [
                    {
                        "row_id": "scifact:p",
                        "source_dataset": "scifact",
                        "source_query_id": "p",
                        "candidate_sources": ["qrel", "q3", "bm25"],
                        "hard_negative_eligible": [False, True, True],
                    },
                    {
                        "row_id": "scifact:g",
                        "source_dataset": "scifact",
                        "source_query_id": "g",
                        "candidate_sources": ["qrel", "q3", "q3"],
                        "hard_negative_eligible": [False, True, True],
                    },
                ],
            )
            buckets = {
                ("scifact", "p"): "promotion_11_20",
                ("scifact", "g"): "guard_recall_quality",
            }
            rows, counts, by_dataset = mod.transform_rows(rows_path, buckets, "other")

        self.assertEqual(rows[0]["candidate_sources"], ["qrel", "q3", "bm25"])
        self.assertEqual(rows[1]["candidate_sources"], ["qrel", "other", "other"])
        self.assertEqual(rows[1]["hard_negative_eligible"], [False, True, True])
        self.assertEqual(counts["q3_sources_masked"], 2)
        self.assertEqual(by_dataset["scifact"]["promotion_rows_q3_sources_preserved"], 1)
        self.assertEqual(by_dataset["scifact"]["rows_with_q3_sources_masked"], 1)

    def test_cli_writes_manifest_and_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows_path = root / "rows.jsonl"
            records_path = root / "records.json"
            out_path = root / "out.jsonl"
            manifest_path = root / "manifest.json"
            write_jsonl(
                rows_path,
                [
                    {
                        "row_id": "nfcorpus:guard",
                        "source_dataset": "nfcorpus",
                        "source_query_id": "guard",
                        "candidate_sources": ["qrel", "q3"],
                        "hard_negative_eligible": [False, True],
                        "legal_gates": {
                            "train_allowed_for_research": True,
                            "release_train_allowed": False,
                            "commercial_use_allowed": False,
                        },
                    }
                ],
            )
            records_path.write_text(
                json.dumps({"records": {"nfcorpus": [{"query_id": "guard", "selected_bucket": "guard_recall_quality"}]}}),
                encoding="utf-8",
            )
            completed = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--input-jsonl",
                    str(rows_path),
                    "--selector-records-json",
                    str(records_path),
                    "--output-jsonl",
                    str(out_path),
                    "--manifest",
                    str(manifest_path),
                ],
                check=True,
                text=True,
                capture_output=True,
            )
            rows = [json.loads(line) for line in out_path.read_text(encoding="utf-8").splitlines()]
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

        self.assertIn("q3_sources_masked=1", completed.stdout)
        self.assertEqual(rows[0]["candidate_sources"], ["qrel", "other"])
        self.assertEqual(manifest["counts"]["q3_sources_masked"], 1)
        self.assertEqual(manifest["counts"]["candidate_source_counts"]["other"], 1)
        self.assertEqual(manifest["legal_gates"]["release_train_allowed"], False)
        self.assertIn("output_jsonl", manifest["sha256"])


if __name__ == "__main__":
    unittest.main()
