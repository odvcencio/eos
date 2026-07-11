#!/usr/bin/env python3
"""Focused tests for build_release_vector_distill_data.py."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import build_release_vector_distill_data as adapter  # noqa: E402


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def write_qrels(path: Path, rows: list[tuple[str, str, int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("query-id\tcorpus-id\tscore\n")
        for query_id, document_id, score in rows:
            handle.write(f"{query_id}\t{document_id}\t{score}\n")


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


class ReleaseVectorDistillAdapterTest(unittest.TestCase):
    def make_dataset(
        self,
        root: Path,
        dataset: str = "tiny",
        train: list[tuple[str, str, int]] | None = None,
        dev: list[tuple[str, str, int]] | None = None,
        test: list[tuple[str, str, int]] | None = None,
    ) -> Path:
        dataset_dir = root / "raw" / dataset / dataset
        write_jsonl(
            dataset_dir / "queries.jsonl",
            [
                {"_id": "q1", "text": "query one"},
                {"_id": "q2", "text": "query two"},
                {"_id": "q3", "text": "query three"},
            ],
        )
        write_jsonl(
            dataset_dir / "corpus.jsonl",
            [
                {"_id": "d1", "title": "Doc One", "text": "body one"},
                {"_id": "d2", "title": "", "text": "body two"},
                {"_id": "d3", "title": "", "text": "body three"},
            ],
        )
        write_qrels(dataset_dir / "qrels" / "train.tsv", train or [("q1", "d1", 1), ("q2", "d2", 1)])
        if dev is not None:
            write_qrels(dataset_dir / "qrels" / "dev.tsv", dev)
        if test is not None:
            write_qrels(dataset_dir / "qrels" / "test.tsv", test)
        return dataset_dir

    def write_license(self, root: Path, dataset: str = "tiny", raw_sha256: str | None = None, **flags: bool) -> Path:
        raw_sha256 = raw_sha256 or adapter.sha256_directory(root / "raw" / dataset)
        entry = {
            "dataset": dataset,
            "license": "Test License",
            "source_url": "https://example.invalid/tiny",
            "raw_sha256": raw_sha256,
            "train_allowed": flags.get("train_allowed", True),
            "release_train_allowed": flags.get("release_train_allowed", True),
            "commercial_use_allowed": flags.get("commercial_use_allowed", True),
        }
        path = root / "licenses.json"
        path.write_text(json.dumps({"datasets": [entry]}, sort_keys=True), encoding="utf-8")
        return path

    def run_adapter(self, root: Path, license_path: Path, dataset: str = "tiny") -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                sys.executable,
                str(SCRIPT_DIR / "build_release_vector_distill_data.py"),
                "--raw-root",
                str(root / "raw"),
                "--datasets",
                dataset,
                "--license-manifest",
                str(license_path),
                "--output-dir",
                str(root / "out"),
            ],
            text=True,
            capture_output=True,
            check=False,
        )

    def test_positive_success_writes_canonical_inputs_and_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.make_dataset(root)
            license_path = self.write_license(root)

            result = self.run_adapter(root, license_path)

            self.assertEqual(result.returncode, 0, result.stderr)
            queries = read_jsonl(root / "out" / "queries.jsonl")
            documents = read_jsonl(root / "out" / "documents.jsonl")
            relations = read_jsonl(root / "out" / "relations.jsonl")
            manifest = json.loads((root / "out" / "manifest.json").read_text(encoding="utf-8"))

        self.assertEqual([row["id"] for row in queries], ["tiny:q1", "tiny:q2"])
        self.assertEqual([row["source_document_id"] for row in documents], ["d1", "d2"])
        self.assertEqual(documents[0]["text"], "Doc One\n\nbody one")
        self.assertEqual([(row["query_id"], row["positive_doc_id"]) for row in relations], [("tiny:q1", "tiny:d1"), ("tiny:q2", "tiny:d2")])
        self.assertTrue(manifest["release_allowed"])
        self.assertFalse(manifest["teacher_embeddings_release_cleared"])
        self.assertEqual(manifest["legal_gates"], {"train_allowed": True, "release_train_allowed": True, "commercial_use_allowed": True})
        self.assertEqual(manifest["outputs"]["relations"]["rows"], 2)
        self.assertEqual(manifest["datasets"][0]["dropped"]["heldout_document"], 0)

    def test_split_leakage_filters_heldout_positive_document(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.make_dataset(root, train=[("q1", "d1", 1), ("q2", "d2", 1)], test=[("q3", "d2", 1)])
            license_path = self.write_license(root)

            result = self.run_adapter(root, license_path)

            self.assertEqual(result.returncode, 0, result.stderr)
            relations = read_jsonl(root / "out" / "relations.jsonl")
            manifest = json.loads((root / "out" / "manifest.json").read_text(encoding="utf-8"))

        self.assertEqual([(row["source_query_id"], row["source_document_id"]) for row in relations], [("q1", "d1")])
        self.assertEqual(manifest["datasets"][0]["dropped"]["heldout_document"], 1)

    def test_hash_mismatch_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.make_dataset(root)
            license_path = self.write_license(root, raw_sha256="0" * 64)

            result = self.run_adapter(root, license_path)

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("raw_sha256 mismatch", result.stderr)

    def test_missing_license_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.make_dataset(root)
            license_path = root / "licenses.json"
            license_path.write_text(json.dumps({"datasets": []}), encoding="utf-8")

            result = self.run_adapter(root, license_path)

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("license manifest missing dataset entries", result.stderr)

    def test_false_legal_flag_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.make_dataset(root)
            license_path = self.write_license(root, commercial_use_allowed=False)

            result = self.run_adapter(root, license_path)

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("does not grant commercial_use_allowed=true", result.stderr)


if __name__ == "__main__":
    unittest.main()
