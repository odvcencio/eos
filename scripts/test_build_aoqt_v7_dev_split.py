"""Tests for the AOQT V7 dev split harness."""

from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import build_aoqt_stage2_calibration as stage2  # noqa: E402
import build_aoqt_v7_dev_split as splitter  # noqa: E402


def fake_sha(label: str) -> str:
    return stage2.sha256_json({"fixture": label})


def write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def official_registry(qids_by_dataset: dict[str, list[str]], source_by_file: dict[str, str]) -> dict:
    return {
        "schema": splitter.heldout_gate.TRUSTED_WORKLOAD_REGISTRY_SCHEMA,
        "status": "active",
        "domains": {
            dataset: {
                "qrels_path_suffix": f"datasets/manta-embed-v1/raw/{dataset}/{dataset}/qrels/test.tsv",
                "qrels_sha256": source_by_file[f"datasets/manta-embed-v1/raw/{dataset}/{dataset}/qrels/test.tsv"],
                "qid_set_sha256": splitter.sha256_json(qids_by_dataset[dataset]),
                "query_count": len(qids_by_dataset[dataset]),
            }
            for dataset in stage2.ALLOWED_DATASETS
        },
    }


class AOQTV7DevSplitTests(unittest.TestCase):
    def make_fixture(self, root: Path) -> tuple[Path, Path, dict]:
        source_sha_by_dataset = {dataset: fake_sha(f"{dataset}-source") for dataset in stage2.ALLOWED_DATASETS}
        rows = []
        for dataset in stage2.ALLOWED_DATASETS:
            for qindex in range(1, 5):
                qid = f"{dataset}-train-{qindex}"
                for bits in (3, 5):
                    row_id = f"{dataset}.q{bits}.top10.{qid}"
                    rows.append(
                        {
                            "row_id": row_id,
                            "dataset": dataset,
                            "bits": bits,
                            "bucket": "top10_guard",
                            "qid": qid,
                            "rank_window": [1, 10],
                            "candidate_doc_ids": [f"{qid}-pos", f"{qid}-neg"],
                            "positive_doc_ids": [f"{qid}-pos"],
                            "negative_doc_ids": [f"{qid}-neg"],
                            "pair_ids": [
                                {
                                    "pair_id": f"{row_id}:{qid}-pos:{qid}-neg",
                                    "positive_doc_id": f"{qid}-pos",
                                    "negative_doc_id": f"{qid}-neg",
                                }
                            ],
                        }
                    )
                if dataset == "nfcorpus":
                    row_id = f"{dataset}.q3.nf80_120.{qid}"
                    rows.append(
                        {
                            "row_id": row_id,
                            "dataset": dataset,
                            "bits": 3,
                            "bucket": "nf_boundary80_120_guard",
                            "qid": qid,
                            "rank_window": [80, 120],
                            "candidate_doc_ids": [f"{qid}-boundary-pos", f"{qid}-boundary-neg"],
                            "positive_doc_ids": [f"{qid}-boundary-pos"],
                            "negative_doc_ids": [f"{qid}-boundary-neg"],
                            "pair_ids": [
                                {
                                    "pair_id": f"{row_id}:{qid}-boundary-pos:{qid}-boundary-neg",
                                    "positive_doc_id": f"{qid}-boundary-pos",
                                    "negative_doc_id": f"{qid}-boundary-neg",
                                }
                            ],
                        }
                    )
        rows = sorted(rows, key=lambda row: row["row_id"])
        official_qids = {dataset: [f"{dataset}-official-1"] for dataset in stage2.ALLOWED_DATASETS}
        official_path = root / "official-test.json"
        source_by_file = {
            f"datasets/manta-embed-v1/raw/{dataset}/{dataset}/qrels/test.tsv": fake_sha(f"{dataset}-official-qrels")
            for dataset in stage2.ALLOWED_DATASETS
        }
        official_payload = {
            "schema": stage2.EXCLUSION_SCHEMA,
            "name": "official-test",
            "qids_by_dataset": official_qids,
            "source_sha256": splitter.sha256_json(dict(sorted(source_by_file.items()))),
            "source_sha256_by_file": source_by_file,
        }
        write_json(official_path, official_payload)
        plan = {
            "schema": stage2.SCHEMA,
            "mode": "plan_only",
            "actual_cache_generation_ran": False,
            "actual_training_ran": False,
            "actual_eval_ran": False,
            "selection_policy": {"split": "train"},
            "exclusions": {
                "sets": [
                    {
                        "name": "official-test",
                        "path": str(official_path),
                        "manifest_sha256": splitter.sha256_file(official_path),
                        "qids_by_dataset_sha256": splitter.sha256_json(official_qids),
                        "qid_count_by_dataset": {dataset: 1 for dataset in stage2.ALLOWED_DATASETS},
                    }
                ]
            },
            "provenance": {
                "plan_sha256": fake_sha("plan"),
                "dataset_source_provenance_by_dataset": {
                    dataset: {"sha256": source_sha_by_dataset[dataset], "components": {"q3_score": source_sha_by_dataset[dataset]}}
                    for dataset in stage2.ALLOWED_DATASETS
                },
            },
            "workload": {"row_count": len(rows), "pair_count": len(rows)},
            "row_ids_sha256": splitter.sha256_json([row["row_id"] for row in rows]),
            "rows_sha256": splitter.sha256_json(rows),
            "rows": rows,
        }
        plan_path = write_json(root / "plan.json", plan)
        return plan_path, official_path, official_registry(official_qids, source_by_file)

    def build(self, plan: Path, official: Path, registry: dict, output: Path) -> dict:
        with mock.patch.object(splitter.heldout_gate, "PRODUCTION_TRUSTED_WORKLOAD_REGISTRY", copy.deepcopy(registry)):
            manifest = splitter.build_manifest(plan, official, 3, "fixture-seed", 0.0)
            write_json(output, manifest)
            return manifest

    def test_build_is_deterministic_grouped_and_leak_free(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            plan, official, registry = self.make_fixture(root)
            first = self.build(plan, official, registry, root / "split.json")
            second = self.build(plan, official, registry, root / "split2.json")
        self.assertEqual(first, second)
        self.assertEqual(first["schema"], splitter.SCHEMA)
        self.assertTrue(first["leakage_proof"]["passed"])
        self.assertEqual(first["leakage_proof"]["official_overlap_count"], 0)
        self.assertEqual(first["population"]["group_count"], 12)
        self.assertEqual(first["population"]["row_count"], 28)
        for fold in first["folds"]:
            train = set(fold["train"]["group_ids"])
            dev = set(fold["dev"]["group_ids"])
            self.assertFalse(train & dev)
            for group in first["groups"]:
                if group["group_id"] in dev:
                    self.assertTrue(set(group["row_ids"]).issubset(set(fold["dev"]["row_ids"])))
                if group["group_id"] in train:
                    self.assertTrue(set(group["row_ids"]).issubset(set(fold["train"]["row_ids"])))

    def test_validate_rebuilds_manifest_and_rejects_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            plan, official, registry = self.make_fixture(root)
            output = root / "split.json"
            manifest = self.build(plan, official, registry, output)
            with mock.patch.object(splitter.heldout_gate, "PRODUCTION_TRUSTED_WORKLOAD_REGISTRY", copy.deepcopy(registry)):
                result = splitter.validate_manifest(output, plan, official)
            self.assertTrue(result["passed"])
            manifest["folds"][0]["dev"]["group_ids"].append("fiqa:forged")
            write_json(output, manifest)
            with mock.patch.object(splitter.heldout_gate, "PRODUCTION_TRUSTED_WORKLOAD_REGISTRY", copy.deepcopy(registry)):
                with self.assertRaisesRegex(splitter.SplitError, "manifest_sha256 mismatch"):
                    splitter.validate_manifest(output, plan, official)

    def test_official_overlap_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            plan, official, registry = self.make_fixture(root)
            payload = json.loads(plan.read_text(encoding="utf-8"))
            payload["rows"][0]["qid"] = "fiqa-official-1"
            payload["rows"][0]["row_id"] = "fiqa.q3.top10.fiqa-official-1"
            payload["rows"][0]["pair_ids"][0]["pair_id"] = "fiqa.q3.top10.fiqa-official-1:p:n"
            write_json(plan, payload)
            with mock.patch.object(splitter.heldout_gate, "PRODUCTION_TRUSTED_WORKLOAD_REGISTRY", copy.deepcopy(registry)):
                with self.assertRaisesRegex(splitter.SplitError, "official heldout qids"):
                    splitter.build_manifest(plan, official, 3, "fixture-seed", 0.0)

    def test_rejects_unpinned_official_registry(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            plan, official, registry = self.make_fixture(root)
            payload = json.loads(official.read_text(encoding="utf-8"))
            payload["source_sha256_by_file"]["datasets/manta-embed-v1/raw/fiqa/fiqa/qrels/test.tsv"] = fake_sha("mutated")
            write_json(official, payload)
            with mock.patch.object(splitter.heldout_gate, "PRODUCTION_TRUSTED_WORKLOAD_REGISTRY", copy.deepcopy(registry)):
                with self.assertRaisesRegex(splitter.SplitError, "source qrels hash registry mismatch|source_sha256 rollup mismatch"):
                    splitter.build_manifest(plan, official, 3, "fixture-seed", 0.0)


if __name__ == "__main__":
    unittest.main()
