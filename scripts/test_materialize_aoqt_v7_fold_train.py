"""Tests for the AOQT V7 fold-train materializer."""

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
import materialize_aoqt_v7_fold_train as materializer  # noqa: E402


def fake_sha(label: str) -> str:
    return stage2.sha256_json({"fixture": label})


def write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows), encoding="utf-8")
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


class AOQTV7FoldTrainMaterializerTests(unittest.TestCase):
    def make_fixture(self, root: Path) -> dict[str, Path | dict]:
        plan_rows = []
        materialized_rows = []
        source_sha_by_dataset = {dataset: fake_sha(f"{dataset}-source") for dataset in stage2.ALLOWED_DATASETS}
        qrels_by_dataset = {dataset: fake_sha(f"{dataset}-qrels") for dataset in stage2.ALLOWED_DATASETS}
        for dataset in stage2.ALLOWED_DATASETS:
            for qindex in range(1, 5):
                qid = f"{dataset}-train-{qindex}"
                for bits in (3, 5):
                    row_id = f"{dataset}.q{bits}.top10.{qid}"
                    plan_rows.append(
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
                            "pair_ids": [{"pair_id": f"{row_id}:p:n", "positive_doc_id": "p", "negative_doc_id": "n"}],
                        }
                    )
                    materialized_rows.append(
                        {
                            "schema": "eos.q3_aoqt_sidecar_row.v1",
                            "row_id": row_id,
                            "dataset": dataset,
                            "query_id": qid,
                            "query_vector_id": f"{dataset}:query:{qid}",
                            "query_vector_sha256": fake_sha(f"{row_id}:qv"),
                            "candidate_doc_ids": [f"{qid}-pos", f"{qid}-neg"],
                            "candidate_vector_ids": [f"{dataset}:doc:p", f"{dataset}:doc:n"],
                            "candidate_vector_sha256": [fake_sha(f"{row_id}:p"), fake_sha(f"{row_id}:n")],
                            "qrel_gains": [1, 0],
                            "candidate_sources": ["q3_top10", "q3_top10"],
                            "eligible_pair_mask": [[False, True], [True, False]],
                            "guard_class": "top10_guard",
                            "anchor_scores": {"dense": [0.8, 0.7], "q3": [0.8, 0.7], "q5": [0.8, 0.7]},
                            "anchor_ranks": {"dense": [1, 2], "q3": [1, 2], "q5": [1, 2]},
                            "weights": {
                                "q3_gain": 1,
                                "q3_order_guard": 0.5,
                                "q3_score_distill": 0.25,
                                "q5_order_guard": 0.5,
                                "q5_score_distill": 0.25,
                                "nf_boundary_guard": 0,
                            },
                            "source_artifact_hash": fake_sha(f"{row_id}:source"),
                            "qrels_sha256": qrels_by_dataset[dataset],
                            "split_proof": {
                                "split": "train",
                                "train_only": True,
                                "proof_sha256": fake_sha("split-proof"),
                                "exclusion_identities": [
                                    "dev",
                                    "dev4",
                                    "reserve",
                                    "reserve4",
                                    "test",
                                    "official",
                                    "official-test",
                                    "proxy",
                                ],
                            },
                            "compatibility_digest": fake_sha("compat"),
                            "legal_gates": {
                                "research_train_allowed": True,
                                "release_train_allowed": False,
                                "commercial_use_allowed": False,
                                "free_open_release_allowed": False,
                            },
                        }
                    )
        plan_rows = sorted(plan_rows, key=lambda row: row["row_id"])
        materialized_rows = sorted(materialized_rows, key=lambda row: row["row_id"])

        official_qids = {dataset: [f"{dataset}-official-1"] for dataset in stage2.ALLOWED_DATASETS}
        source_by_file = {
            f"datasets/manta-embed-v1/raw/{dataset}/{dataset}/qrels/test.tsv": fake_sha(f"{dataset}-official-qrels")
            for dataset in stage2.ALLOWED_DATASETS
        }
        official_path = root / "official-test.json"
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
            "workload": {"row_count": len(plan_rows), "pair_count": len(plan_rows)},
            "row_ids_sha256": splitter.sha256_json([row["row_id"] for row in plan_rows]),
            "rows_sha256": splitter.sha256_json(plan_rows),
            "rows": plan_rows,
        }
        plan_path = write_json(root / "plan.json", plan)
        registry = official_registry(official_qids, source_by_file)
        with mock.patch.object(splitter.heldout_gate, "PRODUCTION_TRUSTED_WORKLOAD_REGISTRY", copy.deepcopy(registry)):
            split = splitter.build_manifest(plan_path, official_path, 2, "fixture-seed", 0.0)
            split_path = write_json(root / "split.json", split)
            validation = splitter.validate_manifest(split_path, plan_path, official_path)
        validation_path = write_json(root / "split.validation.json", validation)

        row_ids = [row["row_id"] for row in materialized_rows]
        source_manifest = {
            "schema": "eos.q3_aoqt_sidecar_manifest.v1",
            "created_at_utc": "2026-09-05T00:00:00Z",
            "anchor_artifact_path": "anchor.mll",
            "anchor_artifact_sha256": fake_sha("anchor"),
            "anchor_package_manifest_sha256": fake_sha("anchor-manifest"),
            "anchor_embedding_space_id": fake_sha("space"),
            "dim": 384,
            "topology": {"kind": "aoqt_givens_v1", "dim": 384, "stages": 8, "pairs_per_stage": 192, "angle_count": 1536, "seed": 191, "pairings_sha256": fake_sha("pairs")},
            "turboquant_seed": 5581486560434873699,
            "quant_surfaces": [
                {"bit_width": 3, "seed": 5581486560434873699, "score_surface": "turboquant.ip.prepared_v1", "prepared_query": True},
                {"bit_width": 5, "seed": 5581486560434873699, "score_surface": "turboquant.ip.prepared_v1", "prepared_query": True},
            ],
            "objective_contract": {
                "dim": 384,
                "turboquant_seed": 5581486560434873699,
                "gain_bit": 3,
                "q3_guard_bit": 3,
                "q5_guard_bit": 5,
                "score_surface": "turboquant.ip.prepared_v1",
                "gain_cutoff": 10,
                "gain_tau": 0.05,
                "gain_margin": 0,
                "guard_tau": 0.05,
                "guard_margin": 0,
                "score_distill_tau": 1,
                "nf_boundary_source": "nf_boundary80_120",
                "weight_sums": {
                    "q3_gain": len(row_ids),
                    "q3_order_guard": len(row_ids) * 0.5,
                    "q3_score_distill": len(row_ids) * 0.25,
                    "q5_order_guard": len(row_ids) * 0.5,
                    "q5_score_distill": len(row_ids) * 0.25,
                    "nf_boundary_guard": 0,
                },
            },
            "qrels_sha256_by_dataset": qrels_by_dataset,
            "split_proof": materialized_rows[0]["split_proof"],
            "source_artifact_hashes": sorted({row["source_artifact_hash"] for row in materialized_rows}),
            "vector_cache_hashes": [fake_sha("query-cache"), fake_sha("doc-cache")],
            "row_count": len(row_ids),
            "row_id_sha256": materializer.aoqt_row_id_sha256(row_ids),
            "compatibility_digest": fake_sha("compat"),
            "legal_gates": materialized_rows[0]["legal_gates"],
            "training_contract": "aoqt-trainonly-score-distill-joint-budget-v1",
            "candidate_eligibility_policy": {
                "dense_max_abs_delta_tolerance": 0.0005,
                "angle_max_abs_cap": 0.04,
                "require_objective_activation": True,
                "q3_gain_allowed_loss_increase": 0,
                "q3_order_guard_allowed_loss_increase": 0,
                "q3_score_distill_allowed_loss_increase": 0.0001,
                "q5_order_guard_allowed_loss_increase": 0,
                "q5_score_distill_allowed_loss_increase": 0.00002,
                "nf_boundary_guard_allowed_loss_increase": 0,
            },
        }
        source_dir = root / "materialized"
        source_manifest_path = write_json(source_dir / "manifest.json", source_manifest)
        source_rows_path = write_jsonl(source_dir / "rows.jsonl", materialized_rows)
        source_preflight = {
            "schema": "eos.q3_aoqt_sidecar_materializer_preflight.v1",
            "created_at_utc": "2026-09-05T00:00:00Z",
            "quality_claim": False,
            "research_only": True,
            "actual_training_ran": False,
            "actual_eval_ran": False,
            "actual_vector_export_ran": False,
            "plan_sha256": materializer.sha256_file(plan_path),
            "input_sha256": {materializer.display_path(plan_path): materializer.sha256_file(plan_path)},
            "row_count": len(row_ids),
            "candidate_count": len(row_ids) * 2,
            "pair_count": len(row_ids) * 2,
            "turboquant_seed": 5581486560434873699,
            "topology_seed": 191,
            "quant_score_surface": "turboquant.ip.prepared_v1",
            "qrels_sha256_by_dataset": qrels_by_dataset,
            "source_semantics": ["q3_top10", "q5_top10", "nf_boundary80_120"],
            "outputs": {"manifest_json": str(source_manifest_path), "preflight_json": str(source_dir / "preflight.json"), "rows_jsonl": str(source_rows_path)},
            "calibration_manifest_sha256": materializer.go_struct_sha256(source_manifest),
            "legal_gates": source_manifest["legal_gates"],
            "objective_contract": source_manifest["objective_contract"],
            "training_contract": source_manifest["training_contract"],
            "candidate_eligibility_policy": source_manifest["candidate_eligibility_policy"],
        }
        source_preflight_path = write_json(source_dir / "preflight.json", source_preflight)
        return {
            "registry": registry,
            "plan": plan_path,
            "split": split_path,
            "validation": validation_path,
            "manifest": source_manifest_path,
            "rows": source_rows_path,
            "preflight": source_preflight_path,
        }

    def args_for(self, fixture: dict[str, Path | dict], output: Path, **overrides: str) -> object:
        values = {
            "split_manifest": fixture["split"],
            "split_validation": fixture["validation"],
            "fold_id": "fold-0",
            "source_plan": fixture["plan"],
            "source_materialized_manifest": fixture["manifest"],
            "source_materialized_rows": fixture["rows"],
            "source_materialized_preflight": fixture["preflight"],
            "output_dir": output,
            "expected_split_manifest_sha256": materializer.sha256_file(fixture["split"]),
            "expected_split_validation_sha256": materializer.sha256_file(fixture["validation"]),
            "expected_source_plan_sha256": materializer.sha256_file(fixture["plan"]),
            "expected_source_materialized_manifest_sha256": materializer.sha256_file(fixture["manifest"]),
            "expected_source_materialized_rows_sha256": materializer.sha256_file(fixture["rows"]),
            "expected_source_materialized_preflight_sha256": materializer.sha256_file(fixture["preflight"]),
        }
        values.update(overrides)
        return type("Args", (), values)

    def test_materializes_exact_fold_rows_and_runner_counts(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            fixture = self.make_fixture(root)
            out = root / "out"
            with mock.patch.object(materializer.splitter.heldout_gate, "PRODUCTION_TRUSTED_WORKLOAD_REGISTRY", copy.deepcopy(fixture["registry"])):
                receipt = materializer.materialize(self.args_for(fixture, out))
            manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
            preflight = json.loads((out / "preflight.json").read_text(encoding="utf-8"))
            row_ids = [json.loads(line)["row_id"] for line in (out / "rows.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(row_ids, receipt["row_ids"])
            self.assertEqual(manifest["row_count"], len(row_ids))
            self.assertEqual(manifest["row_id_sha256"], materializer.aoqt_row_id_sha256(row_ids))
            self.assertEqual(preflight["row_count"], len(row_ids))
            self.assertEqual(preflight["candidate_count"], len(row_ids) * 2)
            self.assertEqual(preflight["pair_count"], len(row_ids) * 2)
            self.assertEqual(preflight["calibration_manifest_sha256"], receipt["outputs"]["manifest_sha256"])

    def test_rejects_stale_expected_hash(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            fixture = self.make_fixture(Path(temp))
            with mock.patch.object(materializer.splitter.heldout_gate, "PRODUCTION_TRUSTED_WORKLOAD_REGISTRY", copy.deepcopy(fixture["registry"])):
                with self.assertRaisesRegex(materializer.MaterializeError, "split manifest sha256 mismatch"):
                    materializer.materialize(self.args_for(fixture, Path(temp) / "out", expected_split_manifest_sha256="0" * 64))

    def test_rejects_failed_validation_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            fixture = self.make_fixture(root)
            receipt = json.loads(Path(fixture["validation"]).read_text(encoding="utf-8"))
            receipt["passed"] = False
            write_json(Path(fixture["validation"]), receipt)
            args = self.args_for(fixture, root / "out")
            with mock.patch.object(materializer.splitter.heldout_gate, "PRODUCTION_TRUSTED_WORKLOAD_REGISTRY", copy.deepcopy(fixture["registry"])):
                with self.assertRaisesRegex(materializer.MaterializeError, "validation receipt did not pass"):
                    materializer.materialize(args)

    def test_rejects_missing_fold_row_id(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            fixture = self.make_fixture(root)
            rows = [json.loads(line) for line in Path(fixture["rows"]).read_text(encoding="utf-8").splitlines()]
            split = json.loads(Path(fixture["split"]).read_text(encoding="utf-8"))
            target = split["folds"][0]["train"]["row_ids"][0]
            rows = [row for row in rows if row["row_id"] != target]
            write_jsonl(Path(fixture["rows"]), rows)
            args = self.args_for(fixture, root / "out")
            with mock.patch.object(materializer.splitter.heldout_gate, "PRODUCTION_TRUSTED_WORKLOAD_REGISTRY", copy.deepcopy(fixture["registry"])):
                with self.assertRaisesRegex(materializer.MaterializeError, "source rows missing"):
                    materializer.materialize(args)

    def test_rejects_duplicate_source_row_id(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            fixture = self.make_fixture(root)
            rows = [json.loads(line) for line in Path(fixture["rows"]).read_text(encoding="utf-8").splitlines()]
            rows.append(copy.deepcopy(rows[0]))
            rows = sorted(rows, key=lambda row: row["row_id"])
            write_jsonl(Path(fixture["rows"]), rows)
            args = self.args_for(fixture, root / "out")
            with mock.patch.object(materializer.splitter.heldout_gate, "PRODUCTION_TRUSTED_WORKLOAD_REGISTRY", copy.deepcopy(fixture["registry"])):
                with self.assertRaisesRegex(materializer.MaterializeError, "duplicate source row_id"):
                    materializer.materialize(args)

    def test_refuses_to_overwrite_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            fixture = self.make_fixture(root)
            out = root / "out"
            out.mkdir()
            (out / "rows.jsonl").write_text("", encoding="utf-8")
            with mock.patch.object(materializer.splitter.heldout_gate, "PRODUCTION_TRUSTED_WORKLOAD_REGISTRY", copy.deepcopy(fixture["registry"])):
                with self.assertRaisesRegex(materializer.MaterializeError, "refusing to overwrite"):
                    materializer.materialize(self.args_for(fixture, out))


if __name__ == "__main__":
    unittest.main()
