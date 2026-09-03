#!/usr/bin/env python3
"""Tests for the AOQT Stage 2 calibration plan builder."""

from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import build_aoqt_stage2_calibration as builder  # noqa: E402


def write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def fake_sha(seed: str) -> str:
    return builder.sha256_json({"seed": seed})


class BuildAOQTStage2CalibrationTest(unittest.TestCase):
    def common_binding(self, anchor_sha: str, anchor_payload: dict, source_sha: str, dataset_source: dict[str, str]) -> dict[str, object]:
        return {
            "anchor": {
                "package_sha256": anchor_payload["package_sha256"],
                "manifest_sha256": anchor_sha,
            },
            "topology": {"id": builder.TOPOLOGY["id"], "sha256": builder.TOPOLOGY_SHA256},
            "legal_scope": copy.deepcopy(builder.LEGAL_SCOPE),
            "source_sha256": source_sha,
            "qrels_sha256": fake_sha("qrels"),
            "dataset_source_provenance": dict(sorted(dataset_source.items())),
            "dataset_source_provenance_sha256": builder.sha256_json(dict(sorted(dataset_source.items()))),
        }

    def fixture(self, root: Path) -> dict[str, object]:
        anchor = {
            "schema": builder.ANCHOR_SCHEMA,
            "package_path": "runs/anchor/d384-pre.mll",
            "package_sha256": fake_sha("anchor-package"),
            "embedding_dim": 384,
            "topology": copy.deepcopy(builder.TOPOLOGY),
            "legal_scope": copy.deepcopy(builder.LEGAL_SCOPE),
        }
        anchor_path = write_json(root / "anchor.json", anchor)
        anchor_manifest_sha = builder.sha256_file(anchor_path)
        exclusions = {
            name: {
                "schema": builder.EXCLUSION_SCHEMA,
                "name": name,
                "qids_by_dataset": {dataset: [f"{name}-{dataset}-qid"] for dataset in builder.ALLOWED_DATASETS},
                "source_sha256": fake_sha(f"{name}-source"),
            }
            for name in builder.REQUIRED_EXCLUSION_NAMES
        }
        query_ids = {dataset: [f"{dataset}-q1", f"{dataset}-q2"] for dataset in builder.ALLOWED_DATASETS}
        doc_ids = {dataset: [f"{dataset}-d{i:03d}" for i in range(1, 126)] for dataset in builder.ALLOWED_DATASETS}
        vector_paths = []
        vector_source = {
            dataset: {
                "query_vector": fake_sha(f"{dataset}-query-source"),
                "doc_vector": fake_sha(f"{dataset}-doc-source"),
            }
            for dataset in builder.ALLOWED_DATASETS
        }
        for dataset in builder.ALLOWED_DATASETS:
            for role, ids in (("query", query_ids[dataset]), ("doc", doc_ids[dataset])):
                component = f"{role}_vector"
                vector_paths.append(
                    write_json(
                        root / "vectors" / f"{dataset}.{role}.json",
                        {
                            "schema": builder.VECTOR_CACHE_SCHEMA,
                            "dataset": dataset,
                            "split": "train",
                            "role": role,
                            "cache_sha256": fake_sha(f"{dataset}-{role}"),
                            "ids": ids,
                            **self.common_binding(anchor_manifest_sha, anchor, vector_source[dataset][component], {component: vector_source[dataset][component]}),
                        },
                    )
                )
        score_paths = []
        for dataset in builder.ALLOWED_DATASETS:
            for bits in (3, 5):
                query_manifest = root / "vectors" / f"{dataset}.query.json"
                doc_manifest = root / "vectors" / f"{dataset}.doc.json"
                score_component = f"q{bits}_score"
                score_source = fake_sha(f"{dataset}-{score_component}-source")
                dataset_source = {
                    **vector_source[dataset],
                    score_component: score_source,
                }
                rows = []
                for qid in query_ids[dataset]:
                    docs = []
                    for rank, doc_id in enumerate(doc_ids[dataset][:120], start=1):
                        gain = 0
                        if rank == 1:
                            gain = 1
                        if dataset == "nfcorpus" and rank == 80:
                            gain = 2
                        docs.append({"doc_id": doc_id, "rank": rank, "gain": gain})
                    rows.append({"qid": qid, "docs": docs})
                score_paths.append(
                    write_json(
                        root / "scores" / f"{dataset}.q{bits}.json",
                        {
                            "schema": builder.SCORE_CACHE_SCHEMA,
                            "dataset": dataset,
                            "split": "train",
                            "bits": bits,
                            "score_mode": "prepared_ip",
                            "top_k": 120,
                            "turboquant": {"score_mode": "prepared_ip", "bits": bits, "seed": builder.DEFAULT_TURBOQUANT_SEED},
                            "vector_cache": {
                                "query_manifest_sha256": builder.sha256_file(query_manifest),
                                "query_cache_sha256": fake_sha(f"{dataset}-query"),
                                "doc_manifest_sha256": builder.sha256_file(doc_manifest),
                                "doc_cache_sha256": fake_sha(f"{dataset}-doc"),
                            },
                            **self.common_binding(anchor_manifest_sha, anchor, score_source, dataset_source),
                            "rows": rows,
                        },
                    )
                )
        exclusion_paths = [write_json(root / "exclusions" / f"{name}.json", payload) for name, payload in exclusions.items()]
        output = root / "out" / "plan.json"
        return {
            "anchor": anchor_path,
            "exclusions": exclusion_paths,
            "vectors": vector_paths,
            "scores": score_paths,
            "output": output,
            "anchor_payload": anchor,
            "score_paths": score_paths,
            "vector_paths": vector_paths,
        }

    def build_args(self, fixture: dict[str, object]) -> list[str]:
        argv = ["--anchor-manifest", str(fixture["anchor"]), "--output-plan", str(fixture["output"])]
        for path in fixture["exclusions"]:
            argv.extend(["--exclusion-qids", str(path)])
        for path in fixture["vectors"]:
            argv.extend(["--vector-cache", str(path)])
        for path in fixture["scores"]:
            argv.extend(["--score-cache", str(path)])
        return argv

    def test_writes_deterministic_plan_without_vectors_or_numeric_scores(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fixture = self.fixture(Path(tmp))
            args = builder.parse_args(self.build_args(fixture))
            plan1 = builder.build_plan(args)
            plan2 = builder.build_plan(args)
            volatile = "created_utc"
            self.assertEqual({k: v for k, v in plan1.items() if k != volatile}, {k: v for k, v in plan2.items() if k != volatile})
            self.assertEqual(plan1["schema"], builder.SCHEMA)
            self.assertEqual(plan1["mode"], "plan_only")
            self.assertEqual(plan1["topology_seed"], builder.DEFAULT_TOPOLOGY_SEED)
            self.assertEqual(plan1["turboquant"]["seed"], builder.DEFAULT_TURBOQUANT_SEED)
            self.assertFalse(plan1["actual_training_ran"])
            self.assertFalse(plan1["actual_eval_ran"])
            self.assertEqual(plan1["topology"]["angle_count"], 1536)
            self.assertEqual(plan1["legal_scope"], builder.LEGAL_SCOPE)
            self.assertEqual(plan1["workload"]["row_count"], 14)
            self.assertEqual(plan1["workload"]["bucket_counts"], {"nf_boundary80_120_guard": 2, "top10_guard": 12})
            self.assertEqual(plan1["workload"]["actual_train_pairs"], 0)
            self.assertEqual(plan1["workload"]["actual_vectors_written"], 0)
            self.assertEqual(plan1["workload"]["actual_similarity_values_written"], 0)
            self.assertEqual(plan1["row_selection_audit"]["skipped_row_count"], 0)
            self.assertEqual(plan1["row_selection_audit"]["skip_counts"], {})
            serialized = json.dumps(plan1, sort_keys=True)
            self.assertNotIn("embedding", json.dumps(plan1["rows"], sort_keys=True).lower())
            self.assertNotIn("vector", json.dumps(plan1["rows"], sort_keys=True).lower())
            self.assertNotIn('"score":', serialized)
            builder.require_output_has_no_vectors_or_numeric_scores(plan1)

    def test_main_writes_plan_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fixture = self.fixture(Path(tmp))
            self.assertEqual(builder.main(self.build_args(fixture)), 0)
            output = fixture["output"]
            assert isinstance(output, Path)
            plan = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(plan["schema"], builder.SCHEMA)
            self.assertEqual(plan["workload"]["bit_counts"], {"3": 8, "5": 6})

    def test_rejects_heldout_or_test_split_score_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fixture = self.fixture(Path(tmp))
            path = fixture["score_paths"][0]
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["split"] = "test"
            write_json(path, payload)
            with self.assertRaisesRegex(builder.PlanError, "split must be train"):
                builder.build_plan(builder.parse_args(self.build_args(fixture)))

    def test_rejects_forbidden_path_tokens(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fixture = self.fixture(Path(tmp))
            path = fixture["vector_paths"][0]
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["source_path"] = "datasets/dev/corpus.jsonl"
            write_json(path, payload)
            with self.assertRaisesRegex(builder.PlanError, "forbidden split token"):
                builder.build_plan(builder.parse_args(self.build_args(fixture)))

    def test_rejects_raw_vectors(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fixture = self.fixture(Path(tmp))
            path = fixture["vector_paths"][0]
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["vectors"] = [[0.1, 0.2]]
            write_json(path, payload)
            with self.assertRaisesRegex(builder.PlanError, "raw vector-like field"):
                builder.build_plan(builder.parse_args(self.build_args(fixture)))

    def test_rejects_numeric_score_fields_in_score_cache_docs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fixture = self.fixture(Path(tmp))
            path = fixture["score_paths"][0]
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["rows"][0]["docs"][0]["score"] = 99.0
            write_json(path, payload)
            with self.assertRaisesRegex(builder.PlanError, "doc_id/rank/gain"):
                builder.build_plan(builder.parse_args(self.build_args(fixture)))

    def test_rejects_qid_only_exclusion_violations(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fixture = self.fixture(Path(tmp))
            path = fixture["exclusions"][0]
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["doc_ids"] = ["leak"]
            write_json(path, payload)
            with self.assertRaisesRegex(builder.PlanError, "qid-only"):
                builder.build_plan(builder.parse_args(self.build_args(fixture)))

    def test_rejects_legacy_global_exclusion_qids(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fixture = self.fixture(Path(tmp))
            path = fixture["exclusions"][0]
            payload = json.loads(path.read_text(encoding="utf-8"))
            del payload["qids_by_dataset"]
            payload["qids"] = ["ambiguous-qid"]
            write_json(path, payload)
            with self.assertRaisesRegex(builder.PlanError, "ambiguous legacy global qids"):
                builder.build_plan(builder.parse_args(self.build_args(fixture)))

    def test_rejects_exclusion_missing_dataset_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fixture = self.fixture(Path(tmp))
            path = fixture["exclusions"][0]
            payload = json.loads(path.read_text(encoding="utf-8"))
            del payload["qids_by_dataset"]["scifact"]
            write_json(path, payload)
            with self.assertRaisesRegex(builder.PlanError, "missing required dataset coverage"):
                builder.build_plan(builder.parse_args(self.build_args(fixture)))

    def test_rejects_nested_exclusion_payload_even_without_doc_ids_field(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fixture = self.fixture(Path(tmp))
            path = fixture["exclusions"][0]
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["payload"] = {"qrels": [{"qid": "secret", "doc_id": "d1", "score": 1.0}]}
            write_json(path, payload)
            with self.assertRaisesRegex(builder.PlanError, "unexpected fields"):
                builder.build_plan(builder.parse_args(self.build_args(fixture)))

    def test_rejects_exclusion_without_source_hash(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fixture = self.fixture(Path(tmp))
            path = fixture["exclusions"][0]
            payload = json.loads(path.read_text(encoding="utf-8"))
            del payload["source_sha256"]
            write_json(path, payload)
            with self.assertRaisesRegex(builder.PlanError, "source hash provenance"):
                builder.build_plan(builder.parse_args(self.build_args(fixture)))

    def test_rejects_duplicate_exclusion_names_before_qid_loss(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture = self.fixture(root)
            score_path = fixture["score_paths"][0]
            score = json.loads(score_path.read_text(encoding="utf-8"))
            original_dev4 = fixture["exclusions"][0]
            original_payload = json.loads(original_dev4.read_text(encoding="utf-8"))
            original_payload["qids_by_dataset"][score["dataset"]] = [score["rows"][0]["qid"]]
            write_json(original_dev4, original_payload)
            duplicate_payload = copy.deepcopy(original_payload)
            duplicate_payload["qids_by_dataset"][score["dataset"]] = ["replacement-dev4-qid"]
            duplicate_path = write_json(root / "exclusions" / "dev4-duplicate.json", duplicate_payload)
            fixture["exclusions"] = [*fixture["exclusions"], duplicate_path]
            with self.assertRaisesRegex(builder.PlanError, "duplicate qid-only set 'dev4'"):
                builder.build_plan(builder.parse_args(self.build_args(fixture)))

    def test_rejects_vector_cache_anchor_binding_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fixture = self.fixture(Path(tmp))
            path = fixture["vector_paths"][0]
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["anchor"]["package_sha256"] = fake_sha("other-anchor")
            write_json(path, payload)
            with self.assertRaisesRegex(builder.PlanError, "anchor package_sha256 binding mismatch"):
                builder.build_plan(builder.parse_args(self.build_args(fixture)))

    def test_rejects_vector_cache_without_qrels_hash(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fixture = self.fixture(Path(tmp))
            path = fixture["vector_paths"][0]
            payload = json.loads(path.read_text(encoding="utf-8"))
            del payload["qrels_sha256"]
            write_json(path, payload)
            with self.assertRaisesRegex(builder.PlanError, "qrels_sha256 provenance"):
                builder.build_plan(builder.parse_args(self.build_args(fixture)))

    def test_rejects_fiqa_q3_qrels_provenance_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fixture = self.fixture(Path(tmp))
            for path in fixture["score_paths"]:
                if str(path).endswith("fiqa.q3.json"):
                    payload = json.loads(path.read_text(encoding="utf-8"))
                    payload["qrels_sha256"] = fake_sha("fiqa-q3-drifted-qrels")
                    write_json(path, payload)
                    break
            with self.assertRaisesRegex(builder.PlanError, "qrels provenance mismatch for dataset fiqa"):
                builder.build_plan(builder.parse_args(self.build_args(fixture)))

    def test_rejects_dataset_source_provenance_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fixture = self.fixture(Path(tmp))
            for path in fixture["score_paths"]:
                if str(path).endswith("fiqa.q5.json"):
                    payload = json.loads(path.read_text(encoding="utf-8"))
                    payload["dataset_source_provenance"]["query_vector"] = fake_sha("fiqa-query-drifted-source")
                    payload["dataset_source_provenance_sha256"] = builder.sha256_json(payload["dataset_source_provenance"])
                    write_json(path, payload)
                    break
            with self.assertRaisesRegex(builder.PlanError, "source provenance mismatch for dataset fiqa"):
                builder.build_plan(builder.parse_args(self.build_args(fixture)))

    def test_rejects_vector_manifest_source_sha_component_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fixture = self.fixture(Path(tmp))
            path = fixture["vector_paths"][0]
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["source_sha256"] = fake_sha("self-consistent-but-wrong-query-vector-artifact")
            write_json(path, payload)
            with self.assertRaisesRegex(builder.PlanError, "query_vector source_sha256 does not match"):
                builder.build_plan(builder.parse_args(self.build_args(fixture)))

    def test_rejects_score_manifest_source_sha_component_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fixture = self.fixture(Path(tmp))
            for path in fixture["score_paths"]:
                if str(path).endswith("fiqa.q3.json"):
                    payload = json.loads(path.read_text(encoding="utf-8"))
                    payload["source_sha256"] = fake_sha("self-consistent-but-wrong-q3-score-artifact")
                    write_json(path, payload)
                    break
            with self.assertRaisesRegex(builder.PlanError, "q3_score source_sha256 does not match"):
                builder.build_plan(builder.parse_args(self.build_args(fixture)))

    def test_rejects_dataset_source_provenance_signature_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fixture = self.fixture(Path(tmp))
            path = fixture["vector_paths"][0]
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["dataset_source_provenance_sha256"] = fake_sha("wrong-dataset-source-signature")
            write_json(path, payload)
            with self.assertRaisesRegex(builder.PlanError, "dataset_source_provenance_sha256 mismatch"):
                builder.build_plan(builder.parse_args(self.build_args(fixture)))

    def test_rejects_score_cache_turboquant_seed_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fixture = self.fixture(Path(tmp))
            path = fixture["score_paths"][0]
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["turboquant"]["seed"] = builder.DEFAULT_TOPOLOGY_SEED
            write_json(path, payload)
            with self.assertRaisesRegex(builder.PlanError, "TurboQuant prepared-IP config binding mismatch"):
                builder.build_plan(builder.parse_args(self.build_args(fixture)))

    def test_accepts_explicit_turboquant_seed_without_changing_topology_seed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fixture = self.fixture(Path(tmp))
            seed = 1234567
            for path in fixture["score_paths"]:
                payload = json.loads(path.read_text(encoding="utf-8"))
                payload["turboquant"]["seed"] = seed
                write_json(path, payload)
            args = builder.parse_args(self.build_args(fixture) + ["--turboquant-seed", str(seed)])
            plan = builder.build_plan(args)
            self.assertEqual(plan["seed"], builder.DEFAULT_TOPOLOGY_SEED)
            self.assertEqual(plan["topology_seed"], builder.DEFAULT_TOPOLOGY_SEED)
            self.assertEqual(plan["turboquant"]["seed"], seed)

    def test_rejects_score_cache_vector_cache_binding_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fixture = self.fixture(Path(tmp))
            path = fixture["score_paths"][0]
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["vector_cache"]["doc_cache_sha256"] = fake_sha("wrong-doc-cache")
            write_json(path, payload)
            with self.assertRaisesRegex(builder.PlanError, "vector cache binding mismatch"):
                builder.build_plan(builder.parse_args(self.build_args(fixture)))

    def test_rejects_missing_top10_guard_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fixture = self.fixture(Path(tmp))
            path = fixture["score_paths"][0]
            payload = json.loads(path.read_text(encoding="utf-8"))
            for row in payload["rows"]:
                for doc in row["docs"][:10]:
                    doc["gain"] = 0
                row["docs"][79]["gain"] = 1
            write_json(path, payload)
            with self.assertRaisesRegex(builder.PlanError, "missing top10 guard coverage"):
                builder.build_plan(builder.parse_args(self.build_args(fixture)))

    def test_skips_all_positive_and_all_zero_guard_windows_with_deterministic_audit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fixture = self.fixture(Path(tmp))
            for path in fixture["score_paths"]:
                if str(path).endswith("fiqa.q3.json"):
                    payload = json.loads(path.read_text(encoding="utf-8"))
                    for doc in payload["rows"][0]["docs"][:10]:
                        doc["gain"] = 1
                    write_json(path, payload)
                if str(path).endswith("scifact.q5.json"):
                    payload = json.loads(path.read_text(encoding="utf-8"))
                    for doc in payload["rows"][0]["docs"]:
                        doc["gain"] = 0
                    write_json(path, payload)
            args = builder.parse_args(self.build_args(fixture))
            plan1 = builder.build_plan(args)
            plan2 = builder.build_plan(args)

            audit = plan1["row_selection_audit"]
            self.assertEqual(audit, plan2["row_selection_audit"])
            self.assertEqual(audit["skipped_row_count"], 2)
            self.assertEqual(
                audit["skip_counts"],
                {
                    "top10_guard:no_positive_candidate": 1,
                    "top10_guard:no_zero_gain_candidate": 1,
                },
            )
            skipped_ids = {row["row_id"] for row in audit["skipped_rows"]}
            emitted_ids = {row["row_id"] for row in plan1["rows"]}
            self.assertFalse(skipped_ids & emitted_ids)
            self.assertIn("fiqa.q3.top10.fiqa-q1", skipped_ids)
            self.assertIn("scifact.q5.top10.scifact-q1", skipped_ids)

    def test_all_positive_top10_rows_fail_closed_when_required_coverage_disappears(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fixture = self.fixture(Path(tmp))
            for path in fixture["score_paths"]:
                if str(path).endswith("fiqa.q3.json"):
                    payload = json.loads(path.read_text(encoding="utf-8"))
                    for row in payload["rows"]:
                        for doc in row["docs"][:10]:
                            doc["gain"] = 1
                    write_json(path, payload)
                    break
            with self.assertRaisesRegex(builder.PlanError, "missing top10 guard coverage"):
                builder.build_plan(builder.parse_args(self.build_args(fixture)))

    def test_emitted_guard_rows_always_have_eligible_pairs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fixture = self.fixture(Path(tmp))
            plan = builder.build_plan(builder.parse_args(self.build_args(fixture)))
            top10 = next(row for row in plan["rows"] if row["row_id"] == "fiqa.q3.top10.fiqa-q1")

            self.assertEqual(top10["positive_doc_ids"], ["fiqa-d001"])
            self.assertEqual(top10["negative_doc_ids"], [f"fiqa-d{i:03d}" for i in range(2, 11)])
            self.assertEqual(len(top10["pair_ids"]), 9)
            for row in plan["rows"]:
                self.assertTrue(row["positive_doc_ids"], row["row_id"])
                self.assertTrue(row["negative_doc_ids"], row["row_id"])
                self.assertTrue(row["pair_ids"], row["row_id"])

    def test_rejects_missing_nfcorpus_q3_boundary_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fixture = self.fixture(Path(tmp))
            for path in fixture["score_paths"]:
                if str(path).endswith("nfcorpus.q3.json"):
                    payload = json.loads(path.read_text(encoding="utf-8"))
                    for row in payload["rows"]:
                        for doc in row["docs"][79:120]:
                            doc["gain"] = 0
                    write_json(path, payload)
                    break
            with self.assertRaisesRegex(builder.PlanError, "missing NFCorpus q3 boundary"):
                builder.build_plan(builder.parse_args(self.build_args(fixture)))

    def test_does_not_emit_nfcorpus_q5_boundary_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fixture = self.fixture(Path(tmp))
            plan = builder.build_plan(builder.parse_args(self.build_args(fixture)))
            self.assertFalse(any(row["dataset"] == "nfcorpus" and row["bits"] == 5 and row["bucket"] == "nf_boundary80_120_guard" for row in plan["rows"]))

    def test_malformed_bits_and_top_k_are_plan_errors(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fixture = self.fixture(Path(tmp))
            path = fixture["score_paths"][0]
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["bits"] = "q3"
            write_json(path, payload)
            with self.assertRaisesRegex(builder.PlanError, "bits must be an integer"):
                builder.build_plan(builder.parse_args(self.build_args(fixture)))
            payload["bits"] = 3
            payload["top_k"] = "one-twenty"
            write_json(path, payload)
            with self.assertRaisesRegex(builder.PlanError, "top_k must be an integer"):
                builder.build_plan(builder.parse_args(self.build_args(fixture)))

    def test_rejects_numeric_strings_for_integer_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fixture = self.fixture(Path(tmp))
            path = fixture["score_paths"][0]
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["bits"] = "3"
            write_json(path, payload)
            with self.assertRaisesRegex(builder.PlanError, "bits must be an integer"):
                builder.build_plan(builder.parse_args(self.build_args(fixture)))
            payload["bits"] = 3
            payload["top_k"] = "120"
            write_json(path, payload)
            with self.assertRaisesRegex(builder.PlanError, "top_k must be an integer"):
                builder.build_plan(builder.parse_args(self.build_args(fixture)))

    def test_rejects_float_integer_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fixture = self.fixture(Path(tmp))
            path = fixture["score_paths"][0]
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["bits"] = 3.0
            write_json(path, payload)
            with self.assertRaisesRegex(builder.PlanError, "bits must be an integer"):
                builder.build_plan(builder.parse_args(self.build_args(fixture)))
            payload["bits"] = 3
            payload["top_k"] = 120.0
            write_json(path, payload)
            with self.assertRaisesRegex(builder.PlanError, "top_k must be an integer"):
                builder.build_plan(builder.parse_args(self.build_args(fixture)))

    def test_rejects_score_cache_doc_rank_bool(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fixture = self.fixture(Path(tmp))
            path = fixture["score_paths"][0]
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["rows"][0]["docs"][0]["rank"] = True
            write_json(path, payload)
            with self.assertRaisesRegex(builder.PlanError, "rank must be an integer"):
                builder.build_plan(builder.parse_args(self.build_args(fixture)))

    def test_rejects_score_cache_doc_gain_bool(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fixture = self.fixture(Path(tmp))
            path = fixture["score_paths"][0]
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["rows"][0]["docs"][0]["gain"] = True
            write_json(path, payload)
            with self.assertRaisesRegex(builder.PlanError, "gain must be an integer"):
                builder.build_plan(builder.parse_args(self.build_args(fixture)))

    def test_rejects_score_cache_doc_rank_and_gain_non_integer_values(self) -> None:
        invalid_cases = [
            ("rank", 1.0, "rank must be an integer"),
            ("rank", "1", "rank must be an integer"),
            ("rank", None, "rank must be an integer"),
            ("rank", ["1"], "rank must be an integer"),
            ("rank", {"value": 1}, "rank must be an integer"),
            ("gain", 1.0, "gain must be an integer"),
            ("gain", "1", "gain must be an integer"),
            ("gain", None, "gain must be an integer"),
            ("gain", ["1"], "gain must be an integer"),
            ("gain", {"value": 1}, "gain must be an integer"),
        ]
        for field, value, pattern in invalid_cases:
            with self.subTest(field=field, value=value):
                with tempfile.TemporaryDirectory() as tmp:
                    fixture = self.fixture(Path(tmp))
                    path = fixture["score_paths"][0]
                    payload = json.loads(path.read_text(encoding="utf-8"))
                    payload["rows"][0]["docs"][0][field] = value
                    write_json(path, payload)
                    with self.assertRaisesRegex(builder.PlanError, pattern):
                        builder.build_plan(builder.parse_args(self.build_args(fixture)))

    def test_skips_same_dataset_excluded_source_qid_with_deterministic_audit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fixture = self.fixture(Path(tmp))
            exclusion_path = fixture["exclusions"][0]
            score_path = fixture["score_paths"][0]
            exclusion = json.loads(exclusion_path.read_text(encoding="utf-8"))
            score = json.loads(score_path.read_text(encoding="utf-8"))
            excluded_qid = score["rows"][0]["qid"]
            exclusion_set_name = exclusion["name"]
            exclusion["qids_by_dataset"][score["dataset"]] = [excluded_qid]
            write_json(exclusion_path, exclusion)
            args = builder.parse_args(self.build_args(fixture))
            plan1 = builder.build_plan(args)
            plan2 = builder.build_plan(args)

            audit = plan1["row_selection_audit"]["source_exclusions"]
            self.assertEqual(audit, plan2["row_selection_audit"]["source_exclusions"])
            self.assertEqual(audit["excluded_source_row_count"], 2)
            self.assertEqual(audit["excluded_source_row_count_by_dataset"], {score["dataset"]: 2})
            self.assertEqual(
                audit["excluded_source_row_count_by_dataset_exclusion_set"],
                {f"{score['dataset']}:{exclusion_set_name}": 2},
            )
            self.assertEqual(
                audit["excluded_source_row_count_by_dataset_bits"],
                {f"{score['dataset']}:q3": 1, f"{score['dataset']}:q5": 1},
            )
            self.assertEqual(
                audit["excluded_source_qid_hashes_by_dataset_sha256"],
                builder.sha256_json(
                    {
                        dataset: [builder.sha256_json({"dataset": dataset, "qid": excluded_qid})] if dataset == score["dataset"] else []
                        for dataset in builder.ALLOWED_DATASETS
                    }
                ),
            )
            self.assertNotIn(excluded_qid, json.dumps(audit, sort_keys=True))
            self.assertNotIn(excluded_qid, {row["qid"] for row in plan1["rows"]})
            self.assertFalse(any(excluded_qid in row["row_id"] for row in plan1["rows"]))
            self.assertEqual(plan1["workload"]["row_count"], 12)
            self.assertEqual(plan1["workload"]["bit_counts"], {"3": 7, "5": 5})

            score_audits = [
                item
                for item in plan1["input_manifests"]["score_caches"]
                if item["dataset"] == score["dataset"]
            ]
            self.assertEqual({item["excluded_source_row_count"] for item in score_audits}, {1})
            self.assertEqual({item["source_qid_count"] for item in score_audits}, {2})
            self.assertEqual({item["qid_count"] for item in score_audits}, {1})

    def test_excluded_source_qid_still_must_obey_score_cache_row_schema(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fixture = self.fixture(Path(tmp))
            exclusion_path = fixture["exclusions"][0]
            score_path = fixture["score_paths"][0]
            exclusion = json.loads(exclusion_path.read_text(encoding="utf-8"))
            score = json.loads(score_path.read_text(encoding="utf-8"))
            exclusion["qids_by_dataset"][score["dataset"]] = [score["rows"][0]["qid"]]
            score["rows"][0]["docs"][0]["score"] = 99.0
            write_json(exclusion_path, exclusion)
            write_json(score_path, score)
            with self.assertRaisesRegex(builder.PlanError, "doc_id/rank/gain"):
                builder.build_plan(builder.parse_args(self.build_args(fixture)))

    def test_allows_cross_dataset_qid_collision_in_exclusions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fixture = self.fixture(Path(tmp))
            exclusion_path = fixture["exclusions"][0]
            score_path = fixture["score_paths"][0]
            exclusion = json.loads(exclusion_path.read_text(encoding="utf-8"))
            score = json.loads(score_path.read_text(encoding="utf-8"))
            other_dataset = next(dataset for dataset in builder.ALLOWED_DATASETS if dataset != score["dataset"])
            exclusion["qids_by_dataset"][other_dataset] = [score["rows"][0]["qid"]]
            write_json(exclusion_path, exclusion)
            plan = builder.build_plan(builder.parse_args(self.build_args(fixture)))
            self.assertEqual(plan["selection_policy"]["official_exclusion_mode"], "dataset_scoped_qid_only")
            self.assertIn(score["rows"][0]["qid"], {row["qid"] for row in plan["rows"] if row["dataset"] == score["dataset"]})
            self.assertEqual(plan["row_selection_audit"]["source_exclusions"]["excluded_source_row_count"], 0)

    def test_exclusion_filter_fails_closed_when_required_guard_coverage_vanishes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fixture = self.fixture(Path(tmp))
            exclusion_path = fixture["exclusions"][0]
            exclusion = json.loads(exclusion_path.read_text(encoding="utf-8"))
            exclusion["qids_by_dataset"]["fiqa"] = ["fiqa-q1", "fiqa-q2"]
            write_json(exclusion_path, exclusion)
            with self.assertRaisesRegex(builder.PlanError, "missing top10 guard coverage"):
                builder.build_plan(builder.parse_args(self.build_args(fixture)))

    def test_rejects_missing_q5_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fixture = self.fixture(Path(tmp))
            scores = [path for path in fixture["scores"] if ".q5." not in str(path)]
            fixture["scores"] = scores
            with self.assertRaisesRegex(builder.PlanError, "missing train .* q5"):
                builder.build_plan(builder.parse_args(self.build_args(fixture)))

    def test_rejects_anchor_pin_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fixture = self.fixture(Path(tmp))
            argv = self.build_args(fixture) + ["--expected-anchor-sha256", fake_sha("wrong")]
            with self.assertRaisesRegex(builder.PlanError, "expected pin"):
                builder.build_plan(builder.parse_args(argv))

    def test_rejects_output_numeric_score_recursively(self) -> None:
        with self.assertRaisesRegex(builder.PlanError, "numeric score"):
            builder.require_output_has_no_vectors_or_numeric_scores({"nested": [{"score": 1.0}]})


if __name__ == "__main__":
    unittest.main()
