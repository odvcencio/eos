#!/usr/bin/env python3
"""Tests for q3 R4 proxy preflight helpers."""

from __future__ import annotations

import csv
import json
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

import q3_r4_proxy_preflight as preflight


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def write_qrels(path: Path, rows: list[tuple[str, str, int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["query-id", "corpus-id", "score"])
        writer.writerows(rows)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def per_query(
    qid: str,
    gains: list[int],
    *,
    dataset: str = "fixture",
    ndcg: float = 0.5,
    recall10: float = 0.5,
    recall100: float = 1.0,
) -> dict:
    top_k = []
    for index in range(100):
        top_k.append(
            {
                "rank": index + 1,
                "doc_id": f"D{index + 1}",
                "score": 1.0 - index / 1000.0,
                "relevance": gains[index] if index < len(gains) else 0,
            }
        )
    return {
        "schema": "manta.embedding_turboquant_retrieval_per_query.v1",
        "dataset": dataset,
        "query_id": qid,
        "method": "turboquant_ip_b3",
        "bits": 3,
        "scoring_surface": "turboquant_ip_prepared",
        "quantizer_seed": 5581486560434873699,
        "relevant_count": 2,
        "quality": {"ndcg_at_10": ndcg, "recall_at_10": recall10, "recall_at_100": recall100},
        "top_k": top_k,
    }


def metric_payload(dataset: str, artifact: str, qrels: dict, ndcg: float, recall100: float) -> dict:
    return {
        "schema": "manta.embedding_turboquant_retrieval_metrics.v1",
        "dataset": dataset,
        "artifact": artifact,
        "config": {"bits": [3], "quantizer_seed": 5581486560434873699, "top_k": 100},
        "inputs": {
            "qrels_path": qrels["path"],
            "qrels_sha256": qrels["sha256"],
            "queries": qrels["qid_count"],
            "relevant_pairs": qrels["relevant_rows"],
            "scored_pairs": 100,
        },
        "rows": [
            {
                "method": "turboquant_ip_b3",
                "bits": 3,
                "candidate_count": 100,
                "candidates_scored": 100,
                "quality": {"ndcg_at_10": ndcg, "recall_at_100": recall100},
            }
        ],
    }


TSV_HEADER = [
    "dataset",
    "row",
    "bits",
    "method",
    "rerank_overfetch",
    "rerank_storage",
    "rerank_bits",
    "ndcg_at_10",
    "ndcg_at_100",
    "mrr_at_10",
    "precision_at_1",
    "precision_at_5",
    "precision_at_10",
    "hit_at_1",
    "hit_at_5",
    "hit_at_10",
    "map_at_10",
    "map_at_100",
    "recall_at_10",
    "recall_at_100",
    "ndcg_at_10_delta",
    "recall_at_100_delta",
    "vector_bytes",
    "dense_vector_bytes",
    "rerank_sidecar_bytes",
    "total_vector_bytes",
    "compression_ratio",
    "total_compression_ratio",
    "scores_per_second",
    "candidate_count",
    "candidates_scored",
    "candidates_pruned",
    "pruning_supported",
    "pruning_used",
    "candidate_decisions_per_second",
    "query_latency_p50_ms",
    "query_latency_p95_ms",
    "query_latency_p99_ms",
    "query_latency_max_ms",
    "docs_per_second",
    "rerank_scores",
]


def write_metrics_tsv(path: Path, dataset: str, ndcg: float, recall100: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    dense = {key: "" for key in TSV_HEADER}
    dense.update(
        {
            "dataset": dataset,
            "row": "dense",
            "method": "float32",
            "ndcg_at_10": "0.500000",
            "recall_at_100": "1.000000",
            "candidate_count": "100",
            "candidates_scored": "100",
        }
    )
    q3 = {key: "" for key in TSV_HEADER}
    q3.update(
        {
            "dataset": dataset,
            "row": "quantized",
            "bits": "3",
            "method": "turboquant_ip_b3",
            "ndcg_at_10": f"{ndcg:.6f}",
            "recall_at_100": f"{recall100:.6f}",
            "candidate_count": "100",
            "candidates_scored": "100",
        }
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=TSV_HEADER)
        writer.writeheader()
        writer.writerow(dense)
        writer.writerow(q3)


class GateFixture:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.policy_path = root / "proxy-eval/q3-r4-proxy-metric-aligned-policy.json"
        self.summary_path = root / "proxy-eval/q3-r4-proxy-gate-summary.json"
        self.anchor_package = root / "anchor/d384-pre.mll"
        self.candidate_package = root / "candidate/d384-pre.mll"
        self.pkg = {
            "target_sha256": "pkg",
            "canonical_content_rollup_sha256": "roll",
            "sibling_count": 1,
            "sibling_hashes": [{"name": "d384-pre.mll", "bytes": 1, "sha256": "pkg"}],
        }
        self.qrels: dict[str, dict] = {}
        self.outputs: dict[str, dict[str, dict[str, str]]] = {"anchor": {}, "candidate": {}}
        self.exclusion_paths = [root / "excluded.dev.json", root / "excluded.reserve.json"]
        self.split_manifest_path = root / "qid-splits.manifest.json"
        self.heldout_paths = {dataset: root / f"heldout/{dataset}.tsv" for dataset in preflight.DOMAINS}

    def write_support_files(self) -> None:
        write_jsonl(self.root / "train.jsonl", [
            {"source_dataset": dataset, "source_query_id": f"Q-{dataset}", "selection_bucket": preflight.BUCKETS["promotion"]}
            for dataset in preflight.DOMAINS
        ])
        for split, path in (("dev4", self.exclusion_paths[0]), ("reserve4", self.exclusion_paths[1])):
            write_json(path, {"split": split, "selected_qids": {dataset: [] for dataset in preflight.DOMAINS}})
        write_json(
            self.split_manifest_path,
            {
                "schema": "fixture.qid_splits.v1",
                "splits": {
                    "dev4": str(self.exclusion_paths[0]),
                    "reserve4": str(self.exclusion_paths[1]),
                },
                "overlap": {dataset: 0 for dataset in preflight.DOMAINS},
            },
        )
        for dataset in preflight.DOMAINS:
            write_qrels(self.heldout_paths[dataset], [(f"T-{dataset}", "D9", 1)])

    def source_inputs(self) -> dict:
        return {
            "exclusion_qid_json": [preflight.exclusion_source_record(path) for path in self.exclusion_paths],
            "qid_split_manifests": [preflight.qid_split_manifest_record(self.split_manifest_path)],
            "heldout_test_qrels": {
                dataset: [preflight.qrels_all_stats(self.heldout_paths[dataset])]
                for dataset in preflight.DOMAINS
            },
        }

    def build(self) -> "GateFixture":
        self.write_support_files()
        train_rows = []
        manifest_datasets = {}
        for dataset in preflight.DOMAINS:
            qid = f"Q-{dataset}"
            qrels_path = self.root / f"proxy-qrels-r4/{dataset}/qrels/r4-train-proxy.tsv"
            write_qrels(qrels_path, [(qid, "D1", 1), (qid, "D2", 1)])
            stats = preflight.qrels_stats(qrels_path)
            stats["source_qrels"] = str(self.root / f"source/{dataset}/train.tsv")
            write_qrels(Path(stats["source_qrels"]), [(qid, "D1", 1), (qid, "D2", 1)])
            stats["source_qrels_sha256"] = preflight.sha256_file(Path(stats["source_qrels"]))
            stats["source_qrels_bytes"] = Path(stats["source_qrels"]).stat().st_size
            self.qrels[dataset] = stats
            manifest_datasets[dataset] = stats
            train_rows.append({"source_dataset": dataset, "source_query_id": qid, "selection_bucket": preflight.BUCKETS["promotion"]})
            for label in ("anchor", "candidate"):
                stem = self.root / f"proxy-eval/{label}.{dataset}.q3.r4-proxy"
                self.outputs[label][dataset] = {
                    "metrics_json": str(stem.with_suffix(".json")),
                    "metrics_tsv": str(stem.with_suffix(".tsv")),
                    "per_query_jsonl": str(stem.with_suffix(".per-query.jsonl")),
                }
        manifest_path = self.root / "proxy-qrels-r4/manifest.json"
        manifest = {
            "schema": f"{preflight.SCHEMA}.proxy_qrels_manifest.v1",
            "policy_path": str(self.policy_path),
            "source_inputs": self.source_inputs(),
            "datasets": manifest_datasets,
            "heldout_overlap": {
                dataset: {
                    "selected_qids": 1,
                    "frozen_dev_reserve_overlap": 0,
                    "official_test_overlap": 0,
                    "frozen_dev_reserve_overlap_qids": [],
                    "official_test_overlap_qids": [],
                }
                for dataset in preflight.DOMAINS
            },
        }
        write_json(manifest_path, manifest)
        policy = {
            "schema": preflight.POLICY_SCHEMA,
            "created_utc": "fixture",
            "status": "no_results_yet",
            "evaluation_executed": False,
            "packages": {"anchor": self.pkg, "candidate": self.pkg},
            "source_inputs": self.source_inputs(),
            "proxy_qrels": self.qrels,
            "proxy_qrels_manifest": {
                "path": str(manifest_path),
                "sha256": preflight.sha256_file(manifest_path),
                "bytes": manifest_path.stat().st_size,
            },
            "expected_outputs": self.outputs,
            "gates": {
                "macro_q3_ndcg_delta_min_exclusive": 0.0,
                "macro_q3_recall_at_100_delta_min": 0.0,
                "domain_q3_ndcg_delta_min": {"nfcorpus": 0.006, "fiqa": -0.0005, "scifact": -0.0005},
                "domain_recall_at_100_delta_min": {"nfcorpus": 0.0, "fiqa": 0.0, "scifact": 0.0},
            },
        }
        write_json(self.policy_path, policy)
        write_json(self.root / "proxy-eval/q3-r4-proxy-execution-plan-audit.json", {"policy_sha256": preflight.sha256_file(self.policy_path)})
        time.sleep(0.01)
        for dataset in preflight.DOMAINS:
            for label, ndcg in (("anchor", 0.5), ("candidate", 0.51 if dataset == "nfcorpus" else 0.5001)):
                self.write_result(label, dataset, ndcg, 1.0)
        return self

    def write_result(self, label: str, dataset: str, ndcg: float, recall100: float) -> None:
        paths = self.outputs[label][dataset]
        package = self.anchor_package if label == "anchor" else self.candidate_package
        qid = f"Q-{dataset}"
        write_json(Path(paths["metrics_json"]), metric_payload(dataset, str(package), self.qrels[dataset], ndcg, recall100))
        write_metrics_tsv(Path(paths["metrics_tsv"]), dataset, ndcg, recall100)
        write_jsonl(Path(paths["per_query_jsonl"]), [per_query(qid, [1, 1], dataset=dataset, ndcg=ndcg, recall10=1.0, recall100=recall100)])

    def rewrite_policy_manifest_and_plan(self, policy: dict) -> None:
        manifest_path = Path(policy["proxy_qrels_manifest"]["path"])
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["source_inputs"] = policy["source_inputs"]
        write_json(manifest_path, manifest)
        policy["proxy_qrels_manifest"] = {
            "path": str(manifest_path),
            "sha256": preflight.sha256_file(manifest_path),
            "bytes": manifest_path.stat().st_size,
        }
        write_json(self.policy_path, policy)
        write_json(self.root / "proxy-eval/q3-r4-proxy-execution-plan-audit.json", {"policy_sha256": preflight.sha256_file(self.policy_path)})

    def patches(self):
        return (
            mock.patch.object(preflight, "verify_fixed_inputs", return_value={}),
            mock.patch.object(preflight, "package_sibling_hashes", return_value=self.pkg),
            mock.patch.object(preflight, "EXPECTED_CANONICAL_PACKAGE_ROLLUPS", {"anchor": "roll", "candidate": "roll"}),
            mock.patch.object(preflight, "ANCHOR_PACKAGE", self.anchor_package),
            mock.patch.object(preflight, "CANDIDATE_PACKAGE", self.candidate_package),
            mock.patch.object(preflight, "TRAIN_JSONL", self.root / "train.jsonl"),
            mock.patch.object(preflight, "RAW_QRELS", {dataset: Path(self.qrels[dataset]["source_qrels"]) for dataset in preflight.DOMAINS}),
            mock.patch.object(preflight, "HELDOUT_QREL_PATHS", {dataset: [self.heldout_paths[dataset]] for dataset in preflight.DOMAINS}),
            mock.patch.object(preflight, "EXCLUSION_JSONS", self.exclusion_paths),
            mock.patch.object(preflight, "QID_SPLIT_MANIFESTS", [self.split_manifest_path]),
        )


class Q3R4ProxyPreflightTests(unittest.TestCase):
    def run_gate_fixture(self, fixture: GateFixture) -> dict:
        patches = fixture.patches()
        with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5], patches[6], patches[7], patches[8], patches[9]:
            return preflight.gate(fixture.policy_path, fixture.summary_path)

    def test_write_subset_qrels_dedupes_qids_and_preserves_positive_gains(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            raw = root / "raw.tsv"
            out = root / "subset.tsv"
            write_qrels(raw, [("Q1", "D1", 2), ("Q1", "D2", 1), ("Q1", "D3", 0), ("Q2", "D4", 1)])

            summary = preflight.write_subset_qrels(raw, {"Q1"}, out)

            self.assertEqual(summary["qid_count"], 1)
            self.assertEqual(summary["relevant_rows"], 2)
            self.assertEqual(out.read_text(encoding="utf-8").splitlines(), ["query-id\tcorpus-id\tscore", "Q1\tD1\t2", "Q1\tD2\t1"])

    def test_write_subset_qrels_fails_closed_when_selected_qid_missing_qrels(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            raw = root / "raw.tsv"
            write_qrels(raw, [("Q1", "D1", 1)])

            with self.assertRaisesRegex(preflight.PreflightError, "missing train qrels"):
                preflight.write_subset_qrels(raw, {"Q2"}, root / "subset.tsv")

    def test_substitution_audit_rejects_lower_gain_replacement(self) -> None:
        anchor = per_query("Q1", [2, 0, 0, 0, 0, 0, 0, 0, 0, 0], ndcg=0.7, recall10=0.5)
        candidate = per_query("Q1", [0, 1, 0, 0, 0, 0, 0, 0, 0, 0], ndcg=0.6, recall10=0.5)
        candidate["top_k"][0]["doc_id"] = "DX"

        audit = preflight.substitution_audit(anchor, candidate)

        self.assertTrue(audit["affected_top10"])
        self.assertFalse(audit["matching_ok"])
        self.assertFalse(audit["affected_metric_ok"])

    def test_load_per_query_fails_on_seed_or_surface_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "bad.jsonl"
            row = per_query("Q1", [1])
            row["quantizer_seed"] = 7
            path.write_text(json.dumps(row) + "\n", encoding="utf-8")

            with self.assertRaisesRegex(preflight.PreflightError, "seed/surface mismatch"):
                preflight.load_per_query(path)

    def test_build_preflight_fails_if_expected_outputs_already_exist(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            train = root / "train.jsonl"
            row = {
                "source_dataset": "fiqa",
                "source_query_id": "Q1",
                "selection_bucket": preflight.BUCKETS["promotion"],
            }
            train.write_text(json.dumps(row) + "\n", encoding="utf-8")
            for dataset in preflight.DOMAINS:
                write_qrels(root / dataset / "train.tsv", [("Q1", "D1", 1)])
                write_qrels(root / dataset / "test.tsv", [("T1", "D2", 1)])
            write_json(root / "dev.json", {"selected_qids": {dataset: [] for dataset in preflight.DOMAINS}})
            existing = root / "proxy-eval/anchor.fiqa.q3.r4-proxy.json"
            existing.parent.mkdir(parents=True)
            existing.write_text("{}\n", encoding="utf-8")

            with mock.patch.object(preflight, "verify_fixed_inputs", return_value={"train_jsonl": {"sha256": "x"}}), \
                mock.patch.object(preflight, "TRAIN_JSONL", train), \
                mock.patch.object(preflight, "RAW_QRELS", {dataset: root / dataset / "train.tsv" for dataset in preflight.DOMAINS}), \
                mock.patch.object(preflight, "HELDOUT_QREL_PATHS", {dataset: [root / dataset / "test.tsv"] for dataset in preflight.DOMAINS}), \
                mock.patch.object(preflight, "EXCLUSION_JSONS", [root / "dev.json"]), \
                mock.patch.object(preflight, "R4_ROOT", root), \
                mock.patch.object(preflight, "package_sibling_hashes", return_value={"target_sha256": "p"}):
                with self.assertRaisesRegex(preflight.PreflightError, "proxy outputs already exist"):
                    preflight.build_preflight()

    def test_gate_complete_valid_fixture_writes_summary_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            fixture = GateFixture(Path(td)).build()

            summary = self.run_gate_fixture(fixture)

            self.assertTrue(summary["gate_pass"])
            self.assertTrue(fixture.summary_path.is_file())
            self.assertEqual(summary["proxy_qrels_audit"]["manifest_sha256"], preflight.sha256_file(Path(summary["proxy_qrels_audit"]["manifest_path"])))
            self.assertEqual(len(summary["output_hashes"]), 18)

    def test_gate_rejects_extra_output(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            fixture = GateFixture(Path(td)).build()
            (Path(td) / "proxy-eval/extra.fiqa.q3.r4-proxy.json").write_text("{}\n", encoding="utf-8")

            with self.assertRaisesRegex(preflight.PreflightError, "unexpected proxy output extras"):
                self.run_gate_fixture(fixture)

    def test_gate_rejects_stale_output(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            fixture = GateFixture(Path(td)).build()
            stale = Path(fixture.outputs["anchor"]["fiqa"]["metrics_json"])
            old = fixture.policy_path.stat().st_mtime - 10
            os.utime(stale, (old, old))

            with self.assertRaisesRegex(preflight.PreflightError, "stale proxy output"):
                self.run_gate_fixture(fixture)

    def test_gate_rejects_proxy_qrels_file_hash_qid_or_gain_drift(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            fixture = GateFixture(Path(td)).build()
            write_qrels(Path(fixture.qrels["fiqa"]["path"]), [("Q-fiqa", "D1", 2), ("Q-fiqa", "D2", 1)])

            with self.assertRaisesRegex(preflight.PreflightError, "fiqa proxy qrels .* drift"):
                self.run_gate_fixture(fixture)

    def test_gate_rejects_manifest_hash_drift(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            fixture = GateFixture(Path(td)).build()
            manifest = Path(json.loads(fixture.policy_path.read_text())["proxy_qrels_manifest"]["path"])
            payload = json.loads(manifest.read_text())
            payload["extra"] = True
            write_json(manifest, payload)

            with self.assertRaisesRegex(preflight.PreflightError, "manifest hash drift"):
                self.run_gate_fixture(fixture)

    def test_gate_rejects_manifest_source_inputs_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            fixture = GateFixture(Path(td)).build()
            policy = json.loads(fixture.policy_path.read_text(encoding="utf-8"))
            manifest_path = Path(policy["proxy_qrels_manifest"]["path"])
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["source_inputs"]["exclusion_qid_json"][0]["qid_counts"]["fiqa"] = 99
            write_json(manifest_path, manifest)
            policy["proxy_qrels_manifest"]["sha256"] = preflight.sha256_file(manifest_path)
            policy["proxy_qrels_manifest"]["bytes"] = manifest_path.stat().st_size
            write_json(fixture.policy_path, policy)
            write_json(fixture.root / "proxy-eval/q3-r4-proxy-execution-plan-audit.json", {"policy_sha256": preflight.sha256_file(fixture.policy_path)})

            with self.assertRaisesRegex(preflight.PreflightError, "manifest source_inputs mismatch"):
                self.run_gate_fixture(fixture)

    def test_gate_rejects_exclusion_source_hash_content_or_qid_drift(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            fixture = GateFixture(Path(td)).build()
            write_json(fixture.exclusion_paths[0], {"split": "dev4", "selected_qids": {"fiqa": ["Q-fiqa"], "nfcorpus": [], "scifact": []}})

            with self.assertRaisesRegex(preflight.PreflightError, "exclusion source .* drift"):
                self.run_gate_fixture(fixture)

    def test_gate_rejects_heldout_qrels_hash_row_gain_or_qid_drift(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            fixture = GateFixture(Path(td)).build()
            write_qrels(fixture.heldout_paths["fiqa"], [("Q-fiqa", "D9", 2)])

            with self.assertRaisesRegex(preflight.PreflightError, "heldout qrels .* drift"):
                self.run_gate_fixture(fixture)

    def test_gate_rejects_missing_or_duplicate_pinned_leakage_sources(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            fixture = GateFixture(Path(td)).build()
            fixture.exclusion_paths[0].unlink()

            with self.assertRaisesRegex(preflight.PreflightError, "missing pinned exclusion source"):
                self.run_gate_fixture(fixture)

        with tempfile.TemporaryDirectory() as td:
            fixture = GateFixture(Path(td)).build()
            policy = json.loads(fixture.policy_path.read_text(encoding="utf-8"))
            policy["source_inputs"]["exclusion_qid_json"].append(policy["source_inputs"]["exclusion_qid_json"][0])
            fixture.rewrite_policy_manifest_and_plan(policy)

            with self.assertRaisesRegex(preflight.PreflightError, "pinned exclusion source count mismatch|duplicate pinned exclusion"):
                self.run_gate_fixture(fixture)

    def test_gate_rejects_aggregate_seed_surface_topk_qrels_or_count_drift(self) -> None:
        cases = [
            ("config", ("config", "quantizer_seed"), 7, "bits/seed mismatch"),
            ("config", ("config", "top_k"), 10, "top_k mismatch"),
            ("inputs", ("inputs", "qrels_path"), "other.tsv", "qrels path mismatch"),
            ("inputs", ("inputs", "queries"), 99, "query count mismatch"),
            ("inputs", ("inputs", "relevant_pairs"), 99, "qrel count mismatch"),
            ("surface", ("rows", 0, "scoring_surface"), "other", "scoring surface mismatch"),
        ]
        for _name, path_keys, value, pattern in cases:
            with self.subTest(pattern=pattern), tempfile.TemporaryDirectory() as td:
                fixture = GateFixture(Path(td)).build()
                metrics_path = Path(fixture.outputs["anchor"]["fiqa"]["metrics_json"])
                payload = json.loads(metrics_path.read_text())
                target = payload
                for key in path_keys[:-1]:
                    target = target[key]
                target[path_keys[-1]] = value
                write_json(metrics_path, payload)

                with self.assertRaisesRegex(preflight.PreflightError, pattern):
                    self.run_gate_fixture(fixture)

    def test_gate_rejects_asymmetric_per_query_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            fixture = GateFixture(Path(td)).build()
            write_jsonl(Path(fixture.outputs["candidate"]["fiqa"]["per_query_jsonl"]), [per_query("Q-other", [1, 1], dataset="fiqa")])

            with self.assertRaisesRegex(preflight.PreflightError, "per-query qid set mismatch"):
                self.run_gate_fixture(fixture)

    def test_gate_rejects_nonfinite_metric(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            fixture = GateFixture(Path(td)).build()
            metrics_path = Path(fixture.outputs["candidate"]["fiqa"]["metrics_json"])
            payload = json.loads(metrics_path.read_text())
            payload["rows"][0]["quality"]["ndcg_at_10"] = float("nan")
            metrics_path.write_text(json.dumps(payload, allow_nan=True) + "\n", encoding="utf-8")

            with self.assertRaisesRegex(preflight.PreflightError, "nonfinite metric"):
                self.run_gate_fixture(fixture)

    def test_gate_rejects_macro_or_domain_threshold_failures(self) -> None:
        cases = [
            ("nfcorpus", 0.505, 1.0, "nfcorpus"),
            ("fiqa", 0.499, 1.0, "fiqa"),
            ("scifact", 0.499, 1.0, "scifact"),
            ("nfcorpus", 0.51, 0.99, "nfcorpus"),
        ]
        for dataset, ndcg, recall, failure in cases:
            with self.subTest(dataset=dataset, ndcg=ndcg, recall=recall), tempfile.TemporaryDirectory() as td:
                fixture = GateFixture(Path(td)).build()
                fixture.write_result("candidate", dataset, ndcg, recall)

                summary = self.run_gate_fixture(fixture)

                self.assertFalse(summary["gate_pass"])
                self.assertIn(failure, summary["failures"])

    def test_substitution_audit_deterministic_multi_lost_gain_matching(self) -> None:
        anchor = per_query("Q1", [2, 1, 0, 0, 0, 0, 0, 0, 0, 0], ndcg=0.8, recall10=1.0)
        candidate = per_query("Q1", [1, 2, 0, 0, 0, 0, 0, 0, 0, 0], ndcg=0.8, recall10=1.0)
        anchor["top_k"][0]["doc_id"] = "A2"
        anchor["top_k"][1]["doc_id"] = "A1"
        candidate["top_k"][0]["doc_id"] = "B1"
        candidate["top_k"][1]["doc_id"] = "B2"

        audit = preflight.substitution_audit(anchor, candidate)

        self.assertTrue(audit["matching_ok"])
        self.assertEqual(audit["matches"][0]["lost"], ("A2", 2))
        self.assertEqual(audit["matches"][0]["entered"], ("B2", 2))

    def test_gate_rejects_top10_gain_ndcg_or_recall_loss(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            fixture = GateFixture(Path(td)).build()
            row = per_query("Q-fiqa", [0, 1], dataset="fiqa", ndcg=0.5001, recall10=0.5, recall100=1.0)
            row["top_k"][0]["doc_id"] = "D3"
            write_jsonl(Path(fixture.outputs["candidate"]["fiqa"]["per_query_jsonl"]), [row])

            summary = self.run_gate_fixture(fixture)

            self.assertFalse(summary["gate_pass"])
            self.assertIn("fiqa", summary["failures"])

    def test_gate_rejects_worst_perquery_ndcg_and_recall100_diagnostics(self) -> None:
        cases = [
            ("ndcg", 0.48, 1.0, "fiqa"),
            ("recall-loss-docs", 0.5001, 0.0, "fiqa"),
            ("recall-loss-delta", 0.5001, 0.96, "fiqa"),
        ]
        for _name, ndcg, recall100, failure in cases:
            with self.subTest(case=_name), tempfile.TemporaryDirectory() as td:
                fixture = GateFixture(Path(td)).build()
                write_jsonl(Path(fixture.outputs["candidate"]["fiqa"]["per_query_jsonl"]), [
                    per_query("Q-fiqa", [1, 1], dataset="fiqa", ndcg=ndcg, recall10=1.0, recall100=recall100)
                ])

                summary = self.run_gate_fixture(fixture)

                self.assertFalse(summary["gate_pass"])
                self.assertIn(failure, summary["failures"])

    def test_gate_rejects_anchor_candidate_config_asymmetry(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            fixture = GateFixture(Path(td)).build()
            metrics_path = Path(fixture.outputs["candidate"]["fiqa"]["metrics_json"])
            payload = json.loads(metrics_path.read_text())
            payload["inputs"]["documents"] = 123
            write_json(metrics_path, payload)

            with self.assertRaisesRegex(preflight.PreflightError, "anchor/candidate inputs asymmetry"):
                self.run_gate_fixture(fixture)

    def test_gate_rejects_tsv_malformed_empty_duplicate_or_extra_rows(self) -> None:
        cases = [
            ("malformed", "fixture\n"),
            ("empty", ""),
            ("duplicate", None),
            ("extra", None),
        ]
        for name, content in cases:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as td:
                fixture = GateFixture(Path(td)).build()
                path = Path(fixture.outputs["anchor"]["fiqa"]["metrics_tsv"])
                if content is not None:
                    path.write_text(content, encoding="utf-8")
                else:
                    rows = path.read_text(encoding="utf-8").splitlines()
                    path.write_text("\n".join(rows + [rows[-1]]) + "\n", encoding="utf-8")

                with self.assertRaisesRegex(preflight.PreflightError, "TSV schema/header mismatch|TSV expected dense\\+q3 rows"):
                    self.run_gate_fixture(fixture)

    def test_gate_rejects_tsv_wrong_dataset_method_bits_or_nonfinite(self) -> None:
        cases = [
            ("dataset", {"dataset": "other"}, "dataset mismatch"),
            ("method", {"method": "turboquant_ip_b5"}, "missing/duplicate expected dense or q3 row"),
            ("bits", {"bits": "5"}, "missing/duplicate expected dense or q3 row"),
            ("nonfinite", {"ndcg_at_10": "nan"}, "nonfinite or nonnumeric TSV metric"),
        ]
        for name, updates, pattern in cases:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as td:
                fixture = GateFixture(Path(td)).build()
                path = Path(fixture.outputs["anchor"]["fiqa"]["metrics_tsv"])
                with path.open("r", encoding="utf-8", newline="") as handle:
                    reader = csv.DictReader(handle, delimiter="\t")
                    rows = list(reader)
                    fieldnames = reader.fieldnames or TSV_HEADER
                rows[1].update(updates)
                with path.open("w", encoding="utf-8", newline="") as handle:
                    writer = csv.DictWriter(handle, delimiter="\t", fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(rows)

                with self.assertRaisesRegex(preflight.PreflightError, pattern):
                    self.run_gate_fixture(fixture)

    def test_gate_rejects_tsv_json_metric_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            fixture = GateFixture(Path(td)).build()
            write_metrics_tsv(Path(fixture.outputs["candidate"]["fiqa"]["metrics_tsv"]), "fiqa", 0.6, 1.0)

            with self.assertRaisesRegex(preflight.PreflightError, "TSV-vs-JSON ndcg_at_10 mismatch"):
                self.run_gate_fixture(fixture)


if __name__ == "__main__":
    unittest.main()
