#!/usr/bin/env python3
"""Dependency-free tests for the EOS embedder deployment envelope summarizer."""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

import summarize_eos_embedder_deployment_envelope as env


def write_json(path: Path, data: dict[str, object]) -> Path:
    path.write_text(json.dumps(data, sort_keys=True) + "\n", encoding="utf-8")
    return path


def valid_turboquant_metrics() -> dict[str, object]:
    return {
        "schema": env.TURBOQUANT_RETRIEVAL_SCHEMA,
        "dataset": "scifact",
        "dense": {"vector_bytes": 15360, "scores_per_second": 100.0},
        "rows": [
            {
                "bits": 4,
                "method": "turboquant_ip_b4",
                "turboquant_version": "v0.1.0",
                "codebook_version": "v0.1.0",
                "quality": {"ndcg_at_10": 0.53, "recall_at_100": 0.74},
                "ndcg_at_10_delta": -0.02,
                "recall_at_100_delta": -0.01,
                "vector_bytes": 5600,
                "dense_vector_bytes": 15360,
                "compression_ratio": 2.742857,
                "total_vector_bytes": 5600,
                "total_compression_ratio": 2.742857,
                "scores_per_second": 120.0,
                "query_latency": {"p95_ms": 3.0},
            }
        ],
        "config": {"quantizer_seed": 123},
    }


class DeploymentEnvelopeTest(unittest.TestCase):
    def test_fixture_summarizes_measured_projected_missing_and_blocked_states(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifact = root / "model.mll"
            artifact.write_bytes(b"sealed-model")
            tokenizer = root / "tokenizer.mll"
            tokenizer.write_bytes(b"tok")
            leaderboard = root / "leaderboard.tsv"
            leaderboard.write_text(
                "\n".join(
                    [
                        "dataset\tbackend\tdocuments\tqueries\tndcg_at_10\tmrr_at_10\trecall_at_10\trecall_at_100\tscores_per_second",
                        "scifact\tbm25\t10\t2\t0.60\t0.50\t0.40\t0.80\t1000",
                        "scifact\tcuda\t10\t2\t0.55\t0.45\t0.35\t0.75\t2000",
                        "fiqa\tbm25\t20\t3\t0.20\t0.30\t0.10\t0.40\t3000",
                        "fiqa\tcuda\t20\t3\t0.10\t0.20\t0.05\t0.30\t4000",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            legal = write_json(
                root / "manifest.json",
                {
                    "schema": "fixture",
                    "legal_gates": {
                        "commercial_use_allowed": False,
                        "release_train_allowed": False,
                        "train_allowed_for_research": True,
                    },
                },
            )
            tq = write_json(root / "tq.json", valid_turboquant_metrics())
            summary = env.build_summary(
                artifact=artifact,
                tokenizer=tokenizer,
                dimension=384,
                leaderboard_tsv=leaderboard,
                turboquant_metrics_json=[tq],
                legal_manifest_json=legal,
                corpus_sizes=[1_000_000],
                bits=[1, 4],
                model_family="fixture-imported-384d",
                clock=lambda: "2026-09-01T00:00:00Z",
            )

        self.assertEqual(summary["schema"], env.SCHEMA)
        self.assertEqual(summary["artifact_footprint"]["files"]["artifact"]["state"], "measured")
        self.assertEqual(summary["artifact_footprint"]["files"]["weights"]["state"], "missing")
        self.assertEqual(summary["artifact_footprint"]["deployable_artifact_bytes"], len(b"sealed-model"))
        self.assertEqual(summary["artifact_footprint"]["supplied_files_total_bytes"], len(b"sealed-model") + len(b"tok"))
        self.assertEqual(summary["retrieval_leaderboard"]["macro"]["cuda"]["ndcg_at_10"], 0.325)
        self.assertEqual(summary["legal_release_gates"]["state"], "blocked")
        self.assertEqual(summary["summary"]["commercial_release_state"], "blocked")
        q1 = [
            row
            for row in summary["vector_payload_projections"]["rows"]
            if row["method"] == "turboquant_ip_b1" and row["corpus_vectors"] == 1_000_000
        ][0]
        self.assertEqual(q1["mse_bits"], 0)
        self.assertEqual(q1["sign_bits"], 384)
        self.assertEqual(q1["mse_bytes"], 0)
        self.assertEqual(q1["sign_bytes"], 48)
        self.assertEqual(q1["bytes_per_vector"], 56)
        q4 = [
            row
            for row in summary["vector_payload_projections"]["rows"]
            if row["method"] == "turboquant_ip_b4" and row["corpus_vectors"] == 1_000_000
        ][0]
        self.assertEqual(q4["payload_bytes"], 200_000_000)
        self.assertAlmostEqual(q4["compression_vs_dense_f32"], 7.68)
        self.assertEqual(summary["turboquant_metrics"]["rows"][0]["p95_ms"], 3.0)
        self.assertEqual(summary["turboquant_metrics"]["rows"][0]["turboquant_version"], "v0.1.0")
        self.assertEqual(summary["turboquant_metrics"]["rows"][0]["codebook_version"], "v0.1.0")
        self.assertEqual(summary["turboquant_metrics"]["rows"][0]["quantizer_seed"], 123)
        self.assertIn("turboquant_metrics", summary["summary"]["measured_evidence_sections"])
        self.assertIn("energy", summary["summary"]["missing_evidence_sections"])
        self.assertIn("throughput_gate", summary["summary"]["missing_evidence_sections"])

    def test_turboquant_projection_ceilings_are_separate_for_unaligned_dimensions(self) -> None:
        row = env.turboquant_ip_bytes_per_vector(257, 2)

        self.assertEqual(row["mse_bits"], 257)
        self.assertEqual(row["sign_bits"], 257)
        self.assertEqual(row["mse_bytes"], 33)
        self.assertEqual(row["sign_bytes"], 33)
        self.assertEqual(row["packed_bytes"], 66)
        self.assertEqual(row["bytes_per_vector"], 74)
        self.assertIn(5, env.DEFAULT_BITS)

    def test_optional_energy_and_throughput_evidence_are_validated_and_measured(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifact = root / "model.mll"
            artifact.write_bytes(b"abc")
            energy = write_json(
                root / "energy.json",
                {
                    "schema": env.ENERGY_SCHEMA,
                    "overall_status": "measured",
                    "workload_mode": "native",
                    "phases": [
                        {
                            "phase": "encoder_only",
                            "telemetry_status": "measured",
                            "telemetry_source": "fixture",
                            "wall_seconds": 1.0,
                            "query_count": 1,
                            "document_count": 2,
                            "workload_item_count": 3,
                            "energy_joules": 4.0,
                            "energy_joules_per_workload_item": 1.333,
                        }
                    ],
                },
            )
            throughput = write_json(
                root / "throughput.json",
                {
                    "schema": env.THROUGHPUT_GATE_SCHEMA,
                    "cold_load_ms": 10.0,
                    "first_query_encode_ms": 2.0,
                    "warm_batch64_docs_per_second": 64.0,
                    "peak_rss_mb": 128.0,
                },
            )

            summary = env.build_summary(
                artifact=artifact,
                dimension=256,
                energy_json=energy,
                throughput_gate_json=throughput,
                clock=lambda: "2026-09-01T00:00:00Z",
            )

        self.assertEqual(summary["serving_energy"]["state"], "measured")
        self.assertEqual(summary["startup_load_encode_throughput_gate"]["state"], "measured")
        self.assertIn("energy", summary["summary"]["measured_evidence_sections"])
        self.assertIn("throughput_gate", summary["summary"]["measured_evidence_sections"])

    def test_tsv_schema_contains_state_source_and_projection_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifact = root / "model.mll"
            artifact.write_bytes(b"abc")
            summary = env.build_summary(
                artifact=artifact,
                dimension=384,
                corpus_sizes=[1_000_000],
                bits=[4],
                clock=lambda: "2026-09-01T00:00:00Z",
            )
            rows = env.flatten_tsv(summary)
            tsv = root / "summary.tsv"
            env.write_tsv(tsv, rows)
            with tsv.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle, delimiter="\t")
                loaded = list(reader)

        self.assertEqual(reader.fieldnames, env.TSV_COLUMNS)
        self.assertTrue(any(row["section"] == "vector_payload_projection" and row["evidence_state"] == "projected" for row in loaded))
        self.assertTrue(any(row["section"] == "legal_release_gates" and row["evidence_state"] == "unknown" for row in loaded))
        self.assertTrue(any(row["section"] == "retrieval_leaderboard" and row["evidence_state"] == "missing" for row in loaded))

    def test_invalid_inputs_fail(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifact = root / "model.mll"
            artifact.write_bytes(b"abc")
            with self.assertRaises(env.EnvelopeError):
                env.build_summary(artifact=artifact, dimension=0)
            with self.assertRaises(env.EnvelopeError):
                env.parse_int_list("0", label="--corpus-size", minimum=1)
            with self.assertRaises(env.EnvelopeError):
                env.parse_int_list("9", label="--bits", minimum=1, maximum=8)
            with self.assertRaises(env.EnvelopeError):
                env.build_summary(artifact=root / "missing.mll", dimension=384)
            bad_tq = write_json(
                root / "bad-tq.json",
                {"schema": env.TURBOQUANT_RETRIEVAL_SCHEMA, "dense": {}, "rows": {}},
            )
            with self.assertRaises(env.EnvelopeError):
                env.build_summary(artifact=artifact, dimension=384, turboquant_metrics_json=[bad_tq])
            bad_energy = write_json(
                root / "bad-energy.json",
                {"schema": env.ENERGY_SCHEMA, "overall_status": "measured", "phases": {}},
            )
            with self.assertRaises(env.EnvelopeError):
                env.build_summary(artifact=artifact, dimension=384, energy_json=bad_energy)
            bad_throughput = write_json(
                root / "bad-throughput.json",
                {"schema": env.THROUGHPUT_GATE_SCHEMA, "cold_load_ms": 0},
            )
            with self.assertRaises(env.EnvelopeError):
                env.build_summary(artifact=artifact, dimension=384, throughput_gate_json=bad_throughput)

    def test_malformed_turboquant_optional_numeric_fields_fail(self) -> None:
        optional_fields = [
            ("total_vector_bytes", ("rows", 0, "total_vector_bytes")),
            ("total_compression_ratio", ("rows", 0, "total_compression_ratio")),
            ("ndcg_at_10", ("rows", 0, "quality", "ndcg_at_10")),
            ("ndcg_at_10_delta", ("rows", 0, "ndcg_at_10_delta")),
            ("recall_at_100", ("rows", 0, "quality", "recall_at_100")),
            ("recall_at_100_delta", ("rows", 0, "recall_at_100_delta")),
        ]
        for field_name, path_keys in optional_fields:
            with self.subTest(field_name=field_name), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                artifact = root / "model.mll"
                artifact.write_bytes(b"abc")
                data = valid_turboquant_metrics()
                target = data
                for key in path_keys[:-1]:
                    target = target[key]  # type: ignore[index]
                target[path_keys[-1]] = "not-a-number"  # type: ignore[index]
                bad_tq = write_json(root / "bad-tq.json", data)

                with self.assertRaises(env.EnvelopeError):
                    env.build_summary(artifact=artifact, dimension=384, turboquant_metrics_json=[bad_tq])

    def test_cli_writes_json_and_tsv(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifact = root / "model.mll"
            artifact.write_bytes(b"abc")
            out_json = root / "out" / "summary.json"
            out_tsv = root / "out" / "summary.tsv"
            rc = env.main(
                [
                    "--artifact",
                    str(artifact),
                    "--dimension",
                    "256",
                    "--corpus-size",
                    "1000",
                    "--bits",
                    "4",
                    "--model-family",
                    "native-eos-stage-b",
                    "--output-json",
                    str(out_json),
                    "--output-tsv",
                    str(out_tsv),
                ]
            )
            self.assertEqual(rc, 0)
            self.assertTrue(out_json.is_file())
            self.assertTrue(out_tsv.is_file())
            data = json.loads(out_json.read_text(encoding="utf-8"))
            self.assertEqual(data["identity"]["dimension"], 256)
            self.assertEqual(data["identity"]["model_family"], "native-eos-stage-b")


if __name__ == "__main__":
    unittest.main()
