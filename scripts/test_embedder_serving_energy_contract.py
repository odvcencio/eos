#!/usr/bin/env python3
"""Contract checks for the default embedder serving energy benchmark."""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "bench_eos_default_embedder_serving_energy.fw"
BENCHMARKS = REPO_ROOT / "docs" / "benchmarks.md"
PRODUCTION = REPO_ROOT / "docs" / "production-embedding.md"


SUMMARY_COLUMNS = [
    "phase",
    "workload",
    "telemetry_status",
    "telemetry_source",
    "gpu_index",
    "wall_seconds",
    "query_count",
    "document_count",
    "workload_item_count",
    "sample_count",
    "average_power_watts",
    "energy_joules",
    "energy_joules_per_query",
    "energy_joules_per_workload_item",
    "output_json",
    "output_tsv",
]


class EmbedderServingEnergyContractTest(unittest.TestCase):
    def test_script_declares_auditable_artifact_contract(self) -> None:
        source = SCRIPT.read_text(encoding="utf-8")

        self.assertIn('const schema = "eos.default_embedder_serving_energy.v1"', source)
        self.assertIn("power-samples.jsonl", source)
        self.assertIn("summary.tsv", source)
        self.assertIn("manifest.json", source)
        self.assertIn("EOS_ENERGY_BENCH_REQUIRE_POWER", source)
        self.assertIn("EOS_ENERGY_BENCH_FAKE_POWER_WATTS", source)
        self.assertIn('"unsupported"', source)
        self.assertIn('"indeterminate"', source)

    def test_script_separates_encoder_from_index_scoring(self) -> None:
        source = SCRIPT.read_text(encoding="utf-8")

        self.assertIn('"export-retrieval-vectors"', source)
        self.assertIn('"eval-retrieval-turboquant"', source)
        self.assertIn('Phase:                       "encoder_only"', source)
        self.assertIn('Phase:                "index_scoring"', source)
        self.assertIn("encoderWorkloadItems := encoderDocs + encoderQueries", source)
        self.assertIn("EnergyJoulesPerWorkloadItem: energyCost(encoderEnergy, encoderWorkloadItems)", source)
        self.assertIn("EnergyJoulesPerQuery: energyCost(scoringEnergy, scoringQueries)", source)
        self.assertNotIn("EnergyJoulesPerQuery: encoder", source)

    def test_summary_tsv_contract_accepts_measured_and_unsupported_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "summary.tsv"
            rows = [
                {
                    "phase": "encoder_only",
                    "workload": "eos export-retrieval-vectors",
                    "telemetry_status": "measured",
                    "telemetry_source": "nvidia-smi",
                    "gpu_index": "0",
                    "wall_seconds": "2.000000",
                    "query_count": "20",
                    "document_count": "300",
                    "workload_item_count": "320",
                    "sample_count": "11",
                    "average_power_watts": "55.000000",
                    "energy_joules": "110.000000",
                    "energy_joules_per_query": "",
                    "energy_joules_per_workload_item": "0.343750",
                    "output_json": "encoder-vectors.manifest.json",
                    "output_tsv": "",
                },
                {
                    "phase": "index_scoring",
                    "workload": "eos eval-retrieval-turboquant",
                    "telemetry_status": "unsupported",
                    "telemetry_source": "nvidia-smi",
                    "gpu_index": "0",
                    "wall_seconds": "1.000000",
                    "query_count": "20",
                    "document_count": "0",
                    "workload_item_count": "20",
                    "sample_count": "0",
                    "average_power_watts": "0.000000",
                    "energy_joules": "0.000000",
                    "energy_joules_per_query": "",
                    "energy_joules_per_workload_item": "",
                    "output_json": "scifact.turboquant-serving.metrics.json",
                    "output_tsv": "scifact.turboquant-serving.metrics.tsv",
                },
            ]
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=SUMMARY_COLUMNS, delimiter="\t")
                writer.writeheader()
                writer.writerows(rows)

            with path.open("r", encoding="utf-8", newline="") as handle:
                loaded = list(csv.DictReader(handle, delimiter="\t"))

        self.assertEqual(loaded[0]["phase"], "encoder_only")
        self.assertEqual(loaded[1]["phase"], "index_scoring")
        self.assertEqual(loaded[0]["telemetry_status"], "measured")
        self.assertEqual(loaded[1]["telemetry_status"], "unsupported")
        self.assertEqual(list(loaded[0].keys()), SUMMARY_COLUMNS)
        self.assertEqual(loaded[0]["energy_joules_per_query"], "")
        self.assertEqual(loaded[0]["energy_joules_per_workload_item"], "0.343750")

    def test_manifest_contract_records_partial_or_unsupported_power_status(self) -> None:
        manifest = {
            "schema": "eos.default_embedder_serving_energy.v1",
            "overall_status": "partial",
            "unsupported_reason": "index_scoring: fewer than two usable GPU power samples",
            "phases": [
                {"phase": "encoder_only", "telemetry_status": "measured"},
                {"phase": "index_scoring", "telemetry_status": "unsupported"},
            ],
        }
        encoded = json.dumps(manifest)
        decoded = json.loads(encoded)

        self.assertEqual(decoded["schema"], "eos.default_embedder_serving_energy.v1")
        self.assertEqual(decoded["overall_status"], "partial")
        self.assertIn("unsupported_reason", decoded)
        self.assertEqual(
            {phase["phase"]: phase["telemetry_status"] for phase in decoded["phases"]},
            {"encoder_only": "measured", "index_scoring": "unsupported"},
        )

    def test_docs_reference_energy_gate_and_unsupported_status(self) -> None:
        docs = BENCHMARKS.read_text(encoding="utf-8") + PRODUCTION.read_text(encoding="utf-8")

        self.assertIn("scripts/bench_eos_default_embedder_serving_energy.fw", docs)
        self.assertIn("power-samples.jsonl", docs)
        self.assertIn("encoder-only", docs)
        self.assertIn("index/scoring", docs)
        self.assertIn("documents + queries", docs)
        self.assertIn("energy_joules_per_query", docs)
        self.assertIn("unsupported", docs)


if __name__ == "__main__":
    unittest.main()
