#!/usr/bin/env python3
"""Contract checks for the default embedder serving energy benchmark."""

from __future__ import annotations

import csv
import json
import re
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


def _function_body(source: str, name: str) -> str:
    match = re.search(rf"\bfunc\s+{re.escape(name)}\s*\(", source)
    if not match:
        raise AssertionError(f"function not found: {name}")
    start = source.find("{", match.end())
    if start < 0:
        raise AssertionError(f"function body not found: {name}")
    depth = 0
    for index in range(start, len(source)):
        char = source[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return source[start + 1 : index]
    raise AssertionError(f"unterminated function body: {name}")


def _assert_ordered(testcase: unittest.TestCase, haystack: str, needles: list[str]) -> None:
    cursor = 0
    for needle in needles:
        found = haystack.find(needle, cursor)
        testcase.assertNotEqual(found, -1, f"missing or out-of-order token: {needle!r}")
        cursor = found + len(needle)


class EmbedderServingEnergyContractTest(unittest.TestCase):
    def test_script_declares_auditable_artifact_contract(self) -> None:
        source = SCRIPT.read_text(encoding="utf-8")

        self.assertIn('const schema = "eos.default_embedder_serving_energy.v1"', source)
        self.assertIn("WorkloadMode", source)
        self.assertIn("CommandProvenance", source)
        self.assertIn("EosBinarySHA256", source)
        self.assertIn("BuildCommand", source)
        self.assertIn("EOS_ENERGY_BENCH_WORKLOAD_MODE", source)
        self.assertIn("power-samples.jsonl", source)
        self.assertIn("summary.tsv", source)
        self.assertIn("manifest.json", source)
        self.assertIn("EOS_ENERGY_BENCH_REQUIRE_POWER", source)
        self.assertIn("EOS_ENERGY_REQUIRE_DEVICE_ENCODER", source)
        self.assertIn("EOS_ENERGY_BENCH_FAKE_POWER_WATTS", source)
        self.assertIn('"unsupported"', source)
        self.assertIn('"indeterminate"', source)

    def test_script_separates_encoder_from_index_scoring(self) -> None:
        source = SCRIPT.read_text(encoding="utf-8")

        self.assertIn('"export-retrieval-vectors"', source)
        self.assertIn('"eval-retrieval-turboquant"', source)
        self.assertIn('"export-pretrained-bert-retrieval-vectors"', source)
        self.assertIn('"eval-retrieval-vectors-turboquant"', source)
        self.assertIn('Phase:                       "encoder_only"', source)
        self.assertIn('Phase:                "index_scoring"', source)
        self.assertIn("encoderWorkloadItems := encoderDocs + encoderQueries", source)
        self.assertIn("EnergyJoulesPerWorkloadItem: energyCost(encoderEnergy, encoderWorkloadItems)", source)
        self.assertIn("EnergyJoulesPerQuery: energyCost(scoringEnergy, scoringQueries)", source)
        self.assertNotIn("EnergyJoulesPerQuery: encoder", source)

    def test_measured_phases_use_run_local_binary_not_go_run(self) -> None:
        source = SCRIPT.read_text(encoding="utf-8")

        self.assertIn('buildArgs := []string{"go", "build", "-o", eosBin, "./cmd/eos"}', source)
        self.assertIn('EosBinary:         eosBin', source)
        self.assertIn('EosBinarySHA256:   eosBinarySHA256', source)
        self.assertIn('BuildCommand:      buildCommand', source)
        self.assertIn('CommandProvenance: []commandRecord{buildCommand, encoderCommand, scoringCommand}', source)
        self.assertNotIn('"go", "run", "./cmd/eos", "export-retrieval-vectors"', source)
        self.assertNotIn('"go", "run", "./cmd/eos", "export-pretrained-bert-retrieval-vectors"', source)
        self.assertNotIn('"go", "run", "./cmd/eos", "eval-retrieval-turboquant"', source)
        self.assertNotIn('"go", "run", "./cmd/eos", "eval-retrieval-vectors-turboquant"', source)

    def test_workload_mode_validation_is_strict_and_default_native(self) -> None:
        source = SCRIPT.read_text(encoding="utf-8")

        self.assertIn('return "native", nil', source)
        self.assertIn('case "native":', source)
        self.assertIn('case "imported_bert":', source)
        self.assertIn('EOS_ENERGY_BENCH_WORKLOAD_MODE must be native or imported_bert', source)

    def test_native_command_shape_stays_current_artifact_path(self) -> None:
        source = SCRIPT.read_text(encoding="utf-8")

        self.assertIn('eosBin, "export-retrieval-vectors"', source)
        self.assertIn('}, "eos export-retrieval-vectors", nil', source)
        self.assertIn('[]string{eosBin, "eval-retrieval-turboquant"}', source)
        self.assertIn('args = append(args, "--batch-size", batchSize)', source)
        self.assertIn('args = append(args, artifact, datasetDir)', source)
        self.assertIn('return args, "eos eval-retrieval-turboquant", nil', source)

    def test_imported_bert_command_shape_uses_package_role_contract_and_cache_eval(self) -> None:
        source = SCRIPT.read_text(encoding="utf-8")

        self.assertIn('eosBin, "export-pretrained-bert-retrieval-vectors"', source)
        self.assertIn('"--package", artifact', source)
        self.assertIn('"--use-package-role-contract"', source)
        self.assertIn('getenvBool("EOS_ENERGY_REQUIRE_DEVICE_ENCODER", false)', source)
        self.assertIn('args = append(args, "--require-device-encoder")', source)
        self.assertIn('return args, "eos export-pretrained-bert-retrieval-vectors", nil', source)
        self.assertIn('[]string{eosBin, "eval-retrieval-vectors-turboquant"}', source)
        self.assertIn('"--doc-vectors", filepath.Join(vectorDir, "doc-vectors.jsonl")', source)
        self.assertIn('"--query-vectors", filepath.Join(vectorDir, "query-vectors.jsonl")', source)
        self.assertIn('return args, "eos eval-retrieval-vectors-turboquant", nil', source)

    def test_imported_bert_generated_argv_order_is_structural(self) -> None:
        source = SCRIPT.read_text(encoding="utf-8")
        encoder_body = _function_body(source, "buildEncoderArgs")
        scoring_body = _function_body(source, "buildScoringArgs")

        _assert_ordered(
            self,
            encoder_body,
            [
                'if mode == "imported_bert"',
                'eosBin, "export-pretrained-bert-retrieval-vectors"',
                '"--package", artifact',
                '"--use-package-role-contract"',
                '"--dataset", dataset',
                '"--split", split',
                '"--batch-size", batchSize',
                '"--max-docs", maxDocs',
                '"--max-queries", maxQueries',
                '"--manifest-json", vectorManifest',
                'if getenvBool("EOS_ENERGY_REQUIRE_DEVICE_ENCODER", false)',
                'args = append(args, "--require-device-encoder")',
                "args = append(args, datasetDir, vectorDir)",
                'return args, "eos export-pretrained-bert-retrieval-vectors", nil',
            ],
        )
        _assert_ordered(
            self,
            scoring_body,
            [
                'if mode == "imported_bert"',
                'args := []string{eosBin, "eval-retrieval-vectors-turboquant"}',
                "args = append(args, commonFlags...)",
                "args = append(args,",
                '"--backend", "imported-pretrained-bert-turboquant"',
                '"--artifact", artifact',
                '"--doc-vectors", filepath.Join(vectorDir, "doc-vectors.jsonl")',
                '"--query-vectors", filepath.Join(vectorDir, "query-vectors.jsonl")',
                "datasetDir",
                'return args, "eos eval-retrieval-vectors-turboquant", nil',
            ],
        )

        self.assertRegex(
            scoring_body,
            r'"--query-vectors", filepath\.Join\(vectorDir, "query-vectors\.jsonl"\),\s+datasetDir,\s+\)\s+'
            r'return args, "eos eval-retrieval-vectors-turboquant", nil',
        )

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
            "workload_mode": "imported_bert",
            "eos_binary": "runs/example/bin/eos",
            "eos_binary_sha256": "abc123",
            "build_command": {"label": "build-eos", "args": ["go", "build", "-o", "runs/example/bin/eos", "./cmd/eos"]},
            "command_provenance": [
                {"label": "build-eos", "args": ["go", "build", "-o", "runs/example/bin/eos", "./cmd/eos"]},
                {"label": "encoder-only", "args": ["eos", "export-pretrained-bert-retrieval-vectors"]},
                {"label": "index-scoring", "args": ["eos", "eval-retrieval-vectors-turboquant"]},
            ],
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
        self.assertEqual(decoded["workload_mode"], "imported_bert")
        self.assertEqual(decoded["build_command"]["label"], "build-eos")
        self.assertEqual(decoded["eos_binary"], "runs/example/bin/eos")
        self.assertEqual(decoded["eos_binary_sha256"], "abc123")
        self.assertEqual(len(decoded["command_provenance"]), 3)
        self.assertEqual(decoded["overall_status"], "partial")
        self.assertIn("unsupported_reason", decoded)
        self.assertEqual(
            {phase["phase"]: phase["telemetry_status"] for phase in decoded["phases"]},
            {"encoder_only": "measured", "index_scoring": "unsupported"},
        )

    def test_docs_reference_energy_gate_and_unsupported_status(self) -> None:
        docs = BENCHMARKS.read_text(encoding="utf-8") + PRODUCTION.read_text(encoding="utf-8")

        self.assertIn("scripts/bench_eos_default_embedder_serving_energy.fw", docs)
        self.assertIn("EOS_ENERGY_BENCH_WORKLOAD_MODE=imported_bert", docs)
        self.assertIn("EOS_ENERGY_REQUIRE_DEVICE_ENCODER=1", docs)
        self.assertIn("power-samples.jsonl", docs)
        self.assertIn("encoder-only", docs)
        self.assertIn("index/scoring", docs)
        self.assertIn("documents + queries", docs)
        self.assertIn("energy_joules_per_query", docs)
        self.assertIn("unsupported", docs)


if __name__ == "__main__":
    unittest.main()
