#!/usr/bin/env python3
"""Focused tests for q3 R6 infraretry1 post-train attestation helpers."""

from __future__ import annotations

import copy
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import q3_r6_infraretry1_posttrain_attestation as att  # noqa: E402


EXPECTED_SPAN_FUNCTIONS = {
    Path("runtime/embedding_train_runner.go"): [
        "validateTurboQuantTopKScoreSpectrumOnlyRunConfig",
        "clearScoreSpectrumIncompatibleRunConfig",
        "applyScoreSpectrumRunOverrides",
    ],
    Path("runtime/embedding_trainer.go"): [
        "accumulateTurboQuantPreparedIPTopKScoreSpectrumRowGrads",
    ],
    Path("runtime/embedding_train_manifest.go"): [
        "mllValues",
        "scoreSpectrumPolicyFromAuthoredDoc",
    ],
    Path("runtime/package_manifest.go"): [
        "encodePackageManifestMLL",
        "decodePackageManifestMLL",
    ],
}


def write(path: Path, data: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(data, encoding="utf-8")
    return path


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class PosttrainAttestationTest(unittest.TestCase):
    def patch_path(self, name: str, value) -> None:
        self.addCleanup(setattr, att, name, getattr(att, name))
        setattr(att, name, value)

    def patch_expected(self, updates: dict[str, str]) -> None:
        original = att.EXPECTED_HASHES
        self.addCleanup(setattr, att, "EXPECTED_HASHES", original)
        next_expected = copy.deepcopy(original)
        next_expected.update(updates)
        att.EXPECTED_HASHES = next_expected

    def test_package_sibling_rollup_matches_launcher_compact_json_formula(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write(root / "d384-pre.mll", "root\n")
            write(root / "d384-pre.package.mll", "package\n")
            rollup, siblings = att.package_sibling_rollup(root)
            h = hashlib.sha256()
            for item in siblings:
                h.update(json.dumps({"bytes": item["bytes"], "name": item["name"], "sha256": item["sha256"]}, separators=(",", ":"), sort_keys=True).encode("utf-8"))
                h.update(b"\n")
            self.assertEqual(rollup, h.hexdigest())

    def test_base_leakage_audit_requires_explicit_zero_base_and_aux_active(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.patch_path("REPO_ROOT", root)
            train = root / "train.jsonl"
            rows = []
            for i in range(1155):
                rows.append(
                    json.dumps(
                        {
                            "row_id": f"r{i}",
                            "selection_bucket": "frontier_context_q3",
                            "base_loss_weight": 0.0,
                            "hard_loss_weight": 0.0,
                            "soft_loss_weight": 0.0,
                            "recovery_loss_weight": 0.0,
                            "turboquant_topk_loss_weight": 1.0,
                            "turboquant_topk_recall_loss_weight": 0.0,
                        },
                        sort_keys=True,
                    )
                )
            write(train, "\n".join(rows) + "\n")
            metrics = write(root / "metrics.json", "{}\n")
            pkg = root / "pkg"
            write(pkg / "d384-pre.mll", "root\n")
            write(pkg / "d384-pre.package.mll", "package\n")
            rollup, _ = att.package_sibling_rollup(pkg)
            self.patch_path("TRAIN_JSONL", train)
            self.patch_path("METRICS_JSON", metrics)
            self.patch_path("CANDIDATE_ROOT_ARTIFACT", pkg / "d384-pre.mll")
            self.patch_path("CANDIDATE_PACKAGE_DIR", pkg)
            self.patch_path("BASE_LEAKAGE_AUDIT_JSON", root / "base-audit.json")
            self.patch_expected({"train_jsonl": sha(train), "metrics_json": sha(metrics), "posttrain_rollup": rollup})
            with mock.patch.object(att, "source_binding_records", return_value=[]):
                payload = att.build_base_leakage_audit()
            self.assertEqual(payload["runtime_semantics"]["base_hard_soft_recovery_work_units"], 0)
            self.assertTrue(payload["runtime_semantics"]["aux_only"])

            bad = json.loads(rows[0])
            bad["base_loss_weight"] = 0.1
            write(train, json.dumps(bad) + "\n" + "\n".join(rows[1:]) + "\n")
            self.patch_expected({"train_jsonl": sha(train), "metrics_json": sha(metrics), "posttrain_rollup": rollup})
            with mock.patch.object(att, "source_binding_records", return_value=[]):
                with self.assertRaisesRegex(att.AttestationError, "non-aux-only"):
                    att.build_base_leakage_audit()

    def test_validate_metrics_requires_topk_semantics_and_empty_generic_prefix_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.patch_path("REPO_ROOT", root)
            metrics = root / "metrics.json"
            payload = {
                "schema": "manta.embedding_train_metrics.v1",
                "config": {
                    "turboquant_prefix_seed": 5581486560434873699,
                    "turboquant_topk_objectives": [{"dim": 384, "bit_width": 3, "weight": 0.04}],
                    "turboquant_topk_loss": "lambdandcg",
                    "turboquant_topk_cutoff": 10,
                    "turboquant_topk_tau": 0.05,
                    "turboquant_topk_margin": 0.002,
                    "turboquant_topk_negative_mask": "q3",
                    "turboquant_topk_recall_weight": 0.50,
                    "turboquant_topk_recall_cutoff": 100,
                    "turboquant_topk_recall_tau": 0.05,
                    "turboquant_topk_recall_negative_mask": "q3",
                },
                "workload": {
                    "train_examples": 1155,
                    "completed_epochs": 2,
                    "train_batches_per_epoch": 289,
                    "planned_train_pairs": 54162,
                    "actual_train_pairs": 54162,
                    "actual_eval_pairs": 0,
                },
            }
            write(metrics, json.dumps(payload) + "\n")
            self.patch_path("METRICS_JSON", metrics)
            self.assertTrue(att.validate_metrics()["checks"]["generic_prefix_score_mode_absent_or_empty"])
            payload["config"]["turboquant_prefix_score_mode"] = "prepared_ip"
            write(metrics, json.dumps(payload) + "\n")
            with self.assertRaisesRegex(att.AttestationError, "generic_prefix_score_mode_absent_or_empty"):
                att.validate_metrics()

    def test_extracted_source_binding_spans_are_file_relative(self) -> None:
        def expected_span(path: Path, name: str) -> tuple[int, int, str]:
            text = (att.REPO_ROOT / path).read_text(encoding="utf-8")
            match = __import__("re").search(rf"func (?:\([^)]*\) )?{__import__('re').escape(name)}\(", text)
            self.assertIsNotNone(match, name)
            start = match.start()
            brace = text.find("{", match.end())
            depth = 0
            end = None
            for idx in range(brace, len(text)):
                if text[idx] == "{":
                    depth += 1
                elif text[idx] == "}":
                    depth -= 1
                    if depth == 0:
                        end = idx + 1
                        break
            self.assertIsNotNone(end, name)
            source = text[start:end]
            start_line = text[:start].count("\n") + 1
            end_line = start_line + source.count("\n")
            return start_line, end_line, hashlib.sha256(source.encode("utf-8")).hexdigest()

        for path, names in EXPECTED_SPAN_FUNCTIONS.items():
            for name in names:
                with self.subTest(path=str(path), name=name):
                    actual = att.extract_function_source(path, name)
                    start, end, body_sha = expected_span(path, name)
                    self.assertEqual(actual["line_start"], start)
                    self.assertEqual(actual["line_end"], end)
                    self.assertEqual(actual["source_sha256"], body_sha)
                    self.assertGreater(actual["line_start"], 1)


if __name__ == "__main__":
    unittest.main()
