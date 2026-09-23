#!/usr/bin/env python3
"""Tests for the q3 R5 broad-focus dataset/audit artifacts."""

from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import build_q3_r5_broadfocus_dataset as builder  # noqa: E402
import q3_r5_training_launcher as launcher  # noqa: E402


R5_ROOT = REPO_ROOT / "runs/eos-d384-q3-boundary-goal-v1-20260902T084144Z/headroom-q3r5-broadfocus-fiqa33-nf74-sf13-nfrecall74-rw050"
TRAIN = R5_ROOT / "data/beir294.q3r5-broadfocus-fiqa33-nf74-sf13-nfrecall74.train.jsonl"
MANIFEST = R5_ROOT / "data/beir294.q3r5-broadfocus-fiqa33-nf74-sf13-nfrecall74.manifest.json"
COVERAGE = R5_ROOT / "q3r5-broadfocus-fiqa33-nf74-sf13-nfrecall74.coverage.json"
LAUNCH_AUDIT = R5_ROOT / "reports/q3-r5-training-launch-audit.json"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_rows() -> list[dict]:
    return [json.loads(line) for line in TRAIN.read_text(encoding="utf-8").splitlines() if line.strip()]


def require_real_artifacts(testcase: unittest.TestCase) -> None:
    missing = [str(path) for path in (TRAIN, MANIFEST, COVERAGE, LAUNCH_AUDIT) if not path.is_file()]
    if missing:
        testcase.skipTest(f"real R5 artifacts not present: {missing}")


class BuildQ3R5BroadfocusDatasetTest(unittest.TestCase):
    def test_real_artifact_counts_hashes_and_legal_scope(self) -> None:
        require_real_artifacts(self)
        manifest = read_json(MANIFEST)
        rows = read_rows()
        self.assertEqual(builder.sha256_file(TRAIN), manifest["sha256"]["train_score_spectrum"])
        self.assertEqual(len(rows), 294)
        self.assertEqual(manifest["counts"]["workload"], builder.EXPECTED_WORKLOAD)
        self.assertEqual(manifest["counts"]["bucket_counts"], builder.EXPECTED_BUCKETS)
        self.assertEqual(manifest["counts"]["dataset_qid_counts"], builder.EXPECTED_DOMAIN_QIDS)
        self.assertEqual(manifest["nf_recall_audit"]["union_count"], 74)
        self.assertEqual(manifest["nf_recall_audit"]["overlap_count"], 14)
        self.assertEqual(manifest["nf_recall_audit"]["rank80_100_positive_count"], 57)
        self.assertFalse(manifest["release_train_allowed"])
        self.assertFalse(manifest["commercial_use_allowed"])
        self.assertFalse(manifest["quality_claim"])
        focus = [row for row in rows if row["selection_bucket"] == builder.FOCUS_BUCKET]
        self.assertEqual(len(focus), 110)
        self.assertTrue(all(len(row["candidate_doc_ids"]) == 2 for row in focus))
        self.assertTrue(all(row["positive_count"] == 1 and row["q3_negative_count"] == 1 for row in focus))
        self.assertTrue(all(row["turboquant_topk_loss_weight"] == 1.0 for row in focus))

    def test_real_launch_audit_proves_plan_only_no_mutation(self) -> None:
        require_real_artifacts(self)
        audit = read_json(LAUNCH_AUDIT)
        self.assertEqual(audit["schema"], launcher.LAUNCH_AUDIT_SCHEMA)
        self.assertFalse(audit["actual_training_ran"])
        self.assertTrue(audit["all_checks_passed"])
        self.assertEqual(audit["parsed_workload"], launcher.EXPECTED_PARSED_WORKLOAD)
        checks = audit["checks"]
        for check in (
            "candidate_pretrain_matches_anchor",
            "candidate_unchanged_by_plan",
            "canonical_anchor_unchanged",
            "metrics_target_absent_after_plan",
            "no_no_tokenizer_in_plan_or_training_argv",
            "workload_exact",
        ):
            self.assertTrue(checks[check], check)
        self.assertNotIn("--plan-only", audit["future_training_argv"])
        self.assertIn("--plan-only", audit["plan_argv"])
        self.assertEqual(audit["candidate"]["pre_plan"]["content_rollup_sha256"], "17335593ec850a1c01c8b31b17a32fe002368a508c06d431b761d45f1d9a76b0")
        self.assertEqual(audit["candidate"]["post_plan"]["content_rollup_sha256"], audit["candidate"]["pre_plan"]["content_rollup_sha256"])

    def test_contract_mutations_rejected_by_launcher(self) -> None:
        require_real_artifacts(self)
        original = read_json(LAUNCH_AUDIT)
        for mutate, needle in (
            (lambda data: data.__setitem__("actual_training_ran", True), "actual_training_ran=false"),
            (lambda data: data["legal_scope"].__setitem__("commercial_use_allowed", True), "legal_scope drift"),
            (lambda data: data["future_training_argv"].insert(2, "--plan-only"), "future_training_argv contains forbidden --plan-only"),
            (lambda data: data["future_training_argv"].append("--no-tokenizer"), "forbidden --no-tokenizer present"),
            (lambda data: data.__setitem__("parsed_workload", {"train": 1}), "parsed workload drift"),
            (lambda data: data["candidate"]["pre_plan"].__setitem__("content_rollup_sha256", "bad"), "content_rollup_sha256 mismatch"),
        ):
            with tempfile.TemporaryDirectory() as tmp:
                mutated = copy.deepcopy(original)
                mutate(mutated)
                audit_path = Path(tmp) / "launch.json"
                audit_path.write_text(json.dumps(mutated, indent=2, sort_keys=True) + "\n", encoding="utf-8")
                args = launcher.parse_args(
                    [
                        "--launch-audit", str(audit_path),
                        "--preflight-only",
                        "--attempt-id", "mutation",
                        "--stdout-capture", str(Path(tmp) / "stdout.txt"),
                        "--stderr-capture", str(Path(tmp) / "stderr.txt"),
                        "--output-audit", str(Path(tmp) / "preflight.json"),
                    ]
                )
                with self.assertRaises(launcher.PreflightError) as ctx:
                    launcher.preflight(args)
                self.assertIn(needle, "; ".join(ctx.exception.errors))


if __name__ == "__main__":
    unittest.main()
