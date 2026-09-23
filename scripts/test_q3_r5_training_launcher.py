#!/usr/bin/env python3
"""Tests for the audited Q3/R5 launcher wrapper constants."""

from __future__ import annotations

import copy
import json
import stat
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import q3_r5_training_launcher as launcher  # noqa: E402


def write(path: Path, data: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(data, encoding="utf-8")
    return path


class Q3R5TrainingLauncherTest(unittest.TestCase):
    def make_fixture(self, root: Path) -> dict[str, Path | dict]:
        anchor_dir = root / "anchor"
        candidate_dir = root / "candidate"
        for dirname in (anchor_dir, candidate_dir):
            write(dirname / "d384-pre.embed-train.mll", "embed train\n")
            write(dirname / "d384-pre.mll", "package\n")
            write(dirname / "d384-pre.tokenizer.mll", "tokenizer\n")
            write(dirname / "d384-pre.weights.mll", "weights\n")
        binary = write(root / "bin" / "eos", "#!/bin/sh\nexit 99\n")
        binary.chmod(binary.stat().st_mode | stat.S_IXUSR)
        train = write(root / "data" / "train.jsonl", "{}\n")
        manifest = write(root / "data" / "manifest.json", "{}\n")
        coverage = write(root / "data" / "coverage.json", "{}\n")
        source = write(root / "scripts" / "builder.py", "pass\n")
        stdout = write(root / "reports" / "plan.stdout.txt", "plan ok\n")
        stderr = write(root / "reports" / "plan.stderr.txt", "")
        metrics = root / "metrics" / "train.metrics.json"
        candidate_target = candidate_dir / "d384-pre.mll"
        anchor_target = anchor_dir / "d384-pre.mll"
        plan_argv = [
            str(binary),
            "train-embed",
            "--plan-only",
            "--score-spectrum-train",
            "--allow-research-only-score-spectrum",
            "--metrics-json",
            str(metrics),
            str(candidate_target),
            str(train),
        ]
        future_argv = [token for token in plan_argv if token != "--plan-only"]
        audit = {
            "schema": launcher.LAUNCH_AUDIT_SCHEMA,
            "created_utc": "2026-09-02T00:00:00Z",
            "actual_training_ran": False,
            "all_checks_passed": True,
            "legal_scope": dict(launcher.EXPECTED_LEGAL_SCOPE),
            "argv_delta": dict(launcher.EXPECTED_ARGV_DELTA),
            "plan_argv": plan_argv,
            "future_training_argv": future_argv,
            "metrics_target": str(metrics),
            "parsed_workload": dict(launcher.EXPECTED_PARSED_WORKLOAD),
            "plan_result": {
                "exit_status": 0,
                "stdout_path": str(stdout),
                "stdout_sha256": launcher.sha256_file(stdout),
                "stderr_path": str(stderr),
                "stderr_sha256": launcher.sha256_file(stderr),
            },
            "binary": {"path": str(binary), "sha256": launcher.sha256_file(binary)},
            "artifacts": {
                "train_jsonl": {"path": str(train), "sha256": launcher.sha256_file(train)},
                "manifest": {"path": str(manifest), "sha256": launcher.sha256_file(manifest)},
                "coverage": {"path": str(coverage), "sha256": launcher.sha256_file(coverage)},
            },
            "sources": {"builder": {"path": str(source), "sha256": launcher.sha256_file(source)}},
            "canonical_anchor": {
                "pre": launcher.package_state(anchor_target),
                "post": launcher.package_state(anchor_target),
            },
            "candidate": {
                "target": str(candidate_target),
                "pre_plan": launcher.package_state(candidate_target),
                "post_plan": launcher.package_state(candidate_target),
            },
            "checks": {"all_contract_checks_true": True},
        }
        audit_path = root / "reports" / "launch-audit.json"
        audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return {"root": root, "audit": audit, "audit_path": audit_path}

    def args_for(self, fixture: dict[str, Path | dict]):
        root = fixture["root"]
        audit_path = fixture["audit_path"]
        assert isinstance(root, Path)
        assert isinstance(audit_path, Path)
        return launcher.parse_args(
            [
                "--launch-audit",
                str(audit_path),
                "--preflight-only",
                "--attempt-id",
                "test",
                "--stdout-capture",
                str(root / "reports" / "attempt.stdout.txt"),
                "--stderr-capture",
                str(root / "reports" / "attempt.stderr.txt"),
                "--output-audit",
                str(root / "reports" / "attempt-preflight.json"),
            ]
        )

    def assert_rejected(self, fixture: dict[str, Path | dict], audit: dict, needle: str) -> None:
        path = fixture["audit_path"]
        assert isinstance(path, Path)
        path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        with self.assertRaises(launcher.PreflightError) as ctx:
            launcher.preflight(self.args_for(fixture))
        self.assertIn(needle, "; ".join(ctx.exception.errors))

    def test_preflight_accepts_r5_schema_and_workload(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fixture = self.make_fixture(Path(tmp))
            result = launcher.preflight(self.args_for(fixture))
            self.assertTrue(result["checks"]["all_checks_true"])
            self.assertEqual(result["schema"], launcher.PREFLIGHT_SCHEMA)
            self.assertEqual(result["expected_workload"], launcher.EXPECTED_PARSED_WORKLOAD)
            self.assertFalse(result["trainer_process_spawned"])

    def test_rejects_r4_schema_or_r4_workload(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fixture = self.make_fixture(Path(tmp))
            audit = copy.deepcopy(fixture["audit"])
            audit["schema"] = "eos.q3_r4_broad_gap.training_launch_audit.v1"
            self.assert_rejected(fixture, audit, "launch audit schema drift")
        with tempfile.TemporaryDirectory() as tmp:
            fixture = self.make_fixture(Path(tmp))
            audit = copy.deepcopy(fixture["audit"])
            audit["parsed_workload"] = {"train": 134, "batch": 4, "steps_per_epoch": 34, "train_pairs_per_epoch": 46912, "planned_pairs": 93824, "actual_train_pairs": 0}
            self.assert_rejected(fixture, audit, "parsed workload drift")


if __name__ == "__main__":
    unittest.main()
