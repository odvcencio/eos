#!/usr/bin/env python3
"""Tests for the audited Q3/R4 training launcher."""

from __future__ import annotations

import copy
import json
import os
import stat
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import q3_r4_training_launcher as launcher  # noqa: E402


def write(path: Path, data: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(data, encoding="utf-8")
    return path


class Q3R4TrainingLauncherTest(unittest.TestCase):
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
            "legal_scope": copy.deepcopy(launcher.EXPECTED_LEGAL_SCOPE),
            "argv_delta": copy.deepcopy(launcher.EXPECTED_ARGV_DELTA),
            "plan_argv": plan_argv,
            "future_training_argv": future_argv,
            "metrics_target": str(metrics),
            "parsed_workload": copy.deepcopy(launcher.EXPECTED_PARSED_WORKLOAD),
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
            "checks": {
                "all_contract_checks_true": True,
                "argv_delta_remove_plan_only_only": True,
                "candidate_pretrain_matches_anchor": True,
                "candidate_target_differs_from_anchor": True,
                "candidate_unchanged_by_plan": True,
                "canonical_anchor_unchanged": True,
                "metrics_target_absent_after_plan": True,
                "metrics_target_in_plan_argv": True,
                "metrics_target_in_training_argv": True,
                "no_no_tokenizer_in_plan_or_training_argv": True,
                "post_alias_zero_hard_rows_absent": True,
                "workload_exact": True,
            },
        }
        audit_path = root / "reports" / "launch-audit.json"
        audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return {
            "root": root,
            "audit": audit,
            "audit_path": audit_path,
            "anchor_target": anchor_target,
            "candidate_target": candidate_target,
            "metrics": metrics,
        }

    def args_for(self, fixture: dict[str, Path | dict], **kwargs: object):
        root = fixture["root"]
        assert isinstance(root, Path)
        audit_path = fixture["audit_path"]
        assert isinstance(audit_path, Path)
        argv = [
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
        if "expected_sha" in kwargs:
            argv.extend(["--expected-audit-sha256", str(kwargs["expected_sha"])])
        if kwargs.get("execute"):
            argv.remove("--preflight-only")
            argv.insert(2, "--execute")
        return launcher.parse_args(argv)

    def rewrite_audit(self, fixture: dict[str, Path | dict], audit: dict) -> None:
        path = fixture["audit_path"]
        assert isinstance(path, Path)
        path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    def assert_rejected(self, fixture: dict[str, Path | dict], audit: dict, needle: str) -> None:
        self.rewrite_audit(fixture, audit)
        with self.assertRaises(launcher.PreflightError) as ctx:
            launcher.preflight(self.args_for(fixture))
        self.assertIn(needle, "; ".join(ctx.exception.errors))

    def test_rollup_serialization_requires_line_records_with_final_newline(self) -> None:
        items = [
            {"bytes": 3, "name": "b.mll", "sha256": "b" * 64},
            {"bytes": 1, "name": "a.mll", "sha256": "a" * 64},
        ]
        expected_bytes = (
            b'{"bytes":3,"name":"b.mll","sha256":"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"}\n'
            b'{"bytes":1,"name":"a.mll","sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"}\n'
        )
        self.assertEqual(launcher.sibling_content_rollup(items), launcher.sha256_bytes(expected_bytes))
        self.assertNotEqual(launcher.sibling_content_rollup(items), launcher.sibling_json_array_rollup(items))
        self.assertNotEqual(launcher.sibling_content_rollup(items), launcher.sibling_no_final_newline_rollup(items))
        self.assertNotEqual(launcher.sibling_content_rollup(items), launcher.sibling_content_rollup(list(reversed(items))))
        self.assertNotEqual(launcher.sibling_content_rollup(items), launcher.sibling_content_rollup(items[:1]))

    def test_preflight_accepts_fixture_and_does_not_spawn_or_mutate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fixture = self.make_fixture(Path(tmp))
            root = fixture["root"]
            assert isinstance(root, Path)
            marker = root / "marker"
            audit = copy.deepcopy(fixture["audit"])
            audit["future_training_argv"] = [str(root / "bin" / "eos"), "train-embed", "--metrics-json", str(fixture["metrics"]), str(fixture["candidate_target"]), str(root / "data" / "train.jsonl")]
            audit["plan_argv"] = audit["future_training_argv"][:2] + ["--plan-only"] + audit["future_training_argv"][2:]
            self.rewrite_audit(fixture, audit)
            before_target_sha = launcher.sha256_file(fixture["candidate_target"])
            result = launcher.preflight(self.args_for(fixture))
            self.assertTrue(result["checks"]["all_checks_true"])
            self.assertFalse(result["trainer_process_spawned"])
            self.assertFalse(marker.exists())
            self.assertEqual(before_target_sha, launcher.sha256_file(fixture["candidate_target"]))
            self.assertFalse((root / "reports" / "attempt.stdout.txt").exists())
            self.assertFalse((root / "reports" / "attempt.stderr.txt").exists())

    def test_rejects_anchor_candidate_equality(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fixture = self.make_fixture(Path(tmp))
            audit = copy.deepcopy(fixture["audit"])
            audit["candidate"]["target"] = str(fixture["anchor_target"])
            audit["candidate"]["pre_plan"] = copy.deepcopy(audit["canonical_anchor"]["pre"])
            audit["candidate"]["post_plan"] = copy.deepcopy(audit["canonical_anchor"]["post"])
            audit["future_training_argv"][-2] = str(fixture["anchor_target"])
            audit["plan_argv"][-2] = str(fixture["anchor_target"])
            self.assert_rejected(fixture, audit, "candidate target equals canonical anchor target")

    def test_rejects_hash_drift(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fixture = self.make_fixture(Path(tmp))
            candidate_target = fixture["candidate_target"]
            assert isinstance(candidate_target, Path)
            candidate_target.write_text("package drift\n", encoding="utf-8")
            with self.assertRaises(launcher.PreflightError) as ctx:
                launcher.preflight(self.args_for(fixture))
            self.assertIn("candidate.pre_plan sibling_hashes mismatch", "; ".join(ctx.exception.errors))

    def test_rejects_preexisting_metrics_or_capture_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fixture = self.make_fixture(Path(tmp))
            metrics = fixture["metrics"]
            assert isinstance(metrics, Path)
            write(metrics, "{}\n")
            with self.assertRaises(launcher.PreflightError) as ctx:
                launcher.preflight(self.args_for(fixture))
            self.assertIn("metrics target already exists", "; ".join(ctx.exception.errors))

        with tempfile.TemporaryDirectory() as tmp:
            fixture = self.make_fixture(Path(tmp))
            root = fixture["root"]
            assert isinstance(root, Path)
            write(root / "reports" / "attempt.stdout.txt", "old\n")
            with self.assertRaises(launcher.PreflightError) as ctx:
                launcher.preflight(self.args_for(fixture))
            self.assertIn("stdout capture already exists", "; ".join(ctx.exception.errors))

        with tempfile.TemporaryDirectory() as tmp:
            fixture = self.make_fixture(Path(tmp))
            root = fixture["root"]
            assert isinstance(root, Path)
            write(root / "reports" / "attempt.stderr.txt", "old\n")
            with self.assertRaises(launcher.PreflightError) as ctx:
                launcher.preflight(self.args_for(fixture))
            self.assertIn("stderr capture already exists", "; ".join(ctx.exception.errors))

        with tempfile.TemporaryDirectory() as tmp:
            fixture = self.make_fixture(Path(tmp))
            root = fixture["root"]
            assert isinstance(root, Path)
            write(root / "reports" / "attempt-preflight.json", "{}\n")
            with self.assertRaises(launcher.PreflightError) as ctx:
                launcher.preflight(self.args_for(fixture))
            self.assertIn("output audit already exists", "; ".join(ctx.exception.errors))

    def test_rejects_wrong_audit_sha(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fixture = self.make_fixture(Path(tmp))
            with self.assertRaises(launcher.PreflightError) as ctx:
                launcher.preflight(self.args_for(fixture, expected_sha="0" * 64))
            self.assertIn("launch audit sha256 mismatch", "; ".join(ctx.exception.errors))

    def test_execute_requires_expected_audit_sha(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fixture = self.make_fixture(Path(tmp))
            with self.assertRaises(launcher.PreflightError) as ctx:
                launcher.preflight(self.args_for(fixture, execute=True))
            self.assertIn("execute mode requires --expected-audit-sha256", "; ".join(ctx.exception.errors))

    def test_rejects_plan_only_no_tokenizer_and_anchor_target(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fixture = self.make_fixture(Path(tmp))
            audit = copy.deepcopy(fixture["audit"])
            audit["future_training_argv"].insert(2, "--plan-only")
            self.assert_rejected(fixture, audit, "future_training_argv contains forbidden --plan-only")

        with tempfile.TemporaryDirectory() as tmp:
            fixture = self.make_fixture(Path(tmp))
            audit = copy.deepcopy(fixture["audit"])
            audit["future_training_argv"].insert(2, "--no-tokenizer")
            self.assert_rejected(fixture, audit, "forbidden --no-tokenizer present")

        with tempfile.TemporaryDirectory() as tmp:
            fixture = self.make_fixture(Path(tmp))
            audit = copy.deepcopy(fixture["audit"])
            audit["future_training_argv"][-2] = str(fixture["anchor_target"])
            self.assert_rejected(fixture, audit, "future_training_argv missing candidate target")


if __name__ == "__main__":
    unittest.main()
