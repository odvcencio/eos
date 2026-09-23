#!/usr/bin/env python3
"""Focused tests for the Q3-R6-INFRARETRY1 probe launcher."""

from __future__ import annotations

import copy
import hashlib
import json
import stat
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import q3_r6_infraretry1_probe_launcher as launcher  # noqa: E402


def write(path: Path, data: str | bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(data, bytes):
        path.write_bytes(data)
    else:
        path.write_text(data, encoding="utf-8")
    return path


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Proc:
    def __init__(self, returncode: int = 0, stdout: bytes = b"{}", stderr: bytes = b""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


class ProbeLauncherTest(unittest.TestCase):
    def make_fixture(self, root: Path) -> dict[str, object]:
        binary = write(root / "bin" / "eos", "#!/bin/sh\nexit 0\n")
        binary.chmod(binary.stat().st_mode | stat.S_IXUSR)
        candidate = write(root / "packages" / "candidate" / "d384-pre.mll", b"root")
        for suffix in (".embed-train.mll", ".embedding.mll", ".memory.mll", ".package.mll", ".tokenizer.mll", ".train-profile.mll", ".train.mll", ".weights.mll"):
            write(candidate.with_name("d384-pre" + suffix), suffix)
        metrics = write(root / "reports" / "metrics.json", "{}\n")
        att = write(root / "reports" / "attestation.json", "{}\n")
        base = write(root / "reports" / "base.json", "{}\n")
        qrels_root = root / "qrels"
        outputs_root = root / "probe" / "future-outputs"
        commands = []
        expected_outputs = []
        surfaces = {}
        for surface_id, (bucket, dataset) in launcher.preflight.r6.EXPECTED_SURFACES.items():
            qrels = write(qrels_root / dataset / f"{surface_id}.qrels.tsv", "query-id\tcorpus-id\tscore\nq1\td1\t1\n")
            surfaces[surface_id] = {
                "bucket": bucket,
                "dataset": dataset,
                "dataset_dir": f"datasets/{dataset}",
                "qrels": {"path": str(qrels), "sha256": sha(qrels), "qid_count": 1, "qrel_count": 1},
            }
            for label, package in (("anchor", str(root / "anchor" / "d384-pre.mll")), ("candidate", str(candidate))):
                stem = outputs_root / surface_id / f"{label}.{surface_id}.q3.r6-probe"
                outputs = {
                    "metrics_json": str(stem.with_suffix(".json")),
                    "metrics_tsv": str(stem.with_suffix(".tsv")),
                    "per_query_jsonl": str(stem.with_suffix(".per-query.jsonl")),
                }
                expected_outputs.extend(outputs.values())
                argv = [
                    str(binary),
                    "eval-retrieval-turboquant",
                    "--bits", "3",
                    "--quantizer-seed", str(launcher.preflight.r6.Q3_SEED),
                    "--top-k", "100",
                    "--per-query-top-k", "100",
                    "--dataset", dataset,
                    "--qrels", str(qrels),
                    "--metrics-json", outputs["metrics_json"],
                    "--metrics-tsv", outputs["metrics_tsv"],
                    "--per-query-jsonl", outputs["per_query_jsonl"],
                    package,
                    f"datasets/{dataset}",
                ]
                commands.append({"surface_id": surface_id, "label": label, "bucket": bucket, "dataset": dataset, "outputs": outputs, "argv": argv, "package": package})
        manifest = {
            "schema": launcher.preflight.r6.PROBE_SCHEMA,
            "binary": {"path": str(binary), "sha256": sha(binary)},
            "candidate_package": {"path": str(candidate), "sibling_rollup_sha256": "rollup"},
            "candidate_train_metrics": {"path": str(metrics), "sha256": sha(metrics)},
            "posttrain_attestation": {"path": str(att), "sha256": sha(att), "status": "passed"},
            "candidate_effective_base_leakage_audit": {"path": str(base), "sha256": sha(base), "status": "passed", "base_hard_soft_recovery_work_units": 0, "aux_only": True},
            "surfaces": surfaces,
            "commands": commands,
            "expected_outputs": sorted(expected_outputs),
            "future_paths": {"outputs_root": str(outputs_root)},
            "surface_count": 7,
            "command_count": 14,
            "expected_output_count": 42,
            "command_argv_sha256": "argv-sha",
        }
        manifest_path = write(root / "probe" / "manifest.json", json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        wrapper = write(root / "scripts" / "q3_r6_infraretry1_probe_preflight.py", "wrapper\n")
        return {"root": root, "manifest": manifest, "manifest_path": manifest_path, "wrapper": wrapper, "binary": binary, "candidate": candidate, "outputs_root": outputs_root}

    def patch_globals(self, fixture: dict[str, object]) -> None:
        root = fixture["root"]
        manifest = fixture["manifest"]
        assert isinstance(root, Path)
        assert isinstance(manifest, dict)
        for name in (
            "REPO_ROOT", "PROBE_MANIFEST", "REPORTS_ROOT", "DEFAULT_PLAN_AUDIT", "DEFAULT_PLAN_RECEIPT",
            "DEFAULT_PLAN_STDOUT", "DEFAULT_PLAN_STDERR", "DEFAULT_EXECUTE_CAPTURE_DIR", "DEFAULT_EXECUTE_AUDIT",
            "EXPECTED_MANIFEST_SHA256", "EXPECTED_WRAPPER_SHA256", "EXPECTED_BINARY_SHA256",
            "EXPECTED_CANDIDATE_ROLLUP_SHA256", "EXPECTED_METRICS_SHA256", "EXPECTED_ATTESTATION_SHA256",
            "EXPECTED_BASE_AUDIT_SHA256", "EXPECTED_SEMANTIC_DIFF_AUDIT_SHA256", "GATE_COMMAND",
        ):
            self.addCleanup(setattr, launcher, name, getattr(launcher, name))
        launcher.REPO_ROOT = root
        launcher.PROBE_MANIFEST = fixture["manifest_path"]
        launcher.REPORTS_ROOT = root / "probe" / "reports"
        launcher.DEFAULT_PLAN_AUDIT = launcher.REPORTS_ROOT / "plan-audit.json"
        launcher.DEFAULT_PLAN_RECEIPT = launcher.REPORTS_ROOT / "plan-receipt.json"
        launcher.DEFAULT_PLAN_STDOUT = launcher.REPORTS_ROOT / "plan.stdout.txt"
        launcher.DEFAULT_PLAN_STDERR = launcher.REPORTS_ROOT / "plan.stderr.txt"
        launcher.DEFAULT_EXECUTE_CAPTURE_DIR = launcher.REPORTS_ROOT / "execute-captures"
        launcher.DEFAULT_EXECUTE_AUDIT = launcher.REPORTS_ROOT / "execute-audit.json"
        launcher.EXPECTED_MANIFEST_SHA256 = sha(fixture["manifest_path"])
        launcher.EXPECTED_WRAPPER_SHA256 = sha(fixture["wrapper"])
        launcher.EXPECTED_BINARY_SHA256 = manifest["binary"]["sha256"]
        launcher.EXPECTED_CANDIDATE_ROLLUP_SHA256 = launcher.sibling_rollup(manifest["candidate_package"]["path"])["sibling_rollup_sha256"]
        manifest["candidate_package"]["sibling_rollup_sha256"] = launcher.EXPECTED_CANDIDATE_ROLLUP_SHA256
        Path(fixture["manifest_path"]).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        launcher.EXPECTED_MANIFEST_SHA256 = sha(fixture["manifest_path"])
        launcher.EXPECTED_METRICS_SHA256 = manifest["candidate_train_metrics"]["sha256"]
        launcher.EXPECTED_ATTESTATION_SHA256 = manifest["posttrain_attestation"]["sha256"]
        launcher.EXPECTED_BASE_AUDIT_SHA256 = manifest["candidate_effective_base_leakage_audit"]["sha256"]
        launcher.EXPECTED_SEMANTIC_DIFF_AUDIT_SHA256 = "semantic"
        launcher.GATE_COMMAND = ["python", "scripts/q3_r6_infraretry1_probe_preflight.py", "--probe-manifest", str(fixture["manifest_path"]), "--expected-manifest-sha256", launcher.EXPECTED_MANIFEST_SHA256]

        self.preflight_patcher = mock.patch.object(launcher.preflight, "validate_manifest", side_effect=lambda *a, **k: copy.deepcopy(manifest))
        self.preflight_patcher.start()
        self.addCleanup(self.preflight_patcher.stop)
        self.addCleanup(setattr, launcher.preflight, "SEMANTIC_DIFF_AUDIT_SHA256", launcher.preflight.SEMANTIC_DIFF_AUDIT_SHA256)
        launcher.preflight.SEMANTIC_DIFF_AUDIT_SHA256 = "semantic"

    def args(self, fixture: dict[str, object], *extra: str) -> list[str]:
        return ["--probe-manifest", str(fixture["manifest_path"]), "--expected-manifest-sha256", launcher.EXPECTED_MANIFEST_SHA256, *extra]

    def test_plan_only_zero_spawn_and_no_output_dirs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fixture = self.make_fixture(Path(tmp))
            self.patch_globals(fixture)
            with mock.patch.object(launcher.subprocess, "run") as run:
                rc = launcher.main(self.args(fixture, "--plan-only"))
            self.assertEqual(rc, 0)
            run.assert_not_called()
            self.assertFalse(Path(fixture["outputs_root"]).exists())
            audit = json.loads(launcher.DEFAULT_PLAN_AUDIT.read_text(encoding="utf-8"))
            self.assertEqual(audit["counts"]["future_probe_spawns"], 14)
            self.assertEqual(audit["counts"]["future_gate_spawns"], 1)
            self.assertEqual(audit["counts"]["current_probe_spawns"], 0)

    def prepare_plan(self, fixture: dict[str, object]) -> str:
        self.assertEqual(launcher.main(self.args(fixture, "--plan-only")), 0)
        return sha(launcher.DEFAULT_PLAN_AUDIT)

    def test_execute_exact_14_plus_1_gate_once(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fixture = self.make_fixture(Path(tmp))
            self.patch_globals(fixture)
            plan_sha = self.prepare_plan(fixture)

            def fake_run(argv, **_kwargs):
                if argv == launcher.GATE_COMMAND:
                    return Proc(0, b'{"passed": true, "command_count": 14, "output_count": 42}\n')
                mj = Path(argv[argv.index("--metrics-json") + 1])
                mt = Path(argv[argv.index("--metrics-tsv") + 1])
                pq = Path(argv[argv.index("--per-query-jsonl") + 1])
                write(mj, "{}\n")
                write(mt, "x\n")
                write(pq, "{}\n")
                return Proc(0, b"ok\n")

            with mock.patch.object(launcher.subprocess, "run", side_effect=fake_run) as run:
                rc = launcher.main(self.args(fixture, "--execute", "--expected-plan-audit-sha256", plan_sha))
            self.assertEqual(rc, 0)
            self.assertEqual(run.call_count, 15)
            audit = json.loads(launcher.DEFAULT_EXECUTE_AUDIT.read_text(encoding="utf-8"))
            self.assertEqual(audit["spawn_count"], 14)
            self.assertEqual(audit["gate_spawn_count"], 1)
            self.assertTrue(audit["passed"])

    def test_execute_rejects_preexisting_output_capture_or_audit(self) -> None:
        for kind in ("output", "capture", "audit"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as tmp:
                fixture = self.make_fixture(Path(tmp))
                self.patch_globals(fixture)
                plan_sha = self.prepare_plan(fixture)
                if kind == "output":
                    write(Path(fixture["manifest"]["expected_outputs"][0]), "old\n")
                elif kind == "capture":
                    launcher.DEFAULT_EXECUTE_CAPTURE_DIR.mkdir(parents=True)
                else:
                    write(launcher.DEFAULT_EXECUTE_AUDIT, "{}\n")
                with mock.patch.object(launcher.subprocess, "run") as run:
                    rc = launcher.main(self.args(fixture, "--execute", "--expected-plan-audit-sha256", plan_sha))
                self.assertEqual(rc, 1)
                run.assert_not_called()
                if kind != "audit":
                    self.assertTrue(launcher.DEFAULT_EXECUTE_AUDIT.exists())

    def test_execute_parent_creation_timing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fixture = self.make_fixture(Path(tmp))
            self.patch_globals(fixture)
            plan_sha = self.prepare_plan(fixture)
            self.assertFalse(Path(fixture["outputs_root"]).exists())

            def fail_first(argv, **_kwargs):
                mj = Path(argv[argv.index("--metrics-json") + 1])
                self.assertTrue(mj.parent.exists())
                return Proc(1, b"", b"bad\n")

            with mock.patch.object(launcher.subprocess, "run", side_effect=fail_first):
                rc = launcher.main(self.args(fixture, "--execute", "--expected-plan-audit-sha256", plan_sha))
            self.assertEqual(rc, 1)
            self.assertTrue(Path(fixture["outputs_root"]).exists())

    def test_nonzero_stops_no_retry_no_gate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fixture = self.make_fixture(Path(tmp))
            self.patch_globals(fixture)
            plan_sha = self.prepare_plan(fixture)
            with mock.patch.object(launcher.subprocess, "run", return_value=Proc(2)) as run:
                rc = launcher.main(self.args(fixture, "--execute", "--expected-plan-audit-sha256", plan_sha))
            self.assertEqual(rc, 1)
            self.assertEqual(run.call_count, 1)
            audit = json.loads(launcher.DEFAULT_EXECUTE_AUDIT.read_text(encoding="utf-8"))
            self.assertEqual(audit["spawn_count"], 1)
            self.assertEqual(audit["gate_spawn_count"], 0)
            self.assertEqual(audit["retry_count"], 0)

    def test_missing_output_stops(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fixture = self.make_fixture(Path(tmp))
            self.patch_globals(fixture)
            plan_sha = self.prepare_plan(fixture)
            with mock.patch.object(launcher.subprocess, "run", return_value=Proc(0)) as run:
                rc = launcher.main(self.args(fixture, "--execute", "--expected-plan-audit-sha256", plan_sha))
            self.assertEqual(rc, 1)
            self.assertEqual(run.call_count, 1)
            audit = json.loads(launcher.DEFAULT_EXECUTE_AUDIT.read_text(encoding="utf-8"))
            self.assertIn("missing outputs", audit["errors"][0])

    def test_input_candidate_mutation_stops_before_spawn(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fixture = self.make_fixture(Path(tmp))
            self.patch_globals(fixture)
            plan_sha = self.prepare_plan(fixture)
            write(Path(fixture["candidate"]).with_name("d384-pre.weights.mll"), "mutated\n")
            with mock.patch.object(launcher.subprocess, "run") as run:
                rc = launcher.main(self.args(fixture, "--execute", "--expected-plan-audit-sha256", plan_sha))
            self.assertEqual(rc, 1)
            run.assert_not_called()
            self.assertTrue(launcher.DEFAULT_EXECUTE_AUDIT.exists())

    def test_always_write_audit_on_bad_plan_hash(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fixture = self.make_fixture(Path(tmp))
            self.patch_globals(fixture)
            self.prepare_plan(fixture)
            rc = launcher.main(self.args(fixture, "--execute", "--expected-plan-audit-sha256", "0" * 64))
            self.assertEqual(rc, 1)
            self.assertTrue(launcher.DEFAULT_EXECUTE_AUDIT.exists())


if __name__ == "__main__":
    unittest.main()
