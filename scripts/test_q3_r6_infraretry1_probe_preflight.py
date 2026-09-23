#!/usr/bin/env python3
"""Focused tests for the infraretry1 probe preflight wrapper."""

from __future__ import annotations

import copy
import hashlib
import io
import json
import stat
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import q3_r6_infraretry1_probe_preflight as wrapper  # noqa: E402


def write(path: Path, data: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(data, encoding="utf-8")
    return path


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class InfraretryProbePreflightTest(unittest.TestCase):
    def make_fixture(self, root: Path) -> dict[str, object]:
        def rel(path: Path) -> str:
            return str(path.relative_to(root))

        old_qrels_root = root / "old" / "probe" / "qrels"
        qrels = {}
        surfaces = {}
        for surface, (_bucket, dataset) in wrapper.r6.EXPECTED_SURFACES.items():
            path = old_qrels_root / dataset / f"{surface}.qrels.tsv"
            write(path, "query-id\tcorpus-id\tscore\nq1\td1\t1\n")
            qrels[surface] = path
        binary = write(root / "bin" / "eos-r6-frontier-retention-infraretry1", "#!/bin/sh\nexit 0\n")
        binary.chmod(binary.stat().st_mode | stat.S_IXUSR)
        metrics = write(root / "new" / "metrics" / "train.metrics.json", "{}\n")
        attestation = write(root / "new" / "reports" / "posttrain-attestation.json", json.dumps({"status": "passed"}) + "\n")
        base_audit = write(root / "new" / "reports" / "base-audit.json", json.dumps({"status": "passed", "aux_only": True}) + "\n")
        commands = []
        expected_outputs = []
        outputs_root = root / "new" / "probe" / "future-outputs"
        anchor = rel(root / "anchor" / "d384-pre.mll")
        candidate = rel(root / "candidate" / "d384-pre.mll")
        for surface, (bucket, dataset) in wrapper.r6.EXPECTED_SURFACES.items():
            surfaces[surface] = {
                "bucket": bucket,
                "dataset": dataset,
                "dataset_dir": f"datasets/{dataset}",
                "qrels": {"path": rel(qrels[surface]), "sha256": sha(qrels[surface]), "qid_count": 1, "qrel_count": 1},
            }
            for label, package in (("anchor", anchor), ("candidate", candidate)):
                stem = outputs_root / surface / f"{label}.{surface}.q3.r6-probe"
                outputs = {
                    "metrics_json": rel(stem.with_suffix(".json")),
                    "metrics_tsv": rel(stem.with_suffix(".tsv")),
                    "per_query_jsonl": rel(stem.with_suffix(".per-query.jsonl")),
                }
                expected_outputs.extend(outputs.values())
                argv = [
                    rel(binary),
                    "eval-retrieval-turboquant",
                    "--bits", "3",
                    "--quantizer-seed", str(wrapper.r6.Q3_SEED),
                    "--top-k", "100",
                    "--per-query-top-k", "100",
                    "--dataset", dataset,
                    "--qrels", rel(qrels[surface]),
                    "--metrics-json", outputs["metrics_json"],
                    "--metrics-tsv", outputs["metrics_tsv"],
                    "--per-query-jsonl", outputs["per_query_jsonl"],
                    package,
                    f"datasets/{dataset}",
                ]
                commands.append({"surface_id": surface, "label": label, "bucket": bucket, "dataset": dataset, "outputs": outputs, "argv": argv, "package": package})
        manifest = {
            "schema": wrapper.r6.PROBE_SCHEMA,
            "created_utc": "2026-09-02T00:00:00Z",
            "arm_id": wrapper.ARM_ID,
            "binary": {"path": rel(binary), "sha256": sha(binary)},
            "anchor_package": {"path": anchor, "sibling_rollup_sha256": "future integration binding"},
            "candidate_package": {"arm_id": wrapper.ARM_ID, "path": candidate, "sibling_rollup_sha256": "posttrain-rollup"},
            "candidate_effective_base_leakage_audit": {"path": rel(base_audit), "sha256": sha(base_audit), "status": "passed", "base_hard_soft_recovery_work_units": 0, "aux_only": True},
            "candidate_train_metrics": {"path": rel(metrics), "sha256": sha(metrics)},
            "posttrain_attestation": {"path": rel(attestation), "sha256": sha(attestation), "status": "passed"},
            "supported_package_inspect_evidence": {"command": [rel(binary), "inspect", candidate], "exit_status": 0, "package_verify_ok": True, "train_profile_step": 578, "stdout_sha256": "inspect-out", "stderr_sha256": hashlib.sha256(b"").hexdigest()},
            "surfaces": surfaces,
            "surface_count": 7,
            "command_count": 14,
            "expected_output_count": 42,
            "commands": commands,
            "expected_outputs": sorted(expected_outputs),
            "future_paths": {"outputs_root": rel(outputs_root), "anchor_package": anchor, "candidate_package": candidate},
            "all_expected_outputs_absent": True,
            "all_expected_outputs_contained": True,
        }
        manifest_path = root / "new" / "probe" / "q3r6-frontier-retention.probe-manifest.json"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        diff = {"allowed_identity_path_changes_only": True, "checks": {"ok": True}}
        diff_path = root / "new" / "probe" / "q3-r6-infraretry1-probe-semantic-diff-audit.json"
        diff_path.write_text(json.dumps(diff, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return {"root": root, "binary": binary, "metrics": metrics, "attestation": attestation, "base_audit": base_audit, "manifest": manifest, "manifest_path": manifest_path, "diff_path": diff_path}

    def patch_globals(self, fixture: dict[str, object]) -> None:
        self.addCleanup(setattr, wrapper, "REPO_ROOT", wrapper.REPO_ROOT)
        self.addCleanup(setattr, wrapper.r6, "REPO_ROOT", wrapper.r6.REPO_ROOT)
        self.addCleanup(setattr, wrapper, "INFRARETRY_BINARY", wrapper.INFRARETRY_BINARY)
        self.addCleanup(setattr, wrapper, "INFRARETRY_BINARY_SHA256", wrapper.INFRARETRY_BINARY_SHA256)
        self.addCleanup(setattr, wrapper, "INFRARETRY_PROBE_MANIFEST", wrapper.INFRARETRY_PROBE_MANIFEST)
        self.addCleanup(setattr, wrapper, "INFRARETRY_PROBE_MANIFEST_SHA256", wrapper.INFRARETRY_PROBE_MANIFEST_SHA256)
        self.addCleanup(setattr, wrapper, "SEMANTIC_DIFF_AUDIT", wrapper.SEMANTIC_DIFF_AUDIT)
        self.addCleanup(setattr, wrapper, "SEMANTIC_DIFF_AUDIT_SHA256", wrapper.SEMANTIC_DIFF_AUDIT_SHA256)
        self.addCleanup(setattr, wrapper, "POSTTRAIN_ROLLUP_SHA256", wrapper.POSTTRAIN_ROLLUP_SHA256)
        self.addCleanup(setattr, wrapper, "METRICS_PATH", wrapper.METRICS_PATH)
        self.addCleanup(setattr, wrapper, "METRICS_SHA256", wrapper.METRICS_SHA256)
        self.addCleanup(setattr, wrapper, "POSTTRAIN_ATTESTATION", wrapper.POSTTRAIN_ATTESTATION)
        self.addCleanup(setattr, wrapper, "POSTTRAIN_ATTESTATION_SHA256", wrapper.POSTTRAIN_ATTESTATION_SHA256)
        self.addCleanup(setattr, wrapper, "EFFECTIVE_BASE_LEAKAGE_AUDIT", wrapper.EFFECTIVE_BASE_LEAKAGE_AUDIT)
        self.addCleanup(setattr, wrapper, "EFFECTIVE_BASE_LEAKAGE_AUDIT_SHA256", wrapper.EFFECTIVE_BASE_LEAKAGE_AUDIT_SHA256)
        self.addCleanup(setattr, wrapper, "INSPECT_STDOUT_SHA256", wrapper.INSPECT_STDOUT_SHA256)
        root = fixture["root"]
        assert isinstance(root, Path)
        wrapper.REPO_ROOT = root
        wrapper.r6.REPO_ROOT = root
        wrapper.INFRARETRY_BINARY = fixture["binary"]
        wrapper.INFRARETRY_BINARY_SHA256 = sha(fixture["binary"])
        wrapper.INFRARETRY_PROBE_MANIFEST = fixture["manifest_path"]
        wrapper.INFRARETRY_PROBE_MANIFEST_SHA256 = sha(fixture["manifest_path"])
        wrapper.SEMANTIC_DIFF_AUDIT = fixture["diff_path"]
        wrapper.SEMANTIC_DIFF_AUDIT_SHA256 = sha(fixture["diff_path"])
        wrapper.POSTTRAIN_ROLLUP_SHA256 = "posttrain-rollup"
        wrapper.METRICS_PATH = fixture["metrics"]
        wrapper.METRICS_SHA256 = sha(fixture["metrics"])
        wrapper.POSTTRAIN_ATTESTATION = fixture["attestation"]
        wrapper.POSTTRAIN_ATTESTATION_SHA256 = sha(fixture["attestation"])
        wrapper.EFFECTIVE_BASE_LEAKAGE_AUDIT = fixture["base_audit"]
        wrapper.EFFECTIVE_BASE_LEAKAGE_AUDIT_SHA256 = sha(fixture["base_audit"])
        wrapper.INSPECT_STDOUT_SHA256 = "inspect-out"

    def write_manifest(self, fixture: dict[str, object], manifest: dict) -> None:
        path = fixture["manifest_path"]
        assert isinstance(path, Path)
        path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        wrapper.INFRARETRY_PROBE_MANIFEST_SHA256 = sha(path)

    def test_pending_preflight_accepts_bound_binary_and_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fixture = self.make_fixture(Path(tmp))
            self.patch_globals(fixture)
            manifest = wrapper.validate_manifest(fixture["manifest_path"], expected_manifest_sha256=wrapper.INFRARETRY_PROBE_MANIFEST_SHA256, allow_pending_runtime_bindings=True)
            self.assertEqual(manifest["_infraretry1_binding"]["command_count"], 14)
            self.assertEqual(manifest["_infraretry1_binding"]["expected_output_count"], 42)

    def test_strict_without_pending_accepts_posttrain_bindings(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fixture = self.make_fixture(Path(tmp))
            self.patch_globals(fixture)
            manifest = wrapper.validate_manifest(fixture["manifest_path"], expected_manifest_sha256=wrapper.INFRARETRY_PROBE_MANIFEST_SHA256, allow_pending_runtime_bindings=False)
            self.assertEqual(manifest["candidate_package"]["sibling_rollup_sha256"], "posttrain-rollup")

    def test_rejects_manifest_sha_binary_argv_diff(self) -> None:
        cases = [
            ("manifest sha", lambda fixture, manifest: None, "infraretry1 probe manifest sha mismatch", "0" * 64),
            ("binary manifest", lambda fixture, manifest: manifest["binary"].update({"sha256": "0" * 64}), "infraretry1 binary manifest binding mismatch", None),
            ("candidate rollup", lambda fixture, manifest: manifest["candidate_package"].update({"sibling_rollup_sha256": "future post-train integration binding"}), "infraretry1 candidate posttrain rollup mismatch", None),
            ("metrics binding", lambda fixture, manifest: manifest["candidate_train_metrics"].update({"sha256": "0" * 64}), "infraretry1 metrics manifest binding mismatch", None),
            ("attestation binding", lambda fixture, manifest: manifest["posttrain_attestation"].update({"status": "failed"}), "infraretry1 posttrain attestation binding mismatch", None),
            ("base audit binding", lambda fixture, manifest: manifest["candidate_effective_base_leakage_audit"].update({"aux_only": False}), "infraretry1 effective-base leakage audit binding mismatch", None),
            ("inspect evidence", lambda fixture, manifest: manifest["supported_package_inspect_evidence"].update({"package_verify_ok": False}), "infraretry1 supported package inspect evidence mismatch", None),
            ("argv0", lambda fixture, manifest: manifest["commands"][0]["argv"].__setitem__(0, "wrong-bin"), "infraretry1 command 0 binary argv mismatch", None),
            ("diff", lambda fixture, manifest: Path(fixture["diff_path"]).write_text(json.dumps({"allowed_identity_path_changes_only": False, "checks": {"ok": False}}) + "\n", encoding="utf-8"), "infraretry1 semantic diff audit sha mismatch", None),
        ]
        for name, mutate, needle, expected_sha in cases:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as tmp:
                fixture = self.make_fixture(Path(tmp))
                self.patch_globals(fixture)
                manifest = copy.deepcopy(fixture["manifest"])
                mutate(fixture, manifest)
                self.write_manifest(fixture, manifest)
                if name == "diff":
                    wrapper.SEMANTIC_DIFF_AUDIT_SHA256 = sha(fixture["diff_path"])
                    needle = "infraretry1 semantic diff audit failed"
                with self.assertRaises(wrapper.r6.ProbeContractError) as ctx:
                    wrapper.validate_manifest(fixture["manifest_path"], expected_manifest_sha256=expected_sha or wrapper.INFRARETRY_PROBE_MANIFEST_SHA256, allow_pending_runtime_bindings=True)
                self.assertIn(needle, str(ctx.exception))

    def test_preflight_binding_rejects_existing_expected_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fixture = self.make_fixture(Path(tmp))
            self.patch_globals(fixture)
            manifest = copy.deepcopy(fixture["manifest"])
            write(Path(fixture["root"]) / manifest["expected_outputs"][0], "{}\n")
            self.write_manifest(fixture, manifest)
            with self.assertRaises(wrapper.r6.ProbeContractError) as ctx:
                wrapper.validate_manifest(
                    fixture["manifest_path"],
                    expected_manifest_sha256=wrapper.INFRARETRY_PROBE_MANIFEST_SHA256,
                    require_outputs_absent=True,
                    allow_pending_runtime_bindings=True,
                )
            self.assertIn("future output path already exists", str(ctx.exception))

    def test_post_output_binding_permits_existing_outputs_but_keeps_shape_guards(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fixture = self.make_fixture(Path(tmp))
            self.patch_globals(fixture)
            manifest = copy.deepcopy(fixture["manifest"])
            write(Path(fixture["root"]) / manifest["expected_outputs"][0], "{}\n")
            self.write_manifest(fixture, manifest)
            binding = wrapper.validate_infraretry_manifest_binding(
                fixture["manifest_path"],
                wrapper.INFRARETRY_PROBE_MANIFEST_SHA256,
                require_outputs_absent=False,
            )
            self.assertEqual(binding["expected_output_count"], 42)

            duplicate_manifest = copy.deepcopy(manifest)
            duplicate_manifest["expected_outputs"][-1] = duplicate_manifest["expected_outputs"][0]
            self.write_manifest(fixture, duplicate_manifest)
            with self.assertRaises(wrapper.r6.ProbeContractError) as duplicate_ctx:
                wrapper.validate_infraretry_manifest_binding(
                    fixture["manifest_path"],
                    wrapper.INFRARETRY_PROBE_MANIFEST_SHA256,
                    require_outputs_absent=False,
                )
            self.assertIn("expected output set mismatch", str(duplicate_ctx.exception))

            escaping_manifest = copy.deepcopy(manifest)
            escaping_manifest["expected_outputs"][0] = "../escaped.json"
            self.write_manifest(fixture, escaping_manifest)
            with self.assertRaises(wrapper.r6.ProbeContractError) as escape_ctx:
                wrapper.validate_infraretry_manifest_binding(
                    fixture["manifest_path"],
                    wrapper.INFRARETRY_PROBE_MANIFEST_SHA256,
                    require_outputs_absent=False,
                )
            self.assertIn("future output path escapes outputs root", str(escape_ctx.exception))

    def test_normal_main_delegates_to_gate_when_expected_output_exists(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fixture = self.make_fixture(Path(tmp))
            self.patch_globals(fixture)
            manifest = copy.deepcopy(fixture["manifest"])
            write(Path(fixture["root"]) / manifest["expected_outputs"][0], "{}\n")
            self.write_manifest(fixture, manifest)
            original_validate_gate = wrapper.r6.validate_gate
            calls: list[Path] = []

            def fake_validate_gate(manifest_path: Path) -> dict[str, object]:
                calls.append(manifest_path)
                return {"schema": "fake.gate.summary", "ok": True}

            wrapper.r6.validate_gate = fake_validate_gate
            self.addCleanup(setattr, wrapper.r6, "validate_gate", original_validate_gate)
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                rc = wrapper.main(
                    [
                        "--probe-manifest",
                        str(fixture["manifest_path"]),
                        "--expected-manifest-sha256",
                        wrapper.INFRARETRY_PROBE_MANIFEST_SHA256,
                    ]
                )
            self.assertEqual(rc, 0)
            self.assertEqual(calls, [fixture["manifest_path"]])
            self.assertIn("fake.gate.summary", stdout.getvalue())

    def test_normal_main_reaches_base_gate_validation_when_expected_output_exists(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fixture = self.make_fixture(Path(tmp))
            self.patch_globals(fixture)
            manifest = copy.deepcopy(fixture["manifest"])
            write(Path(fixture["root"]) / manifest["expected_outputs"][0], "{}\n")
            self.write_manifest(fixture, manifest)
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                rc = wrapper.main(
                    [
                        "--probe-manifest",
                        str(fixture["manifest_path"]),
                        "--expected-manifest-sha256",
                        wrapper.INFRARETRY_PROBE_MANIFEST_SHA256,
                    ]
                )
            self.assertEqual(rc, 1)
            self.assertIn("missing outputs", stdout.getvalue())
            self.assertNotIn("future output path already exists", stdout.getvalue())


if __name__ == "__main__":
    unittest.main()
