#!/usr/bin/env python3
"""Focused tests for the R6 plan-only training launcher contract."""

from __future__ import annotations

import copy
import inspect
import json
import stat
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import q3_r6_training_launcher as launcher  # noqa: E402


def write(path: Path, data: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(data, encoding="utf-8")
    return path


class Q3R6TrainingLauncherTest(unittest.TestCase):
    def make_fixture(self, root: Path) -> dict[str, object]:
        anchor = root / "anchor" / "d384-pre.mll"
        candidate = root / "candidate" / "d384-pre.mll"
        for package in (anchor, candidate):
            for suffix, body in (
                (".embed-train.mll", "embed\n"),
                (".embedding.mll", "embedding\n"),
                (".memory.mll", "memory\n"),
                (".mll", "package\n"),
                (".package.mll", "pkg\n"),
                (".tokenizer.mll", "tok\n"),
                (".train-profile.mll", "profile\n"),
                (".train.mll", "train\n"),
                (".weights.mll", "weights\n"),
            ):
                write(package.with_name("d384-pre" + suffix), body)
        binary = write(root / "bin" / "eos", "#!/bin/sh\nexit 99\n")
        binary.chmod(binary.stat().st_mode | stat.S_IXUSR)
        launcher_script = write(root / "sources" / "launcher.py", "launcher\n")
        launcher_test = write(root / "sources" / "test_launcher.py", "test launcher\n")
        stdout = write(root / "reports" / "plan.stdout.txt", "planned workload: train=1155 score_spectrum_grouped examples batch=4 steps/epoch=289 train_pairs/epoch=27081 pairs(planned=54162 actual=0) score_spectrum_activation_microbatch=0 score_spectrum_aux_only_rows=1155 turboquant_topk_recall=weight:0.5 cutoff:100 tau:0.05 margin:0 mask:q3\n")
        stderr = write(root / "reports" / "plan.stderr.txt", "")
        metrics = root / "metrics" / "train.metrics.json"
        plan = [
            str(binary),
            "train-embed",
            "--plan-only",
            "--score-spectrum-train",
            "--allow-research-only-score-spectrum",
            "--shuffle=false",
            "--epochs", "2",
            "--batch-size", "4",
            "--seed", "191",
            "--contrastive-loss", "grouped_infonce",
            "--temperature", "0.05",
            "--lr", "0.000002",
            "--restore-best=false",
            "--metrics-json", str(metrics),
            "--score-spectrum-loss-mode", "hard_soft_recovery",
            "--score-spectrum-recovery-weight", "0.05",
            "--score-spectrum-recovery-margin", "0",
            "--score-spectrum-recovery-top-k", "4",
            "--score-spectrum-recovery-tau", "0.05",
            "--turboquant-topk-objectives", "fullDim:3=0.04",
            "--turboquant-topk-loss", "lambdandcg",
            "--turboquant-topk-cutoff", "10",
            "--turboquant-topk-tau", "0.05",
            "--turboquant-topk-margin", "0.002",
            "--turboquant-topk-negative-mask", "q3",
            "--turboquant-topk-recall-weight", "0.50",
            "--turboquant-topk-recall-cutoff", "100",
            "--turboquant-topk-recall-tau", "0.05",
            "--turboquant-topk-recall-margin", "0",
            "--turboquant-topk-recall-negative-mask", "q3",
            "--turboquant-prefix-score-mode", "prepared_ip",
            "--turboquant-prefix-seed", "5581486560434873699",
            str(candidate),
            str(root / "data" / "train.jsonl"),
        ]
        future = [token for token in plan if token != "--plan-only"]
        audit = {
            "schema": launcher.LAUNCH_AUDIT_SCHEMA,
            "created_utc": "2026-09-02T00:00:00Z",
            "actual_training_ran": False,
            "trainer_process_spawned": False,
            "all_checks_passed": True,
            "legal_scope": copy.deepcopy(launcher.EXPECTED_LEGAL_SCOPE),
            "argv_delta": copy.deepcopy(launcher.EXPECTED_ARGV_DELTA),
            "plan_argv": plan,
            "future_training_argv": future,
            "metrics_target": str(metrics),
            "parsed_workload": copy.deepcopy(launcher.EXPECTED_WORKLOAD),
            "effective_scale": copy.deepcopy(launcher.EXPECTED_EFFECTIVE_SCALE),
            "plan_result": {
                "exit_status": 0,
                "stdout_path": str(stdout),
                "stdout_sha256": launcher.sha256_file(stdout),
                "stderr_path": str(stderr),
                "stderr_sha256": launcher.sha256_file(stderr),
            },
            "sources": {
                "launcher": {"path": str(launcher_script), "sha256": launcher.sha256_file(launcher_script)},
                "launcher_test": {"path": str(launcher_test), "sha256": launcher.sha256_file(launcher_test)},
            },
            "canonical_anchor": {"pre": launcher.package_state(anchor), "post": launcher.package_state(anchor)},
            "candidate": {"target": str(candidate), "pre_plan": launcher.package_state(candidate), "post_plan": launcher.package_state(candidate)},
        }
        receipt = copy.deepcopy(audit)
        receipt["attempt_id"] = launcher.EXPECTED_RECOVERY_RECEIPT_ATTEMPT
        receipt["all_checks_passed"] = False
        receipt["parsed_workload"] = {
            "actual_train_pairs": 0,
            "batch": 4,
            "epochs": 0,
            "eval_examples": 0,
            "planned_eval_pairs": 0,
            "planned_total_pairs": 54162,
            "planned_work_units": 54162,
            "score_spectrum_aux_only_rows": 1155,
            "steps_per_epoch": 289,
            "total_train_pairs_per_epoch": 27081,
            "train": 0,
        }
        receipt["checks"] = {
            "all_pinned_path_hashes_match": True,
            "argv_delta_remove_plan_only_only": True,
            "binary_sha256_bound": True,
            "candidate_pretrain_matches_anchor": True,
            "candidate_target_differs_from_anchor": True,
            "candidate_unchanged_by_plan": True,
            "canonical_anchor_unchanged": True,
            "effective_scale_exact": True,
            "explicit_base_zero_rows_bound": True,
            "explicit_turboquant_cutoff10": True,
            "metrics_target_absent_after_plan": True,
            "no_eval_rows": True,
            "plan_exit_status_zero": True,
            "plan_only_no_trainer_spawn": True,
            "probe_binding_sidecar_materialized": True,
            "probe_manifest_sha256_bound": True,
            "train_jsonl_sha256_bound": True,
            "workload_exact": False,
        }
        receipt_path = root / "reports" / "receipt.json"
        receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        clone_inventory = {
            "schema": launcher.CLONE_SCHEMA,
            "created_utc": "2026-09-02T00:00:00Z",
            "clone_matches_anchor": True,
        }
        clone_path = root / "reports" / "clone.json"
        clone_path.write_text(json.dumps(clone_inventory, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        audit_path = root / "reports" / "launch.json"
        audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return {"root": root, "audit": audit, "audit_path": audit_path, "candidate": candidate, "anchor": anchor, "metrics": metrics, "receipt_path": receipt_path, "clone_path": clone_path, "stdout": stdout, "stderr": stderr, "launcher_test_source": launcher_test}

    def args_for(self, fixture: dict[str, object], *, execute: bool = False, expected_sha: str | None = None):
        root = fixture["root"]
        audit_path = fixture["audit_path"]
        assert isinstance(root, Path)
        assert isinstance(audit_path, Path)
        argv = [
            "--launch-audit", str(audit_path),
            "--execute" if execute else "--preflight-only",
            "--attempt-id", "test",
            "--stdout-capture", str(root / "reports" / "attempt.stdout.txt"),
            "--stderr-capture", str(root / "reports" / "attempt.stderr.txt"),
            "--output-audit", str(root / "reports" / "attempt.audit.json"),
        ]
        if expected_sha is not None:
            argv.extend(["--expected-audit-sha256", expected_sha])
        return launcher.parse_args(argv)

    def patch_globals(self, fixture: dict[str, object]):
        root = fixture["root"]
        assert isinstance(root, Path)
        self.addCleanup(setattr, launcher, "EXPECTED_ARTIFACT_SHA256", launcher.EXPECTED_ARTIFACT_SHA256)
        self.addCleanup(setattr, launcher, "EXPECTED_SOURCE_SHA256", launcher.EXPECTED_SOURCE_SHA256)
        self.addCleanup(setattr, launcher, "EXPECTED_RUNTIME_SOURCE_SHA256", launcher.EXPECTED_RUNTIME_SOURCE_SHA256)
        self.addCleanup(setattr, launcher, "EXPECTED_QRELS_SHA256", launcher.EXPECTED_QRELS_SHA256)
        self.addCleanup(setattr, launcher, "R6_BINARY", launcher.R6_BINARY)
        self.addCleanup(setattr, launcher, "CANDIDATE_PACKAGE", launcher.CANDIDATE_PACKAGE)
        self.addCleanup(setattr, launcher, "ANCHOR_PACKAGE", launcher.ANCHOR_PACKAGE)
        self.addCleanup(setattr, launcher, "TRAIN_JSONL", launcher.TRAIN_JSONL)
        self.addCleanup(setattr, launcher, "EXPECTED_BINARY_SHA256", launcher.EXPECTED_BINARY_SHA256)
        self.addCleanup(setattr, launcher, "EXPECTED_ANCHOR_ROLLUP", launcher.EXPECTED_ANCHOR_ROLLUP)
        self.addCleanup(setattr, launcher, "RECOVERY_RECEIPT", launcher.RECOVERY_RECEIPT)
        self.addCleanup(setattr, launcher, "EXPECTED_RECOVERY_RECEIPT_SHA256", launcher.EXPECTED_RECOVERY_RECEIPT_SHA256)
        self.addCleanup(setattr, launcher, "CLONE_INVENTORY", launcher.CLONE_INVENTORY)
        self.addCleanup(setattr, launcher, "PLAN_STDOUT", launcher.PLAN_STDOUT)
        self.addCleanup(setattr, launcher, "PLAN_STDERR", launcher.PLAN_STDERR)
        launcher.EXPECTED_ARTIFACT_SHA256 = {}
        launcher.EXPECTED_SOURCE_SHA256 = {}
        launcher.EXPECTED_RUNTIME_SOURCE_SHA256 = {}
        launcher.EXPECTED_QRELS_SHA256 = {}
        launcher.R6_BINARY = root / "bin" / "eos"
        launcher.CANDIDATE_PACKAGE = fixture["candidate"]
        launcher.ANCHOR_PACKAGE = fixture["anchor"]
        launcher.TRAIN_JSONL = root / "data" / "train.jsonl"
        launcher.EXPECTED_BINARY_SHA256 = launcher.sha256_file(launcher.R6_BINARY)
        launcher.EXPECTED_ANCHOR_ROLLUP = launcher.package_state(launcher.ANCHOR_PACKAGE)["content_rollup_sha256"]
        launcher.RECOVERY_RECEIPT = fixture["receipt_path"]
        launcher.EXPECTED_RECOVERY_RECEIPT_SHA256 = launcher.sha256_file(launcher.RECOVERY_RECEIPT)
        launcher.CLONE_INVENTORY = fixture["clone_path"]
        launcher.PLAN_STDOUT = fixture["stdout"]
        launcher.PLAN_STDERR = fixture["stderr"]

    def valid_metrics(self, fixture: dict[str, object]) -> dict[str, object]:
        golden = Path("runs/longembed-hn-ablate-prefix-only-20260619T063224Z/train.metrics.json")
        data = json.loads(golden.read_text(encoding="utf-8"))
        artifact = launcher.display_path(fixture["candidate"])
        data.update({"schema": launcher.TRAIN_METRICS_SCHEMA, "command": "train-embed", "mode": "train", "artifact": artifact})
        data["tokenizer"] = artifact.replace(".mll", ".tokenizer.mll")
        data["summary"].update({"epochs_completed": 2, "steps_completed": 578, "steps_run": 578, "restored_best": False, "stopped_early": False})
        data["config"].update(
            {
                "epochs": 2,
                "batch_size": 4,
                "shuffle": False,
                "seed": 191,
                "restore_best": False,
                "learning_rate": 0.000002,
                "effective_learning_rate": 0.000002,
                "contrastive_loss": "grouped_infonce",
                "temperature": 0.05,
                "eval_only": False,
                "pairwise_train": False,
                "hard_negative_train": False,
                "score_spectrum_train": True,
                "allow_research_only_score_spectrum": True,
                "score_spectrum_loss_mode": "hard_soft_recovery",
                "score_spectrum_recovery_weight": 0.05,
                "score_spectrum_recovery_top_k": 4,
                "score_spectrum_recovery_tau": 0.05,
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
                "turboquant_prefix_seed": 5581486560434873699,
                "turboquant_prefix_score_mode": "prepared_ip",
            }
        )
        data["workload"].update(
            {
                "train_mode": "score_spectrum_grouped",
                "eval_mode": "",
                "train_examples": 1155,
                "eval_examples": 0,
                "batch_size": 4,
                "planned_epochs": 2,
                "completed_epochs": 2,
                "train_batches_per_epoch": 289,
                "train_pairs_per_epoch": 27081,
                "eval_pairs_per_pass": 0,
                "planned_eval_passes": 0,
                "actual_eval_passes": 0,
                "planned_train_pairs": 54162,
                "actual_train_pairs": 54162,
                "actual_train_examples": 2310,
                "planned_eval_pairs": 0,
                "actual_eval_pairs": 0,
                "actual_eval_examples": 0,
                "planned_total_pairs": 54162,
                "actual_total_pairs": 54162,
                "actual_total_examples": 2310,
            }
        )
        data["final_train"].update({"loss": 1.25, "batch_size": 4})
        data["throughput"].update(
            {
                "elapsed_seconds": 1.0,
                "train_seconds": 1.0,
                "eval_seconds": 0.0,
                "examples_per_second": 2310.0,
                "pairs_per_second": 54162.0,
                "train_examples_per_second": 2310.0,
                "train_pairs_per_second": 54162.0,
                "eval_examples_per_second": 0.0,
                "eval_pairs_per_second": 0.0,
                "optimizer_steps_per_second": 578.0,
            }
        )
        data["accelerators"].update({"forward": "host", "optimizer": "host", "activation": "host", "contrastive": "host"})
        data["package"].update({"artifact": artifact})
        return data

    def attach_recovery(self, fixture: dict[str, object]) -> dict:
        audit = copy.deepcopy(fixture["audit"])
        args = launcher.parse_args([
            "--create-plan-audit",
            "--attempt-id", "test",
            "--allow-existing-plan-artifacts",
            "--recovery-receipt", str(fixture["receipt_path"]),
            "--expected-recovery-receipt-sha256", launcher.EXPECTED_RECOVERY_RECEIPT_SHA256,
        ])
        errors: list[str] = []
        receipt = launcher.validate_recovery_receipt(args, audit["plan_argv"], audit["future_training_argv"], {
            "train": 1155,
            "batch": 4,
            "steps_per_epoch": 289,
            "total_train_pairs_per_epoch": 27081,
            "planned_work_units": 54162,
        }, errors)
        self.assertEqual(errors, [])
        audit["recovery_receipt"] = receipt
        path = fixture["audit_path"]
        assert isinstance(path, Path)
        path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        fixture["audit"] = audit
        return audit

    def assert_rejected(self, fixture: dict[str, object], audit: dict, needle: str) -> None:
        path = fixture["audit_path"]
        assert isinstance(path, Path)
        path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        with self.assertRaises(launcher.PreflightError) as ctx:
            launcher.validate_launch_audit(self.args_for(fixture))
        self.assertIn(needle, "; ".join(ctx.exception.errors))

    def test_parse_plan_stdout_r6_counts(self) -> None:
        parsed = launcher.parse_plan_stdout("planned workload: train=1155 score_spectrum_grouped examples batch=4 steps/epoch=289 train_pairs/epoch=27081 pairs(planned=54162 actual=0) score_spectrum_activation_microbatch=0 score_spectrum_aux_only_rows=1155 turboquant_topk_recall=weight:0.5 cutoff:100 tau:0.05 margin:0 mask:q3\n")
        for key in ("train", "batch", "steps_per_epoch", "total_train_pairs_per_epoch", "planned_work_units", "planned_total_pairs", "actual_train_pairs", "eval_examples", "planned_eval_pairs", "score_spectrum_aux_only_rows"):
            self.assertEqual(parsed[key], launcher.EXPECTED_WORKLOAD[key])
        self.assertNotIn("epochs", parsed)

    def test_preflight_accepts_exact_r6_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fixture = self.make_fixture(Path(tmp))
            self.patch_globals(fixture)
            self.attach_recovery(fixture)
            result = launcher.validate_launch_audit(self.args_for(fixture))
            self.assertTrue(result["checks"]["all_checks_true"])
            self.assertFalse(result["trainer_process_spawned"])

    def make_fake_binary(self, fixture: dict[str, object], body: str) -> Path:
        root = fixture["root"]
        assert isinstance(root, Path)
        binary = root / "bin" / "eos"
        binary.write_text(body, encoding="utf-8")
        binary.chmod(binary.stat().st_mode | stat.S_IXUSR)
        launcher.EXPECTED_BINARY_SHA256 = launcher.sha256_file(binary)
        return binary

    def make_success_binary(self, fixture: dict[str, object], *, mutate: bool = True, metrics: dict[str, object] | None = None, extra: str = "") -> Path:
        candidate = fixture["candidate"]
        metrics_path = fixture["metrics"]
        assert isinstance(candidate, Path)
        assert isinstance(metrics_path, Path)
        metrics_json = json.dumps(metrics if metrics is not None else self.valid_metrics(fixture), sort_keys=True)
        mutation = f"printf 'mutated\\n' > '{candidate}'\n" if mutate else ""
        return self.make_fake_binary(
            fixture,
            f"#!/bin/sh\nprintf 'fake train\\n'\nprintf 'fake err\\n' >&2\nmkdir -p '{metrics_path.parent}'\nprintf '%s\\n' '{metrics_json}' > '{metrics_path}'\n{mutation}{extra}exit 0\n",
        )

    def test_execute_success_spawns_once_and_records_candidate_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fixture = self.make_fixture(Path(tmp))
            self.patch_globals(fixture)
            self.attach_recovery(fixture)
            candidate = fixture["candidate"]
            metrics = fixture["metrics"]
            assert isinstance(candidate, Path)
            assert isinstance(metrics, Path)
            self.make_success_binary(fixture)
            status, audit = launcher.execute_once(self.args_for(fixture, execute=True, expected_sha=launcher.sha256_file(fixture["audit_path"])), launcher.validate_launch_audit(self.args_for(fixture, execute=True, expected_sha=launcher.sha256_file(fixture["audit_path"]))))
            self.assertEqual(status, 0)
            self.assertTrue(audit["checks"]["all_checks_true"])
            self.assertEqual(audit["spawn_count"], 1)
            self.assertEqual(audit["retry_count"], 0)
            self.assertTrue(audit["actual_training_ran"])
            self.assertIn("metrics", audit["execute_result"])
            self.assertNotEqual(audit["post_execute"]["candidate"]["content_rollup_sha256"], audit["pre_spawn"]["candidate"]["content_rollup_sha256"])

    def test_execute_nonzero_records_failure_without_retry(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fixture = self.make_fixture(Path(tmp))
            self.patch_globals(fixture)
            self.attach_recovery(fixture)
            self.make_fake_binary(fixture, "#!/bin/sh\nprintf 'bad\\n'\nexit 7\n")
            preflight = launcher.validate_launch_audit(self.args_for(fixture, execute=True, expected_sha=launcher.sha256_file(fixture["audit_path"])))
            status, audit = launcher.execute_once(self.args_for(fixture, execute=True, expected_sha=launcher.sha256_file(fixture["audit_path"])), preflight)
            self.assertEqual(status, 7)
            self.assertEqual(audit["spawn_count"], 1)
            self.assertEqual(audit["retry_count"], 0)
            self.assertIn("trainer exited nonzero: 7", audit["errors"])

    def test_execute_exit0_requires_valid_metrics(self) -> None:
        cases = [
            ("missing", "#!/bin/sh\nprintf 'mutated\\n' > '{candidate}'\nexit 0\n", "execute metrics missing after exit0"),
            ("malformed", "#!/bin/sh\nmkdir -p '{metrics_parent}'\nprintf 'not-json\\n' > '{metrics}'\nprintf 'mutated\\n' > '{candidate}'\nexit 0\n", "execute metrics JSON parse error"),
            ("wrong", None, "execute metrics validation failed: artifact"),
        ]
        for name, body_template, needle in cases:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as tmp:
                fixture = self.make_fixture(Path(tmp))
                self.patch_globals(fixture)
                self.attach_recovery(fixture)
                candidate = fixture["candidate"]
                metrics = fixture["metrics"]
                assert isinstance(candidate, Path)
                assert isinstance(metrics, Path)
                if body_template is None:
                    wrong = self.valid_metrics(fixture)
                    wrong["artifact"] = "wrong-artifact.mll"
                    self.make_success_binary(fixture, metrics=wrong)
                else:
                    self.make_fake_binary(fixture, body_template.format(candidate=candidate, metrics=metrics, metrics_parent=metrics.parent))
                args = self.args_for(fixture, execute=True, expected_sha=launcher.sha256_file(fixture["audit_path"]))
                status, audit = launcher.execute_once(args, launcher.validate_launch_audit(args))
                self.assertEqual(status, 1)
                self.assertEqual(audit["spawn_count"], 1)
                self.assertFalse(audit["checks"]["all_checks_true"])
                self.assertIn(needle, "; ".join(audit["errors"]))
                self.assertTrue(Path(args.output_audit).is_file())

    def test_execute_metrics_accepts_real_schema_shape_and_rejects_drift(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fixture = self.make_fixture(Path(tmp))
            self.patch_globals(fixture)
            self.attach_recovery(fixture)
            metrics_path = fixture["metrics"]
            assert isinstance(metrics_path, Path)
            metrics_path.parent.mkdir(parents=True)
            metrics_path.write_text(json.dumps(self.valid_metrics(fixture), indent=2, allow_nan=False) + "\n", encoding="utf-8")
            errors: list[str] = []
            record = launcher.validate_execute_metrics(str(metrics_path), self.args_for(fixture, execute=True, expected_sha=launcher.sha256_file(fixture["audit_path"])), errors)
            self.assertEqual(errors, [])
            self.assertTrue(record["ok"])
            self.assertNotIn("score_spectrum_recovery_margin", record["json"]["config"])
            self.assertNotIn("turboquant_topk_recall_margin", record["json"]["config"])

        mutations = [
            ("old invented schema", lambda data: data.update({"schema": "eos.q3_r6_frontier_retention.training_metrics.v1"}), "schema"),
            ("missing artifact", lambda data: data.pop("artifact"), "artifact"),
            ("wrong learning_rate", lambda data: data["config"].update({"learning_rate": 0.001}), "config_learning_rate"),
            ("wrong workload", lambda data: data["workload"].update({"planned_total_pairs": 1}), "workload_planned_total_pairs"),
            ("wrong steps", lambda data: data["summary"].update({"steps_completed": 1}), "summary_steps_completed"),
            ("wrong eval", lambda data: data["workload"].update({"actual_eval_pairs": 1}), "workload_actual_eval_pairs_zero"),
            ("wrong q3 mask", lambda data: data["config"].update({"turboquant_topk_negative_mask": "q5"}), "config_turboquant_topk_negative_mask"),
            ("nan throughput", lambda data: data["throughput"].update({"elapsed_seconds": float("nan")}), "nonfinite numeric value"),
            ("negative counter", lambda data: data["throughput"].update({"pairs_per_second": -1}), "throughput_pairs_per_second_finite_nonnegative"),
        ]
        for name, mutate, needle in mutations:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as tmp:
                fixture = self.make_fixture(Path(tmp))
                self.patch_globals(fixture)
                self.attach_recovery(fixture)
                metrics_path = fixture["metrics"]
                assert isinstance(metrics_path, Path)
                data = self.valid_metrics(fixture)
                mutate(data)
                metrics_path.parent.mkdir(parents=True)
                metrics_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
                errors = []
                launcher.validate_execute_metrics(str(metrics_path), self.args_for(fixture, execute=True, expected_sha=launcher.sha256_file(fixture["audit_path"])), errors)
                self.assertIn(needle, "; ".join(errors))

    def test_execute_metrics_optional_zero_omitempty_fields(self) -> None:
        optional_zero_fields = [
            ("score_spectrum_recovery_margin", "config_score_spectrum_recovery_margin_absent_or_zero"),
            ("turboquant_topk_recall_margin", "config_turboquant_topk_recall_margin_absent_or_zero"),
        ]
        for field, check_name in optional_zero_fields:
            with self.subTest(field=field, value="absent"), tempfile.TemporaryDirectory() as tmp:
                fixture = self.make_fixture(Path(tmp))
                self.patch_globals(fixture)
                self.attach_recovery(fixture)
                metrics_path = fixture["metrics"]
                assert isinstance(metrics_path, Path)
                data = self.valid_metrics(fixture)
                data["config"].pop(field, None)
                metrics_path.parent.mkdir(parents=True)
                metrics_path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")
                errors: list[str] = []
                record = launcher.validate_execute_metrics(str(metrics_path), self.args_for(fixture, execute=True, expected_sha=launcher.sha256_file(fixture["audit_path"])), errors)
                self.assertEqual(errors, [])
                self.assertTrue(record["checks"][check_name])
            with self.subTest(field=field, value="explicit-zero"), tempfile.TemporaryDirectory() as tmp:
                fixture = self.make_fixture(Path(tmp))
                self.patch_globals(fixture)
                self.attach_recovery(fixture)
                metrics_path = fixture["metrics"]
                assert isinstance(metrics_path, Path)
                data = self.valid_metrics(fixture)
                data["config"][field] = 0
                metrics_path.parent.mkdir(parents=True)
                metrics_path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")
                errors = []
                record = launcher.validate_execute_metrics(str(metrics_path), self.args_for(fixture, execute=True, expected_sha=launcher.sha256_file(fixture["audit_path"])), errors)
                self.assertEqual(errors, [])
                self.assertTrue(record["checks"][check_name])
            for bad_value in (0.1, "0", None, float("nan")):
                with self.subTest(field=field, value=repr(bad_value)), tempfile.TemporaryDirectory() as tmp:
                    fixture = self.make_fixture(Path(tmp))
                    self.patch_globals(fixture)
                    self.attach_recovery(fixture)
                    metrics_path = fixture["metrics"]
                    assert isinstance(metrics_path, Path)
                    data = self.valid_metrics(fixture)
                    data["config"][field] = bad_value
                    metrics_path.parent.mkdir(parents=True)
                    metrics_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
                    errors = []
                    launcher.validate_execute_metrics(str(metrics_path), self.args_for(fixture, execute=True, expected_sha=launcher.sha256_file(fixture["audit_path"])), errors)
                    self.assertIn(check_name, "; ".join(errors))

    def test_execute_audit_survives_post_run_package_corruption(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fixture = self.make_fixture(Path(tmp))
            self.patch_globals(fixture)
            self.attach_recovery(fixture)
            candidate = fixture["candidate"]
            assert isinstance(candidate, Path)
            tokenizer = candidate.with_name("d384-pre.tokenizer.mll")
            self.make_success_binary(fixture, extra=f"rm '{tokenizer}'\n")
            args = self.args_for(fixture, execute=True, expected_sha=launcher.sha256_file(fixture["audit_path"]))
            status, audit = launcher.execute_once(args, launcher.validate_launch_audit(args))
            self.assertEqual(status, 1)
            self.assertEqual(audit["spawn_count"], 1)
            self.assertFalse(audit["checks"]["all_checks_true"])
            self.assertIn("post execute candidate package state error", "; ".join(audit["errors"]))
            self.assertTrue(Path(args.output_audit).is_file())

    def test_execute_exit0_no_candidate_mutation_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fixture = self.make_fixture(Path(tmp))
            self.patch_globals(fixture)
            self.attach_recovery(fixture)
            self.make_success_binary(fixture, mutate=False)
            args = self.args_for(fixture, execute=True, expected_sha=launcher.sha256_file(fixture["audit_path"]))
            status, audit = launcher.execute_once(args, launcher.validate_launch_audit(args))
            self.assertEqual(status, 1)
            self.assertEqual(audit["spawn_count"], 1)
            self.assertFalse(audit["checks"]["candidate_changed_after_success"])
            self.assertIn("trainer exit0 produced no candidate package mutation", audit["errors"])

    def test_execute_final_gate_rejects_last_second_input_drift_without_spawn(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fixture = self.make_fixture(Path(tmp))
            self.patch_globals(fixture)
            self.attach_recovery(fixture)
            self.make_success_binary(fixture)
            args = self.args_for(fixture, execute=True, expected_sha=launcher.sha256_file(fixture["audit_path"]))
            preflight = launcher.validate_launch_audit(args)
            binary = fixture["root"] / "bin" / "eos"
            assert isinstance(binary, Path)
            binary.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
            binary.chmod(binary.stat().st_mode | stat.S_IXUSR)
            status, audit = launcher.execute_once(args, preflight)
            self.assertEqual(status, 1)
            self.assertEqual(audit["spawn_count"], 0)
            self.assertFalse(audit["trainer_process_spawned"])
            self.assertIn("final gate pinned path sha256 mismatch", "; ".join(audit["errors"]))
            self.assertTrue(Path(args.output_audit).is_file())

    def test_preflight_execute_rejects_stale_paths_escape_and_second_execution(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fixture = self.make_fixture(Path(tmp))
            self.patch_globals(fixture)
            self.attach_recovery(fixture)
            write(Path(tmp) / "reports" / "attempt.stdout.txt", "old\n")
            with self.assertRaises(launcher.PreflightError) as ctx:
                launcher.validate_launch_audit(self.args_for(fixture, execute=True, expected_sha=launcher.sha256_file(fixture["audit_path"])))
            self.assertIn("stdout_capture already exists", "; ".join(ctx.exception.errors))
        with tempfile.TemporaryDirectory() as tmp:
            fixture = self.make_fixture(Path(tmp))
            self.patch_globals(fixture)
            self.attach_recovery(fixture)
            args = launcher.parse_args([
                "--launch-audit", str(fixture["audit_path"]),
                "--execute",
                "--expected-audit-sha256", launcher.sha256_file(fixture["audit_path"]),
                "--attempt-id", "test",
                "--stdout-capture", str(Path(tmp).parent / "escape.stdout"),
                "--stderr-capture", str(Path(tmp) / "reports" / "attempt.stderr.txt"),
                "--output-audit", str(Path(tmp) / "reports" / "attempt.audit.json"),
            ])
            with self.assertRaises(launcher.PreflightError) as ctx:
                launcher.validate_launch_audit(args)
            self.assertIn("stdout_capture path escapes reports root", "; ".join(ctx.exception.errors))
        with tempfile.TemporaryDirectory() as tmp:
            fixture = self.make_fixture(Path(tmp))
            self.patch_globals(fixture)
            self.attach_recovery(fixture)
            self.make_success_binary(fixture)
            args = self.args_for(fixture, execute=True, expected_sha=launcher.sha256_file(fixture["audit_path"]))
            preflight = launcher.validate_launch_audit(args)
            status, _audit = launcher.execute_once(args, preflight)
            self.assertEqual(status, 0)
            with self.assertRaises(launcher.PreflightError) as ctx:
                launcher.validate_launch_audit(args)
            self.assertIn("stdout_capture already exists", "; ".join(ctx.exception.errors))

    def test_rejects_argv_and_workload_drift(self) -> None:
        cases = [
            ("--turboquant-topk-cutoff", "missing required flag"),
            ("--turboquant-topk-recall-weight", "missing required flag"),
            ("--turboquant-prefix-seed", "missing required flag"),
            ("--epochs", "missing required flag"),
        ]
        for flag, needle in cases:
            with tempfile.TemporaryDirectory() as tmp:
                fixture = self.make_fixture(Path(tmp))
                self.patch_globals(fixture)
                self.attach_recovery(fixture)
                audit = copy.deepcopy(fixture["audit"])
                idx = audit["plan_argv"].index(flag)
                del audit["plan_argv"][idx:idx + 2]
                audit["future_training_argv"] = [token for token in audit["plan_argv"] if token != "--plan-only"]
                self.assert_rejected(fixture, audit, needle)
        with tempfile.TemporaryDirectory() as tmp:
            fixture = self.make_fixture(Path(tmp))
            self.patch_globals(fixture)
            self.attach_recovery(fixture)
            audit = copy.deepcopy(fixture["audit"])
            audit["parsed_workload"]["planned_total_pairs"] = 54161
            self.assert_rejected(fixture, audit, "parsed workload drift")

    def test_rejects_hash_legal_path_and_stale_output_drift(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fixture = self.make_fixture(Path(tmp))
            self.patch_globals(fixture)
            self.attach_recovery(fixture)
            audit = copy.deepcopy(fixture["audit"])
            audit["legal_scope"]["commercial_use_allowed"] = True
            self.assert_rejected(fixture, audit, "legal_scope drift")
        with tempfile.TemporaryDirectory() as tmp:
            fixture = self.make_fixture(Path(tmp))
            self.patch_globals(fixture)
            self.attach_recovery(fixture)
            audit = copy.deepcopy(fixture["audit"])
            audit["future_training_argv"][-2] = str(fixture["anchor"])
            self.assert_rejected(fixture, audit, "future_training_argv missing candidate target")
        with tempfile.TemporaryDirectory() as tmp:
            fixture = self.make_fixture(Path(tmp))
            self.patch_globals(fixture)
            self.attach_recovery(fixture)
            metrics = fixture["metrics"]
            assert isinstance(metrics, Path)
            write(metrics, "{}\n")
            with self.assertRaises(launcher.PreflightError) as ctx:
                launcher.validate_launch_audit(self.args_for(fixture))
            self.assertIn("metrics target already exists", "; ".join(ctx.exception.errors))
        with tempfile.TemporaryDirectory() as tmp:
            fixture = self.make_fixture(Path(tmp))
            self.patch_globals(fixture)
            self.attach_recovery(fixture)
            launcher_test_source = fixture["launcher_test_source"]
            assert isinstance(launcher_test_source, Path)
            launcher_test_source.write_text("drifted test launcher\n", encoding="utf-8")
            with self.assertRaises(launcher.PreflightError) as ctx:
                launcher.validate_launch_audit(self.args_for(fixture, execute=True, expected_sha=launcher.sha256_file(fixture["audit_path"])))
            self.assertIn("launch audit bound path sha256 mismatch: sources.launcher_test", "; ".join(ctx.exception.errors))

    def test_strict_stdout_parser_rejects_eval_duplicate_and_truncated(self) -> None:
        base = "planned workload: train=1155 score_spectrum_grouped examples batch=4 steps/epoch=289 train_pairs/epoch=27081 pairs(planned=54162 actual=0) score_spectrum_activation_microbatch=0 score_spectrum_aux_only_rows=1155 turboquant_topk_recall=weight:0.5 cutoff:100 tau:0.05 margin:0 mask:q3\n"
        bad_cases = [
            (base.replace("batch=4", "eval_examples=99 batch=4"), "nonzero eval work"),
            (base.replace("train=1155", "train=1155 train=1154"), "duplicate plan stdout field: train"),
            (base.replace("pairs(planned=54162 actual=0)", ""), "pairs"),
            ("planned workload: train=1155 batch=4\n", "steps_per_epoch"),
        ]
        for text, needle in bad_cases:
            with self.subTest(needle=needle):
                with self.assertRaises(ValueError) as ctx:
                    launcher.parse_plan_stdout(text)
                self.assertIn(needle, str(ctx.exception))

    def test_recovery_receipt_rejects_drift(self) -> None:
        mutations = [
            ("sha", lambda fixture, receipt: receipt, "recovery receipt sha256 mismatch", "0" * 64),
            ("status", lambda fixture, receipt: receipt["plan_result"].update({"exit_status": 2}), "plan_spawned_once_returned_zero", None),
            ("capture", lambda fixture, receipt: receipt["plan_result"].update({"stdout_sha256": "1" * 64}), "stdout_sha_exact", None),
            ("argv", lambda fixture, receipt: receipt["plan_argv"].append("--bogus"), "plan_argv_exact", None),
            ("timestamp", lambda fixture, receipt: receipt.update({"created_utc": "2025-01-01T00:00:00Z"}), "timestamp_order", None),
            ("package", lambda fixture, receipt: receipt["candidate"]["post_plan"].update({"content_rollup_sha256": "0" * 64}), "candidate_pre_post_stable", None),
        ]
        for _name, mutate, needle, override_sha in mutations:
            with self.subTest(needle=needle), tempfile.TemporaryDirectory() as tmp:
                fixture = self.make_fixture(Path(tmp))
                self.patch_globals(fixture)
                receipt_path = fixture["receipt_path"]
                assert isinstance(receipt_path, Path)
                receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
                maybe = mutate(fixture, receipt)
                if maybe is not None:
                    receipt = maybe
                receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
                if override_sha is None:
                    launcher.EXPECTED_RECOVERY_RECEIPT_SHA256 = launcher.sha256_file(receipt_path)
                args = launcher.parse_args([
                    "--create-plan-audit",
                    "--attempt-id", "test",
                    "--allow-existing-plan-artifacts",
                    "--recovery-receipt", str(receipt_path),
                    "--expected-recovery-receipt-sha256", override_sha or launcher.EXPECTED_RECOVERY_RECEIPT_SHA256,
                ])
                errors: list[str] = []
                launcher.validate_recovery_receipt(args, fixture["audit"]["plan_argv"], fixture["audit"]["future_training_argv"], {
                    "train": 1155,
                    "batch": 4,
                    "steps_per_epoch": 289,
                    "total_train_pairs_per_epoch": 27081,
                    "planned_work_units": 54162,
                }, errors)
                self.assertIn(needle, "; ".join(errors))

    def test_execute_uses_popen_and_no_retry_loop(self) -> None:
        source = inspect.getsource(launcher.execute_once)
        self.assertIn("subprocess.Popen(argv", source)
        self.assertNotIn("for attempt", source)
        self.assertNotIn("while ", source)


if __name__ == "__main__":
    unittest.main()
