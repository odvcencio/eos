#!/usr/bin/env python3
"""Tests for the q3 R4 broad-gap dataset builder."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
SCRIPT = SCRIPT_DIR / "build_q3_r4_broad_gap_dataset.py"
LEGAL_GATES = {
    "train_allowed_for_research": True,
    "release_train_allowed": False,
    "commercial_use_allowed": False,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")


def qrels(path: Path, rows: list[tuple[str, str, int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "query-id\tcorpus-id\tscore\n" + "".join(f"{qid}\t{doc}\t{score}\n" for qid, doc, score in rows),
        encoding="utf-8",
    )


def q3_row(dataset: str, qid: str, positives: dict[int, str], *, top10_positive: str | None = None, high_gap: bool = False) -> dict:
    top_k = []
    for rank in range(1, 101):
        if rank in positives:
            doc_id = positives[rank]
            relevance = 1
        elif rank == 1 and top10_positive:
            doc_id = top10_positive
            relevance = 1
        else:
            doc_id = f"{qid}-N{rank}"
            relevance = 0
        score = 1.0 - rank / 1000.0
        if high_gap and rank in positives:
            score -= 0.1
        top_k.append(
            {
                "rank": rank,
                "doc_id": doc_id,
                "score": score,
                "relevance": relevance,
                "dense_rank": rank,
                "dense_score": score,
                "compact_rank": rank,
                "compact_score": score,
            }
        )
    return {
        "schema": "fixture",
        "dataset": dataset,
        "query_id": qid,
        "method": "turboquant_ip_b3",
        "bits": 3,
        "scoring_surface": "turboquant_ip_prepared",
        "quantizer_seed": 5581486560434873699,
        "candidate_count": 100,
        "candidates_scored": 100,
        "pruning_supported": True,
        "pruning_used": True,
        "quality": {},
        "top_k": top_k,
    }


def sentinel_row(release_allowed: bool = False) -> dict:
    docs = ["MED-5329", "MED-4523"] + [f"MED-S{i}" for i in range(3, 101)]
    gains = [1.0, 1.0] + [0.0] * 98
    return {
        "row_id": "nfcorpus:PLAIN-63:nfrecall24-q3top100",
        "source": "nfcorpus:fixture",
        "query": "plain 63",
        "candidate_doc_ids": [f"nfcorpus:{doc}" for doc in docs],
        "candidate_texts": [f"sentinel text {doc}" for doc in docs],
        "candidate_sources": ["qrel", "qrel"] + ["q3"] * 98,
        "qrel_gains": gains,
        "positive_indexes": [0, 1],
        "selected_positive_index": 0,
        "hard_negative_eligible": [False, False] + [True] * 98,
        "target_probabilities": [0.5, 0.5] + [0.0] * 98,
        "hard_loss_weight": 0.0,
        "soft_loss_weight": 0.0,
        "recovery_loss_weight": 0.0,
        "turboquant_topk_loss_weight": 0.0,
        "turboquant_topk_recall_loss_weight": 1.0,
        "train_policy": "fixture",
        "selection_bucket": "nfcorpus_recall_sentinel_q3top100_rank80_100",
        "guard_margin": None,
        "legal_gates": dict(LEGAL_GATES),
        **(LEGAL_GATES | {"release_train_allowed": release_allowed}),
        "source_artifact_hash": "fixture",
        "source_dataset": "nfcorpus",
        "source_query_id": "PLAIN-63",
        "source_positive_doc_ids": ["MED-5329", "MED-4523"],
        "candidate_positive_doc_ids_pre_dedupe": ["MED-5329", "MED-4523"],
        "all_train_qrel_positive_doc_ids": ["MED-5329", "MED-4523"],
        "positive_count": 2,
        "bm25_negative_count": 0,
        "q3_negative_count": 98,
        "extra_negative_count": 0,
        "q3_evidence": [],
        "anchor_q3_candidate_evidence": [
            {"doc_id": doc, "q3_rank": i + 1, "q3_score": 1.0 - (i + 1) / 1000.0, "relevance": int(gains[i])}
            for i, doc in enumerate(docs)
        ],
    }


def validate_contract(manifest: dict) -> None:
    contract = manifest["go_loader_plan"]
    required = contract["required_flags"]
    workload = contract["workload"]
    if contract["ingestion"]["forbidden_no_tokenizer"] is not True:
        raise AssertionError("forbidden_no_tokenizer drift")
    if "--no-tokenizer" not in contract["ingestion"]["forbidden_flags"]:
        raise AssertionError("forbidden --no-tokenizer missing")
    expected_flags = {
        "score_spectrum_train": True,
        "allow_research_only_score_spectrum": True,
        "score_spectrum_loss_mode": "hard_soft_recovery",
        "score_spectrum_recovery_weight": 0.05,
        "score_spectrum_recovery_margin": 0.0,
        "score_spectrum_recovery_top_k": 4,
        "score_spectrum_recovery_tau": 0.05,
        "contrastive_loss": "grouped_infonce",
        "temperature": 0.05,
        "turboquant_prefix_seed": 5581486560434873699,
        "turboquant_topk_objectives": "fullDim:3=0.04",
        "turboquant_topk_loss": "lambdandcg",
        "turboquant_topk_cutoff": 10,
        "turboquant_topk_tau": 0.05,
        "turboquant_topk_margin": 0.002,
        "turboquant_topk_negative_mask": "q3",
        "turboquant_topk_recall_weight": 0.15,
        "turboquant_topk_recall_cutoff": 100,
        "turboquant_topk_recall_tau": 0.05,
        "turboquant_topk_recall_margin": 0.0,
        "turboquant_topk_recall_negative_mask": "q3",
        "lr": 0.000002,
        "epochs": 2,
        "batch_size": 4,
        "seed": 191,
        "restore_best": False,
    }
    for key, want in expected_flags.items():
        if required.get(key) != want:
            raise AssertionError(f"required flag {key} drift: {required.get(key)!r} != {want!r}")
    for key in ("train", "batch", "steps_per_epoch", "planned_pairs", "actual_train_pairs"):
        if key not in workload:
            raise AssertionError(f"workload missing {key}")
    if contract["legal_gates"] != LEGAL_GATES or contract["quality_claim"] is not False:
        raise AssertionError("legal contract drift")
    if contract["metrics_target"] == "":
        raise AssertionError("metrics target missing")
    if "--metrics-json" not in contract["required_plan_argv_tokens"]:
        raise AssertionError("plan argv missing metrics target")
    if "--metrics-json" not in contract["required_training_argv_tokens"]:
        raise AssertionError("training argv missing metrics target")
    if "--no-tokenizer" in contract["required_plan_argv_tokens"] or "--no-tokenizer" in contract["required_training_argv_tokens"]:
        raise AssertionError("no-tokenizer leaked into argv")
    if contract["argv_delta"] != {
        "removed_for_training": ["--plan-only"],
        "added_for_training": [],
        "exactly_one_semantic_delta": True,
        "semantic_delta": "remove --plan-only only",
    }:
        raise AssertionError(f"argv delta drift: {contract['argv_delta']!r}")
    anchor = contract["anchor"]
    if not anchor["anchor_package"].endswith("d384-pre.mll"):
        raise AssertionError("anchor package drift")
    if not anchor["tokenizer_package"].endswith("d384-pre.tokenizer.mll"):
        raise AssertionError("tokenizer package drift")
    if anchor["sibling_count"] < 2:
        raise AssertionError("anchor sibling rollup incomplete")
    candidate = contract["candidate"]
    if candidate["candidate_target_differs_from_anchor"] is not True:
        raise AssertionError("candidate target does not differ from anchor")
    if candidate["candidate_package"] == anchor["anchor_package"]:
        raise AssertionError("candidate package equals anchor")
    artifacts = contract["artifact_hashes"]
    for key in ("train_score_spectrum", "coverage_report", "manifest"):
        if key not in artifacts:
            raise AssertionError(f"artifact hash missing {key}")


def validate_launch_audit(audit: dict) -> None:
    if audit["schema"] != "eos.q3_r4_broad_gap.training_launch_audit.v1":
        raise AssertionError("launch audit schema drift")
    if audit["actual_training_ran"] is not False:
        raise AssertionError("launch audit recorded actual training")
    if audit["legal_scope"] != {
        "research_only": True,
        "release_train_allowed": False,
        "commercial_use_allowed": False,
        "quality_claim": False,
        "no_dev_reserve_official_eval": True,
    }:
        raise AssertionError("launch legal scope drift")
    if audit["candidate"]["target"] == audit["canonical_anchor"]["pre"]["target"]:
        raise AssertionError("candidate target equals canonical anchor")
    if audit["candidate"]["pre_plan"]["content_rollup_sha256"] != audit["canonical_anchor"]["pre"]["content_rollup_sha256"]:
        raise AssertionError("candidate pretrain rollup differs from anchor")
    if audit["candidate"]["pre_plan"]["sibling_hashes"] != audit["canonical_anchor"]["pre"]["sibling_hashes"]:
        raise AssertionError("candidate sibling hashes differ from anchor")
    if audit["candidate"]["post_plan"]["content_rollup_sha256"] != audit["candidate"]["pre_plan"]["content_rollup_sha256"]:
        raise AssertionError("candidate changed during plan")
    if audit["canonical_anchor"]["post"]["content_rollup_sha256"] != audit["canonical_anchor"]["pre"]["content_rollup_sha256"]:
        raise AssertionError("canonical anchor changed during plan")
    if audit["argv_delta"] != {
        "removed_for_training": ["--plan-only"],
        "added_for_training": [],
        "exactly_one_semantic_delta": True,
        "semantic_delta": "remove --plan-only only",
    }:
        raise AssertionError("launch argv delta drift")
    plan_argv = audit["plan_argv"]
    train_argv = audit["future_training_argv"]
    metrics_target = audit["metrics_target"]
    if "--metrics-json" not in plan_argv or "--metrics-json" not in train_argv:
        raise AssertionError("metrics target flag missing from launch argv")
    if metrics_target not in plan_argv or metrics_target not in train_argv:
        raise AssertionError("metrics target path missing from launch argv")
    if "--no-tokenizer" in plan_argv or "--no-tokenizer" in train_argv:
        raise AssertionError("forbidden --no-tokenizer present in launch argv")
    if audit["plan_result"]["exit_status"] != 0:
        raise AssertionError("plan-only failed")
    if audit["parsed_workload"] != {
        "train": 134,
        "batch": 4,
        "steps_per_epoch": 34,
        "train_pairs_per_epoch": 46912,
        "planned_pairs": 93824,
        "actual_train_pairs": 0,
    }:
        raise AssertionError("parsed workload drift")
    checks = audit["checks"]
    required_checks = (
        "candidate_target_differs_from_anchor",
        "candidate_pretrain_matches_anchor",
        "candidate_unchanged_by_plan",
        "canonical_anchor_unchanged",
        "metrics_target_in_plan_argv",
        "metrics_target_in_training_argv",
        "no_no_tokenizer_in_plan_or_training_argv",
        "argv_delta_remove_plan_only_only",
        "metrics_target_absent_after_plan",
        "workload_exact",
        "all_contract_checks_true",
    )
    for check in required_checks:
        if checks.get(check) is not True:
            raise AssertionError(f"launch check failed: {check}")
    if audit["all_checks_passed"] is not True:
        raise AssertionError("launch audit aggregate check is false")


class BuildQ3R4BroadGapDatasetTest(unittest.TestCase):
    def fixture(self, root: Path, *, illegal_sentinel: bool = False) -> dict[str, Path]:
        dataset_root = root / "raw"
        q3_paths: dict[str, Path] = {}
        rows_by_dataset = {
            "fiqa": [
                q3_row("fiqa", "f-excluded", {11: "f-excluded-P11"}),
                q3_row("fiqa", "f1", {11: "f1-P11"}),
                q3_row("fiqa", "f2", {12: "f2-P12"}),
                q3_row("fiqa", "fg1", {}, top10_positive="fg1-P1"),
                q3_row("fiqa", "fg2", {}, top10_positive="fg2-P1"),
            ],
            "nfcorpus": [
                q3_row("nfcorpus", "n-alias", {11: "n-alias-P11"}),
                q3_row("nfcorpus", "n1", {11: "n1-P11"}),
                q3_row("nfcorpus", "n2", {12: "n2-P12"}),
                q3_row("nfcorpus", "n3", {13: "n3-P13"}),
            ],
            "scifact": [
                q3_row("scifact", "243", {11: "s243-P11"}),
                q3_row("scifact", "sg1", {}, top10_positive="sg1-P1"),
                q3_row("scifact", "sg2", {}, top10_positive="sg2-P1"),
            ],
        }
        for dataset, rows in rows_by_dataset.items():
            q3_path = root / f"{dataset}.q3.jsonl"
            write_jsonl(q3_path, rows)
            q3_paths[dataset] = q3_path
            base = dataset_root / dataset / dataset
            query_ids = [row["query_id"] for row in rows]
            write_jsonl(base / "queries.jsonl", [{"_id": qid, "text": f"query {qid}"} for qid in query_ids])
            docs = {}
            qrel_rows = []
            for row in rows:
                qid = row["query_id"]
                for candidate in row["top_k"]:
                    doc_id = candidate["doc_id"]
                    docs[doc_id] = {"_id": doc_id, "title": "", "text": f"text {doc_id}"}
                    if candidate["relevance"] > 0:
                        qrel_rows.append((qid, doc_id, 1))
            if dataset == "nfcorpus" and "n-alias-P11" in docs:
                for rank in range(1, 11):
                    alias_doc = f"n-alias-N{rank}"
                    if alias_doc in docs:
                        docs[alias_doc]["text"] = docs["n-alias-P11"]["text"]
            write_jsonl(base / "corpus.jsonl", list(docs.values()))
            qrels(base / "qrels" / "train.tsv", qrel_rows)
            qrels(base / "qrels" / "test.tsv", [("heldout", "D", 1)])
        dev = root / "dev.json"
        reserve = root / "reserve.json"
        write_json(dev, {"selected_qids": {"fiqa": ["f-excluded"], "nfcorpus": [], "scifact": []}})
        write_json(reserve, {"selected_qids": {"fiqa": [], "nfcorpus": [], "scifact": []}})
        parent = root / "parent.jsonl"
        write_jsonl(parent, [sentinel_row(release_allowed=illegal_sentinel)])
        parent_manifest = root / "parent.manifest.json"
        parent_coverage = root / "parent.coverage.json"
        write_json(parent_manifest, {"schema": "fixture", "legal_gates": LEGAL_GATES})
        write_json(parent_coverage, {"schema": "fixture", "legal_gates": LEGAL_GATES})
        anchor_dir = root / "packages"
        anchor = anchor_dir / "d384-pre.mll"
        tokenizer = anchor_dir / "d384-pre.tokenizer.mll"
        anchor_dir.mkdir(parents=True, exist_ok=True)
        anchor.write_bytes(b"fixture anchor package\n")
        tokenizer.write_bytes(b"fixture tokenizer package\n")
        return {
            "dataset_root": dataset_root,
            "fiqa_q3": q3_paths["fiqa"],
            "nfcorpus_q3": q3_paths["nfcorpus"],
            "scifact_q3": q3_paths["scifact"],
            "dev": dev,
            "reserve": reserve,
            "parent": parent,
            "parent_manifest": parent_manifest,
            "parent_coverage": parent_coverage,
            "anchor": anchor,
            "out": root / "out.jsonl",
            "manifest": root / "manifest.json",
            "coverage": root / "coverage.json",
        }

    def run_builder(self, paths: dict[str, Path], *extra: str, check: bool = True) -> subprocess.CompletedProcess:
        return subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--dataset-root",
                str(paths["dataset_root"]),
                "--q3-per-query-jsonl",
                str(paths["fiqa_q3"]),
                "--q3-per-query-jsonl",
                str(paths["nfcorpus_q3"]),
                "--q3-per-query-jsonl",
                str(paths["scifact_q3"]),
                "--expected-q3-sha256",
                f"fiqa={sha256_file(paths['fiqa_q3'])}",
                "--expected-q3-sha256",
                f"nfcorpus={sha256_file(paths['nfcorpus_q3'])}",
                "--expected-q3-sha256",
                f"scifact={sha256_file(paths['scifact_q3'])}",
                "--parent-sentinel-train-jsonl",
                str(paths["parent"]),
                "--expected-parent-sentinel-train-sha256",
                sha256_file(paths["parent"]),
                "--parent-sentinel-manifest",
                str(paths["parent_manifest"]),
                "--parent-sentinel-coverage",
                str(paths["parent_coverage"]),
                "--anchor-package",
                str(paths["anchor"]),
                "--exclude-qids-json",
                str(paths["dev"]),
                "--exclude-qids-json",
                str(paths["reserve"]),
                "--fiqa-promotion-count",
                "2",
                "--nfcorpus-promotion-count",
                "2",
                "--expected-nfcorpus-eligible",
                "3",
                "--expected-scifact-eligible",
                "1",
                "--guard-count-per-dataset",
                "2",
                "--sentinel-count",
                "1",
                "--expected-rows",
                "10",
                "--expected-promotion-total",
                "5",
                "--expected-guard-total",
                "4",
                "--expected-steps-per-epoch",
                "3",
                "--output-jsonl",
                str(paths["out"]),
                "--manifest",
                str(paths["manifest"]),
                "--coverage-report",
                str(paths["coverage"]),
                *extra,
            ],
            text=True,
            capture_output=True,
            check=check,
        )

    def test_builds_expected_buckets_weights_plain63_and_replays_deterministically(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = self.fixture(Path(tmp))
            self.run_builder(paths)
            first_sha = sha256_file(paths["out"])
            rows = [json.loads(line) for line in paths["out"].read_text(encoding="utf-8").splitlines()]
            buckets = {bucket: sum(1 for row in rows if row["selection_bucket"] == bucket) for bucket in {row["selection_bucket"] for row in rows}}
            self.assertEqual(buckets["promotion_11_20_to_top10_near_gap"], 5)
            self.assertEqual(buckets["guard_top10_margin"], 4)
            self.assertEqual(buckets["nfcorpus_recall_sentinel_q3top100_rank80_100"], 1)
            self.assertTrue(any(row["source_query_id"] == "PLAIN-63" for row in rows))
            self.assertFalse(any(row["source_query_id"] == "f-excluded" for row in rows))
            self.assertFalse(any(row["source_query_id"] == "n-alias" for row in rows))
            self.assertEqual(sum(1 for row in rows if row["turboquant_topk_loss_weight"] == 1.0), 5)
            self.assertEqual(sum(1 for row in rows if row["turboquant_topk_recall_loss_weight"] == 1.0), 1)
            manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
            validate_contract(manifest)
            self.assertEqual(
                manifest["selection_audit"]["rejected"]["nfcorpus"]["promotion_post_alias_no_q3_hard_negative"],
                1,
            )
            self.assertEqual(manifest["go_loader_plan"]["workload"]["train"], 10)
            self.assertEqual(manifest["go_loader_plan"]["workload"]["steps_per_epoch"], 3)
            self.run_builder(paths)
            self.assertEqual(first_sha, sha256_file(paths["out"]))

    def test_contract_mutations_fail_local_validator(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = self.fixture(Path(tmp))
            self.run_builder(paths)
            manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
            for mutate in (
                lambda data: data["go_loader_plan"]["ingestion"].__setitem__("forbidden_no_tokenizer", False),
                lambda data: data["go_loader_plan"]["required_flags"].__setitem__("turboquant_topk_margin", 0.0),
                lambda data: data["go_loader_plan"]["anchor"].__setitem__("tokenizer_package", "missing"),
                lambda data: data["go_loader_plan"]["candidate"].__setitem__("candidate_target_differs_from_anchor", False),
                lambda data: data["go_loader_plan"]["argv_delta"].__setitem__("added_for_training", ["--bogus"]),
                lambda data: data["go_loader_plan"]["required_training_argv_tokens"].append("--no-tokenizer"),
                lambda data: data["go_loader_plan"]["legal_gates"].__setitem__("commercial_use_allowed", True),
            ):
                mutated = json.loads(json.dumps(manifest))
                mutate(mutated)
                with self.assertRaises(AssertionError):
                    validate_contract(mutated)

    def test_real_r4_manifest_contract_when_present(self) -> None:
        manifest_path = (
            REPO_ROOT
            / "runs/eos-d384-q3-boundary-goal-v1-20260902T084144Z/headroom-q3broad-gap050-dualcut-nf64-fiqa21-sf1-nfrecall24/data/beir134.q3broad-gap050-dualcut-nf64-fiqa21-sf1-nfrecall24.manifest.json"
        )
        if not manifest_path.is_file():
            self.skipTest("real R4 manifest artifact not present")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        validate_contract(manifest)
        self.assertEqual(
            manifest["go_loader_plan"]["anchor"]["anchor_package"],
            "runs/eos-wide384-projection-tail-seed191-repeat-v1-20260902T004753Z/packages/d384-pre.mll",
        )
        self.assertEqual(
            manifest["go_loader_plan"]["candidate"]["candidate_package"],
            "runs/eos-d384-q3-boundary-goal-v1-20260902T084144Z/packages/arm-q3broad-gap050-nf64-fiqa21-sf1-nfrecall24-rec005-q3w004-rw015-m002-lr2e6-e2-r4/d384-pre.mll",
        )
        self.assertEqual(
            manifest["go_loader_plan"]["metrics_target"],
            "runs/eos-d384-q3-boundary-goal-v1-20260902T084144Z/metrics/arm-q3broad-gap050-nf64-fiqa21-sf1-nfrecall24-rec005-q3w004-rw015-m002-lr2e6-e2-r4.train.metrics.json",
        )
        self.assertEqual(
            manifest["go_loader_plan"]["workload"],
            {
                "train": 134,
                "batch": 4,
                "steps_per_epoch": 34,
                "raw_candidates_per_epoch": 3815,
                "canonical_candidates_per_epoch": 3725,
                "main_q3_pairs_per_epoch": 2701,
                "recall_q3_pairs_per_epoch": 40486,
                "total_train_pairs_per_epoch": 46912,
                "planned_pairs": 93824,
                "actual_train_pairs": 0,
            },
        )
        self.assertEqual(
            manifest["sha256"]["train_score_spectrum"],
            "201a9093a827eba809a05e072520dc02e3ae0bd5ba7d1c4953e6c58e8ae84a03",
        )

    def test_real_launch_audit_when_present(self) -> None:
        audit_path = (
            REPO_ROOT
            / "runs/eos-d384-q3-boundary-goal-v1-20260902T084144Z/headroom-q3broad-gap050-dualcut-nf64-fiqa21-sf1-nfrecall24/reports/q3-r4-training-postfix-launch-audit.json"
        )
        if not audit_path.is_file():
            self.skipTest("real R4 launch audit not present")
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        validate_launch_audit(audit)
        for mutate in (
            lambda data: data.__setitem__("actual_training_ran", True),
            lambda data: data["candidate"].__setitem__("target", data["canonical_anchor"]["pre"]["target"]),
            lambda data: data["candidate"]["pre_plan"].__setitem__("content_rollup_sha256", "bad"),
            lambda data: data["candidate"]["post_plan"].__setitem__("content_rollup_sha256", "bad"),
            lambda data: data["canonical_anchor"]["post"].__setitem__("content_rollup_sha256", "bad"),
            lambda data: data["argv_delta"].__setitem__("added_for_training", ["--bogus"]),
            lambda data: data["plan_argv"].append("--no-tokenizer"),
            lambda data: data.__setitem__("metrics_target", "missing"),
            lambda data: data["checks"].__setitem__("canonical_anchor_unchanged", False),
        ):
            mutated = json.loads(json.dumps(audit))
            mutate(mutated)
            with self.assertRaises(AssertionError):
                validate_launch_audit(mutated)

    def test_rejects_stale_q3_source_hash(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = self.fixture(Path(tmp))
            result = self.run_builder(paths, "--expected-q3-sha256", "fiqa=deadbeef", check=False)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("fiqa q3 sha mismatch", result.stderr)

    def test_rejects_wrong_scifact_qid(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = self.fixture(Path(tmp))
            result = self.run_builder(paths, "--scifact-required-qid", "999", check=False)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("scifact eligible", result.stderr)

    def test_rejects_parent_sentinel_legal_gate_drift(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = self.fixture(Path(tmp), illegal_sentinel=True)
            result = self.run_builder(paths, check=False)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("release/commercial gates must be false", result.stderr)


if __name__ == "__main__":
    unittest.main()
