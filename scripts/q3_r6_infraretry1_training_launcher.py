#!/usr/bin/env python3
"""Fail-closed R6 plan-only launcher and preflight audit.

This owns the R6 integration boundary only: clone the pristine D384 seed191
package, run the exact plan-only training argv once, record immutable evidence,
and verify that evidence before any future training approval.  It intentionally
does not execute training.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = Path("runs/eos-d384-q3-boundary-goal-v1-20260902T084144Z")
SOURCE_R6_ROOT = RUN_ROOT / "headroom-q3r6-frontier-retain-alltrain-b0-rw050"
R6_ROOT = RUN_ROOT / "headroom-q3r6-frontier-retain-alltrain-b0-rw050-infraretry1"
SOURCE_ARM_ID = "arm-q3r6-frontier-retain-alltrain-b0-rw050-q3w004-m002-lr2e6-e2"
ARM_ID = "arm-q3r6-frontier-retain-alltrain-b0-rw050-q3w004-m002-lr2e6-e2-infraretry1"
ANCHOR_DIR = Path("runs/eos-wide384-projection-tail-seed191-repeat-v1-20260902T004753Z/packages")
ANCHOR_PACKAGE = ANCHOR_DIR / "d384-pre.mll"
CANDIDATE_DIR = RUN_ROOT / "packages" / ARM_ID
CANDIDATE_PACKAGE = CANDIDATE_DIR / "d384-pre.mll"
QUARANTINE_CANDIDATE_PACKAGE = RUN_ROOT / "packages" / SOURCE_ARM_ID / "d384-pre.mll"
R6_BINARY = RUN_ROOT / "bin/eos-r6-frontier-retention-infraretry1"
TRAIN_JSONL = SOURCE_R6_ROOT / "data/beir1155.q3r6-frontier-retention.train.jsonl"
TRAIN_MANIFEST = SOURCE_R6_ROOT / "data/beir1155.q3r6-frontier-retention.manifest.json"
COVERAGE_JSON = SOURCE_R6_ROOT / "q3r6-frontier-retention.coverage.json"
PROBE_MANIFEST = R6_ROOT / "probe/q3r6-frontier-retention.probe-manifest.json"
PLAN_STDOUT = R6_ROOT / "reports/q3-r6-infraretry1-training-launch-plan.stdout.txt"
PLAN_STDERR = R6_ROOT / "reports/q3-r6-infraretry1-training-launch-plan.stderr.txt"
LAUNCH_AUDIT = R6_ROOT / "reports/q3-r6-infraretry1-training-launch-audit.json"
PREFLIGHT_AUDIT = R6_ROOT / "reports/q3-r6-infraretry1-training-preflight-audit.json"
CLONE_INVENTORY = R6_ROOT / "reports/q3-r6-infraretry1-candidate-clone-inventory.json"
PROBE_BINDING = R6_ROOT / "probe/q3-r6-infraretry1-pretrain-binding-sidecar.json"
PREFLIGHT_STDOUT_CAPTURE = R6_ROOT / "reports/q3-r6-infraretry1-training-preflight.stdout.txt"
PREFLIGHT_STDERR_CAPTURE = R6_ROOT / "reports/q3-r6-infraretry1-training-preflight.stderr.txt"
RECOVERY_RECEIPT = Path("__no_recovery_receipt_for_infraretry1__")
HISTORICAL_FAILED_EXECUTE_AUDIT = SOURCE_R6_ROOT / "reports/q3-r6-training-once-execute-audit.json"
LAUNCHER_SCRIPT = Path("scripts/q3_r6_infraretry1_training_launcher.py")
LAUNCHER_TEST = Path("scripts/test_q3_r6_infraretry1_training_launcher.py")

LAUNCH_AUDIT_SCHEMA = "eos.q3_r6_frontier_retention.training_launch_audit.v1"
PREFLIGHT_SCHEMA = "eos.q3_r6_training_launcher_preflight.v1"
TRAIN_METRICS_SCHEMA = "manta.embedding_train_metrics.v1"
CLONE_SCHEMA = "eos.q3_r6_candidate_clone_inventory.v1"
PROBE_BINDING_SCHEMA = "eos.q3_r6_probe_pretrain_binding.v1"
EXPECTED_ANCHOR_TARGET_SHA256 = "188265db16992ab24be15e678c5f7e175bebad769e8d844e8b0f50ffc23bd5bf"
EXPECTED_ANCHOR_ROLLUP = "17335593ec850a1c01c8b31b17a32fe002368a508c06d431b761d45f1d9a76b0"
EXPECTED_QUARANTINE_ROLLUP = "631005d3b308187a0f9391276f54b4730be34ea5ab333532f35792f81ac18516"
EXPECTED_BINARY_SHA256 = "c5679853f8494c1e144a85d3cf2a345b0a3236f917a42607ddebc2242d344f90"
EXPECTED_TRAIN_JSONL_SHA256 = "41abe77ef5f2eb6adbd6ee5f1afd01e17d6f12bd31546db8a44cf77f309bed09"
EXPECTED_BUILDER_SHA256 = "4b8e6319a9957f62332aabf6972822520ee582fe509cc0d3606ae83f5cc3f64a"
EXPECTED_PROBE_PREFLIGHT_SHA256 = "2e585692ae53d8ef8afc0059a20bbf72c0b337d37594ff783e9c8e238eafb307"
EXPECTED_PROBE_PREFLIGHT_TEST_SHA256 = "0a5d92bb5d17bdfdbfe67d9db219da56e612b486b13277062359c71560105704"
EXPECTED_BUILDER_TEST_SHA256 = "1be7257c9a8e89351acbe1e8c0c46b0f369b93d0907a7e40d357b80854511646"
EXPECTED_PRIOR_PROBE_MANIFEST_SHA256 = "4e7ada4db5eaa5d962bf599cdb89dfd0f044d9664820f08ecaa8f084f915e653"
EXPECTED_PROBE_MANIFEST_SHA256 = "b4f6f57165ea6d285c59af25f23b89eee7508ec842f49bb85116575810f74bf7"
EXPECTED_RUNTIME_REPORT_SHA256 = "16bdf27debb1505db8599ac93b4b4f1758d22b7342341c7f3a98ce4b581b0c82"
EXPECTED_RECOVERY_RECEIPT_SHA256 = ""
EXPECTED_RECOVERY_RECEIPT_ATTEMPT = ""
RECOVERY_RECEIPT_MODE = "no_recovery_for_infraretry1_clean_plan"
EXPECTED_QRELS_SHA256 = {
    "probe/qrels/fiqa/frontier.fiqa.qrels.tsv": "fba02275f4158d6b93911aff2b8cefa86937834bc453f291423c39c73dd101d6",
    "probe/qrels/nfcorpus/frontier.nfcorpus.qrels.tsv": "1be3a853a38680db5a43db16f6bcd238894f7d6314d8b9448769310e3af79d47",
    "probe/qrels/scifact/frontier.scifact.qrels.tsv": "1f25e51d34f130fe2091788ecf599e3b877096dfcff3bba0e436222ff0852998",
    "probe/qrels/fiqa/top10-retention.fiqa.qrels.tsv": "dd80706a1e76f7e9b1288a83c12bbf180288dc121e23117f372e0e5c2fb26429",
    "probe/qrels/nfcorpus/top10-retention.nfcorpus.qrels.tsv": "6de33f6d05b158d45f2678ec61783542729c42c72a2f376156968fdf39cbc973",
    "probe/qrels/scifact/top10-retention.scifact.qrels.tsv": "cd0053c7d3a34e79949435692db26e60e9055e18150ee8b26c06b29f2d155e2c",
    "probe/qrels/nfcorpus/nf-boundary80-100.nfcorpus.qrels.tsv": "e55ea7061008c7f9fd1d3f4ea85d67c96bab082ea962d5205f55d1b7f3c9719e",
}
EXPECTED_SOURCE_SHA256 = {
    "scripts/build_q3_r6_frontier_retention_dataset.py": EXPECTED_BUILDER_SHA256,
    "scripts/q3_r6_infraretry1_probe_preflight.py": EXPECTED_PROBE_PREFLIGHT_SHA256,
    "scripts/test_q3_r6_infraretry1_probe_preflight.py": EXPECTED_PROBE_PREFLIGHT_TEST_SHA256,
    "scripts/test_build_q3_r6_frontier_retention_dataset.py": EXPECTED_BUILDER_TEST_SHA256,
    ".tiller/scratch/codex/q3-r6-aux-only-runtime-report.md": EXPECTED_RUNTIME_REPORT_SHA256,
}
EXPECTED_RUNTIME_SOURCE_SHA256 = {
    "runtime/embedding_score_spectrum_dataset.go": "abc095e43a1a8a947f901c17ab1d5feb2c5d34ea9531ed932a950ec5da7707ba",
    "runtime/embedding_score_spectrum_source.go": "897fee7a51cf15a991be454445776f54cb809e1e016e70bf465fa47138b13df8",
    "runtime/embedding_trainer.go": "59db464e089e7db2a43b64610aad38e64f609f8438ec201f8d226440c94dbdaa",
    "runtime/embedding_train_runner.go": "f193a0941d92e28a301cb3b701af3b7ee100609220aeb58e817bcae5e6c084f7",
    "cmd/eos/main.go": "285f13a2a87939926c02771b92cc52b85b0ee5c0066340d14f7e8cefaeaa65eb",
    "runtime/embedding_score_spectrum_dataset_test.go": "b50e41cc656910f7c328875118692acd9df301624de3bb75ef3bf79041aa0c61",
    "runtime/embedding_score_spectrum_source_test.go": "e2f74143be06527794034c16579b3a91e29e40171512d09bbffa91737ca807b2",
    "runtime/embedding_score_spectrum_trainer_test.go": "72cab82fa4f2246bd68404270fc5a7fa1a337d183abeb886213d10b577502a60",
    "cmd/eos/main_test.go": "78a248acf3bf77af9fd2a737c2f6ed43aed15dc9cb2776deeaf4e7ed50bca544",
}
EXPECTED_ARTIFACT_SHA256 = {
    str(TRAIN_JSONL): EXPECTED_TRAIN_JSONL_SHA256,
    str(TRAIN_MANIFEST): "6455a1b1c4662d0c06b2c334032067e75ace0f17759c7de773d3a69d114041f0",
    str(COVERAGE_JSON): "1cdcb05f2b60fd415f9abe269c70d9af6d647f1ac196ebb60958318f4b93bc79",
    str(PROBE_MANIFEST): EXPECTED_PROBE_MANIFEST_SHA256,
}
EXPECTED_LEGAL_SCOPE = {
    "research_only": True,
    "release_train_allowed": False,
    "commercial_use_allowed": False,
    "free_open_release_allowed": False,
    "quality_claim": False,
    "no_dev_reserve_official_eval": True,
}
EXPECTED_ARGV_DELTA = {
    "removed_for_training": ["--plan-only"],
    "added_for_training": [],
    "exactly_one_semantic_delta": True,
    "semantic_delta": "remove --plan-only only",
}
EXPECTED_WORKLOAD = {
    "train": 1155,
    "batch": 4,
    "steps_per_epoch": 289,
    "epochs": 2,
    "raw_candidates_per_epoch": 11183,
    "canonical_candidates_per_epoch": 11167,
    "alias_drops_per_epoch": 16,
    "main_q3_pairs_per_epoch": 10794,
    "recall_q3_pairs_per_epoch": 5120,
    "total_train_pairs_per_epoch": 27081,
    "planned_work_units": 54162,
    "actual_train_pairs": 0,
    "eval_examples": 0,
    "planned_eval_pairs": 0,
    "planned_total_pairs": 54162,
    "score_spectrum_aux_only_rows": 1155,
}
EXPECTED_EFFECTIVE_SCALE = {
    "main_row_scale": 0.038461538461538464,
    "retention_row_scale": 0.0196078431372549,
    "aggregate_main_to_recall_ratio": 1.578425480769231,
    "aggregate_ratio_lte_2": True,
    "base_hard_soft_recovery_expected_zero_rows": 1155,
    "base_hard_work_units": 0,
    "base_soft_work_units": 0,
    "base_recovery_work_units": 0,
    "base_hard_soft_recovery_contribution": 0,
    "aux_only_all_rows": True,
}
EXPECTED_BUCKETS = {
    "frontier_context_q3": 110,
    "top10_retention_q3": 405,
    "nf_boundary80_100_docretention_q3": 640,
}
EXPECTED_FLAGS = {
    "--score-spectrum-train": None,
    "--allow-research-only-score-spectrum": None,
    "--shuffle": "false",
    "--epochs": "2",
    "--batch-size": "4",
    "--seed": "191",
    "--contrastive-loss": "grouped_infonce",
    "--temperature": "0.05",
    "--lr": "0.000002",
    "--restore-best": "false",
    "--score-spectrum-loss-mode": "hard_soft_recovery",
    "--score-spectrum-recovery-weight": "0.05",
    "--score-spectrum-recovery-margin": "0",
    "--score-spectrum-recovery-top-k": "4",
    "--score-spectrum-recovery-tau": "0.05",
    "--turboquant-topk-objectives": "fullDim:3=0.04",
    "--turboquant-topk-loss": "lambdandcg",
    "--turboquant-topk-cutoff": "10",
    "--turboquant-topk-tau": "0.05",
    "--turboquant-topk-margin": "0.002",
    "--turboquant-topk-negative-mask": "q3",
    "--turboquant-topk-recall-weight": "0.50",
    "--turboquant-topk-recall-cutoff": "100",
    "--turboquant-topk-recall-tau": "0.05",
    "--turboquant-topk-recall-margin": "0",
    "--turboquant-topk-recall-negative-mask": "q3",
    "--turboquant-prefix-score-mode": "prepared_ip",
    "--turboquant-prefix-seed": "5581486560434873699",
}
FORBIDDEN_FLAGS = {"--no-tokenizer", "--score-spectrum-eval", "--retrieval-eval-dir", "--eval-only"}


class PreflightError(RuntimeError):
    def __init__(self, errors: list[str], audit: dict[str, Any]):
        super().__init__("; ".join(errors))
        self.errors = errors
        self.audit = audit


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--create-plan-audit", action="store_true")
    mode.add_argument("--preflight-only", action="store_true")
    mode.add_argument("--execute", action="store_true")
    parser.add_argument("--launch-audit", type=Path, default=LAUNCH_AUDIT)
    parser.add_argument("--expected-audit-sha256")
    parser.add_argument("--attempt-id", required=True)
    parser.add_argument("--stdout-capture", type=Path, default=PREFLIGHT_STDOUT_CAPTURE)
    parser.add_argument("--stderr-capture", type=Path, default=PREFLIGHT_STDERR_CAPTURE)
    parser.add_argument("--output-audit", type=Path, default=PREFLIGHT_AUDIT)
    parser.add_argument("--allow-existing-plan-artifacts", action="store_true")
    parser.add_argument("--recovery-receipt", type=Path)
    parser.add_argument("--expected-recovery-receipt-sha256")
    parser.add_argument("--overwrite-audits", action="store_true")
    return parser.parse_args(argv)


def repo_path(path: str | Path) -> Path:
    parsed = Path(path)
    return parsed if parsed.is_absolute() else REPO_ROOT / parsed


def display_path(path: str | Path) -> str:
    resolved = repo_path(path).resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        return str(resolved)


def utc_stamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with repo_path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def compact_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def write_json_new(path: str | Path, obj: dict[str, Any]) -> None:
    out = repo_path(path)
    if out.exists():
        raise FileExistsError(f"refusing to overwrite output: {display_path(out)}")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_json(path: str | Path, obj: dict[str, Any], *, overwrite: bool = False) -> None:
    out = repo_path(path)
    if out.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite output: {display_path(out)}")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_json_atomic(path: str | Path, obj: dict[str, Any]) -> None:
    out = repo_path(path)
    if out.exists():
        raise FileExistsError(f"refusing to overwrite output: {display_path(out)}")
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(f".{out.name}.tmp.{os.getpid()}")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(out)


def package_stem(package: Path) -> str:
    return package.name[:-4] if package.name.endswith(".mll") else package.stem


def sibling_content_rollup(sibling_hashes: list[dict[str, Any]]) -> str:
    h = hashlib.sha256()
    for item in sibling_hashes:
        h.update(compact_json({"bytes": item["bytes"], "name": item["name"], "sha256": item["sha256"]}).encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def package_state(target: str | Path) -> dict[str, Any]:
    target_path = repo_path(target)
    if not target_path.is_file():
        raise ValueError(f"missing package target: {display_path(target_path)}")
    stem = package_stem(target_path)
    siblings = sorted((p for p in target_path.parent.glob(f"{stem}*") if p.is_file()), key=lambda p: p.name)
    pathless = [{"bytes": p.stat().st_size, "name": p.name, "sha256": sha256_file(p)} for p in siblings]
    with_paths = [
        {"bytes": p.stat().st_size, "name": p.name, "path": display_path(p), "sha256": sha256_file(p)}
        for p in siblings
    ]
    return {
        "target": display_path(target_path),
        "target_absolute": str(target_path.resolve()),
        "target_sha256": sha256_file(target_path),
        "dir": display_path(target_path.parent),
        "sibling_count": len(pathless),
        "sibling_hashes": pathless,
        "sibling_hashes_with_paths": with_paths,
        "content_rollup_sha256": sibling_content_rollup(pathless),
        "tokenizer_package": display_path(target_path.with_name(f"{stem}.tokenizer.mll")),
        "tokenizer_package_sha256": sha256_file(target_path.with_name(f"{stem}.tokenizer.mll")),
    }


def require_within(path: str | Path, root: str | Path, errors: list[str], label: str) -> Path:
    resolved = repo_path(path).resolve()
    resolved_root = repo_path(root).resolve()
    try:
        resolved.relative_to(resolved_root)
    except ValueError:
        errors.append(f"{label} escapes root: {display_path(resolved)} not within {display_path(resolved_root)}")
    return resolved


def copy_anchor_clone() -> dict[str, Any]:
    errors: list[str] = []
    require_within(CANDIDATE_DIR, RUN_ROOT / "packages", errors, "candidate dir")
    if errors:
        raise ValueError("; ".join(errors))
    anchor_state = package_state(ANCHOR_PACKAGE)
    if anchor_state["sibling_count"] != 9:
        raise ValueError(f"anchor sibling count drift: {anchor_state['sibling_count']}")
    if anchor_state["target_sha256"] != EXPECTED_ANCHOR_TARGET_SHA256 or anchor_state["content_rollup_sha256"] != EXPECTED_ANCHOR_ROLLUP:
        raise ValueError("anchor identity drift")
    target_dir = repo_path(CANDIDATE_DIR)
    if target_dir.exists():
        existing = sorted(p.name for p in target_dir.iterdir() if p.is_file())
        raise FileExistsError(f"candidate dir already exists with files: {existing}")
    target_dir.mkdir(parents=True)
    copied = []
    try:
        for sibling in anchor_state["sibling_hashes"]:
            src = repo_path(ANCHOR_DIR / sibling["name"])
            dst = target_dir / sibling["name"]
            shutil.copy2(src, dst)
            copied.append({"from": display_path(src), "to": display_path(dst), "sha256": sha256_file(dst), "bytes": dst.stat().st_size})
    except Exception:
        shutil.rmtree(target_dir)
        raise
    candidate_state = package_state(CANDIDATE_PACKAGE)
    audit = {
        "schema": CLONE_SCHEMA,
        "created_utc": utc_stamp(),
        "arm_id": ARM_ID,
        "source": anchor_state,
        "candidate": candidate_state,
        "candidate_dir_was_absent": True,
        "copied": copied,
        "clone_matches_anchor": candidate_state["sibling_hashes"] == anchor_state["sibling_hashes"],
    }
    if not audit["clone_matches_anchor"]:
        raise ValueError("candidate clone does not match anchor")
    write_json_new(CLONE_INVENTORY, audit)
    return audit


def train_argv(plan_only: bool) -> list[str]:
    tokens = [display_path(R6_BINARY), "train-embed"]
    if plan_only:
        tokens.append("--plan-only")
    tokens.extend(
        [
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
            "--metrics-json", display_path(R6_ROOT / "metrics" / f"{ARM_ID}.train.metrics.json"),
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
            display_path(CANDIDATE_PACKAGE),
            display_path(TRAIN_JSONL),
        ]
    )
    return tokens


def option_value(argv: list[str], flag: str) -> str | None:
    if flag in argv:
        idx = argv.index(flag)
        if idx + 1 < len(argv):
            return argv[idx + 1]
    prefix = f"{flag}="
    for token in argv:
        if token.startswith(prefix):
            return token[len(prefix):]
    return None


def validate_argv_exact(plan: list[str], future: list[str], metrics_target: str, errors: list[str]) -> dict[str, Any]:
    details = {
        "plan_contains_plan_only": "--plan-only" in plan,
        "future_contains_plan_only": "--plan-only" in future,
        "metrics_target": metrics_target,
        "plan_metrics_value": option_value(plan, "--metrics-json"),
        "future_metrics_value": option_value(future, "--metrics-json"),
    }
    if "--plan-only" not in plan:
        errors.append("plan_argv missing --plan-only")
    without_plan = list(plan)
    if "--plan-only" in without_plan:
        without_plan.remove("--plan-only")
    if without_plan != future:
        errors.append("future_training_argv is not exactly plan_argv with one --plan-only removed")
    if "--plan-only" in future:
        errors.append("future_training_argv contains forbidden --plan-only")
    for flag in FORBIDDEN_FLAGS:
        if flag in plan or flag in future or any(token.startswith(f"{flag}=") for token in plan + future):
            errors.append(f"forbidden flag present: {flag}")
    if len(plan) < 2 or plan[1] != "train-embed":
        errors.append("plan_argv is not a train-embed command")
    for flag, expected in EXPECTED_FLAGS.items():
        present = flag in plan or any(token.startswith(f"{flag}=") for token in plan)
        if not present:
            errors.append(f"missing required flag: {flag}")
        if expected is not None and option_value(plan, flag) != expected:
            errors.append(f"required flag drift: {flag}")
    if option_value(plan, "--metrics-json") != metrics_target or option_value(future, "--metrics-json") != metrics_target:
        errors.append("metrics target missing/mismatched in argv")
    normalized_future = [display_path(token) if token.endswith(".mll") or token.endswith(".jsonl") or "/" in token else token for token in future]
    candidate_target = display_path(CANDIDATE_PACKAGE)
    anchor_target = display_path(ANCHOR_PACKAGE)
    train_jsonl = display_path(TRAIN_JSONL)
    if candidate_target not in normalized_future:
        errors.append("future_training_argv missing candidate target")
    if anchor_target in normalized_future:
        errors.append("future_training_argv targets canonical anchor")
    if train_jsonl not in normalized_future:
        errors.append("future_training_argv missing R6 train JSONL")
    return details


def parse_unique_int(line: str, pattern: str, label: str, *, required: bool = True) -> int | None:
    if len(re.findall(rf"(?:^| ){re.escape(label)}=", line)) > 1:
        raise ValueError(f"duplicate plan stdout field: {label}")
    matches = re.findall(pattern, line)
    if not matches:
        if required:
            raise ValueError(f"missing plan stdout field: {label}")
        return None
    values = [m if isinstance(m, str) else m[0] for m in matches]
    if len(values) != 1:
        raise ValueError(f"duplicate plan stdout field: {label}")
    try:
        value = int(values[0])
    except ValueError as exc:
        raise ValueError(f"malformed plan stdout integer: {label}") from exc
    if value < 0:
        raise ValueError(f"negative plan stdout integer: {label}")
    return value


def parse_plan_stdout(stdout: str) -> dict[str, int]:
    line = next((raw for raw in stdout.splitlines() if raw.startswith("planned workload:")), "")
    if not line:
        raise ValueError("missing planned workload line")
    if stdout.count("planned workload:") != 1:
        raise ValueError("duplicate planned workload line")
    train = parse_unique_int(line, r"(?:^| )train(?:_examples)?=([0-9]+)(?: |$)", "train")
    eval_examples = parse_unique_int(line, r"(?:^| )eval_examples=([0-9]+)(?: |$)", "eval_examples", required=False)
    batch = parse_unique_int(line, r"(?:^| )batch=([0-9]+)(?: |$)", "batch")
    steps = parse_unique_int(line, r"(?:^| )steps/epoch=([0-9]+)(?: |$)", "steps_per_epoch")
    epochs = parse_unique_int(line, r"(?:^| )epochs=([0-9]+)(?: |$)", "epochs", required=False)
    train_pairs = parse_unique_int(line, r"(?:^| )train_pairs/epoch=([0-9]+)(?: |$)", "train_pairs_per_epoch")
    eval_pairs = parse_unique_int(line, r"(?:^| )eval_pairs/pass=([0-9]+)(?: |$)", "eval_pairs_per_pass", required=False)
    aux_rows = parse_unique_int(line, r"(?:^| )score_spectrum_aux_only_rows=([0-9]+)(?: |$)", "score_spectrum_aux_only_rows")
    planned_actual = re.findall(r"pairs\(planned=([0-9]+) actual=([0-9]+)\)", line)
    if len(planned_actual) != 1:
        raise ValueError("missing or duplicate pairs(planned= actual=) field")
    planned_total, actual_total = (int(planned_actual[0][0]), int(planned_actual[0][1]))
    eval_passes = re.findall(r"eval_passes\(planned=([0-9]+) actual=([0-9]+)\)", line)
    if len(eval_passes) > 1:
        raise ValueError("duplicate eval_passes field")
    if eval_examples not in (None, 0) or eval_pairs not in (None, 0):
        raise ValueError("plan stdout records nonzero eval work")
    if eval_passes and (int(eval_passes[0][0]) != 0 or int(eval_passes[0][1]) != 0):
        raise ValueError("plan stdout records nonzero eval passes")
    parsed = {
        "train": train or 0,
        "batch": batch or 0,
        "steps_per_epoch": steps or 0,
        "total_train_pairs_per_epoch": train_pairs or 0,
        "planned_total_pairs": planned_total,
        "actual_train_pairs": actual_total,
        "score_spectrum_aux_only_rows": aux_rows or 0,
    }
    if epochs is not None:
        parsed["epochs"] = epochs
    parsed["planned_work_units"] = parsed["planned_total_pairs"]
    parsed["planned_eval_pairs"] = eval_pairs or 0
    parsed["eval_examples"] = eval_examples or 0
    return parsed


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(repo_path(path).read_text(encoding="utf-8"))


def validate_manifest_contract(errors: list[str]) -> dict[str, Any]:
    manifest = read_json(TRAIN_MANIFEST)
    coverage = read_json(COVERAGE_JSON)
    workload = manifest.get("counts", {}).get("workload", {})
    effective = manifest.get("counts", {}).get("effective_weight_audit", {})
    if workload != {k: EXPECTED_WORKLOAD[k] for k in ("train", "batch", "steps_per_epoch", "epochs", "raw_candidates_per_epoch", "canonical_candidates_per_epoch", "alias_drops_per_epoch", "main_q3_pairs_per_epoch", "recall_q3_pairs_per_epoch", "total_train_pairs_per_epoch", "planned_work_units", "actual_train_pairs")}:
        errors.append("manifest workload drift")
    if manifest.get("counts", {}).get("bucket_counts") != EXPECTED_BUCKETS:
        errors.append("bucket count drift")
    if effective.get("base_hard_soft_recovery_expected_zero_rows") != 1155:
        errors.append("base zero row count drift")
    if abs(float(effective.get("main_row_scale", -1)) - EXPECTED_EFFECTIVE_SCALE["main_row_scale"]) > 1e-15:
        errors.append("main effective scale drift")
    if abs(float(effective.get("retention_row_scale", -1)) - EXPECTED_EFFECTIVE_SCALE["retention_row_scale"]) > 1e-15:
        errors.append("recall effective scale drift")
    if abs(float(effective.get("aggregate_main_to_recall_ratio", -1)) - EXPECTED_EFFECTIVE_SCALE["aggregate_main_to_recall_ratio"]) > 1e-12:
        errors.append("aggregate scale ratio drift")
    legal = coverage.get("legal_gates", {})
    if legal != {
        "commercial_use_allowed": False,
        "free_open_release_allowed": False,
        "release_train_allowed": False,
        "train_allowed_for_research": True,
    }:
        errors.append("coverage legal gates drift")
    return {"manifest_workload": workload, "effective_weight_audit": effective, "coverage_legal_gates": legal}


def parse_utc(text: str, label: str, errors: list[str]) -> float | None:
    try:
        return time.mktime(time.strptime(text, "%Y-%m-%dT%H:%M:%SZ"))
    except Exception:
        errors.append(f"{label} timestamp malformed")
        return None


def file_record(path: str | Path) -> dict[str, Any]:
    p = repo_path(path)
    return {"path": display_path(p), "sha256": sha256_file(p), "bytes": p.stat().st_size}


def safe_file_record(path: str | Path, errors: list[str], label: str) -> dict[str, Any]:
    try:
        record = file_record(path)
        record["exists"] = True
        record["ok"] = True
        return record
    except Exception as exc:
        errors.append(f"{label} record error: {exc}")
        return {"path": display_path(path), "exists": repo_path(path).exists(), "ok": False, "error": str(exc)}


def safe_package_state(label: str, target: str | Path, errors: list[str]) -> dict[str, Any]:
    try:
        state = package_state(target)
        state["ok"] = True
        return state
    except Exception as exc:
        errors.append(f"{label} package state error: {exc}")
        return {"target": display_path(target), "ok": False, "error": str(exc)}


def package_changed_siblings(pre: dict[str, Any], post: dict[str, Any]) -> list[dict[str, Any]]:
    before = {item.get("name"): item for item in pre.get("sibling_hashes", [])}
    after = {item.get("name"): item for item in post.get("sibling_hashes", [])}
    changes = []
    for name in sorted(set(before) | set(after)):
        old = before.get(name)
        new = after.get(name)
        if old != new:
            changes.append({"name": name, "before": old, "after": new})
    return changes


def audit_bound_path_hashes(audit: dict[str, Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for section in ("sources", "artifacts"):
        node = audit.get(section, {})
        if not isinstance(node, dict):
            continue
        for label, item in sorted(node.items()):
            if not isinstance(item, dict) or "path" not in item or "sha256" not in item:
                continue
            path = item["path"]
            expected = item["sha256"]
            try:
                actual = sha256_file(path)
                ok = actual == expected
                error = None
            except Exception as exc:
                actual = None
                ok = False
                error = str(exc)
            record = {"section": section, "label": label, "path": display_path(path), "expected_sha256": expected, "actual_sha256": actual, "ok": ok}
            if error is not None:
                record["error"] = error
            records.append(record)
    return records


def metrics_root_for(args: argparse.Namespace) -> Path:
    return repo_path(args.launch_audit).resolve().parent.parent / "metrics"


def prepare_metrics_parent_for_execute(metrics_target: str | None, args: argparse.Namespace) -> dict[str, Any]:
    errors: list[str] = []
    record: dict[str, Any] = {
        "metrics_target": display_path(metrics_target or ""),
        "metrics_root": display_path(metrics_root_for(args)),
        "created_or_existing_parent": False,
        "ok": False,
        "errors": errors,
    }
    if not metrics_target:
        errors.append("metrics target missing before execute")
        return record
    target = repo_path(metrics_target)
    metrics_root = metrics_root_for(args).resolve()
    try:
        target_parent = target.parent
        target.resolve(strict=False).relative_to(metrics_root)
    except ValueError:
        errors.append("metrics target escapes metrics root before execute")
        return record
    if target.exists():
        errors.append("metrics target already exists before execute")
        return record
    if target_parent.exists() and not target_parent.is_dir():
        errors.append("metrics parent exists and is not a directory")
        return record
    try:
        target_parent.mkdir(parents=True, exist_ok=True)
        record["created_or_existing_parent"] = True
    except Exception as exc:
        errors.append(f"metrics parent mkdir failed: {exc}")
        return record
    try:
        resolved_parent = target_parent.resolve(strict=True)
        resolved_parent.relative_to(metrics_root)
        record["resolved_parent"] = display_path(resolved_parent)
        record["parent_is_dir"] = resolved_parent.is_dir()
        record["metrics_file_absent_after_parent_prepare"] = not target.exists()
        if not resolved_parent.is_dir():
            errors.append("metrics parent is not a directory after mkdir")
        if target.exists():
            errors.append("metrics target appeared during parent preparation")
    except Exception as exc:
        errors.append(f"metrics parent resolve/revalidate failed: {exc}")
    record["ok"] = not errors
    return record


def is_finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def collect_nonfinite_numbers(value: Any, path: str = "$") -> list[str]:
    bad: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            bad.extend(collect_nonfinite_numbers(child, f"{path}.{key}"))
    elif isinstance(value, list):
        for idx, child in enumerate(value):
            bad.extend(collect_nonfinite_numbers(child, f"{path}[{idx}]"))
    elif isinstance(value, float) and not math.isfinite(value):
        bad.append(path)
    return bad


def approx_equal(actual: Any, expected: float, tol: float = 1e-9) -> bool:
    return is_finite_number(actual) and abs(float(actual) - expected) <= tol


def int_equal(actual: Any, expected: int) -> bool:
    return isinstance(actual, int) and not isinstance(actual, bool) and actual == expected


def nonnegative_number(actual: Any) -> bool:
    return is_finite_number(actual) and float(actual) >= 0


def absent_or_exact_zero(node: dict[str, Any], key: str, tol: float = 1e-9) -> bool:
    if key not in node:
        return True
    return approx_equal(node.get(key), 0.0, tol)


def objective_matches_full_dim_q3(node: Any) -> bool:
    return (
        isinstance(node, list)
        and len(node) == 1
        and isinstance(node[0], dict)
        and int_equal(node[0].get("dim"), 384)
        and int_equal(node[0].get("bit_width"), 3)
        and approx_equal(node[0].get("weight"), 0.04, 1e-7)
    )


def validate_execute_metrics(metrics_target: str | None, args: argparse.Namespace, errors: list[str]) -> dict[str, Any]:
    record: dict[str, Any] = {"path": display_path(metrics_target or ""), "exists": False, "ok": False}
    if not metrics_target:
        errors.append("execute metrics target missing")
        return record
    path = repo_path(metrics_target).resolve()
    metrics_root = metrics_root_for(args).resolve()
    record["metrics_root"] = display_path(metrics_root)
    try:
        path.relative_to(metrics_root)
        record["contained"] = True
    except ValueError:
        record["contained"] = False
        errors.append("execute metrics target escapes metrics root")
    if not path.exists():
        errors.append("execute metrics missing after exit0")
        return record
    if not path.is_file():
        errors.append("execute metrics target is not a regular file")
        record["exists"] = True
        return record
    record["exists"] = True
    try:
        size = path.stat().st_size
        record["bytes"] = size
        record["sha256"] = sha256_file(path)
    except Exception as exc:
        errors.append(f"execute metrics hash/stat error: {exc}")
        record["error"] = str(exc)
        return record
    if size <= 0:
        errors.append("execute metrics file is empty")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        errors.append(f"execute metrics JSON parse error: {exc}")
        record["json_parse_error"] = str(exc)
        return record
    record["json"] = data
    if not isinstance(data, dict):
        errors.append("execute metrics JSON is not an object")
        return record
    nonfinite_paths = collect_nonfinite_numbers(data)
    for bad_path in nonfinite_paths:
        errors.append(f"execute metrics nonfinite numeric value: {bad_path}")
    config = data.get("config")
    workload = data.get("workload")
    summary = data.get("summary")
    throughput = data.get("throughput")
    accelerators = data.get("accelerators")
    package = data.get("package")
    final_train = data.get("final_train")
    artifact_exact = display_path(CANDIDATE_PACKAGE)
    checks = {
        "schema": data.get("schema") == TRAIN_METRICS_SCHEMA,
        "command": data.get("command") == "train-embed",
        "mode": data.get("mode") == "train",
        "artifact": data.get("artifact") == artifact_exact,
        "config_object": isinstance(config, dict),
        "workload_object": isinstance(workload, dict),
        "summary_object": isinstance(summary, dict),
        "throughput_object": isinstance(throughput, dict),
        "accelerators_object": isinstance(accelerators, dict),
        "package_object": isinstance(package, dict),
        "final_train_object": isinstance(final_train, dict),
        "no_nonfinite_numbers": not nonfinite_paths,
    }
    if isinstance(config, dict):
        checks.update(
            {
                "config_epochs": config.get("epochs") == EXPECTED_WORKLOAD["epochs"],
                "config_batch_size": config.get("batch_size") == EXPECTED_WORKLOAD["batch"],
                "config_seed": config.get("seed") == 191,
                "config_learning_rate": approx_equal(config.get("learning_rate"), 0.000002, 1e-12),
                "config_effective_learning_rate": approx_equal(config.get("effective_learning_rate"), 0.000002, 1e-12),
                "config_shuffle": config.get("shuffle") is False,
                "config_restore_best": config.get("restore_best") is False,
                "config_eval_only_false": config.get("eval_only") is False,
                "config_pairwise_train_false": config.get("pairwise_train") is False,
                "config_hard_negative_train_false": config.get("hard_negative_train") is False,
                "config_score_spectrum_train_true": config.get("score_spectrum_train") is True,
                "config_allow_research_only_score_spectrum_true": config.get("allow_research_only_score_spectrum") is True,
                "config_contrastive_loss": config.get("contrastive_loss") == "grouped_infonce",
                "config_temperature": approx_equal(config.get("temperature"), 0.05, 1e-7),
                "config_score_spectrum_loss_mode": config.get("score_spectrum_loss_mode") == "hard_soft_recovery",
                "config_score_spectrum_recovery_weight": approx_equal(config.get("score_spectrum_recovery_weight"), 0.05, 1e-7),
                "config_score_spectrum_recovery_margin_absent_or_zero": absent_or_exact_zero(config, "score_spectrum_recovery_margin", 1e-7),
                "config_score_spectrum_recovery_top_k": int_equal(config.get("score_spectrum_recovery_top_k"), 4),
                "config_score_spectrum_recovery_tau": approx_equal(config.get("score_spectrum_recovery_tau"), 0.05, 1e-7),
                "config_turboquant_topk_objectives": objective_matches_full_dim_q3(config.get("turboquant_topk_objectives")),
                "config_turboquant_topk_loss": config.get("turboquant_topk_loss") == "lambdandcg",
                "config_turboquant_topk_cutoff": int_equal(config.get("turboquant_topk_cutoff"), 10),
                "config_turboquant_topk_tau": approx_equal(config.get("turboquant_topk_tau"), 0.05, 1e-7),
                "config_turboquant_topk_margin": approx_equal(config.get("turboquant_topk_margin"), 0.002, 1e-7),
                "config_turboquant_topk_negative_mask": config.get("turboquant_topk_negative_mask") == "q3",
                "config_turboquant_topk_recall_weight": approx_equal(config.get("turboquant_topk_recall_weight"), 0.50, 1e-7),
                "config_turboquant_topk_recall_cutoff": int_equal(config.get("turboquant_topk_recall_cutoff"), 100),
                "config_turboquant_topk_recall_tau": approx_equal(config.get("turboquant_topk_recall_tau"), 0.05, 1e-7),
                "config_turboquant_topk_recall_margin_absent_or_zero": absent_or_exact_zero(config, "turboquant_topk_recall_margin", 1e-7),
                "config_turboquant_topk_recall_negative_mask": config.get("turboquant_topk_recall_negative_mask") == "q3",
                "config_turboquant_prefix_seed": int_equal(config.get("turboquant_prefix_seed"), 5581486560434873699),
                "config_turboquant_prefix_score_mode": config.get("turboquant_prefix_score_mode") in {"prepared_ip", "prepared-ip"},
            }
        )
    if isinstance(workload, dict):
        checks.update(
            {
                "workload_train_examples": workload.get("train_examples") == EXPECTED_WORKLOAD["train"],
                "workload_eval_examples_zero": workload.get("eval_examples") == 0,
                "workload_batch_size": workload.get("batch_size") == EXPECTED_WORKLOAD["batch"],
                "workload_planned_epochs": workload.get("planned_epochs") == EXPECTED_WORKLOAD["epochs"],
                "workload_completed_epochs": workload.get("completed_epochs") == EXPECTED_WORKLOAD["epochs"],
                "workload_train_batches_per_epoch": workload.get("train_batches_per_epoch") == EXPECTED_WORKLOAD["steps_per_epoch"],
                "workload_train_pairs_per_epoch": workload.get("train_pairs_per_epoch") == EXPECTED_WORKLOAD["total_train_pairs_per_epoch"],
                "workload_eval_pairs_per_pass_zero": workload.get("eval_pairs_per_pass") == 0,
                "workload_planned_eval_passes_zero": workload.get("planned_eval_passes") == 0,
                "workload_actual_eval_passes_zero": workload.get("actual_eval_passes") == 0,
                "workload_planned_train_pairs": workload.get("planned_train_pairs") == EXPECTED_WORKLOAD["planned_total_pairs"],
                "workload_actual_train_pairs": workload.get("actual_train_pairs") == EXPECTED_WORKLOAD["planned_total_pairs"],
                "workload_actual_train_examples": workload.get("actual_train_examples") == EXPECTED_WORKLOAD["train"] * EXPECTED_WORKLOAD["epochs"],
                "workload_planned_total_pairs": workload.get("planned_total_pairs") == EXPECTED_WORKLOAD["planned_total_pairs"],
                "workload_planned_eval_pairs_zero": workload.get("planned_eval_pairs") == 0,
                "workload_actual_eval_pairs_zero": workload.get("actual_eval_pairs") == 0,
                "workload_actual_eval_examples_zero": workload.get("actual_eval_examples") == 0,
                "workload_actual_total_pairs": workload.get("actual_total_pairs") == EXPECTED_WORKLOAD["planned_total_pairs"],
                "workload_actual_total_examples": workload.get("actual_total_examples") == EXPECTED_WORKLOAD["train"] * EXPECTED_WORKLOAD["epochs"],
                "workload_actual_pair_arithmetic": workload.get("actual_total_pairs") == workload.get("actual_train_pairs", -1) + workload.get("actual_eval_pairs", -2),
                "workload_planned_pair_arithmetic": workload.get("planned_total_pairs") == workload.get("planned_train_pairs", -1) + workload.get("planned_eval_pairs", -2),
            }
        )
    if isinstance(summary, dict):
        checks.update(
            {
                "summary_epochs_completed": summary.get("epochs_completed") == EXPECTED_WORKLOAD["epochs"],
                "summary_steps_completed": summary.get("steps_completed") == EXPECTED_WORKLOAD["steps_per_epoch"] * EXPECTED_WORKLOAD["epochs"],
                "summary_steps_run": summary.get("steps_run") == EXPECTED_WORKLOAD["steps_per_epoch"] * EXPECTED_WORKLOAD["epochs"],
                "summary_restore_best_false": summary.get("restored_best") is False,
                "summary_stopped_early_false": summary.get("stopped_early") is False,
            }
        )
    if isinstance(throughput, dict):
        for key in ("elapsed_seconds", "train_seconds", "eval_seconds", "examples_per_second", "pairs_per_second", "train_examples_per_second", "train_pairs_per_second", "eval_examples_per_second", "eval_pairs_per_second", "optimizer_steps_per_second"):
            checks[f"throughput_{key}_finite_nonnegative"] = nonnegative_number(throughput.get(key))
    if isinstance(accelerators, dict):
        for key in ("forward", "optimizer", "activation", "contrastive"):
            checks[f"accelerator_{key}_valid"] = isinstance(accelerators.get(key), str) and accelerators.get(key).strip() != ""
    if isinstance(final_train, dict):
        checks["final_train_loss_finite_nonnegative"] = nonnegative_number(final_train.get("loss"))
        checks["final_train_batch_size_positive"] = isinstance(final_train.get("batch_size"), int) and final_train.get("batch_size") > 0
    if isinstance(package, dict):
        checks["package_artifact"] = package.get("artifact") == artifact_exact
    for key, ok in checks.items():
        if not ok:
            errors.append(f"execute metrics validation failed: {key}")
    record["checks"] = checks
    record["ok"] = all(checks.values()) and size > 0
    return record


def validate_recovery_receipt(args: argparse.Namespace, intended_plan: list[str], intended_future: list[str], manifest_workload: dict[str, Any], errors: list[str]) -> dict[str, Any] | None:
    if not args.allow_existing_plan_artifacts:
        return None
    if args.recovery_receipt is None or args.expected_recovery_receipt_sha256 is None:
        return None
    receipt_path = repo_path(args.recovery_receipt)
    if display_path(receipt_path) != display_path(RECOVERY_RECEIPT):
        errors.append("recovery receipt path drift")
        return None
    if not receipt_path.is_file():
        errors.append("recovery receipt missing")
        return None
    actual_sha = sha256_file(receipt_path)
    if args.expected_recovery_receipt_sha256 != EXPECTED_RECOVERY_RECEIPT_SHA256 or actual_sha != EXPECTED_RECOVERY_RECEIPT_SHA256:
        errors.append("recovery receipt sha256 mismatch")
    receipt = read_json(receipt_path)
    checks = {}
    checks["schema"] = receipt.get("schema") == LAUNCH_AUDIT_SCHEMA
    checks["attempt"] = receipt.get("attempt_id") == EXPECTED_RECOVERY_RECEIPT_ATTEMPT
    checks["mode"] = receipt.get("mode") is None
    checks["legacy_failed_receipt"] = receipt.get("all_checks_passed") is False
    checks["plan_argv_exact"] = receipt.get("plan_argv") == intended_plan
    checks["future_argv_exact"] = receipt.get("future_training_argv") == intended_future
    checks["argv_delta_exact"] = receipt.get("argv_delta") == EXPECTED_ARGV_DELTA
    checks["plan_spawned_once_returned_zero"] = receipt.get("plan_result", {}).get("exit_status") == 0
    checks["not_training"] = receipt.get("actual_training_ran") is False and receipt.get("trainer_process_spawned") is False
    stdout_path = receipt.get("plan_result", {}).get("stdout_path")
    stderr_path = receipt.get("plan_result", {}).get("stderr_path")
    checks["stdout_path_exact"] = display_path(stdout_path or "") == display_path(PLAN_STDOUT)
    checks["stderr_path_exact"] = display_path(stderr_path or "") == display_path(PLAN_STDERR)
    if stdout_path and repo_path(stdout_path).is_file():
        stdout_record = file_record(stdout_path)
        checks["stdout_sha_exact"] = stdout_record["sha256"] == receipt.get("plan_result", {}).get("stdout_sha256") == "8cf248ac7529f72408c63fbfebee187526f070befae1af0c3b50b04ceefaedf6"
        checks["stdout_nonempty"] = stdout_record["bytes"] > 0
        try:
            stdout_workload = parse_plan_stdout(repo_path(stdout_path).read_text(encoding="utf-8"))
        except ValueError as exc:
            errors.append(f"recovery receipt stdout parse failed: {exc}")
            stdout_workload = {}
    else:
        stdout_record = {}
        stdout_workload = {}
        checks["stdout_sha_exact"] = False
        checks["stdout_nonempty"] = False
    if stderr_path and repo_path(stderr_path).is_file():
        stderr_record = file_record(stderr_path)
        checks["stderr_sha_exact"] = stderr_record["sha256"] == receipt.get("plan_result", {}).get("stderr_sha256") == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        checks["stderr_empty"] = stderr_record["bytes"] == 0
    else:
        stderr_record = {}
        checks["stderr_sha_exact"] = False
        checks["stderr_empty"] = False
    receipt_checks = receipt.get("checks", {})
    failing_checks = sorted(k for k, v in receipt_checks.items() if v is not True)
    checks["specific_parser_only_failure"] = failing_checks == ["workload_exact"]
    receipt_parsed = receipt.get("parsed_workload", {})
    checks["legacy_parser_missed_train_and_epochs"] = receipt_parsed.get("train") == 0 and receipt_parsed.get("epochs") == 0
    anchor_node = receipt.get("canonical_anchor", {})
    checks["canonical_anchor_pre_post_stable"] = anchor_node.get("pre", {}).get("content_rollup_sha256") == anchor_node.get("post", {}).get("content_rollup_sha256") == EXPECTED_ANCHOR_ROLLUP
    candidate_node = receipt.get("candidate", {})
    checks["candidate_pre_post_stable"] = candidate_node.get("pre_plan", {}).get("content_rollup_sha256") == candidate_node.get("post_plan", {}).get("content_rollup_sha256") == EXPECTED_ANCHOR_ROLLUP
    checks["candidate_target_differs_from_anchor"] = receipt.get("candidate", {}).get("target") != receipt.get("canonical_anchor", {}).get("pre", {}).get("target")
    checks["metrics_absent"] = bool(receipt.get("metrics_target")) and not repo_path(receipt["metrics_target"]).exists()
    checks["stdout_matches_manifest_workload"] = stdout_workload.get("train") == manifest_workload.get("train") and stdout_workload.get("batch") == manifest_workload.get("batch") and stdout_workload.get("steps_per_epoch") == manifest_workload.get("steps_per_epoch") and stdout_workload.get("total_train_pairs_per_epoch") == manifest_workload.get("total_train_pairs_per_epoch") and stdout_workload.get("planned_total_pairs") == manifest_workload.get("planned_work_units")
    receipt_time = parse_utc(receipt.get("created_utc", ""), "recovery receipt", errors)
    clone_time = parse_utc(read_json(CLONE_INVENTORY).get("created_utc", ""), "clone inventory", errors) if repo_path(CLONE_INVENTORY).is_file() else None
    checks["timestamp_order"] = clone_time is not None and receipt_time is not None and clone_time <= receipt_time
    for key, ok in checks.items():
        if not ok:
            errors.append(f"recovery receipt validation failed: {key}")
    return {
        "path": display_path(receipt_path),
        "sha256": actual_sha,
        "expected_sha256": args.expected_recovery_receipt_sha256,
        "schema": receipt.get("schema"),
        "attempt_id": receipt.get("attempt_id"),
        "mode": RECOVERY_RECEIPT_MODE,
        "stdout": stdout_record,
        "stderr": stderr_record,
        "stdout_workload": stdout_workload,
        "legacy_parsed_workload": receipt_parsed,
        "failing_checks": failing_checks,
        "plan_subprocess_spawn_count_proven": 1,
        "plan_subprocess_exit_status": receipt.get("plan_result", {}).get("exit_status"),
        "checks": checks,
    }


def validate_clean_plan_receipt(intended_plan: list[str], intended_future: list[str], errors: list[str]) -> dict[str, Any] | None:
    if not repo_path(LAUNCH_AUDIT).is_file():
        errors.append("existing infraretry1 plan captures require prior clean launch audit receipt")
        return None
    receipt = read_json(LAUNCH_AUDIT)
    stdout_path = receipt.get("plan_result", {}).get("stdout_path")
    stderr_path = receipt.get("plan_result", {}).get("stderr_path")
    checks = {
        "schema": receipt.get("schema") == LAUNCH_AUDIT_SCHEMA,
        "all_checks_passed": receipt.get("all_checks_passed") is True,
        "no_training": receipt.get("actual_training_ran") is False and receipt.get("trainer_process_spawned") is False,
        "plan_argv_exact": receipt.get("plan_argv") == intended_plan,
        "future_argv_exact": receipt.get("future_training_argv") == intended_future,
        "plan_exit_zero": receipt.get("plan_result", {}).get("exit_status") == 0,
        "stdout_path_exact": display_path(stdout_path or "") == display_path(PLAN_STDOUT),
        "stderr_path_exact": display_path(stderr_path or "") == display_path(PLAN_STDERR),
        "candidate_stable": receipt.get("candidate", {}).get("pre_plan", {}).get("content_rollup_sha256") == receipt.get("candidate", {}).get("post_plan", {}).get("content_rollup_sha256") == EXPECTED_ANCHOR_ROLLUP,
        "anchor_stable": receipt.get("canonical_anchor", {}).get("pre", {}).get("content_rollup_sha256") == receipt.get("canonical_anchor", {}).get("post", {}).get("content_rollup_sha256") == EXPECTED_ANCHOR_ROLLUP,
        "metrics_absent": bool(receipt.get("metrics_target")) and not repo_path(receipt["metrics_target"]).exists(),
    }
    stdout_record = safe_file_record(stdout_path or "", errors, "clean plan stdout")
    stderr_record = safe_file_record(stderr_path or "", errors, "clean plan stderr")
    checks["stdout_sha_exact"] = stdout_record.get("sha256") == receipt.get("plan_result", {}).get("stdout_sha256")
    checks["stderr_sha_exact"] = stderr_record.get("sha256") == receipt.get("plan_result", {}).get("stderr_sha256")
    for key, ok in checks.items():
        if not ok:
            errors.append(f"clean infraretry1 plan receipt validation failed: {key}")
    return {
        "path": display_path(LAUNCH_AUDIT),
        "sha256": sha256_file(LAUNCH_AUDIT),
        "attempt_id": receipt.get("attempt_id"),
        "plan_subprocess_spawn_count_proven": 1,
        "plan_subprocess_exit_status": receipt.get("plan_result", {}).get("exit_status"),
        "stdout": stdout_record,
        "stderr": stderr_record,
        "checks": checks,
    }


def pinned_path_hashes() -> list[dict[str, Any]]:
    records = []
    expected = {**EXPECTED_ARTIFACT_SHA256, **EXPECTED_SOURCE_SHA256, **EXPECTED_RUNTIME_SOURCE_SHA256}
    for path, expected_sha in sorted(expected.items()):
        records.append({"path": display_path(path), "expected_sha256": expected_sha, "actual_sha256": sha256_file(path), "ok": sha256_file(path) == expected_sha})
    for rel, expected_sha in sorted(EXPECTED_QRELS_SHA256.items()):
        path = SOURCE_R6_ROOT / rel
        records.append({"path": display_path(path), "expected_sha256": expected_sha, "actual_sha256": sha256_file(path), "ok": sha256_file(path) == expected_sha})
    records.append({"path": display_path(R6_BINARY), "expected_sha256": EXPECTED_BINARY_SHA256, "actual_sha256": sha256_file(R6_BINARY), "ok": sha256_file(R6_BINARY) == EXPECTED_BINARY_SHA256})
    return records


def compare_package_node(label: str, expected_node: dict[str, Any], errors: list[str]) -> dict[str, Any]:
    try:
        actual = package_state(expected_node.get("target", ""))
    except Exception as exc:
        errors.append(f"{label} package state error: {exc}")
        return {}
    for key in ("target_sha256", "sibling_count", "sibling_hashes", "content_rollup_sha256"):
        if expected_node.get(key) != actual.get(key):
            errors.append(f"{label} {key} mismatch")
    if expected_node.get("sibling_hashes_with_paths") != actual.get("sibling_hashes_with_paths"):
        errors.append(f"{label} sibling_hashes_with_paths mismatch")
    return actual


def output_paths_contained_and_fresh(args: argparse.Namespace, errors: list[str]) -> dict[str, Any]:
    reports_root = repo_path(args.launch_audit).parent.resolve()
    paths = {
        "stdout_capture": repo_path(args.stdout_capture).resolve(),
        "stderr_capture": repo_path(args.stderr_capture).resolve(),
        "output_audit": repo_path(args.output_audit).resolve(),
    }
    checks: dict[str, Any] = {"reports_root": display_path(reports_root)}
    for label, path in paths.items():
        try:
            path.relative_to(reports_root)
            checks[f"{label}_contained"] = True
        except ValueError:
            checks[f"{label}_contained"] = False
            errors.append(f"{label} path escapes reports root")
        fresh_ok = not path.exists()
        if label == "output_audit" and args.overwrite_audits and not args.execute:
            fresh_ok = True
        checks[f"{label}_fresh"] = fresh_ok
        if not fresh_ok:
            errors.append(f"{label} already exists: {display_path(path)}")
        if args.execute and (path.name.startswith("q3-r6-training-once") or path.name.startswith("q3-r6-infraretry1-training-once")):
            checks[f"{label}_not_historical_once_path"] = False
            errors.append(f"{label} uses historical failed execute path")
        else:
            checks[f"{label}_not_historical_once_path"] = True
    checks["stdout_stderr_distinct"] = paths["stdout_capture"] != paths["stderr_capture"]
    checks["audit_distinct_from_captures"] = paths["output_audit"] not in {paths["stdout_capture"], paths["stderr_capture"]}
    if not checks["stdout_stderr_distinct"] or not checks["audit_distinct_from_captures"]:
        errors.append("execute/preflight output paths must be distinct")
    return checks


def matching_trainer_processes(argv: list[str]) -> list[dict[str, Any]]:
    if not argv:
        return []
    binary = str(repo_path(argv[0]).resolve())
    matches = []
    proc_root = Path("/proc")
    if not proc_root.is_dir():
        return matches
    for entry in proc_root.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            raw = (entry / "cmdline").read_bytes()
        except OSError:
            continue
        if not raw:
            continue
        parts = [p.decode("utf-8", "replace") for p in raw.split(b"\0") if p]
        if len(parts) >= 2 and str(repo_path(parts[0]).resolve()) == binary and parts[1] == "train-embed":
            matches.append({"pid": int(entry.name), "argv": parts})
    return matches


def materialize_probe_binding(anchor_state: dict[str, Any], candidate_state: dict[str, Any]) -> dict[str, Any]:
    sidecar = {
        "schema": PROBE_BINDING_SCHEMA,
        "created_utc": utc_stamp(),
        "arm_id": ARM_ID,
        "probe_manifest": {"path": display_path(PROBE_MANIFEST), "sha256": sha256_file(PROBE_MANIFEST)},
        "anchor_package": {
            "path": display_path(ANCHOR_PACKAGE),
            "target_sha256": anchor_state["target_sha256"],
            "sibling_rollup_sha256": anchor_state["content_rollup_sha256"],
        },
        "candidate_package": {
            "path": display_path(CANDIDATE_PACKAGE),
            "pretrain_target_sha256": candidate_state["target_sha256"],
            "pretrain_sibling_rollup_sha256": candidate_state["content_rollup_sha256"],
            "post_train_sibling_rollup_sha256": "future post-train integration binding",
        },
        "pending_strict_probe_bindings": [
            "candidate_package.post_train_sibling_rollup_sha256",
            "candidate_effective_base_leakage_audit",
            "42 real probe outputs",
        ],
    }
    write_json_new(PROBE_BINDING, sidecar)
    return sidecar


def create_plan_audit(args: argparse.Namespace) -> dict[str, Any]:
    planned_paths = [PLAN_STDOUT, PLAN_STDERR, LAUNCH_AUDIT, PREFLIGHT_AUDIT, CLONE_INVENTORY, PROBE_BINDING, PREFLIGHT_STDOUT_CAPTURE, PREFLIGHT_STDERR_CAPTURE]
    if not args.allow_existing_plan_artifacts:
        existing = [display_path(p) for p in planned_paths if repo_path(p).exists()]
        if existing:
            raise FileExistsError(f"plan artifact already exists: {existing}")
    if args.allow_existing_plan_artifacts and repo_path(CLONE_INVENTORY).is_file():
        clone_audit = read_json(CLONE_INVENTORY)
        if clone_audit.get("schema") != CLONE_SCHEMA or clone_audit.get("clone_matches_anchor") is not True:
            raise ValueError("existing clone inventory is not an accepted R6 clone")
    else:
        clone_audit = copy_anchor_clone()
    errors: list[str] = []
    manifest_contract = validate_manifest_contract(errors)
    pin_records = pinned_path_hashes()
    for item in pin_records:
        if not item["ok"]:
            errors.append(f"pinned path sha256 mismatch: {item['path']}")
    before_anchor = package_state(ANCHOR_PACKAGE)
    before_candidate = package_state(CANDIDATE_PACKAGE)
    if before_anchor["content_rollup_sha256"] != EXPECTED_ANCHOR_ROLLUP:
        errors.append("canonical anchor rollup drift")
    if before_candidate["content_rollup_sha256"] != EXPECTED_ANCHOR_ROLLUP:
        errors.append("candidate pretrain rollup drift")
    plan = train_argv(plan_only=True)
    future = train_argv(plan_only=False)
    metrics_target = option_value(plan, "--metrics-json") or ""
    argv_details = validate_argv_exact(plan, future, metrics_target, errors)
    recovery_receipt = validate_recovery_receipt(args, plan, future, manifest_contract["manifest_workload"], errors)
    clean_plan_receipt = None
    if repo_path(metrics_target).exists():
        errors.append("metrics target exists before plan")
    if errors:
        raise PreflightError(errors, {"errors": errors})
    reused_existing_plan = False
    if args.allow_existing_plan_artifacts and repo_path(PLAN_STDOUT).is_file() and repo_path(PLAN_STDERR).is_file():
        if recovery_receipt is None:
            clean_plan_receipt = validate_clean_plan_receipt(plan, future, errors)
        if errors:
            raise PreflightError(errors, {"errors": errors})
        proc_returncode = int((recovery_receipt or clean_plan_receipt)["plan_subprocess_exit_status"] if recovery_receipt else 0)
        plan_stdout = repo_path(PLAN_STDOUT).read_text(encoding="utf-8")
        plan_stderr = repo_path(PLAN_STDERR).read_text(encoding="utf-8")
        reused_existing_plan = True
    else:
        proc = subprocess.run(plan, cwd=REPO_ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        proc_returncode = proc.returncode
        plan_stdout = proc.stdout
        plan_stderr = proc.stderr
        repo_path(PLAN_STDOUT).parent.mkdir(parents=True, exist_ok=True)
        repo_path(PLAN_STDOUT).write_text(plan_stdout, encoding="utf-8")
        repo_path(PLAN_STDERR).write_text(plan_stderr, encoding="utf-8")
    parsed = parse_plan_stdout(plan_stdout)
    if "epochs" not in parsed:
        argv_epochs = option_value(plan, "--epochs")
        if argv_epochs is None:
            errors.append("missing epochs in plan stdout and argv")
        else:
            parsed["epochs"] = int(argv_epochs)
    manifest_workload = manifest_contract["manifest_workload"]
    parsed.update(
        {
            "raw_candidates_per_epoch": manifest_workload["raw_candidates_per_epoch"],
            "canonical_candidates_per_epoch": manifest_workload["canonical_candidates_per_epoch"],
            "alias_drops_per_epoch": manifest_workload["alias_drops_per_epoch"],
            "main_q3_pairs_per_epoch": manifest_workload["main_q3_pairs_per_epoch"],
            "recall_q3_pairs_per_epoch": manifest_workload["recall_q3_pairs_per_epoch"],
        }
    )
    after_anchor = package_state(ANCHOR_PACKAGE)
    after_candidate = package_state(CANDIDATE_PACKAGE)
    quarantine_state = package_state(QUARANTINE_CANDIDATE_PACKAGE)
    if args.allow_existing_plan_artifacts and repo_path(PROBE_BINDING).is_file():
        sidecar = read_json(PROBE_BINDING)
    else:
        sidecar = materialize_probe_binding(before_anchor, before_candidate)
    audit = {
        "schema": LAUNCH_AUDIT_SCHEMA,
        "created_utc": utc_stamp(),
        "attempt_id": args.attempt_id,
        "arm_id": ARM_ID,
        "actual_training_ran": False,
        "trainer_process_spawned": False,
        "reused_existing_plan_stdout_after_fail_closed_launcher_attempt": reused_existing_plan,
        "recovery_receipt": recovery_receipt,
        "clean_plan_receipt": clean_plan_receipt,
        "all_checks_passed": True,
        "binary_dirty_build_caveat": "binary reports eos dev and is hash-bound to the accepted dirty build",
        "legal_scope": dict(EXPECTED_LEGAL_SCOPE),
        "argv_delta": dict(EXPECTED_ARGV_DELTA),
        "plan_argv": plan,
        "future_training_argv": future,
        "metrics_target": metrics_target,
        "parsed_workload": parsed,
        "authoritative_expected_workload": dict(EXPECTED_WORKLOAD),
        "effective_scale": dict(EXPECTED_EFFECTIVE_SCALE),
        "manifest_contract": manifest_contract,
        "plan_result": {
            "exit_status": proc_returncode,
            "stdout_path": display_path(PLAN_STDOUT),
            "stdout_sha256": sha256_file(PLAN_STDOUT),
            "stderr_path": display_path(PLAN_STDERR),
            "stderr_sha256": sha256_file(PLAN_STDERR),
        },
        "binary": {"path": display_path(R6_BINARY), "sha256": sha256_file(R6_BINARY), "expected_sha256": EXPECTED_BINARY_SHA256},
        "artifacts": {
            "train_jsonl": {"path": display_path(TRAIN_JSONL), "sha256": sha256_file(TRAIN_JSONL)},
            "manifest": {"path": display_path(TRAIN_MANIFEST), "sha256": sha256_file(TRAIN_MANIFEST)},
            "coverage": {"path": display_path(COVERAGE_JSON), "sha256": sha256_file(COVERAGE_JSON)},
            "probe_manifest": {"path": display_path(PROBE_MANIFEST), "sha256": sha256_file(PROBE_MANIFEST)},
            "probe_binding_sidecar": {"path": display_path(PROBE_BINDING), "sha256": sha256_file(PROBE_BINDING)},
        },
        "sources": {
            "builder": {"path": "scripts/build_q3_r6_frontier_retention_dataset.py", "sha256": sha256_file("scripts/build_q3_r6_frontier_retention_dataset.py")},
            "probe_preflight": {"path": "scripts/q3_r6_infraretry1_probe_preflight.py", "sha256": sha256_file("scripts/q3_r6_infraretry1_probe_preflight.py")},
            "probe_preflight_test": {"path": "scripts/test_q3_r6_infraretry1_probe_preflight.py", "sha256": sha256_file("scripts/test_q3_r6_infraretry1_probe_preflight.py")},
            "launcher": {"path": display_path(LAUNCHER_SCRIPT), "sha256": sha256_file(LAUNCHER_SCRIPT)},
            "launcher_test": {"path": display_path(LAUNCHER_TEST), "sha256": sha256_file(LAUNCHER_TEST)},
            "runtime_report": {"path": ".tiller/scratch/codex/q3-r6-aux-only-runtime-report.md", "sha256": sha256_file(".tiller/scratch/codex/q3-r6-aux-only-runtime-report.md")},
        },
        "runtime_sources": [{"path": path, "sha256": sha256_file(path)} for path in sorted(EXPECTED_RUNTIME_SOURCE_SHA256)],
        "qrels": [{"path": display_path(SOURCE_R6_ROOT / rel), "sha256": sha256_file(SOURCE_R6_ROOT / rel)} for rel in sorted(EXPECTED_QRELS_SHA256)],
        "pinned_path_hashes": pin_records,
        "canonical_anchor": {"pre": before_anchor, "post": after_anchor},
        "candidate": {"target": display_path(CANDIDATE_PACKAGE), "pre_plan": before_candidate, "post_plan": after_candidate},
        "clone_inventory": {"path": display_path(CLONE_INVENTORY), "sha256": sha256_file(CLONE_INVENTORY), "clone_matches_anchor": clone_audit["clone_matches_anchor"]},
        "probe_binding": {"path": display_path(PROBE_BINDING), "sha256": sha256_file(PROBE_BINDING), "pending": sidecar["pending_strict_probe_bindings"]},
        "historical_quarantine": {
            "failed_execute_audit": {"path": display_path(HISTORICAL_FAILED_EXECUTE_AUDIT), "sha256": sha256_file(HISTORICAL_FAILED_EXECUTE_AUDIT)},
            "candidate_package": quarantine_state,
            "expected_quarantine_rollup_sha256": EXPECTED_QUARANTINE_ROLLUP,
            "unchanged": quarantine_state["content_rollup_sha256"] == EXPECTED_QUARANTINE_ROLLUP,
        },
        "argv_details": argv_details,
        "checks": {},
    }
    audit["checks"] = {
        "all_pinned_path_hashes_match": all(item["ok"] for item in pin_records),
        "binary_sha256_bound": audit["binary"]["sha256"] == EXPECTED_BINARY_SHA256,
        "train_jsonl_sha256_bound": audit["artifacts"]["train_jsonl"]["sha256"] == EXPECTED_TRAIN_JSONL_SHA256,
        "probe_manifest_sha256_bound": audit["artifacts"]["probe_manifest"]["sha256"] == EXPECTED_PROBE_MANIFEST_SHA256,
        "argv_delta_remove_plan_only_only": audit["argv_delta"] == EXPECTED_ARGV_DELTA and [x for x in plan if x != "--plan-only"] == future,
        "explicit_turboquant_cutoff10": option_value(plan, "--turboquant-topk-cutoff") == "10",
        "explicit_base_zero_rows_bound": manifest_contract["effective_weight_audit"].get("base_hard_soft_recovery_expected_zero_rows") == 1155,
        "effective_scale_exact": audit["effective_scale"] == EXPECTED_EFFECTIVE_SCALE,
        "workload_exact": parsed == EXPECTED_WORKLOAD,
        "plan_exit_status_zero": proc_returncode == 0,
        "candidate_target_differs_from_anchor": repo_path(CANDIDATE_PACKAGE).resolve() != repo_path(ANCHOR_PACKAGE).resolve(),
        "candidate_pretrain_matches_anchor": before_candidate["content_rollup_sha256"] == before_anchor["content_rollup_sha256"] == EXPECTED_ANCHOR_ROLLUP,
        "candidate_unchanged_by_plan": before_candidate["content_rollup_sha256"] == after_candidate["content_rollup_sha256"],
        "canonical_anchor_unchanged": before_anchor["content_rollup_sha256"] == after_anchor["content_rollup_sha256"],
        "metrics_target_absent_after_plan": not repo_path(metrics_target).exists(),
        "plan_only_no_trainer_spawn": True,
        "no_eval_rows": parsed.get("eval_examples") == 0 and parsed.get("planned_eval_pairs") == 0,
        "probe_binding_sidecar_materialized": repo_path(PROBE_BINDING).is_file(),
        "historical_quarantine_unchanged": audit["historical_quarantine"]["unchanged"],
    }
    audit["all_checks_passed"] = proc_returncode == 0 and all(audit["checks"].values())
    if not audit["all_checks_passed"]:
        raise PreflightError(["plan audit contract failed"], audit)
    write_json(LAUNCH_AUDIT, audit, overwrite=args.overwrite_audits)
    return audit


def validate_launch_audit(args: argparse.Namespace) -> dict[str, Any]:
    errors: list[str] = []
    launch_path = repo_path(args.launch_audit)
    if not launch_path.is_file():
        raise FileNotFoundError(f"missing launch audit: {display_path(launch_path)}")
    launch_sha = sha256_file(launch_path)
    audit = read_json(launch_path)
    if audit.get("schema") != LAUNCH_AUDIT_SCHEMA:
        errors.append("launch audit schema drift")
    if args.expected_audit_sha256 and args.expected_audit_sha256 != launch_sha:
        errors.append("launch audit sha256 mismatch")
    if args.execute and not args.expected_audit_sha256:
        errors.append("execute mode requires --expected-audit-sha256")
    if audit.get("legal_scope") != EXPECTED_LEGAL_SCOPE:
        errors.append("legal_scope drift")
    if audit.get("actual_training_ran") is not False or audit.get("trainer_process_spawned") is not False:
        errors.append("launch audit records training/spawn")
    if audit.get("all_checks_passed") is not True:
        errors.append("launch audit all_checks_passed is not true")
    if audit.get("parsed_workload") != EXPECTED_WORKLOAD:
        errors.append("parsed workload drift")
    if audit.get("effective_scale") != EXPECTED_EFFECTIVE_SCALE:
        errors.append("effective scale drift")
    plan = audit.get("plan_argv", [])
    future = audit.get("future_training_argv", [])
    argv_details = validate_argv_exact(plan, future, audit.get("metrics_target", ""), errors)
    recovery = audit.get("recovery_receipt")
    recovery_checks: dict[str, bool] = {}
    if recovery is None:
        recovery_checks = {"clean_plan_no_recovery": True}
    elif not isinstance(recovery, dict):
        errors.append("recovery receipt binding malformed")
    else:
        recovery_checks = {
            "path": recovery.get("path") == display_path(RECOVERY_RECEIPT),
            "sha": recovery.get("sha256") == EXPECTED_RECOVERY_RECEIPT_SHA256 == sha256_file(RECOVERY_RECEIPT),
            "expected_sha": recovery.get("expected_sha256") == EXPECTED_RECOVERY_RECEIPT_SHA256,
            "schema": recovery.get("schema") == LAUNCH_AUDIT_SCHEMA,
            "attempt": recovery.get("attempt_id") == EXPECTED_RECOVERY_RECEIPT_ATTEMPT,
            "mode": recovery.get("mode") == RECOVERY_RECEIPT_MODE,
            "spawn_count": recovery.get("plan_subprocess_spawn_count_proven") == 1,
            "exit_status": recovery.get("plan_subprocess_exit_status") == 0,
            "checks": isinstance(recovery.get("checks"), dict) and all(recovery["checks"].values()),
        }
        for key, ok in recovery_checks.items():
            if not ok:
                errors.append(f"recovery receipt binding drift: {key}")
    pin_records = pinned_path_hashes()
    for item in pin_records:
        if not item["ok"]:
            errors.append(f"pinned path sha256 mismatch: {item['path']}")
    audit_bound_records = audit_bound_path_hashes(audit)
    for item in audit_bound_records:
        if not item["ok"]:
            errors.append(f"launch audit bound path sha256 mismatch: {item['section']}.{item['label']} {item['path']}")
    if not any(item["section"] == "sources" and item["label"] == "launcher_test" for item in audit_bound_records):
        errors.append("launch audit missing launcher_test source binding")
    package_actuals = {
        "canonical_anchor.pre": compare_package_node("canonical_anchor.pre", audit.get("canonical_anchor", {}).get("pre", {}), errors),
        "canonical_anchor.post": compare_package_node("canonical_anchor.post", audit.get("canonical_anchor", {}).get("post", {}), errors),
        "candidate.pre_plan": compare_package_node("candidate.pre_plan", audit.get("candidate", {}).get("pre_plan", {}), errors),
        "candidate.post_plan": compare_package_node("candidate.post_plan", audit.get("candidate", {}).get("post_plan", {}), errors),
    }
    if args.attempt_id in {audit.get("attempt_id"), EXPECTED_RECOVERY_RECEIPT_ATTEMPT, "q3-r6-training-once"}:
        errors.append("attempt_id is not unique for this launch")
    output_path_checks = output_paths_contained_and_fresh(args, errors)
    if audit.get("metrics_target") and repo_path(audit["metrics_target"]).exists():
        errors.append(f"metrics target already exists: {display_path(audit['metrics_target'])}")
    live_matches = matching_trainer_processes(future)
    if args.execute and live_matches:
        errors.append("matching trainer process already live")
    plan_result = audit.get("plan_result", {})
    for stream in ("stdout", "stderr"):
        path = plan_result.get(f"{stream}_path")
        expected = plan_result.get(f"{stream}_sha256")
        if not path or not repo_path(path).is_file():
            errors.append(f"plan_result {stream} missing")
        elif sha256_file(path) != expected:
            errors.append(f"plan_result {stream} sha256 mismatch")
    result = {
        "schema": PREFLIGHT_SCHEMA,
        "created_utc": utc_stamp(),
        "mode": "execute" if args.execute else "preflight-only",
        "attempt_id": args.attempt_id,
        "launch_audit_path": display_path(launch_path),
        "launch_audit_sha256": launch_sha,
        "expected_audit_sha256": args.expected_audit_sha256,
        "trainer_process_spawned": False,
        "actual_training_ran": False,
        "future_training_argv": copy.deepcopy(future),
        "metrics_target": audit.get("metrics_target"),
        "pinned_path_hashes": pin_records,
        "audit_bound_path_hashes": audit_bound_records,
        "package_actuals": package_actuals,
        "argv_details": argv_details,
        "expected_workload": EXPECTED_WORKLOAD,
        "parsed_workload": audit.get("parsed_workload"),
        "checks": {
            "launch_audit_schema": audit.get("schema") == LAUNCH_AUDIT_SCHEMA,
            "expected_audit_sha256_matches": args.expected_audit_sha256 is None or args.expected_audit_sha256 == launch_sha,
            "execute_mode_reserved_fail_closed": not args.execute,
            "execute_requires_expected_audit_sha256": (not args.execute) or bool(args.expected_audit_sha256),
            "legal_scope_exact": audit.get("legal_scope") == EXPECTED_LEGAL_SCOPE,
            "actual_training_not_recorded": audit.get("actual_training_ran") is False and audit.get("trainer_process_spawned") is False,
            "launch_audit_checks_passed": audit.get("all_checks_passed") is True,
            "all_pinned_path_hashes_match": all(item["ok"] for item in pin_records),
            "audit_bound_path_hashes_match": bool(audit_bound_records) and all(item["ok"] for item in audit_bound_records),
            "launcher_test_sha256_bound": any(item["section"] == "sources" and item["label"] == "launcher_test" and item["ok"] for item in audit_bound_records),
            "argv_legal": len([e for e in errors if "argv" in e or "flag" in e or "metrics target missing/mismatched" in e]) == 0,
            "workload_exact": audit.get("parsed_workload") == EXPECTED_WORKLOAD,
            "effective_scale_exact": audit.get("effective_scale") == EXPECTED_EFFECTIVE_SCALE,
            "plan_exit_status_zero": audit.get("plan_result", {}).get("exit_status") == 0,
            "recovery_receipt_bound": bool(recovery_checks) and all(recovery_checks.values()),
            "metrics_target_absent": bool(audit.get("metrics_target")) and not repo_path(audit["metrics_target"]).exists(),
            "stdout_capture_absent": not repo_path(args.stdout_capture).exists(),
            "stderr_capture_absent": not repo_path(args.stderr_capture).exists(),
            "output_audit_absent_before_write": args.overwrite_audits or not repo_path(args.output_audit).exists(),
            "output_paths_contained_and_fresh": all(v for k, v in output_path_checks.items() if k != "reports_root"),
            "no_live_matching_trainer": not live_matches,
            "candidate_target_differs_from_anchor": repo_path(CANDIDATE_PACKAGE).resolve() != repo_path(ANCHOR_PACKAGE).resolve(),
            "candidate_pretrain_matches_anchor": package_actuals["candidate.pre_plan"].get("content_rollup_sha256") == EXPECTED_ANCHOR_ROLLUP,
            "candidate_unchanged_by_plan": package_actuals["candidate.pre_plan"].get("content_rollup_sha256") == package_actuals["candidate.post_plan"].get("content_rollup_sha256"),
            "canonical_anchor_unchanged": package_actuals["canonical_anchor.pre"].get("content_rollup_sha256") == package_actuals["canonical_anchor.post"].get("content_rollup_sha256"),
        },
        "errors": errors,
    }
    if args.execute:
        result["checks"].pop("execute_mode_reserved_fail_closed", None)
        result["checks"]["execute_mode_authorized"] = True
    else:
        result["checks"]["preflight_only_no_spawn"] = True
    result["checks"]["all_checks_true"] = not errors and all(result["checks"].values())
    if errors:
        raise PreflightError(errors, result)
    return result


def final_execute_gate(args: argparse.Namespace, preflight_audit: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    launch_path = repo_path(args.launch_audit)
    output_checks = output_paths_contained_and_fresh(args, errors)
    launch_sha = None
    launch_audit: dict[str, Any] = {}
    try:
        launch_sha = sha256_file(launch_path)
        launch_audit = read_json(launch_path)
    except Exception as exc:
        errors.append(f"launch audit reread/hash error: {exc}")
    expected_launch_sha = preflight_audit.get("launch_audit_sha256")
    if launch_sha != expected_launch_sha or (args.expected_audit_sha256 and launch_sha != args.expected_audit_sha256):
        errors.append("final gate launch audit sha256 mismatch")
    bound_records = audit_bound_path_hashes(launch_audit) if launch_audit else []
    for item in bound_records:
        if not item["ok"]:
            errors.append(f"final gate audit-bound path drift: {item['section']}.{item['label']} {item['path']}")
    pin_records = pinned_path_hashes()
    for item in pin_records:
        if not item["ok"]:
            errors.append(f"final gate pinned path sha256 mismatch: {item['path']}")
    pre_anchor = safe_package_state("final gate anchor", ANCHOR_PACKAGE, errors)
    pre_candidate = safe_package_state("final gate candidate", CANDIDATE_PACKAGE, errors)
    live_matches = matching_trainer_processes(preflight_audit.get("future_training_argv", []))
    checks = {
        "launch_audit_still_matches": launch_sha == expected_launch_sha and (not args.expected_audit_sha256 or launch_sha == args.expected_audit_sha256),
        "output_paths_contained_and_fresh": all(v for k, v in output_checks.items() if k != "reports_root"),
        "audit_bound_path_hashes_match": bool(bound_records) and all(item["ok"] for item in bound_records),
        "launcher_test_sha256_bound": any(item["section"] == "sources" and item["label"] == "launcher_test" and item["ok"] for item in bound_records),
        "pinned_path_hashes_match": all(item["ok"] for item in pin_records),
        "anchor_pristine_before_spawn": pre_anchor.get("ok") is True and pre_anchor.get("content_rollup_sha256") == EXPECTED_ANCHOR_ROLLUP,
        "candidate_pristine_before_spawn": pre_candidate.get("ok") is True and pre_candidate.get("content_rollup_sha256") == EXPECTED_ANCHOR_ROLLUP and pre_candidate.get("sibling_count") == 9,
        "candidate_target_differs_from_anchor": repo_path(CANDIDATE_PACKAGE).resolve() != repo_path(ANCHOR_PACKAGE).resolve(),
        "no_live_matching_trainer": not live_matches,
        "preflight_all_checks_true": preflight_audit.get("checks", {}).get("all_checks_true") is True,
    }
    for key, ok in checks.items():
        if not ok:
            errors.append(f"final execute gate failed: {key}")
    return {
        "checked_utc": utc_stamp(),
        "launch_audit_sha256": launch_sha,
        "expected_launch_audit_sha256": expected_launch_sha,
        "output_path_checks": output_checks,
        "audit_bound_path_hashes": bound_records,
        "pinned_path_hashes": pin_records,
        "anchor": pre_anchor,
        "candidate": pre_candidate,
        "live_matches": live_matches,
        "checks": checks,
        "errors": errors,
    }


def execute_once(args: argparse.Namespace, preflight_audit: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    stdout_path = repo_path(args.stdout_capture)
    stderr_path = repo_path(args.stderr_capture)
    output_audit = repo_path(args.output_audit)
    argv = list(preflight_audit["future_training_argv"])
    start_utc = utc_stamp()
    start_monotonic = time.monotonic()
    final_gate = final_execute_gate(args, preflight_audit)
    audit: dict[str, Any] = {
        "schema": PREFLIGHT_SCHEMA,
        "created_utc": start_utc,
        "mode": "execute",
        "attempt_id": args.attempt_id,
        "launch_audit_path": preflight_audit["launch_audit_path"],
        "launch_audit_sha256": preflight_audit["launch_audit_sha256"],
        "expected_audit_sha256": preflight_audit["expected_audit_sha256"],
        "future_training_argv": argv,
        "metrics_target": preflight_audit["metrics_target"],
        "preflight_checks": preflight_audit["checks"],
        "final_pre_spawn_gate": final_gate,
        "pre_spawn": {
            "anchor": final_gate["anchor"],
            "candidate": final_gate["candidate"],
            "pinned_path_hashes": final_gate["pinned_path_hashes"],
            "audit_bound_path_hashes": final_gate["audit_bound_path_hashes"],
        },
        "trainer_process_spawned": False,
        "actual_training_ran": False,
        "spawn_count": 0,
        "retry_count": 0,
        "checks": {},
        "errors": [],
    }
    return_code = 1
    if final_gate["errors"]:
        metrics_parent = {"ok": False, "not_attempted_due_to_final_gate": True}
    else:
        metrics_parent = prepare_metrics_parent_for_execute(preflight_audit.get("metrics_target"), args)
    audit["metrics_parent_prepared"] = metrics_parent
    if final_gate["errors"] or not metrics_parent.get("ok"):
        audit["errors"].extend(final_gate["errors"])
        audit["errors"].extend(metrics_parent.get("errors", []))
        audit["checks"] = {
            "final_pre_spawn_gate_passed": not final_gate["errors"],
            "metrics_parent_prepared": metrics_parent.get("ok") is True,
            "trainer_process_spawned": False,
            "actual_training_ran": False,
            "spawn_count_zero_before_abort": audit["spawn_count"] == 0,
            "retry_count_zero": audit["retry_count"] == 0,
            "all_checks_true": False,
        }
        audit["end_utc"] = utc_stamp()
        audit["duration_seconds"] = max(0.0, time.monotonic() - start_monotonic)
        audit["execute_result"] = {"exit_status": 1, "not_spawned": True}
        write_json_atomic(output_audit, audit)
        return 1, audit
    try:
        if output_audit.exists():
            raise FileExistsError(f"refusing to overwrite output audit: {display_path(output_audit)}")
        stdout_path.parent.mkdir(parents=True, exist_ok=True)
        stderr_path.parent.mkdir(parents=True, exist_ok=True)
        with stdout_path.open("xb") as stdout_fh, stderr_path.open("xb") as stderr_fh:
            proc = subprocess.Popen(argv, cwd=REPO_ROOT, stdout=stdout_fh, stderr=stderr_fh)
            audit["trainer_process_spawned"] = True
            audit["actual_training_ran"] = True
            audit["spawn_count"] = 1
            audit["pid"] = proc.pid
            return_code = proc.wait()
    except Exception as exc:
        audit["spawn_error"] = str(exc)
        audit["errors"].append(f"execute spawn/write failed: {exc}")
    audit["end_utc"] = utc_stamp()
    audit["duration_seconds"] = max(0.0, time.monotonic() - start_monotonic)
    audit["execute_result"] = {"exit_status": return_code}
    if stdout_path.is_file():
        audit["execute_result"]["stdout_capture"] = safe_file_record(stdout_path, audit["errors"], "stdout_capture")
    if stderr_path.is_file():
        audit["execute_result"]["stderr_capture"] = safe_file_record(stderr_path, audit["errors"], "stderr_capture")
    metrics_target = preflight_audit.get("metrics_target")
    if return_code == 0:
        audit["execute_result"]["metrics"] = validate_execute_metrics(metrics_target, args, audit["errors"])
    elif metrics_target and repo_path(metrics_target).is_file():
        audit["execute_result"]["metrics"] = safe_file_record(metrics_target, audit["errors"], "metrics")
    else:
        audit["execute_result"]["metrics"] = {"path": display_path(metrics_target or ""), "exists": False, "ok": False}
    post_anchor = safe_package_state("post execute anchor", ANCHOR_PACKAGE, audit["errors"])
    post_candidate = safe_package_state("post execute candidate", CANDIDATE_PACKAGE, audit["errors"])
    try:
        post_inputs = pinned_path_hashes()
    except Exception as exc:
        audit["errors"].append(f"post execute pinned path hash error: {exc}")
        post_inputs = []
    try:
        post_bound = audit_bound_path_hashes(read_json(args.launch_audit))
    except Exception as exc:
        audit["errors"].append(f"post execute audit-bound hash error: {exc}")
        post_bound = []
    changed_siblings = package_changed_siblings(audit["pre_spawn"]["candidate"], post_candidate)
    audit["post_execute"] = {
        "anchor": post_anchor,
        "candidate": post_candidate,
        "pinned_path_hashes": post_inputs,
        "audit_bound_path_hashes": post_bound,
    }
    audit["candidate_changed_siblings"] = changed_siblings
    try:
        launch_still_matches = sha256_file(preflight_audit["launch_audit_path"]) == preflight_audit["launch_audit_sha256"]
    except Exception as exc:
        audit["errors"].append(f"launch audit post hash error: {exc}")
        launch_still_matches = False
    audit["checks"] = {
        "final_pre_spawn_gate_passed": all(final_gate["checks"].values()) and not final_gate["errors"],
        "preflight_all_checks_true": preflight_audit["checks"].get("all_checks_true") is True,
        "spawn_count_exactly_one": audit["spawn_count"] == 1,
        "retry_count_zero": audit["retry_count"] == 0,
        "trainer_process_spawned": audit["trainer_process_spawned"] is True,
        "actual_training_ran": audit["actual_training_ran"] is True,
        "stdout_capture_recorded": stdout_path.is_file(),
        "stderr_capture_recorded": stderr_path.is_file(),
        "launch_audit_still_matches": launch_still_matches,
        "anchor_immutable": audit["pre_spawn"]["anchor"].get("content_rollup_sha256") == post_anchor.get("content_rollup_sha256") == EXPECTED_ANCHOR_ROLLUP,
        "input_hashes_immutable": all(item.get("ok") for item in post_inputs) and audit["pre_spawn"]["pinned_path_hashes"] == post_inputs and all(item.get("ok") for item in post_bound) and audit["pre_spawn"]["audit_bound_path_hashes"] == post_bound,
        "candidate_had_pristine_pretrain_before_spawn": audit["pre_spawn"]["candidate"].get("content_rollup_sha256") == EXPECTED_ANCHOR_ROLLUP,
        "candidate_post_state_recorded": post_candidate.get("ok") is True and post_candidate.get("sibling_count") == 9,
        "candidate_target_present_after_exit0": return_code != 0 or post_candidate.get("ok") is True,
        "candidate_changed_after_success": return_code != 0 or (post_candidate.get("content_rollup_sha256") != audit["pre_spawn"]["candidate"].get("content_rollup_sha256") and bool(changed_siblings)),
        "metrics_valid_after_exit0": return_code != 0 or audit["execute_result"].get("metrics", {}).get("ok") is True,
        "no_live_matching_trainer_after": not matching_trainer_processes(argv),
    }
    if return_code != 0:
        audit["errors"].append(f"trainer exited nonzero: {return_code}")
    elif not audit["checks"]["candidate_changed_after_success"]:
        audit["errors"].append("trainer exit0 produced no candidate package mutation")
    audit["checks"]["all_checks_true"] = not audit["errors"] and all(audit["checks"].values())
    write_json_atomic(output_audit, audit)
    if return_code == 0 and not audit["checks"]["all_checks_true"]:
        return_code = 1
    return return_code, audit


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.create_plan_audit:
            audit = create_plan_audit(args)
            print(json.dumps({"ok": True, "launch_audit": display_path(LAUNCH_AUDIT), "sha256": sha256_file(LAUNCH_AUDIT)}, sort_keys=True))
            return 0 if audit["all_checks_passed"] else 1
        audit = validate_launch_audit(args)
        if args.execute:
            status, _execute_audit = execute_once(args, audit)
            print(json.dumps({"ok": status == 0, "output_audit": display_path(args.output_audit), "exit_status": status}, sort_keys=True))
            return status
        write_json(args.output_audit, audit, overwrite=args.overwrite_audits)
    except PreflightError as exc:
        try:
            write_json_new(args.output_audit, exc.audit)
        except Exception:
            pass
        print(f"preflight rejected: {'; '.join(exc.errors)}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"launcher failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({"ok": True, "output_audit": display_path(args.output_audit)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
