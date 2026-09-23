#!/usr/bin/env python3
"""Fail-closed launcher/preflight for the EOS Q3/R4 audited training command.

The launch audit is the authority.  This helper validates the audit, package
siblings, pinned file hashes, argv semantics, output lanes, and workload before
optionally executing ``future_training_argv`` exactly once.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
LAUNCH_AUDIT_SCHEMA = "eos.q3_r4_broad_gap.training_launch_audit.v1"
PREFLIGHT_SCHEMA = "eos.q3_r4_training_launcher_preflight.v1"
EXPECTED_PARSED_WORKLOAD = {
    "train": 134,
    "batch": 4,
    "steps_per_epoch": 34,
    "train_pairs_per_epoch": 46912,
    "planned_pairs": 93824,
    "actual_train_pairs": 0,
}
EXPECTED_LEGAL_SCOPE = {
    "research_only": True,
    "release_train_allowed": False,
    "commercial_use_allowed": False,
    "quality_claim": False,
    "no_dev_reserve_official_eval": True,
}
EXPECTED_ARGV_DELTA = {
    "removed_for_training": ["--plan-only"],
    "added_for_training": [],
    "exactly_one_semantic_delta": True,
    "semantic_delta": "remove --plan-only only",
}


class PreflightError(RuntimeError):
    """Raised when fail-closed preflight rejects a launch."""

    def __init__(self, errors: list[str], audit: dict[str, Any]):
        super().__init__("; ".join(errors))
        self.errors = errors
        self.audit = audit


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launch-audit", required=True, type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight-only", action="store_true")
    mode.add_argument("--execute", action="store_true")
    parser.add_argument("--expected-audit-sha256")
    parser.add_argument("--attempt-id", required=True)
    parser.add_argument("--stdout-capture", required=True, type=Path)
    parser.add_argument("--stderr-capture", required=True, type=Path)
    parser.add_argument("--output-audit", required=True, type=Path)
    return parser.parse_args(argv)


def repo_path(path: str | Path) -> Path:
    parsed = Path(path)
    if parsed.is_absolute():
        return parsed
    return REPO_ROOT / parsed


def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def audit_sha256(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def package_stem(package: Path) -> str:
    return package.name[:-4] if package.name.endswith(".mll") else package.stem


def compact_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def sibling_content_rollup(sibling_hashes: list[dict[str, Any]]) -> str:
    """Canonical launch-audit content rollup.

    Byte serialization is one compact, key-sorted JSON object per sibling using
    pathless ``{"bytes", "name", "sha256"}`` content records, in sorted sibling
    order, followed by ``\\n`` for every object including the last one.
    """

    h = hashlib.sha256()
    for item in sibling_hashes:
        h.update(compact_json({"bytes": item["bytes"], "name": item["name"], "sha256": item["sha256"]}).encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def sibling_json_array_rollup(sibling_hashes: list[dict[str, Any]]) -> str:
    content_items = [{"bytes": item["bytes"], "name": item["name"], "sha256": item["sha256"]} for item in sibling_hashes]
    return sha256_bytes(compact_json(content_items).encode("utf-8"))


def sibling_no_final_newline_rollup(sibling_hashes: list[dict[str, Any]]) -> str:
    lines = [
        compact_json({"bytes": item["bytes"], "name": item["name"], "sha256": item["sha256"]})
        for item in sibling_hashes
    ]
    return sha256_bytes("\n".join(lines).encode("utf-8"))


def sibling_path_array_rollup(sibling_hashes_with_paths: list[dict[str, Any]]) -> str:
    return sha256_bytes(compact_json(sibling_hashes_with_paths).encode("utf-8"))


def package_state(target: Path) -> dict[str, Any]:
    if not target.is_file():
        raise ValueError(f"missing package target: {target}")
    stem = package_stem(target)
    siblings = sorted((path for path in target.parent.glob(f"{stem}*") if path.is_file()), key=lambda p: str(p))
    if target.resolve() not in {path.resolve() for path in siblings}:
        raise ValueError(f"target missing from sibling set: {target}")
    pathless = [{"bytes": p.stat().st_size, "name": p.name, "sha256": sha256_file(p)} for p in siblings]
    with_paths = [
        {"bytes": p.stat().st_size, "name": p.name, "path": display_path(p), "sha256": sha256_file(p)}
        for p in siblings
    ]
    tokenizer = target.with_name(f"{stem}.tokenizer.mll")
    tokenizer_sha = sha256_file(tokenizer) if tokenizer.is_file() else None
    return {
        "target": display_path(target),
        "target_absolute": str(target.resolve()),
        "target_sha256": sha256_file(target),
        "dir": display_path(target.parent),
        "sibling_count": len(pathless),
        "sibling_hashes": pathless,
        "sibling_hashes_with_paths": with_paths,
        "content_rollup_sha256": sibling_content_rollup(pathless),
        "json_array_rollup_sha256": sibling_json_array_rollup(pathless),
        "no_final_newline_rollup_sha256": sibling_no_final_newline_rollup(pathless),
        "path_rollup_sha256": sibling_path_array_rollup(with_paths),
        "tokenizer_package": display_path(tokenizer),
        "tokenizer_package_sha256": tokenizer_sha,
    }


def normalize_path_record_path(path_text: str) -> str:
    return display_path(repo_path(path_text))


def normalize_path_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for record in records:
        item = dict(record)
        if "path" in item:
            item["path"] = normalize_path_record_path(str(item["path"]))
        out.append(item)
    return out


def compare_package_node(label: str, expected_node: dict[str, Any], errors: list[str]) -> dict[str, Any]:
    target_text = expected_node.get("target")
    if not target_text:
        errors.append(f"{label} target missing")
        return {}
    try:
        actual = package_state(repo_path(target_text))
    except (OSError, ValueError) as exc:
        errors.append(f"{label} package state error: {exc}")
        return {}

    expected_siblings = expected_node.get("sibling_hashes")
    if expected_siblings != actual["sibling_hashes"]:
        errors.append(f"{label} sibling_hashes mismatch")
    expected_count = expected_node.get("sibling_count")
    if expected_count != actual["sibling_count"]:
        errors.append(f"{label} sibling_count mismatch")
    if expected_node.get("target_sha256") != actual["target_sha256"]:
        errors.append(f"{label} target_sha256 mismatch")
    expected_tokenizer_sha = expected_node.get("tokenizer_package_sha256")
    if expected_tokenizer_sha != actual["tokenizer_package_sha256"]:
        errors.append(f"{label} tokenizer_package_sha256 mismatch")
    expected_paths = expected_node.get("sibling_hashes_with_paths")
    if expected_paths is not None and normalize_path_records(expected_paths) != actual["sibling_hashes_with_paths"]:
        errors.append(f"{label} sibling_hashes_with_paths mismatch")

    declared_rollup = expected_node.get("content_rollup_sha256")
    if declared_rollup != actual["content_rollup_sha256"]:
        errors.append(f"{label} content_rollup_sha256 mismatch")
    if expected_node.get("path_rollup_sha256") is not None and expected_node.get("path_rollup_sha256") != actual["path_rollup_sha256"]:
        errors.append(f"{label} path_rollup_sha256 mismatch")

    return {
        "target": actual["target"],
        "target_sha256": actual["target_sha256"],
        "sibling_count": actual["sibling_count"],
        "sibling_hashes_match_audit": expected_siblings == actual["sibling_hashes"],
        "sibling_hashes_with_paths_match_audit": expected_paths is None or normalize_path_records(expected_paths) == actual["sibling_hashes_with_paths"],
        "content_rollup_sha256": actual["content_rollup_sha256"],
        "declared_content_rollup_sha256": declared_rollup,
        "json_array_rollup_sha256": actual["json_array_rollup_sha256"],
        "no_final_newline_rollup_sha256": actual["no_final_newline_rollup_sha256"],
        "path_rollup_sha256": actual["path_rollup_sha256"],
        "tokenizer_package": actual["tokenizer_package"],
        "tokenizer_package_sha256": actual["tokenizer_package_sha256"],
    }


def walk_pinned_path_hashes(node: Any, path: tuple[str, ...] = ()) -> list[tuple[str, str, str]]:
    found: list[tuple[str, str, str]] = []
    if isinstance(node, dict):
        if "path" in node and "sha256" in node and isinstance(node["path"], str) and isinstance(node["sha256"], str):
            found.append((".".join(path + ("path",)), node["path"], node["sha256"]))
        for key, value in node.items():
            if key in {"sibling_hashes_with_paths"}:
                continue
            found.extend(walk_pinned_path_hashes(value, path + (str(key),)))
    elif isinstance(node, list):
        for idx, value in enumerate(node):
            found.extend(walk_pinned_path_hashes(value, path + (str(idx),)))
    return found


def validate_pinned_path_hashes(launch_audit: dict[str, Any], errors: list[str]) -> list[dict[str, Any]]:
    checked: list[dict[str, Any]] = []
    for field, path_text, expected_sha in walk_pinned_path_hashes(launch_audit):
        path = repo_path(path_text)
        record = {"field": field, "path": display_path(path), "expected_sha256": expected_sha, "ok": False}
        if not path.is_file():
            errors.append(f"pinned path missing: {field} {path_text}")
        else:
            actual = sha256_file(path)
            record["actual_sha256"] = actual
            record["ok"] = actual == expected_sha
            if actual != expected_sha:
                errors.append(f"pinned path sha256 mismatch: {field} {path_text}")
        checked.append(record)

    plan_result = launch_audit.get("plan_result", {})
    for stream in ("stdout", "stderr"):
        path_key = f"{stream}_path"
        sha_key = f"{stream}_sha256"
        if path_key in plan_result and sha_key in plan_result:
            path = repo_path(plan_result[path_key])
            record = {
                "field": f"plan_result.{path_key}",
                "path": display_path(path),
                "expected_sha256": plan_result[sha_key],
                "ok": False,
            }
            if not path.is_file():
                errors.append(f"plan_result {stream} missing: {plan_result[path_key]}")
            else:
                actual = sha256_file(path)
                record["actual_sha256"] = actual
                record["ok"] = actual == plan_result[sha_key]
                if actual != plan_result[sha_key]:
                    errors.append(f"plan_result {stream} sha256 mismatch: {plan_result[path_key]}")
            checked.append(record)
    return checked


def option_value(argv: list[str], flag: str) -> str | None:
    try:
        idx = argv.index(flag)
    except ValueError:
        return None
    if idx + 1 >= len(argv):
        return None
    return argv[idx + 1]


def remove_one_plan_only(argv: list[str]) -> list[str]:
    out = list(argv)
    out.remove("--plan-only")
    return out


def validate_argv(launch_audit: dict[str, Any], errors: list[str]) -> dict[str, Any]:
    plan = launch_audit.get("plan_argv", [])
    future = launch_audit.get("future_training_argv", [])
    metrics_target = launch_audit.get("metrics_target")
    candidate_target = launch_audit.get("candidate", {}).get("target")
    anchor_target = launch_audit.get("canonical_anchor", {}).get("pre", {}).get("target")
    details = {
        "plan_contains_plan_only": "--plan-only" in plan,
        "future_contains_plan_only": "--plan-only" in future,
        "plan_contains_no_tokenizer": "--no-tokenizer" in plan,
        "future_contains_no_tokenizer": "--no-tokenizer" in future,
        "future_contains_candidate_target": candidate_target in future,
        "future_contains_anchor_target": anchor_target in future,
        "metrics_target": metrics_target,
        "future_metrics_value": option_value(future, "--metrics-json"),
        "plan_metrics_value": option_value(plan, "--metrics-json"),
    }
    if not isinstance(plan, list) or not isinstance(future, list):
        errors.append("plan_argv/future_training_argv must be lists")
        return details
    if "--plan-only" not in plan:
        errors.append("plan_argv missing --plan-only")
    if "--plan-only" in future:
        errors.append("future_training_argv contains forbidden --plan-only")
    if "--no-tokenizer" in plan or "--no-tokenizer" in future:
        errors.append("forbidden --no-tokenizer present in plan or future argv")
    if remove_one_plan_only(plan) != future:
        errors.append("future_training_argv is not exactly plan_argv with one --plan-only removed")
    if launch_audit.get("argv_delta") != EXPECTED_ARGV_DELTA:
        errors.append("argv_delta drift")
    if option_value(plan, "--metrics-json") != metrics_target:
        errors.append("metrics target missing/mismatched in plan_argv")
    if option_value(future, "--metrics-json") != metrics_target:
        errors.append("metrics target missing/mismatched in future_training_argv")
    if candidate_target not in future:
        errors.append("future_training_argv missing candidate target")
    if anchor_target in future:
        errors.append("future_training_argv targets canonical anchor")
    if len(future) < 2 or future[1] != "train-embed":
        errors.append("future_training_argv is not a train-embed command")
    return details


def existing_file(path: str | Path) -> bool:
    return repo_path(path).exists()


def build_preflight_audit(
    *,
    args: argparse.Namespace,
    launch_audit: dict[str, Any],
    launch_audit_sha: str,
    errors: list[str],
    pinned_path_hashes: list[dict[str, Any]],
    package_actuals: dict[str, Any],
    argv_details: dict[str, Any],
) -> dict[str, Any]:
    checks = {
        "launch_audit_schema": launch_audit.get("schema") == LAUNCH_AUDIT_SCHEMA,
        "expected_audit_sha256_matches": args.expected_audit_sha256 is None or args.expected_audit_sha256 == launch_audit_sha,
        "execute_requires_expected_audit_sha256": (not args.execute) or bool(args.expected_audit_sha256),
        "legal_scope_exact": launch_audit.get("legal_scope") == EXPECTED_LEGAL_SCOPE,
        "actual_training_not_already_recorded": launch_audit.get("actual_training_ran") is False,
        "launch_audit_checks_passed": launch_audit.get("all_checks_passed") is True,
        "argv_delta_remove_plan_only_only": launch_audit.get("argv_delta") == EXPECTED_ARGV_DELTA,
        "workload_exact": launch_audit.get("parsed_workload") == EXPECTED_PARSED_WORKLOAD,
        "plan_exit_status_zero": launch_audit.get("plan_result", {}).get("exit_status") == 0,
        "candidate_target_differs_from_anchor": False,
        "candidate_content_rollup_matches_anchor": False,
        "candidate_pre_post_rollup_stable": False,
        "anchor_pre_post_rollup_stable": False,
        "all_pinned_path_hashes_match": all(item.get("ok") for item in pinned_path_hashes),
        "metrics_target_absent": False,
        "stdout_capture_absent": not repo_path(args.stdout_capture).exists(),
        "stderr_capture_absent": not repo_path(args.stderr_capture).exists(),
        "output_audit_absent_before_write": not repo_path(args.output_audit).exists(),
        "preflight_only_no_spawn": bool(args.preflight_only),
        "execute_mode_requested": bool(args.execute),
    }
    candidate_target = launch_audit.get("candidate", {}).get("target")
    anchor_target = launch_audit.get("canonical_anchor", {}).get("pre", {}).get("target")
    if candidate_target and anchor_target:
        checks["candidate_target_differs_from_anchor"] = repo_path(candidate_target).resolve() != repo_path(anchor_target).resolve()
    candidate_pre = package_actuals.get("candidate.pre_plan", {})
    candidate_post = package_actuals.get("candidate.post_plan", {})
    anchor_pre = package_actuals.get("canonical_anchor.pre", {})
    anchor_post = package_actuals.get("canonical_anchor.post", {})
    checks["candidate_content_rollup_matches_anchor"] = (
        candidate_pre.get("content_rollup_sha256") == anchor_pre.get("content_rollup_sha256")
        and bool(candidate_pre.get("content_rollup_sha256"))
    )
    checks["candidate_pre_post_rollup_stable"] = (
        candidate_pre.get("content_rollup_sha256") == candidate_post.get("content_rollup_sha256")
        and bool(candidate_pre.get("content_rollup_sha256"))
    )
    checks["anchor_pre_post_rollup_stable"] = (
        anchor_pre.get("content_rollup_sha256") == anchor_post.get("content_rollup_sha256")
        and bool(anchor_pre.get("content_rollup_sha256"))
    )
    metrics_target = launch_audit.get("metrics_target")
    checks["metrics_target_absent"] = bool(metrics_target) and not existing_file(metrics_target)
    checks["argv_legal"] = not any(
        [
            argv_details.get("future_contains_plan_only"),
            argv_details.get("plan_contains_no_tokenizer"),
            argv_details.get("future_contains_no_tokenizer"),
            not argv_details.get("future_contains_candidate_target"),
            argv_details.get("future_contains_anchor_target"),
            argv_details.get("future_metrics_value") != metrics_target,
            argv_details.get("plan_metrics_value") != metrics_target,
        ]
    )
    required_check_values = [
        value
        for key, value in checks.items()
        if key
        not in {
            "execute_mode_requested",
            "all_checks_true",
        }
    ]
    checks["all_checks_true"] = not errors and all(required_check_values)
    return {
        "schema": PREFLIGHT_SCHEMA,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "mode": "execute" if args.execute else "preflight-only",
        "attempt_id": args.attempt_id,
        "launch_audit_path": display_path(repo_path(args.launch_audit)),
        "launch_audit_sha256": launch_audit_sha,
        "expected_audit_sha256": args.expected_audit_sha256,
        "stdout_capture_path": display_path(repo_path(args.stdout_capture)),
        "stderr_capture_path": display_path(repo_path(args.stderr_capture)),
        "output_audit_path": display_path(repo_path(args.output_audit)),
        "stdout_stderr_capture_note": "capture paths are external launcher outputs and are distinct from plan stdout/stderr and metrics target",
        "trainer_process_spawned": False,
        "actual_training_ran": False,
        "future_training_argv": copy.deepcopy(launch_audit.get("future_training_argv")),
        "metrics_target": launch_audit.get("metrics_target"),
        "rollup_algorithm": {
            "content_rollup_sha256": "sha256 of sorted pathless sibling records; each record is compact key-sorted JSON bytes followed by LF, including the final record",
            "record_fields": ["bytes", "name", "sha256"],
            "sort": "sibling file path/name lexical order",
            "not_used_by_postfix_audit": {
                "json_array_rollup": "old direct compact JSON array hash; rejected for this audit",
                "no_final_newline_rollup": "line format without trailing LF; rejected",
                "path_rollup": "pathful records are checked separately when pinned, but not used for content identity",
            },
        },
        "package_actuals": package_actuals,
        "pinned_path_hashes": pinned_path_hashes,
        "argv_details": argv_details,
        "expected_workload": EXPECTED_PARSED_WORKLOAD,
        "parsed_workload": launch_audit.get("parsed_workload"),
        "checks": checks,
        "errors": errors,
    }


def write_audit_new(path: Path, audit: dict[str, Any]) -> None:
    path = repo_path(path)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite output audit: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def preflight(args: argparse.Namespace) -> dict[str, Any]:
    errors: list[str] = []
    launch_path = repo_path(args.launch_audit)
    if not launch_path.is_file():
        raise FileNotFoundError(f"missing launch audit: {launch_path}")
    launch_audit = json.loads(launch_path.read_text(encoding="utf-8"))
    launch_sha = audit_sha256(launch_path)

    if launch_audit.get("schema") != LAUNCH_AUDIT_SCHEMA:
        errors.append("launch audit schema drift")
    if args.expected_audit_sha256 is not None and args.expected_audit_sha256 != launch_sha:
        errors.append("launch audit sha256 mismatch")
    if args.execute and not args.expected_audit_sha256:
        errors.append("execute mode requires --expected-audit-sha256")
    if launch_audit.get("legal_scope") != EXPECTED_LEGAL_SCOPE:
        errors.append("legal_scope drift")
    if launch_audit.get("actual_training_ran") is not False:
        errors.append("launch audit does not record actual_training_ran=false")
    if launch_audit.get("all_checks_passed") is not True:
        errors.append("launch audit all_checks_passed is not true")
    if launch_audit.get("parsed_workload") != EXPECTED_PARSED_WORKLOAD:
        errors.append("parsed workload drift")
    if launch_audit.get("plan_result", {}).get("exit_status") != 0:
        errors.append("plan_result exit_status is not zero")

    package_actuals: dict[str, Any] = {}
    for label, node in (
        ("canonical_anchor.pre", launch_audit.get("canonical_anchor", {}).get("pre", {})),
        ("canonical_anchor.post", launch_audit.get("canonical_anchor", {}).get("post", {})),
        ("candidate.pre_plan", launch_audit.get("candidate", {}).get("pre_plan", {})),
        ("candidate.post_plan", launch_audit.get("candidate", {}).get("post_plan", {})),
    ):
        package_actuals[label] = compare_package_node(label, node, errors)

    pinned_path_hashes = validate_pinned_path_hashes(launch_audit, errors)
    argv_details = validate_argv(launch_audit, errors)

    candidate_target = launch_audit.get("candidate", {}).get("target")
    anchor_target = launch_audit.get("canonical_anchor", {}).get("pre", {}).get("target")
    if candidate_target and anchor_target and repo_path(candidate_target).resolve() == repo_path(anchor_target).resolve():
        errors.append("candidate target equals canonical anchor target")

    for label, candidate in (
        ("metrics target", launch_audit.get("metrics_target")),
        ("stdout capture", args.stdout_capture),
        ("stderr capture", args.stderr_capture),
        ("output audit", args.output_audit),
    ):
        if candidate and repo_path(candidate).exists():
            errors.append(f"{label} already exists: {candidate}")

    audit = build_preflight_audit(
        args=args,
        launch_audit=launch_audit,
        launch_audit_sha=launch_sha,
        errors=errors,
        pinned_path_hashes=pinned_path_hashes,
        package_actuals=package_actuals,
        argv_details=argv_details,
    )
    if errors:
        raise PreflightError(errors, audit)
    return audit


def execute_once(args: argparse.Namespace, audit: dict[str, Any]) -> dict[str, Any]:
    stdout_path = repo_path(args.stdout_capture)
    stderr_path = repo_path(args.stderr_capture)
    if stdout_path.exists() or stderr_path.exists():
        raise FileExistsError("capture path appeared after preflight")
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stderr_path.parent.mkdir(parents=True, exist_ok=True)
    argv = audit["future_training_argv"]
    with stdout_path.open("xb") as stdout_fh, stderr_path.open("xb") as stderr_fh:
        proc = subprocess.run(argv, stdout=stdout_fh, stderr=stderr_fh, check=False)
    audit["trainer_process_spawned"] = True
    audit["actual_training_ran"] = True
    audit["execute_result"] = {
        "exit_status": proc.returncode,
        "stdout_capture_sha256": sha256_file(stdout_path),
        "stderr_capture_sha256": sha256_file(stderr_path),
    }
    return audit


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        audit = preflight(args)
        if args.execute:
            audit = execute_once(args, audit)
        write_audit_new(args.output_audit, audit)
    except PreflightError as exc:
        try:
            write_audit_new(args.output_audit, exc.audit)
        except FileExistsError:
            pass
        print(f"preflight rejected: {'; '.join(exc.errors)}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"launcher failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({"ok": True, "output_audit": display_path(repo_path(args.output_audit))}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
