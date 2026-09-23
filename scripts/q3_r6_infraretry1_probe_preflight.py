#!/usr/bin/env python3
"""Infraretry1-specific wrapper around the accepted q3 R6 probe preflight.

This keeps the original probe gate semantics intact, but binds the approved
infraretry1 binary identity and exact rebound manifest hash before delegating
to scripts/q3_r6_probe_preflight.py.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import q3_r6_probe_preflight as r6


REPO_ROOT = Path(__file__).resolve().parent.parent
RUN_ROOT = Path("runs/eos-d384-q3-boundary-goal-v1-20260902T084144Z")
SOURCE_R6_ROOT = RUN_ROOT / "headroom-q3r6-frontier-retain-alltrain-b0-rw050"
R6_ROOT = RUN_ROOT / "headroom-q3r6-frontier-retain-alltrain-b0-rw050-infraretry1"
SOURCE_ARM_ID = "arm-q3r6-frontier-retain-alltrain-b0-rw050-q3w004-m002-lr2e6-e2"
ARM_ID = f"{SOURCE_ARM_ID}-infraretry1"
INFRARETRY_BINARY = RUN_ROOT / "bin/eos-r6-frontier-retention-infraretry1"
INFRARETRY_BINARY_SHA256 = "c5679853f8494c1e144a85d3cf2a345b0a3236f917a42607ddebc2242d344f90"
INFRARETRY_PROBE_MANIFEST = R6_ROOT / "probe/q3r6-frontier-retention.probe-manifest.json"
INFRARETRY_PROBE_MANIFEST_SHA256 = "05b6c3de0d81d1e1559dd1f6a4fcf83f0b40a06fa95ac103c8e664bfdff90c54"
SEMANTIC_DIFF_AUDIT = R6_ROOT / "probe/q3-r6-infraretry1-probe-semantic-diff-audit.json"
SEMANTIC_DIFF_AUDIT_SHA256 = "4bfc3cb9698e94ffafccd31ca2efe23780c95d0b1f9b693fd9ae287325dd95e7"
POSTTRAIN_ROLLUP_SHA256 = "65c0a44482353a05682d161a5f3f0edf579a54c3f597b5424e2e7820ef83b6cf"
METRICS_PATH = R6_ROOT / "metrics/arm-q3r6-frontier-retain-alltrain-b0-rw050-q3w004-m002-lr2e6-e2-infraretry1.train.metrics.json"
METRICS_SHA256 = "7f955bd8eed270b9b27ba6d4a2890f40c250e20bbf1f9f6289fc2179e710dc01"
POSTTRAIN_ATTESTATION = R6_ROOT / "reports/q3-r6-infraretry1-posttrain-attestation.json"
POSTTRAIN_ATTESTATION_SHA256 = "b59ab5a41e1d82ef2bfd3eb377d0309f597a301ed34c37fd69177a23af01280e"
EFFECTIVE_BASE_LEAKAGE_AUDIT = R6_ROOT / "reports/q3-r6-infraretry1-candidate-effective-base-leakage-audit.json"
EFFECTIVE_BASE_LEAKAGE_AUDIT_SHA256 = "80308a4a9bb3413a159b63aedf25d86c6b3e4f239e6f4af7202afd22a7ce59f3"
INSPECT_STDOUT_SHA256 = "c98c8b549e5e2422b2bf09012a2ba5a6d1c3e922bda12d08e587f7845979006a"


class InfraretryProbeContractError(r6.ProbeContractError):
    pass


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-manifest", required=True, type=Path)
    parser.add_argument("--expected-manifest-sha256", default=INFRARETRY_PROBE_MANIFEST_SHA256)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--allow-pending-runtime-bindings", action="store_true")
    parser.add_argument("--strict-with-pending", action="store_true", help="validate pretrain manifest and report expected post-train pending bindings")
    return parser.parse_args(argv)


def repo_path(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


def display_path(path: str | Path) -> str:
    resolved = repo_path(path).resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        return str(resolved)


def validate_infraretry_manifest_binding(
    manifest_path: Path,
    expected_manifest_sha256: str,
    *,
    require_outputs_absent: bool = True,
) -> dict[str, Any]:
    resolved_manifest = repo_path(manifest_path).resolve()
    expected_manifest = repo_path(INFRARETRY_PROBE_MANIFEST).resolve()
    if resolved_manifest != expected_manifest:
        raise InfraretryProbeContractError("infraretry1 probe manifest path mismatch")
    actual_manifest_sha = r6.sha256_file(resolved_manifest)
    if actual_manifest_sha != expected_manifest_sha256 or actual_manifest_sha != INFRARETRY_PROBE_MANIFEST_SHA256:
        raise InfraretryProbeContractError("infraretry1 probe manifest sha mismatch")
    diff_path = repo_path(SEMANTIC_DIFF_AUDIT).resolve()
    if r6.sha256_file(diff_path) != SEMANTIC_DIFF_AUDIT_SHA256:
        raise InfraretryProbeContractError("infraretry1 semantic diff audit sha mismatch")
    diff = r6.read_json(diff_path)
    if diff.get("allowed_identity_path_changes_only") is not True or not all(diff.get("checks", {}).values()):
        raise InfraretryProbeContractError("infraretry1 semantic diff audit failed")
    manifest = r6.read_json(resolved_manifest)
    binary = manifest.get("binary", {})
    bound_binary = display_path(INFRARETRY_BINARY)
    if binary.get("path") != bound_binary or binary.get("sha256") != INFRARETRY_BINARY_SHA256:
        raise InfraretryProbeContractError("infraretry1 binary manifest binding mismatch")
    if r6.sha256_file(repo_path(binary["path"])) != INFRARETRY_BINARY_SHA256:
        raise InfraretryProbeContractError("infraretry1 binary file sha mismatch")
    candidate = manifest.get("candidate_package", {})
    if candidate.get("sibling_rollup_sha256") != POSTTRAIN_ROLLUP_SHA256:
        raise InfraretryProbeContractError("infraretry1 candidate posttrain rollup mismatch")
    metrics = manifest.get("candidate_train_metrics", {})
    if metrics.get("path") != display_path(METRICS_PATH) or metrics.get("sha256") != METRICS_SHA256:
        raise InfraretryProbeContractError("infraretry1 metrics manifest binding mismatch")
    if r6.sha256_file(repo_path(metrics["path"])) != METRICS_SHA256:
        raise InfraretryProbeContractError("infraretry1 metrics file sha mismatch")
    attestation = manifest.get("posttrain_attestation", {})
    if attestation.get("path") != display_path(POSTTRAIN_ATTESTATION) or attestation.get("sha256") != POSTTRAIN_ATTESTATION_SHA256 or attestation.get("status") != "passed":
        raise InfraretryProbeContractError("infraretry1 posttrain attestation binding mismatch")
    if r6.sha256_file(repo_path(attestation["path"])) != POSTTRAIN_ATTESTATION_SHA256:
        raise InfraretryProbeContractError("infraretry1 posttrain attestation sha mismatch")
    base_audit = manifest.get("candidate_effective_base_leakage_audit", {})
    if (
        base_audit.get("path") != display_path(EFFECTIVE_BASE_LEAKAGE_AUDIT)
        or base_audit.get("sha256") != EFFECTIVE_BASE_LEAKAGE_AUDIT_SHA256
        or base_audit.get("status") != "passed"
        or base_audit.get("base_hard_soft_recovery_work_units") != 0
        or base_audit.get("aux_only") is not True
    ):
        raise InfraretryProbeContractError("infraretry1 effective-base leakage audit binding mismatch")
    if r6.sha256_file(repo_path(base_audit["path"])) != EFFECTIVE_BASE_LEAKAGE_AUDIT_SHA256:
        raise InfraretryProbeContractError("infraretry1 effective-base leakage audit sha mismatch")
    inspect = manifest.get("supported_package_inspect_evidence", {})
    if (
        inspect.get("command") != [bound_binary, "inspect", candidate.get("path")]
        or inspect.get("exit_status") != 0
        or inspect.get("package_verify_ok") is not True
        or inspect.get("train_profile_step") != 578
        or inspect.get("stdout_sha256") != INSPECT_STDOUT_SHA256
        or inspect.get("stderr_sha256") != r6.sha256(b"").hexdigest()
    ):
        raise InfraretryProbeContractError("infraretry1 supported package inspect evidence mismatch")
    commands = manifest.get("commands", [])
    if len(commands) != 14:
        raise InfraretryProbeContractError("infraretry1 command count mismatch")
    for idx, command in enumerate(commands):
        argv = command.get("argv")
        if not isinstance(argv, list) or not argv or argv[0] != bound_binary:
            raise InfraretryProbeContractError(f"infraretry1 command {idx} binary argv mismatch")
    outputs = manifest.get("expected_outputs", [])
    outputs_root = repo_path(manifest.get("future_paths", {}).get("outputs_root", "")).resolve()
    if len(outputs) != 42 or len(set(outputs)) != 42:
        raise InfraretryProbeContractError("infraretry1 expected output set mismatch")
    for output in outputs:
        output_path = r6.require_resolved_within(output, outputs_root, REPO_ROOT, f"infraretry1 future output path escapes outputs root: {output}")
        if require_outputs_absent and output_path.exists():
            raise InfraretryProbeContractError(f"infraretry1 future output path already exists: {output}")
    return {
        "manifest_path": display_path(resolved_manifest),
        "manifest_sha256": actual_manifest_sha,
        "binary_path": bound_binary,
        "binary_sha256": INFRARETRY_BINARY_SHA256,
        "semantic_diff_audit": {"path": display_path(diff_path), "sha256": SEMANTIC_DIFF_AUDIT_SHA256},
        "posttrain_attestation": {"path": display_path(POSTTRAIN_ATTESTATION), "sha256": POSTTRAIN_ATTESTATION_SHA256},
        "effective_base_leakage_audit": {"path": display_path(EFFECTIVE_BASE_LEAKAGE_AUDIT), "sha256": EFFECTIVE_BASE_LEAKAGE_AUDIT_SHA256},
        "metrics": {"path": display_path(METRICS_PATH), "sha256": METRICS_SHA256},
        "candidate_posttrain_rollup_sha256": POSTTRAIN_ROLLUP_SHA256,
        "command_count": len(commands),
        "expected_output_count": len(outputs),
    }


def validate_manifest(manifest_path: Path, *, expected_manifest_sha256: str = INFRARETRY_PROBE_MANIFEST_SHA256, require_outputs_absent: bool = True, allow_pending_runtime_bindings: bool = True) -> dict[str, Any]:
    binding = validate_infraretry_manifest_binding(manifest_path, expected_manifest_sha256, require_outputs_absent=require_outputs_absent)
    original_binary_sha = r6.EXPECTED_BINARY_SHA256
    try:
        r6.EXPECTED_BINARY_SHA256 = INFRARETRY_BINARY_SHA256
        manifest = r6.validate_manifest(manifest_path, require_outputs_absent=require_outputs_absent, allow_pending_runtime_bindings=allow_pending_runtime_bindings)
    finally:
        r6.EXPECTED_BINARY_SHA256 = original_binary_sha
    manifest["_infraretry1_binding"] = binding
    return manifest


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        allow_pending = args.allow_pending_runtime_bindings or args.strict_with_pending
        if args.preflight_only or args.strict_with_pending:
            validate_manifest(args.probe_manifest, expected_manifest_sha256=args.expected_manifest_sha256, require_outputs_absent=True, allow_pending_runtime_bindings=allow_pending)
            if args.strict_with_pending:
                print(json.dumps({"ok": True, "pending": []}, sort_keys=True))
            else:
                print("q3 R6 infraretry1 probe preflight: OK")
            return 0
        original_binary_sha = r6.EXPECTED_BINARY_SHA256
        try:
            validate_infraretry_manifest_binding(args.probe_manifest, args.expected_manifest_sha256, require_outputs_absent=False)
            r6.EXPECTED_BINARY_SHA256 = INFRARETRY_BINARY_SHA256
            summary = r6.validate_gate(args.probe_manifest)
        finally:
            r6.EXPECTED_BINARY_SHA256 = original_binary_sha
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    except r6.ProbeContractError as exc:
        print(f"ProbeContractError: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
