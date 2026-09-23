#!/usr/bin/env python3
"""Fail-closed launcher for the accepted Q3-R6-INFRARETRY1 movement probe."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import q3_r6_infraretry1_probe_preflight as preflight


REPO_ROOT = Path(__file__).resolve().parent.parent
RUN_ROOT = Path("runs/eos-d384-q3-boundary-goal-v1-20260902T084144Z")
R6_ROOT = RUN_ROOT / "headroom-q3r6-frontier-retain-alltrain-b0-rw050-infraretry1"
PROBE_MANIFEST = R6_ROOT / "probe/q3r6-frontier-retention.probe-manifest.json"
REPORTS_ROOT = R6_ROOT / "probe/reports"
DEFAULT_PLAN_AUDIT = REPORTS_ROOT / "q3-r6-infraretry1-probe-launch-plan-audit.json"
DEFAULT_PLAN_RECEIPT = REPORTS_ROOT / "q3-r6-infraretry1-probe-launch-plan-receipt.json"
DEFAULT_PLAN_STDOUT = REPORTS_ROOT / "q3-r6-infraretry1-probe-launch-plan.stdout.txt"
DEFAULT_PLAN_STDERR = REPORTS_ROOT / "q3-r6-infraretry1-probe-launch-plan.stderr.txt"
DEFAULT_EXECUTE_CAPTURE_DIR = REPORTS_ROOT / "q3-r6-infraretry1-probe-execute-captures"
DEFAULT_EXECUTE_AUDIT = REPORTS_ROOT / "q3-r6-infraretry1-probe-execute-audit.json"

PLAN_SCHEMA = "eos.q3_r6_infraretry1_probe_launch_plan.v1"
RECEIPT_SCHEMA = "eos.q3_r6_infraretry1_probe_launch_receipt.v1"
EXECUTE_SCHEMA = "eos.q3_r6_infraretry1_probe_execute_audit.v1"
EXPECTED_MANIFEST_SHA256 = "05b6c3de0d81d1e1559dd1f6a4fcf83f0b40a06fa95ac103c8e664bfdff90c54"
EXPECTED_WRAPPER_SHA256 = "45388a0e71041b84091e47a0781d8cfda79260b33c8b69f508dd2e7166bba6e0"
EXPECTED_BINARY_SHA256 = "c5679853f8494c1e144a85d3cf2a345b0a3236f917a42607ddebc2242d344f90"
EXPECTED_CANDIDATE_ROLLUP_SHA256 = "65c0a44482353a05682d161a5f3f0edf579a54c3f597b5424e2e7820ef83b6cf"
EXPECTED_METRICS_SHA256 = "7f955bd8eed270b9b27ba6d4a2890f40c250e20bbf1f9f6289fc2179e710dc01"
EXPECTED_ATTESTATION_SHA256 = "b59ab5a41e1d82ef2bfd3eb377d0309f597a301ed34c37fd69177a23af01280e"
EXPECTED_BASE_AUDIT_SHA256 = "80308a4a9bb3413a159b63aedf25d86c6b3e4f239e6f4af7202afd22a7ce59f3"
EXPECTED_SEMANTIC_DIFF_AUDIT_SHA256 = "4bfc3cb9698e94ffafccd31ca2efe23780c95d0b1f9b693fd9ae287325dd95e7"
GATE_COMMAND = [
    "python",
    "scripts/q3_r6_infraretry1_probe_preflight.py",
    "--probe-manifest",
    str(PROBE_MANIFEST),
    "--expected-manifest-sha256",
    EXPECTED_MANIFEST_SHA256,
]


class LauncherError(RuntimeError):
    def __init__(self, errors: list[str], audit: dict[str, Any] | None = None):
        super().__init__("; ".join(errors))
        self.errors = errors
        self.audit = audit or {"errors": errors}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--plan-only", action="store_true")
    mode.add_argument("--execute", action="store_true")
    parser.add_argument("--probe-manifest", type=Path, default=PROBE_MANIFEST)
    parser.add_argument("--expected-manifest-sha256", default=EXPECTED_MANIFEST_SHA256)
    parser.add_argument("--plan-audit", type=Path, default=DEFAULT_PLAN_AUDIT)
    parser.add_argument("--plan-receipt", type=Path, default=DEFAULT_PLAN_RECEIPT)
    parser.add_argument("--stdout-capture", type=Path, default=DEFAULT_PLAN_STDOUT)
    parser.add_argument("--stderr-capture", type=Path, default=DEFAULT_PLAN_STDERR)
    parser.add_argument("--expected-plan-audit-sha256")
    parser.add_argument("--execute-capture-dir", type=Path, default=DEFAULT_EXECUTE_CAPTURE_DIR)
    parser.add_argument("--output-audit", type=Path, default=DEFAULT_EXECUTE_AUDIT)
    parser.add_argument("--overwrite-plan", action="store_true")
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


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with repo_path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(repo_path(path).read_text(encoding="utf-8"))


def stable_json(obj: Any) -> str:
    return json.dumps(obj, indent=2, sort_keys=True) + "\n"


def write_text_new(path: str | Path, text: str, *, overwrite: bool = False) -> None:
    out = repo_path(path)
    if out.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite {display_path(out)}")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")


def write_json_new(path: str | Path, obj: Any, *, overwrite: bool = False) -> None:
    write_text_new(path, stable_json(obj), overwrite=overwrite)


def package_stem(package: Path) -> str:
    return package.name[:-4] if package.name.endswith(".mll") else package.stem


def sibling_rollup(target: str | Path) -> dict[str, Any]:
    target_path = repo_path(target)
    if not target_path.is_file():
        raise LauncherError([f"missing package target: {display_path(target_path)}"])
    stem = package_stem(target_path)
    siblings = sorted((p for p in target_path.parent.glob(f"{stem}*") if p.is_file()), key=lambda p: p.name)
    records = [{"name": p.name, "bytes": p.stat().st_size, "sha256": sha256_file(p)} for p in siblings]
    h = hashlib.sha256()
    for item in records:
        h.update(json.dumps(item, sort_keys=True, separators=(",", ":")).encode("utf-8"))
        h.update(b"\n")
    return {
        "target": display_path(target_path),
        "target_sha256": sha256_file(target_path),
        "sibling_count": len(records),
        "sibling_hashes": records,
        "sibling_rollup_sha256": h.hexdigest(),
    }


def require_within(path: str | Path, root: str | Path, errors: list[str], label: str) -> Path:
    resolved = repo_path(path).resolve()
    resolved_root = repo_path(root).resolve()
    try:
        resolved.relative_to(resolved_root)
    except ValueError:
        errors.append(f"{label} escapes {display_path(resolved_root)}: {display_path(resolved)}")
    return resolved


def file_record(path: str | Path) -> dict[str, Any]:
    p = repo_path(path)
    return {"path": display_path(p), "sha256": sha256_file(p), "bytes": p.stat().st_size}


def capture_path(capture_dir: str | Path, index: int, stream: str) -> Path:
    return repo_path(capture_dir) / f"{index:02d}.{stream}.txt"


def matching_live_processes(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    binary = str(repo_path(manifest["binary"]["path"]).resolve())
    manifest_path = str(repo_path(PROBE_MANIFEST).resolve())
    self_pid = os.getpid()
    matches: list[dict[str, Any]] = []
    proc_root = Path("/proc")
    if not proc_root.is_dir():
        return matches
    for entry in proc_root.iterdir():
        if not entry.name.isdigit() or int(entry.name) == self_pid:
            continue
        try:
            raw = (entry / "cmdline").read_bytes()
        except OSError:
            continue
        parts = [p.decode("utf-8", "replace") for p in raw.split(b"\0") if p]
        if not parts:
            continue
        exe0 = str(repo_path(parts[0]).resolve()) if "/" in parts[0] else parts[0]
        is_probe = exe0 == binary and len(parts) > 1 and parts[1] == "eval-retrieval-turboquant"
        is_trainer = exe0 == binary and len(parts) > 1 and parts[1] == "train-embed"
        is_gate = "scripts/q3_r6_infraretry1_probe_preflight.py" in parts and manifest_path in [str(repo_path(p).resolve()) if "/" in p else p for p in parts]
        if is_probe or is_trainer or is_gate:
            matches.append({"pid": int(entry.name), "argv": parts})
    return matches


def load_and_validate_manifest(*, require_outputs_absent: bool) -> dict[str, Any]:
    manifest_path = repo_path(PROBE_MANIFEST)
    manifest = preflight.validate_manifest(
        manifest_path,
        expected_manifest_sha256=EXPECTED_MANIFEST_SHA256,
        require_outputs_absent=require_outputs_absent,
        allow_pending_runtime_bindings=False,
    )
    return manifest


def validate_static_contract(args: argparse.Namespace, *, require_outputs_absent: bool, reject_live: bool) -> dict[str, Any]:
    errors: list[str] = []
    if repo_path(args.probe_manifest).resolve() != repo_path(PROBE_MANIFEST).resolve():
        errors.append("probe manifest path mismatch")
    if args.expected_manifest_sha256 != EXPECTED_MANIFEST_SHA256:
        errors.append("expected manifest sha mismatch")
    for label, path, expected in (
        ("manifest", PROBE_MANIFEST, EXPECTED_MANIFEST_SHA256),
        ("wrapper", "scripts/q3_r6_infraretry1_probe_preflight.py", EXPECTED_WRAPPER_SHA256),
    ):
        actual = sha256_file(path)
        if actual != expected:
            errors.append(f"{label} sha mismatch: {actual}")
    if errors:
        raise LauncherError(errors)
    try:
        manifest = load_and_validate_manifest(require_outputs_absent=require_outputs_absent)
    except Exception as exc:
        raise LauncherError([f"strict preflight failed: {exc}"]) from exc
    outputs_root = manifest["future_paths"]["outputs_root"]
    commands = manifest["commands"]
    outputs = manifest["expected_outputs"]
    if len(manifest.get("surfaces", {})) != 7 or len(commands) != 14 or len(outputs) != 42:
        errors.append("manifest does not have exact 7/14/42 shape")
    if manifest.get("binary", {}).get("sha256") != EXPECTED_BINARY_SHA256:
        errors.append("binary sha not pinned to accepted infraretry1 binary")
    if manifest.get("candidate_package", {}).get("sibling_rollup_sha256") != EXPECTED_CANDIDATE_ROLLUP_SHA256:
        errors.append("candidate rollup not pinned to accepted posttrain rollup")
    accepted_hashes = {
        "metrics": (manifest.get("candidate_train_metrics", {}).get("sha256"), EXPECTED_METRICS_SHA256),
        "attestation": (manifest.get("posttrain_attestation", {}).get("sha256"), EXPECTED_ATTESTATION_SHA256),
        "base_audit": (manifest.get("candidate_effective_base_leakage_audit", {}).get("sha256"), EXPECTED_BASE_AUDIT_SHA256),
        "semantic_diff_audit": (preflight.SEMANTIC_DIFF_AUDIT_SHA256, EXPECTED_SEMANTIC_DIFF_AUDIT_SHA256),
    }
    for label, (actual, expected) in accepted_hashes.items():
        if actual != expected:
            errors.append(f"{label} accepted sha mismatch")
    outputs_seen = []
    for idx, command in enumerate(commands):
        argv = command.get("argv")
        if not isinstance(argv, list) or not argv:
            errors.append(f"command {idx} argv malformed")
            continue
        if any(not isinstance(part, str) or part == "" for part in argv):
            errors.append(f"command {idx} argv contains non-string/empty part")
        if any(part in {";", "&&", "|"} for part in argv):
            errors.append(f"command {idx} argv contains shell token")
        for output in command.get("outputs", {}).values():
            outputs_seen.append(output)
            require_within(output, outputs_root, errors, f"command {idx} output")
    for output in outputs:
        path = require_within(output, outputs_root, errors, "expected output")
        if require_outputs_absent and path.exists():
            errors.append(f"expected output exists: {display_path(path)}")
    if sorted(outputs_seen) != sorted(outputs):
        errors.append("command outputs are not exactly manifest expected outputs")
    if reject_live:
        live = matching_live_processes(manifest)
        if live:
            errors.append(f"matching live probe/eval/trainer/gate process exists: {live}")
    candidate_state = sibling_rollup(manifest["candidate_package"]["path"])
    if candidate_state["sibling_rollup_sha256"] != EXPECTED_CANDIDATE_ROLLUP_SHA256:
        errors.append("candidate sibling rollup drift")
    binary_actual = sha256_file(manifest["binary"]["path"])
    if binary_actual != EXPECTED_BINARY_SHA256:
        errors.append("binary file sha drift")
    if errors:
        raise LauncherError(errors)
    return {"manifest": manifest, "candidate_state": candidate_state}


def build_plan_payload(args: argparse.Namespace) -> dict[str, Any]:
    validated = validate_static_contract(args, require_outputs_absent=True, reject_live=True)
    manifest = validated["manifest"]
    outputs = sorted(manifest["expected_outputs"])
    output_parents = sorted({display_path(repo_path(p).parent) for p in outputs})
    commands = [{"index": idx, "argv": cmd["argv"], "outputs": cmd["outputs"]} for idx, cmd in enumerate(manifest["commands"])]
    gate = list(GATE_COMMAND)
    return {
        "schema": PLAN_SCHEMA,
        "mode": "plan-only",
        "deterministic": True,
        "probe_manifest": {"path": display_path(PROBE_MANIFEST), "sha256": EXPECTED_MANIFEST_SHA256},
        "wrapper": {"path": "scripts/q3_r6_infraretry1_probe_preflight.py", "sha256": EXPECTED_WRAPPER_SHA256},
        "binary": {"path": manifest["binary"]["path"], "sha256": EXPECTED_BINARY_SHA256},
        "candidate_package": {
            "path": manifest["candidate_package"]["path"],
            "manifest_rollup_sha256": EXPECTED_CANDIDATE_ROLLUP_SHA256,
            "actual_rollup_sha256": validated["candidate_state"]["sibling_rollup_sha256"],
        },
        "accepted_hashes": {
            "metrics": EXPECTED_METRICS_SHA256,
            "attestation": EXPECTED_ATTESTATION_SHA256,
            "base_audit": EXPECTED_BASE_AUDIT_SHA256,
            "semantic_diff_audit": EXPECTED_SEMANTIC_DIFF_AUDIT_SHA256,
        },
        "counts": {
            "surfaces": 7,
            "future_probe_spawns": 14,
            "future_gate_spawns": 1,
            "future_outputs": 42,
            "current_probe_spawns": 0,
            "current_gate_spawns": 0,
            "current_retrieval_spawns": 0,
        },
        "commands": commands,
        "future_gate_command": gate,
        "expected_outputs": outputs,
        "output_parents": output_parents,
        "output_parent_hash": sha256_text("\n".join(output_parents) + "\n"),
        "command_argv_sha256": manifest.get("command_argv_sha256"),
        "strict_preflight": {"called_in_process": True, "normal_gate_called": False, "passed": True},
    }


def write_plan(args: argparse.Namespace) -> dict[str, Any]:
    planned = [args.plan_audit, args.plan_receipt, args.stdout_capture, args.stderr_capture]
    if not args.overwrite_plan:
        existing = [display_path(p) for p in planned if repo_path(p).exists()]
        if existing:
            raise LauncherError([f"plan output already exists: {existing}"])
    path_errors: list[str] = []
    for path in planned:
        require_within(path, REPORTS_ROOT, path_errors, "plan artifact")
    if path_errors:
        raise LauncherError(path_errors)
    payload = build_plan_payload(args)
    stdout = "\n".join(
        [
            "Q3-R6-INFRARETRY1 probe launch plan",
            f"future_probe_spawns={payload['counts']['future_probe_spawns']}",
            f"future_gate_spawns={payload['counts']['future_gate_spawns']}",
            f"future_outputs={payload['counts']['future_outputs']}",
            "current_spawns=0",
            "",
        ]
    )
    stderr = ""
    receipt = {
        "schema": RECEIPT_SCHEMA,
        "plan_audit_path": display_path(args.plan_audit),
        "plan_audit_sha256": sha256_text(stable_json(payload)),
        "stdout_capture": {"path": display_path(args.stdout_capture), "sha256": sha256_text(stdout)},
        "stderr_capture": {"path": display_path(args.stderr_capture), "sha256": sha256_text(stderr)},
        "zero_current_spawns": True,
        "normal_gate_called": False,
        "retrieval_subprocesses_spawned": 0,
    }
    write_json_new(args.plan_audit, payload, overwrite=args.overwrite_plan)
    write_json_new(args.plan_receipt, receipt, overwrite=args.overwrite_plan)
    write_text_new(args.stdout_capture, stdout, overwrite=args.overwrite_plan)
    write_text_new(args.stderr_capture, stderr, overwrite=args.overwrite_plan)
    return payload


def reject_execute_outputs(args: argparse.Namespace, manifest: dict[str, Any]) -> None:
    errors: list[str] = []
    for output in manifest["expected_outputs"]:
        if repo_path(output).exists():
            errors.append(f"expected output already exists: {display_path(output)}")
    capture_dir = repo_path(args.execute_capture_dir)
    if capture_dir.exists():
        errors.append(f"execute capture dir already exists: {display_path(capture_dir)}")
    if repo_path(args.output_audit).exists():
        errors.append(f"execute audit already exists: {display_path(args.output_audit)}")
    require_within(args.execute_capture_dir, REPORTS_ROOT, errors, "execute capture dir")
    require_within(args.output_audit, REPORTS_ROOT, errors, "execute audit")
    if errors:
        raise LauncherError(errors)


def validate_plan_for_execute(args: argparse.Namespace) -> dict[str, Any]:
    errors: list[str] = []
    plan_path = repo_path(args.plan_audit)
    if not plan_path.is_file():
        errors.append("reviewed plan audit missing")
    elif sha256_file(plan_path) != args.expected_plan_audit_sha256:
        errors.append("reviewed plan audit sha mismatch")
    if not args.expected_plan_audit_sha256:
        errors.append("execute requires --expected-plan-audit-sha256")
    if errors:
        raise LauncherError(errors)
    plan = read_json(plan_path)
    if plan.get("schema") != PLAN_SCHEMA or plan.get("mode") != "plan-only":
        errors.append("reviewed plan audit schema/mode mismatch")
    if plan.get("counts", {}).get("future_probe_spawns") != 14 or plan.get("counts", {}).get("future_gate_spawns") != 1:
        errors.append("reviewed plan audit future spawn count mismatch")
    if plan.get("counts", {}).get("current_probe_spawns") != 0 or plan.get("counts", {}).get("current_gate_spawns") != 0:
        errors.append("reviewed plan audit current spawn count mismatch")
    validated = validate_static_contract(args, require_outputs_absent=True, reject_live=True)
    manifest = validated["manifest"]
    if [cmd["argv"] for cmd in plan.get("commands", [])] != [cmd["argv"] for cmd in manifest["commands"]]:
        errors.append("reviewed plan argv mismatch against manifest")
    if plan.get("future_gate_command") != GATE_COMMAND:
        errors.append("reviewed plan gate command mismatch")
    reject_execute_outputs(args, manifest)
    if errors:
        raise LauncherError(errors)
    return {"plan": plan, "manifest": manifest, "pre_candidate_state": validated["candidate_state"]}


def unexpected_outputs(manifest: dict[str, Any]) -> list[str]:
    outputs_root = repo_path(manifest["future_paths"]["outputs_root"])
    expected = {repo_path(p).resolve() for p in manifest["expected_outputs"]}
    if not outputs_root.exists():
        return []
    found = [p.resolve() for p in outputs_root.rglob("*") if p.is_file()]
    return [display_path(p) for p in found if p not in expected]


def write_execute_audit(args: argparse.Namespace, audit: dict[str, Any]) -> None:
    out = repo_path(args.output_audit)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        raise FileExistsError(f"refusing to overwrite execute audit: {display_path(out)}")
    out.write_text(stable_json(audit), encoding="utf-8")


def execute(args: argparse.Namespace) -> int:
    start = time.monotonic()
    audit: dict[str, Any] = {
        "schema": EXECUTE_SCHEMA,
        "mode": "execute",
        "plan_audit": {"path": display_path(args.plan_audit), "expected_sha256": args.expected_plan_audit_sha256},
        "spawn_count": 0,
        "retry_count": 0,
        "gate_spawn_count": 0,
        "commands": [],
        "errors": [],
        "passed": False,
    }
    try:
        state = validate_plan_for_execute(args)
        manifest = state["manifest"]
        final_state = validate_static_contract(args, require_outputs_absent=True, reject_live=True)
        audit["pre_spawn_candidate"] = final_state["candidate_state"]
        for output in manifest["expected_outputs"]:
            repo_path(output).parent.mkdir(parents=True, exist_ok=True)
        repo_path(args.execute_capture_dir).mkdir(parents=True, exist_ok=False)
        repo_path(args.output_audit).parent.mkdir(parents=True, exist_ok=True)
        for idx, command in enumerate(manifest["commands"]):
            stdout_path = capture_path(args.execute_capture_dir, idx, "stdout")
            stderr_path = capture_path(args.execute_capture_dir, idx, "stderr")
            before = time.monotonic()
            proc = subprocess.run(command["argv"], cwd=REPO_ROOT, shell=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
            duration = max(0.0, time.monotonic() - before)
            stdout_path.write_bytes(proc.stdout)
            stderr_path.write_bytes(proc.stderr)
            audit["spawn_count"] += 1
            record = {
                "index": idx,
                "argv": command["argv"],
                "exit_status": proc.returncode,
                "duration_seconds": duration,
                "stdout_capture": file_record(stdout_path),
                "stderr_capture": file_record(stderr_path),
                "outputs": {k: file_record(v) if repo_path(v).is_file() else {"path": display_path(v), "exists": False} for k, v in command["outputs"].items()},
            }
            audit["commands"].append(record)
            missing = [path for path, rec in record["outputs"].items() if rec.get("exists") is False]
            extras = unexpected_outputs(manifest)
            if proc.returncode != 0:
                audit["errors"].append(f"command {idx} returned nonzero {proc.returncode}")
                break
            if missing:
                audit["errors"].append(f"command {idx} missing outputs: {missing}")
                break
            if extras:
                audit["errors"].append(f"unexpected outputs present: {extras}")
                break
        if not audit["errors"] and audit["spawn_count"] == 14:
            post_candidate = sibling_rollup(manifest["candidate_package"]["path"])
            audit["post_probe_candidate"] = post_candidate
            if post_candidate["sibling_rollup_sha256"] != EXPECTED_CANDIDATE_ROLLUP_SHA256:
                audit["errors"].append("candidate rollup mutated during probes")
            try:
                validate_static_contract(args, require_outputs_absent=False, reject_live=True)
            except LauncherError as exc:
                audit["errors"].extend([f"post-probe immutable validation failed: {err}" for err in exc.errors])
            missing_all = [display_path(p) for p in manifest["expected_outputs"] if not repo_path(p).is_file()]
            if missing_all:
                audit["errors"].append(f"missing expected outputs after probes: {missing_all}")
        if not audit["errors"] and audit["spawn_count"] == 14:
            gate_stdout = repo_path(args.execute_capture_dir) / "gate.stdout.txt"
            gate_stderr = repo_path(args.execute_capture_dir) / "gate.stderr.txt"
            proc = subprocess.run(GATE_COMMAND, cwd=REPO_ROOT, shell=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
            gate_stdout.write_bytes(proc.stdout)
            gate_stderr.write_bytes(proc.stderr)
            audit["gate_spawn_count"] = 1
            gate_record = {
                "argv": GATE_COMMAND,
                "exit_status": proc.returncode,
                "stdout_capture": file_record(gate_stdout),
                "stderr_capture": file_record(gate_stderr),
            }
            try:
                summary = json.loads(proc.stdout.decode("utf-8"))
            except Exception as exc:
                summary = None
                audit["errors"].append(f"gate summary parse failed: {exc}")
            gate_record["summary"] = summary
            audit["gate"] = gate_record
            if proc.returncode != 0:
                audit["errors"].append(f"gate returned nonzero {proc.returncode}")
            if not isinstance(summary, dict) or summary.get("passed") is not True or summary.get("command_count") != 14 or summary.get("output_count") != 42:
                audit["errors"].append("gate summary did not pass expected 14/42 contract")
        audit["passed"] = not audit["errors"] and audit["spawn_count"] == 14 and audit["gate_spawn_count"] == 1
        return 0 if audit["passed"] else 1
    except LauncherError as exc:
        audit["errors"].extend(exc.errors)
        return 1
    finally:
        audit["duration_seconds"] = max(0.0, time.monotonic() - start)
        try:
            write_execute_audit(args, audit)
        except Exception as exc:
            print(f"failed to write execute audit: {exc}", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.plan_only:
            plan = write_plan(args)
            print(json.dumps({"ok": True, "plan_audit": display_path(args.plan_audit), "sha256": sha256_file(args.plan_audit), "future_probe_spawns": plan["counts"]["future_probe_spawns"], "future_gate_spawns": plan["counts"]["future_gate_spawns"]}, sort_keys=True))
            return 0
        return execute(args)
    except LauncherError as exc:
        print(f"LauncherError: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"LauncherError: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
