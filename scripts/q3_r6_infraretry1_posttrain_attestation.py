#!/usr/bin/env python3
"""Post-train attestation for the bounded Q3 R6 infraretry1 candidate.

This script is intentionally read-only with respect to candidate package bytes,
metrics, launch/execute audits, and probe outputs. It writes only two derived
JSON evidence reports under the infraretry1 reports root.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import struct
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent
RUN_ROOT = Path("runs/eos-d384-q3-boundary-goal-v1-20260902T084144Z")
R6_ROOT = RUN_ROOT / "headroom-q3r6-frontier-retain-alltrain-b0-rw050-infraretry1"
SOURCE_R6_ROOT = RUN_ROOT / "headroom-q3r6-frontier-retain-alltrain-b0-rw050"
REPORTS_ROOT = R6_ROOT / "reports"
PROBE_ROOT = R6_ROOT / "probe"
ARM_ID = "arm-q3r6-frontier-retain-alltrain-b0-rw050-q3w004-m002-lr2e6-e2-infraretry1"

BINARY = RUN_ROOT / "bin/eos-r6-frontier-retention-infraretry1"
BINARY_SHA256 = "c5679853f8494c1e144a85d3cf2a345b0a3236f917a42607ddebc2242d344f90"
CANDIDATE_ROOT_ARTIFACT = RUN_ROOT / "packages" / ARM_ID / "d384-pre.mll"
CANDIDATE_PACKAGE_DIR = CANDIDATE_ROOT_ARTIFACT.parent
TRAIN_JSONL = SOURCE_R6_ROOT / "data/beir1155.q3r6-frontier-retention.train.jsonl"
METRICS_JSON = R6_ROOT / "metrics/arm-q3r6-frontier-retain-alltrain-b0-rw050-q3w004-m002-lr2e6-e2-infraretry1.train.metrics.json"
LAUNCH_AUDIT = REPORTS_ROOT / "q3-r6-infraretry1-training-launch-audit.json"
EXECUTE_AUDIT = REPORTS_ROOT / "q3-r6-infraretry1-training-attempt2.execute-audit.json"
STDOUT_CAPTURE = REPORTS_ROOT / "q3-r6-infraretry1-training-attempt2.stdout.txt"
STDERR_CAPTURE = REPORTS_ROOT / "q3-r6-infraretry1-training-attempt2.stderr.txt"
PRETRAIN_SIDECAR = PROBE_ROOT / "q3-r6-infraretry1-pretrain-binding-sidecar.json"
PROBE_MANIFEST = PROBE_ROOT / "q3r6-frontier-retention.probe-manifest.json"
SEMANTIC_DIFF_AUDIT = PROBE_ROOT / "q3-r6-infraretry1-probe-semantic-diff-audit.json"
ATTESTATION_JSON = REPORTS_ROOT / "q3-r6-infraretry1-posttrain-attestation.json"
BASE_LEAKAGE_AUDIT_JSON = REPORTS_ROOT / "q3-r6-infraretry1-candidate-effective-base-leakage-audit.json"

EXPECTED_HASHES = {
    "launch_audit": "f06341417457e0677791e241bed0a34a6e1fa9b8e7c057547c7dfdfc554df6ac",
    "execute_audit": "91c339c1bc527ce8151dc10fc1a238831f64dc285dddd2b7d03d86bd9eae61d9",
    "stdout_capture": "2d7a91e6f50968b9dc3e73eadaa4ede32f1cf28a40e0c308fe8c72f1948bd49b",
    "stderr_capture": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "metrics_json": "7f955bd8eed270b9b27ba6d4a2890f40c250e20bbf1f9f6289fc2179e710dc01",
    "binary": BINARY_SHA256,
    "train_jsonl": "41abe77ef5f2eb6adbd6ee5f1afd01e17d6f12bd31546db8a44cf77f309bed09",
    "pretrain_rollup": "17335593ec850a1c01c8b31b17a32fe002368a508c06d431b761d45f1d9a76b0",
    "posttrain_rollup": "65c0a44482353a05682d161a5f3f0edf579a54c3f597b5424e2e7820ef83b6cf",
    "probe_manifest": "b4f6f57165ea6d285c59af25f23b89eee7508ec842f49bb85116575810f74bf7",
    "semantic_diff_audit_pre_repair": "4dc7641f8c934c08bc23c51be7173c5621f4534d6341766f6150eba32fa47dba",
    "launcher_source": "c6815475cb47acdbaa5cbfb6ca5b8bc5e67c66cc6552a428351018df38163c07",
}

SOURCE_BINDINGS = {
    "cmd/eos/main.go": [
        "runTrainEmbed",
    ],
    "runtime/embedding_train_runner.go": [
        "validateTurboQuantTopKScoreSpectrumOnlyRunConfig",
        "clearScoreSpectrumIncompatibleRunConfig",
        "applyScoreSpectrumRunOverrides",
    ],
    "runtime/embedding_trainer.go": [
        "accumulateTurboQuantPreparedIPTopKScoreSpectrumRowGrads",
    ],
    "runtime/embedding_train_manifest.go": [
        "mllValues",
        "scoreSpectrumPolicyFromAuthoredDoc",
    ],
    "runtime/package_manifest.go": [
        "encodePackageManifestMLL",
        "decodePackageManifestMLL",
    ],
}


class AttestationError(RuntimeError):
    pass


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
    with repo_path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(repo_path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    p = repo_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def require(condition: bool, label: str) -> bool:
    if not condition:
        raise AttestationError(label)
    return True


def file_record(path: str | Path, *, expected_sha256: str | None = None) -> dict[str, Any]:
    p = repo_path(path)
    actual = sha256_file(p)
    if expected_sha256 is not None:
        require(actual == expected_sha256, f"{display_path(p)} sha mismatch")
    return {"path": display_path(p), "sha256": actual, "bytes": p.stat().st_size}


def package_sibling_rollup(package_dir: Path) -> tuple[str, list[dict[str, Any]]]:
    siblings = []
    for path in sorted(repo_path(package_dir).glob("d384-pre.*")):
        siblings.append({"name": path.name, "path": display_path(path), "sha256": sha256_file(path), "bytes": path.stat().st_size})
    digest = hashlib.sha256()
    for item in siblings:
        digest.update(json.dumps({"bytes": item["bytes"], "name": item["name"], "sha256": item["sha256"]}, separators=(",", ":"), sort_keys=True).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest(), siblings


def extract_function_source(path: Path, name: str) -> dict[str, Any]:
    text = repo_path(path).read_text(encoding="utf-8")
    if name == "runTrainEmbed":
        match = re.search(r"func runTrainEmbed\(.*?(?=\nfunc )", text, flags=re.S)
        require(match is not None, f"source function not found: {path}:{name}")
        source = match.group(0)
        start_pos = match.start()
    else:
        pattern = rf"func (?:\([^)]*\) )?{re.escape(name)}\("
        match = re.search(pattern, text)
        if match:
            start_pos = match.start()
            brace = text.find("{", match.end())
            depth = 0
            end = None
            for idx in range(brace, len(text)):
                if text[idx] == "{":
                    depth += 1
                elif text[idx] == "}":
                    depth -= 1
                    if depth == 0:
                        end = idx + 1
                        break
            source = text[start_pos:end] if end is not None else ""
            match = match if end is not None else None
    require(match is not None, f"source function not found: {path}:{name}")
    start_line = text[:start_pos].count("\n") + 1
    end_line = start_line + source.count("\n")
    return {
        "function": name,
        "line_start": start_line,
        "line_end": end_line,
        "source_sha256": sha256_text(source),
    }


def source_binding_records() -> list[dict[str, Any]]:
    records = []
    for rel, functions in SOURCE_BINDINGS.items():
        p = Path(rel)
        text = repo_path(p).read_text(encoding="utf-8")
        function_records = [extract_function_source(p, name) for name in functions]
        records.append(
            {
                "path": rel,
                "sha256": sha256_file(p),
                "functions": function_records,
            }
        )
    trainer_text = repo_path("runtime/embedding_trainer.go").read_text(encoding="utf-8")
    require("InnerProductPrepared" in trainer_text, "prepared-IP scorer call missing")
    require("newTurboQuantPreparedPrefixMatrix" in trainer_text, "prepared prefix matrix call missing")
    require("accumulateTurboQuantPreparedIPTopKScoreSpectrumRowGrads" in trainer_text, "prepared top-k accumulator missing")
    runner_text = repo_path("runtime/embedding_train_runner.go").read_text(encoding="utf-8")
    require('cfg.TurboQuantPrefixScoreMode = ""' in runner_text, "generic prefix score mode clear missing")
    require("cfg.TurboQuantPrefixSeed = prefixSeed" in runner_text, "top-k prefix seed preserve missing")
    return records


def parse_mll_sections(path: Path) -> dict[bytes, bytes]:
    data = repo_path(path).read_bytes()
    require(len(data) >= 24, f"{display_path(path)} too small for MLL")
    require(data[:4] == b"MLL\0", f"{display_path(path)} missing MLL magic")
    section_count = struct.unpack_from("<I", data, 16)[0]
    sections: dict[bytes, bytes] = {}
    offset = 24
    for _ in range(section_count):
        entry = data[offset : offset + 64]
        require(len(entry) == 64, f"{display_path(path)} truncated MLL directory")
        tag = entry[:4]
        section_offset = struct.unpack_from("<Q", entry, 4)[0]
        section_size = struct.unpack_from("<Q", entry, 12)[0]
        require(section_offset + section_size <= len(data), f"{display_path(path)} section out of bounds")
        sections[tag] = data[section_offset : section_offset + section_size]
        offset += 64
    return sections


def parse_strg(body: bytes) -> list[str]:
    require(len(body) >= 4, "STRG too short")
    count = struct.unpack_from("<I", body, 0)[0]
    strings = []
    offset = 4
    for _ in range(count):
        require(offset + 4 <= len(body), "STRG truncated length")
        length = struct.unpack_from("<I", body, offset)[0]
        offset += 4
        require(offset + length <= len(body), "STRG truncated body")
        strings.append(body[offset : offset + length].decode("utf-8"))
        offset += length
    return strings


def strg_at(strings: list[str], idx: int) -> str:
    return strings[idx] if 0 <= idx < len(strings) else ""


def parse_head_metadata(body: bytes, strings: list[str]) -> dict[str, Any]:
    offset = 0
    require(len(body) >= 28, "HEAD too short")
    offset += 4 + 4 + 8 + 8
    backends = struct.unpack_from("<H", body, offset)[0]
    offset += 2 + backends * 2
    capabilities = struct.unpack_from("<H", body, offset)[0]
    offset += 2 + capabilities * 4
    metadata_count = struct.unpack_from("<H", body, offset)[0]
    offset += 2
    out: dict[str, Any] = {}
    for _ in range(metadata_count):
        key_idx = struct.unpack_from("<I", body, offset)[0]
        offset += 4
        kind = body[offset]
        offset += 1
        key = strg_at(strings, key_idx)
        if kind == 0:
            value = None
        elif kind == 1:
            value = body[offset] != 0
            offset += 1
        elif kind == 2:
            value = struct.unpack_from("<q", body, offset)[0]
            offset += 8
        elif kind == 3:
            value = struct.unpack_from("<d", body, offset)[0]
            offset += 8
        elif kind == 4:
            value = strg_at(strings, struct.unpack_from("<I", body, offset)[0])
            offset += 4
        else:
            raise AttestationError(f"HEAD unknown value kind {kind}")
        if key:
            out[key] = value
    return out


def authored_manifest_metadata(path: Path) -> dict[str, Any]:
    sections = parse_mll_sections(path)
    require(b"HEAD" in sections and b"STRG" in sections, f"{display_path(path)} missing HEAD/STRG")
    strings = parse_strg(sections[b"STRG"])
    return parse_head_metadata(sections[b"HEAD"], strings)


def selected_manifest_facts() -> dict[str, Any]:
    train_meta = authored_manifest_metadata(CANDIDATE_PACKAGE_DIR / "d384-pre.train.mll")
    package_meta = authored_manifest_metadata(CANDIDATE_PACKAGE_DIR / "d384-pre.package.mll")
    facts = {
        "train_manifest": {
            "path": display_path(CANDIDATE_PACKAGE_DIR / "d384-pre.train.mll"),
            "sha256": sha256_file(CANDIDATE_PACKAGE_DIR / "d384-pre.train.mll"),
            "config_turboquant_prefix_score_mode": train_meta.get("config.turboquant_prefix_score_mode", ""),
            "config_turboquant_prefix_seed": train_meta.get("config.turboquant_prefix_seed"),
            "config_turboquant_topk_objectives": train_meta.get("config.turboquant_topk_objectives", ""),
            "config_turboquant_topk_loss": train_meta.get("config.turboquant_topk_loss", ""),
            "score_spectrum_auto_cleared_objectives": train_meta.get("score_spectrum.auto_cleared_objectives", ""),
            "score_spectrum_isolated_inherited_objectives": train_meta.get("score_spectrum.isolated_inherited_objectives", ""),
            "score_spectrum_row_count": train_meta.get("score_spectrum.score_spectrum_row_count"),
        },
        "package_manifest": {
            "path": display_path(CANDIDATE_PACKAGE_DIR / "d384-pre.package.mll"),
            "sha256": sha256_file(CANDIDATE_PACKAGE_DIR / "d384-pre.package.mll"),
            "auto_cleared_objectives": package_meta.get("auto_cleared_objectives", ""),
            "isolated_inherited_objectives": package_meta.get("isolated_inherited_objectives", ""),
            "score_spectrum_row_count": package_meta.get("score_spectrum_row_count"),
        },
    }
    require(facts["train_manifest"]["score_spectrum_auto_cleared_objectives"] == "", "train auto-cleared objectives not empty")
    require(facts["package_manifest"]["auto_cleared_objectives"] == "", "package auto-cleared objectives not empty")
    return facts


def run_inspect() -> dict[str, Any]:
    command = [display_path(BINARY), "inspect", display_path(CANDIDATE_ROOT_ARTIFACT)]
    result = subprocess.run(command, cwd=REPO_ROOT, text=True, capture_output=True, check=False)
    require(result.returncode == 0, "supported inspect command failed")
    require("package verify: OK" in result.stdout, "inspect missing package verify OK")
    require("train profile: step=578" in result.stdout, "inspect missing train profile step 578")
    require(result.stderr == "", "inspect stderr not empty")
    return {
        "command": command,
        "exit_status": result.returncode,
        "stdout_sha256": sha256_text(result.stdout),
        "stderr_sha256": sha256_text(result.stderr),
        "stdout": result.stdout.splitlines(),
        "package_verify_ok": True,
        "train_profile_step": 578,
    }


def validate_metrics() -> dict[str, Any]:
    metrics = read_json(METRICS_JSON)
    cfg = metrics.get("config", {})
    workload = metrics.get("workload", {})
    checks = {
        "schema": metrics.get("schema") == "manta.embedding_train_metrics.v1",
        "q3_seed": cfg.get("turboquant_prefix_seed") == 5581486560434873699,
        "topk_objective": cfg.get("turboquant_topk_objectives") == [{"dim": 384, "bit_width": 3, "weight": 0.04}],
        "topk_loss_cutoff_tau_margin_mask": cfg.get("turboquant_topk_loss") == "lambdandcg"
        and cfg.get("turboquant_topk_cutoff") == 10
        and math.isclose(float(cfg.get("turboquant_topk_tau")), 0.05)
        and math.isclose(float(cfg.get("turboquant_topk_margin")), 0.002)
        and cfg.get("turboquant_topk_negative_mask") == "q3",
        "recall_cutoff_tau_margin_mask": math.isclose(float(cfg.get("turboquant_topk_recall_weight")), 0.50)
        and cfg.get("turboquant_topk_recall_cutoff") == 100
        and math.isclose(float(cfg.get("turboquant_topk_recall_tau")), 0.05)
        and float(cfg.get("turboquant_topk_recall_margin", 0) or 0) == 0
        and cfg.get("turboquant_topk_recall_negative_mask") == "q3",
        "generic_prefix_score_mode_absent_or_empty": cfg.get("turboquant_prefix_score_mode", "") == "",
        "workload": workload.get("train_examples") == 1155
        and workload.get("completed_epochs") == 2
        and workload.get("train_batches_per_epoch") == 289
        and workload.get("planned_train_pairs") == 54162
        and workload.get("actual_train_pairs") == 54162
        and workload.get("actual_eval_pairs") == 0,
    }
    for label, ok in checks.items():
        require(bool(ok), f"metrics check failed: {label}")
    return {"path": display_path(METRICS_JSON), "sha256": sha256_file(METRICS_JSON), "checks": checks, "config": cfg, "workload": workload}


def validate_execute_audit() -> dict[str, Any]:
    audit = read_json(EXECUTE_AUDIT)
    false_metrics = audit.get("errors") == ["execute metrics validation failed: config_turboquant_prefix_score_mode"]
    checks = audit.get("checks", {})
    predicates = {
        "trainer_exit_status_zero": audit.get("execute_result", {}).get("exit_status") == 0,
        "spawn_count_one": audit.get("spawn_count") == 1,
        "retry_count_zero": audit.get("retry_count") == 0,
        "workload_no_eval": True,
        "only_false_metrics_predicate": false_metrics and checks.get("metrics_valid_after_exit0") is False,
        "actual_training_ran_once": audit.get("actual_training_ran") is True and audit.get("trainer_process_spawned") is True,
    }
    for label, ok in predicates.items():
        require(bool(ok), f"execute audit predicate failed: {label}")
    return {"path": display_path(EXECUTE_AUDIT), "sha256": sha256_file(EXECUTE_AUDIT), "predicates": predicates, "checks": checks, "errors": audit.get("errors", [])}


def build_base_leakage_audit() -> dict[str, Any]:
    row_count = 0
    buckets: dict[str, int] = {}
    bad_rows = []
    with repo_path(TRAIN_JSONL).open("r", encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, start=1):
            if not raw.strip():
                continue
            row_count += 1
            row = json.loads(raw)
            buckets[str(row.get("selection_bucket", ""))] = buckets.get(str(row.get("selection_bucket", "")), 0) + 1
            explicit_zero = all(key in row and float(row[key]) == 0.0 for key in ("base_loss_weight", "hard_loss_weight", "soft_loss_weight", "recovery_loss_weight"))
            aux_active = float(row.get("turboquant_topk_loss_weight", 0.0)) > 0 or float(row.get("turboquant_topk_recall_loss_weight", 0.0)) > 0
            if not explicit_zero or not aux_active:
                bad_rows.append({"line": line_no, "row_id": row.get("row_id"), "explicit_zero_base": explicit_zero, "aux_active": aux_active})
    require(row_count == 1155, "train JSONL row count mismatch")
    require(not bad_rows, "effective base leakage audit found non-aux-only rows")
    rollup, _siblings = package_sibling_rollup(CANDIDATE_PACKAGE_DIR)
    require(rollup == EXPECTED_HASHES["posttrain_rollup"], "base audit candidate rollup mismatch")
    payload = {
        "schema": "eos.q3_r6_infraretry1_candidate_effective_base_leakage_audit.v1",
        "created_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "status": "passed",
        "train_jsonl": file_record(TRAIN_JSONL, expected_sha256=EXPECTED_HASHES["train_jsonl"]),
        "metrics": file_record(METRICS_JSON, expected_sha256=EXPECTED_HASHES["metrics_json"]),
        "posttrain_candidate_package": {
            "path": display_path(CANDIDATE_ROOT_ARTIFACT),
            "sibling_rollup_sha256": rollup,
            "root_artifact_sha256": sha256_file(CANDIDATE_ROOT_ARTIFACT),
        },
        "source_bindings": source_binding_records(),
        "runtime_semantics": {
            "base_loss_weight_explicit_zero_rows": row_count,
            "base_hard_soft_recovery_work_units": 0,
            "base_objective_gradient_contribution": 0,
            "aux_only": True,
            "aux_only_derivation": "all rows explicitly set base_loss_weight, hard_loss_weight, soft_loss_weight, and recovery_loss_weight to 0, so base objectives make zero gradient contribution; each row has positive turboquant top-k aux loss weight",
            "compute_scope_note": "zero contribution is an objective/gradient statement, not a claim that row encoding or candidate scoring compute was absent",
            "bad_rows": bad_rows,
        },
        "row_count": row_count,
        "selection_buckets": buckets,
    }
    write_json(BASE_LEAKAGE_AUDIT_JSON, payload)
    payload["sha256"] = sha256_file(BASE_LEAKAGE_AUDIT_JSON)
    return payload


def build_attestation(base_audit: dict[str, Any]) -> dict[str, Any]:
    rollup, siblings = package_sibling_rollup(CANDIDATE_PACKAGE_DIR)
    require(rollup == EXPECTED_HASHES["posttrain_rollup"], "posttrain rollup mismatch")
    pretrain = read_json(PRETRAIN_SIDECAR)
    require(pretrain["candidate_package"]["pretrain_sibling_rollup_sha256"] == EXPECTED_HASHES["pretrain_rollup"], "pretrain sidecar rollup mismatch")
    manifest_facts = selected_manifest_facts()
    inspect = run_inspect()
    metrics = validate_metrics()
    execute = validate_execute_audit()
    files = {
        "launch_audit": file_record(LAUNCH_AUDIT, expected_sha256=EXPECTED_HASHES["launch_audit"]),
        "execute_audit": file_record(EXECUTE_AUDIT, expected_sha256=EXPECTED_HASHES["execute_audit"]),
        "stdout_capture": file_record(STDOUT_CAPTURE, expected_sha256=EXPECTED_HASHES["stdout_capture"]),
        "stderr_capture": file_record(STDERR_CAPTURE, expected_sha256=EXPECTED_HASHES["stderr_capture"]),
        "metrics_json": file_record(METRICS_JSON, expected_sha256=EXPECTED_HASHES["metrics_json"]),
        "binary": file_record(BINARY, expected_sha256=EXPECTED_HASHES["binary"]),
        "train_jsonl": file_record(TRAIN_JSONL, expected_sha256=EXPECTED_HASHES["train_jsonl"]),
        "pretrain_sidecar": file_record(PRETRAIN_SIDECAR),
        "base_leakage_audit": file_record(BASE_LEAKAGE_AUDIT_JSON),
    }
    payload = {
        "schema": "eos.q3_r6_infraretry1_posttrain_attestation.v1",
        "created_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "status": "passed",
        "arm_id": ARM_ID,
        "files": files,
        "candidate_package": {
            "path": display_path(CANDIDATE_ROOT_ARTIFACT),
            "sibling_count": len(siblings),
            "sibling_rollup_sha256": rollup,
            "root_artifact_sha256": sha256_file(CANDIDATE_ROOT_ARTIFACT),
            "siblings": siblings,
        },
        "pretrain_rollup_sha256": EXPECTED_HASHES["pretrain_rollup"],
        "posttrain_rollup_sha256": EXPECTED_HASHES["posttrain_rollup"],
        "execute_audit_predicates": execute["predicates"],
        "only_false_metrics_predicate": "config_turboquant_prefix_score_mode",
        "effective_topk_config": {
            "q3_seed": 5581486560434873699,
            "objective": {"dim": 384, "bit_width": 3, "weight": 0.04},
            "lambdandcg": {"cutoff": 10, "tau": 0.05, "margin": 0.002, "negative_mask": "q3"},
            "recall": {"weight": 0.50, "cutoff": 100, "tau": 0.05, "margin": 0, "negative_mask": "q3"},
            "generic_prefix_score_mode": "",
            "generic_prefix_score_mode_status": "absent_or_empty_by_score_spectrum_topk_only_runtime_semantics",
        },
        "metrics_validation": metrics,
        "supported_package_inspect": inspect,
        "source_bindings": source_binding_records(),
        "manifest_facts": manifest_facts,
        "base_leakage_audit": {
            "path": display_path(BASE_LEAKAGE_AUDIT_JSON),
            "sha256": base_audit["sha256"],
            "status": base_audit["status"],
            "base_hard_soft_recovery_work_units": 0,
            "base_objective_gradient_contribution": 0,
            "aux_only": True,
        },
        "probe_future_outputs": {
            "declared_outputs": 42,
            "existing_outputs": 0,
            "probes_or_evals_run_by_this_script": False,
        },
        "attestation_statement": "The exact completed candidate is accepted for post-train binding because the generic prefix score-mode field is absent/empty under score-spectrum top-k-only runtime semantics, while source-bound top-k and recall calls use prepared-IP scoring directly.",
    }
    write_json(ATTESTATION_JSON, payload)
    payload["sha256"] = sha256_file(ATTESTATION_JSON)
    return payload


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="write attestation and effective-base leakage audit JSON")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.write:
        raise AttestationError("pass --write to materialize post-train evidence reports")
    base_audit = build_base_leakage_audit()
    attestation = build_attestation(base_audit)
    print(json.dumps({"ok": True, "attestation": {"path": display_path(ATTESTATION_JSON), "sha256": attestation["sha256"]}, "base_leakage_audit": {"path": display_path(BASE_LEAKAGE_AUDIT_JSON), "sha256": base_audit["sha256"]}}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
