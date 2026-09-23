#!/usr/bin/env python3
"""Materialize deterministic AOQT V7 fold-train calibration artifacts.

This wrapper filters an existing AOQT Stage 2 materialized calibration set to
exactly one validated V7 fold's train row IDs.  It does not train, evaluate,
export vectors, read official metrics, or mutate the source artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import OrderedDict
from decimal import Decimal
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import build_aoqt_stage2_calibration as stage2  # noqa: E402
import build_aoqt_v7_dev_split as splitter  # noqa: E402


ROW_ID_RECEIPT_SCHEMA = "eos.aoqt.v7_fold_train_row_ids.v1"
DEFAULT_SPLIT_MANIFEST = Path(".tiller/scratch/codex/aoqt-v7-dev-split/aoqt-v7-dev-split.raw-v4.json")
DEFAULT_SPLIT_VALIDATION = Path(".tiller/scratch/codex/aoqt-v7-dev-split/aoqt-v7-dev-split.raw-v4.validation.json")
DEFAULT_SOURCE_PLAN = Path("runs/eos-d384-aoqt-trainonly-v1-20260903T092452Z/aoqt/dataset-scoped/plan.pairable.raw.v4.json")
DEFAULT_SOURCE_DIR = Path(
    "runs/eos-d384-aoqt-trainonly-v1-20260903T092452Z/aoqt/dataset-scoped/"
    "materialized-raw-v4-score-distill-joint-budget-v1"
)


class MaterializeError(ValueError):
    """Raised when fold materialization cannot be trusted."""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-manifest", type=Path, default=DEFAULT_SPLIT_MANIFEST)
    parser.add_argument("--split-validation", type=Path, default=DEFAULT_SPLIT_VALIDATION)
    parser.add_argument("--fold-id", required=True)
    parser.add_argument("--source-plan", type=Path, default=DEFAULT_SOURCE_PLAN)
    parser.add_argument("--source-materialized-manifest", type=Path, default=DEFAULT_SOURCE_DIR / "manifest.json")
    parser.add_argument("--source-materialized-rows", type=Path, default=DEFAULT_SOURCE_DIR / "rows.jsonl")
    parser.add_argument("--source-materialized-preflight", type=Path, default=DEFAULT_SOURCE_DIR / "preflight.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-split-manifest-sha256", required=True)
    parser.add_argument("--expected-split-validation-sha256", required=True)
    parser.add_argument("--expected-source-plan-sha256", required=True)
    parser.add_argument("--expected-source-materialized-manifest-sha256", required=True)
    parser.add_argument("--expected-source-materialized-rows-sha256", required=True)
    parser.add_argument("--expected-source-materialized-preflight-sha256", required=True)
    return parser.parse_args(argv)


def repo_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def display_path(path: Path) -> str:
    resolved = repo_path(path).resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        return str(resolved)


def require_sha256(value: str, label: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise MaterializeError(f"{label} must be lowercase sha256")
    return value


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with repo_path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_bytes(payload: Any) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha256_json(payload: Any) -> str:
    return hashlib.sha256(canonical_bytes(payload)).hexdigest()


def aoqt_row_id_sha256(row_ids: list[str]) -> str:
    h = hashlib.sha256()
    for row_id in sorted(row_ids):
        h.update(row_id.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(repo_path(path).read_text(encoding="utf-8"))
    except OSError as exc:
        raise MaterializeError(f"{display_path(path)}: cannot read JSON") from exc
    except json.JSONDecodeError as exc:
        raise MaterializeError(f"{display_path(path)}: invalid JSON") from exc
    if not isinstance(payload, dict):
        raise MaterializeError(f"{display_path(path)}: expected JSON object")
    return payload


def write_json_atomic(path: Path, payload: dict[str, Any], *, compact: bool = False) -> None:
    target = repo_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".tmp")
    if compact:
        raw = json.dumps(payload, separators=(",", ":")) + "\n"
    else:
        raw = json.dumps(payload, indent=2, sort_keys=False) + "\n"
    tmp.write_text(raw, encoding="utf-8")
    tmp.replace(target)


def check_file_hash(path: Path, expected: str, label: str) -> str:
    require_sha256(expected, f"expected {label}")
    observed = sha256_file(path)
    if observed != expected:
        raise MaterializeError(f"{label} sha256 mismatch: observed {observed}, expected {expected}")
    return observed


def find_fold(split_manifest: dict[str, Any], fold_id: str) -> dict[str, Any]:
    folds = split_manifest.get("folds")
    if not isinstance(folds, list):
        raise MaterializeError("split manifest folds must be a list")
    matches = [fold for fold in folds if isinstance(fold, dict) and fold.get("name") == fold_id]
    if len(matches) != 1:
        raise MaterializeError(f"split manifest must contain exactly one fold named {fold_id!r}")
    return matches[0]


def require_string_list(value: Any, label: str) -> list[str]:
    if not isinstance(value, list) or any(not isinstance(item, str) or not item for item in value):
        raise MaterializeError(f"{label} must be a non-empty string list")
    if len(value) != len(set(value)):
        raise MaterializeError(f"{label} contains duplicate row IDs")
    return list(value)


def validate_split_inputs(split_path: Path, receipt_path: Path, fold_id: str) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], list[str]]:
    split_manifest = load_json(split_path)
    receipt = load_json(receipt_path)
    if receipt.get("schema") != splitter.VALIDATION_SCHEMA:
        raise MaterializeError(f"{display_path(receipt_path)}: expected {splitter.VALIDATION_SCHEMA}")
    if receipt.get("passed") is not True:
        raise MaterializeError("split validation receipt did not pass")

    rebuilt = splitter.validate_manifest(split_path)
    if rebuilt.get("passed") is not True:
        raise MaterializeError("split manifest deterministic validation did not pass")

    split_file_sha = sha256_file(split_path)
    if receipt.get("manifest_file_sha256") != split_file_sha:
        raise MaterializeError("split validation receipt manifest_file_sha256 is stale")
    expected_payload_sha = split_manifest.get("provenance", {}).get("manifest_sha256")
    if receipt.get("manifest_payload_sha256") != expected_payload_sha:
        raise MaterializeError("split validation receipt manifest_payload_sha256 is stale")
    if receipt.get("manifest_payload_sha256") != rebuilt.get("manifest_payload_sha256"):
        raise MaterializeError("split validation receipt does not match deterministic rebuild")

    leakage = split_manifest.get("leakage_proof")
    if not isinstance(leakage, dict) or leakage.get("passed") is not True or leakage.get("official_overlap_count") != 0:
        raise MaterializeError("split leakage proof failed or includes official overlap")

    fold = find_fold(split_manifest, fold_id)
    train = fold.get("train")
    if not isinstance(train, dict):
        raise MaterializeError(f"{fold_id}: train side is required")
    row_ids = require_string_list(train.get("row_ids"), f"{fold_id}.train.row_ids")
    if row_ids != sorted(row_ids):
        raise MaterializeError(f"{fold_id}.train.row_ids must be sorted")
    if train.get("row_count") != len(row_ids):
        raise MaterializeError(f"{fold_id}.train.row_count does not match row_ids")
    if train.get("row_ids_sha256") != sha256_json(row_ids):
        raise MaterializeError(f"{fold_id}.train.row_ids_sha256 does not bind row_ids")
    proof = fold.get("proof")
    if not isinstance(proof, dict) or proof.get("train_dev_group_overlap_count") != 0:
        raise MaterializeError(f"{fold_id}: train/dev group overlap proof failed")
    return split_manifest, receipt, fold, row_ids


def manifest_for_go_hash(manifest: dict[str, Any]) -> OrderedDict[str, Any]:
    def ordered(keys: tuple[str, ...], source: dict[str, Any]) -> OrderedDict[str, Any]:
        return OrderedDict((key, source[key]) for key in keys if key in source)

    def ordered_map(source: dict[str, Any]) -> OrderedDict[str, Any]:
        return OrderedDict((key, source[key]) for key in sorted(source))

    def ordered_objective(source: dict[str, Any]) -> OrderedDict[str, Any]:
        out = ordered(
            (
                "dim",
                "turboquant_seed",
                "gain_bit",
                "q3_guard_bit",
                "q5_guard_bit",
                "score_surface",
                "gain_cutoff",
                "gain_tau",
                "gain_margin",
                "guard_tau",
                "guard_margin",
                "score_distill_tau",
                "nf_boundary_source",
                "weight_sums",
            ),
            source,
        )
        if isinstance(out.get("weight_sums"), dict):
            out["weight_sums"] = ordered(
                (
                    "q3_gain",
                    "q3_order_guard",
                    "q3_score_distill",
                    "q5_order_guard",
                    "q5_score_distill",
                    "nf_boundary_guard",
                ),
                out["weight_sums"],
            )
        return out

    def ordered_candidate_policy(source: dict[str, Any]) -> OrderedDict[str, Any]:
        return ordered(
            (
                "dense_max_abs_delta_tolerance",
                "angle_max_abs_cap",
                "require_objective_activation",
                "q3_gain_allowed_loss_increase",
                "q3_order_guard_allowed_loss_increase",
                "q3_score_distill_allowed_loss_increase",
                "q5_order_guard_allowed_loss_increase",
                "q5_score_distill_allowed_loss_increase",
                "nf_boundary_guard_allowed_loss_increase",
            ),
            source,
        )

    out: OrderedDict[str, Any] = OrderedDict()
    for key in (
        "schema",
        "created_at_utc",
        "anchor_artifact_path",
        "anchor_artifact_sha256",
        "anchor_package_manifest_sha256",
        "anchor_embedding_space_id",
        "dim",
        "topology",
        "turboquant_seed",
        "quant_surfaces",
        "objective_contract",
        "qrels_sha256_by_dataset",
        "split_proof",
        "source_artifact_hashes",
        "vector_cache_hashes",
        "row_count",
        "row_id_sha256",
        "compatibility_digest",
        "legal_gates",
        "training_contract",
        "candidate_eligibility_policy",
    ):
        value = manifest.get(key)
        if key in {"created_at_utc", "anchor_artifact_path", "training_contract"} and value in (None, ""):
            continue
        if key == "candidate_eligibility_policy" and value is None:
            continue
        if key == "topology" and isinstance(value, dict):
            value = ordered(("kind", "dim", "stages", "pairs_per_stage", "angle_count", "seed", "pairings_sha256"), value)
        elif key == "quant_surfaces" and isinstance(value, list):
            value = [
                ordered(("bit_width", "seed", "score_surface", "prepared_query"), item) if isinstance(item, dict) else item
                for item in value
            ]
        elif key == "objective_contract" and isinstance(value, dict):
            value = ordered_objective(value)
        elif key == "qrels_sha256_by_dataset" and isinstance(value, dict):
            value = ordered_map(value)
        elif key == "split_proof" and isinstance(value, dict):
            value = ordered(("split", "train_only", "proof_sha256", "exclusion_identities"), value)
        elif key == "legal_gates" and isinstance(value, dict):
            value = ordered(
                ("research_train_allowed", "release_train_allowed", "commercial_use_allowed", "free_open_release_allowed"),
                value,
            )
        elif key == "candidate_eligibility_policy" and isinstance(value, dict):
            value = ordered_candidate_policy(value)
        out[key] = value
    return out


def dumps_go_like_json(value: Any) -> str:
    if value is None:
        return "null"
    if value is True:
        return "true"
    if value is False:
        return "false"
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        if not (value == value and value not in (float("inf"), float("-inf"))):
            raise MaterializeError("non-finite float cannot be serialized")
        rendered = format(value, ".15g")
        if "e" in rendered or "E" in rendered:
            rendered = format(Decimal(str(value)), "f").rstrip("0").rstrip(".")
        if rendered == "-0":
            rendered = "0"
        return rendered
    if isinstance(value, str):
        return json.dumps(value, separators=(",", ":"))
    if isinstance(value, list):
        return "[" + ",".join(dumps_go_like_json(item) for item in value) + "]"
    if isinstance(value, dict):
        return "{" + ",".join(
            json.dumps(str(key), separators=(",", ":")) + ":" + dumps_go_like_json(value[key])
            for key in value.keys()
        ) + "}"
    raise MaterializeError(f"cannot serialize {type(value).__name__} in Go-like JSON")


def go_struct_sha256(payload: dict[str, Any]) -> str:
    return hashlib.sha256(dumps_go_like_json(manifest_for_go_hash(payload)).encode("utf-8")).hexdigest()


def go_float_json_value(value: float) -> int | float:
    return int(value) if float(value).is_integer() else value


def count_pairs(row: dict[str, Any]) -> int:
    mask = row.get("eligible_pair_mask")
    if not mask:
        n = len(row.get("candidate_doc_ids", []))
        return n * (n - 1)
    total = 0
    if not isinstance(mask, list):
        raise MaterializeError(f"{row.get('row_id')}: eligible_pair_mask must be a list")
    for line in mask:
        if not isinstance(line, list):
            raise MaterializeError(f"{row.get('row_id')}: eligible_pair_mask rows must be lists")
        total += sum(1 for value in line if value is True)
    return total


def add_weights(total: dict[str, float], row: dict[str, Any]) -> None:
    weights = row.get("weights")
    if not isinstance(weights, dict):
        raise MaterializeError(f"{row.get('row_id')}: weights object is required")
    for key in ("q3_gain", "q3_order_guard", "q3_score_distill", "q5_order_guard", "q5_score_distill", "nf_boundary_guard"):
        value = weights.get(key)
        if not isinstance(value, (int, float)):
            raise MaterializeError(f"{row.get('row_id')}: weights.{key} must be numeric")
        total[key] += float(value)


def stream_filter_rows(source_rows: Path, output_rows: Path, target_row_ids: list[str]) -> dict[str, Any]:
    target = set(target_row_ids)
    seen: set[str] = set()
    source_seen: set[str] = set()
    selected_ids: list[str] = []
    all_source_ids: list[str] = []
    candidate_count = 0
    pair_count = 0
    weight_sums = {key: 0.0 for key in ("q3_gain", "q3_order_guard", "q3_score_distill", "q5_order_guard", "q5_score_distill", "nf_boundary_guard")}

    out_path = repo_path(output_rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_name(out_path.name + ".tmp")
    with repo_path(source_rows).open("r", encoding="utf-8") as src, tmp.open("w", encoding="utf-8") as dst:
        for line_no, line in enumerate(src, start=1):
            raw = line.rstrip("\n")
            if not raw:
                raise MaterializeError(f"{display_path(source_rows)}:{line_no}: blank rows are forbidden")
            try:
                row = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise MaterializeError(f"{display_path(source_rows)}:{line_no}: invalid JSON row") from exc
            if not isinstance(row, dict):
                raise MaterializeError(f"{display_path(source_rows)}:{line_no}: row must be object")
            row_id = row.get("row_id")
            if not isinstance(row_id, str) or not row_id:
                raise MaterializeError(f"{display_path(source_rows)}:{line_no}: row_id is required")
            if row_id in source_seen:
                raise MaterializeError(f"{display_path(source_rows)}:{line_no}: duplicate source row_id {row_id!r}")
            source_seen.add(row_id)
            all_source_ids.append(row_id)
            if row_id not in target:
                continue
            if row_id in seen:
                raise MaterializeError(f"{display_path(source_rows)}:{line_no}: duplicate selected row_id {row_id!r}")
            seen.add(row_id)
            selected_ids.append(row_id)
            candidate_ids = row.get("candidate_doc_ids")
            if not isinstance(candidate_ids, list) or not candidate_ids:
                raise MaterializeError(f"{row_id}: candidate_doc_ids must be non-empty")
            candidate_count += len(candidate_ids)
            pair_count += count_pairs(row)
            add_weights(weight_sums, row)
            dst.write(raw)
            dst.write("\n")
    missing = sorted(target - seen)
    if missing:
        tmp.unlink(missing_ok=True)
        raise MaterializeError(f"source rows missing {len(missing)} fold train row IDs")
    if selected_ids != target_row_ids:
        tmp.unlink(missing_ok=True)
        raise MaterializeError("selected rows do not exactly match split.fold.train.row_ids order")
    tmp.replace(out_path)
    return {
        "selected_row_ids": selected_ids,
        "source_row_ids": all_source_ids,
        "candidate_count": candidate_count,
        "pair_count": pair_count,
        "weight_sums": weight_sums,
    }


def build_fold_manifest(source_manifest: dict[str, Any], row_ids: list[str], weight_sums: dict[str, float]) -> OrderedDict[str, Any]:
    manifest = OrderedDict((key, source_manifest[key]) for key in manifest_for_go_hash(source_manifest).keys() if key in source_manifest)
    manifest["row_count"] = len(row_ids)
    manifest["row_id_sha256"] = aoqt_row_id_sha256(row_ids)
    objective = OrderedDict(manifest["objective_contract"])
    objective["weight_sums"] = OrderedDict((key, weight_sums[key]) for key in (
        "q3_gain",
        "q3_order_guard",
        "q3_score_distill",
        "q5_order_guard",
        "q5_score_distill",
        "nf_boundary_guard",
    ))
    objective["weight_sums"] = OrderedDict((key, go_float_json_value(value)) for key, value in objective["weight_sums"].items())
    manifest["objective_contract"] = objective
    return manifest


def build_fold_preflight(
    source_preflight: dict[str, Any],
    manifest: dict[str, Any],
    manifest_sha256: str,
    rows_path: Path,
    manifest_path: Path,
    preflight_path: Path,
    row_count: int,
    candidate_count: int,
    pair_count: int,
) -> OrderedDict[str, Any]:
    preflight = OrderedDict()
    for key in (
        "schema",
        "created_at_utc",
        "quality_claim",
        "research_only",
        "actual_training_ran",
        "actual_eval_ran",
        "actual_vector_export_ran",
        "plan_sha256",
        "input_sha256",
        "row_count",
        "candidate_count",
        "pair_count",
        "turboquant_seed",
        "topology_seed",
        "quant_score_surface",
        "qrels_sha256_by_dataset",
        "source_semantics",
        "outputs",
        "calibration_manifest_sha256",
        "legal_gates",
        "objective_contract",
        "training_contract",
        "candidate_eligibility_policy",
    ):
        if key in source_preflight:
            preflight[key] = source_preflight[key]
    preflight["row_count"] = row_count
    preflight["candidate_count"] = candidate_count
    preflight["pair_count"] = pair_count
    preflight["outputs"] = OrderedDict(
        (
            ("manifest_json", str(repo_path(manifest_path).resolve())),
            ("preflight_json", str(repo_path(preflight_path).resolve())),
            ("rows_jsonl", str(repo_path(rows_path).resolve())),
        )
    )
    preflight["calibration_manifest_sha256"] = manifest_sha256
    preflight["legal_gates"] = manifest["legal_gates"]
    preflight["objective_contract"] = manifest["objective_contract"]
    preflight["training_contract"] = manifest.get("training_contract")
    if manifest.get("candidate_eligibility_policy") is not None:
        preflight["candidate_eligibility_policy"] = manifest["candidate_eligibility_policy"]
    return preflight


def require_source_bindings(plan_path: Path, source_manifest: dict[str, Any], source_preflight: dict[str, Any], source_row_ids: list[str]) -> None:
    if source_manifest.get("schema") != "eos.q3_aoqt_sidecar_manifest.v1":
        raise MaterializeError("source materialized manifest schema mismatch")
    if source_preflight.get("schema") != "eos.q3_aoqt_sidecar_materializer_preflight.v1":
        raise MaterializeError("source materialized preflight schema mismatch")
    for field in ("quality_claim", "actual_training_ran", "actual_eval_ran", "actual_vector_export_ran"):
        expected = False
        if source_preflight.get(field) is not expected:
            raise MaterializeError(f"source preflight {field} must be false")
    if source_preflight.get("research_only") is not True:
        raise MaterializeError("source preflight research_only must be true")
    plan_sha = sha256_file(plan_path)
    if source_preflight.get("plan_sha256") != plan_sha:
        raise MaterializeError("source preflight plan_sha256 does not match source plan")
    input_sha = source_preflight.get("input_sha256")
    if not isinstance(input_sha, dict) or input_sha.get(display_path(plan_path)) != plan_sha:
        raise MaterializeError("source preflight input_sha256 does not bind source plan")
    if source_manifest.get("row_count") != len(source_row_ids) or source_preflight.get("row_count") != len(source_row_ids):
        raise MaterializeError("source materialized row_count does not match source rows")
    if source_manifest.get("row_id_sha256") != aoqt_row_id_sha256(source_row_ids):
        raise MaterializeError("source materialized row_id_sha256 does not match source rows")


def materialize(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = repo_path(args.output_dir)
    rows_out = output_dir / "rows.jsonl"
    manifest_out = output_dir / "manifest.json"
    preflight_out = output_dir / "preflight.json"
    row_ids_out = output_dir / "row-ids.json"
    existing = [path for path in (rows_out, manifest_out, preflight_out, row_ids_out) if path.exists()]
    if existing:
        raise MaterializeError(f"refusing to overwrite existing fold outputs: {[display_path(path) for path in existing]}")

    split_sha = check_file_hash(args.split_manifest, args.expected_split_manifest_sha256, "split manifest")
    receipt_sha = check_file_hash(args.split_validation, args.expected_split_validation_sha256, "split validation")
    plan_sha = check_file_hash(args.source_plan, args.expected_source_plan_sha256, "source plan")
    source_manifest_raw_sha = check_file_hash(args.source_materialized_manifest, args.expected_source_materialized_manifest_sha256, "source materialized manifest")
    source_rows_raw_sha = check_file_hash(args.source_materialized_rows, args.expected_source_materialized_rows_sha256, "source materialized rows")
    source_preflight_raw_sha = check_file_hash(args.source_materialized_preflight, args.expected_source_materialized_preflight_sha256, "source materialized preflight")

    split_manifest, receipt, fold, train_row_ids = validate_split_inputs(args.split_manifest, args.split_validation, args.fold_id)
    if split_manifest.get("source_plan", {}).get("sha256") != plan_sha:
        raise MaterializeError("split manifest source_plan.sha256 does not match source plan")

    source_manifest = load_json(args.source_materialized_manifest)
    source_preflight = load_json(args.source_materialized_preflight)
    row_stats = stream_filter_rows(args.source_materialized_rows, rows_out, train_row_ids)
    require_source_bindings(args.source_plan, source_manifest, source_preflight, row_stats["source_row_ids"])

    fold_manifest = build_fold_manifest(source_manifest, train_row_ids, row_stats["weight_sums"])
    manifest_sha = go_struct_sha256(fold_manifest)
    fold_preflight = build_fold_preflight(
        source_preflight,
        fold_manifest,
        manifest_sha,
        rows_out,
        manifest_out,
        preflight_out,
        len(train_row_ids),
        row_stats["candidate_count"],
        row_stats["pair_count"],
    )

    write_json_atomic(manifest_out, fold_manifest)
    write_json_atomic(preflight_out, fold_preflight)
    rows_sha = sha256_file(rows_out)
    preflight_sha = sha256_file(preflight_out)
    row_id_receipt = OrderedDict(
        (
            ("schema", ROW_ID_RECEIPT_SCHEMA),
            ("fold_id", args.fold_id),
            ("row_count", len(train_row_ids)),
            ("row_id_sha256", aoqt_row_id_sha256(train_row_ids)),
            ("row_ids_sha256", sha256_json(train_row_ids)),
            ("row_ids", train_row_ids),
            ("split", OrderedDict((
                ("manifest_path", display_path(args.split_manifest)),
                ("manifest_sha256", split_sha),
                ("validation_path", display_path(args.split_validation)),
                ("validation_sha256", receipt_sha),
                ("validation_payload_sha256", receipt["manifest_payload_sha256"]),
                ("fold_train_row_ids_sha256", fold["train"]["row_ids_sha256"]),
            ))),
            ("source", OrderedDict((
                ("plan_path", display_path(args.source_plan)),
                ("plan_sha256", plan_sha),
                ("materialized_manifest_path", display_path(args.source_materialized_manifest)),
                ("materialized_manifest_sha256", source_manifest_raw_sha),
                ("materialized_rows_path", display_path(args.source_materialized_rows)),
                ("materialized_rows_sha256", source_rows_raw_sha),
                ("materialized_preflight_path", display_path(args.source_materialized_preflight)),
                ("materialized_preflight_sha256", source_preflight_raw_sha),
                ("source_row_count", len(row_stats["source_row_ids"])),
                ("source_row_id_sha256", aoqt_row_id_sha256(row_stats["source_row_ids"])),
            ))),
            ("outputs", OrderedDict((
                ("rows_jsonl", display_path(rows_out)),
                ("rows_sha256", rows_sha),
                ("manifest_json", display_path(manifest_out)),
                ("manifest_sha256", manifest_sha),
                ("manifest_raw_file_sha256", sha256_file(manifest_out)),
                ("preflight_json", display_path(preflight_out)),
                ("preflight_sha256", preflight_sha),
                ("row_ids_json", display_path(row_ids_out)),
            ))),
            ("actual_training_ran", False),
            ("actual_eval_ran", False),
            ("actual_official_data_eval_ran", False),
            ("quality_claim", False),
            ("research_only", True),
        )
    )
    write_json_atomic(row_ids_out, row_id_receipt)
    return row_id_receipt


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        receipt = materialize(args)
    except (MaterializeError, splitter.SplitError, stage2.PlanError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(
        "materialized "
        f"fold={receipt['fold_id']} rows={receipt['row_count']} "
        f"manifest_sha256={receipt['outputs']['manifest_sha256']} "
        f"rows_sha256={receipt['outputs']['rows_sha256']} "
        f"preflight_sha256={receipt['outputs']['preflight_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
