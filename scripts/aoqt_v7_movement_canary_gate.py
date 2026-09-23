#!/usr/bin/env python3
"""Train/dev movement canary for AOQT V7 research candidates.

This is not the official heldout promotion gate.  It consumes train-only AOQT
metrics plus a non-official dev proxy report and emits one deterministic JSON
decision.  The fixed v1 policy is intentionally conservative where the
preregistered shorthand used "or" style thresholds: both the absolute and
ratio floors must pass.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
import sys
from pathlib import Path
from typing import Any


SCHEMA = "eos.aoqt.v7_movement_canary_gate.v1"
POLICY_ID = "aoqt-v7-movement-canary-conservative-v1"
HARD_MIN_POLICY_ID = "aoqt-v7-movement-canary-conservative-v1-explicit-hard-min"
TRAIN_METRICS_SCHEMA = "eos.q3_aoqt_sidecar_metrics.v2"
SIDECAR_SCHEMA = "eos.aoqt.givens_transform_sidecar.v1"
DEV_EVIDENCE_SCHEMA = "eos.q3_aoqt_sidecar_dev_evidence.v1"
DEV_PROXY_SCHEMA = "eos.aoqt.v7_non_official_dev_proxy.v1"
TRAIN_SIDECAR_BINDING_SCHEMA = "eos.aoqt.v7_train_sidecar_binding.v1"
TRAIN_SPLIT_BINDING_SCHEMA = "eos.aoqt.v7_train_split_binding.v1"
DEV_PROXY_EVIDENCE_LABEL = "non_official_dev_proxy"
DEV_PROXY_SPLIT_BINDING_SCHEMA = "eos.aoqt.v7_dev_proxy_split_binding.v1"
SPLIT_MANIFEST_SCHEMA = "eos.aoqt.v7_dev_split_manifest.v1"
SPLIT_VALIDATION_SCHEMA = "eos.aoqt.v7_dev_split_validation.v1"
V7_OPTIMIZER_MODE = "aoqt-v7-dev-trust-region-v1"
V7_COORDINATE_SEARCH_STRATEGY = "q3_gain_protected_cone_trust_region_dev_v1"
V7_ACCEPTED_SINGLE_COORDINATE_RULE = "aoqt-v7-dev-preregistered-accepted-single-coordinate-v1"
DOMAINS = ("fiqa", "nfcorpus", "scifact")
SURFACES = ("dense", "q3", "q5")
DIMENSION = 384
CANONICAL_D384_SEED191_PAIRINGS_SHA256 = "dcac29b5b7ab30291bb6791b1e88178e1c755d5d3f995a06efe29a4f430bc6f8"
AOQT_TOPOLOGY = {
    "id": "aoqt_givens_v1",
    "dim": DIMENSION,
    "stages": 8,
    "pairs_per_stage": 192,
    "angle_count": 1536,
    "angle_cap_default": 0.04,
    "angle_cap_hard": 0.08,
    "seed": 191,
}
MACRO_MISMATCH_TOLERANCE = 1e-12
FORBIDDEN_TRAIN_TRUE_CLAIM_KEYS = (
    "quality_claim",
    "official_heldout_gate",
    "official_claim",
    "official_quality_claim",
    "heldout_quality_claim",
    "promotion_quality_claim",
    "production_quality_claim",
    "release_claim",
    "release_train_allowed",
    "quality_release_authorized",
    "commercial_use_allowed",
    "commercial_release_authorized",
    "commercial_claim",
)

DEFAULT_THRESHOLDS: dict[str, Any] = {
    "angle_nonzero_min": 16,
    "angle_nonzero_fraction_min": 0.01,
    "angle_l2_min": 2e-4,
    "unique_nonzero_angles_min": 4,
    "accepted_proposals_min": 8,
    "accepted_proposal_fraction_min": 0.15,
    "q3_top10_churn_min": 25,
    "q3_top10_churn_per_domain_min": 1,
    "q3_macro_ndcg_lift_preferred_min": 0.0006,
    "q3_macro_ndcg_lift_hard_min": 0.0004,
    "q3_domain_ndcg_delta_min": 0.0,
    "q3_domain_recall_at_100_delta_min": 0.0,
    "dense_domain_ndcg_delta_min": -0.0005,
    "q5_domain_ndcg_delta_min": -0.001,
}


class CanaryGateError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_json(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def read_json(path: Path, label: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise CanaryGateError(f"{label}: invalid JSON: {exc}") from exc
    except OSError as exc:
        raise CanaryGateError(f"{label}: cannot read: {exc}") from exc


def mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise CanaryGateError(f"{label}: expected object")
    return value


def finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CanaryGateError(f"{label}: expected finite number")
    result = float(value)
    if not math.isfinite(result):
        raise CanaryGateError(f"{label}: expected finite number")
    return result


def nonnegative_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise CanaryGateError(f"{label}: expected nonnegative integer")
    return value


def optional_nonnegative_int(payload: dict[str, Any], key: str, label: str) -> int:
    if key not in payload:
        return 0
    return nonnegative_int(payload.get(key), label)


def require_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise CanaryGateError(f"{label}: expected non-empty string")
    return value


def require_sha256(value: Any, label: str) -> str:
    text = require_string(value, label)
    if len(text) != 64 or any(char not in "0123456789abcdef" for char in text):
        raise CanaryGateError(f"{label}: expected lowercase sha256 hex")
    return text


def require_false_or_absent_claims(payload: dict[str, Any], label: str) -> None:
    for key in FORBIDDEN_TRAIN_TRUE_CLAIM_KEYS:
        if payload.get(key) is True:
            raise CanaryGateError(f"{label}.{key}: canary input must not claim quality or official heldout status")


def exact_schema(payload: dict[str, Any], expected: str, label: str) -> None:
    observed = payload.get("schema")
    if observed != expected:
        raise CanaryGateError(f"{label}.schema: unsupported schema {observed!r}, want {expected!r}")


def get_path(payload: dict[str, Any], keys: tuple[str, ...], label: str) -> Any:
    current: Any = payload
    for key in keys:
        current = mapping(current, label).get(key)
    return current


def first_present(payload: dict[str, Any], paths: tuple[tuple[str, ...], ...], label: str) -> Any:
    for path in paths:
        current: Any = payload
        found = True
        for key in path:
            if not isinstance(current, dict) or key not in current:
                found = False
                break
            current = current[key]
        if found:
            return current
    raise CanaryGateError(f"{label}: missing required field")


def float32_number(value: Any, label: str) -> float:
    number = finite_number(value, label)
    try:
        return struct.unpack("<f", struct.pack("<f", number))[0]
    except (OverflowError, struct.error) as exc:
        raise CanaryGateError(f"{label}: not representable as float32") from exc


def transform_pairings_sha256(stages: list[dict[str, Any]], label: str) -> str:
    digest = hashlib.sha256()
    for stage_index, stage in enumerate(stages):
        pairs = stage.get("pairs")
        if not isinstance(pairs, list):
            raise CanaryGateError(f"{label}.stages[{stage_index}].pairs: expected array")
        for pair_index, pair in enumerate(pairs):
            if not isinstance(pair, list) or len(pair) != 2 or any(isinstance(item, bool) or not isinstance(item, int) for item in pair):
                raise CanaryGateError(f"{label}.stages[{stage_index}].pairs[{pair_index}]: expected integer pair")
            digest.update(f"{pair[0]},{pair[1]}\n".encode("ascii"))
    return digest.hexdigest()


def transform_angles_sha256(stages: list[dict[str, Any]], label: str) -> str:
    digest = hashlib.sha256()
    for stage_index, stage in enumerate(stages):
        angles = stage.get("angles")
        if not isinstance(angles, list):
            raise CanaryGateError(f"{label}.stages[{stage_index}].angles: expected array")
        for angle_index, angle in enumerate(angles):
            number = float32_number(angle, f"{label}.stages[{stage_index}].angles[{angle_index}]")
            digest.update(f"{number:.9g}\n".encode("ascii"))
    return digest.hexdigest()


def validate_sidecar(sidecar: dict[str, Any], *, canonical_pairings_sha256: str = CANONICAL_D384_SEED191_PAIRINGS_SHA256) -> dict[str, Any]:
    if "schema" in sidecar:
        exact_schema(sidecar, SIDECAR_SCHEMA, "sidecar")
    if sidecar.get("version") != "eos/aoqt-givens-transform/v1":
        raise CanaryGateError("sidecar.version: unsupported version")
    if sidecar.get("kind") != AOQT_TOPOLOGY["id"]:
        raise CanaryGateError("sidecar.kind: unsupported AOQT kind")
    if sidecar.get("dim") != AOQT_TOPOLOGY["dim"]:
        raise CanaryGateError(f"sidecar.dim: expected {AOQT_TOPOLOGY['dim']}")
    if sidecar.get("seed") != AOQT_TOPOLOGY["seed"]:
        raise CanaryGateError(f"sidecar.seed: expected {AOQT_TOPOLOGY['seed']}")
    angle_cap = float32_number(sidecar.get("angle_cap"), "sidecar.angle_cap")
    if not 0.0 < angle_cap <= AOQT_TOPOLOGY["angle_cap_default"]:
        raise CanaryGateError("sidecar.angle_cap: outside accepted default cap")
    if "topology" in sidecar:
        topology = mapping(sidecar.get("topology"), "sidecar.topology")
        if topology != AOQT_TOPOLOGY:
            raise CanaryGateError("sidecar.topology: AOQT topology mismatch")
    else:
        topology = dict(AOQT_TOPOLOGY)
    stages_raw = sidecar.get("stages")
    if not isinstance(stages_raw, list) or len(stages_raw) != AOQT_TOPOLOGY["stages"]:
        raise CanaryGateError(f"sidecar.stages: expected {AOQT_TOPOLOGY['stages']} stages")
    stages: list[dict[str, Any]] = []
    angle_count = 0
    max_abs = 0.0
    for stage_index, stage_value in enumerate(stages_raw):
        stage = mapping(stage_value, f"sidecar.stages[{stage_index}]")
        pairs = stage.get("pairs")
        angles = stage.get("angles")
        if not isinstance(pairs, list) or not isinstance(angles, list):
            raise CanaryGateError(f"sidecar.stages[{stage_index}]: pairs and angles arrays are required")
        if len(pairs) != AOQT_TOPOLOGY["pairs_per_stage"] or len(angles) != AOQT_TOPOLOGY["pairs_per_stage"]:
            raise CanaryGateError(f"sidecar.stages[{stage_index}]: expected {AOQT_TOPOLOGY['pairs_per_stage']} pairs/angles")
        seen_coordinates: set[int] = set()
        for pair_index, pair in enumerate(pairs):
            if not isinstance(pair, list) or len(pair) != 2 or any(isinstance(item, bool) or not isinstance(item, int) for item in pair):
                raise CanaryGateError(f"sidecar.stages[{stage_index}].pairs[{pair_index}]: invalid pair")
            left, right = pair
            if left < 0 or right < 0 or left >= DIMENSION or right >= DIMENSION or left == right or left in seen_coordinates or right in seen_coordinates:
                raise CanaryGateError(f"sidecar.stages[{stage_index}].pairs[{pair_index}]: coordinates are not a disjoint D384 pairing")
            seen_coordinates.update((left, right))
            max_abs = max(max_abs, abs(float32_number(angles[pair_index], f"sidecar.stages[{stage_index}].angles[{pair_index}]")))
        angle_count += len(angles)
        stages.append(stage)
    if angle_count != AOQT_TOPOLOGY["angle_count"]:
        raise CanaryGateError(f"sidecar: expected {AOQT_TOPOLOGY['angle_count']} angles")
    if max_abs > angle_cap + 1e-7:
        raise CanaryGateError("sidecar: angle exceeds declared cap")
    audit = mapping(sidecar.get("audit"), "sidecar.audit")
    pairings_sha = transform_pairings_sha256(stages, "sidecar")
    angles_sha = transform_angles_sha256(stages, "sidecar")
    if require_sha256(audit.get("pairings_sha256"), "sidecar.audit.pairings_sha256") != pairings_sha:
        raise CanaryGateError("sidecar.audit.pairings_sha256: does not bind sidecar stages")
    if require_sha256(audit.get("angles_sha256"), "sidecar.audit.angles_sha256") != angles_sha:
        raise CanaryGateError("sidecar.audit.angles_sha256: does not bind sidecar stages")
    if pairings_sha != canonical_pairings_sha256:
        raise CanaryGateError("sidecar.audit.pairings_sha256: not canonical D384 seed191 pairings")
    if "orthogonality_frobenius_per_dim" not in audit:
        raise CanaryGateError("sidecar.audit.orthogonality_frobenius_per_dim: required")
    if finite_number(audit["orthogonality_frobenius_per_dim"], "sidecar.audit.orthogonality_frobenius_per_dim") < 0:
        raise CanaryGateError("sidecar.audit.orthogonality_frobenius_per_dim: must be non-negative")
    return {
        "schema": SIDECAR_SCHEMA,
        "kind": sidecar["kind"],
        "dim": sidecar["dim"],
        "seed": sidecar["seed"],
        "topology_sha256": sha256_json(topology),
        "content_sha256": sha256_json(sidecar),
        "pairings_sha256": pairings_sha,
        "angles_sha256": angles_sha,
        "angle_cap": angle_cap,
        "angle_count": angle_count,
    }


def require_native_sidecar_hashes(train_metrics: dict[str, Any], sidecar_evidence: dict[str, Any], *, require_present: bool) -> None:
    summary = train_metrics.get("summary")
    if isinstance(summary, dict) and ("angles_sha256" in summary or require_present):
        if require_sha256(summary.get("angles_sha256"), "train_metrics.summary.angles_sha256") != sidecar_evidence["angles_sha256"]:
            raise CanaryGateError("train_metrics.summary.angles_sha256: does not bind the supplied sidecar")

    topology = train_metrics.get("topology")
    if isinstance(topology, dict) and ("pairings_sha256" in topology or require_present):
        if require_sha256(topology.get("pairings_sha256"), "train_metrics.topology.pairings_sha256") != sidecar_evidence["pairings_sha256"]:
            raise CanaryGateError("train_metrics.topology.pairings_sha256: does not bind the supplied sidecar")


def require_sidecar_binding(train_metrics: dict[str, Any], sidecar_evidence: dict[str, Any]) -> None:
    expected = {
        "schema": TRAIN_SIDECAR_BINDING_SCHEMA,
        "sidecar_schema": SIDECAR_SCHEMA,
        "sidecar_content_sha256": sidecar_evidence["content_sha256"],
        "topology_sha256": sidecar_evidence["topology_sha256"],
        "pairings_sha256": sidecar_evidence["pairings_sha256"],
        "angles_sha256": sidecar_evidence["angles_sha256"],
        "angle_count": sidecar_evidence["angle_count"],
    }
    if "sidecar_binding" in train_metrics:
        binding = mapping(train_metrics.get("sidecar_binding"), "train_metrics.sidecar_binding")
        exact_schema(binding, TRAIN_SIDECAR_BINDING_SCHEMA, "train_metrics.sidecar_binding")
        if binding != expected:
            raise CanaryGateError("train_metrics.sidecar_binding: does not bind the supplied sidecar evidence")
        require_native_sidecar_hashes(train_metrics, sidecar_evidence, require_present=False)
        return

    summary = mapping(train_metrics.get("summary"), "train_metrics.summary")
    topology = mapping(train_metrics.get("topology"), "train_metrics.topology")
    require_native_sidecar_hashes(train_metrics, sidecar_evidence, require_present=True)
    if topology.get("kind") != AOQT_TOPOLOGY["id"] or topology.get("dim") != AOQT_TOPOLOGY["dim"] or topology.get("stages") != AOQT_TOPOLOGY["stages"] or topology.get("pairs_per_stage") != AOQT_TOPOLOGY["pairs_per_stage"] or topology.get("angle_count") != AOQT_TOPOLOGY["angle_count"] or topology.get("seed") != AOQT_TOPOLOGY["seed"]:
        raise CanaryGateError("train_metrics.topology: AOQT topology mismatch")


def extract_angles(sidecar: dict[str, Any]) -> list[float]:
    angles: list[float] = []
    stages = sidecar.get("stages")
    if isinstance(stages, list):
        for stage_index, stage in enumerate(stages):
            raw_angles = mapping(stage, f"sidecar.stages[{stage_index}]").get("angles")
            if not isinstance(raw_angles, list):
                raise CanaryGateError(f"sidecar.stages[{stage_index}].angles: expected array")
            for angle_index, value in enumerate(raw_angles):
                angles.append(finite_number(value, f"sidecar.stages[{stage_index}].angles[{angle_index}]"))
        return angles
    raw_angles = sidecar.get("angles")
    if isinstance(raw_angles, list):
        for index, value in enumerate(raw_angles):
            angles.append(finite_number(value, f"sidecar.angles[{index}]"))
        return angles
    raise CanaryGateError("sidecar: missing stages[].angles or angles")


def angle_stats(sidecar: dict[str, Any]) -> dict[str, Any]:
    angles = extract_angles(sidecar)
    if not angles:
        raise CanaryGateError("sidecar: no angles")
    nonzero = [angle for angle in angles if angle != 0.0]
    l2 = math.sqrt(math.fsum(angle * angle for angle in angles))
    max_abs = max(abs(angle) for angle in angles)
    unique_nonzero = len({format(angle, ".9g") for angle in nonzero})
    return {
        "angle_count": len(angles),
        "angle_l2": l2,
        "angle_max_abs": max_abs,
        "nonzero_angles": len(nonzero),
        "nonzero_fraction": len(nonzero) / len(angles),
        "unique_nonzero_angles": unique_nonzero,
        "angles_sha256": sidecar.get("audit", {}).get("angles_sha256") if isinstance(sidecar.get("audit"), dict) else sidecar.get("angles_sha256"),
    }


def train_movement(train_metrics: dict[str, Any], sidecar: dict[str, Any], *, canonical_pairings_sha256: str = CANONICAL_D384_SEED191_PAIRINGS_SHA256) -> dict[str, Any]:
    exact_schema(train_metrics, TRAIN_METRICS_SCHEMA, "train_metrics")
    sidecar_evidence = validate_sidecar(sidecar, canonical_pairings_sha256=canonical_pairings_sha256)
    require_sidecar_binding(train_metrics, sidecar_evidence)
    plan = mapping(train_metrics.get("plan", {}), "train_metrics.plan")
    summary = mapping(train_metrics.get("summary", train_metrics), "train_metrics.summary")
    require_false_or_absent_claims(train_metrics, "train_metrics")
    require_false_or_absent_claims(summary, "train_metrics.summary")
    diagnostics = mapping(summary.get("optimizer_diagnostics", {}), "train_metrics.summary.optimizer_diagnostics")
    stats = angle_stats(sidecar)
    metric_l2 = summary.get("angle_l2")
    if metric_l2 is not None:
        stats["metrics_angle_l2"] = finite_number(metric_l2, "train_metrics.summary.angle_l2")
    metric_max_abs = summary.get("angle_max_abs")
    if metric_max_abs is not None:
        stats["metrics_angle_max_abs"] = finite_number(metric_max_abs, "train_metrics.summary.angle_max_abs")
    accepted = nonnegative_int(
        first_present(
            {"summary": summary},
            (
                ("summary", "optimizer_diagnostics", "accepted_proposals"),
                ("summary", "optimizer_diagnostics", "accepted_steps"),
                ("summary", "steps"),
            ),
            "accepted proposals",
        ),
        "accepted proposals",
    )
    attempts_raw = diagnostics.get("proposal_attempts", diagnostics.get("attempted_steps", accepted))
    attempts = nonnegative_int(attempts_raw, "proposal_attempts")
    if attempts == 0:
        raise CanaryGateError("proposal_attempts: expected positive count")
    plan_optimizer_mode = plan.get("optimizer_mode")
    summary_optimizer_mode = summary.get("optimizer_mode")
    diagnostics_optimizer_mode = diagnostics.get("optimizer_mode")
    coordinate_proposal_attempts = optional_nonnegative_int(diagnostics, "coordinate_proposal_attempts", "train_metrics.summary.optimizer_diagnostics.coordinate_proposal_attempts")
    coordinate_accepted_proposals = optional_nonnegative_int(diagnostics, "coordinate_accepted_proposals", "train_metrics.summary.optimizer_diagnostics.coordinate_accepted_proposals")
    coordinate_rejected_proposals = optional_nonnegative_int(diagnostics, "coordinate_rejected_proposals", "train_metrics.summary.optimizer_diagnostics.coordinate_rejected_proposals")
    trust_region_proposal_attempts = optional_nonnegative_int(diagnostics, "trust_region_proposal_attempts", "train_metrics.summary.optimizer_diagnostics.trust_region_proposal_attempts")
    trust_region_accepted_blocks = optional_nonnegative_int(diagnostics, "trust_region_accepted_blocks", "train_metrics.summary.optimizer_diagnostics.trust_region_accepted_blocks")
    trust_region_accepted_singles = optional_nonnegative_int(diagnostics, "trust_region_accepted_singles", "train_metrics.summary.optimizer_diagnostics.trust_region_accepted_singles")
    trust_region_max_moved_angles = optional_nonnegative_int(diagnostics, "trust_region_max_moved_angles", "train_metrics.summary.optimizer_diagnostics.trust_region_max_moved_angles")
    trust_region_radius_raw = diagnostics.get("trust_region_radius", 0.0)
    trust_region_radius = finite_number(trust_region_radius_raw, "train_metrics.summary.optimizer_diagnostics.trust_region_radius")
    accepted_single_coordinate_rule = diagnostics.get("trust_region_accepted_single_coordinate_rule")
    receipts = diagnostics.get("proposal_receipts")
    if not isinstance(receipts, list) or not receipts:
        raise CanaryGateError("train_metrics.summary.optimizer_diagnostics.proposal_receipts: expected non-empty receipt array")
    accepted_coordinate_receipts = []
    for index, receipt_value in enumerate(receipts):
        receipt = mapping(receipt_value, f"train_metrics.summary.optimizer_diagnostics.proposal_receipts[{index}]")
        if receipt.get("accepted") is not True:
            continue
        if receipt.get("kind") != "coordinate":
            raise CanaryGateError(f"train_metrics.summary.optimizer_diagnostics.proposal_receipts[{index}].kind: accepted receipt must be coordinate")
        receipt_strategy = receipt.get("coordinate_search_strategy", receipt.get("strategy"))
        if receipt_strategy != V7_COORDINATE_SEARCH_STRATEGY:
            raise CanaryGateError(f"train_metrics.summary.optimizer_diagnostics.proposal_receipts[{index}].coordinate_search_strategy: expected {V7_COORDINATE_SEARCH_STRATEGY!r}")
        moved_count = nonnegative_int(receipt.get("moved_angle_count"), f"train_metrics.summary.optimizer_diagnostics.proposal_receipts[{index}].moved_angle_count")
        moved_indices = receipt.get("moved_angle_indices")
        if not isinstance(moved_indices, list) or any(isinstance(item, bool) or not isinstance(item, int) for item in moved_indices):
            raise CanaryGateError(f"train_metrics.summary.optimizer_diagnostics.proposal_receipts[{index}].moved_angle_indices: expected integer array")
        if moved_count != len(moved_indices):
            raise CanaryGateError(f"train_metrics.summary.optimizer_diagnostics.proposal_receipts[{index}].moved_angle_count: does not match moved_angle_indices")
        if moved_count <= 0:
            raise CanaryGateError(f"train_metrics.summary.optimizer_diagnostics.proposal_receipts[{index}].moved_angle_count: accepted receipt must move at least one angle")
        if moved_indices != sorted(moved_indices) or len(set(moved_indices)) != len(moved_indices):
            raise CanaryGateError(f"train_metrics.summary.optimizer_diagnostics.proposal_receipts[{index}].moved_angle_indices: must be strictly sorted and unique")
        for moved_index in moved_indices:
            if moved_index < 0 or moved_index >= AOQT_TOPOLOGY["angle_count"]:
                raise CanaryGateError(f"train_metrics.summary.optimizer_diagnostics.proposal_receipts[{index}].moved_angle_indices: index out of bounds")
        accepted_coordinate_receipts.append({"ordinal": receipt.get("ordinal"), "moved_angle_count": moved_count, "moved_angle_indices": moved_indices})
    if len(accepted_coordinate_receipts) != coordinate_accepted_proposals:
        raise CanaryGateError("train_metrics.summary.optimizer_diagnostics.proposal_receipts: accepted coordinate receipt count does not match coordinate_accepted_proposals")
    accepted_receipt_max_moved = max((receipt["moved_angle_count"] for receipt in accepted_coordinate_receipts), default=0)
    accepted_receipt_multi_angle_count = sum(1 for receipt in accepted_coordinate_receipts if receipt["moved_angle_count"] >= 2)
    stats.update(
        {
            "accepted_proposals": accepted,
            "proposal_attempts": attempts,
            "accepted_proposal_fraction": accepted / attempts,
            "sidecar_binding": sidecar_evidence,
            "optimizer_provenance": {
                "plan_optimizer_mode": plan_optimizer_mode,
                "summary_optimizer_mode": summary_optimizer_mode,
                "diagnostics_optimizer_mode": diagnostics_optimizer_mode,
                "dev_only": diagnostics.get("dev_only"),
                "coordinate_search_strategy": diagnostics.get("coordinate_search_strategy"),
                "trust_region_radius": trust_region_radius,
                "proposal_attempts": attempts,
                "accepted_proposals": accepted,
                "coordinate_proposal_attempts": coordinate_proposal_attempts,
                "coordinate_accepted_proposals": coordinate_accepted_proposals,
                "coordinate_rejected_proposals": coordinate_rejected_proposals,
                "trust_region_proposal_attempts": trust_region_proposal_attempts,
                "trust_region_accepted_blocks": trust_region_accepted_blocks,
                "trust_region_accepted_singles": trust_region_accepted_singles,
                "trust_region_max_moved_angles": trust_region_max_moved_angles,
                "trust_region_accepted_single_coordinate_rule": accepted_single_coordinate_rule,
                "accepted_coordinate_receipt_count": len(accepted_coordinate_receipts),
                "accepted_receipt_max_moved_angle_count": accepted_receipt_max_moved,
                "accepted_receipt_multi_angle_count": accepted_receipt_multi_angle_count,
            },
        }
    )
    return stats


def nested_number(payload: dict[str, Any], paths: tuple[tuple[str, ...], ...], label: str) -> float:
    return finite_number(first_present(payload, paths, label), label)


def domain_delta(domain_node: dict[str, Any], surface: str, metric: str, domain: str) -> float:
    compact_metric = metric.replace("_at_", "")
    paths = (
        ("delta", surface, metric),
        ("delta", surface, compact_metric),
        (surface, f"{metric}_delta"),
        (surface, f"{compact_metric}_delta"),
        (f"{surface}_{metric}_delta",),
        (f"{surface}_{compact_metric}_delta",),
        (f"{surface}_{metric}", "delta"),
        (f"{surface}_{compact_metric}", "delta"),
    )
    return nested_number(domain_node, paths, f"dev_proxy.domains.{domain}.{surface}.{metric}_delta")


def domain_churn(domain_node: dict[str, Any], domain: str) -> int:
    paths = (
        ("q3_top10_churn",),
        ("top10_churn", "q3"),
        ("top10", "q3_churn"),
        ("q3", "top10_churn"),
        ("movement", "q3_top10_churn"),
        ("movement", "top10_churn", "q3"),
    )
    value = first_present(domain_node, paths, f"dev_proxy.domains.{domain}.q3_top10_churn")
    return nonnegative_int(value, f"dev_proxy.domains.{domain}.q3_top10_churn")


def require_split_manifest(split_manifest: dict[str, Any], split_manifest_path: Path) -> dict[str, Any]:
    exact_schema(split_manifest, SPLIT_MANIFEST_SCHEMA, "split_manifest")
    provenance = mapping(split_manifest.get("provenance"), "split_manifest.provenance")
    payload_sha = require_sha256(provenance.get("manifest_sha256"), "split_manifest.provenance.manifest_sha256")
    comparable = json.loads(json.dumps(split_manifest))
    comparable_provenance = mapping(comparable.get("provenance"), "split_manifest.provenance")
    comparable_provenance.pop("manifest_sha256", None)
    observed_payload_sha = sha256_json(comparable)
    if observed_payload_sha != payload_sha:
        raise CanaryGateError("split_manifest.provenance.manifest_sha256: does not bind split payload")
    source_plan = mapping(split_manifest.get("source_plan"), "split_manifest.source_plan")
    official = mapping(split_manifest.get("official_qid_registry"), "split_manifest.official_qid_registry")
    folds = split_manifest.get("folds")
    if not isinstance(folds, list) or not folds:
        raise CanaryGateError("split_manifest.folds: expected non-empty array")
    if split_manifest.get("actual_training_ran") is not False or split_manifest.get("actual_eval_ran") is not False or split_manifest.get("actual_official_data_eval_ran") is not False:
        raise CanaryGateError("split_manifest: must be split-only with no training/eval")
    return {
        "schema": SPLIT_MANIFEST_SCHEMA,
        "path": str(split_manifest_path),
        "file_sha256": sha256_file(split_manifest_path),
        "payload_sha256": payload_sha,
        "source_plan_sha256": require_sha256(source_plan.get("sha256"), "split_manifest.source_plan.sha256"),
        "official_qid_registry_sha256": require_sha256(official.get("manifest_sha256"), "split_manifest.official_qid_registry.manifest_sha256"),
        "official_registry_source_sha256": require_sha256(official.get("source_sha256"), "split_manifest.official_qid_registry.source_sha256"),
        "folds": folds,
    }


def require_split_validation_receipt(split_validation: dict[str, Any], split_validation_path: Path, split_evidence: dict[str, Any]) -> dict[str, Any]:
    exact_schema(split_validation, SPLIT_VALIDATION_SCHEMA, "split_validation")
    if split_validation.get("passed") is not True:
        raise CanaryGateError("split_validation.passed: expected true")
    expected = {
        "manifest_file_sha256": split_evidence["file_sha256"],
        "manifest_payload_sha256": split_evidence["payload_sha256"],
        "source_plan_sha256": split_evidence["source_plan_sha256"],
        "official_qid_registry_sha256": split_evidence["official_qid_registry_sha256"],
    }
    for key, expected_value in expected.items():
        observed = require_sha256(split_validation.get(key), f"split_validation.{key}")
        if observed != expected_value:
            raise CanaryGateError(f"split_validation.{key}: does not bind supplied split manifest")
    leakage = split_validation.get("leakage_proof")
    if isinstance(leakage, dict):
        if leakage.get("passed") is not True or leakage.get("official_overlap_count") != 0:
            raise CanaryGateError("split_validation.leakage_proof: expected passed true with zero official overlap")
    return {
        "schema": SPLIT_VALIDATION_SCHEMA,
        "path": str(split_validation_path),
        "file_sha256": sha256_file(split_validation_path),
        "payload_sha256": sha256_json(split_validation),
        "manifest_file_sha256": expected["manifest_file_sha256"],
        "manifest_payload_sha256": expected["manifest_payload_sha256"],
        "source_plan_sha256": expected["source_plan_sha256"],
        "official_qid_registry_sha256": expected["official_qid_registry_sha256"],
        "passed": True,
    }


def qid_set_hashes(qids_by_dataset: dict[str, Any], label: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for domain in DOMAINS:
        qids = qids_by_dataset.get(domain)
        if not isinstance(qids, list) or any(not isinstance(qid, str) or not qid for qid in qids):
            raise CanaryGateError(f"{label}.{domain}: expected qid array")
        if qids != sorted(qids):
            raise CanaryGateError(f"{label}.{domain}: qids must be sorted")
        result[domain] = sha256_json(qids)
    return result


def split_fold_binding(split_evidence: dict[str, Any], fold_id: str) -> dict[str, Any]:
    fold = None
    for candidate in split_evidence["folds"]:
        if isinstance(candidate, dict) and candidate.get("name") == fold_id:
            fold = candidate
            break
    if fold is None:
        raise CanaryGateError(f"dev_proxy.split_manifest.fold_id: unknown fold {fold_id!r}")
    dev = mapping(fold.get("dev"), f"split_manifest.folds.{fold_id}.dev")
    qids_by_dataset = mapping(dev.get("qids_by_dataset"), f"split_manifest.folds.{fold_id}.dev.qids_by_dataset")
    qid_hashes = qid_set_hashes(qids_by_dataset, f"split_manifest.folds.{fold_id}.dev.qids_by_dataset")
    counts = {domain: nonnegative_int(mapping(dev.get("qid_count_by_dataset"), f"split_manifest.folds.{fold_id}.dev.qid_count_by_dataset").get(domain), f"split_manifest.folds.{fold_id}.dev.qid_count_by_dataset.{domain}") for domain in DOMAINS}
    for domain in DOMAINS:
        if counts[domain] != len(qids_by_dataset[domain]):
            raise CanaryGateError(f"split_manifest.folds.{fold_id}.dev.qid_count_by_dataset.{domain}: count does not match qids")
    qids_by_dataset_sha = require_sha256(dev.get("qids_by_dataset_sha256"), f"split_manifest.folds.{fold_id}.dev.qids_by_dataset_sha256")
    if qids_by_dataset_sha != sha256_json(qids_by_dataset):
        raise CanaryGateError(f"split_manifest.folds.{fold_id}.dev.qids_by_dataset_sha256: does not bind qids")
    return {
        "schema": DEV_PROXY_SPLIT_BINDING_SCHEMA,
        "split_manifest_schema": SPLIT_MANIFEST_SCHEMA,
        "split_manifest_file_sha256": split_evidence["file_sha256"],
        "split_manifest_payload_sha256": split_evidence["payload_sha256"],
        "split_validation_schema": SPLIT_VALIDATION_SCHEMA,
        "split_validation_file_sha256": split_evidence["split_validation"]["file_sha256"],
        "split_validation_payload_sha256": split_evidence["split_validation"]["payload_sha256"],
        "source_plan_sha256": split_evidence["source_plan_sha256"],
        "official_qid_registry_sha256": split_evidence["official_qid_registry_sha256"],
        "official_registry_source_sha256": split_evidence["official_registry_source_sha256"],
        "official_qids_file_sha256": split_evidence["official_qid_registry_sha256"],
        "fold_id": fold_id,
        "dev_qid_count_by_dataset": counts,
        "dev_qid_set_sha256_by_dataset": qid_hashes,
        "dev_qids_by_dataset_sha256": qids_by_dataset_sha,
    }


def split_train_fold_binding(split_evidence: dict[str, Any], fold_id: str) -> dict[str, Any]:
    fold = None
    for candidate in split_evidence["folds"]:
        if isinstance(candidate, dict) and candidate.get("name") == fold_id:
            fold = candidate
            break
    if fold is None:
        raise CanaryGateError(f"train_metrics.split_binding.fold_id: unknown fold {fold_id!r}")
    train = mapping(fold.get("train"), f"split_manifest.folds.{fold_id}.train")
    row_ids = train.get("row_ids")
    if not isinstance(row_ids, list) or any(not isinstance(row_id, str) or not row_id for row_id in row_ids):
        raise CanaryGateError(f"split_manifest.folds.{fold_id}.train.row_ids: expected row_id array")
    if row_ids != sorted(row_ids):
        raise CanaryGateError(f"split_manifest.folds.{fold_id}.train.row_ids: row_ids must be sorted")
    row_count = nonnegative_int(train.get("row_count"), f"split_manifest.folds.{fold_id}.train.row_count")
    if row_count != len(row_ids):
        raise CanaryGateError(f"split_manifest.folds.{fold_id}.train.row_count: count does not match row_ids")
    row_ids_sha = require_sha256(train.get("row_ids_sha256"), f"split_manifest.folds.{fold_id}.train.row_ids_sha256")
    if row_ids_sha != sha256_json(row_ids):
        raise CanaryGateError(f"split_manifest.folds.{fold_id}.train.row_ids_sha256: does not bind row_ids")
    qids_by_dataset = mapping(train.get("qids_by_dataset"), f"split_manifest.folds.{fold_id}.train.qids_by_dataset")
    qid_hashes = qid_set_hashes(qids_by_dataset, f"split_manifest.folds.{fold_id}.train.qids_by_dataset")
    counts = {domain: nonnegative_int(mapping(train.get("qid_count_by_dataset"), f"split_manifest.folds.{fold_id}.train.qid_count_by_dataset").get(domain), f"split_manifest.folds.{fold_id}.train.qid_count_by_dataset.{domain}") for domain in DOMAINS}
    for domain in DOMAINS:
        if counts[domain] != len(qids_by_dataset[domain]):
            raise CanaryGateError(f"split_manifest.folds.{fold_id}.train.qid_count_by_dataset.{domain}: count does not match qids")
    qids_by_dataset_sha = require_sha256(train.get("qids_by_dataset_sha256"), f"split_manifest.folds.{fold_id}.train.qids_by_dataset_sha256")
    if qids_by_dataset_sha != sha256_json(qids_by_dataset):
        raise CanaryGateError(f"split_manifest.folds.{fold_id}.train.qids_by_dataset_sha256: does not bind qids")
    return {
        "schema": TRAIN_SPLIT_BINDING_SCHEMA,
        "split_manifest_schema": SPLIT_MANIFEST_SCHEMA,
        "split_manifest_file_sha256": split_evidence["file_sha256"],
        "split_manifest_sha256": split_evidence["file_sha256"],
        "split_manifest_payload_sha256": split_evidence["payload_sha256"],
        "source_plan_sha256": split_evidence["source_plan_sha256"],
        "official_qid_registry_sha256": split_evidence["official_qid_registry_sha256"],
        "official_registry_source_sha256": split_evidence["official_registry_source_sha256"],
        "fold_id": fold_id,
        "train_row_count": row_count,
        "train_row_ids_sha256": row_ids_sha,
        "train_qid_count_by_dataset": counts,
        "train_qid_set_sha256_by_dataset": qid_hashes,
        "train_qids_by_dataset_sha256": qids_by_dataset_sha,
    }


def require_dev_proxy_split_binding(dev_proxy: dict[str, Any], split_evidence: dict[str, Any]) -> dict[str, Any]:
    supplied = mapping(dev_proxy.get("split_manifest"), "dev_proxy.split_manifest")
    exact_schema(supplied, DEV_PROXY_SPLIT_BINDING_SCHEMA, "dev_proxy.split_manifest")
    fold_id = require_string(supplied.get("fold_id"), "dev_proxy.split_manifest.fold_id")
    expected = split_fold_binding(split_evidence, fold_id)
    if supplied != expected:
        raise CanaryGateError("dev_proxy.split_manifest: does not bind exact split manifest fold")
    return expected


def require_train_split_binding(train_metrics: dict[str, Any], dev_evidence: dict[str, Any], split_evidence: dict[str, Any], fold_id: str) -> dict[str, Any]:
    exact_schema(dev_evidence, DEV_EVIDENCE_SCHEMA, "dev_evidence")
    if dev_evidence.get("dev_only") is not True:
        raise CanaryGateError("dev_evidence.dev_only: expected true")
    if dev_evidence.get("optimizer_mode") != V7_OPTIMIZER_MODE:
        raise CanaryGateError(f"dev_evidence.optimizer_mode: expected {V7_OPTIMIZER_MODE!r}")
    metrics_binding = mapping(train_metrics.get("split_binding"), "train_metrics.split_binding")
    evidence_binding = mapping(dev_evidence.get("split_binding"), "dev_evidence.split_binding")
    if metrics_binding != evidence_binding:
        raise CanaryGateError("dev_evidence.split_binding: does not match train_metrics.split_binding")
    expected_subset = split_train_fold_binding(split_evidence, fold_id)
    for key, expected_value in expected_subset.items():
        if metrics_binding.get(key) != expected_value:
            raise CanaryGateError(f"train_metrics.split_binding.{key}: does not bind exact split manifest train fold")
    for key in (
        "materialized_manifest_sha256",
        "materialized_rows_sha256",
        "materialized_preflight_sha256",
        "materialized_row_ids_sha256",
    ):
        require_sha256(metrics_binding.get(key), f"train_metrics.split_binding.{key}")
    materialized_row_count = nonnegative_int(metrics_binding.get("materialized_row_count"), "train_metrics.split_binding.materialized_row_count")
    if materialized_row_count != expected_subset["train_row_count"]:
        raise CanaryGateError("train_metrics.split_binding.materialized_row_count: does not match split train row count")
    inputs = train_metrics.get("inputs")
    if isinstance(inputs, dict) and inputs.get("dataset_manifest_sha256") is not None:
        if require_sha256(inputs.get("dataset_manifest_sha256"), "train_metrics.inputs.dataset_manifest_sha256") != metrics_binding["materialized_manifest_sha256"]:
            raise CanaryGateError("train_metrics.split_binding.materialized_manifest_sha256: does not match train metrics inputs")
    plan = train_metrics.get("plan")
    if isinstance(plan, dict) and plan.get("row_count") is not None:
        if nonnegative_int(plan.get("row_count"), "train_metrics.plan.row_count") != materialized_row_count:
            raise CanaryGateError("train_metrics.split_binding.materialized_row_count: does not match train metrics plan")
    return dict(metrics_binding)


def require_dev_evidence_file_bindings(dev_evidence: dict[str, Any], sidecar_evidence: dict[str, Any], *, train_metrics_path: Path, sidecar_path: Path, preflight_path: Path) -> dict[str, Any]:
    exact_schema(dev_evidence, DEV_EVIDENCE_SCHEMA, "dev_evidence")
    expected = {
        "metrics_sha256": sha256_file(train_metrics_path),
        "transform_sha256": sha256_file(sidecar_path),
        "pairings_sha256": sidecar_evidence["pairings_sha256"],
        "angles_sha256": sidecar_evidence["angles_sha256"],
        "preflight_sha256": sha256_file(preflight_path),
    }
    for key, expected_value in expected.items():
        observed = require_sha256(dev_evidence.get(key), f"dev_evidence.{key}")
        if observed != expected_value:
            raise CanaryGateError(f"dev_evidence.{key}: does not match supplied {'raw sidecar' if key in {'transform_sha256', 'pairings_sha256', 'angles_sha256'} else 'file'}")
    return expected


def normalize_dev_proxy(dev_proxy: dict[str, Any], split_evidence: dict[str, Any]) -> dict[str, Any]:
    exact_schema(dev_proxy, DEV_PROXY_SCHEMA, "dev_proxy")
    require_false_or_absent_claims(dev_proxy, "dev_proxy")
    if dev_proxy.get("evidence_label") != DEV_PROXY_EVIDENCE_LABEL:
        raise CanaryGateError(f"dev_proxy.evidence_label: expected {DEV_PROXY_EVIDENCE_LABEL!r}")
    split_binding = require_dev_proxy_split_binding(dev_proxy, split_evidence)
    domains = mapping(dev_proxy.get("domains"), "dev_proxy.domains")
    reports: dict[str, Any] = {}
    for domain in DOMAINS:
        node = mapping(domains.get(domain), f"dev_proxy.domains.{domain}")
        reports[domain] = {
            "q3_ndcg_at_10_delta": domain_delta(node, "q3", "ndcg_at_10", domain),
            "q3_recall_at_100_delta": domain_delta(node, "q3", "recall_at_100", domain),
            "dense_ndcg_at_10_delta": domain_delta(node, "dense", "ndcg_at_10", domain),
            "q5_ndcg_at_10_delta": domain_delta(node, "q5", "ndcg_at_10", domain),
            "q3_top10_churn": domain_churn(node, domain),
        }
    macro_value = None
    macro = dev_proxy.get("macro")
    if isinstance(macro, dict):
        for keys in (
            ("q3_ndcg_at_10_delta",),
            ("q3_ndcg10_delta",),
            ("q3", "ndcg_at_10_delta"),
            ("q3", "ndcg10_delta"),
        ):
            try:
                macro_value = nested_number(macro, (keys,), "dev_proxy.macro.q3_ndcg_at_10_delta")
                break
            except CanaryGateError:
                pass
    recomputed_macro = math.fsum(row["q3_ndcg_at_10_delta"] for row in reports.values()) / len(DOMAINS)
    supplied_macro = macro_value
    if supplied_macro is not None and abs(supplied_macro - recomputed_macro) > MACRO_MISMATCH_TOLERANCE:
        raise CanaryGateError("dev_proxy.macro.q3_ndcg_at_10_delta: supplied macro does not equal per-domain average")
    total_churn = math.fsum(row["q3_top10_churn"] for row in reports.values())
    return {
        "schema": DEV_PROXY_SCHEMA,
        "evidence_label": dev_proxy["evidence_label"],
        "split_manifest": split_binding,
        "domains": reports,
        "macro": {
            "q3_ndcg_at_10_delta": recomputed_macro,
            "q3_top10_churn": int(total_churn),
            "supplied_q3_ndcg_at_10_delta": supplied_macro,
            "mismatch_tolerance": MACRO_MISMATCH_TOLERANCE,
        },
    }


def check(name: str, observed: Any, threshold: Any, passed: bool, failures: list[str]) -> dict[str, Any]:
    if not passed:
        failures.append(name)
    return {"name": name, "observed": observed, "threshold": threshold, "pass": passed}


def evaluate(train_metrics: dict[str, Any], sidecar: dict[str, Any], dev_evidence: dict[str, Any], dev_proxy: dict[str, Any], split_manifest: dict[str, Any], split_manifest_path: Path, *, split_validation: dict[str, Any] | None = None, split_validation_path: Path | None = None, macro_lift_policy: str = "preferred", canonical_pairings_sha256: str = CANONICAL_D384_SEED191_PAIRINGS_SHA256, train_metrics_path: Path | None = None, sidecar_path: Path | None = None, preflight_path: Path | None = None) -> dict[str, Any]:
    thresholds = dict(DEFAULT_THRESHOLDS)
    if macro_lift_policy == "preferred":
        policy_id = POLICY_ID
        q3_macro_min = thresholds["q3_macro_ndcg_lift_preferred_min"]
    elif macro_lift_policy == "hard-min-explicit":
        policy_id = HARD_MIN_POLICY_ID
        q3_macro_min = thresholds["q3_macro_ndcg_lift_hard_min"]
    else:
        raise CanaryGateError(f"unknown macro lift policy {macro_lift_policy!r}")

    movement = train_movement(train_metrics, sidecar, canonical_pairings_sha256=canonical_pairings_sha256)
    dev_file_binding = None
    if train_metrics_path is not None or sidecar_path is not None or preflight_path is not None:
        if train_metrics_path is None or sidecar_path is None or preflight_path is None:
            raise CanaryGateError("dev_evidence file binding: train_metrics_path, sidecar_path, and preflight_path are required together")
        dev_file_binding = require_dev_evidence_file_bindings(dev_evidence, movement["sidecar_binding"], train_metrics_path=train_metrics_path, sidecar_path=sidecar_path, preflight_path=preflight_path)
    split_evidence = require_split_manifest(split_manifest, split_manifest_path)
    if split_validation is None or split_validation_path is None:
        raise CanaryGateError("split_validation: required")
    split_evidence["split_validation"] = require_split_validation_receipt(split_validation, split_validation_path, split_evidence)
    proxy = normalize_dev_proxy(dev_proxy, split_evidence)
    train_split_binding = require_train_split_binding(train_metrics, dev_evidence, split_evidence, proxy["split_manifest"]["fold_id"])
    failures: list[str] = []
    checks: list[dict[str, Any]] = []

    checks.append(check("angles_nonzero_absolute", movement["nonzero_angles"], thresholds["angle_nonzero_min"], movement["nonzero_angles"] >= thresholds["angle_nonzero_min"], failures))
    checks.append(check("angles_nonzero_fraction", movement["nonzero_fraction"], thresholds["angle_nonzero_fraction_min"], movement["nonzero_fraction"] >= thresholds["angle_nonzero_fraction_min"], failures))
    checks.append(check("angles_not_single_angle", movement["nonzero_angles"], 2, movement["nonzero_angles"] >= 2, failures))
    checks.append(check("angle_l2", movement["angle_l2"], thresholds["angle_l2_min"], movement["angle_l2"] >= thresholds["angle_l2_min"], failures))
    checks.append(check("unique_nonzero_angles", movement["unique_nonzero_angles"], thresholds["unique_nonzero_angles_min"], movement["unique_nonzero_angles"] >= thresholds["unique_nonzero_angles_min"], failures))
    checks.append(check("accepted_proposals_absolute", movement["accepted_proposals"], thresholds["accepted_proposals_min"], movement["accepted_proposals"] >= thresholds["accepted_proposals_min"], failures))
    checks.append(check("accepted_proposals_fraction", movement["accepted_proposal_fraction"], thresholds["accepted_proposal_fraction_min"], movement["accepted_proposal_fraction"] >= thresholds["accepted_proposal_fraction_min"], failures))
    provenance = mapping(movement.get("optimizer_provenance"), "movement.optimizer_provenance")
    modes = {
        "plan": provenance.get("plan_optimizer_mode"),
        "summary": provenance.get("summary_optimizer_mode"),
        "diagnostics": provenance.get("diagnostics_optimizer_mode"),
    }
    checks.append(check("v7_optimizer_mode", modes, V7_OPTIMIZER_MODE, all(mode == V7_OPTIMIZER_MODE for mode in modes.values()), failures))
    checks.append(check("v7_optimizer_mode_consistent", modes, "plan == summary == diagnostics", len(set(modes.values())) == 1, failures))
    checks.append(check("v7_dev_only", provenance.get("dev_only"), True, provenance.get("dev_only") is True, failures))
    checks.append(check("v7_coordinate_search_strategy", provenance.get("coordinate_search_strategy"), V7_COORDINATE_SEARCH_STRATEGY, provenance.get("coordinate_search_strategy") == V7_COORDINATE_SEARCH_STRATEGY, failures))
    checks.append(check("trust_region_radius_positive", provenance.get("trust_region_radius"), "> 0", provenance.get("trust_region_radius") > 0, failures))
    checks.append(check("coordinate_proposal_attempts_positive", provenance.get("coordinate_proposal_attempts"), "> 0", provenance.get("coordinate_proposal_attempts") > 0, failures))
    checks.append(check("trust_region_proposal_attempts_positive", provenance.get("trust_region_proposal_attempts"), "> 0", provenance.get("trust_region_proposal_attempts") > 0, failures))
    checks.append(check("trust_region_proposal_attempts_match_coordinate", {"trust_region": provenance.get("trust_region_proposal_attempts"), "coordinate": provenance.get("coordinate_proposal_attempts")}, "equal", provenance.get("trust_region_proposal_attempts") == provenance.get("coordinate_proposal_attempts"), failures))
    checks.append(check("coordinate_proposal_accounting", {"attempts": provenance.get("coordinate_proposal_attempts"), "accepted": provenance.get("coordinate_accepted_proposals"), "rejected": provenance.get("coordinate_rejected_proposals")}, "attempts == accepted + rejected", provenance.get("coordinate_proposal_attempts") == provenance.get("coordinate_accepted_proposals") + provenance.get("coordinate_rejected_proposals"), failures))
    checks.append(check("trust_region_accepted_accounting", {"trust_region_blocks": provenance.get("trust_region_accepted_blocks"), "trust_region_singles": provenance.get("trust_region_accepted_singles"), "coordinate_accepted": provenance.get("coordinate_accepted_proposals")}, "blocks + singles == coordinate accepted", provenance.get("trust_region_accepted_blocks") + provenance.get("trust_region_accepted_singles") == provenance.get("coordinate_accepted_proposals"), failures))
    checks.append(check("accepted_proposals_match_coordinate", {"accepted_proposals": provenance.get("accepted_proposals"), "coordinate_accepted": provenance.get("coordinate_accepted_proposals")}, "equal", provenance.get("accepted_proposals") == provenance.get("coordinate_accepted_proposals"), failures))
    checks.append(check("accepted_coordinate_receipt_accounting", provenance.get("accepted_coordinate_receipt_count"), provenance.get("coordinate_accepted_proposals"), provenance.get("accepted_coordinate_receipt_count") == provenance.get("coordinate_accepted_proposals"), failures))
    checks.append(check("trust_region_max_moved_angles_receipt_bound", {"aggregate": provenance.get("trust_region_max_moved_angles"), "receipt_max": provenance.get("accepted_receipt_max_moved_angle_count")}, "equal", provenance.get("trust_region_max_moved_angles") == provenance.get("accepted_receipt_max_moved_angle_count"), failures))
    accepted_block = provenance.get("accepted_receipt_max_moved_angle_count") >= 2
    accepted_preregistered_single = (
        provenance.get("accepted_coordinate_receipt_count") > 0
        and provenance.get("accepted_receipt_max_moved_angle_count") == 1
        and provenance.get("trust_region_accepted_single_coordinate_rule") == V7_ACCEPTED_SINGLE_COORDINATE_RULE
    )
    checks.append(check("accepted_proposal_receipt_multi_angle_or_preregistered_single", {"accepted_receipt_multi_angle_count": provenance.get("accepted_receipt_multi_angle_count"), "accepted_receipt_max_moved_angle_count": provenance.get("accepted_receipt_max_moved_angle_count"), "accepted_single_coordinate_rule": provenance.get("trust_region_accepted_single_coordinate_rule")}, f"accepted receipt moved_angle_count >= 2 or {V7_ACCEPTED_SINGLE_COORDINATE_RULE}", accepted_block or accepted_preregistered_single, failures))
    if "metrics_angle_l2" in movement:
        checks.append(check("metrics_angle_l2_matches_sidecar", {"metrics": movement["metrics_angle_l2"], "sidecar": movement["angle_l2"]}, 1e-7, abs(movement["metrics_angle_l2"] - movement["angle_l2"]) <= 1e-7, failures))
    if "metrics_angle_max_abs" in movement:
        checks.append(check("metrics_angle_max_abs_matches_sidecar", {"metrics": movement["metrics_angle_max_abs"], "sidecar": movement["angle_max_abs"]}, 1e-7, abs(movement["metrics_angle_max_abs"] - movement["angle_max_abs"]) <= 1e-7, failures))

    checks.append(check("dev_proxy_evidence_label", proxy["evidence_label"], DEV_PROXY_EVIDENCE_LABEL, proxy["evidence_label"] == DEV_PROXY_EVIDENCE_LABEL, failures))
    checks.append(check("q3_top10_churn_total", proxy["macro"]["q3_top10_churn"], thresholds["q3_top10_churn_min"], proxy["macro"]["q3_top10_churn"] >= thresholds["q3_top10_churn_min"], failures))
    for domain, report in proxy["domains"].items():
        checks.append(check(f"{domain}_q3_top10_churn", report["q3_top10_churn"], thresholds["q3_top10_churn_per_domain_min"], report["q3_top10_churn"] >= thresholds["q3_top10_churn_per_domain_min"], failures))
        checks.append(check(f"{domain}_q3_ndcg_at_10_delta", report["q3_ndcg_at_10_delta"], thresholds["q3_domain_ndcg_delta_min"], report["q3_ndcg_at_10_delta"] >= thresholds["q3_domain_ndcg_delta_min"], failures))
        checks.append(check(f"{domain}_q3_recall_at_100_delta", report["q3_recall_at_100_delta"], thresholds["q3_domain_recall_at_100_delta_min"], report["q3_recall_at_100_delta"] >= thresholds["q3_domain_recall_at_100_delta_min"], failures))
        checks.append(check(f"{domain}_dense_ndcg_at_10_delta", report["dense_ndcg_at_10_delta"], thresholds["dense_domain_ndcg_delta_min"], report["dense_ndcg_at_10_delta"] >= thresholds["dense_domain_ndcg_delta_min"], failures))
        checks.append(check(f"{domain}_q5_ndcg_at_10_delta", report["q5_ndcg_at_10_delta"], thresholds["q5_domain_ndcg_delta_min"], report["q5_ndcg_at_10_delta"] >= thresholds["q5_domain_ndcg_delta_min"], failures))
    checks.append(check("q3_macro_ndcg_at_10_lift", proxy["macro"]["q3_ndcg_at_10_delta"], q3_macro_min, proxy["macro"]["q3_ndcg_at_10_delta"] >= q3_macro_min, failures))

    result = {
        "schema": SCHEMA,
        "policy_id": policy_id,
        "quality_claim": False,
        "official_heldout_gate": False,
        "decision": "fail" if failures else "pass",
        "pass": not failures,
        "failures": failures,
        "thresholds": thresholds,
        "movement": movement,
        "dev_proxy": proxy,
        "split_manifest": {key: split_evidence[key] for key in ("schema", "path", "file_sha256", "payload_sha256", "source_plan_sha256", "official_qid_registry_sha256", "official_registry_source_sha256")},
        "split_validation": split_evidence["split_validation"],
        "train_split_binding": train_split_binding,
        "checks": checks,
    }
    if dev_file_binding is not None:
        result["dev_evidence_file_binding"] = dev_file_binding
    result["decision_sha256"] = sha256_json({key: result[key] for key in sorted(result) if key != "decision_sha256"})
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate the AOQT V7 movement canary against train/dev evidence.")
    parser.add_argument("--train-metrics", required=True, type=Path)
    parser.add_argument("--sidecar", required=True, type=Path)
    parser.add_argument("--dev-evidence", required=True, type=Path)
    parser.add_argument("--dev-proxy-report", required=True, type=Path)
    parser.add_argument("--split-manifest", required=True, type=Path)
    parser.add_argument("--split-validation", required=True, type=Path)
    parser.add_argument("--preflight", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--macro-lift-policy", choices=("preferred", "hard-min-explicit"), default="preferred")
    parser.add_argument("--fail-exit-code", action=argparse.BooleanOptionalAction, default=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        train_metrics = mapping(read_json(args.train_metrics, "train metrics"), "train metrics")
        sidecar = mapping(read_json(args.sidecar, "sidecar"), "sidecar")
        dev_evidence = mapping(read_json(args.dev_evidence, "dev evidence"), "dev evidence")
        dev_proxy = mapping(read_json(args.dev_proxy_report, "dev proxy report"), "dev proxy report")
        split_manifest = mapping(read_json(args.split_manifest, "split manifest"), "split manifest")
        split_validation = mapping(read_json(args.split_validation, "split validation"), "split validation")
        result = evaluate(
            train_metrics,
            sidecar,
            dev_evidence,
            dev_proxy,
            split_manifest,
            args.split_manifest,
            split_validation=split_validation,
            split_validation_path=args.split_validation,
            macro_lift_policy=args.macro_lift_policy,
            train_metrics_path=args.train_metrics,
            sidecar_path=args.sidecar,
            preflight_path=args.preflight,
        )
        result["inputs"] = {
            "train_metrics": {"path": str(args.train_metrics), "sha256": sha256_file(args.train_metrics)},
            "sidecar": {"path": str(args.sidecar), "sha256": sha256_file(args.sidecar)},
            "dev_evidence": {"path": str(args.dev_evidence), "sha256": sha256_file(args.dev_evidence)},
            "dev_proxy_report": {"path": str(args.dev_proxy_report), "sha256": sha256_file(args.dev_proxy_report)},
            "split_manifest": {"path": str(args.split_manifest), "sha256": sha256_file(args.split_manifest)},
            "split_validation": {"path": str(args.split_validation), "sha256": sha256_file(args.split_validation)},
            "preflight": {"path": str(args.preflight), "sha256": sha256_file(args.preflight)},
        }
        result["decision_sha256"] = sha256_json({key: result[key] for key in sorted(result) if key != "decision_sha256"})
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    except CanaryGateError as exc:
        print(f"aoqt-v7-movement-canary-gate: {exc}", file=sys.stderr)
        return 2
    return 0 if result["pass"] or not args.fail_exit_code else 1


if __name__ == "__main__":
    raise SystemExit(main())
