#!/usr/bin/env python3
"""Build and validate deterministic AOQT V7 train/dev split manifests.

The harness consumes an existing AOQT Stage 2 calibration plan and emits only
query/group membership metadata.  It never reads vectors, scores, official
metrics, or evaluator output.  Official heldout overlap is checked only through
the pinned qid-only official exclusion manifest and the closed production
registry in ``aoqt_heldout_gate.py``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import aoqt_heldout_gate as heldout_gate  # noqa: E402
import build_aoqt_stage2_calibration as calibration  # noqa: E402


SCHEMA = "eos.aoqt.v7_dev_split_manifest.v1"
VALIDATION_SCHEMA = "eos.aoqt.v7_dev_split_validation.v1"
DEFAULT_FOLDS = 3
DEFAULT_SEED = "aoqt-v7-dev-split-v1"
DEFAULT_PLAN = Path("runs/eos-d384-aoqt-trainonly-v1-20260903T092452Z/aoqt/dataset-scoped/plan.pairable.raw.v4.json")
DEFAULT_OFFICIAL_QIDS = Path("runs/eos-d384-aoqt-trainonly-v1-20260903T092452Z/exclusions-scoped-v2/official-test.json")


class SplitError(ValueError):
    """Raised when a V7 dev split manifest cannot be trusted."""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser("build")
    build.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    build.add_argument("--official-qids", type=Path, default=DEFAULT_OFFICIAL_QIDS)
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--folds", type=int, default=DEFAULT_FOLDS)
    build.add_argument("--seed", default=DEFAULT_SEED)
    build.add_argument("--reserve-fraction", type=float, default=0.0)

    validate = sub.add_parser("validate")
    validate.add_argument("--manifest", type=Path, required=True)
    validate.add_argument("--plan", type=Path)
    validate.add_argument("--official-qids", type=Path)
    validate.add_argument("--output", type=Path)
    return parser.parse_args(argv)


def repo_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def display_path(path: Path) -> str:
    resolved = repo_path(path).resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        return str(resolved)


def canonical_bytes(payload: Any) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha256_json(payload: Any) -> str:
    return hashlib.sha256(canonical_bytes(payload)).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with repo_path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(repo_path(path).read_text(encoding="utf-8"))
    except OSError as exc:
        raise SplitError(f"{display_path(path)}: cannot read JSON") from exc
    except json.JSONDecodeError as exc:
        raise SplitError(f"{display_path(path)}: invalid JSON") from exc
    if not isinstance(payload, dict):
        raise SplitError(f"{display_path(path)}: expected JSON object")
    return payload


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    target = repo_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(target)


def require_sha256(value: Any, label: str, field: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise SplitError(f"{label}: {field} must be lowercase sha256")
    return value


def require_qid(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or re.search(r"\s", value):
        raise SplitError(f"{label}: qid must be a non-empty string without whitespace")
    return value


def require_plan(path: Path) -> dict[str, Any]:
    plan = load_json(path)
    label = display_path(path)
    if plan.get("schema") != calibration.SCHEMA:
        raise SplitError(f"{label}: expected {calibration.SCHEMA}")
    if plan.get("mode") != "plan_only":
        raise SplitError(f"{label}: AOQT V7 split requires a plan-only calibration manifest")
    for field in ("actual_cache_generation_ran", "actual_training_ran", "actual_eval_ran"):
        if plan.get(field) is not False:
            raise SplitError(f"{label}: {field} must be false")
    policy = plan.get("selection_policy")
    if not isinstance(policy, dict) or policy.get("split") != "train":
        raise SplitError(f"{label}: selection_policy.split must be train")
    rows = plan.get("rows")
    if not isinstance(rows, list) or not rows:
        raise SplitError(f"{label}: rows must be non-empty")
    return plan


def plan_official_exclusion(plan: dict[str, Any]) -> dict[str, Any]:
    exclusions = plan.get("exclusions")
    if not isinstance(exclusions, dict):
        raise SplitError("plan: exclusions object is required")
    sets = exclusions.get("sets")
    if not isinstance(sets, list):
        raise SplitError("plan: exclusions.sets list is required")
    matches = [item for item in sets if isinstance(item, dict) and item.get("name") == "official-test"]
    if len(matches) != 1:
        raise SplitError("plan: exactly one official-test exclusion set is required")
    return matches[0]


def require_official_registry_manifest(path: Path, plan: dict[str, Any]) -> dict[str, Any]:
    payload = load_json(path)
    label = display_path(path)
    if payload.get("schema") != calibration.EXCLUSION_SCHEMA:
        raise SplitError(f"{label}: expected qid-only exclusion schema")
    if payload.get("name") != "official-test":
        raise SplitError(f"{label}: official exclusion name must be official-test")
    unexpected = sorted(set(payload) - {"schema", "name", "qids_by_dataset", "source_sha256", "source_sha256_by_file"})
    if unexpected:
        raise SplitError(f"{label}: official qid registry must be qid-only; unexpected fields {unexpected}")
    qids_by_dataset = calibration.require_qids_by_dataset(payload.get("qids_by_dataset"), label)
    registry = heldout_gate.PRODUCTION_TRUSTED_WORKLOAD_REGISTRY
    if registry.get("schema") != heldout_gate.TRUSTED_WORKLOAD_REGISTRY_SCHEMA:
        raise SplitError("heldout gate: trusted workload registry schema mismatch")
    source_sha = require_sha256(payload.get("source_sha256"), label, "source_sha256")
    source_by_file = payload.get("source_sha256_by_file")
    if not isinstance(source_by_file, dict):
        raise SplitError(f"{label}: source_sha256_by_file is required")
    expected_by_file: dict[str, str] = {}
    for dataset in calibration.ALLOWED_DATASETS:
        domain = registry["domains"][dataset]
        qids = qids_by_dataset[dataset]
        if len(qids) != domain["query_count"]:
            raise SplitError(f"{label}: {dataset} official query count mismatch")
        if sha256_json(qids) != domain["qid_set_sha256"]:
            raise SplitError(f"{label}: {dataset} official qid_set_sha256 mismatch")
        expected_by_file[domain["qrels_path_suffix"]] = domain["qrels_sha256"]
    if dict(sorted(source_by_file.items())) != dict(sorted(expected_by_file.items())):
        raise SplitError(f"{label}: source qrels hash registry mismatch")
    if source_sha != sha256_json(dict(sorted(expected_by_file.items()))):
        raise SplitError(f"{label}: source_sha256 rollup mismatch")

    plan_official = plan_official_exclusion(plan)
    manifest_sha = sha256_file(path)
    if manifest_sha != plan_official.get("manifest_sha256"):
        raise SplitError(f"{label}: official manifest sha does not match source plan exclusion binding")
    if plan_official.get("qids_by_dataset_sha256") != sha256_json(qids_by_dataset):
        raise SplitError(f"{label}: qids_by_dataset_sha256 does not match source plan binding")

    return {
        "path": label,
        "manifest_sha256": manifest_sha,
        "registry_schema": registry["schema"],
        "registry_status": registry.get("status"),
        "source_sha256": source_sha,
        "source_sha256_by_file": dict(sorted(expected_by_file.items())),
        "qid_count_by_dataset": {dataset: len(qids_by_dataset[dataset]) for dataset in calibration.ALLOWED_DATASETS},
        "qid_set_sha256_by_dataset": {dataset: registry["domains"][dataset]["qid_set_sha256"] for dataset in calibration.ALLOWED_DATASETS},
        "qids_by_dataset_sha256": sha256_json(qids_by_dataset),
        "qids_by_dataset": qids_by_dataset,
    }


def row_strata(row: dict[str, Any], dataset_source_sha: str) -> list[str]:
    dataset = str(row["dataset"])
    bits = int(row["bits"])
    bucket = str(row["bucket"])
    guard_class = "boundary" if "boundary" in bucket else "top10"
    return [
        f"dataset={dataset}",
        f"dataset={dataset}|source={dataset_source_sha}",
        f"dataset={dataset}|q{bits}",
        f"dataset={dataset}|guard={guard_class}",
        f"dataset={dataset}|q{bits}|bucket={bucket}",
    ]


def build_groups(plan: dict[str, Any]) -> list[dict[str, Any]]:
    provenance = plan.get("provenance")
    if not isinstance(provenance, dict):
        raise SplitError("plan: provenance object is required")
    source_by_dataset = provenance.get("dataset_source_provenance_by_dataset")
    if not isinstance(source_by_dataset, dict):
        raise SplitError("plan: dataset source provenance is required")

    by_key: dict[tuple[str, str], dict[str, Any]] = {}
    row_ids: set[str] = set()
    pair_ids: set[str] = set()
    for index, row in enumerate(plan["rows"]):
        label = f"plan.rows[{index}]"
        if not isinstance(row, dict):
            raise SplitError(f"{label}: row must be object")
        dataset = str(row.get("dataset", ""))
        if dataset not in calibration.ALLOWED_DATASETS:
            raise SplitError(f"{label}: unsupported dataset {dataset!r}")
        qid = require_qid(row.get("qid"), label)
        row_id = require_qid(row.get("row_id"), label)
        if row_id in row_ids:
            raise SplitError(f"{label}: duplicate row_id {row_id!r}")
        row_ids.add(row_id)
        bits = row.get("bits")
        if not isinstance(bits, int) or bits not in {3, 5}:
            raise SplitError(f"{label}: bits must be 3 or 5")
        bucket = str(row.get("bucket", ""))
        if bucket not in {"top10_guard", "nf_boundary80_120_guard"}:
            raise SplitError(f"{label}: unsupported bucket {bucket!r}")
        dataset_source = source_by_dataset.get(dataset)
        if not isinstance(dataset_source, dict):
            raise SplitError(f"plan: missing source provenance for {dataset}")
        dataset_source_sha = require_sha256(dataset_source.get("sha256"), f"plan provenance {dataset}", "sha256")
        row_pairs = row.get("pair_ids")
        if not isinstance(row_pairs, list) or not row_pairs:
            raise SplitError(f"{label}: pair_ids must be non-empty")
        for pair in row_pairs:
            if not isinstance(pair, dict):
                raise SplitError(f"{label}: pair_id entry must be object")
            pair_id = require_qid(pair.get("pair_id"), label)
            if pair_id in pair_ids:
                raise SplitError(f"{label}: duplicate pair_id {pair_id!r}")
            pair_ids.add(pair_id)

        group = by_key.setdefault(
            (dataset, qid),
            {
                "dataset": dataset,
                "qid": qid,
                "group_id": f"{dataset}:{qid}",
                "dataset_source_sha256": dataset_source_sha,
                "row_ids": [],
                "pair_count": 0,
                "row_count_by_stratum": Counter(),
                "bits": set(),
                "buckets": set(),
                "guard_classes": set(),
            },
        )
        if group["dataset_source_sha256"] != dataset_source_sha:
            raise SplitError(f"{label}: group source provenance mismatch")
        group["row_ids"].append(row_id)
        group["pair_count"] += len(row_pairs)
        group["bits"].add(bits)
        group["buckets"].add(bucket)
        group["guard_classes"].add("boundary" if "boundary" in bucket else "top10")
        for stratum in row_strata(row, dataset_source_sha):
            group["row_count_by_stratum"][stratum] += 1

    groups = []
    for group in by_key.values():
        rows = sorted(group["row_ids"])
        strata = dict(sorted(group["row_count_by_stratum"].items()))
        groups.append(
            {
                "group_id": group["group_id"],
                "dataset": group["dataset"],
                "qid": group["qid"],
                "qid_sha256": sha256_json({"dataset": group["dataset"], "qid": group["qid"]}),
                "dataset_source_sha256": group["dataset_source_sha256"],
                "row_count": len(rows),
                "pair_count": group["pair_count"],
                "row_ids": rows,
                "row_ids_sha256": sha256_json(rows),
                "bits": sorted(group["bits"]),
                "buckets": sorted(group["buckets"]),
                "guard_classes": sorted(group["guard_classes"]),
                "row_count_by_stratum": strata,
                "strata_sha256": sha256_json(strata),
            }
        )
    return sorted(groups, key=lambda item: (item["dataset"], item["qid"]))


def stable_digest(seed: str, value: str) -> str:
    return hashlib.sha256(f"{seed}\0{value}".encode("utf-8")).hexdigest()


def select_reserve(groups: list[dict[str, Any]], reserve_fraction: float, folds: int, seed: str) -> set[str]:
    if reserve_fraction < 0 or reserve_fraction >= 0.5:
        raise SplitError("reserve_fraction must be >= 0 and < 0.5")
    if reserve_fraction == 0:
        return set()
    target = round(len(groups) * reserve_fraction)
    if target == 0:
        return set()
    by_dataset = Counter(group["dataset"] for group in groups)
    selected: set[str] = set()
    remaining = dict(by_dataset)
    for group in sorted(groups, key=lambda item: stable_digest(seed + ":reserve", item["group_id"])):
        if len(selected) >= target:
            break
        dataset = group["dataset"]
        if remaining[dataset] - 1 < folds:
            continue
        selected.add(group["group_id"])
        remaining[dataset] -= 1
    return selected


def assign_folds(groups: list[dict[str, Any]], folds: int, seed: str) -> list[dict[str, Any]]:
    if folds < 2:
        raise SplitError("folds must be at least 2")
    if len(groups) < folds:
        raise SplitError("not enough groups for requested folds")
    by_dataset = Counter(group["dataset"] for group in groups)
    too_small = sorted(dataset for dataset, count in by_dataset.items() if count < folds)
    if too_small:
        raise SplitError(f"not enough groups per dataset for requested folds: {too_small}")

    total_strata: Counter[str] = Counter()
    for group in groups:
        total_strata.update(group["row_count_by_stratum"])
    target_by_stratum = {key: value / folds for key, value in total_strata.items()}
    target_rows = sum(group["row_count"] for group in groups) / folds
    target_groups = len(groups) / folds
    fold_strata = [Counter() for _ in range(folds)]
    fold_row_counts = [0 for _ in range(folds)]
    fold_group_counts = [0 for _ in range(folds)]
    assignments: dict[str, int] = {}

    def rarity(group: dict[str, Any]) -> tuple[float, str]:
        score = sum(count / max(total_strata[stratum], 1) for stratum, count in group["row_count_by_stratum"].items())
        return (-score, stable_digest(seed + ":order", group["group_id"]))

    for group in sorted(groups, key=rarity):
        best_fold = 0
        best_score: tuple[float, str] | None = None
        for fold in range(folds):
            delta = 0.0
            for stratum, count in group["row_count_by_stratum"].items():
                before = fold_strata[fold][stratum] - target_by_stratum[stratum]
                after = fold_strata[fold][stratum] + count - target_by_stratum[stratum]
                delta += (after * after) - (before * before)
            before_rows = fold_row_counts[fold] - target_rows
            after_rows = fold_row_counts[fold] + group["row_count"] - target_rows
            before_groups = fold_group_counts[fold] - target_groups
            after_groups = fold_group_counts[fold] + 1 - target_groups
            delta += 0.25 * ((after_rows * after_rows) - (before_rows * before_rows))
            delta += 0.10 * ((after_groups * after_groups) - (before_groups * before_groups))
            tie = stable_digest(seed + f":fold:{fold}", group["group_id"])
            candidate = (delta, tie)
            if best_score is None or candidate < best_score:
                best_fold = fold
                best_score = candidate
        assignments[group["group_id"]] = best_fold
        fold_strata[best_fold].update(group["row_count_by_stratum"])
        fold_row_counts[best_fold] += group["row_count"]
        fold_group_counts[best_fold] += 1

    folds_out = []
    for fold in range(folds):
        dev_groups = [group for group in groups if assignments[group["group_id"]] == fold]
        train_groups = [group for group in groups if assignments[group["group_id"]] != fold]
        folds_out.append(fold_manifest(f"fold-{fold}", train_groups, dev_groups))
    return folds_out


def fold_manifest(name: str, train_groups: list[dict[str, Any]], dev_groups: list[dict[str, Any]]) -> dict[str, Any]:
    train_ids = sorted(group["group_id"] for group in train_groups)
    dev_ids = sorted(group["group_id"] for group in dev_groups)
    return {
        "name": name,
        "train": split_side(train_groups),
        "dev": split_side(dev_groups),
        "proof": {
            "train_group_ids_sha256": sha256_json(train_ids),
            "dev_group_ids_sha256": sha256_json(dev_ids),
            "train_dev_group_overlap_count": len(set(train_ids) & set(dev_ids)),
            "train_dev_group_overlap_sha256": sha256_json(sorted(set(train_ids) & set(dev_ids))),
        },
    }


def split_side(groups: list[dict[str, Any]]) -> dict[str, Any]:
    group_ids = sorted(group["group_id"] for group in groups)
    row_ids = sorted(row_id for group in groups for row_id in group["row_ids"])
    by_dataset: dict[str, list[str]] = {dataset: [] for dataset in calibration.ALLOWED_DATASETS}
    row_counts_by_dataset: Counter[str] = Counter()
    row_counts_by_stratum: Counter[str] = Counter()
    for group in groups:
        by_dataset[group["dataset"]].append(group["qid"])
        row_counts_by_dataset[group["dataset"]] += group["row_count"]
        row_counts_by_stratum.update(group["row_count_by_stratum"])
    qids_by_dataset = {dataset: sorted(qids) for dataset, qids in by_dataset.items()}
    return {
        "group_count": len(group_ids),
        "row_count": len(row_ids),
        "pair_count": sum(group["pair_count"] for group in groups),
        "qid_count_by_dataset": {dataset: len(qids_by_dataset[dataset]) for dataset in calibration.ALLOWED_DATASETS},
        "row_count_by_dataset": dict(sorted(row_counts_by_dataset.items())),
        "row_count_by_stratum": dict(sorted(row_counts_by_stratum.items())),
        "qids_by_dataset_sha256": sha256_json(qids_by_dataset),
        "group_ids_sha256": sha256_json(group_ids),
        "row_ids_sha256": sha256_json(row_ids),
        "group_ids": group_ids,
        "row_ids": row_ids,
        "qids_by_dataset": qids_by_dataset,
    }


def leakage_proof(groups: list[dict[str, Any]], folds: list[dict[str, Any]], official: dict[str, Any], reserve: dict[str, Any] | None = None) -> dict[str, Any]:
    group_by_id = {group["group_id"]: group for group in groups}
    row_owner: dict[str, str] = {}
    for group in groups:
        for row_id in group["row_ids"]:
            prior = row_owner.setdefault(row_id, group["group_id"])
            if prior != group["group_id"]:
                raise SplitError(f"row_id {row_id!r} has multiple group owners")
    official_overlap: dict[str, list[str]] = {}
    for group in groups:
        if group["qid"] in set(official["qids_by_dataset"][group["dataset"]]):
            official_overlap.setdefault(group["dataset"], []).append(group["qid"])
    official_overlap = {dataset: sorted(qids) for dataset, qids in sorted(official_overlap.items())}

    fold_membership: dict[str, list[str]] = defaultdict(list)
    for fold in folds:
        for group_id in fold["dev"]["group_ids"]:
            fold_membership[group_id].append(fold["name"])
    duplicate_dev_groups = {group_id: names for group_id, names in sorted(fold_membership.items()) if len(names) != 1}

    train_dev_overlaps = {
        fold["name"]: sorted(set(fold["train"]["group_ids"]) & set(fold["dev"]["group_ids"]))
        for fold in folds
    }
    row_sibling_leaks = {}
    for fold in folds:
        train_rows = set(fold["train"]["row_ids"])
        dev_rows = set(fold["dev"]["row_ids"])
        leaked_rows = sorted(train_rows & dev_rows)
        if leaked_rows:
            row_sibling_leaks[fold["name"]] = leaked_rows
    reserve_groups = set(reserve["group_ids"] if reserve else [])
    cv_groups = set(fold_membership)
    unknown_groups = sorted((cv_groups | reserve_groups) - set(group_by_id))

    return {
        "unit": "dataset_qid_group",
        "group_count": len(groups),
        "row_count": len(row_owner),
        "official_overlap_count": sum(len(qids) for qids in official_overlap.values()),
        "official_overlap_qids_by_dataset": official_overlap,
        "official_overlap_qids_by_dataset_sha256": sha256_json(official_overlap),
        "duplicate_dev_group_membership_count": len(duplicate_dev_groups),
        "duplicate_dev_group_membership": duplicate_dev_groups,
        "train_dev_group_overlap_count_by_fold": {name: len(values) for name, values in sorted(train_dev_overlaps.items())},
        "train_dev_group_overlap_sha256_by_fold": {name: sha256_json(values) for name, values in sorted(train_dev_overlaps.items())},
        "row_sibling_leak_count_by_fold": {name: len(values) for name, values in sorted(row_sibling_leaks.items())},
        "row_sibling_leaks_sha256": sha256_json(row_sibling_leaks),
        "unknown_group_count": len(unknown_groups),
        "unknown_groups_sha256": sha256_json(unknown_groups),
        "passed": (
            not official_overlap
            and not duplicate_dev_groups
            and all(len(values) == 0 for values in train_dev_overlaps.values())
            and not row_sibling_leaks
            and not unknown_groups
        ),
    }


def manifest_hash_payload(manifest: dict[str, Any]) -> dict[str, Any]:
    payload = json.loads(json.dumps(manifest))
    provenance = payload.get("provenance")
    if isinstance(provenance, dict):
        provenance.pop("manifest_sha256", None)
    return payload


def build_manifest(plan_path: Path, official_path: Path, folds: int, seed: str, reserve_fraction: float) -> dict[str, Any]:
    plan = require_plan(plan_path)
    official = require_official_registry_manifest(official_path, plan)
    groups = build_groups(plan)
    official_qids = {dataset: set(official["qids_by_dataset"][dataset]) for dataset in calibration.ALLOWED_DATASETS}
    overlaps = [group for group in groups if group["qid"] in official_qids[group["dataset"]]]
    if overlaps:
        raise SplitError(f"source plan contains official heldout qids: {len(overlaps)} groups")

    reserve_ids = select_reserve(groups, reserve_fraction, folds, seed)
    cv_groups = [group for group in groups if group["group_id"] not in reserve_ids]
    reserve_groups = [group for group in groups if group["group_id"] in reserve_ids]
    fold_entries = assign_folds(cv_groups, folds, seed)
    reserve = split_side(reserve_groups) if reserve_groups else None
    proof = leakage_proof(groups, fold_entries, official, reserve)
    if not proof["passed"]:
        raise SplitError("split leakage proof failed")

    all_group_ids = sorted(group["group_id"] for group in groups)
    all_row_ids = sorted(row_id for group in groups for row_id in group["row_ids"])
    manifest = {
        "schema": SCHEMA,
        "mode": "plan_only_split_manifest",
        "actual_training_ran": False,
        "actual_eval_ran": False,
        "actual_official_data_eval_ran": False,
        "seed": seed,
        "fold_count": folds,
        "reserve_fraction": reserve_fraction,
        "source_plan": {
            "path": display_path(plan_path),
            "schema": plan["schema"],
            "sha256": sha256_file(plan_path),
            "plan_sha256": plan.get("provenance", {}).get("plan_sha256"),
            "rows_sha256": plan.get("rows_sha256"),
            "row_ids_sha256": plan.get("row_ids_sha256"),
            "workload": plan.get("workload"),
        },
        "official_qid_registry": {k: v for k, v in official.items() if k != "qids_by_dataset"},
        "stratification_policy": {
            "unit": "dataset_qid_group",
            "algorithm": "deterministic_greedy_min_squared_stratum_deviation.v1",
            "folds": folds,
            "dimensions": ["dataset", "dataset_source_sha256", "bits", "guard_class", "boundary_bucket"],
            "reserve_policy": "none" if not reserve_groups else "stable_hash_selected_with_per_dataset_fold_floor",
        },
        "population": {
            "group_count": len(groups),
            "row_count": len(all_row_ids),
            "pair_count": sum(group["pair_count"] for group in groups),
            "group_ids_sha256": sha256_json(all_group_ids),
            "row_ids_sha256": sha256_json(all_row_ids),
            "groups_sha256": sha256_json(groups),
            "group_count_by_dataset": dict(sorted(Counter(group["dataset"] for group in groups).items())),
            "row_count_by_dataset": dict(sorted(Counter(row["dataset"] for row in plan["rows"]).items())),
            "row_count_by_stratum": split_side(groups)["row_count_by_stratum"],
        },
        "groups": groups,
        "reserve": reserve,
        "folds": fold_entries,
        "leakage_proof": proof,
        "provenance": {
            "builder": display_path(Path(__file__)),
            "builder_sha256": sha256_file(Path(__file__)),
            "self_hash_excludes": ["provenance.manifest_sha256"],
        },
    }
    manifest["provenance"]["manifest_sha256"] = sha256_json(manifest_hash_payload(manifest))
    return manifest


def validate_manifest(manifest_path: Path, plan_path: Path | None = None, official_path: Path | None = None) -> dict[str, Any]:
    manifest = load_json(manifest_path)
    label = display_path(manifest_path)
    if manifest.get("schema") != SCHEMA:
        raise SplitError(f"{label}: expected {SCHEMA}")
    provenance = manifest.get("provenance")
    if not isinstance(provenance, dict):
        raise SplitError(f"{label}: provenance is required")
    expected_hash = require_sha256(provenance.get("manifest_sha256"), label, "provenance.manifest_sha256")
    observed_hash = sha256_json(manifest_hash_payload(manifest))
    if observed_hash != expected_hash:
        raise SplitError(f"{label}: provenance.manifest_sha256 mismatch")

    source = manifest.get("source_plan")
    if not isinstance(source, dict):
        raise SplitError(f"{label}: source_plan is required")
    resolved_plan = plan_path or Path(str(source.get("path", "")))
    official_source = manifest.get("official_qid_registry")
    if not isinstance(official_source, dict):
        raise SplitError(f"{label}: official_qid_registry is required")
    resolved_official = official_path or Path(str(official_source.get("path", "")))
    rebuilt = build_manifest(
        resolved_plan,
        resolved_official,
        int(manifest.get("fold_count")),
        str(manifest.get("seed")),
        float(manifest.get("reserve_fraction")),
    )
    comparable_manifest = json.loads(json.dumps(manifest))
    comparable_manifest["source_plan"]["path"] = display_path(resolved_plan)
    comparable_manifest["official_qid_registry"]["path"] = display_path(resolved_official)
    if comparable_manifest != rebuilt:
        raise SplitError(f"{label}: manifest does not match deterministic rebuild")
    result = {
        "schema": VALIDATION_SCHEMA,
        "manifest_path": label,
        "manifest_file_sha256": sha256_file(manifest_path),
        "manifest_payload_sha256": expected_hash,
        "source_plan_sha256": rebuilt["source_plan"]["sha256"],
        "official_qid_registry_sha256": rebuilt["official_qid_registry"]["manifest_sha256"],
        "group_count": rebuilt["population"]["group_count"],
        "row_count": rebuilt["population"]["row_count"],
        "fold_count": rebuilt["fold_count"],
        "reserve_group_count": 0 if rebuilt["reserve"] is None else rebuilt["reserve"]["group_count"],
        "leakage_proof": rebuilt["leakage_proof"],
        "passed": True,
    }
    return result


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.command == "build":
            payload = build_manifest(args.plan, args.official_qids, args.folds, args.seed, args.reserve_fraction)
            write_json_atomic(args.output, payload)
        elif args.command == "validate":
            payload = validate_manifest(args.manifest, args.plan, args.official_qids)
            if args.output:
                write_json_atomic(args.output, payload)
            else:
                print(json.dumps(payload, indent=2, sort_keys=True))
        else:  # pragma: no cover
            raise AssertionError(args.command)
    except (SplitError, calibration.PlanError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
