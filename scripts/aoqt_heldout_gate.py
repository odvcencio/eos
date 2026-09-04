#!/usr/bin/env python3
"""Freeze and attest the D384 AOQT heldout retrieval quality gate.

This is a two-phase, evaluator-independent gate.  ``freeze`` (also exposed as
``preflight``) validates a plan and writes a deterministic frozen manifest;
every metric artifact and native receipt path must be absent at that point.
``attest`` (also exposed as ``aggregate``) requires a caller-supplied SHA-256
trust anchor for that frozen manifest, validates six paired native-evaluator
artifacts plus six adjacent receipts, recomputes metrics from per-query rows,
and writes a hash-bound attestation.  The gate never launches an evaluator and
never reads official metric values.

The strict schemas are deliberately explicit.  A plan is
``eos.aoqt.heldout_gate_plan.v2``.  It names D384 anchor/candidate packages,
runtime MLL package manifests (or the strict synthetic manifest schema used by
the tests), JSON package attestations, source/binary/cwd/argv/dataset/workload
records, three domain qrels records with counts, three exact domain-scoped qid-only
exclusions (``dev4``, ``reserve4``, ``official-test``), and six metric output
paths with six receipt paths.  The separately approved workload manifest
pins every query identity, per-domain count and qid-set digest, qrels counts,
and the NFCorpus rank-80..120 boundary qids.

Each result is ``eos.aoqt.heldout_retrieval_artifact.v2`` and has one row per
approved query.  Dense and q5 report only nDCG@10; q3 reports nDCG@10 and
recall@100.  Dense/q5 recall is intentionally absent because it is not a gate
metric.  An adjacent ``eos.aoqt.native_eval_receipt.v2`` must bind the exact
native executable, argv, cwd, package mode, qrels/workload/package/output
paths, q3/q5 bits, seed, top-k, frozen-manifest digest, artifact hash, and a
unique nonce.  Self-authored metric JSON without such a receipt is rejected.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import re
import struct
import sys
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parent.parent

PLAN_SCHEMA = "eos.aoqt.heldout_gate_plan.v2"
FROZEN_SCHEMA = "eos.aoqt.heldout_gate_frozen.v2"
APPROVED_WORKLOAD_SCHEMA = "eos.aoqt.approved_heldout_workload.v1"
WORKLOAD_SCHEMA = "eos.aoqt.heldout_workload.v2"
PACKAGE_ATTESTATION_SCHEMA = "eos.aoqt.package_attestation.v1"
PACKAGE_MANIFEST_SCHEMA = "eos.aoqt.package_manifest.v1"
SIDECAR_SCHEMA = "eos.aoqt.sidecar_attestation.v1"
ARTIFACT_SCHEMA = "eos.aoqt.heldout_retrieval_artifact.v2"
RECEIPT_SCHEMA = "eos.aoqt.native_eval_receipt.v2"
ATTESTATION_SCHEMA = "eos.aoqt.heldout_gate_attestation.v2"

DOMAINS = ("fiqa", "nfcorpus", "scifact")
ROLES = ("anchor", "candidate")
SURFACES = ("dense", "q3", "q5")
METRICS_BY_SURFACE = {
    "dense": ("ndcg_at_10",),
    "q3": ("ndcg_at_10", "recall_at_100"),
    "q5": ("ndcg_at_10",),
}
EXCLUSION_NAMES = ("dev4", "reserve4", "official-test")
DIMENSION = 384
Q3_BITS = 3
Q5_BITS = 5
TURBOQUANT_SEED = 5581486560434873699
TOP_K = 120
AOQT_TOPOLOGY_SEED = 191
PACKAGE_MANIFEST_VERSION = "manta/package/v0alpha1"
AOQT_TRANSFORM_VERSION = "eos/aoqt-givens-transform/v1"
NATIVE_EVAL_SUBCOMMAND = "eval-retrieval-turboquant"
RESULT_OUTPUT_COUNT = len(DOMAINS) * len(ROLES)
EXPECTED_OUTPUT_COUNT = RESULT_OUTPUT_COUNT * 2  # metric artifact + native receipt
COMPARISON_EPSILON = 1e-12

AOQT_TOPOLOGY = {
    "id": "aoqt_givens_v1",
    "dim": DIMENSION,
    "stages": 8,
    "pairs_per_stage": 192,
    "angle_count": 1536,
    "angle_cap_default": 0.04,
    "angle_cap_hard": 0.08,
}
AOQT_LEGAL_SCOPE = {
    "train_allowed_for_research": True,
    "release_train_allowed": False,
    "commercial_use_allowed": False,
    "free_open_release_allowed": False,
    "quality_claim": False,
}

# Quality policy is immutable.  Domain and per-query safety floors are
# intentionally at least as strict as the macro acceptance surface.
THRESHOLDS = {
    "q3_macro_ndcg_at_10_delta_min": 0.0003,
    "dense_macro_ndcg_at_10_delta_min": -0.0005,
    "q5_macro_ndcg_at_10_delta_min": -0.001,
    "q3_macro_recall_at_100_delta_min": 0.0,
    "q3_domain_ndcg_at_10_delta_min": 0.0,
    "dense_domain_ndcg_at_10_delta_min": -0.0005,
    "q5_domain_ndcg_at_10_delta_min": -0.001,
    "q3_domain_recall_at_100_delta_min": 0.0,
    "q3_query_ndcg_at_10_delta_min": 0.0,
    "dense_query_ndcg_at_10_delta_min": -0.0005,
    "q5_query_ndcg_at_10_delta_min": -0.001,
    "q3_query_recall_at_100_delta_min": 0.0,
}

SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
SAFE_ID_RE = re.compile(r"[A-Za-z0-9_.:-]+\Z")
NONCE_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{15,127}\Z")
SHELL_MARKERS = (";", "&&", "|", ">", "<", "`", "$(")
FORBIDDEN_PAYLOAD_KEYS = {
    "gain",
    "gains",
    "score",
    "scores",
    "similarity",
    "similarities",
    "vector",
    "vectors",
    "embedding",
    "embeddings",
    "float32",
}


class GateError(ValueError):
    """Raised when a gate input or output cannot be trusted."""


class GateContractError(GateError):
    """Raised for malformed, substituted, or drifted gate evidence."""


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_bytes(canonical_json(value))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant {value!r}")


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_json_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise GateContractError(f"{label}: cannot read strict JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise GateContractError(f"{label}: JSON root must be an object")
    return payload


def write_json_new(path: Path, payload: dict[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise GateContractError(f"refusing to overwrite existing output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def require_mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise GateContractError(f"{label}: expected object")
    return value


def require_exact_keys(value: dict[str, Any], expected: Iterable[str], label: str) -> None:
    expected_set = set(expected)
    actual_set = set(value)
    missing = sorted(expected_set - actual_set)
    extra = sorted(actual_set - expected_set)
    if missing or extra:
        details = []
        if missing:
            details.append(f"missing {missing}")
        if extra:
            details.append(f"unexpected {extra}")
        raise GateContractError(f"{label}: key set mismatch ({'; '.join(details)})")


def require_string(value: Any, label: str, *, safe_id: bool = False) -> str:
    if not isinstance(value, str) or not value or any(ord(char) < 0x20 for char in value):
        raise GateContractError(f"{label}: expected non-empty string")
    if safe_id and not SAFE_ID_RE.fullmatch(value):
        raise GateContractError(f"{label}: invalid identifier")
    return value


def require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or not SHA256_RE.fullmatch(value):
        raise GateContractError(f"{label}: expected lowercase sha256")
    return value


def require_finite_number(value: Any, label: str, *, bounded: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise GateContractError(f"{label}: expected finite number")
    number = float(value)
    if not math.isfinite(number):
        raise GateContractError(f"{label}: expected finite number")
    if bounded and not 0.0 <= number <= 1.0:
        raise GateContractError(f"{label}: expected a number in [0,1]")
    return number


def require_integer(value: Any, label: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise GateContractError(f"{label}: expected integer")
    if minimum is not None and value < minimum:
        raise GateContractError(f"{label}: expected integer >= {minimum}")
    return value


def require_qid(value: Any, label: str) -> str:
    qid = require_string(value, label)
    if any(char.isspace() for char in qid):
        raise GateContractError(f"{label}: qid must not contain whitespace")
    return qid


def resolve_path(value: Any, base_dir: Path, label: str) -> Path:
    raw = require_string(value, label)
    candidate = Path(raw)
    if not candidate.is_absolute():
        candidate = base_dir / candidate
    try:
        return candidate.resolve(strict=False)
    except OSError as exc:
        raise GateContractError(f"{label}: cannot resolve path: {exc}") from exc


def require_regular_file(path: Path, label: str) -> None:
    if path.is_symlink():
        raise GateContractError(f"{label}: symlinks are not allowed")
    if not path.is_file():
        raise GateContractError(f"{label}: regular file is required: {path}")


def require_directory(path: Path, label: str) -> None:
    if path.is_symlink() or not path.is_dir():
        raise GateContractError(f"{label}: existing non-symlink directory is required: {path}")


def normalize_file_record(value: Any, base_dir: Path, label: str) -> dict[str, Any]:
    record = require_mapping(value, label)
    require_exact_keys(record, {"path", "sha256", "bytes"}, label)
    path = resolve_path(record["path"], base_dir, f"{label}.path")
    require_regular_file(path, label)
    expected_sha = require_sha256(record["sha256"], f"{label}.sha256")
    actual_sha = sha256_file(path)
    if actual_sha != expected_sha:
        raise GateContractError(f"{label}: sha256 mismatch for {path}")
    actual_bytes = path.stat().st_size
    if record["bytes"] != actual_bytes:
        raise GateContractError(f"{label}: byte-count mismatch for {path}")
    return {"path": str(path), "sha256": actual_sha, "bytes": actual_bytes}


def normalize_directory(value: Any, base_dir: Path, label: str) -> str:
    path = resolve_path(value, base_dir, label)
    require_directory(path, label)
    return str(path)


def path_within(path: Path, root: Path, label: str) -> None:
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise GateContractError(f"{label}: {path} escapes {root}") from exc


def require_no_forbidden_payload_keys(node: Any, label: str) -> None:
    if isinstance(node, dict):
        for key, value in node.items():
            if str(key).lower().replace("-", "_") in FORBIDDEN_PAYLOAD_KEYS:
                raise GateContractError(f"{label}: forbidden gain/vector/score field {key!r}")
            require_no_forbidden_payload_keys(value, label)
    elif isinstance(node, list):
        for value in node:
            require_no_forbidden_payload_keys(value, label)


def normalize_qids_by_domain(value: Any, label: str, *, require_nonempty: bool = True) -> dict[str, list[str]]:
    mapping = require_mapping(value, label)
    unexpected = sorted(set(mapping) - set(DOMAINS))
    missing = [domain for domain in DOMAINS if domain not in mapping]
    if unexpected or missing:
        raise GateContractError(f"{label}: exact domain coverage required; missing={missing} unexpected={unexpected}")
    normalized: dict[str, list[str]] = {}
    for domain in DOMAINS:
        raw = mapping[domain]
        if not isinstance(raw, list):
            raise GateContractError(f"{label}.{domain}: expected qid list")
        qids = [require_qid(item, f"{label}.{domain}") for item in raw]
        if require_nonempty and not qids:
            raise GateContractError(f"{label}.{domain}: qid list must be non-empty")
        if len(qids) != len(set(qids)):
            raise GateContractError(f"{label}.{domain}: duplicate qid")
        normalized[domain] = sorted(qids)
    if require_nonempty and not any(normalized.values()):
        raise GateContractError(f"{label}: at least one qid is required")
    return normalized


def qids_sha256_by_domain(qids_by_domain: dict[str, list[str]]) -> dict[str, str]:
    return {domain: sha256_json(qids_by_domain[domain]) for domain in DOMAINS}


def normalize_qrels_record(value: Any, base_dir: Path, label: str) -> dict[str, Any]:
    record = require_mapping(value, label)
    require_exact_keys(record, {"path", "sha256", "bytes", "query_count", "relevant_count"}, label)
    result = normalize_file_record({key: record[key] for key in ("path", "sha256", "bytes")}, base_dir, label)
    result["query_count"] = require_integer(record["query_count"], f"{label}.query_count", minimum=1)
    result["relevant_count"] = require_integer(record["relevant_count"], f"{label}.relevant_count", minimum=0)
    return result


def normalize_sibling_entries(value: Any, base_dir: Path, label: str) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise GateContractError(f"{label}: non-empty sibling entry list is required")
    entries: list[dict[str, Any]] = []
    names: set[str] = set()
    for index, item in enumerate(value):
        record = require_mapping(item, f"{label}[{index}]")
        require_exact_keys(record, {"name", "path", "sha256", "bytes"}, f"{label}[{index}]")
        name = require_string(record["name"], f"{label}[{index}].name")
        if Path(name).name != name or name in {".", ".."} or name in names:
            raise GateContractError(f"{label}[{index}]: sibling names must be unique basenames")
        path = resolve_path(record["path"], base_dir, f"{label}[{index}].path")
        if path.name != name:
            raise GateContractError(f"{label}[{index}]: name/path mismatch")
        file_record = normalize_file_record({key: record[key] for key in ("path", "sha256", "bytes")}, base_dir, f"{label}[{index}]")
        names.add(name)
        entries.append({"name": name, **file_record})
    return sorted(entries, key=lambda item: item["name"])


def _unpack(fmt: str, data: bytes, offset: int, label: str) -> tuple[Any, int]:
    size = struct.calcsize(fmt)
    if offset < 0 or offset + size > len(data):
        raise GateContractError(f"{label}: truncated binary metadata")
    return struct.unpack_from(fmt, data, offset)[0], offset + size


def read_mll_sections(path: Path, label: str) -> dict[bytes, bytes]:
    """Read the bounded MLL directory used by runtime package manifests.

    This intentionally verifies the container bounds and unique section tags;
    the content SHA is still bound by the caller's file record.  Full MLL
    digest verification remains the runtime helper's responsibility.
    """
    data = path.read_bytes()
    if len(data) < 24 or data[:4] != b"MLL\0":
        raise GateContractError(f"{label}: expected an MLL container")
    section_count, _ = _unpack("<I", data, 16, f"{label}.header")
    directory_end = 24 + section_count * 64
    if section_count <= 0 or directory_end > len(data):
        raise GateContractError(f"{label}: invalid MLL section directory")
    sections: dict[bytes, bytes] = {}
    intervals: list[tuple[int, int]] = []
    for index in range(section_count):
        offset = 24 + index * 64
        entry = data[offset : offset + 64]
        tag = entry[:4]
        if tag in sections:
            raise GateContractError(f"{label}: duplicate MLL section tag {tag!r}")
        section_offset, _ = _unpack("<Q", entry, 4, f"{label}.directory[{index}]")
        section_size, _ = _unpack("<Q", entry, 12, f"{label}.directory[{index}]")
        section_end = section_offset + section_size
        if section_offset < directory_end or section_end > len(data) or section_end < section_offset:
            raise GateContractError(f"{label}: MLL section {tag!r} is out of bounds")
        for old_start, old_end in intervals:
            if section_offset < old_end and old_start < section_end:
                raise GateContractError(f"{label}: overlapping MLL sections")
        intervals.append((section_offset, section_end))
        sections[tag] = data[section_offset:section_end]
    return sections


def parse_mll_strings(body: bytes, label: str) -> list[str]:
    count, offset = _unpack("<I", body, 0, label)
    strings: list[str] = []
    for index in range(count):
        length, offset = _unpack("<I", body, offset, f"{label}[{index}]")
        if offset + length > len(body):
            raise GateContractError(f"{label}[{index}]: truncated UTF-8 string")
        try:
            value = body[offset : offset + length].decode("utf-8")
        except UnicodeDecodeError as exc:
            raise GateContractError(f"{label}[{index}]: invalid UTF-8 string") from exc
        strings.append(value)
        offset += length
    if offset != len(body):
        raise GateContractError(f"{label}: trailing string-table bytes")
    return strings


def parse_mll_head(body: bytes, strings: list[str], label: str) -> dict[str, Any]:
    # HEAD begins with name/description string indices and two uint64 fields,
    # followed by backend/capability indices and typed metadata entries.  This
    # mirrors the stable layout consumed by runtime/authored_manifest_mll.go.
    if len(body) < 24:
        raise GateContractError(f"{label}: HEAD is too short")
    offset = 24
    backends, offset = _unpack("<H", body, offset, label)
    if offset + backends * 2 > len(body):
        raise GateContractError(f"{label}: truncated backend list")
    offset += backends * 2
    capabilities, offset = _unpack("<H", body, offset, label)
    if offset + capabilities * 4 > len(body):
        raise GateContractError(f"{label}: truncated capability list")
    offset += capabilities * 4
    metadata_count, offset = _unpack("<H", body, offset, label)
    metadata: dict[str, Any] = {}
    for index in range(metadata_count):
        key_index, offset = _unpack("<I", body, offset, f"{label}.metadata[{index}]")
        kind, offset = _unpack("<B", body, offset, f"{label}.metadata[{index}]")
        if key_index >= len(strings) or not strings[key_index]:
            raise GateContractError(f"{label}.metadata[{index}]: invalid key index")
        key = strings[key_index]
        if key in metadata:
            raise GateContractError(f"{label}: duplicate metadata key {key!r}")
        if kind == 0:
            value: Any = None
        elif kind == 1:
            raw, offset = _unpack("<?", body, offset, f"{label}.metadata[{index}]")
            value = bool(raw)
        elif kind == 2:
            value, offset = _unpack("<q", body, offset, f"{label}.metadata[{index}]")
        elif kind == 3:
            value, offset = _unpack("<d", body, offset, f"{label}.metadata[{index}]")
            if not math.isfinite(value):
                raise GateContractError(f"{label}.metadata[{index}]: non-finite float")
        elif kind == 4:
            string_index, offset = _unpack("<I", body, offset, f"{label}.metadata[{index}]")
            if string_index >= len(strings):
                raise GateContractError(f"{label}.metadata[{index}]: invalid string index")
            value = strings[string_index]
        else:
            raise GateContractError(f"{label}.metadata[{index}]: unknown value kind {kind}")
        metadata[key] = value
    if offset != len(body):
        raise GateContractError(f"{label}: trailing HEAD bytes")
    return metadata


def _string_at(strings: list[str], index: int, label: str) -> str:
    if index >= len(strings):
        raise GateContractError(f"{label}: invalid string index")
    return strings[index]


def parse_runtime_authored_manifest(path: Path, label: str) -> dict[str, Any]:
    sections = read_mll_sections(path, label)
    if b"HEAD" not in sections or b"STRG" not in sections:
        raise GateContractError(f"{label}: authored manifest missing HEAD/STRG")
    strings = parse_mll_strings(sections[b"STRG"], f"{label}.STRG")
    metadata = parse_mll_head(sections[b"HEAD"], strings, f"{label}.HEAD")
    return {"format": "mll", "metadata": metadata}


def parse_runtime_package_manifest(path: Path, package_path: Path, label: str) -> dict[str, Any]:
    sections = read_mll_sections(path, label)
    if b"HEAD" not in sections or b"STRG" not in sections or b"XPKG" not in sections:
        raise GateContractError(f"{label}: package manifest missing HEAD/STRG/XPKG")
    strings = parse_mll_strings(sections[b"STRG"], f"{label}.STRG")
    metadata = parse_mll_head(sections[b"HEAD"], strings, f"{label}.HEAD")
    required_metadata = ("manifest_version", "package_kind", "module_name", "artifact_version", "file_count")
    if any(key not in metadata for key in required_metadata):
        raise GateContractError(f"{label}: missing required package metadata")
    if metadata["manifest_version"] != PACKAGE_MANIFEST_VERSION or metadata["package_kind"] != "embedding":
        raise GateContractError(f"{label}: package manifest version/kind mismatch")
    if not isinstance(metadata["module_name"], str) or not metadata["module_name"] or not isinstance(metadata["artifact_version"], str) or not metadata["artifact_version"]:
        raise GateContractError(f"{label}: package module/artifact metadata must be non-empty strings")
    file_count = metadata["file_count"]
    if isinstance(file_count, bool) or not isinstance(file_count, int) or file_count <= 0:
        raise GateContractError(f"{label}: invalid package file_count")
    body = sections[b"XPKG"]
    count, offset = _unpack("<I", body, 0, f"{label}.XPKG")
    if count != file_count:
        raise GateContractError(f"{label}: package file_count/XPKG count mismatch")
    files: list[dict[str, Any]] = []
    roles: set[str] = set()
    for index in range(count):
        role_index, offset = _unpack("<I", body, offset, f"{label}.XPKG[{index}]")
        path_index, offset = _unpack("<I", body, offset, f"{label}.XPKG[{index}]")
        byte_count, offset = _unpack("<q", body, offset, f"{label}.XPKG[{index}]")
        if offset + 32 > len(body):
            raise GateContractError(f"{label}.XPKG[{index}]: truncated SHA-256")
        digest = body[offset : offset + 32].hex()
        offset += 32
        role = _string_at(strings, role_index, f"{label}.XPKG[{index}].role")
        relative_name = _string_at(strings, path_index, f"{label}.XPKG[{index}].path")
        if not role or not relative_name or Path(relative_name).name != relative_name or relative_name in roles:
            raise GateContractError(f"{label}.XPKG[{index}]: invalid or duplicate file identity")
        if byte_count < 0:
            raise GateContractError(f"{label}.XPKG[{index}]: negative byte count")
        roles.add(role)
        files.append({"role": role, "path": relative_name, "sha256": digest, "bytes": byte_count})
    if offset != len(body):
        raise GateContractError(f"{label}.XPKG: trailing bytes")
    for item in files:
        file_path = (path.parent / item["path"]).resolve(strict=False)
        path_within(file_path, path.parent, f"{label}.files.{item['role']}")
        require_regular_file(file_path, f"{label}.files.{item['role']}")
        if sha256_file(file_path) != item["sha256"] or file_path.stat().st_size != item["bytes"]:
            raise GateContractError(f"{label}.files.{item['role']}: package manifest file hash/size mismatch")
    artifact = next((item for item in files if item["role"] == "artifact"), None)
    if artifact is None or (path.parent / artifact["path"]).resolve(strict=False) != package_path:
        raise GateContractError(f"{label}: artifact file role does not bind package path")
    return {"format": "mll", "metadata": metadata, "files": files}


def runtime_policy_value(metadata: dict[str, Any], key: str, default: Any = None) -> Any:
    return metadata.get(key, default)


def runtime_bool(metadata: dict[str, Any], key: str, default: bool = False) -> bool:
    value = runtime_policy_value(metadata, key, default)
    if not isinstance(value, bool):
        raise GateContractError(f"runtime package metadata {key!r} must be boolean")
    return value


def runtime_string(metadata: dict[str, Any], key: str, *, required: bool = False) -> str:
    value = runtime_policy_value(metadata, key, "")
    if not isinstance(value, str) or (required and not value):
        raise GateContractError(f"runtime package metadata {key!r} must be a string")
    return value


def _runtime_transform_hashes(stages: list[dict[str, Any]], label: str) -> tuple[str, str]:
    pairings = hashlib.sha256()
    angles = hashlib.sha256()
    for stage in stages:
        for pair, angle in zip(stage["pairs"], stage["angles"]):
            pairings.update(f"{pair[0]},{pair[1]}\n".encode("ascii"))
            angles.update(f"{float(angle):.9g}\n".encode("ascii"))
    return pairings.hexdigest(), angles.hexdigest()


def validate_runtime_transform_payload(path: Path, payload: dict[str, Any], *, expected_anchor: dict[str, str] | None, label: str) -> dict[str, Any]:
    required = {"version", "kind", "dim", "seed", "angle_cap", "stages", "audit"}
    optional = {"seed", "angle_cap", "audit"}
    if set(payload) - (required | optional) or not {"version", "kind", "dim", "stages"}.issubset(payload):
        raise GateContractError(f"{label}: runtime AOQT sidecar key set mismatch")
    if payload["version"] != AOQT_TRANSFORM_VERSION or payload["kind"] != "aoqt_givens_v1" or payload["dim"] != DIMENSION:
        raise GateContractError(f"{label}: runtime AOQT sidecar version/kind/dimension mismatch")
    seed = payload.get("seed", AOQT_TOPOLOGY_SEED)
    if seed != AOQT_TOPOLOGY_SEED:
        raise GateContractError(f"{label}: runtime AOQT sidecar seed mismatch")
    angle_cap = require_finite_number(payload.get("angle_cap", AOQT_TOPOLOGY["angle_cap_default"]), f"{label}.angle_cap")
    if angle_cap <= 0 or angle_cap > AOQT_TOPOLOGY["angle_cap_default"]:
        raise GateContractError(f"{label}: runtime AOQT sidecar angle cap exceeds policy")
    stages = payload["stages"]
    if not isinstance(stages, list) or len(stages) != AOQT_TOPOLOGY["stages"]:
        raise GateContractError(f"{label}: runtime AOQT sidecar stage count mismatch")
    normalized_stages: list[dict[str, Any]] = []
    for stage_index, raw_stage in enumerate(stages):
        stage = require_mapping(raw_stage, f"{label}.stages[{stage_index}]")
        require_exact_keys(stage, {"pairs", "angles"}, f"{label}.stages[{stage_index}]")
        pairs = stage["pairs"]
        angles = stage["angles"]
        if not isinstance(pairs, list) or len(pairs) != AOQT_TOPOLOGY["pairs_per_stage"] or not isinstance(angles, list) or len(angles) != len(pairs):
            raise GateContractError(f"{label}.stages[{stage_index}]: pair/angle count mismatch")
        seen: set[int] = set()
        normalized_pairs: list[list[int]] = []
        normalized_angles: list[float] = []
        for pair_index, raw_pair in enumerate(pairs):
            if not isinstance(raw_pair, list) or len(raw_pair) != 2:
                raise GateContractError(f"{label}.stages[{stage_index}].pairs[{pair_index}]: pair must have two coordinates")
            a = require_integer(raw_pair[0], f"{label}.stages[{stage_index}].pairs[{pair_index}][0]", minimum=0)
            b = require_integer(raw_pair[1], f"{label}.stages[{stage_index}].pairs[{pair_index}][1]", minimum=0)
            if a >= DIMENSION or b >= DIMENSION or a == b or a in seen or b in seen:
                raise GateContractError(f"{label}.stages[{stage_index}].pairs[{pair_index}]: coordinate topology mismatch")
            seen.update((a, b))
            angle = require_finite_number(angles[pair_index], f"{label}.stages[{stage_index}].angles[{pair_index}]")
            if abs(angle) > angle_cap + 1e-7:
                raise GateContractError(f"{label}.stages[{stage_index}].angles[{pair_index}]: exceeds angle cap")
            normalized_pairs.append([a, b])
            normalized_angles.append(angle)
        if len(seen) != DIMENSION:
            raise GateContractError(f"{label}.stages[{stage_index}]: every D384 coordinate must be paired exactly once")
        normalized_stages.append({"pairs": normalized_pairs, "angles": normalized_angles})
    pairings_sha, angles_sha = _runtime_transform_hashes(normalized_stages, label)
    audit = payload.get("audit", {})
    if audit is None:
        audit = {}
    audit = require_mapping(audit, f"{label}.audit")
    unknown_audit = sorted(set(audit) - {"pairings_sha256", "angles_sha256", "orthogonality_frobenius_per_dim"})
    if unknown_audit:
        raise GateContractError(f"{label}.audit: unexpected keys {unknown_audit}")
    if "pairings_sha256" in audit and require_sha256(audit["pairings_sha256"], f"{label}.audit.pairings_sha256") != pairings_sha:
        raise GateContractError(f"{label}: runtime sidecar pairings hash mismatch")
    if "angles_sha256" in audit and require_sha256(audit["angles_sha256"], f"{label}.audit.angles_sha256") != angles_sha:
        raise GateContractError(f"{label}: runtime sidecar angles hash mismatch")
    if "orthogonality_frobenius_per_dim" in audit:
        require_finite_number(audit["orthogonality_frobenius_per_dim"], f"{label}.audit.orthogonality_frobenius_per_dim")
    # A runtime transform has no package-level anchor field.  The candidate
    # package attestation supplies that binding; the transform bytes themselves
    # are bound by the sibling rollup and this computed content audit.
    if expected_anchor is not None and not isinstance(expected_anchor, dict):
        raise GateContractError(f"{label}: invalid expected anchor binding")
    return {
        "schema": SIDECAR_SCHEMA,
        "source_schema": AOQT_TRANSFORM_VERSION,
        "topology": copy.deepcopy(AOQT_TOPOLOGY),
        "anchor": copy.deepcopy(expected_anchor),
        "pairings_sha256": pairings_sha,
        "angles_sha256": angles_sha,
        "legal_scope": copy.deepcopy(AOQT_LEGAL_SCOPE),
        "seed": seed,
        "angle_cap": angle_cap,
        "transform_sha256": sha256_file(path),
    }


def validate_sidecar(path: Path, *, expected_anchor: dict[str, str] | None, label: str) -> dict[str, Any]:
    payload = read_json(path, label)
    if payload.get("version") == AOQT_TRANSFORM_VERSION:
        return validate_runtime_transform_payload(path, payload, expected_anchor=expected_anchor, label=label)
    require_exact_keys(payload, {"schema", "topology", "anchor", "pairings_sha256", "angles_sha256", "legal_scope"}, label)
    if payload["schema"] != SIDECAR_SCHEMA or payload["topology"] != AOQT_TOPOLOGY or payload["legal_scope"] != AOQT_LEGAL_SCOPE:
        raise GateContractError(f"{label}: sidecar schema/topology/legal policy mismatch")
    anchor = require_mapping(payload["anchor"], f"{label}.anchor")
    require_exact_keys(anchor, {"package_sha256", "manifest_sha256"}, f"{label}.anchor")
    anchor_binding = {
        "package_sha256": require_sha256(anchor["package_sha256"], f"{label}.anchor.package_sha256"),
        "manifest_sha256": require_sha256(anchor["manifest_sha256"], f"{label}.anchor.manifest_sha256"),
    }
    if expected_anchor is not None and anchor_binding != expected_anchor:
        raise GateContractError(f"{label}: sidecar anchor binding mismatch")
    return {
        "schema": SIDECAR_SCHEMA,
        "source_schema": SIDECAR_SCHEMA,
        "topology": copy.deepcopy(AOQT_TOPOLOGY),
        "anchor": anchor_binding,
        "pairings_sha256": require_sha256(payload["pairings_sha256"], f"{label}.pairings_sha256"),
        "angles_sha256": require_sha256(payload["angles_sha256"], f"{label}.angles_sha256"),
        "legal_scope": copy.deepcopy(AOQT_LEGAL_SCOPE),
        "seed": None,
        "angle_cap": None,
        "transform_sha256": sha256_file(path),
    }


def parse_runtime_hash_map(value: str, label: str) -> dict[str, str]:
    if not value:
        return {}
    result: dict[str, str] = {}
    for row in value.splitlines():
        if not row or "=" not in row:
            raise GateContractError(f"{label}: malformed hash map")
        name, digest = row.split("=", 1)
        if not name or name in result:
            raise GateContractError(f"{label}: duplicate/empty hash-map key")
        result[name] = require_sha256(digest, f"{label}.{name}")
    return dict(sorted(result.items()))


def validate_runtime_package_policy(native_manifest: dict[str, Any], *, package: dict[str, Any], role: str, policy: dict[str, Any], anchor_binding: dict[str, str], embedding_space_id: str, sidecar_record: dict[str, Any] | None, sidecar_policy: dict[str, Any] | None, label: str) -> None:
    metadata = native_manifest["metadata"]
    files = {item["role"]: item for item in native_manifest["files"]}
    required_roles = {"artifact", "embedding_manifest", "weights", "memory_plan"}
    if not required_roles.issubset(files):
        raise GateContractError(f"{label}: runtime package omits required embedding file roles")
    if role == "anchor":
        if "post_pool_transform" in files or runtime_bool(metadata, "aoqt_transform_enabled"):
            raise GateContractError(f"{label}: anchor runtime package must not carry an AOQT transform")
    else:
        if "post_pool_transform" not in files or not runtime_bool(metadata, "aoqt_transform_enabled"):
            raise GateContractError(f"{label}: candidate runtime package must carry an enabled AOQT transform")
    for key in ("score_spectrum_research_only", "listwise_geometry_research_only"):
        if runtime_bool(metadata, key):
            raise GateContractError(f"{label}: runtime package retains restricted {key} data")
    authored_manifest_path = (Path(package["manifest"]["path"]).parent / files["embedding_manifest"]["path"]).resolve(strict=False)
    authored = parse_runtime_authored_manifest(authored_manifest_path, f"{label}.embedding_manifest")
    authored_meta = authored["metadata"]
    output_dim = authored_meta.get("output_dim")
    if output_dim != DIMENSION:
        raise GateContractError(f"{label}: embedding manifest output_dim is not D384")
    post_pool = runtime_string(authored_meta, "post_pool_transform")
    expected_post_pool = "" if role == "anchor" else "aoqt_givens_v1"
    if post_pool not in ({"", "none"} if role == "anchor" else {expected_post_pool}):
        raise GateContractError(f"{label}: embedding manifest post_pool_transform mismatch")
    enabled = runtime_bool(metadata, "aoqt_transform_enabled")
    if enabled != (role == "candidate"):
        raise GateContractError(f"{label}: runtime AOQT enabled flag mismatch")
    if not enabled:
        return
    expected_bools = {
        "aoqt_transform_research_only": True,
        "aoqt_transform_research_train_allowed": True,
        "aoqt_transform_release_train_allowed": False,
        "aoqt_transform_commercial_use_allowed": False,
        "aoqt_transform_free_open_release_allowed": False,
        "aoqt_transform_quality_claim": False,
    }
    if any(runtime_bool(metadata, key) != expected for key, expected in expected_bools.items()):
        raise GateContractError(f"{label}: runtime AOQT research-only legal flags mismatch")
    if runtime_string(metadata, "aoqt_transform_schema", required=True) != "eos.aoqt_transform_policy.v1":
        raise GateContractError(f"{label}: runtime AOQT policy schema mismatch")
    if runtime_string(metadata, "aoqt_transform_anchor_artifact_sha256", required=True) != anchor_binding["package_sha256"] or runtime_string(metadata, "aoqt_transform_anchor_package_manifest_sha256", required=True) != anchor_binding["manifest_sha256"] or runtime_string(metadata, "aoqt_transform_anchor_embedding_space_id", required=True) != embedding_space_id:
        raise GateContractError(f"{label}: runtime AOQT anchor binding mismatch")
    if sidecar_record is None or sidecar_policy is None:
        raise GateContractError(f"{label}: runtime AOQT sidecar record is missing")
    if files["post_pool_transform"]["path"] != Path(sidecar_record["path"]).name or files["post_pool_transform"]["sha256"] != sidecar_record["sha256"] or runtime_string(metadata, "aoqt_transform_transform_sha256", required=True) != sidecar_record["sha256"]:
        raise GateContractError(f"{label}: runtime AOQT transform file binding mismatch")
    if runtime_string(metadata, "aoqt_transform_pairings_sha256", required=True) != sidecar_policy["pairings_sha256"] or runtime_string(metadata, "aoqt_transform_angles_sha256", required=True) != sidecar_policy["angles_sha256"]:
        raise GateContractError(f"{label}: runtime AOQT pairings/angles binding mismatch")
    qrels_map = parse_runtime_hash_map(runtime_string(metadata, "aoqt_transform_qrels_sha256_by_dataset"), f"{label}.runtime.qrels_sha256_by_dataset")
    if set(qrels_map) != set(DOMAINS):
        raise GateContractError(f"{label}: runtime AOQT qrels hash coverage mismatch")
    for key in ("aoqt_transform_dataset_manifest_sha256", "aoqt_transform_compatibility_digest"):
        require_sha256(runtime_string(metadata, key, required=True), f"{label}.{key}")


def normalize_package_record(value: Any, base_dir: Path, label: str, *, expected_role: str | None = None) -> dict[str, Any]:
    record = require_mapping(value, label)
    required_keys = {"path", "sha256", "bytes", "manifest", "attestation", "sibling_rollup_sha256"}
    optional_canonical_keys = {"embedding_space_id", "sidecar", "sidecar_policy", "attestation_policy"}
    missing = sorted(required_keys - set(record))
    extra = sorted(set(record) - required_keys - optional_canonical_keys)
    if missing or extra:
        details = []
        if missing:
            details.append(f"missing {missing}")
        if extra:
            details.append(f"unexpected {extra}")
        raise GateContractError(f"{label}: key set mismatch ({'; '.join(details)})")
    package = normalize_file_record({key: record[key] for key in ("path", "sha256", "bytes")}, base_dir, label)
    package["manifest"] = normalize_file_record(record["manifest"], base_dir, f"{label}.manifest")
    package["attestation"] = normalize_file_record(record["attestation"], base_dir, f"{label}.attestation")
    if len({package["path"], package["manifest"]["path"], package["attestation"]["path"]}) != 3:
        raise GateContractError(f"{label}: package, manifest, and attestation paths must differ")
    package["sibling_rollup_sha256"] = require_sha256(record["sibling_rollup_sha256"], f"{label}.sibling_rollup_sha256")
    attestation_path = Path(package["attestation"]["path"])
    policy = read_json(attestation_path, f"{label}.attestation")
    require_exact_keys(
        policy,
        {"schema", "role", "package_sha256", "manifest_sha256", "dimension", "embedding_space_id", "post_pool_transform", "research_only", "topology", "transform", "anchor", "legal_scope", "sidecar", "sibling_rollup"},
        f"{label}.attestation",
    )
    role = require_string(policy["role"], f"{label}.attestation.role", safe_id=True)
    if role not in ROLES or (expected_role is not None and role != expected_role):
        raise GateContractError(f"{label}: package attestation role mismatch")
    if policy["schema"] != PACKAGE_ATTESTATION_SCHEMA or policy["dimension"] != DIMENSION or policy["research_only"] is not (role == "candidate"):
        raise GateContractError(f"{label}: package attestation schema/dimension mismatch")
    if policy["package_sha256"] != package["sha256"] or policy["manifest_sha256"] != package["manifest"]["sha256"]:
        raise GateContractError(f"{label}: package attestation hash binding mismatch")
    embedding_space_id = require_string(policy["embedding_space_id"], f"{label}.attestation.embedding_space_id", safe_id=True)
    if policy["topology"] != AOQT_TOPOLOGY or policy["legal_scope"] != AOQT_LEGAL_SCOPE:
        raise GateContractError(f"{label}: AOQT topology/legal scope mismatch")
    anchor = require_mapping(policy["anchor"], f"{label}.attestation.anchor")
    require_exact_keys(anchor, {"package_sha256", "manifest_sha256"}, f"{label}.attestation.anchor")
    anchor_binding = {
        "package_sha256": require_sha256(anchor["package_sha256"], f"{label}.attestation.anchor.package_sha256"),
        "manifest_sha256": require_sha256(anchor["manifest_sha256"], f"{label}.attestation.anchor.manifest_sha256"),
    }
    manifest_path = Path(package["manifest"]["path"])
    if manifest_path.read_bytes()[:4] == b"MLL\0":
        manifest_policy = None
        native_manifest = parse_runtime_package_manifest(manifest_path, Path(package["path"]), f"{label}.manifest")
    else:
        native_manifest = None
        manifest_policy = read_json(manifest_path, f"{label}.manifest")
        require_exact_keys(
            manifest_policy,
            {"schema", "role", "package_sha256", "dimension", "embedding_space_id", "post_pool_transform", "research_only", "topology", "transform", "anchor", "legal_scope", "sidecar_sha256"},
            f"{label}.manifest",
        )
        if manifest_policy["schema"] != PACKAGE_MANIFEST_SCHEMA or manifest_policy["role"] != role or manifest_policy["package_sha256"] != package["sha256"] or manifest_policy["dimension"] != DIMENSION or manifest_policy["embedding_space_id"] != embedding_space_id or manifest_policy["research_only"] is not (role == "candidate") or manifest_policy["topology"] != AOQT_TOPOLOGY or manifest_policy["legal_scope"] != AOQT_LEGAL_SCOPE:
            raise GateContractError(f"{label}: package manifest identity/policy mismatch")
        expected_manifest_anchor = None if role == "anchor" else anchor_binding
        if manifest_policy["post_pool_transform"] != policy["post_pool_transform"] or manifest_policy["transform"] != policy["transform"] or manifest_policy["anchor"] != expected_manifest_anchor:
            raise GateContractError(f"{label}: package manifest transform/anchor mismatch")
    manifest_sidecar_sha = manifest_policy["sidecar_sha256"] if manifest_policy is not None else None
    if role == "anchor" and manifest_policy is not None and manifest_sidecar_sha is not None:
        raise GateContractError(f"{label}: anchor package manifest must not declare a sidecar")
    if role == "candidate" and manifest_policy is not None:
        require_sha256(manifest_sidecar_sha, f"{label}.manifest.sidecar_sha256")
    if role == "anchor":
        if policy["post_pool_transform"] != "none" or policy["transform"] is not None or policy["sidecar"] is not None:
            raise GateContractError(f"{label}: anchor must not declare an AOQT sidecar")
        if anchor_binding != {"package_sha256": package["sha256"], "manifest_sha256": package["manifest"]["sha256"]}:
            raise GateContractError(f"{label}: anchor self-identity binding mismatch")
    else:
        if policy["post_pool_transform"] != "aoqt_givens_v1":
            raise GateContractError(f"{label}: candidate must declare AOQT post_pool_transform")
        transform = require_mapping(policy["transform"], f"{label}.attestation.transform")
        require_exact_keys(transform, {"id", "dim", "stages", "pairs_per_stage", "angle_count", "pairings_sha256", "angles_sha256", "angle_cap", "max_angle_cap"}, f"{label}.attestation.transform")
        if transform["id"] != AOQT_TOPOLOGY["id"] or transform["dim"] != DIMENSION or transform["stages"] != AOQT_TOPOLOGY["stages"] or transform["pairs_per_stage"] != AOQT_TOPOLOGY["pairs_per_stage"] or transform["angle_count"] != AOQT_TOPOLOGY["angle_count"]:
            raise GateContractError(f"{label}: candidate transform topology mismatch")
        require_sha256(transform["pairings_sha256"], f"{label}.attestation.transform.pairings_sha256")
        require_sha256(transform["angles_sha256"], f"{label}.attestation.transform.angles_sha256")
        cap = require_finite_number(transform["angle_cap"], f"{label}.attestation.transform.angle_cap")
        hard_cap = require_finite_number(transform["max_angle_cap"], f"{label}.attestation.transform.max_angle_cap")
        if cap < 0 or cap > AOQT_TOPOLOGY["angle_cap_default"] or hard_cap != AOQT_TOPOLOGY["angle_cap_hard"] or cap > hard_cap:
            raise GateContractError(f"{label}: candidate angle cap policy mismatch")
        sidecar_record = normalize_file_record(policy["sidecar"], attestation_path.parent, f"{label}.attestation.sidecar")
        sidecar_policy = validate_sidecar(Path(sidecar_record["path"]), expected_anchor=anchor_binding, label=f"{label}.sidecar")
        if sidecar_policy["anchor"] != anchor_binding:
            raise GateContractError(f"{label}: sidecar/attestation anchor mismatch")
        if transform["pairings_sha256"] != sidecar_policy["pairings_sha256"] or transform["angles_sha256"] != sidecar_policy["angles_sha256"]:
            raise GateContractError(f"{label}: transform/sidecar hash mismatch")
        if manifest_policy is not None and manifest_sidecar_sha != sidecar_record["sha256"]:
            raise GateContractError(f"{label}: package manifest sidecar hash mismatch")
        package["sidecar"] = sidecar_record
        package["sidecar_policy"] = sidecar_policy
    sibling = require_mapping(policy["sibling_rollup"], f"{label}.attestation.sibling_rollup")
    require_exact_keys(sibling, {"entries", "sha256"}, f"{label}.attestation.sibling_rollup")
    entries = normalize_sibling_entries(sibling["entries"], attestation_path.parent, f"{label}.attestation.sibling_rollup.entries")
    if sibling["sha256"] != sha256_json(entries) or package["sibling_rollup_sha256"] != sibling["sha256"]:
        raise GateContractError(f"{label}: sibling rollup digest mismatch")
    entry_names = {item["name"] for item in entries}
    required_names = {Path(package["path"]).name, Path(package["manifest"]["path"]).name}
    if role == "candidate":
        required_names.add(Path(package["sidecar"]["path"]).name)
    if not required_names.issubset(entry_names):
        raise GateContractError(f"{label}: sibling rollup omits bound package files")
    if native_manifest is not None:
        native_names = {Path(item["path"]).name for item in native_manifest["files"]}
        if not native_names.issubset(entry_names):
            raise GateContractError(f"{label}: sibling rollup omits runtime package files")
    if native_manifest is not None:
        validate_runtime_package_policy(native_manifest, package=package, role=role, policy=policy, anchor_binding=anchor_binding, embedding_space_id=embedding_space_id, sidecar_record=package.get("sidecar"), sidecar_policy=package.get("sidecar_policy"), label=label)
    package["embedding_space_id"] = embedding_space_id
    package["attestation_policy"] = {
        "schema": PACKAGE_ATTESTATION_SCHEMA,
        "role": role,
        "package_sha256": package["sha256"],
        "manifest_sha256": package["manifest"]["sha256"],
        "dimension": DIMENSION,
        "embedding_space_id": embedding_space_id,
        "post_pool_transform": policy["post_pool_transform"],
        "research_only": policy["research_only"],
        "topology": copy.deepcopy(AOQT_TOPOLOGY),
        "transform": copy.deepcopy(policy["transform"]),
        "anchor": anchor_binding,
        "legal_scope": copy.deepcopy(AOQT_LEGAL_SCOPE),
        "sidecar": package.get("sidecar"),
        "sibling_rollup": {"entries": entries, "sha256": sibling["sha256"]},
    }
    if "embedding_space_id" in record and record["embedding_space_id"] != package["embedding_space_id"]:
        raise GateContractError(f"{label}: embedding_space_id canonical binding mismatch")
    if "sidecar" in record and record["sidecar"] != package.get("sidecar"):
        raise GateContractError(f"{label}: sidecar canonical binding mismatch")
    if "sidecar_policy" in record and record["sidecar_policy"] != package.get("sidecar_policy"):
        raise GateContractError(f"{label}: sidecar policy canonical binding mismatch")
    if "attestation_policy" in record and record["attestation_policy"] != package["attestation_policy"]:
        raise GateContractError(f"{label}: package attestation policy canonical binding mismatch")
    return package


def validate_package_pair(anchor: dict[str, Any], candidate: dict[str, Any]) -> None:
    if anchor["sha256"] == candidate["sha256"] or anchor["manifest"]["sha256"] == candidate["manifest"]["sha256"] or anchor["attestation"]["sha256"] == candidate["attestation"]["sha256"]:
        raise GateContractError("anchor and candidate package identities must be distinct, not merely different paths")
    if anchor["embedding_space_id"] != candidate["embedding_space_id"]:
        raise GateContractError("anchor/candidate embedding_space_id mismatch")
    candidate_anchor = candidate["attestation_policy"]["anchor"]
    if candidate_anchor != {"package_sha256": anchor["sha256"], "manifest_sha256": anchor["manifest"]["sha256"]}:
        raise GateContractError("candidate AOQT sidecar anchor binding mismatch")


def validate_approved_workload(path: Path, *, gate_id: str, qrels: dict[str, dict[str, Any]], label: str) -> dict[str, Any]:
    payload = read_json(path, label)
    require_exact_keys(
        payload,
        {"schema", "workload_id", "gate_id", "split", "dimension", "domains", "query_ids_by_domain", "qid_set_sha256_by_domain", "query_count_by_domain", "qrels_by_domain", "nfcorpus_boundary_qids", "nfcorpus_boundary_qids_sha256", "nfcorpus_boundary_rank_window", "metric_surfaces", "cutoffs", "turboquant", "expected_result_artifact_count"},
        label,
    )
    require_no_forbidden_payload_keys(payload, label)
    if payload["schema"] != APPROVED_WORKLOAD_SCHEMA or payload["gate_id"] != gate_id or payload["split"] != "heldout" or payload["dimension"] != DIMENSION or payload["domains"] != list(DOMAINS):
        raise GateContractError(f"{label}: approved workload identity mismatch")
    workload_id = require_string(payload["workload_id"], f"{label}.workload_id", safe_id=True)
    qids = normalize_qids_by_domain(payload["query_ids_by_domain"], f"{label}.query_ids_by_domain", require_nonempty=True)
    qid_hashes = require_mapping(payload["qid_set_sha256_by_domain"], f"{label}.qid_set_sha256_by_domain")
    require_exact_keys(qid_hashes, set(DOMAINS), f"{label}.qid_set_sha256_by_domain")
    for domain in DOMAINS:
        if require_sha256(qid_hashes[domain], f"{label}.qid_set_sha256_by_domain.{domain}") != sha256_json(qids[domain]):
            raise GateContractError(f"{label}: qid-set hash mismatch for {domain}")
    counts = require_mapping(payload["query_count_by_domain"], f"{label}.query_count_by_domain")
    require_exact_keys(counts, set(DOMAINS), f"{label}.query_count_by_domain")
    count_by_domain = {domain: require_integer(counts[domain], f"{label}.query_count_by_domain.{domain}", minimum=1) for domain in DOMAINS}
    if count_by_domain != {domain: len(qids[domain]) for domain in DOMAINS}:
        raise GateContractError(f"{label}: approved workload count mismatch")
    qrels_node = require_mapping(payload["qrels_by_domain"], f"{label}.qrels_by_domain")
    require_exact_keys(qrels_node, set(DOMAINS), f"{label}.qrels_by_domain")
    qrels_binding: dict[str, Any] = {}
    for domain in DOMAINS:
        item = require_mapping(qrels_node[domain], f"{label}.qrels_by_domain.{domain}")
        require_exact_keys(item, {"path", "sha256", "bytes", "query_count", "relevant_count"}, f"{label}.qrels_by_domain.{domain}")
        normalized = normalize_qrels_record(item, path.parent, f"{label}.qrels_by_domain.{domain}")
        if normalized != qrels[domain]:
            raise GateContractError(f"{label}: qrels binding mismatch for {domain}")
        if normalized["query_count"] != count_by_domain[domain]:
            raise GateContractError(f"{label}: qrels/query count mismatch for {domain}")
        qrels_binding[domain] = normalized
    if not isinstance(payload["nfcorpus_boundary_qids"], list):
        raise GateContractError(f"{label}.nfcorpus_boundary_qids: expected qid list")
    boundary = [require_qid(item, f"{label}.nfcorpus_boundary_qids") for item in payload["nfcorpus_boundary_qids"]]
    if not boundary or boundary != sorted(set(boundary)) or not set(boundary).issubset(set(qids["nfcorpus"])):
        raise GateContractError(f"{label}: NFCorpus boundary qids must be a non-empty sorted subset")
    if require_sha256(payload["nfcorpus_boundary_qids_sha256"], f"{label}.nfcorpus_boundary_qids_sha256") != sha256_json(boundary):
        raise GateContractError(f"{label}: NFCorpus boundary qid hash mismatch")
    if payload["nfcorpus_boundary_rank_window"] != [80, 120]:
        raise GateContractError(f"{label}: NFCorpus boundary window must be [80,120]")
    if payload["metric_surfaces"] != list(SURFACES) or payload["cutoffs"] != {"ndcg_at_10": 10, "recall_at_100": 100}:
        raise GateContractError(f"{label}: metric surface/cutoff mismatch")
    turboquant = require_mapping(payload["turboquant"], f"{label}.turboquant")
    require_exact_keys(turboquant, {"q3_bits", "q5_bits", "seed", "top_k"}, f"{label}.turboquant")
    if turboquant != {"q3_bits": Q3_BITS, "q5_bits": Q5_BITS, "seed": TURBOQUANT_SEED, "top_k": TOP_K}:
        raise GateContractError(f"{label}: TurboQuant config mismatch")
    if payload["expected_result_artifact_count"] != RESULT_OUTPUT_COUNT:
        raise GateContractError(f"{label}: expected result artifact count mismatch")
    return {
        "schema": APPROVED_WORKLOAD_SCHEMA,
        "workload_id": workload_id,
        "gate_id": gate_id,
        "split": "heldout",
        "dimension": DIMENSION,
        "domains": list(DOMAINS),
        "query_ids_by_domain": qids,
        "qid_set_sha256_by_domain": qids_sha256_by_domain(qids),
        "query_count_by_domain": count_by_domain,
        "qrels_by_domain": qrels_binding,
        "nfcorpus_boundary_qids": boundary,
        "nfcorpus_boundary_qids_sha256": sha256_json(boundary),
        "nfcorpus_boundary_rank_window": [80, 120],
        "metric_surfaces": list(SURFACES),
        "cutoffs": {"ndcg_at_10": 10, "recall_at_100": 100},
        "turboquant": {"q3_bits": Q3_BITS, "q5_bits": Q5_BITS, "seed": TURBOQUANT_SEED, "top_k": TOP_K},
        "expected_result_artifact_count": RESULT_OUTPUT_COUNT,
    }


def validate_workload(path: Path, *, gate_id: str, approved: dict[str, Any], label: str) -> dict[str, Any]:
    payload = read_json(path, label)
    required = {"schema", "workload_id", "gate_id", "split", "dimension", "domains", "query_ids_by_domain", "qid_set_sha256_by_domain", "query_count_by_domain", "nfcorpus_boundary_qids", "nfcorpus_boundary_qids_sha256", "nfcorpus_boundary_rank_window", "metric_surfaces", "cutoffs", "turboquant", "expected_result_artifact_count"}
    require_exact_keys(payload, required, label)
    require_no_forbidden_payload_keys(payload, label)
    if payload["schema"] != WORKLOAD_SCHEMA:
        raise GateContractError(f"{label}: workload schema mismatch")
    if payload["workload_id"] != approved["workload_id"] or payload["gate_id"] != gate_id or payload["split"] != "heldout" or payload["dimension"] != DIMENSION or payload["domains"] != list(DOMAINS):
        raise GateContractError(f"{label}: evaluator workload identity mismatch")
    qids = normalize_qids_by_domain(payload["query_ids_by_domain"], f"{label}.query_ids_by_domain", require_nonempty=True)
    qid_hashes = require_mapping(payload["qid_set_sha256_by_domain"], f"{label}.qid_set_sha256_by_domain")
    require_exact_keys(qid_hashes, set(DOMAINS), f"{label}.qid_set_sha256_by_domain")
    counts = require_mapping(payload["query_count_by_domain"], f"{label}.query_count_by_domain")
    require_exact_keys(counts, set(DOMAINS), f"{label}.query_count_by_domain")
    normalized_counts = {domain: require_integer(counts[domain], f"{label}.query_count_by_domain.{domain}", minimum=1) for domain in DOMAINS}
    if not isinstance(payload["nfcorpus_boundary_qids"], list):
        raise GateContractError(f"{label}.nfcorpus_boundary_qids: expected qid list")
    boundary = [require_qid(item, f"{label}.nfcorpus_boundary_qids") for item in payload["nfcorpus_boundary_qids"]]
    normalized = {
        "schema": WORKLOAD_SCHEMA,
        "workload_id": payload["workload_id"],
        "gate_id": gate_id,
        "split": "heldout",
        "dimension": DIMENSION,
        "domains": list(DOMAINS),
        "query_ids_by_domain": qids,
        "qid_set_sha256_by_domain": qids_sha256_by_domain(qids),
        "query_count_by_domain": normalized_counts,
        "nfcorpus_boundary_qids": sorted(set(boundary)),
        "nfcorpus_boundary_qids_sha256": sha256_json(sorted(set(boundary))),
        "nfcorpus_boundary_rank_window": payload["nfcorpus_boundary_rank_window"],
        "metric_surfaces": payload["metric_surfaces"],
        "cutoffs": payload["cutoffs"],
        "turboquant": payload["turboquant"],
        "expected_result_artifact_count": payload["expected_result_artifact_count"],
    }
    if normalized["qid_set_sha256_by_domain"] != qid_hashes or normalized["query_count_by_domain"] != {domain: len(qids[domain]) for domain in DOMAINS}:
        raise GateContractError(f"{label}: workload qid hash/count mismatch")
    if normalized["nfcorpus_boundary_rank_window"] != [80, 120] or not normalized["nfcorpus_boundary_qids"] or not set(normalized["nfcorpus_boundary_qids"]).issubset(set(qids["nfcorpus"])):
        raise GateContractError(f"{label}: workload NFCorpus boundary mismatch")
    if normalized["metric_surfaces"] != list(SURFACES) or normalized["cutoffs"] != {"ndcg_at_10": 10, "recall_at_100": 100} or normalized["turboquant"] != {"q3_bits": Q3_BITS, "q5_bits": Q5_BITS, "seed": TURBOQUANT_SEED, "top_k": TOP_K} or normalized["expected_result_artifact_count"] != RESULT_OUTPUT_COUNT:
        raise GateContractError(f"{label}: workload metric/config mismatch")
    approved_projection = {key: approved[key] for key in normalized if key in approved}
    for key in ("workload_id", "gate_id", "dimension", "domains", "query_ids_by_domain", "qid_set_sha256_by_domain", "query_count_by_domain", "nfcorpus_boundary_qids", "nfcorpus_boundary_qids_sha256", "nfcorpus_boundary_rank_window", "metric_surfaces", "cutoffs", "turboquant", "expected_result_artifact_count"):
        if normalized[key] != approved_projection.get(key):
            raise GateContractError(f"{label}: approved workload binding mismatch at {key}")
    return normalized


def normalize_exclusion_record(value: Any, base_dir: Path, label: str) -> dict[str, Any]:
    record = require_mapping(value, label)
    require_exact_keys(record, {"kind", "name", "path", "sha256", "bytes"}, label)
    if record["kind"] != "qid_only" or record["name"] not in EXCLUSION_NAMES:
        raise GateContractError(f"{label}: exact qid-only exclusion identity required")
    file_record = normalize_file_record({key: record[key] for key in ("path", "sha256", "bytes")}, base_dir, label)
    payload = read_json(Path(file_record["path"]), f"{label}.payload")
    require_exact_keys(payload, {"schema", "name", "split", "qids_by_domain", "source_sha256", "source_sha256_by_file"}, f"{label}.payload")
    require_no_forbidden_payload_keys(payload, f"{label}.payload")
    if payload["schema"] != "eos.aoqt.heldout_exclusion.v1" or payload["name"] != record["name"] or payload["split"] != record["name"]:
        raise GateContractError(f"{label}: exclusion schema/name/split mismatch")
    qids = normalize_qids_by_domain(payload["qids_by_domain"], f"{label}.qids_by_domain", require_nonempty=True)
    source_sha = require_sha256(payload["source_sha256"], f"{label}.source_sha256")
    source_map = require_mapping(payload["source_sha256_by_file"], f"{label}.source_sha256_by_file")
    if not source_map:
        raise GateContractError(f"{label}.source_sha256_by_file must be non-empty")
    normalized_source_map = {}
    for source_name, source_hash in source_map.items():
        normalized_source_map[require_string(source_name, f"{label}.source_sha256_by_file key")] = require_sha256(source_hash, f"{label}.source_sha256_by_file.{source_name}")
    return {
        **file_record,
        "kind": "qid_only",
        "name": record["name"],
        "semantic": "qid_only_exclusion",
        "contains_metric_payload": False,
        "schema": payload["schema"],
        "split": payload["split"],
        "qids_by_domain": qids,
        "qids_sha256_by_domain": qids_sha256_by_domain(qids),
        "source_sha256": source_sha,
        "source_sha256_by_file": dict(sorted(normalized_source_map.items())),
    }


def validate_argv(argv: Any, *, binary: Path, package: Path, base_dir: Path, label: str) -> list[str]:
    if not isinstance(argv, list) or not argv or any(not isinstance(part, str) or not part for part in argv):
        raise GateContractError(f"{label}: argv must be a non-empty string list")
    for index, part in enumerate(argv):
        if any(marker in part for marker in SHELL_MARKERS) or any(ord(char) < 0x20 for char in part):
            raise GateContractError(f"{label}[{index}]: shell/control token is forbidden")
    if len(argv) < 2 or argv[1] != NATIVE_EVAL_SUBCOMMAND or NATIVE_EVAL_SUBCOMMAND in argv[2:]:
        raise GateContractError(f"{label}: native evaluator subcommand/order mismatch")

    def matches(part: str, target: Path) -> bool:
        try:
            path = Path(part)
            if not path.is_absolute():
                path = base_dir / path
            return path.resolve(strict=False) == target
        except OSError:
            return False

    binary_matches = [index for index, part in enumerate(argv) if matches(part, binary)]
    package_matches = [index for index, part in enumerate(argv) if matches(part, package)]
    target_strings = ((str(binary), binary_matches), (str(package), package_matches))
    for target_string, matches_for_target in target_strings:
        if any(target_string in part and index not in matches_for_target for index, part in enumerate(argv)):
            raise GateContractError(f"{label}: executable/package path mention-count impostor")
    if binary_matches != [0]:
        raise GateContractError(f"{label}: argv[0] must be the bound native executable (wrappers rejected)")
    if len(package_matches) != 1:
        raise GateContractError(f"{label}: package must occur exactly once in argv")
    return list(argv)


def normalize_output_paths(value: Any, base_dir: Path, output_root: Path, label: str) -> list[dict[str, Any]]:
    mapping = require_mapping(value, label)
    unexpected = sorted(set(mapping) - set(DOMAINS))
    missing = [domain for domain in DOMAINS if domain not in mapping]
    if unexpected or missing:
        raise GateContractError(f"{label}: exact domain output coverage required")
    records: list[dict[str, Any]] = []
    for domain in DOMAINS:
        item = require_mapping(mapping[domain], f"{label}.{domain}")
        require_exact_keys(item, {"path", "receipt"}, f"{label}.{domain}")
        metric_path = resolve_path(item["path"], base_dir, f"{label}.{domain}.path")
        receipt_node = require_mapping(item["receipt"], f"{label}.{domain}.receipt")
        require_exact_keys(receipt_node, {"path"}, f"{label}.{domain}.receipt")
        receipt_path = resolve_path(receipt_node["path"], base_dir, f"{label}.{domain}.receipt.path")
        for kind, path in (("metrics", metric_path), ("receipt", receipt_path)):
            path_within(path, output_root, f"{label}.{domain}.{kind}")
            if path == output_root or path.exists() or path.is_symlink():
                raise GateContractError(f"{label}.{domain}.{kind}: expected output must be absent before evaluation")
            records.append({"kind": kind, "role": label.rsplit(".", 1)[-1], "domain": domain, "path": str(path), "state": "must_be_absent"})
    return records


def validate_plan(plan_path: Path) -> dict[str, Any]:
    plan_path = plan_path.resolve(strict=False)
    plan = read_json(plan_path, "plan")
    require_exact_keys(plan, {"schema", "gate_id", "candidate_id", "dimension", "thresholds", "evaluation", "anchor", "candidate", "source", "approved_workload", "qrels", "exclusions", "workload_root", "output_root", "artifacts", "attestation_path"}, "plan")
    if plan["schema"] != PLAN_SCHEMA:
        raise GateContractError("plan: schema mismatch")
    gate_id = require_string(plan["gate_id"], "plan.gate_id", safe_id=True)
    candidate_id = require_string(plan["candidate_id"], "plan.candidate_id", safe_id=True)
    if plan["dimension"] != DIMENSION or plan["thresholds"] != THRESHOLDS:
        raise GateContractError("plan: immutable D384/quality policy mismatch")
    if plan["evaluation"] != {"executed": False, "official": False}:
        raise GateContractError("plan: evaluation must be unexecuted and non-official")
    anchor_node = require_mapping(plan["anchor"], "plan.anchor")
    candidate_node = require_mapping(plan["candidate"], "plan.candidate")
    require_exact_keys(anchor_node, {"id", "package"}, "plan.anchor")
    require_exact_keys(candidate_node, {"id", "package"}, "plan.candidate")
    anchor_id = require_string(anchor_node["id"], "plan.anchor.id", safe_id=True)
    if candidate_node["id"] != candidate_id or anchor_id == candidate_id:
        raise GateContractError("plan: anchor/candidate identity mismatch")
    anchor_package = normalize_package_record(anchor_node["package"], plan_path.parent, "plan.anchor.package", expected_role="anchor")
    candidate_package = normalize_package_record(candidate_node["package"], plan_path.parent, "plan.candidate.package", expected_role="candidate")
    validate_package_pair(anchor_package, candidate_package)

    source = require_mapping(plan["source"], "plan.source")
    require_exact_keys(source, {"manifest", "binary", "cwd", "argv", "dataset", "workload", "approved_workload"}, "plan.source")
    source_manifest = normalize_file_record(source["manifest"], plan_path.parent, "plan.source.manifest")
    binary = normalize_file_record(source["binary"], plan_path.parent, "plan.source.binary")
    dataset = normalize_file_record(source["dataset"], plan_path.parent, "plan.source.dataset")
    cwd = normalize_directory(source["cwd"], plan_path.parent, "plan.source.cwd")
    argv_node = require_mapping(source["argv"], "plan.source.argv")
    require_exact_keys(argv_node, set(ROLES), "plan.source.argv")
    argv = {
        role: validate_argv(argv_node[role], binary=Path(binary["path"]), package=Path((anchor_package if role == "anchor" else candidate_package)["path"]), base_dir=plan_path.parent, label=f"plan.source.argv.{role}")
        for role in ROLES
    }
    qrels_node = require_mapping(plan["qrels"], "plan.qrels")
    require_exact_keys(qrels_node, set(DOMAINS), "plan.qrels")
    qrels = {domain: normalize_qrels_record(qrels_node[domain], plan_path.parent, f"plan.qrels.{domain}") for domain in DOMAINS}
    approved_record = normalize_file_record(plan["approved_workload"], plan_path.parent, "plan.approved_workload")
    approved = validate_approved_workload(Path(approved_record["path"]), gate_id=gate_id, qrels=qrels, label="plan.approved_workload")
    workload_record = normalize_file_record(source["workload"], plan_path.parent, "plan.source.workload")
    source_approved_record = normalize_file_record(source["approved_workload"], plan_path.parent, "plan.source.approved_workload")
    if source_approved_record != approved_record:
        raise GateContractError("plan: source/approved workload record mismatch")
    workload = validate_workload(Path(workload_record["path"]), gate_id=gate_id, approved=approved, label="plan.source.workload")

    raw_exclusions = plan["exclusions"]
    if not isinstance(raw_exclusions, list) or len(raw_exclusions) != len(EXCLUSION_NAMES):
        raise GateContractError(f"plan.exclusions: exactly {len(EXCLUSION_NAMES)} domain-scoped exclusions are required")
    exclusions = [normalize_exclusion_record(item, plan_path.parent, f"plan.exclusions[{index}]") for index, item in enumerate(raw_exclusions)]
    if {item["name"] for item in exclusions} != set(EXCLUSION_NAMES):
        raise GateContractError("plan.exclusions: dev4/reserve4/official-test are all required")
    excluded_by_domain = {domain: set() for domain in DOMAINS}
    for item in exclusions:
        for domain, qids in item["qids_by_domain"].items():
            overlap = excluded_by_domain[domain].intersection(qids)
            if overlap:
                raise GateContractError(f"plan.exclusions: duplicate qids in {domain}: {sorted(overlap)}")
            workload_overlap = set(approved["query_ids_by_domain"][domain]).intersection(qids)
            if workload_overlap:
                raise GateContractError(f"plan.exclusions: workload/exclusion qid intersection in {domain}: {sorted(workload_overlap)}")
            excluded_by_domain[domain].update(qids)

    workload_root = resolve_path(plan["workload_root"], plan_path.parent, "plan.workload_root")
    require_directory(workload_root, "plan.workload_root")
    path_within(Path(workload_record["path"]), workload_root, "plan.source.workload")
    path_within(Path(approved_record["path"]), workload_root, "plan.approved_workload")
    output_root = resolve_path(plan["output_root"], plan_path.parent, "plan.output_root")
    require_directory(output_root, "plan.output_root")
    artifacts_node = require_mapping(plan["artifacts"], "plan.artifacts")
    require_exact_keys(artifacts_node, set(ROLES), "plan.artifacts")
    outputs: list[dict[str, Any]] = []
    for role in ROLES:
        role_outputs = normalize_output_paths(artifacts_node[role], plan_path.parent, output_root, f"plan.artifacts.{role}")
        for record in role_outputs:
            record["role"] = role
        outputs.extend(role_outputs)
    if len(outputs) != EXPECTED_OUTPUT_COUNT or len({item["path"] for item in outputs}) != EXPECTED_OUTPUT_COUNT:
        raise GateContractError("plan.artifacts: exactly twelve unique absent outputs are required")
    attestation_path = resolve_path(plan["attestation_path"], plan_path.parent, "plan.attestation_path")
    if attestation_path.exists() or attestation_path.is_symlink() or attestation_path in {Path(item["path"]) for item in outputs}:
        raise GateContractError("plan.attestation_path must be absent and distinct from retrieval outputs")
    all_input_paths = {
        Path(anchor_package["path"]), Path(candidate_package["path"]), Path(anchor_package["manifest"]["path"]), Path(candidate_package["manifest"]["path"]), Path(anchor_package["attestation"]["path"]), Path(candidate_package["attestation"]["path"]), Path(source_manifest["path"]), Path(binary["path"]), Path(dataset["path"]), Path(workload_record["path"]), Path(approved_record["path"]), *[Path(item["path"]) for item in qrels.values()], *[Path(item["path"]) for item in exclusions],
    }
    if len(all_input_paths) != 11 + len(DOMAINS) + len(exclusions):
        raise GateContractError("plan source/package paths must be unique")
    if any(Path(item["path"]) in all_input_paths for item in outputs) or attestation_path in all_input_paths:
        raise GateContractError("plan output path collides with an input")
    return {
        "schema": PLAN_SCHEMA, "gate_id": gate_id, "candidate_id": candidate_id, "anchor_id": anchor_id, "dimension": DIMENSION, "thresholds": copy.deepcopy(THRESHOLDS), "evaluation": {"executed": False, "official": False},
        "anchor": {"id": anchor_id, "package": anchor_package}, "candidate": {"id": candidate_id, "package": candidate_package},
        "source": {"manifest": source_manifest, "binary": binary, "cwd": cwd, "argv": argv, "dataset": dataset, "workload": workload_record, "approved_workload": approved_record},
        "approved_workload": {"record": approved_record, "payload": approved}, "workload": workload, "qrels": qrels, "exclusions": exclusions,
        "workload_root": str(workload_root), "output_root": str(output_root), "outputs": outputs, "attestation_path": str(attestation_path), "plan_path": str(plan_path), "plan_sha256": sha256_file(plan_path),
    }


def digest_without(payload: dict[str, Any], field: str) -> str:
    return sha256_json({key: value for key, value in payload.items() if key != field})


def freeze_manifest(plan_path: Path, output_path: Path) -> dict[str, Any]:
    plan = validate_plan(plan_path)
    output_path = output_path.resolve(strict=False)
    if output_path.exists() or output_path.is_symlink():
        raise GateContractError(f"frozen manifest must be absent: {output_path}")
    if output_path in {Path(item["path"]) for item in plan["outputs"]} or output_path == Path(plan["attestation_path"]):
        raise GateContractError("frozen manifest path collides with a declared output")
    try:
        output_path.relative_to(Path(plan["output_root"]))
    except ValueError:
        pass
    else:
        raise GateContractError("frozen manifest must be outside retrieval output_root")
    gate_script = Path(__file__).resolve()
    frozen = {
        "schema": FROZEN_SCHEMA, "gate_id": plan["gate_id"], "candidate_id": plan["candidate_id"], "anchor_id": plan["anchor_id"], "dimension": DIMENSION, "thresholds": copy.deepcopy(THRESHOLDS), "evaluation": {"executed": False, "official": False},
        "external_trust_anchor": {"required": True, "algorithm": "sha256_file"}, "anchor": plan["anchor"], "candidate": plan["candidate"], "source": plan["source"], "approved_workload": plan["approved_workload"], "workload": plan["workload"], "qrels": plan["qrels"], "exclusions": plan["exclusions"], "output_root": plan["output_root"], "outputs": plan["outputs"], "attestation": {"path": plan["attestation_path"], "state": "must_be_absent"},
        "provenance": {"plan": {"path": plan["plan_path"], "sha256": plan["plan_sha256"], "bytes": Path(plan["plan_path"]).stat().st_size}, "gate_script": {"path": str(gate_script), "sha256": sha256_file(gate_script), "bytes": gate_script.stat().st_size}, "argv_sha256": {role: sha256_json(plan["source"]["argv"][role]) for role in ROLES}, "approved_workload_sha256": plan["approved_workload"]["record"]["sha256"], "output_set_sha256": sha256_json(plan["outputs"])},
    }
    frozen["manifest_sha256"] = digest_without(frozen, "manifest_sha256")
    write_json_new(output_path, frozen)
    frozen["file_sha256"] = sha256_file(output_path)
    return frozen


def validate_frozen_manifest(frozen_path: Path, *, expected_frozen_manifest_sha256: str | None, require_outputs_absent: bool) -> dict[str, Any]:
    if expected_frozen_manifest_sha256 is None:
        raise GateContractError("external expected frozen manifest sha256 is required")
    expected_external = require_sha256(expected_frozen_manifest_sha256, "expected_frozen_manifest_sha256")
    frozen_path = frozen_path.resolve(strict=False)
    require_regular_file(frozen_path, "frozen manifest")
    actual_external = sha256_file(frozen_path)
    if actual_external != expected_external:
        raise GateContractError("external frozen manifest sha256 trust-anchor mismatch")
    frozen = read_json(frozen_path, "frozen manifest")
    require_exact_keys(frozen, {"schema", "gate_id", "candidate_id", "anchor_id", "dimension", "thresholds", "evaluation", "external_trust_anchor", "anchor", "candidate", "source", "approved_workload", "workload", "qrels", "exclusions", "output_root", "outputs", "attestation", "provenance", "manifest_sha256"}, "frozen manifest")
    if frozen["schema"] != FROZEN_SCHEMA or digest_without(frozen, "manifest_sha256") != require_sha256(frozen["manifest_sha256"], "frozen manifest.manifest_sha256"):
        raise GateContractError("frozen manifest self-digest mismatch")
    if frozen["external_trust_anchor"] != {"required": True, "algorithm": "sha256_file"} or frozen["dimension"] != DIMENSION or frozen["thresholds"] != THRESHOLDS or frozen["evaluation"] != {"executed": False, "official": False}:
        raise GateContractError("frozen manifest immutable policy mismatch")
    gate_id = require_string(frozen["gate_id"], "frozen manifest.gate_id", safe_id=True)
    anchor_node = require_mapping(frozen["anchor"], "frozen manifest.anchor")
    candidate_node = require_mapping(frozen["candidate"], "frozen manifest.candidate")
    require_exact_keys(anchor_node, {"id", "package"}, "frozen manifest.anchor")
    require_exact_keys(candidate_node, {"id", "package"}, "frozen manifest.candidate")
    if anchor_node["id"] != frozen["anchor_id"] or candidate_node["id"] != frozen["candidate_id"] or anchor_node["id"] == candidate_node["id"]:
        raise GateContractError("frozen manifest anchor/candidate identity mismatch")
    anchor_package = normalize_package_record(anchor_node["package"], frozen_path.parent, "frozen manifest.anchor.package", expected_role="anchor")
    candidate_package = normalize_package_record(candidate_node["package"], frozen_path.parent, "frozen manifest.candidate.package", expected_role="candidate")
    if anchor_package != frozen["anchor"]["package"] or candidate_package != frozen["candidate"]["package"]:
        raise GateContractError("frozen manifest package record is not canonical")
    validate_package_pair(anchor_package, candidate_package)
    source = require_mapping(frozen["source"], "frozen manifest.source")
    require_exact_keys(source, {"manifest", "binary", "cwd", "argv", "dataset", "workload", "approved_workload"}, "frozen manifest.source")
    source_manifest = normalize_file_record(source["manifest"], frozen_path.parent, "frozen manifest.source.manifest")
    binary = normalize_file_record(source["binary"], frozen_path.parent, "frozen manifest.source.binary")
    dataset = normalize_file_record(source["dataset"], frozen_path.parent, "frozen manifest.source.dataset")
    cwd = normalize_directory(source["cwd"], frozen_path.parent, "frozen manifest.source.cwd")
    if source_manifest != source["manifest"] or binary != source["binary"] or dataset != source["dataset"] or cwd != source["cwd"]:
        raise GateContractError("frozen manifest source record is not canonical")
    argv_node = require_mapping(source["argv"], "frozen manifest.source.argv")
    require_exact_keys(argv_node, set(ROLES), "frozen manifest.source.argv")
    for role in ROLES:
        validate_argv(argv_node[role], binary=Path(binary["path"]), package=Path((anchor_package if role == "anchor" else candidate_package)["path"]), base_dir=frozen_path.parent, label=f"frozen manifest.source.argv.{role}")
    qrels_node = require_mapping(frozen["qrels"], "frozen manifest.qrels")
    require_exact_keys(qrels_node, set(DOMAINS), "frozen manifest.qrels")
    qrels = {domain: normalize_qrels_record(qrels_node[domain], frozen_path.parent, f"frozen manifest.qrels.{domain}") for domain in DOMAINS}
    if qrels != qrels_node:
        raise GateContractError("frozen manifest qrels record is not canonical")
    approved_record = normalize_file_record(frozen["approved_workload"]["record"], frozen_path.parent, "frozen manifest.approved_workload.record")
    approved = validate_approved_workload(Path(approved_record["path"]), gate_id=gate_id, qrels=qrels, label="frozen manifest.approved_workload")
    if frozen["approved_workload"] != {"record": approved_record, "payload": approved} or source["approved_workload"] != approved_record:
        raise GateContractError("frozen manifest approved workload binding mismatch")
    workload_record = normalize_file_record(source["workload"], frozen_path.parent, "frozen manifest.source.workload")
    workload = validate_workload(Path(workload_record["path"]), gate_id=gate_id, approved=approved, label="frozen manifest.source.workload")
    if isinstance(frozen["workload"], dict) and "record" in frozen["workload"]:
        if workload_record != frozen["workload"]["record"]:
            raise GateContractError("frozen manifest workload record mismatch")
    # The frozen workload is the canonical normalized evaluator payload, not a
    # mutable re-read of arbitrary caller fields.
    if frozen["workload"] != workload:
        raise GateContractError("frozen manifest workload payload drift")
    exclusions_node = frozen["exclusions"]
    if not isinstance(exclusions_node, list) or len(exclusions_node) != len(EXCLUSION_NAMES):
        raise GateContractError("frozen manifest requires exactly three exclusions")
    exclusions = []
    excluded_by_domain = {domain: set() for domain in DOMAINS}
    for index, item in enumerate(exclusions_node):
        item_mapping = require_mapping(item, f"frozen manifest.exclusions[{index}]")
        exclusions.append(normalize_exclusion_record({key: item_mapping[key] for key in ("kind", "name", "path", "sha256", "bytes")}, frozen_path.parent, f"frozen manifest.exclusions[{index}]"))
    if exclusions != exclusions_node or {item["name"] for item in exclusions} != set(EXCLUSION_NAMES):
        raise GateContractError("frozen manifest exclusion binding mismatch")
    for item in exclusions:
        for domain, qids in item["qids_by_domain"].items():
            if excluded_by_domain[domain].intersection(qids) or set(approved["query_ids_by_domain"][domain]).intersection(qids):
                raise GateContractError(f"frozen manifest workload/exclusion qid intersection in {domain}")
            excluded_by_domain[domain].update(qids)
    output_root = resolve_path(frozen["output_root"], frozen_path.parent, "frozen manifest.output_root")
    if str(output_root) != frozen["output_root"]:
        raise GateContractError("frozen manifest output_root is not canonical")
    require_directory(output_root, "frozen manifest.output_root")
    outputs = frozen["outputs"]
    if not isinstance(outputs, list) or len(outputs) != EXPECTED_OUTPUT_COUNT:
        raise GateContractError("frozen manifest requires exactly twelve outputs")
    output_paths: set[Path] = set()
    for index, item in enumerate(outputs):
        record = require_mapping(item, f"frozen manifest.outputs[{index}]")
        require_exact_keys(record, {"kind", "role", "domain", "path", "state"}, f"frozen manifest.outputs[{index}]")
        if record["kind"] not in {"metrics", "receipt"} or record["role"] not in ROLES or record["domain"] not in DOMAINS or record["state"] != "must_be_absent":
            raise GateContractError("frozen manifest output identity/state mismatch")
        path = resolve_path(record["path"], frozen_path.parent, f"frozen manifest.outputs[{index}].path")
        if str(path) != record["path"]:
            raise GateContractError("frozen manifest output path is not canonical")
        path_within(path, output_root, "frozen manifest.output")
        if path in output_paths or (require_outputs_absent and (path.exists() or path.is_symlink())):
            raise GateContractError(f"frozen manifest output is duplicate/present: {path}")
        output_paths.add(path)
    if {(item["kind"], item["role"], item["domain"]) for item in outputs} != {(kind, role, domain) for role in ROLES for domain in DOMAINS for kind in ("metrics", "receipt")}:
        raise GateContractError("frozen manifest output coverage mismatch")
    attestation = require_mapping(frozen["attestation"], "frozen manifest.attestation")
    require_exact_keys(attestation, {"path", "state"}, "frozen manifest.attestation")
    attestation_path = resolve_path(attestation["path"], frozen_path.parent, "frozen manifest.attestation.path")
    if attestation != {"path": str(attestation_path), "state": "must_be_absent"} or attestation_path in output_paths or (require_outputs_absent and (attestation_path.exists() or attestation_path.is_symlink())):
        raise GateContractError("frozen manifest attestation output binding mismatch/presence")
    provenance = require_mapping(frozen["provenance"], "frozen manifest.provenance")
    require_exact_keys(provenance, {"plan", "gate_script", "argv_sha256", "approved_workload_sha256", "output_set_sha256"}, "frozen manifest.provenance")
    plan_record = normalize_file_record(provenance["plan"], frozen_path.parent, "frozen manifest.provenance.plan")
    gate_record = normalize_file_record(provenance["gate_script"], frozen_path.parent, "frozen manifest.provenance.gate_script")
    if plan_record != provenance["plan"] or gate_record != provenance["gate_script"] or Path(gate_record["path"]) != Path(__file__).resolve() or gate_record["sha256"] != sha256_file(Path(__file__).resolve()):
        raise GateContractError("frozen manifest provenance file drift")
    argv_hashes = require_mapping(provenance["argv_sha256"], "frozen manifest.provenance.argv_sha256")
    require_exact_keys(argv_hashes, set(ROLES), "frozen manifest.provenance.argv_sha256")
    if any(argv_hashes[role] != sha256_json(argv_node[role]) for role in ROLES) or provenance["approved_workload_sha256"] != approved_record["sha256"] or provenance["output_set_sha256"] != sha256_json(outputs):
        raise GateContractError("frozen manifest provenance digest mismatch")
    return {"path": str(frozen_path), "file_sha256": actual_external, "expected_external_sha256": expected_external, "manifest": frozen, "anchor": anchor_package, "candidate": candidate_package, "source": source, "qrels": qrels, "approved_workload": approved, "workload": workload, "outputs": outputs, "output_root": output_root, "attestation_path": attestation_path}


def normalize_artifact_source(source: Any, artifact_path: Path, frozen_info: dict[str, Any], role: str, domain: str) -> dict[str, Any]:
    node = require_mapping(source, f"{artifact_path}.source")
    require_exact_keys(node, {"manifest", "qrels", "binary", "cwd", "argv", "dataset", "workload", "approved_workload"}, f"{artifact_path}.source")
    base = artifact_path.parent
    manifest = normalize_file_record(node["manifest"], base, f"{artifact_path}.source.manifest")
    qrels = normalize_qrels_record(node["qrels"], base, f"{artifact_path}.source.qrels")
    binary = normalize_file_record(node["binary"], base, f"{artifact_path}.source.binary")
    dataset = normalize_file_record(node["dataset"], base, f"{artifact_path}.source.dataset")
    cwd = normalize_directory(node["cwd"], base, f"{artifact_path}.source.cwd")
    workload_record = normalize_file_record(node["workload"], base, f"{artifact_path}.source.workload")
    approved_record = normalize_file_record(node["approved_workload"], base, f"{artifact_path}.source.approved_workload")
    expected = frozen_info["source"]
    if manifest != expected["manifest"] or qrels != frozen_info["qrels"][domain] or binary != expected["binary"] or dataset != expected["dataset"] or cwd != expected["cwd"] or workload_record != expected["workload"] or approved_record != expected["approved_workload"] or node["argv"] != expected["argv"][role]:
        raise GateContractError(f"{artifact_path}: source/qrels/argv provenance mismatch")
    validate_argv(node["argv"], binary=Path(binary["path"]), package=Path((frozen_info["anchor"] if role == "anchor" else frozen_info["candidate"])["path"]), base_dir=base, label=f"{artifact_path}.source.argv")
    return {"manifest": manifest, "qrels": qrels, "binary": binary, "cwd": cwd, "argv": list(node["argv"]), "dataset": dataset, "workload": workload_record, "approved_workload": approved_record}


def aggregate_rows(rows: Any, expected_qids: list[str], label: str) -> dict[str, Any]:
    if not isinstance(rows, list) or len(rows) != len(expected_qids):
        raise GateContractError(f"{label}: row count mismatch")
    normalized_rows: list[dict[str, Any]] = []
    seen: list[str] = []
    for index, item in enumerate(rows):
        row = require_mapping(item, f"{label}[{index}]")
        require_exact_keys(row, {"query_id", "metrics"}, f"{label}[{index}]")
        qid = require_qid(row["query_id"], f"{label}[{index}].query_id")
        seen.append(qid)
        metrics_node = require_mapping(row["metrics"], f"{label}[{index}].metrics")
        require_exact_keys(metrics_node, set(SURFACES), f"{label}[{index}].metrics")
        metrics: dict[str, dict[str, float]] = {}
        for surface in SURFACES:
            surface_node = require_mapping(metrics_node[surface], f"{label}[{index}].metrics.{surface}")
            require_exact_keys(surface_node, set(METRICS_BY_SURFACE[surface]), f"{label}[{index}].metrics.{surface}")
            metrics[surface] = {metric: require_finite_number(surface_node[metric], f"{label}[{index}].metrics.{surface}.{metric}", bounded=True) for metric in METRICS_BY_SURFACE[surface]}
        normalized_rows.append({"query_id": qid, "metrics": metrics})
    if seen != expected_qids:
        raise GateContractError(f"{label}: exact approved qid order/set required")
    aggregate = {surface: {metric: math.fsum(row["metrics"][surface][metric] for row in normalized_rows) / len(normalized_rows) for metric in METRICS_BY_SURFACE[surface]} for surface in SURFACES}
    return {"rows": normalized_rows, "rows_by_qid": {row["query_id"]: row for row in normalized_rows}, "aggregate": aggregate, "qids": seen}


def normalize_boundary_evidence(value: Any, *, domain: str, workload: dict[str, Any], label: str) -> dict[str, Any]:
    node = require_mapping(value, label)
    require_exact_keys(node, {"schema", "guard", "rank_window", "qids", "qids_sha256"}, label)
    if domain == "nfcorpus":
        expected_qids = workload["nfcorpus_boundary_qids"]
        expected = {"schema": "eos.aoqt.nfcorpus_boundary_evidence.v1", "guard": "q3_recall_at_100", "rank_window": [80, 120], "qids": expected_qids, "qids_sha256": sha256_json(expected_qids)}
    else:
        expected = {"schema": "eos.aoqt.no_boundary_evidence.v1", "guard": "none", "rank_window": None, "qids": [], "qids_sha256": sha256_json([])}
    if node != expected:
        raise GateContractError(f"{label}: boundary guard evidence mismatch")
    return expected


def validate_receipt(path: Path, *, artifact_path: Path, artifact_sha256: str, frozen_info: dict[str, Any], role: str, domain: str, expected_nonce: str | None = None) -> dict[str, Any]:
    payload = read_json(path, f"{role}/{domain} native receipt")
    require_exact_keys(payload, {"schema", "gate_id", "nonce", "role", "domain", "split", "official", "native_evaluator", "wrapper", "exit_status", "executable", "argv", "argv_sha256", "cwd", "dataset_id", "dataset", "qrels", "workload", "approved_workload", "package", "output", "config", "frozen_manifest", "artifact_sha256", "receipt_digest"}, f"{role}/{domain} native receipt")
    if payload["schema"] != RECEIPT_SCHEMA or digest_without(payload, "receipt_digest") != require_sha256(payload["receipt_digest"], f"{path}.receipt_digest"):
        raise GateContractError(f"{path}: native receipt schema/self-digest mismatch")
    nonce = require_string(payload["nonce"], f"{path}.nonce")
    if not NONCE_RE.fullmatch(nonce) or (expected_nonce is not None and nonce != expected_nonce):
        raise GateContractError(f"{path}: receipt nonce mismatch/invalid")
    if payload["gate_id"] != frozen_info["manifest"]["gate_id"] or payload["role"] != role or payload["domain"] != domain or payload["split"] != "heldout" or payload["official"] is not False or payload["native_evaluator"] is not True or payload["wrapper"] is not False or payload["exit_status"] != 0 or payload["dataset_id"] != domain:
        raise GateContractError(f"{path}: native receipt identity/producer flags mismatch")
    frozen = frozen_info["manifest"]
    executable = normalize_file_record(payload["executable"], path.parent, f"{path}.executable")
    if executable != frozen["source"]["binary"]:
        raise GateContractError(f"{path}: executable binding mismatch")
    package = normalize_package_record(payload["package"], path.parent, f"{path}.package", expected_role=role)
    if package != frozen_info["anchor" if role == "anchor" else "candidate"]:
        raise GateContractError(f"{path}: package binding mismatch")
    qrels = normalize_qrels_record(payload["qrels"], path.parent, f"{path}.qrels")
    if qrels != frozen_info["qrels"][domain]:
        raise GateContractError(f"{path}: qrels binding mismatch")
    workload_record = normalize_file_record(payload["workload"], path.parent, f"{path}.workload")
    approved_record = normalize_file_record(payload["approved_workload"], path.parent, f"{path}.approved_workload")
    dataset_record = normalize_file_record(payload["dataset"], path.parent, f"{path}.dataset")
    if dataset_record != frozen["source"]["dataset"] or workload_record != frozen["source"]["workload"] or approved_record != frozen["source"]["approved_workload"]:
        raise GateContractError(f"{path}: workload binding mismatch")
    cwd = normalize_directory(payload["cwd"], path.parent, f"{path}.cwd")
    if cwd != frozen["source"]["cwd"]:
        raise GateContractError(f"{path}: cwd binding mismatch")
    validate_argv(payload["argv"], binary=Path(executable["path"]), package=Path(package["path"]), base_dir=path.parent, label=f"{path}.argv")
    if payload["argv"] != frozen["source"]["argv"][role] or payload["argv_sha256"] != sha256_json(payload["argv"]):
        raise GateContractError(f"{path}: exact argv binding mismatch")
    config = require_mapping(payload["config"], f"{path}.config")
    if config != {"dimension": DIMENSION, "bits": [Q3_BITS, Q5_BITS], "seed": TURBOQUANT_SEED, "top_k": TOP_K, "package_mode": "sibling", "surfaces": list(SURFACES), "cutoffs": {"ndcg_at_10": 10, "recall_at_100": 100}}:
        raise GateContractError(f"{path}: native evaluator config mismatch")
    output_record = normalize_file_record(payload["output"], path.parent, f"{path}.output")
    if output_record != {"path": str(artifact_path), "sha256": artifact_sha256, "bytes": artifact_path.stat().st_size}:
        raise GateContractError(f"{path}: output/artifact path binding mismatch")
    frozen_binding = require_mapping(payload["frozen_manifest"], f"{path}.frozen_manifest")
    require_exact_keys(frozen_binding, {"path", "file_sha256", "manifest_sha256"}, f"{path}.frozen_manifest")
    if frozen_binding != {"path": frozen_info["path"], "file_sha256": frozen_info["expected_external_sha256"], "manifest_sha256": frozen["manifest_sha256"]}:
        raise GateContractError(f"{path}: frozen manifest binding mismatch")
    if payload["artifact_sha256"] != artifact_sha256:
        raise GateContractError(f"{path}: artifact hash binding mismatch")
    return {"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size, "nonce": nonce, "receipt": payload}


def validate_artifact(path: Path, *, expected_metrics: dict[str, Any], expected_receipt: dict[str, Any], frozen_info: dict[str, Any], role: str, domain: str) -> dict[str, Any]:
    path = path.resolve(strict=False)
    require_regular_file(path, f"{role}/{domain} metrics artifact")
    payload = read_json(path, f"{role}/{domain} metrics artifact")
    require_exact_keys(payload, {"schema", "gate_id", "artifact_path", "role", "domain", "split", "package", "source", "receipt_path", "frozen_manifest_sha256", "receipt_nonce", "observed_workload", "boundary_evidence", "rows", "aggregate"}, f"{role}/{domain} metrics artifact")
    frozen = frozen_info["manifest"]
    if payload["schema"] != ARTIFACT_SCHEMA or payload["gate_id"] != frozen["gate_id"] or payload["role"] != role or payload["domain"] != domain or payload["split"] != "heldout":
        raise GateContractError(f"{path}: artifact identity/schema mismatch")
    artifact_path = resolve_path(payload["artifact_path"], path.parent, f"{path}.artifact_path")
    if artifact_path != path or artifact_path != Path(expected_metrics["path"]):
        raise GateContractError(f"{path}: artifact path substitution/mismatch")
    package = normalize_package_record(payload["package"], path.parent, f"{path}.package", expected_role=role)
    if package != frozen_info["anchor" if role == "anchor" else "candidate"]:
        raise GateContractError(f"{path}: package provenance mismatch")
    normalize_artifact_source(payload["source"], path, frozen_info, role, domain)
    expected_qids = frozen_info["workload"]["query_ids_by_domain"][domain]
    observed = require_mapping(payload["observed_workload"], f"{path}.observed_workload")
    expected_observed = {"workload_id": frozen_info["workload"]["workload_id"], "query_count": len(expected_qids), "qids_sha256": sha256_json(expected_qids), "scored_query_count": len(expected_qids)}
    if observed != expected_observed:
        raise GateContractError(f"{path}: observed workload mismatch")
    boundary = normalize_boundary_evidence(payload["boundary_evidence"], domain=domain, workload=frozen_info["workload"], label=f"{path}.boundary_evidence")
    row_result = aggregate_rows(payload["rows"], expected_qids, f"{path}.rows")
    aggregate_node = require_mapping(payload["aggregate"], f"{path}.aggregate")
    require_exact_keys(aggregate_node, set(SURFACES), f"{path}.aggregate")
    for surface in SURFACES:
        surface_node = require_mapping(aggregate_node[surface], f"{path}.aggregate.{surface}")
        require_exact_keys(surface_node, set(METRICS_BY_SURFACE[surface]), f"{path}.aggregate.{surface}")
        for metric in METRICS_BY_SURFACE[surface]:
            declared = require_finite_number(surface_node[metric], f"{path}.aggregate.{surface}.{metric}", bounded=True)
            if abs(declared - row_result["aggregate"][surface][metric]) > COMPARISON_EPSILON:
                raise GateContractError(f"{path}: declared aggregate mismatch for {surface}.{metric}")
    receipt_path = resolve_path(payload["receipt_path"], path.parent, f"{path}.receipt_path")
    if receipt_path != Path(expected_receipt["path"]):
        raise GateContractError(f"{path}: receipt path substitution/mismatch")
    if not receipt_path.is_file() or receipt_path.is_symlink():
        raise GateContractError(f"{path}: receipt missing/substituted")
    frozen_manifest_sha = require_sha256(payload["frozen_manifest_sha256"], f"{path}.frozen_manifest_sha256")
    if frozen_manifest_sha != frozen["manifest_sha256"]:
        raise GateContractError(f"{path}: frozen manifest digest binding mismatch")
    nonce = require_string(payload["receipt_nonce"], f"{path}.receipt_nonce")
    if not NONCE_RE.fullmatch(nonce):
        raise GateContractError(f"{path}: invalid receipt nonce")
    artifact_sha = sha256_file(path)
    receipt = validate_receipt(receipt_path, artifact_path=path, artifact_sha256=artifact_sha, frozen_info=frozen_info, role=role, domain=domain, expected_nonce=nonce)
    return {"path": str(path), "sha256": artifact_sha, "bytes": path.stat().st_size, "receipt": receipt, "role": role, "domain": domain, "aggregate": row_result["aggregate"], "rows_by_qid": row_result["rows_by_qid"], "qids": row_result["qids"], "boundary": boundary}


def at_least(value: float, threshold: float) -> bool:
    return value + COMPARISON_EPSILON >= threshold


def aggregate_gate(artifacts: dict[tuple[str, str], dict[str, Any]], frozen_info: dict[str, Any]) -> dict[str, Any]:
    failures: list[str] = []
    domain_reports: dict[str, Any] = {}
    for domain in DOMAINS:
        anchor = artifacts[("anchor", domain)]
        candidate = artifacts[("candidate", domain)]
        delta = {surface: {metric: candidate["aggregate"][surface][metric] - anchor["aggregate"][surface][metric] for metric in METRICS_BY_SURFACE[surface]} for surface in SURFACES}
        query_safety: dict[str, Any] = {}
        for qid in frozen_info["workload"]["query_ids_by_domain"][domain]:
            a = anchor["rows_by_qid"][qid]["metrics"]
            c = candidate["rows_by_qid"][qid]["metrics"]
            qdelta = {surface: {metric: c[surface][metric] - a[surface][metric] for metric in METRICS_BY_SURFACE[surface]} for surface in SURFACES}
            checks = {"q3_ndcg_at_10_non_regressing": at_least(qdelta["q3"]["ndcg_at_10"], THRESHOLDS["q3_query_ndcg_at_10_delta_min"]), "dense_ndcg_at_10_within_tolerance": at_least(qdelta["dense"]["ndcg_at_10"], THRESHOLDS["dense_query_ndcg_at_10_delta_min"]), "q5_ndcg_at_10_within_tolerance": at_least(qdelta["q5"]["ndcg_at_10"], THRESHOLDS["q5_query_ndcg_at_10_delta_min"]), "q3_recall_at_100_non_regressing": at_least(qdelta["q3"]["recall_at_100"], THRESHOLDS["q3_query_recall_at_100_delta_min"])}
            if not all(checks.values()):
                failures.extend(f"query:{domain}:{qid}:{name}" for name, passed in checks.items() if not passed)
            query_safety[qid] = {"delta": qdelta, "checks": checks, "pass": all(checks.values())}
        safety = {"q3_ndcg_at_10_non_regressing": at_least(delta["q3"]["ndcg_at_10"], THRESHOLDS["q3_domain_ndcg_at_10_delta_min"]), "dense_ndcg_at_10_within_tolerance": at_least(delta["dense"]["ndcg_at_10"], THRESHOLDS["dense_domain_ndcg_at_10_delta_min"]), "q5_ndcg_at_10_within_tolerance": at_least(delta["q5"]["ndcg_at_10"], THRESHOLDS["q5_domain_ndcg_at_10_delta_min"]), "q3_recall_at_100_non_regressing": at_least(delta["q3"]["recall_at_100"], THRESHOLDS["q3_domain_recall_at_100_delta_min"])}
        failures.extend(f"domain:{domain}:{name}" for name, passed in safety.items() if not passed)
        boundary = None
        if domain == "nfcorpus":
            boundary_qids = frozen_info["workload"]["nfcorpus_boundary_qids"]
            boundary_checks = {qid: query_safety[qid]["checks"]["q3_recall_at_100_non_regressing"] for qid in boundary_qids}
            boundary = {"rank_window": [80, 120], "qids": boundary_qids, "qids_sha256": sha256_json(boundary_qids), "q3_recall_at_100_non_regressing": boundary_checks, "pass": all(boundary_checks.values())}
            if not boundary["pass"]:
                failures.append("nfcorpus:boundary:q3_recall_at_100_non_regressing")
        domain_reports[domain] = {"anchor": anchor["aggregate"], "candidate": candidate["aggregate"], "delta": delta, "query_safety": query_safety, "boundary": boundary, "safety": safety, "pass": all(safety.values()) and all(item["pass"] for item in query_safety.values()) and (boundary is None or boundary["pass"])}
    macro_delta = {surface: {metric: math.fsum(domain_reports[domain]["delta"][surface][metric] for domain in DOMAINS) / len(DOMAINS) for metric in METRICS_BY_SURFACE[surface]} for surface in SURFACES}
    macro_checks = {
        "q3_macro_ndcg_at_10": at_least(macro_delta["q3"]["ndcg_at_10"], THRESHOLDS["q3_macro_ndcg_at_10_delta_min"]),
        "dense_macro_ndcg_at_10": at_least(macro_delta["dense"]["ndcg_at_10"], THRESHOLDS["dense_macro_ndcg_at_10_delta_min"]),
        "q5_macro_ndcg_at_10": at_least(macro_delta["q5"]["ndcg_at_10"], THRESHOLDS["q5_macro_ndcg_at_10_delta_min"]),
        "q3_macro_recall_at_100": at_least(macro_delta["q3"]["recall_at_100"], THRESHOLDS["q3_macro_recall_at_100_delta_min"]),
    }
    failures.extend(f"macro:{name}" for name, passed in macro_checks.items() if not passed)
    return {"domains": domain_reports, "macro": {"q3_ndcg_at_10_delta": macro_delta["q3"]["ndcg_at_10"], "dense_ndcg_at_10_delta": macro_delta["dense"]["ndcg_at_10"], "q5_ndcg_at_10_delta": macro_delta["q5"]["ndcg_at_10"], "q3_recall_at_100_delta": macro_delta["q3"]["recall_at_100"], "checks": macro_checks, "pass": all(macro_checks.values())}, "thresholds": copy.deepcopy(THRESHOLDS), "failures": failures, "gate_pass": not failures}


def attest_manifest(frozen_path: Path, output_path: Path | None = None, *, expected_frozen_manifest_sha256: str | None = None) -> dict[str, Any]:
    frozen_info = validate_frozen_manifest(frozen_path, expected_frozen_manifest_sha256=expected_frozen_manifest_sha256, require_outputs_absent=False)
    frozen = frozen_info["manifest"]
    expected_attestation = frozen_info["attestation_path"]
    if output_path is None:
        output_path = expected_attestation
    output_path = output_path.resolve(strict=False)
    if output_path != expected_attestation or output_path.exists() or output_path.is_symlink():
        raise GateContractError("attestation output path must exactly match absent frozen binding")
    metrics_outputs = {(item["role"], item["domain"]): item for item in frozen_info["outputs"] if item["kind"] == "metrics"}
    receipt_outputs = {(item["role"], item["domain"]): item for item in frozen_info["outputs"] if item["kind"] == "receipt"}
    artifacts: dict[tuple[str, str], dict[str, Any]] = {}
    output_bindings: list[dict[str, Any]] = []
    nonce_set: set[str] = set()
    mtime_floor = Path(frozen_info["path"]).stat().st_mtime_ns
    for role in ROLES:
        for domain in DOMAINS:
            metrics_expected = metrics_outputs[(role, domain)]
            receipt_expected = receipt_outputs[(role, domain)]
            metrics_path = Path(metrics_expected["path"])
            receipt_path = Path(receipt_expected["path"])
            for kind, path in (("metrics", metrics_path), ("receipt", receipt_path)):
                if not path.is_file() or path.is_symlink():
                    raise GateContractError(f"missing/substituted {role}/{domain} {kind} output: {path}")
                if path.stat().st_mtime_ns <= mtime_floor:
                    raise GateContractError(f"stale {role}/{domain} {kind} output predates frozen manifest")
            artifact = validate_artifact(metrics_path, expected_metrics=metrics_expected, expected_receipt=receipt_expected, frozen_info=frozen_info, role=role, domain=domain)
            nonce = artifact["receipt"]["nonce"]
            if nonce in nonce_set:
                raise GateContractError(f"duplicate native receipt nonce: {nonce}")
            nonce_set.add(nonce)
            artifacts[(role, domain)] = artifact
            output_bindings.append({"kind": "metrics", "role": role, "domain": domain, "path": str(metrics_path), "sha256": artifact["sha256"], "bytes": artifact["bytes"], "receipt_path": str(receipt_path), "receipt_sha256": artifact["receipt"]["sha256"], "nonce": nonce, "package_sha256": frozen[role]["package"]["sha256"], "source_manifest_sha256": frozen["source"]["manifest"]["sha256"], "binary_sha256": frozen["source"]["binary"]["sha256"], "argv_sha256": sha256_json(frozen["source"]["argv"][role]), "cwd": frozen["source"]["cwd"], "dataset_sha256": frozen["source"]["dataset"]["sha256"], "workload_sha256": frozen["source"]["workload"]["sha256"], "approved_workload_sha256": frozen["source"]["approved_workload"]["sha256"]})
    summary = aggregate_gate(artifacts, frozen_info)
    report = {"schema": ATTESTATION_SCHEMA, "gate_id": frozen["gate_id"], "candidate_id": frozen["candidate_id"], "anchor_id": frozen["anchor_id"], "dimension": DIMENSION, "split": "heldout", "evaluation_executed": True, "evaluation_invoked_by_gate": False, "official_metric_values_read_by_gate": False, "frozen_manifest": {"path": frozen_info["path"], "expected_file_sha256": frozen_info["expected_external_sha256"], "actual_file_sha256": frozen_info["file_sha256"], "manifest_sha256": frozen["manifest_sha256"]}, "provenance": {"anchor_package": frozen["anchor"]["package"], "candidate_package": frozen["candidate"]["package"], "source_manifest": frozen["source"]["manifest"], "binary": frozen["source"]["binary"], "cwd": frozen["source"]["cwd"], "argv": frozen["source"]["argv"], "argv_sha256": frozen["provenance"]["argv_sha256"], "dataset": frozen["source"]["dataset"], "workload": frozen["source"]["workload"], "approved_workload": frozen["source"]["approved_workload"], "qrels": frozen["qrels"], "exclusions": [{"path": item["path"], "sha256": item["sha256"], "name": item["name"], "semantic": item["semantic"], "contains_metric_payload": item["contains_metric_payload"], "qids_sha256_by_domain": item["qids_sha256_by_domain"]} for item in frozen["exclusions"]]}, "output_bindings": output_bindings, "output_bindings_sha256": sha256_json(output_bindings), "domains": summary["domains"], "macro": summary["macro"], "thresholds": summary["thresholds"], "failures": summary["failures"], "gate_pass": summary["gate_pass"]}
    report["attestation_sha256"] = digest_without(report, "attestation_sha256")
    write_json_new(output_path, report)
    return report


# Explicit aliases keep the phase names discoverable for callers and reviews.
preflight = freeze_manifest
aggregate = attest_manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("freeze", "preflight"):
        command = sub.add_parser(name, help="validate plan and freeze absent output bindings")
        command.add_argument("--plan", "--input-manifest", "--manifest", dest="input_manifest", required=True, type=Path)
        command.add_argument("--output-manifest", "--output", dest="output_manifest", required=True, type=Path)
    for name in ("attest", "aggregate"):
        command = sub.add_parser(name, help="attest native receipts and aggregate paired metrics")
        command.add_argument("--frozen-manifest", "--manifest", dest="frozen_manifest", required=True, type=Path)
        command.add_argument("--expected-frozen-manifest-sha256", required=True)
        command.add_argument("--output-attestation", "--output", dest="output_attestation", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.command in {"freeze", "preflight"}:
            frozen = freeze_manifest(args.input_manifest, args.output_manifest)
            print(json.dumps({"ok": True, "mode": "freeze", "manifest_sha256": frozen["manifest_sha256"], "file_sha256": frozen["file_sha256"]}, sort_keys=True))
            return 0
        report = attest_manifest(args.frozen_manifest, args.output_attestation, expected_frozen_manifest_sha256=args.expected_frozen_manifest_sha256)
        print(json.dumps({"ok": report["gate_pass"], "mode": "attest", "gate_pass": report["gate_pass"], "attestation_sha256": report["attestation_sha256"]}, sort_keys=True))
        return 0 if report["gate_pass"] else 1
    except (GateError, OSError, TypeError, KeyError, struct.error) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
