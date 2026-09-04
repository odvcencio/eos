#!/usr/bin/env python3
"""Fail-closed, two-phase D384 AOQT heldout quality gate.

``freeze`` is the plan-only phase.  It validates every input, parses the
complete qrels files, resolves the three domain-scoped qid-only exclusion
manifests, and writes an immutable manifest whose *external file SHA-256* is a
trust anchor supplied to the next phase.

``run`` is the only production quality-gate path.  It requires the external
frozen-manifest SHA, the externally approved workload-descriptor SHA, the
current pinned D384 anchor identity, and the expected EOS executable SHA.  The
gate creates a fresh nonce-named run directory, constructs the complete EOS
``eval-retrieval-turboquant`` argv itself, preflights all output paths, invokes
the pinned executable, and accepts only native metrics/per-query files written
by that invocation.  Native outputs must carry both the gate binding and an
independently resolved runtime binding.  The gate binding proves which
invocation the harness asked for; the runtime binding proves which executable,
argv, package siblings, and dataset files the native evaluator actually
resolved.  A child that merely echoes caller-authored metadata therefore
cannot become evidence.

``attest``/``aggregate`` are intentionally disabled.  Importing receipts or
metrics supplied by a caller is not a production gate.  The tests call the
internal ``test_mode`` hooks with a local mock executable and synthetic files;
the command-line interface has no synthetic-fixture escape hatch.

The quality policy is immutable: q3 macro nDCG@10 must improve by at least
0.0003, dense nDCG@10 may drop by at most 0.0005, q5 nDCG@10 may drop by at
most 0.001, and q3 recall@100 may not regress.  The same substantive safety
floors are enforced per domain; per-query deltas are retained as diagnostics,
not as an additional blanket acceptance floor.  Dense/q5 recall is
intentionally not a gate metric; only q3 recall@100 is required.  NFCorpus
boundary evidence requires native per-query rankings through ranks 80..120
for every fixed boundary qid.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import re
import struct
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parent.parent

# Schemas are versioned independently from the old caller-authored receipt
# path.  The v3 plan is deliberately not accepted by the old attest code.
PLAN_SCHEMA = "eos.aoqt.heldout_gate_plan.v4"
FROZEN_SCHEMA = "eos.aoqt.heldout_gate_frozen.v4"
APPROVED_WORKLOAD_SCHEMA = "eos.aoqt.approved_heldout_workload.v2"
WORKLOAD_SCHEMA = "eos.aoqt.heldout_workload.v3"
DATASET_SCHEMA = "eos.aoqt.heldout_dataset_manifest.v1"
PACKAGE_ATTESTATION_SCHEMA = "eos.aoqt.native_package_attestation.v1"
PACKAGE_MANIFEST_SCHEMA = "eos.aoqt.package_manifest.v1"  # synthetic tests only
NATIVE_PACKAGE_MANIFEST_SCHEMA = "eos.aoqt.native_package_manifest.v1"
SIDECAR_SCHEMA = "eos.aoqt.aoqt_sidecar_attestation.v1"
NATIVE_METRICS_SCHEMA = "manta.embedding_turboquant_retrieval_metrics.v1"
NATIVE_PER_QUERY_SCHEMA = "manta.embedding_turboquant_retrieval_per_query.v1"
GATE_BINDING_SCHEMA = "eos.aoqt.native_eval_gate_binding.v1"
RUNTIME_BINDING_SCHEMA = "eos.aoqt.native_eval_runtime_binding.v1"
BOUNDARY_EVIDENCE_SCHEMA = "eos.aoqt.nfcorpus_boundary_evidence.v1"
EXECUTION_RECEIPT_SCHEMA = "eos.aoqt.native_eval_execution_receipt.v1"
ATTESTATION_SCHEMA = "eos.aoqt.heldout_gate_attestation.v4"

# Backward-compatible names are retained only for importers that inspect the
# schema constants.  There is no backward-compatible production attest path.
ARTIFACT_SCHEMA = NATIVE_METRICS_SCHEMA
RECEIPT_SCHEMA = EXECUTION_RECEIPT_SCHEMA

DOMAINS = ("fiqa", "nfcorpus", "scifact")
ROLES = ("anchor", "candidate")
SURFACES = ("dense", "q3", "q5")
METRICS_BY_SURFACE = {
    "dense": ("ndcg_at_10",),
    "q3": ("ndcg_at_10", "recall_at_100"),
    "q5": ("ndcg_at_10",),
}
EXCLUSION_NAMES = ("dev4", "reserve4", "official-test")
EXCLUSION_SEMANTIC = "qid_only_no_metric_payload"
DIMENSION = 384
Q3_BITS = 3
Q5_BITS = 5
TURBOQUANT_SEED = 5581486560434873699
TOP_K = 120
BATCH_SIZE = 64
HELDOUT_SPLIT = "test"
NATIVE_EVAL_SUBCOMMAND = "eval-retrieval-turboquant"
RESULT_OUTPUT_COUNT = len(DOMAINS) * len(ROLES)
EXPECTED_OUTPUT_COUNT = RESULT_OUTPUT_COUNT * 4  # metrics, TSV, per-query, receipt
COMPARISON_EPSILON = 1e-12

# These are the immutable current D384 pre-transform anchor identities from
# the train-only artifact audit.  The path is not the trust anchor: all three
# identities are required, and production additionally requires callers to
# provide these same values on the ``run`` command line.  The package-manifest
# identity is the actual native .package.mll XPKG manifest.  The stage-2 audit
# JSON is retained separately because it is provenance metadata, not runtime
# package identity.
D384_ANCHOR_PACKAGE_SHA256 = "188265db16992ab24be15e678c5f7e175bebad769e8d844e8b0f50ffc23bd5bf"
D384_ANCHOR_PACKAGE_MANIFEST_SHA256 = "26058741c0709dc9dc1ac3535a1abce6b30280342869883670bdacd1cc4fd5cd"
D384_ANCHOR_AUDIT_MANIFEST_SHA256 = "e1e28316355590b5c9e55c9af7a44014dcfd810258da319c077a316a38fa7683"
# Public callers historically use this name for the manifest identity.
D384_ANCHOR_MANIFEST_SHA256 = D384_ANCHOR_PACKAGE_MANIFEST_SHA256
D384_ANCHOR_EMBEDDING_SPACE_ID = "99aa06139496a87ca3bd79598f99805d04d38df3398d9886ac386c4e8ca0b0db"
D384_ANCHOR_ARTIFACT_ID = "d384-pre"
D384_ANCHOR_PATH_SUFFIX = "runs/eos-wide384-projection-tail-seed191-repeat-v1-20260902T004753Z/packages/d384-pre.mll"

# The gate must never turn a caller-authored descriptor into an official
# workload.  This is a deliberately explicit registry rather than a value
# derived from the plan.  The qid-only official-test population is
# preregistered independently of any candidate; candidate package eligibility
# and native metrics remain downstream requirements.  Synthetic tests use the
# private test hooks and never consult this registry.
TRUSTED_WORKLOAD_REGISTRY_SCHEMA = "eos.aoqt.trusted_workload_registry.v1"
PRODUCTION_TRUSTED_WORKLOAD_REGISTRY: dict[str, Any] = {
    "schema": TRUSTED_WORKLOAD_REGISTRY_SCHEMA,
    "status": "active",
    "reason": "fixed qid-only official-test workload is preregistered; candidate package and metrics remain required downstream",
    "source_root": "datasets/manta-embed-v1/raw",
    "domains": {
        "fiqa": {
            "qrels_path_suffix": "datasets/manta-embed-v1/raw/fiqa/fiqa/qrels/test.tsv",
            "qrels_sha256": "6adc2a640dcdd22bb8b3858f89107adef2a7c3db20a63550dfa7a0f71e379e44",
            "qid_set_sha256": "7a7ee595b288753a46162e1daec8631d3a77300b1d550afdc8c67325d59fbe33",
            "query_count": 648,
            "qrels_pair_count": 1706,
            "corpus_path_suffix": "datasets/manta-embed-v1/raw/fiqa/fiqa/corpus.jsonl",
            "corpus_sha256": "ff593e4df9933955dc3af83be0c3fa28ac7465f627e08c2e53593e734d506517",
            "queries_path_suffix": "datasets/manta-embed-v1/raw/fiqa/fiqa/queries.jsonl",
            "queries_sha256": "eede1e61d4a0188940239b53ebc2da91f577a6a34679c812d1eb9090c29877bc",
        },
        "nfcorpus": {
            "qrels_path_suffix": "datasets/manta-embed-v1/raw/nfcorpus/nfcorpus/qrels/test.tsv",
            "qrels_sha256": "f8fba6ef3d4dd9c3a242a8ba4ae38276fc3622fce7dcbae764766d564542fd2a",
            "qid_set_sha256": "6778186887db52ede15d04f15607996335fa40e40492404290b378c7c2b7b374",
            "query_count": 323,
            "qrels_pair_count": 12334,
            "corpus_path_suffix": "datasets/manta-embed-v1/raw/nfcorpus/nfcorpus/corpus.jsonl",
            "corpus_sha256": "10cc83ef1826b1425e6a87090b5140b39b27755d5a27e48215a88611c899991f",
            "queries_path_suffix": "datasets/manta-embed-v1/raw/nfcorpus/nfcorpus/queries.jsonl",
            "queries_sha256": "d024e6621b84925d485ae473d316a0c3af31c62c8068a59fb29d22f7613aef2a",
        },
        "scifact": {
            "qrels_path_suffix": "datasets/manta-embed-v1/raw/scifact/scifact/qrels/test.tsv",
            "qrels_sha256": "0864bb985e0ca2367ba217977e72004d549054b2b06666ed9d4825ac7c21284c",
            "qid_set_sha256": "baa4c5ec793edbb5173815db5afbca79aca7f32c9498be84b4de3363f7e82d71",
            "query_count": 300,
            "qrels_pair_count": 339,
            "corpus_path_suffix": "datasets/manta-embed-v1/raw/scifact/scifact/corpus.jsonl",
            "corpus_sha256": "dec31c8182f3d744c7d2c09423756fd1d17cbef75808db13ba01cc0aab4d1ac6",
            "queries_path_suffix": "datasets/manta-embed-v1/raw/scifact/scifact/queries.jsonl",
            "queries_sha256": "8ff84a7c903f722981cd8d595c022660140c51867b27608a6d4910db86080313",
        },
    },
    "nfcorpus_boundary": {
        "source_path_suffix": "runs/eos-d384-aoqt-trainonly-v1-20260903T092452Z/exclusions-scoped-v2/official-test.json",
        "source_sha256": "6f90346fc892faa5135e1bca7fdd3a668ba755a5c617eba87f427e8a0601b2a0",
        "qid_set_sha256": "6778186887db52ede15d04f15607996335fa40e40492404290b378c7c2b7b374",
        "qid_count": 323,
        "rank_window": [80, 120],
    },
}

AOQT_TOPOLOGY = {
    "id": "aoqt_givens_v1",
    "dim": DIMENSION,
    "stages": 8,
    "pairs_per_stage": 192,
    "angle_count": 1536,
    "angle_cap_default": 0.04,
    "angle_cap_hard": 0.08,
}
# This is the independent train-only provenance expected in the native XPKG
# AOQTTransformPolicy.  It is deliberately separate from the heldout
# evaluation dataset/qrels registry below: the candidate's transform was
# trained against the canonical raw-v4 AOQT calibration manifest, not the
# heldout BEIR test files used by this gate.
AOQT_TRAIN_PROVENANCE = {
    "dataset_manifest_sha256": "690b35ae2acaad8f37076cc924fd09d14129bedd10d950228737a65848387ef1",
    "qrels_sha256_by_dataset": {
        "fiqa": "8af86b58ce435d91e9575c5345fa0db6dbeb4f84e32a5558e84b98d70df328dd",
        "nfcorpus": "7e0f4ff8adad1dab6e461c4d0a554f88297991c5402ee95a4259b9dc72309475",
        "scifact": "18b018b2ba34aaf5651062368b2aad0fe9d802adc0db5d09525a3fd65c718408",
    },
    "compatibility_digest": "67c25bda77ef87322dbd16f675ba480fbdc8ce100f29f06778c5ffebc31e5181",
}
AOQT_LEGAL_SCOPE = {
    "train_allowed_for_research": True,
    "release_train_allowed": False,
    "commercial_use_allowed": False,
    "free_open_release_allowed": False,
    "quality_claim": False,
}
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
NONCE_RE = re.compile(r"[a-z0-9][a-z0-9-]{31,63}\Z")
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


def _strict_load(text: str, label: str) -> Any:
    try:
        return json.loads(
            text,
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise GateContractError(f"{label}: cannot read strict JSON: {exc}") from exc


def read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = _strict_load(path.read_text(encoding="utf-8"), label)
    except OSError as exc:
        raise GateContractError(f"{label}: cannot read JSON: {exc}") from exc
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
        details: list[str] = []
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
        if candidate.is_symlink():
            raise GateContractError(f"{label}: symlink paths are not accepted")
        return candidate.resolve(strict=False)
    except OSError as exc:
        raise GateContractError(f"{label}: cannot resolve path: {exc}") from exc


def require_regular_file(path: Path, label: str, *, executable: bool = False) -> None:
    if path.is_symlink() or not path.is_file():
        raise GateContractError(f"{label}: regular non-symlink file is required: {path}")
    if executable and not os.access(path, os.X_OK):
        raise GateContractError(f"{label}: executable permission is required: {path}")


def require_elf_executable(path: Path, label: str) -> None:
    """Reject script/wrapper launchers in the production EOS binary slot."""

    require_regular_file(path, label, executable=True)
    try:
        with path.open("rb") as handle:
            magic = handle.read(4)
    except OSError as exc:
        raise GateContractError(f"{label}: cannot inspect executable format: {exc}") from exc
    if magic != b"\x7fELF":
        raise GateContractError(f"{label}: production evaluator must be a native ELF executable, not a wrapper")


def require_directory(path: Path, label: str) -> None:
    if path.is_symlink() or not path.is_dir():
        raise GateContractError(f"{label}: existing non-symlink directory is required: {path}")


def path_within(path: Path, root: Path, label: str) -> None:
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise GateContractError(f"{label}: {path} escapes {root}") from exc


def normalize_file_record(value: Any, base_dir: Path, label: str, *, executable: bool = False) -> dict[str, Any]:
    record = require_mapping(value, label)
    require_exact_keys(record, {"path", "sha256", "bytes"}, label)
    path = resolve_path(record["path"], base_dir, f"{label}.path")
    require_regular_file(path, label, executable=executable)
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


def require_domain_mapping(value: Any, label: str) -> dict[str, Any]:
    mapping = require_mapping(value, label)
    if set(mapping) != set(DOMAINS):
        raise GateContractError(f"{label}: exact domain coverage required")
    return {domain: mapping[domain] for domain in DOMAINS}


def qids_sha256_by_domain(qids_by_domain: dict[str, list[str]]) -> dict[str, str]:
    return {domain: sha256_json(qids_by_domain[domain]) for domain in DOMAINS}


def parse_qrels(path: Path, label: str) -> dict[str, Any]:
    """Parse a complete BEIR qrels TSV and derive, never trust, its workload."""

    require_regular_file(path, label)
    try:
        raw = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise GateContractError(f"{label}: cannot read qrels: {exc}") from exc
    rels: dict[str, dict[str, float]] = {}
    saw_data = False
    for line_number, raw_line in enumerate(raw.splitlines(), 1):
        line = raw_line.strip()
        if not line:
            continue
        fields = raw_line.rstrip("\r\n").split("\t")
        if len(fields) == 1:
            fields = re.split(r"\s+", line)
        if not saw_data and fields and fields[0].lower() in {"query-id", "qid", "query_id"}:
            saw_data = True
            continue
        saw_data = True
        if len(fields) == 4:
            if fields[1] != "0":
                raise GateContractError(f"{label}:{line_number}: four-column qrels must use a zero iteration field")
            qid_raw, doc_raw, score_raw = fields[0], fields[2], fields[3]
        elif len(fields) == 3:
            qid_raw, doc_raw, score_raw = fields
        else:
            raise GateContractError(f"{label}:{line_number}: expected 3 or 4 qrels columns")
        qid = require_qid(qid_raw, f"{label}:{line_number}.query_id")
        doc_id = require_string(doc_raw, f"{label}:{line_number}.doc_id")
        if any(char.isspace() for char in doc_id):
            raise GateContractError(f"{label}:{line_number}: doc_id must not contain whitespace")
        try:
            relevance = float(score_raw)
        except (TypeError, ValueError) as exc:
            raise GateContractError(f"{label}:{line_number}: invalid relevance") from exc
        if not math.isfinite(relevance) or relevance < 0:
            raise GateContractError(f"{label}:{line_number}: relevance must be finite and non-negative")
        if qid in rels and doc_id in rels[qid]:
            raise GateContractError(f"{label}:{line_number}: duplicate qid/doc pair")
        rels.setdefault(qid, {})[doc_id] = relevance
    if not rels:
        raise GateContractError(f"{label}: qrels are empty")
    qids = sorted(rels)
    positive_qids = [qid for qid in qids if any(value > 0 for value in rels[qid].values())]
    if len(positive_qids) != len(qids):
        raise GateContractError(f"{label}: every qid must have at least one positive relevance")
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "qids": qids,
        "qid_set_sha256": sha256_json(qids),
        "query_count": len(qids),
        "qrels_pair_count": sum(len(rels[qid]) for qid in qids),
        "relevant_pair_count": sum(1 for qid in qids for value in rels[qid].values() if value > 0),
        "rels": rels,
    }


def parse_corpus_ids(path: Path, label: str) -> set[str]:
    """Derive the corpus identity used to reject fabricated ranking docs."""

    require_regular_file(path, label)
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise GateContractError(f"{label}: cannot read corpus: {exc}") from exc
    ids: set[str] = set()
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        node = _strict_load(line, f"{label}:{line_number}")
        if not isinstance(node, dict) or set(node) - {"_id", "title", "text"}:
            raise GateContractError(f"{label}:{line_number}: corpus rows must be native _id/title/text records")
        doc_id = require_string(node.get("_id"), f"{label}:{line_number}._id")
        if any(char.isspace() for char in doc_id) or doc_id in ids:
            raise GateContractError(f"{label}:{line_number}: corpus document identity is invalid or duplicated")
        ids.add(doc_id)
    if not ids:
        raise GateContractError(f"{label}: corpus is empty")
    return ids


def _path_overlaps(left: Path, right: Path) -> bool:
    """Return true for equal or ancestor/descendant paths."""

    try:
        left.relative_to(right)
        return True
    except ValueError:
        pass
    try:
        right.relative_to(left)
        return True
    except ValueError:
        return False


def require_pairwise_disjoint_paths(paths: Iterable[Path], label: str) -> None:
    """Reject aliases and nested roots across independent gate artifacts."""

    normalized = sorted({path.resolve(strict=False) for path in paths}, key=str)
    for index, left in enumerate(normalized):
        for right in normalized[index + 1 :]:
            if _path_overlaps(left, right):
                raise GateContractError(f"{label}: paths must be pairwise disjoint: {left} and {right}")


def normalize_qrels_record(value: Any, base_dir: Path, label: str) -> dict[str, Any]:
    record = require_mapping(value, label)
    # Counts and qid lists are deliberately absent from the input record; they
    # are derived from the complete file by parse_qrels.
    require_exact_keys(record, {"path", "sha256", "bytes"}, label)
    file_record = normalize_file_record(record, base_dir, label)
    parsed = parse_qrels(Path(file_record["path"]), label)
    parsed.pop("rels")
    return {**file_record, **parsed}


def normalize_dataset_record(value: Any, base_dir: Path, domain: str, label: str) -> dict[str, Any]:
    record = require_mapping(value, label)
    require_exact_keys(record, {"dataset_id", "dataset_dir", "corpus", "queries", "manifest"}, label)
    dataset_id = require_string(record["dataset_id"], f"{label}.dataset_id", safe_id=True)
    dataset_dir = Path(normalize_directory(record["dataset_dir"], base_dir, f"{label}.dataset_dir"))
    corpus = normalize_file_record(record["corpus"], base_dir, f"{label}.corpus")
    queries = normalize_file_record(record["queries"], base_dir, f"{label}.queries")
    manifest = normalize_file_record(record["manifest"], base_dir, f"{label}.manifest")
    payload = read_json(Path(manifest["path"]), f"{label}.manifest")
    require_exact_keys(payload, {"schema", "domain", "dataset_id", "split", "corpus_sha256", "queries_sha256"}, f"{label}.manifest")
    if payload["schema"] != DATASET_SCHEMA or payload["domain"] != domain or payload["dataset_id"] != dataset_id or payload["split"] != HELDOUT_SPLIT:
        raise GateContractError(f"{label}: dataset manifest identity mismatch")
    if payload["corpus_sha256"] != corpus["sha256"] or payload["queries_sha256"] != queries["sha256"]:
        raise GateContractError(f"{label}: dataset manifest content binding mismatch")
    return {"dataset_id": dataset_id, "dataset_dir": str(dataset_dir), "corpus": corpus, "queries": queries, "manifest": manifest}


def _resolve_source_file(raw_path: str, base_dir: Path, label: str) -> Path:
    candidates = [base_dir / raw_path, REPO_ROOT / raw_path]
    for candidate in candidates:
        if candidate.is_symlink():
            raise GateContractError(f"{label}: source manifest symlink is not accepted: {raw_path}")
        resolved = candidate.resolve(strict=False)
        if resolved.is_file() and not resolved.is_symlink():
            return resolved
    raise GateContractError(f"{label}: source manifest file does not resolve: {raw_path}")


def normalize_exclusion_record(value: Any, base_dir: Path, expected_name: str, workload_qids: dict[str, list[str]], label: str) -> dict[str, Any]:
    record = require_mapping(value, label)
    require_exact_keys(record, {"kind", "name", "manifest"}, label)
    if record["kind"] != "qid_only" or record["name"] != expected_name:
        raise GateContractError(f"{label}: exact qid-only exclusion identity is required")
    manifest_record = normalize_file_record(record["manifest"], base_dir, f"{label}.manifest")
    payload = read_json(Path(manifest_record["path"]), f"{label}.manifest")
    require_no_forbidden_payload_keys(payload, f"{label}.manifest")
    require_exact_keys(payload, {"schema", "name", "qids_by_dataset", "source_sha256", "source_sha256_by_file"}, f"{label}.manifest")
    if payload["schema"] != "eos.aoqt_stage2.exclusion_qids.v1" or payload["name"] != expected_name:
        raise GateContractError(f"{label}: exclusion schema/name mismatch")
    qids = normalize_qids_by_domain(payload["qids_by_dataset"], f"{label}.manifest.qids_by_dataset")
    source_sha = require_sha256(payload["source_sha256"], f"{label}.manifest.source_sha256")
    source_map = require_mapping(payload["source_sha256_by_file"], f"{label}.manifest.source_sha256_by_file")
    if not source_map:
        raise GateContractError(f"{label}: source manifest map must be non-empty")
    normalized_source: dict[str, str] = {}
    resolved_sources: set[Path] = set()
    for raw_path, expected_sha in sorted(source_map.items()):
        source_name = require_string(raw_path, f"{label}.source.path")
        source_path = _resolve_source_file(source_name, Path(manifest_record["path"]).parent, f"{label}.source")
        if source_path in resolved_sources:
            raise GateContractError(f"{label}: source manifest contains duplicate resolved files")
        resolved_sources.add(source_path)
        normalized_expected = require_sha256(expected_sha, f"{label}.source.sha256")
        actual = sha256_file(source_path)
        if actual != normalized_expected:
            raise GateContractError(f"{label}: source manifest sha256 mismatch for {source_path}")
        # Preserve the manifest's canonical path spelling in the frozen
        # record.  The path is still resolved and hashed above; preserving it
        # keeps source_sha256 (the canonical map digest used by the existing
        # AOQT exclusion schema) stable across freeze/revalidation.
        normalized_source[source_name] = actual
    if sha256_json(dict(sorted(normalized_source.items()))) != source_sha:
        raise GateContractError(f"{label}: source_sha256 does not bind source_sha256_by_file")
    for domain in DOMAINS:
        intersection = sorted(set(qids[domain]) & set(workload_qids[domain]))
        if intersection:
            raise GateContractError(f"{label}: workload/exclusion qid intersection in {domain}: {intersection[:4]}")
    return {
        "kind": "qid_only",
        "semantic": EXCLUSION_SEMANTIC,
        "name": expected_name,
        "manifest": manifest_record,
        "manifest_sha256": manifest_record["sha256"],
        "source_sha256": source_sha,
        "source_sha256_by_file": normalized_source,
        "qids_by_domain": qids,
        "qids_sha256_by_domain": qids_sha256_by_domain(qids),
    }


def _validate_topology(value: Any, label: str) -> dict[str, Any]:
    topology = require_mapping(value, label)
    require_exact_keys(topology, set(AOQT_TOPOLOGY), label)
    normalized = {
        "id": require_string(topology["id"], f"{label}.id", safe_id=True),
        "dim": require_integer(topology["dim"], f"{label}.dim", minimum=1),
        "stages": require_integer(topology["stages"], f"{label}.stages", minimum=1),
        "pairs_per_stage": require_integer(topology["pairs_per_stage"], f"{label}.pairs_per_stage", minimum=1),
        "angle_count": require_integer(topology["angle_count"], f"{label}.angle_count", minimum=1),
        "angle_cap_default": require_finite_number(topology["angle_cap_default"], f"{label}.angle_cap_default"),
        "angle_cap_hard": require_finite_number(topology["angle_cap_hard"], f"{label}.angle_cap_hard"),
    }
    if normalized != AOQT_TOPOLOGY:
        raise GateContractError(f"{label}: AOQT topology mismatch")
    return normalized


def _validate_legal_scope(value: Any, label: str) -> dict[str, Any]:
    scope = require_mapping(value, label)
    require_exact_keys(scope, set(AOQT_LEGAL_SCOPE), label)
    for key in AOQT_LEGAL_SCOPE:
        if not isinstance(scope[key], bool):
            raise GateContractError(f"{label}.{key}: expected boolean")
    normalized = {key: scope[key] for key in AOQT_LEGAL_SCOPE}
    if normalized != AOQT_LEGAL_SCOPE:
        raise GateContractError(f"{label}: AOQT legal scope mismatch")
    return normalized


def _native_mll_section_tags(path: Path, label: str) -> set[bytes]:
    """Perform bounded MLL container validation without loading model tensors."""

    try:
        file_size = path.stat().st_size
        with path.open("rb") as handle:
            header = handle.read(24)
    except OSError as exc:
        raise GateContractError(f"{label}: cannot read MLL: {exc}") from exc
    if file_size < 24 or len(header) != 24 or header[:4] != b"MLL\0":
        raise GateContractError(f"{label}: production packages must be native MLL artifacts; JSON fallback is test-only")
    section_count = int.from_bytes(header[16:20], "little")
    if section_count <= 0 or section_count > 4096:
        raise GateContractError(f"{label}: invalid MLL section directory")
    directory_end = 24 + section_count * 64
    if directory_end > file_size:
        raise GateContractError(f"{label}: invalid MLL section directory")
    try:
        with path.open("rb") as handle:
            handle.seek(24)
            directory = handle.read(section_count * 64)
    except OSError as exc:
        raise GateContractError(f"{label}: cannot read MLL section directory: {exc}") from exc
    if len(directory) != section_count * 64:
        raise GateContractError(f"{label}: truncated MLL section directory")
    tags: set[bytes] = set()
    intervals: list[tuple[int, int]] = []
    for index in range(section_count):
        entry = directory[index * 64 : (index + 1) * 64]
        tag = entry[:4]
        if tag in tags:
            raise GateContractError(f"{label}: duplicate MLL section tag {tag!r}")
        start = int.from_bytes(entry[4:12], "little")
        size = int.from_bytes(entry[12:20], "little")
        end = start + size
        if start < directory_end or end < start or end > file_size:
            raise GateContractError(f"{label}: MLL section bounds invalid")
        if any(start < old_end and old_start < end for old_start, old_end in intervals):
            raise GateContractError(f"{label}: overlapping MLL sections")
        tags.add(tag)
        intervals.append((start, end))
    if not {b"HEAD", b"STRG"}.issubset(tags):
        raise GateContractError(f"{label}: native package missing HEAD/STRG sections")
    return tags


def _read_native_mll_sections(path: Path, label: str) -> dict[bytes, bytes]:
    """Read bounded MLL section bodies for package identity inspection.

    The package manifest's XPKG section is the runtime authority for sibling
    roles.  A caller-provided JSON rollup is not equivalent: it can omit a
    required sibling and still be internally self-consistent.  This parser
    mirrors the fixed MLL directory layout used by EOS and verifies all
    offsets before slicing any section.
    """

    tags = _native_mll_section_tags(path, label)
    try:
        file_size = path.stat().st_size
        if file_size > 64 * 1024 * 1024:
            raise GateContractError(f"{label}: MLL manifest exceeds bounded inspection size")
        data = path.read_bytes()
    except OSError as exc:
        raise GateContractError(f"{label}: cannot read MLL section bodies: {exc}") from exc
    if len(data) != file_size:
        raise GateContractError(f"{label}: MLL size changed during inspection")
    section_count = int.from_bytes(data[16:20], "little")
    sections: dict[bytes, bytes] = {}
    for index in range(section_count):
        entry = data[24 + index * 64 : 24 + (index + 1) * 64]
        tag = entry[:4]
        start = int.from_bytes(entry[4:12], "little")
        size = int.from_bytes(entry[12:20], "little")
        sections[tag] = data[start : start + size]
    if set(sections) != tags:
        raise GateContractError(f"{label}: MLL section inspection changed directory identity")
    return sections


def _mll_strings(body: bytes, label: str) -> list[str]:
    if len(body) < 4:
        raise GateContractError(f"{label}: STRG section is truncated")
    count = int.from_bytes(body[:4], "little")
    if count > 100_000:
        raise GateContractError(f"{label}: STRG string count is unbounded")
    strings: list[str] = []
    offset = 4
    for index in range(count):
        if offset + 4 > len(body):
            raise GateContractError(f"{label}: STRG length is truncated at {index}")
        length = int.from_bytes(body[offset : offset + 4], "little")
        offset += 4
        if length > len(body) - offset:
            raise GateContractError(f"{label}: STRG value is truncated at {index}")
        try:
            value = body[offset : offset + length].decode("utf-8")
        except UnicodeDecodeError as exc:
            raise GateContractError(f"{label}: STRG value is not UTF-8 at {index}") from exc
        strings.append(value)
        offset += length
    if offset != len(body):
        raise GateContractError(f"{label}: STRG has trailing bytes")
    return strings


def _mll_head_metadata(body: bytes, strings: list[str], label: str) -> dict[str, Any]:
    """Decode the scalar HEAD metadata used by EOS package manifests."""

    # version/profile/name/description are fixed-width fields in the MLL
    # HEAD section, followed by backend and capability arrays.
    if len(body) < 28:
        raise GateContractError(f"{label}: HEAD section is truncated")
    offset = 4 + 4 + 8 + 8

    def take(size: int, what: str) -> bytes:
        nonlocal offset
        if size < 0 or offset + size > len(body):
            raise GateContractError(f"{label}: HEAD {what} is truncated")
        chunk = body[offset : offset + size]
        offset += size
        return chunk

    backend_count = int.from_bytes(take(2, "backend count"), "little")
    take(backend_count * 2, "backend list")
    capability_count = int.from_bytes(take(2, "capability count"), "little")
    take(capability_count * 4, "capability list")
    metadata_count = int.from_bytes(take(2, "metadata count"), "little")
    if metadata_count > 4096:
        raise GateContractError(f"{label}: HEAD metadata count is unbounded")
    metadata: dict[str, Any] = {}
    for index in range(metadata_count):
        key_index = int.from_bytes(take(4, f"metadata[{index}] key"), "little")
        kind = take(1, f"metadata[{index}] kind")[0]
        if key_index >= len(strings) or not strings[key_index]:
            raise GateContractError(f"{label}: HEAD metadata[{index}] has invalid key index")
        key = strings[key_index]
        if key in metadata:
            raise GateContractError(f"{label}: duplicate HEAD metadata key {key!r}")
        if kind == 0:
            value: Any = None
        elif kind == 1:
            value = bool(take(1, f"metadata[{index}] bool")[0])
        elif kind == 2:
            value = int.from_bytes(take(8, f"metadata[{index}] int"), "little", signed=True)
        elif kind == 3:
            value = struct.unpack("<d", take(8, f"metadata[{index}] float"))[0]
        elif kind == 4:
            string_index = int.from_bytes(take(4, f"metadata[{index}] string index"), "little")
            if string_index >= len(strings):
                raise GateContractError(f"{label}: HEAD metadata[{index}] has invalid string index")
            value = strings[string_index]
        else:
            raise GateContractError(f"{label}: unsupported HEAD metadata kind {kind}")
        metadata[key] = value
    return metadata


def _parse_aoqt_qrels_hashes(value: Any, label: str) -> dict[str, str]:
    """Decode PackageManifest's canonical ``dataset=sha`` qrels metadata."""

    if not isinstance(value, str) or not value or any(ord(char) < 0x20 and char != "\n" for char in value):
        raise GateContractError(f"{label}: expected non-empty newline-separated qrels map")
    raw = value
    result: dict[str, str] = {}
    for index, line in enumerate(raw.split("\n")):
        if not line:
            raise GateContractError(f"{label}[{index}]: empty dataset entry")
        dataset, separator, digest = line.partition("=")
        if not separator or not dataset or dataset in result:
            raise GateContractError(f"{label}[{index}]: malformed or duplicate dataset entry")
        result[dataset] = require_sha256(digest, f"{label}[{index}].{dataset}")
    if not result:
        raise GateContractError(f"{label}: qrels map must not be empty")
    return result


def _parse_xpkg_aoqt_policy(head: dict[str, Any], role: str, label: str) -> dict[str, Any] | None:
    """Normalize the AOQTTransformPolicy serialized in native XPKG HEAD."""

    bool_keys = (
        "aoqt_transform_enabled",
        "aoqt_transform_research_only",
        "aoqt_transform_research_train_allowed",
        "aoqt_transform_release_train_allowed",
        "aoqt_transform_commercial_use_allowed",
        "aoqt_transform_free_open_release_allowed",
        "aoqt_transform_quality_claim",
    )
    if role == "anchor" and not any(key in head for key in bool_keys):
        # The pinned pre-transform anchor is an older training package whose
        # XPKG HEAD predates AOQT policy metadata.  Its absence is equivalent
        # to the native disabled/empty AOQTTransformPolicy; candidate XPKG
        # manifests must carry the complete policy.
        return None
    for key in bool_keys:
        if not isinstance(head.get(key, False), bool):
            raise GateContractError(f"{label}.HEAD.{key}: expected boolean AOQT policy metadata")
    enabled = head.get("aoqt_transform_enabled", False)
    if not enabled:
        if role == "candidate":
            raise GateContractError(f"{label}: candidate XPKG AOQT transform policy is disabled")
        return None
    required_strings = (
        "aoqt_transform_schema",
        "aoqt_transform_anchor_artifact_sha256",
        "aoqt_transform_anchor_package_manifest_sha256",
        "aoqt_transform_anchor_embedding_space_id",
        "aoqt_transform_dataset_manifest_sha256",
        "aoqt_transform_qrels_sha256_by_dataset",
        "aoqt_transform_compatibility_digest",
        "aoqt_transform_transform_sha256",
        "aoqt_transform_pairings_sha256",
        "aoqt_transform_angles_sha256",
    )
    for key in required_strings:
        if key == "aoqt_transform_qrels_sha256_by_dataset":
            _parse_aoqt_qrels_hashes(head.get(key), f"{label}.HEAD.{key}")
        else:
            require_string(head.get(key), f"{label}.HEAD.{key}")
    policy = {
        "schema": head["aoqt_transform_schema"],
        "enabled": enabled,
        "research_only": head["aoqt_transform_research_only"],
        "research_train_allowed": head["aoqt_transform_research_train_allowed"],
        "release_train_allowed": head["aoqt_transform_release_train_allowed"],
        "commercial_use_allowed": head["aoqt_transform_commercial_use_allowed"],
        "free_open_release_allowed": head["aoqt_transform_free_open_release_allowed"],
        "quality_claim": head["aoqt_transform_quality_claim"],
        "anchor_artifact_sha256": require_sha256(head["aoqt_transform_anchor_artifact_sha256"], f"{label}.HEAD.aoqt_transform_anchor_artifact_sha256"),
        "anchor_package_manifest_sha256": require_sha256(head["aoqt_transform_anchor_package_manifest_sha256"], f"{label}.HEAD.aoqt_transform_anchor_package_manifest_sha256"),
        "anchor_embedding_space_id": require_sha256(head["aoqt_transform_anchor_embedding_space_id"], f"{label}.HEAD.aoqt_transform_anchor_embedding_space_id"),
        "dataset_manifest_sha256": require_sha256(head["aoqt_transform_dataset_manifest_sha256"], f"{label}.HEAD.aoqt_transform_dataset_manifest_sha256"),
        "qrels_sha256_by_dataset": _parse_aoqt_qrels_hashes(head["aoqt_transform_qrels_sha256_by_dataset"], f"{label}.HEAD.aoqt_transform_qrels_sha256_by_dataset"),
        "compatibility_digest": require_sha256(head["aoqt_transform_compatibility_digest"], f"{label}.HEAD.aoqt_transform_compatibility_digest"),
        "transform_sha256": require_sha256(head["aoqt_transform_transform_sha256"], f"{label}.HEAD.aoqt_transform_transform_sha256"),
        "pairings_sha256": require_sha256(head["aoqt_transform_pairings_sha256"], f"{label}.HEAD.aoqt_transform_pairings_sha256"),
        "angles_sha256": require_sha256(head["aoqt_transform_angles_sha256"], f"{label}.HEAD.aoqt_transform_angles_sha256"),
    }
    if policy["schema"] != "eos.aoqt_transform_policy.v1" or policy["research_only"] is not True or policy["research_train_allowed"] is not True or policy["release_train_allowed"] or policy["commercial_use_allowed"] or policy["free_open_release_allowed"] or policy["quality_claim"]:
        raise GateContractError(f"{label}: native XPKG AOQT policy/legal scope mismatch")
    if policy["dataset_manifest_sha256"] != AOQT_TRAIN_PROVENANCE["dataset_manifest_sha256"] or policy["qrels_sha256_by_dataset"] != AOQT_TRAIN_PROVENANCE["qrels_sha256_by_dataset"] or policy["compatibility_digest"] != AOQT_TRAIN_PROVENANCE["compatibility_digest"]:
        raise GateContractError(f"{label}: native XPKG AOQT policy is not bound to pinned raw-v4 training provenance")
    return policy


PACKAGE_ROLE_NAMES = {
    "artifact",
    "embedding_manifest",
    "weights",
    "memory_plan",
    "tokenizer",
    "post_pool_transform",
    "checkpoint",
    "train_manifest",
    "train_profile",
}


def _derived_package_name(artifact_path: Path, role: str) -> str:
    stem = artifact_path.name
    if artifact_path.suffix:
        stem = stem[: -len(artifact_path.suffix)]
    suffix_by_role = {
        "artifact": artifact_path.name,
        "embedding_manifest": f"{stem}.embedding.mll",
        "weights": f"{stem}.weights.mll",
        "memory_plan": f"{stem}.memory.mll",
        "tokenizer": f"{stem}.tokenizer.mll",
        "post_pool_transform": f"{stem}.aoqt.json",
        "checkpoint": f"{stem}.embed-train.mll",
        "train_manifest": f"{stem}.train.mll",
        "train_profile": f"{stem}.train-profile.mll",
    }
    return suffix_by_role[role]


def _parse_xpkg_manifest(path: Path, artifact: dict[str, Any], role: str, label: str) -> dict[str, Any]:
    """Parse the native XPKG role/path/hash map and verify every sibling."""

    sections = _read_native_mll_sections(path, label)
    if b"XPKG" not in sections:
        raise GateContractError(f"{label}: native package manifest is missing XPKG role/hash map")
    strings = _mll_strings(sections[b"STRG"], f"{label}.STRG")
    head = _mll_head_metadata(sections[b"HEAD"], strings, f"{label}.HEAD")
    if head.get("manifest_version") != "manta/package/v0alpha1":
        raise GateContractError(f"{label}: package manifest version is not the native EOS package schema")
    if head.get("package_kind") not in {"embedding", "training"}:
        raise GateContractError(f"{label}: unsupported native package kind")
    body = sections[b"XPKG"]
    if len(body) < 4:
        raise GateContractError(f"{label}: XPKG section is truncated")
    count = int.from_bytes(body[:4], "little")
    if count < 1 or count > len(PACKAGE_ROLE_NAMES) or len(body) != 4 + count * 48:
        raise GateContractError(f"{label}: XPKG entry count/size mismatch")
    package_path = Path(artifact["path"])
    entries: list[dict[str, Any]] = []
    seen_roles: set[str] = set()
    seen_paths: set[str] = set()
    seen_hashes: set[str] = set()
    offset = 4
    for index in range(count):
        role_index = int.from_bytes(body[offset : offset + 4], "little")
        path_index = int.from_bytes(body[offset + 4 : offset + 8], "little")
        byte_count = int.from_bytes(body[offset + 8 : offset + 16], "little", signed=True)
        digest = body[offset + 16 : offset + 48].hex()
        offset += 48
        if role_index >= len(strings) or path_index >= len(strings):
            raise GateContractError(f"{label}: XPKG entry {index} has invalid STRG index")
        entry_role = strings[role_index]
        entry_name = strings[path_index]
        if entry_role not in PACKAGE_ROLE_NAMES or not entry_name or "/" in entry_name or "\\" in entry_name:
            raise GateContractError(f"{label}: XPKG entry {index} has an unsupported role/path")
        if byte_count < 0 or not SHA256_RE.fullmatch(digest):
            raise GateContractError(f"{label}: XPKG entry {index} has invalid size/hash")
        if entry_role in seen_roles or entry_name in seen_paths or digest in seen_hashes:
            raise GateContractError(f"{label}: XPKG roles, paths, and content identities must be unique")
        expected_name = _derived_package_name(package_path, entry_role)
        if entry_name != expected_name:
            raise GateContractError(f"{label}: XPKG {entry_role} path is not the derived sibling {expected_name!r}")
        sibling_path = package_path.parent / entry_name
        normalized = normalize_file_record({"path": str(sibling_path), "sha256": digest, "bytes": byte_count}, package_path.parent, f"{label}.XPKG.{entry_role}")
        entries.append({"role": entry_role, "name": entry_name, **normalized})
        seen_roles.add(entry_role)
        seen_paths.add(entry_name)
        seen_hashes.add(digest)
    required = {"artifact", "embedding_manifest", "weights", "memory_plan"}
    if not required.issubset(seen_roles):
        raise GateContractError(f"{label}: XPKG is missing required roles {sorted(required - seen_roles)}")
    if dict(next(item for item in entries if item["role"] == "artifact")).get("sha256") != artifact["sha256"]:
        raise GateContractError(f"{label}: XPKG artifact hash does not equal the frozen artifact")
    if head.get("file_count") != count:
        raise GateContractError(f"{label}: XPKG file_count HEAD metadata mismatch")
    if role == "candidate" and "post_pool_transform" not in seen_roles:
        raise GateContractError(f"{label}: candidate XPKG must bind an AOQT post_pool_transform sibling")
    if role == "anchor" and "post_pool_transform" in seen_roles:
        raise GateContractError(f"{label}: anchor XPKG must not bind an AOQT post_pool_transform sibling")
    aoqt_policy = _parse_xpkg_aoqt_policy(head, role, label)
    entries.sort(key=lambda item: item["role"])
    return {"schema": "native_xpkg.v1", "head": head, "entries": entries, "roles": {item["role"]: item for item in entries}, "aoqt_policy": aoqt_policy}


def _parse_authored_embedding_manifest(path: Path, role: str, label: str) -> dict[str, Any]:
    """Inspect the actual embedding manifest sibling, not its attestation."""

    sections = _read_native_mll_sections(path, label)
    strings = _mll_strings(sections[b"STRG"], f"{label}.STRG")
    head = _mll_head_metadata(sections[b"HEAD"], strings, f"{label}.HEAD")
    if head.get("manifest_kind") != "embedding_manifest" or head.get("manifest_version") != "manta/embedding-manifest/v0alpha1":
        raise GateContractError(f"{label}: actual sibling is not an EOS embedding manifest")
    if head.get("model_dim") != DIMENSION or head.get("output_dim") != DIMENSION:
        raise GateContractError(f"{label}: embedding manifest dimension is not D384")
    transform = head.get("post_pool_transform") or "none"
    expected_transform = "none" if role == "anchor" else AOQT_TOPOLOGY["id"]
    if transform != expected_transform:
        raise GateContractError(f"{label}: post_pool_transform is {transform!r}, expected {expected_transform!r}")
    return {"schema": "native_embedding_manifest.v1", "head": head, "post_pool_transform": transform}


def _validate_manifest_file(path: Path, package_sha: str, role: str, label: str, *, production: bool) -> dict[str, Any] | None:
    if path.read_bytes()[:4] == b"MLL\0":
        if role not in ROLES:
            raise GateContractError(f"{label}: package role is required before native manifest inspection")
        return _parse_authored_embedding_manifest(path, role, label)
    payload = read_json(path, label)
    if production and payload.get("schema") != NATIVE_PACKAGE_MANIFEST_SCHEMA:
        # The pinned current D384 anchor is accompanied by the train-only
        # stage2 anchor manifest rather than an XPKG package-manifest section.
        # It is admitted only by its exact immutable digest and exact audit
        # fields; arbitrary JSON package metadata remains rejected.
        if not (role == "anchor" and sha256_file(path) == D384_ANCHOR_AUDIT_MANIFEST_SHA256 and payload.get("schema") == "eos.aoqt_stage2.anchor_manifest.v1"):
            raise GateContractError(f"{label}: arbitrary JSON package manifest fallback is test-only")
        if payload.get("embedding_dim") != DIMENSION or payload.get("package_sha256") != package_sha:
            raise GateContractError(f"{label}: pinned anchor manifest identity mismatch")
        return None
    require_exact_keys(payload, {"schema", "role", "package_sha256", "dimension", "embedding_space_id", "post_pool_transform", "research_only", "topology", "transform", "anchor", "legal_scope", "sidecar_sha256"}, label)
    if payload["role"] != role or payload["package_sha256"] != package_sha or payload["dimension"] != DIMENSION:
        raise GateContractError(f"{label}: package manifest identity mismatch")
    return payload


def _validate_transform(value: Any, role: str, label: str) -> dict[str, Any] | None:
    if role == "anchor":
        if value is not None:
            raise GateContractError(f"{label}: anchor must not carry AOQT transform")
        return None
    transform = require_mapping(value, label)
    require_exact_keys(transform, {"id", "dim", "stages", "pairs_per_stage", "angle_count", "pairings_sha256", "angles_sha256", "angle_cap", "max_angle_cap"}, label)
    if transform["id"] != AOQT_TOPOLOGY["id"] or transform["dim"] != DIMENSION or transform["stages"] != AOQT_TOPOLOGY["stages"] or transform["pairs_per_stage"] != AOQT_TOPOLOGY["pairs_per_stage"] or transform["angle_count"] != AOQT_TOPOLOGY["angle_count"]:
        raise GateContractError(f"{label}: AOQT transform topology mismatch")
    pairings = require_sha256(transform["pairings_sha256"], f"{label}.pairings_sha256")
    angles = require_sha256(transform["angles_sha256"], f"{label}.angles_sha256")
    cap = require_finite_number(transform["angle_cap"], f"{label}.angle_cap")
    hard_cap = require_finite_number(transform["max_angle_cap"], f"{label}.max_angle_cap")
    if cap <= 0 or hard_cap != AOQT_TOPOLOGY["angle_cap_hard"] or cap > hard_cap or cap > AOQT_TOPOLOGY["angle_cap_default"]:
        raise GateContractError(f"{label}: AOQT angle cap mismatch")
    return {**{key: transform[key] for key in ("id", "dim", "stages", "pairs_per_stage", "angle_count")}, "pairings_sha256": pairings, "angles_sha256": angles, "angle_cap": cap, "max_angle_cap": hard_cap}


def _transform_pairings_sha256(stages: list[dict[str, Any]], label: str) -> str:
    digest = hashlib.sha256()
    for stage_index, stage in enumerate(stages):
        pairs = stage.get("pairs")
        if not isinstance(pairs, list):
            raise GateContractError(f"{label}.stages[{stage_index}].pairs: expected array")
        for pair_index, pair in enumerate(pairs):
            if not isinstance(pair, list) or len(pair) != 2 or any(isinstance(item, bool) or not isinstance(item, int) for item in pair):
                raise GateContractError(f"{label}.stages[{stage_index}].pairs[{pair_index}]: expected integer pair")
            digest.update(f"{pair[0]},{pair[1]}\n".encode("ascii"))
    return digest.hexdigest()


def _transform_angles_sha256(stages: list[dict[str, Any]], label: str) -> str:
    digest = hashlib.sha256()
    for stage_index, stage in enumerate(stages):
        angles = stage.get("angles")
        if not isinstance(angles, list):
            raise GateContractError(f"{label}.stages[{stage_index}].angles: expected array")
        for angle_index, angle in enumerate(angles):
            number = require_finite_number(angle, f"{label}.stages[{stage_index}].angles[{angle_index}]")
            # AOQTGivensTransform stores angles as float32 and Go hashes
            # fmt.Sprintf("%.9g\n", angle).  Re-round decoded JSON values
            # to float32 before formatting so Python cannot hash a wider
            # float64 representation differently.
            try:
                number32 = struct.unpack("<f", struct.pack("<f", number))[0]
            except (OverflowError, struct.error) as exc:
                raise GateContractError(f"{label}.stages[{stage_index}].angles[{angle_index}]: not representable as float32") from exc
            digest.update(f"{number32:.9g}\n".encode("ascii"))
    return digest.hexdigest()


def _parse_actual_aoqt_transform(path: Path, *, expected_anchor: dict[str, str], expected_policy: dict[str, Any], label: str) -> dict[str, Any]:
    """Parse the native AOQTGivensTransform JSON and bind it to XPKG policy.

    Training provenance lives in the native package manifest's
    AOQTTransformPolicy.  The sidecar itself intentionally contains only the
    runtime transform schema (plus optional audit data); requiring a second
    provenance object in the JSON would reject EOS output and create a
    circular package/sidecar binding.
    """

    payload = read_json(path, label)
    required = {"version", "kind", "dim", "seed", "angle_cap", "stages", "audit"}
    require_exact_keys(payload, required, label)
    if payload["version"] != "eos/aoqt-givens-transform/v1" or payload["kind"] != AOQT_TOPOLOGY["id"] or payload["dim"] != DIMENSION:
        raise GateContractError(f"{label}: actual AOQT transform identity mismatch")
    if payload["seed"] != 191:
        raise GateContractError(f"{label}: actual AOQT topology seed mismatch")
    angle_cap_raw = require_finite_number(payload["angle_cap"], f"{label}.angle_cap")
    try:
        angle_cap = struct.unpack("<f", struct.pack("<f", angle_cap_raw))[0]
    except (OverflowError, struct.error) as exc:
        raise GateContractError(f"{label}.angle_cap: not representable as float32") from exc
    if not 0.0 < angle_cap <= AOQT_TOPOLOGY["angle_cap_default"]:
        raise GateContractError(f"{label}: actual AOQT angle cap is outside the approved candidate policy")
    stages = payload["stages"]
    if not isinstance(stages, list) or len(stages) != AOQT_TOPOLOGY["stages"]:
        raise GateContractError(f"{label}: actual AOQT stage count mismatch")
    angle_count = 0
    nonzero = False
    max_abs = 0.0
    for stage_index, stage_value in enumerate(stages):
        stage = require_mapping(stage_value, f"{label}.stages[{stage_index}]")
        require_exact_keys(stage, {"pairs", "angles"}, f"{label}.stages[{stage_index}]")
        pairs = stage["pairs"]
        angles = stage["angles"]
        if not isinstance(pairs, list) or len(pairs) != AOQT_TOPOLOGY["pairs_per_stage"] or not isinstance(angles, list) or len(angles) != len(pairs):
            raise GateContractError(f"{label}.stages[{stage_index}]: pair/angle count mismatch")
        seen_coordinates: set[int] = set()
        for pair_index, pair in enumerate(pairs):
            if not isinstance(pair, list) or len(pair) != 2 or any(isinstance(item, bool) or not isinstance(item, int) for item in pair):
                raise GateContractError(f"{label}.stages[{stage_index}].pairs[{pair_index}]: invalid pair")
            left, right = pair
            if left < 0 or right < 0 or left >= DIMENSION or right >= DIMENSION or left == right or left in seen_coordinates or right in seen_coordinates:
                raise GateContractError(f"{label}.stages[{stage_index}].pairs[{pair_index}]: coordinates are not a disjoint D384 pairing")
            seen_coordinates.update((left, right))
            number_raw = require_finite_number(angles[pair_index], f"{label}.stages[{stage_index}].angles[{pair_index}]")
            try:
                number = struct.unpack("<f", struct.pack("<f", number_raw))[0]
            except (OverflowError, struct.error) as exc:
                raise GateContractError(f"{label}.stages[{stage_index}].angles[{pair_index}]: not representable as float32") from exc
            absolute = abs(number)
            max_abs = max(max_abs, absolute)
            nonzero = nonzero or absolute > 0.0
        angle_count += len(pairs)
    if angle_count != AOQT_TOPOLOGY["angle_count"] or not nonzero or max_abs > angle_cap + 1e-7:
        raise GateContractError(f"{label}: actual AOQT angle count/nonzero/cap evidence is invalid")
    audit = require_mapping(payload["audit"], f"{label}.audit")
    if set(audit) - {"pairings_sha256", "angles_sha256", "orthogonality_frobenius_per_dim"}:
        raise GateContractError(f"{label}.audit: unexpected audit fields")
    if set(audit) != {"pairings_sha256", "angles_sha256"} and set(audit) != {"pairings_sha256", "angles_sha256", "orthogonality_frobenius_per_dim"}:
        raise GateContractError(f"{label}.audit: pairings_sha256 and angles_sha256 are required")
    pairings_sha = _transform_pairings_sha256(stages, f"{label}.stages")
    angles_sha = _transform_angles_sha256(stages, f"{label}.stages")
    if require_sha256(audit["pairings_sha256"], f"{label}.audit.pairings_sha256") != pairings_sha or require_sha256(audit["angles_sha256"], f"{label}.audit.angles_sha256") != angles_sha:
        raise GateContractError(f"{label}: AOQT audit hashes do not bind actual stages")
    if "orthogonality_frobenius_per_dim" in audit:
        orthogonality = require_finite_number(audit["orthogonality_frobenius_per_dim"], f"{label}.audit.orthogonality_frobenius_per_dim")
        if orthogonality < 0:
            raise GateContractError(f"{label}.audit.orthogonality_frobenius_per_dim: must be non-negative")
    sidecar_sha = sha256_file(path)
    if sidecar_sha != expected_policy["transform_sha256"] or pairings_sha != expected_policy["pairings_sha256"] or angles_sha != expected_policy["angles_sha256"]:
        raise GateContractError(f"{label}: AOQT payload hashes do not match native XPKG policy")
    actual_anchor = {
        "artifact_sha256": expected_policy["anchor_artifact_sha256"],
        "package_manifest_sha256": expected_policy["anchor_package_manifest_sha256"],
        "embedding_space_id": expected_policy["anchor_embedding_space_id"],
    }
    if actual_anchor != expected_anchor:
        raise GateContractError(f"{label}: native XPKG AOQT anchor identity mismatch")
    return {"schema": "native_aoqt_transform.v1", "sha256": sidecar_sha, "seed": payload["seed"], "angle_cap": angle_cap, "stages": len(stages), "pairs_per_stage": AOQT_TOPOLOGY["pairs_per_stage"], "angle_count": angle_count, "pairings_sha256": pairings_sha, "angles_sha256": angles_sha, "max_angle_abs": max_abs, "orthogonality_frobenius_per_dim": audit.get("orthogonality_frobenius_per_dim")}


def _validate_sidecar(path: Path, *, expected_anchor: dict[str, str], transform: dict[str, Any], label: str, production: bool) -> dict[str, Any]:
    if production:
        # The caller passes the package/data policy only after the native
        # package manifest has been parsed.  A production sidecar without the
        # actual stages and cross-binding block is deliberately unusable.
        raise GateContractError(f"{label}: native AOQT transform payload requires package/data policy context")
    payload = read_json(path, label)
    require_no_forbidden_payload_keys(payload, label)
    require_exact_keys(payload, {"schema", "kind", "dim", "stages", "pairs_per_stage", "angle_count", "angle_cap", "max_angle_cap", "pairings_sha256", "angles_sha256", "anchor_package_sha256", "anchor_manifest_sha256", "anchor_embedding_space_id", "legal_scope"}, label)
    if payload["schema"] != SIDECAR_SCHEMA or payload["kind"] != AOQT_TOPOLOGY["id"]:
        raise GateContractError(f"{label}: AOQT sidecar schema/kind mismatch")
    topology = {key: payload[key] for key in ("dim", "stages", "pairs_per_stage", "angle_count", "angle_cap", "max_angle_cap")}
    _validate_transform({"id": payload["kind"], **topology, "pairings_sha256": payload["pairings_sha256"], "angles_sha256": payload["angles_sha256"]}, "candidate", "sidecar.transform")
    if payload["anchor_package_sha256"] != expected_anchor["package_sha256"] or payload["anchor_manifest_sha256"] != expected_anchor["manifest_sha256"] or payload["anchor_embedding_space_id"] != expected_anchor["embedding_space_id"]:
        raise GateContractError(f"{label}: sidecar anchor binding mismatch")
    _validate_legal_scope(payload["legal_scope"], f"{label}.legal_scope")
    if production and path.suffix.lower() != ".json":
        raise GateContractError(f"{label}: native sidecar policy must be JSON")
    return payload


def _normalize_sibling_rollup(value: Any, package_record_base: Path, role: str, package_sha: str, manifest_sha: str, attestation_path: Path, sidecar_path: Path | None, label: str, *, production: bool) -> tuple[list[dict[str, Any]], str]:
    rollup = require_mapping(value, label)
    require_exact_keys(rollup, {"entries", "sha256"}, label)
    entries_raw = rollup["entries"]
    if not isinstance(entries_raw, list) or not entries_raw:
        raise GateContractError(f"{label}: non-empty sibling rollup required")
    entries: list[dict[str, Any]] = []
    seen_paths: set[str] = set()
    seen_hashes: set[str] = set()
    seen_names: set[str] = set()
    for index, item in enumerate(entries_raw):
        entry = require_mapping(item, f"{label}.entries[{index}]")
        require_exact_keys(entry, {"name", "path", "sha256", "bytes"}, f"{label}.entries[{index}]")
        name = require_string(entry["name"], f"{label}.entries[{index}].name")
        path = resolve_path(entry["path"], package_record_base, f"{label}.entries[{index}].path")
        if path.name != name or name in {".", ".."} or "/" in name or "\\" in name:
            raise GateContractError(f"{label}.entries[{index}]: name/path mismatch")
        normalized = normalize_file_record({"path": str(path), "sha256": entry["sha256"], "bytes": entry["bytes"]}, package_record_base, f"{label}.entries[{index}]")
        if name in seen_names or str(path) in seen_paths or normalized["sha256"] in seen_hashes:
            raise GateContractError(f"{label}: sibling entries must have distinct paths and content identities")
        seen_names.add(name)
        seen_paths.add(str(path))
        seen_hashes.add(normalized["sha256"])
        entries.append({"name": name, **normalized})
    entries.sort(key=lambda item: item["name"])
    rollup_sha = require_sha256(rollup["sha256"], f"{label}.sha256")
    if rollup_sha != sha256_json(entries):
        raise GateContractError(f"{label}: sibling rollup digest mismatch")
    names = {entry["name"] for entry in entries}
    if role == "candidate" and sidecar_path is not None and sidecar_path.name not in names:
        raise GateContractError(f"{label}: candidate sidecar missing from sibling rollup")
    if production and len(entries) < 3:
        raise GateContractError(f"{label}: native package sibling rollup is incomplete")
    return entries, rollup_sha


def normalize_package_record(value: Any, base_dir: Path, label: str, *, expected_role: str | None = None, expected_anchor: dict[str, str] | None = None, data_binding: dict[str, Any] | None = None, expected_compatibility_digest: str | None = None, production: bool = True) -> dict[str, Any]:
    record = require_mapping(value, label)
    record_keys = {"path", "sha256", "bytes", "manifest", "attestation", "sibling_rollup_sha256"}
    if production:
        # ``manifest`` is the immutable audit identity (the pinned D384
        # anchor uses its stage-2 manifest); ``package_manifest`` is the
        # native EOS .package.mll containing the XPKG role/path/hash map.
        # Keeping both avoids confusing an audit record with runtime package
        # identity and prevents a free-standing rollup from substituting it.
        record_keys.add("package_manifest")
    require_exact_keys(record, record_keys, label)
    package = normalize_file_record({key: record[key] for key in ("path", "sha256", "bytes")}, base_dir, f"{label}.package")
    package_path = Path(package["path"])
    if production:
        _native_mll_section_tags(package_path, f"{label}.package")
    manifest = normalize_file_record(record["manifest"], base_dir, f"{label}.manifest")
    manifest_payload = _validate_manifest_file(Path(manifest["path"]), package["sha256"], expected_role or "", f"{label}.manifest", production=production)
    xpkg_manifest: dict[str, Any] | None = None
    xpkg_record: dict[str, Any] | None = None
    if production:
        xpkg_record = normalize_file_record(record["package_manifest"], base_dir, f"{label}.package_manifest")
        if xpkg_record["path"] in {package["path"], manifest["path"]}:
            raise GateContractError(f"{label}: native package manifest path must be distinct")
        xpkg_manifest = _parse_xpkg_manifest(Path(xpkg_record["path"]), package, expected_role or "", f"{label}.package_manifest")
        embedding_entry = xpkg_manifest["roles"]["embedding_manifest"]
        _parse_authored_embedding_manifest(Path(embedding_entry["path"]), expected_role or "", f"{label}.package_manifest.embedding_manifest")
    attestation = normalize_file_record(record["attestation"], base_dir, f"{label}.attestation")
    distinct_paths = {package["path"], manifest["path"], attestation["path"]}
    if xpkg_record is not None:
        distinct_paths.add(xpkg_record["path"])
    if len(distinct_paths) != (4 if xpkg_record is not None else 3):
        raise GateContractError(f"{label}: package, manifest, and attestation paths must differ")
    policy = read_json(Path(attestation["path"]), f"{label}.attestation")
    require_exact_keys(policy, {"schema", "role", "package_sha256", "manifest_sha256", "dimension", "embedding_space_id", "post_pool_transform", "research_only", "topology", "transform", "anchor", "legal_scope", "sidecar", "sibling_rollup", "dataset_binding", "compatibility_digest", "identity"}, f"{label}.attestation")
    if policy["schema"] != PACKAGE_ATTESTATION_SCHEMA:
        raise GateContractError(f"{label}: native package attestation schema mismatch")
    role = require_string(policy["role"], f"{label}.attestation.role", safe_id=True)
    if expected_role is not None and role != expected_role:
        raise GateContractError(f"{label}: package role mismatch")
    if policy["package_sha256"] != package["sha256"] or policy["manifest_sha256"] != manifest["sha256"] or policy["dimension"] != DIMENSION:
        raise GateContractError(f"{label}: package attestation hash/dimension binding mismatch")
    embedding_space_id = require_sha256(policy["embedding_space_id"], f"{label}.attestation.embedding_space_id")
    topology = _validate_topology(policy["topology"], f"{label}.attestation.topology")
    legal_scope = _validate_legal_scope(policy["legal_scope"], f"{label}.attestation.legal_scope")
    transform = _validate_transform(policy["transform"], role, f"{label}.attestation.transform")
    if role == "anchor":
        if policy["post_pool_transform"] != "none" or bool(policy["research_only"]):
            raise GateContractError(f"{label}: anchor post-pool/research policy mismatch")
    else:
        if policy["post_pool_transform"] != AOQT_TOPOLOGY["id"] or policy["research_only"] is not True or transform is None:
            raise GateContractError(f"{label}: candidate AOQT post-pool policy mismatch")
    if role == "candidate":
        sidecar_raw = require_mapping(policy["sidecar"], f"{label}.attestation.sidecar")
        sidecar = normalize_file_record(sidecar_raw, Path(attestation["path"]).parent, f"{label}.attestation.sidecar")
        sidecar_path: Path | None = Path(sidecar["path"])
    else:
        if policy["sidecar"] is not None:
            raise GateContractError(f"{label}: anchor sidecar must be null")
        sidecar = None
        sidecar_path = None
    if manifest_payload is not None:
        if manifest_payload.get("schema") == "native_embedding_manifest.v1":
            # Actual embedding-manifest metadata is authoritative for D384
            # and post-pool topology.  Provenance policy is checked below.
            if manifest_payload["post_pool_transform"] != policy["post_pool_transform"]:
                raise GateContractError(f"{label}: native embedding manifest/attestation transform mismatch")
        else:
            if manifest_payload["embedding_space_id"] != embedding_space_id or manifest_payload["post_pool_transform"] != policy["post_pool_transform"] or manifest_payload["research_only"] != policy["research_only"]:
                raise GateContractError(f"{label}: package manifest/attestation mismatch")
            manifest_anchor_ok = manifest_payload["anchor"] == policy["anchor"] or (role == "anchor" and manifest_payload["anchor"] is None)
            expected_sidecar_sha = None if sidecar is None else sidecar["sha256"]
            if manifest_payload["sidecar_sha256"] != expected_sidecar_sha:
                raise GateContractError(f"{label}: package manifest/sidecar hash mismatch")
            if manifest_payload["topology"] != topology or manifest_payload["transform"] != transform or not manifest_anchor_ok or manifest_payload["legal_scope"] != legal_scope:
                raise GateContractError(f"{label}: package manifest policy mismatch")
    identity = require_mapping(policy["identity"], f"{label}.attestation.identity")
    require_exact_keys(identity, {"artifact_sha256", "package_manifest_sha256", "embedding_space_id"}, f"{label}.attestation.identity")
    identity = {"artifact_sha256": require_sha256(identity["artifact_sha256"], f"{label}.identity.artifact_sha256"), "package_manifest_sha256": require_sha256(identity["package_manifest_sha256"], f"{label}.identity.package_manifest_sha256"), "embedding_space_id": require_sha256(identity["embedding_space_id"], f"{label}.identity.embedding_space_id")}
    # The package identity must name the actual native XPKG manifest.  The
    # separate ``manifest`` record is an audit/embedding-manifest attachment
    # and is not sufficient to describe the runtime's sibling graph.
    package_manifest_sha256 = manifest["sha256"] if xpkg_record is None else xpkg_record["sha256"]
    if identity != {"artifact_sha256": package["sha256"], "package_manifest_sha256": package_manifest_sha256, "embedding_space_id": embedding_space_id}:
        raise GateContractError(f"{label}: package identity self-binding mismatch")
    anchor_raw = require_mapping(policy["anchor"], f"{label}.attestation.anchor")
    require_exact_keys(anchor_raw, {"package_sha256", "manifest_sha256", "embedding_space_id"}, f"{label}.attestation.anchor")
    anchor_binding = {"package_sha256": require_sha256(anchor_raw["package_sha256"], f"{label}.anchor.package_sha256"), "manifest_sha256": require_sha256(anchor_raw["manifest_sha256"], f"{label}.anchor.manifest_sha256"), "embedding_space_id": require_sha256(anchor_raw["embedding_space_id"], f"{label}.anchor.embedding_space_id")}
    anchor_identity_view = {"package_sha256": identity["artifact_sha256"], "manifest_sha256": identity["package_manifest_sha256"], "embedding_space_id": identity["embedding_space_id"]}
    if role == "anchor" and anchor_binding != anchor_identity_view:
        raise GateContractError(f"{label}: anchor must self-bind as its own anchor identity")
    expected_anchor_view = None if expected_anchor is None else {"package_sha256": expected_anchor["artifact_sha256"], "manifest_sha256": expected_anchor["package_manifest_sha256"], "embedding_space_id": expected_anchor["embedding_space_id"]}
    if role == "candidate" and expected_anchor_view is not None and anchor_binding != expected_anchor_view:
        raise GateContractError(f"{label}: candidate anchor identity mismatch")
    if role == "candidate":
        sidecar_anchor = anchor_binding if expected_anchor is None else {"package_sha256": expected_anchor["artifact_sha256"], "manifest_sha256": expected_anchor["package_manifest_sha256"], "embedding_space_id": expected_anchor["embedding_space_id"]}
        if production:
            if xpkg_manifest is None or "post_pool_transform" not in xpkg_manifest["roles"]:
                raise GateContractError(f"{label}: candidate native XPKG transform sibling is missing")
            transform_entry = xpkg_manifest["roles"]["post_pool_transform"]
            if Path(transform_entry["path"]) != Path(sidecar["path"]) or transform_entry["sha256"] != sidecar["sha256"]:
                raise GateContractError(f"{label}: AOQT sidecar is not the XPKG-derived transform sibling")
            xpkg_policy = xpkg_manifest.get("aoqt_policy")
            if not isinstance(xpkg_policy, dict):
                raise GateContractError(f"{label}: candidate native XPKG AOQT policy is missing")
            native_expected_anchor = {
                "artifact_sha256": sidecar_anchor["package_sha256"],
                "package_manifest_sha256": sidecar_anchor["manifest_sha256"],
                "embedding_space_id": sidecar_anchor["embedding_space_id"],
            }
            actual_transform = _parse_actual_aoqt_transform(
                Path(sidecar["path"]),
                expected_anchor=native_expected_anchor,
                expected_policy=xpkg_policy,
                label=f"{label}.sidecar",
            )
            if actual_transform["sha256"] != sidecar["sha256"]:
                raise GateContractError(f"{label}: actual AOQT sidecar hash changed during inspection")
            try:
                declared_angle_cap = struct.unpack("<f", struct.pack("<f", float(transform.get("angle_cap", -1.0))))[0] if transform is not None else -1.0
            except (OverflowError, struct.error, TypeError, ValueError) as exc:
                raise GateContractError(f"{label}: declared AOQT angle cap is not float32-representable") from exc
            if transform is None or transform.get("pairings_sha256") != actual_transform["pairings_sha256"] or transform.get("angles_sha256") != actual_transform["angles_sha256"] or declared_angle_cap != actual_transform["angle_cap"]:
                raise GateContractError(f"{label}: declared AOQT transform does not equal actual sidecar payload")
            transform = {**transform, "seed": actual_transform["seed"], "max_angle_abs": actual_transform["max_angle_abs"], "payload_sha256": actual_transform["sha256"]}
        else:
            _validate_sidecar(Path(sidecar["path"]), expected_anchor=sidecar_anchor, transform=transform or {}, label=f"{label}.sidecar", production=production)
    binding = require_mapping(policy["dataset_binding"], f"{label}.attestation.dataset_binding")
    require_exact_keys(binding, {"dataset_manifest_sha256_by_domain", "corpus_sha256_by_domain", "queries_sha256_by_domain", "qrels_sha256_by_domain"}, f"{label}.attestation.dataset_binding")
    dataset_manifest_binding = require_domain_mapping(binding["dataset_manifest_sha256_by_domain"], f"{label}.dataset_binding.dataset_manifest_sha256_by_domain")
    corpus_binding = require_domain_mapping(binding["corpus_sha256_by_domain"], f"{label}.dataset_binding.corpus_sha256_by_domain")
    queries_binding = require_domain_mapping(binding["queries_sha256_by_domain"], f"{label}.dataset_binding.queries_sha256_by_domain")
    qrels_binding = require_domain_mapping(binding["qrels_sha256_by_domain"], f"{label}.dataset_binding.qrels_sha256_by_domain")
    normalized_binding = {
        "dataset_manifest_sha256_by_domain": {domain: require_sha256(dataset_manifest_binding[domain], f"{label}.dataset_binding.dataset_manifest_sha256_by_domain.{domain}") for domain in DOMAINS},
        "corpus_sha256_by_domain": {domain: require_sha256(corpus_binding[domain], f"{label}.dataset_binding.corpus_sha256_by_domain.{domain}") for domain in DOMAINS},
        "queries_sha256_by_domain": {domain: require_sha256(queries_binding[domain], f"{label}.dataset_binding.queries_sha256_by_domain.{domain}") for domain in DOMAINS},
        "qrels_sha256_by_domain": {domain: require_sha256(qrels_binding[domain], f"{label}.dataset_binding.qrels_sha256_by_domain.{domain}") for domain in DOMAINS},
    }
    if data_binding is not None and normalized_binding != data_binding:
        raise GateContractError(f"{label}: AOQT qrels/dataset binding mismatch")
    compatibility_digest = require_sha256(policy["compatibility_digest"], f"{label}.attestation.compatibility_digest")
    if expected_compatibility_digest is not None and compatibility_digest != expected_compatibility_digest:
        raise GateContractError(f"{label}: AOQT compatibility digest mismatch")
    sibling_entries, sibling_sha = _normalize_sibling_rollup(policy["sibling_rollup"], Path(attestation["path"]).parent, role, package["sha256"], manifest["sha256"], Path(attestation["path"]), sidecar_path, f"{label}.attestation.sibling_rollup", production=production)
    bound_sibling_sha = require_sha256(record["sibling_rollup_sha256"], f"{label}.sibling_rollup_sha256")
    if bound_sibling_sha != sibling_sha:
        raise GateContractError(f"{label}: package sibling rollup binding mismatch")
    if production:
        if xpkg_manifest is None:
            raise GateContractError(f"{label}: native XPKG identity is required")
        expected_rollup = sorted(
            ({"name": entry["name"], "path": entry["path"], "sha256": entry["sha256"], "bytes": entry["bytes"]} for entry in xpkg_manifest["entries"]),
            key=lambda item: item["name"],
        )
        actual_rollup = sorted(
            ({"name": entry["name"], "path": entry["path"], "sha256": entry["sha256"], "bytes": entry["bytes"]} for entry in sibling_entries),
            key=lambda item: item["name"],
        )
        if actual_rollup != expected_rollup:
            raise GateContractError(f"{label}: caller sibling rollup does not equal native XPKG role/hash map")
    sibling_paths = {entry["path"] for entry in sibling_entries}
    # The attestation is the canonical evaluation policy being validated and is
    # intentionally outside its own rollup (including its own digest would
    # create an impossible circular content hash).  Every package payload and
    # sidecar, however, must be present in the sibling rollup.
    required_sibling_paths = {package["path"], manifest["path"]}
    if xpkg_record is not None:
        required_sibling_paths = {entry["path"] for entry in xpkg_manifest["entries"]}
        if not production:
            required_sibling_paths.update({package["path"], manifest["path"]})
    if sidecar is not None:
        required_sibling_paths.add(sidecar["path"])
    if not required_sibling_paths.issubset(sibling_paths):
        raise GateContractError(f"{label}: sibling rollup omits a bound package component")
    return {
        **package,
        "manifest": manifest,
        "package_manifest": xpkg_record,
        "xpkg_manifest_sha256": None if xpkg_record is None else xpkg_record["sha256"],
        "xpkg": xpkg_manifest,
        "attestation": attestation,
        "sibling_rollup_sha256": sibling_sha,
        "role": role,
        "embedding_space_id": embedding_space_id,
        "package_identity": identity,
        "anchor_identity": {"artifact_sha256": anchor_binding["package_sha256"], "package_manifest_sha256": anchor_binding["manifest_sha256"], "embedding_space_id": anchor_binding["embedding_space_id"]},
        "topology": topology,
        "transform": transform,
        "legal_scope": legal_scope,
        "compatibility_digest": compatibility_digest,
        "sidecar": sidecar,
        "sibling_entries": sibling_entries,
        "attestation_policy": policy,
    }


def validate_package_pair(anchor: dict[str, Any], candidate: dict[str, Any]) -> None:
    anchor_identity = anchor.get("package_identity") or {}
    candidate_identity = candidate.get("package_identity") or {}
    # AOQT candidate construction intentionally copies the immutable anchor
    # artifact/weights.  Artifact SHA equality is therefore expected; the
    # native XPKG role graph and resulting embedding-space identity distinguish
    # the transformed candidate.  A reused package manifest or full identity
    # still indicates a no-op candidate and remains forbidden.
    if anchor_identity == candidate_identity or anchor_identity.get("package_manifest_sha256") == candidate_identity.get("package_manifest_sha256"):
        raise GateContractError("anchor and candidate package identities must be distinct")
    if candidate.get("anchor_identity") != anchor_identity:
        raise GateContractError("candidate anchor identity does not equal the bound anchor")


def _public_package_record(package: dict[str, Any]) -> dict[str, Any]:
    """Strip derived validator state before serializing a frozen manifest."""

    public = {
        "path": package["path"],
        "sha256": package["sha256"],
        "bytes": package["bytes"],
        "manifest": package["manifest"],
        "attestation": package["attestation"],
        "sibling_rollup_sha256": package["sibling_rollup_sha256"],
    }
    if package.get("package_manifest") is not None:
        public["package_manifest"] = package["package_manifest"]
    return public


APPROVED_DESCRIPTOR_KEYS = {
    "schema", "gate_id", "descriptor_id", "split", "dimension", "domains", "query_ids_by_domain", "qid_set_sha256_by_domain", "query_count_by_domain", "qrels_sha256_by_domain", "qrels_query_count_by_domain", "relevant_pair_count_by_domain", "dataset_manifest_sha256_by_domain", "corpus_sha256_by_domain", "queries_sha256_by_domain", "compatibility_digest", "nfcorpus_boundary_qids", "nfcorpus_boundary_qids_sha256", "nfcorpus_boundary_rank_window", "metric_surfaces", "cutoffs", "turboquant"
}


def validate_approved_workload(path: Path, *, gate_id: str, qrels: dict[str, dict[str, Any]], datasets: dict[str, dict[str, Any]], label: str) -> dict[str, Any]:
    payload = read_json(path, label)
    require_exact_keys(payload, APPROVED_DESCRIPTOR_KEYS, label)
    if payload["schema"] != APPROVED_WORKLOAD_SCHEMA or payload["gate_id"] != gate_id or payload["split"] != HELDOUT_SPLIT or payload["dimension"] != DIMENSION:
        raise GateContractError(f"{label}: approved workload identity mismatch")
    if payload["domains"] != list(DOMAINS):
        raise GateContractError(f"{label}: approved workload domain order mismatch")
    qids = normalize_qids_by_domain(payload["query_ids_by_domain"], f"{label}.query_ids_by_domain")
    if payload["qid_set_sha256_by_domain"] != qids_sha256_by_domain(qids):
        raise GateContractError(f"{label}: qid-set hash mismatch")
    qrels_hashes = require_domain_mapping(payload["qrels_sha256_by_domain"], f"{label}.qrels_sha256_by_domain")
    query_counts = require_domain_mapping(payload["query_count_by_domain"], f"{label}.query_count_by_domain")
    qrel_counts = require_domain_mapping(payload["qrels_query_count_by_domain"], f"{label}.qrels_query_count_by_domain")
    relevant_counts = require_domain_mapping(payload["relevant_pair_count_by_domain"], f"{label}.relevant_pair_count_by_domain")
    dataset_hashes = require_domain_mapping(payload["dataset_manifest_sha256_by_domain"], f"{label}.dataset_manifest_sha256_by_domain")
    corpus_hashes = require_domain_mapping(payload["corpus_sha256_by_domain"], f"{label}.corpus_sha256_by_domain")
    queries_hashes = require_domain_mapping(payload["queries_sha256_by_domain"], f"{label}.queries_sha256_by_domain")
    for domain in DOMAINS:
        actual = qrels[domain]
        if qids[domain] != actual["qids"]:
            raise GateContractError(f"{label}: approved qid allowlist does not equal complete {domain} qrels")
        if require_sha256(qrels_hashes[domain], f"{label}.qrels_sha256_by_domain.{domain}") != actual["sha256"] or require_integer(query_counts[domain], f"{label}.query_count_by_domain.{domain}", minimum=1) != actual["query_count"] or require_integer(qrel_counts[domain], f"{label}.qrels_query_count_by_domain.{domain}", minimum=1) != actual["query_count"] or require_integer(relevant_counts[domain], f"{label}.relevant_pair_count_by_domain.{domain}", minimum=1) != actual["relevant_pair_count"]:
            raise GateContractError(f"{label}: qrels content/count binding mismatch for {domain}")
        if require_sha256(dataset_hashes[domain], f"{label}.dataset_manifest_sha256_by_domain.{domain}") != datasets[domain]["manifest"]["sha256"] or require_sha256(corpus_hashes[domain], f"{label}.corpus_sha256_by_domain.{domain}") != datasets[domain]["corpus"]["sha256"] or require_sha256(queries_hashes[domain], f"{label}.queries_sha256_by_domain.{domain}") != datasets[domain]["queries"]["sha256"]:
            raise GateContractError(f"{label}: dataset identity binding mismatch for {domain}")
    compatibility_digest = require_sha256(payload["compatibility_digest"], f"{label}.compatibility_digest")
    boundary_qids = [require_qid(item, f"{label}.nfcorpus_boundary_qids") for item in payload["nfcorpus_boundary_qids"]] if isinstance(payload["nfcorpus_boundary_qids"], list) else None
    if not boundary_qids or len(boundary_qids) != len(set(boundary_qids)) or not set(boundary_qids).issubset(set(qids["nfcorpus"])):
        raise GateContractError(f"{label}: nonempty NFCorpus boundary qid allowlist is required")
    if payload["nfcorpus_boundary_qids_sha256"] != sha256_json(sorted(boundary_qids)) or payload["nfcorpus_boundary_rank_window"] != [80, 120]:
        raise GateContractError(f"{label}: NFCorpus boundary descriptor mismatch")
    if payload["metric_surfaces"] != list(SURFACES) or payload["cutoffs"] != {"ndcg_at_10": 10, "recall_at_100": 100}:
        raise GateContractError(f"{label}: metric surface/cutoff descriptor mismatch")
    if payload["turboquant"] != {"q3_bits": Q3_BITS, "q5_bits": Q5_BITS, "seed": TURBOQUANT_SEED, "top_k": TOP_K, "score_mode": "turboquant_ip_prepared", "package_mode": "native_mll_sibling"}:
        raise GateContractError(f"{label}: TurboQuant descriptor mismatch")
    return {
        "schema": APPROVED_WORKLOAD_SCHEMA,
        "gate_id": gate_id,
        "descriptor_id": require_string(payload["descriptor_id"], f"{label}.descriptor_id", safe_id=True),
        "split": HELDOUT_SPLIT,
        "dimension": DIMENSION,
        "domains": list(DOMAINS),
        "query_ids_by_domain": qids,
        "qid_set_sha256_by_domain": qids_sha256_by_domain(qids),
        "query_count_by_domain": {domain: qrels[domain]["query_count"] for domain in DOMAINS},
        "qrels_sha256_by_domain": {domain: qrels[domain]["sha256"] for domain in DOMAINS},
        "qrels_query_count_by_domain": {domain: qrels[domain]["query_count"] for domain in DOMAINS},
        "relevant_pair_count_by_domain": {domain: qrels[domain]["relevant_pair_count"] for domain in DOMAINS},
        "dataset_manifest_sha256_by_domain": {domain: datasets[domain]["manifest"]["sha256"] for domain in DOMAINS},
        "corpus_sha256_by_domain": {domain: datasets[domain]["corpus"]["sha256"] for domain in DOMAINS},
        "queries_sha256_by_domain": {domain: datasets[domain]["queries"]["sha256"] for domain in DOMAINS},
        "compatibility_digest": compatibility_digest,
        "nfcorpus_boundary_qids": sorted(boundary_qids),
        "nfcorpus_boundary_qids_sha256": sha256_json(sorted(boundary_qids)),
        "nfcorpus_boundary_rank_window": [80, 120],
        "metric_surfaces": list(SURFACES),
        "cutoffs": {"ndcg_at_10": 10, "recall_at_100": 100},
        "turboquant": {"q3_bits": Q3_BITS, "q5_bits": Q5_BITS, "seed": TURBOQUANT_SEED, "top_k": TOP_K, "score_mode": "turboquant_ip_prepared", "package_mode": "native_mll_sibling"},
    }


def validate_workload(path: Path, *, gate_id: str, approved: dict[str, Any], descriptor_sha256: str, label: str) -> dict[str, Any]:
    payload = read_json(path, label)
    expected_keys = {"schema", "gate_id", "workload_id", "descriptor_sha256", "split", "dimension", "query_ids_by_domain", "qid_set_sha256_by_domain", "query_count_by_domain", "qrels_sha256_by_domain", "dataset_manifest_sha256_by_domain", "corpus_sha256_by_domain", "queries_sha256_by_domain", "compatibility_digest"}
    require_exact_keys(payload, expected_keys, label)
    if payload["schema"] != WORKLOAD_SCHEMA or payload["gate_id"] != gate_id or payload["descriptor_sha256"] != descriptor_sha256:
        raise GateContractError(f"{label}: evaluator workload descriptor binding mismatch")
    normalized_qids = normalize_qids_by_domain(payload["query_ids_by_domain"], f"{label}.query_ids_by_domain")
    projection = {
        "schema": WORKLOAD_SCHEMA,
        "gate_id": gate_id,
        "workload_id": require_string(payload["workload_id"], f"{label}.workload_id", safe_id=True),
        "descriptor_sha256": descriptor_sha256,
        "split": payload["split"],
        "dimension": payload["dimension"],
        "query_ids_by_domain": normalized_qids,
        "qid_set_sha256_by_domain": payload["qid_set_sha256_by_domain"],
        "query_count_by_domain": payload["query_count_by_domain"],
        "qrels_sha256_by_domain": payload["qrels_sha256_by_domain"],
        "dataset_manifest_sha256_by_domain": payload["dataset_manifest_sha256_by_domain"],
        "corpus_sha256_by_domain": payload["corpus_sha256_by_domain"],
        "queries_sha256_by_domain": payload["queries_sha256_by_domain"],
        "compatibility_digest": payload["compatibility_digest"],
    }
    expected = {key: value for key, value in projection.items() if key not in {"schema", "workload_id"}}
    approved_projection = {
        "gate_id": approved["gate_id"], "descriptor_sha256": descriptor_sha256, "split": approved["split"], "dimension": approved["dimension"], "query_ids_by_domain": approved["query_ids_by_domain"], "qid_set_sha256_by_domain": approved["qid_set_sha256_by_domain"], "query_count_by_domain": approved["query_count_by_domain"], "qrels_sha256_by_domain": approved["qrels_sha256_by_domain"], "dataset_manifest_sha256_by_domain": approved["dataset_manifest_sha256_by_domain"], "corpus_sha256_by_domain": approved["corpus_sha256_by_domain"], "queries_sha256_by_domain": approved["queries_sha256_by_domain"], "compatibility_digest": approved["compatibility_digest"]
    }
    if expected != approved_projection:
        raise GateContractError(f"{label}: workload does not equal approved full-qrels descriptor")
    return projection


def normalize_source(value: Any, base_dir: Path, label: str) -> dict[str, Any]:
    source = require_mapping(value, label)
    require_exact_keys(source, {"manifest", "binary", "cwd", "dataset_by_domain", "workload", "approved_workload"}, label)
    manifest = normalize_file_record(source["manifest"], base_dir, f"{label}.manifest")
    binary = normalize_file_record(source["binary"], base_dir, f"{label}.binary", executable=True)
    cwd = normalize_directory(source["cwd"], base_dir, f"{label}.cwd")
    dataset_map = require_mapping(source["dataset_by_domain"], f"{label}.dataset_by_domain")
    if set(dataset_map) != set(DOMAINS):
        raise GateContractError(f"{label}.dataset_by_domain: exact domain coverage required")
    datasets = {domain: normalize_dataset_record(dataset_map[domain], base_dir, domain, f"{label}.dataset_by_domain.{domain}") for domain in DOMAINS}
    workload = normalize_file_record(source["workload"], base_dir, f"{label}.workload")
    approved_workload = normalize_file_record(source["approved_workload"], base_dir, f"{label}.approved_workload")
    return {"manifest": manifest, "binary": binary, "cwd": cwd, "dataset_by_domain": datasets, "workload": workload, "approved_workload": approved_workload}


def _data_binding(approved: dict[str, Any]) -> dict[str, Any]:
    return {"dataset_manifest_sha256_by_domain": approved["dataset_manifest_sha256_by_domain"], "corpus_sha256_by_domain": approved["corpus_sha256_by_domain"], "queries_sha256_by_domain": approved["queries_sha256_by_domain"], "qrels_sha256_by_domain": approved["qrels_sha256_by_domain"]}


PLAN_KEYS = {"schema", "gate_id", "candidate_id", "dimension", "thresholds", "evaluation", "anchor", "candidate", "source", "approved_workload", "qrels", "exclusions", "workload_root", "output_root"}


def _require_production_registry() -> None:
    registry = PRODUCTION_TRUSTED_WORKLOAD_REGISTRY
    if registry.get("schema") != TRUSTED_WORKLOAD_REGISTRY_SCHEMA or registry.get("status") != "active":
        reason = registry.get("reason", "trusted workload registry is unavailable")
        raise GateContractError(f"production heldout gate is unavailable: {reason}; caller-selected workload is forbidden")
    domains = registry.get("domains")
    boundary = registry.get("nfcorpus_boundary")
    if not isinstance(domains, dict) or set(domains) != set(DOMAINS) or not isinstance(boundary, dict):
        raise GateContractError("production trusted workload registry is malformed")


def _registry_path_matches(path: Path, suffix: Any, label: str) -> None:
    """Bind a production artifact to its fixed registry-relative path."""

    expected = require_string(suffix, label).replace("\\", "/").lstrip("/")
    actual = path.resolve(strict=False).as_posix()
    if actual != expected and not actual.endswith("/" + expected):
        raise GateContractError(f"{label}: path is not the fixed trusted registry artifact: {path}")


def _validate_production_registry_binding(
    approved: dict[str, Any],
    qrels: dict[str, dict[str, Any]],
    datasets: dict[str, dict[str, Any]],
    exclusions: dict[str, dict[str, Any]] | list[dict[str, Any]],
    *,
    label: str,
) -> None:
    """Require production inputs to equal the independently fixed registry."""

    registry = PRODUCTION_TRUSTED_WORKLOAD_REGISTRY
    domains = registry.get("domains")
    boundary = registry.get("nfcorpus_boundary")
    if not isinstance(domains, dict) or not isinstance(boundary, dict):
        raise GateContractError(f"{label}: production trusted workload registry is malformed")
    if set(domains) != set(DOMAINS):
        raise GateContractError(f"{label}: production trusted workload registry domain coverage is malformed")

    for domain in DOMAINS:
        spec = require_mapping(domains[domain], f"{label}.registry.domains.{domain}")
        required = {
            "qrels_path_suffix", "qrels_sha256", "qid_set_sha256", "query_count",
            "qrels_pair_count", "corpus_path_suffix", "corpus_sha256",
            "queries_path_suffix", "queries_sha256",
        }
        if set(spec) != required:
            raise GateContractError(f"{label}.registry.domains.{domain}: fixed identity fields are malformed")
        actual_qrels = qrels[domain]
        actual_dataset = datasets[domain]
        _registry_path_matches(Path(actual_qrels["path"]), spec["qrels_path_suffix"], f"{label}.{domain}.qrels")
        _registry_path_matches(Path(actual_dataset["corpus"]["path"]), spec["corpus_path_suffix"], f"{label}.{domain}.corpus")
        _registry_path_matches(Path(actual_dataset["queries"]["path"]), spec["queries_path_suffix"], f"{label}.{domain}.queries")
        expected_sha = require_sha256(spec["qrels_sha256"], f"{label}.registry.{domain}.qrels_sha256")
        expected_qid_sha = require_sha256(spec["qid_set_sha256"], f"{label}.registry.{domain}.qid_set_sha256")
        expected_corpus_sha = require_sha256(spec["corpus_sha256"], f"{label}.registry.{domain}.corpus_sha256")
        expected_queries_sha = require_sha256(spec["queries_sha256"], f"{label}.registry.{domain}.queries_sha256")
        expected_query_count = require_integer(spec["query_count"], f"{label}.registry.{domain}.query_count", minimum=1)
        expected_pair_count = require_integer(spec["qrels_pair_count"], f"{label}.registry.{domain}.qrels_pair_count", minimum=1)
        if (
            actual_qrels["sha256"] != expected_sha
            or actual_qrels["qid_set_sha256"] != expected_qid_sha
            or actual_qrels["query_count"] != expected_query_count
            or actual_qrels["qrels_pair_count"] != expected_pair_count
        ):
            raise GateContractError(f"{label}: registry qrels identity/count mismatch for {domain}")
        if actual_dataset["corpus"]["sha256"] != expected_corpus_sha or actual_dataset["queries"]["sha256"] != expected_queries_sha:
            raise GateContractError(f"{label}: registry corpus/query identity mismatch for {domain}")
        # Check the serialized approved descriptor too. The normalized return
        # value is derived from qrels/datasets, so this catches a frozen
        # descriptor that was edited after source validation.
        if (
            approved["query_ids_by_domain"][domain] != actual_qrels["qids"]
            or approved["qid_set_sha256_by_domain"][domain] != expected_qid_sha
            or approved["query_count_by_domain"][domain] != expected_query_count
            or approved["qrels_sha256_by_domain"][domain] != expected_sha
            or approved["qrels_query_count_by_domain"][domain] != expected_query_count
            or approved["corpus_sha256_by_domain"][domain] != expected_corpus_sha
            or approved["queries_sha256_by_domain"][domain] != expected_queries_sha
        ):
            raise GateContractError(f"{label}: approved descriptor is not the fixed registry workload for {domain}")

    if isinstance(exclusions, dict):
        official = exclusions.get("official-test")
    else:
        official = next((item for item in exclusions if item.get("name") == "official-test"), None)
    if not isinstance(official, dict):
        raise GateContractError(f"{label}: fixed official-test exclusion manifest is required")
    boundary_required = {"source_path_suffix", "source_sha256", "qid_set_sha256", "qid_count", "rank_window"}
    if set(boundary) != boundary_required:
        raise GateContractError(f"{label}.registry.nfcorpus_boundary: fixed identity fields are malformed")
    boundary_source_sha = require_sha256(boundary["source_sha256"], f"{label}.registry.nfcorpus_boundary.source_sha256")
    boundary_qid_sha = require_sha256(boundary["qid_set_sha256"], f"{label}.registry.nfcorpus_boundary.qid_set_sha256")
    boundary_qid_count = require_integer(boundary["qid_count"], f"{label}.registry.nfcorpus_boundary.qid_count", minimum=1)
    if boundary["rank_window"] != [80, 120]:
        raise GateContractError(f"{label}.registry.nfcorpus_boundary: rank window must be [80, 120]")
    boundary_suffix = require_string(boundary["source_path_suffix"], f"{label}.registry.nfcorpus_boundary.source_path_suffix").replace("\\", "/").lstrip("/")
    _registry_path_matches(Path(official["manifest"]["path"]), boundary_suffix, f"{label}.official-test.manifest")
    if official["manifest"]["sha256"] != boundary_source_sha:
        raise GateContractError(f"{label}: official-test source is not the fixed registry artifact")
    official_nf_qids = official["qids_by_domain"].get("nfcorpus")
    if not isinstance(official_nf_qids, list):
        raise GateContractError(f"{label}: official-test NFCorpus qid identity is malformed")
    official_nf_qids = sorted(official_nf_qids)
    if (
        len(official_nf_qids) != boundary_qid_count
        or sha256_json(official_nf_qids) != boundary_qid_sha
        or approved["nfcorpus_boundary_qids"] != official_nf_qids
        or approved["nfcorpus_boundary_qids_sha256"] != boundary_qid_sha
    ):
        raise GateContractError(f"{label}: NFCorpus boundary qid set is not the fixed official-test workload")

def validate_plan(plan_path: Path, *, production: bool = True) -> dict[str, Any]:
    plan_path = plan_path.resolve(strict=False)
    if production:
        _require_production_registry()
    plan = read_json(plan_path, "plan")
    require_exact_keys(plan, PLAN_KEYS, "plan")
    if plan["schema"] != PLAN_SCHEMA or plan["dimension"] != DIMENSION:
        raise GateContractError("plan: schema or D384 dimension mismatch")
    gate_id = require_string(plan["gate_id"], "plan.gate_id", safe_id=True)
    candidate_id = require_string(plan["candidate_id"], "plan.candidate_id", safe_id=True)
    if plan["thresholds"] != THRESHOLDS:
        raise GateContractError("plan: quality thresholds are immutable")
    if plan["evaluation"] != {"executed": False, "official": False}:
        raise GateContractError("plan: evaluation must be an unexecuted non-official plan")
    source = normalize_source(plan["source"], plan_path.parent, "plan.source")
    if production:
        require_elf_executable(Path(source["binary"]["path"]), "plan.source.binary")
    if len({dataset["dataset_id"] for dataset in source["dataset_by_domain"].values()}) != len(DOMAINS):
        raise GateContractError("plan.source.dataset_by_domain: dataset identities must be distinct")
    for component in ("manifest", "corpus", "queries"):
        if len({dataset[component]["path"] for dataset in source["dataset_by_domain"].values()}) != len(DOMAINS):
            raise GateContractError(f"plan.source.dataset_by_domain: {component} paths must be distinct per domain")
    source_workload = normalize_file_record(plan["source"]["workload"], plan_path.parent, "plan.source.workload")
    source_approved = normalize_file_record(plan["source"]["approved_workload"], plan_path.parent, "plan.source.approved_workload")
    if source_workload != source["workload"] or source_approved != source["approved_workload"]:
        raise GateContractError("plan.source workload provenance changed during validation")
    approved_record = normalize_file_record(plan["approved_workload"], plan_path.parent, "plan.approved_workload")
    if approved_record != source_approved:
        raise GateContractError("plan: approved workload record must equal source binding")
    qrels_raw = require_mapping(plan["qrels"], "plan.qrels")
    if set(qrels_raw) != set(DOMAINS):
        raise GateContractError("plan.qrels: exact domain coverage required")
    qrels = {domain: normalize_qrels_record(qrels_raw[domain], plan_path.parent, f"plan.qrels.{domain}") for domain in DOMAINS}
    if len({qrels[domain]["path"] for domain in DOMAINS}) != len(DOMAINS):
        raise GateContractError("plan.qrels: qrels paths must be distinct per domain")
    approved = validate_approved_workload(Path(approved_record["path"]), gate_id=gate_id, qrels=qrels, datasets=source["dataset_by_domain"], label="plan.approved_workload")
    workload = validate_workload(Path(source_workload["path"]), gate_id=gate_id, approved=approved, descriptor_sha256=approved_record["sha256"], label="plan.source.workload")
    if source_workload["path"] == approved_record["path"]:
        raise GateContractError("plan: workload and approved descriptor must be distinct files")
    exclusions_raw = plan["exclusions"]
    if not isinstance(exclusions_raw, list) or len(exclusions_raw) != len(EXCLUSION_NAMES):
        raise GateContractError("plan.exclusions: exactly dev4/reserve4/official-test are required")
    by_name: dict[str, dict[str, Any]] = {}
    for item in exclusions_raw:
        item_map = require_mapping(item, "plan.exclusions[]")
        if item_map.get("name") in by_name:
            raise GateContractError("plan.exclusions: duplicate exclusion identity")
        name = require_string(item_map.get("name"), "plan.exclusions[].name", safe_id=False)
        if name not in EXCLUSION_NAMES:
            raise GateContractError("plan.exclusions: exact qid-only exclusion identity is required")
        by_name[name] = normalize_exclusion_record(item_map, plan_path.parent, name, approved["query_ids_by_domain"], f"plan.exclusions.{name}")
    if set(by_name) != set(EXCLUSION_NAMES):
        raise GateContractError("plan.exclusions: exact dev4/reserve4/official-test coverage required")
    if production:
        _validate_production_registry_binding(
            approved,
            qrels,
            source["dataset_by_domain"],
            by_name,
            label="plan",
        )
    expected_anchor_identity = {"artifact_sha256": D384_ANCHOR_PACKAGE_SHA256, "package_manifest_sha256": D384_ANCHOR_MANIFEST_SHA256, "embedding_space_id": D384_ANCHOR_EMBEDDING_SPACE_ID}
    anchor_node = require_mapping(plan["anchor"], "plan.anchor")
    candidate_node = require_mapping(plan["candidate"], "plan.candidate")
    require_exact_keys(anchor_node, {"id", "package"}, "plan.anchor")
    require_exact_keys(candidate_node, {"id", "package"}, "plan.candidate")
    anchor_id = require_string(anchor_node["id"], "plan.anchor.id", safe_id=True)
    if production and anchor_id != D384_ANCHOR_ARTIFACT_ID:
        raise GateContractError("plan.anchor.id: current D384 anchor identity is pinned")
    anchor = normalize_package_record(anchor_node["package"], plan_path.parent, "plan.anchor.package", expected_role="anchor", expected_anchor=expected_anchor_identity, data_binding=_data_binding(approved), expected_compatibility_digest=approved["compatibility_digest"], production=production)
    if production:
        if anchor["package_identity"] != expected_anchor_identity:
            raise GateContractError("plan.anchor.package: current D384 anchor package/manifest/embedding identity mismatch")
        if not str(Path(anchor["path"])).endswith(D384_ANCHOR_PATH_SUFFIX):
            raise GateContractError("plan.anchor.package: current D384 anchor artifact path mismatch")
    else:
        # Synthetic fixtures deliberately use a local identity, but the
        # identity is still frozen and must be supplied back to ``run_harness``
        # by the test caller.  Production never reaches this branch.
        expected_anchor_identity = anchor["package_identity"]
    candidate = normalize_package_record(candidate_node["package"], plan_path.parent, "plan.candidate.package", expected_role="candidate", expected_anchor=anchor["package_identity"], data_binding=_data_binding(approved), expected_compatibility_digest=approved["compatibility_digest"], production=production)
    validate_package_pair(anchor, candidate)
    workload_root = Path(normalize_directory(plan["workload_root"], plan_path.parent, "plan.workload_root"))
    output_root = Path(normalize_directory(plan["output_root"], plan_path.parent, "plan.output_root"))
    if output_root == workload_root:
        raise GateContractError("plan.output_root must be distinct from workload_root")
    all_inputs = {Path(source["manifest"]["path"]), Path(source["binary"]["path"]), Path(source_workload["path"]), Path(approved_record["path"]), *[Path(item["path"]) for item in qrels.values()]}
    all_inputs.update(Path(dataset[key]["path"]) for dataset in source["dataset_by_domain"].values() for key in ("manifest", "corpus", "queries"))
    all_inputs.update(Path(item["manifest"]["path"]) for item in by_name.values())
    for item in by_name.values():
        exclusion_manifest = Path(item["manifest"]["path"])
        all_inputs.update(_resolve_source_file(raw_path, exclusion_manifest.parent, "plan.exclusions.source") for raw_path in item["source_sha256_by_file"])
    all_package_files = {Path(item["path"]) for package in (anchor, candidate) for item in (package, package["manifest"], package["attestation"]) if isinstance(item, dict) and "path" in item}
    all_package_files.update(Path(package["package_manifest"]["path"]) for package in (anchor, candidate) if package.get("package_manifest") is not None)
    all_package_files.update(Path(entry["path"]) for package in (anchor, candidate) for entry in package["sibling_entries"])
    if len(all_inputs | all_package_files) != len(all_inputs) + len(all_package_files):
        raise GateContractError("plan: input and package artifact paths must be pairwise distinct")
    require_pairwise_disjoint_paths(all_inputs | all_package_files, "plan input/package artifacts")
    require_pairwise_disjoint_paths((workload_root, output_root), "plan input/output roots")
    if any(_path_overlaps(output_root, path) for path in all_inputs | all_package_files) or any(_path_overlaps(workload_root, path) for path in all_inputs | all_package_files):
        raise GateContractError("plan: workload/output roots must be disjoint from all input/package artifacts")
    if any(_path_overlaps(output_root, Path(dataset["dataset_dir"])) for dataset in source["dataset_by_domain"].values()) or any(_path_overlaps(workload_root, Path(dataset["dataset_dir"])) for dataset in source["dataset_by_domain"].values()):
        raise GateContractError("plan: workload/output roots must be disjoint from dataset directories")
    return {"schema": PLAN_SCHEMA, "gate_id": gate_id, "candidate_id": candidate_id, "anchor_id": anchor_id, "dimension": DIMENSION, "thresholds": copy.deepcopy(THRESHOLDS), "evaluation": {"executed": False, "official": False}, "anchor": anchor, "candidate": candidate, "source": source, "approved_workload": {"path": approved_record["path"], "sha256": approved_record["sha256"], "bytes": approved_record["bytes"], "descriptor": approved}, "workload": {"path": source_workload["path"], "sha256": source_workload["sha256"], "bytes": source_workload["bytes"], "descriptor": workload}, "qrels": qrels, "exclusions": [by_name[name] for name in EXCLUSION_NAMES], "workload_root": str(workload_root), "output_root": str(output_root), "plan_path": str(plan_path), "plan_sha256": sha256_file(plan_path), "expected_anchor_identity": expected_anchor_identity, "input_paths": sorted(str(path) for path in all_inputs | all_package_files)}


def digest_without(payload: dict[str, Any], field: str) -> str:
    clone = copy.deepcopy(payload)
    clone.pop(field, None)
    return sha256_json(clone)


OUTPUT_LAYOUT = {
    "metrics": "{role}.{domain}.metrics.json",
    "metrics_tsv": "{role}.{domain}.metrics.tsv",
    "per_query": "{role}.{domain}.per-query.jsonl",
    "receipt": "{role}.{domain}.execution-receipt.json",
    "attestation": "gate-attestation.json",
}


def _freeze_manifest_impl(plan_path: Path, output_path: Path, *, production: bool) -> dict[str, Any]:
    """Validate a plan and freeze all immutable inputs before any evaluator run."""

    info = validate_plan(plan_path, production=production)
    gate_script = Path(__file__).resolve()
    require_regular_file(gate_script, "gate script")
    gate_script_record = {"path": str(gate_script), "sha256": sha256_file(gate_script), "bytes": gate_script.stat().st_size}
    output_path = output_path.resolve(strict=False)
    if output_path.exists() or output_path.is_symlink():
        raise GateContractError(f"frozen manifest output must be absent: {output_path}")
    frozen = {
        "schema": FROZEN_SCHEMA,
        "gate_id": info["gate_id"],
        "candidate_id": info["candidate_id"],
        "anchor_id": info["anchor_id"],
        "dimension": DIMENSION,
        "thresholds": copy.deepcopy(THRESHOLDS),
        "evaluation": {"executed": False, "official": False},
        "external_trust_anchor": {"required": True, "algorithm": "sha256_file", "purpose": "caller-supplied immutable frozen-manifest identity"},
        "expected_anchor_identity": info["expected_anchor_identity"],
        "anchor": _public_package_record(info["anchor"]),
        "candidate": _public_package_record(info["candidate"]),
        "source": info["source"],
        "approved_workload": info["approved_workload"],
        "workload": info["workload"],
        "qrels": info["qrels"],
        "exclusions": info["exclusions"],
        "workload_root": info["workload_root"],
        "output_root": info["output_root"],
        "output_layout": copy.deepcopy(OUTPUT_LAYOUT),
        "provenance": {"plan_path": info["plan_path"], "plan_sha256": info["plan_sha256"], "gate_script": gate_script_record, "binary_sha256": info["source"]["binary"]["sha256"], "approved_workload_sha256": info["approved_workload"]["sha256"], "anchor_identity": info["expected_anchor_identity"]},
    }
    frozen["manifest_sha256"] = digest_without(frozen, "manifest_sha256")
    write_json_new(output_path, frozen)
    return {"schema": FROZEN_SCHEMA, "manifest_sha256": frozen["manifest_sha256"], "file_sha256": sha256_file(output_path), "path": str(output_path), "manifest": frozen}


def freeze_manifest(plan_path: Path, output_path: Path) -> dict[str, Any]:
    """Production freeze entry point; synthetic fixtures cannot reach it."""

    return _freeze_manifest_impl(plan_path, output_path, production=True)


def _freeze_manifest_for_tests(plan_path: Path, output_path: Path) -> dict[str, Any]:
    """Private synthetic-fixture hook; never exposed by the CLI."""

    return _freeze_manifest_impl(plan_path, output_path, production=False)


def _validate_frozen_files(frozen: dict[str, Any], frozen_path: Path, *, production: bool) -> dict[str, Any]:
    base = frozen_path.parent
    source = frozen["source"]
    require_exact_keys(source, {"manifest", "binary", "cwd", "dataset_by_domain", "workload", "approved_workload"}, "frozen.source")
    require_domain_mapping(source["dataset_by_domain"], "frozen.source.dataset_by_domain")
    qrels_node = require_domain_mapping(frozen["qrels"], "frozen.qrels")
    provenance = require_mapping(frozen["provenance"], "frozen.provenance")
    require_exact_keys(provenance, {"plan_path", "plan_sha256", "gate_script", "binary_sha256", "approved_workload_sha256", "anchor_identity"}, "frozen.provenance")
    plan_path = resolve_path(provenance["plan_path"], base, "frozen.provenance.plan_path")
    require_regular_file(plan_path, "frozen.provenance.plan")
    plan_sha = require_sha256(provenance["plan_sha256"], "frozen.provenance.plan_sha256")
    if sha256_file(plan_path) != plan_sha:
        raise GateContractError("frozen provenance plan sha256 mismatch")
    gate_script = normalize_file_record(provenance["gate_script"], base, "frozen.provenance.gate_script")
    if gate_script["path"] != str(Path(__file__).resolve()):
        raise GateContractError("frozen provenance gate script path mismatch")
    binary = normalize_file_record(source["binary"], base, "frozen.source.binary", executable=True)
    if binary != source["binary"]:
        raise GateContractError("frozen source binary changed after freeze")
    if production:
        require_elf_executable(Path(binary["path"]), "frozen.source.binary")
    source_manifest = normalize_file_record(source["manifest"], base, "frozen.source.manifest")
    if source_manifest != source["manifest"]:
        raise GateContractError("frozen source manifest binding changed")
    require_directory(Path(source["cwd"]), "frozen.source.cwd")
    datasets: dict[str, dict[str, Any]] = {}
    for domain in DOMAINS:
        datasets[domain] = normalize_dataset_record(source["dataset_by_domain"][domain], base, domain, f"frozen.source.dataset_by_domain.{domain}")
        if datasets[domain] != source["dataset_by_domain"][domain]:
            raise GateContractError(f"frozen dataset binding changed for {domain}")
    qrels: dict[str, dict[str, Any]] = {}
    for domain in DOMAINS:
        require_exact_keys(qrels_node[domain], {"path", "sha256", "bytes", "qids", "qid_set_sha256", "query_count", "qrels_pair_count", "relevant_pair_count"}, f"frozen.qrels.{domain}")
        current = normalize_qrels_record({key: qrels_node[domain][key] for key in ("path", "sha256", "bytes")}, base, f"frozen.qrels.{domain}")
        full_qrels = parse_qrels(Path(current["path"]), f"frozen.qrels.{domain}")
        for key in ("path", "sha256", "bytes", "qids", "qid_set_sha256", "query_count", "qrels_pair_count", "relevant_pair_count"):
            if current[key] != qrels_node[domain][key]:
                raise GateContractError(f"frozen qrels binding changed for {domain}")
        qrels[domain] = {**current, "rels": full_qrels["rels"]}
    approved_rec = normalize_file_record({key: frozen["approved_workload"][key] for key in ("path", "sha256", "bytes")}, base, "frozen.approved_workload")
    workload_rec = normalize_file_record({key: frozen["workload"][key] for key in ("path", "sha256", "bytes")}, base, "frozen.workload")
    if approved_rec != {key: frozen["approved_workload"][key] for key in ("path", "sha256", "bytes")}:
        raise GateContractError("frozen approved workload descriptor changed")
    if workload_rec != {key: frozen["workload"][key] for key in ("path", "sha256", "bytes")}:
        raise GateContractError("frozen workload descriptor changed")
    if provenance["binary_sha256"] != binary["sha256"] or provenance["approved_workload_sha256"] != approved_rec["sha256"]:
        raise GateContractError("frozen provenance input hash binding mismatch")
    provenance_anchor = require_mapping(provenance["anchor_identity"], "frozen.provenance.anchor_identity")
    require_exact_keys(provenance_anchor, {"artifact_sha256", "package_manifest_sha256", "embedding_space_id"}, "frozen.provenance.anchor_identity")
    if provenance_anchor != frozen["expected_anchor_identity"]:
        raise GateContractError("frozen provenance anchor identity mismatch")
    approved = validate_approved_workload(Path(approved_rec["path"]), gate_id=frozen["gate_id"], qrels=qrels, datasets=datasets, label="frozen.approved_workload")
    if approved != frozen["approved_workload"]["descriptor"]:
        raise GateContractError("frozen approved workload derived descriptor changed")
    workload = validate_workload(Path(workload_rec["path"]), gate_id=frozen["gate_id"], approved=approved, descriptor_sha256=approved_rec["sha256"], label="frozen.workload")
    if workload != frozen["workload"]["descriptor"]:
        raise GateContractError("frozen workload descriptor changed")
    expected_anchor = frozen["expected_anchor_identity"]
    anchor = normalize_package_record(frozen["anchor"], base, "frozen.anchor", expected_role="anchor", expected_anchor=expected_anchor, data_binding=_data_binding(approved), expected_compatibility_digest=approved["compatibility_digest"], production=production)
    if production and not str(Path(anchor["path"])).endswith(D384_ANCHOR_PATH_SUFFIX):
        raise GateContractError("frozen.anchor.package: current D384 anchor artifact path mismatch")
    candidate = normalize_package_record(frozen["candidate"], base, "frozen.candidate", expected_role="candidate", expected_anchor=anchor["package_identity"], data_binding=_data_binding(approved), expected_compatibility_digest=approved["compatibility_digest"], production=production)
    validate_package_pair(anchor, candidate)
    normalized_exclusions: list[dict[str, Any]] = []
    for name, item in zip(EXCLUSION_NAMES, frozen["exclusions"], strict=True):
        current = normalize_exclusion_record({"kind": item["kind"], "name": item["name"], "manifest": item["manifest"]}, base, name, approved["query_ids_by_domain"], f"frozen.exclusions.{name}")
        if current != item:
            raise GateContractError(f"frozen exclusion binding changed for {name}")
        normalized_exclusions.append(current)
    if production:
        _validate_production_registry_binding(
            approved,
            qrels,
            datasets,
            normalized_exclusions,
            label="frozen",
        )
    workload_root_raw = require_string(frozen["workload_root"], "frozen.workload_root")
    workload_root = resolve_path(workload_root_raw, base, "frozen.workload_root")
    if str(workload_root) != workload_root_raw:
        raise GateContractError("frozen.workload_root must be a canonical absolute path")
    require_directory(workload_root, "frozen.workload_root")
    output_root_raw = require_string(frozen["output_root"], "frozen.output_root")
    output_root = resolve_path(output_root_raw, base, "frozen.output_root")
    if str(output_root) != output_root_raw:
        raise GateContractError("frozen.output_root must be a canonical absolute path")
    require_directory(output_root, "frozen.output_root")
    if output_root == workload_root or output_root == Path(source["cwd"]) or output_root in {Path(dataset["dataset_dir"]) for dataset in datasets.values()}:
        raise GateContractError("frozen output_root must be distinct from all input roots")
    all_inputs = {
        Path(source_manifest["path"]),
        Path(binary["path"]),
        Path(workload_rec["path"]),
        Path(approved_rec["path"]),
        *[Path(item["path"]) for item in qrels.values()],
    }
    all_inputs.update(Path(dataset[key]["path"]) for dataset in datasets.values() for key in ("manifest", "corpus", "queries"))
    for item in normalized_exclusions:
        exclusion_manifest = Path(item["manifest"]["path"])
        all_inputs.add(exclusion_manifest)
        all_inputs.update(_resolve_source_file(raw_path, exclusion_manifest.parent, "frozen.exclusions.source") for raw_path in item["source_sha256_by_file"])
    all_package_files = {
        Path(item["path"])
        for package in (anchor, candidate)
        for item in (package, package["manifest"], package["attestation"])
        if isinstance(item, dict) and "path" in item
    }
    all_package_files.update(Path(package["package_manifest"]["path"]) for package in (anchor, candidate) if package.get("package_manifest") is not None)
    all_package_files.update(Path(entry["path"]) for package in (anchor, candidate) for entry in package["sibling_entries"])
    if len(all_inputs | all_package_files) != len(all_inputs) + len(all_package_files):
        raise GateContractError("frozen input and package artifact paths must be pairwise distinct")
    require_pairwise_disjoint_paths(all_inputs | all_package_files, "frozen input/package artifacts")
    require_pairwise_disjoint_paths((workload_root, output_root), "frozen input/output roots")
    if any(_path_overlaps(output_root, path) for path in all_inputs | all_package_files) or any(_path_overlaps(workload_root, path) for path in all_inputs | all_package_files):
        raise GateContractError("frozen workload/output roots must be disjoint from all input/package artifacts")
    if any(_path_overlaps(output_root, Path(dataset["dataset_dir"])) for dataset in datasets.values()) or any(_path_overlaps(workload_root, Path(dataset["dataset_dir"])) for dataset in datasets.values()):
        raise GateContractError("frozen workload/output roots must be disjoint from dataset directories")
    return {"path": str(frozen_path), "file_sha256": sha256_file(frozen_path), "manifest": frozen, "anchor": anchor, "candidate": candidate, "source": {**source, "binary": binary, "dataset_by_domain": datasets}, "qrels": qrels, "approved_workload": {**approved_rec, "descriptor": approved}, "workload": {**workload_rec, "descriptor": workload}, "workload_root": workload_root, "output_root": output_root}


def validate_frozen_manifest(frozen_path: Path, *, expected_frozen_manifest_sha256: str | None, require_outputs_absent: bool = False, run_dir: Path | None = None, production: bool = True) -> dict[str, Any]:
    if production:
        _require_production_registry()
    if expected_frozen_manifest_sha256 is None:
        raise GateContractError("external expected frozen manifest sha256 is required")
    expected_external = require_sha256(expected_frozen_manifest_sha256, "expected frozen manifest sha256")
    frozen_path = frozen_path.resolve(strict=False)
    require_regular_file(frozen_path, "frozen manifest")
    actual_external = sha256_file(frozen_path)
    if actual_external != expected_external:
        raise GateContractError("external frozen manifest sha256 trust-anchor mismatch")
    frozen = read_json(frozen_path, "frozen manifest")
    required = {"schema", "gate_id", "candidate_id", "anchor_id", "dimension", "thresholds", "evaluation", "external_trust_anchor", "expected_anchor_identity", "anchor", "candidate", "source", "approved_workload", "workload", "qrels", "exclusions", "workload_root", "output_root", "output_layout", "provenance", "manifest_sha256"}
    require_exact_keys(frozen, required, "frozen manifest")
    require_string(frozen["gate_id"], "frozen.gate_id", safe_id=True)
    require_string(frozen["candidate_id"], "frozen.candidate_id", safe_id=True)
    require_string(frozen["anchor_id"], "frozen.anchor_id", safe_id=True)
    if frozen["schema"] != FROZEN_SCHEMA or frozen["dimension"] != DIMENSION or frozen["thresholds"] != THRESHOLDS or frozen["evaluation"] != {"executed": False, "official": False}:
        raise GateContractError("frozen manifest immutable policy mismatch")
    if production and frozen["anchor_id"] != D384_ANCHOR_ARTIFACT_ID:
        raise GateContractError("frozen anchor id is not the pinned current D384 anchor")
    anchor_identity = require_mapping(frozen["expected_anchor_identity"], "frozen.expected_anchor_identity")
    require_exact_keys(anchor_identity, {"artifact_sha256", "package_manifest_sha256", "embedding_space_id"}, "frozen.expected_anchor_identity")
    anchor_identity = {key: require_sha256(anchor_identity[key], f"frozen.expected_anchor_identity.{key}") for key in ("artifact_sha256", "package_manifest_sha256", "embedding_space_id")}
    if production and anchor_identity != {"artifact_sha256": D384_ANCHOR_PACKAGE_SHA256, "package_manifest_sha256": D384_ANCHOR_MANIFEST_SHA256, "embedding_space_id": D384_ANCHOR_EMBEDDING_SPACE_ID}:
        raise GateContractError("frozen anchor identity is not the pinned current D384 anchor")
    require_exact_keys(frozen["approved_workload"], {"path", "sha256", "bytes", "descriptor"}, "frozen.approved_workload")
    require_exact_keys(frozen["workload"], {"path", "sha256", "bytes", "descriptor"}, "frozen.workload")
    require_domain_mapping(frozen["qrels"], "frozen.qrels")
    if not isinstance(frozen["exclusions"], list) or len(frozen["exclusions"]) != len(EXCLUSION_NAMES):
        raise GateContractError("frozen.exclusions: exact dev4/reserve4/official-test coverage required")
    if digest_without(frozen, "manifest_sha256") != require_sha256(frozen["manifest_sha256"], "frozen manifest.manifest_sha256"):
        raise GateContractError("frozen manifest self-digest mismatch")
    if frozen["external_trust_anchor"] != {"required": True, "algorithm": "sha256_file", "purpose": "caller-supplied immutable frozen-manifest identity"}:
        raise GateContractError("frozen external trust-anchor policy mismatch")
    if frozen["output_layout"] != OUTPUT_LAYOUT:
        raise GateContractError("frozen output layout mismatch")
    info = _validate_frozen_files(frozen, frozen_path, production=production)
    info["expected_external_sha256"] = expected_external
    info["file_sha256"] = actual_external
    if require_outputs_absent:
        if run_dir is None:
            raise GateContractError("run_dir is required for output absence preflight")
        run_dir = run_dir.resolve(strict=False)
        require_directory(run_dir, "evaluator run directory")
        for path in _planned_output_paths(info, run_dir):
            if path.exists() or path.is_symlink():
                raise GateContractError(f"frozen output must be absent before evaluation: {path}")
    return info


def _planned_output_paths(info: dict[str, Any], run_dir: Path) -> list[Path]:
    return [run_dir / OUTPUT_LAYOUT[key].format(role=role, domain=domain) for role in ROLES for domain in DOMAINS for key in ("metrics", "metrics_tsv", "per_query", "receipt")] + [run_dir / OUTPUT_LAYOUT["attestation"]]


def _canonical_evaluator_argv(frozen_info: dict[str, Any], role: str, domain: str, outputs: dict[str, Path]) -> list[str]:
    if role not in ROLES or domain not in DOMAINS:
        raise GateContractError("invalid evaluator role/domain")
    source = frozen_info["source"]
    dataset = source["dataset_by_domain"][domain]
    qrels = frozen_info["qrels"][domain]
    package = frozen_info["anchor" if role == "anchor" else "candidate"]
    argv = [
        source["binary"]["path"],
        NATIVE_EVAL_SUBCOMMAND,
        "--dataset", domain,
        "--split", HELDOUT_SPLIT,
        "--qrels", qrels["path"],
        "--batch-size", str(BATCH_SIZE),
        "--top-k", str(TOP_K),
        "--bits", f"{Q3_BITS},{Q5_BITS}",
        "--quantizer-seed", str(TURBOQUANT_SEED),
        "--max-docs", "0",
        "--max-queries", "0",
        "--per-query-top-k", str(TOP_K),
        "--metrics-json", str(outputs["metrics"]),
        "--metrics-tsv", str(outputs["metrics_tsv"]),
        "--per-query-jsonl", str(outputs["per_query"]),
        package["path"],
        dataset["dataset_dir"],
    ]
    return argv


def _validate_evaluator_argv(argv: list[str], frozen_info: dict[str, Any], role: str, domain: str, outputs: dict[str, Path]) -> None:
    expected = _canonical_evaluator_argv(frozen_info, role, domain, outputs)
    if argv != expected:
        raise GateContractError(f"{role}/{domain}: evaluator argv is not the exact pinned command")
    if any(any(marker in token for marker in SHELL_MARKERS) for token in argv):
        raise GateContractError("constructed argv contains shell metacharacters")
    source = frozen_info["source"]
    package = frozen_info["anchor" if role == "anchor" else "candidate"]
    dataset = source["dataset_by_domain"][domain]
    if argv[0] != source["binary"]["path"] or argv[1] != NATIVE_EVAL_SUBCOMMAND or argv[-2] != package["path"] or argv[-1] != dataset["dataset_dir"]:
        raise GateContractError("constructed argv semantic binding mismatch")


def build_evaluator_argv(frozen_info: dict[str, Any], role: str, domain: str, outputs: dict[str, Path]) -> list[str]:
    argv = _canonical_evaluator_argv(frozen_info, role, domain, outputs)
    _validate_evaluator_argv(argv, frozen_info, role, domain, outputs)
    return argv


def _binding_for_run(frozen_info: dict[str, Any], role: str, domain: str, nonce: str, argv: list[str], outputs: dict[str, Path]) -> dict[str, Any]:
    frozen = frozen_info["manifest"]
    source = frozen_info["source"]
    package = frozen_info["anchor" if role == "anchor" else "candidate"]
    dataset = source["dataset_by_domain"][domain]
    binding: dict[str, Any] = {
        "schema": GATE_BINDING_SCHEMA,
        "gate_id": frozen["gate_id"],
        "role": role,
        "domain": domain,
        "nonce": nonce,
        "frozen_manifest_sha256": frozen_info["expected_external_sha256"],
        "frozen_manifest_digest": frozen["manifest_sha256"],
        "package_sha256": package["package_identity"]["artifact_sha256"],
        "package_manifest_sha256": package["package_identity"]["package_manifest_sha256"],
        "sibling_rollup_sha256": package["sibling_rollup_sha256"],
        "sidecar_sha256": None if package["sidecar"] is None else package["sidecar"]["sha256"],
        "embedding_space_id": package["embedding_space_id"],
        "source_manifest_sha256": source["manifest"]["sha256"],
        "gate_script_sha256": frozen["provenance"]["gate_script"]["sha256"],
        "binary_sha256": source["binary"]["sha256"],
        "argv": argv,
        "argv_sha256": sha256_json(argv),
        "cwd": source["cwd"],
        "dataset_id": domain,
        "dataset_manifest_sha256": dataset["manifest"]["sha256"],
        "corpus_sha256": dataset["corpus"]["sha256"],
        "queries_sha256": dataset["queries"]["sha256"],
        "qrels_sha256": frozen_info["qrels"][domain]["sha256"],
        "compatibility_digest": frozen["approved_workload"]["descriptor"]["compatibility_digest"],
        "workload_sha256": frozen["workload"]["sha256"],
        "approved_workload_sha256": frozen["approved_workload"]["sha256"],
        "workload_qid_set_sha256_by_domain": frozen["approved_workload"]["descriptor"]["qid_set_sha256_by_domain"],
        "workload_query_count_by_domain": frozen["approved_workload"]["descriptor"]["query_count_by_domain"],
        "workload_qrels_sha256_by_domain": frozen["approved_workload"]["descriptor"]["qrels_sha256_by_domain"],
        "config": {"dimension": DIMENSION, "bits": [Q3_BITS, Q5_BITS], "seed": TURBOQUANT_SEED, "top_k": TOP_K, "batch_size": BATCH_SIZE, "max_docs": 0, "max_queries": 0, "per_query_top_k": TOP_K, "rerank_overfetch": [], "package_mode": "native_mll_sibling", "score_mode": "turboquant_ip_prepared", "split": HELDOUT_SPLIT},
        "outputs": {key: str(outputs[key]) for key in ("metrics", "metrics_tsv", "per_query")},
        "nfcorpus_boundary_qids_sha256": frozen["approved_workload"]["descriptor"]["nfcorpus_boundary_qids_sha256"],
        "nfcorpus_boundary_rank_window": [80, 120],
    }
    binding["binding_sha256"] = digest_without(binding, "binding_sha256")
    return binding


def _runtime_binding_for_run(frozen_info: dict[str, Any], role: str, domain: str, nonce: str, argv: list[str], outputs: dict[str, Path]) -> dict[str, Any]:
    """Compute the binding the native evaluator must independently emit.

    This is intentionally separate from ``_binding_for_run``.  The latter is
    gate-owned freshness/provenance metadata; this record describes resolved
    runtime inputs and must be computed by the pinned EOS executable itself.
    Passing the gate record through an environment variable would permit a
    child that merely echoes caller data to masquerade as native evidence.
    """

    frozen = frozen_info["manifest"]
    source = frozen_info["source"]
    package = frozen_info["anchor" if role == "anchor" else "candidate"]
    dataset = source["dataset_by_domain"][domain]
    xpkg_entries = []
    if package.get("xpkg") is not None:
        xpkg_entries = [
            {"role": entry["role"], "name": entry["name"], "path": entry["path"], "sha256": entry["sha256"], "bytes": entry["bytes"]}
            for entry in sorted(package["xpkg"]["entries"], key=lambda item: item["role"])
        ]
    qrels = frozen_info["qrels"][domain]
    approved = frozen["approved_workload"]["descriptor"]
    binding: dict[str, Any] = {
        "schema": RUNTIME_BINDING_SCHEMA,
        "producer": "eos-native-runtime-resolved-inputs",
        "gate_id": frozen["gate_id"],
        "role": role,
        "domain": domain,
        "split": HELDOUT_SPLIT,
        "nonce": nonce,
        "frozen_manifest_sha256": frozen_info["expected_external_sha256"],
        "binary_path": source["binary"]["path"],
        "binary_sha256": source["binary"]["sha256"],
        "argv": list(argv),
        "argv_sha256": sha256_json(argv),
        "cwd": source["cwd"],
        "dimension": DIMENSION,
        "score_mode": "turboquant_ip_prepared",
        "package_mode": "native_mll_sibling",
        "package": {
            "artifact_path": package["path"],
            "artifact_sha256": package["sha256"],
            "package_manifest_path": None if package.get("package_manifest") is None else package["package_manifest"]["path"],
            "package_manifest_sha256": package.get("xpkg_manifest_sha256"),
            "roles": xpkg_entries,
            "roles_sha256": sha256_json(xpkg_entries),
        },
        "dataset": {
            "dataset_id": dataset["dataset_id"],
            "dataset_dir": dataset["dataset_dir"],
            "manifest_path": dataset["manifest"]["path"],
            "manifest_sha256": dataset["manifest"]["sha256"],
            "corpus_path": dataset["corpus"]["path"],
            "corpus_sha256": dataset["corpus"]["sha256"],
            "queries_path": dataset["queries"]["path"],
            "queries_sha256": dataset["queries"]["sha256"],
        },
        "qrels": {
            "path": qrels["path"],
            "sha256": qrels["sha256"],
            "qid_set_sha256": qrels["qid_set_sha256"],
            "query_count": qrels["query_count"],
            "qrels_pair_count": qrels["qrels_pair_count"],
            "relevant_pair_count": qrels["relevant_pair_count"],
        },
        "workload": {
            "path": frozen["workload"]["path"],
            "sha256": frozen["workload"]["sha256"],
            "descriptor_sha256": frozen["approved_workload"]["sha256"],
            "qid_set_sha256_by_domain": approved["qid_set_sha256_by_domain"],
            "query_count_by_domain": approved["query_count_by_domain"],
        },
        "config": {
            "dimension": DIMENSION,
            "bits": [Q3_BITS, Q5_BITS],
            "seed": TURBOQUANT_SEED,
            "top_k": TOP_K,
            "per_query_top_k": TOP_K,
            "batch_size": BATCH_SIZE,
            "max_docs": 0,
            "max_queries": 0,
            "split": HELDOUT_SPLIT,
            "score_mode": "turboquant_ip_prepared",
            "package_mode": "native_mll_sibling",
            "rerank_overfetch": [],
            "rerank_bits": 0,
        },
        "outputs": {key: str(outputs[key]) for key in ("metrics", "metrics_tsv", "per_query")},
    }
    binding["binding_sha256"] = digest_without(binding, "binding_sha256")
    return binding


def _harness_environment(frozen_info: dict[str, Any], nonce: str, binding: dict[str, Any], *, test_mode: bool) -> dict[str, str]:
    """Build a reproducible evaluator environment with no caller preload hooks."""

    # The evaluator is launched by absolute path, so PATH is only retained
    # for the synthetic test shebang.  In production, omitting LD_PRELOAD,
    # LD_LIBRARY_PATH, PYTHONPATH, and similar ambient variables prevents a
    # wrapper or preload from changing the pinned executable's behavior.
    env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin") if test_mode else "/usr/bin:/bin", "LANG": "C", "LC_ALL": "C"}
    if test_mode:
        for key in ("MOCK_MODE", "MOCK_COPY_FROM"):
            value = os.environ.get(key)
            if value is not None:
                env[key] = value
    # The gate binding is an input to the native gate validator, not runtime
    # evidence.  The executable must independently resolve argv/files and emit
    # RUNTIME_BINDING_SCHEMA; the parser below rejects a runtime record that is
    # merely an echo of this channel.
    binding_raw = canonical_json(binding).decode("utf-8")
    env.update(
        {
            "EOS_AOQT_GATE_BINDING_JSON": binding_raw,
            # Native LoadTurboQuantGateBindingFromEnvironment compares this
            # channel to the binding's self-digest, not to the digest of the
            # object that still contains that self-digest.
            "EOS_AOQT_GATE_BINDING_SHA256": binding["binding_sha256"],
            "EOS_AOQT_GATE_NONCE": nonce,
            "EOS_AOQT_FROZEN_MANIFEST_SHA256": frozen_info["expected_external_sha256"],
            "EOS_AOQT_FROZEN_MANIFEST_PATH": frozen_info["path"],
        }
    )
    return env


def _assert_run_directory(path: Path, identity: tuple[int, int], label: str) -> None:
    require_directory(path, label)
    try:
        stat_result = path.stat()
    except OSError as exc:
        raise GateContractError(f"{label}: cannot stat fresh run directory: {exc}") from exc
    if (stat_result.st_dev, stat_result.st_ino) != identity:
        raise GateContractError(f"{label}: fresh run directory was replaced")


def _require_binding(value: Any, expected: dict[str, Any], label: str) -> None:
    binding = require_mapping(value, label)
    if binding != expected:
        raise GateContractError(f"{label}: native output gate binding mismatch (nonce/path/digest substitution)")
    if digest_without(binding, "binding_sha256") != binding["binding_sha256"]:
        raise GateContractError(f"{label}: native output gate binding self-digest mismatch")


def _require_runtime_binding(value: Any, expected: dict[str, Any], label: str) -> None:
    """Require an exact binding independently generated by the EOS binary."""

    binding = require_mapping(value, label)
    if binding != expected:
        raise GateContractError(f"{label}: native runtime binding mismatch (resolved input/package/argv substitution)")
    if binding.get("schema") != RUNTIME_BINDING_SCHEMA or binding.get("producer") != "eos-native-runtime-resolved-inputs":
        raise GateContractError(f"{label}: output is not a native runtime binding")
    if digest_without(binding, "binding_sha256") != binding.get("binding_sha256"):
        raise GateContractError(f"{label}: native runtime binding self-digest mismatch")


def _metric_value(node: dict[str, Any], key: str, label: str) -> float:
    if key not in node:
        raise GateContractError(f"{label}: missing {key}")
    return require_finite_number(node[key], f"{label}.{key}", bounded=True)


def _quality_pair(value: Any, label: str) -> dict[str, float]:
    node = require_mapping(value, label)
    return {"ndcg_at_10": _metric_value(node, "ndcg_at_10", label), "recall_at_100": _metric_value(node, "recall_at_100", label)}


def _ranking_fingerprint(top_k: list[dict[str, Any]], label: str) -> str:
    # Scores are included in the fingerprint so a boundary receipt cannot
    # point at a different score window while retaining the same document IDs.
    return sha256_json(
        [
            {"rank": require_integer(item["rank"], f"{label}.rank", minimum=1), "doc_id": require_string(item["doc_id"], f"{label}.doc_id"), "score": require_finite_number(item["score"], f"{label}.score")}
            for item in top_k
        ]
    )


def _validate_nf_boundary_evidence(value: Any, *, qid: str, compact_top: list[dict[str, Any]], dense_top: list[dict[str, Any]], label: str) -> dict[str, Any]:
    """Validate native rank-window safety evidence for one NFCorpus query."""

    evidence = require_mapping(value, label)
    required = {"schema", "qid", "rank_window", "candidate_count", "exit_rank", "substitution_count", "margin_to_exit", "compact_top_k_sha256", "dense_top_k_sha256", "window", "boundary_complete"}
    require_exact_keys(evidence, required, label)
    if evidence["schema"] != BOUNDARY_EVIDENCE_SCHEMA or evidence["qid"] != qid or evidence["rank_window"] != [80, 120] or evidence["boundary_complete"] is not True:
        raise GateContractError(f"{label}: native NF boundary identity is incomplete")
    if require_integer(evidence["candidate_count"], f"{label}.candidate_count", minimum=TOP_K) < TOP_K or require_integer(evidence["exit_rank"], f"{label}.exit_rank", minimum=TOP_K) < TOP_K:
        raise GateContractError(f"{label}: evaluator did not execute through the pinned rank window")
    substitution_count = require_integer(evidence["substitution_count"], f"{label}.substitution_count", minimum=0)
    margin = require_finite_number(evidence["margin_to_exit"], f"{label}.margin_to_exit")
    if margin < 0:
        raise GateContractError(f"{label}: native NF boundary margin is negative")
    if require_sha256(evidence["compact_top_k_sha256"], f"{label}.compact_top_k_sha256") != _ranking_fingerprint(compact_top, f"{label}.compact_top_k") or require_sha256(evidence["dense_top_k_sha256"], f"{label}.dense_top_k_sha256") != _ranking_fingerprint(dense_top, f"{label}.dense_top_k"):
        raise GateContractError(f"{label}: native NF boundary fingerprints do not bind generated rankings")
    window = evidence["window"]
    if not isinstance(window, list) or len(window) != 2:
        raise GateContractError(f"{label}: native NF boundary window evidence must contain ranks 80 and 120")
    by_rank: dict[int, dict[str, Any]] = {}
    for index, item in enumerate(window):
        node = require_mapping(item, f"{label}.window[{index}]")
        require_exact_keys(node, {"rank", "doc_id", "score"}, f"{label}.window[{index}]")
        rank = require_integer(node["rank"], f"{label}.window[{index}].rank", minimum=1)
        if rank not in {80, 120} or rank in by_rank:
            raise GateContractError(f"{label}: native NF boundary window ranks are not exactly 80 and 120")
        by_rank[rank] = node
    if set(by_rank) != {80, 120}:
        raise GateContractError(f"{label}: native NF boundary window ranks are incomplete")
    compact_by_rank = {item["rank"]: item for item in compact_top}
    for rank, node in by_rank.items():
        expected = compact_by_rank[rank]
        if node["doc_id"] != expected["doc_id"] or abs(require_finite_number(node["score"], f"{label}.window.{rank}.score") - expected["score"]) > COMPARISON_EPSILON:
            raise GateContractError(f"{label}: native NF boundary window is not bound to compact ranking")
    return {
        "schema": BOUNDARY_EVIDENCE_SCHEMA,
        "qid": qid,
        "rank_window": [80, 120],
        "candidate_count": evidence["candidate_count"],
        "exit_rank": evidence["exit_rank"],
        # Quantized and dense rankings may legitimately contain different
        # documents.  Keep the native diagnostic, but do not confuse ranking
        # substitutions with provenance substitution of the package/input.
        "substitution_count": substitution_count,
        "margin_to_exit": margin,
        "compact_top_k_sha256": evidence["compact_top_k_sha256"],
        "dense_top_k_sha256": evidence["dense_top_k_sha256"],
        "window": [{"rank": rank, "doc_id": by_rank[rank]["doc_id"], "score": require_finite_number(by_rank[rank]["score"], f"{label}.window.{rank}.score")} for rank in (80, 120)],
        "boundary_complete": True,
    }


def _derive_nf_boundary_evidence(qid: str, compact_top: list[dict[str, Any]], dense_top: list[dict[str, Any]]) -> dict[str, Any]:
    """Derive the fixed rank-window record from native top-120 output.

    The native evaluator already binds the full q3 ranking, scores, and corpus
    IDs in the per-query contract. Requiring a second bespoke receipt would
    duplicate those facts and could reject an otherwise complete producer.
    """

    compact_by_rank = {item["rank"]: item for item in compact_top}
    return {
        "schema": BOUNDARY_EVIDENCE_SCHEMA,
        "qid": qid,
        "rank_window": [80, 120],
        "candidate_count": len(compact_top),
        "exit_rank": compact_top[-1]["rank"],
        "substitution_count": sum(
            compact_top[index]["doc_id"] != dense_top[index]["doc_id"]
            for index in range(min(len(compact_top), len(dense_top)))
        ),
        "margin_to_exit": 0.0,
        "compact_top_k_sha256": _ranking_fingerprint(compact_top, "derived.compact_top_k"),
        "dense_top_k_sha256": _ranking_fingerprint(dense_top, "derived.dense_top_k"),
        "window": [
            {"rank": rank, "doc_id": compact_by_rank[rank]["doc_id"], "score": compact_by_rank[rank]["score"]}
            for rank in (80, 120)
        ],
        "boundary_complete": True,
        "source": "native_per_query_top120",
    }


def _rank_metrics(top_k: list[dict[str, Any]], rels: dict[str, float], corpus_ids: set[str], label: str) -> dict[str, float]:
    if len(top_k) != TOP_K:
        raise GateContractError(f"{label}: native ranking must contain exactly top-k=120 entries")
    ranks: list[tuple[int, str, float]] = []
    seen_rank: set[int] = set()
    seen_doc: set[str] = set()
    previous_score: float | None = None
    for index, item in enumerate(top_k, 1):
        node = require_mapping(item, f"{label}.top_k[{index - 1}]")
        required = {"rank", "doc_id", "score", "relevance"}
        if not required.issubset(node):
            raise GateContractError(f"{label}.top_k[{index - 1}]: native ranking fields missing")
        rank = require_integer(node["rank"], f"{label}.top_k[{index - 1}].rank", minimum=1)
        doc_id = require_string(node["doc_id"], f"{label}.top_k[{index - 1}].doc_id")
        if rank != index or rank in seen_rank or doc_id in seen_doc:
            raise GateContractError(f"{label}: ranking ranks/doc IDs are not unique and ordered")
        score = require_finite_number(node["score"], f"{label}.top_k[{index - 1}].score")
        if doc_id not in corpus_ids:
            raise GateContractError(f"{label}: ranking document {doc_id!r} is absent from the frozen corpus")
        if previous_score is not None and score > previous_score + COMPARISON_EPSILON:
            raise GateContractError(f"{label}: ranking scores are not sorted in descending order")
        previous_score = score
        relevance = require_finite_number(node["relevance"], f"{label}.top_k[{index - 1}].relevance")
        expected_relevance = rels.get(doc_id, 0.0)
        if abs(relevance - expected_relevance) > COMPARISON_EPSILON:
            raise GateContractError(f"{label}: output relevance disagrees with frozen qrels")
        seen_rank.add(rank)
        seen_doc.add(doc_id)
        ranks.append((rank, doc_id, relevance))
    positive_rels = [value for value in rels.values() if value > 0]
    ideal = sorted(positive_rels, reverse=True)[:10]
    # EOS retrieval_eval.go defines graded nDCG with linear positive
    # relevance gains.  Keep the gate recomputation on that established
    # convention; exponential gains would silently change candidate deltas.
    idcg = sum(value / math.log2(index + 2) for index, value in enumerate(ideal))
    dcg = sum(relevance / math.log2(rank + 1) for rank, _doc, relevance in ranks if rank <= 10 and relevance > 0)
    ndcg = dcg / idcg if idcg else 0.0
    hit = sum(1 for rank, doc_id, relevance in ranks if rank <= 100 and relevance > 0 and doc_id in rels)
    recall = hit / len(positive_rels) if positive_rels else 0.0
    return {"ndcg_at_10": ndcg, "recall_at_100": recall}


def _parse_native_result(metrics_path: Path, tsv_path: Path, per_query_path: Path, *, frozen_info: dict[str, Any], role: str, domain: str, gate_binding: dict[str, Any], runtime_binding: dict[str, Any], production: bool) -> dict[str, Any]:
    for path, label in ((metrics_path, "native metrics"), (tsv_path, "native metrics TSV"), (per_query_path, "native per-query output")):
        require_regular_file(path, f"{role}/{domain} {label}")
        if path.stat().st_size == 0:
            raise GateContractError(f"{role}/{domain} {label}: empty output")
    metrics = read_json(metrics_path, f"{role}/{domain} native metrics")
    required_metrics = {"schema", "dataset", "artifact", "backend", "inputs", "config", "dense", "rows", "gate_binding", "runtime_binding"}
    if not required_metrics.issubset(metrics):
        raise GateContractError(f"{role}/{domain} native metrics: native evaluator fields missing")
    if metrics["schema"] != NATIVE_METRICS_SCHEMA or metrics["dataset"] != domain:
        raise GateContractError(f"{role}/{domain}: native metrics schema/dataset mismatch")
    _require_binding(metrics["gate_binding"], gate_binding, f"{role}/{domain} native metrics.gate_binding")
    _require_runtime_binding(metrics["runtime_binding"], runtime_binding, f"{role}/{domain} native metrics.runtime_binding")
    package = frozen_info["anchor" if role == "anchor" else "candidate"]
    artifact_path = resolve_path(metrics["artifact"], metrics_path.parent, f"{role}/{domain}.native_metrics.artifact")
    if artifact_path != Path(package["path"]):
        raise GateContractError(f"{role}/{domain}: native metrics artifact/package substitution")
    inputs = require_mapping(metrics["inputs"], f"{role}/{domain}.native_metrics.inputs")
    dataset = frozen_info["source"]["dataset_by_domain"][domain]
    if inputs.get("corpus_path") != dataset["corpus"]["path"] or inputs.get("corpus_sha256") != dataset["corpus"]["sha256"] or inputs.get("queries_path") != dataset["queries"]["path"] or inputs.get("queries_sha256") != dataset["queries"]["sha256"] or inputs.get("qrels_path") != frozen_info["qrels"][domain]["path"] or inputs.get("qrels_sha256") != frozen_info["qrels"][domain]["sha256"] or inputs.get("workload_sha256") != frozen_info["workload"]["sha256"] or inputs.get("approved_workload_sha256") != frozen_info["approved_workload"]["sha256"]:
        raise GateContractError(f"{role}/{domain}: native metrics dataset/qrels binding mismatch")
    if inputs.get("queries") != frozen_info["qrels"][domain]["query_count"] or inputs.get("relevant_pairs") != frozen_info["qrels"][domain]["qrels_pair_count"]:
        raise GateContractError(f"{role}/{domain}: native metrics workload counts mismatch")
    config = require_mapping(metrics["config"], f"{role}/{domain}.native_metrics.config")
    if config.get("dimension") != DIMENSION or config.get("split") != HELDOUT_SPLIT or config.get("score_mode") != "turboquant_ip_prepared" or config.get("package_mode") != "native_mll_sibling" or config.get("batch_size") != BATCH_SIZE or config.get("top_k") != TOP_K or config.get("per_query_top_k") != TOP_K or config.get("bits") != [Q3_BITS, Q5_BITS] or config.get("quantizer_seed") != TURBOQUANT_SEED or config.get("max_docs", 0) != 0 or config.get("max_queries", 0) != 0 or config.get("rerank_overfetch", []) != [] or config.get("rerank_bits", 0) != 0:
        raise GateContractError(f"{role}/{domain}: native evaluator config mismatch")
    rows = metrics["rows"]
    if not isinstance(rows, list) or len(rows) != 2:
        raise GateContractError(f"{role}/{domain}: native metrics must contain exactly q3 and q5 rows")
    row_by_bits: dict[int, dict[str, Any]] = {}
    for index, row in enumerate(rows):
        node = require_mapping(row, f"{role}/{domain}.native_metrics.rows[{index}]")
        bits = require_integer(node.get("bits"), f"{role}/{domain}.native_metrics.rows[{index}].bits", minimum=1)
        if bits in row_by_bits or bits not in {Q3_BITS, Q5_BITS}:
            raise GateContractError(f"{role}/{domain}: unexpected/duplicate TurboQuant bits")
        expected_method = f"turboquant_ip_b{bits}"
        if node.get("method") != expected_method or node.get("rerank_overfetch", 0) != 0:
            raise GateContractError(f"{role}/{domain}: scoring method/package-mode mismatch")
        row_by_bits[bits] = node
    if set(row_by_bits) != {Q3_BITS, Q5_BITS}:
        raise GateContractError(f"{role}/{domain}: q3/q5 rows are incomplete")
    dense_node = require_mapping(metrics["dense"], f"{role}/{domain}.native_metrics.dense")
    dense_native = _quality_pair(dense_node.get("quality"), f"{role}/{domain}.native_metrics.dense.quality")
    qrels = frozen_info["qrels"][domain]
    corpus_ids = parse_corpus_ids(Path(frozen_info["source"]["dataset_by_domain"][domain]["corpus"]["path"]), f"{role}/{domain}.corpus")
    if not all(set(rels).issubset(corpus_ids) for rels in qrels["rels"].values()):
        raise GateContractError(f"{role}/{domain}: frozen qrels contain documents absent from the frozen corpus")
    per_query_rows: dict[int, dict[str, dict[str, Any]]] = {Q3_BITS: {}, Q5_BITS: {}}
    try:
        lines = per_query_path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise GateContractError(f"{role}/{domain}: cannot read native per-query output: {exc}") from exc
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        payload = _strict_load(line, f"{role}/{domain}.per-query:{line_number}")
        row = require_mapping(payload, f"{role}/{domain}.per-query:{line_number}")
        required = {"schema", "dataset", "query_id", "method", "bits", "scoring_surface", "quantizer_seed", "relevant_count", "quality", "top_k", "dense_quality", "dense_top_k", "gate_binding", "runtime_binding"}
        if not required.issubset(row):
            raise GateContractError(f"{role}/{domain}.per-query:{line_number}: native evidence fields missing")
        _require_binding(row["gate_binding"], gate_binding, f"{role}/{domain}.per-query:{line_number}.gate_binding")
        _require_runtime_binding(row["runtime_binding"], runtime_binding, f"{role}/{domain}.per-query:{line_number}.runtime_binding")
        if row["schema"] != NATIVE_PER_QUERY_SCHEMA or row["dataset"] != domain or row["scoring_surface"] != "turboquant_ip_prepared" or row["quantizer_seed"] != TURBOQUANT_SEED:
            raise GateContractError(f"{role}/{domain}.per-query:{line_number}: native per-query identity mismatch")
        bits = require_integer(row["bits"], f"{role}/{domain}.per-query:{line_number}.bits", minimum=1)
        if bits not in per_query_rows or row["method"] != f"turboquant_ip_b{bits}":
            raise GateContractError(f"{role}/{domain}.per-query:{line_number}: unexpected scoring method/bits")
        qid = require_qid(row["query_id"], f"{role}/{domain}.per-query:{line_number}.query_id")
        if qid not in qrels["rels"] or qid in per_query_rows[bits]:
            raise GateContractError(f"{role}/{domain}.per-query:{line_number}: qid set is not exactly the frozen workload")
        rels = qrels["rels"][qid]
        if row["relevant_count"] != len(rels):
            raise GateContractError(f"{role}/{domain}.per-query:{line_number}: relevant-count mismatch")
        compact_top = row["top_k"]
        dense_top = row["dense_top_k"]
        if not isinstance(compact_top, list) or not isinstance(dense_top, list):
            raise GateContractError(f"{role}/{domain}.per-query:{line_number}: rankings must be arrays")
        compact_metrics = _rank_metrics(compact_top, rels, corpus_ids, f"{role}/{domain}.per-query:{line_number}.compact")
        dense_metrics = _rank_metrics(dense_top, rels, corpus_ids, f"{role}/{domain}.per-query:{line_number}.dense")
        quality = _quality_pair(row["quality"], f"{role}/{domain}.per-query:{line_number}.quality")
        dense_quality = _quality_pair(row["dense_quality"], f"{role}/{domain}.per-query:{line_number}.dense_quality")
        if abs(quality["ndcg_at_10"] - compact_metrics["ndcg_at_10"]) > COMPARISON_EPSILON or abs(quality["recall_at_100"] - compact_metrics["recall_at_100"]) > COMPARISON_EPSILON:
            raise GateContractError(f"{role}/{domain}.per-query:{line_number}: q3/q5 quality is not recomputed from native ranking")
        if abs(dense_quality["ndcg_at_10"] - dense_metrics["ndcg_at_10"]) > COMPARISON_EPSILON or abs(dense_quality["recall_at_100"] - dense_metrics["recall_at_100"]) > COMPARISON_EPSILON:
            raise GateContractError(f"{role}/{domain}.per-query:{line_number}: dense quality is not recomputed from native ranking")
        boundary = None
        if domain == "nfcorpus" and qid in frozen_info["approved_workload"]["descriptor"]["nfcorpus_boundary_qids"]:
            if "boundary_evidence" in row:
                boundary = _validate_nf_boundary_evidence(row["boundary_evidence"], qid=qid, compact_top=compact_top, dense_top=dense_top, label=f"{role}/{domain}.per-query:{line_number}.boundary_evidence")
            else:
                boundary = _derive_nf_boundary_evidence(qid, compact_top, dense_top)
        elif "boundary_evidence" in row:
            raise GateContractError(f"{role}/{domain}.per-query:{line_number}: unexpected NF boundary evidence for non-boundary query")
        per_query_rows[bits][qid] = {"qid": qid, "q3": compact_metrics, "q5": compact_metrics, "dense": dense_metrics, "quality": quality, "dense_quality": dense_quality, "compact_top": compact_top, "dense_top": dense_top, "boundary": boundary}
    expected_qids = set(qrels["qids"])
    if any(set(rows_by_qid) != expected_qids for rows_by_qid in per_query_rows.values()):
        raise GateContractError(f"{role}/{domain}: native per-query qids do not equal complete frozen qrels workload")
    for qid in qrels["qids"]:
        q3_dense = per_query_rows[Q3_BITS][qid]
        q5_dense = per_query_rows[Q5_BITS][qid]
        if q3_dense["dense_top"] != q5_dense["dense_top"] or q3_dense["dense"] != q5_dense["dense"]:
            raise GateContractError(f"{role}/{domain}: dense evidence differs between q3 and q5 rows for {qid}")
    aggregates = {
        "dense": {**{metric: math.fsum(per_query_rows[Q3_BITS][qid]["dense"][metric] for qid in qrels["qids"]) / len(qrels["qids"]) for metric in METRICS_BY_SURFACE["dense"]}, "recall_at_100": math.fsum(per_query_rows[Q3_BITS][qid]["dense"]["recall_at_100"] for qid in qrels["qids"]) / len(qrels["qids"])},
        "q3": {metric: math.fsum(per_query_rows[Q3_BITS][qid]["q3"][metric] for qid in qrels["qids"]) / len(qrels["qids"]) for metric in METRICS_BY_SURFACE["q3"]},
        "q5": {**{metric: math.fsum(per_query_rows[Q5_BITS][qid]["q5"][metric] for qid in qrels["qids"]) / len(qrels["qids"]) for metric in METRICS_BY_SURFACE["q5"]}, "recall_at_100": math.fsum(per_query_rows[Q5_BITS][qid]["q5"]["recall_at_100"] for qid in qrels["qids"]) / len(qrels["qids"])},
    }
    if any(
        abs(actual - expected) > COMPARISON_EPSILON
        for actual, expected in (
            (dense_native["ndcg_at_10"], aggregates["dense"]["ndcg_at_10"]),
            (dense_native["recall_at_100"], aggregates["dense"]["recall_at_100"]),
            (row_by_bits[Q3_BITS]["quality"]["ndcg_at_10"], aggregates["q3"]["ndcg_at_10"]),
            (row_by_bits[Q3_BITS]["quality"]["recall_at_100"], aggregates["q3"]["recall_at_100"]),
            (row_by_bits[Q5_BITS]["quality"]["ndcg_at_10"], aggregates["q5"]["ndcg_at_10"]),
            (row_by_bits[Q5_BITS]["quality"]["recall_at_100"], aggregates["q5"]["recall_at_100"]),
        )
    ):
        raise GateContractError(f"{role}/{domain}: native aggregate metrics disagree with recomputed per-query evidence")
    boundary_records = {qid: per_query_rows[Q3_BITS][qid]["boundary"] for qid in frozen_info["approved_workload"]["descriptor"]["nfcorpus_boundary_qids"]} if domain == "nfcorpus" else {}
    if domain == "nfcorpus" and any(value is None for value in boundary_records.values()):
        raise GateContractError(f"{role}/{domain}: native NF boundary evidence is incomplete")
    return {"role": role, "domain": domain, "metrics_path": str(metrics_path), "metrics_sha256": sha256_file(metrics_path), "metrics_tsv_path": str(tsv_path), "metrics_tsv_sha256": sha256_file(tsv_path), "per_query_path": str(per_query_path), "per_query_sha256": sha256_file(per_query_path), "aggregate": aggregates, "rows_by_qid": {qid: {"dense": per_query_rows[Q3_BITS][qid]["dense"], "q3": per_query_rows[Q3_BITS][qid]["q3"], "q5": per_query_rows[Q5_BITS][qid]["q5"], "compact_top": per_query_rows[Q3_BITS][qid]["compact_top"], "dense_top": per_query_rows[Q3_BITS][qid]["dense_top"]} for qid in qrels["qids"]}, "boundary_evidence": {"rank_window": [80, 120], "qids": frozen_info["approved_workload"]["descriptor"]["nfcorpus_boundary_qids"] if domain == "nfcorpus" else [], "records": boundary_records, "source": "native_per_query_boundary_evidence"}}


def at_least(value: float, threshold: float) -> bool:
    return value + COMPARISON_EPSILON >= threshold


def _aggregate_nf_boundary_safety(anchor: dict[str, Any], candidate: dict[str, Any], frozen_info: dict[str, Any]) -> dict[str, Any]:
    """Compare candidate recall on an anchor-defined rank-window doc set.

    The boundary qids come from the frozen workload descriptor, not from
    candidate outcomes.  For each fixed qid, relevant documents appearing in
    the anchor q3 ranking at ranks 80..120 form the reusable denominator.  We
    compare recall@100 for that same set in anchor and candidate rankings and
    aggregate the per-qid recalls across all evaluable fixed qids.  A qid with
    no anchor relevant document in the window is reported as not evaluable;
    the descriptor must provide at least one evaluable qid for this guard to
    be meaningful.
    """

    descriptor = frozen_info["approved_workload"]["descriptor"]
    qids = list(descriptor["nfcorpus_boundary_qids"])
    qrels = frozen_info["qrels"]["nfcorpus"]["rels"]
    per_qid: dict[str, Any] = {}
    anchor_recalls: list[float] = []
    candidate_recalls: list[float] = []
    for qid in qids:
        rels = qrels[qid]
        anchor_top = anchor["rows_by_qid"][qid]["compact_top"]
        candidate_top = candidate["rows_by_qid"][qid]["compact_top"]
        anchor_docs = {
            item["doc_id"]
            for item in anchor_top
            if 80 <= item["rank"] <= 120 and rels.get(item["doc_id"], 0.0) > 0
        }
        if not anchor_docs:
            per_qid[qid] = {"anchor_defined_relevant_docs": [], "anchor_recall_at_100": None, "candidate_recall_at_100": None, "evaluable": False, "pass": True}
            continue
        anchor_hits = sum(1 for item in anchor_top if item["rank"] <= 100 and item["doc_id"] in anchor_docs)
        candidate_hits = sum(1 for item in candidate_top if item["rank"] <= 100 and item["doc_id"] in anchor_docs)
        anchor_recall = anchor_hits / len(anchor_docs)
        candidate_recall = candidate_hits / len(anchor_docs)
        anchor_recalls.append(anchor_recall)
        candidate_recalls.append(candidate_recall)
        per_qid[qid] = {
            "anchor_defined_relevant_docs": sorted(anchor_docs),
            "anchor_recall_at_100": anchor_recall,
            "candidate_recall_at_100": candidate_recall,
            "delta": candidate_recall - anchor_recall,
            "evaluable": True,
            "pass": at_least(candidate_recall, anchor_recall),
        }
    evaluable_count = len(anchor_recalls)
    if evaluable_count:
        anchor_mean = math.fsum(anchor_recalls) / evaluable_count
        candidate_mean = math.fsum(candidate_recalls) / evaluable_count
        passed = at_least(candidate_mean, anchor_mean)
    else:
        anchor_mean = None
        candidate_mean = None
        passed = False
    return {
        "schema": "eos.aoqt.nfcorpus_boundary_safety.v1",
        "rank_window": [80, 120],
        "recall_cutoff": 100,
        "qids": qids,
        "aggregation": "macro_over_fixed_evaluable_qids",
        "evaluable_qid_count": evaluable_count,
        "anchor_recall_at_100": anchor_mean,
        "candidate_recall_at_100": candidate_mean,
        "delta": None if anchor_mean is None or candidate_mean is None else candidate_mean - anchor_mean,
        "per_qid": per_qid,
        "pass": passed,
    }


def aggregate_gate(results: dict[tuple[str, str], dict[str, Any]], frozen_info: dict[str, Any]) -> dict[str, Any]:
    failures: list[str] = []
    domain_reports: dict[str, Any] = {}
    for domain in DOMAINS:
        anchor = results[("anchor", domain)]
        candidate = results[("candidate", domain)]
        delta = {surface: {metric: candidate["aggregate"][surface][metric] - anchor["aggregate"][surface][metric] for metric in METRICS_BY_SURFACE[surface]} for surface in SURFACES}
        query_safety: dict[str, Any] = {}
        for qid in frozen_info["approved_workload"]["descriptor"]["query_ids_by_domain"][domain]:
            qdelta = {surface: {metric: candidate["rows_by_qid"][qid][surface][metric] - anchor["rows_by_qid"][qid][surface][metric] for metric in METRICS_BY_SURFACE[surface]} for surface in SURFACES}
            checks = {"q3_ndcg_at_10_non_regressing": at_least(qdelta["q3"]["ndcg_at_10"], THRESHOLDS["q3_query_ndcg_at_10_delta_min"]), "dense_ndcg_at_10_within_tolerance": at_least(qdelta["dense"]["ndcg_at_10"], THRESHOLDS["dense_query_ndcg_at_10_delta_min"]), "q5_ndcg_at_10_within_tolerance": at_least(qdelta["q5"]["ndcg_at_10"], THRESHOLDS["q5_query_ndcg_at_10_delta_min"]), "q3_recall_at_100_non_regressing": at_least(qdelta["q3"]["recall_at_100"], THRESHOLDS["q3_query_recall_at_100_delta_min"])}
            query_safety[qid] = {"delta": qdelta, "checks": checks, "pass": all(checks.values())}
        safety = {"q3_ndcg_at_10_non_regressing": at_least(delta["q3"]["ndcg_at_10"], THRESHOLDS["q3_domain_ndcg_at_10_delta_min"]), "dense_ndcg_at_10_within_tolerance": at_least(delta["dense"]["ndcg_at_10"], THRESHOLDS["dense_domain_ndcg_at_10_delta_min"]), "q5_ndcg_at_10_within_tolerance": at_least(delta["q5"]["ndcg_at_10"], THRESHOLDS["q5_domain_ndcg_at_10_delta_min"]), "q3_recall_at_100_non_regressing": at_least(delta["q3"]["recall_at_100"], THRESHOLDS["q3_domain_recall_at_100_delta_min"])}
        failures.extend(f"domain:{domain}:{name}" for name, passed in safety.items() if not passed)
        boundary = None
        if domain == "nfcorpus":
            boundary = _aggregate_nf_boundary_safety(anchor, candidate, frozen_info)
            if not boundary["pass"]:
                failures.append("domain:nfcorpus:nfcorpus_boundary_safety")
        # Per-query checks remain diagnostics.  Acceptance is defined by the
        # requested macro and per-domain safety contract plus the fixed NF
        # boundary guard; a single query is not an additional blanket floor.
        domain_pass = all(safety.values()) and (boundary is None or boundary["pass"])
        domain_reports[domain] = {"anchor": anchor["aggregate"], "candidate": candidate["aggregate"], "delta": delta, "query_safety": query_safety, "boundary": boundary, "safety": safety, "pass": domain_pass}
    macro_delta = {surface: {metric: math.fsum(domain_reports[domain]["delta"][surface][metric] for domain in DOMAINS) / len(DOMAINS) for metric in METRICS_BY_SURFACE[surface]} for surface in SURFACES}
    macro_checks = {"q3_macro_ndcg_at_10": at_least(macro_delta["q3"]["ndcg_at_10"], THRESHOLDS["q3_macro_ndcg_at_10_delta_min"]), "dense_macro_ndcg_at_10": at_least(macro_delta["dense"]["ndcg_at_10"], THRESHOLDS["dense_macro_ndcg_at_10_delta_min"]), "q5_macro_ndcg_at_10": at_least(macro_delta["q5"]["ndcg_at_10"], THRESHOLDS["q5_macro_ndcg_at_10_delta_min"]), "q3_macro_recall_at_100": at_least(macro_delta["q3"]["recall_at_100"], THRESHOLDS["q3_macro_recall_at_100_delta_min"])}
    failures.extend(f"macro:{name}" for name, passed in macro_checks.items() if not passed)
    return {"domains": domain_reports, "macro": {"q3_ndcg_at_10_delta": macro_delta["q3"]["ndcg_at_10"], "dense_ndcg_at_10_delta": macro_delta["dense"]["ndcg_at_10"], "q5_ndcg_at_10_delta": macro_delta["q5"]["ndcg_at_10"], "q3_recall_at_100_delta": macro_delta["q3"]["recall_at_100"], "checks": macro_checks, "pass": all(macro_checks.values())}, "thresholds": copy.deepcopy(THRESHOLDS), "failures": failures, "gate_pass": not failures}


def _validate_metrics_tsv(path: Path, domain: str, label: str) -> None:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise GateContractError(f"{label}: cannot read TSV: {exc}") from exc
    if len(lines) < 3 or not lines[0].startswith("dataset\trow\tbits\tmethod\t"):
        raise GateContractError(f"{label}: native metrics TSV header/rows missing")
    seen: set[int] = set()
    dense_seen = False
    for line in lines[1:]:
        if not line.strip():
            continue
        fields = line.split("\t")
        if len(fields) < 4 or fields[0] != domain:
            raise GateContractError(f"{label}: native TSV contains an unexpected row")
        if fields[1] == "dense":
            if dense_seen or fields[2] or fields[3] != "float32":
                raise GateContractError(f"{label}: native dense TSV row is malformed or duplicated")
            dense_seen = True
        elif fields[1] == "quantized":
            if fields[2] not in {str(Q3_BITS), str(Q5_BITS)} or fields[3] != f"turboquant_ip_b{fields[2]}":
                raise GateContractError(f"{label}: native TSV q3/q5 row semantics mismatch")
            bit = int(fields[2])
            if bit in seen:
                raise GateContractError(f"{label}: native TSV q3/q5 row duplicated")
            seen.add(bit)
        else:
            raise GateContractError(f"{label}: native TSV row kind is not dense/quantized")
    if not dense_seen or seen != {Q3_BITS, Q5_BITS}:
        raise GateContractError(f"{label}: native TSV q3/q5 rows missing")


def _write_execution_receipt(path: Path, *, frozen_info: dict[str, Any], role: str, domain: str, nonce: str, argv: list[str], binding: dict[str, Any], runtime_binding: dict[str, Any], outputs: dict[str, Path], result: dict[str, Any], exit_status: int) -> dict[str, Any]:
    source = frozen_info["source"]
    package = frozen_info["anchor" if role == "anchor" else "candidate"]
    dataset = source["dataset_by_domain"][domain]
    receipt: dict[str, Any] = {
        "schema": EXECUTION_RECEIPT_SCHEMA,
        "gate_id": frozen_info["manifest"]["gate_id"],
        "nonce": nonce,
        "role": role,
        "domain": domain,
        "producer": "aoqt-heldout-gate-local-subprocess",
        "native_evaluator": True,
        "exit_status": exit_status,
        "executable": source["binary"],
        "binary_sha256": source["binary"]["sha256"],
        "gate_script": frozen_info["manifest"]["provenance"]["gate_script"],
        "argv": argv,
        "argv_sha256": sha256_json(argv),
        "cwd": source["cwd"],
        "config": binding["config"],
        "package": package,
        "dataset": {"domain": domain, "dataset_id": dataset["dataset_id"], "dataset_dir": dataset["dataset_dir"], "manifest": dataset["manifest"], "corpus": dataset["corpus"], "queries": dataset["queries"]},
        "qrels": frozen_info["qrels"][domain],
        "workload": frozen_info["workload"],
        "approved_workload": frozen_info["approved_workload"],
        "frozen_manifest": {"path": frozen_info["path"], "file_sha256": frozen_info["expected_external_sha256"], "manifest_sha256": frozen_info["manifest"]["manifest_sha256"]},
        "gate_binding": binding,
        "runtime_binding": runtime_binding,
        "outputs": {"metrics": {"path": str(outputs["metrics"]), "sha256": result["metrics_sha256"], "bytes": outputs["metrics"].stat().st_size}, "metrics_tsv": {"path": str(outputs["metrics_tsv"]), "sha256": result["metrics_tsv_sha256"], "bytes": outputs["metrics_tsv"].stat().st_size}, "per_query": {"path": str(outputs["per_query"]), "sha256": result["per_query_sha256"], "bytes": outputs["per_query"].stat().st_size}},
        "boundary_evidence": result["boundary_evidence"],
    }
    receipt["receipt_sha256"] = digest_without(receipt, "receipt_sha256")
    write_json_new(path, receipt)
    return {"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size, "payload": receipt}


def run_harness(frozen_path: Path, *, expected_frozen_manifest_sha256: str, expected_workload_manifest_sha256: str, expected_binary_sha256: str, expected_anchor_package_sha256: str, expected_anchor_manifest_sha256: str, expected_anchor_embedding_space_id: str, timeout_seconds: int = 3600, test_mode: bool = False) -> dict[str, Any]:
    """Run the pinned evaluator and aggregate only its fresh bound outputs."""

    if timeout_seconds <= 0:
        raise GateContractError("timeout_seconds must be positive")
    expected_binary = require_sha256(expected_binary_sha256, "expected EOS binary sha256")
    expected_workload = require_sha256(expected_workload_manifest_sha256, "expected approved workload manifest sha256")
    supplied_anchor = {"artifact_sha256": require_sha256(expected_anchor_package_sha256, "expected anchor package sha256"), "package_manifest_sha256": require_sha256(expected_anchor_manifest_sha256, "expected anchor manifest sha256"), "embedding_space_id": require_sha256(expected_anchor_embedding_space_id, "expected anchor embedding-space id")}
    if not test_mode:
        known = {"artifact_sha256": D384_ANCHOR_PACKAGE_SHA256, "package_manifest_sha256": D384_ANCHOR_MANIFEST_SHA256, "embedding_space_id": D384_ANCHOR_EMBEDDING_SPACE_ID}
        if supplied_anchor != known:
            raise GateContractError("caller-supplied anchor identity is not the pinned current D384 anchor")
    # Check caller-provided immutable identities before opening the frozen
    # manifest.  This keeps a weak-anchor attempt from being masked by an
    # unrelated package-format failure in a production plan.
    frozen_info = validate_frozen_manifest(frozen_path, expected_frozen_manifest_sha256=expected_frozen_manifest_sha256, production=not test_mode)
    if frozen_info["source"]["binary"]["sha256"] != expected_binary:
        raise GateContractError("expected EOS binary sha256 does not match frozen binary")
    if frozen_info["approved_workload"]["sha256"] != expected_workload:
        raise GateContractError("external approved workload manifest sha256 mismatch")
    if frozen_info["anchor"]["package_identity"] != supplied_anchor:
        raise GateContractError("frozen anchor identity does not equal caller-supplied immutable anchor identity")
    source = frozen_info["source"]
    binary_path = Path(source["binary"]["path"])
    require_regular_file(binary_path, "frozen EOS executable", executable=True)
    output_root = frozen_info["output_root"]
    nonce = uuid.uuid4().hex
    if not NONCE_RE.fullmatch(nonce):
        raise GateContractError("internal nonce generation failed")
    run_dir = output_root / f"aoqt-{nonce}"
    if run_dir.exists() or run_dir.is_symlink():
        raise GateContractError("fresh nonce run directory already exists")
    run_dir.mkdir(mode=0o700)
    run_dir_stat = run_dir.stat()
    run_dir_identity = (run_dir_stat.st_dev, run_dir_stat.st_ino)
    results: dict[tuple[str, str], dict[str, Any]] = {}
    receipts: list[dict[str, Any]] = []
    commands: list[dict[str, Any]] = []
    try:
        for role in ROLES:
            for domain in DOMAINS:
                _assert_run_directory(run_dir, run_dir_identity, "fresh evaluator run directory")
                outputs = {"metrics": run_dir / OUTPUT_LAYOUT["metrics"].format(role=role, domain=domain), "metrics_tsv": run_dir / OUTPUT_LAYOUT["metrics_tsv"].format(role=role, domain=domain), "per_query": run_dir / OUTPUT_LAYOUT["per_query"].format(role=role, domain=domain)}
                receipt_path = run_dir / OUTPUT_LAYOUT["receipt"].format(role=role, domain=domain)
                for path in (*outputs.values(), receipt_path, run_dir / OUTPUT_LAYOUT["attestation"]):
                    if path.exists() or path.is_symlink():
                        raise GateContractError(f"output path was not absent before evaluator invocation: {path}")
                    path_within(path, run_dir, "evaluator output")
                argv = build_evaluator_argv(frozen_info, role, domain, outputs)
                # Keep a second independent comparison in the execution
                # path: tests can monkeypatch the builder, but production
                # still refuses any argv that is not the canonical command.
                _validate_evaluator_argv(argv, frozen_info, role, domain, outputs)
                binding = _binding_for_run(frozen_info, role, domain, nonce, argv, outputs)
                runtime_binding = _runtime_binding_for_run(frozen_info, role, domain, nonce, argv, outputs)
                env = _harness_environment(frozen_info, nonce, binding, test_mode=test_mode)
                try:
                    completed = subprocess.run(argv, cwd=source["cwd"], env=env, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=timeout_seconds, check=False)
                except (OSError, subprocess.SubprocessError) as exc:
                    raise GateContractError(f"{role}/{domain}: evaluator invocation failed: {exc}") from exc
                commands.append({"role": role, "domain": domain, "argv": argv, "argv_sha256": sha256_json(argv), "cwd": source["cwd"], "exit_status": completed.returncode, "stdout_sha256": sha256_bytes(completed.stdout.encode("utf-8")), "stderr_sha256": sha256_bytes(completed.stderr.encode("utf-8"))})
                if completed.returncode != 0:
                    raise GateContractError(f"{role}/{domain}: native evaluator exited {completed.returncode}: {completed.stderr[-500:]}")
                _assert_run_directory(run_dir, run_dir_identity, "fresh evaluator run directory")
                # The child is allowed to read frozen inputs but never to
                # rewrite them.  Revalidate every bound file after each
                # subprocess returns so a malicious/wrong evaluator cannot
                # mutate the package, qrels, dataset, exclusions, or binary
                # and still have its output accepted against the old digest.
                validate_frozen_manifest(frozen_path, expected_frozen_manifest_sha256=frozen_info["expected_external_sha256"], production=not test_mode)
                for path in outputs.values():
                    require_regular_file(path, f"{role}/{domain} evaluator output")
                _validate_metrics_tsv(outputs["metrics_tsv"], domain, f"{role}/{domain}.metrics_tsv")
                result = _parse_native_result(outputs["metrics"], outputs["metrics_tsv"], outputs["per_query"], frozen_info=frozen_info, role=role, domain=domain, gate_binding=binding, runtime_binding=runtime_binding, production=not test_mode)
                receipt = _write_execution_receipt(receipt_path, frozen_info=frozen_info, role=role, domain=domain, nonce=nonce, argv=argv, binding=binding, runtime_binding=runtime_binding, outputs=outputs, result=result, exit_status=completed.returncode)
                result["receipt"] = receipt
                results[(role, domain)] = result
                receipts.append(receipt)
        summary = aggregate_gate(results, frozen_info)
        _assert_run_directory(run_dir, run_dir_identity, "fresh evaluator run directory")
        command_by_key = {(item["role"], item["domain"]): item for item in commands}
        output_bindings = [{"role": role, "domain": domain, "metrics": results[(role, domain)]["metrics_sha256"], "metrics_tsv": results[(role, domain)]["metrics_tsv_sha256"], "per_query": results[(role, domain)]["per_query_sha256"], "receipt": results[(role, domain)]["receipt"]["sha256"], "nonce": nonce, "argv_sha256": command_by_key[(role, domain)]["argv_sha256"], "runtime_binding_sha256": results[(role, domain)]["receipt"]["payload"]["runtime_binding"]["binding_sha256"]} for role in ROLES for domain in DOMAINS]
        report: dict[str, Any] = {
            "schema": ATTESTATION_SCHEMA,
            "gate_id": frozen_info["manifest"]["gate_id"],
            "candidate_id": frozen_info["manifest"]["candidate_id"],
            "anchor_id": frozen_info["manifest"]["anchor_id"],
            "dimension": DIMENSION,
            "split": HELDOUT_SPLIT,
            "evaluation_executed": True,
            "evaluation_invoked_by_gate": True,
            "official_metric_values_read_by_gate": False,
            "nonce": nonce,
            "run_dir": str(run_dir),
            "frozen_manifest": {"path": frozen_info["path"], "file_sha256": frozen_info["expected_external_sha256"], "manifest_sha256": frozen_info["manifest"]["manifest_sha256"]},
            "approved_workload_sha256": frozen_info["approved_workload"]["sha256"],
            "anchor_identity": supplied_anchor,
            "gate_script_sha256": frozen_info["manifest"]["provenance"]["gate_script"]["sha256"],
            "binary_sha256": source["binary"]["sha256"],
            "commands": commands,
            "receipts": receipts,
            "output_bindings": output_bindings,
            "output_bindings_sha256": sha256_json(output_bindings),
            "domains": summary["domains"],
            "macro": summary["macro"],
            "thresholds": summary["thresholds"],
            "failures": summary["failures"],
            "gate_pass": summary["gate_pass"],
        }
        report_path = run_dir / OUTPUT_LAYOUT["attestation"]
        report["attestation_sha256"] = digest_without(report, "attestation_sha256")
        write_json_new(report_path, report)
        return {**report, "attestation_path": str(report_path)}
    except Exception:
        # Leave the fresh run directory for forensic review, but never turn a
        # partial invocation into a quality pass or overwrite its outputs.
        raise


def attest_manifest(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
    raise GateContractError("attest/aggregate receipt import is disabled; production quality gates must use run")


preflight = freeze_manifest
aggregate = attest_manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    freeze = sub.add_parser("freeze", aliases=["preflight"], help="validate plan and freeze absent input/output bindings")
    freeze.add_argument("--plan", "--input-manifest", "--manifest", dest="input_manifest", required=True, type=Path)
    freeze.add_argument("--output-manifest", "--output", dest="output_manifest", required=True, type=Path)
    run = sub.add_parser("run", help="execute the pinned native EOS evaluator and aggregate fresh outputs")
    run.add_argument("--frozen-manifest", "--manifest", dest="frozen_manifest", required=True, type=Path)
    run.add_argument("--expected-frozen-manifest-sha256", required=True)
    run.add_argument("--expected-workload-manifest-sha256", "--expected-approved-workload-sha256", dest="expected_workload_manifest_sha256", required=True)
    run.add_argument("--expected-binary-sha256", required=True)
    run.add_argument("--expected-anchor-package-sha256", required=True)
    run.add_argument("--expected-anchor-manifest-sha256", required=True)
    run.add_argument("--expected-anchor-embedding-space-id", required=True)
    run.add_argument("--timeout-seconds", type=int, default=3600)
    for name in ("attest", "aggregate"):
        sub.add_parser(name, help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.command in {"freeze", "preflight"}:
            frozen = freeze_manifest(args.input_manifest, args.output_manifest)
            print(json.dumps({"ok": True, "mode": "freeze", "manifest_sha256": frozen["manifest_sha256"], "file_sha256": frozen["file_sha256"]}, sort_keys=True))
            return 0
        if args.command == "run":
            report = run_harness(args.frozen_manifest, expected_frozen_manifest_sha256=args.expected_frozen_manifest_sha256, expected_workload_manifest_sha256=args.expected_workload_manifest_sha256, expected_binary_sha256=args.expected_binary_sha256, expected_anchor_package_sha256=args.expected_anchor_package_sha256, expected_anchor_manifest_sha256=args.expected_anchor_manifest_sha256, expected_anchor_embedding_space_id=args.expected_anchor_embedding_space_id, timeout_seconds=args.timeout_seconds)
            print(json.dumps({"ok": report["gate_pass"], "mode": "run", "gate_pass": report["gate_pass"], "attestation_sha256": report["attestation_sha256"], "attestation_path": report["attestation_path"]}, sort_keys=True))
            return 0 if report["gate_pass"] else 1
        raise GateContractError("attest/aggregate receipt import is disabled; use run")
    except (GateError, OSError, TypeError, KeyError, subprocess.SubprocessError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
