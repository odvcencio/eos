#!/usr/bin/env python3
"""Prepare release-gated BEIR inputs for vector distillation.

This adapter is intentionally pre-teacher. It materializes canonical
query/document/relation JSONL inputs from local BEIR assets and records release
eligibility from an explicit reviewed license manifest. It does not run teacher
embedding and does not claim teacher vectors are release-cleared.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

SCHEMA = "eos.release_vector_distill_inputs.v1"
QUERY_SCHEMA = "eos.release_vector_distill_query.v1"
DOCUMENT_SCHEMA = "eos.release_vector_distill_document.v1"
RELATION_SCHEMA = "eos.release_vector_distill_relation.v1"

DEFAULT_DATASETS = ("scifact", "nfcorpus", "fiqa")
REQUIRED_LICENSE_FIELDS = (
    "dataset",
    "license",
    "source_url",
    "raw_sha256",
    "train_allowed",
    "release_train_allowed",
    "commercial_use_allowed",
)


@dataclass(frozen=True)
class Qrel:
    query_id: str
    document_id: str
    score: float


@dataclass(frozen=True)
class RawAsset:
    path: Path
    sha256: str
    kind: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_directory(path: Path) -> str:
    digest = hashlib.sha256()
    for child in sorted(p for p in path.rglob("*") if p.is_file()):
        rel = child.relative_to(path).as_posix()
        digest.update(rel.encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256_file(child).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def sha256_jsonl(path: Path) -> str:
    return sha256_file(path)


def canonical_id(dataset: str, source_id: str) -> str:
    return f"{dataset}:{source_id}"


def compact_json(row: dict[str, Any]) -> str:
    return json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(compact_json(row))
            handle.write("\n")
            count += 1
    return count


def load_jsonl_map(path: Path, id_field: str = "_id") -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{line_no}: invalid JSON: {exc}")
            row_id = row.get(id_field) or row.get("id")
            if row_id is None:
                raise SystemExit(f"{path}:{line_no}: missing {id_field!r}/'id'")
            out[str(row_id)] = row
    return out


def parse_score(value: str, path: Path, line_no: int) -> float:
    try:
        return float(value)
    except ValueError as exc:
        raise SystemExit(f"{path}:{line_no}: invalid qrels score {value!r}") from exc


def read_qrels(path: Path) -> list[Qrel]:
    qrels: list[Qrel] = []
    if not path.exists():
        return qrels
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        for line_no, cols in enumerate(reader, start=1):
            if not cols or all(not c.strip() for c in cols):
                continue
            lowered = [c.strip().lower() for c in cols]
            if line_no == 1 and lowered[0] in {"query-id", "qid", "query_id"}:
                continue
            if len(cols) >= 4:
                query_id, document_id, score_text = cols[0].strip(), cols[2].strip(), cols[3].strip()
            elif len(cols) >= 3:
                query_id, document_id, score_text = cols[0].strip(), cols[1].strip(), cols[2].strip()
            else:
                raise SystemExit(f"{path}:{line_no}: expected 3-column or 4-column qrels row")
            score = parse_score(score_text, path, line_no)
            if score > 0:
                qrels.append(Qrel(query_id=query_id, document_id=document_id, score=score))
    return qrels


def qrel_id_sets(qrels: Iterable[Qrel]) -> tuple[set[str], set[str], set[tuple[str, str]]]:
    queries: set[str] = set()
    documents: set[str] = set()
    pairs: set[tuple[str, str]] = set()
    for qrel in qrels:
        queries.add(qrel.query_id)
        documents.add(qrel.document_id)
        pairs.add((qrel.query_id, qrel.document_id))
    return queries, documents, pairs


def resolve_dataset_dir(raw_root: Path, dataset: str) -> Path:
    nested = raw_root / dataset / dataset
    if nested.is_dir():
        return nested
    flat = raw_root / dataset
    if flat.is_dir() and (flat / "queries.jsonl").is_file():
        return flat
    raise SystemExit(f"dataset {dataset!r} not found under {raw_root} (expected {nested})")


def resolve_raw_asset(raw_root: Path, dataset: str) -> RawAsset:
    archive = raw_root / f"{dataset}.zip"
    if archive.is_file():
        return RawAsset(path=archive, sha256=sha256_file(archive), kind="archive_file")
    extracted = raw_root / dataset
    if extracted.is_dir():
        return RawAsset(path=extracted, sha256=sha256_directory(extracted), kind="extracted_directory")
    raise SystemExit(f"raw asset for dataset {dataset!r} not found under {raw_root}")


def load_license_manifest(path: Path) -> dict[str, dict[str, Any]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"{path}: invalid JSON: {exc}") from exc

    if isinstance(payload, dict) and isinstance(payload.get("datasets"), list):
        entries = payload["datasets"]
    elif isinstance(payload, list):
        entries = payload
    elif isinstance(payload, dict) and "dataset" in payload:
        entries = [payload]
    elif isinstance(payload, dict):
        entries = []
        for dataset, entry in payload.items():
            if isinstance(entry, dict):
                merged = dict(entry)
                merged.setdefault("dataset", dataset)
                entries.append(merged)
    else:
        raise SystemExit(f"{path}: license manifest must be an object or list")

    by_dataset: dict[str, dict[str, Any]] = {}
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise SystemExit(f"{path}: datasets[{index}] must be an object")
        missing = [key for key in REQUIRED_LICENSE_FIELDS if key not in entry]
        if missing:
            raise SystemExit(f"{path}: dataset entry {index} missing required fields: {', '.join(missing)}")
        dataset = str(entry["dataset"])
        if dataset in by_dataset:
            raise SystemExit(f"{path}: duplicate license entry for dataset {dataset!r}")
        by_dataset[dataset] = entry
    return by_dataset


def validate_license(dataset: str, entry: dict[str, Any], raw_asset: RawAsset) -> dict[str, Any]:
    if str(entry["dataset"]) != dataset:
        raise SystemExit(f"license entry dataset mismatch for {dataset!r}: {entry['dataset']!r}")
    if str(entry["raw_sha256"]).lower() != raw_asset.sha256:
        raise SystemExit(
            f"license raw_sha256 mismatch for {dataset}: manifest={entry['raw_sha256']} actual={raw_asset.sha256}"
        )
    for key in ("train_allowed", "release_train_allowed", "commercial_use_allowed"):
        if entry.get(key) is not True:
            raise SystemExit(f"license manifest for {dataset} does not grant {key}=true")
    return {
        "dataset": dataset,
        "license": str(entry["license"]),
        "source_url": str(entry["source_url"]),
        "raw_sha256": raw_asset.sha256,
        "raw_asset_path": str(raw_asset.path),
        "raw_asset_kind": raw_asset.kind,
        "train_allowed": True,
        "release_train_allowed": True,
        "commercial_use_allowed": True,
    }


def document_text(row: dict[str, Any]) -> tuple[str, str, str]:
    title = str(row.get("title") or "")
    body = str(row.get("text") or "")
    if title and body:
        text = f"{title}\n\n{body}"
    else:
        text = title or body
    return title, body, text


def materialize_dataset(raw_root: Path, dataset: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    dataset_dir = resolve_dataset_dir(raw_root, dataset)
    queries_by_id = load_jsonl_map(dataset_dir / "queries.jsonl")
    docs_by_id = load_jsonl_map(dataset_dir / "corpus.jsonl")
    qrels_dir = dataset_dir / "qrels"
    train_qrels = read_qrels(qrels_dir / "train.tsv")
    dev_qrels = read_qrels(qrels_dir / "dev.tsv")
    test_qrels = read_qrels(qrels_dir / "test.tsv")

    heldout_queries, heldout_documents, heldout_pairs = qrel_id_sets([*dev_qrels, *test_qrels])
    selected_qrels: list[Qrel] = []
    dropped = {
        "non_resolvable_query": 0,
        "non_resolvable_document": 0,
        "heldout_query": 0,
        "heldout_document": 0,
        "heldout_pair": 0,
    }
    seen_pairs: set[tuple[str, str]] = set()
    for qrel in train_qrels:
        pair = (qrel.query_id, qrel.document_id)
        if qrel.query_id not in queries_by_id:
            dropped["non_resolvable_query"] += 1
            continue
        if qrel.document_id not in docs_by_id:
            dropped["non_resolvable_document"] += 1
            continue
        if pair in heldout_pairs:
            dropped["heldout_pair"] += 1
            continue
        if qrel.query_id in heldout_queries:
            dropped["heldout_query"] += 1
            continue
        if qrel.document_id in heldout_documents:
            dropped["heldout_document"] += 1
            continue
        if pair in seen_pairs:
            continue
        seen_pairs.add(pair)
        selected_qrels.append(qrel)

    if not selected_qrels:
        raise SystemExit(f"dataset {dataset}: no train-positive qrels remain after leakage/resolvability filtering")

    selected_query_ids = {q.query_id for q in selected_qrels}
    selected_document_ids = {q.document_id for q in selected_qrels}

    query_rows: list[dict[str, Any]] = []
    for source_id in sorted(selected_query_ids):
        source = queries_by_id[source_id]
        query_rows.append(
            {
                "schema": QUERY_SCHEMA,
                "id": canonical_id(dataset, source_id),
                "dataset": dataset,
                "source_query_id": source_id,
                "text": str(source.get("text") or ""),
            }
        )

    document_rows: list[dict[str, Any]] = []
    for source_id in sorted(selected_document_ids):
        source = docs_by_id[source_id]
        title, body, text = document_text(source)
        document_rows.append(
            {
                "schema": DOCUMENT_SCHEMA,
                "id": canonical_id(dataset, source_id),
                "dataset": dataset,
                "source_document_id": source_id,
                "title": title,
                "body": body,
                "text": text,
            }
        )

    relation_rows: list[dict[str, Any]] = []
    for qrel in sorted(selected_qrels, key=lambda r: (r.query_id, r.document_id, r.score)):
        relation_rows.append(
            {
                "schema": RELATION_SCHEMA,
                "dataset": dataset,
                "query_id": canonical_id(dataset, qrel.query_id),
                "positive_doc_id": canonical_id(dataset, qrel.document_id),
                "source_query_id": qrel.query_id,
                "source_document_id": qrel.document_id,
                "score": qrel.score,
            }
        )

    split_manifest = {
        "dataset": dataset,
        "dataset_dir": str(dataset_dir),
        "train_qrels_path": str(qrels_dir / "train.tsv"),
        "dev_qrels_path": str(qrels_dir / "dev.tsv") if (qrels_dir / "dev.tsv").exists() else None,
        "test_qrels_path": str(qrels_dir / "test.tsv") if (qrels_dir / "test.tsv").exists() else None,
        "train_positive_qrels": len(train_qrels),
        "dev_positive_qrels": len(dev_qrels),
        "test_positive_qrels": len(test_qrels),
        "selected_train_positive_qrels": len(relation_rows),
        "selected_queries": len(query_rows),
        "selected_documents": len(document_rows),
        "dropped": dropped,
        "heldout_query_ids_seen": len(heldout_queries),
        "heldout_document_ids_seen": len(heldout_documents),
        "heldout_pair_ids_seen": len(heldout_pairs),
    }
    return query_rows, document_rows, relation_rows, split_manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=Path("datasets/manta-embed-v1/raw"))
    parser.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS), help="Dataset names under --raw-root.")
    parser.add_argument("--license-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=None)
    args = parser.parse_args(argv)
    if args.manifest is None:
        args.manifest = args.output_dir / "manifest.json"
    return args


def build(args: argparse.Namespace) -> dict[str, Any]:
    licenses = load_license_manifest(args.license_manifest)
    missing = [dataset for dataset in args.datasets if dataset not in licenses]
    if missing:
        raise SystemExit(f"license manifest missing dataset entries: {', '.join(missing)}")

    all_queries: list[dict[str, Any]] = []
    all_documents: list[dict[str, Any]] = []
    all_relations: list[dict[str, Any]] = []
    dataset_manifests: list[dict[str, Any]] = []
    license_entries: list[dict[str, Any]] = []

    for dataset in args.datasets:
        raw_asset = resolve_raw_asset(args.raw_root, dataset)
        license_entries.append(validate_license(dataset, licenses[dataset], raw_asset))
        queries, documents, relations, split_manifest = materialize_dataset(args.raw_root, dataset)
        dataset_manifests.append(split_manifest)
        all_queries.extend(queries)
        all_documents.extend(documents)
        all_relations.extend(relations)

    all_queries.sort(key=lambda row: (row["dataset"], row["source_query_id"]))
    all_documents.sort(key=lambda row: (row["dataset"], row["source_document_id"]))
    all_relations.sort(key=lambda row: (row["dataset"], row["source_query_id"], row["source_document_id"]))

    queries_path = args.output_dir / "queries.jsonl"
    documents_path = args.output_dir / "documents.jsonl"
    relations_path = args.output_dir / "relations.jsonl"
    query_count = write_jsonl(queries_path, all_queries)
    document_count = write_jsonl(documents_path, all_documents)
    relation_count = write_jsonl(relations_path, all_relations)

    manifest = {
        "schema": SCHEMA,
        "raw_root": str(args.raw_root),
        "license_manifest_path": str(args.license_manifest),
        "release_allowed": True,
        "teacher_embeddings_release_cleared": False,
        "teacher_embedding_note": "This adapter emits pre-teacher inputs only. Run build_vector_distill_rows.py separately and review teacher/model licensing before marking teacher vectors release-cleared.",
        "legal_gates": {
            "train_allowed": True,
            "release_train_allowed": True,
            "commercial_use_allowed": True,
        },
        "datasets": dataset_manifests,
        "licenses": license_entries,
        "outputs": {
            "queries": {"path": str(queries_path), "rows": query_count, "sha256": sha256_jsonl(queries_path)},
            "documents": {"path": str(documents_path), "rows": document_count, "sha256": sha256_jsonl(documents_path)},
            "relations": {"path": str(relations_path), "rows": relation_count, "sha256": sha256_jsonl(relations_path)},
        },
        "compatible_next_step": {
            "script": "scripts/build_vector_distill_rows.py",
            "example_args": [
                "--queries",
                str(queries_path),
                "--documents",
                str(documents_path),
                "--relations",
                str(relations_path),
                "--group-order",
                "--output",
                "PATH/teacher-vector-distill.jsonl",
            ],
            "caveat": "Do not propagate release_train_allowed to teacher-vector rows without separate teacher/model/data review.",
        },
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = build(args)
    print(
        "wrote release vector-distill inputs: "
        f"queries={manifest['outputs']['queries']['rows']} "
        f"documents={manifest['outputs']['documents']['rows']} "
        f"relations={manifest['outputs']['relations']['rows']} -> {args.output_dir}"
    )
    print(f"manifest -> {args.manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
