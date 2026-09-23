#!/usr/bin/env python3
"""Mask q3 candidate sources on non-promotion score-spectrum guard rows.

The score-spectrum trainer's TurboQuant top-k objective with
`--turboquant-topk-negative-mask q3` selects negatives by candidate source.
This transformer preserves hard-negative eligibility and q3 evidence, but
renames candidate source `q3` to supported non-q3 source `other` for rows whose selector bucket is
not `promotion_11_20`. Base hard/soft/recovery losses still see those
candidates; only the q3 LambdaNDCG auxiliary objective is limited to true
promotion rows.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any


SCHEMA = "eos.score_spectrum_guard_q3_mask.v1"
LEGAL_GATES = {
    "train_allowed_for_research": True,
    "release_train_allowed": False,
    "commercial_use_allowed": False,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", required=True, type=Path)
    parser.add_argument("--selector-records-json", required=True, type=Path)
    parser.add_argument("--output-jsonl", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--masked-source", default="other")
    return parser.parse_args()


def utc_stamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                yield line_number, json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")


def load_selector_buckets(path: Path) -> dict[tuple[str, str], str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload.get("records")
    if not isinstance(records, dict):
        raise ValueError(f"{path}: expected records object")
    buckets: dict[tuple[str, str], str] = {}
    for dataset, rows in records.items():
        if not isinstance(rows, list):
            raise ValueError(f"{path}: records[{dataset!r}] must be a list")
        for row in rows:
            qid = str(row.get("query_id", ""))
            bucket = str(row.get("selected_bucket", ""))
            if not qid or not bucket:
                raise ValueError(f"{path}: selector row missing query_id/selected_bucket")
            buckets[(str(dataset).lower(), qid)] = bucket
    return buckets


def transform_rows(
    input_jsonl: Path,
    buckets: dict[tuple[str, str], str],
    masked_source: str,
) -> tuple[list[dict[str, Any]], Counter[str], dict[str, Counter[str]]]:
    rows: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    by_dataset: dict[str, Counter[str]] = {}
    for _line_number, row in iter_jsonl(input_jsonl):
        dataset = str(row.get("source_dataset", "")).lower()
        qid = str(row.get("source_query_id", ""))
        key = (dataset, qid)
        if key not in buckets:
            raise ValueError(f"{input_jsonl}: missing selector bucket for {dataset}:{qid}")
        bucket = buckets[key]
        row = dict(row)
        row["selector_bucket"] = bucket
        row["q3_topk_candidate_source_policy"] = (
            "q3_topk_only_on_promotion_11_20; masked guard q3 candidates remain hard/recovery eligible"
        )
        dataset_counts = by_dataset.setdefault(dataset, Counter())
        dataset_counts[f"bucket_{bucket}"] += 1
        if bucket != "promotion_11_20":
            sources = list(row.get("candidate_sources", []))
            masked = sum(1 for source in sources if source == "q3")
            if masked:
                row["candidate_sources"] = [masked_source if source == "q3" else source for source in sources]
                counts["q3_sources_masked"] += masked
                counts["rows_with_q3_sources_masked"] += 1
                dataset_counts["q3_sources_masked"] += masked
                dataset_counts["rows_with_q3_sources_masked"] += 1
        else:
            counts["promotion_rows_q3_sources_preserved"] += 1
            dataset_counts["promotion_rows_q3_sources_preserved"] += 1
        rows.append(row)
    return rows, counts, by_dataset


def main() -> None:
    args = parse_args()
    if args.masked_source == "q3":
        raise SystemExit("--masked-source must not be q3")
    buckets = load_selector_buckets(args.selector_records_json)
    rows, counts, by_dataset = transform_rows(args.input_jsonl, buckets, args.masked_source)
    write_jsonl(args.output_jsonl, rows)
    manifest = {
        "schema": SCHEMA,
        "created_utc": utc_stamp(),
        "transformer": "scripts/mask_score_spectrum_guard_q3_sources.py",
        "inputs": {
            "input_jsonl": str(args.input_jsonl),
            "input_jsonl_sha256": sha256_file(args.input_jsonl),
            "selector_records_json": str(args.selector_records_json),
            "selector_records_json_sha256": sha256_file(args.selector_records_json),
        },
        "outputs": {
            "output_jsonl": str(args.output_jsonl),
            "manifest": str(args.manifest),
        },
        "config": {
            "masked_source": args.masked_source,
            "preserved_source": "q3",
            "preserved_bucket": "promotion_11_20",
        },
        "counts": {
            **dict(counts),
            "rows": len(rows),
            "candidate_source_counts": dict(Counter(source for row in rows for source in row.get("candidate_sources", []))),
        },
        "datasets": {dataset: dict(counter) for dataset, counter in sorted(by_dataset.items())},
        "legal_gates": dict(LEGAL_GATES),
        **LEGAL_GATES,
        "quality_claim": False,
    }
    manifest["sha256"] = {"output_jsonl": sha256_file(args.output_jsonl)}
    write_json(args.manifest, manifest)
    print(
        "masked score-spectrum guard q3 sources: "
        f"rows={len(rows)} q3_sources_masked={counts['q3_sources_masked']} "
        f"output={args.output_jsonl}"
    )


if __name__ == "__main__":
    main()
