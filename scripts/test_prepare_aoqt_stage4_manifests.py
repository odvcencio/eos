#!/usr/bin/env python3
"""Tests for AOQT Stage 4 skinny manifest preparation."""

from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import build_aoqt_stage2_calibration as builder  # noqa: E402
import prepare_aoqt_stage4_manifests as prep  # noqa: E402


def write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    return path


def fake_sha(seed: str) -> str:
    return builder.sha256_json({"seed": seed})


def vector(item_id: str, dim: int = 384) -> dict[str, object]:
    values = [0.0] * dim
    values[0] = 1.0
    return {"id": item_id, "embedding": values}


class PrepareAOQTStage4ManifestsTest(unittest.TestCase):
    def anchor(self, root: Path) -> Path:
        return write_json(
            root / "anchor.json",
            {
                "schema": builder.ANCHOR_SCHEMA,
                "package_path": "runs/anchor/d384-pre.mll",
                "package_sha256": fake_sha("anchor-package"),
                "embedding_dim": 384,
                "topology": copy.deepcopy(builder.TOPOLOGY),
                "legal_scope": copy.deepcopy(builder.LEGAL_SCOPE),
            },
        )

    def retrieval_manifest(self, root: Path, dataset: str, anchor: Path) -> Path:
        return write_json(
            root / f"{dataset}.retrieval.json",
            {
                "schema": prep.RETRIEVAL_EXPORT_SCHEMA,
                "dataset": dataset,
                "split": "train",
                "dimension": 384,
                "anchor": {
                    "package_sha256": json.loads(anchor.read_text(encoding="utf-8"))["package_sha256"],
                    "manifest_sha256": builder.sha256_file(anchor),
                },
            },
        )

    def make_vector_manifest(self, root: Path, dataset: str, role: str, ids: list[str], qrels_sha: str) -> Path:
        anchor = self.anchor(root) if not (root / "anchor.json").exists() else root / "anchor.json"
        retrieval = self.retrieval_manifest(root, dataset, anchor)
        vectors = write_jsonl(root / f"{dataset}.{role}.vectors.jsonl", [vector(item_id) for item_id in ids])
        output = root / f"{dataset}.{role}.manifest.json"
        args = prep.parse_args(
            [
                "adapt-vector-cache",
                "--retrieval-manifest",
                str(retrieval),
                "--vector-jsonl",
                str(vectors),
                "--anchor-manifest",
                str(anchor),
                "--dataset",
                dataset,
                "--role",
                role,
                "--qrels-sha256",
                qrels_sha,
                "--output",
                str(output),
            ]
        )
        payload = prep.build_manifest(args)
        write_json(output, payload)
        return output

    def make_score_manifest(self, root: Path, dataset: str, bits: int, query_ids: list[str], doc_ids: list[str], qrels_sha: str, seed: int | None = None) -> Path:
        seed = builder.DEFAULT_TURBOQUANT_SEED if seed is None else seed
        query_manifest = self.make_vector_manifest(root, dataset, "query", query_ids, qrels_sha)
        doc_manifest = self.make_vector_manifest(root, dataset, "doc", doc_ids, qrels_sha)
        rows = []
        for qid in query_ids:
            rows.append(
                {
                    "bits": bits,
                    "dataset": dataset,
                    "query_id": qid,
                    "quantizer_seed": seed,
                    "scoring_surface": "turboquant_ip_prepared",
                    "split": "train",
                    "top_k_limit": 120,
                    "top_k": [
                        {"doc_id": doc_id, "rank": rank, "score": 100.0 - rank, "relevance": 1 if rank == 1 or (dataset == "nfcorpus" and bits == 3 and rank == 80) else 0}
                        for rank, doc_id in enumerate(doc_ids[:120], start=1)
                    ],
                }
            )
        scores = write_jsonl(root / "scores" / f"{dataset}.q{bits}.jsonl", rows)
        output = root / "scores" / f"{dataset}.q{bits}.manifest.json"
        args = prep.parse_args(
            [
                "convert-score-cache",
                "--per-query-jsonl",
                str(scores),
                "--query-vector-manifest",
                str(query_manifest),
                "--doc-vector-manifest",
                str(doc_manifest),
                "--anchor-manifest",
                str(root / "anchor.json"),
                "--dataset",
                dataset,
                "--bits",
                str(bits),
                "--turboquant-seed",
                str(seed),
                "--qrels-sha256",
                qrels_sha,
                "--output",
                str(output),
            ]
        )
        payload = prep.build_manifest(args)
        write_json(output, payload)
        return output

    def test_normalize_exclusions_is_qid_only_and_does_not_copy_qrels_columns(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            legacy = write_json(
                root / "selected.json",
                {
                    "selected_qids": {
                        "fiqa": ["dev-q2", "dev-q1"],
                        "nfcorpus": ["nf-dev-q"],
                        "scifact": ["sci-dev-q"],
                    }
                },
            )
            fiqa_qrels = root / "fiqa.official.qrels"
            fiqa_qrels.write_text("test-q2 0 doc-a 1\ntest-q1 0 doc-b 0\n", encoding="utf-8")
            nf_qrels = root / "nfcorpus.official.qrels"
            nf_qrels.write_text("nf-test-q 0 doc-c 1\n", encoding="utf-8")
            sci_qrels = root / "scifact.official.qrels"
            sci_qrels.write_text("sci-test-q 0 doc-d 1\n", encoding="utf-8")
            args = prep.parse_args(
                [
                    "normalize-exclusions",
                    "--name",
                    "official-test",
                    "--legacy-selected-qids",
                    str(legacy),
                    "--official-test-qrels",
                    str(fiqa_qrels),
                    "--official-test-qrels",
                    str(nf_qrels),
                    "--official-test-qrels",
                    str(sci_qrels),
                    "--output",
                    str(root / "out.json"),
                ]
            )
            payload = prep.build_manifest(args)

        self.assertEqual(payload["schema"], builder.EXCLUSION_SCHEMA)
        self.assertEqual(payload["qids_by_dataset"]["fiqa"], ["dev-q1", "dev-q2", "test-q1", "test-q2"])
        self.assertEqual(payload["qids_by_dataset"]["nfcorpus"], ["nf-dev-q", "nf-test-q"])
        self.assertEqual(payload["qids_by_dataset"]["scifact"], ["sci-dev-q", "sci-test-q"])
        self.assertEqual(set(payload), {"schema", "name", "qids_by_dataset", "source_sha256", "source_sha256_by_file"})
        serialized = json.dumps(payload, sort_keys=True)
        self.assertNotIn("doc-a", serialized)
        self.assertNotIn("doc-b", serialized)

    def test_normalize_exclusions_dedupes_repeated_qrels_qids_stably(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fiqa_qrels = root / "fiqa.official.qrels"
            fiqa_qrels.write_text("q2 0 doc-a 1\nq1 0 doc-b 0\nq2 0 doc-c 1\n", encoding="utf-8")
            nf_qrels = root / "nfcorpus.official.qrels"
            nf_qrels.write_text("nf-q 0 doc-d 1\n", encoding="utf-8")
            sci_qrels = root / "scifact.official.qrels"
            sci_qrels.write_text("sci-q 0 doc-e 1\n", encoding="utf-8")
            args = prep.parse_args(
                [
                    "normalize-exclusions",
                    "--name",
                    "official-test",
                    "--official-test-qrels",
                    str(fiqa_qrels),
                    "--official-test-qrels",
                    str(nf_qrels),
                    "--official-test-qrels",
                    str(sci_qrels),
                    "--output",
                    str(root / "out.json"),
                ]
            )
            payload = prep.build_manifest(args)
        self.assertEqual(payload["qids_by_dataset"]["fiqa"], ["q1", "q2"])

    def test_normalize_exclusions_rejects_nested_doc_or_score_leakage(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            legacy = write_json(root / "selected.json", {"selected_qids": {"fiqa": ["q1"], "nfcorpus": ["q2"], "scifact": ["q3"]}, "doc_ids": ["d1"]})
            args = prep.parse_args(["normalize-exclusions", "--name", "dev4", "--legacy-selected-qids", str(legacy), "--output", str(root / "out.json")])
            with self.assertRaisesRegex(prep.ManifestError, "forbidden field"):
                prep.build_manifest(args)

    def test_normalize_exclusions_rejects_ambiguous_global_qids(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            legacy = write_json(root / "selected.json", {"selected_qids": ["q1"]})
            args = prep.parse_args(["normalize-exclusions", "--name", "dev4", "--legacy-selected-qids", str(legacy), "--output", str(root / "out.json")])
            with self.assertRaisesRegex(prep.ManifestError, "ambiguous legacy global qids"):
                prep.build_manifest(args)

    def test_normalize_exclusions_rejects_missing_dataset_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            legacy = write_json(root / "selected.json", {"selected_qids": {"fiqa": ["q1"], "nfcorpus": ["q2"]}})
            args = prep.parse_args(["normalize-exclusions", "--name", "dev4", "--legacy-selected-qids", str(legacy), "--output", str(root / "out.json")])
            with self.assertRaisesRegex(prep.ManifestError, "missing required dataset coverage"):
                prep.build_manifest(args)

    def test_adapt_vector_cache_outputs_ids_and_hashes_without_vectors(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            qrels_sha = fake_sha("fiqa-qrels")
            anchor = self.anchor(root)
            retrieval = self.retrieval_manifest(root, "fiqa", anchor)
            vectors = write_jsonl(root / "query-vectors.jsonl", [vector("q1"), vector("q2")])
            args = prep.parse_args(
                [
                    "adapt-vector-cache",
                    "--retrieval-manifest",
                    str(retrieval),
                    "--vector-jsonl",
                    str(vectors),
                    "--anchor-manifest",
                    str(anchor),
                    "--dataset",
                    "fiqa",
                    "--role",
                    "query",
                    "--qrels-sha256",
                    qrels_sha,
                    "--output",
                    str(root / "out.json"),
                ]
            )
            payload = prep.build_manifest(args)
            expected_cache_sha256 = builder.sha256_file(vectors)

        self.assertEqual(payload["schema"], builder.VECTOR_CACHE_SCHEMA)
        self.assertEqual(payload["ids"], ["q1", "q2"])
        self.assertEqual(payload["cache_sha256"], expected_cache_sha256)
        self.assertEqual(payload["qrels_sha256"], qrels_sha)
        self.assertNotIn("embedding", json.dumps(payload, sort_keys=True).lower())

    def test_adapt_vector_cache_rejects_dimension_and_duplicate_id_failures(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            anchor = self.anchor(root)
            retrieval = self.retrieval_manifest(root, "fiqa", anchor)
            bad_dim = write_jsonl(root / "bad-dim.jsonl", [vector("q1", dim=383)])
            args = prep.parse_args(
                [
                    "adapt-vector-cache",
                    "--retrieval-manifest",
                    str(retrieval),
                    "--vector-jsonl",
                    str(bad_dim),
                    "--anchor-manifest",
                    str(anchor),
                    "--dataset",
                    "fiqa",
                    "--role",
                    "query",
                    "--qrels-sha256",
                    fake_sha("qrels"),
                    "--output",
                    str(root / "out.json"),
                ]
            )
            with self.assertRaisesRegex(prep.ManifestError, "dimension"):
                prep.build_manifest(args)

            duplicate = write_jsonl(root / "duplicate.jsonl", [vector("q1"), vector("q1")])
            args.vector_jsonl = duplicate
            with self.assertRaisesRegex(prep.ManifestError, "duplicate vector id"):
                prep.build_manifest(args)

    def test_adapt_vector_cache_requires_explicit_retrieval_export_manifest_identity(self) -> None:
        required_fields = ["schema", "dataset", "split", "dimension", "anchor"]
        for missing in required_fields:
            with self.subTest(missing=missing):
                with tempfile.TemporaryDirectory() as tmp:
                    root = Path(tmp)
                    anchor = self.anchor(root)
                    retrieval = self.retrieval_manifest(root, "fiqa", anchor)
                    payload = json.loads(retrieval.read_text(encoding="utf-8"))
                    del payload[missing]
                    write_json(retrieval, payload)
                    vectors = write_jsonl(root / "query-vectors.jsonl", [vector("q1")])
                    args = prep.parse_args(
                        [
                            "adapt-vector-cache",
                            "--retrieval-manifest",
                            str(retrieval),
                            "--vector-jsonl",
                            str(vectors),
                            "--anchor-manifest",
                            str(anchor),
                            "--dataset",
                            "fiqa",
                            "--role",
                            "query",
                            "--qrels-sha256",
                            fake_sha("qrels"),
                            "--output",
                            str(root / "out.json"),
                        ]
                    )
                    with self.assertRaisesRegex((prep.ManifestError, builder.PlanError), "schema mismatch|missing required source field"):
                        prep.build_manifest(args)

    def test_adapt_vector_cache_rejects_empty_or_missing_retrieval_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            anchor = self.anchor(root)
            vectors = write_jsonl(root / "query-vectors.jsonl", [vector("q1")])
            args = prep.parse_args(
                [
                    "adapt-vector-cache",
                    "--retrieval-manifest",
                    str(root / "empty.json"),
                    "--vector-jsonl",
                    str(vectors),
                    "--anchor-manifest",
                    str(anchor),
                    "--dataset",
                    "fiqa",
                    "--role",
                    "query",
                    "--qrels-sha256",
                    fake_sha("qrels"),
                    "--output",
                    str(root / "out.json"),
                ]
            )
            write_json(root / "empty.json", {})
            with self.assertRaisesRegex(builder.PlanError, "schema mismatch"):
                prep.build_manifest(args)
            args.retrieval_manifest = root / "missing.json"
            with self.assertRaisesRegex(prep.ManifestError, "cannot read JSON"):
                prep.build_manifest(args)

    def test_adapt_vector_cache_rejects_retrieval_manifest_anchor_identity_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            anchor = self.anchor(root)
            retrieval = self.retrieval_manifest(root, "fiqa", anchor)
            payload = json.loads(retrieval.read_text(encoding="utf-8"))
            payload["anchor"]["package_sha256"] = fake_sha("other-package")
            write_json(retrieval, payload)
            vectors = write_jsonl(root / "query-vectors.jsonl", [vector("q1")])
            args = prep.parse_args(
                [
                    "adapt-vector-cache",
                    "--retrieval-manifest",
                    str(retrieval),
                    "--vector-jsonl",
                    str(vectors),
                    "--anchor-manifest",
                    str(anchor),
                    "--dataset",
                    "fiqa",
                    "--role",
                    "query",
                    "--qrels-sha256",
                    fake_sha("qrels"),
                    "--output",
                    str(root / "out.json"),
                ]
            )
            with self.assertRaisesRegex(prep.ManifestError, "anchor package_sha256 identity mismatch"):
                prep.build_manifest(args)

    def test_adapt_vector_cache_rejects_retrieval_manifest_metadata_mismatch(self) -> None:
        cases = [
            ("dataset", "nfcorpus", "dataset mismatch"),
            ("split", "dev", "forbidden split token"),
            ("dimension", 383, "embedding dimension must be 384"),
            ("anchor.manifest_sha256", fake_sha("other-manifest"), "anchor manifest_sha256 identity mismatch"),
        ]
        for field, value, pattern in cases:
            with self.subTest(field=field):
                with tempfile.TemporaryDirectory() as tmp:
                    root = Path(tmp)
                    anchor = self.anchor(root)
                    retrieval = self.retrieval_manifest(root, "fiqa", anchor)
                    payload = json.loads(retrieval.read_text(encoding="utf-8"))
                    if field.startswith("anchor."):
                        payload["anchor"][field.split(".", 1)[1]] = value
                    else:
                        payload[field] = value
                    write_json(retrieval, payload)
                    vectors = write_jsonl(root / "query-vectors.jsonl", [vector("q1")])
                    args = prep.parse_args(
                        [
                            "adapt-vector-cache",
                            "--retrieval-manifest",
                            str(retrieval),
                            "--vector-jsonl",
                            str(vectors),
                            "--anchor-manifest",
                            str(anchor),
                            "--dataset",
                            "fiqa",
                            "--role",
                            "query",
                            "--qrels-sha256",
                            fake_sha("qrels"),
                            "--output",
                            str(root / "out.json"),
                        ]
                    )
                    with self.assertRaisesRegex((prep.ManifestError, builder.PlanError), pattern):
                        prep.build_manifest(args)

    def test_convert_score_cache_strips_numeric_scores_and_binds_prepared_ip_seed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            qrels_sha = fake_sha("nf-qrels")
            query_manifest = self.make_vector_manifest(root, "nfcorpus", "query", ["q1"], qrels_sha)
            doc_ids = [f"d{i:03d}" for i in range(1, 121)]
            doc_manifest = self.make_vector_manifest(root, "nfcorpus", "doc", doc_ids, qrels_sha)
            rows = [
                {
                    "bits": 3,
                    "dataset": "nfcorpus",
                    "quantizer_seed": builder.DEFAULT_TURBOQUANT_SEED,
                    "query_id": "q1",
                    "scoring_surface": "turboquant_ip_prepared",
                    "split": "train",
                    "top_k_limit": 120,
                    "top_k": [
                        {"doc_id": doc_id, "rank": rank, "score": 99.0 - rank, "relevance": 1 if rank == 80 else 0}
                        for rank, doc_id in enumerate(doc_ids, start=1)
                    ],
                }
            ]
            scores = write_jsonl(root / "scores.jsonl", rows)
            args = prep.parse_args(
                [
                    "convert-score-cache",
                    "--per-query-jsonl",
                    str(scores),
                    "--query-vector-manifest",
                    str(query_manifest),
                    "--doc-vector-manifest",
                    str(doc_manifest),
                    "--anchor-manifest",
                    str(root / "anchor.json"),
                    "--dataset",
                    "nfcorpus",
                    "--bits",
                    "3",
                    "--qrels-sha256",
                    qrels_sha,
                    "--output",
                    str(root / "out.json"),
                ]
            )
            payload = prep.build_manifest(args)

        self.assertEqual(payload["schema"], builder.SCORE_CACHE_SCHEMA)
        self.assertEqual(payload["turboquant"], {"score_mode": "prepared_ip", "bits": 3, "seed": builder.DEFAULT_TURBOQUANT_SEED})
        self.assertEqual(payload["top_k"], 120)
        self.assertEqual(payload["rows"][0]["docs"][79], {"doc_id": "d080", "rank": 80, "gain": 1})
        self.assertNotIn('"score"', json.dumps(payload, sort_keys=True))
        builder.require_output_has_no_vectors_or_numeric_scores(payload)

    def test_convert_score_cache_rejects_non_top120_and_supports_q5_seed_binding(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            qrels_sha = fake_sha("scifact-qrels")
            query_manifest = self.make_vector_manifest(root, "scifact", "query", ["q1"], qrels_sha)
            doc_ids = [f"d{i:03d}" for i in range(1, 121)]
            doc_manifest = self.make_vector_manifest(root, "scifact", "doc", doc_ids, qrels_sha)
            scores = write_jsonl(
                root / "scores.jsonl",
                [
                    {
                        "bits": 5,
                        "dataset": "scifact",
                        "qid": "q1",
                        "quantizer_seed": 777,
                        "scoring_surface": "turboquant_ip_prepared",
                        "split": "train",
                        "top_k_limit": 120,
                        "top_k": [{"doc_id": doc_id, "rank": rank, "score": 12.0 - rank, "relevance": 1 if rank == 1 else 0} for rank, doc_id in enumerate(doc_ids, start=1)],
                    }
                ],
            )
            q5 = prep.parse_args(
                [
                    "convert-score-cache",
                    "--per-query-jsonl",
                    str(scores),
                    "--query-vector-manifest",
                    str(query_manifest),
                    "--doc-vector-manifest",
                    str(doc_manifest),
                    "--anchor-manifest",
                    str(root / "anchor.json"),
                    "--dataset",
                    "scifact",
                    "--bits",
                    "5",
                    "--turboquant-seed",
                    "777",
                    "--qrels-sha256",
                    qrels_sha,
                    "--output",
                    str(root / "out.json"),
                ]
            )
            self.assertEqual(prep.build_manifest(q5)["turboquant"], {"score_mode": "prepared_ip", "bits": 5, "seed": 777})
            q5.top_k = 100
            with self.assertRaisesRegex(prep.ManifestError, "top_k must be exactly 120"):
                prep.build_manifest(q5)

    def test_convert_score_cache_rejects_stale_or_reranked_source_metadata(self) -> None:
        cases = [
            ("dataset", "fiqa", "source dataset mismatch"),
            ("split", "test", "split must be train"),
            ("bits", 3, "source bits mismatch"),
            ("quantizer_seed", builder.DEFAULT_TURBOQUANT_SEED + 1, "source quantizer seed mismatch"),
            ("scoring_surface", "turboquant_ip_prepared_overfetch_rerank", "scoring_surface must be turboquant_ip_prepared"),
            ("rerank_overfetch", 200, "rerank source rows are not allowed"),
            ("top_k_limit", 100, "source per-query top_k must be 120"),
        ]
        for field, value, pattern in cases:
            with self.subTest(field=field):
                with tempfile.TemporaryDirectory() as tmp:
                    root = Path(tmp)
                    qrels_sha = fake_sha("scifact-qrels")
                    query_manifest = self.make_vector_manifest(root, "scifact", "query", ["q1"], qrels_sha)
                    doc_ids = [f"d{i:03d}" for i in range(1, 121)]
                    doc_manifest = self.make_vector_manifest(root, "scifact", "doc", doc_ids, qrels_sha)
                    row = {
                        "bits": 5,
                        "dataset": "scifact",
                        "query_id": "q1",
                        "quantizer_seed": builder.DEFAULT_TURBOQUANT_SEED,
                        "scoring_surface": "turboquant_ip_prepared",
                        "split": "train",
                        "top_k_limit": 120,
                        "top_k": [
                            {"doc_id": doc_id, "rank": rank, "score": 10.0 - rank, "relevance": 1 if rank == 1 else 0}
                            for rank, doc_id in enumerate(doc_ids, start=1)
                        ],
                    }
                    row[field] = value
                    scores = write_jsonl(root / "scores.jsonl", [row])
                    args = prep.parse_args(
                        [
                            "convert-score-cache",
                            "--per-query-jsonl",
                            str(scores),
                            "--query-vector-manifest",
                            str(query_manifest),
                            "--doc-vector-manifest",
                            str(doc_manifest),
                            "--anchor-manifest",
                            str(root / "anchor.json"),
                            "--dataset",
                            "scifact",
                            "--bits",
                            "5",
                            "--qrels-sha256",
                            qrels_sha,
                            "--output",
                            str(root / "out.json"),
                        ]
                    )
                    with self.assertRaisesRegex((prep.ManifestError, builder.PlanError), pattern):
                        prep.build_manifest(args)

    def test_convert_score_cache_rejects_missing_required_source_metadata(self) -> None:
        required_fields = ["dataset", "split", "bits", "quantizer_seed", "scoring_surface", "top_k_limit"]
        for missing in required_fields:
            with self.subTest(missing=missing):
                with tempfile.TemporaryDirectory() as tmp:
                    root = Path(tmp)
                    qrels_sha = fake_sha("scifact-qrels")
                    query_manifest = self.make_vector_manifest(root, "scifact", "query", ["q1"], qrels_sha)
                    doc_ids = [f"d{i:03d}" for i in range(1, 121)]
                    doc_manifest = self.make_vector_manifest(root, "scifact", "doc", doc_ids, qrels_sha)
                    row = {
                        "bits": 5,
                        "dataset": "scifact",
                        "query_id": "q1",
                        "quantizer_seed": builder.DEFAULT_TURBOQUANT_SEED,
                        "scoring_surface": "turboquant_ip_prepared",
                        "split": "train",
                        "top_k_limit": 120,
                        "top_k": [
                            {"doc_id": doc_id, "rank": rank, "score": 10.0 - rank, "relevance": 1 if rank == 1 else 0}
                            for rank, doc_id in enumerate(doc_ids, start=1)
                        ],
                    }
                    del row[missing]
                    scores = write_jsonl(root / "scores.jsonl", [row])
                    args = prep.parse_args(
                        [
                            "convert-score-cache",
                            "--per-query-jsonl",
                            str(scores),
                            "--query-vector-manifest",
                            str(query_manifest),
                            "--doc-vector-manifest",
                            str(doc_manifest),
                            "--anchor-manifest",
                            str(root / "anchor.json"),
                            "--dataset",
                            "scifact",
                            "--bits",
                            "5",
                            "--qrels-sha256",
                            qrels_sha,
                            "--output",
                            str(root / "out.json"),
                        ]
                    )
                    with self.assertRaisesRegex(prep.ManifestError, "missing required source field"):
                        prep.build_manifest(args)

    def test_convert_score_cache_rejects_legacy_docs_source_shape(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            qrels_sha = fake_sha("scifact-qrels")
            query_manifest = self.make_vector_manifest(root, "scifact", "query", ["q1"], qrels_sha)
            doc_ids = [f"d{i:03d}" for i in range(1, 121)]
            doc_manifest = self.make_vector_manifest(root, "scifact", "doc", doc_ids, qrels_sha)
            scores = write_jsonl(
                root / "scores.jsonl",
                [
                    {
                        "bits": 5,
                        "dataset": "scifact",
                        "query_id": "q1",
                        "quantizer_seed": builder.DEFAULT_TURBOQUANT_SEED,
                        "scoring_surface": "turboquant_ip_prepared",
                        "split": "train",
                        "top_k_limit": 120,
                        "docs": [{"doc_id": doc_id, "rank": rank, "gain": 1 if rank == 1 else 0} for rank, doc_id in enumerate(doc_ids, start=1)],
                    }
                ],
            )
            args = prep.parse_args(
                [
                    "convert-score-cache",
                    "--per-query-jsonl",
                    str(scores),
                    "--query-vector-manifest",
                    str(query_manifest),
                    "--doc-vector-manifest",
                    str(doc_manifest),
                    "--anchor-manifest",
                    str(root / "anchor.json"),
                    "--dataset",
                    "scifact",
                    "--bits",
                    "5",
                    "--qrels-sha256",
                    qrels_sha,
                    "--output",
                    str(root / "out.json"),
                ]
            )
            with self.assertRaisesRegex(prep.ManifestError, "direct top_k candidate shape"):
                prep.build_manifest(args)

    def test_adapter_outputs_feed_stage3_plan_builder_with_canonical_source_maps(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            exclusions = [
                write_json(
                    root / "exclusions" / "dev4.json",
                    {
                        "schema": builder.EXCLUSION_SCHEMA,
                        "name": "dev4",
                        "qids_by_dataset": {dataset: [f"{dataset}-dev-q"] for dataset in builder.ALLOWED_DATASETS},
                        "source_sha256": fake_sha("dev4"),
                    },
                ),
                write_json(
                    root / "exclusions" / "reserve4.json",
                    {
                        "schema": builder.EXCLUSION_SCHEMA,
                        "name": "reserve4",
                        "qids_by_dataset": {dataset: [f"{dataset}-reserve-q"] for dataset in builder.ALLOWED_DATASETS},
                        "source_sha256": fake_sha("reserve4"),
                    },
                ),
                write_json(
                    root / "exclusions" / "official-test.json",
                    {
                        "schema": builder.EXCLUSION_SCHEMA,
                        "name": "official-test",
                        "qids_by_dataset": {dataset: [f"{dataset}-official-q"] for dataset in builder.ALLOWED_DATASETS},
                        "source_sha256": fake_sha("official"),
                    },
                ),
            ]
            vector_manifests = []
            score_manifests = []
            for dataset in builder.ALLOWED_DATASETS:
                qrels_sha = fake_sha(f"{dataset}-qrels")
                query_ids = [f"{dataset}-q1"]
                doc_ids = [f"{dataset}-d{i:03d}" for i in range(1, 126)]
                vector_manifests.append(self.make_vector_manifest(root / dataset, dataset, "query", query_ids, qrels_sha))
                vector_manifests.append(self.make_vector_manifest(root / dataset, dataset, "doc", doc_ids, qrels_sha))
                score_manifests.append(self.make_score_manifest(root / dataset, dataset, 3, query_ids, doc_ids, qrels_sha))
                score_manifests.append(self.make_score_manifest(root / dataset, dataset, 5, query_ids, doc_ids, qrels_sha))
            argv = ["--anchor-manifest", str(root / "fiqa" / "anchor.json"), "--output-plan", str(root / "plan.json")]
            for path in exclusions:
                argv.extend(["--exclusion-qids", str(path)])
            for path in vector_manifests:
                argv.extend(["--vector-cache", str(path)])
            for path in score_manifests:
                argv.extend(["--score-cache", str(path)])
            plan = builder.build_plan(builder.parse_args(argv))
        self.assertEqual(plan["schema"], builder.SCHEMA)
        self.assertEqual(set(plan["provenance"]["dataset_source_provenance_by_dataset"]), set(builder.ALLOWED_DATASETS))
        builder.require_output_has_no_vectors_or_numeric_scores(plan)


if __name__ == "__main__":
    unittest.main()
