#!/usr/bin/env python3
"""Tests for the q3 R6 frontier-retention train/probe artifacts."""

from __future__ import annotations

import copy
import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import build_q3_r6_frontier_retention_dataset as builder  # noqa: E402
import q3_r6_probe_preflight as probe_gate  # noqa: E402
from build_q3_r3_incumbent_dataset import canonical_stats, eligible_topk_pairs  # noqa: E402
from build_q3_r4_broad_gap_dataset import normalize_candidate_text  # noqa: E402


R6_ROOT = REPO_ROOT / "runs/eos-d384-q3-boundary-goal-v1-20260902T084144Z/headroom-q3r6-frontier-retain-alltrain-b0-rw050"
TRAIN = R6_ROOT / "data/beir1155.q3r6-frontier-retention.train.jsonl"
MANIFEST = R6_ROOT / "data/beir1155.q3r6-frontier-retention.manifest.json"
COVERAGE = R6_ROOT / "q3r6-frontier-retention.coverage.json"
PROBE_MANIFEST = R6_ROOT / "probe/q3r6-frontier-retention.probe-manifest.json"
EXPECTED_SURFACE_IDS = {
    "frontier.fiqa",
    "frontier.nfcorpus",
    "frontier.scifact",
    "top10-retention.fiqa",
    "top10-retention.nfcorpus",
    "top10-retention.scifact",
    "nf-boundary80-100.nfcorpus",
}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_rows(path: Path = TRAIN) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def read_qrels(path: Path) -> dict[str, list[tuple[str, int]]]:
    out: dict[str, list[tuple[str, int]]] = {}
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            out.setdefault(str(row["query-id"]), []).append((str(row["corpus-id"]), int(row["score"])))
    return out


def write_gate_fixture_outputs(probe: dict, outputs_root: Path) -> dict:
    mutated = copy.deepcopy(probe)
    mutated["future_paths"]["outputs_root"] = str(outputs_root)
    mutated["candidate_package"]["sibling_rollup_sha256"] = "fixture-bound"
    mutated["candidate_effective_base_leakage_audit"] = {
        "base_hard_soft_recovery_work_units": 0,
        "aux_only": True,
    }
    expected_outputs: list[str] = []
    for command in mutated["commands"]:
        surface_id = command["surface_id"]
        label = command["label"]
        surface = mutated["surfaces"][surface_id]
        stem = outputs_root / surface_id / f"{label}.{surface_id}.q3.r6-probe"
        command["outputs"] = {
            "metrics_json": f"{stem}.json",
            "metrics_tsv": f"{stem}.tsv",
            "per_query_jsonl": f"{stem}.per-query.jsonl",
        }
        command["argv"] = [
            mutated["binary"]["path"],
            "eval-retrieval-turboquant",
            "--bits",
            "3",
            "--quantizer-seed",
            str(builder.r4.Q3_SEED),
            "--top-k",
            "100",
            "--per-query-top-k",
            "100",
            "--dataset",
            command["dataset"],
            "--qrels",
            surface["qrels"]["path"],
            "--metrics-json",
            command["outputs"]["metrics_json"],
            "--metrics-tsv",
            command["outputs"]["metrics_tsv"],
            "--per-query-jsonl",
            command["outputs"]["per_query_jsonl"],
            command["package"],
            surface["dataset_dir"],
        ]
        expected_outputs.extend(command["outputs"].values())
    mutated["expected_outputs"] = sorted(expected_outputs)
    mutated["command_argv_sha256"] = builder.sha256_json([command["argv"] for command in mutated["commands"]])
    for surface_id, surface in mutated["surfaces"].items():
        qrels = read_qrels(REPO_ROOT / surface["qrels"]["path"])
        for label in ("anchor", "candidate"):
            outputs = next(cmd["outputs"] for cmd in mutated["commands"] if cmd["surface_id"] == surface_id and cmd["label"] == label)
            Path(outputs["metrics_json"]).parent.mkdir(parents=True, exist_ok=True)
            ndcg = 1.0
            recall = 1.0
            metrics = {
                "schema": probe_gate.METRICS_SCHEMA,
                "dataset": surface["dataset"],
                "artifact": next(cmd["package"] for cmd in mutated["commands"] if cmd["surface_id"] == surface_id and cmd["label"] == label),
                "config": {"bits": [3], "quantizer_seed": builder.r4.Q3_SEED, "top_k": 100},
                "inputs": {
                    "qrels_path": surface["qrels"]["path"],
                    "qrels_sha256": surface["qrels"]["sha256"],
                    "queries": surface["qrels"]["qid_count"],
                    "relevant_pairs": surface["qrels"]["qrel_count"],
                },
                "rows": [{"method": probe_gate.Q3_METHOD, "bits": 3, "quality": {"ndcg_at_10": ndcg, "recall_at_100": recall}}],
            }
            Path(outputs["metrics_json"]).write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            Path(outputs["metrics_tsv"]).write_text(
                "dataset\tartifact\tmethod\tbits\tndcg_at_10\trecall_at_100\n"
                f"{surface['dataset']}\t{metrics['artifact']}\t{probe_gate.Q3_METHOD}\t3\t{ndcg:.6f}\t{recall:.6f}\n",
                encoding="utf-8",
            )
            lines = []
            for qid, docs in sorted(qrels.items()):
                relevant = {doc_id: gain for doc_id, gain in docs}
                top: list[dict] = []
                if surface["bucket"] == builder.NF_BOUNDARY_BUCKET:
                    relevant_ranks = list(range(80, 80 + len(docs)))
                else:
                    relevant_ranks = list(range(1, 1 + len(docs)))
                relevant_by_rank = dict(zip(relevant_ranks, docs))
                for rank in range(1, 101):
                    if rank in relevant_by_rank:
                        doc_id, gain = relevant_by_rank[rank]
                        score = 1000.0 - rank
                        relevance = gain
                    else:
                        doc_id = f"__nonrel_{surface_id}_{qid}_{rank}"
                        score = -1000.0 - rank
                        relevance = 0
                    top.append({"rank": rank, "doc_id": doc_id, "score": score, "relevance": relevance})
                lines.append(
                    json.dumps(
                        {
                            "schema": probe_gate.PER_QUERY_SCHEMA,
                            "dataset": surface["dataset"],
                            "artifact": metrics["artifact"],
                            "query_id": qid,
                            "method": probe_gate.Q3_METHOD,
                            "bits": 3,
                            "quantizer_seed": builder.r4.Q3_SEED,
                            "scoring_surface": probe_gate.Q3_SURFACE,
                            "relevant_count": len(relevant),
                            "top_k": top,
                        },
                        sort_keys=True,
                    )
                )
            Path(outputs["per_query_jsonl"]).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return mutated


def require_real_artifacts(testcase: unittest.TestCase) -> None:
    missing = [str(path) for path in (TRAIN, MANIFEST, COVERAGE, PROBE_MANIFEST) if not path.is_file()]
    if missing:
        testcase.skipTest(f"real R6 artifacts not present: {missing}")


class BuildQ3R6FrontierRetentionDatasetTest(unittest.TestCase):
    def test_real_artifact_counts_hashes_and_legal_scope(self) -> None:
        require_real_artifacts(self)
        manifest = read_json(MANIFEST)
        coverage = read_json(COVERAGE)
        probe = read_json(PROBE_MANIFEST)
        rows = read_rows()
        self.assertEqual(manifest["schema"], builder.SCHEMA)
        self.assertEqual(coverage["schema"], f"{builder.SCHEMA}.coverage.v1")
        self.assertEqual(probe["schema"], builder.PROBE_SCHEMA)
        self.assertEqual(builder.sha256_file(TRAIN), manifest["sha256"]["train_score_spectrum"])
        self.assertEqual(len(rows), 1155)
        self.assertEqual(manifest["counts"]["workload"], builder.EXPECTED_WORKLOAD)
        self.assertEqual(manifest["counts"]["bucket_counts"], builder.EXPECTED_BUCKETS)
        self.assertEqual(manifest["counts"]["domain_bucket_counts"], builder.EXPECTED_DOMAIN_BUCKETS)
        self.assertEqual(manifest["counts"]["alias_audit"], {"raw_candidates": 11183, "canonical_candidates": 11167, "alias_drops": 16})
        self.assertFalse(manifest["release_train_allowed"])
        self.assertFalse(manifest["commercial_use_allowed"])
        self.assertFalse(manifest["free_open_release_allowed"])
        self.assertFalse(manifest["quality_claim"])
        self.assertTrue(manifest["train_allowed_for_research"])
        self.assertEqual(coverage["validation"]["effective_weight_audit"]["base_hard_soft_recovery_expected_zero_rows"], 1155)
        self.assertEqual(coverage["validation"]["effective_weight_audit"]["runtime_base0_support"], "pending new binary verification")

    def test_real_artifact_bucket_pair_contract(self) -> None:
        require_real_artifacts(self)
        rows = read_rows()
        by_bucket = {
            bucket: [row for row in rows if row["selection_bucket"] == bucket]
            for bucket in builder.EXPECTED_BUCKETS
        }
        self.assertEqual(sum(eligible_topk_pairs(row, False, "q3") for row in by_bucket[builder.FRONTIER_BUCKET]), 4825)
        self.assertEqual(sum(eligible_topk_pairs(row, False, "q3") for row in by_bucket[builder.TOP10_BUCKET]), 5969)
        self.assertEqual(sum(eligible_topk_pairs(row, True, "q3") for row in by_bucket[builder.NF_BOUNDARY_BUCKET]), 5120)
        frontier_pos_pos = 0
        frontier_pos_nonrel = 0
        for row in by_bucket[builder.FRONTIER_BUCKET]:
            for high in row["qrel_gains"]:
                for low in row["qrel_gains"]:
                    if high <= low:
                        continue
                    if low > 0:
                        frontier_pos_pos += 1
                    else:
                        frontier_pos_nonrel += 1
        self.assertEqual(frontier_pos_pos, 1816)
        self.assertEqual(frontier_pos_nonrel, 3009)

    def test_real_boundary_rows_are_canonical_unique_nine_candidate_recall_rows(self) -> None:
        require_real_artifacts(self)
        rows = [row for row in read_rows() if row["selection_bucket"] == builder.NF_BOUNDARY_BUCKET]
        self.assertEqual(len(rows), 640)
        for row in rows:
            self.assertEqual(len(row["candidate_doc_ids"]), 9, row["row_id"])
            self.assertEqual(canonical_stats(row, row["row_id"])["canonical_candidates"], 9, row["row_id"])
            self.assertEqual(row["positive_count"], 1, row["row_id"])
            self.assertEqual(row["q3_negative_count"], 8, row["row_id"])
            self.assertEqual(sum(1 for value in row["hard_negative_eligible"] if value), 8, row["row_id"])
            self.assertEqual(eligible_topk_pairs(row, False, "q3"), 0, row["row_id"])
            self.assertEqual(eligible_topk_pairs(row, True, "q3"), 8, row["row_id"])
            aliases = [normalize_candidate_text(text) for text in row["candidate_texts"]]
            self.assertEqual(len(aliases), len(set(aliases)), row["row_id"])
            self.assertEqual(float(row["base_loss_weight"]), 0.0)
            self.assertEqual(float(row["turboquant_topk_loss_weight"]), 0.0)
            self.assertEqual(float(row["turboquant_topk_recall_loss_weight"]), 1.0)

    def test_probe_manifest_contract(self) -> None:
        require_real_artifacts(self)
        probe = read_json(PROBE_MANIFEST)
        self.assertEqual(probe["arm_id"], builder.ARM_ID)
        self.assertEqual(set(probe["surfaces"]), EXPECTED_SURFACE_IDS)
        self.assertEqual(probe["surface_count"], 7)
        self.assertEqual(probe["command_count"], 14)
        self.assertEqual(probe["expected_output_count"], 42)
        self.assertTrue(probe["all_expected_outputs_absent"])
        self.assertTrue(probe["all_expected_outputs_contained"])
        self.assertEqual(probe["binary"]["sha256"], builder.R6_BINARY_SHA256)
        self.assertEqual(probe["q3_settings"]["bits"], 3)
        self.assertEqual(probe["q3_settings"]["quantizer_seed"], builder.r4.Q3_SEED)
        self.assertEqual(probe["train_contract"], builder.EXPECTED_WORKLOAD)
        self.assertEqual(sum(surface["row_count"] for surface in probe["surfaces"].values()), 1155)
        self.assertEqual(probe["surfaces"]["frontier.fiqa"]["row_count"], 33)
        self.assertEqual(probe["surfaces"]["frontier.nfcorpus"]["row_count"], 64)
        self.assertEqual(probe["surfaces"]["frontier.scifact"]["row_count"], 13)
        self.assertEqual(probe["surfaces"]["top10-retention.fiqa"]["row_count"], 108)
        self.assertEqual(probe["surfaces"]["top10-retention.nfcorpus"]["row_count"], 234)
        self.assertEqual(probe["surfaces"]["top10-retention.scifact"]["row_count"], 63)
        self.assertEqual(probe["surfaces"]["nf-boundary80-100.nfcorpus"]["row_count"], 640)
        self.assertEqual(probe["surfaces"]["nf-boundary80-100.nfcorpus"]["qid_count"], 211)
        self.assertEqual(probe["surfaces"]["nf-boundary80-100.nfcorpus"]["qrel_count"], 640)
        expected_commands = set()
        expected_outputs = []
        for surface_id, surface in probe["surfaces"].items():
            qrels_path = REPO_ROOT / surface["qrels"]["path"]
            self.assertTrue(qrels_path.is_file(), qrels_path)
            self.assertEqual(builder.sha256_file(qrels_path), surface["qrels"]["sha256"])
            self.assertIn(f"/probe/qrels/{surface['dataset']}/", f"/{surface['qrels']['path']}")
            self.assertEqual(qrels_path.name, f"{surface_id}.qrels.tsv")
            qrels = read_qrels(qrels_path)
            self.assertEqual(len(qrels), surface["qrels"]["qid_count"])
            self.assertEqual(sum(len(docs) for docs in qrels.values()), surface["qrels"]["qrel_count"])
            expected_commands.add((surface_id, "anchor"))
            expected_commands.add((surface_id, "candidate"))
        self.assertEqual({(cmd["surface_id"], cmd["label"]) for cmd in probe["commands"]}, expected_commands)
        for cmd in probe["commands"]:
            surface = probe["surfaces"][cmd["surface_id"]]
            outputs = cmd["outputs"]
            expected_outputs.extend(outputs.values())
            self.assertEqual(
                cmd["argv"],
                [
                    probe["binary"]["path"],
                    "eval-retrieval-turboquant",
                    "--bits",
                    "3",
                    "--quantizer-seed",
                    str(builder.r4.Q3_SEED),
                    "--top-k",
                    "100",
                    "--per-query-top-k",
                    "100",
                    "--dataset",
                    cmd["dataset"],
                    "--qrels",
                    surface["qrels"]["path"],
                    "--metrics-json",
                    outputs["metrics_json"],
                    "--metrics-tsv",
                    outputs["metrics_tsv"],
                    "--per-query-jsonl",
                    outputs["per_query_jsonl"],
                    cmd["package"],
                    surface["dataset_dir"],
                ],
            )
            self.assertNotIn("eval-retrieval", cmd["argv"])
            self.assertNotIn("--turboquant-bits", cmd["argv"])
        self.assertEqual(sorted(expected_outputs), probe["expected_outputs"])
        probe_gate.validate_manifest(PROBE_MANIFEST, require_outputs_absent=True, allow_pending_runtime_bindings=True)

    def test_probe_preflight_rejects_review_p1_and_output_contract_mutations(self) -> None:
        require_real_artifacts(self)
        probe = read_json(PROBE_MANIFEST)
        mutations = []
        bad_cli = copy.deepcopy(probe)
        bad_cli["commands"][0]["argv"][1] = "eval-retrieval"
        mutations.append((bad_cli, "argv shape"))
        bad_domain = copy.deepcopy(probe)
        bad_domain["commands"][0]["dataset"] = "nfcorpus"
        mutations.append((bad_domain, "bucket/dataset"))
        bad_qrels_namespace = copy.deepcopy(probe)
        bad_qrels_namespace["surfaces"]["frontier.fiqa"]["qrels"]["path"] = probe["surfaces"]["frontier.nfcorpus"]["qrels"]["path"]
        bad_qrels_namespace["surfaces"]["frontier.fiqa"]["qrels"]["sha256"] = probe["surfaces"]["frontier.nfcorpus"]["qrels"]["sha256"]
        for cmd in bad_qrels_namespace["commands"]:
            if cmd["surface_id"] == "frontier.fiqa":
                qrels_index = cmd["argv"].index("--qrels") + 1
                cmd["argv"][qrels_index] = bad_qrels_namespace["surfaces"]["frontier.fiqa"]["qrels"]["path"]
        mutations.append((bad_qrels_namespace, "qrels path"))
        bad_escape = copy.deepcopy(probe)
        bad_escape["expected_outputs"][0] = "../escape.json"
        mutations.append((bad_escape, "escapes"))
        bad_sibling_prefix = copy.deepcopy(probe)
        evil_output = str(Path(probe["future_paths"]["outputs_root"]).parent / "future-outputs-evil" / "escape.json")
        bad_sibling_prefix["expected_outputs"][0] = evil_output
        mutations.append((bad_sibling_prefix, "escapes"))
        bad_output_equality = copy.deepcopy(probe)
        alien_output = str(Path(probe["future_paths"]["outputs_root"]) / "frontier.fiqa" / "alien-contained-output.json")
        bad_output_equality["commands"][0]["outputs"]["metrics_json"] = alien_output
        metrics_index = bad_output_equality["commands"][0]["argv"].index("--metrics-json") + 1
        bad_output_equality["commands"][0]["argv"][metrics_index] = alien_output
        mutations.append((bad_output_equality, "command outputs"))
        bad_qrels_sha = copy.deepcopy(probe)
        bad_qrels_sha["surfaces"]["frontier.fiqa"]["qrels"]["sha256"] = "0" * 64
        mutations.append((bad_qrels_sha, "qrels sha"))
        for payload, needle in mutations:
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "probe.json"
                path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
                with self.assertRaises(probe_gate.ProbeContractError) as ctx:
                    probe_gate.validate_manifest(path, require_outputs_absent=True, allow_pending_runtime_bindings=True)
                self.assertIn(needle, str(ctx.exception))

    def test_probe_gate_fixture_outputs_and_numeric_crosscheck(self) -> None:
        require_real_artifacts(self)
        probe = read_json(PROBE_MANIFEST)
        with tempfile.TemporaryDirectory() as tmp:
            outputs_root = Path(tmp) / "future-outputs"
            fixture = write_gate_fixture_outputs(probe, outputs_root)
            path = Path(tmp) / "probe.json"
            path.write_text(json.dumps(fixture, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            summary = probe_gate.validate_gate(path)
            self.assertTrue(summary["passed"])
            self.assertEqual(summary["output_count"], 42)
            self.assertEqual(summary["relevant_top10_exits"], 0)
            self.assertEqual(summary["substitution_failures"], 0)
            self.assertGreaterEqual(summary["nf_boundary_entries"], summary["nf_boundary_exits"])
            bad = copy.deepcopy(fixture)
            bad_cmd = next(cmd for cmd in bad["commands"] if cmd["surface_id"] == "frontier.fiqa" and cmd["label"] == "candidate")
            metrics_path = Path(bad_cmd["outputs"]["metrics_json"])
            metrics = read_json(metrics_path)
            metrics["rows"][0]["quality"]["ndcg_at_10"] = 0.5
            metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            Path(bad_cmd["outputs"]["metrics_tsv"]).write_text(
                "dataset\tartifact\tmethod\tbits\tndcg_at_10\trecall_at_100\n"
                f"fiqa\t{bad_cmd['package']}\t{probe_gate.Q3_METHOD}\t3\t0.500000\t1.000000\n",
                encoding="utf-8",
            )
            with self.assertRaises(probe_gate.ProbeContractError) as ctx:
                probe_gate.validate_gate(path)
            self.assertIn("nDCG delta negative", str(ctx.exception))

    def test_probe_gate_rejects_artifact_and_relevance_truth_mutations(self) -> None:
        require_real_artifacts(self)
        probe = read_json(PROBE_MANIFEST)
        with tempfile.TemporaryDirectory() as tmp:
            outputs_root = Path(tmp) / "future-outputs"
            fixture = write_gate_fixture_outputs(probe, outputs_root)
            path = Path(tmp) / "probe.json"
            path.write_text(json.dumps(fixture, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            probe_gate.validate_gate(path)

            candidate_cmd = next(cmd for cmd in fixture["commands"] if cmd["surface_id"] == "frontier.fiqa" and cmd["label"] == "candidate")
            anchor_cmd = next(cmd for cmd in fixture["commands"] if cmd["surface_id"] == "frontier.fiqa" and cmd["label"] == "anchor")

            missing_artifact = read_json(Path(candidate_cmd["outputs"]["metrics_json"]))
            missing_artifact.pop("artifact")
            Path(candidate_cmd["outputs"]["metrics_json"]).write_text(json.dumps(missing_artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            with self.assertRaises(probe_gate.ProbeContractError) as ctx:
                probe_gate.validate_gate(path)
            self.assertIn("artifact mismatch", str(ctx.exception))

            fixture = write_gate_fixture_outputs(probe, outputs_root)
            path.write_text(json.dumps(fixture, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            candidate_cmd = next(cmd for cmd in fixture["commands"] if cmd["surface_id"] == "frontier.fiqa" and cmd["label"] == "candidate")
            anchor_cmd = next(cmd for cmd in fixture["commands"] if cmd["surface_id"] == "frontier.fiqa" and cmd["label"] == "anchor")
            Path(candidate_cmd["outputs"]["metrics_json"]).write_text(Path(anchor_cmd["outputs"]["metrics_json"]).read_text(encoding="utf-8"), encoding="utf-8")
            with self.assertRaises(probe_gate.ProbeContractError) as ctx:
                probe_gate.validate_gate(path)
            self.assertIn("artifact mismatch", str(ctx.exception))

            fixture = write_gate_fixture_outputs(probe, outputs_root)
            path.write_text(json.dumps(fixture, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            candidate_cmd = next(cmd for cmd in fixture["commands"] if cmd["surface_id"] == "frontier.fiqa" and cmd["label"] == "candidate")
            metrics = read_json(Path(candidate_cmd["outputs"]["metrics_json"]))
            metrics["artifact"] = "runs/wrong/d384-pre.mll"
            Path(candidate_cmd["outputs"]["metrics_json"]).write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            with self.assertRaises(probe_gate.ProbeContractError) as ctx:
                probe_gate.validate_gate(path)
            self.assertIn("artifact mismatch", str(ctx.exception))

            fixture = write_gate_fixture_outputs(probe, outputs_root)
            path.write_text(json.dumps(fixture, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            candidate_cmd = next(cmd for cmd in fixture["commands"] if cmd["surface_id"] == "frontier.fiqa" and cmd["label"] == "candidate")
            per_query_path = Path(candidate_cmd["outputs"]["per_query_jsonl"])
            rows = [json.loads(line) for line in per_query_path.read_text(encoding="utf-8").splitlines() if line]
            rows[0]["relevant_count"] += 1
            per_query_path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n", encoding="utf-8")
            with self.assertRaises(probe_gate.ProbeContractError) as ctx:
                probe_gate.validate_gate(path)
            self.assertIn("relevant_count mismatch", str(ctx.exception))

            fixture = write_gate_fixture_outputs(probe, outputs_root)
            path.write_text(json.dumps(fixture, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            candidate_cmd = next(cmd for cmd in fixture["commands"] if cmd["surface_id"] == "frontier.fiqa" and cmd["label"] == "candidate")
            per_query_path = Path(candidate_cmd["outputs"]["per_query_jsonl"])
            rows = [json.loads(line) for line in per_query_path.read_text(encoding="utf-8").splitlines() if line]
            rows[0]["top_k"][0]["relevance"] = 0
            per_query_path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n", encoding="utf-8")
            with self.assertRaises(probe_gate.ProbeContractError) as ctx:
                probe_gate.validate_gate(path)
            self.assertIn("top_k relevance mismatch", str(ctx.exception))

            fixture = write_gate_fixture_outputs(probe, outputs_root)
            path.write_text(json.dumps(fixture, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            candidate_cmd = next(cmd for cmd in fixture["commands"] if cmd["surface_id"] == "frontier.fiqa" and cmd["label"] == "candidate")
            per_query_path = Path(candidate_cmd["outputs"]["per_query_jsonl"])
            rows = [json.loads(line) for line in per_query_path.read_text(encoding="utf-8").splitlines() if line]
            qrels = read_qrels(REPO_ROOT / fixture["surfaces"]["frontier.fiqa"]["qrels"]["path"])
            qid = rows[0]["query_id"]
            first_doc = rows[0]["top_k"][0]["doc_id"]
            rows[0]["top_k"][0] = {"rank": 1, "doc_id": f"__masking_nonrel_{qid}", "score": 9999.0, "relevance": 1}
            for rank, (doc_id, gain) in enumerate(qrels[qid], start=11):
                rows[0]["top_k"][rank - 1] = {"rank": rank, "doc_id": doc_id, "score": 1000.0 - rank, "relevance": gain}
            rows[0]["top_k"][99] = {"rank": 100, "doc_id": first_doc, "score": 1.0, "relevance": qrels[qid][0][1]}
            per_query_path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n", encoding="utf-8")
            with self.assertRaises(probe_gate.ProbeContractError) as ctx:
                probe_gate.validate_gate(path)
            self.assertIn("top_k relevance mismatch", str(ctx.exception))

    def test_mutations_rejected(self) -> None:
        require_real_artifacts(self)
        original = read_rows()
        args = builder.parse_args([])
        builder.default_paths(args)
        q3_rows, _ids = builder.r4.load_q3_rows(args.q3_per_query_jsonl, builder.expected_q3_hashes(args))
        self.assertTrue(q3_rows)
        sources = {dataset: builder.r4.load_domain_sources(builder.repo_path(args.dataset_root), dataset) for dataset in ("fiqa", "nfcorpus", "scifact")}
        excluded = builder.r4.read_excluded_qids(args.exclude_qids_json)
        for mutate, needle in (
            (lambda rows: rows[0].__setitem__("commercial_use_allowed", True), "legal gate"),
            (lambda rows: rows[0].__setitem__("base_loss_weight", 1.0), "base_loss_weight"),
            (lambda rows: rows[0].__setitem__("turboquant_topk_loss_weight", 0.0), "inactive"),
            (lambda rows: rows[-1]["candidate_texts"].__setitem__(2, rows[-1]["candidate_texts"][1]), "workload drift"),
        ):
            rows = copy.deepcopy(original)
            mutate(rows)
            with self.assertRaises(ValueError) as ctx:
                builder.validate_rows(rows, args, excluded, sources)
            self.assertIn(needle, str(ctx.exception))

    def test_clean_temp_regeneration_train_and_probe_qrel_bytes_match(self) -> None:
        require_real_artifacts(self)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "r6"
            builder.main(["--output-root", str(out)])
            self.assertEqual(builder.sha256_file(out / "data/beir1155.q3r6-frontier-retention.train.jsonl"), builder.sha256_file(TRAIN))
            for surface_id in EXPECTED_SURFACE_IDS:
                dataset = surface_id.rsplit(".", 1)[1]
                self.assertEqual(
                    builder.sha256_file(out / "probe" / "qrels" / dataset / f"{surface_id}.qrels.tsv"),
                    builder.sha256_file(R6_ROOT / "probe" / "qrels" / dataset / f"{surface_id}.qrels.tsv"),
                    surface_id,
                )
            regenerated = read_json(out / "probe/q3r6-frontier-retention.probe-manifest.json")
            current = read_json(PROBE_MANIFEST)
            self.assertEqual(set(regenerated["surfaces"]), set(current["surfaces"]))
            for surface_id in EXPECTED_SURFACE_IDS:
                for field in ("surface_id", "bucket", "dataset", "row_count", "qid_count", "qrel_count", "row_ids_sha256", "qids_sha256"):
                    self.assertEqual(regenerated["surfaces"][surface_id][field], current["surfaces"][surface_id][field], f"{surface_id}.{field}")
            self.assertEqual(regenerated["surface_count"], current["surface_count"])
            self.assertEqual(regenerated["command_count"], current["command_count"])
            self.assertEqual(regenerated["expected_output_count"], current["expected_output_count"])


if __name__ == "__main__":
    unittest.main()
