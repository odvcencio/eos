"""Synthetic contract tests for :mod:`aoqt_heldout_gate`.

The fixtures are deliberately local and fake: they contain no qrels results,
vectors, evaluator invocation, or official metric values. Every test exercises
one of the gate's provenance, binding, freshness, or quality fail-closed
contracts.
"""

from __future__ import annotations

import copy
import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import aoqt_heldout_gate as gate  # noqa: E402


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_raw(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def record(path: Path) -> dict[str, object]:
    return {"path": str(path.resolve()), "sha256": gate.sha256_file(path), "bytes": path.stat().st_size}


class Fixture:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.inputs = self.root / "inputs"
        self.outputs = self.root / "retrieval-outputs"
        self.inputs.mkdir(parents=True)
        self.outputs.mkdir()
        self.plan_path = self.root / "gate-plan.json"
        self.frozen_path = self.root / "frozen.json"
        self.attestation_path = self.root / "attestation.json"

        self.binary = self.inputs / "eos-native-evaluator"
        self.source_manifest = self.inputs / "source-manifest.json"
        self.dataset = self.inputs / "heldout-dataset-manifest.json"
        self.approved_workload = self.inputs / "approved-workload.json"
        self.workload = self.inputs / "evaluator-workload.json"
        self.qrels = {domain: self.inputs / f"{domain}.qrels" for domain in gate.DOMAINS}
        self.exclusions = {name: self.inputs / f"{name}.exclusion.json" for name in gate.EXCLUSION_NAMES}

        self.anchor_package = self.inputs / "anchor/d384-anchor.mll"
        self.anchor_manifest = self.inputs / "anchor/d384-anchor.package.json"
        self.anchor_attestation = self.inputs / "anchor/d384-anchor.attestation.json"
        self.candidate_package = self.inputs / "candidate/d384-candidate.mll"
        self.candidate_manifest = self.inputs / "candidate/d384-candidate.package.json"
        self.candidate_attestation = self.inputs / "candidate/d384-candidate.attestation.json"
        self.candidate_sidecar = self.inputs / "candidate/d384-candidate.sidecar.json"

        self.plan: dict = {}

    def _transform(self) -> dict:
        return {
            "id": gate.AOQT_TOPOLOGY["id"],
            "dim": gate.DIMENSION,
            "stages": gate.AOQT_TOPOLOGY["stages"],
            "pairs_per_stage": gate.AOQT_TOPOLOGY["pairs_per_stage"],
            "angle_count": gate.AOQT_TOPOLOGY["angle_count"],
            "pairings_sha256": gate.sha256_bytes(b"fixture-pairings"),
            "angles_sha256": gate.sha256_bytes(b"fixture-angles"),
            "angle_cap": 0.01,
            "max_angle_cap": gate.AOQT_TOPOLOGY["angle_cap_hard"],
        }

    def _manifest_policy(self, *, role: str, package: Path, anchor: dict | None, sidecar_sha: str | None, transform: dict | None) -> dict:
        return {
            "schema": gate.PACKAGE_MANIFEST_SCHEMA,
            "role": role,
            "package_sha256": gate.sha256_file(package),
            "dimension": gate.DIMENSION,
            "embedding_space_id": "fixture-space",
            "post_pool_transform": "none" if role == "anchor" else "aoqt_givens_v1",
            "research_only": role == "candidate",
            "topology": copy.deepcopy(gate.AOQT_TOPOLOGY),
            "transform": copy.deepcopy(transform),
            "anchor": copy.deepcopy(anchor),
            "legal_scope": copy.deepcopy(gate.AOQT_LEGAL_SCOPE),
            "sidecar_sha256": sidecar_sha,
        }

    def _sibling_entries(self, paths: list[Path]) -> list[dict]:
        return sorted(({"name": path.name, **record(path)} for path in paths), key=lambda item: item["name"])

    def _attestation_policy(self, *, role: str, package: Path, manifest: Path, anchor: dict, sidecar: Path | None, sibling_entries: list[dict], transform: dict | None) -> dict:
        sidecar_policy = record(sidecar) if sidecar is not None else None
        return {
            "schema": gate.PACKAGE_ATTESTATION_SCHEMA,
            "role": role,
            "package_sha256": gate.sha256_file(package),
            "manifest_sha256": gate.sha256_file(manifest),
            "dimension": gate.DIMENSION,
            "embedding_space_id": "fixture-space",
            "post_pool_transform": "none" if role == "anchor" else "aoqt_givens_v1",
            "research_only": role == "candidate",
            "topology": copy.deepcopy(gate.AOQT_TOPOLOGY),
            "transform": copy.deepcopy(transform),
            "anchor": copy.deepcopy(anchor),
            "legal_scope": copy.deepcopy(gate.AOQT_LEGAL_SCOPE),
            "sidecar": sidecar_policy,
            "sibling_rollup": {"entries": sibling_entries, "sha256": gate.sha256_json(sibling_entries)},
        }

    def build(self) -> "Fixture":
        self.binary.write_bytes(b"synthetic-native-evaluator-binary")
        write_json(self.source_manifest, {"schema": "fixture.source.v2", "source_id": "synthetic-native"})
        write_json(self.dataset, {"schema": "fixture.dataset-manifest.v1", "dataset_id": "synthetic-heldout", "split": "heldout", "domains": list(gate.DOMAINS)})
        for domain, path in self.qrels.items():
            path.write_text(f"synthetic-{domain}-qrels\n", encoding="utf-8")

        self.anchor_package.parent.mkdir(parents=True, exist_ok=True)
        self.candidate_package.parent.mkdir(parents=True, exist_ok=True)
        self.anchor_package.write_bytes(b"synthetic-anchor-package")
        self.candidate_package.write_bytes(b"synthetic-candidate-package")

        anchor_binding = {"package_sha256": gate.sha256_file(self.anchor_package), "manifest_sha256": "0" * 64}
        anchor_manifest_policy = self._manifest_policy(role="anchor", package=self.anchor_package, anchor=None, sidecar_sha=None, transform=None)
        write_json(self.anchor_manifest, anchor_manifest_policy)
        anchor_binding["manifest_sha256"] = gate.sha256_file(self.anchor_manifest)

        transform = self._transform()
        sidecar_payload = {
            "schema": gate.SIDECAR_SCHEMA,
            "topology": copy.deepcopy(gate.AOQT_TOPOLOGY),
            "anchor": copy.deepcopy(anchor_binding),
            "pairings_sha256": transform["pairings_sha256"],
            "angles_sha256": transform["angles_sha256"],
            "legal_scope": copy.deepcopy(gate.AOQT_LEGAL_SCOPE),
        }
        write_json(self.candidate_sidecar, sidecar_payload)
        candidate_manifest_policy = self._manifest_policy(role="candidate", package=self.candidate_package, anchor=anchor_binding, sidecar_sha=gate.sha256_file(self.candidate_sidecar), transform=transform)
        write_json(self.candidate_manifest, candidate_manifest_policy)

        anchor_entries = self._sibling_entries([self.anchor_package, self.anchor_manifest])
        candidate_entries = self._sibling_entries([self.candidate_package, self.candidate_manifest, self.candidate_sidecar])
        anchor_policy = self._attestation_policy(role="anchor", package=self.anchor_package, manifest=self.anchor_manifest, anchor=anchor_binding, sidecar=None, sibling_entries=anchor_entries, transform=None)
        candidate_policy = self._attestation_policy(role="candidate", package=self.candidate_package, manifest=self.candidate_manifest, anchor=anchor_binding, sidecar=self.candidate_sidecar, sibling_entries=candidate_entries, transform=transform)
        write_json(self.anchor_attestation, anchor_policy)
        write_json(self.candidate_attestation, candidate_policy)

        anchor_package_record = {
            **record(self.anchor_package),
            "manifest": record(self.anchor_manifest),
            "attestation": record(self.anchor_attestation),
            "sibling_rollup_sha256": anchor_policy["sibling_rollup"]["sha256"],
        }
        candidate_package_record = {
            **record(self.candidate_package),
            "manifest": record(self.candidate_manifest),
            "attestation": record(self.candidate_attestation),
            "sibling_rollup_sha256": candidate_policy["sibling_rollup"]["sha256"],
        }

        qrels = {domain: {**record(path), "query_count": 1, "relevant_count": 1} for domain, path in self.qrels.items()}
        qids = {domain: [f"H-{domain}"] for domain in gate.DOMAINS}
        approved_payload = {
            "schema": gate.APPROVED_WORKLOAD_SCHEMA,
            "workload_id": "fixture-approved-heldout-workload",
            "gate_id": "fixture-aoqt-gate",
            "split": "heldout",
            "dimension": gate.DIMENSION,
            "domains": list(gate.DOMAINS),
            "query_ids_by_domain": qids,
            "qid_set_sha256_by_domain": {domain: gate.sha256_json(qids[domain]) for domain in gate.DOMAINS},
            "query_count_by_domain": {domain: 1 for domain in gate.DOMAINS},
            "qrels_by_domain": qrels,
            "nfcorpus_boundary_qids": qids["nfcorpus"],
            "nfcorpus_boundary_qids_sha256": gate.sha256_json(qids["nfcorpus"]),
            "nfcorpus_boundary_rank_window": [80, 120],
            "metric_surfaces": list(gate.SURFACES),
            "cutoffs": {"ndcg_at_10": 10, "recall_at_100": 100},
            "turboquant": {"q3_bits": gate.Q3_BITS, "q5_bits": gate.Q5_BITS, "seed": gate.TURBOQUANT_SEED, "top_k": gate.TOP_K},
            "expected_result_artifact_count": gate.RESULT_OUTPUT_COUNT,
        }
        write_json(self.approved_workload, approved_payload)
        workload_payload = {key: value for key, value in approved_payload.items() if key != "qrels_by_domain"}
        workload_payload["schema"] = gate.WORKLOAD_SCHEMA
        write_json(self.workload, workload_payload)

        for name, path in self.exclusions.items():
            exclusion_payload = {
                "schema": "eos.aoqt.heldout_exclusion.v1",
                "name": name,
                "split": name,
                "qids_by_domain": {domain: [f"E-{name}-{domain}"] for domain in gate.DOMAINS},
                "source_sha256": gate.sha256_json({"fixture": name}),
                "source_sha256_by_file": {f"{name}.source": gate.sha256_json({"source": name})},
            }
            write_json(path, exclusion_payload)

        artifacts = {
            role: {
                domain: {"path": str(self.outputs / f"{role}.{domain}.json"), "receipt": {"path": str(self.outputs / f"{role}.{domain}.receipt.json")}}
                for domain in gate.DOMAINS
            }
            for role in gate.ROLES
        }
        self.plan = {
            "schema": gate.PLAN_SCHEMA,
            "gate_id": "fixture-aoqt-gate",
            "candidate_id": "fixture-candidate",
            "dimension": gate.DIMENSION,
            "thresholds": copy.deepcopy(gate.THRESHOLDS),
            "evaluation": {"executed": False, "official": False},
            "anchor": {"id": "fixture-anchor", "package": anchor_package_record},
            "candidate": {"id": "fixture-candidate", "package": candidate_package_record},
            "source": {
                "manifest": record(self.source_manifest),
                "binary": record(self.binary),
                "dataset": record(self.dataset),
                "cwd": str(self.root),
                "argv": {
                    "anchor": [str(self.binary), gate.NATIVE_EVAL_SUBCOMMAND, "--package", str(self.anchor_package)],
                    "candidate": [str(self.binary), gate.NATIVE_EVAL_SUBCOMMAND, "--package", str(self.candidate_package)],
                },
                "workload": record(self.workload),
                "approved_workload": record(self.approved_workload),
            },
            "approved_workload": record(self.approved_workload),
            "qrels": qrels,
            "exclusions": [{"kind": "qid_only", "name": name, **record(path)} for name, path in self.exclusions.items()],
            "workload_root": str(self.inputs),
            "output_root": str(self.outputs),
            "artifacts": artifacts,
            "attestation_path": str(self.attestation_path),
        }
        write_json(self.plan_path, self.plan)
        return self

    def freeze(self) -> dict:
        return gate.freeze_manifest(self.plan_path, self.frozen_path)

    def artifact_payload(self, role: str, domain: str, *, q3_ndcg: float = 0.5, dense_ndcg: float = 0.7, q5_ndcg: float = 0.6, q3_recall: float = 0.8, nonce: str | None = None) -> dict:
        frozen = json.loads(self.frozen_path.read_text(encoding="utf-8"))
        qid = frozen["workload"]["query_ids_by_domain"][domain][0]
        metrics = {"dense": {"ndcg_at_10": dense_ndcg}, "q3": {"ndcg_at_10": q3_ndcg, "recall_at_100": q3_recall}, "q5": {"ndcg_at_10": q5_ndcg}}
        if nonce is None:
            nonce = f"fixture-{role}-{domain}-nonce01"
        if domain == "nfcorpus":
            boundary = {"schema": "eos.aoqt.nfcorpus_boundary_evidence.v1", "guard": "q3_recall_at_100", "rank_window": [80, 120], "qids": [qid], "qids_sha256": gate.sha256_json([qid])}
        else:
            boundary = {"schema": "eos.aoqt.no_boundary_evidence.v1", "guard": "none", "rank_window": None, "qids": [], "qids_sha256": gate.sha256_json([])}
        rows = [{"query_id": qid, "metrics": metrics}]
        return {
            "schema": gate.ARTIFACT_SCHEMA,
            "gate_id": frozen["gate_id"],
            "artifact_path": str(self.outputs / f"{role}.{domain}.json"),
            "role": role,
            "domain": domain,
            "split": "heldout",
            "package": frozen[role]["package"],
            "source": {"manifest": frozen["source"]["manifest"], "qrels": frozen["qrels"][domain], "binary": frozen["source"]["binary"], "cwd": frozen["source"]["cwd"], "argv": frozen["source"]["argv"][role], "dataset": frozen["source"]["dataset"], "workload": frozen["source"]["workload"], "approved_workload": frozen["source"]["approved_workload"]},
            "receipt_path": str(self.outputs / f"{role}.{domain}.receipt.json"),
            "frozen_manifest_sha256": frozen["manifest_sha256"],
            "receipt_nonce": nonce,
            "observed_workload": {"workload_id": frozen["workload"]["workload_id"], "query_count": 1, "qids_sha256": gate.sha256_json([qid]), "scored_query_count": 1},
            "boundary_evidence": boundary,
            "rows": rows,
            "aggregate": metrics,
        }

    def receipt_payload(self, role: str, domain: str, artifact_path: Path, artifact_sha: str, nonce: str) -> dict:
        frozen = json.loads(self.frozen_path.read_text(encoding="utf-8"))
        argv = frozen["source"]["argv"][role]
        payload = {
            "schema": gate.RECEIPT_SCHEMA,
            "gate_id": frozen["gate_id"],
            "nonce": nonce,
            "role": role,
            "domain": domain,
            "split": "heldout",
            "official": False,
            "native_evaluator": True,
            "wrapper": False,
            "exit_status": 0,
            "executable": frozen["source"]["binary"],
            "argv": argv,
            "argv_sha256": gate.sha256_json(argv),
            "cwd": frozen["source"]["cwd"],
            "dataset_id": domain,
            "dataset": frozen["source"]["dataset"],
            "qrels": frozen["qrels"][domain],
            "workload": frozen["source"]["workload"],
            "approved_workload": frozen["source"]["approved_workload"],
            "package": frozen[role]["package"],
            "output": record(artifact_path),
            "config": {"dimension": gate.DIMENSION, "bits": [gate.Q3_BITS, gate.Q5_BITS], "seed": gate.TURBOQUANT_SEED, "top_k": gate.TOP_K, "package_mode": "sibling", "surfaces": list(gate.SURFACES), "cutoffs": {"ndcg_at_10": 10, "recall_at_100": 100}},
            "frozen_manifest": {"path": str(self.frozen_path), "file_sha256": gate.sha256_file(self.frozen_path), "manifest_sha256": frozen["manifest_sha256"]},
            "artifact_sha256": artifact_sha,
        }
        payload["receipt_digest"] = gate.digest_without(payload, "receipt_digest")
        return payload

    def write_artifacts(self, candidate_q3: float = 0.5003, candidate_dense: float = 0.6998, candidate_q5: float = 0.5995, candidate_recall: float = 0.8, *, nonce_by_role_domain: dict[tuple[str, str], str] | None = None) -> None:
        time.sleep(0.01)
        nonce_by_role_domain = nonce_by_role_domain or {}
        for role in gate.ROLES:
            for domain in gate.DOMAINS:
                nonce = nonce_by_role_domain.get((role, domain), f"fixture-{role}-{domain}-nonce01")
                values = {} if role == "anchor" else {"q3_ndcg": candidate_q3, "dense_ndcg": candidate_dense, "q5_ndcg": candidate_q5, "q3_recall": candidate_recall}
                artifact = self.artifact_payload(role, domain, nonce=nonce, **values)
                artifact_path = self.outputs / f"{role}.{domain}.json"
                write_json(artifact_path, artifact)
                artifact_sha = gate.sha256_file(artifact_path)
                receipt_path = self.outputs / f"{role}.{domain}.receipt.json"
                write_json(receipt_path, self.receipt_payload(role, domain, artifact_path, artifact_sha, nonce))

    def rebind_receipt(self, role: str, domain: str) -> None:
        artifact_path = self.outputs / f"{role}.{domain}.json"
        receipt_path = self.outputs / f"{role}.{domain}.receipt.json"
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        receipt["artifact_sha256"] = gate.sha256_file(artifact_path)
        receipt["output"] = record(artifact_path)
        receipt["receipt_digest"] = gate.digest_without(receipt, "receipt_digest")
        write_json(receipt_path, receipt)


class AOQTHeldoutGateTest(unittest.TestCase):
    def make_fixture(self) -> Fixture:
        return Fixture(Path(tempfile.mkdtemp())).build()

    def test_freeze_and_attest_passes_synthetic_thresholds(self) -> None:
        fixture = self.make_fixture()
        frozen = fixture.freeze()
        self.assertEqual(frozen["schema"], gate.FROZEN_SCHEMA)
        self.assertEqual(len(frozen["outputs"]), gate.EXPECTED_OUTPUT_COUNT)
        fixture.write_artifacts()
        report = gate.attest_manifest(fixture.frozen_path, fixture.attestation_path, expected_frozen_manifest_sha256=frozen["file_sha256"])
        self.assertTrue(report["gate_pass"])
        self.assertAlmostEqual(report["macro"]["q3_ndcg_at_10_delta"], 0.0003, places=12)
        self.assertFalse(report["evaluation_invoked_by_gate"])
        self.assertFalse(report["official_metric_values_read_by_gate"])
        self.assertEqual(len(report["output_bindings"]), gate.RESULT_OUTPUT_COUNT)
        self.assertEqual(report["output_bindings_sha256"], gate.sha256_json(report["output_bindings"]))

    def test_attest_requires_external_frozen_manifest_sha(self) -> None:
        fixture = self.make_fixture()
        fixture.freeze()
        with self.assertRaisesRegex(gate.GateError, "external expected frozen manifest sha256 is required"):
            gate.attest_manifest(fixture.frozen_path, fixture.attestation_path)

    def test_rewrite_and_rehash_frozen_manifest_rejected_by_external_anchor(self) -> None:
        fixture = self.make_fixture()
        frozen = fixture.freeze()
        rewritten = json.loads(fixture.frozen_path.read_text(encoding="utf-8"))
        rewritten["candidate_id"] = "rewritten-candidate"
        rewritten["manifest_sha256"] = gate.digest_without(rewritten, "manifest_sha256")
        write_json(fixture.frozen_path, rewritten)
        with self.assertRaisesRegex(gate.GateError, "external frozen manifest sha256 trust-anchor mismatch"):
            gate.attest_manifest(fixture.frozen_path, fixture.attestation_path, expected_frozen_manifest_sha256=frozen["file_sha256"])

    def test_freeze_rejects_arbitrary_small_approved_workload(self) -> None:
        fixture = self.make_fixture()
        approved = json.loads(fixture.approved_workload.read_text(encoding="utf-8"))
        approved["query_ids_by_domain"]["fiqa"] = []
        write_json(fixture.approved_workload, approved)
        fixture.plan["approved_workload"] = record(fixture.approved_workload)
        fixture.plan["source"]["approved_workload"] = record(fixture.approved_workload)
        write_json(fixture.plan_path, fixture.plan)
        with self.assertRaisesRegex(gate.GateError, "qid-set hash mismatch|must be non-empty"):
            fixture.freeze()

    def test_freeze_rejects_duplicate_json_keys(self) -> None:
        fixture = self.make_fixture()
        raw = fixture.workload.read_text(encoding="utf-8")
        marker = '"schema": "eos.aoqt.heldout_workload.v2"'
        write_raw(fixture.workload, raw.replace(marker, marker + ', "schema": "duplicate"', 1))
        fixture.plan["source"]["workload"] = record(fixture.workload)
        write_json(fixture.plan_path, fixture.plan)
        with self.assertRaisesRegex(gate.GateError, "duplicate JSON key"):
            fixture.freeze()

    def test_freeze_rejects_loose_or_intersecting_exclusion(self) -> None:
        fixture = self.make_fixture()
        fixture.plan["exclusions"][0]["name"] = "global"
        write_json(fixture.plan_path, fixture.plan)
        with self.assertRaisesRegex(gate.GateError, "exact qid-only exclusion identity"):
            fixture.freeze()

        fixture = self.make_fixture()
        path = fixture.exclusions["dev4"]
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["qids_by_domain"]["fiqa"] = ["H-fiqa"]
        write_json(path, payload)
        for item in fixture.plan["exclusions"]:
            if item["name"] == "dev4":
                item.update(record(path))
        write_json(fixture.plan_path, fixture.plan)
        with self.assertRaisesRegex(gate.GateError, "workload/exclusion qid intersection"):
            fixture.freeze()

    def test_freeze_rejects_exclusion_metric_payload(self) -> None:
        fixture = self.make_fixture()
        path = fixture.exclusions["dev4"]
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["gains"] = {"H-fiqa": 1}
        write_json(path, payload)
        for item in fixture.plan["exclusions"]:
            if item["name"] == "dev4":
                item.update(record(path))
        write_json(fixture.plan_path, fixture.plan)
        with self.assertRaisesRegex(gate.GateError, "key set mismatch|forbidden gain/vector/score field"):
            fixture.freeze()

    def test_freeze_rejects_package_attestation_policy_drift(self) -> None:
        fixture = self.make_fixture()
        policy = json.loads(fixture.candidate_attestation.read_text(encoding="utf-8"))
        policy["topology"]["stages"] = 7
        write_json(fixture.candidate_attestation, policy)
        fixture.plan["candidate"]["package"]["attestation"] = record(fixture.candidate_attestation)
        write_json(fixture.plan_path, fixture.plan)
        with self.assertRaisesRegex(gate.GateError, "AOQT topology/legal scope mismatch"):
            fixture.freeze()

    def test_package_pair_rejects_same_package_identity_even_on_different_path(self) -> None:
        fixture = self.make_fixture()
        anchor = gate.normalize_package_record(fixture.plan["anchor"]["package"], fixture.plan_path.parent, "test.anchor", expected_role="anchor")
        candidate = gate.normalize_package_record(fixture.plan["candidate"]["package"], fixture.plan_path.parent, "test.candidate", expected_role="candidate")
        candidate["sha256"] = anchor["sha256"]
        with self.assertRaisesRegex(gate.GateError, "identities must be distinct"):
            gate.validate_package_pair(anchor, candidate)

    def test_freeze_rejects_wrapper_or_argv_token_substitution(self) -> None:
        fixture = self.make_fixture()
        fixture.plan["source"]["argv"]["candidate"] = ["/bin/sh", gate.NATIVE_EVAL_SUBCOMMAND, "-c", str(fixture.binary), str(fixture.candidate_package)]
        write_json(fixture.plan_path, fixture.plan)
        with self.assertRaisesRegex(gate.GateError, r"argv\[0\] must be the bound native executable"):
            fixture.freeze()

    def test_attest_rejects_receipt_config_or_artifact_substitution(self) -> None:
        fixture = self.make_fixture()
        frozen = fixture.freeze()
        fixture.write_artifacts()
        receipt_path = fixture.outputs / "candidate.fiqa.receipt.json"
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        receipt["config"]["top_k"] = 119
        receipt["receipt_digest"] = gate.digest_without(receipt, "receipt_digest")
        write_json(receipt_path, receipt)
        with self.assertRaisesRegex(gate.GateError, "native evaluator config mismatch"):
            gate.attest_manifest(fixture.frozen_path, fixture.attestation_path, expected_frozen_manifest_sha256=frozen["file_sha256"])

        fixture = self.make_fixture()
        frozen = fixture.freeze()
        fixture.write_artifacts()
        receipt_path = fixture.outputs / "candidate.fiqa.receipt.json"
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        receipt["artifact_sha256"] = gate.sha256_file(fixture.outputs / "anchor.fiqa.json")
        receipt["receipt_digest"] = gate.digest_without(receipt, "receipt_digest")
        write_json(receipt_path, receipt)
        with self.assertRaisesRegex(gate.GateError, "artifact hash binding mismatch"):
            gate.attest_manifest(fixture.frozen_path, fixture.attestation_path, expected_frozen_manifest_sha256=frozen["file_sha256"])

    def test_attest_rejects_non_native_or_stale_receipts(self) -> None:
        fixture = self.make_fixture()
        frozen = fixture.freeze()
        fixture.write_artifacts()
        receipt_path = fixture.outputs / "anchor.fiqa.receipt.json"
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        receipt["native_evaluator"] = False
        receipt["receipt_digest"] = gate.digest_without(receipt, "receipt_digest")
        write_json(receipt_path, receipt)
        with self.assertRaisesRegex(gate.GateError, "producer flags mismatch"):
            gate.attest_manifest(fixture.frozen_path, fixture.attestation_path, expected_frozen_manifest_sha256=frozen["file_sha256"])

        fixture = self.make_fixture()
        frozen = fixture.freeze()
        fixture.write_artifacts()
        artifact_path = fixture.outputs / "anchor.fiqa.json"
        os.utime(artifact_path, ns=(artifact_path.stat().st_mtime_ns, Path(fixture.frozen_path).stat().st_mtime_ns))
        with self.assertRaisesRegex(gate.GateError, "stale anchor/fiqa metrics output"):
            gate.attest_manifest(fixture.frozen_path, fixture.attestation_path, expected_frozen_manifest_sha256=frozen["file_sha256"])

    def test_attest_rejects_duplicate_receipt_nonce(self) -> None:
        fixture = self.make_fixture()
        frozen = fixture.freeze()
        fixture.write_artifacts()
        candidate_artifact_path = fixture.outputs / "candidate.fiqa.json"
        candidate = json.loads(candidate_artifact_path.read_text(encoding="utf-8"))
        candidate["receipt_nonce"] = "fixture-anchor-fiqa-nonce01"
        write_json(candidate_artifact_path, candidate)
        receipt_path = fixture.outputs / "candidate.fiqa.receipt.json"
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        receipt["nonce"] = candidate["receipt_nonce"]
        write_json(receipt_path, receipt)
        fixture.rebind_receipt("candidate", "fiqa")
        with self.assertRaisesRegex(gate.GateError, "duplicate native receipt nonce"):
            gate.attest_manifest(fixture.frozen_path, fixture.attestation_path, expected_frozen_manifest_sha256=frozen["file_sha256"])

    def test_attest_rejects_boundary_evidence_substitution(self) -> None:
        fixture = self.make_fixture()
        frozen = fixture.freeze()
        fixture.write_artifacts()
        path = fixture.outputs / "candidate.nfcorpus.json"
        artifact = json.loads(path.read_text(encoding="utf-8"))
        artifact["boundary_evidence"]["qids"] = []
        write_json(path, artifact)
        fixture.rebind_receipt("candidate", "nfcorpus")
        with self.assertRaisesRegex(gate.GateError, "boundary guard evidence mismatch"):
            gate.attest_manifest(fixture.frozen_path, fixture.attestation_path, expected_frozen_manifest_sha256=frozen["file_sha256"])

    def test_attest_rejects_per_query_quality_regression_fail_closed(self) -> None:
        fixture = self.make_fixture()
        frozen = fixture.freeze()
        fixture.write_artifacts(candidate_q3=0.5001, candidate_dense=0.6990, candidate_q5=0.5980, candidate_recall=0.79)
        report = gate.attest_manifest(fixture.frozen_path, fixture.attestation_path, expected_frozen_manifest_sha256=frozen["file_sha256"])
        self.assertFalse(report["gate_pass"])
        self.assertTrue(report["failures"])
        self.assertIn("macro:q3_macro_ndcg_at_10", report["failures"])
        self.assertIn("query:fiqa:H-fiqa:q3_recall_at_100_non_regressing", report["failures"])
        self.assertIn("nfcorpus:boundary:q3_recall_at_100_non_regressing", report["failures"])

    def test_attest_rejects_declared_aggregate_substitution(self) -> None:
        fixture = self.make_fixture()
        frozen = fixture.freeze()
        fixture.write_artifacts()
        path = fixture.outputs / "anchor.fiqa.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["aggregate"]["q3"]["ndcg_at_10"] = 0.99
        write_json(path, payload)
        fixture.rebind_receipt("anchor", "fiqa")
        with self.assertRaisesRegex(gate.GateError, "declared aggregate mismatch"):
            gate.attest_manifest(fixture.frozen_path, fixture.attestation_path, expected_frozen_manifest_sha256=frozen["file_sha256"])


if __name__ == "__main__":
    unittest.main()
