"""Synthetic tests for the AOQT V7 movement canary gate."""

from __future__ import annotations

import copy
import contextlib
import io
import json
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import aoqt_v7_movement_canary_gate as gate  # noqa: E402


_CANONICAL_SEED191_PAIRINGS: list[list[list[int]]] | None = None


def canonical_seed191_pairings() -> list[list[list[int]]]:
    global _CANONICAL_SEED191_PAIRINGS
    if _CANONICAL_SEED191_PAIRINGS is not None:
        return copy.deepcopy(_CANONICAL_SEED191_PAIRINGS)
    program = textwrap.dedent(
        """
        package main

        import (
            "crypto/sha256"
            "encoding/hex"
            "encoding/json"
            "fmt"
            "math/rand"
        )

        func main() {
            rng := rand.New(rand.NewSource(191))
            stages := make([][][2]int, 8)
            digest := sha256.New()
            for stage := range stages {
                perm := rng.Perm(384)
                pairs := make([][2]int, 192)
                for i := range pairs {
                    a, b := perm[2*i], perm[2*i+1]
                    if a > b {
                        a, b = b, a
                    }
                    pairs[i] = [2]int{a, b}
                    fmt.Fprintf(digest, "%d,%d\\n", a, b)
                }
                stages[stage] = pairs
            }
            data, err := json.Marshal(stages)
            if err != nil {
                panic(err)
            }
            fmt.Println(hex.EncodeToString(digest.Sum(nil)))
            fmt.Println(string(data))
        }
        """
    )
    with tempfile.TemporaryDirectory() as temp:
        source = Path(temp) / "main.go"
        source.write_text(program, encoding="utf-8")
        proc = subprocess.run(["go", "run", str(source)], cwd=SCRIPT_DIR.parent, check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    observed_sha, pairings_json = proc.stdout.splitlines()
    if observed_sha != gate.CANONICAL_D384_SEED191_PAIRINGS_SHA256:
        raise AssertionError(f"canonical pairings sha = {observed_sha}, want {gate.CANONICAL_D384_SEED191_PAIRINGS_SHA256}")
    pairings = json.loads(pairings_json)
    if len(pairings) != gate.AOQT_TOPOLOGY["stages"] or any(len(stage) != gate.AOQT_TOPOLOGY["pairs_per_stage"] for stage in pairings):
        raise AssertionError("canonical pairings fixture has unexpected topology shape")
    _CANONICAL_SEED191_PAIRINGS = pairings
    return copy.deepcopy(pairings)


def write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def safe_angles() -> list[float]:
    return ([0.0001, -0.00011, 0.00012, -0.00013] * 8 + [0.0] * 160) * 8


def sidecar_with_angles(angles: list[float], *, canonical_pairings: bool = True) -> dict:
    if len(angles) != gate.AOQT_TOPOLOGY["angle_count"]:
        raise AssertionError(f"fixture angle count = {len(angles)}")
    pairings = canonical_seed191_pairings() if canonical_pairings else None
    stages = []
    for stage_index in range(gate.AOQT_TOPOLOGY["stages"]):
        start = stage_index * gate.AOQT_TOPOLOGY["pairs_per_stage"]
        stop = start + gate.AOQT_TOPOLOGY["pairs_per_stage"]
        stages.append(
            {
                "pairs": pairings[stage_index] if pairings is not None else [[2 * i, 2 * i + 1] for i in range(gate.AOQT_TOPOLOGY["pairs_per_stage"])],
                "angles": angles[start:stop],
            }
        )
    payload = {
        "schema": gate.SIDECAR_SCHEMA,
        "version": "eos/aoqt-givens-transform/v1",
        "kind": gate.AOQT_TOPOLOGY["id"],
        "dim": gate.AOQT_TOPOLOGY["dim"],
        "seed": gate.AOQT_TOPOLOGY["seed"],
        "angle_cap": gate.AOQT_TOPOLOGY["angle_cap_default"],
        "topology": copy.deepcopy(gate.AOQT_TOPOLOGY),
        "stages": stages,
        "audit": {},
    }
    payload["audit"] = {
        "pairings_sha256": gate.transform_pairings_sha256(stages, "fixture"),
        "angles_sha256": gate.transform_angles_sha256(stages, "fixture"),
        "orthogonality_frobenius_per_dim": 0.0,
    }
    return payload


def train_metrics(
    angles: list[float],
    sidecar: dict,
    *,
    accepted: int = 12,
    attempts: int = 60,
    optimizer_mode: str | None = gate.V7_OPTIMIZER_MODE,
    summary_optimizer_mode: str | None = gate.V7_OPTIMIZER_MODE,
    diagnostics_optimizer_mode: str | None = gate.V7_OPTIMIZER_MODE,
    dev_only: bool = True,
    strategy: str = gate.V7_COORDINATE_SEARCH_STRATEGY,
    trust_region_radius: float = 0.01,
    coordinate_proposal_attempts: int | None = None,
    coordinate_accepted_proposals: int | None = None,
    coordinate_rejected_proposals: int | None = None,
    trust_region_proposal_attempts: int | None = None,
    trust_region_accepted_blocks: int | None = None,
    trust_region_accepted_singles: int = 0,
    trust_region_max_moved_angles: int = 2,
    trust_region_accepted_single_coordinate_rule: str | None = None,
    receipt_strategy: str = gate.V7_COORDINATE_SEARCH_STRATEGY,
    manifest: dict | None = None,
    path: Path = Path(__file__),
) -> dict:
    angle_l2 = sum(angle * angle for angle in angles) ** 0.5
    coordinate_proposal_attempts = attempts if coordinate_proposal_attempts is None else coordinate_proposal_attempts
    coordinate_accepted_proposals = accepted if coordinate_accepted_proposals is None else coordinate_accepted_proposals
    coordinate_rejected_proposals = coordinate_proposal_attempts - coordinate_accepted_proposals if coordinate_rejected_proposals is None else coordinate_rejected_proposals
    trust_region_proposal_attempts = coordinate_proposal_attempts if trust_region_proposal_attempts is None else trust_region_proposal_attempts
    trust_region_accepted_blocks = accepted - trust_region_accepted_singles if trust_region_accepted_blocks is None else trust_region_accepted_blocks
    diagnostics = {
        "optimizer_mode": diagnostics_optimizer_mode,
        "dev_only": dev_only,
        "accepted_steps": accepted,
        "accepted_proposals": accepted,
        "proposal_attempts": attempts,
        "coordinate_search_strategy": strategy,
        "coordinate_proposal_attempts": coordinate_proposal_attempts,
        "coordinate_accepted_proposals": coordinate_accepted_proposals,
        "coordinate_rejected_proposals": coordinate_rejected_proposals,
        "trust_region_radius": trust_region_radius,
        "trust_region_proposal_attempts": trust_region_proposal_attempts,
        "trust_region_accepted_blocks": trust_region_accepted_blocks,
        "trust_region_accepted_singles": trust_region_accepted_singles,
        "trust_region_max_moved_angles": trust_region_max_moved_angles,
    }
    if trust_region_accepted_single_coordinate_rule is not None:
        diagnostics["trust_region_accepted_single_coordinate_rule"] = trust_region_accepted_single_coordinate_rule
    receipts = []
    for ordinal in range(1, attempts + 1):
        is_accepted = ordinal <= coordinate_accepted_proposals
        moved_count = trust_region_max_moved_angles if is_accepted else 1
        receipts.append(
            {
                "ordinal": ordinal,
                "kind": "coordinate",
                "accepted": is_accepted,
                "reason": "accepted" if is_accepted else "loss_increase",
                "coordinate_search_strategy": receipt_strategy,
                "moved_angle_count": moved_count,
                "moved_angle_indices": list(range(moved_count)),
            }
        )
    diagnostics["proposal_receipts"] = receipts
    manifest = manifest or split_manifest(path)
    return {
        "schema": gate.TRAIN_METRICS_SCHEMA,
        "plan": {"optimizer_mode": optimizer_mode, "row_count": train_split_binding(manifest, path)["train_row_count"]},
        "sidecar_binding": sidecar_binding(sidecar),
        "inputs": {"dataset_manifest_sha256": "d" * 64},
        "split_binding": train_split_binding(manifest, path),
        "summary": {
            "optimizer_mode": summary_optimizer_mode,
            "steps": accepted,
            "angle_l2": angle_l2,
            "angle_max_abs": max(abs(angle) for angle in angles),
            "optimizer_diagnostics": diagnostics,
            "quality_claim": False,
        },
    }


def sidecar_binding(sidecar: dict) -> dict:
    evidence = gate.validate_sidecar(sidecar)
    return {
        "schema": gate.TRAIN_SIDECAR_BINDING_SCHEMA,
        "sidecar_schema": gate.SIDECAR_SCHEMA,
        "sidecar_content_sha256": evidence["content_sha256"],
        "topology_sha256": evidence["topology_sha256"],
        "pairings_sha256": evidence["pairings_sha256"],
        "angles_sha256": evidence["angles_sha256"],
        "angle_count": evidence["angle_count"],
    }


def split_manifest(path: Path = Path(__file__)) -> dict:
    dev_qids_by_dataset = {
        "fiqa": ["fiqa-dev-1", "fiqa-dev-2"],
        "nfcorpus": ["nfcorpus-dev-1"],
        "scifact": ["scifact-dev-1", "scifact-dev-2", "scifact-dev-3"],
    }
    train_qids_by_dataset = {
        "fiqa": ["fiqa-train-1", "fiqa-train-2", "fiqa-train-3"],
        "nfcorpus": ["nfcorpus-train-1", "nfcorpus-train-2"],
        "scifact": ["scifact-train-1", "scifact-train-2", "scifact-train-3"],
    }
    train_row_ids = sorted(
        f"{domain}.q3.top10.{qid}"
        for domain, qids in train_qids_by_dataset.items()
        for qid in qids
    )
    dev = {
        "qid_count_by_dataset": {domain: len(qids) for domain, qids in dev_qids_by_dataset.items()},
        "qids_by_dataset": dev_qids_by_dataset,
        "qids_by_dataset_sha256": gate.sha256_json(dev_qids_by_dataset),
    }
    train = {
        "row_count": len(train_row_ids),
        "row_ids": train_row_ids,
        "row_ids_sha256": gate.sha256_json(train_row_ids),
        "qid_count_by_dataset": {domain: len(qids) for domain, qids in train_qids_by_dataset.items()},
        "qids_by_dataset": train_qids_by_dataset,
        "qids_by_dataset_sha256": gate.sha256_json(train_qids_by_dataset),
    }
    manifest = {
        "schema": gate.SPLIT_MANIFEST_SCHEMA,
        "actual_training_ran": False,
        "actual_eval_ran": False,
        "actual_official_data_eval_ran": False,
        "source_plan": {"sha256": "a" * 64},
        "official_qid_registry": {"manifest_sha256": "b" * 64, "source_sha256": "c" * 64},
        "folds": [{"name": "fold-0", "train": train, "dev": dev}],
        "provenance": {},
    }
    manifest["provenance"]["manifest_sha256"] = gate.sha256_json(manifest)
    return manifest


def split_binding(manifest: dict, path: Path = Path(__file__), fold_id: str = "fold-0", validation_path: Path | None = None) -> dict:
    split_evidence = gate.require_split_manifest(manifest, path)
    validation_path = validation_path or path
    split_evidence["split_validation"] = gate.require_split_validation_receipt(split_validation(manifest, path), validation_path, split_evidence)
    return gate.split_fold_binding(split_evidence, fold_id)


def split_validation(manifest: dict, path: Path = Path(__file__)) -> dict:
    split_evidence = gate.require_split_manifest(manifest, path)
    return {
        "schema": gate.SPLIT_VALIDATION_SCHEMA,
        "manifest_path": str(path),
        "manifest_file_sha256": split_evidence["file_sha256"],
        "manifest_payload_sha256": split_evidence["payload_sha256"],
        "source_plan_sha256": split_evidence["source_plan_sha256"],
        "official_qid_registry_sha256": split_evidence["official_qid_registry_sha256"],
        "group_count": 8,
        "row_count": 8,
        "fold_count": 1,
        "reserve_group_count": 0,
        "leakage_proof": {"passed": True, "official_overlap_count": 0},
        "passed": True,
    }


def train_split_binding(manifest: dict, path: Path = Path(__file__), fold_id: str = "fold-0") -> dict:
    split_evidence = gate.require_split_manifest(manifest, path)
    binding = gate.split_train_fold_binding(split_evidence, fold_id)
    binding.update(
        {
            "split_manifest_path": str(path),
            "materialized_manifest_sha256": "d" * 64,
            "materialized_rows_sha256": "e" * 64,
            "materialized_preflight_sha256": "f" * 64,
            "materialized_row_ids_sha256": "1" * 64,
            "materialized_row_count": binding["train_row_count"],
        }
    )
    return binding


def dev_evidence(
    manifest: dict | None = None,
    path: Path = Path(__file__),
    fold_id: str = "fold-0",
    *,
    metrics_path: Path | None = None,
    sidecar_path: Path | None = None,
    preflight_path: Path | None = None,
    sidecar: dict | None = None,
) -> dict:
    manifest = manifest or split_manifest(path)
    transform_sha = gate.sha256_file(sidecar_path) if sidecar_path is not None else "b" * 64
    pairings_sha = sidecar["audit"]["pairings_sha256"] if sidecar is not None else "c" * 64
    angles_sha = sidecar["audit"]["angles_sha256"] if sidecar is not None else "d" * 64
    metrics_sha = gate.sha256_file(metrics_path) if metrics_path is not None else "a" * 64
    preflight_sha = gate.sha256_file(preflight_path) if preflight_path is not None else "f" * 64
    return {
        "schema": gate.DEV_EVIDENCE_SCHEMA,
        "dev_only": True,
        "optimizer_mode": gate.V7_OPTIMIZER_MODE,
        "metrics_path": "metrics.json",
        "metrics_sha256": metrics_sha,
        "preflight_path": "preflight.json",
        "preflight_sha256": preflight_sha,
        "transform_path": "sidecar.json",
        "transform_sha256": transform_sha,
        "pairings_sha256": pairings_sha,
        "angles_sha256": angles_sha,
        "split_binding": train_split_binding(manifest, path, fold_id),
    }


def dev_proxy(*, macro: float = 0.0007, churns: dict[str, int] | None = None, manifest: dict | None = None, path: Path = Path(__file__), validation_path: Path | None = None) -> dict:
    churns = churns or {"fiqa": 9, "nfcorpus": 8, "scifact": 8}
    manifest = manifest or split_manifest(path)
    return {
        "schema": gate.DEV_PROXY_SCHEMA,
        "evidence_label": gate.DEV_PROXY_EVIDENCE_LABEL,
        "split_manifest": split_binding(manifest, path, validation_path=validation_path),
        "domains": {
            domain: {
                "delta": {
                    "q3": {"ndcg_at_10": macro, "recall_at_100": 0.0},
                    "dense": {"ndcg_at_10": 0.0},
                    "q5": {"ndcg_at_10": 0.0},
                },
                "q3_top10_churn": churn,
            }
            for domain, churn in churns.items()
        },
        "macro": {"q3_ndcg_at_10_delta": macro},
    }


class MovementCanaryGateTest(unittest.TestCase):
    def evaluate(self, metrics: dict, sidecar: dict, proxy: dict, *, dev: dict | None = None, manifest: dict | None = None, path: Path = Path(__file__), canonical_pairings_sha256: str | None = None, **kwargs: object) -> dict:
        manifest = manifest or split_manifest(path)
        dev = dev or dev_evidence(manifest, path)
        split_receipt = split_validation(manifest, path)
        if canonical_pairings_sha256 is None:
            canonical_pairings_sha256 = gate.CANONICAL_D384_SEED191_PAIRINGS_SHA256
        return gate.evaluate(metrics, sidecar, dev, proxy, manifest, path, split_validation=split_receipt, split_validation_path=path, canonical_pairings_sha256=canonical_pairings_sha256, **kwargs)

    def evaluate_file_bound(self, root: Path, metrics: dict, sidecar: dict, *, dev: dict | None = None, manifest: dict | None = None, preflight: dict | None = None) -> dict:
        manifest_path = write_json(root / "split.json", manifest or split_manifest(root / "split.json"))
        manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        sidecar_path = write_json(root / "sidecar.json", sidecar)
        preflight_path = write_json(root / "preflight.json", preflight or {"schema": "fixture.preflight.v1"})
        split_validation_path = write_json(root / "split-validation.json", split_validation(manifest_payload, manifest_path))
        metrics_path = write_json(root / "metrics.json", metrics)
        dev_payload = dev or dev_evidence(manifest=manifest_payload, path=manifest_path, metrics_path=metrics_path, sidecar_path=sidecar_path, preflight_path=preflight_path, sidecar=sidecar)
        return gate.evaluate(
            metrics,
            sidecar,
            dev_payload,
            dev_proxy(manifest=manifest_payload, path=manifest_path, validation_path=split_validation_path),
            manifest_payload,
            manifest_path,
            split_validation=json.loads(split_validation_path.read_text(encoding="utf-8")),
            split_validation_path=split_validation_path,
            train_metrics_path=metrics_path,
            sidecar_path=sidecar_path,
            preflight_path=preflight_path,
        )

    def test_synthetic_safe_moving_fixture_passes(self) -> None:
        angles = safe_angles()
        sidecar = sidecar_with_angles(angles)
        result = self.evaluate(train_metrics(angles, sidecar), sidecar, dev_proxy())
        self.assertTrue(result["pass"], result["failures"])
        self.assertEqual(result["decision"], "pass")
        self.assertEqual(result["schema"], gate.SCHEMA)
        self.assertFalse(result["official_heldout_gate"])
        self.assertEqual(result["movement"]["sidecar_binding"]["angle_count"], gate.AOQT_TOPOLOGY["angle_count"])

    def test_v6_canonical_zero_movement_fails_specifically_movement(self) -> None:
        angles = [0.0] * gate.AOQT_TOPOLOGY["angle_count"]
        sidecar = sidecar_with_angles(angles)
        result = self.evaluate(train_metrics(angles, sidecar, accepted=0, attempts=37, optimizer_mode="", summary_optimizer_mode="", diagnostics_optimizer_mode="", dev_only=False, strategy="q3_gain_protected_cone_micro_tail_v2", trust_region_radius=0.0, coordinate_proposal_attempts=0, coordinate_accepted_proposals=0, coordinate_rejected_proposals=0, trust_region_proposal_attempts=0, trust_region_accepted_blocks=0, trust_region_accepted_singles=0, trust_region_max_moved_angles=0), sidecar, dev_proxy())
        self.assertFalse(result["pass"])
        self.assertIn("angle_l2", result["failures"])
        self.assertIn("angles_nonzero_absolute", result["failures"])
        self.assertIn("unique_nonzero_angles", result["failures"])
        self.assertIn("accepted_proposals_absolute", result["failures"])
        self.assertIn("v7_optimizer_mode", result["failures"])
        self.assertIn("v7_dev_only", result["failures"])

    def test_rejects_single_angle_tiny_norm_even_with_dev_lift(self) -> None:
        angles = [1e-5] + [0.0] * (gate.AOQT_TOPOLOGY["angle_count"] - 1)
        sidecar = sidecar_with_angles(angles)
        result = self.evaluate(train_metrics(angles, sidecar, accepted=12, attempts=60), sidecar, dev_proxy(macro=0.01))
        self.assertFalse(result["pass"])
        self.assertIn("angles_not_single_angle", result["failures"])
        self.assertIn("angle_l2", result["failures"])

    def test_rejects_ambiguous_or_threshold_when_only_absolute_acceptance_passes(self) -> None:
        angles = safe_angles()
        sidecar = sidecar_with_angles(angles)
        result = self.evaluate(train_metrics(angles, sidecar, accepted=8, attempts=100), sidecar, dev_proxy())
        self.assertFalse(result["pass"])
        self.assertIn("accepted_proposals_fraction", result["failures"])

    def test_rejects_dev_proxy_floor_failures(self) -> None:
        angles = safe_angles()
        sidecar = sidecar_with_angles(angles)
        proxy = dev_proxy(macro=0.00059, churns={"fiqa": 25, "nfcorpus": 0, "scifact": 1})
        proxy["domains"]["fiqa"]["delta"]["dense"]["ndcg_at_10"] = -0.0006
        proxy["domains"]["scifact"]["delta"]["q5"]["ndcg_at_10"] = -0.002
        result = self.evaluate(train_metrics(angles, sidecar), sidecar, proxy)
        self.assertFalse(result["pass"])
        self.assertIn("q3_macro_ndcg_at_10_lift", result["failures"])
        self.assertIn("nfcorpus_q3_top10_churn", result["failures"])
        self.assertIn("fiqa_dense_ndcg_at_10_delta", result["failures"])
        self.assertIn("scifact_q5_ndcg_at_10_delta", result["failures"])

    def test_hard_min_macro_lift_requires_explicit_policy(self) -> None:
        angles = safe_angles()
        sidecar = sidecar_with_angles(angles)
        preferred = self.evaluate(train_metrics(angles, sidecar), sidecar, dev_proxy(macro=0.0005))
        explicit = self.evaluate(train_metrics(angles, sidecar), sidecar, dev_proxy(macro=0.0005), macro_lift_policy="hard-min-explicit")
        self.assertFalse(preferred["pass"])
        self.assertTrue(explicit["pass"], explicit["failures"])
        self.assertEqual(explicit["policy_id"], gate.HARD_MIN_POLICY_ID)

    def test_cli_writes_deterministic_fail_json_without_success_exit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            angles = [0.0] * gate.AOQT_TOPOLOGY["angle_count"]
            sidecar = sidecar_with_angles(angles)
            sidecar_path = write_json(root / "sidecar.json", sidecar)
            manifest_path = write_json(root / "split.json", split_manifest(root / "split.json"))
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            metrics_path = write_json(root / "metrics.json", train_metrics(angles, sidecar, accepted=0, attempts=37, manifest=manifest, path=manifest_path))
            preflight_path = write_json(root / "preflight.json", {"schema": "fixture.preflight.v1"})
            dev_path = write_json(root / "dev-evidence.json", dev_evidence(manifest=manifest, path=manifest_path, metrics_path=metrics_path, sidecar_path=sidecar_path, preflight_path=preflight_path, sidecar=sidecar))
            split_validation_path = write_json(root / "split-validation.json", split_validation(manifest, manifest_path))
            proxy_path = write_json(root / "proxy.json", dev_proxy(manifest=manifest, path=manifest_path, validation_path=split_validation_path))
            output = root / "out.json"
            code = gate.main(
                [
                    "--train-metrics",
                    str(metrics_path),
                    "--sidecar",
                    str(sidecar_path),
                    "--dev-evidence",
                    str(dev_path),
                    "--dev-proxy-report",
                    str(proxy_path),
                    "--split-manifest",
                    str(manifest_path),
                    "--split-validation",
                    str(split_validation_path),
                    "--preflight",
                    str(preflight_path),
                    "--output-json",
                    str(output),
                ]
            )
            self.assertEqual(code, 1)
            first = output.read_text(encoding="utf-8")
            code = gate.main(
                [
                    "--train-metrics",
                    str(metrics_path),
                    "--sidecar",
                    str(sidecar_path),
                    "--dev-evidence",
                    str(dev_path),
                    "--dev-proxy-report",
                    str(proxy_path),
                    "--split-manifest",
                    str(manifest_path),
                    "--split-validation",
                    str(root / "split-validation.json"),
                    "--preflight",
                    str(preflight_path),
                    "--output-json",
                    str(output),
                    "--no-fail-exit-code",
                ]
            )
            self.assertEqual(code, 0)
            self.assertEqual(first, output.read_text(encoding="utf-8"))
            self.assertEqual(json.loads(first)["decision"], "fail")

    def test_cli_requires_preflight_argument(self) -> None:
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as ctx:
                gate.build_parser().parse_args(
                    [
                        "--train-metrics",
                        "metrics.json",
                        "--sidecar",
                        "sidecar.json",
                        "--dev-evidence",
                        "dev.json",
                        "--dev-proxy-report",
                        "proxy.json",
                        "--split-manifest",
                        "split.json",
                        "--split-validation",
                        "split-validation.json",
                        "--output-json",
                        "out.json",
                    ]
                )
        self.assertNotEqual(ctx.exception.code, 0)

    def test_rejects_non_finite_inputs(self) -> None:
        angles = safe_angles()
        sidecar = sidecar_with_angles(angles)
        bad = copy.deepcopy(dev_proxy())
        bad["domains"]["fiqa"]["delta"]["q3"]["ndcg_at_10"] = float("nan")
        with self.assertRaisesRegex(gate.CanaryGateError, "finite number"):
            self.evaluate(train_metrics(angles, sidecar), sidecar, bad)

    def test_rejects_missing_train_schema(self) -> None:
        angles = safe_angles()
        sidecar = sidecar_with_angles(angles)
        metrics = train_metrics(angles, sidecar)
        del metrics["schema"]
        with self.assertRaisesRegex(gate.CanaryGateError, "train_metrics.schema"):
            self.evaluate(metrics, sidecar, dev_proxy())

    def test_rejects_schema_less_sidecar(self) -> None:
        angles = safe_angles()
        sidecar = sidecar_with_angles(angles)
        metrics = train_metrics(angles, sidecar)
        del metrics["sidecar_binding"]
        metrics["topology"] = {
            "kind": gate.AOQT_TOPOLOGY["id"],
            "dim": gate.AOQT_TOPOLOGY["dim"],
            "stages": gate.AOQT_TOPOLOGY["stages"],
            "pairs_per_stage": gate.AOQT_TOPOLOGY["pairs_per_stage"],
            "angle_count": gate.AOQT_TOPOLOGY["angle_count"],
            "seed": gate.AOQT_TOPOLOGY["seed"],
            "pairings_sha256": sidecar["audit"]["pairings_sha256"],
        }
        metrics["summary"]["angles_sha256"] = sidecar["audit"]["angles_sha256"]
        del sidecar["schema"]
        result = self.evaluate(metrics, sidecar, dev_proxy())
        self.assertTrue(result["pass"], result["failures"])

    def test_rejects_missing_dev_proxy_label(self) -> None:
        angles = safe_angles()
        sidecar = sidecar_with_angles(angles)
        proxy = dev_proxy()
        del proxy["evidence_label"]
        with self.assertRaisesRegex(gate.CanaryGateError, "dev_proxy.evidence_label"):
            self.evaluate(train_metrics(angles, sidecar), sidecar, proxy)

    def test_rejects_missing_dev_proxy_schema(self) -> None:
        angles = safe_angles()
        sidecar = sidecar_with_angles(angles)
        proxy = dev_proxy()
        del proxy["schema"]
        with self.assertRaisesRegex(gate.CanaryGateError, "dev_proxy.schema"):
            self.evaluate(train_metrics(angles, sidecar), sidecar, proxy)

    def test_rejects_official_dev_proxy_label(self) -> None:
        angles = safe_angles()
        sidecar = sidecar_with_angles(angles)
        proxy = dev_proxy()
        proxy["evidence_label"] = "official_heldout"
        with self.assertRaisesRegex(gate.CanaryGateError, "dev_proxy.evidence_label"):
            self.evaluate(train_metrics(angles, sidecar), sidecar, proxy)

    def test_rejects_macro_inflation_mismatch(self) -> None:
        angles = safe_angles()
        sidecar = sidecar_with_angles(angles)
        proxy = dev_proxy(macro=0.0007)
        proxy["macro"]["q3_ndcg_at_10_delta"] = 0.7
        with self.assertRaisesRegex(gate.CanaryGateError, "supplied macro"):
            self.evaluate(train_metrics(angles, sidecar), sidecar, proxy)

    def test_rejects_stale_mismatched_sidecar_metrics_binding(self) -> None:
        angles = safe_angles()
        sidecar = sidecar_with_angles(angles)
        stale_sidecar = sidecar_with_angles([0.0002 if angle != 0.0 else 0.0 for angle in angles])
        with self.assertRaisesRegex(gate.CanaryGateError, "sidecar_binding"):
            self.evaluate(train_metrics(angles, sidecar), stale_sidecar, dev_proxy())

    def test_rejects_bad_sidecar_dimension(self) -> None:
        angles = safe_angles()
        sidecar = sidecar_with_angles(angles)
        metrics = train_metrics(angles, sidecar)
        sidecar["dim"] = gate.AOQT_TOPOLOGY["dim"] - 1
        with self.assertRaisesRegex(gate.CanaryGateError, "sidecar.dim"):
            self.evaluate(metrics, sidecar, dev_proxy())

    def test_rejects_bad_sidecar_angle_count(self) -> None:
        angles = safe_angles()
        sidecar = sidecar_with_angles(angles)
        metrics = train_metrics(angles, sidecar)
        sidecar["stages"][-1]["angles"] = sidecar["stages"][-1]["angles"][:-1]
        with self.assertRaisesRegex(gate.CanaryGateError, "pairs/angles"):
            self.evaluate(metrics, sidecar, dev_proxy())

    def test_rejects_missing_split_manifest_binding(self) -> None:
        angles = safe_angles()
        sidecar = sidecar_with_angles(angles)
        proxy = dev_proxy()
        del proxy["split_manifest"]
        with self.assertRaisesRegex(gate.CanaryGateError, "dev_proxy.split_manifest"):
            self.evaluate(train_metrics(angles, sidecar), sidecar, proxy)

    def test_rejects_wrong_split_manifest_file_sha(self) -> None:
        angles = safe_angles()
        sidecar = sidecar_with_angles(angles)
        proxy = dev_proxy()
        proxy["split_manifest"]["split_manifest_file_sha256"] = "0" * 64
        with self.assertRaisesRegex(gate.CanaryGateError, "exact split manifest fold"):
            self.evaluate(train_metrics(angles, sidecar), sidecar, proxy)

    def test_rejects_fold_qid_count_and_hash_mismatch(self) -> None:
        angles = safe_angles()
        sidecar = sidecar_with_angles(angles)
        proxy = dev_proxy()
        proxy["split_manifest"]["dev_qid_count_by_dataset"]["fiqa"] += 1
        with self.assertRaisesRegex(gate.CanaryGateError, "exact split manifest fold"):
            self.evaluate(train_metrics(angles, sidecar), sidecar, proxy)
        proxy = dev_proxy()
        proxy["split_manifest"]["dev_qid_set_sha256_by_dataset"]["scifact"] = "0" * 64
        with self.assertRaisesRegex(gate.CanaryGateError, "exact split manifest fold"):
            self.evaluate(train_metrics(angles, sidecar), sidecar, proxy)

    def test_rejects_sequential_noncanonical_pairings_under_published_sha(self) -> None:
        angles = safe_angles()
        sidecar = sidecar_with_angles(angles, canonical_pairings=False)
        with self.assertRaisesRegex(gate.CanaryGateError, "not canonical D384 seed191"):
            self.evaluate(
                train_metrics(angles, sidecar),
                sidecar,
                dev_proxy(),
                canonical_pairings_sha256=gate.CANONICAL_D384_SEED191_PAIRINGS_SHA256,
            )

    def test_native_sidecar_and_metrics_are_normalized_without_explicit_envelope(self) -> None:
        angles = safe_angles()
        sidecar = sidecar_with_angles(angles)
        metrics = train_metrics(angles, sidecar)
        del metrics["sidecar_binding"]
        metrics["topology"] = {
            "kind": gate.AOQT_TOPOLOGY["id"],
            "dim": gate.AOQT_TOPOLOGY["dim"],
            "stages": gate.AOQT_TOPOLOGY["stages"],
            "pairs_per_stage": gate.AOQT_TOPOLOGY["pairs_per_stage"],
            "angle_count": gate.AOQT_TOPOLOGY["angle_count"],
            "seed": gate.AOQT_TOPOLOGY["seed"],
            "pairings_sha256": sidecar["audit"]["pairings_sha256"],
        }
        metrics["summary"]["angles_sha256"] = sidecar["audit"]["angles_sha256"]
        native_sidecar = copy.deepcopy(sidecar)
        del native_sidecar["schema"]
        del native_sidecar["topology"]
        result = self.evaluate(metrics, native_sidecar, dev_proxy())
        self.assertTrue(result["pass"], result["failures"])

    def test_rejects_forged_native_metrics_envelope(self) -> None:
        angles = safe_angles()
        sidecar = sidecar_with_angles(angles)
        metrics = train_metrics(angles, sidecar)
        del metrics["sidecar_binding"]
        metrics["topology"] = {
            "kind": gate.AOQT_TOPOLOGY["id"],
            "dim": gate.AOQT_TOPOLOGY["dim"],
            "stages": gate.AOQT_TOPOLOGY["stages"],
            "pairs_per_stage": gate.AOQT_TOPOLOGY["pairs_per_stage"],
            "angle_count": gate.AOQT_TOPOLOGY["angle_count"],
            "seed": gate.AOQT_TOPOLOGY["seed"],
            "pairings_sha256": "0" * 64,
        }
        metrics["summary"]["angles_sha256"] = sidecar["audit"]["angles_sha256"]
        native_sidecar = copy.deepcopy(sidecar)
        del native_sidecar["schema"]
        del native_sidecar["topology"]
        with self.assertRaisesRegex(gate.CanaryGateError, "topology.pairings_sha256"):
            self.evaluate(metrics, native_sidecar, dev_proxy())

    def test_explicit_sidecar_binding_still_rejects_contradictory_native_pairings_hash(self) -> None:
        angles = safe_angles()
        sidecar = sidecar_with_angles(angles)
        metrics = train_metrics(angles, sidecar)
        metrics["topology"] = {
            "kind": gate.AOQT_TOPOLOGY["id"],
            "dim": gate.AOQT_TOPOLOGY["dim"],
            "stages": gate.AOQT_TOPOLOGY["stages"],
            "pairs_per_stage": gate.AOQT_TOPOLOGY["pairs_per_stage"],
            "angle_count": gate.AOQT_TOPOLOGY["angle_count"],
            "seed": gate.AOQT_TOPOLOGY["seed"],
            "pairings_sha256": "0" * 64,
        }
        with self.assertRaisesRegex(gate.CanaryGateError, "topology.pairings_sha256"):
            self.evaluate(metrics, sidecar, dev_proxy())

    def test_explicit_sidecar_binding_still_rejects_contradictory_native_angles_hash(self) -> None:
        angles = safe_angles()
        sidecar = sidecar_with_angles(angles)
        metrics = train_metrics(angles, sidecar)
        metrics["summary"]["angles_sha256"] = "0" * 64
        with self.assertRaisesRegex(gate.CanaryGateError, "summary.angles_sha256"):
            self.evaluate(metrics, sidecar, dev_proxy())

    def test_rejects_production_micro_tail_provenance_even_with_movement(self) -> None:
        angles = safe_angles()
        sidecar = sidecar_with_angles(angles)
        metrics = train_metrics(
            angles,
            sidecar,
            optimizer_mode="",
            summary_optimizer_mode="",
            diagnostics_optimizer_mode="",
            dev_only=False,
            strategy="q3_gain_protected_cone_micro_tail_v2",
            trust_region_radius=0.0,
            coordinate_proposal_attempts=60,
            coordinate_accepted_proposals=12,
            coordinate_rejected_proposals=48,
            trust_region_proposal_attempts=0,
            trust_region_accepted_blocks=0,
            trust_region_max_moved_angles=2,
        )
        result = self.evaluate(metrics, sidecar, dev_proxy())
        self.assertFalse(result["pass"])
        self.assertIn("v7_optimizer_mode", result["failures"])
        self.assertIn("v7_dev_only", result["failures"])
        self.assertIn("v7_coordinate_search_strategy", result["failures"])
        self.assertIn("trust_region_radius_positive", result["failures"])
        self.assertIn("trust_region_proposal_attempts_positive", result["failures"])

    def test_rejects_generic_mislabeled_v7_without_trust_region_shape(self) -> None:
        angles = safe_angles()
        sidecar = sidecar_with_angles(angles)
        metrics = train_metrics(
            angles,
            sidecar,
            strategy="generic_coordinate_search",
            trust_region_radius=0.0,
            coordinate_proposal_attempts=0,
            coordinate_accepted_proposals=0,
            coordinate_rejected_proposals=0,
            trust_region_proposal_attempts=0,
            trust_region_accepted_blocks=0,
            trust_region_max_moved_angles=0,
        )
        result = self.evaluate(metrics, sidecar, dev_proxy())
        self.assertFalse(result["pass"])
        self.assertIn("v7_coordinate_search_strategy", result["failures"])
        self.assertIn("coordinate_proposal_attempts_positive", result["failures"])
        self.assertIn("accepted_proposal_receipt_multi_angle_or_preregistered_single", result["failures"])

    def test_rejects_v7_no_coordinate_path_even_with_generic_acceptance(self) -> None:
        angles = safe_angles()
        sidecar = sidecar_with_angles(angles)
        metrics = train_metrics(
            angles,
            sidecar,
            coordinate_proposal_attempts=0,
            coordinate_accepted_proposals=0,
            coordinate_rejected_proposals=0,
            trust_region_proposal_attempts=1,
            trust_region_accepted_blocks=1,
            trust_region_max_moved_angles=2,
        )
        result = self.evaluate(metrics, sidecar, dev_proxy())
        self.assertFalse(result["pass"])
        self.assertIn("coordinate_proposal_attempts_positive", result["failures"])
        self.assertIn("trust_region_proposal_attempts_match_coordinate", result["failures"])
        self.assertIn("trust_region_accepted_accounting", result["failures"])

    def test_rejects_v7_inconsistent_proposal_counters(self) -> None:
        angles = safe_angles()
        sidecar = sidecar_with_angles(angles)
        metrics = train_metrics(
            angles,
            sidecar,
            coordinate_proposal_attempts=60,
            coordinate_accepted_proposals=12,
            coordinate_rejected_proposals=7,
            trust_region_proposal_attempts=59,
            trust_region_accepted_blocks=11,
            trust_region_accepted_singles=0,
        )
        result = self.evaluate(metrics, sidecar, dev_proxy())
        self.assertFalse(result["pass"])
        self.assertIn("trust_region_proposal_attempts_match_coordinate", result["failures"])
        self.assertIn("coordinate_proposal_accounting", result["failures"])
        self.assertIn("trust_region_accepted_accounting", result["failures"])

    def test_accepts_preregistered_single_coordinate_rule(self) -> None:
        angles = safe_angles()
        sidecar = sidecar_with_angles(angles)
        metrics = train_metrics(
            angles,
            sidecar,
            coordinate_accepted_proposals=12,
            trust_region_accepted_blocks=0,
            trust_region_accepted_singles=12,
            trust_region_max_moved_angles=1,
            trust_region_accepted_single_coordinate_rule=gate.V7_ACCEPTED_SINGLE_COORDINATE_RULE,
        )
        result = self.evaluate(metrics, sidecar, dev_proxy())
        self.assertTrue(result["pass"], result["failures"])

    def test_rejects_forged_train_split_binding(self) -> None:
        angles = safe_angles()
        sidecar = sidecar_with_angles(angles)
        metrics = train_metrics(angles, sidecar)
        metrics["split_binding"]["split_manifest_payload_sha256"] = "0" * 64
        dev = dev_evidence()
        dev["split_binding"] = copy.deepcopy(metrics["split_binding"])
        with self.assertRaisesRegex(gate.CanaryGateError, "train_metrics.split_binding.split_manifest_payload_sha256"):
            self.evaluate(metrics, sidecar, dev_proxy(), dev=dev)

    def test_rejects_wrong_train_split_fold(self) -> None:
        angles = safe_angles()
        sidecar = sidecar_with_angles(angles)
        metrics = train_metrics(angles, sidecar)
        dev = dev_evidence()
        metrics["split_binding"]["fold_id"] = "fold-1"
        dev["split_binding"] = copy.deepcopy(metrics["split_binding"])
        with self.assertRaisesRegex(gate.CanaryGateError, "split_binding.fold_id"):
            self.evaluate(metrics, sidecar, dev_proxy(), dev=dev)

    def test_rejects_wrong_train_row_binding(self) -> None:
        angles = safe_angles()
        sidecar = sidecar_with_angles(angles)
        metrics = train_metrics(angles, sidecar)
        dev = dev_evidence()
        metrics["split_binding"]["train_row_ids_sha256"] = "0" * 64
        dev["split_binding"] = copy.deepcopy(metrics["split_binding"])
        with self.assertRaisesRegex(gate.CanaryGateError, "train_metrics.split_binding.train_row_ids_sha256"):
            self.evaluate(metrics, sidecar, dev_proxy(), dev=dev)

    def test_rejects_aggregate_block_when_accepted_receipts_move_one_angle(self) -> None:
        angles = safe_angles()
        sidecar = sidecar_with_angles(angles)
        metrics = train_metrics(
            angles,
            sidecar,
            trust_region_accepted_blocks=12,
            trust_region_accepted_singles=0,
            trust_region_max_moved_angles=2,
        )
        for receipt in metrics["summary"]["optimizer_diagnostics"]["proposal_receipts"]:
            if receipt["accepted"]:
                receipt["moved_angle_count"] = 1
                receipt["moved_angle_indices"] = [0]
        result = self.evaluate(metrics, sidecar, dev_proxy())
        self.assertFalse(result["pass"])
        self.assertIn("trust_region_max_moved_angles_receipt_bound", result["failures"])
        self.assertIn("accepted_proposal_receipt_multi_angle_or_preregistered_single", result["failures"])

    def test_rejects_train_quality_claims(self) -> None:
        angles = safe_angles()
        sidecar = sidecar_with_angles(angles)
        metrics = train_metrics(angles, sidecar)
        metrics["quality_claim"] = True
        with self.assertRaisesRegex(gate.CanaryGateError, "train_metrics.quality_claim"):
            self.evaluate(metrics, sidecar, dev_proxy())
        metrics = train_metrics(angles, sidecar)
        metrics["summary"]["quality_claim"] = True
        with self.assertRaisesRegex(gate.CanaryGateError, "train_metrics.summary.quality_claim"):
            self.evaluate(metrics, sidecar, dev_proxy())
        metrics = train_metrics(angles, sidecar)
        metrics["summary"]["official_heldout_gate"] = True
        with self.assertRaisesRegex(gate.CanaryGateError, "official_heldout_gate"):
            self.evaluate(metrics, sidecar, dev_proxy())

    def test_rejects_file_bound_dev_evidence_hash_forgery(self) -> None:
        angles = safe_angles()
        sidecar = sidecar_with_angles(angles)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest_path = write_json(root / "split.json", split_manifest(root / "split.json"))
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            split_validation_path = write_json(root / "split-validation.json", split_validation(manifest, manifest_path))
            sidecar_path = write_json(root / "sidecar.json", sidecar)
            preflight_path = write_json(root / "preflight.json", {"schema": "fixture.preflight.v1"})
            metrics = train_metrics(angles, sidecar, manifest=manifest, path=manifest_path)
            metrics_path = write_json(root / "metrics.json", metrics)
            good = dev_evidence(manifest=manifest, path=manifest_path, metrics_path=metrics_path, sidecar_path=sidecar_path, preflight_path=preflight_path, sidecar=sidecar)
            for key in ("metrics_sha256", "transform_sha256", "pairings_sha256", "angles_sha256", "preflight_sha256"):
                bad = copy.deepcopy(good)
                bad[key] = "0" * 64
                with self.assertRaisesRegex(gate.CanaryGateError, f"dev_evidence.{key}"):
                    gate.evaluate(
                        metrics,
                        sidecar,
                        bad,
                        dev_proxy(manifest=manifest, path=manifest_path),
                        manifest,
                        manifest_path,
                        split_validation=json.loads(split_validation_path.read_text(encoding="utf-8")),
                        split_validation_path=split_validation_path,
                        train_metrics_path=metrics_path,
                        sidecar_path=sidecar_path,
                        preflight_path=preflight_path,
                    )

    def test_rejects_count_only_accepted_receipt_forgery(self) -> None:
        angles = safe_angles()
        sidecar = sidecar_with_angles(angles)
        metrics = train_metrics(angles, sidecar)
        del metrics["summary"]["optimizer_diagnostics"]["proposal_receipts"][0]["moved_angle_indices"]
        with self.assertRaisesRegex(gate.CanaryGateError, "moved_angle_indices"):
            self.evaluate(metrics, sidecar, dev_proxy())

    def test_rejects_unsorted_duplicate_or_out_of_bounds_moved_indices(self) -> None:
        angles = safe_angles()
        sidecar = sidecar_with_angles(angles)
        mutations = ([1, 0], [0, 0], [0, gate.AOQT_TOPOLOGY["angle_count"]])
        for moved_indices in mutations:
            metrics = train_metrics(angles, sidecar)
            receipt = metrics["summary"]["optimizer_diagnostics"]["proposal_receipts"][0]
            receipt["moved_angle_indices"] = list(moved_indices)
            receipt["moved_angle_count"] = len(moved_indices)
            with self.assertRaisesRegex(gate.CanaryGateError, "moved_angle_indices"):
                self.evaluate(metrics, sidecar, dev_proxy())

    def test_rejects_accepted_receipt_without_coordinate_kind_or_strategy(self) -> None:
        angles = safe_angles()
        sidecar = sidecar_with_angles(angles)
        metrics = train_metrics(angles, sidecar)
        metrics["summary"]["optimizer_diagnostics"]["proposal_receipts"][0]["kind"] = "generic"
        with self.assertRaisesRegex(gate.CanaryGateError, "accepted receipt must be coordinate"):
            self.evaluate(metrics, sidecar, dev_proxy())
        metrics = train_metrics(angles, sidecar)
        metrics["summary"]["optimizer_diagnostics"]["proposal_receipts"][0]["coordinate_search_strategy"] = "generic_coordinate_search"
        with self.assertRaisesRegex(gate.CanaryGateError, "coordinate_search_strategy"):
            self.evaluate(metrics, sidecar, dev_proxy())

    def test_rejects_generic_accepted_count_that_exceeds_coordinate_receipts(self) -> None:
        angles = safe_angles()
        sidecar = sidecar_with_angles(angles)
        metrics = train_metrics(angles, sidecar, accepted=12, coordinate_accepted_proposals=11, coordinate_rejected_proposals=49, trust_region_accepted_blocks=11)
        result = self.evaluate(metrics, sidecar, dev_proxy())
        self.assertFalse(result["pass"])
        self.assertIn("accepted_proposals_match_coordinate", result["failures"])

    def test_rejects_missing_or_stale_split_validation(self) -> None:
        angles = safe_angles()
        sidecar = sidecar_with_angles(angles)
        metrics = train_metrics(angles, sidecar)
        manifest = split_manifest()
        proxy = dev_proxy(manifest=manifest)
        dev = dev_evidence(manifest)
        with self.assertRaisesRegex(gate.CanaryGateError, "split_validation"):
            gate.evaluate(metrics, sidecar, dev, proxy, manifest, Path(__file__))
        bad = split_validation(manifest)
        bad["passed"] = False
        with self.assertRaisesRegex(gate.CanaryGateError, "split_validation.passed"):
            gate.evaluate(metrics, sidecar, dev, proxy, manifest, Path(__file__), split_validation=bad, split_validation_path=Path(__file__))
        bad = split_validation(manifest)
        bad["official_qid_registry_sha256"] = "0" * 64
        with self.assertRaisesRegex(gate.CanaryGateError, "split_validation.official_qid_registry_sha256"):
            gate.evaluate(metrics, sidecar, dev, proxy, manifest, Path(__file__), split_validation=bad, split_validation_path=Path(__file__))

    def test_rejects_dev_proxy_forbidden_claim_flags(self) -> None:
        angles = safe_angles()
        sidecar = sidecar_with_angles(angles)
        for key in ("official_heldout_gate", "quality_claim", "official_claim", "release_claim", "commercial_claim"):
            proxy = dev_proxy()
            proxy[key] = True
            with self.assertRaisesRegex(gate.CanaryGateError, key):
                self.evaluate(train_metrics(angles, sidecar), sidecar, proxy)

    def test_rejects_missing_or_negative_sidecar_orthogonality(self) -> None:
        angles = safe_angles()
        sidecar = sidecar_with_angles(angles)
        del sidecar["audit"]["orthogonality_frobenius_per_dim"]
        with self.assertRaisesRegex(gate.CanaryGateError, "orthogonality_frobenius_per_dim"):
            self.evaluate(train_metrics(angles, sidecar), sidecar, dev_proxy())
        sidecar = sidecar_with_angles(angles)
        sidecar["audit"]["orthogonality_frobenius_per_dim"] = -1.0
        with self.assertRaisesRegex(gate.CanaryGateError, "orthogonality_frobenius_per_dim"):
            self.evaluate(train_metrics(angles, sidecar), sidecar, dev_proxy())


if __name__ == "__main__":
    unittest.main()
