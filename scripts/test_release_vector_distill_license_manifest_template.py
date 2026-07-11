#!/usr/bin/env python3
"""Validate the release vector-distill license manifest template."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import build_release_vector_distill_data as adapter  # noqa: E402

TEMPLATE_PATH = REPO_ROOT / "docs" / "release-vector-distill-license-manifest.template.json"


def load_template() -> dict:
    return json.loads(TEMPLATE_PATH.read_text(encoding="utf-8"))


class ReleaseVectorDistillLicenseManifestTemplateTest(unittest.TestCase):
    def test_template_is_valid_json_with_required_datasets(self) -> None:
        payload = load_template()
        datasets = payload["datasets"]
        self.assertEqual({entry["dataset"] for entry in datasets}, set(adapter.DEFAULT_DATASETS))
        self.assertEqual([entry["dataset"] for entry in datasets], ["scifact", "nfcorpus", "fiqa"])

    def test_dataset_entries_are_adapter_field_compatible(self) -> None:
        payload = load_template()
        for entry in payload["datasets"]:
            missing = [field for field in adapter.REQUIRED_LICENSE_FIELDS if field not in entry]
            self.assertEqual(missing, [], entry["dataset"])
            self.assertIsInstance(entry["dataset"], str)
            self.assertIsInstance(entry["license"], str)
            self.assertIsInstance(entry["source_url"], str)
            self.assertIsInstance(entry["raw_sha256"], str)

    def test_all_legal_grants_default_false(self) -> None:
        payload = load_template()
        grant_fields = ("train_allowed", "release_train_allowed", "commercial_use_allowed")
        self.assertEqual(payload["teacher_embeddings_release_cleared"], False)
        for field in grant_fields:
            self.assertEqual(payload["legal_gates"][field], False)
        for entry in payload["datasets"]:
            for field in grant_fields:
                self.assertIs(entry[field], False, f"{entry['dataset']} {field}")

    def test_review_marker_is_explicit_and_non_approving(self) -> None:
        payload = load_template()
        review = payload["review"]
        self.assertEqual(review["status"], "review_required_not_approved")
        self.assertIn("REQUIRED", review["owner"])
        self.assertIn("Template only", review["approval_note"])
        self.assertIn("Do not use as approval", review["approval_note"])

    def test_raw_hashes_are_clearly_labeled_placeholders(self) -> None:
        payload = load_template()
        for entry in payload["datasets"]:
            self.assertTrue(entry["raw_sha256"].startswith("REVIEW_REQUIRED:"), entry["dataset"])
            self.assertEqual(entry["raw_sha256_status"], "placeholder_not_reviewed")


if __name__ == "__main__":
    unittest.main(verbosity=2)
