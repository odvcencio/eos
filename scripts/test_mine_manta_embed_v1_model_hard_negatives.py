#!/usr/bin/env python3
"""Policy regression tests for persisted model-mined hard negatives."""

from __future__ import annotations

import unittest
from pathlib import Path
import re


SCRIPT = Path(__file__).resolve().parent / "mine_manta_embed_v1_model_hard_negatives.fw"


class ModelHardNegativeMinerPolicyTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = SCRIPT.read_text(encoding="utf-8")

    def test_output_schema_retains_join_keys_and_aligned_doc_ids(self) -> None:
        self.assertRegex(self.source, re.compile(r"Dataset\s+string\s+`json:\"dataset,omitempty\"`"))
        self.assertRegex(self.source, re.compile(r"QueryID\s+string\s+`json:\"query_id,omitempty\"`"))
        self.assertRegex(self.source, re.compile(r"PositiveDocID\s+string\s+`json:\"positive_doc_id,omitempty\"`"))
        self.assertRegex(self.source, re.compile(r"NegativeDocIDs\s+\[\]string\s+`json:\"negative_doc_ids,omitempty\"`"))
        self.assertIn("negative_doc_ids length", self.source)
        self.assertIn("does not match negatives length", self.source)

    def test_mined_rows_get_stable_dataset_source_for_stage_b_merge(self) -> None:
        self.assertIn("records[i].Dataset = dataset", self.source)
        self.assertIn('records[i].Source = dataset + ":model-hard-negatives"', self.source)


if __name__ == "__main__":
    unittest.main()
