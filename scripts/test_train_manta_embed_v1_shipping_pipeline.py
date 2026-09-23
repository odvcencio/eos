#!/usr/bin/env python3
"""Policy regression tests for the shipping repair pipeline wrapper."""

from __future__ import annotations

import re
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parent / "train_manta_embed_v1_shipping_pipeline.fw"


class ShippingRepairPipelinePolicyTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = SCRIPT.read_text(encoding="utf-8")

    def test_repair_mode_blocks_corkscrew_install(self) -> None:
        self.assertIn("guard !cfg.InstallCorkScrew", self.source)
        self.assertIn("repair mode will not install CorkScrew assets", self.source)

    def test_stage_a_uses_inference_bootstrap_initializer(self) -> None:
        self.assertIn("--bootstrap-from-inference", self.source)
        match = re.search(
            r"func initRepairStageAArtifact.*?runCmd\(.*?\[\]string\{(?P<args>.*?)\}\)",
            self.source,
            re.DOTALL,
        )
        self.assertIsNotNone(match)
        args = match.group("args")
        self.assertIn('"init-model"', args)
        self.assertIn('"--bootstrap-from-inference", cfg.BootstrapArtifact', args)
        self.assertIn("artifact", args)
        for forbidden in (
            "--name",
            "--vocab-size",
            "--max-seq",
            "--architecture",
            "--model-dim",
            "--embedding-dim",
            "--output-dim",
            "--hidden-dim",
            "--weight-dtype",
            "--attention-heads",
            "--encoder-repeats",
        ):
            self.assertNotIn(forbidden, args)

    def test_explicit_tokenizer_is_verified_against_bootstrap_target(self) -> None:
        self.assertIn('"crypto/sha256"', self.source)
        self.assertIn('"encoding/hex"', self.source)
        self.assertIn("func sha256File", self.source)
        self.assertIn("hex.EncodeToString(hash.Sum(nil))", self.source)
        self.assertIn("func verifyExplicitTokenizerMatchesTarget", self.source)
        self.assertIn("cfg.TokenizerExplicit", self.source)
        self.assertIn("sha256File(cfg.Tokenizer)", self.source)
        self.assertIn("sha256File(targetTokenizer)", self.source)
        self.assertIn("does not match bootstrap-extracted tokenizer", self.source)

    def test_repair_does_not_auto_use_bootstrap_sibling_tokenizer(self) -> None:
        self.assertNotIn("filepath.Join(filepath.Dir(cfg.BootstrapArtifact), cfg.ModelName+\".tokenizer.mll\")", self.source)
        self.assertIn("func initRepairStageAArtifact", self.source)
        self.assertIn('fmt.Errorf("bootstrap-from-inference did not write target tokenizer', self.source)

    def test_score_spectrum_training_has_no_legacy_eval_positional_arg(self) -> None:
        self.assertIn('"--score-spectrum-eval", evalJSONL', self.source)
        self.assertIn('EOS_SHIP_SCORE_SPECTRUM_MAX_BATCH_CANDIDATES", 4096', self.source)
        self.assertIn('"--score-spectrum-max-batch-candidates", strconv.Itoa(cfg.ScoreSpectrumMaxBatchCandidates)', self.source)
        self.assertIn('EOS_SHIP_SCORE_SPECTRUM_ACTIVATION_MICROBATCH_SIZE", 8', self.source)
        self.assertIn('"--score-spectrum-activation-microbatch-size", strconv.Itoa(cfg.ScoreSpectrumActivationMicrobatchSize)', self.source)
        self.assertNotIn("args = append(args, artifact, trainJSONL, evalJSONL)", self.source)
        self.assertRegex(
            self.source,
            re.compile(r"args = append\(args, artifact, trainJSONL\).*?trainCommand = append\(trainCommand, args\.\.\.\)\s+err = runCmd", re.DOTALL),
        )

    def test_repair_training_defaults_legacy_host_route_and_disables_packed_fallback(self) -> None:
        self.assertIn('"EOS_TRAIN_ENABLE_COMPACT_RESIDENT_TRAIN=" + residentTrain', self.source)
        self.assertIn('residentTrain := "0"', self.source)
        self.assertIn('residentTrain = "1"', self.source)
        self.assertIn('envEnabled("EOS_SHIP_TRAIN_ENABLE_COMPACT_RESIDENT_TRAIN")', self.source)
        self.assertIn('"EOS_TRAIN_ENABLE_COMPACT_PACKED_FORWARD=0"', self.source)
        self.assertIn('runCmd(cfg, label, runID+".train.log", "env", trainCommand)', self.source)


if __name__ == "__main__":
    unittest.main()
