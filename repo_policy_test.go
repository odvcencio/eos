package eos_test

import (
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

func TestNoTrackedSpecOrPlanDocs(t *testing.T) {
	cmd := exec.Command("git", "ls-files")
	out, err := cmd.Output()
	if err != nil {
		t.Skipf("git ls-files unavailable: %v", err)
	}
	var bad []string
	for _, raw := range strings.Split(string(out), "\n") {
		path := strings.TrimSpace(raw)
		if path == "" {
			continue
		}
		if isSpecOrPlanDoc(path) {
			bad = append(bad, path)
		}
	}
	if len(bad) > 0 {
		t.Fatalf("spec/plan docs must not be tracked:\n%s", strings.Join(bad, "\n"))
	}
}

func isSpecOrPlanDoc(path string) bool {
	path = strings.ToLower(filepath.ToSlash(path))
	if strings.HasPrefix(path, "specs/") ||
		strings.HasPrefix(path, "plans/") ||
		strings.HasPrefix(path, "docs/specs/") ||
		strings.HasPrefix(path, "docs/plans/") {
		return true
	}
	base := filepath.Base(path)
	switch base {
	case "spec.md", "specs.md", "specification.md", "specifications.md", "plan.md", "plans.md":
		return true
	}
	return strings.HasSuffix(path, ".spec.md") || strings.HasSuffix(path, ".plan.md")
}

func TestTrainMantaCandidateEvalOnlyUsesPairEvalAndExportsBeforeGates(t *testing.T) {
	data, err := os.ReadFile(filepath.Join("scripts", "train_manta_embed_v1_candidate.fw"))
	if err != nil {
		t.Fatalf("read candidate wrapper: %v", err)
	}
	src := string(data)
	if strings.Contains(src, "appendEvalModeArgs") {
		t.Fatalf("candidate wrapper must not append training-mode flags to final eval-only commands")
	}
	exportIdx := strings.Index(src, `"Export sealed MLL"`)
	finalEvalIdx := strings.Index(src, `"Final validation eval-only"`)
	if exportIdx < 0 || finalEvalIdx < 0 {
		t.Fatalf("candidate wrapper missing expected export or eval-only step")
	}
	if exportIdx > finalEvalIdx {
		t.Fatalf("sealed export must run before final eval-only gates so a trained artifact is not left stale")
	}
	finalEvalBlock := sliceBetween(t, src, `evalArgs := []string{"train-embed", "--eval-only"`, `err = runCmd(cfg, "Final validation eval-only"`)
	hardEvalBlock := sliceBetween(t, src, `hardArgs := []string{"train-embed", "--eval-only"`, `err = runCmd(cfg, "Hard holdout eval-only"`)
	for name, block := range map[string]string{"final": finalEvalBlock, "hard": hardEvalBlock} {
		if strings.Contains(block, "--hard-negative-train") || strings.Contains(block, "--pairwise-train") {
			t.Fatalf("%s eval-only command must not inherit training-mode flags:\n%s", name, block)
		}
		if !strings.Contains(block, "--no-tokenizer") {
			t.Fatalf("%s eval-only command must preserve token JSONL no-tokenizer handling:\n%s", name, block)
		}
	}
}

func TestTrainMantaCandidateDoesNotCarryStaleSealedArtifacts(t *testing.T) {
	data, err := os.ReadFile(filepath.Join("scripts", "train_manta_embed_v1_candidate.fw"))
	if err != nil {
		t.Fatalf("read candidate wrapper: %v", err)
	}
	src := string(data)
	if !strings.Contains(src, "func isSealedArtifactPath") {
		t.Fatalf("candidate wrapper must identify sealed artifacts explicitly")
	}
	copyPretrainedBlock := sliceBetween(t, src, `func copyPretrainedArtifact`, `func isSealedArtifactPath`)
	copyEvalBlock := sliceBetween(t, src, `func copyPackageForEval`, `func exportSealedArtifact`)
	for name, block := range map[string]string{"pretrained": copyPretrainedBlock, "eval": copyEvalBlock} {
		if !strings.Contains(block, "isSealedArtifactPath(src)") {
			t.Fatalf("%s package copy must skip sealed artifacts to avoid stale sealed outputs:\n%s", name, block)
		}
	}
	exportBlock := sliceBetween(t, src, `func exportSealedArtifact`, `func prepareConfig`)
	for _, want := range []string{
		`tmpSealed := cfg.SealedArtifact + ".tmp"`,
		`os.Remove(tmpSealed)`,
		`"export-mll", cfg.Artifact, tmpSealed`,
		`"inspect", tmpSealed`,
		`os.Rename(tmpSealed, cfg.SealedArtifact)`,
		`"inspect", cfg.SealedArtifact`,
	} {
		if !strings.Contains(exportBlock, want) {
			t.Fatalf("atomic sealed export missing %q:\n%s", want, exportBlock)
		}
	}
	if strings.Contains(exportBlock, `"export-mll", cfg.Artifact, cfg.SealedArtifact`) {
		t.Fatalf("sealed export must not write directly to the final path:\n%s", exportBlock)
	}
}

func TestTrainMantaCandidatePassesPerEpochRetrievalEvalFlags(t *testing.T) {
	data, err := os.ReadFile(filepath.Join("scripts", "train_manta_embed_v1_candidate.fw"))
	if err != nil {
		t.Fatalf("read candidate wrapper: %v", err)
	}
	src := string(data)
	commonBlock := sliceBetween(t, src, `func trainCommonArgs`, `func pairwiseTrainEnabled`)
	for _, want := range []string{
		`EOS_TRAIN_RETRIEVAL_EVAL_DIR`,
		`"--retrieval-eval-dir"`,
		`"--retrieval-eval-split"`,
		`"--retrieval-eval-max-docs"`,
		`"--retrieval-eval-max-queries"`,
		`"--retrieval-eval-batch-size"`,
		`"--retrieval-eval-top-k"`,
		`"--retrieval-eval-tokenizer"`,
	} {
		if !strings.Contains(src, want) && !strings.Contains(commonBlock, want) {
			t.Fatalf("candidate wrapper missing train retrieval eval passthrough %q", want)
		}
	}
}

func TestTrainMantaCandidateRecordsInitialArtifactAndDeterministicRetrievalEval(t *testing.T) {
	data, err := os.ReadFile(filepath.Join("scripts", "train_manta_embed_v1_candidate.fw"))
	if err != nil {
		t.Fatalf("read candidate wrapper: %v", err)
	}
	src := string(data)
	for _, want := range []string{
		`InitialArtifact`,
		`InitialArtifactSHA256`,
		`EOS_INITIAL_ARTIFACT`,
		`cfg.InitialArtifactSHA256, err = sha256File(cfg.InitialArtifact)`,
		`"initial_artifact": %s`,
		`"initial_artifact_sha256": %s`,
		`evalRunID := cfg.RunID + "-retrieval"`,
		`"EOS_EVAL_RUN_ID=" + evalRunID`,
		`"retrieval_eval_leaderboard": %s`,
	} {
		if !strings.Contains(src, want) {
			t.Fatalf("candidate wrapper missing continuation/retrieval provenance %q", want)
		}
	}
	if strings.Contains(src, `let initialArtifact = os.Getenv("EOS_INITIAL_ARTIFACT")`) {
		t.Fatalf("candidate wrapper must resolve EOS_INITIAL_ARTIFACT during config prep, not inside runCandidate")
	}
}

func TestTrainMantaCandidatePlumbsScoreSpectrumAndRetrievalSelection(t *testing.T) {
	data, err := os.ReadFile(filepath.Join("scripts", "train_manta_embed_v1_candidate.fw"))
	if err != nil {
		t.Fatalf("read candidate wrapper: %v", err)
	}
	src := string(data)
	commonBlock := sliceBetween(t, src, `func trainCommonArgs`, `func pairwiseTrainEnabled`)
	runBlock := sliceBetween(t, src, `func runCandidate`, `func main`)
	for _, want := range []string{
		`EOS_SCORE_SPECTRUM_TRAIN`,
		`EOS_ALLOW_RESEARCH_ONLY_SCORE_SPECTRUM`,
		`EOS_SCORE_SPECTRUM_EVAL`,
		`EOS_SCORE_SPECTRUM_LOSS_MODE`,
		`EOS_SCORE_SPECTRUM_RECOVERY_WEIGHT`,
		`EOS_SCORE_SPECTRUM_RECOVERY_MARGIN`,
		`EOS_SCORE_SPECTRUM_RECOVERY_TOP_K`,
		`EOS_SCORE_SPECTRUM_RECOVERY_TAU`,
		`EOS_SCORE_SPECTRUM_MAX_BATCH_CANDIDATES`,
		`EOS_SCORE_SPECTRUM_ACTIVATION_MICROBATCH_SIZE`,
		`"--score-spectrum-eval"`,
		`"--score-spectrum-loss-mode"`,
		`"--score-spectrum-recovery-weight"`,
		`"--score-spectrum-recovery-margin"`,
		`"--score-spectrum-recovery-top-k"`,
		`"--score-spectrum-recovery-tau"`,
		`"--score-spectrum-max-batch-candidates"`,
		`"--score-spectrum-activation-microbatch-size"`,
	} {
		if !strings.Contains(src, want) && !strings.Contains(commonBlock, want) {
			t.Fatalf("candidate wrapper missing score-spectrum passthrough %q", want)
		}
	}
	for _, want := range []string{
		`scoreSpectrumTrainEnabled(cfg)`,
		`"--score-spectrum-train"`,
		`"--allow-research-only-score-spectrum"`,
		`envEnabled("EOS_PRETOKENIZE_JSONL") && !scoreSpectrumTrainEnabled(cfg)`,
	} {
		if !strings.Contains(runBlock, want) {
			t.Fatalf("candidate wrapper missing score-spectrum run flow %q", want)
		}
	}
	for _, want := range []string{
		`trainRetrievalEvalConfigured(cfg)`,
		`cfg.SelectMetric = "retrieval_ndcg"`,
		`cfg.SelectMetric = "score_margin"`,
		`choose only one of EOS_PAIRWISE_TRAIN, EOS_HARD_NEGATIVE_TRAIN, or EOS_SCORE_SPECTRUM_TRAIN`,
	} {
		if !strings.Contains(src, want) {
			t.Fatalf("candidate wrapper missing retrieval-selection or mutual-exclusion guard %q", want)
		}
	}
}

func TestTrainMantaCandidateScoreSpectrumDoesNotUsePositionalPairEval(t *testing.T) {
	data, err := os.ReadFile(filepath.Join("scripts", "train_manta_embed_v1_candidate.fw"))
	if err != nil {
		t.Fatalf("read candidate wrapper: %v", err)
	}
	src := string(data)
	runBlock := sliceBetween(t, src, `func runCandidate`, `func main`)
	for _, want := range []string{
		`if scoreSpectrumTrainEnabled(cfg) {
			args = append(args, cfg.Artifact, trainJSONLForRun)
		} else {
			args = append(args, cfg.Artifact, trainJSONLForRun, evalJSONLForRun)
		}`,
		`if scoreSpectrumTrainEnabled(cfg) {
			planArgs = append(planArgs, cfg.Artifact, trainJSONLForRun)
		} else {
			planArgs = append(planArgs, cfg.Artifact, trainJSONLForRun, evalJSONLForRun)
		}`,
		`scoreSpectrumEvalOnly := scoreSpectrumTrainEnabled(cfg) && cfg.ScoreSpectrumEval != ""`,
		`if scoreSpectrumEvalOnly {
		evalArgs = append(evalArgs, evalCandidate, trainJSONLForRun)
	} else {
		evalArgs = append(evalArgs, evalCandidate, evalJSONLForRun)
	}`,
		`if scoreSpectrumEvalOnly {
			hardArgs = append(hardArgs, hardCandidate, trainJSONLForRun)
		} else {
			hardArgs = append(hardArgs, hardCandidate, hardEvalJSONLForRun)
		}`,
	} {
		if !strings.Contains(runBlock, want) {
			t.Fatalf("candidate wrapper missing score-spectrum native-eval command shape %q", want)
		}
	}
}

func TestTrainMantaCandidateDoesNotForceResidentRepairRoute(t *testing.T) {
	data, err := os.ReadFile(filepath.Join("scripts", "train_manta_embed_v1_candidate.fw"))
	if err != nil {
		t.Fatalf("read candidate wrapper: %v", err)
	}
	src := string(data)
	block := sliceBetween(t, src, `func applyCandidateRuntimeEnv`, `func runCandidate`)
	if strings.Contains(block, `EOS_TRAIN_ENABLE_COMPACT_RESIDENT_TRAIN`+`=1`) {
		t.Fatalf("candidate wrapper must not force resident training for repair mode: %s", block)
	}
	for _, want := range []string{
		`EOS_TRAIN_ENABLE_COMPACT_RESIDENT_TRAIN`,
		`os.Setenv("EOS_TRAIN_ENABLE_COMPACT_RESIDENT_TRAIN", "0")`,
	} {
		if !strings.Contains(block, want) {
			t.Fatalf("candidate wrapper missing explicit resident-off default %q: %s", want, block)
		}
	}
}

func sliceBetween(t *testing.T, src string, start string, end string) string {
	t.Helper()
	startIdx := strings.Index(src, start)
	if startIdx < 0 {
		t.Fatalf("missing start marker %q", start)
	}
	endIdx := strings.Index(src[startIdx:], end)
	if endIdx < 0 {
		t.Fatalf("missing end marker %q after %q", end, start)
	}
	return src[startIdx : startIdx+endIdx]
}
