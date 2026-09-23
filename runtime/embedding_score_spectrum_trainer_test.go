package eosruntime

import (
	"math"
	"path/filepath"
	"strconv"
	"strings"
	"testing"

	"m31labs.dev/eos/runtime/backends/cuda"
	"m31labs.dev/eos/runtime/backends/metal"
)

func TestEmbeddingTrainerTrainScoreSpectrumStep(t *testing.T) {
	trainer := newTinyTrainableAttentionEmbeddingTrainer(t, 0.005)
	trainer.config.Temperature = 0.05

	metrics, err := trainer.TrainScoreSpectrumStep(tinyEmbeddingScoreSpectrumDataset())
	if err != nil {
		t.Fatalf("train score-spectrum step: %v", err)
	}
	if metrics.BatchSize != 4 {
		t.Fatalf("batch size = %d, want 4 row-local query-candidate scores", metrics.BatchSize)
	}
	if metrics.Loss < 0 {
		t.Fatalf("loss = %f, want non-negative", metrics.Loss)
	}
	if trainer.step != 1 {
		t.Fatalf("step = %d, want 1", trainer.step)
	}

}

func TestEmbeddingTrainerTrainScoreSpectrumStepFoldsSelectedOnlyPositive(t *testing.T) {
	trainer := newTinyTrainableAttentionEmbeddingTrainer(t, 0.005)
	trainer.config.Temperature = 0.05

	selected := 0
	batch := []EmbeddingScoreSpectrumExample{
		{
			QueryTokens:             []int32{0},
			QueryMask:               []int32{1},
			CandidateTokens:         [][]int32{{0}, {1}},
			CandidateMasks:          [][]int32{{1}, {1}},
			SelectedPositiveIndex:   &selected,
			HardNegativeEligible:    []bool{false, true},
			TargetProbabilities:     []float32{1, 0},
			CommercialUseAllowed:    true,
			TrainAllowedForResearch: false,
		},
	}

	metrics, err := trainer.TrainScoreSpectrumStep(batch)
	if err != nil {
		t.Fatalf("train score-spectrum selected-only step: %v", err)
	}
	if metrics.BatchSize != 2 || metrics.Loss < 0 {
		t.Fatalf("metrics = %+v, want one row with two candidates and non-negative loss", metrics)
	}
	if len(batch[0].PositiveIndexes) != 0 {
		t.Fatalf("caller batch positive indexes mutated to %+v, want unchanged empty slice", batch[0].PositiveIndexes)
	}
}

func TestEmbeddingTrainerEvaluateScoreSpectrumNativeMetricsAndNoOptimizerUpdate(t *testing.T) {
	selected0 := 0
	examples := []EmbeddingScoreSpectrumExample{
		{
			PositiveIndexes:       []int{0},
			SelectedPositiveIndex: &selected0,
			HardNegativeEligible:  []bool{false, true},
			TargetProbabilities:   []float32{1, 0},
		},
		{
			PositiveIndexes:       []int{0, 1},
			SelectedPositiveIndex: &selected0,
			HardNegativeEligible:  []bool{false, false, true},
			TargetProbabilities:   []float32{0.2, 0.7, 0.1},
		},
	}

	metrics, err := evaluateScoreSpectrumEncodings(
		[]*embeddingEncodedSequence{
			{pooled: []float32{1, 0}},
			{pooled: []float32{1, 1}},
		},
		[]*embeddingEncodedSequence{
			{pooled: []float32{1, 0}},
			{pooled: []float32{0, 1}},
			{pooled: []float32{1, 0}},
			{pooled: []float32{1, 1}},
			{pooled: []float32{0, 1}},
		},
		[]embeddingCandidateSpan{{Start: 0, End: 2}, {Start: 2, End: 5}},
		examples,
		1,
	)
	if err != nil {
		t.Fatalf("evaluate score-spectrum: %v", err)
	}
	if metrics.RowCount != 2 || metrics.CandidateCount != 5 {
		t.Fatalf("row/candidate counts = %d/%d, want 2/5", metrics.RowCount, metrics.CandidateCount)
	}
	if metrics.AnyPositiveTop1 != 1 || metrics.AnyPositiveRowCount != 2 {
		t.Fatalf("any-positive top1/count = %v/%d, want 1/2", metrics.AnyPositiveTop1, metrics.AnyPositiveRowCount)
	}
	if metrics.OriginalPositiveTop1 != 0.5 || metrics.OriginalPositiveRowCount != 2 {
		t.Fatalf("original-positive top1/count = %v/%d, want 0.5/2", metrics.OriginalPositiveTop1, metrics.OriginalPositiveRowCount)
	}
	if metrics.AlternateRelevantRecovery != 1 || metrics.AlternateRecoveryRowCount != 1 {
		t.Fatalf("alternate recovery/count = %v/%d, want 1/1", metrics.AlternateRelevantRecovery, metrics.AlternateRecoveryRowCount)
	}
	wantMargin := float32((1 + (1 - 1/math.Sqrt2)) / 2)
	if math.Abs(float64(metrics.BestPositiveHardestNegativeMargin-wantMargin)) > 1e-5 || metrics.MarginRowCount != 2 {
		t.Fatalf("margin/count = %v/%d, want %v/2", metrics.BestPositiveHardestNegativeMargin, metrics.MarginRowCount, wantMargin)
	}
	if metrics.TargetCrossEntropy <= 0 || metrics.TargetKL < 0 || metrics.Loss < 0 {
		t.Fatalf("loss/ce/kl = %v/%v/%v, want valid positive metrics", metrics.Loss, metrics.TargetCrossEntropy, metrics.TargetKL)
	}
	if metrics.TargetDistributionRowCount != 2 {
		t.Fatalf("target distribution rows = %d, want 2", metrics.TargetDistributionRowCount)
	}

	trainer := newTinyTrainable3DEmbeddingTrainer(t, 0.05)
	startProfile := trainer.TrainProfile()
	startStep := trainer.step
	if _, err := trainer.EvaluateScoreSpectrum(tinyEmbeddingScoreSpectrumDataset()); err != nil {
		t.Fatalf("evaluate score-spectrum through trainer: %v", err)
	}
	if trainer.step != startStep {
		t.Fatalf("step = %d, want unchanged %d", trainer.step, startStep)
	}
	endProfile := trainer.TrainProfile()
	if endProfile.Step != startProfile.Step || endProfile.Optimizer.UpdateCalls != startProfile.Optimizer.UpdateCalls {
		t.Fatalf("optimizer state changed after eval: start step/update=%d/%d end step/update=%d/%d", startProfile.Step, startProfile.Optimizer.UpdateCalls, endProfile.Step, endProfile.Optimizer.UpdateCalls)
	}
}

func TestEmbeddingTrainerEvaluateScoreSpectrumBatchedMatchesFullEval(t *testing.T) {
	selected0 := 0
	selected1 := 1
	examples := []EmbeddingScoreSpectrumExample{
		{
			QueryTokens:             []int32{0},
			QueryMask:               []int32{1},
			CandidateTokens:         [][]int32{{0}, {1}},
			CandidateMasks:          [][]int32{{1}, {1}},
			PositiveIndexes:         []int{0},
			SelectedPositiveIndex:   &selected0,
			HardNegativeEligible:    []bool{false, true},
			TargetProbabilities:     []float32{0.85, 0.15},
			CommercialUseAllowed:    true,
			TrainAllowedForResearch: false,
		},
		{
			QueryTokens:             []int32{1},
			QueryMask:               []int32{1},
			CandidateTokens:         [][]int32{{2}, {1}, {0}},
			CandidateMasks:          [][]int32{{1}, {1}, {1}},
			PositiveIndexes:         []int{0, 1},
			SelectedPositiveIndex:   &selected1,
			HardNegativeEligible:    []bool{false, false, true},
			TargetProbabilities:     []float32{0.35, 0.55, 0.10},
			CommercialUseAllowed:    true,
			TrainAllowedForResearch: false,
		},
		{
			QueryTokens:             []int32{2},
			QueryMask:               []int32{1},
			CandidateTokens:         [][]int32{{1}, {2}},
			CandidateMasks:          [][]int32{{1}, {1}},
			PositiveIndexes:         []int{1},
			SelectedPositiveIndex:   &selected1,
			HardNegativeEligible:    []bool{true, false},
			TargetProbabilities:     []float32{0, 1},
			CommercialUseAllowed:    true,
			TrainAllowedForResearch: false,
		},
	}

	trainer := newTinyTrainable3DEmbeddingTrainer(t, 0.05)
	full, err := trainer.EvaluateScoreSpectrum(examples)
	if err != nil {
		t.Fatalf("full score-spectrum eval: %v", err)
	}
	batched, err := trainer.EvaluateScoreSpectrumBatched(examples, 1)
	if err != nil {
		t.Fatalf("batched score-spectrum eval: %v", err)
	}
	assertScoreSpectrumEvalMetricsClose(t, batched, full)
	if batched.TargetDistributionRowCount != 3 {
		t.Fatalf("target distribution rows = %d, want 3", batched.TargetDistributionRowCount)
	}
}

func assertScoreSpectrumEvalMetricsClose(t *testing.T, got, want EmbeddingScoreSpectrumEvalMetrics) {
	t.Helper()
	const tol = 1e-5
	checkFloat := func(name string, got, want float32) {
		t.Helper()
		if math.Abs(float64(got-want)) > tol {
			t.Fatalf("%s = %v, want %v", name, got, want)
		}
	}
	checkFloat("loss", got.Loss, want.Loss)
	checkFloat("average score", got.AverageScore, want.AverageScore)
	checkFloat("any positive top1", got.AnyPositiveTop1, want.AnyPositiveTop1)
	checkFloat("original positive top1", got.OriginalPositiveTop1, want.OriginalPositiveTop1)
	checkFloat("alternate recovery", got.AlternateRelevantRecovery, want.AlternateRelevantRecovery)
	checkFloat("margin", got.BestPositiveHardestNegativeMargin, want.BestPositiveHardestNegativeMargin)
	checkFloat("target cross entropy", got.TargetCrossEntropy, want.TargetCrossEntropy)
	checkFloat("target kl", got.TargetKL, want.TargetKL)
	if got.RowCount != want.RowCount ||
		got.CandidateCount != want.CandidateCount ||
		got.AnyPositiveRowCount != want.AnyPositiveRowCount ||
		got.OriginalPositiveRowCount != want.OriginalPositiveRowCount ||
		got.AlternateRecoveryRowCount != want.AlternateRecoveryRowCount ||
		got.MarginRowCount != want.MarginRowCount ||
		got.TargetDistributionRowCount != want.TargetDistributionRowCount {
		t.Fatalf("denominators = %+v, want %+v", got, want)
	}
}

func TestEmbeddingTrainerEvaluateScoreSpectrumFoldsSelectedOnlyPositive(t *testing.T) {
	selected := 0
	examples := []EmbeddingScoreSpectrumExample{
		{
			QueryTokens:             []int32{0},
			QueryMask:               []int32{1},
			CandidateTokens:         [][]int32{{0}, {1}},
			CandidateMasks:          [][]int32{{1}, {1}},
			SelectedPositiveIndex:   &selected,
			HardNegativeEligible:    []bool{false, true},
			TargetProbabilities:     []float32{1, 0},
			CommercialUseAllowed:    true,
			TrainAllowedForResearch: false,
		},
	}

	metrics, err := newTinyTrainable3DEmbeddingTrainer(t, 0.05).EvaluateScoreSpectrum(examples)
	if err != nil {
		t.Fatalf("evaluate score-spectrum selected-only row: %v", err)
	}
	if metrics.RowCount != 1 || metrics.CandidateCount != 2 {
		t.Fatalf("row/candidate counts = %d/%d, want 1/2", metrics.RowCount, metrics.CandidateCount)
	}
	if metrics.AnyPositiveRowCount != 1 || metrics.OriginalPositiveRowCount != 1 {
		t.Fatalf("positive denominators = any %d original %d, want 1/1", metrics.AnyPositiveRowCount, metrics.OriginalPositiveRowCount)
	}
	if len(examples[0].PositiveIndexes) != 0 {
		t.Fatalf("caller examples positive indexes mutated to %+v, want unchanged empty slice", examples[0].PositiveIndexes)
	}
}

func TestEmbeddingTrainerTrainScoreSpectrumRejectsPositiveMarkedHardEligible(t *testing.T) {
	trainer := newTinyTrainableAttentionEmbeddingTrainer(t, 0.005)
	batch := tinyEmbeddingScoreSpectrumDataset()
	batch[0].HardNegativeEligible[0] = true

	_, err := trainer.TrainScoreSpectrumStep(batch)
	if err == nil || !strings.Contains(err.Error(), "cannot be hard-negative eligible") {
		t.Fatalf("error = %v, want positive hard-eligible rejection", err)
	}
	if trainer.step != 0 {
		t.Fatalf("step = %d, want no optimizer update", trainer.step)
	}
}

func TestEmbeddingTrainerScoreSpectrumRejectsSelectedPositiveMarkedHardEligible(t *testing.T) {
	selected := 0
	batch := []EmbeddingScoreSpectrumExample{
		{
			QueryTokens:             []int32{0},
			QueryMask:               []int32{1},
			CandidateTokens:         [][]int32{{0}, {1}},
			CandidateMasks:          [][]int32{{1}, {1}},
			SelectedPositiveIndex:   &selected,
			HardNegativeEligible:    []bool{true, true},
			TargetProbabilities:     []float32{1, 0},
			CommercialUseAllowed:    true,
			TrainAllowedForResearch: false,
		},
	}

	trainer := newTinyTrainableAttentionEmbeddingTrainer(t, 0.005)
	if _, err := trainer.TrainScoreSpectrumStep(batch); err == nil || !strings.Contains(err.Error(), "cannot be hard-negative eligible") {
		t.Fatalf("train error = %v, want selected-positive hard-eligible rejection", err)
	}
	if _, err := newTinyTrainable3DEmbeddingTrainer(t, 0.05).EvaluateScoreSpectrum(batch); err == nil || !strings.Contains(err.Error(), "cannot be hard-negative eligible") {
		t.Fatalf("eval error = %v, want selected-positive hard-eligible rejection", err)
	}
}

func TestEmbeddingTrainerScoreSpectrumExcludedHardCandidatesHaveZeroGradient(t *testing.T) {
	queries := []*embeddingEncodedSequence{{pooled: []float32{1, 0}}}
	candidates := []*embeddingEncodedSequence{
		{pooled: []float32{1, 0}},
		{pooled: []float32{0, 1}},
		{pooled: []float32{-1, 0}},
	}
	queryGrads := [][]float32{{0, 0}}
	candidateGrads := [][]float32{{0, 0}, {0, 0}, {0, 0}}
	examples := []EmbeddingScoreSpectrumExample{{
		PositiveIndexes:      []int{0},
		HardNegativeEligible: []bool{false, true, false},
		TargetProbabilities:  []float32{1, 0, 0},
		HardLossWeight:       1,
		SoftLossWeight:       0,
	}}

	_, _, _, err := accumulateScoreSpectrumGrads(queries, candidates, []embeddingCandidateSpan{{Start: 0, End: 3}}, examples, EmbeddingTrainConfig{Temperature: 1, ScoreSpectrumLossMode: ScoreSpectrumLossModeHardSoft, ScoreSpectrumRecoveryTopK: 4, ScoreSpectrumRecoveryTau: 1}, queryGrads, candidateGrads)
	if err != nil {
		t.Fatalf("accumulate score-spectrum grads: %v", err)
	}
	if candidateGrads[2][0] != 0 || candidateGrads[2][1] != 0 {
		t.Fatalf("excluded candidate grad = %v, want zero", candidateGrads[2])
	}
}

func TestEmbeddingTrainerScoreSpectrumRecoveryChangesLossAndGradients(t *testing.T) {
	queries := []*embeddingEncodedSequence{{pooled: []float32{1, 0}}}
	candidates := []*embeddingEncodedSequence{
		{pooled: []float32{1, 0}},
		{pooled: []float32{0, 1}},
		{pooled: []float32{-1, 0}},
	}
	examples := []EmbeddingScoreSpectrumExample{{
		PositiveIndexes:      []int{0},
		HardNegativeEligible: []bool{false, true, true},
		TargetProbabilities:  []float32{1, 0, 0},
	}}
	baseQueryGrads := [][]float32{{0, 0}}
	baseCandidateGrads := [][]float32{{0, 0}, {0, 0}, {0, 0}}
	baseLoss, _, _, err := accumulateScoreSpectrumGrads(queries, candidates, []embeddingCandidateSpan{{Start: 0, End: 3}}, examples, EmbeddingTrainConfig{
		Temperature:               0.5,
		ScoreSpectrumLossMode:     ScoreSpectrumLossModeHardSoft,
		ScoreSpectrumRecoveryTopK: 4,
		ScoreSpectrumRecoveryTau:  0.5,
	}, baseQueryGrads, baseCandidateGrads)
	if err != nil {
		t.Fatalf("base score-spectrum grads: %v", err)
	}

	recoveryQueryGrads := [][]float32{{0, 0}}
	recoveryCandidateGrads := [][]float32{{0, 0}, {0, 0}, {0, 0}}
	recoveryLoss, _, _, err := accumulateScoreSpectrumGrads(queries, candidates, []embeddingCandidateSpan{{Start: 0, End: 3}}, examples, EmbeddingTrainConfig{
		Temperature:                 0.5,
		ScoreSpectrumLossMode:       ScoreSpectrumLossModeHardSoftRecovery,
		ScoreSpectrumRecoveryWeight: 1,
		ScoreSpectrumRecoveryMargin: 0.1,
		ScoreSpectrumRecoveryTopK:   1,
		ScoreSpectrumRecoveryTau:    0.25,
	}, recoveryQueryGrads, recoveryCandidateGrads)
	if err != nil {
		t.Fatalf("recovery score-spectrum grads: %v", err)
	}
	if recoveryLoss <= baseLoss {
		t.Fatalf("recovery loss = %f, want above base %f", recoveryLoss, baseLoss)
	}
	if recoveryCandidateGrads[1][0] == baseCandidateGrads[1][0] && recoveryCandidateGrads[1][1] == baseCandidateGrads[1][1] {
		t.Fatalf("selected hard-negative gradient did not change: base=%v recovery=%v", baseCandidateGrads[1], recoveryCandidateGrads[1])
	}
	if recoveryCandidateGrads[2][0] != baseCandidateGrads[2][0] || recoveryCandidateGrads[2][1] != baseCandidateGrads[2][1] {
		t.Fatalf("non-topK hard-negative recovery gradient changed: base=%v recovery=%v", baseCandidateGrads[2], recoveryCandidateGrads[2])
	}
}

func TestEmbeddingTrainerScoreSpectrumBaseLossWeightScalesBaseBranch(t *testing.T) {
	queries := []*embeddingEncodedSequence{{pooled: []float32{1, 0}}}
	candidates := []*embeddingEncodedSequence{
		{pooled: []float32{1, 0}},
		{pooled: []float32{0, 1}},
		{pooled: []float32{-1, 0}},
	}
	example := EmbeddingScoreSpectrumExample{
		PositiveIndexes:      []int{0},
		HardNegativeEligible: []bool{false, true, true},
		TargetProbabilities:  []float32{1, 0, 0},
	}
	cfg := EmbeddingTrainConfig{
		Temperature:               1,
		ScoreSpectrumLossMode:     ScoreSpectrumLossModeHardSoft,
		ScoreSpectrumRecoveryTopK: 4,
		ScoreSpectrumRecoveryTau:  1,
	}
	baseQueryGrads := [][]float32{{0, 0}}
	baseCandidateGrads := [][]float32{{0, 0}, {0, 0}, {0, 0}}
	baseLoss, _, _, err := accumulateScoreSpectrumGrads(queries, candidates, []embeddingCandidateSpan{{Start: 0, End: 3}}, []EmbeddingScoreSpectrumExample{example}, cfg, baseQueryGrads, baseCandidateGrads)
	if err != nil {
		t.Fatalf("base score-spectrum grads: %v", err)
	}

	for _, weight := range []float32{0.25, 2} {
		t.Run(strconv.FormatFloat(float64(weight), 'g', -1, 32), func(t *testing.T) {
			weighted := example
			weighted.BaseLossWeight = float32Ptr(weight)
			queryGrads := [][]float32{{0, 0}}
			candidateGrads := [][]float32{{0, 0}, {0, 0}, {0, 0}}
			loss, _, _, err := accumulateScoreSpectrumGrads(queries, candidates, []embeddingCandidateSpan{{Start: 0, End: 3}}, []EmbeddingScoreSpectrumExample{weighted}, cfg, queryGrads, candidateGrads)
			if err != nil {
				t.Fatalf("weighted score-spectrum grads: %v", err)
			}
			assertClose32(t, loss, baseLoss*weight, 1e-6, "loss")
			for i := range queryGrads[0] {
				assertClose32(t, queryGrads[0][i], baseQueryGrads[0][i]*weight, 1e-6, "query grad")
			}
			for row := range candidateGrads {
				for col := range candidateGrads[row] {
					assertClose32(t, candidateGrads[row][col], baseCandidateGrads[row][col]*weight, 1e-6, "candidate grad")
				}
			}
		})
	}
}

func TestEmbeddingTrainerScoreSpectrumBaseLossWeightZeroKeepsAuxDenominatorExact(t *testing.T) {
	queries := []*embeddingEncodedSequence{{pooled: []float32{1, 0}}}
	candidates := []*embeddingEncodedSequence{
		{pooled: []float32{1, 0}},
		{pooled: []float32{0, 1}},
	}
	example := EmbeddingScoreSpectrumExample{
		RowID:                    "aux-only",
		Source:                   "unit",
		CandidateIDs:             []string{"p", "n"},
		CandidateSources:         []string{"qrel", "q3"},
		QrelGains:                []float32{1, 0},
		PositiveIndexes:          []int{0},
		HardNegativeEligible:     []bool{false, false},
		TargetProbabilities:      []float32{1, 0},
		BaseLossWeight:           float32Ptr(0),
		TurboQuantTopKLossWeight: float32Ptr(1),
	}
	cfg := EmbeddingTrainConfig{
		Temperature:                1,
		TurboQuantTopKObjectives:   []TurboQuantPrefixObjective{{Dim: 2, BitWidth: 2, Weight: 0.04}},
		TurboQuantTopKLoss:         TurboQuantTopKLossLambdaNDCG,
		TurboQuantTopKCutoff:       2,
		TurboQuantTopKTau:          0.05,
		TurboQuantTopKNegativeMask: TurboQuantTopKNegativeMaskAll,
		ScoreSpectrumLossMode:      ScoreSpectrumLossModeHardSoft,
		ScoreSpectrumRecoveryTopK:  4,
		ScoreSpectrumRecoveryTau:   1,
	}
	queryGrads := [][]float32{{0, 0}}
	candidateGrads := [][]float32{{0, 0}, {0, 0}}
	loss, _, pairs, err := accumulateScoreSpectrumGrads(queries, candidates, []embeddingCandidateSpan{{Start: 0, End: 2}}, []EmbeddingScoreSpectrumExample{example}, cfg, queryGrads, candidateGrads)
	if err != nil {
		t.Fatalf("aux-only score-spectrum grads: %v", err)
	}
	wantQueryGrads := [][]float32{{0, 0}}
	wantCandidateGrads := [][]float32{{0, 0}, {0, 0}}
	wantLoss, _, wantPairs, err := accumulateTurboQuantPreparedIPTopKScoreSpectrumRowGrads(queries, candidates, embeddingCandidateSpan{Start: 0, End: 2}, 0, example, cfg.TurboQuantTopKObjectives, cfg, 1, 1, cfg.TurboQuantTopKCutoff, cfg.TurboQuantTopKTau, cfg.TurboQuantTopKMargin, cfg.TurboQuantTopKNegativeMask, 1.04, wantQueryGrads, wantCandidateGrads)
	if err != nil {
		t.Fatalf("direct aux score-spectrum grads: %v", err)
	}
	if pairs != 2+wantPairs {
		t.Fatalf("pairs = %d, want dense scoring count plus aux pairs %d", pairs, 2+wantPairs)
	}
	assertClose32(t, loss, wantLoss, 1e-6, "aux-only loss")
	for i := range queryGrads[0] {
		assertClose32(t, queryGrads[0][i], wantQueryGrads[0][i], 1e-6, "query aux grad")
	}
	for row := range candidateGrads {
		for col := range candidateGrads[row] {
			assertClose32(t, candidateGrads[row][col], wantCandidateGrads[row][col], 1e-6, "candidate aux grad")
		}
	}
}

func TestEmbeddingTrainerScoreSpectrumBaseLossWeightZeroBypassesRecoveryHardNegativeRequirement(t *testing.T) {
	example := EmbeddingScoreSpectrumExample{
		RowID:                    "aux-only-no-hard",
		Source:                   "unit",
		QueryTokens:              []int32{0},
		QueryMask:                []int32{1},
		CandidateIDs:             []string{"p", "n"},
		CandidateSources:         []string{"qrel", "q3"},
		QrelGains:                []float32{1, 0},
		CandidateTokens:          [][]int32{{0}, {1}},
		CandidateMasks:           [][]int32{{1}, {1}},
		PositiveIndexes:          []int{0},
		HardNegativeEligible:     []bool{false, false},
		TargetProbabilities:      []float32{1, 0},
		BaseLossWeight:           float32Ptr(0),
		TurboQuantTopKLossWeight: float32Ptr(1),
		CommercialUseAllowed:     true,
	}
	cfg := EmbeddingTrainRunConfig{
		Epochs:                      1,
		BatchSize:                   1,
		TurboQuantTopKObjectives:    []TurboQuantPrefixObjective{{Dim: 3, BitWidth: 2, Weight: 0.04}},
		TurboQuantTopKLoss:          TurboQuantTopKLossLambdaNDCG,
		TurboQuantTopKCutoff:        2,
		TurboQuantTopKNegativeMask:  TurboQuantTopKNegativeMaskAll,
		ScoreSpectrumLossMode:       ScoreSpectrumLossModeRecovery,
		ScoreSpectrumRecoveryWeight: 1,
		ScoreSpectrumRecoveryTopK:   1,
		ScoreSpectrumRecoveryTau:    0.25,
	}
	if _, err := newTinyTrainable3DEmbeddingTrainer(t, 0.05).FitScoreSpectrum([]EmbeddingScoreSpectrumExample{example}, nil, cfg); err != nil {
		t.Fatalf("aux-only recovery fit: %v", err)
	}
	activeBase := example
	activeBase.BaseLossWeight = nil
	_, err := newTinyTrainable3DEmbeddingTrainer(t, 0.05).FitScoreSpectrum([]EmbeddingScoreSpectrumExample{activeBase}, nil, cfg)
	if err == nil || !strings.Contains(err.Error(), "recovery loss requires at least one eligible hard-negative candidate") {
		t.Fatalf("active-base recovery error = %v, want missing hard-negative rejection", err)
	}
}

func TestEmbeddingTrainerScoreSpectrumBaseLossWeightZeroWithoutEffectiveAuxFails(t *testing.T) {
	example := EmbeddingScoreSpectrumExample{
		RowID:                   "runtime-no-objective",
		Source:                  "unit",
		QueryTokens:             []int32{0},
		QueryMask:               []int32{1},
		CandidateTokens:         [][]int32{{0}, {1}},
		CandidateMasks:          [][]int32{{1}, {1}},
		PositiveIndexes:         []int{0},
		HardNegativeEligible:    []bool{false, true},
		TargetProbabilities:     []float32{1, 0},
		BaseLossWeight:          float32Ptr(0),
		CommercialUseAllowed:    true,
		TrainAllowedForResearch: false,
	}
	_, err := newTinyTrainable3DEmbeddingTrainer(t, 0.05).TrainScoreSpectrumStep([]EmbeddingScoreSpectrumExample{example})
	if err == nil || !strings.Contains(err.Error(), "row_id=\"runtime-no-objective\"") || !strings.Contains(err.Error(), "source=\"unit\"") || !strings.Contains(err.Error(), "no active objective") {
		t.Fatalf("error = %v, want current-config no-active-objective diagnostics", err)
	}
}

func TestEmbeddingTrainerScoreSpectrumRecoveryFoldsSelectedOnlyPositive(t *testing.T) {
	selected := 0
	batch := []EmbeddingScoreSpectrumExample{{
		QueryTokens:             []int32{0},
		QueryMask:               []int32{1},
		CandidateTokens:         [][]int32{{0}, {1}},
		CandidateMasks:          [][]int32{{1}, {1}},
		SelectedPositiveIndex:   &selected,
		HardNegativeEligible:    []bool{false, true},
		TargetProbabilities:     []float32{1, 0},
		CommercialUseAllowed:    true,
		TrainAllowedForResearch: false,
		RecoveryLossWeight:      2,
	}}
	trainer := newTinyTrainableAttentionEmbeddingTrainer(t, 0.005)
	trainer.config.ScoreSpectrumLossMode = ScoreSpectrumLossModeRecovery
	trainer.config.ScoreSpectrumRecoveryWeight = 1
	trainer.config.ScoreSpectrumRecoveryTopK = 1
	trainer.config.ScoreSpectrumRecoveryTau = 0.05
	metrics, err := trainer.TrainScoreSpectrumStep(batch)
	if err != nil {
		t.Fatalf("train recovery selected-only score-spectrum row: %v", err)
	}
	if metrics.Loss <= 0 || trainer.step != 1 {
		t.Fatalf("metrics/step = %+v/%d, want positive loss and one update", metrics, trainer.step)
	}
}

func TestEmbeddingTrainerScoreSpectrumTurboQuantTopKLambdaNDCGUsesQrelGainsAndMasks(t *testing.T) {
	example := EmbeddingScoreSpectrumExample{
		QueryTokens:             []int32{0},
		QueryMask:               []int32{1},
		CandidateIDs:            []string{"q3-neg", "strong", "weak", "bm25-neg"},
		CandidateSources:        []string{"q3", "qrel", "qrel", "bm25"},
		QrelGains:               []float32{0, 2, 1, 0},
		CandidateTokens:         [][]int32{{1}, {0}, {2}, {1}},
		CandidateMasks:          [][]int32{{1}, {1}, {1}, {1}},
		PositiveIndexes:         []int{1, 2},
		HardNegativeEligible:    []bool{true, false, false, true},
		TargetProbabilities:     []float32{0, 0.7, 0.3, 0},
		CommercialUseAllowed:    true,
		TrainAllowedForResearch: false,
	}
	if got := scoreSpectrumTopKStaticEligiblePairCount(example, TurboQuantTopKNegativeMaskQ3); got != 3 {
		t.Fatalf("q3 static pairs = %d, want 3", got)
	}
	if got := scoreSpectrumTopKStaticEligiblePairCount(example, TurboQuantTopKNegativeMaskBM25); got != 3 {
		t.Fatalf("bm25 static pairs = %d, want 3", got)
	}
	if got := scoreSpectrumTopKStaticEligiblePairCount(example, TurboQuantTopKNegativeMaskHard); got != 5 {
		t.Fatalf("hard static pairs = %d, want 5", got)
	}
	allExample := example
	allExample.CandidateIDs = []string{"q3-neg", "strong", "weak", "bm25-neg", "unmarked-neg"}
	allExample.CandidateSources = []string{"q3", "qrel", "qrel", "bm25", "manual"}
	allExample.QrelGains = []float32{0, 2, 1, 0, 0}
	allExample.CandidateTokens = [][]int32{{1}, {0}, {2}, {1}, {2}}
	allExample.CandidateMasks = [][]int32{{1}, {1}, {1}, {1}, {1}}
	allExample.HardNegativeEligible = []bool{true, false, false, true, false}
	allExample.TargetProbabilities = []float32{0, 0.7, 0.3, 0, 0}
	if got := scoreSpectrumTopKStaticEligiblePairCount(allExample, TurboQuantTopKNegativeMaskHard); got != 5 {
		t.Fatalf("hard static pairs with unmarked lower = %d, want 5", got)
	}
	if got := scoreSpectrumTopKStaticEligiblePairCount(allExample, TurboQuantTopKNegativeMaskAll); got != 7 {
		t.Fatalf("all static pairs with unmarked lower = %d, want 7", got)
	}

	trainer := newTinyTrainable3DEmbeddingTrainer(t, 0.05)
	trainer.config.TurboQuantTopKObjectives = []TurboQuantPrefixObjective{{Dim: 3, BitWidth: 2, Weight: 0.5}}
	trainer.config.TurboQuantTopKLoss = TurboQuantTopKLossLambdaNDCG
	trainer.config.TurboQuantTopKCutoff = 2
	trainer.config.TurboQuantTopKTau = 0.05
	trainer.config.TurboQuantTopKNegativeMask = TurboQuantTopKNegativeMaskQ3
	metrics, err := trainer.TrainScoreSpectrumStep([]EmbeddingScoreSpectrumExample{example})
	if err != nil {
		t.Fatalf("train score-spectrum top-k step: %v", err)
	}
	if metrics.BatchSize != 7 {
		t.Fatalf("batch size = %d, want 4 dense candidates + 3 q3 top-k label pairs", metrics.BatchSize)
	}
	if metrics.Loss < 0 || math.IsNaN(float64(metrics.Loss)) || math.IsInf(float64(metrics.Loss), 0) {
		t.Fatalf("loss = %f, want finite non-negative", metrics.Loss)
	}
	if trainer.step != 1 {
		t.Fatalf("step = %d, want 1", trainer.step)
	}

	disabled := example
	disabled.TurboQuantTopKLossWeight = float32Ptr(0)
	disabledTrainer := newTinyTrainable3DEmbeddingTrainer(t, 0.05)
	disabledTrainer.config.TurboQuantTopKObjectives = trainer.config.TurboQuantTopKObjectives
	disabledTrainer.config.TurboQuantTopKLoss = trainer.config.TurboQuantTopKLoss
	disabledTrainer.config.TurboQuantTopKCutoff = trainer.config.TurboQuantTopKCutoff
	disabledTrainer.config.TurboQuantTopKTau = trainer.config.TurboQuantTopKTau
	disabledTrainer.config.TurboQuantTopKNegativeMask = trainer.config.TurboQuantTopKNegativeMask
	disabledMetrics, err := disabledTrainer.TrainScoreSpectrumStep([]EmbeddingScoreSpectrumExample{disabled})
	if err != nil {
		t.Fatalf("train score-spectrum top-k disabled row: %v", err)
	}
	if disabledMetrics.BatchSize != 4 {
		t.Fatalf("disabled-row batch size = %d, want 4 dense candidates and zero top-k label pairs", disabledMetrics.BatchSize)
	}
}

func TestEmbeddingTrainerFitScoreSpectrumTurboQuantTopKWorkloadAndFullDimValidation(t *testing.T) {
	batch := []EmbeddingScoreSpectrumExample{{
		QueryTokens:             []int32{0},
		QueryMask:               []int32{1},
		CandidateIDs:            []string{"n", "p2", "p1"},
		CandidateSources:        []string{"q3", "qrel", "qrel"},
		QrelGains:               []float32{0, 2, 1},
		CandidateTokens:         [][]int32{{1}, {0}, {2}},
		CandidateMasks:          [][]int32{{1}, {1}, {1}},
		PositiveIndexes:         []int{1, 2},
		HardNegativeEligible:    []bool{true, false, false},
		TargetProbabilities:     []float32{0, 0.8, 0.2},
		CommercialUseAllowed:    true,
		TrainAllowedForResearch: false,
	}}
	cfg := EmbeddingTrainRunConfig{
		Epochs:                     1,
		BatchSize:                  1,
		TurboQuantTopKObjectives:   []TurboQuantPrefixObjective{{Dim: 3, BitWidth: 2, Weight: 0.5}},
		TurboQuantTopKLoss:         TurboQuantTopKLossLambdaNDCG,
		TurboQuantTopKCutoff:       2,
		TurboQuantTopKTau:          0.05,
		TurboQuantTopKNegativeMask: TurboQuantTopKNegativeMaskQ3,
	}
	summary, err := newTinyTrainable3DEmbeddingTrainer(t, 0.05).FitScoreSpectrum(batch, nil, cfg)
	if err != nil {
		t.Fatalf("fit score-spectrum top-k: %v", err)
	}
	if summary.Workload.PlannedTrainPairs != 6 || summary.Workload.ActualTrainPairs != 6 {
		t.Fatalf("planned/actual train pairs = %d/%d, want 6", summary.Workload.PlannedTrainPairs, summary.Workload.ActualTrainPairs)
	}
	if got := summary.Config.TurboQuantTopKNegativeMask; got != TurboQuantTopKNegativeMaskQ3 {
		t.Fatalf("top-k mask = %q, want q3", got)
	}

	disabled := batch
	disabled[0].TurboQuantTopKLossWeight = float32Ptr(0)
	summary, err = newTinyTrainable3DEmbeddingTrainer(t, 0.05).FitScoreSpectrum(disabled, nil, cfg)
	if err != nil {
		t.Fatalf("fit score-spectrum disabled top-k row: %v", err)
	}
	if summary.Workload.PlannedTrainPairs != 3 || summary.Workload.ActualTrainPairs != 3 {
		t.Fatalf("disabled-row planned/actual train pairs = %d/%d, want 3", summary.Workload.PlannedTrainPairs, summary.Workload.ActualTrainPairs)
	}

	cfg.TurboQuantTopKObjectives = []TurboQuantPrefixObjective{{Dim: 2, BitWidth: 2, Weight: 0.5}}
	_, err = newTinyTrainable3DEmbeddingTrainer(t, 0.05).FitScoreSpectrum(batch, nil, cfg)
	if err == nil || !strings.Contains(err.Error(), "must equal served embedding dimension") {
		t.Fatalf("error = %v, want full-dim top-k rejection", err)
	}
}

func TestEmbeddingTrainerFitScoreSpectrumTurboQuantTopKRecallBranchAccounting(t *testing.T) {
	batch := []EmbeddingScoreSpectrumExample{{
		QueryTokens:                    []int32{0},
		QueryMask:                      []int32{1},
		CandidateIDs:                   []string{"p", "q3-neg", "bm25-neg"},
		CandidateSources:               []string{"qrel", "q3", "bm25"},
		QrelGains:                      []float32{1, 0, 0},
		CandidateTokens:                [][]int32{{0}, {1}, {2}},
		CandidateMasks:                 [][]int32{{1}, {1}, {1}},
		PositiveIndexes:                []int{0},
		HardNegativeEligible:           []bool{false, true, true},
		TargetProbabilities:            []float32{1, 0, 0},
		TurboQuantTopKLossWeight:       float32Ptr(0),
		TurboQuantTopKRecallLossWeight: nil,
		CommercialUseAllowed:           true,
		TrainAllowedForResearch:        false,
	}}
	cfg := EmbeddingTrainRunConfig{
		Epochs:                           1,
		BatchSize:                        1,
		TurboQuantTopKObjectives:         []TurboQuantPrefixObjective{{Dim: 3, BitWidth: 2, Weight: 0.5}},
		TurboQuantTopKLoss:               TurboQuantTopKLossLambdaNDCG,
		TurboQuantTopKCutoff:             2,
		TurboQuantTopKTau:                0.05,
		TurboQuantTopKNegativeMask:       TurboQuantTopKNegativeMaskQ3,
		TurboQuantTopKRecallWeight:       0.25,
		TurboQuantTopKRecallCutoff:       100,
		TurboQuantTopKRecallTau:          0.07,
		TurboQuantTopKRecallMargin:       0.01,
		TurboQuantTopKRecallNegativeMask: TurboQuantTopKNegativeMaskBM25,
	}
	summary, err := newTinyTrainable3DEmbeddingTrainer(t, 0.05).FitScoreSpectrum(batch, nil, cfg)
	if err != nil {
		t.Fatalf("fit score-spectrum top-k recall: %v", err)
	}
	if summary.Workload.PlannedTrainPairs != 4 || summary.Workload.ActualTrainPairs != 4 || summary.FinalTrain.BatchSize != 1 {
		t.Fatalf("workload/final train = %+v final=%+v, want 3 dense + 1 recall pair and 1 example", summary.Workload, summary.FinalTrain)
	}
	if summary.Config.TurboQuantTopKRecallWeight != 0.25 || summary.Config.TurboQuantTopKRecallCutoff != 100 || summary.Config.TurboQuantTopKRecallTau != 0.07 || summary.Config.TurboQuantTopKRecallMargin != 0.01 || summary.Config.TurboQuantTopKRecallNegativeMask != TurboQuantTopKNegativeMaskBM25 {
		t.Fatalf("summary recall config = %+v, want exact configured recall branch", summary.Config)
	}

	batch[0].TurboQuantTopKRecallLossWeight = float32Ptr(0)
	summary, err = newTinyTrainable3DEmbeddingTrainer(t, 0.05).FitScoreSpectrum(batch, nil, cfg)
	if err != nil {
		t.Fatalf("fit score-spectrum disabled top-k recall row: %v", err)
	}
	if summary.Workload.PlannedTrainPairs != 3 || summary.Workload.ActualTrainPairs != 3 {
		t.Fatalf("disabled recall workload = %+v, want only dense candidates", summary.Workload)
	}
}

func TestEmbeddingTrainerFitScoreSpectrumTurboQuantTopKRecallDefaultCutoffIncludesLatePairs(t *testing.T) {
	candidateIDs := make([]string, 12)
	candidateSources := make([]string, 12)
	qrelGains := make([]float32, 12)
	candidateTokens := make([][]int32, 12)
	candidateMasks := make([][]int32, 12)
	hardNegativeEligible := make([]bool, 12)
	targetProbabilities := make([]float32, 12)
	for i := range candidateIDs {
		candidateIDs[i] = "n" + strconv.Itoa(i)
		candidateSources[i] = "manual"
		candidateTokens[i] = []int32{int32(i % 3)}
		candidateMasks[i] = []int32{1}
	}
	candidateIDs[10] = "p-late"
	candidateSources[10] = "qrel"
	qrelGains[10] = 1
	targetProbabilities[10] = 1
	candidateIDs[11] = "bm25-tail"
	candidateSources[11] = "bm25"
	hardNegativeEligible[11] = true

	batch := []EmbeddingScoreSpectrumExample{{
		QueryTokens:                    []int32{0},
		QueryMask:                      []int32{1},
		CandidateIDs:                   candidateIDs,
		CandidateSources:               candidateSources,
		QrelGains:                      qrelGains,
		CandidateTokens:                candidateTokens,
		CandidateMasks:                 candidateMasks,
		PositiveIndexes:                []int{10},
		HardNegativeEligible:           hardNegativeEligible,
		TargetProbabilities:            targetProbabilities,
		TurboQuantTopKLossWeight:       float32Ptr(0),
		TurboQuantTopKRecallLossWeight: nil,
		CommercialUseAllowed:           true,
		TrainAllowedForResearch:        false,
	}}
	cfg := EmbeddingTrainRunConfig{
		Epochs:                           1,
		BatchSize:                        1,
		TurboQuantTopKObjectives:         []TurboQuantPrefixObjective{{Dim: 3, BitWidth: 2, Weight: 0.5}},
		TurboQuantTopKLoss:               TurboQuantTopKLossLambdaNDCG,
		TurboQuantTopKCutoff:             2,
		TurboQuantTopKNegativeMask:       TurboQuantTopKNegativeMaskQ3,
		TurboQuantTopKRecallWeight:       0.25,
		TurboQuantTopKRecallNegativeMask: TurboQuantTopKNegativeMaskBM25,
	}
	defaultCutoffSummary, err := newTinyTrainable3DEmbeddingTrainer(t, 0.05).FitScoreSpectrum(batch, nil, cfg)
	if err != nil {
		t.Fatalf("fit score-spectrum default top-k recall cutoff: %v", err)
	}
	if defaultCutoffSummary.Config.TurboQuantTopKRecallCutoff != 100 {
		t.Fatalf("default recall cutoff = %d, want 100", defaultCutoffSummary.Config.TurboQuantTopKRecallCutoff)
	}
	if defaultCutoffSummary.Workload.PlannedTrainPairs != 13 || defaultCutoffSummary.Workload.ActualTrainPairs != 13 {
		t.Fatalf("default recall cutoff workload = %+v, want 12 dense + 1 late recall pair", defaultCutoffSummary.Workload)
	}

	cfg.TurboQuantTopKRecallCutoff = 10
	narrowCutoffSummary, err := newTinyTrainable3DEmbeddingTrainer(t, 0.05).FitScoreSpectrum(batch, nil, cfg)
	if err != nil {
		t.Fatalf("fit score-spectrum explicit narrow top-k recall cutoff: %v", err)
	}
	if narrowCutoffSummary.Config.TurboQuantTopKRecallCutoff != 10 {
		t.Fatalf("narrow recall cutoff = %d, want explicit 10", narrowCutoffSummary.Config.TurboQuantTopKRecallCutoff)
	}
}

func TestEmbeddingTrainerScoreSpectrumTurboQuantTopKRecallDisabledMatchesLegacyTopK(t *testing.T) {
	example := EmbeddingScoreSpectrumExample{
		QueryTokens:             []int32{0},
		QueryMask:               []int32{1},
		CandidateIDs:            []string{"p", "q3-neg"},
		CandidateSources:        []string{"qrel", "q3"},
		QrelGains:               []float32{1, 0},
		CandidateTokens:         [][]int32{{0}, {1}},
		CandidateMasks:          [][]int32{{1}, {1}},
		PositiveIndexes:         []int{0},
		HardNegativeEligible:    []bool{false, true},
		TargetProbabilities:     []float32{1, 0},
		CommercialUseAllowed:    true,
		TrainAllowedForResearch: false,
	}
	baseTrainer := newTinyTrainable3DEmbeddingTrainer(t, 0.05)
	baseTrainer.config.TurboQuantTopKObjectives = []TurboQuantPrefixObjective{{Dim: 3, BitWidth: 2, Weight: 0.5}}
	baseTrainer.config.TurboQuantTopKLoss = TurboQuantTopKLossLambdaNDCG
	baseTrainer.config.TurboQuantTopKCutoff = 2
	baseTrainer.config.TurboQuantTopKTau = 0.05
	baseTrainer.config.TurboQuantTopKNegativeMask = TurboQuantTopKNegativeMaskQ3
	baseMetrics, err := baseTrainer.TrainScoreSpectrumStep([]EmbeddingScoreSpectrumExample{example})
	if err != nil {
		t.Fatalf("base top-k train: %v", err)
	}

	recallDisabledTrainer := newTinyTrainable3DEmbeddingTrainer(t, 0.05)
	recallDisabledTrainer.config = baseTrainer.config
	recallDisabledTrainer.config.TurboQuantTopKRecallWeight = 0
	recallDisabledMetrics, err := recallDisabledTrainer.TrainScoreSpectrumStep([]EmbeddingScoreSpectrumExample{example})
	if err != nil {
		t.Fatalf("recall-disabled top-k train: %v", err)
	}
	if recallDisabledMetrics.Loss != baseMetrics.Loss || recallDisabledMetrics.AverageScore != baseMetrics.AverageScore || recallDisabledMetrics.BatchSize != baseMetrics.BatchSize {
		t.Fatalf("recall-disabled metrics = %+v, want legacy %+v", recallDisabledMetrics, baseMetrics)
	}
}

func TestEmbeddingTrainerFitScoreSpectrumTurboQuantTopKPreservesSeedAndTelemetry(t *testing.T) {
	batch := []EmbeddingScoreSpectrumExample{{
		QueryTokens:             []int32{0},
		QueryMask:               []int32{1},
		CandidateIDs:            []string{"p", "hn", "unmarked"},
		CandidateSources:        []string{"qrel", "q3", "manual"},
		QrelGains:               []float32{1, 0, 0},
		CandidateTokens:         [][]int32{{0}, {1}, {2}},
		CandidateMasks:          [][]int32{{1}, {1}, {1}},
		PositiveIndexes:         []int{0},
		HardNegativeEligible:    []bool{false, true, false},
		TargetProbabilities:     []float32{1, 0, 0},
		CommercialUseAllowed:    true,
		TrainAllowedForResearch: false,
	}}
	trainer := newTinyTrainable3DEmbeddingTrainer(t, 0.05)
	trainer.config.TurboQuantPrefixBits = []int{2}
	trainer.config.TurboQuantPrefixWeight = 0.25
	trainer.config.TurboQuantPrefixSeed = 1234
	var progress []EmbeddingTrainProgress
	summary, err := trainer.FitScoreSpectrum(batch, nil, EmbeddingTrainRunConfig{
		Epochs:                     1,
		BatchSize:                  1,
		TurboQuantTopKObjectives:   []TurboQuantPrefixObjective{{Dim: 3, BitWidth: 2, Weight: 0.5}},
		TurboQuantTopKLoss:         TurboQuantTopKLossLambdaNDCG,
		TurboQuantTopKNegativeMask: TurboQuantTopKNegativeMaskAll,
		TurboQuantPrefixSeed:       1234,
		ProgressEverySteps:         1,
		Progress:                   func(update EmbeddingTrainProgress) { progress = append(progress, update) },
	})
	if err != nil {
		t.Fatalf("fit score-spectrum top-k seed/telemetry: %v", err)
	}
	if summary.Config.TurboQuantPrefixSeed != 1234 || trainer.config.TurboQuantPrefixSeed != 1234 {
		t.Fatalf("prefix seed summary/trainer = %d/%d, want 1234", summary.Config.TurboQuantPrefixSeed, trainer.config.TurboQuantPrefixSeed)
	}
	if len(summary.Config.TurboQuantPrefixBits) != 0 || len(trainer.config.TurboQuantPrefixBits) != 0 {
		t.Fatalf("prefix bits summary/trainer = %v/%v, want isolated", summary.Config.TurboQuantPrefixBits, trainer.config.TurboQuantPrefixBits)
	}
	if summary.Workload.PlannedTrainPairs != 5 || summary.Workload.ActualTrainPairs != 5 {
		t.Fatalf("planned/actual pairs = %d/%d, want 5", summary.Workload.PlannedTrainPairs, summary.Workload.ActualTrainPairs)
	}
	if len(progress) == 0 {
		t.Fatalf("expected score-spectrum progress")
	}
	if progress[0].FirstSpanCandidates != 3 || progress[0].BatchPairs != 5 {
		t.Fatalf("first progress candidates/pairs = %d/%d, want 3/5", progress[0].FirstSpanCandidates, progress[0].BatchPairs)
	}
	artifactPath := filepath.Join(t.TempDir(), "tiny_train_embed_q8.mll")
	paths, err := trainer.WriteTrainingPackage(artifactPath)
	if err != nil {
		t.Fatalf("write training package: %v", err)
	}
	manifest, err := ReadEmbeddingTrainManifestFile(paths.TrainManifestPath)
	if err != nil {
		t.Fatalf("read train manifest: %v", err)
	}
	if manifest.Config.TurboQuantPrefixSeed != 1234 {
		t.Fatalf("manifest prefix seed = %d, want 1234", manifest.Config.TurboQuantPrefixSeed)
	}
	checkpoint, err := ReadEmbeddingTrainCheckpointFile(paths.CheckpointPath)
	if err != nil {
		t.Fatalf("read checkpoint: %v", err)
	}
	if checkpoint.Config.TurboQuantPrefixSeed != 1234 {
		t.Fatalf("checkpoint prefix seed = %d, want 1234", checkpoint.Config.TurboQuantPrefixSeed)
	}
	reloaded, err := LoadEmbeddingTrainerPackage(artifactPath)
	if err != nil {
		t.Fatalf("reload training package: %v", err)
	}
	if reloaded.config.TurboQuantPrefixSeed != 1234 {
		t.Fatalf("reloaded prefix seed = %d, want 1234", reloaded.config.TurboQuantPrefixSeed)
	}
}

func TestEmbeddingTrainerFitScoreSpectrumRejectsNoAuthorizedUseRows(t *testing.T) {
	batch := tinyEmbeddingScoreSpectrumDataset()
	batch[0].ReleaseTrainAllowed = false
	batch[0].CommercialUseAllowed = false
	batch[0].TrainAllowedForResearch = false
	_, err := newTinyTrainable3DEmbeddingTrainer(t, 0.05).FitScoreSpectrum(batch, nil, EmbeddingTrainRunConfig{
		Epochs:    1,
		BatchSize: 1,
	})
	if err == nil || !strings.Contains(err.Error(), "no authorized training use") {
		t.Fatalf("fit no-authorized-use error = %v, want fail-closed legal gate", err)
	}
}

func TestEmbeddingTrainerFitScoreSpectrumEvalOnlyDoesNotUpdate(t *testing.T) {
	trainer := newTinyTrainable3DEmbeddingTrainer(t, 0.05)
	startStep := trainer.step

	summary, err := trainer.FitScoreSpectrum(tinyEmbeddingScoreSpectrumDataset(), tinyEncoderPairDataset(), EmbeddingTrainRunConfig{
		EvalOnly:  true,
		BatchSize: 1,
	})
	if err != nil {
		t.Fatalf("fit score-spectrum eval-only: %v", err)
	}
	if trainer.step != startStep || summary.StepsRun != 0 || summary.DeltaProfile.Step != 0 || summary.DeltaProfile.Optimizer.UpdateCalls != 0 {
		t.Fatalf("optimizer changed trainer_step=%d start=%d stepsRun=%d deltaStep=%d updateCalls=%d", trainer.step, startStep, summary.StepsRun, summary.DeltaProfile.Step, summary.DeltaProfile.Optimizer.UpdateCalls)
	}
	if summary.FinalEval == nil || summary.Workload.ActualEvalPairs == 0 {
		t.Fatalf("missing eval metrics/workload: final=%v actualEvalPairs=%d", summary.FinalEval, summary.Workload.ActualEvalPairs)
	}
}

func TestEmbeddingTrainerFitScoreSpectrumRetrievalOnlyEvalSelectionAndHistory(t *testing.T) {
	dataset := writeTinyRetrievalGateFixture(t)
	corpusPath, queriesPath, qrelsPath := BEIRRetrievalPaths(dataset, "test")
	tok := tinyEmbeddingTokenizerFile()
	summary, err := newTinyTrainable3DEmbeddingTrainer(t, 0.05).FitScoreSpectrum(tinyEmbeddingScoreSpectrumDataset(), nil, EmbeddingTrainRunConfig{
		Epochs:               1,
		BatchSize:            1,
		EvalEveryEpoch:       1,
		EvalEverySteps:       1,
		RestoreBest:          true,
		SelectMetric:         "retrieval_map",
		ScoreSpectrumEval:    tinyEmbeddingScoreSpectrumDataset(),
		RetrievalEvalRuntime: New(cuda.New(), metal.New()),
		RetrievalEval: RetrievalEvalConfig{
			DatasetName: "tiny",
			CorpusPath:  corpusPath,
			QueriesPath: queriesPath,
			QrelsPath:   qrelsPath,
			BatchSize:   2,
		},
		RetrievalEvalTokenizer: &tok,
	})
	if err != nil {
		t.Fatalf("fit score-spectrum with retrieval-only eval: %v", err)
	}
	if !summary.RestoredBest {
		t.Fatal("expected restore-best to run from retrieval-only selection")
	}
	if summary.BestEval == nil || summary.BestEval.RetrievalMAPAt100 <= 0 {
		t.Fatalf("best eval = %+v, want retrieval MAP > 0", summary.BestEval)
	}
	if summary.FinalEval == nil || summary.FinalEval.RetrievalMAPAt100 <= 0 || summary.FinalEval.PairCount != 0 {
		t.Fatalf("final eval = %+v, want retrieval-only metrics with pair_count=0", summary.FinalEval)
	}
	if summary.FinalScoreSpectrumEval == nil || summary.BestScoreSpectrumEval == nil {
		t.Fatalf("missing score-spectrum eval metrics: final=%+v best=%+v", summary.FinalScoreSpectrumEval, summary.BestScoreSpectrumEval)
	}
	if len(summary.EvalHistory) != summary.Workload.ActualEvalPasses {
		t.Fatalf("eval history len = %d, want actual eval passes %d", len(summary.EvalHistory), summary.Workload.ActualEvalPasses)
	}
	for i, record := range summary.EvalHistory {
		if record.Eval == nil || record.Eval.RetrievalMAPAt100 <= 0 || record.Eval.PairCount != 0 {
			t.Fatalf("eval history[%d] eval = %+v, want retrieval-only metrics", i, record.Eval)
		}
		if record.ScoreSpectrumEval == nil {
			t.Fatalf("eval history[%d] missing score-spectrum eval metrics", i)
		}
	}
}

func TestEmbeddingTrainerFitScoreSpectrumPreservesInheritedRecoveryWeight(t *testing.T) {
	trainer := newTinyTrainable3DEmbeddingTrainer(t, 0.05)
	trainer.config.ScoreSpectrumLossMode = ScoreSpectrumLossModeRecovery
	trainer.config.ScoreSpectrumRecoveryWeight = 1.5
	trainer.config.ScoreSpectrumRecoveryTopK = 1
	trainer.config.ScoreSpectrumRecoveryTau = 0.05

	summary, err := trainer.FitScoreSpectrum(tinyEmbeddingScoreSpectrumDataset(), nil, EmbeddingTrainRunConfig{
		Epochs:    1,
		BatchSize: 2,
	})
	if err != nil {
		t.Fatalf("fit score-spectrum with inherited recovery weight: %v", err)
	}
	if summary.Config.ScoreSpectrumLossMode != ScoreSpectrumLossModeRecovery {
		t.Fatalf("summary loss mode = %q, want %q", summary.Config.ScoreSpectrumLossMode, ScoreSpectrumLossModeRecovery)
	}
	if summary.Config.ScoreSpectrumRecoveryWeight != 1.5 {
		t.Fatalf("summary recovery weight = %v, want 1.5", summary.Config.ScoreSpectrumRecoveryWeight)
	}
	if trainer.config.ScoreSpectrumRecoveryWeight != 1.5 {
		t.Fatalf("trainer recovery weight = %v, want 1.5", trainer.config.ScoreSpectrumRecoveryWeight)
	}
}

func TestEmbeddingTrainerFitScoreSpectrumResearchOnlyRequiresExplicitFlag(t *testing.T) {
	_, err := newTinyTrainable3DEmbeddingTrainer(t, 0.05).FitScoreSpectrum(tinyResearchOnlyEmbeddingScoreSpectrumDataset(), nil, EmbeddingTrainRunConfig{
		Epochs:    1,
		BatchSize: 2,
	})
	if err == nil || !strings.Contains(err.Error(), "AllowResearchOnlyScoreSpectrum") {
		t.Fatalf("error = %v, want explicit research flag rejection", err)
	}

	summary, err := newTinyTrainable3DEmbeddingTrainer(t, 0.05).FitScoreSpectrum(tinyResearchOnlyEmbeddingScoreSpectrumDataset(), nil, EmbeddingTrainRunConfig{
		Epochs:                         1,
		BatchSize:                      2,
		AllowResearchOnlyScoreSpectrum: true,
	})
	if err != nil {
		t.Fatalf("fit research-only score-spectrum: %v", err)
	}
	if summary.StepsRun != 1 || summary.Workload.ActualTrainExamples != 2 || summary.Workload.ActualTrainPairs != 4 {
		t.Fatalf("steps/examples/pairs = %d/%d/%d, want 1/2/4", summary.StepsRun, summary.Workload.ActualTrainExamples, summary.Workload.ActualTrainPairs)
	}
}

func TestEmbeddingTrainerFitScoreSpectrumUsesQueryAverageLossAndPairWorkload(t *testing.T) {
	summary, err := newTinyTrainable3DEmbeddingTrainer(t, 0.05).FitScoreSpectrum(tinyEmbeddingScoreSpectrumDataset(), nil, EmbeddingTrainRunConfig{
		Epochs:    1,
		BatchSize: 2,
	})
	if err != nil {
		t.Fatalf("fit score-spectrum: %v", err)
	}
	if summary.FinalTrain.BatchSize != 2 {
		t.Fatalf("final train batch size = %d, want 2 query examples", summary.FinalTrain.BatchSize)
	}
	if summary.Workload.ActualTrainPairs != 4 {
		t.Fatalf("actual train pairs = %d, want 4 row-local candidate scores", summary.Workload.ActualTrainPairs)
	}
}

func TestEmbeddingTrainerFitScoreSpectrumRejectsSingleTargetObjectives(t *testing.T) {
	tests := []struct {
		name string
		cfg  EmbeddingTrainRunConfig
		want string
	}{
		{
			name: "matryoshka",
			cfg:  EmbeddingTrainRunConfig{MatryoshkaDims: []int{2}, MatryoshkaWeights: []float32{1}},
			want: "matryoshka",
		},
		{
			name: "turboquant prefix",
			cfg:  EmbeddingTrainRunConfig{TurboQuantPrefixBits: []int{4}},
			want: "turboquant prefix",
		},
		{
			name: "turboquant compact",
			cfg:  EmbeddingTrainRunConfig{TurboQuantCompactObjectives: []TurboQuantPrefixObjective{{Dim: 3, BitWidth: 4, Weight: 1}}},
			want: "turboquant compact",
		},
		{
			name: "turboquant rank margin",
			cfg:  EmbeddingTrainRunConfig{TurboQuantRankMarginObjectives: []TurboQuantPrefixObjective{{Dim: 3, BitWidth: 4, Weight: 1}}},
			want: "turboquant rank-margin",
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			cfg := tc.cfg
			cfg.Epochs = 1
			cfg.BatchSize = 2
			_, err := newTinyTrainable3DEmbeddingTrainer(t, 0.05).FitScoreSpectrum(tinyEmbeddingScoreSpectrumDataset(), nil, cfg)
			if err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("error = %v, want %q", err, tc.want)
			}
		})
	}
}

func TestEmbeddingTrainerFitScoreSpectrumIsolatesInheritedCompactPackageObjectives(t *testing.T) {
	source := newTinyTrainable3DEmbeddingTrainer(t, 0.05)
	source.config.MatryoshkaDims = []int{2}
	source.config.MatryoshkaWeights = []float32{1}
	source.config.TurboQuantCompactObjectives = []TurboQuantPrefixObjective{{Dim: 2, BitWidth: 2, Weight: 0.25}}
	source.config.TurboQuantPrefixSeed = 11
	artifactPath := filepath.Join(t.TempDir(), "tiny_train_embed_q8.mll")
	if _, err := source.WriteTrainingPackage(artifactPath); err != nil {
		t.Fatalf("write source package: %v", err)
	}

	trainer, err := LoadEmbeddingTrainerPackage(artifactPath)
	if err != nil {
		t.Fatalf("load source package: %v", err)
	}
	summary, err := trainer.FitScoreSpectrum(tinyEmbeddingScoreSpectrumDataset(), nil, EmbeddingTrainRunConfig{
		Epochs:    1,
		BatchSize: 2,
	})
	if err != nil {
		t.Fatalf("fit score-spectrum from compact package: %v", err)
	}
	if len(summary.Config.TurboQuantCompactObjectives) != 0 || len(trainer.config.TurboQuantCompactObjectives) != 0 {
		t.Fatalf("compact objectives were not isolated: summary=%+v trainer=%+v", summary.Config.TurboQuantCompactObjectives, trainer.config.TurboQuantCompactObjectives)
	}
	if len(summary.Config.MatryoshkaDims) != 0 || len(trainer.config.MatryoshkaDims) != 0 {
		t.Fatalf("matryoshka objectives were not isolated: summary=%v trainer=%v", summary.Config.MatryoshkaDims, trainer.config.MatryoshkaDims)
	}
	if trainer.config.TurboQuantPrefixSeed != 0 {
		t.Fatalf("prefix seed = %d, want cleared", trainer.config.TurboQuantPrefixSeed)
	}

	paths, err := trainer.WriteTrainingPackage(artifactPath)
	if err != nil {
		t.Fatalf("rewrite isolated package: %v", err)
	}
	trainManifest, err := ReadEmbeddingTrainManifestFile(paths.TrainManifestPath)
	if err != nil {
		t.Fatalf("read train manifest: %v", err)
	}
	if len(trainManifest.Config.TurboQuantCompactObjectives) != 0 {
		t.Fatalf("train manifest compact objectives = %+v, want cleared", trainManifest.Config.TurboQuantCompactObjectives)
	}
	checkpoint, err := ReadEmbeddingTrainCheckpointFile(paths.CheckpointPath)
	if err != nil {
		t.Fatalf("read checkpoint: %v", err)
	}
	if len(checkpoint.Config.TurboQuantCompactObjectives) != 0 {
		t.Fatalf("checkpoint compact objectives = %+v, want cleared", checkpoint.Config.TurboQuantCompactObjectives)
	}
	packageManifest, err := ReadPackageManifestFile(paths.PackageManifestPath)
	if err != nil {
		t.Fatalf("read package manifest: %v", err)
	}
	if err := packageManifest.VerifyFiles(map[string]string{
		"artifact":           paths.ArtifactPath,
		"embedding_manifest": paths.EmbeddingManifestPath,
		"weights":            paths.WeightFilePath,
		"memory_plan":        paths.MemoryPlanPath,
		"train_manifest":     paths.TrainManifestPath,
		"checkpoint":         paths.CheckpointPath,
		"train_profile":      paths.TrainProfilePath,
	}); err != nil {
		t.Fatalf("verify package manifest: %v", err)
	}
	for _, want := range []string{"matryoshka", "turboquant_compact_objectives", "turboquant_prefix_seed"} {
		if !hasScoreSpectrumObjectiveName(packageManifest.ScoreSpectrum.AutoClearedObjectives, want) {
			t.Fatalf("auto-cleared objectives = %v, missing %q", packageManifest.ScoreSpectrum.AutoClearedObjectives, want)
		}
		if !hasScoreSpectrumObjectiveName(trainManifest.ScoreSpectrum.IsolatedInheritedObjectives, want) {
			t.Fatalf("isolated objectives = %v, missing %q", trainManifest.ScoreSpectrum.IsolatedInheritedObjectives, want)
		}
	}
}

func TestEmbeddingTrainerFitScoreSpectrumIsolatesInheritedPrefixRankAndMatryoshkaObjectives(t *testing.T) {
	trainer := newTinyTrainable3DEmbeddingTrainer(t, 0.05)
	trainer.config.MatryoshkaDims = []int{2}
	trainer.config.MatryoshkaWeights = []float32{0.5}
	trainer.config.TurboQuantPrefixBits = []int{2}
	trainer.config.TurboQuantPrefixObjectives = []TurboQuantPrefixObjective{{Dim: 2, BitWidth: 2, Weight: 0.25}}
	trainer.config.TurboQuantPrefixWeight = 0.75
	trainer.config.TurboQuantPrefixSeed = 17
	trainer.config.TurboQuantPrefixScoreMode = TurboQuantPrefixScoreModePreparedIP
	trainer.config.TurboQuantRankMarginObjectives = []TurboQuantPrefixObjective{{Dim: 2, BitWidth: 2, Weight: 0.5}}
	trainer.config.TurboQuantRankMargin = 0.03

	summary, err := trainer.FitScoreSpectrum(tinyEmbeddingScoreSpectrumDataset(), nil, EmbeddingTrainRunConfig{
		Epochs:    1,
		BatchSize: 2,
	})
	if err != nil {
		t.Fatalf("fit score-spectrum with inherited objectives: %v", err)
	}
	if err := validateScoreSpectrumTrainerConfig(trainer.config); err != nil {
		t.Fatalf("trainer config still has incompatible objectives: %v", err)
	}
	if err := validateScoreSpectrumRunConfig(summary.Config); err != nil {
		t.Fatalf("summary config still has incompatible objectives: %v", err)
	}
	for _, want := range []string{"matryoshka", "turboquant_prefix_bits", "turboquant_prefix_objectives", "turboquant_prefix_weight", "turboquant_prefix_seed", "turboquant_prefix_score_mode", "turboquant_rank_margin_objectives", "turboquant_rank_margin"} {
		if !hasScoreSpectrumObjectiveName(trainer.scoreSpectrumLineage.AutoClearedObjectives, want) {
			t.Fatalf("auto-cleared objectives = %v, missing %q", trainer.scoreSpectrumLineage.AutoClearedObjectives, want)
		}
	}
}

func hasScoreSpectrumObjectiveName(values []string, want string) bool {
	for _, value := range values {
		if value == want {
			return true
		}
	}
	return false
}

func TestEstimateScoreSpectrumTrainWorkloadCountsRowLocalCandidates(t *testing.T) {
	workload := EstimateScoreSpectrumTrainWorkload(tinyEmbeddingScoreSpectrumDataset(), 3, EmbeddingTrainRunConfig{
		Epochs:    2,
		BatchSize: 2,
	})
	if workload.TrainMode != "score_spectrum_grouped" {
		t.Fatalf("train mode = %q", workload.TrainMode)
	}
	if workload.TrainPairsPerEpoch != 4 || workload.PlannedTrainPairs != 8 {
		t.Fatalf("train pairs per epoch/planned = %d/%d, want 4/8", workload.TrainPairsPerEpoch, workload.PlannedTrainPairs)
	}
}

func TestEstimateScoreSpectrumTrainWorkloadCountsAuxOnlyRows(t *testing.T) {
	trainSet := tinyEmbeddingScoreSpectrumDataset()
	trainSet[1].BaseLossWeight = float32Ptr(0)
	workload := EstimateScoreSpectrumTrainWorkload(trainSet, 0, EmbeddingTrainRunConfig{
		Epochs:                     1,
		BatchSize:                  2,
		TurboQuantTopKObjectives:   []TurboQuantPrefixObjective{{Dim: 3, BitWidth: 2, Weight: 0.04}},
		TurboQuantTopKLoss:         TurboQuantTopKLossLambdaNDCG,
		TurboQuantTopKNegativeMask: TurboQuantTopKNegativeMaskHard,
	})
	if workload.ScoreSpectrumAuxOnlyRows != 1 {
		t.Fatalf("aux-only rows = %d, want 1", workload.ScoreSpectrumAuxOnlyRows)
	}
}

func TestScoreSpectrumActivationMicrobatchRunConfig(t *testing.T) {
	if err := validateScoreSpectrumRunConfig(EmbeddingTrainRunConfig{ScoreSpectrumActivationMicrobatchSize: 8, ScoreSpectrumRecoveryTopK: 1, ScoreSpectrumRecoveryTau: 1}); err != nil {
		t.Fatalf("positive activation microbatch config rejected: %v", err)
	}
	if err := validateScoreSpectrumRunConfig(EmbeddingTrainRunConfig{ScoreSpectrumActivationMicrobatchSize: -1}); err == nil || !strings.Contains(err.Error(), "score_spectrum_activation_microbatch_size") {
		t.Fatalf("negative activation microbatch config error = %v, want non-negative rejection", err)
	}
	got := normalizedScoreSpectrumRunConfig(EmbeddingTrainRunConfig{ScoreSpectrumActivationMicrobatchSize: 8})
	if got.ScoreSpectrumActivationMicrobatchSize != 8 {
		t.Fatalf("normalized activation microbatch size = %d, want 8", got.ScoreSpectrumActivationMicrobatchSize)
	}
}

func TestEstimateScoreSpectrumTrainWorkloadMixedEvalModeUsesActualEvalSets(t *testing.T) {
	workload := estimateScoreSpectrumTrainWorkload(tinyEmbeddingScoreSpectrumDataset(), 2, 1, 4, EmbeddingTrainRunConfig{
		Epochs:    1,
		BatchSize: 2,
	})
	if workload.EvalMode != "mixed" {
		t.Fatalf("eval mode = %q, want mixed", workload.EvalMode)
	}

	workload = estimateScoreSpectrumTrainWorkload(tinyEmbeddingScoreSpectrumDataset(), 0, 1, 2, EmbeddingTrainRunConfig{
		Epochs:    1,
		BatchSize: 2,
	})
	if workload.EvalMode != "score_spectrum_grouped" {
		t.Fatalf("eval mode = %q, want score_spectrum_grouped", workload.EvalMode)
	}
}

func tinyEmbeddingScoreSpectrumDataset() []EmbeddingScoreSpectrumExample {
	return []EmbeddingScoreSpectrumExample{
		{
			QueryTokens:             []int32{0},
			QueryMask:               []int32{1},
			CandidateTokens:         [][]int32{{0}, {1}},
			CandidateMasks:          [][]int32{{1}, {1}},
			PositiveIndexes:         []int{0},
			HardNegativeEligible:    []bool{false, true},
			TargetProbabilities:     []float32{1, 0},
			CommercialUseAllowed:    true,
			TrainAllowedForResearch: false,
		},
		{
			QueryTokens:             []int32{1},
			QueryMask:               []int32{1},
			CandidateTokens:         [][]int32{{1}, {0}},
			CandidateMasks:          [][]int32{{1}, {1}},
			PositiveIndexes:         []int{0},
			HardNegativeEligible:    []bool{false, true},
			TargetProbabilities:     []float32{1, 0},
			CommercialUseAllowed:    true,
			TrainAllowedForResearch: false,
		},
	}
}

func tinyResearchOnlyEmbeddingScoreSpectrumDataset() []EmbeddingScoreSpectrumExample {
	out := tinyEmbeddingScoreSpectrumDataset()
	for i := range out {
		out[i].ReleaseTrainAllowed = false
		out[i].CommercialUseAllowed = false
		out[i].TrainAllowedForResearch = true
	}
	return out
}
