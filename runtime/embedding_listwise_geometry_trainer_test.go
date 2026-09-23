package eosruntime

import (
	"bytes"
	"context"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"m31labs.dev/eos/runtime/backends/cuda"
	"m31labs.dev/eos/runtime/backends/metal"
)

func TestTokenizeEmbeddingListwiseGeometryBatchesPreservesMatrixAlignment(t *testing.T) {
	tokenizer := tinyListwiseGeometryTokenizer(t)
	out, err := TokenizeEmbeddingListwiseGeometryBatches(tinyListwiseGeometryBatches(false), tokenizer)
	if err != nil {
		t.Fatalf("tokenize listwise geometry: %v", err)
	}
	if len(out) != 1 {
		t.Fatalf("tokenized batches = %d, want 1", len(out))
	}
	got := out[0]
	if len(got.QueryTokens) != 2 || len(got.DocumentTokens) != 2 {
		t.Fatalf("tokenized shape = %dx%d, want 2x2", len(got.QueryTokens), len(got.DocumentTokens))
	}
	if got.QueryIDs[0] != "q0" || got.QueryIDs[1] != "q1" || got.DocumentIDs[0] != "d0" || got.DocumentIDs[1] != "d1" {
		t.Fatalf("ids = q%v d%v, want original order", got.QueryIDs, got.DocumentIDs)
	}
	if got.TeacherSimilarity[0][0] != 0.9 || got.TeacherSimilarity[0][1] != 0.1 || got.TeacherSimilarity[1][0] != 0.2 || got.TeacherSimilarity[1][1] != 0.8 {
		t.Fatalf("teacher matrix = %+v, want original alignment", got.TeacherSimilarity)
	}
}

func assertFloat32SlicesClose(t *testing.T, got, want []float32, tol float64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("length = %d, want %d; got=%v want=%v", len(got), len(want), got, want)
	}
	for i := range got {
		if math.Abs(float64(got[i]-want[i])) > tol {
			t.Fatalf("value[%d] = %f, want %f within %g; got=%v want=%v", i, got[i], want[i], tol, got, want)
		}
	}
}

func TestEmbeddingTrainerTrainListwiseGeometryStep(t *testing.T) {
	trainer := newTinyTrainable3DEmbeddingTrainer(t, 0.05)
	trainer.config.Temperature = 0.05

	metrics, err := trainer.TrainListwiseGeometryStep(tinyTokenizedListwiseGeometryBatches(false))
	if err != nil {
		t.Fatalf("train listwise geometry step: %v", err)
	}
	if metrics.BatchSize != 4 {
		t.Fatalf("batch size = %d, want 4 query-document scores", metrics.BatchSize)
	}
	if metrics.Loss < 0 || math.IsNaN(float64(metrics.Loss)) || math.IsInf(float64(metrics.Loss), 0) {
		t.Fatalf("loss = %f, want finite non-negative", metrics.Loss)
	}
	if trainer.step != 1 {
		t.Fatalf("step = %d, want 1", trainer.step)
	}
	if metrics.Movement != nil {
		t.Fatalf("movement metrics = %+v, want nil by default", metrics.Movement)
	}

	diagnosticTrainer := newTinyTrainable3DEmbeddingTrainer(t, 0.05)
	diagnosticTrainer.config.Temperature = 0.05
	diagnosticMetrics, err := diagnosticTrainer.TrainListwiseGeometryStepWithDiagnostics(tinyTokenizedListwiseGeometryBatches(false), true)
	if err != nil {
		t.Fatalf("train diagnostic listwise geometry step: %v", err)
	}
	if diagnosticMetrics.Movement == nil {
		t.Fatalf("diagnostic movement metrics missing")
	}
	if diagnosticMetrics.Movement.Gradient.L2Norm <= 0 || diagnosticMetrics.Movement.Gradient.MaxAbs <= 0 || diagnosticMetrics.Movement.Gradient.NonzeroCount <= 0 || diagnosticMetrics.Movement.Gradient.TotalCount <= 0 {
		t.Fatalf("gradient movement = %+v, want nonzero aggregate", diagnosticMetrics.Movement.Gradient)
	}
	if diagnosticMetrics.Movement.ParameterDelta.L2Norm <= 0 || diagnosticMetrics.Movement.ParameterDelta.MaxAbs <= 0 || diagnosticMetrics.Movement.ParameterDelta.NonzeroCount <= 0 || diagnosticMetrics.Movement.ParameterDelta.TotalCount <= 0 {
		t.Fatalf("parameter delta movement = %+v, want nonzero aggregate", diagnosticMetrics.Movement.ParameterDelta)
	}
}

func TestEmbeddingTrainerEvaluateListwiseGeometryMetricsAndCELowerWhenStudentMatchesTeacher(t *testing.T) {
	batch := EmbeddingTokenizedListwiseGeometryBatch{
		QueryIDs:       []string{"q0", "q1"},
		QueryTokens:    [][]int32{{0}, {1}},
		DocumentIDs:    []string{"d0", "d1"},
		DocumentTokens: [][]int32{{0}, {1}},
		TeacherSimilarity: [][]float32{
			{1, 0},
			{0, 1},
		},
		Examples: []EmbeddingListwiseGeometryExample{
			{QueryID: "q0", PositiveDocID: "d0"},
			{QueryID: "q1", PositiveDocID: "d1"},
		},
	}
	matchedQueries := []*embeddingEncodedSequence{
		{pooled: []float32{1, 0}},
		{pooled: []float32{0, 1}},
	}
	matchedDocs := []*embeddingEncodedSequence{
		{pooled: []float32{1, 0}},
		{pooled: []float32{0, 1}},
	}
	matched, err := evaluateListwiseGeometryEncodings(matchedQueries, matchedDocs, []embeddingCandidateSpan{{Start: 0, End: 2}}, []EmbeddingTokenizedListwiseGeometryBatch{batch}, 0.2)
	if err != nil {
		t.Fatalf("matched listwise eval: %v", err)
	}
	swappedDocs := []*embeddingEncodedSequence{
		{pooled: []float32{0, 1}},
		{pooled: []float32{1, 0}},
	}
	swapped, err := evaluateListwiseGeometryEncodings(matchedQueries, swappedDocs, []embeddingCandidateSpan{{Start: 0, End: 2}}, []EmbeddingTokenizedListwiseGeometryBatch{batch}, 0.2)
	if err != nil {
		t.Fatalf("swapped listwise eval: %v", err)
	}
	if matched.TeacherCrossEntropy >= swapped.TeacherCrossEntropy {
		t.Fatalf("matched CE = %f, swapped CE = %f; want matched lower", matched.TeacherCrossEntropy, swapped.TeacherCrossEntropy)
	}
	if matched.Loss != matched.TeacherCrossEntropy {
		t.Fatalf("loss = %f, teacher CE = %f; want aliases equal", matched.Loss, matched.TeacherCrossEntropy)
	}
	if matched.QueryCount != 2 || matched.DocumentCellCount != 4 || matched.BatchCount != 1 {
		t.Fatalf("matched counts = %+v, want 2 queries/4 cells/1 batch", matched)
	}
	if matched.TeacherTop1Agreement != 1 || matched.AnyPositiveTop1 != 1 || matched.AnyPositiveQueryCount != 2 {
		t.Fatalf("matched top1 metrics = %+v, want perfect teacher/positive top1", matched)
	}
	if swapped.TeacherTop1Agreement != 0 || swapped.AnyPositiveTop1 != 0 {
		t.Fatalf("swapped top1 metrics = %+v, want zero teacher/positive top1", swapped)
	}
	if matched.TeacherKL < -1e-6 || math.IsNaN(float64(matched.TeacherKL)) {
		t.Fatalf("teacher KL = %f, want finite non-negative", matched.TeacherKL)
	}
}

func TestEmbeddingTrainerFitListwiseGeometryEvalOnlyRecordsMetricsAndDoesNotUpdate(t *testing.T) {
	trainer := newTinyTrainable3DEmbeddingTrainer(t, 0.05)
	startProfile := trainer.TrainProfile()
	startStep := trainer.step
	summary, err := trainer.FitListwiseGeometry(nil, nil, EmbeddingTrainRunConfig{
		EvalOnly:             true,
		BatchSize:            1,
		Temperature:          0.05,
		ListwiseGeometryEval: tinyTokenizedListwiseGeometryBatches(false),
	})
	if err != nil {
		t.Fatalf("fit listwise geometry eval-only: %v", err)
	}
	if trainer.step != startStep || summary.StepsRun != 0 || summary.StepsCompleted != startStep {
		t.Fatalf("steps trainer=%d summary=%+v, want no optimizer steps", trainer.step, summary)
	}
	if summary.DeltaProfile.Optimizer.UpdateCalls != 0 {
		t.Fatalf("optimizer updates = %d, want 0", summary.DeltaProfile.Optimizer.UpdateCalls)
	}
	if trainer.TrainProfile().Optimizer.UpdateCalls != startProfile.Optimizer.UpdateCalls {
		t.Fatalf("profile optimizer updates = %d, want unchanged %d", trainer.TrainProfile().Optimizer.UpdateCalls, startProfile.Optimizer.UpdateCalls)
	}
	if summary.FinalListwiseGeometryEval == nil || summary.LastListwiseGeometryEval == nil || summary.BestListwiseGeometryEval == nil {
		t.Fatalf("missing listwise eval metrics: %+v", summary)
	}
	if summary.FinalEval != nil {
		t.Fatalf("pairwise eval = %+v, want nil for listwise-only eval", summary.FinalEval)
	}
	if summary.FinalTrain.Movement != nil {
		t.Fatalf("final train movement = %+v, want nil for eval-only", summary.FinalTrain.Movement)
	}
	if summary.Workload.EvalMode != "listwise_geometry" || summary.Workload.ActualEvalPasses != 1 || summary.Workload.ActualEvalExamples != 2 || summary.Workload.ActualEvalPairs != 4 || summary.Workload.ActualTrainPairs != 0 {
		t.Fatalf("workload = %+v, want one listwise eval pass over 2 query rows/4 cells and no train pairs", summary.Workload)
	}
}

func TestEmbeddingTrainerFitListwiseGeometryRejectsWorkloadLimit(t *testing.T) {
	_, err := newTinyTrainable3DEmbeddingTrainer(t, 0.05).FitListwiseGeometry(tinyTokenizedListwiseGeometryBatches(false), nil, EmbeddingTrainRunConfig{
		Epochs:                        1,
		BatchSize:                     1,
		Temperature:                   0.05,
		MaxListwiseGeometryTrainPairs: 3,
	})
	if err == nil || !strings.Contains(err.Error(), "train_pairs/epoch 4 exceeds max_listwise_geometry_train_pairs 3") {
		t.Fatalf("train workload limit error = %v, want train pair cap rejection", err)
	}

	_, err = newTinyTrainable3DEmbeddingTrainer(t, 0.05).FitListwiseGeometry(nil, nil, EmbeddingTrainRunConfig{
		EvalOnly:                     true,
		BatchSize:                    1,
		Temperature:                  0.05,
		ListwiseGeometryEval:         tinyTokenizedListwiseGeometryBatches(false),
		MaxListwiseGeometryEvalPairs: 3,
	})
	if err == nil || !strings.Contains(err.Error(), "eval_pairs/pass 4 exceeds max_listwise_geometry_eval_pairs 3") {
		t.Fatalf("eval workload limit error = %v, want eval pair cap rejection", err)
	}
}

func TestEmbeddingTrainerFitListwiseGeometryReportsPreStepProgress(t *testing.T) {
	trainer := newTinyTrainable3DEmbeddingTrainer(t, 0.05)
	var reports []EmbeddingTrainProgress
	summary, err := trainer.FitListwiseGeometry(tinyTokenizedListwiseGeometryBatches(false), nil, EmbeddingTrainRunConfig{
		Epochs:             1,
		BatchSize:          1,
		Temperature:        0.05,
		ProgressEverySteps: 1,
		Progress: func(progress EmbeddingTrainProgress) {
			reports = append(reports, progress)
		},
	})
	if err != nil {
		t.Fatalf("fit listwise geometry: %v", err)
	}
	if summary.StepsRun != 1 {
		t.Fatalf("steps run = %d, want 1", summary.StepsRun)
	}
	if len(reports) != 2 {
		t.Fatalf("progress reports = %d, want train_start and train", len(reports))
	}
	if reports[0].Phase != "train_start" || reports[0].Batch != 1 || reports[0].Batches != 1 || reports[0].Step != 0 || reports[0].BatchPairs != 4 {
		t.Fatalf("pre-step progress = %+v, want train_start for first 4-pair batch before optimizer step", reports[0])
	}
	if reports[1].Phase != "train" || reports[1].Step != 1 || reports[1].BatchPairs != 4 || reports[1].EpochTrainPairs != 4 {
		t.Fatalf("post-step progress = %+v, want completed first 4-pair batch", reports[1])
	}
}

func TestEmbeddingTrainerFitListwiseGeometrySmokeAndResearchGate(t *testing.T) {
	_, err := newTinyTrainable3DEmbeddingTrainer(t, 0.05).FitListwiseGeometry(tinyTokenizedListwiseGeometryBatches(true), nil, EmbeddingTrainRunConfig{
		Epochs:         1,
		BatchSize:      1,
		EvalEveryEpoch: 1,
		Temperature:    0.05,
	})
	if err == nil || !strings.Contains(err.Error(), "research-only") {
		t.Fatalf("research-only error = %v, want gate rejection", err)
	}

	summary, err := newTinyTrainable3DEmbeddingTrainer(t, 0.05).FitListwiseGeometry(tinyTokenizedListwiseGeometryBatches(true), nil, EmbeddingTrainRunConfig{
		Epochs:                            1,
		BatchSize:                         1,
		EvalEveryEpoch:                    1,
		Temperature:                       0.05,
		AllowResearchOnlyListwiseGeometry: true,
	})
	if err != nil {
		t.Fatalf("fit listwise geometry: %v", err)
	}
	if summary.StepsCompleted != 1 || summary.Workload.TrainMode != "listwise_geometry" || summary.Workload.ActualTrainPairs != 4 {
		t.Fatalf("summary = %+v workload=%+v, want one step and 4 train pairs", summary, summary.Workload)
	}
	if summary.EffectiveLearningRate != 0.05 {
		t.Fatalf("effective learning rate = %g, want 0.05", summary.EffectiveLearningRate)
	}
	if summary.FinalTrain.Movement != nil {
		t.Fatalf("final train movement = %+v, want nil by default", summary.FinalTrain.Movement)
	}

	diagnosticSummary, err := newTinyTrainable3DEmbeddingTrainer(t, 0.05).FitListwiseGeometry(tinyTokenizedListwiseGeometryBatches(true), nil, EmbeddingTrainRunConfig{
		Epochs:                            1,
		BatchSize:                         1,
		EvalEveryEpoch:                    1,
		Temperature:                       0.05,
		AllowResearchOnlyListwiseGeometry: true,
		MovementDiagnostics:               true,
	})
	if err != nil {
		t.Fatalf("fit diagnostic listwise geometry: %v", err)
	}
	if diagnosticSummary.FinalTrain.Movement == nil || diagnosticSummary.FinalTrain.Movement.Gradient.NonzeroCount <= 0 || diagnosticSummary.FinalTrain.Movement.ParameterDelta.NonzeroCount <= 0 {
		t.Fatalf("final train movement = %+v, want nonzero gradient and parameter delta counts", diagnosticSummary.FinalTrain.Movement)
	}
	if !diagnosticSummary.Config.MovementDiagnostics {
		t.Fatalf("movement diagnostics config = false, want true")
	}

	_, err = newTinyTrainable3DEmbeddingTrainer(t, 0.05).FitListwiseGeometry(tinyTokenizedListwiseGeometryBatches(false), nil, EmbeddingTrainRunConfig{
		Epochs:                            1,
		BatchSize:                         1,
		EvalEveryEpoch:                    1,
		Temperature:                       0.05,
		AllowResearchOnlyListwiseGeometry: true,
	})
	if err == nil || !strings.Contains(err.Error(), "must be explicitly research-only") {
		t.Fatalf("allow flag with non-research row error = %v, want strict rejection", err)
	}
}

func TestEmbeddingTrainerFitListwiseGeometryIsolatesInheritedCompactPackageObjectives(t *testing.T) {
	trainer := newTinyTrainable3DEmbeddingTrainer(t, 0.05)
	trainer.config.MatryoshkaDims = []int{2}
	trainer.config.MatryoshkaWeights = []float32{1}
	trainer.config.TurboQuantCompactObjectives = []TurboQuantPrefixObjective{{Dim: 2, BitWidth: 2, Weight: 0.25}}
	trainer.config.TurboQuantPrefixSeed = 11

	summary, err := trainer.FitListwiseGeometry(tinyTokenizedListwiseGeometryBatches(false), nil, EmbeddingTrainRunConfig{
		Epochs:         1,
		BatchSize:      1,
		EvalEveryEpoch: 1,
		Temperature:    0.05,
	})
	if err != nil {
		t.Fatalf("fit listwise geometry with inherited compact objectives: %v", err)
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

	artifactPath := filepath.Join(t.TempDir(), "tiny_train_embed_q8.mll")
	paths, err := trainer.WriteTrainingPackage(artifactPath)
	if err != nil {
		t.Fatalf("write isolated package: %v", err)
	}
	trainManifest, err := ReadEmbeddingTrainManifestFile(paths.TrainManifestPath)
	if err != nil {
		t.Fatalf("read train manifest: %v", err)
	}
	packageManifest, err := ReadPackageManifestFile(paths.PackageManifestPath)
	if err != nil {
		t.Fatalf("read package manifest: %v", err)
	}
	for _, want := range []string{"matryoshka", "turboquant_compact_objectives", "turboquant_prefix_seed"} {
		if !hasScoreSpectrumObjectiveName(trainManifest.ListwiseGeometry.AutoClearedObjectives, want) {
			t.Fatalf("train manifest listwise auto-cleared objectives = %v, missing %q", trainManifest.ListwiseGeometry.AutoClearedObjectives, want)
		}
		if !hasScoreSpectrumObjectiveName(packageManifest.ListwiseGeometry.IsolatedInheritedObjectives, want) {
			t.Fatalf("package manifest listwise isolated objectives = %v, missing %q", packageManifest.ListwiseGeometry.IsolatedInheritedObjectives, want)
		}
	}
}

func TestEmbeddingTrainerFitListwiseGeometryExplicitFullDimCompactOverridesInheritedPackageConfig(t *testing.T) {
	seed := int64(17)
	path := filepath.Join(t.TempDir(), "inherited-listwise-compact.mll")
	source := newTinyTrainable3DEmbeddingTrainer(t, 0.05)
	source.config.MatryoshkaDims = []int{2}
	source.config.MatryoshkaWeights = []float32{1}
	source.config.TurboQuantPrefixBits = []int{2}
	source.config.TurboQuantPrefixWeight = 0.25
	source.config.TurboQuantRankMarginObjectives = []TurboQuantPrefixObjective{{Dim: 2, BitWidth: 2, Weight: 0.5}}
	source.config.TurboQuantRankMargin = 0.02
	source.config.TurboQuantCompactObjectives = []TurboQuantPrefixObjective{{Dim: 2, BitWidth: 2, Weight: 0.25}}
	source.config.TurboQuantPrefixSeed = 11
	if _, err := source.WriteTrainingPackage(path); err != nil {
		t.Fatalf("write inherited package: %v", err)
	}

	limited, err := LoadEmbeddingTrainerPackage(path)
	if err != nil {
		t.Fatalf("load inherited package for cap check: %v", err)
	}
	_, err = limited.FitListwiseGeometry(tinyTokenizedListwiseGeometryBatches(false), nil, EmbeddingTrainRunConfig{
		Epochs:                        1,
		BatchSize:                     1,
		EvalEveryEpoch:                1,
		Temperature:                   0.05,
		TurboQuantCompactObjectives:   []TurboQuantPrefixObjective{{Dim: 3, BitWidth: 3, Weight: 0.5}},
		TurboQuantPrefixSeed:          seed,
		MaxListwiseGeometryTrainPairs: 7,
	})
	if err == nil || !strings.Contains(err.Error(), "max_listwise_geometry_train_pairs 7") {
		t.Fatalf("cap error = %v, want dense+compact train pair cap rejection", err)
	}

	denseSummary, err := newTinyTrainable3DEmbeddingTrainer(t, 0.05).FitListwiseGeometry(tinyTokenizedListwiseGeometryBatches(false), nil, EmbeddingTrainRunConfig{
		Epochs:         1,
		BatchSize:      1,
		EvalEveryEpoch: 1,
		Temperature:    0.05,
	})
	if err != nil {
		t.Fatalf("fit dense listwise geometry: %v", err)
	}
	trainer, err := LoadEmbeddingTrainerPackage(path)
	if err != nil {
		t.Fatalf("load inherited package: %v", err)
	}
	summary, err := trainer.FitListwiseGeometry(tinyTokenizedListwiseGeometryBatches(false), nil, EmbeddingTrainRunConfig{
		Epochs:                        1,
		BatchSize:                     1,
		EvalEveryEpoch:                1,
		Temperature:                   0.05,
		TurboQuantCompactObjectives:   []TurboQuantPrefixObjective{{Dim: 3, BitWidth: 3, Weight: 0.5}},
		TurboQuantPrefixSeed:          seed,
		MaxListwiseGeometryTrainPairs: 8,
	})
	if err != nil {
		t.Fatalf("fit explicit full-dim compact listwise geometry: %v", err)
	}
	if math.IsNaN(float64(summary.FinalTrain.Loss)) || math.IsInf(float64(summary.FinalTrain.Loss), 0) || summary.FinalTrain.Loss <= 0 {
		t.Fatalf("compact final train loss = %f, want finite positive", summary.FinalTrain.Loss)
	}
	if math.Abs(float64(summary.FinalTrain.Loss-denseSummary.FinalTrain.Loss)) < 1e-7 {
		t.Fatalf("compact loss = %f, dense loss = %f, want distinct objective result", summary.FinalTrain.Loss, denseSummary.FinalTrain.Loss)
	}
	if summary.FinalTrain.BatchSize != 8 || summary.Workload.TrainPairsPerEpoch != 8 || summary.Workload.PlannedTrainPairs != 8 || summary.Workload.ActualTrainPairs != 8 || summary.Workload.PlannedTotalPairs != 8 || summary.Workload.ActualTotalPairs != 8 {
		t.Fatalf("workload/final train = %+v final=%+v, want exactly dense+one compact objective (8 pairs)", summary.Workload, summary.FinalTrain)
	}
	if got := summary.Config.TurboQuantCompactObjectives; len(got) != 1 || got[0].Dim != 3 || got[0].BitWidth != 3 || got[0].Weight != 0.5 {
		t.Fatalf("summary compact objectives = %+v, want explicit full-dim q3", got)
	}
	if got := trainer.config.TurboQuantCompactObjectives; len(got) != 1 || got[0].Dim != 3 || got[0].BitWidth != 3 || got[0].Weight != 0.5 {
		t.Fatalf("trainer compact objectives = %+v, want explicit full-dim q3", got)
	}
	if len(trainer.config.MatryoshkaDims) != 0 || len(trainer.config.TurboQuantPrefixBits) != 0 || len(trainer.config.TurboQuantRankMarginObjectives) != 0 {
		t.Fatalf("inherited incompatible trainer objectives not cleared: %+v", trainer.config)
	}
	for _, want := range []string{"matryoshka", "turboquant_prefix_bits", "turboquant_rank_margin_objectives", "turboquant_compact_objectives", "turboquant_prefix_seed"} {
		if !hasScoreSpectrumObjectiveName(trainer.listwiseGeometryLineage.IsolatedInheritedObjectives, want) {
			t.Fatalf("isolated inherited objectives = %v, missing %q", trainer.listwiseGeometryLineage.IsolatedInheritedObjectives, want)
		}
	}
}

func TestEmbeddingTrainerFitListwiseGeometryRejectsExplicitIncompatibleObjectivesBeforeIsolation(t *testing.T) {
	trainer := newTinyTrainable3DEmbeddingTrainer(t, 0.05)
	trainer.config.TurboQuantCompactObjectives = []TurboQuantPrefixObjective{{Dim: 2, BitWidth: 2, Weight: 0.25}}

	_, err := trainer.FitListwiseGeometry(tinyTokenizedListwiseGeometryBatches(false), nil, EmbeddingTrainRunConfig{
		Epochs:                      1,
		BatchSize:                   1,
		EvalEveryEpoch:              1,
		Temperature:                 0.05,
		MatryoshkaDims:              []int{2},
		TurboQuantCompactObjectives: []TurboQuantPrefixObjective{{Dim: 2, BitWidth: 2, Weight: 0.25}},
	})
	if err == nil || !strings.Contains(err.Error(), "listwise geometry training does not support matryoshka objectives") {
		t.Fatalf("error = %v, want explicit listwise matryoshka rejection", err)
	}
	if len(trainer.config.TurboQuantCompactObjectives) == 0 {
		t.Fatalf("inherited objectives were cleared after explicit run-config rejection")
	}
}

func TestEmbeddingTrainerFitListwiseGeometryPairwiseEvalSelectionAndAccounting(t *testing.T) {
	summary, err := newTinyTrainable3DEmbeddingTrainer(t, 0.05).FitListwiseGeometry(tinyTokenizedListwiseGeometryBatches(false), tinyEncoderPairDataset(), EmbeddingTrainRunConfig{
		Epochs:                2,
		BatchSize:             1,
		EvalEveryEpoch:        1,
		EvalEverySteps:        1,
		EarlyStoppingPatience: 3,
		RestoreBest:           true,
		SelectMetric:          "loss",
		Temperature:           0.05,
	})
	if err != nil {
		t.Fatalf("fit listwise geometry with eval: %v", err)
	}
	if summary.FinalEval == nil || summary.BestEval == nil || len(summary.History) == 0 {
		t.Fatalf("summary missing eval/history: %+v", summary)
	}
	if len(summary.History) != 2 {
		t.Fatalf("history len = %d, want one epoch record per epoch", len(summary.History))
	}
	if summary.Workload.ActualEvalPasses != 6 || summary.Workload.ActualEvalPairs != int64(6*len(tinyEncoderPairDataset())) || summary.Workload.ActualEvalExamples != int64(6*len(tinyEncoderPairDataset())) {
		t.Fatalf("eval accounting = passes %d pairs %d examples %d, want 6/%d/%d", summary.Workload.ActualEvalPasses, summary.Workload.ActualEvalPairs, summary.Workload.ActualEvalExamples, 6*len(tinyEncoderPairDataset()), 6*len(tinyEncoderPairDataset()))
	}
	if len(summary.EvalHistory) != summary.Workload.ActualEvalPasses {
		t.Fatalf("eval history len = %d, want actual eval passes %d", len(summary.EvalHistory), summary.Workload.ActualEvalPasses)
	}
	triggerCounts := map[string]int{}
	for i, record := range summary.EvalHistory {
		if record.EvalPass != i+1 {
			t.Fatalf("eval history[%d] pass = %d, want %d", i, record.EvalPass, i+1)
		}
		if record.Eval == nil {
			t.Fatalf("eval history[%d] missing pairwise eval: %+v", i, record)
		}
		triggerCounts[record.Trigger]++
	}
	if triggerCounts["initial"] != 1 || triggerCounts["step"] != 2 || triggerCounts["epoch"] != 2 || triggerCounts["final"] != 1 {
		t.Fatalf("eval history trigger counts = %+v, want initial=1 step=2 epoch=2 final=1", triggerCounts)
	}
	if summary.Workload.PlannedEvalPasses != 6 || summary.Workload.PlannedEvalPairs != int64(6*len(tinyEncoderPairDataset())) {
		t.Fatalf("planned eval accounting = passes %d pairs %d, want 6/%d", summary.Workload.PlannedEvalPasses, summary.Workload.PlannedEvalPairs, 6*len(tinyEncoderPairDataset()))
	}
	if !summary.RestoredBest || summary.BestEval == nil {
		t.Fatalf("restored/best = %v/%v, want restore-best selection", summary.RestoredBest, summary.BestEval)
	}
	for _, record := range summary.History {
		if record.Eval == nil {
			t.Fatalf("history record missing eval: %+v", record)
		}
	}
}

func TestEmbeddingTrainerFitListwiseGeometryRetrievalOnlyEvalSelectionAndHistory(t *testing.T) {
	dataset := writeTinyRetrievalGateFixture(t)
	corpusPath, queriesPath, qrelsPath := BEIRRetrievalPaths(dataset, "test")
	tok := tinyEmbeddingTokenizerFile()
	summary, err := newTinyTrainable3DEmbeddingTrainer(t, 0.05).FitListwiseGeometry(tinyTokenizedListwiseGeometryBatches(false), nil, EmbeddingTrainRunConfig{
		Epochs:               2,
		BatchSize:            1,
		EvalEveryEpoch:       1,
		EvalEverySteps:       1,
		RestoreBest:          true,
		SelectMetric:         "retrieval_recall",
		Temperature:          0.05,
		ListwiseGeometryEval: tinyTokenizedListwiseGeometryBatches(false),
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
		t.Fatalf("fit listwise geometry with retrieval-only eval: %v", err)
	}
	if !summary.RestoredBest {
		t.Fatal("expected restore-best to run from retrieval-only selection")
	}
	if summary.BestEval == nil || summary.BestEval.RetrievalRecallAt100 <= 0 {
		t.Fatalf("best eval retrieval recall = %+v, want > 0", summary.BestEval)
	}
	if summary.FinalEval == nil || summary.FinalEval.RetrievalRecallAt100 <= 0 || summary.FinalEval.PairCount != 0 {
		t.Fatalf("final eval = %+v, want retrieval-only metrics with pair_count=0", summary.FinalEval)
	}
	if len(summary.History) != 2 {
		t.Fatalf("history len = %d, want 2 epochs", len(summary.History))
	}
	for _, record := range summary.History {
		if record.Eval == nil || record.Eval.RetrievalRecallAt100 <= 0 || record.Eval.PairCount != 0 {
			t.Fatalf("epoch %d eval = %+v, want retrieval-only metrics", record.Epoch, record.Eval)
		}
		if record.ListwiseGeometryEval == nil {
			t.Fatalf("epoch %d missing listwise eval metrics", record.Epoch)
		}
	}
	if len(summary.EvalHistory) != summary.Workload.ActualEvalPasses {
		t.Fatalf("eval history len = %d, want actual eval passes %d", len(summary.EvalHistory), summary.Workload.ActualEvalPasses)
	}
	triggerCounts := map[string]int{}
	for i, record := range summary.EvalHistory {
		if record.EvalPass != i+1 {
			t.Fatalf("eval history[%d] pass = %d, want %d", i, record.EvalPass, i+1)
		}
		if record.Eval == nil || record.Eval.RetrievalRecallAt100 <= 0 || record.Eval.PairCount != 0 {
			t.Fatalf("eval history[%d] eval = %+v, want retrieval-only metrics", i, record.Eval)
		}
		triggerCounts[record.Trigger]++
	}
	if triggerCounts["initial"] != 1 || triggerCounts["step"] != 2 || triggerCounts["epoch"] != 2 || triggerCounts["final"] != 1 {
		t.Fatalf("eval history trigger counts = %+v, want initial=1 step=2 epoch=2 final=1", triggerCounts)
	}
}

func TestListwiseGeometryGradientScaleIsQueryRowAveragedAcrossPackedRows(t *testing.T) {
	queries := []*embeddingEncodedSequence{
		{pooled: []float32{1, 0}},
		{pooled: []float32{0, 1}},
	}
	combinedDocs := []*embeddingEncodedSequence{
		{pooled: []float32{1, 0}},
		{pooled: []float32{0, 1}},
	}
	combinedBatch := []EmbeddingTokenizedListwiseGeometryBatch{{
		QueryTokens:       [][]int32{{0}, {1}},
		DocumentTokens:    [][]int32{{0}, {1}},
		TeacherSimilarity: [][]float32{{0.9, 0.1}, {0.2, 0.8}},
	}}
	combinedQueryGrads := [][]float32{{0, 0}, {0, 0}}
	combinedDocGrads := [][]float32{{0, 0}, {0, 0}}
	combinedLoss, _, _, combinedQueries, err := accumulateListwiseGeometryGrads(queries, combinedDocs, []embeddingCandidateSpan{{Start: 0, End: 2}}, combinedBatch, EmbeddingTrainConfig{Temperature: 0.5}, combinedQueryGrads, combinedDocGrads)
	if err != nil {
		t.Fatalf("combined grads: %v", err)
	}

	splitDocs := []*embeddingEncodedSequence{
		{pooled: []float32{1, 0}},
		{pooled: []float32{0, 1}},
		{pooled: []float32{1, 0}},
		{pooled: []float32{0, 1}},
	}
	splitBatch := []EmbeddingTokenizedListwiseGeometryBatch{
		{QueryTokens: [][]int32{{0}}, DocumentTokens: [][]int32{{0}, {1}}, TeacherSimilarity: [][]float32{{0.9, 0.1}}},
		{QueryTokens: [][]int32{{1}}, DocumentTokens: [][]int32{{0}, {1}}, TeacherSimilarity: [][]float32{{0.2, 0.8}}},
	}
	splitQueryGrads := [][]float32{{0, 0}, {0, 0}}
	splitDocGrads := [][]float32{{0, 0}, {0, 0}, {0, 0}, {0, 0}}
	splitLoss, _, _, splitQueries, err := accumulateListwiseGeometryGrads(queries, splitDocs, []embeddingCandidateSpan{{Start: 0, End: 2}, {Start: 2, End: 4}}, splitBatch, EmbeddingTrainConfig{Temperature: 0.5}, splitQueryGrads, splitDocGrads)
	if err != nil {
		t.Fatalf("split grads: %v", err)
	}

	if combinedQueries != 2 || splitQueries != 2 {
		t.Fatalf("query counts = %d/%d, want 2/2", combinedQueries, splitQueries)
	}
	if math.Abs(float64(combinedLoss/float32(combinedQueries)-splitLoss/float32(splitQueries))) > 1e-6 {
		t.Fatalf("query-averaged loss combined=%f split=%f", combinedLoss/float32(combinedQueries), splitLoss/float32(splitQueries))
	}
	for i := range combinedQueryGrads {
		assertFloat32SlicesClose(t, splitQueryGrads[i], combinedQueryGrads[i], 1e-6)
	}
	assertFloat32SlicesClose(t, []float32{splitDocGrads[0][0] + splitDocGrads[2][0], splitDocGrads[0][1] + splitDocGrads[2][1]}, combinedDocGrads[0], 1e-6)
	assertFloat32SlicesClose(t, []float32{splitDocGrads[1][0] + splitDocGrads[3][0], splitDocGrads[1][1] + splitDocGrads[3][1]}, combinedDocGrads[1], 1e-6)
}

func TestEstimateListwiseGeometryTrainWorkloadCountsMatrixCells(t *testing.T) {
	workload := EstimateListwiseGeometryTrainWorkload(tinyTokenizedListwiseGeometryBatches(false), 1, EmbeddingTrainRunConfig{
		Epochs:         2,
		BatchSize:      1,
		EvalEveryEpoch: 1,
	})
	if workload.TrainMode != "listwise_geometry" || workload.TrainPairsPerEpoch != 4 || workload.PlannedTrainPairs != 8 {
		t.Fatalf("workload = %+v, want listwise matrix cell counts", workload)
	}

	workload = EstimateListwiseGeometryTrainWorkload(nil, 0, EmbeddingTrainRunConfig{
		EvalOnly:             true,
		BatchSize:            1,
		ListwiseGeometryEval: tinyTokenizedListwiseGeometryBatches(false),
	})
	if workload.EvalMode != "listwise_geometry" || workload.EvalExamples != 2 || workload.EvalPairsPerPass != 4 || workload.PlannedEvalPairs != 4 {
		t.Fatalf("eval workload = %+v, want 2 query rows and 4 matrix cells", workload)
	}
}

func TestTrainEmbeddingPackageFromTextContrastiveFilesResearchOnlyListwiseWritesRejectableEmbeddingLineage(t *testing.T) {
	trainer := newTinyTrainable3DEmbeddingTrainer(t, 0.05)
	path := writeTinyTrainingPackage(t, trainer)
	tokenizerPath := filepath.Join(t.TempDir(), "tokenizer.mll")
	if err := tinyListwiseGeometryTokenizerFile().WriteFile(tokenizerPath); err != nil {
		t.Fatalf("write tokenizer: %v", err)
	}
	trainPath := writeTinyListwiseGeometryJSONL(t, tinyListwiseGeometryBatches(true))

	if _, _, err := TrainEmbeddingPackageFromTextContrastiveFiles(path, tokenizerPath, trainPath, "", EmbeddingTrainRunConfig{
		ListwiseGeometryTrain:             true,
		AllowResearchOnlyListwiseGeometry: true,
		Epochs:                            1,
		BatchSize:                         1,
		EvalEveryEpoch:                    1,
		Temperature:                       0.05,
	}); err != nil {
		t.Fatalf("train research-only listwise package: %v", err)
	}
	reloaded, err := LoadEmbeddingTrainerPackage(path)
	if err != nil {
		t.Fatalf("reload training package: %v", err)
	}
	embeddingPath := filepath.Join(t.TempDir(), "embedding.mll")
	paths, err := reloaded.WriteEmbeddingPackage(embeddingPath)
	if err != nil {
		t.Fatalf("write embedding package: %v", err)
	}
	manifest, err := ReadPackageManifestFile(paths.PackageManifestPath)
	if err != nil {
		t.Fatalf("read package manifest: %v", err)
	}
	if !manifest.ListwiseGeometry.ListwiseGeometryResearchOnly || manifest.ListwiseGeometry.ListwiseGeometryBatchCount != 1 {
		t.Fatalf("listwise embedding lineage = %+v, want research-only batch count 1", manifest.ListwiseGeometry)
	}
	if _, err := New().LoadEmbeddingPackage(context.Background(), embeddingPath); err == nil || !strings.Contains(err.Error(), "research-only listwise geometry") {
		t.Fatalf("load embedding package error = %v, want research-only listwise rejection", err)
	}
}

func TestTrainEmbeddingPackageFromTextContrastiveFilesListwiseGeometrySmoke(t *testing.T) {
	trainer := newTinyTrainable3DEmbeddingTrainer(t, 0.05)
	path := writeTinyTrainingPackage(t, trainer)
	tokenizerPath := filepath.Join(t.TempDir(), "tokenizer.mll")
	if err := tinyListwiseGeometryTokenizerFile().WriteFile(tokenizerPath); err != nil {
		t.Fatalf("write tokenizer: %v", err)
	}
	trainPath := writeTinyListwiseGeometryJSONL(t, tinyListwiseGeometryBatches(false))

	summary, _, err := TrainEmbeddingPackageFromTextContrastiveFiles(path, tokenizerPath, trainPath, "", EmbeddingTrainRunConfig{
		ListwiseGeometryTrain: true,
		Epochs:                1,
		BatchSize:             1,
		EvalEveryEpoch:        1,
		Temperature:           0.05,
	})
	if err != nil {
		t.Fatalf("train package listwise geometry: %v", err)
	}
	if summary.StepsCompleted != 1 || summary.Workload.TrainMode != "listwise_geometry" {
		t.Fatalf("summary = %+v, want listwise geometry training", summary)
	}
}

func TestTrainEmbeddingPackageFromTextContrastiveFilesListwiseGeometryPositionalEval(t *testing.T) {
	trainer := newTinyTrainable3DEmbeddingTrainer(t, 0.05)
	path := writeTinyTrainingPackage(t, trainer)
	tokenizerPath := filepath.Join(t.TempDir(), "tokenizer.mll")
	if err := tinyListwiseGeometryTokenizerFile().WriteFile(tokenizerPath); err != nil {
		t.Fatalf("write tokenizer: %v", err)
	}
	trainPath := writeTinyListwiseGeometryJSONL(t, tinyListwiseGeometryBatches(false))
	evalPath := writeTinyListwiseGeometryJSONL(t, tinyListwiseGeometryBatches(false))

	summary, _, err := TrainEmbeddingPackageFromTextContrastiveFiles(path, tokenizerPath, trainPath, evalPath, EmbeddingTrainRunConfig{
		ListwiseGeometryTrain: true,
		Epochs:                1,
		BatchSize:             1,
		EvalEveryEpoch:        1,
		Temperature:           0.05,
	})
	if err != nil {
		t.Fatalf("train package listwise geometry with positional eval: %v", err)
	}
	if summary.FinalListwiseGeometryEval == nil {
		t.Fatalf("summary = %+v, want final listwise geometry eval", summary)
	}
	if summary.Workload.EvalMode != "listwise_geometry" || summary.Workload.EvalExamples != 2 || summary.Workload.EvalPairsPerPass != 4 {
		t.Fatalf("workload = %+v, want listwise eval query/cell counts", summary.Workload)
	}
}

func TestTrainEmbeddingPackageFromTextContrastiveFilesListwiseEvalOnlyDoesNotRewritePackage(t *testing.T) {
	trainer := newTinyTrainable3DEmbeddingTrainer(t, 0.05)
	path := writeTinyTrainingPackage(t, trainer)
	tokenizerPath := filepath.Join(t.TempDir(), "tokenizer.mll")
	if err := tinyListwiseGeometryTokenizerFile().WriteFile(tokenizerPath); err != nil {
		t.Fatalf("write tokenizer: %v", err)
	}
	evalPath := writeTinyListwiseGeometryJSONL(t, tinyListwiseGeometryBatches(false))
	packageFiles := []string{
		path,
		DefaultEmbeddingManifestPath(path),
		DefaultWeightFilePath(path),
		DefaultMemoryPlanPath(path),
		DefaultEmbeddingTrainManifestPath(path),
		DefaultEmbeddingCheckpointPath(path),
		DefaultEmbeddingTrainProfilePath(path),
		DefaultPackageManifestPath(path),
	}
	before := map[string][]byte{}
	for _, file := range packageFiles {
		data, err := os.ReadFile(file)
		if err != nil {
			t.Fatalf("read package file %s: %v", file, err)
		}
		before[file] = data
	}

	summary, paths, err := TrainEmbeddingPackageFromTextContrastiveFiles(path, tokenizerPath, evalPath, "", EmbeddingTrainRunConfig{
		EvalOnly:              true,
		ListwiseGeometryTrain: true,
		BatchSize:             1,
		Temperature:           0.05,
	})
	if err != nil {
		t.Fatalf("eval-only listwise package: %v", err)
	}
	if paths.ArtifactPath != path || paths.CheckpointPath != DefaultEmbeddingCheckpointPath(path) || paths.TrainProfilePath != DefaultEmbeddingTrainProfilePath(path) {
		t.Fatalf("paths = %+v, want default package paths for %s", paths, path)
	}
	if summary.FinalListwiseGeometryEval == nil || summary.Workload.ActualEvalExamples != 2 || summary.Workload.ActualEvalPairs != 4 || summary.DeltaProfile.Optimizer.UpdateCalls != 0 {
		t.Fatalf("summary = %+v workload=%+v, want read-only listwise eval metrics", summary, summary.Workload)
	}
	for _, file := range packageFiles {
		after, err := os.ReadFile(file)
		if err != nil {
			t.Fatalf("read package file after eval %s: %v", file, err)
		}
		if !bytes.Equal(after, before[file]) {
			t.Fatalf("package file %s changed during listwise eval-only", file)
		}
	}
}

func tinyListwiseGeometryBatches(researchOnly bool) []EmbeddingListwiseGeometryBatch {
	return []EmbeddingListwiseGeometryBatch{{
		Schema:  "eos.listwise_geometry_batch.v1",
		BatchID: "batch-0",
		Examples: []EmbeddingListwiseGeometryExample{
			{RowID: "r0", Source: "unit", QueryID: "q0", PositiveDocID: "d0", NegativeDocIDs: []string{"d1"}},
			{RowID: "r1", Source: "unit", QueryID: "q1", PositiveDocID: "d1", NegativeDocIDs: []string{"d0"}},
		},
		Queries: []EmbeddingListwiseGeometryQuery{
			{ID: "q0", Text: "a"},
			{ID: "q1", Text: "b"},
		},
		Documents: []EmbeddingListwiseGeometryDocument{
			{ID: "d0", Text: "a"},
			{ID: "d1", Text: "b"},
		},
		TeacherSimilarity:       [][]float32{{0.9, 0.1}, {0.2, 0.8}},
		Score:                   "cosine",
		TrainAllowedForResearch: researchOnly,
		ReleaseTrainAllowed:     !researchOnly,
		CommercialUseAllowed:    !researchOnly,
	}}
}

func tinyTokenizedListwiseGeometryBatches(researchOnly bool) []EmbeddingTokenizedListwiseGeometryBatch {
	batch := tinyListwiseGeometryBatches(researchOnly)[0]
	return []EmbeddingTokenizedListwiseGeometryBatch{{
		Schema:                  batch.Schema,
		BatchID:                 batch.BatchID,
		Examples:                batch.Examples,
		QueryIDs:                []string{"q0", "q1"},
		QueryTokens:             [][]int32{{0}, {1}},
		QueryMasks:              [][]int32{{1}, {1}},
		DocumentIDs:             []string{"d0", "d1"},
		DocumentTokens:          [][]int32{{0}, {1}},
		DocumentMasks:           [][]int32{{1}, {1}},
		TeacherSimilarity:       cloneFloat32Matrix(batch.TeacherSimilarity),
		Score:                   batch.Score,
		TrainAllowedForResearch: batch.TrainAllowedForResearch,
		ReleaseTrainAllowed:     batch.ReleaseTrainAllowed,
		CommercialUseAllowed:    batch.CommercialUseAllowed,
	}}
}

func tinyListwiseGeometryTokenizer(t *testing.T) *BPETokenizer {
	t.Helper()
	tokenizer, err := NewBPETokenizer(tinyListwiseGeometryTokenizerFile(), TokenizerManifest{VocabSize: 6, MaxSequence: 8})
	if err != nil {
		t.Fatalf("new tokenizer: %v", err)
	}
	return tokenizer
}

func tinyListwiseGeometryTokenizerFile() TokenizerFile {
	return TokenizerFile{
		Version: TokenizerFileVersion,
		Tokens:  []string{"[UNK]", "a", "b"},
	}
}

func writeTinyListwiseGeometryJSONL(t *testing.T, batches []EmbeddingListwiseGeometryBatch) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "listwise-geometry.jsonl")
	f, err := os.Create(path)
	if err != nil {
		t.Fatalf("create listwise geometry jsonl: %v", err)
	}
	defer f.Close()
	enc := json.NewEncoder(f)
	for _, batch := range batches {
		record := embeddingListwiseGeometryBatchRecord{
			Schema:                  batch.Schema,
			BatchID:                 batch.BatchID,
			SourceCounts:            batch.SourceCounts,
			Examples:                batch.Examples,
			Queries:                 batch.Queries,
			Documents:               batch.Documents,
			TeacherSimilarity:       batch.TeacherSimilarity,
			TeacherModelID:          batch.TeacherModelID,
			Score:                   batch.Score,
			Normalized:              batch.Normalized,
			TrainPolicy:             batch.TrainPolicy,
			ReleaseTrainAllowed:     batch.ReleaseTrainAllowed,
			CommercialUseAllowed:    batch.CommercialUseAllowed,
			TrainAllowedForResearch: batch.TrainAllowedForResearch,
			SourceArtifactHash:      batch.SourceArtifactHash,
			ExtraFields:             batch.ExtraFields,
		}
		if err := enc.Encode(record); err != nil {
			t.Fatalf("encode listwise geometry jsonl: %v", err)
		}
	}
	return path
}
