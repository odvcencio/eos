package eosruntime

import (
	"fmt"
	"math"
	"math/rand"
	"os"
	"runtime/debug"
	"sort"
	"strconv"
	"strings"
	"time"
)

const (
	VectorDistillOptimizerSyncDeferred  = "deferred"
	VectorDistillOptimizerSyncImmediate = "immediate"
)

// EmbeddingTrainRunConfig controls dataset-level native training.
type EmbeddingTrainRunConfig struct {
	Epochs                int
	BatchSize             int
	Shuffle               bool
	Seed                  int64
	EvalEveryEpoch        int
	EvalEverySteps        int
	EarlyStoppingPatience int
	SelectMetric          string
	MinDelta              float32
	RestoreBest           bool
	EvalOnly              bool
	PairwiseTrain         bool
	HardNegativeTrain     bool
	ScoreSpectrumTrain    bool
	ListwiseGeometryTrain bool
	VectorDistillTrain    bool
	// VectorDistillDefaultRole is the role ("query", "document", or "raw")
	// applied to vector-distill rows that omit an explicit "role" field.
	// Defaults to "query" in normalizedTrainRunConfig to preserve pre-role
	// behavior (all vector-distill rows previously trained under query role).
	VectorDistillDefaultRole string
	// VectorDistillRelationalWeight enables (when > 0) an in-batch relational
	// (similarity-matrix) distillation term over the RAW student pooled
	// vectors, supervising the served geometry directly. 0 disables it.
	VectorDistillRelationalWeight     float32
	VectorDistillOptimizerSync        string
	MovementDiagnostics               bool
	AllowResearchOnlyScoreSpectrum    bool
	AllowResearchOnlyListwiseGeometry bool
	AllowResearchOnlyVectorDistill    bool
	HardNegativesPerQuery             int
	HardNegativeSourceWeights         map[string]int
	LengthBucketBatches               bool
	LearningRate                      float32
	ContrastiveLoss                   string
	Temperature                       float32
	GroupedLossWeight                 float32
	TeacherLossWeight                 float32
	TeacherLossWeightSet              bool
	TeacherTemperature                float32
	TeacherSourceTemperatures         map[string]float32
	TeacherSourceWeights              map[string]float32
	MatryoshkaDims                    []int
	MatryoshkaWeights                 []float32
	ClearTurboQuantPrefix             bool
	TurboQuantPrefixBits              []int
	TurboQuantPrefixObjectives        []TurboQuantPrefixObjective
	TurboQuantPrefixWeight            float32
	TurboQuantPrefixSeed              int64
	TurboQuantPrefixScoreMode         string
	TurboQuantCompactObjectives       []TurboQuantPrefixObjective
	ClearTurboQuantRankMargin         bool
	TurboQuantRankMarginObjectives    []TurboQuantPrefixObjective
	TurboQuantRankMargin              float32
	ScoreSpectrumEval                 []EmbeddingScoreSpectrumExample
	ScoreSpectrumEvalPath             string
	ListwiseGeometryEval              []EmbeddingTokenizedListwiseGeometryBatch
	MaxListwiseGeometryTrainPairs     int64
	MaxListwiseGeometryEvalPairs      int64
	ScoreSpectrumLossMode             string
	ScoreSpectrumRecoveryWeight       float32
	ScoreSpectrumRecoveryMargin       float32
	ScoreSpectrumRecoveryTopK         int
	ScoreSpectrumRecoveryTau          float32
	TeacherScoreNormalization         string
	ProgressEverySteps                int
	Progress                          EmbeddingTrainProgressFunc
	// Retrieval eval gate (optional). When RetrievalEvalRuntime and a complete
	// RetrievalEval (corpus/queries/qrels) are set, each eval also reports
	// nDCG@10, MAP@100, and recall@100 over that held-out set, usable as
	// -select-metric retrieval_ndcg(_at_10), retrieval_map, or retrieval_recall.
	// RetrievalEvalTokenizer is required to embed the raw text.
	RetrievalEvalRuntime   *Runtime
	RetrievalEval          RetrievalEvalConfig
	RetrievalEvalTokenizer *TokenizerFile
	// EvalPairs supplies labeled pairwise eval examples for contrastive
	// training runs. When set, FitContrastive evaluates them per epoch (and
	// per EvalEverySteps) for selection, early stopping, and best-checkpoint
	// restore — pairwise eval data otherwise cannot drive per-epoch selection
	// because the contrastive eval set stays empty.
	EvalPairs []EmbeddingPairExample
}

// EmbeddingTrainProgressFunc receives incremental training progress updates.
type EmbeddingTrainProgressFunc func(EmbeddingTrainProgress)

// EmbeddingTrainProgress reports one completed optimizer step.
type EmbeddingTrainProgress struct {
	Phase              string
	Epoch              int
	Batch              int
	Batches            int
	Step               int
	EvalPass           int
	BatchExamples      int
	BatchPairs         int64
	EpochTrainExamples int64
	EpochTrainPairs    int64
	PlannedEpochPairs  int64
	EvalExamples       int64
	EvalPairs          int64
	Loss               float32
	AverageScore       float32
	Elapsed            time.Duration
}

// EmbeddingTrainEpochSummary records one epoch of training progress.
type EmbeddingTrainEpochSummary struct {
	Epoch                int
	Step                 int
	Train                EmbeddingTrainMetrics
	Eval                 *EmbeddingEvalMetrics
	ScoreSpectrumEval    *EmbeddingScoreSpectrumEvalMetrics
	ListwiseGeometryEval *EmbeddingListwiseGeometryEvalMetrics
	Improved             bool
}

// EmbeddingTrainWorkload summarizes planned and actual pairwise work for a run.
type EmbeddingTrainWorkload struct {
	TrainMode            string
	EvalMode             string
	TrainExamples        int
	EvalExamples         int
	BatchSize            int
	PlannedEpochs        int
	CompletedEpochs      int
	TrainBatchesPerEpoch int
	TrainPairsPerEpoch   int64
	EvalPairsPerPass     int64
	PlannedEvalPasses    int
	ActualEvalPasses     int
	PlannedTrainPairs    int64
	ActualTrainPairs     int64
	ActualTrainExamples  int64
	PlannedEvalPairs     int64
	ActualEvalPairs      int64
	ActualEvalExamples   int64
	PlannedTotalPairs    int64
	ActualTotalPairs     int64
	ActualTotalExamples  int64
}

// EmbeddingTrainRunSummary summarizes a full train/eval run.
type EmbeddingTrainRunSummary struct {
	Config                    EmbeddingTrainRunConfig
	Workload                  EmbeddingTrainWorkload
	EpochsCompleted           int
	StepsCompleted            int
	StepsRun                  int
	BestEpoch                 int
	BestStep                  int
	FinalTrain                EmbeddingTrainMetrics
	LastEval                  *EmbeddingEvalMetrics
	BestEval                  *EmbeddingEvalMetrics
	FinalEval                 *EmbeddingEvalMetrics
	LastScoreSpectrumEval     *EmbeddingScoreSpectrumEvalMetrics
	BestScoreSpectrumEval     *EmbeddingScoreSpectrumEvalMetrics
	FinalScoreSpectrumEval    *EmbeddingScoreSpectrumEvalMetrics
	LastListwiseGeometryEval  *EmbeddingListwiseGeometryEvalMetrics
	BestListwiseGeometryEval  *EmbeddingListwiseGeometryEvalMetrics
	FinalListwiseGeometryEval *EmbeddingListwiseGeometryEvalMetrics
	EffectiveLearningRate     float32
	RestoredBest              bool
	StoppedEarly              bool
	History                   []EmbeddingTrainEpochSummary
	EvalHistory               []EmbeddingTrainEvalSummary
	StartProfile              EmbeddingTrainProfile
	EndProfile                EmbeddingTrainProfile
	DeltaProfile              EmbeddingTrainProfile
	Elapsed                   time.Duration
	TrainDuration             time.Duration
	EvalDuration              time.Duration
}

// EmbeddingTrainEvalSummary records one auditable evaluation pass.
type EmbeddingTrainEvalSummary struct {
	Epoch                int
	Step                 int
	EvalPass             int
	Trigger              string
	Improved             bool
	Eval                 *EmbeddingEvalMetrics
	ScoreSpectrumEval    *EmbeddingScoreSpectrumEvalMetrics
	ListwiseGeometryEval *EmbeddingListwiseGeometryEvalMetrics
}

func appendTrainEvalHistory(summary *EmbeddingTrainRunSummary, epoch, step int, trigger string, improved bool, eval *EmbeddingEvalMetrics, scoreEval *EmbeddingScoreSpectrumEvalMetrics, listwiseEval *EmbeddingListwiseGeometryEvalMetrics) {
	if summary == nil {
		return
	}
	summary.EvalHistory = append(summary.EvalHistory, EmbeddingTrainEvalSummary{
		Epoch:                epoch,
		Step:                 step,
		EvalPass:             summary.Workload.ActualEvalPasses,
		Trigger:              trigger,
		Improved:             improved,
		Eval:                 cloneEvalMetricsPtr(eval),
		ScoreSpectrumEval:    cloneScoreSpectrumEvalMetricsPtr(scoreEval),
		ListwiseGeometryEval: cloneListwiseGeometryEvalMetricsPtr(listwiseEval),
	})
}

// Fit trains over a dataset, periodically evaluates, and can restore the best checkpoint.
func (t *EmbeddingTrainer) Fit(trainSet, evalSet []EmbeddingPairExample, cfg EmbeddingTrainRunConfig) (EmbeddingTrainRunSummary, error) {
	if t == nil {
		return EmbeddingTrainRunSummary{}, fmt.Errorf("embedding trainer is not initialized")
	}
	cfg = normalizedTrainRunConfig(cfg)
	if err := validateClearTurboQuantPrefixRunConfig(cfg); err != nil {
		return EmbeddingTrainRunSummary{}, err
	}
	if err := validateClearTurboQuantRankMarginRunConfig(cfg); err != nil {
		return EmbeddingTrainRunSummary{}, err
	}
	if len(cfg.TurboQuantRankMarginObjectives) > 0 {
		return EmbeddingTrainRunSummary{}, fmt.Errorf("turboquant_rank_margin_objectives require hard-negative training")
	}
	if len(cfg.TurboQuantCompactObjectives) > 0 {
		return EmbeddingTrainRunSummary{}, fmt.Errorf("turboquant_compact_objectives require hard-negative training")
	}
	if cfg.EvalOnly {
		if len(evalSet) == 0 {
			return EmbeddingTrainRunSummary{}, fmt.Errorf("eval dataset is empty")
		}
	} else {
		if len(trainSet) == 0 {
			return EmbeddingTrainRunSummary{}, fmt.Errorf("training dataset is empty")
		}
		if cfg.Epochs <= 0 {
			return EmbeddingTrainRunSummary{}, fmt.Errorf("epochs must be positive")
		}
		if cfg.BatchSize <= 0 {
			return EmbeddingTrainRunSummary{}, fmt.Errorf("batch_size must be positive")
		}
	}
	if cfg.EvalEveryEpoch <= 0 {
		return EmbeddingTrainRunSummary{}, fmt.Errorf("eval_every_epoch must be positive")
	}
	if cfg.EvalEverySteps < 0 {
		return EmbeddingTrainRunSummary{}, fmt.Errorf("eval_every_steps must be non-negative")
	}
	if cfg.EarlyStoppingPatience < 0 {
		return EmbeddingTrainRunSummary{}, fmt.Errorf("early_stopping_patience must be non-negative")
	}
	if cfg.ProgressEverySteps < 0 {
		return EmbeddingTrainRunSummary{}, fmt.Errorf("progress_every_steps must be non-negative")
	}
	if !validTrainSelectionMetric(cfg.SelectMetric) {
		return EmbeddingTrainRunSummary{}, fmt.Errorf("unsupported select_metric %q", cfg.SelectMetric)
	}
	t.configureRetrievalEval(cfg.RetrievalEvalRuntime, cfg.RetrievalEval, cfg.RetrievalEvalTokenizer)
	if err := t.applyTrainRunOverrides(cfg); err != nil {
		return EmbeddingTrainRunSummary{}, err
	}
	cfg = t.syncTrainRunObjectiveConfig(cfg)
	if len(cfg.TurboQuantRankMarginObjectives) > 0 {
		return EmbeddingTrainRunSummary{}, fmt.Errorf("turboquant_rank_margin_objectives require hard-negative training")
	}
	if len(cfg.TurboQuantCompactObjectives) > 0 {
		return EmbeddingTrainRunSummary{}, fmt.Errorf("turboquant_compact_objectives require hard-negative training")
	}

	runStart := time.Now()
	startStep := t.step
	summary := EmbeddingTrainRunSummary{
		Config:                cfg,
		EffectiveLearningRate: t.config.LearningRate,
		StartProfile:          t.TrainProfile(),
		Workload:              EstimatePairwiseTrainWorkload(len(trainSet), len(evalSet), cfg),
	}
	if cfg.EvalOnly {
		evalStart := time.Now()
		maybeReportEvalProgress(cfg, "eval_start", 0, t.step, 1, int64(len(evalSet)), int64(len(evalSet)), runStart)
		finalEval, err := t.EvaluatePairs(evalSet)
		if err != nil {
			return EmbeddingTrainRunSummary{}, fmt.Errorf("eval: %w", err)
		}
		summary.EvalDuration = time.Since(evalStart)
		summary.StepsCompleted = t.step
		summary.FinalEval = cloneEvalMetrics(finalEval)
		summary.LastEval = cloneEvalMetrics(finalEval)
		summary.BestEval = cloneEvalMetrics(finalEval)
		summary.BestStep = t.step
		summary.Workload.ActualEvalPasses = 1
		summary.Workload.ActualEvalPairs = int64(len(evalSet))
		summary.Workload.ActualEvalExamples = int64(len(evalSet))
		maybeReportEvalProgress(cfg, "eval_done", 0, t.step, summary.Workload.ActualEvalPasses, summary.Workload.ActualEvalExamples, summary.Workload.ActualEvalPairs, runStart)
		appendTrainEvalHistory(&summary, 0, t.step, "eval_only", true, summary.FinalEval, nil, nil)
		summary.EndProfile = t.TrainProfile()
		summary.DeltaProfile = diffTrainProfile(summary.StartProfile, summary.EndProfile)
		summary.Workload.ActualTotalPairs = summary.Workload.ActualEvalPairs
		summary.Workload.ActualTotalExamples = summary.Workload.ActualEvalExamples
		summary.Elapsed = time.Since(runStart)
		return summary, nil
	}

	indices := make([]int, len(trainSet))
	for i := range indices {
		indices[i] = i
	}
	rng := rand.New(rand.NewSource(cfg.Seed))

	var (
		bestCheckpoint EmbeddingTrainCheckpoint
		haveBest       bool
		noImproveEvals int
	)

	for epoch := 1; epoch <= cfg.Epochs; epoch++ {
		if cfg.Shuffle {
			rng.Shuffle(len(indices), func(i, j int) {
				indices[i], indices[j] = indices[j], indices[i]
			})
		}

		trainStart := time.Now()
		trainMetrics, err := t.runEpoch(trainSet, indices, cfg.BatchSize, cfg, epoch, runStart)
		if err != nil {
			return EmbeddingTrainRunSummary{}, fmt.Errorf("epoch %d: %w", epoch, err)
		}
		summary.TrainDuration += time.Since(trainStart)
		record := EmbeddingTrainEpochSummary{
			Epoch: epoch,
			Step:  t.step,
			Train: trainMetrics,
		}
		summary.FinalTrain = trainMetrics
		summary.EpochsCompleted = epoch
		summary.Workload.CompletedEpochs = epoch
		summary.Workload.ActualTrainPairs += int64(trainMetrics.BatchSize)
		summary.Workload.ActualTrainExamples += int64(trainMetrics.BatchSize)

		if len(evalSet) > 0 && epoch%cfg.EvalEveryEpoch == 0 {
			evalStart := time.Now()
			evalPass := summary.Workload.ActualEvalPasses + 1
			maybeReportEvalProgress(cfg, "eval_start", epoch, t.step, evalPass, int64(len(evalSet)), int64(len(evalSet)), runStart)
			evalMetrics, err := t.EvaluatePairs(evalSet)
			if err != nil {
				return EmbeddingTrainRunSummary{}, fmt.Errorf("epoch %d eval: %w", epoch, err)
			}
			summary.EvalDuration += time.Since(evalStart)
			summary.Workload.ActualEvalPasses++
			summary.Workload.ActualEvalPairs += int64(len(evalSet))
			summary.Workload.ActualEvalExamples += int64(len(evalSet))
			record.Eval = cloneEvalMetrics(evalMetrics)
			summary.LastEval = cloneEvalMetrics(evalMetrics)
			maybeReportEvalProgress(cfg, "eval_done", epoch, t.step, summary.Workload.ActualEvalPasses, summary.Workload.ActualEvalExamples, summary.Workload.ActualEvalPairs, runStart)
			if !haveBest || betterEvalMetrics(evalMetrics, *summary.BestEval, cfg.SelectMetric, cfg.MinDelta) {
				bestCheckpoint, err = t.Checkpoint()
				if err != nil {
					return EmbeddingTrainRunSummary{}, fmt.Errorf("epoch %d checkpoint: %w", epoch, err)
				}
				haveBest = true
				record.Improved = true
				summary.BestEval = cloneEvalMetrics(evalMetrics)
				summary.BestEpoch = epoch
				summary.BestStep = t.step
				noImproveEvals = 0
				appendTrainEvalHistory(&summary, epoch, t.step, "epoch", true, record.Eval, nil, nil)
			} else {
				noImproveEvals++
				appendTrainEvalHistory(&summary, epoch, t.step, "epoch", false, record.Eval, nil, nil)
				if cfg.EarlyStoppingPatience > 0 && noImproveEvals >= cfg.EarlyStoppingPatience {
					summary.StoppedEarly = true
					summary.History = append(summary.History, record)
					break
				}
			}
		}

		summary.History = append(summary.History, record)
	}

	summary.StepsCompleted = t.step
	summary.StepsRun = t.step - startStep
	preRestoreEndProfile := t.TrainProfile()
	restoreStartProfile := EmbeddingTrainProfile{}
	restored := false
	if cfg.RestoreBest && haveBest {
		if err := t.restoreCheckpoint(bestCheckpoint); err != nil {
			return EmbeddingTrainRunSummary{}, err
		}
		summary.RestoredBest = true
		restoreStartProfile = t.TrainProfile()
		restored = true
	}
	if len(evalSet) > 0 {
		evalStart := time.Now()
		evalPass := summary.Workload.ActualEvalPasses + 1
		maybeReportEvalProgress(cfg, "eval_start", summary.EpochsCompleted, t.step, evalPass, int64(len(evalSet)), int64(len(evalSet)), runStart)
		finalEval, err := t.EvaluatePairs(evalSet)
		if err != nil {
			return EmbeddingTrainRunSummary{}, fmt.Errorf("final eval: %w", err)
		}
		summary.EvalDuration += time.Since(evalStart)
		summary.Workload.ActualEvalPasses++
		summary.Workload.ActualEvalPairs += int64(len(evalSet))
		summary.Workload.ActualEvalExamples += int64(len(evalSet))
		summary.FinalEval = cloneEvalMetrics(finalEval)
		maybeReportEvalProgress(cfg, "eval_done", summary.EpochsCompleted, t.step, summary.Workload.ActualEvalPasses, summary.Workload.ActualEvalExamples, summary.Workload.ActualEvalPairs, runStart)
		appendTrainEvalHistory(&summary, summary.EpochsCompleted, t.step, "final", false, summary.FinalEval, nil, nil)
		if summary.BestEval == nil {
			summary.BestEval = cloneEvalMetrics(finalEval)
			if summary.BestEpoch == 0 {
				summary.BestEpoch = summary.EpochsCompleted
			}
			if summary.BestStep == 0 {
				summary.BestStep = summary.StepsCompleted
			}
		}
	}
	finalProfile := t.TrainProfile()
	if restored {
		preRestoreDelta := diffTrainProfile(summary.StartProfile, preRestoreEndProfile)
		postRestoreDelta := diffTrainProfile(restoreStartProfile, finalProfile)
		summary.DeltaProfile = addTrainProfileDelta(preRestoreDelta, postRestoreDelta)
		summary.EndProfile = applyTrainProfileDelta(preRestoreEndProfile, postRestoreDelta)
	} else {
		summary.EndProfile = finalProfile
		summary.DeltaProfile = diffTrainProfile(summary.StartProfile, summary.EndProfile)
	}
	summary.Workload.ActualTotalPairs = summary.Workload.ActualTrainPairs + summary.Workload.ActualEvalPairs
	summary.Workload.ActualTotalExamples = summary.Workload.ActualTrainExamples + summary.Workload.ActualEvalExamples
	summary.Elapsed = time.Since(runStart)
	return summary, nil
}

// FitContrastive trains over query-positive examples using in-batch negatives.
func (t *EmbeddingTrainer) FitContrastive(trainSet, evalSet []EmbeddingContrastiveExample, cfg EmbeddingTrainRunConfig) (EmbeddingTrainRunSummary, error) {
	if t == nil {
		return EmbeddingTrainRunSummary{}, fmt.Errorf("embedding trainer is not initialized")
	}
	cfg = normalizedTrainRunConfig(cfg)
	if err := validateClearTurboQuantPrefixRunConfig(cfg); err != nil {
		return EmbeddingTrainRunSummary{}, err
	}
	if err := validateClearTurboQuantRankMarginRunConfig(cfg); err != nil {
		return EmbeddingTrainRunSummary{}, err
	}
	if len(cfg.TurboQuantRankMarginObjectives) > 0 {
		return EmbeddingTrainRunSummary{}, fmt.Errorf("turboquant_rank_margin_objectives require hard-negative training")
	}
	if len(cfg.TurboQuantCompactObjectives) > 0 {
		return EmbeddingTrainRunSummary{}, fmt.Errorf("turboquant_compact_objectives require hard-negative training")
	}
	if cfg.EvalOnly {
		if len(evalSet) == 0 {
			return EmbeddingTrainRunSummary{}, fmt.Errorf("eval dataset is empty")
		}
	} else {
		if len(trainSet) == 0 {
			return EmbeddingTrainRunSummary{}, fmt.Errorf("training dataset is empty")
		}
		if cfg.Epochs <= 0 {
			return EmbeddingTrainRunSummary{}, fmt.Errorf("epochs must be positive")
		}
		if cfg.BatchSize <= 1 {
			return EmbeddingTrainRunSummary{}, fmt.Errorf("batch_size must be at least 2 for contrastive training")
		}
	}
	if cfg.EvalEveryEpoch <= 0 {
		return EmbeddingTrainRunSummary{}, fmt.Errorf("eval_every_epoch must be positive")
	}
	if cfg.EvalEverySteps < 0 {
		return EmbeddingTrainRunSummary{}, fmt.Errorf("eval_every_steps must be non-negative")
	}
	if cfg.EarlyStoppingPatience < 0 {
		return EmbeddingTrainRunSummary{}, fmt.Errorf("early_stopping_patience must be non-negative")
	}
	if cfg.ProgressEverySteps < 0 {
		return EmbeddingTrainRunSummary{}, fmt.Errorf("progress_every_steps must be non-negative")
	}
	if !validTrainSelectionMetric(cfg.SelectMetric) {
		return EmbeddingTrainRunSummary{}, fmt.Errorf("unsupported select_metric %q", cfg.SelectMetric)
	}
	t.configureRetrievalEval(cfg.RetrievalEvalRuntime, cfg.RetrievalEval, cfg.RetrievalEvalTokenizer)
	if err := t.applyTrainRunOverrides(cfg); err != nil {
		return EmbeddingTrainRunSummary{}, err
	}
	cfg = t.syncTrainRunObjectiveConfig(cfg)
	if len(cfg.TurboQuantRankMarginObjectives) > 0 {
		return EmbeddingTrainRunSummary{}, fmt.Errorf("turboquant_rank_margin_objectives require hard-negative training")
	}

	runStart := time.Now()
	startStep := t.step
	workload := EstimateContrastiveTrainWorkload(len(trainSet), len(evalSet), cfg)
	if len(cfg.EvalPairs) > 0 {
		workload = RetargetWorkloadToPairwiseEval(workload, len(cfg.EvalPairs), cfg)
	}
	summary := EmbeddingTrainRunSummary{
		Config:                cfg,
		EffectiveLearningRate: t.config.LearningRate,
		StartProfile:          t.TrainProfile(),
		Workload:              workload,
	}
	if cfg.EvalOnly {
		evalStart := time.Now()
		maybeReportEvalProgress(cfg, "eval_start", 0, t.step, 1, int64(len(evalSet)), int64(len(evalSet)*len(evalSet)), runStart)
		finalEval, err := t.EvaluateContrastive(evalSet)
		if err != nil {
			return EmbeddingTrainRunSummary{}, fmt.Errorf("eval: %w", err)
		}
		summary.EvalDuration = time.Since(evalStart)
		summary.StepsCompleted = t.step
		summary.FinalEval = cloneEvalMetrics(finalEval)
		summary.LastEval = cloneEvalMetrics(finalEval)
		summary.BestEval = cloneEvalMetrics(finalEval)
		summary.BestStep = t.step
		summary.Workload.ActualEvalPasses = 1
		summary.Workload.ActualEvalPairs = int64(len(evalSet) * len(evalSet))
		summary.Workload.ActualEvalExamples = int64(len(evalSet))
		maybeReportEvalProgress(cfg, "eval_done", 0, t.step, summary.Workload.ActualEvalPasses, summary.Workload.ActualEvalExamples, summary.Workload.ActualEvalPairs, runStart)
		summary.EndProfile = t.TrainProfile()
		summary.DeltaProfile = diffTrainProfile(summary.StartProfile, summary.EndProfile)
		summary.Workload.ActualTotalPairs = summary.Workload.ActualEvalPairs
		summary.Workload.ActualTotalExamples = summary.Workload.ActualEvalExamples
		summary.Elapsed = time.Since(runStart)
		return summary, nil
	}

	indices := make([]int, len(trainSet))
	for i := range indices {
		indices[i] = i
	}
	rng := rand.New(rand.NewSource(cfg.Seed))
	var (
		bestCheckpoint EmbeddingTrainCheckpoint
		haveBest       bool
		noImproveEvals int
	)
	runEval := func() (EmbeddingEvalMetrics, error) {
		if len(cfg.EvalPairs) > 0 {
			return t.EvaluatePairs(cfg.EvalPairs)
		}
		return t.EvaluateContrastive(evalSet)
	}
	evalPairsPerPass := func() int64 {
		if len(cfg.EvalPairs) > 0 {
			return int64(len(cfg.EvalPairs))
		}
		return int64(len(evalSet) * len(evalSet))
	}
	evalExamplesPerPass := func() int64 {
		if len(cfg.EvalPairs) > 0 {
			return int64(len(cfg.EvalPairs))
		}
		return int64(len(evalSet))
	}
	hasEval := len(evalSet) > 0 || len(cfg.EvalPairs) > 0
	recordEval := func(epoch int, trigger string) (*EmbeddingEvalMetrics, bool, error) {
		evalStart := time.Now()
		evalPass := summary.Workload.ActualEvalPasses + 1
		maybeReportEvalProgress(cfg, "eval_start", epoch, t.step, evalPass, evalExamplesPerPass(), evalPairsPerPass(), runStart)
		evalMetrics, err := runEval()
		if err != nil {
			return nil, false, err
		}
		summary.EvalDuration += time.Since(evalStart)
		summary.Workload.ActualEvalPasses++
		summary.Workload.ActualEvalPairs += evalPairsPerPass()
		summary.Workload.ActualEvalExamples += evalExamplesPerPass()
		summary.LastEval = cloneEvalMetrics(evalMetrics)
		maybeReportEvalProgress(cfg, "eval_done", epoch, t.step, summary.Workload.ActualEvalPasses, summary.Workload.ActualEvalExamples, summary.Workload.ActualEvalPairs, runStart)
		improved := false
		if !haveBest || betterEvalMetrics(evalMetrics, *summary.BestEval, cfg.SelectMetric, cfg.MinDelta) {
			bestCheckpoint, err = t.Checkpoint()
			if err != nil {
				return nil, false, err
			}
			haveBest = true
			improved = true
			summary.BestEval = cloneEvalMetrics(evalMetrics)
			summary.BestEpoch = epoch
			summary.BestStep = t.step
			noImproveEvals = 0
		} else {
			noImproveEvals++
		}
		evalMetricsCopy := cloneEvalMetrics(evalMetrics)
		appendTrainEvalHistory(&summary, epoch, t.step, trigger, improved, evalMetricsCopy, nil, nil)
		return evalMetricsCopy, improved, nil
	}
	if hasEval && cfg.RestoreBest {
		if _, _, err := recordEval(0, "initial"); err != nil {
			return EmbeddingTrainRunSummary{}, fmt.Errorf("initial eval: %w", err)
		}
	}

	for epoch := 1; epoch <= cfg.Epochs; epoch++ {
		if cfg.Shuffle {
			rng.Shuffle(len(indices), func(i, j int) {
				indices[i], indices[j] = indices[j], indices[i]
			})
		}
		if cfg.LengthBucketBatches {
			bucketContrastiveOrderByLength(trainSet, indices, cfg.BatchSize)
		}
		trainStart := time.Now()
		var afterBatch contrastiveEpochBatchHook
		if hasEval && cfg.EvalEverySteps > 0 {
			afterBatch = func(progress EmbeddingTrainProgress) error {
				if progress.Batch <= 0 || progress.Batch%cfg.EvalEverySteps != 0 {
					return nil
				}
				if _, _, err := recordEval(epoch, "step"); err != nil {
					return fmt.Errorf("step %d eval: %w", progress.Step, err)
				}
				return nil
			}
		}
		trainMetrics, err := t.runContrastiveEpoch(trainSet, indices, cfg.BatchSize, cfg, epoch, runStart, afterBatch)
		if err != nil {
			return EmbeddingTrainRunSummary{}, fmt.Errorf("epoch %d: %w", epoch, err)
		}
		summary.TrainDuration += time.Since(trainStart)
		record := EmbeddingTrainEpochSummary{
			Epoch: epoch,
			Step:  t.step,
			Train: trainMetrics,
		}
		summary.FinalTrain = trainMetrics
		summary.EpochsCompleted = epoch
		summary.Workload.CompletedEpochs = epoch
		summary.Workload.ActualTrainPairs += int64(trainMetrics.BatchSize)
		summary.Workload.ActualTrainExamples += int64(contrastiveUsableExampleCount(len(indices), cfg.BatchSize))

		if hasEval && epoch%cfg.EvalEveryEpoch == 0 {
			evalMetrics, improved, err := recordEval(epoch, "epoch")
			if err != nil {
				return EmbeddingTrainRunSummary{}, fmt.Errorf("epoch %d eval: %w", epoch, err)
			}
			record.Eval = evalMetrics
			record.Improved = improved
			if !improved {
				if cfg.EarlyStoppingPatience > 0 && noImproveEvals >= cfg.EarlyStoppingPatience {
					summary.StoppedEarly = true
					summary.History = append(summary.History, record)
					break
				}
			}
		}

		summary.History = append(summary.History, record)
	}

	summary.StepsCompleted = t.step
	summary.StepsRun = t.step - startStep
	preRestoreEndProfile := t.TrainProfile()
	restoreStartProfile := EmbeddingTrainProfile{}
	restored := false
	if cfg.RestoreBest && haveBest {
		if err := t.restoreCheckpoint(bestCheckpoint); err != nil {
			return EmbeddingTrainRunSummary{}, err
		}
		summary.RestoredBest = true
		restoreStartProfile = t.TrainProfile()
		restored = true
	}
	if hasEval {
		evalStart := time.Now()
		evalPass := summary.Workload.ActualEvalPasses + 1
		maybeReportEvalProgress(cfg, "eval_start", summary.EpochsCompleted, t.step, evalPass, evalExamplesPerPass(), evalPairsPerPass(), runStart)
		finalEval, err := runEval()
		if err != nil {
			return EmbeddingTrainRunSummary{}, fmt.Errorf("final eval: %w", err)
		}
		summary.EvalDuration += time.Since(evalStart)
		summary.Workload.ActualEvalPasses++
		summary.Workload.ActualEvalPairs += evalPairsPerPass()
		summary.Workload.ActualEvalExamples += evalExamplesPerPass()
		summary.FinalEval = cloneEvalMetrics(finalEval)
		maybeReportEvalProgress(cfg, "eval_done", summary.EpochsCompleted, t.step, summary.Workload.ActualEvalPasses, summary.Workload.ActualEvalExamples, summary.Workload.ActualEvalPairs, runStart)
		appendTrainEvalHistory(&summary, summary.EpochsCompleted, t.step, "final", false, summary.FinalEval, nil, nil)
		if summary.BestEval == nil {
			summary.BestEval = cloneEvalMetrics(finalEval)
			if summary.BestEpoch == 0 {
				summary.BestEpoch = summary.EpochsCompleted
			}
			if summary.BestStep == 0 {
				summary.BestStep = summary.StepsCompleted
			}
		}
	}
	finalProfile := t.TrainProfile()
	if restored {
		preRestoreDelta := diffTrainProfile(summary.StartProfile, preRestoreEndProfile)
		postRestoreDelta := diffTrainProfile(restoreStartProfile, finalProfile)
		summary.DeltaProfile = addTrainProfileDelta(preRestoreDelta, postRestoreDelta)
		summary.EndProfile = applyTrainProfileDelta(preRestoreEndProfile, postRestoreDelta)
	} else {
		summary.EndProfile = finalProfile
		summary.DeltaProfile = diffTrainProfile(summary.StartProfile, summary.EndProfile)
	}
	summary.Workload.ActualTotalPairs = summary.Workload.ActualTrainPairs + summary.Workload.ActualEvalPairs
	summary.Workload.ActualTotalExamples = summary.Workload.ActualTrainExamples + summary.Workload.ActualEvalExamples
	summary.Elapsed = time.Since(runStart)
	return summary, nil
}

// FitHardNegatives trains over query-positive examples with explicit hard negatives.
func (t *EmbeddingTrainer) FitHardNegatives(trainSet []EmbeddingHardNegativeExample, evalSet []EmbeddingPairExample, cfg EmbeddingTrainRunConfig) (EmbeddingTrainRunSummary, error) {
	if t == nil {
		return EmbeddingTrainRunSummary{}, fmt.Errorf("embedding trainer is not initialized")
	}
	cfg = normalizedTrainRunConfig(cfg)
	if err := validateClearTurboQuantPrefixRunConfig(cfg); err != nil {
		return EmbeddingTrainRunSummary{}, err
	}
	if err := validateClearTurboQuantRankMarginRunConfig(cfg); err != nil {
		return EmbeddingTrainRunSummary{}, err
	}
	if cfg.EvalOnly {
		if len(evalSet) == 0 {
			return EmbeddingTrainRunSummary{}, fmt.Errorf("eval dataset is empty")
		}
	} else {
		if len(trainSet) == 0 {
			return EmbeddingTrainRunSummary{}, fmt.Errorf("training dataset is empty")
		}
		if cfg.Epochs <= 0 {
			return EmbeddingTrainRunSummary{}, fmt.Errorf("epochs must be positive")
		}
		if cfg.BatchSize <= 0 {
			return EmbeddingTrainRunSummary{}, fmt.Errorf("batch_size must be positive")
		}
	}
	if cfg.EvalEveryEpoch <= 0 {
		return EmbeddingTrainRunSummary{}, fmt.Errorf("eval_every_epoch must be positive")
	}
	if cfg.EvalEverySteps < 0 {
		return EmbeddingTrainRunSummary{}, fmt.Errorf("eval_every_steps must be non-negative")
	}
	if cfg.EarlyStoppingPatience < 0 {
		return EmbeddingTrainRunSummary{}, fmt.Errorf("early_stopping_patience must be non-negative")
	}
	if cfg.ProgressEverySteps < 0 {
		return EmbeddingTrainRunSummary{}, fmt.Errorf("progress_every_steps must be non-negative")
	}
	if !validTrainSelectionMetric(cfg.SelectMetric) {
		return EmbeddingTrainRunSummary{}, fmt.Errorf("unsupported select_metric %q", cfg.SelectMetric)
	}
	t.configureRetrievalEval(cfg.RetrievalEvalRuntime, cfg.RetrievalEval, cfg.RetrievalEvalTokenizer)
	if !cfg.EvalOnly {
		var err error
		trainSet, err = normalizeHardNegativeTeacherScoresForRun(trainSet, cfg.TeacherScoreNormalization)
		if err != nil {
			return EmbeddingTrainRunSummary{}, err
		}
	}
	if err := t.applyTrainRunOverrides(cfg); err != nil {
		return EmbeddingTrainRunSummary{}, err
	}
	cfg = t.syncTrainRunObjectiveConfig(cfg)

	runStart := time.Now()
	startStep := t.step
	summary := EmbeddingTrainRunSummary{
		Config:                cfg,
		EffectiveLearningRate: t.config.LearningRate,
		StartProfile:          t.TrainProfile(),
		Workload:              EstimateHardNegativeTrainWorkload(len(trainSet), cfg.HardNegativesPerQuery, len(evalSet), cfg),
	}
	if cfg.EvalOnly {
		evalStart := time.Now()
		maybeReportEvalProgress(cfg, "eval_start", 0, t.step, 1, int64(len(evalSet)), int64(len(evalSet)), runStart)
		finalEval, err := t.EvaluatePairs(evalSet)
		if err != nil {
			return EmbeddingTrainRunSummary{}, fmt.Errorf("eval: %w", err)
		}
		summary.EvalDuration = time.Since(evalStart)
		summary.StepsCompleted = t.step
		summary.FinalEval = cloneEvalMetrics(finalEval)
		summary.LastEval = cloneEvalMetrics(finalEval)
		summary.BestEval = cloneEvalMetrics(finalEval)
		summary.BestStep = t.step
		summary.Workload.ActualEvalPasses = 1
		summary.Workload.ActualEvalPairs = int64(len(evalSet))
		summary.Workload.ActualEvalExamples = int64(len(evalSet))
		maybeReportEvalProgress(cfg, "eval_done", 0, t.step, summary.Workload.ActualEvalPasses, summary.Workload.ActualEvalExamples, summary.Workload.ActualEvalPairs, runStart)
		summary.EndProfile = t.TrainProfile()
		summary.DeltaProfile = diffTrainProfile(summary.StartProfile, summary.EndProfile)
		summary.Workload.ActualTotalPairs = summary.Workload.ActualEvalPairs
		summary.Workload.ActualTotalExamples = summary.Workload.ActualEvalExamples
		summary.Elapsed = time.Since(runStart)
		return summary, nil
	}

	indices := make([]int, len(trainSet))
	for i := range indices {
		indices[i] = i
	}
	rng := rand.New(rand.NewSource(cfg.Seed))
	var (
		bestCheckpoint EmbeddingTrainCheckpoint
		haveBest       bool
		noImproveEvals int
	)
	recordEval := func(epoch int, trigger string) (*EmbeddingEvalMetrics, bool, error) {
		evalStart := time.Now()
		evalPass := summary.Workload.ActualEvalPasses + 1
		maybeReportEvalProgress(cfg, "eval_start", epoch, t.step, evalPass, int64(len(evalSet)), int64(len(evalSet)), runStart)
		evalMetrics, err := t.EvaluatePairs(evalSet)
		if err != nil {
			return nil, false, err
		}
		summary.EvalDuration += time.Since(evalStart)
		summary.Workload.ActualEvalPasses++
		summary.Workload.ActualEvalPairs += int64(len(evalSet))
		summary.Workload.ActualEvalExamples += int64(len(evalSet))
		summary.LastEval = cloneEvalMetrics(evalMetrics)
		maybeReportEvalProgress(cfg, "eval_done", epoch, t.step, summary.Workload.ActualEvalPasses, summary.Workload.ActualEvalExamples, summary.Workload.ActualEvalPairs, runStart)
		improved := false
		if !haveBest || betterEvalMetrics(evalMetrics, *summary.BestEval, cfg.SelectMetric, cfg.MinDelta) {
			bestCheckpoint, err = t.Checkpoint()
			if err != nil {
				return nil, false, err
			}
			haveBest = true
			improved = true
			summary.BestEval = cloneEvalMetrics(evalMetrics)
			summary.BestEpoch = epoch
			summary.BestStep = t.step
			noImproveEvals = 0
		} else {
			noImproveEvals++
		}
		evalMetricsCopy := cloneEvalMetrics(evalMetrics)
		appendTrainEvalHistory(&summary, epoch, t.step, trigger, improved, evalMetricsCopy, nil, nil)
		return evalMetricsCopy, improved, nil
	}
	if len(evalSet) > 0 && cfg.RestoreBest {
		if _, _, err := recordEval(0, "initial"); err != nil {
			return EmbeddingTrainRunSummary{}, fmt.Errorf("initial eval: %w", err)
		}
	}

	for epoch := 1; epoch <= cfg.Epochs; epoch++ {
		if cfg.Shuffle {
			rng.Shuffle(len(indices), func(i, j int) {
				indices[i], indices[j] = indices[j], indices[i]
			})
		}
		if len(cfg.HardNegativeSourceWeights) > 0 {
			indices = hardNegativeSourceWeightedOrder(trainSet, indices, cfg.BatchSize, cfg.HardNegativeSourceWeights, cfg.LengthBucketBatches)
		} else if cfg.LengthBucketBatches {
			bucketHardNegativeOrderByLength(trainSet, indices, cfg.BatchSize)
		}
		trainStart := time.Now()
		var afterBatch contrastiveEpochBatchHook
		if len(evalSet) > 0 && cfg.EvalEverySteps > 0 {
			afterBatch = func(progress EmbeddingTrainProgress) error {
				if progress.Batch <= 0 || progress.Batch%cfg.EvalEverySteps != 0 {
					return nil
				}
				if _, _, err := recordEval(epoch, "step"); err != nil {
					return fmt.Errorf("step %d eval: %w", progress.Step, err)
				}
				return nil
			}
		}
		trainMetrics, err := t.runHardNegativeEpoch(trainSet, indices, cfg.BatchSize, cfg, epoch, runStart, afterBatch)
		if err != nil {
			return EmbeddingTrainRunSummary{}, fmt.Errorf("epoch %d: %w", epoch, err)
		}
		summary.TrainDuration += time.Since(trainStart)
		record := EmbeddingTrainEpochSummary{
			Epoch: epoch,
			Step:  t.step,
			Train: trainMetrics,
		}
		summary.FinalTrain = trainMetrics
		summary.EpochsCompleted = epoch
		summary.Workload.CompletedEpochs = epoch
		summary.Workload.ActualTrainPairs += int64(trainMetrics.BatchSize)
		summary.Workload.ActualTrainExamples += int64(len(indices))

		if len(evalSet) > 0 && epoch%cfg.EvalEveryEpoch == 0 {
			evalMetrics, improved, err := recordEval(epoch, "epoch")
			if err != nil {
				return EmbeddingTrainRunSummary{}, fmt.Errorf("epoch %d eval: %w", epoch, err)
			}
			record.Eval = evalMetrics
			record.Improved = improved
			if !improved && cfg.EarlyStoppingPatience > 0 && noImproveEvals >= cfg.EarlyStoppingPatience {
				summary.StoppedEarly = true
				summary.History = append(summary.History, record)
				break
			}
		}

		summary.History = append(summary.History, record)
	}

	summary.StepsCompleted = t.step
	summary.StepsRun = t.step - startStep
	preRestoreEndProfile := t.TrainProfile()
	restoreStartProfile := EmbeddingTrainProfile{}
	restored := false
	if cfg.RestoreBest && haveBest {
		if err := t.restoreCheckpoint(bestCheckpoint); err != nil {
			return EmbeddingTrainRunSummary{}, err
		}
		summary.RestoredBest = true
		restoreStartProfile = t.TrainProfile()
		restored = true
	}
	if len(evalSet) > 0 {
		evalStart := time.Now()
		evalPass := summary.Workload.ActualEvalPasses + 1
		maybeReportEvalProgress(cfg, "eval_start", summary.EpochsCompleted, t.step, evalPass, int64(len(evalSet)), int64(len(evalSet)), runStart)
		finalEval, err := t.EvaluatePairs(evalSet)
		if err != nil {
			return EmbeddingTrainRunSummary{}, fmt.Errorf("final eval: %w", err)
		}
		summary.EvalDuration += time.Since(evalStart)
		summary.Workload.ActualEvalPasses++
		summary.Workload.ActualEvalPairs += int64(len(evalSet))
		summary.Workload.ActualEvalExamples += int64(len(evalSet))
		summary.FinalEval = cloneEvalMetrics(finalEval)
		maybeReportEvalProgress(cfg, "eval_done", summary.EpochsCompleted, t.step, summary.Workload.ActualEvalPasses, summary.Workload.ActualEvalExamples, summary.Workload.ActualEvalPairs, runStart)
		appendTrainEvalHistory(&summary, summary.EpochsCompleted, t.step, "final", false, summary.FinalEval, nil, nil)
		if summary.BestEval == nil {
			summary.BestEval = cloneEvalMetrics(finalEval)
			if summary.BestEpoch == 0 {
				summary.BestEpoch = summary.EpochsCompleted
			}
			if summary.BestStep == 0 {
				summary.BestStep = summary.StepsCompleted
			}
		}
	}
	finalProfile := t.TrainProfile()
	if restored {
		preRestoreDelta := diffTrainProfile(summary.StartProfile, preRestoreEndProfile)
		postRestoreDelta := diffTrainProfile(restoreStartProfile, finalProfile)
		summary.DeltaProfile = addTrainProfileDelta(preRestoreDelta, postRestoreDelta)
		summary.EndProfile = applyTrainProfileDelta(preRestoreEndProfile, postRestoreDelta)
	} else {
		summary.EndProfile = finalProfile
		summary.DeltaProfile = diffTrainProfile(summary.StartProfile, summary.EndProfile)
	}
	summary.Workload.ActualTotalPairs = summary.Workload.ActualTrainPairs + summary.Workload.ActualEvalPairs
	summary.Workload.ActualTotalExamples = summary.Workload.ActualTrainExamples + summary.Workload.ActualEvalExamples
	summary.Elapsed = time.Since(runStart)
	return summary, nil
}

// FitScoreSpectrum trains over row-local ranked score-spectrum examples.
func (t *EmbeddingTrainer) FitScoreSpectrum(trainSet []EmbeddingScoreSpectrumExample, evalSet []EmbeddingPairExample, cfg EmbeddingTrainRunConfig) (EmbeddingTrainRunSummary, error) {
	if t == nil {
		return EmbeddingTrainRunSummary{}, fmt.Errorf("embedding trainer is not initialized")
	}
	cfg = normalizedTrainRunConfig(cfg)
	if err := validateClearTurboQuantPrefixRunConfig(cfg); err != nil {
		return EmbeddingTrainRunSummary{}, err
	}
	if err := validateClearTurboQuantRankMarginRunConfig(cfg); err != nil {
		return EmbeddingTrainRunSummary{}, err
	}
	if err := validateScoreSpectrumRunConfig(cfg); err != nil {
		return EmbeddingTrainRunSummary{}, err
	}
	scoreSpectrumEvalSet := cfg.ScoreSpectrumEval
	if cfg.EvalOnly {
		if len(evalSet) == 0 && len(scoreSpectrumEvalSet) == 0 {
			return EmbeddingTrainRunSummary{}, fmt.Errorf("eval dataset is empty")
		}
	} else {
		if len(trainSet) == 0 {
			return EmbeddingTrainRunSummary{}, fmt.Errorf("training dataset is empty")
		}
		if err := validateScoreSpectrumTrainSetResearchGates(trainSet, cfg.AllowResearchOnlyScoreSpectrum); err != nil {
			return EmbeddingTrainRunSummary{}, err
		}
		if cfg.Epochs <= 0 {
			return EmbeddingTrainRunSummary{}, fmt.Errorf("epochs must be positive")
		}
		if cfg.BatchSize <= 0 {
			return EmbeddingTrainRunSummary{}, fmt.Errorf("batch_size must be positive")
		}
	}
	if cfg.EvalEveryEpoch <= 0 {
		return EmbeddingTrainRunSummary{}, fmt.Errorf("eval_every_epoch must be positive")
	}
	if cfg.EvalEverySteps < 0 {
		return EmbeddingTrainRunSummary{}, fmt.Errorf("eval_every_steps must be non-negative")
	}
	if cfg.EarlyStoppingPatience < 0 {
		return EmbeddingTrainRunSummary{}, fmt.Errorf("early_stopping_patience must be non-negative")
	}
	if cfg.ProgressEverySteps < 0 {
		return EmbeddingTrainRunSummary{}, fmt.Errorf("progress_every_steps must be non-negative")
	}
	if !validTrainSelectionMetric(cfg.SelectMetric) {
		return EmbeddingTrainRunSummary{}, fmt.Errorf("unsupported select_metric %q", cfg.SelectMetric)
	}
	t.configureRetrievalEval(cfg.RetrievalEvalRuntime, cfg.RetrievalEval, cfg.RetrievalEvalTokenizer)
	if err := t.applyTrainRunOverrides(cfg); err != nil {
		return EmbeddingTrainRunSummary{}, err
	}
	cfg = t.syncTrainRunObjectiveConfig(cfg)
	cfg = t.isolateScoreSpectrumObjectiveConfig(cfg)
	if err := t.applyScoreSpectrumRunOverrides(cfg); err != nil {
		return EmbeddingTrainRunSummary{}, err
	}
	if err := validateScoreSpectrumRunConfig(cfg); err != nil {
		return EmbeddingTrainRunSummary{}, err
	}
	if err := validateScoreSpectrumTrainerConfig(t.config); err != nil {
		return EmbeddingTrainRunSummary{}, err
	}

	runStart := time.Now()
	startStep := t.step
	summary := EmbeddingTrainRunSummary{
		Config:                cfg,
		EffectiveLearningRate: t.config.LearningRate,
		StartProfile:          t.TrainProfile(),
		Workload:              estimateScoreSpectrumTrainWorkload(trainSet, len(evalSet), len(scoreSpectrumEvalSet), scoreSpectrumEvalWorkCount(evalSet, scoreSpectrumEvalSet), cfg),
	}
	if cfg.EvalOnly {
		evalStart := time.Now()
		evalExamples, evalPairs := scoreSpectrumEvalProgressWork(evalSet, scoreSpectrumEvalSet)
		maybeReportEvalProgress(cfg, "eval_start", 0, t.step, 1, evalExamples, evalPairs, runStart)
		if len(evalSet) > 0 {
			finalEval, err := t.EvaluatePairs(evalSet)
			if err != nil {
				return EmbeddingTrainRunSummary{}, fmt.Errorf("eval: %w", err)
			}
			summary.FinalEval = cloneEvalMetrics(finalEval)
			summary.LastEval = cloneEvalMetrics(finalEval)
			summary.BestEval = cloneEvalMetrics(finalEval)
		} else if retrievalEvalMetrics, err := t.retrievalOnlyEvalMetrics(true); err != nil {
			return EmbeddingTrainRunSummary{}, fmt.Errorf("retrieval eval: %w", err)
		} else if retrievalEvalMetrics != nil {
			summary.FinalEval = cloneEvalMetrics(*retrievalEvalMetrics)
			summary.LastEval = cloneEvalMetrics(*retrievalEvalMetrics)
			summary.BestEval = cloneEvalMetrics(*retrievalEvalMetrics)
		}
		if len(scoreSpectrumEvalSet) > 0 {
			finalScoreEval, err := t.EvaluateScoreSpectrumBatched(scoreSpectrumEvalSet, cfg.BatchSize)
			if err != nil {
				return EmbeddingTrainRunSummary{}, fmt.Errorf("score-spectrum eval: %w", err)
			}
			summary.FinalScoreSpectrumEval = cloneScoreSpectrumEvalMetrics(finalScoreEval)
			summary.LastScoreSpectrumEval = cloneScoreSpectrumEvalMetrics(finalScoreEval)
			summary.BestScoreSpectrumEval = cloneScoreSpectrumEvalMetrics(finalScoreEval)
		}
		summary.EvalDuration = time.Since(evalStart)
		summary.StepsCompleted = t.step
		summary.BestStep = t.step
		summary.Workload.ActualEvalPasses = 1
		summary.Workload.ActualEvalPairs = evalPairs
		summary.Workload.ActualEvalExamples = evalExamples
		maybeReportEvalProgress(cfg, "eval_done", 0, t.step, summary.Workload.ActualEvalPasses, summary.Workload.ActualEvalExamples, summary.Workload.ActualEvalPairs, runStart)
		appendTrainEvalHistory(&summary, 0, t.step, "eval_only", true, summary.FinalEval, summary.FinalScoreSpectrumEval, nil)
		summary.EndProfile = t.TrainProfile()
		summary.DeltaProfile = diffTrainProfile(summary.StartProfile, summary.EndProfile)
		summary.Workload.ActualTotalPairs = summary.Workload.ActualEvalPairs
		summary.Workload.ActualTotalExamples = summary.Workload.ActualEvalExamples
		summary.Elapsed = time.Since(runStart)
		return summary, nil
	}

	indices := make([]int, len(trainSet))
	for i := range indices {
		indices[i] = i
	}
	rng := rand.New(rand.NewSource(cfg.Seed))
	var (
		bestCheckpoint EmbeddingTrainCheckpoint
		haveBest       bool
		noImproveEvals int
	)
	recordEval := func(epoch int, trigger string) (*EmbeddingEvalMetrics, *EmbeddingScoreSpectrumEvalMetrics, bool, error) {
		evalStart := time.Now()
		evalPass := summary.Workload.ActualEvalPasses + 1
		evalExamples, evalPairs := scoreSpectrumEvalProgressWork(evalSet, scoreSpectrumEvalSet)
		maybeReportEvalProgress(cfg, "eval_start", epoch, t.step, evalPass, evalExamples, evalPairs, runStart)
		var evalMetrics *EmbeddingEvalMetrics
		if len(evalSet) > 0 {
			metrics, err := t.EvaluatePairs(evalSet)
			if err != nil {
				return nil, nil, false, err
			}
			evalMetrics = cloneEvalMetrics(metrics)
			summary.LastEval = cloneEvalMetrics(metrics)
		} else if retrievalEvalMetrics, err := t.retrievalOnlyEvalMetrics(true); err != nil {
			return nil, nil, false, err
		} else if retrievalEvalMetrics != nil {
			evalMetrics = retrievalEvalMetrics
			summary.LastEval = cloneEvalMetrics(*retrievalEvalMetrics)
		}
		var scoreEvalMetrics *EmbeddingScoreSpectrumEvalMetrics
		if len(scoreSpectrumEvalSet) > 0 {
			metrics, err := t.EvaluateScoreSpectrumBatched(scoreSpectrumEvalSet, cfg.BatchSize)
			if err != nil {
				return nil, nil, false, err
			}
			scoreEvalMetrics = cloneScoreSpectrumEvalMetrics(metrics)
			summary.LastScoreSpectrumEval = cloneScoreSpectrumEvalMetrics(metrics)
		}
		summary.EvalDuration += time.Since(evalStart)
		summary.Workload.ActualEvalPasses++
		summary.Workload.ActualEvalPairs += evalPairs
		summary.Workload.ActualEvalExamples += evalExamples
		maybeReportEvalProgress(cfg, "eval_done", epoch, t.step, summary.Workload.ActualEvalPasses, summary.Workload.ActualEvalExamples, summary.Workload.ActualEvalPairs, runStart)
		improved := false
		if scoreSpectrumSelectionMetric(cfg.SelectMetric) {
			if scoreEvalMetrics == nil {
				return nil, nil, false, fmt.Errorf("select_metric %q requires score-spectrum eval data", cfg.SelectMetric)
			}
			improved = !haveBest || betterScoreSpectrumEvalMetrics(*scoreEvalMetrics, *summary.BestScoreSpectrumEval, cfg.SelectMetric, cfg.MinDelta)
		} else {
			if evalMetrics == nil || (len(evalSet) == 0 && !retrievalSelectionMetric(cfg.SelectMetric)) {
				return nil, nil, false, fmt.Errorf("select_metric %q requires pairwise eval data", cfg.SelectMetric)
			}
			improved = !haveBest || betterEvalMetrics(*evalMetrics, *summary.BestEval, cfg.SelectMetric, cfg.MinDelta)
		}
		if improved {
			var err error
			bestCheckpoint, err = t.Checkpoint()
			if err != nil {
				return nil, nil, false, err
			}
			haveBest = true
			if evalMetrics != nil {
				summary.BestEval = cloneEvalMetrics(*evalMetrics)
			}
			if scoreEvalMetrics != nil {
				summary.BestScoreSpectrumEval = cloneScoreSpectrumEvalMetrics(*scoreEvalMetrics)
			}
			summary.BestEpoch = epoch
			summary.BestStep = t.step
			noImproveEvals = 0
		} else {
			noImproveEvals++
		}
		appendTrainEvalHistory(&summary, epoch, t.step, trigger, improved, evalMetrics, scoreEvalMetrics, nil)
		return evalMetrics, scoreEvalMetrics, improved, nil
	}
	if (len(evalSet) > 0 || len(scoreSpectrumEvalSet) > 0) && cfg.RestoreBest {
		if _, _, _, err := recordEval(0, "initial"); err != nil {
			return EmbeddingTrainRunSummary{}, fmt.Errorf("initial eval: %w", err)
		}
	}

	for epoch := 1; epoch <= cfg.Epochs; epoch++ {
		if cfg.Shuffle {
			rng.Shuffle(len(indices), func(i, j int) {
				indices[i], indices[j] = indices[j], indices[i]
			})
		}
		trainStart := time.Now()
		var afterBatch contrastiveEpochBatchHook
		if (len(evalSet) > 0 || len(scoreSpectrumEvalSet) > 0) && cfg.EvalEverySteps > 0 {
			afterBatch = func(progress EmbeddingTrainProgress) error {
				if progress.Batch <= 0 || progress.Batch%cfg.EvalEverySteps != 0 {
					return nil
				}
				if _, _, _, err := recordEval(epoch, "step"); err != nil {
					return fmt.Errorf("step %d eval: %w", progress.Step, err)
				}
				return nil
			}
		}
		trainMetrics, err := t.runScoreSpectrumEpoch(trainSet, indices, cfg.BatchSize, cfg, epoch, runStart, afterBatch)
		if err != nil {
			return EmbeddingTrainRunSummary{}, fmt.Errorf("epoch %d: %w", epoch, err)
		}
		summary.TrainDuration += time.Since(trainStart)
		record := EmbeddingTrainEpochSummary{
			Epoch: epoch,
			Step:  t.step,
			Train: trainMetrics,
		}
		summary.FinalTrain = trainMetrics
		summary.EpochsCompleted = epoch
		summary.Workload.CompletedEpochs = epoch
		_, epochPairs := scoreSpectrumBatchWork(scoreSpectrumOrderedExamples(trainSet, indices), cfg.BatchSize)
		summary.Workload.ActualTrainPairs += epochPairs
		summary.Workload.ActualTrainExamples += int64(len(indices))

		if (len(evalSet) > 0 || len(scoreSpectrumEvalSet) > 0) && epoch%cfg.EvalEveryEpoch == 0 {
			evalMetrics, scoreEvalMetrics, improved, err := recordEval(epoch, "epoch")
			if err != nil {
				return EmbeddingTrainRunSummary{}, fmt.Errorf("epoch %d eval: %w", epoch, err)
			}
			record.Eval = evalMetrics
			record.ScoreSpectrumEval = scoreEvalMetrics
			record.Improved = improved
			if !improved && cfg.EarlyStoppingPatience > 0 && noImproveEvals >= cfg.EarlyStoppingPatience {
				summary.StoppedEarly = true
				summary.History = append(summary.History, record)
				break
			}
		}

		summary.History = append(summary.History, record)
	}

	summary.StepsCompleted = t.step
	summary.StepsRun = t.step - startStep
	preRestoreEndProfile := t.TrainProfile()
	restoreStartProfile := EmbeddingTrainProfile{}
	restored := false
	if cfg.RestoreBest && haveBest {
		if err := t.restoreCheckpoint(bestCheckpoint); err != nil {
			return EmbeddingTrainRunSummary{}, err
		}
		summary.RestoredBest = true
		restoreStartProfile = t.TrainProfile()
		restored = true
	}
	if len(evalSet) > 0 || len(scoreSpectrumEvalSet) > 0 {
		evalStart := time.Now()
		evalPass := summary.Workload.ActualEvalPasses + 1
		evalExamples, evalPairs := scoreSpectrumEvalProgressWork(evalSet, scoreSpectrumEvalSet)
		maybeReportEvalProgress(cfg, "eval_start", summary.EpochsCompleted, t.step, evalPass, evalExamples, evalPairs, runStart)
		if len(evalSet) > 0 {
			finalEval, err := t.EvaluatePairs(evalSet)
			if err != nil {
				return EmbeddingTrainRunSummary{}, fmt.Errorf("final eval: %w", err)
			}
			summary.FinalEval = cloneEvalMetrics(finalEval)
		} else if retrievalEvalMetrics, err := t.retrievalOnlyEvalMetrics(true); err != nil {
			return EmbeddingTrainRunSummary{}, fmt.Errorf("final retrieval eval: %w", err)
		} else if retrievalEvalMetrics != nil {
			summary.FinalEval = cloneEvalMetrics(*retrievalEvalMetrics)
		}
		if len(scoreSpectrumEvalSet) > 0 {
			finalScoreEval, err := t.EvaluateScoreSpectrumBatched(scoreSpectrumEvalSet, cfg.BatchSize)
			if err != nil {
				return EmbeddingTrainRunSummary{}, fmt.Errorf("final score-spectrum eval: %w", err)
			}
			summary.FinalScoreSpectrumEval = cloneScoreSpectrumEvalMetrics(finalScoreEval)
		}
		summary.EvalDuration += time.Since(evalStart)
		summary.Workload.ActualEvalPasses++
		summary.Workload.ActualEvalPairs += evalPairs
		summary.Workload.ActualEvalExamples += evalExamples
		maybeReportEvalProgress(cfg, "eval_done", summary.EpochsCompleted, t.step, summary.Workload.ActualEvalPasses, summary.Workload.ActualEvalExamples, summary.Workload.ActualEvalPairs, runStart)
		appendTrainEvalHistory(&summary, summary.EpochsCompleted, t.step, "final", false, summary.FinalEval, summary.FinalScoreSpectrumEval, nil)
		if summary.FinalEval != nil && summary.BestEval == nil {
			summary.BestEval = cloneEvalMetrics(*summary.FinalEval)
		}
		if summary.FinalScoreSpectrumEval != nil && summary.BestScoreSpectrumEval == nil {
			summary.BestScoreSpectrumEval = cloneScoreSpectrumEvalMetrics(*summary.FinalScoreSpectrumEval)
		}
		if summary.BestEval != nil || summary.BestScoreSpectrumEval != nil {
			if summary.BestEpoch == 0 {
				summary.BestEpoch = summary.EpochsCompleted
			}
			if summary.BestStep == 0 {
				summary.BestStep = summary.StepsCompleted
			}
		}
	}
	finalProfile := t.TrainProfile()
	if restored {
		preRestoreDelta := diffTrainProfile(summary.StartProfile, preRestoreEndProfile)
		postRestoreDelta := diffTrainProfile(restoreStartProfile, finalProfile)
		summary.DeltaProfile = addTrainProfileDelta(preRestoreDelta, postRestoreDelta)
		summary.EndProfile = applyTrainProfileDelta(preRestoreEndProfile, postRestoreDelta)
	} else {
		summary.EndProfile = finalProfile
		summary.DeltaProfile = diffTrainProfile(summary.StartProfile, summary.EndProfile)
	}
	summary.Workload.ActualTotalPairs = summary.Workload.ActualTrainPairs + summary.Workload.ActualEvalPairs
	summary.Workload.ActualTotalExamples = summary.Workload.ActualTrainExamples + summary.Workload.ActualEvalExamples
	summary.Elapsed = time.Since(runStart)
	return summary, nil
}

// FitListwiseGeometry trains over tokenized listwise query-document geometry batches.
func (t *EmbeddingTrainer) FitListwiseGeometry(trainSet []EmbeddingTokenizedListwiseGeometryBatch, evalSet []EmbeddingPairExample, cfg EmbeddingTrainRunConfig) (EmbeddingTrainRunSummary, error) {
	if t == nil {
		return EmbeddingTrainRunSummary{}, fmt.Errorf("embedding trainer is not initialized")
	}
	cfg = normalizedTrainRunConfig(cfg)
	listwiseEvalSet := cfg.ListwiseGeometryEval
	if cfg.EvalOnly {
		if len(evalSet) == 0 && len(listwiseEvalSet) == 0 {
			return EmbeddingTrainRunSummary{}, fmt.Errorf("eval dataset is empty")
		}
	} else {
		if len(trainSet) == 0 {
			return EmbeddingTrainRunSummary{}, fmt.Errorf("training dataset is empty")
		}
		if cfg.Epochs <= 0 {
			return EmbeddingTrainRunSummary{}, fmt.Errorf("epochs must be positive")
		}
		if cfg.BatchSize <= 0 {
			return EmbeddingTrainRunSummary{}, fmt.Errorf("batch_size must be positive")
		}
		if err := validateListwiseGeometryTrainSetResearchGates(trainSet, cfg.AllowResearchOnlyListwiseGeometry); err != nil {
			return EmbeddingTrainRunSummary{}, err
		}
	}
	if cfg.EvalEveryEpoch <= 0 {
		return EmbeddingTrainRunSummary{}, fmt.Errorf("eval_every_epoch must be positive")
	}
	if cfg.EvalEverySteps < 0 {
		return EmbeddingTrainRunSummary{}, fmt.Errorf("eval_every_steps must be non-negative")
	}
	if cfg.EarlyStoppingPatience < 0 {
		return EmbeddingTrainRunSummary{}, fmt.Errorf("early_stopping_patience must be non-negative")
	}
	if cfg.ProgressEverySteps < 0 {
		return EmbeddingTrainRunSummary{}, fmt.Errorf("progress_every_steps must be non-negative")
	}
	if !validTrainSelectionMetric(cfg.SelectMetric) {
		return EmbeddingTrainRunSummary{}, fmt.Errorf("unsupported select_metric %q", cfg.SelectMetric)
	}
	if err := validateListwiseGeometryRunConfig(cfg); err != nil {
		return EmbeddingTrainRunSummary{}, err
	}
	t.configureRetrievalEval(cfg.RetrievalEvalRuntime, cfg.RetrievalEval, cfg.RetrievalEvalTokenizer)
	if err := t.applyTrainRunOverrides(cfg); err != nil {
		return EmbeddingTrainRunSummary{}, err
	}
	cfg = t.syncTrainRunObjectiveConfig(cfg)
	cfg = t.isolateListwiseGeometryObjectiveConfig(cfg)
	if err := validateListwiseGeometryRunConfig(cfg); err != nil {
		return EmbeddingTrainRunSummary{}, err
	}
	if err := validateListwiseGeometryTrainerConfig(t.config); err != nil {
		return EmbeddingTrainRunSummary{}, err
	}

	runStart := time.Now()
	startStep := t.step
	summary := EmbeddingTrainRunSummary{
		Config:                cfg,
		EffectiveLearningRate: t.config.LearningRate,
		StartProfile:          t.TrainProfile(),
		Workload:              estimateListwiseGeometryTrainWorkload(trainSet, len(evalSet), listwiseGeometryQueryCount(listwiseEvalSet), len(evalSet)+listwiseGeometryQueryCount(listwiseEvalSet), len(evalSet)+listwiseGeometryCellCount(listwiseEvalSet), cfg),
	}
	if err := validateListwiseGeometryWorkloadLimits(summary.Workload, cfg); err != nil {
		return EmbeddingTrainRunSummary{}, err
	}
	if cfg.EvalOnly {
		evalStart := time.Now()
		evalExamples, evalPairs := listwiseGeometryEvalProgressWork(evalSet, listwiseEvalSet)
		maybeReportEvalProgress(cfg, "eval_start", 0, t.step, 1, evalExamples, evalPairs, runStart)
		if len(evalSet) > 0 {
			finalEval, err := t.EvaluatePairs(evalSet)
			if err != nil {
				return EmbeddingTrainRunSummary{}, fmt.Errorf("eval: %w", err)
			}
			summary.FinalEval = cloneEvalMetrics(finalEval)
			summary.LastEval = cloneEvalMetrics(finalEval)
			summary.BestEval = cloneEvalMetrics(finalEval)
		} else if retrievalEvalMetrics, err := t.retrievalOnlyEvalMetrics(true); err != nil {
			return EmbeddingTrainRunSummary{}, fmt.Errorf("retrieval eval: %w", err)
		} else if retrievalEvalMetrics != nil {
			summary.FinalEval = cloneEvalMetrics(*retrievalEvalMetrics)
			summary.LastEval = cloneEvalMetrics(*retrievalEvalMetrics)
			summary.BestEval = cloneEvalMetrics(*retrievalEvalMetrics)
		}
		if len(listwiseEvalSet) > 0 {
			finalListwiseEval, err := t.EvaluateListwiseGeometryBatched(listwiseEvalSet, cfg.BatchSize)
			if err != nil {
				return EmbeddingTrainRunSummary{}, fmt.Errorf("listwise geometry eval: %w", err)
			}
			summary.FinalListwiseGeometryEval = cloneListwiseGeometryEvalMetrics(finalListwiseEval)
			summary.LastListwiseGeometryEval = cloneListwiseGeometryEvalMetrics(finalListwiseEval)
			summary.BestListwiseGeometryEval = cloneListwiseGeometryEvalMetrics(finalListwiseEval)
		}
		summary.EvalDuration = time.Since(evalStart)
		summary.StepsCompleted = t.step
		summary.BestStep = t.step
		summary.Workload.ActualEvalPasses = 1
		summary.Workload.ActualEvalPairs = evalPairs
		summary.Workload.ActualEvalExamples = evalExamples
		maybeReportEvalProgress(cfg, "eval_done", 0, t.step, summary.Workload.ActualEvalPasses, summary.Workload.ActualEvalExamples, summary.Workload.ActualEvalPairs, runStart)
		appendTrainEvalHistory(&summary, 0, t.step, "eval_only", true, summary.FinalEval, nil, summary.FinalListwiseGeometryEval)
		summary.EndProfile = t.TrainProfile()
		summary.DeltaProfile = diffTrainProfile(summary.StartProfile, summary.EndProfile)
		summary.Workload.ActualTotalPairs = summary.Workload.ActualEvalPairs
		summary.Workload.ActualTotalExamples = summary.Workload.ActualEvalExamples
		summary.Elapsed = time.Since(runStart)
		return summary, nil
	}

	indices := make([]int, len(trainSet))
	for i := range indices {
		indices[i] = i
	}
	rng := rand.New(rand.NewSource(cfg.Seed))
	var (
		bestCheckpoint EmbeddingTrainCheckpoint
		haveBest       bool
		noImproveEvals int
	)
	recordEval := func(epoch int, trigger string) (*EmbeddingEvalMetrics, *EmbeddingListwiseGeometryEvalMetrics, bool, error) {
		evalStart := time.Now()
		evalPass := summary.Workload.ActualEvalPasses + 1
		evalExamples, evalPairs := listwiseGeometryEvalProgressWork(evalSet, listwiseEvalSet)
		maybeReportEvalProgress(cfg, "eval_start", epoch, t.step, evalPass, evalExamples, evalPairs, runStart)
		var evalMetrics *EmbeddingEvalMetrics
		if len(evalSet) > 0 {
			metrics, err := t.EvaluatePairs(evalSet)
			if err != nil {
				return nil, nil, false, err
			}
			evalMetrics = cloneEvalMetrics(metrics)
			summary.LastEval = cloneEvalMetrics(metrics)
		} else if retrievalEvalMetrics, err := t.retrievalOnlyEvalMetrics(true); err != nil {
			return nil, nil, false, err
		} else if retrievalEvalMetrics != nil {
			evalMetrics = retrievalEvalMetrics
			summary.LastEval = cloneEvalMetrics(*retrievalEvalMetrics)
		}
		var listwiseEvalMetrics *EmbeddingListwiseGeometryEvalMetrics
		if len(listwiseEvalSet) > 0 {
			metrics, err := t.EvaluateListwiseGeometryBatched(listwiseEvalSet, cfg.BatchSize)
			if err != nil {
				return nil, nil, false, err
			}
			listwiseEvalMetrics = cloneListwiseGeometryEvalMetrics(metrics)
			summary.LastListwiseGeometryEval = cloneListwiseGeometryEvalMetrics(metrics)
		}
		summary.EvalDuration += time.Since(evalStart)
		summary.Workload.ActualEvalPasses++
		summary.Workload.ActualEvalPairs += evalPairs
		summary.Workload.ActualEvalExamples += evalExamples
		maybeReportEvalProgress(cfg, "eval_done", epoch, t.step, summary.Workload.ActualEvalPasses, summary.Workload.ActualEvalExamples, summary.Workload.ActualEvalPairs, runStart)
		improved := false
		selectableEval := evalMetrics != nil && (len(evalSet) > 0 || retrievalSelectionMetric(cfg.SelectMetric))
		if selectableEval && (!haveBest || betterEvalMetrics(*evalMetrics, *summary.BestEval, cfg.SelectMetric, cfg.MinDelta)) {
			checkpoint, err := t.Checkpoint()
			if err != nil {
				return nil, nil, false, err
			}
			bestCheckpoint = checkpoint
			haveBest = true
			improved = true
			summary.BestEval = cloneEvalMetrics(*evalMetrics)
			if listwiseEvalMetrics != nil {
				summary.BestListwiseGeometryEval = cloneListwiseGeometryEvalMetrics(*listwiseEvalMetrics)
			}
			summary.BestEpoch = epoch
			summary.BestStep = t.step
			noImproveEvals = 0
		} else if selectableEval {
			noImproveEvals++
		}
		appendTrainEvalHistory(&summary, epoch, t.step, trigger, improved, evalMetrics, nil, listwiseEvalMetrics)
		return evalMetrics, listwiseEvalMetrics, improved, nil
	}
	if (len(evalSet) > 0 || (len(listwiseEvalSet) > 0 && retrievalSelectionMetric(cfg.SelectMetric) && t.retrievalEvalEnabled)) && cfg.RestoreBest {
		if _, _, _, err := recordEval(0, "initial"); err != nil {
			return EmbeddingTrainRunSummary{}, fmt.Errorf("initial eval: %w", err)
		}
	}

	for epoch := 1; epoch <= cfg.Epochs; epoch++ {
		if cfg.Shuffle {
			rng.Shuffle(len(indices), func(i, j int) {
				indices[i], indices[j] = indices[j], indices[i]
			})
		}
		trainStart := time.Now()
		var afterBatch contrastiveEpochBatchHook
		if (len(evalSet) > 0 || len(listwiseEvalSet) > 0) && cfg.EvalEverySteps > 0 {
			afterBatch = func(progress EmbeddingTrainProgress) error {
				if progress.Batch <= 0 || progress.Batch%cfg.EvalEverySteps != 0 {
					return nil
				}
				if _, _, _, err := recordEval(epoch, "step"); err != nil {
					return fmt.Errorf("step %d eval: %w", progress.Step, err)
				}
				return nil
			}
		}
		trainMetrics, err := t.runListwiseGeometryEpoch(trainSet, indices, cfg.BatchSize, cfg, epoch, runStart, afterBatch)
		if err != nil {
			return EmbeddingTrainRunSummary{}, fmt.Errorf("epoch %d: %w", epoch, err)
		}
		summary.TrainDuration += time.Since(trainStart)
		record := EmbeddingTrainEpochSummary{
			Epoch: epoch,
			Step:  t.step,
			Train: trainMetrics,
		}
		summary.FinalTrain = trainMetrics
		summary.EpochsCompleted = epoch
		summary.Workload.CompletedEpochs = epoch
		_, epochPairs := listwiseGeometryBatchWork(listwiseGeometryOrderedBatches(trainSet, indices), cfg.BatchSize)
		summary.Workload.ActualTrainPairs += epochPairs
		summary.Workload.ActualTrainExamples += int64(len(indices))
		if (len(evalSet) > 0 || len(listwiseEvalSet) > 0) && epoch%cfg.EvalEveryEpoch == 0 {
			evalMetrics, listwiseEvalMetrics, improved, err := recordEval(epoch, "epoch")
			if err != nil {
				return EmbeddingTrainRunSummary{}, fmt.Errorf("epoch %d eval: %w", epoch, err)
			}
			record.Eval = evalMetrics
			record.ListwiseGeometryEval = listwiseEvalMetrics
			record.Improved = improved
			if (len(evalSet) > 0 || (t.retrievalEvalEnabled && retrievalSelectionMetric(cfg.SelectMetric))) && !improved && cfg.EarlyStoppingPatience > 0 && noImproveEvals >= cfg.EarlyStoppingPatience {
				summary.StoppedEarly = true
				summary.History = append(summary.History, record)
				break
			}
		}
		summary.History = append(summary.History, record)
	}
	summary.StepsCompleted = t.step
	summary.StepsRun = t.step - startStep
	preRestoreEndProfile := t.TrainProfile()
	restoreStartProfile := EmbeddingTrainProfile{}
	restored := false
	if cfg.RestoreBest && haveBest {
		if err := t.restoreCheckpoint(bestCheckpoint); err != nil {
			return EmbeddingTrainRunSummary{}, err
		}
		summary.RestoredBest = true
		restoreStartProfile = t.TrainProfile()
		restored = true
	}
	if len(evalSet) > 0 || len(listwiseEvalSet) > 0 {
		evalStart := time.Now()
		evalPass := summary.Workload.ActualEvalPasses + 1
		evalExamples, evalPairs := listwiseGeometryEvalProgressWork(evalSet, listwiseEvalSet)
		maybeReportEvalProgress(cfg, "eval_start", summary.EpochsCompleted, t.step, evalPass, evalExamples, evalPairs, runStart)
		if len(evalSet) > 0 {
			finalEval, err := t.EvaluatePairs(evalSet)
			if err != nil {
				return EmbeddingTrainRunSummary{}, fmt.Errorf("final eval: %w", err)
			}
			summary.FinalEval = cloneEvalMetrics(finalEval)
		} else if retrievalEvalMetrics, err := t.retrievalOnlyEvalMetrics(true); err != nil {
			return EmbeddingTrainRunSummary{}, fmt.Errorf("final retrieval eval: %w", err)
		} else if retrievalEvalMetrics != nil {
			summary.FinalEval = cloneEvalMetrics(*retrievalEvalMetrics)
		}
		if len(listwiseEvalSet) > 0 {
			finalListwiseEval, err := t.EvaluateListwiseGeometryBatched(listwiseEvalSet, cfg.BatchSize)
			if err != nil {
				return EmbeddingTrainRunSummary{}, fmt.Errorf("final listwise geometry eval: %w", err)
			}
			summary.FinalListwiseGeometryEval = cloneListwiseGeometryEvalMetrics(finalListwiseEval)
		}
		summary.EvalDuration += time.Since(evalStart)
		summary.Workload.ActualEvalPasses++
		summary.Workload.ActualEvalPairs += evalPairs
		summary.Workload.ActualEvalExamples += evalExamples
		maybeReportEvalProgress(cfg, "eval_done", summary.EpochsCompleted, t.step, summary.Workload.ActualEvalPasses, summary.Workload.ActualEvalExamples, summary.Workload.ActualEvalPairs, runStart)
		appendTrainEvalHistory(&summary, summary.EpochsCompleted, t.step, "final", false, summary.FinalEval, nil, summary.FinalListwiseGeometryEval)
		if summary.BestEval == nil {
			if summary.FinalEval != nil {
				summary.BestEval = cloneEvalMetrics(*summary.FinalEval)
			}
			if summary.BestEpoch == 0 {
				summary.BestEpoch = summary.EpochsCompleted
			}
			if summary.BestStep == 0 {
				summary.BestStep = summary.StepsCompleted
			}
		}
		if summary.FinalListwiseGeometryEval != nil && summary.BestListwiseGeometryEval == nil {
			summary.BestListwiseGeometryEval = cloneListwiseGeometryEvalMetrics(*summary.FinalListwiseGeometryEval)
		}
	}
	finalProfile := t.TrainProfile()
	if restored {
		preRestoreDelta := diffTrainProfile(summary.StartProfile, preRestoreEndProfile)
		postRestoreDelta := diffTrainProfile(restoreStartProfile, finalProfile)
		summary.DeltaProfile = addTrainProfileDelta(preRestoreDelta, postRestoreDelta)
		summary.EndProfile = applyTrainProfileDelta(preRestoreEndProfile, postRestoreDelta)
	} else {
		summary.EndProfile = finalProfile
		summary.DeltaProfile = diffTrainProfile(summary.StartProfile, summary.EndProfile)
	}
	summary.Workload.ActualTotalPairs = summary.Workload.ActualTrainPairs + summary.Workload.ActualEvalPairs
	summary.Workload.ActualTotalExamples = summary.Workload.ActualTrainExamples + summary.Workload.ActualEvalExamples
	summary.Elapsed = time.Since(runStart)
	return summary, nil
}

// EstimatePairwiseTrainWorkload returns planned pairwise work for supervised pair training.
func EstimatePairwiseTrainWorkload(trainExamples, evalExamples int, cfg EmbeddingTrainRunConfig) EmbeddingTrainWorkload {
	cfg = normalizedTrainRunConfig(cfg)
	batches := batchCount(trainExamples, cfg.BatchSize, 1)
	evalPasses := plannedEvalPassCount(evalExamples, cfg.Epochs, cfg.EvalEveryEpoch)
	trainPairsPerEpoch := int64(trainExamples)
	evalPairsPerPass := int64(evalExamples)
	if cfg.EvalOnly {
		batches = 0
		trainPairsPerEpoch = 0
		if evalExamples > 0 {
			evalPasses = 1
		}
	}
	return EmbeddingTrainWorkload{
		TrainMode:            "pairwise",
		EvalMode:             workloadEvalMode(evalExamples, "pairwise"),
		TrainExamples:        trainExamples,
		EvalExamples:         evalExamples,
		BatchSize:            cfg.BatchSize,
		PlannedEpochs:        cfg.Epochs,
		TrainBatchesPerEpoch: batches,
		TrainPairsPerEpoch:   trainPairsPerEpoch,
		EvalPairsPerPass:     evalPairsPerPass,
		PlannedEvalPasses:    evalPasses,
		PlannedTrainPairs:    trainPairsPerEpoch * int64(cfg.Epochs),
		PlannedEvalPairs:     evalPairsPerPass * int64(evalPasses),
		PlannedTotalPairs:    trainPairsPerEpoch*int64(cfg.Epochs) + evalPairsPerPass*int64(evalPasses),
	}
}

// EstimateContrastiveTrainWorkload returns planned pairwise work for contrastive training with in-batch negatives.
// RetargetWorkloadToPairwiseEval rewrites the eval side of a contrastive
// workload estimate for runs whose evals score labeled pairs (cfg.EvalPairs)
// instead of in-batch contrastive examples.
func RetargetWorkloadToPairwiseEval(workload EmbeddingTrainWorkload, evalPairs int, cfg EmbeddingTrainRunConfig) EmbeddingTrainWorkload {
	cfg = normalizedTrainRunConfig(cfg)
	passes := plannedEvalPassCount(evalPairs, cfg.Epochs, cfg.EvalEveryEpoch)
	if cfg.RestoreBest && evalPairs > 0 {
		passes++
	}
	if cfg.EvalEverySteps > 0 && evalPairs > 0 {
		passes += (workload.TrainBatchesPerEpoch / cfg.EvalEverySteps) * cfg.Epochs
	}
	workload.EvalMode = workloadEvalMode(evalPairs, "pairwise")
	workload.EvalExamples = evalPairs
	workload.EvalPairsPerPass = int64(evalPairs)
	workload.PlannedEvalPasses = passes
	workload.PlannedEvalPairs = int64(evalPairs) * int64(passes)
	workload.PlannedTotalPairs = workload.PlannedTrainPairs + workload.PlannedEvalPairs
	return workload
}

func EstimateContrastiveTrainWorkload(trainExamples, evalExamples int, cfg EmbeddingTrainRunConfig) EmbeddingTrainWorkload {
	cfg = normalizedTrainRunConfig(cfg)
	batches, trainPairsPerEpoch := contrastiveBatchWork(trainExamples, cfg.BatchSize)
	trainPairsPerEpoch *= int64(compactPrefixObjectiveMultiplier(cfg))
	evalPairsPerPass := contrastiveEvalPairs(evalExamples)
	evalPasses := plannedEvalPassCount(evalExamples, cfg.Epochs, cfg.EvalEveryEpoch)
	if cfg.RestoreBest && evalExamples > 0 {
		evalPasses++
	}
	if cfg.EvalEverySteps > 0 && evalExamples > 0 {
		evalPasses += (batches / cfg.EvalEverySteps) * cfg.Epochs
	}
	if cfg.EvalOnly {
		batches = 0
		trainPairsPerEpoch = 0
		if evalExamples > 0 {
			evalPasses = 1
		}
	}
	return EmbeddingTrainWorkload{
		TrainMode:            "contrastive",
		EvalMode:             workloadEvalMode(evalExamples, "contrastive"),
		TrainExamples:        trainExamples,
		EvalExamples:         evalExamples,
		BatchSize:            cfg.BatchSize,
		PlannedEpochs:        cfg.Epochs,
		TrainBatchesPerEpoch: batches,
		TrainPairsPerEpoch:   trainPairsPerEpoch,
		EvalPairsPerPass:     evalPairsPerPass,
		PlannedEvalPasses:    evalPasses,
		PlannedTrainPairs:    trainPairsPerEpoch * int64(cfg.Epochs),
		PlannedEvalPairs:     evalPairsPerPass * int64(evalPasses),
		PlannedTotalPairs:    trainPairsPerEpoch*int64(cfg.Epochs) + evalPairsPerPass*int64(evalPasses),
	}
}

// EstimateHardNegativeTrainWorkload returns planned work for explicit hard-negative contrastive training.
func EstimateHardNegativeTrainWorkload(trainExamples, negativesPerExample, evalExamples int, cfg EmbeddingTrainRunConfig) EmbeddingTrainWorkload {
	cfg = normalizedTrainRunConfig(cfg)
	if negativesPerExample < 0 {
		negativesPerExample = 0
	}
	batches, trainPairsPerEpoch := hardNegativeBatchWork(trainExamples, cfg.BatchSize, negativesPerExample, cfg.ContrastiveLoss, cfg.TeacherLossWeight, compactPrefixObjectiveMultiplier(cfg), turboQuantRankMarginObjectiveCount(cfg.TurboQuantRankMarginObjectives), turboQuantRankMarginObjectiveCount(cfg.TurboQuantCompactObjectives))
	evalPasses := plannedEvalPassCount(evalExamples, cfg.Epochs, cfg.EvalEveryEpoch)
	if cfg.RestoreBest && evalExamples > 0 {
		evalPasses++
	}
	if cfg.EvalEverySteps > 0 && evalExamples > 0 {
		evalPasses += (batches / cfg.EvalEverySteps) * cfg.Epochs
	}
	if cfg.EvalOnly {
		batches = 0
		trainPairsPerEpoch = 0
		if evalExamples > 0 {
			evalPasses = 1
		}
	}
	evalPairsPerPass := int64(evalExamples)
	return EmbeddingTrainWorkload{
		TrainMode:            hardNegativeTrainMode(cfg.ContrastiveLoss),
		EvalMode:             workloadEvalMode(evalExamples, "pairwise"),
		TrainExamples:        trainExamples,
		EvalExamples:         evalExamples,
		BatchSize:            cfg.BatchSize,
		PlannedEpochs:        cfg.Epochs,
		TrainBatchesPerEpoch: batches,
		TrainPairsPerEpoch:   trainPairsPerEpoch,
		EvalPairsPerPass:     evalPairsPerPass,
		PlannedEvalPasses:    evalPasses,
		PlannedTrainPairs:    trainPairsPerEpoch * int64(cfg.Epochs),
		PlannedEvalPairs:     evalPairsPerPass * int64(evalPasses),
		PlannedTotalPairs:    trainPairsPerEpoch*int64(cfg.Epochs) + evalPairsPerPass*int64(evalPasses),
	}
}

// EstimateScoreSpectrumTrainWorkload returns planned row-local candidate scoring work.
func EstimateScoreSpectrumTrainWorkload(trainSet []EmbeddingScoreSpectrumExample, evalExamples int, cfg EmbeddingTrainRunConfig) EmbeddingTrainWorkload {
	scoreSpectrumEvalExamples := 0
	if len(cfg.ScoreSpectrumEval) > 0 || strings.TrimSpace(cfg.ScoreSpectrumEvalPath) != "" {
		scoreSpectrumEvalExamples = evalExamples
	}
	return estimateScoreSpectrumTrainWorkload(trainSet, len(cfg.EvalPairs), scoreSpectrumEvalExamples, evalExamples, cfg)
}

// EstimateListwiseGeometryTrainWorkload returns planned query-document matrix scoring work.
func EstimateListwiseGeometryTrainWorkload(trainSet []EmbeddingTokenizedListwiseGeometryBatch, evalExamples int, cfg EmbeddingTrainRunConfig) EmbeddingTrainWorkload {
	if len(cfg.ListwiseGeometryEval) > 0 {
		listwiseEvalExamples := listwiseGeometryQueryCount(cfg.ListwiseGeometryEval)
		evalPairs := evalExamples + listwiseGeometryCellCount(cfg.ListwiseGeometryEval)
		return estimateListwiseGeometryTrainWorkload(trainSet, evalExamples, listwiseEvalExamples, evalExamples+listwiseEvalExamples, evalPairs, cfg)
	}
	return estimateListwiseGeometryTrainWorkload(trainSet, evalExamples, 0, evalExamples, evalExamples, cfg)
}

func estimateListwiseGeometryTrainWorkload(trainSet []EmbeddingTokenizedListwiseGeometryBatch, pairwiseEvalExamples, listwiseEvalExamples, evalExamples, evalPairs int, cfg EmbeddingTrainRunConfig) EmbeddingTrainWorkload {
	cfg = normalizedTrainRunConfig(cfg)
	batches, trainPairsPerEpoch := listwiseGeometryBatchWork(trainSet, cfg.BatchSize)
	evalPasses := plannedEvalPassCount(evalExamples, cfg.Epochs, cfg.EvalEveryEpoch)
	if cfg.RestoreBest && evalExamples > 0 {
		evalPasses++
	}
	if cfg.EvalEverySteps > 0 && evalExamples > 0 {
		evalPasses += (batches / cfg.EvalEverySteps) * cfg.Epochs
	}
	if cfg.EvalOnly {
		batches = 0
		trainPairsPerEpoch = 0
		if evalExamples > 0 {
			evalPasses = 1
		}
	}
	evalPairsPerPass := int64(evalPairs)
	evalMode := workloadEvalMode(evalExamples, "pairwise")
	if listwiseEvalExamples > 0 {
		evalMode = "listwise_geometry"
		if pairwiseEvalExamples > 0 {
			evalMode = "mixed"
		}
	}
	return EmbeddingTrainWorkload{
		TrainMode:            "listwise_geometry",
		EvalMode:             evalMode,
		TrainExamples:        len(trainSet),
		EvalExamples:         evalExamples,
		BatchSize:            cfg.BatchSize,
		PlannedEpochs:        cfg.Epochs,
		TrainBatchesPerEpoch: batches,
		TrainPairsPerEpoch:   trainPairsPerEpoch,
		EvalPairsPerPass:     evalPairsPerPass,
		PlannedEvalPasses:    evalPasses,
		PlannedTrainPairs:    trainPairsPerEpoch * int64(cfg.Epochs),
		PlannedEvalPairs:     evalPairsPerPass * int64(evalPasses),
		PlannedTotalPairs:    trainPairsPerEpoch*int64(cfg.Epochs) + evalPairsPerPass*int64(evalPasses),
	}
}

func listwiseGeometryEvalProgressWork(evalSet []EmbeddingPairExample, listwiseEvalSet []EmbeddingTokenizedListwiseGeometryBatch) (int64, int64) {
	examples := int64(len(evalSet) + listwiseGeometryQueryCount(listwiseEvalSet))
	pairs := int64(len(evalSet) + listwiseGeometryCellCount(listwiseEvalSet))
	return examples, pairs
}

func listwiseGeometryQueryCount(batches []EmbeddingTokenizedListwiseGeometryBatch) int {
	total := 0
	for _, batch := range batches {
		total += len(batch.QueryTokens)
	}
	return total
}

func listwiseGeometryCellCount(batches []EmbeddingTokenizedListwiseGeometryBatch) int {
	total := 0
	for _, batch := range batches {
		total += len(batch.QueryTokens) * len(batch.DocumentTokens)
	}
	return total
}

func estimateScoreSpectrumTrainWorkload(trainSet []EmbeddingScoreSpectrumExample, pairwiseEvalExamples, scoreSpectrumEvalExamples, evalExamples int, cfg EmbeddingTrainRunConfig) EmbeddingTrainWorkload {
	cfg = normalizedTrainRunConfig(cfg)
	batches, trainPairsPerEpoch := scoreSpectrumBatchWork(trainSet, cfg.BatchSize)
	evalPasses := plannedEvalPassCount(evalExamples, cfg.Epochs, cfg.EvalEveryEpoch)
	if cfg.RestoreBest && evalExamples > 0 {
		evalPasses++
	}
	if cfg.EvalEverySteps > 0 && evalExamples > 0 {
		evalPasses += (batches / cfg.EvalEverySteps) * cfg.Epochs
	}
	if cfg.EvalOnly {
		batches = 0
		trainPairsPerEpoch = 0
		if evalExamples > 0 {
			evalPasses = 1
		}
	}
	evalPairsPerPass := int64(evalExamples)
	evalMode := workloadEvalMode(evalExamples, "pairwise")
	if scoreSpectrumEvalExamples > 0 {
		if evalExamples > 0 {
			evalMode = "score_spectrum_grouped"
			if pairwiseEvalExamples > 0 {
				evalMode = "mixed"
			}
		}
	}
	return EmbeddingTrainWorkload{
		TrainMode:            "score_spectrum_grouped",
		EvalMode:             evalMode,
		TrainExamples:        len(trainSet),
		EvalExamples:         evalExamples,
		BatchSize:            cfg.BatchSize,
		PlannedEpochs:        cfg.Epochs,
		TrainBatchesPerEpoch: batches,
		TrainPairsPerEpoch:   trainPairsPerEpoch,
		EvalPairsPerPass:     evalPairsPerPass,
		PlannedEvalPasses:    evalPasses,
		PlannedTrainPairs:    trainPairsPerEpoch * int64(cfg.Epochs),
		PlannedEvalPairs:     evalPairsPerPass * int64(evalPasses),
		PlannedTotalPairs:    trainPairsPerEpoch*int64(cfg.Epochs) + evalPairsPerPass*int64(evalPasses),
	}
}

func scoreSpectrumEvalWorkCount(evalSet []EmbeddingPairExample, scoreSpectrumEvalSet []EmbeddingScoreSpectrumExample) int {
	if len(scoreSpectrumEvalSet) == 0 {
		return len(evalSet)
	}
	return len(evalSet) + scoreSpectrumCandidateCount(scoreSpectrumEvalSet)
}

func scoreSpectrumEvalProgressWork(evalSet []EmbeddingPairExample, scoreSpectrumEvalSet []EmbeddingScoreSpectrumExample) (int64, int64) {
	examples := int64(len(evalSet) + len(scoreSpectrumEvalSet))
	pairs := int64(len(evalSet) + scoreSpectrumCandidateCount(scoreSpectrumEvalSet))
	return examples, pairs
}

func scoreSpectrumCandidateCount(examples []EmbeddingScoreSpectrumExample) int {
	total := 0
	for _, example := range examples {
		total += len(example.CandidateTokens)
	}
	return total
}

func batchCount(total, batchSize, minBatch int) int {
	if total <= 0 || batchSize <= 0 {
		return 0
	}
	count := 0
	for start := 0; start < total; start += batchSize {
		end := start + batchSize
		if end > total {
			end = total
		}
		if end-start < minBatch {
			break
		}
		count++
	}
	return count
}

func contrastiveUsableExampleCount(total, batchSize int) int {
	if total <= 0 || batchSize <= 1 {
		return 0
	}
	used := 0
	for start := 0; start < total; start += batchSize {
		end := start + batchSize
		if end > total {
			end = total
		}
		if end-start < 2 {
			break
		}
		used += end - start
	}
	return used
}

func contrastiveBatchWork(total, batchSize int) (int, int64) {
	if total <= 0 || batchSize <= 1 {
		return 0, 0
	}
	var pairs int64
	batches := 0
	for start := 0; start < total; start += batchSize {
		end := start + batchSize
		if end > total {
			end = total
		}
		n := end - start
		if n < 2 {
			break
		}
		batches++
		pairs += int64(n) * int64(n)
	}
	return batches, pairs
}

func contrastiveEvalPairs(total int) int64 {
	if total <= 0 {
		return 0
	}
	return int64(total) * int64(total)
}

func hardNegativeBatchWork(total, batchSize, negativesPerExample int, loss string, teacherLossWeight float32, matryoshkaMultiplier, rankMarginObjectives, compactObjectives int) (int, int64) {
	if total <= 0 || batchSize <= 0 {
		return 0, 0
	}
	if negativesPerExample < 0 {
		negativesPerExample = 0
	}
	if matryoshkaMultiplier <= 0 {
		matryoshkaMultiplier = 1
	}
	var pairs int64
	batches := 0
	candidatesPerExample := int64(1 + negativesPerExample)
	for start := 0; start < total; start += batchSize {
		end := start + batchSize
		if end > total {
			end = total
		}
		n := end - start
		if n <= 0 {
			break
		}
		candidates := int64(n) * candidatesPerExample
		basePairs := int64(0)
		if loss == "grouped_infonce" {
			if candidatesPerExample < 2 {
				break
			}
			batches++
			basePairs = int64(n) * candidatesPerExample
		} else if loss == "hybrid_infonce" {
			if candidates < 2 {
				break
			}
			batches++
			basePairs = int64(n) * candidates
			if candidatesPerExample >= 2 {
				basePairs += int64(n) * candidatesPerExample
			}
		} else {
			if candidates < 2 {
				break
			}
			batches++
			basePairs = int64(n) * candidates
		}
		pairs += basePairs * int64(matryoshkaMultiplier)
		if teacherLossWeight > 0 && candidatesPerExample >= 2 {
			pairs += int64(n) * candidatesPerExample
		}
		if rankMarginObjectives > 0 && candidatesPerExample >= 2 {
			pairs += int64(n) * int64(rankMarginObjectives)
		}
		if compactObjectives > 0 && candidatesPerExample >= 2 {
			pairs += int64(n) * candidatesPerExample * int64(compactObjectives)
		}
	}
	return batches, pairs
}

func hardNegativeTrainMode(loss string) string {
	switch loss {
	case "grouped_infonce":
		return "hard_negative_grouped_infonce"
	case "hybrid_infonce":
		return "hard_negative_hybrid_infonce"
	default:
		return "hard_negative_contrastive"
	}
}

func hardNegativeBatchPairCount(batch []EmbeddingHardNegativeExample) int64 {
	if len(batch) == 0 {
		return 0
	}
	candidates := 0
	for _, example := range batch {
		candidates += 1 + len(example.NegativeTokens)
	}
	return int64(len(batch)) * int64(candidates)
}

func scoreSpectrumBatchWork(trainSet []EmbeddingScoreSpectrumExample, batchSize int) (int, int64) {
	if len(trainSet) == 0 || batchSize <= 0 {
		return 0, 0
	}
	var pairs int64
	batches := 0
	for start := 0; start < len(trainSet); start += batchSize {
		end := start + batchSize
		if end > len(trainSet) {
			end = len(trainSet)
		}
		batchPairs := scoreSpectrumBatchPairCount(trainSet[start:end])
		if batchPairs <= 0 {
			break
		}
		batches++
		pairs += batchPairs
	}
	return batches, pairs
}

func scoreSpectrumBatchPairCount(batch []EmbeddingScoreSpectrumExample) int64 {
	var pairs int64
	for _, example := range batch {
		pairs += int64(len(example.CandidateTokens))
	}
	return pairs
}

func listwiseGeometryBatchWork(trainSet []EmbeddingTokenizedListwiseGeometryBatch, batchSize int) (int, int64) {
	if len(trainSet) == 0 || batchSize <= 0 {
		return 0, 0
	}
	var pairs int64
	batches := 0
	for start := 0; start < len(trainSet); start += batchSize {
		end := start + batchSize
		if end > len(trainSet) {
			end = len(trainSet)
		}
		batchPairs := listwiseGeometryBatchPairCount(trainSet[start:end])
		if batchPairs <= 0 {
			break
		}
		batches++
		pairs += batchPairs
	}
	return batches, pairs
}

func listwiseGeometryBatchPairCount(batch []EmbeddingTokenizedListwiseGeometryBatch) int64 {
	var pairs int64
	for _, row := range batch {
		pairs += int64(len(row.QueryTokens)) * int64(len(row.DocumentTokens))
	}
	return pairs
}

func listwiseGeometryBatchQueryCount(batch []EmbeddingTokenizedListwiseGeometryBatch) int {
	total := 0
	for _, row := range batch {
		total += len(row.QueryTokens)
	}
	return total
}

func scoreSpectrumOrderedExamples(trainSet []EmbeddingScoreSpectrumExample, order []int) []EmbeddingScoreSpectrumExample {
	ordered := make([]EmbeddingScoreSpectrumExample, 0, len(order))
	for _, idx := range order {
		ordered = append(ordered, trainSet[idx])
	}
	return ordered
}

func listwiseGeometryOrderedBatches(trainSet []EmbeddingTokenizedListwiseGeometryBatch, order []int) []EmbeddingTokenizedListwiseGeometryBatch {
	ordered := make([]EmbeddingTokenizedListwiseGeometryBatch, 0, len(order))
	for _, idx := range order {
		ordered = append(ordered, trainSet[idx])
	}
	return ordered
}

func validateScoreSpectrumTrainSetResearchGates(trainSet []EmbeddingScoreSpectrumExample, allowResearchOnly bool) error {
	for i, example := range trainSet {
		if err := validateTokenizedScoreSpectrumShape(example); err != nil {
			return fmt.Errorf("score-spectrum example %d: %w", i, err)
		}
		researchOnly := example.TrainAllowedForResearch && !example.ReleaseTrainAllowed && !example.CommercialUseAllowed
		if researchOnly && !allowResearchOnly {
			return fmt.Errorf("score-spectrum example %d is research-only; set AllowResearchOnlyScoreSpectrum to train it", i)
		}
	}
	return nil
}

func validateListwiseGeometryTrainSetResearchGates(trainSet []EmbeddingTokenizedListwiseGeometryBatch, allowResearchOnly bool) error {
	for i, batch := range trainSet {
		if err := validateTokenizedListwiseGeometryBatch(batch); err != nil {
			return fmt.Errorf("listwise geometry batch %d: %w", i, err)
		}
		researchOnly := batch.TrainAllowedForResearch && !batch.ReleaseTrainAllowed && !batch.CommercialUseAllowed
		if allowResearchOnly && !researchOnly {
			return fmt.Errorf("listwise geometry batch %d must be explicitly research-only when AllowResearchOnlyListwiseGeometry is set", i)
		}
		if researchOnly && !allowResearchOnly {
			return fmt.Errorf("listwise geometry batch %d is research-only; set AllowResearchOnlyListwiseGeometry to train it", i)
		}
	}
	return nil
}

func spreadHardNegativeOrderByQuery(trainSet []EmbeddingHardNegativeExample, order []int) []int {
	if len(trainSet) == 0 || len(order) < 2 {
		return order
	}
	type queryBucket struct {
		indexes []int
		next    int
	}
	buckets := []queryBucket{}
	bucketByKey := map[string]int{}
	for _, idx := range order {
		if idx < 0 || idx >= len(trainSet) {
			return order
		}
		example := trainSet[idx]
		key := embeddingBatchSequenceKey(example.QueryTokens, example.QueryMask, 0)
		bucketIndex, ok := bucketByKey[key]
		if !ok {
			bucketIndex = len(buckets)
			bucketByKey[key] = bucketIndex
			buckets = append(buckets, queryBucket{})
		}
		buckets[bucketIndex].indexes = append(buckets[bucketIndex].indexes, idx)
	}
	if len(buckets) == len(order) {
		return order
	}
	active := make([]int, 0, len(buckets))
	for i := range buckets {
		active = append(active, i)
	}
	out := make([]int, 0, len(order))
	for len(active) > 0 {
		nextActive := make([]int, 0, len(active))
		for _, bucketIndex := range active {
			bucket := &buckets[bucketIndex]
			out = append(out, bucket.indexes[bucket.next])
			bucket.next++
			if bucket.next < len(bucket.indexes) {
				nextActive = append(nextActive, bucketIndex)
			}
		}
		active = nextActive
	}
	if len(out) != len(order) {
		return order
	}
	return out
}

func hardNegativeSourceWeightedOrder(trainSet []EmbeddingHardNegativeExample, order []int, batchSize int, weights map[string]int, lengthBucket bool) []int {
	weights = normalizeHardNegativeSourceWeights(weights)
	if len(trainSet) == 0 || len(order) < 2 || batchSize <= 0 || len(weights) == 0 {
		return order
	}
	type sourceQueue struct {
		indexes []int
		next    int
		weight  int
	}
	groups := []sourceQueue{}
	groupByKey := map[string]int{}
	for _, idx := range order {
		if idx < 0 || idx >= len(trainSet) {
			return order
		}
		key := hardNegativeSourceGroupKey(weights, trainSet[idx].Source)
		groupIndex, ok := groupByKey[key]
		if !ok {
			groupIndex = len(groups)
			groupByKey[key] = groupIndex
			groups = append(groups, sourceQueue{
				weight: hardNegativeSourceWeight(weights, trainSet[idx].Source),
			})
		}
		groups[groupIndex].indexes = append(groups[groupIndex].indexes, idx)
	}
	if len(groups) == 0 {
		return order
	}
	for i := range groups {
		if lengthBucket {
			bucketHardNegativeOrderByLength(trainSet, groups[i].indexes, batchSize)
		}
		groups[i].indexes = spreadHardNegativeOrderByQuery(trainSet, groups[i].indexes)
	}
	out := make([]int, 0, len(order))
	remaining := len(order)
	for remaining > 0 {
		batchStart := len(out)
		for len(out)-batchStart < batchSize && remaining > 0 {
			progressed := false
			for i := range groups {
				group := &groups[i]
				for repeat := 0; repeat < group.weight && len(out)-batchStart < batchSize; repeat++ {
					if group.next >= len(group.indexes) {
						break
					}
					out = append(out, group.indexes[group.next])
					group.next++
					remaining--
					progressed = true
				}
			}
			if !progressed {
				break
			}
		}
		if len(out) == batchStart {
			break
		}
	}
	if len(out) != len(order) {
		return order
	}
	return out
}

func hardNegativeSourceGroupKey(weights map[string]int, source string) string {
	exact := normalizedHardNegativeSource(source)
	if _, ok := weights[exact]; ok {
		return exact
	}
	return hardNegativeSourceFamily(source)
}

func hardNegativeSourceWeight(weights map[string]int, source string) int {
	if len(weights) == 0 {
		return 1
	}
	exact := normalizedHardNegativeSource(source)
	if weight := weights[exact]; weight > 0 {
		return weight
	}
	family := hardNegativeSourceFamily(source)
	if weight := weights[family]; weight > 0 {
		return weight
	}
	if weight := weights["*"]; weight > 0 {
		return weight
	}
	return 1
}

func hardNegativeTeacherTemperature(temperatures map[string]float32, source string, fallback float32) float32 {
	if fallback <= 0 {
		fallback = 1
	}
	if len(temperatures) == 0 {
		return fallback
	}
	exact := normalizedHardNegativeSource(source)
	if temp := temperatures[exact]; temp > 0 {
		return temp
	}
	family := hardNegativeSourceFamily(source)
	if temp := temperatures[family]; temp > 0 {
		return temp
	}
	if temp := temperatures["*"]; temp > 0 {
		return temp
	}
	return fallback
}

func hardNegativeTeacherWeight(weights map[string]float32, source string) float32 {
	if len(weights) == 0 {
		return 1
	}
	exact := normalizedHardNegativeSource(source)
	if weight, ok := weights[exact]; ok {
		return weight
	}
	family := hardNegativeSourceFamily(source)
	if weight, ok := weights[family]; ok {
		return weight
	}
	if weight, ok := weights["*"]; ok {
		return weight
	}
	return 1
}

func hardNegativeSourceFamily(source string) string {
	source = normalizedHardNegativeSource(source)
	if idx := strings.IndexByte(source, ':'); idx > 0 {
		return source[:idx]
	}
	return source
}

func normalizedHardNegativeSource(source string) string {
	source = strings.ToLower(strings.TrimSpace(source))
	if source == "" {
		return "unknown"
	}
	return source
}

func normalizeHardNegativeSourceWeights(weights map[string]int) map[string]int {
	if len(weights) == 0 {
		return nil
	}
	out := map[string]int{}
	for source, weight := range weights {
		key := normalizedHardNegativeSource(source)
		if weight <= 0 {
			continue
		}
		out[key] = weight
	}
	if len(out) == 0 {
		return nil
	}
	return out
}

func normalizeHardNegativeTeacherTemperatures(temperatures map[string]float32) map[string]float32 {
	if len(temperatures) == 0 {
		return nil
	}
	out := map[string]float32{}
	for source, temp := range temperatures {
		key := normalizedHardNegativeSource(source)
		if temp <= 0 {
			continue
		}
		out[key] = temp
	}
	if len(out) == 0 {
		return nil
	}
	return out
}

func normalizeHardNegativeTeacherWeights(weights map[string]float32) map[string]float32 {
	if len(weights) == 0 {
		return nil
	}
	out := map[string]float32{}
	for source, weight := range weights {
		key := normalizedHardNegativeSource(source)
		out[key] = weight
	}
	if len(out) == 0 {
		return nil
	}
	return out
}

func normalizeTeacherScoreNormalization(mode string) string {
	mode = strings.ToLower(strings.TrimSpace(mode))
	mode = strings.ReplaceAll(mode, "-", "_")
	if mode == "" {
		return "none"
	}
	switch mode {
	case "off", "disabled":
		return "none"
	case "source", "source_z", "source_zscore":
		return "source_zscore"
	case "family", "family_z", "source_family_zscore":
		return "family_zscore"
	case "example", "example_z", "query_zscore":
		return "example_zscore"
	default:
		return mode
	}
}

func validTeacherScoreNormalization(mode string) bool {
	switch normalizeTeacherScoreNormalization(mode) {
	case "none", "source_zscore", "family_zscore", "example_zscore":
		return true
	default:
		return false
	}
}

type hardNegativeTeacherScoreMoments struct {
	Count int
	Sum   float64
	SumSq float64
}

func (m *hardNegativeTeacherScoreMoments) Add(score float32) {
	if m == nil {
		return
	}
	value := float64(score)
	m.Count++
	m.Sum += value
	m.SumSq += value * value
}

func (m hardNegativeTeacherScoreMoments) MeanStd() (float32, float32) {
	if m.Count <= 0 {
		return 0, 1
	}
	mean := m.Sum / float64(m.Count)
	variance := m.SumSq/float64(m.Count) - mean*mean
	if variance < 1e-12 {
		return float32(mean), 1
	}
	return float32(mean), float32(math.Sqrt(variance))
}

func normalizeHardNegativeTeacherScoresForRun(examples []EmbeddingHardNegativeExample, mode string) ([]EmbeddingHardNegativeExample, error) {
	mode = normalizeTeacherScoreNormalization(mode)
	switch mode {
	case "none":
		return examples, nil
	case "source_zscore", "family_zscore", "example_zscore":
	default:
		return nil, fmt.Errorf("unsupported teacher_score_normalization %q", mode)
	}
	if len(examples) == 0 {
		return examples, nil
	}
	out := append([]EmbeddingHardNegativeExample(nil), examples...)
	if mode == "example_zscore" {
		for i := range out {
			out[i].TeacherScores = zscoreTeacherScoreSlice(out[i].TeacherScores)
		}
		return out, nil
	}
	moments := map[string]*hardNegativeTeacherScoreMoments{}
	for _, example := range out {
		if len(example.TeacherScores) == 0 {
			continue
		}
		key := normalizedHardNegativeSource(example.Source)
		if mode == "family_zscore" {
			key = hardNegativeSourceFamily(example.Source)
		}
		stats := moments[key]
		if stats == nil {
			stats = &hardNegativeTeacherScoreMoments{}
			moments[key] = stats
		}
		for _, score := range example.TeacherScores {
			stats.Add(score)
		}
	}
	sourceScale := make(map[string][2]float32, len(moments))
	for source, stats := range moments {
		mean, std := stats.MeanStd()
		sourceScale[source] = [2]float32{mean, std}
	}
	for i := range out {
		if len(out[i].TeacherScores) == 0 {
			continue
		}
		key := normalizedHardNegativeSource(out[i].Source)
		if mode == "family_zscore" {
			key = hardNegativeSourceFamily(out[i].Source)
		}
		scale := sourceScale[key]
		out[i].TeacherScores = normalizeTeacherScoreSlice(out[i].TeacherScores, scale[0], scale[1])
	}
	return out, nil
}

func zscoreTeacherScoreSlice(scores []float32) []float32 {
	if len(scores) == 0 {
		return nil
	}
	moments := hardNegativeTeacherScoreMoments{}
	for _, score := range scores {
		moments.Add(score)
	}
	mean, std := moments.MeanStd()
	return normalizeTeacherScoreSlice(scores, mean, std)
}

func normalizeTeacherScoreSlice(scores []float32, mean, std float32) []float32 {
	if len(scores) == 0 {
		return nil
	}
	if std <= 1e-6 || math.IsNaN(float64(std)) || math.IsInf(float64(std), 0) {
		std = 1
	}
	out := make([]float32, len(scores))
	for i, score := range scores {
		out[i] = (score - mean) / std
	}
	return out
}

func plannedEvalPassCount(evalExamples, epochs, evalEvery int) int {
	if evalExamples <= 0 || epochs <= 0 {
		return 0
	}
	if evalEvery <= 0 {
		evalEvery = 1
	}
	return epochs/evalEvery + 1
}

func workloadEvalMode(evalExamples int, mode string) string {
	if evalExamples <= 0 {
		return ""
	}
	return mode
}

func (t *EmbeddingTrainer) runEpoch(trainSet []EmbeddingPairExample, order []int, batchSize int, cfg EmbeddingTrainRunConfig, epoch int, runStart time.Time) (EmbeddingTrainMetrics, error) {
	totalLoss := float32(0)
	totalScore := float32(0)
	totalExamples := 0
	batchIndex := 0
	totalBatches := batchCount(len(order), batchSize, 1)
	for start := 0; start < len(order); start += batchSize {
		end := start + batchSize
		if end > len(order) {
			end = len(order)
		}
		batch := make([]EmbeddingPairExample, 0, end-start)
		for _, idx := range order[start:end] {
			batch = append(batch, trainSet[idx])
		}
		metrics, err := t.TrainStep(batch)
		if err != nil {
			return EmbeddingTrainMetrics{}, err
		}
		totalLoss += metrics.Loss * float32(metrics.BatchSize)
		totalScore += metrics.AverageScore * float32(metrics.BatchSize)
		totalExamples += metrics.BatchSize
		batchIndex++
		maybeReportTrainProgress(cfg, EmbeddingTrainProgress{
			Phase:              "train",
			Epoch:              epoch,
			Batch:              batchIndex,
			Batches:            totalBatches,
			Step:               t.step,
			BatchExamples:      metrics.BatchSize,
			BatchPairs:         int64(metrics.BatchSize),
			EpochTrainExamples: int64(totalExamples),
			EpochTrainPairs:    int64(totalExamples),
			PlannedEpochPairs:  int64(len(order)),
			Loss:               metrics.Loss,
			AverageScore:       metrics.AverageScore,
			Elapsed:            time.Since(runStart),
		})
	}
	if totalExamples == 0 {
		return EmbeddingTrainMetrics{}, fmt.Errorf("training epoch has no examples")
	}
	inv := float32(1) / float32(totalExamples)
	return EmbeddingTrainMetrics{
		Loss:         totalLoss * inv,
		AverageScore: totalScore * inv,
		BatchSize:    totalExamples,
	}, nil
}

type contrastiveEpochBatchHook func(EmbeddingTrainProgress) error

func (t *EmbeddingTrainer) runContrastiveEpoch(trainSet []EmbeddingContrastiveExample, order []int, batchSize int, cfg EmbeddingTrainRunConfig, epoch int, runStart time.Time, afterBatch contrastiveEpochBatchHook) (EmbeddingTrainMetrics, error) {
	totalLoss := float32(0)
	totalScore := float32(0)
	totalExamples := 0
	totalTrainExamples := 0
	var totalPairs int64
	batchIndex := 0
	totalBatches, plannedEpochPairs := contrastiveBatchWork(len(order), batchSize)
	plannedEpochPairs *= int64(compactPrefixObjectiveMultiplier(cfg))
	for start := 0; start < len(order); start += batchSize {
		end := start + batchSize
		if end > len(order) {
			end = len(order)
		}
		if end-start < 2 {
			break
		}
		batch := make([]EmbeddingContrastiveExample, 0, end-start)
		for _, idx := range order[start:end] {
			batch = append(batch, trainSet[idx])
		}
		metrics, err := t.TrainContrastiveStep(batch)
		if err != nil {
			return EmbeddingTrainMetrics{}, err
		}
		totalLoss += metrics.Loss * float32(metrics.BatchSize)
		totalScore += metrics.AverageScore * float32(metrics.BatchSize)
		totalExamples += metrics.BatchSize
		totalTrainExamples += end - start
		totalPairs += int64(metrics.BatchSize)
		batchIndex++
		// Bound RSS: per-step activations are unreferenced after the step, but
		// Go's scavenger returns pages to the OS lazily, so RSS can climb past a
		// memory cap (e.g. WSL2's) and trigger an OOM/SIGTERM before reclaim —
		// especially for wider models. Force the scavenger periodically.
		if n := trainMemReclaimEvery(); n > 0 && batchIndex%n == 0 {
			debug.FreeOSMemory()
		}
		progress := EmbeddingTrainProgress{
			Phase:              "train",
			Epoch:              epoch,
			Batch:              batchIndex,
			Batches:            totalBatches,
			Step:               t.step,
			BatchExamples:      end - start,
			BatchPairs:         int64(metrics.BatchSize),
			EpochTrainExamples: int64(totalTrainExamples),
			EpochTrainPairs:    totalPairs,
			PlannedEpochPairs:  plannedEpochPairs,
			Loss:               metrics.Loss,
			AverageScore:       metrics.AverageScore,
			Elapsed:            time.Since(runStart),
		}
		maybeReportTrainProgress(cfg, progress)
		if afterBatch != nil {
			if err := afterBatch(progress); err != nil {
				return EmbeddingTrainMetrics{}, err
			}
		}
	}
	if totalExamples == 0 {
		return EmbeddingTrainMetrics{}, fmt.Errorf("training epoch has no usable contrastive batches")
	}
	inv := float32(1) / float32(totalExamples)
	return EmbeddingTrainMetrics{
		Loss:         totalLoss * inv,
		AverageScore: totalScore * inv,
		BatchSize:    totalExamples,
	}, nil
}

// trainMemReclaimEvery returns how often (in batches) to force the Go scavenger
// to return freed activation memory to the OS during training, bounding RSS.
// Per-step activations are unreferenced after each step, but the scavenger
// returns pages lazily, so for wider models RSS can climb past a memory cap
// (e.g. WSL2's ~27GB) and get SIGTERM'd before reclaim. Default every 8 batches;
// 0 disables. Override with EOS_TRAIN_RECLAIM_EVERY.
func trainMemReclaimEvery() int {
	if v := strings.TrimSpace(os.Getenv("EOS_TRAIN_RECLAIM_EVERY")); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n >= 0 {
			return n
		}
	}
	return 8
}

func (t *EmbeddingTrainer) runHardNegativeEpoch(trainSet []EmbeddingHardNegativeExample, order []int, batchSize int, cfg EmbeddingTrainRunConfig, epoch int, runStart time.Time, afterBatch contrastiveEpochBatchHook) (EmbeddingTrainMetrics, error) {
	totalLoss := float32(0)
	totalScore := float32(0)
	totalExamples := 0
	totalTrainExamples := 0
	var totalPairs int64
	batchIndex := 0
	if len(cfg.HardNegativeSourceWeights) == 0 {
		order = spreadHardNegativeOrderByQuery(trainSet, order)
	}
	totalBatches, plannedEpochPairs := hardNegativeBatchWork(len(order), batchSize, cfg.HardNegativesPerQuery, cfg.ContrastiveLoss, cfg.TeacherLossWeight, compactPrefixObjectiveMultiplier(cfg), turboQuantRankMarginObjectiveCount(cfg.TurboQuantRankMarginObjectives), turboQuantRankMarginObjectiveCount(cfg.TurboQuantCompactObjectives))
	for start := 0; start < len(order); start += batchSize {
		end := start + batchSize
		if end > len(order) {
			end = len(order)
		}
		batch := make([]EmbeddingHardNegativeExample, 0, end-start)
		for _, idx := range order[start:end] {
			batch = append(batch, trainSet[idx])
		}
		if hardNegativeBatchPairCount(batch) < 2 {
			break
		}
		metrics, err := t.TrainHardNegativeContrastiveStep(batch)
		if err != nil {
			return EmbeddingTrainMetrics{}, err
		}
		totalLoss += metrics.Loss * float32(metrics.BatchSize)
		totalScore += metrics.AverageScore * float32(metrics.BatchSize)
		totalExamples += metrics.BatchSize
		totalTrainExamples += end - start
		totalPairs += int64(metrics.BatchSize)
		batchIndex++
		// Bound RSS: per-step activations are unreferenced after the step, but
		// Go's scavenger returns pages to the OS lazily, so RSS can climb past a
		// memory cap (e.g. WSL2's) and trigger an OOM/SIGTERM before reclaim —
		// especially for wider models. Force the scavenger periodically.
		if n := trainMemReclaimEvery(); n > 0 && batchIndex%n == 0 {
			debug.FreeOSMemory()
		}
		progress := EmbeddingTrainProgress{
			Phase:              "train",
			Epoch:              epoch,
			Batch:              batchIndex,
			Batches:            totalBatches,
			Step:               t.step,
			BatchExamples:      end - start,
			BatchPairs:         int64(metrics.BatchSize),
			EpochTrainExamples: int64(totalTrainExamples),
			EpochTrainPairs:    totalPairs,
			PlannedEpochPairs:  plannedEpochPairs,
			Loss:               metrics.Loss,
			AverageScore:       metrics.AverageScore,
			Elapsed:            time.Since(runStart),
		}
		maybeReportTrainProgress(cfg, progress)
		if afterBatch != nil {
			if err := afterBatch(progress); err != nil {
				return EmbeddingTrainMetrics{}, err
			}
		}
	}
	if totalExamples == 0 {
		return EmbeddingTrainMetrics{}, fmt.Errorf("training epoch has no usable hard-negative batches")
	}
	inv := float32(1) / float32(totalExamples)
	return EmbeddingTrainMetrics{
		Loss:         totalLoss * inv,
		AverageScore: totalScore * inv,
		BatchSize:    totalExamples,
	}, nil
}

func (t *EmbeddingTrainer) runScoreSpectrumEpoch(trainSet []EmbeddingScoreSpectrumExample, order []int, batchSize int, cfg EmbeddingTrainRunConfig, epoch int, runStart time.Time, afterBatch contrastiveEpochBatchHook) (EmbeddingTrainMetrics, error) {
	totalLoss := float32(0)
	totalScore := float32(0)
	totalTrainExamples := 0
	var totalPairs int64
	batchIndex := 0
	ordered := make([]EmbeddingScoreSpectrumExample, 0, len(order))
	for _, idx := range order {
		ordered = append(ordered, trainSet[idx])
	}
	totalBatches, plannedEpochPairs := scoreSpectrumBatchWork(ordered, batchSize)
	for start := 0; start < len(order); start += batchSize {
		end := start + batchSize
		if end > len(order) {
			end = len(order)
		}
		batch := make([]EmbeddingScoreSpectrumExample, 0, end-start)
		for _, idx := range order[start:end] {
			batch = append(batch, trainSet[idx])
		}
		if scoreSpectrumBatchPairCount(batch) <= 0 {
			break
		}
		metrics, err := t.TrainScoreSpectrumStep(batch)
		if err != nil {
			return EmbeddingTrainMetrics{}, err
		}
		// TrainScoreSpectrumStep reports query-averaged loss and pair-averaged
		// score. Keep epoch loss query-averaged; use pair counts only for score
		// aggregation, workload accounting, and progress.
		totalLoss += metrics.Loss * float32(len(batch))
		totalScore += metrics.AverageScore * float32(metrics.BatchSize)
		totalTrainExamples += end - start
		totalPairs += int64(metrics.BatchSize)
		batchIndex++
		if n := trainMemReclaimEvery(); n > 0 && batchIndex%n == 0 {
			debug.FreeOSMemory()
		}
		progress := EmbeddingTrainProgress{
			Phase:              "train",
			Epoch:              epoch,
			Batch:              batchIndex,
			Batches:            totalBatches,
			Step:               t.step,
			BatchExamples:      end - start,
			BatchPairs:         int64(metrics.BatchSize),
			EpochTrainExamples: int64(totalTrainExamples),
			EpochTrainPairs:    totalPairs,
			PlannedEpochPairs:  plannedEpochPairs,
			Loss:               metrics.Loss,
			AverageScore:       metrics.AverageScore,
			Elapsed:            time.Since(runStart),
		}
		maybeReportTrainProgress(cfg, progress)
		if afterBatch != nil {
			if err := afterBatch(progress); err != nil {
				return EmbeddingTrainMetrics{}, err
			}
		}
	}
	if totalTrainExamples == 0 {
		return EmbeddingTrainMetrics{}, fmt.Errorf("training epoch has no usable score-spectrum batches")
	}
	lossInv := float32(1) / float32(totalTrainExamples)
	scoreAvg := float32(0)
	if totalPairs > 0 {
		scoreAvg = totalScore / float32(totalPairs)
	}
	return EmbeddingTrainMetrics{
		Loss:         totalLoss * lossInv,
		AverageScore: scoreAvg,
		BatchSize:    totalTrainExamples,
	}, nil
}

func (t *EmbeddingTrainer) runListwiseGeometryEpoch(trainSet []EmbeddingTokenizedListwiseGeometryBatch, order []int, batchSize int, cfg EmbeddingTrainRunConfig, epoch int, runStart time.Time, afterBatch contrastiveEpochBatchHook) (EmbeddingTrainMetrics, error) {
	totalLoss := float32(0)
	totalScore := float32(0)
	totalTrainExamples := 0
	totalTrainQueries := 0
	var totalPairs int64
	var movement EmbeddingTrainMovementMetrics
	haveMovement := false
	batchIndex := 0
	ordered := make([]EmbeddingTokenizedListwiseGeometryBatch, 0, len(order))
	for _, idx := range order {
		ordered = append(ordered, trainSet[idx])
	}
	totalBatches, plannedEpochPairs := listwiseGeometryBatchWork(ordered, batchSize)
	for start := 0; start < len(order); start += batchSize {
		end := start + batchSize
		if end > len(order) {
			end = len(order)
		}
		batch := make([]EmbeddingTokenizedListwiseGeometryBatch, 0, end-start)
		for _, idx := range order[start:end] {
			batch = append(batch, trainSet[idx])
		}
		batchPairs := listwiseGeometryBatchPairCount(batch)
		if batchPairs <= 0 {
			break
		}
		batchQueries := listwiseGeometryBatchQueryCount(batch)
		nextBatchIndex := batchIndex + 1
		maybeReportTrainProgress(cfg, EmbeddingTrainProgress{
			Phase:              "train_start",
			Epoch:              epoch,
			Batch:              nextBatchIndex,
			Batches:            totalBatches,
			Step:               t.step,
			BatchExamples:      end - start,
			BatchPairs:         batchPairs,
			EpochTrainExamples: int64(totalTrainExamples),
			EpochTrainPairs:    totalPairs,
			PlannedEpochPairs:  plannedEpochPairs,
			Elapsed:            time.Since(runStart),
		})
		metrics, err := t.TrainListwiseGeometryStepWithDiagnostics(batch, cfg.MovementDiagnostics)
		if err != nil {
			return EmbeddingTrainMetrics{}, err
		}
		totalLoss += metrics.Loss * float32(batchQueries)
		totalScore += metrics.AverageScore * float32(metrics.BatchSize)
		totalTrainExamples += end - start
		totalTrainQueries += batchQueries
		totalPairs += int64(metrics.BatchSize)
		if metrics.Movement != nil {
			mergeEmbeddingTrainMovementMetrics(&movement, metrics.Movement)
			haveMovement = true
		}
		batchIndex++
		if n := trainMemReclaimEvery(); n > 0 && batchIndex%n == 0 {
			debug.FreeOSMemory()
		}
		progress := EmbeddingTrainProgress{
			Phase:              "train",
			Epoch:              epoch,
			Batch:              batchIndex,
			Batches:            totalBatches,
			Step:               t.step,
			BatchExamples:      end - start,
			BatchPairs:         int64(metrics.BatchSize),
			EpochTrainExamples: int64(totalTrainExamples),
			EpochTrainPairs:    totalPairs,
			PlannedEpochPairs:  plannedEpochPairs,
			Loss:               metrics.Loss,
			AverageScore:       metrics.AverageScore,
			Elapsed:            time.Since(runStart),
		}
		maybeReportTrainProgress(cfg, progress)
		if afterBatch != nil {
			if err := afterBatch(progress); err != nil {
				return EmbeddingTrainMetrics{}, err
			}
		}
	}
	if totalTrainExamples == 0 {
		return EmbeddingTrainMetrics{}, fmt.Errorf("training epoch has no usable listwise geometry batches")
	}
	if totalTrainQueries == 0 {
		return EmbeddingTrainMetrics{}, fmt.Errorf("training epoch has no usable listwise geometry queries")
	}
	scoreAvg := float32(0)
	if totalPairs > 0 {
		scoreAvg = totalScore / float32(totalPairs)
	}
	out := EmbeddingTrainMetrics{
		Loss:         totalLoss / float32(totalTrainQueries),
		AverageScore: scoreAvg,
		BatchSize:    totalTrainExamples,
	}
	if haveMovement {
		out.Movement = &movement
	}
	return out, nil
}

func (t *EmbeddingTrainer) restoreCheckpoint(checkpoint EmbeddingTrainCheckpoint) error {
	restored, err := NewEmbeddingTrainerFromCheckpoint(t.module, checkpoint)
	if err != nil {
		return err
	}
	// The rebuilt trainer never saw configureRetrievalEval; preserve the
	// retrieval gate across restore or the post-restore final eval silently
	// reports zero retrieval metrics.
	retrievalRuntime := t.retrievalEvalRuntime
	retrievalConfig := t.retrievalEvalConfig
	retrievalTokenizer := t.retrievalEvalTokenizer
	*t = *restored
	t.configureRetrievalEval(retrievalRuntime, retrievalConfig, retrievalTokenizer)
	return nil
}

func maybeReportTrainProgress(cfg EmbeddingTrainRunConfig, progress EmbeddingTrainProgress) {
	if cfg.Progress == nil || cfg.ProgressEverySteps <= 0 || progress.Batch <= 0 {
		return
	}
	if progress.Batch%cfg.ProgressEverySteps != 0 && progress.Batch != progress.Batches {
		return
	}
	if progress.Phase == "" {
		progress.Phase = "train"
	}
	cfg.Progress(progress)
}

func maybeReportEvalProgress(cfg EmbeddingTrainRunConfig, phase string, epoch, step, evalPass int, evalExamples, evalPairs int64, runStart time.Time) {
	if cfg.Progress == nil || cfg.ProgressEverySteps <= 0 {
		return
	}
	cfg.Progress(EmbeddingTrainProgress{
		Phase:        phase,
		Epoch:        epoch,
		Step:         step,
		EvalPass:     evalPass,
		EvalExamples: evalExamples,
		EvalPairs:    evalPairs,
		Elapsed:      time.Since(runStart),
	})
}

func bucketContrastiveOrderByLength(trainSet []EmbeddingContrastiveExample, order []int, batchSize int) {
	if len(trainSet) == 0 || len(order) < 2 || batchSize <= 1 {
		return
	}
	windowSize := contrastiveLengthBucketWindow(batchSize, len(order))
	for start := 0; start < len(order); start += windowSize {
		end := start + windowSize
		if end > len(order) {
			end = len(order)
		}
		window := order[start:end]
		sort.SliceStable(window, func(i, j int) bool {
			left := contrastiveExampleSortLength(trainSet[window[i]])
			right := contrastiveExampleSortLength(trainSet[window[j]])
			return left < right
		})
	}
}

func bucketHardNegativeOrderByLength(trainSet []EmbeddingHardNegativeExample, order []int, batchSize int) {
	if len(trainSet) == 0 || len(order) == 0 || batchSize <= 0 {
		return
	}
	windowSize := contrastiveLengthBucketWindow(batchSize, len(order))
	for start := 0; start < len(order); start += windowSize {
		end := start + windowSize
		if end > len(order) {
			end = len(order)
		}
		window := order[start:end]
		sort.SliceStable(window, func(i, j int) bool {
			left := hardNegativeExampleSortLength(trainSet[window[i]])
			right := hardNegativeExampleSortLength(trainSet[window[j]])
			return left < right
		})
	}
}

func contrastiveLengthBucketWindow(batchSize, total int) int {
	if total <= 0 {
		return 0
	}
	if batchSize <= 1 {
		return total
	}
	windowSize := batchSize * 4
	if raw := trainEnv("EOS_TRAIN_LENGTH_BUCKET_WINDOW"); raw != "" {
		if parsed, err := strconv.Atoi(raw); err == nil && parsed > 0 {
			windowSize = parsed
		}
	}
	if windowSize < batchSize {
		windowSize = batchSize
	}
	if windowSize > total {
		windowSize = total
	}
	return windowSize
}

func contrastiveExampleSortLength(example EmbeddingContrastiveExample) int {
	length := len(example.QueryTokens)
	if len(example.PositiveTokens) > length {
		length = len(example.PositiveTokens)
	}
	return length
}

func hardNegativeExampleSortLength(example EmbeddingHardNegativeExample) int {
	length := len(example.QueryTokens)
	if len(example.PositiveTokens) > length {
		length = len(example.PositiveTokens)
	}
	for _, tokens := range example.NegativeTokens {
		if len(tokens) > length {
			length = len(tokens)
		}
	}
	return length
}

func expandContrastiveExamples(examples []EmbeddingContrastiveExample) []EmbeddingPairExample {
	if len(examples) == 0 {
		return nil
	}
	out := make([]EmbeddingPairExample, 0, len(examples)*len(examples))
	for i, example := range examples {
		out = append(out, EmbeddingPairExample{
			Source:      example.Source,
			LeftTokens:  append([]int32(nil), example.QueryTokens...),
			RightTokens: append([]int32(nil), example.PositiveTokens...),
			LeftMask:    append([]int32(nil), example.QueryMask...),
			RightMask:   append([]int32(nil), example.PositiveMask...),
			Target:      1,
		})
		for j, negative := range examples {
			if i == j {
				continue
			}
			out = append(out, EmbeddingPairExample{
				Source:      example.Source,
				LeftTokens:  append([]int32(nil), example.QueryTokens...),
				RightTokens: append([]int32(nil), negative.PositiveTokens...),
				LeftMask:    append([]int32(nil), example.QueryMask...),
				RightMask:   append([]int32(nil), negative.PositiveMask...),
				Target:      -1,
			})
		}
	}
	return out
}

// retrievalEvalGateConfigured reports whether cfg carries a complete
// BEIR-style retrieval eval gate (runtime plus corpus/queries/qrels paths),
// mirroring EmbeddingTrainer.configureRetrievalEval's own enablement check.
// Callers use this to decide whether an unset SelectMetric should default to
// the retrieval-gated metric instead of the legacy pairwise default.
func retrievalEvalGateConfigured(cfg EmbeddingTrainRunConfig) bool {
	return cfg.RetrievalEvalRuntime != nil &&
		strings.TrimSpace(cfg.RetrievalEval.CorpusPath) != "" &&
		strings.TrimSpace(cfg.RetrievalEval.QueriesPath) != "" &&
		strings.TrimSpace(cfg.RetrievalEval.QrelsPath) != ""
}

func normalizedTrainRunConfig(cfg EmbeddingTrainRunConfig) EmbeddingTrainRunConfig {
	if cfg.EvalOnly {
		cfg.Epochs = 0
	} else if cfg.Epochs == 0 {
		cfg.Epochs = 1
	}
	if cfg.BatchSize == 0 {
		cfg.BatchSize = 8
	}
	if cfg.EvalEveryEpoch == 0 {
		cfg.EvalEveryEpoch = 1
	}
	if cfg.SelectMetric == "" {
		// Direct API callers who configure a complete retrieval eval gate but
		// leave SelectMetric unset get the retrieval-gated default instead of
		// the legacy pairwise metric, matching the CLI's auto-upgrade in
		// cmd/eos's runTrainEmbed. Callers who explicitly set SelectMetric
		// (including explicitly to "score_margin" or any pairwise metric)
		// are never touched here.
		if retrievalEvalGateConfigured(cfg) {
			cfg.SelectMetric = "retrieval_ndcg"
		} else {
			cfg.SelectMetric = "score_margin"
		}
	}
	if cfg.Seed == 0 {
		cfg.Seed = 1
	}
	if cfg.HardNegativesPerQuery == 0 {
		cfg.HardNegativesPerQuery = 1
	}
	if cfg.VectorDistillDefaultRole == "" {
		cfg.VectorDistillDefaultRole = EmbeddingRoleQuery
	}
	if mode, err := NormalizeVectorDistillOptimizerSyncMode(cfg.VectorDistillOptimizerSync); err == nil {
		cfg.VectorDistillOptimizerSync = mode
	}
	cfg.GroupedLossWeight = effectiveGroupedLossWeight(cfg.ContrastiveLoss, cfg.GroupedLossWeight)
	cfg.HardNegativeSourceWeights = normalizeHardNegativeSourceWeights(cfg.HardNegativeSourceWeights)
	cfg.TeacherSourceTemperatures = normalizeHardNegativeTeacherTemperatures(cfg.TeacherSourceTemperatures)
	cfg.TeacherSourceWeights = normalizeHardNegativeTeacherWeights(cfg.TeacherSourceWeights)
	if dims, weights, err := normalizeMatryoshkaDimsAndWeights(cfg.MatryoshkaDims, cfg.MatryoshkaWeights, 0); err == nil {
		cfg.MatryoshkaDims = dims
		cfg.MatryoshkaWeights = weights
	}
	if bits, err := normalizeTurboQuantPrefixBits(cfg.TurboQuantPrefixBits); err == nil {
		cfg.TurboQuantPrefixBits = bits
	}
	if objectives, err := normalizeTurboQuantPrefixObjectives(cfg.TurboQuantPrefixObjectives, cfg.MatryoshkaDims, 0); err == nil {
		cfg.TurboQuantPrefixObjectives = objectives
	}
	if objectives, err := normalizeTurboQuantPrefixObjectives(cfg.TurboQuantRankMarginObjectives, cfg.MatryoshkaDims, 0); err == nil {
		cfg.TurboQuantRankMarginObjectives = objectives
	}
	if objectives, err := normalizeTurboQuantPrefixObjectives(cfg.TurboQuantCompactObjectives, cfg.MatryoshkaDims, 0); err == nil {
		cfg.TurboQuantCompactObjectives = objectives
	}
	if len(cfg.TurboQuantRankMarginObjectives) > 0 && cfg.TurboQuantRankMargin == 0 {
		cfg.TurboQuantRankMargin = effectiveTurboQuantRankMargin(cfg.TurboQuantRankMargin)
	}
	if len(cfg.TurboQuantPrefixBits) > 0 || len(cfg.TurboQuantPrefixObjectives) > 0 || len(cfg.TurboQuantCompactObjectives) > 0 || len(cfg.TurboQuantRankMarginObjectives) > 0 {
		if cfg.TurboQuantPrefixWeight == 0 {
			if len(cfg.TurboQuantPrefixBits) > 0 {
				cfg.TurboQuantPrefixWeight = 1
			}
		}
		cfg.TurboQuantPrefixSeed = effectiveTurboQuantPrefixSeed(cfg.TurboQuantPrefixSeed)
	}
	if len(cfg.TurboQuantPrefixBits) > 0 || len(cfg.TurboQuantPrefixObjectives) > 0 || len(cfg.TurboQuantRankMarginObjectives) > 0 {
		if mode, err := normalizeTurboQuantPrefixScoreMode(cfg.TurboQuantPrefixScoreMode); err == nil {
			cfg.TurboQuantPrefixScoreMode = mode
		}
	}
	cfg = normalizedScoreSpectrumRunConfig(cfg)
	cfg.TeacherScoreNormalization = normalizeTeacherScoreNormalization(cfg.TeacherScoreNormalization)
	return cfg
}

func NormalizeVectorDistillOptimizerSyncMode(mode string) (string, error) {
	switch strings.ToLower(strings.TrimSpace(strings.ReplaceAll(mode, "-", "_"))) {
	case "", VectorDistillOptimizerSyncDeferred:
		return VectorDistillOptimizerSyncDeferred, nil
	case VectorDistillOptimizerSyncImmediate:
		return VectorDistillOptimizerSyncImmediate, nil
	default:
		return "", fmt.Errorf("unsupported vector_distill_optimizer_sync %q (supported: %s, %s)", mode, VectorDistillOptimizerSyncDeferred, VectorDistillOptimizerSyncImmediate)
	}
}

func normalizedScoreSpectrumRunConfig(cfg EmbeddingTrainRunConfig) EmbeddingTrainRunConfig {
	if strings.TrimSpace(cfg.ScoreSpectrumLossMode) != "" {
		mode, err := normalizeScoreSpectrumLossMode(cfg.ScoreSpectrumLossMode)
		if err == nil {
			cfg.ScoreSpectrumLossMode = mode
		}
	}
	if cfg.ScoreSpectrumLossMode != "" && cfg.ScoreSpectrumRecoveryWeight == 0 && scoreSpectrumLossModeIncludesRecovery(cfg.ScoreSpectrumLossMode) {
		cfg.ScoreSpectrumRecoveryWeight = 1
	}
	if cfg.ScoreSpectrumRecoveryTopK == 0 {
		cfg.ScoreSpectrumRecoveryTopK = 4
	}
	if cfg.ScoreSpectrumRecoveryTau == 0 {
		if cfg.Temperature > 0 {
			cfg.ScoreSpectrumRecoveryTau = cfg.Temperature
		} else {
			cfg.ScoreSpectrumRecoveryTau = 0.05
		}
	}
	return cfg
}

func finalizedScoreSpectrumRunConfig(cfg EmbeddingTrainRunConfig) EmbeddingTrainRunConfig {
	mode, err := normalizeScoreSpectrumLossMode(cfg.ScoreSpectrumLossMode)
	if err == nil {
		cfg.ScoreSpectrumLossMode = mode
	}
	if cfg.ScoreSpectrumRecoveryWeight == 0 && scoreSpectrumLossModeIncludesRecovery(cfg.ScoreSpectrumLossMode) {
		cfg.ScoreSpectrumRecoveryWeight = 1
	}
	if cfg.ScoreSpectrumRecoveryTopK == 0 {
		cfg.ScoreSpectrumRecoveryTopK = 4
	}
	if cfg.ScoreSpectrumRecoveryTau == 0 {
		if cfg.Temperature > 0 {
			cfg.ScoreSpectrumRecoveryTau = cfg.Temperature
		} else {
			cfg.ScoreSpectrumRecoveryTau = 0.05
		}
	}
	return cfg
}

func validateClearTurboQuantPrefixRunConfig(cfg EmbeddingTrainRunConfig) error {
	if !cfg.ClearTurboQuantPrefix {
		return nil
	}
	if len(cfg.TurboQuantPrefixBits) > 0 {
		return fmt.Errorf("clear_turboquant_prefix is mutually exclusive with turboquant_prefix_bits")
	}
	if len(cfg.TurboQuantPrefixObjectives) > 0 {
		return fmt.Errorf("clear_turboquant_prefix is mutually exclusive with turboquant_prefix_objectives")
	}
	if cfg.TurboQuantPrefixWeight != 0 {
		return fmt.Errorf("clear_turboquant_prefix is mutually exclusive with turboquant_prefix_weight")
	}
	seedOrModeOnlyForRankMargin := len(cfg.TurboQuantRankMarginObjectives) > 0 || len(cfg.TurboQuantCompactObjectives) > 0
	if cfg.TurboQuantPrefixSeed != 0 {
		if !seedOrModeOnlyForRankMargin {
			return fmt.Errorf("clear_turboquant_prefix is mutually exclusive with turboquant_prefix_seed")
		}
	}
	if strings.TrimSpace(cfg.TurboQuantPrefixScoreMode) != "" {
		if !seedOrModeOnlyForRankMargin {
			return fmt.Errorf("clear_turboquant_prefix is mutually exclusive with turboquant_prefix_score_mode")
		}
	}
	return nil
}

func validateClearTurboQuantRankMarginRunConfig(cfg EmbeddingTrainRunConfig) error {
	if !cfg.ClearTurboQuantRankMargin {
		return nil
	}
	if len(cfg.TurboQuantRankMarginObjectives) > 0 {
		return fmt.Errorf("clear_turboquant_rank_margin is mutually exclusive with turboquant_rank_margin_objectives")
	}
	if cfg.TurboQuantRankMargin != 0 {
		return fmt.Errorf("clear_turboquant_rank_margin is mutually exclusive with turboquant_rank_margin")
	}
	return nil
}

func validateScoreSpectrumRunConfig(cfg EmbeddingTrainRunConfig) error {
	if len(cfg.MatryoshkaDims) > 0 {
		return fmt.Errorf("score-spectrum training does not support matryoshka objectives in v1")
	}
	if len(cfg.TurboQuantPrefixBits) > 0 || len(cfg.TurboQuantPrefixObjectives) > 0 {
		return fmt.Errorf("score-spectrum training does not support turboquant prefix objectives in v1")
	}
	if len(cfg.TurboQuantCompactObjectives) > 0 {
		return fmt.Errorf("score-spectrum training does not support turboquant compact objectives in v1")
	}
	if len(cfg.TurboQuantRankMarginObjectives) > 0 {
		return fmt.Errorf("score-spectrum training does not support turboquant rank-margin objectives in v1")
	}
	if err := validateScoreSpectrumRecoveryConfig(cfg.ScoreSpectrumLossMode, cfg.ScoreSpectrumRecoveryWeight, cfg.ScoreSpectrumRecoveryMargin, cfg.ScoreSpectrumRecoveryTopK, cfg.ScoreSpectrumRecoveryTau); err != nil {
		return err
	}
	return nil
}

func validateListwiseGeometryRunConfig(cfg EmbeddingTrainRunConfig) error {
	if cfg.MaxListwiseGeometryTrainPairs < 0 {
		return fmt.Errorf("max_listwise_geometry_train_pairs must be non-negative")
	}
	if cfg.MaxListwiseGeometryEvalPairs < 0 {
		return fmt.Errorf("max_listwise_geometry_eval_pairs must be non-negative")
	}
	if len(cfg.MatryoshkaDims) > 0 {
		return fmt.Errorf("listwise geometry training does not support matryoshka objectives in v1")
	}
	if len(cfg.TurboQuantPrefixBits) > 0 || len(cfg.TurboQuantPrefixObjectives) > 0 {
		return fmt.Errorf("listwise geometry training does not support turboquant prefix objectives in v1")
	}
	if len(cfg.TurboQuantCompactObjectives) > 0 {
		return fmt.Errorf("listwise geometry training does not support turboquant compact objectives in v1")
	}
	if len(cfg.TurboQuantRankMarginObjectives) > 0 {
		return fmt.Errorf("listwise geometry training does not support turboquant rank-margin objectives in v1")
	}
	return nil
}

func validateListwiseGeometryWorkloadLimits(workload EmbeddingTrainWorkload, cfg EmbeddingTrainRunConfig) error {
	if cfg.MaxListwiseGeometryTrainPairs > 0 && workload.TrainPairsPerEpoch > cfg.MaxListwiseGeometryTrainPairs {
		return fmt.Errorf("listwise geometry train_pairs/epoch %d exceeds max_listwise_geometry_train_pairs %d", workload.TrainPairsPerEpoch, cfg.MaxListwiseGeometryTrainPairs)
	}
	if cfg.MaxListwiseGeometryEvalPairs > 0 && workload.EvalPairsPerPass > cfg.MaxListwiseGeometryEvalPairs {
		return fmt.Errorf("listwise geometry eval_pairs/pass %d exceeds max_listwise_geometry_eval_pairs %d", workload.EvalPairsPerPass, cfg.MaxListwiseGeometryEvalPairs)
	}
	return nil
}

func (t *EmbeddingTrainer) isolateScoreSpectrumObjectiveConfig(cfg EmbeddingTrainRunConfig) EmbeddingTrainRunConfig {
	if t == nil {
		return cfg
	}
	cleared := scoreSpectrumIncompatibleObjectiveNames(t.config)
	if len(cleared) == 0 {
		return cfg
	}
	t.config = clearScoreSpectrumIncompatibleTrainConfig(t.config)
	cfg = clearScoreSpectrumIncompatibleRunConfig(cfg)
	t.scoreSpectrumLineage.AutoClearedObjectives = mergeScoreSpectrumObjectiveNames(t.scoreSpectrumLineage.AutoClearedObjectives, cleared)
	t.scoreSpectrumLineage.IsolatedInheritedObjectives = mergeScoreSpectrumObjectiveNames(t.scoreSpectrumLineage.IsolatedInheritedObjectives, cleared)
	t.scoreSpectrumLineage.ScoreSpectrumTrain = true
	return cfg
}

func (t *EmbeddingTrainer) isolateListwiseGeometryObjectiveConfig(cfg EmbeddingTrainRunConfig) EmbeddingTrainRunConfig {
	if t == nil {
		return cfg
	}
	cleared := scoreSpectrumIncompatibleObjectiveNames(t.config)
	if len(cleared) == 0 {
		return cfg
	}
	t.config = clearScoreSpectrumIncompatibleTrainConfig(t.config)
	cfg = clearScoreSpectrumIncompatibleRunConfig(cfg)
	t.listwiseGeometryLineage.AutoClearedObjectives = mergeScoreSpectrumObjectiveNames(t.listwiseGeometryLineage.AutoClearedObjectives, cleared)
	t.listwiseGeometryLineage.IsolatedInheritedObjectives = mergeScoreSpectrumObjectiveNames(t.listwiseGeometryLineage.IsolatedInheritedObjectives, cleared)
	t.listwiseGeometryLineage.ListwiseGeometryTrain = true
	return cfg
}

func scoreSpectrumIncompatibleObjectiveNames(cfg EmbeddingTrainConfig) []string {
	names := []string{}
	if len(cfg.MatryoshkaDims) > 0 || len(cfg.MatryoshkaWeights) > 0 {
		names = append(names, "matryoshka")
	}
	if len(cfg.TurboQuantPrefixBits) > 0 {
		names = append(names, "turboquant_prefix_bits")
	}
	if len(cfg.TurboQuantPrefixObjectives) > 0 {
		names = append(names, "turboquant_prefix_objectives")
	}
	if cfg.TurboQuantPrefixWeight != 0 && (len(cfg.TurboQuantPrefixBits) > 0 || len(cfg.TurboQuantPrefixObjectives) > 0) {
		names = append(names, "turboquant_prefix_weight")
	}
	if len(cfg.TurboQuantCompactObjectives) > 0 {
		names = append(names, "turboquant_compact_objectives")
	}
	if len(cfg.TurboQuantRankMarginObjectives) > 0 {
		names = append(names, "turboquant_rank_margin_objectives")
	}
	if cfg.TurboQuantRankMargin != 0 {
		names = append(names, "turboquant_rank_margin")
	}
	if cfg.TurboQuantPrefixSeed != 0 {
		names = append(names, "turboquant_prefix_seed")
	}
	if strings.TrimSpace(cfg.TurboQuantPrefixScoreMode) != "" {
		names = append(names, "turboquant_prefix_score_mode")
	}
	return normalizeScoreSpectrumObjectiveNames(names)
}

func clearScoreSpectrumIncompatibleTrainConfig(cfg EmbeddingTrainConfig) EmbeddingTrainConfig {
	cfg.MatryoshkaDims = nil
	cfg.MatryoshkaWeights = nil
	cfg.TurboQuantPrefixBits = nil
	cfg.TurboQuantPrefixObjectives = nil
	cfg.TurboQuantPrefixWeight = 0
	cfg.TurboQuantPrefixSeed = 0
	cfg.TurboQuantPrefixScoreMode = ""
	cfg.TurboQuantCompactObjectives = nil
	cfg.TurboQuantRankMarginObjectives = nil
	cfg.TurboQuantRankMargin = 0
	return cfg
}

func clearScoreSpectrumIncompatibleRunConfig(cfg EmbeddingTrainRunConfig) EmbeddingTrainRunConfig {
	cfg.MatryoshkaDims = nil
	cfg.MatryoshkaWeights = nil
	cfg.TurboQuantPrefixBits = nil
	cfg.TurboQuantPrefixObjectives = nil
	cfg.TurboQuantPrefixWeight = 0
	cfg.TurboQuantPrefixSeed = 0
	cfg.TurboQuantPrefixScoreMode = ""
	cfg.TurboQuantCompactObjectives = nil
	cfg.TurboQuantRankMarginObjectives = nil
	cfg.TurboQuantRankMargin = 0
	return cfg
}

func mergeScoreSpectrumObjectiveNames(existing []string, added []string) []string {
	if len(existing) == 0 {
		return normalizeScoreSpectrumObjectiveNames(added)
	}
	merged := append([]string(nil), existing...)
	merged = append(merged, added...)
	return normalizeScoreSpectrumObjectiveNames(merged)
}

func (t *EmbeddingTrainer) applyTrainRunOverrides(cfg EmbeddingTrainRunConfig) error {
	if t == nil {
		return nil
	}
	next := t.config
	changed := false
	if cfg.LearningRate > 0 {
		next.LearningRate = cfg.LearningRate
		changed = true
	}
	if cfg.ContrastiveLoss != "" {
		next.ContrastiveLoss = cfg.ContrastiveLoss
		changed = true
	}
	if cfg.Temperature > 0 {
		next.Temperature = cfg.Temperature
		changed = true
	}
	if cfg.GroupedLossWeight > 0 {
		next.GroupedLossWeight = cfg.GroupedLossWeight
		changed = true
	}
	if cfg.TeacherLossWeightSet {
		next.TeacherLossWeight = cfg.TeacherLossWeight
		if cfg.TeacherLossWeight == 0 && len(cfg.TeacherSourceTemperatures) == 0 {
			next.TeacherSourceTemperatures = nil
		}
		if cfg.TeacherLossWeight == 0 && len(cfg.TeacherSourceWeights) == 0 {
			next.TeacherSourceWeights = nil
		}
		changed = true
	} else if cfg.TeacherLossWeight > 0 {
		next.TeacherLossWeight = cfg.TeacherLossWeight
		changed = true
	}
	if cfg.TeacherTemperature > 0 {
		next.TeacherTemperature = cfg.TeacherTemperature
		changed = true
	}
	if len(cfg.TeacherSourceTemperatures) > 0 {
		next.TeacherSourceTemperatures = normalizeHardNegativeTeacherTemperatures(cfg.TeacherSourceTemperatures)
		changed = true
	}
	if len(cfg.TeacherSourceWeights) > 0 {
		next.TeacherSourceWeights = normalizeHardNegativeTeacherWeights(cfg.TeacherSourceWeights)
		changed = true
	}
	if len(cfg.MatryoshkaDims) > 0 {
		next.MatryoshkaDims = append([]int(nil), cfg.MatryoshkaDims...)
		next.MatryoshkaWeights = append([]float32(nil), cfg.MatryoshkaWeights...)
		changed = true
	}
	if cfg.ClearTurboQuantPrefix {
		next.TurboQuantPrefixBits = nil
		next.TurboQuantPrefixObjectives = nil
		next.TurboQuantPrefixWeight = 0
		if len(next.TurboQuantCompactObjectives) == 0 && len(next.TurboQuantRankMarginObjectives) == 0 {
			next.TurboQuantPrefixSeed = 0
			next.TurboQuantPrefixScoreMode = ""
		}
		changed = true
	}
	if cfg.ClearTurboQuantRankMargin {
		next.TurboQuantRankMarginObjectives = nil
		next.TurboQuantRankMargin = 0
		if len(next.TurboQuantPrefixBits) == 0 && len(next.TurboQuantPrefixObjectives) == 0 && len(next.TurboQuantCompactObjectives) == 0 {
			next.TurboQuantPrefixSeed = 0
			next.TurboQuantPrefixScoreMode = ""
		}
		changed = true
	}
	if len(cfg.TurboQuantPrefixBits) > 0 {
		next.TurboQuantPrefixBits = append([]int(nil), cfg.TurboQuantPrefixBits...)
		next.TurboQuantPrefixObjectives = nil
		next.TurboQuantPrefixWeight = cfg.TurboQuantPrefixWeight
		next.TurboQuantPrefixSeed = cfg.TurboQuantPrefixSeed
		next.TurboQuantPrefixScoreMode = cfg.TurboQuantPrefixScoreMode
		changed = true
	}
	if len(cfg.TurboQuantRankMarginObjectives) > 0 {
		next.TurboQuantRankMarginObjectives = append([]TurboQuantPrefixObjective(nil), cfg.TurboQuantRankMarginObjectives...)
		next.TurboQuantRankMargin = cfg.TurboQuantRankMargin
		next.TurboQuantPrefixSeed = cfg.TurboQuantPrefixSeed
		next.TurboQuantPrefixScoreMode = cfg.TurboQuantPrefixScoreMode
		changed = true
	}
	if len(cfg.TurboQuantCompactObjectives) > 0 {
		next.TurboQuantCompactObjectives = append([]TurboQuantPrefixObjective(nil), cfg.TurboQuantCompactObjectives...)
		next.TurboQuantPrefixSeed = cfg.TurboQuantPrefixSeed
		changed = true
	}
	if len(cfg.TurboQuantPrefixObjectives) > 0 {
		next.TurboQuantPrefixObjectives = append([]TurboQuantPrefixObjective(nil), cfg.TurboQuantPrefixObjectives...)
		next.TurboQuantPrefixBits = nil
		next.TurboQuantPrefixWeight = cfg.TurboQuantPrefixWeight
		next.TurboQuantPrefixSeed = cfg.TurboQuantPrefixSeed
		next.TurboQuantPrefixScoreMode = cfg.TurboQuantPrefixScoreMode
		changed = true
	}
	if !changed {
		return nil
	}
	next = normalizedTrainConfig(next, t.tokenParam, t.attnQParam, t.attnKParam, t.attnVParam, t.attnOParam, t.hiddenParam, t.projParam)
	var err error
	next, err = normalizeMatryoshkaTrainConfig(next, trainerEmbeddingDim(t))
	if err != nil {
		return err
	}
	if err := validateTrainConfig(next); err != nil {
		return err
	}
	t.config = next
	return nil
}

func (t *EmbeddingTrainer) applyScoreSpectrumRunOverrides(cfg EmbeddingTrainRunConfig) error {
	if t == nil {
		return nil
	}
	next := t.config
	next.ScoreSpectrumLossMode = cfg.ScoreSpectrumLossMode
	next.ScoreSpectrumRecoveryWeight = cfg.ScoreSpectrumRecoveryWeight
	next.ScoreSpectrumRecoveryMargin = cfg.ScoreSpectrumRecoveryMargin
	next.ScoreSpectrumRecoveryTopK = cfg.ScoreSpectrumRecoveryTopK
	next.ScoreSpectrumRecoveryTau = cfg.ScoreSpectrumRecoveryTau
	next = normalizedTrainConfig(next, t.tokenParam, t.attnQParam, t.attnKParam, t.attnVParam, t.attnOParam, t.hiddenParam, t.projParam)
	if err := validateTrainConfig(next); err != nil {
		return err
	}
	t.config = next
	return nil
}

func (t *EmbeddingTrainer) syncTrainRunObjectiveConfig(cfg EmbeddingTrainRunConfig) EmbeddingTrainRunConfig {
	if t == nil {
		return cfg
	}
	if cfg.ContrastiveLoss == "" {
		cfg.ContrastiveLoss = t.config.ContrastiveLoss
	}
	if cfg.GroupedLossWeight == 0 {
		cfg.GroupedLossWeight = t.config.GroupedLossWeight
	}
	if cfg.TeacherLossWeight == 0 && !cfg.TeacherLossWeightSet {
		cfg.TeacherLossWeight = t.config.TeacherLossWeight
	}
	if cfg.TeacherTemperature == 0 {
		cfg.TeacherTemperature = t.config.TeacherTemperature
	}
	if len(cfg.TeacherSourceTemperatures) == 0 && len(t.config.TeacherSourceTemperatures) > 0 && !(cfg.TeacherLossWeightSet && cfg.TeacherLossWeight == 0) {
		cfg.TeacherSourceTemperatures = t.config.TeacherSourceTemperatures
	}
	if len(cfg.TeacherSourceWeights) == 0 && len(t.config.TeacherSourceWeights) > 0 && !(cfg.TeacherLossWeightSet && cfg.TeacherLossWeight == 0) {
		cfg.TeacherSourceWeights = t.config.TeacherSourceWeights
	}
	if len(cfg.MatryoshkaDims) == 0 && len(t.config.MatryoshkaDims) > 0 {
		cfg.MatryoshkaDims = append([]int(nil), t.config.MatryoshkaDims...)
		cfg.MatryoshkaWeights = append([]float32(nil), t.config.MatryoshkaWeights...)
	}
	if cfg.ClearTurboQuantPrefix {
		cfg.TurboQuantPrefixBits = nil
		cfg.TurboQuantPrefixObjectives = nil
		cfg.TurboQuantPrefixWeight = 0
		if len(cfg.TurboQuantCompactObjectives) == 0 && len(cfg.TurboQuantRankMarginObjectives) == 0 {
			cfg.TurboQuantPrefixSeed = 0
			cfg.TurboQuantPrefixScoreMode = ""
		} else if len(cfg.TurboQuantRankMarginObjectives) == 0 {
			cfg.TurboQuantPrefixScoreMode = ""
		}
	} else if len(cfg.TurboQuantPrefixObjectives) == 0 && len(cfg.TurboQuantPrefixBits) == 0 && len(t.config.TurboQuantPrefixBits) > 0 {
		cfg.TurboQuantPrefixBits = append([]int(nil), t.config.TurboQuantPrefixBits...)
		cfg.TurboQuantPrefixWeight = t.config.TurboQuantPrefixWeight
		cfg.TurboQuantPrefixSeed = t.config.TurboQuantPrefixSeed
		cfg.TurboQuantPrefixScoreMode = t.config.TurboQuantPrefixScoreMode
	}
	if cfg.ClearTurboQuantRankMargin {
		cfg.TurboQuantRankMarginObjectives = nil
		cfg.TurboQuantRankMargin = 0
		if len(cfg.TurboQuantPrefixBits) == 0 && len(cfg.TurboQuantPrefixObjectives) == 0 && len(cfg.TurboQuantCompactObjectives) == 0 {
			cfg.TurboQuantPrefixSeed = 0
			cfg.TurboQuantPrefixScoreMode = ""
		} else if len(cfg.TurboQuantPrefixBits) == 0 && len(cfg.TurboQuantPrefixObjectives) == 0 {
			cfg.TurboQuantPrefixScoreMode = ""
		}
	} else if !cfg.EvalOnly && len(cfg.TurboQuantRankMarginObjectives) == 0 && len(t.config.TurboQuantRankMarginObjectives) > 0 {
		cfg.TurboQuantRankMarginObjectives = append([]TurboQuantPrefixObjective(nil), t.config.TurboQuantRankMarginObjectives...)
		cfg.TurboQuantRankMargin = t.config.TurboQuantRankMargin
		cfg.TurboQuantPrefixSeed = t.config.TurboQuantPrefixSeed
		cfg.TurboQuantPrefixScoreMode = t.config.TurboQuantPrefixScoreMode
	}
	if !cfg.EvalOnly && len(cfg.TurboQuantCompactObjectives) == 0 && len(t.config.TurboQuantCompactObjectives) > 0 {
		cfg.TurboQuantCompactObjectives = append([]TurboQuantPrefixObjective(nil), t.config.TurboQuantCompactObjectives...)
		cfg.TurboQuantPrefixSeed = t.config.TurboQuantPrefixSeed
	}
	if !cfg.ClearTurboQuantPrefix && len(cfg.TurboQuantPrefixBits) == 0 && len(cfg.TurboQuantPrefixObjectives) == 0 && len(t.config.TurboQuantPrefixObjectives) > 0 {
		cfg.TurboQuantPrefixObjectives = append([]TurboQuantPrefixObjective(nil), t.config.TurboQuantPrefixObjectives...)
		cfg.TurboQuantPrefixWeight = t.config.TurboQuantPrefixWeight
		cfg.TurboQuantPrefixSeed = t.config.TurboQuantPrefixSeed
		cfg.TurboQuantPrefixScoreMode = t.config.TurboQuantPrefixScoreMode
	}
	if cfg.ScoreSpectrumLossMode == "" {
		cfg.ScoreSpectrumLossMode = t.config.ScoreSpectrumLossMode
	}
	if cfg.ScoreSpectrumRecoveryWeight == 0 {
		cfg.ScoreSpectrumRecoveryWeight = t.config.ScoreSpectrumRecoveryWeight
	}
	if cfg.ScoreSpectrumRecoveryMargin == 0 {
		cfg.ScoreSpectrumRecoveryMargin = t.config.ScoreSpectrumRecoveryMargin
	}
	if cfg.ScoreSpectrumRecoveryTopK == 0 {
		cfg.ScoreSpectrumRecoveryTopK = t.config.ScoreSpectrumRecoveryTopK
	}
	if cfg.ScoreSpectrumRecoveryTau == 0 {
		cfg.ScoreSpectrumRecoveryTau = t.config.ScoreSpectrumRecoveryTau
	}
	cfg.GroupedLossWeight = effectiveGroupedLossWeight(cfg.ContrastiveLoss, cfg.GroupedLossWeight)
	cfg.TeacherSourceTemperatures = normalizeHardNegativeTeacherTemperatures(cfg.TeacherSourceTemperatures)
	cfg.TeacherSourceWeights = normalizeHardNegativeTeacherWeights(cfg.TeacherSourceWeights)
	cfg.MatryoshkaDims, cfg.MatryoshkaWeights, _ = normalizeMatryoshkaDimsAndWeights(cfg.MatryoshkaDims, cfg.MatryoshkaWeights, trainerEmbeddingDim(t))
	cfg.TurboQuantPrefixBits, _ = normalizeTurboQuantPrefixBits(cfg.TurboQuantPrefixBits)
	cfg.TurboQuantPrefixObjectives, _ = normalizeTurboQuantPrefixObjectives(cfg.TurboQuantPrefixObjectives, cfg.MatryoshkaDims, trainerEmbeddingDim(t))
	cfg.TurboQuantRankMarginObjectives, _ = normalizeTurboQuantPrefixObjectives(cfg.TurboQuantRankMarginObjectives, cfg.MatryoshkaDims, trainerEmbeddingDim(t))
	cfg.TurboQuantCompactObjectives, _ = normalizeTurboQuantPrefixObjectives(cfg.TurboQuantCompactObjectives, cfg.MatryoshkaDims, trainerEmbeddingDim(t))
	if len(cfg.TurboQuantRankMarginObjectives) > 0 && cfg.TurboQuantRankMargin == 0 {
		cfg.TurboQuantRankMargin = effectiveTurboQuantRankMargin(cfg.TurboQuantRankMargin)
	}
	if len(cfg.TurboQuantPrefixBits) > 0 || len(cfg.TurboQuantPrefixObjectives) > 0 || len(cfg.TurboQuantCompactObjectives) > 0 || len(cfg.TurboQuantRankMarginObjectives) > 0 {
		if cfg.TurboQuantPrefixWeight == 0 {
			if len(cfg.TurboQuantPrefixBits) > 0 {
				cfg.TurboQuantPrefixWeight = 1
			}
		}
		cfg.TurboQuantPrefixSeed = effectiveTurboQuantPrefixSeed(cfg.TurboQuantPrefixSeed)
	}
	if len(cfg.TurboQuantPrefixBits) > 0 || len(cfg.TurboQuantPrefixObjectives) > 0 || len(cfg.TurboQuantRankMarginObjectives) > 0 {
		if mode, err := normalizeTurboQuantPrefixScoreMode(cfg.TurboQuantPrefixScoreMode); err == nil {
			cfg.TurboQuantPrefixScoreMode = mode
		}
	} else {
		cfg.TurboQuantPrefixScoreMode = ""
	}
	cfg = finalizedScoreSpectrumRunConfig(cfg)
	cfg.TeacherScoreNormalization = normalizeTeacherScoreNormalization(cfg.TeacherScoreNormalization)
	return cfg
}

func canonicalTrainSelectionMetric(metric string) string {
	switch metric {
	case "retrieval_ndcg_at_10":
		return "retrieval_ndcg"
	case "retrieval_map":
		return "retrieval_map_at_100"
	case "retrieval_recall":
		return "retrieval_recall_at_100"
	default:
		return metric
	}
}

func retrievalSelectionMetric(metric string) bool {
	switch canonicalTrainSelectionMetric(metric) {
	case "retrieval_ndcg", "retrieval_map_at_100", "retrieval_recall_at_100":
		return true
	default:
		return false
	}
}

func validTrainSelectionMetric(metric string) bool {
	switch canonicalTrainSelectionMetric(metric) {
	case "loss", "pair_accuracy", "threshold_accuracy", "score_margin", "auc", "top1_accuracy", "top5_accuracy", "top10_accuracy", "mrr", "mean_positive_rank", "mean_rank", "retrieval_ndcg", "retrieval_map_at_100", "retrieval_recall_at_100":
		return true
	case "score_spectrum_any_positive_top1", "score_spectrum_alternate_recovery", "score_spectrum_best_positive_hardest_negative_margin", "score_spectrum_original_positive_top1":
		return true
	default:
		return false
	}
}

func scoreSpectrumSelectionMetric(metric string) bool {
	switch metric {
	case "score_spectrum_any_positive_top1", "score_spectrum_alternate_recovery", "score_spectrum_best_positive_hardest_negative_margin", "score_spectrum_original_positive_top1":
		return true
	default:
		return false
	}
}

func betterEvalMetrics(current, best EmbeddingEvalMetrics, metric string, minDelta float32) bool {
	const eps = 1e-6
	primaryDelta := float32(math.Max(float64(minDelta), eps))
	switch canonicalTrainSelectionMetric(metric) {
	case "loss":
		if current.Loss < best.Loss-primaryDelta {
			return true
		}
		if math.Abs(float64(current.Loss-best.Loss)) <= eps {
			if current.ScoreMargin > best.ScoreMargin+float32(eps) {
				return true
			}
			if math.Abs(float64(current.ScoreMargin-best.ScoreMargin)) <= eps && current.PairAccuracy > best.PairAccuracy+float32(eps) {
				return true
			}
		}
	case "pair_accuracy":
		if current.PairAccuracy > best.PairAccuracy+primaryDelta {
			return true
		}
		if math.Abs(float64(current.PairAccuracy-best.PairAccuracy)) <= eps {
			if current.ScoreMargin > best.ScoreMargin+float32(eps) {
				return true
			}
			if math.Abs(float64(current.ScoreMargin-best.ScoreMargin)) <= eps && current.Loss < best.Loss-float32(eps) {
				return true
			}
		}
	case "threshold_accuracy":
		if current.ThresholdAccuracy > best.ThresholdAccuracy+primaryDelta {
			return true
		}
		if math.Abs(float64(current.ThresholdAccuracy-best.ThresholdAccuracy)) <= eps {
			if current.ROCAUC > best.ROCAUC+float32(eps) {
				return true
			}
			if math.Abs(float64(current.ROCAUC-best.ROCAUC)) <= eps && current.ScoreMargin > best.ScoreMargin+float32(eps) {
				return true
			}
		}
	case "auc", "top1_accuracy", "top5_accuracy", "top10_accuracy", "mrr", "retrieval_ndcg", "retrieval_map_at_100", "retrieval_recall_at_100":
		currentRankMetric := evalRankMetric(current, metric)
		bestRankMetric := evalRankMetric(best, metric)
		if currentRankMetric > bestRankMetric+primaryDelta {
			return true
		}
		if math.Abs(float64(currentRankMetric-bestRankMetric)) <= eps {
			if current.ScoreMargin > best.ScoreMargin+float32(eps) {
				return true
			}
			if math.Abs(float64(current.ScoreMargin-best.ScoreMargin)) <= eps && current.Loss < best.Loss-float32(eps) {
				return true
			}
		}
	case "mean_positive_rank", "mean_rank":
		if current.MeanPositiveRank < best.MeanPositiveRank-primaryDelta {
			return true
		}
		if math.Abs(float64(current.MeanPositiveRank-best.MeanPositiveRank)) <= eps {
			if current.Top1Accuracy > best.Top1Accuracy+float32(eps) {
				return true
			}
			if math.Abs(float64(current.Top1Accuracy-best.Top1Accuracy)) <= eps && current.ScoreMargin > best.ScoreMargin+float32(eps) {
				return true
			}
		}
	default:
		if current.ScoreMargin > best.ScoreMargin+primaryDelta {
			return true
		}
		if math.Abs(float64(current.ScoreMargin-best.ScoreMargin)) <= eps {
			if current.PairAccuracy > best.PairAccuracy+float32(eps) {
				return true
			}
			if math.Abs(float64(current.PairAccuracy-best.PairAccuracy)) <= eps && current.Loss < best.Loss-float32(eps) {
				return true
			}
		}
	}
	return false
}

func evalRankMetric(metrics EmbeddingEvalMetrics, metric string) float32 {
	switch canonicalTrainSelectionMetric(metric) {
	case "auc":
		return metrics.ROCAUC
	case "top5_accuracy":
		return metrics.Top5Accuracy
	case "top10_accuracy":
		return metrics.Top10Accuracy
	case "mrr":
		return metrics.MeanReciprocalRank
	case "retrieval_ndcg":
		return metrics.RetrievalNDCGAt10
	case "retrieval_map_at_100":
		return metrics.RetrievalMAPAt100
	case "retrieval_recall_at_100":
		return metrics.RetrievalRecallAt100
	default:
		return metrics.Top1Accuracy
	}
}

func cloneEvalMetrics(metrics EmbeddingEvalMetrics) *EmbeddingEvalMetrics {
	out := metrics
	return &out
}

func (t *EmbeddingTrainer) retrievalOnlyEvalMetrics(noPairwiseEval bool) (*EmbeddingEvalMetrics, error) {
	if t == nil || !noPairwiseEval || !t.retrievalEvalEnabled {
		return nil, nil
	}
	metrics := EmbeddingEvalMetrics{}
	if err := t.augmentRetrievalMetrics(&metrics); err != nil {
		return nil, err
	}
	return cloneEvalMetrics(metrics), nil
}

func cloneEvalMetricsPtr(metrics *EmbeddingEvalMetrics) *EmbeddingEvalMetrics {
	if metrics == nil {
		return nil
	}
	return cloneEvalMetrics(*metrics)
}

func betterScoreSpectrumEvalMetrics(current, best EmbeddingScoreSpectrumEvalMetrics, metric string, minDelta float32) bool {
	const eps = 1e-6
	primaryDelta := float32(math.Max(float64(minDelta), eps))
	currentMetric := scoreSpectrumEvalMetric(current, metric)
	bestMetric := scoreSpectrumEvalMetric(best, metric)
	if currentMetric > bestMetric+primaryDelta {
		return true
	}
	if math.Abs(float64(currentMetric-bestMetric)) <= eps {
		if current.BestPositiveHardestNegativeMargin > best.BestPositiveHardestNegativeMargin+float32(eps) {
			return true
		}
		if math.Abs(float64(current.BestPositiveHardestNegativeMargin-best.BestPositiveHardestNegativeMargin)) <= eps && current.Loss < best.Loss-float32(eps) {
			return true
		}
	}
	return false
}

func scoreSpectrumEvalMetric(metrics EmbeddingScoreSpectrumEvalMetrics, metric string) float32 {
	switch metric {
	case "score_spectrum_original_positive_top1":
		return metrics.OriginalPositiveTop1
	case "score_spectrum_alternate_recovery":
		return metrics.AlternateRelevantRecovery
	case "score_spectrum_best_positive_hardest_negative_margin":
		return metrics.BestPositiveHardestNegativeMargin
	default:
		return metrics.AnyPositiveTop1
	}
}

func cloneScoreSpectrumEvalMetrics(metrics EmbeddingScoreSpectrumEvalMetrics) *EmbeddingScoreSpectrumEvalMetrics {
	out := metrics
	return &out
}

func cloneScoreSpectrumEvalMetricsPtr(metrics *EmbeddingScoreSpectrumEvalMetrics) *EmbeddingScoreSpectrumEvalMetrics {
	if metrics == nil {
		return nil
	}
	return cloneScoreSpectrumEvalMetrics(*metrics)
}

func cloneListwiseGeometryEvalMetrics(metrics EmbeddingListwiseGeometryEvalMetrics) *EmbeddingListwiseGeometryEvalMetrics {
	out := metrics
	return &out
}

func cloneListwiseGeometryEvalMetricsPtr(metrics *EmbeddingListwiseGeometryEvalMetrics) *EmbeddingListwiseGeometryEvalMetrics {
	if metrics == nil {
		return nil
	}
	return cloneListwiseGeometryEvalMetrics(*metrics)
}
