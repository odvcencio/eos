package eosruntime

import (
	"fmt"
	"math"
	"math/rand"
	"os"
	"runtime/debug"
	"sort"
	"time"

	"m31labs.dev/eos/runtime/backend"
)

// vectorDistillProjectionState holds the train-time-only 128→384 distillation
// projection matrix and its AdamW moments.  It is NEVER persisted to disk;
// the served model uses the native 128-d student output exclusively.
type vectorDistillProjectionState struct {
	W            []float32 // shape [inputDim * teacherDim], row-major
	Mom1         []float32 // AdamW first moment
	Mom2         []float32 // AdamW second moment
	InputDim     int
	OutDim       int
	Step         int // AdamW step counter (separate from student step)
	ResidentName string
}

type vectorDistillBatchScratch struct {
	batch       []EmbeddingTokenizedVectorDistillExample
	inputs      []embeddingSequenceInput
	students    [][]float32
	teachers    [][]float32
	gradW       []float32
	projVec     []float32
	gradProj    []float32
	gradStudent []float32
}

type vectorDistillResidentBucket struct {
	seqLen      int
	uniqueSlots []int
	req         backend.CompactTrainForwardRequest
	handle      backend.CompactTrainHandle
	pooled      *backend.Tensor
}

type vectorDistillResidentForward struct {
	uniqueForInput []int
	uniqueEncoded  []*embeddingEncodedSequence
	buckets        []vectorDistillResidentBucket
	refs           []backend.CompactForwardResidentRef
	stepID         uint64
}

type pendingCompactResidentOptimizerUpdate struct {
	item       *CompactEmbeddingTrainTensor
	mom1, mom2 *backend.Tensor
	assignMom1 bool
	assignMom2 bool
	ref        backend.ResidentGradientRef
	cfg        backend.OptimizerUpdateConfig
}

// newVectorDistillProjectionState allocates and Kaiming-initialises a fresh
// projection matrix.
func newVectorDistillProjectionState(inputDim, outDim int, rng *rand.Rand) *vectorDistillProjectionState {
	n := inputDim * outDim
	W := make([]float32, n)
	// Kaiming uniform: fan_in = inputDim, bound = sqrt(1/inputDim)
	bound := float32(math.Sqrt(1.0 / float64(inputDim)))
	for i := range W {
		W[i] = (rng.Float32()*2 - 1) * bound
	}
	return &vectorDistillProjectionState{
		W:            W,
		Mom1:         make([]float32, n),
		Mom2:         make([]float32, n),
		InputDim:     inputDim,
		OutDim:       outDim,
		ResidentName: fmt.Sprintf("vector_distill_projection_%p", &W[0]),
	}
}

// FitVectorDistill trains the compact student using MSE+cosine distillation
// from a BGE-style teacher's 384-d embeddings.  The distillation projection
// (128→384) lives only in this function's scope and is never written to disk.
// The retrieval eval gate embeds via the native 128-d student path.
func (t *EmbeddingTrainer) FitVectorDistill(
	trainSet []EmbeddingTokenizedVectorDistillExample,
	cfg EmbeddingTrainRunConfig,
) (EmbeddingTrainRunSummary, error) {
	if t == nil {
		return EmbeddingTrainRunSummary{}, fmt.Errorf("embedding trainer is not initialized")
	}
	if err := t.failIfOptimizerPoisoned(); err != nil {
		return EmbeddingTrainRunSummary{}, err
	}
	cfg = normalizedTrainRunConfig(cfg)
	if cfg.EvalOnly {
		// eval-only path: retrieval eval only — no training data needed
	} else {
		if len(trainSet) == 0 {
			return EmbeddingTrainRunSummary{}, fmt.Errorf("vector distillation training dataset is empty")
		}
		if cfg.Epochs <= 0 {
			return EmbeddingTrainRunSummary{}, fmt.Errorf("epochs must be positive")
		}
		if cfg.BatchSize <= 0 {
			return EmbeddingTrainRunSummary{}, fmt.Errorf("batch_size must be positive")
		}
		if err := validateVectorDistillResearchGates(trainSet, cfg.AllowResearchOnlyVectorDistill); err != nil {
			return EmbeddingTrainRunSummary{}, err
		}
	}
	if cfg.EvalEveryEpoch <= 0 {
		return EmbeddingTrainRunSummary{}, fmt.Errorf("eval_every_epoch must be positive")
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
	optimizerSyncMode, err := NormalizeVectorDistillOptimizerSyncMode(cfg.VectorDistillOptimizerSync)
	if err != nil {
		return EmbeddingTrainRunSummary{}, err
	}
	cfg.VectorDistillOptimizerSync = optimizerSyncMode
	t.configureRetrievalEval(cfg.RetrievalEvalRuntime, cfg.RetrievalEval, cfg.RetrievalEvalTokenizer)
	if err := t.applyTrainRunOverrides(cfg); err != nil {
		return EmbeddingTrainRunSummary{}, err
	}
	cfg = t.syncTrainRunObjectiveConfig(cfg)
	previousDeferredSync := t.deferOptimizerSync
	t.deferOptimizerSync = optimizerSyncMode == VectorDistillOptimizerSyncDeferred
	defer func() {
		t.deferOptimizerSync = previousDeferredSync
	}()

	runStart := time.Now()
	startStep := t.step
	summary := EmbeddingTrainRunSummary{
		Config:                cfg,
		EffectiveLearningRate: t.config.LearningRate,
		StartProfile:          t.TrainProfile(),
		Workload:              EstimateVectorDistillTrainWorkload(len(trainSet), 0, cfg),
	}

	// --- eval-only path ---
	if cfg.EvalOnly {
		evalStart := time.Now()
		maybeReportEvalProgress(cfg, "eval_start", 0, t.step, 1, 0, 0, runStart)
		if retrievalMetrics, err := t.retrievalOnlyEvalMetrics(true); err != nil {
			return EmbeddingTrainRunSummary{}, fmt.Errorf("retrieval eval: %w", err)
		} else if retrievalMetrics != nil {
			summary.FinalEval = cloneEvalMetrics(*retrievalMetrics)
			summary.LastEval = cloneEvalMetrics(*retrievalMetrics)
			summary.BestEval = cloneEvalMetrics(*retrievalMetrics)
		}
		summary.EvalDuration = time.Since(evalStart)
		summary.StepsCompleted = t.step
		summary.BestStep = t.step
		summary.Workload.ActualEvalPasses = 1
		appendTrainEvalHistory(&summary, 0, t.step, "eval_only", true, summary.FinalEval, nil, nil)
		summary.EndProfile = t.TrainProfile()
		summary.DeltaProfile = diffTrainProfile(summary.StartProfile, summary.EndProfile)
		summary.Elapsed = time.Since(runStart)
		return summary, nil
	}

	// determine teacher dim from the first example
	if len(trainSet[0].TeacherVector) == 0 {
		return EmbeddingTrainRunSummary{}, fmt.Errorf("vector distillation: teacher_vector is empty in first example")
	}
	teacherDim := len(trainSet[0].TeacherVector)

	// distillation projection state — persists across epochs, discarded on return
	var proj *vectorDistillProjectionState
	rng := rand.New(rand.NewSource(cfg.Seed))

	indices := make([]int, len(trainSet))
	for i := range indices {
		indices[i] = i
	}
	var (
		bestCheckpoint EmbeddingTrainCheckpoint
		haveBest       bool
		noImproveEvals int
	)

	// recordEval runs the retrieval gate eval and returns updated metrics.
	recordEval := func(epoch int, trigger string) (*EmbeddingEvalMetrics, bool, error) {
		evalStart := time.Now()
		evalPass := summary.Workload.ActualEvalPasses + 1
		maybeReportEvalProgress(cfg, "eval_start", epoch, t.step, evalPass, 0, 0, runStart)
		var evalMetrics *EmbeddingEvalMetrics
		if retrievalMetrics, err := t.retrievalOnlyEvalMetrics(true); err != nil {
			return nil, false, err
		} else if retrievalMetrics != nil {
			evalMetrics = retrievalMetrics
			summary.LastEval = cloneEvalMetrics(*retrievalMetrics)
		}
		summary.EvalDuration += time.Since(evalStart)
		summary.Workload.ActualEvalPasses++
		maybeReportEvalProgress(cfg, "eval_done", epoch, t.step, evalPass, 0, 0, runStart)
		improved := false
		selectable := evalMetrics != nil && retrievalSelectionMetric(cfg.SelectMetric)
		if selectable && (!haveBest || betterEvalMetrics(*evalMetrics, *summary.BestEval, cfg.SelectMetric, cfg.MinDelta)) {
			checkpoint, err := t.Checkpoint()
			if err != nil {
				return nil, false, fmt.Errorf("checkpoint: %w", err)
			}
			bestCheckpoint = checkpoint
			haveBest = true
			improved = true
			summary.BestEval = cloneEvalMetrics(*evalMetrics)
			summary.BestEpoch = epoch
			summary.BestStep = t.step
			noImproveEvals = 0
		} else if selectable {
			noImproveEvals++
		}
		appendTrainEvalHistory(&summary, epoch, t.step, trigger, improved, evalMetrics, nil, nil)
		return evalMetrics, improved, nil
	}

	// initial eval before any training (when retrieval eval is configured and restore-best is on)
	if t.retrievalEvalEnabled && retrievalSelectionMetric(cfg.SelectMetric) && cfg.RestoreBest {
		if _, _, err := recordEval(0, "initial"); err != nil {
			return EmbeddingTrainRunSummary{}, fmt.Errorf("initial eval: %w", err)
		}
	}

	for epoch := 1; epoch <= cfg.Epochs; epoch++ {
		if cfg.Shuffle {
			rng.Shuffle(len(indices), func(i, j int) { indices[i], indices[j] = indices[j], indices[i] })
		}
		trainStart := time.Now()
		trainMetrics, updatedProj, err := t.runVectorDistillEpoch(trainSet, indices, cfg.BatchSize, cfg, teacherDim, proj, epoch, runStart)
		if err != nil {
			return EmbeddingTrainRunSummary{}, fmt.Errorf("epoch %d: %w", epoch, err)
		}
		proj = updatedProj
		summary.TrainDuration += time.Since(trainStart)
		record := EmbeddingTrainEpochSummary{
			Epoch: epoch,
			Step:  t.step,
			Train: trainMetrics,
		}
		summary.FinalTrain = trainMetrics
		summary.EpochsCompleted = epoch
		summary.Workload.CompletedEpochs = epoch
		summary.Workload.ActualTrainPairs += int64(len(indices))
		summary.Workload.ActualTrainExamples += int64(len(indices))

		if t.retrievalEvalEnabled && epoch%cfg.EvalEveryEpoch == 0 {
			evalMetrics, improved, err := recordEval(epoch, "epoch")
			if err != nil {
				return EmbeddingTrainRunSummary{}, fmt.Errorf("epoch %d eval: %w", epoch, err)
			}
			record.Eval = evalMetrics
			record.Improved = improved
			if t.retrievalEvalEnabled && retrievalSelectionMetric(cfg.SelectMetric) && !improved &&
				cfg.EarlyStoppingPatience > 0 && noImproveEvals >= cfg.EarlyStoppingPatience {
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

	// final eval
	if t.retrievalEvalEnabled {
		evalStart := time.Now()
		evalPass := summary.Workload.ActualEvalPasses + 1
		maybeReportEvalProgress(cfg, "eval_start", summary.EpochsCompleted, t.step, evalPass, 0, 0, runStart)
		if retrievalMetrics, err := t.retrievalOnlyEvalMetrics(true); err != nil {
			return EmbeddingTrainRunSummary{}, fmt.Errorf("final retrieval eval: %w", err)
		} else if retrievalMetrics != nil {
			summary.FinalEval = cloneEvalMetrics(*retrievalMetrics)
		}
		summary.EvalDuration += time.Since(evalStart)
		summary.Workload.ActualEvalPasses++
		maybeReportEvalProgress(cfg, "eval_done", summary.EpochsCompleted, t.step, summary.Workload.ActualEvalPasses, 0, 0, runStart)
		appendTrainEvalHistory(&summary, summary.EpochsCompleted, t.step, "final", false, summary.FinalEval, nil, nil)
		if summary.BestEval == nil {
			if summary.FinalEval != nil {
				summary.BestEval = cloneEvalMetrics(*summary.FinalEval)
			}
		}
	}

	summary.Workload.ActualTotalPairs = summary.Workload.ActualTrainPairs + summary.Workload.ActualEvalPairs
	summary.Workload.ActualTotalExamples = summary.Workload.ActualTrainExamples + summary.Workload.ActualEvalExamples
	if restored {
		summary.EndProfile = diffTrainProfile(preRestoreEndProfile, restoreStartProfile)
	} else {
		summary.EndProfile = t.TrainProfile()
	}
	summary.DeltaProfile = diffTrainProfile(summary.StartProfile, t.TrainProfile())
	summary.Elapsed = time.Since(runStart)
	return summary, nil
}

// runVectorDistillEpoch runs one epoch of vector-distillation training.
// It returns the epoch metrics and the (possibly newly-created) projection state.
func (t *EmbeddingTrainer) runVectorDistillEpoch(
	trainSet []EmbeddingTokenizedVectorDistillExample,
	order []int,
	batchSize int,
	cfg EmbeddingTrainRunConfig,
	teacherDim int,
	proj *vectorDistillProjectionState,
	epoch int,
	runStart time.Time,
) (EmbeddingTrainMetrics, *vectorDistillProjectionState, error) {
	totalLoss := float32(0)
	totalExamples := 0
	totalBatches := batchCount(len(order), batchSize, 1)
	batchIndex := 0
	var scratch vectorDistillBatchScratch

	for start := 0; start < len(order); start += batchSize {
		end := start + batchSize
		if end > len(order) {
			end = len(order)
		}
		batch := ensureVectorDistillExampleScratch(scratch.batch, end-start)
		for _, idx := range order[start:end] {
			batch = append(batch, trainSet[idx])
		}
		scratch.batch = batch

		maybeReportTrainProgress(cfg, EmbeddingTrainProgress{
			Phase:              "train_start",
			Epoch:              epoch,
			Batch:              batchIndex + 1,
			Batches:            totalBatches,
			Step:               t.step,
			BatchExamples:      len(batch),
			BatchPairs:         int64(len(batch)),
			EpochTrainExamples: int64(totalExamples),
			EpochTrainPairs:    int64(totalExamples),
			PlannedEpochPairs:  int64(len(order)),
			Elapsed:            time.Since(runStart),
		})

		metrics, updatedProj, err := t.trainVectorDistillBatchWithScratch(batch, proj, teacherDim, cfg.VectorDistillDefaultRole, cfg.VectorDistillRelationalWeight, &scratch)
		if err != nil {
			return EmbeddingTrainMetrics{}, proj, err
		}
		proj = updatedProj

		totalLoss += metrics.Loss * float32(len(batch))
		totalExamples += len(batch)
		batchIndex++

		if n := trainMemReclaimEvery(); n > 0 && batchIndex%n == 0 {
			debug.FreeOSMemory()
		}

		maybeReportTrainProgress(cfg, EmbeddingTrainProgress{
			Phase:              "train",
			Epoch:              epoch,
			Batch:              batchIndex,
			Batches:            totalBatches,
			Step:               t.step,
			BatchExamples:      len(batch),
			BatchPairs:         int64(len(batch)),
			EpochTrainExamples: int64(totalExamples),
			EpochTrainPairs:    int64(totalExamples),
			PlannedEpochPairs:  int64(len(order)),
			Loss:               metrics.Loss,
			Elapsed:            time.Since(runStart),
		})
	}

	if totalExamples == 0 {
		return EmbeddingTrainMetrics{}, proj, fmt.Errorf("vector distillation epoch has no usable examples")
	}
	return EmbeddingTrainMetrics{
		Loss:      totalLoss / float32(totalExamples),
		BatchSize: totalExamples,
	}, proj, nil
}

// vectorDistillRoleIndex resolves a per-example role string ("query",
// "document", "raw", or "" for legacy rows written before the role field
// existed) to the model's role-embedding index. Empty roles fall back to
// defaultRole; the returned bool tells the caller whether the fallback fired
// so it can log a one-time warning.
func (t *EmbeddingTrainer) vectorDistillRoleIndex(role, defaultRole string) (int32, bool, error) {
	usedFallback := false
	effective := role
	if effective == "" {
		effective = defaultRole
		usedFallback = true
	}
	switch effective {
	case EmbeddingRoleQuery:
		return t.queryRoleIndex(), usedFallback, nil
	case EmbeddingRoleDocument:
		return t.documentRoleIndex(), usedFallback, nil
	case EmbeddingRoleRaw:
		return t.rawRoleIndex(), usedFallback, nil
	default:
		return 0, false, fmt.Errorf("unsupported vector-distill role %q (want %q, %q, or %q)", effective, EmbeddingRoleQuery, EmbeddingRoleDocument, EmbeddingRoleRaw)
	}
}

// trainVectorDistillBatch runs one optimizer step over a batch of
// (text, teacher_vector) examples.  The distillation projection state is
// returned (created on first call) so the caller can persist it across batches.
// defaultRole is applied to examples with no explicit per-row role (legacy
// rows); relationalWeight, when > 0, activates the in-batch relational
// similarity-matrix term over the raw (pre-projection) student vectors.
func (t *EmbeddingTrainer) trainVectorDistillBatch(
	batch []EmbeddingTokenizedVectorDistillExample,
	proj *vectorDistillProjectionState,
	teacherDim int,
	defaultRole string,
	relationalWeight float32,
) (EmbeddingTrainMetrics, *vectorDistillProjectionState, error) {
	return t.trainVectorDistillBatchWithScratch(batch, proj, teacherDim, defaultRole, relationalWeight, nil)
}

func (t *EmbeddingTrainer) trainVectorDistillBatchWithScratch(
	batch []EmbeddingTokenizedVectorDistillExample,
	proj *vectorDistillProjectionState,
	teacherDim int,
	defaultRole string,
	relationalWeight float32,
	scratch *vectorDistillBatchScratch,
) (EmbeddingTrainMetrics, *vectorDistillProjectionState, error) {
	if !t.isCompactTrainer() {
		return EmbeddingTrainMetrics{}, proj, fmt.Errorf("vector distillation requires compact_transformer_v1")
	}
	if err := t.failIfOptimizerPoisoned(); err != nil {
		return EmbeddingTrainMetrics{}, proj, err
	}
	if len(batch) == 0 {
		return EmbeddingTrainMetrics{}, proj, fmt.Errorf("vector distillation batch is empty")
	}
	if defaultRole == "" {
		defaultRole = EmbeddingRoleQuery
	}

	// Build sequence inputs, resolving each example's own role (falling back
	// to defaultRole for legacy rows without an explicit "role" field).
	var inputs []embeddingSequenceInput
	if scratch != nil {
		inputs = ensureVectorDistillInputScratch(scratch.inputs, len(batch))
	} else {
		inputs = make([]embeddingSequenceInput, len(batch))
	}
	for i, ex := range batch {
		roleIndex, usedFallback, rerr := t.vectorDistillRoleIndex(ex.Role, defaultRole)
		if rerr != nil {
			return EmbeddingTrainMetrics{}, proj, fmt.Errorf("example %d (%s): %w", i, ex.ID, rerr)
		}
		if usedFallback && !t.vectorDistillDefaultRoleWarned {
			fmt.Fprintf(os.Stderr, "vector-distill: example %q has no explicit \"role\"; falling back to default role %q (add \"role\" to JSONL rows or pass --vector-distill-default-role to silence this warning)\n", ex.ID, defaultRole)
			t.vectorDistillDefaultRoleWarned = true
		}
		inputs[i] = embeddingSequenceInput{
			tokens: ex.Tokens,
			mask:   ex.Mask,
			role:   roleIndex,
			label:  fmt.Sprintf("batch %d", i),
		}
	}
	if scratch != nil {
		scratch.inputs = inputs
	}
	forward := t.prepareForwardWeights()
	if forward == nil || forward.compact == nil {
		return EmbeddingTrainMetrics{}, proj, fmt.Errorf("missing compact forward weights")
	}
	if compactResidentTrainEnabled() && t.compactTrainAccel != nil {
		return t.trainVectorDistillBatchResidentWithScratch(batch, inputs, proj, teacherDim, forward, relationalWeight, scratch)
	}
	encodeStart := time.Now()
	encoded, err := t.encodeSequenceInputs(inputs, forward, true)
	t.vectorDistillPhases.EncodeNanos += time.Since(encodeStart).Nanoseconds()
	t.vectorDistillPhases.EncodeCalls++
	if err != nil {
		return EmbeddingTrainMetrics{}, proj, err
	}
	defer t.releaseEncodedSequences(encoded)

	// Lazily initialise the projection on first call (student dim known now)
	if proj == nil {
		if len(encoded) == 0 || len(encoded[0].pooled) == 0 {
			return EmbeddingTrainMetrics{}, nil, fmt.Errorf("vector distillation: encoded sequence has empty pooled output")
		}
		studentDim := len(encoded[0].pooled)
		proj = newVectorDistillProjectionState(studentDim, teacherDim, rand.New(rand.NewSource(42)))
	}

	// Optional in-batch relational term over the RAW (pre-projection) student
	// vectors — computed once for the whole batch since it depends on every
	// pair, then merged per-example into gradStudent below.
	var relationalLoss float32
	var relationalGrads [][]float32
	if relationalWeight > 0 {
		var students [][]float32
		var teachers [][]float32
		if scratch != nil {
			students = ensureVectorDistillFloat32SlicesScratch(scratch.students, len(encoded))
			teachers = ensureVectorDistillFloat32SlicesScratch(scratch.teachers, len(encoded))
		} else {
			students = make([][]float32, len(encoded))
			teachers = make([][]float32, len(encoded))
		}
		for i, enc := range encoded {
			students[i] = enc.pooled
			teachers[i] = batch[i].TeacherVector
		}
		if scratch != nil {
			scratch.students = students
			scratch.teachers = teachers
		}
		relationalLoss, relationalGrads, err = t.vectorDistillRelationalLossAndGrad(students, teachers, relationalWeight)
		if err != nil {
			return EmbeddingTrainMetrics{}, proj, fmt.Errorf("relational loss: %w", err)
		}
	}

	grads, gradW, totalLoss, err := t.computeVectorDistillBatchGradientsWithScratch(batch, encoded, proj, forward, relationalGrads, scratch)
	if err != nil {
		return EmbeddingTrainMetrics{}, proj, err
	}

	// Scale: average over batch
	batchScale := float32(1) / float32(len(batch))

	// Update student parameters
	if err := t.applyCompactOptimizerUpdates(grads, batchScale); err != nil {
		return EmbeddingTrainMetrics{}, proj, err
	}

	// Update projection with its own AdamW (same LR/hyper-params as student)
	proj.Step++
	optimizerStart := time.Now()
	if err := t.applyVectorDistillProjectionAdamW(proj, gradW); err != nil {
		return EmbeddingTrainMetrics{}, proj, err
	}
	t.vectorDistillPhases.OptimizerNanos += time.Since(optimizerStart).Nanoseconds()
	t.vectorDistillPhases.OptimizerCalls++

	return EmbeddingTrainMetrics{
		// relationalLoss is already a batch-level mean (not per-example), so
		// it is added once rather than scaled by batchScale.
		Loss:      totalLoss*batchScale + relationalLoss,
		BatchSize: len(batch),
	}, proj, nil
}

func (t *EmbeddingTrainer) trainVectorDistillBatchResidentWithScratch(
	batch []EmbeddingTokenizedVectorDistillExample,
	inputs []embeddingSequenceInput,
	proj *vectorDistillProjectionState,
	teacherDim int,
	forward *embeddingForwardWeights,
	relationalWeight float32,
	scratch *vectorDistillBatchScratch,
) (EmbeddingTrainMetrics, *vectorDistillProjectionState, error) {
	if t == nil || !t.isCompactTrainer() || t.compactTrainAccel == nil {
		return EmbeddingTrainMetrics{}, proj, fmt.Errorf("compact resident vector-distill train is not available")
	}
	t.compactForwardSelected = false
	residentOpt, ok := t.optimizerAccel.(backend.ResidentGradientOptimizerAccelerator)
	if !ok {
		return EmbeddingTrainMetrics{}, proj, fmt.Errorf("compact resident vector-distill train requires resident-gradient optimizer")
	}
	residentForward, err := t.runVectorDistillResidentForward(inputs, forward)
	if err != nil {
		return EmbeddingTrainMetrics{}, proj, err
	}
	stepCompleted := false
	aborted := false
	defer func() {
		if !stepCompleted && !aborted {
			_ = t.compactTrainAccel.AbortCompactTrainStep(residentForward.stepID)
		}
	}()
	encoded := residentForward.uniqueEncoded
	if proj == nil {
		if len(encoded) == 0 || len(encoded[0].pooled) == 0 {
			return EmbeddingTrainMetrics{}, nil, fmt.Errorf("vector distillation: resident encoded sequence has empty pooled output")
		}
		proj = newVectorDistillProjectionState(len(encoded[0].pooled), teacherDim, rand.New(rand.NewSource(42)))
	}
	var relationalLoss float32
	var relationalGrads [][]float32
	if relationalWeight > 0 {
		students := make([][]float32, len(batch))
		teachers := make([][]float32, len(batch))
		for i, unique := range residentForward.uniqueForInput {
			students[i] = encoded[unique].pooled
			teachers[i] = batch[i].TeacherVector
		}
		relationalLoss, relationalGrads, err = t.vectorDistillRelationalLossAndGrad(students, teachers, relationalWeight)
		if err != nil {
			return EmbeddingTrainMetrics{}, proj, fmt.Errorf("relational loss: %w", err)
		}
	}
	uniqueGradPooled, gradW, totalLoss, err := t.computeVectorDistillResidentPooledGradientsWithScratch(batch, residentForward, proj, relationalGrads, scratch)
	if err != nil {
		return EmbeddingTrainMetrics{}, proj, err
	}
	var residentGradRefs []backend.ResidentGradientRef
	backwardStart := time.Now()
	for bi := range residentForward.buckets {
		bucket := &residentForward.buckets[bi]
		gradData := make([]float32, 0, len(bucket.uniqueSlots)*bucket.req.Shape.OutputDim)
		for _, unique := range bucket.uniqueSlots {
			grad := uniqueGradPooled[unique]
			if len(grad) != bucket.req.Shape.OutputDim {
				return EmbeddingTrainMetrics{}, proj, fmt.Errorf("compact resident train pooled grad for unique %d has dim %d, want %d", unique, len(grad), bucket.req.Shape.OutputDim)
			}
			gradData = append(gradData, grad...)
		}
		result, err := t.compactTrainAccel.RunCompactTrainBackward(backend.CompactTrainBackwardRequest{
			Handle:     bucket.handle,
			GradPooled: backend.NewTensorF32([]int{len(bucket.uniqueSlots), bucket.req.Shape.OutputDim}, gradData),
		})
		if err != nil {
			return EmbeddingTrainMetrics{}, proj, fmt.Errorf("compact resident train backward T=%d B=%d: %w", bucket.seqLen, len(bucket.uniqueSlots), err)
		}
		bucket.handle = backend.CompactTrainHandle{}
		residentGradRefs = result.ResidentGradRefs
	}
	t.vectorDistillPhases.BackwardNanos += time.Since(backwardStart).Nanoseconds()
	t.vectorDistillPhases.BackwardCalls++
	if err := t.compactTrainAccel.EndCompactTrainStep(residentForward.stepID); err != nil {
		_ = t.compactTrainAccel.AbortCompactTrainStep(residentForward.stepID)
		aborted = true
		return EmbeddingTrainMetrics{}, proj, fmt.Errorf("compact resident train end step: %w", err)
	}
	batchScale := float32(1) / float32(len(batch))
	optimizerStart := time.Now()
	pendingUpdates, err := t.preflightCompactOptimizerUpdatesWithResidentGrads(residentOpt, residentGradRefs, batchScale, int(residentForward.stepID))
	if err != nil {
		t.compactForwardSelected = false
		return EmbeddingTrainMetrics{}, proj, err
	}
	prospectiveProjStep := proj.Step + 1
	if err := t.preflightVectorDistillProjectionAdamWForStep(proj, gradW, prospectiveProjStep); err != nil {
		t.compactForwardSelected = false
		return EmbeddingTrainMetrics{}, proj, err
	}
	if err := t.applyPreflightedCompactOptimizerUpdatesWithResidentGrads(residentOpt, pendingUpdates); err != nil {
		return EmbeddingTrainMetrics{}, proj, t.poisonAfterOptimizerLaunch(err)
	}
	if err := t.applyVectorDistillProjectionAdamWForStepStrict(proj, gradW, prospectiveProjStep); err != nil {
		return EmbeddingTrainMetrics{}, proj, t.poisonAfterOptimizerLaunch(err)
	}
	t.step = int(residentForward.stepID)
	t.compactState.Step = t.step
	t.compactOptimizerUpdates++
	t.compactForwardSelected = true
	proj.Step = prospectiveProjStep
	t.vectorDistillPhases.OptimizerNanos += time.Since(optimizerStart).Nanoseconds()
	t.vectorDistillPhases.OptimizerCalls++
	stepCompleted = true
	return EmbeddingTrainMetrics{
		Loss:      totalLoss*batchScale + relationalLoss,
		BatchSize: len(batch),
	}, proj, nil
}

func (t *EmbeddingTrainer) runVectorDistillResidentForward(inputs []embeddingSequenceInput, forward *embeddingForwardWeights) (*vectorDistillResidentForward, error) {
	if t == nil || forward == nil || forward.compact == nil {
		return nil, fmt.Errorf("compact resident train forward requires compact weights")
	}
	t.compactForwardSelected = false
	compactForward := forward.compact
	if err := t.prepareCompactTrainAccelerator(compactForward); err != nil {
		return nil, err
	}
	out := &vectorDistillResidentForward{
		uniqueForInput: make([]int, len(inputs)),
		stepID:         uint64(t.step + 1),
	}
	sequenceCache := map[string]int{}
	groupOrder := make([]int, 0)
	groups := map[int][]int{}
	tokensByUnique := make([][]int32, 0, len(inputs))
	masksByUnique := make([][]int32, 0, len(inputs))
	rolesByUnique := make([]int32, 0, len(inputs))
	for i, input := range inputs {
		mask, err := t.prepareMask(input.tokens, input.mask)
		if err != nil {
			return nil, fmt.Errorf("%s: %w", embeddingSequenceInputLabel(input, i), err)
		}
		normalizedMask := normalizeCompactForwardMask(mask)
		key := embeddingBatchSequenceKey(input.tokens, normalizedMask, input.role)
		unique, ok := sequenceCache[key]
		if !ok {
			unique = len(tokensByUnique)
			sequenceCache[key] = unique
			tokensByUnique = append(tokensByUnique, append([]int32(nil), input.tokens...))
			masksByUnique = append(masksByUnique, normalizedMask)
			rolesByUnique = append(rolesByUnique, input.role)
			seqLen := len(input.tokens)
			if _, exists := groups[seqLen]; !exists {
				groupOrder = append(groupOrder, seqLen)
			}
			groups[seqLen] = append(groups[seqLen], unique)
		}
		out.uniqueForInput[i] = unique
	}
	sort.Ints(groupOrder)
	refs, err := t.compactForwardResidentRefs(compactForward)
	if err != nil {
		return nil, err
	}
	out.refs = refs
	for _, seqLen := range groupOrder {
		uniqueSlots := groups[seqLen]
		req := backend.CompactTrainForwardRequest{
			Shape:        t.compactForwardShape(compactForward, len(uniqueSlots), seqLen),
			Tokens:       make([][]int32, len(uniqueSlots)),
			Masks:        make([][]int32, len(uniqueSlots)),
			Roles:        make([]int32, len(uniqueSlots)),
			ResidentRefs: refs,
			GELUMode:     compactForwardGELUMode(),
			StepID:       out.stepID,
		}
		for i, unique := range uniqueSlots {
			req.Tokens[i] = tokensByUnique[unique]
			req.Masks[i] = masksByUnique[unique]
			req.Roles[i] = rolesByUnique[unique]
		}
		if preflight, ok := t.compactTrainAccel.(backend.CompactTrainPreflight); ok {
			if err := preflight.PreflightCompactTrainForward(req); err != nil {
				return nil, fmt.Errorf("compact resident train forward preflight T=%d B=%d: %w", seqLen, len(uniqueSlots), err)
			}
		}
		out.buckets = append(out.buckets, vectorDistillResidentBucket{seqLen: seqLen, uniqueSlots: uniqueSlots, req: req})
	}
	if len(out.buckets) == 0 {
		return nil, fmt.Errorf("compact resident train has no exact-length buckets")
	}
	if err := t.compactTrainAccel.BeginCompactTrainStep(out.stepID, refs); err != nil {
		return nil, fmt.Errorf("compact resident train begin step: %w", err)
	}
	began := true
	defer func() {
		if began {
			_ = t.compactTrainAccel.AbortCompactTrainStep(out.stepID)
		}
	}()
	out.uniqueEncoded = make([]*embeddingEncodedSequence, len(tokensByUnique))
	encodeStart := time.Now()
	for bi := range out.buckets {
		bucket := &out.buckets[bi]
		result, err := t.compactTrainAccel.RunCompactTrainForward(bucket.req)
		if err != nil {
			_ = t.releaseVectorDistillResidentHandles(out)
			return nil, fmt.Errorf("compact resident train forward T=%d B=%d: %w", bucket.seqLen, len(bucket.uniqueSlots), err)
		}
		bucket.handle = result.Handle
		bucket.pooled = result.Pooled
		if result.Pooled == nil || len(result.Pooled.Shape) != 2 || result.Pooled.Shape[0] != len(bucket.uniqueSlots) || result.Pooled.Shape[1] != bucket.req.Shape.OutputDim {
			_ = t.releaseVectorDistillResidentHandles(out)
			return nil, fmt.Errorf("compact resident train forward T=%d B=%d pooled shape %v, want [%d %d]", bucket.seqLen, len(bucket.uniqueSlots), tensorShapeForError(result.Pooled), len(bucket.uniqueSlots), bucket.req.Shape.OutputDim)
		}
		for row, unique := range bucket.uniqueSlots {
			start := row * bucket.req.Shape.OutputDim
			end := start + bucket.req.Shape.OutputDim
			out.uniqueEncoded[unique] = &embeddingEncodedSequence{
				pooled: append([]float32(nil), result.Pooled.F32[start:end]...),
				tokens: tokensByUnique[unique],
				role:   rolesByUnique[unique],
			}
		}
	}
	t.vectorDistillPhases.EncodeNanos += time.Since(encodeStart).Nanoseconds()
	t.vectorDistillPhases.EncodeCalls++
	began = false
	return out, nil
}

func (t *EmbeddingTrainer) releaseVectorDistillResidentHandles(forward *vectorDistillResidentForward) error {
	if t == nil || t.compactTrainAccel == nil || forward == nil {
		return nil
	}
	var firstErr error
	for i := range forward.buckets {
		handle := forward.buckets[i].handle
		if handle.Token == nil {
			continue
		}
		if err := t.compactTrainAccel.ReleaseCompactTrainHandle(handle); err != nil && firstErr == nil {
			firstErr = err
		}
		forward.buckets[i].handle = backend.CompactTrainHandle{}
	}
	return firstErr
}

func (t *EmbeddingTrainer) computeVectorDistillResidentPooledGradientsWithScratch(
	batch []EmbeddingTokenizedVectorDistillExample,
	forward *vectorDistillResidentForward,
	proj *vectorDistillProjectionState,
	relationalGrads [][]float32,
	scratch *vectorDistillBatchScratch,
) ([][]float32, []float32, float32, error) {
	if forward == nil || len(forward.uniqueEncoded) == 0 {
		return nil, nil, 0, fmt.Errorf("compact resident train has no encoded rows")
	}
	uniqueGrads := make([][]float32, len(forward.uniqueEncoded))
	for i, enc := range forward.uniqueEncoded {
		if enc == nil || len(enc.pooled) == 0 {
			return nil, nil, 0, fmt.Errorf("compact resident train unique row %d has empty pooled output", i)
		}
		uniqueGrads[i] = make([]float32, len(enc.pooled))
	}
	if relationalGrads != nil && len(relationalGrads) != len(batch) {
		return nil, nil, 0, fmt.Errorf("compact resident train relational gradient count %d, want %d", len(relationalGrads), len(batch))
	}
	var gradW []float32
	if scratch != nil {
		gradW = ensureFloat32Scratch(scratch.gradW, proj.InputDim*proj.OutDim)
		scratch.gradW = gradW
	} else {
		gradW = make([]float32, proj.InputDim*proj.OutDim)
	}
	totalLoss := float32(0)
	for i, unique := range forward.uniqueForInput {
		enc := forward.uniqueEncoded[unique]
		student := enc.pooled
		teacher := batch[i].TeacherVector
		projectionLossStart := time.Now()
		var projVec []float32
		if scratch != nil {
			projVec = ensureFloat32Scratch(scratch.projVec, proj.OutDim)
			scratch.projVec = projVec
		} else {
			projVec = make([]float32, proj.OutDim)
		}
		if err := t.projectVectorDistillStudent(student, proj, projVec); err != nil {
			return nil, nil, 0, fmt.Errorf("example %d projection: %w", i, err)
		}
		var lossResult VectorDistillLossResult
		var lerr error
		if scratch != nil {
			lossResult, lerr = vectorDistillLossAndGradInto(projVec, teacher, scratch.gradProj)
			scratch.gradProj = lossResult.GradProj
		} else {
			lossResult, lerr = VectorDistillLossAndGrad(projVec, teacher)
		}
		if lerr != nil {
			return nil, nil, 0, fmt.Errorf("example %d: %w", i, lerr)
		}
		totalLoss += lossResult.Loss
		t.vectorDistillPhases.ProjectionLossNanos += time.Since(projectionLossStart).Nanoseconds()
		t.vectorDistillPhases.ProjectionLossCalls++
		backwardStart := time.Now()
		gradStudent := make([]float32, proj.InputDim)
		if err := t.accumulateVectorDistillProjectionGrads(student, lossResult.GradProj, proj, gradStudent, gradW); err != nil {
			return nil, nil, 0, fmt.Errorf("example %d projection backward: %w", i, err)
		}
		if relationalGrads != nil {
			if len(relationalGrads[i]) != len(gradStudent) {
				return nil, nil, 0, fmt.Errorf("compact resident train relational gradient %d dim %d, want %d", i, len(relationalGrads[i]), len(gradStudent))
			}
			n := float32(len(batch))
			for k, g := range relationalGrads[i] {
				gradStudent[k] += g * n
			}
		}
		for k, g := range gradStudent {
			uniqueGrads[unique][k] += g
		}
		t.vectorDistillPhases.BackwardNanos += time.Since(backwardStart).Nanoseconds()
		t.vectorDistillPhases.BackwardCalls++
	}
	return uniqueGrads, gradW, totalLoss, nil
}

func (t *EmbeddingTrainer) applyCompactOptimizerUpdatesWithResidentGrads(residentOpt backend.ResidentGradientOptimizerAccelerator, refs []backend.ResidentGradientRef, scale float32, stepID int) error {
	pending, err := t.preflightCompactOptimizerUpdatesWithResidentGrads(residentOpt, refs, scale, stepID)
	if err != nil {
		return err
	}
	return t.applyPreflightedCompactOptimizerUpdatesWithResidentGrads(residentOpt, pending)
}

func (t *EmbeddingTrainer) preflightCompactOptimizerUpdatesWithResidentGrads(residentOpt backend.ResidentGradientOptimizerAccelerator, refs []backend.ResidentGradientRef, scale float32, stepID int) ([]pendingCompactResidentOptimizerUpdate, error) {
	if t == nil || t.compactState == nil {
		return nil, nil
	}
	if residentOpt == nil {
		return nil, fmt.Errorf("compact resident train requires resident-gradient optimizer")
	}
	refByName := make(map[string]backend.ResidentGradientRef, len(refs))
	for _, ref := range refs {
		if ref.Name == "" {
			return nil, fmt.Errorf("compact resident train returned empty gradient ref name")
		}
		if _, exists := refByName[ref.Name]; exists {
			return nil, fmt.Errorf("compact resident train returned duplicate gradient ref %q", ref.Name)
		}
		refByName[ref.Name] = ref
	}
	expected := map[string]bool{}
	for _, item := range compactTrainStateOptimizerItems(t.compactState) {
		if item == nil || item.Name == "" || item.Tensor == nil {
			continue
		}
		expected[item.Name] = true
		ref, ok := refByName[item.Name]
		if !ok {
			return nil, fmt.Errorf("compact resident train missing gradient ref %q", item.Name)
		}
		if ref.Elements != len(item.Tensor.F32) {
			return nil, fmt.Errorf("compact resident train gradient ref %q elements %d, want %d", item.Name, ref.Elements, len(item.Tensor.F32))
		}
	}
	for name := range refByName {
		if !expected[name] {
			return nil, fmt.Errorf("compact resident train unexpected gradient ref %q", name)
		}
	}
	preflight, ok := residentOpt.(backend.ResidentGradientOptimizerPreflightAccelerator)
	if !ok {
		return nil, fmt.Errorf("compact resident train requires resident-gradient optimizer preflight")
	}
	items := compactTrainStateOptimizerItems(t.compactState)
	pending := make([]pendingCompactResidentOptimizerUpdate, 0, len(items))
	for _, item := range items {
		if item == nil || item.Name == "" || item.Tensor == nil {
			continue
		}
		mom1, mom2 := item.Moment1, item.Moment2
		assignMom1, assignMom2 := false, false
		if mom1 == nil {
			mom1 = zeroLikeMaster(item.Tensor)
			assignMom1 = true
		}
		if mom2 == nil {
			mom2 = zeroLikeMaster(item.Tensor)
			assignMom2 = true
		}
		cfg := t.optimizerUpdateConfig(scale)
		cfg.Step = stepID
		cfg.DeferSync = t.deferOptimizerSync && t.canBridgeResidentOptimizerParam(item.Name, item.Tensor)
		pending = append(pending, pendingCompactResidentOptimizerUpdate{
			item: item, mom1: mom1, mom2: mom2,
			assignMom1: assignMom1, assignMom2: assignMom2,
			ref: refByName[item.Name], cfg: cfg,
		})
	}
	// Validate the complete exact ordered set before assigning moments or
	// launching the first compact student update.
	for _, update := range pending {
		if err := preflight.PreflightApplyUpdateWithResidentGrad(update.item.Name, update.cfg, update.item.Tensor, update.mom1, update.mom2, update.ref); err != nil {
			return nil, fmt.Errorf("compact resident optimizer preflight %q: %w", update.item.Name, err)
		}
	}
	return pending, nil
}

func (t *EmbeddingTrainer) applyPreflightedCompactOptimizerUpdatesWithResidentGrads(residentOpt backend.ResidentGradientOptimizerAccelerator, pending []pendingCompactResidentOptimizerUpdate) error {
	if t == nil || residentOpt == nil {
		return nil
	}
	for _, update := range pending {
		if update.assignMom1 {
			update.item.Moment1 = update.mom1
		}
		if update.assignMom2 {
			update.item.Moment2 = update.mom2
		}
	}
	for _, update := range pending {
		if err := residentOpt.ApplyUpdateWithResidentGrad(update.item.Name, update.cfg, update.item.Tensor, update.mom1, update.mom2, update.ref); err != nil {
			return fmt.Errorf("compact resident optimizer update %q: %w", update.item.Name, err)
		}
	}
	t.momentsDirty = true
	t.invalidateForwardWeights()
	return nil
}

// computeVectorDistillBatchGradients runs the per-example projection forward
// pass, the pointwise MSE+cosine loss, and backprop through the projection
// and encoder for every example in the batch, merging in the (optional)
// in-batch relational gradient for each example's raw student vector. It
// returns the RAW (pre-batchScale, pre-AdamW) accumulated encoder gradient
// state, the raw projection-weight gradient, and the summed (not yet
// batch-scaled) pointwise loss. This is split out of trainVectorDistillBatch
// so tests can inspect the merged gradient directly — the effective weight of
// the relational term (see the "Pre-multiply by len(batch)" comment below) is
// only observable before the caller's batchScale/AdamW step, since AdamW's
// per-coordinate normalization on the very first optimizer step erases most
// gradient-magnitude information.
func (t *EmbeddingTrainer) computeVectorDistillBatchGradients(
	batch []EmbeddingTokenizedVectorDistillExample,
	encoded []*embeddingEncodedSequence,
	proj *vectorDistillProjectionState,
	forward *embeddingForwardWeights,
	relationalGrads [][]float32,
) (*compactEmbeddingGradState, []float32, float32, error) {
	return t.computeVectorDistillBatchGradientsWithScratch(batch, encoded, proj, forward, relationalGrads, nil)
}

func (t *EmbeddingTrainer) computeVectorDistillBatchGradientsWithScratch(
	batch []EmbeddingTokenizedVectorDistillExample,
	encoded []*embeddingEncodedSequence,
	proj *vectorDistillProjectionState,
	forward *embeddingForwardWeights,
	relationalGrads [][]float32,
	scratch *vectorDistillBatchScratch,
) (*compactEmbeddingGradState, []float32, float32, error) {
	grads := newCompactEmbeddingGradState(t.compactState)
	var gradW []float32
	if scratch != nil {
		gradW = ensureFloat32Scratch(scratch.gradW, proj.InputDim*proj.OutDim)
		scratch.gradW = gradW
	} else {
		gradW = make([]float32, proj.InputDim*proj.OutDim)
	}
	totalLoss := float32(0)

	for i, enc := range encoded {
		student := enc.pooled
		teacher := batch[i].TeacherVector

		// Forward through projection: proj_vec = W @ student
		projectionLossStart := time.Now()
		var projVec []float32
		if scratch != nil {
			projVec = ensureFloat32Scratch(scratch.projVec, proj.OutDim)
			scratch.projVec = projVec
		} else {
			projVec = make([]float32, proj.OutDim)
		}
		if err := t.projectVectorDistillStudent(student, proj, projVec); err != nil {
			return nil, nil, 0, fmt.Errorf("example %d projection: %w", i, err)
		}

		// Loss and gradient w.r.t. projected vector
		var lossResult VectorDistillLossResult
		var lerr error
		if scratch != nil {
			lossResult, lerr = vectorDistillLossAndGradInto(projVec, teacher, scratch.gradProj)
			scratch.gradProj = lossResult.GradProj
		} else {
			lossResult, lerr = VectorDistillLossAndGrad(projVec, teacher)
		}
		if lerr != nil {
			return nil, nil, 0, fmt.Errorf("example %d: %w", i, lerr)
		}
		totalLoss += lossResult.Loss
		t.vectorDistillPhases.ProjectionLossNanos += time.Since(projectionLossStart).Nanoseconds()
		t.vectorDistillPhases.ProjectionLossCalls++

		// Backprop through projection → gradStudent and gradW
		backwardStart := time.Now()
		var gradStudent []float32
		if scratch != nil {
			gradStudent = ensureFloat32Scratch(scratch.gradStudent, proj.InputDim)
			scratch.gradStudent = gradStudent
		} else {
			gradStudent = make([]float32, proj.InputDim)
		}
		if err := t.accumulateVectorDistillProjectionGrads(student, lossResult.GradProj, proj, gradStudent, gradW); err != nil {
			return nil, nil, 0, fmt.Errorf("example %d projection backward: %w", i, err)
		}

		// Merge in the relational term's gradient w.r.t. the same raw student
		// vector (computed directly, not through the projection).
		//
		// VectorDistillRelationalLossAndGrad returns the complete analytic
		// gradient of the whole-batch relational loss. The batchScale the
		// caller applies afterwards (1/len(batch)) is correct for the
		// pointwise term — it converts a sum over examples into a mean —
		// but it would incorrectly shrink the relational term's effective
		// weight by 1/N as well, since that term is already a batch-level
		// (not per-example-summed) quantity. Pre-multiply by len(batch) here
		// so the later batchScale exactly cancels, keeping the relational
		// term's effective weight equal to relationalWeight regardless of
		// batch size. See TestVectorDistillRelationalGradientMergeIsBatchSizeInvariant.
		if relationalGrads != nil {
			n := float32(len(batch))
			for k, g := range relationalGrads[i] {
				gradStudent[k] += g * n
			}
		}
		t.vectorDistillPhases.BackwardNanos += time.Since(backwardStart).Nanoseconds()
		t.vectorDistillPhases.BackwardCalls++

		// Backprop gradStudent through the encoder
		if berr := t.backpropCompactEncodedSequence(enc, gradStudent, forward.compact, grads); berr != nil {
			return nil, nil, 0, fmt.Errorf("example %d backprop: %w", i, berr)
		}
	}

	return grads, gradW, totalLoss, nil
}

func (t *EmbeddingTrainer) projectVectorDistillStudent(student []float32, proj *vectorDistillProjectionState, out []float32) error {
	if proj == nil || len(student) == 0 || len(out) != proj.OutDim {
		return nil
	}
	if name := proj.residentName(); name != "" {
		weights := backend.NewTensorF32([]int{proj.InputDim, proj.OutDim}, proj.W)
		if accelerated, ok, err := t.tryTrainerMatMulBoundRightChecked(student, 1, proj.InputDim, name, weights, false, false); err != nil {
			return err
		} else if ok && len(accelerated) == proj.OutDim {
			copy(out, accelerated)
			return nil
		}
		if err := t.syncVectorDistillProjectionState(proj, "host_fallback"); err != nil {
			return err
		}
	}
	if accelerated, ok := t.tryTrainerMatMul(student, 1, proj.InputDim, proj.W, proj.InputDim, proj.OutDim, false, false); ok && len(accelerated) == proj.OutDim {
		copy(out, accelerated)
		return nil
	}
	for si := 0; si < proj.InputDim; si++ {
		base := si * proj.OutDim
		sv := student[si]
		for k := 0; k < proj.OutDim; k++ {
			out[k] += proj.W[base+k] * sv
		}
	}
	return nil
}

func (t *EmbeddingTrainer) accumulateVectorDistillProjectionGrads(
	student []float32,
	gradProj []float32,
	proj *vectorDistillProjectionState,
	gradStudent []float32,
	gradW []float32,
) error {
	if proj == nil || len(gradProj) == 0 || len(student) == 0 {
		return nil
	}
	studentGradOK := false
	if name := proj.residentName(); name != "" {
		weights := backend.NewTensorF32([]int{proj.InputDim, proj.OutDim}, proj.W)
		if accelerated, ok, err := t.tryTrainerMatMulBoundRightChecked(gradProj, 1, proj.OutDim, name, weights, false, true); err != nil {
			return err
		} else if ok && len(accelerated) == proj.InputDim {
			for i, v := range accelerated {
				gradStudent[i] += v
			}
			studentGradOK = true
		} else {
			if err := t.syncVectorDistillProjectionState(proj, "host_fallback"); err != nil {
				return err
			}
		}
	}
	if !studentGradOK {
		if accelerated, ok := t.tryTrainerMatMul(gradProj, 1, proj.OutDim, proj.W, proj.InputDim, proj.OutDim, false, true); ok && len(accelerated) == proj.InputDim {
			for i, v := range accelerated {
				gradStudent[i] += v
			}
			studentGradOK = true
		}
	}
	weightGradOK := false
	if accelerated, ok := t.tryTrainerMatMul(student, 1, proj.InputDim, gradProj, 1, proj.OutDim, true, false); ok && len(accelerated) == len(gradW) {
		for i, v := range accelerated {
			gradW[i] += v
		}
		weightGradOK = true
	}
	if studentGradOK && weightGradOK {
		return nil
	}
	if studentGradOK {
		for i := 0; i < proj.InputDim; i++ {
			si := student[i]
			base := i * proj.OutDim
			for k := 0; k < proj.OutDim; k++ {
				gradW[base+k] += si * gradProj[k]
			}
		}
		return nil
	}
	if weightGradOK {
		for i := 0; i < proj.InputDim; i++ {
			base := i * proj.OutDim
			var gs float32
			for k := 0; k < proj.OutDim; k++ {
				gs += proj.W[base+k] * gradProj[k]
			}
			gradStudent[i] += gs
		}
		return nil
	}
	accumulateVectorDistillProjectionGrads(student, gradProj, proj.W, gradStudent, gradW)
	return nil
}

func (t *EmbeddingTrainer) applyVectorDistillProjectionAdamW(proj *vectorDistillProjectionState, gradW []float32) error {
	if proj == nil {
		return nil
	}
	return t.applyVectorDistillProjectionAdamWForStep(proj, gradW, proj.Step)
}

func (t *EmbeddingTrainer) preflightVectorDistillProjectionAdamWForStep(proj *vectorDistillProjectionState, gradW []float32, step int) error {
	if proj == nil || len(proj.W) == 0 || step <= 0 {
		return nil
	}
	if len(gradW) != len(proj.W) {
		return fmt.Errorf("vector-distill projection optimizer preflight: grad size %d does not match weight size %d", len(gradW), len(proj.W))
	}
	if proj.InputDim <= 0 || proj.OutDim <= 0 || proj.InputDim*proj.OutDim != len(proj.W) {
		return fmt.Errorf("vector-distill projection optimizer preflight: shape [%d %d] does not match weight size %d", proj.InputDim, proj.OutDim, len(proj.W))
	}
	if len(proj.Mom1) != len(proj.W) || len(proj.Mom2) != len(proj.W) {
		return fmt.Errorf("vector-distill projection optimizer preflight: moment sizes %d/%d do not match weight size %d", len(proj.Mom1), len(proj.Mom2), len(proj.W))
	}
	if t == nil || t.optimizerAccel == nil {
		return nil
	}
	preflight, ok := t.optimizerAccel.(backend.OptimizerPreflightAccelerator)
	if !ok {
		return fmt.Errorf("vector-distill projection optimizer preflight requires optimizer preflight support")
	}
	shape := []int{proj.InputDim, proj.OutDim}
	weights := backend.NewTensorF32(shape, proj.W)
	mom1 := backend.NewTensorF32(shape, proj.Mom1)
	mom2 := backend.NewTensorF32(shape, proj.Mom2)
	name := proj.residentName()
	cfg := backend.OptimizerUpdateConfig{
		Optimizer:    t.config.Optimizer,
		Step:         step,
		LearningRate: t.config.LearningRate,
		WeightDecay:  t.config.WeightDecay,
		Beta1:        t.config.Beta1,
		Beta2:        t.config.Beta2,
		Epsilon:      t.config.Epsilon,
		Scale:        1,
		DeferSync:    t.deferOptimizerSync && t.canBridgeResidentOptimizerParam(name, weights),
	}
	if err := preflight.PreflightApplyUpdate(name, cfg, weights, mom1, mom2, backend.NewTensorF32(shape, gradW)); err != nil {
		return fmt.Errorf("vector-distill projection optimizer preflight %q: %w", name, err)
	}
	return nil
}

func (t *EmbeddingTrainer) applyVectorDistillProjectionAdamWForStep(proj *vectorDistillProjectionState, gradW []float32, step int) error {
	return t.applyVectorDistillProjectionAdamWForStepWithFallback(proj, gradW, step, true)
}

func (t *EmbeddingTrainer) applyVectorDistillProjectionAdamWForStepStrict(proj *vectorDistillProjectionState, gradW []float32, step int) error {
	return t.applyVectorDistillProjectionAdamWForStepWithFallback(proj, gradW, step, false)
}

func (t *EmbeddingTrainer) applyVectorDistillProjectionAdamWForStepWithFallback(proj *vectorDistillProjectionState, gradW []float32, step int, allowHostFallback bool) error {
	if proj == nil || len(proj.W) == 0 || step <= 0 {
		return nil
	}
	if t != nil && t.optimizerAccel != nil && len(gradW) == len(proj.W) {
		shape := []int{proj.InputDim, proj.OutDim}
		weights := backend.NewTensorF32(shape, proj.W)
		mom1 := backend.NewTensorF32(shape, proj.Mom1)
		mom2 := backend.NewTensorF32(shape, proj.Mom2)
		name := proj.residentName()
		deferSync := t.deferOptimizerSync && t.canBridgeResidentOptimizerParam(name, weights)
		cfg := backend.OptimizerUpdateConfig{
			Optimizer:    t.config.Optimizer,
			Step:         step,
			LearningRate: t.config.LearningRate,
			WeightDecay:  t.config.WeightDecay,
			Beta1:        t.config.Beta1,
			Beta2:        t.config.Beta2,
			Epsilon:      t.config.Epsilon,
			Scale:        1,
			DeferSync:    deferSync,
		}
		if err := t.optimizerAccel.ApplyUpdate(name, cfg, weights, mom1, mom2, backend.NewTensorF32(shape, gradW)); err == nil {
			t.momentsDirty = true
			if deferSync {
				if err := t.bindForwardMatrix(name, weights); err != nil {
					if syncErr := t.syncVectorDistillProjectionState(proj, "resident_bind_fallback"); syncErr != nil {
						return fmt.Errorf("vector-distill projection bind: %w; forced sync: %v", err, syncErr)
					}
				}
				return nil
			}
			if err := t.syncOptimizerBinding(name, weights, mom1, mom2, true, "host_fallback"); err == nil {
				copy(proj.W, weights.F32)
				copy(proj.Mom1, mom1.F32)
				copy(proj.Mom2, mom2.F32)
			} else {
				return err
			}
			return nil
		} else if !allowHostFallback || deferSync || t.hasResidentOptimizerParam(name) {
			return fmt.Errorf("vector-distill projection optimizer update: %w", err)
		}
	}
	applyVectorDistillProjectionAdamW(
		proj.W, proj.Mom1, proj.Mom2, gradW,
		t.config.LearningRate, t.config.Beta1, t.config.Beta2, t.config.Epsilon, t.config.WeightDecay,
		step,
	)
	return nil
}

func (p *vectorDistillProjectionState) residentName() string {
	if p == nil {
		return ""
	}
	if p.ResidentName == "" {
		p.ResidentName = fmt.Sprintf("vector_distill_projection_%p", p)
	}
	return p.ResidentName
}

func (t *EmbeddingTrainer) syncVectorDistillProjectionState(proj *vectorDistillProjectionState, reason string) error {
	if t == nil || proj == nil || len(proj.W) == 0 {
		return nil
	}
	if err := t.failIfOptimizerPoisoned(); err != nil {
		return err
	}
	if t.optimizerAccel == nil {
		return nil
	}
	shape := []int{proj.InputDim, proj.OutDim}
	weights := backend.NewTensorF32(shape, proj.W)
	mom1 := backend.NewTensorF32(shape, proj.Mom1)
	mom2 := backend.NewTensorF32(shape, proj.Mom2)
	if err := t.syncOptimizerBinding(proj.residentName(), weights, mom1, mom2, true, reason); err != nil {
		return err
	}
	copy(proj.W, weights.F32)
	copy(proj.Mom1, mom1.F32)
	copy(proj.Mom2, mom2.F32)
	return nil
}

func (t *EmbeddingTrainer) vectorDistillRelationalLossAndGrad(students, teachers [][]float32, weight float32) (float32, [][]float32, error) {
	n := len(students)
	if weight <= 0 || n < 2 {
		return 0, nil, nil
	}
	if len(teachers) != n {
		return 0, nil, fmt.Errorf("vector_distill_relational: student count %d != teacher count %d", n, len(teachers))
	}
	studentData, studentDim, studentOK := flattenVectorDistillRelationalRows(students)
	teacherData, teacherDim, teacherOK := flattenVectorDistillRelationalRows(teachers)
	var studentScores, teacherScores []float32
	if studentOK {
		if out, ok := t.tryTrainerMatMul(studentData, n, studentDim, studentData, n, studentDim, false, true); ok && len(out) == n*n {
			studentScores = out
		}
	}
	if teacherOK {
		if out, ok := t.tryTrainerMatMul(teacherData, n, teacherDim, teacherData, n, teacherDim, false, true); ok && len(out) == n*n {
			teacherScores = out
		}
	}
	return vectorDistillRelationalLossAndGradWithScores(students, teachers, weight, studentScores, teacherScores)
}

func flattenVectorDistillRelationalRows(rows [][]float32) ([]float32, int, bool) {
	if len(rows) == 0 || len(rows[0]) == 0 {
		return nil, 0, false
	}
	width := len(rows[0])
	data := make([]float32, 0, len(rows)*width)
	for _, row := range rows {
		if len(row) != width {
			return nil, 0, false
		}
		data = append(data, row...)
	}
	return data, width, true
}

func ensureVectorDistillExampleScratch(s []EmbeddingTokenizedVectorDistillExample, n int) []EmbeddingTokenizedVectorDistillExample {
	if cap(s) < n {
		return make([]EmbeddingTokenizedVectorDistillExample, 0, n)
	}
	return s[:0]
}

func ensureVectorDistillInputScratch(s []embeddingSequenceInput, n int) []embeddingSequenceInput {
	if cap(s) < n {
		return make([]embeddingSequenceInput, n)
	}
	s = s[:n]
	clear(s)
	return s
}

func ensureVectorDistillFloat32SlicesScratch(s [][]float32, n int) [][]float32 {
	if cap(s) < n {
		return make([][]float32, n)
	}
	s = s[:n]
	clear(s)
	return s
}

// EstimateVectorDistillTrainWorkload returns a planned-work summary for
// vector-distillation runs (one example = one training pair).
func EstimateVectorDistillTrainWorkload(trainExamples, evalExamples int, cfg EmbeddingTrainRunConfig) EmbeddingTrainWorkload {
	cfg = normalizedTrainRunConfig(cfg)
	batches := batchCount(trainExamples, cfg.BatchSize, 1)
	evalPasses := plannedEvalPassCount(evalExamples, cfg.Epochs, cfg.EvalEveryEpoch)
	trainPairsPerEpoch := int64(trainExamples)
	if cfg.EvalOnly {
		batches = 0
		trainPairsPerEpoch = 0
		evalPasses = 1 // retrieval-only eval
	} else {
		if cfg.RestoreBest && trainExamples > 0 {
			evalPasses++
		}
		if cfg.EvalEverySteps > 0 && trainExamples > 0 {
			evalPasses += (batches / cfg.EvalEverySteps) * cfg.Epochs
		}
		// initial eval pass if restoring best
		if cfg.RestoreBest && trainExamples > 0 {
			evalPasses++
		}
	}
	evalPairsPerPass := int64(evalExamples)
	return EmbeddingTrainWorkload{
		TrainMode:            "vector_distill",
		EvalMode:             workloadEvalModeVectorDistill(evalExamples, cfg),
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

func workloadEvalModeVectorDistill(evalExamples int, cfg EmbeddingTrainRunConfig) string {
	if cfg.RetrievalEval.CorpusPath != "" {
		return "retrieval"
	}
	if evalExamples > 0 {
		return "pairwise"
	}
	return ""
}
