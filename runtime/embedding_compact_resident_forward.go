package eosruntime

import (
	"fmt"
	"sort"

	"m31labs.dev/eos/runtime/backend"
)

const (
	// These limits protect the legacy host-packed path when a caller disables
	// resident training or runs on a host-only backend. They are deliberately
	// conservative for the 16 GiB training host: the estimate covers only
	// sequence activations and excludes Go/runtime overhead, gradients, and
	// model weights. A resident backend has no such limit because its activation
	// arena is sized and validated by the backend before launch.
	compactPackedScoreSpectrumMaxCandidates = 8192
	compactPackedScoreSpectrumMaxTokens     = 1 << 20
	compactPackedScoreSpectrumMaxBytes      = 1 << 30
	compactResidentMaxUint64                = ^uint64(0)
)

// guardCompactScoreSpectrumPackedBatch rejects a legacy packed batch before
// building candidateInputs or allocating any token-level activation state.
// It is intentionally based on the caller's already-materialized rows and
// performs checked arithmetic so malformed/adversarial dimensions fail with a
// useful diagnostic instead of an integer overflow followed by a huge make.
func (t *EmbeddingTrainer) guardCompactScoreSpectrumPackedBatch(batch []EmbeddingScoreSpectrumExample) error {
	if t == nil || !t.isCompactTrainer() {
		return nil
	}
	var candidates uint64
	var tokens uint64
	for row, example := range batch {
		rowCandidates := uint64(len(example.CandidateTokens))
		if rowCandidates == 0 {
			continue
		}
		if candidates > compactResidentMaxUint64-rowCandidates {
			return fmt.Errorf("compact packed score-spectrum row %d candidate count overflows; reduce batch size or enable resident training before allocating activations", row)
		}
		candidates += rowCandidates
		if candidates > compactPackedScoreSpectrumMaxCandidates {
			return fmt.Errorf("compact packed score-spectrum batch has %d candidates (limit %d); enable resident training or reduce batch size before allocating activations", candidates, compactPackedScoreSpectrumMaxCandidates)
		}
		rowTokens := uint64(len(example.QueryTokens))
		for _, candidate := range example.CandidateTokens {
			if rowTokens > compactResidentMaxUint64-uint64(len(candidate)) {
				return fmt.Errorf("compact packed score-spectrum row %d token count overflows; enable resident training before allocating activations", row)
			}
			rowTokens += uint64(len(candidate))
		}
		if tokens > compactResidentMaxUint64-rowTokens {
			return fmt.Errorf("compact packed score-spectrum token count overflows; enable resident training before allocating activations")
		}
		tokens += rowTokens
		if tokens > compactPackedScoreSpectrumMaxTokens {
			return fmt.Errorf("compact packed score-spectrum batch has %d tokens (limit %d); enable resident training or reduce batch size before allocating activations", tokens, compactPackedScoreSpectrumMaxTokens)
		}
	}

	// Each token participates in several D-wide activation rows and one
	// attention score matrix per layer. The score matrix dominates for long
	// sequences; use a checked, intentionally over-estimated lower bound that
	// catches unsafe batches without needing to touch the model tensors.
	if tokens > 0 {
		modelDim := uint64(1)
		layers := uint64(1)
		if state := t.compactState; state != nil {
			if state.Manifest.ModelDim > 0 {
				modelDim = uint64(state.Manifest.ModelDim)
			}
			if state.Manifest.EncoderRepeats > 0 {
				layers = uint64(state.Manifest.EncoderRepeats)
			}
		}
		// 16 float32 rows per layer cover Q/K/V, residual/hidden/FFN outputs,
		// gradients, and retained intermediates; add 4 bytes per attention
		// score cell as a separate term. All products are overflow checked.
		rowBytes, ok := checkedUint64Product(tokens, modelDim, layers, 16*4)
		if !ok {
			return fmt.Errorf("compact packed score-spectrum activation estimate overflows; enable resident training before allocating activations")
		}
		seqLenBound := uint64(1)
		for _, example := range batch {
			if uint64(len(example.QueryTokens)) > seqLenBound {
				seqLenBound = uint64(len(example.QueryTokens))
			}
			for _, candidate := range example.CandidateTokens {
				if uint64(len(candidate)) > seqLenBound {
					seqLenBound = uint64(len(candidate))
				}
			}
		}
		scoreCells, ok := checkedUint64Product(candidates, seqLenBound, seqLenBound, layers, 4)
		if !ok || compactResidentMaxUint64-rowBytes < scoreCells {
			return fmt.Errorf("compact packed score-spectrum activation estimate overflows; enable resident training before allocating activations")
		}
		if rowBytes+scoreCells > compactPackedScoreSpectrumMaxBytes {
			return fmt.Errorf("compact packed score-spectrum estimated activations are %d bytes (limit %d); enable resident training or reduce batch size before allocating activations", rowBytes+scoreCells, compactPackedScoreSpectrumMaxBytes)
		}
	}
	return nil
}

func checkedUint64Product(values ...uint64) (uint64, bool) {
	product := uint64(1)
	for _, value := range values {
		if value != 0 && product > compactResidentMaxUint64/value {
			return 0, false
		}
		product *= value
	}
	return product, true
}

// runCompactResidentForward runs the compact resident-train forward ABI for
// the unique exact-length sequence buckets in inputs. It intentionally
// returns only pooled outputs and opaque liveness handles: all token-level
// activations remain owned by the accelerator until the matching backward
// call consumes each handle. Callers must either backprop every returned
// bucket or abort the step.
//
// The returned type is shared with vector distillation because both training
// objectives have the same resident-forward lifetime and duplicate-sequence
// semantics. In particular, uniqueForInput maps every original row back to a
// single pooled output, so callers can sum row-local pooled gradients before
// issuing one backward per unique sequence.
func (t *EmbeddingTrainer) runCompactResidentForward(inputs []embeddingSequenceInput, forward *embeddingForwardWeights) (*vectorDistillResidentForward, error) {
	if t == nil || forward == nil || forward.compact == nil {
		return nil, fmt.Errorf("compact resident train forward requires compact weights")
	}
	if t.compactTrainAccel == nil {
		return nil, fmt.Errorf("compact resident train accelerator is not initialized")
	}
	if len(inputs) == 0 {
		return nil, fmt.Errorf("compact resident train forward has no inputs")
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
	for bi := range out.buckets {
		bucket := &out.buckets[bi]
		result, err := t.compactTrainAccel.RunCompactTrainForward(bucket.req)
		if err != nil {
			_ = t.releaseVectorDistillResidentHandles(out)
			return nil, fmt.Errorf("compact resident train forward T=%d B=%d: %w", bucket.seqLen, len(bucket.uniqueSlots), err)
		}
		bucket.handle = result.Handle
		bucket.pooled = result.Pooled
		if result.Pooled == nil || result.Pooled.DType != "f32" || len(result.Pooled.Shape) != 2 || result.Pooled.Shape[0] != len(bucket.uniqueSlots) || result.Pooled.Shape[1] != bucket.req.Shape.OutputDim || len(result.Pooled.F32) != len(bucket.uniqueSlots)*bucket.req.Shape.OutputDim {
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
	began = false
	return out, nil
}

// runCompactScoreSpectrumBatchUpdateResident performs the score-spectrum
// objective against pooled outputs returned by the resident-train ABI. The
// score/loss code is intentionally shared with the host packed path; only the
// encoder forward/backward and optimizer publication change. This keeps hard,
// soft, and recovery terms exactly row-local and preserves all positives and
// eligible candidates.
func (t *EmbeddingTrainer) runCompactScoreSpectrumBatchUpdateResident(batch []EmbeddingScoreSpectrumExample) (EmbeddingTrainMetrics, error) {
	if t == nil || t.compactState == nil {
		return EmbeddingTrainMetrics{}, fmt.Errorf("compact embedding trainer is not initialized")
	}
	if len(batch) == 0 {
		return EmbeddingTrainMetrics{}, fmt.Errorf("score-spectrum training batch is empty")
	}
	if len(t.config.MatryoshkaDims) > 0 {
		return EmbeddingTrainMetrics{}, fmt.Errorf("compact_transformer_v1 score-spectrum training does not support matryoshka objectives")
	}
	if len(t.config.TurboQuantPrefixBits) > 0 || len(t.config.TurboQuantPrefixObjectives) > 0 || len(turboQuantPrefixObjectivesForConfig(t.config)) > 0 {
		return EmbeddingTrainMetrics{}, fmt.Errorf("compact_transformer_v1 score-spectrum training does not support turboquant prefix objectives")
	}
	if len(t.config.TurboQuantCompactObjectives) > 0 || len(turboQuantCompactObjectivesForConfig(t.config)) > 0 {
		return EmbeddingTrainMetrics{}, fmt.Errorf("compact_transformer_v1 score-spectrum training does not support turboquant compact objectives")
	}
	if len(t.config.TurboQuantRankMarginObjectives) > 0 || len(turboQuantRankMarginObjectivesForConfig(t.config)) > 0 {
		return EmbeddingTrainMetrics{}, fmt.Errorf("compact_transformer_v1 score-spectrum training does not support turboquant rank-margin objectives")
	}
	if err := validateScoreSpectrumTrainerConfig(t.config); err != nil {
		return EmbeddingTrainMetrics{}, err
	}
	residentOpt, ok := t.optimizerAccel.(backend.ResidentGradientOptimizerAccelerator)
	if !ok {
		return EmbeddingTrainMetrics{}, fmt.Errorf("compact resident score-spectrum training requires resident-gradient optimizer")
	}
	canonicalBatch, err := canonicalizeTokenizedScoreSpectrumExamples(batch)
	if err != nil {
		return EmbeddingTrainMetrics{}, err
	}
	queryInputs, candidateInputs, candidateSpans, err := scoreSpectrumSequenceInputs(canonicalBatch, t.queryRoleIndex(), t.documentRoleIndex())
	if err != nil {
		return EmbeddingTrainMetrics{}, err
	}
	forward := t.prepareForwardWeights()
	if forward == nil || forward.compact == nil {
		return EmbeddingTrainMetrics{}, fmt.Errorf("missing compact forward weights")
	}
	allInputs := make([]embeddingSequenceInput, 0, len(queryInputs)+len(candidateInputs))
	allInputs = append(allInputs, queryInputs...)
	allInputs = append(allInputs, candidateInputs...)
	residentForward, err := t.runCompactResidentForward(allInputs, forward)
	if err != nil {
		return EmbeddingTrainMetrics{}, err
	}
	stepCompleted := false
	aborted := false
	defer func() {
		if !stepCompleted && !aborted {
			_ = t.compactTrainAccel.AbortCompactTrainStep(residentForward.stepID)
		}
	}()

	queries := make([]*embeddingEncodedSequence, len(queryInputs))
	for i := range queries {
		unique := residentForward.uniqueForInput[i]
		if unique < 0 || unique >= len(residentForward.uniqueEncoded) {
			return EmbeddingTrainMetrics{}, fmt.Errorf("compact resident score-spectrum query %d maps to invalid unique row %d", i, unique)
		}
		queries[i] = residentForward.uniqueEncoded[unique]
	}
	candidates := make([]*embeddingEncodedSequence, len(candidateInputs))
	for i := range candidates {
		inputIndex := len(queryInputs) + i
		unique := residentForward.uniqueForInput[inputIndex]
		if unique < 0 || unique >= len(residentForward.uniqueEncoded) {
			return EmbeddingTrainMetrics{}, fmt.Errorf("compact resident score-spectrum candidate %d maps to invalid unique row %d", i, unique)
		}
		candidates[i] = residentForward.uniqueEncoded[unique]
	}
	queryGrads := make([][]float32, len(queries))
	candidateGrads := make([][]float32, len(candidates))
	for i, query := range queries {
		if query == nil || len(query.pooled) == 0 {
			return EmbeddingTrainMetrics{}, fmt.Errorf("compact resident score-spectrum query %d has empty pooled output", i)
		}
		queryGrads[i] = make([]float32, len(query.pooled))
	}
	for i, candidate := range candidates {
		if candidate == nil || len(candidate.pooled) == 0 {
			return EmbeddingTrainMetrics{}, fmt.Errorf("compact resident score-spectrum candidate %d has empty pooled output", i)
		}
		candidateGrads[i] = make([]float32, len(candidate.pooled))
	}
	totalLoss, totalScore, pairCount, err := accumulateScoreSpectrumGrads(queries, candidates, candidateSpans, canonicalBatch, t.config, queryGrads, candidateGrads)
	if err != nil {
		return EmbeddingTrainMetrics{}, err
	}
	if pairCount == 0 {
		return EmbeddingTrainMetrics{}, fmt.Errorf("score-spectrum training batch has no usable candidates")
	}
	uniqueGradPooled := make([][]float32, len(residentForward.uniqueEncoded))
	for unique, encoded := range residentForward.uniqueEncoded {
		if encoded == nil || len(encoded.pooled) == 0 {
			return EmbeddingTrainMetrics{}, fmt.Errorf("compact resident score-spectrum unique row %d has empty pooled output", unique)
		}
		uniqueGradPooled[unique] = make([]float32, len(encoded.pooled))
	}
	for i, grad := range queryGrads {
		unique := residentForward.uniqueForInput[i]
		addFloat32Slice(uniqueGradPooled[unique], grad)
	}
	for i, grad := range candidateGrads {
		unique := residentForward.uniqueForInput[len(queryInputs)+i]
		addFloat32Slice(uniqueGradPooled[unique], grad)
	}

	var residentGradRefs []backend.ResidentGradientRef
	for bi := range residentForward.buckets {
		bucket := &residentForward.buckets[bi]
		gradData := make([]float32, 0, len(bucket.uniqueSlots)*bucket.req.Shape.OutputDim)
		for _, unique := range bucket.uniqueSlots {
			if unique < 0 || unique >= len(uniqueGradPooled) {
				return EmbeddingTrainMetrics{}, fmt.Errorf("compact resident score-spectrum bucket T=%d maps to invalid unique row %d", bucket.seqLen, unique)
			}
			grad := uniqueGradPooled[unique]
			if len(grad) != bucket.req.Shape.OutputDim {
				return EmbeddingTrainMetrics{}, fmt.Errorf("compact resident score-spectrum pooled grad for unique %d has dim %d, want %d", unique, len(grad), bucket.req.Shape.OutputDim)
			}
			gradData = append(gradData, grad...)
		}
		result, err := t.compactTrainAccel.RunCompactTrainBackward(backend.CompactTrainBackwardRequest{
			Handle:     bucket.handle,
			GradPooled: backend.NewTensorF32([]int{len(bucket.uniqueSlots), bucket.req.Shape.OutputDim}, gradData),
		})
		if err != nil {
			return EmbeddingTrainMetrics{}, fmt.Errorf("compact resident score-spectrum backward T=%d B=%d: %w", bucket.seqLen, len(bucket.uniqueSlots), err)
		}
		bucket.handle = backend.CompactTrainHandle{}
		// The backend returns cumulative step-scoped gradient handles. Keep the
		// final response (after the last bucket) for the resident optimizer.
		residentGradRefs = result.ResidentGradRefs
	}
	if err := t.compactTrainAccel.EndCompactTrainStep(residentForward.stepID); err != nil {
		_ = t.compactTrainAccel.AbortCompactTrainStep(residentForward.stepID)
		aborted = true
		return EmbeddingTrainMetrics{}, fmt.Errorf("compact resident score-spectrum end step: %w", err)
	}
	batchScale := float32(1) / float32(len(queries))
	pendingUpdates, err := t.preflightCompactOptimizerUpdatesWithResidentGrads(residentOpt, residentGradRefs, batchScale, int(residentForward.stepID))
	if err != nil {
		t.compactForwardSelected = false
		return EmbeddingTrainMetrics{}, err
	}
	if err := t.applyPreflightedCompactOptimizerUpdatesWithResidentGrads(residentOpt, pendingUpdates); err != nil {
		return EmbeddingTrainMetrics{}, t.poisonAfterOptimizerLaunch(err)
	}
	if releaser, ok := t.compactTrainAccel.(backend.CompactTrainGradientReleaser); ok {
		if err := releaser.ReleaseCompactTrainGradients(residentForward.stepID); err != nil {
			return EmbeddingTrainMetrics{}, t.poisonAfterOptimizerLaunch(err)
		}
	}
	t.step = int(residentForward.stepID)
	t.compactState.Step = t.step
	t.compactOptimizerUpdates++
	t.compactForwardSelected = true
	stepCompleted = true
	return EmbeddingTrainMetrics{
		Loss:         totalLoss * batchScale,
		AverageScore: totalScore / float32(pairCount),
		BatchSize:    pairCount,
	}, nil
}
