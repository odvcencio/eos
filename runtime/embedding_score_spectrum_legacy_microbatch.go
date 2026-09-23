package eosruntime

import "fmt"

// legacyScoreSpectrumActivationMicrobatch is the host-side activation bound
// for the legacy score-spectrum trainer. A score-spectrum optimizer batch is
// intentionally kept as one logical batch: the value only bounds the number
// of encoded sequences held at once. In particular, it must not be confused
// with a second optimizer batch size (which would change Adam moments and
// update semantics).
//
// The implementation makes one pooled-only forward pass over the complete
// query-first/candidate-second sequence order, computes the row-local loss
// from those pooled vectors, and then replays that exact sequence order in
// bounded activation chunks for backward. The forward weights are prepared
// once by the caller and no optimizer update occurs until every replay chunk
// has contributed to the same gradient buffers.
func (t *EmbeddingTrainer) runLegacyScoreSpectrumActivationMicrobatchUpdate(batch []EmbeddingScoreSpectrumExample, microbatchSize int) (EmbeddingTrainMetrics, error) {
	if t == nil {
		return EmbeddingTrainMetrics{}, fmt.Errorf("embedding trainer is not initialized")
	}
	if compactResidentTrainEnabled() {
		return EmbeddingTrainMetrics{}, fmt.Errorf(
			"unsupported architecture for compact resident score-spectrum training: architecture_version=%q; requires architecture_version=%q (%s=1)",
			embeddingTrainerArchitectureVersion(t),
			EmbeddingArchitectureCompactTransformerV1,
			compactResidentTrainEnv,
		)
	}
	if t.isCompactTrainer() {
		return EmbeddingTrainMetrics{}, fmt.Errorf("legacy score-spectrum activation microbatching is unavailable for compact trainers")
	}
	if len(batch) == 0 {
		return EmbeddingTrainMetrics{}, fmt.Errorf("score-spectrum training batch is empty")
	}
	if microbatchSize <= 0 {
		return EmbeddingTrainMetrics{}, fmt.Errorf("score-spectrum activation microbatch size must be positive")
	}

	canonicalBatch, err := canonicalizeTokenizedScoreSpectrumExamples(batch)
	if err != nil {
		return EmbeddingTrainMetrics{}, err
	}
	queryInputs, candidateInputs, candidateSpans, err := scoreSpectrumSequenceInputs(canonicalBatch, t.queryRoleIndex(), t.documentRoleIndex())
	if err != nil {
		return EmbeddingTrainMetrics{}, err
	}

	// Prepare the quantized/forward tensors exactly once. The optimizer is not
	// called until both the pooled loss pass and the replayed backward pass are
	// complete, so every chunk observes the same weights.
	forward := t.prepareForwardWeights()
	if forward == nil {
		return EmbeddingTrainMetrics{}, fmt.Errorf("missing forward weights")
	}
	t.primeForwardWeightResidency(forward.attnQ, forward.attnK, forward.attnV, forward.attnO, forward.hidden, forward.proj)

	allInputs := make([]embeddingSequenceInput, 0, len(queryInputs)+len(candidateInputs))
	allInputs = append(allInputs, queryInputs...)
	allInputs = append(allInputs, candidateInputs...)
	pooled, err := t.poolLegacyScoreSpectrumInputs(allInputs, forward, microbatchSize)
	if err != nil {
		return EmbeddingTrainMetrics{}, err
	}
	queries := pooled[:len(queryInputs)]
	candidates := pooled[len(queryInputs):]

	gradToken := make([]float32, len(t.tokenEmbed.F32))
	gradRole := make([]float32, tensorDataLen(t.roleEmbed))
	gradAttnQ := make([]float32, tensorDataLen(t.attentionQuery))
	gradAttnK := make([]float32, tensorDataLen(t.attentionKey))
	gradAttnV := make([]float32, tensorDataLen(t.attentionValue))
	gradAttnO := make([]float32, tensorDataLen(t.attentionOutput))
	gradHidden := make([]float32, len(t.hiddenProjectionData()))
	gradProj := make([]float32, len(t.projection.F32))
	queryGrads := make([][]float32, len(queries))
	candidateGrads := make([][]float32, len(candidates))
	for i := range queries {
		queryGrads[i] = make([]float32, len(queries[i].pooled))
	}
	for i := range candidates {
		candidateGrads[i] = make([]float32, len(candidates[i].pooled))
	}

	totalLoss, totalScore, pairCount, err := accumulateScoreSpectrumGrads(queries, candidates, candidateSpans, canonicalBatch, t.config, queryGrads, candidateGrads)
	if err != nil {
		return EmbeddingTrainMetrics{}, err
	}
	if pairCount == 0 {
		return EmbeddingTrainMetrics{}, fmt.Errorf("score-spectrum training batch has no usable candidates")
	}

	// Replay queries first and candidates second, matching the legacy trainer's
	// accumulation order. Each replay chunk is discarded after its backward
	// contribution, while the parameter gradient buffers remain for the single
	// optimizer transaction below.
	if err := t.replayLegacyScoreSpectrumBackward(queryInputs, queryGrads, forward, microbatchSize, "query", gradToken, gradRole, gradAttnQ, gradAttnK, gradAttnV, gradAttnO, gradHidden, gradProj); err != nil {
		return EmbeddingTrainMetrics{}, err
	}
	if err := t.replayLegacyScoreSpectrumBackward(candidateInputs, candidateGrads, forward, microbatchSize, "candidate", gradToken, gradRole, gradAttnQ, gradAttnK, gradAttnV, gradAttnO, gradHidden, gradProj); err != nil {
		return EmbeddingTrainMetrics{}, err
	}

	batchScale := float32(1) / float32(len(queries))
	// Keep the historical step ordering: the update receives the incremented
	// step, and this is the only optimizer update for the logical batch.
	t.step++
	if err := t.applyEmbeddingOptimizerUpdates(gradToken, gradRole, gradAttnQ, gradAttnK, gradAttnV, gradAttnO, gradHidden, gradProj, batchScale); err != nil {
		return EmbeddingTrainMetrics{}, err
	}
	return EmbeddingTrainMetrics{
		Loss:         totalLoss * batchScale,
		AverageScore: totalScore / float32(pairCount),
		BatchSize:    pairCount,
	}, nil
}

// poolLegacyScoreSpectrumInputs runs forward in bounded sequence chunks and
// returns pooled-only sequence views. No layer activation is retained after a
// chunk is released; tokens and role are retained only as lightweight input
// identity fields on the pooled view.
func (t *EmbeddingTrainer) poolLegacyScoreSpectrumInputs(inputs []embeddingSequenceInput, forward *embeddingForwardWeights, microbatchSize int) ([]*embeddingEncodedSequence, error) {
	if len(inputs) == 0 {
		return nil, fmt.Errorf("score-spectrum input sequence set is empty")
	}
	if forward == nil {
		return nil, fmt.Errorf("missing forward weights")
	}
	if microbatchSize <= 0 {
		microbatchSize = len(inputs)
	}
	pooled := make([]*embeddingEncodedSequence, len(inputs))
	for start := 0; start < len(inputs); start += microbatchSize {
		end := start + microbatchSize
		if end > len(inputs) {
			end = len(inputs)
		}
		encoded, err := t.encodeSequenceInputs(inputs[start:end], forward, false)
		if err != nil {
			t.discardLegacyScoreSpectrumEncodedSequences(encoded)
			return nil, fmt.Errorf("score-spectrum forward sequences %d-%d: %w", start, end-1, err)
		}
		if len(encoded) != end-start {
			t.discardLegacyScoreSpectrumEncodedSequences(encoded)
			return nil, fmt.Errorf("score-spectrum forward sequences %d-%d returned %d encodings, want %d", start, end-1, len(encoded), end-start)
		}
		for i, sequence := range encoded {
			if sequence == nil || len(sequence.pooled) == 0 {
				t.discardLegacyScoreSpectrumEncodedSequences(encoded)
				return nil, fmt.Errorf("score-spectrum forward sequence %d is empty", start+i)
			}
			// The pooled vector is the only activation that crosses chunk
			// boundaries. Input slices are lightweight references to the
			// canonical batch and role is copied for diagnostics/lifecycle
			// parity with a normal encoded sequence.
			pooled[start+i] = &embeddingEncodedSequence{
				pooled: append([]float32(nil), sequence.pooled...),
				tokens: inputs[start+i].tokens,
				role:   inputs[start+i].role,
			}
		}
		t.discardLegacyScoreSpectrumEncodedSequences(encoded)
	}
	return pooled, nil
}

// replayLegacyScoreSpectrumBackward re-encodes one side of the logical
// score-spectrum batch in bounded chunks and contributes to shared parameter
// gradients. The pooled gradients are indexed in the same order as inputs.
func (t *EmbeddingTrainer) replayLegacyScoreSpectrumBackward(inputs []embeddingSequenceInput, pooledGrads [][]float32, forward *embeddingForwardWeights, microbatchSize int, side string, gradToken, gradRole, gradAttnQ, gradAttnK, gradAttnV, gradAttnO, gradHidden, gradProj []float32) error {
	if len(inputs) == 0 {
		return nil
	}
	if len(inputs) != len(pooledGrads) {
		return fmt.Errorf("score-spectrum %s backward input/gradient count mismatch: inputs=%d gradients=%d", side, len(inputs), len(pooledGrads))
	}
	if forward == nil {
		return fmt.Errorf("missing forward weights")
	}
	if microbatchSize <= 0 {
		microbatchSize = len(inputs)
	}
	for start := 0; start < len(inputs); start += microbatchSize {
		end := start + microbatchSize
		if end > len(inputs) {
			end = len(inputs)
		}
		encoded, err := t.encodeSequenceInputs(inputs[start:end], forward, true)
		if err != nil {
			t.discardLegacyScoreSpectrumEncodedSequences(encoded)
			return fmt.Errorf("score-spectrum %s backward forward sequences %d-%d: %w", side, start, end-1, err)
		}
		if len(encoded) != end-start {
			t.discardLegacyScoreSpectrumEncodedSequences(encoded)
			return fmt.Errorf("score-spectrum %s backward sequences %d-%d returned %d encodings, want %d", side, start, end-1, len(encoded), end-start)
		}
		if err := t.backpropLegacyScoreSpectrumEncodedChunk(encoded, pooledGrads[start:end], forward, side, start, gradToken, gradRole, gradAttnQ, gradAttnK, gradAttnV, gradAttnO, gradHidden, gradProj); err != nil {
			t.discardLegacyScoreSpectrumEncodedSequences(encoded)
			return err
		}
		t.discardLegacyScoreSpectrumEncodedSequences(encoded)
	}
	return nil
}

func (t *EmbeddingTrainer) backpropLegacyScoreSpectrumEncodedChunk(encoded []*embeddingEncodedSequence, pooledGrads [][]float32, forward *embeddingForwardWeights, side string, offset int, gradToken, gradRole, gradAttnQ, gradAttnK, gradAttnV, gradAttnO, gradHidden, gradProj []float32) error {
	if len(encoded) != len(pooledGrads) {
		return fmt.Errorf("score-spectrum %s backward chunk at %d encoding/gradient count mismatch: encodings=%d gradients=%d", side, offset, len(encoded), len(pooledGrads))
	}
	// The legacy role-conditioned route intentionally declines the batched
	// backward helper so role gradients can be accumulated. For unconditioned
	// legacy models, retain the existing batched helper when available; the
	// chunk boundary only limits live activations and does not create an
	// optimizer update.
	if t.tryBackpropContrastiveBatch(
		encoded,
		nil,
		pooledGrads,
		nil,
		forward.attnQ,
		forward.attnK,
		forward.attnV,
		forward.attnO,
		forward.hidden,
		forward.proj,
		gradToken,
		gradAttnQ,
		gradAttnK,
		gradAttnV,
		gradAttnO,
		gradHidden,
		gradProj,
	) {
		if err := t.attentionResidentTrainStepErr(); err != nil {
			return fmt.Errorf("score-spectrum %s backward chunk at %d: %w", side, offset, err)
		}
		return nil
	}
	if err := t.attentionResidentTrainStepErr(); err != nil {
		return fmt.Errorf("score-spectrum %s backward chunk at %d: %w", side, offset, err)
	}
	for i, sequence := range encoded {
		if sequence == nil {
			return fmt.Errorf("score-spectrum %s backward sequence %d is nil", side, offset+i)
		}
		inputGrad := t.backpropEncodedSequence(sequence, pooledGrads[i], forward.attnQ, forward.attnK, forward.attnV, forward.attnO, forward.hidden, forward.proj, gradToken, gradAttnQ, gradAttnK, gradAttnV, gradAttnO, gradHidden, gradProj)
		if err := t.attentionResidentTrainStepErr(); err != nil {
			return fmt.Errorf("score-spectrum %s backward sequence %d: %w", side, offset+i, err)
		}
		t.accumulateInputTokenGrad(sequence.tokens, inputGrad, gradToken)
		t.accumulateInputRoleGrad(sequence.role, len(sequence.tokens), inputGrad, gradRole)
	}
	return nil
}

// discardLegacyScoreSpectrumEncodedSequences releases bindings and clears
// references to all full activation buffers immediately after a bounded
// forward/backward chunk. This is deliberately separate from
// releaseEncodedSequences, whose callers may still need the encoded objects'
// activations after unbinding.
func (t *EmbeddingTrainer) discardLegacyScoreSpectrumEncodedSequences(sequences []*embeddingEncodedSequence) {
	if t == nil {
		return
	}
	t.releaseEncodedSequences(sequences)
	for _, sequence := range sequences {
		if sequence == nil {
			continue
		}
		sequence.layers = nil
		sequence.pooled = nil
		sequence.tokens = nil
	}
}

// evaluateLegacyScoreSpectrumActivationMicrobatched computes score-spectrum
// eval metrics with the same pooled-only activation bound used by training.
// rowBatchSize retains the evaluator's existing row chunking/accounting while
// microbatchSize bounds the total query+candidate sequences in each forward
// replay.
func (t *EmbeddingTrainer) evaluateLegacyScoreSpectrumActivationMicrobatched(examples []EmbeddingScoreSpectrumExample, rowBatchSize, microbatchSize int, forward *embeddingForwardWeights) (EmbeddingScoreSpectrumEvalMetrics, error) {
	if len(examples) == 0 {
		return EmbeddingScoreSpectrumEvalMetrics{}, fmt.Errorf("score-spectrum eval dataset is empty")
	}
	if forward == nil {
		return EmbeddingScoreSpectrumEvalMetrics{}, fmt.Errorf("missing forward weights")
	}
	canonicalExamples, err := canonicalizeTokenizedScoreSpectrumExamples(examples)
	if err != nil {
		return EmbeddingScoreSpectrumEvalMetrics{}, err
	}
	examples = canonicalExamples
	if rowBatchSize <= 0 || rowBatchSize > len(examples) {
		rowBatchSize = len(examples)
	}
	if microbatchSize <= 0 {
		microbatchSize = rowBatchSize
	}
	var aggregate EmbeddingScoreSpectrumEvalMetrics
	for start := 0; start < len(examples); start += rowBatchSize {
		end := start + rowBatchSize
		if end > len(examples) {
			end = len(examples)
		}
		rows := examples[start:end]
		queryInputs, candidateInputs, candidateSpans, err := scoreSpectrumSequenceInputs(rows, t.queryRoleIndex(), t.documentRoleIndex())
		if err != nil {
			return EmbeddingScoreSpectrumEvalMetrics{}, fmt.Errorf("score-spectrum eval rows %d-%d: %w", start, end-1, err)
		}
		allInputs := make([]embeddingSequenceInput, 0, len(queryInputs)+len(candidateInputs))
		allInputs = append(allInputs, queryInputs...)
		allInputs = append(allInputs, candidateInputs...)
		pooled, err := t.poolLegacyScoreSpectrumInputs(allInputs, forward, microbatchSize)
		if err != nil {
			return EmbeddingScoreSpectrumEvalMetrics{}, fmt.Errorf("score-spectrum eval rows %d-%d: %w", start, end-1, err)
		}
		chunkMetrics, err := evaluateScoreSpectrumEncodings(pooled[:len(queryInputs)], pooled[len(queryInputs):], candidateSpans, rows, t.config.Temperature)
		// Pooled views contain no activations, but clear them promptly so a very
		// large eval set does not retain every vector while aggregate metrics are
		// being accumulated.
		for _, sequence := range pooled {
			if sequence != nil {
				sequence.pooled = nil
				sequence.tokens = nil
			}
		}
		if err != nil {
			return EmbeddingScoreSpectrumEvalMetrics{}, fmt.Errorf("score-spectrum eval rows %d-%d: %w", start, end-1, err)
		}
		mergeScoreSpectrumEvalMetrics(&aggregate, chunkMetrics)
	}
	normalizeScoreSpectrumEvalMetrics(&aggregate)
	return aggregate, nil
}
