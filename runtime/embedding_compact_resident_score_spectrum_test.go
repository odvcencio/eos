package eosruntime

import (
	"fmt"
	"strings"
	"testing"

	eosartifact "m31labs.dev/eos/artifact/eos"
	"m31labs.dev/eos/runtime/backend"
)

// scoreSpectrumResidentGradientToken carries a host-computed gradient only
// for the parity fake below. Production backends keep this payload in their
// own device-resident allocation behind the same opaque ABI.
type scoreSpectrumResidentGradientToken struct {
	name       string
	data       []float32
	generation uint64
	stepID     uint64
	alive      bool
}

func (*scoreSpectrumResidentGradientToken) ResidentGradientToken() {}

func (t *scoreSpectrumResidentGradientToken) Backend() eosartifact.BackendKind {
	return eosartifact.BackendCUDA
}

func (t *scoreSpectrumResidentGradientToken) Generation() uint64 {
	if t == nil {
		return 0
	}
	return t.generation
}

func (t *scoreSpectrumResidentGradientToken) StepID() uint64 {
	if t == nil {
		return 0
	}
	return t.stepID
}

func (t *scoreSpectrumResidentGradientToken) Alive() bool {
	return t != nil && t.alive
}

// scoreSpectrumHostBackedCompactTrainAccelerator emulates the resident ABI
// while computing pooled outputs and gradients through the existing host
// encoder. It is deliberately test-only: it lets the resident score-spectrum
// route prove loss/update parity without requiring CUDA.
type scoreSpectrumHostBackedCompactTrainAccelerator struct {
	fakeCompactTrainAccelerator
	host       *EmbeddingTrainer
	forward    *compactEmbeddingForwardWeights
	encoded    map[uint64][]*embeddingEncodedSequence
	gradients  map[string][]float32
	gradientID uint64
}

func (a *scoreSpectrumHostBackedCompactTrainAccelerator) BeginCompactTrainStep(stepID uint64, refs []backend.CompactForwardResidentRef) error {
	a.encoded = map[uint64][]*embeddingEncodedSequence{}
	a.gradients = map[string][]float32{}
	return a.fakeCompactTrainAccelerator.BeginCompactTrainStep(stepID, refs)
}

func (a *scoreSpectrumHostBackedCompactTrainAccelerator) RunCompactTrainForward(req backend.CompactTrainForwardRequest) (backend.CompactTrainForwardResult, error) {
	if a.host == nil || a.forward == nil {
		return backend.CompactTrainForwardResult{}, fmt.Errorf("host-backed compact test accelerator is missing encoder")
	}
	a.forwardCalls++
	a.forwardRequests = append(a.forwardRequests, req)
	a.nextGeneration++
	handleToken := &fakeCompactTrainHandleToken{alive: true, generation: a.nextGeneration, stepID: req.StepID}
	if a.live == nil {
		a.live = map[uint64]*fakeCompactTrainHandleToken{}
	}
	a.live[handleToken.generation] = handleToken
	a.stats.ForwardCalls++
	a.stats.HandlesCreated++
	a.stats.LiveHandles++
	pooled := make([]float32, req.Shape.Batch*req.Shape.OutputDim)
	encoded := make([]*embeddingEncodedSequence, req.Shape.Batch)
	for i := range req.Tokens {
		seq, err := a.host.encodeCompactSequence(req.Tokens[i], req.Masks[i], req.Roles[i], a.forward)
		if err != nil {
			return backend.CompactTrainForwardResult{}, err
		}
		encoded[i] = seq
		if len(seq.pooled) != req.Shape.OutputDim {
			return backend.CompactTrainForwardResult{}, fmt.Errorf("host-backed pooled width %d, want %d", len(seq.pooled), req.Shape.OutputDim)
		}
		copy(pooled[i*req.Shape.OutputDim:], seq.pooled)
	}
	a.encoded[handleToken.generation] = encoded
	a.stats.PooledDownloadedBytes += int64(len(pooled) * 4)
	return backend.CompactTrainForwardResult{
		Handle: backend.CompactTrainHandle{
			Backend:    eosartifact.BackendCUDA,
			Token:      handleToken,
			Shape:      req.Shape,
			Generation: handleToken.generation,
			StepID:     handleToken.stepID,
		},
		Pooled: backend.NewTensorF32([]int{req.Shape.Batch, req.Shape.OutputDim}, pooled),
	}, nil
}

func (a *scoreSpectrumHostBackedCompactTrainAccelerator) RunCompactTrainBackward(req backend.CompactTrainBackwardRequest) (backend.CompactTrainBackwardResult, error) {
	if a.host == nil || a.forward == nil {
		return backend.CompactTrainBackwardResult{}, fmt.Errorf("host-backed compact test accelerator is missing encoder")
	}
	token, ok := req.Handle.Token.(*fakeCompactTrainHandleToken)
	if !ok || token == nil || !token.alive {
		return backend.CompactTrainBackwardResult{}, fmt.Errorf("host-backed compact test backward requires live handle")
	}
	if req.GradPooled == nil || len(req.GradPooled.Shape) != 2 {
		return backend.CompactTrainBackwardResult{}, fmt.Errorf("host-backed compact test backward pooled gradient is malformed")
	}
	encoded := a.encoded[token.generation]
	if len(encoded) != req.GradPooled.Shape[0] {
		return backend.CompactTrainBackwardResult{}, fmt.Errorf("host-backed encoded rows %d, want %d", len(encoded), req.GradPooled.Shape[0])
	}
	grads := newCompactEmbeddingGradState(a.host.compactState)
	for i, seq := range encoded {
		start := i * req.GradPooled.Shape[1]
		end := start + req.GradPooled.Shape[1]
		if err := a.host.backpropCompactEncodedSequence(seq, req.GradPooled.F32[start:end], a.forward, grads); err != nil {
			return backend.CompactTrainBackwardResult{}, err
		}
	}
	items := compactTrainStateOptimizerItems(a.host.compactState)
	slices := compactEmbeddingGradSlices(grads)
	if len(items) != len(slices) {
		return backend.CompactTrainBackwardResult{}, fmt.Errorf("host-backed gradients %d, want %d", len(slices), len(items))
	}
	for i, item := range items {
		if item == nil {
			continue
		}
		if a.gradients[item.Name] == nil {
			a.gradients[item.Name] = make([]float32, len(slices[i]))
		}
		addFloat32Slice(a.gradients[item.Name], slices[i])
	}
	token.alive = false
	delete(a.live, token.generation)
	a.backwardCalls++
	a.stats.BackwardCalls++
	a.stats.HandlesReleased++
	a.stats.LiveHandles--
	refs := make([]backend.ResidentGradientRef, 0, len(items))
	for _, item := range items {
		if item == nil || item.Tensor == nil {
			continue
		}
		a.gradientID++
		gradToken := &scoreSpectrumResidentGradientToken{
			name:       item.Name,
			data:       append([]float32(nil), a.gradients[item.Name]...),
			generation: a.gradientID,
			stepID:     req.Handle.StepID,
			alive:      true,
		}
		refs = append(refs, backend.ResidentGradientRef{
			Name:       item.Name,
			Backend:    eosartifact.BackendCUDA,
			Token:      gradToken,
			Elements:   len(gradToken.data),
			Generation: gradToken.generation,
			StepID:     gradToken.stepID,
		})
	}
	return backend.CompactTrainBackwardResult{ResidentGradRefs: refs}, nil
}

type scoreSpectrumHostBackedResidentOptimizer struct {
	fakeResidentOptimizerAccelerator
}

func (o *scoreSpectrumHostBackedResidentOptimizer) ApplyUpdateWithResidentGrad(name string, cfg backend.OptimizerUpdateConfig, tensor, mom1, mom2 *backend.Tensor, grad backend.ResidentGradientRef) error {
	token, ok := grad.Token.(*scoreSpectrumResidentGradientToken)
	if !ok || token == nil || !token.alive || token.name != name {
		return fmt.Errorf("host-backed resident optimizer received invalid gradient %q", name)
	}
	o.residentApplyCalls++
	return o.fakeResidentOptimizerAccelerator.ApplyUpdate(name, cfg, tensor, mom1, mom2, backend.NewTensorF32(tensor.Shape, token.data))
}

func compactTrainStateTensorValues(state *CompactEmbeddingTrainState) [][]float32 {
	values := make([][]float32, 0)
	for _, snapshot := range snapshotCompactTrainStateTensors(state) {
		values = append(values, append([]float32(nil), snapshot.tensor.F32...))
	}
	return values
}

func TestCompactScoreSpectrumResidentUsesPooledBucketsAndAggregatesDuplicates(t *testing.T) {
	t.Setenv(compactResidentTrainEnv, "1")
	trainer := newCompactEmbeddingTrainerForTest(t, 3)
	trainer.optimizerAccel = &fakeResidentOptimizerAccelerator{}
	train := &fakeCompactTrainAccelerator{supportBackward: true}
	trainer.compactTrainAccel = train
	trainer.compactTrainBackend = eosartifact.BackendCUDA

	selected := 0
	batch := []EmbeddingScoreSpectrumExample{
		{
			QueryTokens:           []int32{1, 2},
			QueryMask:             []int32{1, 1},
			CandidateTokens:       [][]int32{{3, 4}, {0}},
			CandidateMasks:        [][]int32{{1, 1}, {1}},
			PositiveIndexes:       []int{0},
			SelectedPositiveIndex: &selected,
			HardNegativeEligible:  []bool{false, true},
			TargetProbabilities:   []float32{1, 0},
		},
		{
			// The query and first candidate intentionally duplicate row 0.
			QueryTokens:           []int32{1, 2},
			QueryMask:             []int32{1, 1},
			CandidateTokens:       [][]int32{{3, 4}, {2}},
			CandidateMasks:        [][]int32{{1, 1}, {1}},
			PositiveIndexes:       []int{0},
			SelectedPositiveIndex: &selected,
			HardNegativeEligible:  []bool{false, true},
			TargetProbabilities:   []float32{1, 0},
		},
	}

	metrics, err := trainer.TrainScoreSpectrumStep(batch)
	if err != nil {
		t.Fatalf("resident score-spectrum step: %v", err)
	}
	if !compactTestFinite(metrics.Loss) || !compactTestFinite(metrics.AverageScore) || metrics.BatchSize != 4 {
		t.Fatalf("resident score-spectrum metrics = %+v, want finite four-candidate metrics", metrics)
	}
	if trainer.step != 1 || !trainer.compactForwardSelected {
		t.Fatalf("resident step/selected = %d/%t, want 1/true", trainer.step, trainer.compactForwardSelected)
	}
	if train.beginCalls != 1 || train.endCalls != 1 || train.abortCalls != 0 {
		t.Fatalf("resident lifecycle begin/end/abort = %d/%d/%d, want 1/1/0", train.beginCalls, train.endCalls, train.abortCalls)
	}
	// Unique inputs are {query T2, candidate T2, candidate T1, candidate T1},
	// so the resident path should issue exactly two exact-length buckets. The
	// duplicate query/candidate references must not create extra handles.
	if train.forwardCalls != 2 || train.backwardCalls != 2 {
		t.Fatalf("resident forward/backward calls = %d/%d, want two buckets", train.forwardCalls, train.backwardCalls)
	}
	if len(train.forwardRequests) != 2 || len(train.backwardGrads) != 2 {
		t.Fatalf("resident request/gradient records = %d/%d, want two buckets", len(train.forwardRequests), len(train.backwardGrads))
	}
	if train.forwardRequests[0].Shape.Tokens != 1 || train.forwardRequests[0].Shape.Batch != 2 || train.forwardRequests[1].Shape.Tokens != 2 || train.forwardRequests[1].Shape.Batch != 2 {
		t.Fatalf("resident bucket shapes = %+v/%+v, want T1/B2 then T2/B2", train.forwardRequests[0].Shape, train.forwardRequests[1].Shape)
	}
	for i, grad := range train.backwardGrads {
		if grad == nil || len(grad.Shape) != 2 || grad.Shape[0] != 2 || grad.Shape[1] != 3 || len(grad.F32) != 6 {
			t.Fatalf("resident backward gradient %d = %+v, want [2 3] pooled gradient", i, grad)
		}
	}
}

func TestCompactScoreSpectrumResidentMatchesHostPackedLossAndUpdate(t *testing.T) {
	// Keep the reference copy explicitly on the packed route, then enable the
	// resident route only for the resident copy. The runtime intentionally no
	// longer falls back when the resident flag is set and its accelerator is
	// unavailable.
	t.Setenv(compactResidentTrainEnv, "0")
	host := newCompactEmbeddingTrainerForTest(t, 3)
	resident := newCompactEmbeddingTrainerForTest(t, 3)
	encoder := newCompactEmbeddingTrainerForTest(t, 3)
	host.config.Temperature = 0.05
	resident.config.Temperature = 0.05
	encoder.config.Temperature = 0.05
	// Force the reference trainer through the existing host packed path even
	// when a CUDA compact-train implementation is registered in the test
	// process. The probe below supplies the resident ABI for the second copy.
	host.compactTrainAccel = nil
	host.optimizerAccel = nil
	encoder.compactTrainAccel = nil
	probe := &scoreSpectrumHostBackedCompactTrainAccelerator{
		fakeCompactTrainAccelerator: fakeCompactTrainAccelerator{supportBackward: true},
		host:                        encoder,
		forward:                     encoder.prepareCompactForwardWeights(),
	}
	optimizer := &scoreSpectrumHostBackedResidentOptimizer{}
	resident.optimizerAccel = optimizer
	resident.compactTrainAccel = probe

	batch := tinyEmbeddingScoreSpectrumDataset()
	hostBefore := compactTrainStateTensorValues(host.compactState)
	residentBefore := compactTrainStateTensorValues(resident.compactState)
	hostMetrics, err := host.TrainScoreSpectrumStep(batch)
	if err != nil {
		t.Fatalf("host packed score-spectrum step: %v", err)
	}
	t.Setenv(compactResidentTrainEnv, "1")
	residentMetrics, err := resident.TrainScoreSpectrumStep(batch)
	if err != nil {
		t.Fatalf("resident score-spectrum step: %v", err)
	}
	if residentMetrics.BatchSize != hostMetrics.BatchSize {
		t.Fatalf("resident/host batch size = %d/%d, want parity", residentMetrics.BatchSize, hostMetrics.BatchSize)
	}
	assertClose(t, residentMetrics.Loss, hostMetrics.Loss, 1e-6)
	assertClose(t, residentMetrics.AverageScore, hostMetrics.AverageScore, 1e-6)
	if resident.step != 1 || host.step != 1 {
		t.Fatalf("resident/host steps = %d/%d, want one update each", resident.step, host.step)
	}
	if len(hostBefore) != len(residentBefore) {
		t.Fatalf("host/resident snapshot sizes = %d/%d, want parity", len(hostBefore), len(residentBefore))
	}
	residentAfter := compactTrainStateTensorValues(resident.compactState)
	hostAfter := compactTrainStateTensorValues(host.compactState)
	hostMoved := 0
	residentMoved := 0
	for i := range hostAfter {
		assertCloseF32Slice(t, fmt.Sprintf("resident tensor %d", i), residentAfter[i], hostAfter[i], 1e-5)
		if !float32SlicesClose(hostAfter[i], hostBefore[i], 0) {
			hostMoved++
		}
		if !float32SlicesClose(residentAfter[i], residentBefore[i], 0) {
			residentMoved++
		}
	}
	if hostMoved == 0 || residentMoved == 0 {
		t.Fatalf("host/resident updates moved no tensors: %d/%d", hostMoved, residentMoved)
	}
	if optimizer.residentApplyCalls == 0 {
		t.Fatal("resident optimizer did not receive resident gradients")
	}
}

func TestScoreSpectrumResidentRejectsLegacyBeforeHostAllocations(t *testing.T) {
	t.Setenv(compactResidentTrainEnv, "1")
	trainer := newTinyTrainable3DEmbeddingTrainer(t, 0.05)
	_, err := trainer.TrainScoreSpectrumStep(tinyEmbeddingScoreSpectrumDataset())
	if err == nil || !strings.Contains(err.Error(), "unsupported architecture") ||
		!strings.Contains(err.Error(), EmbeddingArchitectureLegacyV1) ||
		!strings.Contains(err.Error(), EmbeddingArchitectureCompactTransformerV1) {
		t.Fatalf("legacy resident error = %v, want unsupported architecture with legacy/compact versions", err)
	}
	if trainer.step != 0 {
		t.Fatalf("legacy resident rejection step = %d, want unchanged", trainer.step)
	}
}

func TestCompactScoreSpectrumResidentRejectsUnavailableAccelerator(t *testing.T) {
	t.Setenv(compactResidentTrainEnv, "1")
	trainer := newCompactEmbeddingTrainerForTest(t, 3)
	trainer.compactTrainAccel = nil
	_, err := trainer.TrainScoreSpectrumStep(tinyEmbeddingScoreSpectrumDataset())
	if err == nil || !strings.Contains(err.Error(), "compact train accelerator is unavailable") ||
		!strings.Contains(err.Error(), "packed fallback is disabled") {
		t.Fatalf("unavailable resident accelerator error = %v, want fail-closed packed fallback error", err)
	}
	if trainer.step != 0 {
		t.Fatalf("unavailable resident rejection step = %d, want unchanged", trainer.step)
	}
}

func TestCompactScoreSpectrumPackedRouteReportsTrainStartDiagnostics(t *testing.T) {
	t.Setenv(compactResidentTrainEnv, "0")
	t.Setenv(compactPackedForwardEnv, "0")
	trainer := newCompactEmbeddingTrainerForTest(t, 3)
	t.Cleanup(func() { trainer.Close() })
	source, err := NewEmbeddingScoreSpectrumSliceSource(tinyEmbeddingScoreSpectrumDataset())
	if err != nil {
		t.Fatalf("new score-spectrum source: %v", err)
	}
	defer source.Close()
	var reports []EmbeddingTrainProgress
	_, err = trainer.FitScoreSpectrumSource(source, nil, EmbeddingTrainRunConfig{
		Epochs:                          1,
		BatchSize:                       2,
		ScoreSpectrumMaxBatchCandidates: 2,
		Shuffle:                         false,
		ProgressEverySteps:              99,
		Progress:                        func(progress EmbeddingTrainProgress) { reports = append(reports, progress) },
	})
	if err != nil {
		t.Fatalf("compact packed score-spectrum fit: %v", err)
	}
	if len(reports) == 0 {
		t.Fatal("compact packed score-spectrum fit emitted no progress")
	}
	start := reports[0]
	if start.Phase != "train_start" || start.Architecture != EmbeddingArchitectureCompactTransformerV1 ||
		start.Route != "compact_packed" || start.Branch != "compact_packed" || start.CompactBackend != "host" ||
		start.ResidentRequested || start.CandidateCap != 2 || start.FirstSpanRows != 1 || start.FirstSpanCandidates != 2 {
		t.Fatalf("compact packed train_start = %+v, want route/cap/first-span diagnostics", start)
	}
}

func TestCompactScoreSpectrumResidentRequiresResidentGradientOptimizer(t *testing.T) {
	t.Setenv(compactResidentTrainEnv, "1")
	trainer := newCompactEmbeddingTrainerForTest(t, 3)
	trainer.optimizerAccel = nil
	trainer.compactTrainAccel = &fakeCompactTrainAccelerator{supportBackward: true}
	_, err := trainer.TrainScoreSpectrumStep(tinyEmbeddingScoreSpectrumDataset())
	if err == nil || !strings.Contains(err.Error(), "resident-gradient optimizer") {
		t.Fatalf("resident optimizer error = %v, want resident-gradient optimizer failure", err)
	}
	if trainer.step != 0 {
		t.Fatalf("resident optimizer failure step = %d, want unchanged", trainer.step)
	}
}

func TestCompactScoreSpectrumPackedGuardFailsBeforeActivationPath(t *testing.T) {
	trainer := newCompactEmbeddingTrainerForTest(t, 3)
	batch := []EmbeddingScoreSpectrumExample{{
		QueryTokens:     []int32{1},
		CandidateTokens: make([][]int32, compactPackedScoreSpectrumMaxCandidates+1),
	}}
	for i := range batch[0].CandidateTokens {
		batch[0].CandidateTokens[i] = []int32{int32(i % 4)}
	}
	if err := trainer.guardCompactScoreSpectrumPackedBatch(batch); err == nil || !strings.Contains(err.Error(), "before allocating activations") {
		t.Fatalf("packed guard error = %v, want preallocation guard failure", err)
	}

	// A safe small batch remains admissible, proving the guard is bounded rather
	// than a blanket prohibition on the host fallback.
	if err := trainer.guardCompactScoreSpectrumPackedBatch(tinyEmbeddingScoreSpectrumDataset()); err != nil {
		t.Fatalf("small packed score-spectrum batch rejected: %v", err)
	}
}
