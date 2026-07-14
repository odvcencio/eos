package eosruntime

import (
	"fmt"
	"math"
	"math/rand"
	"strings"
	"testing"

	eosartifact "m31labs.dev/eos/artifact/eos"
	"m31labs.dev/eos/runtime/backend"
)

// TestVectorDistillRoleIndexResolvesExplicitAndFallbackRoles verifies the
// per-example role resolution helper: explicit roles map to their own role
// index (never a hardcoded one), empty roles fall back to defaultRole and
// report that the fallback fired, and unknown roles error.
func TestVectorDistillRoleIndexResolvesExplicitAndFallbackRoles(t *testing.T) {
	trainer := newCompactEmbeddingTrainerForTest(t, 3)
	cases := []struct {
		role, defaultRole string
		want              int32
		wantFallback      bool
	}{
		{EmbeddingRoleQuery, EmbeddingRoleQuery, trainer.queryRoleIndex(), false},
		{EmbeddingRoleDocument, EmbeddingRoleQuery, trainer.documentRoleIndex(), false},
		{EmbeddingRoleRaw, EmbeddingRoleQuery, trainer.rawRoleIndex(), false},
		{"", EmbeddingRoleDocument, trainer.documentRoleIndex(), true},
		{"", EmbeddingRoleQuery, trainer.queryRoleIndex(), true},
	}
	for _, tc := range cases {
		got, fallback, err := trainer.vectorDistillRoleIndex(tc.role, tc.defaultRole)
		if err != nil {
			t.Fatalf("role=%q default=%q: %v", tc.role, tc.defaultRole, err)
		}
		if got != tc.want || fallback != tc.wantFallback {
			t.Errorf("role=%q default=%q got=(%d,%v) want=(%d,%v)", tc.role, tc.defaultRole, got, fallback, tc.want, tc.wantFallback)
		}
	}
	if _, _, err := trainer.vectorDistillRoleIndex("passage", EmbeddingRoleQuery); err == nil {
		t.Fatal("expected error for unknown role")
	}
}

func vectorDistillTestExample(id, role string, tokens []int32, teacher []float32) EmbeddingTokenizedVectorDistillExample {
	mask := make([]int32, len(tokens))
	for i := range mask {
		mask[i] = 1
	}
	return EmbeddingTokenizedVectorDistillExample{
		ID:            id,
		Tokens:        tokens,
		Mask:          mask,
		TeacherVector: append([]float32(nil), teacher...),
		Role:          role,
	}
}

// TestTrainVectorDistillBatchUsesPerExampleRoleNotHardcodedQuery is the core
// regression test for the query-role hardcoding bug: identical tokens trained
// under an explicit "document" role must produce a different loss than the
// same tokens trained under "query" (the additive role embedding changes the
// pooled encoding), and a legacy row with no role falling back to a default
// of "document" must behave identically to an explicit "document" row.
func TestTrainVectorDistillBatchUsesPerExampleRoleNotHardcodedQuery(t *testing.T) {
	teacher := []float32{0.1, -0.2, 0.05, 0.3}
	tokens := []int32{1, 2, 3}

	runBatch := func(role, defaultRole string) EmbeddingTrainMetrics {
		t.Helper()
		trainer := newCompactEmbeddingTrainerForTest(t, 3)
		batch := []EmbeddingTokenizedVectorDistillExample{vectorDistillTestExample("ex1", role, tokens, teacher)}
		metrics, _, err := trainer.trainVectorDistillBatch(batch, nil, len(teacher), defaultRole, 0)
		if err != nil {
			t.Fatalf("trainVectorDistillBatch role=%q default=%q: %v", role, defaultRole, err)
		}
		return metrics
	}

	documentMetrics := runBatch(EmbeddingRoleDocument, EmbeddingRoleQuery)
	queryMetrics := runBatch(EmbeddingRoleQuery, EmbeddingRoleQuery)
	if documentMetrics.Loss == queryMetrics.Loss {
		t.Fatalf("document-role and query-role losses are identical (%v); role is not affecting training (regression of the hardcoded query-role bug)", documentMetrics.Loss)
	}

	fallbackToDocument := runBatch("", EmbeddingRoleDocument)
	if fallbackToDocument.Loss != documentMetrics.Loss {
		t.Fatalf("fallback-to-document loss %v != explicit document-role loss %v, want equal", fallbackToDocument.Loss, documentMetrics.Loss)
	}
}

// TestTrainVectorDistillBatchRoleFallbackWarningFiresOnce verifies the
// one-time warning flag is set the first time a row without an explicit role
// falls back to the default, and TestTrainVectorDistillBatchNoWarningForExplicitRole
// verifies explicit-role rows never trigger it.
func TestTrainVectorDistillBatchRoleFallbackWarningFiresOnce(t *testing.T) {
	trainer := newCompactEmbeddingTrainerForTest(t, 3)
	if trainer.vectorDistillDefaultRoleWarned {
		t.Fatal("fresh trainer must not have warned yet")
	}
	batch := []EmbeddingTokenizedVectorDistillExample{
		vectorDistillTestExample("a", "", []int32{1, 2, 3}, []float32{0.1, -0.1, 0.2}),
	}
	if _, _, err := trainer.trainVectorDistillBatch(batch, nil, 3, EmbeddingRoleQuery, 0); err != nil {
		t.Fatalf("trainVectorDistillBatch: %v", err)
	}
	if !trainer.vectorDistillDefaultRoleWarned {
		t.Fatal("expected vectorDistillDefaultRoleWarned=true after a fallback row")
	}
}

func TestTrainVectorDistillBatchNoWarningForExplicitRole(t *testing.T) {
	trainer := newCompactEmbeddingTrainerForTest(t, 3)
	batch := []EmbeddingTokenizedVectorDistillExample{
		vectorDistillTestExample("a", EmbeddingRoleDocument, []int32{1, 2, 3}, []float32{0.1, -0.1, 0.2}),
	}
	if _, _, err := trainer.trainVectorDistillBatch(batch, nil, 3, EmbeddingRoleQuery, 0); err != nil {
		t.Fatalf("trainVectorDistillBatch: %v", err)
	}
	if trainer.vectorDistillDefaultRoleWarned {
		t.Fatal("explicit-role row must not trigger the fallback warning")
	}
}

func TestTrainVectorDistillBatchRejectsUnknownRole(t *testing.T) {
	trainer := newCompactEmbeddingTrainerForTest(t, 3)
	batch := []EmbeddingTokenizedVectorDistillExample{
		vectorDistillTestExample("a", "passage", []int32{1, 2, 3}, []float32{0.1, -0.1, 0.2}),
	}
	if _, _, err := trainer.trainVectorDistillBatch(batch, nil, 3, EmbeddingRoleQuery, 0); err == nil {
		t.Fatal("expected error for unsupported role")
	}
}

// TestTrainVectorDistillBatchRelationalWeightMovesEncoder is the "tiny model"
// integration check for the opt-in relational term: with
// --vector-distill-relational-weight active on a 2-example batch, the loss is
// finite and the encoder's trainable tensors move (confirming gradients flow
// from the relational term into the encoder, not just the ephemeral
// projection).
func TestTrainVectorDistillBatchRelationalWeightMovesEncoder(t *testing.T) {
	trainer := newCompactEmbeddingTrainerForTest(t, 3)
	batch := []EmbeddingTokenizedVectorDistillExample{
		vectorDistillTestExample("a", EmbeddingRoleQuery, []int32{1, 2, 3}, []float32{0.2, -0.1, 0.05, 0.3}),
		vectorDistillTestExample("b", EmbeddingRoleDocument, []int32{3, 2, 1}, []float32{-0.1, 0.4, 0.2, -0.05}),
	}
	snapshots := snapshotCompactTrainStateTensors(trainer.compactState)
	metrics, proj, err := trainer.trainVectorDistillBatch(batch, nil, 4, EmbeddingRoleQuery, 0.5)
	if err != nil {
		t.Fatalf("trainVectorDistillBatch: %v", err)
	}
	if !compactTestFinite(metrics.Loss) {
		t.Fatalf("loss is not finite: %v", metrics.Loss)
	}
	if proj == nil {
		t.Fatal("expected projection state to be initialized")
	}
	delta := aggregateParameterDeltaStats(snapshots)
	if delta.NonzeroCount == 0 || delta.L2Norm == 0 {
		t.Fatalf("relational-weight training did not move encoder tensors: %+v", delta)
	}
}

// TestTrainVectorDistillBatchRelationalWeightInactiveForSingleExampleBatch
// verifies the relational term is a graceful no-op for a batch of size 1: the
// loss must be identical whether the relational weight is on or off, and no
// error is raised.
func TestTrainVectorDistillBatchRelationalWeightInactiveForSingleExampleBatch(t *testing.T) {
	single := []EmbeddingTokenizedVectorDistillExample{
		vectorDistillTestExample("a", EmbeddingRoleQuery, []int32{1, 2, 3}, []float32{0.2, -0.1, 0.05, 0.3}),
	}

	trainerOff := newCompactEmbeddingTrainerForTest(t, 3)
	metricsOff, _, err := trainerOff.trainVectorDistillBatch(single, nil, 4, EmbeddingRoleQuery, 0)
	if err != nil {
		t.Fatalf("trainVectorDistillBatch (weight=0): %v", err)
	}

	trainerOn := newCompactEmbeddingTrainerForTest(t, 3)
	metricsOn, _, err := trainerOn.trainVectorDistillBatch(single, nil, 4, EmbeddingRoleQuery, 0.5)
	if err != nil {
		t.Fatalf("trainVectorDistillBatch (weight=0.5, batch size 1): %v", err)
	}

	if metricsOff.Loss != metricsOn.Loss {
		t.Fatalf("relational term active for a batch size of 1: loss(weight=0)=%v loss(weight=0.5)=%v, want equal (inactive)", metricsOff.Loss, metricsOn.Loss)
	}
}

// TestVectorDistillRelationalGradientMergeIsBatchSizeInvariant is the
// regression test for the relational-gradient batch-size-scaling bug fixed
// in computeVectorDistillBatchGradients (see the "Pre-multiply by
// len(batch)" comment there): VectorDistillRelationalLossAndGrad returns the
// complete analytic gradient of the whole-batch relational loss, which the
// caller's batchScale (1/len(batch)) would otherwise shrink by an extra,
// spurious factor of 1/N on top of its intended pointwise sum->mean
// conversion. This test duplicates a fixed two-example cluster into batches
// of size 2 and 4 and checks that the relational term's EFFECTIVE
// contribution to the raw (pre-AdamW) encoder gradient — isolated by
// differencing a real merge call against one with the relational term
// disabled, then applying the same batchScale the trainer applies — does not
// shrink anywhere near the naive 1/N-per-doubling factor the bug produced.
//
// The assertion threshold of 0.5 is derived analytically: for this exact
// "two clusters duplicated to double the batch" construction, the correct
// (post-fix) ratio between the N=4 and N=2 contributions is 2/3 (dilution is
// expected and correct, since the relational term is a mean over a larger set
// of same-cluster pairs whose disagreement is zero), while the pre-fix bug's
// ratio for the same construction is (2/3)/2 = 1/3 (an extra halving from
// the spurious 1/N batchScale applied twice). 0.5 sits cleanly between them.
func TestVectorDistillRelationalGradientMergeIsBatchSizeInvariant(t *testing.T) {
	teacherA := []float32{0.2, -0.1, 0.05, 0.3}
	teacherB := []float32{-0.15, 0.35, 0.1, -0.2}
	exampleA := vectorDistillTestExample("a", EmbeddingRoleQuery, []int32{1, 2, 3}, teacherA)
	exampleB := vectorDistillTestExample("b", EmbeddingRoleDocument, []int32{3, 2, 1}, teacherB)

	batch2 := []EmbeddingTokenizedVectorDistillExample{exampleA, exampleB}
	batch4 := []EmbeddingTokenizedVectorDistillExample{exampleA, exampleB, exampleA, exampleB}

	const relationalWeight = float32(0.5)
	const teacherDim = 4

	norm2 := relationalOnlyEncoderGradNorm(t, batch2, relationalWeight, teacherDim)
	norm4 := relationalOnlyEncoderGradNorm(t, batch4, relationalWeight, teacherDim)

	if norm2 == 0 || norm4 == 0 {
		t.Fatalf("expected nonzero relational-only encoder gradient contribution: norm(N=2)=%v norm(N=4)=%v", norm2, norm4)
	}

	ratio := norm4 / norm2
	if ratio < 0.5 {
		t.Fatalf("relational-only encoder gradient contribution shrank as batch size doubled via duplicated examples (regression of batch-size-dependent relational weight): norm(N=2)=%v norm(N=4)=%v ratio=%v, want >= 0.5", norm2, norm4, ratio)
	}
}

func TestComputeVectorDistillBatchGradientsScratchMatchesUnscratchedExact(t *testing.T) {
	trainer := newCompactEmbeddingTrainerForTest(t, 3)
	batch := []EmbeddingTokenizedVectorDistillExample{
		vectorDistillTestExample("a", EmbeddingRoleQuery, []int32{1, 2, 3}, []float32{0.2, -0.1, 0.05, 0.3}),
		vectorDistillTestExample("b", EmbeddingRoleDocument, []int32{3, 2, 1}, []float32{-0.1, 0.4, 0.2, -0.05}),
	}

	inputs := make([]embeddingSequenceInput, len(batch))
	for i, ex := range batch {
		roleIndex, _, err := trainer.vectorDistillRoleIndex(ex.Role, EmbeddingRoleQuery)
		if err != nil {
			t.Fatalf("vectorDistillRoleIndex: %v", err)
		}
		inputs[i] = embeddingSequenceInput{tokens: ex.Tokens, mask: ex.Mask, role: roleIndex, label: fmt.Sprintf("batch %d", i)}
	}
	forward := trainer.prepareForwardWeights()
	if forward == nil || forward.compact == nil {
		t.Fatal("missing compact forward weights")
	}
	encoded, err := trainer.encodeSequenceInputs(inputs, forward, true)
	if err != nil {
		t.Fatalf("encodeSequenceInputs: %v", err)
	}
	defer trainer.releaseEncodedSequences(encoded)

	students := make([][]float32, len(encoded))
	teachers := make([][]float32, len(encoded))
	for i, enc := range encoded {
		students[i] = enc.pooled
		teachers[i] = batch[i].TeacherVector
	}
	_, relationalGrads, err := VectorDistillRelationalLossAndGrad(students, teachers, 0.5)
	if err != nil {
		t.Fatalf("VectorDistillRelationalLossAndGrad: %v", err)
	}

	proj := newVectorDistillProjectionState(len(encoded[0].pooled), 4, rand.New(rand.NewSource(42)))
	wantGrads, wantGradW, wantLoss, err := trainer.computeVectorDistillBatchGradients(batch, encoded, proj, forward, relationalGrads)
	if err != nil {
		t.Fatalf("computeVectorDistillBatchGradients: %v", err)
	}
	var scratch vectorDistillBatchScratch
	gotGrads, gotGradW, gotLoss, err := trainer.computeVectorDistillBatchGradientsWithScratch(batch, encoded, proj, forward, relationalGrads, &scratch)
	if err != nil {
		t.Fatalf("computeVectorDistillBatchGradientsWithScratch: %v", err)
	}
	if gotLoss != wantLoss {
		t.Fatalf("totalLoss = %v, want exact %v", gotLoss, wantLoss)
	}
	assertExactFloat32Slice(t, "gradW", gotGradW, wantGradW)

	gotSlices := compactEmbeddingGradSlices(gotGrads)
	wantSlices := compactEmbeddingGradSlices(wantGrads)
	if len(gotSlices) != len(wantSlices) {
		t.Fatalf("gradient slice count = %d, want %d", len(gotSlices), len(wantSlices))
	}
	for i := range gotSlices {
		assertExactFloat32Slice(t, fmt.Sprintf("grad slice %d", i), gotSlices[i], wantSlices[i])
	}
}

func TestVectorDistillHostFallbackRecordsPhaseTimers(t *testing.T) {
	trainer := newCompactEmbeddingTrainerForTest(t, 3)
	trainer.forwardMatMul = nil
	trainer.optimizerAccel = nil
	batch := []EmbeddingTokenizedVectorDistillExample{
		vectorDistillTestExample("a", EmbeddingRoleQuery, []int32{1, 2, 3}, []float32{0.2, -0.1, 0.05, 0.3}),
		vectorDistillTestExample("b", EmbeddingRoleDocument, []int32{3, 2, 1}, []float32{-0.1, 0.4, 0.2, -0.05}),
	}

	start := trainer.TrainProfile()
	metrics, proj, err := trainer.trainVectorDistillBatch(batch, nil, 4, EmbeddingRoleQuery, 0)
	if err != nil {
		t.Fatalf("trainVectorDistillBatch host fallback: %v", err)
	}
	if proj == nil {
		t.Fatal("expected projection state")
	}
	if !compactTestFinite(metrics.Loss) {
		t.Fatalf("loss is not finite: %v", metrics.Loss)
	}
	delta := diffTrainProfile(start, trainer.TrainProfile())
	if delta.VectorDistillPhases.EncodeNanos <= 0 {
		t.Fatalf("encode timer = %d, want > 0", delta.VectorDistillPhases.EncodeNanos)
	}
	if delta.VectorDistillPhases.ProjectionLossNanos <= 0 {
		t.Fatalf("projection/loss timer = %d, want > 0", delta.VectorDistillPhases.ProjectionLossNanos)
	}
	if delta.VectorDistillPhases.BackwardNanos <= 0 {
		t.Fatalf("backward timer = %d, want > 0", delta.VectorDistillPhases.BackwardNanos)
	}
	if delta.VectorDistillPhases.OptimizerNanos <= 0 {
		t.Fatalf("optimizer timer = %d, want > 0", delta.VectorDistillPhases.OptimizerNanos)
	}
}

func TestFitVectorDistillOptimizerSyncModeControlsDeferredUpdates(t *testing.T) {
	trainSet := []EmbeddingTokenizedVectorDistillExample{
		vectorDistillTestExample("a", EmbeddingRoleQuery, []int32{1, 2, 3}, []float32{0.2, -0.1, 0.05, 0.3}),
		vectorDistillTestExample("b", EmbeddingRoleDocument, []int32{3, 2, 1}, []float32{-0.1, 0.4, 0.2, -0.05}),
	}

	run := func(mode string) (EmbeddingTrainRunSummary, *fakeResidentOptimizerAccelerator, error) {
		t.Helper()
		trainer := newCompactEmbeddingTrainerForTest(t, 3)
		opt := &fakeResidentOptimizerAccelerator{}
		trainer.optimizerAccel = opt
		trainer.forwardMatMul = &residentAwareCountingMatMulAccelerator{}
		summary, err := trainer.FitVectorDistill(trainSet, EmbeddingTrainRunConfig{
			Epochs:                     1,
			BatchSize:                  2,
			Shuffle:                    false,
			VectorDistillOptimizerSync: mode,
		})
		if trainer.deferOptimizerSync {
			t.Fatalf("deferOptimizerSync was not restored after mode %q", mode)
		}
		return summary, opt, err
	}

	defaultSummary, defaultOpt, err := run("")
	if err != nil {
		t.Fatalf("default vector-distill fit: %v", err)
	}
	if defaultSummary.Config.VectorDistillOptimizerSync != VectorDistillOptimizerSyncDeferred {
		t.Fatalf("default sync mode = %q, want %q", defaultSummary.Config.VectorDistillOptimizerSync, VectorDistillOptimizerSyncDeferred)
	}
	if defaultOpt.stats.DeferredSyncUpdates == 0 {
		t.Fatalf("default deferred sync updates = %d, want > 0", defaultOpt.stats.DeferredSyncUpdates)
	}
	if defaultSummary.DeltaProfile.Optimizer.DeferredSyncUpdates == 0 {
		t.Fatalf("default profile deferred sync updates = %d, want > 0", defaultSummary.DeltaProfile.Optimizer.DeferredSyncUpdates)
	}

	immediateSummary, immediateOpt, err := run(VectorDistillOptimizerSyncImmediate)
	if err != nil {
		t.Fatalf("immediate vector-distill fit: %v", err)
	}
	if immediateSummary.Config.VectorDistillOptimizerSync != VectorDistillOptimizerSyncImmediate {
		t.Fatalf("immediate sync mode = %q, want %q", immediateSummary.Config.VectorDistillOptimizerSync, VectorDistillOptimizerSyncImmediate)
	}
	if immediateOpt.stats.DeferredSyncUpdates != 0 {
		t.Fatalf("immediate deferred sync updates = %d, want 0", immediateOpt.stats.DeferredSyncUpdates)
	}
	if immediateSummary.DeltaProfile.Optimizer.DeferredSyncUpdates != 0 {
		t.Fatalf("immediate profile deferred sync updates = %d, want 0", immediateSummary.DeltaProfile.Optimizer.DeferredSyncUpdates)
	}
}

func TestFitVectorDistillOptimizerSyncModeRejectsInvalidMode(t *testing.T) {
	trainer := newCompactEmbeddingTrainerForTest(t, 3)
	opt := &fakeResidentOptimizerAccelerator{}
	trainer.optimizerAccel = opt
	trainer.forwardMatMul = &residentAwareCountingMatMulAccelerator{}
	_, err := trainer.FitVectorDistill([]EmbeddingTokenizedVectorDistillExample{
		vectorDistillTestExample("a", EmbeddingRoleQuery, []int32{1, 2, 3}, []float32{0.2, -0.1, 0.05, 0.3}),
	}, EmbeddingTrainRunConfig{
		Epochs:                     1,
		BatchSize:                  1,
		VectorDistillOptimizerSync: "sometimes",
	})
	if err == nil || !strings.Contains(err.Error(), "unsupported vector_distill_optimizer_sync") {
		t.Fatalf("invalid sync mode error = %v, want unsupported mode", err)
	}
	if opt.stats.UpdateCalls != 0 {
		t.Fatalf("optimizer updates after invalid mode = %d, want 0", opt.stats.UpdateCalls)
	}
}

func TestVectorDistillProjectionRoutesMatMulsAndOptimizerThroughAccelerators(t *testing.T) {
	trainer := newCompactEmbeddingTrainerForTest(t, 3)
	matmulAccel := &fakeVectorDistillMatMulAccelerator{}
	optimizerAccel := &fakeVectorDistillOptimizerAccelerator{}
	trainer.forwardMatMul = matmulAccel
	trainer.optimizerAccel = optimizerAccel

	proj := &vectorDistillProjectionState{
		W:        []float32{0.2, -0.1, 0.05, 0.3, -0.4, 0.25},
		Mom1:     []float32{0.01, -0.02, 0.03, -0.01, 0.02, -0.03},
		Mom2:     []float32{0.001, 0.002, 0.003, 0.004, 0.005, 0.006},
		InputDim: 2,
		OutDim:   3,
		Step:     2,
	}
	student := []float32{0.7, -0.2}
	gradProj := []float32{0.4, -0.3, 0.1}

	gotProj := make([]float32, proj.OutDim)
	trainer.projectVectorDistillStudent(student, proj, gotProj)
	wantProj := make([]float32, proj.OutDim)
	fillHostMatMul(student, 1, proj.InputDim, proj.W, proj.OutDim, wantProj)
	assertTensorClose(t, backend.NewTensorF32([]int{1, proj.OutDim}, gotProj), []int{1, proj.OutDim}, wantProj)
	if matmulAccel.stats.RunCalls != 1 {
		t.Fatalf("matmul run calls after projection forward = %d, want 1", matmulAccel.stats.RunCalls)
	}

	gotGradStudent := make([]float32, proj.InputDim)
	gotGradW := make([]float32, len(proj.W))
	trainer.accumulateVectorDistillProjectionGrads(student, gradProj, proj, gotGradStudent, gotGradW)
	wantGradStudent := make([]float32, proj.InputDim)
	wantGradW := make([]float32, len(proj.W))
	accumulateVectorDistillProjectionGrads(student, gradProj, proj.W, wantGradStudent, wantGradW)
	assertTensorClose(t, backend.NewTensorF32([]int{1, proj.InputDim}, gotGradStudent), []int{1, proj.InputDim}, wantGradStudent)
	assertTensorClose(t, backend.NewTensorF32([]int{proj.InputDim, proj.OutDim}, gotGradW), []int{proj.InputDim, proj.OutDim}, wantGradW)
	if matmulAccel.stats.RunCalls != 3 {
		t.Fatalf("matmul run calls after projection backward = %d, want 3", matmulAccel.stats.RunCalls)
	}

	hostProj := &vectorDistillProjectionState{
		W:        append([]float32(nil), proj.W...),
		Mom1:     append([]float32(nil), proj.Mom1...),
		Mom2:     append([]float32(nil), proj.Mom2...),
		InputDim: proj.InputDim,
		OutDim:   proj.OutDim,
		Step:     proj.Step,
	}
	if err := trainer.applyVectorDistillProjectionAdamW(proj, gotGradW); err != nil {
		t.Fatalf("apply projection optimizer update: %v", err)
	}
	applyVectorDistillProjectionAdamW(hostProj.W, hostProj.Mom1, hostProj.Mom2, gotGradW, trainer.config.LearningRate, trainer.config.Beta1, trainer.config.Beta2, trainer.config.Epsilon, trainer.config.WeightDecay, hostProj.Step)
	assertTensorClose(t, backend.NewTensorF32([]int{proj.InputDim, proj.OutDim}, proj.W), []int{proj.InputDim, proj.OutDim}, hostProj.W)
	assertTensorClose(t, backend.NewTensorF32([]int{proj.InputDim, proj.OutDim}, proj.Mom1), []int{proj.InputDim, proj.OutDim}, hostProj.Mom1)
	assertTensorClose(t, backend.NewTensorF32([]int{proj.InputDim, proj.OutDim}, proj.Mom2), []int{proj.InputDim, proj.OutDim}, hostProj.Mom2)
	if optimizerAccel.stats.UpdateCalls != 1 {
		t.Fatalf("optimizer update calls = %d, want 1", optimizerAccel.stats.UpdateCalls)
	}
	if optimizerAccel.stats.SyncCalls != 1 {
		t.Fatalf("optimizer sync calls = %d, want 1", optimizerAccel.stats.SyncCalls)
	}
}

func TestVectorDistillProjectionResidentUpdateErrorFailsClosed(t *testing.T) {
	trainer := newCompactEmbeddingTrainerForTest(t, 3)
	proj := &vectorDistillProjectionState{
		W:        []float32{0.2, -0.1, 0.05, 0.3, -0.4, 0.25},
		Mom1:     []float32{0, 0, 0, 0, 0, 0},
		Mom2:     []float32{0, 0, 0, 0, 0, 0},
		InputDim: 2,
		OutDim:   3,
		Step:     1,
	}
	name := proj.residentName()
	before := append([]float32(nil), proj.W...)
	trainer.optimizerAccel = &fakeResidentOptimizerAccelerator{
		applyErr: fmt.Errorf("forced projection update failure"),
		resident: map[string]*fakeResidentOptimizerToken{
			name: {tensor: backend.NewTensorF32([]int{2, 3}, append([]float32(nil), proj.W...)), generation: 1, alive: true},
		},
	}
	trainer.forwardMatMul = &residentAwareCountingMatMulAccelerator{}
	trainer.deferOptimizerSync = true
	err := trainer.applyVectorDistillProjectionAdamW(proj, []float32{0.1, 0.2, 0.3, 0.4, 0.5, 0.6})
	if err == nil || !strings.Contains(err.Error(), "forced projection update failure") {
		t.Fatalf("projection update error = %v, want forced resident failure", err)
	}
	assertCloseF32Slice(t, "projection after failed resident update", proj.W, before, 0)
}

func TestVectorDistillStudentResidentUpdateErrorStopsBeforeProjectionUpdate(t *testing.T) {
	trainer := newCompactEmbeddingTrainerForTest(t, 3)
	trainer.optimizerAccel = &fakeResidentOptimizerAccelerator{
		applyErr: fmt.Errorf("forced student optimizer failure"),
	}
	trainer.forwardMatMul = &residentAwareCountingMatMulAccelerator{}
	trainer.deferOptimizerSync = true
	proj := &vectorDistillProjectionState{
		W:        []float32{0.2, -0.1, 0.05, 0.3, -0.4, 0.25, 0.15, 0.2, -0.05, 0.1, -0.3, 0.4},
		Mom1:     make([]float32, 12),
		Mom2:     make([]float32, 12),
		InputDim: 3,
		OutDim:   4,
	}
	before := append([]float32(nil), proj.W...)
	batch := []EmbeddingTokenizedVectorDistillExample{
		vectorDistillTestExample("a", EmbeddingRoleQuery, []int32{1, 2, 3}, []float32{0.2, -0.1, 0.05, 0.3}),
		vectorDistillTestExample("b", EmbeddingRoleDocument, []int32{3, 2, 1}, []float32{-0.1, 0.4, 0.2, -0.05}),
	}

	_, gotProj, err := trainer.trainVectorDistillBatch(batch, proj, 4, EmbeddingRoleQuery, 0)
	if err == nil || !strings.Contains(err.Error(), "forced student optimizer failure") {
		t.Fatalf("trainVectorDistillBatch error = %v, want student optimizer failure", err)
	}
	if gotProj != proj {
		t.Fatal("trainVectorDistillBatch returned a different projection after failure")
	}
	if proj.Step != 0 {
		t.Fatalf("projection step = %d, want 0 because projection update must not run", proj.Step)
	}
	assertCloseF32Slice(t, "projection after failed student update", proj.W, before, 0)
}

func TestVectorDistillResidentRouteFailuresAbortAndKeepSelectedFalse(t *testing.T) {
	cases := []struct {
		name      string
		proj      *vectorDistillProjectionState
		setup     func(*fakeCompactTrainAccelerator, *fakeResidentOptimizerAccelerator)
		clear     func(*fakeCompactTrainAccelerator, *fakeResidentOptimizerAccelerator)
		wantAbort int64
		wantEnd   int64
		wantErr   string
	}{
		{
			name: "forward",
			proj: residentRouteProjectionForTest(),
			setup: func(train *fakeCompactTrainAccelerator, _ *fakeResidentOptimizerAccelerator) {
				train.forwardErr = fmt.Errorf("forced forward failure")
			},
			clear: func(train *fakeCompactTrainAccelerator, _ *fakeResidentOptimizerAccelerator) {
				train.forwardErr = nil
			},
			wantAbort: 1,
			wantErr:   "forced forward failure",
		},
		{
			name: "loss",
			proj: &vectorDistillProjectionState{
				W:        make([]float32, 3*3),
				Mom1:     make([]float32, 3*3),
				Mom2:     make([]float32, 3*3),
				InputDim: 3,
				OutDim:   3,
			},
			wantAbort: 1,
			wantErr:   "proj length",
		},
		{
			name: "backward",
			proj: residentRouteProjectionForTest(),
			setup: func(train *fakeCompactTrainAccelerator, _ *fakeResidentOptimizerAccelerator) {
				train.backwardErr = fmt.Errorf("forced backward validation failure")
			},
			clear: func(train *fakeCompactTrainAccelerator, _ *fakeResidentOptimizerAccelerator) {
				train.backwardErr = nil
			},
			wantAbort: 1,
			wantErr:   "forced backward validation failure",
		},
		{
			name: "end",
			proj: residentRouteProjectionForTest(),
			setup: func(train *fakeCompactTrainAccelerator, _ *fakeResidentOptimizerAccelerator) {
				train.endErr = fmt.Errorf("forced end failure")
			},
			clear: func(train *fakeCompactTrainAccelerator, _ *fakeResidentOptimizerAccelerator) {
				train.endErr = nil
			},
			wantAbort: 1,
			wantEnd:   1,
			wantErr:   "forced end failure",
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			trainer := newCompactEmbeddingTrainerForTest(t, 3)
			t.Cleanup(trainer.Close)
			t.Setenv(compactResidentTrainEnv, "1")
			opt := &fakeResidentOptimizerAccelerator{applyErrNames: map[string]error{}}
			train := &fakeCompactTrainAccelerator{supportBackward: true}
			trainer.optimizerAccel = opt
			trainer.forwardMatMul = &residentAwareCountingMatMulAccelerator{}
			trainer.compactTrainAccel = train
			trainer.compactTrainBackend = eosartifact.BackendCUDA
			trainer.compactForwardSelected = true
			if tc.setup != nil {
				tc.setup(train, opt)
			}
			proj := tc.proj
			proj.ResidentName = "resident_route_projection"
			beforeStep := trainer.step
			beforeUpdates := trainer.compactOptimizerUpdates
			_, gotProj, err := trainer.trainVectorDistillBatch([]EmbeddingTokenizedVectorDistillExample{
				vectorDistillTestExample("a", EmbeddingRoleQuery, []int32{1, 2, 3}, []float32{0.2, -0.1, 0.05, 0.3}),
			}, proj, 4, EmbeddingRoleQuery, 0)
			if err == nil || !strings.Contains(err.Error(), tc.wantErr) {
				t.Fatalf("resident route error = %v, want %q", err, tc.wantErr)
			}
			if gotProj != proj {
				t.Fatal("resident route returned a different projection on failure")
			}
			if trainer.compactForwardSelected {
				t.Fatal("resident route failure left compactForwardSelected=true")
			}
			if train.abortCalls != tc.wantAbort || train.endCalls != tc.wantEnd {
				t.Fatalf("abort/end calls = %d/%d, want %d/%d", train.abortCalls, train.endCalls, tc.wantAbort, tc.wantEnd)
			}
			if train.stats.LiveHandles != 0 {
				t.Fatalf("live handles after failure = %d, want 0", train.stats.LiveHandles)
			}
			if train.activeStep != 0 || train.sealedStep != 0 || len(train.gradientTokens) != 0 {
				t.Fatalf("step residue after failure active/sealed/grad_tokens=%d/%d/%d", train.activeStep, train.sealedStep, len(train.gradientTokens))
			}
			if trainer.step != beforeStep || trainer.compactState.Step != beforeStep || trainer.compactOptimizerUpdates != beforeUpdates {
				t.Fatalf("published state changed on failure: step=%d stateStep=%d updates=%d, want %d/%d", trainer.step, trainer.compactState.Step, trainer.compactOptimizerUpdates, beforeStep, beforeUpdates)
			}
			if tc.clear != nil {
				tc.clear(train, opt)
			}
			retryProj := residentRouteProjectionForTest()
			retryProj.ResidentName = "resident_route_projection_retry"
			_, _, err = trainer.trainVectorDistillBatch([]EmbeddingTokenizedVectorDistillExample{
				vectorDistillTestExample("retry", EmbeddingRoleQuery, []int32{1, 2, 3}, []float32{0.2, -0.1, 0.05, 0.3}),
			}, retryProj, 4, EmbeddingRoleQuery, 0)
			if err != nil {
				t.Fatalf("retry after %s failure: %v", tc.name, err)
			}
			if !trainer.compactForwardSelected || trainer.step != beforeStep+1 || trainer.compactState.Step != beforeStep+1 || train.stats.LiveHandles != 0 {
				t.Fatalf("retry after %s did not publish clean state: selected=%t step=%d stateStep=%d live=%d", tc.name, trainer.compactForwardSelected, trainer.step, trainer.compactState.Step, train.stats.LiveHandles)
			}
		})
	}
}

func TestVectorDistillResidentRouteSuccessPublishesSelectedAfterFullStep(t *testing.T) {
	trainer := newCompactEmbeddingTrainerForTest(t, 3)
	t.Cleanup(trainer.Close)
	t.Setenv(compactResidentTrainEnv, "1")
	opt := &fakeResidentOptimizerAccelerator{applyErrNames: map[string]error{}}
	train := &fakeCompactTrainAccelerator{supportBackward: true}
	trainer.optimizerAccel = opt
	trainer.forwardMatMul = &residentAwareCountingMatMulAccelerator{}
	trainer.compactTrainAccel = train
	trainer.compactTrainBackend = eosartifact.BackendCUDA
	proj := residentRouteProjectionForTest()
	proj.ResidentName = "resident_route_projection"
	_, gotProj, err := trainer.trainVectorDistillBatch([]EmbeddingTokenizedVectorDistillExample{
		vectorDistillTestExample("a", EmbeddingRoleQuery, []int32{1, 2, 3}, []float32{0.2, -0.1, 0.05, 0.3}),
	}, proj, 4, EmbeddingRoleQuery, 0)
	if err != nil {
		t.Fatalf("resident route success: %v", err)
	}
	if gotProj != proj || proj.Step != 1 || trainer.step != 1 || trainer.compactState.Step != 1 || trainer.compactOptimizerUpdates != 1 || !trainer.compactForwardSelected {
		t.Fatalf("published success state projStep=%d step=%d stateStep=%d updates=%d selected=%t", proj.Step, trainer.step, trainer.compactState.Step, trainer.compactOptimizerUpdates, trainer.compactForwardSelected)
	}
	if train.abortCalls != 0 || train.endCalls != 1 || train.stats.LiveHandles != 0 {
		t.Fatalf("success abort/end/live = %d/%d/%d, want 0/1/0", train.abortCalls, train.endCalls, train.stats.LiveHandles)
	}
}

func TestVectorDistillResidentOptimizerLatePreflightFailureMutatesNothing(t *testing.T) {
	trainer := newCompactEmbeddingTrainerForTest(t, 3)
	t.Cleanup(trainer.Close)
	t.Setenv(compactResidentTrainEnv, "1")
	items := compactTrainStateOptimizerItems(trainer.compactState)
	if len(items) < 2 {
		t.Fatalf("compact optimizer items = %d, want at least two", len(items))
	}
	lateName := items[len(items)-1].Name
	items[0].Moment1 = nil
	items[0].Moment2 = nil
	opt := &fakeResidentOptimizerAccelerator{
		applyErrNames:     map[string]error{},
		preflightErrNames: map[string]error{lateName: fmt.Errorf("forced late predictable preflight failure")},
	}
	train := &fakeCompactTrainAccelerator{supportBackward: true}
	trainer.optimizerAccel = opt
	trainer.forwardMatMul = &residentAwareCountingMatMulAccelerator{}
	trainer.compactTrainAccel = train
	trainer.compactTrainBackend = eosartifact.BackendCUDA
	allTensors := make([]*backend.Tensor, 0, len(items)*3)
	for _, item := range items {
		allTensors = append(allTensors, item.Tensor, item.Moment1, item.Moment2)
	}
	before := snapshotEmbeddingTrainTensors(allTensors...)
	proj := residentRouteProjectionForTest()
	projWBefore := append([]float32(nil), proj.W...)
	projMom1Before := append([]float32(nil), proj.Mom1...)
	projMom2Before := append([]float32(nil), proj.Mom2...)
	_, _, err := trainer.trainVectorDistillBatch([]EmbeddingTokenizedVectorDistillExample{
		vectorDistillTestExample("a", EmbeddingRoleQuery, []int32{1, 2, 3}, []float32{0.2, -0.1, 0.05, 0.3}),
	}, proj, 4, EmbeddingRoleQuery, 0)
	if err == nil || !strings.Contains(err.Error(), "forced late predictable preflight failure") {
		t.Fatalf("resident route error = %v, want late preflight failure", err)
	}
	if opt.preflightCalls != int64(len(items)) || opt.residentApplyCalls != 0 {
		t.Fatalf("preflight/resident update calls = %d/%d, want %d/0", opt.preflightCalls, opt.residentApplyCalls, len(items))
	}
	if delta := aggregateParameterDeltaStats(before); delta.NonzeroCount != 0 {
		t.Fatalf("parameter/moment mutation after preflight failure: %+v", delta)
	}
	if items[0].Moment1 != nil || items[0].Moment2 != nil {
		t.Fatal("preflight failure assigned previously nil compact moments")
	}
	assertCloseF32Slice(t, "projection weights after preflight failure", proj.W, projWBefore, 0)
	assertCloseF32Slice(t, "projection moment1 after preflight failure", proj.Mom1, projMom1Before, 0)
	assertCloseF32Slice(t, "projection moment2 after preflight failure", proj.Mom2, projMom2Before, 0)
	if trainer.compactForwardSelected || trainer.step != 0 || trainer.compactState.Step != 0 || trainer.compactOptimizerUpdates != 0 {
		t.Fatalf("preflight failure published state: selected=%t step=%d stateStep=%d updates=%d", trainer.compactForwardSelected, trainer.step, trainer.compactState.Step, trainer.compactOptimizerUpdates)
	}
	if train.abortCalls != 1 || train.activeStep != 0 || train.sealedStep != 0 || len(train.gradientTokens) != 0 {
		t.Fatalf("post-End preflight abort residue abort/active/sealed/grad_tokens=%d/%d/%d/%d", train.abortCalls, train.activeStep, train.sealedStep, len(train.gradientTokens))
	}
	delete(opt.preflightErrNames, lateName)
	if _, _, err := trainer.trainVectorDistillBatch([]EmbeddingTokenizedVectorDistillExample{
		vectorDistillTestExample("retry", EmbeddingRoleQuery, []int32{1, 2, 3}, []float32{0.2, -0.1, 0.05, 0.3}),
	}, proj, 4, EmbeddingRoleQuery, 0); err != nil {
		t.Fatalf("retry after preflight failure: %v", err)
	}
	if !trainer.compactForwardSelected || trainer.step != 1 || opt.residentApplyCalls != int64(len(items)) {
		t.Fatalf("retry state selected=%t step=%d residentUpdates=%d, want true/1/%d", trainer.compactForwardSelected, trainer.step, opt.residentApplyCalls, len(items))
	}
}

func TestVectorDistillResidentProjectionPreflightFailureMutatesNothingAndRetries(t *testing.T) {
	trainer := newCompactEmbeddingTrainerForTest(t, 3)
	t.Cleanup(trainer.Close)
	t.Setenv(compactResidentTrainEnv, "1")
	items := compactTrainStateOptimizerItems(trainer.compactState)
	opt := &fakeResidentOptimizerAccelerator{
		applyErrNames:     map[string]error{},
		preflightErrNames: map[string]error{"resident_route_projection": fmt.Errorf("forced projection preflight failure")},
	}
	train := &fakeCompactTrainAccelerator{supportBackward: true}
	trainer.optimizerAccel = opt
	trainer.forwardMatMul = &residentAwareCountingMatMulAccelerator{}
	trainer.compactTrainAccel = train
	trainer.compactTrainBackend = eosartifact.BackendCUDA
	allTensors := make([]*backend.Tensor, 0, len(items)*3)
	for _, item := range items {
		allTensors = append(allTensors, item.Tensor, item.Moment1, item.Moment2)
	}
	before := snapshotEmbeddingTrainTensors(allTensors...)
	proj := residentRouteProjectionForTest()
	proj.ResidentName = "resident_route_projection"
	projBefore := append([]float32(nil), proj.W...)
	projMom1Before := append([]float32(nil), proj.Mom1...)
	projMom2Before := append([]float32(nil), proj.Mom2...)
	_, _, err := trainer.trainVectorDistillBatch([]EmbeddingTokenizedVectorDistillExample{
		vectorDistillTestExample("a", EmbeddingRoleQuery, []int32{1, 2, 3}, []float32{0.2, -0.1, 0.05, 0.3}),
	}, proj, 4, EmbeddingRoleQuery, 0)
	if err == nil || !strings.Contains(err.Error(), "forced projection preflight failure") {
		t.Fatalf("resident route error = %v, want projection preflight failure", err)
	}
	if opt.preflightCalls != int64(len(items)) || opt.preflightApplyCalls != 1 || opt.residentApplyCalls != 0 || opt.applyCalls != 0 {
		t.Fatalf("preflight/apply calls resident=%d projection=%d residentApply=%d apply=%d, want %d/1/0/0", opt.preflightCalls, opt.preflightApplyCalls, opt.residentApplyCalls, opt.applyCalls, len(items))
	}
	if delta := aggregateParameterDeltaStats(before); delta.NonzeroCount != 0 {
		t.Fatalf("student parameter/moment mutation after projection preflight failure: %+v", delta)
	}
	assertCloseF32Slice(t, "projection weights after preflight failure", proj.W, projBefore, 0)
	assertCloseF32Slice(t, "projection moment1 after preflight failure", proj.Mom1, projMom1Before, 0)
	assertCloseF32Slice(t, "projection moment2 after preflight failure", proj.Mom2, projMom2Before, 0)
	if trainer.optimizerPoisonErr != nil || trainer.compactForwardSelected || trainer.step != 0 || trainer.compactState.Step != 0 || trainer.compactOptimizerUpdates != 0 {
		t.Fatalf("projection preflight failure state poison=%v selected=%t step=%d stateStep=%d updates=%d", trainer.optimizerPoisonErr, trainer.compactForwardSelected, trainer.step, trainer.compactState.Step, trainer.compactOptimizerUpdates)
	}
	if train.abortCalls != 1 || train.activeStep != 0 || train.sealedStep != 0 || len(train.gradientTokens) != 0 {
		t.Fatalf("projection preflight abort residue abort/active/sealed/grad_tokens=%d/%d/%d/%d", train.abortCalls, train.activeStep, train.sealedStep, len(train.gradientTokens))
	}
	delete(opt.preflightErrNames, "resident_route_projection")
	_, _, err = trainer.trainVectorDistillBatch([]EmbeddingTokenizedVectorDistillExample{
		vectorDistillTestExample("retry", EmbeddingRoleQuery, []int32{1, 2, 3}, []float32{0.2, -0.1, 0.05, 0.3}),
	}, proj, 4, EmbeddingRoleQuery, 0)
	if err != nil {
		t.Fatalf("retry after projection preflight failure: %v", err)
	}
	if !trainer.compactForwardSelected || trainer.step != 1 || trainer.compactState.Step != 1 || opt.residentApplyCalls != int64(len(items)) {
		t.Fatalf("retry state selected=%t step=%d stateStep=%d residentUpdates=%d, want true/1/1/%d", trainer.compactForwardSelected, trainer.step, trainer.compactState.Step, opt.residentApplyCalls, len(items))
	}
}

func TestVectorDistillResidentStudentLateApplyFailurePoisonsAndRefusesRetry(t *testing.T) {
	trainer := newCompactEmbeddingTrainerForTest(t, 3)
	t.Cleanup(trainer.Close)
	t.Setenv(compactResidentTrainEnv, "1")
	trainer.config.WeightDecay = 0.01
	items := compactTrainStateOptimizerItems(trainer.compactState)
	if len(items) < 2 {
		t.Fatalf("compact optimizer items = %d, want at least two", len(items))
	}
	lateName := items[1].Name
	opt := &fakeResidentOptimizerAccelerator{
		applyErrNames:     map[string]error{lateName: fmt.Errorf("forced late student apply failure")},
		preflightErrNames: map[string]error{},
	}
	train := &fakeCompactTrainAccelerator{supportBackward: true}
	trainer.optimizerAccel = opt
	trainer.forwardMatMul = &residentAwareCountingMatMulAccelerator{}
	trainer.compactTrainAccel = train
	trainer.compactTrainBackend = eosartifact.BackendCUDA
	allTensors := make([]*backend.Tensor, 0, len(items)*3)
	for _, item := range items {
		allTensors = append(allTensors, item.Tensor, item.Moment1, item.Moment2)
	}
	before := snapshotEmbeddingTrainTensors(allTensors...)
	proj := residentRouteProjectionForTest()
	proj.ResidentName = "resident_route_projection"
	projBefore := append([]float32(nil), proj.W...)
	_, _, err := trainer.trainVectorDistillBatch([]EmbeddingTokenizedVectorDistillExample{
		vectorDistillTestExample("a", EmbeddingRoleQuery, []int32{1, 2, 3}, []float32{0.2, -0.1, 0.05, 0.3}),
	}, proj, 4, EmbeddingRoleQuery, 0)
	if err == nil || !strings.Contains(err.Error(), "poisoned") || !strings.Contains(err.Error(), "forced late student apply failure") {
		t.Fatalf("resident route error = %v, want poisoned late student apply failure", err)
	}
	if trainer.optimizerPoisonErr == nil || trainer.compactForwardSelected || trainer.step != 0 || trainer.compactState.Step != 0 || trainer.compactOptimizerUpdates != 0 {
		t.Fatalf("late student failure state poison=%v selected=%t step=%d stateStep=%d updates=%d", trainer.optimizerPoisonErr, trainer.compactForwardSelected, trainer.step, trainer.compactState.Step, trainer.compactOptimizerUpdates)
	}
	if opt.preflightCalls != int64(len(items)) || opt.preflightApplyCalls != 1 || opt.residentApplyCalls < 2 || opt.residentApplyCalls >= int64(len(items)) || opt.applyCalls != opt.residentApplyCalls {
		t.Fatalf("late student calls preflight=%d projectionPreflight=%d residentApply=%d apply=%d len=%d", opt.preflightCalls, opt.preflightApplyCalls, opt.residentApplyCalls, opt.applyCalls, len(items))
	}
	if delta := aggregateParameterDeltaStats(before); delta.NonzeroCount == 0 {
		t.Fatal("late student apply failure did not mutate any prior student optimizer state; test must model unsafe retry")
	}
	if proj.Step != 0 {
		t.Fatalf("projection step after student late failure = %d, want 0", proj.Step)
	}
	assertCloseF32Slice(t, "projection after late student failure", proj.W, projBefore, 0)
	if train.abortCalls != 1 || train.activeStep != 0 || train.sealedStep != 0 || len(train.gradientTokens) != 0 {
		t.Fatalf("late student abort residue abort/active/sealed/grad_tokens=%d/%d/%d/%d", train.abortCalls, train.activeStep, train.sealedStep, len(train.gradientTokens))
	}
	delete(opt.applyErrNames, lateName)
	if _, _, err := trainer.trainVectorDistillBatch([]EmbeddingTokenizedVectorDistillExample{
		vectorDistillTestExample("retry", EmbeddingRoleQuery, []int32{1, 2, 3}, []float32{0.2, -0.1, 0.05, 0.3}),
	}, proj, 4, EmbeddingRoleQuery, 0); err == nil || !strings.Contains(err.Error(), "close and create a new trainer") {
		t.Fatalf("retry after poison error = %v, want fail-closed poison", err)
	}
	if _, err := trainer.TrainStep(nil); err == nil || !strings.Contains(err.Error(), "poisoned") {
		t.Fatalf("TrainStep after poison error = %v, want poisoned", err)
	}
	assertPoisonedTrainerRejectsOptimizerPublication(t, trainer, opt, train, proj)
}

func TestVectorDistillResidentProjectionLateApplyFailurePoisonsAndRefusesRetry(t *testing.T) {
	trainer := newCompactEmbeddingTrainerForTest(t, 3)
	t.Cleanup(trainer.Close)
	t.Setenv(compactResidentTrainEnv, "1")
	trainer.config.WeightDecay = 0.01
	items := compactTrainStateOptimizerItems(trainer.compactState)
	opt := &fakeResidentOptimizerAccelerator{
		applyErrNames:     map[string]error{"resident_route_projection": fmt.Errorf("forced late projection apply failure")},
		preflightErrNames: map[string]error{},
	}
	train := &fakeCompactTrainAccelerator{supportBackward: true}
	trainer.optimizerAccel = opt
	trainer.forwardMatMul = &residentAwareCountingMatMulAccelerator{}
	trainer.compactTrainAccel = train
	trainer.compactTrainBackend = eosartifact.BackendCUDA
	allTensors := make([]*backend.Tensor, 0, len(items)*3)
	for _, item := range items {
		allTensors = append(allTensors, item.Tensor, item.Moment1, item.Moment2)
	}
	before := snapshotEmbeddingTrainTensors(allTensors...)
	proj := residentRouteProjectionForTest()
	proj.ResidentName = "resident_route_projection"
	projBefore := append([]float32(nil), proj.W...)
	projMom1Before := append([]float32(nil), proj.Mom1...)
	projMom2Before := append([]float32(nil), proj.Mom2...)
	_, _, err := trainer.trainVectorDistillBatch([]EmbeddingTokenizedVectorDistillExample{
		vectorDistillTestExample("a", EmbeddingRoleQuery, []int32{1, 2, 3}, []float32{0.2, -0.1, 0.05, 0.3}),
	}, proj, 4, EmbeddingRoleQuery, 0)
	if err == nil || !strings.Contains(err.Error(), "poisoned") || !strings.Contains(err.Error(), "forced late projection apply failure") {
		t.Fatalf("resident route error = %v, want poisoned projection apply failure", err)
	}
	if trainer.optimizerPoisonErr == nil || trainer.compactForwardSelected || trainer.step != 0 || trainer.compactState.Step != 0 || trainer.compactOptimizerUpdates != 0 {
		t.Fatalf("late projection failure state poison=%v selected=%t step=%d stateStep=%d updates=%d", trainer.optimizerPoisonErr, trainer.compactForwardSelected, trainer.step, trainer.compactState.Step, trainer.compactOptimizerUpdates)
	}
	if opt.preflightCalls != int64(len(items)) || opt.preflightApplyCalls != 1 || opt.residentApplyCalls != int64(len(items)) || opt.applyCalls != int64(len(items)+1) {
		t.Fatalf("late projection calls preflight=%d projectionPreflight=%d residentApply=%d apply=%d, want %d/1/%d/%d", opt.preflightCalls, opt.preflightApplyCalls, opt.residentApplyCalls, opt.applyCalls, len(items), len(items), len(items)+1)
	}
	if delta := aggregateParameterDeltaStats(before); delta.NonzeroCount == 0 {
		t.Fatal("late projection apply failure did not mutate student optimizer state before failing; test must model unsafe retry")
	}
	if proj.Step != 0 {
		t.Fatalf("projection step after projection late failure = %d, want 0", proj.Step)
	}
	assertCloseF32Slice(t, "projection weights after failed projection apply", proj.W, projBefore, 0)
	assertCloseF32Slice(t, "projection moment1 after failed projection apply", proj.Mom1, projMom1Before, 0)
	assertCloseF32Slice(t, "projection moment2 after failed projection apply", proj.Mom2, projMom2Before, 0)
	if train.abortCalls != 1 || train.activeStep != 0 || train.sealedStep != 0 || len(train.gradientTokens) != 0 {
		t.Fatalf("late projection abort residue abort/active/sealed/grad_tokens=%d/%d/%d/%d", train.abortCalls, train.activeStep, train.sealedStep, len(train.gradientTokens))
	}
	delete(opt.applyErrNames, "resident_route_projection")
	if _, _, err := trainer.trainVectorDistillBatch([]EmbeddingTokenizedVectorDistillExample{
		vectorDistillTestExample("retry", EmbeddingRoleQuery, []int32{1, 2, 3}, []float32{0.2, -0.1, 0.05, 0.3}),
	}, proj, 4, EmbeddingRoleQuery, 0); err == nil || !strings.Contains(err.Error(), "close and create a new trainer") {
		t.Fatalf("retry after projection poison error = %v, want fail-closed poison", err)
	}
	assertPoisonedTrainerRejectsOptimizerPublication(t, trainer, opt, train, proj)
}

func assertPoisonedTrainerRejectsOptimizerPublication(t *testing.T, trainer *EmbeddingTrainer, opt *fakeResidentOptimizerAccelerator, train *fakeCompactTrainAccelerator, proj *vectorDistillProjectionState) {
	t.Helper()
	if trainer.optimizerPoisonErr == nil {
		t.Fatal("trainer is not poisoned")
	}
	if !trainer.momentsDirty {
		t.Fatal("poisoned trainer cleared momentsDirty; want forensic dirty state preserved")
	}
	baseSyncs := opt.stats.SyncCalls
	baseDownloads := opt.stats.DownloadedBytes

	assertNoSync := func(label string) {
		t.Helper()
		if opt.stats.SyncCalls != baseSyncs || opt.stats.DownloadedBytes != baseDownloads {
			t.Fatalf("%s changed sync/download counters: sync %d->%d downloaded %d->%d", label, baseSyncs, opt.stats.SyncCalls, baseDownloads, opt.stats.DownloadedBytes)
		}
		if !trainer.momentsDirty {
			t.Fatalf("%s cleared momentsDirty on poisoned trainer", label)
		}
	}
	wantPoison := func(label string, err error) {
		t.Helper()
		if err == nil || !strings.Contains(err.Error(), "poisoned") {
			t.Fatalf("%s error = %v, want poison", label, err)
		}
		assertNoSync(label)
	}

	wantPoison("direct optimizer sync", trainer.syncOptimizerStateWithReason(true, "test_poison_direct_sync"))
	_, err := trainer.Checkpoint()
	wantPoison("checkpoint", err)
	weights, err := trainer.ExportInferenceWeights()
	if weights != nil {
		t.Fatalf("export returned %d weights on poisoned trainer", len(weights))
	}
	wantPoison("export", err)
	_, err = trainer.EvaluatePairs(nil)
	wantPoison("eval", err)
	_, err = trainer.EvalBatch(nil)
	wantPoison("eval batch", err)
	if proj != nil {
		wantPoison("projection sync", trainer.syncVectorDistillProjectionState(proj, "test_poison_projection_sync"))
	}

	priorCloseErr := fmt.Errorf("prior close failure")
	trainer.closeErr = priorCloseErr
	err = trainer.CloseWithError()
	wantPoison("close", err)
	if !strings.Contains(err.Error(), priorCloseErr.Error()) {
		t.Fatalf("close error = %v, want prior close error preserved", err)
	}
	if trainer.CloseError() == nil || !strings.Contains(trainer.CloseError().Error(), "poisoned") {
		t.Fatalf("CloseError after poison close = %v, want poison", trainer.CloseError())
	}
	if !opt.closed {
		t.Fatal("optimizer accelerator was not released by poisoned close")
	}
	if !train.closed {
		t.Fatal("compact train accelerator was not released by poisoned close")
	}
	if trainer.optimizerAccel != nil || trainer.compactTrainAccel != nil || trainer.forwardMatMul != nil {
		t.Fatalf("poisoned close left accelerators optimizer=%T compactTrain=%T forward=%T", trainer.optimizerAccel, trainer.compactTrainAccel, trainer.forwardMatMul)
	}
	assertNoSync("close release")
	secondCloseErr := trainer.CloseWithError()
	if secondCloseErr != err {
		t.Fatalf("second poisoned CloseWithError = %v, want stable first close error %v", secondCloseErr, err)
	}
	if strings.Count(secondCloseErr.Error(), "poisoned") != 1 {
		t.Fatalf("second poisoned CloseWithError duplicated poison context: %v", secondCloseErr)
	}
	assertNoSync("second close")
}

func TestVectorDistillResidentRouteVaryingBucketsDuplicateAliasesAndRetry(t *testing.T) {
	trainer := newCompactEmbeddingTrainerForTest(t, 3)
	t.Cleanup(trainer.Close)
	t.Setenv(compactResidentTrainEnv, "1")
	opt := &fakeResidentOptimizerAccelerator{applyErrNames: map[string]error{}, preflightErrNames: map[string]error{}}
	train := &fakeCompactTrainAccelerator{supportBackward: true}
	trainer.optimizerAccel = opt
	trainer.forwardMatMul = &residentAwareCountingMatMulAccelerator{}
	trainer.compactTrainAccel = train
	trainer.compactTrainBackend = eosartifact.BackendCUDA
	batch := []EmbeddingTokenizedVectorDistillExample{
		vectorDistillTestExample("t3-a", EmbeddingRoleQuery, []int32{1, 2, 3}, []float32{0.2, -0.1, 0.05, 0.3}),
		vectorDistillTestExample("t2-a", EmbeddingRoleDocument, []int32{3, 4}, []float32{-0.1, 0.4, 0.2, -0.05}),
		vectorDistillTestExample("t3-alias", EmbeddingRoleQuery, []int32{1, 2, 3}, []float32{0.3, 0.1, -0.2, 0.05}),
		vectorDistillTestExample("t4-a", EmbeddingRoleRaw, []int32{4, 3, 2, 1}, []float32{0.15, -0.25, 0.35, -0.05}),
		vectorDistillTestExample("t2-alias", EmbeddingRoleDocument, []int32{3, 4}, []float32{-0.2, 0.3, 0.1, 0.25}),
	}
	badProj := &vectorDistillProjectionState{W: make([]float32, 3*3), Mom1: make([]float32, 3*3), Mom2: make([]float32, 3*3), InputDim: 3, OutDim: 3}
	if _, _, err := trainer.trainVectorDistillBatch(batch, badProj, 4, EmbeddingRoleQuery, 0); err == nil || !strings.Contains(err.Error(), "proj length") {
		t.Fatalf("pre-backward injection error = %v, want projection/loss failure", err)
	}
	if train.abortCalls != 1 || train.backwardCalls != 0 || train.stats.LiveHandles != 0 || trainer.compactForwardSelected {
		t.Fatalf("pre-backward abort state abort=%d backward=%d live=%d selected=%t", train.abortCalls, train.backwardCalls, train.stats.LiveHandles, trainer.compactForwardSelected)
	}

	proj := residentRouteProjectionForTest()
	projWReference := append([]float32(nil), proj.W...)
	if _, _, err := trainer.trainVectorDistillBatch(batch, proj, 4, EmbeddingRoleQuery, 0); err != nil {
		t.Fatalf("retry resident route: %v", err)
	}
	if !trainer.compactForwardSelected || trainer.step != 1 || train.beginCalls != 2 || train.forwardCalls != 6 || train.backwardCalls != 3 || train.endCalls != 1 {
		t.Fatalf("route counters selected=%t step=%d begin/forward/backward/end=%d/%d/%d/%d", trainer.compactForwardSelected, trainer.step, train.beginCalls, train.forwardCalls, train.backwardCalls, train.endCalls)
	}
	requests := train.forwardRequests[len(train.forwardRequests)-3:]
	for i, wantT := range []int{2, 3, 4} {
		if requests[i].Shape.Tokens != wantT || requests[i].Shape.Batch != 1 {
			t.Fatalf("bucket %d shape B/T=%d/%d, want 1/%d", i, requests[i].Shape.Batch, requests[i].Shape.Tokens, wantT)
		}
	}
	if len(train.backwardGrads) != 3 {
		t.Fatalf("backward gradients = %d, want one per exact-T bucket", len(train.backwardGrads))
	}

	// Host/reference aggregation: the fake resident forward returns zero pooled
	// rows. Compute each input's host projection gradient independently, then
	// sum the two exact aliases into their single pooled resident row.
	perInput := make([][]float32, len(batch))
	for i := range batch {
		loss, err := VectorDistillLossAndGrad(make([]float32, proj.OutDim), batch[i].TeacherVector)
		if err != nil {
			t.Fatalf("host reference loss %d: %v", i, err)
		}
		perInput[i] = make([]float32, proj.InputDim)
		for k := 0; k < proj.InputDim; k++ {
			for j := 0; j < proj.OutDim; j++ {
				perInput[i][k] += loss.GradProj[j] * projWReference[k*proj.OutDim+j]
			}
		}
	}
	sum := func(indices ...int) []float32 {
		out := make([]float32, proj.InputDim)
		for _, index := range indices {
			for k := range out {
				out[k] += perInput[index][k]
			}
		}
		return out
	}
	wantByBucket := [][]float32{sum(1, 4), sum(0, 2), sum(3)}
	for i, want := range wantByBucket {
		assertCloseF32Slice(t, fmt.Sprintf("bucket %d pooled alias gradient", i), train.backwardGrads[i].F32, want, 0)
	}
}

func residentRouteProjectionForTest() *vectorDistillProjectionState {
	return &vectorDistillProjectionState{
		W:        []float32{0.2, -0.1, 0.05, 0.3, -0.4, 0.25, 0.15, 0.2, -0.05, 0.1, -0.3, 0.4},
		Mom1:     make([]float32, 12),
		Mom2:     make([]float32, 12),
		InputDim: 3,
		OutDim:   4,
	}
}

func TestVectorDistillProjectionResidentMatMulFallbackSyncFailureFailsClosed(t *testing.T) {
	trainer := newCompactEmbeddingTrainerForTest(t, 3)
	proj := &vectorDistillProjectionState{
		W:        []float32{0.2, -0.1, 0.05, 0.3, -0.4, 0.25},
		Mom1:     []float32{0, 0, 0, 0, 0, 0},
		Mom2:     []float32{0, 0, 0, 0, 0, 0},
		InputDim: 2,
		OutDim:   3,
		Step:     1,
	}
	name := proj.residentName()
	trainer.optimizerAccel = &fakeResidentOptimizerAccelerator{
		syncErr: fmt.Errorf("forced projection fallback sync failure"),
		resident: map[string]*fakeResidentOptimizerToken{
			name: {tensor: backend.NewTensorF32([]int{2, 3}, append([]float32(nil), proj.W...)), generation: 1, alive: true},
		},
	}
	trainer.forwardMatMul = &countingMatMulAccelerator{}
	trainer.deferOptimizerSync = true
	trainer.momentsDirty = true

	out := make([]float32, proj.OutDim)
	err := trainer.projectVectorDistillStudent([]float32{0.7, -0.2}, proj, out)
	if err == nil || !strings.Contains(err.Error(), "forced projection fallback sync failure") {
		t.Fatalf("projectVectorDistillStudent error = %v, want fallback sync failure", err)
	}
	assertCloseF32Slice(t, "projection output after failed fallback", out, make([]float32, proj.OutDim), 0)
}

func TestVectorDistillCompactForwardResidencyAndQKVCounters(t *testing.T) {
	trainer := newCompactEmbeddingTrainerForTest(t, 3)
	if trainer.forwardMatMul != nil {
		trainer.forwardMatMul.Close()
	}
	fake := &countingMatMulAccelerator{}
	trainer.forwardMatMul = fake
	trainer.forwardBackend = eosartifact.BackendCUDA
	trainer.optimizerAccel = nil

	batch := []EmbeddingTokenizedVectorDistillExample{
		vectorDistillTestExample("a", EmbeddingRoleQuery, []int32{1, 2, 3}, []float32{0.2, -0.1, 0.05, 0.3}),
		vectorDistillTestExample("b", EmbeddingRoleDocument, []int32{3, 2, 1}, []float32{-0.1, 0.4, 0.2, -0.05}),
	}
	metrics, proj, err := trainer.trainVectorDistillBatch(batch, nil, 4, EmbeddingRoleQuery, 0)
	if err != nil {
		t.Fatalf("trainVectorDistillBatch: %v", err)
	}
	if proj == nil || !compactTestFinite(metrics.Loss) {
		t.Fatalf("metrics=%+v projection=%v, want finite loss and projection", metrics, proj)
	}

	stats := trainer.ForwardResidencyStats()
	if stats.MatMul.BindCalls == 0 {
		t.Fatalf("matmul bind calls = %d, want compact forward weights resident", stats.MatMul.BindCalls)
	}
	if stats.MatMul.BoundRightCalls == 0 {
		t.Fatalf("bound-right calls = %d, want resident compact weights used", stats.MatMul.BoundRightCalls)
	}
	if fake.multiBoundRuns == 0 {
		t.Fatalf("q/k/v multi-bound runs = %d, want coalesced compact QKV", fake.multiBoundRuns)
	}
	if stats.MatMul.RunCalls == 0 {
		t.Fatalf("run calls = %d, want accelerated compact matmuls recorded", stats.MatMul.RunCalls)
	}

	bindCallsAfterStep := fake.bindCalls
	forward := trainer.prepareForwardWeights()
	if forward == nil || forward.compact == nil {
		t.Fatal("missing compact forward cache")
	}
	if fake.bindCalls <= bindCallsAfterStep {
		t.Fatalf("bind calls after post-update prepare = %d, want > %d", fake.bindCalls, bindCallsAfterStep)
	}
	mask, err := trainer.prepareMask([]int32{1, 2, 3}, nil)
	if err != nil {
		t.Fatalf("prepare mask: %v", err)
	}
	accelerated, err := trainer.encodeSequence([]int32{1, 2, 3}, mask, trainer.queryRoleIndex(), nil, nil, nil, nil, nil, nil, nil, nil, false)
	if err != nil {
		t.Fatalf("accelerated encode after update: %v", err)
	}
	trainer.forwardMatMul = nil
	host, err := trainer.encodeSequence([]int32{1, 2, 3}, mask, trainer.queryRoleIndex(), nil, nil, nil, nil, nil, nil, nil, nil, false)
	if err != nil {
		t.Fatalf("host encode after update: %v", err)
	}
	assertTensorClose(t, backend.NewTensorF32([]int{1, len(accelerated.pooled)}, accelerated.pooled), []int{1, len(host.pooled)}, host.pooled)
	trainer.forwardMatMul = fake

	bindCalls := fake.bindCalls
	trainer.prepareForwardWeights()
	if fake.bindCalls != bindCalls {
		t.Fatalf("redundant compact bind calls = %d, want %d", fake.bindCalls, bindCalls)
	}
	if trainer.ForwardResidencyStats().BindSkips == 0 {
		t.Fatalf("bind skips = %d, want compact residency reuse recorded", trainer.ForwardResidencyStats().BindSkips)
	}
}

func TestVectorDistillCompactQKVMultiBoundFallbacks(t *testing.T) {
	for _, mode := range []string{"wrong_order", "wrong_shape", "backend_error"} {
		t.Run(mode, func(t *testing.T) {
			trainer := newCompactEmbeddingTrainerForTest(t, 3)
			if trainer.forwardMatMul != nil {
				trainer.forwardMatMul.Close()
			}
			fake := &qkvFallbackMatMulAccelerator{mode: mode}
			trainer.forwardMatMul = fake
			trainer.forwardBackend = eosartifact.BackendCUDA

			mask, err := trainer.prepareMask([]int32{1, 2, 3}, nil)
			if err != nil {
				t.Fatalf("prepare mask: %v", err)
			}
			accelerated, err := trainer.encodeSequence([]int32{1, 2, 3}, mask, trainer.queryRoleIndex(), nil, nil, nil, nil, nil, nil, nil, nil, false)
			if err != nil {
				t.Fatalf("accelerated encode with %s qkv fallback: %v", mode, err)
			}
			if fake.multiBoundRuns == 0 {
				t.Fatalf("multi-bound runs = %d, want attempted QKV coalescing", fake.multiBoundRuns)
			}
			if fake.boundRightRuns <= 3 {
				t.Fatalf("bound-right runs = %d, want fallback single-RHS matmuls after failed QKV coalescing", fake.boundRightRuns)
			}

			trainer.forwardMatMul = nil
			host, err := trainer.encodeSequence([]int32{1, 2, 3}, mask, trainer.queryRoleIndex(), nil, nil, nil, nil, nil, nil, nil, nil, false)
			if err != nil {
				t.Fatalf("host encode: %v", err)
			}
			assertTensorClose(t, backend.NewTensorF32([]int{1, len(accelerated.pooled)}, accelerated.pooled), []int{1, len(host.pooled)}, host.pooled)
		})
	}
}

func TestVectorDistillRelationalLossRoutesSimilarityMatricesThroughMatMul(t *testing.T) {
	trainer := newCompactEmbeddingTrainerForTest(t, 3)
	matmulAccel := &fakeVectorDistillMatMulAccelerator{}
	trainer.forwardMatMul = matmulAccel

	students := [][]float32{
		{0.6, -0.3, 0.2},
		{0.1, 0.8, -0.4},
		{-0.5, 0.2, 0.9},
	}
	teachers := [][]float32{
		{0.2, 0.5, -0.1, 0.4},
		{-0.3, 0.4, 0.6, -0.2},
		{0.7, -0.2, 0.1, 0.3},
	}
	const weight = float32(0.7)

	wantLoss, wantGrads, err := VectorDistillRelationalLossAndGrad(students, teachers, weight)
	if err != nil {
		t.Fatalf("host VectorDistillRelationalLossAndGrad: %v", err)
	}
	gotLoss, gotGrads, err := trainer.vectorDistillRelationalLossAndGrad(students, teachers, weight)
	if err != nil {
		t.Fatalf("accelerated vectorDistillRelationalLossAndGrad: %v", err)
	}
	if matmulAccel.stats.RunCalls != 2 {
		t.Fatalf("matmul run calls = %d, want 2", matmulAccel.stats.RunCalls)
	}
	if gotLoss != wantLoss {
		t.Fatalf("loss = %v, want exact %v", gotLoss, wantLoss)
	}
	if len(gotGrads) != len(wantGrads) {
		t.Fatalf("grad rows = %d, want %d", len(gotGrads), len(wantGrads))
	}
	for i := range gotGrads {
		assertExactFloat32Slice(t, fmt.Sprintf("grad row %d", i), gotGrads[i], wantGrads[i])
	}
}

func BenchmarkVectorDistillCompactForwardResidencyCounters(b *testing.B) {
	for _, tc := range []struct {
		name       string
		disableQKV string
	}{
		{name: "qkv_multi_disabled", disableQKV: "1"},
		{name: "qkv_multi_enabled", disableQKV: ""},
	} {
		b.Run(tc.name, func(b *testing.B) {
			b.Setenv("EOS_TRAIN_DISABLE_QKV_MULTI_BOUND", tc.disableQKV)
			trainer := newCompactEmbeddingTrainerForTest(b, 3)
			if trainer.forwardMatMul != nil {
				trainer.forwardMatMul.Close()
			}
			fake := &countingMatMulAccelerator{}
			trainer.forwardMatMul = fake
			trainer.forwardBackend = eosartifact.BackendCUDA
			trainer.optimizerAccel = nil
			batch := []EmbeddingTokenizedVectorDistillExample{
				vectorDistillTestExample("a", EmbeddingRoleQuery, []int32{1, 2, 3}, []float32{0.2, -0.1, 0.05, 0.3}),
				vectorDistillTestExample("b", EmbeddingRoleDocument, []int32{3, 2, 1}, []float32{-0.1, 0.4, 0.2, -0.05}),
			}
			var proj *vectorDistillProjectionState
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				var err error
				if _, proj, err = trainer.trainVectorDistillBatch(batch, proj, 4, EmbeddingRoleQuery, 0); err != nil {
					b.Fatalf("trainVectorDistillBatch: %v", err)
				}
			}
			b.StopTimer()
			stats := trainer.ForwardResidencyStats()
			steps := float64(b.N)
			if steps == 0 {
				steps = 1
			}
			b.ReportMetric(float64(stats.MatMul.BindCalls)/steps, "bind_calls/step")
			b.ReportMetric(float64(stats.BindSkips)/steps, "bind_skips/step")
			b.ReportMetric(float64(stats.MatMul.RunCalls)/steps, "matmul_rhs_runs/step")
			b.ReportMetric(float64(stats.MatMul.BoundRightCalls)/steps, "bound_right/step")
			b.ReportMetric(float64(stats.MatMul.RunUploadedBytes)/steps, "run_upload_B/step")
			b.ReportMetric(float64(stats.MatMul.RunDownloadedBytes)/steps, "run_download_B/step")
			b.ReportMetric(float64(fake.multiBoundRuns)/steps, "qkv_multi_dispatches/step")
		})
	}
}

// relationalOnlyEncoderGradNorm drives the real production merge path
// (computeVectorDistillBatchGradients) to compute the L2 norm of the RAW
// (pre-batchScale, pre-AdamW) encoder gradient contributed solely by the
// in-batch relational term for the given batch. It isolates the relational
// term's contribution by differencing a call with the relational term active
// against an otherwise-identical call with it disabled (same encoded
// sequences and projection weights, so the pointwise contribution is
// identical and cancels out of the difference). The result is scaled by the
// same batchScale = 1/len(batch) the real trainer applies before AdamW, so it
// reflects the EFFECTIVE per-batch relational contribution actually reaching
// the optimizer — testing at this pre-AdamW level (rather than post-AdamW
// parameter deltas) avoids the confound of AdamW's first-step per-coordinate
// normalization, which saturates to roughly sign(gradient) and would mask
// the magnitude differences this test is designed to catch.
func relationalOnlyEncoderGradNorm(t *testing.T, batch []EmbeddingTokenizedVectorDistillExample, relationalWeight float32, teacherDim int) float64 {
	t.Helper()
	trainer := newCompactEmbeddingTrainerForTest(t, 3)

	inputs := make([]embeddingSequenceInput, len(batch))
	for i, ex := range batch {
		roleIndex, _, err := trainer.vectorDistillRoleIndex(ex.Role, EmbeddingRoleQuery)
		if err != nil {
			t.Fatalf("vectorDistillRoleIndex: %v", err)
		}
		inputs[i] = embeddingSequenceInput{tokens: ex.Tokens, mask: ex.Mask, role: roleIndex, label: fmt.Sprintf("batch %d", i)}
	}
	forward := trainer.prepareForwardWeights()
	if forward == nil || forward.compact == nil {
		t.Fatal("missing compact forward weights")
	}
	encoded, err := trainer.encodeSequenceInputs(inputs, forward, true)
	if err != nil {
		t.Fatalf("encodeSequenceInputs: %v", err)
	}
	defer trainer.releaseEncodedSequences(encoded)

	students := make([][]float32, len(encoded))
	teachers := make([][]float32, len(encoded))
	for i, enc := range encoded {
		students[i] = enc.pooled
		teachers[i] = batch[i].TeacherVector
	}
	_, relationalGrads, err := VectorDistillRelationalLossAndGrad(students, teachers, relationalWeight)
	if err != nil {
		t.Fatalf("VectorDistillRelationalLossAndGrad: %v", err)
	}
	if relationalGrads == nil {
		t.Fatalf("expected non-nil relational grads for batch size %d", len(batch))
	}

	studentDim := len(encoded[0].pooled)
	proj := newVectorDistillProjectionState(studentDim, teacherDim, rand.New(rand.NewSource(42)))

	gradsWith, _, _, err := trainer.computeVectorDistillBatchGradients(batch, encoded, proj, forward, relationalGrads)
	if err != nil {
		t.Fatalf("computeVectorDistillBatchGradients (with relational): %v", err)
	}
	gradsWithout, _, _, err := trainer.computeVectorDistillBatchGradients(batch, encoded, proj, forward, nil)
	if err != nil {
		t.Fatalf("computeVectorDistillBatchGradients (without relational): %v", err)
	}

	withSlices := compactEmbeddingGradSlices(gradsWith)
	withoutSlices := compactEmbeddingGradSlices(gradsWithout)
	if len(withSlices) != len(withoutSlices) {
		t.Fatalf("gradient slice count mismatch: with=%d without=%d", len(withSlices), len(withoutSlices))
	}

	batchScale := float32(1) / float32(len(batch))
	var sumSq float64
	for si := range withSlices {
		a, b := withSlices[si], withoutSlices[si]
		if len(a) != len(b) {
			t.Fatalf("gradient slice %d length mismatch: with=%d without=%d", si, len(a), len(b))
		}
		for k := range a {
			diff := float64(a[k]-b[k]) * float64(batchScale)
			sumSq += diff * diff
		}
	}
	return math.Sqrt(sumSq)
}

func assertExactFloat32Slice(t *testing.T, name string, got, want []float32) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s length = %d, want %d", name, len(got), len(want))
	}
	for i := range got {
		if got[i] != want[i] {
			t.Fatalf("%s[%d] = %v, want exact %v", name, i, got[i], want[i])
		}
	}
}

type fakeVectorDistillMatMulAccelerator struct {
	stats backend.MatMulAcceleratorStats
}

func (a *fakeVectorDistillMatMulAccelerator) Backend() eosartifact.BackendKind {
	return eosartifact.BackendKind("mock")
}

func (a *fakeVectorDistillMatMulAccelerator) RunMatMul(inputs []*backend.Tensor, outputType eosartifact.ValueType) (backend.StepDispatchResult, error) {
	return a.RunMatMulWithTranspose(inputs, outputType, false, false)
}

func (a *fakeVectorDistillMatMulAccelerator) RunMatMulWithTranspose(inputs []*backend.Tensor, outputType eosartifact.ValueType, transposeLeft, transposeRight bool) (backend.StepDispatchResult, error) {
	if len(inputs) != 2 || inputs[0] == nil || inputs[1] == nil {
		return backend.StepDispatchResult{}, fmt.Errorf("fake matmul expects two inputs")
	}
	lhs := inputs[0]
	rhs := inputs[1]
	if len(lhs.Shape) != 2 || len(rhs.Shape) != 2 {
		return backend.StepDispatchResult{}, fmt.Errorf("fake matmul expects rank-2 inputs")
	}
	rows, _, cols, ok := trainerMatMulDims(lhs.Shape[0], lhs.Shape[1], rhs.Shape[0], rhs.Shape[1], transposeLeft, transposeRight)
	if !ok {
		return backend.StepDispatchResult{}, fmt.Errorf("fake matmul shape mismatch")
	}
	out := make([]float32, rows*cols)
	fillHostMatMulTranspose(lhs.F32, lhs.Shape[0], lhs.Shape[1], rhs.F32, rhs.Shape[0], rhs.Shape[1], transposeLeft, transposeRight, out)
	a.stats.RunCalls++
	a.stats.RunUploadedBytes += int64((len(lhs.F32) + len(rhs.F32)) * 4)
	a.stats.RunDownloadedBytes += int64(len(out) * 4)
	return backend.StepDispatchResult{Outputs: []*backend.Tensor{backend.NewTensorF32([]int{rows, cols}, out)}, VariantEntry: "fake_matmul"}, nil
}

func (a *fakeVectorDistillMatMulAccelerator) BindMatrix(name string, tensor *backend.Tensor) error {
	a.stats.BindCalls++
	return nil
}

func (a *fakeVectorDistillMatMulAccelerator) UnbindMatrix(name string) error {
	return nil
}

func (a *fakeVectorDistillMatMulAccelerator) RunMatMulWithBoundLeft(leftName string, rhs *backend.Tensor, outputType eosartifact.ValueType, transposeLeft, transposeRight bool) (backend.StepDispatchResult, error) {
	return backend.StepDispatchResult{}, fmt.Errorf("fake bound-left matmul is not implemented")
}

func (a *fakeVectorDistillMatMulAccelerator) RunMatMulWithBoundRight(lhs *backend.Tensor, rightName string, outputType eosartifact.ValueType, transposeLeft, transposeRight bool) (backend.StepDispatchResult, error) {
	return backend.StepDispatchResult{}, fmt.Errorf("fake bound-right matmul is not implemented")
}

func (a *fakeVectorDistillMatMulAccelerator) Stats() backend.MatMulAcceleratorStats {
	return a.stats
}

func (a *fakeVectorDistillMatMulAccelerator) Close() {}

type qkvFallbackMatMulAccelerator struct {
	countingMatMulAccelerator
	mode string
}

func (a *qkvFallbackMatMulAccelerator) RunMatMulWithBoundRights(lhs *backend.Tensor, rightNames []string, outputType eosartifact.ValueType, transposeLeft, transposeRight bool) ([]backend.StepDispatchResult, error) {
	if a.mode == "backend_error" {
		a.runCalls += len(rightNames)
		a.multiBoundRuns++
		a.boundRightRuns += len(rightNames)
		return nil, fmt.Errorf("forced qkv multi-bound error")
	}
	results, err := a.countingMatMulAccelerator.RunMatMulWithBoundRights(lhs, rightNames, outputType, transposeLeft, transposeRight)
	if err != nil {
		return nil, err
	}
	switch a.mode {
	case "wrong_order":
		if len(results) >= 2 {
			results[0], results[1] = results[1], results[0]
		}
	case "wrong_shape":
		if len(results) > 0 && len(results[0].Outputs) == 1 && results[0].Outputs[0] != nil {
			results[0].Outputs[0] = backend.NewTensorF32([]int{1, 1}, []float32{0})
		}
	}
	return results, nil
}

type fakeVectorDistillOptimizerAccelerator struct {
	stats backend.OptimizerAcceleratorStats
}

func (a *fakeVectorDistillOptimizerAccelerator) Backend() eosartifact.BackendKind {
	return eosartifact.BackendKind("mock")
}

func (a *fakeVectorDistillOptimizerAccelerator) ApplyUpdate(name string, cfg backend.OptimizerUpdateConfig, tensor, mom1, mom2, grad *backend.Tensor) error {
	if tensor == nil || mom1 == nil || mom2 == nil || grad == nil {
		return fmt.Errorf("fake optimizer expects tensors")
	}
	applyOptimizerUpdate(EmbeddingTrainConfig{
		Optimizer:    cfg.Optimizer,
		LearningRate: cfg.LearningRate,
		WeightDecay:  cfg.WeightDecay,
		Beta1:        cfg.Beta1,
		Beta2:        cfg.Beta2,
		Epsilon:      cfg.Epsilon,
	}, cfg.Step, tensor, mom1, mom2, grad.F32, cfg.Scale)
	a.stats.UpdateCalls++
	a.stats.UploadedBytes += int64((len(tensor.F32) + len(mom1.F32) + len(mom2.F32) + len(grad.F32)) * 4)
	a.stats.DownloadedBytes += int64((len(tensor.F32) + len(mom1.F32) + len(mom2.F32)) * 4)
	return nil
}

func (a *fakeVectorDistillOptimizerAccelerator) SyncState(name string, tensor, mom1, mom2 *backend.Tensor, includeMoments bool) error {
	a.stats.SyncCalls++
	return nil
}

func (a *fakeVectorDistillOptimizerAccelerator) Stats() backend.OptimizerAcceleratorStats {
	return a.stats
}

func (a *fakeVectorDistillOptimizerAccelerator) Close() {}
