package eosruntime

import (
	"fmt"
	"math"
	"math/rand"
	"testing"
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
