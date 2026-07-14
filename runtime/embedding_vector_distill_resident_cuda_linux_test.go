//go:build linux && cgo

package eosruntime

import (
	"math"
	"testing"

	_ "m31labs.dev/eos/runtime/backends/cuda"
)

func TestVectorDistillCompactResidentTrainCUDAOneAndTwoStepParity(t *testing.T) {
	trainSet := []EmbeddingTokenizedVectorDistillExample{
		vectorDistillTestExample("a", EmbeddingRoleQuery, []int32{1, 2}, []float32{0.10, -0.20, 0.05}),
		vectorDistillTestExample("b", EmbeddingRoleDocument, []int32{2, 1}, []float32{-0.05, 0.15, 0.20}),
	}
	cfg := EmbeddingTrainRunConfig{
		Epochs:                     1,
		BatchSize:                  1,
		Shuffle:                    false,
		EvalEveryEpoch:             1,
		VectorDistillOptimizerSync: VectorDistillOptimizerSyncImmediate,
		VectorDistillDefaultRole:   EmbeddingRoleQuery,
	}
	run := func(name, gate, syncMode string) (*EmbeddingTrainer, EmbeddingTrainRunSummary) {
		t.Helper()
		t.Setenv(compactResidentTrainEnv, gate)
		t.Setenv(compactPackedForwardEnv, "0")
		trainer := newCompactEmbeddingTrainerForTest(t, 3)
		localCfg := cfg
		localCfg.VectorDistillOptimizerSync = syncMode
		summary, err := trainer.FitVectorDistill(trainSet, localCfg)
		if err != nil {
			t.Fatalf("%s FitVectorDistill: %v", name, err)
		}
		if summary.StepsRun != 2 {
			t.Fatalf("%s steps_run = %d, want 2", name, summary.StepsRun)
		}
		return trainer, summary
	}
	host, hostSummary := run("host_immediate", "0", VectorDistillOptimizerSyncImmediate)
	residentImmediate, residentImmediateSummary := run("resident_immediate", "1", VectorDistillOptimizerSyncImmediate)
	residentDeferred, residentDeferredSummary := run("resident_deferred", "1", VectorDistillOptimizerSyncDeferred)
	if err := host.syncOptimizerStateWithReason(true, "test_compare"); err != nil {
		t.Fatalf("sync host state: %v", err)
	}
	if err := residentImmediate.syncOptimizerStateWithReason(true, "test_compare"); err != nil {
		t.Fatalf("sync resident immediate state: %v", err)
	}
	if err := residentDeferred.syncOptimizerStateWithReason(true, "test_compare"); err != nil {
		t.Fatalf("sync resident deferred state: %v", err)
	}

	assertCompactTrainerStateClose(t, "host_vs_resident_immediate", host, residentImmediate, 3e-8, 1e-6)
	assertCompactTrainerStateClose(t, "host_vs_resident_deferred", host, residentDeferred, 3e-8, 1e-6)
	assertCompactTrainerStateClose(t, "resident_immediate_vs_deferred", residentImmediate, residentDeferred, 0, 0)
	if residentImmediateSummary.DeltaProfile.CompactTrain == nil || residentImmediateSummary.DeltaProfile.CompactTrain.BackwardCalls != 2 || residentImmediateSummary.DeltaProfile.CompactTrain.LiveHandles != 0 {
		t.Fatalf("resident immediate compact train profile = %+v", residentImmediateSummary.DeltaProfile.CompactTrain)
	}
	if residentImmediateSummary.DeltaProfile.CompactForward != nil && (residentImmediateSummary.DeltaProfile.CompactForward.PackedDownloads != 0 || residentImmediateSummary.DeltaProfile.CompactForward.PackedBytes != 0) {
		t.Fatalf("resident immediate packed counters = %+v", residentImmediateSummary.DeltaProfile.CompactForward)
	}
	if residentImmediateSummary.DeltaProfile.Optimizer.ResidentGradUpdateCalls == 0 || residentImmediateSummary.DeltaProfile.Optimizer.ResidentGradUploadBytesAvoided == 0 {
		t.Fatalf("resident immediate optimizer profile = %+v", residentImmediateSummary.DeltaProfile.Optimizer)
	}
	if residentImmediateSummary.DeltaProfile.Optimizer.UploadedBytes >= hostSummary.DeltaProfile.Optimizer.UploadedBytes {
		t.Fatalf("resident immediate optimizer uploaded bytes = %d, host = %d", residentImmediateSummary.DeltaProfile.Optimizer.UploadedBytes, hostSummary.DeltaProfile.Optimizer.UploadedBytes)
	}
	if residentDeferredSummary.DeltaProfile.CompactTrain == nil || residentDeferredSummary.DeltaProfile.CompactTrain.LiveHandles != 0 || residentDeferredSummary.DeltaProfile.CompactTrain.FallbackOrUnhandled != 0 {
		t.Fatalf("resident deferred compact train profile = %+v", residentDeferredSummary.DeltaProfile.CompactTrain)
	}
	t.Logf("A host/immediate optimizer_uploaded=%d optimizer_resident_grad_updates=%d", hostSummary.DeltaProfile.Optimizer.UploadedBytes, hostSummary.DeltaProfile.Optimizer.ResidentGradUpdateCalls)
	t.Logf("B resident/immediate compact_train=%+v optimizer=%+v", *residentImmediateSummary.DeltaProfile.CompactTrain, residentImmediateSummary.DeltaProfile.Optimizer)
	t.Logf("C resident/deferred compact_train=%+v optimizer=%+v", *residentDeferredSummary.DeltaProfile.CompactTrain, residentDeferredSummary.DeltaProfile.Optimizer)
}

func TestVectorDistillCompactResidentTrainCUDAVaryingTWholeBatchParity(t *testing.T) {
	batch := []EmbeddingTokenizedVectorDistillExample{
		vectorDistillTestExample("t2", EmbeddingRoleDocument, []int32{3, 4}, []float32{-0.10, 0.40, 0.20, -0.05}),
		vectorDistillTestExample("t3", EmbeddingRoleQuery, []int32{1, 2, 3}, []float32{0.20, -0.10, 0.05, 0.30}),
		vectorDistillTestExample("t4", EmbeddingRoleRaw, []int32{4, 3, 2, 1}, []float32{0.15, -0.25, 0.35, -0.05}),
		vectorDistillTestExample("t3-alias", EmbeddingRoleQuery, []int32{1, 2, 3}, []float32{0.30, 0.10, -0.20, 0.05}),
	}
	newProjection := func() *vectorDistillProjectionState {
		return &vectorDistillProjectionState{
			W:    []float32{0.2, -0.1, 0.05, 0.3, -0.4, 0.25, 0.15, -0.05, 0.35, -0.2, 0.1, 0.45},
			Mom1: make([]float32, 12), Mom2: make([]float32, 12), InputDim: 3, OutDim: 4,
		}
	}
	run := func(name, gate string) (*EmbeddingTrainer, *vectorDistillProjectionState, EmbeddingTrainProfile) {
		t.Helper()
		t.Setenv(compactResidentTrainEnv, gate)
		t.Setenv(compactPackedForwardEnv, "0")
		trainer := newCompactEmbeddingTrainerForTest(t, 3)
		start := trainer.TrainProfile()
		_, proj, err := trainer.trainVectorDistillBatch(batch, newProjection(), 4, EmbeddingRoleQuery, 0.5)
		if err != nil {
			trainer.Close()
			t.Fatalf("%s varying-T batch: %v", name, err)
		}
		return trainer, proj, diffTrainProfile(start, trainer.TrainProfile())
	}
	host, hostProj, _ := run("host", "0")
	defer host.Close()
	resident, residentProj, profile := run("resident", "1")
	defer resident.Close()
	if err := host.syncOptimizerStateWithReason(true, "test_compare"); err != nil {
		t.Fatalf("sync host state: %v", err)
	}
	if err := resident.syncOptimizerStateWithReason(true, "test_compare"); err != nil {
		t.Fatalf("sync resident state: %v", err)
	}
	assertCompactTrainerStateClose(t, "varying_t_host_vs_resident", host, resident, 1e-6, 1e-6)
	assertTensorMaxAbs(t, "varying_t projection", hostProj.W, residentProj.W, 1e-6, 1e-6)
	assertTensorMaxAbs(t, "varying_t projection mom1", hostProj.Mom1, residentProj.Mom1, 1e-6, 1e-6)
	assertTensorMaxAbs(t, "varying_t projection mom2", hostProj.Mom2, residentProj.Mom2, 1e-6, 1e-6)
	stats := profile.CompactTrain
	if stats == nil {
		t.Fatal("varying-T resident compact train stats are nil")
	}
	const expectedPooledBytes = int64(3 * 3 * 4)
	if stats.ForwardCalls != 3 || stats.BackwardCalls != 3 || stats.HandlesCreated != 3 || stats.HandlesReleased != 3 || stats.LiveHandles != 0 {
		t.Fatalf("varying-T handle counters = %+v", *stats)
	}
	if stats.PooledDownloadedBytes != expectedPooledBytes || stats.GradPooledUploadedBytes != expectedPooledBytes {
		t.Fatalf("varying-T pooled D2H/H2D = %d/%d, want %d/%d", stats.PooledDownloadedBytes, stats.GradPooledUploadedBytes, expectedPooledBytes, expectedPooledBytes)
	}
	if stats.FallbackOrUnhandled != 0 {
		t.Fatalf("varying-T fallback/unhandled = %d, want 0", stats.FallbackOrUnhandled)
	}
	if profile.CompactForward != nil && (profile.CompactForward.PackedDownloads != 0 || profile.CompactForward.PackedBytes != 0) {
		t.Fatalf("varying-T packed-state D2H = %+v", *profile.CompactForward)
	}
	t.Logf("varying-T CUDA exact buckets=2,3,4 aliases=1 compact_train=%+v", *stats)
}

func assertCompactTrainerStateClose(t *testing.T, label string, left, right *EmbeddingTrainer, target, hard float32) {
	t.Helper()
	leftItems := compactTrainStateOptimizerItems(left.compactState)
	rightItems := compactTrainStateOptimizerItems(right.compactState)
	if len(leftItems) != len(rightItems) {
		t.Fatalf("%s item count = %d/%d", label, len(leftItems), len(rightItems))
	}
	for i := range leftItems {
		l, r := leftItems[i], rightItems[i]
		if l.Name != r.Name {
			t.Fatalf("%s item %d name = %q/%q", label, i, l.Name, r.Name)
		}
		assertTensorMaxAbs(t, label+" "+l.Name+" tensor", l.Tensor.F32, r.Tensor.F32, target, hard)
		assertTensorMaxAbs(t, label+" "+l.Name+" mom1", l.Moment1.F32, r.Moment1.F32, target, hard)
		assertTensorMaxAbs(t, label+" "+l.Name+" mom2", l.Moment2.F32, r.Moment2.F32, target, hard)
	}
}

func assertTensorMaxAbs(t *testing.T, label string, left, right []float32, target, hard float32) {
	t.Helper()
	if len(left) != len(right) {
		t.Fatalf("%s len = %d/%d", label, len(left), len(right))
	}
	maxAbs := float32(0)
	for i := range left {
		d := float32(math.Abs(float64(left[i] - right[i])))
		if d > maxAbs {
			maxAbs = d
		}
	}
	if hard == 0 {
		if maxAbs != 0 {
			t.Fatalf("%s max_abs = %.9g, want exact 0", label, maxAbs)
		}
		return
	}
	if maxAbs > hard {
		t.Fatalf("%s max_abs = %.9g, hard %.9g target %.9g", label, maxAbs, hard, target)
	}
	if maxAbs > target {
		t.Logf("%s max_abs = %.9g exceeds target %.9g but within hard %.9g", label, maxAbs, target, hard)
	}
}
