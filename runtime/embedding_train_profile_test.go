package eosruntime

import (
	"path/filepath"
	"testing"

	eosartifact "m31labs.dev/eos/artifact/eos"
	"m31labs.dev/eos/runtime/backend"
)

func TestDefaultEmbeddingTrainProfilePath(t *testing.T) {
	got := DefaultEmbeddingTrainProfilePath("/tmp/tiny_train_embed_q8.mll")
	if want := "/tmp/tiny_train_embed_q8.train-profile.mll"; got != want {
		t.Fatalf("training profile path = %q, want %q", got, want)
	}
}

func TestEmbeddingTrainProfileRoundTrip(t *testing.T) {
	want := EmbeddingTrainProfile{
		Version:           EmbeddingTrainProfileVersion,
		Step:              7,
		ForwardBackend:    "cuda",
		OptimizerBackend:  "cuda",
		ActivationBackend: "cuda",
		ForwardResidency: EmbeddingForwardResidencyStats{
			BindSkips: 3,
			MatMul: backend.MatMulAcceleratorStats{
				BindCalls:          8,
				UploadedBytes:      512,
				QuantizePasses:     5,
				QuantizedBytes:     320,
				BindNanos:          1000,
				QuantizeNanos:      500,
				BoundMatrices:      2,
				RunCalls:           13,
				BoundLeftCalls:     4,
				BoundRightCalls:    6,
				RunUploadedBytes:   2048,
				RunDownloadedBytes: 1536,
				RunNanos:           7000,
			},
		},
		VectorDistillPhases: EmbeddingVectorDistillPhaseTimers{
			EncodeNanos:         11,
			ProjectionLossNanos: 22,
			BackwardNanos:       33,
			OptimizerNanos:      44,
			EncodeCalls:         1,
			ProjectionLossCalls: 2,
			BackwardCalls:       2,
			OptimizerCalls:      1,
		},
		Optimizer: backend.OptimizerAcceleratorStats{
			LogicalSteps:                    4,
			TensorUpdateCalls:               13,
			UpdateCalls:                     4,
			ResidentGradBatchCalls:          2,
			ResidentGradBatchKernelLaunches: 20,
			ResidentGradBatchKernelSyncs:    2,
			DeferredSyncUpdates:             9,
			SyncCalls:                       2,
			ForcedSyncCalls:                 1,
			LastForcedSyncReason:            "restore-best-test",
			UploadedBytes:                   1024,
			DownloadedBytes:                 512,
			UpdateNanos:                     100,
			SyncNanos:                       40,
			ResidentParams:                  3,
		},
		Activation: backend.ActivationAcceleratorStats{
			BindCalls:              4,
			GELUBackwardCalls:      3,
			SoftmaxBackwardCalls:   2,
			LayerNormBackwardCalls: 1,
			UploadedBytes:          768,
			DownloadedBytes:        256,
			RunNanos:               80,
			BoundTensors:           2,
		},
	}
	path := filepath.Join(t.TempDir(), "tiny.train-profile.mll")
	if err := want.WriteFile(path); err != nil {
		t.Fatalf("write training profile: %v", err)
	}
	got, err := ReadEmbeddingTrainProfileFile(path)
	if err != nil {
		t.Fatalf("read training profile: %v", err)
	}
	if got.Version != want.Version {
		t.Fatalf("version = %q, want %q", got.Version, want.Version)
	}
	if got.Step != want.Step {
		t.Fatalf("step = %d, want %d", got.Step, want.Step)
	}
	if got.ForwardResidency.BindSkips != want.ForwardResidency.BindSkips {
		t.Fatalf("bind skips = %d, want %d", got.ForwardResidency.BindSkips, want.ForwardResidency.BindSkips)
	}
	if got.ForwardResidency.MatMul.BindCalls != want.ForwardResidency.MatMul.BindCalls {
		t.Fatalf("bind calls = %d, want %d", got.ForwardResidency.MatMul.BindCalls, want.ForwardResidency.MatMul.BindCalls)
	}
	if got.ForwardResidency.MatMul.QuantizePasses != want.ForwardResidency.MatMul.QuantizePasses {
		t.Fatalf("quantize passes = %d, want %d", got.ForwardResidency.MatMul.QuantizePasses, want.ForwardResidency.MatMul.QuantizePasses)
	}
	if got.ForwardResidency.MatMul.RunCalls != want.ForwardResidency.MatMul.RunCalls {
		t.Fatalf("run calls = %d, want %d", got.ForwardResidency.MatMul.RunCalls, want.ForwardResidency.MatMul.RunCalls)
	}
	if got.ForwardResidency.MatMul.RunUploadedBytes != want.ForwardResidency.MatMul.RunUploadedBytes {
		t.Fatalf("run uploaded bytes = %d, want %d", got.ForwardResidency.MatMul.RunUploadedBytes, want.ForwardResidency.MatMul.RunUploadedBytes)
	}
	assertVectorDistillProfilePhases(t, got.VectorDistillPhases, want.VectorDistillPhases)
	if got.Optimizer.UpdateCalls != want.Optimizer.UpdateCalls {
		t.Fatalf("optimizer update calls = %d, want %d", got.Optimizer.UpdateCalls, want.Optimizer.UpdateCalls)
	}
	if got.Optimizer.LogicalSteps != want.Optimizer.LogicalSteps {
		t.Fatalf("optimizer logical steps = %d, want %d", got.Optimizer.LogicalSteps, want.Optimizer.LogicalSteps)
	}
	if got.Optimizer.TensorUpdateCalls != want.Optimizer.TensorUpdateCalls {
		t.Fatalf("optimizer tensor update calls = %d, want %d", got.Optimizer.TensorUpdateCalls, want.Optimizer.TensorUpdateCalls)
	}
	if got.Optimizer.ResidentGradBatchCalls != want.Optimizer.ResidentGradBatchCalls {
		t.Fatalf("optimizer resident-gradient batch calls = %d, want %d", got.Optimizer.ResidentGradBatchCalls, want.Optimizer.ResidentGradBatchCalls)
	}
	if got.Optimizer.ResidentGradBatchKernelLaunches != want.Optimizer.ResidentGradBatchKernelLaunches {
		t.Fatalf("optimizer resident-gradient batch kernel launches = %d, want %d", got.Optimizer.ResidentGradBatchKernelLaunches, want.Optimizer.ResidentGradBatchKernelLaunches)
	}
	if got.Optimizer.ResidentGradBatchKernelSyncs != want.Optimizer.ResidentGradBatchKernelSyncs {
		t.Fatalf("optimizer resident-gradient batch kernel synchronizations = %d, want %d", got.Optimizer.ResidentGradBatchKernelSyncs, want.Optimizer.ResidentGradBatchKernelSyncs)
	}
	if got.Optimizer.DeferredSyncUpdates != want.Optimizer.DeferredSyncUpdates {
		t.Fatalf("optimizer deferred sync updates = %d, want %d", got.Optimizer.DeferredSyncUpdates, want.Optimizer.DeferredSyncUpdates)
	}
	if got.Optimizer.ForcedSyncCalls != want.Optimizer.ForcedSyncCalls {
		t.Fatalf("optimizer forced sync calls = %d, want %d", got.Optimizer.ForcedSyncCalls, want.Optimizer.ForcedSyncCalls)
	}
	if got.Optimizer.LastForcedSyncReason != want.Optimizer.LastForcedSyncReason {
		t.Fatalf("optimizer last forced sync reason = %q, want %q", got.Optimizer.LastForcedSyncReason, want.Optimizer.LastForcedSyncReason)
	}
	if got.Activation.GELUBackwardCalls != want.Activation.GELUBackwardCalls {
		t.Fatalf("activation gelu calls = %d, want %d", got.Activation.GELUBackwardCalls, want.Activation.GELUBackwardCalls)
	}
	if got.Activation.BindCalls != want.Activation.BindCalls {
		t.Fatalf("activation bind calls = %d, want %d", got.Activation.BindCalls, want.Activation.BindCalls)
	}
}

func TestEmbeddingTrainerFitCapturesProfileDelta(t *testing.T) {
	trainer := newTinyTrainableAttentionEmbeddingTrainer(t, 0.05)
	if trainer.forwardMatMul != nil {
		trainer.forwardMatMul.Close()
	}
	fake := &countingMatMulAccelerator{}
	trainer.forwardMatMul = fake
	trainer.forwardBackend = eosartifact.BackendCUDA

	summary, err := trainer.FitContrastive(tinyEmbeddingContrastiveDataset(), tinyEmbeddingContrastiveDataset(), EmbeddingTrainRunConfig{
		Epochs:      2,
		BatchSize:   2,
		Shuffle:     false,
		Seed:        1,
		RestoreBest: true,
	})
	if err != nil {
		t.Fatalf("fit contrastive: %v", err)
	}
	if summary.StartProfile.Version != EmbeddingTrainProfileVersion {
		t.Fatalf("start profile version = %q, want %q", summary.StartProfile.Version, EmbeddingTrainProfileVersion)
	}
	if summary.EndProfile.Version != EmbeddingTrainProfileVersion {
		t.Fatalf("end profile version = %q, want %q", summary.EndProfile.Version, EmbeddingTrainProfileVersion)
	}
	if summary.DeltaProfile.Step != summary.StepsCompleted {
		t.Fatalf("delta profile step = %d, want %d", summary.DeltaProfile.Step, summary.StepsCompleted)
	}
	if summary.DeltaProfile.ForwardResidency.MatMul.BindCalls <= 0 {
		t.Fatalf("delta bind calls = %d, want positive", summary.DeltaProfile.ForwardResidency.MatMul.BindCalls)
	}
	if summary.EndProfile.ForwardResidency.MatMul.BindCalls < summary.StartProfile.ForwardResidency.MatMul.BindCalls {
		t.Fatalf("end profile bind calls = %d, want at least start count %d", summary.EndProfile.ForwardResidency.MatMul.BindCalls, summary.StartProfile.ForwardResidency.MatMul.BindCalls)
	}
	if summary.EndProfile.Optimizer.UpdateCalls < summary.StartProfile.Optimizer.UpdateCalls {
		t.Fatalf("end optimizer update calls = %d, want at least start count %d", summary.EndProfile.Optimizer.UpdateCalls, summary.StartProfile.Optimizer.UpdateCalls)
	}
	if summary.EndProfile.Activation.SoftmaxBackwardCalls < summary.StartProfile.Activation.SoftmaxBackwardCalls {
		t.Fatalf("end activation softmax calls = %d, want at least start count %d", summary.EndProfile.Activation.SoftmaxBackwardCalls, summary.StartProfile.Activation.SoftmaxBackwardCalls)
	}
	if summary.EndProfile.Activation.BindCalls < summary.StartProfile.Activation.BindCalls {
		t.Fatalf("end activation bind calls = %d, want at least start count %d", summary.EndProfile.Activation.BindCalls, summary.StartProfile.Activation.BindCalls)
	}
	if summary.DeltaProfile.Optimizer.UpdateCalls != summary.EndProfile.Optimizer.UpdateCalls-summary.StartProfile.Optimizer.UpdateCalls {
		t.Fatalf("optimizer delta update calls = %d, want %d", summary.DeltaProfile.Optimizer.UpdateCalls, summary.EndProfile.Optimizer.UpdateCalls-summary.StartProfile.Optimizer.UpdateCalls)
	}
	if summary.DeltaProfile.Activation.SoftmaxBackwardCalls != summary.EndProfile.Activation.SoftmaxBackwardCalls-summary.StartProfile.Activation.SoftmaxBackwardCalls {
		t.Fatalf("activation delta softmax calls = %d, want %d", summary.DeltaProfile.Activation.SoftmaxBackwardCalls, summary.EndProfile.Activation.SoftmaxBackwardCalls-summary.StartProfile.Activation.SoftmaxBackwardCalls)
	}
	if summary.DeltaProfile.Activation.BindCalls != summary.EndProfile.Activation.BindCalls-summary.StartProfile.Activation.BindCalls {
		t.Fatalf("activation delta bind calls = %d, want %d", summary.DeltaProfile.Activation.BindCalls, summary.EndProfile.Activation.BindCalls-summary.StartProfile.Activation.BindCalls)
	}
	if fake.bindCalls <= 0 {
		t.Fatalf("fake bind calls = %d, want positive", fake.bindCalls)
	}
}

func TestTrainProfileOptimizerCounterDeltaMergeAndApply(t *testing.T) {
	left := EmbeddingTrainProfile{
		Version:          EmbeddingTrainProfileVersion,
		OptimizerBackend: eosartifact.BackendCUDA,
		VectorDistillPhases: EmbeddingVectorDistillPhaseTimers{
			EncodeNanos:         10,
			ProjectionLossNanos: 20,
			BackwardNanos:       30,
			OptimizerNanos:      40,
			EncodeCalls:         1,
			ProjectionLossCalls: 2,
			BackwardCalls:       2,
			OptimizerCalls:      1,
		},
		Optimizer: backend.OptimizerAcceleratorStats{
			LogicalSteps:                    2,
			TensorUpdateCalls:               9,
			UpdateCalls:                     2,
			ResidentGradBatchCalls:          2,
			ResidentGradBatchKernelLaunches: 30,
			ResidentGradBatchKernelSyncs:    2,
			DeferredSyncUpdates:             7,
			SyncCalls:                       1,
			ForcedSyncCalls:                 1,
			LastForcedSyncReason:            "left-pressure",
			ResidentParams:                  3,
		},
	}
	right := EmbeddingTrainProfile{
		Version:          EmbeddingTrainProfileVersion,
		OptimizerBackend: eosartifact.BackendCUDA,
		VectorDistillPhases: EmbeddingVectorDistillPhaseTimers{
			EncodeNanos:         100,
			ProjectionLossNanos: 200,
			BackwardNanos:       300,
			OptimizerNanos:      400,
			EncodeCalls:         3,
			ProjectionLossCalls: 6,
			BackwardCalls:       6,
			OptimizerCalls:      3,
		},
		Optimizer: backend.OptimizerAcceleratorStats{
			LogicalSteps:                    3,
			TensorUpdateCalls:               11,
			UpdateCalls:                     3,
			ResidentGradBatchCalls:          3,
			ResidentGradBatchKernelLaunches: 45,
			ResidentGradBatchKernelSyncs:    3,
			DeferredSyncUpdates:             5,
			SyncCalls:                       2,
			ForcedSyncCalls:                 2,
			LastForcedSyncReason:            "right-capacity",
			ResidentParams:                  4,
		},
	}

	merged := addTrainProfileDelta(left, right)
	assertOptimizerProfileCounters(t, merged.Optimizer, backend.OptimizerAcceleratorStats{
		LogicalSteps:                    5,
		TensorUpdateCalls:               20,
		UpdateCalls:                     5,
		ResidentGradBatchCalls:          5,
		ResidentGradBatchKernelLaunches: 75,
		ResidentGradBatchKernelSyncs:    5,
		DeferredSyncUpdates:             12,
		SyncCalls:                       3,
		ForcedSyncCalls:                 3,
		LastForcedSyncReason:            "right-capacity",
		ResidentParams:                  4,
	})
	assertVectorDistillProfilePhases(t, merged.VectorDistillPhases, EmbeddingVectorDistillPhaseTimers{
		EncodeNanos:         110,
		ProjectionLossNanos: 220,
		BackwardNanos:       330,
		OptimizerNanos:      440,
		EncodeCalls:         4,
		ProjectionLossCalls: 8,
		BackwardCalls:       8,
		OptimizerCalls:      4,
	})

	base := EmbeddingTrainProfile{
		Version:          EmbeddingTrainProfileVersion,
		OptimizerBackend: eosartifact.BackendKind("host"),
		VectorDistillPhases: EmbeddingVectorDistillPhaseTimers{
			EncodeNanos:         1000,
			ProjectionLossNanos: 2000,
			BackwardNanos:       3000,
			OptimizerNanos:      4000,
			EncodeCalls:         10,
			ProjectionLossCalls: 20,
			BackwardCalls:       20,
			OptimizerCalls:      10,
		},
		Optimizer: backend.OptimizerAcceleratorStats{
			LogicalSteps:                    10,
			TensorUpdateCalls:               100,
			UpdateCalls:                     10,
			ResidentGradBatchCalls:          10,
			ResidentGradBatchKernelLaunches: 150,
			ResidentGradBatchKernelSyncs:    10,
			DeferredSyncUpdates:             50,
			SyncCalls:                       10,
			ForcedSyncCalls:                 4,
			LastForcedSyncReason:            "base-sync",
			ResidentParams:                  8,
		},
	}
	applied := applyTrainProfileDelta(base, right)
	assertOptimizerProfileCounters(t, applied.Optimizer, backend.OptimizerAcceleratorStats{
		LogicalSteps:                    13,
		TensorUpdateCalls:               111,
		UpdateCalls:                     13,
		ResidentGradBatchCalls:          13,
		ResidentGradBatchKernelLaunches: 195,
		ResidentGradBatchKernelSyncs:    13,
		DeferredSyncUpdates:             55,
		SyncCalls:                       12,
		ForcedSyncCalls:                 6,
		LastForcedSyncReason:            "right-capacity",
		ResidentParams:                  4,
	})
	assertVectorDistillProfilePhases(t, applied.VectorDistillPhases, EmbeddingVectorDistillPhaseTimers{
		EncodeNanos:         1100,
		ProjectionLossNanos: 2200,
		BackwardNanos:       3300,
		OptimizerNanos:      4400,
		EncodeCalls:         13,
		ProjectionLossCalls: 26,
		BackwardCalls:       26,
		OptimizerCalls:      13,
	})
}

func TestTrainProfileRestoreBestMergePreservesOptimizerCounters(t *testing.T) {
	start := EmbeddingTrainProfile{
		Version: EmbeddingTrainProfileVersion,
		Step:    10,
		Optimizer: backend.OptimizerAcceleratorStats{
			LogicalSteps:                    10,
			TensorUpdateCalls:               100,
			UpdateCalls:                     10,
			ResidentGradBatchCalls:          10,
			ResidentGradBatchKernelLaunches: 150,
			ResidentGradBatchKernelSyncs:    10,
			DeferredSyncUpdates:             80,
			SyncCalls:                       2,
			ForcedSyncCalls:                 1,
			LastForcedSyncReason:            "startup",
			ResidentParams:                  2,
		},
	}
	preRestoreEnd := EmbeddingTrainProfile{
		Version: EmbeddingTrainProfileVersion,
		Step:    14,
		Optimizer: backend.OptimizerAcceleratorStats{
			LogicalSteps:                    14,
			TensorUpdateCalls:               140,
			UpdateCalls:                     14,
			ResidentGradBatchCalls:          14,
			ResidentGradBatchKernelLaunches: 210,
			ResidentGradBatchKernelSyncs:    14,
			DeferredSyncUpdates:             120,
			SyncCalls:                       5,
			ForcedSyncCalls:                 2,
			LastForcedSyncReason:            "pre-restore-pressure",
			ResidentParams:                  4,
		},
	}
	restoreStart := EmbeddingTrainProfile{
		Version: EmbeddingTrainProfileVersion,
		Step:    12,
		Optimizer: backend.OptimizerAcceleratorStats{
			LogicalSteps:                    12,
			TensorUpdateCalls:               120,
			UpdateCalls:                     12,
			ResidentGradBatchCalls:          12,
			ResidentGradBatchKernelLaunches: 180,
			ResidentGradBatchKernelSyncs:    12,
			DeferredSyncUpdates:             96,
			SyncCalls:                       3,
			ForcedSyncCalls:                 1,
			LastForcedSyncReason:            "startup",
			ResidentParams:                  3,
		},
	}
	final := EmbeddingTrainProfile{
		Version: EmbeddingTrainProfileVersion,
		Step:    13,
		Optimizer: backend.OptimizerAcceleratorStats{
			LogicalSteps:                    13,
			TensorUpdateCalls:               133,
			UpdateCalls:                     13,
			ResidentGradBatchCalls:          13,
			ResidentGradBatchKernelLaunches: 195,
			ResidentGradBatchKernelSyncs:    13,
			DeferredSyncUpdates:             108,
			SyncCalls:                       4,
			ForcedSyncCalls:                 2,
			LastForcedSyncReason:            "final-eval-sync",
			ResidentParams:                  5,
		},
	}

	preRestoreDelta := diffTrainProfile(start, preRestoreEnd)
	postRestoreDelta := diffTrainProfile(restoreStart, final)
	mergedDelta := addTrainProfileDelta(preRestoreDelta, postRestoreDelta)
	endProfile := applyTrainProfileDelta(preRestoreEnd, postRestoreDelta)

	assertOptimizerProfileCounters(t, mergedDelta.Optimizer, backend.OptimizerAcceleratorStats{
		LogicalSteps:                    5,
		TensorUpdateCalls:               53,
		UpdateCalls:                     5,
		ResidentGradBatchCalls:          5,
		ResidentGradBatchKernelLaunches: 75,
		ResidentGradBatchKernelSyncs:    5,
		DeferredSyncUpdates:             52,
		SyncCalls:                       4,
		ForcedSyncCalls:                 2,
		LastForcedSyncReason:            "final-eval-sync",
		ResidentParams:                  5,
	})
	assertOptimizerProfileCounters(t, endProfile.Optimizer, backend.OptimizerAcceleratorStats{
		LogicalSteps:                    15,
		TensorUpdateCalls:               153,
		UpdateCalls:                     15,
		ResidentGradBatchCalls:          15,
		ResidentGradBatchKernelLaunches: 225,
		ResidentGradBatchKernelSyncs:    15,
		DeferredSyncUpdates:             132,
		SyncCalls:                       6,
		ForcedSyncCalls:                 3,
		LastForcedSyncReason:            "final-eval-sync",
		ResidentParams:                  5,
	})
}

func TestTrainProfileOptimizerCounterActivity(t *testing.T) {
	cases := []struct {
		name  string
		stats backend.OptimizerAcceleratorStats
	}{
		{name: "logical steps", stats: backend.OptimizerAcceleratorStats{LogicalSteps: 1}},
		{name: "tensor update calls", stats: backend.OptimizerAcceleratorStats{TensorUpdateCalls: 1}},
		{name: "resident-gradient batch calls", stats: backend.OptimizerAcceleratorStats{ResidentGradBatchCalls: 1}},
		{name: "resident-gradient batch kernel launches", stats: backend.OptimizerAcceleratorStats{ResidentGradBatchKernelLaunches: 1}},
		{name: "resident-gradient batch kernel synchronizations", stats: backend.OptimizerAcceleratorStats{ResidentGradBatchKernelSyncs: 1}},
		{name: "deferred sync updates", stats: backend.OptimizerAcceleratorStats{DeferredSyncUpdates: 1}},
		{name: "forced sync calls", stats: backend.OptimizerAcceleratorStats{ForcedSyncCalls: 1}},
		{name: "last forced sync reason", stats: backend.OptimizerAcceleratorStats{LastForcedSyncReason: "memory-pressure"}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if !hasTrainProfileActivity(EmbeddingTrainProfile{Optimizer: tc.stats}) {
				t.Fatal("expected optimizer counter activity")
			}
		})
	}
}

func TestDiffCompactTrainStatsExcludesWarmupAllocationsKeepsMeasuredReuse(t *testing.T) {
	start := &backend.CompactTrainAcceleratorStats{
		ArenaAllocations:            1,
		ArenaReuseHits:              1,
		GradientAllocations:         1,
		GradientReuseHits:           1,
		LiveHandles:                 2,
		ActivationArenaBytes:        4096,
		WorkspaceArenaBytes:         2048,
		LastShape:                   backend.CompactForwardShape{Batch: 1, Tokens: 4, ModelDim: 8},
		LastForwardLaunches:         3,
		LastBackwardLaunches:        4,
		LastForwardCublasGemmCalls:  1,
		LastBackwardCublasGemmCalls: 2,
		LastForwardSyncs:            1,
		LastBackwardSyncs:           1,
	}
	end := &backend.CompactTrainAcceleratorStats{
		ArenaAllocations:            1,
		ArenaReuseHits:              2,
		GradientAllocations:         1,
		GradientReuseHits:           2,
		LiveHandles:                 1,
		ActivationArenaBytes:        8192,
		WorkspaceArenaBytes:         4096,
		LastShape:                   backend.CompactForwardShape{Batch: 1, Tokens: 4, ModelDim: 8, FFNDim: 16},
		LastForwardLaunches:         5,
		LastBackwardLaunches:        6,
		LastForwardCublasGemmCalls:  3,
		LastBackwardCublasGemmCalls: 4,
		LastForwardSyncs:            2,
		LastBackwardSyncs:           2,
	}

	got := diffCompactTrainStats(start, end)
	if got == nil {
		t.Fatal("compact train diff = nil, want stats")
	}
	if got.ArenaAllocations != 0 || got.GradientAllocations != 0 {
		t.Fatalf("warm-up allocations = arena %d, gradient %d; want both zero", got.ArenaAllocations, got.GradientAllocations)
	}
	if got.ArenaReuseHits != 1 || got.GradientReuseHits != 1 {
		t.Fatalf("measured reuse = arena %d, gradient %d; want both one", got.ArenaReuseHits, got.GradientReuseHits)
	}
	if got.LiveHandles != end.LiveHandles {
		t.Fatalf("live handles = %d, want end snapshot %d", got.LiveHandles, end.LiveHandles)
	}
	if got.ActivationArenaBytes != end.ActivationArenaBytes || got.WorkspaceArenaBytes != end.WorkspaceArenaBytes {
		t.Fatalf("arena bytes = activation %d/workspace %d, want end snapshot %d/%d", got.ActivationArenaBytes, got.WorkspaceArenaBytes, end.ActivationArenaBytes, end.WorkspaceArenaBytes)
	}
	if got.LastShape != end.LastShape || got.LastForwardLaunches != end.LastForwardLaunches || got.LastBackwardLaunches != end.LastBackwardLaunches || got.LastForwardCublasGemmCalls != end.LastForwardCublasGemmCalls || got.LastBackwardCublasGemmCalls != end.LastBackwardCublasGemmCalls || got.LastForwardSyncs != end.LastForwardSyncs || got.LastBackwardSyncs != end.LastBackwardSyncs {
		t.Fatalf("last compact train snapshots = %+v, want end snapshots %+v", got, end)
	}
}

func TestCompactTrainProfileRestoreMergePreservesPoolDeltasAndSnapshots(t *testing.T) {
	start := EmbeddingTrainProfile{
		Version: EmbeddingTrainProfileVersion,
		CompactTrain: &backend.CompactTrainAcceleratorStats{
			LiveHandles:          2,
			ArenaReuseHits:       1,
			ArenaAllocations:     1,
			GradientReuseHits:    1,
			GradientAllocations:  1,
			ResidentGradBytes:    100,
			ActivationArenaBytes: 1000,
			WorkspaceArenaBytes:  2000,
		},
	}
	preRestoreEnd := EmbeddingTrainProfile{
		Version: EmbeddingTrainProfileVersion,
		CompactTrain: &backend.CompactTrainAcceleratorStats{
			LiveHandles:          1,
			ArenaReuseHits:       3,
			ArenaAllocations:     2,
			GradientReuseHits:    4,
			GradientAllocations:  2,
			ResidentGradBytes:    150,
			ActivationArenaBytes: 1100,
			WorkspaceArenaBytes:  2200,
		},
	}
	restoreStart := EmbeddingTrainProfile{
		Version: EmbeddingTrainProfileVersion,
		CompactTrain: &backend.CompactTrainAcceleratorStats{
			LiveHandles:          4,
			ArenaReuseHits:       10,
			ArenaAllocations:     5,
			GradientReuseHits:    11,
			GradientAllocations:  5,
			ResidentGradBytes:    200,
			ActivationArenaBytes: 3000,
			WorkspaceArenaBytes:  6000,
		},
	}
	final := EmbeddingTrainProfile{
		Version: EmbeddingTrainProfileVersion,
		CompactTrain: &backend.CompactTrainAcceleratorStats{
			LiveHandles:          3,
			ArenaReuseHits:       12,
			ArenaAllocations:     6,
			GradientReuseHits:    13,
			GradientAllocations:  6,
			ResidentGradBytes:    230,
			ActivationArenaBytes: 3300,
			WorkspaceArenaBytes:  6600,
		},
	}

	preRestoreDelta := diffTrainProfile(start, preRestoreEnd)
	postRestoreDelta := diffTrainProfile(restoreStart, final)
	mergedDelta := addTrainProfileDelta(preRestoreDelta, postRestoreDelta)
	endProfile := applyTrainProfileDelta(preRestoreEnd, postRestoreDelta)

	assertCompactTrainProfileStats(t, mergedDelta.CompactTrain, &backend.CompactTrainAcceleratorStats{
		LiveHandles:          3,
		ArenaReuseHits:       4,
		ArenaAllocations:     2,
		GradientReuseHits:    5,
		GradientAllocations:  2,
		ResidentGradBytes:    80,
		ActivationArenaBytes: 3300,
		WorkspaceArenaBytes:  6600,
	})
	assertCompactTrainProfileStats(t, endProfile.CompactTrain, &backend.CompactTrainAcceleratorStats{
		LiveHandles:          3,
		ArenaReuseHits:       5,
		ArenaAllocations:     3,
		GradientReuseHits:    6,
		GradientAllocations:  3,
		ResidentGradBytes:    180,
		ActivationArenaBytes: 3300,
		WorkspaceArenaBytes:  6600,
	})
}

func assertCompactTrainProfileStats(t *testing.T, got, want *backend.CompactTrainAcceleratorStats) {
	t.Helper()
	if got == nil {
		t.Fatalf("compact train stats = nil, want %+v", want)
	}
	if got.LiveHandles != want.LiveHandles || got.ArenaReuseHits != want.ArenaReuseHits || got.ArenaAllocations != want.ArenaAllocations || got.GradientReuseHits != want.GradientReuseHits || got.GradientAllocations != want.GradientAllocations || got.ResidentGradBytes != want.ResidentGradBytes || got.ActivationArenaBytes != want.ActivationArenaBytes || got.WorkspaceArenaBytes != want.WorkspaceArenaBytes {
		t.Fatalf("compact train stats = %+v, want selected counters/snapshots %+v", *got, *want)
	}
}

func assertOptimizerProfileCounters(t *testing.T, got, want backend.OptimizerAcceleratorStats) {
	t.Helper()
	if got.LogicalSteps != want.LogicalSteps {
		t.Fatalf("logical steps = %d, want %d", got.LogicalSteps, want.LogicalSteps)
	}
	if got.TensorUpdateCalls != want.TensorUpdateCalls {
		t.Fatalf("tensor update calls = %d, want %d", got.TensorUpdateCalls, want.TensorUpdateCalls)
	}
	if got.UpdateCalls != want.UpdateCalls {
		t.Fatalf("update calls = %d, want %d", got.UpdateCalls, want.UpdateCalls)
	}
	if got.ResidentGradBatchCalls != want.ResidentGradBatchCalls {
		t.Fatalf("resident-gradient batch calls = %d, want %d", got.ResidentGradBatchCalls, want.ResidentGradBatchCalls)
	}
	if got.ResidentGradBatchKernelLaunches != want.ResidentGradBatchKernelLaunches {
		t.Fatalf("resident-gradient batch kernel launches = %d, want %d", got.ResidentGradBatchKernelLaunches, want.ResidentGradBatchKernelLaunches)
	}
	if got.ResidentGradBatchKernelSyncs != want.ResidentGradBatchKernelSyncs {
		t.Fatalf("resident-gradient batch kernel synchronizations = %d, want %d", got.ResidentGradBatchKernelSyncs, want.ResidentGradBatchKernelSyncs)
	}
	if got.DeferredSyncUpdates != want.DeferredSyncUpdates {
		t.Fatalf("deferred sync updates = %d, want %d", got.DeferredSyncUpdates, want.DeferredSyncUpdates)
	}
	if got.SyncCalls != want.SyncCalls {
		t.Fatalf("sync calls = %d, want %d", got.SyncCalls, want.SyncCalls)
	}
	if got.ForcedSyncCalls != want.ForcedSyncCalls {
		t.Fatalf("forced sync calls = %d, want %d", got.ForcedSyncCalls, want.ForcedSyncCalls)
	}
	if got.LastForcedSyncReason != want.LastForcedSyncReason {
		t.Fatalf("last forced sync reason = %q, want %q", got.LastForcedSyncReason, want.LastForcedSyncReason)
	}
	if got.ResidentParams != want.ResidentParams {
		t.Fatalf("resident params = %d, want %d", got.ResidentParams, want.ResidentParams)
	}
}

func assertVectorDistillProfilePhases(t *testing.T, got, want EmbeddingVectorDistillPhaseTimers) {
	t.Helper()
	if got.EncodeNanos != want.EncodeNanos {
		t.Fatalf("encode nanos = %d, want %d", got.EncodeNanos, want.EncodeNanos)
	}
	if got.ProjectionLossNanos != want.ProjectionLossNanos {
		t.Fatalf("projection/loss nanos = %d, want %d", got.ProjectionLossNanos, want.ProjectionLossNanos)
	}
	if got.BackwardNanos != want.BackwardNanos {
		t.Fatalf("backward nanos = %d, want %d", got.BackwardNanos, want.BackwardNanos)
	}
	if got.OptimizerNanos != want.OptimizerNanos {
		t.Fatalf("optimizer nanos = %d, want %d", got.OptimizerNanos, want.OptimizerNanos)
	}
	if got.EncodeCalls != want.EncodeCalls {
		t.Fatalf("encode calls = %d, want %d", got.EncodeCalls, want.EncodeCalls)
	}
	if got.ProjectionLossCalls != want.ProjectionLossCalls {
		t.Fatalf("projection/loss calls = %d, want %d", got.ProjectionLossCalls, want.ProjectionLossCalls)
	}
	if got.BackwardCalls != want.BackwardCalls {
		t.Fatalf("backward calls = %d, want %d", got.BackwardCalls, want.BackwardCalls)
	}
	if got.OptimizerCalls != want.OptimizerCalls {
		t.Fatalf("optimizer calls = %d, want %d", got.OptimizerCalls, want.OptimizerCalls)
	}
}
