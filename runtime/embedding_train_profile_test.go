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
		Optimizer: backend.OptimizerAcceleratorStats{
			LogicalSteps:         4,
			TensorUpdateCalls:    13,
			UpdateCalls:          4,
			DeferredSyncUpdates:  9,
			SyncCalls:            2,
			ForcedSyncCalls:      1,
			LastForcedSyncReason: "restore-best-test",
			UploadedBytes:        1024,
			DownloadedBytes:      512,
			UpdateNanos:          100,
			SyncNanos:            40,
			ResidentParams:       3,
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
	if got.Optimizer.UpdateCalls != want.Optimizer.UpdateCalls {
		t.Fatalf("optimizer update calls = %d, want %d", got.Optimizer.UpdateCalls, want.Optimizer.UpdateCalls)
	}
	if got.Optimizer.LogicalSteps != want.Optimizer.LogicalSteps {
		t.Fatalf("optimizer logical steps = %d, want %d", got.Optimizer.LogicalSteps, want.Optimizer.LogicalSteps)
	}
	if got.Optimizer.TensorUpdateCalls != want.Optimizer.TensorUpdateCalls {
		t.Fatalf("optimizer tensor update calls = %d, want %d", got.Optimizer.TensorUpdateCalls, want.Optimizer.TensorUpdateCalls)
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
		Optimizer: backend.OptimizerAcceleratorStats{
			LogicalSteps:         2,
			TensorUpdateCalls:    9,
			UpdateCalls:          2,
			DeferredSyncUpdates:  7,
			SyncCalls:            1,
			ForcedSyncCalls:      1,
			LastForcedSyncReason: "left-pressure",
			ResidentParams:       3,
		},
	}
	right := EmbeddingTrainProfile{
		Version:          EmbeddingTrainProfileVersion,
		OptimizerBackend: eosartifact.BackendCUDA,
		Optimizer: backend.OptimizerAcceleratorStats{
			LogicalSteps:         3,
			TensorUpdateCalls:    11,
			UpdateCalls:          3,
			DeferredSyncUpdates:  5,
			SyncCalls:            2,
			ForcedSyncCalls:      2,
			LastForcedSyncReason: "right-capacity",
			ResidentParams:       4,
		},
	}

	merged := addTrainProfileDelta(left, right)
	assertOptimizerProfileCounters(t, merged.Optimizer, backend.OptimizerAcceleratorStats{
		LogicalSteps:         5,
		TensorUpdateCalls:    20,
		UpdateCalls:          5,
		DeferredSyncUpdates:  12,
		SyncCalls:            3,
		ForcedSyncCalls:      3,
		LastForcedSyncReason: "right-capacity",
		ResidentParams:       4,
	})

	base := EmbeddingTrainProfile{
		Version:          EmbeddingTrainProfileVersion,
		OptimizerBackend: eosartifact.BackendKind("host"),
		Optimizer: backend.OptimizerAcceleratorStats{
			LogicalSteps:         10,
			TensorUpdateCalls:    100,
			UpdateCalls:          10,
			DeferredSyncUpdates:  50,
			SyncCalls:            10,
			ForcedSyncCalls:      4,
			LastForcedSyncReason: "base-sync",
			ResidentParams:       8,
		},
	}
	applied := applyTrainProfileDelta(base, right)
	assertOptimizerProfileCounters(t, applied.Optimizer, backend.OptimizerAcceleratorStats{
		LogicalSteps:         13,
		TensorUpdateCalls:    111,
		UpdateCalls:          13,
		DeferredSyncUpdates:  55,
		SyncCalls:            12,
		ForcedSyncCalls:      6,
		LastForcedSyncReason: "right-capacity",
		ResidentParams:       4,
	})
}

func TestTrainProfileRestoreBestMergePreservesOptimizerCounters(t *testing.T) {
	start := EmbeddingTrainProfile{
		Version: EmbeddingTrainProfileVersion,
		Step:    10,
		Optimizer: backend.OptimizerAcceleratorStats{
			LogicalSteps:         10,
			TensorUpdateCalls:    100,
			UpdateCalls:          10,
			DeferredSyncUpdates:  80,
			SyncCalls:            2,
			ForcedSyncCalls:      1,
			LastForcedSyncReason: "startup",
			ResidentParams:       2,
		},
	}
	preRestoreEnd := EmbeddingTrainProfile{
		Version: EmbeddingTrainProfileVersion,
		Step:    14,
		Optimizer: backend.OptimizerAcceleratorStats{
			LogicalSteps:         14,
			TensorUpdateCalls:    140,
			UpdateCalls:          14,
			DeferredSyncUpdates:  120,
			SyncCalls:            5,
			ForcedSyncCalls:      2,
			LastForcedSyncReason: "pre-restore-pressure",
			ResidentParams:       4,
		},
	}
	restoreStart := EmbeddingTrainProfile{
		Version: EmbeddingTrainProfileVersion,
		Step:    12,
		Optimizer: backend.OptimizerAcceleratorStats{
			LogicalSteps:         12,
			TensorUpdateCalls:    120,
			UpdateCalls:          12,
			DeferredSyncUpdates:  96,
			SyncCalls:            3,
			ForcedSyncCalls:      1,
			LastForcedSyncReason: "startup",
			ResidentParams:       3,
		},
	}
	final := EmbeddingTrainProfile{
		Version: EmbeddingTrainProfileVersion,
		Step:    13,
		Optimizer: backend.OptimizerAcceleratorStats{
			LogicalSteps:         13,
			TensorUpdateCalls:    133,
			UpdateCalls:          13,
			DeferredSyncUpdates:  108,
			SyncCalls:            4,
			ForcedSyncCalls:      2,
			LastForcedSyncReason: "final-eval-sync",
			ResidentParams:       5,
		},
	}

	preRestoreDelta := diffTrainProfile(start, preRestoreEnd)
	postRestoreDelta := diffTrainProfile(restoreStart, final)
	mergedDelta := addTrainProfileDelta(preRestoreDelta, postRestoreDelta)
	endProfile := applyTrainProfileDelta(preRestoreEnd, postRestoreDelta)

	assertOptimizerProfileCounters(t, mergedDelta.Optimizer, backend.OptimizerAcceleratorStats{
		LogicalSteps:         5,
		TensorUpdateCalls:    53,
		UpdateCalls:          5,
		DeferredSyncUpdates:  52,
		SyncCalls:            4,
		ForcedSyncCalls:      2,
		LastForcedSyncReason: "final-eval-sync",
		ResidentParams:       5,
	})
	assertOptimizerProfileCounters(t, endProfile.Optimizer, backend.OptimizerAcceleratorStats{
		LogicalSteps:         15,
		TensorUpdateCalls:    153,
		UpdateCalls:          15,
		DeferredSyncUpdates:  132,
		SyncCalls:            6,
		ForcedSyncCalls:      3,
		LastForcedSyncReason: "final-eval-sync",
		ResidentParams:       5,
	})
}

func TestTrainProfileOptimizerCounterActivity(t *testing.T) {
	cases := []struct {
		name  string
		stats backend.OptimizerAcceleratorStats
	}{
		{name: "logical steps", stats: backend.OptimizerAcceleratorStats{LogicalSteps: 1}},
		{name: "tensor update calls", stats: backend.OptimizerAcceleratorStats{TensorUpdateCalls: 1}},
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
