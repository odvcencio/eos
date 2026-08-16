package eosruntime

import (
	"encoding/json"
	"path/filepath"
	"strings"
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

func TestCompactTrainTelemetryDeltaAndAggregation(t *testing.T) {
	forward := backend.CompactTrainTelemetryPhaseIndex(backend.CompactTrainPhaseForwardDirect)
	if forward < 0 {
		t.Fatal("forward telemetry phase has no stable index")
	}
	start := &backend.CompactTrainTelemetry{Enabled: true}
	start.Phases[forward] = backend.CompactTrainPhaseTelemetry{GoCalls: 2, HostNanos: 10, FailIndex: 3, HasFailIndex: true, Failures: 1, FailureStage: "launch"}
	end := *start
	end.EventTiming = true
	end.Phases[forward] = backend.CompactTrainPhaseTelemetry{GoCalls: 7, DriverCalls: 5, HostNanos: 40, DeviceElapsedNanos: 9, Failures: 2, FailIndex: 4, HasFailIndex: true, FailureStage: "launch"}
	delta := backend.DiffCompactTrainTelemetry(start, &end)
	got := delta.Phases[forward]
	if got.GoCalls != 5 || got.DriverCalls != 5 || got.HostNanos != 30 || got.DeviceElapsedNanos != 9 || got.FailIndex != 4 || !got.HasFailIndex || got.FailureStage != "launch" || !delta.EventTiming {
		t.Fatalf("telemetry delta = %+v, want go=5 driver=5 host=30 device=9 fail=4 enabled event", got)
	}
	merged := backend.AddCompactTrainTelemetry(start, delta)
	if merged.Phases[forward] != end.Phases[forward] || !merged.Enabled || !merged.EventTiming {
		t.Fatalf("telemetry aggregation = %+v, want end phase %+v", merged.Phases[forward], end.Phases[forward])
	}
	if got := backend.AddCompactTrainTelemetry(&end, nil).Phases[forward].FailIndex; got != end.Phases[forward].FailIndex {
		t.Fatalf("zero telemetry aggregation changed failure index to %d", got)
	}
}

func TestCompactTrainTelemetryEventControlsAndViews(t *testing.T) {
	index := backend.CompactTrainTelemetryPhaseIndex(backend.CompactTrainPhaseK5Batch)
	start := &backend.CompactTrainTelemetry{
		Enabled: true,
		View:    backend.CompactTrainTelemetryViewCompactTrain,
	}
	start.Phases[index] = backend.CompactTrainPhaseTelemetry{Completed: 3}
	start.SetPhasePresent(index)
	end := backend.CloneCompactTrainTelemetry(start)
	end.View = backend.CompactTrainTelemetryViewOptimizer
	end.Phases[index].EventControlFailures = 1
	end.Phases[index].EventControlStage = "event_query"
	end.SetPhasePresent(index)
	delta := backend.DiffCompactTrainTelemetry(start, end)
	got := delta.Phases[index]
	if delta.View != backend.CompactTrainTelemetryViewMixed || got.Completed != 0 || got.Failures != 0 || got.EventControlFailures != 1 || got.EventControlStage != "event_query" {
		t.Fatalf("event-control delta = view=%q phase=%+v, want mixed/event-only delta", delta.View, got)
	}
	merged := backend.AddCompactTrainTelemetry(start, delta)
	if merged.View != backend.CompactTrainTelemetryViewMixed || merged.Phases[index].Completed != 3 || merged.Phases[index].EventControlFailures != 1 || merged.Phases[index].Failures != 0 {
		t.Fatalf("event-control aggregation = view=%q phase=%+v", merged.View, merged.Phases[index])
	}
	encoded, err := json.Marshal(merged)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(encoded), `"view":"mixed"`) || !strings.Contains(string(encoded), `"event_control_stage":"event_query"`) {
		t.Fatalf("event-control/view JSON = %s", encoded)
	}
	var decoded backend.CompactTrainTelemetry
	if err := json.Unmarshal(encoded, &decoded); err != nil {
		t.Fatal(err)
	}
	if decoded.View != merged.View || decoded.Phases[index] != merged.Phases[index] {
		t.Fatalf("event-control/view round-trip = %+v, want %+v", decoded, *merged)
	}
}

func TestCompactTrainTelemetryOptionalNamedJSONAndDeepCopy(t *testing.T) {
	zeroStats, err := json.Marshal(backend.CompactTrainAcceleratorStats{})
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(zeroStats), "telemetry") {
		t.Fatalf("default compact stats unexpectedly serialized telemetry: %s", zeroStats)
	}
	zeroProfile, err := json.Marshal(EmbeddingTrainProfile{Version: EmbeddingTrainProfileVersion})
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(zeroProfile), "compact_train_telemetry") || strings.Contains(string(zeroProfile), "\"telemetry\"") {
		t.Fatalf("default profile unexpectedly serialized telemetry: %s", zeroProfile)
	}

	index := backend.CompactTrainTelemetryPhaseIndex(backend.CompactTrainPhaseForwardDirect)
	telemetry := &backend.CompactTrainTelemetry{Enabled: true}
	telemetry.Phases[index] = backend.CompactTrainPhaseTelemetry{GoCalls: 2, Failures: 1, FailIndex: 0, HasFailIndex: true, FailureStage: "launch"}
	telemetry.SetPhasePresent(index)
	stats := backend.CompactTrainAcceleratorStats{Telemetry: telemetry}
	encoded, err := json.Marshal(stats)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(encoded), "\"schema_version\":1") || !strings.Contains(string(encoded), "forward_direct") || strings.Contains(string(encoded), "\"phases\":[") || strings.Contains(string(encoded), "context_attempt") || strings.Contains(string(encoded), "context_success") {
		t.Fatalf("enabled telemetry schema is not named/versioned: %s", encoded)
	}
	var decoded backend.CompactTrainAcceleratorStats
	if err := json.Unmarshal(encoded, &decoded); err != nil {
		t.Fatal(err)
	}
	if decoded.Telemetry == nil || decoded.Telemetry.Phases[index] != telemetry.Phases[index] {
		t.Fatalf("named telemetry round-trip lost phase: got %+v want %+v", decoded.Telemetry, telemetry)
	}
	clone := backend.CloneCompactTrainTelemetry(telemetry)
	clone.Phases[index].GoCalls = 99
	if telemetry.Phases[index].GoCalls == 99 {
		t.Fatal("telemetry clone aliases source phase array")
	}
	profileClone := addCompactTrainStats(&stats, nil)
	profileClone.Telemetry.Phases[index].GoCalls = 77
	if stats.Telemetry.Phases[index].GoCalls == 77 {
		t.Fatal("profile stats clone aliases source telemetry")
	}
}

func TestCompactTrainTelemetryFailureDeltaPresenceAndReset(t *testing.T) {
	index := backend.CompactTrainTelemetryPhaseIndex(backend.CompactTrainPhaseForwardDirect)
	start := &backend.CompactTrainTelemetry{Enabled: true}
	start.Phases[index] = backend.CompactTrainPhaseTelemetry{Failures: 1, FailIndex: 0, HasFailIndex: true, FailureStage: "launch"}
	start.SetPhasePresent(index)
	noActivity := backend.DiffCompactTrainTelemetry(start, start)
	noActivityJSON, err := json.Marshal(noActivity)
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(noActivityJSON), "forward_direct") {
		t.Fatalf("no-activity delta re-emitted stale empty phase: %s", noActivityJSON)
	}
	success := *start
	success.Phases[index].GoCalls = 3
	delta := backend.DiffCompactTrainTelemetry(start, &success)
	if got := delta.Phases[index]; got.Failures != 0 || got.HasFailIndex || got.FailIndex != 0 || got.FailureStage != "" {
		t.Fatalf("no-new-failure delta repeated metadata: %+v", got)
	}
	merged := backend.AddCompactTrainTelemetry(start, delta)
	if got := merged.Phases[index]; !got.HasFailIndex || got.FailIndex != 0 || got.FailureStage != "launch" {
		t.Fatalf("success add lost prior failure metadata: %+v", got)
	}
	backend.ResetCompactTrainTelemetry(merged)
	if merged.Enabled || merged.Phases[index] != (backend.CompactTrainPhaseTelemetry{}) {
		t.Fatalf("telemetry reset retained state: %+v", merged)
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
		CompactTrain: &backend.CompactTrainAcceleratorStats{
			ForwardInputUploadBatchCalls:      7,
			ForwardInputUploadContextSets:     8,
			ForwardInputUploadDeviceCopies:    9,
			ForwardInputUploadFailures:        10,
			ForwardInputUploadScalarFallbacks: 11,
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
	assertCompactTrainInputUploadCounters(t, got.CompactTrain, want.CompactTrain)
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

func TestTrainProfileCompactForwardReadbackCounterActivity(t *testing.T) {
	for _, tc := range []struct {
		name  string
		stats backend.CompactTrainAcceleratorStats
	}{
		{name: "batch entries", stats: backend.CompactTrainAcceleratorStats{ForwardReadbackBatchEntries: 1}},
		{name: "context sets", stats: backend.CompactTrainAcceleratorStats{ForwardReadbackContextSets: 1}},
		{name: "device copies", stats: backend.CompactTrainAcceleratorStats{ForwardReadbackDeviceCopies: 1}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if !hasTrainProfileActivity(EmbeddingTrainProfile{CompactTrain: &tc.stats}) {
				t.Fatal("expected compact-forward readback counter activity")
			}
		})
	}
}

func TestTrainProfileCompactForwardInputUploadCounterActivity(t *testing.T) {
	for _, tc := range []struct {
		name  string
		stats backend.CompactTrainAcceleratorStats
	}{
		{name: "batch calls", stats: backend.CompactTrainAcceleratorStats{ForwardInputUploadBatchCalls: 1}},
		{name: "context sets", stats: backend.CompactTrainAcceleratorStats{ForwardInputUploadContextSets: 1}},
		{name: "device copies", stats: backend.CompactTrainAcceleratorStats{ForwardInputUploadDeviceCopies: 1}},
		{name: "failures", stats: backend.CompactTrainAcceleratorStats{ForwardInputUploadFailures: 1}},
		{name: "scalar fallbacks", stats: backend.CompactTrainAcceleratorStats{ForwardInputUploadScalarFallbacks: 1}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if !hasTrainProfileActivity(EmbeddingTrainProfile{CompactTrain: &tc.stats}) {
				t.Fatal("expected compact-forward input-upload counter activity")
			}
		})
	}
}

func TestTrainProfileCompactForwardGraphCounterActivity(t *testing.T) {
	cases := []struct {
		name string
		set  func(*backend.CompactTrainAcceleratorStats)
	}{
		{name: "graph captures", set: func(stats *backend.CompactTrainAcceleratorStats) { stats.GraphCaptures = 1 }},
		{name: "graph replays", set: func(stats *backend.CompactTrainAcceleratorStats) { stats.GraphReplays = 1 }},
		{name: "graph launches", set: func(stats *backend.CompactTrainAcceleratorStats) { stats.GraphLaunches = 1 }},
		{name: "graph nodes", set: func(stats *backend.CompactTrainAcceleratorStats) { stats.GraphNodes = 1 }},
		{name: "graph capture failures", set: func(stats *backend.CompactTrainAcceleratorStats) { stats.GraphCaptureFailures = 1 }},
		{name: "graph replay failures", set: func(stats *backend.CompactTrainAcceleratorStats) { stats.GraphReplayFailures = 1 }},
		{name: "graph invalidations", set: func(stats *backend.CompactTrainAcceleratorStats) { stats.GraphInvalidations = 1 }},
		{name: "graph parity failures", set: func(stats *backend.CompactTrainAcceleratorStats) { stats.GraphParityFailures = 1 }},
		{name: "graph fallbacks", set: func(stats *backend.CompactTrainAcceleratorStats) { stats.GraphFallbacks = 1 }},
		{name: "graph synchronizations", set: func(stats *backend.CompactTrainAcceleratorStats) { stats.GraphSynchronizations = 1 }},
		{name: "direct forward submissions", set: func(stats *backend.CompactTrainAcceleratorStats) { stats.DirectForwardSubmissions = 1 }},
		{name: "graph executed nodes", set: func(stats *backend.CompactTrainAcceleratorStats) { stats.GraphExecutedNodes = 1 }},
		{name: "forward device kernel work", set: func(stats *backend.CompactTrainAcceleratorStats) { stats.ForwardDeviceKernelWork = 1 }},
		{name: "last forward direct submissions", set: func(stats *backend.CompactTrainAcceleratorStats) { stats.LastForwardDirectSubmissions = 1 }},
		{name: "last forward device kernel work", set: func(stats *backend.CompactTrainAcceleratorStats) { stats.LastForwardDeviceKernelWork = 1 }},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			stats := backend.CompactTrainAcceleratorStats{}
			tc.set(&stats)
			if !hasTrainProfileActivity(EmbeddingTrainProfile{CompactTrain: &stats}) {
				t.Fatalf("expected compact-train graph activity for %+v", stats)
			}
		})
	}
}

func TestDiffCompactTrainStatsExcludesWarmupAllocationsKeepsMeasuredReuse(t *testing.T) {
	start := &backend.CompactTrainAcceleratorStats{
		ArenaAllocations:                  1,
		ArenaReuseHits:                    1,
		GradientAllocations:               1,
		GradientReuseHits:                 1,
		LiveHandles:                       2,
		ActivationArenaBytes:              4096,
		WorkspaceArenaBytes:               2048,
		ForwardReadbackBatchEntries:       4,
		ForwardReadbackContextSets:        5,
		ForwardReadbackDeviceCopies:       12,
		ForwardInputUploadBatchCalls:      2,
		ForwardInputUploadContextSets:     3,
		ForwardInputUploadDeviceCopies:    4,
		ForwardInputUploadFailures:        5,
		ForwardInputUploadScalarFallbacks: 6,
		GraphCaptures:                     2,
		GraphReplays:                      3,
		GraphLaunches:                     4,
		GraphNodes:                        5,
		GraphCaptureFailures:              6,
		GraphReplayFailures:               7,
		GraphInvalidations:                8,
		GraphParityFailures:               9,
		GraphFallbacks:                    10,
		GraphSynchronizations:             11,
		DirectForwardSubmissions:          12,
		GraphExecutedNodes:                13,
		ForwardDeviceKernelWork:           14,
		LastShape:                         backend.CompactForwardShape{Batch: 1, Tokens: 4, ModelDim: 8},
		LastForwardLaunches:               3,
		LastBackwardLaunches:              4,
		LastForwardCublasGemmCalls:        1,
		LastBackwardCublasGemmCalls:       2,
		LastForwardSyncs:                  1,
		LastBackwardSyncs:                 1,
		LastForwardDirectSubmissions:      1,
		LastForwardDeviceKernelWork:       2,
	}
	end := &backend.CompactTrainAcceleratorStats{
		ArenaAllocations:                  1,
		ArenaReuseHits:                    2,
		GradientAllocations:               1,
		GradientReuseHits:                 2,
		LiveHandles:                       1,
		ActivationArenaBytes:              8192,
		WorkspaceArenaBytes:               4096,
		ForwardReadbackBatchEntries:       6,
		ForwardReadbackContextSets:        7,
		ForwardReadbackDeviceCopies:       18,
		ForwardInputUploadBatchCalls:      5,
		ForwardInputUploadContextSets:     8,
		ForwardInputUploadDeviceCopies:    10,
		ForwardInputUploadFailures:        12,
		ForwardInputUploadScalarFallbacks: 14,
		GraphCaptures:                     11,
		GraphReplays:                      15,
		GraphLaunches:                     17,
		GraphNodes:                        19,
		GraphCaptureFailures:              21,
		GraphReplayFailures:               23,
		GraphInvalidations:                25,
		GraphParityFailures:               27,
		GraphFallbacks:                    29,
		GraphSynchronizations:             31,
		DirectForwardSubmissions:          33,
		GraphExecutedNodes:                35,
		ForwardDeviceKernelWork:           37,
		LastShape:                         backend.CompactForwardShape{Batch: 1, Tokens: 4, ModelDim: 8, FFNDim: 16},
		LastForwardLaunches:               5,
		LastBackwardLaunches:              6,
		LastForwardCublasGemmCalls:        3,
		LastBackwardCublasGemmCalls:       4,
		LastForwardSyncs:                  2,
		LastBackwardSyncs:                 2,
		LastForwardDirectSubmissions:      3,
		LastForwardDeviceKernelWork:       4,
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
	if got.ForwardReadbackBatchEntries != 2 || got.ForwardReadbackContextSets != 2 || got.ForwardReadbackDeviceCopies != 6 {
		t.Fatalf("forward readback counters = %d/%d/%d, want 2/2/6", got.ForwardReadbackBatchEntries, got.ForwardReadbackContextSets, got.ForwardReadbackDeviceCopies)
	}
	if got.ForwardInputUploadBatchCalls != 3 || got.ForwardInputUploadContextSets != 5 || got.ForwardInputUploadDeviceCopies != 6 || got.ForwardInputUploadFailures != 7 || got.ForwardInputUploadScalarFallbacks != 8 {
		t.Fatalf("forward input-upload counters = %d/%d/%d/%d/%d, want 3/5/6/7/8", got.ForwardInputUploadBatchCalls, got.ForwardInputUploadContextSets, got.ForwardInputUploadDeviceCopies, got.ForwardInputUploadFailures, got.ForwardInputUploadScalarFallbacks)
	}
	assertCompactTrainGraphCounters(t, got, &backend.CompactTrainAcceleratorStats{
		GraphCaptures:            9,
		GraphReplays:             12,
		GraphLaunches:            13,
		GraphNodes:               14,
		GraphCaptureFailures:     15,
		GraphReplayFailures:      16,
		GraphInvalidations:       17,
		GraphParityFailures:      18,
		GraphFallbacks:           19,
		GraphSynchronizations:    20,
		DirectForwardSubmissions: 21,
		GraphExecutedNodes:       22,
		ForwardDeviceKernelWork:  23,
	})
	assertCompactTrainGraphSnapshots(t, got, &backend.CompactTrainAcceleratorStats{
		LastForwardDirectSubmissions: 3,
		LastForwardDeviceKernelWork:  4,
	})
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
			LiveHandles:                       2,
			ArenaReuseHits:                    1,
			ArenaAllocations:                  1,
			GradientReuseHits:                 1,
			GradientAllocations:               1,
			ResidentGradBytes:                 100,
			ActivationArenaBytes:              1000,
			WorkspaceArenaBytes:               2000,
			ForwardReadbackBatchEntries:       1,
			ForwardReadbackContextSets:        1,
			ForwardReadbackDeviceCopies:       3,
			ForwardInputUploadBatchCalls:      2,
			ForwardInputUploadContextSets:     3,
			ForwardInputUploadDeviceCopies:    4,
			ForwardInputUploadFailures:        5,
			ForwardInputUploadScalarFallbacks: 6,
			GraphCaptures:                     1,
			GraphReplays:                      2,
			GraphLaunches:                     3,
			GraphNodes:                        4,
			GraphCaptureFailures:              5,
			GraphReplayFailures:               6,
			GraphInvalidations:                7,
			GraphParityFailures:               8,
			GraphFallbacks:                    9,
			GraphSynchronizations:             10,
			DirectForwardSubmissions:          11,
			GraphExecutedNodes:                12,
			ForwardDeviceKernelWork:           13,
			LastForwardDirectSubmissions:      14,
			LastForwardDeviceKernelWork:       15,
		},
	}
	preRestoreEnd := EmbeddingTrainProfile{
		Version: EmbeddingTrainProfileVersion,
		CompactTrain: &backend.CompactTrainAcceleratorStats{
			LiveHandles:                       1,
			ArenaReuseHits:                    3,
			ArenaAllocations:                  2,
			GradientReuseHits:                 4,
			GradientAllocations:               2,
			ResidentGradBytes:                 150,
			ActivationArenaBytes:              1100,
			WorkspaceArenaBytes:               2200,
			ForwardReadbackBatchEntries:       3,
			ForwardReadbackContextSets:        3,
			ForwardReadbackDeviceCopies:       9,
			ForwardInputUploadBatchCalls:      5,
			ForwardInputUploadContextSets:     8,
			ForwardInputUploadDeviceCopies:    10,
			ForwardInputUploadFailures:        12,
			ForwardInputUploadScalarFallbacks: 14,
			GraphCaptures:                     4,
			GraphReplays:                      6,
			GraphLaunches:                     8,
			GraphNodes:                        10,
			GraphCaptureFailures:              12,
			GraphReplayFailures:               14,
			GraphInvalidations:                16,
			GraphParityFailures:               18,
			GraphFallbacks:                    20,
			GraphSynchronizations:             22,
			DirectForwardSubmissions:          24,
			GraphExecutedNodes:                26,
			ForwardDeviceKernelWork:           28,
			LastForwardDirectSubmissions:      29,
			LastForwardDeviceKernelWork:       30,
		},
	}
	restoreStart := EmbeddingTrainProfile{
		Version: EmbeddingTrainProfileVersion,
		CompactTrain: &backend.CompactTrainAcceleratorStats{
			LiveHandles:                       4,
			ArenaReuseHits:                    10,
			ArenaAllocations:                  5,
			GradientReuseHits:                 11,
			GradientAllocations:               5,
			ResidentGradBytes:                 200,
			ActivationArenaBytes:              3000,
			WorkspaceArenaBytes:               6000,
			ForwardReadbackBatchEntries:       4,
			ForwardReadbackContextSets:        4,
			ForwardReadbackDeviceCopies:       12,
			ForwardInputUploadBatchCalls:      10,
			ForwardInputUploadContextSets:     20,
			ForwardInputUploadDeviceCopies:    30,
			ForwardInputUploadFailures:        40,
			ForwardInputUploadScalarFallbacks: 50,
			GraphCaptures:                     30,
			GraphReplays:                      40,
			GraphLaunches:                     50,
			GraphNodes:                        60,
			GraphCaptureFailures:              70,
			GraphReplayFailures:               80,
			GraphInvalidations:                90,
			GraphParityFailures:               100,
			GraphFallbacks:                    110,
			GraphSynchronizations:             120,
			DirectForwardSubmissions:          130,
			GraphExecutedNodes:                140,
			ForwardDeviceKernelWork:           150,
			LastForwardDirectSubmissions:      31,
			LastForwardDeviceKernelWork:       32,
		},
	}
	final := EmbeddingTrainProfile{
		Version: EmbeddingTrainProfileVersion,
		CompactTrain: &backend.CompactTrainAcceleratorStats{
			LiveHandles:                       3,
			ArenaReuseHits:                    12,
			ArenaAllocations:                  6,
			GradientReuseHits:                 13,
			GradientAllocations:               6,
			ResidentGradBytes:                 230,
			ActivationArenaBytes:              3300,
			WorkspaceArenaBytes:               6600,
			ForwardReadbackBatchEntries:       6,
			ForwardReadbackContextSets:        6,
			ForwardReadbackDeviceCopies:       18,
			ForwardInputUploadBatchCalls:      15,
			ForwardInputUploadContextSets:     28,
			ForwardInputUploadDeviceCopies:    40,
			ForwardInputUploadFailures:        52,
			ForwardInputUploadScalarFallbacks: 64,
			GraphCaptures:                     35,
			GraphReplays:                      47,
			GraphLaunches:                     59,
			GraphNodes:                        71,
			GraphCaptureFailures:              83,
			GraphReplayFailures:               95,
			GraphInvalidations:                107,
			GraphParityFailures:               119,
			GraphFallbacks:                    131,
			GraphSynchronizations:             143,
			DirectForwardSubmissions:          155,
			GraphExecutedNodes:                167,
			ForwardDeviceKernelWork:           179,
			LastForwardDirectSubmissions:      41,
			LastForwardDeviceKernelWork:       42,
		},
	}

	preRestoreDelta := diffTrainProfile(start, preRestoreEnd)
	postRestoreDelta := diffTrainProfile(restoreStart, final)
	mergedDelta := addTrainProfileDelta(preRestoreDelta, postRestoreDelta)
	endProfile := applyTrainProfileDelta(preRestoreEnd, postRestoreDelta)

	assertCompactTrainProfileStats(t, mergedDelta.CompactTrain, &backend.CompactTrainAcceleratorStats{
		LiveHandles:                       3,
		ArenaReuseHits:                    4,
		ArenaAllocations:                  2,
		GradientReuseHits:                 5,
		GradientAllocations:               2,
		ResidentGradBytes:                 80,
		ActivationArenaBytes:              3300,
		WorkspaceArenaBytes:               6600,
		ForwardReadbackBatchEntries:       4,
		ForwardReadbackContextSets:        4,
		ForwardReadbackDeviceCopies:       12,
		ForwardInputUploadBatchCalls:      8,
		ForwardInputUploadContextSets:     13,
		ForwardInputUploadDeviceCopies:    16,
		ForwardInputUploadFailures:        19,
		ForwardInputUploadScalarFallbacks: 22,
		GraphCaptures:                     8,
		GraphReplays:                      11,
		GraphLaunches:                     14,
		GraphNodes:                        17,
		GraphCaptureFailures:              20,
		GraphReplayFailures:               23,
		GraphInvalidations:                26,
		GraphParityFailures:               29,
		GraphFallbacks:                    32,
		GraphSynchronizations:             35,
		DirectForwardSubmissions:          38,
		GraphExecutedNodes:                41,
		ForwardDeviceKernelWork:           44,
		LastForwardDirectSubmissions:      41,
		LastForwardDeviceKernelWork:       42,
	})
	assertCompactTrainProfileStats(t, endProfile.CompactTrain, &backend.CompactTrainAcceleratorStats{
		LiveHandles:                       3,
		ArenaReuseHits:                    5,
		ArenaAllocations:                  3,
		GradientReuseHits:                 6,
		GradientAllocations:               3,
		ResidentGradBytes:                 180,
		ActivationArenaBytes:              3300,
		WorkspaceArenaBytes:               6600,
		ForwardReadbackBatchEntries:       5,
		ForwardReadbackContextSets:        5,
		ForwardReadbackDeviceCopies:       15,
		ForwardInputUploadBatchCalls:      10,
		ForwardInputUploadContextSets:     16,
		ForwardInputUploadDeviceCopies:    20,
		ForwardInputUploadFailures:        24,
		ForwardInputUploadScalarFallbacks: 28,
		GraphCaptures:                     8,
		GraphReplays:                      11,
		GraphLaunches:                     14,
		GraphNodes:                        17,
		GraphCaptureFailures:              20,
		GraphReplayFailures:               23,
		GraphInvalidations:                26,
		GraphParityFailures:               29,
		GraphFallbacks:                    32,
		GraphSynchronizations:             35,
		DirectForwardSubmissions:          38,
		GraphExecutedNodes:                41,
		ForwardDeviceKernelWork:           44,
		LastForwardDirectSubmissions:      41,
		LastForwardDeviceKernelWork:       42,
	})
}

func assertCompactTrainProfileStats(t *testing.T, got, want *backend.CompactTrainAcceleratorStats) {
	t.Helper()
	if got == nil {
		t.Fatalf("compact train stats = nil, want %+v", want)
	}
	if got.LiveHandles != want.LiveHandles || got.ArenaReuseHits != want.ArenaReuseHits || got.ArenaAllocations != want.ArenaAllocations || got.GradientReuseHits != want.GradientReuseHits || got.GradientAllocations != want.GradientAllocations || got.ResidentGradBytes != want.ResidentGradBytes || got.ActivationArenaBytes != want.ActivationArenaBytes || got.WorkspaceArenaBytes != want.WorkspaceArenaBytes || got.ForwardReadbackBatchEntries != want.ForwardReadbackBatchEntries || got.ForwardReadbackContextSets != want.ForwardReadbackContextSets || got.ForwardReadbackDeviceCopies != want.ForwardReadbackDeviceCopies || got.ForwardInputUploadBatchCalls != want.ForwardInputUploadBatchCalls || got.ForwardInputUploadContextSets != want.ForwardInputUploadContextSets || got.ForwardInputUploadDeviceCopies != want.ForwardInputUploadDeviceCopies || got.ForwardInputUploadFailures != want.ForwardInputUploadFailures || got.ForwardInputUploadScalarFallbacks != want.ForwardInputUploadScalarFallbacks {
		t.Fatalf("compact train stats = %+v, want selected counters/snapshots %+v", *got, *want)
	}
}

func assertCompactTrainInputUploadCounters(t *testing.T, got, want *backend.CompactTrainAcceleratorStats) {
	t.Helper()
	if got == nil || want == nil {
		if got != want {
			t.Fatalf("compact train input-upload stats = %+v, want %+v", got, want)
		}
		return
	}
	if got.ForwardInputUploadBatchCalls != want.ForwardInputUploadBatchCalls || got.ForwardInputUploadContextSets != want.ForwardInputUploadContextSets || got.ForwardInputUploadDeviceCopies != want.ForwardInputUploadDeviceCopies || got.ForwardInputUploadFailures != want.ForwardInputUploadFailures || got.ForwardInputUploadScalarFallbacks != want.ForwardInputUploadScalarFallbacks {
		t.Fatalf("compact train input-upload counters = %d/%d/%d/%d/%d, want %d/%d/%d/%d/%d", got.ForwardInputUploadBatchCalls, got.ForwardInputUploadContextSets, got.ForwardInputUploadDeviceCopies, got.ForwardInputUploadFailures, got.ForwardInputUploadScalarFallbacks, want.ForwardInputUploadBatchCalls, want.ForwardInputUploadContextSets, want.ForwardInputUploadDeviceCopies, want.ForwardInputUploadFailures, want.ForwardInputUploadScalarFallbacks)
	}
}

func assertCompactTrainGraphCounters(t *testing.T, got, want *backend.CompactTrainAcceleratorStats) {
	t.Helper()
	if got.GraphCaptures != want.GraphCaptures ||
		got.GraphReplays != want.GraphReplays ||
		got.GraphLaunches != want.GraphLaunches ||
		got.GraphNodes != want.GraphNodes ||
		got.GraphCaptureFailures != want.GraphCaptureFailures ||
		got.GraphReplayFailures != want.GraphReplayFailures ||
		got.GraphInvalidations != want.GraphInvalidations ||
		got.GraphParityFailures != want.GraphParityFailures ||
		got.GraphFallbacks != want.GraphFallbacks ||
		got.GraphSynchronizations != want.GraphSynchronizations ||
		got.DirectForwardSubmissions != want.DirectForwardSubmissions ||
		got.GraphExecutedNodes != want.GraphExecutedNodes ||
		got.ForwardDeviceKernelWork != want.ForwardDeviceKernelWork {
		t.Fatalf("compact-train graph counters = %+v, want %+v", *got, *want)
	}
}

func assertCompactTrainGraphSnapshots(t *testing.T, got, want *backend.CompactTrainAcceleratorStats) {
	t.Helper()
	if got.LastForwardDirectSubmissions != want.LastForwardDirectSubmissions || got.LastForwardDeviceKernelWork != want.LastForwardDeviceKernelWork {
		t.Fatalf("compact-train graph snapshots = direct=%d/work=%d, want direct=%d/work=%d", got.LastForwardDirectSubmissions, got.LastForwardDeviceKernelWork, want.LastForwardDirectSubmissions, want.LastForwardDeviceKernelWork)
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
