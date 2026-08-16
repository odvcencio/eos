//go:build linux && cgo

package eosruntime

import (
	"fmt"
	"os"
	"strings"
	"testing"
	"time"

	eosartifact "m31labs.dev/eos/artifact/eos"
	"m31labs.dev/eos/runtime/backend"

	_ "m31labs.dev/eos/runtime/backends/cuda"
)

// EOS_RUN_COMPACT_RESIDENT_CUDA_WARM_BENCH is deliberately opt-in. The
// benchmark runs the real compact resident CUDA train path and must not be
// pulled into an ordinary `go test` or an unbounded `-bench .` invocation.
const compactResidentCUDAWarmBenchGate = "EOS_RUN_COMPACT_RESIDENT_CUDA_WARM_BENCH"

// BenchmarkVectorDistillCompactResidentTrainCUDAWarm measures a bounded warm
// resident step.  The first step is intentionally excluded from timing and
// counters reported for the measured loop: it establishes the exact-shape
// arena and gradient pools.  This is a diagnostic gate, not a release-shape
// throughput claim.
//
// The default fixture is a prevalidated canonical synthetic shape built from
// the existing test helpers' model contract (B=4,T=4,D=4,H=6,L=2,O=3). Set
// EOS_CUDA_WARM_BENCH_PROFILE=next (or descriptor) to construct the
// prevalidated next-run descriptor shape
// (B=1,T=256,D=128,H=256,L=2,O=128,V=16384,heads=4). The opt-in route must be
// invoked with -benchtime=1x; the code-level b.N guard prevents calibration
// from executing additional measured steps.
func BenchmarkVectorDistillCompactResidentTrainCUDAWarm(b *testing.B) {
	if strings.TrimSpace(os.Getenv(compactResidentCUDAWarmBenchGate)) != "1" {
		b.Skipf("opt-in benchmark: set %s=1", compactResidentCUDAWarmBenchGate)
	}
	if b.N != 1 {
		b.Fatalf("opt-in benchmark requires exactly one measured step; rerun with -benchtime=1x (b.N=%d)", b.N)
	}

	shape, err := compactResidentCUDAWarmBenchShapeFromEnv()
	if err != nil {
		b.Fatalf("compact resident CUDA warm benchmark shape: %v", err)
	}

	// Probe the registered CUDA resident-train factory before constructing the
	// trainer. Any opt-in factory error or backend mismatch is a benchmark
	// failure; only the gate-unset path above is an intentional skip.
	probe, kind, err := backend.NewPreferredCompactTrainAccelerator(eosartifact.BackendCUDA)
	if err != nil {
		b.Fatalf("CUDA resident-train accelerator factory failed: %v", err)
	}
	probeBackend := eosartifact.BackendKind("")
	if probe != nil {
		probeBackend = probe.Backend()
	}
	if probe == nil || kind != eosartifact.BackendCUDA || probeBackend != eosartifact.BackendCUDA {
		if probe != nil {
			closeCompactResidentCUDAWarmBenchAccelerator(probe)
		}
		b.Fatalf("CUDA resident-train accelerator selection failed (registered_backend=%q accelerator_backend=%q accel=%T)", kind, probeBackend, probe)
	}
	closeCompactResidentCUDAWarmBenchAccelerator(probe)

	b.Setenv(compactResidentTrainEnv, "1")
	b.Setenv(compactPackedForwardEnv, "0")
	// The exact K7 launch contract is the compact boundary configuration. Set
	// these before constructing the accelerator so constructor-time flags cannot
	// select the per-launch or cuBLAS variants.
	b.Setenv("EOS_CUDA_COMPACT_SYNC_EACH_LAUNCH", "0")
	b.Setenv("EOS_CUDA_COMPACT_TRAIN_CUBLAS", "0")
	graphEnabled := false
	switch strings.ToLower(strings.TrimSpace(os.Getenv("EOS_CUDA_COMPACT_TRAIN_FORWARD_GRAPH"))) {
	case "1", "true", "yes":
		graphEnabled = true
	}

	checkpoint := compactResidentCUDAWarmBenchCheckpoint(shape)
	state, err := LoadCompactEmbeddingTrainStateFromCheckpoint(checkpoint, checkpoint.Manifest)
	if err != nil {
		b.Fatalf("load compact resident CUDA warm benchmark state: %v", err)
	}
	trainer, err := newCompactEmbeddingTrainerFromTrainState(&eosartifact.Module{Name: "compact-warm-bench"}, state)
	if err != nil {
		b.Fatalf("new compact resident CUDA warm benchmark trainer: %v", err)
	}
	defer trainer.Close()
	trainerAccelBackend := eosartifact.BackendKind("")
	if trainer.compactTrainAccel != nil {
		trainerAccelBackend = trainer.compactTrainAccel.Backend()
	}
	if trainer.compactTrainAccel == nil || trainer.compactTrainBackend != eosartifact.BackendCUDA || trainerAccelBackend != eosartifact.BackendCUDA {
		b.Fatalf("compact resident CUDA path selection failed after setup (configured_backend=%q accelerator_backend=%q accel=%T)", trainer.compactTrainBackend, trainerAccelBackend, trainer.compactTrainAccel)
	}

	batch := compactResidentCUDAWarmBenchBatch(shape)
	var scratch vectorDistillBatchScratch
	var projection *vectorDistillProjectionState
	initialProfile := trainer.TrainProfile()
	if initialProfile.CompactTrainBackend != eosartifact.BackendCUDA || initialProfile.CompactTrain == nil {
		b.Fatalf("compact resident CUDA profile selection failed before warm-up (backend=%q stats=%v)", initialProfile.CompactTrainBackend, initialProfile.CompactTrain)
	}

	// Warm-up is outside the benchmark timer and outside the reported measured
	// delta.  It is exactly one bounded resident training step, not Fit or an
	// epoch/training run.
	if _, projection, err = trainer.trainVectorDistillBatchWithScratch(batch, projection, shape.OutputDim, EmbeddingRoleQuery, 0, &scratch); err != nil {
		b.Fatalf("compact resident CUDA warm-up step: %v", err)
	}
	warmProfile := trainer.TrainProfile()
	warmDelta := diffTrainProfile(initialProfile, warmProfile)
	warmStats := compactResidentCUDAWarmBenchCompactStats(warmDelta)
	if warmStats == nil {
		b.Fatalf("compact resident CUDA warm-up profile has no compact-train stats")
	}
	if warmStats.FallbackOrUnhandled != 0 {
		b.Fatalf("compact resident CUDA warm-up reported compact-train fallback/unhandled=%d", warmStats.FallbackOrUnhandled)
	}
	assertCompactResidentCUDAWarmBenchGraphStep(b, "warmup", warmStats, shape, graphEnabled, false)
	assertCompactResidentCUDAWarmBenchK5(b, "warmup", warmDelta.Optimizer)
	b.Logf("scope=diagnostic_only warmup_excluded=true warmup_steps=1 profile=%s shape=%s", compactResidentCUDAWarmBenchProfileLabel(shape), shape.String())
	b.Logf("warmup_counters compact_train=%+v optimizer=%+v", compactResidentCUDAWarmBenchCompactStats(warmDelta), warmDelta.Optimizer)

	b.ResetTimer()
	measureStart := time.Now()
	for i := 0; i < b.N; i++ {
		if _, projection, err = trainer.trainVectorDistillBatchWithScratch(batch, projection, shape.OutputDim, EmbeddingRoleQuery, 0, &scratch); err != nil {
			b.StopTimer()
			b.Fatalf("compact resident CUDA measured step %d/%d: %v", i+1, b.N, err)
		}
	}
	b.StopTimer()
	measuredElapsed := time.Since(measureStart)

	measuredProfile := trainer.TrainProfile()
	measuredDelta := diffTrainProfile(warmProfile, measuredProfile)
	stats := compactResidentCUDAWarmBenchCompactStats(measuredDelta)
	if stats == nil {
		b.Fatalf("compact resident CUDA measured profile has no compact-train stats")
	}
	if measuredProfile.CompactTrainBackend != eosartifact.BackendCUDA {
		b.Fatalf("compact resident CUDA measured profile selected backend=%q", measuredProfile.CompactTrainBackend)
	}
	if stats.FallbackOrUnhandled != 0 {
		b.Fatalf("compact resident CUDA measured step reported compact-train fallback/unhandled=%d", stats.FallbackOrUnhandled)
	}
	assertCompactResidentCUDAWarmBenchGraphStep(b, "measured", stats, shape, graphEnabled, true)
	optimizerStats := measuredDelta.Optimizer
	assertCompactResidentCUDAWarmBenchK5(b, "measured", optimizerStats)
	steps := b.N
	if steps <= 0 {
		steps = 1
	}
	perStep := func(value int64) float64 { return float64(value) / float64(steps) }

	// testing.B reports the standard ns/op.  wall_ns/op is emitted explicitly
	// so the report remains clear that this is host-observed wall time and not
	// a device event measurement.
	b.ReportMetric(float64(measuredElapsed.Nanoseconds())/float64(steps), "wall_ns/op")
	b.ReportMetric(perStep(stats.ArenaAllocations), "arena_allocations/step")
	b.ReportMetric(perStep(stats.ArenaReuseHits), "arena_reuse_hits/step")
	b.ReportMetric(perStep(stats.GradientAllocations), "gradient_allocations/step")
	b.ReportMetric(perStep(stats.GradientReuseHits), "gradient_reuse_hits/step")
	b.ReportMetric(perStep(stats.KernelLaunches), "kernel_launches/step")
	b.ReportMetric(perStep(stats.KernelSynchronizations), "kernel_syncs/step")
	b.ReportMetric(perStep(stats.UploadedBytes), "compact_upload_B/step")
	b.ReportMetric(perStep(stats.DownloadedBytes), "compact_download_B/step")
	b.ReportMetric(perStep(stats.PooledDownloadedBytes), "pooled_download_B/step")
	b.ReportMetric(perStep(stats.ForwardReadbackBatchEntries), "compact_forward_readback_batch_entries/step")
	b.ReportMetric(perStep(stats.ForwardReadbackContextSets), "compact_forward_readback_context_sets/step")
	b.ReportMetric(perStep(stats.ForwardReadbackDeviceCopies), "compact_forward_readback_device_copies/step")
	b.ReportMetric(perStep(stats.GradPooledUploadedBytes), "grad_pooled_upload_B/step")
	b.ReportMetric(perStep(stats.FallbackOrUnhandled), "compact_train_fallback_or_unhandled/step")
	b.ReportMetric(perStep(stats.GraphCaptures), "graph_captures/step")
	b.ReportMetric(perStep(stats.GraphReplays), "graph_replays/step")
	b.ReportMetric(perStep(stats.GraphLaunches), "graph_launches/step")
	b.ReportMetric(perStep(stats.GraphNodes), "graph_nodes_recorded/step")
	b.ReportMetric(perStep(stats.GraphExecutedNodes), "graph_executed_nodes/step")
	b.ReportMetric(perStep(stats.DirectForwardSubmissions), "direct_forward_submissions/step")
	b.ReportMetric(perStep(stats.ForwardDeviceKernelWork), "forward_device_kernel_work/step")
	b.ReportMetric(perStep(stats.KernelLaunches+stats.GraphExecutedNodes), "direct_plus_graph_nodes/step")
	b.ReportMetric(perStep(stats.GraphCaptureFailures), "graph_capture_failures/step")
	b.ReportMetric(perStep(stats.GraphReplayFailures), "graph_replay_failures/step")
	b.ReportMetric(perStep(stats.GraphInvalidations), "graph_invalidations/step")
	b.ReportMetric(perStep(stats.GraphParityFailures), "graph_parity_failures/step")
	b.ReportMetric(perStep(stats.GraphFallbacks), "graph_fallbacks/step")
	b.ReportMetric(perStep(stats.GraphSynchronizations), "graph_syncs/step")
	b.ReportMetric(perStep(measuredDelta.Optimizer.UploadedBytes), "optimizer_upload_B/step")
	b.ReportMetric(perStep(measuredDelta.Optimizer.DownloadedBytes), "optimizer_download_B/step")
	b.ReportMetric(perStep(optimizerStats.ResidentGradBatchCalls), "resident_grad_batch_calls/step")
	b.ReportMetric(perStep(optimizerStats.ResidentGradBatchKernelLaunches), "resident_grad_batch_kernel_launches/step")
	b.ReportMetric(perStep(optimizerStats.ResidentGradBatchKernelSyncs), "resident_grad_batch_kernel_syncs/step")
	b.ReportMetric(float64(shape.Batch), "shape_B")
	b.ReportMetric(float64(shape.Tokens), "shape_T")
	b.ReportMetric(float64(shape.ModelDim), "shape_D")
	b.ReportMetric(float64(shape.FFNDim), "shape_H")
	b.ReportMetric(float64(shape.Layers), "shape_L")
	b.ReportMetric(float64(shape.OutputDim), "shape_O")
	b.Logf("scope=diagnostic_only warmup_excluded=true measured_steps=%d shape=%s profile=%s wall_ns=%d", b.N, shape.String(), compactResidentCUDAWarmBenchProfileLabel(shape), measuredElapsed.Nanoseconds())
	b.Logf("measured_counters compact_train=%+v optimizer=%+v phases=%+v", stats, measuredDelta.Optimizer, measuredDelta.VectorDistillPhases)
}

type compactResidentCUDAWarmBenchShape struct {
	Batch       int
	Tokens      int
	ModelDim    int
	FFNDim      int
	Layers      int
	OutputDim   int
	Heads       int
	HeadDim     int
	Vocab       int
	MaxSequence int
	Profile     string
}

func (s compactResidentCUDAWarmBenchShape) String() string {
	return fmt.Sprintf("B=%d,T=%d,D=%d,H=%d,L=%d,O=%d,heads=%d,head_dim=%d,vocab=%d,max_seq=%d", s.Batch, s.Tokens, s.ModelDim, s.FFNDim, s.Layers, s.OutputDim, s.Heads, s.HeadDim, s.Vocab, s.MaxSequence)
}

func compactResidentCUDAWarmBenchProfileLabel(shape compactResidentCUDAWarmBenchShape) string {
	if shape.Profile == "" {
		return "canonical_fixture"
	}
	return shape.Profile
}

func TestCompactResidentCUDAWarmBenchExactProfiles(t *testing.T) {
	for _, tc := range []struct {
		name                  string
		profile               string
		wantForward, wantBack int64
		wantWhole             int64
	}{
		{name: "canonical", profile: "canonical", wantForward: 24, wantBack: 44, wantWhole: 68},
		{name: "next", profile: "next", wantForward: 22, wantBack: 43, wantWhole: 65},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Setenv("EOS_CUDA_WARM_BENCH_PROFILE", tc.profile)
			shape, err := compactResidentCUDAWarmBenchShapeFromEnv()
			if err != nil {
				t.Fatalf("profile shape: %v", err)
			}
			forward, backward, whole := compactResidentCUDAWarmBenchExpectedKernelCounts(shape)
			if forward != tc.wantForward || backward != tc.wantBack || whole != tc.wantWhole {
				t.Fatalf("profile %s exact counts = forward/backward/whole %d/%d/%d, want %d/%d/%d", tc.profile, forward, backward, whole, tc.wantForward, tc.wantBack, tc.wantWhole)
			}
			if shape.Profile == "canonical_fixture" && (shape.Batch != 4 || shape.Tokens != 4 || shape.ModelDim != 4 || shape.FFNDim != 6 || shape.Layers != 2 || shape.OutputDim != 3) {
				t.Fatalf("canonical profile shape = %s", shape.String())
			}
			if shape.Profile == "next_descriptor_shape" && (shape.Batch != 1 || shape.Tokens != 256 || shape.ModelDim != 128 || shape.FFNDim != 256 || shape.Layers != 2 || shape.OutputDim != 128) {
				t.Fatalf("next profile shape = %s", shape.String())
			}
		})
	}
}

func compactResidentCUDAWarmBenchShapeFromEnv() (compactResidentCUDAWarmBenchShape, error) {
	profile := strings.TrimSpace(os.Getenv("EOS_CUDA_WARM_BENCH_PROFILE"))
	var shape compactResidentCUDAWarmBenchShape
	switch profile {
	case "", "canonical":
		shape = compactResidentCUDAWarmBenchCanonicalShape()
	case "next", "descriptor":
		shape = compactResidentCUDAWarmBenchNextDescriptorShape()
	default:
		return compactResidentCUDAWarmBenchShape{}, fmt.Errorf("EOS_CUDA_WARM_BENCH_PROFILE=%q (want canonical, next, or descriptor)", profile)
	}
	if err := validateCompactResidentCUDAWarmBenchShape(shape); err != nil {
		return compactResidentCUDAWarmBenchShape{}, err
	}
	return shape, nil
}

func compactResidentCUDAWarmBenchCanonicalShape() compactResidentCUDAWarmBenchShape {
	return compactResidentCUDAWarmBenchShape{
		Batch:       4,
		Tokens:      4,
		ModelDim:    4,
		FFNDim:      6,
		Layers:      2,
		OutputDim:   3,
		Heads:       2,
		HeadDim:     2,
		Vocab:       5,
		MaxSequence: 8,
		Profile:     "canonical_fixture",
	}
}

func compactResidentCUDAWarmBenchNextDescriptorShape() compactResidentCUDAWarmBenchShape {
	return compactResidentCUDAWarmBenchShape{
		Batch:       1,
		Tokens:      256,
		ModelDim:    128,
		FFNDim:      256,
		Layers:      2,
		OutputDim:   128,
		Heads:       4,
		HeadDim:     32,
		Vocab:       16384,
		MaxSequence: 256,
		Profile:     "next_descriptor_shape",
	}
}

func validateCompactResidentCUDAWarmBenchShape(shape compactResidentCUDAWarmBenchShape) error {
	var expected compactResidentCUDAWarmBenchShape
	switch shape.Profile {
	case "canonical_fixture":
		expected = compactResidentCUDAWarmBenchCanonicalShape()
	case "next_descriptor_shape":
		expected = compactResidentCUDAWarmBenchNextDescriptorShape()
	default:
		return fmt.Errorf("benchmark profile %q is not prevalidated (want canonical or next/descriptor)", shape.Profile)
	}
	if shape != expected {
		return fmt.Errorf("benchmark profile %q must use prevalidated shape %s; got %s", shape.Profile, expected.String(), shape.String())
	}
	return nil
}

func compactResidentCUDAWarmBenchCheckpoint(shape compactResidentCUDAWarmBenchShape) EmbeddingTrainCheckpoint {
	manifest := EmbeddingManifest{
		Name:                  "compact-warm-bench",
		ArchitectureVersion:   EmbeddingArchitectureCompactTransformerV1,
		EncoderRepeats:        shape.Layers,
		ModelDim:              shape.ModelDim,
		OutputDim:             shape.OutputDim,
		AttentionHeads:        shape.Heads,
		HeadDim:               shape.HeadDim,
		FFNDim:                shape.FFNDim,
		ParameterTying:        EmbeddingParameterTyingUntied,
		TokenEmbeddingParam:   "token_embedding",
		RoleConditioning:      EmbeddingRoleConditioningAdditiveV1,
		RoleEmbeddingParam:    "role_embedding",
		AttentionMaskMode:     EmbeddingAttentionMaskModeKey,
		AttentionScoreScale:   EmbeddingAttentionScoreScaleKeyDimRSQ,
		PositionEncoding:      EmbeddingPositionEncodingRoPE,
		AttentionQueryParam:   "layer0_attn_q",
		AttentionKeyParam:     "layer0_attn_k",
		AttentionValueParam:   "layer0_attn_v",
		AttentionOutputParam:  "layer0_attn_o",
		HiddenProjectionParam: "layer0_ffn_up",
		ProjectionParam:       "layer0_ffn_down",
		Tokenizer:             TokenizerManifest{VocabSize: shape.Vocab, MaxSequence: shape.MaxSequence},
	}
	if shape.OutputDim != shape.ModelDim {
		manifest.OutputProjectionParam = "output_projection"
	}
	tensors := map[string]*backend.Tensor{
		"token_embedding": compactTrainStateTestTensor([]int{shape.Vocab, shape.ModelDim}, 0.03),
		"role_embedding":  compactTrainStateTestTensor([]int{3, shape.ModelDim}, 0.01),
	}
	for layer := 0; layer < shape.Layers; layer++ {
		offset := float32(layer+1) * 0.02
		tensors[compactLayerTensorName(layer, "attn_q")] = compactTrainStateTestTensor([]int{shape.ModelDim, shape.ModelDim}, 0.04+offset)
		tensors[compactLayerTensorName(layer, "attn_k")] = compactTrainStateTestTensor([]int{shape.ModelDim, shape.ModelDim}, 0.05+offset)
		tensors[compactLayerTensorName(layer, "attn_v")] = compactTrainStateTestTensor([]int{shape.ModelDim, shape.ModelDim}, 0.06+offset)
		tensors[compactLayerTensorName(layer, "attn_o")] = compactTrainStateTestTensor([]int{shape.ModelDim, shape.ModelDim}, 0.07+offset)
		tensors[compactLayerTensorName(layer, "ffn_up")] = compactTrainStateTestTensor([]int{shape.ModelDim, shape.FFNDim}, 0.08+offset)
		tensors[compactLayerTensorName(layer, "ffn_down")] = compactTrainStateTestTensor([]int{shape.FFNDim, shape.ModelDim}, 0.09+offset)
	}
	if manifest.OutputProjectionParam != "" {
		tensors[manifest.OutputProjectionParam] = compactTrainStateTestTensor([]int{shape.ModelDim, shape.OutputDim}, 0.11)
	}
	moments := make(map[string]*backend.Tensor, len(tensors)*2)
	for name, tensor := range tensors {
		moments[name+"_moment_1"] = zeroLikeMaster(tensor)
		moments[name+"_moment_2"] = zeroLikeMaster(tensor)
	}
	return EmbeddingTrainCheckpoint{
		Version:       EmbeddingTrainCheckpointVersion,
		Manifest:      manifest,
		Config:        EmbeddingTrainConfig{Optimizer: "adamw"},
		Tensors:       tensors,
		MomentTensors: moments,
	}
}

func compactResidentCUDAWarmBenchBatch(shape compactResidentCUDAWarmBenchShape) []EmbeddingTokenizedVectorDistillExample {
	batch := make([]EmbeddingTokenizedVectorDistillExample, shape.Batch)
	for row := range batch {
		tokens := make([]int32, shape.Tokens)
		mask := make([]int32, shape.Tokens)
		for pos := range tokens {
			// The affine pattern is deterministic and gives each row a distinct
			// exact-length sequence for the default and descriptor vocabularies.
			tokens[pos] = int32(1 + ((row*31 + pos*17) % (shape.Vocab - 1)))
			mask[pos] = 1
		}
		teacher := make([]float32, shape.OutputDim)
		for col := range teacher {
			teacher[col] = float32(((row+col)%17)-8) * 0.01
		}
		batch[row] = EmbeddingTokenizedVectorDistillExample{
			ID:            fmt.Sprintf("cuda-warm-%d", row),
			Tokens:        tokens,
			Mask:          mask,
			TeacherVector: teacher,
			Role:          EmbeddingRoleQuery,
		}
	}
	return batch
}

func compactResidentCUDAWarmBenchCompactStats(profile EmbeddingTrainProfile) *backend.CompactTrainAcceleratorStats {
	return profile.CompactTrain
}

func compactResidentCUDAWarmBenchExpectedKernelCounts(shape compactResidentCUDAWarmBenchShape) (forward, backward, whole int64) {
	// K7's exact profiles are the only accepted benchmark shapes. P is one
	// when the output projection is present, so F=2+10L+2P yields 24/22.
	if shape.OutputDim != shape.ModelDim {
		return 24, 44, 68
	}
	return 22, 43, 65
}

func assertCompactResidentCUDAWarmBenchGraphStep(b *testing.B, label string, stats *backend.CompactTrainAcceleratorStats, shape compactResidentCUDAWarmBenchShape, graphEnabled, measured bool) {
	b.Helper()
	if stats == nil {
		b.Fatalf("%s compact-train stats are nil", label)
	}
	forward, backward, whole := compactResidentCUDAWarmBenchExpectedKernelCounts(shape)
	wantCaptures, wantReplays, wantGraphLaunches, wantGraphNodes := int64(0), int64(0), int64(0), int64(0)
	wantDirectForward, wantGraphExecuted := forward, int64(0)
	wantWholeDirect := whole
	if graphEnabled {
		if !measured {
			wantCaptures, wantGraphNodes = 1, forward
		} else {
			wantReplays, wantGraphLaunches = 1, 1
			wantDirectForward, wantGraphExecuted = 0, forward
			wantWholeDirect = backward
		}
	}
	if stats.GraphCaptures != wantCaptures || stats.GraphReplays != wantReplays || stats.GraphLaunches != wantGraphLaunches || stats.GraphNodes != wantGraphNodes {
		b.Fatalf("%s graph capture/replay telemetry = captures=%d replays=%d launches=%d nodes=%d, want %d/%d/%d/%d (graph_enabled=%t)", label, stats.GraphCaptures, stats.GraphReplays, stats.GraphLaunches, stats.GraphNodes, wantCaptures, wantReplays, wantGraphLaunches, wantGraphNodes, graphEnabled)
	}
	if stats.DirectForwardSubmissions != wantDirectForward || stats.LastForwardDirectSubmissions != wantDirectForward {
		b.Fatalf("%s direct forward submissions = cumulative=%d last=%d, want %d/%d", label, stats.DirectForwardSubmissions, stats.LastForwardDirectSubmissions, wantDirectForward, wantDirectForward)
	}
	if stats.GraphExecutedNodes != wantGraphExecuted || stats.ForwardDeviceKernelWork != forward || stats.LastForwardDeviceKernelWork != forward {
		b.Fatalf("%s forward device work = graph_executed=%d cumulative=%d last=%d, want %d/%d/%d", label, stats.GraphExecutedNodes, stats.ForwardDeviceKernelWork, stats.LastForwardDeviceKernelWork, wantGraphExecuted, forward, forward)
	}
	if stats.GraphCaptureFailures != 0 || stats.GraphReplayFailures != 0 || stats.GraphInvalidations != 0 || stats.GraphParityFailures != 0 || stats.GraphFallbacks != 0 || stats.GraphSynchronizations != 0 {
		b.Fatalf("%s graph failure/fallback/sync telemetry = capture_failures=%d replay_failures=%d invalidations=%d parity_failures=%d fallbacks=%d graph_syncs=%d, want all zero", label, stats.GraphCaptureFailures, stats.GraphReplayFailures, stats.GraphInvalidations, stats.GraphParityFailures, stats.GraphFallbacks, stats.GraphSynchronizations)
	}
	if stats.KernelLaunches != wantWholeDirect || stats.KernelSynchronizations != 2 || stats.KernelLaunches+stats.GraphExecutedNodes != whole {
		b.Fatalf("%s compact direct/device work = direct_kernels=%d compact_syncs=%d direct_plus_graph=%d, want %d/2/%d", label, stats.KernelLaunches, stats.KernelSynchronizations, stats.KernelLaunches+stats.GraphExecutedNodes, wantWholeDirect, whole)
	}
	if stats.ForwardReadbackBatchEntries != 1 || stats.ForwardReadbackContextSets != 1 || stats.ForwardReadbackDeviceCopies != 3 {
		b.Fatalf("%s K6 readback telemetry = %d/%d/%d, want 1/1/3", label, stats.ForwardReadbackBatchEntries, stats.ForwardReadbackContextSets, stats.ForwardReadbackDeviceCopies)
	}
}

func assertCompactResidentCUDAWarmBenchK5(b *testing.B, label string, stats backend.OptimizerAcceleratorStats) {
	b.Helper()
	if stats.ResidentGradBatchCalls != 1 ||
		stats.ResidentGradBatchKernelSyncs != 1 ||
		stats.ResidentGradBatchKernelLaunches != stats.ResidentGradUpdateCalls ||
		stats.ResidentGradUpdateCalls <= 0 {
		b.Fatalf("%s K5 batch telemetry = calls=%d kernel_launches=%d kernel_syncs=%d resident_grad_updates=%d; want calls=1, syncs=1, launches=updates>0", label, stats.ResidentGradBatchCalls, stats.ResidentGradBatchKernelLaunches, stats.ResidentGradBatchKernelSyncs, stats.ResidentGradUpdateCalls)
	}
}

func closeCompactResidentCUDAWarmBenchAccelerator(accel backend.CompactTrainAccelerator) {
	if closer, ok := accel.(interface{ Close() }); ok {
		closer.Close()
	}
}
