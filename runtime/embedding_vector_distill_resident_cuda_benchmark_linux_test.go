//go:build linux && cgo

package eosruntime

import (
	"encoding/json"
	"fmt"
	"os"
	"strconv"
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
const compactResidentCUDAWarmBenchUploadBatchEnv = "EOS_CUDA_COMPACT_TRAIN_FORWARD_UPLOAD_BATCH"

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
	graphEnabled := compactResidentCUDAWarmBenchFlagEnabled("EOS_CUDA_COMPACT_TRAIN_FORWARD_GRAPH")
	uploadBatchEnabled := compactResidentCUDAWarmBenchFlagEnabled(compactResidentCUDAWarmBenchUploadBatchEnv)

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
	assertCompactResidentCUDAWarmBenchInputUpload(b, "warmup", warmStats, uploadBatchEnabled, false)
	assertCompactResidentCUDAWarmBenchK5(b, "warmup", warmDelta.Optimizer, shape)
	b.Logf("scope=diagnostic_only warmup_excluded=true warmup_steps=1 profile=%s shape=%s graph_enabled=%t input_upload_batch_enabled=%t", compactResidentCUDAWarmBenchProfileLabel(shape), shape.String(), graphEnabled, uploadBatchEnabled)
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
	assertCompactResidentCUDAWarmBenchInputUpload(b, "measured", stats, uploadBatchEnabled, true)
	optimizerStats := measuredDelta.Optimizer
	assertCompactResidentCUDAWarmBenchK5(b, "measured", optimizerStats, shape)
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
	b.ReportMetric(perStep(stats.ForwardInputUploadBatchCalls), "forward_input_upload_batch_calls/step")
	b.ReportMetric(perStep(stats.ForwardInputUploadContextSets), "forward_input_upload_context_sets/step")
	b.ReportMetric(perStep(stats.ForwardInputUploadDeviceCopies), "forward_input_upload_device_copies/step")
	b.ReportMetric(perStep(stats.ForwardInputUploadFailures), "forward_input_upload_failures/step")
	b.ReportMetric(perStep(stats.ForwardInputUploadScalarFallbacks), "forward_input_upload_scalar_fallbacks/step")
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
	b.Logf("scope=diagnostic_only warmup_excluded=true measured_steps=%d shape=%s profile=%s graph_enabled=%t input_upload_batch_enabled=%t wall_ns=%d", b.N, shape.String(), compactResidentCUDAWarmBenchProfileLabel(shape), graphEnabled, uploadBatchEnabled, measuredElapsed.Nanoseconds())
	b.Logf("measured_input_upload_counters calls=%d context_sets=%d device_copies=%d failures=%d scalar_fallbacks=%d", stats.ForwardInputUploadBatchCalls, stats.ForwardInputUploadContextSets, stats.ForwardInputUploadDeviceCopies, stats.ForwardInputUploadFailures, stats.ForwardInputUploadScalarFallbacks)
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
		name                               string
		profile                            string
		wantForward, wantBack, wantWhole   int64
		wantForwardUpload, wantWholeUpload int64
		wantK5Updates                      int64
	}{
		{name: "canonical", profile: "canonical", wantForward: 24, wantBack: 44, wantWhole: 68, wantForwardUpload: 148, wantWholeUpload: 196, wantK5Updates: 15},
		{name: "next", profile: "next", wantForward: 22, wantBack: 43, wantWhole: 65, wantForwardUpload: 2056, wantWholeUpload: 2568, wantK5Updates: 14},
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
			forwardUpload, _, wholeUpload := compactResidentCUDAWarmBenchExpectedUploadBytes(shape)
			if forwardUpload != tc.wantForwardUpload || wholeUpload != tc.wantWholeUpload {
				t.Fatalf("profile %s exact uploads = forward/whole %d/%d, want %d/%d", tc.profile, forwardUpload, wholeUpload, tc.wantForwardUpload, tc.wantWholeUpload)
			}
			if k5Updates := compactResidentCUDAWarmBenchExpectedK5Updates(shape); k5Updates != tc.wantK5Updates {
				t.Fatalf("profile %s exact K5 updates = %d, want %d", tc.profile, k5Updates, tc.wantK5Updates)
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

func compactResidentCUDAWarmBenchFlagEnabled(name string) bool {
	switch strings.ToLower(strings.TrimSpace(os.Getenv(name))) {
	case "1", "true", "yes":
		return true
	default:
		return false
	}
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
	wantForwardUpload, wantGradUpload, wantCompactUpload := compactResidentCUDAWarmBenchExpectedUploadBytes(shape)
	if stats.UploadedBytes != wantCompactUpload || stats.GradPooledUploadedBytes != wantGradUpload || stats.UploadedBytes-stats.GradPooledUploadedBytes != wantForwardUpload {
		b.Fatalf("%s compact H2D bytes = whole=%d forward=%d grad=%d, want whole=%d forward=%d grad=%d", label, stats.UploadedBytes, stats.UploadedBytes-stats.GradPooledUploadedBytes, stats.GradPooledUploadedBytes, wantCompactUpload, wantForwardUpload, wantGradUpload)
	}
	if stats.ForwardReadbackBatchEntries != 1 || stats.ForwardReadbackContextSets != 1 || stats.ForwardReadbackDeviceCopies != 3 {
		b.Fatalf("%s K6 readback telemetry = %d/%d/%d, want 1/1/3", label, stats.ForwardReadbackBatchEntries, stats.ForwardReadbackContextSets, stats.ForwardReadbackDeviceCopies)
	}
}

func compactResidentCUDAWarmBenchExpectedUploadBytes(shape compactResidentCUDAWarmBenchShape) (forward, grad, whole int64) {
	forward = int64((2*shape.Batch*shape.Tokens + shape.Batch + 1) * 4)
	grad = int64(shape.Batch * shape.OutputDim * 4)
	return forward, grad, forward + grad
}

func compactResidentCUDAWarmBenchExpectedK5Updates(shape compactResidentCUDAWarmBenchShape) int64 {
	if shape.OutputDim == shape.ModelDim {
		return 14
	}
	return 15
}

func assertCompactResidentCUDAWarmBenchInputUpload(b *testing.B, label string, stats *backend.CompactTrainAcceleratorStats, enabled, measured bool) {
	b.Helper()
	if stats == nil {
		b.Fatalf("%s compact-train stats are nil", label)
	}
	if !enabled {
		if stats.ForwardInputUploadBatchCalls != 0 || stats.ForwardInputUploadContextSets != 0 || stats.ForwardInputUploadDeviceCopies != 0 || stats.ForwardInputUploadFailures != 0 || stats.ForwardInputUploadScalarFallbacks != 0 {
			b.Fatalf("%s input-upload counters with flag off = %d/%d/%d/%d/%d, want 0/0/0/0/0", label, stats.ForwardInputUploadBatchCalls, stats.ForwardInputUploadContextSets, stats.ForwardInputUploadDeviceCopies, stats.ForwardInputUploadFailures, stats.ForwardInputUploadScalarFallbacks)
		}
		return
	}
	if !measured {
		if stats.ForwardInputUploadBatchCalls != 0 || stats.ForwardInputUploadContextSets != 0 || stats.ForwardInputUploadDeviceCopies != 0 || stats.ForwardInputUploadFailures != 0 || stats.ForwardInputUploadScalarFallbacks <= 0 {
			b.Fatalf("%s cold input-upload counters with flag on = %d/%d/%d/%d/%d, want 0/0/0/0/scalar>0", label, stats.ForwardInputUploadBatchCalls, stats.ForwardInputUploadContextSets, stats.ForwardInputUploadDeviceCopies, stats.ForwardInputUploadFailures, stats.ForwardInputUploadScalarFallbacks)
		}
		return
	}
	if stats.ForwardInputUploadBatchCalls != 1 || stats.ForwardInputUploadContextSets != 1 || stats.ForwardInputUploadDeviceCopies != 4 || stats.ForwardInputUploadFailures != 0 || stats.ForwardInputUploadScalarFallbacks != 0 {
		b.Fatalf("%s warm input-upload counters with flag on = %d/%d/%d/%d/%d, want 1/1/4/0/0", label, stats.ForwardInputUploadBatchCalls, stats.ForwardInputUploadContextSets, stats.ForwardInputUploadDeviceCopies, stats.ForwardInputUploadFailures, stats.ForwardInputUploadScalarFallbacks)
	}
}

func assertCompactResidentCUDAWarmBenchK5(b *testing.B, label string, stats backend.OptimizerAcceleratorStats, shape compactResidentCUDAWarmBenchShape) {
	b.Helper()
	wantUpdates := compactResidentCUDAWarmBenchExpectedK5Updates(shape)
	if stats.ResidentGradBatchCalls != 1 ||
		stats.ResidentGradBatchKernelSyncs != 1 ||
		stats.ResidentGradBatchKernelLaunches != stats.ResidentGradUpdateCalls ||
		stats.ResidentGradUpdateCalls != wantUpdates {
		b.Fatalf("%s K5 batch telemetry = calls=%d kernel_launches=%d kernel_syncs=%d resident_grad_updates=%d; want calls=1, launches=updates=%d, syncs=1", label, stats.ResidentGradBatchCalls, stats.ResidentGradBatchKernelLaunches, stats.ResidentGradBatchKernelSyncs, stats.ResidentGradUpdateCalls, wantUpdates)
	}
}

func closeCompactResidentCUDAWarmBenchAccelerator(accel backend.CompactTrainAccelerator) {
	if closer, ok := accel.(interface{ Close() }); ok {
		closer.Close()
	}
}

// BenchmarkVectorDistillCompactResidentTrainCUDAK9TelemetryMatrix is the
// process-isolated accounting gate used by cmd/eos-k9-matrix. It deliberately
// has a separate name and gate so the original one-step warm benchmark above
// keeps its historical b.N==1 behavior and benchmark-local environment setup.
const (
	compactResidentCUDAK9MatrixGate       = "EOS_RUN_COMPACT_RESIDENT_CUDA_K9_MATRIX"
	compactResidentCUDAK9MeasuredStepsEnv = "EOS_CUDA_K9_MEASURED_STEPS"
	compactResidentCUDAK9Schema           = "eos-k9-cuda-telemetry/v1"
	compactResidentCUDAK9SchemaVersion    = 1
)

type compactResidentCUDAK9Modes struct {
	Graph         int `json:"graph"`
	Upload        int `json:"upload"`
	Events        int `json:"events"`
	CUBLAS        int `json:"cublas"`
	SyncEach      int `json:"sync_each"`
	ResidentTrain int `json:"resident_train"`
	PackedForward int `json:"packed_forward"`
}

type compactResidentCUDAK9Step struct {
	WallNanos         int64
	TopLevelHostNanos int64
	DeviceEventNanos  int64
	ResidualFraction  float64
}

type compactResidentCUDAK9Warmup struct {
	ColdSteps          int `json:"cold_steps"`
	ReplayPrimingSteps int `json:"replay_priming_steps"`
}

type compactResidentCUDAK9Result struct {
	Schema               string                                        `json:"schema"`
	SchemaVersion        int                                           `json:"schema_version"`
	Profile              string                                        `json:"profile"`
	Shape                string                                        `json:"shape"`
	Modes                compactResidentCUDAK9Modes                    `json:"modes"`
	Graph                int                                           `json:"graph"`
	Upload               int                                           `json:"upload"`
	Events               int                                           `json:"events"`
	WarmupSteps          int                                           `json:"warmup_steps"`
	Warmup               compactResidentCUDAK9Warmup                   `json:"warmup"`
	MeasuredSteps        int                                           `json:"measured_steps"`
	WallNanosPerStep     []int64                                       `json:"wall_ns_per_step"`
	TopLevelHostNanos    []int64                                       `json:"top_level_host_ns"`
	DeviceEventNanos     []int64                                       `json:"device_event_ns,omitempty"`
	DispatchHostNanos    []int64                                       `json:"dispatch_host_ns"`
	ResidualFractions    []float64                                     `json:"residual_fraction"`
	LegacyStats          *backend.CompactTrainAcceleratorStats         `json:"legacy_stats"`
	TelemetryOwner       *backend.CompactTrainTelemetry                `json:"telemetry_owner,omitempty"`
	TelemetryMirror      *backend.CompactTrainTelemetry                `json:"telemetry_mirror,omitempty"`
	TelemetryMirrorK5    *backend.CompactTrainPhaseTelemetry           `json:"optimizer_k5_mirror,omitempty"`
	TelemetryMirrorEqual bool                                          `json:"telemetry_mirror_equal"`
	PhaseTelemetry       map[string]backend.CompactTrainPhaseTelemetry `json:"phase_telemetry,omitempty"`
}

func BenchmarkVectorDistillCompactResidentTrainCUDAK9TelemetryMatrix(b *testing.B) {
	b.StopTimer()
	if strings.TrimSpace(os.Getenv(compactResidentCUDAK9MatrixGate)) != "1" {
		b.Skipf("opt-in benchmark: set %s=1", compactResidentCUDAK9MatrixGate)
	}
	if b.N != 1 {
		b.Fatalf("K9 telemetry benchmark requires exactly one Go benchmark iteration; rerun with -benchtime=1x (b.N=%d)", b.N)
	}
	modes, err := compactResidentCUDAK9ModesFromEnv()
	if err != nil {
		b.Fatal(err)
	}
	measuredSteps, err := compactResidentCUDAK9MeasuredSteps()
	if err != nil {
		b.Fatal(err)
	}
	shape, err := compactResidentCUDAWarmBenchShapeFromEnv()
	if err != nil {
		b.Fatalf("K9 telemetry benchmark shape: %v", err)
	}

	// Probe the factory before constructing the trainer. All controls above
	// must already have been present during package initialization; this path
	// intentionally never calls b.Setenv.
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

	checkpoint := compactResidentCUDAWarmBenchCheckpoint(shape)
	state, err := LoadCompactEmbeddingTrainStateFromCheckpoint(checkpoint, checkpoint.Manifest)
	if err != nil {
		b.Fatalf("load K9 telemetry benchmark state: %v", err)
	}
	trainer, err := newCompactEmbeddingTrainerFromTrainState(&eosartifact.Module{Name: "compact-k9-telemetry-bench"}, state)
	if err != nil {
		b.Fatalf("new K9 telemetry benchmark trainer: %v", err)
	}
	defer trainer.Close()
	if trainer.compactTrainAccel == nil || trainer.compactTrainBackend != eosartifact.BackendCUDA || trainer.compactTrainAccel.Backend() != eosartifact.BackendCUDA {
		b.Fatalf("K9 telemetry benchmark selected non-CUDA resident train (configured=%q accel=%T)", trainer.compactTrainBackend, trainer.compactTrainAccel)
	}

	batch := compactResidentCUDAWarmBenchBatch(shape)
	var scratch vectorDistillBatchScratch
	var projection *vectorDistillProjectionState
	initialProfile := trainer.TrainProfile()

	// Exactly one cold warmup establishes arenas, gradients, and (when events
	// are enabled) the first event pairs. Graph capture is also excluded here.
	warmProfileBefore := initialProfile
	if _, projection, err = trainer.trainVectorDistillBatchWithScratch(batch, projection, shape.OutputDim, EmbeddingRoleQuery, 0, &scratch); err != nil {
		b.Fatalf("K9 telemetry cold warmup: %v", err)
	}
	coldWarmProfile := trainer.TrainProfile()
	coldWarmDelta := diffTrainProfile(warmProfileBefore, coldWarmProfile)
	compactResidentCUDAK9AssertWarmup(b, "cold warmup", coldWarmDelta, shape, modes, false)

	primingSteps := 0
	if modes.Graph == 1 && modes.Events == 1 {
		// The graph replay event pair is first created on this excluded replay.
		// Priming it here keeps every measured replay step identical.
		before := trainer.TrainProfile()
		if _, projection, err = trainer.trainVectorDistillBatchWithScratch(batch, projection, shape.OutputDim, EmbeddingRoleQuery, 0, &scratch); err != nil {
			b.Fatalf("K9 telemetry replay-priming warmup: %v", err)
		}
		after := trainer.TrainProfile()
		primeDelta := diffTrainProfile(before, after)
		compactResidentCUDAK9AssertWarmup(b, "replay-priming warmup", primeDelta, shape, modes, true)
		primingSteps = 1
	}
	warmProfile := trainer.TrainProfile()

	observations := make([]compactResidentCUDAK9Step, 0, measuredSteps)
	b.ResetTimer()
	b.StopTimer()
	for i := 0; i < measuredSteps; i++ {
		// Profile snapshots are intentionally outside the wall timer. This keeps
		// JSON cloning/diffing and telemetry map work out of the measured delta.
		stepBefore := trainer.TrainProfile()
		b.StartTimer()
		start := time.Now()
		if _, projection, err = trainer.trainVectorDistillBatchWithScratch(batch, projection, shape.OutputDim, EmbeddingRoleQuery, 0, &scratch); err != nil {
			b.StopTimer()
			b.Fatalf("K9 telemetry measured step %d/%d: %v", i+1, measuredSteps, err)
		}
		wallNanos := time.Since(start).Nanoseconds()
		stepAfter := trainer.TrainProfile()
		b.StopTimer()
		stepDelta := diffTrainProfile(stepBefore, stepAfter)
		hostNanos := compactResidentCUDAK9TopLevelHostNanos(stepDelta.VectorDistillPhases)
		deviceNanos := compactResidentCUDAK9DeviceNanos(stepDelta.CompactTrain, modes.Graph == 1)
		if wallNanos <= 0 || hostNanos < 0 {
			b.Fatalf("K9 telemetry measured step %d has invalid wall/host time %d/%d", i+1, wallNanos, hostNanos)
		}
		if modes.Events == 1 && hostNanos < deviceNanos {
			b.Fatalf("K9 telemetry measured step %d has negative dispatch time: host=%d device=%d", i+1, hostNanos, deviceNanos)
		}
		residual := float64(absInt64(wallNanos-hostNanos)) / float64(wallNanos)
		if residual > 0.05 {
			b.Fatalf("K9 telemetry measured step %d residual %.6f exceeds 0.05 (wall=%d host=%d device=%d)", i+1, residual, wallNanos, hostNanos, deviceNanos)
		}
		observations = append(observations, compactResidentCUDAK9Step{
			WallNanos: wallNanos, TopLevelHostNanos: hostNanos,
			DeviceEventNanos: deviceNanos, ResidualFraction: residual,
		})
	}
	b.StopTimer()

	finalProfile := trainer.TrainProfile()
	measuredDelta := diffTrainProfile(warmProfile, finalProfile)
	stats := measuredDelta.CompactTrain
	if stats == nil {
		b.Fatalf("K9 telemetry measured profile has no compact-train stats")
	}
	compactResidentCUDAK9AssertMeasured(b, measuredDelta, stats, shape, modes, measuredSteps)

	wall := make([]int64, len(observations))
	host := make([]int64, len(observations))
	device := make([]int64, 0, len(observations))
	dispatch := make([]int64, len(observations))
	residual := make([]float64, len(observations))
	for i, observation := range observations {
		wall[i] = observation.WallNanos
		host[i] = observation.TopLevelHostNanos
		dispatch[i] = observation.TopLevelHostNanos - observation.DeviceEventNanos
		residual[i] = observation.ResidualFraction
		if modes.Events == 1 {
			device = append(device, observation.DeviceEventNanos)
		}
	}
	owner := (*backend.CompactTrainTelemetry)(nil)
	var mirror *backend.CompactTrainTelemetry
	if stats != nil {
		owner = stats.Telemetry
	}
	mirror = measuredDelta.Optimizer.CompactTrainTelemetry
	phaseTelemetry := compactResidentCUDAK9PhaseMap(owner)
	result := compactResidentCUDAK9Result{
		Schema: compactResidentCUDAK9Schema, SchemaVersion: compactResidentCUDAK9SchemaVersion,
		Profile: compactResidentCUDAK9ProfileLabel(shape), Shape: shape.String(), Modes: modes,
		Graph: modes.Graph, Upload: modes.Upload, Events: modes.Events,
		WarmupSteps:   1 + primingSteps,
		Warmup:        compactResidentCUDAK9Warmup{ColdSteps: 1, ReplayPrimingSteps: primingSteps},
		MeasuredSteps: measuredSteps, WallNanosPerStep: wall, TopLevelHostNanos: host,
		DeviceEventNanos: device, DispatchHostNanos: dispatch, ResidualFractions: residual,
		LegacyStats: stats, TelemetryOwner: owner, TelemetryMirror: mirror,
		PhaseTelemetry: phaseTelemetry,
	}
	if owner != nil && mirror != nil {
		ownerK5 := owner.Phase(backend.CompactTrainPhaseK5Batch)
		mirrorK5 := mirror.Phase(backend.CompactTrainPhaseK5Batch)
		result.TelemetryMirrorK5 = &mirrorK5
		result.TelemetryMirrorEqual = ownerK5 == mirrorK5
		if !result.TelemetryMirrorEqual {
			b.Fatalf("K9 telemetry optimizer mirror K5 differs from owner: owner=%+v mirror=%+v", ownerK5, mirrorK5)
		}
		if compactResidentCUDAK9TelemetryOnlyK5(owner, mirror) {
			b.Fatalf("K9 telemetry owner and optimizer mirror unexpectedly contain only the same K5 view")
		}
	} else if owner != nil || mirror != nil {
		b.Fatalf("K9 telemetry owner/mirror nil mismatch: owner=%v mirror=%v", owner != nil, mirror != nil)
	}
	if modes.Events == 0 && result.TelemetryMirrorEqual {
		b.Fatalf("K9 event-off telemetry mirror unexpectedly equal")
	}
	encoded, err := json.Marshal(result)
	if err != nil {
		b.Fatalf("marshal K9 telemetry result: %v", err)
	}
	// This is intentionally the sole machine-readable record emitted by the
	// process, and it is printed only after every identity assertion passes.
	fmt.Printf("k9_result_json=%s\n", encoded)
}

func compactResidentCUDAK9ModesFromEnv() (compactResidentCUDAK9Modes, error) {
	flag := func(name string) (int, error) {
		value := strings.TrimSpace(os.Getenv(name))
		if value != "0" && value != "1" {
			return 0, fmt.Errorf("K9 requires %s to be process-start env 0 or 1 (got %q)", name, value)
		}
		parsed, _ := strconv.Atoi(value)
		return parsed, nil
	}
	var modes compactResidentCUDAK9Modes
	var err error
	if modes.Graph, err = flag("EOS_CUDA_COMPACT_TRAIN_FORWARD_GRAPH"); err != nil {
		return modes, err
	}
	if modes.Upload, err = flag(compactResidentCUDAWarmBenchUploadBatchEnv); err != nil {
		return modes, err
	}
	if modes.Events, err = flag("EOS_CUDA_COMPACT_PROFILE_EVENTS"); err != nil {
		return modes, err
	}
	if modes.CUBLAS, err = flag("EOS_CUDA_COMPACT_TRAIN_CUBLAS"); err != nil {
		return modes, err
	}
	if modes.SyncEach, err = flag("EOS_CUDA_COMPACT_SYNC_EACH_LAUNCH"); err != nil {
		return modes, err
	}
	resident, residentErr := flag(compactResidentTrainEnv)
	if residentErr != nil {
		return modes, residentErr
	}
	packed, packedErr := flag(compactPackedForwardEnv)
	if packedErr != nil {
		return modes, packedErr
	}
	modes.ResidentTrain, modes.PackedForward = resident, packed
	if modes.CUBLAS != 0 || modes.SyncEach != 0 {
		return modes, fmt.Errorf("K9 requires EOS_CUDA_COMPACT_TRAIN_CUBLAS=0 and EOS_CUDA_COMPACT_SYNC_EACH_LAUNCH=0 (got %d/%d)", modes.CUBLAS, modes.SyncEach)
	}
	if modes.ResidentTrain != 1 || modes.PackedForward != 0 {
		return modes, fmt.Errorf("K9 requires resident_train=1 and packed_forward=0 (got %d/%d)", modes.ResidentTrain, modes.PackedForward)
	}
	return modes, nil
}

func compactResidentCUDAK9MeasuredSteps() (int, error) {
	const defaultSteps, maxSteps = 20, 1000
	value := strings.TrimSpace(os.Getenv(compactResidentCUDAK9MeasuredStepsEnv))
	if value == "" {
		return defaultSteps, nil
	}
	steps, err := strconv.Atoi(value)
	if err != nil || steps < 1 || steps > maxSteps {
		return 0, fmt.Errorf("%s must be a decimal integer in [1,%d] (got %q)", compactResidentCUDAK9MeasuredStepsEnv, maxSteps, value)
	}
	return steps, nil
}

func compactResidentCUDAK9ProfileLabel(shape compactResidentCUDAWarmBenchShape) string {
	switch shape.Profile {
	case "canonical_fixture":
		return "canonical"
	case "next_descriptor_shape":
		return "next"
	default:
		return shape.Profile
	}
}

func compactResidentCUDAK9TopLevelHostNanos(phases EmbeddingVectorDistillPhaseTimers) int64 {
	return phases.EncodeNanos + phases.ProjectionLossNanos + phases.BackwardNanos + phases.OptimizerNanos
}

func compactResidentCUDAK9DeviceNanos(stats *backend.CompactTrainAcceleratorStats, graph bool) int64 {
	if stats == nil || stats.Telemetry == nil {
		return 0
	}
	forward := backend.CompactTrainPhaseForwardDirect
	if graph {
		forward = backend.CompactTrainPhaseForwardReplay
	}
	return stats.Telemetry.Phase(forward).DeviceElapsedNanos +
		stats.Telemetry.Phase(backend.CompactTrainPhaseBackwardFinal).DeviceElapsedNanos +
		stats.Telemetry.Phase(backend.CompactTrainPhaseBackwardFFN).DeviceElapsedNanos +
		stats.Telemetry.Phase(backend.CompactTrainPhaseBackwardAttention).DeviceElapsedNanos +
		stats.Telemetry.Phase(backend.CompactTrainPhaseK5Batch).DeviceElapsedNanos
}

func compactResidentCUDAK9AssertWarmup(b *testing.B, label string, delta EmbeddingTrainProfile, shape compactResidentCUDAWarmBenchShape, modes compactResidentCUDAK9Modes, replay bool) {
	b.Helper()
	stats := delta.CompactTrain
	if stats == nil {
		b.Fatalf("%s has no compact-train stats", label)
	}
	if stats.FallbackOrUnhandled != 0 || stats.CublasGemmCalls != 0 {
		b.Fatalf("%s fallback/cublas counters = %d/%d, want 0/0", label, stats.FallbackOrUnhandled, stats.CublasGemmCalls)
	}
	if stats.LastForwardCublasGemmCalls != 0 || stats.LastBackwardCublasGemmCalls != 0 {
		b.Fatalf("%s last-step cublas counters = %d/%d, want 0/0", label, stats.LastForwardCublasGemmCalls, stats.LastBackwardCublasGemmCalls)
	}
	assertCompactResidentCUDAWarmBenchGraphStep(b, label, stats, shape, modes.Graph == 1, replay)
	if replay {
		// The helper's measured form checks one replay and no capture. The
		// first warmup is the only capture and is already checked below.
		if stats.GraphCaptures != 0 || stats.GraphReplays != 1 || stats.GraphLaunches != 1 || stats.GraphNodes != 0 {
			b.Fatalf("%s replay graph counters = %d/%d/%d/%d, want 0/1/1/0", label, stats.GraphCaptures, stats.GraphReplays, stats.GraphLaunches, stats.GraphNodes)
		}
	}
	assertCompactResidentCUDAWarmBenchInputUpload(b, label, stats, modes.Upload == 1, replay)
	assertCompactResidentCUDAWarmBenchK5(b, label, delta.Optimizer, shape)
	if modes.Events == 0 {
		if stats.Telemetry != nil || delta.Optimizer.CompactTrainTelemetry != nil {
			b.Fatalf("%s event-off telemetry unexpectedly present", label)
		}
		return
	}
	if stats.Telemetry == nil || !stats.Telemetry.Enabled {
		b.Fatalf("%s event-on owner telemetry missing", label)
	}
	if !replay && modes.Graph == 1 {
		capture := stats.Telemetry.Phase(backend.CompactTrainPhaseForwardCapture)
		forward, _, _ := compactResidentCUDAWarmBenchExpectedKernelCounts(shape)
		if capture.GraphBegin != 1 || capture.GraphEnd != 1 || capture.GraphInstantiate != 1 || capture.GraphNodes != forward || capture.GraphKernelNodes != forward || capture.GraphCublasNodes != 0 || capture.GraphUnknownNodes != 0 || capture.DriverCalls != forward+3 || capture.Attempted != forward+3 || capture.Completed != forward+3 || capture.Failures != 0 {
			b.Fatalf("%s capture telemetry = %+v, want exact F=%d capture identity", label, capture, forward)
		}
	}
}

func compactResidentCUDAK9AssertMeasured(b *testing.B, delta EmbeddingTrainProfile, stats *backend.CompactTrainAcceleratorStats, shape compactResidentCUDAWarmBenchShape, modes compactResidentCUDAK9Modes, steps int) {
	b.Helper()
	if steps <= 0 || stats == nil {
		b.Fatalf("K9 measured assertion has invalid steps/stats %d/%v", steps, stats)
	}
	expect := func(name string, got, perStep int64) {
		if got != perStep*int64(steps) {
			b.Fatalf("measured %s=%d, want %d (%d/step x %d)", name, got, perStep*int64(steps), perStep, steps)
		}
	}
	expectLast := func(name string, got, want int64) {
		if got != want {
			b.Fatalf("measured %s last=%d, want %d", name, got, want)
		}
	}
	forward, backward, whole := compactResidentCUDAWarmBenchExpectedKernelCounts(shape)
	if modes.Graph == 1 {
		whole = backward
	}
	wantForward, wantGrad, wantWholeUpload := compactResidentCUDAWarmBenchExpectedUploadBytes(shape)
	// Trainer-level and compact aggregate counters.
	expect("forward_calls", stats.ForwardCalls, 1)
	expect("backward_calls", stats.BackwardCalls, 1)
	expect("handles_created", stats.HandlesCreated, 1)
	expect("handles_released", stats.HandlesReleased, 1)
	expect("arena_reuse_hits", stats.ArenaReuseHits, 1)
	expect("arena_allocations", stats.ArenaAllocations, 0)
	expect("gradient_reuse_hits", stats.GradientReuseHits, 1)
	expect("gradient_allocations", stats.GradientAllocations, 0)
	expect("gradient_zero_calls", stats.GradientZeroCalls, 1)
	expect("uploaded_bytes", stats.UploadedBytes, wantWholeUpload)
	expect("grad_pooled_uploaded_bytes", stats.GradPooledUploadedBytes, wantGrad)
	expect("downloaded_bytes", stats.DownloadedBytes, int64(4+shape.Batch*shape.OutputDim*4+shape.Batch*4))
	expect("pooled_downloaded_bytes", stats.PooledDownloadedBytes, int64(shape.Batch*shape.OutputDim*4))
	wantStatusDownloaded := int64(8)
	if shape.OutputDim != shape.ModelDim {
		wantStatusDownloaded = 20
	}
	expect("status_downloaded_bytes", stats.StatusDownloadedBytes, wantStatusDownloaded)
	expect("readback_batch_entries", stats.ForwardReadbackBatchEntries, 1)
	expect("readback_context_sets", stats.ForwardReadbackContextSets, 1)
	expect("readback_device_copies", stats.ForwardReadbackDeviceCopies, 3)
	expect("kernel_launches", stats.KernelLaunches, whole)
	expect("kernel_syncs", stats.KernelSynchronizations, 2)
	expect("cublas_calls", stats.CublasGemmCalls, 0)
	if stats.LastForwardCublasGemmCalls != 0 || stats.LastBackwardCublasGemmCalls != 0 {
		b.Fatalf("measured last-step cublas counters = %d/%d, want 0/0", stats.LastForwardCublasGemmCalls, stats.LastBackwardCublasGemmCalls)
	}
	expect("graph_captures", stats.GraphCaptures, 0)
	expect("graph_replays", stats.GraphReplays, compactResidentCUDAK9BoolInt64(modes.Graph == 1))
	expect("graph_launches", stats.GraphLaunches, compactResidentCUDAK9BoolInt64(modes.Graph == 1))
	expect("graph_nodes", stats.GraphNodes, 0)
	expect("graph_executed_nodes", stats.GraphExecutedNodes, compactResidentCUDAK9BoolInt64(modes.Graph == 1)*forward)
	expect("direct_forward_submissions", stats.DirectForwardSubmissions, compactResidentCUDAK9BoolInt64(modes.Graph == 0)*forward)
	expect("forward_device_kernel_work", stats.ForwardDeviceKernelWork, forward)
	expect("fallback_or_unhandled", stats.FallbackOrUnhandled, 0)
	expect("graph_capture_failures", stats.GraphCaptureFailures, 0)
	expect("graph_replay_failures", stats.GraphReplayFailures, 0)
	expect("graph_invalidations", stats.GraphInvalidations, 0)
	expect("graph_parity_failures", stats.GraphParityFailures, 0)
	expect("graph_fallbacks", stats.GraphFallbacks, 0)
	expect("graph_syncs", stats.GraphSynchronizations, 0)
	if stats.LiveHandles != 0 {
		b.Fatalf("measured live handles=%d, want 0", stats.LiveHandles)
	}
	wantForwardUploadCalls := int64(0)
	wantForwardUploadContext := int64(0)
	wantForwardUploadCopies := int64(0)
	if modes.Upload == 1 {
		wantForwardUploadCalls, wantForwardUploadContext, wantForwardUploadCopies = 1, 1, 4
	}
	expect("forward_input_upload_batch_calls", stats.ForwardInputUploadBatchCalls, wantForwardUploadCalls)
	expect("forward_input_upload_context_sets", stats.ForwardInputUploadContextSets, wantForwardUploadContext)
	expect("forward_input_upload_device_copies", stats.ForwardInputUploadDeviceCopies, wantForwardUploadCopies)
	expect("forward_input_upload_failures", stats.ForwardInputUploadFailures, 0)
	expect("forward_input_upload_scalar_fallbacks", stats.ForwardInputUploadScalarFallbacks, 0)
	if stats.LastShape == (backend.CompactForwardShape{}) {
		b.Fatalf("measured last shape is zero, want %s", shape.String())
	}
	// LastShape is the backend's canonical fixture and is checked below by
	// field so a future addition to the shape struct does not weaken this gate
	// silently.
	if stats.LastShape.Batch != shape.Batch || stats.LastShape.Tokens != shape.Tokens || stats.LastShape.ModelDim != shape.ModelDim || stats.LastShape.FFNDim != shape.FFNDim || stats.LastShape.Heads != shape.Heads || stats.LastShape.HeadDim != shape.HeadDim || stats.LastShape.Layers != shape.Layers || stats.LastShape.OutputDim != shape.OutputDim || stats.LastShape.HasOutputProjection != (shape.OutputDim != shape.ModelDim) {
		b.Fatalf("measured last shape=%+v, want %s", stats.LastShape, shape.String())
	}
	expectLast("last_forward_launches", stats.LastForwardLaunches, func() int64 {
		if modes.Graph == 1 {
			return 0
		}
		return forward
	}())
	_, expectedBackward, _ := compactResidentCUDAWarmBenchExpectedKernelCounts(shape)
	expectLast("last_backward_launches", stats.LastBackwardLaunches, expectedBackward)
	expectLast("last_forward_syncs", stats.LastForwardSyncs, 1)
	expectLast("last_backward_syncs", stats.LastBackwardSyncs, 1)
	expectLast("last_forward_direct_submissions", stats.LastForwardDirectSubmissions, func() int64 {
		if modes.Graph == 1 {
			return 0
		}
		return forward
	}())
	expectLast("last_forward_device_kernel_work", stats.LastForwardDeviceKernelWork, forward)

	opt := delta.Optimizer
	expectOpt := func(name string, got, perStep int64) {
		if got != perStep*int64(steps) {
			b.Fatalf("measured optimizer %s=%d, want %d (%d/step x %d)", name, got, perStep*int64(steps), perStep, steps)
		}
	}
	wantK5 := compactResidentCUDAWarmBenchExpectedK5Updates(shape)
	expectOpt("resident_grad_batch_calls", opt.ResidentGradBatchCalls, 1)
	expectOpt("resident_grad_batch_kernel_launches", opt.ResidentGradBatchKernelLaunches, wantK5)
	expectOpt("resident_grad_batch_kernel_syncs", opt.ResidentGradBatchKernelSyncs, 1)
	expectOpt("resident_grad_update_calls", opt.ResidentGradUpdateCalls, wantK5)

	owner := stats.Telemetry
	mirror := opt.CompactTrainTelemetry
	if modes.Events == 0 {
		if owner != nil || mirror != nil {
			b.Fatalf("event-off measured telemetry owner/mirror = %v/%v, want nil/nil", owner, mirror)
		}
		return
	}
	if owner == nil || mirror == nil || !owner.Enabled || !owner.EventTiming || owner.View != backend.CompactTrainTelemetryViewCompactTrain || mirror.View != backend.CompactTrainTelemetryViewOptimizer || !mirror.Enabled || !mirror.EventTiming {
		b.Fatalf("event-on measured telemetry metadata owner=%+v mirror=%+v", owner, mirror)
	}
	compactResidentCUDAK9AssertTelemetryPhases(b, owner, shape, modes, steps)
	ownerK5 := owner.Phase(backend.CompactTrainPhaseK5Batch)
	mirrorK5 := mirror.Phase(backend.CompactTrainPhaseK5Batch)
	if ownerK5 != mirrorK5 {
		b.Fatalf("event-on optimizer mirror K5 = %+v, owner K5 = %+v", mirrorK5, ownerK5)
	}
	if compactResidentCUDAK9TelemetryOnlyK5(owner, mirror) {
		b.Fatalf("event-on owner/mirror telemetry unexpectedly equal as whole views")
	}
	if ownerK5.HostNanos <= 0 || ownerK5.DeviceElapsedNanos <= 0 {
		b.Fatalf("event-on K5 timing = %+v, want positive host/device", ownerK5)
	}
	_ = wantForward
}

func compactResidentCUDAK9AssertTelemetryPhases(b *testing.B, telemetry *backend.CompactTrainTelemetry, shape compactResidentCUDAWarmBenchShape, modes compactResidentCUDAK9Modes, steps int) {
	b.Helper()
	if telemetry == nil {
		b.Fatalf("event-on telemetry is nil")
	}
	perStep := func(value int64) int64 { return value * int64(steps) }
	phase := func(name backend.CompactTrainTelemetryPhase) backend.CompactTrainPhaseTelemetry {
		return telemetry.Phase(name)
	}
	exact := func(name string, got, want int64) {
		if got != want {
			b.Fatalf("event-on phase %s field=%d, want %d", name, got, want)
		}
	}
	exactPhase := func(name string, got backend.CompactTrainPhaseTelemetry, want backend.CompactTrainPhaseTelemetry) {
		got.HostNanos = 0
		got.DeviceElapsedNanos = 0
		if got != want {
			b.Fatalf("event-on phase %s=%+v, want %+v", name, got, want)
		}
	}
	forward, _, _ := compactResidentCUDAWarmBenchExpectedKernelCounts(shape)
	_, gradBytes, _ := compactResidentCUDAWarmBenchExpectedUploadBytes(shape)
	forwardBytes, _, _ := compactResidentCUDAWarmBenchExpectedUploadBytes(shape)
	if modes.Upload == 0 {
		input := phase(backend.CompactTrainPhaseInputH2D)
		exactPhase("input_h2d", input, backend.CompactTrainPhaseTelemetry{GoCalls: perStep(4), ContextSets: perStep(4), DriverCalls: perStep(4), H2DCopies: perStep(4), H2DBytes: perStep(forwardBytes), Attempted: perStep(4), Completed: perStep(4)})
	} else {
		input := phase(backend.CompactTrainPhaseInputH2D)
		exactPhase("input_h2d", input, backend.CompactTrainPhaseTelemetry{GoCalls: perStep(1), ContextSets: perStep(1), DriverCalls: perStep(4), H2DCopies: perStep(4), H2DBytes: perStep(forwardBytes), Attempted: perStep(4), Completed: perStep(4)})
	}
	exactPhase("begin_zero", phase(backend.CompactTrainPhaseBeginZero), backend.CompactTrainPhaseTelemetry{GoCalls: perStep(1), ContextSets: perStep(1), DriverCalls: perStep(1), MemsetCalls: perStep(1), Attempted: perStep(1), Completed: perStep(1)})
	exactPhase("grad_h2d", phase(backend.CompactTrainPhaseGradH2D), backend.CompactTrainPhaseTelemetry{GoCalls: perStep(1), ContextSets: perStep(1), DriverCalls: perStep(1), H2DCopies: perStep(1), H2DBytes: perStep(gradBytes), Attempted: perStep(1), Completed: perStep(1)})
	wantForwardPhase := backend.CompactTrainPhaseTelemetry{}
	if modes.Graph == 0 {
		wantForwardPhase = backend.CompactTrainPhaseTelemetry{GoCalls: perStep(forward + 3), ContextSets: perStep(forward + 3), DriverCalls: perStep(forward + 3), KernelLaunches: perStep(forward), EventRecords: perStep(2), EventQueries: perStep(1), Attempted: perStep(forward), Enqueued: perStep(forward), Completed: perStep(forward)}
	} else {
		wantForwardPhase = backend.CompactTrainPhaseTelemetry{GoCalls: perStep(4), ContextSets: perStep(4), DriverCalls: perStep(4), GraphLaunches: perStep(1), EventRecords: perStep(2), EventQueries: perStep(1), Attempted: perStep(1), Enqueued: perStep(1), Completed: perStep(1)}
	}
	forwardName := backend.CompactTrainPhaseForwardDirect
	oppositeForwardName := backend.CompactTrainPhaseForwardReplay
	if modes.Graph == 1 {
		forwardName = backend.CompactTrainPhaseForwardReplay
		oppositeForwardName = backend.CompactTrainPhaseForwardDirect
	}
	if phase(backend.CompactTrainPhaseForwardCapture) != (backend.CompactTrainPhaseTelemetry{}) ||
		phase(oppositeForwardName) != (backend.CompactTrainPhaseTelemetry{}) ||
		phase(backend.CompactTrainPhaseOuterOptimizer) != (backend.CompactTrainPhaseTelemetry{}) {
		b.Fatalf("event-on inactive telemetry phases are nonzero: capture=%+v opposite=%s:%+v outer=%+v", phase(backend.CompactTrainPhaseForwardCapture), oppositeForwardName, phase(oppositeForwardName), phase(backend.CompactTrainPhaseOuterOptimizer))
	}
	forwardPhase := phase(forwardName)
	forwardPhaseCounts := forwardPhase
	forwardPhaseCounts.HostNanos = 0
	forwardPhaseCounts.DeviceElapsedNanos = 0
	if forwardPhaseCounts != wantForwardPhase || forwardPhase.DeviceElapsedNanos <= 0 || forwardPhase.Failures != 0 || forwardPhase.EventControlFailures != 0 {
		b.Fatalf("event-on phase %s=%+v, want %+v with positive device time", forwardName, forwardPhase, wantForwardPhase)
	}
	exactPhase("forward_boundary", phase(backend.CompactTrainPhaseForwardBoundary), backend.CompactTrainPhaseTelemetry{GoCalls: perStep(1), ContextSets: perStep(1), DriverCalls: perStep(1), StreamSynchronizes: perStep(1), Attempted: perStep(1), Completed: perStep(1)})
	wantFinal := backend.CompactTrainPhaseTelemetry{GoCalls: perStep(2 + 3), ContextSets: perStep(2 + 3), DriverCalls: perStep(2 + 3), KernelLaunches: perStep(2), EventRecords: perStep(2), EventQueries: perStep(1), Attempted: perStep(2), Enqueued: perStep(2), Completed: perStep(2)}
	final := phase(backend.CompactTrainPhaseBackwardFinal)
	finalCounts := final
	finalCounts.HostNanos = 0
	finalCounts.DeviceElapsedNanos = 0
	if finalCounts != wantFinal || final.DeviceElapsedNanos <= 0 || final.Failures != 0 || final.EventControlFailures != 0 {
		b.Fatalf("event-on phase backward_final=%+v, want %+v with positive device time", final, wantFinal)
	}
	wantFFN := backend.CompactTrainPhaseTelemetry{GoCalls: perStep(22), ContextSets: perStep(22), DriverCalls: perStep(22), KernelLaunches: perStep(16), EventRecords: perStep(4), EventQueries: perStep(2), Attempted: perStep(16), Enqueued: perStep(16), Completed: perStep(16)}
	ffn := phase(backend.CompactTrainPhaseBackwardFFN)
	ffnCounts := ffn
	ffnCounts.HostNanos = 0
	ffnCounts.DeviceElapsedNanos = 0
	if ffnCounts != wantFFN || ffn.DeviceElapsedNanos <= 0 || ffn.Failures != 0 || ffn.EventControlFailures != 0 {
		b.Fatalf("event-on phase backward_ffn=%+v, want %+v with positive device time", ffn, wantFFN)
	}
	wantAttention := backend.CompactTrainPhaseTelemetry{GoCalls: perStep(38), ContextSets: perStep(38), DriverCalls: perStep(38), KernelLaunches: perStep(26), EventRecords: perStep(8), EventQueries: perStep(4), Attempted: perStep(26), Enqueued: perStep(26), Completed: perStep(26)}
	attention := phase(backend.CompactTrainPhaseBackwardAttention)
	attentionCounts := attention
	attentionCounts.HostNanos = 0
	attentionCounts.DeviceElapsedNanos = 0
	if attentionCounts != wantAttention || attention.DeviceElapsedNanos <= 0 || attention.Failures != 0 || attention.EventControlFailures != 0 {
		b.Fatalf("event-on phase backward_attention=%+v, want %+v with positive device time", attention, wantAttention)
	}
	exactPhase("backward_boundary", phase(backend.CompactTrainPhaseBackwardBoundary), backend.CompactTrainPhaseTelemetry{GoCalls: perStep(1), ContextSets: perStep(1), DriverCalls: perStep(1), StreamSynchronizes: perStep(1), Attempted: perStep(1), Completed: perStep(1)})
	_, _, activeBytes := compactResidentCUDAWarmBenchExpectedUploadBytes(shape)
	_ = activeBytes
	wantD2H := int64(4 + shape.Batch*shape.OutputDim*4 + shape.Batch*4)
	exactPhase("k6_readback", phase(backend.CompactTrainPhaseK6Readback), backend.CompactTrainPhaseTelemetry{GoCalls: perStep(1), ContextSets: perStep(1), DriverCalls: perStep(3), D2HCopies: perStep(3), D2HBytes: perStep(wantD2H), Attempted: perStep(3), Completed: perStep(3)})
	wantK5 := compactResidentCUDAWarmBenchExpectedK5Updates(shape)
	wantK5Phase := backend.CompactTrainPhaseTelemetry{GoCalls: perStep(6), ContextSets: perStep(4), DriverCalls: perStep(wantK5 + 4), KernelLaunches: perStep(wantK5), HostDescriptorAllocs: perStep(1), StreamSynchronizes: perStep(1), EventRecords: perStep(2), EventQueries: perStep(1), Attempted: perStep(wantK5), Enqueued: perStep(wantK5), Completed: perStep(wantK5)}
	k5 := phase(backend.CompactTrainPhaseK5Batch)
	k5Counts := k5
	k5Counts.HostNanos = 0
	k5Counts.DeviceElapsedNanos = 0
	if k5Counts != wantK5Phase || k5.DeviceElapsedNanos <= 0 || k5.Failures != 0 || k5.EventControlFailures != 0 {
		b.Fatalf("event-on phase k5_batch=%+v, want %+v with positive device time", k5, wantK5Phase)
	}
	// Keep named phases authoritative; this total is only a diagnostic
	// cross-check for the corrected event-control delta.
	if modes.Graph == 0 {
		exact("event-on owner GoCalls", phase(backend.CompactTrainPhaseBeginZero).GoCalls+phase(backend.CompactTrainPhaseInputH2D).GoCalls+forwardPhase.GoCalls+phase(backend.CompactTrainPhaseForwardBoundary).GoCalls+phase(backend.CompactTrainPhaseGradH2D).GoCalls+final.GoCalls+ffn.GoCalls+attention.GoCalls+phase(backend.CompactTrainPhaseBackwardBoundary).GoCalls+phase(backend.CompactTrainPhaseK6Readback).GoCalls+k5.GoCalls, perStep(forward+83))
	}
	goCalls, contextSets, driverCalls := compactResidentCUDAK9TelemetryCallTotals(telemetry)
	wantGo, wantContext, wantDriver := compactResidentCUDAK9ExpectedEventOnCallTotals(shape, modes)
	wantGo *= int64(steps)
	wantContext *= int64(steps)
	wantDriver *= int64(steps)
	if goCalls != wantGo || contextSets != wantContext || driverCalls != wantDriver {
		b.Fatalf("event-on aggregate Go/Context/Driver=%d/%d/%d, want %d/%d/%d (event-off baseline +29/+27/+27)", goCalls, contextSets, driverCalls, wantGo, wantContext, wantDriver)
	}
}

func compactResidentCUDAK9TelemetryCallTotals(telemetry *backend.CompactTrainTelemetry) (goCalls, contextSets, driverCalls int64) {
	if telemetry == nil {
		return 0, 0, 0
	}
	for _, name := range []backend.CompactTrainTelemetryPhase{
		backend.CompactTrainPhaseBeginZero, backend.CompactTrainPhaseInputH2D,
		backend.CompactTrainPhaseForwardDirect, backend.CompactTrainPhaseForwardCapture,
		backend.CompactTrainPhaseForwardReplay, backend.CompactTrainPhaseForwardBoundary,
		backend.CompactTrainPhaseGradH2D, backend.CompactTrainPhaseBackwardFinal,
		backend.CompactTrainPhaseBackwardFFN, backend.CompactTrainPhaseBackwardAttention,
		backend.CompactTrainPhaseBackwardBoundary, backend.CompactTrainPhaseK6Readback,
		backend.CompactTrainPhaseK5Batch, backend.CompactTrainPhaseOuterOptimizer,
	} {
		value := telemetry.Phase(name)
		goCalls += value.GoCalls
		contextSets += value.ContextSets
		driverCalls += value.DriverCalls
	}
	return goCalls, contextSets, driverCalls
}

func compactResidentCUDAK9ExpectedEventOffCallTotals(shape compactResidentCUDAWarmBenchShape, modes compactResidentCUDAK9Modes) (goCalls, contextSets, driverCalls int64) {
	if modes.Graph == 1 {
		if shape.OutputDim == shape.ModelDim {
			goCalls, driverCalls = 54, 70
		} else {
			goCalls, driverCalls = 55, 72
		}
	} else if shape.OutputDim == shape.ModelDim {
		goCalls, driverCalls = 75, 91
	} else {
		goCalls, driverCalls = 78, 95
	}
	if modes.Upload == 1 {
		goCalls -= 3
	}
	contextSets = goCalls
	return goCalls, contextSets, driverCalls
}

func compactResidentCUDAK9ExpectedEventOnCallTotals(shape compactResidentCUDAWarmBenchShape, modes compactResidentCUDAK9Modes) (goCalls, contextSets, driverCalls int64) {
	goCalls, contextSets, driverCalls = compactResidentCUDAK9ExpectedEventOffCallTotals(shape, modes)
	return goCalls + 29, contextSets + 27, driverCalls + 27
}

func TestCompactResidentCUDAK9EventControlAggregateIdentities(t *testing.T) {
	for _, shape := range []compactResidentCUDAWarmBenchShape{compactResidentCUDAWarmBenchCanonicalShape(), compactResidentCUDAWarmBenchNextDescriptorShape()} {
		for graph := 0; graph <= 1; graph++ {
			for upload := 0; upload <= 1; upload++ {
				modes := compactResidentCUDAK9Modes{Graph: graph, Upload: upload, Events: 1}
				offGo, offContext, offDriver := compactResidentCUDAK9ExpectedEventOffCallTotals(shape, modes)
				onGo, onContext, onDriver := compactResidentCUDAK9ExpectedEventOnCallTotals(shape, modes)
				if onGo-offGo != 29 || onContext-offContext != 27 || onDriver-offDriver != 27 {
					t.Fatalf("shape=%s graph/upload=%d/%d event-control delta=%d/%d/%d, want 29/27/27", shape.Profile, graph, upload, onGo-offGo, onContext-offContext, onDriver-offDriver)
				}
			}
		}
	}
}

func compactResidentCUDAK9PhaseMap(telemetry *backend.CompactTrainTelemetry) map[string]backend.CompactTrainPhaseTelemetry {
	if telemetry == nil {
		return nil
	}
	phases := map[string]backend.CompactTrainPhaseTelemetry{}
	for _, name := range []backend.CompactTrainTelemetryPhase{
		backend.CompactTrainPhaseBeginZero, backend.CompactTrainPhaseInputH2D,
		backend.CompactTrainPhaseForwardDirect, backend.CompactTrainPhaseForwardCapture,
		backend.CompactTrainPhaseForwardReplay, backend.CompactTrainPhaseForwardBoundary,
		backend.CompactTrainPhaseGradH2D, backend.CompactTrainPhaseBackwardFinal,
		backend.CompactTrainPhaseBackwardFFN, backend.CompactTrainPhaseBackwardAttention,
		backend.CompactTrainPhaseBackwardBoundary, backend.CompactTrainPhaseK6Readback,
		backend.CompactTrainPhaseK5Batch, backend.CompactTrainPhaseOuterOptimizer,
	} {
		if value := telemetry.Phase(name); value != (backend.CompactTrainPhaseTelemetry{}) {
			phases[string(name)] = value
		}
	}
	return phases
}

func compactResidentCUDAK9TelemetryOnlyK5(owner, mirror *backend.CompactTrainTelemetry) bool {
	if owner == nil || mirror == nil {
		return false
	}
	for _, name := range []backend.CompactTrainTelemetryPhase{
		backend.CompactTrainPhaseBeginZero, backend.CompactTrainPhaseInputH2D,
		backend.CompactTrainPhaseForwardDirect, backend.CompactTrainPhaseForwardCapture,
		backend.CompactTrainPhaseForwardReplay, backend.CompactTrainPhaseForwardBoundary,
		backend.CompactTrainPhaseGradH2D, backend.CompactTrainPhaseBackwardFinal,
		backend.CompactTrainPhaseBackwardFFN, backend.CompactTrainPhaseBackwardAttention,
		backend.CompactTrainPhaseBackwardBoundary, backend.CompactTrainPhaseK6Readback,
		backend.CompactTrainPhaseOuterOptimizer,
	} {
		if owner.Phase(name) != (backend.CompactTrainPhaseTelemetry{}) || mirror.Phase(name) != (backend.CompactTrainPhaseTelemetry{}) {
			return false
		}
	}
	return owner.Phase(backend.CompactTrainPhaseK5Batch) == mirror.Phase(backend.CompactTrainPhaseK5Batch)
}

func absInt64(value int64) int64 {
	if value < 0 {
		return -value
	}
	return value
}

func compactResidentCUDAK9BoolInt64(value bool) int64 {
	if value {
		return 1
	}
	return 0
}
