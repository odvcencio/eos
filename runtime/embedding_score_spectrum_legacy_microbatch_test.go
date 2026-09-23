package eosruntime

import (
	"math"
	"strconv"
	"strings"
	"testing"

	"m31labs.dev/eos/runtime/backend"
)

// TestLegacyScoreSpectrumActivationMicrobatchPreservesOneStepParity exercises
// the role-conditioned legacy path, which deliberately uses scalar backward
// accumulation. Keeping batched forward/backward disabled makes this a tight
// reference test for row/candidate ordering and Adam state, including the
// hard+soft+recovery objective and repeated candidates.
func TestLegacyScoreSpectrumActivationMicrobatchPreservesOneStepParity(t *testing.T) {
	t.Setenv("EOS_TRAIN_DISABLE_BATCHED_FORWARD", "1")
	t.Setenv("EOS_TRAIN_DISABLE_BATCHED_BACKWARD", "1")

	for _, microbatchSize := range []int{1, 2, 8} {
		t.Run("size_"+strconv.Itoa(microbatchSize), func(t *testing.T) {
			baseline := newLegacyScoreSpectrumParityTrainer(t, 0)
			microbatched := newLegacyScoreSpectrumParityTrainer(t, microbatchSize)
			batch := legacyScoreSpectrumParityBatch()

			want, err := baseline.TrainScoreSpectrumStep(batch)
			if err != nil {
				t.Fatalf("baseline score-spectrum step: %v", err)
			}
			got, err := microbatched.TrainScoreSpectrumStep(legacyScoreSpectrumParityBatch())
			if err != nil {
				t.Fatalf("microbatched score-spectrum step (size %d): %v", microbatchSize, err)
			}
			assertLegacyScoreSpectrumTrainMetricsClose(t, got, want)
			if baseline.step != 1 || microbatched.step != 1 {
				t.Fatalf("steps baseline/microbatched = %d/%d, want one optimizer step each", baseline.step, microbatched.step)
			}
			assertLegacyScoreSpectrumTensorStateClose(t, microbatched, baseline, 1e-6)
		})
	}
}

func TestLegacyScoreSpectrumActivationMicrobatchBatchedReferenceParity(t *testing.T) {
	// The unconditioned attention trainer exercises the existing batched
	// backward helper. The microbatch route may group each replay chunk
	// independently, but it must still produce the same single-step result.
	t.Setenv("EOS_TRAIN_DISABLE_BATCHED_FORWARD", "")
	t.Setenv("EOS_TRAIN_DISABLE_BATCHED_BACKWARD", "")
	for _, microbatchSize := range []int{1, 2, 8} {
		t.Run("size_"+strconv.Itoa(microbatchSize), func(t *testing.T) {
			baseline := newTinyTrainableAttentionEmbeddingTrainer(t, 0.01)
			microbatched := newTinyTrainableAttentionEmbeddingTrainer(t, 0.01)
			configureLegacyScoreSpectrumParityTrainer(baseline, 0)
			configureLegacyScoreSpectrumParityTrainer(microbatched, microbatchSize)

			want, err := baseline.TrainScoreSpectrumStep(legacyScoreSpectrumAttentionParityBatch())
			if err != nil {
				t.Fatalf("batched baseline score-spectrum step: %v", err)
			}
			got, err := microbatched.TrainScoreSpectrumStep(legacyScoreSpectrumAttentionParityBatch())
			if err != nil {
				t.Fatalf("batched microbatched score-spectrum step (size %d): %v", microbatchSize, err)
			}
			assertLegacyScoreSpectrumTrainMetricsClose(t, got, want)
			if baseline.step != 1 || microbatched.step != 1 {
				t.Fatalf("steps baseline/microbatched = %d/%d, want one optimizer step each", baseline.step, microbatched.step)
			}
			assertLegacyScoreSpectrumTensorStateClose(t, microbatched, baseline, 1e-5)
		})
	}
}

func TestLegacyScoreSpectrumActivationMicrobatchKeepsOversizedRowIntact(t *testing.T) {
	t.Setenv("EOS_TRAIN_DISABLE_BATCHED_FORWARD", "1")
	t.Setenv("EOS_TRAIN_DISABLE_BATCHED_BACKWARD", "1")
	trainer := newLegacyScoreSpectrumParityTrainer(t, 2)
	batch := []EmbeddingScoreSpectrumExample{legacyScoreSpectrumParityBatch()[0]}
	if len(batch[0].CandidateTokens) <= trainer.config.ScoreSpectrumActivationMicrobatchSize {
		t.Fatalf("test row candidate count = %d, want greater than microbatch size %d", len(batch[0].CandidateTokens), trainer.config.ScoreSpectrumActivationMicrobatchSize)
	}

	metrics, err := trainer.TrainScoreSpectrumStep(batch)
	if err != nil {
		t.Fatalf("oversized-row score-spectrum step: %v", err)
	}
	if metrics.BatchSize != len(batch[0].CandidateTokens) {
		t.Fatalf("oversized-row pair count = %d, want %d candidates", metrics.BatchSize, len(batch[0].CandidateTokens))
	}
	if trainer.step != 1 {
		t.Fatalf("oversized-row step = %d, want one optimizer update", trainer.step)
	}
}

func TestLegacyScoreSpectrumActivationMicrobatchEvalParity(t *testing.T) {
	baseline := newLegacyScoreSpectrumParityTrainer(t, 0)
	microbatched := newLegacyScoreSpectrumParityTrainer(t, 2)
	examples := legacyScoreSpectrumParityBatch()

	want, err := baseline.EvaluateScoreSpectrumBatched(examples, 2)
	if err != nil {
		t.Fatalf("baseline score-spectrum eval: %v", err)
	}
	got, err := microbatched.EvaluateScoreSpectrumBatched(legacyScoreSpectrumParityBatch(), 2)
	if err != nil {
		t.Fatalf("microbatched score-spectrum eval: %v", err)
	}
	assertScoreSpectrumEvalMetricsClose(t, got, want)
	if baseline.step != 0 || microbatched.step != 0 {
		t.Fatalf("eval changed steps baseline/microbatched = %d/%d", baseline.step, microbatched.step)
	}
}

func TestLegacyScoreSpectrumActivationMicrobatchReleasesSequenceBindings(t *testing.T) {
	t.Setenv("EOS_TRAIN_ENABLE_SEQUENCE_MATMUL_BINDINGS", "1")
	t.Setenv("EOS_TRAIN_DISABLE_BATCHED_FORWARD", "1")
	t.Setenv("EOS_TRAIN_DISABLE_BATCHED_BACKWARD", "1")
	trainer := newLegacyScoreSpectrumParityTrainer(t, 2)
	if trainer.forwardMatMul != nil {
		trainer.forwardMatMul.Close()
	}
	fake := &countingMatMulAccelerator{}
	trainer.forwardMatMul = fake

	if _, err := trainer.TrainScoreSpectrumStep(legacyScoreSpectrumParityBatch()); err != nil {
		t.Fatalf("microbatched score-spectrum step with bindings: %v", err)
	}
	for name := range fake.bound {
		if strings.HasPrefix(name, "seq_") {
			t.Fatalf("sequence binding %q retained after microbatched step; bound=%v", name, fake.bound)
		}
	}
	if len(fake.bound) != 1 || fake.bound["projection"] == nil {
		t.Fatalf("forward bindings after microbatched step = %v, want only persistent projection binding", fake.bound)
	}
}

func TestLegacyScoreSpectrumActivationMicrobatchResidentRequestFailsClosed(t *testing.T) {
	t.Setenv(compactResidentTrainEnv, "1")
	trainer := newLegacyScoreSpectrumParityTrainer(t, 2)
	_, err := trainer.TrainScoreSpectrumStep(legacyScoreSpectrumParityBatch())
	if err == nil || !strings.Contains(err.Error(), "unsupported architecture") ||
		!strings.Contains(err.Error(), EmbeddingArchitectureLegacyV1) ||
		!strings.Contains(err.Error(), EmbeddingArchitectureCompactTransformerV1) {
		t.Fatalf("legacy resident error = %v, want fail-closed unsupported-architecture error", err)
	}
	if trainer.step != 0 {
		t.Fatalf("resident rejection step = %d, want unchanged", trainer.step)
	}
}

func TestLegacyScoreSpectrumActivationMicrobatchRejectsNegativeTrainerConfig(t *testing.T) {
	trainer := newLegacyScoreSpectrumParityTrainer(t, 0)
	trainer.config.ScoreSpectrumActivationMicrobatchSize = -1
	_, err := trainer.TrainScoreSpectrumStep(legacyScoreSpectrumParityBatch())
	if err == nil || !strings.Contains(err.Error(), "score_spectrum_activation_microbatch_size") {
		t.Fatalf("negative trainer activation microbatch error = %v, want non-negative validation", err)
	}
}

func newLegacyScoreSpectrumParityTrainer(t *testing.T, microbatchSize int) *EmbeddingTrainer {
	t.Helper()
	trainer := newTinyRoleEmbeddingTrainer(t, 0.01)
	t.Cleanup(trainer.Close)
	configureLegacyScoreSpectrumParityTrainer(trainer, microbatchSize)
	return trainer
}

func configureLegacyScoreSpectrumParityTrainer(trainer *EmbeddingTrainer, microbatchSize int) {
	trainer.config.Temperature = 0.2
	trainer.config.ScoreSpectrumLossMode = ScoreSpectrumLossModeHardSoftRecovery
	trainer.config.ScoreSpectrumRecoveryWeight = 1
	trainer.config.ScoreSpectrumRecoveryMargin = 0.07
	trainer.config.ScoreSpectrumRecoveryTopK = 2
	trainer.config.ScoreSpectrumRecoveryTau = 0.3
	trainer.config.ScoreSpectrumActivationMicrobatchSize = microbatchSize
}

func legacyScoreSpectrumParityBatch() []EmbeddingScoreSpectrumExample {
	selected0 := 2
	selected1 := 2
	return []EmbeddingScoreSpectrumExample{
		{
			QueryTokens:           []int32{0, 1},
			QueryMask:             []int32{1, 1},
			CandidateTokens:       [][]int32{{0}, {1, 2}, {0}, {3, 1}},
			CandidateMasks:        [][]int32{{1}, {1, 1}, {1}, {1, 1}},
			PositiveIndexes:       []int{0, 2},
			SelectedPositiveIndex: &selected0,
			HardNegativeEligible:  []bool{false, true, false, true},
			TargetProbabilities:   []float32{0.3, 0.2, 0.4, 0.1},
			RecoveryLossWeight:    1,
			CommercialUseAllowed:  true,
		},
		{
			QueryTokens:           []int32{2, 3},
			QueryMask:             []int32{1, 1},
			CandidateTokens:       [][]int32{{3}, {2, 3}, {3}, {0, 1}},
			CandidateMasks:        [][]int32{{1}, {1, 1}, {1}, {1, 1}},
			PositiveIndexes:       []int{1, 2},
			SelectedPositiveIndex: &selected1,
			HardNegativeEligible:  []bool{true, false, false, true},
			TargetProbabilities:   []float32{0.15, 0.4, 0.35, 0.1},
			RecoveryLossWeight:    1,
			CommercialUseAllowed:  true,
		},
	}
}

func legacyScoreSpectrumAttentionParityBatch() []EmbeddingScoreSpectrumExample {
	selected0 := 2
	selected1 := 2
	return []EmbeddingScoreSpectrumExample{
		{
			QueryTokens:           []int32{0, 1},
			QueryMask:             []int32{1, 1},
			CandidateTokens:       [][]int32{{0}, {1, 2}, {0}, {2, 1}},
			CandidateMasks:        [][]int32{{1}, {1, 1}, {1}, {1, 1}},
			PositiveIndexes:       []int{0, 2},
			SelectedPositiveIndex: &selected0,
			HardNegativeEligible:  []bool{false, true, false, true},
			TargetProbabilities:   []float32{0.3, 0.2, 0.4, 0.1},
			RecoveryLossWeight:    1,
			CommercialUseAllowed:  true,
		},
		{
			QueryTokens:           []int32{2, 0},
			QueryMask:             []int32{1, 1},
			CandidateTokens:       [][]int32{{2}, {0, 1}, {2}, {1, 0}},
			CandidateMasks:        [][]int32{{1}, {1, 1}, {1}, {1, 1}},
			PositiveIndexes:       []int{1, 2},
			SelectedPositiveIndex: &selected1,
			HardNegativeEligible:  []bool{true, false, false, true},
			TargetProbabilities:   []float32{0.15, 0.4, 0.35, 0.1},
			RecoveryLossWeight:    1,
			CommercialUseAllowed:  true,
		},
	}
}

func assertLegacyScoreSpectrumTrainMetricsClose(t *testing.T, got, want EmbeddingTrainMetrics) {
	t.Helper()
	const tolerance = float32(1e-6)
	if math.Abs(float64(got.Loss-want.Loss)) > float64(tolerance) {
		t.Fatalf("loss = %.9g, want %.9g", got.Loss, want.Loss)
	}
	if math.Abs(float64(got.AverageScore-want.AverageScore)) > float64(tolerance) {
		t.Fatalf("average score = %.9g, want %.9g", got.AverageScore, want.AverageScore)
	}
	if got.BatchSize != want.BatchSize {
		t.Fatalf("batch size = %d, want %d", got.BatchSize, want.BatchSize)
	}
}

func legacyScoreSpectrumTensorState(trainer *EmbeddingTrainer) map[string][]float32 {
	state := map[string][]float32{}
	add := func(name string, tensor *backend.Tensor) {
		if tensor != nil {
			state[name] = append([]float32(nil), tensor.F32...)
		}
	}
	add("token", trainer.tokenEmbed)
	add("role", trainer.roleEmbed)
	add("attn_q", trainer.attentionQuery)
	add("attn_k", trainer.attentionKey)
	add("attn_v", trainer.attentionValue)
	add("attn_o", trainer.attentionOutput)
	add("hidden", trainer.hiddenProjection)
	add("projection", trainer.projection)
	add("token_mom1", trainer.tokenMom1)
	add("token_mom2", trainer.tokenMom2)
	add("role_mom1", trainer.roleMom1)
	add("role_mom2", trainer.roleMom2)
	add("attn_q_mom1", trainer.attnQMom1)
	add("attn_q_mom2", trainer.attnQMom2)
	add("attn_k_mom1", trainer.attnKMom1)
	add("attn_k_mom2", trainer.attnKMom2)
	add("attn_v_mom1", trainer.attnVMom1)
	add("attn_v_mom2", trainer.attnVMom2)
	add("attn_o_mom1", trainer.attnOMom1)
	add("attn_o_mom2", trainer.attnOMom2)
	add("hidden_mom1", trainer.hiddenMom1)
	add("hidden_mom2", trainer.hiddenMom2)
	add("projection_mom1", trainer.projMom1)
	add("projection_mom2", trainer.projMom2)
	return state
}

func assertLegacyScoreSpectrumTensorStateClose(t *testing.T, got, want *EmbeddingTrainer, tolerance float32) {
	t.Helper()
	gotState := legacyScoreSpectrumTensorState(got)
	wantState := legacyScoreSpectrumTensorState(want)
	if len(gotState) != len(wantState) {
		t.Fatalf("tensor state key count = %d, want %d (got=%v want=%v)", len(gotState), len(wantState), mapKeysForTest(gotState), mapKeysForTest(wantState))
	}
	for name, wantValues := range wantState {
		gotValues, ok := gotState[name]
		if !ok {
			t.Fatalf("tensor state missing %q", name)
		}
		assertCloseF32Slice(t, name, gotValues, wantValues, tolerance)
	}
}

func mapKeysForTest(values map[string][]float32) []string {
	keys := make([]string, 0, len(values))
	for key := range values {
		keys = append(keys, key)
	}
	return keys
}
