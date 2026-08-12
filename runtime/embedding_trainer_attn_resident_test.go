package eosruntime

import (
	"fmt"
	"math"
	"testing"

	eosartifact "m31labs.dev/eos/artifact/eos"
	"m31labs.dev/eos/runtime/backend"
)

// RunAttentionBlockResident implements backend.AttentionResidentAccelerator
// on the shared countingMatMulAccelerator test fake with a correct,
// hardware-independent reference computation: it reads whatever tensors are
// currently bound under req.QueryName/KeyName/ValueName (exactly like the
// real CUDA implementation reads its residentMatrices map), so a trainer-side
// staleness bug -- failing to refresh a binding after an optimizer step --
// reproduces here exactly as it would against real CUDA.
func (a *countingMatMulAccelerator) RunAttentionBlockResident(req backend.AttentionResidentRequest) (backend.AttentionResidentResult, error) {
	a.runCalls++
	a.attnResidentRuns++
	if a.attnResidentErr != nil {
		return backend.AttentionResidentResult{}, a.attnResidentErr
	}
	if req.Input == nil || req.Batch <= 0 || req.SeqLen <= 0 || req.ModelDim <= 0 {
		return backend.AttentionResidentResult{}, fmt.Errorf("fake attention resident accelerator: invalid request")
	}
	rows := req.Batch * req.SeqLen
	d := req.ModelDim
	seq := req.SeqLen
	if len(req.Input.F32) != rows*d {
		return backend.AttentionResidentResult{}, fmt.Errorf("fake attention resident accelerator: input shape mismatch")
	}
	wq := a.bound[req.QueryName]
	wk := a.bound[req.KeyName]
	wv := a.bound[req.ValueName]
	if wq == nil || wk == nil || wv == nil || len(wq.F32) != d*d || len(wk.F32) != d*d || len(wv.F32) != d*d {
		return backend.AttentionResidentResult{}, fmt.Errorf("fake attention resident accelerator: missing or mismatched resident binding")
	}

	query := make([]float32, rows*d)
	key := make([]float32, rows*d)
	value := make([]float32, rows*d)
	fillHostMatMul(req.Input.F32, rows, d, wq.F32, d, query)
	fillHostMatMul(req.Input.F32, rows, d, wk.F32, d, key)
	fillHostMatMul(req.Input.F32, rows, d, wv.F32, d, value)

	scores := make([]float32, req.Batch*seq*seq)
	mixed := make([]float32, rows*d)
	for b := 0; b < req.Batch; b++ {
		base := b * seq * d
		scoreBase := b * seq * seq
		qb := query[base : base+seq*d]
		kb := key[base : base+seq*d]
		vb := value[base : base+seq*d]
		sb := scores[scoreBase : scoreBase+seq*seq]
		fillHostMatMulTranspose(qb, seq, d, kb, seq, d, false, true, sb)
		softmaxRowsInPlace(sb, seq, seq)
		mb := mixed[base : base+seq*d]
		fillHostMatMul(sb, seq, seq, vb, d, mb)
	}

	a.uploadedBytes += int64(len(req.Input.F32) * 4)
	a.downloadedBytes += int64((len(query) + len(key) + len(value) + len(scores) + len(mixed)) * 4)

	return backend.AttentionResidentResult{
		Query:  backend.NewTensorF32([]int{rows, d}, query),
		Key:    backend.NewTensorF32([]int{rows, d}, key),
		Value:  backend.NewTensorF32([]int{rows, d}, value),
		Scores: backend.NewTensorF32([]int{rows, seq}, scores),
		Mixed:  backend.NewTensorF32([]int{rows, d}, mixed),
	}, nil
}

func tinyAttnResidentContrastiveBatch(n int) []EmbeddingContrastiveExample {
	all := []EmbeddingContrastiveExample{
		{QueryTokens: []int32{0, 2}, PositiveTokens: []int32{0, 0}, QueryMask: []int32{1, 1}, PositiveMask: []int32{1, 1}},
		{QueryTokens: []int32{1, 2}, PositiveTokens: []int32{1, 1}, QueryMask: []int32{1, 1}, PositiveMask: []int32{1, 1}},
		{QueryTokens: []int32{2, 0}, PositiveTokens: []int32{0, 2}, QueryMask: []int32{1, 1}, PositiveMask: []int32{1, 1}},
	}
	if n > len(all) {
		n = len(all)
	}
	return append([]EmbeddingContrastiveExample(nil), all[:n]...)
}

// TestEmbeddingTrainerAttnResidentDisabledByDefault confirms EOS_EMBED_ATTN_RESIDENT
// defaults OFF: an accelerator that implements the resident fast path must
// not be used unless the flag is explicitly set.
func TestEmbeddingTrainerAttnResidentDisabledByDefault(t *testing.T) {
	trainer := newTinyTrainableAttentionEmbeddingTrainer(t, 0.05)
	if trainer.forwardMatMul != nil {
		trainer.forwardMatMul.Close()
	}
	fake := &countingMatMulAccelerator{}
	trainer.forwardMatMul = fake
	trainer.config.ContrastiveLoss = "infonce"
	trainer.config.Temperature = 0.05

	if _, err := trainer.TrainContrastiveStep(tinyAttnResidentContrastiveBatch(2)); err != nil {
		t.Fatalf("train step: %v", err)
	}
	if fake.attnResidentRuns != 0 {
		t.Fatalf("attention resident runs = %d, want 0 when EOS_EMBED_ATTN_RESIDENT is unset", fake.attnResidentRuns)
	}
	if trainer.attnResidentQueryBinding != "" {
		t.Fatalf("attnResidentQueryBinding = %q, want empty when the flag is off", trainer.attnResidentQueryBinding)
	}
}

// TestEmbeddingTrainerAttnResidentQueryBindingBakesScoreScale verifies
// primeAttentionResidentQueryBinding bakes the model's attention score scale
// into the resident query copy so the device chain never needs a separate
// scale kernel: scale*(Q@K^T) == (scale*Q)@K^T.
func TestEmbeddingTrainerAttnResidentQueryBindingBakesScoreScale(t *testing.T) {
	t.Setenv("EOS_EMBED_ATTN_RESIDENT", "1")
	trainer := newTinyTrainableAttentionEmbeddingTrainer(t, 0.05)
	if trainer.forwardMatMul != nil {
		trainer.forwardMatMul.Close()
	}
	fake := &countingMatMulAccelerator{}
	trainer.forwardMatMul = fake
	trainer.manifest.AttentionScoreScale = EmbeddingAttentionScoreScaleKeyDimRSQ

	attnQ := backend.NewTensorF32([]int{2, 2}, []float32{2, 4, 6, 8})
	trainer.primeAttentionResidentQueryBinding(attnQ)

	if trainer.attnResidentQueryBinding == "" {
		t.Fatal("expected a resident query binding name")
	}
	bound := fake.bound[trainer.attnResidentQueryBinding]
	if bound == nil {
		t.Fatal("expected the scaled query weight to be bound on the accelerator")
	}
	scale := float32(1 / math.Sqrt(2))
	want := []float32{2 * scale, 4 * scale, 6 * scale, 8 * scale}
	assertCloseF32Slice(t, "scaled attn_q", bound.F32, want, 1e-6)

	// A backend without the resident interface must clear the binding name
	// rather than leave a stale one pointing at a previous weight generation
	// (fail closed): swap in an accelerator that implements
	// backend.MatMulAccelerator only, deliberately without
	// RunAttentionBlockResident.
	trainer.forwardMatMul = &minimalMatMulAccelerator{}
	trainer.attnResidentQueryBinding = "stale_from_before"
	trainer.primeAttentionResidentQueryBinding(attnQ)
	if trainer.attnResidentQueryBinding != "" {
		t.Fatalf("attnResidentQueryBinding = %q, want empty when the accelerator lacks the resident interface", trainer.attnResidentQueryBinding)
	}

	// Flag off must also clear it, even with a capable accelerator bound.
	trainer.forwardMatMul = fake
	trainer.attnResidentQueryBinding = "stale_from_before"
	t.Setenv("EOS_EMBED_ATTN_RESIDENT", "0")
	trainer.primeAttentionResidentQueryBinding(attnQ)
	if trainer.attnResidentQueryBinding != "" {
		t.Fatalf("attnResidentQueryBinding = %q, want empty when the flag is off", trainer.attnResidentQueryBinding)
	}
}

// minimalMatMulAccelerator implements backend.MatMulAccelerator only,
// deliberately without RunAttentionBlockResident, matching how a backend
// without the resident fast path looks to primeAttentionResidentQueryBinding
// and tryAttentionBlockResident's type assertions.
type minimalMatMulAccelerator struct {
	bound map[string]*backend.Tensor
}

func (a *minimalMatMulAccelerator) Backend() eosartifact.BackendKind {
	return eosartifact.BackendCUDA
}

func (a *minimalMatMulAccelerator) RunMatMul([]*backend.Tensor, eosartifact.ValueType) (backend.StepDispatchResult, error) {
	return backend.StepDispatchResult{}, nil
}

func (a *minimalMatMulAccelerator) RunMatMulWithTranspose([]*backend.Tensor, eosartifact.ValueType, bool, bool) (backend.StepDispatchResult, error) {
	return backend.StepDispatchResult{}, nil
}

func (a *minimalMatMulAccelerator) BindMatrix(name string, tensor *backend.Tensor) error {
	if a.bound == nil {
		a.bound = map[string]*backend.Tensor{}
	}
	a.bound[name] = tensor
	return nil
}

func (a *minimalMatMulAccelerator) UnbindMatrix(name string) error {
	delete(a.bound, name)
	return nil
}

func (a *minimalMatMulAccelerator) RunMatMulWithBoundLeft(string, *backend.Tensor, eosartifact.ValueType, bool, bool) (backend.StepDispatchResult, error) {
	return backend.StepDispatchResult{}, nil
}

func (a *minimalMatMulAccelerator) RunMatMulWithBoundRight(*backend.Tensor, string, eosartifact.ValueType, bool, bool) (backend.StepDispatchResult, error) {
	return backend.StepDispatchResult{}, nil
}

func (a *minimalMatMulAccelerator) Stats() backend.MatMulAcceleratorStats {
	return backend.MatMulAcceleratorStats{}
}

func (a *minimalMatMulAccelerator) Close() {}

// TestEmbeddingTrainerAttnResidentSkipsMaskedMode confirms the stage (a)
// fast path stays off for masked attention: the device forward-softmax
// kernel it uses has no mask input, so masked models must keep using the
// host path.
func TestEmbeddingTrainerAttnResidentSkipsMaskedMode(t *testing.T) {
	t.Setenv("EOS_EMBED_ATTN_RESIDENT", "1")
	trainer := newTinyTrainableAttentionEmbeddingTrainer(t, 0.05)
	if trainer.forwardMatMul != nil {
		trainer.forwardMatMul.Close()
	}
	fake := &countingMatMulAccelerator{}
	trainer.forwardMatMul = fake
	trainer.manifest.AttentionMaskMode = EmbeddingAttentionMaskModeKey
	trainer.config.ContrastiveLoss = "infonce"
	trainer.config.Temperature = 0.05

	if _, err := trainer.TrainContrastiveStep(tinyAttnResidentContrastiveBatch(2)); err != nil {
		t.Fatalf("train step: %v", err)
	}
	if fake.attnResidentRuns != 0 {
		t.Fatalf("attention resident runs = %d, want 0 for attention_mask_mode=key", fake.attnResidentRuns)
	}
}

// TestEmbeddingTrainerAttnResidentParityAcrossShapes exercises
// tryAttentionBlockResident's batching/unflattening against a
// hardware-independent reference accelerator across a few batch shapes,
// comparing flag-on (resident) and flag-off (host) forward output.
func TestEmbeddingTrainerAttnResidentParityAcrossShapes(t *testing.T) {
	for _, tc := range []struct {
		name string
		n    int
	}{
		{name: "batch_1", n: 1},
		{name: "batch_2", n: 2},
		{name: "batch_3", n: 3},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Setenv("EOS_EMBED_ATTN_RESIDENT", "1")
			trainer := newTinyTrainableAttentionEmbeddingTrainer(t, 0.05)
			if trainer.forwardMatMul != nil {
				trainer.forwardMatMul.Close()
			}
			fake := &countingMatMulAccelerator{}
			trainer.forwardMatMul = fake
			trainer.manifest.AttentionScoreScale = EmbeddingAttentionScoreScaleKeyDimRSQ
			trainer.config.ContrastiveLoss = "infonce"
			trainer.config.Temperature = 0.05

			batch := tinyAttnResidentContrastiveBatch(tc.n)

			resident, err := trainer.EvaluateContrastive(batch)
			if err != nil {
				t.Fatalf("evaluate resident: %v", err)
			}
			if fake.attnResidentRuns == 0 {
				t.Fatal("expected the resident fast path to run")
			}

			t.Setenv("EOS_EMBED_ATTN_RESIDENT", "0")
			host, err := trainer.EvaluateContrastive(batch)
			if err != nil {
				t.Fatalf("evaluate host: %v", err)
			}

			if math.IsNaN(float64(resident.Loss)) || math.IsNaN(float64(host.Loss)) {
				t.Fatalf("nan loss: resident=%v host=%v", resident.Loss, host.Loss)
			}
			if diff := math.Abs(float64(resident.Loss - host.Loss)); diff > 1e-4 {
				t.Fatalf("resident loss %v vs host loss %v, diff %v > 1e-4", resident.Loss, host.Loss, diff)
			}
			if diff := math.Abs(float64(resident.AverageScore - host.AverageScore)); diff > 1e-4 {
				t.Fatalf("resident average score %v vs host average score %v, diff %v > 1e-4", resident.AverageScore, host.AverageScore, diff)
			}
		})
	}
}

// TestEmbeddingTrainerAttnResidentReflectsWeightsAfterOptimizerStep is the
// stale-state guard: it mutates Wq/Wk/Wv/Wo via a real optimizer step and
// asserts the next resident-fast-path forward reflects the update rather
// than a cached pre-update binding -- the class of bug CHANGELOG.md's
// pre-eval device-weight sync fix addressed for the compact lane.
func TestEmbeddingTrainerAttnResidentReflectsWeightsAfterOptimizerStep(t *testing.T) {
	t.Setenv("EOS_EMBED_ATTN_RESIDENT", "1")
	trainer := newTinyTrainableAttentionEmbeddingTrainer(t, 0.05)
	if trainer.forwardMatMul != nil {
		trainer.forwardMatMul.Close()
	}
	fake := &countingMatMulAccelerator{}
	trainer.forwardMatMul = fake
	trainer.manifest.AttentionScoreScale = EmbeddingAttentionScoreScaleKeyDimRSQ
	trainer.config.ContrastiveLoss = "infonce"
	trainer.config.Temperature = 0.05

	batch := tinyAttnResidentContrastiveBatch(2)

	// Baseline: resident-fast-path forward before any weight update.
	before, err := trainer.EvaluateContrastive(batch)
	if err != nil {
		t.Fatalf("evaluate before: %v", err)
	}
	if fake.attnResidentRuns == 0 {
		t.Fatal("expected the resident fast path to run before the optimizer step")
	}

	// Mutate Wq/Wk/Wv/Wo (and the token embedding) via a real optimizer step
	// while the resident flag is on.
	if _, err := trainer.TrainContrastiveStep(batch); err != nil {
		t.Fatalf("train step: %v", err)
	}

	// Flag ON: forward again on the SAME (post-update) trainer/weights.
	afterOn, err := trainer.EvaluateContrastive(batch)
	if err != nil {
		t.Fatalf("evaluate after (resident): %v", err)
	}

	// Flag OFF: independently computed forward on the identical post-update
	// weights -- the ground truth this test checks the resident path against.
	t.Setenv("EOS_EMBED_ATTN_RESIDENT", "0")
	afterOff, err := trainer.EvaluateContrastive(batch)
	if err != nil {
		t.Fatalf("evaluate after (host): %v", err)
	}

	if math.IsNaN(float64(afterOn.Loss)) || math.IsNaN(float64(afterOff.Loss)) {
		t.Fatalf("nan loss after optimizer step: resident=%v host=%v", afterOn.Loss, afterOff.Loss)
	}
	if diff := math.Abs(float64(afterOn.Loss - afterOff.Loss)); diff > 1e-4 {
		t.Fatalf("post-update resident loss %v vs host loss %v, diff %v > 1e-4 (stale weights?)", afterOn.Loss, afterOff.Loss, diff)
	}
	if math.Abs(float64(afterOn.Loss-before.Loss)) < 1e-6 {
		t.Fatalf("post-update loss %v did not move from the pre-update baseline %v; resident forward may be reusing stale weights", afterOn.Loss, before.Loss)
	}
}

// TestEmbeddingTrainerAttnResidentCUDAParity is the hardware parity check:
// flag-on (device-resident) forward output must match flag-off (host)
// forward output within 1e-4 across a few batch shapes on the real CUDA
// backend. It skips (rather than fails) when no accelerator implementing
// backend.AttentionResidentAccelerator is available, matching how the
// existing bound-matmul accelerator tests skip without CUDA hardware.
func TestEmbeddingTrainerAttnResidentCUDAParity(t *testing.T) {
	trainer := newTinyTrainableAttentionEmbeddingTrainer(t, 0.05)
	if _, ok := trainer.forwardMatMul.(backend.AttentionResidentAccelerator); !ok {
		t.Skip("matmul accelerator does not implement the attention-resident fast path")
	}
	trainer.manifest.AttentionScoreScale = EmbeddingAttentionScoreScaleKeyDimRSQ
	trainer.config.ContrastiveLoss = "infonce"
	trainer.config.Temperature = 0.05

	for _, tc := range []struct {
		name string
		n    int
	}{
		{name: "batch_1", n: 1},
		{name: "batch_2", n: 2},
		{name: "batch_3", n: 3},
	} {
		t.Run(tc.name, func(t *testing.T) {
			batch := tinyAttnResidentContrastiveBatch(tc.n)

			t.Setenv("EOS_EMBED_ATTN_RESIDENT", "1")
			resident, err := trainer.EvaluateContrastive(batch)
			if err != nil {
				t.Fatalf("evaluate resident: %v", err)
			}

			t.Setenv("EOS_EMBED_ATTN_RESIDENT", "0")
			host, err := trainer.EvaluateContrastive(batch)
			if err != nil {
				t.Fatalf("evaluate host: %v", err)
			}

			if math.IsNaN(float64(resident.Loss)) || math.IsNaN(float64(host.Loss)) {
				t.Fatalf("nan loss: resident=%v host=%v", resident.Loss, host.Loss)
			}
			if diff := math.Abs(float64(resident.Loss - host.Loss)); diff > 1e-4 {
				t.Fatalf("cuda resident loss %v vs host loss %v, diff %v > 1e-4", resident.Loss, host.Loss, diff)
			}
			if diff := math.Abs(float64(resident.AverageScore - host.AverageScore)); diff > 1e-4 {
				t.Fatalf("cuda resident average score %v vs host average score %v, diff %v > 1e-4", resident.AverageScore, host.AverageScore, diff)
			}
		})
	}
}
