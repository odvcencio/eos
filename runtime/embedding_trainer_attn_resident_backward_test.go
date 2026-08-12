package eosruntime

import (
	"fmt"
	"math"
	"testing"

	eosartifact "m31labs.dev/eos/artifact/eos"
	"m31labs.dev/eos/runtime/backend"
)

// fakeAttentionResidentTrainHandleToken implements
// backend.AttentionResidentTrainHandleToken on the shared
// countingMatMulAccelerator test fake.
type fakeAttentionResidentTrainHandleToken struct {
	id    int
	alive bool
}

func (tk *fakeAttentionResidentTrainHandleToken) AttentionResidentTrainHandleToken() {}
func (tk *fakeAttentionResidentTrainHandleToken) Backend() eosartifact.BackendKind {
	return eosartifact.BackendCUDA
}
func (tk *fakeAttentionResidentTrainHandleToken) Generation() uint64 { return 0 }
func (tk *fakeAttentionResidentTrainHandleToken) StepID() uint64     { return 0 }
func (tk *fakeAttentionResidentTrainHandleToken) Alive() bool        { return tk != nil && tk.alive }

// fakeAttentionResidentTrainHandleState is the device-resident data one
// RunAttentionBlockResidentTrainForward call would keep on the device; the
// fake keeps it as plain host slices instead.
type fakeAttentionResidentTrainHandleState struct {
	token          *fakeAttentionResidentTrainHandleToken
	seqLen, d      int
	scale          float32
	input          []float32
	q, k, v, probs []float32
	wq, wk, wv     []float32
}

// RunAttentionBlockResidentTrainForward implements
// backend.AttentionResidentTrainAccelerator on countingMatMulAccelerator: a
// correct, hardware-independent reference computation using unscaled Q/K/V
// (req.Scale applied explicitly, mirroring the CUDA train accelerator, NOT
// stage (a)'s pre-scaled-Q convention) and the production masked-softmax
// helper (softmaxRowsMaskedColumnsInPlace) so a trainer-side masking bug
// reproduces here exactly as it would against real CUDA.
func (a *countingMatMulAccelerator) RunAttentionBlockResidentTrainForward(req backend.AttentionResidentTrainForwardRequest) (backend.AttentionResidentTrainForwardResult, error) {
	a.attnResidentTrainForwardRuns++
	if !a.attnResidentTrainStepActive || a.attnResidentTrainStepID != req.StepID {
		return backend.AttentionResidentTrainForwardResult{}, fmt.Errorf("fake attention resident train forward: step %d is not active", req.StepID)
	}
	if req.Input == nil || req.Batch <= 0 || req.SeqLen <= 0 || req.ModelDim <= 0 {
		return backend.AttentionResidentTrainForwardResult{}, fmt.Errorf("fake attention resident train forward: invalid request")
	}
	batch, seq, d := req.Batch, req.SeqLen, req.ModelDim
	rows := batch * seq
	if len(req.Input.F32) != rows*d {
		return backend.AttentionResidentTrainForwardResult{}, fmt.Errorf("fake attention resident train forward: input shape mismatch")
	}
	if req.Mask != nil && len(req.Mask) != batch {
		return backend.AttentionResidentTrainForwardResult{}, fmt.Errorf("fake attention resident train forward: mask has %d sequences, want %d", len(req.Mask), batch)
	}
	wq := a.bound[req.QueryName]
	wk := a.bound[req.KeyName]
	wv := a.bound[req.ValueName]
	if wq == nil || wk == nil || wv == nil || len(wq.F32) != d*d || len(wk.F32) != d*d || len(wv.F32) != d*d {
		return backend.AttentionResidentTrainForwardResult{}, fmt.Errorf("fake attention resident train forward: missing or mismatched resident binding")
	}
	scale := req.Scale
	if scale == 0 {
		scale = 1
	}

	query := make([]float32, rows*d)
	key := make([]float32, rows*d)
	value := make([]float32, rows*d)
	fillHostMatMul(req.Input.F32, rows, d, wq.F32, d, query)
	fillHostMatMul(req.Input.F32, rows, d, wk.F32, d, key)
	fillHostMatMul(req.Input.F32, rows, d, wv.F32, d, value)

	scores := make([]float32, batch*seq*seq)
	mixed := make([]float32, rows*d)
	for b := 0; b < batch; b++ {
		base := b * seq * d
		scoreBase := b * seq * seq
		qb := query[base : base+seq*d]
		kb := key[base : base+seq*d]
		vb := value[base : base+seq*d]
		sb := scores[scoreBase : scoreBase+seq*seq]
		fillHostMatMulTranspose(qb, seq, d, kb, seq, d, false, true, sb)
		scaleFloat32Slice(sb, scale)
		if req.Mask != nil {
			if len(req.Mask[b]) != seq {
				return backend.AttentionResidentTrainForwardResult{}, fmt.Errorf("fake attention resident train forward: mask sequence %d has %d entries, want %d", b, len(req.Mask[b]), seq)
			}
			softmaxRowsMaskedColumnsInPlace(sb, seq, seq, req.Mask[b])
		} else {
			softmaxRowsInPlace(sb, seq, seq)
		}
		mb := mixed[base : base+seq*d]
		fillHostMatMul(sb, seq, seq, vb, d, mb)
	}

	if a.attnResidentTrainHandles == nil {
		a.attnResidentTrainHandles = map[int]*fakeAttentionResidentTrainHandleState{}
	}
	a.attnResidentTrainNextID++
	id := a.attnResidentTrainNextID
	token := &fakeAttentionResidentTrainHandleToken{id: id, alive: true}
	a.attnResidentTrainHandles[id] = &fakeAttentionResidentTrainHandleState{
		token:  token,
		seqLen: seq,
		d:      d,
		scale:  scale,
		input:  append([]float32(nil), req.Input.F32...),
		q:      query,
		k:      key,
		v:      value,
		probs:  scores,
		wq:     append([]float32(nil), wq.F32...),
		wk:     append([]float32(nil), wk.F32...),
		wv:     append([]float32(nil), wv.F32...),
	}

	a.uploadedBytes += int64(len(req.Input.F32) * 4)
	a.downloadedBytes += int64((len(query) + len(key) + len(value) + len(scores) + len(mixed)) * 4)

	return backend.AttentionResidentTrainForwardResult{
		Handle: backend.AttentionResidentTrainHandle{
			Backend:  eosartifact.BackendCUDA,
			Token:    token,
			Batch:    batch,
			SeqLen:   seq,
			ModelDim: d,
			StepID:   req.StepID,
		},
		Query:  backend.NewTensorF32([]int{rows, d}, query),
		Key:    backend.NewTensorF32([]int{rows, d}, key),
		Value:  backend.NewTensorF32([]int{rows, d}, value),
		Scores: backend.NewTensorF32([]int{rows, seq}, scores),
		Mixed:  backend.NewTensorF32([]int{rows, d}, mixed),
	}, nil
}

// RunAttentionBlockResidentTrainBackward implements
// backend.AttentionResidentTrainAccelerator, independently re-deriving the
// same Q/K/V-path algebra backpropAttentionSequences computes (gradScores,
// gradPreSoftmax via backwardSoftmaxRow, gradV/gradK/gradQ, gradWq/gradWk/
// gradWv, gradInput) using the state RunAttentionBlockResidentTrainForward
// stashed under the handle, then always releases the handle.
func (a *countingMatMulAccelerator) RunAttentionBlockResidentTrainBackward(req backend.AttentionResidentTrainBackwardRequest) (backend.AttentionResidentTrainBackwardResult, error) {
	a.attnResidentTrainBackwardRuns++
	token, ok := req.Handle.Token.(*fakeAttentionResidentTrainHandleToken)
	if !ok || token == nil || !token.alive {
		return backend.AttentionResidentTrainBackwardResult{}, fmt.Errorf("fake attention resident train backward: stale or unknown handle")
	}
	state := a.attnResidentTrainHandles[token.id]
	if state == nil {
		return backend.AttentionResidentTrainBackwardResult{}, fmt.Errorf("fake attention resident train backward: handle %d is not registered", token.id)
	}
	defer func() {
		token.alive = false
		delete(a.attnResidentTrainHandles, token.id)
	}()
	seq, d := state.seqLen, state.d
	if req.GradMixed == nil || len(req.GradMixed.F32) != seq*d {
		return backend.AttentionResidentTrainBackwardResult{}, fmt.Errorf("fake attention resident train backward: gradMixed shape mismatch")
	}

	gradScores := make([]float32, seq*seq)
	fillHostMatMulTranspose(req.GradMixed.F32, seq, d, state.v, seq, d, false, true, gradScores)
	gradPreSoftmax := make([]float32, seq*seq)
	for row := 0; row < seq; row++ {
		backwardSoftmaxRow(gradPreSoftmax[row*seq:(row+1)*seq], gradScores[row*seq:(row+1)*seq], state.probs[row*seq:(row+1)*seq])
	}
	scaleFloat32Slice(gradPreSoftmax, state.scale)

	gradV := make([]float32, seq*d)
	fillHostMatMulTranspose(state.probs, seq, seq, req.GradMixed.F32, seq, d, true, false, gradV)
	gradK := make([]float32, seq*d)
	fillHostMatMulTranspose(gradPreSoftmax, seq, seq, state.q, seq, d, true, false, gradK)
	gradQ := make([]float32, seq*d)
	fillHostMatMul(gradPreSoftmax, seq, seq, state.k, d, gradQ)

	gradWq := make([]float32, d*d)
	fillHostMatMulTranspose(state.input, seq, d, gradQ, seq, d, true, false, gradWq)
	gradWk := make([]float32, d*d)
	fillHostMatMulTranspose(state.input, seq, d, gradK, seq, d, true, false, gradWk)
	gradWv := make([]float32, d*d)
	fillHostMatMulTranspose(state.input, seq, d, gradV, seq, d, true, false, gradWv)

	gradInput := make([]float32, seq*d)
	fillHostMatMulTranspose(gradQ, seq, d, state.wq, d, d, false, true, gradInput)
	term2 := make([]float32, seq*d)
	fillHostMatMulTranspose(gradK, seq, d, state.wk, d, d, false, true, term2)
	term3 := make([]float32, seq*d)
	fillHostMatMulTranspose(gradV, seq, d, state.wv, d, d, false, true, term3)
	addFloat32Slice(gradInput, term2)
	addFloat32Slice(gradInput, term3)

	a.uploadedBytes += int64(len(req.GradMixed.F32) * 4)
	a.downloadedBytes += int64((len(gradInput) + len(gradWq) + len(gradWk) + len(gradWv)) * 4)

	return backend.AttentionResidentTrainBackwardResult{
		GradInput:       backend.NewTensorF32([]int{seq, d}, gradInput),
		GradQueryWeight: backend.NewTensorF32([]int{d, d}, gradWq),
		GradKeyWeight:   backend.NewTensorF32([]int{d, d}, gradWk),
		GradValueWeight: backend.NewTensorF32([]int{d, d}, gradWv),
	}, nil
}

// BeginAttentionResidentTrainStep implements backend.AttentionResidentTrainAccelerator.
// It defensively releases any handle left over from a prior step, mirroring
// the CUDA accelerator's generation bump.
func (a *countingMatMulAccelerator) BeginAttentionResidentTrainStep(stepID uint64) error {
	a.releaseAllFakeAttentionResidentTrainHandles()
	a.attnResidentTrainStepID = stepID
	a.attnResidentTrainStepActive = true
	a.attnResidentTrainBeginCalls++
	return nil
}

// EndAttentionResidentTrainStep implements backend.AttentionResidentTrainAccelerator.
func (a *countingMatMulAccelerator) EndAttentionResidentTrainStep(stepID uint64) error {
	if !a.attnResidentTrainStepActive || a.attnResidentTrainStepID != stepID {
		return fmt.Errorf("fake attention resident train: step %d is not the active step", stepID)
	}
	a.releaseAllFakeAttentionResidentTrainHandles()
	a.attnResidentTrainStepActive = false
	a.attnResidentTrainEndCalls++
	return nil
}

// AbortAttentionResidentTrainStep implements backend.AttentionResidentTrainAccelerator.
func (a *countingMatMulAccelerator) AbortAttentionResidentTrainStep(stepID uint64) error {
	if !a.attnResidentTrainStepActive || a.attnResidentTrainStepID != stepID {
		return fmt.Errorf("fake attention resident train: step %d is not the active step", stepID)
	}
	a.releaseAllFakeAttentionResidentTrainHandles()
	a.attnResidentTrainStepActive = false
	a.attnResidentTrainAbortCalls++
	return nil
}

// ReleaseAttentionResidentTrainHandle implements backend.AttentionResidentTrainAccelerator.
func (a *countingMatMulAccelerator) ReleaseAttentionResidentTrainHandle(handle backend.AttentionResidentTrainHandle) error {
	token, ok := handle.Token.(*fakeAttentionResidentTrainHandleToken)
	if !ok || token == nil {
		return nil
	}
	token.alive = false
	delete(a.attnResidentTrainHandles, token.id)
	a.attnResidentTrainReleaseCalls++
	return nil
}

func (a *countingMatMulAccelerator) releaseAllFakeAttentionResidentTrainHandles() {
	for id, state := range a.attnResidentTrainHandles {
		state.token.alive = false
		delete(a.attnResidentTrainHandles, id)
	}
}

// tinyRaggedAttnResidentContrastiveBatch returns a 2-token-sequence
// contrastive batch where every query/positive mask has its final position
// zeroed except one, exercising the "key" mask mode's actual masking effect
// (tinyAttnResidentContrastiveBatch's masks are all-ones, a no-op).
func tinyRaggedAttnResidentContrastiveBatch() []EmbeddingContrastiveExample {
	return []EmbeddingContrastiveExample{
		{QueryTokens: []int32{0, 2}, PositiveTokens: []int32{0, 0}, QueryMask: []int32{1, 1}, PositiveMask: []int32{1, 0}},
		{QueryTokens: []int32{1, 2}, PositiveTokens: []int32{1, 1}, QueryMask: []int32{1, 0}, PositiveMask: []int32{1, 1}},
		{QueryTokens: []int32{2, 0}, PositiveTokens: []int32{0, 2}, QueryMask: []int32{1, 1}, PositiveMask: []int32{1, 1}},
	}
}

// newAttnResidentTrainTestTrainer builds a fresh trainer + fake accelerator
// pair configured for the shipped default model's attention shape: masked
// ("key"), scaled, with attention residual and layer norm enabled, matching
// models/default_embedding.go's DefaultEmbeddingManifest.
func newAttnResidentTrainTestTrainer(t *testing.T) (*EmbeddingTrainer, *countingMatMulAccelerator) {
	t.Helper()
	trainer := newTinyTrainableAttentionEmbeddingTrainer(t, 0.05)
	if trainer.forwardMatMul != nil {
		trainer.forwardMatMul.Close()
	}
	fake := &countingMatMulAccelerator{}
	trainer.forwardMatMul = fake
	trainer.manifest.AttentionMaskMode = EmbeddingAttentionMaskModeKey
	trainer.manifest.AttentionScoreScale = EmbeddingAttentionScoreScaleKeyDimRSQ
	trainer.manifest.AttentionResidual = true
	trainer.manifest.AttentionLayerNorm = true
	trainer.config.ContrastiveLoss = "infonce"
	trainer.config.Temperature = 0.05
	return trainer, fake
}

// TestEmbeddingTrainerAttnResidentTrainBackwardMatchesHostStep is the S3(b)
// end-to-end parity test: one full TrainContrastiveStep (forward, backward,
// optimizer update) with the resident train fast path enabled must land on
// the same post-step weights as the identical step run entirely on the
// host path, within 1e-4, across a ragged-mask batch (some positions
// actually masked out, not just all-ones) and the shipped default model's
// attention_residual/attention_layernorm=true configuration.
func TestEmbeddingTrainerAttnResidentTrainBackwardMatchesHostStep(t *testing.T) {
	t.Setenv("EOS_EMBED_ATTN_RESIDENT", "1")
	resident, residentFake := newAttnResidentTrainTestTrainer(t)
	batch := tinyRaggedAttnResidentContrastiveBatch()
	if _, err := resident.TrainContrastiveStep(batch); err != nil {
		t.Fatalf("resident train step: %v", err)
	}
	if residentFake.attnResidentTrainForwardRuns == 0 {
		t.Fatal("expected the resident-train forward fast path to run")
	}
	if residentFake.attnResidentTrainBackwardRuns == 0 {
		t.Fatal("expected the resident-train backward fast path to run")
	}
	if residentFake.attnResidentTrainBeginCalls == 0 || residentFake.attnResidentTrainEndCalls == 0 {
		t.Fatalf("begin/end calls = %d/%d, want at least one each", residentFake.attnResidentTrainBeginCalls, residentFake.attnResidentTrainEndCalls)
	}
	if len(residentFake.attnResidentTrainHandles) != 0 {
		t.Fatalf("%d resident-train handles still live after the step; expected all released", len(residentFake.attnResidentTrainHandles))
	}

	t.Setenv("EOS_EMBED_ATTN_RESIDENT", "0")
	host, hostFake := newAttnResidentTrainTestTrainer(t)
	if _, err := host.TrainContrastiveStep(batch); err != nil {
		t.Fatalf("host train step: %v", err)
	}
	if hostFake.attnResidentTrainForwardRuns != 0 {
		t.Fatalf("host run unexpectedly used the resident-train forward path %d times", hostFake.attnResidentTrainForwardRuns)
	}

	assertCloseF32Slice(t, "attn_q", resident.attentionQuery.F32, host.attentionQuery.F32, 1e-4)
	assertCloseF32Slice(t, "attn_k", resident.attentionKey.F32, host.attentionKey.F32, 1e-4)
	assertCloseF32Slice(t, "attn_v", resident.attentionValue.F32, host.attentionValue.F32, 1e-4)
	assertCloseF32Slice(t, "attn_o", resident.attentionOutput.F32, host.attentionOutput.F32, 1e-4)
	assertCloseF32Slice(t, "token_embedding", resident.tokenEmbed.F32, host.tokenEmbed.F32, 1e-4)
	assertCloseF32Slice(t, "projection", resident.projection.F32, host.projection.F32, 1e-4)
}

// TestEmbeddingTrainerAttnResidentTrainDisabledByDefault confirms
// EOS_EMBED_ATTN_RESIDENT defaults OFF for the S3(b) backward path too: an
// accelerator implementing AttentionResidentTrainAccelerator must not be
// used unless the flag is explicitly set.
func TestEmbeddingTrainerAttnResidentTrainDisabledByDefault(t *testing.T) {
	trainer, fake := newAttnResidentTrainTestTrainer(t)
	if _, err := trainer.TrainContrastiveStep(tinyRaggedAttnResidentContrastiveBatch()); err != nil {
		t.Fatalf("train step: %v", err)
	}
	if fake.attnResidentTrainForwardRuns != 0 || fake.attnResidentTrainBackwardRuns != 0 {
		t.Fatalf("resident-train forward/backward runs = %d/%d, want 0/0 when EOS_EMBED_ATTN_RESIDENT is unset", fake.attnResidentTrainForwardRuns, fake.attnResidentTrainBackwardRuns)
	}
	if fake.attnResidentTrainBeginCalls != 0 {
		t.Fatalf("resident-train begin calls = %d, want 0 when the flag is off", fake.attnResidentTrainBeginCalls)
	}
}

// TestEmbeddingTrainerAttnResidentTrainStepBoundaryInvalidatesPriorBatch is
// the step-boundary invalidation guard at the trainer level: a second
// TrainContrastiveStep call must not reuse or otherwise observe the first
// call's resident activations -- every handle from step N is gone (its
// backward already consumed and released it, or, for a handle backward
// never reached, BeginAttentionResidentTrainStep's own defensive sweep
// released it) by the time step N+1's Begin runs.
func TestEmbeddingTrainerAttnResidentTrainStepBoundaryInvalidatesPriorBatch(t *testing.T) {
	t.Setenv("EOS_EMBED_ATTN_RESIDENT", "1")
	trainer, fake := newAttnResidentTrainTestTrainer(t)
	batch := tinyRaggedAttnResidentContrastiveBatch()

	if _, err := trainer.TrainContrastiveStep(batch); err != nil {
		t.Fatalf("first train step: %v", err)
	}
	firstStepID := fake.attnResidentTrainStepID
	if len(fake.attnResidentTrainHandles) != 0 {
		t.Fatalf("%d handles still live after the first step", len(fake.attnResidentTrainHandles))
	}

	if _, err := trainer.TrainContrastiveStep(batch); err != nil {
		t.Fatalf("second train step: %v", err)
	}
	if fake.attnResidentTrainStepID == firstStepID {
		t.Fatalf("second step reused stepID %d from the first step", firstStepID)
	}
	if len(fake.attnResidentTrainHandles) != 0 {
		t.Fatalf("%d handles still live after the second step", len(fake.attnResidentTrainHandles))
	}
	if fake.attnResidentTrainBeginCalls != 2 {
		t.Fatalf("begin calls = %d, want 2 (one per step)", fake.attnResidentTrainBeginCalls)
	}
}

// TestEmbeddingTrainerAttnResidentTrainForwardAfterOptimizerStepUsesFreshWeights
// is the optimizer-step-then-forward guard: a resident-train step that runs
// immediately after a prior optimizer update must reflect the updated
// Wq/Wk/Wv, not weights cached from before the update. Since
// TrainContrastiveStep primes forward-weight residency (and therefore, via
// primeAttentionResidentQueryBinding's shared invalidation gate, the plain
// Wq/Wk/Wv bindings the resident-train path reads) before every forward
// pass, two consecutive real steps -- which each apply an optimizer update
// -- must produce two different loss values rather than the second
// silently repeating the first.
func TestEmbeddingTrainerAttnResidentTrainForwardAfterOptimizerStepUsesFreshWeights(t *testing.T) {
	t.Setenv("EOS_EMBED_ATTN_RESIDENT", "1")
	trainer, fake := newAttnResidentTrainTestTrainer(t)
	batch := tinyRaggedAttnResidentContrastiveBatch()

	first, err := trainer.TrainContrastiveStep(batch)
	if err != nil {
		t.Fatalf("first train step: %v", err)
	}
	if fake.attnResidentTrainForwardRuns == 0 {
		t.Fatal("expected the resident-train forward fast path to run on the first step")
	}
	firstAttnQ := append([]float32(nil), trainer.attentionQuery.F32...)

	second, err := trainer.TrainContrastiveStep(batch)
	if err != nil {
		t.Fatalf("second train step: %v", err)
	}
	if math.IsNaN(float64(first.Loss)) || math.IsNaN(float64(second.Loss)) {
		t.Fatalf("nan loss: first=%v second=%v", first.Loss, second.Loss)
	}
	if math.Abs(float64(second.Loss-first.Loss)) < 1e-7 && sliceAllCloseForTest(trainer.attentionQuery.F32, firstAttnQ, 1e-9) {
		t.Fatal("second step's loss and Wq did not move from the first step's; resident-train forward may be reusing stale weights")
	}
}

func sliceAllCloseForTest(a, b []float32, tol float32) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if abs32(a[i]-b[i]) > tol {
			return false
		}
	}
	return true
}
