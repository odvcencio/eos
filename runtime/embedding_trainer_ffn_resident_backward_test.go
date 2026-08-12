package eosruntime

import (
	"fmt"
	"testing"

	eosartifact "m31labs.dev/eos/artifact/eos"
	"m31labs.dev/eos/runtime/backend"
)

// fakeFFNResidentTrainHandleToken implements
// backend.FFNResidentTrainHandleToken on the shared countingMatMulAccelerator
// test fake, mirroring fakeAttentionResidentTrainHandleToken.
type fakeFFNResidentTrainHandleToken struct {
	id    int
	alive bool
}

func (tk *fakeFFNResidentTrainHandleToken) FFNResidentTrainHandleToken() {}
func (tk *fakeFFNResidentTrainHandleToken) Backend() eosartifact.BackendKind {
	return eosartifact.BackendCUDA
}
func (tk *fakeFFNResidentTrainHandleToken) Generation() uint64 { return 0 }
func (tk *fakeFFNResidentTrainHandleToken) StepID() uint64     { return 0 }
func (tk *fakeFFNResidentTrainHandleToken) Alive() bool        { return tk != nil && tk.alive }

// fakeFFNResidentTrainHandleState is the device-resident data one
// RunFFNBlockResidentTrainForward call would keep on the device; the fake
// keeps it as plain host slices instead, mirroring
// fakeAttentionResidentTrainHandleState.
type fakeFFNResidentTrainHandleState struct {
	token                *fakeFFNResidentTrainHandleToken
	seqLen, d, h, e      int
	input                []float32
	ffnHidden, activated []float32
	wup, wdown           []float32
}

// RunFFNBlockResidentTrainForward implements
// backend.FFNResidentTrainAccelerator on countingMatMulAccelerator: a
// correct, hardware-independent reference computation using the exact
// (tanh) GELU, matching the production accelerator's FastGELU=false-only
// contract.
func (a *countingMatMulAccelerator) RunFFNBlockResidentTrainForward(req backend.FFNResidentTrainForwardRequest) (backend.FFNResidentTrainForwardResult, error) {
	a.ffnResidentTrainForwardRuns++
	if !a.attnResidentTrainStepActive || a.attnResidentTrainStepID != req.StepID {
		return backend.FFNResidentTrainForwardResult{}, fmt.Errorf("fake ffn resident train forward: step %d is not active", req.StepID)
	}
	if req.FastGELU {
		return backend.FFNResidentTrainForwardResult{}, fmt.Errorf("fake ffn resident train forward: fast GELU is not supported")
	}
	seqLen, d, h, e := req.SeqLen, req.InputDim, req.HiddenDim, req.OutputDim
	if req.Input == nil || seqLen <= 0 || d <= 0 || h <= 0 || e <= 0 {
		return backend.FFNResidentTrainForwardResult{}, fmt.Errorf("fake ffn resident train forward: invalid request")
	}
	if len(req.Input.F32) != seqLen*d {
		return backend.FFNResidentTrainForwardResult{}, fmt.Errorf("fake ffn resident train forward: input shape mismatch")
	}
	wup := a.bound[req.HiddenWeightName]
	wdown := a.bound[req.OutputWeightName]
	if wup == nil || wdown == nil || len(wup.F32) != d*h || len(wdown.F32) != h*e {
		return backend.FFNResidentTrainForwardResult{}, fmt.Errorf("fake ffn resident train forward: missing or mismatched resident binding")
	}

	ffnHidden := make([]float32, seqLen*h)
	fillHostMatMul(req.Input.F32, seqLen, d, wup.F32, h, ffnHidden)
	activated := make([]float32, seqLen*h)
	fillGELUForward(activated, ffnHidden, false)
	ffnOutput := make([]float32, seqLen*e)
	fillHostMatMul(activated, seqLen, h, wdown.F32, e, ffnOutput)

	if a.ffnResidentTrainHandles == nil {
		a.ffnResidentTrainHandles = map[int]*fakeFFNResidentTrainHandleState{}
	}
	a.ffnResidentTrainNextID++
	id := a.ffnResidentTrainNextID
	token := &fakeFFNResidentTrainHandleToken{id: id, alive: true}
	a.ffnResidentTrainHandles[id] = &fakeFFNResidentTrainHandleState{
		token:     token,
		seqLen:    seqLen,
		d:         d,
		h:         h,
		e:         e,
		input:     append([]float32(nil), req.Input.F32...),
		ffnHidden: ffnHidden,
		activated: activated,
		wup:       append([]float32(nil), wup.F32...),
		wdown:     append([]float32(nil), wdown.F32...),
	}

	a.uploadedBytes += int64(len(req.Input.F32) * 4)
	a.downloadedBytes += int64((len(ffnHidden) + len(activated) + len(ffnOutput)) * 4)

	return backend.FFNResidentTrainForwardResult{
		Handle: backend.FFNResidentTrainHandle{
			Backend:   eosartifact.BackendCUDA,
			Token:     token,
			SeqLen:    seqLen,
			InputDim:  d,
			HiddenDim: h,
			OutputDim: e,
			StepID:    req.StepID,
		},
		FFNHidden: backend.NewTensorF32([]int{seqLen, h}, ffnHidden),
		Activated: backend.NewTensorF32([]int{seqLen, h}, activated),
		FFNOutput: backend.NewTensorF32([]int{seqLen, e}, ffnOutput),
	}, nil
}

// RunFFNBlockResidentTrainBackward implements
// backend.FFNResidentTrainAccelerator, independently re-deriving the same
// hidden-projection/GELU/output-projection algebra
// backpropProjectedFFNSequence computes, using the state
// RunFFNBlockResidentTrainForward stashed under the handle, then always
// releases the handle.
func (a *countingMatMulAccelerator) RunFFNBlockResidentTrainBackward(req backend.FFNResidentTrainBackwardRequest) (backend.FFNResidentTrainBackwardResult, error) {
	a.ffnResidentTrainBackwardRuns++
	token, ok := req.Handle.Token.(*fakeFFNResidentTrainHandleToken)
	if !ok || token == nil || !token.alive {
		return backend.FFNResidentTrainBackwardResult{}, fmt.Errorf("fake ffn resident train backward: stale or unknown handle")
	}
	state := a.ffnResidentTrainHandles[token.id]
	if state == nil {
		return backend.FFNResidentTrainBackwardResult{}, fmt.Errorf("fake ffn resident train backward: handle %d is not registered", token.id)
	}
	defer func() {
		token.alive = false
		delete(a.ffnResidentTrainHandles, token.id)
	}()
	seqLen, d, h, e := state.seqLen, state.d, state.h, state.e
	if req.GradFFNOutput == nil || len(req.GradFFNOutput.F32) != seqLen*e {
		return backend.FFNResidentTrainBackwardResult{}, fmt.Errorf("fake ffn resident train backward: gradFFNOutput shape mismatch")
	}

	gradOutputWeight := make([]float32, h*e)
	fillHostMatMulTranspose(state.activated, seqLen, h, req.GradFFNOutput.F32, seqLen, e, true, false, gradOutputWeight)
	gradActivatedPre := make([]float32, seqLen*h)
	fillHostMatMulTranspose(req.GradFFNOutput.F32, seqLen, e, state.wdown, h, e, false, true, gradActivatedPre)
	gradActivated := make([]float32, seqLen*h)
	fillGELUBackwardMul(gradActivated, gradActivatedPre, state.ffnHidden, false)
	gradHiddenWeight := make([]float32, d*h)
	fillHostMatMulTranspose(state.input, seqLen, d, gradActivated, seqLen, h, true, false, gradHiddenWeight)
	gradInput := make([]float32, seqLen*d)
	fillHostMatMulTranspose(gradActivated, seqLen, h, state.wup, d, h, false, true, gradInput)

	a.uploadedBytes += int64(len(req.GradFFNOutput.F32) * 4)
	a.downloadedBytes += int64((len(gradInput) + len(gradHiddenWeight) + len(gradOutputWeight)) * 4)

	return backend.FFNResidentTrainBackwardResult{
		GradInput:        backend.NewTensorF32([]int{seqLen, d}, gradInput),
		GradHiddenWeight: backend.NewTensorF32([]int{d, h}, gradHiddenWeight),
		GradOutputWeight: backend.NewTensorF32([]int{h, e}, gradOutputWeight),
	}, nil
}

// ReleaseFFNResidentTrainHandle implements backend.FFNResidentTrainAccelerator.
func (a *countingMatMulAccelerator) ReleaseFFNResidentTrainHandle(handle backend.FFNResidentTrainHandle) error {
	token, ok := handle.Token.(*fakeFFNResidentTrainHandleToken)
	if !ok || token == nil {
		return nil
	}
	token.alive = false
	delete(a.ffnResidentTrainHandles, token.id)
	a.ffnResidentTrainReleaseCalls++
	return nil
}

// newFFNResidentTrainTestTrainer builds a fresh trainer + fake accelerator
// pair with both an attention block and an FFN block (hidden projection,
// GELU, output projection), matching the shipped default model's shape
// (attention residual+layernorm, FFN residual+layernorm, masked "key"
// attention). EOS_TRAIN_DISABLE_BATCHED_BACKWARD forces the per-sequence
// backward path (backpropAttentionSequence/backpropProjectedFFNSequence),
// matching the shipped default (role-conditioned) model's behavior even
// though this tiny test trainer has no role embedding of its own -- see
// tryFFNBlockResidentTrain's doc comment for why the per-sequence path is
// the integration point that matters in production.
func newFFNResidentTrainTestTrainer(t *testing.T) (*EmbeddingTrainer, *countingMatMulAccelerator) {
	t.Helper()
	t.Setenv("EOS_TRAIN_DISABLE_BATCHED_BACKWARD", "1")
	trainer := newTinyTrainableEncoderEmbeddingTrainer(t, 0.05)
	if trainer.forwardMatMul != nil {
		trainer.forwardMatMul.Close()
	}
	fake := &countingMatMulAccelerator{}
	trainer.forwardMatMul = fake
	trainer.manifest.AttentionMaskMode = EmbeddingAttentionMaskModeKey
	trainer.manifest.AttentionScoreScale = EmbeddingAttentionScoreScaleKeyDimRSQ
	trainer.config.ContrastiveLoss = "infonce"
	trainer.config.Temperature = 0.05
	return trainer, fake
}

// tinyRaggedFFNResidentContrastiveBatch mirrors
// tinyRaggedAttnResidentContrastiveBatch: a small contrastive batch with a
// mix of unmasked and ragged (partially masked) sequences, so both the
// attention and FFN resident paths exercise a real masking effect, not just
// an all-ones no-op mask.
func tinyRaggedFFNResidentContrastiveBatch() []EmbeddingContrastiveExample {
	return []EmbeddingContrastiveExample{
		{QueryTokens: []int32{0, 2}, PositiveTokens: []int32{0, 0}, QueryMask: []int32{1, 1}, PositiveMask: []int32{1, 0}},
		{QueryTokens: []int32{1, 2}, PositiveTokens: []int32{1, 1}, QueryMask: []int32{1, 0}, PositiveMask: []int32{1, 1}},
		{QueryTokens: []int32{2, 0}, PositiveTokens: []int32{0, 2}, QueryMask: []int32{1, 1}, PositiveMask: []int32{1, 1}},
	}
}

// TestEmbeddingTrainerFFNResidentTrainBackwardMatchesHostStep is the S3(c)
// FFN end-to-end parity test: one full TrainContrastiveStep (forward,
// backward, optimizer update) with the resident train fast path enabled
// must land on the same post-step weights as the identical step run
// entirely on the host path, within 1e-4, across a ragged-mask batch, with
// both attention and FFN blocks device-resident.
func TestEmbeddingTrainerFFNResidentTrainBackwardMatchesHostStep(t *testing.T) {
	t.Setenv("EOS_EMBED_ATTN_RESIDENT", "1")
	resident, residentFake := newFFNResidentTrainTestTrainer(t)
	batch := tinyRaggedFFNResidentContrastiveBatch()
	if _, err := resident.TrainContrastiveStep(batch); err != nil {
		t.Fatalf("resident train step: %v", err)
	}
	if residentFake.ffnResidentTrainForwardRuns == 0 {
		t.Fatal("expected the FFN resident-train forward fast path to run")
	}
	if residentFake.ffnResidentTrainBackwardRuns == 0 {
		t.Fatal("expected the FFN resident-train backward fast path to run")
	}
	if residentFake.attnResidentTrainForwardRuns == 0 || residentFake.attnResidentTrainBackwardRuns == 0 {
		t.Fatal("expected the attention resident-train fast path to also run alongside FFN")
	}
	if len(residentFake.ffnResidentTrainHandles) != 0 {
		t.Fatalf("%d FFN resident-train handles still live after the step; expected all released", len(residentFake.ffnResidentTrainHandles))
	}

	t.Setenv("EOS_EMBED_ATTN_RESIDENT", "0")
	host, hostFake := newFFNResidentTrainTestTrainer(t)
	if _, err := host.TrainContrastiveStep(batch); err != nil {
		t.Fatalf("host train step: %v", err)
	}
	if hostFake.ffnResidentTrainForwardRuns != 0 {
		t.Fatalf("host run unexpectedly used the FFN resident-train forward path %d times", hostFake.ffnResidentTrainForwardRuns)
	}

	assertCloseF32Slice(t, "attn_q", resident.attentionQuery.F32, host.attentionQuery.F32, 1e-4)
	assertCloseF32Slice(t, "attn_k", resident.attentionKey.F32, host.attentionKey.F32, 1e-4)
	assertCloseF32Slice(t, "attn_v", resident.attentionValue.F32, host.attentionValue.F32, 1e-4)
	assertCloseF32Slice(t, "attn_o", resident.attentionOutput.F32, host.attentionOutput.F32, 1e-4)
	assertCloseF32Slice(t, "ffn_up", resident.hiddenProjection.F32, host.hiddenProjection.F32, 1e-4)
	assertCloseF32Slice(t, "projection", resident.projection.F32, host.projection.F32, 1e-4)
	assertCloseF32Slice(t, "token_embedding", resident.tokenEmbed.F32, host.tokenEmbed.F32, 1e-4)
}

// TestEmbeddingTrainerFFNResidentTrainDisabledByDefault confirms
// EOS_EMBED_ATTN_RESIDENT defaults OFF for the S3(c) FFN path too.
func TestEmbeddingTrainerFFNResidentTrainDisabledByDefault(t *testing.T) {
	trainer, fake := newFFNResidentTrainTestTrainer(t)
	if _, err := trainer.TrainContrastiveStep(tinyRaggedFFNResidentContrastiveBatch()); err != nil {
		t.Fatalf("train step: %v", err)
	}
	if fake.ffnResidentTrainForwardRuns != 0 || fake.ffnResidentTrainBackwardRuns != 0 {
		t.Fatalf("FFN resident-train forward/backward runs = %d/%d, want 0/0 when EOS_EMBED_ATTN_RESIDENT is unset", fake.ffnResidentTrainForwardRuns, fake.ffnResidentTrainBackwardRuns)
	}
}

// TestEmbeddingTrainerFFNResidentTrainStepBoundaryInvalidatesPriorBatch is
// the step-boundary invalidation guard at the trainer level for FFN
// handles, mirroring
// TestEmbeddingTrainerAttnResidentTrainStepBoundaryInvalidatesPriorBatch.
func TestEmbeddingTrainerFFNResidentTrainStepBoundaryInvalidatesPriorBatch(t *testing.T) {
	t.Setenv("EOS_EMBED_ATTN_RESIDENT", "1")
	trainer, fake := newFFNResidentTrainTestTrainer(t)
	batch := tinyRaggedFFNResidentContrastiveBatch()

	if _, err := trainer.TrainContrastiveStep(batch); err != nil {
		t.Fatalf("first train step: %v", err)
	}
	if len(fake.ffnResidentTrainHandles) != 0 {
		t.Fatalf("%d FFN handles still live after the first step", len(fake.ffnResidentTrainHandles))
	}

	if _, err := trainer.TrainContrastiveStep(batch); err != nil {
		t.Fatalf("second train step: %v", err)
	}
	if len(fake.ffnResidentTrainHandles) != 0 {
		t.Fatalf("%d FFN handles still live after the second step", len(fake.ffnResidentTrainHandles))
	}
}

// TestEmbeddingTrainerFFNResidentTrainDuplicateReferenceMatchesHost exercises
// the duplicate-content case cachedEmbeddingBatchSequence's per-batch dedup
// cache creates: two contrastive examples sharing identical
// (tokens, mask, role) text so the same cached embeddingBatchSequence's
// states backpropagate twice in one step. The first backward reference
// consumes the resident-train handle; the second must fall back to the
// still-valid downloaded host ffnHidden/activated copies -- which stage (c)
// leaves in place for FFN precisely so this stays correct without any
// duplicate-reference tracking (see tryFFNBlockResidentTrain's doc comment).
func TestEmbeddingTrainerFFNResidentTrainDuplicateReferenceMatchesHost(t *testing.T) {
	t.Setenv("EOS_EMBED_ATTN_RESIDENT", "1")
	resident, residentFake := newFFNResidentTrainTestTrainer(t)
	dup := EmbeddingContrastiveExample{QueryTokens: []int32{0, 2}, PositiveTokens: []int32{1, 2}, QueryMask: []int32{1, 1}, PositiveMask: []int32{1, 1}}
	other := EmbeddingContrastiveExample{QueryTokens: []int32{0, 2}, PositiveTokens: []int32{2, 1}, QueryMask: []int32{1, 1}, PositiveMask: []int32{1, 1}}
	batch := []EmbeddingContrastiveExample{dup, other}
	if _, err := resident.TrainContrastiveStep(batch); err != nil {
		t.Fatalf("resident train step: %v", err)
	}
	if residentFake.ffnResidentTrainForwardRuns == 0 {
		t.Fatal("expected the FFN resident-train forward fast path to run")
	}
	if len(residentFake.ffnResidentTrainHandles) != 0 {
		t.Fatalf("%d FFN resident-train handles still live after the step; expected all released", len(residentFake.ffnResidentTrainHandles))
	}

	t.Setenv("EOS_EMBED_ATTN_RESIDENT", "0")
	host, _ := newFFNResidentTrainTestTrainer(t)
	if _, err := host.TrainContrastiveStep(batch); err != nil {
		t.Fatalf("host train step: %v", err)
	}

	assertCloseF32Slice(t, "ffn_up", resident.hiddenProjection.F32, host.hiddenProjection.F32, 1e-4)
	assertCloseF32Slice(t, "projection", resident.projection.F32, host.projection.F32, 1e-4)
	assertCloseF32Slice(t, "token_embedding", resident.tokenEmbed.F32, host.tokenEmbed.F32, 1e-4)
}
