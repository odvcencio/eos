//go:build linux && cgo

package cuda

/*
#cgo CFLAGS: -I/usr/local/cuda/include
#include <cuda.h>
*/
import "C"

import (
	"fmt"
	"time"

	eosartifact "m31labs.dev/eos/artifact/eos"
	"m31labs.dev/eos/runtime/backend"
)

// attentionResidentTrainState is the S3(b) attention-resident-train handle
// registry, embedded in deviceRuntime. It mirrors CompactTrainAccelerator's
// step/generation bookkeeping: BeginAttentionResidentTrainStep bumps
// generation and defensively frees any handle a prior caller forgot to
// release or end, so a handle from an earlier step can never be read by a
// later one (see releaseAllAttentionResidentTrainHandlesLocked).
type attentionResidentTrainState struct {
	generation uint64
	stepID     uint64
	stepActive bool
	nextHandle uint64
	handles    map[uint64]*attentionResidentTrainHandleState
}

// attentionResidentTrainHandleState is the device-private data one
// RunAttentionBlockResidentTrainForward call registers and the matching
// RunAttentionBlockResidentTrainBackward call consumes. Q, K, V, scores
// (post-softmax probs), and the flattened input stay resident here across
// the forward/backward boundary; mixed is not kept (backward only needs the
// caller-supplied upstream gradMixed, never the forward mixed activation
// itself -- see the package doc comment on runAttentionBlockResidentTrainBackward).
type attentionResidentTrainHandleState struct {
	id         uint64
	generation uint64
	stepID     uint64
	batch      int
	seqLen     int
	modelDim   int
	scale      float32
	queryName  string
	keyName    string
	valueName  string
	inputBuf   C.CUdeviceptr
	qBuf       C.CUdeviceptr
	kBuf       C.CUdeviceptr
	vBuf       C.CUdeviceptr
	scoresBuf  C.CUdeviceptr
	released   bool
}

// attentionResidentTrainToken implements backend.AttentionResidentTrainHandleToken.
type attentionResidentTrainToken struct {
	state *attentionResidentTrainHandleState
}

func (t *attentionResidentTrainToken) AttentionResidentTrainHandleToken() {}

func (t *attentionResidentTrainToken) Backend() eosartifact.BackendKind {
	return eosartifact.BackendCUDA
}

func (t *attentionResidentTrainToken) Generation() uint64 {
	if t == nil || t.state == nil {
		return 0
	}
	return t.state.generation
}

func (t *attentionResidentTrainToken) StepID() uint64 {
	if t == nil || t.state == nil {
		return 0
	}
	return t.state.stepID
}

func (t *attentionResidentTrainToken) Alive() bool {
	return t != nil && t.state != nil && !t.state.released
}

// beginAttentionResidentTrainStep starts a new resident-train step,
// invalidating (and freeing the device buffers of) every handle still
// registered from a previous step -- the stale-state guard a caller that
// skipped End/Abort, or a second batch's Begin racing a first batch's
// leftover handle, both rely on. This also sweeps ffnTrain's registry (see
// ffn_resident_train_accel_linux.go): S3(c) FFN-resident-train handles
// share attention's step generation/stepID/stepActive bookkeeping rather
// than opening a second, parallel step.
func (rt *deviceRuntime) beginAttentionResidentTrainStep(stepID uint64) error {
	if rt == nil || rt.ptr == nil {
		return fmt.Errorf("cuda runtime is not initialized")
	}
	rt.releaseAllAttentionResidentTrainHandles()
	rt.releaseAllFFNResidentTrainHandles()
	rt.attnTrain.generation++
	rt.attnTrain.stepID = stepID
	rt.attnTrain.stepActive = true
	return nil
}

// endOrAbortAttentionResidentTrainStep validates stepID against the active
// step, then releases every handle still registered (a normal occurrence:
// not every forward call is followed by a backward call, for example an
// eval-only encode within a mixed step) and closes the step out. Also
// sweeps ffnTrain's registry (see beginAttentionResidentTrainStep).
func (rt *deviceRuntime) endOrAbortAttentionResidentTrainStep(stepID uint64) error {
	if rt == nil || rt.ptr == nil {
		return fmt.Errorf("cuda runtime is not initialized")
	}
	if !rt.attnTrain.stepActive || rt.attnTrain.stepID != stepID {
		return fmt.Errorf("cuda attention resident train: step %d is not the active step", stepID)
	}
	rt.releaseAllAttentionResidentTrainHandles()
	rt.releaseAllFFNResidentTrainHandles()
	rt.attnTrain.stepActive = false
	return nil
}

// releaseAllAttentionResidentTrainHandles frees every live handle's device
// buffers and clears the registry. Safe to call with zero live handles.
func (rt *deviceRuntime) releaseAllAttentionResidentTrainHandles() {
	if rt == nil {
		return
	}
	for id, state := range rt.attnTrain.handles {
		rt.freeAttentionResidentTrainHandleBuffers(state)
		delete(rt.attnTrain.handles, id)
	}
}

func (rt *deviceRuntime) freeAttentionResidentTrainHandleBuffers(state *attentionResidentTrainHandleState) {
	if rt == nil || state == nil || state.released {
		return
	}
	state.released = true
	for _, ptr := range []C.CUdeviceptr{state.inputBuf, state.qBuf, state.kBuf, state.vBuf, state.scoresBuf} {
		_ = rt.freeBuffer(ptr)
	}
}

// releaseAttentionResidentTrainHandle releases one handle's device buffers
// (idempotent: releasing an already-released or unknown handle is a no-op,
// not an error, matching CompactTrainAccelerator's ReleaseCompactTrainHandle
// tolerance for a handle the caller already consumed via a successful
// backward call).
func (rt *deviceRuntime) releaseAttentionResidentTrainHandle(handle backend.AttentionResidentTrainHandle) error {
	if rt == nil || rt.ptr == nil {
		return fmt.Errorf("cuda runtime is not initialized")
	}
	token, ok := handle.Token.(*attentionResidentTrainToken)
	if !ok || token == nil || token.state == nil {
		return nil
	}
	state := token.state
	rt.freeAttentionResidentTrainHandleBuffers(state)
	delete(rt.attnTrain.handles, state.id)
	return nil
}

// lookupAttentionResidentTrainHandle validates a handle against the current
// generation/step and registry membership before a backward call touches
// its buffers -- the guard behind the step-boundary invalidation contract:
// a handle from a step that has since Begin'd again, or already
// ended/aborted/released, is rejected with an error instead of silently
// reading freed or reused device memory.
func (rt *deviceRuntime) lookupAttentionResidentTrainHandle(handle backend.AttentionResidentTrainHandle) (*attentionResidentTrainHandleState, error) {
	token, ok := handle.Token.(*attentionResidentTrainToken)
	if !ok || token == nil || token.state == nil {
		return nil, fmt.Errorf("cuda attention resident train: handle is not a live cuda resident-train handle")
	}
	state := token.state
	if state.released {
		return nil, fmt.Errorf("cuda attention resident train: handle %d was already released", state.id)
	}
	if state.generation != rt.attnTrain.generation {
		return nil, fmt.Errorf("cuda attention resident train: handle %d is from a prior step generation (stale)", state.id)
	}
	if registered, ok := rt.attnTrain.handles[state.id]; !ok || registered != state {
		return nil, fmt.Errorf("cuda attention resident train: handle %d is not registered", state.id)
	}
	return state, nil
}

func allOnesInt32(n int) []int32 {
	out := make([]int32, n)
	for i := range out {
		out[i] = 1
	}
	return out
}

// runAttentionBlockResidentTrainForward runs one attention block's Q/K/V
// projection, per-sequence scaled dot-product scores, masked forward
// softmax, and value-mix, exactly like runAttentionBlockResident, but keeps
// Q, K, V, scores, and the flattened input device-resident afterward (under
// the returned handle) instead of only using them to compute the
// downloaded results. Q/K/V here are the caller's plain (unscaled) resident
// weight bindings -- req.Scale is applied inside this call's kernels
// instead of being pre-baked into a scaled Q copy -- so the backward chain
// below can mirror the host backpropAttentionSequences algebra term for
// term.
func (rt *deviceRuntime) runAttentionBlockResidentTrainForward(req backend.AttentionResidentTrainForwardRequest) (backend.AttentionResidentTrainForwardResult, error) {
	if rt == nil || rt.ptr == nil {
		return backend.AttentionResidentTrainForwardResult{}, fmt.Errorf("cuda runtime is not initialized")
	}
	if !rt.attnTrain.stepActive || rt.attnTrain.stepID != req.StepID {
		return backend.AttentionResidentTrainForwardResult{}, fmt.Errorf("cuda attention resident train forward: step %d is not the active step", req.StepID)
	}
	batch, seqLen, d := req.Batch, req.SeqLen, req.ModelDim
	if batch <= 0 || seqLen <= 0 || d <= 0 {
		return backend.AttentionResidentTrainForwardResult{}, fmt.Errorf("cuda attention resident train forward requires positive batch, seq_len, and model_dim")
	}
	rows := batch * seqLen
	if req.Input == nil || len(req.Input.F32) != rows*d {
		return backend.AttentionResidentTrainForwardResult{}, fmt.Errorf("cuda attention resident train forward input must hold %d x %d floats", rows, d)
	}
	maskFlat, err := flattenAttentionResidentMask(req.Mask, batch, seqLen)
	if err != nil {
		return backend.AttentionResidentTrainForwardResult{}, err
	}
	if maskFlat == nil {
		maskFlat = allOnesInt32(batch * seqLen)
	}
	scale := req.Scale
	if scale == 0 {
		scale = 1
	}
	wq, ok := rt.residentMatrices[req.QueryName]
	if !ok {
		return backend.AttentionResidentTrainForwardResult{}, fmt.Errorf("cuda attention resident train forward: query binding %q is not resident", req.QueryName)
	}
	wk, ok := rt.residentMatrices[req.KeyName]
	if !ok {
		return backend.AttentionResidentTrainForwardResult{}, fmt.Errorf("cuda attention resident train forward: key binding %q is not resident", req.KeyName)
	}
	wv, ok := rt.residentMatrices[req.ValueName]
	if !ok {
		return backend.AttentionResidentTrainForwardResult{}, fmt.Errorf("cuda attention resident train forward: value binding %q is not resident", req.ValueName)
	}
	for _, w := range []struct {
		label string
		m     residentMatrix
	}{{"query", wq}, {"key", wk}, {"value", wv}} {
		if w.m.rows != d || w.m.cols != d {
			return backend.AttentionResidentTrainForwardResult{}, fmt.Errorf("cuda attention resident train forward: %s weight shape %dx%d does not match model_dim %d", w.label, w.m.rows, w.m.cols, d)
		}
	}
	kernel, err := rt.ensureSoftmaxForwardMaskedKernel()
	if err != nil {
		return backend.AttentionResidentTrainForwardResult{}, err
	}

	start := time.Now()
	// Every buffer here is a fresh per-handle allocation, not the shared
	// named-scratch cache runAttentionBlockResident uses: with multiple
	// encoder layers and multiple length-groups all forwarding before any
	// of them backward, several handles are live concurrently, so a fixed
	// scratch name would let a later forward call clobber an earlier one's
	// still-needed buffers.
	inputBuf, err := rt.allocFloat32(rows * d)
	if err != nil {
		return backend.AttentionResidentTrainForwardResult{}, err
	}
	if err := rt.copyFloat32ToBuffer(inputBuf, req.Input.F32); err != nil {
		_ = rt.freeBuffer(inputBuf)
		return backend.AttentionResidentTrainForwardResult{}, err
	}
	maskBuf, err := rt.uploadInt32(maskFlat)
	if err != nil {
		_ = rt.freeBuffer(inputBuf)
		return backend.AttentionResidentTrainForwardResult{}, err
	}
	defer rt.freeBuffer(maskBuf)

	qBuf, err := rt.allocFloat32(rows * d)
	if err != nil {
		_ = rt.freeBuffer(inputBuf)
		return backend.AttentionResidentTrainForwardResult{}, err
	}
	kBuf, err := rt.allocFloat32(rows * d)
	if err != nil {
		_ = rt.freeBuffer(inputBuf)
		_ = rt.freeBuffer(qBuf)
		return backend.AttentionResidentTrainForwardResult{}, err
	}
	vBuf, err := rt.allocFloat32(rows * d)
	if err != nil {
		_ = rt.freeBuffer(inputBuf)
		_ = rt.freeBuffer(qBuf)
		_ = rt.freeBuffer(kBuf)
		return backend.AttentionResidentTrainForwardResult{}, err
	}
	scoresBuf, err := rt.allocFloat32(batch * seqLen * seqLen)
	if err != nil {
		_ = rt.freeBuffer(inputBuf)
		_ = rt.freeBuffer(qBuf)
		_ = rt.freeBuffer(kBuf)
		_ = rt.freeBuffer(vBuf)
		return backend.AttentionResidentTrainForwardResult{}, err
	}
	// mixedBuf is forward-scoped only: the host still needs Mixed for its
	// existing (unchanged) Wo forward matmul, but no backward step below
	// ever reads the forward Mixed activation again -- backward's own
	// upstream gradient (gradMixed) arrives fresh from the caller instead.
	mixedBuf, err := rt.allocFloat32(rows * d)
	if err != nil {
		_ = rt.freeBuffer(inputBuf)
		_ = rt.freeBuffer(qBuf)
		_ = rt.freeBuffer(kBuf)
		_ = rt.freeBuffer(vBuf)
		_ = rt.freeBuffer(scoresBuf)
		return backend.AttentionResidentTrainForwardResult{}, err
	}
	defer rt.freeBuffer(mixedBuf)

	freeKept := func() {
		_ = rt.freeBuffer(inputBuf)
		_ = rt.freeBuffer(qBuf)
		_ = rt.freeBuffer(kBuf)
		_ = rt.freeBuffer(vBuf)
		_ = rt.freeBuffer(scoresBuf)
	}

	if err := rt.matMulCublasWithBetaNoSync(inputBuf, wq.ptr, qBuf, C.int(rows), C.int(d), C.int(d), C.int(d), false, false, 0); err != nil {
		freeKept()
		return backend.AttentionResidentTrainForwardResult{}, fmt.Errorf("cuda attention resident train forward: query projection: %w", err)
	}
	if err := rt.matMulCublasWithBetaNoSync(inputBuf, wk.ptr, kBuf, C.int(rows), C.int(d), C.int(d), C.int(d), false, false, 0); err != nil {
		freeKept()
		return backend.AttentionResidentTrainForwardResult{}, fmt.Errorf("cuda attention resident train forward: key projection: %w", err)
	}
	if err := rt.matMulCublasWithBetaNoSync(inputBuf, wv.ptr, vBuf, C.int(rows), C.int(d), C.int(d), C.int(d), false, false, 0); err != nil {
		freeKept()
		return backend.AttentionResidentTrainForwardResult{}, fmt.Errorf("cuda attention resident train forward: value projection: %w", err)
	}
	if err := rt.matMulCublasStridedBatched(qBuf, kBuf, scoresBuf, C.int(batch), C.int(seqLen), C.int(d), C.int(seqLen), C.int(d), false, true); err != nil {
		freeKept()
		return backend.AttentionResidentTrainForwardResult{}, fmt.Errorf("cuda attention resident train forward: scores: %w", err)
	}
	grid, block, err := checkedLaunch1D("cuda attention resident train softmax", rows, 128)
	if err != nil {
		freeKept()
		return backend.AttentionResidentTrainForwardResult{}, err
	}
	if err := rt.launchAuxSoftmaxForwardRowsMasked(kernel, uint(grid), uint(block), scoresBuf, maskBuf, rows, seqLen, seqLen, scale); err != nil {
		freeKept()
		return backend.AttentionResidentTrainForwardResult{}, fmt.Errorf("cuda attention resident train forward: masked softmax: %w", err)
	}
	if err := rt.matMulCublasStridedBatched(scoresBuf, vBuf, mixedBuf, C.int(batch), C.int(seqLen), C.int(seqLen), C.int(seqLen), C.int(d), false, false); err != nil {
		freeKept()
		return backend.AttentionResidentTrainForwardResult{}, fmt.Errorf("cuda attention resident train forward: mixed: %w", err)
	}
	if err := rt.synchronize(); err != nil {
		freeKept()
		return backend.AttentionResidentTrainForwardResult{}, err
	}

	queryHost := make([]float32, rows*d)
	keyHost := make([]float32, rows*d)
	valueHost := make([]float32, rows*d)
	scoresHost := make([]float32, batch*seqLen*seqLen)
	mixedHost := make([]float32, rows*d)
	for _, dl := range []struct {
		dst []float32
		src C.CUdeviceptr
	}{
		{queryHost, qBuf},
		{keyHost, kBuf},
		{valueHost, vBuf},
		{scoresHost, scoresBuf},
		{mixedHost, mixedBuf},
	} {
		if err := rt.downloadFloat32(dl.dst, dl.src); err != nil {
			freeKept()
			return backend.AttentionResidentTrainForwardResult{}, err
		}
	}

	if rt.attnTrain.handles == nil {
		rt.attnTrain.handles = map[uint64]*attentionResidentTrainHandleState{}
	}
	rt.attnTrain.nextHandle++
	handleID := rt.attnTrain.nextHandle
	state := &attentionResidentTrainHandleState{
		id:         handleID,
		generation: rt.attnTrain.generation,
		stepID:     rt.attnTrain.stepID,
		batch:      batch,
		seqLen:     seqLen,
		modelDim:   d,
		scale:      scale,
		queryName:  req.QueryName,
		keyName:    req.KeyName,
		valueName:  req.ValueName,
		inputBuf:   inputBuf,
		qBuf:       qBuf,
		kBuf:       kBuf,
		vBuf:       vBuf,
		scoresBuf:  scoresBuf,
	}
	rt.attnTrain.handles[handleID] = state

	uploadedBytes := int64((len(req.Input.F32) + len(maskFlat)) * 4)
	downloadedBytes := int64((len(queryHost) + len(keyHost) + len(valueHost) + len(scoresHost) + len(mixedHost)) * 4)
	rt.recordMatMulRun(start, uploadedBytes, downloadedBytes, false, true)

	return backend.AttentionResidentTrainForwardResult{
		Handle: backend.AttentionResidentTrainHandle{
			Backend:    eosartifact.BackendCUDA,
			Token:      &attentionResidentTrainToken{state: state},
			Batch:      batch,
			SeqLen:     seqLen,
			ModelDim:   d,
			Generation: state.generation,
			StepID:     state.stepID,
		},
		Query:  backend.NewTensorF32([]int{rows, d}, queryHost),
		Key:    backend.NewTensorF32([]int{rows, d}, keyHost),
		Value:  backend.NewTensorF32([]int{rows, d}, valueHost),
		Scores: backend.NewTensorF32([]int{rows, seqLen}, scoresHost),
		Mixed:  backend.NewTensorF32([]int{rows, d}, mixedHost),
	}, nil
}

// runAttentionBlockResidentTrainBackward backpropagates through the
// attention block runAttentionBlockResidentTrainForward computed: given the
// upstream gradient at Mixed (gradMixed -- the caller's host code has
// already backpropagated through Wo and any attention residual/layer norm
// to produce it, exactly as backpropAttentionSequences does today before it
// reaches the Q/K/V portion of the block), this chain mirrors the rest of
// backpropAttentionSequences entirely device-side:
//
//	gradScores    = gradMixed @ V^T                       (tryTrainerBatchedMatMulTranspose analog)
//	gradPreSoftmax = scale * softmaxBackward(gradScores, probs)  (trySoftmaxBackwardRows + scaleFloat32Slice, fused)
//	gradV  = probs^T @ gradMixed                           (tryCombinedAttentionValueKeyGradMatMul's V term)
//	gradK  = gradPreSoftmax^T @ Q                           (tryCombinedAttentionValueKeyGradMatMul's K term)
//	gradQ  = gradPreSoftmax @ K
//	gradWq = input^T @ gradQ ; gradWk = input^T @ gradK ; gradWv = input^T @ gradV
//	gradInput = gradQ@Wq^T + gradK@Wk^T + gradV@Wv^T        (tryAccumulatedAttentionInputGradMatMul analog)
//
// probs already carries zeros at masked key columns (from the forward
// masked-softmax kernel), so the softmax-backward step needs no mask input
// of its own: a zero probability zeroes that column's contribution to both
// the row dot product and the output row exactly as it does on the host
// (see softmaxRowsMaskedColumnsInPlace / backwardSoftmaxRow).
//
// gradInput is the Q/K/V-path contribution only -- the caller still adds
// any attention-residual pass-through itself, matching
// backpropAttentionSequences' own composition order. The handle is
// released (its device buffers freed) whether this call succeeds or fails,
// since each handle is consumed by exactly one backward call.
func (rt *deviceRuntime) runAttentionBlockResidentTrainBackward(req backend.AttentionResidentTrainBackwardRequest) (backend.AttentionResidentTrainBackwardResult, error) {
	if rt == nil || rt.ptr == nil {
		return backend.AttentionResidentTrainBackwardResult{}, fmt.Errorf("cuda runtime is not initialized")
	}
	state, err := rt.lookupAttentionResidentTrainHandle(req.Handle)
	if err != nil {
		return backend.AttentionResidentTrainBackwardResult{}, err
	}
	defer func() {
		rt.freeAttentionResidentTrainHandleBuffers(state)
		delete(rt.attnTrain.handles, state.id)
	}()

	batch, seqLen, d := state.batch, state.seqLen, state.modelDim
	rows := batch * seqLen
	if req.GradMixed == nil || len(req.GradMixed.F32) != rows*d {
		return backend.AttentionResidentTrainBackwardResult{}, fmt.Errorf("cuda attention resident train backward gradMixed must hold %d x %d floats", rows, d)
	}
	wq, ok := rt.residentMatrices[state.queryName]
	if !ok {
		return backend.AttentionResidentTrainBackwardResult{}, fmt.Errorf("cuda attention resident train backward: query binding %q is not resident", state.queryName)
	}
	wk, ok := rt.residentMatrices[state.keyName]
	if !ok {
		return backend.AttentionResidentTrainBackwardResult{}, fmt.Errorf("cuda attention resident train backward: key binding %q is not resident", state.keyName)
	}
	wv, ok := rt.residentMatrices[state.valueName]
	if !ok {
		return backend.AttentionResidentTrainBackwardResult{}, fmt.Errorf("cuda attention resident train backward: value binding %q is not resident", state.valueName)
	}
	softmaxKernel, err := rt.ensureAttnSoftmaxBackwardScaledKernel()
	if err != nil {
		return backend.AttentionResidentTrainBackwardResult{}, err
	}

	start := time.Now()
	gradMixedBuf, err := rt.allocFloat32(rows * d)
	if err != nil {
		return backend.AttentionResidentTrainBackwardResult{}, err
	}
	defer rt.freeBuffer(gradMixedBuf)
	if err := rt.copyFloat32ToBuffer(gradMixedBuf, req.GradMixed.F32); err != nil {
		return backend.AttentionResidentTrainBackwardResult{}, err
	}

	gradScoresBuf, err := rt.allocFloat32(batch * seqLen * seqLen)
	if err != nil {
		return backend.AttentionResidentTrainBackwardResult{}, err
	}
	defer rt.freeBuffer(gradScoresBuf)
	gradPreSoftmaxBuf, err := rt.allocFloat32(batch * seqLen * seqLen)
	if err != nil {
		return backend.AttentionResidentTrainBackwardResult{}, err
	}
	defer rt.freeBuffer(gradPreSoftmaxBuf)
	gradQBuf, err := rt.allocFloat32(rows * d)
	if err != nil {
		return backend.AttentionResidentTrainBackwardResult{}, err
	}
	defer rt.freeBuffer(gradQBuf)
	gradKBuf, err := rt.allocFloat32(rows * d)
	if err != nil {
		return backend.AttentionResidentTrainBackwardResult{}, err
	}
	defer rt.freeBuffer(gradKBuf)
	gradVBuf, err := rt.allocFloat32(rows * d)
	if err != nil {
		return backend.AttentionResidentTrainBackwardResult{}, err
	}
	defer rt.freeBuffer(gradVBuf)
	gradWqBuf, err := rt.allocFloat32(d * d)
	if err != nil {
		return backend.AttentionResidentTrainBackwardResult{}, err
	}
	defer rt.freeBuffer(gradWqBuf)
	gradWkBuf, err := rt.allocFloat32(d * d)
	if err != nil {
		return backend.AttentionResidentTrainBackwardResult{}, err
	}
	defer rt.freeBuffer(gradWkBuf)
	gradWvBuf, err := rt.allocFloat32(d * d)
	if err != nil {
		return backend.AttentionResidentTrainBackwardResult{}, err
	}
	defer rt.freeBuffer(gradWvBuf)
	gradInputBuf, err := rt.allocFloat32(rows * d)
	if err != nil {
		return backend.AttentionResidentTrainBackwardResult{}, err
	}
	defer rt.freeBuffer(gradInputBuf)

	// gradScores[b] = gradMixed[b] @ V[b]^T
	if err := rt.matMulCublasStridedBatched(gradMixedBuf, state.vBuf, gradScoresBuf, C.int(batch), C.int(seqLen), C.int(d), C.int(seqLen), C.int(d), false, true); err != nil {
		return backend.AttentionResidentTrainBackwardResult{}, fmt.Errorf("cuda attention resident train backward: gradScores: %w", err)
	}
	// gradPreSoftmax = scale * softmaxBackward(gradScores, probs)
	grid, block, err := checkedLaunch1D("cuda attention resident train softmax backward", rows, 128)
	if err != nil {
		return backend.AttentionResidentTrainBackwardResult{}, err
	}
	if err := rt.launchAuxAttnSoftmaxBackwardScaledRows(softmaxKernel, uint(grid), uint(block), gradScoresBuf, state.scoresBuf, gradPreSoftmaxBuf, rows, seqLen, state.scale); err != nil {
		return backend.AttentionResidentTrainBackwardResult{}, fmt.Errorf("cuda attention resident train backward: softmax backward: %w", err)
	}
	// gradV[b] = probs[b]^T @ gradMixed[b]
	if err := rt.matMulCublasStridedBatched(state.scoresBuf, gradMixedBuf, gradVBuf, C.int(batch), C.int(seqLen), C.int(seqLen), C.int(seqLen), C.int(d), true, false); err != nil {
		return backend.AttentionResidentTrainBackwardResult{}, fmt.Errorf("cuda attention resident train backward: gradV: %w", err)
	}
	// gradK[b] = gradPreSoftmax[b]^T @ Q[b]
	if err := rt.matMulCublasStridedBatched(gradPreSoftmaxBuf, state.qBuf, gradKBuf, C.int(batch), C.int(seqLen), C.int(seqLen), C.int(seqLen), C.int(d), true, false); err != nil {
		return backend.AttentionResidentTrainBackwardResult{}, fmt.Errorf("cuda attention resident train backward: gradK: %w", err)
	}
	// gradQ[b] = gradPreSoftmax[b] @ K[b]
	if err := rt.matMulCublasStridedBatched(gradPreSoftmaxBuf, state.kBuf, gradQBuf, C.int(batch), C.int(seqLen), C.int(seqLen), C.int(seqLen), C.int(d), false, false); err != nil {
		return backend.AttentionResidentTrainBackwardResult{}, fmt.Errorf("cuda attention resident train backward: gradQ: %w", err)
	}
	// gradWq = input^T @ gradQ ; gradWk = input^T @ gradK ; gradWv = input^T @ gradV,
	// each one GEMM contracting over every row (batch*seqLen) at once.
	if err := rt.matMulCublasWithBetaNoSync(state.inputBuf, gradQBuf, gradWqBuf, C.int(rows), C.int(d), C.int(rows), C.int(d), true, false, 0); err != nil {
		return backend.AttentionResidentTrainBackwardResult{}, fmt.Errorf("cuda attention resident train backward: gradWq: %w", err)
	}
	if err := rt.matMulCublasWithBetaNoSync(state.inputBuf, gradKBuf, gradWkBuf, C.int(rows), C.int(d), C.int(rows), C.int(d), true, false, 0); err != nil {
		return backend.AttentionResidentTrainBackwardResult{}, fmt.Errorf("cuda attention resident train backward: gradWk: %w", err)
	}
	if err := rt.matMulCublasWithBetaNoSync(state.inputBuf, gradVBuf, gradWvBuf, C.int(rows), C.int(d), C.int(rows), C.int(d), true, false, 0); err != nil {
		return backend.AttentionResidentTrainBackwardResult{}, fmt.Errorf("cuda attention resident train backward: gradWv: %w", err)
	}
	// gradInput = gradQ@Wq^T + gradK@Wk^T + gradV@Wv^T, accumulated in place
	// via beta (term1 writes, term2/term3 accumulate).
	if err := rt.matMulCublasWithBetaNoSync(gradQBuf, wq.ptr, gradInputBuf, C.int(rows), C.int(d), C.int(d), C.int(d), false, true, 0); err != nil {
		return backend.AttentionResidentTrainBackwardResult{}, fmt.Errorf("cuda attention resident train backward: gradInput query term: %w", err)
	}
	if err := rt.matMulCublasWithBetaNoSync(gradKBuf, wk.ptr, gradInputBuf, C.int(rows), C.int(d), C.int(d), C.int(d), false, true, 1); err != nil {
		return backend.AttentionResidentTrainBackwardResult{}, fmt.Errorf("cuda attention resident train backward: gradInput key term: %w", err)
	}
	if err := rt.matMulCublasWithBetaNoSync(gradVBuf, wv.ptr, gradInputBuf, C.int(rows), C.int(d), C.int(d), C.int(d), false, true, 1); err != nil {
		return backend.AttentionResidentTrainBackwardResult{}, fmt.Errorf("cuda attention resident train backward: gradInput value term: %w", err)
	}
	if err := rt.synchronize(); err != nil {
		return backend.AttentionResidentTrainBackwardResult{}, err
	}

	gradInputHost := make([]float32, rows*d)
	gradWqHost := make([]float32, d*d)
	gradWkHost := make([]float32, d*d)
	gradWvHost := make([]float32, d*d)
	for _, dl := range []struct {
		dst []float32
		src C.CUdeviceptr
	}{
		{gradInputHost, gradInputBuf},
		{gradWqHost, gradWqBuf},
		{gradWkHost, gradWkBuf},
		{gradWvHost, gradWvBuf},
	} {
		if err := rt.downloadFloat32(dl.dst, dl.src); err != nil {
			return backend.AttentionResidentTrainBackwardResult{}, err
		}
	}

	uploadedBytes := int64(len(req.GradMixed.F32) * 4)
	downloadedBytes := int64((len(gradInputHost) + len(gradWqHost) + len(gradWkHost) + len(gradWvHost)) * 4)
	rt.recordMatMulRun(start, uploadedBytes, downloadedBytes, true, false)

	return backend.AttentionResidentTrainBackwardResult{
		GradInput:       backend.NewTensorF32([]int{rows, d}, gradInputHost),
		GradQueryWeight: backend.NewTensorF32([]int{d, d}, gradWqHost),
		GradKeyWeight:   backend.NewTensorF32([]int{d, d}, gradWkHost),
		GradValueWeight: backend.NewTensorF32([]int{d, d}, gradWvHost),
	}, nil
}
