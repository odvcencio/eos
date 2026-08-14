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

// ffnResidentTrainState is the S3(c) FFN-resident-train handle registry,
// embedded in deviceRuntime. Unlike attentionResidentTrainState it carries
// no generation/stepID/stepActive fields of its own: FFN blocks and
// attention blocks share exactly one step per training step (see
// beginAttentionResidentTrainStep/endOrAbortAttentionResidentTrainStep in
// attention_resident_train_accel_linux.go, which now also sweep this
// registry), so every FFN handle below is stamped with rt.attnTrain's
// generation/stepID at creation time and validated against it at lookup
// time, exactly as attention handles are against themselves.
type ffnResidentTrainState struct {
	nextHandle uint64
	handles    map[uint64]*ffnResidentTrainHandleState
	// gradAccum is S3(d)'s step-scoped device-side hidden/output
	// weight-gradient accumulator, keyed by resident weight binding name.
	// Mirrors attentionResidentTrainState.gradAccum exactly (see
	// attention_resident_train_accel_linux.go for the shared
	// residentGradAccumBuffer/accumulateResidentGradWeight/
	// flushResidentGradAccum/releaseResidentGradAccum helpers); cleared by
	// the same begin/end/abort sweep that clears handles.
	gradAccum map[string]*residentGradAccumBuffer
}

// ffnResidentTrainHandleState is the device-private data one
// RunFFNBlockResidentTrainForward call registers and the matching
// RunFFNBlockResidentTrainBackward call consumes. The flattened input,
// pre-GELU ffnHidden, and post-GELU activated stay resident here across the
// forward/backward boundary; ffnOutput is not kept (backward only needs the
// caller-supplied upstream gradFFNOutput, never the forward ffnOutput
// activation itself -- mirrors attentionResidentTrainHandleState not
// keeping mixed).
type ffnResidentTrainHandleState struct {
	id               uint64
	generation       uint64
	stepID           uint64
	seqLen           int
	inputDim         int
	hiddenDim        int
	outputDim        int
	hiddenWeightName string
	outputWeightName string
	inputBuf         C.CUdeviceptr
	ffnHiddenBuf     C.CUdeviceptr
	activatedBuf     C.CUdeviceptr
	released         bool
}

// ffnResidentTrainToken implements backend.FFNResidentTrainHandleToken.
type ffnResidentTrainToken struct {
	state *ffnResidentTrainHandleState
}

func (t *ffnResidentTrainToken) FFNResidentTrainHandleToken() {}

func (t *ffnResidentTrainToken) Backend() eosartifact.BackendKind {
	return eosartifact.BackendCUDA
}

func (t *ffnResidentTrainToken) Generation() uint64 {
	if t == nil || t.state == nil {
		return 0
	}
	return t.state.generation
}

func (t *ffnResidentTrainToken) StepID() uint64 {
	if t == nil || t.state == nil {
		return 0
	}
	return t.state.stepID
}

func (t *ffnResidentTrainToken) Alive() bool {
	return t != nil && t.state != nil && !t.state.released
}

// releaseAllFFNResidentTrainHandles frees every live FFN handle's device
// buffers and clears the registry. Safe to call with zero live handles.
// Called from both deviceRuntime.close() and the shared attention/FFN step
// boundary sweep (beginAttentionResidentTrainStep,
// endOrAbortAttentionResidentTrainStep).
func (rt *deviceRuntime) releaseAllFFNResidentTrainHandles() {
	if rt == nil {
		return
	}
	for id, state := range rt.ffnTrain.handles {
		rt.freeFFNResidentTrainHandleBuffers(state)
		delete(rt.ffnTrain.handles, id)
	}
}

func (rt *deviceRuntime) freeFFNResidentTrainHandleBuffers(state *ffnResidentTrainHandleState) {
	if rt == nil || state == nil || state.released {
		return
	}
	state.released = true
	for _, ptr := range []C.CUdeviceptr{state.inputBuf, state.ffnHiddenBuf, state.activatedBuf} {
		_ = rt.freeBuffer(ptr)
	}
}

// releaseFFNResidentTrainHandle releases one handle's device buffers
// (idempotent, mirrors releaseAttentionResidentTrainHandle).
func (rt *deviceRuntime) releaseFFNResidentTrainHandle(handle backend.FFNResidentTrainHandle) error {
	if rt == nil || rt.ptr == nil {
		return fmt.Errorf("cuda runtime is not initialized")
	}
	token, ok := handle.Token.(*ffnResidentTrainToken)
	if !ok || token == nil || token.state == nil {
		return nil
	}
	state := token.state
	rt.freeFFNResidentTrainHandleBuffers(state)
	delete(rt.ffnTrain.handles, state.id)
	return nil
}

// lookupFFNResidentTrainHandle validates a handle against the shared
// attention/FFN step generation and registry membership before a backward
// call touches its buffers, mirroring lookupAttentionResidentTrainHandle.
func (rt *deviceRuntime) lookupFFNResidentTrainHandle(handle backend.FFNResidentTrainHandle) (*ffnResidentTrainHandleState, error) {
	token, ok := handle.Token.(*ffnResidentTrainToken)
	if !ok || token == nil || token.state == nil {
		return nil, fmt.Errorf("cuda ffn resident train: handle is not a live cuda resident-train handle")
	}
	state := token.state
	if state.released {
		return nil, fmt.Errorf("cuda ffn resident train: handle %d was already released", state.id)
	}
	if state.generation != rt.attnTrain.generation {
		return nil, fmt.Errorf("cuda ffn resident train: handle %d is from a prior step generation (stale)", state.id)
	}
	if registered, ok := rt.ffnTrain.handles[state.id]; !ok || registered != state {
		return nil, fmt.Errorf("cuda ffn resident train: handle %d is not registered", state.id)
	}
	return state, nil
}

// runFFNBlockResidentTrainForward runs one FFN block's hidden projection,
// GELU, and output projection, keeping the flattened input, pre-GELU
// ffnHidden, and post-GELU activated device-resident afterward (under the
// returned handle) instead of only using them to compute the downloaded
// result. This is the S3(c) counterpart of
// runAttentionBlockResidentTrainForward: same handle lifecycle, same
// single-upload/multi-download shape, same "caller still adds residual/
// layer norm on the host" boundary as the attention chain's Wo/residual/LN.
func (rt *deviceRuntime) runFFNBlockResidentTrainForward(req backend.FFNResidentTrainForwardRequest) (backend.FFNResidentTrainForwardResult, error) {
	if rt == nil || rt.ptr == nil {
		return backend.FFNResidentTrainForwardResult{}, fmt.Errorf("cuda runtime is not initialized")
	}
	if !rt.attnTrain.stepActive || rt.attnTrain.stepID != req.StepID {
		return backend.FFNResidentTrainForwardResult{}, fmt.Errorf("cuda ffn resident train forward: step %d is not the active step", req.StepID)
	}
	seqLen, d, h, e := req.SeqLen, req.InputDim, req.HiddenDim, req.OutputDim
	if seqLen <= 0 || d <= 0 || h <= 0 || e <= 0 {
		return backend.FFNResidentTrainForwardResult{}, fmt.Errorf("cuda ffn resident train forward requires positive seq_len, input_dim, hidden_dim, and output_dim")
	}
	if req.FastGELU {
		// Only the exact (tanh) GELU kernel is wired (see
		// ensureGeluForwardKernel/ensureGeluBackwardKernel): the fast/rational
		// approximation host callers may select via fastGELUEnabled() is not
		// implemented here, so a caller must fail closed to the host path
		// rather than silently compute the wrong GELU variant.
		return backend.FFNResidentTrainForwardResult{}, fmt.Errorf("cuda ffn resident train forward: fast GELU is not supported by the resident kernel")
	}
	if req.Input == nil || len(req.Input.F32) != seqLen*d {
		return backend.FFNResidentTrainForwardResult{}, fmt.Errorf("cuda ffn resident train forward input must hold %d x %d floats", seqLen, d)
	}
	wup, ok := rt.residentMatrices[req.HiddenWeightName]
	if !ok {
		return backend.FFNResidentTrainForwardResult{}, fmt.Errorf("cuda ffn resident train forward: hidden weight binding %q is not resident", req.HiddenWeightName)
	}
	if wup.rows != d || wup.cols != h {
		return backend.FFNResidentTrainForwardResult{}, fmt.Errorf("cuda ffn resident train forward: hidden weight shape %dx%d does not match input_dim=%d hidden_dim=%d", wup.rows, wup.cols, d, h)
	}
	wdown, ok := rt.residentMatrices[req.OutputWeightName]
	if !ok {
		return backend.FFNResidentTrainForwardResult{}, fmt.Errorf("cuda ffn resident train forward: output weight binding %q is not resident", req.OutputWeightName)
	}
	if wdown.rows != h || wdown.cols != e {
		return backend.FFNResidentTrainForwardResult{}, fmt.Errorf("cuda ffn resident train forward: output weight shape %dx%d does not match hidden_dim=%d output_dim=%d", wdown.rows, wdown.cols, h, e)
	}
	geluKernel, err := rt.ensureGeluForwardKernel()
	if err != nil {
		return backend.FFNResidentTrainForwardResult{}, err
	}

	start := time.Now()
	// Every buffer here is a fresh per-handle allocation, not shared named
	// scratch: with multiple encoder layers and multiple length-groups all
	// forwarding before any of them backward, several handles are live
	// concurrently (see runAttentionBlockResidentTrainForward's identical
	// reasoning).
	inputBuf, err := rt.allocFloat32(seqLen * d)
	if err != nil {
		return backend.FFNResidentTrainForwardResult{}, err
	}
	if err := rt.copyFloat32ToBuffer(inputBuf, req.Input.F32); err != nil {
		_ = rt.freeBuffer(inputBuf)
		return backend.FFNResidentTrainForwardResult{}, err
	}
	ffnHiddenBuf, err := rt.allocFloat32(seqLen * h)
	if err != nil {
		_ = rt.freeBuffer(inputBuf)
		return backend.FFNResidentTrainForwardResult{}, err
	}
	activatedBuf, err := rt.allocFloat32(seqLen * h)
	if err != nil {
		_ = rt.freeBuffer(inputBuf)
		_ = rt.freeBuffer(ffnHiddenBuf)
		return backend.FFNResidentTrainForwardResult{}, err
	}
	// ffnOutputBuf is forward-scoped only: the host still needs FFNOutput to
	// continue with its existing (unchanged) FFN-residual/layer-norm path,
	// but no backward step below ever reads the forward FFNOutput activation
	// again -- backward's own upstream gradient (gradFFNOutput) arrives
	// fresh from the caller instead.
	ffnOutputBuf, err := rt.allocFloat32(seqLen * e)
	if err != nil {
		_ = rt.freeBuffer(inputBuf)
		_ = rt.freeBuffer(ffnHiddenBuf)
		_ = rt.freeBuffer(activatedBuf)
		return backend.FFNResidentTrainForwardResult{}, err
	}
	defer rt.freeBuffer(ffnOutputBuf)

	freeKept := func() {
		_ = rt.freeBuffer(inputBuf)
		_ = rt.freeBuffer(ffnHiddenBuf)
		_ = rt.freeBuffer(activatedBuf)
	}

	// ffnHidden = input @ Wup
	if err := rt.matMulCublasWithBetaNoSync(inputBuf, wup.ptr, ffnHiddenBuf, C.int(seqLen), C.int(d), C.int(d), C.int(h), false, false, 0); err != nil {
		freeKept()
		return backend.FFNResidentTrainForwardResult{}, fmt.Errorf("cuda ffn resident train forward: hidden projection: %w", err)
	}
	// activated = GELU(ffnHidden)
	grid, block, err := checkedLaunch1D("cuda ffn resident train gelu forward", seqLen*h, 128)
	if err != nil {
		freeKept()
		return backend.FFNResidentTrainForwardResult{}, err
	}
	if err := rt.launchAuxGeluForward(geluKernel, uint(grid), uint(block), ffnHiddenBuf, activatedBuf, seqLen*h); err != nil {
		freeKept()
		return backend.FFNResidentTrainForwardResult{}, fmt.Errorf("cuda ffn resident train forward: gelu: %w", err)
	}
	// ffnOutput = activated @ Wdown
	if err := rt.matMulCublasWithBetaNoSync(activatedBuf, wdown.ptr, ffnOutputBuf, C.int(seqLen), C.int(h), C.int(h), C.int(e), false, false, 0); err != nil {
		freeKept()
		return backend.FFNResidentTrainForwardResult{}, fmt.Errorf("cuda ffn resident train forward: output projection: %w", err)
	}
	if err := rt.synchronize(); err != nil {
		freeKept()
		return backend.FFNResidentTrainForwardResult{}, err
	}

	// S3(d): SkipUnreadDownload elides the ffnHidden/activated D2H copy when
	// the caller has proven host code will never read them before the
	// matching backward call -- see the field's doc comment in backend.go.
	// FFNOutput is never skipped: the caller's forward continuation
	// (residual + layer norm) always reads it. The underlying device
	// buffers are computed and kept resident either way; only their host
	// copy is conditional.
	skipHiddenActivated := req.SkipUnreadDownload
	var ffnHiddenHost, activatedHost []float32
	if !skipHiddenActivated {
		ffnHiddenHost = make([]float32, seqLen*h)
		activatedHost = make([]float32, seqLen*h)
	}
	ffnOutputHost := make([]float32, seqLen*e)
	downloads := []struct {
		dst []float32
		src C.CUdeviceptr
	}{
		{ffnOutputHost, ffnOutputBuf},
	}
	if !skipHiddenActivated {
		downloads = append(downloads,
			struct {
				dst []float32
				src C.CUdeviceptr
			}{ffnHiddenHost, ffnHiddenBuf},
			struct {
				dst []float32
				src C.CUdeviceptr
			}{activatedHost, activatedBuf},
		)
	}
	for _, dl := range downloads {
		if err := rt.downloadFloat32(dl.dst, dl.src); err != nil {
			freeKept()
			return backend.FFNResidentTrainForwardResult{}, err
		}
	}

	if rt.ffnTrain.handles == nil {
		rt.ffnTrain.handles = map[uint64]*ffnResidentTrainHandleState{}
	}
	rt.ffnTrain.nextHandle++
	handleID := rt.ffnTrain.nextHandle
	state := &ffnResidentTrainHandleState{
		id:               handleID,
		generation:       rt.attnTrain.generation,
		stepID:           rt.attnTrain.stepID,
		seqLen:           seqLen,
		inputDim:         d,
		hiddenDim:        h,
		outputDim:        e,
		hiddenWeightName: req.HiddenWeightName,
		outputWeightName: req.OutputWeightName,
		inputBuf:         inputBuf,
		ffnHiddenBuf:     ffnHiddenBuf,
		activatedBuf:     activatedBuf,
	}
	rt.ffnTrain.handles[handleID] = state

	uploadedBytes := int64(len(req.Input.F32) * 4)
	downloadedBytes := int64((len(ffnHiddenHost) + len(activatedHost) + len(ffnOutputHost)) * 4)
	rt.recordMatMulRun(start, uploadedBytes, downloadedBytes, false, true)

	result := backend.FFNResidentTrainForwardResult{
		Handle: backend.FFNResidentTrainHandle{
			Backend:    eosartifact.BackendCUDA,
			Token:      &ffnResidentTrainToken{state: state},
			SeqLen:     seqLen,
			InputDim:   d,
			HiddenDim:  h,
			OutputDim:  e,
			Generation: state.generation,
			StepID:     state.stepID,
		},
		FFNOutput: backend.NewTensorF32([]int{seqLen, e}, ffnOutputHost),
	}
	if !skipHiddenActivated {
		result.FFNHidden = backend.NewTensorF32([]int{seqLen, h}, ffnHiddenHost)
		result.Activated = backend.NewTensorF32([]int{seqLen, h}, activatedHost)
	}
	return result, nil
}

// runFFNBlockResidentTrainBackward backpropagates through the FFN block
// runFFNBlockResidentTrainForward computed: given the upstream gradient at
// FFNOutput (gradFFNOutput -- the caller's host code has already
// backpropagated through the FFN residual and any layer norm to produce it,
// exactly as backpropProjectedFFNSequence does today before it reaches the
// output-projection portion of the block), this chain mirrors the rest of
// backpropProjectedFFNSequence entirely device-side:
//
//	gradOutputWeight = activated^T @ gradFFNOutput          (Wdown grad)
//	gradActivatedPre = gradFFNOutput @ Wdown^T
//	gradActivated    = geluBackward(gradActivatedPre, ffnHidden)
//	gradHiddenWeight = input^T @ gradActivated               (Wup grad)
//	gradHidden       = gradActivated @ Wup^T
//
// gradHidden is the hidden-projection/GELU/output-projection contribution
// only -- the caller still adds any FFN-residual pass-through itself,
// matching backpropProjectedFFNSequence's own composition order. The handle
// is released (its device buffers freed) whether this call succeeds or
// fails, since each handle is consumed by exactly one backward call.
func (rt *deviceRuntime) runFFNBlockResidentTrainBackward(req backend.FFNResidentTrainBackwardRequest) (backend.FFNResidentTrainBackwardResult, error) {
	if rt == nil || rt.ptr == nil {
		return backend.FFNResidentTrainBackwardResult{}, fmt.Errorf("cuda runtime is not initialized")
	}
	state, err := rt.lookupFFNResidentTrainHandle(req.Handle)
	if err != nil {
		return backend.FFNResidentTrainBackwardResult{}, err
	}
	defer func() {
		rt.freeFFNResidentTrainHandleBuffers(state)
		delete(rt.ffnTrain.handles, state.id)
	}()

	seqLen, d, h, e := state.seqLen, state.inputDim, state.hiddenDim, state.outputDim
	if req.GradFFNOutput == nil || len(req.GradFFNOutput.F32) != seqLen*e {
		return backend.FFNResidentTrainBackwardResult{}, fmt.Errorf("cuda ffn resident train backward gradFFNOutput must hold %d x %d floats", seqLen, e)
	}
	wup, ok := rt.residentMatrices[state.hiddenWeightName]
	if !ok {
		return backend.FFNResidentTrainBackwardResult{}, fmt.Errorf("cuda ffn resident train backward: hidden weight binding %q is not resident", state.hiddenWeightName)
	}
	wdown, ok := rt.residentMatrices[state.outputWeightName]
	if !ok {
		return backend.FFNResidentTrainBackwardResult{}, fmt.Errorf("cuda ffn resident train backward: output weight binding %q is not resident", state.outputWeightName)
	}
	geluBackwardKernel, err := rt.ensureGeluBackwardKernel()
	if err != nil {
		return backend.FFNResidentTrainBackwardResult{}, err
	}

	start := time.Now()
	gradFFNOutputBuf, err := rt.allocFloat32(seqLen * e)
	if err != nil {
		return backend.FFNResidentTrainBackwardResult{}, err
	}
	defer rt.freeBuffer(gradFFNOutputBuf)
	if err := rt.copyFloat32ToBuffer(gradFFNOutputBuf, req.GradFFNOutput.F32); err != nil {
		return backend.FFNResidentTrainBackwardResult{}, err
	}

	gradActivatedPreBuf, err := rt.allocFloat32(seqLen * h)
	if err != nil {
		return backend.FFNResidentTrainBackwardResult{}, err
	}
	defer rt.freeBuffer(gradActivatedPreBuf)
	gradActivatedBuf, err := rt.allocFloat32(seqLen * h)
	if err != nil {
		return backend.FFNResidentTrainBackwardResult{}, err
	}
	defer rt.freeBuffer(gradActivatedBuf)
	gradInputBuf, err := rt.allocFloat32(seqLen * d)
	if err != nil {
		return backend.FFNResidentTrainBackwardResult{}, err
	}
	defer rt.freeBuffer(gradInputBuf)

	// S3(d): gradOutputWeight += activated^T @ gradFFNOutput, accumulated
	// with beta=1 into this step's device-resident per-name accumulator
	// instead of downloaded here -- see flushFFNResidentTrainWeightGradients.
	if err := rt.accumulateResidentGradWeight(&rt.ffnTrain.gradAccum, state.outputWeightName, h, e, rt.attnTrain.stepID, state.activatedBuf, gradFFNOutputBuf, C.int(seqLen), C.int(h), C.int(seqLen), C.int(e), true, false); err != nil {
		return backend.FFNResidentTrainBackwardResult{}, fmt.Errorf("cuda ffn resident train backward: gradOutputWeight accumulate: %w", err)
	}
	// gradActivatedPre = gradFFNOutput @ Wdown^T
	if err := rt.matMulCublasWithBetaNoSync(gradFFNOutputBuf, wdown.ptr, gradActivatedPreBuf, C.int(seqLen), C.int(e), C.int(h), C.int(e), false, true, 0); err != nil {
		return backend.FFNResidentTrainBackwardResult{}, fmt.Errorf("cuda ffn resident train backward: gradActivatedPre: %w", err)
	}
	// gradActivated = geluBackward(gradActivatedPre, ffnHidden)
	grid, block, err := checkedLaunch1D("cuda ffn resident train gelu backward", seqLen*h, 128)
	if err != nil {
		return backend.FFNResidentTrainBackwardResult{}, err
	}
	if err := rt.launchAuxElementWise(geluBackwardKernel, uint(grid), uint(block), gradActivatedPreBuf, state.ffnHiddenBuf, gradActivatedBuf, seqLen*h); err != nil {
		return backend.FFNResidentTrainBackwardResult{}, fmt.Errorf("cuda ffn resident train backward: gelu backward: %w", err)
	}
	// S3(d): gradHiddenWeight += input^T @ gradActivated, accumulated the
	// same way as gradOutputWeight above.
	if err := rt.accumulateResidentGradWeight(&rt.ffnTrain.gradAccum, state.hiddenWeightName, d, h, rt.attnTrain.stepID, state.inputBuf, gradActivatedBuf, C.int(seqLen), C.int(d), C.int(seqLen), C.int(h), true, false); err != nil {
		return backend.FFNResidentTrainBackwardResult{}, fmt.Errorf("cuda ffn resident train backward: gradHiddenWeight accumulate: %w", err)
	}
	// gradInput = gradActivated @ Wup^T
	if err := rt.matMulCublasWithBetaNoSync(gradActivatedBuf, wup.ptr, gradInputBuf, C.int(seqLen), C.int(h), C.int(d), C.int(h), false, true, 0); err != nil {
		return backend.FFNResidentTrainBackwardResult{}, fmt.Errorf("cuda ffn resident train backward: gradInput: %w", err)
	}
	if err := rt.synchronize(); err != nil {
		return backend.FFNResidentTrainBackwardResult{}, err
	}

	gradInputHost := make([]float32, seqLen*d)
	if err := rt.downloadFloat32(gradInputHost, gradInputBuf); err != nil {
		return backend.FFNResidentTrainBackwardResult{}, err
	}

	uploadedBytes := int64(len(req.GradFFNOutput.F32) * 4)
	downloadedBytes := int64(len(gradInputHost) * 4)
	rt.recordMatMulRun(start, uploadedBytes, downloadedBytes, true, false)

	return backend.FFNResidentTrainBackwardResult{
		GradInput: backend.NewTensorF32([]int{seqLen, d}, gradInputHost),
	}, nil
}

// flushFFNResidentTrainWeightGradients implements
// backend.FFNResidentTrainAccelerator.FlushFFNResidentTrainWeightGradients:
// downloads and frees this step's device-accumulated hidden/output weight
// gradient sums exactly once, mirroring
// flushAttentionResidentTrainWeightGradients. A name with no contributions
// this step yields a nil tensor for that slot, not an error.
func (rt *deviceRuntime) flushFFNResidentTrainWeightGradients(hiddenWeightName, outputWeightName string) (gradHiddenWeight, gradOutputWeight *backend.Tensor, err error) {
	if rt == nil || rt.ptr == nil {
		return nil, nil, fmt.Errorf("cuda runtime is not initialized")
	}
	if gradHiddenWeight, err = flushResidentGradAccum(rt, rt.ffnTrain.gradAccum, hiddenWeightName, rt.attnTrain.stepID); err != nil {
		return nil, nil, err
	}
	if gradOutputWeight, err = flushResidentGradAccum(rt, rt.ffnTrain.gradAccum, outputWeightName, rt.attnTrain.stepID); err != nil {
		return nil, nil, err
	}
	return gradHiddenWeight, gradOutputWeight, nil
}

// Note: the exported backend.FFNResidentTrainAccelerator methods
// (RunFFNBlockResidentTrainForward, RunFFNBlockResidentTrainBackward,
// FlushFFNResidentTrainWeightGradients, ReleaseFFNResidentTrainHandle) live
// on *matMulAccelerator in matmul_accel.go, not on *deviceRuntime directly
// -- t.forwardMatMul is always a *matMulAccelerator wrapping a
// *deviceRuntime via a named field (not an embedded one), so deviceRuntime's
// methods are not promoted. Mirrors exactly how
// attention_resident_train_accel_linux.go's lowercase
// runAttentionBlockResidentTrainForward/Backward are wrapped.
