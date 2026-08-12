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

	"m31labs.dev/eos/runtime/backend"
)

// runAttentionBlockResident executes one attention block -- Q/K/V projection,
// per-sequence scaled dot-product scores, forward softmax, and value-mix --
// entirely device-to-device against already-bound resident weights (see
// bindMatMulRight/residentMatrices). It uploads the block's input once and
// downloads each of Q, K, V, the post-softmax scores, and the mixed output
// once; no intermediate activation crosses the host/device boundary more
// than once.
//
// This is the stage (a) forward-only residency boundary: the caller
// (EmbeddingTrainer) still runs backward on the host using these downloaded
// host copies, and score scaling is baked into the caller-supplied
// QueryName binding (see primeAttentionResidentQueryBinding in the runtime
// package) so this method never launches a separate device-side scale
// kernel: scale*(Q@K^T) == (scale*Q)@K^T == (input @ (scale*Wq)) @ K^T.
//
// Masked softmax is out of scope for stage (a); callers must only route
// AttentionMaskMode "none" sequences here.
func (rt *deviceRuntime) runAttentionBlockResident(req backend.AttentionResidentRequest) (backend.AttentionResidentResult, error) {
	if rt == nil || rt.ptr == nil {
		return backend.AttentionResidentResult{}, fmt.Errorf("cuda runtime is not initialized")
	}
	batch, seqLen, d := req.Batch, req.SeqLen, req.ModelDim
	if batch <= 0 || seqLen <= 0 || d <= 0 {
		return backend.AttentionResidentResult{}, fmt.Errorf("cuda attention resident block requires positive batch, seq_len, and model_dim")
	}
	rows := batch * seqLen
	if req.Input == nil || len(req.Input.F32) != rows*d {
		return backend.AttentionResidentResult{}, fmt.Errorf("cuda attention resident block input must hold %d x %d floats", rows, d)
	}
	wq, ok := rt.residentMatrices[req.QueryName]
	if !ok {
		return backend.AttentionResidentResult{}, fmt.Errorf("cuda attention resident block: query binding %q is not resident", req.QueryName)
	}
	wk, ok := rt.residentMatrices[req.KeyName]
	if !ok {
		return backend.AttentionResidentResult{}, fmt.Errorf("cuda attention resident block: key binding %q is not resident", req.KeyName)
	}
	wv, ok := rt.residentMatrices[req.ValueName]
	if !ok {
		return backend.AttentionResidentResult{}, fmt.Errorf("cuda attention resident block: value binding %q is not resident", req.ValueName)
	}
	for _, w := range []struct {
		label string
		m     residentMatrix
	}{{"query", wq}, {"key", wk}, {"value", wv}} {
		if w.m.rows != d || w.m.cols != d {
			return backend.AttentionResidentResult{}, fmt.Errorf("cuda attention resident block: %s weight shape %dx%d does not match model_dim %d", w.label, w.m.rows, w.m.cols, d)
		}
	}
	kernel, err := rt.ensureSoftmaxForwardKernel()
	if err != nil {
		return backend.AttentionResidentResult{}, err
	}

	start := time.Now()
	inputBuf, err := rt.uploadMatMulScratchFloat32("attn_resident_input", req.Input.F32)
	if err != nil {
		return backend.AttentionResidentResult{}, err
	}
	qBuf, err := rt.matMulScratchFloat32("attn_resident_q", rows*d)
	if err != nil {
		return backend.AttentionResidentResult{}, err
	}
	kBuf, err := rt.matMulScratchFloat32("attn_resident_k", rows*d)
	if err != nil {
		return backend.AttentionResidentResult{}, err
	}
	vBuf, err := rt.matMulScratchFloat32("attn_resident_v", rows*d)
	if err != nil {
		return backend.AttentionResidentResult{}, err
	}
	scoresBuf, err := rt.matMulScratchFloat32("attn_resident_scores", batch*seqLen*seqLen)
	if err != nil {
		return backend.AttentionResidentResult{}, err
	}
	mixedBuf, err := rt.matMulScratchFloat32("attn_resident_mixed", rows*d)
	if err != nil {
		return backend.AttentionResidentResult{}, err
	}

	// Q = input @ Wq ; K = input @ Wk ; V = input @ Wv, all device-to-device
	// against the already-resident (and, for Q, score-scale-baked) weights.
	if err := rt.matMulCublasWithBetaNoSync(inputBuf, wq.ptr, qBuf, C.int(rows), C.int(d), C.int(d), C.int(d), false, false, 0); err != nil {
		return backend.AttentionResidentResult{}, fmt.Errorf("cuda attention resident block: query projection: %w", err)
	}
	if err := rt.matMulCublasWithBetaNoSync(inputBuf, wk.ptr, kBuf, C.int(rows), C.int(d), C.int(d), C.int(d), false, false, 0); err != nil {
		return backend.AttentionResidentResult{}, fmt.Errorf("cuda attention resident block: key projection: %w", err)
	}
	if err := rt.matMulCublasWithBetaNoSync(inputBuf, wv.ptr, vBuf, C.int(rows), C.int(d), C.int(d), C.int(d), false, false, 0); err != nil {
		return backend.AttentionResidentResult{}, fmt.Errorf("cuda attention resident block: value projection: %w", err)
	}
	// scores[b] = Q[b] @ K[b]^T, one seqLen x seqLen block per sequence,
	// contiguous within scoresBuf (batch-strided GEMM keeps sequences from
	// attending across the batch boundary).
	if err := rt.matMulCublasStridedBatched(qBuf, kBuf, scoresBuf, C.int(batch), C.int(seqLen), C.int(d), C.int(seqLen), C.int(d), false, true); err != nil {
		return backend.AttentionResidentResult{}, fmt.Errorf("cuda attention resident block: scores: %w", err)
	}
	grid, block, err := checkedLaunch1D("cuda attention resident softmax", rows, 128)
	if err != nil {
		return backend.AttentionResidentResult{}, err
	}
	if err := rt.launchAuxSoftmaxForwardRows(kernel, uint(grid), uint(block), scoresBuf, rows, seqLen); err != nil {
		return backend.AttentionResidentResult{}, fmt.Errorf("cuda attention resident block: softmax: %w", err)
	}
	// mixed[b] = scores[b] @ V[b]
	if err := rt.matMulCublasStridedBatched(scoresBuf, vBuf, mixedBuf, C.int(batch), C.int(seqLen), C.int(seqLen), C.int(seqLen), C.int(d), false, false); err != nil {
		return backend.AttentionResidentResult{}, fmt.Errorf("cuda attention resident block: mixed: %w", err)
	}
	if err := rt.synchronize(); err != nil {
		return backend.AttentionResidentResult{}, err
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
			return backend.AttentionResidentResult{}, err
		}
	}

	downloadedBytes := int64((len(queryHost) + len(keyHost) + len(valueHost) + len(scoresHost) + len(mixedHost)) * 4)
	rt.recordMatMulRun(start, int64(len(req.Input.F32)*4), downloadedBytes, false, true)

	return backend.AttentionResidentResult{
		Query:  backend.NewTensorF32([]int{rows, d}, queryHost),
		Key:    backend.NewTensorF32([]int{rows, d}, keyHost),
		Value:  backend.NewTensorF32([]int{rows, d}, valueHost),
		Scores: backend.NewTensorF32([]int{rows, seqLen}, scoresHost),
		Mixed:  backend.NewTensorF32([]int{rows, d}, mixedHost),
	}, nil
}
