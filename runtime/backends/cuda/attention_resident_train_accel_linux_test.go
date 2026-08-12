//go:build linux && cgo

package cuda

import (
	"testing"

	"m31labs.dev/eos/runtime/backend"
)

// hostAttentionResidentTrainForward is an independent host reference for
// runAttentionBlockResidentTrainForward, built from the generic
// (transpose/beta-aware) hostRowMajorMatMulForCUDATest primitive and the
// masked-softmax reference softmaxMaskedHost already used by the compact
// forward CUDA tests, rather than by mirroring the device code's own
// arithmetic -- so a shape/transpose slip in the device chain has an
// independent reference to disagree with.
func hostAttentionResidentTrainForward(input, wq, wk, wv []float32, mask [][]int32, batch, seqLen, d int, scale float32) (q, k, v, probs, mixed []float32) {
	rows := batch * seqLen
	q = hostRowMajorMatMulForCUDATest(input, wq, make([]float32, rows*d), rows, d, d, d, false, false, 0)
	k = hostRowMajorMatMulForCUDATest(input, wk, make([]float32, rows*d), rows, d, d, d, false, false, 0)
	v = hostRowMajorMatMulForCUDATest(input, wv, make([]float32, rows*d), rows, d, d, d, false, false, 0)
	probs = make([]float32, batch*seqLen*seqLen)
	mixed = make([]float32, rows*d)
	for b := 0; b < batch; b++ {
		base := b * seqLen * d
		scoreBase := b * seqLen * seqLen
		qb := q[base : base+seqLen*d]
		kb := k[base : base+seqLen*d]
		vb := v[base : base+seqLen*d]
		sb := hostRowMajorMatMulForCUDATest(qb, kb, make([]float32, seqLen*seqLen), seqLen, d, seqLen, d, false, true, 0)
		for i := range sb {
			sb[i] *= scale
		}
		for row := 0; row < seqLen; row++ {
			softmaxMaskedHost(sb[row*seqLen:(row+1)*seqLen], mask[b])
		}
		copy(probs[scoreBase:scoreBase+seqLen*seqLen], sb)
		mb := hostRowMajorMatMulForCUDATest(sb, vb, make([]float32, seqLen*d), seqLen, seqLen, seqLen, d, false, false, 0)
		copy(mixed[base:base+seqLen*d], mb)
	}
	return q, k, v, probs, mixed
}

func softmaxBackwardRowHostForCUDATest(dOut, probs []float32) []float32 {
	dot := float32(0)
	for i := range probs {
		dot += dOut[i] * probs[i]
	}
	out := make([]float32, len(probs))
	for i := range probs {
		out[i] = probs[i] * (dOut[i] - dot)
	}
	return out
}

// hostAttentionResidentTrainBackward is the backward counterpart of
// hostAttentionResidentTrainForward, independently re-deriving
// backpropAttentionSequences' Q/K/V-path algebra via
// hostRowMajorMatMulForCUDATest rather than mirroring the device kernels.
func hostAttentionResidentTrainBackward(input, wq, wk, wv, q, k, v, probs, gradMixed []float32, batch, seqLen, d int, scale float32) (gradInput, gradWq, gradWk, gradWv []float32) {
	rows := batch * seqLen
	gradQ := make([]float32, rows*d)
	gradK := make([]float32, rows*d)
	gradV := make([]float32, rows*d)
	for b := 0; b < batch; b++ {
		base := b * seqLen * d
		scoreBase := b * seqLen * seqLen
		qb := q[base : base+seqLen*d]
		kb := k[base : base+seqLen*d]
		pb := probs[scoreBase : scoreBase+seqLen*seqLen]
		gmb := gradMixed[base : base+seqLen*d]
		vb := v[base : base+seqLen*d]

		gradScores := hostRowMajorMatMulForCUDATest(gmb, vb, make([]float32, seqLen*seqLen), seqLen, d, seqLen, d, false, true, 0)
		gradPreSoftmax := make([]float32, seqLen*seqLen)
		for row := 0; row < seqLen; row++ {
			rowOut := softmaxBackwardRowHostForCUDATest(gradScores[row*seqLen:(row+1)*seqLen], pb[row*seqLen:(row+1)*seqLen])
			for c, val := range rowOut {
				gradPreSoftmax[row*seqLen+c] = val * scale
			}
		}
		gvb := hostRowMajorMatMulForCUDATest(pb, gmb, make([]float32, seqLen*d), seqLen, seqLen, seqLen, d, true, false, 0)
		gkb := hostRowMajorMatMulForCUDATest(gradPreSoftmax, qb, make([]float32, seqLen*d), seqLen, seqLen, seqLen, d, true, false, 0)
		gqb := hostRowMajorMatMulForCUDATest(gradPreSoftmax, kb, make([]float32, seqLen*d), seqLen, seqLen, seqLen, d, false, false, 0)
		copy(gradQ[base:base+seqLen*d], gqb)
		copy(gradK[base:base+seqLen*d], gkb)
		copy(gradV[base:base+seqLen*d], gvb)
	}
	gradWq = hostRowMajorMatMulForCUDATest(input, gradQ, make([]float32, d*d), rows, d, rows, d, true, false, 0)
	gradWk = hostRowMajorMatMulForCUDATest(input, gradK, make([]float32, d*d), rows, d, rows, d, true, false, 0)
	gradWv = hostRowMajorMatMulForCUDATest(input, gradV, make([]float32, d*d), rows, d, rows, d, true, false, 0)
	gradInput = hostRowMajorMatMulForCUDATest(gradQ, wq, make([]float32, rows*d), rows, d, d, d, false, true, 0)
	term2 := hostRowMajorMatMulForCUDATest(gradK, wk, make([]float32, rows*d), rows, d, d, d, false, true, 0)
	term3 := hostRowMajorMatMulForCUDATest(gradV, wv, make([]float32, rows*d), rows, d, d, d, false, true, 0)
	for i := range gradInput {
		gradInput[i] += term2[i] + term3[i]
	}
	return gradInput, gradWq, gradWk, gradWv
}

func bindAttentionResidentTrainWeights(t *testing.T, rt *deviceRuntime, prefix string, d int, wq, wk, wv []float32) (queryName, keyName, valueName string) {
	t.Helper()
	queryName, keyName, valueName = prefix+"_wq", prefix+"_wk", prefix+"_wv"
	if err := rt.bindMatMulRight(queryName, backend.NewTensorF32([]int{d, d}, wq)); err != nil {
		t.Fatalf("bind wq: %v", err)
	}
	if err := rt.bindMatMulRight(keyName, backend.NewTensorF32([]int{d, d}, wk)); err != nil {
		t.Fatalf("bind wk: %v", err)
	}
	if err := rt.bindMatMulRight(valueName, backend.NewTensorF32([]int{d, d}, wv)); err != nil {
		t.Fatalf("bind wv: %v", err)
	}
	t.Cleanup(func() {
		rt.unbindMatMulRight(queryName)
		rt.unbindMatMulRight(keyName)
		rt.unbindMatMulRight(valueName)
	})
	return queryName, keyName, valueName
}

// raggedMaskSet builds one all-ones mask per sequence except the last
// active `keep` positions in the final sequence, matching how a
// length-bucketed batch pads shorter sequences within a shared seqLen.
func raggedMaskSet(batch, seqLen int, keepInLast int) [][]int32 {
	masks := make([][]int32, batch)
	for b := 0; b < batch; b++ {
		row := make([]int32, seqLen)
		for i := range row {
			row[i] = 1
		}
		if b == batch-1 {
			for i := keepInLast; i < seqLen; i++ {
				row[i] = 0
			}
		}
		masks[b] = row
	}
	return masks
}

// TestAttentionResidentTrainForwardBackwardMatchesHostReference is the S3(b)
// parity test: forward (masked, scaled) and backward (gradWq/gradWk/gradWv/
// gradInput) must match the independent host reference within 1e-4 across a
// few shapes, including a ragged mask (a shorter sequence padded out to a
// shared seqLen within the batch).
func TestAttentionResidentTrainForwardBackwardMatchesHostReference(t *testing.T) {
	rt, err := newDeviceRuntime()
	if err != nil {
		t.Skipf("cuda runtime unavailable: %v", err)
	}
	defer rt.close()

	cases := []struct {
		name   string
		batch  int
		seqLen int
		d      int
		ragged int // keepInLast; 0 means fully unmasked (every mask row all-ones)
		scale  float32
	}{
		{name: "batch1_seq2_d4_unmasked", batch: 1, seqLen: 2, d: 4, ragged: 0, scale: 0.5},
		{name: "batch2_seq3_d4_unmasked", batch: 2, seqLen: 3, d: 4, ragged: 0, scale: 1.0 / 2.0},
		{name: "batch3_seq4_d6_ragged", batch: 3, seqLen: 4, d: 6, ragged: 2, scale: 0.25},
		{name: "batch4_seq5_d4_ragged", batch: 4, seqLen: 5, d: 4, ragged: 3, scale: 0.4},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			rows := tc.batch * tc.seqLen
			input := seqData(rows*tc.d, 0.023, -0.041)
			wq := seqData(tc.d*tc.d, 0.017, -0.013)
			wk := seqData(tc.d*tc.d, -0.019, 0.011)
			wv := seqData(tc.d*tc.d, 0.014, 0.007)
			gradMixed := seqData(rows*tc.d, -0.009, 0.031)

			var mask [][]int32
			if tc.ragged > 0 {
				mask = raggedMaskSet(tc.batch, tc.seqLen, tc.ragged)
			} else {
				mask = raggedMaskSet(tc.batch, tc.seqLen, tc.seqLen)
			}

			queryName, keyName, valueName := bindAttentionResidentTrainWeights(t, rt, "art_"+tc.name, tc.d, wq, wk, wv)

			if err := rt.beginAttentionResidentTrainStep(1); err != nil {
				t.Fatalf("begin step: %v", err)
			}
			fwd, err := rt.runAttentionBlockResidentTrainForward(backend.AttentionResidentTrainForwardRequest{
				Batch:     tc.batch,
				SeqLen:    tc.seqLen,
				ModelDim:  tc.d,
				Input:     backend.NewTensorF32([]int{rows, tc.d}, input),
				QueryName: queryName,
				KeyName:   keyName,
				ValueName: valueName,
				Mask:      mask,
				Scale:     tc.scale,
				StepID:    1,
			})
			if err != nil {
				t.Fatalf("forward: %v", err)
			}
			wantQ, wantK, wantV, wantProbs, wantMixed := hostAttentionResidentTrainForward(input, wq, wk, wv, mask, tc.batch, tc.seqLen, tc.d, tc.scale)
			assertFloatSlicesClose(t, fwd.Query.F32, wantQ, 1e-4)
			assertFloatSlicesClose(t, fwd.Key.F32, wantK, 1e-4)
			assertFloatSlicesClose(t, fwd.Value.F32, wantV, 1e-4)
			assertFloatSlicesClose(t, fwd.Scores.F32, wantProbs, 1e-4)
			assertFloatSlicesClose(t, fwd.Mixed.F32, wantMixed, 1e-4)

			bwd, err := rt.runAttentionBlockResidentTrainBackward(backend.AttentionResidentTrainBackwardRequest{
				Handle:    fwd.Handle,
				GradMixed: backend.NewTensorF32([]int{rows, tc.d}, gradMixed),
			})
			if err != nil {
				t.Fatalf("backward: %v", err)
			}
			wantGradInput, wantGradWq, wantGradWk, wantGradWv := hostAttentionResidentTrainBackward(input, wq, wk, wv, wantQ, wantK, wantV, wantProbs, gradMixed, tc.batch, tc.seqLen, tc.d, tc.scale)
			assertFloatSlicesClose(t, bwd.GradInput.F32, wantGradInput, 1e-4)
			assertFloatSlicesClose(t, bwd.GradQueryWeight.F32, wantGradWq, 1e-4)
			assertFloatSlicesClose(t, bwd.GradKeyWeight.F32, wantGradWk, 1e-4)
			assertFloatSlicesClose(t, bwd.GradValueWeight.F32, wantGradWv, 1e-4)

			if err := rt.endOrAbortAttentionResidentTrainStep(1); err != nil {
				t.Fatalf("end step: %v", err)
			}
		})
	}
}

// TestAttentionResidentTrainBackwardRejectsStaleHandleAcrossStepBoundary is
// the step-boundary invalidation guard: a handle from step 1 must not be
// usable once step 2 has begun, even though step 1 never explicitly ended
// it -- the class of bug that would let a second batch silently read a
// first batch's still-resident activations.
func TestAttentionResidentTrainBackwardRejectsStaleHandleAcrossStepBoundary(t *testing.T) {
	rt, err := newDeviceRuntime()
	if err != nil {
		t.Skipf("cuda runtime unavailable: %v", err)
	}
	defer rt.close()

	const d = 3
	const seqLen = 2
	input := seqData(seqLen*d, 0.031, 0.019)
	wq := seqData(d*d, 0.011, 0.005)
	wk := seqData(d*d, -0.013, 0.021)
	wv := seqData(d*d, 0.017, -0.009)
	queryName, keyName, valueName := bindAttentionResidentTrainWeights(t, rt, "art_stale", d, wq, wk, wv)
	mask := raggedMaskSet(1, seqLen, seqLen)

	if err := rt.beginAttentionResidentTrainStep(1); err != nil {
		t.Fatalf("begin step 1: %v", err)
	}
	fwd, err := rt.runAttentionBlockResidentTrainForward(backend.AttentionResidentTrainForwardRequest{
		Batch: 1, SeqLen: seqLen, ModelDim: d,
		Input:     backend.NewTensorF32([]int{seqLen, d}, input),
		QueryName: queryName, KeyName: keyName, ValueName: valueName,
		Mask: mask, Scale: 1, StepID: 1,
	})
	if err != nil {
		t.Fatalf("forward step 1: %v", err)
	}
	// A second batch's step begins WITHOUT step 1 ever calling
	// End/AbortAttentionResidentTrainStep -- the exact scenario the
	// generation bump in beginAttentionResidentTrainStep exists to guard.
	if err := rt.beginAttentionResidentTrainStep(2); err != nil {
		t.Fatalf("begin step 2: %v", err)
	}

	gradMixed := seqData(seqLen*d, -0.01, 0.02)
	_, err = rt.runAttentionBlockResidentTrainBackward(backend.AttentionResidentTrainBackwardRequest{
		Handle:    fwd.Handle,
		GradMixed: backend.NewTensorF32([]int{seqLen, d}, gradMixed),
	})
	if err == nil {
		t.Fatal("expected an error backpropagating a handle from a step that has since been superseded by a new Begin")
	}

	if err := rt.endOrAbortAttentionResidentTrainStep(2); err != nil {
		t.Fatalf("end step 2: %v", err)
	}
}

// TestAttentionResidentTrainBackwardRejectsHandleAfterRelease confirms a
// handle already consumed by one backward call (or explicitly released)
// cannot be replayed into a second backward call.
func TestAttentionResidentTrainBackwardRejectsHandleAfterRelease(t *testing.T) {
	rt, err := newDeviceRuntime()
	if err != nil {
		t.Skipf("cuda runtime unavailable: %v", err)
	}
	defer rt.close()

	const d = 3
	const seqLen = 2
	input := seqData(seqLen*d, 0.031, 0.019)
	wq := seqData(d*d, 0.011, 0.005)
	wk := seqData(d*d, -0.013, 0.021)
	wv := seqData(d*d, 0.017, -0.009)
	queryName, keyName, valueName := bindAttentionResidentTrainWeights(t, rt, "art_release", d, wq, wk, wv)
	mask := raggedMaskSet(1, seqLen, seqLen)

	if err := rt.beginAttentionResidentTrainStep(1); err != nil {
		t.Fatalf("begin step: %v", err)
	}
	fwd, err := rt.runAttentionBlockResidentTrainForward(backend.AttentionResidentTrainForwardRequest{
		Batch: 1, SeqLen: seqLen, ModelDim: d,
		Input:     backend.NewTensorF32([]int{seqLen, d}, input),
		QueryName: queryName, KeyName: keyName, ValueName: valueName,
		Mask: mask, Scale: 1, StepID: 1,
	})
	if err != nil {
		t.Fatalf("forward: %v", err)
	}
	gradMixed := seqData(seqLen*d, -0.01, 0.02)
	if _, err := rt.runAttentionBlockResidentTrainBackward(backend.AttentionResidentTrainBackwardRequest{
		Handle:    fwd.Handle,
		GradMixed: backend.NewTensorF32([]int{seqLen, d}, gradMixed),
	}); err != nil {
		t.Fatalf("first backward: %v", err)
	}

	if _, err := rt.runAttentionBlockResidentTrainBackward(backend.AttentionResidentTrainBackwardRequest{
		Handle:    fwd.Handle,
		GradMixed: backend.NewTensorF32([]int{seqLen, d}, gradMixed),
	}); err == nil {
		t.Fatal("expected an error replaying a handle backward already consumed")
	}

	if err := rt.endOrAbortAttentionResidentTrainStep(1); err != nil {
		t.Fatalf("end step: %v", err)
	}
}

// TestAttentionResidentTrainForwardAfterOptimizerStepUsesFreshWeights is the
// optimizer-step-then-forward guard: after Wq is re-bound under the same
// name (as an optimizer step does), a fresh Begin/forward on a new step
// must reflect the new weight, not a value cached from the handle registry
// or scratch buffers of the prior step.
func TestAttentionResidentTrainForwardAfterOptimizerStepUsesFreshWeights(t *testing.T) {
	rt, err := newDeviceRuntime()
	if err != nil {
		t.Skipf("cuda runtime unavailable: %v", err)
	}
	defer rt.close()

	const d = 3
	const seqLen = 2
	input := seqData(seqLen*d, 0.031, 0.019)
	wqA := seqData(d*d, 0.011, 0.005)
	wk := seqData(d*d, -0.013, 0.021)
	wv := seqData(d*d, 0.017, -0.009)
	queryName, keyName, valueName := bindAttentionResidentTrainWeights(t, rt, "art_optstep", d, wqA, wk, wv)
	mask := raggedMaskSet(1, seqLen, seqLen)

	req := backend.AttentionResidentTrainForwardRequest{
		Batch: 1, SeqLen: seqLen, ModelDim: d,
		Input:     backend.NewTensorF32([]int{seqLen, d}, input),
		QueryName: queryName, KeyName: keyName, ValueName: valueName,
		Mask: mask, Scale: 1,
	}

	if err := rt.beginAttentionResidentTrainStep(1); err != nil {
		t.Fatalf("begin step 1: %v", err)
	}
	req.StepID = 1
	fwdA, err := rt.runAttentionBlockResidentTrainForward(req)
	if err != nil {
		t.Fatalf("forward step 1: %v", err)
	}
	gradMixed := seqData(seqLen*d, -0.01, 0.02)
	if _, err := rt.runAttentionBlockResidentTrainBackward(backend.AttentionResidentTrainBackwardRequest{
		Handle:    fwdA.Handle,
		GradMixed: backend.NewTensorF32([]int{seqLen, d}, gradMixed),
	}); err != nil {
		t.Fatalf("backward step 1: %v", err)
	}
	if err := rt.endOrAbortAttentionResidentTrainStep(1); err != nil {
		t.Fatalf("end step 1: %v", err)
	}

	// Simulate an optimizer step re-binding Wq under the same name with new data.
	wqB := seqData(d*d, 0.037, -0.027)
	if err := rt.bindMatMulRight(queryName, backend.NewTensorF32([]int{d, d}, wqB)); err != nil {
		t.Fatalf("rebind wqB: %v", err)
	}

	if err := rt.beginAttentionResidentTrainStep(2); err != nil {
		t.Fatalf("begin step 2: %v", err)
	}
	req.StepID = 2
	fwdB, err := rt.runAttentionBlockResidentTrainForward(req)
	if err != nil {
		t.Fatalf("forward step 2: %v", err)
	}
	wantQB := hostRowMajorMatMulForCUDATest(input, wqB, make([]float32, seqLen*d), seqLen, d, d, d, false, false, 0)
	assertFloatSlicesClose(t, fwdB.Query.F32, wantQB, 1e-4)
	same := true
	for i := range fwdA.Query.F32 {
		if abs32ForCUDATest(fwdA.Query.F32[i]-fwdB.Query.F32[i]) > 1e-6 {
			same = false
			break
		}
	}
	if same {
		t.Fatal("query projection after optimizer re-bind matches the pre-update result; the resident-train forward may be reusing stale weights")
	}
	if err := rt.endOrAbortAttentionResidentTrainStep(2); err != nil {
		t.Fatalf("end step 2: %v", err)
	}
}
