//go:build linux && cgo

package cuda

import (
	"math"
	"testing"

	"m31labs.dev/eos/runtime/backend"
)

// hostGeluBackwardForCUDATest is an independent host reference for the GELU
// backward-multiply kernel (manta_gelu_backward_mul /
// geluBackwardMulKernelSource): the closed-form derivative of the tanh-
// approximation GELU used by hostGeluForward (native_linux.go), matching
// the trainer package's own geluBackward.
func hostGeluBackwardForCUDATest(gradOut, preAct []float32) []float32 {
	out := make([]float32, len(preAct))
	for i, x := range preAct {
		x64 := float64(x)
		cubic := x64 * x64 * x64
		inner := 0.7978845608 * (x64 + 0.044715*cubic)
		tanhInner := math.Tanh(inner)
		sech2 := 1 - tanhInner*tanhInner
		innerGrad := 0.7978845608 * (1 + 3*0.044715*x64*x64)
		grad := 0.5*(1+tanhInner) + 0.5*x64*sech2*innerGrad
		out[i] = gradOut[i] * float32(grad)
	}
	return out
}

// hostFFNResidentTrainForward is an independent host reference for
// runFFNBlockResidentTrainForward, built from hostRowMajorMatMulForCUDATest
// and hostGeluForward rather than by mirroring the device code's own
// arithmetic.
func hostFFNResidentTrainForward(input, wup, wdown []float32, seqLen, d, h, e int) (ffnHidden, activated, ffnOutput []float32) {
	ffnHidden = hostRowMajorMatMulForCUDATest(input, wup, make([]float32, seqLen*h), seqLen, d, d, h, false, false, 0)
	activated = make([]float32, seqLen*h)
	for i, v := range ffnHidden {
		activated[i] = hostGeluForward(v)
	}
	ffnOutput = hostRowMajorMatMulForCUDATest(activated, wdown, make([]float32, seqLen*e), seqLen, h, h, e, false, false, 0)
	return ffnHidden, activated, ffnOutput
}

// hostFFNResidentTrainBackward is the backward counterpart of
// hostFFNResidentTrainForward, independently re-deriving
// backpropProjectedFFNSequence's hidden-projection/GELU/output-projection
// algebra via hostRowMajorMatMulForCUDATest and hostGeluBackwardForCUDATest.
func hostFFNResidentTrainBackward(input, wup, wdown, ffnHidden, activated, gradFFNOutput []float32, seqLen, d, h, e int) (gradInput, gradWup, gradWdown []float32) {
	gradWdown = hostRowMajorMatMulForCUDATest(activated, gradFFNOutput, make([]float32, h*e), seqLen, h, seqLen, e, true, false, 0)
	gradActivatedPre := hostRowMajorMatMulForCUDATest(gradFFNOutput, wdown, make([]float32, seqLen*h), seqLen, e, h, e, false, true, 0)
	gradActivated := hostGeluBackwardForCUDATest(gradActivatedPre, ffnHidden)
	gradWup = hostRowMajorMatMulForCUDATest(input, gradActivated, make([]float32, d*h), seqLen, d, seqLen, h, true, false, 0)
	gradInput = hostRowMajorMatMulForCUDATest(gradActivated, wup, make([]float32, seqLen*d), seqLen, h, d, h, false, true, 0)
	return gradInput, gradWup, gradWdown
}

func bindFFNResidentTrainWeights(t *testing.T, rt *deviceRuntime, prefix string, d, h, e int, wup, wdown []float32) (hiddenWeightName, outputWeightName string) {
	t.Helper()
	hiddenWeightName, outputWeightName = prefix+"_wup", prefix+"_wdown"
	if err := rt.bindMatMulRight(hiddenWeightName, backend.NewTensorF32([]int{d, h}, wup)); err != nil {
		t.Fatalf("bind wup: %v", err)
	}
	if err := rt.bindMatMulRight(outputWeightName, backend.NewTensorF32([]int{h, e}, wdown)); err != nil {
		t.Fatalf("bind wdown: %v", err)
	}
	t.Cleanup(func() {
		rt.unbindMatMulRight(hiddenWeightName)
		rt.unbindMatMulRight(outputWeightName)
	})
	return hiddenWeightName, outputWeightName
}

// TestFFNResidentTrainForwardBackwardMatchesHostReference is the S3(c) FFN
// parity test: forward (ffnHidden/activated/ffnOutput) and backward
// (gradHiddenWeight/gradOutputWeight/gradInput) must match the independent
// host reference within 1e-4 across a few non-square shapes (input dim,
// hidden dim, and output dim all different, exercising every matmul's
// transpose/shape wiring independently).
func TestFFNResidentTrainForwardBackwardMatchesHostReference(t *testing.T) {
	rt, err := newDeviceRuntime()
	if err != nil {
		t.Skipf("cuda runtime unavailable: %v", err)
	}
	defer rt.close()

	cases := []struct {
		name            string
		seqLen, d, h, e int
	}{
		{name: "seq2_d4_h8_e4", seqLen: 2, d: 4, h: 8, e: 4},
		{name: "seq3_d4_h6_e5", seqLen: 3, d: 4, h: 6, e: 5},
		{name: "seq5_d6_h3_e6", seqLen: 5, d: 6, h: 3, e: 6},
		{name: "seq1_d3_h5_e2", seqLen: 1, d: 3, h: 5, e: 2},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			input := seqData(tc.seqLen*tc.d, 0.023, -0.041)
			wup := seqData(tc.d*tc.h, 0.017, -0.013)
			wdown := seqData(tc.h*tc.e, -0.019, 0.011)
			gradFFNOutput := seqData(tc.seqLen*tc.e, -0.009, 0.031)

			hiddenWeightName, outputWeightName := bindFFNResidentTrainWeights(t, rt, "fft_"+tc.name, tc.d, tc.h, tc.e, wup, wdown)

			if err := rt.beginAttentionResidentTrainStep(1); err != nil {
				t.Fatalf("begin step: %v", err)
			}
			fwd, err := rt.runFFNBlockResidentTrainForward(backend.FFNResidentTrainForwardRequest{
				SeqLen:           tc.seqLen,
				InputDim:         tc.d,
				HiddenDim:        tc.h,
				OutputDim:        tc.e,
				Input:            backend.NewTensorF32([]int{tc.seqLen, tc.d}, input),
				HiddenWeightName: hiddenWeightName,
				OutputWeightName: outputWeightName,
				StepID:           1,
			})
			if err != nil {
				t.Fatalf("forward: %v", err)
			}
			wantFFNHidden, wantActivated, wantFFNOutput := hostFFNResidentTrainForward(input, wup, wdown, tc.seqLen, tc.d, tc.h, tc.e)
			assertFloatSlicesClose(t, fwd.FFNHidden.F32, wantFFNHidden, 1e-4)
			assertFloatSlicesClose(t, fwd.Activated.F32, wantActivated, 1e-4)
			assertFloatSlicesClose(t, fwd.FFNOutput.F32, wantFFNOutput, 1e-4)

			bwd, err := rt.runFFNBlockResidentTrainBackward(backend.FFNResidentTrainBackwardRequest{
				Handle:        fwd.Handle,
				GradFFNOutput: backend.NewTensorF32([]int{tc.seqLen, tc.e}, gradFFNOutput),
			})
			if err != nil {
				t.Fatalf("backward: %v", err)
			}
			wantGradInput, wantGradWup, wantGradWdown := hostFFNResidentTrainBackward(input, wup, wdown, wantFFNHidden, wantActivated, gradFFNOutput, tc.seqLen, tc.d, tc.h, tc.e)
			assertFloatSlicesClose(t, bwd.GradInput.F32, wantGradInput, 1e-4)
			gradWup, gradWdown, err := rt.flushFFNResidentTrainWeightGradients(hiddenWeightName, outputWeightName)
			if err != nil {
				t.Fatalf("flush gradients: %v", err)
			}
			if gradWup == nil || gradWdown == nil {
				t.Fatalf("flush gradients returned nil tensors: wup=%v wdown=%v", gradWup, gradWdown)
			}
			assertFloatSlicesClose(t, gradWup.F32, wantGradWup, 1e-4)
			assertFloatSlicesClose(t, gradWdown.F32, wantGradWdown, 1e-4)

			if err := rt.endOrAbortAttentionResidentTrainStep(1); err != nil {
				t.Fatalf("end step: %v", err)
			}
		})
	}
}

// TestFFNResidentTrainBackwardRejectsStaleHandleAcrossStepBoundary mirrors
// TestAttentionResidentTrainBackwardRejectsStaleHandleAcrossStepBoundary for
// the FFN block: a handle from step 1 must not be usable once step 2 has
// begun, even though step 1 never explicitly ended it.
func TestFFNResidentTrainBackwardRejectsStaleHandleAcrossStepBoundary(t *testing.T) {
	rt, err := newDeviceRuntime()
	if err != nil {
		t.Skipf("cuda runtime unavailable: %v", err)
	}
	defer rt.close()

	const seqLen, d, h, e = 2, 3, 4, 3
	input := seqData(seqLen*d, 0.031, 0.019)
	wup := seqData(d*h, 0.011, 0.005)
	wdown := seqData(h*e, -0.013, 0.021)
	hiddenWeightName, outputWeightName := bindFFNResidentTrainWeights(t, rt, "fft_stale", d, h, e, wup, wdown)

	if err := rt.beginAttentionResidentTrainStep(1); err != nil {
		t.Fatalf("begin step 1: %v", err)
	}
	fwd, err := rt.runFFNBlockResidentTrainForward(backend.FFNResidentTrainForwardRequest{
		SeqLen: seqLen, InputDim: d, HiddenDim: h, OutputDim: e,
		Input:            backend.NewTensorF32([]int{seqLen, d}, input),
		HiddenWeightName: hiddenWeightName, OutputWeightName: outputWeightName,
		StepID: 1,
	})
	if err != nil {
		t.Fatalf("forward step 1: %v", err)
	}
	// A second batch's step begins WITHOUT step 1 ever calling
	// End/AbortAttentionResidentTrainStep -- the exact scenario the shared
	// generation bump in beginAttentionResidentTrainStep exists to guard,
	// for FFN handles exactly as for attention handles.
	if err := rt.beginAttentionResidentTrainStep(2); err != nil {
		t.Fatalf("begin step 2: %v", err)
	}

	gradFFNOutput := seqData(seqLen*e, -0.01, 0.02)
	_, err = rt.runFFNBlockResidentTrainBackward(backend.FFNResidentTrainBackwardRequest{
		Handle:        fwd.Handle,
		GradFFNOutput: backend.NewTensorF32([]int{seqLen, e}, gradFFNOutput),
	})
	if err == nil {
		t.Fatal("expected an error backpropagating a handle from a step that has since been superseded by a new Begin")
	}

	if err := rt.endOrAbortAttentionResidentTrainStep(2); err != nil {
		t.Fatalf("end step 2: %v", err)
	}
}

// TestFFNResidentTrainBackwardRejectsHandleAfterRelease mirrors
// TestAttentionResidentTrainBackwardRejectsHandleAfterRelease for the FFN
// block: a handle already consumed by one backward call cannot be replayed
// into a second backward call.
func TestFFNResidentTrainBackwardRejectsHandleAfterRelease(t *testing.T) {
	rt, err := newDeviceRuntime()
	if err != nil {
		t.Skipf("cuda runtime unavailable: %v", err)
	}
	defer rt.close()

	const seqLen, d, h, e = 2, 3, 4, 3
	input := seqData(seqLen*d, 0.031, 0.019)
	wup := seqData(d*h, 0.011, 0.005)
	wdown := seqData(h*e, -0.013, 0.021)
	hiddenWeightName, outputWeightName := bindFFNResidentTrainWeights(t, rt, "fft_release", d, h, e, wup, wdown)

	if err := rt.beginAttentionResidentTrainStep(1); err != nil {
		t.Fatalf("begin step: %v", err)
	}
	fwd, err := rt.runFFNBlockResidentTrainForward(backend.FFNResidentTrainForwardRequest{
		SeqLen: seqLen, InputDim: d, HiddenDim: h, OutputDim: e,
		Input:            backend.NewTensorF32([]int{seqLen, d}, input),
		HiddenWeightName: hiddenWeightName, OutputWeightName: outputWeightName,
		StepID: 1,
	})
	if err != nil {
		t.Fatalf("forward: %v", err)
	}
	gradFFNOutput := seqData(seqLen*e, -0.01, 0.02)
	if _, err := rt.runFFNBlockResidentTrainBackward(backend.FFNResidentTrainBackwardRequest{
		Handle:        fwd.Handle,
		GradFFNOutput: backend.NewTensorF32([]int{seqLen, e}, gradFFNOutput),
	}); err != nil {
		t.Fatalf("first backward: %v", err)
	}

	if _, err := rt.runFFNBlockResidentTrainBackward(backend.FFNResidentTrainBackwardRequest{
		Handle:        fwd.Handle,
		GradFFNOutput: backend.NewTensorF32([]int{seqLen, e}, gradFFNOutput),
	}); err == nil {
		t.Fatal("expected an error replaying a handle backward already consumed")
	}

	if err := rt.endOrAbortAttentionResidentTrainStep(1); err != nil {
		t.Fatalf("end step: %v", err)
	}
}

// TestFFNResidentTrainForwardRejectsFastGELU confirms the resident kernel
// fails closed (rather than silently computing the wrong GELU variant) when
// asked for the fast/rational GELU approximation the host path can select
// via fastGELUEnabled() -- only the exact (tanh) kernel is wired.
func TestFFNResidentTrainForwardRejectsFastGELU(t *testing.T) {
	rt, err := newDeviceRuntime()
	if err != nil {
		t.Skipf("cuda runtime unavailable: %v", err)
	}
	defer rt.close()

	const seqLen, d, h, e = 2, 3, 4, 3
	input := seqData(seqLen*d, 0.031, 0.019)
	wup := seqData(d*h, 0.011, 0.005)
	wdown := seqData(h*e, -0.013, 0.021)
	hiddenWeightName, outputWeightName := bindFFNResidentTrainWeights(t, rt, "fft_fastgelu", d, h, e, wup, wdown)

	if err := rt.beginAttentionResidentTrainStep(1); err != nil {
		t.Fatalf("begin step: %v", err)
	}
	defer rt.endOrAbortAttentionResidentTrainStep(1)

	_, err = rt.runFFNBlockResidentTrainForward(backend.FFNResidentTrainForwardRequest{
		SeqLen: seqLen, InputDim: d, HiddenDim: h, OutputDim: e,
		Input:            backend.NewTensorF32([]int{seqLen, d}, input),
		HiddenWeightName: hiddenWeightName, OutputWeightName: outputWeightName,
		FastGELU: true,
		StepID:   1,
	})
	if err == nil {
		t.Fatal("expected an error requesting fast GELU from the resident kernel")
	}
}
