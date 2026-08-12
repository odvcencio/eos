//go:build linux && cgo

package cuda

import (
	"testing"

	"m31labs.dev/eos/runtime/backend"
)

// TestAttentionResidentBlockMatchesHostReference verifies
// deviceRuntime.runAttentionBlockResident's Q/K/V projection, per-sequence
// scaled-dot-product scores, forward softmax, and value-mix against a host
// reference across a few (batch, seq_len, model_dim) shapes. It skips
// without a CUDA runtime, matching the other CUDA accelerator tests in this
// package.
func TestAttentionResidentBlockMatchesHostReference(t *testing.T) {
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
	}{
		{name: "batch1_seq2_d4", batch: 1, seqLen: 2, d: 4},
		{name: "batch2_seq3_d4", batch: 2, seqLen: 3, d: 4},
		{name: "batch3_seq4_d6", batch: 3, seqLen: 4, d: 6},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			rows := tc.batch * tc.seqLen
			input := seqData(rows*tc.d, 0.023, -0.041)
			wq := seqData(tc.d*tc.d, 0.017, -0.013)
			wk := seqData(tc.d*tc.d, -0.019, 0.011)
			wv := seqData(tc.d*tc.d, 0.014, 0.007)

			const queryName = "test_attn_resident_wq"
			const keyName = "test_attn_resident_wk"
			const valueName = "test_attn_resident_wv"
			if err := rt.bindMatMulRight(queryName, backend.NewTensorF32([]int{tc.d, tc.d}, wq)); err != nil {
				t.Fatalf("bind wq: %v", err)
			}
			defer rt.unbindMatMulRight(queryName)
			if err := rt.bindMatMulRight(keyName, backend.NewTensorF32([]int{tc.d, tc.d}, wk)); err != nil {
				t.Fatalf("bind wk: %v", err)
			}
			defer rt.unbindMatMulRight(keyName)
			if err := rt.bindMatMulRight(valueName, backend.NewTensorF32([]int{tc.d, tc.d}, wv)); err != nil {
				t.Fatalf("bind wv: %v", err)
			}
			defer rt.unbindMatMulRight(valueName)

			result, err := rt.runAttentionBlockResident(backend.AttentionResidentRequest{
				Batch:     tc.batch,
				SeqLen:    tc.seqLen,
				ModelDim:  tc.d,
				Input:     backend.NewTensorF32([]int{rows, tc.d}, input),
				QueryName: queryName,
				KeyName:   keyName,
				ValueName: valueName,
			})
			if err != nil {
				t.Fatalf("run attention resident block: %v", err)
			}

			wantQuery := matmulHost(input, rows, tc.d, wq, tc.d)
			wantKey := matmulHost(input, rows, tc.d, wk, tc.d)
			wantValue := matmulHost(input, rows, tc.d, wv, tc.d)
			wantScores := make([]float32, tc.batch*tc.seqLen*tc.seqLen)
			wantMixed := make([]float32, rows*tc.d)
			for b := 0; b < tc.batch; b++ {
				base := b * tc.seqLen * tc.d
				scoreBase := b * tc.seqLen * tc.seqLen
				qb := wantQuery[base : base+tc.seqLen*tc.d]
				kb := wantKey[base : base+tc.seqLen*tc.d]
				vb := wantValue[base : base+tc.seqLen*tc.d]
				sb := hostRowMajorMatMulForCUDATest(qb, kb, make([]float32, tc.seqLen*tc.seqLen), tc.seqLen, tc.d, tc.seqLen, tc.d, false, true, 0)
				hostSoftmaxRows(sb, tc.seqLen, tc.seqLen)
				copy(wantScores[scoreBase:scoreBase+tc.seqLen*tc.seqLen], sb)
				mb := matmulHost(sb, tc.seqLen, tc.seqLen, vb, tc.d)
				copy(wantMixed[base:base+tc.seqLen*tc.d], mb)
			}

			if result.Query == nil || result.Key == nil || result.Value == nil || result.Scores == nil || result.Mixed == nil {
				t.Fatal("expected every result tensor to be populated")
			}
			assertFloatSlicesClose(t, result.Query.F32, wantQuery, 1e-4)
			assertFloatSlicesClose(t, result.Key.F32, wantKey, 1e-4)
			assertFloatSlicesClose(t, result.Value.F32, wantValue, 1e-4)
			assertFloatSlicesClose(t, result.Scores.F32, wantScores, 1e-4)
			assertFloatSlicesClose(t, result.Mixed.F32, wantMixed, 1e-4)
		})
	}
}

// TestAttentionResidentBlockRejectsMissingBinding confirms the resident
// block fails closed (with an error, not a silent zero result) when the
// caller has not bound the query/key/value weights it names.
func TestAttentionResidentBlockRejectsMissingBinding(t *testing.T) {
	rt, err := newDeviceRuntime()
	if err != nil {
		t.Skipf("cuda runtime unavailable: %v", err)
	}
	defer rt.close()

	_, err = rt.runAttentionBlockResident(backend.AttentionResidentRequest{
		Batch:     1,
		SeqLen:    2,
		ModelDim:  2,
		Input:     backend.NewTensorF32([]int{2, 2}, []float32{1, 2, 3, 4}),
		QueryName: "missing_q",
		KeyName:   "missing_k",
		ValueName: "missing_v",
	})
	if err == nil {
		t.Fatal("expected an error for unbound query/key/value weights")
	}
}

// TestAttentionResidentBlockRepeatedCallsRefreshWeights confirms two calls
// against the same names but different bound weight data (as
// bindMatMulRight/BindMatrix produces after an optimizer step re-binds a
// changed weight) each use their own current weights rather than a
// stale/cached first-call result -- the device-level half of the S3(a)
// stale-weight guard tested at the trainer level in
// TestEmbeddingTrainerAttnResidentReflectsWeightsAfterOptimizerStep.
func TestAttentionResidentBlockRepeatedCallsRefreshWeights(t *testing.T) {
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

	const queryName = "test_attn_resident_refresh_wq"
	const keyName = "test_attn_resident_refresh_wk"
	const valueName = "test_attn_resident_refresh_wv"
	if err := rt.bindMatMulRight(queryName, backend.NewTensorF32([]int{d, d}, wqA)); err != nil {
		t.Fatalf("bind wqA: %v", err)
	}
	defer rt.unbindMatMulRight(queryName)
	if err := rt.bindMatMulRight(keyName, backend.NewTensorF32([]int{d, d}, wk)); err != nil {
		t.Fatalf("bind wk: %v", err)
	}
	defer rt.unbindMatMulRight(keyName)
	if err := rt.bindMatMulRight(valueName, backend.NewTensorF32([]int{d, d}, wv)); err != nil {
		t.Fatalf("bind wv: %v", err)
	}
	defer rt.unbindMatMulRight(valueName)

	req := backend.AttentionResidentRequest{
		Batch: 1, SeqLen: seqLen, ModelDim: d,
		Input:     backend.NewTensorF32([]int{seqLen, d}, input),
		QueryName: queryName, KeyName: keyName, ValueName: valueName,
	}
	resultA, err := rt.runAttentionBlockResident(req)
	if err != nil {
		t.Fatalf("run A: %v", err)
	}

	// Re-bind the query weight under the SAME name with different data,
	// exactly like primeAttentionResidentQueryBinding does after an
	// optimizer step.
	wqB := seqData(d*d, 0.037, -0.027)
	if err := rt.bindMatMulRight(queryName, backend.NewTensorF32([]int{d, d}, wqB)); err != nil {
		t.Fatalf("bind wqB: %v", err)
	}
	resultB, err := rt.runAttentionBlockResident(req)
	if err != nil {
		t.Fatalf("run B: %v", err)
	}

	wantA := matmulHost(input, seqLen, d, wqA, d)
	wantB := matmulHost(input, seqLen, d, wqB, d)
	assertFloatSlicesClose(t, resultA.Query.F32, wantA, 1e-4)
	assertFloatSlicesClose(t, resultB.Query.F32, wantB, 1e-4)

	same := true
	for i := range resultA.Query.F32 {
		if abs32ForCUDATest(resultA.Query.F32[i]-resultB.Query.F32[i]) > 1e-6 {
			same = false
			break
		}
	}
	if same {
		t.Fatal("query projection after re-bind matches the pre-rebind result; the resident block may be caching stale weights")
	}
}

func abs32ForCUDATest(v float32) float32 {
	if v < 0 {
		return -v
	}
	return v
}
