//go:build linux && cgo

package cuda

import (
	"math"
	"strings"
	"testing"

	eosartifact "m31labs.dev/eos/artifact/eos"
	"m31labs.dev/eos/runtime/backend"
)

func TestBERTCUDAFoundationStatusIsFailClosed(t *testing.T) {
	status := newBERTCUDAEncoderFoundationStatus()
	if status.Contract != pretrainedBERTCUDAFoundationContract {
		t.Fatalf("contract = %q", status.Contract)
	}
	if status.FullDeviceReady {
		t.Fatal("foundation status unexpectedly claims full-device readiness")
	}
	if len(status.MissingComponents) == 0 {
		t.Fatal("foundation status should name missing full-device components")
	}
}

func TestBERTCUDAFoundationRejectsNonBGEAttrs(t *testing.T) {
	step := eosartifact.Step{
		Kind: eosartifact.StepBERTEmbedder,
		Attributes: map[string]string{
			"num_hidden_layers":   "12",
			"num_attention_heads": "12",
			"hidden_act":          "gelu",
			"pooling":             "masked_mean",
			"normalization":       "l2",
			"epsilon":             "1e-12",
		},
	}
	status, err := validateBGEPretrainedBERTEmbedderStep(step, nil)
	if err == nil || !strings.Contains(err.Error(), "pooling") {
		t.Fatalf("error = %v, want pooling rejection", err)
	}
	if status.ShapeValidated || status.FullDeviceReady {
		t.Fatalf("status should remain unvalidated/fail-closed: %+v", status)
	}
}

func TestCUDAInt32UploadCopyDownloadRoundTrip(t *testing.T) {
	rt, err := newDeviceRuntime()
	if err != nil {
		t.Skipf("CUDA unavailable: %v", err)
	}
	defer rt.close()

	ptr, err := rt.uploadInt32([]int32{1, -2, 3, 4})
	if err != nil {
		t.Fatalf("upload int32: %v", err)
	}
	defer rt.freeBuffer(ptr)
	if err := rt.copyInt32ToBuffer(ptr, []int32{9, 8, -7, 6}); err != nil {
		t.Fatalf("copy int32: %v", err)
	}
	got := make([]int32, 4)
	if err := rt.downloadInt32(got, ptr); err != nil {
		t.Fatalf("download int32: %v", err)
	}
	want := []int32{9, 8, -7, 6}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("got[%d]=%d want %d (all=%v)", i, got[i], want[i], got)
		}
	}
}

func TestCheckedInt32ProductGuardsKernelOffsets(t *testing.T) {
	const maxInt32 = int(1<<31 - 1)
	tests := []struct {
		name    string
		dims    []int
		wantErr string
	}{
		{name: "max offset product allowed", dims: []int{maxInt32}},
		{name: "row hidden overflows", dims: []int{maxInt32/2 + 1, 2}, wantErr: "int32 index range"},
		{name: "embedding table overflows", dims: []int{1024, maxInt32/1024 + 1}, wantErr: "int32 index range"},
		{name: "exact gelu element count overflows", dims: []int{maxInt32 + 1}, wantErr: "int32 index range"},
		{name: "residual row hidden overflows", dims: []int{maxInt32/3 + 1, 3}, wantErr: "int32 index range"},
		{name: "cls batch tokens hidden overflows", dims: []int{2, 2, maxInt32/4 + 1}, wantErr: "int32 index range"},
		{name: "negative dimension rejected", dims: []int{4, -1}, wantErr: "negative"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := checkedInt32Product("test cuda bert kernel offset", tt.dims...)
			if tt.wantErr == "" {
				if err != nil {
					t.Fatalf("checkedInt32Product() error = %v, want nil", err)
				}
				return
			}
			if err == nil || !strings.Contains(err.Error(), tt.wantErr) {
				t.Fatalf("checkedInt32Product() error = %v, want %q", err, tt.wantErr)
			}
		})
	}
}

func TestCUDABERTEmbeddingAffineLayerNormMatchesHost(t *testing.T) {
	rt, err := newDeviceRuntime()
	if err != nil {
		t.Skipf("CUDA unavailable: %v", err)
	}
	defer rt.close()

	tokenEmb := backend.NewTensorF32([]int{4, 3}, []float32{
		0.10, -0.20, 0.30,
		0.00, 0.25, -0.15,
		-0.40, 0.50, 0.20,
		0.70, -0.10, -0.30,
	})
	positionEmb := backend.NewTensorF32([]int{3, 3}, []float32{
		0.01, 0.02, 0.03,
		-0.01, 0.04, -0.02,
		0.05, -0.03, 0.01,
	})
	typeEmb := backend.NewTensorF32([]int{2, 3}, []float32{
		0.00, 0.10, -0.10,
		0.20, -0.05, 0.05,
	})
	gamma := backend.NewTensorF32([]int{3}, []float32{1.1, 0.9, -1.2})
	beta := backend.NewTensorF32([]int{3}, []float32{0.01, -0.02, 0.03})
	inputIDs := backend.NewTensorI32([]int{2, 3}, []int32{0, 1, 2, 3, 0, 1})
	positionIDs := backend.NewTensorI32([]int{2, 3}, []int32{0, 1, 2, 0, 1, 2})
	tokenTypeIDs := backend.NewTensorI32([]int{2, 3}, []int32{0, 1, 0, 1, 0, 1})

	want, err := backend.BERTEmbeddingsReference(tokenEmb, positionEmb, typeEmb, gamma, beta, inputIDs, positionIDs, tokenTypeIDs, 1e-12)
	if err != nil {
		t.Fatalf("host embedding ref: %v", err)
	}
	got, err := rt.runBERTEmbeddingAffineLayerNorm(tokenEmb, positionEmb, typeEmb, gamma, beta, inputIDs, tokenTypeIDs, 1e-12)
	if err != nil {
		t.Fatalf("cuda embedding layernorm: %v", err)
	}
	assertBERTCloseF32(t, got.F32, want.F32, 1e-5)

	badIDs := backend.NewTensorI32([]int{1, 1}, []int32{99})
	badTypes := backend.NewTensorI32([]int{1, 1}, []int32{0})
	if _, err := rt.runBERTEmbeddingAffineLayerNorm(tokenEmb, positionEmb, typeEmb, gamma, beta, badIDs, badTypes, 1e-12); err == nil || !strings.Contains(err.Error(), "out of range") {
		t.Fatalf("bad ids error = %v, want out-of-range failure", err)
	}

	for _, tt := range []struct {
		name          string
		tokenEmb      *backend.Tensor
		positionEmb   *backend.Tensor
		typeEmb       *backend.Tensor
		gamma         *backend.Tensor
		beta          *backend.Tensor
		wantSubstring string
	}{
		{name: "token table", tokenEmb: badDTypeClone(tokenEmb, "q4"), positionEmb: positionEmb, typeEmb: typeEmb, gamma: gamma, beta: beta, wantSubstring: "token_embeddings"},
		{name: "position table", tokenEmb: tokenEmb, positionEmb: badDTypeClone(positionEmb, "q4"), typeEmb: typeEmb, gamma: gamma, beta: beta, wantSubstring: "position_embeddings"},
		{name: "type table", tokenEmb: tokenEmb, positionEmb: positionEmb, typeEmb: badDTypeClone(typeEmb, "q4"), gamma: gamma, beta: beta, wantSubstring: "token_type_embeddings"},
		{name: "gamma", tokenEmb: tokenEmb, positionEmb: positionEmb, typeEmb: typeEmb, gamma: badDTypeClone(gamma, "q4"), beta: beta, wantSubstring: "gamma"},
		{name: "beta", tokenEmb: tokenEmb, positionEmb: positionEmb, typeEmb: typeEmb, gamma: gamma, beta: badDTypeClone(beta, "q4"), wantSubstring: "beta"},
	} {
		t.Run("rejects bad dtype "+tt.name, func(t *testing.T) {
			_, err := rt.runBERTEmbeddingAffineLayerNorm(tt.tokenEmb, tt.positionEmb, tt.typeEmb, tt.gamma, tt.beta, inputIDs, tokenTypeIDs, 1e-12)
			if err == nil || !strings.Contains(err.Error(), "dtype") || !strings.Contains(err.Error(), tt.wantSubstring) {
				t.Fatalf("bad dtype error = %v, want dtype failure mentioning %q", err, tt.wantSubstring)
			}
		})
	}
}

func TestCUDABERTExactGELUMatchesHost(t *testing.T) {
	rt, err := newDeviceRuntime()
	if err != nil {
		t.Skipf("CUDA unavailable: %v", err)
	}
	defer rt.close()

	src := backend.NewTensorF32([]int{9}, []float32{-3, -1.5, -0.2, 0, 0.1, 0.5, 1, 2, 4})
	got, err := rt.runBERTExactGELU(src)
	if err != nil {
		t.Fatalf("cuda exact gelu: %v", err)
	}
	want := make([]float32, len(src.F32))
	for i, x := range src.F32 {
		want[i] = float32(0.5 * float64(x) * (1 + math.Erf(float64(x)/math.Sqrt2)))
	}
	assertBERTCloseF32(t, got.F32, want, 1e-6)
}

func TestCUDABERTResidualAffineLayerNormMatchesHost(t *testing.T) {
	rt, err := newDeviceRuntime()
	if err != nil {
		t.Skipf("CUDA unavailable: %v", err)
	}
	defer rt.close()

	src := backend.NewTensorF32([]int{2, 2, 3}, []float32{
		0.10, 0.20, -0.30,
		0.40, -0.10, 0.05,
		-0.20, 0.70, 0.30,
		0.15, -0.25, 0.35,
	})
	residual := backend.NewTensorF32([]int{2, 2, 3}, []float32{
		0.02, -0.01, 0.03,
		-0.04, 0.06, 0.08,
		0.10, -0.20, 0.05,
		0.00, 0.03, -0.06,
	})
	gamma := backend.NewTensorF32([]int{3}, []float32{1.2, -0.7, 0.5})
	beta := backend.NewTensorF32([]int{3}, []float32{0.01, 0.02, -0.03})
	got, err := rt.runBERTResidualAffineLayerNorm(src, residual, gamma, beta, 1e-12)
	if err != nil {
		t.Fatalf("cuda residual layernorm: %v", err)
	}
	wantInput := make([]float32, len(src.F32))
	for i := range wantInput {
		wantInput[i] = src.F32[i] + residual.F32[i]
	}
	want := affineLayerNormReference(wantInput, 4, 3, gamma.F32, beta.F32, 1e-12)
	assertBERTCloseF32(t, got.F32, want, 1e-5)
}

func TestCUDABERTResidualAffineLayerNormRejectsZeroHiddenAndBadDTypes(t *testing.T) {
	rt, err := newDeviceRuntime()
	if err != nil {
		t.Skipf("CUDA unavailable: %v", err)
	}
	defer rt.close()

	zeroHidden := backend.NewTensorF32([]int{1, 0}, nil)
	gamma := backend.NewTensorF32([]int{0}, nil)
	beta := backend.NewTensorF32([]int{0}, nil)
	if _, err := rt.runBERTResidualAffineLayerNorm(zeroHidden, zeroHidden, gamma, beta, 1e-12); err == nil || !strings.Contains(err.Error(), "hidden size") {
		t.Fatalf("zero hidden error = %v, want hidden-size failure", err)
	}

	src := backend.NewTensorF32([]int{1, 2}, []float32{1, 2})
	residual := backend.NewTensorF32([]int{1, 2}, []float32{3, 4})
	gamma = backend.NewTensorF32([]int{2}, []float32{1, 1})
	beta = backend.NewTensorF32([]int{2}, []float32{0, 0})
	for _, tt := range []struct {
		name          string
		src           *backend.Tensor
		residual      *backend.Tensor
		gamma         *backend.Tensor
		beta          *backend.Tensor
		wantSubstring string
	}{
		{name: "src", src: badDTypeClone(src, "q8"), residual: residual, gamma: gamma, beta: beta, wantSubstring: "src"},
		{name: "residual", src: src, residual: badDTypeClone(residual, "q8"), gamma: gamma, beta: beta, wantSubstring: "residual"},
		{name: "gamma", src: src, residual: residual, gamma: badDTypeClone(gamma, "q8"), beta: beta, wantSubstring: "gamma"},
		{name: "beta", src: src, residual: residual, gamma: gamma, beta: badDTypeClone(beta, "q8"), wantSubstring: "beta"},
	} {
		t.Run("rejects bad dtype "+tt.name, func(t *testing.T) {
			_, err := rt.runBERTResidualAffineLayerNorm(tt.src, tt.residual, tt.gamma, tt.beta, 1e-12)
			if err == nil || !strings.Contains(err.Error(), "dtype") || !strings.Contains(err.Error(), tt.wantSubstring) {
				t.Fatalf("bad dtype error = %v, want dtype failure mentioning %q", err, tt.wantSubstring)
			}
		})
	}
}

func TestCUDABERTCLSL2MatchesHost(t *testing.T) {
	rt, err := newDeviceRuntime()
	if err != nil {
		t.Skipf("CUDA unavailable: %v", err)
	}
	defer rt.close()

	hidden := backend.NewTensorF32([]int{3, 2, 4}, []float32{
		1, 2, 2, 1,
		9, 9, 9, 9,
		0, 0, 0, 0,
		1, 1, 1, 1,
		-1, 0.5, 0.5, 2,
		3, 3, 3, 3,
	})
	got, err := rt.runBERTCLSL2(hidden)
	if err != nil {
		t.Fatalf("cuda cls l2: %v", err)
	}
	want := []float32{
		1.0 / 3.1622777, 2.0 / 3.1622777, 2.0 / 3.1622777, 1.0 / 3.1622777,
		0, 0, 0, 0,
		-1.0 / 2.345208, 0.5 / 2.345208, 0.5 / 2.345208, 2.0 / 2.345208,
	}
	assertBERTCloseF32(t, got.F32, want, 1e-5)
}

func affineLayerNormReference(input []float32, rows, hidden int, gamma, beta []float32, epsilon float64) []float32 {
	out := make([]float32, len(input))
	for row := 0; row < rows; row++ {
		base := row * hidden
		mean := 0.0
		for d := 0; d < hidden; d++ {
			mean += float64(input[base+d])
		}
		mean /= float64(hidden)
		variance := 0.0
		for d := 0; d < hidden; d++ {
			centered := float64(input[base+d]) - mean
			variance += centered * centered
		}
		variance /= float64(hidden)
		invStd := 1 / math.Sqrt(variance+epsilon)
		for d := 0; d < hidden; d++ {
			out[base+d] = float32(((float64(input[base+d])-mean)*invStd)*float64(gamma[d]) + float64(beta[d]))
		}
	}
	return out
}

func assertBERTCloseF32(t *testing.T, got, want []float32, tol float64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("len got=%d want=%d", len(got), len(want))
	}
	for i := range got {
		if diff := math.Abs(float64(got[i] - want[i])); diff > tol {
			t.Fatalf("value[%d] got=%g want=%g diff=%g tol=%g", i, got[i], want[i], diff, tol)
		}
	}
}

func badDTypeClone(tensor *backend.Tensor, dtype string) *backend.Tensor {
	out := tensor.Clone()
	out.DType = dtype
	return out
}
