//go:build linux && cgo

package cuda

import (
	"math"
	"strings"
	"testing"

	"m31labs.dev/eos/runtime/backend"
)

func TestCUDABERTOneLayerResidentFixtureMatchesHostNoIntermediateDownloads(t *testing.T) {
	rt, err := newDeviceRuntime()
	if err != nil {
		t.Skipf("CUDA unavailable: %v", err)
	}
	defer rt.close()

	const (
		batch        = 2
		tokens       = 3
		vocab        = 7
		hidden       = 4
		heads        = 2
		intermediate = 6
		epsilon      = 1e-12
	)
	inputIDs := backend.NewTensorI32([]int{batch, tokens}, []int32{0, 2, 4, 3, 1, 5})
	attentionMask := backend.NewTensorI32([]int{batch, tokens}, []int32{1, 1, 0, 1, 1, 1})
	tokenTypeIDs := backend.NewTensorI32([]int{batch, tokens}, []int32{0, 0, 0, 0, 1, 1})
	positionIDs := backend.NewTensorI32([]int{batch, tokens}, []int32{0, 1, 2, 0, 1, 2})

	weights := oneLayerBERTFixtureWeights(vocab, tokens, hidden, intermediate)
	names := bindOneLayerBERTFixtureResidents(t, rt, weights)

	embedded, err := backend.BERTEmbeddingsReference(
		weights["token_embeddings"],
		weights["position_embeddings"],
		weights["token_type_embeddings"],
		weights["embedding_layernorm_weight"],
		weights["embedding_layernorm_bias"],
		inputIDs,
		positionIDs,
		tokenTypeIDs,
		epsilon,
	)
	if err != nil {
		t.Fatalf("host embeddings: %v", err)
	}
	want, err := backend.BERTEncoderLayerReference(
		embedded,
		attentionMask,
		weights["attention_query_weight"],
		weights["attention_query_bias"],
		weights["attention_key_weight"],
		weights["attention_key_bias"],
		weights["attention_value_weight"],
		weights["attention_value_bias"],
		weights["attention_output_weight"],
		weights["attention_output_bias"],
		weights["attention_layernorm_weight"],
		weights["attention_layernorm_bias"],
		weights["intermediate_weight"],
		weights["intermediate_bias"],
		weights["output_weight"],
		weights["output_bias"],
		weights["output_layernorm_weight"],
		weights["output_layernorm_bias"],
		heads,
		epsilon,
		"gelu",
	)
	if err != nil {
		t.Fatalf("host layer: %v", err)
	}

	got, transfers, err := rt.runBERTOneLayerResidentFixture(inputIDs, attentionMask, tokenTypeIDs, names, heads, epsilon)
	if err != nil {
		t.Fatalf("cuda one-layer fixture: %v", err)
	}
	if got.DType != "f32" || !got.EqualShape(want) {
		t.Fatalf("got dtype/shape = %s %v, want %s %v", got.DType, got.Shape, want.DType, want.Shape)
	}
	maxAbs, minCos := bertFixtureMaxAbsAndMinCos(got.F32, want.F32, batch, tokens, hidden)
	if maxAbs > 5e-4 || minCos < 0.99999 {
		t.Fatalf("parity max_abs=%g min_cos=%g, want <=5e-4 and >=0.99999", maxAbs, minCos)
	}
	if transfers.IntermediateDownloadedBytes != 0 {
		t.Fatalf("intermediate downloaded bytes=%d, want 0", transfers.IntermediateDownloadedBytes)
	}
	wantFinalDownload := int64(batch * tokens * hidden * 4)
	if transfers.FinalDownloadedBytes != wantFinalDownload {
		t.Fatalf("final downloaded bytes=%d, want final tensor bytes %d", transfers.FinalDownloadedBytes, wantFinalDownload)
	}
	if transfers.StatusDownloadedBytes != 4 {
		t.Fatalf("status downloaded bytes=%d, want 4", transfers.StatusDownloadedBytes)
	}
	if transfers.DownloadedBytes != wantFinalDownload+transfers.StatusDownloadedBytes {
		t.Fatalf("downloaded bytes total=%d, want final+status %d", transfers.DownloadedBytes, wantFinalDownload+transfers.StatusDownloadedBytes)
	}
	wantUploads := int64((len(inputIDs.I32)+len(attentionMask.I32)+len(tokenTypeIDs.I32))*4 + 4)
	if transfers.UploadedBytes != wantUploads {
		t.Fatalf("uploaded bytes=%d, want input/status uploads %d", transfers.UploadedBytes, wantUploads)
	}
	if transfers.ResidentWeightBytesReferenced == 0 {
		t.Fatal("resident weight bytes referenced should be nonzero")
	}
	if newBERTCUDAEncoderFoundationStatus().FullDeviceReady {
		t.Fatal("one-layer fixture must not promote full-device readiness")
	}
	t.Logf("one-layer BERT CUDA parity max_abs=%.8g min_cos=%.8g uploads=%d total_download=%d final_download=%d status_download=%d intermediate_download=%d resident_weight_bytes=%d", maxAbs, minCos, transfers.UploadedBytes, transfers.DownloadedBytes, transfers.FinalDownloadedBytes, transfers.StatusDownloadedBytes, transfers.IntermediateDownloadedBytes, transfers.ResidentWeightBytesReferenced)
}

func TestCUDABERTOneLayerResidentFixtureHandlesNonzeroDeviceStatus(t *testing.T) {
	rt, err := newDeviceRuntime()
	if err != nil {
		t.Skipf("CUDA unavailable: %v", err)
	}
	defer rt.close()

	const (
		batch        = 1
		tokens       = 2
		vocab        = 7
		hidden       = 4
		heads        = 2
		intermediate = 6
		epsilon      = 1e-12
	)
	inputIDs := backend.NewTensorI32([]int{batch, tokens}, []int32{0, 2})
	attentionMask := backend.NewTensorI32([]int{batch, tokens}, []int32{1, 1})
	tokenTypeIDs := backend.NewTensorI32([]int{batch, tokens}, []int32{0, 1})
	weights := oneLayerBERTFixtureWeights(vocab, tokens, hidden, intermediate)
	names := bindOneLayerBERTFixtureResidents(t, rt, weights)

	_, transfers, err := rt.runBERTOneLayerResidentFixtureWithOptions(inputIDs, attentionMask, tokenTypeIDs, names, heads, epsilon, bertCUDAOneLayerFixtureOptions{InjectStatusBeforeDownload: 99})
	if err == nil || !strings.Contains(err.Error(), "embedding status=99") {
		t.Fatalf("error=%v transfers=%+v, want injected nonzero device status", err, transfers)
	}
	if transfers.StatusDownloadedBytes != 4 || transfers.DownloadedBytes != 4 {
		t.Fatalf("downloaded bytes=%+v, want status-only download accounting", transfers)
	}
	if transfers.FinalDownloadedBytes != 0 || transfers.IntermediateDownloadedBytes != 0 {
		t.Fatalf("final/intermediate downloads=%+v, want no tensor download after status failure", transfers)
	}
	wantUploads := int64((len(inputIDs.I32)+len(attentionMask.I32)+len(tokenTypeIDs.I32))*4 + 4 + 4)
	if transfers.UploadedBytes != wantUploads {
		t.Fatalf("uploaded bytes=%d, want inputs/status init/status injection %d", transfers.UploadedBytes, wantUploads)
	}
}

func TestCUDABERTOneLayerResidentFixtureFailsClosedOnMissingResidentWeight(t *testing.T) {
	rt, err := newDeviceRuntime()
	if err != nil {
		t.Skipf("CUDA unavailable: %v", err)
	}
	defer rt.close()

	inputIDs := backend.NewTensorI32([]int{1, 1}, []int32{0})
	attentionMask := backend.NewTensorI32([]int{1, 1}, []int32{1})
	tokenTypeIDs := backend.NewTensorI32([]int{1, 1}, []int32{0})
	names := map[string]string{"token_embeddings": "missing"}
	_, transfers, err := rt.runBERTOneLayerResidentFixture(inputIDs, attentionMask, tokenTypeIDs, names, 1, 1e-12)
	if err == nil || !strings.Contains(err.Error(), "not bound") {
		t.Fatalf("error=%v transfers=%+v, want unbound resident failure", err, transfers)
	}
	if transfers.DownloadedBytes != 0 || transfers.IntermediateDownloadedBytes != 0 {
		t.Fatalf("failed run should not download bytes: %+v", transfers)
	}
}

func TestCUDABERTOneLayerResidentFixturePreflightRejectsUnboundAndBadShape(t *testing.T) {
	rt, err := newDeviceRuntime()
	if err != nil {
		t.Skipf("CUDA unavailable: %v", err)
	}
	defer rt.close()

	inputIDs := backend.NewTensorI32([]int{1, 2}, []int32{0, 1})
	attentionMask := backend.NewTensorI32([]int{1, 2}, []int32{1, 1})
	tokenTypeIDs := backend.NewTensorI32([]int{1, 2}, []int32{0, 0})
	weights := oneLayerBERTFixtureWeights(7, 2, 4, 6)
	names := bindOneLayerBERTFixtureResidents(t, rt, weights)

	t.Run("present but unbound tensor", func(t *testing.T) {
		badNames := cloneStringMap(names)
		badNames["attention_output_bias"] = "fixture_unbound_attention_output_bias"
		_, transfers, err := rt.runBERTOneLayerResidentFixture(inputIDs, attentionMask, tokenTypeIDs, badNames, 2, 1e-12)
		if err == nil || !strings.Contains(err.Error(), "not bound") || transfers.UploadedBytes != 0 || transfers.DownloadedBytes != 0 {
			t.Fatalf("error=%v transfers=%+v, want preflight unbound tensor failure with no transfers", err, transfers)
		}
	})
	t.Run("present but unbound matrix", func(t *testing.T) {
		badNames := cloneStringMap(names)
		badNames["attention_key_weight"] = "fixture_unbound_attention_key_weight"
		_, transfers, err := rt.runBERTOneLayerResidentFixture(inputIDs, attentionMask, tokenTypeIDs, badNames, 2, 1e-12)
		if err == nil || !strings.Contains(err.Error(), "not bound") || transfers.UploadedBytes != 0 || transfers.DownloadedBytes != 0 {
			t.Fatalf("error=%v transfers=%+v, want preflight unbound matrix failure with no transfers", err, transfers)
		}
	})
	t.Run("bad resident matrix shape", func(t *testing.T) {
		badWeights := cloneTensorMap(weights)
		badWeights["attention_query_weight"] = backend.NewTensorF32([]int{5, 4}, deterministicF32(20, 0.03, 0.018))
		badRT, err := newDeviceRuntime()
		if err != nil {
			t.Skipf("CUDA unavailable: %v", err)
		}
		defer badRT.close()
		badNames := bindOneLayerBERTFixtureResidents(t, badRT, badWeights)
		_, transfers, err := badRT.runBERTOneLayerResidentFixture(inputIDs, attentionMask, tokenTypeIDs, badNames, 2, 1e-12)
		if err == nil || !strings.Contains(err.Error(), "attention_query_weight shape") || transfers.UploadedBytes != 0 || transfers.DownloadedBytes != 0 {
			t.Fatalf("error=%v transfers=%+v, want preflight bad-shape failure with no transfers", err, transfers)
		}
	})
}

func TestCUDABERTOneLayerResidentFixturePreflightRejectsInvalidIDs(t *testing.T) {
	rt, err := newDeviceRuntime()
	if err != nil {
		t.Skipf("CUDA unavailable: %v", err)
	}
	defer rt.close()

	weights := oneLayerBERTFixtureWeights(7, 3, 4, 6)
	names := bindOneLayerBERTFixtureResidents(t, rt, weights)
	validIDs := backend.NewTensorI32([]int{1, 3}, []int32{0, 1, 2})
	validMask := backend.NewTensorI32([]int{1, 3}, []int32{1, 1, 1})
	validTypes := backend.NewTensorI32([]int{1, 3}, []int32{0, 1, 0})

	for _, tt := range []struct {
		name          string
		inputIDs      *backend.Tensor
		attentionMask *backend.Tensor
		tokenTypeIDs  *backend.Tensor
		want          string
	}{
		{name: "token", inputIDs: backend.NewTensorI32([]int{1, 3}, []int32{0, 7, 2}), attentionMask: validMask, tokenTypeIDs: validTypes, want: "input_ids[1]"},
		{name: "type", inputIDs: validIDs, attentionMask: validMask, tokenTypeIDs: backend.NewTensorI32([]int{1, 3}, []int32{0, 2, 0}), want: "token_type_ids[1]"},
		{name: "mask", inputIDs: validIDs, attentionMask: backend.NewTensorI32([]int{1, 3}, []int32{1, 2, 1}), tokenTypeIDs: validTypes, want: "attention_mask[1]"},
		{name: "position", inputIDs: backend.NewTensorI32([]int{1, 4}, []int32{0, 1, 2, 3}), attentionMask: backend.NewTensorI32([]int{1, 4}, []int32{1, 1, 1, 1}), tokenTypeIDs: backend.NewTensorI32([]int{1, 4}, []int32{0, 0, 1, 1}), want: "position"},
	} {
		t.Run(tt.name, func(t *testing.T) {
			_, transfers, err := rt.runBERTOneLayerResidentFixture(tt.inputIDs, tt.attentionMask, tt.tokenTypeIDs, names, 2, 1e-12)
			if err == nil || !strings.Contains(err.Error(), tt.want) || transfers.UploadedBytes != 0 || transfers.DownloadedBytes != 0 {
				t.Fatalf("error=%v transfers=%+v, want invalid %s preflight failure with no transfers", err, transfers, tt.want)
			}
		})
	}
}

func TestCUDABERTOneLayerResidentFixtureParityIsSensitiveToWeightsAndInputs(t *testing.T) {
	rt, err := newDeviceRuntime()
	if err != nil {
		t.Skipf("CUDA unavailable: %v", err)
	}
	defer rt.close()

	const (
		batch        = 2
		tokens       = 3
		vocab        = 7
		hidden       = 4
		heads        = 2
		intermediate = 6
		epsilon      = 1e-12
	)
	inputIDs := backend.NewTensorI32([]int{batch, tokens}, []int32{0, 2, 4, 3, 1, 5})
	attentionMask := backend.NewTensorI32([]int{batch, tokens}, []int32{1, 1, 0, 1, 1, 1})
	tokenTypeIDs := backend.NewTensorI32([]int{batch, tokens}, []int32{0, 0, 0, 0, 1, 1})
	weights := oneLayerBERTFixtureWeights(vocab, tokens, hidden, intermediate)

	baselineHost := mustOneLayerHostReference(t, weights, inputIDs, attentionMask, tokenTypeIDs, heads, epsilon)
	names := bindOneLayerBERTFixtureResidents(t, rt, weights)
	baselineCUDA, _, err := rt.runBERTOneLayerResidentFixture(inputIDs, attentionMask, tokenTypeIDs, names, heads, epsilon)
	if err != nil {
		t.Fatalf("baseline cuda: %v", err)
	}
	if maxAbs, minCos := bertFixtureMaxAbsAndMinCos(baselineCUDA.F32, baselineHost.F32, batch, tokens, hidden); maxAbs > 5e-4 || minCos < 0.99999 {
		t.Fatalf("baseline parity max_abs=%g min_cos=%g", maxAbs, minCos)
	}

	changedWeights := cloneTensorMap(weights)
	changedWeights["attention_value_weight"].F32[0] += 0.75
	changedHost := mustOneLayerHostReference(t, changedWeights, inputIDs, attentionMask, tokenTypeIDs, heads, epsilon)
	if maxAbsF32(baselineHost.F32, changedHost.F32) < 1e-5 {
		t.Fatal("changed weight did not move host reference enough; parity test would be vacuous")
	}
	changedNames := bindOneLayerBERTFixtureResidents(t, rt, changedWeights)
	changedCUDA, _, err := rt.runBERTOneLayerResidentFixture(inputIDs, attentionMask, tokenTypeIDs, changedNames, heads, epsilon)
	if err != nil {
		t.Fatalf("changed-weight cuda: %v", err)
	}
	if maxAbs, minCos := bertFixtureMaxAbsAndMinCos(changedCUDA.F32, changedHost.F32, batch, tokens, hidden); maxAbs > 5e-4 || minCos < 0.99999 {
		t.Fatalf("changed-weight parity max_abs=%g min_cos=%g", maxAbs, minCos)
	}

	changedInputIDs := backend.NewTensorI32([]int{batch, tokens}, []int32{6, 2, 4, 3, 1, 5})
	changedInputHost := mustOneLayerHostReference(t, changedWeights, changedInputIDs, attentionMask, tokenTypeIDs, heads, epsilon)
	if maxAbsF32(changedHost.F32, changedInputHost.F32) < 1e-5 {
		t.Fatal("changed input did not move host reference enough; parity test would be vacuous")
	}
	changedInputCUDA, _, err := rt.runBERTOneLayerResidentFixture(changedInputIDs, attentionMask, tokenTypeIDs, changedNames, heads, epsilon)
	if err != nil {
		t.Fatalf("changed-input cuda: %v", err)
	}
	if maxAbs, minCos := bertFixtureMaxAbsAndMinCos(changedInputCUDA.F32, changedInputHost.F32, batch, tokens, hidden); maxAbs > 5e-4 || minCos < 0.99999 {
		t.Fatalf("changed-input parity max_abs=%g min_cos=%g", maxAbs, minCos)
	}
}

func oneLayerBERTFixtureWeights(vocab, positions, hidden, intermediate int) map[string]*backend.Tensor {
	weights := map[string]*backend.Tensor{
		"token_embeddings":           backend.NewTensorF32([]int{vocab, hidden}, deterministicF32(vocab*hidden, 0.17, 0.015)),
		"position_embeddings":        backend.NewTensorF32([]int{positions, hidden}, deterministicF32(positions*hidden, -0.08, 0.012)),
		"token_type_embeddings":      backend.NewTensorF32([]int{2, hidden}, deterministicF32(2*hidden, 0.04, 0.02)),
		"embedding_layernorm_weight": backend.NewTensorF32([]int{hidden}, deterministicPositiveF32(hidden, 0.9, 0.05)),
		"embedding_layernorm_bias":   backend.NewTensorF32([]int{hidden}, deterministicF32(hidden, 0.01, 0.01)),
		"attention_query_weight":     backend.NewTensorF32([]int{hidden, hidden}, deterministicF32(hidden*hidden, 0.03, 0.018)),
		"attention_query_bias":       backend.NewTensorF32([]int{hidden}, deterministicF32(hidden, -0.02, 0.006)),
		"attention_key_weight":       backend.NewTensorF32([]int{hidden, hidden}, deterministicF32(hidden*hidden, -0.01, 0.017)),
		"attention_key_bias":         backend.NewTensorF32([]int{hidden}, deterministicF32(hidden, 0.015, 0.005)),
		"attention_value_weight":     backend.NewTensorF32([]int{hidden, hidden}, deterministicF32(hidden*hidden, 0.025, 0.013)),
		"attention_value_bias":       backend.NewTensorF32([]int{hidden}, deterministicF32(hidden, -0.005, 0.004)),
		"attention_output_weight":    backend.NewTensorF32([]int{hidden, hidden}, deterministicF32(hidden*hidden, -0.02, 0.016)),
		"attention_output_bias":      backend.NewTensorF32([]int{hidden}, deterministicF32(hidden, 0.012, 0.007)),
		"attention_layernorm_weight": backend.NewTensorF32([]int{hidden}, deterministicPositiveF32(hidden, 0.85, 0.04)),
		"attention_layernorm_bias":   backend.NewTensorF32([]int{hidden}, deterministicF32(hidden, -0.015, 0.008)),
		"intermediate_weight":        backend.NewTensorF32([]int{intermediate, hidden}, deterministicF32(intermediate*hidden, 0.02, 0.014)),
		"intermediate_bias":          backend.NewTensorF32([]int{intermediate}, deterministicF32(intermediate, 0.01, 0.006)),
		"output_weight":              backend.NewTensorF32([]int{hidden, intermediate}, deterministicF32(hidden*intermediate, -0.025, 0.015)),
		"output_bias":                backend.NewTensorF32([]int{hidden}, deterministicF32(hidden, 0.005, 0.009)),
		"output_layernorm_weight":    backend.NewTensorF32([]int{hidden}, deterministicPositiveF32(hidden, 0.95, 0.03)),
		"output_layernorm_bias":      backend.NewTensorF32([]int{hidden}, deterministicF32(hidden, 0.0, 0.006)),
	}
	return weights
}

func bindOneLayerBERTFixtureResidents(t *testing.T, rt *deviceRuntime, weights map[string]*backend.Tensor) map[string]string {
	t.Helper()
	names := map[string]string{}
	for slot, tensor := range weights {
		name := "fixture_" + slot
		names[slot] = name
		if len(tensor.Shape) == 2 && strings.HasSuffix(slot, "_weight") {
			if err := rt.bindMatMulRight(name, tensor); err != nil {
				t.Fatalf("bind matrix %s: %v", slot, err)
			}
			continue
		}
		if err := rt.bindBERTResidentTensor(name, tensor); err != nil {
			t.Fatalf("bind tensor %s: %v", slot, err)
		}
	}
	return names
}

func mustOneLayerHostReference(t *testing.T, weights map[string]*backend.Tensor, inputIDs, attentionMask, tokenTypeIDs *backend.Tensor, heads int, epsilon float64) *backend.Tensor {
	t.Helper()
	positionIDs := generatedPositionIDs(inputIDs.Shape[0], inputIDs.Shape[1])
	embedded, err := backend.BERTEmbeddingsReference(
		weights["token_embeddings"],
		weights["position_embeddings"],
		weights["token_type_embeddings"],
		weights["embedding_layernorm_weight"],
		weights["embedding_layernorm_bias"],
		inputIDs,
		positionIDs,
		tokenTypeIDs,
		epsilon,
	)
	if err != nil {
		t.Fatalf("host embeddings: %v", err)
	}
	out, err := backend.BERTEncoderLayerReference(
		embedded,
		attentionMask,
		weights["attention_query_weight"],
		weights["attention_query_bias"],
		weights["attention_key_weight"],
		weights["attention_key_bias"],
		weights["attention_value_weight"],
		weights["attention_value_bias"],
		weights["attention_output_weight"],
		weights["attention_output_bias"],
		weights["attention_layernorm_weight"],
		weights["attention_layernorm_bias"],
		weights["intermediate_weight"],
		weights["intermediate_bias"],
		weights["output_weight"],
		weights["output_bias"],
		weights["output_layernorm_weight"],
		weights["output_layernorm_bias"],
		heads,
		epsilon,
		"gelu",
	)
	if err != nil {
		t.Fatalf("host layer: %v", err)
	}
	return out
}

func generatedPositionIDs(batch, tokens int) *backend.Tensor {
	values := make([]int32, batch*tokens)
	for row := range values {
		values[row] = int32(row % tokens)
	}
	return backend.NewTensorI32([]int{batch, tokens}, values)
}

func cloneStringMap(in map[string]string) map[string]string {
	out := make(map[string]string, len(in))
	for key, value := range in {
		out[key] = value
	}
	return out
}

func cloneTensorMap(in map[string]*backend.Tensor) map[string]*backend.Tensor {
	out := make(map[string]*backend.Tensor, len(in))
	for key, value := range in {
		clone := &backend.Tensor{
			DType: value.DType,
			Shape: append([]int(nil), value.Shape...),
			F32:   append([]float32(nil), value.F32...),
			I32:   append([]int32(nil), value.I32...),
		}
		out[key] = clone
	}
	return out
}

func maxAbsF32(a, b []float32) float64 {
	maxAbs := 0.0
	for i := range a {
		diff := math.Abs(float64(a[i] - b[i]))
		if diff > maxAbs {
			maxAbs = diff
		}
	}
	return maxAbs
}

func deterministicF32(n int, offset, scale float64) []float32 {
	out := make([]float32, n)
	for i := range out {
		v := offset + scale*math.Sin(float64(i+1)*1.7) + scale*0.5*math.Cos(float64(i+3)*0.31)
		out[i] = float32(v)
	}
	return out
}

func deterministicPositiveF32(n int, offset, scale float64) []float32 {
	out := deterministicF32(n, offset, scale)
	for i := range out {
		if out[i] < 0.1 {
			out[i] = 0.1
		}
	}
	return out
}

func bertFixtureMaxAbsAndMinCos(got, want []float32, batch, tokens, hidden int) (float64, float64) {
	maxAbs := 0.0
	minCos := 1.0
	for row := 0; row < batch*tokens; row++ {
		dot, gotNorm, wantNorm := 0.0, 0.0, 0.0
		for d := 0; d < hidden; d++ {
			idx := row*hidden + d
			diff := math.Abs(float64(got[idx] - want[idx]))
			if diff > maxAbs {
				maxAbs = diff
			}
			g := float64(got[idx])
			w := float64(want[idx])
			dot += g * w
			gotNorm += g * g
			wantNorm += w * w
		}
		if gotNorm > 0 && wantNorm > 0 {
			cos := dot / math.Sqrt(gotNorm*wantNorm)
			if cos < minCos {
				minCos = cos
			}
		}
	}
	return maxAbs, minCos
}
