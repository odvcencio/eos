//go:build linux && cgo

package cuda

import (
	"context"
	"math"
	"strings"
	"testing"

	eosartifact "m31labs.dev/eos/artifact/eos"
	"m31labs.dev/eos/runtime/backend"
	turboquant "m31labs.dev/turboquant"
)

func TestCUDATurboQuantStepsMatchTurboQuantSpec(t *testing.T) {
	rt, err := newDeviceRuntime()
	if err != nil {
		t.Skipf("no cuda runtime available: %v", err)
	}
	if rt == nil {
		t.Skip("no cuda runtime available")
	}
	defer rt.close()

	input := backend.NewTensorF32([]int{1, 5, 2, 2}, []float32{
		0.10, -0.20, 0.30, -0.40,
		0.50, -0.60, 0.70, -0.80,
		0.90, -1.00, 1.10, -1.20,
		1.30, -1.40, 1.50, -1.60,
		1.70, -1.80, 1.90, -2.00,
	})
	step := eosartifact.Step{Kind: eosartifact.StepTurboQEncode, Attributes: map[string]string{"bits": "4", "seed": "77", "rounds": "1"}}
	cfg, ok := planBuiltinTurboQEncode(step, []*backend.Tensor{input})
	if !ok {
		t.Fatal("turboquant encode should be supported")
	}
	encoded, err := rt.runTurboQEncodeStep([]*backend.Tensor{input}, scalarF32ValueType(), cfg)
	if err != nil {
		t.Fatalf("run turboquant encode: %v", err)
	}
	if encoded.VariantEntry != "__builtin_cuda_turboquant_encode" {
		t.Fatalf("encode variant = %q", encoded.VariantEntry)
	}
	if encoded.Metadata["device_execution"] != true {
		t.Fatalf("encode device_execution = %v, want true", encoded.Metadata["device_execution"])
	}
	wantCoords, wantNorms := hostTurboQEncode(input, 4, 1, 77)
	assertTensorClose(t, encoded.Outputs[0], input.Shape, wantCoords)
	assertTensorClose(t, encoded.Outputs[1], []int{1, 2, 2}, wantNorms)

	decodeType := eosartifact.ValueType{
		Kind:   eosartifact.ValueTensor,
		Tensor: &eosartifact.TensorType{DType: "f32"},
	}
	decodeStep := eosartifact.Step{Kind: eosartifact.StepTurboQDecode, Attributes: map[string]string{"bits": "4", "seed": "77", "rounds": "1"}}
	decodeCfg, ok := planBuiltinTurboQDecode(decodeStep, encoded.Outputs)
	if !ok {
		t.Fatal("turboquant decode should be supported")
	}
	decoded, err := rt.runTurboQDecodeStep(encoded.Outputs, decodeType, decodeCfg)
	if err != nil {
		t.Fatalf("run turboquant decode: %v", err)
	}
	if decoded.VariantEntry != "__builtin_cuda_turboquant_decode" {
		t.Fatalf("decode variant = %q", decoded.VariantEntry)
	}
	wantDecoded := hostTurboQDecode(encoded.Outputs[0], encoded.Outputs[1], 4, 1, 77)
	assertTensorClose(t, decoded.Outputs[0], input.Shape, wantDecoded)
}

func TestCUDATurboQuantRejectsDefaultMultiRoundSpec(t *testing.T) {
	input := backend.NewTensorF32([]int{1, 4, 1, 1}, []float32{0.10, -0.20, 0.30, -0.40})
	step := eosartifact.Step{Kind: eosartifact.StepTurboQEncode, Attributes: map[string]string{"bits": "4", "seed": "77"}}
	if _, ok := planBuiltinTurboQEncode(step, []*backend.Tensor{input}); ok {
		t.Fatal("cuda turboquant encode planned default multi-round Hadamard spec; want fail-closed until CUDA supports it")
	}
	step.Attributes["rounds"] = "3"
	if _, ok := planBuiltinTurboQEncode(step, []*backend.Tensor{input}); ok {
		t.Fatal("cuda turboquant encode planned explicit multi-round Hadamard spec; want fail-closed until CUDA supports it")
	}
	coords := backend.NewTensorQ4([]int{1, 4, 1, 1}, []float32{1, 2, 3, 4})
	norms := backend.NewTensorQNorm([]int{1, 1, 1}, []float32{128})
	decodeStep := eosartifact.Step{Kind: eosartifact.StepTurboQDecode, Attributes: map[string]string{"bits": "4", "seed": "77"}}
	if _, ok := planBuiltinTurboQDecode(decodeStep, []*backend.Tensor{coords, norms}); ok {
		t.Fatal("cuda turboquant decode planned default multi-round Hadamard spec; want fail-closed until CUDA supports it")
	}
	decodeStep.Attributes["rounds"] = "3"
	if _, ok := planBuiltinTurboQDecode(decodeStep, []*backend.Tensor{coords, norms}); ok {
		t.Fatal("cuda turboquant decode planned explicit multi-round Hadamard spec; want fail-closed until CUDA supports it")
	}
	step.Attributes["rounds"] = "1"
	if _, ok := planBuiltinTurboQEncode(step, []*backend.Tensor{input}); !ok {
		t.Fatal("cuda turboquant encode rejected explicit rounds=1")
	}
	decodeStep.Attributes["rounds"] = "1"
	if _, ok := planBuiltinTurboQDecode(decodeStep, []*backend.Tensor{coords, norms}); !ok {
		t.Fatal("cuda turboquant decode rejected explicit rounds=1")
	}
	step.Attributes["rounds"] = "bogus"
	if _, ok := planBuiltinTurboQEncode(step, []*backend.Tensor{input}); ok {
		t.Fatal("cuda turboquant encode planned malformed rounds")
	}
	decodeStep.Attributes["rounds"] = "bogus"
	if _, ok := planBuiltinTurboQDecode(decodeStep, []*backend.Tensor{coords, norms}); ok {
		t.Fatal("cuda turboquant decode planned malformed rounds")
	}
}

func TestTurboQuantReferenceRejectsMalformedRounds(t *testing.T) {
	input := backend.NewTensorF32([]int{1, 4, 1, 1}, []float32{0.10, -0.20, 0.30, -0.40})
	if _, _, err := backend.TurboQuantEncodeReference(input, map[string]string{"bits": "4", "rounds": "bogus"}); err == nil || !strings.Contains(err.Error(), "rounds") {
		t.Fatalf("encode malformed rounds error = %v, want rounds error", err)
	}
	coords := backend.NewTensorQ4([]int{1, 4, 1, 1}, []float32{1, 2, 3, 4})
	norms := backend.NewTensorQNorm([]int{1, 1, 1}, []float32{128})
	if _, err := backend.TurboQuantDecodeReference(coords, norms, map[string]string{"bits": "4", "rounds": "bogus"}); err == nil || !strings.Contains(err.Error(), "rounds") {
		t.Fatalf("decode malformed rounds error = %v, want rounds error", err)
	}
}

func TestCUDATurboQuantExecutorFailsClosedForUnsupportedRounds(t *testing.T) {
	rt, err := newDeviceRuntime()
	if err != nil {
		t.Skipf("no cuda runtime available: %v", err)
	}
	if rt == nil {
		t.Skip("no cuda runtime available")
	}
	defer rt.close()

	input := backend.NewTensorF32([]int{1, 4, 1, 1}, []float32{0.10, -0.20, 0.30, -0.40})
	for _, tc := range []struct {
		name    string
		attrs   map[string]string
		wantErr string
	}{
		{name: "default-multi-round", attrs: map[string]string{"bits": "4", "seed": "77"}, wantErr: "requires rounds=1"},
		{name: "explicit-multi-round", attrs: map[string]string{"bits": "4", "seed": "77", "rounds": "3"}, wantErr: "requires rounds=1"},
		{name: "malformed-rounds", attrs: map[string]string{"bits": "4", "seed": "77", "rounds": "bogus"}, wantErr: "invalid"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			_, err := runCUDATurboQuantEncodeModule(rt, tc.attrs, input)
			if err == nil || !strings.Contains(err.Error(), tc.wantErr) {
				t.Fatalf("run error = %v, want %q", err, tc.wantErr)
			}
		})
	}

	result, err := runCUDATurboQuantEncodeModule(rt, map[string]string{"bits": "4", "seed": "77", "rounds": "1"}, input)
	if err != nil {
		t.Fatalf("run rounds=1: %v", err)
	}
	if got := result.Outputs["coords"].Metadata["execution_mode"]; got != "cuda_device" {
		t.Fatalf("rounds=1 execution_mode = %v, want cuda_device", got)
	}
	if got := result.Outputs["coords"].Metadata["rounds"]; got != 1 {
		t.Fatalf("rounds metadata = %v, want 1", got)
	}
}

func TestCUDATurboQuantDecodeExecutorFailsClosedForUnsupportedRounds(t *testing.T) {
	rt, err := newDeviceRuntime()
	if err != nil {
		t.Skipf("no cuda runtime available: %v", err)
	}
	if rt == nil {
		t.Skip("no cuda runtime available")
	}
	defer rt.close()

	coords := backend.NewTensorQ4([]int{1, 4, 1, 1}, []float32{1, 2, 3, 4})
	norms := backend.NewTensorQNorm([]int{1, 1, 1}, []float32{128})
	for _, tc := range []struct {
		name    string
		attrs   map[string]string
		wantErr string
	}{
		{name: "default-multi-round", attrs: map[string]string{"bits": "4", "seed": "77"}, wantErr: "requires rounds=1"},
		{name: "explicit-multi-round", attrs: map[string]string{"bits": "4", "seed": "77", "rounds": "3"}, wantErr: "requires rounds=1"},
		{name: "malformed-rounds", attrs: map[string]string{"bits": "4", "seed": "77", "rounds": "bogus"}, wantErr: "invalid"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			_, err := runCUDATurboQuantDecodeModule(rt, tc.attrs, coords, norms)
			if err == nil || !strings.Contains(err.Error(), tc.wantErr) {
				t.Fatalf("run error = %v, want %q", err, tc.wantErr)
			}
		})
	}

	result, err := runCUDATurboQuantDecodeModule(rt, map[string]string{"bits": "4", "seed": "77", "rounds": "1"}, coords, norms)
	if err != nil {
		t.Fatalf("run rounds=1: %v", err)
	}
	if got := result.Outputs["out"].Metadata["execution_mode"]; got != "cuda_device" {
		t.Fatalf("rounds=1 execution_mode = %v, want cuda_device", got)
	}
	if got := result.Outputs["out"].Metadata["rounds"]; got != 1 {
		t.Fatalf("rounds metadata = %v, want 1", got)
	}
}

func runCUDATurboQuantEncodeModule(rt *deviceRuntime, attrs map[string]string, input *backend.Tensor) (backend.Result, error) {
	mod := eosartifact.NewModule("cuda_turboquant_rounds_contract")
	mod.EntryPoints = []eosartifact.EntryPoint{{
		Name: "quantize",
		Kind: eosartifact.EntryPointPipeline,
		Inputs: []eosartifact.ValueBinding{
			{Name: "x", Type: cudaTensorValueType("f32", []string{"1", "4", "1", "1"})},
		},
		Outputs: []eosartifact.ValueBinding{
			{Name: "coords", Type: cudaTensorValueType("q4", []string{"1", "4", "1", "1"})},
			{Name: "norms", Type: cudaTensorValueType("q_norm", []string{"1", "1", "1"})},
		},
	}}
	mod.Buffers = []eosartifact.Buffer{
		{Name: "coords", DType: "q4", Shape: []string{"1", "4", "1", "1"}},
		{Name: "norms", DType: "q_norm", Shape: []string{"1", "1", "1"}},
	}
	mod.Steps = []eosartifact.Step{
		{Entry: "quantize", Kind: eosartifact.StepTurboQEncode, Name: "encode", Inputs: []string{"x"}, Outputs: []string{"coords", "norms"}, Attributes: cloneTurboRoundAttrs(attrs)},
		{Entry: "quantize", Kind: eosartifact.StepReturn, Name: "return", Outputs: []string{"coords", "norms"}},
	}
	exec := &executor{module: mod, device: rt}
	return backend.ExecuteSymbolic(context.Background(), mod, nil, nil, exec.dispatchKernel, exec.dispatchStep, eosartifact.BackendCUDA, backend.Request{
		Entry:  "quantize",
		Inputs: map[string]any{"x": input},
	})
}

func runCUDATurboQuantDecodeModule(rt *deviceRuntime, attrs map[string]string, coords, norms *backend.Tensor) (backend.Result, error) {
	mod := eosartifact.NewModule("cuda_turboquant_decode_rounds_contract")
	mod.EntryPoints = []eosartifact.EntryPoint{{
		Name: "decode",
		Kind: eosartifact.EntryPointPipeline,
		Inputs: []eosartifact.ValueBinding{
			{Name: "coords", Type: cudaTensorValueType("q4", []string{"1", "4", "1", "1"})},
			{Name: "norms", Type: cudaTensorValueType("q_norm", []string{"1", "1", "1"})},
		},
		Outputs: []eosartifact.ValueBinding{
			{Name: "out", Type: cudaTensorValueType("f32", []string{"1", "4", "1", "1"})},
		},
	}}
	mod.Buffers = []eosartifact.Buffer{
		{Name: "out", DType: "f32", Shape: []string{"1", "4", "1", "1"}},
	}
	mod.Steps = []eosartifact.Step{
		{Entry: "decode", Kind: eosartifact.StepTurboQDecode, Name: "decode", Inputs: []string{"coords", "norms"}, Outputs: []string{"out"}, Attributes: cloneTurboRoundAttrs(attrs)},
		{Entry: "decode", Kind: eosartifact.StepReturn, Name: "return", Outputs: []string{"out"}},
	}
	exec := &executor{module: mod, device: rt}
	return backend.ExecuteSymbolic(context.Background(), mod, nil, nil, exec.dispatchKernel, exec.dispatchStep, eosartifact.BackendCUDA, backend.Request{
		Entry: "decode",
		Inputs: map[string]any{
			"coords": coords,
			"norms":  norms,
		},
	})
}

func cudaTensorValueType(dtype string, shape []string) eosartifact.ValueType {
	return eosartifact.ValueType{
		Kind:   eosartifact.ValueTensor,
		Tensor: &eosartifact.TensorType{DType: dtype, Shape: shape},
	}
}

func cloneTurboRoundAttrs(in map[string]string) map[string]string {
	if in == nil {
		return nil
	}
	out := make(map[string]string, len(in))
	for key, value := range in {
		out[key] = value
	}
	return out
}

func hostTurboQEncode(input *backend.Tensor, bits, rounds int, seed int64) ([]float32, []float32) {
	q := turboquant.NewHadamardRoundsWithSeed(input.Shape[1], bits, rounds, seed)
	n, channels, height, width := input.Shape[0], input.Shape[1], input.Shape[2], input.Shape[3]
	coords := make([]float32, len(input.F32))
	norms := make([]float32, n*height*width)
	vec := make([]float32, channels)
	indices := make([]int, channels)
	for b := 0; b < n; b++ {
		for y := 0; y < height; y++ {
			for x := 0; x < width; x++ {
				for c := 0; c < channels; c++ {
					vec[c] = input.F32[offset4(input.Shape, b, c, y, x)]
				}
				norm := q.QuantizeIndicesTo(indices, vec)
				for c := 0; c < channels; c++ {
					coords[offset4(input.Shape, b, c, y, x)] = float32(indices[c])
				}
				norms[(b*height+y)*width+x] = float32(quantizeQNormForTest(norm))
			}
		}
	}
	return coords, norms
}

func hostTurboQDecode(coords, norms *backend.Tensor, bits, rounds int, seed int64) []float32 {
	q := turboquant.NewHadamardRoundsWithSeed(coords.Shape[1], bits, rounds, seed)
	n, channels, height, width := coords.Shape[0], coords.Shape[1], coords.Shape[2], coords.Shape[3]
	out := make([]float32, len(coords.F32))
	vec := make([]float32, channels)
	indices := make([]int, channels)
	for b := 0; b < n; b++ {
		for y := 0; y < height; y++ {
			for x := 0; x < width; x++ {
				for c := 0; c < channels; c++ {
					indices[c] = clampIntForTest(int(math.Round(float64(coords.F32[offset4(coords.Shape, b, c, y, x)]))), 0, (1<<bits)-1)
				}
				q.DequantizeIndicesTo(vec, indices)
				norm := dequantizeQNormForTest(byte(clampIntForTest(int(math.Round(float64(norms.F32[(b*height+y)*width+x]))), 0, 255)))
				for c := 0; c < channels; c++ {
					out[offset4(coords.Shape, b, c, y, x)] = vec[c] * norm
				}
			}
		}
	}
	return out
}

func quantizeQNormForTest(norm float32) byte {
	if norm <= 0 || math.IsNaN(float64(norm)) {
		return 0
	}
	if math.IsInf(float64(norm), 1) {
		return 255
	}
	t := (math.Log(float64(norm)) + 16.0) / 32.0
	if t <= 0 {
		return 0
	}
	if t >= 1 {
		return 255
	}
	return byte(math.Round(t * 255))
}

func dequantizeQNormForTest(encoded byte) float32 {
	t := float64(encoded) / 255
	return float32(math.Exp(-16.0 + t*32.0))
}
