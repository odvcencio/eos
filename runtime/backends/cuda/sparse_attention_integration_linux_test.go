//go:build linux && cgo

package cuda

import (
	"context"
	"strings"
	"testing"

	eosartifact "m31labs.dev/eos/artifact/eos"
	"m31labs.dev/eos/runtime/backend"
)

func TestCUDASparseAttentionStepMatchesReference(t *testing.T) {
	rt, err := newDeviceRuntime()
	if err != nil {
		t.Skipf("no cuda runtime available: %v", err)
	}
	if rt == nil {
		t.Skip("no cuda runtime available")
	}
	defer rt.close()

	query := backend.NewTensorF16([]int{2, 2}, []float32{
		1, 0,
		0, 1,
	})
	key := backend.NewTensorF16([]int{3, 2}, []float32{
		1, 0,
		0, 1,
		-1, 0,
	})
	value := backend.NewTensorF16([]int{3, 2}, []float32{
		10, 0,
		0, 20,
		-10, 0,
	})
	step := eosartifact.Step{Kind: eosartifact.StepSparseAttention, Attributes: map[string]string{"top_k": "1"}}
	cfg, ok := planBuiltinSparseAttention(step, []*backend.Tensor{query, key, value})
	if !ok {
		t.Fatal("sparse_attention should be supported")
	}
	outputType := eosartifact.ValueType{
		Kind:   eosartifact.ValueTensor,
		Tensor: &eosartifact.TensorType{DType: "f16"},
	}
	got, err := rt.runSparseAttentionStep([]*backend.Tensor{query, key, value}, outputType, cfg)
	if err != nil {
		t.Fatalf("run sparse_attention: %v", err)
	}
	if got.VariantEntry != "__builtin_cuda_sparse_attention" {
		t.Fatalf("variant = %q", got.VariantEntry)
	}
	if got.Metadata["device_execution"] != true {
		t.Fatalf("device_execution = %v, want true", got.Metadata["device_execution"])
	}
	if got.Metadata["selected_key_count"] != 1 || got.Metadata["candidate_key_budget"] != 3 {
		t.Fatalf("sparse attention budget metadata = %+v", got.Metadata)
	}
	want, err := backend.SparseAttentionReference(query, key, value, step.Attributes)
	if err != nil {
		t.Fatalf("reference sparse_attention: %v", err)
	}
	assertTensorClose(t, got.Outputs[0], want.Shape, want.F32)
}

func TestCUDATurboSparseAttentionStepMatchesReference(t *testing.T) {
	rt, err := newDeviceRuntime()
	if err != nil {
		t.Skipf("no cuda runtime available: %v", err)
	}
	if rt == nil {
		t.Skip("no cuda runtime available")
	}
	defer rt.close()

	query := backend.NewTensorF16([]int{2, 2}, []float32{
		1, 0,
		0, 1,
	})
	keyNCHW := backend.NewTensorF16([]int{1, 2, 3, 1}, []float32{
		1, 0, -1,
		0, 1, 0,
	})
	valueNCHW := backend.NewTensorF16([]int{1, 2, 3, 1}, []float32{
		10, 0, -10,
		0, 20, 0,
	})
	attrs := map[string]string{"bits": "4", "seed": "77", "rounds": "1", "top_k": "1"}
	keyCoords, keyNorms, err := backend.TurboQuantEncodeReference(keyNCHW, attrs)
	if err != nil {
		t.Fatal(err)
	}
	valueCoords, valueNorms, err := backend.TurboQuantEncodeReference(valueNCHW, attrs)
	if err != nil {
		t.Fatal(err)
	}
	step := eosartifact.Step{Kind: eosartifact.StepTurboSparseAttention, Attributes: attrs}
	inputs := []*backend.Tensor{query, keyCoords, keyNorms, valueCoords, valueNorms}
	cfg, ok := planBuiltinTurboSparseAttention(step, inputs)
	if !ok {
		t.Fatal("turbo_sparse_attention should be supported")
	}
	outputType := eosartifact.ValueType{
		Kind:   eosartifact.ValueTensor,
		Tensor: &eosartifact.TensorType{DType: "f16"},
	}
	got, err := rt.runTurboSparseAttentionStep(inputs, outputType, cfg)
	if err != nil {
		t.Fatalf("run turbo_sparse_attention: %v", err)
	}
	if got.VariantEntry != "__builtin_cuda_turbo_sparse_attention" {
		t.Fatalf("variant = %q", got.VariantEntry)
	}
	if got.Metadata["dense_kv_materialized"] != false {
		t.Fatalf("dense_kv_materialized = %v, want false", got.Metadata["dense_kv_materialized"])
	}
	if got.Metadata["kv_decode"] != "cuda_turboquant_inline" {
		t.Fatalf("kv_decode = %v, want cuda_turboquant_inline", got.Metadata["kv_decode"])
	}
	if got.Metadata["selected_key_count"] != 1 || got.Metadata["candidate_key_budget"] != 3 || got.Metadata["score_count_fraction"] != 1.0 {
		t.Fatalf("turbo sparse attention budget metadata = %+v", got.Metadata)
	}
	want, err := backend.TurboSparseAttentionReference(query, keyCoords, keyNorms, valueCoords, valueNorms, attrs)
	if err != nil {
		t.Fatalf("reference turbo_sparse_attention: %v", err)
	}
	assertTensorClose(t, got.Outputs[0], want.Shape, want.F32)
}

func TestCUDATurboSparseAttentionRoundPlanningContract(t *testing.T) {
	query, keyCoords, keyNorms, valueCoords, valueNorms := turboSparseRoundContractInputs(t)
	base := map[string]string{"bits": "4", "seed": "77", "top_k": "1"}
	step := eosartifact.Step{Kind: eosartifact.StepTurboSparseAttention, Attributes: cloneTurboRoundAttrs(base)}
	inputs := []*backend.Tensor{query, keyCoords, keyNorms, valueCoords, valueNorms}
	if _, ok := planBuiltinTurboSparseAttention(step, inputs); ok {
		t.Fatal("cuda turbo_sparse_attention planned default multi-round Hadamard spec; want fail-closed until CUDA supports it")
	}
	step.Attributes["rounds"] = "3"
	if _, ok := planBuiltinTurboSparseAttention(step, inputs); ok {
		t.Fatal("cuda turbo_sparse_attention planned explicit multi-round Hadamard spec; want fail-closed until CUDA supports it")
	}
	step.Attributes["rounds"] = "bogus"
	if _, ok := planBuiltinTurboSparseAttention(step, inputs); ok {
		t.Fatal("cuda turbo_sparse_attention planned malformed rounds")
	}
	step.Attributes["rounds"] = "1"
	if _, ok := planBuiltinTurboSparseAttention(step, inputs); !ok {
		t.Fatal("cuda turbo_sparse_attention rejected explicit rounds=1")
	}
}

func TestCUDATurboSparseAttentionExecutorFailsClosedForUnsupportedRounds(t *testing.T) {
	rt, err := newDeviceRuntime()
	if err != nil {
		t.Skipf("no cuda runtime available: %v", err)
	}
	if rt == nil {
		t.Skip("no cuda runtime available")
	}
	defer rt.close()

	query, keyCoords, keyNorms, valueCoords, valueNorms := turboSparseRoundContractInputs(t)
	for _, tc := range []struct {
		name    string
		attrs   map[string]string
		wantErr string
	}{
		{name: "default-multi-round", attrs: map[string]string{"bits": "4", "seed": "77", "top_k": "1"}, wantErr: "requires rounds=1"},
		{name: "explicit-multi-round", attrs: map[string]string{"bits": "4", "seed": "77", "rounds": "3", "top_k": "1"}, wantErr: "requires rounds=1"},
		{name: "malformed-rounds", attrs: map[string]string{"bits": "4", "seed": "77", "rounds": "bogus", "top_k": "1"}, wantErr: "invalid"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			_, err := runCUDATurboSparseModule(rt, tc.attrs, query, keyCoords, keyNorms, valueCoords, valueNorms)
			if err == nil || !strings.Contains(err.Error(), tc.wantErr) {
				t.Fatalf("run error = %v, want %q", err, tc.wantErr)
			}
		})
	}

	result, err := runCUDATurboSparseModule(rt, map[string]string{"bits": "4", "seed": "77", "rounds": "1", "top_k": "1"}, query, keyCoords, keyNorms, valueCoords, valueNorms)
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

func turboSparseRoundContractInputs(t *testing.T) (*backend.Tensor, *backend.Tensor, *backend.Tensor, *backend.Tensor, *backend.Tensor) {
	t.Helper()
	attrs := map[string]string{"bits": "4", "seed": "77", "rounds": "1", "top_k": "1"}
	query := backend.NewTensorF16([]int{2, 2}, []float32{
		1, 0,
		0, 1,
	})
	keyNCHW := backend.NewTensorF16([]int{1, 2, 3, 1}, []float32{
		1, 0, -1,
		0, 1, 0,
	})
	valueNCHW := backend.NewTensorF16([]int{1, 2, 3, 1}, []float32{
		10, 0, -10,
		0, 20, 0,
	})
	keyCoords, keyNorms, err := backend.TurboQuantEncodeReference(keyNCHW, attrs)
	if err != nil {
		t.Fatal(err)
	}
	valueCoords, valueNorms, err := backend.TurboQuantEncodeReference(valueNCHW, attrs)
	if err != nil {
		t.Fatal(err)
	}
	return query, keyCoords, keyNorms, valueCoords, valueNorms
}

func runCUDATurboSparseModule(rt *deviceRuntime, attrs map[string]string, query, keyCoords, keyNorms, valueCoords, valueNorms *backend.Tensor) (backend.Result, error) {
	mod := eosartifact.NewModule("cuda_turbo_sparse_rounds_contract")
	mod.EntryPoints = []eosartifact.EntryPoint{{
		Name: "attend",
		Kind: eosartifact.EntryPointPipeline,
		Inputs: []eosartifact.ValueBinding{
			{Name: "q", Type: cudaTensorValueType("f16", []string{"2", "2"})},
			{Name: "kc", Type: cudaTensorValueType("q4", []string{"1", "2", "3", "1"})},
			{Name: "kn", Type: cudaTensorValueType("q_norm", []string{"1", "3", "1"})},
			{Name: "vc", Type: cudaTensorValueType("q4", []string{"1", "2", "3", "1"})},
			{Name: "vn", Type: cudaTensorValueType("q_norm", []string{"1", "3", "1"})},
		},
		Outputs: []eosartifact.ValueBinding{
			{Name: "out", Type: cudaTensorValueType("f16", []string{"2", "2"})},
		},
	}}
	mod.Buffers = []eosartifact.Buffer{
		{Name: "out", DType: "f16", Shape: []string{"2", "2"}},
	}
	mod.Steps = []eosartifact.Step{
		{Entry: "attend", Kind: eosartifact.StepTurboSparseAttention, Name: "attend", Inputs: []string{"q", "kc", "kn", "vc", "vn"}, Outputs: []string{"out"}, Attributes: cloneTurboRoundAttrs(attrs)},
		{Entry: "attend", Kind: eosartifact.StepReturn, Name: "return", Outputs: []string{"out"}},
	}
	exec := &executor{module: mod, device: rt}
	return backend.ExecuteSymbolic(context.Background(), mod, nil, nil, exec.dispatchKernel, exec.dispatchStep, eosartifact.BackendCUDA, backend.Request{
		Entry: "attend",
		Inputs: map[string]any{
			"q":  query,
			"kc": keyCoords,
			"kn": keyNorms,
			"vc": valueCoords,
			"vn": valueNorms,
		},
	})
}

func TestCUDATurboSparseAttentionBatchedStepMatchesReference(t *testing.T) {
	rt, err := newDeviceRuntime()
	if err != nil {
		t.Skipf("no cuda runtime available: %v", err)
	}
	if rt == nil {
		t.Skip("no cuda runtime available")
	}
	defer rt.close()

	query := backend.NewTensorF16([]int{2, 1, 2}, []float32{
		1, 0,
		0, 1,
	})
	keyNCHW := backend.NewTensorF16([]int{2, 2, 3, 1}, []float32{
		1, 0, -1,
		0, 1, 0,
		0, 1, 0,
		1, 0, -1,
	})
	valueNCHW := backend.NewTensorF16([]int{2, 2, 3, 1}, []float32{
		10, 0, -10,
		0, 20, 0,
		1, 2, 3,
		100, 200, 300,
	})
	attrs := map[string]string{"bits": "4", "seed": "91", "rounds": "1", "top_k": "1"}
	keyCoords, keyNorms, err := backend.TurboQuantEncodeReference(keyNCHW, attrs)
	if err != nil {
		t.Fatal(err)
	}
	valueCoords, valueNorms, err := backend.TurboQuantEncodeReference(valueNCHW, attrs)
	if err != nil {
		t.Fatal(err)
	}
	step := eosartifact.Step{Kind: eosartifact.StepTurboSparseAttention, Attributes: attrs}
	inputs := []*backend.Tensor{query, keyCoords, keyNorms, valueCoords, valueNorms}
	cfg, ok := planBuiltinTurboSparseAttention(step, inputs)
	if !ok {
		t.Fatal("batched turbo_sparse_attention should be supported")
	}
	outputType := eosartifact.ValueType{
		Kind:   eosartifact.ValueTensor,
		Tensor: &eosartifact.TensorType{DType: "f16"},
	}
	got, err := rt.runTurboSparseAttentionStep(inputs, outputType, cfg)
	if err != nil {
		t.Fatalf("run batched turbo_sparse_attention: %v", err)
	}
	want, err := backend.TurboSparseAttentionReference(query, keyCoords, keyNorms, valueCoords, valueNorms, attrs)
	if err != nil {
		t.Fatalf("reference batched turbo_sparse_attention: %v", err)
	}
	assertTensorClose(t, got.Outputs[0], want.Shape, want.F32)
}

func TestCUDATurboSparseAttentionRoutedStepMatchesReference(t *testing.T) {
	rt, err := newDeviceRuntime()
	if err != nil {
		t.Skipf("no cuda runtime available: %v", err)
	}
	if rt == nil {
		t.Skip("no cuda runtime available")
	}
	defer rt.close()

	query := backend.NewTensorF16([]int{1, 2}, []float32{1, 0})
	keyNCHW := backend.NewTensorF16([]int{1, 2, 6, 1}, []float32{
		0, 0, 1, 2, 0, 0,
		1, 0, 0, 0, 0, 1,
	})
	valueNCHW := backend.NewTensorF16([]int{1, 2, 6, 1}, []float32{
		1, 2, 30, 40, 5, 6,
		10, 20, 300, 400, 50, 60,
	})
	attrs := map[string]string{
		"bits":             "4",
		"seed":             "119",
		"rounds":           "1",
		"top_k":            "1",
		"route_block_size": "2",
		"route_top_blocks": "1",
	}
	keyCoords, keyNorms, err := backend.TurboQuantEncodeReference(keyNCHW, attrs)
	if err != nil {
		t.Fatal(err)
	}
	valueCoords, valueNorms, err := backend.TurboQuantEncodeReference(valueNCHW, attrs)
	if err != nil {
		t.Fatal(err)
	}
	step := eosartifact.Step{Kind: eosartifact.StepTurboSparseAttention, Attributes: attrs}
	inputs := []*backend.Tensor{query, keyCoords, keyNorms, valueCoords, valueNorms}
	cfg, ok := planBuiltinTurboSparseAttention(step, inputs)
	if !ok {
		t.Fatal("routed turbo_sparse_attention should be supported")
	}
	outputType := eosartifact.ValueType{
		Kind:   eosartifact.ValueTensor,
		Tensor: &eosartifact.TensorType{DType: "f16"},
	}
	got, err := rt.runTurboSparseAttentionStep(inputs, outputType, cfg)
	if err != nil {
		t.Fatalf("run routed turbo_sparse_attention: %v", err)
	}
	if got.Metadata["routing"] != "block_anchor" {
		t.Fatalf("routing = %v, want block_anchor", got.Metadata["routing"])
	}
	if got.Metadata["candidate_key_budget"] != 2 {
		t.Fatalf("candidate_key_budget = %v, want 2", got.Metadata["candidate_key_budget"])
	}
	if got.Metadata["route_block_count"] != 3 || got.Metadata["selected_route_blocks"] != 1 || got.Metadata["estimated_score_count_per_query"] != 5 {
		t.Fatalf("routed budget metadata = %+v", got.Metadata)
	}
	if got.Metadata["subquadratic_score_plan"] != true {
		t.Fatalf("subquadratic_score_plan = %v, want true", got.Metadata["subquadratic_score_plan"])
	}
	want, err := backend.TurboSparseAttentionReference(query, keyCoords, keyNorms, valueCoords, valueNorms, attrs)
	if err != nil {
		t.Fatalf("reference routed turbo_sparse_attention: %v", err)
	}
	assertTensorClose(t, got.Outputs[0], want.Shape, want.F32)
}
