package backend

import (
	"context"
	"testing"

	eosartifact "m31labs.dev/eos/artifact/eos"
)

func TestExecuteSymbolicRecordsDeviceAndHostAccounting(t *testing.T) {
	tensorType := eosartifact.ValueType{Kind: eosartifact.ValueTensor, Tensor: &eosartifact.TensorType{DType: "f32", Shape: []string{"T", "D"}}}
	mod := &eosartifact.Module{
		Name: "accounting",
		EntryPoints: []eosartifact.EntryPoint{{
			Name:    "run",
			Inputs:  []eosartifact.ValueBinding{{Name: "x", Type: tensorType}},
			Outputs: []eosartifact.ValueBinding{{Name: "y", Type: tensorType}},
		}},
		Kernels: []eosartifact.Kernel{{
			Name:    "norm",
			Inputs:  []eosartifact.ValueBinding{{Name: "x", Type: tensorType}},
			Outputs: []eosartifact.ValueBinding{{Name: "y", Type: tensorType}},
			Body: []eosartifact.KernelOp{
				{Kind: eosartifact.KernelOpPointwise, Op: "normalize"},
				{Kind: eosartifact.KernelOpReturn, Op: "return", Outputs: []string{"y"}},
			},
		}},
		Steps: []eosartifact.Step{
			{Entry: "run", Kind: eosartifact.StepLaunchKernel, Name: "norm_step", Kernel: "norm", Inputs: []string{"x"}, Outputs: []string{"y"}},
			{Entry: "run", Kind: eosartifact.StepReturn, Outputs: []string{"y"}},
		},
	}
	input := NewTensorF32([]int{1, 2}, []float32{1, 2})
	compiled := map[string]CompiledKernel{"norm": {Name: "norm", Backend: eosartifact.BackendCUDA, Entry: "norm_cuda", Source: "library:norm"}}
	result, err := ExecuteSymbolic(context.Background(), mod, nil, compiled, func(_ context.Context, _ eosartifact.Kernel, inputs []*Tensor) (KernelDispatchResult, error) {
		return KernelDispatchResult{
			Outputs: []*Tensor{inputs[0].Clone()},
			Metadata: map[string]any{
				"device_execution":      true,
				"execution_mode":        "cuda_device",
				"uploaded_bytes":        int64(12),
				"downloaded_bytes":      int64(8),
				"sync_count":            int64(1),
				"graph_captures":        int64(1),
				"graph_replays":         int64(2),
				"resident_cache_hits":   int64(3),
				"resident_cache_misses": int64(1),
			},
		}, nil
	}, nil, eosartifact.BackendCUDA, Request{Entry: "run", Inputs: map[string]any{"x": input}})
	if err != nil {
		t.Fatalf("execute: %v", err)
	}
	if got := result.Accounting.KernelSteps; got != 1 {
		t.Fatalf("kernel steps = %d, want 1", got)
	}
	if got := result.Accounting.DeviceKernelLaunches; got != 1 {
		t.Fatalf("device kernel launches = %d, want 1", got)
	}
	if got := result.Accounting.DeviceSteps; got != 1 {
		t.Fatalf("device steps = %d, want 1", got)
	}
	if got := result.Accounting.HostSteps; got != 1 {
		t.Fatalf("host steps = %d, want return step counted once", got)
	}
	if result.Accounting.FullDeviceExecution() {
		t.Fatal("full device execution should be false when the return step is host orchestration")
	}
	if got, want := result.Accounting.UploadBytes, int64(12); got != want {
		t.Fatalf("upload bytes = %d, want %d", got, want)
	}
	if got, want := result.Accounting.ResidencyHits, int64(3); got != want {
		t.Fatalf("residency hits = %d, want %d", got, want)
	}
	if got, want := result.Metadata["device_kernel_launch_count"], "1"; got != want {
		t.Fatalf("metadata device kernel launches = %q, want %q", got, want)
	}
	if got, want := result.Metadata["full_device_execution"], "false"; got != want {
		t.Fatalf("metadata full_device_execution = %q, want %q", got, want)
	}
}

func TestExecuteSymbolicRecordsFallbackReason(t *testing.T) {
	tensorType := eosartifact.ValueType{Kind: eosartifact.ValueTensor, Tensor: &eosartifact.TensorType{DType: "f32", Shape: []string{"T"}}}
	mod := &eosartifact.Module{
		Name:        "fallback-accounting",
		EntryPoints: []eosartifact.EntryPoint{{Name: "run", Inputs: []eosartifact.ValueBinding{{Name: "x", Type: tensorType}}, Outputs: []eosartifact.ValueBinding{{Name: "y", Type: tensorType}}}},
		Kernels:     []eosartifact.Kernel{{Name: "unary", Inputs: []eosartifact.ValueBinding{{Name: "x", Type: tensorType}}, Outputs: []eosartifact.ValueBinding{{Name: "y", Type: tensorType}}, Body: []eosartifact.KernelOp{{Kind: eosartifact.KernelOpPointwise, Op: "gelu"}, {Kind: eosartifact.KernelOpReturn, Op: "return"}}}},
		Steps:       []eosartifact.Step{{Entry: "run", Kind: eosartifact.StepLaunchKernel, Kernel: "unary", Inputs: []string{"x"}, Outputs: []string{"y"}}, {Entry: "run", Kind: eosartifact.StepReturn, Outputs: []string{"y"}}},
	}
	compiled := map[string]CompiledKernel{"unary": {Name: "unary", Backend: eosartifact.BackendCUDA, Entry: "unary_cuda", Source: "library:unary"}}
	result, err := ExecuteSymbolic(context.Background(), mod, nil, compiled, func(_ context.Context, _ eosartifact.Kernel, inputs []*Tensor) (KernelDispatchResult, error) {
		return KernelDispatchResult{Outputs: []*Tensor{inputs[0].Clone()}, Metadata: map[string]any{"execution_mode": "host_fallback", "fallback_reason": "unsupported_input_shape"}}, nil
	}, nil, eosartifact.BackendCUDA, Request{Entry: "run", Inputs: map[string]any{"x": NewTensorF32([]int{1}, []float32{1})}})
	if err != nil {
		t.Fatalf("execute fallback: %v", err)
	}
	if got := result.Accounting.FallbackSteps; got != 1 {
		t.Fatalf("fallback steps = %d, want 1", got)
	}
	if got, want := result.Accounting.FallbackReasons["unsupported_input_shape"], 1; got != want {
		t.Fatalf("fallback reason count = %d, want %d", got, want)
	}
	if got, want := result.Metadata["fallback_reasons"], "unsupported_input_shape=1"; got != want {
		t.Fatalf("fallback reasons metadata = %q, want %q", got, want)
	}
}
