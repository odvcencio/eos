package backend

import (
	"crypto/sha256"
	"encoding/hex"
	"testing"

	eosartifact "m31labs.dev/eos/artifact/eos"
	"m31labs.dev/eos/compiler"
)

func TestCompileNativeKernelProgramCarriesOfflineImage(t *testing.T) {
	bundle, err := compiler.Build(nil, compiler.Options{ModuleName: "tiny_embed", Preset: compiler.PresetTinyEmbed})
	if err != nil {
		t.Fatal(err)
	}
	data := []byte("fake-ptx")
	sum := sha256.Sum256(data)
	for i := range bundle.Artifact.Kernels[0].Variants {
		variant := &bundle.Artifact.Kernels[0].Variants[i]
		if variant.Backend != eosartifact.BackendCUDA {
			continue
		}
		variant.Binary = &eosartifact.KernelBinary{Format: "ptx", Arch: "90", SourceSHA256: variant.ABI.SourceSHA256, SHA256: hex.EncodeToString(sum[:]), Data: data}
	}
	compiled, err := CompileVariants(bundle.Artifact, eosartifact.BackendCUDA)
	if err != nil {
		t.Fatal(err)
	}
	kernel := bundle.Artifact.Kernels[0]
	prog, err := CompileNativeKernelProgram(eosartifact.BackendCUDA, kernel, compiled[kernel.Name])
	if err != nil {
		t.Fatal(err)
	}
	if got, want := prog.LaunchConfig["launch_compiler"], "offline_ptx"; got != want {
		t.Fatalf("launch_compiler = %v, want %v", got, want)
	}
	if prog.Compiled.Binary == nil || string(prog.Compiled.Binary.Data) != string(data) {
		t.Fatalf("compiled offline binary = %+v", prog.Compiled.Binary)
	}
}

func TestCompileNativeKernelProgramCUDAConfig(t *testing.T) {
	bundle, err := compiler.Build(nil, compiler.Options{ModuleName: "tiny_embed", Preset: compiler.PresetTinyEmbed})
	if err != nil {
		t.Fatalf("build: %v", err)
	}
	compiled, err := CompileVariants(bundle.Artifact, eosartifact.BackendCUDA)
	if err != nil {
		t.Fatalf("compile variants: %v", err)
	}
	kernel := bundle.Artifact.Kernels[0]
	prog, err := CompileNativeKernelProgram(eosartifact.BackendCUDA, kernel, compiled[kernel.Name])
	if err != nil {
		t.Fatalf("compile native kernel: %v", err)
	}
	if got := prog.LaunchConfig["launch_api"]; got != "cuLaunchKernel" {
		t.Fatalf("launch_api = %v, want cuLaunchKernel", got)
	}
	if got := prog.LaunchConfig["launch_block_size"]; got != 128 {
		t.Fatalf("launch_block_size = %v, want 128", got)
	}
	if got := prog.LaunchConfig["dispatch_mode"]; got != "backend_native" {
		t.Fatalf("dispatch_mode = %v, want backend_native", got)
	}
}

func TestCompileNativeKernelProgramMetalConfig(t *testing.T) {
	bundle, err := compiler.Build(nil, compiler.Options{ModuleName: "tiny_embed", Preset: compiler.PresetTinyEmbed})
	if err != nil {
		t.Fatalf("build: %v", err)
	}
	compiled, err := CompileVariants(bundle.Artifact, eosartifact.BackendMetal)
	if err != nil {
		t.Fatalf("compile variants: %v", err)
	}
	kernel := bundle.Artifact.Kernels[0]
	prog, err := CompileNativeKernelProgram(eosartifact.BackendMetal, kernel, compiled[kernel.Name])
	if err != nil {
		t.Fatalf("compile native kernel: %v", err)
	}
	if got := prog.LaunchConfig["launch_api"]; got != "dispatchThreadgroups" {
		t.Fatalf("launch_api = %v, want dispatchThreadgroups", got)
	}
	if got := prog.LaunchConfig["launch_threadgroup_size"]; got != 128 {
		t.Fatalf("launch_threadgroup_size = %v, want 128", got)
	}
}

func TestCompileNativeKernelProgramPortableGPUConfigs(t *testing.T) {
	bundle, err := compiler.Build(nil, compiler.Options{ModuleName: "tiny_embed", Preset: compiler.PresetTinyEmbed})
	if err != nil {
		t.Fatalf("build: %v", err)
	}
	kernel := bundle.Artifact.Kernels[0]
	cases := []struct {
		kind      eosartifact.BackendKind
		launchAPI string
		sizeKey   string
	}{
		{kind: eosartifact.BackendVulkan, launchAPI: "vkCmdDispatch", sizeKey: "launch_workgroup_size"},
		{kind: eosartifact.BackendDirectML, launchAPI: "IDMLCommandRecorder::RecordDispatch", sizeKey: "launch_threadgroup_size"},
		{kind: eosartifact.BackendWebGPU, launchAPI: "GPUComputePassEncoder.dispatchWorkgroups", sizeKey: "launch_workgroup_size"},
	}
	for _, tc := range cases {
		t.Run(string(tc.kind), func(t *testing.T) {
			compiled, err := CompileVariants(bundle.Artifact, tc.kind)
			if err != nil {
				t.Fatalf("compile variants: %v", err)
			}
			prog, err := CompileNativeKernelProgram(tc.kind, kernel, compiled[kernel.Name])
			if err != nil {
				t.Fatalf("compile native kernel: %v", err)
			}
			if got := prog.LaunchConfig["launch_api"]; got != tc.launchAPI {
				t.Fatalf("launch_api = %v, want %s", got, tc.launchAPI)
			}
			if got := prog.LaunchConfig[tc.sizeKey]; got != 128 {
				t.Fatalf("%s = %v, want 128", tc.sizeKey, got)
			}
		})
	}
}

func TestCompileNativeKernelProgramRejectsBackendSourceMismatch(t *testing.T) {
	kernel := eosartifact.Kernel{Name: "bad"}
	compiled := CompiledKernel{
		Name:    "bad",
		Backend: eosartifact.BackendCUDA,
		Entry:   "bad_cuda",
		Source:  "kernel void bad_cuda() {}",
	}
	_, err := CompileNativeKernelProgram(eosartifact.BackendCUDA, kernel, compiled)
	if err == nil {
		t.Fatal("expected backend source mismatch")
	}
}

func TestCompileNativeKernelProgramValidatesLaunchContract(t *testing.T) {
	bundle, err := compiler.Build(nil, compiler.Options{ModuleName: "tiny_embed", Preset: compiler.PresetTinyEmbed})
	if err != nil {
		t.Fatalf("build: %v", err)
	}
	kernel := bundle.Artifact.Kernels[0]
	compiled, err := CompileVariants(bundle.Artifact, eosartifact.BackendCUDA)
	if err != nil {
		t.Fatalf("compile variants: %v", err)
	}
	prog, err := CompileNativeKernelProgram(eosartifact.BackendCUDA, kernel, compiled[kernel.Name])
	if err != nil {
		t.Fatalf("compile native kernel: %v", err)
	}
	if got := prog.LaunchConfig["launch_abi_status"]; got != "validated" {
		t.Fatalf("launch_abi_status = %v, want validated", got)
	}
	if got := prog.LaunchConfig["launch_contract_shape"]; got != "row_wise" {
		t.Fatalf("launch_contract_shape = %v, want row_wise", got)
	}
	if got := prog.LaunchConfig["launch_arg_count"]; got != 4 {
		t.Fatalf("launch_arg_count = %v, want 4", got)
	}
	if prog.LaunchContract.Fingerprint == "" {
		t.Fatal("launch contract fingerprint is empty")
	}

	compiled[kernel.Name].ABI.Args[1].Name = "wrong_output"
	if _, err := CompileNativeKernelProgram(eosartifact.BackendCUDA, kernel, compiled[kernel.Name]); err == nil {
		t.Fatal("expected launch ABI argument mismatch")
	}
}

func TestCompileNativeKernelProgramMetalContractOmitsBuiltins(t *testing.T) {
	bundle, err := compiler.Build(nil, compiler.Options{ModuleName: "tiny_embed", Preset: compiler.PresetTinyEmbed})
	if err != nil {
		t.Fatalf("build: %v", err)
	}
	kernel := bundle.Artifact.Kernels[0]
	compiled, err := CompileVariants(bundle.Artifact, eosartifact.BackendMetal)
	if err != nil {
		t.Fatalf("compile variants: %v", err)
	}
	prog, err := CompileNativeKernelProgram(eosartifact.BackendMetal, kernel, compiled[kernel.Name])
	if err != nil {
		t.Fatalf("compile native kernel: %v", err)
	}
	if got := prog.LaunchConfig["launch_arg_count"]; got != 4 {
		t.Fatalf("launch_arg_count = %v, want 4 runtime buffer args", got)
	}
	if got := prog.LaunchConfig["launch_abi_status"]; got != "validated" {
		t.Fatalf("launch_abi_status = %v, want validated", got)
	}
}
