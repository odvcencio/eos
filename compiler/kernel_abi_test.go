package compiler

import (
	"strings"
	"testing"

	eosartifact "m31labs.dev/eos/artifact/eos"
)

func TestExtractKernelABICUDA(t *testing.T) {
	source := `#include <cuda_fp16.h>
extern "C" __global__ void score_cuda(const float* query, const float* docs, float* out0, int rows, int cols) {
  int row = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if (row >= rows) return;
  out0[row] = query[0] + docs[0];
}`
	abi, err := extractKernelABI(eosartifact.BackendCUDA, "score_cuda", source)
	if err != nil {
		t.Fatal(err)
	}
	if got, want := abi.Parser, kernelABIParserFull; got != want {
		t.Fatalf("parser = %q, want %q", got, want)
	}
	if got, want := len(abi.Args), 5; got != want {
		t.Fatalf("arg count = %d, want %d", got, want)
	}
	if got, want := abi.Args[0].Type, "const float*"; got != want {
		t.Fatalf("query type = %q, want %q", got, want)
	}
	if got, want := abi.Args[0].Access, "read"; got != want {
		t.Fatalf("query access = %q, want %q", got, want)
	}
	if got, want := abi.Args[0].AddressSpace, "global"; got != want {
		t.Fatalf("query address space = %q, want %q", got, want)
	}
	if got, want := abi.Args[2].Access, "write"; got != want {
		t.Fatalf("out access = %q, want %q", got, want)
	}
	if got, want := abi.Args[3].Location, "value"; got != want {
		t.Fatalf("rows location = %q, want %q", got, want)
	}
}

func TestExtractKernelABIMetal(t *testing.T) {
	source := `#include <metal_stdlib>
using namespace metal;
kernel void embed_metal(const device float* in0 [[buffer(0)]], device float* out0 [[buffer(1)]], constant int& rows [[buffer(2)]], uint gid [[thread_position_in_grid]]) {
  if ((int)gid >= rows) return;
  out0[gid] = in0[gid];
}`
	abi, err := extractKernelABI(eosartifact.BackendMetal, "embed_metal", source)
	if err != nil {
		t.Fatal(err)
	}
	if got, want := len(abi.Args), 4; got != want {
		t.Fatalf("arg count = %d, want %d", got, want)
	}
	for i, want := range []string{"buffer:0", "buffer:1", "buffer:2", "builtin:thread_position_in_grid"} {
		if got := abi.Args[i].Location; got != want {
			t.Fatalf("arg %d location = %q, want %q", i, got, want)
		}
	}
	if got, want := abi.Args[0].AddressSpace, "device"; got != want {
		t.Fatalf("in address space = %q, want %q", got, want)
	}
	if got, want := abi.Args[2].Access, "value"; got != want {
		t.Fatalf("rows access = %q, want %q", got, want)
	}
}

func TestBuildAttachesKernelABI(t *testing.T) {
	bundle, err := Build(nil, Options{ModuleName: "tiny_embed", Preset: PresetTinyEmbed})
	if err != nil {
		t.Fatal(err)
	}
	if len(bundle.Artifact.Kernels) == 0 {
		t.Fatal("artifact has no kernels")
	}
	for _, variant := range bundle.Artifact.Kernels[0].Variants {
		if variant.Backend != eosartifact.BackendCUDA && variant.Backend != eosartifact.BackendMetal {
			continue
		}
		if variant.ABI == nil {
			t.Fatalf("%s variant has no ABI", variant.Backend)
		}
		if !strings.HasPrefix(variant.ABI.Parser, "gotreesitter-cpp-") {
			t.Fatalf("%s parser = %q", variant.Backend, variant.ABI.Parser)
		}
	}
}

func TestExtractKernelABIRejectsMissingEntry(t *testing.T) {
	_, err := extractKernelABI(eosartifact.BackendCUDA, "missing", `extern "C" __global__ void present(const float* in0) {}`)
	if err == nil || !strings.Contains(err.Error(), "missing") {
		t.Fatalf("error = %v, want missing entry error", err)
	}
}
