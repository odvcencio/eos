//go:build linux && cgo

package cuda

import (
	"context"
	"strings"
	"testing"

	eosartifact "m31labs.dev/eos/artifact/eos"
	"m31labs.dev/eos/compiler"
	"m31labs.dev/eos/runtime/backend"
)

// staleRoPECUDASource is byte-for-byte the pre-fix "rope" kernel body that
// ships in the real stale runs/*/*.mll artifacts sealed before the seq_len
// fix (see runs/eos-encoder-v21-controlled-20260626T190850Z/eos-embed-v1.mll
// and its siblings): a 4-parameter signature whose rotary angle is derived
// directly from the flattened batch row index instead of the token's
// position within its own sequence.
const staleRoPECUDASource = `
extern "C" __global__ void rope_cuda(const float* in0, float* out0, int rows, int cols) {

    int row = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (row >= rows) return;
    int base = row * cols;
    for (int c = 0; c + 1 < cols; c += 2) {
        float theta = ((float)row) / powf(10000.0f, ((float)c) / (float)cols);
        float cos_theta = cosf(theta);
        float sin_theta = sinf(theta);
        float x0 = in0[base + c];
        float x1 = in0[base + c + 1];
        out0[base + c] = x0 * cos_theta - x1 * sin_theta;
        out0[base + c + 1] = x0 * sin_theta + x1 * cos_theta;
    }
    if ((cols & 1) != 0) {
        out0[base + cols - 1] = in0[base + cols - 1];
    }
}
`

// TestValidateRoPEKernelABIRejectsStaleFourParamSource is a pure unit test on
// the guard function itself: a synthetic CompiledKernel carrying the OLD
// 4-param rope source (byte-identical to the real stale runs/*.mll
// artifacts) must be rejected with the stale-ABI error, regardless of
// whatever the Meta map says.
func TestValidateRoPEKernelABIRejectsStaleFourParamSource(t *testing.T) {
	compiled := backend.CompiledKernel{
		Name:   "rope",
		Entry:  "rope_cuda",
		Source: staleRoPECUDASource,
		// No "rope_abi" key at all -- matches the real stale artifacts,
		// which predate the tag entirely.
		Meta: map[string]string{"tile": "[128]"},
	}
	err := validateRoPEKernelABI("rope", compiled)
	if err == nil {
		t.Fatal("expected stale rope kernel ABI to be rejected, got nil error")
	}
	if !strings.Contains(err.Error(), "stale rope kernel ABI") {
		t.Fatalf("error = %q, want it to mention stale rope kernel ABI", err.Error())
	}
	if !strings.Contains(err.Error(), "recompile/reseal") {
		t.Fatalf("error = %q, want actionable recompile/reseal guidance", err.Error())
	}
}

// TestValidateRoPEKernelABIRejectsStaleSourceEvenWithSpoofedMetaTag proves the
// guard does not simply trust a "rope_abi": "v2" Meta tag -- it always
// re-derives ABI compatibility from the actual compiled Source text, so a
// desynced/stale/hand-edited Meta tag can never mask a stale kernel body.
func TestValidateRoPEKernelABIRejectsStaleSourceEvenWithSpoofedMetaTag(t *testing.T) {
	compiled := backend.CompiledKernel{
		Name:   "rope",
		Entry:  "rope_cuda",
		Source: staleRoPECUDASource,
		Meta:   map[string]string{"rope_abi": "v2"},
	}
	if err := validateRoPEKernelABI("rope", compiled); err == nil {
		t.Fatal("expected stale source to be rejected even with a fresh-looking rope_abi meta tag")
	}
}

// TestValidateRoPEKernelABIAcceptsFreshFiveParamSource guards the accept
// path: a kernel actually emitted by the current compiler's
// emitCUDARoPEKernel (5-param in0/out0/rows/cols/seq_len signature) must be
// accepted so real, current rope kernels keep dispatching to the device.
func TestValidateRoPEKernelABIAcceptsFreshFiveParamSource(t *testing.T) {
	const freshSource = `extern "C" __global__ void rope_cuda(const float* in0, float* out0, int rows, int cols, int seq_len) {
    int row = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (row >= rows) return;
    int pos = (seq_len > 0) ? (row % seq_len) : row;
}
`
	compiled := backend.CompiledKernel{
		Name:   "rope",
		Entry:  "rope_cuda",
		Source: freshSource,
		Meta:   map[string]string{"rope_abi": "v2"},
	}
	if err := validateRoPEKernelABI("rope", compiled); err != nil {
		t.Fatalf("expected fresh 5-param rope source to be accepted, got: %v", err)
	}
}

// TestValidateRoPEKernelABIRejectsMissingEntry covers the defensive branch
// where the named entry point can't be located in the source at all (e.g. a
// corrupt or mismatched artifact) -- it must fail the same way as a stale
// kernel rather than silently accepting.
func TestValidateRoPEKernelABIRejectsMissingEntry(t *testing.T) {
	compiled := backend.CompiledKernel{
		Name:   "rope",
		Entry:  "rope_cuda",
		Source: `extern "C" __global__ void something_else(const float* in0, float* out0, int rows, int cols, int seq_len) {}`,
	}
	if err := validateRoPEKernelABI("rope", compiled); err == nil {
		t.Fatal("expected missing entry point to be rejected")
	}
}

// TestCUDALoadRejectsStaleRoPEArtifact exercises the full Load path (not
// just the guard function in isolation): it compiles a real rope pipeline,
// downgrades its CUDA "rope" kernel variant's Source to the exact pre-fix
// 4-param text the real stale runs/*.mll artifacts carry, and asserts that
// Backend.Load fails loudly with the stale-ABI error rather than silently
// falling back to host execution or the row-wise launcher. This is the same
// mechanism (classifyCUDAKernel -> attachDeviceExecution ->
// validateRoPEKernelABI) that would reject
// runs/eos-encoder-v21-controlled-20260626T190850Z/eos-embed-v1.mll and its
// siblings today.
func TestCUDALoadRejectsStaleRoPEArtifact(t *testing.T) {
	const src = `
param token_embedding: f16[V, D] @weight("weights/token_embedding")

pipeline rope_probe(tokens: i32[T]) -> f16[T, D] {
    let hidden = gather(token_embedding, tokens)
    return rope(hidden)
}
`
	bundle, err := compiler.Build([]byte(src), compiler.Options{ModuleName: "rope_stale_probe"})
	if err != nil {
		t.Fatalf("build: %v", err)
	}

	found := false
	for i := range bundle.Artifact.Kernels {
		kernel := &bundle.Artifact.Kernels[i]
		if len(kernel.Body) == 0 || kernel.Body[0].Op != "rope" {
			continue
		}
		for j := range kernel.Variants {
			variant := &kernel.Variants[j]
			if variant.Backend != eosartifact.BackendCUDA {
				continue
			}
			if !strings.Contains(variant.Source, "seq_len") {
				t.Fatalf("freshly compiled CUDA rope variant should declare seq_len, got: %s", variant.Source)
			}
			// Downgrade to the exact pre-fix artifact contents.
			variant.Source = staleRoPECUDASource
			variant.Entry = "rope_cuda"
			// Simulate a pre-kernel-ABI artifact: old sealed modules do not
			// carry the new source fingerprint, so the dedicated rope guard
			// remains the actionable diagnostic for this regression.
			variant.ABI = nil
			delete(variant.Meta, "rope_abi")
			found = true
		}
	}
	if !found {
		t.Fatal("expected a CUDA rope kernel variant in the compiled bundle")
	}

	weights := map[string]backend.WeightBinding{
		"token_embedding": {Name: "token_embedding", Data: backend.NewTensorF16([]int{4, 2}, []float32{
			0, 0,
			1, 0,
			0, 1,
			1, 1,
		})},
	}
	b := New()
	_, err = b.Load(context.Background(), bundle.Artifact, weights)
	if err == nil {
		t.Fatal("expected Load to reject the stale rope kernel ABI, got nil error")
	}
	if !strings.Contains(err.Error(), "stale rope kernel ABI") {
		t.Fatalf("Load error = %q, want it to mention stale rope kernel ABI", err.Error())
	}
	if !strings.Contains(err.Error(), "recompile/reseal") {
		t.Fatalf("Load error = %q, want actionable recompile/reseal guidance", err.Error())
	}
}
