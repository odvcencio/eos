//go:build linux && cgo

package cuda

import (
	"context"
	"math"
	"testing"

	"m31labs.dev/eos/compiler"
	eosruntime "m31labs.dev/eos/runtime"
	"m31labs.dev/eos/runtime/backend"
	"m31labs.dev/eos/runtime/backends/metal"
)

// These tests guard the CUDA embedding forward pass against the host reference
// across a range of hidden dimensions. They exist because a regression made the
// CUDA work stream non-blocking (CU_STREAM_NON_BLOCKING) while device<->host
// transfers used the synchronous cuMemcpy on the default stream — the copies
// raced in-flight kernels/GEMMs and produced non-deterministic, size-dependent
// garbage (near-orthogonal embeddings at D>=128). Per-kernel parity tests all
// passed because they run at tiny sizes; only an end-to-end sweep at realistic
// dimensions catches it. Cross-backend cosine must be ~1.0 at every dimension.

func cosine(a, b []float32) float64 {
	if len(a) != len(b) || len(a) == 0 {
		return math.NaN()
	}
	var dot, na, nb float64
	for i := range a {
		dot += float64(a[i]) * float64(b[i])
		na += float64(a[i]) * float64(a[i])
		nb += float64(b[i]) * float64(b[i])
	}
	return dot / (math.Sqrt(na) * math.Sqrt(nb))
}

func parityGenTensor(dtype string, rows, cols, seed int) *backend.Tensor {
	vals := make([]float32, rows*cols)
	for i := range vals {
		n := (i*1103515245 + seed*12345 + 1013904223) % 2003
		if n < 0 {
			n += 2003
		}
		vals[i] = (float32(n)/2003.0)*2 - 1
	}
	return &backend.Tensor{DType: dtype, Shape: []int{rows, cols}, F32: vals}
}

func parityRunEmbed(t *testing.T, b backend.Backend, src []byte, preset compiler.Preset, weights map[string]*backend.Tensor, tokens int) []float32 {
	t.Helper()
	opts := compiler.Options{ModuleName: "parity_probe", Preset: preset}
	if preset != "" {
		src = nil
	}
	bundle, err := compiler.Build(src, opts)
	if err != nil {
		t.Fatalf("build: %v", err)
	}
	loadOpts := make([]eosruntime.LoadOption, 0, len(weights))
	for name, tensor := range weights {
		loadOpts = append(loadOpts, eosruntime.WithWeight(name, tensor))
	}
	prog, err := eosruntime.New(b).Load(context.Background(), bundle.Artifact, loadOpts...)
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	toks := make([]int32, tokens)
	mask := make([]int32, tokens)
	for i := 0; i < tokens; i++ {
		toks[i] = int32(i % 8)
		mask[i] = 1
	}
	inputs := map[string]any{
		"tokens":         backend.NewTensorI32([]int{tokens}, toks),
		"attention_mask": backend.NewTensorI32([]int{tokens}, mask),
	}
	if preset == compiler.PresetEncoderTrainableQ8x2 || preset == compiler.PresetEncoderTrainableQ4x2 {
		roles := make([]int32, tokens)
		for i := range roles {
			roles[i] = int32(i % 3)
		}
		inputs["role_ids"] = backend.NewTensorI32([]int{tokens}, roles)
	}
	result, err := prog.Run(context.Background(), backend.Request{
		Entry:  "embed_pooled",
		Inputs: inputs,
	})
	if err != nil {
		t.Fatalf("run: %v", err)
	}
	for _, o := range result.Outputs {
		if tn, ok := o.Data.(*backend.Tensor); ok {
			return tn.F32
		}
	}
	t.Fatal("no output tensor in result")
	return nil
}

// TestCUDAEmbedSimplePooledMatchesHost guards the simple embedding tail
// (gather -> matmul -> normalize -> masked mean_pool) across a dimension sweep.
func TestCUDAEmbedSimplePooledMatchesHost(t *testing.T) {
	const src = `
param token_embedding: f16[V, D] @weight("weights/token_embedding")
param projection: f16[D, D] @weight("weights/projection")

pipeline embed_pooled(tokens: i32[T], attention_mask: i32[T]) -> f16[D] {
    let hidden = gather(token_embedding, tokens)
    let projected = @matmul(hidden, projection)
    let normalized = normalize(projected)
    return mean_pool(normalized, attention_mask)
}
`
	for _, d := range []int{64, 128, 256} {
		weights := map[string]*backend.Tensor{
			"token_embedding": parityGenTensor("f16", 8, d, 1),
			"projection":      parityGenTensor("f16", d, d, 2),
		}
		host := parityRunEmbed(t, metal.New(), []byte(src), "", weights, 16)
		dev := parityRunEmbed(t, New(), []byte(src), "", weights, 16)
		cos := cosine(host, dev)
		t.Logf("simple D=%-4d cosine(host,cuda)=%.6f", d, cos)
		if math.IsNaN(cos) || cos < 0.999 {
			t.Errorf("simple embedding diverges on CUDA at D=%d: cosine=%.6f (want >=0.999)", d, cos)
		}
	}
}

// TestCUDARoPEBatchedRowsAreBatchPositionInvariant guards the CUDA "rope"
// kernel against a regression where it used the flattened [B*T] row index
// (blockIdx.x*blockDim.x+threadIdx.x) as the rotary position directly,
// instead of the token's position within its own sequence (row % seq_len).
// That bug made a token's rotated hidden state -- and therefore its final
// embedding -- depend on which batch row it happened to land in during a
// batched device call, i.e. on `--batch-size`, even for byte-identical
// input content. See compiler.go's cudaNativeEmitter.ropeBody and
// runtime/backends/cuda/native_linux.go's runRoPEKernel/classifyCUDAKernel.
//
// It builds a rank-3 [Q, T, D] batched gather+rope pipeline, replicates the
// SAME token sequence across every batch row, and asserts: (1) every batch
// row matches row 0 on both host and CUDA (position must reset per
// sequence, not accumulate across rows), and (2) CUDA matches the host
// reference row-for-row.
func TestCUDARoPEBatchedRowsAreBatchPositionInvariant(t *testing.T) {
	const src = `
param token_embedding: f16[V, D] @weight("weights/token_embedding")

pipeline rope_probe(tokens: i32[Q, T]) -> f16[Q, T, D] {
    let hidden = gather(token_embedding, tokens)
    return rope(hidden)
}
`
	const q, tlen, dim, vocab = 4, 5, 8, 12
	weights := map[string]*backend.Tensor{
		"token_embedding": parityGenTensor("f16", vocab, dim, 1),
	}
	toks := make([]int32, q*tlen)
	for row := 0; row < q; row++ {
		for tpos := 0; tpos < tlen; tpos++ {
			toks[row*tlen+tpos] = int32(tpos % vocab)
		}
	}
	host := parityRunRopeBatched(t, metal.New(), []byte(src), weights, toks, q, tlen)
	dev := parityRunRopeBatched(t, New(), []byte(src), weights, toks, q, tlen)

	rowWidth := tlen * dim
	for row := 0; row < q; row++ {
		hostRow := host[row*rowWidth : (row+1)*rowWidth]
		hostRow0 := host[0:rowWidth]
		if idx, ok := nearlyEqualF32(hostRow0, hostRow, 1e-5); !ok {
			t.Fatalf("host reference row %d diverges from row 0 at %d (host=%g row=%g) for identical input rows -- host reference regressed too", row, idx, hostRow0[idx], hostRow[idx])
		}
		devRow := dev[row*rowWidth : (row+1)*rowWidth]
		devRow0 := dev[0:rowWidth]
		if idx, ok := nearlyEqualF32(devRow0, devRow, 1e-4); !ok {
			t.Fatalf("CUDA rope row %d diverges from row 0 at %d (row0=%g row%d=%g) for IDENTICAL input tokens -- rotary position is leaking the batch row index (compiler.go cudaNativeEmitter.ropeBody)", row, idx, devRow0[idx], row, devRow[idx])
		}
		if idx, ok := nearlyEqualF32(hostRow, devRow, 1e-3); !ok {
			t.Fatalf("CUDA rope row %d diverges from host reference at %d (host=%g cuda=%g)", row, idx, hostRow[idx], devRow[idx])
		}
	}
}

// parityRunRopeBatched builds and runs a rank-3 [Q, T] -> [Q, T, D] pipeline
// (used to probe RoPE's batched row behavior directly) and returns the flat
// f32 output.
func parityRunRopeBatched(t *testing.T, b backend.Backend, src []byte, weights map[string]*backend.Tensor, tokens []int32, q, tlen int) []float32 {
	t.Helper()
	bundle, err := compiler.Build(src, compiler.Options{ModuleName: "rope_probe"})
	if err != nil {
		t.Fatalf("build: %v", err)
	}
	loadOpts := make([]eosruntime.LoadOption, 0, len(weights))
	for name, tensor := range weights {
		loadOpts = append(loadOpts, eosruntime.WithWeight(name, tensor))
	}
	prog, err := eosruntime.New(b).Load(context.Background(), bundle.Artifact, loadOpts...)
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	result, err := prog.Run(context.Background(), backend.Request{
		Entry: "rope_probe",
		Inputs: map[string]any{
			"tokens": backend.NewTensorI32([]int{q, tlen}, tokens),
		},
	})
	if err != nil {
		t.Fatalf("run: %v", err)
	}
	for _, o := range result.Outputs {
		if tn, ok := o.Data.(*backend.Tensor); ok {
			return tn.F32
		}
	}
	t.Fatal("no output tensor in result")
	return nil
}

// TestCUDAEncoderEmbedMatchesHost guards the full encoder forward (the real
// manta-embed-v1 shape: 2x [attention(q/k/v/o, transpose, softmax) -> residual
// + layernorm -> ffn(gelu) -> residual + layernorm] -> normalize -> mean_pool)
// across a dimension sweep. This is the configuration that regressed.
func TestCUDAEncoderEmbedMatchesHost(t *testing.T) {
	for _, d := range []int{64, 128, 256} {
		h := d * 2
		weights := map[string]*backend.Tensor{
			"token_embedding": parityGenTensor("q8", 8, d, 1),
			"role_embedding":  parityGenTensor("q8", 3, d, 8),
			"attn_q":          parityGenTensor("q8", d, d, 2),
			"attn_k":          parityGenTensor("q8", d, d, 3),
			"attn_v":          parityGenTensor("q8", d, d, 4),
			"attn_o":          parityGenTensor("q8", d, d, 5),
			"ffn_up":          parityGenTensor("q8", d, h, 6),
			"projection":      parityGenTensor("q8", h, d, 7),
		}
		host := parityRunEmbed(t, metal.New(), nil, compiler.PresetEncoderTrainableQ8x2, weights, 16)
		dev := parityRunEmbed(t, New(), nil, compiler.PresetEncoderTrainableQ8x2, weights, 16)
		cos := cosine(host, dev)
		t.Logf("encoder D=%-4d H=%-4d cosine(host,cuda)=%.6f", d, h, cos)
		if math.IsNaN(cos) || cos < 0.999 {
			t.Errorf("CUDA encoder forward diverges from host at D=%d: cosine=%.6f (want >=0.999)", d, cos)
		}
	}
}
